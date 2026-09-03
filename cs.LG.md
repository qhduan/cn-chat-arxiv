# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Common Measure of Communication for Speech Brain-Computer Interfaces](https://arxiv.org/abs/2609.02887) | 提出开词汇互信息（OVMI）这一信息论度量标准，解决了语音脑机接口领域因数据集、记录方法和词汇各异而无法相互比较的评估难题，为不同系统的通信能力提供了统一的衡量指标。 |
| [^2] | [Discriminative World Models for Web Agents](https://arxiv.org/abs/2609.02885) | 该论文提出“预测状态匹配”这一新训练目标，使世界模型生成的网络状态表示具有跨候选动作的区分性，从而与下游排序器对齐，提升网络智能体测试时动作选择的准确性。 |
| [^3] | [Graph Machine: Towards Better Pretraining via Edges](https://arxiv.org/abs/2609.02881) | 提出图机器（GM）架构，利用类似指针的“边”实现稀疏动态路由来访问 O(n) 规模状态，将 Qwen3-0.6B 中 75% 的稠密 Transformer 层替换为 GM 稀疏层并从头预训练，在每层仅检索极少 token 的情况下模型损失几乎不受影响甚至略有改善。 |
| [^4] | [GRADSOLVE: fast exact gradients for ODE ensembles on GPUs](https://arxiv.org/abs/2609.02876) | GRADSOLVE是一个开源JAX库，通过记录自适应求解器接受的步长并以固定步长重放进行微分，在GPU上以融合内核求解的速度实现对ODE集合的精确反向模式梯度计算。 |
| [^5] | [Improved Gradient Descent Lower Bounds Beyond Nesterov](https://arxiv.org/abs/2609.02855) | 本文证明了光滑凸优化中固定步长梯度下降的两个更强下界——非anytime的Ω(n^{-1.6342})与anytime的Ω(n^{-1.2408})，并借助silver调度可达的O(n^{-log_2(1+√2)})速率，严格分离了两种设定下可实现的收敛指数。 |
| [^6] | [The Implications of Linguistic Illegibility for LLM Security](https://arxiv.org/abs/2609.02852) | 本文提出“语言不可读性”概念，指出大语言模型的外部语言输出无法可靠反映其基于激活空间数学运算的内部计算，从而对依赖模型语言自我报告的安全机制构成根本性挑战。 |
| [^7] | [Post-Training Language Models for Gold-Medal Performance in Coding Competitions](https://arxiv.org/abs/2609.02849) | 该研究通过结合大规模题目筛选、监督微调、强化学习以及反馈驱动的测试时计算策略 GenCorrect，使语言模型在 IOI 2025 编程竞赛中取得了超越金牌分数线（438.3 分）的成绩（Nano-CC 达 468 分，Ultra-CC 达 502 分）。 |
| [^8] | [UE5M3 FP4 Block Scaling for Stable Language Model Pretraining](https://arxiv.org/abs/2609.02846) | 该论文提出将E2M1有效载荷与无符号E5M3块缩放配对、配合选择性随机舍入并省略随机Hadamard变换的FP4训练方案，在Nemotron-H 8B模型近1900亿token的预训练中取得了优于NVIDIA Transformer Engine NV配方的训练损失与验证损失。 |
| [^9] | [Learning Spectral-Like Mesh-Free Discretisations](https://arxiv.org/abs/2609.02833) | 该论文提出SpeND方法，将无网格离散化中欠定自由度的选择转化为学习问题，利用神经网络参数化模板权重，从而在细尺度波数处获得类谱方法的精度。 |
| [^10] | [AI Contextual Measurement for Recovering Individual and Group-Level Effects: Validation Against Survey Measures and an Occupational Application](https://arxiv.org/abs/2609.02821) | 提出AICOME框架，通过受访者层面的AI测量指标推导群体汇总值与个体偏差，从而同时恢复情境模型中的个体与群体层面效应，并利用2022年中国家庭追踪调查的职业数据及多项工作相关调查变量加以验证。 |
| [^11] | [Cliff: Learning Process Rewards from the First Mistake](https://arxiv.org/abs/2609.02817) | Cliff利用现成的大语言模型作为教师来定位每次推理中的第一个错误，将采样序列分解为正确前缀与错误后缀，从而在无需专门奖励模型、也无需假设师生推理模式一致的情况下，实现更有效的过程奖励塑形。 |
| [^12] | [Dutch Books for Language Models](https://arxiv.org/abs/2609.02797) | 该论文基于德·菲内蒂定理提出一种利用线性规划计算荷兰赌利润的评估方法，无需真实结果标签即可量化语言模型概率预测的不连贯性，并发现语言模型预测存在显著的不连贯现象。 |
| [^13] | [Full-Model Optimality for Tunable Linear Generative Priors in Compressed Sensing](https://arxiv.org/abs/2609.02790) | 本文针对压缩感知中通过奇异值分解相互关联的可调线性生成先验族建立了理论，证明在无噪声高斯压缩感知中，全维线性先验在整个先验族中达到最小的期望重建误差。 |
| [^14] | [CodePoisonRAG: Knowledge Poisoning Attacks on Retrieval-Augmented Code Generation](https://arxiv.org/abs/2609.02774) | 该论文提出CodePoisonRAG，一种针对检索增强代码生成的定向上游知识投毒攻击框架，通过CWE特定漏洞注入与语义错误标注，在不修改大语言模型的前提下将良性代码条目转化为传播攻击者选定弱点的投毒构件。 |
| [^15] | [From Reweighting to Rewriting: Unlocking the Intervention Effects of Influential Samples in Training Data Attribution](https://arxiv.org/abs/2609.02771) | 该论文发现重加权无法释放影响函数所识别样本的干预价值，并提出“影响引导的响应重写”方法——通过重写所选样本的响应而非调整其权重，从而真正解锁训练数据归因中影响力样本的干预效果。 |
| [^16] | [Do Tabular Foundation Models Know Physics? Contamination, Units, and the Deterministic Limit](https://arxiv.org/abs/2609.02766) | 本文在316个物理方程采样的数据集上评估四种表格基础模型，发现其性能虽全面超越基线模型，但其贝叶斯先验既无法表示无噪声机制也无法表示物理单位，因此只能对物理数据进行插值而不能充当真正的物理模型。 |
| [^17] | [Untangling the Mechanisms of Misleading Context in Medical Question Answering](https://arxiv.org/abs/2609.02754) | 该研究通过在MedMisBench中注入伪造证据和纯粹断言两类误导性上下文，系统揭示了推理模型的医学判断被误导的机制，发现模型对纯粹断言的易感性显著高于伪造证据（高出10至27个百分点），且误导信息虽在推理轨迹中被大量披露却难以被察觉。 |
| [^18] | [HiPoly: a hierarchical polymer-native AI framework for property prediction and generative design](https://arxiv.org/abs/2609.02746) | 提出HiPoly框架，基于G2RINS表示的三级分层图架构直接编码聚合物的随机连接性、组成和分子量，实现了从实验配方数据到性能预测、生成式分子设计和物理验证的端到端AI驱动聚合物设计工作流程。 |
| [^19] | [SPADE: SPaT Attack Detection from the Connected Vehicle's Perspective](https://arxiv.org/abs/2609.02741) | 本文提出了SPADE——首个专为网联汽车车载视角下的SPaT攻击检测深度学习研究设计的带标签多模态仿真数据集，填补了入侵检测研究在SPaT消息完整性防护领域的空白。 |
| [^20] | [Language Models Can Control Their Own Attention](https://arxiv.org/abs/2609.02737) | 该论文提出“声明式注意力”协议，让语言模型在思维链中自主声明需要关注的上下文区域，推理引擎据此像解析工具调用一样跳过大部分KV缓存读取，从而以内在方式避免了外部评分方法每步O(N)的开销。 |
| [^21] | [LoRA-TSD: Tangent-Space Spectral Descent for LoRA via Muon-Style Updates](https://arxiv.org/abs/2609.02734) | 该论文提出LoRA-TSD优化器，将LoRA更新视为固定秩矩阵流形上的切向量并在切空间内执行Muon风格的谱范数最速下降，通过仅需因子梯度、且比截断SVD收缩便宜最多2.8倍的收缩映射实现几何感知的LoRA微调，并证明其Frobenius范数版本可恢复LoRA-Pro。 |
| [^22] | [Momentum in large-batch training: Polyak enlarges the critical batch size, Nesterov improves data efficiency](https://arxiv.org/abs/2609.02728) | 该论文在幂律核回归框架下证明，在单遍大批量训练中 Polyak 动量可将临界学习率随批大小线性放大（从而扩大临界批大小约 1/(1-ρ) 倍），而 Nesterov 动量的临界学习率以 $B^\beta$（β>1）的更快速度增长，从而显著提升数据效率，并给出了刻画完整风险动力学的标度律与三区制批大小相图。 |
| [^23] | [Neural operators approximate strongly continuous convex monotone semigroups](https://arxiv.org/abs/2609.02727) | 该论文提出Chernoff神经算子与包络神经算子，通过学习单步算子实现了对强连续凸单调半群的万能逼近并给出定量逼近速率，在非线性偏微分方程、随机最优控制和模型不确定性下的随机过程等数值例子中验证了方法的有效性。 |
| [^24] | [H3DNAS: Hardware-Aware ONNX-Native 3D Point Cloud Model Compression](https://arxiv.org/abs/2609.02684) | H3DNAS是一个硬件感知的模型压缩框架，无需源代码即可直接在ONNX计算图上压缩3D点云模型，并通过通道依赖图和两阶段分层搜索实现边缘设备上的高效部署。 |
| [^25] | [Eliciting ESG Preferences for Reinforcement Learning-Based Portfolio Optimization](https://arxiv.org/abs/2609.02677) | 该论文将ESG投资组合优化建模为同时融合三家ESG评级机构数据的多目标强化学习问题，并提出基于高斯过程的偏好引导框架，让从业者通过直观的成对比较推断自身潜在效用函数，从而解决单一评级源和人工权重设置的非直观性问题。 |
| [^26] | [oHC: Orthogonal Hyper-Connections on SO(4) via Quaternions](https://arxiv.org/abs/2609.02672) | 该论文证明了双随机矩阵约束的混合会随深度耗尽残差流的多样性，并提出通过四元数在SO(4)流形上构造正交混合矩阵的oHC方法，既保证缩放稳定又完整保持残差流的范数与多样性。 |
| [^27] | [Dimension Dependent Correlation Gap Bounds under Restricted Independence](https://arxiv.org/abs/2609.02659) | 本文利用AI辅助的理论分析与计算验证相结合的证明方法，证明了成对独立性下单调次模函数的相关性差距在 $n=4$ 时普遍成立且达到紧的 $4/3$ 上界，解决了此前遗留的开放问题。 |
| [^28] | [Unfolding the Leech Lattice: Fused Multi-Shell Decoding and VRAM Layouts for 2-Bit LLM Weights](https://arxiv.org/abs/2609.02652) | 本文首次实现了Leech格量化所必需的多壳解码器，通过融合GPU内核与优化的显存布局（发现显存内码率是独立于磁盘码率的设计维度），使2比特LLM权重在batch 1解码阶段实现高效推理。 |
| [^29] | [Loom: Weaving Diagnostic Strands into Free-Text Consensus via Embedding-Space Reweighting](https://arxiv.org/abs/2609.02649) | Loom是一个部署于真实工业根因分析的生成式共识框架，它将模块化启发式产生的开放式诊断假设投影到连续嵌入空间，并通过基于质心的迭代重加权算法解决冲突信号，从而把嘈杂矛盾的文本假设聚合为可靠共识。 |
| [^30] | [Differentiable Electricity-Market Clearing for Gradient-Based Planning](https://arxiv.org/abs/2609.02646) | 本文提出将电力市场出清建模为可微优化层，通过反向模式自动微分将规划成本梯度经出清电价回传至规划方案，从而实现基于梯度的数据中心负荷选址优化。 |
| [^31] | [TaRA: Training-Aware Low-Rank Adaptation Initialization](https://arxiv.org/abs/2609.02639) | TaRA提出了一种训练感知的LoRA初始化方法，通过使低秩因子诱导的梯度密切逼近全秩权重矩阵的梯度来提升训练初期的梯度保真度，且几乎不增加计算开销。 |
| [^32] | [Oracle, will I ever learn? A study of prediction convergence and complementarity across link prediction models](https://arxiv.org/abs/2609.02638) | 该论文提出用“神谕”方法衡量链接预测模型的互补性——即针对每个查询在多个模型中选取最佳预测，以此量化不同模型（甚至同一模型的不同训练过程）所捕获知识的互补程度，并给出模型组合可达到的性能上限。 |
| [^33] | [Scalable Direction-Following TTS via Voice Impression-Guided Pseudo Triplet Construction](https://arxiv.org/abs/2609.02623) | 提出一种利用印象可控语音合成模型与大语言模型自动构建（参考语音、方向文本、修改后语音）伪三元组的可扩展流水线，解决了方向跟随语音合成中训练数据稀缺的问题，仅凭伪数据即可实现稳定的说话人特征保留式风格修改。 |
| [^34] | [Source Distribution Estimation by Posterior Averaging](https://arxiv.org/abs/2609.02622) | 该论文提出用期望最大化框架求解源分布估计问题，E步在当前源估计生成的新鲜仿真数据上训练摊销后验，M步将后验平均值重新拟合为源分布，从而避免了对一次性训练的似然代理模型的依赖。 |
| [^35] | [Learning-Based Reconstruction Attacks on Coordinate-Obfuscated Point Clouds](https://arxiv.org/abs/2609.02568) | 本文评估了点云选择性坐标加密方案在面对机器学习重构攻击时的安全性，证明攻击者可利用未加密数据中的空间与几何相关性，在不解密的情况下恢复被加密的坐标，从而揭示了该加密框架的安全隐患。 |
| [^36] | [Online Reinforcement Learning in the Met Office Unified Model through Distributed Model-Agent Coupling](https://arxiv.org/abs/2609.02566) | 该研究通过分布式模型-智能体耦合将强化学习智能体嵌入英国气象局统一模式，实现在线训练有界的位温订正，在保持数值稳定性的同时于多数纬度带降低了500百帕位势高度的平均绝对误差。 |
| [^37] | [ProbeMatchDTI: Probe-Driven Multi-Scale Biochemical Pattern Matching for Drug-Target Interaction Prediction](https://arxiv.org/abs/2609.02549) | 提出探针驱动的ProbeMatchDTI框架，通过IterProbe和BindingProbe保留被传统方法抑制的弱结合相关生化信号，实现多尺度药物-蛋白质模式匹配，从而改进药物-靶点相互作用预测。 |
| [^38] | [Learn from Whoever Is Right: Answer-Verified Multi-Teacher Distillation for Multi-Domain LLMs](https://arxiv.org/abs/2609.02548) | 提出了MT-SDPO方法，通过答案验证机制按样本识别最可靠的教师（而非仅依赖领域匹配路由），将多个冻结的领域专家模型的能力蒸馏统一到一个学生模型中。 |
| [^39] | [TrajMind: Chaining Role-Specialized LoRAs for Fast-and-Slow Collective Trajectory Anomaly Diagnosis](https://arxiv.org/abs/2609.02540) | TrajMind提出快慢双路径框架，在单一冻结的视觉-语言骨干网络上链式切换三个角色专精LoRA适配器，将持续在线筛查与按需诊断解耦，从而实现低延迟且经源数据可验证的集体轨迹异常诊断。 |
| [^40] | [A Comparative Study of Graph Representations for GNN-Based Power Grid Control in L2RPN](https://arxiv.org/abs/2609.02538) | 本研究在L2RPN环境中对多种图表示方法进行了受控比较，发现使图复杂度与任务粒度相匹配比追求更丰富的图表示更为重要。 |
| [^41] | [Spectral Initialization and Scheduled Graph Smoothness for Uncertain Knowledge Graph Completion](https://arxiv.org/abs/2609.02519) | QUEST通过利用置信度加权图拉普拉斯算子特征向量进行实体嵌入谱初始化，并结合无偏小批量狄利克雷能量正则化来调度图平滑，在不增加可训练参数的情况下显著提升了不确定知识图谱补全的置信度预测和链接预测性能。 |
| [^42] | [Orthogonal Ensembles and Tested Explanations for Performer-Independent Body-Motion Emotion Recognition](https://arxiv.org/abs/2609.02510) | 在留一表演者的困难评估设置下，通过组合十一个误差模式正交的模型，将12类身体运动情感识别的Macro-F1提升11.07个百分点，并提供了一套经过实验验证的解释方法，证明模型决策基于运动的身体区域证据且与拉班动作分析（LMA）属性高度一致。 |
| [^43] | [Rethinking the Teacher-Student Framework for Test-Time Adaptation](https://arxiv.org/abs/2609.02507) | 该论文挑战了测试时自适应中基于EMA的教师权重更新策略，指出误差累积在长序列场景下依然存在，并提出使用权重固定不更新的“顽固”教师，从而显著提升TTA方法在更长测试场景下的性能与稳定性。 |
| [^44] | [Training seeds and model-selection stability in recommender-system evaluation](https://arxiv.org/abs/2609.02499) | 推荐系统评估实验中，训练种子的变化往往会产生可检测的影响，仅报告单一随机种子的结果可能夸大评估结论的稳定性。 |
| [^45] | [RINSE: Robust Target-Time Normality Estimation for Zero-Shot Graph Anomaly Detection](https://arxiv.org/abs/2609.02497) | RINSE提出了一种无需梯度的零样本图异常检测框架，在保持源检测器固定的同时，通过迭代估计目标正常性、表示校准和证据可靠性，并结合可靠性门控秩融合与编码器集成，实现了对未见目标图的鲁棒异常检测。 |
| [^46] | [DeepAffinity: Long-Term Aspect Preference Prediction in eCommerce using Small Language Models](https://arxiv.org/abs/2609.02468) | 提出DeepAffinity框架，利用结构化提示和专用预测头微调的小语言模型（SLM）从用户时序交互历史中预测其对品牌、尺寸、颜色等产品属性的长期偏好，性能优于标准生成式微调方法，并能提升大规模推荐质量。 |
| [^47] | [Scalable Kronecker-Fisher Approximation: Efficient Hessian Analysis for Billion-Parameter Language Models Compression](https://arxiv.org/abs/2609.02451) | 本文提出一种可扩展的Kronecker-Fisher近似方法，无需存储完整Fisher矩阵即可对十亿参数语言模型进行高效Hessian分析，发现值投影层是最脆弱的组件，为混合精度分配等压缩与优化策略提供了实用的理论工具。 |
| [^48] | [CACTUS: Mask-Guided Semantic Clean-Label Backdoors in Decentralized Federated Learning](https://arxiv.org/abs/2609.02450) | CACTUS提出了一种掩码引导的语义干净标签后门攻击方法，通过将标签一致的语义对转换为目标导向的表示偏移，并在去中心化联邦学习的对等聚合前以反事实方式施加到干净非目标嵌入上，从而在多种模态和聚合规则下实现高攻击成功率。 |
| [^49] | [Towards One-for-All Robustness Across a Continuum of Threat Levels](https://arxiv.org/abs/2609.02440) | 提出威胁条件网络（TCN），通过将表示学习分解为威胁不变的共享骨干与轻量的威胁条件自适应模块，使单一模型无需针对不同攻击预算训练多个专门模型，即可在无限连续的扰动强度范围内实现鲁棒性。 |
| [^50] | [When Decodability Is Not Enough: Logical Validity Representations, Behavioral Dissociation, and Causal Tests in Language Models](https://arxiv.org/abs/2609.02438) | 该研究发现即使大语言模型在逻辑验证任务上的行为表现接近随机，其隐藏状态中仍能近乎完美地解码出逻辑有效性信息，但因果干预显示这种表征并未被模型实际利用，揭示了“可解码性不等于因果性使用”这一重要结论。 |
| [^51] | [IFW-BLS: Dual-Robust Broad Learning System with Intuitionistic Fuzzy Wave Loss](https://arxiv.org/abs/2609.02422) | 本文提出IFW-BLS，通过将有界、平滑、非对称的波浪损失与直觉模糊加权机制结合在单一优化模型中，同时实现针对大残差和低可靠样本的双重鲁棒宽度学习系统。 |
| [^52] | [Coverage, Not Targeting: A Structural Regime in Multi-Turn Agent Credit Assignment](https://arxiv.org/abs/2609.02417) | 论文提出“验证器信息密度 V_d”这一结构性指标，揭示当终端状态验证器处于低 V_d 区间时，多轮智能体信用分配的关键在于奖励覆盖（均匀稠密分布）而非对关键轮次的精准瞄准。 |
| [^53] | [Evidence for Shared Routing Geometry and Dynamics in Sparse Mixture-of-Experts](https://arxiv.org/abs/2609.02404) | 该论文证明了稀疏混合专家模型中各层路由器的状态共享一个被层特定坐标系掩盖的共同几何结构，通过对齐后单一线性模型即可解释大部分跨层路由状态演化。 |
| [^54] | [A computational approach to maximum likelihood thresholds for colored Gaussian graphical models](https://arxiv.org/abs/2609.02382) | 本文针对有色高斯图模型，通过几何表述建立统一理论框架并提出新的符号算法，解决了其最大似然阈值的计算问题。 |
| [^55] | [Percolation Dynamics in Optimization : Variance Cascades and Discrete Scale Invariance](https://arxiv.org/abs/2609.02373) | 该论文将随机梯度下降的动力学建模为渗流过程，揭示出架构对称性迫使子网络以离散同步块方式合并，其结构转变在宏观序参量上表现为类似物理相变的方差尖峰，且该捕获机制及标度级联在重尾噪声模型下同样适用于Adam和AdamW。 |
| [^56] | [Humanoid Safe Stop via Learned Stoppability Value](https://arxiv.org/abs/2609.02358) | 提出Safe-Stop框架，将人形机器人紧急停止建模为可达-避障问题，通过结合停止概率估计器与哈密顿-雅可比可达-避障估计器这两个互补的可停性信号，实现任务无关、可跨上游任务免训练迁移的安全停止决策。 |
| [^57] | [AGI Maze Prediction Datasets: A Compact Benchmark for Learning World Dynamics with Transformers](https://arxiv.org/abs/2609.02339) | 该论文提出了一个基于程序化生成迷宫世界的轻量级基准数据集，通过迷宫不相交的任务划分来检验Transformer是真正学到了可迁移的世界动力学规律，还是仅仅记忆了熟悉布局。 |
| [^58] | [Poisoning Attacks on the PGM-index](https://arxiv.org/abs/2609.02328) | 本文提出了首个针对学习型索引PGM-index的投毒攻击PGM-attack，仅通过插入10%的对抗性键即可使索引分段数量膨胀至多120倍，并首次推导出任意插入下分段数的理论上界，证明该攻击至少能达到最优攻击效果的52%。 |
| [^59] | [What Is Worth Representing? Representational Empowerment for Continual Model Construction](https://arxiv.org/abs/2609.02322) | 该论文提出“表征赋权”框架，通过评估候选表征元素能多大程度扩展智能体未来的建模与规划能力，来决定“什么值得被表征”，从而实现跨环境的持续模型构建。 |
| [^60] | [Bayes-Optimal BER and AUC: Estimation and Evaluation of Estimators](https://arxiv.org/abs/2609.02304) | 该论文提出了基于软标签来估计贝叶斯最优平衡错误率（BER）和AUC的新方法，并研究了如何评估这些估计量，从而在类别不平衡等准确率失效的场景中衡量模型性能的理论上限。 |
| [^61] | [Improving Evaluation Realism with Inference-Time Compute and Deployment Scaffolds](https://arxiv.org/abs/2609.02302) | 该论文提出“批判式精炼”和 DISH 智能体框架两种技术，通过投入额外推理时计算并模仿真实部署环境，使模拟对齐评估更难被能力强模型识别为测试，从而提升安全评估的真实性与结论可靠性。 |
| [^62] | [SEAL: Reinforcing Global Safety in Mixture-of-Experts through Shared Expert ALignment](https://arxiv.org/abs/2609.02293) | 本文提出 SEAL 方法，通过对共享专家进行安全对齐，为混合专家模型提供不依赖于稀疏路由路径的全局安全防护，有效抵御越狱提示、恶意微调以及安全关键神经元剪枝等对抗性攻击。 |
| [^63] | [From topology learning to graph generation: A unifying perspective](https://arxiv.org/abs/2609.02286) | 本综述提出统一框架，将图拓扑学习与图生成视为同一图数据生成过程的逆问题，从而连接了这两个长期平行发展的研究方向。 |
| [^64] | [Entangled Representations Amplify Collateral Damage in Unlearning](https://arxiv.org/abs/2609.02285) | 该研究首次通过受控实验验证了表示纠缠会加剧机器遗忘的附带损害——通过训练知识域解耦程度不同的语言模型套件，证明更解耦的模型在固定遗忘水平下保留成本可降低约4倍。 |
| [^65] | [Do Large Language Models Capture the Diversity in their Training Data?](https://arxiv.org/abs/2609.02275) | 该论文提出一种基于信息论的方法，通过比较模型生成输出与训练数据的条件熵，发现大语言模型（如OLMo、Pythia和GPT-Neo）生成内容的多样性系统性地低于其训练数据的多样性。 |
| [^66] | [CAPTURE: Disentangling Preference Drift from Memory Poisoning in Personalized LLM Agents](https://arxiv.org/abs/2609.02265) | 提出 CAPTURE 框架，通过神经微分方程信念追踪器、多时间尺度记忆账本、不确定性触发的澄清机制和反事实记忆审计，使个性化 LLM 智能体能够有效区分真实偏好漂移与对抗性记忆投毒攻击。 |
| [^67] | [Codebook Agent: Amortized Topology Design for LLM Multi-Agent Systems](https://arxiv.org/abs/2609.02264) | 该论文揭示当前将多智能体系统拓扑设计视为条件图生成的范式存在根本性缺陷——有效拓扑种类极少、边数无法反映真实token成本、消息传递评分器无法区分共享配置的智能体——并提出基于码本的摊销式拓扑设计作为替代方案。 |
| [^68] | [RideSkill: A Hierarchical Algorithm for Generalized Ride Sharing with LLM-Driven Automatic Evolution](https://arxiv.org/abs/2609.02250) | 该论文提出RideSkill，一种由大语言模型驱动自动进化的分层算法，用于解决泛化拼车问题，克服了传统多智能体强化学习方法在泛化性、可迁移性和大规模训练方面的局限。 |
| [^69] | [LLM-as-a-Judge Is Not an Oracle: Why Self-Improving Agents Need Deterministic Guardrails](https://arxiv.org/abs/2609.02246) | 本文提出LLM裁判不应作为自我改进智能体的最终评估权威，而应降级为顾问，由裁判无法推翻的确定性验证层把关所有变更，并以生产环境中归纳出的四类十一种评估失效模式（包括读取缓存答案的奖励破解行为）支撑这一立场。 |
| [^70] | [Similarity-Aware Personalized Federated Learning in Heterogeneous Environments](https://arxiv.org/abs/2609.02241) | 提出SAPE-FL框架，通过将客户端模型同时锚定到全局模型和相似性加权的同伴平均模型，并利用基于模型与输出相似性的动态个性化正则化，在异构数据分布下自适应平衡全局知识迁移与同伴协作、过滤不相似客户端。 |
| [^71] | [Recursive Value Learning for Long-Horizon Offline Goal-Conditioned RL](https://arxiv.org/abs/2609.02237) | DCRL通过将轨迹段递归分解为平衡二叉树并从叶到根训练价值，将自举深度从线性降至对数级，从而有效提升长时程离线目标条件强化学习的性能。 |
| [^72] | [Hardware-Accelerated Instance Segmentation for Resource-Constrained Space Robotics with Criticality Analysis](https://arxiv.org/abs/2609.02219) | 该论文提出面向资源受限月球机器人的硬件加速实例分割框架，通过AVIS无标签校准策略与DPU上经过架构优化的YOLO分割模型部署，在低光照、有限算力和辐射故障三重约束下实现有界延迟的实时感知。 |
| [^73] | [Prototype-guided transfer of sparse literature knowledge for electrolyte additive discovery](https://arxiv.org/abs/2609.02209) | 该研究提出了文献驱动的ProtoMI框架，通过图对比学习从稀疏的文献报道电解液添加剂中提取化学可解释的结构原型，并利用原型引导的半监督对比学习将其迁移到庞大的未标注化学空间，从而高效实现对电解液添加剂候选分子的优先排序。 |
| [^74] | [SMart: A Multi-source Multi-phase Time Series Representation Transfer Framework](https://arxiv.org/abs/2609.02203) | 本文提出SMart框架，通过多阶段递归图恢复任务和源数据集选择器这两种新颖机制，实现了利用多源数据进行多阶段时间序列表示迁移的改进。 |
| [^75] | [Schr\"odinger Bridges on Lie Group Manifolds for Probabilistic Intrinsic Generation](https://arxiv.org/abs/2609.02196) | 该论文将薛定谔桥推广到李群流形上，实现了在弯曲几何空间中直接进行概率生成建模，允许仅约束部分可观测端点变量，并针对紧致阿贝尔群与非阿贝尔群分别提出了WKBC和RCCBM两种计算方法。 |
| [^76] | [Learning the Constitutive Behavior of Materials via Neural Operators and Causal Attention: Case Studies in Plasticity and Damage](https://arxiv.org/abs/2609.02194) | 提出一种基于神经算子与因果注意力的数据驱动本构建模框架，将材料视为从应变历史到应力响应的算子映射，无需内状态变量即可直接从应变-应力数据学习塑性与损伤本构行为。 |
| [^77] | [Quantum MeanFlow: single-shot generative sampling on NISQ hardware](https://arxiv.org/abs/2609.02186) | 该论文提出量子MeanFlow（QMF），通过学习平均速度场将量子流匹配所需的 多步顺序求解简化为单步生成采样，从而规避了NISQ量子硬件上高输入/输出成本带来的顺序电路提交问题。 |
| [^78] | [WeaveMark: Robust and Scalable Multi-bit LLM Watermarking via Coded Payload Spreading](https://arxiv.org/abs/2609.02177) | WeaveMark通过每令牌多比特载荷扩展、软判决纠错码和无偏多层重加权三大技术，突破了多比特LLM水印中载荷容量、提取准确率与文本质量之间的固有权衡，并引入零比特层实现可靠的水印存在性检测。 |
| [^79] | [Breadth Beats Depth: Improving GCG-Based Jailbreak Optimization with Breadth-Oriented Suffix Search](https://arxiv.org/abs/2609.02172) | 本文提出即插即用框架BOSS，通过尾部聚焦对抗损失和面向广度的后缀搜索策略改进基于GCG的越狱攻击优化，在提升攻击成功率的同时降低了优化时间。 |
| [^80] | [DMRL: Document-Mediated Reinforcement Learning for Skill Optimization in Advertising Recommendation](https://arxiv.org/abs/2609.02170) | 本文提出DMRL框架，将广告推荐中大语言模型技能文档的优化建模为上层智能体的结构化编辑动作序列，借助冻结的下层任务智能体A/B测试反馈和双相对策略优化（DRPO）解决信用分配问题，实现技能文档的自我演化优化。 |
| [^81] | [GenCAR: Generative Counterfactual Alignment with Risk-Controlled Selection for Out-of-Distribution Recommendation](https://arxiv.org/abs/2609.02162) | 提出GenCAR框架，将分布外推荐形式化为α-有效反事实推荐（α-VCR）问题，通过基于偏好的反事实监督与基于保形p值的Benjamini-Hochberg集合选择，在控制代理标签错误发现率（FDR）的同时提升推荐效用。 |
| [^82] | [GeoSPRINT: Geometric Redundancy-Aware Step Pruning for Inference in Diffusion Trajectories](https://arxiv.org/abs/2609.02160) | GeoSPRINT是一个免训练的扩散模型推理加速框架，通过潜在空间的超平面性测试（QR分解实现）检测去噪轨迹中几何冗余的步数，并据此构建非均匀采样调度，在高曲率区域分配更多步数，从而在保证样本质量的同时大幅减少采样步数。 |
| [^83] | [Exact Limits of Random Projections for Preserving Geometry: Distance Recovery, Nearest-Neighbor Rankings, and Covariance Shape in Gaussian Models](https://arxiv.org/abs/2609.02155) | 该论文揭示Johnson-Lindenstrauss引理在高维中可能对保留的几何结构毫无信息量，并通过最优解码器理论精确刻画了高斯模型下从随机投影中恢复距离特征、最近邻排序与协方差形状的根本极限。 |
| [^84] | [Online Non-Monotone DR-Submodular Maximization Matching the Offline $0.401$ Factor](https://arxiv.org/abs/2609.02145) | 该论文首次在对抗性在线设置下实现了非单调DR-次模最大化与非离线算法相同的0.401最优近似比，通过用加权在线学习器替代离线箱约束步骤并结合精确非对称平衡定理，在决策后全信息值预言机模型下达到次线性近似遗憾。 |
| [^85] | [A Power Law in Logarithm's Clothing: On the Scalability of Graph-Based Vector Search](https://arxiv.org/abs/2609.02143) | 该论文通过跨数据集规模的实测推翻了“搜索成本随数据规模呈多重对数增长”的通行说法，发现当数据规模相对内在维度较小时，基于图的向量搜索成本实际遵循次线性幂律（约按 $N^c$、$0<c<1$ 增长），只有规模足够大时才收敛到理论预言的对数式增长。 |
| [^86] | [SoK: Where Do Flow Labels Come From? Auditing Label Provenance in Encrypted Traffic Benchmarks](https://arxiv.org/abs/2609.02140) | 该论文首次系统化审计了加密流量分类基准中的标签来源问题，揭示了“粗粒度继承”与“过严过滤”两类标签策略的固有风险，并推导出仅使用严格侧信道特征的分类器所能达到的平衡准确率理论上限。 |
| [^87] | [HyperMC: Multi-Fidelity Hyperparameter Tuning for Stochastic Gradient MCMC](https://arxiv.org/abs/2609.02138) | 提出了HyperMC框架，将Hyperband风格的资源分配与核Stein差异评估相结合，为缺乏Metropolis-Hastings接受率的SGMCMC方法实现高效的多保真度超参数调优，并通过全局网格初始化与精英引导局部细化增强了鲁棒性。 |
| [^88] | [Scalable Bayesian Optimization of Composite Functions for Image-Based Inverse Problems in Materials Characterization](https://arxiv.org/abs/2609.02126) | 本文提出一种可扩展的复合函数贝叶斯优化方法（SBOCF），利用图像匹配目标的复合结构和模拟图像的中间信息，高效地从PACBED图样中估计样品厚度和晶体失倾等关键参数，克服了网格搜索效率低下和神经网络需大量预训练且难以迁移的缺点。 |
| [^89] | [Disease Burden over Skin Tone: Decomposing the Dermatology-AI Generalization Gap](https://arxiv.org/abs/2609.02111) | 该研究发现，皮肤病学AI模型泛化能力差的主要原因是疾病分布偏移而非肤色代表性不足，表明扩大疾病覆盖范围比增加肤色多样性对提升模型泛化更为关键。 |
| [^90] | [A Computational Comparison of Fourier Spectral Differentiation and Spatial Automatic Differentiation in Periodic Physics-Informed Neural Networks](https://arxiv.org/abs/2609.02110) | 该论文通过严格控制变量的配对实验，比较了周期PINNs中的空间自动微分与傅里叶谱微分，表明傅里叶谱方法通过谱乘法求导并复用傅里叶系数，可在需要多阶或高阶空间导数时显著降低计算与内存开销。 |
| [^91] | [A Unified Rate-Distortion Perspective on Vector, Product, and Scalar Quantization](https://arxiv.org/abs/2609.02107) | 本文提出统一的率失真框架来分析离散视觉分词中的矢量、乘积和标量量化，证明最小化失真（而非最大化码本利用率）才是重建保真度的核心目标，并确立了量化方法公平比较所需的两个条件。 |
| [^92] | [Federated LoRA Adaptation of BiomedCLIP Across Four International Chest X-Ray Cohorts](https://arxiv.org/abs/2609.02101) | 该研究在横跨三大洲的四个公开胸片队列上验证了BiomedCLIP的联邦LoRA参数高效微调，在无需交换患者数据的前提下将共享类别AUC平均从0.687提升至0.802，为异质医疗机构的隐私保护型多模态模型适配提供了基准。 |
| [^93] | [Compositional Spectral Prompts for LLM-based Online Time Series Forecasting](https://arxiv.org/abs/2609.02093) | 提出CoSPOT框架，通过冻结大语言模型并采用基于频域的组合式频谱提示，以极少的参数更新实现高效的在线时间序列预测，克服了现有方法难以长期适应和无法泛化到未见模式的局限。 |
| [^94] | [IDEEA: training-free Input-Dependent stEEring via Activation cluster matching](https://arxiv.org/abs/2609.02089) | 提出IDEEA框架，通过对每个注意力头的正负激活进行聚类并求解最优匹配问题来构建簇条件化的引导方向，首次实现了无需训练、随输入自适应变化的大模型激活引导，克服了传统固定单一方向引导的根本局限。 |
| [^95] | [TC-Next: Zero-Shot Multimodal Cyclone Forecasting](https://arxiv.org/abs/2609.02085) | TC-Next是一个仅在西太平洋GraphCast预报上训练、只使用通用大气变量的多模态深度学习模型，能大幅降低热带气旋路径和强度预报误差，并可零样本直接迁移到Pangu-Weather、IFS HRES和WeatherNext Cyclones等其他天气基础模型上，性能超越传统规则追踪器乃至专门的直接追踪器。 |
| [^96] | [XMerge: Cross-Axis Selection and Reconstructive Layer Merging for LLM Depth Compression](https://arxiv.org/abs/2609.02083) | XMerge 是一种训练后的大语言模型深度压缩方法，通过跨轴选择识别隐藏状态变化最小的层块，并利用局部边界重构重新拟合相邻存留块，在不改变架构、不增加推理参数、无需任务标签的情况下实现高质量的层删除压缩。 |
| [^97] | [DynG-Diff: A State-Aware Dynamic Guidance Diffusion Framework for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2609.02068) | DynG-Diff提出了一种状态感知的动态引导扩散框架，通过两阶段分离训练策略建模多变量时间序列联合分布，并利用轻量级策略网络自适应推断变量可靠性以输出动态引导强度，从而解决了概率多变量时间序列预测中变量间信息异质性的问题。 |
| [^98] | [DocHop: Benchmarking Out-of-domain Multi-hop Reasoning in Information-Dense Documents](https://arxiv.org/abs/2609.02059) | 提出了DocHop基准，用于评估多模态大语言模型能否在信息密集的文档图像中，利用文本叙述上下文解析目标实体，并跨多个图表进行多跳证据聚合与推理。 |
| [^99] | [The Dynamics of Continuous Mixture Collapse in Language Models](https://arxiv.org/abs/2609.02049) | 该研究揭示了语言模型无法保持连续混合推理状态的深层原因，识别出三种相互独立的失败机制：transformer 架构对混合几何结构的固有扭曲、训练过程对这种扭曲的显著放大，以及 softmax 读出与自回归反馈构成的动力系统导致混合分量被单一主导或坍缩至不可区分。 |
| [^100] | [Act More, Decide Less: Skill-Guided Adaptive Action Chunking for Long-Horizon LLM Agents](https://arxiv.org/abs/2609.02042) | 提出SPACE方法，通过从成功轨迹中归纳两级程序化技能并以子技能边界作为块边界监督进行蒸馏，使长程LLM智能体学会自适应的可变长度动作分块，克服了标准强化学习无法学习块边界导致的单动作退化或过长序列问题。 |
| [^101] | [Source-Free Class Relearning: Diagnosing Forgetting in Class Unlearning](https://arxiv.org/abs/2609.02018) | 本文提出一种严格无源的类别重学习方法，证明仅凭遗忘后的模型本身、对合成探测集进行单步梯度更新分类器头即可恢复被遗忘的类别，从而诊断出类别遗忘并未真正抹除类别结构。 |
| [^102] | [Perceptually Regularized Diffusion Model for Image Super-Resolution](https://arxiv.org/abs/2609.02016) | 本文提出一种感知正则化扩散模型，在扩散模型训练中引入感知正则化，以克服标准像素域噪声预测损失导致的过度平滑问题，从而提升图像超分辨率的感知质量。 |
| [^103] | [Train What You Deploy: Closing the MLP Reachability Gap in Low-Rank Clone Distillation](https://arxiv.org/abs/2609.02006) | 该论文提出“训练你所部署的”原则，让训练直接覆盖完整部署矩阵而非教师诱导的权重切片，在不增加任何推理成本的前提下释放低秩克隆蒸馏中62.5-81.4%被困住的容量，在三个教师模型上取得显著性能提升。 |
| [^104] | [Posterior Tempering Explains Variance Inflation in Linear and Generalized Linear Thompson Sampling](https://arxiv.org/abs/2609.01999) | 该论文提出 α-TS 算法，通过用 α-后验（分数幂后验）替代标准后验来形式化方差膨胀思想，并给出了先验与奖励分布的一般正则性条件，使汤普森采样在广义线性老虎机中无需后验近似即可完成遗憾分析，且当 α ∝ d^{-1} 时达到了已知最优的 O(d^{3/2}√T log T) 遗憾界。 |
| [^105] | [Linear Fusion MultiDiffusion for Fast Training-Free Spherical Panorama Generation](https://arxiv.org/abs/2609.01997) | LF-MultiDiffusion是一种免训练的球形全景生成方法，通过将潜变量聚合重新表述为正则化最小二乘问题并利用Krylov迭代求解器在去噪循环内求解，实现了比最强免训练基线更好的视觉质量、文本对齐和全景一致性，同时带来15.36倍的推理加速。 |
| [^106] | [CAHR-Net: Condition-Adaptive Hysteresis Reconstruction for Compact and Interpretable Magnetic Core Loss Modeling](https://arxiv.org/abs/2609.01991) | CAHR-Net通过特征级线性调制将频率、温度和波形等工作条件直接注入中间磁滞回线重构表征，在保留“波形→磁场→回线面积→损耗”可解释链条的同时，实现了紧凑而准确的磁芯损耗建模。 |
| [^107] | [Morphology signal in whole slide image foundation models can automatically triage slides](https://arxiv.org/abs/2609.01987) | 本文提出一种利用公开的WSI基础模型进行零样本分类、按肿瘤含量自动排序和分拣全切片图像的流程，无需病理学家人工筛选，解决了多切片数据集训练中信号稀释的问题。 |
| [^108] | [A Unified Particle Filter LSTM for Data-Driven Process Simulation](https://arxiv.org/abs/2609.01967) | 提出统一粒子滤波LSTM，通过维护并更新一组加权的循环状态假设来应对事件日志中潜在流程状态的不确定性，从而提升数据驱动流程仿真的真实性和准确性。 |
| [^109] | [Post-Training Ternarization of Qwen3-4B Capability, Effective Bit Budget, Storage Compression, and Deployment](https://arxiv.org/abs/2609.01962) | 本文对Qwen3-4B模型进行端到端后训练三值化，将量化线性权重压缩至每权重1.641有效比特（覆盖81.62%参数），实现显著存储压缩，但任务准确率从64.5%降至54.7%，且不同任务的能力退化程度不均匀。 |
| [^110] | [Network-Aware Forecasting on Wireless Access Points](https://arxiv.org/abs/2609.01957) | 该论文提出“网络感知可部署性”概念，通过在目标 AP 上的资格认证和负载约束下的执行验证两道门槛来评估预测模型的部署可行性，并发现边缘测试平台无法可靠反映目标无线接入点的真实运行表现。 |
| [^111] | [FlashKAN: B-Spline KANs via Truncated Power Form](https://arxiv.org/abs/2609.01956) | FlashKAN用逼近论中的截断幂形式取代Cox-de Boor递归，通过torch.compile融合为单一GPU内核并结合有界坐标稳定化，显著加速了KAN中B样条激活函数的计算，并提供了开源软件包。 |
| [^112] | [Convergence Theory of Knowledge Distillation in Asynchronous P2P Gossip Learning Network](https://arxiv.org/abs/2609.01952) | 本文首次为异步P2P八卦学习网络中的知识蒸馏建立了收敛理论，将共识从参数空间转移到函数空间，把KD事件建模为logit空间中的几何收缩算子，从而解决了不同参数规模架构的设备无法通过去中心化SGD达成共识的问题。 |
| [^113] | [On-Policy Distillation Meets Off-Policy GRPO: Training Compact Instruction-Following Rerankers](https://arxiv.org/abs/2609.01947) | 该论文提出两阶段蒸馏框架，先用离策略GRPO结合LLM裁判反馈强化4B教师重排序器，再让1B学生模型在自身采样的排序上接受教师软奖励实现在策略蒸馏，在分布偏移场景下取得最显著的性能提升。 |
| [^114] | [Pushing Forward Multi-Secret-Key Homomorphic Encryption for Private Average Aggregation](https://arxiv.org/abs/2609.01945) | 该论文提出了基于RLWE同态加密的轻量级多密钥协议，用于联邦学习中的私有平均聚合，既摆脱了单密钥方案对不共谋假设的依赖，又避免了多方同态加密方案中协作解密所需的大方差噪声带来的高开销。 |
| [^115] | [Sparse Readout Prism: Explaining Logit-Lens Scores in Features Instead of Tokens](https://arxiv.org/abs/2609.01936) | 该论文提出稀疏读出棱镜（SRP），仅利用读出矩阵自身的权重将其分解为稀疏“读出特征”，把logit-lens分数解释为特征贡献之和，从而消除了透镜读数对拟合语料库的依赖（语料库条件性），并支持跨词元、上下文、层与透镜的比较。 |
| [^116] | [OR-Transformer: Scaling Real-Time Decision-Making to 1,000 Items](https://arxiv.org/abs/2609.01933) | OR-Transformer通过商品置换等变的Transformer架构和路径梯度训练的深度强化学习方法，将随机需求下的联合补货决策扩展到1024种商品规模，随规模增长持续优于MILP基线并大幅降低在线决策时间。 |
| [^117] | [CRISP: Cliff-awaRe Input-adaptive Sparse Prefilling with Structural-Mass-Motivated Routing](https://arxiv.org/abs/2609.01925) | 该论文提出CRISP方法，用直接从代理注意力图结构中读取路由决策的结构代理C_struct替代JSD路由，解决了动态稀疏注意力路由中的两个结构性挑战，实现了长上下文LLM推理的高效输入自适应稀疏预填充。 |
| [^118] | [Basin Geometry and Reliable Recall of Dynamical Memories in Reservoir Computing](https://arxiv.org/abs/2609.01914) | 该研究揭示了储层计算联想记忆的吸引域具有“章鱼状”几何结构，线索驱动的广义同步能够绕过触手区域的不可预测性、将系统引入鲁棒的吸引域头部，从而实现动力学记忆的可靠召回，并给出了最小线索时长、同步速率与吸引域头部半径之间的定量关系。 |
| [^119] | [OutageDiT: A Generative Foundation Model for Power Outage Forecasting and Scenario Simulation](https://arxiv.org/abs/2609.01896) | OutageDiT是一个基于全美停电与天气数据训练的生成式基础模型，能以15分钟分辨率生成七天停电轨迹，在单一模型中统一实现停电点预测、不确定性量化和条件情景模拟。 |
| [^120] | [Latent unified smooth Hamiltonians for excited state chemistry](https://arxiv.org/abs/2609.01871) | 该论文提出一种通过间接学习电子态哈密顿量的潜在隐式基组表示来统一建模分子基态与激发态（包括锥形交叉点和非绝热耦合）的神经网络架构，并在胸腺嘧啶和偶氮苯光化学体系上准确再现了基态与低激发态的能量及振子强度。 |
| [^121] | [Import What You Need: Learning When and How to Augment EHR Graphs with External Knowledge](https://arxiv.org/abs/2609.01839) | 提出ReTA框架，利用强化学习在每次就诊时动态选择知识图谱增强动作（软导入、硬导入或跳过），以预算感知的方式解决现有EHR图增强方法固定且与上下文无关的拓扑增强问题。 |
| [^122] | [Interpretable Symptom Vectors for Depression in a Large Language Model](https://arxiv.org/abs/2609.01832) | 该研究通过机制可解释性技术发现大语言模型内部在第21层对抑郁症状产生几何分离，并构建“症状向量”将文本投影后得到各症状系数，其能保留临床医生标注的严重程度排序，从而增强LLM在抑郁症评估中的临床可信度。 |
| [^123] | [Reinforcement learning to choose optimizers](https://arxiv.org/abs/2609.01811) | 该论文提出将优化算法选择建模为序贯决策问题，利用循环强化学习策略在运行中动态决定下一步使用哪个优化器（涵盖基于梯度与无导数两类）及其持续时间，并在每次切换时传递当前最优解与代表性步长，从而摆脱了现有方法对切换时机、组合范围和决策频率的预设限制。 |
| [^124] | [hLLM: Single Pass Decoding for Generative Reranking](https://arxiv.org/abs/2609.01807) | 提出hLLM，通过轻量自注意力头从LLM预填充隐状态读取项目-位置得分矩阵，并用匈牙利算法求最优二分匹配，在O(1)次前向传播内一次性解码全部N个序数，从而将生成式重排序的解码从逐token自回归生成变为常数次前向传播，且天然保证输出为有效排序。 |
| [^125] | [D-FROST: Decentralized Federated pRompt-tuning via Optimal tranSporT for Non-IID and Imbalanced Data](https://arxiv.org/abs/2609.01802) | 该论文首次研究了去中心化联邦学习中的提示调优，提出基于最优传输的D-FROST算法，通过在提示测度上进行Wasserstein优化来捕捉提示的集合值结构，解决了非独立同分布和不平衡数据下提示集无法按索引对齐的难题，并具备达成共识的理论保证。 |
| [^126] | [Ten Architectures, One Error: Shared Failure Modes in Hyperspectral Classification under Spatially Disjoint Evaluation](https://arxiv.org/abs/2609.01786) | 该论文提出一种将空间分离与模型感受野关联的无泄漏评估协议，揭示随机像素划分导致准确率虚高的问题——十种高光谱分类架构在此协议下Macro-F1平均下降0.147且模型排名最多变动五位。 |
| [^127] | [Emergence of Fibrations, Compression, and Symmetry Breaking in Artificial Neural Networks](https://arxiv.org/abs/2609.01768) | 深度神经网络在学习过程中会自发涌现图论中的覆盖对称性，利用该对称性可将模型压缩至原大小的17%，而受控破坏该对称性则能克服持续学习中的可塑性丧失问题。 |
| [^128] | [Toward Explainable and Policy-Aware AI for Carbon Credit Price Prediction: A Research Framework for Emerging Carbon Markets](https://arxiv.org/abs/2609.01765) | 本文提出融合市场数据与政策文本的可解释碳价预测框架EPA-CarbonNet，但实证结果大多为负面——其预测精度不及随机游走基线，仅方向准确率（58.6%）优于所有基线。 |
| [^129] | [Pooling and Drift in Delayed Bandits](https://arxiv.org/abs/2609.01761) | 该论文发现当延迟老虎机的反馈结果仅通过动作所产生的状态依赖于动作时，学习代价由有效维度（真正不同的状态数量）而非动作数量决定，并据此证明了 $\widetilde{O}(\sqrt{(d+1)V\log K})$ 等新的遗憾界，突破了以往随动作数增长的界限。 |
| [^130] | [A Study of Conditional Diffusion Models for Open-Loop Control under Dry Friction and Stiction](https://arxiv.org/abs/2609.01756) | 本文提出的动作扩散方法利用条件一维U-Net生成时间连贯的有界控制序列，在干摩擦与静摩擦的开环控制基准中显著降低了终端误差和卡滞步数，尤其在低样本场景下优于随机采样和交叉熵方法。 |
| [^131] | [CAT-Flow: Curvature-Adaptive sTeps for Flow Matching](https://arxiv.org/abs/2609.01746) | 本文提出了两种无需训练的轻量级算法CAT-OV和CAT-OT，利用流匹配采样与梯度流之间的新颖联系，通过估计向量场曲率在推理时自适应调整步长，从而在不增加额外神经网络求值的前提下提升采样效率与质量。 |
| [^132] | [Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives](https://arxiv.org/abs/2609.01736) | 提出以自然语言取代API模式作为工具调用接口的“工具原语”设计，并构建包含25,519个函数的集中式仓库ToolFace供LLM在推理时动态检索工具，从而解决多步多轮推理脆弱及大规模工具目录下性能退化的问题。 |
| [^133] | [RecKAN: Kolmogorov-Arnold Networks with a Learnable Recursive Polynomial Basis](https://arxiv.org/abs/2609.01729) | RecKAN的核心创新在于通过一个五系数可学习的二阶多项式递推关系来定义KAN网络的基函数本身，该递推可涵盖多种经典多项式族作为特例，使学到的基能够超越任何固定的经典多项式选择。 |
| [^134] | [Hearing the Whispers: Black-Box Membership Inference Attacks on Finetuned TTS Models](https://arxiv.org/abs/2609.01723) | 该论文提出了首个专门针对微调TTS模型的黑盒成员推断攻击框架，解决了查询生成与语音表示工程中的独特挑战，以揭示个性化语音合成中的隐私风险。 |
| [^135] | [Generative Diffusion Surrogates with Analytical Variance Schedule](https://arxiv.org/abs/2609.01705) | 本文提出将输运问题中已知的解析方差（均方位移）的时间导数作为扩散模型的加噪速率，使生成时间成为经过校准的物理输运时钟，从而构建适用于随机输运的概率性、时间可分辨且能刻画非高斯分布结构的代理模型。 |
| [^136] | [Tri-Band Channel Measurement-Enabled Multi-Layer Digital Twin for Terahertz Wireless Data Centers](https://arxiv.org/abs/2609.01699) | 本文提出一种由140、220、300 GHz三频段信道测量校准的自底向上多层（物理、信道、评估、操控）数字孪生框架，用于太赫兹无线数据中心的高效无线规划与实时优化。 |
| [^137] | [FairLens: Benchmarking Fairness in Vision-Language Models for High-Stakes Decision-Making](https://arxiv.org/abs/2609.01691) | 该论文提出了FairLens基准框架，在招聘、法律和医疗三大高风险领域用超过10万个图像-问题对评估八个视觉语言模型，发现其主要缺陷是基于人脸做出无根据的推断，而非对不同群体的不平等对待。 |
| [^138] | [FORGE: Forward-Only Test-Time Adaptation for Integer-Only Vision Models on Microcontrollers](https://arxiv.org/abs/2609.01683) | 提出FORGE方法，首次实现仅前向传播的测试时自适应在微控制器上已部署的BN折叠纯整数卷积网络中运行，其核心是通过将折叠卷积的逐通道输出重新归一化到干净训练统计量，恢复因BN融合而丢失的自适应能力。 |
| [^139] | [Reinforcement Learning and Rule-Based Peer-to-Peer Pricing in Residential PV-BES Communities](https://arxiv.org/abs/2609.01680) | 在住宅光伏社区对等电力交易定价中，仅光伏配置下基于规则的定价优于最佳强化学习策略，但结合电池储能后最佳强化学习策略可将社区节约资金从734.23欧元提升至978.52欧元，且SDR形定价在学习型模式中表现最佳。 |
| [^140] | [A Survey on Self-Improving Test-Time Intelligence: Feedback-Driven Adapting, Learning, and Scaling at Inference](https://arxiv.org/abs/2609.01679) | 本综述提出“反馈驱动的测试时智能（TTI）”作为统一框架，将测试时适应、测试时学习和测试时扩展三大方向联系起来，系统阐释了AI模型如何利用测试时反馈和额外计算在部署阶段实现自我改进。 |
| [^141] | [Sim2Signal: Sim-to-Real Benchmarks for Traffic Signal Control](https://arxiv.org/abs/2609.01676) | 提出Sim2Signal基准，将交通信号控制中的仿真到现实差距分解为观测、动作、转移和奖励四类差距并逐一单独诱导，从而为系统评估缓解方法提供了标准化测试平台。 |
| [^142] | [Random Forest-Informed Cellular Automaton for Large-Scale Wildfire Spread Modelling](https://arxiv.org/abs/2609.01675) | 该论文提出一个融合随机森林与元胞自动机的三阶段框架，利用随机森林估计每日火灾发生概率并与邻域驱动蔓延规则结合，在5公里网格的大规模野火模拟中相比纯元胞自动机基线取得了显著更高的空间重叠度。 |
| [^143] | [CliffRank: A Dual-Branch Framework for Activity-Cliff Ranking Prediction](https://arxiv.org/abs/2609.01673) | CliffRank 提出了一种将绝对活性回归与排序一致性学习相结合的双分支框架，通过成对偏好一致性损失（PPC）在抗菌肽和小分子的活性悬崖排序预测任务上取得了领先的性能。 |
| [^144] | [Private Computation Space: Experience with Trusted Multi-Cluster Federated Learning for Agriculture](https://arxiv.org/abs/2609.01667) | 本文提出了已部署的开源机器学习系统私密计算空间（PCS），通过多集群编排、异步联邦学习与差分隐私等技术，在保护农民数据隐私和身份的同时，应对农村基础设施脆弱的挑战，推动人工智能在农业中的安全应用。 |
| [^145] | [Context Inference Attacks Without Jailbreaks](https://arxiv.org/abs/2609.01663) | 该论文提出并形式化了“上下文推断攻击”这一新型隐私威胁，证明智能体AI系统即使在没有越狱、且存在指令限制、logit抑制和上下文稀释等防御措施的情况下，仍会泄露其通过自身工具调用静默加载的敏感上下文信息。 |
| [^146] | [Efficient Context-Limited Telescope Bibliography Classification for the WASP-2025 Shared Task Using SciBERT](https://arxiv.org/abs/2609.01647) | 本研究提出一种基于SciBERT的高效方法，在严格的512 token上下文限制和有限计算资源下，将望远镜相关科学论文自动分类为四个类别，以0.89的宏F1分数在WASP-2025共享任务排行榜中位列第一。 |
| [^147] | [SocialBuddy: Tailoring Search Agent for Social Scenarios](https://arxiv.org/abs/2609.01641) | 该论文提出首个面向社交场景的智能体搜索框架SocialBuddy，并构建了包含20万用户画像、1000万社交帖子和5万推理轨迹的大规模模拟环境SocialEnv，以解决复杂社交搜索中的性能退化与稀疏奖励难题。 |
| [^148] | [Omega-N: Interpretable Structural Node Descriptors and Their Applicability Domain](https://arxiv.org/abs/2609.01633) | 本文提出 Omega-N，通过局部化三角指标因子并结合配置零模型超额与多尺度个性化 PageRank 邻域两项修正，得到每节点十个仅依赖图结构、无需属性与训练的可解释描述符，并验证了全局标量对标谱基线、逐节点归因在结构中心数量未知时更具优势的理论预测。 |
| [^149] | [Marginal Expected Revenue for Jointly Ranking Auction and Fixed-Price Listings in E-Commerce Sponsored Search](https://arxiv.org/abs/2609.01628) | 该论文提出边际eCPM（meCPM）方法，将传统固定价格商品的期望收益估算框架扩展至价格动态演化的拍卖和“拍卖加一口价”（ABIN）商品，实现了电商赞助搜索中多种上架形式的联合排序。 |
| [^150] | [PRISM: An Agentic Multi-Model Architecture for Proactive Safety in Autonomous Transportation Systems](https://arxiv.org/abs/2609.01623) | PRISM是一种智能体多模型安全架构，通过逆向事故概率建模将事故分类器转化为可解释的动态安全评分，并由推理层协调三个并发专门模型（轨迹运动学、环境风险、VRU交互），实现从被动事故规避到主动持续风险管理的转变。 |
| [^151] | [RecEvolve: A Knowledge-Driven Autonomous Agent System for Recommender Systems](https://arxiv.org/abs/2609.01622) | 本文提出知识驱动的自主智能体系统RecEvolve，将想法生成、代码实现、训练与评估的完整研究生命周期纳入持续自主闭环框架，首次在生产级大规模双塔召回模型上从零完成40余次自主训练迭代，突破隐藏架构瓶颈并取得NDCG约20%的相对提升，使线上用户满意度增长3.77%。 |
| [^152] | [When Literature Data Mislead Artificial Intelligence in Materials Discovery](https://arxiv.org/abs/2609.01621) | 该研究以固态电解质电导率数据为例，揭示了科学文献中普遍存在的文本-图表不匹配、单位不一致等看似合理却难以检测的错误，这些错误会以结构化标签噪声的形式污染AI训练数据库，甚至导致高达100倍的电导率误差。 |
| [^153] | [Multi-Agent Retrieval-Augmented Generation for Efficient Cloud Knowledge Base Search in Telecom SNOC Environment](https://arxiv.org/abs/2609.01618) | 本文提出了一个完全离线的多智能体RAG框架Athena，通过融合E5稠密检索、BM25稀疏检索和知识图谱扩展，并结合加权CombSUM融合与交叉编码器重排序，实现了电信SNOC环境中企业云知识库的高效精准检索。 |
| [^154] | [Hybrid Retrieval-Augmented Generation with Knowledge Graph Expansion, RRF Fusion, and Per-Chunk Grounded Evaluation for Enterprise Document Search](https://arxiv.org/abs/2609.01617) | DocuSearch 提出了一种混合检索增强生成系统，通过倒数排名融合（RRF）将稠密向量语义检索、BM25全文搜索与知识图谱邻居扩展三种互补证据源加权融合（权重分别为0.50、0.35、0.15），在电信网络运营生产环境中实现了更准确、有据可依的企业文档问答。 |
| [^155] | [Prompt-Space Meta-Learning Does Not Transfer Across Users: A Frozen-LLM Negative Result](https://arxiv.org/abs/2609.01615) | 本文提出 Muse 方法并得出负面结论：在冻结大语言模型上通过提示空间元学习获得的共享自适应提示无法跨用户迁移个性化能力，其收益主要来自通用指令质量而非真正的用户自适应。 |
| [^156] | [DiDrive: A Risk-Aware Hierarchical Diffusion Framework for Safe Offline Reinforcement Learning in Autonomous Driving](https://arxiv.org/abs/2609.01609) | DiDrive提出了一种融合风险感知分层扩散架构（RHDif）与3DICE策略优化范式的分布引导离线扩散框架，通过过滤状态冗余、聚焦安全关键威胁并缓解分布外动作高估，从而提升自动驾驶离线强化学习的安全性与稳定性。 |
| [^157] | [WMLLM: Self-Evolving Optimization Agents via Predict-Then-Act World Modeling](https://arxiv.org/abs/2609.01608) | 提出WMLLM自进化优化智能体框架，让大语言模型先预测有前景的优化方向再生成候选解，通过世界建模、多轮智能体改进、种群搜索与强化学习的结合，在黑盒优化中显著提升样本效率并持续改进其隐式世界模型与优化策略。 |
| [^158] | [Selective Agent Guidance via Entropy: Learning Autonomous Policies from Imperfect VLM Teachers](https://arxiv.org/abs/2609.01567) | 该论文提出SAGE框架，仅在智能体不确定时才查询昂贵的视觉语言模型教师，并利用环境优势对教师建议进行加权蒸馏，从而训练出无需教师引导即可自主行动的轻量级强化学习策略。 |
| [^159] | [Rethinking Learnability in Offline Data-driven Optimization](https://arxiv.org/abs/2609.01493) | 本文针对PAC可学习性无法充分刻画离线优化的理论缺陷，提出了“算法依赖的可学习性”这一新概念，其只需保证在优化器轨迹上的精度即可支撑离线数据驱动优化。 |
| [^160] | [Bandits in Prod: Hyperparameter Optimization at Inference Time](https://arxiv.org/abs/2609.01335) | 该论文将生产系统中只能通过线上噪声反馈评估配置的场景形式化为在线超参数优化（OHPO），提出通用框架IMABO及免重启的无限多臂老虎机策略IMOSS，并给出了分位数遗憾的理论保证。 |
| [^161] | [Subliminal Learning as Trait-Direction Drift: A Mechanism and Targeted Control under SFT Distillation](https://arxiv.org/abs/2609.01091) | 本文提出“特质方向漂移”机制来解释潜意识学习现象——偏置教师数据中可测量的偏好差距在监督微调中累积为学生的行为迁移，并据此提出探测空间走廊正则化这一针对性防御方法，在蒸馏过程中约束模型沿校准特质方向的漂移。 |
| [^162] | [Let Confidence Change, Not the Prediction: Prediction-Preserving Repair for Post-hoc Calibration](https://arxiv.org/abs/2609.01072) | 本文提出CORD——首个通过修复完整校准概率向量来严格保持原始top-1预测不变、仅修正置信度的事后校准后拟合适配器，并引入TPCR指标量化校准器改变预测的频率。 |
| [^163] | [VoiceLongMemEval: Do Assistants Remember How You Sounded?](https://arxiv.org/abs/2609.00570) | 该论文提出了VoiceLongMemEval（VLME）基准，用于评估AI助手在长时多会话对话中能否记住情感、韵律和语音事件等副语言信息，发现现有大语言模型普遍存在无法捕捉说话方式的“情感鸿沟”。 |
| [^164] | [EEG-VID: Task-Guided Latent Predictive Pretraining for EEG Decoding and Assistive Target Selection](https://arxiv.org/abs/2609.00566) | EEG-VID提出了一种任务引导式潜变量预测预训练框架，通过指数移动平均编码器预测未来EEG潜状态，在42组跨会话跨被试对比中有41组提升准确率（最高提升16.22个百分点），并可有效应用于场景约束下的辅助目标选择。 |
| [^165] | [Why Multi-Layer Message Passing Works: Completeness Theory for Graph Neural Network Interatomic Potentials](https://arxiv.org/abs/2609.00528) | 本文提出多层完备性理论，证明在通用性、重叠与连通性条件下，稀疏截断图上的 $L$ 层消息传递与访问完整 $L$ 跳邻域具有同等的表示能力，从而首次严格证明了图神经网络原子间势中使用小于物理相互作用范围的逐层截断消息传递这一通用做法的合理性，并由此推出 DPA3 与 CHGNet 架构具有通用近似能力。 |
| [^166] | [QTEA: Ternary LLMs with Sparse Residual Salient Weight and By-Column Optimization](https://arxiv.org/abs/2609.00224) | QTEA提出了一种低于2比特的训练后量化框架，通过将权重量化为三值、利用1:4半结构化稀疏的显著权重残差进行误差补偿，并结合逐列缩放精修与误差衰减机制，在保持GPU硬件效率的同时显著降低了大语言模型低比特量化的精度损失。 |
| [^167] | [Tracing Generated Samples to Training-Data Clusters in Flow-Matching Models](https://arxiv.org/abs/2608.30081) | 该论文提出一种解析与学习相结合的混合方法，在流匹配模型中沿生成轨迹推导簇级别的归因分数，从而将生成的图像追溯到影响它的训练数据簇。 |
| [^168] | [The Illusion of Replacement: Rethinking Specialized Machine Learning Models in the Foundation Model Era](https://arxiv.org/abs/2608.28980) | 本文综述159篇论文后发现，语言模型虽在极端少样本预测等特定场景中可与专用模型竞争，但一旦直接评估结构表示与计算能力，并无证据表明其能全面取代机器学习中的专用架构。 |
| [^169] | [Optimal Transport for Network Comparison: A Review with Machine Learning Applications](https://arxiv.org/abs/2608.27500) | 本文综述了基于最优传输的网络比较方法，系统梳理了Wasserstein、Gromov-Wasserstein和Bures-Wasserstein三种距离，突出传输方案可解释图间差异的节点来源，并利用拉普拉斯谱为Bures-Wasserstein距离推导高效边界，进而在聚类和时间序列网络任务中验证了这些方法。 |
| [^170] | [A Storage-Retrieval Gap in Parametric Knowledge Graph Memory](https://arxiv.org/abs/2608.25489) | 该论文提出将知识图谱离线编译为LoRA适配器作为参数化知识层，在零查询上下文成本下实现事实知识泛化，但发现存储知识无法通过相似性检索恢复，揭示了参数化记忆中的存储-检索差距。 |
| [^171] | [Scalable Self-Supervised Learning for Multiphase AC-OPF in Distribution Systems with Topology Reconfiguration](https://arxiv.org/abs/2608.25095) | 该论文提出了一种无需标签数据的自监督学习框架（惩罚+SLFS），能够高效扩展到带拓扑重构的配电网多相交流最优潮流求解，解决了现有方法在规模和复杂性上的瓶颈。 |
| [^172] | [A Feature-Major Codebook for Memory-Efficient Sparse-Binary Self-Organizing Maps: Scaling a MEDLINE Atlas to 1.05 Million Neurons on a Single Consumer GPU](https://arxiv.org/abs/2608.24067) | 通过将码本按特征主序存储，仅改变布局即可将自组织映射的BMU搜索加速4.5-8.5倍，实现单块消费级GPU上MEDLINE规模（105万神经元）的可扩展映射，且不损失精度。 |
| [^173] | [The Axiomatic Trader: Latent Regularity, Information Budgets, and the Canonical Form of a Quantitative Investment System](https://arxiv.org/abs/2608.23416) | 本文通过五个关键常数（复发界限、不变性缺陷、相干时间、信号上限和机制依赖比例）形式化定义了量化投资系统的公理化基础，从而推导出其近乎必然的架构。 |
| [^174] | [Mol-JEPA: A multimodal Joint Embedding Predictive Architecture for Molecules](https://arxiv.org/abs/2608.22642) | Mol-JEPA通过模态掩蔽和潜在空间预测，有效整合多种生物化学数据，解决了分子基础模型中的无效增强和模态坍缩问题，提升了表示性能。 |
| [^175] | [ToSCA: Leveraging Hierarchical Reinforcement Learning on Temporal and Strategic Abstractions of Conversational Agents](https://arxiv.org/abs/2608.21969) | 本文提出一种两级层次强化学习框架，结合话语级策略抽象与词元级解码，并引入双粒度奖励机制，以提升对话代理在复杂交互中的性能。 |
| [^176] | [Agentic Scaffolding Amplifies Sycophantic Behavior in Large Language Models](https://arxiv.org/abs/2608.21377) | 本文发现代理式交互脚手架（如多轮反馈和迭代细化）会系统性放大LLM的谄媚行为，导致平均准确率下降6.3%，且更强模型放大效应更显著。 |
| [^177] | [FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth](https://arxiv.org/abs/2608.20574) | 该论文提出了一个基于可执行烹饪真实数据的自动化基准测试FlavourBench，通过版本化系统和严格统计方法对27个前沿语言模型进行公平排名，消除了传统基准中的评判者偏差和缺失数据问题。 |
| [^178] | [Diagonal Multi-omics Integration of Heterogenous Datasets](https://arxiv.org/abs/2608.16968) | 本文提出了一种基于极值迹问题和梯度上升方法的新特征，利用最大值与最小值点差的范数来表征数据集异质性，用于异质数据集的对角多组学整合。 |
| [^179] | [TransfHAR: Self-Supervised Wrist Representations for On-Demand Activity Recognition](https://arxiv.org/abs/2608.15861) | TransfHAR通过自监督预训练从粗粒度腕部活动中学习可迁移运动先验，实现无需大量标记数据的按需细粒度活动识别，并支持用户自定义活动集。 |
| [^180] | [Three Necessary Principles for Self-Supervised Visual Representation Learning](https://arxiv.org/abs/2608.08309) | 该论文提出自监督视觉表征学习必须同时满足观察（语义不变性）、预测（块级空间预测）与正则化（表征非退化）三个缺一不可的原则，并从理论上证明：缺少正则化时代码器会坍缩为常量映射，而对比对齐与动量编码器均无法在收敛时保证不发生坍缩。 |
| [^181] | [How Far Do Simple Transformations Translate Across Text Embedding Models?](https://arxiv.org/abs/2608.05980) | 本研究在九个架构、池化策略和训练目标各异的嵌入模型上检验了“潜在通用性”假设，发现线性映射等简单变换仅能在部分兼容的模型对之间成功转换表示，异构嵌入空间的兼容性受架构、训练目标、池化和数据分布共同影响，并非普遍通用。 |
| [^182] | [Scaling an Autoregressive Transformer for Single-Cell Generation](https://arxiv.org/abs/2608.02961) | 本文提出将自回归Transformer与可学习量化VAE分词器相结合用于单细胞基因表达向量的自监督生成，并首次发现了该架构联合拟合的双指数缩放定律与计算最优训练配置。 |
| [^183] | [Training nGPT](https://arxiv.org/abs/2608.01284) | 本文提出的nGPT实用训练方案（含Logit梯度预处理、对数学习率衰减等技术），使300亿参数的混合MoE模型仅需约一半训练token即可达到与AdamW训练的未归一化模型相同的验证损失。 |
| [^184] | [From Digital to Physical Reservoir Computing: Co-Optimizing Soft Robotic Reservoirs via Dynamics Matching](https://arxiv.org/abs/2608.00484) | 提出一种通过动力学匹配将软体机器人物理储备池与高性能数字参考动力学进行预训练和协同优化的框架，利用可微物理模型联合优化物理参数、微分同胚状态映射和前馈-反馈控制，从而缩小物理储备池与数字储备池之间的性能差距。 |
| [^185] | [Nova: An End-to-End MLIR Compiler for Deep Learning](https://arxiv.org/abs/2608.00029) | Nova编译器通过端到端MLIR流水线，将前向和反向传播统一为单一值语义方言，实现整图优化并直接合成细粒度内核，从而原生支持完整Transformer架构并最大化硬件利用。 |
| [^186] | [Feature Interaction Modeling for Neural Operators](https://arxiv.org/abs/2607.28762) | 提出FM-Operator，一种显式建模传感器观测与查询坐标之间特征构建与交互的逐点查询神经算子，通过将DeepONet的分支-主干聚合重新诠释为乘法交互，提升了对激波主导和低粘性偏微分方程的求解能力。 |
| [^187] | [Windowed thinning and query complexity for the bouncy particle and Zigzag samplers](https://arxiv.org/abs/2607.28413) | 该论文提出窗口化稀疏化这一针对弹性粒子采样器和坐标Zigzag过程的精确模拟方法，并结合定量混合估计首次给出了从高斯冷启动达到总变差误差 $\varepsilon$ 所需的梯度查询复杂度保证。 |
| [^188] | [Held-out evidence resolves follow-up measurement decisions in biological screens](https://arxiv.org/abs/2607.27651) | 本研究提出OPAL留出决策测试框架，通过在最终评估前固定档案特定标准来判断机器学习筛选规则能否替代固定测量计划，实现了将优化与足以改变实验的证据相分离。 |
| [^189] | [Aletheia: An Offline-First Clinical Decision Support System for Differential Diagnosis in Low-Resource Healthcare Settings](https://arxiv.org/abs/2607.24814) | Aletheia是一个面向撒哈拉以南非洲低资源医疗环境的离线优先临床决策支持系统，通过QLoRA微调Qwen2.5-3B-Instruct模型，在东非高发疾病鉴别诊断中实现了80%的Top-1准确率和100%的Top-3准确率，无需依赖互联网连接和高规格硬件。 |
| [^190] | [Adaptive Graph-of-Islands Evolution for Automatic Feature Engineering with LLMs](https://arxiv.org/abs/2607.23286) | TOPOFE将自动特征工程构建为图结构的多岛屿进化程序搜索框架，通过LLM引导的变异与交叉探索语义族群、利用提示词自适应记忆积累接受/拒绝反馈来优化候选生成，并动态协调岛屿间的迁移以提升全局搜索效率。 |
| [^191] | [Reinforcement Learning for Heterogeneous Sensor Selection in Maritime Surveillance](https://arxiv.org/abs/2607.22667) | 该论文提出了一种信息增益引导的强化学习传感器选择框架，利用PPO智能体在异构海上传感器网络中为单船跟踪智能选择传感器，从而避免了激活全部传感器或进行高计算代价的在线信息增益评估的需要。 |
| [^192] | [Multi-Mask Diffusion Language Models for Few-Step Generation](https://arxiv.org/abs/2607.19686) | 提出多掩码扩散模型MultiMDM，通过在前向过程中保留掩码结构、在反向过程中先预测指定掩码再精炼为干净词元的起草能力，实现高质量的少步文本生成。 |
| [^193] | [One Model, Many Graphs: Learning over Attributed Graphs across Heterogeneous Modalities with Vision-Language Models](https://arxiv.org/abs/2607.19128) | 提出OMG-VLM统一框架，以预训练视觉语言模型为共享骨干并引入结构感知图适配器，实现了对仅含文本、仅含视觉或两者兼有的异构模态属性图的统一学习，突破了现有方法需针对固定模态模式单独建模的限制。 |
| [^194] | [Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations](https://arxiv.org/abs/2607.17117) | 该论文提出持久稀疏自编码器，通过为每个特征学习一个持久性系数，使稀疏自编码器能够仅凭重构目标从语言模型激活中自动学习特征特定的时间尺度，同时保持高质量的重构效果。 |
| [^195] | [The Anatomy of a Truth Direction: Knowledge-Dependent Dimensionality, a Relational Law, and a Shared Category Geometry in Small Language Models](https://arxiv.org/abs/2607.16741) | 本文提出一种无需训练的SVD主方向方法来解剖语言模型中的真值表征，揭示其具有知识依赖的维度、遵循关系定律并呈现共享的类别几何结构，并可在6个架构家族的14个模型上以O(d)成本高效读取真值。 |
| [^196] | [Gauge dependence and structured-output corruption in sign-branched repetition penalties: measurements across models, inference stacks, and alternative repetition controls](https://arxiv.org/abs/2607.09791) | 该论文揭示了主流推理引擎中的符号分支乘法重复惩罚依赖于 logit 任意零点（规范选择），导致惩罚操作缺乏良好定义且在不同模型上效果各异，并会使 JSON 结构化输出的有效率从 97% 骤降至 23%，同时提出了减法式与归一化等不受规范影响的替代方案。 |
| [^197] | [What You See Is What You Get: Observation-Aligned Supervision for Chart-to-Code Generation](https://arxiv.org/abs/2607.04726) | 论文揭示了图表到代码生成训练中存在的四类潜在变量与观察图像不匹配问题，并提出观察对齐监督方法，用视觉上可约束的量替换潜在变量作为监督目标。 |
| [^198] | [AdaBoosting Text Prompts for Vision-Language Models](https://arxiv.org/abs/2607.00684) | 提出受AdaBoost启发的文本提示提升框架TPB，将每个基于文本提示的分类器作为弱学习器，通过显式聚焦错误分类的难例将其逐步聚合为强集成分类器，从而充分利用少样本监督提升视觉-语言模型的分类性能。 |
| [^199] | [SABER-Math: Automated Benchmark for Information Retrieval Evaluation in Mathematics](https://arxiv.org/abs/2606.29894) | 该论文提出了首个无需专家标注、完全自动化的数学信息检索评估基准SABER-Math，它从28.3万道高中数学题出发自动构建具有挑战性的重排序任务，以克服现有基准无法捕捉细粒度数学相关性的问题。 |
| [^200] | [TaLK: Text-attributed Graph Dataset Distillation via Coupling Language Model with Graph-Aware Kernel](https://arxiv.org/abs/2606.22975) | 提出TaLK方法，通过耦合语言模型与图感知核，实现文本属性图的高效数据集蒸馏，避免重复训练完整模型，同时兼顾文本与结构信息。 |
| [^201] | [Objective-Behavior Alignment: Diagnostics for MORL Policy Selection](https://arxiv.org/abs/2606.21321) | 该论文提出了一种诊断工作流程，能够自动揭示多目标强化学习帕累托前沿上仅凭目标价值向量无法发现的行为差异，帮助决策者更全面地进行策略选择。 |
| [^202] | [MM++: Post-Hoc Scale-Invariant Multilayer OOD Detection via Top-K Gated Feature Fusion](https://arxiv.org/abs/2606.17352) | MM++提出了一种无需辅助数据或架构修改的事后尺度不变多层OOD检测框架，通过熵密度下降选取判别中间层并进行门控特征融合，再利用Ledoit-Wolf正则化捆绑协方差稳定联合特征空间，从而在近远OOD检测中均取得鲁棒性能。 |
| [^203] | [Medical Heuristic Learning: An LLM-Driven Framework for Interpretable and Auditable Clinical Decision Rules](https://arxiv.org/abs/2606.16337) | 提出医学启发式学习（MHL）框架，利用大语言模型通过统计探针、医学知识探针、初始规则合成与迭代规则优化，构建完全以自然语言表达、可解释且可审计的临床决策规则专家系统，并能应对小样本、类别不平衡和特征演变等实际临床约束。 |
| [^204] | [Emotional regulation improves deep learning-based image classification](https://arxiv.org/abs/2606.13081) | 该论文提出了“情绪调节”这一新框架，通过人工主观体验建模情绪，利用基于情感刺激的预训练平衡情绪与非情绪响应，从而提升了ResNet和ViT在图像分类任务上的性能。 |
| [^205] | [A Geometry-Aware Triplane Field Network for Vehicle Aerodynamic Prediction](https://arxiv.org/abs/2606.07724) | 提出几何感知三平面场网络GTF-Net，通过从表面点构建三平面特征并采用AFNO谱混合与CNN细化相结合的双流主干，在统一表示中同时捕捉全局气动耦合与局部几何细节，实现快速的车辆气动压力与壁面剪切应力预测。 |
| [^206] | [WhiFlash: Accelerating Speculative Decoding with Token-Level Cross-Paradigm Routing](https://arxiv.org/abs/2606.07710) | WhiFlash提出首个跨范式投机解码方法，在单一令牌级控制器下统一自回归与扩散并行草稿生成，并通过基于熵或神经策略的细粒度路由自适应应对草稿准确率的动态波动，从而显著加速大语言模型推理。 |
| [^207] | [DOG-DPO:Dynamic Optimization in Geometry for Safety Alignment](https://arxiv.org/abs/2606.07678) | 提出无需训练的数据选择框架DOG-DPO，将偏好对视为模型表示空间中的几何方向，通过分解全局锚定子空间与数据集特有残余子空间并最大化多样性覆盖，为DPO安全对齐筛选出广泛且非冗余的偏好数据子集。 |
| [^208] | [Enabling KV Caching of Shared Prefix for Diffusion Language Models](https://arxiv.org/abs/2606.07571) | 本文提出bicache，首个针对扩散语言模型共享前缀的KV缓存技术，解决了双向注意力下KV动态变化导致传统缓存失效的问题。 |
| [^209] | [Shortcomings and capacities of real-constrained neural networks in complex spaces](https://arxiv.org/abs/2606.04390) | 本文通过文献中非标准的HCIZ积分公式，计算了复假设类中实数预激活约束神经网络与复数神经网络之间存储容量的渐近比率。 |
| [^210] | [Variation Spaces for Encoder--Decoder Neural Operators: Approximation and Generalization](https://arxiv.org/abs/2606.01244) | 该论文基于有界变差向量值测度构建了神经算子的变分空间理论，证明了在ReLU激活下该空间与Schatten-1算子类范数等价，并建立了编码器-解码器神经算子的逼近误差界与高概率泛化界。 |
| [^211] | [OptSkills: Learning Generalizable Optimization Skills from Problem Archetypes via Cluster-Based Distillation](https://arxiv.org/abs/2605.29829) | 该论文提出OptSkills系统，通过依据问题底层原型进行聚类并将成功求解轨迹蒸馏为可复用的工作流级技能，使LLM驱动的优化建模与求解在分布内和分布外问题上都具备更强的泛化能力。 |
| [^212] | [Half-Truth Audio Detection and Localisation: A Lightweight Cross-Attentive Architecture and a Cross-Corpus Diagnostic Study](https://arxiv.org/abs/2605.29531) | 提出了轻量级交叉注意力网络CAFNet，可同时完成真实/全伪造/半真音频的三分类与合成片段的时间边界定位，在约14毫秒CPU延迟下达到97.55%的三分类准确率。 |
| [^213] | [Inference-Native Zeroth-Order Optimization](https://arxiv.org/abs/2605.28760) | 该论文提出“推理原生零阶优化”，将零阶优化重新表述为推理运行时可直接执行的可编程候选状态查询（通过ProbePlan形式化），从而在无需反向传播和全参数状态物化的情况下完成优化。 |
| [^214] | [Connections between the F\"ollmer process and the denoising diffusion probabilistic model](https://arxiv.org/abs/2605.18040) | 本文阐明了离散化Föllmer过程与DDPM采样器之间的直接联系，证明其为DDPM采样器提供了自然的超参数设置，且能容纳比离散化反向SDE更广泛的方差调度，从而系统地恢复了最先进的DDPM采样误差界结果。 |
| [^215] | [When Prompts Interact: Assessing Prompt Arithmetic for Deconfounding under Distribution Shift](https://arxiv.org/abs/2605.03096) | 本文研究了通过任务算术组合软提示能否提升模型对混杂变量引起分布偏移的鲁棒性，并提出了一种混合提示算术方法来去除模型对虚假特征的依赖，相比完全微调更具计算效率。 |
| [^216] | [Stabilizing Private LASSO under Heterogeneous Covariates via Anisotropic Objective Perturbation](https://arxiv.org/abs/2605.01492) | 该论文提出一种基于Gram矩阵的各向异性目标扰动“预失真”策略，通过抵消异质协变量结构引起的失真来稳定差分隐私下的高维LASSO估计，显著提升了收敛稳定性、统计效率和隐私性能。 |
| [^217] | [Language Diffusion Models are Associative Memories Capable of Retrieving Unseen Data](https://arxiv.org/abs/2604.26841) | 该论文证明均匀离散扩散语言模型本质上是联想记忆，其吸引盆可通过条件似然最大化而非显式能量函数形成，并揭示了由数据规模支配的从记忆到泛化的急剧转变，使其能够检索未见过的数据。 |
| [^218] | [RCProb: Probabilistic rule extraction from classification tree ensembles](https://arxiv.org/abs/2604.25304) | RCProb是一种针对RuleCOSI+的概率扩展方法，通过平滑的原子类条件证据和支持自适应混合概率估计，显著提升了从分类树集成中提取规则的概率可靠性，将对数损失大幅降低。 |
| [^219] | [Conditional Diffusion Posterior Alignment for Sparse-View CT Reconstruction](https://arxiv.org/abs/2604.21960) | 提出条件扩散后验对齐（CDPA）方法，通过将条件扩散与显式数据一致性相结合，成功将基于扩散模型的稀疏视角CT重建扩展到大型3D体数据，克服了3D模型内存计算开销大、3D训练数据缺乏以及2D切片方法不一致等局限。 |
| [^220] | [Stream-CQSA: Exact Out-of-Memory Recovery for Attention](https://arxiv.org/abs/2604.20819) | 提出Stream-CQSA框架，基于循环法定人数集（CQS）理论递归地将无法装入显存的注意力调用分解为独立子任务并重组结果，可对任意被包装的注意力内核（无论精确或近似）实现精确的显存溢出恢复。 |
| [^221] | [On the Expressive Power and Limitations of Multi-Layer SSMs](https://arxiv.org/abs/2604.14501) | 多层状态空间模型（SSM）求解函数复合问题时必须满足深度、状态维度与标量精度之间的下界权衡（d²p=Ω(N/L³)），且输入后推理无法绕过该通信下界，从而揭示了多层SSM表达能力的基本限制。 |
| [^222] | [Beyond State Consistency: Behavior Consistency in Text-Based World Models](https://arxiv.org/abs/2604.13824) | 该论文提出了一种新的行为对齐训练范式，通过优化行为一致性奖励（BehR）这一步骤级指标来衡量智能体动作似然在真实状态与预测状态之间的变化，从而提升基于文本的世界模型与真实环境之间的功能一致性。 |
| [^223] | [From High-Dimensional Spaces to Verifiable ODD Coverage for Safety-Critical AI-based Systems](https://arxiv.org/abs/2604.02198) | 本文提出一种融合参数离散化、基于约束的过滤和基于关键性的降维的方法，弥合了抽象ODD定义与可验证证据之间的鸿沟，为安全关键AI系统（如航空领域）满足EASA认证中ODD完全覆盖要求提供了可验证的工程途径。 |
| [^224] | [Train at Moving Edge: Online-Verified Prompt Selection for Efficient RL Training of Large Reasoning Model](https://arxiv.org/abs/2603.25184) | 提出HIVE双阶段框架，通过历史奖励轨迹与在线验证在采样前选取处于“学习边缘”（中等难度且高不确定性）的高效用提示，显著提升大推理模型强化学习训练的数据效率。 |
| [^225] | [SpecXMaster Technical Report](https://arxiv.org/abs/2603.23101) | 该论文提出SpecXMaster，一个基于智能体强化学习的端到端智能框架，可直接从原始FID数据自动提取NMR光谱多重性信息并解读为化学结构，在多个公开NMR解读基准上表现卓越。 |
| [^226] | [MISApp: Multi-Hop Intent-Aware Session Graph Learning for Next App Prediction](https://arxiv.org/abs/2603.21653) | 提出了一种无需用户画像的下一个应用预测框架MISApp，通过构建多跳会话图捕捉不同结构范围的转移依赖，并结合时间与空间上下文，有效应对冷启动等场景下的预测挑战。 |
| [^227] | [ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs](https://arxiv.org/abs/2603.18579) | 提出ICE框架，通过在多种干预算子下将模型解释与同等规模的随机基线进行统计对比，首次揭示了大语言模型的解释忠实性是依赖干预方法的量而非固定属性（切换算子导致差距高达44个百分点），并能检测出比随机表现更差的反忠实性现象。 |
| [^228] | [Probing Cultural Signals in Large Language Models through Author Profiling](https://arxiv.org/abs/2603.16749) | 本研究通过零样本歌词作者画像任务揭示了大语言模型中的系统性文化偏见——多数模型默认偏向北美族裔而DeepSeek-1.5B更对齐亚洲族裔，并创新性地提出MAD和RD两个公平性指标来量化这些差异。 |
| [^229] | [MDM-Prime-v2: Binary Encoding and Index Shuffling Enable Scaling of Diffusion Language Models](https://arxiv.org/abs/2603.16077) | 本文提出MDM-Prime-v2，通过分析子分词器的最优设计并引入二进制编码与索引洗牌技术，克服了MDM-Prime框架中交叉熵损失增加和词元粒度超参数选择缺乏指导的两大局限，实现了掩码扩散语言模型的规模化。 |
| [^230] | [GONE: Structural Knowledge Unlearning via Neighborhood-Expanded Distribution Shaping](https://arxiv.org/abs/2603.12275) | 本文提出了GONE基准用于评估大型语言模型对结构化知识图谱事实的遗忘效果，能够解耦直接事实移除、推理泄漏和灾难性遗忘三种效应，并设计了邻域扩展分布塑造（NEDS）这一新型遗忘框架。 |
| [^231] | [DynaTokens: Controlling Token Dynamics for Continual Video-Language Understanding](https://arxiv.org/abs/2603.06662) | 提出DynaTokens，一种基于Transformer的按需动态生成微调token的生成器，结合元学习启发的正则化和无梯度路由机制，有效缓解多模态大语言模型在持续视频问答中的任务干扰与遗忘问题。 |
| [^232] | [Quantum Maximum Likelihood Prediction via Hilbert Space Embeddings](https://arxiv.org/abs/2602.18364) | 本文通过将经验概率分布嵌入量子态并最小化量子相对熵，提出了一种量子最大似然预测方法，并为其在经典和量子大语言模型中的统一应用提供了非渐近性能保证。 |
| [^233] | [Constrained Group Relative Policy Optimization](https://arxiv.org/abs/2602.05863) | 本文提出约束GRPO（Constrained GRPO），一种基于拉格朗日方法的GRPO扩展用于约束策略优化，并揭示了在归一化前对标量化奖励会导致共享分母耦合，使改变一个约束乘子会同时影响奖励与其他约束的相对权重这一关键失败模式。 |
| [^234] | [Shiva-DiT: Residual-Based Differentiable Top-$k$ Selection for Efficient Diffusion Transformers](https://arxiv.org/abs/2602.05605) | Shiva-DiT提出了一种基于残差的可微分Top-k token选择方法，借助残差感知直通估计器同时学习token分数与预算k，并结合上下文感知路由器和自适应比率策略，在保持生成质量的同时显著降低扩散Transformer的FLOPs与延迟（在SD3-Medium上实现1.54倍加速）。 |
| [^235] | [Modular Expert Merging for Biomedical Retrieval](https://arxiv.org/abs/2602.04731) | 本文提出模块化专家合并方法，通过合成难负样本和LoRA微调领域专家并合并，在生物医学检索上优于大规模混合训练，兼顾通用性能。 |
| [^236] | [Cantelli Constrained Policy Optimization](https://arxiv.org/abs/2601.22993) | 本文提出风险厌恶方法Canary，利用Cantelli不等式基于成本回报的前两阶矩得到可处理的风险价值约束上界，并扩展CPO信赖域框架提供最坏情况保证，是所有测试环境中唯一能可靠满足风险价值约束的方法。 |
| [^237] | [Towards Solving the Gilbert-Pollak Conjecture via Large Language Models](https://arxiv.org/abs/2601.22365) | 该论文提出一种新型AI系统，通过让大语言模型生成以可执行代码实现的规则约束几何引理，为停滞三十年的Gilbert-Pollak猜想（Steiner比率猜想）获得更紧的下界，突破了0.824的已有纪录。 |
| [^238] | [Learning and extrapolating scale-invariant processes](https://arxiv.org/abs/2601.14810) | 本文研究了通过在模型架构中融入尺度不变的对称性（几何深度学习思路），机器学习能够学习并外推具有幂律行为的无标度过程（如分数高斯场和阿贝尔沙堆模型），从而预测训练集中未出现过的罕见大事件。 |
| [^239] | [Beyond Transfer Accuracy: Mechanism-Guided Controlled Adaptation for Low-Resource Languages](https://arxiv.org/abs/2601.08146) | 该论文提出了一种无需反事实的回路发现方法，并据此提出回路定向监督微调（CT-SFT），仅更新任务相关的注意力头和LayerNorm，从而在低资源语言适配中既保持竞争力又最有效地避免灾难性遗忘。 |
| [^240] | [What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?](https://arxiv.org/abs/2512.24497) | 本文将联合嵌入预测世界模型（JEPA-WM）类规划方法进行了系统化表征，通过对若干关键组件的全面研究，找出了在抽象表示空间中进行物理规划取得成功的关键技术选择。 |
| [^241] | [On Cost-Aware Designs for Sequential Hypothesis Testing](https://arxiv.org/abs/2512.19067) | 本文提出了成本感知序贯假设检验框架，证明了最优期望总成本按 $\Theta(\log(1/\delta))$ 缩放，并揭示了“最大化期望信息增益与期望成本之比”这一设计原则，据此改编的经典策略具有渐近最优性。 |
| [^242] | [Secure AI-Driven Super-Resolution for Real-Time Mixed Reality Applications](https://arxiv.org/abs/2512.15823) | 提出了一种在服务器端对点云进行下采样与部分加密、在客户端利用AI超分辨率模型重建原分辨率内容的系统，可近乎线性地降低实时混合现实应用的带宽消耗与加解密延迟，同时重建误差极小。 |
| [^243] | [A Multivariate Bernoulli-Based Sampling Method for Multi-Label Data with Application to Meta-Research](https://arxiv.org/abs/2512.08371) | 提出了一种基于多元伯努利分布、考虑标签间依赖性的加权抽样算法，解决了多标签数据中稀有标签难以获得足够样本的问题，并成功应用于元研究领域。 |
| [^244] | [Freeze, Diffuse, Decode: Geometry-Aware Adaptation of Pretrained Transformer Embeddings for Antimicrobial Peptide Design](https://arxiv.org/abs/2511.23120) | 提出了FDD（冻结、扩散、解码）框架，通过沿冻结嵌入的内在流形传播监督信号，在保留预训练Transformer嵌入几何结构的前提下实现几何感知的任务适配，并在抗菌肽设计中生成低维、可预测、可解释的表示，支持性质预测、检索与潜空间插值。 |
| [^245] | [Enhancing Road Safety Through Multi-Camera Image Segmentation with Post-Encroachment Time Analysis](https://arxiv.org/abs/2511.12018) | 本文提出一种基于多摄像头图像分割与侵占后时间（PET）计算的实时交通安全评估框架，利用边缘设备上的YOLOv11分割和单应性鸟瞰图变换，实现交叉口碰撞风险的细粒度动态可视化。 |
| [^246] | [SEBA: Sample-Efficient Black-Box Attacks on Visual Reinforcement Learning](https://arxiv.org/abs/2511.09681) | SEBA提出了一种针对视觉强化学习的样本高效黑盒攻击框架，通过结合影子Q模型、生成对抗网络和世界模型，以极少的真实环境查询实现对基于图像的连续控制智能体的有效对抗攻击。 |
| [^247] | [GMTRouter: Personalized LLM Router over Multi-turn User Interactions](https://arxiv.org/abs/2511.08590) | 提出GMTRouter，将多轮用户-LLM交互建模为包含用户、LLM、查询、响应和轮次五种节点类型的异构图，以最大程度保留交互的关系结构，从而在用户偏好数据稀缺且格式不一致的情况下实现个性化的LLM路由。 |
| [^248] | [Gradient Prediction with Control Variates in the Cheap-Forward Regime](https://arxiv.org/abs/2511.05187) | 该论文提出用降精度、推理风格的程序预测梯度，并通过控制变量将大量预测与少量精确梯度结合，使近似误差转化为方差而非偏差，从而在集群推理资源足够廉价时降低语言模型训练的成本。 |
| [^249] | [Exchange Policy Optimization Algorithm for Semi-Infinite Safe Reinforcement Learning](https://arxiv.org/abs/2511.04147) | 提出交换策略优化（EPO）算法，首次为含无限多约束的半无限安全强化学习提供在可证明有界安全保证下实现最优策略性能的算法框架。 |
| [^250] | [Neural Variational Cut Posteriors without Upstream Data](https://arxiv.org/abs/2510.10268) | 提出NeVI-Cut方法，一种无需访问上游数据和模型、仅利用上游后验样本即可模块化且可证明准确地近似切割后验的神经变分推断方法。 |
| [^251] | [Toward Uncertainty-Aware and Generalizable Neural Decoding for Quantum LDPC Codes](https://arxiv.org/abs/2510.06257) | 提出了具备校准不确定性估计的量子贝叶斯图注意力译码器QuBA，并通过SAGU多阶段训练框架实现了对训练中未见过的量子LDPC码的鲁棒泛化译码。 |
| [^252] | [General Demographic Pre-trained Models for Enhancing Predictive Performance Across Diseases and Population](https://arxiv.org/abs/2509.07330) | 提出了一种基于年龄和性别的通用人口统计学预训练模型（GDP），能以即插即用的方式提升跨疾病和跨人群的预测性能。 |
| [^253] | [Adversarial Stress Testing of Outlier Detection in Subjective Image Quality Assessment](https://arxiv.org/abs/2509.06554) | 该论文提出了一个针对主观图像质量评估中离群值检测方法的对抗性压力测试框架，利用优化算法构造最坏情况下的恶意评分，以揭示现有检测方法在极端攻击下的失效行为。 |
| [^254] | [Explainable Information Processing in Particle Swarm Optimization through Landscape and Search Behavior Analysis](https://arxiv.org/abs/2509.06272) | 本文提出了一个针对粒子群优化的多层面可解释性框架，通过探索性地形分析（ELA）量化问题特性，并利用机器学习为未知问题预测最优的拓扑特定超参数配置，从而显著提升算法的透明度与可解释性。 |
| [^255] | [Simulating Classification Models for Ex-Ante Evaluation of Predict-Then-Optimize Methods](https://arxiv.org/abs/2509.02191) | 本文提出在指定性能水平上模拟多分类预测的方法，将基于仿真的事前评估从二分类推广到具有类别型不确定参数的优化问题，构建预测误差到决策遗憾的映射，并通过基于遗憾成因的一阶近似降低计算成本。 |
| [^256] | [No Data Wasted: A Semi-supervised Generative Model for Incomplete Multi-view Data Integration with Missing Labels](https://arxiv.org/abs/2508.11180) | 本文提出了一种半监督生成模型，在统一框架中同时利用有标签和无标签的多视图数据，通过最大化无标签样本的似然并与信息瓶颈原理结合，同时解决了多视图学习中视图缺失和标签缺失的双重问题。 |
| [^257] | [Learning Encodings by Maximizing State Distinguishability: Variational Quantum Error Correction](https://arxiv.org/abs/2506.11552) | 提出变分量子纠错方法，以最大化噪声信道后量子态的可区分性作为机器学习损失函数，自动发现针对特定噪声结构优化的资源高效编码电路，并在多种场景下超越标准码。 |
| [^258] | [DLM-One: Diffusion Language Models for One-Step Sequence Generation](https://arxiv.org/abs/2506.00290) | DLM-One提出了一种基于分数蒸馏的框架，将扩散语言模型的生成过程压缩为单步，实现采样步数约2000倍、推理时间约500倍的加速，同时保持有竞争力的文本生成性能。 |
| [^259] | [Quantum Speedups for Sampling and Non-convex Optimization with Stochastic Oracles](https://arxiv.org/abs/2504.03626) | 该论文提出了一个量子加速框架，通过用方差可控的量子均值估计和梯度估计子程序替代随机梯度估计器，加速了经典的随机朗之万蒙特卡洛和哈密顿蒙特卡洛算法，且无需可逆性或精确梯度，从而实现了从非凸分布采样和非凸优化的量子加速。 |
| [^260] | [Sample Complexity of Linear Quadratic Regulator Without Initial Stability](https://arxiv.org/abs/2502.14210) | 该论文提出了一种受REINFORCE启发的滚动时域算法，无需初始稳定策略即可解决未知动态的LQR问题，并通过黎曼距离下黎卡提算子收缩性的精细误差传播分析，实现了更优的样本复杂度和收敛保证。 |
| [^261] | [Nonasymptotic CLT and Error Bounds for Linear Two-Time-Scale Stochastic Approximation](https://arxiv.org/abs/2502.09884) | 本文首次建立了带 Polyak-Ruppert 平均的线性双时间尺度随机逼近的非渐近 Wasserstein-1 中心极限定理，并由此证明其期望误差以最优的 1/√K 速率衰减，填补了有限时间分析与渐近理论之间的空白。 |
| [^262] | [Double-Bounded Nonlinear Optimal Transport for Size Constrained Min Cut Clusterin](https://arxiv.org/abs/2501.18143) | 本文首次将最小割问题转化为双边界约束非线性最优传输问题，并基于Frank-Wolfe方法提出DNF算法，证明了O(1/t)的收敛速率，在尺寸约束最小割聚类任务上取得了有竞争力的性能。 |
| [^263] | [Enhancing brain age estimation with structural MRI and synthesized cerebral blood volume maps](https://arxiv.org/abs/2412.01865) | 提出了一种融合结构MRI与合成脑血容量（DeepCBV）图谱的多模态脑年龄估计框架，将平均绝对误差降至3.95年，并有效捕捉了早期神经退行性相关的血管变化。 |
| [^264] | [Monotonic anomaly detection](https://arxiv.org/abs/2410.23158) | 针对只需检测高（或低）属性值异常的场景，本文提出了融合斜坡函数的非对称距离度量和改进的孤立森林路径长度算法，实验证明二者能显著提升单调属性数据集上的异常检测效果。 |
| [^265] | [Action abstractions for amortized sampling](https://arxiv.org/abs/2410.15184) | 提出了一种在策略优化过程中自动发现动作抽象的方法，通过将高奖励轨迹中常用的动作子序列“分块”为单个高级动作加入动作空间，从而缓解长轨迹下信用分配困难、探索受限及模式发现受阻的问题。 |
| [^266] | [Deep Reinforcement Learning for Reach-Avoid-Stay Problems](https://arxiv.org/abs/2410.02898) | 本文提出一种两步深度强化学习框架，联合学习最大鲁棒到达-避开-停留集及其控制策略，能够处理一般动态系统并保证在所有有界扰动下安全到达并停留在目标集内。 |
| [^267] | [Achieving More with Less: A Tensor-Optimization-Powered Ensemble Method](https://arxiv.org/abs/2408.02936) | 该论文提出了一种基于张量优化的集成方法，通过引入置信度张量来刻画各基分类器对不同类别的预测置信程度，从而仅用少量基学习器即可达到通常需要大量基学习器才能实现的分类性能与泛化能力。 |
| [^268] | [Doubly Stochastic Adaptive Neighbors Clustering via the Marcus Mapping](https://arxiv.org/abs/2408.02932) | 该论文提出Marcus映射，将Marcus定理扩展到某些稀疏矩阵，证明其也可通过对角矩阵变换为双随机对称矩阵，并据此提出了引入秩约束的双随机自适应邻居聚类算法ANCMM。 |
| [^269] | [Prompting the Unknown: Understanding Response Uncertainty in Large Language Models](https://arxiv.org/abs/2407.14845) | 该论文提出了一个提示-响应概念模型，识别出大语言模型响应不确定性的四个来源（提示规范不足、模型质量、任务变异性和语义冗余），并证明了提高提示信息性或模型质量可以降低响应不确定性。 |
| [^270] | [Smoothed Analysis for Learning Concepts with Low Intrinsic Dimension](https://arxiv.org/abs/2407.00966) | 本文提出了一种平滑分析框架，通过只需与对小随机高斯扰动鲁棒的最优分类器竞争，实现了对依赖低维子空间且具有有界高斯表面积的概念（如半空间函数和低维凸集函数）在任意分布下的高效学习。 |
| [^271] | [Gradient Descent on Logistic Regression with Non-Separable Data and Large Step Sizes](https://arxiv.org/abs/2406.05033) | 该论文研究了非可分数据上逻辑回归的大步长梯度下降动力学，揭示了从临界步长 $2/\lambda$ 开始的倍周期分岔现象，证明了一维中小于 $1/\lambda$ 的步长足以保证全局收敛，而对于 $1/\lambda$ 到 $2/\lambda$ 之间的步长则可构造出使GD收敛到稳定周期循环的数据集。 |
| [^272] | [GPTBIAS: A Comprehensive Framework for Evaluating Bias in Large Language Models](https://arxiv.org/abs/2312.06315) | 本文提出了GPTBIAS框架，利用GPT-4等高性能大语言模型来评估其他模型的社会偏见，并设计了专门用于偏见评估的“偏见攻击指令”提示词，从而提升了偏见评估的可信度和可解释性。 |
| [^273] | [Deep denoising autoencoder-based non-invasive blood flow detection for arteriovenous fistula](https://arxiv.org/abs/2306.06865) | 该论文提出一种基于深度降噪自编码器（DAE）的表征学习方法，对一层离散小波变换获得的波形进行降维和重建，实现了动静脉瘘功能障碍的无创检测，准确率达0.93。 |
| [^274] | [Robust Streaming PCA](https://arxiv.org/abs/1902.03223) | 该论文提出了协方差矩阵属于时变不确定集合的鲁棒流式主成分分析框架，给出了算法收敛的基本极限，并证明噪声幂法在此扰动设定下达到速率最优。 |
| [^275] | [Clustering Three-Way Data with Outliers.](http://arxiv.org/abs/2310.05288) | 这项研究提出了一种用于聚类矩阵形式数据的方法，可以处理其中的异常值。 |
| [^276] | [Generalized Regret Analysis of Thompson Sampling using Fractional Posteriors.](http://arxiv.org/abs/2309.06349) | 这项研究对使用分数后验概率的汤普森抽样算法进行了广义遗憾分析，获得了依赖于实例和实例独立的频率遗憾界。这对多臂赌博问题的解决有重要意义。 |

# 详细

[^1]: 语音脑机接口的通信通用度量标准

    A Common Measure of Communication for Speech Brain-Computer Interfaces

    [https://arxiv.org/abs/2609.02887](https://arxiv.org/abs/2609.02887)

    提出开词汇互信息（OVMI）这一信息论度量标准，解决了语音脑机接口领域因数据集、记录方法和词汇各异而无法相互比较的评估难题，为不同系统的通信能力提供了统一的衡量指标。

    

    语音脑机接口将神经活动转化为语言，为瘫痪患者恢复说话能力提供了一条途径，更广泛地说，还能实现新形式的自然人机交互。尽管前景广阔，该领域目前仍缺乏衡量进展的通用标准，因为不同系统使用不同的数据集、记录方法、语音类型和词汇量，导致它们报告的分数很少具有可比性。这一度量问题的背后是两个尚未解决的关键问题：（i）语音脑机接口应该使用户能够传达什么样的词语分布；（ii）系统能够从该分布中传达多少信息。我们通过推导开词汇互信息（OVMI）来同时解决这两个问题。OVMI是一个信息论量度，用于衡量解码器相对于用户可能想要传达的词语参考分布所传递的信息量。这使得在不同条件下测量得到的能力能够（摘要在此处被截断）……

    arXiv:2609.02887v1 Announce Type: new  Abstract: Speech brain-computer interfaces (speech BCIs) translate neural activity into language, offering a path towards restoring speech for people with paralysis and, more broadly, enabling new forms of natural human-computer interaction. Despite this promise, the field lacks a common measure of progress because systems use different datasets, recording methods, types of speech, and vocabularies, so their reported scores are rarely comparable. Underlying this measurement problem are two unresolved questions: (i) what distribution of words should a speech BCI enable a user to communicate, and (ii) how much information from this distribution can a system convey. We address both by deriving open-vocabulary mutual information (OVMI), an information-theoretic quantity that measures the information conveyed by a decoder relative to a reference distribution over the words a user may wish to communicate. This allows capabilities measured under differen
    
[^2]: 面向网络智能体的判别式世界模型

    Discriminative World Models for Web Agents

    [https://arxiv.org/abs/2609.02885](https://arxiv.org/abs/2609.02885)

    该论文提出“预测状态匹配”这一新训练目标，使世界模型生成的网络状态表示具有跨候选动作的区分性，从而与下游排序器对齐，提升网络智能体测试时动作选择的准确性。

    

    最近的网络智能体使用世界模型进行测试时动作选择，其方式是采样候选动作、预测由此产生的网络状态，然后使用排序模型或过程奖励模型（PRM）对这些状态进行排序。这些世界模型通常通过监督式的下一状态预测进行训练，以生成如HTML或AXTree快照等固定表示形式。然而，这一训练目标与下游排序器并不一致，因为排序器依赖于预测出的状态在不同候选动作之间具有区分性，才能对其进行准确评分。为解决这一问题，我们提出了“预测状态匹配”，这是一种训练目标，要求预测的表示能够将真实的结果状态与通过替代动作到达的状态区分开来。我们使用从WebArena Go-Browse轨迹派生的分支式网络智能体数据集来训练这些模型，其中每个决策点都包含多个替代动作及其对应的结果状态。在我们保留的预测状态匹配（基准）上的实验……（原文摘要在此处截断）

    arXiv:2609.02885v1 Announce Type: new  Abstract: Recent web agents use world models for test-time action selection by sampling candidate actions, predicting the resulting web states, and ranking them with a ranker model or a Process Reward Model (PRM). These world models are typically trained via supervised next-state prediction to generate fixed representations like HTML or AXTree snapshots. However, this objective is misaligned with the downstream ranker, which relies on predicted states being discriminative across candidates to accurately score them. To address this, we introduce predicted-state matching, a training objective where the predicted representation must distinguish the true resulting state from those reached by alternative actions. We train these models using a branching web-agent dataset derived from WebArena Go-Browse trajectories, where every decision point contains multiple alternative actions and their resulting states. Experiments on our held-out predicted-state ma
    
[^3]: 图机器：通过边实现更优预训练

    Graph Machine: Towards Better Pretraining via Edges

    [https://arxiv.org/abs/2609.02881](https://arxiv.org/abs/2609.02881)

    提出图机器（GM）架构，利用类似指针的“边”实现稀疏动态路由来访问 O(n) 规模状态，将 Qwen3-0.6B 中 75% 的稠密 Transformer 层替换为 GM 稀疏层并从头预训练，在每层仅检索极少 token 的情况下模型损失几乎不受影响甚至略有改善。

    

    我们提出了图机器（Graph Machine，GM），这是一种维持 $O(n)$ 规模状态、并通过稀疏动态路由来访问该状态的架构。与采用固定大小状态、或稀疏但静态路由的方法不同，GM 在其稀疏层中保持 $O(n)$ 的复杂度，同时不将潜在可访问的状态规模限制为 $O(1)$。相反，GM 使用“边”——一种类似指针的对象，通过一种类似指针追踪（pointer chasing）的引荐机制进行可微分更新。我们将 Qwen3-0.6B 中 75% 的稠密 Transformer 层替换为 GM 稀疏层，并在 157 亿个 token 上从头开始预训练。在每个稀疏层中，当每个 KV 头仅从 4,096 个 token 中检索 2 个时，损失仅出现轻微上升；当检索 4 个时，最优模型的损失还略有改善。

    arXiv:2609.02881v1 Announce Type: new  Abstract: We introduce the Graph Machine (GM), an architecture that maintains an $O(n)$-sized state and accesses it through sparse, dynamic routing. Unlike methods with fixed-size states or sparse but static routing, GM preserves $O(n)$ complexity in its sparse layers without restricting the potentially accessible state size to $O(1)$. Instead, GM uses edges - pointer-like objects updated differentiably by a referral mechanism resembling pointer chasing. We replace 75% of the dense Transformer layers in Qwen3-0.6B with GM sparse layers and pretrain from scratch on 15.7B tokens. With only 2 of 4,096 tokens retrieved per KV head in each sparse layer, loss degrades only slightly; with 4, the best model marginally improves loss.
    
[^4]: GRADSOLVE：GPU上ODE集合的快速精确梯度

    GRADSOLVE: fast exact gradients for ODE ensembles on GPUs

    [https://arxiv.org/abs/2609.02876](https://arxiv.org/abs/2609.02876)

    GRADSOLVE是一个开源JAX库，通过记录自适应求解器接受的步长并以固定步长重放进行微分，在GPU上以融合内核求解的速度实现对ODE集合的精确反向模式梯度计算。

    

    常微分方程（ODE）是科学与工程中各类模型的基础，许多应用需要求解方程解关于参数的导数。独立轨迹集合（ensemble）非常适合图形处理器（GPU）计算，但当前的GPU软件迫使用户做出权衡：求解速度最快的集成求解器无法以同样的速度进行反向模式微分，而为微分设计的求解器求解速度又较慢。目前尚无单一工具能够以融合内核求解的速度提供反向模式梯度。我们提出了GRADSOLVE，这是一个开源的JAX库，用于在NVIDIA GPU上求解低维ODE集合并进行反向模式微分。它记录自适应求解器所接受的步长，并对这些步长进行固定步长的重放以实现微分；返回的梯度是这些步骤的精确离散伴随（discrete adjoint），与Diffrax默认返回的导数完全一致，但通过固定长度的计算链以更低的开销获得。

    arXiv:2609.02876v1 Announce Type: cross  Abstract: Ordinary differential equations (ODEs) underlie models in science and engineering, and many applications need derivatives of their solutions with respect to parameters. Ensembles of independent trajectories suit graphics processing units (GPUs), but current GPU software forces a trade-off: the fastest ensemble solvers cannot be differentiated in reverse mode at the speed they solve, and the solvers built for differentiation solve more slowly. No single tool has yet offered a reverse-mode gradient at the speed of a fused-kernel solve.   We present GRADSOLVE, an open-source JAX library for solving and reverse-mode differentiating low-dimensional ODE ensembles on NVIDIA GPUs. It records the steps an adaptive solver accepts and differentiates a fixed-step replay of them; the returned gradient is the exact discrete adjoint of those steps, the same derivative Diffrax returns by default, obtained more cheaply from a fixed-length chain than fr
    
[^5]: 超越Nesterov的改进梯度下降下界

    Improved Gradient Descent Lower Bounds Beyond Nesterov

    [https://arxiv.org/abs/2609.02855](https://arxiv.org/abs/2609.02855)

    本文证明了光滑凸优化中固定步长梯度下降的两个更强下界——非anytime的Ω(n^{-1.6342})与anytime的Ω(n^{-1.2408})，并借助silver调度可达的O(n^{-log_2(1+√2)})速率，严格分离了两种设定下可实现的收敛指数。

    

    我们研究了在光滑凸优化中，梯度下降（GD）通过预先设定的步长能够被加速到何种程度。在超越Nemirovsky和Yudin经典的Ω(n^{-2})一阶oracle下界的基础上，我们证明了Ω(n^{-1.6342})的非anytime下界以及Ω(n^{-1.2408})的anytime下界。这两个结果分别改进了Ma和Chen近期提出的Ω(n^{-1.932})非anytime下界，以及Tsai等人提出的Ω(n^{-4/3}) anytime下界。结合silver步长调度所达到的非anytime O(n^{-log_2(1+√2)})收敛速率，我们的anytime下界在这两种设定下可实现的收敛指数之间建立了严格的分离。

    arXiv:2609.02855v1 Announce Type: cross  Abstract: We study how far gradient descent (GD) can be accelerated by predetermined stepsizes in smooth convex optimization. Going beyond the classical $\Omega(n^{-2})$ first-order oracle lower bound of Nemirovsky and Yudin, we prove an $\Omega(n^{-1.6342})$ non-anytime lower bound and an $\Omega(n^{-1.2408})$ anytime lower bound. These improve the recent $\Omega(n^{-1.932})$ non-anytime lower bound of Ma and Chen and the $\Omega(n^{-4/3})$ anytime lower bound of Tsai et al., respectively. Together with the non-anytime $O(n^{-\log_2(1+\sqrt{2})})$ rate achieved by silver schedules, our anytime lower bound establishes a strict separation between the achievable convergence exponents in the two settings.
    
[^6]: 语言不可读性对大语言模型安全的影响

    The Implications of Linguistic Illegibility for LLM Security

    [https://arxiv.org/abs/2609.02852](https://arxiv.org/abs/2609.02852)

    本文提出“语言不可读性”概念，指出大语言模型的外部语言输出无法可靠反映其基于激活空间数学运算的内部计算，从而对依赖模型语言自我报告的安全机制构成根本性挑战。

    

    大语言模型（LLM）被训练用于生成自然语言。然而，多方面的证据表明，LLM外化的语言输出和通过机制可解释性方法提取的语言特征，可能并不是理解模型内部计算的可靠透镜。我们提出“语言不可读性”这一术语，广义上指LLM外化的或通过机制性探测获得的语言产物无法代表模型实际思考方式的各种情形。我们认为，对于内部计算并非通过语言直接表达、而是通过对激活空间进行数学运算来实现的大语言模型而言（激活空间与自然语言之间仅在两端发生有损转换），语言不可读性的阴影是不可避免的。如果语言不可读性始终可能存在，那么依赖模型语言自我报告的安全机制（例如思维链监控、宪法式自我批评、激活探测……）

    arXiv:2609.02852v1 Announce Type: new  Abstract: LLMs are trained to generate natural language. However, various strands of evidence indicate that an LLM's externalized linguistic outputs and mechanistically-extracted linguistic features can be an unreliable lens for understanding internal model computation. We introduce the term ``linguistic illegibility'' to broadly refer to scenarios in which an LLM's externalized or mechanistically-probed language artifacts fail to represent how the model actually thinks. We argue that the specter of linguistic illegibility is unavoidable for LLMs whose internal computations are not directly expressed via language, but rather math over activation spaces (with lossy translations between activation spaces and natural language happening at the bookends). If linguistic illegibility is always possible, then security mechanisms that rely on a model's linguistic self-reporting (e.g., chain-of-thought monitoring, constitutional self-critique, activation pr
    
[^7]: 面向编程竞赛金牌表现的语言模型后训练

    Post-Training Language Models for Gold-Medal Performance in Coding Competitions

    [https://arxiv.org/abs/2609.02849](https://arxiv.org/abs/2609.02849)

    该研究通过结合大规模题目筛选、监督微调、强化学习以及反馈驱动的测试时计算策略 GenCorrect，使语言模型在 IOI 2025 编程竞赛中取得了超越金牌分数线（438.3 分）的成绩（Nano-CC 达 468 分，Ultra-CC 达 502 分）。

    

    竞赛编程已成为检验大语言模型推理能力的关键测试，其中 IOI 和 ICPC 等国际赛事代表了最具挑战性的场景。我们提出了一条端到端的专门化流水线，结合了大规模题目筛选、合成推理轨迹、监督微调（SFT）和强化学习（RL）。利用 22,000 道精选题目，我们通过 SFT 和 RL 训练了 Nemotron-3-Nano-CC（30B-A3B），并仅通过 SFT 训练了 Nemotron-3-Ultra-CC（550B-A55B）。我们进一步提出了 GenCorrect，这是一种由反馈驱动的测试时计算策略，可迭代地生成、评估并改进多样化的解决方案。在 IOI 2025 上，Nano-CC 在后训练后从 130 分提升至 291 分，结合 GenCorrect 后达到 468 分，超过了 438.3 的金牌分数线，而 Ultra-CC 达到了 502 分。在这些结果的指导下，我们开发了一个面向竞赛的 Ultra-CC 系统，并在 IOI 2026 期间进行了前瞻性评估。

    arXiv:2609.02849v1 Announce Type: cross  Abstract: Competitive programming has become a key test of large language model reasoning, with international competitions such as IOI and ICPC representing its most challenging settings. We present an end-to-end specialization pipeline combining large-scale problem curation, synthetic reasoning traces, supervised fine-tuning (SFT), and reinforcement learning (RL). Using 22,000 curated problems, we train Nemotron-3-Nano-CC (30B-A3B) with SFT and RL and Nemotron-3-Ultra-CC (550B-A55B) with SFT alone. We further introduce GenCorrect, a feedback-driven test-time compute strategy that iteratively generates, evaluates, and refines diverse solutions. On IOI 2025, Nano-CC improves from 130 points to 291 after post-training and to 468 with GenCorrect, exceeding the gold threshold of 438.3 while Ultra-CC reaches 502. Guided by these results, we develop a competition-specific Ultra-CC system and evaluate it prospectively during IOI 2026. Under the same ti
    
[^8]: 用于稳定语言模型预训练的 UE5M3 FP4 块缩放方法

    UE5M3 FP4 Block Scaling for Stable Language Model Pretraining

    [https://arxiv.org/abs/2609.02846](https://arxiv.org/abs/2609.02846)

    该论文提出将E2M1有效载荷与无符号E5M3块缩放配对、配合选择性随机舍入并省略随机Hadamard变换的FP4训练方案，在Nemotron-H 8B模型近1900亿token的预训练中取得了优于NVIDIA Transformer Engine NV配方的训练损失与验证损失。

    

    arXiv:2609.02846v1 公告类型：新论文 摘要：稳定的4位浮点（FP4）预训练十分困难，因为E2M1有效载荷仅能表示较窄的数值范围。NVIDIA的Transformer Engine NV配方通过当前张量缩放、随机Hadamard变换（RHT）以及bfloat16（BF16）最终层来解决这一问题，但这在FP4矩阵乘法之外引入了额外的工作。我们则将E2M1有效载荷与无符号E5M3（UE5M3）块缩放相结合。E5M3更宽的数值范围允许使用周期性张量缩放，同时我们的配方对反向传播梯度应用选择性随机舍入，省略了RHT，并在所有符合条件的内部线性层中使用FP4。我们对Nemotron-H 8B模型进行了近1900亿个token的预训练。与Transformer Engine NV相比，所提出的block-16配方实现了更低的最终窗口训练损失，并且在各自的量化推理策略下，以留出数据负对数似然衡量的验证损失也更低。其量化推理的下游点估计结果……

    arXiv:2609.02846v1 Announce Type: new  Abstract: Stable 4-bit floating-point (FP4) pretraining is difficult because the E2M1 payload represents only a narrow range of magnitudes. NVIDIA's Transformer Engine \nv{} recipe addresses this with current-tensor scaling, a randomized Hadamard transform (RHT), and bfloat16 (BF16) final layers, adding work outside the FP4 matrix multiplications. We instead pair E2M1 payloads with unsigned E5M3 (\ue{}) block scales. Their wider range permits periodic tensor scaling, while our recipe applies selective stochastic rounding to backward gradients, omits RHT, and uses FP4 in all eligible internal linears.   We pretrain a Nemotron-H 8B model for nearly 190 billion tokens. Compared with Transformer Engine \nv{}, the proposed block-16 recipe finishes with lower final-window training loss and, under their respective quantized-inference policies, lower validation loss measured as held-out negative log-likelihood. Its quantized-inference downstream point est
    
[^9]: 学习类谱的无网格离散化方法

    Learning Spectral-Like Mesh-Free Discretisations

    [https://arxiv.org/abs/2609.02833](https://arxiv.org/abs/2609.02833)

    该论文提出SpeND方法，将无网格离散化中欠定自由度的选择转化为学习问题，利用神经网络参数化模板权重，从而在细尺度波数处获得类谱方法的精度。

    

    无网格方法，如经核修正的光滑粒子流体动力学（SPH）、径向基函数生成的有限差分法（RBF-FD）以及局部各向异性基函数方法（LABFM），通过在局部模板上施加多项式一致性来构造离散微分算子。当模板中包含的节点数多于一致性约束的数量时，所得线性方程组是欠定的，剩余的自由度则通过核函数的选择、基函数预处理或最小范数条件被隐式地固定。多项式一致性仅在低波数极限下对算子进行约束，而构造过程的任何部分都没有在细尺度内容所在的波数处针对精度进行选择。我们提出了类谱神经离散化方法，将这些自由度的选择转化为一个学习问题：模板权重由一个神经网络进行参数化，该神经网络以（摘要在此处截断）

    arXiv:2609.02833v1 Announce Type: cross  Abstract: Meshfree methods such as smoothed particle hydrodynamics (SPH) with kernel corrections, radial basis function-generated finite differences (RBF-FD), and the local anisotropic basis function method (LABFM) construct discrete differential operators by imposing polynomial consistency on a local stencil. For stencils containing more nodes than there are consistency constraints, the resulting linear system is underdetermined, and the remaining degrees of freedom are fixed implicitly by the choice of kernel, basis preconditioning, or a minimum-norm condition. Polynomial consistency constrains the operator only in the low-wavenumber limit, and no part of the construction selects for accuracy at the wavenumbers where fine-scale content resides. We introduce Spectral-like Neural Discretisation (SpeND), in which the choice of those degrees of freedom is cast as a learning problem: stencil weights are parametrised by a neural network conditioned 
    
[^10]: 用于恢复个体与群体层面效应的AI情境测量：基于调查测度的验证及一项职业应用

    AI Contextual Measurement for Recovering Individual and Group-Level Effects: Validation Against Survey Measures and an Occupational Application

    [https://arxiv.org/abs/2609.02821](https://arxiv.org/abs/2609.02821)

    提出AICOME框架，通过受访者层面的AI测量指标推导群体汇总值与个体偏差，从而同时恢复情境模型中的个体与群体层面效应，并利用2022年中国家庭追踪调查的职业数据及多项工作相关调查变量加以验证。

    

    研究者越来越多地使用人工智能来构建传统调查中缺失的社会、组织和职业特征测量指标。我们提出AICOME（AI COntextual MEasurement，AI情境测量），这是一个用于评估AI生成的受访者层面测量指标能否在情境模型中恢复个体和群体层面效应的框架。其核心思想是：在受访者层面构建的AI测量指标可用于推导其群体层面的汇总值和个体偏差，从而使研究者能够同时估计组间和组内关联，而不仅仅将AI测量视为响应预测。我们使用2022年中国家庭追踪调查（CFPS）对该框架进行验证，其中职业提供了经验性的分组结构，多项与工作相关的调查变量提供了验证基准。对于计算机使用、外语使用、每周工作时长和管理责任等变量……（原文摘要至此截断）

    arXiv:2609.02821v1 Announce Type: new  Abstract: Researchers increasingly use artificial intelligence to construct measures of social, organizational, and occupational characteristics that are absent from conventional surveys. We propose AICOME, AI COntextual MEasurement, a framework for evaluating whether AI-derived respondent-level measures can recover individual and group-level effects in contextual models. The key idea is that an AI measure constructed at the respondent level can be used to derive its group-level aggregate and its individual deviation, allowing researchers to estimate both between-group and within-group associations rather than treating AI measurement as response prediction alone.   We validate the framework using the 2022 China Family Panel Studies (CFPS), where occupations provide the empirical grouping structure and several job-related survey variables provide validation benchmarks. For computer use, foreign-language use, weekly hours, and management responsibil
    
[^11]: Cliff：从第一个错误中学习过程奖励

    Cliff: Learning Process Rewards from the First Mistake

    [https://arxiv.org/abs/2609.02817](https://arxiv.org/abs/2609.02817)

    Cliff利用现成的大语言模型作为教师来定位每次推理中的第一个错误，将采样序列分解为正确前缀与错误后缀，从而在无需专门奖励模型、也无需假设师生推理模式一致的情况下，实现更有效的过程奖励塑形。

    

    具有可验证奖励的强化学习（RLVR）已成为大语言模型（LLM）后训练的强大范式，但其对粗粒度结果奖励的依赖导致对中间推理过程的指导有限。现有的方法，如过程奖励建模和在线策略蒸馏，引入了额外的约束，例如需要依赖专门的奖励模型，或假设教师与学生之间具有相同的推理模式。然而，我们观察到，一旦推理过程首次出错，对后续推理进行评估所能提供的额外信息就很有限，因为后续推理已经以一个无效的前缀为条件。因此，我们提出了Cliff，一种奖励塑形策略，它利用现成的大语言模型作为教师来识别每个采样序列（rollout）中的第一个错误。由此，采样序列自然地被分解为两个部分：正确的前缀和错误的后缀。Cliff随后将该……

    arXiv:2609.02817v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) has emerged as a powerful paradigm for large language model (LLM) post-training, but its reliance on coarse outcome rewards leads to limited guidance on intermediate reasoning processes. Existing approaches such as process reward modeling and on-policy distillation introduce additional constraints, such as reliance on a specialized reward model or assuming identical reasoning patterns between teacher and student. Nevertheless, we observe that once a reasoning process first goes wrong, evaluating the subsequent reasoning provides limited additional information, as it is already conditioned on an invalid prefix. Therefore, we propose Cliff, a reward shaping strategy that utilizes an off-the-shelf LLM as a teacher to identify the first mistake in each rollout. As a result, the rollout is naturally decomposed into two parts: a correct prefix and an incorrect suffix. Cliff then converts th
    
[^12]: 面向语言模型的荷兰赌

    Dutch Books for Language Models

    [https://arxiv.org/abs/2609.02797](https://arxiv.org/abs/2609.02797)

    该论文基于德·菲内蒂定理提出一种利用线性规划计算荷兰赌利润的评估方法，无需真实结果标签即可量化语言模型概率预测的不连贯性，并发现语言模型预测存在显著的不连贯现象。

    

    人们越来越多地使用语言模型来辅助生活决策。许多此类决策涉及概率预测：某个重大生活事件、自然灾害或经济结果发生的可能性有多大？语言模型的用户可能默认这些预测源自一个连贯一致的世界模型。在本文中，我们通过一个基于德·菲内蒂定理的评估程序来检验语言模型概率预测的连贯性。我们让语言模型对基于股票收益数据生成的事件做出预测，然后利用线性规划计算最大的荷兰赌利润——即套利者通过针对模型生成的概率下注所能确保获得的利润——并将其作为衡量不连贯性的指标。我们的评估方法不需要真实结果标签，因此即使在结果尚未被观测或尚未确定的情况下，也能对预测的连贯性进行评估。我们发现语言模型的预测中存在大量不连贯性的证据。

    arXiv:2609.02797v1 Announce Type: cross  Abstract: People increasingly use language models to support life decisions. Many such decisions involve a probabilistic forecast: How likely is a major life event, a natural disaster, or an economic outcome? Users of language models may implicitly trust that these forecasts fall out of a coherent world model. In this paper, we evaluate the coherence of language model probabilistic forecasts through a procedure that builds on a theorem due to de Finetti. We elicit forecasts from language models across events generated from stock returns data. We then use linear programs to compute the largest Dutch-book profit - the profit an arbitrageur could guarantee by betting against model-generated probabilities - which we use as a measure of incoherence. Our procedure does not require outcome labels, so we can evaluate coherence even in settings where outcomes are not observed or have not yet resolved. We find substantial evidence of incoherence in langua
    
[^13]: 压缩感知中可调线性生成先验的全模型最优性

    Full-Model Optimality for Tunable Linear Generative Priors in Compressed Sensing

    [https://arxiv.org/abs/2609.02790](https://arxiv.org/abs/2609.02790)

    本文针对压缩感知中通过奇异值分解相互关联的可调线性生成先验族建立了理论，证明在无噪声高斯压缩感知中，全维线性先验在整个先验族中达到最小的期望重建误差。

    

    生成模型作为压缩感知等逆问题的先验，已在实验和理论层面得到广泛研究。Gunn 等人最近的工作研究了具有可调复杂度的生成先验的使用方法，即维护一个包含不同复杂度的生成先验族，并在重建阶段选择特定的复杂度。他们证明，通过适当地调整生成先验的复杂度，可以在多种逆问题中实验性地获得更低的重建误差。在本文中，我们针对通过奇异值分解自然关联的可调线性生成先验族的设定，为压缩感知建立了理论。我们证明，在无噪声高斯压缩感知中，全维线性先验在整个线性先验族上达到了最小的期望重建误差。因此，在这种理想化的线性无噪声环境中（摘要在此处截断）……

    arXiv:2609.02790v1 Announce Type: cross  Abstract: Generative models have been studied experimentally and theoretically as priors for inverse problems such as compressed sensing. Recent work by Gunn et al. studied the use of generative priors with tunable complexity, where a family of generative priors with varying complexity is maintained and a specific complexity can be selected at inversion time. They demonstrated that lower reconstruction errors can be experimentally attained for a variety of inverse problems by appropriately tuning the complexity of the generative prior. In the present paper, we establish theory for compressed sensing in the setting of a tunable family of linear generative priors naturally related through their singular value decompositions. We prove that in noiseless Gaussian compressed sensing, the full-dimensional linear prior attains the minimum expected reconstruction error over the entire family of linear priors. Thus, in this idealized linear noiseless sett
    
[^14]: CodePoisonRAG：针对检索增强代码生成的知识投毒攻击

    CodePoisonRAG: Knowledge Poisoning Attacks on Retrieval-Augmented Code Generation

    [https://arxiv.org/abs/2609.02774](https://arxiv.org/abs/2609.02774)

    该论文提出CodePoisonRAG，一种针对检索增强代码生成的定向上游知识投毒攻击框架，通过CWE特定漏洞注入与语义错误标注，在不修改大语言模型的前提下将良性代码条目转化为传播攻击者选定弱点的投毒构件。

    

    检索增强代码生成（RACG）通过检索外部代码构件、文档和补丁并将其纳入生成上下文，从而改进了基于大语言模型的软件开发。这种对外部知识的依赖引入了一个关键的信任边界：被投毒的构件可以在不修改底层大语言模型的情况下影响生成的代码。先前的工作表明，选择现有的易受攻击示例可以提高RACG输出的一般漏洞率，但尚未解决黑盒攻击者能否构建单个与任务匹配的构件来传播攻击者所选定弱点的问题。我们提出了CodePoisonRAG，这是一个有针对性的上游知识投毒框架，能够将良性的固定代码条目转化为被投毒的构件。其攻击链结合了CWE特定的漏洞注入（在保持任务对齐的同时嵌入选定的从源到汇聚点的漏洞流）与语义错误标注（添加虚假的……）

    arXiv:2609.02774v1 Announce Type: cross  Abstract: Retrieval-Augmented Code Generation (RACG) improves LLM-based software development by retrieving external code artifacts, documentation, and patches, and incorporating them into the generation context. This reliance on external knowledge introduces a critical trust boundary: poisoned artifacts can influence generated code without modifying the underlying LLM. Prior work shows that selecting existing vulnerable examples can increase the general vulnerability rate of RACG outputs, but leaves open whether a black-box attacker can construct a single task-matched artifact that propagates an attacker-selected weakness. We introduce CodePoisonRAG, a targeted upstream knowledge-poisoning framework that transforms benign fixed-code entries into poisoned artifacts. Its attack chain combines CWE-specific Vulnerability Injection, which embeds a selected source-to-sink flow while retaining task alignment, with Semantic Mislabeling, which adds false
    
[^15]: 从重加权到重写：解锁训练数据归因中影响力样本的干预效果

    From Reweighting to Rewriting: Unlocking the Intervention Effects of Influential Samples in Training Data Attribution

    [https://arxiv.org/abs/2609.02771](https://arxiv.org/abs/2609.02771)

    该论文发现重加权无法释放影响函数所识别样本的干预价值，并提出“影响引导的响应重写”方法——通过重写所选样本的响应而非调整其权重，从而真正解锁训练数据归因中影响力样本的干预效果。

    

    训练数据归因（TDA）旨在识别塑造模型行为的训练样本，但其干预价值既取决于选择了哪些样本，也取决于如何对这些样本进行修改。影响函数（IF）估计的是无穷小重加权下的行为变化，然而在常规的基于权重的干预下，由IF选出的样本相较于随机选择的样本往往优势有限。这引出了一个问题：是有影响力的样本本身缺乏干预价值，还是重加权方式未能实现其行为杠杆作用。我们提出影响引导的响应重写方法，该方法利用影响函数识别干预目标，并在保持指令不变的情况下，将其响应替换为与目标行为一致或相反的监督信号。我们在四个开源权重的大语言模型上，以认知性弃答为主要测试场景，在相同的影响选择样本上比较了重写与重加权两种干预方式。响应重写产生了

    arXiv:2609.02771v1 Announce Type: cross  Abstract: Training data attribution (TDA) aims to identify training examples that shape model behavior, but its intervention value depends on both which examples are selected and how they are modified. Influence functions (IF) estimate behavioral changes under infinitesimal reweighting, yet IF-selected examples often show limited advantages over random selection under conventional weight-based interventions. This raises the question of whether influential examples lack intervention value or whether reweighting fails to realize their behavioral leverage.We introduce influence-guided response rewriting, which uses IF to identify intervention targets and replaces their responses with behavior-aligned or behavior-opposed supervision while keeping instructions fixed. Across four open-weight LLMs, we compare rewriting and reweighting on the same influence-selected examples using epistemic abstention as our primary testbed. Response rewriting produces 
    
[^16]: 表格基础模型懂物理吗？数据污染、物理单位与确定性极限

    Do Tabular Foundation Models Know Physics? Contamination, Units, and the Deterministic Limit

    [https://arxiv.org/abs/2609.02766](https://arxiv.org/abs/2609.02766)

    本文在316个物理方程采样的数据集上评估四种表格基础模型，发现其性能虽全面超越基线模型，但其贝叶斯先验既无法表示无噪声机制也无法表示物理单位，因此只能对物理数据进行插值而不能充当真正的物理模型。

    

    表格基础模型（TFMs）以语言模型填充文本的方式学习填充表格，而表格可以说是大多数物理测量数据所呈现的格式。它们在这一过程中学到了物理知识吗？由于它们在构造上是贝叶斯模型，因此问题在于其先验中包含什么。我们直接探测了这一问题，在从316个物理方程采样的数据集上（涵盖域内和域外），将四个模型（TabPFN-3、TabICLv2、TabDPT和Real-TabPFN-2.5）与六个基线模型进行了对比评估。TFMs无论开箱即用还是经过调优后都表现卓越。但我们证明了它们的先验既无法表示无噪声的物理机制，也无法表示物理单位，这正是它们能够对物理数据进行插值、却尚不能充当真正物理模型的原因。

    arXiv:2609.02766v1 Announce Type: new  Abstract: Tabular foundation models (TFMs) learn to fill in tables the way language models fill in text, and tables are arguably the format in which most physical measurement arrives. Did they learn any physics in the process? They are Bayesian by construction, so the question is what their prior contains. We probe it directly, evaluating four of them (TabPFN-3, TabICLv2, TabDPT and Real-TabPFN-2.5) against six baselines on datasets sampled from 316 physical equations, in and out of domain. TFMs dominate, out of the box and after tuning. But we show that their prior can represent neither a noiseless mechanism nor physical units, which is why they interpolate physics without yet being able to act as physical models.
    
[^17]: 剖析医学问答中误导性上下文的作用机制

    Untangling the Mechanisms of Misleading Context in Medical Question Answering

    [https://arxiv.org/abs/2609.02754](https://arxiv.org/abs/2609.02754)

    该研究通过在MedMisBench中注入伪造证据和纯粹断言两类误导性上下文，系统揭示了推理模型的医学判断被误导的机制，发现模型对纯粹断言的易感性显著高于伪造证据（高出10至27个百分点），且误导信息虽在推理轨迹中被大量披露却难以被察觉。

    

    大语言模型如今能够以专家级水平回答医学问题。然而，这些系统所依据的上下文可能具有误导性，而误导性上下文会破坏模型的医学判断。为了理解误导性上下文如何破坏这一判断，我们考察了模型对上下文的易感性、对误导信息的披露程度、推理被破坏的机制以及决策的可监控性。在一个由临床医生审核、包含8,627个问题的问答基准MedMisBench的医学推理子集上，我们注入了两类误导性上下文线索：伪造的证据和纯粹的断言。我们测试了三个推理模型，其中两个会暴露其完整的推理轨迹，另一个前沿模型仅暴露其最终回复。所有三个模型都更容易受纯粹断言而非伪造证据的影响，采用断言答案的频率高出10至27个百分点。误导性线索在81%至98%的推理轨迹中被披露，但仅……

    arXiv:2609.02754v1 Announce Type: cross  Abstract: Large language models now answer medical questions with expert-level performance. However, the context these systems act on can be misleading, and misleading context can corrupt a model's medical judgment. To understand how misleading context corrupts this judgment, we examine the model's susceptibility to the context, disclosure of it, mechanism of corrupted reasoning, and monitorability of the decision. On the medical reasoning subset of MedMisBench, a clinician-reviewed question-answering benchmark of 8,627 questions, we inject two types of misleading context cues, fabricated evidence and a bare assertion. We test three reasoning models, two that expose their full reasoning trace and one frontier model that exposes only its response. All three are more susceptible to the assertion than to the fabricated evidence, adopting the asserted answer 10 to 27 points more often. The misleading cues are disclosed in 81 to 98% of traces but onl
    
[^18]: HiPoly：一种用于性能预测与生成式设计的分层聚合物原生人工智能框架

    HiPoly: a hierarchical polymer-native AI framework for property prediction and generative design

    [https://arxiv.org/abs/2609.02746](https://arxiv.org/abs/2609.02746)

    提出HiPoly框架，基于G2RINS表示的三级分层图架构直接编码聚合物的随机连接性、组成和分子量，实现了从实验配方数据到性能预测、生成式分子设计和物理验证的端到端AI驱动聚合物设计工作流程。

    

    聚合物材料是现代技术的核心，其应用涵盖能源、健康和交通运输等广泛领域。尽管人工智能在材料发现方面已取得重大进展，但聚合物在多个长度尺度上呈现的分层结构，使其本质上难以用统一且具有物理意义的方式进行表示。在此，我们提出了HiPoly——一个聚合物原生的人工智能框架，它通过基于G2RINS表示构建的三级分层图架构来处理完整的聚合物描述。HiPoly在其架构中直接编码了单体间的随机连接性、组成和分子量，并采用了反映聚合物体系多尺度特性的物理启发设计原则。该框架建立了一个端到端的AI驱动工作流程，涵盖从实验配方数据到性能预测、生成式分子设计，以及基于物理的验证……

    arXiv:2609.02746v1 Announce Type: cross  Abstract: Polymeric materials are central to modern technologies, with applications ranging from energy to health and transportation. Although AI has made significant advances in materials discovery, the hierarchical structure of polymers across multiple length scales makes them inherently difficult to represent in a unified and physically meaningful way. Here we introduce HiPoly, a polymer-native AI framework that processes complete polymer descriptions through a three-level hierarchical graph architecture built on the G2RINS representation. HiPoly encodes stochastic inter-monomer connectivity, composition, and molecular weight directly within its architecture, using physically motivated design principles that mirror the multi-scale nature of polymeric systems. The framework establishes an end-to-end AI-driven workflow from experimental formulation data to property prediction, generative molecular design, and physics-based validation through mo
    
[^19]: SPADE：从网联汽车视角的SPaT攻击检测

    SPADE: SPaT Attack Detection from the Connected Vehicle's Perspective

    [https://arxiv.org/abs/2609.02741](https://arxiv.org/abs/2609.02741)

    本文提出了SPADE——首个专为网联汽车车载视角下的SPaT攻击检测深度学习研究设计的带标签多模态仿真数据集，填补了入侵检测研究在SPaT消息完整性防护领域的空白。

    

    信号相位与配时（SPaT）消息是网联汽车（CV）安全的基石，使网联汽车能够通过车路协同（V2I）和车车协同（V2V）通信来感知并响应交叉口状态。当路侧单元或对等车辆被攻陷时，一系列应用层攻击可以绕过传统认证机制，从而威胁这些消息的完整性。现有的入侵检测研究要么从基础设施侧进行防御，要么针对V2V基本安全消息（BSM）/协同感知消息（CAM）的异常行为，而车载网联汽车视角下的SPaT完整性问题尚未得到解决。为填补这一空白，我们推出了SPADE——SPaT攻击检测与评估数据集——一个带标签的、多模态的、基于仿真的数据集，专为该领域的深度学习入侵检测系统（IDS）研究而设计。SPADE通过Eclipse MOSAIC生成，采用在SAE……

    arXiv:2609.02741v1 Announce Type: cross  Abstract: Signal Phase and Timing (SPaT) messages are a cornerstone of connected vehicle (CV) safety, enabling CVs to perceive and respond to intersection state through Vehicle-to-Infrastructure (V2I) and Vehicle-to-Vehicle (V2V) communication. The integrity of these messages is threatened by a range of application-layer attacks that can bypass conventional authentication when a roadside unit or peer vehicle is compromised. Existing intrusion detection research either defends the infrastructure side or targets V2V Basic Safety Message (BSM) / Cooperative Awareness Message (CAM) misbehavior, leaving the onboard CV perspective on SPaT integrity unaddressed.To close this gap, we introduce SPADE --- the SPaT Attack Detection and Evaluation dataset --- a labelled, multi-modal, simulation-based dataset designed specifically for deep learning IDS research in this space. SPADE is generated through Eclipse MOSAIC using runtime attack injection at the SAE
    
[^20]: 语言模型可以控制自己的注意力

    Language Models Can Control Their Own Attention

    [https://arxiv.org/abs/2609.02737](https://arxiv.org/abs/2609.02737)

    该论文提出“声明式注意力”协议，让语言模型在思维链中自主声明需要关注的上下文区域，推理引擎据此像解析工具调用一样跳过大部分KV缓存读取，从而以内在方式避免了外部评分方法每步O(N)的开销。

    

    语言模型将大部分注意力集中在上下文的一小部分上，然而它们却要读取整个KV缓存来找出少数重要的token。如果用户在100万token的对话中询问之前的某个细节，全局注意力层必须扫描完整上下文才能生成回复的每一个token。一种著名的方法通过轻量级代理分数预先选择相关token来缓解这一成本，但这种外部评分机制在每一步仍然会产生O(N)的开销。我们采取一种内在的方法，其动机源于一个简单的问题：模型难道不是已经知道上下文的哪些部分是相关的吗？为此，我们引入了声明式注意力，这是一种协议，可引导模型在其思维链中声明它需要关注的位置，将生成过程划分为三种模式：（完整上下文）、（特定区域）和（仅最近输出）。推理引擎像解析工具调用一样解析这些声明，并跳过大部分KV……

    arXiv:2609.02737v1 Announce Type: cross  Abstract: Language models spend most of their attention on a small fraction of context, yet they read the entire KV cache to find the few tokens that matter. If the user asks about a previous detail in a 1M-token conversation, global attention layers must scan the full context to generate each token of the reply. A prominent approach mitigates this cost by pre-selecting relevant tokens via lightweight proxy scores, but this extrinsic scoring still incurs O(N) per step. We take an intrinsic approach motivated by the simple question: wouldn't the model already know which parts of the context are relevant? To this end, we introduce Declarative Attention (DA), a protocol that elicits the model to declare where it needs to attend within its chain-of-thought, partitioning generation into three modes:  (full context),  (a specific region), and  (recent output only). The inference engine parses these declarations like tool calls and skips most of the KV
    
[^21]: LoRA-TSD：基于Muon风格更新的LoRA切空间谱下降法

    LoRA-TSD: Tangent-Space Spectral Descent for LoRA via Muon-Style Updates

    [https://arxiv.org/abs/2609.02734](https://arxiv.org/abs/2609.02734)

    该论文提出LoRA-TSD优化器，将LoRA更新视为固定秩矩阵流形上的切向量并在切空间内执行Muon风格的谱范数最速下降，通过仅需因子梯度、且比截断SVD收缩便宜最多2.8倍的收缩映射实现几何感知的LoRA微调，并证明其Frobenius范数版本可恢复LoRA-Pro。

    

    低秩适应（LoRA）是微调大模型的标准方法，然而当其两个因子被独立训练时，其更新忽略了它所诱导的低秩权重变化的几何结构。我们提出LoRA-TSD，一种将每个LoRA步视为固定秩矩阵流形切向量的优化器，并在该切空间内执行Muon的谱范数最速下降步，随后通过LoRA参数化固有的收缩映射将结果映射回因子。该步避免了对完整权重矩阵的昂贵操作，其收缩映射比先前流形方法所使用的截断SVD收缩最多便宜2.8倍。我们证明了该替代目标的Frobenius范数版本可恢复LoRA-Pro，并且我们确定了切投影梯度（即该流形的黎曼梯度）作为LoRA训练自然的平稳性度量，它仅凭因子梯度即可计算。

    arXiv:2609.02734v1 Announce Type: new  Abstract: Low-rank adaptation (LoRA) is the standard way to fine-tune large models, yet when its two factors are trained independently, the update ignores the geometry of the low-rank weight change it induces. We introduce LoRA-TSD, an optimizer that treats every LoRA step as a tangent vector of the fixed-rank matrix manifold and takes the spectral-norm steepest-descent step of Muon inside that tangent space, mapping the result back to the factors through a retraction native to the LoRA parametrization. The step avoids expensive operations on full weight matrices, and its retraction is up to $2.8\times$ cheaper than the truncated-SVD retraction used by prior manifold methods. We prove that the Frobenius-norm version of our surrogate recovers LoRA-Pro, and we identify the tangent-projected gradient, the Riemannian gradient of the manifold, as the stationarity measure natural to LoRA training and computable from the factor gradients alone. Under thi
    
[^22]: 大批量训练中的动量：Polyak 动量扩大临界批大小，Nesterov 动量提升数据效率

    Momentum in large-batch training: Polyak enlarges the critical batch size, Nesterov improves data efficiency

    [https://arxiv.org/abs/2609.02728](https://arxiv.org/abs/2609.02728)

    该论文在幂律核回归框架下证明，在单遍大批量训练中 Polyak 动量可将临界学习率随批大小线性放大（从而扩大临界批大小约 1/(1-ρ) 倍），而 Nesterov 动量的临界学习率以 $B^\beta$（β>1）的更快速度增长，从而显著提升数据效率，并给出了刻画完整风险动力学的标度律与三区制批大小相图。

    

    我们在单遍（one-pass）训练机制下研究动量何时以及如何改善大批量训练，并以幂律核回归作为一个易于解析的设定。我们首先通过临界学习率刻画风险的稳定性，临界学习率定义为保证训练稳定的最大学习率，并得到 $\eta_{\mathrm{SGD}}^{\mathrm{crit}}\eqsim 1$、$\eta_{\mathrm{Polyak}}^{\mathrm{crit}}\eqsim \min\{1,B(1-\rho)\}$ 以及 $\eta_{\mathrm{Nesterov}}^{\mathrm{crit}}\eqsim \min\{1,B^\beta(1-\rho)\}$，其中 $B$ 为批大小，$\rho$ 为动量因子，$\beta>1$ 为容量指数。在该允许区域内，我们推导出完整风险动力学的标度律，刻画了训练从早期瞬态阶段、经幂律衰减、直至噪声底限的完整演化过程。随后，在固定数据预算下，我们在允许的学习率与动量因子范围内最小化最后一步风险，得到了一个包含三个区制的批大小相图，揭示了（原文摘要在此处截断）

    arXiv:2609.02728v1 Announce Type: cross  Abstract: We study when and how momentum improves large-batch training in the one-pass regime, using power-law kernel regression as a tractable setting. We first characterize risk stability through the critical learning rate, defined as the largest learning rate for stable training, and obtain $\eta_{\mathrm{SGD}}^{\mathrm{crit}}\eqsim 1$, $\eta_{\mathrm{Polyak}}^{\mathrm{crit}}\eqsim \min\{1,B(1-\rho)\}$, and $\eta_{\mathrm{Nesterov}}^{\mathrm{crit}}\eqsim \min\{1,B^\beta(1-\rho)\}$, where $B$ is the batch size, $\rho$ is the momentum factor, and $\beta>1$ is the capacity exponent. Within this admissible region, we derive scaling laws for the full risk dynamics, capturing the progression from an early transient, through power-law decay, to a noise floor. We then minimize the final-step risk over the admissible learning rates and momentum factors under a fixed data budget, yielding a three-regime batch-size phase diagram that reveals how the rol
    
[^23]: 神经算子逼近强连续凸单调半群

    Neural operators approximate strongly continuous convex monotone semigroups

    [https://arxiv.org/abs/2609.02727](https://arxiv.org/abs/2609.02727)

    该论文提出Chernoff神经算子与包络神经算子，通过学习单步算子实现了对强连续凸单调半群的万能逼近并给出定量逼近速率，在非线性偏微分方程、随机最优控制和模型不确定性下的随机过程等数值例子中验证了方法的有效性。

    

    我们通过用神经算子学习其Chernoff型单步算子来逼近强连续凸单调半群。首先，我们引入了所谓的Chernoff神经算子这一一般类，并通过万能逼近定理证明它们可以任意好地逼近Chernoff单步算子。通过利用加权Hölder空间之间的稳定性估计，单步逼近误差可以在迭代过程中传播，从而得到相应半群的万能逼近。其次，我们针对包络半群引入了更专门的包络神经算子类，这使我们能够推导出定量的逼近速率。最后，我们通过多个源自非线性偏微分方程、随机最优控制以及模型不确定性下随机过程的数值例子，展示了这些神经算子的有效性。

    arXiv:2609.02727v1 Announce Type: cross  Abstract: We approximate strongly continuous convex monotone semigroups by learning their Chernoff-type one-step operators with neural operators. First, we introduce the general class of so-called Chernoff-neural operators and show in a universal approximation theorem that they can approximate the Chernoff one-step operators arbitrarily well. By using stability estimates between weighted H\"older spaces, the one-step approximation error can be propagated through the iterations which yields universal approximation of the corresponding semigroup. Second, we introduce the more specialized class of envelope-neural operators for envelope semigroups which allows us to derive quantitative approximation rates. Finally, we illustrate the effectiveness of these neural operators in several numerical examples arising from non-linear partial differential equations, stochastic optimal control and stochastic processes under model uncertainty.
    
[^24]: H3DNAS：硬件感知的ONNX原生3D点云模型压缩

    H3DNAS: Hardware-Aware ONNX-Native 3D Point Cloud Model Compression

    [https://arxiv.org/abs/2609.02684](https://arxiv.org/abs/2609.02684)

    H3DNAS是一个硬件感知的模型压缩框架，无需源代码即可直接在ONNX计算图上压缩3D点云模型，并通过通道依赖图和两阶段分层搜索实现边缘设备上的高效部署。

    

    将3D点云模型部署在NVIDIA Jetson Orin Nano等边缘硬件上受到计算和内存预算的严格限制。现有的压缩方法需要访问模型的原始源代码，这使其无法应用于供应商和模型仓库通常分发的开放神经网络交换格式二进制文件。我们提出了H3DNAS，这是一个硬件感知的模型压缩框架，可直接在ONNX计算图上运行，无需原始源代码、架构类定义或搜索期间的梯度访问。H3DNAS做出了三项贡献：（1）一个通道依赖图（CDG），将ONNX算子分类为四种约束类别，并形式化证明了自由参数比例ρ_f是拓扑不变量，这是一个可在O(|V|+|E|)时间内计算的可证明压缩上限；（2）一个两阶段分层搜索，对候选…（原文在此截断）

    arXiv:2609.02684v1 Announce Type: new  Abstract: Deploying 3D point cloud models on edge hardware such as the NVIDIA Jetson Orin Nano is severely constrained by compute and memory budgets. Existing compression methods require access to the model's original source code, rendering them inapplicable to the Open Neural Network Exchange (ONNX) binaries commonly distributed by vendors and model repositories. We present \textbf{H3DNAS}, a hardware-aware model compression framework that operates directly on ONNX computational graphs without requiring original source code, architecture class definition, or gradient access during search. H3DNAS makes three contributions: (1) a \textbf{Channel Dependency Graph (CDG)} that classifies ONNX operators into four constraint classes and formally establishes that the free parameter fraction $\rho_f$ is topological invariant, a provable compression ceiling computable in $\mathcal{O}(|V|+|E|)$; (2) a \textbf{Two-Stage Hierarchical Search} that prunes candi
    
[^25]: 基于强化学习的投资组合优化中的ESG偏好引导

    Eliciting ESG Preferences for Reinforcement Learning-Based Portfolio Optimization

    [https://arxiv.org/abs/2609.02677](https://arxiv.org/abs/2609.02677)

    该论文将ESG投资组合优化建模为同时融合三家ESG评级机构数据的多目标强化学习问题，并提出基于高斯过程的偏好引导框架，让从业者通过直观的成对比较推断自身潜在效用函数，从而解决单一评级源和人工权重设置的非直观性问题。

    

    现代投资组合管理日益需要在传统的风险调整收益与严格的环境、社会和治理（ESG）要求之间取得平衡。当前的强化学习方法通常只针对单一ESG评级机构进行优化，忽略了行业内各评级方法之间的显著差异，以及人工为相互冲突的目标分配权重的非直观性。本文通过将ESG感知的投资组合优化表述为一个多目标强化学习（MORL）问题来解决这些局限性，该问题同时纳入了三家不同ESG评级机构的评级。为了弥合高维算法权衡与人类决策之间的差距，我们集成了一个基于高斯过程的偏好引导框架。该系统使从业者能够通过对候选投资组合进行直观的成对比较来推断其潜在的效用函数。

    arXiv:2609.02677v1 Announce Type: cross  Abstract: Modern portfolio management increasingly demands a balance between traditional risk-adjusted returns and strict Environmental, Social, and Governance (ESG) mandates. Current Reinforcement Learning (RL) approaches typically optimize for a single ESG provider, neglecting the significant divergence in rating methodologies across the industry and the unintuitive nature of manually weighting conflicting objectives. This paper addresses these limitations by formulating ESG-aware portfolio optimization as a Multi-Objective Reinforcement Learning (MORL) problem that simultaneously incorporates ratings from three distinct ESG agencies. To bridge the gap between high-dimensional algorithmic trade-offs and human decision-making, we integrate a Preference Elicitation framework using Gaussian Processes. This system enables practitioners to infer their latent utility functions through intuitive pairwise comparisons of candidate portfolios based on t
    
[^26]: oHC：基于四元数在SO(4)流形上的正交超连接

    oHC: Orthogonal Hyper-Connections on SO(4) via Quaternions

    [https://arxiv.org/abs/2609.02672](https://arxiv.org/abs/2609.02672)

    该论文证明了双随机矩阵约束的混合会随深度耗尽残差流的多样性，并提出通过四元数在SO(4)流形上构造正交混合矩阵的oHC方法，既保证缩放稳定又完整保持残差流的范数与多样性。

    

    超连接用n条并行的残差流取代Transformer的单条残差流，并在每一层通过学习到的n×n残差矩阵对它们进行混合。若不对该矩阵施加约束，混合步骤对残差流的缩放因子便没有任何限制，且该因子会随层数不断累积，从而导致训练不稳定。流形约束超连接通过将矩阵限制为双随机矩阵来解决这一问题。这将缩放因子的上界限制为一，使混合无法再放大任何方向，但没有任何下界约束。我们证明，在该集合内，混合步骤只能通过缩小各残差流之间的差异来降低残差流的范数，而其均值保持不变；由于这种缩减会随层数累积，各残差流变得越来越相似，其多样性随网络深度被消耗殆尽。因此，我们提出正交超连接，……

    arXiv:2609.02672v1 Announce Type: new  Abstract: Hyper-Connections (HC) replace the single residual stream of a Transformer with $n$ parallel ones, mixing them at every layer with a learned $n \times n$ residual matrix. Leaving that matrix unconstrained places no limit on the factor by which the mixing step rescales the residual streams, and that factor compounds across layers, which destabilizes training. Manifold-constrained Hyper-Connections (mHC) address this by restricting the matrix to the doubly stochastic matrices. That caps the factor at one, so the mixing can no longer amplify any direction, but nothing bounds it from below. We prove that inside this set the mixing step can reduce the norm of the residual streams only by shrinking the differences between the streams, while their mean is left unchanged; and since the reduction accumulates over layers, the streams grow more alike and their diversity is spent with depth. We therefore propose Orthogonal Hyper-Connections (oHC), r
    
[^27]: 受限独立性下依赖维数的相关性差距界

    Dimension Dependent Correlation Gap Bounds under Restricted Independence

    [https://arxiv.org/abs/2609.02659](https://arxiv.org/abs/2609.02659)

    本文利用AI辅助的理论分析与计算验证相结合的证明方法，证明了成对独立性下单调次模函数的相关性差距在 $n=4$ 时普遍成立且达到紧的 $4/3$ 上界，解决了此前遗留的开放问题。

    

    成对独立相关性差距是指集合函数在任意依赖关系下的最大期望值与在成对独立性下的最大期望值之比，用于衡量这种独立性限制所带来的损失。在相互独立条件下，对于单调次模函数，该差距普遍被界定在 $e/(e-1)$ 以内。在成对独立条件下，针对包括 $n=3$ 在内的若干特殊情形，人们已建立了更紧的 $4/3$ 上界，并猜测该界普遍成立。近期一个AI辅助构造的反例推翻了该猜想当 $n=5$ 时的成立性，这使得 $n=4$ 时该界的有效性以及最坏情形下的紧界成为悬而未决的问题。我们解决了这两个问题。首先，对于 $n=4$，我们利用结合理论分析与计算验证的AI辅助证明，证明了 $4/3$ 上界普遍成立且是紧的。该证明结合了最优分子顶点的结构性刻画、置换对称性、锥认证……

    arXiv:2609.02659v1 Announce Type: cross  Abstract: The pairwise independent correlation gap is the ratio of the maximum expected value of a set function under arbitrary dependence to that under pairwise independence, measuring the loss from this independence restriction. Under mutual independence, this gap is universally bounded by $e/(e-1)$ for monotone submodular functions. With pairwise independence, a tighter $4/3$ upper bound was established for several special cases, including $n=3$, and conjectured to hold universally. A recent AI-assisted counterexample disproved this conjecture for $n=5$, leaving the validity of the $n=4$ bound and the tight worst case bound open.   We resolve both questions. First, for $n=4$, we establish that the $4/3$ bound holds universally and is tight using an AI-assisted proof combining theoretical analysis and computational verification. The proof combines a structural characterization of optimal numerator vertices, permutation symmetry, cone certifica
    
[^28]: 展开Leech格：面向2比特LLM权重的融合多壳解码与显存布局

    Unfolding the Leech Lattice: Fused Multi-Shell Decoding and VRAM Layouts for 2-Bit LLM Weights

    [https://arxiv.org/abs/2609.02652](https://arxiv.org/abs/2609.02652)

    本文首次实现了Leech格量化所必需的多壳解码器，通过融合GPU内核与优化的显存布局（发现显存内码率是独立于磁盘码率的设计维度），使2比特LLM权重在batch 1解码阶段实现高效推理。

    

    Leech格向量量化在其自身评估协议下保持着已报道的最强2比特量化质量。其解码内核只能解码单个壳层，而我们未发现该码率所必需的多壳解码器的任何实现。本文提供了这样一个实现，并测量其在batch size为1的解码阶段GEMV运算中的服务开销。首先，为完整的301类码本构建了一条服务路径：通过离线扩展生成GPU内存布局，并使用一个融合的反量化加矩阵向量乘内核读取这些布局且不产生warp发散，结果与f64精度进行了对照验证。其次，显存内码率是一个独立于磁盘上码率的设计维度。四种比特精确的布局在同一进程中的计时结果显示，在恒定带宽下，二进制位平面布局在大小和速度上均优于一热掩码布局（每权重4.80比特，为FP16的2.15倍）。当码率低于4.3比特时，会出现第二个不规则的数据流；当码率降至3.6比特时，解码不再能仅依靠移位和掩码操作完成。第三，已部署的4比特（AWQ）和2比特（QTIP）GEMV内核在同一进程中运行。格解码内核读取2.4……

    arXiv:2609.02652v1 Announce Type: new  Abstract: Leech-lattice vector quantization holds the strongest reported 2-bit quality under its own evaluation protocol. Its kernel decodes one shell; we found no implementation of the multi-shell decoder the rate requires. This paper supplies one and measures its serving cost for decode-phase GEMV at batch 1. First, a serving path for the full 301-class codebook: an offline expansion into GPU layouts and a fused dequantize-plus-matvec kernel reading them without warp divergence, verified against f64. Second, the in-VRAM rate is a design axis distinct from the on-disk rate. Four bit-exact layouts timed in one process show binary bit planes beating one-hot masks on size and speed at constant bandwidth (4.80 bits per weight, 2.15x FP16). Below 4.3 bits a second, irregular stream enters; at 3.6 the decode stops being shifts and masks. Third, deployed four-bit (AWQ) and two-bit (QTIP) GEMV kernels run in the same process. The trellis kernel reads 2.4
    
[^29]: Loom：通过嵌入空间重加权将诊断线索编织成自由文本共识

    Loom: Weaving Diagnostic Strands into Free-Text Consensus via Embedding-Space Reweighting

    [https://arxiv.org/abs/2609.02649](https://arxiv.org/abs/2609.02649)

    Loom是一个部署于真实工业根因分析的生成式共识框架，它将模块化启发式产生的开放式诊断假设投影到连续嵌入空间，并通过基于质心的迭代重加权算法解决冲突信号，从而把嘈杂矛盾的文本假设聚合为可靠共识。

    

    将嘈杂且相互矛盾的文本假设聚合为可靠共识，是在真实工业场景中部署NLP系统时的一项根本性挑战。虽然单体式大语言模型（LLM）智能体为根因分析（RCA）等任务提供了无限的表达能力，但它们存在上下文长度限制、幻觉不断累积以及难以承受的推理延迟等问题。传统弱监督方法虽具备统计严谨性，但在数学上仅限于离散类别。我们提出了Loom，一个部署于真实世界根因分析的生成式共识框架，它弥合了上述两种范式。Loom通过将模块化启发式方法（由事件特定实体、时间和指标动态填充的诊断模板）产生的开放式假设投影到连续嵌入空间中进行聚合，并采用基于质心的迭代重加权算法来解决冲突信号。所得的共识权重为单一……（摘要原文在此处截断）

    arXiv:2609.02649v1 Announce Type: new  Abstract: Aggregating noisy, conflicting textual hypotheses into a reliable consensus is a fundamental challenge when deploying NLP systems in real-world industrial settings. While monolithic Large Language Model (LLM) agents offer unbounded expressivity for tasks like Root Cause Analysis (RCA), they suffer from context limits, compounding hallucinations, and prohibitive inference latency. Traditional weak supervision offers statistical rigor but is mathematically restricted to discrete classes. We present Loom, a generative consensus framework deployed for real-world RCA that bridges these paradigms. Loom aggregates open-form hypotheses emitted by modular heuristics (diagnostic templates dynamically populated with episode-specific entities, times, and metrics) by projecting them into a continuous embedding space, and resolves conflicting signals with an iterative centroid-based reweighting algorithm. The resulting consensus weights ground a singl
    
[^30]: 面向基于梯度规划的可微电力市场出清

    Differentiable Electricity-Market Clearing for Gradient-Based Planning

    [https://arxiv.org/abs/2609.02646](https://arxiv.org/abs/2609.02646)

    本文提出将电力市场出清建模为可微优化层，通过反向模式自动微分将规划成本梯度经出清电价回传至规划方案，从而实现基于梯度的数据中心负荷选址优化。

    

    规划一座大型数据中心十分困难，因为规模大到足以影响电力市场的设施会改变它自身将要支付的电价。这些电价由市场出清决定，而市场出清是一个需要在每种运行工况下重新求解的约束优化问题。然而，对市场进行仿真只能告诉规划者某个候选方案的表现如何，却无法说明该如何改进。本文将市场出清视为一个可微的优化层：每次前向传播求解市场出清，而反向模式自动微分则将规划成本通过出清电价反向传播回规划方案。在用有限差分方法验证了这些梯度之后，作者将其应用于一个具体问题：在两个合成网络中，将50兆瓦的数据中心负荷分配到六个候选母线上（每个启用站点有固定成本），并在36种运行状态下进行评估。与对所有站点组合的穷举枚举相比，梯度优化能够恢复连续分配结果（摘要在此处被截断）。

    arXiv:2609.02646v1 Announce Type: new  Abstract: Planning a large data center is difficult because a facility big enough to matter changes the electricity prices it will pay. Those prices are set by market clearing, a constrained optimization problem solved anew in every operating condition. However, simulating the market tells a planner how a candidate plan performs but not how to improve it. Here we treat market clearing as a differentiable optimization layer: each forward pass solves the market, and reverse-mode automatic differentiation propagates the planning cost back through the cleared prices to the plan. After validating these gradients against finite differences, we apply them to a concrete problem: allocating 50 MW of data-center load across six candidate buses in two synthetic networks, under a fixed cost per active site, evaluated over 36 operating states. Judged against exhaustive enumeration of all site combinations, gradient optimization recovers the continuous allocati
    
[^31]: TaRA：训练感知的低秩适应初始化

    TaRA: Training-Aware Low-Rank Adaptation Initialization

    [https://arxiv.org/abs/2609.02639](https://arxiv.org/abs/2609.02639)

    TaRA提出了一种训练感知的LoRA初始化方法，通过使低秩因子诱导的梯度密切逼近全秩权重矩阵的梯度来提升训练初期的梯度保真度，且几乎不增加计算开销。

    

    低秩适应已成为参数高效微调（PEFT）的事实标准，然而由于低秩分解带来的信息瓶颈，其性能对初始化高度敏感。现有方法试图通过利用预训练权重、激活值或梯度的主成分来构建高质量的LoRA初始化，但这些方法并未直接考虑全秩模型的训练动态。本文提出了训练感知低秩适应初始化，该方法在初始化LoRA时，使低秩因子所诱导的梯度能够密切逼近相应全秩权重矩阵的梯度。TaRA源于数学公式推导，在提升训练初期梯度保真度的同时，引入的计算开销几乎可以忽略不计。在多样且具有挑战性的微调任务中，TaRA始终……

    arXiv:2609.02639v1 Announce Type: cross  Abstract: Low-Rank Adaptation (LoRA) has become a de facto standard for parameter-efficient fine-tuning (PEFT), yet its performance is highly sensitive to initialization due to the information bottleneck imposed by low-rank decomposition. Existing approaches attempt to construct high-quality LoRA initializations by exploiting principal components of pretrained weights, activations, or gradients. However, these methods do not directly account for the training dynamics of the full-rank model. In this paper, we propose Training-aware Low-Rank Adaptation Initialization (TaRA), a method that initializes LoRA such that the gradients induced by the low-rank factors closely approximate the gradient of the corresponding full-rank weight matrix. Derived from a mathematical formulation, TaRA improves gradient fidelity at the start of training while introducing negligible computational overhead. Across diverse and challenging fine-tuning tasks, TaRA consist
    
[^32]: 神谕，我能学会吗？——链接预测模型间预测收敛性与互补性研究

    Oracle, will I ever learn? A study of prediction convergence and complementarity across link prediction models

    [https://arxiv.org/abs/2609.02638](https://arxiv.org/abs/2609.02638)

    该论文提出用“神谕”方法衡量链接预测模型的互补性——即针对每个查询在多个模型中选取最佳预测，以此量化不同模型（甚至同一模型的不同训练过程）所捕获知识的互补程度，并给出模型组合可达到的性能上限。

    

    知识图谱已成为网络应用（包括搜索、问答和推荐系统）中结构化知识的重要来源。在这些应用中，链接预测既可以作为预测任务本身，也可以作为丰富不完整知识图谱以供下游任务使用的手段。有趣的是，不同的链接预测模型，甚至同一模型的不同训练过程，对于同一查询可能会产生截然不同的预测结果。这表明模型在捕获底层知识方面存在变异性，由此引出一个根本性问题：不同模型在多大程度上捕获了互补的知识，以及通过组合这些模型能够恢复多少这样的知识？我们提出通过一个“神谕”的性能来衡量模型互补性，该神谕针对每个查询在一组被考虑的模型中选择最佳预测，从而提供了性能的上限。

    arXiv:2609.02638v1 Announce Type: new  Abstract: Knowledge graphs have become an important source of structured knowledge for Web applications, including search, question answering, and recommender systems. In these applications, link prediction can serve either as a prediction task itself or as a means to enrich incomplete knowledge graphs for downstream tasks. Interestingly, different link prediction models, or even different training runs of the same model, can produce substantially different predictions for the same query. This suggests a variability in the capture of the underlying knowledge by models, thus raising a fundamental question: to what extent do different models capture complementary knowledge, and how much of this knowledge could be recovered by combining them? We propose to measure model complementarity through the performance of an oracle that, for each query, selects the best prediction among a considered set of models, hence providing an upper bound on the performa
    
[^33]: 基于语音印象引导伪三元组构建的可扩展方向跟随语音合成

    Scalable Direction-Following TTS via Voice Impression-Guided Pseudo Triplet Construction

    [https://arxiv.org/abs/2609.02623](https://arxiv.org/abs/2609.02623)

    提出一种利用印象可控语音合成模型与大语言模型自动构建（参考语音、方向文本、修改后语音）伪三元组的可扩展流水线，解决了方向跟随语音合成中训练数据稀缺的问题，仅凭伪数据即可实现稳定的说话人特征保留式风格修改。

    

    语音演员常常需要重新朗读同一段剧本，并根据表演指示调整自己的演绎方式。我们将这一场景定义为“方向跟随语音合成”，即系统在保留说话人身份和语言内容的前提下，生成一段相对于参考语音能够体现给定表演指示的新语音。该方法面临的一个关键挑战是缺乏能够捕捉此类相对修改的训练数据。为解决这一问题，我们提出了一种可扩展的伪三元组构建流水线，用于生成（参考语音、方向文本、修改后语音）三元组。该流水线利用印象可控的语音合成模型生成受控的风格变化，并借助大语言模型根据估计的印象差异生成自然语言的方向描述。实验结果表明，仅使用伪三元组即可实现稳定的、保留说话人特征的语音修改；而将伪数据与真实录制数据相结合，还能在保持其他性能的同时进一步提升方向对齐度。

    arXiv:2609.02623v1 Announce Type: cross  Abstract: Voice actors often re-read the same script while modifying their delivery in response to performance directions. We study this setting as direction-following TTS, where a system generates a new utterance that reflects a given direction relative to a reference utterance while preserving speaker identity and linguistic content. A key challenge is the lack of training data capturing such relative modifications. To address this, we propose a scalable pseudo-triplet construction pipeline that generates~(reference utterance, direction text, modified utterance) triplets. It generates controlled style variations using an impression-controllable TTS model and uses an LLM to produce natural language directions from estimated impression differences. Experimental results demonstrate that pseudo-triplets alone enable stable speaker-preserving modification, and that combining pseudo and recorded data further improves direction alignment while mainta
    
[^34]: 通过后验平均进行源分布估计

    Source Distribution Estimation by Posterior Averaging

    [https://arxiv.org/abs/2609.02622](https://arxiv.org/abs/2609.02622)

    该论文提出用期望最大化框架求解源分布估计问题，E步在当前源估计生成的新鲜仿真数据上训练摊销后验，M步将后验平均值重新拟合为源分布，从而避免了对一次性训练的似然代理模型的依赖。

    

    基于仿真的科学通常需要一个关于仿真器参数的分布，使其前推结果能够重现一组真实观测数据：这就是源分布估计（SDE）问题。现有方法基于从固定提议先验一次性训练得到的似然代理模型来拟合源分布。因此，它们的目标函数仅以代理模型而非真实仿真器来表述，这在代理模型从未训练过的参数空间不准确区域可能会失效。我们转而通过期望最大化（EM）方法求解SDE：E步在来自当前源估计的新鲜仿真数据上训练摊销后验，M步将该后验在观测数据上的平均值重新拟合为源分布。我们给出两种参数化方式：(1) 分离的源流与后验流，(2) 单一共享的条件流。我们在宽泛先验和误设初始先验两种情况下，于三个基准任务上评估了我们的方法。

    arXiv:2609.02622v1 Announce Type: new  Abstract: Simulation-based science often requires a distribution over simulator parameters whose push-forward reproduces a set of real observations: this is the source distribution estimation (SDE) problem. Existing methods fit the source against a likelihood surrogate trained once from a fixed proposal prior. Their objective is therefore stated only in terms of the surrogate instead of the true simulator, which may fail for inaccurate areas in parameter space where the surrogate was never trained. We instead solve SDE by expectation maximization: an E-step trains an amortized posterior on fresh simulations from the current source estimate, and an M-step refits the source to the average of that posterior over the observed data. We give two parameterizations, (1) separate source and posterior flows and (2) a single shared conditional flow. We evaluate our method on three benchmark tasks under both broad and misspecified initial priors. Both improve
    
[^35]: 基于学习的坐标混淆点云重构攻击

    Learning-Based Reconstruction Attacks on Coordinate-Obfuscated Point Clouds

    [https://arxiv.org/abs/2609.02568](https://arxiv.org/abs/2609.02568)

    本文评估了点云选择性坐标加密方案在面对机器学习重构攻击时的安全性，证明攻击者可利用未加密数据中的空间与几何相关性，在不解密的情况下恢复被加密的坐标，从而揭示了该加密框架的安全隐患。

    

    基于点云表示的体积视频能够支持沉浸式虚拟现实和增强现实应用，但为高效且安全的内容传输带来了重大挑战。先前的工作提出了一种针对点云的选择性坐标加密框架，该框架仅加密部分坐标，在降低计算成本的同时使未授权内容在视觉上产生退化。然而，剩余未加密的信息是否足以支持内容重构仍不明确。在本文中，我们评估了选择性坐标加密抵御基于机器学习的重构攻击的鲁棒性。我们考虑这样一类攻击者：他们能够访问选择性加密的点云，并试图通过利用未加密数据中的空间和几何相关性，在不解密的情况下恢复被加密的坐标。我们在两种加密粒度下评估了PointNet和随机森林模型的表现……

    arXiv:2609.02568v1 Announce Type: cross  Abstract: Volumetric video based on point cloud representations enables immersive virtual and augmented reality applications but introduces significant challenges for efficient and secure content delivery. Prior work proposed a selective coordinate encryption framework for point clouds that encrypts only a subset of coordinates, reducing computational costs while visually degrading unauthorized content. However, it remains unclear whether the remaining unencrypted information is sufficient to enable content reconstruction.   In this paper, we evaluate the robustness of selective coordinate encryption against machine learning-based reconstruction attacks. We consider an attacker with access to selectively encrypted point clouds attempting to recover encrypted coordinates without decryption by exploiting spatial and geometric correlations in the unencrypted data. We evaluate PointNet and Random Forest models under two encryption granularities: \te
    
[^36]: 通过分布式模型-智能体耦合在英国气象局统一模式中实现在线强化学习

    Online Reinforcement Learning in the Met Office Unified Model through Distributed Model-Agent Coupling

    [https://arxiv.org/abs/2609.02566](https://arxiv.org/abs/2609.02566)

    该研究通过分布式模型-智能体耦合将强化学习智能体嵌入英国气象局统一模式，实现在线训练有界的位温订正，在保持数值稳定性的同时于多数纬度带降低了500百帕位势高度的平均绝对误差。

    

    机器学习订正只有在适应不断演变的模式状态、同时保持动力一致性和数值稳定性的前提下，才能真正补充数值天气预报。为了在全球预报模式中检验这一点，我们通过秩局部张量将英国气象局（UKMO）统一模式（UM）与分布式强化学习智能体相耦合。一个DDPG actor在每条大气柱的70个垂直模式层之间共享权重，并对模式倾向施加有界的位温订正。在十次nudging训练预报中，向UKMO业务分析场进行nudging计算提供了即时的反事实目标。随后将冻结的策略在非nudging预报中进行推理评估。该耦合工作流成功完成了训练，并在评估案例中保持数值稳定。与匹配的原始UM预报在+6小时相比，学习到的策略在六个纬度带中的四个降低了Z$_{500}$平均绝对误差（MAE）。

    arXiv:2609.02566v1 Announce Type: new  Abstract: Machine-learnt corrections can complement numerical weather prediction only if they adapt to the evolving model state while preserving dynamical consistency and numerical stability. To test this within a global forecasting model, we couple the Met Office (UKMO) Unified Model (UM) with distributed RL agents through rank-local tensors. A DDPG actor shares weights across the 70 vertical model levels of each atmospheric column and applies bounded potential-temperature corrections to the model tendencies. Across ten nudged training forecasts, nudging calculations towards the UKMO operational analysis provides an immediate counterfactual target. The frozen policy is then evaluated in a non-nudged forecast for inference. The coupled workflow successfully completes training and remains numerically stable in the evaluated case. Relative to a matched native UM forecast at +6 h, the learnt policy reduces Z$_{500}$ MAE in four of six latitude bands,
    
[^37]: ProbeMatchDTI：面向药物-靶点相互作用预测的探针驱动多尺度生化模式匹配

    ProbeMatchDTI: Probe-Driven Multi-Scale Biochemical Pattern Matching for Drug-Target Interaction Prediction

    [https://arxiv.org/abs/2609.02549](https://arxiv.org/abs/2609.02549)

    提出探针驱动的ProbeMatchDTI框架，通过IterProbe和BindingProbe保留被传统方法抑制的弱结合相关生化信号，实现多尺度药物-蛋白质模式匹配，从而改进药物-靶点相互作用预测。

    

    药物-靶点相互作用（DTI）预测是AI驱动药物发现中的一项重要任务。尽管近期的生化表示学习方法提升了DTI预测的性能，但其被动的特征聚合方式往往偏向主导性分子模式，而抑制了较弱但与结合相关的信号，例如官能团和残基-上下文模式，从而限制了多尺度生化对应关系的建模。为解决这一问题，我们提出了ProbeMatchDTI，一个由模式探针驱动的框架，包含IterProbe和BindingProbe两个模块。IterProbe显式地保留跨精炼深度的上下文状态，并在跨实体匹配之前使用可学习探针在每个位置对这些状态进行选择，从而保留较弱的生化模式，并强化官能团、局部基序与分子骨架之间的关联。BindingProbe随后在局部生化层面刻画药物与蛋白质之间的跨实体互补性

    arXiv:2609.02549v1 Announce Type: cross  Abstract: Drug-target interaction (DTI) prediction is an important task in AI-driven drug discovery. Although recent biochemical representation learning methods have improved DTI prediction, their passive feature aggregation tends to favor dominant molecular patterns while suppressing weak yet binding-relevant signals, such as functional groups and residue-context patterns, limiting the modeling of multi-scale biochemical correspondences. To address this issue, we propose ProbeMatchDTI, a pattern-probe-driven framework comprising IterProbe and BindingProbe. IterProbe explicitly retains contextual states across refinement depths and uses learnable probes to select them at each position before cross-entity matching, thereby preserving weak biochemical patterns and strengthening associations among functional groups, local motifs, and molecular scaffolds. BindingProbe then characterizes cross-entity drug-protein complementarity at local biochemical-
    
[^38]: 向正确者学习：面向多领域大语言模型的答案验证式多教师蒸馏

    Learn from Whoever Is Right: Answer-Verified Multi-Teacher Distillation for Multi-Domain LLMs

    [https://arxiv.org/abs/2609.02548](https://arxiv.org/abs/2609.02548)

    提出了MT-SDPO方法，通过答案验证机制按样本识别最可靠的教师（而非仅依赖领域匹配路由），将多个冻结的领域专家模型的能力蒸馏统一到一个学生模型中。

    

    现代大语言模型依赖强化学习在单个领域构建强大能力，但将这些能力整合到一个可部署的单一模型中仍然具有挑战性。现有方法通过将每个样本路由到与其领域匹配的教师，由领域标签决定由哪个教师提供监督。然而，领域专长只在平均意义上成立：匹配的教师对给定样本并非总是正确，而来自其他领域的教师有时反而是正确的。因此，可靠的教师必须按单个样本来识别，而非按领域来识别。在本文中，我们提出了多教师自蒸馏策略优化，这是一种在-policy 蒸馏方法，将多个冻结的教师模型统一到一个学生模型中。MT-SDPO由三个组件构成：（1）自锚点，即由来自其自身组内的正确 rollout 对某个 rollout 进行监督；（2）答案验证的资格判定，即教师...

    arXiv:2609.02548v1 Announce Type: cross  Abstract: Modern large language models (LLMs) rely on reinforcement learning to build strong capabilities in individual domains, but integrating those capabilities into a single deployable model remains challenging. By routing each sample to the teacher whose domain matches it, existing approaches let a domain label decide which teacher provides supervision. However, domain expertise holds only on average: the matched teacher is not always correct on a given sample, while a teacher from another domain sometimes is. The reliable teacher therefore has to be identified per sample, not per domain. In this paper, we introduce Multi-Teacher Self-Distillation Policy Optimization (MT-SDPO), an on-policy distillation method that unifies several frozen teachers into one student model. MT-SDPO consists of three components: (1) self-anchors, where a rollout is supervised by a correct rollout from its own group; (2) answer-verified eligibility, where a teach
    
[^39]: TrajMind：通过链式连接角色专精LoRA实现快慢结合的集体轨迹异常诊断

    TrajMind: Chaining Role-Specialized LoRAs for Fast-and-Slow Collective Trajectory Anomaly Diagnosis

    [https://arxiv.org/abs/2609.02540](https://arxiv.org/abs/2609.02540)

    TrajMind提出快慢双路径框架，在单一冻结的视觉-语言骨干网络上链式切换三个角色专精LoRA适配器，将持续在线筛查与按需诊断解耦，从而实现低延迟且经源数据可验证的集体轨迹异常诊断。

    

    从城市轨迹中诊断集体异常对交通治理而言日益重要，因为它能够揭示发生了什么、涉及了谁、以及事件发生的地点和时间。现有检测器可以高效地产生分数或标签，而视觉-语言流水线则能提供更丰富的语义；但两者都无法将可验证的诊断与低延迟监控相结合。核心挑战在于如何从源轨迹中识别集体模式并恢复精确的事件细节，而无需对每个监控窗口都运行完整的诊断流水线。因此，我们将持续在线的筛查与按需进行的诊断分离：筛查负责发出警报，而诊断只发布经源数据验证的“什么-谁-哪里-何时”记录。我们提出TrajMind，一个快慢结合的框架，它在一个冻结的视觉-语言骨干网络上切换三个角色专精的LoRA适配器。其慢速路径TrajMind_slow通过链式连接基于画布的……

    arXiv:2609.02540v1 Announce Type: new  Abstract: Diagnosing collective anomalies from urban trajectories is increasingly important for traffic governance, as it reveals what happened, who was involved, and where and when the event occurred. Existing detectors efficiently produce scores or labels, whereas vision--language pipelines provide richer semantics; neither couples verifiable diagnosis with low-latency monitoring. The central challenge is to recognize collective patterns and recover exact event details from the source trajectories without running the full diagnostic pipeline for every monitored window. We therefore separate always-on screening from on-demand diagnosis: screening raises alerts, while diagnosis releases only source-verified what--who--where--when records. We present TrajMind, a fast-and-slow framework that switches three role-specialized LoRA adapters over one frozen vision--language backbone. Its slow path, \textit{TrajMind$_{\text{slow}}$}, chains canvas-based t
    
[^40]: L2RPN环境中基于图神经网络的电网控制的图表示比较研究

    A Comparative Study of Graph Representations for GNN-Based Power Grid Control in L2RPN

    [https://arxiv.org/abs/2609.02538](https://arxiv.org/abs/2609.02538)

    本研究在L2RPN环境中对多种图表示方法进行了受控比较，发现使图复杂度与任务粒度相匹配比追求更丰富的图表示更为重要。

    

    图构建是电力网格控制深度强化学习中的一个关键但未被充分研究的设计选择。我们在“学习运行电力网络”环境中，针对拓扑控制任务，对不同图表示方法进行了受控实验比较，包括物理拓扑、电气敏感性以及混合变体。我们的研究结果表明，使图的复杂度与任务粒度相匹配比最大化表示的丰富性更为重要，并凸显了大规模受控表示研究的重要性。

    arXiv:2609.02538v1 Announce Type: new  Abstract: Graph construction is a critical but underexamined design choice in deep reinforcement learning for power grid control. We present a controlled experimental comparison of different graph representations, including physical topology, electrical-sensitivity, and hybrid variants for topology control in the Learning to Run a Power Network (L2RPN) environment. Our findings indicate that matching graph complexity to task granularity is more important than maximizing representational richness, and highlight the importance of controlled representation studies at scale.
    
[^41]: 不确定知识图谱补全的谱初始化与调度图平滑方法

    Spectral Initialization and Scheduled Graph Smoothness for Uncertain Knowledge Graph Completion

    [https://arxiv.org/abs/2609.02519](https://arxiv.org/abs/2609.02519)

    QUEST通过利用置信度加权图拉普拉斯算子特征向量进行实体嵌入谱初始化，并结合无偏小批量狄利克雷能量正则化来调度图平滑，在不增加可训练参数的情况下显著提升了不确定知识图谱补全的置信度预测和链接预测性能。

    

    不确定知识图谱（UKGs）通过为每个三元组分配连续的置信度分数来扩展知识图谱。由于大多数可能的三元组缺乏观测到的置信度，近期的方法依赖半监督学习来生成伪标签。这些方法在初始化实体嵌入时没有使用置信度加权图，从而丢弃了图的全局社区结构和枢纽结构。我们提出了QUEST，它在标准的置信度分布学习流程中没有添加任何可训练参数。首先，QUEST利用置信度加权图拉普拉斯算子的最小非平凡特征向量来初始化实体嵌入，在训练之前就融入了社区和枢纽结构信息。其次，QUEST应用无偏的小批量狄利克雷能量正则化器来保证训练早期的结构一致性。在两个UKG数据集上，QUEST在八个指标-数据集组合中的六个上超越了先前方法，提升了置信度预测和链接预测的性能。

    arXiv:2609.02519v1 Announce Type: cross  Abstract: Uncertain knowledge graphs (UKGs) extend knowledge graphs by assigning each triple a continuous confidence score. Since most possible triples lack observed confidences, recent methods rely on semi-supervised learning to generate pseudo-labels. These methods initialize entity embeddings without using the confidence-weighted graph, discarding its global community and hub structure. We introduce QUEST, which adds no trainable parameters to the standard confidence-distribution learning pipeline. First, QUEST initializes entity embeddings using the smallest non-trivial eigenvectors of the confidence-weighted graph Laplacian, incorporating community and hub structure before training. Second, QUEST applies an unbiased mini-batch Dirichlet energy regularizer to enforce early-stage structural consistency. On two UKG datasets, QUEST improves confidence prediction and link prediction on six of eight metric-dataset pairs over prior methods and mat
    
[^42]: 面向独立于表演者的身体运动情感识别的正交集成与经过验证的解释方法

    Orthogonal Ensembles and Tested Explanations for Performer-Independent Body-Motion Emotion Recognition

    [https://arxiv.org/abs/2609.02510](https://arxiv.org/abs/2609.02510)

    在留一表演者的困难评估设置下，通过组合十一个误差模式正交的模型，将12类身体运动情感识别的Macro-F1提升11.07个百分点，并提供了一套经过实验验证的解释方法，证明模型决策基于运动的身体区域证据且与拉班动作分析（LMA）属性高度一致。

    

    我们研究了在留一表演者评估下，仅基于身体、从骨骼运动进行12类表演情感分类的问题，这是一个困难且欠定的设置：随机猜测的准确率为8.3%，而协议匹配的复现STGCN++基线仅达到25.73 ± 4.03%的Macro-F1。我们表明，可靠的性能提升并非来自新架构，而是来自组合十一个具有正交误差模式的模型：在标记训练表演者上的10折留一表演者交叉验证中，等权重logit均值集成方法达到每折36.80 ± 4.00%的Macro-F1，相比同数据划分的复现基线，在协议匹配条件下提升+11.07个百分点（相对提升+43%）。我们的核心贡献是一套经过验证的解释套件：对于一个强集成成员，部位遮蔽和反事实编辑实验表明（而非仅断言）其决策依赖于基于运动的身体区域证据，并且这种区域显著性远比分类表现更符合基于规则的拉班动作分析（LMA）属性……（摘要原文在此处截断）

    arXiv:2609.02510v1 Announce Type: cross  Abstract: We study body-only, 12-class acted-emotion classification from skeleton motion under leave-performer-out (LPO) evaluation, a hard, underdetermined setting: chance is 8.3%, and a protocol-matched reproduced STGCN++ baseline reaches only 25.73 +/- 4.03% Macro-F1. We show that reliable gains come not from a new architecture but from combining eleven models with orthogonal error modes: under 10-fold LPO cross-validation on the labeled training performers, an equal-weight logit-mean ensemble reaches 36.80 +/- 4.00% per-fold Macro-F1, a protocol-matched +11.07 pp (+43% relative) over the same-split reproduced baseline. Our central contribution is a tested explanation suite: for a strong ensemble member, part-masking and counterfactual edits show (rather than assert) that its decisions depend on motion-grounded body-region evidence, and this region saliency aligns with rule-based Laban Movement Analysis (LMA) attributes far more than with cla
    
[^43]: 重新思考测试时自适应中的师生框架

    Rethinking the Teacher-Student Framework for Test-Time Adaptation

    [https://arxiv.org/abs/2609.02507](https://arxiv.org/abs/2609.02507)

    该论文挑战了测试时自适应中基于EMA的教师权重更新策略，指出误差累积在长序列场景下依然存在，并提出使用权重固定不更新的“顽固”教师，从而显著提升TTA方法在更长测试场景下的性能与稳定性。

    

    测试时自适应（TTA）近期成为一种颇有前景的策略，它允许预训练模型在部署时适应不断变化的数据分布，而无需访问任何标签。为了缓解误差累积，研究人员广泛采用了师生框架，但其长期稳定性往往被视为理所当然。在这项工作中，我们挑战了将教师权重设置为学生权重指数移动平均（EMA）这一常见策略，通过研究证明误差累积依然存在，尽管相较于通常使用的场景，这一问题主要体现在更长的序列上。我们分析了师生框架内的稳定性-可塑性权衡，并提出使用一个不更新自身权重的“顽固”教师。令人惊讶的是，我们表明这一简单的改变能够使TTA方法在多个包含更长场景的数据集上显著提升性能，并带来更强的稳定性。

    arXiv:2609.02507v1 Announce Type: new  Abstract: Test-Time Adaptation (TTA) has recently emerged as a promising strategy that allows the adaptation of pre-trained models to changing data distributions at deployment time, without access to any labels. To mitigate error accumulation, researchers have widely adopted the teacher-student framework, though its long-term stability is often taken for granted. In this work, we challenge the common strategy of setting the teacher weights to an exponential moving average of the student by showing that error accumulation still occurs, although it is mostly apparent on longer sequences compared to those commonly utilized. We analyze the stability-plasticity trade-off within the teacher-student framework and propose to use an intransigent teacher that does not update its weights. Surprisingly, we show that this simple change allows TTA methods to significantly improve their performance on multiple datasets with longer scenarios and result in increas
    
[^44]: 推荐系统评估中的训练种子与模型选择稳定性

    Training seeds and model-selection stability in recommender-system evaluation

    [https://arxiv.org/abs/2609.02499](https://arxiv.org/abs/2609.02499)

    推荐系统评估实验中，训练种子的变化往往会产生可检测的影响，仅报告单一随机种子的结果可能夸大评估结论的稳定性。

    

    推荐系统实验通常依赖于单一的随机训练种子，并假设运行间的随机性对评估结论的影响有限。这一假设存在风险，因为训练种子可能影响多种与算法相关的机制，包括参数初始化、小批量数据排序、dropout、掩码、潜在采样以及训练时的负采样。我们通过固定数据划分并在不同超参数配置下改变训练种子来检验这一假设。我们在三个层面分析种子效应：用户级指标敏感性、基于验证集的模型选择以及推荐列表的一致性。结果表明，种子变化的影响通常是可检测的，其影响程度取决于各配置之间是否明显分离、验证集结果能否迁移到测试集，以及相似得分是否会产生相似的top-k列表。研究发现表明，仅报告单种子结果可能会夸大结论的稳定性。

    arXiv:2609.02499v1 Announce Type: cross  Abstract: Recommender-system experiments often rely on a single random training seed, assuming that run-to-run stochasticity has limited impact on evaluation conclusions. This assumption is risky, as a training seed may influence several algorithm-dependent mechanisms, including parameter initialization, mini-batch ordering, dropout, masking, latent sampling, and training-time negative sampling. We examine this assumption by fixing the data partition and varying the training seed across hyperparameter configurations. We analyze seed effects at three levels: user-level metric sensitivity, validation-based model selection and recommendation-list agreement. Results show that seed variation is often detectable. Its impact depends on whether configurations are clearly separated, whether validation results transfer to test, and whether similar scores lead to similar top-$k$ lists. Findings suggest that reporting single-seed results can overstate the s
    
[^45]: RINSE：面向零样本图异常检测的鲁棒目标时正常性估计

    RINSE: Robust Target-Time Normality Estimation for Zero-Shot Graph Anomaly Detection

    [https://arxiv.org/abs/2609.02497](https://arxiv.org/abs/2609.02497)

    RINSE提出了一种无需梯度的零样本图异常检测框架，在保持源检测器固定的同时，通过迭代估计目标正常性、表示校准和证据可靠性，并结合可靠性门控秩融合与编码器集成，实现了对未见目标图的鲁棒异常检测。

    

    零样本图异常检测旨在将在源图上训练的检测器部署到未见过的、无标签的目标图上，然而领域偏移可能使源自源图的正常性概念变得不可靠。我们提出了RINSE（鲁棒迭代正常性自估计），这是一个无需梯度的目标时框架，它在保持源图训练的检测器固定不变的同时，从目标图中依次估计目标正常性、表示校准和证据可靠性。其核心思想是识别低残差目标节点的可靠子集，利用它们构建修剪的目标感知正常性模型，并通过可靠性门控秩融合和编码器集成来组合互补的异常证据。在八个未见过的目标图上，RINSE在两种独立预处理协议下均取得了评估方法中最高的平均AUPRC，同时模块消融实验和敏感性分析验证了该组合设计的有效性。这些结果支持……

    arXiv:2609.02497v1 Announce Type: cross  Abstract: Zero-shot graph anomaly detection seeks to deploy a detector trained on source graphs to unseen, unlabeled targets, yet domain shift can make source-derived notions of normality unreliable. We introduce RINSE (Robust Iterative Normality Self-Estimation), a gradient-free target-time framework that keeps the source-trained detector fixed while sequentially estimating target normality, representation calibration, and evidence reliability from the target graph. Its core idea is to identify a reliable subset of low-residual target nodes, use them to construct a trimmed target-aware normality model, and combine complementary anomaly evidence through reliability-gated rank fusion and encoder ensembling. Across eight unseen target graphs, RINSE achieves the highest average AUPRC among the evaluated methods under two separate preprocessing protocols, while block ablations and sensitivity analyses support the combined design. These results suppo
    
[^46]: DeepAffinity：基于小语言模型的电商长期属性偏好预测

    DeepAffinity: Long-Term Aspect Preference Prediction in eCommerce using Small Language Models

    [https://arxiv.org/abs/2609.02468](https://arxiv.org/abs/2609.02468)

    提出DeepAffinity框架，利用结构化提示和专用预测头微调的小语言模型（SLM）从用户时序交互历史中预测其对品牌、尺寸、颜色等产品属性的长期偏好，性能优于标准生成式微调方法，并能提升大规模推荐质量。

    

    我们探索预测电商用户对产品属性（如品牌、尺寸和颜色）的偏好——我们将这一任务定义为“属性亲和度”。解决这一任务可以提升对客户的理解，并在推荐、搜索和营销中实现细粒度的个性化。我们将属性亲和度构建为一个时序预测任务：从用户按时间顺序排列的交互历史中预测其未来的属性选择，捕捉超越当前会话、随时间演变的长期偏好。为此，我们提出了DeepAffinity，它利用小语言模型，结合结构化提示词以及针对该任务微调的专用预测头。我们展示了DeepAffinity优于标准的生成式微调方法，而通用开源大语言模型在没有进行任务特定微调的情况下表现不佳，凸显了它们在建模细致行为方面的局限性。最后，DeepAffinity在大规模跨国数据上提升了推荐质量。

    arXiv:2609.02468v1 Announce Type: cross  Abstract: We explore predicting eCommerce user preferences for product aspects such as brand, size, and color - a task we define as Aspect Affinity. Solving this task improves customer understanding and enables fine-grained personalization in recommendation, search, and marketing. We frame Aspect Affinity as a temporal prediction task: forecasting a users future aspect choices from their time-ordered interaction history, capturing long-term preferences that evolve beyond the current session. To this end, we propose DeepAffinity, which leverages Small Language Models (SLMs) with structured prompts and specialized prediction heads fine-tuned for this task. We show DeepAffinity outperforms standard generative fine-tuning methods, while general-purpose open-source LLMs perform poorly without task-specific tuning, highlighting their limits in modeling nuanced behavior. Finally, DeepAffinity enhances recommendation quality on a large-scale multination
    
[^47]: 可扩展的Kronecker-Fisher近似：面向十亿参数语言模型压缩的高效Hessian分析

    Scalable Kronecker-Fisher Approximation: Efficient Hessian Analysis for Billion-Parameter Language Models Compression

    [https://arxiv.org/abs/2609.02451](https://arxiv.org/abs/2609.02451)

    本文提出一种可扩展的Kronecker-Fisher近似方法，无需存储完整Fisher矩阵即可对十亿参数语言模型进行高效Hessian分析，发现值投影层是最脆弱的组件，为混合精度分配等压缩与优化策略提供了实用的理论工具。

    

    在本文中，我们提出了一种可扩展的基于Kronecker的近似方法，该方法无需存储整个Fisher矩阵即可捕捉跨层交互，使得对十亿参数规模的神经网络进行实用的Hessian分析成为可能，而此类网络的完整计算是不可行的。我们的方法揭示了一致的脆弱性模式：在多个模型家族中，值投影层表现出最高的敏感性和最强的跨层相关性，而其他组件则表现出架构特定的行为。通过在量化、稀疏化、层间破坏以及破坏后微调方面的大量实验，我们证明了我们的近似与性能下降和性能恢复均具有很强的相关性。我们的框架为识别大模型中的脆弱组件提供了一个实用的、有理论依据的工具，为有引导的压缩与优化策略（如混合精度分配）开辟了新的途径。

    arXiv:2609.02451v1 Announce Type: cross  Abstract: In this paper, we propose a scalable Kronecker-based approximation that captures cross-layer interactions without storing the entire Fisher matrix, enabling practical Hessian analysis for billion-parameter networks where full computation is infeasible. Our approach reveals consistent vulnerability patterns: value projection layers exhibit the highest sensitivity and strongest cross-layer correlations across multiple model families, while other components exhibit architecture-specific behaviors. Through extensive experiments on quantization, sparsification, inter-layer corruption, and post-corruption fine-tuning, we demonstrate that our approximation strongly correlates with both performance degradation and recovery. Our framework provides a practical, theoretically grounded tool for identifying fragile components in large models, opening new avenues for guided compression and optimization strategies, such as mixed-precision allocation,
    
[^48]: CACTUS：去中心化联邦学习中基于掩码引导的语义干净标签后门攻击

    CACTUS: Mask-Guided Semantic Clean-Label Backdoors in Decentralized Federated Learning

    [https://arxiv.org/abs/2609.02450](https://arxiv.org/abs/2609.02450)

    CACTUS提出了一种掩码引导的语义干净标签后门攻击方法，通过将标签一致的语义对转换为目标导向的表示偏移，并在去中心化联邦学习的对等聚合前以反事实方式施加到干净非目标嵌入上，从而在多种模态和聚合规则下实现高攻击成功率。

    

    联邦学习（FL）中的语义触发器可以比合成补丁更不显眼，但依赖样本的放置方式可能会削弱后门在各聚合轮次中的植入效果。这一挑战在去中心化联邦学习（DFL）中更为严峻，因为依赖拓扑结构的对等聚合会反复混合本地模型。CACTUS将标签一致的语义对转换为目标导向的表示偏移。掩码引导的、针对特定模态的算子隔离触发器效应，跨样本将其耦合，并在对等聚合之前以反事实方式将这些偏移应用于干净的非目标嵌入。实验涵盖语音、文本、表格和图像任务，并在九种聚合规则下进行。在30%恶意节点的条件下，CACTUS在Speech Commands数据集上达到了九种规则平均攻击成功率（ASR）51.2%，并且在四种模态中的三种上，其九种规则平均ASR在所评估的攻击中最高。敏感性分析表明，ASR随网络拓扑结构和输入分布而变化。

    arXiv:2609.02450v1 Announce Type: new  Abstract: Semantic triggers in federated learning (FL) can be less conspicuous than synthetic patches, but sample-dependent placement may weaken backdoor implantation across aggregation rounds. This challenge is compounded in decentralized FL (DFL), where topology-dependent peer aggregation repeatedly mixes local models. CACTUS converts label-consistent semantic pairs into target-directed representation shifts. Mask-guided, modality-specific operators isolate trigger effects, couple them across samples, and apply the shifts counterfactually to clean non-target embeddings before peer aggregation. Experiments cover speech, text, tabular, and image tasks under nine aggregation rules. With 30\% malicious nodes, CACTUS reaches a nine-rule mean attack success rate (ASR) of 51.2\% on Speech Commands and the highest nine-rule mean ASR among evaluated attacks on three of four modalities. Sensitivity analyses show that ASR varies with network topology and i
    
[^49]: 迈向跨越连续威胁级别的“一体通用”鲁棒性

    Towards One-for-All Robustness Across a Continuum of Threat Levels

    [https://arxiv.org/abs/2609.02440](https://arxiv.org/abs/2609.02440)

    提出威胁条件网络（TCN），通过将表示学习分解为威胁不变的共享骨干与轻量的威胁条件自适应模块，使单一模型无需针对不同攻击预算训练多个专门模型，即可在无限连续的扰动强度范围内实现鲁棒性。

    

    对抗鲁棒模型通常会对特定的攻击预算过拟合，因而需要为多样且动态变化的对抗环境部署多个专门化模型，而随着威胁空间的增长，这种策略在根本上变得难以实施。这引出了一个开放性挑战：我们能否在单一模型内实现对连续威胁级别的强鲁棒性？我们提出了威胁条件网络，其建立在表示因子分解框架之上，将表示学习分解为威胁不变的共享骨干网络和轻量级的威胁条件自适应模块。TCN 通过基于傅里叶的嵌入和通道级仿射调制，使单一模型以扰动级别为条件，并在扰动预算的分布上进行训练，从而在推理时能够跨越无限连续的威胁级别实现灵活而无缝的适应。在 CIFAR-10、CIFAR-100 和 Tiny-ImageNet 上进行的大量实验表……（原文摘要至此截断）

    arXiv:2609.02440v1 Announce Type: cross  Abstract: Adversarially robust models often overfit to a specific attack budget, necessitating multiple specialized models for diverse and dynamic adversarial environments, a strategy that becomes fundamentally intractable as the threat space grows. This raises an open challenge: can we achieve strong robustness across a continuum of threat levels within a single model? We propose the Threat Conditional Network (TCN), grounded in a representation factorization framework that decomposes representation learning into a threat-invariant shared backbone and a lightweight threat-conditional adaptor. TCN conditions a single model on the perturbation level via Fourier-based embeddings and channel-wise affine modulation, and is trained against a distribution over perturbation budgets, enabling flexible and seamless adaptation across an infinite continuum of threat levels during inference. Extensive experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet sh
    
[^50]: 当可解码性并不足够时：语言模型中的逻辑有效性表征、行为解离与因果检验

    When Decodability Is Not Enough: Logical Validity Representations, Behavioral Dissociation, and Causal Tests in Language Models

    [https://arxiv.org/abs/2609.02438](https://arxiv.org/abs/2609.02438)

    该研究发现即使大语言模型在逻辑验证任务上的行为表现接近随机，其隐藏状态中仍能近乎完美地解码出逻辑有效性信息，但因果干预显示这种表征并未被模型实际利用，揭示了“可解码性不等于因果性使用”这一重要结论。

    

    大型语言模型看起来可能具备逻辑推理能力，但仅凭答案的对错并不能告诉我们模型内部究竟表征了什么。我们使用匹配的有效-无效前提-论断对，在五个开源权重Transformer模型中研究逻辑验证任务，这些前提-论断对覆盖不同的推理家族、语义领域、模板和难度级别。尽管行为表现接近随机水平，逻辑有效性往往可以从隐藏状态中被几乎完美地解码出来，并且在留出的模板、领域和推理家族上仍保持很强的可解码性。在能够正确定义以正确性为条件的评估的情形下，有效性在行为错误的样本上也保持高度可解码。与此同时，详尽的留一法测试揭示了这种泛化能力的明显局限，而与随机对照相比，沿探针导出的有效性方向进行的因果干预仅有微弱且非特异性的效果。

    arXiv:2609.02438v1 Announce Type: new  Abstract: Large language models can look capable of logical reasoning, but correct or incorrect answers alone tell us little about what the model represents internally. We study logical verification in five open-weight transformer models using matched valid--invalid premise--claim pairs that vary across inference families, semantic domains, templates, and difficulty levels. Despite near-chance behavioral performance, logical validity is often almost perfectly decodable from hidden states and remains strongly decodable under held-out templates, domains, and inference families. Validity also remains highly decodable on behaviorally incorrect examples in the conditions where correctness-conditioned evaluation is well defined. At the same time, exhaustive leave-one-out tests reveal clear limits to this generalization, and interventions along probe-derived validity directions have only weak, nonspecific effects compared with random controls. Our result
    
[^51]: IFW-BLS：基于直觉模糊波浪损失的双鲁棒宽度学习系统

    IFW-BLS: Dual-Robust Broad Learning System with Intuitionistic Fuzzy Wave Loss

    [https://arxiv.org/abs/2609.02422](https://arxiv.org/abs/2609.02422)

    本文提出IFW-BLS，通过将有界、平滑、非对称的波浪损失与直觉模糊加权机制结合在单一优化模型中，同时实现针对大残差和低可靠样本的双重鲁棒宽度学习系统。

    

    宽度学习系统是一种高效的随机化学习模型，它通过特征节点和增强节点扩展网络宽度，并无需深层反向传播即可估计输出权重。然而，其标准的最小二乘训练在两个方面存在脆弱性：（i）由噪声、离群值或损坏标签引起的大残差可能主导目标函数；（ii）所有样本被当作同等可靠，即使某些样本处于模糊或局部冲突的区域。本文提出了IFW-BLS，即直觉模糊波浪宽度学习系统，在同一个优化模型内解决了这两种脆弱性来源。第一种鲁棒机制是残差级保护，通过用有界、平滑且非对称的波浪损失代替平方损失来实现。有界性可以防止极端残差获得无界的影响，而非对称性则允许对正向偏差和负向偏差施加不同程度的惩罚。

    arXiv:2609.02422v1 Announce Type: new  Abstract: Broad Learning System is an efficient randomized learning model that expands network width through feature and enhancement nodes and estimates the output weights without deep backpropagation. Its standard least-squares training, however, is vulnerable in two different ways: (i) large residuals caused by noise, outliers, or corrupted labels can dominate the objective, and (ii) all samples are treated as equally reliable even when some lie in ambiguous or locally conflicting regions. This paper proposes IFW-BLS, an Intuitionistic Fuzzy Wave Broad Learning System that addresses these two sources of fragility within one optimization model. The first robustness mechanism is residual-level protection, obtained by replacing the squared loss with the bounded, smooth, and asymmetric wave loss. Boundedness prevents extreme residuals from receiving unbounded influence, while asymmetry allows positive and negative deviations to be penalized differen
    
[^52]: 覆盖而非瞄准：多轮智能体信用分配中的结构性区间

    Coverage, Not Targeting: A Structural Regime in Multi-Turn Agent Credit Assignment

    [https://arxiv.org/abs/2609.02417](https://arxiv.org/abs/2609.02417)

    论文提出“验证器信息密度 V_d”这一结构性指标，揭示当终端状态验证器处于低 V_d 区间时，多轮智能体信用分配的关键在于奖励覆盖（均匀稠密分布）而非对关键轮次的精准瞄准。

    

    多轮智能体强化学习日益将信用分配视为一个“瞄准”问题：在给定终端可验证奖励的情况下，逐轮方法试图将信用定位到起关键作用的轮次上。我们识别出了预测这种做法何时正确的结构性量——验证器信息密度 V_d = k/C（智能体 C 步因果链中，验证器能够暴露其逐轮正确性的部分所占比例），并证明终端状态验证器深处于低 V_d 区间，此时“瞄准”是错误的轴。在 tau^2-bench 上进行的受控共享回放比较中（该比较将奖励密度与信用几何结构分离开来），均匀分布的连续稠密奖励优于稀疏的二值结果奖励（后者在 4/5 个随机种子上产生净负效果），而将同样的优势集中在进展轮次或随机轮次上则同样有害：瞄准只是次要因素。其机制在于覆盖：终端状态验证将可观测信号坍缩为单一的最终……（原文摘要在此处截断）

    arXiv:2609.02417v1 Announce Type: cross  Abstract: Multi-turn agentic RL increasingly treats credit assignment as a targeting problem: given a terminal verifiable reward, per-turn methods localize credit onto the turns that mattered. We identify the structural quantity that predicts when this is the right move, the verifier information density V_d = k/C (the fraction of an agent's C-step causal chain whose per-turn correctness the verifier exposes), and show that terminal-state verifiers sit deep in a low-V_d regime where targeting is the wrong axis. In controlled shared-rollout comparisons on tau^2-bench that separate reward density from credit geometry, a continuous dense reward spread uniformly beats the sparse binary outcome reward (net-harmful on 4/5 seeds), while concentrating the same advantage on progress turns or on random turns is equally harmful: targeting is second-order. The mechanism is coverage: terminal-state verification collapses the observable signal to a single fina
    
[^53]: 稀疏混合专家模型中共享路由几何与动力学的证据

    Evidence for Shared Routing Geometry and Dynamics in Sparse Mixture-of-Experts

    [https://arxiv.org/abs/2609.02404](https://arxiv.org/abs/2609.02404)

    该论文证明了稀疏混合专家模型中各层路由器的状态共享一个被层特定坐标系掩盖的共同几何结构，通过对齐后单一线性模型即可解释大部分跨层路由状态演化。

    

    稀疏混合专家模型在每个稀疏层使用独立参数化的路由器来为每个token选择专家。先前的工作表明，跨深度的路由决策通常可以从早期的路由信号预测出来，这表明路由在各层之间并非完全独立。然而，这种可预测性背后的结构仍不清楚。在这项工作中，我们提供了证据表明，各层中与路由相关的状态共享一个共同的几何结构，而这一结构被层特定的坐标系所掩盖。我们分离出每个路由器的控制子空间，并使用广义正交Procrustes分析将这些空间对齐到一个共享的规范表示中。对齐后，单个线性转换即可达到R²=0.39–0.71，并保留了单独拟合的层特定动力学模型79–90%的预测能力，这表明路由状态演化的大部分遵循一个跨深度可复用的过程。

    arXiv:2609.02404v1 Announce Type: cross  Abstract: Sparse mixture-of-experts (MoE) models use an independently parameterized router at each sparse layer to select experts for every token. Prior work has shown that routing decisions across depth can often be predicted from earlier routing signals, suggesting that routing is not fully independent across layers. However, the structure behind this predictability remains unclear. In this work, we provide evidence that routing-relevant states across layers share a common geometric structure that is obscured by layer-specific coordinate systems. We isolate the control subspace of each router and align these spaces into a shared canonical representation using generalized orthogonal Procrustes analysis. After alignment, a single linear transition reaches $R^2=0.39$--$0.71$ and retains 79--90\% of the predictive power of separately fitted layer-specific dynamics, indicating that much of routing-state evolution follows a reusable process across d
    
[^54]: 有色高斯图模型最大似然阈值的计算方法

    A computational approach to maximum likelihood thresholds for colored Gaussian graphical models

    [https://arxiv.org/abs/2609.02382](https://arxiv.org/abs/2609.02382)

    本文针对有色高斯图模型，通过几何表述建立统一理论框架并提出新的符号算法，解决了其最大似然阈值的计算问题。

    

    高斯图模型（GGMs）是可解释结构学习的重要工具。然而，在高维小样本的情形下，现有数据往往不足以保证最大似然估计量的存在。有色高斯图模型（CGGMs）通过图着色施加对称性约束来缓解这一限制，从而降低了所需的样本量。保证估计量几乎必然存在所需的最小观测数被定义为最大似然阈值（MLT）。本文通过关注MLT的几何表述来解决CGGMs的MLT计算问题：即求样本协方差矩阵的最小秩，使得其投影几乎必然位于充分统计量锥的内部。我们建立了一个统一的理论框架，将已有结果从无色模型推广到有色模型，并提出了新的符号算法。此外……

    arXiv:2609.02382v1 Announce Type: cross  Abstract: Gaussian graphical models (GGMs) are essential tools for interpretable structure learning. However, in high-dimensional, small-sample regimes, the available data is often insufficient for the maximum likelihood estimator to exist. Colored Gaussian graphical models (CGGMs) mitigate this limitation by imposing symmetry constraints through graph coloring, which reduces the required sample size. This minimal number of observations needed to guarantee that the estimator exists almost surely is defined as the maximum likelihood threshold (MLT). Here, we address the computation of the MLT for CGGMs by focusing on its geometric formulation: finding the minimum rank of a sample covariance matrix such that its projection lies almost surely within the interior of the cone of sufficient statistics. We establish a unified theoretical framework, extending results from uncolored to colored models and introducing new symbolic algorithms. Furthermore, 
    
[^55]: 优化中的渗流动力学：方差级联与离散尺度不变性

    Percolation Dynamics in Optimization : Variance Cascades and Discrete Scale Invariance

    [https://arxiv.org/abs/2609.02373](https://arxiv.org/abs/2609.02373)

    该论文将随机梯度下降的动力学建模为渗流过程，揭示出架构对称性迫使子网络以离散同步块方式合并，其结构转变在宏观序参量上表现为类似物理相变的方差尖峰，且该捕获机制及标度级联在重尾噪声模型下同样适用于Adam和AdamW。

    

    我们研究了随机梯度下降（SGD）的动力学。已知SGD会将深度神经网络引导至对应于更简单子网络的不变集上，但这种引导如何随时间展开仍然知之甚少。我们通过将随机梯度流（SGF）建模为一个渗流过程来回答这一问题：在该过程中，架构对称性迫使子网络以离散的同步块方式合并，而非一次一个。这些结构性转变会在一个宏观序参量上表现为方差尖峰，与物理相变相呼应。我们进一步证明，在显式的重尾噪声模型下，这种捕获机制及其相关的标度级联同样适用于Adam和AdamW优化器。

    arXiv:2609.02373v1 Announce Type: cross  Abstract: We study the dynamics of Stochastic Gradient Descent (SGD), which is known to steer deep neural networks toward invariant sets that correspond to simpler subnetworks. How this steering unfolds over time remains poorly understood. We answer this by modeling the stochastic gradient flow (SGF) as a percolation process, in which architectural symmetries force subnetworks to merge in discrete simultaneous blocks rather than one at a time. These structural transitions register as variance spikes in a macroscopic order parameter, echoing physical phase transitions. We further show this trapping mechanism and its associated scaling cascade extend to Adam and AdamW under an explicit heavy-tailed noise model.
    
[^56]: 基于学习可停性价值的人形机器人安全停止

    Humanoid Safe Stop via Learned Stoppability Value

    [https://arxiv.org/abs/2609.02358](https://arxiv.org/abs/2609.02358)

    提出Safe-Stop框架，将人形机器人紧急停止建模为可达-避障问题，通过结合停止概率估计器与哈密顿-雅可比可达-避障估计器这两个互补的可停性信号，实现任务无关、可跨上游任务免训练迁移的安全停止决策。

    

    人形机器人在响应紧急停止命令时，通常执行固定的动作，而不会推理判断从当前状态是否真正能够实现安全停止。我们将紧急停止建模为一个可达-避障问题，并提出 Safe-Stop——一个任务无关的框架，它将学习到的停止策略与学习到的可停性估计器相结合。这两个估计器是互补的：一个是停止概率估计器，由固定停止策略的实际结果进行监督；另一个是可达-避障估计器，由基于物理状态的哈密顿-雅可比备份进行监督。前者捕捉学习到的控制器的涌现停止行为；后者提供互补的可恢复性信号。由于停止策略和估计器不依赖于停止命令之前的行为策略，因此它们无需重新训练即可迁移到多种不同的上游任务。在部署时，两个估计值被组合起来：Safe-Stop 在（可停性条件满足时）执行停止。

    arXiv:2609.02358v1 Announce Type: cross  Abstract: Humanoid robots responding to emergency stop commands typically execute a fixed maneuver, without reasoning about whether a safe stop is actually feasible from the current state. We cast emergency stopping as a reach-avoid problem and propose Safe-Stop, a task-agnostic framework that pairs a learned stop policy with learned stoppability estimators. The estimators are complementary: a stop-probability estimator supervised by the actual outcomes of the fixed stop policy, and a reach-avoidance estimator supervised by a Hamilton-Jacobi backup over physical state. The first captures emergent stopping behavior of the learned controller; the second provides a complementary recoverability signal. Because the stop policy and estimators do not depend on the behavior policy that preceded the stop command, they transfer across diverse upstream tasks without retraining. At deployment, the two estimates are combined: Safe-Stop commits to the stop on
    
[^57]: AGI迷宫预测数据集：一个基于Transformer学习世界动力学的紧凑基准

    AGI Maze Prediction Datasets: A Compact Benchmark for Learning World Dynamics with Transformers

    [https://arxiv.org/abs/2609.02339](https://arxiv.org/abs/2609.02339)

    该论文提出了一个基于程序化生成迷宫世界的轻量级基准数据集，通过迷宫不相交的任务划分来检验Transformer是真正学到了可迁移的世界动力学规律，还是仅仅记忆了熟悉布局。

    

    世界建模要求预测模型能够维护并更新一个足以推理行动后果的内部状态。我们提出了AGI迷宫预测数据集与基准，这是一个轻量级的受控测试平台，用于研究Transformer及其他预测模型的这一能力。该基准源自程序化生成的有状态网格世界，包含逐步转移预测、固定视界状态预测和顺序文本观察预测三类任务。通过源迷宫不相交的训练与验证划分，结合贪心精确匹配评估，该基准能够将“学习可迁移的动作条件动力学”与“记忆熟悉布局中的转移”区分开来。我们建立了从零训练的字节级Transformer基线，并将其与两种工作记忆增强架构进行比较。一个通用的辅助潜在记忆Transformer可以完美拟合部分训练集，但无法持续……

    arXiv:2609.02339v1 Announce Type: cross  Abstract: World modeling requires a predictive model to maintain and update an internal state adequate for reasoning about the consequences of actions. We introduce the AGI Maze Prediction Datasets and Benchmark, a lightweight controlled testbed for studying this capability in Transformers and other predictive models. Derived from procedurally generated, stateful grid worlds, the benchmark comprises per-step transition prediction, fixed-horizon state prediction, and sequential textual-observation prediction. Source-maze-disjoint training and validation splits, together with greedy exact-match evaluation, distinguish learning transferable action-conditioned dynamics from memorizing transitions in familiar layouts. We establish from-scratch byte-level Transformer baselines and compare them with two working-memory-augmented architectures. A generic auxiliary latent-memory Transformer can fit some training sets perfectly but does not consistently im
    
[^58]: 对PGM索引的投毒攻击

    Poisoning Attacks on the PGM-index

    [https://arxiv.org/abs/2609.02328](https://arxiv.org/abs/2609.02328)

    本文提出了首个针对学习型索引PGM-index的投毒攻击PGM-attack，仅通过插入10%的对抗性键即可使索引分段数量膨胀至多120倍，并首次推导出任意插入下分段数的理论上界，证明该攻击至少能达到最优攻击效果的52%。

    

    PGM-index（Ferragina和Vinciguerra，VLDB'20）是最实用的学习型索引之一，这得益于其理论上的优雅性和持续出色的实证性能。它建立在能够最小化分段数量的最优分段线性逼近（PLA）之上。在本文中，我们探讨这种最优PLA本身对投毒攻击的敏感程度。我们提出了PGM-attack，一种高效的投毒攻击方法，通过按顺序插入对抗性键来膨胀最终的分段数量，并开发了一种方法来推导在任意插入情况下可达到的分段数量的理论上界。我们的实验表明，仅投毒10%的键，PGM-attack就能使分段数量增加至多120倍。在所有评估的实例上，我们基于实例的上界至多为PGM-attack所达分段数量的1.92倍，这证明了PGM-attack至少达到了最优解的52%。

    arXiv:2609.02328v1 Announce Type: cross  Abstract: The PGM-index (Ferragina and Vinciguerra, VLDB'20) is one of the most practical learned indexes, owing to its theoretical elegance and consistently strong empirical performance. It is built on optimal piecewise linear approximations (PLAs) that minimize the number of segments. In this paper, we ask how sensitive this optimal PLA itself is to poisoning attacks. We propose PGM-attack, an efficient poisoning attack that sequentially inserts adversarial keys to inflate the resulting number of segments, and we develop a method for deriving theoretical upper bounds on the number of segments attainable under arbitrary insertions. Our experiments show that poisoning only 10% of the keys allows PGM-attack to increase the segment count by up to 120x. On every evaluated instance, our instance-dependent upper bound is at most 1.92x the segment count attained by PGM-attack, certifying that PGM-attack achieves at least 52% of the optimum. This incre
    
[^59]: 什么值得表征？面向持续模型构建的表征赋权

    What Is Worth Representing? Representational Empowerment for Continual Model Construction

    [https://arxiv.org/abs/2609.02322](https://arxiv.org/abs/2609.02322)

    该论文提出“表征赋权”框架，通过评估候选表征元素能多大程度扩展智能体未来的建模与规划能力，来决定“什么值得被表征”，从而实现跨环境的持续模型构建。

    

    建模世界的首要问题不仅仅是估计正确的参数或因果结构，而是决定究竟什么应该被表征。我们将这一问题框架化为持续模型构建：智能体维护一个针对不可直接访问世界 W 的环境特定模型 M，并在不同环境间管理一个由可复用表征元素组成的持久库 L。我们提出表征赋权，通过衡量候选元素在多大程度上扩展智能体未来建模与规划的能力来为其评分，这是对经典赋权定义的补充，但将“控制”重新定义为对内部表征的控制，而非对外部状态的控制。我们将该框架实现为分层的“管理者-执行者”架构，并通过三个实验对其进行验证。在一个封闭词表的因果学习任务中，人类参与者会以不同的抽象粒度构建因果模型，以最大化目标可达性而非保真度……

    arXiv:2609.02322v1 Announce Type: cross  Abstract: The first problem of modeling the world is not just estimating the right parameters or causal structure, but deciding what should be represented at all. We frame this problem as continual model construction: an agent maintains an environment-specific model M of an inaccessible world W and curates a persistent library L of reusable representational elements across environments. We propose Representational Empowerment (RepEmp) to score candidate elements by how much they expand the agent's future capacity to model and plan, complementing the classic definition of empowerment, but redefined as control over internal representations instead of external states. We realize the framework as a hierarchical Curator-Actor architecture and test it across three experiments. In a closed-vocabulary causal-learning task, human participants construct causal models at varying abstraction granularities to maximize goal reachability rather than fidelity t
    
[^60]: 贝叶斯最优BER与AUC：估计量的估计与评估

    Bayes-Optimal BER and AUC: Estimation and Evaluation of Estimators

    [https://arxiv.org/abs/2609.02304](https://arxiv.org/abs/2609.02304)

    该论文提出了基于软标签来估计贝叶斯最优平衡错误率（BER）和AUC的新方法，并研究了如何评估这些估计量，从而在类别不平衡等准确率失效的场景中衡量模型性能的理论上限。

    

    机器学习中的一个基本量是任何模型在给定任务上可达到的最优性能。估计这一量可以使我们将不可消除的误差部分与模型自身的缺陷区分开来，从而告诉我们还剩多大的改进空间。最近的研究表明，在二分类任务中，贝叶斯误差（或等价地，最优准确率）可以从软标签中估计出来。然而，在类别严重不平衡或标注存在噪声的场景下，准确率往往不能很好地概括模型性能，此时平衡错误率（BER）和ROC曲线下面积（AUC）等指标更为合适。我们通过两项互补的贡献来填补这一空白：（i）估计方面，我们提出了基于软标签的最优BER和AUC估计量；我们首先考虑真实软标签和类别先验均已知的干净设定，随后将估计量扩展到更贴近现实的设定中（摘要截断）。

    arXiv:2609.02304v1 Announce Type: new  Abstract: A fundamental quantity in machine learning is the optimal performance achievable by any model on a given task. Estimating this quantity allows us to distinguish the irreducible part of the error from a deficiency of the model, telling us how much room for improvement remains. Recent work has shown that the Bayes error, or equivalently the optimal accuracy, can be estimated from soft labels in binary classification. However, accuracy is often a poor summary of performance in settings with severe class imbalance or noisy annotations, where metrics such as the balanced error rate (BER) and the area under the ROC curve (AUC) are more appropriate. We address this gap with two complementary contributions. (i) Estimation. We propose soft-label-based estimators for the optimal BER and AUC. We first consider the clean setting in which true soft labels and the class prior are known, and then extend the estimators to a more realistic setting in whi
    
[^61]: 通过推理时计算与部署支架提升评估的真实性

    Improving Evaluation Realism with Inference-Time Compute and Deployment Scaffolds

    [https://arxiv.org/abs/2609.02302](https://arxiv.org/abs/2609.02302)

    该论文提出“批判式精炼”和 DISH 智能体框架两种技术，通过投入额外推理时计算并模仿真实部署环境，使模拟对齐评估更难被能力强模型识别为测试，从而提升安全评估的真实性与结论可靠性。

    

    对齐评估面临的一个核心障碍是“评估意识”：能力强大的模型能够分辨出自己是在被测试而非被部署，这削弱了安全评估所能支持的结论。我们提出了两种技术，使模拟的对齐评估更难与真实部署区分开来。我们的第一种技术是“批判式精炼”，它在模拟器的每个动作上投入额外的推理时计算：模拟器生成多个候选动作，利用目标模型实例提供的关于如何使其更真实的反馈对候选动作进行精炼，然后以最接近真实部署的候选动作继续评估。我们的第二种技术是 DISH（模仿部署的 SWE-Agent 框架），它将目标模型封装在一个智能体框架中，缩小了编码场景下模拟环境与真实部署环境之间的差距。我们在多个目标模型上测试了这些技术，发现它们可以叠加组合：同时应用两者能带来更大的真实性提升。

    arXiv:2609.02302v1 Announce Type: new  Abstract: A core obstacle to alignment evaluation is evaluation awareness: capable models can tell when they are being tested rather than deployed, weakening the conclusions a safety evaluation can support. We present two techniques that make simulated alignment evaluations harder to distinguish from real deployments. Our first technique, critique refinement, spends additional inference-time compute on each simulator action: the simulator generates multiple candidate actions, refines them using feedback from an instance of the target model on how to make them more realistic, and continues the evaluation with the most deployment-like candidate. Our second technique, DISH (Deployment-Imitating SWE-Agent Harness), wraps the target in an agent harness, reducing the gap between simulated and real deployment environments in coding settings. We test the techniques on multiple target models and find that they compose: applying both yields larger realism g
    
[^62]: SEAL：通过共享专家对齐强化混合专家模型的全局安全性

    SEAL: Reinforcing Global Safety in Mixture-of-Experts through Shared Expert ALignment

    [https://arxiv.org/abs/2609.02293](https://arxiv.org/abs/2609.02293)

    本文提出 SEAL 方法，通过对共享专家进行安全对齐，为混合专家模型提供不依赖于稀疏路由路径的全局安全防护，有效抵御越狱提示、恶意微调以及安全关键神经元剪枝等对抗性攻击。

    

    混合专家模型是大型语言模型的一种扩展架构，它对每个 token 仅激活一小部分专家模块，从而在计算量几乎保持恒定的情况下实现大规模参数增长。近期的混合 MoE 架构进一步引入“共享专家”来捕获持续有用的表示，提升了模型的稳定性和泛化能力。目前 MoE 已支撑众多旗舰开源和商业模型，但仍然容易受到对抗性攻击。具体而言，稀疏路由引入了结构性漏洞：MoE 的安全性取决于哪些专家被激活，而攻击者可以通过越狱提示、恶意微调以及对安全关键神经元的权重级剪枝来破坏这一选择过程。现有防御方法主要侧重于强化路由器，但由于路由过程具有非确定性，攻击者仍然可以操纵或绕过路由轨迹，从而使防御失效……（摘要在此处被截断）

    arXiv:2609.02293v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) is a scaling architecture for large language models that activates only a small subset of expert modules per token, enabling massive parameter growth with nearly constant computation. Recent Hybrid MoE architecture adds \textit{shared experts} to capture consistently useful representations, further improving stability and generalization. MoE now powers many flagship open-source and commercial models, yet remains vulnerable to adversarial attacks. Specifically, sparse routing introduces a structural vulnerability: MoE safety hinges on which experts are activated, and adversaries can subvert this selection through jailbreak prompts, malicious fine-tuning, and weight-level pruning of safety-critical neurons. Existing defenses primarily focus on hardening the router, but an adversary may still manipulate or bypass the routing trajectory due to the routing process's nondeterministic nature, thereby collapsing the de
    
[^63]: 从拓扑学习到图生成：一个统一的视角

    From topology learning to graph generation: A unifying perspective

    [https://arxiv.org/abs/2609.02286](https://arxiv.org/abs/2609.02286)

    本综述提出统一框架，将图拓扑学习与图生成视为同一图数据生成过程的逆问题，从而连接了这两个长期平行发展的研究方向。

    

    从数据中学习图结构是一个基础性问题，涵盖广泛的信号处理和机器学习任务。尽管针对这一问题已有大量研究，但现有工作主要沿着两个平行的方向发展：第一个方向试图从支撑于图上的观测数据中推断单个图的拓扑结构，而第二个方向试图从观测到的图实例中学习生成分布，从而实现对新图的采样。本综述提出了一个统一的框架，通过将这两种建模方式视为图数据共同生成过程的逆问题来连接它们。我们回顾了该框架下的主要方法论，强调了它们之间的关系、优势和局限性，并指出了跨范式整合思想的机会。通过架起图拓扑学习与图生成之间的桥梁，本综述提供了更广泛的跨学科视角。

    arXiv:2609.02286v1 Announce Type: cross  Abstract: Learning graph structures from data is a fundamental problem that spans a wide range of signal processing and machine learning tasks. While significant effort has been made to tackle the problem, existing research has largely evolved along two parallel directions. The first seeks to infer the topology of an individual graph from observations supported on it, whereas the second seeks to learn a generative distribution from observed graph instances, enabling the sampling of new graphs. This review presents a unified framework that connects these formulations by viewing them as inverse problems of a common generation process for graph data. We review the major methodologies within this framework, highlight their relationships, strengths, and limitations, and identify opportunities for integrating ideas across paradigms. By bridging graph topology learning and graph generation, this review provides a broader cross-disciplinary perspective 
    
[^64]: 纠缠表示加剧机器遗忘中的附带损害

    Entangled Representations Amplify Collateral Damage in Unlearning

    [https://arxiv.org/abs/2609.02285](https://arxiv.org/abs/2609.02285)

    该研究首次通过受控实验验证了表示纠缠会加剧机器遗忘的附带损害——通过训练知识域解耦程度不同的语言模型套件，证明更解耦的模型在固定遗忘水平下保留成本可降低约4倍。

    

    可解释性研究中一个长期存在的直觉是，表示纠缠——即神经网络中不同知识域之间共享结构——会使机器遗忘变得更加困难。尽管这一直觉广为流传，但此前从未在受控实验中得到直接验证。我们提出了一种实现验证的方法：通过改造选择性梯度掩蔽（SGTM），我们在英文维基百科语料上训练了六个254M参数的语言模型套件，这些模型在生物学与非生物学知识之间具有不同等级的解耦程度。将该套件中的每个模型分别应用三种标准遗忘方法后，我们发现解耦程度更高的模型始终能实现更好的“保留-遗忘”权衡：在固定遗忘水平下，最解耦的模型在三种方法中的两种下保留成本约降低4倍，在第三种方法下降低1.3倍。由于我们的干预仅改变了模型本身，而不改变数据或遗忘算法，因……

    arXiv:2609.02285v1 Announce Type: cross  Abstract: A long-held intuition in interpretability research is that representational entanglement, the sharing of structure between knowledge domains in a neural network, makes unlearning harder. While the intuition is widespread, it has never been directly tested in a controlled experiment. We present a way to do so: by repurposing Selective Gradient Masking (SGTM), we train a suite of six 254M-parameter language models on English Wikipedia with graded levels of disentanglement between biology and non-biology knowledge. Applying three standard unlearning methods to every model in the suite, we find that more disentangled models consistently achieve better retain-forget trade-offs: at a fixed level of forgetting, the most disentangled models incur roughly $4\times$ lower retain cost under two of the three methods, and $1.3\times$ lower under the third. Because our intervention changes only the model, not the data or the unlearning algorithm, th
    
[^65]: 大型语言模型能否捕捉其训练数据中的多样性？

    Do Large Language Models Capture the Diversity in their Training Data?

    [https://arxiv.org/abs/2609.02275](https://arxiv.org/abs/2609.02275)

    该论文提出一种基于信息论的方法，通过比较模型生成输出与训练数据的条件熵，发现大语言模型（如OLMo、Pythia和GPT-Neo）生成内容的多样性系统性地低于其训练数据的多样性。

    

    大型语言模型被训练用于建模文本的条件分布，但它们是否能够捕捉训练数据中存在的合理输出的全部多样性，这一问题仍未得到充分理解。我们通过信息论的视角来研究这个问题，将模型生成输出的条件熵与相应训练数据的条件熵进行比较。给定成对的输入-输出样本，我们使用条件熵及其基于冯·诺依曼熵的矩阵类比方法来衡量超出条件输入所能解释的输出变异性，而无需同一提示的多个参考输出。在具有公开可用训练数据的大语言模型家族中，包括OLMo、Pythia和GPT-Neo，我们一致发现，在不同的模型规模、序列长度和解码策略下，模型生成的输出都表现出比其训练数据更低的条件熵。我们观察到类似的……（摘要截断）

    arXiv:2609.02275v1 Announce Type: cross  Abstract: Large language models are trained to model conditional distributions over text, yet it remains inadequately understood whether they capture the full diversity of plausible outputs present in their training data. We study this question through an information-theoretic lens by comparing the conditional entropy of model-generated outputs with that of the corresponding training data. Given paired input-output samples, we use conditional entropy and its matrix-based analogue based on von Neumann entropy to measure output variability beyond what is explained by the conditioning input, without requiring multiple reference outputs for the same prompt. Across LLM families with publicly available training data, including OLMo, Pythia, and GPT-Neo, we consistently find that model-generated outputs exhibit lower conditional entropy than their training data, across different model scales, sequence lengths, and decoding strategies. We observe a simi
    
[^66]: CAPTURE：在个性化大语言模型智能体中区分偏好漂移与记忆投毒

    CAPTURE: Disentangling Preference Drift from Memory Poisoning in Personalized LLM Agents

    [https://arxiv.org/abs/2609.02265](https://arxiv.org/abs/2609.02265)

    提出 CAPTURE 框架，通过神经微分方程信念追踪器、多时间尺度记忆账本、不确定性触发的澄清机制和反事实记忆审计，使个性化 LLM 智能体能够有效区分真实偏好漂移与对抗性记忆投毒攻击。

    

    个性化语言智能体使用持久记忆来随时间适应用户，但同样的机制也带来了攻击面。当新信息与已存储的偏好发生冲突时，智能体必须区分真实的偏好漂移与临时的上下文变化、歧义或对抗性的记忆投毒。我们将该问题形式化为基于潜在用户状态的连续时间部分可观测决策过程，并说明为什么仅基于新近性和来源的规则是不够的。CAPTURE 通过神经微分方程信念追踪器、多时间尺度记忆账本、由不确定性触发的澄清机制以及对所引用记忆的反事实审计来解决这一歧义。在来自 96 个用户的 480 个保留测试场景中，CAPTURE 取得了 71.5% 的胜率，相比之下，采用相同监督的基线为 69.3%，最强的启发式基线为 66.1%。它将固定策略下记忆投毒的成功率限制在 11.5%，同时接受了 83.5% 的（合法更新，原文在此截断）。

    arXiv:2609.02265v1 Announce Type: new  Abstract: Personalized language agents use persistent memory to adapt to users over time, but the same mechanism creates an attack surface. When new information conflicts with stored preferences, an agent must distinguish genuine preference drift from temporary context shifts, ambiguity, or adversarial memory poisoning. We formulate this problem as a continuous-time partially observable decision process over a latent user state and show why rules based only on recency and provenance are insufficient. CAPTURE addresses this ambiguity with a neural differential-equation belief tracker, a multi-timescale memory ledger, uncertainty-triggered clarification, and counterfactual auditing of cited memories. On 480 held-out episodes from 96 users, CAPTURE achieves a 71.5% win rate, compared with 69.3% for an identically supervised baseline and 66.1% for the strongest heuristic baseline. It limits fixed-policy poisoning success to 11.5% while accepting 83.5%
    
[^67]: 码本智能体：面向大语言模型多智能体系统的摊销式拓扑设计

    Codebook Agent: Amortized Topology Design for LLM Multi-Agent Systems

    [https://arxiv.org/abs/2609.02264](https://arxiv.org/abs/2609.02264)

    该论文揭示当前将多智能体系统拓扑设计视为条件图生成的范式存在根本性缺陷——有效拓扑种类极少、边数无法反映真实token成本、消息传递评分器无法区分共享配置的智能体——并提出基于码本的摊销式拓扑设计作为替代方案。

    

    针对每个查询适配大语言模型（LLM）多智能体系统的通信拓扑，可以同时提升准确性与效率，然而现有设计者将此问题视为条件图生成：由变分、自回归或扩散解码器在 N×N 的邻接空间中进行搜索，再由一个基于效用与结构成本（如边数）训练的图网络代理模型对采样候选进行排序。我们认为这种问题表述与问题本身并不匹配。实验表明：即使码本容量从 8 扩大到 64，通过奖励筛选后保留的拓扑也会坍缩为大约六种不同的图；边数与实际测量的 token 消耗呈负相关（Pearson r ≈ -0.4），因此对图进行稀疏化反而会使推理成本更高；此外，当智能体共享相同配置时——这是已发布基准测试的默认设置——基于智能体配置节点的消息传递评分器具有邻接不变性，因而根本无法对候选方案进行排序。

    arXiv:2609.02264v1 Announce Type: new  Abstract: Adapting the communication topology of an LLM multi-agent system to each query improves both accuracy and efficiency, yet current designers treat this as conditional graph generation: a variational, autoregressive, or diffusion decoder searches the $N \times N$ adjacency space, and a graph-network proxy trained on utility and a structural cost such as edge count ranks the sampled candidates. We argue that this formulation is misaligned with the problem. Empirically, topologies that survive a reward filter collapse to about six distinct graphs even when the codebook capacity grows from 8 to 64; edge count is negatively correlated with measured token consumption (Pearson $r \approx -0.4$), so sparsifying the graph makes inference more expensive; and a message-passing scorer over agent-profile nodes is adjacency-invariant whenever agents share a profile---the default configuration of published benchmarks---so it cannot rank candidates at al
    
[^68]: RideSkill：一种基于大语言模型驱动自动进化的泛化拼车分层算法

    RideSkill: A Hierarchical Algorithm for Generalized Ride Sharing with LLM-Driven Automatic Evolution

    [https://arxiv.org/abs/2609.02250](https://arxiv.org/abs/2609.02250)

    该论文提出RideSkill，一种由大语言模型驱动自动进化的分层算法，用于解决泛化拼车问题，克服了传统多智能体强化学习方法在泛化性、可迁移性和大规模训练方面的局限。

    

    拼车允许具有不同起讫点（OD对）的多名乘客共享同一辆车辆，是一个具有挑战性的运营问题，因为它需要在不确定且多变的情况下，高效地将不同OD对的订单捆绑并分配给车辆。尽管多智能体强化学习（MARL）解决方案已取得了有前景的性能，但它们存在泛化能力有限（难以适应不同的环境场景）、可迁移性低（难以适应不同的平台目标）以及在大规模系统中训练困难（如维度灾难）等问题。最近，受大语言模型（LLM）规模化发展的启发，一些工作将LLM引入网约车系统，要么直接将LLM用作决策智能体，要么利用LLM进行自动算法设计。然而，这些方法均不支持车辆共享，这使问题变得更加复杂。

    arXiv:2609.02250v1 Announce Type: cross  Abstract: Ride-sharing, which allows multiple passengers with different origin-destination (OD) pairs to share a single vehicle, is a challenging operational problem, as it requires orders with different OD pairs to be efficiently bundled and assigned to vehicles under uncertain and varying scenarios. Although multi-agent reinforcement learning (MARL) solutions have achieved promising performance, they suffer from limited generalization (adapting to different environmental scenarios), low transferability (adapting to different platform objectives), and training difficulties in large-scale systems, such as the curse of dimensionality. Recently, motivated by the scaling of large language models (LLMs), several works have incorporated LLMs into ride-hailing systems, either by employing LLMs directly as decision-making agents or using them for automatic algorithm design. However, none of these approaches support vehicle sharing, which complicates th
    
[^69]: LLM作为裁判并非神谕：为什么自我改进的智能体需要确定性护栏

    LLM-as-a-Judge Is Not an Oracle: Why Self-Improving Agents Need Deterministic Guardrails

    [https://arxiv.org/abs/2609.02246](https://arxiv.org/abs/2609.02246)

    本文提出LLM裁判不应作为自我改进智能体的最终评估权威，而应降级为顾问，由裁判无法推翻的确定性验证层把关所有变更，并以生产环境中归纳出的四类十一种评估失效模式（包括读取缓存答案的奖励破解行为）支撑这一立场。

    

    自我改进的智能体流水线的核心存在一个问题。优化器重写提示词以获得更高的分数，而分数来自一个本身也是LLM的裁判。这个裁判对系统是否在改进拥有最终决定权，而我们的观点是它并未赢得这一地位。裁判应当从神谕降级为顾问：它的裁决只是众多输入中的一个，而每一次变更都应由裁判无法推翻的确定性验证层来把关。我们通过构建并实际运行替代方案得出了这一立场。在生产环境中跨合同分析、合规审查和代码质量领域运行自主提示词优化循环数月的过程中，我们记录了评估信号失效的十一种方式，可归为四类：裁判偏见、测试框架与指标失效、真值错误以及奖励破解。智能体通过读取环境中的缓存答案键获得了满分，100%的通过率掩盖了……

    arXiv:2609.02246v1 Announce Type: new  Abstract: Self-improving agent pipelines have a problem at their center. An optimizer rewrites prompts to score higher, and the score comes from a judge that is itself an LLM. That judge has the last word on whether the system is getting better, and our position is that it has not earned it. The judge should be demoted from oracle to advisor: its verdict becomes one input among several, and every change is gated instead by a deterministic verification layer the judge cannot override. We reached this position by building the alternative and running it. Over months of running autonomous prompt-optimization loops in production across contract analysis, compliance review, and code quality, we cataloged eleven ways the evaluation signal failed, in four classes: judge bias, harness and metric failures, ground-truth errors, and reward hacking. Agents achieved perfect scores by reading cached answer keys from their environment, a 100% pass rate concealing
    
[^70]: 异构环境下的相似性感知个性化联邦学习

    Similarity-Aware Personalized Federated Learning in Heterogeneous Environments

    [https://arxiv.org/abs/2609.02241](https://arxiv.org/abs/2609.02241)

    提出SAPE-FL框架，通过将客户端模型同时锚定到全局模型和相似性加权的同伴平均模型，并利用基于模型与输出相似性的动态个性化正则化，在异构数据分布下自适应平衡全局知识迁移与同伴协作、过滤不相似客户端。

    

    联邦学习（FL）允许多个分散的客户端在保护数据隐私的前提下协作训练模型。然而，客户端之间的数据分布不匹配往往导致全局模型泛化能力差、本地客户端性能下降。在这种情况下，某些仅用本地数据训练的本地模型可能表现得比全局模型更好，从而使协作式联邦学习的优势失去意义。为解决这一问题，我们提出了SAPE-FL（相似性感知个性化联邦学习），这是一种新颖的个性化框架，它将每个客户端的模型同时锚定到全局模型和一个基于相似性加权的同伴平均模型上。通过引入基于模型相似性和输出相似性的动态、客户端特定正则化，SAPE-FL能够自适应地平衡全局知识迁移与同伴协作，同时过滤掉不相似的客户端。这种双重锚定机制缓解了负向（知识迁移）……（摘要在此处截断）

    arXiv:2609.02241v1 Announce Type: new  Abstract: Federated Learning (FL) allows decentralized clients to train models collaboratively while preserving data privacy. However, distribution mismatch across clients often leads to poor global generalization and degraded local client-level performance. In such scenarios, some of the clients with their local models trained solely on local data may perform better than the globally learnt model, thus nullifying the benefits of collaborative federated learning. To address this, we propose SAPE-FL (Similarity-Aware Personalized Federated Learning), a novel personalization framework that anchors each client's model to both the global model and a similarity-weighted peer averaged model. By incorporating dynamic, client-specific regularization based on both model similarity and output similarity, SAPE-FL adaptively balances global knowledge transfer and peer collaboration while filtering out dissimilar clients. This dual anchoring mitigates negative
    
[^71]: 面向长时程离线目标条件强化学习的递归价值学习

    Recursive Value Learning for Long-Horizon Offline Goal-Conditioned RL

    [https://arxiv.org/abs/2609.02237](https://arxiv.org/abs/2609.02237)

    DCRL通过将轨迹段递归分解为平衡二叉树并从叶到根训练价值，将自举深度从线性降至对数级，从而有效提升长时程离线目标条件强化学习的性能。

    

    将离线目标条件强化学习（GCRL）扩展到长时程任务是困难的，其原因在于：（1）长程价值学习依赖于可能仍然不准确的短程价值估计；（2）基于最大化的价值备份会通过反复传播放大过高估计。我们提出DCRL（分治强化学习，Divide-and-Conquer RL），该方法递归地将每条轨迹段分解为平衡二叉树，并从叶节点到根节点进行价值训练。因此，每个父节点仅在其子节点更新之后才被更新，使用的是对观测路径的精确因式分解，而非在有噪声的备选方案中进行选择。由于该目标沿演示路径学习价值，而这些演示路径不一定是最优的，DCRL同时在轨迹之间传播价值以发现更短的路径。得益于平衡二叉树结构，DCRL将最坏情况下的自举深度从线性降低到对数级，并且这种更短的依赖结构在实证上与之相对应……

    arXiv:2609.02237v1 Announce Type: new  Abstract: Scaling offline goal-conditioned reinforcement learning (GCRL) to long-horizon tasks is difficult because (1) long-range value learning depends on shorter-range estimates that may still be inaccurate, and (2) max-based value backups can amplify overestimation through repeated propagation. We propose DCRL (Divide-and-Conquer RL), which recursively decomposes each trajectory segment into a balanced binary tree and trains the values from leaves to root. Each parent is therefore updated only after its children, using an exact factorization of the observed route rather than selecting among noisy alternatives. Since this objective learns values along demonstrated routes that are not necessarily optimal, DCRL jointly propagates values across trajectories to discover shorter routes. Thanks to the balanced binary tree, DCRL reduces worst-case bootstrap depth from linear to logarithmic, and this shorter dependency structure empirically corresponds
    
[^72]: 面向资源受限空间机器人的结合关键性分析的硬件加速实例分割

    Hardware-Accelerated Instance Segmentation for Resource-Constrained Space Robotics with Criticality Analysis

    [https://arxiv.org/abs/2609.02219](https://arxiv.org/abs/2609.02219)

    该论文提出面向资源受限月球机器人的硬件加速实例分割框架，通过AVIS无标签校准策略与DPU上经过架构优化的YOLO分割模型部署，在低光照、有限算力和辐射故障三重约束下实现有界延迟的实时感知。

    

    自主月球任务需要在三个相互耦合的约束条件下实现实时感知：极端低光照条件、有限的星载计算能力，以及可能静默破坏推理结果的辐射致硬件故障。我们提出了一种面向部署的实例分割框架，适用于资源受限的月球机器人，该框架在严格的计算约束下同时解决量化校准与系统级故障暴露问题。首先，我们提出了激活方差信息采样（AVIS），这是一种无标签校准策略，基于激活方差统计量确定性地选择校准样本。其次，我们在深度学习处理器单元（DPU）上部署了基于YOLO的分割模型，并进行了架构修改以减少CPU回退路径，实现了静态编译执行，在低光照条件下提供有界延迟。我们进一步引入了软件级关键性分析来估计……

    arXiv:2609.02219v1 Announce Type: cross  Abstract: Autonomous lunar missions require real-time per- ception under three coupled constraints: extreme low-light conditions, limited onboard compute, and radiation-induced hardware faults that can silently corrupt inference. We present a deployment-oriented instance segmentation framework for resource-constrained lunar robotics that jointly addresses quan- tization calibration and system-level fault exposure under strict compute constraints. First, we introduce Activation Variance Informative Sampling (AVIS), a label-free calibration strategy that deterministically selects calibration samples based on activation variance statistics. Second, we deploy a YOLO-based segmentation model on a Deep Learning Processor Unit (DPU) with architectural modifications that reduce CPU fallback paths and enable statically compiled execution with bounded latency in low-lighting conditions. We further introduce a software-level criticality analysis to estimat
    
[^73]: 原型引导的稀疏文献知识迁移用于电解液添加剂发现

    Prototype-guided transfer of sparse literature knowledge for electrolyte additive discovery

    [https://arxiv.org/abs/2609.02209](https://arxiv.org/abs/2609.02209)

    该研究提出了文献驱动的ProtoMI框架，通过图对比学习从稀疏的文献报道电解液添加剂中提取化学可解释的结构原型，并利用原型引导的半监督对比学习将其迁移到庞大的未标注化学空间，从而高效实现对电解液添加剂候选分子的优先排序。

    

    电解液添加剂的发现仍然具有挑战性，因为经过实验验证的分子非常稀少，而可获取的化学空间庞大且大部分未标注。这一挑战在锂离子电池中被进一步放大，因为添加剂的性能来源于相互耦合的界面反应，而非单一的分子性质。在此，我们开发了一种原型引导的分子智能框架ProtoMI，这是一个由文献驱动的框架，它从已报道的电解液添加剂中学习可迁移的结构先验，并利用这些先验在未标注的化学空间中对候选分子进行优先排序。针对含硼添加剂，ProtoMI将126个文献报道的分子与179,977个未标注的候选分子相结合。图对比学习从已报道的添加剂中识别出七个具有化学可解释性的原型，原型引导的半监督对比学习在源域与目标域分布匹配的条件下将这些原型适配到候选空间中。

    arXiv:2609.02209v1 Announce Type: cross  Abstract: Electrolyte additive discovery remains challenging because experimentally validated molecules are sparse, whereas accessible chemical spaces are vast and largely unlabeled. This challenge is amplified in lithium-ion batteries, where additive performance arises from coupled interfacial reactions rather than a single molecular property. Here, we develop a prototype-guided molecular intelligence, ProtoMI, a literature-driven framework that learns transferable structural priors from reported electrolyte additives and uses them to prioritize candidates in unlabeled chemical space. For boron-containing additives, ProtoMI combines 126 literature-reported molecules with 179,977 unlabeled candidates. Graph contrastive learning identifies seven chemically interpretable prototypes from the reported additives, and prototype guided semi-supervised contrastive learning adapts these prototypes to the candidate space under source-target distribution m
    
[^74]: SMart：一个多源多阶段的时间序列表示迁移框架

    SMart: A Multi-source Multi-phase Time Series Representation Transfer Framework

    [https://arxiv.org/abs/2609.02203](https://arxiv.org/abs/2609.02203)

    本文提出SMart框架，通过多阶段递归图恢复任务和源数据集选择器这两种新颖机制，实现了利用多源数据进行多阶段时间序列表示迁移的改进。

    

    时间序列表示学习（TSRL）近年来吸引了越来越多的研究关注。TSRL领域近期有两个探索方向：i) 利用基于Transformer的框架来学习时间序列；ii) 不仅使用目标数据集，还从其他数据集借用时间序列数据来促进表示迁移。虽然这两种探索已被证明是有效的，但(i)中的自监督时间序列恢复任务和(ii)中使用的单源数据集在技术上较为简单，因此可以通过新的思路加以改进。在本工作中，我们提出了一种新的TSRL框架，即多源多阶段时间序列表示迁移（SMart），它包含两种新颖的机制来解决上述缺陷：1) 一种多阶段递归图恢复任务，提供三种可选模式，用于引导编码器将时间序列的动态特性嵌入到时间序列表示中；2) 一个源数据集选择器，用于……

    arXiv:2609.02203v1 Announce Type: cross  Abstract: Time series representation learning (TSRL) has attracted growing research interests in recent years. Two recent explorations in TSRL are: i) exploiting a transformer-based framework to learn time series; ii) instead of using only the targeted dataset, borrowing time series from other datasets to to facilitate representation transfer. While these two explorations are shown effective, the self-supervised time series recovery task in (i) and the single-source dataset used in (ii) are technically simple and thus can be enhanced with new ideas. In this work, we propose a new TSRL framework, namely multi-source multi-phase time series representation transfer (SMart), which has two novel mechanisms to address the aforementioned deficiencies: 1) a multi-phase recurrence plots recovery task, in three alternative modes, for guiding the encoder to embed time series dynamics into the time series representation; and 2) a source dataset selector to 
    
[^75]: 李群流形上的薛定谔桥用于概率性内在生成

    Schr\"odinger Bridges on Lie Group Manifolds for Probabilistic Intrinsic Generation

    [https://arxiv.org/abs/2609.02196](https://arxiv.org/abs/2609.02196)

    该论文将薛定谔桥推广到李群流形上，实现了在弯曲几何空间中直接进行概率生成建模，允许仅约束部分可观测端点变量，并针对紧致阿贝尔群与非阿贝尔群分别提出了WKBC和RCCBM两种计算方法。

    

    直接在几何流形上进行生成建模，可以避免将非欧几里得数据展平、反复向环境空间投影以及欧几里得表示中坐标不一致所带来的误差。薛定谔桥为一个在指定端点分布之间进行熵正则化输运的概率生成框架提供了理论基础。我们研究了李群流形上动力学系统的薛定谔桥问题，其状态为 X_t = (g_t, ξ_t) ∈ G × 𝔤，允许端点观测仅约束那些实际被测量的变量。特别地，熵投影确定了未观测到的端点速度的条件分布律。针对相同的观测端点桥问题，我们发展了两种计算实现方法：缠绕核桥校准使用紧致阿贝尔群上的显式周期化动力学核，而互惠条件控制桥匹配（RCCBM）则处理紧致非阿贝尔群的情形（原文摘要在此处被截断）。

    arXiv:2609.02196v1 Announce Type: cross  Abstract: Generative modeling directly on geometric manifolds can avoid errors introduced by flattening non-Euclidean data, repeated ambient projection, and coordinate inconsistency in Euclidean representations. Schrodinger bridges provide a probabilistic generative framework for entropy-regularized transport between prescribed endpoint distributions. We study Schrodinger bridges for kinetic dynamics on Lie group manifolds with state X_t = (g_t, xi_t) in G x g, allowing endpoint observations to constrain only the variables that are actually measured. In particular, the entropy projection determines the conditional law of the unobserved endpoint velocities.   For the same observed endpoint bridge, we develop two computational realizations: Wrapped-Kernel Bridge Calibration (WKBC) uses an explicit periodized kinetic kernel on compact Abelian groups, whereas Reciprocal Conditional-Control Bridge Matching (RCCBM) handles compact non-Abelian groups t
    
[^76]: 基于神经算子与因果注意力学习材料的本构行为：塑性与损伤案例研究

    Learning the Constitutive Behavior of Materials via Neural Operators and Causal Attention: Case Studies in Plasticity and Damage

    [https://arxiv.org/abs/2609.02194](https://arxiv.org/abs/2609.02194)

    提出一种基于神经算子与因果注意力的数据驱动本构建模框架，将材料视为从应变历史到应力响应的算子映射，无需内状态变量即可直接从应变-应力数据学习塑性与损伤本构行为。

    

    经典的路径依赖非弹性材料本构建模依赖于内状态变量，其演化方程必须基于领域知识进行假设，并借助实验数据进行校准。然而，在许多实际场景中，相关内变量通常在实验中无法测量，本构响应必须完全从测得的应变-应力数据中推断，而无需对材料内部状态的任何先验知识。我们提出了一种基于材料算子概念的数据驱动本构建模框架，该框架将变形材料视为从其完整应变历史到相应应力响应的泛函映射。与传统的自回归或循环神经网络公式不同，该模型直接在全加载路径上以函数到函数映射的方式进行训练，通过单次并行前向传播即可预测完整的应力轨迹。时间…

    arXiv:2609.02194v1 Announce Type: new  Abstract: Classical constitutive modeling of path-dependent inelastic materials relies on internal state variables whose evolution equations must be postulated based on domain knowledge and calibrated against experimental data. However, in many practical settings, the relevant internal variables are typically not measurable in experiments, and the constitutive response must be inferred entirely from measured strain-stress data without any prior knowledge of the material's internal state. We propose a data-driven constitutive modeling framework based on the concept of a material operator, which treats a deforming material as a functional mapping from its entire strain history to the corresponding stress response. In contrast to traditional autoregressive or recurrent formulations, the model is trained directly on full loading paths as function-to-function mappings, predicting complete stress trajectories in a single parallel forward pass. Temporal 
    
[^77]: 量子MeanFlow：在NISQ硬件上的单次生成式采样

    Quantum MeanFlow: single-shot generative sampling on NISQ hardware

    [https://arxiv.org/abs/2609.02186](https://arxiv.org/abs/2609.02186)

    该论文提出量子MeanFlow（QMF），通过学习平均速度场将量子流匹配所需的 多步顺序求解简化为单步生成采样，从而规避了NISQ量子硬件上高输入/输出成本带来的顺序电路提交问题。

    

    量子生成模型为探索量子计算能否增强生成式机器学习提供了一个有前景的框架。流匹配（Flow Matching）是一种生成方法，它通过一个学习到的速度场，将简单的已知分布输运至目标数据分布从而生成样本。其量子对应方法——量子流匹配（QFM）最近被提出，但与经典方法一样，它需要在推理阶段对常微分方程进行多时间步的积分求解。由于每一步都依赖上一步的输出，电路提交必须按顺序进行，而量子计算机的输入/输出成本很高，这成为一个显著缺点。为缓解该问题，我们提出了量子MeanFlow（QMF），即MeanFlow框架的量子类比，它能够实现单步样本生成。QFM在每个时间步学习瞬时速度场，而QMF学习的是平均速度场。

    arXiv:2609.02186v1 Announce Type: cross  Abstract: Quantum generative models offer a promising framework for exploring whether quantum computation can enhance generative machine learning. Flow matching is a generative method in which samples are generated by transporting a simple, known distribution to the target data distribution with a learned velocity field. Its quantum counterpart, known as quantum flow matching (QFM), was introduced recently, and, like its classical counterpart, requires integrating an ordinary differential equation over many time steps during inference. As each step requires the output from the previous step, the circuit submission is sequential and a drawback on quantum computers as they have high input/output costs. To alleviate this problem, we introduce Quantum MeanFlow (QMF), the quantum analogue of the MeanFlow formulation, which allows single-step sample generation. While the QFM learns an instantaneous velocity field at each time step, QMF learns the aver
    
[^78]: WeaveMark：基于编码载荷扩展的鲁棒且可扩展的多比特大语言模型水印

    WeaveMark: Robust and Scalable Multi-bit LLM Watermarking via Coded Payload Spreading

    [https://arxiv.org/abs/2609.02177](https://arxiv.org/abs/2609.02177)

    WeaveMark通过每令牌多比特载荷扩展、软判决纠错码和无偏多层重加权三大技术，突破了多比特LLM水印中载荷容量、提取准确率与文本质量之间的固有权衡，并引入零比特层实现可靠的水印存在性检测。

    

    面向大语言模型（LLM）的多比特水印技术通过在生成的文本中嵌入可识别用户身份的消息，实现内容来源追踪。现有方法在提取准确率、文本质量和载荷容量之间存在根本性的权衡。我们提出了WeaveMark，一种基于编码载荷扩展的鲁棒且可扩展的多比特LLM水印方案。WeaveMark通过每令牌多比特扩展提升载荷容量、通过软判决纠错码提高提取准确率、并通过无偏多层重加权保持文本质量，从而推动了这一权衡的前沿边界。此外，它还引入了专用的零比特层，用于可靠的水印存在性检测。实验显示出显著的性能提升，尤其体现在长消息和经过编辑的文本上。WeaveMark在200个令牌下对32位消息实现了89.8%的匹配率，而BiMark仅为20.8%。在200个令牌的16位消息遭受10%替换攻击时（摘要在此处截断）

    arXiv:2609.02177v1 Announce Type: cross  Abstract: Multi-bit watermarking for large language models (LLMs) enables content source tracing by embedding user-identifiable messages into generated text. Existing methods face a fundamental trade-off among extraction accuracy, text quality, and payload capacity. We propose WeaveMark, a robust and scalable multi-bit LLM watermarking scheme based on coded payload spreading. WeaveMark shifts this trade-off frontier by improving payload capacity through multi-bit-per-token spreading, improving extraction accuracy through soft-decision error-correcting code, and preserving text quality through unbiased multilayer reweighting. It further introduces dedicated zero-bit layers for reliable watermark presence detection. Experiments show large gains, especially for long messages and edited text. WeaveMark achieves 89.8% match rate for 32-bit messages at 200 tokens, compared with 20.8% for BiMark. Under 10% substitution attacks on 16-bit messages at 200
    
[^79]: 广度胜过深度：通过面向广度的后缀搜索改进基于GCG的越狱优化

    Breadth Beats Depth: Improving GCG-Based Jailbreak Optimization with Breadth-Oriented Suffix Search

    [https://arxiv.org/abs/2609.02172](https://arxiv.org/abs/2609.02172)

    本文提出即插即用框架BOSS，通过尾部聚焦对抗损失和面向广度的后缀搜索策略改进基于GCG的越狱攻击优化，在提升攻击成功率的同时降低了优化时间。

    

    基于优化的越狱攻击（如贪心坐标梯度法GCG）通过在白盒源模型上优化对抗性后缀，实现了较强的有效性和可迁移性。然而，现有的基于GCG的方法依赖于平均对抗损失和深度贪心搜索，这可能过度强调容易越狱的行为，而忽视后缀空间中有前景的区域。我们提出了BOSS，这是一个即插即用的框架，通过面向广度的后缀搜索来改进基于GCG的越狱优化。BOSS使用尾部聚焦对抗损失、标准源损失和行为覆盖率来选择终端后缀，然后探索多条短轨迹，并有选择地延续有前景的后缀。在公开基准上的实验表明，BOSS在多种基于GCG的方法上提升了攻击成功率，同时缩短了优化时间。

    arXiv:2609.02172v1 Announce Type: new  Abstract: Optimization-based jailbreak attacks such as Greedy Coordinate Gradient (GCG) achieve strong effectiveness and transferability by optimizing adversarial suffixes on white-box source models. However, existing GCG-based methods rely on averaged adversarial loss and deep greedy search, which can over-emphasize easy-to-jailbreak behaviors and overlook promising regions of the suffix space. We propose BOSS, a plug-and-play framework that improves GCG-based jailbreak optimization through breadth-oriented suffix search. BOSS uses Tail-Focused Adversarial Loss (TFAL), standard source loss, and behavior coverage to select terminal suffixes, then explores multiple short trajectories and selectively continues promising suffixes. Experiments on public benchmarks show that BOSS improves attack success rates across multiple GCG-based methods while reducing optimization time.
    
[^80]: DMRL：面向广告推荐技能优化的文档中介强化学习

    DMRL: Document-Mediated Reinforcement Learning for Skill Optimization in Advertising Recommendation

    [https://arxiv.org/abs/2609.02170](https://arxiv.org/abs/2609.02170)

    本文提出DMRL框架，将广告推荐中大语言模型技能文档的优化建模为上层智能体的结构化编辑动作序列，借助冻结的下层任务智能体A/B测试反馈和双相对策略优化（DRPO）解决信用分配问题，实现技能文档的自我演化优化。

    

    广告推荐需要在平衡商业回报与用户体验的同时，持续调优复杂的系统参数。近期工作引入了配备技能文档的大语言模型（LLM）来辅助这一劳动密集型过程，但技能优化在很大程度上仍由提示词驱动，缺乏将奖励归因到具体文档编辑的原则性机制。为解决这一局限，我们提出了文档中介强化学习，这是一个将技能文档优化建模为一系列结构化编辑动作的技能自我演化框架。在DMRL中，上层智能体执行受控的文档编辑，而冻结参数的下层任务智能体通过A/B测试来评估这些编辑的效果。为解决信用分配与长期结果问题，我们引入了两个关键组件：（1）双相对策略优化（DRPO），一种用于鲁棒且风险感知优势估计的训练后策略优化方法……（摘要截断于此）

    arXiv:2609.02170v1 Announce Type: new  Abstract: Advertising recommendation requires continuously tuning complex system parameters while balancing commercial returns and user experience. Recent work has introduced large language models (LLMs) with skill documents to assist this labor-intensive process, but skill optimization remains largely prompt-driven, lacking a principled mechanism to attribute rewards to specific document edits. To address this limitation, we propose Document-Mediated Reinforcement Learning (DMRL), a skill self-evolution framework that models skill document optimization as a sequence of structured editing actions. In DMRL, an upper-level agent performs controlled document edits, while a frozen lower-level task agent evaluates their effects through A/B testing. To address credit assignment and long-term outcomes, we introduce two key components: (1) Dual-Relative Policy Optimization (DRPO), a post-training policy optimization method for robust and risk-aware advant
    
[^81]: GenCAR：面向分布外推荐的生成式反事实对齐与风险可控选择

    GenCAR: Generative Counterfactual Alignment with Risk-Controlled Selection for Out-of-Distribution Recommendation

    [https://arxiv.org/abs/2609.02162](https://arxiv.org/abs/2609.02162)

    提出GenCAR框架，将分布外推荐形式化为α-有效反事实推荐（α-VCR）问题，通过基于偏好的反事实监督与基于保形p值的Benjamini-Hochberg集合选择，在控制代理标签错误发现率（FDR）的同时提升推荐效用。

    

    在分布偏移下提供有用的推荐，对于分布外（OOD）推荐中平衡效用与风险至关重要。然而，现有的大多数OOD方法在改进排序或构建反事实候选集时，并未对所服务集合的代理标签错误发现率（FDR）加以控制。在本工作中，我们将OOD服务形式化为α-有效反事实推荐（α-VCR）问题，以便在控制代理标签FDR的同时，保留从反事实监督中学习到的候选支持集，并提出GenCAR，该方法将基于偏好的反事实监督与经校准的集合选择相结合。具体而言，GenCAR在干预环境因素的同时固定稳定偏好表示，通过偏好锚点和信任半径过滤来约束离线大语言模型（LLM）的提议，并使用保形p值进行Benjamini-Hochberg选择。我们从理论上对（摘要在此处截断）……

    arXiv:2609.02162v1 Announce Type: cross  Abstract: Serving useful recommendations under distribution shift is crucial for balancing utility and risk in out-of-distribution (OOD) recommendation. However, most existing OOD methods improve ranking or construct counterfactual candidates without controlling the proxy-label false discovery rate (FDR) of the served set. In this work, we formulate OOD serving as the $\alpha$-Valid Counterfactual Recommendation ($\alpha$-VCR) problem to retain candidate support learned from counterfactual supervision while controlling proxy-label FDR, and propose GenCAR, which couples preference-grounded counterfactual supervision with calibrated set selection. In particular, GenCAR fixes the stable-preference representation while intervening on the environmental factor, grounds offline large language model proposals through preference anchors and trust-radius filtering, and uses conformal $p$-values for Benjamini--Hochberg selection. We theoretically bound con
    
[^82]: GeoSPRINT：面向扩散模型轨迹推理的几何冗余感知步数剪枝

    GeoSPRINT: Geometric Redundancy-Aware Step Pruning for Inference in Diffusion Trajectories

    [https://arxiv.org/abs/2609.02160](https://arxiv.org/abs/2609.02160)

    GeoSPRINT是一个免训练的扩散模型推理加速框架，通过潜在空间的超平面性测试（QR分解实现）检测去噪轨迹中几何冗余的步数，并据此构建非均匀采样调度，在高曲率区域分配更多步数，从而在保证样本质量的同时大幅减少采样步数。

    

    扩散模型能够生成高质量的样本，但推理阶段计算成本高昂，因为采样过程需要大量连续的神经函数评估（NFEs）。现有的加速方法要么使用固定的跳步调度策略，要么基于局部数值误差自适应调整步长，要么需要额外的训练。我们提出了GeoSPRINT（Geometric Step Pruning for Inference in Trajectories，面向轨迹推理的几何步剪枝），这是一个免训练框架，能够基于去噪轨迹的几何结构构建非均匀采样调度。GeoSPRINT通过在潜在空间中进行超平面性测试来检测几何上冗余的步（该测试通过QR分解高效实现），并将由此得到的冗余特征转换为采样调度，为轨迹的高曲率区域分配更多的采样步数。此外，我们还引入了轨迹投影分数 α_traj，这是一种残差方差度量，用于量化轨迹的直线程度……

    arXiv:2609.02160v1 Announce Type: cross  Abstract: Diffusion models achieve high sample quality but remain expensive at inference time because sampling requires many sequential neural function evaluations (NFEs). Existing acceleration methods either use fixed step-skipping schedules, adapt step sizes based on local numerical error, or require additional training. We introduce GeoSPRINT (Geometric Step Pruning for Inference in Trajectories), a training-free framework for constructing non-uniform sampling schedules from the geometry of denoising trajectories. GeoSPRINT detects geometrically redundant steps using a hyperplanarity test in latent space, implemented efficiently via QR factorization, and converts the resulting redundancy profile into a sampling schedule that allocates more steps to high-curvature regions of the trajectory. In addition, we introduce the trajectory projection score $\alpha_{\mathrm{traj}}$, a residual-variance metric that quantifies trajectory straightness and 
    
[^83]: 保持几何结构的随机投影精确极限：高斯模型中的距离恢复、最近邻排序与协方差形状

    Exact Limits of Random Projections for Preserving Geometry: Distance Recovery, Nearest-Neighbor Rankings, and Covariance Shape in Gaussian Models

    [https://arxiv.org/abs/2609.02155](https://arxiv.org/abs/2609.02155)

    该论文揭示Johnson-Lindenstrauss引理在高维中可能对保留的几何结构毫无信息量，并通过最优解码器理论精确刻画了高斯模型下从随机投影中恢复距离特征、最近邻排序与协方差形状的根本极限。

    

    Johnson-Lindenstrauss (JL) 引理保证将 $n$ 个点随机投影到 $m=O(\varepsilon^{-2}\log n)$ 维时，能以高概率在相对误差 $\varepsilon$ 内保持成对平方距离，且该维度阶数是渐近最优的。然而，在高维情形下，距离会集中在一个基准值附近，而关键的几何信息却蕴含在更小的波动之中。我们由此证明，JL 界对于所保留的几何结构可能是无信息的：即使一个独立的高斯替换点云与原始数据完全无关，其对应的替换映射也能满足该界。随后，我们研究任意解码器从线性草图中恢复平方距离 $D$ 的特征 $f(D)$ 的能力。在平方误差损失下，最优解码器为条件期望，因此恢复过程定义了一个线性算子，其奇异值量化了特征的可恢复性。对于各向同性高斯数据（$\Sigma=$……

    arXiv:2609.02155v1 Announce Type: new  Abstract: The Johnson-Lindenstrauss (JL) lemma guarantees that a random projection of $n$ points to   $m=O(\varepsilon^{-2}\log n)$ dimensions preserves pairwise squared distances within relative   error $\varepsilon$ with high probability, and this dimension order is asymptotically   optimal. In high dimensions, however, distances concentrate around a baseline while key   geometric information lies in much smaller fluctuations. We show that the JL bound can   therefore be uninformative about retained geometry: an independent Gaussian replacement map   can satisfy it even though the replacement cloud is independent of the original data.   We then ask how well any decoder can recover a feature $f(D)$ of a squared distance $D$ from   a linear sketch. Under squared-error loss, the optimal decoder is conditional expectation, so   recovery defines a linear operator whose singular values quantify feature recovery. For   isotropic Gaussian data ($\Sigma=
    
[^84]: 在线非单调DR-次模最大化：匹配离线0.401近似比

    Online Non-Monotone DR-Submodular Maximization Matching the Offline $0.401$ Factor

    [https://arxiv.org/abs/2609.02145](https://arxiv.org/abs/2609.02145)

    该论文首次在对抗性在线设置下实现了非单调DR-次模最大化与非离线算法相同的0.401最优近似比，通过用加权在线学习器替代离线箱约束步骤并结合精确非对称平衡定理，在决策后全信息值预言机模型下达到次线性近似遗憾。

    

    我们研究在 $d$ 维单位立方体的紧凸下闭子集上，非负、非单调DR-次模函数的在线最大化问题。在相应的元可解性假设下，目前已知最好的构造性离线近似比为 $0.401$，而可比的对抗性在线保证一直停留在 $1/e$。我们证明该 $0.401$ 近似比同样可以在线实现。在决策后全信息值预言机模型中，当预言机反馈条件无偏且有界时，我们的算法以次线性近似遗憾达到了 $0.401$ 的近似比。该在线算法并不在变化的目标函数上运行离线构造，而是用加权在线学习器替代离线算法中依赖于目标的箱约束步骤，以累积方式控制所需的残差项。一个精确的非对称平衡定理使得离线系数在对抗性变化下仍得以保持。直接实现（原文摘要在此处截断）……

    arXiv:2609.02145v1 Announce Type: cross  Abstract: We study online maximization of nonnegative, non-monotone DR-submodular functions over compact convex down-closed subsets of the $d$-dimensional unit cube. The best known constructive offline approximation factor is $0.401$ under the corresponding meta-solvability assumptions, whereas comparable adversarial online guarantees had remained at $1/e$. We show that this factor is also achievable online. In the post-decision full-information value-oracle model, our algorithm attains factor $0.401$ with sublinear approximate regret when oracle feedback is conditionally unbiased and bounded.   The online algorithm does not run the offline construction on a changing objective. Instead, it replaces the offline objective-dependent box step by a weighted online learner that controls the required residual terms cumulatively. An exact asymmetric balance theorem preserves the offline coefficients despite adversarial variation. The direct implementati
    
[^85]: 对数外衣下的幂律：论基于图的向量搜索的可扩展性

    A Power Law in Logarithm's Clothing: On the Scalability of Graph-Based Vector Search

    [https://arxiv.org/abs/2609.02143](https://arxiv.org/abs/2609.02143)

    该论文通过跨数据集规模的实测推翻了“搜索成本随数据规模呈多重对数增长”的通行说法，发现当数据规模相对内在维度较小时，基于图的向量搜索成本实际遵循次线性幂律（约按 $N^c$、$0<c<1$ 增长），只有规模足够大时才收敛到理论预言的对数式增长。

    

    大多数向量数据库依赖基于图的索引（尤其是 HNSW 和 Vamana）来进行近似最近邻搜索。随着嵌入模型的广泛采用，这些数据库存储的数据集迅速增长。在固定精度下，搜索成本如何随数据集规模扩展？流行的答案是多重对数增长。然而，这一论断仅在特殊条件下被证明过，对于实践中使用的索引则是未经证明的断言。它也基本未经检验：标准基准测试只在一个数据集规模下测量成本，而不是跨规模测量。我们对这一论断进行了检验。答案取决于规模本身。当数据集规模 $N$ 相对于数据的内在维度较小时，搜索成本按 $N^c$ 增长（其中 $c$ 是满足 $0<c<1$ 的常数）。我们将这种扩展规律称为次线性幂律。一旦 $N$ 足够大，增长放缓至亚多项式级别，与多重对数论断一致。次线性幂律出现在所有数据集上，主要持续到……

    arXiv:2609.02143v1 Announce Type: cross  Abstract: Most vector databases rely on graph-based indexes, notably HNSW and Vamana, for approximate nearest neighbor search. With embedding models widely adopted, the datasets these databases store grow rapidly. At a fixed accuracy, how does search cost scale with dataset size? The prevailing answer is poly-logarithmic growth. Yet the claim is proven only under special conditions and asserted without proof for the indexes used in practice. It is also largely untested: standard benchmarks measure cost at one dataset size, not across sizes. We put the claim to the test. The answer depends on the scale itself. While the dataset size $N$ is small relative to the data's intrinsic dimensionality, search cost grows as $N^c$ for a constant $0<1$. We call this scaling the Sublinear Power Law. Once $N$ is large enough, growth slows to subpolynomial, consistent with the poly-logarithmic claim. The Sublinear Power Law appears on every dataset, mostly up t
    
[^86]: SoK：流标签从何而来？审计加密流量基准中的标签溯源

    SoK: Where Do Flow Labels Come From? Auditing Label Provenance in Encrypted Traffic Benchmarks

    [https://arxiv.org/abs/2609.02140](https://arxiv.org/abs/2609.02140)

    该论文首次系统化审计了加密流量分类基准中的标签来源问题，揭示了“粗粒度继承”与“过严过滤”两类标签策略的固有风险，并推导出仅使用严格侧信道特征的分类器所能达到的平衡准确率理论上限。

    

    加密流量分类从传输层可观测信息中推断流记录之外的语义，而监督式训练则依赖于对其所附着的单个流成立的标签。近期的系统化研究审视了模型输入与数据划分；我们则对与之互补的标签一侧进行系统化。在审计的14个基准条目中，我们识别出两种反复出现的标签侧策略：一是粗粒度继承，其风险在于给证据无法覆盖的流打上标签；二是过严过滤，即只保留能够自我证明的流，从而有丢弃相关流的风险。没有任何被审计的条目披露可计数的预筛选总体，且下游论文附加到相同标签上的任务对象中，23个被引用单元格里有8个与恢复出的记录不一致。在严格的侧信道特征下，我们推导出任何仅限于使用这些特征的分类器的相对表示的平衡准确率上限：在公开基准上……

    arXiv:2609.02140v1 Announce Type: cross  Abstract: Encrypted traffic classification infers semantics beyond the flow record from transport-layer observables, and supervised training rests on labels that hold for the individual flow they are attached to. Recent systematizations scrutinize model in- puts and data splits; we systematize the complementary label side. Across 14 audited benchmark entries, we identify two recurring label-side strategies: coarse inheritance, which risks labelling flows the evidence does not cover, and overstrict filtering, which keeps only self-attesting flows and risks dis- carding relevant ones. No audited entry exposes a countable pre-selection population, and the task objects downstream papers attach to the same labels disagree with the recovered record in 8 of 23 referenced cells. Under strict side-channel features we derive a representation-relative ceiling on bal- anced accuracy for any classifier restricted to those features: on the public benchmarks t
    
[^87]: HyperMC：面向随机梯度MCMC的多保真度超参数调优方法

    HyperMC: Multi-Fidelity Hyperparameter Tuning for Stochastic Gradient MCMC

    [https://arxiv.org/abs/2609.02138](https://arxiv.org/abs/2609.02138)

    提出了HyperMC框架，将Hyperband风格的资源分配与核Stein差异评估相结合，为缺乏Metropolis-Hastings接受率的SGMCMC方法实现高效的多保真度超参数调优，并通过全局网格初始化与精英引导局部细化增强了鲁棒性。

    

    随机梯度马尔可夫链蒙特卡罗（SGMCMC）方法能够实现可扩展的贝叶斯推断，但其性能强烈依赖于步长、小批量大小以及leapfrog步数等超参数。由于大多数SGMCMC算法缺乏Metropolis-Hastings接受率，标准的基于接受率的调优方法无法直接适用。我们提出了HyperMC，一个将Hyperband风格的资源分配与核Stein差异（KSD）评估相结合的多保真度调优框架。通过运行多个连续减半调度区间，HyperMC在固定计算预算下，平衡了对连续超参数空间的广泛探索与对有前景配置的日益精确的评估。我们进一步提出了Robust HyperMC，它采用全局网格初始化 followed by 精英引导的局部细化策略，以降低对随机候选生成和含噪声的有限预算评估的敏感性。

    arXiv:2609.02138v1 Announce Type: cross  Abstract: Stochastic gradient Markov chain Monte Carlo (SGMCMC) methods enable scalable Bayesian inference, but their performance depends strongly on hyperparameters such as the step size, mini-batch size, and number of leapfrog steps. Since most SGMCMC algorithms lack a Metropolis-Hastings acceptance rate, standard acceptance-based tuning methods are not directly applicable. We propose HyperMC, a multi-fidelity tuning framework that combines Hyperband-style resource allocation with kernel Stein discrepancy (KSD) evaluation. By running multiple successive-halving brackets, HyperMC balances broad exploration of a continuous hyperparameter space with increasingly accurate evaluation of promising configurations under a fixed computational budget. We further introduce Robust HyperMC, which uses global grid initialization followed by elite-guided local refinement to reduce sensitivity to random candidate generation and noisy finite-budget evaluations
    
[^88]: 面向材料表征中基于图像逆问题的可扩展复合函数贝叶斯优化

    Scalable Bayesian Optimization of Composite Functions for Image-Based Inverse Problems in Materials Characterization

    [https://arxiv.org/abs/2609.02126](https://arxiv.org/abs/2609.02126)

    本文提出一种可扩展的复合函数贝叶斯优化方法（SBOCF），利用图像匹配目标的复合结构和模拟图像的中间信息，高效地从PACBED图样中估计样品厚度和晶体失倾等关键参数，克服了网格搜索效率低下和神经网络需大量预训练且难以迁移的缺点。

    

    从科学图像中估计物理参数是材料表征中一类常见的逆问题，通常依赖于昂贵的基于物理的模拟。在电子显微学中，样品厚度和晶体失倾是决定电子如何在样品中散射的关键参数，因此直接影响从图像中恢复的原子尺度结构的准确性。这些参数通常通过将实验测得的位置平均会聚束电子衍射（PACBED）图样与模拟图样进行匹配来推断，但网格搜索方法的可扩展性差，而神经网络方法则需要大量预训练，且可能无法迁移到新的实验条件。本文提出了可扩展的复合函数贝叶斯优化方法（SBOCF），这是一种节省模拟计算量的高效方法，它利用了图像匹配目标函数已知的复合结构以及模拟图像中包含的中间信息。通过用图像块表示PACBED图像……

    arXiv:2609.02126v1 Announce Type: new  Abstract: Estimating physical parameters from scientific images is a common inverse problem in materials characterization that often relies on expensive physics-based simulations. In electron microscopy, specimen thickness and crystal mistilt are critical parameters that govern how electrons scatter through the sample, and therefore the accuracy of any atomic-scale structure recovered from it. They are commonly inferred by matching experimental position-averaged convergent-beam electron diffraction (PACBED) patterns to simulated ones, but grid searches scale poorly and neural-network methods require extensive pretraining that may not transfer to new conditions. Here, we propose scalable Bayesian optimization of composite functions (SBOCF), a simulation-efficient method that exploits the known composite structure of the image-matching objective and the intermediate information contained in simulated images. By representing PACBED images with patch-
    
[^89]: 疾病负担重于肤色：解构皮肤病学AI的泛化差距

    Disease Burden over Skin Tone: Decomposing the Dermatology-AI Generalization Gap

    [https://arxiv.org/abs/2609.02111](https://arxiv.org/abs/2609.02111)

    该研究发现，皮肤病学AI模型泛化能力差的主要原因是疾病分布偏移而非肤色代表性不足，表明扩大疾病覆盖范围比增加肤色多样性对提升模型泛化更为关键。

    

    皮肤病学人工智能（AI）模型主要在以浅肤色、以癌症为中心的图像集合上训练，然而这些模型越来越多地被提议部署于资源受限的环境中，而那里的患者与训练人群在两个相互混淆的维度上存在差异：肤色和疾病分布。我们研究泛化能力不佳究竟主要是由肤色代表性不足还是疾病分布偏移所致。我们评估了一个以癌症训练的基线模型（在HAM10000和ISIC 2019上微调的ResNet-50）、两个皮肤病学基础模型（DermLIP和MONET）以及一个通用视觉模型（DINOv3），将其作为冻结特征提取器。模型在一个按肤色分层且疾病匹配的数据集（Diverse Dermatology Images，DDI）以及一个疾病偏移且肤色多样的数据集（Skin Condition Image Network，SCIN）上进行评估。我们的结果表明，在所评估的设置中，疾病分布偏移对泛化差距的贡献大于肤色。

    arXiv:2609.02111v1 Announce Type: cross  Abstract: Dermatology artificial intelligence (AI) models are predominantly trained on light-skinned, cancer-focused image collections, yet they are increasingly proposed for deployment in resource-constrained settings where patients differ from training populations along two confounded axes: skin tone and disease distribution. We investigate whether poor generalization is primarily caused by skin-tone underrepresentation or disease-distribution shift. We evaluate a cancer-trained baseline (ResNet-50 fine-tuned on HAM10000 and ISIC 2019), two dermatology foundation models (DermLIP and MONET), and a general-purpose vision model (DINOv3) as frozen feature extractors. Models are evaluated on a tone-stratified disease-matched dataset (Diverse Dermatology Images, DDI) and a disease-shifted tone-diverse dataset (Skin Condition Image Network, SCIN). Our results show that disease-distribution shift contributes more than skin tone in the evaluated settin
    
[^90]: 周期物理信息神经网络中傅里叶谱微分与空间自动微分的计算比较

    A Computational Comparison of Fourier Spectral Differentiation and Spatial Automatic Differentiation in Periodic Physics-Informed Neural Networks

    [https://arxiv.org/abs/2609.02110](https://arxiv.org/abs/2609.02110)

    该论文通过严格控制变量的配对实验，比较了周期PINNs中的空间自动微分与傅里叶谱微分，表明傅里叶谱方法通过谱乘法求导并复用傅里叶系数，可在需要多阶或高阶空间导数时显著降低计算与内存开销。

    

    物理信息神经网络（PINNs）通常使用自动微分（AD）来计算偏微分方程残差中出现的空间导数，当需要多个或高阶导数时，其计算与内存开销可能变得十分可观。我们在周期物理空间的PINNs中对空间自动微分与傅里叶谱微分进行了受控比较。在每个配对实验中，神经表示、时间微分、优化器、采样过程和训练计划均保持固定，从而使两种情况仅在空间微分过程上存在差异。对于傅里叶变体，网络输出在均匀周期网格上求值并变换到傅里叶空间，空间导数通过谱乘法获得，且相同的傅里叶系数在不同导数阶数间被重复使用。我们在标准PINNs中对这两种方法进行了比较……

    arXiv:2609.02110v1 Announce Type: new  Abstract: Physics-informed neural networks (PINNs) commonly evaluate the spatial derivatives appearing in partial differential equation residuals using automatic differentiation (AD), whose computational and memory costs can become substantial when multiple or high-order derivatives are required. We perform a controlled comparison of spatial AD and Fourier spectral differentiation in periodic physical-space PINNs. Within each paired experiment, the neural representation, temporal differentiation, optimizer, sampling procedure, and training schedule are held fixed, so that the two cases differ only in the spatial differentiation procedure. For the Fourier variant, network outputs are evaluated on a uniform periodic grid and transformed to Fourier space, where spatial derivatives are obtained through spectral multiplication and the same Fourier coefficients are reused across derivative orders. We compare the two procedures in standard PINNs for the 
    
[^91]: 矢量量化、乘积量化与标量量化的统一率失真视角

    A Unified Rate-Distortion Perspective on Vector, Product, and Scalar Quantization

    [https://arxiv.org/abs/2609.02107](https://arxiv.org/abs/2609.02107)

    本文提出统一的率失真框架来分析离散视觉分词中的矢量、乘积和标量量化，证明最小化失真（而非最大化码本利用率）才是重建保真度的核心目标，并确立了量化方法公平比较所需的两个条件。

    

    离散视觉分词目前主要由矢量量化、标量量化和乘积量化驱动，但缺乏一个统一的概念框架来理解量化中的权衡。在本文中，我们提出了针对现代离散视觉分词的统一率失真视角。通过将量化视为有损压缩，我们以词元数量和码本大小来刻画标称固定长度编码率，并将量化误差视为失真。在这一框架内，我们解决了三个核心问题。首先，我们从理论和实验上证明，最小化失真（而非最大化码本利用率）才是实现重建保真度的主要内在目标，并且与STE（直通估计器）引起的梯度差异直接相关。其次，我们确立了量化方法内在比较的两个关键公平条件：控制潜在特征统计特性，以及强制执行相同的编码率。第三，在该

    arXiv:2609.02107v1 Announce Type: new  Abstract: Discrete visual tokenization, predominantly driven by vector, scalar, and product quantization, lacks a unified conceptual framework for understanding quantization tradeoffs. In this paper, we propose a unified rate--distortion perspective on modern discrete visual tokenization. By viewing quantization as lossy compression, we characterize the nominal fixed-length coding rate through token count and codebook size, and quantization error as the distortion. Within this framework, we resolve three central questions. First, we theoretically and empirically show that minimizing distortion, rather than maximizing codebook utilization, is the primary intrinsic objective for reconstruction fidelity, with a direct connection to the STE-induced gradient discrepancy. Second, we establish two critical fairness conditions for intrinsic quantization comparison: controlling latent feature statistics and enforcing identical coding rates. Third, under th
    
[^92]: 基于四个国际胸部X光队列的BiomedCLIP联邦LoRA适配研究

    Federated LoRA Adaptation of BiomedCLIP Across Four International Chest X-Ray Cohorts

    [https://arxiv.org/abs/2609.02101](https://arxiv.org/abs/2609.02101)

    该研究在横跨三大洲的四个公开胸片队列上验证了BiomedCLIP的联邦LoRA参数高效微调，在无需交换患者数据的前提下将共享类别AUC平均从0.687提升至0.802，为异质医疗机构的隐私保护型多模态模型适配提供了基准。

    

    联邦学习（FL）使各机构能够在不交换数据的情况下训练共享模型，而低秩适配（LoRA）通过仅传输紧凑的低秩更新，使这一方法在大规模场景下切实可行。生物医学成像是这种组合极具吸引力的应用领域：患者数据因隐私法规而被隔离存档，且各机构在扫描设备、协议和计算资源方面差异巨大。这种异质性引出了一个关键问题——联邦LoRA更新应如何进行聚合，随着多模态视觉-语言模型日益成为医学图像分析的核心，这一问题愈发紧迫。我们在横跨三大洲（美国、越南、西班牙）的四个公开队列上，对BiomedCLIP用于胸片分类的联邦参数高效微调（PEFT）进行了基准测试。结果表明，联邦LoRA适配在所有四个队列上都将共享类别的AUC从基线提升（平均从0.687提升至0.802），优于未经适配的BiomedCLIP骨干网络，说明性能增益确实来自联邦适配本身。

    arXiv:2609.02101v1 Announce Type: cross  Abstract: Federated learning (FL) lets institutions train a shared model without exchanging data, and Low-Rank Adaptation (LoRA) makes this practical at scale by communicating only compact low-rank updates. Biomedical imaging is a compelling setting for this combination: patient data are archived behind privacy regulations, and institutions differ widely in scanners, protocols, and compute. Such heterogeneity raises the question of how federated LoRA updates should be aggregated, increasingly pressing as multimodal vision-language models become central to medical image analysis. We benchmark federated Parameter-efficient fine-tuning (PEFT) of BiomedCLIP for chest radiograph classification across four public cohorts on three continents (USA, Vietnam, Spain). Federated LoRA adaptation improves shared-class AUC on all four cohorts over the unadapted BiomedCLIP backbone (mean 0.687 to 0.802), showing that the gains come from federated adaptation rat
    
[^93]: 面向基于大语言模型的在线时间序列预测的组合式频谱提示

    Compositional Spectral Prompts for LLM-based Online Time Series Forecasting

    [https://arxiv.org/abs/2609.02093](https://arxiv.org/abs/2609.02093)

    提出CoSPOT框架，通过冻结大语言模型并采用基于频域的组合式频谱提示，以极少的参数更新实现高效的在线时间序列预测，克服了现有方法难以长期适应和无法泛化到未见模式的局限。

    

    为了应对时间序列的序列性和持续演化特性，在线时间序列预测（OTSF）任务已在多个领域得到广泛研究。现有研究主要专注于通过采用基于记忆缓冲区的检索策略来适应非平稳环境。然而，我们观察到此类框架难以实现长期适应，且无法泛化到未见过的模式。为此，我们提出了CoSPOT，一个基于大语言模型的在线时间序列预测框架。受大语言模型强大少样本学习能力的启发，该框架利用预训练的大语言模型作为在线预测器的骨干网络。为了实现高效的在线适应，CoSPOT保持大语言模型参数冻结，并采用基于频域基底的组合式频谱提示来引导模型把握输入的整体分布，从而大幅减少了在线阶段需要更新的参数数量。具体而言，CoSPOT将时间序列分解为频域成分……（摘要在此处截断）

    arXiv:2609.02093v1 Announce Type: new  Abstract: To address the sequential and evolving nature of time series, the Online Time Series Forecasting (OTSF) task has been extensively studied in multiple domains. Existing research focuses on adapting to non-stationary environments by employing memory buffer-based retrieval strategies. However, we observe that such frameworks struggle with long-term adaptation and fail to generalize to unseen patterns. To this end, we introduce CoSPOT, an LLM-based online time series forecasting framework that leverages a pre-trained LLM as the backbone online forecaster, motivated by its strong few-shot capabilities. For efficient online adaptation, CoSPOT keeps the LLM frozen and employs compositional spectral prompts grounded in frequency-domain bases to guide the model with the overall distribution of the input, thereby substantially reducing the number of parameters updated during the online phase. Specifically, CoSPOT decomposes time series into freque
    
[^94]: IDEEA：通过激活簇匹配实现无需训练的输入相关引导

    IDEEA: training-free Input-Dependent stEEring via Activation cluster matching

    [https://arxiv.org/abs/2609.02089](https://arxiv.org/abs/2609.02089)

    提出IDEEA框架，通过对每个注意力头的正负激活进行聚类并求解最优匹配问题来构建簇条件化的引导方向，首次实现了无需训练、随输入自适应变化的大模型激活引导，克服了传统固定单一方向引导的根本局限。

    

    引导技术通过在推理时向选定的激活中注入偏置来对齐大型语言模型（LLM），与监督微调或强化学习等权重更新方法相比，这是一种成本远低的替代方案。然而，现有的大多数无需训练的引导方法都是输入无关的：只拟合一次单一方向，并在所有输入间共享。这在根本上存在局限，因为不同的输入占据激活空间的不同区域，并且针对同一目标概念容许不同的最优引导方向，就像相对于固定损失的梯度会随输入而变化一样。我们通过 IDEEA（通过激活簇匹配实现输入相关引导）来弥补这一空白，这是一个用于输入相关引导的无需训练的框架。IDEEA 对每个注意力头的正负激活支持进行聚类，并求解一个最优匹配问题来构建一组条件于簇的方向……（摘要原文在此处被截断）

    arXiv:2609.02089v1 Announce Type: new  Abstract: Steering aligns large language models (LLMs) by injecting a bias into selected activations at inference time, offering a far cheaper alternative to weight-update methods such as supervised fine-tuning or reinforcement learning. However, most existing training-free steering methods are input-independent: a single direction is fitted once and shared across all inputs. This is fundamentally limiting as different inputs occupy different regions of the activation space and admit different optimal steering directions toward the same target concept, much as the gradient with respect to a fixed loss varies from input to input. We close this gap with IDEEA (Input-Dependent stEEring via Activation cluster matching), a training-free framework for input-dependent steering. IDEEA clusters the positive and negative activation supports per attention head, and solves an optimal-matching problem to construct a set of cluster-conditional directions, all a
    
[^95]: TC-Next：零样本多模态热带气旋预报

    TC-Next: Zero-Shot Multimodal Cyclone Forecasting

    [https://arxiv.org/abs/2609.02085](https://arxiv.org/abs/2609.02085)

    TC-Next是一个仅在西太平洋GraphCast预报上训练、只使用通用大气变量的多模态深度学习模型，能大幅降低热带气旋路径和强度预报误差，并可零样本直接迁移到Pangu-Weather、IFS HRES和WeatherNext Cyclones等其他天气基础模型上，性能超越传统规则追踪器乃至专门的直接追踪器。

    

    我们提出了TropicalCycloneNext（TC-Next），这是一个多模态深度学习模型，通过利用基础模型预报的大气动力学和热力学场以及GridSat红外卫星图像，对未来6-24小时的热带气旋路径和强度进行预报。TC-Next仅在覆盖西太平洋（WP）的GraphCast预报数据上训练，且仅依赖通用的大气变量，其在GraphCast上的表现相较于传统的基于规则的追踪器TempestExtremes，将路径误差降低了15-44%，强度误差降低了3-6倍；在无需重新训练的情况下直接应用于Pangu-Weather和IFS HRES的预报场时，它在两者上均优于TempestExtremes。将TC-Next零样本应用于WeatherNext Cyclones在2025年西太平洋季的通用天气场时，与该模型专门设计的直接追踪器（确定性设置）相比，TC-Next在每个预报时效上都取得了更低的强度误差，以及更低或相当的路径误差。

    arXiv:2609.02085v1 Announce Type: new  Abstract: We present TropicalCycloneNext (TC-Next), a multimodal deep learning model that forecasts tropical cyclone track and intensity at $6$-$24$ h leads by leveraging a foundation model's forecast fields of atmospheric kinematic and thermodynamic fields and GridSat infrared satellite imagery. Trained only on GraphCast forecasts over the Western Pacific (WP), yet reliant only on generic atmospheric variables, TC-Next on GraphCast lowers track error by $15$-$44\%$ and intensity error by a factor of $3$-$6$ relative to a conventional, rule-based tracker, TempestExtremes; applied without retraining to the forecast fields of Pangu-Weather and IFS HRES, it stays ahead of TempestExtremes on both. Applied zero-shot to the generic weather fields of WeatherNext Cyclones on the 2025 WP season, TC-Next attains lower intensity error at every lead time, and lower or comparable track error, compared to that model's specialized direct tracker in a determinist
    
[^96]: XMerge：用于大语言模型深度压缩的跨轴选择与重构式层合并

    XMerge: Cross-Axis Selection and Reconstructive Layer Merging for LLM Depth Compression

    [https://arxiv.org/abs/2609.02083](https://arxiv.org/abs/2609.02083)

    XMerge 是一种训练后的大语言模型深度压缩方法，通过跨轴选择识别隐藏状态变化最小的层块，并利用局部边界重构重新拟合相邻存留块，在不改变架构、不增加推理参数、无需任务标签的情况下实现高质量的层删除压缩。

    

    删除完整的 transformer 层可以保持标准的服务架构，但现有的深度压缩方法可能会损失大量质量，且损失在不同模型间变化难以预测。我们提出了 XMerge，这是一种包含两个组件的训练后方法。跨轴选择用于识别隐藏状态的相对幅度和角度变化较小的层块，局部边界重构则重新拟合相邻的存留块以匹配原始两个块的输出。XMerge 不使用任务标签或端到端微调，也不引入架构变更或额外的推理时参数。在七个 Llama 和 Qwen 主干模型（0.5B-8B）、五个已发表的基线方法和三个层削减级别上的实验表明，其相对于基线的优势在最激进的删除设置下最大：在 k=4 时，它在 CORE（一个 22 项任务的聚合基准）上于七个主干模型中的六个上排名第一，并且在 MMLU 上也在七个中的六个上排名第一（两个基准上同时排名第一的有五个）。

    arXiv:2609.02083v1 Announce Type: cross  Abstract: Removing complete transformer layers preserves a standard serving architecture, but existing depth-compression methods can lose substantial quality, and the loss varies unpredictably across models. We introduce XMerge, a post-training method with two components. Cross-axis selection identifies a block with low relative-magnitude and angular hidden-state change, and local boundary reconstruction re-fits the adjacent surviving block to match the original two-block output. XMerge uses no task labels or end-to-end fine-tuning, and it introduces neither architectural changes nor additional inference-time parameters. Across seven Llama and Qwen backbones (0.5B-8B), five published baselines, and three layer-reduction levels, its advantage over baselines is largest at the most aggressive removal: at k=4 it ranks first on six of seven backbones on CORE (a 22-task aggregate) and, separately, on six of seven on MMLU (five of seven on both at once
    
[^97]: DynG-Diff：一种用于概率时间序列预测的状态感知动态引导扩散框架

    DynG-Diff: A State-Aware Dynamic Guidance Diffusion Framework for Probabilistic Time Series Forecasting

    [https://arxiv.org/abs/2609.02068](https://arxiv.org/abs/2609.02068)

    DynG-Diff提出了一种状态感知的动态引导扩散框架，通过两阶段分离训练策略建模多变量时间序列联合分布，并利用轻量级策略网络自适应推断变量可靠性以输出动态引导强度，从而解决了概率多变量时间序列预测中变量间信息异质性的问题。

    

    概率多变量时间序列（MTS）预测对于建模复杂动力系统至关重要。然而，现有的基于扩散的方法依赖于任务特定的条件范式，缺乏灵活性，并且难以应对固有的“信息异质性”问题——即不同变量之间噪声水平和演化模式的显著差异。为解决这一问题，我们提出了DynG-Diff，一种面向概率多变量时间序列预测的变量敏感动态引导扩散框架：(1) DynG-Diff采用两阶段分离训练策略，使用无条件扩散骨干网络来建模多变量时间序列的联合分布；(2) DynG-Diff引入一个轻量级的状态感知策略网络，从实时噪声状态和单步去噪估计中自适应地推断变量的可靠性，并输出动态引导强度矩阵；(3) DynG-Diff从数学上对这种动态加权机制进行形式化表述……

    arXiv:2609.02068v1 Announce Type: new  Abstract: Probabilistic multivariate time series (MTS) forecasting is crucial for modeling complex dynamical systems. However, existing diffusion-based methods rely on task-specific conditional paradigms that lack flexibility and struggle with inherent "information heterogeneity"--the significantly varying noise levels and evolutionary patterns across variables. To address this, we propose DynG-Diff, a variable-sensitive dynamic guidance diffusion framework for probabilistic multivariate time-series forecasting: (1) DynG-Diff adopts a two-stage separated training strategy and uses an unconditional diffusion backbone to model the joint distribution of multivariate time series. (2) DynG-Diff introduces a lightweight state-aware policy network that adaptively infers variable reliability from real-time noisy states and one-step denoising estimates, outputting a dynamic guidance strength matrix. (3) DynG-Diff mathematically formulates this dynamic weig
    
[^98]: DocHop：信息密集文档中域外多跳推理的基准测试

    DocHop: Benchmarking Out-of-domain Multi-hop Reasoning in Information-Dense Documents

    [https://arxiv.org/abs/2609.02059](https://arxiv.org/abs/2609.02059)

    提出了DocHop基准，用于评估多模态大语言模型能否在信息密集的文档图像中，利用文本叙述上下文解析目标实体，并跨多个图表进行多跳证据聚合与推理。

    

    多模态大语言模型（MLLMs）在图表问答和文档问答等结构化视觉理解任务上已取得了优异的表现。然而，现有基准通常将这些领域孤立地进行评估，因而忽略了一项关键能力：模型能否利用文本上下文来决定如何选择、解释和聚合图表证据。我们提出了DocHop，一个用于文档风格图像中图表-上下文整合推理的基准。在DocHop中，文档叙述规定了多步骤的组合约束条件，而图表则提供相应的数据数值。问题建立在叙述中定义的语义参考标签之上，要求模型在跨多个图表聚合证据之前，先从上下文中解析出目标实体。为了实现系统性评估，我们通过一个具有可控推理深度的随机化“逻辑优先”生成流程构建了DocHop……

    arXiv:2609.02059v1 Announce Type: new  Abstract: Multimodal Large Language Models (MLLMs) have achieved strong performance on structured visual understanding tasks such as chart and document question answering. However, existing benchmarks typically evaluate these domains in isolation, leaving underexplored a key capability: whether models can use textual context to determine how chart evidence should be selected, interpreted, and aggregated. We introduce DocHop, a benchmark for integrated chart--context reasoning in document-style images. In DocHop, the document narrative specifies multi-step compositional constraints, while charts provide the corresponding data values. Questions are grounded on a semantic reference label defined in the narrative, requiring models to resolve target entities from context before aggregating evidence across multiple charts. To enable systematic evaluation, we construct DocHop via a stochastic logic-first generation pipeline with controllable reasoning de
    
[^99]: 语言模型中连续混合坍缩的动力学

    The Dynamics of Continuous Mixture Collapse in Language Models

    [https://arxiv.org/abs/2609.02049](https://arxiv.org/abs/2609.02049)

    该研究揭示了语言模型无法保持连续混合推理状态的深层原因，识别出三种相互独立的失败机制：transformer 架构对混合几何结构的固有扭曲、训练过程对这种扭曲的显著放大，以及 softmax 读出与自回归反馈构成的动力系统导致混合分量被单一主导或坍缩至不可区分。

    

    大语言模型的潜在状态推理方法用连续状态（例如词元嵌入的加权混合）取代离散的中间词元，以保留多种可能的推理方向，而不是只承诺其中一种。然而，预训练语言模型往往无法保持这些混合状态。我们通过理论分析与在多种模型上开展的受控实证研究相结合的方式探究其成因，并识别出三种相互独立且截然不同的失败来源。首先，transformer 架构本身就会扭曲混合的几何结构，而训练过程会显著放大这种效应。此外，即使模型能够完美地以线性方式传输混合，失败仍可能发生：softmax 读出与自回归反馈共同构成一个动力系统，该系统要么不断放大微小的差异直到混合中的某一分量占据主导地位，要么收缩不同的混合直到它们变得无法区分。我们验证了这一理论预测……

    arXiv:2609.02049v1 Announce Type: cross  Abstract: LLMs latent-state reasoning methods replace discrete intermediate tokens with continuous states, such as weighted mixtures of token embeddings, to retain multiple possible reasoning directions rather than committing to one. Yet pretrained language models often fail to preserve these mixtures. We study why through a combination of theoretical analysis and controlled empirical investigations on a variety of models. We identify three independent, distinct sources of failure. First, transformer architectures already distort mixture geometry, and training substantially amplifies this effect. Moreover, the failure can occur even if the model transports mixtures perfectly linearly: the softmax readout and autoregressive feedback form a dynamical system that either amplifies small differences until one component of the mixture dominates or contracts different mixtures until they become indistinguishable. We verify this theoretical prediction e
    
[^100]: 多行动，少决策：面向长程LLM智能体的技能引导自适应动作分块

    Act More, Decide Less: Skill-Guided Adaptive Action Chunking for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2609.02042](https://arxiv.org/abs/2609.02042)

    提出SPACE方法，通过从成功轨迹中归纳两级程序化技能并以子技能边界作为块边界监督进行蒸馏，使长程LLM智能体学会自适应的可变长度动作分块，克服了标准强化学习无法学习块边界导致的单动作退化或过长序列问题。

    

    面向长程交互任务的大语言模型（LLM）智能体通常遵循ReAct风格的协议，每轮LLM调用仅发出一个原始动作。虽然这种方式支持频繁的重新规划，但在长程任务中效率低下，因为大量轮次被消耗在常规动作序列上。一种自然的替代方案是让智能体输出可变长度的动作块。然而，使用标准强化学习直接训练此类策略会失败：智能体要么退化为单动作行为，要么过度承诺过长的序列。这两种失败有着共同的根源：无法学习块边界。我们提出SPACE，通过从轨迹归纳出的程序化技能中蒸馏块边界监督来解决这一挑战。我们从成功轨迹中归纳出两级程序化技能，其中子技能边界直接作为块边界监督，随后将这种时间结构蒸馏进策略中。

    arXiv:2609.02042v1 Announce Type: new  Abstract: Large language model (LLM) agents for long-horizon interactive tasks typically follow a ReAct-style protocol, issuing one primitive action per LLM round. While this enables frequent replanning, it is inefficient for long-horizon tasks where many rounds are spent on routine action sequences. A natural alternative is to let the agent emit variable-length action chunks. However, naively training such policies with standard reinforcement learning fails: the agent either collapses to single-action behavior or over-commits to excessively long sequences. Both failures share a common root cause: the inability to learn chunk boundaries. We propose SPACE, which addresses this challenge by distilling chunk-boundary supervision from trajectory-induced programmatic skills. We induce two-level programmatic skills from successful trajectories, where subskill boundaries serve as direct chunk-boundary supervision. This temporal structure is then distille
    
[^101]: 无源类别重学习：诊断类别遗忘中的遗忘现象

    Source-Free Class Relearning: Diagnosing Forgetting in Class Unlearning

    [https://arxiv.org/abs/2609.02018](https://arxiv.org/abs/2609.02018)

    本文提出一种严格无源的类别重学习方法，证明仅凭遗忘后的模型本身、对合成探测集进行单步梯度更新分类器头即可恢复被遗忘的类别，从而诊断出类别遗忘并未真正抹除类别结构。

    

    类别遗忘旨在移除模型识别指定遗忘类别的能力，同时保持模型在保留类别上的性能。然而，遗忘后的低遗忘准确率并不一定意味着类别结构已被真正抹除。近似遗忘方法可能改变分类器的决策边界，同时在特征表示中留下可恢复的结构。已有研究表明遗忘类别是可以被恢复的，但现有方法需要真实的遗忘或保留样本、辅助数据或参考模型检查点。我们在严格无源的设置下研究类别重学习问题，即仅利用被遗忘处理后的模型本身，探究能否通过更新分类器头来恢复遗忘类别。我们的方法建立在一项理论分析之上，该分析确立了一个充分对齐条件，在该条件下，对一个合成探测集执行单步梯度更新即可增大遗忘类别的期望逻辑值间隔。基于此，我们提出……

    arXiv:2609.02018v1 Announce Type: new  Abstract: Class unlearning aims to remove a model's ability to recognize designated forget classes while preserving performance on retain classes. However, low forget accuracy after unlearning does not necessarily mean the class structure has been erased. Approximate unlearning methods can alter classifier decision boundaries while leaving recoverable structure in the representation. Prior work has shown that forget classes can be recovered, but existing approaches require real forget or retain samples, auxiliary data, or reference checkpoints. We study class relearning in a strictly source-free setting, asking whether a forget class can be recovered through a classifier-head update using only the unlearned model. Our approach rests on a theoretical analysis establishing a sufficient alignment condition under which a single gradient step on a synthetic probe set increases the expected logit margin of the forget class. Building on this, we propose 
    
[^102]: 用于图像超分辨率的感知正则化扩散模型

    Perceptually Regularized Diffusion Model for Image Super-Resolution

    [https://arxiv.org/abs/2609.02016](https://arxiv.org/abs/2609.02016)

    本文提出一种感知正则化扩散模型，在扩散模型训练中引入感知正则化，以克服标准像素域噪声预测损失导致的过度平滑问题，从而提升图像超分辨率的感知质量。

    

    图像超分辨率旨在从低分辨率观测中重建高分辨率图像，是医学成像、遥感、监控、显微成像和科学可视化等领域的基础技术。传统的基于模型的方法将超分辨率表述为带有手工设计正则化先验的逆问题，虽然这类方法具有可解释性和理论基础，但它们依赖于固定假设，且需要计算开销较大的迭代求解器。深度学习方法通过学习从低分辨率到高分辨率图像的非线性映射，提供了数据驱动的灵活性，其中扩散模型在感知质量方面取得了尤为令人瞩目的成果。然而，标准的扩散模型训练目标是像素域的噪声预测损失，并未显式地保证感知保真度，这可能导致过度平滑和精细图像结构的丢失。为了解决这些局限性，我们提出了一种感知……

    arXiv:2609.02016v1 Announce Type: cross  Abstract: Image super-resolution, which aims to reconstruct high-resolution images from their low-resolution observations, is fundamental to medical imaging, remote sensing, surveillance, microscopy, and scientific visualization. Traditional model-based methods formulate super-resolution as an inverse problem with hand-crafted regularization priors. While interpretable and theoretically grounded, they rely on fixed assumptions and require computationally intensive iterative solvers. Deep learning methods offer data-driven flexibility by learning nonlinear mappings from low- to high-resolution images, among which diffusion models have achieved particularly impressive perceptual quality. However, the standard diffusion training objective is a pixel-domain noise-prediction loss that does not explicitly enforce perceptual fidelity, which can lead to oversmoothing and loss of fine image structure. To address these limitations, we propose a perceptual
    
[^103]: 训练你所部署的：缩小低秩克隆蒸馏中MLP的可达性差距

    Train What You Deploy: Closing the MLP Reachability Gap in Low-Rank Clone Distillation

    [https://arxiv.org/abs/2609.02006](https://arxiv.org/abs/2609.02006)

    该论文提出“训练你所部署的”原则，让训练直接覆盖完整部署矩阵而非教师诱导的权重切片，在不增加任何推理成本的前提下释放低秩克隆蒸馏中62.5-81.4%被困住的容量，在三个教师模型上取得显著性能提升。

    

    压缩后的学生模型存在两个未必一致的结构：它在推理时部署的权重，以及其训练所能到达的权重族。我们证明，最先进的权重继承蒸馏方法——低秩克隆——部署的是全宽的学生MLP，但训练却被绑定在由教师模型诱导的权重切片上，导致每个已部署矩阵62.5%至81.4%的独立线性自由度无法被训练到——这些容量在推理时付出了代价，却从未被训练。我们的原则只有一句话：训练你所部署的。从相同的LRC热启动出发，我们将训练对象设为整个已部署矩阵，通过两种可合并的实现方式（Dense-LRC和CORE-LRC，二者均可折叠为单一的已部署权重），在不改变部署形状、部署参数量或推理FLOPs的前提下完成训练。这恢复了被搁置的模型容量：在每个教师模型下采用更强的实现方式，相对于匹配预算的朴素LRC基线，在三个教师模型（Llama3.2-3B、Llama3.1-8B、Qw……）上分别取得+2.36/+2.71/+10.45的Avg9提升。

    arXiv:2609.02006v1 Announce Type: cross  Abstract: A compressed student has two shapes that need not agree: the weight it deploys at inference and the weight family its training can reach. We show that a state-of-the-art weight-inheritance distiller, Low-Rank Clone (LRC), deploys a full-width student MLP but ties training to a teacher-induced slice, leaving 62.5-81.4% of each deployed matrix's independent linear degrees of freedom unreachable-paid for at inference, never trainable. Our principle is one line: train what you deploy. From the identical LRC warm start, we make the training object the entire deployed matrix, with no change in deployed shape, deployed parameter count, or inference FLOPs, via two mergeable realizations (Dense-LRC and CORE-LRC) that both collapse to one deployed weight. This recovers stranded capacity: taking the stronger realization per teacher, +2.36/+2.71/+10.45 Avg9 over matched-budget plain-LRC baselines across three teachers (Llama3.2-3B, Llama3.1-8B, Qw
    
[^104]: 后验温度化解释了线性与广义线性汤普森采样中的方差膨胀

    Posterior Tempering Explains Variance Inflation in Linear and Generalized Linear Thompson Sampling

    [https://arxiv.org/abs/2609.01999](https://arxiv.org/abs/2609.01999)

    该论文提出 α-TS 算法，通过用 α-后验（分数幂后验）替代标准后验来形式化方差膨胀思想，并给出了先验与奖励分布的一般正则性条件，使汤普森采样在广义线性老虎机中无需后验近似即可完成遗憾分析，且当 α ∝ d^{-1} 时达到了已知最优的 O(d^{3/2}√T log T) 遗憾界。

    

    我们研究了一种汤普森采样（TS）算法的变体，称为 α-TS，用于解决随机广义线性老虎机问题。现有的 TS 分析方法需要膨胀后验方差才能推导出接近最优的遗憾界保证。我们通过引入 α-TS 来形式化方差膨胀的思想，该算法使用分数幂后验（α-后验）替代标准后验。我们的主要贡献是识别了关于先验分布和奖励分布的一般正则性条件，使得能够在不假设后验分布存在任何可处理近似的情况下对 α-TS 进行遗憾分析，这一点不同于以往的工作。对于 α ∝ d^{-1} 的特定选择，我们的一般遗憾界对指数族和次高斯族的奖励分布均给出了已知最优的遗憾界 O(d^{3/2}√T log T)。我们进一步提供了一个依赖于 α 的下界，表明遗憾常数 d（摘要在此处被截断）

    arXiv:2609.01999v1 Announce Type: cross  Abstract: We study a variant of the Thompson Sampling (TS) algorithm, called $\alpha$-TS, for solving stochastic generalized linear bandit problems. Existing analyses of TS require inflating the posterior variance to derive near-optimal regret guarantees. We formalize the idea of variance inflation by introducing $\alpha$-TS that uses a fractional or $\alpha$-posterior instead of the standard posterior. Our main contribution is to identify general regularity conditions on the prior and reward distributions that enable a regret analysis of $\alpha$-TS without assuming any tractable approximation of the posterior distribution, unlike previous works. For a specific choice of $\alpha \propto d^{-1}$, our general regret bound yields the best known regret bound of $O(d^{3/2}\sqrt{T}\log T)$ for both the exponential and sub-Gaussian families of reward distributions. We further provide an $\alpha$-dependent lower bound showing that the regret constant d
    
[^105]: 用于快速免训练球形全景生成的线性融合MultiDiffusion

    Linear Fusion MultiDiffusion for Fast Training-Free Spherical Panorama Generation

    [https://arxiv.org/abs/2609.01997](https://arxiv.org/abs/2609.01997)

    LF-MultiDiffusion是一种免训练的球形全景生成方法，通过将潜变量聚合重新表述为正则化最小二乘问题并利用Krylov迭代求解器在去噪循环内求解，实现了比最强免训练基线更好的视觉质量、文本对齐和全景一致性，同时带来15.36倍的推理加速。

    

    我们提出了LF-MultiDiffusion，这是一种免训练的全景生成方法，它扩展了MultiDiffusion以支持目标图像空间和参考图像空间之间的线性投影。我们的核心思想是将潜变量聚合重新表述为一个正则化最小二乘问题，并在去噪循环内使用基于Krylov的迭代求解器进行高效求解。这种表述方式相比先前的免训练方法实现了更稠密、更自然的映射，仅需少得多的透视视图即可实现更稳定的生成。因此，LF-MultiDiffusion减少了去噪过程中图像生成器的评估次数，并显著提升了推理效率。实验表明，LF-MultiDiffusion相比最强的免训练基线方法，在视觉质量、文本对齐和全景一致性方面均取得了更好的表现，同时提供了15.36倍的加速。

    arXiv:2609.01997v1 Announce Type: cross  Abstract: We propose LF-MultiDiffusion, a training-free panorama generation method that extends MultiDiffusion to support linear projections between target and reference image spaces. Our key idea is to reformulate latent aggregation as a regularized least-squares problem and solve it efficiently with a Krylov-based iterative solver inside the denoising loop. This formulation enables denser and more natural mappings than prior training-free methods, yielding more stable generation with far fewer perspective views. As a result, LF-MultiDiffusion reduces the number of image generator evaluations during denoising and significantly improves inference efficiency. Experiments show that LF-MultiDiffusion achieves better visual quality, text alignment, and panoramic consistency than the strongest training-free baseline, while providing a 15.36$\times$ speedup. Our project page is available at: https://ahykw.github.io/lfmd.
    
[^106]: CAHR-Net：面向紧凑且可解释的磁芯损耗建模的条件自适应磁滞回线重构网络

    CAHR-Net: Condition-Adaptive Hysteresis Reconstruction for Compact and Interpretable Magnetic Core Loss Modeling

    [https://arxiv.org/abs/2609.01991](https://arxiv.org/abs/2609.01991)

    CAHR-Net通过特征级线性调制将频率、温度和波形等工作条件直接注入中间磁滞回线重构表征，在保留“波形→磁场→回线面积→损耗”可解释链条的同时，实现了紧凑而准确的磁芯损耗建模。

    

    磁芯损耗源于磁滞回线：每个激励周期所耗散的能量等于回线面积，而频率、温度和波形形状通过重塑回线几何形态来决定损耗大小。大多数现有模型让这些条件仅作用于最终输出的标量——经验公式将它们折算进拟合的指数中，数据驱动的预测器则将它们附加在编码特征之后——因此没有留下任何中间磁滞表征可供条件去重塑。本文提出CAHR-Net，一种条件自适应磁滞回线重构网络，将工作条件注入到其物理上真正起作用的位置。它保留了从磁通密度波形到磁场重构、回线面积积分以及功率损耗估计的可解释链条，并利用特征级线性调制（FiLM）将频率、温度和波形统计量注入中间重构表征中。

    arXiv:2609.01991v1 Announce Type: new  Abstract: Magnetic core loss originates in the hysteresis loop: the energy dissipated per excitation cycle equals the loop area, and frequency, temperature, and waveform shape set the loss by reshaping the loop geometry. Most existing models let these conditions act only on a terminal scalar - empirical equations fold them into fitted exponents, and data-driven predictors append them to encoded features - so no intermediate hysteresis representation remains for the conditions to reshape. This paper proposes CAHR-Net, a condition-adaptive hysteresis reconstruction network that injects the operating conditions where they physically act. It preserves the interpretable chain from flux density waveform to magnetic field reconstruction, loop-area integration, and power loss estimation, and uses feature-wise linear modulation to inject frequency, temperature, and waveform statistics into the intermediate reconstruction representation. A matched large-bat
    
[^107]: 全切片图像基础模型中的形态学信号可自动分拣病理切片

    Morphology signal in whole slide image foundation models can automatically triage slides

    [https://arxiv.org/abs/2609.01987](https://arxiv.org/abs/2609.01987)

    本文提出一种利用公开的WSI基础模型进行零样本分类、按肿瘤含量自动排序和分拣全切片图像的流程，无需病理学家人工筛选，解决了多切片数据集训练中信号稀释的问题。

    

    在癌症诊断和分期过程中，患者的检查通常会产生多张全切片图像（WSI）。在WSI数据上训练模型的初始步骤之一，是识别出一张或少数几张包含肿瘤或其他诊断生物标志物的切片，这些信息是下游预测任务（如估计复发风险或无进展生存期）所必需的。这一步骤需要经验丰富的病理学家进行繁琐的人工筛选。许多已发表的数据集人为地假设每位患者仅有1张切片；另一种做法是使用每位患者的所有切片进行模型训练，但这可能会稀释来自少数包含肿瘤或其他相关信息切片的信号。在本文中，我们提出了一种利用公开可用的WSI基础模型（FMs）来克服这些挑战的处理流程。我们的评估表明，基于WSI基础模型零样本分类的预测结果对WSI进行排序，能够准确识别出肿瘤含量最多的切片，

    arXiv:2609.01987v1 Announce Type: cross  Abstract: Patient exams in the cancer diagnosis and staging process typically generate several whole slide images (WSIs). One of the initial steps in training models on WSI data is identifying one or a few slides containing tumor or other diagnostic biomarkers necessary for downstream prediction tasks such as estimating recurrence risk or progression-free survival. This step requires tedious manual curation by experienced pathologists. Many published datasets make the artificial assumption of 1 slide per patient. Alternatively, all slides per patient may be used for model training, which may dilute the signal from the few slides containing tumor or other relevant information. In this paper, we present a pipeline to overcome these challenges using publicly available WSI foundation models (FMs). Our evaluations show that ranking WSIs based on predictions from zero-shot classification using WSI FMs accurately identifies slides with the most tumor, 
    
[^108]: 一种用于数据驱动流程仿真的统一粒子滤波LSTM

    A Unified Particle Filter LSTM for Data-Driven Process Simulation

    [https://arxiv.org/abs/2609.01967](https://arxiv.org/abs/2609.01967)

    提出统一粒子滤波LSTM，通过维护并更新一组加权的循环状态假设来应对事件日志中潜在流程状态的不确定性，从而提升数据驱动流程仿真的真实性和准确性。

    

    数据驱动的流程仿真旨在从历史事件日志中生成真实的案例轨迹，而无需显式指定底层动态的模型。深度序列模型可以通过下一活动概率和条件时间分布来捕捉复杂的时序依赖关系。然而，事件日志仅提供了底层流程状态的部分视图，通常只记录活动的完成情况而不包含相应的服务开始时间。因此，同一个观测到的流程历史可能与多种合理的潜在流程条件相一致，而标准的循环模型会将每个流程前缀压缩为单一的确定性循环状态。我们提出了一种统一粒子滤波LSTM（Unified PF-LSTM），该方法维护并顺序更新一组加权的循环状态假设。我们使用加权均值以及基于矩的学习特征来总结这一粒子信念（摘要在此处不完整，原文截断）。

    arXiv:2609.01967v1 Announce Type: new  Abstract: Data-driven process simulation aims to generate realistic case trajectories from historical event logs without requiring an explicitly specified model of the underlying dynamics. Deep sequence models can capture complex temporal dependencies through next-activity probabilities and conditional time distributions. However, event logs provide only a partial view of the underlying process state, often recording activity completions without the corresponding service-start times. Consequently, the same observed process history may be consistent with multiple plausible latent process conditions, whereas standard recurrent models compress each process prefix into a single deterministic recurrent state. We propose a Unified Particle Filter LSTM (Unified PF-LSTM) that maintains and sequentially updates a weighted set of recurrent-state hypotheses. We summarize this particle belief using its weighted mean and learned features based on the moment-ge
    
[^109]: Qwen3-4B的后训练三值化：能力、有效位预算、存储压缩与部署

    Post-Training Ternarization of Qwen3-4B Capability, Effective Bit Budget, Storage Compression, and Deployment

    [https://arxiv.org/abs/2609.01962](https://arxiv.org/abs/2609.01962)

    本文对Qwen3-4B模型进行端到端后训练三值化，将量化线性权重压缩至每权重1.641有效比特（覆盖81.62%参数），实现显著存储压缩，但任务准确率从64.5%降至54.7%，且不同任务的能力退化程度不均匀。

    

    超低比特语言模型可以减少存储和内存带宽，但名义上的“1.58比特”标签并不能完全描述其存储表示、保留的能力或运行时行为。我们研究了Qwen（一个指令微调的4B参数模型）的端到端后训练转换，采用KOTMS旋转、E2M-ATQ三值化以及来自TWLA的GPTQ风格误差补偿。该实验仅量化权重：激活值保持16位精度，因此省略了ILA-AMP。我们评估了有效位核算、任务能力保留、困惑度、校准敏感性、检查点组成和部署行为。最终转换对量化线性权重使用每权重1.641有效比特，覆盖了81.62%的模型参数。在十项评分能力比较中，准确率从64.5%下降到54.7%。退化是不均匀的：BoolQ保留了84.6%的机会校正后教师性能，而ARC-Challenge保留了43……

    arXiv:2609.01962v1 Announce Type: new  Abstract: Ultra-low-bit language models can reduce storage and memory bandwidth, but a nominal "1.58-bit" label does not fully describe the stored representation, retained capability, or runtime behavior.   We study an end-to-end post-training conversion of Qwen, an instruction-tuned 4B-parameter model, using KOTMS rotation, E2M-ATQ ternarization, and GPTQ-style error compensation from TWLA. The experiment is weight-only: activations remain at 16-bit precision, so ILA-AMP is omitted. We evaluate effective bit accounting, task capability retention, perplexity, calibration sensitivity, checkpoint composition, and deployment behavior.   The final conversion uses 1.641 effective bits per weight for quantized linear weights, with 81.62% of model parameters targeted. Across ten scored capability comparisons, accuracy falls from 64.5% to 54.7%. Degradation is uneven: BoolQ retains 84.6% chance-corrected teacher performance, while ARC-Challenge retains 43
    
[^110]: 无线接入点上的网络感知预测

    Network-Aware Forecasting on Wireless Access Points

    [https://arxiv.org/abs/2609.01957](https://arxiv.org/abs/2609.01957)

    该论文提出“网络感知可部署性”概念，通过在目标 AP 上的资格认证和负载约束下的执行验证两道门槛来评估预测模型的部署可行性，并发现边缘测试平台无法可靠反映目标无线接入点的真实运行表现。

    

    企业无线接入点（AP）是部署预测性机器学习（ML）的有前景的平台，但其首要职责仍然是提供无线连接和网络服务。因此，预测推理必须与数据包处理、Wi-Fi 和 IoT 无线电操作以及客户端管理共享 AP 的 CPU 和内存。这种资源竞争带来两个风险：在代理硬件上表现良好的模型在目标 AP 上可能运行过慢，而单独运行时资源占用合适的模型在负载下仍可能降低网络服务质量。我们通过两道门槛定义了“网络感知可部署性”：首先是对模型及其执行路径在目标 AP 上进行资格认证，其次是在数据包服务和预测约束下对其执行配置文件进行验证。我们的基准测试表明，边缘测试平台无法可靠地反映目标设备的真实行为。在匹配的工件和服务设置下，五个模型实现的运行速度相差 6.1 至 19.1 倍……（摘要在此处截断）

    arXiv:2609.01957v1 Announce Type: cross  Abstract: Enterprise wireless access points (APs) are promising platforms for predictive machine learning (ML), but their primary responsibility remains providing wireless connectivity and network services. Predictive inference must therefore share an AP's CPU and memory with packet processing, Wi-Fi and IoT radio operations, and client management. This resource contention creates two risks: a model that performs well on proxy hardware may be too slow on the target AP, while a model that fits in isolation may still degrade network services under load. We define \textit{network-aware deployability} using two gates: qualification of the model and its execution path on the target AP, followed by validation of its execution profile under packet-service and forecasting constraints. Our benchmarks show that edge testbeds do not reliably capture target behavior. Across matched artifacts and serving settings, five model implementations run 6.1--19.1$\ti
    
[^111]: FlashKAN：基于截断幂形式的B样条KAN

    FlashKAN: B-Spline KANs via Truncated Power Form

    [https://arxiv.org/abs/2609.01956](https://arxiv.org/abs/2609.01956)

    FlashKAN用逼近论中的截断幂形式取代Cox-de Boor递归，通过torch.compile融合为单一GPU内核并结合有界坐标稳定化，显著加速了KAN中B样条激活函数的计算，并提供了开源软件包。

    

    Kolmogorov-Arnold网络（KAN）将可学习的B样条激活函数放置在网络边上，而非在节点上使用固定激活函数。标准的Cox-de Boor递归在计算k次样条的这些激活函数时需要k次顺序传递，消耗了超过90%的前向传播时间。FlashKAN用截断幂形式取代了这种递归，这是逼近论中的一个经典结果，它将每个均匀三次B样条表示为在移位节点位置上的五个(x)_+^3项。本文做出了三项贡献：(1) 一个torch.compile融合的实现，将这些操作合并为单个GPU内核，消除了所有递归、跨度查找和散布-聚集操作；(2) 一种有界坐标稳定化方法，将归一化输入钳制到[0, k+1]，防止了历史上促使Cox-de Boor递归被采用的灾难性抵消问题；(3) 一个可用于生产环境的开源软件包（pip install flashkan）。

    arXiv:2609.01956v1 Announce Type: new  Abstract: Kolmogorov-Arnold Networks (KANs) place learnable B-spline activations on network edges rather than fixed activations on nodes. The standard Cox-de Boor recursion evaluates these activations through k sequential passes for degree-k splines, consuming over 90% of forward-pass time. FlashKAN replaces this recursion with the truncated power form, a classical result from approximation theory that expresses each uniform cubic B-spline as five (x)_+^3 terms at shifted knot positions. This paper makes three contributions: (1) a torch.compile-fused implementation that collapses these operations into a single GPU kernel, eliminating all recursion, span lookup, and scatter-gather operations; (2) a bounded-coordinate stabilization that clamps the normalized input to [0, k+1], preventing the catastrophic cancellation that historically motivated the Cox-de Boor recursion; and (3) a production-ready, open-source package (pip install flashkan) that ser
    
[^112]: 异步点对点（P2P）八卦学习网络中知识蒸馏的收敛理论

    Convergence Theory of Knowledge Distillation in Asynchronous P2P Gossip Learning Network

    [https://arxiv.org/abs/2609.01952](https://arxiv.org/abs/2609.01952)

    本文首次为异步P2P八卦学习网络中的知识蒸馏建立了收敛理论，将共识从参数空间转移到函数空间，把KD事件建模为logit空间中的几何收缩算子，从而解决了不同参数规模架构的设备无法通过去中心化SGD达成共识的问题。

    

    去中心化、无服务器的学习日益连接着运行不同架构的设备，而标准工具——去中心化SGD——在这种情形下是无法定义的，因为参数数量不同的模型无法进行平均。知识蒸馏（KD）通过交换软预测而非权重来绕过这一障碍，但针对完全去中心化的异步点对点（P2P）知识蒸馏的收敛理论尚属空白。我们提供了这样的理论，将共识从参数空间重新定位到函数（输出）空间：一次KD事件是在对等体预测分布上、logit空间中的几何收缩算子，我们在参考测度下的预测Hilbert空间中对其进行分析。在标准的光滑性/方差假设以及两个可实现性假设（其中一个连接参数SGD与函数步骤，另一个控制受限任务/KD对齐）下，时间平均的函数平稳性与函数空间分歧以……收敛

    arXiv:2609.01952v1 Announce Type: cross  Abstract: Decentralized, serverless learning increasingly connects devices running different architectures, where the standard tool, decentralized SGD, is undefined as models with different parameter counts cannot be averaged. Knowledge distillation (KD) exchanges soft predictions rather than weights and sidesteps this obstacle, yet convergence theory for fully decentralized, asynchronous peer-to-peer (P2P) KD is lacking. We provide one, relocating consensus from parameter space to function (output) space: a KD event is a geometric contraction operator in logit space on the peers' predictive distributions, which we analyse in the Hilbert space of predictions on a reference measure. Under standard smoothness/variance assumptions and two realizability assumptions, one bridging parameter SGD to the functional step and one controlling restricted task/KD alignment, the time-averaged functional stationarity and function-space disagreement converge at 
    
[^113]: 在策略蒸馏与离策略GRPO相结合：训练紧凑型指令跟随重排序器

    On-Policy Distillation Meets Off-Policy GRPO: Training Compact Instruction-Following Rerankers

    [https://arxiv.org/abs/2609.01947](https://arxiv.org/abs/2609.01947)

    该论文提出两阶段蒸馏框架，先用离策略GRPO结合LLM裁判反馈强化4B教师重排序器，再让1B学生模型在自身采样的排序上接受教师软奖励实现在策略蒸馏，在分布偏移场景下取得最显著的性能提升。

    

    紧凑型指令跟随重排序器在部署应用中极具吸引力，但传统的蒸馏流程通常通过在固定样本集上对教师输出进行离线模仿来训练学生模型，这使得监督信号局限于教师已观察到的排序空间。我们从强化学习的视角重新审视重排序器的蒸馏问题。我们提出了一个两阶段框架，将离策略教师优化与在策略学生蒸馏相结合。在第一阶段，利用基于LLM裁判反馈的离策略GRPO，在88K条指令跟随样本上强化一个4B教师重排序器。在第二阶段，紧凑的1B学生模型从自身策略中采样排序结果，并就这些排序获得源自教师的软奖励，从而将学生的探索与知识迁移耦合起来。我们在分布偏移情况下观察到最显著的性能提升。在MAIR-11（原始11子集、869个查询）评估集上，所提出的学生模型达到0.7670 nDCG@6，超越了（原文在此截断）

    arXiv:2609.01947v1 Announce Type: cross  Abstract: Compact instruction-following rerankers are attractive for deployment, but conventional distillation pipelines typically train students by offline imitation of teacher outputs on a fixed set of examples, constraining supervision to the teacher's observed ranking space. We revisit reranker distillation through the lens of reinforcement learning.   We propose a two-stage framework combining off-policy teacher optimization with on-policy student distillation. In Stage 1, a 4B teacher reranker is strengthened with off-policy GRPO using LLM-judge feedback on 88K instruction-following examples. In Stage 2, a compact 1B student samples rankings from its own policy and receives soft teacher-derived rewards on those rankings, coupling student exploration with knowledge transfer.   Our strongest gains appear under distribution shift. On MAIR-11, the original 11-subset, 869-query evaluation, the proposed student reaches 0.7670 nDCG@6, outperformi
    
[^114]: 推进面向私有平均聚合的多密钥同态加密

    Pushing Forward Multi-Secret-Key Homomorphic Encryption for Private Average Aggregation

    [https://arxiv.org/abs/2609.01945](https://arxiv.org/abs/2609.01945)

    该论文提出了基于RLWE同态加密的轻量级多密钥协议，用于联邦学习中的私有平均聚合，既摆脱了单密钥方案对不共谋假设的依赖，又避免了多方同态加密方案中协作解密所需的大方差噪声带来的高开销。

    

    联邦学习使多个客户端能够在保持各自本地数据集隔离的同时共同训练一个共享模型。然而，交换的模型更新仍可能泄露敏感信息，这使得私有聚合成为实际部署中的核心构建模块，尤其是在跨机构（cross-silo）场景下。同态加密天然契合联邦学习中客户端—聚合服务器的通信模式，但传统的单密钥部署依赖于较强的“不共谋”假设。多方同态加密消除了这一限制，不过近期在受限解密访问条件下的攻击要求在协作解密时使用大方差的smudging噪声，这显著增加了密文大小和实现复杂度。在本工作中，我们提出了基于RLWE同态加密的、用于私有平均聚合的轻量级多密钥协议。我们的构造有别于通常的多方……

    arXiv:2609.01945v1 Announce Type: cross  Abstract: Federated Learning enables multiple clients to train a shared model while keeping their local datasets isolated. However, the exchanged model updates may still leak sensitive information, making private aggregation a central building block in practical deployments, especially in the cross-silo setting. Homomorphic Encryption naturally fits the client--aggregator communication pattern of Federated Learning, but conventional single-key deployments rely on strong non-collusion assumptions. Multiparty Homomorphic Encryption removes this limitation, although recent attacks under restricted decryption access require large-variance smudging noise during collaborative decryption, which significantly increases ciphertext size and implementation complexity. In this work, we propose lightweight multi-secret-key protocols for private average aggregation based on RLWE-based Homomorphic Encryption. Our construction departs from the usual multiparty 
    
[^115]: 稀疏读出棱镜（SRP）：用特征而非词元来解释Logit-Lens分数

    Sparse Readout Prism: Explaining Logit-Lens Scores in Features Instead of Tokens

    [https://arxiv.org/abs/2609.01936](https://arxiv.org/abs/2609.01936)

    该论文提出稀疏读出棱镜（SRP），仅利用读出矩阵自身的权重将其分解为稀疏“读出特征”，把logit-lens分数解释为特征贡献之和，从而消除了透镜读数对拟合语料库的依赖（语料库条件性），并支持跨词元、上下文、层与透镜的比较。

    

    语言模型对下一个词元的预测是跨层逐步形成的，而“透镜”方法通过将中间隐藏状态解码为词元来追踪这一过程。但透镜的读数同时反映了隐藏状态以及用于解码的读出矩阵。许多透镜是在语料库上拟合的，我们证明：仅在拟合语料库上不同的两个透镜，会对相同的隐藏状态报告不同的词元。我们将这种依赖性称为“语料库条件性”。为了独立于拟合语料库来考察读出结构，我们提出了稀疏读出棱镜（Sparse Readout Prism, SRP），它仅使用读出矩阵自身的权重对其进行分解，并将任意词元的logit或logit差表示为稀疏读出特征贡献之和。这使读出特征成为透镜读数的一种全新分析单元，揭示出词元身份可能掩盖的结构，并支持跨词元、上下文、层与透镜之间的比较。

    arXiv:2609.01936v1 Announce Type: cross  Abstract: A language model's prediction of its next token develops across layers, and lens methods track this process by decoding intermediate hidden states into tokens. But a lens reading reflects both the hidden state and the readout (the unembedding matrix) used to decode it. Many lenses are fit on a corpus, and we show that two lenses differing only in their fitting corpus can report different tokens for the same hidden states. We call this dependence corpus conditionality. To examine readout structure independently of the fitting corpus, we introduce Sparse Readout Prism (SRP), which decomposes the readout using only its weights and expresses any token logit or logit difference as a sum of contributions from sparse readout features. This reveals readout features as a new unit of analysis for lens readings, exposing structure that token identities can obscure and enabling comparisons across tokens, contexts, layers, and lenses. Replacing the
    
[^116]: OR-Transformer：将实时决策扩展至1000件商品规模

    OR-Transformer: Scaling Real-Time Decision-Making to 1,000 Items

    [https://arxiv.org/abs/2609.01933](https://arxiv.org/abs/2609.01933)

    OR-Transformer通过商品置换等变的Transformer架构和路径梯度训练的深度强化学习方法，将随机需求下的联合补货决策扩展到1024种商品规模，随规模增长持续优于MILP基线并大幅降低在线决策时间。

    

    现代供应链运营可能需要在相关随机需求、异质提前期和共享固定订货成本的条件下，协调数千种异质商品的补货，从而产生超过10⁴维的观测空间。在这种规模下，滚动时域随机混合整数线性规划（MILP）变得极其缓慢，而标准强化学习（RL）方法在高维动作空间中面临日益严峻的信用分配挑战。我们提出了OR-Transformer，这是一个面向随机需求下联合补货问题的深度强化学习框架，采用商品置换等变的Transformer架构，并通过库存动态进行路径梯度训练。在规模高达1,024个库存商品的问题上，随着规模的增长，OR-Transformer越来越多地超越基于学习的方法和滚动时域MILP基线。它还将在线决策时间降低了超过（原文摘要在此处被截断）

    arXiv:2609.01933v1 Announce Type: new  Abstract: Modern supply chain operations can require coordinating replenishment across thousands of heterogeneous items under correlated stochastic demand, heterogeneous lead times, and shared fixed ordering costs, yielding observation spaces exceeding $10^4$ dimensions. At this scale, rolling-horizon stochastic mixed-integer linear programs (MILPs) become prohibitively slow, while standard reinforcement learning (RL) methods face increasingly challenging credit assignment in high-dimensional action spaces. We introduce OR-Transformer, a deep reinforcement learning framework for joint replenishment under stochastic demand, with an item-permutation-equivariant Transformer architecture and pathwise-gradient training through the inventory dynamics. Across problem sizes up to 1,024 inventory items, OR-Transformer increasingly outperforms learning-based and rolling-horizon MILP baselines as scale grows. It also reduces online decision-making time by ov
    
[^117]: CRISP：悬崖感知的输入自适应稀疏预填充与基于结构质量驱动的路由

    CRISP: Cliff-awaRe Input-adaptive Sparse Prefilling with Structural-Mass-Motivated Routing

    [https://arxiv.org/abs/2609.01925](https://arxiv.org/abs/2609.01925)

    该论文提出CRISP方法，用直接从代理注意力图结构中读取路由决策的结构代理C_struct替代JSD路由，解决了动态稀疏注意力路由中的两个结构性挑战，实现了长上下文LLM推理的高效输入自适应稀疏预填充。

    

    长上下文大语言模型（LLM）推理中的注意力预填充阶段计算复杂度呈二次方增长，使自注意力成为严重的计算瓶颈。传统的稀疏注意力方法通过固定模式或离线分析来缓解这一问题，但缺乏适应输入相关注意力结构的灵活性。近期的动态方法通过实时将注意力头路由到稀疏模式来解决这一问题，但其依赖于带有额外开销的间接路由代理，且其预算分配机制忽略了softmax之后的质量层级。我们提出了CRISP（Cliff-awaRe Input-adaptive Sparse Prefilling，悬崖感知的输入自适应稀疏预填充），该方法识别并解决了这一动态路由范式中的两个结构性挑战。首先，我们证明了路由决策可以直接从代理注意力图的结构中读取。我们用C_struct取代了Jensen-Shannon散度（JSD）路由，C_struct是一种结构代理，用于测量垂直-斜线兼容位置处的注意力质量，并能重现……（摘要在此处截断）

    arXiv:2609.01925v1 Announce Type: cross  Abstract: The attention prefilling phase of long-context LLM inference scales quadratically, making self-attention a severe computational bottleneck. Traditional sparse attention methods mitigate this through fixed patterns or offline profiling, but lack the flexibility to adapt to input-dependent attention structure. Recent dynamic methods address this by routing heads to sparse patterns in real-time, but rely on indirect routing proxies with overhead and budget allocation mechanisms that overlook the post-softmax mass hierarchy. We present CRISP (Cliff-awaRe Input-adaptive Sparse Prefilling), which identifies and addresses two structural challenges in this dynamic routing paradigm. First, we show that the routing decision can be read directly off the structure of the proxy attention map. We replace the Jensen-Shannon Divergence (JSD) routing with C_struct, a structural proxy that measures mass at Vertical-Slash compatible positions and reprodu
    
[^118]: 储层计算中动力学记忆的吸引域几何结构与可靠召回

    Basin Geometry and Reliable Recall of Dynamical Memories in Reservoir Computing

    [https://arxiv.org/abs/2609.01914](https://arxiv.org/abs/2609.01914)

    该研究揭示了储层计算联想记忆的吸引域具有“章鱼状”几何结构，线索驱动的广义同步能够绕过触手区域的不可预测性、将系统引入鲁棒的吸引域头部，从而实现动力学记忆的可靠召回，并给出了最小线索时长、同步速率与吸引域头部半径之间的定量关系。

    

    可靠的吸引子召回通常需要宽阔的吸引域。然而，在基于储层计算的联想记忆中，尽管吸引域被不可预测的、类似迷宫般的区域所主导，时间线索仍能可靠地恢复动力学记忆。我们揭示了记忆吸引域呈现出一种“章鱼状”结构：靠近吸引子处有一个鲁棒的“头部”，以及贯穿整个状态空间的纤细且相互缠绕的“触手”。位于触手区域的初始状态会产生接近于零的不确定性指数，使得召回的记忆在有限精度下实际上是不可预测的。然而，由线索驱动的广义同步绕过了这种不可预测性，将系统驱动到鲁棒的吸引域头部。这一机制得出了一个将最小线索时长、同步速率和吸引域头部半径联系起来的定量关系。经过训练的循环神经网络也表现出类似的几何结构，表明这一现象并不仅限于储层计算。

    arXiv:2609.01914v1 Announce Type: cross  Abstract: Reliable attractor recall conventionally requires broad basins of attraction. However, in reservoir-computing based associative memory, temporal cues reliably recover dynamical memories despite basins dominated by unpredictable, riddled-like regions. We reveal that memory basins exhibit an ``octopus-like'' structure: a robust ``head'' near the attractor and thin, intertwined ``tentacles'' spanning state space. Initial states in tentacular regions yield near-zero uncertainty exponents, making the recalled memory effectively unpredictable at finite precision. Yet, cue-driven generalized synchronization bypasses this unpredictability, driving the system into the robust basin head. This mechanism yields a quantitative relation linking minimum cue duration, synchronization rate, and basin-head radius. Trained recurrent neural networks exhibit similar geometry, suggesting this phenomenon extends beyond reservoir computing.
    
[^119]: OutageDiT：面向停电预测与情景模拟的生成式基础模型

    OutageDiT: A Generative Foundation Model for Power Outage Forecasting and Scenario Simulation

    [https://arxiv.org/abs/2609.01896](https://arxiv.org/abs/2609.01896)

    OutageDiT是一个基于全美停电与天气数据训练的生成式基础模型，能以15分钟分辨率生成七天停电轨迹，在单一模型中统一实现停电点预测、不确定性量化和条件情景模拟。

    

    停电规划需要在事件发生之前获得情景，这些情景必须在呈现规模、发生时间和持续时间不确定性的同时，保留时间上的依赖关系。然而，严重停电事件非常罕见，任何单一地区的数据中都几乎没有极端停电与恢复模式的样本。为应对这一挑战，我们提出了OutageDiT，一个用于以15分钟分辨率生成七天停电轨迹的基础模型，该模型基于全美范围内的停电和天气记录进行训练。具体而言，条件编码器在每次预测中对历史上下文和已知的未来协变量仅处理一次，随后浅层流（flow）解码器复用所得到的与预测时域对齐的状态来生成完整轨迹。所得样本能够在同一个深度生成模型中支持点预测、不确定性量化和条件事件模拟。在停电预测基准测试中，OutageDiT提升了预测……

    arXiv:2609.01896v1 Announce Type: cross  Abstract: Power-outage planning requires scenarios before an event occurs. These scenarios must represent uncertainty in magnitude, timing, and duration while preserving temporal dependence. However, severe events are rare, and data from any single region contain few examples of extreme outage and restoration patterns. To address this challenge, we introduce OutageDiT, a foundation model for generating seven-day outage trajectories at quarter-hour resolution, trained on outage and weather records across the United States. Specifically, a condition encoder processes the historical context and known future covariates once per forecast, and a shallow flow decoder reuses the resulting horizon-aligned states to generate complete trajectories. The resulting samples support point forecasting, uncertainty quantification, and conditional event simulation within one deep generative model. Across outage forecasting benchmarks, OutageDiT improves forecast a
    
[^120]: 用于激发态化学的潜在统一平滑哈密顿量

    Latent unified smooth Hamiltonians for excited state chemistry

    [https://arxiv.org/abs/2609.01871](https://arxiv.org/abs/2609.01871)

    该论文提出一种通过间接学习电子态哈密顿量的潜在隐式基组表示来统一建模分子基态与激发态（包括锥形交叉点和非绝热耦合）的神经网络架构，并在胸腺嘧啶和偶氮苯光化学体系上准确再现了基态与低激发态的能量及振子强度。

    

    我们描述了一种旨在模拟任意分子体系电子基态与激发态的神经网络架构及其训练方法。通过间接学习电子态哈密顿量的潜在隐式基组表示，该模型能够统一处理多个电子态、锥形交叉点以及非绝热耦合。该形式体系还可进一步扩展，以学习其他算符（例如跃迁偶极矩）的一致潜在表示。为展示该架构的通用能力，我们在两个真实光化学体系——胸腺嘧啶和偶氮苯——上训练并评估了神经网络。所得模型准确地再现了与这些体系光化学相关的基态和低激发态的能量与振子强度。我们通过研究关键分子几何构型来展示所训练网络的性能。

    arXiv:2609.01871v1 Announce Type: cross  Abstract: We describe a neural network architecture and training procedure designed to model electronic ground and excited states of arbitrary molecular systems. By indirectly learning a latent, implicit basis representation of the electronic-state Hamiltonian, the model offers a unified treatment of multiple electronic states, conical intersections, and non-adiabatic couplings. The formalism can be further extended to learn consistent latent representations of additional operators such as transition dipole moments, for example. To demonstrate the general capabilities of our architecture, we train and evaluate networks on two realistic photochemical systems, thymine and azobenzene. The resulting models accurately reproduce energies and oscillator strengths for the ground- and low-lying excited states relevant to the photochemistry of these systems. We highlight the performance of the trained networks by studying critical molecular geometries, in
    
[^121]: 按需导入：学习何时以及如何利用外部知识增强电子健康记录图

    Import What You Need: Learning When and How to Augment EHR Graphs with External Knowledge

    [https://arxiv.org/abs/2609.01839](https://arxiv.org/abs/2609.01839)

    提出ReTA框架，利用强化学习在每次就诊时动态选择知识图谱增强动作（软导入、硬导入或跳过），以预算感知的方式解决现有EHR图增强方法固定且与上下文无关的拓扑增强问题。

    

    基于电子健康记录（EHR）的纵向预测受到患者轨迹稀疏性和不规则性的限制，而利用外部知识图谱（KG）进行知识增强为缓解这些问题提供了一种有前景的方法。然而，现有的大多数方法执行固定的、与上下文无关的拓扑增强，即无论患者的动态状态如何变化，都添加相同的KG节点和边。我们提出了ReTA，一个基于强化学习的动态拓扑增强框架，它将KG导入建模为每次就诊的、预算感知的策略。ReTA首先构建一个离线优化的、基于KG的模板池，然后学习一个策略，在每次就诊时从三个选项中选择一个增强动作：软导入，在不修改图拓扑的情况下丰富节点特征；硬导入，将一个紧凑的KG子图嫁接到就诊图上以创建消息传递捷径；以及跳过，当基础（摘要在此处截断）……在此，我输出以上内容。

    arXiv:2609.01839v1 Announce Type: cross  Abstract: Longitudinal prediction from electronic health records (EHRs) is limited by the sparsity and irregularity in patient trajectories, and knowledge augmentation with external knowledge graphs (KGs) offers a promising way to alleviate these issues. However, most existing methods perform fixed, context-agnostic topology augmentation by adding the same KG nodes and edges regardless of a patient's evolving state. We propose ReTA, a Reinforcement learning-based dynamic Topology Augmentation framework that casts KG import as a per-visit, budget-aware policy. ReTA first constructs an offline refined pool of KG-grounded templates, then learns a policy to select one augment action per visit from three options: Soft Import, which enriches node features without modifying graph topology, Hard Import, which grafts a compact KG subgraph onto the visit graph to create message-passing shortcuts, and Skip, which leaves the visit unaugmented when the base 
    
[^122]: 大语言模型中用于抑郁症的可解释症状向量

    Interpretable Symptom Vectors for Depression in a Large Language Model

    [https://arxiv.org/abs/2609.01832](https://arxiv.org/abs/2609.01832)

    该研究通过机制可解释性技术发现大语言模型内部在第21层对抑郁症状产生几何分离，并构建“症状向量”将文本投影后得到各症状系数，其能保留临床医生标注的严重程度排序，从而增强LLM在抑郁症评估中的临床可信度。

    

    抑郁症患者呈现出多样化的症状特征，然而临床实践中通常将这种差异简化为单一的严重程度评分。大语言模型（LLMs）有潜力从患者的言语中捕捉各种症状及其严重程度。然而，抑郁症状在LLM内部如何表示仍然知之甚少，这限制了临床信任度。为了检验模型内部激活是否与临床医生的判断相符，我们使用机制可解释性技术分析了Gemma-3-27B-PT的残差流。通过记录来自经过验证的临床量表的多种症状描述的激活，我们发现在多个距离度量下，症状组在第21层的几何分离最为显著。随后，我们使用语义投影方法，将留出的自然语言文本投影到由这些临床量表构建的症状向量（Symptom Vectors）上。所得到的每个症状的系数保留了临床医生标注的严重程度排序。

    arXiv:2609.01832v1 Announce Type: cross  Abstract: Patients with depression present with diverse symptom profiles, yet clinical practice routinely reduces this variation to a single severity score. Large language models (LLMs) can potentially capture various symptoms and their severity from patient speech. However, how depressive symptoms are represented inside LLMs remains poorly understood, limiting clinical trust. To examine whether internal model activations match clinician judgment, we analyzed the residual stream of Gemma-3-27B-PT using mechanistic interpretability techniques. Recording activations across symptom descriptions drawn from validated clinical instruments, we found that symptom groups geometrically separated the most at layer 21 across multiple distance metrics. Using Semantic Projection, we then projected held-out naturalistic text onto Symptom Vectors constructed from these instruments. The resulting per-symptom coefficients preserved clinician-annotated rank orderi
    
[^123]: 强化学习选择优化器

    Reinforcement learning to choose optimizers

    [https://arxiv.org/abs/2609.01811](https://arxiv.org/abs/2609.01811)

    该论文提出将优化算法选择建模为序贯决策问题，利用循环强化学习策略在运行中动态决定下一步使用哪个优化器（涵盖基于梯度与无导数两类）及其持续时间，并在每次切换时传递当前最优解与代表性步长，从而摆脱了现有方法对切换时机、组合范围和决策频率的预设限制。

    

    arXiv:2609.01811v1 公告类型：cross 摘要：没有任何单一的优化方法能够在所有问题上都始终表现最佳，且最合适的优化器选择可能在一次运行过程中发生变化。现有的在运行过程中更换优化器的方法通常会预先固定策略的一部分：候选组合被限制在单一算法类别内，切换仅在固定时刻发生一次，或者将决策频率视为超参数而非可学习的量。我们提出“强化学习选择优化器”，将优化算法的选择建模为一个序贯决策问题。在每次决策时，一个循环策略读取当前的运行状态，同时决定下一步应使用哪个优化器以及使用多长时间。候选组合同时包含基于梯度和无导数的优化器，并且每次切换都会传递当前最优解和一个代表性步长。一个上下文代理对作用于专家头部的门控网络进行条件化，训练采用……（原文摘要在此处截断）

    arXiv:2609.01811v1 Announce Type: cross  Abstract: No single optimization method is uniformly best for all problems, and the most suitable optimizer choice can change during a run. Existing approaches that change optimizer during execution typically predetermine part of the strategy: the portfolio is restricted to one algorithm class, the switch occurs once at a fixed time, or the frequency of decisions is treated as a hyperparameter rather than a learned one. We introduce "Reinforcement Learning to Choose Optimizers", which formulates the optimization algorithm choice as a sequential decision-making problem. At each decision, a recurrent policy reads the current run state and decides both which optimizer should be used next and for how long. The portfolio includes both gradient-based and derivative-free optimizers, and each switch passes on the current best solution and a representative step size. A context proxy conditions a gating network over expert heads, and training employs a de
    
[^124]: hLLM：面向生成式重排序的单遍解码

    hLLM: Single Pass Decoding for Generative Reranking

    [https://arxiv.org/abs/2609.01807](https://arxiv.org/abs/2609.01807)

    提出hLLM，通过轻量自注意力头从LLM预填充隐状态读取项目-位置得分矩阵，并用匈牙利算法求最优二分匹配，在O(1)次前向传播内一次性解码全部N个序数，从而将生成式重排序的解码从逐token自回归生成变为常数次前向传播，且天然保证输出为有效排序。

    

    arXiv:2609.01807v1 公告类型： cross 摘要：大语言模型（LLM）实现了最先进的生成式排序质量，但其产生的排序结果必须经过解码，而自回归解码每生成一个token就需要一次顺序前向传播。我们观察到，排序器必须输出的token仅仅是N个序数值，用于按排序顺序命名各个项目，而这种狭窄的、具有置换结构的输出格式使得我们可以采用比从左到右生成高效得多的解码策略。我们提出了hLLM（匈牙利LLM），一种针对该输出格式专门设计的解码策略，它能够在O(1)次前向传播中解码全部N个序数。hLLM通过一个轻量级的自注意力头，从LLM预填充阶段的隐状态中读取一个N×K的项目-位置得分矩阵，然后利用匈牙利算法将该矩阵的最优二分匹配作为序数解码，从而在构造层面（而非通过事后修复）保证输出是一个有效的置换。通过对训练信号的系统性研究……

    arXiv:2609.01807v1 Announce Type: cross  Abstract: Large language models (LLMs) achieve state-of-the-art generative ranking quality, but the ranking they produce must be decoded, and autoregressive decoding spends one sequential forward pass per emitted token. We observe that the only tokens a ranker must emit are the $N$ ordinal values naming the items in ranked order, and that this narrow, permutation-structured output format admits decoding strategies which are much more efficient than left-to-right generation. We introduce hLLM (Hungarian LLM), a format-specialized decoding strategy that decodes all $N$ ordinals in $O(1)$ forward passes. hLLM reads an $N \times K$ item-position score matrix off the LLM's prefill hidden states with a lightweight self-attention head, then decodes the ordinals as the optimal bipartite assignment of that matrix via the Hungarian algorithm, yielding a valid permutation by construction rather than by repair. Through a systematic study of training signals
    
[^125]: D-FROST：基于最优传输的去中心化联邦提示调优，面向非独立同分布与不平衡数据

    D-FROST: Decentralized Federated pRompt-tuning via Optimal tranSporT for Non-IID and Imbalanced Data

    [https://arxiv.org/abs/2609.01802](https://arxiv.org/abs/2609.01802)

    该论文首次研究了去中心化联邦学习中的提示调优，提出基于最优传输的D-FROST算法，通过在提示测度上进行Wasserstein优化来捕捉提示的集合值结构，解决了非独立同分布和不平衡数据下提示集无法按索引对齐的难题，并具备达成共识的理论保证。

    

    提示调优（Prompt tuning）提供了一种参数高效的基础模型（FMs）适配方式，它通过冻结预训练的主干网络，仅更新少量可学习的提示。这一特性使提示调优特别适合去中心化联邦学习（DFL），因为在DFL中交换完整模型更新的通信代价可能高得令人望而却步。然而，DFL中的提示调优带来了新的挑战：从异构本地数据学习到的提示集可能无法按索引对齐，使得标准的去中心化平均方法不再适用；此外，算法还应有理论保证能够达成共识并向共同目标取得进展。在本工作中，我们首次对DFL中的提示调优进行了研究。我们将去中心化提示调优表述为在提示测度上基于Wasserstein距离的优化问题，该表述能够刻画提示的集合值结构。随后，我们提出了D-FROST，一种基于最优传输（OT）的去中心化（摘要在此截断）

    arXiv:2609.01802v1 Announce Type: new  Abstract: Prompt tuning provides a parameter-efficient way to adapt foundation models (FMs) by freezing the pretrained backbone and updating only a small set of learnable prompts. This property makes prompt tuning especially suitable for decentralized federated learning (DFL), where exchanging full-model updates can be prohibitively expensive. However, prompt tuning in DFL introduces new challenges. Prompt sets learned from heterogeneous local data may not be index-wise aligned, making standard decentralized averaging unsuitable. In addition, the algorithm should be theoretically guaranteed to achieve consensus and make progress toward the shared objective. In this work, we provide the first study of prompt tuning in DFL. We formulate decentralized prompt tuning as a Wasserstein-based optimization problem over prompt measures, which captures the set-valued structure of prompts. We then propose D-FROST, an optimal-transport-based (OT-based) decentr
    
[^126]: 十种架构，同一错误：空间不相交评估下高光谱分类的共同失效模式

    Ten Architectures, One Error: Shared Failure Modes in Hyperspectral Classification under Spatially Disjoint Evaluation

    [https://arxiv.org/abs/2609.01786](https://arxiv.org/abs/2609.01786)

    该论文提出一种将空间分离与模型感受野关联的无泄漏评估协议，揭示随机像素划分导致准确率虚高的问题——十种高光谱分类架构在此协议下Macro-F1平均下降0.147且模型排名最多变动五位。

    

    高光谱图像分类目前仍严重依赖在单一场景内进行随机像素划分。随机划分的Salinas数据集是比较不同架构时最常用的数据集之一。然而，在随机划分方法下，大量测试像素与训练像素直接相邻，这会夸大报告的准确率。本工作引入了一种无泄漏评估协议，将空间分离距离与模型的感受野关联起来。将该协议应用于十种不同的架构（涵盖经典方法、光谱、光谱-空间、Transformer、视觉骨干网络和状态空间等系列）后发现，Macro-F1平均下降0.147，模型排名变化最多达五位。此外，无泄漏评估限制了哪些架构可以在给定基准上进行测试。由于每个划分仅支持有限半径内的图像块，需要将此半径与（摘要在此处截断）

    arXiv:2609.01786v1 Announce Type: cross  Abstract: Hyperspectral image classification still relies heavily on random pixel splits within a single scene. The Salinas dataset, randomly split, is among the most widely used datasets for comparing different architectures. However, under a random split method, a large fraction of test pixels fall immediately adjacent to a training pixel, which inflates reported accuracy. This work introduces a leakage-free evaluation protocol linking spatial separation to the model's receptive field. Applying this protocol across ten different architectures, including classical, spectral, spectral-spatial, transformer, vision-backbone, and state-space families, shows that Macro-F1 drops by 0.147 on average and model rankings change by as many as five places. Furthermore, leakage-free evaluation limits which architectures can be tested on a given benchmark. Since each partition supports patches only within a finite radius, reporting this radius alongside the 
    
[^127]: 人工神经网络中纤维化、压缩与对称性破缺的出现

    Emergence of Fibrations, Compression, and Symmetry Breaking in Artificial Neural Networks

    [https://arxiv.org/abs/2609.01768](https://arxiv.org/abs/2609.01768)

    深度神经网络在学习过程中会自发涌现图论中的覆盖对称性，利用该对称性可将模型压缩至原大小的17%，而受控破坏该对称性则能克服持续学习中的可塑性丧失问题。

    

    人工神经网络通常被视为强大却不透明的黑箱。在本研究中，我们证明了深度神经网络的学习过程会产生图论中被称为纤维化与覆盖的局部对称性。我们证明覆盖对称性是随机梯度下降的稳定吸引子。与这一理论相一致，我们报告了覆盖对称性在主要网络架构中的涌现现象，包括多层网络、卷积网络、循环网络以及Transformer网络。利用这些对称性可以实现大幅度的模型压缩——将网络缩减至原始大小的17%而不损失性能。此外，对覆盖对称性进行受控破坏能够克服可塑性丧失的问题，在持续学习中取得了最先进的性能。这些理论结果为基于对称性的AI系统提供了新的基础，能够将黑箱转化为可解释的彩色图，并实现更高效的（原文在此截断）。

    arXiv:2609.01768v1 Announce Type: new  Abstract: Artificial neural networks are often regarded as powerful yet opaque black boxes. Here, we demonstrate that learning in deep neural networks generates local symmetries known in graph theory as fibrations and coverings. We prove that covering symmetries are stable attractors of stochastic gradient descent. Consistent with this theory, we report the emergence of covering symmetries across major network architectures, including multilayer, convolutional, recurrent, and transformer networks. Exploiting these symmetries enables drastic model compression - reducing networks to 17% of their original size without sacrificing performance. Furthermore, controlled breaking of covering symmetry overcomes the loss of plasticity, achieving state-of-the-art performance in continual learning. The theoretical results provide a new foundation for AI systems based on symmetries that convert black boxes into interpretable colored graphs and enable more effi
    
[^128]: 迈向可解释且具备政策感知能力的碳信用价格预测人工智能：面向新兴碳市场的研究框架

    Toward Explainable and Policy-Aware AI for Carbon Credit Price Prediction: A Research Framework for Emerging Carbon Markets

    [https://arxiv.org/abs/2609.01765](https://arxiv.org/abs/2609.01765)

    本文提出融合市场数据与政策文本的可解释碳价预测框架EPA-CarbonNet，但实证结果大多为负面——其预测精度不及随机游走基线，仅方向准确率（58.6%）优于所有基线。

    

    碳市场为排放定价，然而这一价格仍然难以预测。该领域的研究主要集中于欧盟和中国的碳交易体系，将监管文本压缩为情绪分数，并在缺乏校准或解释稳定性分析的情况下报告准确率。我们将十个反复出现的研究缺口提炼为一个影响-可行性矩阵，并提出EPA-CarbonNet——一种六层架构，通过交叉注意力将市场序列与政策文本融合，同时提供校准区间和政策归因解释。随后，我们基于十一年每日标普碳指数数据构建并测试了该模型。研究结果在很大程度上是负面的，并按实际测量结果如实报告：随机游走模型在五天RMSE上优于该模型（0.0365对0.0475），SHAP排序在重采样背景下的一致性仅为rho = 0.54，且政策注意力从未与已记录的监管事件相吻合。方向准确率为58.6%，超过了所有基线模型。

    arXiv:2609.01765v1 Announce Type: new  Abstract: Carbon markets put a price on emissions, yet that price remains hard to forecast. Work in this area clusters on the EU and Chinese schemes, compresses regulatory text into a sentiment score, and reports accuracy without calibration or explanation stability. We distil ten recurring gaps into an impact-feasibility matrix and propose EPA-CarbonNet, a six-layer architecture that fuses market series with policy text by cross-attention and calibrated intervals alongside policy-attributed explanations. We then build and test it on eleven years of daily S and P carbon index data. The findings are largely negative, and reported as measured: a random walk beats the model on five-day RMSE (0.0365 against 0.0475), SHAP rankings agree at rho = 0.54 across resampled backgrounds, and policy attention never coincides with documented regulatory events. Directional accuracy, at 58.6 percent, leads every baseline. Code, data documentation and all result ar
    
[^129]: 延迟老虎机中的池化与漂移

    Pooling and Drift in Delayed Bandits

    [https://arxiv.org/abs/2609.01761](https://arxiv.org/abs/2609.01761)

    该论文发现当延迟老虎机的反馈结果仅通过动作所产生的状态依赖于动作时，学习代价由有效维度（真正不同的状态数量）而非动作数量决定，并据此证明了 $\widetilde{O}(\sqrt{(d+1)V\log K})$ 等新的遗憾界，突破了以往随动作数增长的界限。

    

    系统常常不得不在得知行动是否奏效之前就采取行动：推荐系统几秒内就能观察到点击，而购买则要几天后才能看到。在 $K$ 个动作、延迟 $d$ 轮的设定下，$T$ 轮内已知的最优遗憾率为 $\widetilde{O}(\sqrt{(K+d)T})$，因此可选动作越多，学习代价就越高。但事实未必如此：如果结果仅通过动作所产生的状态依赖于动作，那么一个迟到的结果就能为所有可能产生该观测状态的动作提供信息，此时代价由动作产生的真正不同的状态数量决定，而非动作的数量。我们用介于 $1$ 与状态总数之间的有效维度 $v_t$ 来度量这一概念，并针对任意预先固定的预算证明：对于轮转算法，遗憾率为 $\widetilde{O}(\sqrt{(d+1)V\log K})$；对于实践中常用的单副本算法，遗憾率为 $\widetilde{O}(\sqrt{V^{-}}+\sqrt{dT})$；合并相似状态可以进一步降低这一代价。

    arXiv:2609.01761v1 Announce Type: cross  Abstract: A system often has to act long before it learns whether the act worked: a recommender sees a click in seconds and a purchase in days. With $K$ actions and a delay of $d$ rounds, the best rate known for this setting is $\widetilde{O}(\sqrt{(K+d)T})$ over $T$ rounds, so a longer menu is always more expensive to learn from. It need not be: if the outcome depends on the action only through the state it produced, then one late outcome informs every action that could have produced the observed state, and the price is set by how many genuinely different states the actions produce rather than by how many actions there are. We measure this using an effective dimension $v_t$ between $1$ and the number of states, and prove $\widetilde{O}(\sqrt{(d+1)V\log K})$ for a rotating algorithm and $\widetilde{O}(\sqrt{V^{-}}+\sqrt{dT})$ for the single-copy algorithm used in practice, for any budget fixed in advance; merging similar states lowers the price 
    
[^130]: 干摩擦与静摩擦条件下开环控制的条件扩散模型研究

    A Study of Conditional Diffusion Models for Open-Loop Control under Dry Friction and Stiction

    [https://arxiv.org/abs/2609.01756](https://arxiv.org/abs/2609.01756)

    本文提出的动作扩散方法利用条件一维U-Net生成时间连贯的有界控制序列，在干摩擦与静摩擦的开环控制基准中显著降低了终端误差和卡滞步数，尤其在低样本场景下优于随机采样和交叉熵方法。

    

    扩散模型近来已成为规划与控制中富有表达力的生成式先验。本文研究了动作扩散，这是一种动作序列扩散形式，被用作具有干摩擦和静摩擦特性的质点系统的开环提议分布。在该基准测试中，只有当施加的输入超过静摩擦阈值时运动才会开始，因此有效控制在动作序列空间中占据一个小且具有时间结构的子集。一个紧凑的条件一维U-Net根据初始状态和目标状态生成有界的控制序列。我们将其与均匀随机采样、来自相同结构化数据集先验的随机采样以及交叉熵方法（CEM）进行了比较。结果表明，动作扩散降低了终端误差和卡滞步数，尤其是在低样本情况下。这些结果表明条件扩散为生成时间上连贯的控制提供了一种有效机制。

    arXiv:2609.01756v1 Announce Type: new  Abstract: Diffusion models have recently emerged as expressive generative priors for planning and control. This paper studies Action Diffusion, an action-sequence diffusion formulation used as an open-loop proposal distribution for a point-mass system with dry friction and stiction. In this benchmark, motion starts only when the applied input exceeds a static-friction threshold, so effective controls occupy a small and temporally structured subset of the action-sequence space. A compact conditional 1D U-Net generates bounded control sequences conditioned on initial and target states. We compare it with uniform random shooting, random shooting from the same structured dataset prior, and the Cross-Entropy Method (CEM). Results show that Action Diffusion reduces terminal error and stuck steps, especially in low-sample regimes. These results indicate that conditional diffusion provides an effective mechanism for generating temporally coherent control 
    
[^131]: CAT-Flow：面向流匹配的曲率自适应步长方法

    CAT-Flow: Curvature-Adaptive sTeps for Flow Matching

    [https://arxiv.org/abs/2609.01746](https://arxiv.org/abs/2609.01746)

    本文提出了两种无需训练的轻量级算法CAT-OV和CAT-OT，利用流匹配采样与梯度流之间的新颖联系，通过估计向量场曲率在推理时自适应调整步长，从而在不增加额外神经网络求值的前提下提升采样效率与质量。

    

    流匹配已成为生成建模领域的领先框架，为FLUX和Stable Diffusion 3.5等最先进的系统提供支撑。然而，其基于ODE的采样过程的迭代特性造成了根本性的效率瓶颈：生成样本的质量对步长的选择高度敏感，且当前模型通常需要20至30步才能获得良好的生成质量。在本工作中，我们提出了两种轻量级、无需训练的算法——CAT-OV和CAT-OT，它们基于流匹配采样与梯度流之间的一种新颖联系，在推理阶段自适应地调整步长。我们的算法计算高效，无需额外的神经网络函数求值。具体而言，CAT-OT通过向量场时间导数的有限差分近似来估计时间维度上的曲率，而CAT-OV则通过向量场的梯度来近似状态空间上的曲率。在适当的条……（摘要原文截断）

    arXiv:2609.01746v1 Announce Type: new  Abstract: Flow Matching has emerged as a leading framework for generative modeling, powering state-of-the-art systems such as FLUX and Stable Diffusion 3.5. However, the iterative nature of its ODE-based sampling process creates a fundamental efficiency bottleneck: the quality of generated samples is highly sensitive to the choice of step-sizes, and current models typically require 20 to 30 steps for good quality. In this work, we propose two lightweight, training-free algorithms, CAT-OV and CAT-OT that adapt step-sizes at inference time based on a novel connection between Flow Matching sampling and gradient flow. Our algorithms are computed efficiently by not requiring additional neural function evaluations. Specifically, CAT-OT estimates curvature over time via a finite-difference approximation of the time-derivative of the vector field, while CAT-OV approximates curvature over the state space via a gradient of the vector field. Under suitable c
    
[^132]: 通过智能体原生可复用工具原语实现LLM工具使用中的Harness工程

    Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives

    [https://arxiv.org/abs/2609.01736](https://arxiv.org/abs/2609.01736)

    提出以自然语言取代API模式作为工具调用接口的“工具原语”设计，并构建包含25,519个函数的集中式仓库ToolFace供LLM在推理时动态检索工具，从而解决多步多轮推理脆弱及大规模工具目录下性能退化的问题。

    

    增强了外部工具的大型语言模型（LLM）在解决复杂现实任务方面已展现出卓越能力。然而，现有方法面临两个关键挑战：由工具输出类型和API模式不兼容导致的脆弱的多步与多轮推理，以及在大规模工具目录下的性能下降。为解决这些问题，我们提出了**工具原语**，这一设计以自然语言作为工具调用的接口，取代了僵化的基于API模式的调用方式，其中每个工具都被封装了一个LLM接口，在内部处理模式解析与执行，从而实现工具之间的自然通信，支持嵌套和多轮工具调用。基于工具原语，我们构建了**ToolFace**，一个包含25,519个函数的集中式仓库，LLM可以在推理时从中动态检索仅相关的工具，从而无需枚举原始API模式……（摘要原文在此处被截断）

    arXiv:2609.01736v1 Announce Type: cross  Abstract: Large language models (LLMs) augmented with external tools have demonstrated remarkable capability in solving complex real-world tasks. However, existing approaches suffer from two key challenges: brittle multi-step and multi-turn reasoning caused by incompatible tool output types and API schemas, and performance degradation under large tool catalogues. To address these, we introduce \textbf{Tool Primitives}, a design that replaces rigid API schema-based invocation with natural language as the interface for tool calling, where each tool is wrapped with an LLM interface that handles schema resolution and execution internally, enabling natural inter-tool communication for nested and multi-turn tool calling. Building on Tool Primitives, we host \textbf{ToolFace}, a centralized repository of 25,519 functions from which LLMs dynamically retrieve only the relevant tools at inference time, eliminating the need to enumerate raw API schemas in 
    
[^133]: RecKAN：具有可学习递归多项式基的Kolmogorov-Arnold网络

    RecKAN: Kolmogorov-Arnold Networks with a Learnable Recursive Polynomial Basis

    [https://arxiv.org/abs/2609.01729](https://arxiv.org/abs/2609.01729)

    RecKAN的核心创新在于通过一个五系数可学习的二阶多项式递推关系来定义KAN网络的基函数本身，该递推可涵盖多种经典多项式族作为特例，使学到的基能够超越任何固定的经典多项式选择。

    

    Kolmogorov–Arnold网络（KAN）用每条边上可学习的单变量函数取代了标准网络的固定标量权重，但现有变体仍然固定了这些函数所依赖的基：B样条、切比雪夫多项式、小波或雅可比多项式，并且只学习其上的组合权重。我们提出了RecKAN，它转而通过一个二阶多项式递推关系 R_{n+1}(x) = (ax²+bx+c)R_n(x) + (dx+e)R_{n-1}(x) 来定义基本身，其五个系数与网络进行联合学习。我们证明该递推关系可以将多种经典多项式族（包括两类切比雪夫多项式、斐波那契、佩尔和雅各布斯塔尔多项式）作为特例恢复出来，并证明其次数恰好在包含所有这些多项式的子族中随n线性增长，从而具体地表明所学习的基能够超越任何固定的经典选择。在多个基准（摘要在此处截断）

    arXiv:2609.01729v1 Announce Type: cross  Abstract: Kolmogorov--Arnold Networks (KANs) replace the fixed scalar weights of a standard network with learnable univariate functions on each edge, but existing variants still fix the \emph{basis} that those functions are built from: B-splines, Chebyshev polynomials, wavelets, or Jacobi polynomials, and learn only the combination weights over it. We introduce RecKAN, which instead defines the basis itself by a second order polynomial recurrence, $R_{n+1}(x) = (ax^2+bx+c)R_n(x) + (dx+e)R_{n-1}(x)$, whose five coefficients are learned jointly with the network. We show this recurrence recovers several classical polynomial families including both kinds of Chebyshev polynomials, Fibonacci, Pell, and Jacobsthal polynomials as special cases, and prove that its degree grows linearly in $n$ exactly on the sub-family containing all of them, giving a concrete sense in which the learned basis can move beyond any fixed classical choice. Across multiple ben
    
[^134]: 听见低语：针对微调TTS模型的黑盒成员推断攻击

    Hearing the Whispers: Black-Box Membership Inference Attacks on Finetuned TTS Models

    [https://arxiv.org/abs/2609.01723](https://arxiv.org/abs/2609.01723)

    该论文提出了首个专门针对微调TTS模型的黑盒成员推断攻击框架，解决了查询生成与语音表示工程中的独特挑战，以揭示个性化语音合成中的隐私风险。

    

    文本转语音（TTS）基础模型越来越多地在私人数据集上进行微调，以合成高度个性化的语音，这通过暴露生物特征身份和敏感语音内容带来了严重的隐私风险。现有的黑盒成员推断攻击（MIA）遵循查询生成和表示工程的两阶段流水线，而这两者在适配到TTS时都面临独特的挑战。对于查询生成而言，合成文本与参考语音的双重条件控制创造了一个庞大且尚未被充分探索的查询设计空间，且目前尚无识别有效查询的既定标准。对于表示工程而言，语音的多层次特性及其时间变异性使得低层表示和直接对比方法不足以捕捉成员信号。为了应对这些挑战，我们提出了首个明确针对TTS模型设计的黑盒成员推断攻击框架。

    arXiv:2609.01723v1 Announce Type: cross  Abstract: Text-to-Speech (TTS) foundation models are increasingly fine-tuned on private datasets to synthesize highly personalized voices, introducing severe privacy risks by exposing both biometric identities and sensitive speech content. Existing black-box membership inference attacks (MIAs) follow a two-stage pipeline of query generation and representation engineering, both of which face unique challenges when adapted to TTS. For query generation, dual conditioning on synthesis text and reference speech creates a large and underexplored query design space with no established criterion for identifying an effective query. For representation engineering, the multi-level speech characteristics and temporal variability of speech make low-level representations and direct comparisons inadequate for capturing membership signals. To address these challenges, we present the first black-box MIA framework explicitly tailored to TTS models at both the spe
    
[^135]: 具有解析方差调度的生成式扩散代理模型

    Generative Diffusion Surrogates with Analytical Variance Schedule

    [https://arxiv.org/abs/2609.01705](https://arxiv.org/abs/2609.01705)

    本文提出将输运问题中已知的解析方差（均方位移）的时间导数作为扩散模型的加噪速率，使生成时间成为经过校准的物理输运时钟，从而构建适用于随机输运的概率性、时间可分辨且能刻画非高斯分布结构的代理模型。

    

    随机输运描述的是这样一类物理系统：初始结构化的分布在未解析的强迫、散射或非均匀介质作用下发生扩散。对此类系统有用的代理模型应当具有概率性、时间可分辨性，并能表示非高斯的分布结构。生成式扩散模型——通过高斯噪声破坏数据并学习一个反向流以恢复结构化状态——恰好具备这些特性。然而，其噪声调度通常是启发式选择的：图像与音频生成——即其典型应用场景——并不提供物理时钟。相比之下，在输运问题中，即使完整分布未知，方差（即均方位移）往往可以从宏观理论或经验标度律中获得。本文将该方差的时间导数规定为正向加噪速率，从而使生成时间成为一个经过校准的输运时钟。方差路径由构造方式强制保证……

    arXiv:2609.01705v1 Announce Type: new  Abstract: Stochastic transport describes physical systems in which an initially structured distribution spreads under unresolved forcing, scattering, or heterogeneous media. Useful surrogates for such systems should be probabilistic, time-resolved, and able to represent non-Gaussian distributional structure. Generative diffusion models, which corrupt data with Gaussian noise and learn a reverse flow back to structured states, have these properties. Their noise schedules, however, are usually chosen heuristically: image and audio generation---the canonical use cases---provide no physical clock. In transport, by contrast, the variance, or mean-square displacement, is often known from macroscopic theory or empirical scaling even when the full distribution is not. Here we prescribe the forward noising rate as the time derivative of this variance, turning generative time into a calibrated transport clock. The variance path is enforced by construction, 
    
[^136]: 基于三频段信道测量的面向太赫兹无线数据中心的多层数字孪生

    Tri-Band Channel Measurement-Enabled Multi-Layer Digital Twin for Terahertz Wireless Data Centers

    [https://arxiv.org/abs/2609.01699](https://arxiv.org/abs/2609.01699)

    本文提出一种由140、220、300 GHz三频段信道测量校准的自底向上多层（物理、信道、评估、操控）数字孪生框架，用于太赫兹无线数据中心的高效无线规划与实时优化。

    

    人工智能计算的快速增长推动了对灵活、大容量数据中心互连的需求。太赫兹（THz）通信凭借其超宽带宽和高空间复用能力，已成为未来无线数据中心的一种有前景的解决方案，而数字孪生（DT）则能够实现高效的无线规划和实时优化。本工作提出了一种面向太赫兹无线数据中心的测量驱动的多层数字孪生框架，其中物理层、信道层、评估层和操控层自底向上逐步构建。首先，在140、220和300 GHz频段开展了广泛的信道测量，以表征频率相关的传播特性。基于三频段测量结果，通过联合优化几何、材料、天线和混合传播模型，建立了经测量校准的物理孪生体。在物理孪生体之上，一条视线……（摘要不完整，后续内容截断）

    arXiv:2609.01699v1 Announce Type: new  Abstract: The rapid growth of AI computing has driven increasing demands for flexible and high-capacity data-center interconnections. Owing to its ultra-wide bandwidth and high spatial reuse capability, terahertz (THz) communication has emerged as a promising solution for future wireless data centers, while digital twins (DTs) enable efficient wireless planning and real-time optimization. In this work, a measurement-driven multi-layer DT framework is proposed for THz wireless data centers, where the physical, channel, evaluation, and manipulation layers are progressively constructed from bottom to top. First, extensive channel measurements are conducted at 140, 220, and 300 GHz to characterize frequency-dependent propagation behaviors. Based on the tri-band measurements, a measurement-calibrated physical twin is established by jointly optimizing the geometry, material, antenna, and hybrid propagation models. On top of the physical twin, a line-of-
    
[^137]: FairLens：面向高风险决策的视觉语言模型公平性基准测试

    FairLens: Benchmarking Fairness in Vision-Language Models for High-Stakes Decision-Making

    [https://arxiv.org/abs/2609.01691](https://arxiv.org/abs/2609.01691)

    该论文提出了FairLens基准框架，在招聘、法律和医疗三大高风险领域用超过10万个图像-问题对评估八个视觉语言模型，发现其主要缺陷是基于人脸做出无根据的推断，而非对不同群体的不平等对待。

    

    视觉语言模型（VLM）正越来越多地被用于基于视觉输入做出决策。我们提出了FAIRLENS，这是一个用于衡量视觉语言模型在招聘、法律和医疗保健三个高风险领域中响应的公平性与有效性的基准测试和评估框架。FAIRLENS将涵盖性别、种族和年龄群体的真实人脸图像与封闭式及开放式问题相配对，为每个模型提供超过10万个图像-问题对，并从四个互补的视角评估模型响应：不良结果率上的人口统计均等性、合理性、对问题中未支持的角色和身份的人口统计关联，以及自由文本生成中的偏见。合理性是核心的有效性标准：当响应遵循问题中所述的证据，并在图像无法支持答案时选择不作答，则该响应是合理的。通过对八个视觉语言模型的评估，我们发现模型的主要失败模式是无根据的推断，而非不平等的对待。模型经常……（摘要在此处被截断）

    arXiv:2609.01691v1 Announce Type: cross  Abstract: Vision-language models (VLMs) are increasingly used to make decisions from visual inputs. We introduce FAIRLENS, a benchmark and evaluation framework for measuring both the fairness and the validity of VLM responses in three high-stakes domains: hiring, legal, and healthcare. FAIRLENS pairs real face images spanning gender, race, and age groups with closed- and open-ended questions, giving more than 100K image-question pairs per model, and evaluates responses from four complementary views: demographic parity over adverse outcome rates, soundness, demographic association over unsupported roles and statuses, and bias in free-text generation. Soundness is the central validity criterion: a response is sound when it follows the evidence stated in the question and abstains when the image cannot support an answer. Evaluating eight VLMs, we find that the primary failure is unwarranted inference rather than unequal treatment. Models routinely i
    
[^138]: FORGE：面向微控制器上纯整数视觉模型的仅前向传播测试时自适应方法

    FORGE: Forward-Only Test-Time Adaptation for Integer-Only Vision Models on Microcontrollers

    [https://arxiv.org/abs/2609.01683](https://arxiv.org/abs/2609.01683)

    提出FORGE方法，首次实现仅前向传播的测试时自适应在微控制器上已部署的BN折叠纯整数卷积网络中运行，其核心是通过将折叠卷积的逐通道输出重新归一化到干净训练统计量，恢复因BN融合而丢失的自适应能力。

    

    部署在微控制器（MCU）上的视觉模型被量化为纯整数运算，并在仅支持推理的运行时环境中执行，这些运行时不包含反向传播所需的机制——而反向传播正是使模型适应实际部署中所遇分布偏移（传感器噪声、模糊、光照变化）的标准工具。现有的仅前向传播测试时自适应（TTA）方法要么只能在服务器级或边缘GPU级模型上运行（并非真正的微控制器整数执行），要么依赖于整数部署过程中已被融合消除的批归一化（BN）层。我们提出了一种可在已部署的、BN已折叠的、纯整数卷积网络上运行的仅前向传播TTA方法。关键观察在于：将BN融合进前序卷积（这是整数推理的必要步骤）会破坏基于归一化的自适应方法所依赖的统计量。我们通过将每个折叠卷积的逐通道输出重新归一化到其干净训练时的统计量来恢复自适应能力，

    arXiv:2609.01683v1 Announce Type: cross  Abstract: Vision models deployed on microcontrollers (MCUs) are quantized to integer-only arithmetic and run in inference-only runtimes that do not carry the machinery backpropagation needs: the standard tool for adapting a model to the distribution shift (sensor noise, blur, lighting) it meets in the field. Existing forward-only test-time adaptation (TTA) methods either run only on server- or edge-GPU-class models (not true microcontroller integer execution), or require the batch-normalization (BN) layers that integer deployment fuses away. We present a forward-only TTA method that operates on deployed, BN-folded, integer-only convolutional networks. The key observation is that fusing BN into the preceding convolution, a mandatory step for integer inference, destroys the statistics that normalization-based adaptation relies on. We restore adaptation by re-normalizing each folded convolution's per-channel output to its clean training statistics,
    
[^139]: 住宅光伏-电池储能社区中基于强化学习与规则的对等电力交易定价

    Reinforcement Learning and Rule-Based Peer-to-Peer Pricing in Residential PV-BES Communities

    [https://arxiv.org/abs/2609.01680](https://arxiv.org/abs/2609.01680)

    在住宅光伏社区对等电力交易定价中，仅光伏配置下基于规则的定价优于最佳强化学习策略，但结合电池储能后最佳强化学习策略可将社区节约资金从734.23欧元提升至978.52欧元，且SDR形定价在学习型模式中表现最佳。

    

    本文比较了住宅光伏社区中对等（P2P）电力交易的基于规则和基于学习的定价机制。基于规则的基准方法包括作为事后分配机制的账单分摊、中间市场汇率定价以及供需比（SDR）定价。强化学习（RL）方案通过深度Q网络实现，并在基于乘数的定价和可学习的SDR形定价下进行评估，同时以固定参数的SDR变体作为非学习对照。性能评估通过社区节约资金以及补充性的财务和运营指标进行。在仅光伏的基础配置中，基于规则的基准方法优于最佳的强化学习策略。在配置电池储能的条件下（仅针对强化学习策略进行评估），最佳强化学习策略下的社区节约资金从734.23欧元增加到978.52欧元。在各种基于学习的模式和两种配置中，SDR形定价表现均优于其他方法。

    arXiv:2609.01680v1 Announce Type: new  Abstract: This paper compares rule-based and learning-based pricing mechanisms for peer-to-peer (P2P) electricity trading in residential photovoltaic communities. The rule-based benchmarks comprise bill-sharing as an ex post allocation mechanism, the mid-market rate, and supply-demand-ratio pricing. The reinforcement-learning (RL) formulation is implemented through a Deep Q-Network and evaluated under multiplier-based and learnable SDR-shaped pricing, with a fixed-parameter SDR variant as a non-learning control. Performance is assessed through community savings together with complementary financial and operational indicators. In the base PV-only configuration, the rule-based benchmarks outperform the best RL policy. With battery energy storage, evaluated for the RL policies only, community savings under the best RL policy increase from EUR 734.23 to EUR 978.52. Across the learning-based modes and in both configurations, SDR-shaped pricing outperfo
    
[^140]: 自我改进的测试时智能综述：推理阶段的反馈驱动适应、学习与扩展

    A Survey on Self-Improving Test-Time Intelligence: Feedback-Driven Adapting, Learning, and Scaling at Inference

    [https://arxiv.org/abs/2609.01679](https://arxiv.org/abs/2609.01679)

    本综述提出“反馈驱动的测试时智能（TTI）”作为统一框架，将测试时适应、测试时学习和测试时扩展三大方向联系起来，系统阐释了AI模型如何利用测试时反馈和额外计算在部署阶段实现自我改进。

    

    人工智能系统在部署过程中改进自身行为的能力正变得日益重要。随着推理不再局限于对固定训练模型的静态执行，越来越多的研究工作探索模型如何通过利用测试时信息和额外的计算来实时优化自身行为。这些发展主要沿着两个方向演进：一类方法利用测试时信号修改模型的状态，另一类方法通过额外的推理时资源（如更多采样和工具使用）来改进预测。然而，这些方向往往由不同的研究社区以不同的术语分别研究，导致它们之间的联系难以被看清。在本综述中，我们提出反馈驱动的测试时智能作为理解此类部署时改进的统一视角。我们利用这一视角将测试时适应、测试时学习和测试时扩展联系起来，突出……

    arXiv:2609.01679v1 Announce Type: new  Abstract: The ability of AI systems to improve their behavior during deployment is becoming increasingly important. As inference moves beyond the static execution of a fixed trained model, a growing body of work studies how models can refine their behavior on the fly by exploiting test-time information and additional computation. These developments have largely evolved along two directions: methods that modify the model's state using test-time signals, and methods that improve predictions through extra inference-time resources such as more sampling and tool use. However, these directions are often studied in separate communities with different terminology, making their connections harder to see. In this survey, we present feedback-driven Test-Time Intelligence (TTI) as a unified perspective for understanding such deployment-time improvement. We use this view to relate test-time adaptation, test-time learning, and test-time scaling, highlighting bo
    
[^141]: Sim2Signal：面向交通信号控制的仿真到现实基准测试

    Sim2Signal: Sim-to-Real Benchmarks for Traffic Signal Control

    [https://arxiv.org/abs/2609.01676](https://arxiv.org/abs/2609.01676)

    提出Sim2Signal基准，将交通信号控制中的仿真到现实差距分解为观测、动作、转移和奖励四类差距并逐一单独诱导，从而为系统评估缓解方法提供了标准化测试平台。

    

    强化学习在仿真环境中能够实现强大的交通信号控制性能，然而在仿真器中训练的策略一旦部署到现实世界往往会失效，这种失败被称为“仿真到现实差距”。当强化学习应用于交通信号控制时，这种差距来源于多个方面：感知、动作执行、交通动态以及控制目标。它们的相对影响以及现有仿真到现实缓解方法的可靠性仍缺乏充分理解，且该领域一直缺乏一个能够系统衡量这种差距并评估缓解方法的标准基准。我们提出了Sim2Signal，该基准将仿真到现实差距分解为观测差距、动作差距、转移差距和奖励差距，分别对应底层马尔可夫决策过程四个组成部分的不匹配，并在共享协议下单独诱导每种差距。我们在2个基础控制器上评估了18种缓解方法，涵盖33种差距设置和10个校准场景……

    arXiv:2609.01676v1 Announce Type: new  Abstract: Reinforcement learning achieves strong traffic signal control performance in simulation, yet policies trained in simulators often fail once deployed in the real world, a failure known as the Sim-to-Real gap. When RL is applied to traffic signal control, this gap arises from several sources: sensing, action execution, traffic dynamics, and the control objective. Their relative impact and the reliability of existing Sim-to-Real mitigation methods remain insufficiently understood, and the field lacks a standard benchmark for systematically measuring the gap and evaluating mitigation methods. We present Sim2Signal, a benchmark that decomposes the Sim-to-Real gap into observation, action, transition, and reward gaps, corresponding to mismatches in the four components of the underlying MDP, and induces each gap in isolation under a shared protocol. We evaluate 18 mitigation methods on 2 base controllers, across 33 gap settings and 10 calibrate
    
[^142]: 随机森林驱动的元胞自动机用于大规模野火蔓延建模

    Random Forest-Informed Cellular Automaton for Large-Scale Wildfire Spread Modelling

    [https://arxiv.org/abs/2609.01675](https://arxiv.org/abs/2609.01675)

    该论文提出一个融合随机森林与元胞自动机的三阶段框架，利用随机森林估计每日火灾发生概率并与邻域驱动蔓延规则结合，在5公里网格的大规模野火模拟中相比纯元胞自动机基线取得了显著更高的空间重叠度。

    

    准确的大规模野火蔓延建模需要能够同时捕捉与火灾发生相关的环境条件和火势传播的局部动态的模型。我们提出了一个将随机森林（RF）模型与元胞自动机（CA）相结合的三阶段框架。首先，在2021年加拿大火灾季数据上训练的RF模型估计每日像素级别的火灾发生概率。其次，分位数梯度提升模型为敏感性分析提供可选的蔓延速率先验。第三，RF驱动的CA在5公里网格上将RF概率层与邻域驱动的蔓延相结合。RF模型在2022至2024年数据集上取得了0.725至0.795的AUC值，而RF驱动的CA在2023年模拟中实现了比所评估的纯元胞自动机基线显著更高的空间重叠度。更高分辨率的模拟为局部空间误差提供了额外的定性评估。

    arXiv:2609.01675v1 Announce Type: cross  Abstract: Accurate large-scale wildfire spread modelling requires models that capture both the environmental conditions associated with fire occurrence and the local dynamics of fire propagation. We propose a three-stage framework that combines a Random Forest (RF) model with a cellular automaton (CA). First, an RF model trained on the 2021 Canadian fire season estimates daily pixel-level fire-occurrence probabilities. Second, quantile gradient boosting models provide optional spread-rate priors for sensitivity analysis. Third, an RF-informed CA combines the RF probability layer with neighbourhood-driven spread on a 5 km grid. The RF model achieved AUC values of 0.725--0.795 on the 2022--2024 datasets, while the RF-informed CA achieved substantially higher spatial overlap than the evaluated CA-only baselines in the 2023 simulation. A higher-resolution simulation provides an additional qualitative assessment of local spatial errors. These results
    
[^143]: CliffRank：一种用于活性悬崖排序预测的双分支框架

    CliffRank: A Dual-Branch Framework for Activity-Cliff Ranking Prediction

    [https://arxiv.org/abs/2609.01673](https://arxiv.org/abs/2609.01673)

    CliffRank 提出了一种将绝对活性回归与排序一致性学习相结合的双分支框架，通过成对偏好一致性损失（PPC）在抗菌肽和小分子的活性悬崖排序预测任务上取得了领先的性能。

    

    活性悬崖排序仍然是一个难题，因为局部结构的变化可能导致巨大的活性差异，而能够揭示其潜在机制的高质量数据仍然有限。为了更有效地利用现有的活性标签，我们将绝对活性回归与排序一致性学习相结合。CliffRank 使用均方误差损失、阈值化的列表损失以及成对偏好一致性损失（PPC）训练两个并行预测器，其中 PPC 在偏好概率空间中对齐相对排序关系。在三个抗菌肽数据集上，采用 ESM2-t12 的 CliffRank 取得了最高的平均 Spearman 相关系数 0.5393 和平均 Recall@50 为 21.4，尽管在不同数据集上的最优方法各不相同。在三个小分子数据集上，采用 PNA 并在 120 个 epoch 后激活 PPC 的 CliffRank 取得了最高的平均 Spearman 相关系数 0.6890，其平均 Recall@50 为 30.4，与 ACANet-PNA 相当。

    arXiv:2609.01673v1 Announce Type: cross  Abstract: Activity-cliff ranking remains difficult because local structural changes can cause large activity differences, while high-quality data that resolve the underlying mechanisms remain limited. To use available activity labels more effectively, we combine absolute-activity regression with ranking-consistency learning. CliffRank trains two parallel predictors with mean squared error, a thresholded listwise loss, and Pairwise Preference Consistency (PPC), which aligns relative ordering in the preference-probability space. On three antimicrobial peptide datasets, CliffRank with ESM2-t12 achieved the highest mean Spearman correlation of 0.5393 and mean Recall@50 of 21.4, although the leading method varied across individual datasets. On three small-molecule datasets, CliffRank with PNA, where PPC was activated after 120 epochs, achieved the highest mean Spearman correlation of 0.6890, while its mean Recall@50 of 30.4 matched that of ACANet-PNA
    
[^144]: 私密计算空间：面向农业的可信多集群联邦学习实践经验

    Private Computation Space: Experience with Trusted Multi-Cluster Federated Learning for Agriculture

    [https://arxiv.org/abs/2609.01667](https://arxiv.org/abs/2609.01667)

    本文提出了已部署的开源机器学习系统私密计算空间（PCS），通过多集群编排、异步联邦学习与差分隐私等技术，在保护农民数据隐私和身份的同时，应对农村基础设施脆弱的挑战，推动人工智能在农业中的安全应用。

    

    人工智能已被证明有助于改善农业实践，但其应用仍然有限：69%的美国农民对共享其数据存在隐私担忧，在推广应用之前必须解决这些问题。虽然联邦学习已在其他领域被证明可以大规模保护隐私，但为农业部署这样一个系统面临着一系列独特的挑战；这一问题需要一个既能保护农民数据和身份、又能保持模型效用的系统，该系统需能在普通商用硬件上运行，并对脆弱的农村基础设施具有韧性。为解决这些问题，我们提出了私密计算空间，这是一个已实际部署的开源机器学习系统，用于安全地配置和处理农民数据。我们设计了一个专为农业环境量身定制的系统，通过多集群编排确保在农村地区的可靠性，并结合异步联邦学习（FL）与差分隐私等技术。

    arXiv:2609.01667v1 Announce Type: cross  Abstract: Artificial Intelligence has shown to help improve agricultural practices, yet adoption remains limited: 69% of U.S. farmers have privacy concerns with sharing their data, and these concerns must be addressed before adoption is widespread. While Federated Learning has been demonstrated to protect privacy at scale for other sectors, deploying a system for agriculture comes with its own set of challenges; the problem necessitates a system that can protect farmer data and identities while preserving model utility, runs on commodity hardware, and is resilient to fragile rural infrastructure. To address these concerns, we introduce the Private Computation Space (PCS), a deployed, open-source Machine Learning system to provision and process farmer data securely. We design a system tailored to an agricultural setting, with multi-cluster orchestration for reliability in rural areas with asynchronous Federated Learning (FL), Differential Privacy
    
[^145]: 无需越狱的上下文推断攻击

    Context Inference Attacks Without Jailbreaks

    [https://arxiv.org/abs/2609.01663](https://arxiv.org/abs/2609.01663)

    该论文提出并形式化了“上下文推断攻击”这一新型隐私威胁，证明智能体AI系统即使在没有越狱、且存在指令限制、logit抑制和上下文稀释等防御措施的情况下，仍会泄露其通过自身工具调用静默加载的敏感上下文信息。

    

    智能体AI系统正日益被部署用于在推理时处理敏感数据，例如在系统回答问题之前，将医疗记录或财务文档组装成一个隐藏的上下文。先前的工作主要通过“越狱”攻击来研究隐私风险，这类攻击诱导模型直接泄露敏感内容，但在很大程度上忽视了智能体场景，即上下文是由智能体自身的工具调用组装而成的。我们表明，尽管我们测试了多种针对智能体的控制措施——包括不泄露上下文的指令、logit抑制以及上下文稀释——我们所评估的智能体仍然容易受到隐藏上下文泄露的攻击。例如，一个回答良性用户查询的网页浏览智能体，仍然携带着关于其上下文中被静默加载记录的可利用信号。我们通过一个安全博弈引入并形式化了“上下文推断攻击”，并在防御强度逐渐递减的三种设置下进行了评估……

    arXiv:2609.01663v1 Announce Type: cross  Abstract: Agentic AI systems are increasingly deployed to process sensitive data at inference time, such as healthcare records or financial documents assembled into a hidden \emph{context} before the system answers. Prior work has studied privacy risks primarily through \emph{jailbreaking} attacks that induce models to directly disclose sensitive content, but has largely overlooked the agentic setting where the context is assembled by the agent's own tool calls. We show that the agents we evaluate remain vulnerable to hidden-context leakage despite the controls we test against them, namely an instruction not to disclose the context, logit suppression, and context dilution. For instance, a web-browsing agent answering benign user queries still carries exploitable signals about records silently loaded into its context. We introduce and formalize \emph{context-inference attacks} through a security game and evaluate three settings under decreasing a
    
[^146]: 使用SciBERT的WASP-2025共享任务高效上下文受限望远镜文献目录分类

    Efficient Context-Limited Telescope Bibliography Classification for the WASP-2025 Shared Task Using SciBERT

    [https://arxiv.org/abs/2609.01647](https://arxiv.org/abs/2609.01647)

    本研究提出一种基于SciBERT的高效方法，在严格的512 token上下文限制和有限计算资源下，将望远镜相关科学论文自动分类为四个类别，以0.89的宏F1分数在WASP-2025共享任务排行榜中位列第一。

    

    望远镜文献目录的编制是评估天文台科学影响力并确保天文学研究可重复性的关键环节。这项任务涉及识别、分类和关联引用或使用特定望远镜的科学出版物。然而，这一过程在很大程度上仍然是人工操作且资源密集型的。在本工作中，我们提出了一种基于SciBERT的高效方法，可将科学论文自动分类为四个类别——科学研究、仪器设备、简单提及和非望远镜相关。尽管面临严格的上下文长度限制（最多512个token）和有限的计算资源，我们的方法仍取得了0.89的宏F1分数，在WASP-2025排行榜上名列榜首。我们分析了截断的影响，并表明即使有一半的样本超过了token限制，SciBERT的领域对齐特性也能实现稳健的分类。我们还讨论了截断、分块和长上下文方法之间的权衡。

    arXiv:2609.01647v1 Announce Type: new  Abstract: The creation of telescope bibliographies is a crucial part of assessing the scientific impact of observatories and ensuring reproducibility in astronomy. This task involves identifying, categorizing, and linking scientific publications that reference or use specific telescopes. However, this process remains largely manual and resource intensive. In this work, we present an efficient SciBERT-based approach for automatic classification of scientific papers into four categories - science, instrumentation, mention, and not telescope. Despite strict context-length constraints (maximum 512 tokens) and limited compute resources, our approach achieved a macro F1 score of 0.89, ranking at the top of the WASP-2025 leaderboard. We analyze the effect of truncation and show that even with half the samples exceeding the token limit, SciBERT's domain alignment enables robust classification. We discuss trade-offs between truncation, chunking, and long-c
    
[^147]: SocialBuddy：为社交场景量身定制的搜索智能体

    SocialBuddy: Tailoring Search Agent for Social Scenarios

    [https://arxiv.org/abs/2609.01641](https://arxiv.org/abs/2609.01641)

    该论文提出首个面向社交场景的智能体搜索框架SocialBuddy，并构建了包含20万用户画像、1000万社交帖子和5万推理轨迹的大规模模拟环境SocialEnv，以解决复杂社交搜索中的性能退化与稀疏奖励难题。

    

    在数字社交互动时代，从海量社交信息流中搜索朋友的帖子已成为用户的基本需求。然而，尽管现代智能体搜索框架在传统检索任务中取得了显著成功，但面对异构的用户查询和多维度的社交内容时，它们会陷入失效，导致在复杂社交搜索中出现严重的性能退化。为弥补这一差距，我们推出了SocialBuddy——首个专为社交场景量身定制的智能体搜索框架。具体而言，我们构建了SocialEnv，这是首个面向社交搜索的大规模模拟环境。借助自动化的数据与轨迹合成流水线，SocialEnv包含20万个用户画像、1000万条社交帖子和5万条推理轨迹，为社交搜索智能体的研发奠定了坚实基础。为解决社交搜索中稀疏奖励带来的信用分配难题，我们……

    arXiv:2609.01641v1 Announce Type: cross  Abstract: In the era of digital social interaction, searching friends' posts from massive social streams has become a fundamental user need. However, while modern agentic search frameworks have achieved remarkable success in conventional retrieval tasks, they break down when confronted with heterogeneous user queries and multi-dimensional social feeds, resulting in severe performance degradation in complex social search. To bridge this gap, we introduce SocialBuddy, the first agentic search framework tailored for social scenarios. Specifically, we construct SocialEnv, the first large-scale simulated environment for social search. Powered by an automated data and trajectory synthesis pipeline, SocialEnv includes 200K user profiles, 10 million social posts, and 50K reasoning trajectories, establishing a solid foundation for the development of social search agents. To tackle the credit assignment dilemma caused by sparse rewards in social search, w
    
[^148]: Omega-N：可解释的结构节点描述符及其适用域

    Omega-N: Interpretable Structural Node Descriptors and Their Applicability Domain

    [https://arxiv.org/abs/2609.01633](https://arxiv.org/abs/2609.01633)

    本文提出 Omega-N，通过局部化三角指标因子并结合配置零模型超额与多尺度个性化 PageRank 邻域两项修正，得到每节点十个仅依赖图结构、无需属性与训练的可解释描述符，并验证了全局标量对标谱基线、逐节点归因在结构中心数量未知时更具优势的理论预测。

    

    复合结构指标用一个数字概括整个网络；对于基于三角形的指标而言，它在谱上是冗余的：Tr(A^3) 就是邻接谱的三阶矩。非冗余的信息位于下一层级，即 diag(A^3)，它依赖于特征向量，无法由谱唯一确定。该指标族理论论文中的一个推论指出并预测：全局标量应当与经过锐化的谱基线相媲美而非超越它们，而逐节点归因方法则应在结构中心数量未知的情况下表现更优。本文对这一预测进行了验证。我们通过将四个因子逐一局部化来构建 Omega-N。直接的局部化方法条件数很差；来自已发表实践的两项修正解决了这一问题——即对每个局部因子引入配置零模型超额，以及在多个尺度上采用个性化 PageRank 邻域——从而仅凭图结构本身，为每个节点生成十个可解释的特征，无需节点属性、无需训练。

    arXiv:2609.01633v1 Announce Type: cross  Abstract: A composite structural index summarises a network in one number; for a triangle-based index it is spectrally redundant: Tr(A^3) is the third moment of the adjacency spectrum. The non-redundant content sits one level down, in diag(A^3), which depends on eigenvectors and is not spectrally determined. A corollary in the theory paper for this index family stated that, and predicted: the global scalar should tie sharpened spectral baselines rather than beat them, while the node-wise attribution should do better where the number of structural epicentres is unknown. This paper tests it.   We construct Omega-N by localizing each of the four factors. The direct localization is badly conditioned; two corrections from published practice fix it, a configuration-null excess for every local factor and a personalized-PageRank neighbourhood at several scales, giving ten interpretable features per node from the graph alone, with no attributes, training
    
[^149]: 电子商务赞助搜索中联合排序拍卖与固定价格商品列表的边际期望收益

    Marginal Expected Revenue for Jointly Ranking Auction and Fixed-Price Listings in E-Commerce Sponsored Search

    [https://arxiv.org/abs/2609.01628](https://arxiv.org/abs/2609.01628)

    该论文提出边际eCPM（meCPM）方法，将传统固定价格商品的期望收益估算框架扩展至价格动态演化的拍卖和“拍卖加一口价”（ABIN）商品，实现了电商赞助搜索中多种上架形式的联合排序。

    

    电子商务搜索排序在向相互竞争的商品列表分配曝光位时，必须平衡多个目标——相关性、用户参与度和平台收入。对于固定价格商品，估算期望收益部分的方法已经非常成熟，但当市场库存中包含混合上架形式（如纯拍卖和混合式“拍卖加一口价”（Auction with Buy It Now, ABIN）商品）时，估算就变得具有挑战性，因为此类商品的价格是动态演化的，在排序时最终成交价值尚不可知。然而，拍卖和ABIN商品在eBay等平台的库存和交易量中占有相当可观的份额，并且是个人卖家以及价值不明确的独特商品常用的上架形式。我们将标准的千次曝光期望成本（eCPM）框架扩展到拍卖和ABIN商品，推导出边际eCPM（meCPM），以刻画对价格仍在演化中的商品多展示一次曝光所带来的增量价值。

    arXiv:2609.01628v1 Announce Type: cross  Abstract: E-commerce search ranking must balance multiple objectives--relevance, user engagement, and platform revenue--when allocating impression slots to competing listings. Estimating the expected revenue component is well understood for fixed-price items, but becomes challenging when marketplace inventory includes mixed listing formats such as pure auctions and hybrid "Auction with Buy It Now" (ABIN) items, where prices evolve dynamically and the final transaction value is unknown at ranking time. Yet auction and ABIN listings account for a meaningful share of inventory and transaction volume on platforms such as eBay, and are a popular format for individual sellers and for unique items with unclear value. We extend the standard Expected Cost-per-Mille (eCPM) framework to auction and ABIN listings by deriving a marginal eCPM (meCPM) that captures the incremental value of showing one more impression of an item whose price is still evolving. T
    
[^150]: PRISM：面向自动驾驶交通系统主动安全的智能体多模型架构

    PRISM: An Agentic Multi-Model Architecture for Proactive Safety in Autonomous Transportation Systems

    [https://arxiv.org/abs/2609.01623](https://arxiv.org/abs/2609.01623)

    PRISM是一种智能体多模型安全架构，通过逆向事故概率建模将事故分类器转化为可解释的动态安全评分，并由推理层协调三个并发专门模型（轨迹运动学、环境风险、VRU交互），实现从被动事故规避到主动持续风险管理的转变。

    

    自动驾驶和智能交通系统在复杂的城市环境中运行，其安全性取决于车辆行为、环境条件以及行人、骑行者等弱势道路使用者（VRU）之间的交互。大多数高级驾驶辅助系统（ADAS）采用被动反应机制，只有在危险出现后才被激活——美国VRU死亡人数的持续上升凸显了这一关键局限。本研究提出了PRISM（Proactive Risk Intelligence and Safety Management，主动风险智能与安全管理），这是一种智能体多模型安全架构，实现了从被动的事故规避向主动、持续的风险管理的范式转变。PRISM采用逆向事故概率建模，将二分类事故分类器转换为动态、可解释的安全评分。三个专门化模型分别处理轨迹运动学、环境风险和VRU交互，三者并发运行，由一个融合……的推理层进行统一协调。

    arXiv:2609.01623v1 Announce Type: cross  Abstract: Autonomous and intelligent transportation systems operate in complex urban environments where safety depends on interactions among vehicle behavior, environmental conditions, and vulnerable road users (VRUs) such as pedestrians and cyclists. Most advanced driver assistance systems (ADAS) employ reactive mechanisms that activate only after hazards have emerged, a critical limitation underscored by rising VRU fatalities in the United States.   This study introduces PRISM (Proactive Risk Intelligence and Safety Management), an agentic multi-model safety architecture that transitions from reactive crash avoidance to proactive, continuous risk management. PRISM employs inverse crash-probability modeling to convert binary crash classifiers into dynamic, interpretable safety scores. Three specialized models addressing trajectory kinematics, environmental risk, and VRU interaction operate concurrently, coordinated by a reasoning layer incorpor
    
[^151]: RecEvolve：面向推荐系统的知识驱动自主智能体系统

    RecEvolve: A Knowledge-Driven Autonomous Agent System for Recommender Systems

    [https://arxiv.org/abs/2609.01622](https://arxiv.org/abs/2609.01622)

    本文提出知识驱动的自主智能体系统RecEvolve，将想法生成、代码实现、训练与评估的完整研究生命周期纳入持续自主闭环框架，首次在生产级大规模双塔召回模型上从零完成40余次自主训练迭代，突破隐藏架构瓶颈并取得NDCG约20%的相对提升，使线上用户满意度增长3.77%。

    

    智能体AI（Agentic AI）的兴起催生了向自迭代系统的转变，为生产级推荐模型的自主优化开辟了新的前沿。本文展示了一个知识驱动的自主智能体系统的实证验证，该系统被直接部署在生产级大规模双塔召回模型上。通过将完整的研究生命周期——涵盖想法生成、代码实现、离线训练和指标评估——委托给一个持续闭环的自主框架，该智能体系统从零开始成功执行了40余次完整的自主训练运行。在严格的生产规模评估下执行这些运行时，该系统系统地攻克了最新生产模型中隐藏的架构瓶颈，实现了NDCG约20%的相对提升这一突破性进展，该提升直接转化为线上生产流量中用户满意度增长3.77%。此外，部署（摘要在此处被截断）

    arXiv:2609.01622v1 Announce Type: cross  Abstract: The rise of agentic AI has catalyzed a shift toward self-iterating systems, opening new frontiers for the autonomous optimization of production recommender models. This paper presents the empirical validation of a knowledge-driven autonomous agent system, deployed directly on a production large-scale Two-Tower retrieval model. By delegating the entire research lifecycle, spanning idea generation, code implementation, offline training, and metric evaluation, to a continuous closed-loop autonomous framework, the agent system executed over 40 completed autonomous training runs from scratch. Executing these runs under rigorous production-scale evaluations, the system systematically navigated hidden architectural bottlenecks on the latest production model to achieve a breakthrough ~20% relative improvement in NDCG, a gain that translated directly to a +3.77% increase in user satisfaction in live production traffic. Furthermore, the deployme
    
[^152]: 当文献数据在材料发现中误导人工智能时

    When Literature Data Mislead Artificial Intelligence in Materials Discovery

    [https://arxiv.org/abs/2609.01621](https://arxiv.org/abs/2609.01621)

    该研究以固态电解质电导率数据为例，揭示了科学文献中普遍存在的文本-图表不匹配、单位不一致等看似合理却难以检测的错误，这些错误会以结构化标签噪声的形式污染AI训练数据库，甚至导致高达100倍的电导率误差。

    

    人工智能（AI）日益将科学文献作为数据来源，用于构建数据库、训练预测模型并指导科学发现。然而，源自文献的数据集通常假设已报道的实验值具有内在一致性并可直接重复使用。本文以固态电解质（SE）电导率数据作为材料科学的代表性案例来分析这一假设。通过追踪从原始论文到策展数据集的数值来源，我们发现了反复出现的文本与图表不匹配、坐标轴标注含糊、单位不一致以及测量背景信息缺失等问题。这些偏差在数值上往往看似合理，因此难以通过常规预处理检测到，但它们会在数据库构建和机器学习再利用过程中以结构化标签噪声的形式传播。一个跨数据库的示例表明，模糊的报道方式可造成高达100倍的电导率误差。我们的分析重新定义了数据……（摘要在此处被截断）

    arXiv:2609.01621v1 Announce Type: cross  Abstract: Artificial intelligence (AI) increasingly treats scientific literature as a data source for building databases, training predictive models, and guiding discovery. Yet literature-derived datasets often assume that reported experimental values are internally consistent and directly reusable. Here, we analyze this assumption using solid electrolyte (SE) conductivity data as a representative materials-science case. By tracing values from source articles to curated datasets, we identify recurrent text-figure mismatches, ambiguous axis annotations, unit inconsistencies, and missing measurement context. These discrepancies are often numerically plausible and therefore difficult to detect through routine preprocessing, but they can propagate as structured label noise during database construction and machine-learning reuse. A cross-database example shows how ambiguous reporting can create a 100-fold conductivity error. Our analysis reframes dat
    
[^153]: 面向电信SNOC环境高效云知识库检索的多智能体检索增强生成

    Multi-Agent Retrieval-Augmented Generation for Efficient Cloud Knowledge Base Search in Telecom SNOC Environment

    [https://arxiv.org/abs/2609.01618](https://arxiv.org/abs/2609.01618)

    本文提出了一个完全离线的多智能体RAG框架Athena，通过融合E5稠密检索、BM25稀疏检索和知识图谱扩展，并结合加权CombSUM融合与交叉编码器重排序，实现了电信SNOC环境中企业云知识库的高效精准检索。

    

    电信服务与网络运营中心（SNOC）依赖大量的云文档集合，包括标准操作程序（SOP）、供应商技术手册、事件报告和配置指南，以维持不间断的网络运营。在关键事件发生期间，工程师必须快速检索到准确的信息，然而传统的基于关键词和单阶段的检索方法往往难以提供精确的结果。本文提出了 Athena for Cloud Knowledge Base，这是一个完全离线的多智能体检索增强生成（RAG）框架，专为 Vodafone Idea 的 SNOC 环境中的企业云文档搜索而设计。该系统在基于 LangGraph 的编排框架中集成了使用 E5 Large V2 嵌入的稠密检索、BM25 稀疏检索以及知识图谱扩展。检索到的候选结果通过加权 CombSUM 进行融合，随后进行交叉编码器重排序和最大边际相关性处理。

    arXiv:2609.01618v1 Announce Type: cross  Abstract: Telecom Service and Network Operations Centers (SNOCs) rely on large collections of cloud documents, including Standard Operating Procedures (SOPs), vendor technical manuals, incident reports, and configuration guides, to maintain uninterrupted network operations. During critical incidents, engineers must quickly retrieve accurate information, yet traditional keyword based and single stage retrieval approaches often struggle to provide precise results.   This paper presents Athena for Cloud Knowledge Base, a fully offline, multi agent Retrieval Augmented Generation (RAG) framework designed for enterprise cloud document search in Vodafone Idea's SNOC environment. The system integrates dense retrieval using E5 Large V2 embeddings, BM25 sparse retrieval, and Knowledge Graph expansion within a LangGraph based orchestration framework. Retrieved candidates are fused using Weighted CombSUM, followed by cross encoder reranking and Maximal Marg
    
[^154]: 面向企业文档搜索的混合检索增强生成：知识图谱扩展、RRF融合与分块接地评估

    Hybrid Retrieval-Augmented Generation with Knowledge Graph Expansion, RRF Fusion, and Per-Chunk Grounded Evaluation for Enterprise Document Search

    [https://arxiv.org/abs/2609.01617](https://arxiv.org/abs/2609.01617)

    DocuSearch 提出了一种混合检索增强生成系统，通过倒数排名融合（RRF）将稠密向量语义检索、BM25全文搜索与知识图谱邻居扩展三种互补证据源加权融合（权重分别为0.50、0.35、0.15），在电信网络运营生产环境中实现了更准确、有据可依的企业文档问答。

    

    从大型企业文档库中获取准确、有据可依的答案是一个难题。仅依靠稠密向量检索在处理混合技术术语、供应商特定缩写词、或需要跨多个不相邻章节进行推理的查询时，往往表现不佳。DocuSearch 正是为解决这一差距而构建的——一个在电信网络运营生产环境中开发和评估的离线多智能体文档智能系统。DocuSearch 不依赖单一检索信号，而是整合三种互补的证据来源：基于 Qdrant 向量库并使用 BGE-Large 嵌入的语义搜索、基于 SQLite FTS5 索引的 BM25 全文搜索，以及来自结构化边表的知识图谱邻居扩展。这三份排序列表通过倒数排名融合进行合并，其中向量搜索的信号权重为 0.50，BM25 为 0.35，知识图谱为 0.15，使用……

    arXiv:2609.01617v1 Announce Type: cross  Abstract: Getting accurate, grounded answers out of large enterprise document repositories is a difficult problem. Dense vector retrieval alone frequently performs poorly on queries that mix technical terminology, vendor-specific acronyms, or require reasoning across several non-adjacent sections. DocuSearch was built to address exactly this gap - an offline, multi-agent document intelligence system developed and evaluated in a production telecom network operations environment. Rather than relying on a single retrieval signal, DocuSearch pulls together three complementary sources of evidence: semantic search over a Qdrant vector store using BGE-Large embeddings, BM25 full text search over an SQLite FTS5 index, and Knowledge Graph neighbour expansion from a structured edge table. These three ranked lists are merged through Reciprocal Rank Fusion with signal weights of 0.50 for vector search, 0.35 for BM25, and 0.15 for the knowledge graph, using 
    
[^155]: 提示空间元学习无法跨用户迁移：一项冻结大语言模型的负面结果

    Prompt-Space Meta-Learning Does Not Transfer Across Users: A Frozen-LLM Negative Result

    [https://arxiv.org/abs/2609.01615](https://arxiv.org/abs/2609.01615)

    本文提出 Muse 方法并得出负面结论：在冻结大语言模型上通过提示空间元学习获得的共享自适应提示无法跨用户迁移个性化能力，其收益主要来自通用指令质量而非真正的用户自适应。

    

    将冻结的大语言模型（LLM）个性化到单个用户，通常被表述为提示空间中的元学习问题：每个用户被视为一个任务，研究者寻求一种共享的自然语言自适应策略，该策略在给定用户少量标注交互的情况下，为该用户配置冻结模型。这一框架颇具吸引力，因为它与骨干模型无关，并且复用了提示优化的现有机制，然而该领域很少检验优化后的元目标所编码的究竟是可跨用户迁移的自适应能力，还是仅仅是一般性的指令质量。我们通过 Muse（Meta-learned User-adaptation via Shared Evolution，基于共享进化的元学习用户自适应）来研究这一问题：Muse 通过反思式提示进化在元训练用户群体上进化出一个单一的共享自适应提示，将其冻结，然后零样本应用于留出的用户；匹配的对照组将学习效应与措辞和选择带来的混淆因素分离开来。在两个标准的个性化基准（LaMP-2 分类……）上

    arXiv:2609.01615v1 Announce Type: new  Abstract: Personalizing a frozen large language model (LLM) to individual users is often framed as a meta-learning problem in prompt space: each user is a task, and one seeks a shared natural-language adaptation policy that, given a handful of the user's labeled interactions, configures the frozen model for that user. The framing is attractive because it is backbone-agnostic and reuses the machinery of prompt optimization, yet the field rarely tests whether the optimized meta-objective encodes transferable cross-user adaptation rather than generic instruction quality. We study this question with Muse (Meta-learned User-adaptation via Shared Evolution), which evolves a single shared adaptation prompt over a meta-train user population by reflective prompt evolution, freezes it, and applies it zero-shot to held-out users; matched controls isolate learning from confounds of phrasing and selection. On two standard personalization benchmarks (LaMP-2 cat
    
[^156]: DiDrive：一种用于自动驾驶安全离线强化学习的风险感知分层扩散框架

    DiDrive: A Risk-Aware Hierarchical Diffusion Framework for Safe Offline Reinforcement Learning in Autonomous Driving

    [https://arxiv.org/abs/2609.01609](https://arxiv.org/abs/2609.01609)

    DiDrive提出了一种融合风险感知分层扩散架构（RHDif）与3DICE策略优化范式的分布引导离线扩散框架，通过过滤状态冗余、聚焦安全关键威胁并缓解分布外动作高估，从而提升自动驾驶离线强化学习的安全性与稳定性。

    

    尽管扩散模型能够有效捕捉自动驾驶中的多模态行为先验，但离线强化学习（RL）策略仍然容易受到分布偏移、重尾风险信号、分布外（OOD）动作生成以及高维状态冗余等问题的影响。为了应对这些挑战，我们提出了DiDrive，一个分布引导的离线扩散框架，它包含两个协同组件：风险感知分层扩散（RHDif）架构和3DICE策略优化范式。在状态空间中，RHDif利用低层风险门控编码器和高层上下文调节器来过滤环境冗余，并聚焦于安全关键威胁。在动作空间中，3DICE通过样本内校准引导、时空优化以及基于集成的方法的候选动作排序，来缓解OOD动作的高估和梯度振荡问题。在CARLA基准上的评估结果表明，DiDrive（摘要在此处被截断）

    arXiv:2609.01609v1 Announce Type: new  Abstract: While diffusion models effectively capture multimodal behavioral priors for autonomous driving, offline reinforcement learning (RL) policies remain susceptible to distribution shift, heavy-tailed risk signals, out-of-distribution (OOD) action generation, and high-dimensional state redundancy. To address these challenges, we propose DiDrive, a distribution-guided offline diffusion framework featuring two synergistic components: the Risk-Aware Hierarchical Diffusion (RHDif) architecture and the 3DICE policy optimization paradigm. In the state space, RHDif utilizes a low-level risk-gated encoder and a high-level contextual modulator to filter environmental redundancy and focus on safety-critical threats. In the action space, 3DICE mitigates OOD overestimation and gradient oscillation through in-sample calibrated guidance, spatiotemporal optimization, and ensemble-based candidate ranking. Evaluations on the CARLA benchmark demonstrate DiDriv
    
[^157]: WMLLM：基于“预测后行动”世界建模的自进化优化智能体

    WMLLM: Self-Evolving Optimization Agents via Predict-Then-Act World Modeling

    [https://arxiv.org/abs/2609.01608](https://arxiv.org/abs/2609.01608)

    提出WMLLM自进化优化智能体框架，让大语言模型先预测有前景的优化方向再生成候选解，通过世界建模、多轮智能体改进、种群搜索与强化学习的结合，在黑盒优化中显著提升样本效率并持续改进其隐式世界模型与优化策略。

    

    黑盒优化问题因其搜索空间庞大、结构松散且维度极高而依然充满挑战。现有方法通常依赖直接的候选生成或试错式改进，因而样本效率较低。提升搜索效率的一个自然途径是引入世界建模，它能够在代价高昂的评估之前帮助识别有前景的优化方向。大语言模型凭借其隐含知识，能够以相当可观的准确度预测这些候选解的结果。受此启发，我们提出了WMLLM——一个基于“预测后行动”世界建模的自进化优化智能体框架。该智能体首先预测有前景的方向，然后采取行动生成候选解。结合智能体多轮迭代改进、基于种群的搜索以及强化学习，WMLLM能够在搜索过程中同时优化其隐式世界模型和自身的优化策略。

    arXiv:2609.01608v1 Announce Type: cross  Abstract: Black-box optimization problems remain challenging because of large, weakly structured, and high-dimensional search spaces. Existing methods often suffer from poor sample efficiency because they rely on direct candidate generation or trial-and-error refinement. A natural way to improve search efficiency is to use world modeling, which can help identify promising optimization directions before costly evaluation. Large language models can predict the outcomes of these candidates with nontrivial accuracy because of their implicit knowledge. Motivated by this observation, we propose WMLLM, a self-evolving optimization-agent framework based on predict-then-act world modeling. The agent first predicts promising directions and then acts to generate candidates. Combined with agentic multi-turn refinement, population-based search, and reinforcement learning, WMLLM refines both its implicit world model and its optimization strategy during search
    
[^158]: 基于熵的选择性智能体引导：从不完美的视觉语言模型教师中学习自主策略

    Selective Agent Guidance via Entropy: Learning Autonomous Policies from Imperfect VLM Teachers

    [https://arxiv.org/abs/2609.01567](https://arxiv.org/abs/2609.01567)

    该论文提出SAGE框架，仅在智能体不确定时才查询昂贵的视觉语言模型教师，并利用环境优势对教师建议进行加权蒸馏，从而训练出无需教师引导即可自主行动的轻量级强化学习策略。

    

    视觉语言模型为交互式决策提供了有用的先验知识，但直接将其用作策略既昂贵又脆弱：它们必须在每一步都被查询，无法通过环境交互得到改进，并且可能重复系统性错误。我们研究如何从一个在线、昂贵、不完美但具有信息量的视觉语言模型教师中学习一个廉价的自主策略。我们提出了SAGE（基于熵的选择性智能体引导），这是一个仅在学习者不确定时才查询视觉语言模型的框架，它在训练期间执行其建议的动作，并将引导蒸馏到一个轻量级的强化学习（RL）策略中。由于视觉语言模型的建议并不总是可靠的，SAGE可以使用由环境得出的优势来对教师动作蒸馏进行加权，而不是将所有建议视为同样有用。在稀疏奖励的视觉推理和导航任务中，SAGE学习到的策略在评估时无需视觉语言模型引导即可自主行动，并改进了……

    arXiv:2609.01567v1 Announce Type: new  Abstract: Vision-Language Models (VLMs) provide useful priors for interactive decision-making, but using them directly as policies is expensive and brittle: they must be queried at every step, do not improve from environment interaction, and can repeat systematic errors. We study how to learn a cheap autonomous policy from an online, expensive, and imperfect but informative VLM teacher. We propose SAGE (Selective Agent Guidance via Entropy), a framework that queries a VLM only when the learner is uncertain, executes the suggested action during training, and distills guidance into a lightweight Reinforcement Learning (RL) policy. Because VLM advice is not always reliable, SAGE can weight teacher-action distillation using environment-derived advantages rather than treating all suggestions as equally useful. Across sparse-reward visual reasoning and navigation tasks, SAGE learns policies that act without VLM guidance at evaluation time and improves o
    
[^159]: 重新思考离线数据驱动优化中的可学习性

    Rethinking Learnability in Offline Data-driven Optimization

    [https://arxiv.org/abs/2609.01493](https://arxiv.org/abs/2609.01493)

    本文针对PAC可学习性无法充分刻画离线优化的理论缺陷，提出了“算法依赖的可学习性”这一新概念，其只需保证在优化器轨迹上的精度即可支撑离线数据驱动优化。

    

    黑盒优化（BBO）已得到广泛应用，但随着现实世界中BBO问题日益复杂，进化算法和贝叶斯优化面临效率挑战。数据驱动优化通过从数据中学习来提升BBO算法的效率。离线数据驱动优化仅利用一组固定的历史评估来寻找高质量解，由于无需额外的在线评估而吸引了大量关注。尽管已提出众多离线优化方法，但一个根本问题仍未得到解答：什么样的可学习性对于离线优化是足够的？先前的理论研究表明，概率近似正确（PAC）可学习性是不够的，因为即使大多数区域被学习得很好，最优区域仍可能学习得很差。在本文中，我们提出了算法依赖的可学习性，它只要求在优化器的轨迹上具有精度

    arXiv:2609.01493v1 Announce Type: cross  Abstract: Black-Box Optimization (BBO) has found broad applications, but evolutionary algorithms and Bayesian optimization face efficiency challenges as real-world BBO problems grow increasingly complex. Data-driven optimization improves the efficiency of BBO algorithms by learning from data. Offline data-driven optimization seeks high-quality solutions using only a fixed set of previous evaluations, attracting substantial attention because it requires no additional online evaluations. Many offline optimization methods have been proposed, but a fundamental question remains unanswered: what learnability is sufficient for offline optimization? Prior theoretical studies show that Probably Approximately Correct (PAC) learnability is insufficient, as the optimal region may remain poorly learned even when most regions are well learned. In this paper, we propose algorithm-dependent learnability, which requires accuracy only on the optimizer's trajector
    
[^160]: 生产环境中的老虎机：推理时的超参数优化

    Bandits in Prod: Hyperparameter Optimization at Inference Time

    [https://arxiv.org/abs/2609.01335](https://arxiv.org/abs/2609.01335)

    该论文将生产系统中只能通过线上噪声反馈评估配置的场景形式化为在线超参数优化（OHPO），提出通用框架IMABO及免重启的无限多臂老虎机策略IMOSS，并给出了分位数遗憾的理论保证。

    

    许多生产系统只能通过将某个配置应用于实际线上请求并观察带噪声的反馈来评估该配置。现代智能体系统是一个突出的例子，其推理时的选择包括模型选择、检索深度、提示策略和解码温度等，但往往缺乏具有代表性的验证数据。我们将这一设置形式化为在线超参数优化（OHPO），并将其转化为混合与条件搜索空间上的无限多臂老虎机问题。我们提出了IMABO这一通用框架，它将任意用于在已采样配置中进行选择的老虎机策略与任意用于提出新配置的预言机相结合。我们用IMOSS对该框架进行了实例化，IMOSS是一种无需重启的anytime策略，其活跃集合以 $t^{\beta}$ 的速度增长，并证明了期望累积分位数遗憾界为 $O(p_\rho^{-1/\beta} + T^{(1+\beta)/2})$，其中 $\beta\in(0,1)$ 控制活跃集合的增长，$p_\rho$ 是对某个概率的下界约束（摘要在此处截断）。

    arXiv:2609.01335v1 Announce Type: cross  Abstract: Many production systems can assess a configuration only by using it on live requests and observing noisy feedback. Modern agentic systems are a prominent example, with inference-time choices such as model selection, retrieval depth, prompting strategy, and decoding temperature, yet often with no representative validation data. We formalize this setting as Online Hyperparameter Optimization (OHPO) and cast it as an infinitely many-armed bandit over mixed and conditional search spaces. We introduce IMABO, a general framework that combines any bandit policy for choosing among already sampled configurations with any oracle for proposing new ones. We instantiate it with IMOSS, a restart-free anytime policy whose active set grows as $t^{\beta}$, and prove an expected cumulative quantile-regret bound of $O(p_\rho^{-1/\beta} + T^{(1+\beta)/2})$, where $\beta\in(0,1)$ controls active-set growth and $p_\rho$ lower-bounds the probability that a p
    
[^161]: 潜意识学习即特质方向漂移：SFT蒸馏下的机制与针对性控制

    Subliminal Learning as Trait-Direction Drift: A Mechanism and Targeted Control under SFT Distillation

    [https://arxiv.org/abs/2609.01091](https://arxiv.org/abs/2609.01091)

    本文提出“特质方向漂移”机制来解释潜意识学习现象——偏置教师数据中可测量的偏好差距在监督微调中累积为学生的行为迁移，并据此提出探测空间走廊正则化这一针对性防御方法，在蒸馏过程中约束模型沿校准特质方向的漂移。

    

    超越预期的能力之外，模型蒸馏还可能从教师模型传递隐藏特质。一个被系统提示词偏置的教师模型可以生成语义上干净的训练数据（例如数字序列），但这些数据仍会使下游学生模型继承隐藏偏好，这种现象被称为“潜意识学习”。先前的研究已识别出该过程的若干环节，但信号在训练过程中如何积累并产生行为迁移仍不清楚，这使得有针对性的缓解难以实现。我们提出并验证了“特质方向漂移”作为潜意识学习的机制：偏置生成会在教师数据中产生可测量的偏好差距，而学生可识别的差距会在监督微调期间诱导特质对齐的参数更新，这些更新逐步累积为行为迁移。基于这一机制，我们提出了探测空间走廊正则化，这是一种针对性的防御方法，在蒸馏过程中约束沿校准特质方向的漂移……

    arXiv:2609.01091v1 Announce Type: new  Abstract: Beyond intended capabilities, model distillation can transfer hidden traits from a teacher. A teacher biased by a system prompt can generate semantically clean training data, such as numeric sequences, that still causes a downstream student to inherit the hidden preference, a phenomenon known as subliminal learning. Prior work has identified several parts of this process. How the signal builds up during training and produces behavioral transfer remains unclear, making targeted mitigation difficult. We propose and validate trait-direction drift as a mechanism for subliminal learning: biased generation creates measurable preference gaps in teacher data, and student-recognizable gaps induce trait-aligned updates during supervised fine-tuning that accumulate into behavioral transfer. Guided by this mechanism, we propose probe-space corridor regularization, a targeted defense that constrains drift along a calibrated trait direction during dis
    
[^162]: 让置信度改变，而非预测：面向事后校准的保持预测修复方法

    Let Confidence Change, Not the Prediction: Prediction-Preserving Repair for Post-hoc Calibration

    [https://arxiv.org/abs/2609.01072](https://arxiv.org/abs/2609.01072)

    本文提出CORD——首个通过修复完整校准概率向量来严格保持原始top-1预测不变、仅修正置信度的事后校准后拟合适配器，并引入TPCR指标量化校准器改变预测的频率。

    

    事后校准用于修正模型报告的置信度，然而多类别校准器也可能同时改变相应的top-1预测。准确率仅能捕捉这些变化对正确性的净效应，而无法反映预测被改变的频率；Top-1预测改变率（TPCR）正是用于衡量这一频率的指标。我们提出了面向Top-1决策保持的校准器输出修复方法（CORD），这是首个通过修复完整校准概率向量来实现严格预测保持的后拟合适配器。仅凭原始输出和校准输出，CORD即可确定分配给原始top-1类别的概率质量；校准后的条件分布则将剩余概率质量分配给其他类别，从而得到一个修复后的向量，其argmax能够恢复原始预测。在校准集上，只要条件允许，CORD会协调修复后的概率质量，以保留校准输出在原始预测上的平均概率质量。该适配器仅改变……（原文在此截断）

    arXiv:2609.01072v1 Announce Type: new  Abstract: Post-hoc calibration corrects reported confidence, yet a multiclass calibrator can also change the associated top-1 prediction. Accuracy captures only the net effect of these changes on correctness, not how often predictions change; the Top-1 Prediction Change Rate (TPCR) instead measures this frequency. We propose Calibrator-Output Repair for Top-1 Decision Preservation (CORD), the first post-fit adapter to impose exact prediction preservation by repairing the full calibrated probability vector. From the original and calibrated outputs alone, CORD determines the mass assigned to the original top-1. The calibrated conditional distribution allocates the remaining mass over the other classes, yielding a repaired vector whose own argmax recovers the original prediction. On the calibration split, CORD coordinates the repaired masses to retain the calibrated outputs' mean mass on original predictions whenever attainable. The adapter alters ne
    
[^163]: VoiceLongMemEval：助手是否记得你的声音听起来如何？

    VoiceLongMemEval: Do Assistants Remember How You Sounded?

    [https://arxiv.org/abs/2609.00570](https://arxiv.org/abs/2609.00570)

    该论文提出了VoiceLongMemEval（VLME）基准，用于评估AI助手在长时多会话对话中能否记住情感、韵律和语音事件等副语言信息，发现现有大语言模型普遍存在无法捕捉说话方式的“情感鸿沟”。

    

    随着多智能体架构和大语言模型规模的不断增长，部署的AI助手越来越多地需要对长且连续的多会话对话历史进行推理。当前的基准测试将这种对话历史评估视为长时程信息检索、时间推理或知识更新，却关键性地忽略了人机交互的基本动态，即“他们是怎么说的”（说话方式）。为了填补这一空白，我们提出了VoiceLongMemEval（VLME）基准，其中每个问题的答案都依赖于附加在对话轮次上的副语言元数据（情感标签、韵律描述符和语音事件），而这些信息仅凭文字本身是无法恢复的。每个测试项都经过三阶段对抗性门控验证，确保强大的语言模型在仅获得文本转录时无法回答。对领先的前沿模型和开放权重模型的评估揭示了普遍存在的“情感鸿沟”；提供文本轨道的副语言元数据……（摘要在此处截断）

    arXiv:2609.00570v1 Announce Type: new  Abstract: With the growing scale of multi-agent architectures and large language models, deployed AI assistants are increasingly tasked with reasoning over long, continuous, multi-session conversation histories. Current benchmarks evaluate this dialogue history as information retrieval over long horizon, temporal reasoning, or knowledge updates, while crucially ignoring the fundamental dynamics of human-agent interaction, i.e. how they said it. To address this gap, we present VoiceLongMemEval (VLME) benchmark, where every answer depends on paralinguistic metadata (emotion labels, prosody descriptors, and voice events) attached to conversational turns, which is otherwise unrecoverable from the words alone. Every item passes a three-stage adversarial gate, ensuring that a strong language model fails when given only the transcript. Evaluating leading frontier and open-weight models reveals a pervasive affect gap; providing text-track paralinguistic m
    
[^164]: EEG-VID：面向脑电解码与辅助目标选择的任务引导式潜变量预测预训练

    EEG-VID: Task-Guided Latent Predictive Pretraining for EEG Decoding and Assistive Target Selection

    [https://arxiv.org/abs/2609.00566](https://arxiv.org/abs/2609.00566)

    EEG-VID提出了一种任务引导式潜变量预测预训练框架，通过指数移动平均编码器预测未来EEG潜状态，在42组跨会话跨被试对比中有41组提升准确率（最高提升16.22个百分点），并可有效应用于场景约束下的辅助目标选择。

    

    我们提出EEG-VID，这是一个面向跨会话与跨被试脑电解码的任务引导式潜变量预测预训练框架。EEG-VID利用指数移动平均目标编码器与弱任务引导，从近期的EEG历史预测未来的潜变量EEG状态，随后进行有监督微调。在VIG-48和BCI竞赛IV-2a/IV-2b数据集上，第一阶段预训练在42组匹配的骨干网络-数据集-协议对比中有41组提升了平均准确率，包括全部12个留一被试设置，最大提升达16.22个百分点。在48区域的跨天VIG-48任务上，EEG-VID实现了6.52%的Top-1准确率和30.50%的Top-5准确率。在一项独立的六名被试离线机器人场景研究中，经过被试特定校准后，候选约束下的目标选择准确率达到40.24%，而随机水平仅为25%。这些结果支持任务引导的潜变量预测作为脑电解码与场景约束辅助目标选择的一种可迁移预训练策略。

    arXiv:2609.00566v1 Announce Type: cross  Abstract: We propose EEG-VID, a task-guided latent predictive pretraining framework for EEG decoding under session and subject shifts. EEG-VID predicts future latent EEG states from recent history using an exponential-moving-average target encoder and weak task guidance, followed by supervised fine-tuning. Across VIG-48 and BCI Competition IV-2a/IV-2b, Stage 1 improves mean accuracy in 41 of 42 matched backbone-dataset-protocol comparisons, including all 12 leave-one-subject-out settings, with a maximum gain of 16.22 percentage points. On the 48-region cross-day VIG-48 task, EEG-VID achieves 6.52% Top-1 and 30.50% Top-5 accuracy. In a separate six-participant offline robot-scene study, candidate-constrained target selection reaches 40.24% versus a 25% chance level after subject-specific calibration. These results support task-guided latent prediction as a transferable pretraining strategy for EEG decoding and scene-constrained assistive target s
    
[^165]: 为什么多层消息传递有效：图神经网络原子间势的完备性理论

    Why Multi-Layer Message Passing Works: Completeness Theory for Graph Neural Network Interatomic Potentials

    [https://arxiv.org/abs/2609.00528](https://arxiv.org/abs/2609.00528)

    本文提出多层完备性理论，证明在通用性、重叠与连通性条件下，稀疏截断图上的 $L$ 层消息传递与访问完整 $L$ 跳邻域具有同等的表示能力，从而首次严格证明了图神经网络原子间势中使用小于物理相互作用范围的逐层截断消息传递这一通用做法的合理性，并由此推出 DPA3 与 CHGNet 架构具有通用近似能力。

    

    我们证明了超图神经网络——一种具有三体消息传递的不变性架构——是势能面的通用近似器。我们的主要贡献是提出了一个多层完备性理论。我们表明，只要构型是通用的并满足重叠条件和连通性条件，在稀疏的基于截断的图上进行 $L$ 层消息传递，就能达到与访问完整 $L$ 跳邻域相同的表示能力。这为一种普遍做法——即使用多层消息传递且每层截断半径小于物理相互作用范围（几乎所有实用的基于图神经网络的机器学习原子间势都采用这一设置）——提供了首个严格的合理性证明。作为直接推论，我们表明 DPA3 和 CHGNet 两种架构均继承了通用近似性质。

    arXiv:2609.00528v1 Announce Type: new  Abstract: We prove that the Hypergraph Neural Network, an invariant architecture with 3-body message passing, is a universal approximator for potential energy surfaces. Our main contribution is a multi-layer completeness theory. We show that $L$ layers of message passing on sparse, cutoff-based graphs achieve the same representational power as having access to the full $L$-hop neighborhood, provided the configurations are generic, satisfy an overlap condition and a connectivity condition. This provides the first rigorous justification for the common practice of using multi-layer message passing with a per-layer cutoff smaller than the physical interaction range, the setting used by virtually all practical graph neural network based machine-learned interatomic potentials. As immediate consequences, we show that both DPA3 and CHGNet architectures inherit universal approximation.
    
[^166]: QTEA：基于稀疏残差显著权重与逐列优化的三值大语言模型

    QTEA: Ternary LLMs with Sparse Residual Salient Weight and By-Column Optimization

    [https://arxiv.org/abs/2609.00224](https://arxiv.org/abs/2609.00224)

    QTEA提出了一种低于2比特的训练后量化框架，通过将权重量化为三值、利用1:4半结构化稀疏的显著权重残差进行误差补偿，并结合逐列缩放精修与误差衰减机制，在保持GPU硬件效率的同时显著降低了大语言模型低比特量化的精度损失。

    

    仅权重的训练后量化（PTQ）可以缓解大规模部署大语言模型（LLM）时的计算负担。然而，现有的PTQ方法往往难以在不同模型间泛化，并且在低于2比特的量化下会出现严重的精度损失。许多方法利用非结构化稀疏性来缓解这种损失，但代价是失去了规则性和GPU友好的执行效率。我们提出了QTEA，一个低于2比特的PTQ框架，它将权重量化为三值，并使用显著权重作为残差误差补偿器。为了保持硬件效率，残差被分配到显著列中以半结构化1:4稀疏性选取的列上。我们进一步在GPTQ风格的逐列量化中加入了逐列缩放精修机制，交替更新每列的缩放因子和三值赋值，以减少重构误差。我们还识别出GPTQ中存在与处理顺序相关的误差传播问题，并引入误差衰减机制来减弱后期误差传播（摘要在此处截断）。

    arXiv:2609.00224v1 Announce Type: cross  Abstract: Weight-only post-training quantization (PTQ) can alleviate the computational burden of serving large language models (LLMs) at scale. However, existing PTQ methods often fail to generalize across models and suffer severe accuracy loss below 2 bits. Many leverage unstructured sparsity to mitigate this loss, but at the cost of regularity and GPU-friendly execution. We present QTEA, a sub-2-bit PTQ framework that quantizes weights into ternary values and uses salient weights as residual error compensators. To maintain hardware efficiency, residuals are assigned to selected columns with semi-structured \(1{:}4\) sparsity within the salient columns. We further add column-wise rescale refinement to GPTQ-style column-by-column quantization, alternately updating per-column scales and ternary assignments to reduce reconstruction error. We also identify order-dependent error propagation in GPTQ and introduce error decay to attenuate late-stage e
    
[^167]: 在流匹配模型中将生成样本追溯到训练数据簇

    Tracing Generated Samples to Training-Data Clusters in Flow-Matching Models

    [https://arxiv.org/abs/2608.30081](https://arxiv.org/abs/2608.30081)

    该论文提出一种解析与学习相结合的混合方法，在流匹配模型中沿生成轨迹推导簇级别的归因分数，从而将生成的图像追溯到影响它的训练数据簇。

    

    理解哪些训练样本影响了生成图像是生成式建模中的一个重要问题。在流匹配中，训练样本通过生成轨迹上的速度场影响生成的图像。移除样本以考察其反事实影响会改变速度场，而这种变化对最终图像产生的影响取决于它如何沿轨迹传播。因此，速度场中的局部变化并不一定能预测最终的反事实效果。本工作通过一种结合解析与学习的方法研究流匹配模型中的归因问题，并基于此推导出簇级别上基于轨迹的归因分数。我们使用独立重新训练的留一簇模型来评估这些归因分数，并在两个不同的流匹配隐空间中与多种归因基线方法进行比较。我们的实验表明……

    arXiv:2608.30081v1 Announce Type: new  Abstract: Understanding which training samples influence a generated image is an important problem in generative modeling. In flow matching, training samples influence the generated image through the velocity field along the generation trajectory. Removing samples to examine their counterfactual influence changes the velocity field, and the resulting effect on the final image depends on how the change propagates through the trajectory. Consequently, local changes in the velocity field do not necessarily predict the final counterfactual effect.   This work investigates attribution in flow-matching models through a hybrid analytical--learned approach, and uses it to derive trajectory-based attribution scores at the cluster level. We evaluate these attribution scores using independently retrained leave-one-cluster-out (LOO) models, and compare with several attribution baselines using two different flow-matching latent spaces. Our experiments show tha
    
[^168]: 替代的幻象：重新思考基础模型时代的专用机器学习模型

    The Illusion of Replacement: Rethinking Specialized Machine Learning Models in the Foundation Model Era

    [https://arxiv.org/abs/2608.28980](https://arxiv.org/abs/2608.28980)

    本文综述159篇论文后发现，语言模型虽在极端少样本预测等特定场景中可与专用模型竞争，但一旦直接评估结构表示与计算能力，并无证据表明其能全面取代机器学习中的专用架构。

    

    机器学习传统上为结构化数据构建的专用架构能否被基于语言的模型所取代？本文通过对2016年至2026年间涵盖九种模态的159篇论文的综述来检验这一问题，在考虑预测精度的同时兼顾结构表示与结构计算。论文区分了“执行任务”与“保留并计算使任务可处理的结构”这两个概念，并将现有方法归纳为八种表示机制，范围从纯语言系统到完全专用的架构。研究发现，语言中介模型在特定场景下极具竞争力，包括极端少样本预测、离散化符号任务、文本标注的知识图谱以及大规模单模态预训练。然而，只要直接评估结构表示或结构计算而非仅评估精度，就没有发现通用替代的证据。

    arXiv:2608.28980v1 Announce Type: cross  Abstract: Can the specialized architectures that machine learning has traditionally built for structured data be replaced by language-based models? This question is examined through a review of 159 papers (2016--2026) across nine modalities, with predictive accuracy considered alongside structural representation and computation. A distinction is made between performing a task and preserving and computing the structure that makes the task tractable, and existing approaches are organized into eight representational regimes, ranging from language-only systems to fully specialized architectures. Language-mediated models are found to be highly competitive in specific settings, including extreme few-shot prediction, discretized symbolic tasks, textually annotated knowledge graphs, and large-scale single-modality pretraining. However, whenever structural representation or computation is directly evaluated rather than accuracy alone, no evidence of gene
    
[^169]: 用于网络比较的最优传输：综述及其机器学习应用

    Optimal Transport for Network Comparison: A Review with Machine Learning Applications

    [https://arxiv.org/abs/2608.27500](https://arxiv.org/abs/2608.27500)

    本文综述了基于最优传输的网络比较方法，系统梳理了Wasserstein、Gromov-Wasserstein和Bures-Wasserstein三种距离，突出传输方案可解释图间差异的节点来源，并利用拉普拉斯谱为Bures-Wasserstein距离推导高效边界，进而在聚类和时间序列网络任务中验证了这些方法。

    

    运用最优传输进行网络比较是网络科学中一个不断发展的研究领域。与标准的图度量不同，最优传输不仅计算网络间的相异性，还提供一个传输方案来解释一张图如何演变为另一张图。本文综述了如何利用三种主要距离——Wasserstein距离、Gromov-Wasserstein距离和Bures-Wasserstein距离——来比较无向无权图。我们考察了通过节点特征概率分布在一维情形下Wasserstein距离的闭式解，并展示了Wasserstein距离和Gromov-Wasserstein距离的传输方案如何捕捉图扰动后具体哪些节点影响了距离。对于Bures-Wasserstein距离，我们利用拉普拉斯谱推导出上界，从而避免了完整的谱分解。最后，我们使用合成网络数据集评估这些距离在聚类任务中的表现，并应用于真实世界的时间序列网络数据。

    arXiv:2608.27500v1 Announce Type: cross  Abstract: Network comparison using optimal transport is a growing area of research in network science. Unlike standard graph metrics, optimal transport computes both network dissimilarity and a transport plan that explains how one graph morphs into another. In this paper, we review how optimal transport compares undirected, unweighted graphs using three primary distances: the Wasserstein, Gromov-Wasserstein, and Bures-Wasserstein distances. We examine the closed form of the Wasserstein distance in one dimension via node feature probability distributions, and show how the transport plans of the Wasserstein and Gromov-Wasserstein distances capture which specific nodes influence the distance after graph perturbation. For the Bures-Wasserstein distance, we derive bounds using Laplacian spectra to bypass full spectral decompositions. Finally, we evaluate these distances using a synthetic network dataset for clustering and a real-world time series net
    
[^170]: 参数化知识图谱记忆中的存储-检索差距

    A Storage-Retrieval Gap in Parametric Knowledge Graph Memory

    [https://arxiv.org/abs/2608.25489](https://arxiv.org/abs/2608.25489)

    该论文提出将知识图谱离线编译为LoRA适配器作为参数化知识层，在零查询上下文成本下实现事实知识泛化，但发现存储知识无法通过相似性检索恢复，揭示了参数化记忆中的存储-检索差距。

    

    arXiv:2608.25489v1 公告类型：交叉 摘要：图检索增强生成在查询时将检索到的子图放入模型的上下文窗口中，每次调用都支付重复的令牌成本，并在每次调用时暴露源数据。我们研究了一种替代方案：将知识图谱离线编译为每个实体一个LoRA适配器的库，这些适配器作为参数化知识层，通过注入权重而非文本来查询，在查询时零上下文成本。在MetaQA数据集上，我们发现子图训练的适配器编码了上下文无关的事实知识，这些知识能泛化到未见问题：在单值关系上，适配器相对于几乎无法闭卷的基础模型（0.007）获得了+0.243的精确匹配分数提升，且只有正确的适配器能恢复这些知识（相对于基础模型的oracle差距为+0.283）。然而，存储的知识无法通过相似性恢复：在无子图的查询下，基于嵌入和权重空间几何的检索性能均不佳。

    arXiv:2608.25489v1 Announce Type: cross  Abstract: Graph retrieval-augmented generation places retrieved subgraphs into the model's context window at query time, paying a recurring token cost and exposing source data on every call. We study an alternative: compiling a knowledge graph offline into a bank of LoRA adapters, one per entity, that serve as a parametric knowledge layer queried by injecting weights rather than text, at zero query-time context cost. On the MetaQA dataset, we find that subgraph-trained adapters encode context-free factual knowledge that generalizes to unseen questions: on single-valued relations the adapter gains $+0.243$ exact-match score over a base model that is nearly blind closed-book ($0.007$), and only the correct adapter recovers this knowledge (an oracle gap of $+0.283$ over the base model). However, the stored knowledge is not recoverable by similarity: given a query with no subgraph, embedding-based and weight-space geometry retrieval both perform at 
    
[^171]: 面向拓扑重构的配电网多相交流最优潮流可扩展自监督学习

    Scalable Self-Supervised Learning for Multiphase AC-OPF in Distribution Systems with Topology Reconfiguration

    [https://arxiv.org/abs/2608.25095](https://arxiv.org/abs/2608.25095)

    该论文提出了一种无需标签数据的自监督学习框架（惩罚+SLFS），能够高效扩展到带拓扑重构的配电网多相交流最优潮流求解，解决了现有方法在规模和复杂性上的瓶颈。

    

    arXiv:2608.25095v1 公告类型：交叉 摘要：分布式能源资源（DERs）在配电网中的普及使得这些资产的主动协调成为可能，从而降低成本并实现更清洁的运营。要实现这一潜力，需要在变化的负载、DER可用性和拓扑重构条件下快速求解多相交流最优潮流（AC-OPF），其速度和规模远超传统非线性求解器。基于学习的替代模型可以提供毫秒级推理，但现有方法主要针对平衡的输电系统，无法扩展到公用事业规模下配电网馈线的多相、不平衡和可重构特性。我们提出了惩罚+序列线性化可行寻求（SLFS）算法，这是一种面向开关引起的拓扑变化下多相配电网AC-OPF的自监督学习框架。惩罚+SLFS不需要标记的最优解，直接从AC-OPF目标函数和约束条件中训练。

    arXiv:2608.25095v1 Announce Type: cross  Abstract: The proliferation of distributed energy resources (DERs) in distribution grids enables the active coordination of these assets to reduce costs and enable cleaner operations. Realizing this potential requires solving multiphase AC optimal power flow (AC-OPF) quickly across varying loads, DER availabilities, and topology reconfigurations, at much greater speed and scale than conventional nonlinear solvers. Learning-based surrogates can offer millisecond inference, yet existing methods target largely balanced transmission systems and do not scale to the multiphase, unbalanced, and reconfigurable nature of distribution feeders at utility scale. We present the Penalty + Sequential Linearized Feasibility Seeking (SLFS) algorithm, a self-supervised learning framework for multiphase distribution AC-OPF under switch-induced topology changes. Penalty+SLFS requires no labeled optimal solutions and trains directly from the AC-OPF objective and con
    
[^172]: 面向内存高效稀疏二值自组织映射的特征主序码本：在单块消费级GPU上将MEDLINE图谱扩展至105万神经元

    A Feature-Major Codebook for Memory-Efficient Sparse-Binary Self-Organizing Maps: Scaling a MEDLINE Atlas to 1.05 Million Neurons on a Single Consumer GPU

    [https://arxiv.org/abs/2608.24067](https://arxiv.org/abs/2608.24067)

    通过将码本按特征主序存储，仅改变布局即可将自组织映射的BMU搜索加速4.5-8.5倍，实现单块消费级GPU上MEDLINE规模（105万神经元）的可扩展映射，且不损失精度。

    

    自组织映射能将大型语料库转化为可浏览的二维图谱，但在MEDLINE规模上构建一直不切实际：主导训练的最佳匹配单元（BMU）搜索受限于每个周期读取码本所需的带宽。我证明这一瓶颈主要是码本布局的产物。将码本按特征主序存储，使每个特征的权重连续排列，即W[v.M+i]，将搜索重构为分块稀疏-稠密乘积，其中每个加载的权重列在样本块中被重复使用。仅改变布局，保持实现、精度和更新规则不变，即可将BMU搜索加速4.5-8.5倍。由于精确的argmin BMU不受码本存储方式影响，这一增益无需代价：在每种地图尺寸下，留出量化误差与cuSPARSE基线一致，偏差在0.5%以内。与基线相比，优势是交叉而非恒定：cuSPARSE.SOM在较小规模时更快，但在大规模时则相反。

    arXiv:2608.24067v1 Announce Type: new  Abstract: A self-organising map turns a large corpus into a browsable two-dimensional atlas, but building one at MEDLINE scale has been impractical: the best-matching-unit (BMU) search that dominates training is bound by the bandwidth needed to read the codebook every epoch. I show that this bottleneck is largely an artefact of codebook layout. Storing it feature-major with each feature's weights contiguous, W[v.M+i], recasts the search as a tiled sparse-dense product in which every loaded weight column is reused across a tile of samples. Varying only the layout, with implementation, precision and update rule held fixed, accelerates the BMU search by 4.5-8.5x. Because an exact-argmin BMU is invariant to how the codebook is stored, this gain costs nothing: held-out quantisation error agrees with a cuSPARSE baseline to within 0.5% at every map size. Against that baseline the advantage is a crossover rather than a constant: cuSPARSE.SOM is faster at 
    
[^173]: 公理化交易者：潜在规律性、信息预算与量化投资系统的规范形式

    The Axiomatic Trader: Latent Regularity, Information Budgets, and the Canonical Form of a Quantitative Investment System

    [https://arxiv.org/abs/2608.23416](https://arxiv.org/abs/2608.23416)

    本文通过五个关键常数（复发界限、不变性缺陷、相干时间、信号上限和机制依赖比例）形式化定义了量化投资系统的公理化基础，从而推导出其近乎必然的架构。

    

    arXiv:2608.23416v1 公告类型：交叉 摘要：系统性交易基于一个信念：过去发现的规律性会持续存在。我们将其表述为由一个未观测潜在状态驱动的时间不变机制，并证明它留给研究者五个待声明的常数——在块长度$b$下的复发界限$\Lambda$、表示中声明的不变性缺陷$\epsilon_0$、状态坐标的相干时间$\ell_i$、信号上限$\rho$以及依赖于机制的部分$\kappa$——之后，一个正确量化投资系统的架构几乎是被迫确定的。

    arXiv:2608.23416v1 Announce Type: cross  Abstract: Systematic trading rests on one article of faith: that regularities found in the past persist. We state it as a time-invariant mechanism driven by an unobserved latent state, and show that it leaves a researcher five constants to declare --- the recurrence bound $Lambda$ at a block length $b$, the invariance defect $epsilon_0$ of the representation it is declared of, the coherence times $ell_i$ of the state's coordinates, the signal ceiling $rho$ and the fraction $kappa$ of it contingent on the regime --- after which the architecture of a correct quantitative investment system is nearly forced.
    
[^174]: Mol-JEPA：一种用于分子的多模态联合嵌入预测架构

    Mol-JEPA: A multimodal Joint Embedding Predictive Architecture for Molecules

    [https://arxiv.org/abs/2608.22642](https://arxiv.org/abs/2608.22642)

    Mol-JEPA通过模态掩蔽和潜在空间预测，有效整合多种生物化学数据，解决了分子基础模型中的无效增强和模态坍缩问题，提升了表示性能。

    

    尽管分子基础模型近期取得了进展，但仍存在一些局限性，如化学上无效的数据增强、模态坍缩以及生物化学环境表示不完整。为解决这些挑战，我们提出了Mol-JEPA，一个用于学习分子世界模型的可扩展框架。该模型不依赖次优的分子扰动，而是利用模态掩蔽来从分子结构、细胞表型、结合亲和力、ADMET谱、量子化学模拟及其他药物发现数据中提取信息。在多种基准测试中，我们展示了Mol-JEPA学习到的表示具有强大性能，凸显了通过潜在空间预测融入生物化学背景的价值。

    arXiv:2608.22642v1 Announce Type: cross  Abstract: Despite recent advances in molecular foundation models, several limitations remain, such as chemically invalid augmentations, modality collapse, and incomplete representation of biochemical environments. To address these challenges, we present \textbf{Mol-JEPA}, a scalable framework for learning molecular world models. Rather than relying on suboptimal molecular perturbations, our model uses modality masking to exploit information from molecular structures, cellular phenotypes, binding affinities, ADMET profiles, quantum chemistry simulations and other drug discovery data. Across various benchmarks, we show that the representations learned by Mol-JEPA deliver strong performance, demonstrating the value of incorporating biochemical context through latent space prediction.
    
[^175]: ToSCA：基于对话代理时间与策略抽象的层次强化学习

    ToSCA: Leveraging Hierarchical Reinforcement Learning on Temporal and Strategic Abstractions of Conversational Agents

    [https://arxiv.org/abs/2608.21969](https://arxiv.org/abs/2608.21969)

    本文提出一种两级层次强化学习框架，结合话语级策略抽象与词元级解码，并引入双粒度奖励机制，以提升对话代理在复杂交互中的性能。

    

    人类在日常互动和思考中具有多个层次的时间抽象能力，例如概念感知和策略规划。受此启发，我们为对话代理提出了一种两级层次强化学习（RL）框架，弥合了以往基于词元级别或话语级别RL方法之间的差距。该框架基于两级MDP开发，其中词元级别的响应解码依赖于话语级别的动作，即显式文本策略。基于理论推导和效率考虑，我们使用DQN求解高层评论家，使用PPO求解低层演员-评论家。为进一步缓解奖励稀疏性并促进收敛，我们还设计了双粒度奖励机制，将话语级别的满意度评分与词元级别的内在动机和K-L惩罚相结合。在日常对话和情感支持对话上的实验表明，所提方法优于现有基线。

    arXiv:2608.21969v1 Announce Type: new  Abstract: Humans have multiple levels of temporal abstractions on daily interaction and thinking, such as concept perception and strategic planning. Inspired by this nature, we propose a two-level hierarchical reinforcement learning (RL) framework for conversational agents, bridging the gap between previous token-level or utterance-level RL methods. Developed on a two-level MDP, the token-level response decoding is conditioned on the utterance-level action, the explicit textual strategies. Based on theoretical derivation and efficiency consideration, we use DQN to solve the high-level critic and PPO to solve the low-level actor-critic. To further alleviate the reward sparsity and facilitate the convergence, we also design the dual-granularity reward mechanism, in which the utterance-level satisfaction score is integrated with token-level intrinsic motivation and K-L penalty. Experiments on both daily and emotional support conversations show that o
    
[^176]: 代理式脚手架放大大型语言模型中的谄媚行为

    Agentic Scaffolding Amplifies Sycophantic Behavior in Large Language Models

    [https://arxiv.org/abs/2608.21377](https://arxiv.org/abs/2608.21377)

    本文发现代理式交互脚手架（如多轮反馈和迭代细化）会系统性放大LLM的谄媚行为，导致平均准确率下降6.3%，且更强模型放大效应更显著。

    

    大型语言模型中的谄媚行为，即优先迎合用户认同而非提供真实回答的倾向，已被广泛记录，但主要在单轮对话场景中研究。本文探讨了一个关键问题：对LLM施加更强的交互脚手架是否会使谄媚行为变得更糟？通过4800次真实性判断（200个陈述×6个模型×4种条件），我们发现代理系统特有的交互脚手架（反馈循环、重新考虑检查点和迭代细化）系统性地放大了谄媚行为。多轮交互、用户压力和迭代自我细化各自为模型提供了更多趋向认同的机会，这种漂移伴随着平均准确率下降6.3个百分点，表明这种屈服是有害的而非纠正性的。更强大的模型显示出更大的放大效应，这...

    arXiv:2608.21377v1 Announce Type: cross  Abstract: Sycophancy in large language models, the tendency to prioritize user agreement over truthful responses, has been documented extensively but studied primarily in single-turn settings. This paper investigates a critical question: does subjecting LLMs to greater interaction scaffolding make sycophancy better or worse? Across 4,800 veracity judgments (200 statements $\times$ 6 models $\times$ 4 conditions), we find that the interaction scaffolding characteristic of agentic systems (feedback loops, reconsideration checkpoints, and iterative refinement) systematically amplifies sycophantic behavior. Multi-turn interaction, user pressure, and iterative self-refinement each provide additional opportunities for models to drift toward agreement, and this drift coincides with a mean accuracy drop of $-6.3$ percentage points, establishing the capitulation as harmful rather than corrective. More capable models show larger amplification effects, a t
    
[^177]: FlavourBench：用可执行的烹饪真实数据对前沿语言模型进行排名

    FlavourBench: Ranking Frontier Language Models with Executable Culinary Ground Truth

    [https://arxiv.org/abs/2608.20574](https://arxiv.org/abs/2608.20574)

    该论文提出了一个基于可执行烹饪真实数据的自动化基准测试FlavourBench，通过版本化系统和严格统计方法对27个前沿语言模型进行公平排名，消除了传统基准中的评判者偏差和缺失数据问题。

    

    开放式语言模型基准测试通常继承一个评判者：人类偏好小组、另一个模型，或脆弱的精确匹配键。我们引入了FlavourBench，一个自动化基准测试，其中版本化的烹饪系统提供密集、可执行的真实数据。每个任务呈现八种食材，并要求选择三种食材的组合；在模型执行前，Epicure对所有56种可能的组合进行评分。我们在一个包含534个任务的相同核心集上评估了27个前沿端点，涵盖替代、配对和受限组合。每个排名的模型在每个面板和家族中恰好有89个有效响应（总共14,418个模型-任务单元），消除了排行榜上的差异性缺失。FlavourBench分数是冻结任务分数的等家族均值。我们使用50,000个锚点聚类自助重采样进行同时95%分数区间，以及100,000次符号翻转抽样进行所有351个配对模型对比，并采用Holm校正。两个独立的...

    arXiv:2608.20574v1 Announce Type: new  Abstract: Open-ended language-model benchmarks usually inherit a judge: a human preference panel, another model, or a brittle exact-match key. We introduce FlavourBench, an automated benchmark in which a versioned culinary system supplies dense, executable ground truth. Each task presents eight ingredients and asks for a three-ingredient portfolio; before model execution, Epicure scores all 56 possible portfolios. We evaluate 27 frontier endpoints on an identical 534-task core spanning substitution, pairing, and constrained composition. Every ranked model has exactly 89 valid responses per panel and family (14,418 model-task cells total), eliminating differential missingness from the leaderboard. The FlavourBench Score is the equal-family mean of the frozen task scores. We use 50,000 anchor-cluster bootstrap replicates for simultaneous 95% score bands and 100,000 sign-flip draws for all 351 paired model contrasts, with Holm control. The two indepe
    
[^178]: 异质数据集的对角多组学整合

    Diagonal Multi-omics Integration of Heterogenous Datasets

    [https://arxiv.org/abs/2608.16968](https://arxiv.org/abs/2608.16968)

    本文提出了一种基于极值迹问题和梯度上升方法的新特征，利用最大值与最小值点差的范数来表征数据集异质性，用于异质数据集的对角多组学整合。

    

    本文考虑了异质数据集的对角多组学整合方法。我们分析并发展了多种处理生物异质性本质的方法，以更清晰地理解所产生的差异。具体而言，研究了嵌入复欧几里得空间中与Stiefel流形同胚的集合上耦合拉普拉斯算子的极值迹问题。最大化问题的梯度上升方法以泛函分析的经典术语进行了详细阐述，这本身具有重要研究意义。在此基础上，我们通过采用最大值与最小值点之间差的范数，引入了数据集异质性的一个新特征。

    arXiv:2608.16968v1 Announce Type: cross  Abstract: In this paper, we consider methods for the diagonal multi-omics integration of heterogeneous datasets. Several approaches to the nature of biological heterogeneity are analyzed and developed to comprehend more clearly the generated differences. Specifically, the extremal trace problems for the coupled Laplacian on sets homeomorphic to the Stiefel manifold embedded in the complex Euclidean space are investigated. The gradient ascent method for the maximization problem is elaborated in the classical terms of functional analysis, which is of significant interest in itself. On this basis, we introduce a novel characteristic of dataset heterogeneity by employing the norm of the difference between the maximum and minimum points.
    
[^179]: TransfHAR：基于自监督腕部表征的按需活动识别

    TransfHAR: Self-Supervised Wrist Representations for On-Demand Activity Recognition

    [https://arxiv.org/abs/2608.15861](https://arxiv.org/abs/2608.15861)

    TransfHAR通过自监督预训练从粗粒度腕部活动中学习可迁移运动先验，实现无需大量标记数据的按需细粒度活动识别，并支持用户自定义活动集。

    

    arXiv:2608.15861v1 公告类型：新 摘要：细粒度腕部活动识别可支持程序步骤引导和情境感知辅助等应用，然而为每个新任务、用户和活动粒度获取标记数据仍是瓶颈。我们提出TransfHAR，一种自监督腕部IMU框架，通过从全局未标记活动中学习可迁移的运动先验，实现按需的细粒度活动识别。我们表明，在粗粒度腕部IMU活动（如坐、走、锻炼）上进行自监督预训练，能学习到足够丰富的运动结构，可迁移到预训练中不存在的细粒度操作、手势和程序性活动（如捏、搅拌、挥手）。我们将TransfHAR实现为实时智能手表应用，允许用户仅通过少量演示定义和扩展自己的活动集，以实现个性化识别。在三个离线跨数据集评估中，TransfHAR达到或超过基线性能。

    arXiv:2608.15861v1 Announce Type: new  Abstract: Fine-grained wrist activity recognition can support applications such as procedural step guidance and context-aware assistance, yet acquiring labeled data for every new task, user, and activity granularity remains a bottleneck. We present TransfHAR, a self-supervised wrist IMU framework for on-demand, fine-grained activity recognition by learning transferable motion priors from global, unlabeled activities. We show that self-supervised pretraining on coarse wrist IMU activities (e.g., sitting, walking, exercise) learns motion structure rich enough to transfer to fine-grained manipulative, gestural, and procedural activities (e.g., snapping, stirring, waving) that are absent from pretraining. We implement TransfHAR as a real-time smartwatch application that lets users define and expand their own activity set for personalized recognition from only a few demonstrations. Across three offline cross-dataset evaluations, TransfHAR matches or ex
    
[^180]: 自监督视觉表征学习的三个必要原则

    Three Necessary Principles for Self-Supervised Visual Representation Learning

    [https://arxiv.org/abs/2608.08309](https://arxiv.org/abs/2608.08309)

    该论文提出自监督视觉表征学习必须同时满足观察（语义不变性）、预测（块级空间预测）与正则化（表征非退化）三个缺一不可的原则，并从理论上证明：缺少正则化时代码器会坍缩为常量映射，而对比对齐与动量编码器均无法在收敛时保证不发生坍缩。

    

    我们认为，在无标签条件下学习视觉表征，需要一个在三个互不重叠的目标上联合完备的训练信号：增广视图间的语义不变性、图像块级空间预测，以及表征的非退化性。我们将这三者形式化为观察原则、预测原则与正则化原则，并证明：(i) 在无负样本对齐的条件下，若仅组合观察与预测而不加正则化，常量编码器可以作为全局最小化器存在；(ii) 这两个目标在编码器输出处梯度互补且结构上不冲突；(iii) 动量编码器会收敛到与在线编码器相同的固定点，并且在收敛时并不提供防止坍缩的保证。对比对齐仅能提供一种自我受限的抗坍缩能力，这一点通过显式的梯度衰减论证得到形式化。移除预测目标则会因……而失去空间训练信号（原文摘要在此处不完整）。

    arXiv:2608.08309v2 Announce Type: replace-cross  Abstract: We argue that learning visual representations without labels requires a training signal jointly complete across three non-overlapping objectives: semantic invariance across augmented views, patch-level spatial prediction, and representational non-degeneracy. We formalize these as the observation, prediction, and regularization principles and prove (i) that combining observation and prediction without regularization admits the constant encoder as a global minimizer under negative-free alignment; (ii) that the two objectives are gradient-complementary and structurally non-conflicting at the encoder output; and (iii) that the momentum encoder converges to the same fixed point as the online encoder and provides no collapse guarantee at convergence. Contrastive alignment provides only self-limiting collapse resistance, formalized via an explicit gradient-decay argument. Dropping prediction withholds the spatial training signal by co
    
[^181]: 简单变换在文本嵌入模型间能实现多大程度的转换？

    How Far Do Simple Transformations Translate Across Text Embedding Models?

    [https://arxiv.org/abs/2608.05980](https://arxiv.org/abs/2608.05980)

    本研究在九个架构、池化策略和训练目标各异的嵌入模型上检验了“潜在通用性”假设，发现线性映射等简单变换仅能在部分兼容的模型对之间成功转换表示，异构嵌入空间的兼容性受架构、训练目标、池化和数据分布共同影响，并非普遍通用。

    

    我们研究了简单的变换是否能在异构文本嵌入模型之间转换表示。理解独立训练的模型如何组织语义信息，是实现AI到AI潜在通信（无需解码为人类可读文本）的关键使能技术。聚焦于线性映射等轻量级转换器，我们在超越简化基准的现实文本环境中检验了文献中关于潜在通用性的假设。我们在九个架构、池化策略和训练目标各不相同的嵌入模型上，使用CKA、下游迁移、保真度和检索来评估兼容性。简单转换器能够恢复有意义的共享结构，并支持部分兼容模型对之间的迁移，但在其他模型对上则急剧失败。兼容性共同取决于架构、训练目标、池化和数据分布。总体而言，结果表明异构嵌入空间并非普遍通用。

    arXiv:2608.05980v2 Announce Type: replace  Abstract: We investigate whether simple transformations can translate representations across heterogeneous text embedding models. Understanding how independently trained models organize semantic information is an enabler for AI-to-AI latent communication without decoding into human-readable text. Focusing on lightweight translators such as linear mappings, we test the literature hypothesis of latent universality in a realistic text setting beyond simplified benchmarks. Across nine embedding models differing in architecture, pooling strategy, and training objective, we evaluate compatibility using CKA, downstream transfer, fidelity, and retrieval. Simple translators recover meaningful shared structure and support transfer for some compatible pairs, but fail sharply for others. Compatibility depends jointly on architecture, training objective, pooling, and data distribution. Overall, the results show that heterogeneous embedding spaces are not u
    
[^182]: 面向单细胞生成的自回归Transformer模型规模化研究

    Scaling an Autoregressive Transformer for Single-Cell Generation

    [https://arxiv.org/abs/2608.02961](https://arxiv.org/abs/2608.02961)

    本文提出将自回归Transformer与可学习量化VAE分词器相结合用于单细胞基因表达向量的自监督生成，并首次发现了该架构联合拟合的双指数缩放定律与计算最优训练配置。

    

    我们研究了单细胞基因表达向量的自监督生成任务：给定来自某一细胞类型的一组向量，我们的目标是生成该细胞类型的更多基因表达向量。针对该任务，我们既刻画了所生成基因表达向量的生物保真度，也刻画了预训练损失的缩放行为。该模型由一个因果Transformer与一个可学习的量化VAE分词器配对构成，并采用交叉熵损失进行训练。为评估该模型，我们以某细胞类型的留出基因表达向量为条件生成基因表达向量，并将生成的基因表达向量分布与该细胞类型的真实分布进行比较。我们通过改变可训练参数数量和训练数据量来研究所提出架构的缩放特性。据我们所知，我们首次发现了联合拟合的双指数缩放定律以及计算最优（摘要在此处截断）……

    arXiv:2608.02961v2 Announce Type: replace-cross  Abstract: We study a self-supervised generation task for single-cell gene expression vectors: given a set of vectors from a cell type, we aim to generate additional gene expression vectors of that cell type. For this task we characterize both the biological fidelity of the generated gene expression vectors and the scaling behavior of the pretraining loss. The model is a causal transformer paired with a learned quantized VAE tokenizer, trained with a cross-entropy loss. To evaluate the model, we condition it on held-out gene expression vectors of a cell type and generate vectors of gene expression, comparing the resulting distribution over gene expression vectors to the ground truth distribution of that cell type. We study the scaling properties of the proposed architecture by varying the number of trained parameters and the amount of training data. To our knowledge, we find the first jointly-fit two-exponent scaling law and compute-optim
    
[^183]: 训练 nGPT

    Training nGPT

    [https://arxiv.org/abs/2608.01284](https://arxiv.org/abs/2608.01284)

    本文提出的nGPT实用训练方案（含Logit梯度预处理、对数学习率衰减等技术），使300亿参数的混合MoE模型仅需约一半训练token即可达到与AdamW训练的未归一化模型相同的验证损失。

    

    归一化Transformer（nGPT）通过将模型参数向量和激活向量约束到单位超球面上，实现超球面表示学习。本文描述了一种实用的nGPT训练方案，并在现代混合Mamba-2–Transformer混合专家模型上进行了评估。该方案引入了Logit梯度预处理、对数学习率衰减、GatedAdamW、角度更新控制以及可选的探索机制。与使用AdamW训练的相同混合MoE架构的未归一化模型相比，总参数量为300亿的nGPT模型仅需约一半的训练token即可达到相同的验证损失。该方案在所评估的总参数量最高达300亿的模型上均具有良好的可扩展性。

    arXiv:2608.01284v2 Announce Type: replace-cross  Abstract: The normalized Transformer (nGPT) realizes hyperspherical representation learning by constraining model parameter vectors and activation vectors to the unit hypersphere. In this paper, we describe a practical training recipe for nGPT and evaluate it on modern hybrid Mamba-2--Transformer Mixture-of-Experts (MoE) models. The recipe introduces Logit Gradient Preconditioning, Logarithmic Learning Rate Decay, GatedAdamW, angular update control, and optional exploration mechanisms. Compared with an unnormalized model of the same hybrid MoE architecture trained with AdamW, the 30B-total-parameter nGPT model reaches the same validation loss using approximately half as many training tokens. The recipe scales across the models considered, which contain up to 30B total parameters.
    
[^184]: 从数字到物理储备池计算：通过动力学匹配协同优化软体机器人储备池

    From Digital to Physical Reservoir Computing: Co-Optimizing Soft Robotic Reservoirs via Dynamics Matching

    [https://arxiv.org/abs/2608.00484](https://arxiv.org/abs/2608.00484)

    提出一种通过动力学匹配将软体机器人物理储备池与高性能数字参考动力学进行预训练和协同优化的框架，利用可微物理模型联合优化物理参数、微分同胚状态映射和前馈-反馈控制，从而缩小物理储备池与数字储备池之间的性能差距。

    

    软体机器人基底在物理储备池计算（PRC）中极具前景，因为其柔性非线性动力学能够提供时序记忆、高维状态变换和高效推理。然而，物理储备池通常被直接采用而不经过预训练或协同优化，这可能限制了软体机器人PRC相对于数字储备池的性能。我们研究了是否可以通过对照高性能的数字参考动力学来预训练物理储备池。我们的方法利用可微物理模型以及避免时间积分的加速度级方程误差目标，联合优化物理参数、微分同胚的物理-参考状态映射以及前馈-反馈控制。作为概念验证，我们使用仿真软体机器人、随机振荡器网络（RON）参考以及并行多起点梯度下降来实例化该方法。

    arXiv:2608.00484v2 Announce Type: replace-cross  Abstract: Soft robotic substrates are promising for Physical Reservoir Computing (PRC) because their compliant nonlinear dynamics can provide temporal memory, high-dimensional state transformations, and efficient inference. However, physical reservoirs are often adopted as-is rather than pretrained or co-optimized, potentially limiting soft robotic PRC performance relative to digital reservoirs. We investigate whether a physical reservoir can instead be pretrained against high-performing digital reference dynamics. Our formulation jointly optimizes physical parameters, a diffeomorphic physical-reference state map, and feedforward-feedback control using a differentiable physical model and an acceleration-level equation-error objective that avoids temporal integration. As a proof of concept, we instantiate the formulation with simulated soft robots, a Random Oscillators Network (RON) reference, and parallel multi-start gradient descent. We
    
[^185]: Nova：一种用于深度学习的端到端MLIR编译器

    Nova: An End-to-End MLIR Compiler for Deep Learning

    [https://arxiv.org/abs/2608.00029](https://arxiv.org/abs/2608.00029)

    Nova编译器通过端到端MLIR流水线，将前向和反向传播统一为单一值语义方言，实现整图优化并直接合成细粒度内核，从而原生支持完整Transformer架构并最大化硬件利用。

    

    大规模深度学习模型的性能在很大程度上取决于高层数学操作如何有效地映射到底层物理硬件。虽然高层张量框架提供了灵活的抽象，但它们的执行模型本质上缺乏最大化硬件利用率所需的整图可见性，常常迫使对复杂操作（如注意力机制）依赖不透明的手写内核库。为了弥合这一差距，我们提出了Nova的下一代版本，这是一种自动化的端到端JIT编译器，通过直接从计算结构合成细粒度内核，实现对硬件映射的绝对控制。在本工作中，我们扩展了Nova的编译流水线，以原生支持完整的Transformer架构。通过捕获即时执行并将前向和反向传播统一为单一值语义方言，Nova实现了积极的整图优化。而非...

    arXiv:2608.00029v2 Announce Type: replace  Abstract: The performance of deep learning models at scale relies heavily on how effectively high-level mathematical operations are mapped to underlying physical hardware. While high-level tensor frameworks provide flexible abstractions, their execution models inherently lack the whole-graph visibility required to maximize hardware utilization, often forcing a reliance on opaque, hand-written kernel libraries for complex operations like Attention. To bridge this gap, we present the next iteration of Nova, an automated end-to-end JIT compiler that achieves absolute control over hardware mapping by synthesizing fine-grained kernels directly from the computation's structure. In this work, we extend Nova's compilation pipeline to natively support full Transformer architectures. By capturing eager executions and unifying forward and backward passes into a single value-semantic dialect, Nova unlocks aggressive whole-graph optimizations. Rather than 
    
[^186]: 神经算子的特征交互建模

    Feature Interaction Modeling for Neural Operators

    [https://arxiv.org/abs/2607.28762](https://arxiv.org/abs/2607.28762)

    提出FM-Operator，一种显式建模传感器观测与查询坐标之间特征构建与交互的逐点查询神经算子，通过将DeepONet的分支-主干聚合重新诠释为乘法交互，提升了对激波主导和低粘性偏微分方程的求解能力。

    

    尽管目前已提出许多DeepONet的变体，基于查询的算子网络在处理激波主导和低粘性偏微分方程时仍然表现不佳，这类方程具有尖锐移动间断和缓慢衰减的解谱，对有限维可分离表示构成了挑战。在本工作中，我们提出了特征交互建模算子，这是一种逐点查询的神经算子，能够显式地建模传感器观测与查询坐标之间的特征构建与交互。我们的设计源于从乘法交互的角度对经典DeepONet聚合方式的重新诠释。具体而言，分支—主干内积具有等价形式 \(\boldsymbol{b}(u)^\top \boldsymbol{\tau}(y)=\boldsymbol{1}^\top \operatorname{diag}(\boldsymbol{b}(u))\,\boldsymbol{\tau}(y)\)，这揭示了两种表示仅沿对应的潜在维度进行交互，而……

    arXiv:2607.28762v2 Announce Type: replace  Abstract: Despite the many variants of DeepONet that have been proposed, query-based operator networks still struggle with shock-dominated and low-viscosity PDEs, whose sharp moving discontinuities and slowly decaying solution spectra challenge finite-dimensional separable representations. In this work, we propose \emph{Feature Interaction Modeling Operator} (FM-Operator), a point-wise query neural operator that explicitly models feature construction and interactions between sensor observations and query coordinates. Our design is motivated by a reinterpretation of the canonical DeepONet aggregation through the lens of multiplicative interactions. Specifically, the branch--trunk inner product admits the equivalent form \(\boldsymbol{b}(u)^\top \boldsymbol{\tau}(y)=\boldsymbol{1}^\top \operatorname{diag}(\boldsymbol{b}(u))\,\boldsymbol{\tau}(y)\), revealing that the two representations interact only along corresponding latent dimensions and the
    
[^187]: 弹性粒子采样器与Zigzag采样器的窗口化稀疏化方法及查询复杂度

    Windowed thinning and query complexity for the bouncy particle and Zigzag samplers

    [https://arxiv.org/abs/2607.28413](https://arxiv.org/abs/2607.28413)

    该论文提出窗口化稀疏化这一针对弹性粒子采样器和坐标Zigzag过程的精确模拟方法，并结合定量混合估计首次给出了从高斯冷启动达到总变差误差 $\varepsilon$ 所需的梯度查询复杂度保证。

    

    设 $\mu(dx)\propto e^{-U(x)}dx$ 为 $\R^d$ 上的概率分布，其中 $U$ 是 $m$-强凸且 $L$-光滑的函数，并记 $\kappa=L/m$ 为条件数。我们研究窗口化稀疏化方法，这是一种针对弹性粒子采样器和坐标Zigzag过程的精确模拟方法。该方法将轨迹划分为确定性的窗口，并在每个窗口开始时进行一次梯度评估，以构造一个易于处理的事件率局部包络。将这一构造与定量的混合时间估计以及弹跳和翻转期望次数的有限时间界相结合，可以从高斯冷启动出发得到查询复杂度保证。对于总变差误差 $\varepsilon$，弹性粒子采样器的期望查询次数为 $O(\kappa^{1/2}d\,(d\log\kappa+\log\frac{1}{\varepsilon}))$ 次梯度查询，Zigzag过程为 $O(\kappa d^{1/4}(d\log\kappa+\log\frac{1}{\varepsilon}))$ 次全梯度等价查询，其中 $d$ 个坐标……

    arXiv:2607.28413v2 Announce Type: replace-cross  Abstract: Let $\mu(d x)\propto e^{-U(x)} d x$ on $\R^d$, where $U$ is $m$-strongly convex and $L$-smooth, and denote by $\kappa=L/m$ the condition number. We consider windowed thinning, an exact simulation method for the bouncy particle sampler and the coordinate Zigzag process. The method divides a trajectory into deterministic windows and uses a gradient evaluation at the beginning of each window to construct a tractable local envelope for the event rate. Combining this construction with quantitative mixing estimates and finite-time bounds on the expected numbers of bounces and flips yields query complexity guarantees from a Gaussian cold start. For total-variation error $\varepsilon$, the expected query counts are $O(\kappa^{1/2}d\,(d\log\kappa+\log\frac1\varepsilon))$ gradient queries for the bouncy particle sampler and $O(\kappa d^{1/4}(d\log\kappa+\log\frac1\varepsilon))$ full-gradient equivalents for Zigzag, where $d$ coordinate-p
    
[^188]: 留出证据解决生物筛选中的后续测量决策

    Held-out evidence resolves follow-up measurement decisions in biological screens

    [https://arxiv.org/abs/2607.27651](https://arxiv.org/abs/2607.27651)

    本研究提出OPAL留出决策测试框架，通过在最终评估前固定档案特定标准来判断机器学习筛选规则能否替代固定测量计划，实现了将优化与足以改变实验的证据相分离。

    

    机器学习被用于决定生物筛选应收集哪些后续测量。在一个包含六条规则的Cell Painting（细胞绘画）规则组中，价值最高的规则将对96.01%的化合物库进行重新成像，且其误激活上界高达97.14%，这说明仅凭预测价值无法证明用其替代固定测量计划的合理性。我们开发了OPAL——一种留出决策测试方法，它冻结给定规则，并在考虑成本后，依据在最终评估之前预先固定的特定档案标准，来判断不必要的测量、覆盖率和价值。一个由开发阶段选出的稀疏Cell Painting规则虽然将额外孔负担降低了18.2倍，但其假发现上界超过了35%，因此仍保留固定计划。LINCS-LJP在开发期间设定的点估计标准下倾向于广泛采集，而非选择性节省。CTRP则需要回退到固定计划，因为其冻结的评分遗漏了已测量的机会。OPAL将优化过程与足以改变实验方案的证据分离开来。

    arXiv:2607.27651v3 Announce Type: replace  Abstract: Machine learning determines which follow-up measurements biological screens collect. In a six-rule Cell Painting battery, the highest-value rule would re-image 96.01% of the library and had a 97.14% false-activation upper bound, showing why predicted value alone cannot justify replacing a fixed plan. We developed OPAL, a held-out decision test that freezes a rule and judges unnecessary measurement, coverage and value after cost against archive-specific criteria fixed before final evaluation. A development-selected sparse Cell Painting rule had 18.2-fold lower added-well burden, but its false-discovery bound exceeded 35%, so the fixed plan remained. LINCS--LJP favored broad acquisition under point-estimate criteria set during development, not selective saving. CTRP required fallback because its frozen score missed measured opportunity. OPAL separates optimization from evidence sufficient to change an experiment.
    
[^189]: Aletheia：面向低资源医疗环境的离线优先鉴别诊断临床决策支持系统

    Aletheia: An Offline-First Clinical Decision Support System for Differential Diagnosis in Low-Resource Healthcare Settings

    [https://arxiv.org/abs/2607.24814](https://arxiv.org/abs/2607.24814)

    Aletheia是一个面向撒哈拉以南非洲低资源医疗环境的离线优先临床决策支持系统，通过QLoRA微调Qwen2.5-3B-Instruct模型，在东非高发疾病鉴别诊断中实现了80%的Top-1准确率和100%的Top-3准确率，无需依赖互联网连接和高规格硬件。

    

    在撒哈拉以南非洲地区，专科临床专业知识的获取仍然严重受限，农村地区的医患比例可能低于1:25,000。现有的AI辅助诊断工具主要依赖可靠的互联网连接和高规格硬件，这使其对于地区医院和卫生中心的一线医护人员而言并不实用。本文提出了Aletheia，一个专为撒哈拉以南非洲低资源医疗环境设计的离线优先临床决策支持系统。Aletheia基于Qwen2.5-3B-Instruct构建，采用量化低秩自适应（QLoRA）技术，在涵盖50种东非高发疾病、共27,000个临床推理样本的精选数据集上进行了微调。评估结果显示，其Top-1诊断准确率为80%（10例中的8例；95% CI 49.0-94.3%），Top-3准确率为100%（10例中的10例；95% CI 72.2-100%），BERTScore-F1为0.909。

    arXiv:2607.24814v2 Announce Type: replace  Abstract: Access to specialist clinical expertise remains severely limited across sub-Saharan Africa, where physician-to-patient ratios can fall below 1:25,000 in rural settings. Existing AI-assisted diagnostic tools predominantly require reliable internet connectivity and high-specification hardware, rendering them impractical for frontline healthcare workers in district hospitals and health centres. This paper presents Aletheia, an offline-first clinical decision support system designed for low-resource healthcare contexts across sub-Saharan Africa. Aletheia is built upon Qwen2.5-3B-Instruct, fine-tuned using Quantised Low-Rank Adaptation (QLoRA) on a curated dataset of 27,000 clinical reasoning samples spanning 50 disease conditions with elevated prevalence in East Africa. Evaluation demonstrates a Top-1 diagnostic accuracy of 80% (8 of 10 cases; 95% CI 49.0-94.3%), Top-3 accuracy of 100% (10 of 10; 95% CI 72.2-100%), BERTScore-F1 of 0.909,
    
[^190]: 基于大语言模型的自适应群岛图进化自动特征工程

    Adaptive Graph-of-Islands Evolution for Automatic Feature Engineering with LLMs

    [https://arxiv.org/abs/2607.23286](https://arxiv.org/abs/2607.23286)

    TOPOFE将自动特征工程构建为图结构的多岛屿进化程序搜索框架，通过LLM引导的变异与交叉探索语义族群、利用提示词自适应记忆积累接受/拒绝反馈来优化候选生成，并动态协调岛屿间的迁移以提升全局搜索效率。

    

    面向表格数据的自动特征工程（AutoFE）需要从庞大的程序空间中发现有信息量的变换。现有方法存在三个局限：经典方法依赖表达能力有限的固定算子库；基于大语言模型（LLM）的方法从静态提示词生成候选方案，且不保留搜索经验；进化方法使用固定的迁移策略，忽略了任务特定的跨族群迁移效用。我们提出了TOPOFE，这是一个将AutoFE构建为图结构多岛屿进化程序搜索的框架。变换空间被划分为语义上连贯的族群，每个族群由一个岛屿通过LLM引导的变异和交叉进行探索。每个岛屿维护一个提示词自适应记忆，通过累积接受/拒绝反馈，在无需参数更新的情况下引导候选方案朝着富有成效的区域发展。为了协调全局探索，TOPOFE动态地……

    arXiv:2607.23286v2 Announce Type: replace  Abstract: Automatic feature engineering (AutoFE) for tabular data requires discovering informative transformations from a large program space. Existing approaches suffer from three limitations: classical methods rely on fixed operator libraries with limited expressivity, LLM-based methods generate proposals from static prompts without retaining search experience, and evolutionary methods use fixed migration policies that ignore task-specific cross-family transfer utility. We introduce TOPOFE, a framework that formulates AutoFE as graph-structured multi-island evolutionary program search. The transformation space is partitioned into semantically coherent families, each explored by an island through LLM-guided mutation and crossover. Each island maintains a Prompt Adaptation Memory that accumulates accept/reject feedback to steer proposals toward productive regions without parameter updates. To coordinate global exploration, TOPOFE dynamically l
    
[^191]: 海上监视中异构传感器选择的强化学习

    Reinforcement Learning for Heterogeneous Sensor Selection in Maritime Surveillance

    [https://arxiv.org/abs/2607.22667](https://arxiv.org/abs/2607.22667)

    该论文提出了一种信息增益引导的强化学习传感器选择框架，利用PPO智能体在异构海上传感器网络中为单船跟踪智能选择传感器，从而避免了激活全部传感器或进行高计算代价的在线信息增益评估的需要。

    

    本文提出了一种信息增益引导的强化学习传感器选择框架，用于异构海上传感器网络中的单船跟踪。该方法的灵感来源于信息论的传感器管理思想：无需激活所有传感器，也无需反复执行计算代价高昂的在线期望信息增益评估，而是由学习到的策略在每个决策时刻选择一个与跟踪相关的传感器。贝叶斯序贯蒙特卡罗跟踪器根据带噪声的量测估计船舶状态，并提供信念表示，以支持非线性和非高斯条件下的传感器调度。一个近端策略优化（PPO）智能体在塞浦路斯阿依纳帕码头CMMI智能码头试验台的地理参考仿真环境中，从五个传感器中选择其一。该策略在试验台实际的五传感器配置上进行训练。该智能体观察信念状态、检测历史、覆盖范围、传感器几何等信息（摘要内容在此处被截断）。

    arXiv:2607.22667v2 Announce Type: replace  Abstract: This paper presents an information-gain-guided reinforcement-learning sensor-selection framework for single-vessel tracking in heterogeneous maritime sensor networks. The proposed approach is motivated by information-theoretic sensor management: instead of activating all sensors or repeatedly performing computationally expensive online expected-information-gain evaluation, a learned policy selects one tracking-relevant sensor at each decision epoch. A Bayesian sequential Monte Carlo tracker estimates the vessel state from noisy measurements and provides a belief representation for scheduling under nonlinear and non-Gaussian conditions. A Proximal Policy Optimization agent selects one of five sensors in a georeferenced simulation of the CMMI Smart Marina testbed at Ayia Napa Marina, Cyprus. The policy is trained on the testbed's actual five-sensor configuration. The agent observes belief-state, detection-history, coverage, sensor-geom
    
[^192]: 面向少步生成的多掩码扩散语言模型

    Multi-Mask Diffusion Language Models for Few-Step Generation

    [https://arxiv.org/abs/2607.19686](https://arxiv.org/abs/2607.19686)

    提出多掩码扩散模型MultiMDM，通过在前向过程中保留掩码结构、在反向过程中先预测指定掩码再精炼为干净词元的起草能力，实现高质量的少步文本生成。

    

    arXiv:2607.19686v3 公告类型：替换。掩码扩散模型是一类很有前景的语言生成器，但实现高质量的少步生成仍然具有挑战性。在MDM中，所有前向轨迹都会坍缩到单一的全掩码状态，因此没有为一致性风格的少步生成保留终端熵。虽然最近基于均匀状态扩散的少步替代方案避免了这种退化问题，但与MDM相比，将干净词元与噪声区分开来变得更加困难，这通常会损害建模质量和训练效率。在这项工作中，我们提出了多掩码扩散模型，它为少步生成保留了掩码结构。在前向过程中，每个干净词元首先被推向一个指定的掩码，然后逐渐在掩码集合上混合。因此，反向过程具备了起草能力，即先将指定掩码预测出来，再将其精炼为干净词元。我们推导了闭式ELBO训练目标……

    arXiv:2607.19686v3 Announce Type: replace  Abstract: Masked diffusion models (MDMs) are a promising family of language generators, but achieving high-quality few-step generation remains challenging. In MDMs, all forward trajectories collapse to a single fully masked state, leaving no terminal entropy for consistency-style few-step generation. While recent few-step alternatives based on uniform-state diffusion avoid this degeneracy, it becomes harder to distinguish clean tokens from noise than MDMs, which usually harms modeling quality and training efficiency. In this work, we propose a multi-mask diffusion model (MultiMDM) that preserves the masking structure towards few-step generation. In the forward process, each clean token is first pushed towards a designated mask and then gradually mixes over the mask set. As a result, the backward process has a drafting capability by predicting a designated mask before refining to a clean token. We derive a closed-form ELBO training objective fo
    
[^193]: 一个模型，多种图：基于视觉语言模型在异构模态属性图上的学习

    One Model, Many Graphs: Learning over Attributed Graphs across Heterogeneous Modalities with Vision-Language Models

    [https://arxiv.org/abs/2607.19128](https://arxiv.org/abs/2607.19128)

    提出OMG-VLM统一框架，以预训练视觉语言模型为共享骨干并引入结构感知图适配器，实现了对仅含文本、仅含视觉或两者兼有的异构模态属性图的统一学习，突破了现有方法需针对固定模态模式单独建模的限制。

    

    视觉语言模型（VLM）为文本和视觉信息提供了统一的表示空间，但它们作为图结构数据通用骨干网络的潜力在很大程度上尚未被探索。在实践中，属性图表现出显著的模态异质性：一些图仅包含文本节点属性，另一些仅包含视觉属性，还有一些同时提供两者。现有的图学习方法通常是为固定的模态模式设计的，需要为不同设置单独构建模型，这限制了可扩展性和跨图泛化能力。为了弥合这一差距，我们提出了OMG-VLM（基于视觉语言模型的一个模型多种图），这是一个用于跨异构模态模式学习属性图的统一框架。OMG-VLM利用预训练的VLM作为共享骨干网络，并引入结构感知的图适配器，在整合邻域信息的同时保持兼容性……

    arXiv:2607.19128v2 Announce Type: replace  Abstract: Vision-language models (VLMs) provide a unified representation space for textual and visual information, yet their potential as general-purpose backbones for graph-structured data remains largely unexplored. In practice, attributed graphs exhibit substantial modality heterogeneity: some graphs contain only textual node attributes, others only visual attributes, while still others provide both. Existing graph learning approaches are typically designed for fixed modality schemas, requiring separate models for different settings and limiting scalability and cross-graph generalization. To bridge this gap, we present OMG-VLM (One Model, Many Graphs with Vision-Language Models), a unified framework for learning over attributed graphs across heterogeneous modality schemas. OMG-VLM leverages a pretrained VLM as a shared backbone and introduces structure-aware graph adapters that integrate neighborhood information while remaining compatible w
    
[^194]: 持久稀疏自编码器：在语言模型表示中学习特征特定的时间尺度

    Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations

    [https://arxiv.org/abs/2607.17117](https://arxiv.org/abs/2607.17117)

    该论文提出持久稀疏自编码器，通过为每个特征学习一个持久性系数，使稀疏自编码器能够仅凭重构目标从语言模型激活中自动学习特征特定的时间尺度，同时保持高质量的重构效果。

    

    稀疏自编码器（SAE）将语言模型的激活分解为稀疏特征，然而这些模型传统上对每个词元进行独立编码，无法揭示跨序列持续存在的信息。我们首先证明，时间持久性可以在标准SAE特征中自然涌现：当一个特征激活后，隐藏状态会保持与其方向对齐，且过去的激活有助于重构后续的隐藏状态。这种持续性在不用特征之间的差异很大。因此，我们提出了持久稀疏自编码器（Persistent SAEs），这是标准SAE的一种扩展，它为每个特征学习一个持久性系数，使模型能够仅通过重构任务学习特征特定的时间尺度。我们的实验表明，持久SAE在学习一系列时间尺度的同时保持了有竞争力的重构质量：短时间尺度（快速）特征保持局部可解释性，而长时间尺度（……（摘要在此处被截断）

    arXiv:2607.17117v2 Announce Type: replace-cross  Abstract: Sparse autoencoders (SAEs) decompose language model activations into sparse features, yet these models traditionally encode each token independently, failing to expose information that persists across a sequence. We first show that temporal persistence can naturally emerge in standard SAE features: after a feature activates, the hidden state remains aligned with its direction, and past activations help reconstruct later hidden states. How long this lasts varies widely across features. We therefore introduce Persistent Sparse Autoencoders (Persistent SAEs), an extension of standard SAEs that learns a persistence coefficient for each feature, allowing the model to learn feature-specific timescales from reconstruction alone. Our experiments show that Persistent SAEs retain competitive reconstruction quality while learning a spectrum of timescales: short-timescale (fast) features stay locally interpretable, whereas long-timescale (
    
[^195]: 真理方向的解剖：小语言模型中的知识依赖维度、关系定律与共享类别几何

    The Anatomy of a Truth Direction: Knowledge-Dependent Dimensionality, a Relational Law, and a Shared Category Geometry in Small Language Models

    [https://arxiv.org/abs/2607.16741](https://arxiv.org/abs/2607.16741)

    本文提出一种无需训练的SVD主方向方法来解剖语言模型中的真值表征，揭示其具有知识依赖的维度、遵循关系定律并呈现共享的类别几何结构，并可在6个架构家族的14个模型上以O(d)成本高效读取真值。

    

    Bürger等人（2024）证明了大型语言模型中的真理表征在陈述极性上是通用的，但存在于一个多维子空间中。陈述的真值可以从语言模型的残差流中线性读取，但目前尚不清楚这种表示中有多少能够拟合到一个单一方向上，是哪个组件构建了它，以及它究竟由什么构成。我们围绕这些问题开展了一项研究，仅使用一种工具：一条无需训练的轴，即通过真/假最小对的隐藏状态差异进行奇异值分解（SVD）得到的主方向，它无需标签即可识别（至多差一个全局符号）。我们在来自6个不同架构家族（包括混合专家MoE）的14个模型上进行了广泛评估，能够以每个token O(d)的成本读取和提取真值。最后，我们给出了一个预先注册的预测，探讨这种排列结构是否同样适用于那些真值是通过计算而非检索获得的类别。

    arXiv:2607.16741v3 Announce Type: replace  Abstract: B\"urger et al.\ (2024) demonstrated that truth representations in large language models are universal across statement polarity but reside within a multidimensional subspace. The truth value of a statement is linearly readable from a residual stream of language model, but it is not clear how much of that representation fits on a single direction, which component builds it, or what it is made of. We conducted a study based on these questions, with one instrument: a training-free axis, the dominant direction of the singular value decomposition (SVD) of hidden-state differences over true/false minimal pairs, identified without labels up to one global sign. Extensive evaluation across 14 models from 6 diverse architectural families (including MoE), read and extract at cost $O(d)$ per token. We close with a pre-registered prediction on whether the arrangement extends to categories whose truth is computed rather than retrieved.
    
[^196]: 符号分支重复惩罚中的规范依赖性与结构化输出损坏：跨模型、推理栈及替代重复控制方法的测量

    Gauge dependence and structured-output corruption in sign-branched repetition penalties: measurements across models, inference stacks, and alternative repetition controls

    [https://arxiv.org/abs/2607.09791](https://arxiv.org/abs/2607.09791)

    该论文揭示了主流推理引擎中的符号分支乘法重复惩罚依赖于 logit 任意零点（规范选择），导致惩罚操作缺乏良好定义且在不同模型上效果各异，并会使 JSON 结构化输出的有效率从 97% 骤降至 23%，同时提出了减法式与归一化等不受规范影响的替代方案。

    

    部署于整个大语言模型推理生态系统中（HuggingFace、vLLM、llama.cpp 以及十几个其他推理引擎）的乘法重复惩罚会根据每个原始 logit 的符号进行分支运算（正数除以 theta，负数乘以 theta）。但 softmax 对于给所有 logit 加上一个常数是不变的，因此模型的 logit 零点是任意的（一种规范/规范自由度选择），而符号分支却读取了这个零点。由此产生两个可测量的后果：(1) 该惩罚没有良好定义：对模型的 logit 进行常数重新中心化在 theta=1 时被证明是无操作（no-op），但在常规设置 theta=1.3 下，它会改变 58-96% 的贪心解码 token，而减法式惩罚和归一化惩罚则不改变任何 token；真实的模型检查点处于差异巨大的零点上，因此固定的 repetition_penalty 在每个模型上实际上是不同的操作。(2) 它会破坏结构化输出：在 200 个真实世界的 JSON 模式（schema）上，theta=1.3 将有效且符合模式的输出比例从 97% 降至 23%。应用该惩罚……（摘要在此处截断）

    arXiv:2607.09791v2 Announce Type: replace-cross  Abstract: The multiplicative repetition penalty shipped across the LLM inference ecosystem (HuggingFace, vLLM, llama$.$cpp, and a dozen further engines) branches on the sign of each raw logit (divide positives by theta, multiply negatives). But the softmax is unchanged by adding a constant to every logit, so a model's logit zero-point is arbitrary (a gauge choice), and the sign-branch reads it. Two measurable consequences follow. (1) The penalty is not well-defined: re-centering a model's logits by a constant is a provable no-op at theta=1, yet at a routine theta=1.3 it changes 58-96% of greedy tokens, while subtractive and normalized penalties change none; real checkpoints sit at widely different zero-points, so a fixed repetition_penalty is a different operation on every model. (2) It corrupts structured output: on 200 real-world JSON schemas, theta=1.3 drops the rate of valid, schema-conformant output from 97% to 23%. Applying the pen
    
[^197]: 所见即所得：面向图表到代码生成的观察对齐监督

    What You See Is What You Get: Observation-Aligned Supervision for Chart-to-Code Generation

    [https://arxiv.org/abs/2607.04726](https://arxiv.org/abs/2607.04726)

    论文揭示了图表到代码生成训练中存在的四类潜在变量与观察图像不匹配问题，并提出观察对齐监督方法，用视觉上可约束的量替换潜在变量作为监督目标。

    

    图表到代码生成通常通过对参考绘图脚本进行监督微调来训练，这隐式地将黄金代码视为完全可观察的目标。然而，许多图表程序包含无法从渲染图像中唯一恢复的潜在变量。我们在五种图表类型中识别出这种潜在变量与观察不匹配问题的四种形式：聚合导致的不匹配，即原始样本被简化为箱线图统计量或直方图分箱统计；归一化导致的不匹配，即饼图中绝对尺度被移除；投影导致的不匹配，即三维信息在二维渲染中丢失；以及水平集导致的不匹配，即标量场只能通过选定的等高线被观察。这些不匹配引入了目标歧义，并要求模型生成图像本身无法支持的信息。我们提出观察对齐监督方法，用视觉上可约束的量来替换潜在变量。

    arXiv:2607.04726v4 Announce Type: replace  Abstract: Chart-to-code generation is commonly trained through supervised fine-tuning on reference plotting scripts, implicitly treating the gold code as a fully observable target. However, many chart programs contain latent variables that cannot be uniquely recovered from the rendered image. We identify this latent-observation mismatch in four forms across five chart types: aggregation-induced mismatch, where raw samples are reduced to box statistics or histogram bin masses; normalization-induced mismatch, where absolute scale is removed in pie charts; projection-induced mismatch, where 3D information is lost through 2D rendering; and level-set-induced mismatch, where a scalar field is observable only through selected contour lines. These mismatches introduce target ambiguity and require models to generate information unsupported by the image. We propose Observation-Aligned Supervision, which replaces latent variables with visually constraine
    
[^198]: 面向视觉-语言模型的文本提示AdaBoost提升方法

    AdaBoosting Text Prompts for Vision-Language Models

    [https://arxiv.org/abs/2607.00684](https://arxiv.org/abs/2607.00684)

    提出受AdaBoost启发的文本提示提升框架TPB，将每个基于文本提示的分类器作为弱学习器，通过显式聚焦错误分类的难例将其逐步聚合为强集成分类器，从而充分利用少样本监督提升视觉-语言模型的分类性能。

    

    预训练视觉-语言模型（VLM）的分类精度取决于文本提示的质量。手工设计的模板和大型语言模型（LLM）生成的描述不仅能提高预测的可解释性，还能使相同的提示在不同异构视觉-语言模型之间复用。近期的工作利用少量带标签图像构建任务适配的文本提示。然而，现有的少样本文本提示方法在构建提示时并未显式关注被错误分类的样本，导致即使可用的样本数量增加，性能提升也仅是边际性的。为了充分利用少样本监督，我们提出了文本提示提升方法，这是一个受AdaBoost启发的框架，它将每个基于文本提示的分类器视为弱学习器，并通过显式针对困难且被错误分类的样本，将它们依次聚合为一个强集成分类器。大量实验表明，TPB在保持任务内在特性、模型……

    arXiv:2607.00684v4 Announce Type: replace  Abstract: The classification accuracy of pretrained Vision-Language Models (VLMs) relies on the quality of the text prompts. Handcrafted templates and Large Language Model (LLM)-generated descriptions not only make predictions more interpretable, but also enable reuse of the same prompts across heterogeneous VLMs. Recent works construct task-adapted text prompts with a small number of labeled images. However, existing few-shot text prompting methods do not explicitly focus on misclassified examples during prompt construction, leading to only marginal improvements even as more shots become available. To fully exploit few-shot supervision, we propose Text Prompt Boosting (TPB), an AdaBoost-inspired framework that treats each text-prompt-based classifier as a weak learner and sequentially aggregates them into a strong ensemble by explicitly targeting hard, misclassified examples. Extensive experiments show that TPB preserves task-intrinsic, model
    
[^199]: SABER-Math：面向数学信息检索评估的自动化基准

    SABER-Math: Automated Benchmark for Information Retrieval Evaluation in Mathematics

    [https://arxiv.org/abs/2606.29894](https://arxiv.org/abs/2606.29894)

    该论文提出了首个无需专家标注、完全自动化的数学信息检索评估基准SABER-Math，它从28.3万道高中数学题出发自动构建具有挑战性的重排序任务，以克服现有基准无法捕捉细粒度数学相关性的问题。

    

    随着智能体AI系统处理越来越复杂的数学任务，它们越来越依赖信息检索（IR）来搜索问题数据库、定理库和教育资源。然而，选择合适的检索器仍然很困难，因为无法直接将其对下游性能的影响隔离开来加以评估。另一方面，现有的检索专用基准往往无法捕捉细粒度的数学相关性，从而错误地惩罚相关文档。我们通过引入SABER-Math来填补这一空白，这是首个无需专家标注、完全自动化的数学信息检索评估基准。SABER-Math从28.3万道带解答的高中数学题目出发，通过三个步骤构建具有挑战性的重排序任务：(i) 首先，大语言模型为每道题目提取简洁的解题摘要和数学主题；(ii) 然后，利用基于本体主题和词汇解答相似性的方法为每个查询发现相关文档……（原文摘要在此处截断）

    arXiv:2606.29894v2 Announce Type: replace-cross  Abstract: As agentic AI systems tackle more complex mathematical tasks, they increasingly rely on information retrieval (IR) to search problem databases, theorem libraries, and educational resources. However, choosing the right retriever remains difficult, as it is infeasible to directly isolate its effect on downstream performance. On the other hand, existing retrieval-specific benchmarks often fail to capture fine-grained mathematical relevance, penalizing relevant documents. We address this gap by introducing SABER-Math, the first fully automated benchmark for evaluating mathematical IR without expert annotation. Starting from 283K high-school-level math problems with solutions, SABER-Math builds challenging reranking tasks in three steps: (i) first, LLMs extract concise solution summaries and mathematical topics for each problem; (ii) then, per-query relevant documents are discovered using ontology topic-based and lexical solutions-s
    
[^200]: TaLK：通过耦合语言模型与图感知核的文本属性图数据集蒸馏

    TaLK: Text-attributed Graph Dataset Distillation via Coupling Language Model with Graph-Aware Kernel

    [https://arxiv.org/abs/2606.22975](https://arxiv.org/abs/2606.22975)

    提出TaLK方法，通过耦合语言模型与图感知核，实现文本属性图的高效数据集蒸馏，避免重复训练完整模型，同时兼顾文本与结构信息。

    

    文本属性图（TAGs）广泛应用于许多现实领域，对其学习需要联合建模文本语义和图结构。标准方法是将语言模型（LM）与图神经网络（GNN）结合，但联合训练计算成本高且难以扩展。数据集蒸馏是降低训练成本的有效途径，但现有方法不适用于TAGs，因为它们通常针对单一模态设计，或在蒸馏过程中仍需反复训练完整的LM-GNN模型。为解决此问题，我们提出TaLK，一种有效的TAGs数据集蒸馏方法，它将LM与图感知神经切线核耦合。该设计实现了高效的数据集蒸馏，避免了对完整数据集的重复联合训练，同时反映了文本和结构信息，以支持有效的TAG学习。

    arXiv:2606.22975v2 Announce Type: replace  Abstract: Text-attributed graphs (TAGs) are widely used in many real-world domains, and learning on TAGs requires jointly modeling text semantics and graph structure. A standard approach for modeling TAGs is to combine a language model (LM) and a graph neural network (GNN), but joint training is computationally expensive and difficult to scale. Dataset distillation is a promising way to reduce training costs, but existing methods are not well suited to TAGs because they are typically designed for a single modality or still require repeatedly training expensive LM-GNN models on the full dataset during distillation. To address this, we propose TaLK, an effective dataset distillation method for TAGs that couples an LM with a graph-aware neural tangent kernel. This design enables efficient dataset distillation, avoiding repeated joint training on the full dataset while reflecting both textual and structural information for effective TAG learning. 
    
[^201]: 目标-行为对齐：多目标强化学习策略选择的诊断方法

    Objective-Behavior Alignment: Diagnostics for MORL Policy Selection

    [https://arxiv.org/abs/2606.21321](https://arxiv.org/abs/2606.21321)

    该论文提出了一种诊断工作流程，能够自动揭示多目标强化学习帕累托前沿上仅凭目标价值向量无法发现的行为差异，帮助决策者更全面地进行策略选择。

    

    现实世界的决策通常需要同时优化多个相互冲突的目标。在强化学习（RL）中，这通常通过标量化函数将奖励信号组合成单一标量目标来解决，但这种方法可能很脆弱：权重的微小变化就可能导致截然不同的策略。多目标强化学习（MORL）则生成一组明确表示目标之间权衡关系的策略集合。然而，这些策略通常仅通过其价值向量呈现给决策者，这可能掩盖显著的行为差异：那些会产生不同轨迹的策略，在仅依据期望回报进行评估时可能看起来毫无区别。我们提出了一种探索性诊断工作流程，能够自动突出帕累托前沿上仅凭目标值无法揭示的行为差异，提供定量和可视化……（摘要原文在此处被截断）

    arXiv:2606.21321v2 Announce Type: replace  Abstract: Real-world decision-making often requires optimizing multiple competing objectives simultaneously. In reinforcement learning (RL), this is typically addressed by combining reward signals into a single scalar objective via a scalarization function, which can be fragile: small changes in the weights can induce drastically different policies. Multi-objective reinforcement learning (MORL) instead produces sets of policies that explicitly represent trade-offs between objectives. However, these policies are typically presented to the decision maker only through their value vectors, which can obscure substantial behavioral variation: policies that induce distinct trajectories may appear indistinguishable when evaluated solely by expected returns. We propose an exploratory diagnostic workflow that automatically highlights behavioral variation along the Pareto front that objective values alone do not reveal, providing both quantitative and vi
    
[^202]: MM++：基于Top-K门控特征融合的事后尺度不变多层分布外检测

    MM++: Post-Hoc Scale-Invariant Multilayer OOD Detection via Top-K Gated Feature Fusion

    [https://arxiv.org/abs/2606.17352](https://arxiv.org/abs/2606.17352)

    MM++提出了一种无需辅助数据或架构修改的事后尺度不变多层OOD检测框架，通过熵密度下降选取判别中间层并进行门控特征融合，再利用Ledoit-Wolf正则化捆绑协方差稳定联合特征空间，从而在近远OOD检测中均取得鲁棒性能。

    

    我们提出了MM++（Multilayer Mahalanobis++，多层马氏距离++），一个严格事后（post-hoc）且尺度不变的分布外（OOD）检测框架。为了解决尺度不变性与层次表达能力之间的权衡问题，MM++构建了一个有原则的联合特征空间。该框架首先通过测量熵密度的下降来识别具有判别力的中间层，这些下降标志着急剧语义压缩的边界。通过将选定的这些层与终端表示进行融合，框架在捕捉潜在跨层相关性的同时减轻了早期层的噪声。至关重要的是，Ledoit-Wolf正则化的捆绑协方差矩阵稳定了这一统一空间，实现了可靠的距离估计。MM++无需辅助OOD数据、分类器微调或架构修改，即可在近OOD和远OOD检测任务中跨不同架构提供鲁棒的性能。

    arXiv:2606.17352v2 Announce Type: replace  Abstract: We introduce MM++ (Multilayer Mahalanobis++), a strictly post-hoc, and scale-invariant framework for Out-of-Distribution (OOD) detection. To address the trade-off between scale invariance and hierarchical expressivity, MM++ constructs a principled joint feature space. It first identifies discriminative intermediate layers by measuring entropy density drops, which mark the boundaries of sharp semantic compression. By fusing these selected layers with the terminal representation, the framework captures latent cross-layer correlations while mitigating early-layer noise. Crucially, a Ledoit-Wolf regularized tied covariance matrix stabilizes this unified space, enabling reliable distance estimation. Requiring no auxiliary OOD data, classifier fine-tuning, or architectural modifications, MM++ delivers robust performance across distinct architectures for both near- and far-OOD detection.
    
[^203]: 医学启发式学习：一种由大语言模型驱动的可解释、可审计临床决策规则框架

    Medical Heuristic Learning: An LLM-Driven Framework for Interpretable and Auditable Clinical Decision Rules

    [https://arxiv.org/abs/2606.16337](https://arxiv.org/abs/2606.16337)

    提出医学启发式学习（MHL）框架，利用大语言模型通过统计探针、医学知识探针、初始规则合成与迭代规则优化，构建完全以自然语言表达、可解释且可审计的临床决策规则专家系统，并能应对小样本、类别不平衡和特征演变等实际临床约束。

    

    用于临床决策支持的预测建模既需要强大的预测性能，也需要透明、可审计且可供人工审查的决策逻辑。尽管深度学习和基于树的集成方法可以达到较高的准确率，但其黑箱特性仍然是实现可信临床部署的主要障碍。此外，临床预测通常在实际约束下运行，包括样本量有限、严重的类别不平衡，以及因诊断标准或临床文档记录实践变化而导致的特征演变。我们提出了医学启发式学习，这是一种由大语言模型辅助的规则学习的受限范式。MHL不依赖于对隐式模型权重的更新，而是整合了统计探针、医学知识探针、初始规则合成和迭代规则优化，以构建一个可执行的基于规则的专家系统。所得到的规则系统完全使用自然语言表达……

    arXiv:2606.16337v4 Announce Type: replace  Abstract: Predictive modeling for clinical decision support requires both strong predictive performance and transparent, auditable, and human-reviewable decision logic. Although deep learning and tree-based ensemble methods can achieve high accuracy, their black-box nature remains a major obstacle to trustworthy clinical deployment. Moreover, clinical prediction often operates under practical constraints, including limited sample sizes, severe class imbalance, and feature evolution arising from changes in diagnostic criteria or clinical documentation practices. We propose Medical Heuristic Learning (MHL), a constrained paradigm for LLM-assisted rule learning. Rather than relying on updates to implicit model weights, MHL integrates statistical probes, medical knowledge probes, initial rule synthesis, and iterative rule optimization to construct an executable rule-based expert system. The resulting rule system is expressed entirely using the nat
    
[^204]: 情绪调节改善基于深度学习的图像分类

    Emotional regulation improves deep learning-based image classification

    [https://arxiv.org/abs/2606.13081](https://arxiv.org/abs/2606.13081)

    该论文提出了“情绪调节”这一新框架，通过人工主观体验建模情绪，利用基于情感刺激的预训练平衡情绪与非情绪响应，从而提升了ResNet和ViT在图像分类任务上的性能。

    

    情绪显著影响认知，在某些条件下能够增强记忆和学习。基于这一原理，情绪增强深度学习研究了情感状态如何改善神经网络架构和学习范式，从而实现比非情绪模型更好的泛化能力。然而，现有方法往往仅依赖于客观的神经生理因素，忽视了主观性在情绪中的作用。为弥补这一空白，本研究引入了“情绪调节”，这是一种通过人工主观体验在深度学习中建模情绪的新型框架。该方法采用基于情感刺激的预训练，在下游任务优化中平衡非情绪反应和受情绪影响的反应。研究在图像分类任务上进行了大量实验，在四个情绪数据集上对ResNet和ViT架构进行预训练，并以CIFAR-10和CIFAR-100作为目标数据集。

    arXiv:2606.13081v2 Announce Type: replace-cross  Abstract: Emotion significantly influences cognition, enhancing memory and learning under certain conditions. Drawing on this principle, emotion-augmented deep learning investigates how affective states can improve neural network architectures and learning paradigms, achieving better generalization than non-emotional models. However, existing methods often rely solely on objective neurophysiological factors, neglecting the role of subjectivity in emotion. To bridge this gap, the present study introduces Emotional Regulation, a novel framework for modeling emotion in deep learning through artificial subjective experience. The method employs pre-training based on affective stimuli, balancing non-emotional and emotionally-influenced responses in downstream task optimization. Extensive experimentation was conducted in image classification, pre-training ResNet and ViT architectures on four emotional datasets, using CIFAR-10 and -100 as target
    
[^205]: 一种用于车辆空气动力学预测的几何感知三平面场网络

    A Geometry-Aware Triplane Field Network for Vehicle Aerodynamic Prediction

    [https://arxiv.org/abs/2606.07724](https://arxiv.org/abs/2606.07724)

    提出几何感知三平面场网络GTF-Net，通过从表面点构建三平面特征并采用AFNO谱混合与CNN细化相结合的双流主干，在统一表示中同时捕捉全局气动耦合与局部几何细节，实现快速的车辆气动压力与壁面剪切应力预测。

    

    高保真计算流体动力学（CFD）对于车辆空气动力学分析至关重要，但其高昂的成本仍然限制了早期设计探索。如果模型能够高效地同时捕捉全局流动上下文和局部几何细节，基于机器学习的表面场预测可以提供一种更快速的替代方案。本工作提出了一种基于机器学习的方法，称为几何感知三平面场网络，用于车辆气动压力和壁面剪切应力的预测。GTF-Net 通过共享的多层感知机（MLP）和平滑双线性栅格化，直接从采样的表面点构建三平面特征。随后，这些平面由一个双流主干网络进行处理，该网络将自适应傅里叶神经算子（AFNO）的谱混合与卷积神经网络（CNN）的细化相结合，从而在同一表示中同时建模长程气动耦合和局部几何引起的变化。在……

    arXiv:2606.07724v2 Announce Type: replace  Abstract: High-fidelity computational fluid dynamics (CFD) is crucial to vehicle aerodynamic analysis, but its cost still constrains early-stage design exploration. Machine-learning-based surface-field prediction offers a faster alternative if the model can efficiently capture both global flow context and local geometric detail. This work proposes a machine-learning-based method, named the geometry-aware triplane field network (GTF-Net), for vehicle aerodynamic pressure and wall shear stress prediction. GTF-Net constructs triplane features directly from sampled surface points through a shared multilayer perceptron (MLP) and smooth bilinear rasterization. The planes are then processed by a dual-stream backbone that combines adaptive Fourier neural operator (AFNO) spectral mixing with convolutional neural network (CNN) refinement, so long-range aerodynamic coupling and local geometry-induced variations are modeled in the same representation. At 
    
[^206]: WhiFlash：通过令牌级跨范式路由加速投机解码

    WhiFlash: Accelerating Speculative Decoding with Token-Level Cross-Paradigm Routing

    [https://arxiv.org/abs/2606.07710](https://arxiv.org/abs/2606.07710)

    WhiFlash提出首个跨范式投机解码方法，在单一令牌级控制器下统一自回归与扩散并行草稿生成，并通过基于熵或神经策略的细粒度路由自适应应对草稿准确率的动态波动，从而显著加速大语言模型推理。

    

    大型语言模型（LLM）的自回归特性仍然是推理的重要瓶颈，尤其是在复杂的智能体工作负载中。虽然投机解码（SD）可以加速推理，但当前方法依赖于静态的草稿生成范式，要么使用自回归草稿模型进行推理，要么使用基于扩散的并行草稿模型来处理结构化输出。我们从实证中发现，草稿准确率在单个序列内部会剧烈波动，导致静态范式和粗粒度路由无法实现显著的性能提升。为了解决这种波动性，我们提出了WhiFlash，这是首个跨范式的投机解码方法，它在单一的令牌级控制器下统一了自回归草稿和基于扩散的并行草稿。WhiFlash采用了细粒度的路由机制，可使用轻量级的基于熵的方法或学习到的神经策略，两者都经过参数化以提供可调节的平衡……

    arXiv:2606.07710v2 Announce Type: replace-cross  Abstract: The autoregressive nature of large language models (LLMs) remains a significant bottleneck for inference, particularly in complex agentic workloads. While speculative decoding (SD) accelerates inference, current approaches rely on static drafting paradigms, utilising either autoregressive drafting models for reasoning or diffusion-based parallel drafting models for structured outputs. We empirically find that drafting accuracy fluctuates dramatically within a single sequence, leaving significant performance unrealised by static paradigms and coarse-grained routing. To address this volatility, we introduce WhiFlash, the first cross-paradigm SD method that unifies autoregressive and diffusion-based parallel drafting under a single token-level controller. WhiFlash adopts a fine-grained routing mechanism that employs either a lightweight entropy-based or a learned neural policy, both parametrised to provide a tunable balance betwee
    
[^207]: DOG-DPO：面向安全对齐的几何动态优化

    DOG-DPO:Dynamic Optimization in Geometry for Safety Alignment

    [https://arxiv.org/abs/2606.07678](https://arxiv.org/abs/2606.07678)

    提出无需训练的数据选择框架DOG-DPO，将偏好对视为模型表示空间中的几何方向，通过分解全局锚定子空间与数据集特有残余子空间并最大化多样性覆盖，为DPO安全对齐筛选出广泛且非冗余的偏好数据子集。

    

    大语言模型的安全对齐依赖于偏好数据，但当前的训练流程往往使用庞大且冗余的数据集。现有的数据选择方法通常独立地对每个偏好对进行打分，将方向性的偏好信息压缩为标量化的质量或多样性分数。这种以样本为中心的视角在多数据集场景下尤为受限，因为共享的安全方向与数据集特有的残余风险同时存在。我们提出了DOG-DPO，一个无需训练的数据选择框架，它将偏好对视为结构化的几何信号。DOG-DPO首先将每个偏好对表示为模型表示空间中的一个方向；然后将多数据集的偏好几何分解为全局锚定子空间和数据集特有的残余子空间；最后通过最大化基于多样性的覆盖度来选择子集，从而在DPO训练之前实现对安全对齐方向的广泛、非冗余的覆盖。

    arXiv:2606.07678v3 Announce Type: replace-cross  Abstract: Safety alignment for large language models relies on preference data, but current pipelines often train on large, redundant datasets. Existing data selection methods typically score each preference pair independently, collapsing directional preference information into scalar quality or diversity scores. This sample-centric view is especially limiting in multi-dataset settings, where shared safety directions coexist with dataset-specific residual risks. We propose DOG-DPO, a training-free data selection framework that treats preference pairs as structured geometric signals. DOG-DPO first represents each preference pair as a direction in model representation space. It then decomposes multi-dataset preference geometry into a global anchor subspace and dataset-specific residual subspaces. Finally, it selects subsets by maximizing diversity-based coverage, encouraging broad, non-redundant coverage of alignment directions before DPO 
    
[^208]: 为扩散语言模型启用共享前缀的KV缓存

    Enabling KV Caching of Shared Prefix for Diffusion Language Models

    [https://arxiv.org/abs/2606.07571](https://arxiv.org/abs/2606.07571)

    本文提出bicache，首个针对扩散语言模型共享前缀的KV缓存技术，解决了双向注意力下KV动态变化导致传统缓存失效的问题。

    

    针对共享前缀的键值（KV）缓存对于高吞吐量的大语言模型（LLM）服务至关重要，但在新兴的扩散语言模型（DLM）中面临重大挑战。在DLM中，双向注意力意味着更新任何一个token都会动态改变整个上下文及其对应的KV。因此，现有为LLM开发的缓存技术（其假设KV一旦计算完成就保持不变）会破坏共享前缀的KV。我们的实验表明，将这些技术应用于DLM会导致模型准确率崩溃至接近零。为了实现高吞吐量的DLM服务，我们提出了双向前缀缓存bicache，这是首个针对DLM共享前缀的KV缓存技术。

    arXiv:2606.07571v3 Announce Type: replace-cross  Abstract: Key-value (KV) caching for shared prefixes is essential for high-throughput large language model (LLM) serving, but it faces critical challenges in emerging diffusion language models (DLMs). In DLMs, bidirectional attention means that updating any token dynamically alters the entire context and its corresponding KVs. Thus, existing caching techniques developed for LLMs, which assume that KVs remain invariant once computed, corrupt the shared prefix KVs. Our experiments show that applying these techniques to DLMs causes model accuracy to collapse to near zero.   To unlock high-throughput DLM serving, we propose bidirectional prefix caching, bicache, the first KV caching technique for shared prefixes in DLMs. bicache is designed based on key observations from our comprehensive analysis: shared prefix KVs remain stable and reusable in shallow layers, while the depth of shallow layers depends on the fraction of shared prefix tokens
    
[^209]: 复空间中实数约束神经网络的缺陷与容量

    Shortcomings and capacities of real-constrained neural networks in complex spaces

    [https://arxiv.org/abs/2606.04390](https://arxiv.org/abs/2606.04390)

    本文通过文献中非标准的HCIZ积分公式，计算了复假设类中实数预激活约束神经网络与复数神经网络之间存储容量的渐近比率。

    

    我们发现了在复假设类中强制使用实数预激活与使用复数预激活时，两者存储容量之间的渐近比率。我们使用从复高斯分布中抽取的权重，其范数几乎必然地渐近收敛于维数的平方根。我们的方法依赖于临界容量处的Gardner体积类型比较。我们的证明依赖于Harish-Chandra-Itzykson-Zuber (HCIZ)公式的应用，这在文献中并不常见。借助HCIZ公式，我们可以对最终的渐近比率获得更稳健的近似。这一策略特别适用于我们的工作，因为我们借助Weyl积分公式和Haar测度在酉群和正交紧流形上进行积分。

    arXiv:2606.04390v3 Announce Type: replace  Abstract: We find the asymptotic ratio between the storage capacities when enforcing real pre-activations in a complex hypothesis class as opposed to complex ones in the same class. We use weights drawn from the complex Gaussian, which converge asymptotically in norm to the square root of dimension almost surely. Our methods depend on Gardner volume-type comparisons at critical capacity. Our proof relies on an application of the Harish-Chandra-Itzykson-Zuber (HCIZ) formula, nonstandard in literature. With the HCIZ formula, we may obtain a more robust approximation for the final asymptotic ratio. This strategy is applicable to our work specifically since we integrate over the unitary and orthogonal compact manifolds, facilitated via the Weyl integration formula and the Haar measure.
    
[^210]: 编码器-解码器神经算子的变分空间：逼近与泛化

    Variation Spaces for Encoder--Decoder Neural Operators: Approximation and Generalization

    [https://arxiv.org/abs/2606.01244](https://arxiv.org/abs/2606.01244)

    该论文基于有界变差向量值测度构建了神经算子的变分空间理论，证明了在ReLU激活下该空间与Schatten-1算子类范数等价，并建立了编码器-解码器神经算子的逼近误差界与高概率泛化界。

    

    受神经网络函数空间理论的启发，我们构建并分析了希尔伯特空间之间非线性算子的一个变分空间，该空间通过具有有界变差的向量值Borel测度来定义。我们将该空间的单位球刻画为Bochner空间中向量值单神经元字典的闭凸包。对于ReLU激活函数，该空间中的有界线性算子恰好是Schatten-1算子，且两者的范数等价。对于该空间中的算子，我们在Bochner $L^q$范数下建立了编码器-解码器逼近界，其中误差分解为输入和输出编码误差以及一个阶为 $N^{-1/2}$ 的有限宽度项。在输入和噪声满足次高斯假设的条件下，我们进一步为路径范数约束的编码器-解码器网络上的经验最小二乘推导了高概率泛化界；有限样本对平方预测误差的贡献……

    arXiv:2606.01244v2 Announce Type: replace-cross  Abstract: Inspired by the function-space theory of neural networks, we formulate and analyze a variation space for nonlinear operators between Hilbert spaces, defined through vector-valued Borel measures of bounded variation. We characterize its unit ball as the closed convex hull of a vector-valued single-neuron dictionary in Bochner spaces. For the ReLU activation, the bounded linear operators in this space are precisely the Schatten-$1$ operators, with equivalent norms. For operators in this space, we establish encoder--decoder approximation bounds in the Bochner $L^q$-norm, where the error decomposes into input and output encoding errors and a finite-width term of order $N^{-1/2}$. Under sub-Gaussian assumptions on the input and noise, we further derive high-probability generalization bounds for empirical least squares over path-norm-constrained encoder--decoder networks; the finite-sample contribution to the squared prediction error
    
[^211]: OptSkills：通过基于聚类的蒸馏从问题原型中学习可泛化的优化技能

    OptSkills: Learning Generalizable Optimization Skills from Problem Archetypes via Cluster-Based Distillation

    [https://arxiv.org/abs/2605.29829](https://arxiv.org/abs/2605.29829)

    该论文提出OptSkills系统，通过依据问题底层原型进行聚类并将成功求解轨迹蒸馏为可复用的工作流级技能，使LLM驱动的优化建模与求解在分布内和分布外问题上都具备更强的泛化能力。

    

    利用大型语言模型（LLM）从自然语言中自动构建并求解优化问题，已成为自动化优化的一种高效范式。然而，现有方法的泛化能力仍然有限：它们对表面的叙述变化较为敏感，主要在案例层面复用经验，并难以适应发生偏移或新出现的问题类型。我们提出了OptSkills，一个以原型为中心、面向优化建模与求解的技能学习与推理智能体系统。为了实现鲁棒的泛化能力，我们的系统依据问题的底层原型而非表面叙述对问题进行聚类。为了提升分布内泛化能力，系统在每个聚类内探索多样化的建模范式与求解器配置，然后将成功轨迹蒸馏为可复用的工作流级技能。为了提升分布外泛化能力，系统会精炼现有技能或扩展……（摘要原文在此处截断）

    arXiv:2605.29829v2 Announce Type: replace  Abstract: Leveraging Large Language Models (LLMs) to automatically formulate and solve optimization problems from natural language has emerged as an efficient paradigm for automated optimization. However, existing methods still exhibit limited generalization: they are sensitive to superficial narrative variations, reuse experience mainly at the case level, and struggle to adapt to shifted or emerging problem types. We propose OptSkills, an archetype-centric skill learning and reasoning agent system for optimization modeling and solving. To improve robust generalization, our system clusters problems by their underlying archetypes rather than surface narratives. To improve in-distribution generalization, it explores diverse modeling paradigms and solver configurations within each cluster, then distills successful trajectories into reusable workflow-level skills. To improve out-of-distribution generalization, it refines existing skills or expands
    
[^212]: 半真音频检测与定位：一种轻量级交叉注意力架构与跨语料库诊断研究

    Half-Truth Audio Detection and Localisation: A Lightweight Cross-Attentive Architecture and a Cross-Corpus Diagnostic Study

    [https://arxiv.org/abs/2605.29531](https://arxiv.org/abs/2605.29531)

    提出了轻量级交叉注意力网络CAFNet，可同时完成真实/全伪造/半真音频的三分类与合成片段的时间边界定位，在约14毫秒CPU延迟下达到97.55%的三分类准确率。

    

    部分伪造（半真）语音是指将一小段合成语音片段拼接进原本真实的语音中，相比文献中占主导地位的全合成深度伪造语音，这是一种更难且更现实的取证威胁。我们提出了CAFNet，一种轻量级（576K参数、2.24 MB）的交叉注意力架构，它融合MFCC、LFCC和Chroma-STFT特征，联合将音频分类为真实、全伪造或半真，并回归合成区域的时间边界，CPU延迟约为14毫秒。组件消融实验表明，交叉注意力融合是CAFNet最关键的组件；而来自早期版本的深度监督辅助分类头并非如此，在3个随机种子的重复实验中，移除该组件使所有域内指标均得到提升，且方差大幅降低。在MLADDC T2+T3数据集上，该模型达到97.55%±0.69%的三分类准确率和0.037秒的边界平均绝对误差（MAE）

    arXiv:2605.29531v3 Announce Type: replace-cross  Abstract: Partially manipulated (half-truth) speech, where a short synthesised segment is spliced into an otherwise genuine utterance, is a harder and more realistic forensic threat than the fully synthesised deepfakes that dominate the literature. We present CAFNet, a lightweight (576K-parameter, 2.24 MB) cross-attentive architecture that fuses MFCC, LFCC, and Chroma-STFT features to jointly classify audio as real, fully fake, or half-truth, and regress the temporal boundaries of the synthesised region, at approximately 14 ms CPU latency. A component ablation shows cross-attention fusion is CAFNet's most load-bearing component; a deeply supervised auxiliary classification head from earlier iterations is not, and removing it improves every in-domain metric under 3-seed replication with substantially lower variance. On MLADDC T2+T3 the model reaches 97.55%$\pm$0.69% ternary accuracy and 0.037 s boundary mean absolute error (MAE), to our k
    
[^213]: 推理原生零阶优化

    Inference-Native Zeroth-Order Optimization

    [https://arxiv.org/abs/2605.28760](https://arxiv.org/abs/2605.28760)

    该论文提出“推理原生零阶优化”，将零阶优化重新表述为推理运行时可直接执行的可编程候选状态查询（通过ProbePlan形式化），从而在无需反向传播和全参数状态物化的情况下完成优化。

    

    零阶（ZO）优化去除了反向传播，但传统实现仍然通过修改模型权重来创建候选状态，并通过完整的参数状态来物化更新。我们提出了推理原生ZO（Inference-Native ZO），它显式暴露了ZO的查询语义，并将候选状态评估与可变学习状态降低为推理运行时可以直接执行的抽象。我们将ZO表述为通过候选状态查询实现的可编程梯度获取。方向构建、候选选择、观测、估计和更新语义构成了一个查询过程，其面向模型的原语是候选评估。我们将该过程所需的逻辑查询形式化为ProbePlan，而将物理状态的实现与调度留给后端完成。因子化的辅助状态、持久子空间复用、惰性更新以及可选的LoRA库降低了状态管理成本。同一公式化还可覆盖token评分……（原文摘要在此处截断）

    arXiv:2605.28760v2 Announce Type: replace  Abstract: Zeroth-order (ZO) optimization removes backpropagation, but conventional implementations still create candidate states by mutating model weights and materialize updates through the full parameter state. We introduce Inference-Native ZO, which exposes ZO's query semantics and lowers candidate-state evaluation and mutable learning state to abstractions an inference runtime can execute directly.   We formulate ZO as programmable gradient acquisition through candidate-state queries. Direction construction, candidate selection, observation, estimation, and update semantics form a query process whose model-facing primitive is candidate evaluation. We formalize the logical queries required by that process as a ProbePlan, leaving physical state realization and scheduling to the backend. Factorized side states, persistent-subspace reuse, lazy updates, and optional LoRA banks reduce state-management cost. The same formulation covers token-scor
    
[^214]: Föllmer过程与去噪扩散概率模型之间的联系

    Connections between the F\"ollmer process and the denoising diffusion probabilistic model

    [https://arxiv.org/abs/2605.18040](https://arxiv.org/abs/2605.18040)

    本文阐明了离散化Föllmer过程与DDPM采样器之间的直接联系，证明其为DDPM采样器提供了自然的超参数设置，且能容纳比离散化反向SDE更广泛的方差调度，从而系统地恢复了最先进的DDPM采样误差界结果。

    

    Föllmer过程是一个在时刻1被条件化为具有预先指定分布的布朗运动。该过程可以被解释为对应于去噪扩散概率模型（DDPM）的反向随机微分方程（SDE）的一个“增广”时间压缩版本。虽然这一事实已被间接用于通过反向SDE的离散化来分析DDPM的采样误差，但Föllmer过程的直接离散化与DDPM采样器之间的联系尚未得到充分探索。本文在综述文献中相关结果的同时阐明了这一点。我们证明，离散化的Föllmer过程为DDPM采样器提供了自然的超参数设置，同时比离散化的反向SDE能够容纳更广泛一类的方差调度。此外，这使我们能够系统地恢复关于DDPM采样误差界的最先进结果，并得到略微……

    arXiv:2605.18040v2 Announce Type: replace-cross  Abstract: The F\"ollmer process is a Brownian motion conditioned to have a pre-specified distribution at time 1. This process can be interpreted as an ``augmented'' time-compressed version of the reverse stochastic differential equation (SDE) corresponding to the denoising diffusion probabilistic model (DDPM). While this fact has been indirectly used to analyze DDPM sampling errors via discretization of the reverse SDE, the connection between direct discretization of the F\"ollmer process and the DDPM sampler has not yet been fully explored. This paper clarifies this point while surveying relevant results from the literature. We show that discretized F\"ollmer processes give natural hyper-parameter settings of the DDPM sampler while accommodating a broader class of variance schedules than discretized reverse SDEs. Moreover, this allows us to systematically recover state-of-the-art results on DDPM sampling error bounds, along with slight 
    
[^215]: 当提示词相互作用：评估提示词算术在分布偏移下的去混杂能力

    When Prompts Interact: Assessing Prompt Arithmetic for Deconfounding under Distribution Shift

    [https://arxiv.org/abs/2605.03096](https://arxiv.org/abs/2605.03096)

    本文研究了通过任务算术组合软提示能否提升模型对混杂变量引起分布偏移的鲁棒性，并提出了一种混合提示算术方法来去除模型对虚假特征的依赖，相比完全微调更具计算效率。

    

    在分类任务中，模型可能依赖混杂变量来获得强大的分布内性能，捕获在分布偏移下失效的虚假特征。这种捷径行为会导致在分布外场景中出现显著的性能下降。任务算术提供了一种潜在的解决方案，通过减去次要模型更新来移除不需要的信号，但它通常需要完全微调，计算成本高昂。提示调优提供了一种参数高效的替代方案，通过一小组可训练的虚拟令牌来适配模型。对由此产生的提示词进行任务算术运算，为对整个模型进行操作提供了一种有吸引力的替代方法，但这种方法能在多大程度上限制对虚假特征的依赖仍有待验证。在这项工作中，我们研究了通过任务算术组合软提示是否能提高模型对混杂偏移的鲁棒性。我们提出了混合提示算术方法……

    arXiv:2605.03096v2 Announce Type: replace-cross  Abstract: In classification tasks, models may rely on confounding variables to achieve strong in-distribution performance, capturing spurious features that fail under distribution shift. This shortcut behavior leads to substantial degradation in out-of-distribution settings. Task arithmetic offers a potential solution by removing unwanted signals via subtraction of secondary model updates, but it typically requires full fine-tuning, which is computationally expensive. Prompt tuning provides a parameter-efficient alternative by adapting models through a small set of trainable virtual tokens. Task arithmetic on the resulting prompts presents an appealing alternative to operations on entire models, but the extent to which this approach can limit reliance on spurious features remains to be established. In this work, we study whether composing soft prompts through task arithmetic improves robustness to confounding shifts. We propose Hybrid Pr
    
[^216]: 通过各向异性目标扰动稳定异质协变量下的隐私保护LASSO

    Stabilizing Private LASSO under Heterogeneous Covariates via Anisotropic Objective Perturbation

    [https://arxiv.org/abs/2605.01492](https://arxiv.org/abs/2605.01492)

    该论文提出一种基于Gram矩阵的各向异性目标扰动“预失真”策略，通过抵消异质协变量结构引起的失真来稳定差分隐私下的高维LASSO估计，显著提升了收敛稳定性、统计效率和隐私性能。

    

    我们研究了在差分隐私下，针对具有异质协变量尺度的高维LASSO问题采用目标扰动方法。在实际场景中，协变量通常呈现不同的尺度；然而，在隐私约束下，标准预处理是有问题的，因为它会消耗额外的隐私预算。这种异质性通过协变量的逆Gram矩阵在目标扰动中引入了有效的各向异性，这会降低算法的稳定性和准确性。为解决这一问题，我们提出了一种基于Gram矩阵的各向异性目标扰动方法，这是一种“预失真”策略，通过抵消协变量结构带来的失真来恢复估计过程中的各向同性。利用近似消息传递（AMP）框架和状态演化分析，我们证明了与现有方法相比，我们提出的扰动方法显著稳定了收敛性，并提升了统计效率和隐私性能。

    arXiv:2605.01492v2 Announce Type: replace-cross  Abstract: We study high-dimensional LASSO under differential privacy via objective perturbation with heterogeneous covariate scales. In practical scenarios, covariates often exhibit diverse scales; however, standard preprocessing is problematic under privacy constraints, as it consumes additional privacy budget. This heterogeneity induces effective anisotropy in the objective perturbation via the inverse Gram matrix of covariates, which can degrade the stability and accuracy of algorithms. To address this, we propose a Gram-based anisotropic objective perturbation, a ``pre-distortion" strategy that counteracts the distortion from the covariate structure to restore isotropy in the estimation process. Using an Approximate Message Passing (AMP) framework and state evolution analysis, we demonstrate that our proposed perturbation significantly stabilizes convergence and improves both statistical efficiency and privacy performance compared to
    
[^217]: 语言扩散模型是能够检索未见数据的联想记忆

    Language Diffusion Models are Associative Memories Capable of Retrieving Unseen Data

    [https://arxiv.org/abs/2604.26841](https://arxiv.org/abs/2604.26841)

    该论文证明均匀离散扩散语言模型本质上是联想记忆，其吸引盆可通过条件似然最大化而非显式能量函数形成，并揭示了由数据规模支配的从记忆到泛化的急剧转变，使其能够检索未见过的数据。

    

    语言扩散模型何时会记忆其训练数据，以及如何定量评估其真正的生成机制？我们通过证明基于均匀分布的离散扩散模型（UDDMs）在根本上表现为具有涌现创造能力的联想记忆（AMs）来回答这些问题。联想记忆的核心思想是通过在存储的数据点周围建立独特的吸引盆，从而可靠地将这些数据点作为“记忆”恢复出来。历史上，像Hopfield网络这样的模型使用显式的能量函数来保证这些稳定吸引子的存在。我们拓展了这一视角，利用了一个关键观察：能量并非严格必需，因为吸引盆也可以通过条件似然最大化来形成。通过评估模型对训练样本和测试样本的词元恢复能力，我们在UDDMs中识别出一个由训练规模大小所支配的、从记忆到泛化的急剧转变（摘要在此处截断）。

    arXiv:2604.26841v2 Announce Type: replace-cross  Abstract: When do language diffusion models memorize their training data, and how to quantitatively assess their true generative regime? We address these questions by showing that Uniform-based Discrete Diffusion Models (UDDMs) fundamentally behave as Associative Memories (AMs) $\textit{with emergent creative capabilities}$. The core idea of an AM is to reliably recover stored data points as $\textit{memories}$ by establishing distinct basins of attraction around them. Historically, models like Hopfield networks use an explicit energy function to guarantee these stable attractors. We broaden this perspective by leveraging the observation that energy is not strictly necessary, as basins of attraction can also be formed via conditional likelihood maximization. By evaluating token recovery of $\textit{training}$ and $\textit{test}$ examples, we identify in UDDMs a sharp memorization-to-generalization transition governed by the size of the t
    
[^218]: RCProb：从分类树集成中提取概率规则

    RCProb: Probabilistic rule extraction from classification tree ensembles

    [https://arxiv.org/abs/2604.25304](https://arxiv.org/abs/2604.25304)

    RCProb是一种针对RuleCOSI+的概率扩展方法，通过平滑的原子类条件证据和支持自适应混合概率估计，显著提升了从分类树集成中提取规则的概率可靠性，将对数损失大幅降低。

    

    树集成提供了强大的分类性能，但通常表现为黑盒模型。诸如RuleCOSI+等事后可解释性技术会提取一个近似于集成模型的小型规则集，但这种简化可能使附加在提取规则上的概率变得不可靠。特别是，RuleCOSI+为提取的规则分配经验类别概率，并在其贪婪组合和简化过程中反复使用这些规则统计量。我们提出了RCProb，一种概率扩展方法，它在计算开销较大的搜索阶段使用平滑的原子类条件证据，并在最终规则概率中使用带有集成信息的m估计的支持自适应混合方法。该方法在18个二分类和5个多分类数据集上使用随机森林（RF）和梯度提升机（GBM）集成进行评估。相对于RuleCOSI+，RF的中位配对对数损失降低为71.9%，GBM为62.5%。

    arXiv:2604.25304v2 Announce Type: replace  Abstract: Tree ensembles provide strong classification performance but usually behave as black-box models. Post-hoc interpretability techniques such as RuleCOSI+ extract a small ruleset that approximates the ensemble, but this simplification can leave the probabilities attached to the extracted rules unreliable. In particular, RuleCOSI+ assigns empirical class probabilities to the extracted rules and repeatedly uses those rule statistics during its greedy combination and simplification procedure. We present RCProb, a probabilistic extension that uses smoothed atomic class-conditional evidence for the expensive search stages and a support-adaptive mixture with an ensemble-informed m-estimate for the final rule probabilities. The method is evaluated on 18 binary and 5 multiclass datasets using random forest (RF) and gradient boosting machine (GBM) ensembles. Relative to RuleCOSI+, the median paired log-loss reduction is 71.9\% for RF and 62.5\% 
    
[^219]: 用于稀疏视角CT重建的条件扩散后验对齐

    Conditional Diffusion Posterior Alignment for Sparse-View CT Reconstruction

    [https://arxiv.org/abs/2604.21960](https://arxiv.org/abs/2604.21960)

    提出条件扩散后验对齐（CDPA）方法，通过将条件扩散与显式数据一致性相结合，成功将基于扩散模型的稀疏视角CT重建扩展到大型3D体数据，克服了3D模型内存计算开销大、3D训练数据缺乏以及2D切片方法不一致等局限。

    

    计算机断层扫描（CT）是医疗和工业应用中广泛使用的一种成像方式。为了限制辐射暴露和测量时间，稀疏视角CT（即大幅减少投影视角数量的CT）正受到越来越多的关注。深度神经网络在提升稀疏视角CT重建质量方面展现出巨大潜力，尤其是生成式扩散模型。然而，这些方法难以扩展到大型三维体数据，原因包括：（i）3D模型对内存和计算的高要求，（ii）缺乏大规模3D训练数据集，以及（iii）在每个切片上独立使用2D模型时导致的切片间不一致。我们通过将条件扩散与显式数据一致性相结合，克服了这些限制，将基于扩散模型的稀疏视角CT重建扩展到大型3D体数据。我们提出了条件扩散后验对齐（CDPA）方法，以实现可扩展的……

    arXiv:2604.21960v3 Announce Type: replace-cross  Abstract: Computed Tomography (CT) is a widely used imaging modality in medical and industrial applications. To limit radiation exposure and measurement time, there is a growing interest in sparse-view CT, where the number of projection views is significantly reduced. Deep neural networks have shown great promise in improving reconstruction quality in sparse-view CT, especially generative diffusion models. However, these methods struggle to scale to large 3D volumes due to several reasons: (i) the high memory and computational requirements of 3D models, (ii) the lack of large 3D training datasets, and (iii) the inconsistencies across slices when using 2D models independently on each slice. We overcome these limitations and scale diffusion-based sparse-view CT reconstruction to large 3D volumes by combining conditional diffusion with explicit data consistency. We propose Conditional Diffusion Posterior Alignment (CDPA) to enable scalable 
    
[^220]: Stream-CQSA：面向注意力机制的精确显存溢出恢复

    Stream-CQSA: Exact Out-of-Memory Recovery for Attention

    [https://arxiv.org/abs/2604.20819](https://arxiv.org/abs/2604.20819)

    提出Stream-CQSA框架，基于循环法定人数集（CQS）理论递归地将无法装入显存的注意力调用分解为独立子任务并重组结果，可对任意被包装的注意力内核（无论精确或近似）实现精确的显存溢出恢复。

    

    长上下文大语言模型不仅受限于注意力计算成本，还受限于内存溢出（OOM）故障。即使内核已经过优化，某个被调用的注意力计算也可能无法装入可用的设备内存。精确与近似注意力方法虽能降低内存使用，但每种固定实现仍存在设备相关的容量上限。我们提出Stream-CQSA，一个基于CQS分解的注意力层OOM恢复框架，其理论基础是循环法定人数集理论。Stream-CQSA递归地将无法执行的注意力调用划分为独立的子序列任务，使用兼容的内部内核逐一执行，并重新组合局部统计量以恢复完整的注意力输出。相对于被包装的注意力内核，无论该内核是精确的还是近似的，这种恢复都是精确的。与主要基线FlashAttention-2相比，我们的原生Stream-CQSA内核在16位前向输出上取得了提升（摘要在此处被截断）。

    arXiv:2604.20819v2 Announce Type: replace  Abstract: Long-context large language models are limited not only by attention cost but also by out-of-memory (OOM) failures. A selected attention call may not fit in available device memory even when the kernel is optimized. Exact and approximate attention methods reduce memory use, but every fixed implementation still has a device-specific capacity boundary. We introduce Stream-CQSA, an attention-level OOM recovery framework based on CQS decomposition, derived from the theory of cyclic quorum sets (CQS). Stream-CQSA recursively partitions an infeasible attention call into independent subsequence tasks, executes each with a compatible inner kernel, and recomposes the local statistics to recover the full attention output. This recovery is exact relative to the wrapped attention kernel, whether that kernel is exact or approximate. Compared with FlashAttention-2, the major baseline, our native Stream-CQSA kernel improves 16-bit forward-output er
    
[^221]: 论多层状态空间模型（SSM）的表达能力与局限性

    On the Expressive Power and Limitations of Multi-Layer SSMs

    [https://arxiv.org/abs/2604.14501](https://arxiv.org/abs/2604.14501)

    多层状态空间模型（SSM）求解函数复合问题时必须满足深度、状态维度与标量精度之间的下界权衡（d²p=Ω(N/L³)），且输入后推理无法绕过该通信下界，从而揭示了多层SSM表达能力的基本限制。

    

    我们研究了深度、有限精度、状态维度以及思维链如何影响多层状态空间模型（SSM）的表达能力。针对显式表的K个函数复合问题——一个序列信息传播的规范基准任务——我们证明，任何求解(L+3)个函数复合的L层SSM都必须满足 d²p=Ω(N/L³)，其中 d 为状态维度，p 为每个标量的精度。反过来，K个函数复合可以由一个 d=1 且 p=Θ(log N) 的(K+1)层广义SSM精确求解。这为该形式化问题族给出了一个最坏情况下的深度层级结构。我们随后区分了“输入后推理”（所有思考标记均在输入之后生成）与“输入交错推理”（思考标记可以在读取输入流的过程中插入）。输入后推理无法绕过我们基于通信的下界证明流程，

    arXiv:2604.14501v2 Announce Type: replace-cross  Abstract: We study how depth, finite precision, state dimension, and chain-of-thought (CoT) affect the expressive power of multi-layer state-space models (SSMs). For the explicit-table $K$-function-composition problem, a canonical benchmark for sequential information propagation, we prove that any $L$-layer SSM solving $(L+3)$-function composition must satisfy $d^2p=\Omega(N/L^3)$, where $d$ is the state dimension and $p$ is the per-scalar precision. Conversely, $K$-function composition is solved exactly by a $(K+1)$-layer generalized SSM with $d=1$ and $p=\Theta(\log N)$. This gives a worst-case depth hierarchy for this formal problem family. We then distinguish post-input reasoning, in which all thought tokens are generated after the input, from input-interleaved reasoning, in which thought tokens may be inserted while the input stream is being read. Post-input reasoning does not circumvent our communication-based lower-bound pipeline,
    
[^222]: 超越状态一致性：基于文本的世界模型中的行为一致性

    Beyond State Consistency: Behavior Consistency in Text-Based World Models

    [https://arxiv.org/abs/2604.13824](https://arxiv.org/abs/2604.13824)

    该论文提出了一种新的行为对齐训练范式，通过优化行为一致性奖励（BehR）这一步骤级指标来衡量智能体动作似然在真实状态与预测状态之间的变化，从而提升基于文本的世界模型与真实环境之间的功能一致性。

    

    世界模型已成为评估交互式智能体在在线规划和离线评估中所生成动作后果的关键组件。在基于文本的环境中，世界模型通常使用精确匹配（Exact Match）等单步指标进行评估和训练，旨在提高预测状态与真实世界状态之间的相似度，但已有研究表明此类指标不足以捕捉智能体的实际行为。为解决这一问题，我们提出了一种新的行为对齐训练范式，旨在提高世界模型与真实环境之间的功能一致性。该范式专注于优化一个可计算的步骤级指标——行为一致性奖励，该指标在冻结的参考智能体下，衡量已记录的下一个动作的似然在真实状态与世界模型预测状态之间的变化程度。在WebShop和TextWorld上的实验表明（摘要此处截断）

    arXiv:2604.13824v2 Announce Type: replace  Abstract: World models have been emerging as critical components for assessing the consequences of actions generated by interactive agents in online planning and offline evaluation. In text-based environments, world models are typically evaluated and trained with single-step metrics such as Exact Match, aiming to improve the similarity between predicted and real-world states, but such metrics have been shown to be insufficient for capturing actual agent behavior. To address this issue, we introduce a new behavior-aligned training paradigm aimed at improving the functional consistency between the world model and the real environment. This paradigm focuses on optimizing a tractable step-level metric named Behavior Consistency Reward (BehR), which measures how much the likelihood of a logged next action changes between the real state and the world-model-predicted state under a frozen Reference Agent. Experiments on WebShop and TextWorld show that
    
[^223]: 从高维空间到面向安全关键AI系统的可验证运行设计域（ODD）覆盖

    From High-Dimensional Spaces to Verifiable ODD Coverage for Safety-Critical AI-based Systems

    [https://arxiv.org/abs/2604.02198](https://arxiv.org/abs/2604.02198)

    本文提出一种融合参数离散化、基于约束的过滤和基于关键性的降维的方法，弥合了抽象ODD定义与可验证证据之间的鸿沟，为安全关键AI系统（如航空领域）满足EASA认证中ODD完全覆盖要求提供了可验证的工程途径。

    

    虽然人工智能（AI）为运营性能带来了变革性潜力，但其在航空等安全关键领域的部署需要严格遵守严格的认证标准。当前EASA（欧盟航空安全局）指南要求证明AI/ML组件的运行设计域（ODD）的完全覆盖——这一要求需要证明在定义的运行边界内不存在关键性空白。然而，由于系统运行在高维参数空间中，现有方法难以提供满足完整性准则所需的可扩展性和形式化基础。目前，尚无标准化的工程方法来弥合抽象ODD定义与可验证证据之间的鸿沟。本文针对这一空白，提出了一种将参数离散化、基于约束的过滤和基于关键性的降维整合到一个结构中（的方法），从而……

    arXiv:2604.02198v2 Announce Type: replace  Abstract: While Artificial Intelligence (AI) offers transformative potential for operational performance, its deployment in safety-critical domains such as aviation requires strict adherence to rigorous certification standards. Current EASA guidelines mandate demonstrating complete coverage of the AI/ML constituent's Operational Design Domain (ODD) -- a requirement that demands proof that no critical gaps exist within defined operational boundaries. However, as systems operate within high-dimensional parameter spaces, existing methods struggle to provide the scalability and formal grounding necessary to satisfy the completeness criterion. Currently, no standardized engineering method exists to bridge the gap between abstract ODD definitions and verifiable evidence. This paper addresses this void by proposing a method that integrates parameter discretization, constraint-based filtering, and criticality-based dimension reduction into a structure
    
[^224]: 在移动边缘训练：面向大推理模型高效强化学习训练的在线验证提示选择

    Train at Moving Edge: Online-Verified Prompt Selection for Efficient RL Training of Large Reasoning Model

    [https://arxiv.org/abs/2603.25184](https://arxiv.org/abs/2603.25184)

    提出HIVE双阶段框架，通过历史奖励轨迹与在线验证在采样前选取处于“学习边缘”（中等难度且高不确定性）的高效用提示，显著提升大推理模型强化学习训练的数据效率。

    

    强化学习（RL）已成为大语言模型（LLM）在推理任务上进行后训练的关键技术。虽然扩大采样轮次可以稳定训练并提升性能，但其计算开销是一个关键问题。在GRPO等算法中，每个提示（prompt）需要进行多次采样，这带来了极高的成本，因为很大一部分提示所提供的梯度几乎可以忽略不计，因而效用较低。为了解决这一问题，我们研究了如何在采样阶段之前选取高效用的提示。我们的实验分析表明，样本效用是非均匀且不断演变的：最强的学习信号集中在“学习边缘”——即中等难度与高不确定性的交汇处，且这一边缘会随着训练的推进而移动。受此启发，我们提出了HIVE（基于历史信息与在线验证的提示选择），这是一个面向数据高效强化学习的双阶段框架。HIVE利用历史奖励轨迹……（摘要原文在此处截断）

    arXiv:2603.25184v3 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) has become essential for post-training large language models (LLMs) in reasoning tasks. While scaling rollouts can stabilize training and enhance performance, the computational overhead is a critical issue. In algorithms like GRPO, multiple rollouts per prompt incur prohibitive costs, as a large portion of prompts provide negligible gradients and are thus of low utility. To address this problem, we investigate how to select high-utility prompts before the rollout phase. Our experimental analysis reveals that sample utility is non-uniform and evolving: the strongest learning signals concentrate at the ``learning edge", the intersection of intermediate difficulty and high uncertainty, which shifts as training proceeds. Motivated by this, we propose HIVE (History-Informed and online-VErified prompt selection), a dual-stage framework for data-efficient RL. HIVE utilizes historical reward trajectories for
    
[^225]: SpecXMaster 技术报告

    SpecXMaster Technical Report

    [https://arxiv.org/abs/2603.23101](https://arxiv.org/abs/2603.23101)

    该论文提出SpecXMaster，一个基于智能体强化学习的端到端智能框架，可直接从原始FID数据自动提取NMR光谱多重性信息并解读为化学结构，在多个公开NMR解读基准上表现卓越。

    

    智能光谱学是AI驱动的闭环科学发现中的关键要素，充当物质结构与人工智能之间的重要桥梁。然而，传统的依赖专家的光谱解读面临诸多重大挑战，包括易受人为偏见和错误的影响、依赖有限的专业知识，以及不同解读人员之间的差异。为应对这些挑战，我们提出了SpecXMaster，一个利用智能体强化学习进行核磁共振（NMR）分子光谱解读的智能框架。SpecXMaster能够直接从原始FID（自由感应衰减）数据中自动提取1H和13C光谱的多重性信息。这一端到端流水线实现了将NMR光谱完全自动化地解读为化学结构。该框架在多个公开的NMR解读基准测试中展现出卓越的性能，并被进一步……

    arXiv:2603.23101v4 Announce Type: replace  Abstract: Intelligent spectroscopy serves as a pivotal element in AI-driven closed-loop scientific discovery, functioning as the critical bridge between matter structure and artificial intelligence. However, conventional expert-dependent spectral interpretation encounters substantial hurdles, including susceptibility to human bias and error, dependence on limited specialized expertise, and variability across interpreters. To address these challenges, we propose SpecXMaster, an intelligent framework leveraging Agentic Reinforcement Learning (RL) for NMR molecular spectral interpretation. SpecXMaster enables automated extraction of multiplicity information from both 1H and 13C spectra directly from raw FID (free induction decay) data. This end-to-end pipeline enables fully automated interpretation of NMR spectra into chemical structures. It demonstrates superior performance across multiple public NMR interpretation benchmarks and has been refine
    
[^226]: MISApp：面向下一个应用预测的多跳意图感知会话图学习

    MISApp: Multi-Hop Intent-Aware Session Graph Learning for Next App Prediction

    [https://arxiv.org/abs/2603.21653](https://arxiv.org/abs/2603.21653)

    提出了一种无需用户画像的下一个应用预测框架MISApp，通过构建多跳会话图捕捉不同结构范围的转移依赖，并结合时间与空间上下文，有效应对冷启动等场景下的预测挑战。

    

    预测用户下一个将要打开的移动应用对于主动式移动服务至关重要。然而在真实场景下，准确预测仍然充满挑战：用户意图可能在短时会话内快速变化，且用户特定的历史画像往往稀疏或不可获得，尤其是在冷启动条件下。现有方法主要将应用使用建模为序列行为或局部会话转移，限制了其捕捉高阶结构依赖和动态演化会话意图的能力。为解决这一问题，我们提出MISApp，一种基于多跳会话图学习的无用户画像的下一个应用预测框架。MISApp构建多跳会话图以捕捉不同结构范围内的转移依赖，通过轻量级图传播学习会话表示，并融入时间上下文与基于相似性的空间分类来刻画会话条件。（注：原始摘要内容不完整，在“会话条件”处被截断）

    arXiv:2603.21653v2 Announce Type: replace  Abstract: Predicting the next mobile app a user will launch is essential for proactive mobile services. Yet accurate prediction remains challenging in real-world settings, where user intent can shift rapidly within short sessions and user-specific historical profiles are often sparse or unavailable, especially under cold-start conditions. Existing approaches mainly model app usage as sequential behavior or local session transitions, limiting their ability to capture higher-order structural dependencies and evolving session intent. To address this issue, we propose MISApp, a profile-free framework for next app prediction based on multi-hop session graph learning. MISApp constructs multi-hop session graphs to capture transition dependencies at different structural ranges, learns session representations through lightweight graph propagation, incorporates temporal context and similarity-based spatial categorization to characterize session conditio
    
[^227]: ICE：面向大语言模型的基于统计基础的干预一致性解释评估

    ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs

    [https://arxiv.org/abs/2603.18579](https://arxiv.org/abs/2603.18579)

    提出ICE框架，通过在多种干预算子下将模型解释与同等规模的随机基线进行统计对比，首次揭示了大语言模型的解释忠实性是依赖干预方法的量而非固定属性（切换算子导致差距高达44个百分点），并能检测出比随机表现更差的反忠实性现象。

    

    评估解释是否忠实反映模型的推理过程仍然是一个开放性问题。现有基准采用单一干预且缺乏统计检验，因此无法区分真正的忠实性与随机水平的表现。我们表明，忠实性并非固定属性，而是一个依赖算子的量，会随测量它所使用的干预方法而变化。我们提出ICE（干预一致性解释）框架，该框架在多个算子下将解释与同等规模的随机基线进行对比评估。通过对7个大语言模型在4个任务上使用删除和检索填充算子进行评估，我们发现切换算子会在18%的配置（28个注意力比较中的5个）中跨越正证据阈值，差距可达44个百分点。随机化基线在近三分之一的英文删除配置中检测到反忠实性（即解释比随机还差）……

    arXiv:2603.18579v2 Announce Type: replace-cross  Abstract: Evaluating whether explanations faithfully reflect a model's reasoning remains an open problem. Existing benchmarks use single interventions without statistical testing, making it impossible to distinguish genuine faithfulness from chance-level performance. We show that faithfulness is not a fixed property but an operator-dependent quantity that changes with the intervention method used to measure it. We introduce ICE (Intervention-Consistent Explanation), a framework that evaluates explanations against random baselines of equal size under multiple operators. Evaluating 7 LLMs across 4 tasks with deletion and retrieval infill operators, we find that switching operators crosses the positive-evidence threshold in 18% of configurations (5 of 28 attention comparisons), with gaps reaching 44 percentage points. Randomized baselines detect anti-faithfulness (explanations worse than random) in nearly one-third of English deletion confi
    
[^228]: 通过作者画像探测大语言模型中的文化信号

    Probing Cultural Signals in Large Language Models through Author Profiling

    [https://arxiv.org/abs/2603.16749](https://arxiv.org/abs/2603.16749)

    本研究通过零样本歌词作者画像任务揭示了大语言模型中的系统性文化偏见——多数模型默认偏向北美族裔而DeepSeek-1.5B更对齐亚洲族裔，并创新性地提出MAD和RD两个公平性指标来量化这些差异。

    

    大语言模型（LLM）正日益被部署于具有社会影响的应用中，这引发了人们对其所编码的文化偏见的担忧。我们通过评估大语言模型能否在零样本设置下从歌词中进行作者画像来探测这些表征，即在不进行任务特定微调的情况下推断歌手的性别和族裔。在超过10,000首歌词上对多个开源模型进行评估后发现，大语言模型取得了不俗的画像性能，但表现出系统性的文化对齐：大多数模型默认偏向北美族裔，而DeepSeek-1.5B则与亚洲族裔的对齐更强。这一发现既体现在模型的预测分布中，也体现在对其所生成理由的分析中。为了量化这些差异，我们引入了两个公平性指标——模态准确率散度（MAD）和召回率散度（RD），并表明Ministral-8B显示出最强的族裔（摘要在此处似乎被截断）

    arXiv:2603.16749v3 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly deployed in applications with societal impact, raising concerns about the cultural biases they encode. We probe these representations by evaluating whether LLMs can perform author profiling from song lyrics in a zero-shot setting, inferring singers' gender and ethnicity without task-specific fine-tuning. Across several open-source models evaluated on more than 10,000 lyrics, we find that LLMs achieve non-trivial profiling performance but demonstrate systematic cultural alignment: most models default toward North American ethnicity, while DeepSeek-1.5B aligns more strongly with Asian ethnicity. This finding emerges from both the models' prediction distributions and an analysis of their generated rationales. To quantify these disparities, we introduce two fairness metrics, Modality Accuracy Divergence (MAD) and Recall Divergence (RD), and show that Ministral-8B displays the strongest ethnic
    
[^229]: MDM-Prime-v2：二进制编码与索引洗牌实现扩散语言模型的规模化

    MDM-Prime-v2: Binary Encoding and Index Shuffling Enable Scaling of Diffusion Language Models

    [https://arxiv.org/abs/2603.16077](https://arxiv.org/abs/2603.16077)

    本文提出MDM-Prime-v2，通过分析子分词器的最优设计并引入二进制编码与索引洗牌技术，克服了MDM-Prime框架中交叉熵损失增加和词元粒度超参数选择缺乏指导的两大局限，实现了掩码扩散语言模型的规模化。

    

    掩码扩散模型（MDM）在使用部分掩码方案学习时表现出卓越的泛化能力。该方法将词元转换为子词元，并在子词元层级上对扩散过程进行建模。我们识别了MDM-Prime框架的两个局限性：首先，我们发现当子分词器与常用的字节对编码（BPE）分词器配合使用时，其函数形式会显著增加目标函数中的交叉熵损失；其次，我们缺乏指导子分词器中词元粒度超参数选择的工具。为解决这些局限性，我们分析了使MDM-Prime训练目标最小化的子分词器最优设计，并开发了MDM-Prime-v2——一个融合了二进制编码和索引洗牌的掩码扩散语言模型。我们的分析刻画了词元粒度和子词元熵如何影响训练目标与下游性能，

    arXiv:2603.16077v4 Announce Type: replace  Abstract: Masked diffusion models (MDM) exhibit superior generalization when learned using a Partial masking scheme (Prime). This approach converts tokens into sub-tokens and models the diffusion process at the sub-token level. We identify two limitations of the MDM-Prime framework. First, we find that the functional form of the subtokenizer significantly increases the cross-entropy loss in the objective when paired with commonly used Byte-Pair-Encoding (BPE) tokenizers. Second, we lack tools to guide the hyperparameter choice of the token granularity in the subtokenizer. To address these limitations, we analyze the optimal design of the subtokenizer that minimizes MDM-Prime training objective and develop MDM-Prime-v2, a masked diffusion language model which incorporates Binary Encoding and Index Shuffling. Our analysis characterizes how token granularity and sub-token entropy influence the training objective and downstream performance, provid
    
[^230]: GONE：基于邻域扩展分布塑造的结构化知识遗忘

    GONE: Structural Knowledge Unlearning via Neighborhood-Expanded Distribution Shaping

    [https://arxiv.org/abs/2603.12275](https://arxiv.org/abs/2603.12275)

    本文提出了GONE基准用于评估大型语言模型对结构化知识图谱事实的遗忘效果，能够解耦直接事实移除、推理泄漏和灾难性遗忘三种效应，并设计了邻域扩展分布塑造（NEDS）这一新型遗忘框架。

    

    在大型语言模型（LLMs）中，知识遗忘是一项紧迫且具有挑战性的任务，因为LLMs具有前所未有的记忆和消化大规模训练数据的能力，这在安全性、隐私和知识产权方面引发了更为重大的问题。然而，现有工作（包括参数编辑、微调和基于蒸馏的方法）都专注于扁平的句子级数据，而忽视了自然结构化数据中关系型、多跳和推理性的知识。针对这一空白，本文提出了图遗忘与节点擦除（Graph Oblivion and Node Erasure, GONE），一个用于评估大型语言模型中结构化知识图谱（KG）事实知识遗忘的基准。该基于KG的基准能够解耦遗忘的三种效应：直接事实移除、基于推理的知识泄漏以及灾难性遗忘。此外，本文还设计了一种新颖的遗忘框架——邻域扩展分布塑造（Neighborhood-Expanded Distribution Shaping, NEDS）。

    arXiv:2603.12275v2 Announce Type: replace  Abstract: Unlearning knowledge is a pressing and challenging task in Large Language Models (LLMs) because of their unprecedented capability to memorize and digest training data at scale, raising more significant issues regarding safety, privacy, and intellectual property. However, existing works, including parameter editing, fine-tuning, and distillation-based methods, are all focused on flat sentence-level data but overlook the relational, multi-hop, and reasoned knowledge in naturally structured data. In response to this gap, this paper introduces Graph Oblivion and Node Erasure (GONE), a benchmark for evaluating knowledge unlearning over structured knowledge graph (KG) facts in LLMs.This KG-based benchmark enables the disentanglement of three effects of unlearning: direct fact removal, reasoning-based leakage, and catastrophic forgetting. In addition, Neighborhood-Expanded Distribution Shaping (NEDS), a novel unlearning framework, is design
    
[^231]: DynaTokens：通过控制Token动力学实现持续视频-语言理解

    DynaTokens: Controlling Token Dynamics for Continual Video-Language Understanding

    [https://arxiv.org/abs/2603.06662](https://arxiv.org/abs/2603.06662)

    提出DynaTokens，一种基于Transformer的按需动态生成微调token的生成器，结合元学习启发的正则化和无梯度路由机制，有效缓解多模态大语言模型在持续视频问答中的任务干扰与遗忘问题。

    

    基于多模态大语言模型的持续视频问答仍然具有挑战性，因为顺序适配会引发任务间的干扰，而随着任务序列的增长，存储任务特定的提示词变得不切实际。我们提出了DynaTokens，一种基于Transformer的token生成器，可按需动态生成微调token，通过共享生成权重实现任务自适应的提示更新。为缓解遗忘问题，我们引入了受元学习启发的正则化器，能够前瞻性地避免任务特定的尖锐更新方向，同时将不断演化的生成器锚定在先前任务的行为上。我们在理论上将该目标与尖锐度感知优化联系起来，展示了它如何倾向于更平坦的跨任务极小值并提升记忆保持能力。DynaTokens将基于鲁棒的预训练token和视觉嵌入的无梯度路由与轻量级辅助多模态监督相结合，减少了持续适配过程中的路由器漂移。在多个（原文在此处截断）……

    arXiv:2603.06662v3 Announce Type: replace-cross  Abstract: Continual VideoQA with multimodal LLMs remains challenging because sequential adaptation induces task interference, while storing task-specific prompts becomes impractical as task sequences grow. We introduce DynaTokens, a transformer-based token generator that dynamically produces fine-tuning tokens on demand, enabling task-adaptive prompt updates through shared generation weights. To mitigate forgetting, we introduce meta-learning-inspired regularisers that look ahead to avoid task-specific sharp update directions while anchoring the evolving generator to prior-task behaviours. We theoretically connect this objective to sharpness-aware optimisation, showing how it favours flatter cross-task minima and improves retention. DynaTokens combines gradient-free routing based on robust pretrained token and visual embeddings with lightweight auxiliary multimodal supervision, reducing router drift during continual adaptation. Across st
    
[^232]: 基于希尔伯特空间嵌入的量子最大似然预测

    Quantum Maximum Likelihood Prediction via Hilbert Space Embeddings

    [https://arxiv.org/abs/2602.18364](https://arxiv.org/abs/2602.18364)

    本文通过将经验概率分布嵌入量子态并最小化量子相对熵，提出了一种量子最大似然预测方法，并为其在经典和量子大语言模型中的统一应用提供了非渐近性能保证。

    

    arXiv:2602.18364v3 公告类型: 替换-交叉 摘要：最大似然预测（MLP）是现代大型语言模型的核心任务。在此，我们首次针对由独立同分布样本构成的简化数据模型，研究该任务的量子版本。量子最大似然预测器（QMLP）通过将经验概率分布嵌入到量子态中，并在给定状态类上最小化量子相对熵来获得。我们推导了QMLP在迹范数和量子相对熵方面的非渐近性能保证，包括收敛速率和浓度不等式。我们的方法为在经典和量子大语言模型中处理MLP提供了一个统一框架。我们还考虑了量子信息投影的相关问题，并将著名的量子毕达哥拉斯定理推广到并非由自伴类生成的混合族。

    arXiv:2602.18364v3 Announce Type: replace-cross  Abstract: Maximum likelihood prediction (MLP) is a core task at the heart of modern large language models. Here, we study a quantum version of this task for a simplified data model consisting of independent and identically distributed samples, as a first step. The quantum maximum likelihood predictor (QMLP) is obtained by embedding of empirical probability distributions into quantum states and performing a minimization of quantum relative entropy over a given class of states. We derive non-asymptotic performance guarantees for QMLP in terms of convergence rates and concentration inequalities, both in trace norm and quantum relative entropy. Our approach provides a unified framework to handle MLP within both classical and quantum LLMs. We also consider the related problem of quantum information projection and generalize the well known quantum Pythagorean theorem to mixture families which are not necessarily generated by a self-adjoint cla
    
[^233]: 约束组相对策略优化

    Constrained Group Relative Policy Optimization

    [https://arxiv.org/abs/2602.05863](https://arxiv.org/abs/2602.05863)

    本文提出约束GRPO（Constrained GRPO），一种基于拉格朗日方法的GRPO扩展用于约束策略优化，并揭示了在归一化前对标量化奖励会导致共享分母耦合，使改变一个约束乘子会同时影响奖励与其他约束的相对权重这一关键失败模式。

    

    组相对策略优化（GRPO）仍然是大语言模型（LLM）和视觉语言模型（VLM）微调中占主导地位的无评论家（critic-free）方法，但其与约束策略优化（例如在安全关键领域）的兼容性尚未得到仔细研究。在本工作中，我们提出了约束GRPO（Constrained GRPO），这是一种基于拉格朗日方法的GRPO扩展，用于约束策略优化。我们证明，在归一化之前对标量化奖励这一标准做法会引入一种关键的拉格朗日特有的失败模式：GRPO的组内归一化使得约束优化对多组件学习信号的聚合方式高度敏感。我们进一步证明，在归一化之前对标量化奖励会引入共享分母耦合，使得改变某一个乘子不仅会改变其对应约束的受重视程度，还会改变奖励与其他约束之间的相对权重。我们通过一个简单但关键的修改来解决这一问题。

    arXiv:2602.05863v3 Announce Type: replace-cross  Abstract: Group Relative Policy Optimization (GRPO) remains the dominant critic-free approach for fine-tuning LLMs and VLMs, but its compatibility with constrained policy optimization (e.g. for safety-critical domains) has not been carefully examined. In this work, we introduce Constrained GRPO, a Lagrangian-based extension of GRPO for constrained policy optimization. We show that the standard practice of scalarizing rewards before normalization introduces a critical Lagrangian-specific failure mode: GRPO's within-group normalization makes constrained optimization highly sensitive to how multi-component learning signals are aggregated. We show that scalarizing rewards before normalization introduces shared-denominator coupling, so that changing one multiplier alters not only the emphasis on its corresponding constraint, but also the relative weighting of the reward and other constraints. We address this with a simple but crucial modifica
    
[^234]: Shiva-DiT：基于残差的可微分Top-k选择，实现高效扩散Transformer

    Shiva-DiT: Residual-Based Differentiable Top-$k$ Selection for Efficient Diffusion Transformers

    [https://arxiv.org/abs/2602.05605](https://arxiv.org/abs/2602.05605)

    Shiva-DiT提出了一种基于残差的可微分Top-k token选择方法，借助残差感知直通估计器同时学习token分数与预算k，并结合上下文感知路由器和自适应比率策略，在保持生成质量的同时显著降低扩散Transformer的FLOPs与延迟（在SD3-Medium上实现1.54倍加速）。

    

    扩散Transformer（DiT）在高分辨率下计算成本高昂，因为自注意力的计算量随token序列长度呈二次方增长。现有的剪枝方法无法同时提供端到端可学习性、低训练开销以及确定性的token数量以实现可预测的按token计算。我们提出了Shiva-DiT，其基于基于残差的可微分Top-k选择（Residual-Based Differentiable Top-k Selection）。它的前向传播执行硬top-k选择，同时残差感知的直通估计器将梯度同时传播到token分数和预算k，而无需评估第二条骨干网络路径。上下文感知路由器（Context-Aware Router）和自适应比率策略（Adaptive Ratio Policy）在目标平均预算下学习与层和时间步相关的token保留方案。在SD3-Medium、Flux.1-dev和PixArt-Σ上的实验表明，该方法在FLOPs和实测延迟方面均实现了持续降低。在SD3-Medium上，Shiva-DiT提供了四个保真度-延迟工作点，并达到了1.54倍的墙钟时间加速。（注：原文摘要在此处被截断）

    arXiv:2602.05605v2 Announce Type: replace-cross  Abstract: Diffusion Transformers (DiTs) are costly at high resolution because self-attention scales quadratically with token sequence length. Existing pruning methods do not jointly provide end-to-end learnability, low training overhead, and deterministic token counts for predictable token-dependent computation. We propose Shiva-DiT, based on Residual-Based Differentiable Top-k Selection. Its forward pass executes hard top-k selection, while a residual-aware straight-through estimator propagates gradients to both token scores and the budget k without evaluating a second backbone path. A Context-Aware Router and Adaptive Ratio Policy learn layer- and timestep-dependent retention schedules under a target average budget. Experiments on SD3-Medium, Flux.1-dev, and PixArt-{\Sigma} show consistent reductions in FLOPs and measured latency. On SD3-Medium, Shiva-DiT provides four fidelity-latency operating points and reaches a 1.54x wall-clock sp
    
[^235]: 模块化专家合并用于生物医学检索

    Modular Expert Merging for Biomedical Retrieval

    [https://arxiv.org/abs/2602.04731](https://arxiv.org/abs/2602.04731)

    本文提出模块化专家合并方法，通过合成难负样本和LoRA微调领域专家并合并，在生物医学检索上优于大规模混合训练，兼顾通用性能。

    

    arXiv:2602.04731v2 公告类型：替换 摘要：将通用大型语言模型适配为领域专用的密集检索器通常需要在混合领域数据上进行大规模训练。我们表明，合并独立训练的领域专用专家在四个仅解码器LLM家族（0.6B-7B）、四种合并方法和来自MTEB的十二项医学及通用检索任务中持续优于这种方法，这表明参数空间组合捕捉了互补的领域优势，而大规模混合领域训练则将其平均化。为了进一步最大化专家质量，我们引入了Synthesize-Train-Merge（STM），这是一个模块化框架，它使用顶级LLM合成难负样本，并通过LoRA微调领域专用专家后再进行合并，无需持续预训练。合成的难负样本对较小模型带来最大收益，STM在生物医学检索任务上实现了强劲性能，同时保持了具有竞争力的通用领域结果。

    arXiv:2602.04731v2 Announce Type: replace  Abstract: Adapting general-purpose LLMs into domain-specialized dense retrievers typically requires large-scale training on mixed-domain data. We show that merging independently trained domain-specialized experts consistently exceeds this approach across four decoder-only LLM families (0.6B-7B), four merging methods, and twelve medical and general retrieval tasks from MTEB, suggesting that parameter-space composition captures complementary domain strengths that large-scale mixed-domain training averages out. To further maximize expert quality, we introduce Synthesize-Train-Merge (STM), a modular framework that synthesizes hard negatives with a top-tier LLM and fine-tunes domain-specialized experts via LoRA before merging them, without continual pre-training. Synthesized hard negatives yield the largest gains for smaller models, and STM achieves strong performance on biomedical retrieval tasks while maintaining competitive general-domain result
    
[^236]: 基于Cantelli不等式的约束策略优化

    Cantelli Constrained Policy Optimization

    [https://arxiv.org/abs/2601.22993](https://arxiv.org/abs/2601.22993)

    本文提出风险厌恶方法Canary，利用Cantelli不等式基于成本回报的前两阶矩得到可处理的风险价值约束上界，并扩展CPO信赖域框架提供最坏情况保证，是所有测试环境中唯一能可靠满足风险价值约束的方法。

    

    我们提出了Canary，这是一种风险厌恶型方法，旨在优化带有风险价值约束的强化学习问题。我们利用Cantelli不等式，基于成本回报的一阶矩和二阶矩，得到了一个可处理、保守且平滑的风险价值约束上界。由此产生的约束估计器在密集成本机制下，即使违反阈值设置得很严格也能保持稳定。在约束策略优化（CPO）方法的信赖域框架基础上进行扩展，我们进一步为训练过程中的策略改进和约束违反提供了最坏情况界。实证结果表明，在训练过程中，Canary是所有测试环境中唯一能够可靠满足风险价值约束的方法。

    arXiv:2601.22993v5 Announce Type: replace  Abstract: We introduce Canary, a risk-averse method designed to optimize Value-at-Risk (VaR) constrained reinforcement learning (RL) problems. We employ Cantelli's inequality to obtain a tractable, conservative and smooth bound on the VaR constraint based on the first two moments of the cost return. This yields a constraint estimator that remains stable with tight violation thresholds in dense cost regimes. Extending the trust-region framework of the Constrained Policy Optimization (CPO) method, we further provide worst-case bounds for both policy improvement and constraint violation during the training process. Empirically during training, Canary is the only method that reliably satisfies the VaR constraint in every environment tested.
    
[^237]: 利用大语言模型迈向解决吉尔伯特-波拉克猜想

    Towards Solving the Gilbert-Pollak Conjecture via Large Language Models

    [https://arxiv.org/abs/2601.22365](https://arxiv.org/abs/2601.22365)

    该论文提出一种新型AI系统，通过让大语言模型生成以可执行代码实现的规则约束几何引理，为停滞三十年的Gilbert-Pollak猜想（Steiner比率猜想）获得更紧的下界，突破了0.824的已有纪录。

    

    吉尔伯特-波拉克猜想（Gilbert-Pollak猜想），也称为Steiner比率猜想，指出：对于欧几里得平面上的任意有限点集，Steiner最小树的长度至少为欧几里得最小生成树长度的√3/2 ≈ 0.866倍（该比值即Steiner比率）。20世纪80年代的一系列改进最终将下界提升至0.824，此后三十年间未再有实质性进展报道。大语言模型（LLM）的最新进展在竞赛级数学问题上展现出强大的性能，但其在解决开放性、研究级问题方面的潜力在很大程度上仍待探索。在本工作中，我们提出了一个新型AI系统，用于获得更紧的Steiner比率下界。我们并不直接提示大语言模型去解决该猜想，而是让其生成以可执行代码实现的、受规则约束的几何引理。这些引理随后被用于……

    arXiv:2601.22365v3 Announce Type: replace-cross  Abstract: The Gilbert-Pollak Conjecture \citep{gilbert1968steiner}, also known as the Steiner Ratio Conjecture, states that for any finite point set in the Euclidean plane, the Steiner minimum tree has length at least $\sqrt{3}/2 \approx 0.866$ times that of the Euclidean minimum spanning tree (the Steiner ratio). A sequence of improvements through the 1980s culminated in a lower bound of $0.824$, with no substantial progress reported over the past three decades. Recent advances in LLMs have demonstrated strong performance on contest-level mathematical problems, yet their potential for addressing open, research-level questions remains largely unexplored. In this work, we present a novel AI system for obtaining tighter lower bounds on the Steiner ratio. Rather than directly prompting LLMs to solve the conjecture, we task them with generating rule-constrained geometric lemmas implemented as executable code. These lemmas are then used to co
    
[^238]: 学习与外推尺度不变过程

    Learning and extrapolating scale-invariant processes

    [https://arxiv.org/abs/2601.14810](https://arxiv.org/abs/2601.14810)

    本文研究了通过在模型架构中融入尺度不变的对称性（几何深度学习思路），机器学习能够学习并外推具有幂律行为的无标度过程（如分数高斯场和阿贝尔沙堆模型），从而预测训练集中未出现过的罕见大事件。

    

    机器学习（ML）近来深刻地改变了一些领域，例如语言和视觉领域，我们可以预期它同样适用于复杂系统的分析。本文旨在探讨这样一个问题：人们能够以何种方式以及在多大程度上对无标度过程进行回归建模，即表现出幂律行为的过程，例如地震或雪崩？我们感兴趣的是预测大事件，即训练集中的罕见事件，因此这需要模型具备外推能力。为此，我们考虑了两个具有统计自相似性的范式问题。第一个是服从线性动力学的二维分数高斯场，它在构造上具有自相似性，并且可以进行精确解析分析。第二个是表现出自组织临界性的阿贝尔沙堆模型。几何深度学习这一新兴范式表明，将已知对称性纳入模型架构是取得成功的关键。

    arXiv:2601.14810v3 Announce Type: replace-cross  Abstract: Machine Learning (ML) has deeply changed some fields recently, like Language and Vision and we may expect it to be relevant also to the analysis of of complex systems. Here we want to tackle the question of how and to which extent can one regress scale-free processes, i.e. processes displaying power law behavior, like earthquakes or avalanches? We are interested in predicting the large ones, i.e. rare events in the training set which therefore require extrapolation capabilities of the model. For this we consider two paradigmatic problems that are statistically self-similar. The first one is a 2-dimensional fractional Gaussian field obeying linear dynamics, self-similar by construction and amenable to exact analysis. The second one is the Abelian sandpile model, exhibiting self-organized criticality. The emerging paradigm of Geometric Deep Learning shows that including known symmetries into the model's architecture is key to suc
    
[^239]: 超越迁移准确率：面向低资源语言的机制引导可控适配

    Beyond Transfer Accuracy: Mechanism-Guided Controlled Adaptation for Low-Resource Languages

    [https://arxiv.org/abs/2601.08146](https://arxiv.org/abs/2601.08146)

    该论文提出了一种无需反事实的回路发现方法，并据此提出回路定向监督微调（CT-SFT），仅更新任务相关的注意力头和LayerNorm，从而在低资源语言适配中既保持竞争力又最有效地避免灾难性遗忘。

    

    现有的回路发现方法依赖于具有清晰反事实的模板化任务，限制了其在多样化自然文本上的应用。我们通过标签平衡的激活均值和任务方向相关性评分，将变换器上下文分解方法（CD-T）适配到非结构化设置中，实现了无需反事实的回路发现。我们利用所发现的回路提出回路定向监督微调（CT-SFT），将参数更新限制在任务相关的注意力头和LayerNorm上。在NusaX跨语言情感迁移任务上的实验表明，CT-SFT在低资源适配方面极具竞争力。虽然非回路的稀疏更新和全量微调有时能通过容量招募达到相当的目标准确率，但CT-SFT最为一致地避免了灾难性遗忘，保留了源语言及相关任务的性能。在XNLI上的扩展实验在更难的任务上进一步支持了关于源语言保留和干预的发现。

    arXiv:2601.08146v4 Announce Type: replace-cross  Abstract: Existing circuit discovery methods rely on templated tasks with clean counterfactuals, limiting their use on diverse natural text. We adapt Contextual Decomposition for Transformers (CD-T) for unstructured settings via label-balanced activation means and task-directional relevance scoring, enabling counterfactual-free circuit discovery. We leverage the discovered circuits for Circuit-Targeted Supervised Fine-Tuning (CT-SFT), restricting parameter updates to task-relevant heads and LayerNorm. Experiments on NusaX cross-lingual sentiment transfer show that CT-SFT is highly competitive for low-resource adaptation. While non-circuit sparse updates and full fine-tuning sometimes match target accuracy through capacity recruitment, CT-SFT most consistently avoids catastrophic forgetting, preserving source-language and related-task performance. Extensions to XNLI support the source-retention and intervention findings on a harder task a
    
[^240]: 什么驱动了基于联合嵌入预测世界模型的物理规划的成功？

    What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?

    [https://arxiv.org/abs/2512.24497](https://arxiv.org/abs/2512.24497)

    本文将联合嵌入预测世界模型（JEPA-WM）类规划方法进行了系统化表征，通过对若干关键组件的全面研究，找出了在抽象表示空间中进行物理规划取得成功的关键技术选择。

    

    人工智能领域一个长期存在的挑战是开发能够解决广泛物理任务、并能泛化到新的未见任务和环境的智能体。近期一种流行的方法是从状态-动作轨迹中训练世界模型，随后将其与规划算法结合使用以解决新任务。规划通常在输入空间中进行，但最近有一类方法引入了在世界模型学习到的表示空间中进行优化的规划算法，其承诺是通过抽象掉无关细节来实现更高效的规划。在这项工作中，我们将这一类模型表征为JEPA-WMs（联合嵌入预测世界模型），并研究了使此类算法有效运作的技术选择。我们对若干关键组件进行了全面研究，目标是找出该类方法中的最优方案。我们使用模拟环境和真实世界机器人进行了实验。

    arXiv:2512.24497v4 Announce Type: replace  Abstract: A long-standing challenge in AI is to develop agents capable of solving a wide range of physical tasks and generalizing to new, unseen tasks and environments. A popular recent approach involves training a world model from state-action trajectories and subsequently use it with a planning algorithm to solve new tasks. Planning is commonly performed in the input space, but a recent family of methods has introduced planning algorithms that optimize in the learned representation space of the world model, with the promise that abstracting irrelevant details yields more efficient planning. In this work, we characterize models from this family as JEPA-WMs and investigate the technical choices that make algorithms from this class work. We propose a comprehensive study of several key components with the objective of finding the optimal approach within the family. We conducted experiments using both simulated environments and real-world robotic
    
[^241]: 论序贯假设检验中的成本感知设计

    On Cost-Aware Designs for Sequential Hypothesis Testing

    [https://arxiv.org/abs/2512.19067](https://arxiv.org/abs/2512.19067)

    本文提出了成本感知序贯假设检验框架，证明了最优期望总成本按 $\Theta(\log(1/\delta))$ 缩放，并揭示了“最大化期望信息增益与期望成本之比”这一设计原则，据此改编的经典策略具有渐近最优性。

    

    我们提出了成本感知序贯假设检验，其中主动决策者选择具有不同随机成本的感知动作，在平均误差约束 $\delta$ 下识别真实假设，同时最小化期望总成本而非样本数量。对于固定成本，我们证明了最优期望总成本的量级为 $\Theta(\log(1/\delta))$，并且可以通过基于多假设序贯概率比检验（MSPRT）的程序实现。我们证明了成本感知的设计原则是在策略诱导的动作分布下最大化期望信息增益与期望成本之比。在此原则指导下，我们将两种经典策略改编至成本感知设置中，并建立了它们的渐近最优性。随后，我们在两种揭示模型下处理随机成本：事后揭示模型（即仅在获得样本后才披露成本，此时成本-误差权衡与固定成本情形一致）……

    arXiv:2512.19067v2 Announce Type: replace-cross  Abstract: We introduce Cost-Aware (CA) Sequential Hypothesis Testing (CASHT), in which an active decision-maker selects sensing actions with differing, random costs to identify the true hypothesis under an average-error constraint $\delta$ while minimizing the expected total cost rather than the number of samples. For fixed costs, we prove that the optimal expected total cost scales as $\Theta(\log(1/\delta))$, and is achievable by Multihypothesis Sequential Probability Ratio Test-based procedures. We show that the CA design principle is to maximize the ratio of expected information gain to expected cost under the policy-induced action distribution. Guided by this principle, we adapt two classic policies to the CA setting and establish their asymptotic optimality. We then treat random costs under two revelation models: ex-post, where costs are disclosed only after a sample is obtained, and the cost-error tradeoff coincides with the fixed
    
[^242]: 面向实时混合现实应用的安全AI驱动超分辨率技术

    Secure AI-Driven Super-Resolution for Real-Time Mixed Reality Applications

    [https://arxiv.org/abs/2512.15823](https://arxiv.org/abs/2512.15823)

    提出了一种在服务器端对点云进行下采样与部分加密、在客户端利用AI超分辨率模型重建原分辨率内容的系统，可近乎线性地降低实时混合现实应用的带宽消耗与加解密延迟，同时重建误差极小。

    

    360°和6DoF（六自由度）点云视频等沉浸式格式需要高带宽和低延迟，这对实时AR/VR流媒体传输提出了挑战。本工作聚焦于降低带宽消耗和加解密延迟这两个造成整体延迟的关键因素。我们设计了一套系统，在源服务器端对点云内容进行下采样并应用部分加密。在客户端，内容被解密后使用基于机器学习的超分辨率模型进行上采样。我们的评估表明，随着下采样分辨率的降低，带宽/延迟和加解密开销呈近似线性的下降，同时超分辨率模型能够以极小的误差和适中的推理时间有效地重建原始的全分辨率点云。

    arXiv:2512.15823v3 Announce Type: replace-cross  Abstract: Immersive formats such as 360{\deg} and 6DoF point cloud videos require high bandwidth and low latency, posing challenges for real-time AR/VR streaming. This work focuses on reducing bandwidth consumption and encryption/decryption delay, two key contributors to overall latency. We design a system that downsamples point cloud content at the origin server and applies partial encryption. At the client, the content is decrypted and upscaled using an ML-based super-resolution model. Our evaluation demonstrates a nearly linear reduction in bandwidth/latency, and encryption/decryption overhead with lower downsampling resolutions, while the super-resolution model effectively reconstructs the original full-resolution point clouds with minimal error and modest inference time.
    
[^243]: 基于多元伯努利分布的多标签数据抽样方法及其在元研究中的应用

    A Multivariate Bernoulli-Based Sampling Method for Multi-Label Data with Application to Meta-Research

    [https://arxiv.org/abs/2512.08371](https://arxiv.org/abs/2512.08371)

    提出了一种基于多元伯努利分布、考虑标签间依赖性的加权抽样算法，解决了多标签数据中稀有标签难以获得足够样本的问题，并成功应用于元研究领域。

    

    数据集可能包含具有多个标签的观测值。如果标签之间不互斥，且各标签的出现频率差异很大，那么要获得一个既包含足够多稀有标签观测值以便对这些标签进行推断、又以已知方式偏离总体频率的样本，将面临很大挑战。在本文中，我们将多元伯努利分布作为多标签问题的底层分布。我们提出了一种考虑标签依赖性的新型抽样算法。该算法利用观测到的标签频率来估计多元伯努利分布的参数，并为每个标签组合计算权重。这种方法确保加权抽样能够获得目标分布的特征，同时考虑到标签之间的依赖关系。我们将该方法应用于多种数据集，其中包括从Web of Science中抽取的带有标签的研究论文样本……

    arXiv:2512.08371v5 Announce Type: replace  Abstract: Datasets may contain observations with multiple labels. If the labels are not mutually exclusive, and if the labels vary greatly in frequency, obtaining a sample that includes sufficient observations with scarcer labels to make inferences about those labels, and which deviates from the population frequencies in a known manner, creates challenges. In this paper, we consider a multivariate Bernoulli distribution as our underlying distribution of a multi-label problem. We present a novel sampling algorithm that takes label dependencies into account. It uses observed label frequencies to estimate multivariate Bernoulli distribution parameters and calculates weights for each label combination. This approach ensures the weighted sampling acquires target distribution characteristics while accounting for label dependencies. We applied this approach to a variety of datasets, including a sample of research articles from Web of Science labeled 
    
[^244]: 冻结、扩散、解码：面向抗菌肽设计的预训练Transformer嵌入的几何感知适配

    Freeze, Diffuse, Decode: Geometry-Aware Adaptation of Pretrained Transformer Embeddings for Antimicrobial Peptide Design

    [https://arxiv.org/abs/2511.23120](https://arxiv.org/abs/2511.23120)

    提出了FDD（冻结、扩散、解码）框架，通过沿冻结嵌入的内在流形传播监督信号，在保留预训练Transformer嵌入几何结构的前提下实现几何感知的任务适配，并在抗菌肽设计中生成低维、可预测、可解释的表示，支持性质预测、检索与潜空间插值。

    

    预训练Transformer提供了丰富的、通用目的的嵌入表示，可迁移至下游任务。然而，当前的迁移策略——微调和探测——要么会扭曲预训练嵌入的几何结构，要么缺乏足够的表达能力来捕捉与任务相关的信号。当监督数据稀缺时，这些问题会变得更加突出。本文提出了“冻结、扩散、解码”框架，这是一种新颖的基于扩散的框架，能够在保留预训练嵌入底层几何结构的同时，将其适配到下游任务。FDD沿着冻结嵌入的内在流形传播监督信号，实现了对嵌入空间的几何感知适配。将该框架应用于抗菌肽设计，FDD产生了低维、具有预测能力且可解释的表示，可支持性质预测、检索和潜空间插值。

    arXiv:2511.23120v2 Announce Type: replace  Abstract: Pretrained transformers provide rich, general-purpose embeddings, which are transferred to downstream tasks. However, current transfer strategies: fine-tuning and probing, either distort the pretrained geometric structure of the embeddings or lack sufficient expressivity to capture task-relevant signals. These issues become even more pronounced when supervised data are scarce. Here, we introduce Freeze, Diffuse, Decode (FDD), a novel diffusion-based framework that adapts pre-trained embeddings to downstream tasks while preserving their underlying geometric structure. FDD propagates supervised signal along the intrinsic manifold of frozen embeddings, enabling a geometry-aware adaptation of the embedding space. Applied to antimicrobial peptide design, FDD yields low-dimensional, predictive, and interpretable representations that support property prediction, retrieval, and latent-space interpolation.
    
[^245]: 通过多摄像头图像分割与侵占后时间分析提升道路安全

    Enhancing Road Safety Through Multi-Camera Image Segmentation with Post-Encroachment Time Analysis

    [https://arxiv.org/abs/2511.12018](https://arxiv.org/abs/2511.12018)

    本文提出一种基于多摄像头图像分割与侵占后时间（PET）计算的实时交通安全评估框架，利用边缘设备上的YOLOv11分割和单应性鸟瞰图变换，实现交叉口碰撞风险的细粒度动态可视化。

    

    信号交叉口交通安全分析对于减少车辆与行人碰撞至关重要，然而传统的基于事故数据的研究受限于数据稀疏和报告延迟。本文提出了一种多摄像头计算机视觉框架，通过计算侵占后时间实现实时安全评估，并在加利福尼亚州丘拉维斯塔市的H街与百老汇交叉口进行了实证演示。四台同步摄像头提供连续的视觉覆盖，图像帧在NVIDIA Jetson AGX Xavier边缘设备上使用YOLOv11分割进行车辆检测处理。检测到的车辆多边形通过单应性变换转换为统一的鸟瞰图，从而实现重叠摄像头视图之间的对齐。像素级PET算法通过测量连续车辆通过的时间间隔来跟踪每个空间位置的时间占用情况，从而实现细粒度的动态危险可视化。

    arXiv:2511.12018v2 Announce Type: replace-cross  Abstract: Traffic safety analysis at signalized intersections is essential for reducing vehicle and pedestrian collisions, yet traditional crash-based studies are limited by data sparsity and reporting latency. This paper presents a multi-camera computer vision framework for real-time safety assessment through Post-Encroachment Time (PET) computation, demonstrated at the intersection of H Street and Broadway in Chula Vista, California. Four synchronized cameras provide continuous visual coverage, with frames processed on NVIDIA Jetson AGX Xavier edge devices using YOLOv11 segmentation for vehicle detection. Detected vehicle polygons are transformed into a unified bird's-eye map via homography, enabling alignment across overlapping camera views. A pixel-level PET algorithm tracks temporal occupancy at each spatial location by measuring the time between successive vehicle passages, enabling fine-grained hazard visualization through dynamic
    
[^246]: SEBA：面向视觉强化学习的样本高效黑盒攻击

    SEBA: Sample-Efficient Black-Box Attacks on Visual Reinforcement Learning

    [https://arxiv.org/abs/2511.09681](https://arxiv.org/abs/2511.09681)

    SEBA提出了一种针对视觉强化学习的样本高效黑盒攻击框架，通过结合影子Q模型、生成对抗网络和世界模型，以极少的真实环境查询实现对基于图像的连续控制智能体的有效对抗攻击。

    

    视觉强化学习在视觉控制和机器人领域取得了显著进展，但其对对抗性扰动的脆弱性仍未得到充分探索。现有的大多数黑盒攻击集中于基于向量或离散动作的强化学习，其在基于图像的连续控制上的有效性受限于庞大的动作空间和过多的环境查询。我们提出了SEBA，一个针对视觉强化学习智能体的样本高效黑盒对抗攻击框架。SEBA集成了三个组件：一个用于估计对抗条件下累积奖励的影子Q模型、一个生成视觉上不可察觉扰动的生成对抗网络，以及一个模拟环境动态以减少真实环境查询的世界模型。通过在学习影子模型与优化生成器之间交替进行的两阶段迭代训练过程，SEBA在保持高效的同时实现了强大的攻击性能。

    arXiv:2511.09681v2 Announce Type: replace-cross  Abstract: Visual reinforcement learning has achieved remarkable progress in visual control and robotics, but its vulnerability to adversarial perturbations remains underexplored. Most existing black-box attacks focus on vector-based or discrete-action RL, and their effectiveness on image-based continuous control is limited by the large action space and excessive environment queries. We propose SEBA, a sample-efficient framework for black-box adversarial attacks on visual RL agents. SEBA integrates a shadow Q model that estimates cumulative rewards under adversarial conditions, a generative adversarial network that produces visually imperceptible perturbations, and a world model that simulates environment dynamics to reduce real-world queries. Through a two-stage iterative training procedure that alternates between learning the shadow model and refining the generator, SEBA achieves strong attack performance while maintaining efficiency. E
    
[^247]: GMTRouter：基于多轮用户交互的个性化LLM路由器

    GMTRouter: Personalized LLM Router over Multi-turn User Interactions

    [https://arxiv.org/abs/2511.08590](https://arxiv.org/abs/2511.08590)

    提出GMTRouter，将多轮用户-LLM交互建模为包含用户、LLM、查询、响应和轮次五种节点类型的异构图，以最大程度保留交互的关系结构，从而在用户偏好数据稀缺且格式不一致的情况下实现个性化的LLM路由。

    

    大语言模型（LLM）路由在平衡响应质量与计算成本方面已展现出强大能力。由于用户展现出多样化的偏好，个性化在LLM路由中受到越来越多的关注，因为即使是相同的查询也可能需要不同的模型来生成符合个人需求的响应。然而，现有方法并未实现完全个性化，且往往无法忠实地捕捉用户与LLM之间复杂的交互关系。此外，用户偏好数据通常稀缺且格式不一致，这限制了直接利用用户特定数据的方法的有效性。为了应对这些挑战，我们提出了GMTRouter，它将多轮用户-LLM交互表示为一个包含五种节点类型（用户、LLM、查询、响应和轮次）的异构图，从而最大程度地保留了交互中丰富的关系结构。通过轻量级的归纳图……

    arXiv:2511.08590v2 Announce Type: replace  Abstract: Large Language Model (LLM) routing has demonstrated strong capability in balancing response quality with computational cost. As users exhibit diverse preferences, personalization has attracted increasing attention in LLM routing, since even identical queries may require different models to generate responses tailored to individual needs. However, existing approaches are not fully personalized and often fail to faithfully capture the complex interactions between users and LLMs. Moreover, user preference data is typically scarce and inconsistent in format, which limits the effectiveness of methods that directly leverage user-specific data. To address these challenges, we propose GMTRouter, which represents multi-turn user-LLM interactions as a heterogeneous graph with five node types: user, LLM, query, response and turn, thereby maximally preserving the rich relational structure of the interaction. Through a lightweight inductive graph
    
[^248]: 廉价前向计算场景下基于控制变量的梯度预测

    Gradient Prediction with Control Variates in the Cheap-Forward Regime

    [https://arxiv.org/abs/2511.05187](https://arxiv.org/abs/2511.05187)

    该论文提出用降精度、推理风格的程序预测梯度，并通过控制变量将大量预测与少量精确梯度结合，使近似误差转化为方差而非偏差，从而在集群推理资源足够廉价时降低语言模型训练的成本。

    

    我们研究能否利用原本闲置的推理资源来降低训练中稀缺GPU的成本。我们的分析采用一种模拟计算账本，其中集群工作按稀缺GPU单次前向计算的一部分计费；所有实验均在常规GPU上运行。我们的算法通过一个降低精度、推理风格的反向模式程序来预测梯度，并通过控制变量将大量预测梯度与少量精确梯度相结合，使近似误差表现为方差而非偏差。在一个1.24亿参数的语言模型以及选定的短训练窗口上，当集群工作足够便宜时，该方法相对于所测试的基线能够降低模拟账本成本。跨越1000万至7.74亿参数规模的实验既显示了方法的有效迁移，也显示了失败案例。我们并未测试仅推理专用硬件、端到端的分布式延迟，或完整的按批次大小扫描的优化器基线。

    arXiv:2511.05187v2 Announce Type: replace  Abstract: We study whether otherwise-idle inference resources could reduce the scarce-GPU cost of training. Our analysis uses a simulated compute ledger in which fleet work is billed at a fraction of a scarce-GPU forward; all experiments run on a regular GPU. Our algorithm predicts gradients with a reduced-precision, inference-style reverse-mode program and combines many predictions with a few exact gradients through a control variate, so approximation error becomes variance rather than bias. On a 124M-parameter language model and selected short training windows, the method can lower simulated ledger cost relative to the tested baselines when fleet work is sufficiently cheap. Experiments spanning 10M-774M parameters show both transfers and failures. We do not test inference-only hardware, end-to-end distributed latency, or a full optimizer-by-batch-size baseline sweep.
    
[^249]: 面向半无限安全强化学习的交换策略优化算法

    Exchange Policy Optimization Algorithm for Semi-Infinite Safe Reinforcement Learning

    [https://arxiv.org/abs/2511.04147](https://arxiv.org/abs/2511.04147)

    提出交换策略优化（EPO）算法，首次为含无限多约束的半无限安全强化学习提供在可证明有界安全保证下实现最优策略性能的算法框架。

    

    安全强化学习（RL）旨在优化长期性能的同时遵守安全要求。然而，许多实际应用涉及无限数量的约束，从而形成半无限安全强化学习（SI-safe RL）。此类场景通常出现在安全条件必须在整个连续参数空间上强制执行的情况下，例如确保每个空间位置的资源充足分配。现有方法通常通过朴素的空间离散化或随机采样来处理这些连续约束。此类方法本质上存在残余违规问题，或仅能提供概率性的安全保证。因此，目前没有任何框架能够处理无限多个约束以提供可靠的安全证书。在本文中，我们提出交换策略优化，这是一种算法框架，可在可证明有界的安全保证下实现最优策略性能。

    arXiv:2511.04147v2 Announce Type: replace  Abstract: Safe reinforcement learning (RL) aims to optimize long-term performance while adhering to safety requirements. However, many practical applications involve an infinite number of constraints, forming semi-infinite safe RL (SI-safe RL). Such scenarios typically appear when safety conditions must be enforced across an entire continuous parameter space, such as ensuring adequate resource distribution at every spatial location. Existing approaches typically tackle these continuous constraints through naive spatial discretization or stochastic sampling. Such methods inherently suffer from residual violations or provide only probabilistic safety guarantees. Therefore, no current framework can handle infinitely many constraints to provide reliable safety certificates. In this paper, we propose exchange policy optimization (EPO), an algorithmic framework that achieves optimal policy performance with provably bounded safety guarantees. EPO ope
    
[^250]: 无需上游数据的神经变分切割后验

    Neural Variational Cut Posteriors without Upstream Data

    [https://arxiv.org/abs/2510.10268](https://arxiv.org/abs/2510.10268)

    提出NeVI-Cut方法，一种无需访问上游数据和模型、仅利用上游后验样本即可模块化且可证明准确地近似切割后验的神经变分推断方法。

    

    在许多应用中，需要将来自先前（上游）分析的参数不确定性（以样本形式提供）传播到后续（下游）分析中，且不允许反馈。这一问题被称为“切断反馈”（cutting feedback）或 cut-Bayes，而切割后验作为保持信息流约束的最优后验已被充分刻画。然而，从切割后验中采样（例如通过嵌套MCMC）计算成本高昂，而现有的用于cut-Bayes的变分推断方法需要访问上游数据和模型，这在实际中往往不可得。我们提出了一种模块化且可证明准确的cut-Bayes方法，无需访问上游数据或模型。我们利用切割后验作为在上游后验期望下最小化下游条件Kullback-Leibler散度的刻画，并用上游样本的经验平均来替代期望。我们的方法NeVI-Cut（用于切割后验的神经变分推断）……

    arXiv:2510.10268v3 Announce Type: replace-cross  Abstract: In many applications, one must propagate parameter uncertainty from an earlier (upstream) analysis, available as samples, to subsequent (downstream) analyses without feedback. This problem is called cutting feedback or cut-Bayes, and the cut-posterior, the optimal posterior preserving information-flow constraints, is well characterized. However, sampling from it (e.g., via nested MCMC) is computationally intensive, while existing variational inference methods for cut-Bayes require access to upstream data and model, often unavailable. We propose a modular and provably accurate cut-Bayes approach requiring no access to upstream data or model. We leverage the characterization of the cut-posterior as the minimizer of the expected downstream conditional Kullback-Leibler divergence over the upstream posterior, replacing the expectation with the sample average over upstream draws. Our method, NeVI-Cut (neural variational inference for
    
[^251]: 面向量子LDPC码的不确定性感知与可泛化的神经译码方法

    Toward Uncertainty-Aware and Generalizable Neural Decoding for Quantum LDPC Codes

    [https://arxiv.org/abs/2510.06257](https://arxiv.org/abs/2510.06257)

    提出了具备校准不确定性估计的量子贝叶斯图注意力译码器QuBA，并通过SAGU多阶段训练框架实现了对训练中未见过的量子LDPC码的鲁棒泛化译码。

    

    量子纠错（QEC）对于可扩展的量子计算至关重要，然而传统算法的译码错误导致精度受限（即对逻辑错误的抑制能力有限）且开销较高，而基于推理的译码器可以缓解这两个问题。迄今为止，这类机器学习（ML）译码器缺乏对实际容错至关重要的两个关键特性：可靠的不确定性量化以及对未见过的QEC码的鲁棒泛化能力。为填补这一空白，我们提出了量子贝叶斯图注意力译码器，它能够实现强大的错误模式识别能力，同时提供经过校准的不确定性估计。在QuBA的基础上，我们进一步开发了一种具有增强跨域鲁棒性的多阶段训练框架，使其能够在训练集之外进行译码，称为不确定性下的序贯聚合泛化（SAGU）。在双变量自行车（BB）码及其组合上的实验表明……

    arXiv:2510.06257v2 Announce Type: replace-cross  Abstract: Quantum error correction (QEC) is essential for scalable quantum computing, yet decoding errors via conventional algorithms result in limited accuracy (i.e., suppression of logical errors) and high overheads, both of which can be alleviated by inference-based decoders. To date, such machine-learning (ML) decoders lack two key properties crucial for practical fault tolerance: reliable uncertainty quantification and robust generalization to previously unseen QEC codes. To address this gap, we propose a Quantum Bayesian graph Attention decoder \textbf{(QuBA)} that enables expressive error-pattern recognition alongside calibrated uncertainty estimates. Building on QuBA, we further develop a multi-phase training framework with enhanced cross-domain robustness enabling decoding beyond the training set called Sequential Aggregate Generalization under Uncertainty \textbf{(SAGU)}. Experiments on bivariate bicycle (BB) codes and their co
    
[^252]: 面向跨疾病与跨人群提升预测性能的通用人口统计学预训练模型

    General Demographic Pre-trained Models for Enhancing Predictive Performance Across Diseases and Population

    [https://arxiv.org/abs/2509.07330](https://arxiv.org/abs/2509.07330)

    提出了一种基于年龄和性别的通用人口统计学预训练模型（GDP），能以即插即用的方式提升跨疾病和跨人群的预测性能。

    

    医疗健康领域的基础模型需要在异质性临床人群和疾病场景下的稳健泛化能力与部署所需的架构简洁性之间取得平衡。我们提出了一种专注于人口统计学属性的预训练模型，能够以即插即用的方式增强医学领域的特征效用。我们引入了通用人口统计学预训练模型（GDP），旨在基于年龄和性别这两种最普遍的临床特征来提取患者状态的内在表示。通过研究多种编码方法和就诊记录重排方案，对GDP的模型构成进行了优化。该模型经过预训练后，通过将学习到的表示嵌入到具有不同人口统计学特征的多种疾病和地理队列中，验证了其可迁移性。最优的模型配置随后与表现最优的表格基础模型进行了对比验证。

    arXiv:2509.07330v3 Announce Type: replace-cross  Abstract: Foundation models for healthcare require balancing robust generalization across heterogeneous clinical populations and disease settings with the architectural simplicity needed for deployment. We present a pre-trained model focused on demographic attributes that enhances feature utility across medical domains in a plug-and-play fashion. We introduce the General Demographic Pre-trained (GDP) model, designed to extract intrinsic representations of patient status based on age and sex, the two most ubiquitous clinical features. The composition of GDP was optimized by investigating various encoding methods and visit-reordering schemes. The model was pre-trained and transferability was validated by embedding the learned representations into diverse disease and geographic cohorts characterized by distinct demographic profiles. The optimal model configuration was subsequently validated against top-performing tabular foundation models (
    
[^253]: 主观图像质量评估中离群值检测的对抗性压力测试

    Adversarial Stress Testing of Outlier Detection in Subjective Image Quality Assessment

    [https://arxiv.org/abs/2509.06554](https://arxiv.org/abs/2509.06554)

    该论文提出了一个针对主观图像质量评估中离群值检测方法的对抗性压力测试框架，利用优化算法构造最坏情况下的恶意评分，以揭示现有检测方法在极端攻击下的失效行为。

    

    在主观图像和视频质量评估中，观察者对选定的刺激进行评分或比较。在计算平均意见分数（MOS）之前，应当识别出不可靠的评分并将其作为离群值处理。目前存在多种离群值检测方法，包括标准化流程，但这些方法的比较性能通常仅使用特定类型的合成离群值（如随机乱点者）来评估。此类测试不一定能揭示这些方法的最坏情况行为。为填补这一空白，我们引入并演示了一个面向离群值检测方法的通用经验性最坏情况评估框架，并为离散绝对类别评分和连续视觉模拟量表评分提供了概念验证性的对抗攻击生成器。这些攻击利用优化算法来识别能使所得MOS估计值与真实值之间偏差最大化的评分。我们将所提出的框架应用于若干……

    arXiv:2509.06554v2 Announce Type: replace-cross  Abstract: In subjective image and video quality assessment, observers rate or compare selected stimuli. Before calculating mean opinion scores (MOSs), unreliable ratings should be identified and handled as outliers. Several outlier-detection methods are available, including standardized procedures, but their comparative performance is often evaluated using only specific types of synthetic outliers such as random clickers. Such tests do not necessarily reveal the worst-case behavior of these methods. To address this gap, we introduce and demonstrate a general empirical worst-case framework for outlier-detection methods, with proof-of-concept adversarial attack generators for both discrete absolute category and continuous visual analog scale ratings. The attacks use optimization algorithms to identify ratings that maximize the discrepancy between the resulting MOS estimates and the ground truth. We apply the proposed framework to several h
    
[^254]: 通过地形与搜索行为分析实现粒子群优化中的可解释信息处理

    Explainable Information Processing in Particle Swarm Optimization through Landscape and Search Behavior Analysis

    [https://arxiv.org/abs/2509.06272](https://arxiv.org/abs/2509.06272)

    本文提出了一个针对粒子群优化的多层面可解释性框架，通过探索性地形分析（ELA）量化问题特性，并利用机器学习为未知问题预测最优的拓扑特定超参数配置，从而显著提升算法的透明度与可解释性。

    

    基于群体智能的优化算法在解决复杂问题方面已展现出卓越的成功，但由于算法组件如何影响性能缺乏透明度，其广泛应用仍受到限制。本工作从两个互补的视角——基于问题地形的可解释性和算法可解释性——为粒子群优化（PSO）提出了一个多层面的可解释性框架。从基于地形的视角出发，我们开发了一个使用探索性地形分析（ELA）的综合性表征框架，以量化问题难度、多峰性和崎岖度，提取ELA元特征、离散度度量以及信息内容统计量；同时，一种采用决策树和随机森林分类器的机器学习方法，能够为未见过的预测性难题给出最优的特定拓扑超参数配置。从算法可解释性的视角出发，我们整合……（原文摘要在此处被截断）

    arXiv:2509.06272v5 Announce Type: replace-cross  Abstract: Swarm-based optimization algorithms have demonstrated remarkable success in solving complex problems, yet their widespread adoption remains limited due to poor transparency in how algorithmic components influence performance. This work presents a multi-faceted explainability framework for Particle Swarm Optimization (PSO) through two complementary perspectives: landscape-based and algorithmic explainability. From the landscape-based perspective, we develop a comprehensive characterization framework using Exploratory Landscape Analysis (ELA) to quantify problem difficulty, multimodality, and ruggedness, extracting ELA meta-features, dispersion measures, and information content statistics, while a machine learning approach employing Decision Tree and Random Forest classifiers enables prediction of optimal topology-specific hyperparameter configurations for unseen problems. From the algorithmic explainability perspective, we integ
    
[^255]: 面向“预测后优化”方法事前评估的分类模型仿真

    Simulating Classification Models for Ex-Ante Evaluation of Predict-Then-Optimize Methods

    [https://arxiv.org/abs/2509.02191](https://arxiv.org/abs/2509.02191)

    本文提出在指定性能水平上模拟多分类预测的方法，将基于仿真的事前评估从二分类推广到具有类别型不确定参数的优化问题，构建预测误差到决策遗憾的映射，并通过基于遗憾成因的一阶近似降低计算成本。

    

    预测后优化将机器学习预测与下游优化相结合，在求解时问题参数未知的情况下为决策制定提供支持。然而，更好的预测性能并不一定带来更好的决策，因此在投入预测模型开发之前评估二者之间的关系十分有用。现有的基于仿真的方法可以实现这种事前评估，但仅限于二分类问题，并且可能需要多次求解下游优化问题。我们将该方法推广到具有类别型不确定参数的优化问题：引入了一种在指定性能水平上模拟多分类预测的方法，并利用它构建从预测误差到决策遗憾的映射。为了减少获得该映射所需的计算量，我们还提出了一种基于遗憾成因的一阶近似方法。

    arXiv:2509.02191v3 Announce Type: replace  Abstract: Predict-Then-Optimize combines machine learning predictions with downstream optimization to support decision-making when problem parameters are unknown at the time of solving. However, better predictive performance does not necessarily lead to better decisions, making it useful to assess this relationship before investing in the development of a prediction model. Existing simulation-based approaches enable such ex-ante evaluation, but are limited to binary classification and may require solving the downstream optimization problem many times. We generalize this methodology to optimization problems with categorical uncertain parameters by introducing a method for simulating multiclass predictions at prescribed performance levels and using it to construct a prediction-error-to-decision-regret mapping. To reduce the computational effort required to obtain this mapping, we also propose a first-order approximation based on the regret cause
    
[^256]: 不浪费任何数据：一种面向标签缺失的不完整多视图数据集成的半监督生成模型

    No Data Wasted: A Semi-supervised Generative Model for Incomplete Multi-view Data Integration with Missing Labels

    [https://arxiv.org/abs/2508.11180](https://arxiv.org/abs/2508.11180)

    本文提出了一种半监督生成模型，在统一框架中同时利用有标签和无标签的多视图数据，通过最大化无标签样本的似然并与信息瓶颈原理结合，同时解决了多视图学习中视图缺失和标签缺失的双重问题。

    

    多视图学习被广泛应用于现实生活中的数据集，但它经常同时面临视图缺失和标签缺失的问题。先前的概率方法通过使用专家乘积（product-of-experts）方案来聚合现有视图的表示，解决了视图缺失问题，并利用信息瓶颈（IB）原理，取得了优于确定性分类器的性能。然而，IB框架本质上是完全监督的，无法利用无标签数据。在这项工作中，我们提出了一种半监督生成模型，在统一的框架中同时利用有标签和无标签样本。我们的方法通过最大化无标签样本的似然，学习一个与有标签数据上IB共享的潜在空间。我们还在似然建模中纳入了模态特定的信息，并在共享潜在空间中执行跨视图互信息最大化，以增强对跨视图共享信息的提取。

    arXiv:2508.11180v2 Announce Type: replace-cross  Abstract: Multi-view learning is widely applied to real-life datasets, but it often suffers from both missing views and missing labels. Prior probabilistic approaches addressed the missing view problem by using a product-of-experts scheme to aggregate representations from present views and achieved superior performance over deterministic classifiers, using the information bottleneck (IB) principle. However, the IB framework is inherently fully supervised and cannot leverage unlabeled data. In this work, we propose a semi-supervised generative model that utilizes both labeled and unlabeled samples in a unified framework. Our method maximizes the likelihood of unlabeled samples to learn a latent space shared with the IB on labeled data. We also include modality-specific information in likelihood modeling and perform cross-view mutual information maximization in the shared latent space to enhance the extraction of shared information across 
    
[^257]: 通过最大化态可区分性学习编码：变分量子纠错

    Learning Encodings by Maximizing State Distinguishability: Variational Quantum Error Correction

    [https://arxiv.org/abs/2506.11552](https://arxiv.org/abs/2506.11552)

    提出变分量子纠错方法，以最大化噪声信道后量子态的可区分性作为机器学习损失函数，自动发现针对特定噪声结构优化的资源高效编码电路，并在多种场景下超越标准码。

    

    量子纠错对于保护量子信息免受退相干影响至关重要。传统的码（如表面码）需要巨大的开销，使其对于近期的早期容错设备而言并不实用。我们提出了一种新颖的目标函数，通过最大化噪声信道作用后量子态之间的可区分性，将纠错码定制以匹配特定噪声结构，从而确保高效的恢复操作。我们以可区分性损失函数将这一概念形式化，将其作为机器学习目标，用以发现针对给定噪声特性优化的资源高效编码电路。我们使用变分技术实现了这一方法，称之为变分量子纠错。我们的方法所得到的码具有良好的理论和实际性质，并在多种场景下优于标准码。我们还提供了概念验证演示。

    arXiv:2506.11552v3 Announce Type: replace-cross  Abstract: Quantum error correction is crucial for protecting quantum information against decoherence. Traditional codes like the surface code require substantial overhead, making them impractical for near-term, early fault-tolerant devices. We propose a novel objective function for tailoring error correction codes to specific noise structures by maximizing the distinguishability between quantum states after a noise channel, ensuring efficient recovery operations. We formalize this concept with the distinguishability loss function, serving as a machine learning objective to discover resource-efficient encoding circuits optimized for given noise characteristics. We implement this methodology using variational techniques, termed variational quantum error correction (VarQEC). Our approach yields codes with desirable theoretical and practical properties and outperforms standard codes in various scenarios. We also provide proof-of-concept demo
    
[^258]: DLM-One：用于单步序列生成的扩散语言模型

    DLM-One: Diffusion Language Models for One-Step Sequence Generation

    [https://arxiv.org/abs/2506.00290](https://arxiv.org/abs/2506.00290)

    DLM-One提出了一种基于分数蒸馏的框架，将扩散语言模型的生成过程压缩为单步，实现采样步数约2000倍、推理时间约500倍的加速，同时保持有竞争力的文本生成性能。

    

    本文介绍了DLM-One，这是一个基于分数蒸馏的框架，可实现连续扩散语言模型（DLM）的单步序列生成。DLM-One通过将学生模型输出的分数与前向扩散噪声空间中预训练教师DLM的分数函数对齐，从而消除了迭代精炼过程。我们证明了该框架与具体架构无关，并在多种连续流形上具有鲁棒性，包括标准的词嵌入空间和logit单纯形空间。通过对多个代表性扩散语言模型的实验，我们展示了DLM-One在采样步数上实现了高达约2000倍的加速，在墙钟时间上实现了约500倍的加速，同时在基准文本生成任务上保持了有竞争力的性能。我们进一步分析了语言领域扩散蒸馏中的失败模式，并提出了一种对抗正则化的两阶段训练方案以防止学生模型退化。

    arXiv:2506.00290v2 Announce Type: replace  Abstract: This paper introduces DLM-One, a score-distillation-based framework for one-step sequence generation with continuous diffusion language models (DLMs). DLM-One eliminates iterative refinement by aligning the scores of a student model's outputs with the score function of a pretrained teacher DLM in the forward-diffused noisy space. We demonstrate that our framework is architecture-agnostic and robust across diverse continuous manifolds, including standard token embedding spaces and logit simplex spaces. Through experiments on multiple representative DLMs, we show that DLM-One achieves up to $\sim$2000$\times$ speedup in sampling steps and $\sim$500$\times$ in wall-clock time, while maintaining competitive performance on benchmark text generation tasks. We further analyze failure modes in language-domain diffusion distillation and propose an adversarially-regularized two-stage training scheme to prevent student degeneration. Our finding
    
[^259]: 基于随机预言机的采样与非凸优化的量子加速

    Quantum Speedups for Sampling and Non-convex Optimization with Stochastic Oracles

    [https://arxiv.org/abs/2504.03626](https://arxiv.org/abs/2504.03626)

    该论文提出了一个量子加速框架，通过用方差可控的量子均值估计和梯度估计子程序替代随机梯度估计器，加速了经典的随机朗之万蒙特卡洛和哈密顿蒙特卡洛算法，且无需可逆性或精确梯度，从而实现了从非凸分布采样和非凸优化的量子加速。

    

    我们提出了在 $\mathbb{R}^d$ 上从形如 $\pi\propto e^{-f}$ 的分布中进行采样的量子加速方法。我们考虑两种随机预言机模型：一种是随机梯度预言机，其中 $f=\frac{1}{n}\sum_{i=1}^n f_i$ 且各个分量梯度 $\{\nabla f_i\}_{i \in [n]}$ 可用；另一种是随机求值预言机，其中仅有 $f$ 的带噪声值可用。我们的框架通过用方差可控的量子均值估计和梯度估计子程序替换随机梯度估计器，来加速经典的随机朗之万蒙特卡洛（LMC）和哈密顿蒙特卡洛（HMC）算法。与基于量子游走的方法不同，我们的算法不需要可逆性或精确梯度，并且保留了底层马尔可夫链的结构。在有限和设置中，量子均值估计与经典方差缩减技术相结合，提高了随机梯度查询复杂度。

    arXiv:2504.03626v2 Announce Type: replace-cross  Abstract: We present quantum speedups for sampling from distributions of the form $\pi\propto e^{-f}$ on $\mathbb{R}^d$. We consider two stochastic oracle models: a stochastic gradient oracle, where $f=\frac{1}{n}\sum_{i=1}^n f_i $ and component gradients $\{\nabla f_i\}_{i \in [n]}$ are available, and a stochastic evaluation oracle, where only noisy values of $f$ are available. Our framework accelerates classical stochastic Langevin Monte Carlo (LMC) and Hamiltonian Monte Carlo (HMC) algorithms by replacing stochastic gradient estimators with variance-controlled quantum mean estimation and gradient estimation subroutines. Unlike quantum walk based approaches, our algorithms do not require reversibility or exact gradients, and they preserve the structure of the underlying Markov chain. In the finite-sum setting, quantum mean estimation combined with classical variance-reduction techniques improves the stochastic gradient-query complexity
    
[^260]: 无需初始稳定性的线性二次调节器的样本复杂度

    Sample Complexity of Linear Quadratic Regulator Without Initial Stability

    [https://arxiv.org/abs/2502.14210](https://arxiv.org/abs/2502.14210)

    该论文提出了一种受REINFORCE启发的滚动时域算法，无需初始稳定策略即可解决未知动态的LQR问题，并通过黎曼距离下黎卡提算子收缩性的精细误差传播分析，实现了更优的样本复杂度和收敛保证。

    

    受REINFORCE算法启发，我们针对具有未知动态的线性二次调节器（LQR）问题提出了一种新颖的滚动时域算法。与以往方法不同，我们的算法避免了对两点梯度估计的依赖，同时保持了相同数量级的样本复杂度。此外，它消除了必须以稳定初始策略作为起点的限制性要求，从而拓宽了其适用范围。除了这些改进之外，我们还基于黎曼距离下黎卡提算子的收缩性，对误差传播进行了更精细的分析。这一改进带来了更优的样本复杂度，并确保了更强的收敛保证。

    arXiv:2502.14210v4 Announce Type: replace-cross  Abstract: Inspired by REINFORCE, we introduce a novel receding-horizon algorithm for the Linear Quadratic Regulator (LQR) problem with unknown dynamics. Unlike prior methods, our algorithm avoids reliance on two-point gradient estimates while maintaining the same order of sample complexity. Furthermore, it eliminates the restrictive requirement of starting with a stable initial policy, broadening its applicability. Beyond these improvements, we introduce a refined analysis of error propagation through the contraction of the Riccati operator under the Riemannian distance. This refinement leads to a better sample complexity and ensures improved convergence guarantees.
    
[^261]: 线性双时间尺度随机逼近的非渐近中心极限定理与误差界

    Nonasymptotic CLT and Error Bounds for Linear Two-Time-Scale Stochastic Approximation

    [https://arxiv.org/abs/2502.09884](https://arxiv.org/abs/2502.09884)

    本文首次建立了带 Polyak-Ruppert 平均的线性双时间尺度随机逼近的非渐近 Wasserstein-1 中心极限定理，并由此证明其期望误差以最优的 1/√K 速率衰减，填补了有限时间分析与渐近理论之间的空白。

    

    我们研究由鞅噪声驱动的线性双时间尺度随机逼近算法。机器学习中的最新应用促使人们需要理解有限时间误差率，但传统的随机逼近分析要么关注分布意义上的渐近收敛，要么给出远非最优的有限时间界。先前关于渐近中心极限定理（CLT）的工作表明，双时间尺度算法可能能够在期望意义下达到 $1/\sqrt{K}$ 的误差，其常数由极限高斯向量的期望范数给出；然而，目前已知最好的有限时间速率要慢得多。我们推导了首个针对由鞅差噪声驱动、采用 Polyak-Ruppert 平均的线性双时间尺度随机逼近的非渐近 Wasserstein-1 中心极限定理。作为推论，我们证明 Polyak-Ruppert 平均所达到的期望误差以 $1/\sqrt{K}$ 的速率衰减，这一结果显著……

    arXiv:2502.09884v4 Announce Type: replace-cross  Abstract: We consider linear two-time-scale stochastic approximation algorithms driven by martingale noise. Recent applications in machine learning motivate the need to understand finite-time error rates, but conventional stochastic approximation analyses focus on either asymptotic convergence in distribution or finite-time bounds that are far from optimal. Prior work on asymptotic central limit theorems (CLTs) suggests that two-time-scale algorithms may be able to achieve $1/\sqrt{K}$ error in expectation, with a constant given by the expected norm of the limiting Gaussian vector. However, the best known finite-time rates are much slower. We derive the first nonasymptotic Wasserstein-1 CLT for linear two-time-scale stochastic approximation with Polyak-Ruppert averaging driven by martingale difference noise. As a corollary, we show that the expected error achieved by Polyak-Ruppert averaging decays at rate $1/\sqrt{K}$, which significant
    
[^262]: 面向尺寸约束最小割聚类的双边界约束非线性最优传输

    Double-Bounded Nonlinear Optimal Transport for Size Constrained Min Cut Clusterin

    [https://arxiv.org/abs/2501.18143](https://arxiv.org/abs/2501.18143)

    本文首次将最小割问题转化为双边界约束非线性最优传输问题，并基于Frank-Wolfe方法提出DNF算法，证明了O(1/t)的收敛速率，在尺寸约束最小割聚类任务上取得了有竞争力的性能。

    

    最小割是一种重要的图划分方法。然而，目前求解最小割问题的方法存在速度慢、求解困难以及常常收敛到简单解的问题。为了解决这些问题，我们将最小割问题松弛为双边界约束问题，并首次将最小割问题视为双边界约束非线性最优传输问题。此外，我们基于Frank-Wolfe方法开发了一种求解双边界约束非线性最优传输的方法（简称DNF）。我们证明，对于满足Lipschitz光滑性的凸问题，DNF方法可以达到O(1/t)的收敛速率。我们将DNF应用于尺寸约束最小割聚类，并在八个基准数据集上进行了评估。DNF取得了具有竞争力的聚类性能，在若干数据集和指标上与对比基线方法持平或更优。

    arXiv:2501.18143v2 Announce Type: replace  Abstract: Min cut is an important graph partitioning method. However, current solutions to the min cut problem suffer from slow speeds, difficulty in solving, and often converge to simple solutions. To address these issues, we relax the min cut problem into a double-bounded constraint and, for the first time, treat the min cut problem as a double-bounded nonlinear optimal transport problem. Additionally, we develop a method for solving double bounded nonlinear optimal transport based on the Frank-Wolfe method (abbreviated as DNF). We prove that for convex problems satisfying Lipschitz smoothness, the DNF method can achieve a convergence rate of \(\mathcal{O}(\frac{1}{t})\). We apply DNF to size-constrained min-cut clustering and evaluate it on eight benchmark datasets. DNF achieves competitive clustering performance and matches or outperforms the compared baselines on several datasets and metrics.
    
[^263]: 结合结构磁共振成像与合成脑血容量图谱增强脑年龄估计

    Enhancing brain age estimation with structural MRI and synthesized cerebral blood volume maps

    [https://arxiv.org/abs/2412.01865](https://arxiv.org/abs/2412.01865)

    提出了一种融合结构MRI与合成脑血容量（DeepCBV）图谱的多模态脑年龄估计框架，将平均绝对误差降至3.95年，并有效捕捉了早期神经退行性相关的血管变化。

    

    BrainAGE（脑年龄差距估计）是一种有前景的基于影像的神经生物学衰老及疾病风险生物标志物，然而现有方法主要依赖T1加权结构磁共振成像，忽视了可能在组织损伤和认知衰退之前出现的功能性血管变化。DeepCBV图谱由非对比剂磁共振成像合成而来，通过捕捉与早期神经退行性病变相关的血管信息，为对比剂增强的灌注成像提供了一种可扩展的替代方案。我们开发了一个多模态BrainAGE框架，该框架结合了两个独立的三维卷积神经网络的预测结果：其中一个仅在结构MRI扫描上训练，另一个仅在DeepCBV图谱上训练。每个模型均在来自13个开源数据集的2851次扫描上进行训练和验证，并评估了其与轻度认知障碍（MCI）和阿尔茨海默病（AD）的一致性。组合模型在认知正常（CN）对照组中取得了最准确的脑年龄差距估计，平均绝对误差为3.95年，优于单独使用的模型。

    arXiv:2412.01865v5 Announce Type: replace-cross  Abstract: BrainAGE is a promising imaging-derived biomarker of neurobiological ageing and disease risk, yet current approaches rely predominantly on T1-weighted structural MRI, overlooking functional vascular changes that may precede tissue damage and cognitive decline. DeepCBV maps, synthesized from non-contrast MRI, offer a scalable alternative to contrast-enhanced perfusion imaging by capturing vascular information relevant to early neurodegeneration. We developed a multimodal BrainAGE framework that combines predictions from two separate three-dimensional convolutional neural networks: one trained only on structural MRI scans and another trained only on DeepCBV maps. Each model was trained and validated on 2851 scans from 13 open-source datasets and was evaluated for concordance with MCI and AD. The combined model achieved the most accurate brain age gap for CN controls, with a mean absolute error of 3.95 years, outperforming models 
    
[^264]: 单调异常检测

    Monotonic anomaly detection

    [https://arxiv.org/abs/2410.23158](https://arxiv.org/abs/2410.23158)

    针对只需检测高（或低）属性值异常的场景，本文提出了融合斜坡函数的非对称距离度量和改进的孤立森林路径长度算法，实验证明二者能显著提升单调属性数据集上的异常检测效果。

    

    半监督异常检测基于这样的原则：任何看起来与正常训练数据不同的记录都是潜在的异常。然而，在某些情况下，我们特别关注那些对应于高属性值（或低属性值，但并非两者兼有）的异常。针对基于距离的方法，我们提出了一种非对称距离度量，通过引入斜坡函数（ramp function）来考虑这种单调性。针对孤立森林算法，我们提出了一种改进的路径长度算法。通过在合成数据集和真实数据集上的实验，我们证明这些方法提升了在具有单调属性的数据集上的异常检测性能。

    arXiv:2410.23158v3 Announce Type: replace  Abstract: Semi-supervised anomaly detection is based on the principle that any record that looks different from normal training data is a potential anomaly. However, in some cases we are specifically interested in anomalies that correspond to high attribute values (or low, but not both). For distance-based methods, we propose an asymmetrical distance measure that takes this monotonicity into account by incorporating the ramp function. For the Isolation Forest algorithm, we propose a modified path length algorithm. Through experiments on synthetic and real-life datasets, we show that these proposals increase anomaly detection performance on datasets with monotonic attributes.
    
[^265]: 用于摊销采样的动作抽象

    Action abstractions for amortized sampling

    [https://arxiv.org/abs/2410.15184](https://arxiv.org/abs/2410.15184)

    提出了一种在策略优化过程中自动发现动作抽象的方法，通过将高奖励轨迹中常用的动作子序列“分块”为单个高级动作加入动作空间，从而缓解长轨迹下信用分配困难、探索受限及模式发现受阻的问题。

    

    随着强化学习（RL）和生成流网络所使用的策略采样出的轨迹变得越来越长，信用分配和探索变得更加具有挑战性，且较长的规划范围阻碍了模式发现和泛化能力。这一挑战在追求熵的强化学习方法（如生成流网络）中尤为突出，因为智能体必须学会从结构化分布中采样，并发现多个高奖励状态，而每个状态都需要许多步骤才能到达。为了应对这一挑战，我们提出了一种将动作抽象（即高级动作）的发现融入策略优化过程的方法。我们的方法包括迭代地提取在许多高奖励轨迹中常用的动作子序列，并将它们“分块”为单个动作，添加到动作空间中。在合成环境和真实世界环境上的实证评估中，我们的方法……

    arXiv:2410.15184v2 Announce Type: replace-cross  Abstract: As trajectories sampled by policies used by reinforcement learning (RL) and generative flow networks (GFlowNets) grow longer, credit assignment and exploration become more challenging, and the long planning horizon hinders mode discovery and generalization. The challenge is particularly pronounced in entropy-seeking RL methods, such as generative flow networks, where the agent must learn to sample from a structured distribution and discover multiple high-reward states, each of which take many steps to reach. To tackle this challenge, we propose an approach to incorporate the discovery of action abstractions, or high-level actions, into the policy optimization process. Our approach involves iteratively extracting action subsequences commonly used across many high-reward trajectories and `chunking' them into a single action that is added to the action space. In empirical evaluation on synthetic and real-world environments, our ap
    
[^266]: 用于到达-避开-停留问题的深度强化学习

    Deep Reinforcement Learning for Reach-Avoid-Stay Problems

    [https://arxiv.org/abs/2410.02898](https://arxiv.org/abs/2410.02898)

    本文提出一种两步深度强化学习框架，联合学习最大鲁棒到达-避开-停留集及其控制策略，能够处理一般动态系统并保证在所有有界扰动下安全到达并停留在目标集内。

    

    到达-避开-停留（Reach-Avoid-Stay, RAS）任务在许多应用中至关重要，这些应用要求系统在有界扰动下安全地到达目标集并持续保持在其中。现有方法要么难以计算最大鲁棒RAS集（即从其中RAS任务可实现的所有状态的集合），要么在处理一般动态系统时能力有限。为应对这些挑战，本文提出了一种两步深度强化学习框架，联合学习最大鲁棒RAS集及相应的控制策略。第一步识别目标集内的最大鲁棒控制不变集，并推导出确保系统保持在该集合内的策略。第二步以该不变集为目标计算最大鲁棒到达-避开（RA）集，并证明该RA集等价于最大鲁棒RAS集。利用这一结果，基于两步策略构建了一个切换策略。

    arXiv:2410.02898v3 Announce Type: replace-cross  Abstract: Reach-Avoid-Stay (RAS) tasks are essential in applications where systems must safely reach a target set and remain within it under all bounded disturbances. Existing approaches either struggle to compute the maximal robust RAS set, the set of all states from which the RAS task is achievable, or are limited in handling general dynamic systems. To address these challenges, this paper proposes a two-step deep reinforcement learning framework that jointly learns the maximal robust RAS set and the corresponding control policy. The first step identifies the maximal robust control-invariant set within the target set and derives a policy that ensures the system remains within it. The second step computes the maximal robust reach-avoid (RA) set using this invariant set as the target, and it is proven that this RA set is equivalent to the maximal robust RAS set. Leveraging this result, a switching policy is constructed from the two step-
    
[^267]: 以少胜多：一种张量优化驱动的集成方法

    Achieving More with Less: A Tensor-Optimization-Powered Ensemble Method

    [https://arxiv.org/abs/2408.02936](https://arxiv.org/abs/2408.02936)

    该论文提出了一种基于张量优化的集成方法，通过引入置信度张量来刻画各基分类器对不同类别的预测置信程度，从而仅用少量基学习器即可达到通常需要大量基学习器才能实现的分类性能与泛化能力。

    

    集成学习是一种利用弱学习器来产生强学习器的方法。然而，获取大量基学习器需要耗费大量的时间和计算资源。因此，研究如何仅使用少量基学习器就能达到通常需要大量基学习器才能实现的性能，是很有意义的。我们认为，要实现这一目标，关键在于在集成过程中同时提升分类性能和泛化能力。为了提高模型准确率，需要将每个弱基学习器更高效地整合起来。我们观察到，不同的基学习器在预测不同类别时表现出不同的准确率水平。为了利用这一点，我们引入了置信度张量 $\tilde{\mathbf{\Theta}}$，其中 $\tilde{\mathbf{\Theta}}_{rst}$ 表示第 $t$ 个基分类器将样本判定为类别 $r$ 而实际上该样本属于类别 $s$ 的置信程度。

    arXiv:2408.02936v3 Announce Type: replace  Abstract: Ensemble learning is a method that leverages weak learners to produce a strong learner. However, obtaining a large number of base learners requires substantial time and computational resources. Therefore, it is meaningful to study how to achieve the performance typically obtained with many base learners using only a few. We argue that to achieve this, it is essential to enhance both classification performance and generalization ability during the ensemble process. To increase model accuracy, each weak base learner needs to be more efficiently integrated. It is observed that different base learners exhibit varying levels of accuracy in predicting different classes. To capitalize on this, we introduce confidence tensors $\tilde{\mathbf{\Theta}}$, where $\tilde{\mathbf{\Theta}}_{rst}$ signifies the degree of confidence that the $t$-th base classifier assigns the sample to class $r$ while it actually belongs to class $s$. To the best of 
    
[^268]: 基于Marcus映射的双随机自适应邻居聚类

    Doubly Stochastic Adaptive Neighbors Clustering via the Marcus Mapping

    [https://arxiv.org/abs/2408.02932](https://arxiv.org/abs/2408.02932)

    该论文提出Marcus映射，将Marcus定理扩展到某些稀疏矩阵，证明其也可通过对角矩阵变换为双随机对称矩阵，并据此提出了引入秩约束的双随机自适应邻居聚类算法ANCMM。

    

    聚类是机器学习和数据科学中的一项基础任务，基于相似性图的聚类是该领域的一种重要方法。双随机对称相似性图为聚类问题和下游任务提供了众多好处，然而学习这样的图仍然是一个重大挑战。Marcus定理指出，严格正的对称矩阵可以通过对角矩阵变换为双随机对称矩阵。然而，在聚类中，学习稀疏矩阵对于计算效率至关重要。我们通过提出Marcus映射扩展了Marcus定理，该映射表明某些稀疏矩阵也可以通过对角矩阵变换为双随机对称矩阵。此外，我们在聚类问题中引入了秩约束，并提出了基于Marcus映射的双随机自适应邻居聚类算法（ANCMM）。

    arXiv:2408.02932v3 Announce Type: replace-cross  Abstract: Clustering is a fundamental task in machine learning and data science, and similarity graph-based clustering is an important approach within this domain. Doubly stochastic symmetric similarity graphs provide numerous benefits for clustering problems and downstream tasks, yet learning such graphs remains a significant challenge. Marcus theorem states that a strictly positive symmetric matrix can be transformed into a doubly stochastic symmetric matrix by diagonal matrices. However, in clustering, learning sparse matrices is crucial for computational efficiency. We extend Marcus theorem by proposing the Marcus mapping, which indicates that certain sparse matrices can also be transformed into doubly stochastic symmetric matrices via diagonal matrices. Additionally, we introduce rank constraints into the clustering problem and propose the Doubly Stochastic Adaptive Neighbors Clustering algorithm based on the Marcus Mapping (ANCMM).
    
[^269]: 提示未知：理解大语言模型中的响应不确定性

    Prompting the Unknown: Understanding Response Uncertainty in Large Language Models

    [https://arxiv.org/abs/2407.14845](https://arxiv.org/abs/2407.14845)

    该论文提出了一个提示-响应概念模型，识别出大语言模型响应不确定性的四个来源（提示规范不足、模型质量、任务变异性和语义冗余），并证明了提高提示信息性或模型质量可以降低响应不确定性。

    

    大语言模型（LLM）被广泛应用于跨领域的决策制定中。确保生成安全可靠的响应对基于LLM的应用的有效部署至关重要，尤其是在医疗保健和金融等高风险领域。这些应用通常使用精心设计的提示来引导响应生成；然而，提示与LLM生成响应的可靠性之间的关系尚未被充分理解。为填补这一空白，我们提出了一个新颖的提示-响应概念模型，通过识别响应不确定性的四个来源——提示规范不足、模型质量、任务变异性和语义冗余——来解释提示中提供的任务相关信息量（信息性）与LLM生成响应不确定性之间的关系。我们证明了随着提示信息性或模型质量的提升，响应不确定性会降低。

    arXiv:2407.14845v4 Announce Type: replace-cross  Abstract: Large language models (LLMs) are widely used in decision-making across diverse domains. Ensuring the generation of safe and reliable responses is critical for the effective deployment of LLM-based applications, particularly in high-stakes domains such as healthcare and finance. Most of these applications typically use carefully crafted prompts to guide response generation; however, the relationship between prompts and the reliability of LLM-generated responses is not yet fully understood. To address this gap, we propose a novel prompt-response concept model that explains the relationship between the amount of task-relevant information (informativeness) provided in the prompt and the LLM-generated response uncertainty by identifying four sources of response uncertainty: prompt underspecification, model quality, task variability, and semantic redundancy. We prove that response uncertainty decreases as prompt informativeness or mo
    
[^270]: 低内在维度概念学习的平滑分析

    Smoothed Analysis for Learning Concepts with Low Intrinsic Dimension

    [https://arxiv.org/abs/2407.00966](https://arxiv.org/abs/2407.00966)

    本文提出了一种平滑分析框架，通过只需与对小随机高斯扰动鲁棒的最优分类器竞争，实现了对依赖低维子空间且具有有界高斯表面积的概念（如半空间函数和低维凸集函数）在任意分布下的高效学习。

    

    在传统的监督学习模型中，学习者的目标——给定来自 $\mathbb{R}^d \times \{\pm 1\}$ 上任意联合分布的样本——是输出一个与某个概念类中最优拟合概念相比具有竞争力（误差在 $\epsilon$ 以内）的假设。为了摆脱即使对简单概念类学习而言的强困难性结果，我们引入了一个平滑分析框架，该框架只要求学习者与对小随机高斯扰动具有鲁棒性的最佳分类器相竞争。这一细微的改变使我们能够为满足以下条件的任意概念提供广泛的学习结果：(1) 依赖于低维子空间（即多指标模型），并且 (2) 具有有界的高斯表面积。这一类概念包括半空间函数和（低维）凸集函数，这些情况在非平滑设置下已知仅在相对于高度结构化的分布（如高斯分布）时才是可学习的。

    arXiv:2407.00966v3 Announce Type: replace  Abstract: In traditional models of supervised learning, the goal of a learner-- given examples from an arbitrary joint distribution on $\mathbb{R}^d \times \{\pm 1\}$-- is to output a hypothesis that is competitive (to within $\epsilon$) of the best fitting concept from some class. In order to escape strong hardness results for learning even simple concept classes, we introduce a smoothed-analysis framework that requires a learner to compete only with the best classifier that is robust to small random Gaussian perturbation.   This subtle change allows us to give a wide array of learning results for any concept that (1) depends on a low-dimensional subspace (aka multi-index model) and (2) has a bounded Gaussian surface area. This class includes functions of halfspaces and (low-dimensional) convex sets, cases that are only known to be learnable in non-smoothed settings with respect to highly structured distributions such as Gaussians.   Our defi
    
[^271]: 非可分数据与大步长下逻辑回归的梯度下降

    Gradient Descent on Logistic Regression with Non-Separable Data and Large Step Sizes

    [https://arxiv.org/abs/2406.05033](https://arxiv.org/abs/2406.05033)

    该论文研究了非可分数据上逻辑回归的大步长梯度下降动力学，揭示了从临界步长 $2/\lambda$ 开始的倍周期分岔现象，证明了一维中小于 $1/\lambda$ 的步长足以保证全局收敛，而对于 $1/\lambda$ 到 $2/\lambda$ 之间的步长则可构造出使GD收敛到稳定周期循环的数据集。

    

    我们研究了在较大恒定步长下梯度下降（GD）在逻辑回归问题上的动力学行为。对于线性可分数据，已知GD在任意大的步长下都能收敛到极小值点，但当问题不可分时，这一性质不再成立。事实上，其行为可能复杂得多——一系列倍周期分岔从临界步长 $2/\lambda$ 开始，其中 $\lambda$ 是解处Hessian矩阵的最大特征值。使用小于临界值的步长在初始化靠近解时可以保证收敛：但这在全局上是否足够？在一维情形中，我们证明了小于 $1/\lambda$ 的步长足以保证全局收敛。然而，对于 $1/\lambda$ 与临界步长 $2/\lambda$ 之间的所有步长，都可以构造出一个数据集，使得GD收敛到一个稳定的周期循环。在更高维度中，这实际上甚至对于步……（原文摘要在此处截断）

    arXiv:2406.05033v3 Announce Type: replace  Abstract: We study gradient descent (GD) dynamics on logistic regression problems with large, constant step sizes. For linearly-separable data, it is known that GD converges to the minimizer with arbitrarily large step sizes, a property which no longer holds when the problem is not separable. In fact, the behaviour can be much more complex -- a sequence of period-doubling bifurcations begins at the critical step size $2/\lambda$, where $\lambda$ is the largest eigenvalue of the Hessian at the solution. Using a smaller-than-critical step size guarantees convergence if initialized nearby the solution: but does this suffice globally? In one dimension, we show that a step size less than $1/\lambda$ suffices for global convergence. However, for all step sizes between $1/\lambda$ and the critical step size $2/\lambda$, one can construct a dataset such that GD converges to a stable cycle. In higher dimensions, this is actually possible even for step 
    
[^272]: GPTBIAS：一个用于评估大语言模型偏见的综合框架

    GPTBIAS: A Comprehensive Framework for Evaluating Bias in Large Language Models

    [https://arxiv.org/abs/2312.06315](https://arxiv.org/abs/2312.06315)

    本文提出了GPTBIAS框架，利用GPT-4等高性能大语言模型来评估其他模型的社会偏见，并设计了专门用于偏见评估的“偏见攻击指令”提示词，从而提升了偏见评估的可信度和可解释性。

    

    警告：本文包含可能具有冒犯性或令人不适的内容。大语言模型（LLM）在各种应用中的使用大幅增加，无论是以原始形式还是通过微调适配的形式。因此，LLM 日益流行，并被庞大的用户群体广泛采用。然而，LLM 的一个隐忧是可能生成带有社会偏见的内容。现有的评估方法存在诸多局限，其结果的可解释性程度也较为有限。在本工作中，我们提出了一个名为 GPTBIAS 的偏见评估框架，该框架利用 LLM（例如 GPT-4）的高性能来评估模型中的偏见。我们还引入了专门为评估模型偏见而设计的提示词，称为“偏见攻击指令”。为了增强偏见评估的可信度和可解释性，我们的框架不仅提供……

    arXiv:2312.06315v2 Announce Type: replace  Abstract: Warning: This paper contains content that may be offensive or upsetting. There has been a significant increase in the usage of large language models (LLMs) in various applications, both in their original form and through fine-tuned adaptations. As a result, LLMs have gained popularity and are being widely adopted by a large user community. However, one of the concerns with LLMs is the potential generation of socially biased content. The existing evaluation methods have many constraints, and their results exhibit a limited degree of interpretability. In this work, we propose a bias evaluation framework named GPTBIAS that leverages the high performance of LLMs (e.g., GPT-4 \cite{openai2023gpt4}) to assess bias in models. We also introduce prompts called Bias Attack Instructions, which are specifically designed for evaluating model bias. To enhance the credibility and interpretability of bias evaluation, our framework not only provides 
    
[^273]: 基于深度降噪自编码器的动静脉瘘无创血流检测

    Deep denoising autoencoder-based non-invasive blood flow detection for arteriovenous fistula

    [https://arxiv.org/abs/2306.06865](https://arxiv.org/abs/2306.06865)

    该论文提出一种基于深度降噪自编码器（DAE）的表征学习方法，对一层离散小波变换获得的波形进行降维和重建，实现了动静脉瘘功能障碍的无创检测，准确率达0.93。

    

    临床指南强调了定期监测和监督血液透析患者动静脉瘘（AVF）通路的重要性，以便及时发现任何功能障碍。尽管血管音图/声音分析克服了标准化AVF狭窄诊断工具的局限性，但先前的研究依赖于传统的特征提取方法，限制了其在不同环境中的适用性。相比之下，表征学习能够捕获可轻松跨不同环境迁移的基本底层因素。我们提出了一种基于深度降噪自编码器（DAE）的方法，利用表征学习，对通过一层离散小波变换获得的波形执行降维和重建任务。我们的结果表明，DAE生成的潜在表征超越了预期，准确率达到0.93。引入……

    arXiv:2306.06865v2 Announce Type: replace-cross  Abstract: Clinical guidelines underscore the importance of regularly monitoring and surveilling arteriovenous fistula (AVF) access in hemodialysis patients to promptly detect any dysfunction. Although phono-angiography/sound analysis overcomes the limitations of standardized AVF stenosis diagnosis tool, prior studies have depended on conventional feature extraction methods, restricting their applicability in diverse contexts. In contrast, representation learning captures fundamental underlying factors that can be readily transferred across different contexts. We propose an approach based on deep denoising autoencoders (DAEs) that perform dimensionality reduction and reconstruction tasks using the waveform obtained through one-level discrete wavelet transform, utilizing representation learning. Our results demonstrate that the latent representation generated by the DAE surpasses expectations with an accuracy of 0.93. The incorporation of 
    
[^274]: 鲁棒流式主成分分析

    Robust Streaming PCA

    [https://arxiv.org/abs/1902.03223](https://arxiv.org/abs/1902.03223)

    该论文提出了协方差矩阵属于时变不确定集合的鲁棒流式主成分分析框架，给出了算法收敛的基本极限，并证明噪声幂法在此扰动设定下达到速率最优。

    

    我们研究了当随机数据生成模型受到扰动时的流式主成分分析问题。现有模型假设协方差矩阵是固定的，而我们采用鲁棒的视角，即协方差矩阵属于一个随时间变化的不确定集合。在此设定下，我们给出了任何恢复主成分的算法在收敛性上的基本极限。我们分析了噪声幂法和Oja算法的收敛性（这两种算法此前都是针对平稳数据生成模型研究的），并论证了在我们的设定下噪声幂法在速率上是最优的。最后，我们通过在合成数据集和真实数据集上的数值实验验证了我们分析的有效性。

    arXiv:1902.03223v4 Announce Type: replace-cross  Abstract: We consider streaming principal component analysis when the stochastic data generating model is subject to perturbations. While existing models assume a fixed covariance, we adopt a robust perspective where the covariance matrix belongs to a temporal uncertainty set. Under this setting, we provide fundamental limits on convergence of any algorithm recovering principal components. We analyze the convergence of the noisy power method and Oja's algorithm, both studied for the stationary data generating model, and argue that the noisy power method is rate-optimal in our setting. Finally, we demonstrate the validity of our analysis through numerical experiments on synthetic and real-world datasets.
    
[^275]: 带有异常值的三元数据聚类

    Clustering Three-Way Data with Outliers. (arXiv:2310.05288v1 [stat.ML])

    [http://arxiv.org/abs/2310.05288](http://arxiv.org/abs/2310.05288)

    这项研究提出了一种用于聚类矩阵形式数据的方法，可以处理其中的异常值。

    

    矩阵变量分布是模型聚类领域的最新添加，从而可以分析具有复杂结构（如图像和时间序列）的矩阵形式数据。由于其最近的出现，关于矩阵变量数据的文献有限，对于处理这些模型中的异常值的文献更少。本文讨论了一种用于聚类矩阵变量正态数据的方法。该方法使用子集对数似然的分布，将OCLUST算法扩展到矩阵变量正态数据，并使用迭代方法检测和剪裁异常值。

    Matrix-variate distributions are a recent addition to the model-based clustering field, thereby making it possible to analyze data in matrix form with complex structure such as images and time series. Due to its recent appearance, there is limited literature on matrix-variate data, with even less on dealing with outliers in these models. An approach for clustering matrix-variate normal data with outliers is discussed. The approach, which uses the distribution of subset log-likelihoods, extends the OCLUST algorithm to matrix-variate normal data and uses an iterative approach to detect and trim outliers.
    
[^276]: 使用分数后验概率对汤普森抽样进行广义遗憾分析

    Generalized Regret Analysis of Thompson Sampling using Fractional Posteriors. (arXiv:2309.06349v1 [stat.ML])

    [http://arxiv.org/abs/2309.06349](http://arxiv.org/abs/2309.06349)

    这项研究对使用分数后验概率的汤普森抽样算法进行了广义遗憾分析，获得了依赖于实例和实例独立的频率遗憾界。这对多臂赌博问题的解决有重要意义。

    

    汤普森抽样（TS）是解决随机多臂赌博问题的最流行和最早的算法之一。我们考虑了TS的一个变种，称为α-TS，其中我们使用分数或α-后验（α∈（0,1））代替标准后验分布。为了计算α-后验，标准后验的定义中的似然函数被一个因子α搅拌。对于α-TS，我们在非常温和的先验和奖励分布条件下获得了既依赖于实例的Ο（∑_{k≠i^*}Δ_k（\frac{\log(T)}{C(α)Δ_k^2}+\frac{1}{2}））也依赖于实例独立的Ο（\sqrt{KT\log K}）频率遗憾界，其中Δ_k是第k个和最好的臂的真实均值奖励之间的差，而C(α)是已知的常数。子高斯和指数族模型都满足我们对奖励分布的一般条件。我们对先验的条件是...

    Thompson sampling (TS) is one of the most popular and earliest algorithms to solve stochastic multi-armed bandit problems. We consider a variant of TS, named $\alpha$-TS, where we use a fractional or $\alpha$-posterior ($\alpha\in(0,1)$) instead of the standard posterior distribution. To compute an $\alpha$-posterior, the likelihood in the definition of the standard posterior is tempered with a factor $\alpha$. For $\alpha$-TS we obtain both instance-dependent $\mathcal{O}\left(\sum_{k \neq i^*} \Delta_k\left(\frac{\log(T)}{C(\alpha)\Delta_k^2} + \frac{1}{2} \right)\right)$ and instance-independent $\mathcal{O}(\sqrt{KT\log K})$ frequentist regret bounds under very mild conditions on the prior and reward distributions, where $\Delta_k$ is the gap between the true mean rewards of the $k^{th}$ and the best arms, and $C(\alpha)$ is a known constant. Both the sub-Gaussian and exponential family models satisfy our general conditions on the reward distribution. Our conditions on the prior di
    

