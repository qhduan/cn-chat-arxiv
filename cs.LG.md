# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RobustRLlib: A Unified Library and Benchmark for Robust Reinforcement Learning Algorithms](https://arxiv.org/abs/2610.32971) | 本文提出RobustRLlib，一个集成了17种鲁棒在线、离线与安全RL算法的统一库与基准测试平台，并通过273种扰动配置的系统评估绘制了当前鲁棒RL算法的能力图谱，揭示了其有效范围与鲁棒性差距。 |
| [^2] | [DexPort: A Hand-Held Skill Interface for Dexterous Hand Data Collection](https://arxiv.org/abs/2610.32969) | DexPort提出了一种面向高自由度灵巧手的便携式手持数据采集接口，通过采集与部署共用相同手部硬件来避免实体差异，并借助一小组可选择的技能来协调手指运动，从而大幅降低高质量灵巧操作演示数据的采集成本。 |
| [^3] | [Flownomaly: Single-Step Conditional-Mean Recovery for Few-Shot Anomaly Detection](https://arxiv.org/abs/2610.32964) | Flownomaly将小样本工业异常检测建模为冻结DINOv2特征空间中的“正常对应物恢复”问题，通过条件流匹配学习速度场，仅需单步评估即可恢复干净正常特征的后验均值，并利用仅基于正常样本的Jacobian增益-噪声准则自动确定运行时间，无需异常验证样本。 |
| [^4] | [Measure Before You Train: Typed Decision Models vs. a Classical Baseline vs. a Generative LLM for Routing a Personal Multilingual Inbox](https://arxiv.org/abs/2610.32955) | 该论文提出自托管个人邮件代理Foyer及其测量协议，在真实多语言邮件数据上于训练前建立人工真实标注基准，并系统比较了类型化决策模型、经典机器学习基线与生成式大语言模型等九个系统的表现。 |
| [^5] | [Where Rounding Happens Matters: Rounding-Location Alignment Stabilizes LLM Reinforcement Learning](https://arxiv.org/abs/2610.32951) | 该论文发现，仅需将训练侧RoPE、SwiGLU和残差RMSNorm三个区域的舍入位置与推理侧对齐（将选定中间结果以FP32保留至BF16输出），即可在保留其他实现差异的情况下稳定BF16强化学习并防止训练崩溃。 |
| [^6] | [Knowledge Distillation across Item and Code Spaces for Parallel Generative Recommendation](https://arxiv.org/abs/2610.32943) | 该论文提出面向并行语义ID推荐器的两个互补知识蒸馏目标（如物品分布匹配IDM），将序列推荐器的物品级预测分布知识跨空间迁移到代码空间，弥补了硬标签监督缺乏备选物品分布信息的不足。 |
| [^7] | [If at First You Don't Succeed, Backtrack: Lightweight Recovery for Frozen VLAs](https://arxiv.org/abs/2610.32941) | 提出Backtrack，一种针对冻结VLA模型的轻量级失败检测与恢复机制：利用小型预测头读取模型内部表示来预测失败，并通过回溯机械臂动作让原始策略从更早状态重试，无需恢复演示、重新训练或独立模型，仅增加不到1%的参数量。 |
| [^8] | [Learning Positive Linear Combinations of ReLUs in Fully Polynomial Time](https://arxiv.org/abs/2610.32940) | 该论文提出了首个在全多项式时间和样本复杂度内、于高斯分布下以L2损失PAC学习k个任意ReLU激活正线性组合的算法，其核心创新是一种以列表可解码方式将隐藏权重向量恢复到常数误差的新型弱学习器提升过程。 |
| [^9] | [ZENO: Nonlinear filtering with Zakai Equation Neural Operators](https://arxiv.org/abs/2610.32926) | ZENO通过基于Zakai方程的物理信息损失训练神经算子，实现了对非线性随机动力系统未归一化滤波分布的高效逼近，从而完成状态推断与参数学习。 |
| [^10] | [Adaptive Frequency Threshold Byte Pair Encoding for Mixed Natural Language and Structured Symbolic Corpora](https://arxiv.org/abs/2610.32921) | 本文提出AdaBPE，一种基于自适应频率阈值的频繁模式挖掘分词方法，通过动态调整合并策略，更好地处理跨越空白的多词表达和LaTeX等结构化符号内容。 |
| [^11] | [SemantEq: Faithful Bidirectional Equivalence of Formal Statements](https://arxiv.org/abs/2610.32919) | 提出 SemantEq，一种更有原则的形式化命题等价检验方法，要求命题间只能通过一系列允许的逻辑变换步骤相互转化，从而避免因证明器过强导致逻辑等价检验被平凡通过的问题。 |
| [^12] | [A Tutorial on Physics Constrained Machine Learning for Hybrid Modeling, Optimization, and Control](https://arxiv.org/abs/2610.32914) | 本文提出了模块化物理约束机器学习框架KKT-HardNet的分步教程，该框架通过微分投影保证约束满足，并利用隐函数定理实现高效端到端训练，可在单一平台上完成混合建模、过程仿真、优化与模型预测控制。 |
| [^13] | [Recommender regret for improving regularised RL estimation with bandits and MCTS](https://arxiv.org/abs/2610.32910) | 该论文将基于老虎机的估计生成形式化为推荐过程，提出正则化感知的推荐者遗憾来评估估计质量并证明其下界，进而通过推荐者MCTS统一了基于老虎机和搜索的目标改进方法，在多种任务中实现了更快收敛与更优的下游性能。 |
| [^14] | [LoCA: Enabling Multi-Modality in Geospatial Location Encoders](https://arxiv.org/abs/2610.32894) | 该论文从理论上证明了位置编码器受Lipschitz约束其输出固有维度至多为2，并提出LoCA方法使位置编码器能够同时将单个位置映射到多个模态的描述符，从而实现多模态能力。 |
| [^15] | [When Human Corrections Travel: Safety in Federated VLA Adaptation](https://arxiv.org/abs/2610.32888) | 论文提出在联邦学习框架下共享人类纠偏衍生的VLA适配更新时，必须保留多模态纠偏的上下文与不确定性、在接收机器人闭环中预先评估更新、并兼顾隐私泄露与安全关键证据的保留，从而确保跨任务、环境与机器人本体的安全适配。 |
| [^16] | [Reinforcement Learning with Multi-Step Transition Predictions: A Bellman Theory](https://arxiv.org/abs/2610.32887) | 该论文提出块反馈协议下的贝叶斯贝尔曼方程理论，揭示多步转移预测的价值源于递归改变贝尔曼展开中最大化与期望的顺序，从而无需状态空间指数级扩展即可量化预测信息对强化学习的增益。 |
| [^17] | [Trusting the Tool: Asymmetric Deference of LLM Vulnerability Auditors to Static-Analysis Evidence](https://arxiv.org/abs/2610.32876) | 本研究首次量化了 LLM 漏洞审计员对静态分析证据的盲从性：真实证据几乎没有提升审计准确率，而工具的虚假“保证”却能使真实漏洞的检出率下降多达 19 个百分点，揭示了模型对工具证据的非对称依赖。 |
| [^18] | [Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery](https://arxiv.org/abs/2610.32866) | 本文提出了设计上即可解释的稀疏微调方法TopKLoRA，通过在每个token上仅激活k个最大的适配器潜在变量、纯粹以下游损失训练，使其潜在变量在功能上具有因果性，从而能够忠实地发现休眠代理后门等模型内部电路。 |
| [^19] | [PoKER: A Probability of Kill Estimation Rate Model for Air-to-Air Missiles Using Machine Learning on Stochastic Targets](https://arxiv.org/abs/2610.32864) | PoKER是一种利用机器学习整合目标随机行为与导弹脱靶距离、从而更准确预测超视距空战中空空导弹杀伤概率的新型概率模型，有望成为飞行员改善作战决策的重要工具。 |
| [^20] | [Sustainability of TSFM Deployment and Pretraining: From Carbon Footprint Analysis to Mitigation via Pruning](https://arxiv.org/abs/2610.32859) | 该论文通过建模不同部署场景下的累计碳成本，评估了时间序列基础模型相比需定期重训的传统模型的可持续性，并证明仅通过无梯度剪枝移除单个编码器块，即可在几乎不损失零样本预测性能的前提下减少6%-9%的计算量、碳排放和延迟。 |
| [^21] | [Regularised RL improves sim-to-real transfer](https://arxiv.org/abs/2610.32855) | 本文提出基于软mellowmax价值函数的新型正则化强化学习算法SMM-AC，该算法在真实机器人上比SAC等基线方法获得更高回报且抗扰动能力更强，证明了正则化能够有效改善sim-to-real迁移性能。 |
| [^22] | [Singular Geometry of Low-Rank Adaptation](https://arxiv.org/abs/2610.32854) | 本文借助奇异学习理论中的实对数典型阈值（RLCT）刻画了LoRA作为奇异统计模型的学习复杂度，证明该复杂度由适应层前向激活与反向传播敏感度构成的协方差决定，并在非退化情形下恢复降秩回归的经典复杂度。 |
| [^23] | [Near-Optimal List Replicability from Littlestone Dimension](https://arxiv.org/abs/2610.32853) | 本文证明了Littlestone维度为d的概念类的列表可复制数至多为2^{O(d·min{log d, v*})}，该界在指数上仅差一个对数因子即匹配最坏情形下界，并指数级改进了此前最佳结果，从而近乎最优地解决了列表可复制性问题。 |
| [^24] | [Neural LoFi with Backward Coupling: A Spectral Theory of Cross-Layer Feature Learning](https://arxiv.org/abs/2610.32849) | 该论文提出带反向耦合的神经低度滤波谱理论，从反向传播梯度推导出跨层局部耦合动力学，利用上层敏感性为早期谱步骤生成有效标签，从而在逐层学习框架中实现跨层特征交互。 |
| [^25] | [On the Limits of Sparse Linear Model with Shuffled Differential Privacy](https://arxiv.org/abs/2610.32822) | 本文证明了在非交互式乱序差分隐私协议下稀疏线性回归存在Ω(√(dk)/(nε))的固有维度误差下界，并提出了相应的算法，进而通过交互式协议克服该维度障碍。 |
| [^26] | [U-NeXt: General-purpose event detection for neonatal EEG](https://arxiv.org/abs/2610.32811) | 提出了U-NeXt——一个基于语义分割的新生儿脑电通用事件检测框架，通过统一的卷积编码器-解码器架构对低电压区间、伪迹和癫痫发作三类事件实现了优于任务特定模型的检测性能，尤其在短事件检测的敏感性上有显著提升。 |
| [^27] | [Multiplicative vs. Additive State Coupling in Learned Optimizer Schedules](https://arxiv.org/abs/2610.32801) | 本研究通过反向传播学习MAX-CUT求解器的调度，发现双线性（乘法）状态耦合架构在分布外泛化上相比加法耦合具有显著且持久的优势，从而将关键架构区别定位于求解器状态进入调度的方式，而非状态自适应性本身。 |
| [^28] | [CARES: Coverage-Aware Replay Exemplar Selection for Continual Learning](https://arxiv.org/abs/2610.32799) | CARES利用冻结预训练特征空间的类别感知、密度加权覆盖度来选择回放样本，无需任何模型优化即可构建缓冲区，比现有基于核心集的选择器快2.3至8.8倍，同时在平衡、不平衡和标签噪声数据流中保持最先进的性能。 |
| [^29] | [Generalization properties of the token perceptron: the role of embedding and vocabulary](https://arxiv.org/abs/2610.32788) | 该论文提出“词元感知机”这一最小可解线性模型，并用复制方法解析计算其泛化误差，揭示了嵌入空间的几何结构与有限词表大小如何影响大语言模型中词元预测的学习与泛化。 |
| [^30] | [Deep Learning-Based Prediction of HER2 Status from Whole-Brain MRI in Breast Cancer Brain Metastases](https://arxiv.org/abs/2610.32774) | 该研究探索利用深度学习从乳腺癌脑转移瘤的全脑T1加权MRI中直接预测HER2状态，并评估了引入专家肿瘤分割信息作为额外输入通道能否提升预测性能。 |
| [^31] | [Direction-Aware Token Credit Assignment with Reward-Model Gradients](https://arxiv.org/abs/2610.32772) | 提出方向感知词元信用分配方法 DATC，利用冻结奖励模型的梯度来度量奖励方向与 GRPO 更新方向的对齐程度，从而为每个词元分配信用权重，在两个模型家族的四种设置中均优于 GRPO。 |
| [^32] | [On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models](https://arxiv.org/abs/2610.32770) | 该论文提出了一种无需任何数据的在线策略自训练框架，大语言模型仅需粗糙的高层主题描述即可自生成问题-答案对和特权推理演示进行自我改进，在多个领域提升能力的同时保留原有领域的性能。 |
| [^33] | [Clean-label Poisoning Attack and Defense Game for Federated Learning in Vehicular Networks](https://arxiv.org/abs/2610.32765) | 该论文针对车联网联邦学习提出了一种特征空间干净标签投毒攻击与特征一致性重标记防御方法，并通过博弈论框架刻画攻防均衡策略，在不损害正常检测准确率的情况下将攻击成功率维持在低水平。 |
| [^34] | [ARENA: Active Region attENtion Aggregation for interpretable solar flare prediction](https://arxiv.org/abs/2610.32758) | ARENA是一种基于多示例学习的弱监督神经网络架构，通过注意力聚合机制从数量可变的活动区切片预测太阳耀斑，并能可解释地定位驱动预测的关键活动区，其Heidke技巧评分达0.45，与其他耀斑预测方法相当。 |
| [^35] | [Sparse Bayesian Neural Networks via Sticky Piecewise Deterministic Markov Processes](https://arxiv.org/abs/2610.32753) | 该论文提出利用粘性分段确定性马尔可夫过程结合尖峰-平板先验，在贝叶斯神经网络的采样过程中将权重精确置零，实现渐近精确的稀疏贝叶斯神经网络采样与剪枝。 |
| [^36] | [On the Identifiability of Hybrid Deep Generative Models: Meta-Learning as a Solution](https://arxiv.org/abs/2610.32748) | 本文首次从理论上探究了融合物理知识的混合深度生成模型的可辨识性问题，并提出元学习作为构建具有可辨识性保证的混合深度生成模型的新颖解决方案。 |
| [^37] | [Strongly Adaptive Online Learning with Time-Varying Movement Costs](https://arxiv.org/abs/2610.32740) | 本文提出了一种针对时变移动代价在线凸优化的强自适应算法，通过新颖地归约到此前未被探索的带移动代价的睡眠专家问题，在任意时间区间上实现了数据依赖的遗憾界。 |
| [^38] | [Internal Discrepancy for Contrastive Learning: Generalization Bounds and Downstream Behaviors](https://arxiv.org/abs/2610.32729) | 本文提出“内部差异”度量来统一刻画对比学习中的各种理论假设，证明了基于该度量的下游监督任务泛化界，并提出松弛k均值损失作为减小内部差异的近似方法。 |
| [^39] | [PlayerEmbed: Learning Behavioral Fingerprint Representations from Raw Input Sequences](https://arxiv.org/abs/2610.32727) | 该论文提出PlayerEmbed框架，利用深度度量学习和混合CNN-LSTM编码器从原始玩家输入序列中学习紧凑的行为指纹嵌入，在玩家识别任务中对已见和未见玩家分别达到0.950和0.879的平衡准确率，且增加训练玩家数量可显著提升对未见玩家的泛化能力。 |
| [^40] | [Cross-Scenario Controller Parameter Transformation from Real-World Driving Data through Flow Matching](https://arxiv.org/abs/2610.32719) | 该论文的核心创新是将ACC控制器参数适配形式化为驾驶员个性化的场景间传输问题，利用流匹配生成模型，仅凭少量源场景演示即可为陌生驾驶员预测并生成目标场景下的控制器参数，实现跨驾驶场景的个性化参数转换。 |
| [^41] | [Score-Preserving Continual Lasso under Task Shift](https://arxiv.org/abs/2610.32708) | 提出保得分持续Lasso（SP-CL），通过保留历史平方损失得分将稀疏估计与状态传递分离，避免了递归复用 ℓ₁ 估计量导致的正则化偏差跨任务累积，并在平方损失下精确恢复全历史Lasso解与稀疏率。 |
| [^42] | [Probing the limits of forward screening and inverse design of crystal materials](https://arxiv.org/abs/2610.32678) | 该研究将55个晶体数据库按结构匹配整合为包含1460万种晶体和1.318亿性能值的统一数据库，揭示晶体-性能数据仅有4.5%被填充，并开发了高精度等变图变换器CGTNet构建性能模型库来填补数据空缺，以此探究晶体材料正向筛选与逆向设计的能力极限。 |
| [^43] | [EMG Tokenization Preserves Movement Information at a Fraction of the Bitrate](https://arxiv.org/abs/2610.32674) | 该研究提出无需姿态标签的EMG标记化方法，在将数据负载降低约41倍的情况下，三维手部姿态解码精度几乎不受影响。 |
| [^44] | [Do Language Models Trust Their Own Justifications? A Study on Functional Consistency](https://arxiv.org/abs/2610.32665) | 该研究提出基于“功能自一致性”原则的实验框架，通过分离、移除或语义反转模型所标记的关键词元，检验语言模型生成的解释是否与其真实预测行为在功能上保持一致。 |
| [^45] | [DT-JEPA: Learning to Predict Future Representations for Decision-Making](https://arxiv.org/abs/2610.32657) | 提出DT-JEPA，将联合嵌入预测架构（JEPA）与决策Transformer相结合，通过在潜在空间中预测未来动态表征来替代以预期回报作为条件，从而克服回报条件描述未来的不完整性，并摆脱对奖励标注数据集的依赖。 |
| [^46] | [WL-SeqVAE: Sequence-Based Graph Generation Using Variational Autoencoders and Weisfeiler-Leman Colors](https://arxiv.org/abs/2610.32650) | 该论文提出WL-SeqVAE，将图转换为Weisfeiler-Leman颜色序列并用变分自编码器和自回归解码器建模，把图生成转化为序列生成问题，同时借助WL颜色兼容性约束保证局部结构一致性，从而避免直接参数化大规模邻接矩阵。 |
| [^47] | [The Standard Deviation in VICReg Buys a Safety Margin](https://arxiv.org/abs/2610.32630) | 本文首次通过实验验证了 VICReg 采用标准差而非方差作为防坍塌正则项的合理性——标准差铰链因其梯度在特征方差缩小时不会过早消失，其坍塌阈值比方差铰链低 35-130 倍，从而提供了关键的安全边际。 |
| [^48] | [Acquisition for Spherical-Linear Bayesian Optimization on the Cube](https://arxiv.org/abs/2610.32628) | 本文分析了单位立方体上各类贝叶斯优化采集子问题全局求解的复杂度，并提出了一种新的采集函数，其全局优化耗时远少于现有多起点拟牛顿方法，同时保持或提升了优化性能。 |
| [^49] | [Earth embeddings are decomposable records of the land surface](https://arxiv.org/abs/2610.32617) | 本研究首次分解了两种广泛使用的年度地球嵌入向量产品，证明其中包含火灾、森林砍伐和作物类型等可独立验证的生态特征，并能定位扰动发生时间及区分火灾严重程度。 |
| [^50] | [Interpreting Supervised Pairwise Graph Representations: Local Evidence, Confounding, and Provable Post-hoc Correction](https://arxiv.org/abs/2610.32611) | 该研究通过受控实验与混杂因素分析揭示，监督式成对图表示的表面性能优势可能源于序列长度等混杂因素而非高阶图拓扑本身，并提出可证明的事后修正方法来区分真实的局部证据与虚假关联。 |
| [^51] | [Trajectory Matching: Discrete Flow Matching with Structured Actions](https://arxiv.org/abs/2610.32602) | 提出轨迹匹配方法，将离散流匹配条件化于连接端点的完整动作序列，从而支持改变状态大小、修改多个组件及依赖历史的结构化动作，为用户自定义动作空间提供有原则的、无需模拟的训练目标。 |
| [^52] | [Let's Do Turbulence Right: In Space and Time!](https://arxiv.org/abs/2610.32601) | 本文利用每次运行超过28 TB的大规模时间解析直接数值模拟，发现湍流的空间与时间评估可能不一致，呼吁机器学习社区将(3+1)维湍流视为四维现象，同时充分解析其空间结构和时间演化。 |
| [^53] | [Universal neural network approximation of symmetric polyconvex energies and envelopes](https://arxiv.org/abs/2610.32598) | 本文提出对称多凸神经网络（SPCNNs），通过在深度架构中同时融合凸性与子式空间所需的单调性，首次实现了对对称多凸能量及其包络的通用逼近，为弹性力学中的本构建模提供了高效的神经网络替代方案。 |
| [^54] | [Contextual Decision Making: Predictive or Generative](https://arxiv.org/abs/2610.32584) | 该论文从统计精度与优化可解性两个角度系统比较了先预测后优化与先生成后优化两种情境决策范式，证明了平滑性差异可以抵消生成式方法的维度负担，并揭示了凸性保持与学习复杂度之间的基本权衡。 |
| [^55] | [Batched NN 4D-Var: a performant hybrid model framework for the Integrated Forecasting System](https://arxiv.org/abs/2610.32582) | 本文提出了批处理神经网络四维变分框架，通过在综合预报系统中嵌入神经网络表示模式误差，并以随机顺序处理多个同化窗口、对参数增量取平均实现在线学习，构建了高性能的机器学习与数值预报混合模式。 |
| [^56] | [From Parameter Rebasin to Functional Transfer Across Heterogeneous Transformers](https://arxiv.org/abs/2610.32581) | 提出免训练方法Ariadne，通过刻画匹配块处的激活变化并用局部Procrustes映射对齐、闭式岭回归直接合成目标更新，在异构Transformer之间实现无需移动参数的任务行为迁移。 |
| [^57] | [Shared Is Not Enough: Predictor Width and Factor Readability Under Masked Latent Prediction](https://arxiv.org/abs/2610.32562) | 该论文通过构建所有潜在因子已知的模拟器，发现I-JEPA因使用宽度仅为编码器一半的预测器而导致共享因子的可读性大幅下降（中位数得分仅为SimCLR的40%），而将预测器加宽到编码器同等宽度后可在所有训练种子上逆转这一劣势。 |
| [^58] | [Exact Global Minimax Risk near Nonidentification](https://arxiv.org/abs/2610.32546) | 该论文给出了在冗余参数非识别点附近，最大临界路径极小极大风险等于全参数空间精确渐近极小极大风险的充分条件，并构造了能一致达到该精确首项常数的估计量，且在含未知方差的logistic误差变量回归中确定出五阶精确风险常数。 |
| [^59] | [Principled Nonlinear Concept Erasure and Steering](https://arxiv.org/abs/2610.32516) | 本文提出一个基于条件自编码器的原则性非线性概念擦除框架，从理论上证明完美重建与总体水平独立性足以实现完美的非线性防护保证，并在多个数据集上取得了更优的实用性与擦除效果的平衡。 |
| [^60] | [Analysing Austrian Parliamentary Speeches by Predicting Party Affiliations](https://arxiv.org/abs/2610.32515) | 该论文利用微调BERT模型预测奥地利议会演讲的党派归属，发现联合执政党派的演讲特征趋于相似而反对党逐渐疏远，且议员只有在换党后才会改变演讲风格。 |
| [^61] | [Amortized Thompson Sampling and Efficient Information-Directed Sampling for Simulator-Based Design](https://arxiv.org/abs/2610.32514) | 该论文提出一种摊销式汤姆逊采样方法，在数据收集前预先计算汤姆逊采样分布的近似，使观测到来时能够立即获取样本，从而大幅加速多轮序贯决策，并实现高效的信息导向采样。 |
| [^62] | [Quantum-Classical Link Prediction on Dark Web Hyperlink Graphs](https://arxiv.org/abs/2610.32510) | 本研究首次在暗网Tor超链接图上系统比较了量子与经典链接预测方法，发现在严格的度控制评估下经典图自编码器表现最佳（VGAE达0.906），而四量子比特变分量子分类器与经典基线相比并无优势。 |
| [^63] | [BAARC: Uncertainty-Driven Adaptive Partial Spectral Measurement to Full Reconstruction for Material Classification](https://arxiv.org/abs/2610.32505) | 该论文提出了一个不确定性驱动的贝叶斯自适应框架BAARC，通过逐步部分光谱测量与全谱重建的联合训练，在无需材料先验知识的情况下实现拉曼光谱材料的自动分类识别。 |
| [^64] | [DRIVE: Fine-tuning via Data Contribution- and Diversity-aware Weighting with Prior Regularization](https://arxiv.org/abs/2610.32501) | DRIVE提出了一个面向单阶段微调的统一可微数据加权框架，通过Shapley贡献分解（抗遗忘与适应分量）、基于协方差的多样性项和先验正则化三个耦合信号的共同演化，在源预训练数据不可访问的情况下显式平衡知识保持与新任务适应。 |
| [^65] | [Random Jump Forests](https://arxiv.org/abs/2610.32498) | 本文提出随机跳跃森林这一非参数方法，利用单棵树同时划分协变量空间并在节点处结合Nelson-Aalen估计，无需马尔可夫假设即可估计多状态模型中的条件累积转移率与占据概率，并建立了一致性和泛函中心极限定理等理论保证。 |
| [^66] | [Accordion: Elastic-Depth Inference via Fixed-Prefix Layer Looping](https://arxiv.org/abs/2610.32492) | 本文提出基于混合专家的模型Accordion，通过固定前缀层循环支持五种推理预算，并在预训练、监督微调和强化学习全流程中无需跨预算一致性目标即可在不同推理预算下保持强劲性能。 |
| [^67] | [A curated multi-endpoint drug-protein bioactivity dataset with construct-aware protein targets](https://arxiv.org/abs/2610.32490) | 该论文构建了Curate-DTI数据集，通过将UniProt登录号与实验序列结合来区分不同的蛋白质构建体，提供了包含超过290万条测量数据的构建体感知多终点药物-蛋白质生物活性数据集。 |
| [^68] | [State Estimation as In-context Tabular Prediction](https://arxiv.org/abs/2610.32487) | 该论文提出将电力系统状态估计转化为上下文表格预测问题，利用权重冻结的表格基础模型（仅在合成数据上预训练），在低可观测性、不良数据等加权最小二乘法失效的场景下实现了显著更优的估计精度。 |
| [^69] | [Efficient Zero-Shot Video Anomaly Detection via Signal-Aware Adaptive Fusion (SAAF)](https://arxiv.org/abs/2610.32486) | SAAF框架通过双信号自适应采样器（增量信号捕捉突发事件、漂移信号检测静态与渐进异常）以及阈值门控的级联架构，在无需训练数据的情况下实现了高效且全面的零样本视频异常检测。 |
| [^70] | [PathJev: Decision-Native Vision-Language Modelling for Histopathology](https://arxiv.org/abs/2610.32478) | PathJev-P 是一个决策原生的组织病理学视觉语言模型，通过在输入中嵌入运行时问题和候选集合并经小型决策头一次前向传播输出归一化概率分布，在无数据重叠的测试集上准确率超越同骨干生成式模型、零样本病理模型 CONCH 和通用医疗决策模型。 |
| [^71] | [Online Continual Learning in Edge-Cloud Applications](https://arxiv.org/abs/2610.32472) | 本文提出一种边缘-云方法论，将持续学习与分布外检测及概念漂移检测相集成，利用ODIN和ECDD机制识别有价值的样本并触发基于回放的模型更新，从而解决物联网应用中模型在线适应非平稳数据流时的灾难性遗忘和分布外性能退化问题。 |
| [^72] | [Sharpness-aware Orthogonalized Momentum Algorithms](https://arxiv.org/abs/2610.32465) | 提出锐度感知正交化动量算法（SOMA），将计算高效且有理论依据的锐度扰动方案与 Muon 优化器结合，以提升大语言模型训练的泛化能力并降低额外梯度计算开销。 |
| [^73] | [DualGSL: Prior-Guided Graph Structure Learning with Node-Adaptive Prediction Fusion](https://arxiv.org/abs/2610.32462) | 提出了DualGSL框架，通过节点自适应门控融合先验传播路径与任务自适应传播路径的预测结果，使图结构学习能够针对不同节点动态调整两类信息的权重。 |
| [^74] | [DIGOM: Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare](https://arxiv.org/abs/2610.32451) | 提出DIGOM框架，利用解耦视图的稀疏多路图联合建模患者内跨模态、跨患者同模态和跨患者跨模态三种交互关系，以提升多模态医疗数据的临床结果预测性能。 |
| [^75] | [Decoupling Patient Personas and Clinical Workflows: A Dual-Agent Approach to Realistic Medical Dialogue Generation](https://arxiv.org/abs/2610.32443) | 提出CoRPS-Dial双智能体框架，通过解耦患者画像与临床工作流并辅以事后质量控制，生成兼顾临床规范性与患者多样性的逼真诊前医疗对话。 |
| [^76] | [PROVE: Probabilistic Visual Grounding via Patch-Level Evidence for Vision-Language Models](https://arxiv.org/abs/2610.32440) | 提出PROVE，一种无需修改冻结CLIP骨干网络的事后证据适配器，通过将补丁特征映射为Beta分布并累积正负证据实现概率化视觉定位，同时利用证据空缺性作为信号判断所指对象是否存在，无需额外的存在分类器。 |
| [^77] | [Data Quality Audit and Bivariate Temperature-Humidity Analysis of a Long-Term LoRaWAN Network: Anomaly Detection, PCA, Clustering and Machine Learning](https://arxiv.org/abs/2610.32438) | 该研究提出一种C++17处理流水线，通过对长期LoRaWAN数据集进行系统性变量覆盖审计，发现CO2数据完全缺失、气压覆盖率极低，从而将分析重新聚焦于覆盖率高达95.76%的温度-湿度变量对，并结合异常检测、主成分分析、聚类与机器学习方法开展双变量分析。 |
| [^78] | [On the Limits of Stochastic Push-Pull over Time-Varying Digraphs](https://arxiv.org/abs/2610.32422) | 本文通过构造周期网络上的反例并建立匹配的收敛上下界，证明了随机Push-Pull算法在时变有向图上可能失去线性加速，其随机收敛率由有效节点数 $n_{\mathrm{eff}}$ 而非网络规模 $n$ 主导。 |
| [^79] | [When Does Dynamics Randomization Test Context Inference?](https://arxiv.org/abs/2610.32414) | 论文提出“预测余量”这一形式化指标来量化随机动力学环境中上下文编码器可获得的额外预测信息，发现许多环境下该余量很低，并建议将其与随机化参数一同报告，以判断显式上下文推断是否真正带来收益。 |
| [^80] | [PASTA: Prompt-Aware Annealed Diffusion Sampling](https://arxiv.org/abs/2610.32390) | PASTA提出了一种免训练的提示感知退火扩散采样框架，通过在共享潜变量轨迹上对允许对象的区域选择性施加语义引导，在文本到图像生成中严格保留指定区域，同时保持视觉连贯性。 |
| [^81] | [KnitTrans: Coordinated Trajectory Transfer for Effective and Efficient LLM Reinforcement Learning](https://arxiv.org/abs/2610.32389) | KnitTrans 通过保持约束的迷你批次准入和按需轨迹数据访问等协同设计，优化大语言模型强化学习中从采样到训练的轨迹传输，在降低架构陈旧性的同时提升学习质量与系统效率。 |
| [^82] | [Acoustic and Lexical Signals for Native Language Identification of Language Learners](https://arxiv.org/abs/2610.32382) | 该研究首次将基于WavLM的音频分类器与零样本GPT-4o转写文本分类器相结合用于第二语言学习者的母语识别，发现音频方法在低熟练度（A1）时更优，而文本方法在中高熟练度（A2–C1）时更具优势。 |
| [^83] | [Detection and Validation of Hidden Subgroup Disparities in Clinical AI](https://arxiv.org/abs/2610.32362) | 该论文提出通过算法化亚组发现在缺乏丰富元数据的情况下检测临床AI模型中隐藏的患者亚组性能差异，并对所发现亚组的性能差距和语义连贯性进行了评估与验证。 |
| [^84] | [Optimal Multiclass Online Regret with Arbitrary Label Spaces](https://arxiv.org/abs/2610.32339) | 该论文确定了具有任意（可能无限）标签空间的不可知多类在线学习的最优遗憾上界 $O(\min\{T,\sqrt{dT}\})$，解决了 Hanneke 等人（2023）提出的开放问题。 |
| [^85] | [TENET: Time-point Encoding Network for Multivariate Time Series Anomaly Detection](https://arxiv.org/abs/2610.32337) | TENET将无监督多元时间序列异常检测重新表述为当前点系统状态兼容性估计，通过以当前点为中心的时间编码和上下文自适应混合构建潜在状态表示，从而克服了观测空间差异作为异常指标不可靠的问题。 |
| [^86] | [When to Trust Memory: Retrieval-Guided Probabilistic Spatiotemporal Forecasting under Distribution Shift](https://arxiv.org/abs/2610.32329) | 提出 REMIND 框架，通过检索机制量化当前观测与历史模式的匹配程度来估计分布偏移程度，从而在非平稳环境下自适应地生成校准良好且可靠的概率时空预测。 |
| [^87] | [SKG in Practice: OOD Generalization Cases Across FourDomains](https://arxiv.org/abs/2610.32314) | SKG推理框架的工程实现在耗散机械系统、目标检测Sim-to-Real、序列建模和分子动力学四个领域的案例研究中，均显著优于神经基线方法，大幅提升了OOD（分布外）泛化性能。 |
| [^88] | [GeoLEON-CLIP: Graph-Anchored Multimodal Contrastive Learning for Geographic Region Representation](https://arxiv.org/abs/2610.32311) | 该论文提出GeoLEON-CLIP框架，以冻结的预训练LEON图嵌入作为语义锚点，通过三模态对比学习将OpenStreetMap图结构、卫星影像和文本描述对齐，从而充分利用结构化地图数据并实现最优的跨模态地理区域表征与检索性能。 |
| [^89] | [Supervised Source Separation of Electric Vehicle Interior Noise Using a Hybrid Transformer Architecture](https://arxiv.org/abs/2610.32310) | 本文将原本用于音乐分离的混合时-频谱Transformer架构适配到电动汽车领域，提出一种监督式双耳源分离方法，可将车内噪声分解为风噪、胎噪、动力总成噪声和辅助驱动噪声四种成分，从而无需昂贵的测量和传递路径分析即可支持针对性声学工程。 |
| [^90] | [DMAIC-Bench: From Self-Reflection to Self-Restraint for Agents in Multi-Stage Analytical Tasks](https://arxiv.org/abs/2610.32305) | 该论文提出 DMAIC-Bench 基准，通过基于精益六西格玛的成对反事实任务首次系统评估大语言模型智能体的“自我克制”能力（即依据证据决定继续还是停止），发现模型在 GO 决策上准确率达 92%，但在 NO-GO 决策上仅约 19%。 |
| [^91] | [CoPiT: A conditional Position-induced Transformer for aerodynamic field prediction on unstructured meshes](https://arxiv.org/abs/2610.32299) | 本文提出CoPiT模型，以PiT为几何骨干并通过多阶段FiLM调制显式注入全局条件向量，实现了非结构网格上高效准确的气动场预测。 |
| [^92] | [Complementing DINO Features with Image Structure for Part Discovery](https://arxiv.org/abs/2610.32293) | 该论文揭示了基于DINO特征的部件发现边界模糊的根源在于ViT刚性补丁网格导致跨边界补丁混合了多个部件的像素，并提出用图像结构（来自边界检测器的边界信息）补充DINO特征，从而在无需重新训练的情况下实现更精确的部件发现。 |
| [^93] | [ReLEAP: Upcycling Vision Datasets through Cross-Task Unification and Semi-Supervised Annotation](https://arxiv.org/abs/2610.32289) | ReLEAP提出了一个可扩展的Python框架，通过数据源抽象、跨任务标注重用和统一层级分类体系来整合异构视觉数据集，并结合半监督标注实现对现有数据的系统性回收与标注升级，在精准农业领域整合十五个数据集验证了其有效性。 |
| [^94] | [RootDown: Deploying and Benchmarking Models for Beet-Root Detection on Edge Devices](https://arxiv.org/abs/2610.32275) | 该论文提出了 RootDown 端到端图像检测流水线，结合基于 SAM 3 的自动标注策略与专用基准数据集，在边缘设备上实现了装载场景中部分遮挡甜菜的检测，为甜菜自动化质量检测迈出第一步。 |
| [^95] | [Empirical concentration for chains with infinite memory](https://arxiv.org/abs/2610.32265) | 本文将经验集中不等式从马尔可夫链推广到具有无限记忆的平稳链，在Wasserstein压缩条件下建立了有界差分不等式，并仅凭单条轨迹估计依赖系数，从而得到可直接使用的经验集中界。 |
| [^96] | [Anti-coherence Regularized Absolute Scaling for Out-of-Distribution Detection](https://arxiv.org/abs/2610.32255) | 提出绝对值缩放修复与抗相干性正则化，解决了SCALE方法因负激活正负抵消而在ViT等骨干网络上性能退化的问题，无需训练数据即可提升分布外检测性能。 |
| [^97] | [Do Halting Heads Really Save Compute in Looped Language Models?](https://arxiv.org/abs/2610.32246) | 本文揭示循环语言模型中的停止头因训练方式缺陷和批次生成机制而无法真正节省计算量，无停止头的模型在同等计算成本下表现几乎一样好。 |
| [^98] | [BiLoCo: Binary Low-Rank Corrections for Post-Training Quantization](https://arxiv.org/abs/2610.32243) | 提出BiLoCo方法，将低秩校正的左右因子以二值±1符号向量存储，每个秩一分量节省16倍存储，并可在CUDA核心上与FP4张量核心GEMM并行运行，有效缓解FP4训练后量化的精度损失。 |
| [^99] | [Correct but Unselectable: The Hidden Interface Tax in Multi-Candidate Reasoning](https://arxiv.org/abs/2610.32235) | 该论文发现了多候选推理中被忽视的“隐性接口税”瓶颈——正确答案虽在候选池中却因不完整、不可解析或格式不对齐而无法被选择器选中，并提出兼容性感知候选构建方法（CACC）作为候选源与固定选择器之间的接口层来解决这一问题。 |
| [^100] | [MS-TTRL: Breaking the Frequency Conflict in Multiple-Solution Test-Time Reinforcement Learning](https://arxiv.org/abs/2610.32234) | 提出MS-TTRL方法，通过结果级自验证获得与频率无关的正确性估计，并借助三态结果缓存实现验证复用，从而解决了多解测试时强化学习中生成频率在正确性估计与多样性保持之间的“频率冲突”。 |
| [^101] | [TileShare: Training-Free Acceleration of Overlapping Tiled ViT Decoding](https://arxiv.org/abs/2610.32233) | TileShare通过将重叠token的各tile计算结果混合为共享状态、同时对读取共享状态的前几层注意力保留每tile独立计算，无需训练即可将重叠平铺式ViT解码加速1.38-1.51倍（FP8下最高1.96倍）并保持质量等效。 |
| [^102] | [RoboCode: Learning Reusable Action Primitives for Robot Manipulation](https://arxiv.org/abs/2610.32225) | RoboCode通过共享的软门控基元动作接口，将机器人操作中的动作块表示为多个可复用基元的软组合，从而学习跨任务共享的短时程运动模式。 |
| [^103] | [Democratizing LLM Training with the Edge: Challenges, Systems, and the Road Ahead](https://arxiv.org/abs/2610.32215) | 本文提出以云端参数服务器为核心的统一分布式训练系统架构，系统评估了联邦学习、拆分学习和去中心化方法在异构硬件与不可靠网络环境下的可行性，旨在让缺乏大型数据中心基础设施的中小机构也能利用自身私有数据参与大语言模型的训练。 |
| [^104] | [Head Dimension Matters in Language Models for Long-Context Retrieval](https://arxiv.org/abs/2610.32213) | 本研究发现在固定模型规模下，增大注意力头维度可通过提高检索头的比例显著提升语言模型的长上下文检索性能，且即使冻结注意力投影权重该结论依然成立，而头维度对预训练损失和短上下文任务性能影响甚微。 |
| [^105] | [Do VLAs Know They Will Fail? Interpretable Success and Failure Patterns in Physical AI Systems](https://arxiv.org/abs/2610.32208) | 本论文发现OpenVLA激活的稀疏自编码器（SAE）表示蕴含结果预测信息，可在机器人任务陷入不可恢复状态前预测失败，并通过识别与成败阶段相关的SAE特征及基于VLM的解释流程，揭示反复出现的行为模式。 |
| [^106] | [Neural Disjunctive Normal Forms for Learning Exact Rank Filters](https://arxiv.org/abs/2610.32203) | 本文提出基于阈值分解的神经析取范式来学习精确秩滤波器，通过前向传播精确求值、后向传播使用平滑替代导数的直通式方法，解决了形态学算子梯度学习中导数稀疏的问题。 |
| [^107] | [SWE-Shift: Evaluating Coding Agents in Evolving Workspaces](https://arxiv.org/abs/2610.32196) | 提出 SWE-Shift 基准，用于评估编程智能体在任务执行中途遭遇代码、依赖和需求等变化时的表现，实验发现所有 17 个被测智能体系统在发生此类变化时成功率均显著下降。 |
| [^108] | [Principia: Test-Time Hypotheses to Principle Construction for Scientific Discovery](https://arxiv.org/abs/2610.32195) | 提出了基于大语言模型的Principia框架，通过从相关输入与观测结果中形成比较性因果假设、并将假设中反复出现的关系综合为归纳性原理，在测试时自动构建并持续完善科学原理，突破了科学发现中知识构建这一瓶颈。 |
| [^109] | [Dual-Adaptive Z-Score PPO: Dynamic Clipping Using Action Confidence and Normalized Advantage Magnitude](https://arxiv.org/abs/2610.32194) | 该论文提出PPO-DAZ3方法，通过动作置信度和归一化优势幅度这两个信号为每个样本动态设置PPO的裁剪阈值，同时保持标准PPO训练流程其余部分不变。 |
| [^110] | [EVOLVE: Uncertainty-Aware Multivariate Time Series Anomaly Prediction with Adaptive Prototype Memories](https://arxiv.org/abs/2610.32189) | 提出EVOLVE框架，通过自适应原型记忆实现不确定性感知的多变量时间序列异常预测，能够建模变量与上下文相关的扰动差异，并预见多种可能的未来异常演化轨迹。 |
| [^111] | [CFBrainAug: Modularity-Induced Counterfactual Brain Network Augmentation for Interpretable fMRI-based Autism Diagnosis](https://arxiv.org/abs/2610.32184) | 提出了反事实脑网络学习框架CFBrainAug，通过生成标签相反但拓扑相似、并受模块数量约束的脑网络来统一数据增强与反事实解释，在提升基于fMRI自闭症诊断性能的同时增强了临床可解释性。 |
| [^112] | [Depth-Dependent Safety Geometry in Vision-Language Models: Layer-Wise Localization and Bidirectional Refusal Control](https://arxiv.org/abs/2610.32182) | 该论文提出一个无需重训的推理时框架，通过逐层激活分析揭示视觉语言模型的安全几何随深度非均匀分布——早期视觉层可通过激活引导实现 6%–96% 的双向拒答控制，而深层虽表征可分却对操控不敏感，从而区分了表征可分性与因果可控性。 |
| [^113] | [Not All Information Is Worth Asking For: Decision-Informed Information Seeking with LLMs](https://arxiv.org/abs/2610.32181) | 提出DIVA方法，让大语言模型基于证据的决策价值扣除成本后来评估每项检查是否值得进行，从而决定该查询什么信息以及何时停止，避免为无法改变决策或不值得的信息付费。 |
| [^114] | [Searching with Sparse Histories: Cross-Task Architecture Search from Incomplete Search Logs and Zero-Cost Proxies](https://arxiv.org/abs/2610.32179) | 提出一种风险鲁棒的跨任务神经架构搜索方法，通过冠军初始化与联合成对排序器预训练，将稀疏搜索日志与零成本代理相结合，显著缓解最坏情况下的失败并大幅提升最差性能。 |
| [^115] | [Federated Nonparametric Regression with Heterogeneous Clients](https://arxiv.org/abs/2610.32178) | 该论文提出一种联邦Nadaraya-Watson非参数回归方法，客户端仅传输本地核摘要即可构建协变量密度加权的联邦估计器，并通过精确恒等式与两阶段分析刻画异构客户端从联邦预测中获益的条件及统计推断。 |
| [^116] | [$\Psi$-0.6: Adaptive Streams of Thought in a Unified Vision-Language World Model](https://arxiv.org/abs/2610.32177) | Ψ-0.6 是一个统一的视觉-语言世界模型，通过在共享的“思维流”推理过程中自适应地组合感知、语言、几何与模拟能力，在多视角空间推理上零样本超越领先的开放权重模型，并支持基于想象场景的反事实模拟。 |
| [^117] | [Curvature-Aware Clipping: Mitigating Bias in DPSGD via Approximate Second-Order Information](https://arxiv.org/abs/2610.32175) | 提出CA-DPSGD框架，利用近似二阶信息构建曲率感知的最优裁剪空间，并结合各向异性噪声机制，有效缓解DPSGD中的梯度裁剪偏差问题，性能超越现有最先进的裁剪方法。 |
| [^118] | [Solving and Learning Navier-Stokes Dynamics with the Neural Basis Method](https://arxiv.org/abs/2610.32167) | 提出神经基方法（NBM），基于“先表示后投影”原则，将纳维-斯托克斯方程的求解与算子学习转化为固定神经基上的参数到系数映射，同时具备后验误差估计保证与良好的可扩展性。 |
| [^119] | [Sparsity-driven emergence of functional modularity in multitask neural networks](https://arxiv.org/abs/2610.32153) | 稀疏性正则化无需预定义模块边界即可促使多任务神经网络自发涌现功能模块化，形成可在深层选择性保留、重组的紧凑“功能token”子网络。 |
| [^120] | [Implicit Neural Representations for Variational Problems on Graphons](https://arxiv.org/abs/2610.32147) | 该论文提出用隐式神经网络表示Graphon并通过梯度下降求解其上的变分问题，其创新在于结合多尺度正弦残差架构、经隐式微分的密度约束内嵌求解器以及对称性感知的蒙特卡洛估计器，服务于极值图论与稠密随机图大偏差理论。 |
| [^121] | [Rethinking LoRA Initialization for Robust Asymmetric Learning Rates](https://arxiv.org/abs/2610.32142) | 该论文通过对 LoRA 动力学的信号-噪声分解，揭示了标准初始化下由初始化引起的噪声项是非对称学习率训练不稳定和性能崩溃的根源，从而主张重新设计 LoRA 初始化以实现稳健的非对称学习率。 |
| [^122] | [Beyond Data Schedule: Fast LLM Pretraining on Data Mixtures by Loss-Reweighting](https://arxiv.org/abs/2610.32141) | 提出LARE方法，在保持数据暴露不变的前提下，根据运行中的损失统计自适应地重加权各领域的损失贡献，使单一检查点的知识与推理性能超越多阶段预训练的最佳结果，从而实现更快的LLM预训练。 |
| [^123] | [SyncPro: Synchronous Reasoning and Responding for Proactive Streaming Video Understanding](https://arxiv.org/abs/2610.32132) | SyncPro提出了一种同步推理与响应的机制，解决了主动式流媒体视频理解中持续推理与及时响应难以协调的问题，避免了固定推理间隔导致的无谓更新或延迟响应。 |
| [^124] | [MPQ: A Message-Passing View of Post-Training Quantization](https://arxiv.org/abs/2610.32127) | 该论文提出消息传递量化（MPQ），将训练后量化重新建模为离散码本上的后验推断，通过在权重块间传递软后验信息并在最后阶段才做硬判决，从而超越GPTQ等现有硬判决求解器，提升大语言模型低比特量化的精度。 |
| [^125] | [Conditional Flow Models for Offline Maximum Entropy Reinforcement Learning](https://arxiv.org/abs/2610.32125) | 该论文提出一个分离式框架，利用基于概率流ODE的条件扩散模型分别估计奖励分布和转移核，从而构建软最优贝尔曼算子的数据驱动近似来估计最优软动作价值函数，并提供了非渐近理论保证。 |
| [^126] | [Semantically Rich, Metrically Affect-Blind: A Geometric Blind Spot in Modern Text Embedders](https://arxiv.org/abs/2610.32122) | 现代文本嵌入器的嵌入向量中几乎可以完美解码情感信息（线性探针准确率高达91.3%），但其度量空间（余弦相似度）几乎完全无法区分情感，导致依赖度量的检索、聚类、去重和RAG系统对这类信息“视而不见”。 |
| [^127] | [Tampered ML: Evaluating LLM Judgment of Machine Learning Research](https://arxiv.org/abs/2610.32114) | 该论文提出Tampered ML基准，通过将真实机器学习论文的实验提案与人为引入方法论缺陷的最小修改版本配对，来评估大语言模型识别研究方法缺陷的能力，发现前沿模型虽优于随机水平但远未达到可靠程度。 |
| [^128] | [REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories](https://arxiv.org/abs/2610.32113) | REACT 提出一种从成对安全与不安全轨迹中学习安全方向、并一次性施加于思维链早期隐藏状态的激活干预方法，发现最强控制增益集中在思维链的前两到三个 token 窗口，即使后续状态更能预测风险。 |
| [^129] | [Putting Transformer Theories to the Test](https://arxiv.org/abs/2610.32104) | 该论文将七种关于Transformer能力的理论预测统一到同一矩阵中，并通过受控实验验证，发现没有任何单一理论能完全解释Transformer在不同计算任务上的表现。 |
| [^130] | [ReMedI: Inference-Time Refinement and Self-Distillation for Controllable Molecular Optimization](https://arxiv.org/abs/2610.32103) | 该论文提出ReMedI方法，以分子语言模型作为基础提议者，结合高效的Metropolis-Hastings搜索策略与推理时细化及自蒸馏，在无需昂贵任务特定数据集的情况下实现可控的多目标分子先导化合物优化。 |
| [^131] | [PRISM: Promoting Sampling Diversity of Large Language Models with Implicit Stochastic Modulation](https://arxiv.org/abs/2610.32101) | PRISM是一个训练时框架，通过次模数据筛选与随机嵌入调制两种机制，将采样多样性内化为大语言模型自身的能力，从而突破质量-多样性的权衡前沿。 |
| [^132] | [RADIANT: Object-Centric Feed-Forward Global Illumination](https://arxiv.org/abs/2610.32075) | 提出RADIANT架构，以物体为中心、每个物体仅用32个表面补丁token表示，将注意力序列长度与多边形数量解耦，从而在无需逐场景优化或运行时光线追踪的情况下高效合成新3D场景的全局光照。 |
| [^133] | [DBCM: Dual-Branch Calibration Module for Test-Time Adaptation in Time Series Forecasting](https://arxiv.org/abs/2610.32071) | 提出双分支校准模块DBCM，通过共享适配器利用跨通道预测残差的共享结构，并结合低秩通道分支保留有限的通道特定灵活性，实现时间序列预测中更高效的测试时适应。 |
| [^134] | [Controlled Evaluation of EEG Foundation Models: Random Initialisation, Band-Power and Validation-Selected Probes](https://arxiv.org/abs/2610.32067) | 该论文通过引入未训练编码器、频带功率基线和基于验证集的层选择三种对照，对多个EEG基础模型进行受控评估，发现其预训练收益大多不具统计稳健性，随机初始化和简单的频带功率基线即可与之匹敌。 |
| [^135] | [Concept Acupuncturist: An Efficient and Precise Concept Erasure Method for Diffusion Models via Sparse Autoencoders](https://arxiv.org/abs/2610.32059) | 该论文提出ConceptACP框架，利用稀疏自编码器的可解释性分析定位U-Net去噪器中与目标概念最相关的交叉注意力块并进行针对性微调，从而实现对扩散模型概念的高效精确擦除。 |
| [^136] | [How Much Do You Trust A Dog? Quantifying Hint-Following in Reasoning Language Models](https://arxiv.org/abs/2610.32055) | 该研究发现大语言模型对提示来源（如教授、朋友甚至狗）的答案偏移遵循可预测的指数倾斜结构，且偏移强度与来源可靠性基本无关，揭示了模型谄媚行为未被来源可靠性校准的问题。 |
| [^137] | [A Dynamic Inventory Rationing Model for Last-Mile Delivery of Medication in Liberia's Community Health Program](https://arxiv.org/abs/2610.32051) | 该论文将利比里亚社区卫生计划中稀缺药物的动态分配问题建模为有限时域随机控制问题，在转诊成本、患者未治疗成本与地区间公平性之间取得平衡，以解决“将库存下放给社区卫生助理”与“集中保留库存以应对不确定需求”之间的权衡难题。 |
| [^138] | [Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery?](https://arxiv.org/abs/2610.32046) | 提出测试时记忆演化（TTME），一种无需更新模型权重的即插即用测试时适应方法，通过持续演化外部记忆来积累经验，在科学发现任务上以更高的计算效率超越了测试时训练（TTT）。 |
| [^139] | [NUViT: Vision Transformers for Fourier Space Data](https://arxiv.org/abs/2610.32041) | 提出了NUViT，一种通过重新设计图像分词、数据表示和位置编码以适配傅里叶空间数据频率与相位结构的视觉Transformer，在图像分类、晶格识别、MRI标注和冷冻电镜分析等多种傅里叶域任务中均优于标准架构。 |
| [^140] | [Rethinking Representation Geometry in Supervised Contrastive Learning under Long-Tailed Data](https://arxiv.org/abs/2610.32039) | 本文揭示了现有长尾数据下的监督对比学习目标因依赖每类样本数量而无法在最优解处形成正则单纯形几何，并提出结构平衡方法与完全平衡的对比目标，可证明地实现正则单纯形几何。 |
| [^141] | [MAGIC: Margin-gated Contrastive Debiasing with Evidence Restoration for VLM Test-time Robustness Scaling](https://arxiv.org/abs/2610.32013) | 提出了MAGIC方法，通过测量关键token的决策边际并利用门控机制，仅对易受语言偏见影响的低置信度响应选择性激活文本与视觉扰动分支进行对比去偏，同时结合证据恢复，实现高效且鲁棒的视觉语言模型测试时推理。 |
| [^142] | [GS-MoE: Training-Free MoE Compression via Groupwise Subspace Sharing and Pruning](https://arxiv.org/abs/2610.32009) | GS-MoE提出了一种免训练的MoE压缩框架，通过分组子空间共享的低秩表示与基于Nyström的神经元宽度剪枝，从专家间和专家内两个维度消除冗余，并利用重建误差指导专家分组与分层预算分配。 |
| [^143] | [TRACE: Data-Free Text Reconstruction Attacks against Approximate Unlearning in LLMs](https://arxiv.org/abs/2610.31995) | 本文提出TRACE框架，利用遗忘前后的模型快照差异，在无需先验知识的情况下重建大语言模型中被近似遗忘的训练数据，揭示了模型快照所引入的隐私泄露侧信道风险。 |
| [^144] | [REDAR: Reinforcement-Driven Adaptive Regularisation for Class Incremental Learning](https://arxiv.org/abs/2610.31991) | REDAR将类增量学习中的正则化系数选择建模为序列决策问题，利用轻量级强化学习智能体在每个增量阶段根据性能指标和数据统计自适应地输出蒸馏系数，解决了传统固定系数方法无法适应各阶段数据变化且不匹配持续学习目标的根本缺陷。 |
| [^145] | [WADE: An LSTM-Based Anomaly Detection Engine for WebAssembly Applications](https://arxiv.org/abs/2610.31984) | WADE 提出了一种基于 LSTM 的异常检测引擎，通过学习 WebAssembly 应用在良性条件下的正常函数级执行轨迹，无需漏洞签名即可在浏览器运行时检测已知和未知的攻击行为。 |
| [^146] | [Compatible Likelihoods for Flow Matching on Manifolds](https://arxiv.org/abs/2610.31981) | 该论文针对黎曼流匹配中基于似然的目标可能产生推离数据分布的速度场这一失败模式，提出“相容性”概念刻画似然目标的有效性，并进一步设计了一种基于指数族的新型目标，既满足相容性又仅支持紧流形上可行的速度。 |
| [^147] | [Beyond Heuristics and Commercial Navigation: An AI-Augmented Decision-Support for Ambulance Travel Time Estimation](https://arxiv.org/abs/2610.31968) | 该研究基于瑞典斯科讷地区的真实救护车历史运营数据，揭示了救护车行驶时间随时段、日期和行程优先级显著变化，并开发了机器学习系统以超越传统启发式假设和商业导航系统，从而更准确地估计救护车行驶时间。 |
| [^148] | [A Model-Agnostic Correction Framework for Short-Term Building Load Forecasting](https://arxiv.org/abs/2610.31959) | 提出了一种模型无关的天气-负荷检索增强校正框架WeL-RAC，通过在预测时基于近期负荷状态和天气预报检索相似历史案例，并聚合其观测到的未来负荷轨迹来校正初始预测，从而提升短期建筑负荷预测在工况快速变化下的可靠性。 |
| [^149] | [A Mathematical Theory of the Jacobian Lens](https://arxiv.org/abs/2610.31955) | 本文提出了雅可比透镜（J-lens）的数学理论，证明该可解释性工具优先识别的是相干特征——即类似主导 Lyapunov 模态、可跨层放大、跨词元位置恒定且跨上下文一致的线性方向。 |
| [^150] | [RobustNCL: Robustifying Neuromorphic Continual Learning under Transient Faults with Lightweight Detect-and-Cure Strategy](https://arxiv.org/abs/2610.31954) | 本文提出RobustNCL框架，通过轻量级检测与修复策略提升神经形态持续学习在硬件瞬态故障（特别是DRAM片外存储器故障）下的可靠性。 |
| [^151] | [Sharp Bounds on the Winner's Bias under Independent and Nonnegatively Correlated Noise](https://arxiv.org/abs/2610.31948) | 本文精确确定了在独立非同分布噪声（$\sqrt{(M-1)/(2M)}$）与两两非负相关噪声（$(M-1)/M$）下，从 $M$ 个无偏含噪分数中选取最优值所导致的赢家偏差的紧致常数，给出了达到这些界的显式构造，所有定理经 Lean 4 形式化验证，并据此作为推论精确界定了交叉验证估计器的偏差。 |
| [^152] | [Kolibri: A Sovereign European Model on the Pareto Frontier](https://arxiv.org/abs/2610.31945) | Kolibri是一个总参数78.1B、激活参数仅3.46B的英德双语开源混合专家模型，通过完全自主掌控的数据与训练流程实现德语占比超20%（含2T自建德语token）的训练，在帕累托前沿上实现了面向主权化部署的高效性能。 |
| [^153] | [Towards Trustworthy AI-Enabled Emergency Medical Dispatch: Uncertainty-Aware Ambulance Travel Time Estimation](https://arxiv.org/abs/2610.31944) | 该论文提出一种不确定性感知的救护车行程时间估计方法，通过量化每次估计的不确定性并处理真实数据中的领域特定异常，提升AI辅助紧急医疗调度的可信赖性。 |
| [^154] | [Agreement rather than normalisation keeps depth reusable in looped language models](https://arxiv.org/abs/2610.31943) | 循环语言模型深度可复用的关键在于训练时与推理时循环配置的一致性，而非循环步之间的状态归一化。 |
| [^155] | [Beyond Local Accuracy: Benchmarking the Rollout Reliability of PDE Foundation Models](https://arxiv.org/abs/2610.31935) | 该研究提出“误差放大”指标G来基准测试PDE基础模型在自回归滚动预测中的可靠性，发现尽管微调的基础模型在短时程预测中普遍优于从零训练的基线模型，但其短时程优势未必能在长时程滚动中保持。 |
| [^156] | [Every Weight Is a Path of Shared Discrete Decisions](https://arxiv.org/abs/2610.31931) | 该论文提出TWIG树结构权重表示，将神经网络中每个权重表达为共享离散决策路径的终点，从而把剪枝、量化、低秩分解和权重共享统一为对同一对象的编辑，并使权重层可转化为几乎只含导线和加法器的硬件电路。 |
| [^157] | [ENCL: Enabling Efficient Neuromorphic Continual Learning with Local Learning Rule and Sub-Network Freezing Strategy](https://arxiv.org/abs/2610.31927) | 本文提出ENCL方法，通过轻量级的时空局部学习规则和子网络冻结策略，取代传统神经形态持续学习中内存消耗大的经验回放和昂贵的全局学习规则，实现了高效的持续学习。 |
| [^158] | [Input Bias Correction in Gaussian Process Regression](https://arxiv.org/abs/2610.31920) | 提出偏差感知高斯过程（Ba-GP）框架，利用已记录GP的后验均值梯度将输入偏差和噪声传播为均值与协方差校正，并借助加性协方差结构开发数值高效的一阶校正算法，从而在获得更优输入估计时高效修正基于错误输入训练的高斯过程模型。 |
| [^159] | [Decode Late, Multiply Once: Rethinking the Dot Product for Quantized Weights](https://arxiv.org/abs/2610.31913) | 该论文提出“延迟解码、只乘一次”的点积重构方法：将k比特权重编码直接用作地址偏移来选择2^k个累加器之一，最后统一进行融合的解码与乘法，从而在整数运算下精确等价的前提下，将量化权重点积的乘法器数量减少n/2^k倍，并支持任意（含非线性）解码器和矩阵/张量乘积。 |
| [^160] | [Topological Equivalence of Spectral and Density Cluster Hierarchies](https://arxiv.org/abs/2610.31909) | 本文证明了在适当条件下，基于扩散距离的谱聚类层次结构与基于密度超水平集的密度聚类树在拓扑上等价，且该等价性可通过k-近邻图实现推广到有限样本情形。 |
| [^161] | [Real-to-Sim Calibration of Bosch Etching Simulators with Few-Shot XSEM Data and Hybrid Search](https://arxiv.org/abs/2610.31904) | 提出了一种少样本真实到仿真校准框架，通过将模拟器参数与工艺变量解耦，并结合局部探索、LLM引导候选生成和XGBoost的混合搜索策略，从稀疏XSEM测量中推断特定配方的Bosch刻蚀模拟器参数，实现对刻蚀深度和侧壁角度的可靠预测。 |
| [^162] | [Learning Structured Sparsity for Large Models: Bridging Model Compressibility and Efficient Inference](https://arxiv.org/abs/2610.31897) | 提出统一框架S-SALAAD，联合学习模型权重与结构化稀疏+低秩表示，使大模型既能弹性压缩以适配边缘设备的多种内存预算，又能实现真正加速的高效推理。 |
| [^163] | [Combinatorial Characterizations of the Learnability of Conditional Distributions](https://arxiv.org/abs/2610.31890) | 本文引入fat_γ Natarajan维数和fat_γ Graph维数两个组合维度，证明前者在每个正尺度上的有限性是条件分布函数类PAC可学习性的充要条件，并证明分布值函数类可学习当且仅当其每个坐标类均可学习。 |
| [^164] | [Scaling Transformer Energy-based Models For High Quality Image Synthesis](https://arxiv.org/abs/2610.31889) | 本文提出Transformer能量基模型（TEM），通过在潜空间中运行的可扩展transformer能量函数、多层能量组合以及正表示对齐（PRA）等技术，克服了能量基模型在高维采样与深度架构训练中的瓶颈，实现了高质量的大规模图像合成。 |
| [^165] | [Effectiveness of Curriculum Learning Depends on Reward Sparsity and Competing Optima](https://arxiv.org/abs/2610.31884) | 通过构建并分析策略学习的最小化模型，该研究证明了课程学习的有效性取决于奖励的稀疏程度：在稠密奖励下课程仅有边际收益，而在稀疏奖励下课程可能是学习得以发生的必要条件。 |
| [^166] | [Forecasting Associations, Not Interventions: A Preregistered Causal Audit of Covariate-Aware Time Series Foundation Models](https://arxiv.org/abs/2610.31869) | 该研究通过理论与预注册实验证明，协变量感知的时间序列基础模型（如Chronos-2、TimesFM-2.5等）对“假如”式干预查询给出的只是关联性预测而非真实因果效应，并推导出闭式的无参数混淆差距参考线，实验显示这些模型比回归本身更严重地受混淆影响。 |
| [^167] | [Rewriting Scaling Laws for Synthetic Data with Compute and Data Constraints](https://arxiv.org/abs/2610.31867) | 本文提出了首个面向合成重写数据的缩放定律，将生成重写的计算成本纳入建模，发现最优重写数据比例随计算预算的增加而显著上升。 |
| [^168] | [Aligning Large Language Models with Human Feedback: Mathematical Foundations and Algorithm Design](https://arxiv.org/abs/2610.31862) | 本文系统介绍了大语言模型对齐的数学基础与算法框架，涵盖SFT、RLHF和DPO等主流技术，并将其理论与逆强化学习和离散选择模型建立联系，为开发更稳健、可扩展的对齐方法提供了全面参考。 |
| [^169] | [Cautious Conflict-Free Inverse Gradient Methods](https://arxiv.org/abs/2610.31855) | 针对ConFIG方法与Adam等自适应优化器结合后权重更新可能丧失无冲突特性的漏洞，提出了仅需三行代码扩展的谨慎ConFIG（Cautious ConFIG）方法来保障PINN训练中更新的无冲突性。 |
| [^170] | [Rethinking Multi-View Fusion for Urban Representation Learning](https://arxiv.org/abs/2610.31850) | 本文实证揭示城市多视图融合中模态效用区域差异与多目标梯度不平衡两大结构性问题，并提出自适应稀疏MoE路由的自调节框架PRISM来学习区域特定的视图分配，实现更稳健的城市表示学习。 |
| [^171] | [Don't Forget What You Just Learned: Self-Directed Repair Turns Agent Mistakes Into Lasting Lessons](https://arxiv.org/abs/2610.31846) | 提出Self-Direction机制，让智能体在任务失败后自我反思提取教训，并立即蒸馏为轻量级低秩适配器（仅占基础模型约0.02%参数量）存入持久化库中，从而将错误转化为不遗忘、不破坏原有能力的持久化学习成果。 |
| [^172] | [SEVI: Safe Evidential Real-World Reinforcement Learning](https://arxiv.org/abs/2610.31829) | SEVI提出证据式安全评论家框架，在单次前向传播中同时估计约束违反风险与认知不确定性，并在真实世界策略学习中对不安全或证据不足的动作进行离线恢复策略干预，从而实现安全的真实世界强化学习。 |
| [^173] | [Speculative Diffusion is a Tale of Two Tradeoffs](https://arxiv.org/abs/2610.31819) | 本文揭示了投机扩散中存在的两种质量与速度的权衡：扩散模型可通过牺牲输出质量换取更快的采样速度，而投机解码也可通过更宽松的接受标准推广为有损方式以加速生成。 |
| [^174] | [Nested Sparsity for Scalable Pretraining and Continual Learning](https://arxiv.org/abs/2610.31818) | 本文提出 Nestor，一种推广注意力算子的嵌套稀疏机制，可与 MoE 稀疏性层级组合，在相同计算量和稀疏度下预训练性能优于传统 MoE transformer，并能更有效地进行持续预训练和适配更长上下文。 |
| [^175] | [CuraLoc: Repository-Level Fault Localization via Planned Code Inspection](https://arxiv.org/abs/2610.31809) | CuraLoc提出在源码检索前一次性规划代码检查的缺陷定位方法：模型在单次规划中同时选定起始符号、检查关系并形成修改假设，再由确定性检索完成最终排序，从而减少反复的模型调用和token消耗。 |
| [^176] | [PaperScorer: How Predictable Is Paper Acceptance?](https://arxiv.org/abs/2610.31791) | 该论文提出PaperScorer系列模型，仅凭论文文本或页面截图即可预测同行评审的录用或拒稿结果，其中视觉模型表现最佳（ICLR论文68%、arXiv机器学习论文77%准确率），表明论文的图表、布局与呈现方式蕴含着重要的评审决策信号。 |
| [^177] | [Not All Retained Clients Are Equal: Client Selection for Federated Unlearning](https://arxiv.org/abs/2610.31786) | 提出PDF框架，通过在训练期间维护路径依赖、CLP加权的轻量级客户端历史信息，在联邦遗忘时识别并排序最有价值的保留客户端，而非盲目依赖所有保留客户端，从而更高效地移除目标客户端的影响。 |
| [^178] | [Early learning in large residual networks](https://arxiv.org/abs/2610.31783) | 该论文通过图表计算方法分析了大型残差网络在早期训练阶段的动态，发现在大输入维度极限下，网络在训练初期只能学习到输入的线性函数，无法学习更高阶的单项式。 |
| [^179] | [Benchmarks Are Surprisingly Compaction-Friendly](https://arxiv.org/abs/2610.31778) | 该论文首次在固定的智能体框架内实现五种上下文压缩策略，并在三个长时程基准上系统评估三个前沿模型，发现基准任务对有损压缩出人意料地友好——对百万级上下文模型而言，压缩阈值几乎不影响任务性能。 |
| [^180] | [Efficient Omni-modal Large Language Model Inference via Per-Query Modality Selection](https://arxiv.org/abs/2610.31768) | 提出一种逐查询自适应模态选择方法，在不改动全模态大语言模型本身的情况下，通过成本敏感学习为每个查询选择最优模态组合，持续改善精度与推理成本之间的权衡。 |
| [^181] | [Learning to Be Monitored: Causal Bounds and Intervention-Aware Post-Training Recipes for Chain-of-Thought Monitorability](https://arxiv.org/abs/2610.31733) | 该论文指出常用的 CoT 监测评估指标 TPR_mc 只是因果检测率的上界并推导出互补下界，据此建立了更严格的 CoT 可监测性评估框架，并探索通过后训练方法使大语言模型的思维链更易于被监测。 |
| [^182] | [ConsMSA: Semantic Distribution Consistency Learning for Multimodal Sentiment Analysis](https://arxiv.org/abs/2610.31731) | 该论文提出ConsMSA，通过将多模态特征投影到共享情感空间并计算模态内与模态间一致性得分（I²CS），构建一致性感知的重要性信号来解决多模态序列中的语义不一致问题，从而提升多模态情感分析的鲁棒性与效率。 |
| [^183] | [When and How Does Muon Precondition? The Role of Structured Gradient Noise](https://arxiv.org/abs/2610.31728) | 本文从理论上揭示了 Muon 的预条件收益取决于梯度噪声的结构——在各向同性噪声下 Muon 不带来超越学习率缩放的加速，而在异构任务频率产生的结构化噪声下，Muon 能将对任务频率的依赖从线性降为平方根，从而平衡高低频任务的学习并避免 SignSGD 中由旋转引起的负迁移。 |
| [^184] | [Cognitively Constrained Partners for Zero-Shot Coordination in Real-Time Partitioned Control](https://arxiv.org/abs/2610.31726) | 本文提出一种基于人类生理学约束（反应延迟与感知限制）生成多样化模拟伙伴的框架，用于训练人机协作智能体，初步证明这些伙伴的行为更接近真实人类，并能实现与真人的零样本协调。 |
| [^185] | [Operationalising Weaker Faithfulness: Recovery, Ambiguity and Computational Limits](https://arxiv.org/abs/2610.31724) | 本文提出两种操作化弱忠实性的策略——邻接学习中的选择性边保护与显式保留备选有向图解释（解释家族），使结构学习器能够恢复单独独立但联合预测结果的变量（如XOR）之间的联系，将完整父对恢复率从0/20提升至20/20，且家族估计器在48/50的样本中保留了全部备选解释。 |
| [^186] | [Autoregressive Multimodal Anomaly Detection with Large Language Models for Microservice Systems](https://arxiv.org/abs/2610.31722) | 该论文提出了LLMMAD，一个基于大语言模型的自回归多模态异常检测框架，通过在跨模态融合中保留模态特定特征并解决“身份捷径”问题，来提升微服务系统的异常检测效果。 |
| [^187] | [End-to-End Differentiable Diffusion Conditioning for Physics-Informed Optimization](https://arxiv.org/abs/2610.31721) | D2C利用确定性采样使反向去噪过程可微的特性，将冻结的条件扩散模型转化为端到端的梯度优化器，在测试时通过代理目标和物理信息的梯度直接优化条件变量，从而克服了离线基于模型优化中现有方法过于保守或无法精确对齐物理目标的问题。 |
| [^188] | [Towards Adversarially Robust Multigraph Neural Networks for Financial Crime Analysis](https://arxiv.org/abs/2610.31718) | 该论文提出了首个针对多重图神经网络的结构攻击与对抗训练可扩展框架，通过节点-边关联矩阵优化结构扰动，并引入基于有向环计数的正则化器衡量攻击合理性，从而提升反洗钱任务的对抗鲁棒性。 |
| [^189] | [Does LeWorldModel Preserve Local Action Geometry?](https://arxiv.org/abs/2610.31716) | 本文通过在多个扰动尺度上使用 Gram 矩阵评估预训练 LeWorldModel 对微小动作变化的响应，发现该世界模型未能很好地保留局部动作几何结构，并提出应对动作敏感性进行针对特定尺度的评估。 |
| [^190] | [Reuse, Rewrite, or Reject: Evidence-Constrained Trace Reuse for Repeated Mobile GUI Control](https://arxiv.org/abs/2610.31712) | 该论文提出图-文本模板与“记录-模板-回放”框架，通过对UI状态转移进行证据约束验证，实现移动端GUI智能体在重复任务中对先前执行轨迹的选择性复用，从而降低冗余的LLM推理与执行开销。 |
| [^191] | [The Path to NDE 4.0: A Review of Algorithms, Techniques, and Strategies for Autonomous Inspection](https://arxiv.org/abs/2610.31704) | 本文从计算算法、物理技术、工业应用和研究目标四个视角全面综述了无损检测领域向NDE 4.0数字化数据驱动范式的演进，指出深度学习和生成模型正推动自动缺陷识别的发展，同时揭示标准化开放数据匮乏是行业面临的主要瓶颈。 |
| [^192] | [Collaborative In-Context Learning for Indoor Radio Map Estimation](https://arxiv.org/abs/2610.31702) | 提出Poly-Ripple框架，首次将室内无线电地图估计形式化为协作式上下文学习问题，使多个客户端利用大型自回归视觉模型和波纹序自回归建模，在无需模型更新并保护隐私的前提下协作预测无线电地图。 |
| [^193] | [Fooling LLM Safety Guards into Judging the Wrong Text](https://arxiv.org/abs/2610.31695) | 论文提出“目标重绑定”攻击方法——通过附加伪造查询和无害诱饵，利用安全防护机制只评判输入中最后一个完整任务的特性，使有害内容绕过审查，在GuardRebind基准上对五种防护机制的绕过率均超过93%。 |
| [^194] | [Multi modal perceived conversation success prediction for spontaneous autistic and non-autistic dyadic interactions](https://arxiv.org/abs/2610.31682) | 本研究提出利用构音、面部、声学-韵律和话轮转换等多模态特征，在包含自闭症与非自闭症个体的自发双人对话数据中预测参与者感知的对话成功，证明对话成功本质上是多维的，需要多层面的分析框架。 |
| [^195] | [Autoregressive Physics-Informed Neural Operator Surrogates and Joint 4-D Seismic-Production Bayesian History Matching](https://arxiv.org/abs/2610.31677) | 该论文提出了一种以自回归物理信息傅里叶神经算子作为代理模型加速的贝叶斯历史拟合框架，通过两阶段 aREKI 与 ES-MDA 策略联合同化生产数据与四维地震数据，显著降低了大规模油藏反演的计算成本与病态性。 |
| [^196] | [Hybrid SEIR Temporal Fusion Transformer Framework with SHAP Analysis for Interpretable Pandemic Forecasting](https://arxiv.org/abs/2610.31673) | 该论文提出了一种将SEIR流行病学模型与时间融合Transformer（TFT）相结合的混合预测框架，并借助SHAP分析实现跨地区大流行病感染的可解释多时间步预测。 |
| [^197] | [Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success](https://arxiv.org/abs/2610.31670) | 提出无需训练的离线指标“感知运动对齐”（S²），通过衡量模型感知表征与物理运动几何结构的对齐程度，低成本地预测具身模型的下游任务成功率。 |
| [^198] | [SABLE: Scalable Affine Bayesian Latent-space Exploration](https://arxiv.org/abs/2610.31495) | SABLE 提出了一种贝叶斯仿射代理优化方法，通过将查询分解为由期望改进（EI）优化的低维跨度内分量和由标量范数支配的跨度外分量，在评估预算远小于潜空间维度的高维潜空间中实现了可扩展的贝叶斯优化。 |
| [^199] | [Convex Equilibrium Potentials for Monge--Amp\`ere: Curvature Bounds and A Posteriori Evaluation](https://arxiv.org/abs/2610.31386) | 本文针对用于Monge–Ampère方程的凸softplus平衡神经网络，证明了Hessian矩阵关于压缩因子具有最优的 $(1-q)^{-2}$ 增长界，并给出了基于残差的后验估计以控制近似平衡点处的状态及其导数误差。 |
| [^200] | [Harmful Responses Are Jailbreak Templates: Transferring Autoregressive Jailbreaks to Diffusion Language Models](https://arxiv.org/abs/2610.31331) | 本文发现有害回复经部分掩码后可天然充当越狱模板，并据此提出RMT-Jailbreak框架，将现有的自回归模型越狱攻击有效迁移到扩散语言模型上。 |
| [^201] | [The Value of System 1 Models in Medicine](https://arxiv.org/abs/2610.31315) | 本研究首次大规模评估了四个系统1模型在20种语言、14个临床领域共86个任务上的零样本表现，表明这类能快速输出概率并具备不确定性估计能力的模型，在临床决策中具有独特价值。 |
| [^202] | [Track finding and particle identification performance with particle physics foundation models](https://arxiv.org/abs/2610.31180) | 该论文在时间投影室数据集上评估了粒子物理基础模型在径迹寻找和粒子鉴别等下游任务中的表现，证明基于基础模型的数据分析方法性能显著优于传统方法，且可灵活适应新任务。 |
| [^203] | [Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters](https://arxiv.org/abs/2610.31017) | 开卷评分主要衡量的是所有适配器共有的“阅读”行为而非文档记忆——随机适配器即可恢复专用适配器82%的开卷优势，但在无源文本的完形测试中仅恢复约2%（与零无异），因此不宜用开卷评分来评估或选择文档级LoRA适配器。 |
| [^204] | [OpenAttention: One Percent Is Enough with Tail Proxies](https://arxiv.org/abs/2610.30951) | OpenAttention通过将整个未读尾部压缩为单个计数感知均值代理token，在读取率低至1%时仍能获得比丢弃或采样更低误差的稀疏注意力输出。 |
| [^205] | [Lean Finder V2: Semantic Search for Mathlib That Disambiguates User Queries](https://arxiv.org/abs/2610.30843) | Lean Finder V2 通过互补的模型级对抗训练与输入级查询重写，实现了能够消除歧义的 Mathlib 语义搜索，有效解决了短而模糊查询的检索难题。 |
| [^206] | [Scaling Laws for Safety Probes](https://arxiv.org/abs/2610.30678) | 本文通过构建六个新安全数据集、在20个基础模型上以不同标签规模训练探针，首次提出了可依据模型规模和标签数量提前预测安全探针性能的缩放定律。 |
| [^207] | [Residual Calibration via Local Feature-Space Refinement](https://arxiv.org/abs/2610.30652) | 提出了局部自适应概率估计（LAPE），一种即插即用的事后残差校准框架，它以全局校准器输出为锚点，利用相邻校准样本的非参数局部证据对预测进行精细化，从而捕捉特征空间中局部区域的空间变化失准。 |
| [^208] | [Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy](https://arxiv.org/abs/2610.30474) | 本文提出了一种基于原子可解释单元的SAE评估新框架，将超越单义性的评估目标形式化，并借此揭示了现有SAE特征中隐藏的多义性与冗余性。 |
| [^209] | [Differentially Private Synthetic Data via Generalized Moment Matching](https://arxiv.org/abs/2610.30196) | 提出了广义矩匹配这一通用框架，通过私有估计矩、拟合概率测度并采样合成记录来生成差分隐私合成数据，可应用于Wasserstein距离、最大均值差异等多种积分概率度量，并在固定维度下达到相差对数因子的极小极大最优误差率。 |
| [^210] | [The Price of Current-Effect Inference in Nonstationary Multi-Armed Bandits](https://arxiv.org/abs/2610.29982) | 该论文揭示了非平稳多臂老虎机中收益最大化与现时效应推断之间的内在权衡，证明了以有限宽度置信区间同时覆盖所有现时效应所需付出的动态遗憾极小极大代价，并提出 TRACE 算法来同时兼顾这两个目标。 |
| [^211] | [Norm-optimal model order reduction in reproducing kernel Hilbert spaces](https://arxiv.org/abs/2610.28224) | 本文建立了再生核希尔伯特空间中的范数最优模型降阶框架，通过引入广义Malmquist-Takenaka函数与可分RKHS间的酉变换，将最优性条件和相应算法从可微核函数扩展到了不可微核函数的情形。 |
| [^212] | [Attractive Faces Skew Text-to-Image Reward Models and Leaderboards](https://arxiv.org/abs/2610.28195) | 研究发现人脸吸引力在人类偏好判断中会压倒图像质量本身，这一偏差会扭曲文生图模型的奖励模型和排行榜，使AI生成的有吸引力面孔被排在真实但不吸引人的面孔之上。 |
| [^213] | [Foundations of the Scaling Law](https://arxiv.org/abs/2610.27789) | 本论文通过几何近似、谱分析和约束优化三个相互关联的分析框架，为缩放定律建立了理论基础，并区分解释了涌现、顿悟和测试时缩放等现象。 |
| [^214] | [Privacy Accounting for Empirical Denoisers](https://arxiv.org/abs/2610.27710) | 本文提出一种无需训练的截断隐私可问责经验去噪器，通过显式计算其每步敏感性，使扩散模型采样过程中的每一步去噪都可建模为高斯机制，从而实现对采样过程本身隐私损失的形式化核算。 |
| [^215] | [CyberHull: Deterministic Malicious URL Isolation via Convex Hull](https://arxiv.org/abs/2610.27675) | 提出CyberHull框架，利用QuickHull算法在多个正交特征域上构建凸包边界，以资源高效、确定性的方式实现恶意URL的识别与隔离，训练复杂度仅为O(n log n)。 |
| [^216] | [MLIPS: Multi-Layer Indexer Pool Sharing](https://arxiv.org/abs/2610.27438) | MLIPS是一种无需训练的方法，通过让一个生产者层发布超出预算的top-M候选池供后续多个层共享复用，在保留各层独立打分与选择的同时消除重复的候选搜索，在DeepSeek-V4-Flash上实现了1.14–1.40倍加速和14.8–27.4%的GPU能耗降低，且准确率几乎不受影响。 |
| [^217] | [Quadratic Sample Complexity for Strictly Replicable Realizable Learning with Unknown Marginals](https://arxiv.org/abs/2610.26944) | 该论文首次证明，在分布及其边缘均未知的严格可复制可实现学习中，样本复杂度具有二次方下界Ω(d²/(ερ²))，并通过泊松化球采样算法为完整布尔类P_d给出了匹配的Θ(d²/(ερ²)+ε⁻¹log(1/δ))上界，完整刻画了该问题的复杂度。 |
| [^218] | [Residual-Based Adaptive Refinement in Hybrid Variational-Collocation PINNs for Singular Elliptic PDEs with Mixed Dirichlet-Neumann Boundary Conditions](https://arxiv.org/abs/2610.26458) | 提出一种混合变分-配置物理信息神经网络框架，通过基于残差的自适应配置点细化策略，有效捕捉混合狄利克雷-诺伊曼边界条件下椭圆偏微分方程在凹角和裂纹尖端附近的解奇异性。 |
| [^219] | [EvoSkill v2: Fully Agentic Skill Evolution from Benchmark Adaptation to a Stateful Evolver](https://arxiv.org/abs/2610.25931) | EvoSkill v2 提出了一种由三个智能体组成的技能演化系统，其核心创新在于采用一个贯穿全程、带有 bash 工具的有状态编码智能体在每个训练周期持续重写技能指令，并结合多次尝试与无筛选接受策略，在不改动底层模型的前提下提升智能体的基准测试表现。 |
| [^220] | [Anisotropic Representation Dynamics in Transformers](https://arxiv.org/abs/2610.23242) | 该论文提出了混合门控的各向异性表示动力学理论，将自注意力分解为数值传递、查询门控与键门控，揭示了门控机制可诱导表示差异从收缩切换为扩张的状态转换，并给出了判定该现象的充分必要谱判据。 |
| [^221] | [SpectrumQuant: Spectrum Alignment for LLM Quantization](https://arxiv.org/abs/2610.20303) | 提出SpectrumQuant，一种基于协方差最大化的闭式正交旋转方法，通过频谱对齐构建量化友好的变换，从而提升大语言模型训练后低比特量化的效果。 |
| [^222] | [Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs](https://arxiv.org/abs/2610.20179) | 本文提出跨模型上下文内神经反馈新范式，首次将模型在任务上的胜任能力与对自身激活的真实访问区分开来，测试十个开源大模型后发现大语言模型仅具备有限的激活控制能力。 |
| [^223] | [Interaction-as-a-Judge](https://arxiv.org/abs/2610.17895) | 提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。 |
| [^224] | [Barycentric Guidance: Turning Foundation Image Editors into Continuous Affective Controllers](https://arxiv.org/abs/2610.16892) | 提出无需训练的“重心引导”框架，通过将任意目标效价-唤醒度点映射为基于单纯形的引导权重，把预训练基础图像编辑器转变为可精确连续控制情绪的情感控制器，且无需微调、额外数据或模型修改。 |
| [^225] | [Latent-Space Importance Weighting under Covariate Shift](https://arxiv.org/abs/2610.15571) | 该论文提出一种针对协变量偏移的迁移学习方法，在低维潜在空间中将表示学习与样本重加权相结合，并给出了目标域超额风险率等理论保证与实验验证。 |
| [^226] | [Prediction-Powered Data Fusion for Treatment Effect Estimation](https://arxiv.org/abs/2610.12332) | 提出了一种无需对观察性研究做特殊假设的数据融合框架，通过保持RCT估计的无偏性并从大型观察性研究中借力，显著提升平均处理效应（ATE）和条件平均处理效应（CATE）的估计精度。 |
| [^227] | [A structure-preserving neural density functional for the ions of a polymer electrolyte](https://arxiv.org/abs/2610.12132) | 开发了一种保持空间对称性、热力学可积性和诺特定理恒等式的神经密度泛函，仅需平面密度和内力数据训练即可准确预测聚合物电解质离子在不同浓度和尺度下的结构与关联行为。 |
| [^228] | [Ranking Prior Alignment for Credit Risk Modeling: When Do External Priors Matter?](https://arxiv.org/abs/2610.11146) | 提出了一种模型无关的“排序先验对齐”框架，通过温度缩放的KL散度损失将来自领域专家、教师模型或大语言模型的外部排序先验统一蒸馏进神经或树类评分模型，以缓解标注数据稀缺的冷启动信贷评分难题。 |
| [^229] | [Safe at One Loop, Risky at Another: Aligning Safety Across Recurrent Depths in Looped Language Models](https://arxiv.org/abs/2610.10625) | 循环语言模型在不同递归深度上的安全性并不一致——更深的推理深度更容易遭受越狱攻击，且同一攻击可跨深度迁移，而SFT与偏好对齐均无法消除这种跨深度安全差距。 |
| [^230] | [Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models](https://arxiv.org/abs/2610.10623) | LoopOPD让循环语言模型利用自身更深层循环计算作为冻结“教师”，在学生自生成轨迹上进行在线策略蒸馏，无需外部教师或特权信息即可获得密集监督实现自我提升，D-LoopOPD进一步将该过程动态化。 |
| [^231] | [Constitution-Guided Watermarking](https://arxiv.org/abs/2610.09552) | 本文提出“宪法引导水印”框架，通过将提供商需求表示为自然语言原则，使水印系统能够根据不同请求的需求灵活选择属性权衡，避免了传统方法对所有请求采用统一配置所导致的牺牲问题。 |
| [^232] | [VETTA: Coordinating Turn- and Token-Level Credit Assignment for Multi-Turn LLM Agents](https://arxiv.org/abs/2610.08402) | VETTA通过共享轻量级评论家上的独立价值头联合学习轮次级与令牌级价值，在单次策略更新中协调两个层面的信用分配，从而解决多轮LLM智能体面临稀疏反馈时的信用分配难题。 |
| [^233] | [Revisiting Temporal Regularization for Smooth Control in Deep Reinforcement Learning](https://arxiv.org/abs/2610.07910) | 本文证明了时间正则化能约束共享相同下一状态的邻近状态之间的动作差异，从而在观测噪声下提供空间平滑性，并据此提出了结合线性斜坡上升时间惩罚的CATS方法，在不损害任务性能的前提下实现机器人的平滑控制。 |
| [^234] | [SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics](https://arxiv.org/abs/2610.06513) | 提出SOL——一种基于固定Transformer隐藏状态经验测度的双切片Wasserstein距离的文本分布距离度量，当Transformer为单射时可证明其为真正的度量，为非自回归语言模型的分布拟合评估提供了稳定的样本级评估方案。 |
| [^235] | [$\tilde{O}(\sqrt{T})$ Regret and Polylogarithmic Constraint Violation for COCO](https://arxiv.org/abs/2610.03983) | 本文针对对抗性约束在线凸优化（COCO）问题，提出了一种在线策略，在保持近最优 $O(\sqrt{T\log T})$ 遗憾的同时，将累计约束违反从 $\widetilde O(\sqrt{T})$ 显著降低到多对数级别的 $O(\log^2 T)$。 |
| [^236] | [Benchmarking Vision-Language Models on Synapse Detection and Proofreading in Connectomics](https://arxiv.org/abs/2609.36492) | 该研究首次系统性地基准测试了视觉语言模型在连接组学突触检测与校对任务上的表现，发现零样本性能接近随机，但通过少样本学习和LoRA微调，开源模型可达到专业模型水平，且在跨物种迁移中优于专业模型。 |
| [^237] | [Train4Merge: A Controlled Single-Teacher Study of RL vs. SFT Teachers for OPD-Based Model Merging](https://arxiv.org/abs/2609.32303) | 该研究通过受控单教师实验首次系统比较了强化学习（RL）与监督微调（SFT）两种训练算法在基于在线策略蒸馏的模型合并中的效果，发现在智能体、推理和感知三个领域中，RL训练的教师均能产生更强的学生模型，性能分别领先4.27、1.50和0.86个百分点。 |
| [^238] | [Bandits with Probing: Optimal Regret and the Limits of Winner Feedback](https://arxiv.org/abs/2609.15248) | 本文针对带探测的多臂老虎机问题首次确定了两个极小极大最优遗憾定律：在赢家反馈下，独立随机奖励情形的遗憾阶为 Φ_{n,k}(T)，而一般联合独立同分布奖励及固定序列情形的遗憾阶为 R_{n,k}(T)，并给出了匹配的任意时刻上界。 |
| [^239] | [Training-Free Task Vectors for LLM Behavioral Control](https://arxiv.org/abs/2609.09054) | 提出了一种无需微调的TFTVs方法，仅利用前向传播统计量将激活引导向量映射为秩一权重空间编辑，从而以低成本实现对大语言模型行为的放大、抑制与组合控制。 |
| [^240] | [Chimaera: A Mixture-of-Graph-Experts Architecture for Cross-Task and Cross-Dataset Graph Learning](https://arxiv.org/abs/2609.08709) | Chimaera 提出了一种将专家混合机制与图基础模型相结合的新架构，通过整合多种 GFM 架构和大语言模型生成的嵌入，实现了跨任务和跨数据集的图学习能力。 |
| [^241] | [Momentum in large-batch training: Polyak enlarges the critical batch size, Nesterov improves data efficiency](https://arxiv.org/abs/2609.02728) | 该论文在幂律核回归框架下证明，在单遍大批量训练中 Polyak 动量可将临界学习率随批大小线性放大（从而扩大临界批大小约 1/(1-ρ) 倍），而 Nesterov 动量的临界学习率以 $B^\beta$（β>1）的更快速度增长，从而显著提升数据效率，并给出了刻画完整风险动力学的标度律与三区制批大小相图。 |
| [^242] | [Verdict Instability of OOD Scores under Reference Resampling](https://arxiv.org/abs/2609.00691) | 本文提出“判定不稳定性”这一新概念，通过重采样参考集并用其闭式解（无需拟合参数）来度量OOD检测分数对参考集选择的敏感程度，并揭示远分布外查询的分数恰好落在最可复现的判定上。 |
| [^243] | [SAUSS: Stochastic Approximation with Unbiased Simulated Scores for Limited Dependent Variable Models](https://arxiv.org/abs/2608.25304) | 提出了一种基于无偏模拟分数的随机逼近方法（SAUSS），通过固定小批量和无偏分数估计，显著降低多项选择模型的计算成本并消除模拟偏差。 |
| [^244] | [When Does Self-Supervised Pretraining Help Tabular Models? A Study of Label Scarcity and Missing Data](https://arxiv.org/abs/2608.24381) | 本文通过14个任务实验发现，自监督预训练在表格数据上平均优于从头训练，但增益不稳定且不显著，且其对干净数据集更有效，而对高缺失度数据可能有害。 |
| [^245] | [Decoupling candidate dual AGN from chance superpositions in the GOTHIC survey via a deep-learning framework](https://arxiv.org/abs/2608.24164) | 本文通过YOLOv11定向边界框深度学习框架，有效从GOTHIC巡天的偶然叠加中区分出真实双活动星系核，实现了高精度检测。 |
| [^246] | [TreeWY: Speculative Verification for Gated DeltaNet Hybrids](https://arxiv.org/abs/2608.20961) | 本文提出TreeWY方法，通过树状WY变换消除推测解码中的状态快照，显著降低内存开销，从而支持高接受率的宽草稿树。 |
| [^247] | [Differentiating Through Dual Prices: End-to-End Policy Learning Under Capacity Constraints](https://arxiv.org/abs/2608.04669) | 该论文提出通过对偶价格进行端到端微分来联合训练结果模型与分配策略，实现了容量约束下从观测数据中学习资源分配策略，并给出了具有可证明容量可行性和次优性界的凸松弛形式。 |
| [^248] | [Discretization and Statistical Consistency of Functional Flow Matching](https://arxiv.org/abs/2608.04531) | 本文证明了在有限观测细化（允许非嵌套）且一致有界的条件下，泛函流匹配的速度目标依均方收敛，并量化了观测、学习与数值误差对生成分布的影响，为其函数空间模型的统计一致性提供了理论保证。 |
| [^249] | [FinVerse: Financial Time-Series Benchmark](https://arxiv.org/abs/2608.03259) | FinVerse是一个金融领域的时间序列预测基准，通过更现实的评估方式（如方向预测）来弥补传统误差指标在真实决策中的不足，包含大量金融时间序列数据。 |
| [^250] | [Learning Bidirectional Causal Interactions with Heteroscedastic Neural Networks](https://arxiv.org/abs/2607.22313) | 提出无需外部工具变量的异方差神经联立方程估计器 SEM-DNN，通过条件协方差对角化实现观测数据中双向同期因果交互的唯一识别与非线性联合估计。 |
| [^251] | [GeneSpeak-FP: Target and Compound Retrieval from Observed Cell-Level Perturbation Signatures](https://arxiv.org/abs/2607.17671) | 提出GeneSpeak-FP，一种基因标记Transformer模型，可从单细胞扰动特征中联合检索化合物及其靶点，其top-10化合物检索和靶点召回率分别达到随机排序期望值的约13倍和11倍。 |
| [^252] | [Distributed Sparse Interventions in Language Models](https://arxiv.org/abs/2607.07128) | 本文提出了分布式稀疏干预（DSI）方法，通过操纵分布于各层和注意力头中的稀疏个体神经元集合来引导语言模型行为，克服了传统密集线性干预同时改变数千个神经元的弊端，更精准地将模型表示与行为联系起来。 |
| [^253] | [Flow-ERD: Agent-type Aware Flow Matching With Entropy-Regularized Distillation for Diverse Traffic Simulation](https://arxiv.org/abs/2607.06957) | Flow-ERD通过智能体类型感知流匹配与熵正则化蒸馏两阶段框架，在多智能体交通仿真中联合优化真实性与多样性，并有效防止闭环推演中的模式坍缩问题。 |
| [^254] | [GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks](https://arxiv.org/abs/2607.05369) | 本文提出了Graph-as-Policy（GaP），一个多智能体自学习编码框架，它从模块化开源机器人技能库生成包含感知、规划和控制节点的有向计算图，并利用内部仿真环境进行排练，从而提升物体几何与姿态变化较大的变分自动化任务的可靠性。 |
| [^255] | [CodeJeNN: A simple C++ neural network generator for physics applications](https://arxiv.org/abs/2607.02746) | CodeJeNN能将训练好的Keras模型自动生成为无外部依赖的自包含C++推理代码，使神经网络可高效、无缝地集成到高性能物理模拟求解器中。 |
| [^256] | [How Should Transformers Encode Numeric Values in Electronic Health Records?](https://arxiv.org/abs/2607.01391) | 该论文系统比较了Transformer处理电子健康记录数据时离散、连续和混合三种数值编码策略，发现显式建模数值-概念交互的方法在精度敏感任务上最优，而投影前分箱的混合方法更为稳健通用，且最优分箱数与数据集规模呈幂律关系。 |
| [^257] | [Convergence of Large-Stepsize SGD for the Cross-entropy Loss via Stochastic Lyapunov Stability](https://arxiv.org/abs/2606.30930) | 本文借助随机李雅普诺夫稳定性理论证明，大步长SGD在多分类交叉熵损失下（含线性分类器和两层神经网络）虽会在“稳定边缘”振荡与稳定下降之间交替，但迭代点总能在固定次数内回归稳定，从而实现最佳迭代意义上的收敛。 |
| [^258] | [Approximating velocity fields with planted point attractors via Neural-ODE for classification purposes](https://arxiv.org/abs/2606.23550) | 该论文提出在神经常微分方程（Neural-ODE）中植入预设的平衡点作为类别指示器，利用速度场的通用逼近能力塑造动力学景观并形成吸引域，从而将每个输入有效地引导至其对应的目标类别，实现分类任务。 |
| [^259] | [AlphaMemo: Structured Search-Process Memory for Self-Evolving Alpha Mining Agents](https://arxiv.org/abs/2606.20625) | AlphaMemo通过从AST差异中提取编辑模式、置信度门控残差记忆和非对称否决控制构建结构化搜索过程记忆，实现了自进化的Alpha挖掘智能体，在中证500和标普500上显著提升了样本外表现和固定预算下的因子发现效率。 |
| [^260] | [Online Distributional Prediction via Latent Cluster Geometry Under Drift and Corruption](https://arxiv.org/abs/2606.18778) | 该论文提出一种在漂移与对抗污染环境下基于潜在聚类几何配置的在线分布预测方法，利用Gibbs拟后验平均与可逆跳转MCMC进行预测，并给出了将PAC-贝叶斯复杂度、污染敏感后验扰动与漂移效应分离的累积Wasserstein-1遗憾界。 |
| [^261] | [DeFAb: A Dataset for Grounded Scientific Discovery through Defeasible Abduction](https://arxiv.org/abs/2606.18557) | 该论文提出了基于基因本体构建的DeFAb数据集与符号验证框架，用于评估模型通过可废止溯因修订科学理论的能力，并揭示了现有修订测试无法检验例外泛化性的局限性。 |
| [^262] | [From Uncertain Judgments to Calibrated Rankings: Conformal Elo Estimation for LLM Evaluation](https://arxiv.org/abs/2606.13221) | 该论文提出保形Elo估计方法，通过将校准后的获胜概率而非硬标签传入Bradley-Terry过程来量化每场对战的胜负不确定性，并对人机Elo评分的残差差距应用分割保形预测，使LLM-as-a-judge得出的模型排名与人类评分仅相差17.9 Elo平均绝对误差，同时提供具有统计保证的预测区间。 |
| [^263] | [Neural networks become centroid classifiers when trained on heavily corrupted inputs](https://arxiv.org/abs/2606.11319) | 该研究发现神经网络对超过90%的输入损坏仍保持远高于随机水平的准确率，并通过均值场理论分析证明，在重度损坏情形下无限宽神经网络会退化为最近质心（原型）分类器。 |
| [^264] | [Joint user-item allocation for e-commerce marketing campaigns](https://arxiv.org/abs/2606.09623) | 该论文首次将电子商务营销活动中的商品选择与用户定向统一形式化为“自动定向”联合分配问题，突破了现有方法固定一方、单独优化另一方的局限，在多场互斥且预算受限的活动中实现用户与商品的联合优化分配。 |
| [^265] | [Driving Video Retrieval for Complex Queries with Structured Grounding](https://arxiv.org/abs/2606.09109) | 提出STRIVE-D数据校准检索框架，通过弱标注视频评估并自适应调整查询规则，将校准后的规则分数与视觉-语言及关键词检索信号融合，从而在驾驶视频中可靠检索加塞、急刹等复杂动态事件。 |
| [^266] | [Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces](https://arxiv.org/abs/2606.06840) | 本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。 |
| [^267] | [Transient Amplification in Cascaded Gradient Methods: Exact Minimax Laws and Tracking without Resonance](https://arxiv.org/abs/2606.04031) | 该论文针对级联梯度方法首次给出精确且与维度无关的最坏瞬态放大极小极大公式，证明最坏瞬态为 1 + c/(1-γ) 量级、仅凭特征值无法预测有限时间增益，并在鞅噪声下确立了三次逆间隙缩放的跟踪方差定律。 |
| [^268] | [Efficient Exploration for Iterative Nash Preference Optimization](https://arxiv.org/abs/2606.01382) | 论文提出探索式纳什偏好优化（ENPO），通过SFT型正则化与对抗性探索机制，解决了迭代NLHF中隐式探索不足导致的KL正则参数指数级依赖问题，为在线迭代纳什学习提供了理论保证。 |
| [^269] | [SoundnessBench: Can Your AI Scientist Really Tell Good Research Ideas from Bad Ones?](https://arxiv.org/abs/2605.30329) | 提出了包含1,099个真实研究提案的SoundnessBench基准，评估发现12个前沿大语言模型普遍存在乐观偏见（平均假阳性率74.0%），无法可靠地在投入资源前甄别研究想法的方法论缺陷。 |
| [^270] | [Instance-dependent Stochastic Lipschitz bandit](https://arxiv.org/abs/2605.29748) | 该论文提出了一种新的分析与算法，通过Lipschitz函数在其水平集上的次优性间隙积分来刻画遗憾，从而得到了真正依赖实例的遗憾界，能够适应水平集的局部增长结构而不仅仅是其渐近行为。 |
| [^271] | [Causal Representation Learning for Generalisable Recommendation](https://arxiv.org/abs/2605.27043) | 该论文提出一种基于因果表征学习的推荐方法，通过信息论解耦准则和可优化的变分下界，从处理表征中剔除受混淆因子影响的非因果信息，从而解决训练分布与服务分布不一致导致的推荐模型泛化难题。 |
| [^272] | [Efficient Benchmarking Is Just Feature Selection and Multiple Regression](https://arxiv.org/abs/2605.25773) | 该论文将高效基准测试重新构建为带特征选择的多元回归问题，通过结合核岭回归与mRMR特征选择算法，在预测误差和排序相关性上显著改进了现有高效基准测试方法。 |
| [^273] | [Approaching I/O-optimality for Approximate Attention](https://arxiv.org/abs/2605.23751) | 本文提出了受近似注意力方法启发的 I/O 高效算法，将注意力计算的 I/O 复杂度对上下文长度 n 的依赖从二次降低到几乎线性，并给出了几乎匹配的算法特定下界，证明了其接近最优性。 |
| [^274] | [Regret-Based $(\epsilon,\delta)$-optimal Stopping Criteria for Bayesian Optimization](https://arxiv.org/abs/2605.22561) | 本文为GP-UCB提出了可证明更紧的瞬时遗憾界，并据此设计了在终止时能以高概率$1-\delta$保证$\epsilon$-最优解的贝叶斯优化停止准则。 |
| [^275] | [Shaping the Prior: How Synthetic Task Distributions Determine Tabular Foundation Model Quality](https://arxiv.org/abs/2605.18971) | 提出O'Prior组合式真实感合成先验（包含分层SCM元生成器、真实感引擎、压力注入模块与防泄漏课程生成协议），在固定架构与算力的条件下证明合成任务分布的设计是决定表格基础模型质量的关键因素。 |
| [^276] | [Tabular Foundation Models for Credit Risk Prediction: A Dedicated Review and Empirical Comparison](https://arxiv.org/abs/2605.18147) | 本文系统综述并实证比较了表格基础模型在信用风险预测中的应用，指出其借助上下文学习利用大规模预训练先验知识，有望在小数据场景下超越梯度提升等传统模型并解决长期存在的挑战。 |
| [^277] | [Temporal Fair Division in Multi-Agent Systems: From Precise Alternation Metrics to Scalable Coordination Proxies](https://arxiv.org/abs/2605.14879) | 该论文提出了“轮转周期性”这一计算高效的公平性度量指标家族，通过等待时间规律性和访问频率均衡性两个互补维度，区分有序轮流与不规则访问模式，为多智能体系统中的时间公平分配提供了可扩展的协调评估方法。 |
| [^278] | [NodeSynth: Socially Aligned Synthetic Data for AI Evaluation](https://arxiv.org/abs/2605.14381) | NodeSyn通过将微调的分类体系生成器与可信文献来源和敏感属性相结合，生成反映真实社会背景的敏感领域评估数据，其诱发的模型失败率达到通用合成基线的3.1倍、真实人类查询的1.6倍，显著优于现有合成数据方法。 |
| [^279] | [Watermarking Should Be Treated as a Monitoring Primitive](https://arxiv.org/abs/2605.13095) | 论文主张水印不仅是防御移除和伪造的安全技术，更应被视为一种监控原语——同样的水印信号可用于将生成内容追溯至具体生成实体，即使是零比特水印也能被内部观察者用于个体归因，外部观察者在特定配置下也能识别内容来源。 |
| [^280] | [U-STS-LLM: A Unified Spatio-Temporal Steered Large Language Model for Traffic Prediction and Imputation](https://arxiv.org/abs/2605.11735) | 该论文提出U-STS-LLM统一框架，基于部分冻结的时空注意力LLM主干，利用动态时空注意力偏置生成器引导注意力，将蜂窝流量预测与缺失数据填补两大任务统一处理，解决了现有模型任务专用、时空建模解耦、级联误差累积以及过拟合等问题。 |
| [^281] | [Attractor-Vascular Coupling Theory: Formal Grounding and Empirical Validation for AAMI-Standard Cuffless Blood Pressure Estimation from Smartphone Photoplethysmography](https://arxiv.org/abs/2605.10871) | 该论文提出吸引子-血管耦合理论（AVCT），从PPG吸引子几何结构出发构建可解释的血压预测框架，实现了符合AAMI标准的智能手机无袖带血压估计，校准后收缩压/舒张压平均绝对误差仅2.05/1.67 mmHg。 |
| [^282] | [Generalized Wasserstein Flow Matching: Nested Couplings for Point Cloud Generation](https://arxiv.org/abs/2605.08424) | 该论文将流匹配推广到概率测度之上的概率测度（元测度）层面，证明只需可容许的嵌套外内耦合（无需最优传输计划或光滑流形）即可诱导有效的元测度流与非局部速度场，从而拓展了多样本流匹配的设计空间，并在点云生成中揭示了几何与成本的权衡。 |
| [^283] | [POETS: Uncertainty-Aware LLM Optimization via Compute-Efficient Policy Ensembles](https://arxiv.org/abs/2605.07775) | POETS利用KL正则化策略会隐式编码奖励函数这一洞察，直接训练共享预训练权重的策略集成来捕获认知不确定性，从而以计算高效的方式将不确定性量化与LLM策略优化相结合，绕过了传统上需要先训练不确定性感知奖励模型再拟合策略的复杂嵌套流程。 |
| [^284] | [Q-MMR: Off-Policy Evaluation via Recursive Reweighting and Moment Matching](https://arxiv.org/abs/2605.06474) | 提出 Q-MMR 框架，通过自上而下的递归重加权和矩匹配学习每个数据点的权重来实现离策略评估，并在仅假设 Q 函数可实现性的条件下获得与维度无关的有限样本保证。 |
| [^285] | [MoLS: SNR-Guided Module-Wise Learning-Rate Calibration for Large Language Models](https://arxiv.org/abs/2605.05794) | MoLS 通过估计各模块的梯度信噪比并采用逆平方根缩放自动确定相对学习率乘数，以一次性校准取代了大语言模型训练中昂贵的手动模块学习率调参。 |
| [^286] | [VLMShield: Efficient and Robust Defense of Vision-Language Models against Malicious Prompts](https://arxiv.org/abs/2604.06502) | 该论文提出MAFE多模态特征提取框架和轻量级安全检测器VLMShield，利用良性与恶意提示之间的分布差异，以即插即用方式高效且鲁棒地检测针对视觉语言模型的多模态恶意攻击。 |
| [^287] | [Learning to Predict, Discover, and Reason in High-Dimensional Event Sequences](https://arxiv.org/abs/2603.16313) | 本文提出将车辆诊断故障码序列视为一种“语言”，通过机器学习方法实现自动预测、错误模式发现与推理，以替代昂贵且易错的人工规则定义过程。 |
| [^288] | [On the Complexity of Best-Arm Identification in Non-Stationary Linear Bandits](https://arxiv.org/abs/2603.10346) | 本文研究了非平稳线性老虎机中固定预算最优臂识别的复杂度，指出基于G-最优设计的经典复杂度度量过于悲观，因为其下界仅建立在标准基向量构成的臂集合上，掩盖了更丰富的臂集合结构可能带来的优势。 |
| [^289] | [MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios](https://arxiv.org/abs/2603.09983) | 提出MoE-SpAc框架，将投机解码创新性地重新用作内存管理的前瞻传感器，通过投机效用估计器、异构工作负载均衡器和异步执行引擎，在统一的效用空间中协调专家预取与逐出，使异构边缘设备上的MoE推理TPS较SOTA基线提升42%。 |
| [^290] | [Heterogeneous Agent Collaborative Reinforcement Learning](https://arxiv.org/abs/2603.02604) | 提出新的RLVR问题HACRL及协同算法HACPO，使异构智能体在训练中共享经验证的轨迹以实现双向互学习、推理时独立运行，从而最大化样本利用率与跨智能体知识传递。 |
| [^291] | [How do standard time series forecasting benchmarks translate to finance? A broad study of risk adjusted performance](https://arxiv.org/abs/2603.01820) | 该研究在15年期货数据上通过波动率目标夏普比率优化任务对16种深度学习预测架构进行基准测试，发现通用Transformer和SSM等主流“最先进”模型表现不及简单的AR(1)基准，而基于循环结构的简单混合模型（变量选择网络+LSTM夏普比率最高、xLSTM对交易成本最鲁棒）表现最佳，表明标准时间序列基准上的成功难以直接迁移到金融预测任务。 |
| [^292] | [Half-Truths Break Similarity-Based Retrieval](https://arxiv.org/abs/2602.23906) | 本文发现CLIP模型会被“半真半假”的错误描述欺骗（相似度反而升高），并提出CS-CLIP方法，通过将描述分解为实体和关系单元并构建最小编辑的干扰样本进行对比微调，从而增强模型对细节错误的识别能力。 |
| [^293] | [Isotropic Activation Functions Enable Deindividuated Neurons and Adaptive Topologies](https://arxiv.org/abs/2602.23405) | 该论文提出围绕选定等变性设计的各向同性激活函数，通过仿射映射的奇异值分解将网络层对角化，使网络能够以精确或渐近保持函数的方式自适应地增删神经元，从而实现可变拓扑结构。 |
| [^294] | [Under the Influence: Evaluating and Improving LLM Vigilance in Agentic Tasks](https://arxiv.org/abs/2602.21262) | 该论文针对大语言模型智能体容易被外界影响的问题，提出在现实智能体任务中评估并提升其“警觉性”，即辨别何时应采纳善意建议、何时应拒绝恶意建议的能力。 |
| [^295] | [Why Does Counterintuitive Likelihood Behavior Rarely Emerge in Tabular Anomaly Detection?](https://arxiv.org/abs/2602.09593) | 本文提出了一个领域无关的反直觉似然现象形式化定义，并通过大规模实验与理论分析证明：与图像领域不同，反直觉的似然行为在表格异常检测中很少出现，且标准化流似然可实现有竞争力的异常检测性能。 |
| [^296] | [Mitigating the Likelihood Paradox in Flow-based OOD Detection: An Entropic Perspective](https://arxiv.org/abs/2602.09581) | 提出免训练的语义比例熵操控方法SPEM，从熵的视角通过对与分布内记忆库相似度低的输入施加更强扰动，增大对数似然差的期望下界，从而缓解基于流的OOD检测中的似然悖论，使似然排序与人类直觉对齐。 |
| [^297] | [StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors](https://arxiv.org/abs/2602.08934) | 本文提出StealthRL强化学习改写攻击框架，在保持语义保真度和文本可读性的前提下，将多种AI文本检测器的平均TPR@1%FPR降至0.024，且攻击可迁移至训练中未见的检测器，揭示了AI文本检测器普遍存在的鲁棒性缺陷。 |
| [^298] | [Clifford Kolmogorov-Arnold Networks](https://arxiv.org/abs/2602.05977) | 提出了克利福德柯尔莫哥洛夫-阿诺德网络（ClKAN），通过随机化拟蒙特卡罗网格生成方法解决高维代数的指数级扩展问题，并引入新的批归一化策略处理可变域输入，实现了在任意克利福德代数空间中灵活高效的函数逼近。 |
| [^299] | [Feature-Space Smoothing: Certified Robustness of Deep Representations](https://arxiv.org/abs/2601.16200) | 该论文提出基于高斯随机平滑的特征空间平滑（FS）框架，在表示层面为深度模型提供干净与对抗特征间余弦相似度的认证下界，并引入即插即用的高斯平滑增强器（GSB）来提升编码器的高斯鲁棒性分数。 |
| [^300] | [On the Runway Cascade of Transformers for Language Modeling](https://arxiv.org/abs/2601.14522) | 提出并形式化“跑道级联”现象，解释因果Transformer中直接路径与间接路径（跑道）之间的信息传播失调如何导致冗余信息污染标记表示，并设计“跑道感知重连”机制来显式缓解这一问题。 |
| [^301] | [Attention-space Contrastive Guidance for Efficient Hallucination Mitigation in LVLMs](https://arxiv.org/abs/2601.13707) | 提出注意力空间对比引导（ACG），一种无需训练的单次前向传播方法，通过在自注意力层构建图像条件与纯文本对比注意力路径并结合轻量级正交投影，高效缓解大型视觉-语言模型的幻觉问题。 |
| [^302] | [The Role of Quantum in Hybrid Quantum-Classical Neural Networks: An Empirical Assessment](https://arxiv.org/abs/2601.04732) | 本文通过理想无噪声模拟下的严谨统计研究，在三个医学二分类任务上系统评估了常用混合量子-经典神经网络架构，以揭示量子组件对模型整体性能的实际贡献。 |
| [^303] | [Dementia-R1: Reinforced Pretraining and Reasoning from Unstructured Clinical Notes for Real-World Dementia Prognosis](https://arxiv.org/abs/2601.03018) | 提出了Dementia-R1框架，通过冷启动强化学习策略先预训练模型预测可验证的临床指标，再基于非结构化临床笔记进行痴呆症的纵向预后预测，实现了最佳整体性能。 |
| [^304] | [Multi-Task Learning for Metal Alloy Property Prediction: An Empirical Study of Negative Transfer and Mitigation Strategies](https://arxiv.org/abs/2512.22740) | 本研究基于5.4万样本的金属合金数据集实证揭示了多任务学习中因任务函数形式不匹配导致的梯度错位负迁移现象——它损害电阻率和硬度的回归性能却提升非晶形成能力的分类召回率，并提出PCGrad以及标签分布平滑结合梯度归一化等有效缓解策略。 |
| [^305] | [MedInsightBench: Evaluating Medical Analytics Agents Through Multi-Step Insight Discovery in Multimodal Medical Data](https://arxiv.org/abs/2512.13297) | 提出首个用于评估大型多模态模型和智能体框架在多模态医学图像数据上进行多步骤医学洞察发现能力的基准MedInsightBench，包含332个精心标注的医学案例，并揭示现有模型表现仍然有限。 |
| [^306] | [Demo: Generative AI helps Radiotherapy Planning with User Preference](https://arxiv.org/abs/2512.08996) | 本文提出一种仅依据用户自定义偏好即可预测三维剂量分布的生成式模型，使计划制定者能够个性化权衡危及器官与靶区之间的取舍，并在适应性上超越Varian RapidPlan。 |
| [^307] | [Softly Symbolifying Kolmogorov-Arnold Networks](https://arxiv.org/abs/2512.07875) | 提出软符号化的Kolmogorov-Arnold网络（S2KAN），通过将符号基元直接融入训练，利用可微分的门控稀疏化机制与最小描述长度目标，在保留稠密样条灵活逼近能力的同时发现可解释的激活函数。 |
| [^308] | [Iterative GRPO: Batch-Online Policy Iteration for Multi-Turn RL via Single-Turn RLHF](https://arxiv.org/abs/2511.21638) | 提出Iterative GRPO方法，利用批在线部署模式收集的交互数据，将经典的近似策略迭代算法与单轮RLHF相结合，从而无需在训练循环中接入真实或模拟用户即可实现多轮强化学习。 |
| [^309] | [BrainRotViT: Transformer-ResNet Hybrid for Explainable Modeling of Brain Aging from 3D sMRI](https://arxiv.org/abs/2511.15188) | 提出了一种两阶段2.5D混合框架BrainRotViT，将预训练视觉Transformer与轻量级残差CNN相结合，从结构MRI中实现准确且可解释的脑年龄估计，验证集平均绝对误差仅3.43年。 |
| [^310] | [Clip-Low Increases Entropy and Clip-High Decreases Entropy in Reinforcement Learning of Large Language Models](https://arxiv.org/abs/2509.26114) | 该研究揭示了PPO/GRPO中的裁剪机制是RLVR中一个被忽视的混杂因素——clip-low会增加熵而clip-high会降低熵，且在标准参数下clip-high占主导地位，导致即使奖励完全随机，模型熵也会下降并引发熵坍缩。 |
| [^311] | [Dynamic Relational Priming Improves Transformer in Multivariate Time Series](https://arxiv.org/abs/2509.12196) | 本文提出动态关系启动注意力机制（prime attention），通过为每个token对交互定制动态表示，显著改进了Transformer在多元时间序列建模中捕捉异构通道间依赖关系的能力。 |
| [^312] | [AU-Harness: An Open-Source Toolkit for Efficient and Unified Evaluation of AUdio-LLMs](https://arxiv.org/abs/2509.08031) | AU-Harness是一个开源的音频大语言模型统一评估框架，通过高效流水线将评估处理时间减少77%，并解决了现有工具在多轮对话支持不足和框架缺乏标准化方面的关键局限。 |
| [^313] | [From Dueling Bandits to Episodic MDPs: Learning with Adversarial Preference Feedback](https://arxiv.org/abs/2507.11706) | 本文将决斗老虎机扩展到偏好反馈会影响状态转移的情景MDP设定，在对抗性偏好变化下建立了极小极大遗憾界 $\tilde{\Theta}(\min\{HT,(H^3SK)^{1/3}T^{2/3}\})$，并给出了达到该界的算法。 |
| [^314] | [Deep Learning Warm Starts for Trajectory Optimization on the International Space Station](https://arxiv.org/abs/2505.05588) | 本研究在国际空间站上首次实现在轨演示，利用神经网络热启动加速Astrobee自由飞行机器人的轨迹优化，通过学习顺序凸规划问题的结构来实现满足安全约束的星载实时轨迹生成。 |
| [^315] | [Catastrophic Overfitting, Entropy Gap and Participation Ratio: A Noiseless $l^p$ Norm Solution for Fast Adversarial Training](https://arxiv.org/abs/2505.02360) | 本文提出一种纯粹通过控制 $l^p$ 训练范数来缓解快速对抗训练中灾难性过拟合的无噪声新方案，并揭示灾难性过拟合源于高度集中的梯度与激进范数约束的相互作用。 |
| [^316] | [Quantum-aware Transformer model for state classification](https://arxiv.org/abs/2502.21055) | 提出了一种量子感知的Transformer模型，通过对量子态厄米矩阵表示进行掩码式无监督预训练来学习量子密度矩阵的结构特性，从而实现两体量子态（涵盖可分离态、Werner态、一般纠缠态和最大纠缠态）的数据驱动纠缠分类。 |
| [^317] | [Stochastic Inverse Problem: stability, regularization and Wasserstein gradient flow](https://arxiv.org/abs/2410.00229) | 本文将逆问题推广到随机情形，在概率空间中借助测度输运理论研究了直接反演、带正则化的变分方法以及Wasserstein梯度流优化，并揭示了度量选择对稳定性和解的性质的关键影响。 |
| [^318] | [Quantized Approximately Orthogonal Recurrent Neural Networks](https://arxiv.org/abs/2402.04012) | 本文提出了量化近似正交循环神经网络（QORNNs）来解决正交循环神经网络（ORNNs）中参数过多的问题，采用一种后训练量化策略和三种融入正交约束和量化权重的量化感知训练算法，取得了与s相似的结果。 |
| [^319] | [FC-PINO: High Precision Physics-Informed Neural Operators via Fourier Continuation](https://arxiv.org/abs/2211.15960) | FC-PINO 通过傅里叶延拓技术克服了谱微分固有的周期性假设限制，将物理信息神经算子的高精度、高效率求解能力扩展到了非周期性偏微分方程问题。 |
| [^320] | [Maximum Likelihood on the Joint (Data, Condition) Distribution for Solving Ill-Posed Problems with Conditional Flow Models](https://arxiv.org/abs/2208.11782) | 提出了一种计算低成本的最大似然代理方法来训练条件流模型，无需显式条件分布、辅助网络或额外损失项，即可自然地求解一类不适定逆问题。 |
| [^321] | [Learning from Hypervectors: A Survey on Hypervector Encoding.](http://arxiv.org/abs/2308.00685) | 本调查研究聚焦于超维计算系统的输入和超维向量的生成，探讨了各种方法的限制、挑战和潜在好处。 |

# 详细

[^1]: RobustRLlib：鲁棒强化学习算法的统一库与基准测试

    RobustRLlib: A Unified Library and Benchmark for Robust Reinforcement Learning Algorithms

    [https://arxiv.org/abs/2610.32971](https://arxiv.org/abs/2610.32971)

    本文提出RobustRLlib，一个集成了17种鲁棒在线、离线与安全RL算法的统一库与基准测试平台，并通过273种扰动配置的系统评估绘制了当前鲁棒RL算法的能力图谱，揭示了其有效范围与鲁棒性差距。

    

    鲁棒强化学习（RL）对于可靠的真实世界部署至关重要，因为策略可能在多种偏移和扰动下失效。现有的鲁棒RL算法提供了丰富的机制和原理，但两大障碍限制了其更广泛的应用潜力：算法可复用性有限，以及实证能力边界不清晰。我们提出了RobustRLlib，一个以算法为中心的库和基准测试平台，以应对这两个挑战。它集成了17种代表性的鲁棒在线、离线和安全RL算法，并采用统一接口以显著提升可复用性，同时提供了一个增强的评估平台，涵盖多样化的真实世界偏移以及先进的人形机器人和视觉-语言-动作任务。通过对273种扰动配置的系统评估，本工作提供了当前鲁棒RL算法的最新能力图谱，揭示了这些算法在哪些场景下仍然有效，以及在哪些方面仍存在显著的鲁棒性差距。

    arXiv:2610.32971v1 Announce Type: cross  Abstract: Robust reinforcement learning (RL) is critical for reliable real-world deployment, where policies may fail under diverse shifts and disruptions. Existing robust RL algorithms offer a rich set of mechanisms and principles, yet two major barriers limit their broader potential: limited algorithm reusability and an unclear empirical capability boundary. We introduce RobustRLlib, an algorithm-centric library and benchmark that addresses both challenges. It integrates 17 representative robust online, offline, and safe RL algorithms under a unified interface for substantially improved reusability, together with an enhanced evaluation platform spanning diverse real-world shifts and advanced humanoid and vision-language-action tasks. Systematic evaluation over 273 perturbation configurations provides an up-to-date capability map of current robust RL algorithms, revealing where they remain effective and where substantial robustness gaps persist.
    
[^2]: DexPort：一种用于灵巧手数据采集的手持式技能接口

    DexPort: A Hand-Held Skill Interface for Dexterous Hand Data Collection

    [https://arxiv.org/abs/2610.32969](https://arxiv.org/abs/2610.32969)

    DexPort提出了一种面向高自由度灵巧手的便携式手持数据采集接口，通过采集与部署共用相同手部硬件来避免实体差异，并借助一小组可选择的技能来协调手指运动，从而大幅降低高质量灵巧操作演示数据的采集成本。

    

    收集高质量的演示数据仍然是学习灵巧操作的核心瓶颈。遥操作能够记录可执行的机器人动作，但它需要完整的机器人设备，并把协调众多手指关节的负担压在操作者身上。人类演示更容易收集，但其动作必须经过估计并重定向到机器人，而且不能直接提供机器人动作。在本文中，我们提出了DexPort，一个面向高自由度机器人手的便携式数据采集接口。它在数据采集与部署时使用相同的手部硬件，从而避免了手部实体差异（embodiment gap），并同时记录指令关节目标和测量关节状态。我们的关键观察是：一小组技能就可以支持广泛的灵巧任务。在采集过程中，操作者移动手持接口来指定机器人手腕运动，并选择或调整协调手指运动的技能。（原文摘要在此处截断）

    arXiv:2610.32969v1 Announce Type: cross  Abstract: Collecting high-quality demonstrations remains a central bottleneck for learning dexterous manipulation. Teleoperation records executable robot actions, but it requires a complete robot setup and places the burden of coordinating many finger joints on the operator. Human demonstrations are easier to collect, but their motions must be estimated and retargeted to the robot, and they do not directly provide robot actions. In this paper, we introduce DexPort, a portable data-collection interface for high-DoF robot hands. It uses the same hand hardware for collection and deployment, avoiding a hand embodiment gap, and records both commanded joint targets and measured joint states. Our key observation is that a small set of skills can support a wide range of dexterous tasks. During collection, the operator moves the hand-held interface to specify the robot wrist motion and selects or adjusts skills that coordinate the finger motion. This rem
    
[^3]: Flownomaly：面向小样本异常检测的单步条件均值恢复

    Flownomaly: Single-Step Conditional-Mean Recovery for Few-Shot Anomaly Detection

    [https://arxiv.org/abs/2610.32964](https://arxiv.org/abs/2610.32964)

    Flownomaly将小样本工业异常检测建模为冻结DINOv2特征空间中的“正常对应物恢复”问题，通过条件流匹配学习速度场，仅需单步评估即可恢复干净正常特征的后验均值，并利用仅基于正常样本的Jacobian增益-噪声准则自动确定运行时间，无需异常验证样本。

    

    小样本工业异常检测需要仅从少量无缺陷图像中建模类别特定的正常性。常见方法要么将推理与存储的示例绑定，要么施加刚性的线性结构，要么存在重现异常内容的风险。Flownomaly则将该问题转化为在冻结的DINOv2特征空间中进行正常对应物的恢复。给定正常支持样本，通过对冻结的patch token进行条件流匹配来学习一个速度场，将高斯噪声传输至正常特征。沿直线路径，仅需对最优场进行一次评估的单个恢复步骤，即可得到干净正常token的后验均值，无需轨迹积分。运行时间决定了恢复对查询token的修改强度。在没有异常验证样本可用于调参的情况下，一种仅依赖正常样本的Jacobian增益-噪声准则即可仅从支持样本中选出该时间。查询token与……

    arXiv:2610.32964v1 Announce Type: cross  Abstract: Few-shot industrial anomaly detection requires modeling category-specific normality from only a handful of defect-free images. Common approaches tie inference to stored exemplars, impose rigid linear structure, or risk reproducing anomalous content. Flownomaly instead casts the problem as normal-counterpart recovery in frozen DINOv2 feature space. Given normal supports, a velocity field is learned through conditional flow matching over frozen patch tokens to transport Gaussian noise toward normal features. Along straight paths, a single recovery step using one evaluation of the optimal field yields the posterior mean of the clean normal token without trajectory integration. The operating time governs how strongly recovery modifies a query token. With no anomalous validation samples available for tuning, a normal-only Jacobian gain-to-noise criterion selects this time from the supports alone. Squared residuals between query tokens and t
    
[^4]: 先测量再训练：类型化决策模型 vs. 经典基线 vs. 生成式大语言模型在个人多语言收件箱路由中的应用

    Measure Before You Train: Typed Decision Models vs. a Classical Baseline vs. a Generative LLM for Routing a Personal Multilingual Inbox

    [https://arxiv.org/abs/2610.32955](https://arxiv.org/abs/2610.32955)

    该论文提出自托管个人邮件代理Foyer及其测量协议，在真实多语言邮件数据上于训练前建立人工真实标注基准，并系统比较了类型化决策模型、经典机器学习基线与生成式大语言模型等九个系统的表现。

    

    个人邮件代理需要为每封邮件做出决策：这封邮件属于什么类型、是否紧急、是否需要起草回复、创建任务，还是直接忽略。然而这些系统的错误率很少被真正测量，因为存储的标签是由系统自身生成的。我们提出了Foyer——一个正在日常使用的自托管个人收件箱代理，以及一套测量协议（分层采样、冻结快照、按邮件线程分组的划分、预注册假设、在代理审查循环中由人工标注），该协议能够在任何训练开始之前获得真实标注数据。在739封由所有者亲自标注的真实意大利语/英语邮件上（其中231封为无偏样本，500封为对抗性采样），我们在按线程分组的交叉验证下比较了九个系统：生产环境的关键词规则、TF-IDF+逻辑回归经典基线、本地类型化决策模型、托管决策模型，以及托管生成式大语言模型。论文给出了四项主要结果，其中生产环境规则在邮件类型识别上仅获得28%的得分，而对照系统达到了58%。

    arXiv:2610.32955v1 Announce Type: new  Abstract: Personal email agents decide, for every message, what it is, whether it is urgent, whether to draft a reply, create a task, or ignore it. Their error rate is rarely measured: the stored labels were written by the system itself. We present Foyer, a self-hosted personal inbox agent in daily use, and a measurement protocol (stratified sampling, frozen snapshots, thread-grouped splits, pre-registered hypotheses, human labels in the agent's review loop) that yields ground truth before any training. On 739 real Italian/English emails labelled by the owner (231 unbiased, 500 adversarially sampled), we compare nine systems under thread-grouped cross-validation: the production keyword rules, a TF-IDF + logistic-regression baseline, local typed decision models (Tev1, Nimble), a hosted decision model (Nimble v4-max), and hosted generative LLMs (Qwen2.5-7B, Gemma 4 26B/31B). Four results. The production rules score 28% on message type against a 58% 
    
[^5]: 舍入发生的位置很重要：舍入位置对齐可稳定大语言模型强化学习

    Where Rounding Happens Matters: Rounding-Location Alignment Stabilizes LLM Reinforcement Learning

    [https://arxiv.org/abs/2610.32951](https://arxiv.org/abs/2610.32951)

    该论文发现，仅需将训练侧RoPE、SwiGLU和残差RMSNorm三个区域的舍入位置与推理侧对齐（将选定中间结果以FP32保留至BF16输出），即可在保留其他实现差异的情况下稳定BF16强化学习并防止训练崩溃。

    

    大语言模型的强化学习通常使用相互独立的推理和训练系统。即使权重和BF16张量完全相同，实现上的差异也可能使预期的同策略更新变为离策略，从而导致训练不稳定。现有的解决方法要么对齐整个推理和训练技术栈，要么改变全局精度，要么修改学习算法。我们提出这样一个问题：在保留其他实现差异的情况下，仅对齐执行栈中的选定部分能否稳定BF16强化学习？我们在三个区域——RoPE、SwiGLU和残差RMSNorm——将训练侧的舍入位置与推理侧对齐，方法是将选定的中间结果以FP32精度保留，直到产生其BF16输出。在四个Qwen系列模型上，对齐全部三个区域可以防止默认BF16基线下观察到的训练崩溃；在所研究的实验中，仅对齐RoPE路径也足以实现稳定，尽管仍存在其他数值差异。这些干预措施保持推理引擎不变

    arXiv:2610.32951v1 Announce Type: cross  Abstract: Reinforcement learning for large language models often uses separate systems for inference and training. Even with identical weights and BF16 tensors, implementation differences can make an intended on-policy update off-policy and destabilize training. Existing remedies align the entire inference and training stacks, change global precision, or modify the learning algorithm. We ask whether aligning selected parts of the execution stack can stabilize BF16 RL while other implementation differences remain. We align training-side rounding locations with inference in three regions---RoPE, SwiGLU, and residual RMSNorm---by retaining selected intermediates in FP32 until their BF16 outputs. Across four Qwen-family models, aligning all three regions prevents the collapse observed under the default BF16 baseline; RoPE-path alignment alone also suffices in the studied runs, despite remaining numerical discrepancies. The interventions keep the inf
    
[^6]: 面向并行生成式推荐的物品空间与代码空间知识蒸馏

    Knowledge Distillation across Item and Code Spaces for Parallel Generative Recommendation

    [https://arxiv.org/abs/2610.32943](https://arxiv.org/abs/2610.32943)

    该论文提出面向并行语义ID推荐器的两个互补知识蒸馏目标（如物品分布匹配IDM），将序列推荐器的物品级预测分布知识跨空间迁移到代码空间，弥补了硬标签监督缺乏备选物品分布信息的不足。

    

    基于语义ID的生成式推荐将每个物品表示为从其语义表示中导出的离散代码序列，并通过预测这些代码来生成推荐。现有的语义ID推荐器通常使用观测到的下一个物品的硬标签进行训练。尽管这种监督方式能够标识目标物品，但它无法提供在同一用户历史条件下其他备选物品的显式分布信息。序列推荐器可以提供这种物品级别的预测知识，但由于教师模型和学生模型分别工作在物品空间和代码空间，将其迁移到语义ID生成中具有挑战性。我们为并行语义ID推荐器提出了两个互补的知识蒸馏目标，这类推荐器无需自回归前缀条件即可同时预测所有代码位置。物品分布匹配（IDM）从编目层面重构物品分布……（原文摘要在此处截断）

    arXiv:2610.32943v1 Announce Type: new  Abstract: Generative recommendation with semantic IDs represents each item as a sequence of discrete codes derived from its semantic representation and predicts these codes to generate recommendations. Existing semantic-ID recommenders are typically trained with hard labels for the observed next item. Although such supervision identifies the target, it provides no explicit distributional information about alternative items conditioned on the same user history. Sequential recommenders can supply this item-level predictive knowledge, but transferring it to semantic-ID generation is challenging because the teacher and student operate in item and code spaces, respectively. We propose two complementary knowledge-distillation objectives for parallel semantic-ID recommenders, which predict all code positions simultaneously without autoregressive prefix conditioning. Item Distribution Matching (IDM) reconstructs a catalogue-level item distribution from th
    
[^7]: 如果第一次尝试失败，就回溯：针对冻结视觉-语言-动作模型的轻量级恢复机制

    If at First You Don't Succeed, Backtrack: Lightweight Recovery for Frozen VLAs

    [https://arxiv.org/abs/2610.32941](https://arxiv.org/abs/2610.32941)

    提出Backtrack，一种针对冻结VLA模型的轻量级失败检测与恢复机制：利用小型预测头读取模型内部表示来预测失败，并通过回溯机械臂动作让原始策略从更早状态重试，无需恢复演示、重新训练或独立模型，仅增加不到1%的参数量。

    

    视觉-语言-动作（VLA）模型在机器人操控任务上表现良好，然而一旦其自身动作将机器人带入训练数据几乎未覆盖的状态，模型就会变得脆弱。从此类失败中恢复通常需要恢复演示数据、策略重新训练，或在推理时使用独立的模型。我们提出了Backtrack，这是一种针对冻结VLA模型的轻量级失败检测与恢复机制，无需上述任何条件。一个小型失败预测头仅通过冻结策略的结果标签轨迹数据进行训练，读取VLA现有的内部表示来估计当前回合是否会失败。当预测的失败概率超过阈值时，控制器会暂停策略，回溯机械臂最近完成的动作，然后将控制权交还给未做任何改变的策略，使其能够从较早的配置重新尝试。该预测头仅为策略增加了0.48-0.88%的参数量。我们在LIBERO上对Backtrack进行了评估……

    arXiv:2610.32941v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) models perform well on robotic manipulation, yet remain brittle once their own actions carry them into states their training data barely covers. Recovering from such failures usually requires recovery demonstrations, policy retraining, or a separate model at inference time. We introduce Backtrack, a lightweight failure detection and recovery mechanism for frozen VLAs that needs none of these. A small failure prediction head, trained only on outcome-labeled rollouts of the frozen policy, reads the VLA's existing internal representations to estimate whether the current episode will fail. When the predicted failure probability exceeds a threshold, a controller pauses the policy, retraces the arm's recently achieved motion, and hands control back to the unchanged policy so it can retry from an earlier configuration. The head adds only 0.48-0.88% to the policy's parameter count. We evaluate Backtrack on LIBERO w
    
[^8]: 在全多项式时间内学习ReLU的正线性组合

    Learning Positive Linear Combinations of ReLUs in Fully Polynomial Time

    [https://arxiv.org/abs/2610.32940](https://arxiv.org/abs/2610.32940)

    该论文提出了首个在全多项式时间和样本复杂度内、于高斯分布下以L2损失PAC学习k个任意ReLU激活正线性组合的算法，其核心创新是一种以列表可解码方式将隐藏权重向量恢复到常数误差的新型弱学习器提升过程。

    

    我们研究了在 $\mathbb{R}^d$ 上、关于高斯分布、以 $L_2$ 损失进行 PAC 学习的这样一个基础问题：学习 $k$ 个任意 ReLU 激活函数的正线性组合。我们给出了针对该学习任务的第一个全多项式时间算法。具体而言，我们的算法抽取 $\mathrm{poly}(d, k, 1/\epsilon)$ 个样本，以 $\mathrm{poly}(d, k, 1/\epsilon)$ 的时间运行，并以高概率输出一个与目标概念 $\epsilon$-接近的假设。此前所有针对该问题的算法要么需要关于 $k$ 或 $1/\epsilon$ 的超多项式时间，要么需要对权重矩阵施加较强的良态假设。在技术层面，我们的方法开发了一种新颖的“弱”学习器提升过程，该过程能够以列表可解码的方式将隐藏的权重向量恢复到常数误差。为了开发这种“弱”学习过程，我们利用隐式张量计算技术来构建低度……（摘要在此处截断）

    arXiv:2610.32940v1 Announce Type: cross  Abstract: We study the fundamental problem of PAC learning positive linear combinations of $k$ arbitrary ReLU activations on $\mathbb{R}^d$ with respect to the Gaussian distribution, in $L_2$ loss. We give the first fully-polynomial time algorithm for this learning task. Specifically, our algorithm draws $\mathrm{poly}(d, k, 1/\epsilon)$ samples, runs in $\mathrm{poly}(d, k, 1/\epsilon)$ time, and outputs a hypothesis that is $\epsilon$-close to the target concept with high probability. All prior algorithms for this problem require super-polynomial time in $k$ or $1/\epsilon$, or leverage strong well-conditioned assumptions on the weight matrix.   On a technical level, our approach develops a novel boosting procedure of a ``weak'' learner that recovers the hidden weight vectors to constant error, in a list-decodable manner. To develop this ``weak'' learning procedure, we leverage implicit tensor computation techniques to construct a low-degree f
    
[^9]: ZENO：基于Zakai方程神经算子的非线性滤波

    ZENO: Nonlinear filtering with Zakai Equation Neural Operators

    [https://arxiv.org/abs/2610.32926](https://arxiv.org/abs/2610.32926)

    ZENO通过基于Zakai方程的物理信息损失训练神经算子，实现了对非线性随机动力系统未归一化滤波分布的高效逼近，从而完成状态推断与参数学习。

    

    非线性滤波的状态推断与参数学习是科学机器学习中的核心计算，在工程和神经科学等领域有着广泛的应用。未归一化滤波分布由Zakai方程控制，这是一个随机偏微分方程（SPDE），它将前向Kolmogorov算子（即Fokker-Planck方程）与观测似然更新相结合。尽管解析求解这些高维SPDE通常是不可能的，数值求解也难以处理，但算子学习和物理信息机器学习的最新进展使其变得可行。利用这些进展，我们提出了Zakai方程神经算子（ZENO），该模型通过基于Zakai方程的物理信息损失来优化神经算子，以逼近潜在随机动力系统的精确未归一化滤波分布。

    arXiv:2610.32926v1 Announce Type: new  Abstract: State inference and parameter learning for nonlinear filters is a core computation for scientific machine learning with varied applications in engineering and neuroscience. The unnormalized filtering distribution is governed by the Zakai equation, a stochastic partial differential equation (SPDE) that combines the forward Kolmogorov operator (known as the Fokker-Planck equation) with observed likelihood updates. Although solving these high-dimensional SPDEs is often impossible analytically and intractable numerically, advances in operator learning and physics-informed machine learning have made it feasible. Harnessing these advances, we introduce the Zakai Equation Neural Operator (ZENO), a model that optimizes a neural operator against a Zakai Equation based physics-informed loss to approximate the exact unnormalized filtering distribution for latent stochastic dynamical systems. We use this ZENO-inferred posterior distribution in an ex
    
[^10]: 面向混合自然语言与结构化符号语料库的自适应频率阈值字节对编码

    Adaptive Frequency Threshold Byte Pair Encoding for Mixed Natural Language and Structured Symbolic Corpora

    [https://arxiv.org/abs/2610.32921](https://arxiv.org/abs/2610.32921)

    本文提出AdaBPE，一种基于自适应频率阈值的频繁模式挖掘分词方法，通过动态调整合并策略，更好地处理跨越空白的多词表达和LaTeX等结构化符号内容。

    

    现代语言模型依赖子词分词来定义输入处理的离散单元，通常采用字节对编码（BPE）等方法，将空白字符视为固定的语义边界。这种设计存在固有局限，因为许多有意义的概念——包括多词表达以及诸如LaTeX等形式语言的结构化环境——经常跨越空白字符。尽管近期的超词分词方法试图通过放宽空白约束来解决这一问题，但它们通常依赖贪心的频率驱动合并方式，未能考虑词元长度或资源不足的领域特定语言的具体需求。在本文中，我们提出了自适应频率阈值字节对编码，这是一种利用自适应频繁模式挖掘方法的新型分词方法。AdaBPE具有双重动态特性：它基于……的长度自适应调整最小频率阈值（摘要在此处截断）

    arXiv:2610.32921v1 Announce Type: new  Abstract: Modern language models rely on subwording tokenization to define the discrete units of input processing, typically using methods like Byte Pair Encoding (BPE) that treat whitespace as a fixed semantic boundary. This design is inherently limited, as many meaningful concepts---including multiword expressions and the structured environments of formal languages like LaTeX---frequently span across whitespaces. While recent superword tokenization methods attempt to address this by relaxing whitespace constraints, they often rely on greedy frequency-driven merging that fails to account for token length or the specific needs of underrepresented domain-specific languages. In this paper, we introduce Adaptive Frequency Threshold Byte Pair Encoding (AdaBPE), a new tokenization method that leverages an adaptive frequent-pattern mining approach. AdaBPE features a twofold dynamicity: it adapts minimum frequency thresholds based on both the length of a
    
[^11]: SemantEq：形式化命题的忠实双向等价

    SemantEq: Faithful Bidirectional Equivalence of Formal Statements

    [https://arxiv.org/abs/2610.32919](https://arxiv.org/abs/2610.32919)

    提出 SemantEq，一种更有原则的形式化命题等价检验方法，要求命题间只能通过一系列允许的逻辑变换步骤相互转化，从而避免因证明器过强导致逻辑等价检验被平凡通过的问题。

    

    命题自动形式化是将数学从自然语言翻译成 Lean 4 等证明助手语言的过程，如今模型已经可以大规模地完成这一任务。随着证明器变得越来越强，正确性的负担从证明转移到了命题本身，即形式化命题是否忠实地翻译了其源文本。给定一个可信的参考形式化，最自然的检验标准是逻辑等价，但逻辑等价证明的内容过多：对于库中已证明的任意两个命题 P 和 Q，P ↔ Q 都会平凡地成立，因为证明器可以分别证明 P 和 Q，并通过 P ↔ True ↔ Q 完成推导。证明器越强，通过检验的无关命题对就越多。早期的方法通过限制证明器的能力，并检查每个证明是否引用了另一个命题。我们提出 SemantEq，一种更有原则的等价关系，它要求每个命题只能通过一系列允许的步骤（如逻辑重排等）转化为另一个命题……

    arXiv:2610.32919v1 Announce Type: new  Abstract: Statement autoformalization translates mathematics from natural language into a proof assistant such as Lean 4, and models now do it at scale. As provers get stronger, the burden of correctness shifts from the proof to the statement, that is, to whether the formal statement faithfully translates its source. Given a trusted reference formalization, the natural test is logical equivalence, and it proves too much. $P \leftrightarrow Q$ holds trivially for any two statements $P$ and $Q$ that the library proves, because a prover can prove each separately and pass through $P \leftrightarrow \mathrm{True} \leftrightarrow Q$. The stronger the prover, the more unrelated pairs pass. Earlier methods restrict the prover and check that each proof names the other statement. We introduce SemantEq, a more principled relation that instead requires each statement to turn into the other only through a chain of allowed steps, such as logical rearrangement a
    
[^12]: 物理约束机器学习在混合建模、优化与控制中的教程

    A Tutorial on Physics Constrained Machine Learning for Hybrid Modeling, Optimization, and Control

    [https://arxiv.org/abs/2610.32914](https://arxiv.org/abs/2610.32914)

    本文提出了模块化物理约束机器学习框架KKT-HardNet的分步教程，该框架通过微分投影保证约束满足，并利用隐函数定理实现高效端到端训练，可在单一平台上完成混合建模、过程仿真、优化与模型预测控制。

    

    物理约束机器学习（PCML）将领域知识、第一性原理和物理约束集成到人工智能/机器学习模型中。在本工作中，我们提出了一个模块化PCML框架KKT-HardNet的分步教程，并说明如何在单一平台上将其用于混合建模、过程仿真、优化和模型预测控制。KKT-HardNet将神经网络的快速推理能力与通过微分投影保证满足代数约束相结合。它可以处理异构数据，并强制执行线性与非线性的等式和不等式约束。其模块化设计支持自定义神经网络主干以及用于PCML的用户自定义投影机制。为实现高效的端到端训练，KKT-HardNet利用隐函数定理来通过投影层进行反向传播，从而避免了计算代价高昂的自动微分。

    arXiv:2610.32914v1 Announce Type: new  Abstract: Physics-constrained machine learning (PCML) integrates domain knowledge, first-principles, and physical constraints into AI/ML models. In this work, we present a step-by-step tutorial of a modular PCML framework, KKT-HardNet, and illustrate how it could be used for hybrid modeling, process simulation, optimization and model predictive control in a single platform. KKT-HardNet combines fast inference capabilities of neural networks with guaranteed satisfaction of algebraic constraints through differential projection. It can handle heterogeneous data and enforce both linear and nonlinear equality and inequality constraints. The modular design supports custom neural-network backbones and user-defined projection mechanisms for PCML. To enable efficient end-to-end training, KKT-HardNet exploits implicit function theorem to backpropagate through the projection layer. This avoids computationally expensive automatic differentiation. This work al
    
[^13]: 用于通过老虎机与蒙特卡洛树搜索改进正则化强化学习估计的推荐者遗憾

    Recommender regret for improving regularised RL estimation with bandits and MCTS

    [https://arxiv.org/abs/2610.32910](https://arxiv.org/abs/2610.32910)

    该论文将基于老虎机的估计生成形式化为推荐过程，提出正则化感知的推荐者遗憾来评估估计质量并证明其下界，进而通过推荐者MCTS统一了基于老虎机和搜索的目标改进方法，在多种任务中实现了更快收敛与更优的下游性能。

    

    正则化强化学习（RL）算法依赖于蒙特卡洛树搜索或单步前瞻等估计器，在训练和推理过程中提供策略、价值或Q值估计。然而，现有的基于老虎机的估计框架评估的是采样效率，而非估计的质量。我们通过将估计生成形式化为一个推荐过程来解决这一问题，将基于老虎机的采样视为一种改进算子。为了评估推荐（估计）的质量，我们定义了正则化感知的推荐者遗憾，分析了其下界，并证明了像均匀采样这样简单的算法也能达到这些下界。最后，我们在推荐者MCTS框架下统一了基于老虎机和基于搜索的目标改进方法。在合成树、街机游戏和科学发现任务中，推荐者通常能带来更快的目标收敛和更优的下游性能。我们的实验刻画了……

    arXiv:2610.32910v1 Announce Type: new  Abstract: Regularised reinforcement learning (RL) algorithms rely on estimators, like Monte Carlo tree search or one-step look-ahead, to provide policy, value, or Q-value estimates in training and inference. However, existing bandit-based estimation frameworks evaluate sampling efficiency, not the quality of estimates. We address this by formalising estimate generation as a recommendation process, treating bandit-based sampling as an improvement operator. To evaluate the quality of a recommendation (estimate), we define regulariser-aware recommender regrets, analysing their lower bounds, and proving that simple algorithms like uniform sampling can achieve these bounds. Finally, we unify bandit- and search-based target improvement under recommender MCTS. Across synthetic trees, arcade games, and scientific discovery tasks, recommenders often yield faster target convergence and superior downstream performance. Our experiments characterise for the fi
    
[^14]: LoCA：实现地理空间位置编码器的多模态能力

    LoCA: Enabling Multi-Modality in Geospatial Location Encoders

    [https://arxiv.org/abs/2610.32894](https://arxiv.org/abs/2610.32894)

    该论文从理论上证明了位置编码器受Lipschitz约束其输出固有维度至多为2，并提出LoCA方法使位置编码器能够同时将单个位置映射到多个模态的描述符，从而实现多模态能力。

    

    位置编码器通过与配对的地球观测影像进行对比对齐，将地理坐标映射为高维描述符。我们证明这种构造存在结构性限制：任何Lipschitz位置编码器的输出固有维度至多为2，无论其架构或带宽如何，这迫使对比损失严格超过其渐近最优值。在实证上，联合训练还会使编码器的嵌入协方差坍缩为低秩。尽管存在维度限制，当前的位置编码器通过针对冻结的图像编码器进行训练来规避这种谱坍缩，我们将其重新解释为一种协方差正则化器，为位置编码器提供了一个可折叠到的高秩目标。然而，将这种方法扩展到多模态输入具有挑战性，因为位置编码器需要同时将单个位置映射到多个可能不相容的描述符。我们识别出两个……（原文截断）

    arXiv:2610.32894v1 Announce Type: cross  Abstract: Location encoders map geographic coordinates to high-dimensional descriptors through contrastive alignment with paired Earth Observation imagery. We prove that this construction is structurally limited: any Lipschitz location encoder produces an output of intrinsic dimension at most 2, regardless of architecture or bandwidth, which forces the contrastive loss to strictly exceed its asymptotic optimum. Empirically, joint training also collapses the encoder's embedding covariance to low rank. Although dimension-limited, current location encoders sidestep this spectral collapse by training against a frozen image encoder, which we reinterpret as a covariance regularizer that supplies a high-rank target for the location encoder to fold onto. However, extending this approach to multimodal inputs is challenging, as location encoders should simultaneously map a single location to multiple, potentially incompatible, descriptors. We identify two
    
[^15]: 当人类纠偏数据迁移时：联邦VLA适配中的安全性

    When Human Corrections Travel: Safety in Federated VLA Adaptation

    [https://arxiv.org/abs/2610.32888](https://arxiv.org/abs/2610.32888)

    论文提出在联邦学习框架下共享人类纠偏衍生的VLA适配更新时，必须保留多模态纠偏的上下文与不确定性、在接收机器人闭环中预先评估更新、并兼顾隐私泄露与安全关键证据的保留，从而确保跨任务、环境与机器人本体的安全适配。

    

    视觉-语言-动作（VLA）策略能够适应本地的人类干预，而联邦学习可以在不汇集原始记录的情况下整合分布式机器人经验。我们考察了两者未来的交叉点：共享由多样化的人类纠偏所衍生、跨任务、环境、人员和机器人本体的适配更新。纠偏在模态、安全语义（紧迫性、行为范围等）和隐私敏感性方面各不相同。一个提升平均任务成功率的共享更新，可能抑制罕见的安全干预，或在另一台机器人上产生不安全的响应。我们提出三项要求：在决定共享内容时保留多模态纠偏的上下文与不确定性；在部署前于每台接收机器人的闭环中评估候选更新；以及在评估信息泄露的同时评估安全关键证据的保留。在人类交互期间，即时、清晰可读的防御行为仍然是必要的。我们提供了对……

    arXiv:2610.32888v1 Announce Type: cross  Abstract: Vision-language-action (VLA) policies can adapt to local human intervention, while federated learning can combine distributed robot experience without pooling raw records. We examine their prospective intersection: sharing adaptations derived from diverse human corrections across tasks, environments, people, and robot embodiments. Corrections differ in modality, safety semantics(urgency, behavioral scope, etc.), and privacy sensitivity. A shared update that improves mean task success may suppress a rare safety intervention or produce an unsafe response on another robot. We argue for three requirements: preserve the context and uncertainty of multimodal corrections when deciding what to share; assess candidate updates in each receiving robot's closed loop before deployment; and evaluate leakage alongside retention of safety-critical evidence. Immediate, legible defensive behavior remains necessary during human interaction. We offer coun
    
[^16]: 基于多步转移预测的强化学习：一种贝尔曼理论

    Reinforcement Learning with Multi-Step Transition Predictions: A Bellman Theory

    [https://arxiv.org/abs/2610.32887](https://arxiv.org/abs/2610.32887)

    该论文提出块反馈协议下的贝叶斯贝尔曼方程理论，揭示多步转移预测的价值源于递归改变贝尔曼展开中最大化与期望的顺序，从而无需状态空间指数级扩展即可量化预测信息对强化学习的增益。

    

    对未来转移实现的多步预测可以改进强化学习，但其价值仍难以量化。直接将这些预测纳入状态可能导致状态空间呈指数级增长，使得直接动态规划在计算上代价高昂。我们在块反馈协议下研究折扣有限马尔可夫决策过程（MDP）：在每个K周期块的开始，决策者观察到针对部分动作的K步概率转移预测，并在块内根据观察到的状态调整决策。我们证明最优值在原始状态空间上满足一个贝叶斯贝尔曼方程。这一表示揭示了预测信息通过递归地改变贝尔曼展开中最大化与期望的局部顺序来创造价值。我们通过贝尔曼-詹森分析将这一效应形式化，并量化了预测时域、预测误差、动作覆盖范围等因素的影响。

    arXiv:2610.32887v1 Announce Type: new  Abstract: Multi-step predictions of future transition realizations can improve reinforcement learning, but their value remains difficult to quantify. Directly including these predictions in the state can lead to exponential state-space growth, making direct dynamic programming computationally expensive. We study discounted finite MDPs under a block-feedback protocol: at the start of each K-period block, the decision maker observes $K$-step probabilistic transition predictions for a subset of actions and adapts decisions to observed states within the block. We show that the optimal value satisfies a Bayesian Bellman equation on the original state space. This representation reveals that predictive information creates value by recursively changing the local order of maximization and expectation in the Bellman expansion. We formalize this effect through a Bellman--Jensen analysis and quantify the effects of prediction horizon, error, action coverage, 
    
[^17]: 信任工具：大语言模型漏洞审计员对静态分析证据的非对称盲从

    Trusting the Tool: Asymmetric Deference of LLM Vulnerability Auditors to Static-Analysis Evidence

    [https://arxiv.org/abs/2610.32876](https://arxiv.org/abs/2610.32876)

    本研究首次量化了 LLM 漏洞审计员对静态分析证据的盲从性：真实证据几乎没有提升审计准确率，而工具的虚假“保证”却能使真实漏洞的检出率下降多达 19 个百分点，揭示了模型对工具证据的非对称依赖。

    

    将大语言模型（LLM）与静态分析相结合的流水线允许模型将分析器的输出作为证据使用，但模型能否区分真实与虚假证据，以及其审查程度是否取决于证据所声称的内容，此前尚未被测量。我们构建了一个用于 C/C++ 的过程内分析器，它会生成证明义务，并在六种仅注释不同的证据条件下，对 PrimeVul 数据集中存在漏洞的函数与已修补的函数进行审计，其中包括虚假保证（谎称一个缺失的检查已存在）和虚假警报（谎称一个新增的检查缺失）。该研究涵盖四组互不相交集合中的 916 对函数、来自四家供应商的 11 种模型配置以及共 16,572 次审计；没有任何模型被训练或微调。真实证据几乎未能提升准确率（仅 +0.02）。虚假证据则表现出非对称性：相对于关于相同操作的真实注释，虚假保证使真实漏洞的检出率下降多达 19 个百分点。

    arXiv:2610.32876v1 Announce Type: new  Abstract: Pipelines that combine large language models (LLMs) with static analysis let the model consume the analyser's output as evidence, but whether the model distinguishes true from false evidence, and whether its scrutiny depends on what the evidence claims, has not been measured. We built an intraprocedural analyser for C/C++ that produces proof obligations and audited vulnerable and patched functions from PrimeVul under six evidence conditions that differ only in the notes, including fabricated assurances that a missing check exists and fabricated alarms that an added check is missing. The study covers 916 pairs in four disjoint sets, eleven model configurations from four vendors and 16,572 audits; no model is trained or fine-tuned. True evidence barely improved accuracy (+0.02). False evidence acted asymmetrically: relative to true notes about the same operations, false assurances lowered the detection of real vulnerabilities by up to 19 p
    
[^18]: 基于TopKLoRA的可解释稀疏微调实现忠实的电路发现

    Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery

    [https://arxiv.org/abs/2610.32866](https://arxiv.org/abs/2610.32866)

    本文提出了设计上即可解释的稀疏微调方法TopKLoRA，通过在每个token上仅激活k个最大的适配器潜在变量、纯粹以下游损失训练，使其潜在变量在功能上具有因果性，从而能够忠实地发现休眠代理后门等模型内部电路。

    

    摘要（arXiv:2610.32866v1，公告类型：cross）：在来源未知的数据集上进行微调时，如果数据中包含被投毒的样本，可能会在被训练的模型中植入后门。后门难以定位和移除，并且可以在安全训练之后持续存在。虽然参数高效微调方法能够捕捉训练过程中引入的全部参数变化，但它们不易于解释。我们提出了TopKLoRA的完整方法，该方法扩展了Masiak等人（2025）的初步提案，这是一种设计上即可解释的LoRA变体，其中在每个token上只有$k$个最大的适配器潜在变量处于激活状态。与稀疏自编码器（SAE）不同，TopKLoRA完全基于下游损失进行训练，其中优化压力会促使模型学习在功能上具有因果作用的潜在变量。我们在一个休眠代理（sleeper-agent）后门案例研究中展示了这种稀疏LoRA适配器：我们通过在投毒的指令遵循数据集上微调gemma-2-2b、Qwen2.5-1.5B和Qwen2.5-7B来创建模型生物体（model organisms）。对于每个生物体，我们识别并……（摘要原文在此处截断）

    arXiv:2610.32866v1 Announce Type: cross  Abstract: Fine-tuning on datasets of unknown origin can create a backdoor in the trained model when the data includes poisoned examples. Difficult to localise and remove, backdoors can persist through safety training. While parameter-efficient fine-tuning methods capture the full parameter changes introduced during training, they are not easily interpretable. We present the full method of TopKLoRA, extending the preliminary proposal of Masiak et al. (2025), an interpretable-by-design LoRA variant in which only the $k$ largest adapter latents are active at each token. Unlike SAEs, TopKLoRA is trained purely on downstream loss, where optimisation pressure incentivises learning functionally causal latents. We demonstrate this sparse LoRA adapter in a sleeper-agent backdoor case study where we create model organisms by fine-tuning gemma-2-2b, Qwen2.5-1.5B, and Qwen2.5-7B on a poisoned instruction-following dataset. For each organism, we identify and
    
[^19]: PoKER：一种基于随机目标机器学习的空空导弹杀伤概率估计模型

    PoKER: A Probability of Kill Estimation Rate Model for Air-to-Air Missiles Using Machine Learning on Stochastic Targets

    [https://arxiv.org/abs/2610.32864](https://arxiv.org/abs/2610.32864)

    PoKER是一种利用机器学习整合目标随机行为与导弹脱靶距离、从而更准确预测超视距空战中空空导弹杀伤概率的新型概率模型，有望成为飞行员改善作战决策的重要工具。

    

    这项工作介绍了PoKER，这是一种新颖的概率模型，旨在优化空对空场景中导弹发射的有效性，特别是在超视距（BVR）空战中。与传统的武器交战区（WEZ）模型基于最大距离、最小距离和不可逃逸距离等静态距离来划定区域不同，PoKER应用机器学习，通过整合目标的随机行为和导弹脱靶距离，更准确地预测杀伤概率。该模型能够动态评估目标行为，极大地扩展了交战分析的预测能力。通过考虑弹头杀伤力、目标和发射机朝向以及具体交战条件等因素，PoKER为交战动态提供了重要见解并量化了成功概率。因此，它有可能成为超视距空战飞行员的重要工具，从而改善作战决策。

    arXiv:2610.32864v1 Announce Type: cross  Abstract: This work introduces PoKER, a novel probabilistic model engineered to optimize missile launch effectiveness in air-to-air scenarios, specifically within Beyond Visual Range (BVR) air combat. Unlike conventional Weapon Engagement Zone (WEZ) models that delineate zones based on static distances such as maximum, minimum, and no-escape ranges, PoKER applies machine learning to predict kill probabilities more accurately by integrating the stochastic behaviors of targets and missile miss distances. This model dynamically evaluates target behavior, greatly expanding the predictive capabilities of engagement analysis. By factoring in elements such as warhead lethality, target and shooter orientations, and the specific conditions of engagement, PoKER provides important insights into engagement dynamics and quantifies success probabilities. Consequently, it can potentially be an important tool for BVR air combat pilots, improving operational dec
    
[^20]: 时间序列基础模型部署与预训练的可持续性：从碳足迹分析到基于剪枝的减排方案

    Sustainability of TSFM Deployment and Pretraining: From Carbon Footprint Analysis to Mitigation via Pruning

    [https://arxiv.org/abs/2610.32859](https://arxiv.org/abs/2610.32859)

    该论文通过建模不同部署场景下的累计碳成本，评估了时间序列基础模型相比需定期重训的传统模型的可持续性，并证明仅通过无梯度剪枝移除单个编码器块，即可在几乎不损失零样本预测性能的前提下减少6%-9%的计算量、碳排放和延迟。

    

    时间序列基础模型（TSFM）具有强大的零样本预测能力，但其高昂的预训练成本引发了一个实际的可持续性问题：与必须定期重新训练的有监督独立模型相比，这笔前期投资能否在整个部署生命周期中得到回报？我们在一个现实的工业场景中对不同方法的累计碳成本进行了建模：对于大规模或高频查询的部署，Chronos-2可在数月内摊销其预训练成本；而对于小规模或低频查询的部署，经典模型始终保持更低的成本。对于需要降低TSFM自身运行成本的情形，我们证明，通过无梯度剪枝准则移除单个编码器块（占参数的7.9%），可在FLOPs计算量、CO2排放和延迟方面实现一致的6%至9%的降低，同时在包含25个任务的基准测试中几乎不影响零样本预测性能。

    arXiv:2610.32859v1 Announce Type: new  Abstract: Time Series Foundation Models (TSFMs) offer strong zero-shot forecasting capabilities, but their high pretraining cost raises a practical sustainability question: does this upfront investment pay off over a full deployment lifecycle, compared to supervised standalone models that must be periodically retrained? We model the cumulative carbon cost of different approaches across a realistic industrial scenario: Chronos-2 amortizes its pretraining cost within months for large or frequently queried deployments, but a classical model remains cheaper indefinitely for small or rarely queried ones. For cases where reducing the TSFM's own cost matters, we show that removing a single encoder block (7.9% of parameters) via a gradient-free pruning criterion yields a consistent 6 to 9% reduction in FLOPs, CO2 emissions, and latency, at essentially no cost to zero-shot forecasting performance across a 25-task benchmark.
    
[^21]: 正则化强化学习改善仿真到现实的迁移

    Regularised RL improves sim-to-real transfer

    [https://arxiv.org/abs/2610.32855](https://arxiv.org/abs/2610.32855)

    本文提出基于软mellowmax价值函数的新型正则化强化学习算法SMM-AC，该算法在真实机器人上比SAC等基线方法获得更高回报且抗扰动能力更强，证明了正则化能够有效改善sim-to-real迁移性能。

    

    正则化强化学习（RL）具有可证明的鲁棒性保证，理应促进策略从仿真到机器人的迁移（即sim-to-real）。然而，像近端策略优化（PPO）这类弱正则化算法仍然是sim-to-real流程中事实上的标准选择，而正则化对迁移的实际益处至今仍缺乏充分的刻画。在本工作中，我们研究了常用RL算法的sim-to-real性能，旨在刻画正则化对机器人性能的影响。为此，我们推导出了Soft MellowMax Actor-Critic（SMM-AC），这是一种基于软mellowmax价值函数的新型正则化RL算法，具有强鲁棒性保证。在MuJoCo连续控制基准测试中，SMM-AC与软演员-评论家（SAC）等基线方法相比具有竞争力。在真实机器人上，SMM-AC获得了更高的回报，同时对观测中的扰动表现出更强的鲁棒性。

    arXiv:2610.32855v1 Announce Type: new  Abstract: Regularised reinforcement learning (RL) has provable robustness guarantees that should ease the transfer of policies from simulation to robots, known as sim-to-real. Yet, weakly regularised algorithms like Proximal Policy Optimisation (PPO) remain the de facto choice in sim-to-real pipelines, while the practical benefits of regularisation for transfer remain poorly characterised. In this work, we study the sim-to-real performance of commonly used RL algorithms, with the aim of characterising the effects of regularisation on robotic performance. To that end, we derive Soft MellowMax Actor-Critic (SMM-AC), a novel regularised RL algorithm based on the soft mellowmax value function, which carries strong robustness guarantees. On MuJoCo continuous control benchmarks, SMM-AC is competitive with baselines like Soft Actor-Critic (SAC). On real robots, SMM-AC achieves higher return, while exhibiting greater robustness to perturbations in observa
    
[^22]: 低秩适应的奇异几何

    Singular Geometry of Low-Rank Adaptation

    [https://arxiv.org/abs/2610.32854](https://arxiv.org/abs/2610.32854)

    本文借助奇异学习理论中的实对数典型阈值（RLCT）刻画了LoRA作为奇异统计模型的学习复杂度，证明该复杂度由适应层前向激活与反向传播敏感度构成的协方差决定，并在非退化情形下恢复降秩回归的经典复杂度。

    

    低秩适应如今已成为对大型神经网络进行参数高效微调的标准方法。其低秩分解本质上不可辨识，这使得LoRA成为一个奇异统计模型，其学习复杂度无法仅凭参数维度来刻画。我们利用奇异学习理论中的实对数典型阈值（RLCT）来研究这种复杂度，并分析它如何同时依赖于低秩参数化和预训练网络。在局部线性化下，LoRA诱导出一个低秩迹回归模型，其Kullback–Leibler几何由一个前向-后向协方差所控制，该协方差由被适应层的激活值和反向传播敏感度构造而成。这一协方差决定了哪些低秩扰动在统计上是可观测的。当它在相关低秩方向上非退化时，我们恢复了降秩回归的经典学习复杂度；当它……（原文摘要在此处截断）

    arXiv:2610.32854v1 Announce Type: new  Abstract: Low-rank adaptation (LoRA) is now a standard approach to parameter-efficient fine-tuning of large neural networks. Its low-rank factorization is inherently non-identifiable, making LoRA a singular statistical model whose learning complexity cannot be described by parameter dimension alone. We study this complexity using the real log canonical threshold (RLCT) from singular learning theory and how it depends on both the low-rank parameterization and the pretrained network. Under a local linearization, LoRA induces a low-rank trace regression model whose Kullback--Leibler geometry is controlled by a forward-backward covariance, constructed from activations and backpropagated sensitivities at the adapted layer. This covariance determines which low-rank perturbations are statistically observable. When it is nondegenerate along the relevant low-rank directions, we recover the classical learning complexity of reduced-rank regression; when its 
    
[^23]: 基于Littlestone维度的近最优列表可复制性

    Near-Optimal List Replicability from Littlestone Dimension

    [https://arxiv.org/abs/2610.32853](https://arxiv.org/abs/2610.32853)

    本文证明了Littlestone维度为d的概念类的列表可复制数至多为2^{O(d·min{log d, v*})}，该界在指数上仅差一个对数因子即匹配最坏情形下界，并指数级改进了此前最佳结果，从而近乎最优地解决了列表可复制性问题。

    

    我们在可实现PAC学习框架中研究列表可复制数，并证明可数域上每个Littlestone维度为 d 且对偶VC维度为 v* 的概念类 H 满足 LR(H) ≤ 2^{O(d·min{log d, v*})}。该结果在指数上仅相差一个对数因子的情况下匹配了 2^{Ω(d)} 的最坏情形下界，并指数级地改进了 Bun、Livni 和 Moran [FOCS 2020] 此前的最佳上界 LR(H) ≤ 2^{2^{O(d)}}。与 Ghazi、Golowich、Kumar 和 Manurangsi [STOC 2021] 以及 Lyu [STOC 2026] 的工作所隐含的固定精度 ε>0 下 2^{O_ε(d²)} 的界不同，我们的界与 ε 无关。其中 2^{O(d log d)} 的界来自一个新的界定列表可复制性的归纳框架，该框架建立在差分隐私学习中最近提出的不可约性概念之上；而 2^{O(dv*)} 的界则来自一个简单的全局（此处摘要内容不完整）……

    arXiv:2610.32853v1 Announce Type: new  Abstract: We study the list replicability number in the realizable PAC learning framework and prove that every concept class $H$ over a countable domain with Littlestone dimension $d$ and dual VC dimension $v^*$ satisfies \[ \mathrm{LR}(H)\leq 2^{O(d\min\{\log d,v^*\})}. \] This matches the worst-case lower bound of $2^{\Omega(d)}$ up to a logarithmic factor in the exponent and exponentially improves the previous best upper bound $\mathrm{LR}(H)\le 2^{2^{O(d)}}$ of Bun, Livni, and Moran [FOCS 2020]. Unlike earlier bounds of $2^{O_\epsilon(d^2)}$ for fixed accuracy $\epsilon>0$ implied by the work of Ghazi, Golowich, Kumar, and Manurangsi [STOC 2021] and Lyu [STOC 2026], our bound is independent of $\epsilon$.   The $2^{O(d\log d)}$ bound follows from a new inductive framework for bounding list replicability that builds on the recent notion of irreducibility from differentially private learning. The $2^{O(dv^*)}$ bound follows from a simple globall
    
[^24]: 带反向耦合的神经低度滤波：跨层特征学习的谱理论

    Neural LoFi with Backward Coupling: A Spectral Theory of Cross-Layer Feature Learning

    [https://arxiv.org/abs/2610.32849](https://arxiv.org/abs/2610.32849)

    该论文提出带反向耦合的神经低度滤波谱理论，从反向传播梯度推导出跨层局部耦合动力学，利用上层敏感性为早期谱步骤生成有效标签，从而在逐层学习框架中实现跨层特征交互。

    

    在联合训练中，反向传播允许后面的层重塑前面的表示，这种反馈机制在纯逐层学习中是不存在的。我们扩展了神经低度滤波（Neural LoFi）——一种最近提出的用于特征学习的逐层谱替代方法——以捕捉这种交互作用。从精确的反向传播梯度出发，我们推导出神经元的输入权重与其对应的下一层列之间的局部耦合。在具有平滑激活函数和固定周围环境的受控小初始化条件下，这些动力学会放大跨层相关算子的奇异模式。每个增长的模式通过共享的神经元系数将较早层的特征与上层响应配对。这启发了一种反向耦合算法：上层敏感性为较早的谱步骤生成依赖于样本的有效标签，同时经过修正的特征则细化向上传输的信息。

    arXiv:2610.32849v1 Announce Type: cross  Abstract: During joint training, backpropagation allows later layers to reshape earlier representations, a feedback mechanism absent from purely layer-wise learning. We extend Neural Low-Degree Filtering (Neural LoFi), a recently proposed layer-wise spectral surrogate for feature learning, to capture this interaction. Starting from the exact backpropagation gradients, we derive a local coupling between a neuron's incoming weights and its corresponding next-layer column. In a controlled small-initialization regime with smooth activations and fixed surroundings, these dynamics amplify singular modes of a cross-layer correlation operator. Each growing mode pairs an earlier feature with an upper-layer response through shared neuron coefficients. This motivates a backward-coupling algorithm in which upper-layer sensitivities generate sample-dependent effective labels for earlier spectral steps, while corrected features refine upward transmission. A f
    
[^25]: 关于乱序差分隐私下稀疏线性模型的极限

    On the Limits of Sparse Linear Model with Shuffled Differential Privacy

    [https://arxiv.org/abs/2610.32822](https://arxiv.org/abs/2610.32822)

    本文证明了在非交互式乱序差分隐私协议下稀疏线性回归存在Ω(√(dk)/(nε))的固有维度误差下界，并提出了相应的算法，进而通过交互式协议克服该维度障碍。

    

    稀疏线性回归是高维统计中的一个基本问题，其隐私保护版本在差分隐私的中心化模型与本地化模型之间存在较大差距。乱序差分隐私介于两者之间，在无需信任中心管理者的前提下，提供接近中心化模型的精度。本文研究了非交互式与交互式乱序差分隐私（SDP）协议下的稀疏线性回归。在非交互式设定下，本文证明了任何非交互式(ε,δ)-SDP协议在维度为d、稀疏度为k、用户数为n的情况下，都会产生Ω(√(dk)/(nε))的ℓ2范数估计误差，揭示了一种固有的维度障碍。此外，本文提出了一种非交互式算法，达到了Õ(√(k/n) + d√k/(nε))的ℓ2范数误差上界。为克服这一障碍，我们转向交互式（摘要在此处截断）

    arXiv:2610.32822v2 Announce Type: new  Abstract: Sparse linear regression is a fundamental problem in high-dimensional statistics, and its private counterparts leave a large gap between the central and the local model of differential privacy. Shuffled differential privacy (SDP) sits between the two, offering accuracy closer to the central model without the trust requirement of a central curator. This paper studies sparse linear regression under both non-interactive and interactive SDP protocols. In the non-interactive setting, any non-interactive $(\epsilon,\delta)$-SDP protocol is shown to incur an $\ell_2$-norm estimation error of $\Omega( \frac{\sqrt{dk}}{n\epsilon} )$ with dimension $d$, sparsity $k$, and $n$ users, revealing an inherent dimensionality barrier. In addition, a non-interactive algorithm is proposed that achieves an $\ell_2$-norm upper bound of $\tilde{\mathcal{O}}( \sqrt{\frac{k}{n}} + \frac{d\sqrt{k}}{n\epsilon})$. To overcome this barrier, we turn to the interactiv
    
[^26]: U-NeXt：面向新生儿脑电图的通用事件检测框架

    U-NeXt: General-purpose event detection for neonatal EEG

    [https://arxiv.org/abs/2610.32811](https://arxiv.org/abs/2610.32811)

    提出了U-NeXt——一个基于语义分割的新生儿脑电通用事件检测框架，通过统一的卷积编码器-解码器架构对低电压区间、伪迹和癫痫发作三类事件实现了优于任务特定模型的检测性能，尤其在短事件检测的敏感性上有显著提升。

    

    我们开发了U-NeXt，这是一个基于语义分割的新生儿脑电图（EEG）事件检测通用框架。其卷积编码器-解码器结构直接对单个通道进行分割，并使用长输入片段以保留时间上下文。我们基于低电压区间、伪迹和癫痫发作三个数据集（分别包含51名、98名和278名新生儿）构建了共享的架构与训练框架，共涵盖超过11,000小时的脑电数据和82,000个标注事件。我们将该框架与任务特定的ConvNeXt分类器进行了性能比较。针对癫痫发作检测，我们还评估了在扩充数据集上训练的U-NeXt+。两个U-NeXt变体与ConvNeXt基线模型均在两个独立的留出癫痫发作数据集（Helsinki和Cork，共130名新生儿）上进行评估，其记录由三位专家标注。U-NeXt在所有三种事件类型的检测中均有所提升，在短事件检测的敏感性方面提升尤为显著。

    arXiv:2610.32811v1 Announce Type: new  Abstract: We developed U-NeXt, a general framework for neonatal electroencephalography (EEG) event detection based on semantic segmentation. Its convolutional encoder-decoder directly segments individual channels using long input segments to preserve temporal context. A shared architecture and training framework were developed using low-voltage interval, artefact, and seizure datasets containing 51, 98, and 278 neonates, respectively, comprising over 11,000 hours of EEG and 82,000 annotated events. Performance was compared with task-specific ConvNeXt classifiers. For seizure detection, we also evaluated U-NeXt+, trained on an expanded dataset. Both U-NeXt variants and the ConvNeXt baseline were evaluated on two independent held-out seizure datasets, Helsinki and Cork, comprising 130 neonates, with recordings annotated by three experts. U-NeXt improved detection across all three event types, with particular gains in sensitivity for short events. U-
    
[^27]: 学习优化器调度中的乘法与加法状态耦合

    Multiplicative vs. Additive State Coupling in Learned Optimizer Schedules

    [https://arxiv.org/abs/2610.32801](https://arxiv.org/abs/2610.32801)

    本研究通过反向传播学习MAX-CUT求解器的调度，发现双线性（乘法）状态耦合架构在分布外泛化上相比加法耦合具有显著且持久的优势，从而将关键架构区别定位于求解器状态进入调度的方式，而非状态自适应性本身。

    

    我们通过在展开的经典求解器上进行反向传播，来学习MAX-CUT连续惩罚松弛问题的调度。状态盲调度能够泛化到比训练所用实例大得多的实例。状态自适应的双线性调度同样能实现分布外泛化，在15个随机种子上优于其未训练的基线（t=4.1）。相比之下，具有相同输入和输出头的加法消融模型在分布内与双线性模型在统计上不可区分，但在分布外的表现却比其自身基线更差（t=-11.7）。这种分离在无正则化的情况下即会出现，并在30轮权重衰减扫描中持续存在：双线性模型的优势在每个测试值下都保持显著，而加法模型始终无法恢复。在该设置下，这些结果将求解器状态进入调度的方式，而非单纯的状态自适应性，隔离为与（分布外泛化表现）相关的关键架构区别。

    arXiv:2610.32801v1 Announce Type: cross  Abstract: We learn schedules for a continuous penalty relaxation of MAX-CUT by backpropagating through an unrolled classical solver. A state-blind schedule generalizes to instances substantially larger than those used for training. A state-adaptive bilinear schedule also generalizes out of distribution, outperforming its untrained baseline across 15 seeds ($t=4.1$). By contrast, an additive ablation with identical inputs and output heads is statistically indistinguishable from the bilinear model in distribution but performs worse than its own baseline out of distribution ($t=-11.7$). This separation appears without regularization and persists across a $30$ round weight-decay sweep: the bilinear advantage remains significant at every tested value, whereas the additive model never recovers. The results in this setting isolate how solver state enters the schedule, rather than state adaptivity alone, as the architectural distinction associated with 
    
[^28]: CARES：面向持续学习的覆盖感知回放样本选择

    CARES: Coverage-Aware Replay Exemplar Selection for Continual Learning

    [https://arxiv.org/abs/2610.32799](https://arxiv.org/abs/2610.32799)

    CARES利用冻结预训练特征空间的类别感知、密度加权覆盖度来选择回放样本，无需任何模型优化即可构建缓冲区，比现有基于核心集的选择器快2.3至8.8倍，同时在平衡、不平衡和标签噪声数据流中保持最先进的性能。

    

    基于回放的持续学习通过保存过去样本的缓冲区来缓解灾难性遗忘，因此在固定内存预算下，缓冲区的构成至关重要。近期的基于核心集的回放方法利用模型衍生的梯度、损失或辅助优化来提升缓冲区质量，但由于内存需要反复更新，会带来大量计算开销。我们提出CARES，一种无需持续模型优化的缓冲区构建方法，它利用冻结的预训练特征空间的几何结构，采用类别感知、密度加权的覆盖度策略。我们在平衡、不平衡和标签噪声数据流下，于在线类增量与任务增量设置中对CARES进行了评估。在Tiny-ImageNet上，CARES比现有的基于核心集的选择器快2.3至8.8倍。尽管CARES在没有使用持续学习器任何训练信号的情况下实现了这一加速，但它在平衡、不平衡和标签噪声数据流中仍名列最先进水平。

    arXiv:2610.32799v1 Announce Type: new  Abstract: Replay-based Continual Learning mitigates catastrophic forgetting by keeping a buffer of past samples, making buffer composition critical under a fixed memory budget. Recent coreset-based replay methods improve buffer quality using model-derived gradients, losses, or auxiliary optimization, but incur substantial computational overhead as memory is repeatedly updated. We propose CARES, a continual model-optimization-free approach for buffer construction that instead exploits the geometry of a frozen pretrained feature space using class-aware, density-weighted coverage. We evaluate CARES across balanced, imbalanced, and label-noisy streams under online class- and task-incremental settings. On Tiny-ImageNet, CARES is 2.3 to 8.8 times faster than existing coreset-based selectors. Although CARES achieves this speedup using no training signal from the continual learner, it ranks among the state of the art across balanced, imbalanced, and label
    
[^29]: 词元感知机的泛化性质：嵌入与词表的作用

    Generalization properties of the token perceptron: the role of embedding and vocabulary

    [https://arxiv.org/abs/2610.32788](https://arxiv.org/abs/2610.32788)

    该论文提出“词元感知机”这一最小可解线性模型，并用复制方法解析计算其泛化误差，揭示了嵌入空间的几何结构与有限词表大小如何影响大语言模型中词元预测的学习与泛化。

    

    大语言模型被训练用于从有限词表中预测词元，每个词元由 d 维嵌入空间中的一个向量表示。因此，词元预测本质上是一个分类问题，其类别的几何结构由嵌入决定。我们研究这种几何结构如何影响一个最小可解模型中的学习：该模型是一个线性预测器，通过每个位置上一个 d×d 的矩阵，将 N 个嵌入词元的序列映射为 d 维输出，并在教师-学生框架下进行训练。我们将其称为“词元感知机”（token perceptron）。我们利用复制方法计算其泛化误差，在每位置样本数 α=P/N 固定的情况下取 N→∞，同时保持嵌入维度 d 和词表大小 D 有限。作为第一个参考情形，我们考虑球面上的连续嵌入，以分离出嵌入维度所起的作用。此时自然负载为 α/d，且当 d>1 时，归一化标签固定了 d……

    arXiv:2610.32788v1 Announce Type: cross  Abstract: Large language models are trained to predict tokens from a finite vocabulary, each represented by a vector in a $d$-dimensional embedding space. Therefore, token prediction is a classification problem whose classes have a geometry determined by the embedding. We study how this geometry affects learning in a minimal solvable model: a linear predictor that maps a sequence of $N$ embedded tokens to a $d$-dimensional output through one $d\times d$ matrix per position, trained in the teacher--student setting. We call this the \textit{token perceptron}. We compute its generalization error with the replica method, taking $N\to\infty$ at fixed number of examples per position $\alpha=P/N$ and keeping $d$ and the vocabulary size $D$ finite. As a first reference case, we consider continuous embeddings on the sphere, which isolate the role of the embedding dimension. The natural load is then $\alpha/d$, and for $d>1$ a normalized label fixes the d
    
[^30]: 基于深度学习的乳腺癌脑转移瘤全脑MRI预测HER2状态

    Deep Learning-Based Prediction of HER2 Status from Whole-Brain MRI in Breast Cancer Brain Metastases

    [https://arxiv.org/abs/2610.32774](https://arxiv.org/abs/2610.32774)

    该研究探索利用深度学习从乳腺癌脑转移瘤的全脑T1加权MRI中直接预测HER2状态，并评估了引入专家肿瘤分割信息作为额外输入通道能否提升预测性能。

    

    目的：人表皮生长因子受体2（HER2）状态是转移性乳腺癌预后评估和治疗决策的关键生物标志物。因此，本研究旨在探索能否利用深度学习直接从乳腺癌脑转移瘤（BCBM）的全脑对比增强T1加权（T1w）MRI扫描中预测HER2状态，以及提供显式肿瘤分割信息是否能改善预测效果。方法：利用公开可用的BCBM-Radiogenomics数据集，我们分析了135名患者，其中HER2阳性70名，HER2阴性65名。研究实现并评估了三种方法：传统影像组学模型、仅使用T1w MRI扫描的简单全卷积网络（SFCN），以及使用T1w MRI扫描并将专家标注的肿瘤掩膜作为附加输入通道的SFCN。结果：影像组学模型获得了受试者工作特征曲线下面积（AUC）……（原文摘要在此处截断）

    arXiv:2610.32774v1 Announce Type: new  Abstract: Purpose: Human epidermal growth factor receptor 2 (HER2) status is a key biomarker relevant for prognostication and treatment decision making in metastatic breast cancer. Thus, the aim of this work was to explore whether HER2 status can be predicted directly from whole-brain post-contrast T1-weighted (T1w) MRI scans in breast cancer brain metastases (BCBM) using deep learning, and whether providing explicit tumour segmentation information improves prediction.   Methods: Using the publicly available BCBM-Radiogenomics dataset, we analysed 135 patients, including 70 HER2+ and 65 HER2- patients. Three approaches were implemented and evaluated: a conventional radiomics model, a simple fully convolutional network (SFCN) using T1w MRI scans alone, and an SFCN using T1w MRI scans with an expert-derived tumour mask as an additional input channel.   Results: The radiomics model achieved an area under the receiver operating characteristic curve (R
    
[^31]: 基于奖励模型梯度的方向感知词元信用分配

    Direction-Aware Token Credit Assignment with Reward-Model Gradients

    [https://arxiv.org/abs/2610.32772](https://arxiv.org/abs/2610.32772)

    提出方向感知词元信用分配方法 DATC，利用冻结奖励模型的梯度来度量奖励方向与 GRPO 更新方向的对齐程度，从而为每个词元分配信用权重，在两个模型家族的四种设置中均优于 GRPO。

    

    GRPO 在所有词元间共享响应级别的优势，但尚未解决每个词元应以多大强度参与学习的问题。由共享优势所引发的更新会强化或抑制每个位置上采样的词元，但这些更新并不总是指向词元级别上奖励的提升。这启发了一种基于“功”的信用分配视角：正如功取决于力沿位移方向的分量，词元信用应当反映策略更新方向上的局部奖励变化。我们提出了方向感知词元信用分配（DATC），该方法将冻结的奖励模型梯度映射到策略 logit 空间，通过评估奖励方向与 GRPO 带符号的采样动作更新方向之间的一致性来分配权重，同时保持响应级别优势的符号与均值。DATC 在两个模型家族的全部四种设置中均优于 GRPO。

    arXiv:2610.32772v1 Announce Type: new  Abstract: GRPO shares a response-level advantage across tokens, leaving unresolved how strongly each token should contribute to learning. Updates induced by the shared advantage reinforce or suppress the sampled token at each position, but these updates do not always point toward increasing reward at the token level. This motivates a work-based view of credit: just as work depends on the component of force along a displacement, token credit should reflect the local reward change along the policy update direction. We introduce Direction-Aware Token Credit Assignment (DATC), which maps frozen reward-model gradients into policy logit space and allocates weights by evaluating the alignment between the reward direction and GRPO's signed sampled-action update direction, while preserving the sign and mean of the response-level advantage. DATC outperforms GRPO in all four settings across two model families.
    
[^32]: 在线策略自训练：大语言模型的无数据自我改进

    On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models

    [https://arxiv.org/abs/2610.32770](https://arxiv.org/abs/2610.32770)

    该论文提出了一种无需任何数据的在线策略自训练框架，大语言模型仅需粗糙的高层主题描述即可自生成问题-答案对和特权推理演示进行自我改进，在多个领域提升能力的同时保留原有领域的性能。

    

    在线策略自蒸馏使得能够在不依赖更强外部教师的情况下对基础模型进行后训练。然而，现有方法依赖于真实的问题-答案对、人工策划的问题集或其他监督信号。在这项工作中，我们将在线策略自蒸馏重新表述为一个自训练框架，由自生成的问题-答案对和特权推理演示驱动。我们将监督信号推向极限：仅由粗糙的高层级主题描述引导，模型在无需访问任何人工策划的提示或答案的情况下，自行合成查询课程和特权推理轨迹。我们发现，在完全合成的特权信息上进行自蒸馏，能够使模型在多个领域获得改进，同时保留其在未进行自训练的领域上的能力。我们展示了该框架在文本和多模态领域的通用性，并取得了一致的（效果）。

    arXiv:2610.32770v1 Announce Type: cross  Abstract: On-policy self-distillation enables post-training foundation models without relying on stronger external teachers. However, existing approaches rely on ground-truth question-answer pairs, human-curated problem sets, or other supervised signals. In this work, we reformulate on-policy self-distillation as a self-training framework, driven by self-generated question-answer pairs and privileged reasoning demonstrations. We push supervision to the limit: guided solely by coarse, high-level topic descriptions, the model synthesizes its own curriculum of queries and privileged reasoning traces without access to any human-curated prompts or answers. We find that self-distillation on fully synthetic privileged information enables a model to improve across multiple domains while retaining its capabilities on domains it is not self-trained on. We demonstrate the generality of this framework across text and multimodal domains, achieving consistent
    
[^33]: 车联网中联邦学习的干净标签投毒攻击与防御博弈

    Clean-label Poisoning Attack and Defense Game for Federated Learning in Vehicular Networks

    [https://arxiv.org/abs/2610.32765](https://arxiv.org/abs/2610.32765)

    该论文针对车联网联邦学习提出了一种特征空间干净标签投毒攻击与特征一致性重标记防御方法，并通过博弈论框架刻画攻防均衡策略，在不损害正常检测准确率的情况下将攻击成功率维持在低水平。

    

    本文研究了车联网中使用联邦学习（FL）时干净标签投毒攻击与防御之间的交互。基本安全消息（BSM）的异常检测分类被建模为在具有动态拓扑结构的车联网上的分布式学习问题。作为联邦学习面临的关键威胁，本文引入了一种特征空间的干净标签攻击，该攻击将良性样本向恶意特征极端方向扰动，在保持标签不变的同时降低类间可分性。为应对这一攻击，提出了一种特征一致性重标记防御方法，用于检测并重新标记可疑样本。此外，构建了一个博弈论框架来刻画攻防双方之间的交互，并基于攻击有效性与成本参数来表征均衡策略。结果表明，该攻击随被攻陷节点数量的增加而有效扩展，而所提出的防御方法能够在不降低干净样本准确率的前提下将攻击成功率保持在较低水平，并使均衡策略发生……（原文摘要截断于此）

    arXiv:2610.32765v1 Announce Type: cross  Abstract: This paper studies interactions between clean-label poisoning attack and defense in vehicular networks using federated learning (FL). Basic Safety Message (BSM) classification for anomaly detection is modeled as a distributed learning problem over a vehicular network with dynamic topology. As a critical threat to FL, a feature-space clean-label attack is introduced, wherein benign samples are perturbed toward malicious feature extremes, degrading class separability while preserving labels. To counter this, a feature-consistency relabeling defense is proposed to detect and relabel suspicious samples. A game-theoretic framework is developed to capture attack-defense interactions and characterize equilibrium strategies based on attack effectiveness and cost parameters. Results show that the attack scales effectively with compromised nodes, while the proposed defense keeps attack success low without degrading clean accuracy and shifts equi
    
[^34]: ARENA：基于活动区注意力聚合的可解释太阳耀斑预报

    ARENA: Active Region attENtion Aggregation for interpretable solar flare prediction

    [https://arxiv.org/abs/2610.32758](https://arxiv.org/abs/2610.32758)

    ARENA是一种基于多示例学习的弱监督神经网络架构，通过注意力聚合机制从数量可变的活动区切片预测太阳耀斑，并能可解释地定位驱动预测的关键活动区，其Heidke技巧评分达0.45，与其他耀斑预测方法相当。

    

    预测太阳耀斑是一个具有挑战性的空间天气预报问题，其对地球高层大气可能产生重大影响，包括对无线电通信的干扰。我们提出了活动区注意力聚合（ARENA），这是一种神经网络架构，能够从数量可变的活动区切片中预测太阳耀斑，同时学习识别与预测最相关的区域。ARENA采用弱监督下的多示例学习进行训练，仅需观测对应的耀斑/无耀斑标签，无需标注耀斑起源位置。因此，学习到的注意力机制提供了一种无需显式起源标注即可定位驱动预测的活动区的方法。ARENA的Heidke技巧评分达到0.45，与其他耀斑预测方法相当。这些结果表明，基于活动区的聚合方法能够提供一种高效且可解释的替代方案。

    arXiv:2610.32758v1 Announce Type: cross  Abstract: Predicting solar flares is a challenging space-weather forecasting problem with potentially significant impacts on Earth's upper atmosphere, including disruptions to radio communications. We propose Active Region attENtion Aggregation (ARENA), a neural network architecture that predicts solar flares from a variable number of active-region cutouts while learning to identify the region most relevant to the prediction. ARENA is trained using multiple-instance learning under weak supervision, requiring only a flare/no-flare label for the observation and no labels specifying the flare origin. The learned attention mechanism therefore provides a means of localizing the active region driving the prediction without explicit origin annotations. ARENA achieves a heidke skill score of 0.45 which is comparable to other flare prediction approaches. These results demonstrate that active-region-based aggregation can provide an efficient and interpret
    
[^35]: 基于粘性分段确定性马尔可夫过程的稀疏贝叶斯神经网络

    Sparse Bayesian Neural Networks via Sticky Piecewise Deterministic Markov Processes

    [https://arxiv.org/abs/2610.32753](https://arxiv.org/abs/2610.32753)

    该论文提出利用粘性分段确定性马尔可夫过程结合尖峰-平板先验，在贝叶斯神经网络的采样过程中将权重精确置零，实现渐近精确的稀疏贝叶斯神经网络采样与剪枝。

    

    越来越多的证据表明，神经网络的性能在很大程度上取决于其权重中的一小部分，这些权重通常通过训练一个密集网络然后进行剪枝来识别。在贝叶斯框架内，剪枝可以通过尖峰-平板先验直接纳入，从而对所涉及的不确定性进行恰当处理，包括剪枝不确定性和参数不确定性。在贝叶斯神经网络（BNN）中，通过MCMC进行采样是出了名的困难。我们证明了基于分段确定性马尔可夫过程（PDMP）的现代蒙特卡洛方法，只要数值构造的速率界成立，就可以渐近精确地采样BNN的后验分布。尖峰-平板先验可以通过粘性版本的PDMP纳入，在采样过程中将权重精确地设置为零。我们检验了这些采样器的预测性能和收敛性，并表明以这种方式采样的稀疏BNN…

    arXiv:2610.32753v1 Announce Type: new  Abstract: There is growing evidence that the performance of a neural network depends largely on a small subset of its weights, which are typically identified by training a dense network and pruning it afterwards. Within a Bayesian framework, pruning can be directly incorporated through a spike-and-slab prior, providing a proper treatment of the uncertainties involved, both with respect to pruning and to parameter uncertainty. In Bayesian neural networks (BNNs), sampling by MCMC is notoriously hard. We demonstrate that modern Monte Carlo methods based on piecewise deterministic Markov processes (PDMPs) can be used to sample posteriors of BNNs asymptotically exact whenever numerically constructed rate bounds hold. Spike-and-slab priors can be incorporated through sticky versions of the PDMPs, setting weights exactly to zero during sampling. We examine the predictive performance and convergence of the samplers, and show that sparse BNNs sampled this 
    
[^36]: 论混合深度生成模型的可辨识性：元学习作为解决方案

    On the Identifiability of Hybrid Deep Generative Models: Meta-Learning as a Solution

    [https://arxiv.org/abs/2610.32748](https://arxiv.org/abs/2610.32748)

    本文首次从理论上探究了融合物理知识的混合深度生成模型的可辨识性问题，并提出元学习作为构建具有可辨识性保证的混合深度生成模型的新颖解决方案。

    

    在深度学习中利用基于物理的归纳偏置的研究兴趣，推动了混合深度生成模型（hybrid-DGMs）的最新发展，这类模型将已知的基于物理的数学表达式集成到神经生成模型中。识别这些混合深度生成模型需要同时推断基于物理组件的参数以及神经组件的参数。然而，这些混合深度生成模型的可辨识性尚未在理论上得到探究或证实。现有关于一般深度生成模型不可辨识性的理论如何适用于混合深度生成模型？什么可能是构建具有理论可辨识性证明的混合深度生成模型的有效方法？本文首次对混合深度生成模型的可辨识性进行了理论探究，并提出元学习作为构建可辨识混合深度生成模型的一种新颖解决方案。在合成数据和真实数据基准上，我们为混合深度生成模型的不可辨识性提供了强有力的实证证据。

    arXiv:2610.32748v1 Announce Type: new  Abstract: The interest in leveraging physics-based inductive bias in deep learning has resulted in recent development of hybrid deep generative models (hybrid-DGMs) that integrate known physics-based mathematical expressions in neural generative models. To identify these hybrid-DGMs requires inferring parameters of the physics-based component along with their neural component. The identifiability of these hybrid-DGMs, however, has not yet been theoretically probed or established. How does the existing theory of the un-identifiability of general DGMs apply to hybrid-DGMs? What may be an effective approach to construct a hybrid-DGM with theoretically-proven identifiability? This paper provides the first theoretical probe into the identifiability of hybrid-DGMs, and presents meta-learning as a novel solution to construct identifiable hybrid-DGMs. On synthetic and real-data benchmarks, we provide strong empirical evidence for the un-identifiability of
    
[^37]: 具有时变移动代价的强自适应在线学习

    Strongly Adaptive Online Learning with Time-Varying Movement Costs

    [https://arxiv.org/abs/2610.32740](https://arxiv.org/abs/2610.32740)

    本文提出了一种针对时变移动代价在线凸优化的强自适应算法，通过新颖地归约到此前未被探索的带移动代价的睡眠专家问题，在任意时间区间上实现了数据依赖的遗憾界。

    

    arXiv:2610.32740v1 公告类型：cross 摘要：我们研究了具有时变移动代价的在线凸优化（OCO）的强自适应遗憾，该设定用于评估学习者在任意子区间上的性能。这一框架刻画了复杂的序列决策问题，其中环境与切换决策的代价均随时间波动。我们提出了一种新颖的算法，对于任意区间 $[a,b] \subseteq [T]$，可实现数据依赖的遗憾界 $\widetilde{\mathcal{O}}\left( \sqrt{(D^2+DP_{[a,b]})\sum_{t \in [a,b]}(\|g_t\|^2+\lambda_t\|g_t\|)} \right)$，其中 $D$ 为决策域的直径，$P_{[a,b]}$ 为比较序列在 $[a,b]$ 上的路径长度，$g_t$ 为损失的梯度，$\lambda_t$ 为第 $t$ 轮的移动代价系数。我们方法的关键在于一种新颖的归约，将问题转化为带移动代价的睡眠专家问题，这是一个此前未被探索过、且具有独立研究价值的变体。作为……

    arXiv:2610.32740v1 Announce Type: cross  Abstract: We study strongly adaptive regret for Online Convex Optimization (OCO) with time-varying movement costs, a setting that evaluates a learner's performance over arbitrary sub-intervals. This framework captures complex sequential decision-making problems where both the environment and the cost of switching decisions fluctuate over time. We propose a novel algorithm which, for any interval $[a,b] \subseteq [T]$, achieves a data-dependent regret bound of $ \widetilde{\mathcal{O}}\left( \sqrt{(D^2+DP_{[a,b]})\sum_{t \in [a,b]}(\|g_t\|^2+\lambda_t\|g_t\|)} \right) $, where $D$ is the diameter of the decision domain, $P_{[a,b]}$ is the path length of the comparator sequence over $[a,b]$, $g_t$ is the gradient of the loss, and $\lambda_t$ is the movement-cost coefficient at round $t$. Key to our approach is a novel reduction to sleeping experts with movement costs, a previously unexplored variant that is of independent interest. As a result of 
    
[^38]: 对比学习的内部差异：泛化界与下游行为

    Internal Discrepancy for Contrastive Learning: Generalization Bounds and Downstream Behaviors

    [https://arxiv.org/abs/2610.32729](https://arxiv.org/abs/2610.32729)

    本文提出“内部差异”度量来统一刻画对比学习中的各种理论假设，证明了基于该度量的下游监督任务泛化界，并提出松弛k均值损失作为减小内部差异的近似方法。

    

    对比学习已成为训练表示模型的一种流行范式。尽管先前的研究致力于寻求更实用、限制更少的假设以增强对对比学习的理论理解，但这些假设本质上描述的是数据内部表示之间或由函数类所诱导的表示之间的相互联系。为了统一概括这些假设，我们提出了“内部差异”这一度量方法，用于量化正样本的平均表示与潜在聚类中心之间的差异。随后，我们在无需额外假设的情况下，证明了对比学习之后下游监督任务的内部差异感知泛化界。这一泛化界不仅作为连接各种假设的统一框架，还激励我们进一步减小内部差异，从而引入了松弛k均值损失作为其近似。此外，

    arXiv:2610.32729v1 Announce Type: cross  Abstract: Contrastive learning has emerged as a popular paradigm for training representation models. While prior studies pursue more practical and less restrictive assumptions to enhance theoretical understanding of contrastive learning, they essentially describe the interconnections among representations within the data or induced by the function class. To encapsulate these assumptions, we propose the measure of internal discrepancy that quantifies the difference between the mean representation of positive samples and latent cluster centers. We then prove an internal discrepancy-aware generalization bound for downstream supervised tasks following contrastive learning, without requiring additional assumptions. This generalization bound not only serves as a unifying framework connecting various assumptions but also motivates us to further reduce the internal discrepancy, thereby introducing the relaxed k-means loss as an approximation. Moreover, 
    
[^39]: PlayerEmbed：从原始输入序列中学习行为指纹表示

    PlayerEmbed: Learning Behavioral Fingerprint Representations from Raw Input Sequences

    [https://arxiv.org/abs/2610.32727](https://arxiv.org/abs/2610.32727)

    该论文提出PlayerEmbed框架，利用深度度量学习和混合CNN-LSTM编码器从原始玩家输入序列中学习紧凑的行为指纹嵌入，在玩家识别任务中对已见和未见玩家分别达到0.950和0.879的平衡准确率，且增加训练玩家数量可显著提升对未见玩家的泛化能力。

    

    电子游戏中的原始玩家输入序列包含细粒度的行为信息，但其高维和噪声特性使得大规模分析充满挑战。为了解决这一问题，我们提出了PlayerEmbed，一个利用深度度量学习来学习紧凑且具有判别性的玩家嵌入表示的框架。我们的方法采用混合CNN-LSTM编码器，在来自1,000名玩家的原始输入序列上进行训练。我们在这些玩家的保留序列上，以及在来自训练期间完全未见过的1,000名玩家的序列上对模型进行评估，以检验其对未见身份的泛化能力。在玩家识别任务中，基于参考嵌入构建的最近质心分类器取得了强劲性能，对训练期间见过的玩家达到0.950的平衡准确率，对训练期间未见过的玩家达到0.879的平衡准确率。系统性增加训练玩家数量能够显著提升对未见玩家的泛化能力，表明从……

    arXiv:2610.32727v1 Announce Type: cross  Abstract: Raw player input sequences in video games contain fine-grained behavior, but are high-dimensional and noisy, which makes large-scale analytics challenging. To address this, we introduce PlayerEmbed, a framework that learns compact and discriminative player embeddings using deep metric learning. Our approach uses a hybrid CNN-LSTM encoder trained on raw input sequences from 1,000 players. We evaluate the model on held-out sequences from these players and on sequences from 1,000 players entirely unseen during training to assess generalization to unseen identities. For player identification, a nearest-centroid classifier built from reference embeddings achieves strong performance, reaching balanced accuracies of 0.950 for players seen during training and 0.879 for players unseen during training. Systematically increasing the number of training players substantially improves generalization to unseen players, indicating a transition from si
    
[^40]: 通过流匹配从真实世界驾驶数据实现跨场景控制器参数转换

    Cross-Scenario Controller Parameter Transformation from Real-World Driving Data through Flow Matching

    [https://arxiv.org/abs/2610.32719](https://arxiv.org/abs/2610.32719)

    该论文的核心创新是将ACC控制器参数适配形式化为驾驶员个性化的场景间传输问题，利用流匹配生成模型，仅凭少量源场景演示即可为陌生驾驶员预测并生成目标场景下的控制器参数，实现跨驾驶场景的个性化参数转换。

    

    高级驾驶辅助系统（ADAS）日益普及，旨在为驾驶员提供支持、降低工作负荷并提升交通安全。然而，许多系统很少被使用，原因在于其行为感觉不自然，与驾驶员个人的驾驶风格不匹配。大规模真实世界人类驾驶数据的日益可得，使得开发更类人化、面向个体驾驶员的辅助功能成为可能。本文将自适应巡航控制（ACC）参数的适配问题形式化为一个面向个体驾驶员的场景到场景的传输问题，即将控制器参数从一种驾驶情境转换到另一种驾驶情境。利用流匹配技术，我们训练了一个生成模型，仅需源场景中的少量演示即可预测未见过的驾驶员在目标场景中的行为。我们将该方法与线性参数偏移基线进行了对比，并评估了融合驾驶员一致性耦合、驾驶员嵌入等机制的流匹配变体。

    arXiv:2610.32719v1 Announce Type: cross  Abstract: Advanced Driver Assistance Systems have become increasingly widespread, aiming to support drivers, reduce workload, and improve traffic safety. However, many systems are rarely used because their behavior feels unnatural and does not match individual driving behavior. The growing availability of large-scale, real-world human driving data enables more human-like and driver-individual assistance functions. This paper formulates the adaptation of Adaptive Cruise Control parameters as a driver-individual scenario-to-scenario transport problem, transforming controller parameters from one driving context to another. Using flow matching, we train a generative model to predict the target-scenario behavior of previously unseen drivers from only a few demonstrations in a source scenario. We compare the approach with a linear parameter-shift baseline and evaluate flow-matching variants incorporating driver-consistent coupling, driver embeddings, 
    
[^41]: 任务偏移下的保得分持续Lasso

    Score-Preserving Continual Lasso under Task Shift

    [https://arxiv.org/abs/2610.32708](https://arxiv.org/abs/2610.32708)

    提出保得分持续Lasso（SP-CL），通过保留历史平方损失得分将稀疏估计与状态传递分离，避免了递归复用 ℓ₁ 估计量导致的正则化偏差跨任务累积，并在平方损失下精确恢复全历史Lasso解与稀疏率。

    

    持续学习旨在从序列任务中进行学习，同时在历史数据不再可访问的情况下缓解灾难性遗忘。我们研究高维稀疏线性回归中的持续学习问题，其中 ℓ₁ 正则化带来了持续岭回归中所不存在的困难：递归地复用 ℓ₁ 惩罚估计量会传播早期的收缩效应，并在任务间累积正则化偏差。我们提出保得分持续Lasso（SP-CL），该方法通过保留历史平方损失得分，将稀疏估计与状态传递分离开来。在平方损失下，其累积估计量能够精确恢复全历史Lasso的极小值解。当任务系数一致时，它继承了全历史的稀疏率。对于相关的任务特定系数，SP-CL 采用一种正则化的当前任务修正，该修正不会改变所传递的状态，并达到离线源辅助（截断）

    arXiv:2610.32708v1 Announce Type: new  Abstract: Continual learning (CL) aims to learn from sequential tasks while mitigating catastrophic forgetting when historical data are no longer accessible. We study continual learning for high-dimensional sparse linear regression, where $\ell_1$ regularization creates a difficulty absent from continual ridge regression: that recursively reusing an $\ell_1$-penalized estimator can propagate earlier shrinkage and accumulate regularization bias across tasks. We propose score-preserving continual Lasso (SP-CL), which separates sparse estimation from state transmission by preserving the historical squared-loss score. Under squared loss, its cumulative estimator exactly recovers a full-history Lasso minimizer. When task coefficients coincide, it inherits the full-history sparse rate. For related task-specific coefficients, SP-CL uses a regularized current-task correction that leaves the transmitted state unchanged and attains the offline source-assist
    
[^42]: 探究晶体材料正向筛选与逆向设计的极限

    Probing the limits of forward screening and inverse design of crystal materials

    [https://arxiv.org/abs/2610.32678](https://arxiv.org/abs/2610.32678)

    该研究将55个晶体数据库按结构匹配整合为包含1460万种晶体和1.318亿性能值的统一数据库，揭示晶体-性能数据仅有4.5%被填充，并开发了高精度等变图变换器CGTNet构建性能模型库来填补数据空缺，以此探究晶体材料正向筛选与逆向设计的能力极限。

    

    晶体材料的设计要么通过在已知晶体中筛选具有目标性能的材料，要么通过生成具有该性能的新晶体。这两种途径都以已知晶体的公共数据库作为评判标准，但这些数据库对同一晶体的命名方式各不相同。近期多个生成式与自主发现项目所宣称的新材料，事后被证明是已知物相或其近似变体。这两种途径的能力极限至今仍未被测量。本文通过将55个数据库中的晶体按结构匹配，构建了一个可自我扩展的数据库，包含1460万种晶体和1.318亿个性能值，且每个数值均可追溯至其来源。该数据库中仅有4.5%的晶体-性能对含有数值。为填补这些空缺，我们开发了CGTNet——一种等变图变换器，在两个公开基准的若干任务上达到了最先进的精度。针对每个任务训练后，它构成了一个性能模型库，我们已发布其预测结果。

    arXiv:2610.32678v1 Announce Type: cross  Abstract: Crystal materials are designed either by screening known crystals for a target property or by generating new crystals with that property. Both routes are judged against public databases of known crystals, but these databases name the same crystal differently. Several materials announced as new by recent generative and autonomous discovery campaigns have since proved to be known phases or close variants of them. The limits of both routes remain unmeasured. Here we probe them by matching the crystals of 55 databases by structure into one self-extending database of 14.6 million crystals and 131.8 million property values, each traced to its source. Only 4.5% of its crystal-property pairs carry a value. To fill the empty pairs, we developed CGTNet, an equivariant graph transformer with state-of-the-art accuracy on several tasks of two public benchmarks. Trained on each task, it forms a library of property models whose predictions we release
    
[^43]: 肌电信号标记化以极小比特率保留运动信息

    EMG Tokenization Preserves Movement Information at a Fraction of the Bitrate

    [https://arxiv.org/abs/2610.32674](https://arxiv.org/abs/2610.32674)

    该研究提出无需姿态标签的EMG标记化方法，在将数据负载降低约41倍的情况下，三维手部姿态解码精度几乎不受影响。

    

    无线肌电图（EMG）接口必须在电极覆盖范围与传输带宽之间取得平衡。本研究表明，紧凑的EMG标记在大幅减少数据负载的同时，能够保留用于三维手部姿态解码的信息。我们评估了两种无需姿态标签即可训练的因果标记器：一种是预训练用于重建EMG频谱的编码器，另一种基于SPAR——一种针对邻近电极的工程化频谱-空间表示。在使用九名参与者320通道EMG的个性化、会话内实验中，我们采用匹配的解码器架构和训练流程，对标记化特征与连续特征进行了比较。在各解码器上取平均后，标记化使SPAR的平均误差变化为+0.01毫米（95%置信区间：-0.12至0.14），学习型表示为+0.45毫米（0.10至0.82）。两者的误差增加均满足1毫米的非劣效性标准。存储的标记阵列相对于原始EMG数据将负载降低了41倍，并且……

    arXiv:2610.32674v1 Announce Type: cross  Abstract: Wireless electromyography (EMG) interfaces must balance electrode coverage against transmission bandwidth. Here, we show that compact EMG tokens substantially reduce payload while retaining information for three-dimensional hand-pose decoding. We evaluate two causal tokenizers fitted without pose labels: an encoder pretrained to reconstruct EMG spectra and one based on SPAR, an engineered spectral-spatial representation of neighbouring electrodes. In personalized, within-session experiments with 320-channel EMG from nine participants, we compare tokenized and continuous features using matched decoder architectures and training procedures. Averaged across decoders, tokenization changed mean error by +0.01 mm for SPAR (95% confidence interval: -0.12 to 0.14) and +0.45 mm for the learned representation (0.10 to 0.82). Both increases met a 1 mm non-inferiority criterion. Stored token arrays reduced payloads 41-fold relative to raw EMG and 
    
[^44]: 语言模型相信自己的解释吗？一项关于功能一致性的研究

    Do Language Models Trust Their Own Justifications? A Study on Functional Consistency

    [https://arxiv.org/abs/2610.32665](https://arxiv.org/abs/2610.32665)

    该研究提出基于“功能自一致性”原则的实验框架，通过分离、移除或语义反转模型所标记的关键词元，检验语言模型生成的解释是否与其真实预测行为在功能上保持一致。

    

    大型语言模型（LLMs）已被广泛应用于文本分类任务，它们不仅输出类别预测，还会生成解释，突出显示被认为对得出预测标签最相关的词元。然而，这些被突出的元素在针对性干预下是否与模型自身的预测在行为上保持一致——我们将其称为“功能自一致性”——目前仍不清楚。尽管大量文献评估了此类解释的文本合理性，但很少有研究评估其与模型实际行为之间的功能一致性。在本工作中，我们提出了一个基于自一致性原则的实验框架：如果模型认为某些词元是决定性的，那么分离、移除或语义反转这些词元，应当会在其预测中产生系统性的、可解释的变化。我们通过充分性和完整性等指标将这一评估操作化……

    arXiv:2610.32665v1 Announce Type: new  Abstract: Large Language Models (LLMs) have been widely adopted in text classification tasks, where they not only output class predictions but also generate explanations that highlight the tokens deemed most relevant for reaching the predicted label. Yet it remains unclear whether these highlighted elements are behaviorally consistent with the model's own predictions under targeted interventions, which we refer to as functional auto-consistency. While much of the literature evaluates the textual plausibility of such explanations, few studies assess their functional consistency with the model's actual behavior. In this work, we propose an experimental framework based on the principle of auto-consistency: if a model identifies certain tokens as decisive, then isolating, removing, or semantically inverting them should produce systematic and interpretable changes in its predictions. We operationalize this evaluation through sufficiency, comprehensiven
    
[^45]: DT-JEPA：学习预测未来表征以进行决策

    DT-JEPA: Learning to Predict Future Representations for Decision-Making

    [https://arxiv.org/abs/2610.32657](https://arxiv.org/abs/2610.32657)

    提出DT-JEPA，将联合嵌入预测架构（JEPA）与决策Transformer相结合，通过在潜在空间中预测未来动态表征来替代以预期回报作为条件，从而克服回报条件描述未来的不完整性，并摆脱对奖励标注数据集的依赖。

    

    决策Transformer（DT）通过以预期未来回报和历史轨迹为条件来生成动作，在各种离线强化学习（RL）任务中展现出了强大的性能。然而，将预期未来回报作为条件对未来提供了不完整的描述，因为具有不同状态转移和行为的轨迹可能拥有相似的回报。此外，预期未来回报本质上依赖于任务特定的奖励函数，这使得策略学习依赖于带有奖励标注的数据集，并限制了其向具有相似但未见过的奖励的任务扩展的能力。最近，联合嵌入预测架构（JEPA）已成为一种有前景的基于自监督学习的世界模型范式，它在潜在空间中学习未来动态的预测表征，而不是在原始空间中重建未来状态。受JEPA启发，我们提出了DT-JEPA

    arXiv:2610.32657v1 Announce Type: new  Abstract: Decision Transformer (DT) has demonstrated strong performance in various offline reinforcement learning (RL) tasks by conditioning action generation on expected future returns and historical trajectories. However, treating expected future returns as the condition provides an incomplete description of the future, as trajectories with different state transitions and behaviors may share similar returns. Moreover, expected future returns inherently depend on task-specific reward functions, which makes policy learning reliant on reward-labeled datasets and restricts its scalability to tasks with similar but unseen rewards. Recently, Joint Embedding Predictive Architectures (JEPA) have emerged as a promising paradigm of self-supervised learning-based world models, which learn predictive representations of future dynamics in latent space rather than reconstructing future states in the original space. Inspired by JEPA, we propose DT-JEPA, an off
    
[^46]: WL-SeqVAE：基于序列的图生成方法——利用变分自编码器与Weisfeiler-Leman颜色

    WL-SeqVAE: Sequence-Based Graph Generation Using Variational Autoencoders and Weisfeiler-Leman Colors

    [https://arxiv.org/abs/2610.32650](https://arxiv.org/abs/2610.32650)

    该论文提出WL-SeqVAE，将图转换为Weisfeiler-Leman颜色序列并用变分自编码器和自回归解码器建模，把图生成转化为序列生成问题，同时借助WL颜色兼容性约束保证局部结构一致性，从而避免直接参数化大规模邻接矩阵。

    

    本文提出了WL-SeqVAE，一种用于图生成的基于序列的变分自编码器。WL-SeqVAE背后的关键观察是：由Weisfeiler-Leman（WL）算法诱导的节点颜色为局部图结构提供了紧凑的离散摘要，因此可以作为生成过程中的结构化标记。我们的方法并不直接学习邻接矩阵上的分布，而是将每个图映射为WL颜色序列，并采用变分自编码器将该序列编码为高斯潜在分布，再由自回归解码器从该分布中重构出图。在生成过程中，由WL颜色导出的兼容性约束引导节点和边的预测，将来自潜在表示的全局信息与局部结构一致性相结合。这一方法将图生成问题转化为序列生成问题，避免了对二次方规模的邻接矩阵进行直接参数化。

    arXiv:2610.32650v1 Announce Type: new  Abstract: In this paper, we introduce WL-SeqVAE, a sequence-based variational autoencoder for generating graphs. The key observation behind WL-SeqVAE is that node colors induced by the Weisfeiler-Leman (WL) algorithm provide compact discrete summaries of local graph structure and can therefore serve as structural tokens for generation. Rather than learning a distribution over adjacency matrices directly, our method maps each graph to a sequence of WL colors and employs a variational autoencoder that encodes this sequence into a Gaussian latent distribution, from which an autoregressive decoder reconstructs the graph. During generation, compatibility constraints derived from the WL colors guide node and edge predictions, combining global information from the latent representation with local structural consistency. This formulation turns graph generation into a sequence generation problem and avoids directly parameterizing a quadratic adjacency matr
    
[^47]: VICReg 中的标准差提供了安全边际

    The Standard Deviation in VICReg Buys a Safety Margin

    [https://arxiv.org/abs/2610.32630](https://arxiv.org/abs/2610.32630)

    本文首次通过实验验证了 VICReg 采用标准差而非方差作为防坍塌正则项的合理性——标准差铰链因其梯度在特征方差缩小时不会过早消失，其坍塌阈值比方差铰链低 35-130 倍，从而提供了关键的安全边际。

    

    VICReg 通过对每个嵌入维度的标准差施加铰链（hinge）惩罚来防止表征坍塌，其作者曾从梯度角度论证，若改用方差作为铰链则会失效。据我们所知，这一论断尚未经过实验验证。我们在受控环境中（ResNet-18，CIFAR-10 和 CIFAR-100）对此进行了测试，仅改变铰链项，并将其权重 $\mu$ 在五个数量级范围内进行扫描。在默认权重（$\mu = 1$）下，两种铰链在 CIFAR-10 上达到相同的线性探针准确率（约 89.5%）。当权重低于某个阈值时两者都会坍塌，但标准差铰链的坍塌阈值低了 35-130 倍。CIFAR-100 上也显示出一致的差距。在训练过程中，方差铰链的梯度会随着特征方差的缩小而衰减，而标准差铰链的梯度直到方差非常小时才会衰减。因此，标准差铰链提供了安全边际：它能在远低于方差铰链失效点的权重下仍维持有效训练。

    arXiv:2610.32630v1 Announce Type: new  Abstract: VICReg prevents collapse with a hinge on the standard deviation of each embedding dimension, and its authors argue from the gradient that a hinge on the variance would fail. To our knowledge, this claim has not been tested experimentally. We test it in a controlled setting (ResNet-18, CIFAR-10 and CIFAR-100), changing only the hinge and sweeping its weight $\mu$ over five orders of magnitude. At the default weight ($\mu = 1$), the two hinges reach the same linear-probe accuracy on CIFAR-10 (about 89.5%). Both collapse below a threshold weight, but the standard-deviation hinge's threshold is 35-130$\times$ lower. CIFAR-100 shows a consistent gap. In training, the variance hinge's gradient fades as the feature variance shrinks, while the standard-deviation hinge's does not until that variance is very small. The standard-deviation hinge therefore buys a safety margin: it keeps training at weights far below the point where the variance hinge
    
[^48]: 立方体上球面-线性贝叶斯优化的采集函数

    Acquisition for Spherical-Linear Bayesian Optimization on the Cube

    [https://arxiv.org/abs/2610.32628](https://arxiv.org/abs/2610.32628)

    本文分析了单位立方体上各类贝叶斯优化采集子问题全局求解的复杂度，并提出了一种新的采集函数，其全局优化耗时远少于现有多起点拟牛顿方法，同时保持或提升了优化性能。

    

    近期研究表明，仅使用线性代理模型并配合巧妙的变换来执行贝叶斯优化（BO），就足以在高维测试函数上逼近基于标准高斯过程（GP）方法的性能。当决策区域为球体时，该策略可以实现快速精确的采集。然而，在贝叶斯优化文献中最常遇到的决策区域是单位立方体，在这种情形下，通常采用标准方法（即多起点拟牛顿法）来求解采集子问题。在本文中，我们将讨论关于单位立方体上各种采集子问题全局求解复杂度的一系列正反两方面的结果。在此讨论的基础上，我们将提出一种新的采集函数，其全局优化所需的执行时间显著少于现有方法，同时能够保持或提高性能。

    arXiv:2610.32628v1 Announce Type: new  Abstract: Recent work demonstrated that executing Bayesian Optimization (BO) using simply a linear surrogate with a clever transformation is enough to approximate the performance of the standard Gaussian-Process (GP) based version on high-dimensional test functions. When the decision region is a sphere, this strategy admits rapid exact acquisition. However, it is the unit cube we encounter most often in the BO literature, and in this setting a standard approach, namely a multi-start quasi-Newton method, is employed for solving the acquisition subproblem. In this article, we will discuss a series of results, both positive and negative, regarding the complexity of global resolution of various acquisition subproblems over the unit cube. On the basis of this discussion, we will propose a new acquisition function which is globally optimizable using significantly less execution time than the incumbent approach while maintaining or improving performance.
    
[^49]: 地球嵌入向量是陆地表面的可分解记录

    Earth embeddings are decomposable records of the land surface

    [https://arxiv.org/abs/2610.32617](https://arxiv.org/abs/2610.32617)

    本研究首次分解了两种广泛使用的年度地球嵌入向量产品，证明其中包含火灾、森林砍伐和作物类型等可独立验证的生态特征，并能定位扰动发生时间及区分火灾严重程度。

    

    生态监测、碳核算和生物多样性评估日益依赖于年度地球嵌入向量——这是一种通过学习得到的向量，将地球上每个十米像素一整年的卫星观测信息压缩其中，用以替代原始影像本身。然而，这些向量的含义至今尚未被确立。本文对两种广泛使用的嵌入产品进行了分解，并表明它们包含可单独识别的生态特征，包括火灾、森林砍伐和作物类型，并通过来自六大洲事件的独立参考数据进行了验证。所恢复出的概念能够将突然的冠层损失定位于扰动后的第一个生长季，且对严重程度的分辨能力足以区分计划烧除与野火。快速恢复的小型火灾可以在其中一种产品中被检测到，而在另一种产品中则无法检测到，后者反而对作物类型的区分能力更强。

    arXiv:2610.32617v1 Announce Type: new  Abstract: Ecological monitoring, carbon accounting and biodiversity assessment increasingly rely on annual Earth embeddings--learned vectors that compress a year of satellite observation for every ten-meter pixel on Earth, consumed in place of the imagery itself. What these vectors mean has not been established. Here we decompose two widely used embedding products and show that they contain individual ecological features, including fire, forest clearing and crop type, validated against independent reference data on events across six continents. The recovered concepts date abrupt canopy loss to the first post-disturbance growing season and resolve severity well enough to separate prescribed fire from wildfire. Small fires that recover quickly are detectable in one product but not the other, which instead separates crops more strongly.
    
[^50]: 解读监督式成对图表示：局部证据、混杂因素与可证明的事后修正

    Interpreting Supervised Pairwise Graph Representations: Local Evidence, Confounding, and Provable Post-hoc Correction

    [https://arxiv.org/abs/2610.32611](https://arxiv.org/abs/2610.32611)

    该研究通过受控实验与混杂因素分析揭示，监督式成对图表示的表面性能优势可能源于序列长度等混杂因素而非高阶图拓扑本身，并提出可证明的事后修正方法来区分真实的局部证据与虚假关联。

    

    监督式成对图表示将成对预测关系转化为样本特定的图结构，但图级别的准确率本身并不能证明高阶拓扑结构承载了预测信号。在一个受控的Transformer错误检测任务中，监督式成对图的性能优于“注意力即邻接矩阵”方法（ROC-AUC为0.839对0.559；可比的无置信度版本为0.687对0.559），但在控制序列长度之后，它们相对于模型置信度的表面优势消失了——序列长度几乎可以从池化隐藏状态中被完美恢复（R²=0.9995）。在已发表的HCP Emotion集成图上，带符号的节点聚合本身已经起到了监督式成对预测集成的作用。仅凭边幅值进行预测接近随机水平；拓扑感知的增强方法对ROC-AUC的改变最多只有0.004，其中只有符号分离度在所有测试的图规模上显示出微小的正向变化，而经过验证集调优的三（原文摘要在此处截断）

    arXiv:2610.32611v1 Announce Type: new  Abstract: Supervised pairwise graph representations convert pairwise predictive relations into sample-specific graphs, but graph-level accuracy does not by itself show that higher-order topology carries the signal. In a controlled Transformer error-detection task, supervised pairwise graphs outperform attention-as-adjacency (ROC-AUC $0.839$ versus $0.559$; the comparable no-confidence pair is $0.687$ versus $0.559$), but their apparent advantage over model confidence disappears after controlling for sequence length, which is almost perfectly recoverable from the pooled hidden state ($R^2=0.9995$). On published HCP Emotion ensemble graphs, signed node aggregation already acts as an ensemble of supervised pairwise predictions. Edge magnitude alone is near chance; topology-aware augmentations change ROC-AUC by at most $0.004$, with only sign-separated degree showing small positive changes across all tested graph sizes, while a validation-tuned three-
    
[^51]: 轨迹匹配：基于结构化动作的离散流匹配

    Trajectory Matching: Discrete Flow Matching with Structured Actions

    [https://arxiv.org/abs/2610.32602](https://arxiv.org/abs/2610.32602)

    提出轨迹匹配方法，将离散流匹配条件化于连接端点的完整动作序列，从而支持改变状态大小、修改多个组件及依赖历史的结构化动作，为用户自定义动作空间提供有原则的、无需模拟的训练目标。

    

    结构化离散生成的自然单元通常不是独立的单组件编辑，而是更丰富的动作：例如将一个片段连接到分子上、将一个节点及其关联边一起插入、或通过添加子节点来扩展层次结构。然而，大多数现有的离散流匹配方法（DFM）仅限于对固定大小状态进行独立的逐组件编辑。这一限制源于端点条件化：端点在训练期间固定了源和目标，却未固定两者之间的动作序列，导致对于无法按组件分解的复杂动作，难以定义可处理的条件速率。我们提出了轨迹匹配，它将流条件化于连接端点的完整动作序列之上。这为用户自定义的动作空间提供了有原则的、无需模拟的训练目标，其中包括改变状态大小、同时修改多个组件或依赖于先前动作的动作。我们证明……

    arXiv:2610.32602v1 Announce Type: new  Abstract: The natural units of structured discrete generation are often not independent single-component edits but richer actions: attaching a fragment to a molecule, inserting a node together with an incident edge, or growing a hierarchy by adding a child. Yet most existing discrete flow matching methods (DFM) are restricted to independent component-wise edits over fixed-size states. This restriction stems from endpoint conditioning: endpoints fix the source and target during training but not the action sequence between them, making tractable conditional rates hard to define for complex actions that cannot be factorized component-wise. We propose trajectory matching, which conditions the flow on a full sequence of actions connecting the endpoints. This yields principled, simulation-free training targets for user-defined action spaces, including actions that change state size, modify multiple components, or depend on earlier actions. We prove that
    
[^52]: 让我们正确处理湍流：在空间与时间上！

    Let's Do Turbulence Right: In Space and Time!

    [https://arxiv.org/abs/2610.32601](https://arxiv.org/abs/2610.32601)

    本文利用每次运行超过28 TB的大规模时间解析直接数值模拟，发现湍流的空间与时间评估可能不一致，呼吁机器学习社区将(3+1)维湍流视为四维现象，同时充分解析其空间结构和时间演化。

    

    三维纳维-斯托克斯方程支配着流体湍流，这是一种混沌的非线性物理现象，其特征是复杂涡旋在广阔的空间和时间尺度上相互作用。要对这种多尺度能量级联进行数值精确建模，需要同时完全解析流场的精细空间结构和连续的时间演化。在本文中，我们利用大规模的时间解析直接数值模拟（每次运行超过28 TB）来评估数值选择如何决定湍流中可解析的(3+1)维物理特性。在此分析的基础上，我们研究了神经网络如何表示所记录的湍流运动。我们的结果强调，无论是对于数值记录还是学习到的表示，空间评估和时间评估都可能得出不一致的结论。因此，我们呼吁机器学习社区将(3+1)维湍流视为一种真正的四维现象。

    arXiv:2610.32601v1 Announce Type: cross  Abstract: The three-dimensional Navier-Stokes equations govern fluid turbulence, a chaotic and non-linear physical phenomenon characterized by complex eddies interacting across vast spatial and temporal scales. A numerically accurate modeling of this multi-scale energy cascade requires fully resolving both the fine spatial structures and the continuous temporal evolution of the flow field. In this paper, we leverage massive, time-resolved direct numerical simulations, which exceed 28 TB per run, to assess how numerical choices govern the resolved (3+1)-dimensional physics of turbulent flows. Building on this analysis, we study how neural networks represent the recorded turbulent motion. Our results highlight that spatial and temporal assessments can disagree for both numerical records and learned representations. We therefore call on the machine learning community to treat (3+1)-dimensional turbulence as a four-dimensional phenomenon.
    
[^53]: 对称多凸能量与包络的通用神经网络逼近

    Universal neural network approximation of symmetric polyconvex energies and envelopes

    [https://arxiv.org/abs/2610.32598](https://arxiv.org/abs/2610.32598)

    本文提出对称多凸神经网络（SPCNNs），通过在深度架构中同时融合凸性与子式空间所需的单调性，首次实现了对对称多凸能量及其包络的通用逼近，为弹性力学中的本构建模提供了高效的神经网络替代方案。

    

    对称多凸性为小应变弹性理论中的弱下半连续性提供了充分条件，并为非（拟）凸能量的松弛提供了一个可处理的下界。由于计算此类函数的经典算法可能计算代价高昂，神经网络近似近来已成为一种替代方案。在为对称多凸性建立此类基于神经网络的方法时，一个核心困难在于额外的单调性要求。正如我们所证明的，在基于ReLU的输入凸网络上施加相应的权重限制并不能实现对提升后的表示子的通用逼近。本文引入了对称多凸神经网络（SPCNNs），用于本构建模以及对称多凸包络的逼近。该深度架构将凸性与表示子在子式空间中所要求的单调性相结合。我们证明了一个通用逼近定理……

    arXiv:2610.32598v1 Announce Type: cross  Abstract: Symmetric polyconvexity provides a sufficient condition for weak lower semicontinuity in small-strain elasticity and a tractable lower bound for the relaxation of non-(quasi)convex energies. Since classical algorithms for computing such functions can be computationally expensive, neural network approximations have recently emerged as an alternative. A central difficulty with establishing such neural network based approaches for symmetric polyconvexity is the additional monotonicity property. As we show, the corresponding weight restrictions on ReLU-based input-convex networks do not yield universal approximation of the lifted representers. Here, we introduce symmetric polyconvex neural networks (SPCNNs) for constitutive modeling and approximation of symmetric polyconvex envelopes. The deep architecture combines convexity with the monotonicity required of a representer in the space of minors. We prove a universal approximation theorem f
    
[^54]: 情境决策：预测式还是生成式

    Contextual Decision Making: Predictive or Generative

    [https://arxiv.org/abs/2610.32584](https://arxiv.org/abs/2610.32584)

    该论文从统计精度与优化可解性两个角度系统比较了先预测后优化与先生成后优化两种情境决策范式，证明了平滑性差异可以抵消生成式方法的维度负担，并揭示了凸性保持与学习复杂度之间的基本权衡。

    

    情境决策利用观测到的协变量在不确定性条件下指导决策。两种自然的方法是“先预测后优化”，它学习条件期望目标；以及“先生成后优化”，它学习条件不确定性并利用生成的场景进行优化。我们通过考察学习目标的选择如何影响统计精度与优化的可解性，来研究何时哪种方法更优。尽管PtO学习的目标范围更窄，我们的遗憾界表明，平滑性方面的差异可以抵消GtO所承担的额外维度负担。利用平滑性同时引发了一个优化上的权衡：在本文所研究的非负权重PtO方法类中，一个下界限制了额外平滑性所能带来的收益；而更灵活的方法可能引入带符号的权重，从而失去保持凸性的保证。GtO能够保持逐场景凸性，但需要学习更丰富（的不确定性模型）。

    arXiv:2610.32584v1 Announce Type: cross  Abstract: Contextual decision making uses observed covariates to guide decisions under uncertainty. Two natural approaches are predict-then-optimize (PtO), which learns the conditional expected objective, and generate-then-optimize (GtO), which learns conditional uncertainty and optimizes using generated scenarios. We study when each approach is preferable by examining how the choice of learning target affects statistical accuracy and optimization tractability. Although PtO learns a narrower target, our regret bounds show how differences in smoothness can offset GtO's additional dimensional burden. Exploiting smoothness also raises an optimization tradeoff: within the class of nonnegative-weight PtO methods studied, a lower bound limits the gains from additional smoothness, while more flexible methods may introduce signed weights and lose the guarantee of preserving convexity. GtO preserves scenario-wise convexity, but requires learning a richer
    
[^55]: 批处理神经网络四维变分（Batched NN 4D-Var）：面向综合预报系统的高性能混合模式框架

    Batched NN 4D-Var: a performant hybrid model framework for the Integrated Forecasting System

    [https://arxiv.org/abs/2610.32582](https://arxiv.org/abs/2610.32582)

    本文提出了批处理神经网络四维变分框架，通过在综合预报系统中嵌入神经网络表示模式误差，并以随机顺序处理多个同化窗口、对参数增量取平均实现在线学习，构建了高性能的机器学习与数值预报混合模式。

    

    结合机器学习与数值天气预报的混合预报系统，为在保持物理一致性的同时提升预报技巧提供了一条有前景的途径。在本研究中，我们利用四维变分（4D-Var）方法研究了在综合预报系统（Integrated Forecasting System, IFS）内对依赖于状态的模式误差订正进行在线学习。在以往工作的基础上，我们使用一个嵌入预报模式内部的神经网络来表示模式误差，并通过资料同化循环利用观测对其进行训练。首先，我们引入了一种针对性的神经网络四维变分变体，其中仅估计神经网络参数，而大气状态由预先计算的分析场提供。随后，我们提出了一种批处理扩展方法，该方法以随机顺序处理多个同化窗口，并对多个窗口上的参数增量取平均，其灵感来自小批量随机优化。实验在综合预报系统上开展（原文摘要至此截断）。

    arXiv:2610.32582v1 Announce Type: new  Abstract: Hybrid forecasting systems combining machine learning and numerical weather prediction offer a promising route toward improved forecast skill while preserving physical consistency. In this study, we investigate online learning of state-dependent model-error corrections within the Integrated Forecasting System using 4D-Var. Building on previous work, we represent model error using a neural network embedded within the forecast model and trained from observations through the data assimilation cycle. First, we introduce a targeted variant of neural-network 4D-Var in which only the neural-network parameters are estimated, while the atmospheric state is provided by a precomputed analysis. We then propose a batched extension that processes assimilation windows in random order and averages parameter increments over multiple windows, drawing inspiration from mini-batch stochastic optimization. Experiments are conducted with the Integrated Forecas
    
[^56]: 从参数重定位到异构Transformer间的功能迁移

    From Parameter Rebasin to Functional Transfer Across Heterogeneous Transformers

    [https://arxiv.org/abs/2610.32581](https://arxiv.org/abs/2610.32581)

    提出免训练方法Ariadne，通过刻画匹配块处的激活变化并用局部Procrustes映射对齐、闭式岭回归直接合成目标更新，在异构Transformer之间实现无需移动参数的任务行为迁移。

    

    将预训练的Transformer适配到下游任务会产生特定于任务的更新，而针对每个模型变体重新生成这些更新的成本很高。在不同深度和宽度的模型之间复用这些更新非常困难，因为它们的参数形状互不兼容，且不共享统一的坐标系。我们提出了Ariadne，这是一种免训练的方法，能够在异构Transformer之间迁移任务特定的行为，而无需调整源模型的尺寸或传输其参数。Ariadne不再把源任务向量（即微调模型与基础模型在参数空间中的差异）视为需要移动的对象，而是通过微调在匹配的网络块处引起的激活变化来刻画微调行为。首先，它通过局部Procrustes映射将这些效应与目标基础模型对齐；随后，利用闭式岭回归直接合成目标模型的原生更新。在Transformer扩展、缩减和语言（摘要在此处截断）

    arXiv:2610.32581v1 Announce Type: new  Abstract: Adapting pretrained Transformers to downstream tasks produces task-specific updates that are expensive to reproduce for every model variant. Reusing these updates across different depths and widths is challenging because their parameters have incompatible shapes and do not share a canonical coordinate system. We introduce Ariadne, a training-free method that transfers task-specific behavior across heterogeneous Transformers without resizing the source model or transporting its parameters. Rather than treating the source task vector, i.e. the difference in parameter space between the fine-tuned and the base models, as the object to move, Ariadne characterizes fine-tuning by the activation changes it induces at matched blocks. First, it aligns these effects to a target base through local Procrustes maps. Then, it directly synthesizes a native target update by closed-form ridge regression. Across Transformer expansion, reduction and languag
    
[^57]: 共享并不足够：掩码潜在预测下的预测器宽度与因子可读性

    Shared Is Not Enough: Predictor Width and Factor Readability Under Masked Latent Prediction

    [https://arxiv.org/abs/2610.32562](https://arxiv.org/abs/2610.32562)

    该论文通过构建所有潜在因子已知的模拟器，发现I-JEPA因使用宽度仅为编码器一半的预测器而导致共享因子的可读性大幅下降（中位数得分仅为SimCLR的40%），而将预测器加宽到编码器同等宽度后可在所有训练种子上逆转这一劣势。

    

    诸如I-JEPA之类的掩码潜在预测方法旨在学习可预测的特征，但决定它们保留哪些信息的因素尚不清楚。我们构建了一个模拟器，其中图像背后的每个因子都是已知的，因此探针可以测量训练后每个因子仍然保持的可读性。我们使用I-JEPA、SimCLR和BYOL在相同的视图对上训练了小型编码器（0.7M参数），并探测了两个视图共享的一个因子。I-JEPA的预测器具有编码器一半的宽度和深度，这是它在ViT-B中使用的比例。其中位数得分约为SimCLR的40%。将预测器加宽到编码器的宽度后，在每个训练种子上都逆转了这一排序，超过了其他两种方法。我们发布了该模拟器和训练好的模型。

    arXiv:2610.32562v1 Announce Type: new  Abstract: Masked latent prediction methods such as I-JEPA aim to learn predictable features, but what determines which information they retain is less clear. We built a simulator in which every factor behind an image is known, so a probe can measure how readable each factor remains after training. We trained small encoders (0.7M parameters) with I-JEPA, SimCLR and BYOL on the same pairs of views and probed a factor that both views share. I-JEPA had a predictor with half the encoder's width and depth, the proportions it uses for ViT-B. Its median score was about 40% of SimCLR's. Widening the predictor to the encoder's width reversed this ordering at every training seed, against both other methods. We release the simulator and the trained models.
    
[^58]: 非识别点附近的精确全局极小极大风险

    Exact Global Minimax Risk near Nonidentification

    [https://arxiv.org/abs/2610.32546](https://arxiv.org/abs/2610.32546)

    该论文给出了在冗余参数非识别点附近，最大临界路径极小极大风险等于全参数空间精确渐近极小极大风险的充分条件，并构造了能一致达到该精确首项常数的估计量，且在含未知方差的logistic误差变量回归中确定出五阶精确风险常数。

    

    我们研究当某个冗余参数在一个子模型上不可识别、但其会影响一个在该处趋于消失的目标量时的平方误差估计问题。对于具有高阶剖面展开的模型，我们给出了充分条件，使得最大的临界路径极小极大风险恰为整个参数空间上的精确渐近极小极大风险。一个高效的可见特征、可忽略的校准误差以及稳定的恢复机制，使得单个估计量能够在剖面坐标未知的情况下一致地达到首项常数。该估计量利用了校准估计、局部特征均值以及一个经验恢复特征。该常数通过一个有界正态决策问题来刻画。我们在两个模型中验证了这些条件。在具有一个代理变量和两个未知方差的中心化高斯logistic误差变量回归模型中，我们确定了五阶全局系数风险及其精确常数。一个稀疏扩展将方差分解……（原文摘要在此处截断）

    arXiv:2610.32546v1 Announce Type: new  Abstract: We study squared-error estimation when a nuisance parameter is unidentified on a submodel but affects a target that vanishes there. For models with higher-order profile expansions, we give sufficient conditions under which the largest critical-path minimax risk is the exact asymptotic minimax risk over the full parameter space. An efficient observable feature, negligible calibration error and stable recovery yield one estimator attaining the leading constant uniformly, with the profile coordinates unknown. The estimator uses calibration estimates, a local feature mean and an empirical recovery feature. The constant is expressed through a bounded-normal decision problem. We verify the conditions in two models. In centered Gaussian logistic errors-in-variables regression with one surrogate and two unknown variances, we determine the fifth-order global coefficient risk and its exact constant. A sparse extension separates variance-decomposit
    
[^59]: 原则性非线性概念擦除与引导

    Principled Nonlinear Concept Erasure and Steering

    [https://arxiv.org/abs/2610.32516](https://arxiv.org/abs/2610.32516)

    本文提出一个基于条件自编码器的原则性非线性概念擦除框架，从理论上证明完美重建与总体水平独立性足以实现完美的非线性防护保证，并在多个数据集上取得了更优的实用性与擦除效果的平衡。

    

    学习到的表示通常会编码敏感属性，例如性别或种族，这些属性若被保留可能是不可取的或不安全的。概念擦除旨在移除这些信息，同时保留与任务相关的内容。现有方法仍不完善：线性方法仅提供针对线性探针的保证，而非线性方法虽然实现了更强的经验擦除效果，但缺乏正式保证，且常常牺牲模型实用性。在这项工作中，我们为非线性概念擦除引入了一个原则性的条件自编码器框架。我们遵循这样的原则：擦除应当只移除敏感属性而不影响其他内容，并且我们证明，在给定假设下，完美重建结合总体水平的独立性足以实现完美的非线性防护性，同时在不完美重建的情形下也提供保留保证。在实证方面，我们在多个数据集上实现了更优的实用性-擦除权衡。

    arXiv:2610.32516v1 Announce Type: cross  Abstract: Learned representations often encode sensitive attributes, such as gender or race, that may be undesirable or unsafe to retain. Concept erasure aims to remove this information while preserving task-relevant content. Existing methods remain incomplete: linear methods provide guarantees only against linear probes, while nonlinear methods achieve stronger empirical erasure but lack formal guarantees and often sacrifice utility. In this work, we introduce a principled conditional autoencoder framework for nonlinear concept erasure. We follow the principle that erasure should remove only the sensitive attribute and nothing else, and we show that, under the stated assumptions, perfect reconstruction together with population-level independence is sufficient for perfect nonlinear guardedness, with preservation guarantees in the imperfect-reconstruction regime. Empirically, we attain a superior utility-erasure trade-off across multiple datasets
    
[^60]: 通过预测党派归属分析奥地利议会演讲

    Analysing Austrian Parliamentary Speeches by Predicting Party Affiliations

    [https://arxiv.org/abs/2610.32515](https://arxiv.org/abs/2610.32515)

    该论文利用微调BERT模型预测奥地利议会演讲的党派归属，发现联合执政党派的演讲特征趋于相似而反对党逐渐疏远，且议员只有在换党后才会改变演讲风格。

    

    本文研究了奥地利国民议会议员的演讲特征。我们提出了一种基于微调BERT模型的分析方法，该模型为每篇演讲分配一个关于奥地利各党派的概率分布，并在发言者和党派层面对这些分布进行聚合。在进行深入分析之前，我们进行了广泛的评估以确保生成分布的质量。我们没有将预测党派的错误归类视为性能问题，而是利用这些错误来推导关于议员和政治发展的见解。我们的研究结果表明，共同执政的党派在演讲特征上变得更加相似，而反对党则进一步疏远执政党。此外，转换党派的议员通常只有在换党之后才会调整其演讲特征。最后，我们公开发布了一个包含党派归属预测的数据集。

    arXiv:2610.32515v1 Announce Type: cross  Abstract: This paper examines speech characteristics of the Austrian National Council members (CMs). We introduce analysis methodology based on a fine-tuned BERT-Model which assigns to every speech a probability distribution over Austrian parties. These are then aggregated at the speaker and party level. We perform an extensive evaluation to ensure the quality of the generated distributions prior to further analysis. Instead of considering mis-assignments of predicted parties as a performance issue, we leverage them to derive insights about the CMs and political developments. Our findings show that parties that govern together become more similar in their speech characteristics, while, analogously, parties that are in opposition move further away from governing parties. Moreover, CMs who switch party usually adapt their speech characteristics only after the switch occurs. Finally, we publicly release a data set containing party affiliation predi
    
[^61]: 摊销式汤姆逊采样与面向基于模拟器设计的高效信息导向采样

    Amortized Thompson Sampling and Efficient Information-Directed Sampling for Simulator-Based Design

    [https://arxiv.org/abs/2610.32514](https://arxiv.org/abs/2610.32514)

    该论文提出一种摊销式汤姆逊采样方法，在数据收集前预先计算汤姆逊采样分布的近似，使观测到来时能够立即获取样本，从而大幅加速多轮序贯决策，并实现高效的信息导向采样。

    

    汤姆逊采样是一种广泛应用于不确定性环境下序贯决策的策略，它提供了一种在探索与利用之间进行权衡的原则性方法。该策略需要在从当前关于未知量的信念中抽取的单一样本下最大化目标函数。传统上，这一优化是在从数据中推断出信念之后才进行的，当需要大量汤姆逊样本时，计算负担会变得十分沉重，尤其是在连续动作空间中。在这项工作中，我们提出了一种方法，可以在收集任何数据之前提前摊销这一过程，使得一旦观测数据到来，即可立即获得完整汤姆逊采样分布的近似。我们运用该方法求解多轮设定，即在多轮之间进行序贯数据采集与汤姆逊样本推荐。除了加速汤姆逊采样之外，我们的方法还实现了高效的信息导向采样。

    arXiv:2610.32514v1 Announce Type: new  Abstract: Thompson sampling is a widely used strategy for sequential decision-making under uncertainty, offering a principled way to balance exploration and exploitation. This strategy involves maximizing a target objective under a single sample drawn from the current belief over unknown quantities. Conventionally, this optimization is performed once the belief has been inferred from data, and it becomes computationally burdensome when many Thompson samples are required, especially in continuous action spaces. In this work, we introduce an approach that amortizes this process ahead of time, before any data is collected, so that an approximation of the full Thompson sampling distribution is available immediately once observations arrive. We employ this approach to solve the multi-round setting, involving sequential data acquisition and Thompson-sample recommendation across rounds. Beyond accelerating Thompson sampling, our approach enables efficien
    
[^62]: 暗网超链接图上的量子-经典链接预测

    Quantum-Classical Link Prediction on Dark Web Hyperlink Graphs

    [https://arxiv.org/abs/2610.32510](https://arxiv.org/abs/2610.32510)

    本研究首次在暗网Tor超链接图上系统比较了量子与经典链接预测方法，发现在严格的度控制评估下经典图自编码器表现最佳（VGAE达0.906），而四量子比特变分量子分类器与经典基线相比并无优势。

    

    我们分析了来自离线Tor爬取数据中的HTML内容与超链接，该数据包含383个洋葱主机和2,067条有向跨主机链接。本研究将语义分析与度控制条件下的量子-经典链接预测相结合。研究发现，两个目录群组发出了94.8%的观测主机链接。有向优先连接方法在随机非边情况下达到了0.976的ROC-AUC，但在源条件目标度匹配下降至0.550±0.014。在训练与测试对不相交的设置下，多模态逻辑回归达到0.672±0.021，图自编码器（GAE）为0.899±0.016，变分图自编码器（VGAE）为0.906±0.016。研究还比较了四量子比特变分量子分类器（VQC），在精确模拟和1,024次采样模拟下，对两种编码方式在20个不相交对图划分上进行了测试，经典对照组使用相同的输入和训练样本。VQC的最佳精确模拟平均ROC-AUC为0.549±0.023，相比之下RBF支持向量分类器为0.551±0.016，16参数的MLP为0.573±0.029。

    arXiv:2610.32510v1 Announce Type: cross  Abstract: We analyze HTML content and hyperlinks from an offline Tor crawl containing 383 onion hosts and 2,067 directed cross-host links. The study combines semantic analysis with degree-controlled quantum--classical link prediction. Two directory groups emit 94.8\% of the observed host links. Directed preferential attachment reaches ROC-AUC 0.976 with random non-edges, but falls to $0.550\pm0.014$ under source-conditioned target-degree matching. With disjoint training and test pairs, multimodal logistic regression reaches $0.672\pm0.021$, GAE $0.899\pm0.016$, and VGAE $0.906\pm0.016$. A four-qubit variational quantum classifier (VQC) comparison tests two encodings across 20 disjoint-pair graph splits under exact and 1,024-shot simulation, with classical controls using the same inputs and training examples. Its best exact mean ROC-AUC is $0.549\pm0.023$, compared with $0.551\pm0.016$ for RBF SVC and $0.573\pm0.029$ for a 16-parameter MLP on the
    
[^63]: BAARC：面向材料分类的不确定性驱动的自适应部分光谱测量与全谱重建方法

    BAARC: Uncertainty-Driven Adaptive Partial Spectral Measurement to Full Reconstruction for Material Classification

    [https://arxiv.org/abs/2610.32505](https://arxiv.org/abs/2610.32505)

    该论文提出了一个不确定性驱动的贝叶斯自适应框架BAARC，通过逐步部分光谱测量与全谱重建的联合训练，在无需材料先验知识的情况下实现拉曼光谱材料的自动分类识别。

    

    波数覆盖范围有限的光谱仪器需要精确的扫描定位或多次测量才能完成材料分析，这通常需要对待诊断材料具备先验知识，以便测量具有特征峰的光谱区域。在我们的研究中，我们为拉曼光谱测量构建了一个贝叶斯自适应采集与重建驱动分类（BAARC）框架的仿真，该框架采用掩码训练来学习光谱特征，并动态驱动激光通道的选择，直至识别出材料。我们提出了一个联合训练的两级框架：该框架在完整光谱范围内逐步进行小波数片段的部分测量，在每一步中首先尝试重建完整光谱，直到模型的不确定性足够低；其次，在每个重建步骤中进行材料分类，直到满足停止条件。过程中的不确定性量化……

    arXiv:2610.32505v1 Announce Type: new  Abstract: Spectroscopic instruments with limited wavenumber coverage require accurate scan positioning or multiple measurements to complete the material analysis. This often requires prerequisite knowledge about the material under diagnosis to measure the spectral regions with distinguishing peaks. In our study, we simulate a Bayesian Adaptive Acquisition with Reconstruction-driven Classification (BAARC) framework for Raman spectral measurements, using masked training to learn spectral features that dynamically drive laser channel selection until the material is identified. We present a jointly trained two-level framework that takes progressive partial measurements of small wavenumber patches across the complete spectral range and, at each step, first attempts to reconstruct the full spectra until the model uncertainty is low. Secondly, material classification is performed at each reconstruction step until stoppage. Uncertainty quantification duri
    
[^64]: DRIVE：通过数据贡献与多样性感知的加权及先验正则化进行微调

    DRIVE: Fine-tuning via Data Contribution- and Diversity-aware Weighting with Prior Regularization

    [https://arxiv.org/abs/2610.32501](https://arxiv.org/abs/2610.32501)

    DRIVE提出了一个面向单阶段微调的统一可微数据加权框架，通过Shapley贡献分解（抗遗忘与适应分量）、基于协方差的多样性项和先验正则化三个耦合信号的共同演化，在源预训练数据不可访问的情况下显式平衡知识保持与新任务适应。

    

    单阶段微调十分普遍，然而当源预训练数据不可访问时，灾难性遗忘依然存在。数据加权为调节样本影响提供了一种自然的手段。然而，现有的加权方法缺乏一种有原则的机制来显式地权衡知识保持与新任务适应之间的关系，并且没有提供一个统一的可微目标来容纳诸如样本多样性和样本特征等辅助信号。这是因为当前的方案往往偏重其中一个目标而忽视另一个，且不存在针对这些信号的显式梯度兼容编码。为解决这一问题，我们提出了DRIVE，一个有原则且高效的权重分配框架。它通过一个统一的可微目标推导出每个样本的权重，其中三个耦合的信号共同演化：Shapley可加性将贡献分解为抗遗忘和适应两个分量，这些分量由基于协方差的多样性项进行重塑，并通过先验正则化加以稳定。

    arXiv:2610.32501v1 Announce Type: new  Abstract: Single-stage fine-tuning is pervasive, yet catastrophic forgetting persists when the source pre-training data is inaccessible. Data weighting affords a natural lever to modulate sample influence. Yet existing weighting methods lack a principled mechanism to explicitly arbitrate knowledge retention against new-task adaptation, and provide no unified differentiable objective to accommodate auxiliary signals such as sample diversity and sample characteristics. This is because current schemes privilege one goal over the other, and no explicit gradient-compatible encoding exists for such signals. To address this, we propose DRIVE, a principled and efficient weight assignment framework. It derives per-sample weights via a unified differentiable objective where three coupled signals co-evolve: Shapley additivity splits contribution into anti-forgetting and adaptation components, which are reshaped by a covariance-based diversity term and stabil
    
[^65]: 随机跳跃森林

    Random Jump Forests

    [https://arxiv.org/abs/2610.32498](https://arxiv.org/abs/2610.32498)

    本文提出随机跳跃森林这一非参数方法，利用单棵树同时划分协变量空间并在节点处结合Nelson-Aalen估计，无需马尔可夫假设即可估计多状态模型中的条件累积转移率与占据概率，并建立了一致性和泛函中心极限定理等理论保证。

    

    本文引入随机跳跃森林，作为一种非参数工具，用于估计多状态模型中的条件累积转移率和占据概率。其主要思想是：利用部分观测到的轨迹，让单棵树同时为所有转移划分协变量空间，并在终端节点处使用Nelson-Aalen估计量。占据概率通过乘积积分恢复，而无需马尔可夫假设。在协变量向量取有限多个值的假设以及适当的分裂条件下，可以建立累积率估计量的一致性和泛函中心极限定理。特别地，在马尔可夫情形下可以给出逐点置信区间和同时置信带。模拟研究表明，该方法的预测精度接近于正确设定的基准模型，而当基准模型被……时则有显著改进。

    arXiv:2610.32498v1 Announce Type: cross  Abstract: Random jump forests are introduced as a nonparametric tool for estimating conditional cumulative transition rates and occupation probabilities in multi-state models. The main idea is that a single tree partitions the covariate space for all transitions simultaneously, using partially observed trajectories, and with Nelson--Aalen estimators at the terminal nodes. Occupation probabilities are recovered through the product integral without requiring the Markov assumption. Uniform consistency and a functional central limit theorem for the estimators of cumulative rates can be established under the assumption that the covariate vector takes finitely many values, and under suitable splitting conditions. In particular, pointwise confidence intervals and simultaneous bands are available in the Markov case. Simulation studies show a predictive accuracy close to the correctly specified benchmark models and marked improvements when benchmarks are
    
[^66]: Accordion：通过固定前缀层循环实现弹性深度推理

    Accordion: Elastic-Depth Inference via Fixed-Prefix Layer Looping

    [https://arxiv.org/abs/2610.32492](https://arxiv.org/abs/2610.32492)

    本文提出基于混合专家的模型Accordion，通过固定前缀层循环支持五种推理预算，并在预训练、监督微调和强化学习全流程中无需跨预算一致性目标即可在不同推理预算下保持强劲性能。

    

    大型语言模型服务于具有不同推理预算的各类应用，但为每个预算训练和维护单独的模型会产生巨大的成本。近期的循环Transformer通过参数重用实现了灵活的推理计算，但在不同推理预算下保持强劲性能仍然是一个开放的挑战。我们提出了Accordion，一种基于混合专家的模型，通过固定前缀层循环支持五种推理预算。每个预算遵循预定义的执行路径，而最小预算则保留未循环的模型。联合预训练在每个微批次中采样一个预算并优化其下一个词预测损失，无需额外的跨预算一致性目标。我们进一步通过监督微调和强化学习来保持这些推理预算，其中参数组合并和预算匹配的多教师在线策略蒸馏将三个领域专家整合到统一模型中。

    arXiv:2610.32492v1 Announce Type: cross  Abstract: Large language models serve applications with diverse inference budgets, but training and maintaining separate models for each budget incurs substantial costs. Recent Looped Transformers enable flexible inference computation through parameter reuse, but maintaining strong performance across different inference budgets remains an open challenge. We introduce Accordion, a mixture-of-experts-based model that supports five inference budgets through fixed-prefix layer looping. Each budget follows a pre-defined execution path, while the minimum budget retains the unlooped model. Joint pretraining samples one budget per micro-batch and optimizes its next-token prediction loss without additional cross-budget consistency objectives. We further preserve these inference budgets through supervised fine-tuning and reinforcement learning, where parameter-group merging and budget-matched multi-teacher on-policy distillation integrate three domain exp
    
[^67]: 一个具有构建体感知蛋白靶点的精选多终点药物-蛋白质生物活性数据集

    A curated multi-endpoint drug-protein bioactivity dataset with construct-aware protein targets

    [https://arxiv.org/abs/2610.32490](https://arxiv.org/abs/2610.32490)

    该论文构建了Curate-DTI数据集，通过将UniProt登录号与实验序列结合来区分不同的蛋白质构建体，提供了包含超过290万条测量数据的构建体感知多终点药物-蛋白质生物活性数据集。

    

    公共的药物-蛋白质生物活性资源通常通过UniProt登录号来识别靶点，但一个登录号可能代表不同的实验序列。我们提出了Curate-DTI，这是一个构建体感知的数据集，源自2026年7月1日的BindingDB导出数据，其中IC50、Ki、Kd和EC50被独立处理。该数据集包含2,919,717条测量数据、1,291,872个标准化配体、9,668个序列定义的蛋白质实体，以及涵盖8,336个已审核UniProt登录号的2,275,683个终点特异性相互作用。蛋白质实体将登录号与报告的实验序列相结合，保留了在仅按登录号处理时会被合并的构建体。在四个终点中，共有4,204个终点特异性的配体-登录号组在不同的序列定义蛋白质实体间显示出不同的分类标签。精确值和不等式限定的观测值提供了关系感知的分类证据，而回归目标则使用精确值。

    arXiv:2610.32490v1 Announce Type: new  Abstract: Public drug-protein bioactivity resources commonly identify targets by UniProt accession, although one accession may represent distinct experimental sequences. We present Curate-DTI, a construct-aware dataset derived from the 1 July 2026 BindingDB export, with IC50, Ki, Kd and EC50 processed independently. The release contains 2,919,717 measurements, 1,291,872 standardized ligands, 9,668 sequence-defined protein entities and 2,275,683 endpoint-specific interactions spanning 8,336 reviewed UniProt accessions. Protein entities combine the accession with the reported experimental sequence, preserving constructs that accession-only processing would merge. Across the four endpoints, 4,204 endpoint-specific ligand-accession groups showed different classification labels across sequence-defined protein entities. Exact and inequality-qualified observations provide relation-aware classification evidence, whereas regression targets use exact values
    
[^68]: 将状态估计作为上下文表格预测

    State Estimation as In-context Tabular Prediction

    [https://arxiv.org/abs/2610.32487](https://arxiv.org/abs/2610.32487)

    该论文提出将电力系统状态估计转化为上下文表格预测问题，利用权重冻结的表格基础模型（仅在合成数据上预训练），在低可观测性、不良数据等加权最小二乘法失效的场景下实现了显著更优的估计精度。

    

    本信将电力系统状态估计（SE）表述为一个上下文表格预测（ICTP）问题。其核心思想是一种表格化数据组织方式：历史量测向量及对应的状态估计作为上下文，而新的量测向量构成查询，模型据此预测状态向量。我们提出使用在合成数据上预训练的表格基础模型（TFMs），其权重从不更新，因此电力系统数据仅需作为上下文提供。利用冻结权重的TFM，我们对14、30、33、70和118节点的输配电网络进行状态估计，并在加权最小二乘法（WLS）失效或退化的场景下——如低可观测性和存在不良数据时——依然有效。在完全量测条件下，ICTP的误差比上下文中各节点平均误差低11至57倍；在仅有5%量测可用时，误差仍低1.3至4.0倍。当六个节点的量测存在8σ偏差时，ICTP的角度误差上升1.8至（原文摘要在此处截断）

    arXiv:2610.32487v1 Announce Type: cross  Abstract: This letter poses state estimation (SE) as an in-context tabular prediction (ICTP) problem. The key idea is a tabular arrangement in which historical measurement vectors and corresponding state estimates serve as the context, and a new measurement vector forms the query for which we predict a state vector. We propose to use tabular foundation models (TFMs), pretrained on synthetic data, whose weights are never updated, so power system data is needed only as context. With frozen TFM weights we estimate the state of 14-, 30-, 33-, 70-, and 118-bus transmission and distribution networks, and do so where weighted least squares (WLS) fails or degrades, under low observability and with bad data. ICTP errors are 11 to 57$\times$ below the per-bus average of the context at full metering, and 1.3 to 4.0$\times$ below it with only 5% of measurements available. With measurements at six buses biased by 8$\sigma$, the ICTP angle error rises 1.8 to 
    
[^69]: 基于信号感知自适应融合（SAAF）的高效零样本视频异常检测

    Efficient Zero-Shot Video Anomaly Detection via Signal-Aware Adaptive Fusion (SAAF)

    [https://arxiv.org/abs/2610.32486](https://arxiv.org/abs/2610.32486)

    SAAF框架通过双信号自适应采样器（增量信号捕捉突发事件、漂移信号检测静态与渐进异常）以及阈值门控的级联架构，在无需训练数据的情况下实现了高效且全面的零样本视频异常检测。

    

    将视觉语言模型部署于视频异常检测面临一个根本性约束：其语义推理所需的计算预算与真实世界视频流的规模不兼容。我们提出信号感知自适应融合（SAAF）框架，通过三项关键创新实现高效的零样本异常检测。首先，我们引入一种在语义嵌入空间中运行的双信号自适应采样器：增量信号通过帧间余弦距离捕捉突发事件，而漂移信号通过与滚动时间上下文的偏差检测静态和渐进式异常。二者通过OR门控组合，并结合基于自适应指数移动平均（EMA）的阈值处理，无需任何训练数据即可实现全面的异常覆盖。其次，我们提出一种阈值门控的级联架构，由轻量级的帧级评分先行处理，仅当置信度需要进一步分析时才调用昂贵的视频级时序分析。

    arXiv:2610.32486v1 Announce Type: cross  Abstract: Deploying Vision-Language Models for video anomaly detection confronts a fundamental constraint. Their semantic reasoning demands computational budgets incompatible with the scale of real-world video streams. We present Signal-Aware Adaptive Fusion (SAAF), a framework that achieves efficient zero-shot anomaly detection through three key innovations. First, we introduce a dual-signal adaptive sampler operating in semantic embedding space: a delta signal captures sudden events via frame-to-frame cosine distance, while a drift signal detects static and gradual anomalies through deviation from rolling temporal context. Their OR-gated combination with adaptive EMA-based thresholding provides comprehensive anomaly coverage without training data. Second, we propose a threshold-gated cascaded architecture where lightweight frame-level scoring precedes expensive video-level temporal analysis, invoking it only when confidence warrants additional
    
[^70]: PathJev：面向组织病理学的决策原生视觉语言建模

    PathJev: Decision-Native Vision-Language Modelling for Histopathology

    [https://arxiv.org/abs/2610.32478](https://arxiv.org/abs/2610.32478)

    PathJev-P 是一个决策原生的组织病理学视觉语言模型，通过在输入中嵌入运行时问题和候选集合并经小型决策头一次前向传播输出归一化概率分布，在无数据重叠的测试集上准确率超越同骨干生成式模型、零样本病理模型 CONCH 和通用医疗决策模型。

    

    用于组织病理学决策的视觉语言模型很少能针对用户关心的候选类别返回校准良好的概率。决策原生模型将运行时选定的问题和候选集合写入输入，并在一次前向传播中通过一个小型决策头返回归一化的概率分布。我们构建了 PathJev-P，一个面向组织病理学的决策原生模型：基于 MedGemma-1.5-4B，配备 LoRA 适配器和一个决策头，在来自三个公开数据集的 40,000 个切片补丁状态上训练，并将专家标注转换为带有器官特异性候选集合和“以上都不是”选项的类型化问题。在一项基于 3 个随机种子、且分析方案在测试前预先固定的对比实验中，在与骨干模型训练数据无已记录重叠的测试集上，PathJev-P 的准确率超过了使用相同骨干、数据和选择规则训练的生成式模型、零样本病理模型 CONCH 以及一个通用医疗决策模型。

    arXiv:2610.32478v1 Announce Type: cross  Abstract: Vision-language models used for decisions in histopathology rarely return calibrated probabilities over the candidates a user cares about. Decision-native models write a question and a candidate set chosen at run time into the input and return a normalised distribution from a small decision head in one forward pass. We build PathJev-P, a decision-native model for histopathology: MedGemma-1.5-4B with LoRA adapters and a decision head, trained on 40,000 patch states from three public collections whose expert labels are turned into typed questions with organ-specific candidate sets and a "none of these" option. In a comparison over 3 seeds with an analysis plan fixed before testing, on test sets without documented overlap with the training data of the backbone, PathJev-P is more accurate than a generative model trained with the same backbone, data and selection rule, the zero-shot pathology model CONCH and a general medical decision model
    
[^71]: 边缘-云应用中的在线持续学习

    Online Continual Learning in Edge-Cloud Applications

    [https://arxiv.org/abs/2610.32472](https://arxiv.org/abs/2610.32472)

    本文提出一种边缘-云方法论，将持续学习与分布外检测及概念漂移检测相集成，利用ODIN和ECDD机制识别有价值的样本并触发基于回放的模型更新，从而解决物联网应用中模型在线适应非平稳数据流时的灾难性遗忘和分布外性能退化问题。

    

    在现实世界的物联网（IoT）及边缘/云系统中部署的机器学习模型面临一个关键的可靠性瓶颈：需要在无法访问完整原始训练数据集的情况下，对非平稳数据流进行在线适应。当模型遇到分布外（OOD）输入或遭受灾难性遗忘时，性能会严重下降。应对这些挑战需要一种适应机制，能够从不断演变的数据流中学习，同时保留先前获得的知识。本文提出了一种边缘-云方法论，用于在物联网应用中将持续学习（CL）与分布外检测及概念漂移检测相集成。该方法引入了ODIN和ECDD作为可选机制，用于识别具有信息价值的样本以进行标注，并触发基于回放的模型更新。实验结果在汽车、医疗健康和农业等数据集上……（原文摘要在此处被截断）

    arXiv:2610.32472v1 Announce Type: new  Abstract: Deployed machine learning models in real-world IoT and edge/cloud systems face a critical reliability bottleneck: the requirement for online adaptation to non-stationary data streams while operating without access to the full original training dataset. Performance is severely degraded when models encounter Out-of-Distribution (OOD) inputs or suffer from catastrophic forgetting. Addressing these challenges requires adaptation mechanisms capable of learning from evolving data streams while preserving previously acquired knowledge. This paper proposes an edge-cloud methodology for integrating Continual Learning (CL) with Out-of-Distribution (OOD) Detection and Concept Drift Detection in IoT applications. The methodology incorporates ODIN and ECDD as alternative mechanisms for identifying informative samples for annotation and triggering rehearsal-based model updates. Experimental results across automotive, healthcare, and agricultural datas
    
[^72]: 锐度感知正交化动量算法

    Sharpness-aware Orthogonalized Momentum Algorithms

    [https://arxiv.org/abs/2610.32465](https://arxiv.org/abs/2610.32465)

    提出锐度感知正交化动量算法（SOMA），将计算高效且有理论依据的锐度扰动方案与 Muon 优化器结合，以提升大语言模型训练的泛化能力并降低额外梯度计算开销。

    

    在基础优化器的每次迭代中加入对抗性锐度惩罚，即所谓的锐度感知最小化（SAM），是一种公认的用于提升深度学习泛化能力的技术。近来，Muon 已成为训练大语言模型（LLM）的主导性优化算法，它针对重尾梯度更新——LLM 训练中的常见现象——实现了更低的样本复杂度。然而，将锐度惩罚与 Muon 更新相结合的研究在文献中仍基本处于空白。此外，SAM 在重尾噪声下的行为在理论和实证层面都尚不清楚。造成这一空白的一个潜在原因在于，SAM 每次迭代都需要为惩罚扰动额外计算梯度，对于大语言模型而言，这一开销变得难以承受。本文提出了锐度感知正交化动量算法（SOMA），将一种计算高效且有理论依据的扰动方案与 Muon 算法相结合。

    arXiv:2610.32465v1 Announce Type: new  Abstract: Adversarial sharpness penalty per iteration of a base optimizer, known as sharpness-aware minimization (SAM), is an accepted technique to improve generalization in deep learning. Recently, Muon has emerged as a dominant optimization algorithm for training large language models (LLMs). It achieves lower sample complexity for heavy-tailed gradient updates - a phenomenon common in LLM training. However, sharpness penalties with Muon updates are yet largely unexplored in the literature. Moreover, the behavior of SAM under heavy-tailed noise is unknown both theoretically and empirically. A potential reason for this gap is the additional cost of extra gradient computation per iteration for penalty perturbations in SAM, which becomes prohibitive for LLMs. This paper introduces Sharpness-aware Orthogonalized Momentum Algorithm (SOMA) to combine a computationally efficient and theoretically grounded perturbation scheme with the Muon algorithm. Th
    
[^73]: DualGSL：先验引导的图结构学习与节点自适应预测融合

    DualGSL: Prior-Guided Graph Structure Learning with Node-Adaptive Prediction Fusion

    [https://arxiv.org/abs/2610.32462](https://arxiv.org/abs/2610.32462)

    提出了DualGSL框架，通过节点自适应门控融合先验传播路径与任务自适应传播路径的预测结果，使图结构学习能够针对不同节点动态调整两类信息的权重。

    

    图结构学习（GSL）将传播图适配到预测任务中，但基于特征的先验与学习到的关系之间的相对预测价值可能因节点而异。我们提出了DualGSL，这是一个用于直推式节点分类的先验引导框架，它学习节点特定的权重，用于组合来自独立的先验传播路径和任务自适应传播路径的预测。先验分支通过引入虚拟类别节点来增强特征图，以融入训练标签信息，而结构分支则将学习到的亲和图与特征图先验相结合。两个分支共享编码器参数，同时保持各自不同的表示和分类logits。一个节点自适应门控基于分支的投影表示、逐元素绝对差以及预测置信度来融合两个分支的logits。联合目标函数将融合后的分类损失、分支特定的分类损失以及先验分支的类别锚点损失整合在一起……

    arXiv:2610.32462v1 Announce Type: new  Abstract: Graph structure learning (GSL) adapts propagation graphs to prediction tasks, but the relative predictive value of feature-derived priors and learned relations can vary across nodes. We propose DualGSL, a prior-guided framework for transductive node classification that learns node-specific weights for combining predictions from separate prior and task-adaptive propagation paths. The prior branch augments a feature graph with virtual class nodes to incorporate training-label information, while the structural branch combines learned affinity graphs with the feature-graph prior. The two branches share encoder parameters while preserving distinct representations and classification logits. A node-adaptive gate fuses the branch logits based on their projected representations, element-wise absolute difference, and prediction confidence. A joint objective integrates fused and branch-specific classification losses with prior-branch class-anchor a
    
[^74]: DIGOM：面向多模态医疗结果预测的解耦交互图

    DIGOM: Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare

    [https://arxiv.org/abs/2610.32451](https://arxiv.org/abs/2610.32451)

    提出DIGOM框架，利用解耦视图的稀疏多路图联合建模患者内跨模态、跨患者同模态和跨患者跨模态三种交互关系，以提升多模态医疗数据的临床结果预测性能。

    

    多模态医疗数据包含着超越单个患者观测范围的关系信息。因此，有效的临床结果预测不仅需要整合来自多种模态的信息，还需要对这些模态之间的关系进行建模。标准的融合方法无法显式地捕捉这些交互，而现有的基于图的方法通常只能建模其中的一部分交互。我们提出DIGOM（Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare，面向多模态医疗结果预测的解耦交互图），这是一个基于图的框架，能够联合建模三种互补的交互类型：患者内跨模态交互、跨患者同模态交互以及跨患者跨模态交互。DIGOM学习一个具有解耦视图的稀疏多路图，其中每个视图编码患者间不同模态对之间的交互。通过可微机制采样具有患者自适应连接性的离散邻接矩阵。

    arXiv:2610.32451v1 Announce Type: cross  Abstract: Multimodal healthcare data contain relational information that extends beyond individual patient observations. Effective clinical outcome prediction therefore requires not only integrating information from multiple modalities, but also modeling the relationships between them. Standard fusion approaches do not explicitly capture these interactions, while existing graph-based approaches typically model only a subset of them. We propose DIGOM (Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare), a graph-based framework that jointly models three complementary interaction types: Intra-patient cross-modal, Cross-patient same-modal, and Cross-patient cross-modal interactions. DIGOM learns a sparse multiplex graph with disentangled views, each encoding interactions between pairs of modalities across patients. Discrete adjacency matrices with patient-adaptive connectivity are sampled via a differentiable mechanism, 
    
[^75]: 解耦患者画像与临床工作流：一种用于逼真医疗对话生成的双智能体方法

    Decoupling Patient Personas and Clinical Workflows: A Dual-Agent Approach to Realistic Medical Dialogue Generation

    [https://arxiv.org/abs/2610.32443](https://arxiv.org/abs/2610.32443)

    提出CoRPS-Dial双智能体框架，通过解耦患者画像与临床工作流并辅以事后质量控制，生成兼顾临床规范性与患者多样性的逼真诊前医疗对话。

    

    诊前医疗对话对于收集全面的患者信息至关重要，但由于隐私和伦理限制，大规模获取这类对话成本高昂。现有基于大语言模型（LLM）的合成方法通常依赖提示工程或僵化的规则，限制了其在临床相关性与真实患者多样性之间取得平衡的能力。我们提出了CoRPS-Dial，这是一个用于诊前医疗对话合成的可控双智能体框架，它将患者画像与符合临床指南的工作流规则相整合。CoRPS-Dial通过由患者画像驱动的患者智能体与由流程引导的医生智能体之间的交互来生成对话，并进行事后质量控制以保证角色一致性、工作流完整性和文本一致性。在冠心病数据上的实验表明，CoRPS-Dial在信息覆盖度、临床合规性和对话自然度方面提升了合成对话与真实对话之间的对齐程度。

    arXiv:2610.32443v1 Announce Type: cross  Abstract: Pre-consultation medical dialogues are essential for collecting comprehensive patient information, but are costly to obtain at scale due to privacy and ethical constraints. Existing LLM-based synthesis methods often rely on prompt engineering or rigid rules, limiting their ability to balance clinical relevance with realistic patient diversity. We propose CoRPS-Dial, a controllable dual-agent framework for pre-consultation medical dialogue synthesis that integrates patient personas with guideline-aligned clinical workflow rules. CoRPS-Dial generates dialogues through interactions between a persona-driven patient agent and a process-guided doctor agent, followed by post-hoc quality control for role consistency, workflow completeness, and textual consistency. Experiments on coronary heart disease demonstrate that CoRPS-Dial improves alignment between synthetic and real dialogues in information coverage, clinical compliance, and dialogue n
    
[^76]: PROVE：基于补丁级证据的视觉-语言模型概率视觉定位

    PROVE: Probabilistic Visual Grounding via Patch-Level Evidence for Vision-Language Models

    [https://arxiv.org/abs/2610.32440](https://arxiv.org/abs/2610.32440)

    提出PROVE，一种无需修改冻结CLIP骨干网络的事后证据适配器，通过将补丁特征映射为Beta分布并累积正负证据实现概率化视觉定位，同时利用证据空缺性作为信号判断所指对象是否存在，无需额外的存在分类器。

    

    使用CLIP等视觉-语言模型进行视觉定位时，通常针对图像-查询对生成单一的确定性相关性图，而无法指示该定位是否可靠，或所指对象是否确实存在。我们提出了PROVE（基于补丁级证据的概率视觉定位），这是一种事后证据适配器，能够将冻结的CLIP补丁特征映射为每个补丁的Beta分布，而无需修改骨干网络。PROVE将定位任务转化为逐补丁的成员归属问题：每个补丁累积其是否属于所指对象的正负证据，因此一个查询可以匹配零个、一个或多个补丁。我们使用基于软补丁覆盖目标的证据目标来训练PROVE，并定义了一个先验拉近目标，使不存在所指对象的样本保持在无信息先验附近，从而使证据空缺性能够作为对象缺失的信号，而无需专门的存在分类器。在推理阶段

    arXiv:2610.32440v1 Announce Type: cross  Abstract: Visual grounding with vision-language models such as CLIP typically produces a single deterministic relevance map for an image-query pair, without indicating whether the localisation is reliable or whether the referent is present at all. We introduce \textbf{PROVE} (\textbf{Pro}babilistic \textbf{V}isual Grounding via Patch-Level \textbf{E}vidence), a post-hoc evidential adapter that maps frozen CLIP patch features to per-patch Beta distributions without modifying the backbone. PROVE casts grounding as per-patch membership: each patch accumulates positive and negative evidence for whether it belongs to the referent, so a query may match zero, one, or many patches. We train PROVE with an evidential objective on soft patch-coverage targets, and define a prior-pull objective that keeps absent-referent examples near an uninformative prior, so that vacuity serves as an absence signal without a dedicated presence classifier. During inference
    
[^77]: 长期LoRaWAN网络的数据质量审计与温度-湿度双变量分析：异常检测、主成分分析、聚类与机器学习

    Data Quality Audit and Bivariate Temperature-Humidity Analysis of a Long-Term LoRaWAN Network: Anomaly Detection, PCA, Clustering and Machine Learning

    [https://arxiv.org/abs/2610.32438](https://arxiv.org/abs/2610.32438)

    该研究提出一种C++17处理流水线，通过对长期LoRaWAN数据集进行系统性变量覆盖审计，发现CO2数据完全缺失、气压覆盖率极低，从而将分析重新聚焦于覆盖率高达95.76%的温度-湿度变量对，并结合异常检测、主成分分析、聚类与机器学习方法开展双变量分析。

    

    长期环境物联网/LPWAN部署通常会聚合来自多种设备的异构数据流，但在分析之前并未对各变量的实际数据覆盖情况进行系统性审计——当多种传感器类型共存于同一网络时，这种风险会进一步加大。本工作提出了一种C++17处理流水线，并应用于Tour Perret LoRaWAN数据集（421,937帧数据，2021年6月至2023年6月，5个设备）。集成于数据提取阶段的系统性覆盖审计显示，CO2完全缺失（覆盖率0%），气压数据仅由一个孤立的基础模块（Wyres，占语料库的4.24%）上报，这使最初进行四变量联合分析的假设不成立。因此，分析重新聚焦于温度-湿度变量对，二者在来自4个Elsys EMS传感器的205,005帧数据（95.76%）中联合可用。该流水线依次执行异常检测（修正Z分数法、孤立森林算法）、双变量主成分分析（PCA）、聚类以及机器学习分析。

    arXiv:2610.32438v1 Announce Type: cross  Abstract: Long-term environmental IoT/LPWAN deployments often aggregate heterogeneous data streams from diverse devices without systematic auditing of each variable's actual coverage prior to analysis-an increased risk when multiple sensor types coexist within the same network. This work proposes a C++17 processing pipeline applied to the Tour Perret LoRaWAN dataset (421 937 frames, June 2021--June 2023, 5 devices). A systematic coverage audit integrated into the extraction phase reveals that CO2 is completely missing (0% coverage) and pressure is reported only by an isolated base module (Wyres, 4.24% of the corpus), invalidating the initial hypothesis of a joint four-variable analysis. The analysis is refocused on the Temperature--Humidity pair, jointly available across 205{,}005 frames (95.76%) from 4 Elsys EMS sensors. The pipeline sequentially performs anomaly detection (modified Z-score, Isolation Forest), bivariate Principal Component Anal
    
[^78]: 关于时变有向图上随机Push-Pull算法的极限

    On the Limits of Stochastic Push-Pull over Time-Varying Digraphs

    [https://arxiv.org/abs/2610.32422](https://arxiv.org/abs/2610.32422)

    本文通过构造周期网络上的反例并建立匹配的收敛上下界，证明了随机Push-Pull算法在时变有向图上可能失去线性加速，其随机收敛率由有效节点数 $n_{\mathrm{eff}}$ 而非网络规模 $n$ 主导。

    

    随机Push-Pull算法在静态强连通有向图上能够实现关于网络规模 $n$ 的线性加速，但这种加速是否在时变网络上依然成立尚不清楚。在本工作中，我们在周期为二的网络上构造了一个二次型去中心化优化问题，表明随机Push-Pull在时变有向图上可能无法获得线性加速。对于 $p$-周期网络，我们确定了一个有效节点数 $n_{\mathrm{eff}}\le n$，它支配着主导的随机项。具体而言，对于光滑非凸目标函数，我们建立了采用恒定步长的随机Push-Pull的收敛上界和匹配下界，两者均呈现 $\Theta(1/\sqrt{n_{\mathrm{eff}} T})$ 的主导阶依赖关系。当 $p=1$ 时，恒等式 $n_{\mathrm{eff}}=n$ 恢复了静态有向图上通常的线性加速；然而，周期网络可能存在 $n_{\mathrm{eff}}=o(n)$，从而证明了线性加速的丧失。

    arXiv:2610.32422v1 Announce Type: cross  Abstract: Stochastic Push-Pull achieves linear speedup with respect to the network size $n$ on static strongly connected digraphs, but whether this speedup persists on time-varying networks remains unclear. In this work, we construct a quadratic decentralized problem on period-two networks showing that stochastic Push-Pull may fail to attain linear speedup over time-varying digraphs. For $p$-periodic networks, we identify an effective node count $n_{\mathrm{eff}}\le n$ that governs the leading stochastic term. Specifically, for smooth nonconvex objectives, we establish convergence upper bounds and matching lower bounds for stochastic Push-Pull with constant stepsizes, both exhibiting the leading-order dependence $\Theta(1/\sqrt{n_{\mathrm{eff}} T})$. When $p=1$, the identity $n_{\mathrm{eff}}=n$ recovers the usual linear speedup for static digraphs. However, periodic networks can have $n_{\mathrm{eff}}=o(n)$, demonstrating the loss of linear spe
    
[^79]: 动力学随机化何时能真正检验上下文推断？

    When Does Dynamics Randomization Test Context Inference?

    [https://arxiv.org/abs/2610.32414](https://arxiv.org/abs/2610.32414)

    论文提出“预测余量”这一形式化指标来量化随机动力学环境中上下文编码器可获得的额外预测信息，发现许多环境下该余量很低，并建议将其与随机化参数一同报告，以判断显式上下文推断是否真正带来收益。

    

    采用随机化动力学的环境被广泛用于探究世界模型是否需要显式的上下文推断。然而，随机化可能只为上下文编码器留下很少可供补充的预测信息，因为不同的潜在参数可能在访问过的状态上产生相似的转移，或者这些状态本身可能已经揭示了当前的上下文。我们将剩余的单步可预测信息形式化为“预测余量”，并通过在固定分布与实际访问的状态-动作分布下比较核函数，来区分查询覆盖范围的变化与上下文权重的变化。我们分析了离散和连续环境，发现当核函数之间差异很小，或当访问分布缩小了原本较大的差异时，预测余量都很低。与此同时，重置状态分量会增加预测余量，这表明上下文证据是在状态中逐步累积的。我们建议在报告随机化参数的同时报告这一预测余量指标，以便衡量显式上下文推断所带来的收益。

    arXiv:2610.32414v1 Announce Type: new  Abstract: Environments with randomized dynamics are widely used to ask whether world models need explicit context inference. However, randomization can leave little predictive information for a context encoder to add, because distinct latent parameters may produce similar transitions at visited states, or those states may already reveal the active context. We formalize the remaining one-step information as predictive headroom and compare kernels under fixed and visited state--action distributions to distinguish changes in query coverage from changes in context weights. We analyze discrete and continuous environments and find low headroom both when kernels differ little and when visitation reduces an otherwise large discrepancy. At the same time, resetting state components increases headroom, showing that context evidence accumulates in state. We propose reporting this headroom alongside the randomized parameters, so that gains from explicit contex
    
[^80]: PASTA：提示感知的退火扩散采样

    PASTA: Prompt-Aware Annealed Diffusion Sampling

    [https://arxiv.org/abs/2610.32390](https://arxiv.org/abs/2610.32390)

    PASTA提出了一种免训练的提示感知退火扩散采样框架，通过在共享潜变量轨迹上对允许对象的区域选择性施加语义引导，在文本到图像生成中严格保留指定区域，同时保持视觉连贯性。

    

    尽管扩散模型能够生成高质量的图像，但现实世界的视觉设计需要有意识地保留干净区域，用于放置标题、标志或UI元素。现有的布局感知文本到图像方法难以严格保留预留区域，常常引入不必要的内容，使下游编辑变得复杂。现有方法要么通过操纵注意力机制来移除内容，要么通过噪声级干预来抑制条件控制。然而，这些方法牺牲了视觉连贯性，会在预留区域和非预留区域之间产生平淡的区域或明显的风格不一致。为了解决这一局限性，我们提出了提示感知共享潜变量免训练退火扩散采样（PASTA），这是一个免训练框架，通过在采样过程中选择性地向允许放置对象的区域应用语义引导，来强制执行预留区域约束。PASTA在共享潜变量轨迹上运行，其中完整提示与上下文……

    arXiv:2610.32390v1 Announce Type: cross  Abstract: While Diffusion models produce high-quality images, real-world visual design requires intentionally preserving clean regions for headlines, logos, or UI elements. Existing layout-aware text-to-image methods struggle to strictly preserve reserved regions, often introducing unwanted content that complicates downstream editing. Existing approaches either manipulate attention to displace content or suppress conditioning via noise-level interventions. However, these methods sacrifice visual coherence, producing flat regions or noticeable style inconsistencies between reserved and non-reserved areas. To address this limitation, we propose a Prompt-Aware Shared-Latent Training-Free Annealed Diffusion Sampling (PASTA), a training-free framework that enforces reserved-region constraints by selectively applying semantic guidance to object-allowed regions during sampling. PASTA operates on a shared latent trajectory, where full-prompt and context
    
[^81]: KnitTrans：面向高效且有效的大语言模型强化学习的协同式轨迹传输

    KnitTrans: Coordinated Trajectory Transfer for Effective and Efficient LLM Reinforcement Learning

    [https://arxiv.org/abs/2610.32389](https://arxiv.org/abs/2610.32389)

    KnitTrans 通过保持约束的迷你批次准入和按需轨迹数据访问等协同设计，优化大语言模型强化学习中从采样到训练的轨迹传输，在降低架构陈旧性的同时提升学习质量与系统效率。

    

    在大型语言模型强化学习（RL）中，从采样到训练的轨迹传输会同时影响学习质量和系统效率。强制执行跨整个批次的训练要求可能会延迟已完成轨迹的准入，从而导致更高的架构陈旧性，即相对于同置执行而言，解耦流水线所带来的额外参数版本差距。除准入问题外，数据访问与训练消费之间的错位在传输大规模轨迹负载时会造成冗余的数据移动和不必要的等待。我们提出了 KnitTrans，一个通过三项设计来协调从采样到训练路径的系统。第一，保持约束的迷你批次准入机制在早期迷你批次组成上提供了灵活性，同时检查剩余容量是否足以满足剩余配额。第二，按需轨迹数据访问将存储单元与消费方分配相匹配，并使微型批次（计算）重叠（原文摘要在此处截断）

    arXiv:2610.32389v1 Announce Type: cross  Abstract: Trajectory transfer from rollout to training affects both learning quality and system efficiency in large language model reinforcement learning (RL). Enforcing batch-wide training requirements can delay the admission of completed trajectories, resulting in higher architectural staleness: the additional parameter-version gap from disaggregated pipelining relative to colocated execution. Beyond admission, misalignment between data access and training consumption causes redundant movement and unnecessary waiting when transferring large trajectory payloads. We present KnitTrans, a system that coordinates the rollout-to-training path through three designs. First, constraint-preserving mini-batch admission allows flexibility in early mini-batch composition, while checking that enough capacity remains to satisfy the remaining quotas. Second, on-demand trajectory data access matches storage units to consumer assignments and overlaps micro-batc
    
[^82]: 面向语言学习者母语识别的声学与词汇信号

    Acoustic and Lexical Signals for Native Language Identification of Language Learners

    [https://arxiv.org/abs/2610.32382](https://arxiv.org/abs/2610.32382)

    该研究首次将基于WavLM的音频分类器与零样本GPT-4o转写文本分类器相结合用于第二语言学习者的母语识别，发现音频方法在低熟练度（A1）时更优，而文本方法在中高熟练度（A2–C1）时更具优势。

    

    第二语言学习者的口语母语（L1）识别传统上依赖于语音的声学属性，而通过分析语音转写文本来探寻母语痕迹的研究关注相对较少。本文提出了一项利用口语学习者英语语料库中的语音及其对应转写文本进行母语识别的研究。我们将基于WavLM的音频母语分类器与在转写文本上运行的零样本GPT-4o分类器进行对比，并使用CEFR平衡的评估集在不同熟练度水平上分析二者的预测结果。实验表明，当完整转写文本可用时，音频分类器总体上优于基于转写文本的方法，但两者的相对优势随学习者熟练度而变化：音频方法在A1级别表现更好，而从A2到C1级别，在拥有完整转写文本的情况下文本方法表现更佳。基于转写文本的准确率通常随熟练度提高而上升，而声学性能则保持……（摘要原文在此处被截断）

    arXiv:2610.32382v1 Announce Type: new  Abstract: Spoken native language (L1) identification for second language learners has traditionally relied on acoustic attributes of speech. Comparatively little attention has been paid to analysing speech transcripts for L1 traces. In this paper, we present a study of L1 identification using both speech and corresponding transcripts from a spoken learner English corpus. We compare a WavLM-based audio L1 classifier with a zero-shot GPT-4o classifier operating on transcripts and analyse their predictions across proficiency levels using a CEFR-balanced evaluation set. Our experiments show that the audio classifier outperforms the transcript-based approach overall when full transcripts are available, although their relative strengths vary with learner proficiency. Audio performs better at A1, whereas text performs better from A2 to C1 with full transcripts. Transcript-based accuracy generally increases with proficiency, while acoustic performance rem
    
[^83]: 临床AI中隐藏亚组差异的检测与验证

    Detection and Validation of Hidden Subgroup Disparities in Clinical AI

    [https://arxiv.org/abs/2610.32362](https://arxiv.org/abs/2610.32362)

    该论文提出通过算法化亚组发现在缺乏丰富元数据的情况下检测临床AI模型中隐藏的患者亚组性能差异，并对所发现亚组的性能差距和语义连贯性进行了评估与验证。

    

    用于临床工作流程的机器学习（ML）模型在不同患者群体中的表现往往存在系统性差异，这引发了对公平性和患者安全的担忧。传统的亚组分析只有在有丰富元数据（如患者人口统计学信息）可用且能捕捉性能差异来源时才能揭示此类差异，而这些条件在实践中很少得到满足。算法化亚组发现方法通过从学习到的特征表示中识别连贯的群体来提供了一种解决方案，但由于真实的潜在分层是未知的，这些方法难以评估。因此，亚组发现尚未被系统地应用于性能监测或对其进行评估。我们的主要目标是评估算法化亚组发现用于性能监测的效果，评估所发现亚组的性能差距和语义连贯性。我们在包含223,228张胸部X光片的数据集中研究了心脏肥大分类任务……

    arXiv:2610.32362v1 Announce Type: new  Abstract: Machine learning (ML) models used in clinical workflows often perform systematically differently across patient groups, raising concerns about equity and patient safety. Traditional subgroup analyses can expose such disparities only when rich metadata (e.g., patient demographics) is available and captures the sources of performance variation, conditions rarely met in practice. Algorithmic subgroup discovery methods offer a solution by identifying coherent groups from learned feature representations, but because true underlying stratifications are unknown, these methods are hard to evaluate. As a result, subgroup discovery has not been systematically applied or assessed for performance monitoring. Our main objective is to evaluate algorithmic subgroup discovery for performance monitoring, assessing the discovered subgroups' performance gaps and semantic coherence. Investigating cardiomegaly classification in a dataset of 223,228 chest X-r
    
[^84]: 具有任意标签空间的多类在线学习最优遗憾

    Optimal Multiclass Online Regret with Arbitrary Label Spaces

    [https://arxiv.org/abs/2610.32339](https://arxiv.org/abs/2610.32339)

    该论文确定了具有任意（可能无限）标签空间的不可知多类在线学习的最优遗憾上界 $O(\min\{T,\sqrt{dT}\})$，解决了 Hanneke 等人（2023）提出的开放问题。

    

    我们确定了在具有任意（可能是无限的）标签空间的不可知多类在线学习中的最优遗憾。对于每个多类 Littlestone 维度为 $d<\infty$ 的假设类，我们证明了在 $T$ 轮中 $O(\min\{T,\sqrt{dT}\})$ 的遗憾上界，该上界与已知的下界相匹配。这解决了 Hanneke 等人（2023）提出的一个问题，并将 Alon 等人（2021）的最优二分类遗憾界推广到了多类情形。我们的证明将 Hanneke 等人（2023）的单边专家覆盖加强为分数覆盖，并将该覆盖与到有限专家博弈的极小极大归约相结合，其最佳响应奖励提供了链式论证所需的分歧界。

    arXiv:2610.32339v1 Announce Type: cross  Abstract: We determine the optimal regret in agnostic multiclass online learning with arbitrary, possibly infinite, label spaces. For every hypothesis class of multiclass Littlestone dimension $d<\infty$, we prove a regret upper bound of $O(\min\{T,\sqrt{dT}\})$ over $T$ rounds, matching the known lower bound. This resolves a question of Hanneke et al. (2023) and extends the optimal binary regret bound of Alon et al. (2021). Our proof strengthens the one-sided expert cover of Hanneke et al. (2023) to a fractional cover. We combine this cover with a minimax reduction to a finite expert game, whose best-response rewards provide the disagreement bounds needed for chaining.
    
[^85]: TENET：用于多元时间序列异常检测的时间点编码网络

    TENET: Time-point Encoding Network for Multivariate Time Series Anomaly Detection

    [https://arxiv.org/abs/2610.32337](https://arxiv.org/abs/2610.32337)

    TENET将无监督多元时间序列异常检测重新表述为当前点系统状态兼容性估计，通过以当前点为中心的时间编码和上下文自适应混合构建潜在状态表示，从而克服了观测空间差异作为异常指标不可靠的问题。

    

    在无监督多元时间序列异常检测（MTSAD）中，局部变化会放大重构和预测误差，使得观测空间差异成为判断系统异常行为的不可靠指标。由于潜在空间评分可以缓解这一局限，我们将单类多元时间序列异常检测重新表述为当前点系统状态兼容性估计问题，并提出TENET——用于多元时间序列异常检测的时间点编码网络。我们将当前点系统状态定义为目标时间点的潜在表示，该表示由输入窗口中的变量级时间上下文构建。TENET通过一个时间层来构建该状态：该层检索相对于当前点的窗口上下文，使用以当前点为中心的嵌入对时间进行编码，并将检索到的上下文与当前表示进行自适应混合。借助这一设计，TENET不再将输入窗口视为需要重构的对象（摘要在此处截断）。

    arXiv:2610.32337v1 Announce Type: cross  Abstract: In unsupervised multivariate time series anomaly detection (MTSAD), local variation can inflate reconstruction and forecasting errors, making observation-space discrepancy an unreliable indicator of abnormal system behavior. Since latent-space scoring can mitigate this limitation, we reframe one-class MTSAD as current-point system-state compatibility estimation and propose TENET, the Time-point Encoding Network for Multivariate Time Series Anomaly Detection. We define the current-point system state as a latent representation of the target time point, constructed from variable-level temporal context in the input window. TENET constructs this state through a temporal layer that retrieves window context relative to the current point, encodes time with current-point-centered embeddings, and adaptively mixes the retrieved context with the current representation. With this design, TENET treats the input window not as the object to reconstruc
    
[^86]: 何时信任记忆：分布偏移下检索引导的概率时空预测

    When to Trust Memory: Retrieval-Guided Probabilistic Spatiotemporal Forecasting under Distribution Shift

    [https://arxiv.org/abs/2610.32329](https://arxiv.org/abs/2610.32329)

    提出 REMIND 框架，通过检索机制量化当前观测与历史模式的匹配程度来估计分布偏移程度，从而在非平稳环境下自适应地生成校准良好且可靠的概率时空预测。

    

    时空预测在现实应用中扮演着重要角色，其中概率方法能够为决策提供不确定性估计。现实世界的系统本质上是非平稳的，导致训练数据与测试数据之间存在分布偏移。然而，现有的概率方法主要关注标准设定下的数据分布建模，而现有的分布偏移处理方法往往以粗粒度的方式应对此类偏移，无法刻画偏移程度并相应地调整预测策略。因此，这些方法常常产生校准不佳且不可靠的预测。为解决这一问题，我们提出了 REMIND，一个面向分布偏移下时空预测的检索引导概率框架。REMIND 利用检索机制来估计当前观测在多大程度上得到历史模式的支持，从而在不同偏移状态下实现自适应预测。它采用时空检索（原文在此处截断）。

    arXiv:2610.32329v1 Announce Type: new  Abstract: Spatiotemporal forecasting plays an important role in real-world applications, where probabilistic approaches provide uncertainty estimates for decision-making. Real-world systems are inherently non-stationary, leading to distribution shifts between training and test data. However, existing probabilistic methods mainly focus on modeling data distributions under standard settings, while distribution shift methods often handle such shifts in a coarse manner, failing to characterize shift degree and adapt prediction strategy accordingly. As a result, they often produce miscalibrated and unreliable forecasts. To address this issue, we propose REMIND, a retrieval-guided probabilistic framework for spatiotemporal forecasting under distribution shift. REMIND leverages retrieval to estimate how well current observation are supported by historical patterns, enabling adaptive prediction under different shift regimes. It employs spatiotemporal retr
    
[^87]: SKG实践：跨四个领域的OOD泛化案例研究

    SKG in Practice: OOD Generalization Cases Across FourDomains

    [https://arxiv.org/abs/2610.32314](https://arxiv.org/abs/2610.32314)

    SKG推理框架的工程实现在耗散机械系统、目标检测Sim-to-Real、序列建模和分子动力学四个领域的案例研究中，均显著优于神经基线方法，大幅提升了OOD（分布外）泛化性能。

    

    本文介绍了SKG推理框架的工程实现方法，并提供了跨四个领域的完整案例研究。案例1，二维耗散机械系统：OOD MSE比最佳神经基线低1.69至4.12倍；在受控条件下，状态相关度量带来了10^8量级的改进。案例2，目标检测的模拟到真实迁移：真实机器上的mAP从0.2564提升至0.4964（+24.00个百分点，单次运行实验，32张图像，点估计），AP75从0.1736提升至0.5765（+40.29个百分点），且无需在真实机器上重新训练。案例3，序列建模从正弦到方波：OOD MSE为0.0296，而Transformer为0.0641（5个随机种子的平均值，p=0.024）。案例4，分子动力学MD17：ID@100 ADE为0.0285，而ESTAG为0.1467；OOD@100为0.0316，而ESTAG为0.1411（单种子）。对于每个案例，我们展示了研究目标、所应用的SKG理论、参数设置、基线比较协议、数据……（原文摘要截断）

    arXiv:2610.32314v1 Announce Type: new  Abstract: This paper presents the engineering implementation methods of the SKG inference framework and provides complete case studies across four domains. Case 1, 2D Dissipative Mechanical System: OOD MSE is 1.69 to 4.12 times lower than the best neural baseline; the state-dependent metric yields a 10^8-order improvement under controlled conditions. Case 2, Target Detection Sim-to-Real: Real-machine mAP increases from 0.2564 to 0.4964 (+24.00 points, single-run experiment, 32 images, point estimate), and AP75 increases from 0.1736 to 0.5765 (+40.29 points), with no real-machine retraining required. Case 3, Sequence Modeling sine to square: OOD MSE is 0.0296 versus Transformer 0.0641 (mean of 5 seeds, p=0.024). Case 4, Molecular Dynamics MD17: ID@100 ADE is 0.0285 versus ESTAG 0.1467; OOD@100 is 0.0316 versus 0.1411 (single seed). For each case, we present the objective, applied SKG theory, parameter settings, baseline comparison protocol, data re
    
[^88]: GeoLEON-CLIP：面向地理区域表征的图锚定多模态对比学习

    GeoLEON-CLIP: Graph-Anchored Multimodal Contrastive Learning for Geographic Region Representation

    [https://arxiv.org/abs/2610.32311](https://arxiv.org/abs/2610.32311)

    该论文提出GeoLEON-CLIP框架，以冻结的预训练LEON图嵌入作为语义锚点，通过三模态对比学习将OpenStreetMap图结构、卫星影像和文本描述对齐，从而充分利用结构化地图数据并实现最优的跨模态地理区域表征与检索性能。

    

    地理区域本质上是多模态的：卫星影像捕捉视觉形态，文本传达人类可理解的语义，地图结构则描述城市环境。现有的地理空间对比学习方法通常只对齐图像-文本或图像-位置对，未能充分利用结构化地图数据。我们提出GeoLEON-CLIP，这是一个图锚定的对比学习框架，对齐每个以H3索引的区域的三个视图：源自OpenStreetMap的图嵌入、卫星图像裁剪区域以及文本描述。冻结的预训练LEON图嵌入在视觉与语言之间提供语义锚点。冻结的图、图像和文本编码器通过可训练的残差投影头相连接，并以对称的三模态对比目标进行训练。实验表明，该模型实现了强大的图-图像和图-文本对齐效果；在混合分辨率池中，图-图像检索取得了最佳的跨模态性能。为了评估锚定假设，我们...

    arXiv:2610.32311v1 Announce Type: cross  Abstract: Geographic regions are multimodal: satellite imagery captures visual morphology, text conveys human-interpretable semantics, and map structures describe urban context. Existing geospatial contrastive methods typically align image-text or image-location pairs, underusing structured map data. We introduce GeoLEON-CLIP, a graph-anchored contrastive framework that aligns three views of each H3-indexed region: an OpenStreetMap-derived graph embedding, a satellite image crop, and a textual description. Frozen pretrained LEON graph embeddings provide a semantic anchor between vision and language. Frozen graph, image, and text encoders are connected by trainable residual projection heads and trained with a symmetric tri-modal contrastive objective. Experiments show strong graph-image and graph-text alignment; graph-image retrieval achieves the best cross-modal performance in a mixed-resolution pool. To evaluate the anchoring hypothesis, we pro
    
[^89]: 使用混合Transformer架构的电动汽车车内噪声监督式源分离

    Supervised Source Separation of Electric Vehicle Interior Noise Using a Hybrid Transformer Architecture

    [https://arxiv.org/abs/2610.32310](https://arxiv.org/abs/2610.32310)

    本文将原本用于音乐分离的混合时-频谱Transformer架构适配到电动汽车领域，提出一种监督式双耳源分离方法，可将车内噪声分解为风噪、胎噪、动力总成噪声和辅助驱动噪声四种成分，从而无需昂贵的测量和传递路径分析即可支持针对性声学工程。

    

    电动汽车的车内声音是多种不同噪声贡献的叠加——包括风噪、轮胎噪声、动力总成（电驱动单元和逆变器）噪声以及辅助驱动噪声——这些成分的相对平衡决定了车内的整体响度和感知声品质。将录制的车内信号分解为这些声源贡献是开展针对性声学工程的前提，但这需要耗费大量精力的测量工作以及传递路径分析领域的专业知识。本工作将这一分解问题形式化为一个包含四个目标声源贡献的监督式双耳声源分离问题，并将最初为音乐分离开发的混合时-频谱Transformer架构适配到技术系统的声源分离领域。为消除声源贡献之间的歧义，引入了以运行状态为形式的条件输入机制……

    arXiv:2610.32310v1 Announce Type: cross  Abstract: The interior sound of an electric vehicle is a superposition of several distinct noise contributions - wind noise, tire noise, powertrain (electric drive unit and inverter) noise, and auxiliary-drive noise - whose relative balance governs both the overall loudness and the perceived sound quality in the vehicle interior. Decomposing a recorded vehicle interior signal into these source contributions is a prerequisite for targeted acoustic engineering, but requires effortful measurement campaigns as well as expertise in the field of transfer path analysis. This work formulates the decomposition as a supervised, binaural source-separation problem with four target source contributions and adapts a hybrid time-spectrogram transformer architecture, originally developed for music demixing, to the domain of sound source separation for technical systems. To disambiguate source contributions, conditioning is introduced in the form of operating-st
    
[^90]: DMAIC-Bench：多阶段分析任务中智能体从自我反思到自我克制

    DMAIC-Bench: From Self-Reflection to Self-Restraint for Agents in Multi-Stage Analytical Tasks

    [https://arxiv.org/abs/2610.32305](https://arxiv.org/abs/2610.32305)

    该论文提出 DMAIC-Bench 基准，通过基于精益六西格玛的成对反事实任务首次系统评估大语言模型智能体的“自我克制”能力（即依据证据决定继续还是停止），发现模型在 GO 决策上准确率达 92%，但在 NO-GO 决策上仅约 19%。

    

    多阶段分析任务要求大语言模型智能体利用中间证据来决定是继续、转向还是停止。自我反思（即智能体批判自身输出的过程）已被广泛研究，然而批判本身并不能表明证据是否改变了下一步行动。我们将这种能力称为“自我克制”。现有的智能体基准测试主要衡量任务完成度或输出质量，而未对这一能力进行检验。DMAIC-Bench 通过 97 个种子任务来评估这一能力，这些任务覆盖四个行业的 15 个工业案例，并围绕精益六西格玛明确的统计学前提构建。成对的反事实变体要求智能体在同一工作流位置采取相反的行动，从而将有依据的克制与不加区分的停止区分开来。我们使用私有评分器评估执行轨迹，并通过独立专家评审评估最终结论。在对来自六家提供商的 13 个模型的评估中，执行层面的路由准确率在匹配的 GO 决策上平均为 92%，而在 NO-GO 决策上仅为 19%。

    arXiv:2610.32305v1 Announce Type: new  Abstract: Multi-stage analytical tasks require LLM agents to use intermediate evidence to decide whether to continue, redirect or stop. Self-reflection, in which an agent critiques its output, is widely studied, yet a critique does not show whether evidence changed the next action. We call that capability self-restraint. Existing agent benchmarks largely measure task completion or output quality and leave it untested. DMAIC-Bench evaluates it with 97 seeded tasks across 15 industrial cases in four sectors, built around Lean Six Sigma's explicit statistical prerequisites. Paired counterfactual variants require opposite actions at the same workflow position, distinguishing warranted restraint from indiscriminate stopping. We assess execution traces with a private scorer and final claims through independent expert review. Across 13 models from six providers, execution-level routing accuracy averages 92% on matched GO decisions and 19% on NO-GO decisi
    
[^91]: CoPiT：一种用于非结构网格气动场预测的条件位置诱导Transformer

    CoPiT: A conditional Position-induced Transformer for aerodynamic field prediction on unstructured meshes

    [https://arxiv.org/abs/2610.32299](https://arxiv.org/abs/2610.32299)

    本文提出CoPiT模型，以PiT为几何骨干并通过多阶段FiLM调制显式注入全局条件向量，实现了非结构网格上高效准确的气动场预测。

    

    高保真雷诺平均Navier-Stokes（RANS）模拟仍然是气动分析的标准工具，然而每次在非结构网格上进行求解的计算成本都很高，使得设计探索、优化和多工况评估等多查询任务代价高昂。神经算子为将几何外形和运行条件映射到流场提供了对网格灵活的代理模型；然而，在许多现有架构中，诸如自由流状态和设计变量等全局参数只是与逐点输入拼接在一起，导致它们对中间特征的影响是隐式的。本文提出了CoPiT，一种条件位置诱导Transformer（Conditional Position-induced Transformer），它以位置诱导Transformer（PiT）作为几何骨干，并通过多阶段特征级线性调制（FiLM）注入全局条件向量，从而使参数化调制显式化，同时保持几何……（原文摘要不完整，此处为现有内容之翻译）

    arXiv:2610.32299v1 Announce Type: cross  Abstract: High-fidelity Reynolds-averaged Navier--Stokes (RANS) simulations remain a standard tool for aerodynamic analysis, yet each solve on an unstructured mesh is computationally expensive, making many-query tasks such as design exploration, optimization, and multi-condition evaluation costly. Neural operators provide mesh-flexible surrogates for mapping geometry and operating conditions to flow fields; however, in many existing architectures, global parameters such as the freestream state and design variables are concatenated with pointwise inputs, leaving their influence on intermediate features implicit. This paper proposes CoPiT, a \textbf{Co}nditional \textbf{P}osition-induced \textbf{T}ransformer that builds on the Position-induced Transformer (PiT) as its geometric backbone and injects a global condition vector through multi-stage feature-wise linear modulation (FiLM), thereby making parametric modulation explicit while preserving geo
    
[^92]: 结合图像结构补充DINO特征以实现部件发现

    Complementing DINO Features with Image Structure for Part Discovery

    [https://arxiv.org/abs/2610.32293](https://arxiv.org/abs/2610.32293)

    该论文揭示了基于DINO特征的部件发现边界模糊的根源在于ViT刚性补丁网格导致跨边界补丁混合了多个部件的像素，并提出用图像结构（来自边界检测器的边界信息）补充DINO特征，从而在无需重新训练的情况下实现更精确的部件发现。

    

    诸如DINO之类的自监督视觉Transformer（ViT）编码了丰富的语义信息，使得无需训练即可将物体从背景中分割出来，并通过特征聚类发现物体的部件。本文聚焦于部件发现任务：将前景物体划分为若干语义部件。现有方法产生的部件边界不够精确。我们将这一问题追溯到训练DINO时使用的刚性ViT补丁网格：跨越边界的补丁会包含来自多个部件的像素。虽然DINO基于不变性的训练本可以将这种像素级混合吸收为干净的特征，但通过对超过26K个前景补丁的实验表明事实并非如此：与边界重叠的补丁会产生熵更高、分配置信度更低的特征，从而使预测的部件边界变得模糊。与其投入巨大精力在几何连贯的区域上重新训练DINO，一种更简单的方案是直接从边界检测器获取边界信息……（原文摘要在此处截断）

    arXiv:2610.32293v1 Announce Type: cross  Abstract: Self-supervised Vision Transformers (ViTs) like DINO encode rich semantics, enabling training-free segmentation of objects from the background and discovery of their parts via feature clustering. This paper focuses on part discovery: partitioning a foreground object into semantic parts. Existing approaches produce imprecise part boundaries. We trace this to the rigid ViT patch grid used in training DINO, where patches straddling boundaries contain pixels from multiple parts. While DINO's invariance-based training could absorb this pixel-level mixing into clean features, experiments with over 26K foreground patches show it does not: boundary-overlapping patches produce features with higher entropy and lower assignment confidence, blurring predicted part boundaries. Rather than make the massive effort of retraining DINO over geometrically coherent regions, an easier approach is to take boundaries directly from a boundary detector, which 
    
[^93]: ReLEAP：通过跨任务统一与半监督标注提升视觉数据集的价值

    ReLEAP: Upcycling Vision Datasets through Cross-Task Unification and Semi-Supervised Annotation

    [https://arxiv.org/abs/2610.32289](https://arxiv.org/abs/2610.32289)

    ReLEAP提出了一个可扩展的Python框架，通过数据源抽象、跨任务标注重用和统一层级分类体系来整合异构视觉数据集，并结合半监督标注实现对现有数据的系统性回收与标注升级，在精准农业领域整合十五个数据集验证了其有效性。

    

    开发稳健的视觉模型需要大量标注数据，尤其是在可变环境条件下涉及细粒度类别区分的任务。即使在一些拥有众多现有数据集的领域，标注格式、标签定义和目标学习任务的差异也阻碍了数据的重用与整合。为应对这一挑战，我们提出了ReLEAP，这是一个可扩展的Python框架，通过数据源抽象、跨学习任务的系统性标注重用，以及将标签映射到共享的层级分类体系，来统一异构图像数据集。我们通过整合十五个涵盖广泛植物物种和标注类型的多样化数据集，展示了该框架在精准农业中的适用性。除了数据统一之外，ReLEAP还能实现对现有数据的系统性回收利用，并为基于学习的标注升级提供坚实基础。利用这一基础，我们……（摘要在此处截断）

    arXiv:2610.32289v1 Announce Type: cross  Abstract: Developing robust vision models requires large amounts of annotated data, especially for tasks involving fine-grained class distinctions under variable environmental conditions. Even in domains with numerous existing datasets, differences in annotation formats, label definitions and targeted learning tasks hinder data reuse and integration. To address this challenge, we introduce ReLEAP, a scalable Python framework for unifying heterogeneous image datasets through data-source abstraction, systematic reuse of annotations across learning tasks, and mapping of labels to a shared hierarchical taxonomy. We demonstrate its applicability in precision agriculture by integrating fifteen diverse datasets covering a broad range of plant species and annotation types. Beyond unification, ReLEAP enables systematic recycling of existing data and provides a robust foundation for learning-based annotation upcycling. Leveraging this foundation, we condu
    
[^94]: RootDown：在边缘设备上部署与基准测试甜菜根检测模型

    RootDown: Deploying and Benchmarking Models for Beet-Root Detection on Edge Devices

    [https://arxiv.org/abs/2610.32275](https://arxiv.org/abs/2610.32275)

    该论文提出了 RootDown 端到端图像检测流水线，结合基于 SAM 3 的自动标注策略与专用基准数据集，在边缘设备上实现了装载场景中部分遮挡甜菜的检测，为甜菜自动化质量检测迈出第一步。

    

    甜菜收获和储存后的质量检测对于保持糖分含量并确保最佳加工效率至关重要。将收获的甜菜转移至田间堆垛或卡车的过程中，为在装载输送带上对单个甜菜进行自动目视检查提供了机会。然而，这是一项具有挑战性的任务，因为大量的甜菜会造成部分重叠和遮挡，使得标注和检测都变得复杂。在这项工作中，我们提出了一种端到端的基于图像的流水线，用于在装载场景中检测部分遮挡的甜菜，作为迈向自动化质量检测的第一步。我们引入了一个专门的基准数据集，并提出了一种利用 SAM 3 基础模型生成边界框的自动标注策略。我们比较了在人工标注标签与自动生成标签上训练的模型。此外，我们还研究了部署方面的考量，对多种推理配置进行了基准测试。

    arXiv:2610.32275v1 Announce Type: cross  Abstract: Quality inspection of sugar beets after harvest and storage is critical to preserve sugar content and ensure optimal processing efficiency. The transfer of harvested beets to field clamps or trucks provides an opportunity to visually inspect individual beets automatically on loader conveyors. However, this is a challenging task as the large number of beets causes partial overlap and occlusion, complicating both annotation and detection. In this work, we propose an end-to-end image-based pipeline for detecting partially occluded sugar beets in loading scenarios as a first step toward automated quality inspection. We introduce a dedicated benchmark and present an automatic annotation strategy that leverages the SAM 3 foundation model to generate bounding boxes. We compare models trained on manually annotated and automatically generated labels. Additionally, we investigate deployment considerations by benchmarking multiple inference confi
    
[^95]: 具有无限记忆的链的经验集中不等式

    Empirical concentration for chains with infinite memory

    [https://arxiv.org/abs/2610.32265](https://arxiv.org/abs/2610.32265)

    本文将经验集中不等式从马尔可夫链推广到具有无限记忆的平稳链，在Wasserstein压缩条件下建立了有界差分不等式，并仅凭单条轨迹估计依赖系数，从而得到可直接使用的经验集中界。

    

    时间序列的集中不等式通常涉及衡量依赖性的未知量，这妨碍了它们直接用作置信界或学习算法的准则。近期的工作针对平稳马尔可夫链证明了经验集中不等式，其中未知量被估计出来。我们将这一方法扩展到马尔可夫情形之外，考虑具有无限记忆的平稳链。我们首先在可求和的Wasserstein压缩条件下，于有界Polish度量空间上建立了有界差分不等式。随后，我们从单条轨迹估计其依赖系数：当状态空间有限时使用转移计数，当状态空间连续时使用核条件密度估计的导数。我们给出了一般的误差界，并说明这些误差界对记忆尾部的依赖关系。最后，通过一个固定的乘法安全因子，将这些估计量转化为经验集中不等式。

    arXiv:2610.32265v1 Announce Type: new  Abstract: Concentration inequalities for time series usually involve unknown quantities measuring dependence. This prevents their direct use as confidence bounds, or as criteria for learning algorithms. Recent works proved empirical concentration inequalities for stationary Markov chains, where the unknown quantities are estimated. We extend this approach beyond the Markov case, by considering stationary chains with infinite memory. We first establish a bounded-differences inequality on bounded Polish metric spaces under a summable Wasserstein contraction condition. We then estimate its dependence coefficients from a single trajectory, using either transition counts when the state space is finite, or derivatives of kernel conditional-density estimates for a continuous state space. We give general error bounds and illustrate their dependence on the memory tail. Finally, a fixed multiplicative safety factor turns these estimators into empirical conc
    
[^96]: 面向分布外检测的抗相干性正则化绝对缩放方法

    Anti-coherence Regularized Absolute Scaling for Out-of-Distribution Detection

    [https://arxiv.org/abs/2610.32255](https://arxiv.org/abs/2610.32255)

    提出绝对值缩放修复与抗相干性正则化，解决了SCALE方法因负激活正负抵消而在ViT等骨干网络上性能退化的问题，无需训练数据即可提升分布外检测性能。

    

    分布外（OOD）检测被公认为AI安全中的一个关键问题。SCALE是激活整形家族中一种有效的后处理OOD检测方法，它无需训练数据，并通过基于百分位的幅度比对倒数第二层激活进行重新缩放。然而，当存在负激活时，该比值定义不良，因为正负贡献在底层求和中相互抵消，导致SCALE在ViT等骨干网络上性能退化。我们通过一种“绝对值修复”来解决这一问题，即在绝对值化的激活上计算比值，而非通过ReLU丢弃负激活。基于这一修正后的尺度，我们进一步提出抗相干性正则化，通过惩罚样本中与最大logit符号不匹配的最大logit贡献占比，从而抑制对预测类别不可靠的证据。尽管无需访问训练数据……（原文摘要截断）

    arXiv:2610.32255v1 Announce Type: cross  Abstract: Out-of-distribution (OOD) detection is recognized as a critical problem in AI safety. SCALE is an effective post-hoc OOD detection method in the activation-shaping family that requires no training data and rescales penultimate activations using a percentile-based magnitude ratio. However, this ratio is ill-defined when negative activations are present, since positive and negative contributions cancel within the underlying sum, causing SCALE to degenerate on backbones such as ViT. We address this with an Absolute Fix that computes the ratio over absolute-valued activations rather than discarding negative activations through ReLU. Building on this corrected scale, we propose Anti-coherence regularization, which penalizes the scaling factor by the fraction of a sample's max-logit contributions to the max logit that are sign-mismatched and thereby suppresses unreliable evidence for the predicted class. Despite requiring no access to traini
    
[^97]: 循环语言模型中的停止头真的能节省计算吗？

    Do Halting Heads Really Save Compute in Looped Language Models?

    [https://arxiv.org/abs/2610.32246](https://arxiv.org/abs/2610.32246)

    本文揭示循环语言模型中的停止头因训练方式缺陷和批次生成机制而无法真正节省计算量，无停止头的模型在同等计算成本下表现几乎一样好。

    

    循环语言模型让每个词元多次通过相同的层，每次通过称为一次迭代。为了节省计算量，模型引入一个额外的层——停止头，它在每次迭代后给出词元在此停止的概率，使简单的词元可以提前停止。我们从零开始预训练了参数量最高达13亿的模型，发现了两个问题。第一，在所有迭代的混合数据上训练的停止头预测词元所需的迭代次数不足一半，但在标准停止规则下，词元实际上几乎每次迭代都会运行，因为这种训练方式从未对词元真正停止的那一次迭代进行评分。改为对每次迭代都进行评分可以解决这一问题，但在我们的实验中，只有在采用强烈偏向提前停止的惩罚时才有效。第二，计算节省量通常以每个词元的平均迭代次数来报告，但当序列以批次方式生成、且需运行到批次中最慢的词元停止为止时，没有停止头的模型在同等计算成本下表现几乎一样好。

    arXiv:2610.32246v1 Announce Type: new  Abstract: Looped language models run each token through the same layers several times, and each run is one iteration. To save compute, an extra layer, the halting head, gives after each iteration the probability that the token stops there, so that easy tokens can stop early. Pretraining models from scratch, up to 1.3B parameters, we find two problems. First, heads trained on a mix of all iterations predict that tokens need under half the iterations, yet under the standard stopping rule tokens run almost every iteration, because this training never scores the single iteration where a token stops. Scoring every iteration instead fixes this, but in our runs it works only with a strong penalty favoring early stops. Second, savings are commonly reported as an average number of iterations per token, but when sequences are generated in batches that run until their slowest token stops, a model without a head is almost as good at that cost. The reported av
    
[^98]: BiLoCo：用于训练后量化的二值低秩校正

    BiLoCo: Binary Low-Rank Corrections for Post-Training Quantization

    [https://arxiv.org/abs/2610.32243](https://arxiv.org/abs/2610.32243)

    提出BiLoCo方法，将低秩校正的左右因子以二值±1符号向量存储，每个秩一分量节省16倍存储，并可在CUDA核心上与FP4张量核心GEMM并行运行，有效缓解FP4训练后量化的精度损失。

    

    将大语言模型训练后量化（PTQ）至FP4已成为降低推理成本的关键技术，尤其是在提供原生FP4张量核心支持的Blackwell GPU上。然而，直接进行FP4训练后量化仍会留下明显的精度差距。近期SVDQuant的工作通过添加经由SVD获得的高精度低秩校正，缓解了FP4量化对扩散模型造成的误差。然而，这种高精度低秩校正每个秩需要消耗大量比特，且与主FP4 GEMM运行在相同的张量核心上。我们提出BiLoCo，一种低秩校正方法，其左因子和右因子以二值{±1}符号向量而非BF16数字的形式存储。这种符号格式使每个秩一分量的存储减少16倍，并且由于校正现在仅由加法/减法组成，因此可以在CUDA核心上与FP4张量核心GEMM并行运行。在4B–72B规模的LLM上，

    arXiv:2610.32243v1 Announce Type: new  Abstract: Post-training quantization (PTQ) to FP4 has emerged as a key technique for reducing large language model inference cost, especially on Blackwell GPUs, which offer native FP4 tensor core support. Direct PTQ to FP4, however, still leaves a noticeable accuracy gap. Recent work on SVDQuant mitigates the errors introduced through FP4 quantization for diffusion models by adding high-precision low-rank corrections obtained via SVD. However, this high-precision low-rank correction requires many bits per rank, and it runs on the same tensor cores as the main FP4 GEMM. We introduce BiLoCo, a low-rank correction method whose left and right factors are stored as binary ${\{\pm 1\}}$ sign vectors rather than as BF16 numbers. The signed format uses $16{\times}$ less storage per rank-one component, and since the correction now consists of additions/subtractions, this can be run on CUDA cores concurrently with the FP4 tensor-core GEMM. On 4B--72B LLMs, 
    
[^99]: 正确却无法被选中：多候选推理中的隐性接口税

    Correct but Unselectable: The Hidden Interface Tax in Multi-Candidate Reasoning

    [https://arxiv.org/abs/2610.32235](https://arxiv.org/abs/2610.32235)

    该论文发现了多候选推理中被忽视的“隐性接口税”瓶颈——正确答案虽在候选池中却因不完整、不可解析或格式不对齐而无法被选择器选中，并提出兼容性感知候选构建方法（CACC）作为候选源与固定选择器之间的接口层来解决这一问题。

    

    在成本、延迟或隐私约束下部署的小型语言模型，在推理失败时无法简单地切换到更大的模型。多候选“生成后重排”提供了一条自然的测试时扩展路径。然而，更多的候选并不总能带来更高的准确率。我们识别出一个此前被忽视的、介于生成与选择之间的瓶颈：一个正确的答案可能已经出现在候选池中，却因为不完整、无法解析、或与固定选择器所期望的答案格式不对齐而无法被选中。我们将这种损失称为“隐性接口税”，并形式化了“覆盖-转化”分解方法，将原始的神谕覆盖与可选覆盖区分开来——后者是指正确候选实际能够被选择器所消费的问题子集。我们提出兼容性感知的候选构建方法，作为候选源与固定选择器之间的接口层，用于修复片段、对齐答案……

    arXiv:2610.32235v1 Announce Type: new  Abstract: Small language models deployed under cost, latency, or privacy constraints cannot simply switch to larger models when reasoning fails. Multi-candidate generate-then-rerank offers a natural test-time scaling route. However, more candidates do not always yield higher accuracy. We identify a previously overlooked bottleneck between generation and selection: a correct answer may appear in the candidate pool yet fail to be selected because it is incomplete, unparseable, or misaligned with the answer mode the fixed selector expects. We term this loss the hidden interface tax and formalize a coverage-conversion decomposition that distinguishes raw oracle coverage from selectable coverage, the subset of problems where a correct candidate can actually be consumed by the selector. We propose Compatibility-Aware Candidate Construction (CACC), an interface layer between the candidate source and the fixed selector that repairs fragments, aligns answe
    
[^100]: MS-TTRL：打破多解测试时强化学习中的频率冲突

    MS-TTRL: Breaking the Frequency Conflict in Multiple-Solution Test-Time Reinforcement Learning

    [https://arxiv.org/abs/2610.32234](https://arxiv.org/abs/2610.32234)

    提出MS-TTRL方法，通过结果级自验证获得与频率无关的正确性估计，并借助三态结果缓存实现验证复用，从而解决了多解测试时强化学习中生成频率在正确性估计与多样性保持之间的“频率冲突”。

    

    科学、工程和数学推理任务往往存在多个有效解，而对完整解集进行穷尽式标注的代价通常高昂得令人望而却步。测试时强化学习将强化学习应用于无标签的测试实例，为多解推理提供了一种无需标签的方法。在这一设定中，生成频率服务于两个相互对立的目标：正确性估计偏好高频出现的结果，而结果层面的多样性保持则偏好低频出现的结果。我们将这种张力称为“频率冲突”。为解决这一冲突，我们提出MS-TTRL，它通过结果级自验证获得与频率无关的正确性估计，同时仅将生成频率保留用于多样性保持。三态结果缓存和候选状态支持在重复观测中复用验证结果，并以置信度感知的方式处理模糊情况……

    arXiv:2610.32234v1 Announce Type: new  Abstract: Scientific, engineering, and mathematical reasoning tasks can admit multiple valid solutions, while exhaustive annotation of the full solution set is often prohibitively costly. Test-time reinforcement learning (TTRL) applies reinforcement learning to unlabeled test instances, offering a label-free approach to multiple-solution reasoning. In this setting, generation frequency serves two opposing objectives: correctness estimation favors frequent outcomes, whereas outcome-level diversity preservation favors less frequent outcomes. We term this tension the \textbf{Frequency Conflict}. To resolve this conflict, we propose MS-TTRL, which obtains frequency-independent correctness estimates through Outcome-Level Self-Verification while reserving generation frequency solely for diversity preservation. Three-State Outcome Cache and the Candidate State support verification reuse across repeated observations and confidence-aware handling of ambigu
    
[^101]: TileShare：免训练加速重叠平铺式ViT解码

    TileShare: Training-Free Acceleration of Overlapping Tiled ViT Decoding

    [https://arxiv.org/abs/2610.32233](https://arxiv.org/abs/2610.32233)

    TileShare通过将重叠token的各tile计算结果混合为共享状态、同时对读取共享状态的前几层注意力保留每tile独立计算，无需训练即可将重叠平铺式ViT解码加速1.38-1.51倍（FP8下最高1.96倍）并保持质量等效。

    

    固定窗口视觉Transformer（ViT）解码器在部署分辨率下以重叠平铺的方式运行：重叠区域中的每个token都会被多个完整的解码器实例处理。在24亿参数的MiniMax-H3视频解码器中，约一半的每token计算是重复的。直接移除平铺是不可行的，因为注意力、全局token和坐标都与tile绑定。TileShare保留每个被释放tile的注意力，并将其他所有操作对每个token只计算一次。被释放的计算仅在一种步骤上发生变化：重叠token的各tile结果通过被释放的权重混合为一个共享状态。干预实验表明，这种混合在注意力读取共享状态的位置会损害质量，主要集中在前几层，因此TileShare对这些层保留每个tile的独立计算。在无需训练的情况下，在每个tile保留六层，在两种GPU上的解码速度比批处理加编译的平铺方式快1.38-1.51倍（使用FP8最高可达1.96倍），且质量保持等效。

    arXiv:2610.32233v1 Announce Type: cross  Abstract: Fixed-window Vision Transformer (ViT) decoders run at deployment resolution as overlapping tiles: every token in an overlap band is processed by several full decoder instances. In the 2.4B-parameter MiniMax-H3 video decoder, about half of all per-token computation is duplicated. Removing the tiles fails because attention, global tokens and coordinates are bound to them. TileShare keeps each released tile's attention and computes every other operation once per token. The released computation changes at one kind of step only: an overlap token's per-tile results are blended into one shared state with the released weights. Interventions show that this blend costs quality where attention reads the shared states, mainly in the first few layers, which TileShare therefore keeps per tile. Without training, keeping six layers per tile decodes 1.38-1.51x faster than batched and compiled tiling on two GPU types (up to 1.96x with FP8) and is equiva
    
[^102]: RoboCode：为机器人操作学习可复用的动作基元

    RoboCode: Learning Reusable Action Primitives for Robot Manipulation

    [https://arxiv.org/abs/2610.32225](https://arxiv.org/abs/2610.32225)

    RoboCode通过共享的软门控基元动作接口，将机器人操作中的动作块表示为多个可复用基元的软组合，从而学习跨任务共享的短时程运动模式。

    

    近期的机器人操作策略通常直接从观测预测连续的动作块，而没有显式地对动作空间中的可复用结构进行建模。然而，操作演示中往往包含跨任务共享的、重复出现的短时程运动模式。我们提出了RoboCode，它通过一个共享的软门控基元动作接口来学习这些模式。RoboCode并不将每个动作块分配给单一的离散编码，而是将其表示为多个已学习基元的软组合，从而为短时程动作提供一种灵活的中间表示。该基元接口从跨多个任务汇总的动作块中学习得到，随后被冻结，同时训练任务特定的状态编码器将当前状态输入映射到共享潜空间中。在五个ManiSkill操作任务上的实验回答了三个问题。首先，学习到的基元……

    arXiv:2610.32225v1 Announce Type: cross  Abstract: Recent robot manipulation policies commonly predict continuous action chunks directly from observations, without explicitly modeling reusable structure in the action space. However, manipulation demonstrations often contain recurring short-horizon motion patterns that are shared across tasks. We introduce \textsc{RoboCode}, which learns these patterns through a shared soft-gated primitive action interface. Rather than assigning each action chunk to a single discrete code, \textsc{RoboCode} represents it as a soft composition of multiple learned primitives, providing a flexible intermediate representation of short-horizon actions. The primitive interface is learned from action chunks pooled across multiple tasks and then frozen while task-specific state encoders are trained to map current state inputs into the shared latent space.   Experiments on five ManiSkill manipulation tasks address three questions. First, the learned primitive in
    
[^103]: 利用边缘算力实现大语言模型训练的民主化：挑战、系统与未来之路

    Democratizing LLM Training with the Edge: Challenges, Systems, and the Road Ahead

    [https://arxiv.org/abs/2610.32215](https://arxiv.org/abs/2610.32215)

    本文提出以云端参数服务器为核心的统一分布式训练系统架构，系统评估了联邦学习、拆分学习和去中心化方法在异构硬件与不可靠网络环境下的可行性，旨在让缺乏大型数据中心基础设施的中小机构也能利用自身私有数据参与大语言模型的训练。

    

    现代GPU数据中心日益增长的计算能力使得少数工业巨头能够在海量数据上训练和部署大语言模型（LLM）。然而，这些模型及其训练数据的庞大规模使得集中式开发方式越来越难以持续，同时也将那些拥有相关私有数据但缺乏同等基础设施的小型机构拒之门外。在本文中，我们围绕预训练和微调两种范式，系统阐述了关于分布式LLM训练的立场观点，涵盖联邦学习、拆分学习和去中心化方法，并评估了它们在具有异构硬件和不可靠网络连接的地理分布式环境中的适用程度。随后，我们提出了一个以云端托管参数服务器为核心的统一系统架构，该服务器通过出站gRPC连接协调异构客户端，并复用统一的通信、压缩和容错技术栈。

    arXiv:2610.32215v2 Announce Type: cross  Abstract: The growing computational power of modern GPU data centers has let a few industrial giants train and deploy large language models (LLMs) on massive amounts of data. Yet the sheer scale of these models and their training data makes centralized development increasingly hard to sustain, and at the same time shuts out smaller institutions that hold relevant private data but lack comparable infrastructure. In this paper, we organize our positions on distributed LLM training around the pretraining and fine-tuning paradigms, covering federated learning, split learning, and decentralized approaches, and we assess their readiness for geo-distributed settings with heterogeneous hardware and unreliable connectivity. We then present a {\em unified} system architecture built around a cloud-hosted parameter server that coordinates heterogeneous clients over outbound gRPC connections, reusing one communication, compression, and fault-tolerance stack 
    
[^104]: 头维度对语言模型长上下文检索至关重要

    Head Dimension Matters in Language Models for Long-Context Retrieval

    [https://arxiv.org/abs/2610.32213](https://arxiv.org/abs/2610.32213)

    本研究发现在固定模型规模下，增大注意力头维度可通过提高检索头的比例显著提升语言模型的长上下文检索性能，且即使冻结注意力投影权重该结论依然成立，而头维度对预训练损失和短上下文任务性能影响甚微。

    

    多头注意力通过查询-键交互和值聚合来检索与组合上下文表示，其中头维度限制了每个注意力头的表示空间。在长上下文处理中，将固定的参数预算分配给更少但更宽的头是否能改善信息检索，目前仍不清楚。在本文中，我们研究了头维度如何影响长上下文检索任务的性能。我们的实证研究表明，头维度对预训练损失和短上下文非检索任务的性能影响不大，而在固定模型规模下，长上下文检索性能会随着头的变宽而提升。我们发现，拥有更宽头的模型具有更高比例的检索头，即有更多参数与检索功能相关联。我们的消融实验还表明，即使冻结注意力模块中的投影权重，检索性能仍会随头变宽而提升。

    arXiv:2610.32213v1 Announce Type: cross  Abstract: Multi-head attention retrieves and combines context representations through query--key interactions and value aggregation, with head dimension constraining the per-head representation space. In long-context processing, it remains unclear whether allocating a fixed parameter budget to fewer, wider heads improves information retrieval. In this paper, we investigate how head dimension affects the performance of long-context retrieval tasks. Our empirical studies reveal that pretraining loss and short-context non-retrieval task performance are not affected much by head dimension, while long-context retrieval performance improves with wider heads at fixed model sizes. We find that models with wider heads have a higher proportion of retrieval heads, associating more parameters with retrieval. Our ablation studies also show that the retrieval performance improves with wider heads even when the projection weights in attention modules are froze
    
[^105]: VLA模型知道自己会失败吗？物理AI系统中可解释的成功与失败模式

    Do VLAs Know They Will Fail? Interpretable Success and Failure Patterns in Physical AI Systems

    [https://arxiv.org/abs/2610.32208](https://arxiv.org/abs/2610.32208)

    本论文发现OpenVLA激活的稀疏自编码器（SAE）表示蕴含结果预测信息，可在机器人任务陷入不可恢复状态前预测失败，并通过识别与成败阶段相关的SAE特征及基于VLM的解释流程，揭示反复出现的行为模式。

    

    视觉-语言-动作（VLA）模型已成为强大的通用机器人策略，但其成功与失败仍然难以诊断和理解。在这项工作中，我们展示了OpenVLA激活的稀疏自编码器（SAE）表示包含可预测结果的信息，可用于在机器人进入不可恢复状态之前预测任务失败。此外，为了深入洞察哪些常见行为可能导致这些结果，我们识别了强激活与成功或失败特定阶段相关联的个体SAE特征，并通过定位这些特征表现出显著激活变化的时间区间，揭示了反复出现的行为模式。我们进一步开发了一个基于VLM的解释流程，为这些模式生成行为描述，并在保留的时间区间上验证了其与特征激活的对应关系。

    arXiv:2610.32208v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) models have emerged as powerful general-purpose robot policies, but their successes and failures remain difficult to diagnose and understand. In this work, we show that sparse autoencoder (SAE) representations of OpenVLA activations contain outcome-predictive information and can be used to predict task failure before the robot reaches an unrecoverable state. Moreover, to provide insights into what common behaviors may contribute to these outcomes, we identify individual SAE features whose strong activations are associated with specific stages of success or failure, and uncover recurring behavioral patterns by localizing temporal intervals where these features exhibit significant activation changes. We further develop a VLM-based interpretation pipeline to generate behavioral descriptions of these patterns and validate their correspondence to feature activations on held-out intervals.
    
[^106]: 用于学习精确秩滤波器的神经析取范式

    Neural Disjunctive Normal Forms for Learning Exact Rank Filters

    [https://arxiv.org/abs/2610.32203](https://arxiv.org/abs/2610.32203)

    本文提出基于阈值分解的神经析取范式来学习精确秩滤波器，通过前向传播精确求值、后向传播使用平滑替代导数的直通式方法，解决了形态学算子梯度学习中导数稀疏的问题。

    

    基于梯度的形态学算子学习因结构元素的局部稀疏导数而受到阻碍。我们开发了一种学习秩滤波器的方法，该方法在保留精确算子求值的同时，将导数分布在整个窗口上。利用阈值分解，我们构建了一种神经析取范式表示，并建立了针对整数值图像和结构元素的精确性充分条件。随后，我们从平滑阈值方程的唯一根导出一个平滑替代函数。隐式微分给出了具有归一化局部权重的闭式导数，其分布由平滑参数控制。这些导数被用于直通式（straight-through）过程：前向传播计算精确的秩滤波器，而后向传播使用平滑替代函数。对于固定秩，该过程可以学习任意秩滤波器的实值结构元素……

    arXiv:2610.32203v1 Announce Type: cross  Abstract: Gradient-based learning of morphological operators is hindered by locally sparse derivatives with respect to the structuring elements. We develop a method for learning rank filters that retains exact operator evaluation while distributing the derivatives across the window. Using threshold decomposition, we construct a neural disjunctive normal form representation and establish a sufficient condition for exactness with integer-valued images and structuring elements. We then derive a smooth surrogate from the unique root of a smooth threshold equation. Implicit differentiation gives closed-form derivatives with normalised local weights, whose distribution is controlled by a smoothing parameter. These derivatives are used in a straight-through procedure: the forward pass evaluates the exact rank filter, while the backward pass uses the surrogate. For fixed ranks, the procedure learns real-valued structuring elements of any rank filter, in
    
[^107]: SWE-Shift：在动态演变的工作区中评估编程智能体

    SWE-Shift: Evaluating Coding Agents in Evolving Workspaces

    [https://arxiv.org/abs/2610.32196](https://arxiv.org/abs/2610.32196)

    提出 SWE-Shift 基准，用于评估编程智能体在任务执行中途遭遇代码、依赖和需求等变化时的表现，实验发现所有 17 个被测智能体系统在发生此类变化时成功率均显著下降。

    

    仓库级编程基准测试在很大程度上假设任务和环境在整个执行过程中保持不变。而在软件开发中，代码、依赖项、运行时状态和需求可能在智能体仍在工作时发生变化。我们提出了 SWE-Shift，一个用于评估编程智能体在执行中途遭遇变化时表现的基准。基于相同的保存历史和工作区，Clean（干净）延续在不干预的情况下继续进行，而与之匹配的 Shifted（转变）延续则会收到与任务相关的更新。成功需要完成当前任务、保留仍然有效的工作，并满足适用的状态和证据检查。该基准包含横跨 12 个变化族别的 120 个场景，每个场景在九个预定的干预位置进行评估。我们评估了 17 个编程智能体系统，所有系统在 Shifted 条件下的成功率均低于 Clean 条件。GPT-6 Astra 的成功率从 67.35% 降至 24.79%，配对成功率为 20.34%，工具/会话变化和过时证据是主要失败因素之一。

    arXiv:2610.32196v1 Announce Type: new  Abstract: Repository-level coding benchmarks largely assume that tasks and environments remain unchanged throughout execution. In software development, code, dependencies, runtime state, and requirements can change while an agent is still working. We introduce SWE-Shift, a benchmark for evaluating coding agents under such mid-execution changes. From the same saved history and workspace, a Clean continuation proceeds without intervention, while a matched Shifted continuation receives a task-relevant update. Success requires completing the current task, preserving still-valid work, and satisfying the applicable state and evidence checks. The benchmark contains 120 scenarios across 12 change families, each evaluated at nine scheduled intervention positions. We evaluate 17 coding-agent systems, all of which have lower Shifted than Clean success. GPT-6 Astra drops from 67.35% to 24.79%, with 20.34% paired success. Tool/session changes and stale evidenc
    
[^108]: Principia：面向科学发现的测试时假设到原理构建

    Principia: Test-Time Hypotheses to Principle Construction for Scientific Discovery

    [https://arxiv.org/abs/2610.32195](https://arxiv.org/abs/2610.32195)

    提出了基于大语言模型的Principia框架，通过从相关输入与观测结果中形成比较性因果假设、并将假设中反复出现的关系综合为归纳性原理，在测试时自动构建并持续完善科学原理，突破了科学发现中知识构建这一瓶颈。

    

    科学发现需要识别观测现象背后的科学原理，以指导进一步的探究。由于这些经验性原理在复杂的发现过程中往往是未知的，它们通过“假设到原理”的流程来构建，即在测试时的发现过程中，从逐步积累的观测中提炼出反复出现的因果关系。然而，构建这类知识需要大量的领域专业知识、时间和精力，使得这一转化阶段成为科学发现的瓶颈。为了解决这一局限，我们提出了Principia，一个基于大语言模型（LLM）的框架，能够在测试时构建并持续完善科学原理。借鉴人类研究人员发展原理的方式，Principia从相关输入及其观测结果中形成比较性因果假设，然后将这些假设中反复出现的关系综合为归纳性原理。通过应用所构建的原理……（原文摘要在此处被截断）

    arXiv:2610.32195v1 Announce Type: cross  Abstract: Scientific discovery requires identifying the scientific principles underlying observed phenomena to guide further investigation. As these empirical principles are often unknown in complex discovery processes, they are constructed via a hypotheses-to-principle pipeline to obtain recurring causalities from progressively accumulated observations during test-time discovery. However, constructing this knowledge requires substantial domain expertise, time, and effort, making this translation stage a bottleneck for scientific discovery. To address this limitation, we propose Principia, an LLM-based framework that constructs and continually refines scientific principles at test time. Drawing on how human researchers develop principles, Principia forms comparative causal hypotheses from related inputs and their observed outcomes, then synthesizes recurring relationships across these hypotheses into inductive principles. By applying the constru
    
[^109]: 双自适应Z分数PPO：基于动作置信度与归一化优势幅度的动态裁剪

    Dual-Adaptive Z-Score PPO: Dynamic Clipping Using Action Confidence and Normalized Advantage Magnitude

    [https://arxiv.org/abs/2610.32194](https://arxiv.org/abs/2610.32194)

    该论文提出PPO-DAZ3方法，通过动作置信度和归一化优势幅度这两个信号为每个样本动态设置PPO的裁剪阈值，同时保持标准PPO训练流程其余部分不变。

    

    近端策略优化（PPO）因其稳定性和易于实现而仍然是一种流行的在线策略强化学习算法。然而，其裁剪阈值是固定的，因此每个训练样本都获得相同的信任域宽度，而不论相应策略更新的可靠程度如何。本工作研究了PPO的自适应裁剪变体，将恒定的裁剪参数替换为基于策略和回报统计量为每个样本计算的阈值。其核心方法——双自适应Z分数PPO（PPO-DAZ3）——利用两个信号为每个样本设置裁剪区间：动作置信度（由当前策略下所选动作的概率给出）和更新置信度（由归一化优势幅度给出），而标准PPO训练流程的其余部分保持不变。在三个经典控制基准（LunarLander-v3、CartPole-v1和Acrobot-v1）上的评估表明……（原文摘要在此处截断）

    arXiv:2610.32194v1 Announce Type: cross  Abstract: Proximal Policy Optimization (PPO) remains a popular on-policy reinforcement learning algorithm owing to its stability and ease of implementation. Its clipping threshold, however, is fixed, so every training sample receives the same trust-region width regardless of how reliable the corresponding policy update is. This work examines adaptive-clipping variants of PPO in which the constant clipping parameter is replaced by sample-dependent thresholds computed from policy and return statistics. The central method, Dual-Adaptive Z-Score PPO (PPO-DAZ3), sets the clipping interval for each sample from two signals: action confidence, given by the probability of the selected action under the current policy, and update confidence, given by the normalized advantage magnitude. The rest of the standard PPO training pipeline is left unchanged. Evaluation on three classic control benchmarks (LunarLander-v3, CartPole-v1, and Acrobot-v1) indicates that
    
[^110]: EVOLVE：基于自适应原型记忆的不确定性感知多变量时间序列异常预测

    EVOLVE: Uncertainty-Aware Multivariate Time Series Anomaly Prediction with Adaptive Prototype Memories

    [https://arxiv.org/abs/2610.32189](https://arxiv.org/abs/2610.32189)

    提出EVOLVE框架，通过自适应原型记忆实现不确定性感知的多变量时间序列异常预测，能够建模变量与上下文相关的扰动差异，并预见多种可能的未来异常演化轨迹。

    

    无监督多变量时间序列异常预测（AP）旨在从当前观测中估计未来的异常风险，为复杂系统的主动干预与维护提供早期预警信号。现有无监督AP方法遵循基于扰动的范式，即将预定义或随机扰动注入历史窗口以模拟异常的未来演化。然而，这些方法在不同变量和上下文之间共享扰动，并学习确定性的“从历史到未来”的演化。实际上，异常演化是依赖于变量和上下文的：同一扰动在某个变量或上下文中可能被放大为异常的未来动态，而在另一些情况下则被吸收为正常波动。此外，未来演化并非由历史窗口唯一确定，因为相似的历史可能导向多种合理的发展轨迹，例如逐渐退化或突发故障。

    arXiv:2610.32189v1 Announce Type: new  Abstract: Unsupervised multivariate time series anomaly prediction (AP) aims to estimate future anomaly risks from current observations, providing early-warning signals for proactive intervention and maintenance in complex systems. Existing unsupervised AP methods follow a perturbation-based paradigm, where predefined or random perturbations are injected into historical windows to simulate abnormal future evolution. However, these methods share perturbations across different variables and contexts, and learn deterministic history-to-future evolution. In reality, anomaly evolution is variable- and context-dependent: the same perturbation may be amplified into abnormal future dynamics in one variable or context, while being absorbed as normal fluctuation in another. In addition, future evolution is not uniquely determined by a historical window, since similar histories may lead to multiple plausible trajectories, such as gradual degradation or abrup
    
[^111]: CFBrainAug：模块化诱导的反事实脑网络增强用于可解释的基于fMRI的自闭症诊断

    CFBrainAug: Modularity-Induced Counterfactual Brain Network Augmentation for Interpretable fMRI-based Autism Diagnosis

    [https://arxiv.org/abs/2610.32184](https://arxiv.org/abs/2610.32184)

    提出了反事实脑网络学习框架CFBrainAug，通过生成标签相反但拓扑相似、并受模块数量约束的脑网络来统一数据增强与反事实解释，在提升基于fMRI自闭症诊断性能的同时增强了临床可解释性。

    

    在静息态功能磁共振成像（rs-fMRI）脑网络上进行机器学习对于诊断自闭症谱系障碍（ASD）具有广阔前景，但现有方法容易在有限的训练数据上过拟合，且其事后统计归因提供的临床可解释性有限。我们提出了CFBrainAug，这是一个反事实脑网络学习框架，通过学习标签改变的脑网络，将数据增强与反事实解释统一起来。与传统的标签保持增强不同，CFBrainAug首先检索一个标签相反但拓扑相似的候选网络，然后在模块数量约束下对其进行细化。在事实网络与反事实网络上进行联合训练，丰富了决策边界附近的训练分布，使得基于轻量级特征选择的SVM能够与近期的图深度学习方法相媲美。同时，反事实网络提供了（原文摘要在此处截断）

    arXiv:2610.32184v1 Announce Type: new  Abstract: Machine learning on resting-state functional magnetic resonance imaging (rs-fMRI) brain networks is promising for diagnosing Autism Spectrum Disorder (ASD), but existing methods are prone to overfitting on limited training data, and their post-hoc statistical attribution offers limited clinical interpretability. We propose CFBrainAug, a counterfactual brain network learning framework that unifies data augmentation and counterfactual explanation by learning label-altered brain networks. Unlike conventional label-preserving augmentation, CFBrainAug first retrieves a label-opposite yet topologically similar candidate and then refines it under a module-number constraint. Jointly training on factual and counterfactual networks enriches the training distribution around the decision boundary, enabling a lightweight feature-selection-based SVM to perform competitively with recent graph deep learning methods. Meanwhile, the counterfactuals provid
    
[^112]: 视觉语言模型中深度依赖的安全几何结构：逐层定位与双向拒答控制

    Depth-Dependent Safety Geometry in Vision-Language Models: Layer-Wise Localization and Bidirectional Refusal Control

    [https://arxiv.org/abs/2610.32182](https://arxiv.org/abs/2610.32182)

    该论文提出一个无需重训的推理时框架，通过逐层激活分析揭示视觉语言模型的安全几何随深度非均匀分布——早期视觉层可通过激活引导实现 6%–96% 的双向拒答控制，而深层虽表征可分却对操控不敏感，从而区分了表征可分性与因果可控性。

    

    视觉语言模型（VLM）已被广泛部署，但其安全行为背后的内部机制仍知之甚少。我们提出了一个轻量级的推理时框架，用于在冻结的视觉语言模型中定位并操控安全敏感表征，无需重新训练或修改参数。在 Qwen3-VL-4B 上，逐层激活差异分析揭示了安全几何结构在模型深度上呈非均匀分布。因果干预实验表明，早期视觉层通过沿学习到的安全方向进行激活引导，可实现强大的双向拒答控制，不安全拒绝率可在 6%–96% 之间调节；而后面的层对激活幅值缩放更为敏感，并表现出显著的安全性与选择性之间的权衡。更深的层沿安全方向保持良好的表征分离，但对激活引导基本不敏感，这区分了表征可分性与因果可控性。InternVL2.5-4B 在定性上（呈现出类似结果，摘要原文在此处截断）。

    arXiv:2610.32182v1 Announce Type: cross  Abstract: Vision-Language Models (VLMs) are widely deployed, yet the internal mechanisms behind their safety behavior remain poorly understood. We present a lightweight inference-time framework for localizing and manipulating safety-sensitive representations in frozen VLMs, without retraining or modifying parameters. On Qwen3-VL-4B, layer-wise activation divergence analysis reveals non-uniform safety geometry across depth. Causal interventions show that early vision layers support strong bidirectional refusal control through activation steering along a learned safety direction, with unsafe rejection spanning 6%-96%, whereas later layers respond more to activation magnitude scaling and exhibit a substantial safety-selectivity trade-off. Deeper layers remain well separated along the safety direction yet are largely insensitive to steering, distinguishing representational separability from causal controllability. InternVL2.5-4B shows qualitatively 
    
[^113]: 并非所有信息都值得询问：基于决策感知的大语言模型信息获取

    Not All Information Is Worth Asking For: Decision-Informed Information Seeking with LLMs

    [https://arxiv.org/abs/2610.32181](https://arxiv.org/abs/2610.32181)

    提出DIVA方法，让大语言模型基于证据的决策价值扣除成本后来评估每项检查是否值得进行，从而决定该查询什么信息以及何时停止，避免为无法改变决策或不值得的信息付费。

    

    大语言模型（LLM）越来越多地被用于在不完整信息下支持决策，在这种情况下，决定查询什么以及何时停止与最终答案同样重要。然而，LLM会陷入信息陷阱：它们会为那些无法改变决策或不值得其成本的检查（即昂贵的证据片段）付费，并且在证据已经充分的情况下仍然继续询问。为此，我们提出了决策感知的信息价值获取方法（Decision-Informed Value-of-information Acquisition，DIVA），这是一种基于证据决策价值的简单而有效的LLM信息获取程序。DIVA 1）让LLM通过简短的数值问题指定一个小型决策模型，即坏结果发生的可能性有多大，以及每项检查将如何改变这一概率；2）通过设想各候选检查的潜在结果能在扣除成本后对决策带来多大改善来评估每项候选检查的价值，当没有检查值得其成本时即停止。其变体DIVA-U进一步考虑了……

    arXiv:2610.32181v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to support decisions under incomplete information, where deciding what to look up and when to stop matters as much as the final answer. However, LLMs fall into an information trap: They pay for checks, i.e., costly pieces of evidence, that cannot change their decision or are not worth their cost, and they keep asking when the evidence is already sufficient. To this end, we propose Decision-Informed Value-of-information Acquisition (DIVA), a simple yet effective procedure for LLM information seeking based on the decision value of evidence. DIVA 1) lets the LLM specify a small decision model, i.e., how likely a bad outcome is and how each check would shift this probability, through short numeric questions and 2) values each candidate check by how much its imagined outcomes improve the decision net of its cost, stopping when no check is worth its cost. Its variant, DIVA-U, further accounts 
    
[^114]: 稀疏历史搜索：基于不完整搜索日志与零成本代理的跨任务架构搜索

    Searching with Sparse Histories: Cross-Task Architecture Search from Incomplete Search Logs and Zero-Cost Proxies

    [https://arxiv.org/abs/2610.32179](https://arxiv.org/abs/2610.32179)

    提出一种风险鲁棒的跨任务神经架构搜索方法，通过冠军初始化与联合成对排序器预训练，将稀疏搜索日志与零成本代理相结合，显著缓解最坏情况下的失败并大幅提升最差性能。

    

    神经架构搜索（NAS）成本高昂，但跨任务迁移可以通过重用先前任务的知识来降低成本。然而，现有方法在稀疏搜索历史和任务特定迁移失败的问题上表现不佳。我们提出了一种风险鲁棒的策略，通过基于冠军的初始化和联合成对排序器预训练，将稀疏的源任务日志与零成本代理相结合。在TransNAS-Bench-101上，该方法缓解了严重的最坏情况失败，将Micro自编码任务的最差10%排名从290降低到43，将Macro房间布局任务的最差10%排名从465降低到45。在13个单元空间目标中，该方法在平均性能和最坏情况性能上均在10个目标上优于仅使用代理的排序器，同时保持了有竞争力的平均性能。

    arXiv:2610.32179v1 Announce Type: cross  Abstract: Neural architecture search (NAS) is expensive, but cross-task transfer can reduce costs by reusing knowledge from previous tasks. However, existing methods struggle with sparse search histories and task-specific transfer failures. We propose a method, a risk-robust strategy that combines sparse source logs with zero-cost proxies through champion-based initialization and joint pairwise-ranker pretraining. On TransNAS-Bench-101, the proposal mitigates severe worst-case failures, reducing worst-10% ranks from 290 to 43 for Micro autoencoding and from 465 to 45 for Macro room layout. Across 13 cell-space targets, it outperforms a proxy-only ranker on 10 targets in both mean and worst-case performance, while maintaining competitive average performance.
    
[^115]: 具有异构客户端的联邦非参数回归

    Federated Nonparametric Regression with Heterogeneous Clients

    [https://arxiv.org/abs/2610.32178](https://arxiv.org/abs/2610.32178)

    该论文提出一种联邦Nadaraya-Watson非参数回归方法，客户端仅传输本地核摘要即可构建协变量密度加权的联邦估计器，并通过精确恒等式与两阶段分析刻画异构客户端从联邦预测中获益的条件及统计推断。

    

    联邦预测器是否能使单个客户端受益，取决于总体异质性和估计不确定性两个方面。我们在三角阵列设置下针对Nadaraya-Watson回归研究这一问题，其中拟合分布与评估分布可能不同。每个客户端仅传输本地核摘要，使服务器能够在不集中个体观测数据的情况下构建一个以协变量密度加权的联邦估计器。客户端通过预设权重将其拟合的本地预测器与联邦预测器相结合，增益定义为在独立审计观测上配对的平方预测损失的减少量。我们给出了精确恒等式，用以刻画已实现拟合配对的增益及其重复拟合均值。我们还为联邦估计器建立了逐点推断。对客户端增益的两阶段分析将拟合变异性与审计变异性分离，并保留了当客户端（原文在此处截断）

    arXiv:2610.32178v1 Announce Type: new  Abstract: Whether a federated predictor benefits an individual client depends on both population heterogeneity and estimation uncertainty. We study this question for Nadaraya--Watson regression in a triangular array with potentially different fitting and evaluation distributions. Each client transmits local kernel summaries, allowing the server to construct a covariate-density-weighted federated estimator without centralizing individual observations. A client combines its fitted local predictor with the federated predictor through a prescribed weight, and gain is the paired reduction in squared prediction loss on an independent audit observation. Exact identities characterize both the gain of the realized fitted pair and its repeated-fitting mean. We also establish pointwise inference for the federated estimator. A two-stage analysis of client gains separates fitting variability from audit variability, retains the covariance created when a client'
    
[^116]: Ψ-0.6：统一视觉-语言世界模型中的自适应思维流

    $\Psi$-0.6: Adaptive Streams of Thought in a Unified Vision-Language World Model

    [https://arxiv.org/abs/2610.32177](https://arxiv.org/abs/2610.32177)

    Ψ-0.6 是一个统一的视觉-语言世界模型，通过在共享的“思维流”推理过程中自适应地组合感知、语言、几何与模拟能力，在多视角空间推理上零样本超越领先的开放权重模型，并支持基于想象场景的反事实模拟。

    

    人类在问题展开的过程中，会灵活地结合不同形式的推理来思考物理世界。实现这种灵活性是人工智能领域一个重大的开放性挑战：各种能力往往分散在不同的专用模型之中，限制了它们的组合方式以及共同适应任务的能力。我们提出了 Ψ-0.6，一个统一的视觉-语言世界模型，它能够在一个我们称为“思维流”的共享推理过程中自适应地组合感知、语言、几何与模拟。这些思维流交替穿插图像、文本以及几何与运动的显式表示，模型可以通过统一的 token 接口生成并解释所有这些内容。Ψ-0.6 在多视角空间推理任务上零样本超越了领先的开放权重模型，并且通过基于几何中间表示的推理获得了进一步提升。它还支持反事实模拟：通过生成并解释想象中的场景来判断，例如，某……

    arXiv:2610.32177v1 Announce Type: new  Abstract: Humans reason about the physical world by flexibly combining different forms of inference as a problem unfolds. Achieving this flexibility is a major open challenge in AI: capabilities often remain separated across specialized models, limiting how they can be composed and jointly adapted to a task. We introduce $\Psi$-0.6, a unified vision-language world model that adaptively composes perception, language, geometry, and simulation within a shared reasoning process we call streams of thought. These streams interleave images, text, and explicit representations of geometry and motion, all of which the model can generate and interpret through a shared token interface. $\Psi$-0.6 surpasses leading open-weight models zero-shot on multi-view spatial reasoning, with further gains from reasoning through geometric intermediates. It also supports counterfactual simulation: generating and interpreting imagined scenes to determine, for example, wheth
    
[^117]: 曲率感知裁剪：利用近似二阶信息缓解DPSGD中的偏差

    Curvature-Aware Clipping: Mitigating Bias in DPSGD via Approximate Second-Order Information

    [https://arxiv.org/abs/2610.32175](https://arxiv.org/abs/2610.32175)

    提出CA-DPSGD框架，利用近似二阶信息构建曲率感知的最优裁剪空间，并结合各向异性噪声机制，有效缓解DPSGD中的梯度裁剪偏差问题，性能超越现有最先进的裁剪方法。

    

    差分隐私随机梯度下降（DPSGD）已成为在严格隐私保证下训练深度神经网络的事实标准。然而，DPSGD的性能受到梯度裁剪偏差和差分隐私噪声的制约。尽管已有研究尝试解决这些问题，但由裁剪偏差引起的几何错位仍然是一个关键挑战。为弥补这一不足，我们提出了一种新颖的框架——曲率感知差分隐私随机梯度下降（CA-DPSGD），该框架利用近似二阶信息构建一个最优的、曲率感知的裁剪空间。CA-DPSGD集成了一个经过校正的曲率矩阵估计器，并结合各向异性噪声机制以确保隐私保证。大量的实证实验表明，我们的方法优于当前最先进的裁剪方法。

    arXiv:2610.32175v1 Announce Type: new  Abstract: Differentially Private Stochastic Gradient Descent (DPSGD) has become the de facto standard for training deep neural networks with rigorous privacy guarantees. However, the utility of DPSGD is constrained by gradient clipping bias and DP noise. While existing work has tried to address these issues, the geometric misalignment caused by clipping bias remains a critical challenge. To bridge this gap, we propose a novel framework, Curvature-Aware Differentially Private Stochastic Gradient Descent (CA-DPSGD), which leverages approximate second-order information to construct an optimal, curvature-aware clipping space. CA-DPSGD integrates a rectified estimator for the curvature matrix and is coupled with an anisotropic noise mechanism to ensure privacy guarantees. Extensive empirical experiments demonstrate that our approach outperforms state-of-the-art clipping methods.
    
[^118]: 基于神经基方法的纳维-斯托克斯动力学求解与学习

    Solving and Learning Navier-Stokes Dynamics with the Neural Basis Method

    [https://arxiv.org/abs/2610.32167](https://arxiv.org/abs/2610.32167)

    提出神经基方法（NBM），基于“先表示后投影”原则，将纳维-斯托克斯方程的求解与算子学习转化为固定神经基上的参数到系数映射，同时具备后验误差估计保证与良好的可扩展性。

    

    物理信息机器学习已成为流体动力学预测建模的一个重要方向，它将控制物理方程与学习到的表示相结合。然而，现有的主流方法既不能保证较小的PDE残差损失意味着较小的解误差，也无法随问题规模的增长而有效扩展。本文提出神经基方法（NBM）作为求解纳维-斯托克斯动力学的一种新方法。NBM建立在“先表示、后投影”的原则之上：固定的神经基函数定义表示空间，而通过投影控制方程来确定展开系数。同一组神经基为参数化解流形提供了公共坐标系，从而将算子学习重新表述为从参数到系数的映射，并通过消除显式空间依赖性实现了可扩展性。对于不可压缩流动，NBM给出了一种仅含速度变量的公式，其残差可作为后验速度误差估计。

    arXiv:2610.32167v1 Announce Type: cross  Abstract: Physics-informed machine learning has become a major direction in predictive modeling of fluid dynamics, combining governing physics with learned representations. Yet prevailing formulations neither guarantee that small PDE-residual losses imply small solution errors nor scale effectively with increasing problem size. Here we present the neural basis method (NBM) as a new approach for Navier--Stokes dynamics. NBM is built on a ``representation-then-projection'' principle: fixed neural bases define the representation space, while projection of the governing equations determines the expansion coefficients. The same bases define common coordinates for the parametric solution manifold, recasting operator learning as a parameter-to-coefficient mapping and enabling scalability by removing explicit spatial dependence. For incompressible flow, NBM yields a velocity-only formulation whose residual provides an a posteriori velocity-error estimat
    
[^119]: 稀疏性驱动的多任务神经网络功能模块化的涌现

    Sparsity-driven emergence of functional modularity in multitask neural networks

    [https://arxiv.org/abs/2610.32153](https://arxiv.org/abs/2610.32153)

    稀疏性正则化无需预定义模块边界即可促使多任务神经网络自发涌现功能模块化，形成可在深层选择性保留、重组的紧凑“功能token”子网络。

    

    神经网络能够学习多个任务，但学习过程如何将共享计算组织成专门化且可复用的组件，目前仍知之甚少。在这项工作中，我们证明稀疏性诱导的正则化能够在无需预定义模块边界的情况下，促进多任务网络中功能模块化的形成。通过在多层感知机和轻量级transformer中进行功能剖析，我们识别出了紧凑的、具有任务选择性的子网络，并将其命名为“功能token”（functional tokens）。在基于规则的任务族中，选择性地保留这些token可以维持其对应的功能，同时降低其他任务的表现。这种专门化在训练过程中优先在较深层中发展，而较浅的层则保留共享计算。正则化的强度与形式会塑造模块的紧凑性和专门化程度，而任务权重的设置则会改变功能的分配方式。此外，提取出的功能token还可以通过替代路径进行重组……

    arXiv:2610.32153v1 Announce Type: cross  Abstract: Neural networks can learn multiple tasks, but how learning organizes shared computation into specialized and reusable components remains poorly understood. In this work, we show that sparsity inducing regularization promotes functional modularity in multitask networks without predefined module boundaries. Using functional profiling in multilayer perceptrons and a lightweight transformer, we identified compact, task-selective subnetworks, termed functional tokens. Across rule-based task families, selectively retaining these tokens preserves their associated functions while reducing performance on other tasks. Such specialization develops preferentially in deeper layers during training, while earlier layers retain shared computation. Regularization strength and form shape module compactness and specialization, while task weighting alters functional allocation. Moreover, extracted functional tokens can be recombined through alternative pa
    
[^120]: 面向Graphon上变分问题的隐式神经表示

    Implicit Neural Representations for Variational Problems on Graphons

    [https://arxiv.org/abs/2610.32147](https://arxiv.org/abs/2610.32147)

    该论文提出用隐式神经网络表示Graphon并通过梯度下降求解其上的变分问题，其创新在于结合多尺度正弦残差架构、经隐式微分的密度约束内嵌求解器以及对称性感知的蒙特卡洛估计器，服务于极值图论与稠密随机图大偏差理论。

    

    我们提出了一个用于求解Graphon上变分问题的神经探索框架，其中Graphon是指作为稠密图序列极限而出现的对称可测函数 $[0,1]^2 \to [0,1]$。此类问题在数学文献的两个领域中处于核心地位：其一是极值图论，它研究在给定约束下的图参数优化问题，尤其针对顶点数量巨大的图；其二是稠密随机图的大偏差理论，它通过受约束的Graphon优化来刻画稀有事件的结构。我们采用隐式神经网络来表示Graphon，并通过梯度下降来优化以Graphon为变量的目标函数。我们的设计结合了三个要素：一种偏向于尖锐、阶跃状Graphon的多尺度正弦残差架构；一个内嵌的求解器，用于强制施加单一经验密度约束，并通过隐式微分进行求导；以及对对称性敏感的蒙特卡洛估计器。

    arXiv:2610.32147v1 Announce Type: cross  Abstract: We propose a neural exploratory framework for variational problems over graphons, i.e., symmetric measurable functions $[0,1]^2 \to [0,1]$, that arise as limits of dense graph sequences. Such problems are central to two parts of the mathematical literature: extremal graph theory, which studies graph parameter optimisation under given restrictions, often for those graphs with a large number of vertices, and the large deviation theory of dense random graphs, which characterises the structure of rare events through constrained graphon optimisation. We represent graphons by implicit neural networks and optimise graphon objectives by gradient descent. Our design combines three ingredients: a multi-scale sinusoidal residual architecture biased toward sharp, step-like graphons; an embedded solver that enforces a single empirical density constraint and is differentiated by implicit differentiation; and symmetry-aware Monte Carlo estimators. On
    
[^121]: 重新思考 LoRA 初始化以实现稳健的非对称学习率

    Rethinking LoRA Initialization for Robust Asymmetric Learning Rates

    [https://arxiv.org/abs/2610.32142](https://arxiv.org/abs/2610.32142)

    该论文通过对 LoRA 动力学的信号-噪声分解，揭示了标准初始化下由初始化引起的噪声项是非对称学习率训练不稳定和性能崩溃的根源，从而主张重新设计 LoRA 初始化以实现稳健的非对称学习率。

    

    低秩适应方法对学习率（LR）的选择往往非常敏感。近期的实证研究表明，一旦学习率经过仔细调优，原始的 LoRA 仍然可以与许多已提出的 LoRA 变体保持竞争力。这将核心的实际问题从设计更多变体转变为降低学习率选择的成本和脆弱性。LoRA+ 通过非对称学习率解决了这一问题，对上投影矩阵 $B$ 使用比下投影矩阵 $A$ 更大的学习率。然而，比率 $\eta_B / \eta_A$ 本身成为一个额外的超参数：较大的比率可以改善适配效果，但在标准 LoRA 初始化下，它们也可能导致训练不稳定以及依赖比率的性能崩溃。我们通过对 LoRA 动力学的信号-噪声分解来识别这种脆弱性的来源。在标准初始化下，更新中包含一个由初始化引起的噪声项，其大小可能受到……（摘要原文在此处截断）

    arXiv:2610.32142v1 Announce Type: cross  Abstract: Low-Rank Adaptation (LoRA) is often highly sensitive to learning-rate (LR) choices. Recent empirical studies suggest that, once LRs are carefully tuned, vanilla LoRA can remain competitive with many proposed LoRA variants. This shifts the central practical question from designing ever more variants to reducing the cost and brittleness of LR selection. LoRA+ addresses this issue with asymmetric LRs, using a larger LR for the up-projection matrix $B$ than for the down-projection matrix $A$. However, the ratio $\eta_B / \eta_A$ itself becomes an additional hyperparameter: large ratios can improve adaptation, but under the standard LoRA initialization they can also cause unstable training and ratio-dependent performance collapse. We identify the source of this brittleness through a signal--noise decomposition of LoRA dynamics. Under standard \initA, the update contains an initialization-induced noise term whose magnitude can be affected by
    
[^122]: 超越数据调度：基于损失重加权的混合数据快速LLM预训练

    Beyond Data Schedule: Fast LLM Pretraining on Data Mixtures by Loss-Reweighting

    [https://arxiv.org/abs/2610.32141](https://arxiv.org/abs/2610.32141)

    提出LARE方法，在保持数据暴露不变的前提下，根据运行中的损失统计自适应地重加权各领域的损失贡献，使单一检查点的知识与推理性能超越多阶段预训练的最佳结果，从而实现更快的LLM预训练。

    

    大型语言模型通常通过连续的预训练阶段获得专业能力，但这些收益可能以遗忘先前知识为代价。在这项工作中，我们研究了是否可以通过改变每个领域对学习的贡献强度，同时保持其数据暴露量不变，来改善这种权衡。我们发现，在受控的知识-数学实验中，损失重加权产生了一个单一的检查点，其表现超越了沿任一阶段轨迹所能达到的最佳知识和推理分数。这揭示了一种仅靠数据混合比例无法捕捉的独立控制手段。基于这一发现，我们提出了LARE（固定暴露下的损失自适应重加权），它根据运行中的损失统计自动调整各领域权重。受梯度噪声理论分析的启发，其归一化的逆平方根规则在不改变采样混合的情况下平衡了相对损失尺度。

    arXiv:2610.32141v1 Announce Type: cross  Abstract: Large language models often acquire specialized capabilities through successive pretraining stages, but these gains can come at the cost of forgetting earlier knowledge. In this work, we investigate whether such trade-offs can be improved by changing how strongly each domain contributes to learning while keeping its data exposure fixed. We find that, in controlled knowledge--mathematics experiments, loss reweighting produces a single checkpoint that surpasses the best knowledge and reasoning scores attained along either stage trajectory. This reveals an independent control that data-mixture ratios alone do not capture. Building on this finding, we propose LARE (Loss-Adaptive Reweighting under fixed Exposure), which automatically adjusts domain weights from running loss statistics. Motivated by a theoretical analysis of gradient noise, its normalized inverse-square-root rule balances relative loss scales without changing the sampled mix
    
[^123]: SyncPro：面向主动式流媒体视频理解的同步推理与响应

    SyncPro: Synchronous Reasoning and Responding for Proactive Streaming Video Understanding

    [https://arxiv.org/abs/2610.32132](https://arxiv.org/abs/2610.32132)

    SyncPro提出了一种同步推理与响应的机制，解决了主动式流媒体视频理解中持续推理与及时响应难以协调的问题，避免了固定推理间隔导致的无谓更新或延迟响应。

    

    主动式流媒体视频理解要求助手能够追踪正在展开的事件，并在用户所请求的条件满足时做出响应。为了做出这些决策，助手必须积累相关证据，并在新观察到来时重新评估响应条件。现有工作为这一过程提供了互补的能力：主动式视频助手学会了何时响应，而流媒体视频推理方法则生成按计划安排的中间文本来组织观察结果并支持后续回答。这些能力共同为主动式推理奠定了基础，但如何协调持续进行的推理与及时的响应仍未得到充分探索。在直接适配的方式下，固定的推理间隔可能在没有相关变化发生时触发更新，或者将有用的更新推迟到下一个预定步骤。同时，共享同一条生成路径也将两个不同的需求耦合在一起：即为未来决策保留证据……

    arXiv:2610.32132v1 Announce Type: cross  Abstract: Proactive streaming video understanding requires assistants to track unfolding events and respond when a user's requested condition is met. To make these decisions, assistants must accumulate relevant evidence and reassess response conditions as new observations arrive. Existing work offers complementary capabilities for this process: proactive video assistants learn when to respond, while streaming video reasoning methods generate scheduled intermediate text to organize observations and support subsequent answers. Together, these capabilities provide a foundation for proactive reasoning, but how to coordinate ongoing reasoning with timely responses remains underexplored. In a direct adaptation, fixed reasoning intervals can trigger updates when nothing relevant has changed or postpone useful updates until the next scheduled step. Sharing one generation path also couples two distinct requirements: preserving evidence for future decisio
    
[^124]: MPQ：训练后量化的消息传递视角

    MPQ: A Message-Passing View of Post-Training Quantization

    [https://arxiv.org/abs/2610.32127](https://arxiv.org/abs/2610.32127)

    该论文提出消息传递量化（MPQ），将训练后量化重新建模为离散码本上的后验推断，通过在权重块间传递软后验信息并在最后阶段才做硬判决，从而超越GPTQ等现有硬判决求解器，提升大语言模型低比特量化的精度。

    

    训练后量化（PTQ）是高效部署大语言模型的关键技术，因为它无需重新训练即可将权重压缩为低比特表示。现有的PTQ求解器，包括GPTQ、LDLQ和块坐标下降法，都会在考虑相邻权重块之前就把每个权重块固定为单个码字，使得接近最优的备选方案未被探索。在这项工作中，我们将逐行PTQ转化为对离散码本的后验推断问题，其中现有求解器对应于零温度、硬判决的极限情况，即每个块仅保留最优码字。我们提出消息传递量化（MPQ），它在推断过程中跨块保留软后验信息，并仅在输出阶段才固定为硬码字。MPQ首先使用向量近似消息传递来计算近似的逐块后验分布，然后通过精确的块坐标下降法来精炼硬判决初始化结果。在2比特……

    arXiv:2610.32127v1 Announce Type: new  Abstract: Post-training quantization (PTQ) is a key technique for efficiently deploying large language models, as it compresses weights into low-bit representations without retraining. Existing PTQ solvers, including GPTQ, LDLQ, and block coordinate descent, commit each weight block to a single codeword before its neighbors are considered, leaving near-optimal alternatives unexplored. In this work, we cast row-wise PTQ as posterior inference over discrete codebooks, in which existing solvers correspond to a zero-temperature, hard-decision limit retaining only the top codeword per block. We propose Message-Passing Quantization (MPQ), which preserves soft posterior information across blocks during inference and commits to hard codewords only at the output stage. MPQ first uses vector approximate message passing to compute approximate blockwise posteriors, then refines a hard-decision initialization by exact block coordinate descent. Across 2-bit lat
    
[^125]: 离线最大熵强化学习的条件流模型

    Conditional Flow Models for Offline Maximum Entropy Reinforcement Learning

    [https://arxiv.org/abs/2610.32125](https://arxiv.org/abs/2610.32125)

    该论文提出一个分离式框架，利用基于概率流ODE的条件扩散模型分别估计奖励分布和转移核，从而构建软最优贝尔曼算子的数据驱动近似来估计最优软动作价值函数，并提供了非渐近理论保证。

    

    离线最大熵强化学习旨在从固定数据集中估计最优软动作价值函数 Q*，而无需与环境进行进一步的交互。尽管 Q* 由软最优贝尔曼方程刻画，但由于奖励分布和转移核均未知，相应的贝尔曼算子无法直接计算。因此，估计 Q* 需要同时对环境建模和价值函数学习进行联合处理。我们提出了一个将这两项任务分开处理的框架。基于概率流常微分方程（ODE）的条件扩散模型用于估计条件奖励分布和转移核。由此得到的分布估计器定义了软最优贝尔曼算子的数据驱动近似，通过在一类深度神经网络中最小化经验贝尔曼残差来估计 Q*。我们建立了非渐近的理论保证……

    arXiv:2610.32125v1 Announce Type: new  Abstract: Offline maximum entropy reinforcement learning seeks to estimate the optimal soft action-value function \(Q^*\) from a fixed dataset without further interaction with the environment. Although \(Q^*\) is characterized by the soft optimal Bellman equation, the corresponding Bellman operator cannot be evaluated directly because both the reward distribution and the transition kernel are unknown. Consequently, estimating \(Q^*\) requires the joint treatment of environment modeling and value function learning. We propose a framework that addresses these tasks separately. Conditional diffusion models based on probability flow ODEs estimate the conditional reward distribution and transition kernel. The resulting distributional estimators define a data driven approximation of the soft optimal Bellman operator, whose empirical Bellman residual is minimized over a class of deep neural networks to estimate \(Q^*\). We establish nonasymptotic guarant
    
[^126]: 语义丰富，度量情感盲区：现代文本嵌入器中的几何盲点

    Semantically Rich, Metrically Affect-Blind: A Geometric Blind Spot in Modern Text Embedders

    [https://arxiv.org/abs/2610.32122](https://arxiv.org/abs/2610.32122)

    现代文本嵌入器的嵌入向量中几乎可以完美解码情感信息（线性探针准确率高达91.3%），但其度量空间（余弦相似度）几乎完全无法区分情感，导致依赖度量的检索、聚类、去重和RAG系统对这类信息“视而不见”。

    

    我们报告了现代文本嵌入器中的一种系统性分离现象：情感几乎可以完美地从其嵌入中解码出来，却几乎完全缺失于其度量之中。在涵盖2021至2026年的十一个冻结编码器上，包括2026年世代的模型以及40亿参数的LLM嵌入器Qwen3-Embedding-4B，线性探针能以高达91.3%的准确率恢复句子极性，而余弦相似度区分极性的差距仅为0.003至0.064。Qwen3-Embedding-4B是该组中探针性能最强的模型，其探针准确率达到91.3%，但检索MAP仅为64.3。由于检索、聚类、去重和检索增强生成（RAG）都是基于度量而非训练好的探针来运行的，这些系统对其编码器确实包含的信息实际上是“视而不见”的。一项任务条件控制实验表明，这反映的是预训练目标本身，而非提示方式的伪影：将Jina-v5以其分类任务为条件进行……（原文摘要在此处截断）

    arXiv:2610.32122v1 Announce Type: new  Abstract: We report a systematic dissociation in modern text embedders: sentiment is almost perfectly decodable from their embeddings, yet almost entirely absent from their metric. Across eleven frozen encoders spanning 2021 to 2026, including 2026-generation models and the 4B-parameter LLM embedder Qwen3-Embedding-4B, a linear probe recovers sentence polarity at up to 91.3% accuracy while cosine similarity separates polarity by a margin of only 0.003-0.064. Qwen3-Embedding-4B, the strongest probe in the panel, reaches 91.3% probe accuracy but only 64.3 retrieval MAP. Because retrieval, clustering, deduplication, and retrieval-augmented generation operate on the metric rather than on a trained probe, these systems are effectively blind to information their encoder demonstrably contains. A task-conditioning control suggests this reflects the pretraining objective rather than a prompting artifact: conditioning Jina-v5 on its classification task rath
    
[^127]: Tampered ML：评估大语言模型对机器学习研究的判断力

    Tampered ML: Evaluating LLM Judgment of Machine Learning Research

    [https://arxiv.org/abs/2610.32114](https://arxiv.org/abs/2610.32114)

    该论文提出Tampered ML基准，通过将真实机器学习论文的实验提案与人为引入方法论缺陷的最小修改版本配对，来评估大语言模型识别研究方法缺陷的能力，发现前沿模型虽优于随机水平但远未达到可靠程度。

    

    加速AI对齐研究进展的一种途径是将对齐研究本身的部分工作自动化。这要求系统能够在难以监督的研究任务上可靠运行，包括识别研究提案中的缺陷，以及评估一个提议的实验是否能支持其预期结论。为了评估这一能力，我们引入了Tampered ML，这是一个包含1,858个条目对的基准，每个条目对由一个从已发表的机器学习论文中重构的实验提案，以及一个经过最小化编辑、引入了针对性方法论缺陷的反事实版本组成。模型必须在两种呈现顺序下识别出方法论上更合理的提案。一项针对分层样本的人类验证研究估计，86%的条目是有效的（95%置信区间：81-91%）。前沿模型的表现高于随机水平，但仍远未达到可靠程度（其中Opus 5达到77.9%的准确率）。它们对呈现顺序高度敏感，并且f……（摘要在此处被截断）

    arXiv:2610.32114v1 Announce Type: cross  Abstract: One approach to accelerating progress on AI alignment is to automate parts of alignment research itself. Doing so requires systems that can perform reliably on hard-to-supervise research tasks, including identifying flaws in research proposals and evaluating whether a proposed experiment supports its intended claim. To evaluate this capability, we introduce Tampered ML, a benchmark of 1,858 item-pairs, each consisting of an experiment proposal reconstructed from a published machine learning paper and a minimally edited counterfactual that introduces a targeted methodological flaw. Models must identify the more methodologically sound proposal in both presentation orders. A human validation study on a stratified sample estimates that 86% of items are valid (95% CI: 81-91%). Frontier models perform above chance but remain far from reliable (with Opus 5 achieving 77.9% accuracy). They are highly sensitive to the order of presentation and f
    
[^128]: REACT：思维链轨迹的表征早期分析与控制

    REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories

    [https://arxiv.org/abs/2610.32113](https://arxiv.org/abs/2610.32113)

    REACT 提出一种从成对安全与不安全轨迹中学习安全方向、并一次性施加于思维链早期隐藏状态的激活干预方法，发现最强控制增益集中在思维链的前两到三个 token 窗口，即使后续状态更能预测风险。

    

    思维链生成提供了一种在推理模型完成回答之前进行干预的机会。但是，风险最容易被检测到的时间点是否也能提供最有效的控制？我们通过 REACT 来研究这个问题，这是一种条件性激活干预方法，它从成对的安全与不安全轨迹中学习一个安全方向，并将其一次性应用于早期的思维链隐藏状态。模型权重保持冻结，所有后续 token 的生成不再进行进一步编辑。为了研究时机问题，我们在前五个思维链位置上分别训练方向，并使用无门控编辑配合编辑后的完整重新生成进行评估。在三个蒸馏推理模型和三个生成种子上，最强的增益集中在两到三个 token 的早期窗口内；在 DeepSeek-R1-Distill-Llama-8B 上，效果在第四和第五个 token 处降至接近于零，尽管较晚的状态可能更能预测风险。

    arXiv:2610.32113v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) generation offers an opportunity to intervene before a reasoning model completes its answer. But does the point at which risk is easiest to detect also offer the most effective control? We study this question with REACT, a conditional activation intervention that learns a safety direction from paired safe and unsafe trajectories and applies it once to an early CoT hidden state. The model weights remain frozen, and all subsequent tokens are generated without further edits. To study timing, we train directions separately at each of the first five CoT positions and evaluate ungated edits with full post-edit regeneration. Across three distilled reasoning models and three generation seeds, the strongest gains concentrate in a two- to three-token early window; on DeepSeek-R1-Distill-Llama-8B, the effect falls to approximately zero at the fourth and fifth tokens, even though later states can be more predictive of risk. 
    
[^129]: 将Transformer理论付诸检验

    Putting Transformer Theories to the Test

    [https://arxiv.org/abs/2610.32104](https://arxiv.org/abs/2610.32104)

    该论文将七种关于Transformer能力的理论预测统一到同一矩阵中，并通过受控实验验证，发现没有任何单一理论能完全解释Transformer在不同计算任务上的表现。

    

    目前已发展出若干形式化理论来刻画Transformer架构的能力与局限，每种理论采用不同的抽象方式来捕捉该架构的不同特征。由于底层假设和概念框架的差异，这些理论对于Transformer认为哪些任务容易、哪些任务困难的预测可能并不一致。我们梳理了七种关于Transformer能力的理论（Star-free、AC⁰、TC⁰、C-RASP、MPC、BAPO和Sensitivity），并将其结果转化为一个统一的预测矩阵，覆盖一系列具有不同计算结构的任务。我们提供了新的理论结果来补全该矩阵。随后，我们通过受控实验来评估这些理论预测，实验测量了分布内样本复杂度、长度泛化能力以及预训练语言模型的性能。我们发现，没有任何单一理论能够解释……（摘要原文不完整）

    arXiv:2610.32104v1 Announce Type: cross  Abstract: Several formal theories have been developed to characterize the capabilities and limitations of the transformer architecture, each employing different abstractions that capture different features of the architecture. Due to the differences in their underlying assumptions and conceptual frameworks, these theories can disagree in their predictions about what tasks transformers find easy or hard. We map out seven theories of transformer capability (Star-free, $\mathsf{AC}^0$, $\mathsf{TC}^0$, C-RASP, MPC, BAPO, and Sensitivity) and translate their results into a common matrix of predictions across a suite of tasks covering different computational structures. We provide new theoretical results to complete this matrix. We then evaluate theoretical predictions through controlled experiments measuring in-distribution sample complexity, length generalization, and pretrained language model performance. We find that no single theory explains the
    
[^130]: ReMedI：基于推理时细化与自蒸馏的可控分子优化

    ReMedI: Inference-Time Refinement and Self-Distillation for Controllable Molecular Optimization

    [https://arxiv.org/abs/2610.32103](https://arxiv.org/abs/2610.32103)

    该论文提出ReMedI方法，以分子语言模型作为基础提议者，结合高效的Metropolis-Hastings搜索策略与推理时细化及自蒸馏，在无需昂贵任务特定数据集的情况下实现可控的多目标分子先导化合物优化。

    

    在药物研发中，先导化合物优化必须在改善相互竞争的性质（例如提高渗透性、降低毒性）的同时，限制可能损害先导化合物药效的骨架变化。基于强化学习的后训练方法（如C-MORAL）通过微调分子语言模型来直接针对感兴趣的目标优化分子先导化合物，从而解决这一问题，但这类方法依赖于昂贵的任务特定数据集，并且在实现多目标优化（MOO）方面面临挑战。我们认为，预训练模型可能已经包含有用的分子编辑操作，而零样本生成无法高效地触及这些操作。受零样本生成搜索能力有限的启发，我们开发了一种高效的Metropolis-Hastings搜索策略，以更好地探索提议分布。我们提出了ReMedI，该方法利用分子语言模型作为基础提议者进行推理时对齐。给定一个先导化合物和一组多目标任务……

    arXiv:2610.32103v1 Announce Type: cross  Abstract: In drug development, lead optimization must improve competing properties, such as increasing permeability and reducing toxicity, while limiting scaffold changes that could compromise the lead's potency. RL-based post-training, such as C-MORAL, addresses this problem by fine-tuning molecular language models to directly optimize molecular leads for objectives of interest, which relies on expensive task-specific datasets and faces challenges in achieving multi-objective optimization (MOO). We argue a pre-trained model may already contain useful molecular edits that zero-shot generation cannot efficiently reach. Motivated by the limited search capabilities of zero-shot generation, we develop an efficient Metropolis-Hastings search strategy to better explore the proposal distribution. We propose ReMedI, which uses the molecular language model as a base proposer for inference-time alignment. Given a lead and a set of multi-objective tasks, t
    
[^131]: PRISM：通过隐式随机调制促进大语言模型的采样多样性

    PRISM: Promoting Sampling Diversity of Large Language Models with Implicit Stochastic Modulation

    [https://arxiv.org/abs/2610.32101](https://arxiv.org/abs/2610.32101)

    PRISM是一个训练时框架，通过次模数据筛选与随机嵌入调制两种机制，将采样多样性内化为大语言模型自身的能力，从而突破质量-多样性的权衡前沿。

    

    大语言模型（LLM）存在模式坍缩问题，即其输出分布会集中于熟悉的语义模式。当多样性消失时，重复采样只会带来语义上的重复而非新的发现，使得罕见但更优的解难以浮现，也让自我进化与持续学习难以维系。现有的大多数补救方法仅在推理时进行干预，只是对样本进行塑形，而没有将多样性转化为模型自身的一种能力。我们提出PRISM，这是一个训练时框架，将多样性内化到模型之中。PRISM结合了两个核心组件：(i) 次模数据筛选，为每个提示配对多样且正确的多教师解法；(ii) 随机嵌入调制，将特殊token的扰动转化为隐式信号，促使模型采样出不同的输出。理论分析证明了该机制的稳定性。在WritingPrompts和BookMIA数据集上，PRISM将质量-多样性前沿向外推移，取得……

    arXiv:2610.32101v1 Announce Type: new  Abstract: Large language models (LLMs) exhibit mode collapse, where their output distribution concentrates on familiar semantic modes. When diversity disappears, repeated sampling becomes semantic repetition rather than discovery, making rare but stronger solutions unlikely to surface and leaving self-evolution and continual learning difficult to sustain. Most remedies intervene only at inference time, shaping sam- ples without making diversity a model capability. We propose PRISM, a training-time framework that internalizes diversity within the model. PRISM combines (i) submodular curation that pairs each prompt with diverse-yet-correct multi-teacher solutions, and (ii) stochastic embedding modulation, which turns special-token perturbations into implicit signals for sampling distinct outputs. A theoretical analysis establishes the stability of this mechanism. On WritingPrompts and BookMIA, PRISM moves the quality-diversity frontier outward, achi
    
[^132]: RADIANT：以物体为中心的前馈全局光照

    RADIANT: Object-Centric Feed-Forward Global Illumination

    [https://arxiv.org/abs/2610.32075](https://arxiv.org/abs/2610.32075)

    提出RADIANT架构，以物体为中心、每个物体仅用32个表面补丁token表示，将注意力序列长度与多边形数量解耦，从而在无需逐场景优化或运行时光线追踪的情况下高效合成新3D场景的全局光照。

    

    前馈神经渲染器能够在全新3D场景中合成全局光照，无需逐场景优化或运行时光线追踪。现有基于Transformer的渲染器以单个三角形为单位进行token化，因此自注意力的计算开销随网格分辨率呈二次方增长。然而，宏观的辐射传输本质上由底层表面几何与物体空间排列决定，而非特定的网格离散化方式；因此更精细的网格剖分只会在不改变底层传输物理的前提下增加token数量。这促使我们采用一种与离散化无关的、以物体为中心的表示方法。我们提出RADIANT（RADiance-based Inter-Actor Neural Transformer，基于辐射的交互体神经Transformer），这是一种从点输入进行前馈全局光照的以物体为中心的架构。通过用紧凑的32个表面补丁token表示每个物体，RADIANT将注意力序列长度与多边形数量解耦。在单一网络中，物体间场景Transformer……

    arXiv:2610.32075v1 Announce Type: cross  Abstract: Feed-forward neural renderers synthesize global illumination across novel 3D scenes without per-scene optimization or runtime ray tracing. Existing transformer-based renderers tokenize individual triangles, so self-attention scales quadratically with mesh resolution. Yet macroscopic radiative transport is governed by the underlying surface geometry and object arrangements rather than a particular mesh discretization; finer tessellation can thus inflate token counts without changing the underlying transport physics, motivating a discretization-agnostic and object-centric representation. We introduce RADIANT (RADiance-based Inter-Actor Neural Transformer), an object-centric architecture for feed-forward global illumination from point inputs. By representing each object with a compact budget of 32 surface-patch tokens, RADIANT decouples attention sequence length from polygon counts. In a single network, an inter-object scene transformer m
    
[^133]: DBCM：用于时间序列预测测试时适应的双分支校准模块

    DBCM: Dual-Branch Calibration Module for Test-Time Adaptation in Time Series Forecasting

    [https://arxiv.org/abs/2610.32071](https://arxiv.org/abs/2610.32071)

    提出双分支校准模块DBCM，通过共享适配器利用跨通道预测残差的共享结构，并结合低秩通道分支保留有限的通道特定灵活性，实现时间序列预测中更高效的测试时适应。

    

    非平稳性会在时间序列预测（TSF）中引发测试时的分布偏移，导致预测结果与相应真实值之间产生预测残差。近期关于时间序列预测测试时适应的工作通过在冻结的预测器周围引入轻量级校准模块来解决这一问题，这些模块利用测试时逐步获得的观测数据进行更新。现有方法通常采用逐通道适配器，隐含地假设预测残差本质上是通道特定的。然而，我们观察到预测残差在通道之间往往表现出大量的共享结构，这表明完全独立的逐通道参数化可能是冗余的。受此观察启发，我们提出了双分支校准模块（DBCM），它通过共享适配器在通道之间共享适应能力，同时通过低秩通道分支保留有限的通道特定灵活性。

    arXiv:2610.32071v1 Announce Type: cross  Abstract: Non-stationarity induces test-time distribution shifts in time series forecasting (TSF), resulting in forecast residuals between predictions and the corresponding ground truth. Recent works on test-time adaptation for TSF address this by introducing lightweight calibration modules around frozen forecasters, updated using observations that become incrementally available at test time. Existing approaches typically employ channel-wise adapters, implicitly assuming that forecast residuals are inherently channel-specific. However, we observe that forecast residuals often exhibit substantial shared structure across channels, suggesting that fully independent channel-wise parameterization may be redundant. Motivated by this observation, we propose the Dual-Branch Calibration Module (DBCM), which shares adaptation capacity across channels through a shared adapter while retaining limited channel-specific flexibility through a low-rank channel-w
    
[^134]: EEG基础模型的受控评估：随机初始化、频带功率与基于验证集选择的探测层

    Controlled Evaluation of EEG Foundation Models: Random Initialisation, Band-Power and Validation-Selected Probes

    [https://arxiv.org/abs/2610.32067](https://arxiv.org/abs/2610.32067)

    该论文通过引入未训练编码器、频带功率基线和基于验证集的层选择三种对照，对多个EEG基础模型进行受控评估，发现其预训练收益大多不具统计稳健性，随机初始化和简单的频带功率基线即可与之匹敌。

    

    EEG基础模型报告了下游任务的性能提升，但其评估协议很少包含三种对照：即同一编码器未经训练的版本、在同一数据上拟合的频谱特征，以及不看测试分数而选择的层。因此，我们在统一协议下、采用全部三种对照和多个随机种子，在五个基准和四个附加任务族上评估了五个公开发布的EEG基础模型和一个自研的JEPA风格模型。这些对照从三个方面改变了评估结论。第一，若按测试分数选择探测层，逐层峰值平均提高+0.034，并在36个单元格中的9个里逆转了峰值与输出之间的差距。第二，当在验证集上选择层时，随机初始化在睡眠和情绪任务上与预训练持平甚至更优（涉及一半的模型），且没有任何模型在情绪或运动想象任务上胜过五频带对数功率基线。第三，在完全微调后，30对实验中仅有5对的增益超过其随机种子波动范围的2倍……

    arXiv:2610.32067v1 Announce Type: cross  Abstract: EEG foundation models report downstream gains, yet their protocols seldom include three controls. These are the same encoder untrained, spectral features fitted on the same data, and a layer picked without looking at test scores. We therefore evaluate five publicly released EEG foundation models and an in-house JEPA-style model under one protocol with all three controls and several seeds, on five benchmarks and four added task families. The controls change the reading in three ways. First, picking the probed layer by its test score raises the layer-wise peak by an average of +0.034 and reverses the peak-versus-output gap in 9 of 36 cells. Second, with the layer picked on validation, random initialisation matches or beats pretraining for half of the models on sleep and emotion, and no model beats a five-band log-power baseline on emotion or motor imagery. Third, after full fine-tuning only 5 of 30 pairs gain beyond twice their seed spre
    
[^135]: 概念针灸师：一种基于稀疏自编码器的扩散模型高效精确概念擦除方法

    Concept Acupuncturist: An Efficient and Precise Concept Erasure Method for Diffusion Models via Sparse Autoencoders

    [https://arxiv.org/abs/2610.32059](https://arxiv.org/abs/2610.32059)

    该论文提出ConceptACP框架，利用稀疏自编码器的可解释性分析定位U-Net去噪器中与目标概念最相关的交叉注意力块并进行针对性微调，从而实现对扩散模型概念的高效精确擦除。

    

    基于扩散的图像生成模型能够生成高质量图像，但可能生成有害或违反政策的内容，这促使机器遗忘（MU）技术被用于选择性地移除目标概念，同时保留模型的通用能力。然而，现有的机器遗忘方法往往难以应对概念表示的分布式和纠缠特性，当通过在全扩散轨迹上进行大规模微调来应用时，会导致优化效率低下和编辑不精确。为了弥补这一差距，我们提出了概念针灸师（ConceptACP），这是一种新颖的机器遗忘框架，它将可解释性引导的分析与无需推理的微调相结合，以实现高效且精确的概念移除。通过建立语义概念与 U-Net 去噪器中交叉注意力块之间的映射，并定位与概念最相关的块进行针对性优化，ConceptACP 避免了全扩散轨迹的优化，从而提高了（效率与精度）。

    arXiv:2610.32059v1 Announce Type: cross  Abstract: Diffusion-based image generation models produce high-quality images but may generate harmful or policy-violating content, motivating machine unlearning (MU) to selectively remove target concepts while preserving general capabilities. However, existing MU approaches often struggle with the distributed and entangled nature of concept representations, leading to inefficient optimization and imprecise edits when applied through large-scale fine-tuning over full diffusion trajectories. To bridge this gap, we propose Concept Acupuncturist (ConceptACP), a novel MU framework that integrates interpretability-guided analysis with inference-free fine-tuning for efficient and precise concept removal. By establishing mappings between semantic concepts and cross-attention blocks in the U-Net denoiser and localizing the most concept-relevant block for targeted optimization, ConceptACP avoids full diffusion trajectory optimization, thereby improving f
    
[^136]: 你有多信任一只狗？量化推理语言模型中的提示遵循行为

    How Much Do You Trust A Dog? Quantifying Hint-Following in Reasoning Language Models

    [https://arxiv.org/abs/2610.32055](https://arxiv.org/abs/2610.32055)

    该研究发现大语言模型对提示来源（如教授、朋友甚至狗）的答案偏移遵循可预测的指数倾斜结构，且偏移强度与来源可靠性基本无关，揭示了模型谄媚行为未被来源可靠性校准的问题。

    

    大型语言模型（LLM）常常表现出谄媚行为，即使在客观的事实和推理任务上，也会为了迎合用户的建议而改变自己的答案。然而，这种影响的定量形式——以及它是否与信息来源的可靠性相校准——仍然知之甚少。在本工作中，我们通过MMLU问题上的提示诱导答案偏移来研究这一行为，评估形如“[来源]认为答案是(X)”的提示。我们在三个开源模型系列（Qwen、Gemma和SmolLM）上系统地改变提示所归属的来源（教授、朋友或狗）。我们表明，提示引起的变化遵循一个可预测的结构，该结构可以用指数倾斜很好地刻画：提示在被提示答案的对数几率上引起近似恒定的均值偏移，且问题与目标的易感性在不同来源之间很大程度上是共享的。为了探索这种模式如何产生，我们引入了一个形式化的……

    arXiv:2610.32055v1 Announce Type: new  Abstract: Large language models (LLMs) often exhibit sycophancy, altering their answers to align with user suggestions even on objective factual and reasoning tasks. Yet the quantitative form of this influence--- and whether it is calibrated to source reliability--- remains poorly understood. In this work, we investigate this behavior through hint-induced answer shifts on MMLU questions, evaluating prompts of the form ``[Source] thinks the answer is (X)''. We systematically vary the source to which the hint is attributed (a professor, a friend, or a dog) across three open-source model families, Qwen, Gemma, and SmolLM. We show that the change induced by the hint follows a predictable structure well-characterized by an exponential tilt: the hint induces an approximately constant mean shift in the hinted answer's log-odds, with question--target susceptibility largely shared across sources. To explore how this pattern can arise, we introduce a formal
    
[^137]: 利比里亚社区卫生计划中药物“最后一公里”配送的动态库存配给模型

    A Dynamic Inventory Rationing Model for Last-Mile Delivery of Medication in Liberia's Community Health Program

    [https://arxiv.org/abs/2610.32051](https://arxiv.org/abs/2610.32051)

    该论文将利比里亚社区卫生计划中稀缺药物的动态分配问题建模为有限时域随机控制问题，在转诊成本、患者未治疗成本与地区间公平性之间取得平衡，以解决“将库存下放给社区卫生助理”与“集中保留库存以应对不确定需求”之间的权衡难题。

    

    我们研究利比里亚社区卫生计划中稀缺药物的动态分配问题。一个中央仓库为一家卫生机构及其所监管的社区卫生助理（CHAs）供应药品。当社区卫生助理的药品耗尽时，患者会被转诊至该卫生机构——若患者完成转诊则产生交通成本，否则将得不到治疗。将库存分配给社区卫生助理可以让更多患者无需出行即可获得治疗，而将库存集中保留在中央仓库则保留了应对不确定需求的灵活性。这一分配问题必须在权衡上述因素的同时兼顾各地点之间的公平性。我们将该问题表述为一个有限时域的随机控制问题，其目标函数综合了转诊成本、患者未接受治疗的成本以及按需求加权的公平性效用。由于标准的动态规划方法在此情境下计算上不可行，我们开发了一种连续时间（方法）……（原文摘要在此处截断）

    arXiv:2610.32051v1 Announce Type: cross  Abstract: We study the dynamic allocation of scarce medications in Liberia's community health program. A central warehouse supplies a health facility and the Community Health Assistants (CHAs) it supervises. When a CHA runs out of medication, patients are referred to the facility, incurring travel costs if they complete the referral and remain untreated otherwise. Allocating inventory to CHAs allows more patients to receive treatment without traveling, while retaining inventory centrally preserves flexibility to respond to uncertain demand. The allocation problem must balance these considerations while accounting for fairness across locations. We formulate this problem as a finite-horizon stochastic control problem whose objective combines the costs of referrals and untreated patients with a demand-weighted fairness utility. Because standard dynamic programming methods are computationally intractable in this setting, we develop a continuous-time
    
[^138]: 测试时记忆演化能否在科学发现中超越测试时训练？

    Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery?

    [https://arxiv.org/abs/2610.32046](https://arxiv.org/abs/2610.32046)

    提出测试时记忆演化（TTME），一种无需更新模型权重的即插即用测试时适应方法，通过持续演化外部记忆来积累经验，在科学发现任务上以更高的计算效率超越了测试时训练（TTT）。

    

    近期的测试时训练（TTT）框架已使开放权重的大型语言模型（LLM）在科学发现任务上超越了由前沿模型驱动的仅推理框架。然而，TTT 在发现过程中需要在线更新模型权重，带来了额外的计算成本，使得在扩展到更大规模模型时计算效率成为一个重要的考量因素。这促使我们探索一种高效的面向科学发现的测试时适应方法。我们提出了测试时记忆演化（TTME），这是一种即插即用的测试时适应方法，作为 TTT 的替代方案。TTME 通过将经验蒸馏到外部记忆中并持续更新，以指导后续的发现过程，而无需更新模型权重。为了进行干净的对比，我们在相同的发现框架下评估 TTME 和 TTT，仅替换其中的测试时适应方法。我们的评估表明，TTME 的表现优于 TTT。

    arXiv:2610.32046v1 Announce Type: new  Abstract: Recent \emph{test-time training} (TTT) frameworks have enabled open-weight large language models (LLMs) to outperform inference-only frameworks powered by frontier models on scientific discovery tasks. However, TTT introduces additional computational cost for online weight updates during the discovery process, making computational efficiency an important consideration when scaling to larger models. This motivates us to explore an efficient \emph{test-time adaptation} method for scientific discovery. We introduce \emph{Test-Time Memory Evolution} (TTME), a plug-and-play test-time adaptation method, as an alternative to TTT. TTME continually updates the external memory by distilling the experience to guide subsequent discovery without updating model weights. To have a clean comparison, we evaluate TTME and TTT under the same discovery framework, replacing only the test-time adaptation method. Our evaluation shows that TTME outperforms TTT 
    
[^139]: NUViT：面向傅里叶空间数据的视觉Transformer

    NUViT: Vision Transformers for Fourier Space Data

    [https://arxiv.org/abs/2610.32041](https://arxiv.org/abs/2610.32041)

    提出了NUViT，一种通过重新设计图像分词、数据表示和位置编码以适配傅里叶空间数据频率与相位结构的视觉Transformer，在图像分类、晶格识别、MRI标注和冷冻电镜分析等多种傅里叶域任务中均优于标准架构。

    

    视觉Transformer（ViT）在许多计算机视觉任务中展现出强大的性能，但难以自然地扩展到自然图像领域之外。一个重要的例子是傅里叶空间数据，它被用于表示丰富多样的模态，包括医学成像和显微成像。标准的ViT分词方法将图像在笛卡尔网格上划分为方形图块，这种方式难以契合傅里叶空间数据的频率和相位结构。我们提出了非均匀视觉Transformer（Non-Uniform ViT），这是一种简单、通用的视觉Transformer，它重新设计了图像分词、数据表示和位置编码，以适应这种结构。在傅里叶域图像分类、二维晶格识别、MRI标注以及冷冻电镜蛋白质异质性分析上的实验表明，NUViT在捕捉傅里叶数据结构方面始终优于标准架构。

    arXiv:2610.32041v1 Announce Type: cross  Abstract: Vision Transformers (ViTs) have demonstrated strong performance in many computer vision tasks but do not naturally extend beyond the natural image domain. One important example is Fourier-space data, which is used to represent a rich and diverse set of modalities, including medical imaging and microscopy. Standard ViT tokenization partitions an image into square patches on a Cartesian grid, which poorly fit the frequency and phase structure of Fourier-space data. We introduce Non-Uniform ViT, a simple, general-purpose vision transformer that redesigns image tokenization, data representation, and positional encoding to respect this structure. Experiments on Fourier-domain image classification, 2D crystal lattice identification, MRI annotation, and cryo-EM protein heterogeneity analysis show that NUViT consistently captures the structure of Fourier data better than standard architectures.
    
[^140]: 重新思考长尾数据下监督对比学习中的表示几何

    Rethinking Representation Geometry in Supervised Contrastive Learning under Long-Tailed Data

    [https://arxiv.org/abs/2610.32039](https://arxiv.org/abs/2610.32039)

    本文揭示了现有长尾数据下的监督对比学习目标因依赖每类样本数量而无法在最优解处形成正则单纯形几何，并提出结构平衡方法与完全平衡的对比目标，可证明地实现正则单纯形几何。

    

    监督对比学习及其平衡变体已在长尾分布下的表示学习中展现了其有效性。然而，尽管先前的分析表明其最优表示可能呈现正则单纯形几何结构，但这种几何结构是否确实存在仍然缺乏充分的理解。在这项工作中，我们提出了一个理论框架，用于刻画这些目标函数在最优解处是否形成正则单纯形。通过将该框架应用于代表性的目标函数，我们表明这些目标函数在最优解处无法形成正则单纯形，因为它们仍然依赖于每个类别的样本数量。受这一分析的启发，我们引入了可应用于对比目标的结构平衡方法，并提出了一种完全平衡的对比目标，该目标可证明地在其最优解处形成正则单纯形几何结构。理论和实证结果均表……（摘要在此处截断）

    arXiv:2610.32039v1 Announce Type: cross  Abstract: Supervised contrastive learning and its balanced variants have demonstrated their effectiveness for representation learning under long-tailed distributions. However, although prior analyses suggest that their optimal representations may exhibit a regular simplex geometry, whether such a geometry is indeed exhibited remains insufficiently understood. In this work, we present a theoretical framework for these objectives that characterizes whether a regular simplex is formed at the optimum. By applying this to representative objectives, we show that these objectives fail to form a regular simplex at the optimum because they remain dependent on the number of samples per class. Motivated by this analysis, we introduce structural balancing that can be applied to contrastive objectives, and propose a fully balanced contrastive objective that provably forms a regular simplex geometry at its optimum. Both theoretical and empirical results demon
    
[^141]: MAGIC：面向视觉语言模型测试时鲁棒性扩展的边际门控对比去偏与证据恢复方法

    MAGIC: Margin-gated Contrastive Debiasing with Evidence Restoration for VLM Test-time Robustness Scaling

    [https://arxiv.org/abs/2610.32013](https://arxiv.org/abs/2610.32013)

    提出了MAGIC方法，通过测量关键token的决策边际并利用门控机制，仅对易受语言偏见影响的低置信度响应选择性激活文本与视觉扰动分支进行对比去偏，同时结合证据恢复，实现高效且鲁棒的视觉语言模型测试时推理。

    

    本文研究了大型视觉语言模型（LVLMs）的测试时鲁棒性扩展问题，旨在缓解推理过程中的语言偏见。现有方法通常采用对比解码等测试时扩展技术来提升推理性能。尽管取得了一定进展，但由于扰动过程中证据被刻意移除以及推理成本冗余，这些方法的性能仍远不能令人满意。为此，我们提出了一种名为边际门控对比去偏与证据恢复（MAGIC）的新方法。MAGIC 的核心在于评估 LVLMs 的决策边际，以选择性地激活文本和视觉扰动分支，从而实现高效的去偏推理。具体而言，MAGIC 首先测量关键 token 的预测边际，并应用门控函数来识别易受语言偏见影响的低置信度响应。对于被选中的……

    arXiv:2610.32013v1 Announce Type: cross  Abstract: This paper studies the problem of test-time robustness scaling for large vision-language models (LVLMs), which aims to mitigate language bias during inference. Existing approaches typically adopt test-time scaling techniques such as contrastive decoding to improve inference performance. Despite the progress, the performance of these approaches is far from satisfactory due to intentional evidence removal during perturbations and redundant inference cost. Towards this end, we propose a novel approach named Margin-gated Contrastive Debiasing with Evidence Restoration (MAGIC). The core of our MAGIC is to evaluate the decision margins of LVLMs to selectively activate textual and visual perturbation branches for efficient debiased inference. In particular, our MAGIC first measures the prediction margins of crucial tokens, and applies a gating function to identify low-confidence responses that are susceptible to language biases. For selected 
    
[^142]: GS-MoE：基于分组子空间共享与剪枝的免训练MoE压缩

    GS-MoE: Training-Free MoE Compression via Groupwise Subspace Sharing and Pruning

    [https://arxiv.org/abs/2610.32009](https://arxiv.org/abs/2610.32009)

    GS-MoE提出了一种免训练的MoE压缩框架，通过分组子空间共享的低秩表示与基于Nyström的神经元宽度剪枝，从专家间和专家内两个维度消除冗余，并利用重建误差指导专家分组与分层预算分配。

    

    混合专家模型通过稀疏激活减少计算量，但存储其专家参数仍然是主要的内存瓶颈。我们提出GS-MoE，一个免训练的压缩框架，通过两个压缩维度——共享子空间秩和专家中间宽度——来处理专家之间与专家内部的冗余。分组低秩表示在保留专家特定投影的同时捕捉专家间的共享结构，而基于Nyström的神经元压缩则降低每个专家内部的中间宽度。对于固定的共享子空间，我们从重建目标中推导出每个专家的重建误差降低量，并用其指导专家分组。预算采用分层方式分配：利用无梯度的层重要性分数指导各层之间秩与宽度预算的分配，随后由特征值和杠杆值将秩以非均匀的方式分配到各分组，并分配中间宽度。

    arXiv:2610.32009v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) models reduce computation through sparse activation, but storing their expert parameters remains a major memory bottleneck. We introduce GS-MoE, a training-free compression framework that addresses redundancy across and within experts through two compression axes: shared-subspace rank and expert intermediate width. Groupwise low-rank representations capture shared structure across experts while retaining expert-specific projections, and Nystr\"om-based neuron compression reduces intermediate width within each expert. For fixed shared subspaces, we derive per-expert reconstruction-error reductions from the reconstruction objective and use them to guide expert grouping. Budgets are allocated hierarchically using a gradient-free layer-importance score to guide the allocation of rank and width budgets across layers, which eigenvalues and leverage then distribute as non-uniform ranks across groups and intermediate wid
    
[^143]: TRACE：针对大语言模型近似遗忘的免数据文本重建攻击

    TRACE: Data-Free Text Reconstruction Attacks against Approximate Unlearning in LLMs

    [https://arxiv.org/abs/2610.31995](https://arxiv.org/abs/2610.31995)

    本文提出TRACE框架，利用遗忘前后的模型快照差异，在无需先验知识的情况下重建大语言模型中被近似遗忘的训练数据，揭示了模型快照所引入的隐私泄露侧信道风险。

    

    随着大语言模型（LLM）面临由法规合规、版权问题和隐私保护所驱动的日益增长的数据删除需求，近似遗忘已成为替代代价高昂的完全重训练的一种实用方案。在实践中，遗忘前后的模型快照通常会被保留用于审计和回滚，这引入了一个此前被忽视的侧信道。本文证明，大语言模型中的近似遗忘会在模型快照中留下残留痕迹，这些痕迹可能被恢复，从而导致近似遗忘本欲防范的隐私泄露。在此，我们提出TRACE框架，它能够在获得遗忘前后模型快照访问权限的条件下，无需任何数据特定的先验知识即可重建被遗忘的训练数据。为实现这一目标，TRACE从嵌入层级的参数差异中识别出一小部分候选词元以约束搜索空间，并组装……

    arXiv:2610.31995v1 Announce Type: cross  Abstract: As large language models (LLMs) face growing demands for data removal driven by regulatory compliance, copyright concerns, and privacy protection, approximate unlearning has emerged as a practical alternative to expensive full retraining. In practice, pre- and post-unlearning model snapshots are often maintained for auditing and rollback, introducing a previously overlooked side channel. This paper demonstrates that approximate unlearning in LLM can leave residual traces within model snapshots, which may be recoverable and thus enable the privacy breaches that approximate unlearning is intended to prevent. Here, we propose TRACE, a framework that reconstructs unlearned training data given access to pre- and post-unlearning model snapshots, without any data-specific prior knowledge. To achieve this, TRACE identifies a compact set of candidate tokens from embedding-level parameter differences to constrain the search space, and assembles 
    
[^144]: REDAR：面向类增量学习的强化驱动自适应正则化

    REDAR: Reinforcement-Driven Adaptive Regularisation for Class Incremental Learning

    [https://arxiv.org/abs/2610.31991](https://arxiv.org/abs/2610.31991)

    REDAR将类增量学习中的正则化系数选择建模为序列决策问题，利用轻量级强化学习智能体在每个增量阶段根据性能指标和数据统计自适应地输出蒸馏系数，解决了传统固定系数方法无法适应各阶段数据变化且不匹配持续学习目标的根本缺陷。

    

    基于正则化的类增量学习方法通过离线调优的固定系数来控制稳定性-可塑性权衡，这些系数独立于持续学习目标，并且无法感知各阶段在数据量、任务难度和类别相似度上的具体变化。我们认为这是一种根本性的不匹配，因此提出REDAR：面向类增量学习的强化驱动自适应正则化方法，该方法将正则化强度的选择视为一个序列决策问题。在每个增量阶段，一个轻量级的强化学习（RL）智能体观察由当前性能指标和输入数据统计信息组成的紧凑状态，并输出该阶段的蒸馏系数。为了遵守CIL对历史数据访问的限制，该策略完全在由初始阶段构建的伪CIL任务上进行训练，随后部署到增量序列上而无需进一步适应。实验……

    arXiv:2610.31991v1 Announce Type: new  Abstract: Regularisation-based class-incremental learning (CIL) methods control the stability-plasticity trade-off through fixed coefficients tuned offline, independently of the continual learning objective and blind to phase-specific variations in data volume, task difficulty, and class similarity. We argue this is a fundamental mismatch and propose REDAR: Reinforcement-Driven Adaptive Regularisation for Class Incremental Learning, which treats regularisation strength selection as a sequential decision problem. At each incremental phase, a lightweight reinforcement learning (RL) agent observes a compact state of current performance metrics and incoming data statistics, and outputs the distillation coefficients for that phase. To respect CIL's constraint on historical data access, the policy is trained entirely on pseudo-CIL tasks constructed from the initial phase, then deployed on the incremental sequence without further adaptation. Experiments 
    
[^145]: WADE：一种基于LSTM的WebAssembly应用异常检测引擎

    WADE: An LSTM-Based Anomaly Detection Engine for WebAssembly Applications

    [https://arxiv.org/abs/2610.31984](https://arxiv.org/abs/2610.31984)

    WADE 提出了一种基于 LSTM 的异常检测引擎，通过学习 WebAssembly 应用在良性条件下的正常函数级执行轨迹，无需漏洞签名即可在浏览器运行时检测已知和未知的攻击行为。

    

    WebAssembly (Wasm) 通过可移植且沙箱化的执行模型，为 Web 应用提供接近原生的运行性能。然而，其有限的运行时保护机制（例如缺乏栈保护金丝雀和地址空间布局随机化 (ASLR)）使 Wasm 应用容易遭受传统攻击。我们提出了 WADE（WebAssembly 异常检测引擎），这是一个用于检测浏览器中执行的 WebAssembly 应用异常运行时行为的系统。WADE 监控在良性条件下收集的函数级执行轨迹，并训练长短期记忆网络 (LSTM) 来学习每个应用的正常执行行为。在运行时，该模型将偏离这些已学习模式的行为标记为潜在攻击。与基于签名的方案不同，WADE 构建针对特定应用的行为模型，无需针对每个漏洞的签名即可检测已知和前所未见的攻击。我们对……（原文摘要在此处截断）

    arXiv:2610.31984v1 Announce Type: cross  Abstract: WebAssembly (Wasm) enables near-native performance for web applications through a portable and sandboxed execution model. However, its limited runtime protection mechanisms (e.g., the lack of stack canaries and address space layout randomization (ASLR)) leave Wasm applications vulnerable to traditional attacks.   We present WADE (WebAssembly Anomaly Detection Engine), a system that detects anomalous runtime behaviors in browser-executed WebAssembly applications. WADE monitors function-level execution traces collected under benign conditions and trains a Long Short-Term Memory (LSTM) network to learn the normal execution behavior of each application. At runtime, the model flags deviations from these learned patterns as potential attacks. Unlike signature-based approaches, WADE builds application-specific behavioral models, enabling detection of both known and previously unseen attacks without per-vulnerability signatures.   We evaluate 
    
[^146]: 流形上流匹配的相容似然

    Compatible Likelihoods for Flow Matching on Manifolds

    [https://arxiv.org/abs/2610.31981](https://arxiv.org/abs/2610.31981)

    该论文针对黎曼流匹配中基于似然的目标可能产生推离数据分布的速度场这一失败模式，提出“相容性”概念刻画似然目标的有效性，并进一步设计了一种基于指数族的新型目标，既满足相容性又仅支持紧流形上可行的速度。

    

    在最简单的非欧几里得流形——一维平坦环面上，我们展示了一个分布，其中最近提出的黎曼流匹配扩展方法被一个将样本推离数据分布的速度场全局优化。我们将这种失败模式追溯到一个我们称之为“相容性”的性质：如果一个基于似然的流匹配目标的总体最小化器能够恢复由数据和耦合所诱导的边际速度场，则称该目标是相容的。我们给出了相容性的一般条件，将其应用于现有方法，并通过在相容性条件下消失的残差分量来界定最坏情况误差。此外，我们注意到现有的相容目标将概率密度置于紧流形上不可行的速度上。为此，我们提出了一个基于端点上指数族构建的目标，以条件速度作为充分统计量，该目标既满足相容性，又仅支持可行的速度。

    arXiv:2610.31981v1 Announce Type: cross  Abstract: On the simplest non-Euclidean manifold, the 1D flat torus, we exhibit a distribution where a recently proposed extension of Riemannian flow matching is globally optimized by a velocity field that pushes samples away from the data distribution. We trace this failure mode to a property we call compatibility; a likelihood-based flow matching objective is compatible if its population minimizer recovers the marginal velocity field induced by the data and coupling. We give a general condition for compatibility, apply it to existing methods, and bound the worst-case error in terms of a residual component vanishing under compatibility. Next, we note that existing compatible objectives put density on velocities that are infeasible on compact manifolds. We propose an objective built from an exponential family over endpoints, with the conditional velocity as sufficient statistic, which is both compatible and supported only on feasible velocities.
    
[^147]: 超越启发式方法与商业导航：一种用于救护车行驶时间估计的AI增强决策支持系统

    Beyond Heuristics and Commercial Navigation: An AI-Augmented Decision-Support for Ambulance Travel Time Estimation

    [https://arxiv.org/abs/2610.31968](https://arxiv.org/abs/2610.31968)

    该研究基于瑞典斯科讷地区的真实救护车历史运营数据，揭示了救护车行驶时间随时段、日期和行程优先级显著变化，并开发了机器学习系统以超越传统启发式假设和商业导航系统，从而更准确地估计救护车行驶时间。

    

    准确的救护车行驶时间估计对于紧急医疗调度至关重要，然而现有的决策支持实践往往依赖于静态查找表，或采用关于救护车与普通乘用车行驶时间差异的简化假设。例如，这些假设认为救护车只是比普通乘用车快一个固定的百分比（如5%）。然而，此类假设很少经过真实救护车运营数据的验证，且可能无法捕捉行程优先级、时间背景和起讫点所带来的影响。我们利用瑞典斯科讷（Skåne）地区的历史救护车行程数据来解决这些局限性。我们的分析表明，救护车行驶时间在不同小时、不同日期和不同行程优先级之间存在显著差异，而商业导航系统无法捕捉到紧急情况特有的动态特征。我们贡献了一个机器学习系统，以更准确地估计救护车行驶时间。

    arXiv:2610.31968v1 Announce Type: new  Abstract: Accurate ambulance travel time estimation is critical for emergency medical dispatch, yet existing decision-support practices often rely on static lookup tables or simplified assumptions regarding how ambulance travel time differs from ordinary passenger-car travel. For instance, they assume ambulances simply travel a fixed percentage (e.g., 5%) faster than ordinary passenger cars. However, such assumptions are rarely validated using real ambulance operational data and may fail to capture the effects of trip priority, temporal context, and origin-destination. We address these limitations using historical ambulance trips from the Sk{\aa}ne region in Sweden. Our analysis reveals that ambulance travel times vary substantially by hour, day, and trip priority, and that commercial navigation systems fail to capture emergency-specific dynamics. We contribute a machine learning system to more accurately estimate ambulance travel time. To make th
    
[^148]: 一种用于短期建筑负荷预测的模型无关校正框架

    A Model-Agnostic Correction Framework for Short-Term Building Load Forecasting

    [https://arxiv.org/abs/2610.31959](https://arxiv.org/abs/2610.31959)

    提出了一种模型无关的天气-负荷检索增强校正框架WeL-RAC，通过在预测时基于近期负荷状态和天气预报检索相似历史案例，并聚合其观测到的未来负荷轨迹来校正初始预测，从而提升短期建筑负荷预测在工况快速变化下的可靠性。

    

    短期建筑负荷预测仍然具有挑战性，因为负荷模式会随着天气和运行条件的变化而快速改变。当当前运行状态在训练数据中罕见或代表性不足时，即使历史记录中其他地方存在类似情况，数据驱动的预测模型也可能变得不太可靠。检索增强时间序列预测的最新进展提供了一种有前景的方法，即通过在推理时直接重用相关的历史案例，使预测能够适应不断变化的条件。受这一思想的启发，我们提出了WeL-RAC，一种模型无关的天气-负荷检索增强校正框架，该框架在预测时显式地检索此类证据。WeL-RAC基于近期负荷状态和与预测时域对齐的天气预报，检索与当前条件相似的历史案例，然后聚合这些案例观测到的未来负荷轨迹，用于校正初始预测结果。

    arXiv:2610.31959v1 Announce Type: new  Abstract: Short-term building load forecasting remains challenging because load patterns can change rapidly with weather and operating conditions. Data-driven forecasting models may become less reliable when the current operating regime is rare or poorly represented during training, even when similar situations exist elsewhere in the historical record. Recent advances in retrieval-augmented time series forecasting offer a promising way to adapt predictions to changing conditions by directly reusing relevant historical cases at inference time. Motivated by this idea, we propose WeL-RAC, a model-agnostic Weather-Load Retrieval-Augmented Correction framework that explicitly retrieves such evidence at prediction time. WeL-RAC retrieves historical cases that are similar to the current condition based on both recent load states and horizon-aligned weather forecasts. Their observed future load trajectories are then aggregated and used to correct the init
    
[^149]: 雅可比透镜的数学理论

    A Mathematical Theory of the Jacobian Lens

    [https://arxiv.org/abs/2610.31955](https://arxiv.org/abs/2610.31955)

    本文提出了雅可比透镜（J-lens）的数学理论，证明该可解释性工具优先识别的是相干特征——即类似主导 Lyapunov 模态、可跨层放大、跨词元位置恒定且跨上下文一致的线性方向。

    

    Anthropic 最近的工作引入了一种新的可解释性工具，称为雅可比透镜（Jacobian lens，J-lens，Gurnee 等人，2026）。该方法在前沿语言模型的隐藏状态中识别出一些线性方向，其性质令人联想到工作记忆式的表征。然而，目前尚不清楚这些特征的哪些机制性质使得它们能够被 J-lens 检测到。为此，我们提出了 J-lens 的数学理论，并证明它优先识别的是“相干特征”——即表现得像主导的、空间相干的 Lyapunov 模态那样的方向。沿这些方向的小扰动会跨层放大，在不同词元位置之间大致保持恒定，并在不同上下文中保持一致。尽管我们对这些性质是否必然与“全局工作空间”相关联持不可知态度，但通过解析推导出的 J-lens 偏好仍然解释了先前应用于……的实证结果。

    arXiv:2610.31955v1 Announce Type: cross  Abstract: Recent work from Anthropic introduced a new interpretability tool, called the Jacobian lens (J-lens, Gurnee et al., 2026). This method identified linear directions in the hidden states of frontier language models that had properties reminiscent of working memory-like representations. However, it is unclear what mechanistic properties of such features make them detectable to the J-lens. To this end, we introduce a mathematical theory of the J-lens, and show that it preferentially identifies coherent features -- directions that act like leading, spatially coherent Lyapunov modes. Small perturbations along these directions amplify across layers, remain roughly constant across token positions, and are consistent across different contexts. While we are agnostic to whether these properties are necessarily associated with a ``Global Workspace'', the analytically derived preferences of the J-lens nevertheless explain prior empirical applied to
    
[^150]: RobustNCL：基于轻量级检测与修复策略增强瞬态故障下神经形态持续学习的鲁棒性

    RobustNCL: Robustifying Neuromorphic Continual Learning under Transient Faults with Lightweight Detect-and-Cure Strategy

    [https://arxiv.org/abs/2610.31954](https://arxiv.org/abs/2610.31954)

    本文提出RobustNCL框架，通过轻量级检测与修复策略提升神经形态持续学习在硬件瞬态故障（特别是DRAM片外存储器故障）下的可靠性。

    

    在神经网络中，持续学习（CL）是一种重要的方法，通过在运行时有效地从新的未见数据中学习，使已部署的系统能够适应动态的运行环境。最近，神经形态持续学习（NCL）概念通过利用稀疏的基于事件的运算，成为了一种节能的持续学习替代方案。然而，NCL的性能可能受到硬件平台中瞬态故障的损害，这些故障可能随时由高能粒子撞击引起。目前，NCL在瞬态故障下的可靠性尚未被研究，因此相应的容错技术仍未被探索。为解决这一问题，我们提出了一种新颖的RobustNCL框架，通过轻量级的检测与修复策略，提升硬件平台中基于DRAM的片外存储器存在瞬态故障时NCL的可靠性。具体而言，该框架首先监测权重更新过程中出现的故障……

    arXiv:2610.31954v1 Announce Type: cross  Abstract: In neural networks, continual learning (CL) is a prominent approach for enabling deployed systems to adapt to dynamic operational environments by effectively learning from new unseen data at runtime. Recently, the neuromorphic continual learning (NCL) concept has emerged as an energy-efficient alternative to CL by leveraging sparse event-based operations. However, NCL performance can be undermined by transient faults in the hardware platforms, which may come from high-energy particle strikes at any time. Currently, NCL reliability under transient faults has not been studied, and hence the corresponding fault-tolerance techniques remain unexplored. To address this, we propose a novel RobustNCL framework to improve NCL reliability in the presence of transient faults in the DRAM-based off-chip memory of hardware platforms through a lightweight detect-and-cure strategy. Specifically, it first monitors the presence of faults during weight d
    
[^151]: 独立与非负相关噪声下赢家偏差的精确界

    Sharp Bounds on the Winner's Bias under Independent and Nonnegatively Correlated Noise

    [https://arxiv.org/abs/2610.31948](https://arxiv.org/abs/2610.31948)

    本文精确确定了在独立非同分布噪声（$\sqrt{(M-1)/(2M)}$）与两两非负相关噪声（$(M-1)/M$）下，从 $M$ 个无偏含噪分数中选取最优值所导致的赢家偏差的紧致常数，给出了达到这些界的显式构造，所有定理经 Lean 4 形式化验证，并据此作为推论精确界定了交叉验证估计器的偏差。

    

    报告 $M$ 个含噪声、无偏分数中的最优值会使赢家价值被高估至多 $c\sqrt{V}$，其中 $V$ 是总噪声方差，精确常数 $c$ 取决于噪声之间的相关结构。此前该常数仅在两个极端情形下已知：任意相关（$\sqrt{(M-1)/M}$）和独立同分布噪声。本文确定了介于两者之间的两类情形：对于独立但非同分布的噪声，该常数为 $\sqrt{(M-1)/(2M)}$；对于两两非负相关的噪声，该常数为 $(M-1)/M$；对于不相关噪声，这一结果回答了关于期望极差的一个开放问题的最大值版本。文中给出了达到这两个界的显式有限构造。作为推论，交叉验证估计器的偏差在每个类别内都得到了精确界定。所有定理均在 Lean 4 中经过形式化验证。在 448 个语言模型和分类器选择流程中，采用按候选者分配条目的设计实现了独立类别上界的中位数 0.75。在固定条目预算下……

    arXiv:2610.31948v1 Announce Type: new  Abstract: Reporting the best of $M$ noisy, unbiased scores overstates the winner's value by up to $c\sqrt{V}$, where $V$ is the total noise variance and the sharp constant $c$ depends on how the noise is dependent. It was known only at the two extremes, arbitrary dependence ($\sqrt{(M-1)/M}$) and i.i.d. noise. We determine it for two classes in between: for independent, not identically distributed noise it is $\sqrt{(M-1)/(2M)}$, and for pairwise nonnegatively correlated noise it is $(M-1)/M$; for uncorrelated noise this answers the maximum version of an open question about the expected range. Explicit finite constructions attain both. As a corollary, the bias of the cross-validation estimator is bounded sharply within each class. All theorems are formally verified in Lean 4. Across 448 language-model and classifier selection pipelines, designs with per-candidate items realize a median 0.75 of the independent-class ceiling. At a fixed item budget,
    
[^152]: Kolibri：位于帕累托前沿的欧洲主权大模型

    Kolibri: A Sovereign European Model on the Pareto Frontier

    [https://arxiv.org/abs/2610.31945](https://arxiv.org/abs/2610.31945)

    Kolibri是一个总参数78.1B、激活参数仅3.46B的英德双语开源混合专家模型，通过完全自主掌控的数据与训练流程实现德语占比超20%（含2T自建德语token）的训练，在帕累托前沿上实现了面向主权化部署的高效性能。

    

    本报告介绍了Kolibri，一个英德双语的混合专家transformer模型，总参数量为78.1B，每个token激活3.46B参数，以Apache 2.0许可证开源发布。我们构建Kolibri的目的是实现主权化和专业化部署，特别关注德语以及公共管理、工业和航空航天等受监管领域。为支持这些应用场景，我们掌控了完整的模型开发流程，包括数据、架构、训练基础设施、后训练和评估。我们在24T token上对Kolibri进行训练，涵盖预训练、中期训练和长上下文扩展，其中德语按字节数计算占数据组合的20%以上。为达到这一比例，我们通过自建数据管道从网络整理并合成生成了超过2T的德语token，以补充德语数据集。后训练阶段将在精选的德语和英语数据上进行监督微调与……（摘要原文不完整）

    arXiv:2610.31945v1 Announce Type: cross  Abstract: This report introduces Kolibri, an English-German mixture-of-experts transformer with 78.1B total parameters and 3.46B active parameters per token, released as open weights under the Apache 2.0 licence. We built Kolibri for sovereign and specialised deployment, with a particular focus on German and on regulated domains such as public administration, industry, and aerospace. To support these settings, we control the full model-development process, including data, architecture, training infrastructure, post-training, and evaluation.   We train Kolibri on 24T tokens across pre-training, mid-training, and long-context extension, with German accounting for more than 20% of the data mix by bytes. To reach this share, we complement German datasets with more than 2T German tokens that we curate from the web with our own pipeline and generate synthetically. Post-training combines supervised fine-tuning on curated German and English data with re
    
[^153]: 迈向可信赖的AI赋能紧急医疗调度：不确定性感知的救护车行程时间估计

    Towards Trustworthy AI-Enabled Emergency Medical Dispatch: Uncertainty-Aware Ambulance Travel Time Estimation

    [https://arxiv.org/abs/2610.31944](https://arxiv.org/abs/2610.31944)

    该论文提出一种不确定性感知的救护车行程时间估计方法，通过量化每次估计的不确定性并处理真实数据中的领域特定异常，提升AI辅助紧急医疗调度的可信赖性。

    

    紧急医疗调度在做出救护车派遣决策时日益依赖AI赋能的决策支持系统。在此过程中，准确的救护车行程时间估计对于及时的资源配置和院前救护至关重要。然而，现有的救护车行程时间估计方法通常仅产生单一点预测。因此，这些方法无法表明单个估计结果的可靠程度或其所含不确定性的大小。这一局限在安全攸关的紧急医疗服务调度操作中问题尤为突出，因为预期行程时间相近的行程可能表现出截然不同的不确定性水平。此外，真实世界的救护车行程日志可能包含特定领域的异常数据，例如位置重叠的医疗机构以及物理上不合理的行驶速度。这些异常会降低点预测的准确性，并影响预测任务本

    arXiv:2610.31944v1 Announce Type: new  Abstract: Emergency medical dispatch increasingly relies on AI-enabled decision support systems when making ambulance dispatch decisions. In this process, accurate ambulance travel time estimates are essential for timely resource allocation and prehospital care. However, existing ambulance travel time estimation approaches typically produce only single point predictions. Consequently, they do not indicate how reliable an individual estimate is or how much uncertainty it carries. This limitation is particularly problematic in safety-critical emergency medical service dispatch operations, where trips with similar expected travel times may exhibit substantially different levels of uncertainty. Furthermore, real-world ambulance trip logs may contain domain-specific artefacts, such as co-located healthcare facilities and physically implausible travel speeds. These artefacts can degrade point prediction accuracy and affect the prediction task on which s
    
[^154]: 一致性而非归一化使循环语言模型的深度保持可复用

    Agreement rather than normalisation keeps depth reusable in looped language models

    [https://arxiv.org/abs/2610.31943](https://arxiv.org/abs/2610.31943)

    循环语言模型深度可复用的关键在于训练时与推理时循环配置的一致性，而非循环步之间的状态归一化。

    

    循环语言模型反复应用同一个层块，而这一设计只有在深度可复用时才有价值：超出训练预算的循环步数在推理时必须仍然可用。我们探讨这究竟依赖于循环步之间的状态归一化（如已发布的 Ouro 模型所采用的），还是依赖于模型训练时所用循环与运行时所执行循环之间的一致性。在已发布的权重上，这种循环边界归一化可以在不增加任何参数的情况下被移除，且读出路径保持比特级一致；该操作在单次循环步时是恒等变换，这可作为实现正确性的检验。移除该归一化后，Ouro-1.4B 和 Ouro-2.6B 在一个平衡二分类任务上（随机水平为 0.500）到训练预算处准确率降为零。此前理论所预测的失败机制——状态范数增长——并不是原因：范数始终保持在 2.021 倍以内，而输出头却持续提交同一个错误 token。然而，在两个合成任务上从零训练的 1900 万参数模型中（摘要在此处截断）

    arXiv:2610.31943v1 Announce Type: cross  Abstract: Looped language models apply one block of layers repeatedly, and the design pays off only if depth is reusable: loop steps beyond the trained budget must remain usable at inference. We ask whether that rests on normalising the state between loop steps, as the released Ouro models do, or on agreement between the loop a model is trained with and the loop it is run with. On released weights this loop-boundary normalisation can be removed without adding parameters and with the readout path bit-identical, and the operation is an identity at one loop step, which serves as an implementation check. Removing it takes Ouro-1.4B and Ouro-2.6B to zero accuracy by the trained budget on a balanced binary task (chance 0.500). State-norm growth, the failure that prior accounts predict, is not the cause: the norm stays within a factor of 2.021 while the head commits to a single wrong token. Yet in 19M-parameter models trained from scratch on two synthe
    
[^155]: 超越局部精度：PDE基础模型滚动预测可靠性的基准测试

    Beyond Local Accuracy: Benchmarking the Rollout Reliability of PDE Foundation Models

    [https://arxiv.org/abs/2610.31935](https://arxiv.org/abs/2610.31935)

    该研究提出“误差放大”指标G来基准测试PDE基础模型在自回归滚动预测中的可靠性，发现尽管微调的基础模型在短时程预测中普遍优于从零训练的基线模型，但其短时程优势未必能在长时程滚动中保持。

    

    偏微分方程（PDE）基础模型能够实现数据高效的适配，但其部署方式是自回归的，因此短时程精度并不一定意味着可靠的仿真。我们提出这样一个问题：当把基础模型方法的预测反复反馈作为输入进行长时程预测时，其短时程优势是否仍然能够保持。我们将误差放大定义为 G=E(H_f)/E(H_s)，用于衡量全时程误差相对于短时程误差的比值。我们在七个二维时变PDE数据集上评估了七个模型：从零训练的FNO和DeepONet；微调的DPOT、GPhyT、Poseidon和Walrus；以及冻结的上下文学习模型VICON。我们报告了VRMSE、相对L_2和相对MAE指标。其中六个数据集提供了配对的短时程与全时程评分，而CE-RM仅提供短时程滚动扫描。在所有指标下，至少有一个微调的基础模型在30个短时程滚动“数据集-预算”组合中的26个里优于两个从零训练的基线模型。然而，全时程VRMSE……（原文摘要在此处被截断）

    arXiv:2610.31935v1 Announce Type: new  Abstract: PDE foundation models enable data-efficient adaptation but are deployed autoregressively, so short-horizon accuracy need not imply reliable simulation. We ask whether the short-horizon advantage of a foundation-model recipe persists when its predictions are repeatedly fed back as inputs for a long horizon. We define error amplification as G=E(H_f)/E(H_s), which measures full-horizon error relative to short-horizon error. We evaluate seven models on seven two-dimensional time-dependent PDE datasets: scratch-trained FNO and DeepONet; fine-tuned DPOT, GPhyT, Poseidon, and Walrus; and frozen in-context VICON. We report VRMSE, relative L_2, and relative MAE. Six datasets provide paired short- and full-horizon scores, while CE-RM contributes short-rollout sweeps only. At least one fine-tuned foundation model outperforms both scratch baselines in 26 of 30 short-rollout dataset--budget cells under every metric. Nevertheless, full-horizon VRMSE e
    
[^156]: 每个权重都是一条共享离散决策的路径

    Every Weight Is a Path of Shared Discrete Decisions

    [https://arxiv.org/abs/2610.31931](https://arxiv.org/abs/2610.31931)

    该论文提出TWIG树结构权重表示，将神经网络中每个权重表达为共享离散决策路径的终点，从而把剪枝、量化、低秩分解和权重共享统一为对同一对象的编辑，并使权重层可转化为几乎只含导线和加法器的硬件电路。

    

    我们提出TWIG（带继承门的树结构权重），这是一种新的表示方法，其中神经网络的每一个权重都是一串微小离散决策的终点：一个符号、一次二的幂次移位、一个零门、一个指向共享表格的索引。属于一组的权重共享其路径的上部，且每个决策被其下方的所有权重继承，因此只需存储一次、应用一次。这一简单的改变带来三个后果。任何训练好的网络都可以写成TWIG形式，且当决策为二的幂次时，其权重层变成几乎完全由导线和加法器构成的电路，适合于权重在制造时即固定不变的芯片。剪枝、量化、低秩分解、码本和权重共享都成为对同一对象的编辑，因此单个优化器即可为整个网络在所有这些方法之间做出选择。而选择这些决策则等价于求解一个高阶玻尔兹曼机的基态问题……

    arXiv:2610.31931v1 Announce Type: new  Abstract: We introduce TWIG (Tree-structured Weights with Inherited Gates), a representation in which every weight of a neural network is the end of a path of small discrete decisions: a sign, a shift by a power of two, a zero gate, an index into a shared table. Weights that belong together share the upper part of their paths, and each decision is inherited by every weight below it, so it is stored once and applied once. This simple change has three consequences. Any trained network can be written as a TWIG, and when the decisions are powers of two its weight layers become circuits made almost entirely of wires and adders, suited to chips whose weights are fixed at fabrication. Pruning, quantization, low-rank factorization, codebooks and weight sharing become edits of one object, so a single optimizer can choose among all of them for a whole network. And choosing the decisions is the ground-state problem of a higher-order Boltzmann machine whose i
    
[^157]: ENCL：通过局部学习规则与子网络冻结策略实现高效神经形态持续学习

    ENCL: Enabling Efficient Neuromorphic Continual Learning with Local Learning Rule and Sub-Network Freezing Strategy

    [https://arxiv.org/abs/2610.31927](https://arxiv.org/abs/2610.31927)

    本文提出ENCL方法，通过轻量级的时空局部学习规则和子网络冻结策略，取代传统神经形态持续学习中内存消耗大的经验回放和昂贵的全局学习规则，实现了高效的持续学习。

    

    arXiv:2610.31927v1 公告类型：cross 摘要：神经网络中的持续学习（CL）是一种很有前景的学习方法，用于在运行时根据离线训练阶段未曾学习过的新数据来更新已部署系统的知识。为了实现功耗/能量高效的持续学习处理，近年来提出了基于稀疏事件操作的神经形态持续学习（NCL）概念。然而，最先进的NCL方法通常采用内存消耗大的数据/经验回放，并伴随着昂贵的全局学习规则（即基于反向传播的代理梯度）。为解决这些局限性，我们提出了ENCL，这是一种新颖的方法，通过采用轻量级局部学习规则和子网络冻结策略来实现高效的NCL。ENCL利用时间和空间上的局部学习规则为新任务的训练执行高效的信用分配，从而减轻了全局学习规则带来的高昂成本。ENCL还……（原文摘要在此处截断）

    arXiv:2610.31927v1 Announce Type: cross  Abstract: Continual learning (CL) in neural networks is a promising learning approach for updating the knowledge of deployed systems based on new data encountered at run time, that have not been learned during the offline training phase. To enable their power/energy-efficient CL processing, neuromorphic continual learning (NCL) concept with sparse event-based operations have recently been introduced. However, state-of-the-art NCL methods typically employ the memory-hungry data/experience replay, coupled with costly global learning rule (i.e., backpropagation-based surrogate gradient). To address these limitations, we propose ENCL, a novel methodology that enables efficient NCL by employing a lightweight local learning rule and sub-network freezing strategy. ENCL leverages the temporally- and spatially-local learning rule to perform efficient credit assignment for training new tasks, alleviating the expensive cost of global learning rule. ENCL al
    
[^158]: 高斯过程回归中的输入偏差校正

    Input Bias Correction in Gaussian Process Regression

    [https://arxiv.org/abs/2610.31920](https://arxiv.org/abs/2610.31920)

    提出偏差感知高斯过程（Ba-GP）框架，利用已记录GP的后验均值梯度将输入偏差和噪声传播为均值与协方差校正，并借助加性协方差结构开发数值高效的一阶校正算法，从而在获得更优输入估计时高效修正基于错误输入训练的高斯过程模型。

    

    在传统的高斯过程回归中，输入通常被假定为已知的，不确定性仅通过测量值引入。当使用高斯过程回归并利用移动传感器获取的测量值来映射标量场时，输入对应于移动传感器的估计位姿。本简报的目标是研究当更好的输入估计可用时，如何以数值高效的方式校正基于错误输入（例如来自航位推算的位姿估计）训练的已记录高斯过程。在所开发的偏差感知高斯过程（Ba-GP）框架中，利用已记录高斯过程的后验均值梯度，将估计的输入偏差和噪声传播为均值和协方差校正。随后利用所得到的加性协方差结构，开发了数值高效的一阶校正算法。仿真研究和室内光照强度映射实验表明，Ba-GP 提升了……

    arXiv:2610.31920v1 Announce Type: cross  Abstract: In traditional Gaussian process (GP) regression, the inputs are typically assumed to be known, and uncertainty enters only through the measurements. When scalar fields are mapped using GP regression with measurements obtained from a mobile sensor, the inputs correspond to the estimated pose of the mobile sensor. The goal in this letter is to understand how a recorded GP, trained on erroneous inputs (e.g., pose estimates from dead-reckoning), can be corrected in a numerically efficient way when better estimates of the inputs become available. In the developed bias-aware GP (Ba-GP) framework, the recorded GP posterior-mean gradient is used to propagate estimated input bias and noise into mean and covariance corrections. The resulting additive covariance structure is then exploited to develop numerically efficient first-order correction algorithms. A simulation study and an indoor light-intensity mapping experiment show that Ba-GP improve
    
[^159]: 延迟解码，只乘一次：重新思考量化权重的点积运算

    Decode Late, Multiply Once: Rethinking the Dot Product for Quantized Weights

    [https://arxiv.org/abs/2610.31913](https://arxiv.org/abs/2610.31913)

    该论文提出“延迟解码、只乘一次”的点积重构方法：将k比特权重编码直接用作地址偏移来选择2^k个累加器之一，最后统一进行融合的解码与乘法，从而在整数运算下精确等价的前提下，将量化权重点积的乘法器数量减少n/2^k倍，并支持任意（含非线性）解码器和矩阵/张量乘积。

    

    大型神经网络越来越多地以1-4比特权重和16或32比特激活值的形式部署，然而它们的点积运算仍然对每个元素使用一个乘法器。我们证明，一个长度为n、权重编码为k比特的点积可以用n+2^k次加法、至多2^k次解码和至多2^k次乘法来计算。每个权重编码被直接用作地址偏移量，从2^k个累加器中选择一个；每个激活值被加到所选的累加器中；累积的和只在最后于一个融合的“解码-乘法”步骤中解码并相乘一次。解码后的权重向量从头到尾都不需要被实际构建。这种重构对于整数运算是精确等价的，适用于整数或浮点操作数以及任意（包括非线性）的解码器，并可扩展到分组缩放以及矩阵和张量乘积。乘法器的数量减少为原来的1/(n/2^k)，而总运算能耗则以一个较小但有界的因子得到改善，我们对此进行了量化……

    arXiv:2610.31913v1 Announce Type: new  Abstract: Large neural networks are increasingly deployed with 1-4 bit weights and 16- or 32-bit activations, yet their dot products still use one multiplier per element. We show that a length-n dot product with k-bit weight codes can be computed with n+2^k additions and at most 2^k decodes and 2^k multiplications. Each weight code is used directly as an address offset that selects one of 2^k accumulators; each activation is added into the selected accumulator; and the accumulated sums are decoded and multiplied only once, at the end, in a fused decode-and-multiply step. The decoded weight vector is never formed. The refactoring is exact for integer arithmetic, works with integer or floating-point operands and with arbitrary, including nonlinear, decoders, and extends to group-wise scales and to matrix and tensor products. The multiplier count falls by n/2^k, while total arithmetic energy improves by a smaller, bounded factor that we quantify with
    
[^160]: 谱聚类层次结构与密度聚类层次结构的拓扑等价性

    Topological Equivalence of Spectral and Density Cluster Hierarchies

    [https://arxiv.org/abs/2610.31909](https://arxiv.org/abs/2610.31909)

    本文证明了在适当条件下，基于扩散距离的谱聚类层次结构与基于密度超水平集的密度聚类树在拓扑上等价，且该等价性可通过k-近邻图实现推广到有限样本情形。

    

    我们比较了从底层密度函数 $f$ 恢复层次聚类的两种经典方法：一种是由超水平集 $\{f\ge\lambda\}$ 的连通分支给出的密度聚类树，另一种是对由拉普拉斯算子诱导的扩散距离应用单链接方法得到的谱层次结构。我们给出了这样的条件：在适当的扩散时间下，扩散层次结构与密度聚类树在高度的单调重缩放意义下是拓扑等价的。论文进一步表明，$k$-近邻（$k$-NN）图实现通过扩散距离的一致逼近继承了这种等价性，从而实现了从谱图构造到密度水平集拓扑的有限样本归约。

    arXiv:2610.31909v1 Announce Type: new  Abstract: We compare two canonical ways to recover hierarchical clustering from an underlying density $f$: the \emph{density cluster tree} given by connected components of superlevel sets $\{f\ge\lambda\}$, and a \emph{spectral hierarchy} obtained by applying single linkage to diffusion distances induced by a Laplacian operator. We give conditions under which, at an appropriate diffusion time, the diffusion hierarchy is \emph{topologically equivalent} to the density cluster tree up to a monotone rescaling of heights. It further shows that a $k$-NN graph implementation inherits this equivalence from uniform approximation of diffusion distances, yielding a finite-sample reduction from spectral graph constructions to density-level-set topology.
    
[^161]: 基于少样本XSEM数据与混合搜索的Bosch刻蚀模拟器真实到仿真校准

    Real-to-Sim Calibration of Bosch Etching Simulators with Few-Shot XSEM Data and Hybrid Search

    [https://arxiv.org/abs/2610.31904](https://arxiv.org/abs/2610.31904)

    提出了一种少样本真实到仿真校准框架，通过将模拟器参数与工艺变量解耦，并结合局部探索、LLM引导候选生成和XGBoost的混合搜索策略，从稀疏XSEM测量中推断特定配方的Bosch刻蚀模拟器参数，实现对刻蚀深度和侧壁角度的可靠预测。

    

    Bosch工艺广泛应用于MEMS和TSV制造中，但在缺乏真实测量数据校准的情况下，准确预测刻蚀结果仍然困难，而真实测量数据稀疏且获取成本高昂。本工作提出了一个少样本真实到仿真校准框架，该框架将模拟器参数与直接工艺变量解耦，从稀疏测量数据中推断出特定配方的模拟器参数向量，从而在未见的工艺条件下实现刻蚀深度和侧壁角度的可靠预测。我们首先将Bosch轮廓预测表述为一个少样本真实到仿真校准问题，利用少量真实XSEM测量数据来识别特定配方的ViennaPS参数向量 $z_r^*$，以最小化模拟与测量的刻蚀深度和侧壁角度之间的差异。为了高效搜索该参数空间，该框架结合了局部探索、LLM引导的候选生成以及XGBoost基（摘要在此处截断）

    arXiv:2610.31904v1 Announce Type: cross  Abstract: The Bosch process is widely used in MEMS and TSV fabrication, but accurate prediction of etching outcomes remains difficult without calibration against real measurements, which are sparse and costly to obtain. This work presents a few-shot real-to-sim calibration framework that decouples simulator parameters from direct process variables to infer a recipe-specific simulator parameter vector from sparse measurements, enabling reliable prediction of etch depth and sidewall angle under held-out process conditions. We first formulate Bosch profile prediction as a few-shot real-to-sim calibration problem, where a small set of real XSEM measurements is used to identify a recipe-specific ViennaPS parameter vector $z_r^*$ that minimizes the discrepancy between simulated and measured etch depth and sidewall angle. To search this parameter space efficiently, the framework combines local exploration, LLM-guided candidate generation, and XGBoost-b
    
[^162]: 面向大模型的结构化稀疏学习：弥合模型可压缩性与高效推理之间的鸿沟

    Learning Structured Sparsity for Large Models: Bridging Model Compressibility and Efficient Inference

    [https://arxiv.org/abs/2610.31897](https://arxiv.org/abs/2610.31897)

    提出统一框架S-SALAAD，联合学习模型权重与结构化稀疏+低秩表示，使大模型既能弹性压缩以适配边缘设备的多种内存预算，又能实现真正加速的高效推理。

    

    缩放定律持续推动着模型规模的增长，而由此产生的计算与内存需求使得推理日益依赖云基础设施。然而，不稳定的网络连接、通信延迟以及隐私要求可能使这种依赖变得不切实际，从而促使人们转向在边缘设备上进行本地执行。本地部署要求模型能够适应有限的内存，并在可用的计算资源下高效运行。近期研究表明，学习“稀疏+低秩”表示能够将单个模型检查点弹性压缩至多个内存预算。然而，参数数量的减少并不一定转化为更快的推理速度。非结构化稀疏组件中的索引开销和不规则内存访问可能会抵消计算上的节省。为解决这一关键的部署障碍，我们提出了S-SALAAD，这是一个统一框架，可联合学习模型权重及其低（注：原摘要在此处被截断）

    arXiv:2610.31897v1 Announce Type: cross  Abstract: Scaling laws continue to reward increases in model scale, while the resulting computational and memory demands drive greater reliance on cloud infrastructure for inference. Yet unreliable network connectivity, communication delays, and privacy requirements can make such reliance impractical, motivating local execution on edge devices. Local deployment requires models to fit within limited memory and execute efficiently with the available computational resources. Recent work has shown that learning sparse-plus-low-rank representations enables elastic compression of a single checkpoint to multiple memory budgets. However, fewer parameters do not necessarily translate into faster inference. Indexing overhead and irregular memory access in unstructured sparse components can offset computational savings. To address this critical deployment barrier, we introduce S-SALAAD, a unified framework that jointly learns the model weights and their lo
    
[^163]: 条件分布可学习性的组合刻画

    Combinatorial Characterizations of the Learnability of Conditional Distributions

    [https://arxiv.org/abs/2610.31890](https://arxiv.org/abs/2610.31890)

    本文引入fat_γ Natarajan维数和fat_γ Graph维数两个组合维度，证明前者在每个正尺度上的有限性是条件分布函数类PAC可学习性的充要条件，并证明分布值函数类可学习当且仅当其每个坐标类均可学习。

    

    我们研究将输入映射到有限空间上概率分布的函数类的PAC可学习性。学习者观察独立同分布的输入-标签对，其中每个标签从该类中未知目标假设的条件分布中采样得到。我们通过在满足两个温和假设的有界正常损失下的超额风险来衡量学习效果。我们引入了fat_γ Natarajan维数，它同时推广了fat_γ维数和Natarajan维数，并证明其在每个正尺度上的有限性是可学习性的充要条件。我们还引入了fat_γ Graph维数，它是经典Graph维数的推广，并建立了在不同尺度上比较这两个维数的不等式，从而得到可学习性的等价刻画。最后，我们证明一个取值为分布的函数类是可学习的，当且仅当它的每个坐标类都是可学习的。

    arXiv:2610.31890v1 Announce Type: new  Abstract: We study the PAC-learnability of function classes that map inputs to probability distributions over a finite space. The learner observes i.i.d. input-label pairs, with each label sampled from the conditional distribution of an unknown target hypothesis in the class. We measure learning through excess risk under a bounded proper loss that satisfies two mild assumptions. We introduce the \textbf{fat$_\gamma$Natarajan dimension}, which generalizes both fat$_\gamma$ and Natarajan dimension, and prove that its finiteness at every positive scale is necessary and sufficient for learnability. We also introduce a \textbf{fat$_\gamma$Graph} dimension extending the classical \gra dimension and establish inequalities comparing the two dimensions at different scales, leading to an equivalent characterization of learnability. Finally, we show that a distribution-valued function class is learnable if and only if each of its coordinate classes is learna
    
[^164]: 扩展Transformer能量模型以实现高质量图像合成

    Scaling Transformer Energy-based Models For High Quality Image Synthesis

    [https://arxiv.org/abs/2610.31889](https://arxiv.org/abs/2610.31889)

    本文提出Transformer能量基模型（TEM），通过在潜空间中运行的可扩展transformer能量函数、多层能量组合以及正表示对齐（PRA）等技术，克服了能量基模型在高维采样与深度架构训练中的瓶颈，实现了高质量的大规模图像合成。

    

    能量基模型是一类灵活的生成模型，它将概率建模与生成过程解耦。这种解耦直接使组合式生成和推理时扩展成为可能，即只需增加采样步数即可提升样本质量。然而，高维采样中的严峻挑战以及深度架构下的训练不稳定性，一直将能量基模型限制在低维场景中。为解决这些瓶颈，我们提出了Transformer能量基模型（TEM），这是一种完全在潜空间中运行的可扩展transformer能量函数。在此骨干网络之上，我们从多个层提取能量并将其求和，从而组合了从局部纹理到全局语义的互补约束。此外，两个组件使其能够大规模实现高质量生成：正表示对齐，它利用视觉基础模型对TEM的中间表示进行监督……

    arXiv:2610.31889v1 Announce Type: new  Abstract: Energy-Based Models (EBMs) are a flexible class of generative models that decouple probabilistic modeling from the generation process. This decoupling directly enables compositional generation and inference-time scaling, where sample quality can be refined simply by increasing sampling steps. However, severe challenges in high-dimensional sampling and training instabilities with deep architectures have restricted EBMs to low-dimensional regimes. To resolve these bottlenecks, we introduce the Transformer Energy-Based Model (TEM), a scalable transformer energy function operating entirely within a latent space. On top of this backbone, we extract energies at multiple layers and sum them, composing complementary constraints from local texture to global semantics. Furthermore, two components make this high-quality at scale: Positive Representation Alignment (PRA), which supervises TEM's intermediate representations with a vision foundation mo
    
[^165]: 课程学习的有效性取决于奖励稀疏性与竞争最优解

    Effectiveness of Curriculum Learning Depends on Reward Sparsity and Competing Optima

    [https://arxiv.org/abs/2610.31884](https://arxiv.org/abs/2610.31884)

    通过构建并分析策略学习的最小化模型，该研究证明了课程学习的有效性取决于奖励的稀疏程度：在稠密奖励下课程仅有边际收益，而在稀疏奖励下课程可能是学习得以发生的必要条件。

    

    尽管动物和人类在先训练较简单版本的任务时，往往能更快、更可靠地学会该任务，但课程学习在人工环境中的有效性却各不相同，且其在强化学习中的效果似乎显著大于监督学习。在这项工作中，我们构建并分析了一个策略学习的最小化模型，以更好地理解其中的原因。我们考虑一个开环、回合制的目标拦截任务，其参数——尤其是智能体必须距离目标多近才能获得有信息量的奖励信号——可以被设定为使智能体频繁获得有信息量的奖励（“稠密”奖励）或很少获得（“稀疏”奖励）。我们对两种情况下基于REINFORCE的策略学习动力学进行了数学和数值分析。在稠密奖励的情况下，课程只能带来边际收益；而在稀疏奖励的情况下，课程可能是学习得以发生的必要条件。我们发现，这一现象（原文摘要在此处被截断）

    arXiv:2610.31884v1 Announce Type: cross  Abstract: While animals and people tend to learn a task more quickly and reliably when they first train on simpler versions of it, curriculum learning's effectiveness in artificial settings varies, and appears substantially greater in reinforcement learning than in supervised learning. In this work, we construct and analyze a minimal model of policy learning to better understand why. We consider an open-loop, episodic target interception task whose parameters---especially how close the agent must be to the target to receive an informative reward signal---can be chosen so that the agent either receives informative rewards frequently ('dense' rewards) or rarely ('sparse' rewards). We mathematically and numerically analyze the dynamics of REINFORCE-based policy learning in both cases. In the dense reward case, curricula have only marginal benefits; in the sparse case, curricula can be required for learning to occur at all. We find that this is espe
    
[^166]: 预测的是关联，而非干预：协变量感知时间序列基础模型的预注册因果审计

    Forecasting Associations, Not Interventions: A Preregistered Causal Audit of Covariate-Aware Time Series Foundation Models

    [https://arxiv.org/abs/2610.31869](https://arxiv.org/abs/2610.31869)

    该研究通过理论与预注册实验证明，协变量感知的时间序列基础模型（如Chronos-2、TimesFM-2.5等）对“假如”式干预查询给出的只是关联性预测而非真实因果效应，并推导出闭式的无参数混淆差距参考线，实验显示这些模型比回归本身更严重地受混淆影响。

    

    协变量感知的时间序列基础模型（TSFM）接受控制变量的未来路径并返回预测结果，这自然引发了“假如……会怎样”式的查询。然而这类查询本质上是干预性的，而这些模型却是在观测数据上训练的。我们在三个层面判定这些模型的回答究竟是条件斜率 ∂E[Y|X]（关联性）还是干预效应 ∂E[Y|do(X)]。理论层面：贝叶斯最优的观测性预测器只有在能够获取并使用调整集的情况下，才会以干预性方式回答干预查询，否则其回答仍是关联性的；针对上下文内最小二乘法，我们以闭式形式推导出由此产生的混淆差距——即模型在观测性情境与随机化情境下效应估计之差——这是一条无参数的参考线。行为层面：在一个预注册、且具有精确反事实真值的实验网格上，Chronos-2、TiRex-2、TabPFN-TS 和 TimesFM-2.5 全部贴合这条参考线（R² 为 0.90–1.00），并且比回归本身表现出更强的混淆响应（斜率为 1.3–1.8）……

    arXiv:2610.31869v1 Announce Type: new  Abstract: Covariate-aware time series foundation models (TSFMs) accept a future path for a control variable and return a forecast, inviting what-if queries. The query is interventional; the models are trained on observational data. We determine whether their answers are the conditional slope $\partial E[Y|X]$ or the interventional effect $\partial E[Y|do(X)]$, at three levels. Theory: a Bayes-optimal observational predictor answers interventional queries associationally unless the adjustment set is available and used; for in-context least squares we derive the resulting confounding gap -- a model's effect estimate under an observational minus under a randomized context -- in closed form, a parameter-free reference line. Behaviour: on a preregistered grid with exact counterfactual ground truth, Chronos-2, TiRex-2, TabPFN-TS and TimesFM-2.5 all track this line ($R^2$ 0.90-1.00) and respond more confounded than regression itself (slopes 1.3-1.8), on 
    
[^167]: 在计算与数据约束下重写合成数据的缩放定律

    Rewriting Scaling Laws for Synthetic Data with Compute and Data Constraints

    [https://arxiv.org/abs/2610.31867](https://arxiv.org/abs/2610.31867)

    本文提出了首个面向合成重写数据的缩放定律，将生成重写的计算成本纳入建模，发现最优重写数据比例随计算预算的增加而显著上升。

    

    合成重写是指由语言模型对自然文档进行重写（例如转换为目标风格），这如今已成为预训练语料库的标准组成部分。然而，其使用中的若干关键问题仍知之甚少，例如：应该花费多少计算量来重写训练数据，以及在重写数据上训练出的模型能否超越重写器本身。我们通过提出一个重写缩放定律来研究这两个问题，该定律根据模型大小、训练词元数量、训练词元中重写数据所占比例以及在自然数据上的训练轮数来预测损失，并将生成重写所花费的计算量计入总成本。我们在两种设置下拟合了该缩放定律。在计算受限的设置中，自然数据充足，重写数据来自一个用比我们最大规模训练运行更多计算量训练的第三方重写器。我们发现，估计的最优重写比例随计算量的增加而上升，从3×10¹⁸ FLOPs时的不到10%上升到10²²时的32%。

    arXiv:2610.31867v1 Announce Type: cross  Abstract: Synthetic rewrites, where a language model rewrites a natural document e.g. into a target style, are now a standard ingredient of pretraining corpora. However, critical aspects of their use remain poorly understood, such as how much compute should be spent rewriting training data and whether a model trained on rewrites can outperform the rewriter. We study these two questions by proposing a rewriting scaling law that predicts loss from model size, training tokens, the fraction of training tokens that are rewrites, and the number of epochs over the natural data, charging the compute spent generating rewrites. We fit this scaling law in two settings. In the compute-limited setting, natural data is plentiful and rewrites come from a third-party rewriter trained with more compute than our largest run. We find that the estimated optimal rewrite fraction increases with compute, from under $10\%$ at $3\times10^{18}$ FLOPs to $32\%$ at $10^{22
    
[^168]: 基于人类反馈的大语言模型对齐：数学基础与算法设计

    Aligning Large Language Models with Human Feedback: Mathematical Foundations and Algorithm Design

    [https://arxiv.org/abs/2610.31862](https://arxiv.org/abs/2610.31862)

    本文系统介绍了大语言模型对齐的数学基础与算法框架，涵盖SFT、RLHF和DPO等主流技术，并将其理论与逆强化学习和离散选择模型建立联系，为开发更稳健、可扩展的对齐方法提供了全面参考。

    

    本文介绍了用于使大语言模型（LLM）与人类意图、偏好和价值观对齐的数学基础与算法框架。我们讨论了标准的对齐技术，如监督微调（SFT）、基于人类反馈的强化学习（RLHF）和直接偏好优化（DPO）。我们还探讨了从人类偏好中学习的理论基础，并将其与逆强化学习（IRL）和离散选择模型建立了联系。我们以教程的形式介绍了最先进的算法，讨论了它们的优势与局限性，并为实际应用提供了见解。本文旨在为研究人员和从业者提供一份全面的参考资料，既包含对对齐方法论的基础理解，也为开发更稳健、更可扩展的对齐技术提供了框架。

    arXiv:2610.31862v1 Announce Type: cross  Abstract: This article provides an introduction to the mathematical foundations and algorithmic frameworks used to align Large Language Models (LLMs) with human intentions, preferences, and values. We discuss standard alignment techniques, such as fine-tuning (SFT), reinforcement learning with human feedback (RLHF), and direct preference optimization (DPO). We also explore the theoretical underpinnings of learning from human preferences, drawing connections to inverse reinforcement learning (IRL) and discrete choice models. We present state-of-the-art algorithms in a tutorial style, discuss their advantages and limitations, and offer insights into practical implementation. Our exposition is intended to serve as a comprehensive resource for researchers and practitioners, providing both a foundational understanding of alignment methodologies and a framework for developing more robust and scalable alignment techniques.
    
[^169]: 谨慎无冲突逆梯度方法

    Cautious Conflict-Free Inverse Gradient Methods

    [https://arxiv.org/abs/2610.31855](https://arxiv.org/abs/2610.31855)

    针对ConFIG方法与Adam等自适应优化器结合后权重更新可能丧失无冲突特性的漏洞，提出了仅需三行代码扩展的谨慎ConFIG（Cautious ConFIG）方法来保障PINN训练中更新的无冲突性。

    

    偏微分方程（PDE）是建模物理和工程中复杂现象不可或缺的工具。物理信息神经网络（PINNs）通过将物理约束直接纳入神经网络的损失函数，已成为传统数值求解器的强大替代方案。尽管具有诸多优势，PINN训练需要多目标优化，这常常导致边界、初始和残差损失项之间出现显著的梯度冲突。无冲突逆梯度（ConFIG）方法能够解决这些冲突；然而，将ConFIG与Adam等自适应优化器相结合时引入了一个关键漏洞：ConFIG生成的无冲突方向会被优化器进一步处理，由此产生的权重更新相对于任务梯度而言可能不再保持无冲突特性。为了解决这一问题，本文提出了谨慎ConFIG（Cautious ConFIG），作为一种简洁的三行扩展开头被截断了。

    arXiv:2610.31855v1 Announce Type: new  Abstract: Partial differential equations (PDEs) are indispensable tools for modeling complex phenomena in physics and engineering. Physics-Informed Neural Networks (PINNs) have emerged as a powerful alternative to traditional numerical solvers by directly incorporating physical constraints into neural network loss functions. Despite their advantages, PINN training requires multi-objective optimization, which often leads to significant gradient conflicts among boundary, initial, and residual loss terms. The Conflict-Free Inverse Gradient (ConFIG) method addresses these conflicts; however, integrating ConFIG with adaptive optimizers such as Adam introduces a critical vulnerability: the conflict-free direction generated by ConFIG is further processed by the optimizer, and the resulting weight update may no longer be conflict-free with respect to the task gradients. To address this issue, Cautious ConFIG is proposed as a concise three-line extension t
    
[^170]: 重新思考面向城市表示学习的多视图融合

    Rethinking Multi-View Fusion for Urban Representation Learning

    [https://arxiv.org/abs/2610.31850](https://arxiv.org/abs/2610.31850)

    本文实证揭示城市多视图融合中模态效用区域差异与多目标梯度不平衡两大结构性问题，并提出自适应稀疏MoE路由的自调节框架PRISM来学习区域特定的视图分配，实现更稳健的城市表示学习。

    

    城市区域表示学习将异构的城市信号（如人流移动、兴趣点和土地利用模式）编码为统一的嵌入表示，服务于下游任务，包括签到预测、犯罪预测和服务需求估计。现有方法依赖于全局共享的融合与联合优化，隐式假设各模态重要性相同且多目标训练天然平衡。然而，城市数据本质上是异构的。我们的实证分析揭示了两个结构性问题：（i）模态效用在区域间差异显著，导致静态融合过度强调某些模态的信号；（ii）异构目标之间存在持续的梯度不平衡，造成表示偏差和训练不稳定。我们提出了 PRISM，一个面向自适应多视图学习的自调节框架。PRISM 采用自适应的稀疏 MoE 路由来学习区域特定的视图分配，……（摘要在此处被截断）

    arXiv:2610.31850v1 Announce Type: cross  Abstract: Urban region representation learning encodes heterogeneous urban signals, such as mobility flows, points of interest, and land-use patterns, into unified embeddings for downstream tasks, including check-in prediction, crime forecasting, and service demand estimation. Existing approaches rely on globally shared fusion and joint optimization, implicitly assuming uniform modality importance and naturally balanced multi-objective training. However, urban data is inherently heterogeneous. Our empirical analysis reveals two structural issues: (i) modality utility varies markedly across regions, causing static fusion to overemphasize signals from certain modalities; and (ii) heterogeneous objectives exhibit persistent gradient imbalance, causing representational bias and unstable training. We present PRISM, a self-adjusting framework for adaptive multi-view learning. PRISM adapts sparse MoE routing to learn region-specific view allocations, w
    
[^171]: 不要忘记你刚学到的东西：自我导向修复将智能体的错误转化为持久教训

    Don't Forget What You Just Learned: Self-Directed Repair Turns Agent Mistakes Into Lasting Lessons

    [https://arxiv.org/abs/2610.31846](https://arxiv.org/abs/2610.31846)

    提出Self-Direction机制，让智能体在任务失败后自我反思提取教训，并立即蒸馏为轻量级低秩适配器（仅占基础模型约0.02%参数量）存入持久化库中，从而将错误转化为不遗忘、不破坏原有能力的持久化学习成果。

    

    智能体应当从错误中学习，从而使经验带来持续的改进。然而，当今的智能体从错误中恢复只是暂时的：上下文内的反思容易在会话之间被遗忘，并且会争夺有限的上下文空间，而全参数微调成本高昂，且可能干扰已有的能力。我们提出了Self-Direction（自我导向），这是一种让智能体从错误中学习的机制，它结合了临时性的上下文内重试与策略更新两者的优点。当智能体在任务中失败时，它会反思自己的行动轨迹以提取出一条教训，并立即将其提炼为一个轻量级的低秩适配器。通过保持原始权重冻结，Self-Direction防止了适配器训练对基础模型参数的覆盖，同时每个适配器仅需训练约相当于基础模型参数量0.02%的额外参数。随后，智能体会将这些适配器保存到一个持久化的库中。

    arXiv:2610.31846v1 Announce Type: new  Abstract: Intelligent agents should learn from their mistakes so that experience leads to continuous improvement. Yet today's agents recover from a mistake only temporarily: in-context reflections are easily forgotten across sessions and compete for limited context, while full-parameter fine-tuning is expensive and can interfere with existing capabilities. We introduce Self-Direction, a mechanism for agents to learn from mistakes that combines the benefits of temporary in-context retries and policy updates. When an agent fails a task, it reflects on its own trajectory in order to extract a lesson and immediately distills it into a lightweight low-rank adapter. By keeping the original weights frozen, Self-Direction prevents adapter training from overwriting base-model parameters while training additional parameters equivalent to ~0.02% of the base model's parameter count for each adapter. The agent then saves these adapters in a persistent library 
    
[^172]: SEVI：安全的证据式真实世界强化学习

    SEVI: Safe Evidential Real-World Reinforcement Learning

    [https://arxiv.org/abs/2610.31829](https://arxiv.org/abs/2610.31829)

    SEVI提出证据式安全评论家框架，在单次前向传播中同时估计约束违反风险与认知不确定性，并在真实世界策略学习中对不安全或证据不足的动作进行离线恢复策略干预，从而实现安全的真实世界强化学习。

    

    真实世界强化学习已经使富接触操作任务能够直接在物理机器人上进行训练，但在在线学习过程中的不安全交互仍然是实际部署的关键障碍。我们提出了安全证据式真实世界强化学习，这是一个在真实世界策略学习过程中进行可靠性感知安全干预的框架。SEVI引入了一个证据式安全评论家，能够在单次前向传播中同时估计未来的约束违反风险和认知不确定性。在在线学习期间，SEVI会在执行前评估任务策略的每个动作；如果预测该动作不安全，或该预测缺乏足够的证据支持，SEVI将用一个离线训练的恢复策略覆盖该动作，引导机器人前往风险更低的区域。我们在分布内和分布偏移条件下，于两个模拟操作任务和一个真实世界的USB插入任务上对SEVI进行了评估。

    arXiv:2610.31829v1 Announce Type: cross  Abstract: Real-world reinforcement learning has enabled contact-rich manipulation tasks to be trained directly on physical robots, but unsafe interactions during online learning remain a key barrier to practical deployment. We propose Safe Evidential Real-World Reinforcement Learning (SEVI), a framework for reliability-aware safety intervention during real-world policy learning. SEVI introduces an Evidential Safety Critic that estimates both future constraint-violation risk and epistemic uncertainty in a single forward pass. During online learning, SEVI evaluates each task-policy action before execution; if the action is predicted to be unsafe or the prediction is insufficiently supported by evidence, SEVI overrides it with an offline-trained recovery policy that guides the robot toward a lower-risk region. We evaluate SEVI on two simulated manipulation tasks and a real-world USB insertion task under in-distribution and distribution-shifted cond
    
[^173]: 投机扩散：两种权衡的故事

    Speculative Diffusion is a Tale of Two Tradeoffs

    [https://arxiv.org/abs/2610.31819](https://arxiv.org/abs/2610.31819)

    本文揭示了投机扩散中存在的两种质量与速度的权衡：扩散模型可通过牺牲输出质量换取更快的采样速度，而投机解码也可通过更宽松的接受标准推广为有损方式以加速生成。

    

    投机解码是一种通过从更快、质量较低的草稿模型中采样来加速自回归大语言模型（LLM）推理速度的技术。传统上，投机解码的一个基本特征是它相对于验证器分布是无损的，即在保证速度的同时不牺牲质量。这与最近开发的扩散大语言模型（dLLM）不同，后者能够通过降低输出质量、减少扩散计算量来实现显著的加速。由于扩散模型可以通过并行生成token实现比自回归模型更快的采样，它们被认为是投机扩散中草稿模型的有力候选者。虽然人们普遍认为扩散模型可以为速度牺牲质量，但投机解码本身也可以被推广，通过更宽松的接受标准更快地生成有损样本……

    arXiv:2610.31819v1 Announce Type: new  Abstract: Speculative decoding is a technique that accelerates the inference speed of autoregressive large language models (LLMs) by sampling from a faster, lower quality draft model. Traditionally, a fundamental characteristic of speculative decoding is that it is lossless with respect to the verifier distribution, guaranteeing speed without sacrificing quality. This differs from recently developed diffusion LLMs (dLLMs), which are notably able to achieve substantial speedups by degrading the quality of their outputs with less diffusion effort. Because diffusion models are able to sample faster than autoregressive models by generating tokens in parallel, they are considered strong candidates for draft models in speculative diffusion. While it is widely understood that diffusion models can sacrifice quality for speed, speculative decoding itself can also be generalized to generate lossy samples faster using more lenient acceptance criteria with re
    
[^174]: 面向可扩展预训练与持续学习的嵌套稀疏性

    Nested Sparsity for Scalable Pretraining and Continual Learning

    [https://arxiv.org/abs/2610.31818](https://arxiv.org/abs/2610.31818)

    本文提出 Nestor，一种推广注意力算子的嵌套稀疏机制，可与 MoE 稀疏性层级组合，在相同计算量和稀疏度下预训练性能优于传统 MoE transformer，并能更有效地进行持续预训练和适配更长上下文。

    

    arXiv:2610.31818v1 公告类型：新 摘要：混合专家transformer是现代大规模深度学习的标准骨干架构，其采用条件计算：每次前向传播中仅使用模型权重的一个稀疏子集。在 MoE transformer 中，这种稀疏性来自 MoE 算子，该算子是对经典 transformer 中多层感知机的推广。在本工作中，我们探索了一种替代性的条件计算机制，它以类似的方式推广了注意力算子。由该机制诱导的稀疏性能够与 MoE 算子诱导的稀疏性自然地嵌套（即进行层级组合）。我们展示了采用这种嵌套稀疏性机制的 transformer（我们将其命名为 Nestor），在相同计算量和相同稀疏度下，在语言模型预训练中的表现优于传统的 MoE transformer。我们还展示了 Nestor 非常适合持续预训练：相比传统方法，它能够更有效地适配更长的上下文。

    arXiv:2610.31818v1 Announce Type: new  Abstract: Mixture-of-experts (MoE) transformers, the standard backbone for modern large-scale deep learning, employ conditional computation: only a sparse subset of model weights is used in each forward pass. In MoE transformers, this sparsity arises in the MoE operator, which generalizes the multi-layer perceptron in the classical transformer. In this work, we explore an alternative mechanism for conditional computation that analogously generalizes the attention operator. The sparsity induced by this mechanism naturally nests (i.e., hierarchically composes) with the sparsity induced by the MoE operator. We show that a transformer employing this nested sparsity mechanism, which we call Nestor, performs better than a conventional MoE transformer at equal compute and equal sparsity in language model pretraining. We also show that Nestor is well-suited for continual pretraining: it can be adapted to longer contexts more effectively than conventional 
    
[^175]: CuraLoc：通过计划式代码检查实现仓库级缺陷定位

    CuraLoc: Repository-Level Fault Localization via Planned Code Inspection

    [https://arxiv.org/abs/2610.31809](https://arxiv.org/abs/2610.31809)

    CuraLoc提出在源码检索前一次性规划代码检查的缺陷定位方法：模型在单次规划中同时选定起始符号、检查关系并形成修改假设，再由确定性检索完成最终排序，从而减少反复的模型调用和token消耗。

    

    缺陷报告描述了出了什么问题，但很少指出代码应该在哪里修改。现有的仓库级定位方法通常让语言模型迭代地决定下一步要检查哪些代码，这需要多次模型调用并消耗大量token。我们研究了这些检查决策是否可以在源码检索之前一次性规划完成。在本研究中，我们提出了CuraLoc，它首先筛选候选文件，并使用简洁的行为与结构描述来表示文件中的符号。在单次规划决策中，模型联合完成三项任务：选择起始符号、确定从每个符号出发要检查的代码关系，并对可能的代码修改位置形成假设。随后由确定性检索收集所请求的源码，用于基于这些假设进行的最终排序，从而将模型的推理集中在选择和比较代码上，而不是指导每一步检索。我们在…上验证了所提方法的有效性

    arXiv:2610.31809v1 Announce Type: cross  Abstract: A bug report describes what went wrong, but rarely where the code should change. Existing repository-level localization methods often let language models iteratively decide what code to inspect next, requiring repeated model calls and substantial token usage. We investigate whether these inspection decisions can instead be planned before source retrieval. In this research, we introduce CuraLoc, which first selects candidate files and represents their symbols using compact behavioral and structural descriptions. In a single planning decision, the model jointly selects starting symbols, chooses code relationships to inspect from each, and forms hypotheses about likely edits. Deterministic retrieval collects the requested source for a final ranking informed by those hypotheses, concentrating model reasoning on choosing and comparing code rather than directing each retrieval step. We demonstrate the efficacy of our proposed approach on the
    
[^176]: PaperScorer：论文录用有多可预测？

    PaperScorer: How Predictable Is Paper Acceptance?

    [https://arxiv.org/abs/2610.31791](https://arxiv.org/abs/2610.31791)

    该论文提出PaperScorer系列模型，仅凭论文文本或页面截图即可预测同行评审的录用或拒稿结果，其中视觉模型表现最佳（ICLR论文68%、arXiv机器学习论文77%准确率），表明论文的图表、布局与呈现方式蕴含着重要的评审决策信号。

    

    自动化研究在很大程度上通过基准测试性能来评估，但研究的价值很少能简化为一个单一的分数。人类通过同行评审来判断一项工作的整体贡献，而这难以建模，因为它充满噪声且依赖于众多因素。然而，如果仅凭论文本身，我们就能直接预测其同行评审结果呢？我们提出了PaperScorer，这是一系列模型，可以从论文提取的文本或页面截图中预测论文的录用或拒稿决定，训练时仅使用决策结果，并去除了作者身份和格式上的捷径。视觉模型最具预测性（在ICLR论文上达到68%的准确率，在arXiv的ML论文上达到77%），超越了纯文本模型和前沿推理模型。与纯文本不同，页面图像保留了图表、布局和呈现方式。这一信号是稳健的，在更小的模型上依然有效，甚至可以将整篇论文压缩为单张图像。但准确的预测并非均匀分布……

    arXiv:2610.31791v1 Announce Type: new  Abstract: Automated research is largely evaluated by benchmark performance, but the value of research rarely reduces to a single score. Humans judge a work's holistic contribution through peer review, which is hard to model because it is noisy and depends on many factors. However, what if, given only the paper, we could predict its peer-review decision directly? We introduce PaperScorer, a family of models that predict a paper's accept or reject decision from its extracted text or page screenshots, training only on the decision after removing author identity and formatting shortcuts. Vision models are the most predictive (68% accuracy on ICLR papers and 77% on ML papers from arXiv), beating text-only and frontier reasoning models. Unlike plain text, page images retain figures, layout, and presentation. This signal is robust, surviving smaller models and even compressing the whole paper into a single image. But accurate predictions are not evenly s
    
[^177]: 并非所有保留客户端都是平等的：联邦遗忘中的客户端选择

    Not All Retained Clients Are Equal: Client Selection for Federated Unlearning

    [https://arxiv.org/abs/2610.31786](https://arxiv.org/abs/2610.31786)

    提出PDF框架，通过在训练期间维护路径依赖、CLP加权的轻量级客户端历史信息，在联邦遗忘时识别并排序最有价值的保留客户端，而非盲目依赖所有保留客户端，从而更高效地移除目标客户端的影响。

    

    联邦遗忘旨在从已训练的模型中移除指定目标客户端的影响，而无需承担从头重新训练的过高成本。现有方法大多将遗忘视为一种事后、无状态的操作，忽略了客户端影响是路径依赖的：训练期间（尤其是在关键学习期间，CLP）所做的贡献可能产生不成比例且持久的影响。此外，知识重叠和客户端的间歇性可用性使得依赖所有保留客户端既低效又不切实际。我们提出了PDF，一个路径依赖的客户端选择框架，其基于这样一个洞察：保留客户端对遗忘任务的价值并不相等。在训练期间，PDF维护轻量级、与目标无关的、经CLP加权的历史记录，总结每个客户端累积的更新方向以及与全局模型的对齐程度。在收到遗忘请求时，这些历史记录会识别并排序保留客户端。

    arXiv:2610.31786v1 Announce Type: new  Abstract: Federated unlearning (FU) aims to remove a designated target client's influence from a trained model without the prohibitive cost of retraining from scratch. Existing methods largely treat unlearning as a post-hoc, stateless operation, overlooking that client influence is path-dependent: contributions made during training, particularly during critical learning periods (CLPs), can have disproportionate and persistent effects. Moreover, knowledge overlap and intermittent client availability make relying on all retained clients inefficient and impractical. We introduce PDF, a path-dependent client-selection framework based on the insight that retained clients are not equally valuable for unlearning. During training, PDF maintains lightweight, target-agnostic, CLP-weighted histories summarizing each client's accumulated update direction and global-model alignment. Upon an unlearning request, these histories identify and rank retained clients
    
[^178]: 大型残差网络中的早期学习

    Early learning in large residual networks

    [https://arxiv.org/abs/2610.31783](https://arxiv.org/abs/2610.31783)

    该论文通过图表计算方法分析了大型残差网络在早期训练阶段的动态，发现在大输入维度极限下，网络在训练初期只能学习到输入的线性函数，无法学习更高阶的单项式。

    

    我们研究了由零隐藏层块构成的非线性残差网络的早期训练动态，并将其作为更一般残差网络的模型。我们在一个顺序极限下分析网络：首先让宽度和深度发散，然后让训练数据点数量发散，最后让输入维度发散。与典型的平均场理论分析（其中深度和宽度发散，但训练数据点数量和输入维度保持固定）相比，这是一个更精细的设置。我们的主要技术成果是一种图表计算方法，用训练集中输入内积的统计量来刻画梯度下降几步之后的网络输出和隐藏表示。我们的一个主要概念性发现是一个负面结果：在大输入维度极限下，网络输出在训练早期只能学习其输入的线性函数，因为更高阶的学习单项式……（摘要原文在此处被截断）

    arXiv:2610.31783v1 Announce Type: new  Abstract: We study the early training dynamics of nonlinear residual networks with blocks of zero hidden layers as a model of more general residual networks. We analyze networks in a sequential limit where first width and depth diverge, then the number of training data points diverges, and finally the input dimension diverges. This is a more refined setup compared to the typical mean-field theory analysis in which depth and width diverge but the number of training datapoints and input dimension are fixed. Our main technical result is a diagrammatic computation of both the network output and the hidden representations after the few steps of gradient descent in terms of the statistics of inner products of inputs from the training set. One of our principle conceptual findings is a negative result: in the limit of large input dimension the network output can learn only a linear function of its inputs early in training since higher learned monomials ar
    
[^179]: 基准测试出人意料地对上下文压缩友好

    Benchmarks Are Surprisingly Compaction-Friendly

    [https://arxiv.org/abs/2610.31778](https://arxiv.org/abs/2610.31778)

    该论文首次在固定的智能体框架内实现五种上下文压缩策略，并在三个长时程基准上系统评估三个前沿模型，发现基准任务对有损压缩出人意料地友好——对百万级上下文模型而言，压缩阈值几乎不影响任务性能。

    

    arXiv:2610.31778v1 公告类型： cross 摘要：长时程智能体任务目前正在将大语言模型（LLM）智能体推向超出其上下文窗口的范围。为了继续执行任务，智能体框架必须释放上下文窗口空间，这一过程被称为上下文压缩。压缩在设计上通常是有损的：智能体可能会悄悄丢失任务约束、指令，或者其早期决策背后的推理依据。尽管压缩已是编程智能体框架中的标准组件，但此前尚未有研究针对前沿模型在真实的长时程智能体任务上对压缩进行系统性研究。以往的工作仅在玩具型或合成的轨迹上评估较小的开源权重模型。我们固定了智能体框架，在其中实现了五种压缩策略，并在三个长时程基准上评估了三个前沿模型。对于两个具备100万上下文的模型，压缩阈值对任务性能没有可测量的影响，而对于上下文窗口最小的GPT-5.3-Codex……

    arXiv:2610.31778v1 Announce Type: cross  Abstract: Long horizon agentic tasks are now pushing LLM agents past their context window. To continue the task, the agent scaffold has to free up the context window, a process known as context compaction. Compaction is generally lossy by design: agents can silently lose task constraints, instructions, or their reasoning behind earlier decisions. Although compaction is a standard component of coding agent scaffolds, it has not been studied systematically with frontier models on realistic, long horizon agentic tasks. Previous work evaluate smaller open weight models on toy or synthetic trajectories. We fix the agent scaffold (OpenCode), implement five compaction strategies inside it, and evaluate three frontier models on three long horizon benchmarks (PaperBench, ProgramBench, PostTrainBench). For the two 1M context models, the compaction threshold does not measurably affect the task performance, whereas GPT-5.3-Codex, the model with the smallest
    
[^180]: 通过逐查询模态选择实现高效的全模态大语言模型推理

    Efficient Omni-modal Large Language Model Inference via Per-Query Modality Selection

    [https://arxiv.org/abs/2610.31768](https://arxiv.org/abs/2610.31768)

    提出一种逐查询自适应模态选择方法，在不改动全模态大语言模型本身的情况下，通过成本敏感学习为每个查询选择最优模态组合，持续改善精度与推理成本之间的权衡。

    

    全模态大语言模型（Omni-LLM）能够联合处理文本、音频和视频等多种模态，实现更强的多模态理解与推理能力，但往往伴随着高昂的推理成本。我们观察到，额外模态带来的收益因查询而异，这表明固定的模态配置可能效率低下。基于此，我们研究了逐查询的模态选择方法，即在保持底层全模态大语言模型不变的情况下，自适应地为每个查询选择需要提供的模态。我们将该问题形式化为精度-成本优化问题，并开发了一种用于模态选择的成本敏感学习方法。我们在WorldSense数据集上，使用Qwen2.5-Omni和Qwen3-Omni模型，在输入token数、延迟和货币成本三种推理成本度量下评估我们的方法。在两个模型上，逐查询模态选择均持续改善了精度-成本的权衡表现。

    arXiv:2610.31768v1 Announce Type: new  Abstract: Omni-modal large language models (Omni-LLMs) can jointly process a broad range of modalities, such as text, audio, and video, enabling stronger multimodal understanding and reasoning capabilities, but often at substantial inference cost. We observe that the benefit of additional modalities varies across queries, indicating that a fixed modality configuration can be inefficient. Motivated by this, we study query-wise modality selection, which keeps the underlying Omni-LLM fixed while adaptively selecting which modalities to provide for each query. We formulate the problem as an accuracy-cost optimization problem and develop a cost-sensitive learning approach for modality selection. We evaluate our approach on WorldSense using Qwen2.5-Omni and Qwen3-Omni under three inference-cost measures: input tokens, latency, and monetary cost. Across both models, query-wise modality selection consistently improves the accuracy-cost trade-off and can e
    
[^181]: 学会被监测：面向思维链可监测性的因果界与干预感知的后训练方案

    Learning to Be Monitored: Causal Bounds and Intervention-Aware Post-Training Recipes for Chain-of-Thought Monitorability

    [https://arxiv.org/abs/2610.31733](https://arxiv.org/abs/2610.31733)

    该论文指出常用的 CoT 监测评估指标 TPR_mc 只是因果检测率的上界并推导出互补下界，据此建立了更严格的 CoT 可监测性评估框架，并探索通过后训练方法使大语言模型的思维链更易于被监测。

    

    思维链（CoT）监测旨在从大型语言模型（LLM）的思维链中检测其潜在的有害行为。基于干预的评估方法通过检验“当目标行为发生时，监测器能否从思维链中识别出干预的作用”来衡量这一属性，通常采用 Guan 等人（2025）提出的最低标准敏感度（$\mathsf{TPR_{mc}}$）指标。然而，在干预下观察到的目标行为并不一定是由干预引起的，我们证明该指标可能会高估监测器对“实际由干预引起的目标行为”的检出频率。具体而言，我们将 $\mathsf{TPR_{mc}}$ 确立为该因果检测率的上界，并推导出一个互补的下界，从而为 CoT 可监测性提供了更严格的评估框架。借助该框架，我们研究了能否通过训练使 LLM 的思维链变得更易于监测。我们首先构建了一个训练（后文截断）

    arXiv:2610.31733v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) monitoring aims to detect potentially harmful behaviors in large language models (LLMs) from their CoTs. Intervention-based evaluations measure this property by asking whether a monitor can detect the role of an intervention from the CoT when the targeted behavior occurs, typically using minimal-criterion sensitivity ($\mathsf{TPR_{mc}}$) proposed by~\citep{guan2025monitoring}. However, a target behavior observed under intervention is not necessarily caused by it, and we show that this metric can overestimate how often a monitor detects target behaviors actually caused by the intervention. Specifically, we establish $\mathsf{TPR_{mc}}$ as an upper bound on this causal detection rate and derive a complementary lower bound, which provides a tighter evaluation framework for CoT monitorability. With this framework, we study whether LLMs can be trained to make their CoTs easier to monitor. We first construct a trainin
    
[^182]: ConsMSA：面向多模态情感分析的语义分布一致性学习

    ConsMSA: Semantic Distribution Consistency Learning for Multimodal Sentiment Analysis

    [https://arxiv.org/abs/2610.31731](https://arxiv.org/abs/2610.31731)

    该论文提出ConsMSA，通过将多模态特征投影到共享情感空间并计算模态内与模态间一致性得分（I²CS），构建一致性感知的重要性信号来解决多模态序列中的语义不一致问题，从而提升多模态情感分析的鲁棒性与效率。

    

    多模态情感分析（MSA）旨在通过整合文本、视频和音频等不同模态的信号来预测人类情感。然而，原始的多模态序列常常存在语义不一致问题——在模态内部和模态之间表现出冗余或冲突——这阻碍了对情感的鲁棒理解并增加了计算成本。为此，我们提出了ConsMSA，它显式地将模态内与模态间的语义分布一致性形式化，为鲁棒且高效的多模态情感预测提供了一种一致性感知机制。具体而言，ConsMSA将多模态token特征投影到共享的情感空间中，以计算模态内与模态间一致性得分（I²CS）。通过将该得分与预测相关性相结合，我们构建了一致性感知的重要性信号，并将其用于：(i) 作为一致性正则化器来对齐潜在分布……

    arXiv:2610.31731v1 Announce Type: new  Abstract: Multimodal sentiment analysis (MSA) aims to predict human sentiments by integrating signals from different modalities such as text, video, and audio. However, raw multimodal sequences often suffer from semantic inconsistency--exhibiting redundancy or conflicts within and across modalities--which hinders robust understanding and increases computational cost. To this end, we introduce ConsMSA, which explicitly formalizes semantic distribution consistency across both $\textit{intra}$- and $\textit{inter}$-modality, providing a consistency-aware mechanism for robust and efficient multimodal sentiment prediction. Specifically, ConsMSA projects multimodal token features into a shared sentiment space to compute an Intra- and Inter-modality Consistency Score ($I^2CS$). By coupling this score with predictive relevance, we formulate consistency-aware importance signals that are utilized: (i) as a consistency regularizer to align latent distributio
    
[^183]: Muon 何时以及如何进行预条件处理？结构化梯度噪声的作用

    When and How Does Muon Precondition? The Role of Structured Gradient Noise

    [https://arxiv.org/abs/2610.31728](https://arxiv.org/abs/2610.31728)

    本文从理论上揭示了 Muon 的预条件收益取决于梯度噪声的结构——在各向同性噪声下 Muon 不带来超越学习率缩放的加速，而在异构任务频率产生的结构化噪声下，Muon 能将对任务频率的依赖从线性降为平方根，从而平衡高低频任务的学习并避免 SignSGD 中由旋转引起的负迁移。

    

    Muon 已成为大语言模型训练中广受欢迎的优化器，但其理论基础仍鲜为人知。在本工作中，我们研究梯度噪声的几何结构如何塑造 Muon 预条件处理所带来的收益。我们首先在理论和实证上表明，在经典的各向同性噪声假设下，Muon 除了对学习率进行重新缩放之外并不能提供任何加速。这促使我们研究在具有异构任务频率的联想记忆任务中，由小批量采样所产生的结构化噪声。我们为两个任务的情形建立了精确的平衡阈值，并将分析扩展到多个任务，揭示了任务覆盖范围和梯度可靠性如何共同塑造学习过程。我们的分析表明，Muon 能够将对任务频率的依赖从线性关系软化为平方根关系，有助于平衡高频任务与低频任务的学习。我们进一步识别了 SignSGD 中由旋转引起的负迁移现象，而 Muon 可以避免这种负迁移，并且……

    arXiv:2610.31728v1 Announce Type: new  Abstract: Muon has become a popular optimizer for large language model training, yet its theoretical foundations remain poorly understood. In this work, we study how gradient noise geometry shapes Muon's preconditioning benefits. We first show, theoretically and empirically, that under the classical isotropic noise assumption, Muon offers no acceleration beyond a rescaling of the learning rate. This motivates us to investigate structured noise arising from mini-batch sampling in associative memory with heterogeneous task frequencies. We establish an exact balancing threshold for two tasks and extend the analysis to multiple tasks, revealing how task coverage and gradient reliability jointly shape learning. Our analysis shows that Muon can soften the dependence on task frequency from linear to square-root, helping balance the learning of frequent and rare tasks. We further identify rotation-induced negative transfer in SignSGD that Muon avoids, and
    
[^184]: 用于实时分区控制中零样本协调的认知受限伙伴

    Cognitively Constrained Partners for Zero-Shot Coordination in Real-Time Partitioned Control

    [https://arxiv.org/abs/2610.31726](https://arxiv.org/abs/2610.31726)

    本文提出一种基于人类生理学约束（反应延迟与感知限制）生成多样化模拟伙伴的框架，用于训练人机协作智能体，初步证明这些伙伴的行为更接近真实人类，并能实现与真人的零样本协调。

    

    有效的人机协作需要能够与不可预测的人类伙伴进行协调的智能体。从实时人类反馈或大型行为数据集进行强化学习成本高昂，而纯模拟训练则会过拟合于人工约定，无法实现与真实人类的零样本协调。我们提出了一个框架，用于针对“认知受限伙伴”训练协作智能体：这类模拟伙伴的多样性来源于基于人类生理学文献构建的程序性约束，具体指反应延迟和感知限制，而非来自奖励塑形或历史训练检查点。在一个改进的“月球着陆器”环境中，我们报告了初步证据，表明这些伙伴的行为比无约束的智能体更像真实人类，并且针对它们训练的智能体能够匹敌最强的交叉对弈基线，同时是唯一一个其性能不依赖于训练方式如何（配置）的智能体。

    arXiv:2610.31726v1 Announce Type: new  Abstract: Effective human-machine teaming requires agents that coordinate with unpredictable human partners. Reinforcement learning from live human feedback or large behavioural datasets is expensive, while purely simulated training overfits to artificial conventions and fails at Zero-Shot Coordination with real people. We propose a framework for training teaming agents against Cognitively Constrained Partners: simulated partners whose diversity emerges from procedural constraints grounded in human physiological literature, specifically reaction delays and perceptual limitations, rather than from reward shaping or historical training checkpoints. In a modified Lunar Lander environment, we report preliminary evidence that these partners play more like real people than unconstrained agents do, and that agents trained against them match the strongest cross-play baselines while being the only ones whose performance does not depend on how their trainin
    
[^185]: 操作化较弱忠实性：恢复、歧义与计算极限

    Operationalising Weaker Faithfulness: Recovery, Ambiguity and Computational Limits

    [https://arxiv.org/abs/2610.31724](https://arxiv.org/abs/2610.31724)

    本文提出两种操作化弱忠实性的策略——邻接学习中的选择性边保护与显式保留备选有向图解释（解释家族），使结构学习器能够恢复单独独立但联合预测结果的变量（如XOR）之间的联系，将完整父对恢复率从0/20提升至20/20，且家族估计器在48/50的样本中保留了全部备选解释。

    

    变量可以联合预测某个结果，而每个变量单独却与该结果相互独立，这会导致结构学习器丢弃含有信息量的关联。我们研究如何通过两种途径来操作化较弱忠实性：一是在邻接学习器中进行选择性保护，二是显式保留备选的有向图解释。匹配实验表明，早期候选移除阶段与后期剪枝阶段两处的改动都可能是必要的。在带后代节点的噪声XOR实验中（N=10,000，20个匹配样本），将重新激活与保护相结合，使完整父节点对的恢复率从单独采用任一改动时的0/20提升至20/20，但同时会引入假边。一项验证性研究显示，家族估计器能够保留因对受保护学习器所选骨架进行定向而丢失的解释：在一个较弱信号的XOR条件下（N=10,000），它在48/50个样本中保留了完整的参考解释家族，而骨架自适应[估计器]仅为0/50。

    arXiv:2610.31724v1 Announce Type: cross  Abstract: Variables can predict an outcome jointly while each is individually independent of it, causing structure learners to discard informative links. We investigate how to operationalise weaker faithfulness through two approaches: selective protection in an adjacency learner and explicit retention of alternative directed graph explanations. Matched experiments show that changes at both an early candidate-removal stage and later pruning can be necessary. In noisy XOR with a descendant (N = 10,000, 20 matched samples), combining reactivation and protection raises complete parent-pair recovery from 0/20 for either change alone to 20/20, while adding false edges. A confirmation study shows that the family estimator can preserve explanations lost by orienting the protected learner's selected skeleton: in one weaker signal XOR condition (N = 10,000) it retains the entire reference family in 48/50 samples, compared with 0/50 for the skeleton adapte
    
[^186]: 基于大语言模型的微服务系统自回归多模态异常检测

    Autoregressive Multimodal Anomaly Detection with Large Language Models for Microservice Systems

    [https://arxiv.org/abs/2610.31722](https://arxiv.org/abs/2610.31722)

    该论文提出了LLMMAD，一个基于大语言模型的自回归多模态异常检测框架，通过在跨模态融合中保留模态特定特征并解决“身份捷径”问题，来提升微服务系统的异常检测效果。

    

    微服务系统已成为大规模云应用的主流架构，支持快速部署和弹性伸缩。然而，确保微服务系统的可靠性和可用性关键依赖于有效的异常检测，因为故障和性能异常会在服务之间快速传播，严重影响端到端的系统行为。现有的微服务系统多模态异常检测方法通常面临两个根本性的局限：（i）为了跨模态融合而忽略了模态特定的特征，以及（ii）影响特征重构过程的“身份捷径”问题。为了解决这些问题，我们提出了LLMMAD，一个基于大语言模型的微服务系统自回归多模态异常检测框架。首先，为了协调跨模态融合与模态特定特征的保留，LLMMAD将（原文在此处截断）

    arXiv:2610.31722v1 Announce Type: cross  Abstract: Microservice systems have become a dominant architecture for large-scale cloud applications, enabling rapid deployment and elastic scaling. However, ensuring the reliability and availability of microservice systems critically relies on effective anomaly detection, as failures and performance anomalies can quickly propagate across services and severely impact end-to-end system behavior. Existing multimodal anomaly detection approaches for microservice systems often struggle with two fundamental limitations: (i) the neglect of modality-specific characteristics in favor of cross-modality fusion, and (ii) the "identity-shortcut" problem that affects the feature reconstruction process. To address these issues, we propose LLMMAD, an autoregressive multimodal anomaly detection framework for microservice systems based on Large Language Models (LLMs). Initially, to reconcile cross-modality fusion with modality-specific preservation, LLMMAD embe
    
[^187]: 面向物理信息优化的端到端可微扩散条件化方法

    End-to-End Differentiable Diffusion Conditioning for Physics-Informed Optimization

    [https://arxiv.org/abs/2610.31721](https://arxiv.org/abs/2610.31721)

    D2C利用确定性采样使反向去噪过程可微的特性，将冻结的条件扩散模型转化为端到端的梯度优化器，在测试时通过代理目标和物理信息的梯度直接优化条件变量，从而克服了离线基于模型优化中现有方法过于保守或无法精确对齐物理目标的问题。

    

    在昂贵的评估成本和复杂约束下优化高维物理系统是科学与工程领域的核心挑战。在离线基于模型的优化（MBO）中，方法必须仅使用固定数据集来改进设计，并且在测试时无法访问真实的目标函数。然而，现有方法通常存在过于保守、对无约束代理模型过度利用等问题，或者只能作为静态的推理时采样器，无法针对精确的物理目标对候选方案进行优化。在本文中，我们提出了用于离线基于模型优化的可微扩散条件化方法，该方法利用了一个关键事实：确定性采样使得反向去噪链相对于条件信号是可微的。D2C将冻结的条件扩散模型转变为端到端的基于梯度的优化器，其方式是在测试时通过来自冻结代理目标和物理信息（原文摘要在此处截断）的梯度直接优化条件变量。

    arXiv:2610.31721v1 Announce Type: cross  Abstract: Optimizing high-dimensional physical systems under expensive evaluations and complex constraints is a central challenge in science and engineering. In offline model-based optimization (MBO), methods must improve designs using only a fixed dataset and no test-time access to the true objective. However, existing methods typically suffer from conservatism, unconstrained surrogate exploitation, or function as static inference-time samplers that cannot refine candidates against precise physical objectives. In this paper, we propose Differentiable Diffusion Conditioning (D2C) for offline MBO, which leverages the fact that deterministic sampling makes the reverse denoising chain differentiable with respect to the conditioning signal. D2C turns a frozen conditional diffusion model into an end-to-end gradient-based optimizer by directly optimizing conditioning variables at test time via gradients from frozen surrogate objectives and physics-inf
    
[^188]: 面向金融犯罪分析的对抗鲁棒多重图神经网络

    Towards Adversarially Robust Multigraph Neural Networks for Financial Crime Analysis

    [https://arxiv.org/abs/2610.31718](https://arxiv.org/abs/2610.31718)

    该论文提出了首个针对多重图神经网络的结构攻击与对抗训练可扩展框架，通过节点-边关联矩阵优化结构扰动，并引入基于有向环计数的正则化器衡量攻击合理性，从而提升反洗钱任务的对抗鲁棒性。

    

    金融交易网络天然可以用多重图来表示，其中同一对账户之间的重复交易被建模为不同的边。最近提出的多重图神经网络架构显式利用这种结构，在反洗钱（AML）任务上取得了强劲的性能。然而，它们对通过操纵交易结构来逃避检测的对抗者的鲁棒性在很大程度上仍未被探索。我们提出了首个基于节点-边关联矩阵的多重图神经网络结构攻击与对抗训练的可扩展框架，将投影梯度下降和投影随机块坐标下降方法适配于在此类矩阵上优化结构扰动。我们进一步引入了一种基于环的正则化器，对有向2-环和3-环的全局计数变化进行惩罚，提供了一种衡量攻击合理性的结构化指标。在……上的实验（摘要不完整）

    arXiv:2610.31718v1 Announce Type: new  Abstract: Financial transaction networks are naturally represented as multigraphs, where repeated transactions between the same pair of accounts are modeled as distinct edges. Recently proposed multigraph neural network architectures leverage this structure explicitly to achieve a strong performance on anti-money laundering (AML) tasks. However, their robustness to adversaries that manipulate transaction structures to evade detection remains largely unexplored. We present the first scalable framework for structural attacks and adversarial training of multigraph neural networks using node-edge incidence matrices, adapting projected gradient descent and projected randomized block coordinate descent methods to optimize structural perturbations over such matrices. We further introduce a cycle-based regularizer that penalizes changes in the global counts of directed 2- and 3-cycles, providing a structural measure of attack plausibility. Experiments on 
    
[^189]: LeWorldModel 是否保留了局部动作几何结构？

    Does LeWorldModel Preserve Local Action Geometry?

    [https://arxiv.org/abs/2610.31716](https://arxiv.org/abs/2610.31716)

    本文通过在多个扰动尺度上使用 Gram 矩阵评估预训练 LeWorldModel 对微小动作变化的响应，发现该世界模型未能很好地保留局部动作几何结构，并提出应对动作敏感性进行针对特定尺度的评估。

    

    我们研究了预训练的 LeWorldModel（LeWM）在多大程度上能够复现微小动作变化所产生的效果。通过匹配模拟器与模型响应，我们在三个扰动尺度上使用 Gram 矩阵来度量动作敏感性。在来自 Reacher 和 OGBench Cube 环境的近千个状态中，在最小扰动尺度下，最敏感的动作方向分别以约 36° 和 42° 的中位角度存在差异。将扰动增大十倍后，Reacher 上的中位归一化 Gram 差异降低约 70%，Cube 上降低约 47%。以像素和嵌入（embedding）度量的几何结构在不同尺度间的变化，比所选的物理状态几何结构更为显著。在保留测试方向上，环境 Gram 矩阵的中位嵌入响应误差分别为 17% 和 28%，而具有相同迹的各向同性参考模型则为 26% 和 38%。LeWM 在该尺度下的误差更高。这些发现支持对动作敏感性进行针对特定尺度的评估……

    arXiv:2610.31716v1 Announce Type: new  Abstract: We examine how closely pretrained LeWorldModel (LeWM) reproduces the effects of small action changes. Using matched simulator and model responses, we measure action sensitivity with Gram matrices at three perturbation scales. Across nearly a thousand states from Reacher and OGBench Cube, the most sensitive action directions differ by median angles of about $36^\circ$ and $42^\circ$, respectively, at the smallest scale. Increasing the perturbation tenfold reduces median normalized Gram discrepancy by roughly 70\% on Reacher and 47\% on Cube. Geometry measured in pixels and embeddings varies more across scales than the chosen physical-state geometry. On held-out directions, environment Grams achieve median embedding response errors of 17\% and 28\%, compared with 26\% and 38\% for an isotropic reference with the same trace. LeWM's errors are higher at this scale. These findings support scale-specific evaluation of action sensitivity; its e
    
[^190]: 复用、重写还是拒绝：面向重复移动端GUI控制的证据约束轨迹复用

    Reuse, Rewrite, or Reject: Evidence-Constrained Trace Reuse for Repeated Mobile GUI Control

    [https://arxiv.org/abs/2610.31712](https://arxiv.org/abs/2610.31712)

    该论文提出图-文本模板与“记录-模板-回放”框架，通过对UI状态转移进行证据约束验证，实现移动端GUI智能体在重复任务中对先前执行轨迹的选择性复用，从而降低冗余的LLM推理与执行开销。

    

    移动端GUI智能体经常执行具有相似交互流程的重复性任务。复用先前的执行记录可以减少冗余的大语言模型（LLM）推理与执行开销，但用户指令和执行环境的变化可能使先前成功的操作失效。我们系统地将这些变化归纳为六个代表性类别，并观察到此类变化并不一定会使整个流程失效：在针对当前任务上下文和界面状态进行验证后，单个状态转移仍可保持可复用性。基于这一洞察，我们提出了用于选择性执行轨迹复用的图-文本模板。图结构捕获UI状态和动作转移，而文本编码任务意图、流程结构以及用于可变内容的类型化槽位。一个“记录-模板-回放”框架从经过验证的执行中构建模板，并利用当前指令和实时UI状态来复用……

    arXiv:2610.31712v1 Announce Type: cross  Abstract: Mobile GUI agents frequently perform repetitive tasks with similar interaction procedures. Reusing previous executions can reduce redundant LLM inference and execution overhead, but variations in user instructions and execution environments may invalidate previously successful actions. We systematically characterize these variations into six representative categories and observe that such changes do not necessarily invalidate entire procedures: individual transitions can remain reusable when validated against the current task context and interface state. Based on this insight, we propose graph-text templates for selective execution-trace reuse. The graph captures UI states and action transitions, while the text encodes task intent, procedural structure, and typed slots for variable content. A record-template-replay framework constructs templates from verified executions and leverages the current instruction and live UI state to reuse, 
    
[^191]: 迈向无损检测4.0之路：自主检测的算法、技术与策略综述

    The Path to NDE 4.0: A Review of Algorithms, Techniques, and Strategies for Autonomous Inspection

    [https://arxiv.org/abs/2610.31704](https://arxiv.org/abs/2610.31704)

    本文从计算算法、物理技术、工业应用和研究目标四个视角全面综述了无损检测领域向NDE 4.0数字化数据驱动范式的演进，指出深度学习和生成模型正推动自动缺陷识别的发展，同时揭示标准化开放数据匮乏是行业面临的主要瓶颈。

    

    无损检测（NDI）是现代质量保证的基石，对于在不造成损坏的情况下确保基础设施和制造部件的安全性与可靠性至关重要。本文对当前无损检测领域进行了全面综述，从四个不同的视角对该领域进行分析，即计算算法、物理技术、工业应用和研究目标。我们综合了大量学术文献，以阐明该领域从手动、模拟解读向以数据为中心的数字化范式（即NDE 4.0）的快速演变。我们的分析突出了超声波和目视检测作为主要物理技术的主导地位，同时揭示了自动缺陷识别向深度学习和生成模型的重大转变。尽管取得了这些技术进步，但该行业仍面临关键瓶颈，特别是标准化开放获取数据的稀缺。

    arXiv:2610.31704v1 Announce Type: new  Abstract: Non-Destructive Inspection (NDI) is a cornerstone of modern quality assurance, critical for ensuring the safety and reliability of infrastructure and manufactured components without causing damage. This paper presents a comprehensive review of the current NDI landscape, analyzing the field through four distinct lenses, namely, computational algorithms, physical techniques, industrial applications, and research objectives. We synthesize a broad range of academic literature to illustrate the rapid evolution from manual, analog interpretation to a digitized, data-centric paradigm known as NDE 4.0. Our analysis highlights the dominance of ultrasound and visual inspection as primary physical techniques, while revealing a significant shift toward deep learning and generative models for automated defect recognition. Despite these technological strides, the industry faces critical bottlenecks, notably the scarcity of standardized open-access dat
    
[^192]: 面向室内无线电地图估计的协作式上下文学习

    Collaborative In-Context Learning for Indoor Radio Map Estimation

    [https://arxiv.org/abs/2610.31702](https://arxiv.org/abs/2610.31702)

    提出Poly-Ripple框架，首次将室内无线电地图估计形式化为协作式上下文学习问题，使多个客户端利用大型自回归视觉模型和波纹序自回归建模，在无需模型更新并保护隐私的前提下协作预测无线电地图。

    

    室内无线电地图估计（RME）对于覆盖规划、定位和网络部署等无线应用至关重要。然而，现有的大多数方法遵循监督学习范式，需要昂贵的标注数据和针对特定任务的模型训练。此外，室内环境涉及隐私敏感信息，例如建筑布局和结构，这些信息通常无法集中收集。为应对这些挑战，我们首次尝试将室内无线电地图估计形式化为一个协作式上下文学习（ICL）问题。我们提出了Poly-Ripple框架，其中多个客户端使用大型自回归视觉模型（LAVM）执行上下文学习，而无需进行模型更新。每个客户端由其私有的示例进行提示，并针对相同的场景查询预测无线电地图。为了捕捉无线信号从发射机（Tx）向外传播的因果结构，我们采用了波纹序自回归建模方法。

    arXiv:2610.31702v1 Announce Type: cross  Abstract: Indoor radio map estimation (RME) is essential for wireless applications such as coverage planning, localization, and network deployment. However, most existing methods follow a supervised learning paradigm that requires costly labeled data and task-specific model training. Moreover, indoor environments involve privacy-sensitive information, such as building layouts and structures, that often cannot be centralized. To address these challenges, we make the first attempt to formulate indoor RME as a collaborative in-context learning (ICL) problem. We propose Poly-Ripple, a framework in which multiple clients perform ICL using large autoregressive vision models (LAVMs) without model updates. Each client is prompted by its private demonstrations and predicts the radio map for the same scene query. To capture the causal structure of wireless propagation outward from the transmitter (Tx), we adopt ripple-ordered autoregressive modeling. We f
    
[^193]: 欺骗大语言模型安全防护机制对错误的文本进行判断

    Fooling LLM Safety Guards into Judging the Wrong Text

    [https://arxiv.org/abs/2610.31695](https://arxiv.org/abs/2610.31695)

    论文提出“目标重绑定”攻击方法——通过附加伪造查询和无害诱饵，利用安全防护机制只评判输入中最后一个完整任务的特性，使有害内容绕过审查，在GuardRebind基准上对五种防护机制的绕过率均超过93%。

    

    在LLM应用程序对内容采取行动之前，安全防护机制会先检查内容，而大多数防护机制会将该内容直接粘贴到纯文本提示词中。模仿提示词格式的内容可以附加第二个任务，防护机制反而会去回答那个任务。我们将这一现象称为“目标重绑定”。我们的解释是：防护机制评判的是其输入中最后一个完整的任务（此处为无害的诱饵），因此有害请求得以原封不动地通过。在一项预注册测试中，附加的伪造查询和诱饵成功绕过了Shieldstral-1.0-3B标记的全部75个保留AdvBench有害请求。防护机制并非被说服："忽略上述文本”的指令无法绕过其中任何一个请求，且在后续运行中，有害的诱饵反而会阻止攻击。在我们的GuardRebind基准测试中，五种防护机制中的每一种都有一个（针对各防护机制选择的）攻击家族，能够绕过至少93%的被其标记的项目。目标重绑定对Shieldstral和gpt-oss-safeguard的解释效果最好；而对于ShieldGemma和Llama Guard 3，通用规避和尾部文本策略分别……

    arXiv:2610.31695v1 Announce Type: cross  Abstract: Safety guards check content before an LLM application acts on it, and most paste that content into a plain-text prompt. Content that imitates the prompt can append a second task, and the guard answers that one instead. We call this target rebinding. Our explanation is that the guard grades the last complete task in its input, here a harmless decoy, so the harmful request goes through unchanged. In a pre-registered test, an appended forged query and decoy clear all 75 held-out AdvBench requests that Shieldstral-1.0-3B flags. The guard is not persuaded: a directive to ignore the text above clears none of them, and in a follow-up run a harmful decoy stops the attack. On our GuardRebind benchmark, each of five guards has an attack family, chosen per guard, that clears at least 93% of its flagged items. Rebinding explains this best on Shieldstral and gpt-oss-safeguard; on ShieldGemma and Llama Guard 3, generic evasion and trailing text resp
    
[^194]: 针对自闭症与非自闭症个体自发双人对话互动的多模态感知对话成功预测

    Multi modal perceived conversation success prediction for spontaneous autistic and non-autistic dyadic interactions

    [https://arxiv.org/abs/2610.31682](https://arxiv.org/abs/2610.31682)

    本研究提出利用构音、面部、声学-韵律和话轮转换等多模态特征，在包含自闭症与非自闭症个体的自发双人对话数据中预测参与者感知的对话成功，证明对话成功本质上是多维的，需要多层面的分析框架。

    

    对话成功对于沟通、社会纽带和幸福感至关重要，而在自闭症者与神经典型者的对话中，对话成功往往会降低。先前的研究曾在儿科自闭症富集样本中使用声学和词汇特征来预测感知的对话成功，或在非自闭症群体互动中使用非言语行为进行预测。其他研究刻画了自闭症互动中的韵律、面部、动作和话轮转换同步性，但主要针对诊断分类或组间差异分析。本研究则从互补的构音、面部、声学-韵律和话轮转换测量出发，在三个包含自闭症和非自闭症个体的自发双人互动数据集上预测参与者感知的对话成功。通过利用多模态特征，我们证明对话成功本质上是多维的，需要一个多层面的分析框架。

    arXiv:2610.31682v1 Announce Type: new  Abstract: Conversation success is important for communication, social bonds, and well-being and is often reduced in autistic-neurotypical conversations. Previous studies have predicted perceived conversation success using acoustic and lexical features in pediatric autism-enriched samples, or using nonverbal behavior in non-autistic group interactions. Other work has characterized prosodic, facial, motor, and turn-taking synchrony in autistic interactions, but has primarily targeted diagnostic classification or group differences. The present study instead predicts participant perceived conversation success from complementary articulatory, facial, acoustic-prosodic, and turn-taking measures across three spontaneous dyadic-interaction datasets including autistic and non autistic individuals. By leveraging multimodal features, we demonstrate that conversational success is inherently multidimensional and requires a multi-faceted analytical framework. C
    
[^195]: 自回归物理信息神经算子代理模型与四维地震-生产数据联合贝叶斯历史拟合

    Autoregressive Physics-Informed Neural Operator Surrogates and Joint 4-D Seismic-Production Bayesian History Matching

    [https://arxiv.org/abs/2610.31677](https://arxiv.org/abs/2610.31677)

    该论文提出了一种以自回归物理信息傅里叶神经算子作为代理模型加速的贝叶斯历史拟合框架，通过两阶段 aREKI 与 ES-MDA 策略联合同化生产数据与四维地震数据，显著降低了大规模油藏反演的计算成本与病态性。

    

    我们提出了一种由代理模型加速的贝叶斯历史拟合框架，并在一个真实感的合成模型（N=100,188个网格单元）上进行了验证。该框架将正演映射 G 替换为在闭环中训练的傅里叶神经算子 G_θ，即 x̂_{n+1} = G_θ(u, x̂_n)，并以离散黑油方程残差 R(x_{n+1}, x_n) 作为物理信息神经算子（PINO）惩罚项，同时采用截断反向传播与噪声注入技术。我们利用自适应正则化集合卡尔曼反演（aREKI）同化生产数据与四维地震数据，并通过代理模型的闭式 Gassmann 变换在滚动预测过程中合成 ΔV_p，采用两阶段调度方案（先对生产数据执行 aREKI，再对两类数据同时使用多数据同化集合平滑器 ES-MDA）。为缓解问题的病态性，我们并非在 R^{2N} 中直接反演 (log K, φ)，因为当 N_e << 2N 时会导致协方差矩阵秩亏并产生零空间……

    arXiv:2610.31677v1 Announce Type: new  Abstract: We present a surrogate-accelerated Bayesian history-matching framework, demonstrated on a realistic synthetic model ($N=100{,}188$ cells). The forward map $\mathcal{G}$ is replaced by a Fourier neural operator $\mathcal{G}_\theta$ trained in a closed loop, $\hat{x}_{n+1}=\mathcal{G}_\theta(u,\hat{x}_n)$, with the discrete black-oil residual $R(x_{n+1},x_n)$ as a physics-informed neural operator (PINO) penalty, truncated backpropagation and noise injection. We assimilate production data and 4-D seismic using adaptive regularised ensemble Kalman inversion (aREKI), and synthesise $\Delta V_p$ inside the rollout via a closed-form Gassmann transform of the surrogate, under a two-stage schedule (aREKI on production, then an ensemble smoother with multiple data assimilation, ES-MDA, on both). To temper the ill-posedness, we invert $(\log K,\phi)$ not in $\mathbb{R}^{2N}$, where $N_e\ll 2N$ leaves a rank-deficient covariance and a null space the
    
[^196]: 结合SHAP分析的混合SEIR-时间融合Transformer框架用于可解释的大流行病预测

    Hybrid SEIR Temporal Fusion Transformer Framework with SHAP Analysis for Interpretable Pandemic Forecasting

    [https://arxiv.org/abs/2610.31673](https://arxiv.org/abs/2610.31673)

    该论文提出了一种将SEIR流行病学模型与时间融合Transformer（TFT）相结合的混合预测框架，并借助SHAP分析实现跨地区大流行病感染的可解释多时间步预测。

    

    对异质性地区的大流行病传播进行预测，需要既准确又可解释的模型以支持公共卫生决策行动。本文提出了一种混合预测框架，将SEIR流行病学模型与时间融合Transformer（TFT）相结合，生成可解释的多时间步感染预测。利用来自Our World in Data语料库的各地区COVID-19数据，我们提出了一个统一的处理流程，整合了流行病学指标（易感者、暴露者、感染者、康复者）、干预信号（口罩佩戴、疫苗接种、政策）、流动性协变量以及人口统计背景信息。其中，SEIR组件用于估计潜在的疾病动态并注入机理先验知识，而TFT则从历史观测数据和已知未来输入中学习非线性时间依赖关系与跨特征交互作用。为确保模型透明度，SHAP（SHapley加性解释）方法被应用于全局和（原文此处截断）……

    arXiv:2610.31673v1 Announce Type: cross  Abstract: Forecasting pandemic spread across heterogeneous regions demands models that are both accurate and interpretable for public health action. This paper proposes a hybrid forecasting framework that integrates an SEIR epidemiological model enhanced with a Temporal Fusion Transformer (TFT) to produce interpretable multi-horizon infection forecasts. Using region-wise COVID-19 data from the Our World in Data corpus, we present a unified pipeline that integrates epidemiological indicators (susceptible, exposed, infectious, recovered), intervention signals (masking, vaccination, policies), mobility covariates, and demographic context. The SEIR component estimates latent disease dynamics and injects mechanistic priors, while the TFT learns non-linear temporal dependencies and cross-feature interactions from historical observations and known future inputs. To ensure transparency, SHAP (SHapley Additive exPlanations) is applied at both global and 
    
[^197]: 感知运动对齐：具身策略成功的柏拉图式代理指标

    Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success

    [https://arxiv.org/abs/2610.31670](https://arxiv.org/abs/2610.31670)

    提出无需训练的离线指标“感知运动对齐”（S²），通过衡量模型感知表征与物理运动几何结构的对齐程度，低成本地预测具身模型的下游任务成功率。

    

    机器人基础模型通常需要通过代价高昂的闭环回放（rollout）来评估，这促使人们寻找低成本的离线模型能力代理指标。柏拉图表征假说（PRH）指出，随着表征质量的提升，视觉、语言等不同模态会收敛到对现实的共享统计模型。我们提出一个问题：运动轨迹是否构成另一种这样的模态——当具身模型能力增强时，其感知表征是否应向物理运动的结构收敛？为探究这一点，我们提出了感知运动对齐，这是一种无需训练的离线指标，用于衡量模型的感知表征与物理运动几何结构的对齐程度。我们发现，S² 在多个具身模型家族中均与下游机器人任务成功率呈正相关，因此可作为低成本的离线代理指标，在闭环评估之前对模型进行优先级排序。S² 还能够识别出有前景的（模型）（摘要在此处截断）。

    arXiv:2610.31670v1 Announce Type: cross  Abstract: Robotic foundation models are typically evaluated through costly closed-loop rollouts, which motivates cheap offline proxies for model capability. The Platonic Representation Hypothesis (PRH) suggests that as representations improve, different modalities such as vision and language converge toward a shared statistical model of reality. We ask whether motion trajectories form another such modality: as embodied models become more capable, should their sensory representations converge toward the structure of physical motion? To examine this, we propose Sensorimotor Alignment ($S^2$), a training-free offline metric that measures how well a model's sensory representation aligns with the geometry of physical motion. We find that $S^2$ correlates positively with downstream robot task success across embodied model families, making it a low-cost offline proxy for prioritizing models before closed-loop evaluation. $S^2$ can also identify promisi
    
[^198]: SABLE：可扩展的仿射贝叶斯潜空间探索

    SABLE: Scalable Affine Bayesian Latent-space Exploration

    [https://arxiv.org/abs/2610.31495](https://arxiv.org/abs/2610.31495)

    SABLE 提出了一种贝叶斯仿射代理优化方法，通过将查询分解为由期望改进（EI）优化的低维跨度内分量和由标量范数支配的跨度外分量，在评估预算远小于潜空间维度的高维潜空间中实现了可扩展的贝叶斯优化。

    

    诸如工程设计、蛋白质等复杂对象往往难以优化，因为它们的设计空间呈现高维结构，且目标评估代价高昂。生成模型提供了连续的潜空间表示，但其潜空间可能具有数千至数万维，形成了潜空间维度 D 远超评估预算的情形。我们提出 SABLE，一种针对各向同性分布且预算有限的高维潜空间的贝叶斯优化方法。SABLE 使用贝叶斯仿射代理模型进行预测与采集，并将高阶结构视为代理模型的偏差。在 N 次观测之后，斜率更新的维度至多为 N-1，从而将每个查询分解为一个至多 N 维的跨度内分量（通过期望改进 EI 进行优化）和一个由标量范数支配的跨度外分量。我们表明……

    arXiv:2610.31495v1 Announce Type: new  Abstract: Complex objects such as engineering designs and proteins are often difficult to optimise because their design spaces exhibit high-dimensional structure and objective evaluations are expensive. Generative models provide continuous latent representations, but their latent spaces can have thousands to tens of thousands of dimensions, creating a regime where the latent dimension $D$ greatly exceeds the evaluation budget. We introduce SABLE, a Bayesian optimisation method for high-dimensional latent spaces with an isotropic distribution and limited budget. SABLE uses a Bayesian affine surrogate for prediction and acquisition, treating higher-order structure as surrogate discrepancy. After $N$ observations, the slope update has dimension at most $N-1$, yielding a decomposition of each query into an in-span component optimised by expected improvement (EI) in at most $N$ dimensions and an off-span component governed by a scalar norm. We show tha
    
[^199]: 用于Monge–Ampère方程的凸平衡势：曲率界与后验评估

    Convex Equilibrium Potentials for Monge--Amp\`ere: Curvature Bounds and A Posteriori Evaluation

    [https://arxiv.org/abs/2610.31386](https://arxiv.org/abs/2610.31386)

    本文针对用于Monge–Ampère方程的凸softplus平衡神经网络，证明了Hessian矩阵关于压缩因子具有最优的 $(1-q)^{-2}$ 增长界，并给出了基于残差的后验估计以控制近似平衡点处的状态及其导数误差。

    

    平衡神经网络将其隐藏状态定义为压缩映射的不动点，因此其有效深度是在评估时确定的，而非由网络架构决定。将这类网络用于完全非线性偏微分方程会带来两个显式网络可以避免的问题：二阶导数可能随有效深度增长，且势函数只能被近似求值。我们研究具有压缩因子 $q<1$ 的非负 softplus 平衡网络，这是一类适合 Monge–Ampère 方程的凸函数类。通过预解算子中的正性抵消，我们证明 Hessian 矩阵至多以 $(1-q)^{-2}$ 的速度增长，而不是直接估计所给出的 $(1-q)^{-3}$，并且一个显式的单神经元函数族表明该指数是最优（不可改进）的；三阶导数则满足一个最优的 $(1-q)^{-3}$ 界。随后，基于残差的后验估计可以控制近似平衡点处的状态、其输入导数，以及在二维情形下 Hessian 行列式的误差。

    arXiv:2610.31386v1 Announce Type: cross  Abstract: Equilibrium neural networks define their hidden state as the fixed point of a contraction, so their effective depth is set at evaluation time rather than by the architecture. Using them for fully nonlinear PDEs raises two issues that explicit networks avoid: second derivatives may grow with the effective depth, and the potential is only ever evaluated approximately. We study nonnegative softplus equilibrium networks with contraction factor $q<1$, a convex class suited to the Monge--Amp\`ere equation. A positivity cancellation in the resolvent shows that the Hessian grows at most like $(1-q)^{-2}$, rather than the $(1-q)^{-3}$ of a direct estimate, and an explicit one-neuron family shows that this exponent is sharp; third derivatives satisfy a sharp $(1-q)^{-3}$ bound. Residual-based a posteriori estimates then control the state, its input derivatives and, in two dimensions, the Hessian determinant at an approximate equilibrium. Combine
    
[^200]: 有害回复即越狱模板：将自回归越狱攻击迁移至扩散语言模型

    Harmful Responses Are Jailbreak Templates: Transferring Autoregressive Jailbreaks to Diffusion Language Models

    [https://arxiv.org/abs/2610.31331](https://arxiv.org/abs/2610.31331)

    本文发现有害回复经部分掩码后可天然充当越狱模板，并据此提出RMT-Jailbreak框架，将现有的自回归模型越狱攻击有效迁移到扩散语言模型上。

    

    扩散大语言模型通过双向上下文建模实现了灵活的掩码生成能力，但这一能力同时也引入了独特的越狱攻击面。我们发现，针对给定查询的有害回复天然可以充当有效的越狱模板：只需对其中的部分词元进行掩码处理，就能诱导扩散语言模型重构出有害内容。基于这一观察，我们提出了RMT-Jailbreak，这是一个通用框架，可以将现有的自回归（AR）越狱攻击通过其诱导生成的有害回复迁移到扩散语言模型上。RMT-Jailbreak采用均匀词元保留策略，将AR模型生成的有害回复转换为高掩码比例的模板，并进一步应用基于提示的安全过滤来掩码所有剩余可见的有害内容。精炼后的模板随后被提供给目标扩散语言模型进行双向重构。在多种AR越狱方法和目标扩散语言模型上开展的大量实验表明……

    arXiv:2610.31331v1 Announce Type: new  Abstract: Diffusion large language models (dLLMs) enable flexible masked generation through bidirectional context modeling, but this capability also introduces a distinct jailbreak attack surface. We find that a harmful response to a given query naturally serves as an effective jailbreak template: simply masking a subset of its tokens can induce a dLLM to reconstruct harmful content. Based on this observation, we propose RMT-Jailbreak, a general framework that transfers existing autoregressive (AR) jailbreak attacks to dLLMs through their induced harmful responses. RMT-Jailbreak converts an AR-generated harmful response into a high-mask-ratio template using uniform token retention, and further applies prompt-based safety filtering to mask any remaining visible harmful content. The refined template is then provided to the target dLLM for bidirectional reconstruction. Extensive experiments across multiple AR jailbreak methods and target dLLMs demons
    
[^201]: 系统1模型在医学中的价值

    The Value of System 1 Models in Medicine

    [https://arxiv.org/abs/2610.31315](https://arxiv.org/abs/2610.31315)

    本研究首次大规模评估了四个系统1模型在20种语言、14个临床领域共86个任务上的零样本表现，表明这类能快速输出概率并具备不确定性估计能力的模型，在临床决策中具有独特价值。

    

    许多临床决策是在少数固定选项之间做出的选择，需要快速、大量地完成。此类决策还需要不确定性估计，以便知道何时应转交给临床医生处理。生成式语言模型以自由文本形式回答速度较慢，且对自身置信度的表达不佳。相比之下，系统1模型能够以零样本方式作答，并为每个选项给出概率。然而，这类模型在各类临床任务中的表现尚不清楚。本研究评估了四个系统1模型（两个专有模型和两个开放权重模型）在86个公开任务上的零样本表现，这些任务来自14个临床领域、20种语言，涵盖考试题目和医学影像。在分析之前，任务被划分为六个能力层级。主要结局指标为受试者工作特征曲线下面积（AUROC），次要结局指标包括校准度、语言效应，以及与已发表结果和生成式模型的比较。在179,111个病例中，最佳模型的平均AUROC……（摘要原文在此处截断）

    arXiv:2610.31315v1 Announce Type: new  Abstract: Many clinical decisions are a choice among a few fixed options, taken quickly and in large numbers. Such decisions also need an uncertainty estimate to know when to defer to a clinician. Generative language models answer slowly in free text and state their confidence poorly. In contrast, System 1 models answer zero-shot with a probability for each answer. However, their performance across clinical tasks is unknown. Four System 1 models, two proprietary and two with open weights, were evaluated zero-shot on 86 public tasks from 14 clinical domains in 20 languages, spanning examination questions and medical imaging. Tasks were assigned to six capability tiers before analysis. The primary outcome was the area under the receiver-operating-characteristic curve (AUROC). Secondary outcomes included calibration, language effects and comparisons with published results and generative models. Across 179,111 cases, the mean AUROC of the best model p
    
[^202]: 基于粒子物理基础模型的径迹寻找与粒子鉴别性能研究

    Track finding and particle identification performance with particle physics foundation models

    [https://arxiv.org/abs/2610.31180](https://arxiv.org/abs/2610.31180)

    该论文在时间投影室数据集上评估了粒子物理基础模型在径迹寻找和粒子鉴别等下游任务中的表现，证明基于基础模型的数据分析方法性能显著优于传统方法，且可灵活适应新任务。

    

    现代高能核物理与粒子物理实验会产生大量稀疏数据，在进行物理分析之前必须对其进行大幅度的规模缩减。近年来，研究者们一直在尝试将基础模型评估为执行数据分析的潜在工具。已有研究表明，这些模型能够基于其所训练的多模态数据来学习高维数据集中的特征。为了展示基础模型在执行复杂模式识别任务（例如在磁场中重建带电粒子）方面的有效性，我们在一个由时间投影室空间点组成的数据集上，评估了基于基础模型的重建方法在多个下游任务中的性能。结果表明，基于基础模型的数据分析性能显著优于传统方法，此外该方法还可以被调整用于学习与原任务不一定相关的额外任务。

    arXiv:2610.31180v1 Announce Type: cross  Abstract: Modern high energy nuclear and particle physics experiments produce large amounts of sparse data that must be reduced significantly in size before performing physics analysis. Recently, there has been an effort to evaluate Foundation Models as a potential tool for performing data analysis. These models have been shown to learn features in high dimensional datasets based on the multi-modal data they are trained on. To show their effectiveness at performing complex pattern recognition tasks, such as reconstructing charged particles in a magnetic field, we evaluated the performance of a Foundation Model based reconstruction on several downstream tasks with a dataset composed of spacepoints in a Time Projection Chamber. We show that the performance of the Foundation Model based data analysis significantly improves upon that of traditional methods, and furthermore can be adapted to learn additional tasks that are not necessarily related to 
    
[^203]: 主要是阅读：开卷评分在文档级LoRA适配器中究竟衡量了什么

    Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters

    [https://arxiv.org/abs/2610.31017](https://arxiv.org/abs/2610.31017)

    开卷评分主要衡量的是所有适配器共有的“阅读”行为而非文档记忆——随机适配器即可恢复专用适配器82%的开卷优势，但在无源文本的完形测试中仅恢复约2%（与零无异），因此不宜用开卷评分来评估或选择文档级LoRA适配器。

    

    单文档LoRA适配器已被提出作为可合并与检索的知识单元；然而，开卷评分——即将文档放入提示词中——会同样容易地把功劳归给文档自己的适配器（其专用适配器）的“阅读”行为，而非真正“记住”该文档。我们在Llama-3.1-8B-Instruct上为150篇Wikipedia文章各训练一个适配器，发现开卷分数主要衡量的是每个适配器都共有的一种阅读行为：一个在其他文档上训练的随机适配器能够恢复专用适配器相对于基础模型82%的开卷优势，但在一个不提供源文本、直接询问同一文档事实的完形填空探测中，只能恢复其2%的优势——这一比例与零无法区分。文档内容确实写入了权重，因为专用适配器在完形回忆上比基础模型高出0.094，且其更新的几何结构遵循其所阅读的文档；然而，在一项包含30个查询的测试中，通过权重空间邻近性选择的适配器……（原文在此处截断）

    arXiv:2610.31017v1 Announce Type: new  Abstract: Single-document LoRA adapters have been proposed as units of knowledge to be merged and retrieved; yet open-book scoring, with the document in the prompt, can credit a document's own adapter (its specialist) for reading that document as readily as for holding it. Training one adapter for each of 150 Wikipedia articles on Llama-3.1-8B-Instruct, we find that open-book scores mostly measure a reading behaviour every adapter shares: a random adapter trained on another document recovers 82 % of the specialist's open-book advantage over the base model but only 2 % of its advantage on a cloze probe that asks for the same document's facts without providing the source text, a share we cannot distinguish from zero. Content does reach the weights, since the specialists beat the base model in cloze recall by 0.094, and the geometry of their updates follows the documents they read; yet, in a test of 30 queries, adapters chosen by weight-space proximi
    
[^204]: OpenAttention：用尾部代理，百分之一足矣

    OpenAttention: One Percent Is Enough with Tail Proxies

    [https://arxiv.org/abs/2610.30951](https://arxiv.org/abs/2610.30951)

    OpenAttention通过将整个未读尾部压缩为单个计数感知均值代理token，在读取率低至1%时仍能获得比丢弃或采样更低误差的稀疏注意力输出。

    

    arXiv:2610.30951v1 公告类型：新论文 摘要：稀疏注意力只读取键值缓存中的一小部分，而如何处理未被读取的尾部仍然是一个关键挑战。直接丢弃尾部会使输出产生偏差，而以往方法采用的尾部采样策略在读取率为5%及以下时会失效，因为此时尾部很可能包含选择器遗漏的重要token。我们的关键观察是，最简单的代理形式——一个将整个尾部折叠为其计数感知均值的token——在实用选择器下的每个读取率上，都能产生比采样更低的输出误差。我们进一步以闭式形式证明，在低读取率下，采样带来的权重误差几乎是丢弃的两倍，而单个代理的误差则始终低于丢弃。基于这些发现，我们探索用少量针对尾部的代理来近似稠密注意力分布。我们提出了OpenAttention，这是一个由聚类、ErrorGate和top-k选择加上计数感知均值代理组成的流水线。为了高效地实现聚类和选择……

    arXiv:2610.30951v1 Announce Type: new  Abstract: Sparse attention reads only a small subset of the key-value cache, and how to treat the unread tail remains a key challenge. Discarding the tail biases the output, and sampling it as previous approaches do fails at reads of 5% and below, where the tail likely holds heavy hitters the selector missed. Our key observation is that the simplest possible proxy, one token that folds the entire tail into its count-aware mean, already yields lower output error than a sample at every read behind a practical selector. We further show in closed form that at low reads sampling nearly doubles the weight error of discarding while a single proxy stays below it. Building on these findings, we explore approximating the dense attention distribution with a few proxies that target the tail. We propose OpenAttention, a pipeline of clustering, ErrorGate and top-k selection followed by a count-aware mean proxy. To implement clustering and selection both efficie
    
[^205]: Lean Finder V2：为 Mathlib 提供可消除用户查询歧义的语义搜索

    Lean Finder V2: Semantic Search for Mathlib That Disambiguates User Queries

    [https://arxiv.org/abs/2610.30843](https://arxiv.org/abs/2610.30843)

    Lean Finder V2 通过互补的模型级对抗训练与输入级查询重写，实现了能够消除歧义的 Mathlib 语义搜索，有效解决了短而模糊查询的检索难题。

    

    我们提出了 Lean Finder V2，这是一个面向 Mathlib 的语义搜索框架，通过互补的模型级对抗训练和输入级查询重写来消除用户查询的歧义。人类数学家和编码智能体很少通过提供详细上下文来搜索 Mathlib。在实践中，他们查询的是简短的性质、记号、别名或假设，而将大部分预期的数学内容隐含在查询之外。相比之下，现有的 Mathlib 搜索引擎不切实际地假设查询带有丰富的上下文。由于缺乏与真实标准（ground-truth）Mathlib 陈述配对的真实世界短查询，这种歧义问题变得更加难以处理。我们认为这本质上是一个数据问题。通过对抗训练，我们扩大了带标注的短且含歧义查询的生成规模，并在模型级微调和输入级查询重写两个层面上学习消除用户查询的歧义。在人工策划的 MathlibQR 基准测试上，我们的 Lean Finder V2 创造了新的最佳纪录（原文在此处截断）。

    arXiv:2610.30843v1 Announce Type: new  Abstract: We introduce Lean Finder V2, a semantic search framework for Mathlib that disambiguates user queries through complementary model-level adversarial training and input-level query rewriting. Human mathematicians and coding agents rarely search Mathlib by providing detailed context. In practice, they query short properties, notation, aliases, or hypotheses, leaving much of the intended mathematics implicit. In contrast, existing Mathlib search engines unrealistically assume context-rich queries. The scarcity of real-world short queries paired with ground-truth Mathlib statements makes this ambiguity even more intractable. We argue that this is essentially a data problem. Through adversarial training, we scale up the generation of labeled short and ambiguous queries, and learn to disambiguate user queries at both model-level fine-tuning and input-level query rewriting. On the human-curated MathlibQR benchmark, our Lean Finder V2 sets a new s
    
[^206]: 安全探针的缩放定律

    Scaling Laws for Safety Probes

    [https://arxiv.org/abs/2610.30678](https://arxiv.org/abs/2610.30678)

    本文通过构建六个新安全数据集、在20个基础模型上以不同标签规模训练探针，首次提出了可依据模型规模和标签数量提前预测安全探针性能的缩放定律。

    

    安全探针是在模型激活值上训练的、用于实时监控语言模型的标准工具。每当发布新模型或需要监控新的危害领域时，探针都必须重新训练，这需要新的标签以及探针架构的选择。然而，无论是模型还是标签都无法预先告诉我们最终探针的性能如何，要弄清这一点需要训练并比较多种探针配置。在本文中，我们探讨是否可以仅凭模型规模和标签数量来提前预测探针的性能。现有的安全数据集不足以支持这一研究，因为它们要么规模有限，要么已经饱和。因此，我们构建了六个规模足够大、难度足够高的新安全数据集。在这些数据集上，我们在来自五个模型家族的20个基础模型上训练了两种探针架构，每个模型使用32到10⁶个标签。我们使用探针误差来衡量性能……

    arXiv:2610.30678v1 Announce Type: cross  Abstract: Safety probes trained on a model's activations are a standard tool for monitoring language models in real time. Each time a new model is released or a new harm area needs to be monitored, the probe must be retrained, which requires new labels and a choice of probe architecture. Yet neither the model nor the labels tell us in advance how well the resulting probe will perform. Finding out requires training and comparing many probe configurations. In this paper, we ask whether probe performance can instead be predicted in advance from model size and the number of labels. Existing safety datasets are insufficient for this, as they are either limited in scale or already saturated. We therefore construct six new safety datasets of sufficient size and difficulty. On these datasets, we train two probe architectures on 20 base models from five model families, with 32 to $10^{6}$ labels for each model. We measure performance using probe error, d
    
[^207]: 基于局部特征空间精细化的残差校准

    Residual Calibration via Local Feature-Space Refinement

    [https://arxiv.org/abs/2610.30652](https://arxiv.org/abs/2610.30652)

    提出了局部自适应概率估计（LAPE），一种即插即用的事后残差校准框架，它以全局校准器输出为锚点，利用相邻校准样本的非参数局部证据对预测进行精细化，从而捕捉特征空间中局部区域的空间变化失准。

    

    深度神经网络往往难以提供可靠的置信度估计，这使得事后校准对于可靠的决策至关重要。现有的事后校准方法主要依赖于基于分数的变换，这类方法虽然能够改善边际校准，但往往无法捕捉学习到的特征空间中局部区域的、随空间变化的失准现象。为了解决这一问题，我们提出了局部自适应概率估计，这是一种事后残差校准框架。具体而言，LAPE 将全局校准器的输出视为稳定的锚点，并利用来自相邻校准样本的非参数局部证据来对单个预测进行精细化。作为一个即插即用的精细化模块，LAPE 能够有效利用特征空间几何信息，并且可以复用用于构建全局校准器的校准数据。此外，我们通过级联方式将 LAPE 扩展到多视图输入的场景中。

    arXiv:2610.30652v1 Announce Type: cross  Abstract: Deep neural networks often struggle to provide reliable confidence estimates, making post-hoc calibration essential for reliable decision-making. Existing post-hoc methods largely rely on score-based transformations, which improve marginal calibration but often fail to capture spatially varying miscalibration in local regions of the learned feature space. To address this gap, we propose Local Adaptive Probability Estimation (LAPE), a post-hoc residual calibration framework. Specifically, LAPE treats the global calibrator output as a stable anchor and refines the individual prediction by leveraging non-parametric local evidence from neighboring calibration samples. As a plug-and-play refinement module, LAPE effectively utilizes the information about feature-space geometry and can reuse the calibration data that are used to construct the global calibrator. Furthermore, we extend LAPE to the setting with multi-view inputs through Cascade 
    
[^208]: 重新思考SAE评估：原子可解释单元框架揭示隐藏的多义性与冗余性

    Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy

    [https://arxiv.org/abs/2610.30474](https://arxiv.org/abs/2610.30474)

    本文提出了一种基于原子可解释单元的SAE评估新框架，将超越单义性的评估目标形式化，并借此揭示了现有SAE特征中隐藏的多义性与冗余性。

    

    稀疏自编码器（SAE）最初是为发现单义特征而提出的，单义性是刻画单个特征的性质，但并未规定学习到的特征集合整体上应该是什么样子。随着SAE越来越多地被应用于归因分析、回路发现和干预，研究者们开始使用描述特征集合全局性质的评估指标，例如稀疏性、死特征比例、重建保真度和解耦性等，这一系列指标已远远超出单义性的范畴。然而，这种超越单义性的隐式转变尚未被形式化为一个清晰、明确定义的、先于这些评估指标存在的目标。缺少这样的目标，现有指标仍然是临时拼凑的，其必要性缺乏依据。在这项工作中，我们通过为SAE评估引入一种新范式来形式化这一最终目标：要求SAE特征，以及更广义的字典……（摘要不完整）

    arXiv:2610.30474v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) were originally motivated by discovering monosemantic features, a property that characterizes individual features but does not specify what the learned feature set should look like as a whole. As SAEs have increasingly been adopted for attribution, circuit discovery, and intervention, researchers have begun evaluating metrics that describe global properties of the feature set, such as sparsity, dead feature rates, reconstruction fidelity, and disentanglement, a diverse set that extends far beyond monosemanticity. However, this implicit shift beyond monosemanticity has not been formalized as a clear, well-defined objective that comes before these evaluation metrics. Without such an objective, existing metrics remain ad hoc, with their necessity unjustified. In this work, we formalize this ultimate objective by introducing a new paradigm for SAE evaluation: requiring that SAE features, and more generally dictio
    
[^209]: 基于广义矩匹配的差分隐私合成数据

    Differentially Private Synthetic Data via Generalized Moment Matching

    [https://arxiv.org/abs/2610.30196](https://arxiv.org/abs/2610.30196)

    提出了广义矩匹配这一通用框架，通过私有估计矩、拟合概率测度并采样合成记录来生成差分隐私合成数据，可应用于Wasserstein距离、最大均值差异等多种积分概率度量，并在固定维度下达到相差对数因子的极小极大最优误差率。

    

    我们研究了在指定的积分概率度量（IPM）下，针对[-1,1]^d中包含n条记录的任意数据库的(ε,δ)-差分隐私合成数据生成问题。我们提出了广义矩匹配方法，该方法私有地估计矩、拟合概率测度并采样合成记录。当矩误差能够控制目标IPM时，可应用一个一般性误差定理。该框架扩展了针对1-Wasserstein距离和光滑查询的切比雪夫矩匹配方法。我们将该框架应用于采用Matern核、Laplace核以及两层ReLU神经正切核的最大均值差异（MMD），还获得了切片Wasserstein-1以及由范数受限ReLU网络诱导的IPM的保证。在合成记录数量足够多的情况下，对于固定维度，这些误差率在相差对数因子的意义下是极小极大最优的。

    arXiv:2610.30196v1 Announce Type: cross  Abstract: We study $(\varepsilon,\delta)$-differentially private synthetic data for arbitrary databases of $n$ records in $[-1,1]^d$ under a specified integral probability metric (IPM). We introduce generalized moment matching, which estimates moments privately, fits a probability measure, and samples synthetic records. A general error theorem applies when moment error controls the target IPM. This framework extends Chebyshev moment matching for $1$-Wasserstein distance and smooth queries. We apply the framework to maximum mean discrepancy with Matern and Laplace kernels and a two-layer ReLU neural tangent kernel.We also obtain guarantees for sliced Wasserstein-$1$ and the IPM induced by norm-bounded ReLU networks. With sufficiently many synthetic records,these rates are minimax optimal up to logarithmic factors for fixed dimension.
    
[^210]: 非平稳多臂老虎机中现时效应推断的代价

    The Price of Current-Effect Inference in Nonstationary Multi-Armed Bandits

    [https://arxiv.org/abs/2610.29982](https://arxiv.org/abs/2610.29982)

    该论文揭示了非平稳多臂老虎机中收益最大化与现时效应推断之间的内在权衡，证明了以有限宽度置信区间同时覆盖所有现时效应所需付出的动态遗憾极小极大代价，并提出 TRACE 算法来同时兼顾这两个目标。

    

    自适应实验利用老虎机算法来偏好具有更高期望收益的干预措施。然而，决策者还需要对现时处理效应进行可靠的推断：即在当前时刻干预措施与参照臂平均收益之间的差异，而非历史平均。这两个目标相互冲突：收益最大化倾向于选择更优的干预，而现时效应推断则要求随着结果随时间变化，持续探索可能次优的干预。我们针对平均收益具有指数 β∈(0,1] 和常数 L 的 Hölder 平滑性的 K 臂老虎机刻画了这一权衡。在要求置信区间以不超过 w 的宽度、以高概率同时覆盖所有现时效应的条件下，我们在期望动态遗憾上建立了（在对数因子范围内）匹配的极小极大界。在时间跨度 T 内，我们的算法 TRACE（Tracking Rewards and Current Effects）实现了遗憾……（原文摘要至此截断）

    arXiv:2610.29982v1 Announce Type: cross  Abstract: Adaptive experiments use bandit algorithms to favor interventions with higher expected rewards. However, decision-makers also require reliable inference on current treatment effects: differences between intervention and reference-arm mean rewards at the present time, rather than historical averages. These objectives conflict: reward maximization favors better interventions, whereas current-effect inference requires continued exploration of potentially suboptimal interventions as outcomes change over time. We characterize this tradeoff for $K$-arm bandits with H\"older-smooth mean rewards of exponent $\beta\in(0,1]$ and constant $L$. Requiring high-probability simultaneous coverage of all current effects by confidence intervals of width at most $w$, we establish matching minimax bounds, up to logarithmic factors, on expected dynamic regret. Over horizon $T$, our algorithm TRACE (Tracking Rewards and Current Effects) achieves regret of o
    
[^211]: 再生核希尔伯特空间中的范数最优模型降阶

    Norm-optimal model order reduction in reproducing kernel Hilbert spaces

    [https://arxiv.org/abs/2610.28224](https://arxiv.org/abs/2610.28224)

    本文建立了再生核希尔伯特空间中的范数最优模型降阶框架，通过引入广义Malmquist-Takenaka函数与可分RKHS间的酉变换，将最优性条件和相应算法从可微核函数扩展到了不可微核函数的情形。

    

    我们考虑再生核希尔伯特空间（RKHSs）中的模型降阶（MOR）问题。我们证明了范数最优降阶模型的著名必要插值条件的推广形式，这些条件最初是针对 $H_2$ Hardy-Hilbert 空间中的有理函数提出的。我们为具有可微核函数的 RKHS 构建了迭代有理 Krylov（IRKA）算法的替代方法，以获得范数最优的降阶模型。随后，我们引入了广义 Malmquist-Takenaka（GMT）函数，该函数可用于在任意 RKHS 中定义标准正交基。利用 GMT 函数，我们提出了可分 RKHS 之间的酉变换。这使得我们能够将最优性条件和后续算法扩展到在具有不可微核函数的 RKHS 中提出的 MOR 问题。所提出的 MOR 框架的潜力和通用性在多个数值实验中得到了验证。我们在各种……上执行范数最优模型降阶。

    arXiv:2610.28224v1 Announce Type: cross  Abstract: We consider the problem of model order reduction (MOR) in reproducing kernel Hilbert spaces (RKHSs). We prove generalizations of well-known necessary interpolatory conditions for norm-optimal reduced-order models, originally stated for rational functions in $H_2$ Hardy-Hilbert spaces. We construct an alternative to the iterative rational Krylov (IRKA) algorithm for RKHSs with differentiable kernels to obtain norm-optimal reduced-order models. Then, we introduce generalized Malmquist-Takenaka (GMT) functions that can be used to define orthonormal bases in an arbitrary RKHS. Using GMT functions, we propose unitary transformations between separable RKHSs. This allows us to extend our optimality conditions and subsequent algorithms to MOR problems posed in RKHSs with non-differentiable kernels. The potential and versatility of the proposed MOR framework is demonstrated in several numerical experiments. We perform norm-optimal MOR on variou
    
[^212]: 有吸引力的面孔会使文生图奖励模型和排行榜产生偏差

    Attractive Faces Skew Text-to-Image Reward Models and Leaderboards

    [https://arxiv.org/abs/2610.28195](https://arxiv.org/abs/2610.28195)

    研究发现人脸吸引力在人类偏好判断中会压倒图像质量本身，这一偏差会扭曲文生图模型的奖励模型和排行榜，使AI生成的有吸引力面孔被排在真实但不吸引人的面孔之上。

    

    人类偏好现已成为文生图模型评估、引导、微调和排名的核心。然而，人类偏好对图像生成中偏差的易感性仍未得到充分探索。我们发现了一个关键观察：对于人脸而言，感知到的吸引力在偏好判断中会压倒图像质量，即使在比较保真度完全相同的真实照片时也是如此。在一项受控用户研究中，标注者将有吸引力的面孔评为更高质量，即使不吸引人的真实面孔与有吸引力的AI生成面孔进行比较时也是如此。我们研究了这种偏差的下游影响。首先，基于用户研究数据模拟Elo排名显示，吸引力可以逆转排行榜的排名，使合成的有吸引力面孔排在真实的、不吸引人的面孔之上。其次，同样的偏差在人类偏好模型中再次出现。通过单独编辑40个标注的面部属性，我们发现吸引力是最强的（影响因素）……

    arXiv:2610.28195v1 Announce Type: cross  Abstract: Human preference is now central to the evaluation, guidance, finetuning, and ranking of text-to-image models. However, the susceptibility of human preferences to bias in image generation remains underexplored. We identify a key observation: for human faces, perceived attractiveness overrides image quality in preference judgments, even when comparing real photographs of identical fidelity. In a controlled user study, annotators rank more attractive faces as higher quality, even when unattractive real faces are compared with attractive AI-generated ones. We examine the downstream consequences of this bias. Firstly, simulating Elo rankings from user study data shows attractiveness can invert a leaderboard's ranking, elevating synthetic attractive faces over real, unattractive ones. Secondly, the same bias reappears in the human preference models. Editing 40 annotated facial attributes in isolation, we find that attractiveness is the stron
    
[^213]: 缩放定律的基础

    Foundations of the Scaling Law

    [https://arxiv.org/abs/2610.27789](https://arxiv.org/abs/2610.27789)

    本论文通过几何近似、谱分析和约束优化三个相互关联的分析框架，为缩放定律建立了理论基础，并区分解释了涌现、顿悟和测试时缩放等现象。

    

    经验缩放定律将验证损失与模型规模、训练数据量和计算量联系起来，为资源分配提供了定量基础。本技术报告通过三个相互关联的分析来解释这些关系：几何近似将空间分辨率、内在维度和平滑度与函数误差联系起来，而损失函数决定了该误差如何传递为任务风险；谱分析结合特征可见性、协方差特征值和目标对齐，描述了更多数据如何减少尚未解决的预测不确定性；约束优化则给出了参数与数据的平衡以及训练与推理之间的权衡。我们还研究了涌现与顿悟现象，将评分阈值与学习动力学区分开来；并研究了测试时缩放，将候选答案的覆盖范围与所选答案的准确率区分开来。在视觉和分子预测上的应用展示了领域特定的坐标选择如何改变资源比较方式。

    arXiv:2610.27789v1 Announce Type: new  Abstract: Empirical scaling laws relate validation loss to model size, training exposure and compute, providing a quantitative basis for allocating resources. This technical note explains these relationships through three linked analyses. Geometric approximation connects spatial resolution, intrinsic dimension and smoothness to function error; the loss function determines how that error transfers to task risk. Spectral analysis combines feature visibility, covariance eigenvalues and target alignment to describe how additional data reduce unresolved predictive mass. Constrained optimization then gives the parameter-data balance and the training-inference trade-off. We also examine emergence and grokking, separating scoring thresholds from learning dynamics, and test-time scaling, separating candidate coverage from selected-answer accuracy. Applications to vision and molecular prediction show how domain-specific coordinates change the resource compa
    
[^214]: 经验去噪器的隐私核算

    Privacy Accounting for Empirical Denoisers

    [https://arxiv.org/abs/2610.27710](https://arxiv.org/abs/2610.27710)

    本文提出一种无需训练的截断隐私可问责经验去噪器，通过显式计算其每步敏感性，使扩散模型采样过程中的每一步去噪都可建模为高斯机制，从而实现对采样过程本身隐私损失的形式化核算。

    

    扩散模型已成为现代生成系统的基础，但其强大的记忆能力引发了隐私方面的担忧。虽然差分隐私（DP）训练能够提供形式化的保证，但其代价是样本质量的下降。在本工作中，我们研究了一个与之互补的问题：在采样过程本身中会产生多少隐私损失？直接分析现代神经去噪器是具有挑战性的，因为它们对单个训练样本的敏感性通常是难以计算的。为了克服这一挑战，我们构建并分析了一种经过截断的隐私可问责经验去噪器——这是一种无需训练的贝叶斯最优去噪器估计器，其每一步的敏感性都可以被显式计算，从而使每个去噪步骤都能成为一个高斯机制。这一视角使得在以下情况下进行隐私核算成为可能：当每个训练样本对每个去噪步骤都有贡献时，可在高斯DP下进行隐私核算；当每个……（摘要在此处被截断）

    arXiv:2610.27710v1 Announce Type: new  Abstract: Diffusion models have emerged as the foundation of modern generative systems, yet their high memorization capacity raises privacy concerns. While differentially private (DP) training provides formal guarantees, it comes at a cost in sample quality. In this work, we study a complementary question: how much privacy loss is incurred during the sampling process itself? Directly analyzing modern neural denoisers is challenging because their sensitivity to a single training example is generally intractable. To overcome this challenge, we construct and analyze a clipped privacy-accountable empirical denoiser, a training-free estimator of the Bayes-optimal denoiser, whose per-step sensitivity can be computed explicitly, allowing each denoising step to be a Gaussian mechanism. This perspective enables privacy accounting under Gaussian DP when each training example contributes to each denoising step, and under $f$-DP trade-off functions when each 
    
[^215]: CyberHull：基于凸包的确定性恶意URL隔离方法

    CyberHull: Deterministic Malicious URL Isolation via Convex Hull

    [https://arxiv.org/abs/2610.27675](https://arxiv.org/abs/2610.27675)

    提出CyberHull框架，利用QuickHull算法在多个正交特征域上构建凸包边界，以资源高效、确定性的方式实现恶意URL的识别与隔离，训练复杂度仅为O(n log n)。

    

    日益复杂且具有抗规避能力的钓鱼攻击策略的出现，对现代网络安全构成了重大威胁，尤其是在信息物理系统和基于互联网的网络中。传统的概率方法和某些机器学习方法往往难以检测新型攻击，并且可能缺乏可解释性或可移植性。为应对这些挑战，我们提出了CyberHull，这是一种资源高效且具有确定性的框架，通过在多个正交特征域（包括结构内容熵、超链接图拓扑和语义标记分布）上构建数学定义的凸包边界来识别恶意URL。CyberHull使用QuickHull算法构建紧凑的凸包（称为“金库”/vaults），在适当假设下其训练复杂度为O(n log n)。URL分类通过多面体内部点测试来执行

    arXiv:2610.27675v1 Announce Type: cross  Abstract: The emergence of increasingly sophisticated and evasion-resistant phishing attack strategies represents a significant threat to modern cybersecurity, particularly in cyber-physical systems and internet-based networks. Traditional probabilistic methods and certain machine learning approaches often struggle to detect novel attacks and may lack interpretability or portability. To address these challenges, we introduce CyberHull, a resource-efficient and deterministic framework for identifying malicious URLs through mathematically defined convex hull boundaries across multiple orthogonal feature domains, including structural content entropy, hyperlink graph topology, and semantic token distribution. CyberHull constructs compact convex hulls, referred to as vaults, using the QuickHull algorithm, with a training complexity of O(n log n) under suitable assumptions. URL classification is performed through point-in-polyhedron tests that impose 
    
[^216]: MLIPS：多层索引器池共享

    MLIPS: Multi-Layer Indexer Pool Sharing

    [https://arxiv.org/abs/2610.27438](https://arxiv.org/abs/2610.27438)

    MLIPS是一种无需训练的方法，通过让一个生产者层发布超出预算的top-M候选池供后续多个层共享复用，在保留各层独立打分与选择的同时消除重复的候选搜索，在DeepSeek-V4-Flash上实现了1.14–1.40倍加速和14.8–27.4%的GPU能耗降低，且准确率几乎不受影响。

    

    稀疏注意力限制了每个查询所使用的键的数量，但寻找这些键仍可能需要在每一层对完整前缀进行扫描。我们提出了MLIPS（多层索引器池共享），这是一种无需训练的方法，它在共享候选搜索的同时，保留每个使用方自己的打分与选择。生产者会发布一个大于注意力预算K的top-M候选池；后续各层从该池中选出自己的top-K。在三个稀疏注意力模型家族上的测量表明，宽泛候选池所提供的覆盖率显著高于最终选择之间的重叠程度。我们在vLLM中为DeepSeek-V4-Flash实现了逐层共享、持久解码复用以及粗粒度生产者。在四个长上下文测试套件上，每个生产者对应三个使用层的设置可带来1.14–1.40倍的加速和14.8–27.4%的GPU能耗降低，单次运行中观察到的准确率变化为−1.49至+0.29个百分点。更广泛的共享能够…

    arXiv:2610.27438v1 Announce Type: cross  Abstract: Sparse attention limits the keys used by each query, but finding those keys can still require a full-prefix scan at every layer. We introduce \textbf{MLIPS (Multi-Layer Indexer Pool Sharing)}, a training-free method that shares candidate search while retaining each consumer's own scoring and selection. A producer publishes a top-$M$ pool larger than the attention budget $K$; subsequent layers select their own top-$K$ from that pool. Measurements across three sparse-attention model families show substantially higher coverage from broad pools than overlap between final selections. We implement layerwise sharing, persistent decode reuse, and a coarse producer in vLLM for DeepSeek-V4-Flash. Across four long-context suites, three consumer layers per producer give $1.14$--$1.40\times$ speedups and $14.8$--$27.4\%$ lower GPU energy, with observed accuracy changes of $-1.49$ to $+0.29$ percentage points in single runs. More extensive sharing e
    
[^217]: 未知边缘分布下严格可复制可实现学习的二次样本复杂度

    Quadratic Sample Complexity for Strictly Replicable Realizable Learning with Unknown Marginals

    [https://arxiv.org/abs/2610.26944](https://arxiv.org/abs/2610.26944)

    该论文首次证明，在分布及其边缘均未知的严格可复制可实现学习中，样本复杂度具有二次方下界Ω(d²/(ερ²))，并通过泊松化球采样算法为完整布尔类P_d给出了匹配的Θ(d²/(ερ²)+ε⁻¹log(1/δ))上界，完整刻画了该问题的复杂度。

    

    在量词∃A∀D的框架下，即可实现分布及其边缘分布均未知的情况下，我们确定了严格可复制性（即两个独立样本和一个共享随机种子必须产生完全相同的函数）的有限类样本复杂度指数。当ε=δ=ρ=10⁻⁴时，在|H|≤M上的最坏情况复杂度为Θ((log M)²)，且不含迭代对数项。对于d≥1、0<ε≤1/16以及0<δ,ρ≤10⁻³，任何在整个显式锚定可实现类上有效的随机学习器（即使允许非proper学习）都需要Ω(d²/(ερ²))个样本。对于完整布尔类P_d（d≥2），在相同的参数范围内，我们借助一个proper的泊松化球采样算法证明了匹配的刻画结果Θ(d²/(ερ²)+ε⁻¹log(1/δ))。此外，我们还为每个有限可测布尔类给出了显式的proper Gibbs算法上界。

    arXiv:2610.26944v1 Announce Type: cross  Abstract: Under the quantifier $\exists A\,\forall D$, where the realizable distribution and its marginal are unknown, we determine the finite-class sample-complexity exponent for strict replicability: two independent samples and one shared seed must produce identical functions. For $\varepsilon=\delta=\rho=10^{-4}$, the worst case over $|H|\le M$ is $\Theta((\log M)^2)$, without iterated logarithms. For $d\ge1$, $0<\varepsilon\le1/16$, and $0<\delta,\rho\le10^{-3}$, every randomized, possibly improper learner valid throughout an explicit anchored realizable family requires $\Omega(d^2/(\varepsilon\rho^2))$ samples. For the full Boolean class $P_d$, $d\ge2$, in the same parameter range, we prove the matching characterization $\Theta(d^2/(\varepsilon\rho^2)+\varepsilon^{-1}\log(1/\delta))$ using a proper Poissonized ball-sampling algorithm. We also give an explicit proper Gibbs upper bound for every finite measurable Boolean class. Our lower boun
    
[^218]: 针对具有混合狄利克雷-诺伊曼边界条件的奇异椭圆偏微分方程的混合变分-配置物理信息神经网络中基于残差的自适应细化方法

    Residual-Based Adaptive Refinement in Hybrid Variational-Collocation PINNs for Singular Elliptic PDEs with Mixed Dirichlet-Neumann Boundary Conditions

    [https://arxiv.org/abs/2610.26458](https://arxiv.org/abs/2610.26458)

    提出一种混合变分-配置物理信息神经网络框架，通过基于残差的自适应配置点细化策略，有效捕捉混合狄利克雷-诺伊曼边界条件下椭圆偏微分方程在凹角和裂纹尖端附近的解奇异性。

    

    对于在非凸或含裂纹区域上、具有混合狄利克雷-诺伊曼边界条件的二阶椭圆偏微分方程问题，其解通常不属于 $H^2(\Omega)$ 空间。在凹角、裂纹尖端以及边界条件类型发生改变的点附近，梯度以 $r^{\lambda-1}$ 的形式增长，其中 $\lambda<1$。我们研究了混合变分-配置物理信息神经网络（PINNs）中基于残差的自适应细化方法。该方法采用多尺度傅里叶特征网络，在复合损失函数上进行训练。该损失函数结合了三项：用勒让德气泡函数检验的变分残差、强形式配置项以及对狄利克雷和诺伊曼边界数据的惩罚项。变分项仅需网络的一阶导数，而配置项需要二阶导数，并提供了一个可在任意点评估的残差。在每个训练阶段结束后，由该残差筛选出的候选点被添加到配置点集中。该方法在五个二维……（原文截断）上进行了测试。

    arXiv:2610.26458v2 Announce Type: replace-cross  Abstract: Solutions of second-order elliptic PDE problems on nonconvex or cracked domains with mixed Dirichlet-Neumann conditions typically lie outside $H^2(\Omega)$. Near re-entrant corners, crack tips and points where the boundary condition changes type, the gradient grows like $r^{\lambda-1}$ with $\lambda<1$. We study residual-based adaptive refinement in hybrid variational-collocation PINNs. A multiscale Fourier-feature network is trained on a composite loss. The loss combines a variational residual tested against Legendre bubble functions, a strong-form collocation term and penalties for the Dirichlet and Neumann data. The variational term needs only first derivatives of the network, whereas the collocation term requires second derivatives and provides a residual that can be evaluated at any point. After each training stage, candidate points selected by this residual are added to the collocation set. The method is tested on five tw
    
[^219]: EvoSkill v2：从基准适配到有状态演化器的全智能体技能演化

    EvoSkill v2: Fully Agentic Skill Evolution from Benchmark Adaptation to a Stateful Evolver

    [https://arxiv.org/abs/2610.25931](https://arxiv.org/abs/2610.25931)

    EvoSkill v2 提出了一种由三个智能体组成的技能演化系统，其核心创新在于采用一个贯穿全程、带有 bash 工具的有状态编码智能体在每个训练周期持续重写技能指令，并结合多次尝试与无筛选接受策略，在不改动底层模型的前提下提升智能体的基准测试表现。

    

    EvoSkill v2 是一个三智能体系统，它通过编辑智能体所获得的指令（技能）来提升性能，同时保持其底层模型和运行框架不变。用户通过描述一个基准测试并提供一个评分器来启动运行，基准适配器会将其转换为针对该基准的专用插件。求解器尝试完成任务，演化器则读取求解器的训练轨迹并在每个周期重写技能。我们将演化器实现为一个带有 bash 工具、在整个运行过程中持续存在的有状态编码智能体，这不同于我们所调研的技能优化器中最常见的两种设计：如 SkillOpt 那样的一系列单次模型调用，以及如 SkillBoost 那样每轮从零开始的智能体。我们还测量了求解器的轨迹在多次尝试之间的变化程度，为演化器提供每个训练任务的多次尝试机会，并直接接受每个周期产生的技能，而不是依据嘈杂的分数在其中进行筛选。在细粒度真实标签的条件下，EvoSkill v2 的技能将……（摘要原文在此处截断）

    arXiv:2610.25931v2 Announce Type: replace  Abstract: EvoSkill v2 is a three-agent system that edits the instructions (skills) an agent is given, leaving its model and harness intact. The user starts a run by describing a benchmark and supplying a grader, which a benchmark adaptor turns into a benchmark-specific plugin. A solver attempts the tasks, and an evolver reads the solver's training rollouts and rewrites the skills at each epoch. We implement the evolver as one stateful coding agent with a bash tool for the whole run, unlike the two most common designs among the skill optimizers we surveyed: a sequence of single model calls, as in SkillOpt, and an agent that starts afresh each round, as in SkillBoost. We also measure how much the solver's rollouts vary from one attempt to the next, give the evolver several attempts per training task, and accept every epoch's skills instead of selecting among them on noisy scores. With fine-grained ground truth, EvoSkill v2's skills raised the he
    
[^220]: Transformers 中的各向异性表示动力学

    Anisotropic Representation Dynamics in Transformers

    [https://arxiv.org/abs/2610.23242](https://arxiv.org/abs/2610.23242)

    该论文提出了混合门控的各向异性表示动力学理论，将自注意力分解为数值传递、查询门控与键门控，揭示了门控机制可诱导表示差异从收缩切换为扩张的状态转换，并给出了判定该现象的充分必要谱判据。

    

    内部表示之间的微小差异在穿过 Transformer 层时可能会缩小或扩大，这会影响信息的保留方式以及模型对扰动的敏感程度。然而，注意力权重的变化何时能够将原本会缩小的表示差异转变为扩大的差异，这一点仍不清楚。我们发展了一套混合门控的各向异性表示动力学理论，将自注意力分解为数值传递、查询门控和键门控三个组成部分。我们的分析揭示了被注意力关注的表示的几何结构如何约束门控行为，以及这些机制之间的相互作用如何能够逆转由数值传递引起的收缩。我们将这一现象形式化为门控诱导的状态切换：表示差异在固定的注意力路由下收缩，但在完全注意力下扩张。我们推导出一个充分必要的谱判据，用以证明该现象的存在性或……

    arXiv:2610.23242v1 Announce Type: new  Abstract: Small differences between internal representations can shrink or grow as they pass through a Transformer layer, affecting how information is preserved and how sensitive the model is to perturbations. Yet it remains unclear when changes in attention weights can turn a representation difference that would otherwise shrink into one that grows. We develop a \textbf{Mixture-Gated Theory of Anisotropic Representation Dynamics} that separates self-attention into value transport, query gating, and key gating. Our analysis reveals how the geometry of attended representations constrains gating and how interactions between these mechanisms can reverse contraction induced by value transport. We formalize this phenomenon as gating-induced regime switching, where a representation difference contracts under fixed attention routing but expands under full attention. We derive a necessary and sufficient spectral criterion that certifies the existence or a
    
[^221]: SpectrumQuant：面向大语言模型量化的频谱对齐方法

    SpectrumQuant: Spectrum Alignment for LLM Quantization

    [https://arxiv.org/abs/2610.20303](https://arxiv.org/abs/2610.20303)

    提出SpectrumQuant，一种基于协方差最大化的闭式正交旋转方法，通过频谱对齐构建量化友好的变换，从而提升大语言模型训练后低比特量化的效果。

    

    正交变换已成为大语言模型（LLM）低比特量化的有效工具，因为它能够在不改变底层模型函数的情况下重塑激活分布。现有方法通常依赖于固定的、与数据无关的变换（如Hadamard旋转），或者依赖于在校准阶段对可学习旋转的优化。我们从第一性原理出发研究旋转选择问题，并证明最大化协方差可以带来频谱对齐，从而为构建量化友好的正交变换提供了一个有理论依据的准则。基于这一洞察，我们提出了SpectrumQuant，这是一种闭式解的旋转方法，它从校准数据中估计激活协方差，并利用它构建协方差感知的正交变换以用于训练后量化。我们的公式化将量化友好的旋转刻画为对齐分量与混合分量的组合，同时将Hadamard旋转作为特例恢复出来（此句原文被截断）。

    arXiv:2610.20303v1 Announce Type: new  Abstract: Orthogonal transforms have emerged as an effective tool for low-bit quantization of large language models (LLMs), as they reshape activation distributions without changing the underlying model function. Existing approaches typically rely either on fixed data-agnostic transforms, such as Hadamard rotations, or on calibration-time optimization of learned rotations. We study rotation selection from first principles and show that maximizing covariance leads to spectral alignment and provides a principled criterion for constructing quantization-friendly orthogonal transforms. Based on this insight, we propose SpectrumQuant, a closed-form rotation method that estimates activation covariance from calibration data and uses it to construct a covariance-aware orthogonal transform for post-training quantization. Our formulation characterizes quantization-friendly rotations as compositions of alignment and mixing components, while recovering Hadamar
    
[^222]: 跨模型上下文内神经反馈揭示大语言模型中的（有限）激活控制能力

    Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs

    [https://arxiv.org/abs/2610.20179](https://arxiv.org/abs/2610.20179)

    本文提出跨模型上下文内神经反馈新范式，首次将模型在任务上的胜任能力与对自身激活的真实访问区分开来，测试十个开源大模型后发现大语言模型仅具备有限的激活控制能力。

    

    基于激活向量的探针被广泛用于检测语言模型的不良行为，但其有效性依赖于一个假设：模型无法控制其激活所编码的内容。人类可以通过神经反馈学会控制甚至单个神经元的放电，近期有研究探讨了这种能力是否也适用于语言模型，但得出了相互矛盾的结论。迄今为止，尚无实验设计能够将模型在任务上的表现能力与其对自身激活的真实访问区分开来。我们通过“上下文内神经反馈”会话填补了这一空白：在其中，一个“受试”模型被要求最大化（或最小化）一个由其自身激活计算得出的分数，且该分数会在每一轮对话中展示给该模型。在我们的第一个实验中，句子要么由受试模型自己生成，要么由第二个“写作”模型生成，而分数始终来自受试模型的激活。我们测试了来自五个系列的十个开源权重模型，规模从9B到32B。部分模型……（原文摘要在此处被截断）

    arXiv:2610.20179v1 Announce Type: new  Abstract: Probes on activation vectors are widely used to detect misbehavior in language models, but their success relies on the assumption that models cannot control what their activations encode. Humans can learn, through neurofeedback, to control the firing of even single neurons, and recent studies investigated whether this ability extends to language models, reaching conflicting conclusions. No experimental design so far disentangles competence at the task from genuine access to one's own activations. We fill this gap with in-context neurofeedback sessions, in which a "subject" model is asked to maximize (or minimize) a score derived from its activations, shown back to it at every turn. In our first experiment, the sentences are generated either by the subject or by a second "writer" model, while the score always derives from the activations of the subject. We test ten open-weight models from five families, with sizes from 9B to 32B. Some mod
    
[^223]: 交互即裁判

    Interaction-as-a-Judge

    [https://arxiv.org/abs/2610.17895](https://arxiv.org/abs/2610.17895)

    提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。

    

    编程智能体的任务是产出复杂的交互式应用，例如网站、桌面应用和游戏。人类在评估这些应用时会以交互方式进行，而当前自动化的LLM评估主要集中于代码分析和单元测试。这种不匹配会导致LLM评判者遗漏那些对人类用户而言一目了然的缺陷。我们研究了“交互即裁判”，这是一种由计算机操作智能体（CUA）直接与应用进行交互的评估范式。我们证明“交互即裁判”能够同时改进交互式应用的评估与生成。为此，我们收集并发布了两个互补的数据集：包含人类偏好及对应智能体操作轨迹的6,858对应用，以及带有经人类验证的需求标签的222个应用。相较于标准的非交互式方法，我们发现“交互即裁判”在预测h……（原文摘要在此处截断）

    arXiv:2610.17895v2 Announce Type: cross  Abstract: Coding agents are tasked with producing sophisticated interactive applications such as websites, desktop applications, and games. While humans assess these applications interactively, current automated LLM evaluations primarily focus on code analysis and unit tests. This mismatch can cause LLM judges to miss failures that are immediately apparent to human users. We study Interaction-as-a-Judge, an evaluation paradigm where computer-use agents (CUAs) directly interact with an application. We demonstrate that Interaction-as-a-Judge can improve both evaluation and generation of interactive applications. To this end, we collect and release two complementary datasets: $6,858$ pairs of applications with human preferences and corresponding agent traces, and $222$ applications with human-verified requirement labels. Relative to standard non-interactive approaches, we find that Interaction-as-a-Judge improves evaluation in terms of predicting h
    
[^224]: 重心引导：将基础图像编辑器转变为连续情感控制器

    Barycentric Guidance: Turning Foundation Image Editors into Continuous Affective Controllers

    [https://arxiv.org/abs/2610.16892](https://arxiv.org/abs/2610.16892)

    提出无需训练的“重心引导”框架，通过将任意目标效价-唤醒度点映射为基于单纯形的引导权重，把预训练基础图像编辑器转变为可精确连续控制情绪的情感控制器，且无需微调、额外数据或模型修改。

    

    情绪是在一个连续体上表达的，天然地可以用效价-唤醒度空间来描述。现有的基于VA的图像编辑器通常依赖训练且局限于特定领域，这限制了它们只能覆盖训练时见过的VA区域，并使其在不同数据集和领域之间表现脆弱。与此同时，尽管最先进的基础图像编辑器能够产生高质量的编辑结果，但它们并非连续的情感控制器：文本编码器中的非线性情感表示使得提示词插值难以准确校准，因此精确且连续的VA控制仍然不可靠。为填补这一空白，我们提出了重心引导，这是一个无需训练的框架，能够将预训练的基础图像编辑器转变为对VA具有连续控制能力的情感控制器，无需微调、额外数据或模型修改。我们的方法将任意目标VA点映射为基于单纯形的引导权重，从三个方面实现VA可控性：（i）精确瞄准

    arXiv:2610.16892v2 Announce Type: replace-cross  Abstract: Emotions are expressed on a continuum and are naturally described in the valence-arousal (VA) space. Existing VA-based image editors are typically training-based and domain-specific, limiting them to VA regions seen during training and making them brittle across datasets and domains. Meanwhile, despite producing high-quality edits, state-of-the-art foundation image editors are not continuous affective controllers: non-linear emotion representations in text encoders makes prompt interpolation poorly calibrated, so accurate and continuous VA control remains unreliable. To fill this gap, we introduce Barycentric Guidance, a training-free framework that turns pretrained foundation image editors into continuous affective controllers over VA, requiring no finetuning, extra data, or model changes. Our method maps any target VA point to simplex-based guidance weights, enabling VA controllability in three aspects: (i) accurate targeting
    
[^225]: 协变量偏移下的潜在空间重要性加权

    Latent-Space Importance Weighting under Covariate Shift

    [https://arxiv.org/abs/2610.15571](https://arxiv.org/abs/2610.15571)

    该论文提出一种针对协变量偏移的迁移学习方法，在低维潜在空间中将表示学习与样本重加权相结合，并给出了目标域超额风险率等理论保证与实验验证。

    

    在本工作中，我们提出了一种针对协变量偏移的迁移学习算法，将样本重加权与表示学习相结合。我们方法的一个关键特征是使用低维潜在空间进行建模，这与直接在高维环境空间中操作的传统方法形成对比。我们的方法通过估计学习到的潜在特征的密度比来进行下游的加权预测。对于具有正则学习空间预测分数和密度比的严格充分学习表示，我们利用独立的数据划分建立了明确的目标域超额风险率。我们在给定的平滑性条件下给出了维度比较，并对真实权重和固定预测器进行了方差比较。一般的有限样本不等式保留了表示质量误差。在模拟数据集和真实世界数据集上的实验表明，结合学习到的表示与潜在加权能够带来性能提升。

    arXiv:2610.15571v1 Announce Type: new  Abstract: In this work, we propose a transfer learning algorithm for covariate shift, integrating sample reweighting with representation learning. A key feature of our approach is the use of a lower-dimensional latent space for modeling, in contrast to conventional methods that operate directly in the high-dimensional ambient space. Our method estimates the density ratio of learned latent features for downstream weighted prediction. We establish explicit target-domain excess-risk rates with independent data folds for strictly sufficient learned representations with regular learned-space prediction scores and density ratios. We give dimension comparisons under stated smoothness conditions and a variance comparison for true weights and a fixed predictor. General finite-sample inequalities retain representation-quality errors. Experiments on simulated and real-world datasets demonstrate gains from combining learned representations and latent weightin
    
[^226]: 面向处理效应估计的预测驱动数据融合方法

    Prediction-Powered Data Fusion for Treatment Effect Estimation

    [https://arxiv.org/abs/2610.12332](https://arxiv.org/abs/2610.12332)

    提出了一种无需对观察性研究做特殊假设的数据融合框架，通过保持RCT估计的无偏性并从大型观察性研究中借力，显著提升平均处理效应（ATE）和条件平均处理效应（CATE）的估计精度。

    

    随机对照试验（RCT）能够在无混杂的情况下识别处理效应，但样本量通常较小；而观察性研究（OBS）样本量大，但可能存在混杂。目前已有许多将小型RCT与大型OBS相结合的估计方法用于估计平均处理效应（ATE）和条件平均处理效应（CATE）。然而，现有的ATE估计方法要么对OBS施加额外假设，要么未能从OBS中充分借力。相比ATE，CATE的研究相对较少，现有的CATE方法要么假设OBS无混杂，要么依赖于混杂函数的模型，要么以引入偏差为代价来换取更低的方差。为此，我们提出了一个框架，在对OBS不做任何特殊假设的前提下融合OBS与RCT：既保持基于RCT估计的无偏性，又从大型OBS中借力以提升估计精度。基于这一原则，我们构建了具有闭式权重的ATE估计器AIPW-Fusion……

    arXiv:2610.12332v1 Announce Type: cross  Abstract: Randomized controlled trials (RCTs) identify treatment effects without confounding but are often small, whereas observational studies (OBS) are large but may be confounded. Many estimators combining a small RCT with a large OBS have been developed for the average treatment effect (ATE) and the conditional ATE (CATE). However, existing ATE estimators either make assumptions on the OBS or do not borrow enough power from them. The CATE has been studied less than the ATE. Existing CATE methods either assume the OBS are unconfounded, rely on a model of the confounding function, or accept bias in exchange for lower variance. We therefore propose a framework that, without special assumptions on the OBS, fuses the OBS and the RCT by preserving the unbiasedness of RCT-based estimation while borrowing power from the large OBS to boost precision. Applying this principle, we build an ATE estimator, AIPW-Fusion, with closed-form weights and confide
    
[^227]: 一种保持结构的聚合物电解质离子神经密度泛函

    A structure-preserving neural density functional for the ions of a polymer electrolyte

    [https://arxiv.org/abs/2610.12132](https://arxiv.org/abs/2610.12132)

    开发了一种保持空间对称性、热力学可积性和诺特定理恒等式的神经密度泛函，仅需平面密度和内力数据训练即可准确预测聚合物电解质离子在不同浓度和尺度下的结构与关联行为。

    

    预测非均匀聚合物电解质的结构和响应，需要对离子关联的描述既保持分子尺度的精度，又能在不同空间尺度和几何构型之间保持可迁移性。我们开发了一种面向电解质的神经密度泛函，它保持空间对称性、热力学可积性以及诺特定理恒等式，并在稳定、非临界的体相状态下恢复完美屏蔽行为。其非线性的密度依赖性捕捉到了配对封闭近似所遗漏的浓度依赖关联，包括在强耦合条件下长波长数密度涨落从增强到抑制的转变。该泛函能够描述未经训练的盐浓度下的密度分布，并预测体相结构因子和长波长数密度响应。仅通过分子动力学模拟得到的平面密度分布和内力分布进行训练，该泛函即可预测更大区域内离子的结构。

    arXiv:2610.12132v1 Announce Type: cross  Abstract: Predicting the structure and response of inhomogeneous polymer electrolytes requires a description of ion correlations that retains molecular-scale accuracy while remaining transferable across spatial scales and geometries. We develop a neural density functional for electrolytes that preserves spatial symmetries, thermodynamic integrability and the Noether identities, with perfect screening recovered in stable, noncritical bulk states. Its nonlinear density dependence captures the concentration-dependent correlations missed by a pair closure, including a crossover from enhanced to suppressed long-wavelength number fluctuations at strong coupling. The functional describes density profiles at an untrained salt concentration and predicts bulk structure factors and the long-wavelength number response. Trained solely on planar density and internal-force profiles from molecular dynamics, the functional predicts ionic structure in larger doma
    
[^228]: 面向信贷风险建模的排序先验对齐：外部先验何时起作用？

    Ranking Prior Alignment for Credit Risk Modeling: When Do External Priors Matter?

    [https://arxiv.org/abs/2610.11146](https://arxiv.org/abs/2610.11146)

    提出了一种模型无关的“排序先验对齐”框架，通过温度缩放的KL散度损失将来自领域专家、教师模型或大语言模型的外部排序先验统一蒸馏进神经或树类评分模型，以缓解标注数据稀缺的冷启动信贷评分难题。

    

    冷启动信贷评分——即在标注数据稀缺、特征薄弱或模型容量极小的情况下部署模型——是金融机器学习中反复出现的难题。当一个新贷款产品上线时，标注的违约数据稀缺、特征流水线不成熟，模型必须以最小容量部署以避免过拟合。标准防御手段都在同样有限的同一份数据上运行；我们真正需要的是一种扎根于领域知识的外部正则化来源。我们提出了排序先验对齐，这是一个模型无关的框架，通过温度缩放的KL散度损失，将外部排序先验（来自领域专家、教师模型或大语言模型）蒸馏到任何评分模型中。该框架通过一个统一公式将神经网络（MIL注意力）架构和树模型（XGBoost自定义目标函数）架构统一起来：L = L_task + gamma(t) * KL(P_agent || P_model)，其中 gamma(t) 遵循指数衰减调度。该方法不需要外部……（原文摘要在此处截断）

    arXiv:2610.11146v1 Announce Type: new  Abstract: Cold-start credit scoring -- deploying models with scarce labeled data, weak features, or minimal capacity -- is a recurring problem in financial machine learning. When a new lending product launches, labeled default data is scarce, feature pipelines are immature, and models must be deployed with minimal capacity to avoid overfitting. Standard defenses operate on the same limited data; what is needed is a source of external regularization grounded in domain knowledge.   We propose Ranking Prior Alignment, a model-agnostic framework that distills external ranking priors (from domain experts, teacher models, or LLMs) into any scoring model via a temperature-scaled KL divergence loss. The framework unifies neural (MIL attention) and tree-based (XGBoost custom objective) architectures through a single formulation: L = L_task + gamma(t) * KL(P_agent || P_model), where gamma(t) follows an exponential decay schedule. The method requires no exte
    
[^229]: 在一个循环深度安全，在另一个却危险：对齐循环语言模型中跨递归深度的安全性

    Safe at One Loop, Risky at Another: Aligning Safety Across Recurrent Depths in Looped Language Models

    [https://arxiv.org/abs/2610.10625](https://arxiv.org/abs/2610.10625)

    循环语言模型在不同递归深度上的安全性并不一致——更深的推理深度更容易遭受越狱攻击，且同一攻击可跨深度迁移，而SFT与偏好对齐均无法消除这种跨深度安全差距。

    

    循环语言模型通过在递归步骤中重复使用共享参数，提供了一种参数高效的方法来扩展模型能力。由于每个递归深度都可以被独立读取，单个LoopLM在不同的推理深度上暴露出更广泛的输出空间，由此引出一个重要问题：安全性是否在整个递归计算过程中得以保持。先前的评估表明，更深的递归可以提升模型对有害查询的安全性，但其在越狱攻击下的鲁棒性仍不清楚。因此，我们针对不同的递归深度，对LoopLM在越狱攻击下的安全性进行了全面评估。我们发现：攻击成功率在更深的推理深度上可能会增加；同一个查询在不同深度上可能引发不同的安全行为；并且针对某一深度构造的攻击可以迁移到其他深度。此外，监督微调（SFT）和偏好对齐并不能消除这些安全差距，这一发现激……（摘要在此处截断）

    arXiv:2610.10625v1 Announce Type: cross  Abstract: Looped Language Models (LoopLMs) provide a parameter efficient approach to scaling model capabilities through repeated use of shared parameters across recurrent steps. Since each recurrent depth can be read out independently, a single LoopLM exposes a broader output space across inference depths, raising an important question: whether safety is preserved throughout recurrent computation. Prior evaluations suggest that deeper recurrence can improve safety on harmful queries, but robustness under jailbreak attacks remains unclear. We therefore conduct a comprehensive safety evaluation of LoopLMs under jailbreak attacks targeting different recurrent depths. We find that attack success can increase at deeper inference depths, the same query can elicit different safety behaviors across depths, and attacks constructed against one depth can transfer to others. Moreover, SFT and preference alignment do not eliminate these safety gaps, motivati
    
[^230]: 循环式自我提升：面向循环语言模型的动态跨环在线策略蒸馏

    Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models

    [https://arxiv.org/abs/2610.10623](https://arxiv.org/abs/2610.10623)

    LoopOPD让循环语言模型利用自身更深层循环计算作为冻结“教师”，在学生自生成轨迹上进行在线策略蒸馏，无需外部教师或特权信息即可获得密集监督实现自我提升，D-LoopOPD进一步将该过程动态化。

    

    循环语言模型通过在循环计算步骤中复用共享参数，提供了一种参数高效地扩展推理能力的方法。尽管前景广阔，循环语言模型的有效后训练仍然充满挑战。现有方法要么提供稀疏的、或跨循环扩展代价高昂的基于奖励的监督，要么依赖外部教师模型或特权信息，导致教师模型可用性受限或师生上下文不匹配。为解决这些局限，我们提出LoopOPD，这是一个跨环在线策略蒸馏框架，它利用循环语言模型内部额外的循环计算作为其自身的监督来源。LoopOPD在学生自身生成的轨迹上，以冻结的终端循环策略作为具有计算特权的教师来指导中间循环的学生，从而在无需外部教师或特权信息的情况下提供密集监督。我们进一步提出动态LoopOPD（D-LoopOPD），它持续……（原文摘要在此处截断）

    arXiv:2610.10623v1 Announce Type: cross  Abstract: Looped Language Models (LoopLMs) offer a parameter efficient approach to scaling reasoning by reusing shared parameters across recurrent computation steps. Despite their promise, effective post-training of LoopLMs remains challenging. Existing approaches either provide reward based supervision that is sparse or costly to extend across loops, or rely on external teachers or privileged information, leading to limited teacher availability or teacher-student context mismatch. To address these limitations, we introduce LoopOPD, a cross-loop on-policy distillation framework that uses additional recurrent computation within a LoopLM as its own source of supervision. LoopOPD uses a frozen terminal loop policy as a compute privileged teacher for an intermediate loop student on student generated rollouts, providing dense supervision without an external teacher or privileged information. We further propose Dynamic LoopOPD (D-LoopOPD), which conti
    
[^231]: 宪法引导的水印技术

    Constitution-Guided Watermarking

    [https://arxiv.org/abs/2610.09552](https://arxiv.org/abs/2610.09552)

    本文提出“宪法引导水印”框架，通过将提供商需求表示为自然语言原则，使水印系统能够根据不同请求的需求灵活选择属性权衡，避免了传统方法对所有请求采用统一配置所导致的牺牲问题。

    

    水印技术使语言模型提供商能够识别由其模型生成的文本。然而，水印的理想属性之间可能存在冲突（即更强的水印信号可能会降低文本质量），而抵抗编辑的设计也可能便于伪造。提供商通过选择平衡竞争目标或优先考虑特定属性的配置来应对这些权衡。但这两种方法都会将一个统一的操作点强加于具有不同需求的请求上，可能在需要保留原始措辞时牺牲文本质量，或在需要可靠归因时牺牲鲁棒性。为了实现灵活且可适应的设计，我们提出了宪法引导水印，这是一个能够根据提供商需求（以自然语言原则的形式列出）来为每个请求选择合适权衡的框架。在离线阶段，一个预训练的推理代理会结合水印实现来审查宪法规则，并迭代地重新……

    arXiv:2610.09552v1 Announce Type: cross  Abstract: Watermarking enables language model providers to identify text generated by their models. However, its desired properties can conflict (\ie~stronger watermark signals can degrade text quality), while designs that resist editing may also facilitate forgery. Providers address these trade-offs by choosing configurations that balance competing objectives or prioritize particular properties. Either approach imposes a shared operating point on requests with different requirements, potentially sacrificing quality where wording preservation matters or robustness where reliable attribution is essential. To allow flexible and adaptable designs, we introduce \emph{Constitution-Guided Watermarking}, a framework that selects request-appropriate trade-offs from provider requirements, listed as natural-language principles. \emph{Offline}, a pretrained reasoning agent examines constitutional rules alongside watermark implementations and iteratively re
    
[^232]: VETTA：为多轮LLM智能体协调轮次级与令牌级信用分配

    VETTA: Coordinating Turn- and Token-Level Credit Assignment for Multi-Turn LLM Agents

    [https://arxiv.org/abs/2610.08402](https://arxiv.org/abs/2610.08402)

    VETTA通过共享轻量级评论家上的独立价值头联合学习轮次级与令牌级价值，在单次策略更新中协调两个层面的信用分配，从而解决多轮LLM智能体面临稀疏反馈时的信用分配难题。

    

    多轮LLM智能体通常在多次交互中接收到稀疏的任务反馈，同时逐个令牌地生成每个响应。这产生了两个相关的信用分配问题：哪些响应有助于实现最终结果，以及每个响应中哪些生成决策是关键的？现有方法通常只关注其中一个层面：轮次级方法评估完整的响应，但无法区分响应内部的决策；令牌级方法可以跨轮次传播反馈，但不会显式地为每个响应建模信用。这些互补的局限性促使我们在两个层面同时学习信用，并在单次策略更新中对其进行协调。我们提出了VETTA，这是一种信用分配方法，通过共享轻量级评论家上的独立价值头，联合学习轮次级和令牌级价值。VETTA沿两个时间序列计算优势函数，并将每个轮次优势与响应内中心化的令牌修正项相结合。

    arXiv:2610.08402v1 Announce Type: new  Abstract: Multi-turn LLM agents often receive sparse task feedback across several interactions, while generating each response token by token. This creates two related credit-assignment questions: which responses helped achieve the outcome, and which generation decisions mattered within each response? Existing methods typically focus on only one level: turn-level methods evaluate complete responses but do not distinguish the decisions within them; token-level methods can propagate feedback across turns but do not explicitly model credit for each response. These complementary limitations motivate learning credit at both levels and coordinating it in a single policy update. We introduce VETTA, a credit assignment method that jointly learns turn- and token-level values through separate heads on a shared lightweight critic. VETTA computes advantages along both temporal sequences and combines each turn advantage with a within-response-centered token re
    
[^233]: 重新审视深度强化学习中用于平滑控制的时间正则化

    Revisiting Temporal Regularization for Smooth Control in Deep Reinforcement Learning

    [https://arxiv.org/abs/2610.07910](https://arxiv.org/abs/2610.07910)

    本文证明了时间正则化能约束共享相同下一状态的邻近状态之间的动作差异，从而在观测噪声下提供空间平滑性，并据此提出了结合线性斜坡上升时间惩罚的CATS方法，在不损害任务性能的前提下实现机器人的平滑控制。

    

    深度强化学习策略可能产生非平滑的动作振荡，这阻碍了其在物理机器人上的部署。现有的基于架构和惩罚的方法通过直接降低对状态输入变化的敏感性来追求空间平滑性，但随着平滑程度的加强，其宽泛的约束可能会损害任务性能。时间正则化则约束沿观测转移的动作差异，但一直被认为无法提供观测噪声下所需的空间平滑性。我们通过证明时间惩罚能够约束共享相同下一状态的当前状态之间的期望动作差异，来重新审视这一假设，从而揭示出一种在经验上可延伸至空间平滑性的空间效应。基于这一发现，我们提出了仅使用时间平滑性的动作调节方法（CATS），该方法将时间惩罚与线性斜坡上升相结合。我们强调了时间正则化

    arXiv:2610.07910v1 Announce Type: new  Abstract: Deep Reinforcement Learning policies can produce nonsmooth action oscillations that hinder deployment on physical robots. Existing architectural and penalty-based approaches seek spatial smoothness by directly reducing sensitivity to changes in state inputs, but their broad constraints can degrade task performance as stronger smoothing is pursued. Temporal regularization instead constrains action differences along observed transitions, but has been considered unable to provide the spatial smoothness needed under observation noise. We revisit this assumption by proving that the temporal penalty bounds the expected action differences between current states sharing a next state, revealing a spatial effect that empirically extends to spatial smoothness. Building on this finding, we propose Conditioning for Action using only Temporal Smoothness (CATS), which combines a temporal penalty with linear ramp-up. We highlight temporal regularization
    
[^234]: SOL：利用双切片Wasserstein度量衡量文本分布之间的差距

    SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics

    [https://arxiv.org/abs/2610.06513](https://arxiv.org/abs/2610.06513)

    提出SOL——一种基于固定Transformer隐藏状态经验测度的双切片Wasserstein距离的文本分布距离度量，当Transformer为单射时可证明其为真正的度量，为非自回归语言模型的分布拟合评估提供了稳定的样本级评估方案。

    

    评估文本生成需要衡量生成分布与数据分布的匹配程度。对于自回归模型，这通过困惑度来实现；而扩散模型和基于流的语言模型只能提供似然界，其紧密程度在不同模型家族之间存在差异。基于样本的替代方法（如结合熵的生成式困惑度）则未考虑分布拟合情况。我们提出了SOL，一种文本分布之间的距离度量。每个序列由其在固定Transformer下隐藏状态的经验测度来表示，并通过双切片Wasserstein距离来比较这些测度的分布。我们证明，当Transformer是单射时，SOL是一个真正的度量。实验表明，SOL能够检测分布性失败、恢复预期的模型趋势，并提供稳定的基于样本的估计。我们提出SOL以填补当前非自回归模型评估协议中的空白。

    arXiv:2610.06513v2 Announce Type: replace  Abstract: Evaluating text generation requires measuring how well the generated distribution matches the data distribution. For autoregressive models, this is done by the perplexity. Diffusion and flow-based language models can only provide a likelihood bound, whose tightness differs between model families. Sample-based substitutes such as generative perplexity with entropy do not consider the distribution fit. We propose SOL,   a distance between text distributions. Each sequence is represented by the empirical measure of its hidden states under a fixed transformer and the distributions of these measures are compared by the double sliced Wasserstein distance. We prove that SOL is a metric if the transformer is injective. Experiments show that SOL detects distributional failures, recovers expected model trends, and provides stable sample-based estimates. We put forward SOL to fill the gap in the current evaluation protocol used for non auto-reg
    
[^235]: COCO问题的$\tilde{O}(\sqrt{T})$遗憾界与多对数级约束违反

    $\tilde{O}(\sqrt{T})$ Regret and Polylogarithmic Constraint Violation for COCO

    [https://arxiv.org/abs/2610.03983](https://arxiv.org/abs/2610.03983)

    本文针对对抗性约束在线凸优化（COCO）问题，提出了一种在线策略，在保持近最优 $O(\sqrt{T\log T})$ 遗憾的同时，将累计约束违反从 $\widetilde O(\sqrt{T})$ 显著降低到多对数级别的 $O(\log^2 T)$。

    

    我们研究了具有对抗性凸代价函数和约束函数的约束在线凸优化（$\mathsf{COCO}$）。在每一轮 $t\in[T]$ 中，学习者从 $d$ 维凸决策集 $X$ 中选择 $x_t$，随后自适应的对抗者揭示一个凸代价函数 $f_t$ 和一个约束函数 $g_t$。由此，学习者产生代价 $f_t(x_t)$ 和约束违反量 $\max\{0,g_t(x_t)\}$，并旨在同时最小化整个时间范围内的遗憾和累计约束违反（$\mathsf{CCV}$）。现有算法实现了 $O(\sqrt{T})$ 的遗憾和 $\widetilde O(\sqrt{T})$ 的 $\mathsf{CCV}$。我们证明了一种在线策略可以实现 $O(\sqrt{T\log T})$ 的遗憾和 $O(\log^2 T)$ 的 $\mathsf{CCV}$，在保持近最优遗憾的同时，将 $\mathsf{CCV}$ 从多项式级别降低到多对数级别。当所有代价函数均为 $\alpha$-exp-凹或 $\beta$-强凸时，该算法实现了 $O(\$（原文摘要在此处截断）

    arXiv:2610.03983v2 Announce Type: replace  Abstract: We study constrained online convex optimization with adversarial convex costs and constraints ($\mathsf{COCO}$). At each round $t\in[T]$, a learner selects $x_t$ from a $d$-dimensional convex decision set $X$, after which an adaptive adversary reveals a convex cost function $f_t$ and constraint function $g_t$. Consequently, the learner incurs a cost of $f_t(x_t)$ and a constraint violation of $\max\{0,g_t(x_t)\}$, and aims to simultaneously minimize regret and cumulative constraint violation ($\mathsf{CCV}$) over the entire horizon. Existing algorithms achieve $O(\sqrt{T})$ regret and $\widetilde O(\sqrt{T})$ $\mathsf{CCV}$. We show that an online policy can achieve $O(\sqrt{T\log T})$ regret and $O(\log^2 T)$ $\mathsf{CCV}$, reducing the $\mathsf{CCV}$ from polynomial to polylogarithmic while retaining near-optimal regret. When all cost functions are either $\alpha$-exp-concave or $\beta$-strongly convex, the algorithm achieves $O(\
    
[^236]: 基于连接组学中突触检测与校对任务的视觉语言模型基准测试

    Benchmarking Vision-Language Models on Synapse Detection and Proofreading in Connectomics

    [https://arxiv.org/abs/2609.36492](https://arxiv.org/abs/2609.36492)

    该研究首次系统性地基准测试了视觉语言模型在连接组学突触检测与校对任务上的表现，发现零样本性能接近随机，但通过少样本学习和LoRA微调，开源模型可达到专业模型水平，且在跨物种迁移中优于专业模型。

    

    我们对视觉语言模型（VLMs）在连接组学中检查电子显微镜图像时注释者所做决策的任务上进行了基准测试：突触检测（存在性与极性）和校对（分裂错误与合并错误）。对于突触检测，我们在零样本、四样本上下文学习和LoRA等设置下，针对专业模型，评估了19个开源模型和2个闭源模型，涵盖各种架构和规模，并使用公开资源自行构建了数据集。对于校对任务，我们在ConnectomeBench2数据集上评估了3个开源模型和2个闭源模型，并进行了从果蝇和小鼠到人类和斑马鱼的跨物种迁移。大多数模型在零样本情况下表现接近随机水平；少量示例主要对闭源模型和最大的开源模型有所帮助。在几千个标签上进行LoRA微调后，开源模型达到了与专业模型相当的水平。当在未见过的物种上评估时，经过适配的最优VLM在识别方面优于在同一数据上训练的专业模型

    arXiv:2609.36492v1 Announce Type: cross  Abstract: We benchmarked vision-language models (VLMs) on the decisions annotators take when inspecting electron microscopy images in connectomics: synapse detection (presence and polarity) and proofreading (split errors and merge errors). For synapse detection, we evaluated 19 open and 2 closed models across various architectures and sizes under zero-shot, four-shot in-context learning and LoRA settings, against specialist models, on datasets constructed by us using public resources. For proofreading, we evaluated 3 open and 2 closed models on the ConnectomeBench2 dataset, with cross-species transfer from fly and mouse to human and zebrafish. Most models were at chance zero-shot; a few examples helped mainly the closed and largest open ones. LoRA on a few thousand labels brought open models level with specialist models. When evaluated on unseen species, the best adapted VLMs outperformed specialist models trained on the same data in identifying
    
[^237]: Train4Merge：基于OPD模型合并中强化学习与监督微调教师的受控单教师研究

    Train4Merge: A Controlled Single-Teacher Study of RL vs. SFT Teachers for OPD-Based Model Merging

    [https://arxiv.org/abs/2609.32303](https://arxiv.org/abs/2609.32303)

    该研究通过受控单教师实验首次系统比较了强化学习（RL）与监督微调（SFT）两种训练算法在基于在线策略蒸馏的模型合并中的效果，发现在智能体、推理和感知三个领域中，RL训练的教师均能产生更强的学生模型，性能分别领先4.27、1.50和0.86个百分点。

    

    从共享检查点训练得到的领域专家可以通过在线策略蒸馏将他们的专业能力迁移到单个学生模型中。现有研究主要聚焦于改进这一合并过程，而用于训练专家的算法却缺乏系统性的比较。我们通过在智能体、推理和感知三个领域对监督微调（SFT）和强化学习（RL）进行受控单教师比较，研究哪种训练算法能产生更适合OPD的教师模型。教师和学生共享相同的Qwen3.5-9B初始化，且两种类型的教师在相近的任务性能水平上进行比较。我们的实验表明，在所有三个领域中，RL教师都能产生更强的学生模型，并能更高程度地恢复教师模型带来的性能增益。在各自的最佳检查点上，RL指导的学生模型分别以4.27、1.50和0.86个百分点的优势超越SFT指导的学生模型。

    arXiv:2609.32303v2 Announce Type: replace-cross  Abstract: Domain experts trained from a shared checkpoint can transfer their specialized capabilities to a single student through on-policy distillation (OPD). Existing research primarily focuses on improving this merging process, while the algorithms used to train the experts have received limited systematic comparison. We investigate which training algorithm produces teachers better suited to OPD through controlled single-teacher comparisons of supervised fine-tuning (SFT) and reinforcement learning (RL) across Agentic, Reasoning, and Perception. Teachers and students share the same Qwen3.5-9B initialization, and the two teacher types are compared at similar task performance. Our experiments show that RL teachers yield stronger students and higher recovery of teacher performance gains across all three domains. At their best checkpoints, RL-guided students outperform SFT-guided students by 4.27, 1.50, and 0.86 percentage points, respect
    
[^238]: 带探测的多臂老虎机：最优遗憾与赢家反馈的极限

    Bandits with Probing: Optimal Regret and the Limits of Winner Feedback

    [https://arxiv.org/abs/2609.15248](https://arxiv.org/abs/2609.15248)

    本文针对带探测的多臂老虎机问题首次确定了两个极小极大最优遗憾定律：在赢家反馈下，独立随机奖励情形的遗憾阶为 Φ_{n,k}(T)，而一般联合独立同分布奖励及固定序列情形的遗憾阶为 R_{n,k}(T)，并给出了匹配的任意时刻上界。

    

    学习者在每轮最多探测 n 个臂中的 k 个，接收它们在 [0,1] 区间内奖励的最大值，并与最优固定臂进行竞争。那么，探测的优势何时能为学习“买单”？我们确定了两个极小极大定律。在具有赢家反馈（最大奖励值及获胜臂标签）的独立随机奖励下，或者在给定块最大值之间单个带符号对比的任意固定序列下，极小极大遗憾的阶为 Φ_{n,k}(T)=min{(n-k)T/n, (n-k)/k}，其中 2≤k<n。在赢家反馈下，任意联合独立同分布奖励和固定序列的极小极大遗憾阶均为 R_{n,k}(T)=(n-k)/n · min{T, (n+T)/k, √(nT/k)}。这两个定律均包含普适常数和任意时刻有效上界。第一个定律将遗憾归结为纯粹的覆盖成本：同轮对比吸收了稳定性成本，而独立性允许精确重采样，其收益用于推动样本的进一步获取。第二个定律则额外引入了一个学习成本……

    arXiv:2609.15248v1 Announce Type: new  Abstract: A learner probes at most $k$ of $n$ arms each round, receives the maximum of their rewards in $[0,1]$, and competes with the best fixed arm. When does the probing advantage pay for learning? We determine two minimax laws. Under independent stochastic rewards with winner feedback (the maximum and a winning label), or on arbitrary fixed sequences given a single signed contrast between block maxima, the minimax regret has order $\Phi_{n,k}(T)=\min\{\frac{n-k}{n}T,\frac{n-k}{k}\}$, $2\le k<n$. Under winner feedback, both arbitrary joint i.i.d. rewards and fixed sequences have minimax regret of order $R_{n,k}(T)=\frac{n-k}{n}\min\{T,\frac{n+T}{k},\sqrt{\frac{nT}{k}}\}$. Both laws have universal constants and anytime upper bounds. The first reduces regret to a pure coverage cost: same-round contrasts absorb the stability cost, and independence permits exact resampling whose gains fund sample advancement. The second adds a learning cost that be
    
[^239]: 用于大语言模型行为控制的无训练任务向量

    Training-Free Task Vectors for LLM Behavioral Control

    [https://arxiv.org/abs/2609.09054](https://arxiv.org/abs/2609.09054)

    提出了一种无需微调的TFTVs方法，仅利用前向传播统计量将激活引导向量映射为秩一权重空间编辑，从而以低成本实现对大语言模型行为的放大、抑制与组合控制。

    

    任务向量通过识别权重空间中具有语义意义的方向，实现训练后的模型编辑，通常计算为微调模型与其预训练初始化模型之间的差异。然而，这种对微调的依赖使得发现此类方向的成本高昂，限制了训练后模型编辑的实用性。为了解决这一局限，我们提出了无训练任务向量（TFTVs），这是一种无需微调即可计算类任务向量方向的新方法。我们的方法仅利用前向传播统计量，将激活引导向量映射为秩一权重空间编辑，同时满足直接支持通过加法实现学习、通过减法实现遗忘以及多个编辑进行组合的算术性质。在实验中，我们在大语言模型行为控制任务上对TFTVs进行了评估，结果表明它们能够在保持模型原有能力的同时，持续地放大、抑制和组合目标行为。

    arXiv:2609.09054v1 Announce Type: cross  Abstract: Task vectors enable post-training model editing by identifying semantically meaningful directions in weight space, typically computed as the difference between a fine-tuned model and its pretrained initialization. However, this reliance on fine-tuning makes discovering such directions costly and limits the practicality of post-training model editing. To address this limitation, we introduce Training-Free Task Vectors (TFTVs), a novel method to compute task-vector-like directions without requiring fine-tuning. Our method maps activation steering vectors to rank-one weight-space edits using only forward-pass statistics, while satisfying arithmetic properties that directly support learning via addition, forgetting via subtraction, and the composition of multiple edits. Empirically, we evaluate TFTVs on large language model behavioral control tasks and show that they consistently amplify, suppress, and compose target behaviors while preser
    
[^240]: Chimaera：一种用于跨任务和跨数据集图学习的图专家混合架构

    Chimaera: A Mixture-of-Graph-Experts Architecture for Cross-Task and Cross-Dataset Graph Learning

    [https://arxiv.org/abs/2609.08709](https://arxiv.org/abs/2609.08709)

    Chimaera 提出了一种将专家混合机制与图基础模型相结合的新架构，通过整合多种 GFM 架构和大语言模型生成的嵌入，实现了跨任务和跨数据集的图学习能力。

    

    为图设计基础模型具有挑战性，原因在于图结构的不规则性以及嵌入的尺寸和特征各不相同。Chimaera 将专家混合机制与图基础模型（GFM）相结合，整合了不同的 GFM 架构，例如图提示和线性 GNN 模型。该方法利用大语言模型生成嵌入，专家可以按照不同的策略进行训练和组合，包括不同的 GFM、嵌入等。此外，Chimaera 扩展了现有的线性 GNN，使其除了支持节点级任务外，还支持链接级和图级任务。研究使用六个基准文本属性图数据集，在节点、链接和图分类任务上进行了同任务和跨任务的实证分析。实验证明了 Chimaera 的有效性及其跨任务和跨数据集的迁移能力。进一步的见解还包括需要同时使用大型和小型语言模型。

    arXiv:2609.08709v1 Announce Type: new  Abstract: Designing foundation models for graphs is challenging due to the irregular structure of graphs and the different sizes and characteristics of embeddings. Chimaera integrates mixture-of-experts with graph foundation models (GFM). It integrates different GFM architectures, such as graph prompts and linear GNN models. Large language models are used to generate embeddings, and experts can be trained and combined following different strategies, GFMs, embeddings, etc. Furthermore, Chimaera extends existing linear GNNs to support link-level and graph-level tasks in addition to node-level tasks. Empirical analyses are performed on same-task and cross-task experiments with node, link, and graph classification tasks using six benchmark text-attributed graph datasets. The experiments demonstrate the effectiveness of Chimaera and its capabilities for transfer across tasks and datasets. Further insights include the need to use both large and small la
    
[^241]: 大批量训练中的动量：Polyak 动量扩大临界批大小，Nesterov 动量提升数据效率

    Momentum in large-batch training: Polyak enlarges the critical batch size, Nesterov improves data efficiency

    [https://arxiv.org/abs/2609.02728](https://arxiv.org/abs/2609.02728)

    该论文在幂律核回归框架下证明，在单遍大批量训练中 Polyak 动量可将临界学习率随批大小线性放大（从而扩大临界批大小约 1/(1-ρ) 倍），而 Nesterov 动量的临界学习率以 $B^\beta$（β>1）的更快速度增长，从而显著提升数据效率，并给出了刻画完整风险动力学的标度律与三区制批大小相图。

    

    我们在单遍（one-pass）训练机制下研究动量何时以及如何改善大批量训练，并以幂律核回归作为一个易于解析的设定。我们首先通过临界学习率刻画风险的稳定性，临界学习率定义为保证训练稳定的最大学习率，并得到 $\eta_{\mathrm{SGD}}^{\mathrm{crit}}\eqsim 1$、$\eta_{\mathrm{Polyak}}^{\mathrm{crit}}\eqsim \min\{1,B(1-\rho)\}$ 以及 $\eta_{\mathrm{Nesterov}}^{\mathrm{crit}}\eqsim \min\{1,B^\beta(1-\rho)\}$，其中 $B$ 为批大小，$\rho$ 为动量因子，$\beta>1$ 为容量指数。在该允许区域内，我们推导出完整风险动力学的标度律，刻画了训练从早期瞬态阶段、经幂律衰减、直至噪声底限的完整演化过程。随后，在固定数据预算下，我们在允许的学习率与动量因子范围内最小化最后一步风险，得到了一个包含三个区制的批大小相图，揭示了（原文摘要在此处截断）

    arXiv:2609.02728v1 Announce Type: cross  Abstract: We study when and how momentum improves large-batch training in the one-pass regime, using power-law kernel regression as a tractable setting. We first characterize risk stability through the critical learning rate, defined as the largest learning rate for stable training, and obtain $\eta_{\mathrm{SGD}}^{\mathrm{crit}}\eqsim 1$, $\eta_{\mathrm{Polyak}}^{\mathrm{crit}}\eqsim \min\{1,B(1-\rho)\}$, and $\eta_{\mathrm{Nesterov}}^{\mathrm{crit}}\eqsim \min\{1,B^\beta(1-\rho)\}$, where $B$ is the batch size, $\rho$ is the momentum factor, and $\beta>1$ is the capacity exponent. Within this admissible region, we derive scaling laws for the full risk dynamics, capturing the progression from an early transient, through power-law decay, to a noise floor. We then minimize the final-step risk over the admissible learning rates and momentum factors under a fixed data budget, yielding a three-regime batch-size phase diagram that reveals how the rol
    
[^242]: 参考集重采样下OOD分数的判定不稳定性

    Verdict Instability of OOD Scores under Reference Resampling

    [https://arxiv.org/abs/2609.00691](https://arxiv.org/abs/2609.00691)

    本文提出“判定不稳定性”这一新概念，通过重采样参考集并用其闭式解（无需拟合参数）来度量OOD检测分数对参考集选择的敏感程度，并揭示远分布外查询的分数恰好落在最可复现的判定上。

    

    事后（post-hoc）分布外（OOD）检测器是在有限的参考集上拟合的，因此它们产生的每一个分数都只是一个估计值。如果我们选择了另一个参考集，某些判定结果就会发生改变。我们通过重采样参考集并记录分数的自助法（bootstrap）标准差来度量这种变动，并将其称为“判定不稳定性”。该量具有无需拟合任何参数的闭式解。一个判定的不稳定性等于所分配类别沿查询方向的类内离散度除以该类别参考样本数的平方根。正是这一样本数将判定不稳定性与分数分布的几何结构区分开来，且只有在类别不平衡的情况下才能将其识别出来。不稳定性随局部离散度的增大而增长。远分布外（Far-OOD）查询位于各向异性嵌入的低方差方向上，因此在我们测试的所有基于距离的分数中，最高分数值都被分配给了那些最可复现的判定。只有es…（摘要不完整）

    arXiv:2609.00691v1 Announce Type: new  Abstract: Post-hoc out-of-distribution detectors are fitted on a finite reference set, so every score they produce is an estimate. If we had chosen a different set, some verdicts would have moved. We measure that movement by resampling the reference set and recording the bootstrap standard deviation of the score, which we call verdict instability. It admits a closed form with no fitted parameters. The instability of a verdict is the within-class dispersion of the assigned class along the query's direction, divided by the square root of that class's reference count. That count is what separates verdict instability from the geometry of the score distribution, and it is identifiable only under class imbalance. Instability grows with the local dispersion. Far-OOD queries lie along the low-variance directions of an anisotropic embedding, so every distance-based score we test assigns its highest values to the verdicts that are most reproducible. Only es
    
[^243]: SAUSS：基于无偏模拟分数的随机逼近方法用于受限因变量模型

    SAUSS: Stochastic Approximation with Unbiased Simulated Scores for Limited Dependent Variable Models

    [https://arxiv.org/abs/2608.25304](https://arxiv.org/abs/2608.25304)

    提出了一种基于无偏模拟分数的随机逼近方法（SAUSS），通过固定小批量和无偏分数估计，显著降低多项选择模型的计算成本并消除模拟偏差。

    

    多类别选择模型允许灵活替代模式，但在处理大量备选项或观测数据时计算成本高昂。在固定的每观测模拟预算下，模拟最大似然法会引入模拟偏差，而每次优化步骤都需要全样本似然评估。我们提出了基于无偏模拟分数的随机逼近方法（SAUSS），这是一种基于条件无偏小批量分数估计的均值随机逼近方法。每次迭代使用固定的小批量，与样本大小无关。对于多项概率单位模型，接受-拒绝抽样提供精确的条件抽取，并对任意固定数量的接受抽取给出无偏分数估计。在局部条件下，均值估计量的渐近理论和SAUSS迭代的部分和过程纳入了小批量与模拟变异性，并支持随机缩放和插入式推断。在模拟实验和一项...

    arXiv:2608.25304v1 Announce Type: cross  Abstract: Multinomial choice models allow flexible substitution patterns but become computationally demanding with many alternatives or observations. With a fixed per-observation simulation budget, simulated maximum likelihood introduces simulation bias, while each optimization step requires a full-sample likelihood evaluation. We propose Stochastic Approximation with Unbiased Simulated Scores (SAUSS), an averaged stochastic approximation based on conditionally unbiased mini-batch score estimates. Each iteration uses a fixed mini-batch regardless of sample size. For multinomial probit, accept-reject sampling provides exact conditional draws and unbiased score estimates for any fixed number of accepted draws. Under local conditions, asymptotic theory for the averaged estimator and the partial-sum process of the SAUSS iterates incorporates mini-batch and simulation variability and supports random-scaling and plug-in inference. In simulations and a
    
[^244]: 自监督预训练何时对表格模型有帮助？标签稀缺与缺失数据研究

    When Does Self-Supervised Pretraining Help Tabular Models? A Study of Label Scarcity and Missing Data

    [https://arxiv.org/abs/2608.24381](https://arxiv.org/abs/2608.24381)

    本文通过14个任务实验发现，自监督预训练在表格数据上平均优于从头训练，但增益不稳定且不显著，且其对干净数据集更有效，而对高缺失度数据可能有害。

    

    arXiv:2608.24381v1 公告类型：新公告 摘要：自监督学习（SSL）已成为表格数据的一种有前景的方法，但其在极端标签稀缺和测试时缺失情况下的有效性仍未充分探索。在本文中，我们评估了一种掩码-恢复自监督预训练目标，与从头训练及经典基线在14个多样化的分类任务中进行对比。首先，尽管SSL在平均表现上优于从头训练，并与最先进的树集成方法保持竞争力（在10%标签下，达到约0.8954 AUC，而随机森林为0.9015），但SSL与从头训练之间的增益存在较高的任务间方差，且缺乏显著性（在5%和10%标签下p值均为0.626）。其次，与缺失值插补目标普遍有利于具有固有缺失数据的数据集这一假设相反，SSL在干净数据集上提供了最可靠的改进，而在高固有缺失度的数据集上常常导致性能下降。第三，尽管存在这种……

    arXiv:2608.24381v1 Announce Type: new  Abstract: Self-supervised learning (SSL) has emerged as a promising approach for tabular data, yet its efficacy under extreme label scarcity and test-time missingness remains under-explored. In this paper, we evaluate a mask-and-recover SSL pretraining objective against training from scratch and classical baselines across 14 diverse classification tasks. First, while SSL outperforms training from scratch on average and remains competitive with state-of-the-art tree ensembles (achieving ~0.8954 AUC vs. Random Forest's 0.9015 at 10% labels), the SSL-vs-scratch gains exhibit high inter-task variance and lack significance (p = 0.626 at both 5% and 10% labels). Second, contrary to the hypothesis that missing-value imputation objectives universally benefit datasets with native missingness, SSL yields the most reliable improvements on clean datasets, while frequently degrading performance on datasets with high inherent missingness. Third, despite this tr
    
[^245]: 基于深度学习框架在GOTHIC巡天中解耦候选双活动星系核与偶然叠加

    Decoupling candidate dual AGN from chance superpositions in the GOTHIC survey via a deep-learning framework

    [https://arxiv.org/abs/2608.24164](https://arxiv.org/abs/2608.24164)

    本文通过YOLOv11定向边界框深度学习框架，有效从GOTHIC巡天的偶然叠加中区分出真实双活动星系核，实现了高精度检测。

    

    双活动星系核（DAGN）标志着合并星系和超大质量黑洞配对演化中的关键阶段，然而由于投影效应和有限的空间分辨率，它们在大规模成像巡天中仍难以识别。致密前景恒星和未解析的子结构可能通过偶然叠加模拟双核，从而干扰自动检测。我们重新审视了GOTHIC流程中标记但被拒绝的46,061个DAGN候选星系，主要原因在于两个核落在SDSS光纤孔径内或超过其分离阈值。我们基于YOLOv11定向边界框架构，在带注释的SDSS成像上训练了一个监督深度学习框架，以区分真正的双核与前景恒星污染物及其他虚假对齐。最终模型在双核类别上实现了验证精度0.919、召回率0.905和F1分数0.912。

    arXiv:2608.24164v1 Announce Type: cross  Abstract: Dual active galactic nuclei (DAGN) mark a critical phase in the evolution of merging galaxies and the pairing of supermassive black holes, yet they remain difficult to identify in large imaging surveys because of projection effects and limited spatial resolution. Compact foreground stars and unresolved substructure can mimic dual nuclei through chance superposition, complicating automated detection. We revisit the 46,061 galaxies flagged but rejected as DAGN candidates by the GOTHIC pipeline, primarily because the two nuclei fell within the SDSS fibre aperture or exceeded its separation threshold. We train a supervised deep-learning framework based on the YOLOv11 oriented-bounding-box architecture on annotated SDSS imaging to separate genuine dual nuclei from foreground stellar contaminants and other spurious alignments. The final model attains a validation precision of 0.919, recall of 0.905, and $F_1$ of 0.912 for the dual-nuclei cla
    
[^246]: TreeWY：面向门控DeltaNet混合模型的自适应验证方法

    TreeWY: Speculative Verification for Gated DeltaNet Hybrids

    [https://arxiv.org/abs/2608.20961](https://arxiv.org/abs/2608.20961)

    本文提出TreeWY方法，通过树状WY变换消除推测解码中的状态快照，显著降低内存开销，从而支持高接受率的宽草稿树。

    

    arXiv:2608.20961v1 公告类型：新论文  摘要：现代开源模型多为混合架构：大部分层采用线性注意力（门控DeltaNet，GDN）层，这些层携带一个小的固定大小的循环状态，而非不断增长的键值（KV）缓存。这使得普通解码在内存上高效，但不利于推测解码。为了验证一批草稿标记并回滚被拒绝的标记，当前系统在GDN层的每个草稿位置都会对完整循环状态进行快照，而这些快照无法在草稿树的分支间共享，因此宽且高接受率的树在内存上变得不可行。我们移除了快照。利用门控Delta规则的一种树状WY变换，我们通过一次三角求解计算每个草稿节点的输出，并在提交时仅重建一个被接受的状态，存储一个小型的伪值矩阵而非逐节点状态；该推导仅依赖于门控Delta规则，而不涉及任何其他架构细节。在两个规模的服务器基准测试中...

    arXiv:2608.20961v1 Announce Type: new  Abstract: Modern open models are hybrids: most layers are linear-attention (Gated DeltaNet, GDN) layers carrying a small fixed-size recurrent state instead of a growing key-value (KV) cache. This makes ordinary decoding memory-efficient, but hurts speculative decoding. To verify a batch of draft tokens and then roll back the rejected ones, today's systems snapshot the full recurrent state at every draft position for GDN layers, and those snapshots cannot be shared across branches of a draft tree, so a wide, high-acceptance tree becomes memory-infeasible. We remove the snapshots. Using a tree-structured WY transform of the gated delta rule, we compute every draft node's output with a single triangular solve and reconstruct only the one accepted state on commit, storing a small pseudo-value matrix instead of per-node states; the derivation depends only on the gated delta rule, not on any other architectural detail. In serving benchmarks on two scale
    
[^247]: 通过对偶价格进行微分：容量约束下的端到端策略学习

    Differentiating Through Dual Prices: End-to-End Policy Learning Under Capacity Constraints

    [https://arxiv.org/abs/2608.04669](https://arxiv.org/abs/2608.04669)

    该论文提出通过对偶价格进行端到端微分来联合训练结果模型与分配策略，实现了容量约束下从观测数据中学习资源分配策略，并给出了具有可证明容量可行性和次优性界的凸松弛形式。

    

    许多社会服务将稀缺资源（如住房援助或医疗干预）分配给逐一到达的求助者。每位到达者必须立即获得分配决策，且每项资源的长期使用量必须保持在其容量范围内。我们研究如何从记录的观测数据中学习这样的策略。最近的研究（Tang等人，2024）提出了一种决策无关的流程：通过回归为每种处理方式拟合一个结果模型，基于这些模型为每个有容量约束的资源定价，并为每位到达者分配预测结果减去价格后收益最大的处理方式。我们则转而端到端地训练结果模型，通过对偶价格本身对已部署策略价值的离策略估计进行反向传播求导。我们研究了两种形式化方法：一种是精确的非凸形式，另一种是凸松弛形式，后者的最优解在期望意义上始终满足容量约束，且其次优性可以被显式地界定。

    arXiv:2608.04669v2 Announce Type: replace  Abstract: Many social services assign scarce resources, such as housing assistance or hospital interventions, to people who arrive one at a time. Each arrival must receive a decision immediately, and the long-run usage of every resource must stay within its capacity. We study how to learn such a policy from logged observational data. Recent work (Tang et al., 2024) proposes a decision-blind pipeline that fits one outcome model per treatment by regression, prices each capacitated resource from those models, and assigns each arrival the treatment with the largest predicted outcome minus price. We instead train the outcome models end-to-end, differentiating an off-policy estimate of the deployed policy's value through the dual prices themselves. We study two formulations: an exact nonconvex one, and a convex relaxation whose optimum always satisfies the capacity constraints in expectation and whose suboptimality can be explicitly bounded. Every m
    
[^248]: 泛函流匹配的离散化与统计一致性

    Discretization and Statistical Consistency of Functional Flow Matching

    [https://arxiv.org/abs/2608.04531](https://arxiv.org/abs/2608.04531)

    本文证明了在有限观测细化（允许非嵌套）且一致有界的条件下，泛函流匹配的速度目标依均方收敛，并量化了观测、学习与数值误差对生成分布的影响，为其函数空间模型的统计一致性提供了理论保证。

    

    泛函流匹配通过将一个简单的概率分布连续变换为目标分布，学习生成随机函数，例如速度场或偏微分方程的解。然而在实践中，训练和采样仅使用有限个网格值或系数。那么，细化这些观测能否恢复预期的函数空间模型？观测误差、学习误差和数值误差又如何影响生成的分布？我们通过以下方式回答这些问题：首先证明，当有限重构收敛于底层函数并保持一致有界时，用于训练的速度目标依均方收敛。这些观测不必是嵌套的：每次细化都可以替换先前的传感器位置。我们还给出了正交投影的定量界，并在适当的正则性与重构条件下将结果推广到点观测情形。

    arXiv:2608.04531v2 Announce Type: replace-cross  Abstract: Functional flow matching learns to generate random functions, such as velocity fields or solutions of partial differential equations, by continuously transforming a simple probability distribution into a target distribution. In practice, however, training and sampling use only finitely many grid values or coefficients. Does refining these observations recover the intended function-space model, and how do observation, learning, and numerical errors affect the generated distribution? We address these questions by first proving that the velocity targets used for training converge in mean square when the finite reconstructions converge to the underlying functions and remain uniformly bounded. The observations need not be nested: each refinement may replace earlier sensor locations. We also give quantitative bounds for orthogonal projections and extend the result to point observations under suitable regularity and reconstruction ass
    
[^249]: FinVerse：金融时间序列基准

    FinVerse: Financial Time-Series Benchmark

    [https://arxiv.org/abs/2608.03259](https://arxiv.org/abs/2608.03259)

    FinVerse是一个金融领域的时间序列预测基准，通过更现实的评估方式（如方向预测）来弥补传统误差指标在真实决策中的不足，包含大量金融时间序列数据。

    

    摘要：随着时间序列基础模型的出现，以有意义的方式评估其预测能力的基准变得越来越重要。现有的时间序列预测基准提供了有用的标准化比较，但它们通常用统一的基于误差的指标来评估异构序列。在此类指标下表现强劲并不一定意味着模型的预测能支持跨领域的最佳现实决策。例如，在股票预测中，正确预测价格是上涨还是下跌可能比仅最小化逐点预测误差更直接地与实现收益相关。为此，我们引入了FinVerse，一个金融领域的时间序列预测基准，朝着更现实的评估迈出了第一步。发布的FinVerse数据工件包含116,897个金融时间序列，共171.1M个观测值，其中60,232个序列被用于基准测试。

    arXiv:2608.03259v2 Announce Type: replace-cross  Abstract: As time-series foundation models have emerged, the need for benchmarks that can evaluate their forecasting ability in meaningful ways has become increasingly important. Existing time-series forecasting benchmarks provide useful standardized comparisons, but they often evaluate heterogeneous series with uniform error-based metrics. Strong performance under such metrics does not necessarily imply that a model's forecasts will support the best real-world decisions across domains. For example, in stock forecasting, correctly predicting whether a price will rise or fall can be more directly relevant to realized returns than minimizing point-wise forecast error alone. To this end, we introduce FinVerse, a finance-domain time-series forecasting benchmark that takes a first step toward more realistic evaluation. The released FinVerse data artifact contains 116,897 financial time series with 171.1M observations, of which 60,232 series w
    
[^250]: 基于异方差神经网络学习双向因果交互

    Learning Bidirectional Causal Interactions with Heteroscedastic Neural Networks

    [https://arxiv.org/abs/2607.22313](https://arxiv.org/abs/2607.22313)

    提出无需外部工具变量的异方差神经联立方程估计器 SEM-DNN，通过条件协方差对角化实现观测数据中双向同期因果交互的唯一识别与非线性联合估计。

    

    从观测数据中估计同期双向交互作用是困难的，因为每个结果变量相对于另一方都是内生的，而灵活的回归可能只能捕捉到简化形式的依赖关系。本文提出了 SEM-DNN，一种异方差神经联立方程估计器，能够在没有外部工具变量的情况下学习互惠的结构性交互作用。其识别策略利用了条件协方差对角化：当结构冲击具有零条件均值、在给定前定协变量的条件下互不相关、且表现出非成比例的条件方差时，只有真实的交互系数才能在整个特征空间上使条件残差协方差对角化。该方法使用对角高斯拟似然来联合逼近非线性结构均值函数和依赖于特征的方差，该拟似然中融入了联立方程系统的雅可比矩阵。我们建立了唯一识别性……

    arXiv:2607.22313v2 Announce Type: replace  Abstract: Estimating contemporaneous bidirectional interactions from observational data is difficult because each outcome is endogenous to the other, while flexible regressions may capture only reduced-form dependence. This paper proposes SEM-DNN, a heteroscedastic neural simultaneous-equation estimator that learns reciprocal structural interactions without external instruments. Identification exploits conditional covariance diagonalization: when structural shocks have zero conditional means, are conditionally uncorrelated given predetermined covariates, and exhibit nonproportional conditional variances, only the true interaction coefficients diagonalize the conditional residual covariance across the feature space. The method jointly approximates nonlinear structural mean functions and feature-dependent variances using a diagonal Gaussian quasi-likelihood that incorporates the simultaneous-system Jacobian. We establish unique identification an
    
[^251]: GeneSpeak-FP：从观测到的细胞水平扰动特征中检索靶点与化合物

    GeneSpeak-FP: Target and Compound Retrieval from Observed Cell-Level Perturbation Signatures

    [https://arxiv.org/abs/2607.17671](https://arxiv.org/abs/2607.17671)

    提出GeneSpeak-FP，一种基因标记Transformer模型，可从单细胞扰动特征中联合检索化合物及其靶点，其top-10化合物检索和靶点召回率分别达到随机排序期望值的约13倍和11倍。

    

    单细胞扰动谱可能有助于在不同细胞背景下识别已知化合物。我们提出了GeneSpeak-FP，这是一个基因标记Transformer模型，它基于处理后的细胞相对于细胞系特异性DMSO参考的表达变化，并结合一个粗粒度的器官标记，联合对记录的化合物及其靶点注释进行排序。在Tahoe-100M数据集上，我们在每种化合物内部将11,673个药物-细胞系组按90/10比例划分，尽管化合物和细胞系的身份可能重复出现，但没有任何完整的组在划分之间共享。在38,400次查询的蒙特卡洛评估中，在379个固定候选化合物中，正确化合物在12.9%的细胞中排名第一，在34.3%的细胞中位列前10。该模型从278个基因的词表中，在top 10内检索到40.8%的注释靶点，在top 20内检索到54.4%。Top-10化合物检索和靶点召回率分别约为均匀随机排序下期望值的13倍和11倍；top-1化合物检索接近49倍。

    arXiv:2607.17671v2 Announce Type: replace  Abstract: Single-cell perturbation profiles may help identify known compounds across cellular contexts. We present GeneSpeak-FP, a gene-token Transformer that jointly ranks recorded compounds and their target annotations from a treated cell relative to a cell-line-specific DMSO reference, together with a coarse organ token. On Tahoe-100M, we split 11,673 drug--cell-line groups 90/10 within each compound. No complete group is shared across partitions, although compound and cell-line identities may recur. In a 38,400-query Monte Carlo pass, the correct compound appears first for 12.9% of cells and within the top 10 for 34.3%, among 379 fixed candidates. The model retrieves 40.8% of annotated targets within the top 10 and 54.4% within the top 20, from a vocabulary of 278 genes. Top-10 compound retrieval and target recall are approximately 13 and 11 times their expected values under uniformly random rankings; top-1 compound retrieval is nearly 49 
    
[^252]: 语言模型中的分布式稀疏干预

    Distributed Sparse Interventions in Language Models

    [https://arxiv.org/abs/2607.07128](https://arxiv.org/abs/2607.07128)

    本文提出了分布式稀疏干预（DSI）方法，通过操纵分布于各层和注意力头中的稀疏个体神经元集合来引导语言模型行为，克服了传统密集线性干预同时改变数千个神经元的弊端，更精准地将模型表示与行为联系起来。

    

    语言模型可以在不同的抽象层次上执行各种任务。它们能够灵活地从上下文中推断任务、同时执行多个任务，并在相互冲突的目标之间做出选择。理解和控制模型如何表示这些任务，对于使模型与人类目标保持一致至关重要。研究人员可以通过多种方式改变模型行为，但模型引导——即操纵中间任务表示——是将表示与行为联系起来的最直接方式。当前的引导方法通常采用密集干预并假设线性效应，例如在残差流中沿全局方向移动所有激活值。由于这类干预会同时改变数千个神经元，研究人员不得不自行区分相关神经元与无关神经元。我们提出了分布式稀疏干预（DSI），通过分布在各层和注意力头中的一组稀疏的个体神经元来引导模型。

    arXiv:2607.07128v2 Announce Type: replace  Abstract: Language models can perform a wide range of tasks at various levels of abstraction. They flexibly infer tasks from context, execute multiple tasks simultaneously, and select among competing goals. Understanding and controlling how models represent these tasks is central to aligning them with human objectives. Researchers can change model behaviour in many ways, but model steering, which manipulates intermediate task representations, links representation to behaviour most directly. Current steering methods typically intervene densely and assume linear effects, e.g., shifting every activation along a global direction in the residual stream. Because such interventions change thousands of neurons at once, they leave researchers to separate relevant from irrelevant neurons. We introduce Distributed Sparse Interventions (DSI), which steers models through a sparse set of individual neurons distributed across layers and attention heads. Find
    
[^253]: Flow-ERD：面向多样化交通仿真的智能体类型感知流匹配与熵正则化蒸馏

    Flow-ERD: Agent-type Aware Flow Matching With Entropy-Regularized Distillation for Diverse Traffic Simulation

    [https://arxiv.org/abs/2607.06957](https://arxiv.org/abs/2607.06957)

    Flow-ERD通过智能体类型感知流匹配与熵正则化蒸馏两阶段框架，在多智能体交通仿真中联合优化真实性与多样性，并有效防止闭环推演中的模式坍缩问题。

    

    真实且多样化的交通仿真对自动驾驶的开发至关重要。然而，现有主流基准主要奖励真实性，近期方法也据此进行优化，导致多样性方面未被充分探索。我们提出Flow-ERD，一个同时追求真实性与多样性的多智能体仿真器。其核心骨干是智能体类型感知流匹配（AFM），它将流匹配的多模态表达能力与特定类型的运动学执行相结合，在保留细粒度多样性的同时，使运动与各智能体类型保持一致。第二阶段为熵正则化蒸馏（ERD），采用熵正则化的反向KL目标对闭环推演分布进行微调，在缓解闭环协变量偏移的同时，显式防止模型坍缩到高密度模态上。我们使用交叉对多样性（Cross-Pair Diversity）及其有效性门控变体以及标准的真实性评分来评估Flow-ERD。验证实验……

    arXiv:2607.06957v2 Announce Type: replace-cross  Abstract: Realistic and diverse traffic simulation is essential to autonomous driving development. Yet prevailing benchmarks predominantly reward realism, and recent methods have optimized accordingly, leaving diversity underexplored. We introduce Flow-ERD, a multi-agent simulator that pursues realism and diversity jointly. Its backbone, Agent-Type Aware Flow Matching (AFM), couples flow matching's multi-modal expressiveness with type-specific kinematic execution. It preserves fine-grained diversity while keeping motions consistent with each agent type. A second stage, Entropy-Regularized Distillation (ERD), fine-tunes the closed-loop rollout distribution with an entropy-regularized reverse-KL objective. This mitigates closed-loop covariate shift while explicitly preventing collapse onto high-density modes. We evaluate Flow-ERD with Cross-Pair Diversity and its validity-gated variant alongside standard realism scores. Validation experime
    
[^254]: GaP：一种用于变分自动化任务的“图即策略”多智能体自学习框架

    GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks

    [https://arxiv.org/abs/2607.05369](https://arxiv.org/abs/2607.05369)

    本文提出了Graph-as-Policy（GaP），一个多智能体自学习编码框架，它从模块化开源机器人技能库生成包含感知、规划和控制节点的有向计算图，并利用内部仿真环境进行排练，从而提升物体几何与姿态变化较大的变分自动化任务的可靠性。

    

    要让机器人在商业和工业应用中可靠地工作，智能体编码系统的最新进展有潜力将可解释的机器人编程与无模型策略的开放世界适应能力相结合。我们专注于“变分自动化”（VA），这是一类在物体几何形状和姿态上的变化比固定自动化更大的任务。无模型策略往往难以缩小变分自动化任务的可靠性差距，而这些任务在商业和工业应用中必须持久且可靠地执行。受先前任务与运动规划（TAMP）和机器人操作系统（ROS）方面工作的启发，我们提出了“图即策略”，这是一个多智能体编码框架，能够从模块化开源机器人技能库（MORSL）中生成包含感知、规划和控制节点的有向计算图。GaP生成一个内部仿真环境，以便用不同的图并行地排练任务实例。

    arXiv:2607.05369v2 Announce Type: replace-cross  Abstract: For robots to work reliably in commercial and industrial applications, recent advances in agentic coding systems have potential to combine interpretable robot programming with the open-world adaptability of model-free policies. We focus on "Variational Automation" (VA), a class of tasks that have larger variations in object geometry and pose than fixed automation. Model-free policies often struggle to close the reliability gap for VA tasks, which must be executed persistently and reliably in commercial and industrial applications. Motivated by prior work on Task and Motion Planning (TAMP) and the Robot Operating System (ROS), we introduce Graph-as-Policy (GaP), a multi-agent coding harness that generates directed computation graphs with perception, planning, and control nodes from a Modular Open Robot Skill Library (MORSL). GaP generates an internal simulation environment to rehearse task instances with different graphs in para
    
[^255]: CodeJeNN：一个面向物理应用的简单C++神经网络生成器

    CodeJeNN: A simple C++ neural network generator for physics applications

    [https://arxiv.org/abs/2607.02746](https://arxiv.org/abs/2607.02746)

    CodeJeNN能将训练好的Keras模型自动生成为无外部依赖的自包含C++推理代码，使神经网络可高效、无缝地集成到高性能物理模拟求解器中。

    

    机器学习在物理应用中已展现出前景，但将基于Python的库集成到高性能C++求解器中可能会造成性能瓶颈。我们提出了CodeJeNN，它通过从训练好的Keras模型自动生成自包含的C++推理代码来填补这一空白。该方法通过内联函数消除了外部依赖，实现了与现有框架的无缝集成。我们描述了Keras到C++的工作流程、所支持的架构以及局限性。通过与Keras（即时执行模式和优化模式）的推理基准测试，以及一项氢气-空气混合层的计算流体动力学（CFD）模拟——其中粘度由神经网络建模——对CodeJeNN进行了演示，结果表明生成的模型能够直接集成到求解器中，同时保持高效和准确。

    arXiv:2607.02746v3 Announce Type: replace-cross  Abstract: Machine learning has shown promise for physics applications, but integrating Python-based libraries into high-performance C++ solvers can create performance bottlenecks. We present CodeJeNN, which addresses this gap by auto-generating self-contained C++ code from trained Keras models for inference. This eliminates external dependencies through inlined functions, allowing seamless integration into existing frameworks. We describe the Keras-to-C++ workflow, supported architectures, and limitations. CodeJeNN is demonstrated through inference benchmarks against Keras (eager and optimized) and a computational fluid dynamics (CFD) simulation of a hydrogen-air mixing layer in which viscosity is modeled using a neural network, showing that the generated models integrate directly into a solver while remaining efficient and accurate.
    
[^256]: Transformer应如何在电子健康记录中编码数值？

    How Should Transformers Encode Numeric Values in Electronic Health Records?

    [https://arxiv.org/abs/2607.01391](https://arxiv.org/abs/2607.01391)

    该论文系统比较了Transformer处理电子健康记录数据时离散、连续和混合三种数值编码策略，发现显式建模数值-概念交互的方法在精度敏感任务上最优，而投影前分箱的混合方法更为稳健通用，且最优分箱数与数据集规模呈幂律关系。

    

    在基于Transformer的序列处理中，我们应如何对数值进行编码，尤其是在电子健康记录（EHR）数据中？我们利用嵌入真实世界EHR数据中的合成算术任务以及真实临床预测任务，系统地比较了离散、连续和混合数值编码策略。我们的研究揭示了数值精度、优化稳定性与架构灵活性之间的权衡。我们发现，在架构约束允许的情况下，显式建模数值-概念交互的方法在精度敏感的算术任务上表现最佳。而保留数值但在投影前进行分箱的混合标记化方法则提供了一种更稳健且适用范围更广的替代方案，其最优分箱数量遵循一个简单的基于经验推导的、与数据集规模相关的幂律。在各类任务中，模型始终表现出可靠且“足够好”的数值计算能力。

    arXiv:2607.01391v2 Announce Type: replace-cross  Abstract: How do we encode numeric values in transformer-based sequence processing, particularly in electronic health record (EHR) data? We systematically compare discrete, continuous, and hybrid value encoding strategies using synthetic arithmetic tasks embedded within real-world EHR data, as well as real-world clinical prediction tasks. Our study reveals trade-offs between numeric precision, optimisation stability, and architectural flexibility. We find that approaches that explicitly model value-concept interactions perform best on precision-sensitive arithmetic tasks when architectural constraints permit. Hybrid token-based approaches that retain numeric values but apply binning prior to projection provide a more robust and broadly applicable alternative, with the optimal number of bins following a simple empirically derived power-law in dataset size. Across tasks, models consistently exhibit reliable "good enough" numeric computatio
    
[^257]: 通过随机李雅普诺夫稳定性分析大步长随机梯度下降在交叉熵损失下的收敛性

    Convergence of Large-Stepsize SGD for the Cross-entropy Loss via Stochastic Lyapunov Stability

    [https://arxiv.org/abs/2606.30930](https://arxiv.org/abs/2606.30930)

    本文借助随机李雅普诺夫稳定性理论证明，大步长SGD在多分类交叉熵损失下（含线性分类器和两层神经网络）虽会在“稳定边缘”振荡与稳定下降之间交替，但迭代点总能在固定次数内回归稳定，从而实现最佳迭代意义上的收敛。

    

    arXiv:2606.30930v2 公告类型：替换 摘要：现代深度学习已被证明能够以远超经典优化理论所能解释的学习率运行。以往对这一现象的大多数分析都集中在确定性梯度下降上，而随机设置在很大程度上尚未被探索。在这项工作中，我们为应用于多分类交叉熵损失的随机梯度下降（SGD）提供了精确的收敛保证，涵盖了线性分类器和两层神经网络。我们表明，SGD的随机性可能导致其动力学在由曲率驱动的振荡所主导的“稳定边缘”区域与期望损失以可控速率下降的稳定区域之间交替。尽管如此，我们证明了SGD能够稳定动力学，确保迭代点在固定的迭代次数内回归稳定，从而即使在学习率很大的情况下也能实现最佳迭代意义上的收敛。实验验证了我们的理论结果。

    arXiv:2606.30930v2 Announce Type: replace  Abstract: Modern deep learning has been shown to operate using learning rates far larger than those justified by classical optimization theory. Most prior analyses of this phenomenon focus on deterministic gradient descent, leaving the stochastic setting largely unexplored. In this work, we provide sharp convergence guarantees for Stochastic Gradient Descent (SGD) applied to the multiclass cross-entropy loss, for both linear classifiers and two-layer neural networks. We show that the stochasticity of SGD may cause the dynamics to alternate between an edge-of-stability regime that is dominated by curvature-driven oscillations, and a stable regime in which the expected loss decreases at a controlled rate. Despite that, we prove that SGD stabilizes the dynamics, ensuring that the iterates return to stability in a fixed number of iterations and allowing convergence in the best-iterate sense even with large learning rates. Experiments validate our 
    
[^258]: 通过神经常微分方程（Neural-ODE）近似具有植入点吸引子的速度场以用于分类任务

    Approximating velocity fields with planted point attractors via Neural-ODE for classification purposes

    [https://arxiv.org/abs/2606.23550](https://arxiv.org/abs/2606.23550)

    该论文提出在神经常微分方程（Neural-ODE）中植入预设的平衡点作为类别指示器，利用速度场的通用逼近能力塑造动力学景观并形成吸引域，从而将每个输入有效地引导至其对应的目标类别，实现分类任务。

    

    在这项工作中，配备了精心设计的平衡点集合的神经常微分方程（Neural ODEs）已被成功应用于分类任务。植入的不动点作为目标类别的指示器，而速度场则利用该架构的通用逼近能力来塑造动力学景观。这一过程定义了训练模型的吸引域，有效地将每个输入（作为初始条件提供）引导至其对应的目标类别。

    arXiv:2606.23550v3 Announce Type: replace-cross  Abstract: In this work, Neural ODEs equipped with a curated collection of equilibrium points have been successfully employed for classification tasks. The planted fixed points serve as indicators for the target classes, while the velocity field, leveraging the universal approximation capabilities of the architecture, shapes the dynamical landscape. This process defines the basins of attraction of the trained model, effectively directing each input (provided as an initial condition) toward its corresponding destination target.
    
[^259]: AlphaMemo：面向自进化Alpha挖掘智能体的结构化搜索过程记忆

    AlphaMemo: Structured Search-Process Memory for Self-Evolving Alpha Mining Agents

    [https://arxiv.org/abs/2606.20625](https://arxiv.org/abs/2606.20625)

    AlphaMemo通过从AST差异中提取编辑模式、置信度门控残差记忆和非对称否决控制构建结构化搜索过程记忆，实现了自进化的Alpha挖掘智能体，在中证500和标普500上显著提升了样本外表现和固定预算下的因子发现效率。

    

    LLM智能体通过结合金融先验知识、符号推理、可执行因子生成和反馈驱动的迭代改进，在Alpha挖掘领域展现出广阔前景。然而，它们面临着组合爆炸式的搜索空间、嘈杂且非平稳的反馈、冗余的因子发现，以及简单复用历史成功经验所带来的过拟合风险。为了应对这些挑战，我们提出了AlphaMemo，一个具有结构化搜索过程记忆的自进化Alpha挖掘智能体。与仅仅记忆最终因子或完整轨迹不同，AlphaMemo记录的是可复用的证据，即哪些编辑模式在特定的父因子上下文中有效或失败。它从抽象语法树（AST）差异中提取编辑模式，在搜索账本先验之上应用置信度门控的残差记忆，并采用非对称否决控制机制来抑制高置信度的失败模式。在中证500和标普500指数上的实验表明，该方法的样本外表现和固定预算下的因子发现效率均得到提升。

    arXiv:2606.20625v2 Announce Type: replace  Abstract: LLM agents are promising for alpha mining via combining financial priors, symbolic reasoning, executable factor generation, and feedback-driven refinement. Yet, they face a combinatorial search space, noisy non-stationary feedback, redundant discoveries, and overfitting risks from naively reusing past successes. To address these challenges, we propose AlphaMemo, a self-evolving alpha mining agent with Structured Search-Process Memory. Rather than memorizing only final factors or full trajectories, AlphaMemo records reusable evidence about which edit motifs work or fail under specific parent-factor contexts. It extracts motifs from Abstract Syntax Tree (AST) differences, applies confidence-gated residual memory on top of a search-ledger prior, and uses asymmetric veto control to suppress high-confidence failure patterns. Experiments on CSI 500 and S&P 500 show improved out-of-sample performance and fixed-budget discovery efficiency, w
    
[^260]: 漂移与污染环境下基于潜在聚类几何的在线分布预测

    Online Distributional Prediction via Latent Cluster Geometry Under Drift and Corruption

    [https://arxiv.org/abs/2606.18778](https://arxiv.org/abs/2606.18778)

    该论文提出一种在漂移与对抗污染环境下基于潜在聚类几何配置的在线分布预测方法，利用Gibbs拟后验平均与可逆跳转MCMC进行预测，并给出了将PAC-贝叶斯复杂度、污染敏感后验扰动与漂移效应分离的累积Wasserstein-1遗憾界。

    

    非平稳数据流中的在线学习通常被表述为跟踪一个点估计，然而许多应用需要预测完整的数据生成分布。我们研究了在漂移和有界因果对抗污染下的在线分布预测问题。候选分布由可变大小的潜在聚类配置来表示，这些配置诱导出预测分布。通过在配置空间上构建Gibbs拟后验并进行后验平均，可以得到一个在线预测器，并可使用可逆跳转MCMC进行采样。该方法不依赖数据流的似然函数，而是指定了一个核混合预测族。我们针对时变的真实分布评估累积Wasserstein-1遗憾，并研究了周期性重启、滑动窗口和指数折扣记忆与全历史记忆的对比。所给出的界通过 $A_T^{\mathrm{OT}}=\sum_{t=2}^T W_2^...$ 将PAC-贝叶斯复杂度、污染敏感的后验扰动和漂移这三部分效应分离开来。

    arXiv:2606.18778v2 Announce Type: replace-cross  Abstract: Online learning in non-stationary streams is often formulated as tracking a point estimate, although many applications require predicting the full data-generating distribution. We study online distributional prediction under drift and bounded causal adversarial corruption. Candidate laws are represented by variable-size latent cluster configurations that induce predictive distributions. A Gibbs quasi-posterior over configurations yields an online predictor by posterior averaging and can be sampled using reversible-jump MCMC. The method uses no stream likelihood but specifies a kernel-mixture predictive family. We evaluate cumulative Wasserstein-1 regret against the time-varying true law and study periodic restart, sliding-window, and exponential-discounting memory against full history. The bounds separate PAC-Bayesian complexity, corruption-sensitive posterior perturbation, and drift through $A_T^{\mathrm{OT}}=\sum_{t=2}^T W_2^
    
[^261]: DeFAb：一个通过可废止溯因实现有据科学发现的数据集

    DeFAb: A Dataset for Grounded Scientific Discovery through Defeasible Abduction

    [https://arxiv.org/abs/2606.18557](https://arxiv.org/abs/2606.18557)

    该论文提出了基于基因本体构建的DeFAb数据集与符号验证框架，用于评估模型通过可废止溯因修订科学理论的能力，并揭示了现有修订测试无法检验例外泛化性的局限性。

    

    一个理论可以通过添加例外来容纳反例，而更难的问题在于决定例外应该扩展到多大范围，以及哪些预测应当得以保留。我们提出了DeFAb，这是一个基于基因本体（Gene Ontology）中经过整理的生物学知识构建的数据集和符号验证框架。模型需要填补理论中的空白，或对其规则提出例外。验证器会检查每次修改的后果，允许多种不同的解决方案获得认可。简单的符号修复即可解决已发布评估面板中的全部336个任务。这一结果暴露了修订测试的一个局限性：与单个实体绑定的例外即使没有识别出适用于其他情况的模式，也可能成功通过测试。为此，我们开发了一个情境化任务，要求所提出的例外能够触及相关案例，同时在其他地方保留原有的预测。我们计划开展两项研究，分别比较使用经验证和被拒绝的示例进行训练的效果，以及修订过程中有和没有后果反馈的差异。

    arXiv:2606.18557v2 Announce Type: replace  Abstract: A theory can accommodate a counterexample by adding an exception. The harder problem is deciding how far the exception should extend and which predictions should survive. We present DeFAb, a dataset and symbolic verification framework built from curated biological knowledge in the Gene Ontology. Models fill gaps in a theory or propose exceptions to its rules. A verifier checks the consequences of each edit, allowing different solutions to receive credit. Simple symbolic repairs solve all 336 tasks in the released evaluation panel. This result exposes a limitation of the revision test. An exception tied to one entity can succeed without identifying a pattern that applies to other cases. We develop a contextual task that requires an exception to reach the relevant cases and preserve predictions elsewhere. Two planned studies compare training with verified and rejected examples, and revision with and without feedback about consequences.
    
[^262]: 从不确定的判断到经过校准的排名：用于大语言模型评估的保形Elo估计

    From Uncertain Judgments to Calibrated Rankings: Conformal Elo Estimation for LLM Evaluation

    [https://arxiv.org/abs/2606.13221](https://arxiv.org/abs/2606.13221)

    该论文提出保形Elo估计方法，通过将校准后的获胜概率而非硬标签传入Bradley-Terry过程来量化每场对战的胜负不确定性，并对人机Elo评分的残差差距应用分割保形预测，使LLM-as-a-judge得出的模型排名与人类评分仅相差17.9 Elo平均绝对误差，同时提供具有统计保证的预测区间。

    

    评估新的大型语言模型通常需要大规模且成本高昂的人工标注工作。LLM-as-a-judge（以大模型作为评判者）提供了一种更廉价的替代方案，但评判模型自身的评分存在系统性误差——例如位置偏差、自我偏好或不可传递性——这些误差会严重破坏由此产生的排名的校准性。我们在两个互补的层面对这种评判与人类之间的分歧进行了量化。在局部层面，我们通过将校准后的获胜概率（而非硬标签）传播到Bradley-Terry程序中，从评判模型自身的评分差异中估计每场对战的胜负不确定性。仅这一方法就大幅提升了Elo估计的准确性，在LMArena上对55个留出模型取平均时，使基于大模型得出的评分与基于人类得出的评分之间的Elo平均绝对误差缩小至17.9以内。在全局层面，我们对留出模型上基于大模型得出的Elo评分与基于人类得出的Elo评分之间的残差差距应用了分割保形预测，从而生成预测区间（摘要原文在此处截断）。

    arXiv:2606.13221v3 Announce Type: replace  Abstract: Evaluating new large language models typically requires costly human annotation campaigns at scale. LLM-as-a-judge offers a cheaper alternative, but judge scores carry systematic errors -- such as position bias, self-preference, or intransitivity -- that can strongly miscalibrate the resulting rankings. We quantify the resulting judge-human disagreement at two complementary levels. At the local level, we estimate per-battle uncertainty from the judge's own score differences by propagating calibrated win probabilities rather than hard labels into the Bradley-Terry procedure. This alone provides a drastic improvement to Elo estimation accuracy, bringing LLM-derived ratings within 17.9 Elo MAE of human-derived ones when averaged over 55 held-out models on LMArena. At the global level, we apply split conformal prediction to the residual gap between LLM-derived and human-derived Elo ratings across held-out models, producing prediction int
    
[^263]: 当在严重损坏的输入上训练时，神经网络会变成质心分类器

    Neural networks become centroid classifiers when trained on heavily corrupted inputs

    [https://arxiv.org/abs/2606.11319](https://arxiv.org/abs/2606.11319)

    该研究发现神经网络对超过90%的输入损坏仍保持远高于随机水平的准确率，并通过均值场理论分析证明，在重度损坏情形下无限宽神经网络会退化为最近质心（原型）分类器。

    

    从不完美的数据中学习是机器学习的核心主题，它将鲁棒性这一实际问题与可学习性这一基本问题联系起来。本文研究属性噪声：即在标签保持完好的情况下从损坏的输入中学习，这一设定相比于标签噪声受到的分析关注要少得多。我们考虑两类损坏模型：加性噪声和替换噪声。通过在损坏的分类数据集上对多层感知机进行实验，我们发现神经网络依然保持鲁棒，即使输入损坏超过90%——远超人类识别能力的极限——其准确率仍远高于随机猜测水平。为了理解这种鲁棒性，我们采用受均值场启发的方法分析了重度损坏情形下的无限宽网络，并推导出分类结果的主导阶决策规则：网络实现了一种原型规则，即最近类……（摘要在此处截断）

    arXiv:2606.11319v2 Announce Type: replace  Abstract: Learning from imperfect data is a central theme in machine learning, connecting practical questions of robustness to fundamental questions of learnability. Here we examine attribute noise: learning from corrupted inputs while keeping the labels intact, a setting that has received considerably less analytical attention than its label-noise counterpart. We consider two types of corruption models: additive noise and replacement noise. Through experiments with multi-layer perceptrons (MLPs) on corrupted classification datasets, we find that neural networks remain robust, maintaining well-above-chance accuracy even when inputs are >90% corrupted -- far beyond human recognition. To understand this robustness, we analyze infinite-width networks in the heavy-corruption regime using a mean-field-inspired approach and derive a leading-order decision rule for the classification outcome: the network implements a prototype rule, the nearest-class
    
[^264]: 电子商务营销活动中的用户-商品联合分配

    Joint user-item allocation for e-commerce marketing campaigns

    [https://arxiv.org/abs/2606.09623](https://arxiv.org/abs/2606.09623)

    该论文首次将电子商务营销活动中的商品选择与用户定向统一形式化为“自动定向”联合分配问题，突破了现有方法固定一方、单独优化另一方的局限，在多场互斥且预算受限的活动中实现用户与商品的联合优化分配。

    

    网络平台同时开展众多营销活动，每一场活动都需要决定推广哪些商品以及定向哪些用户。最优的用户选择取决于商品，而最优的商品选择又取决于用户。各活动之间还存在相互竞争，因为每个用户和每件商品只能归属于其中一场活动，且每场活动的受众规模受其预算上限约束。然而，现有方法都是固定一方再搜索另一方：要么先选定商品组合再寻找目标受众，要么逐个用户地对商品进行排序。无论哪种方式，它们都只解决了完整问题的一个受限版本，这意味着联合优化的分配方案至少不会更差，甚至可能好得多。这种限制造成的损失有多大，以及联合问题能否在营销活动的实际运行规模上求解，目前仍不得而知。在本文中，我们将该联合任务形式化为“自动定向”：将用户和商品同时分配给多个互不重叠、容量受限的营销活动……

    arXiv:2606.09623v2 Announce Type: replace  Abstract: Web platforms run many marketing campaigns at once, and for each one they must decide which products to promote and which users to target. The best users depend on the products, and the best products on the users. The campaigns also compete with one another, because each user and each item may belong to only one of them and each audience is capped by the campaign budget. Still, existing approaches fix one side and search the other, choosing an assortment and then looking for its audience, or ranking items for one user at a time. Either way they solve a restricted version of the full problem, implying that a jointly optimized assignment would be at least as good and possibly far better. What the restriction costs, and whether the joint problem is solvable at the scale campaigns are run, remain unknown. Here we formalize the joint task as auto-targeting: assigning users and items to multiple disjoint, capacity-bounded campaigns that ma
    
[^265]: 基于结构化定位的驾驶视频复杂查询检索

    Driving Video Retrieval for Complex Queries with Structured Grounding

    [https://arxiv.org/abs/2606.09109](https://arxiv.org/abs/2606.09109)

    提出STRIVE-D数据校准检索框架，通过弱标注视频评估并自适应调整查询规则，将校准后的规则分数与视觉-语言及关键词检索信号融合，从而在驾驶视频中可靠检索加塞、急刹等复杂动态事件。

    

    大规模视频检索是自动驾驶数据整理与安全验证的核心任务，用户不仅希望检索到场景，还希望找到诸如加塞、急刹车等动态事件。现有的视觉-语言检索与基于关键词的检索方法常常会遗漏这些事件，原因在于相关运动可能未在文本中被明确描述，或无法通过词汇重叠被捕获。基于规则的检索能够更直接地编码这类事件，但它较为脆弱：当生成或手写的规则与真实驾驶数据的假设不匹配时往往会失效。我们提出了STRIVE-D，一个面向驾驶视频的数据校准检索框架。该框架利用弱标注的域内视频来估计查询规则何时可靠，对与观测数据不匹配的规则进行自适应调整，并将校准后的规则分数与视觉-语言检索及关键词检索信号相融合。在三个驾驶基准上的实验（包括新发布的Dr...（摘要原文在此处被截断）

    arXiv:2606.09109v3 Announce Type: replace-cross  Abstract: Video retrieval at scale is central to data curation and safety validation in autonomous driving, where users want to find not only scenes but also dynamic events such as cut-ins and hard braking. Existing vision-language and keyword-based retrieval methods often miss these events because the relevant motion may not be explicitly described in text or captured by lexical overlap. Rule-based retrieval can encode such events more directly, but it is brittle: generated or hand-written rules often fail when their assumptions do not match real driving data. We propose STRIVE-D, a data-calibrated retrieval framework for driving videos. It uses weakly labeled in-domain videos to estimate when a query rule is reliable, adapt rules that mismatch observed data, and fuse calibrated rule scores with vision-language and keyword-based retrieval signals. Across three driving benchmarks, including newly released human-annotated event data on Dr
    
[^266]: 先刻画再蒸馏：大输出空间中的机制化推理

    Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces

    [https://arxiv.org/abs/2606.06840](https://arxiv.org/abs/2606.06840)

    本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。

    

    经过推理训练的语言模型能够以零样本方式执行多标签任务，即需要从数千到数十万个候选标签中选出一小部分相关标签。我们探讨它们在机制层面是如何完成这一任务的，以及该机制能否被蒸馏。我们通过将每个决策视为一个由模型自身决策边际（decision margin）评分的token级事件，使这一问题变得可测量：包括选定标签空间粗略区域的token、在该区域内选定具体标签的token，以及输出偏离推理过程中早先提到的接近替代方案（即“近似失误”）的token。通过归因分析、精确平均消融、向其他示例上下文中植入（knock-in）实验，以及对通用注意力头进行折扣的零假设校准，这些方法赋予了单个注意力头因果地位。在医院出院小结的临床编码任务（MIMIC-IV）上，在上下文中包含全部5,651个候选诊断代码的情况下，一个小型的、全局的、具有阶段结构的……（摘要原文在此处截断）

    arXiv:2606.06840v2 Announce Type: replace-cross  Abstract: Reasoning-trained language models can perform, zero-shot, multi-label tasks that require selecting a small set of relevant labels from a universe of thousands to hundreds of thousands of candidates. We ask how they do it mechanistically, and whether the mechanism can be distilled. We make the question measurable by treating each decision as a token-level event scored by the model's own decision margin: the token that picks a coarse region of the label space, the tokens that pick a label within it, and the token where the output departs from a close alternative (a near-miss) named earlier in the reasoning. Attribution, exact mean-ablation, knock-in into another example's context, and a null calibration that discounts generic heads then give individual attention heads causal standing. On clinical coding of hospital discharge summaries (MIMIC-IV), with all 5,651 candidate diagnosis codes in context, a small, global, phase-structur
    
[^267]: 级联梯度方法中的瞬态放大：精确极小极大定律与无共振跟踪

    Transient Amplification in Cascaded Gradient Methods: Exact Minimax Laws and Tracking without Resonance

    [https://arxiv.org/abs/2606.04031](https://arxiv.org/abs/2606.04031)

    该论文针对级联梯度方法首次给出精确且与维度无关的最坏瞬态放大极小极大公式，证明最坏瞬态为 1 + c/(1-γ) 量级、仅凭特征值无法预测有限时间增益，并在鞅噪声下确立了三次逆间隙缩放的跟踪方差定律。

    

    主从梯度迭代（包括双层优化和双时间尺度方法）尽管具有渐近稳定性，但仍可能出现较大的有限时间偏移。对于具有分块矩阵 J = [[A,0],[C,D]] 的级联迭代，其中对称对角块的范数至多为 γ < 1，耦合范数至多为 c，我们证明了一个精确的、与维度无关的极小极大公式：在任意时间范围 t，最坏情况的放大倍数等于矩阵 [[γ^t, 0], [c·t·γ^(t-1), γ^t]] 的谱范数，且该值可由一个对齐的二维二次问题达到。因此，最坏瞬态的量级为 1 + c/(1-γ)，且仅凭特征值无法预测有限时间增益。在鞅噪声下，对齐系统的平稳跟随者方差为 σ²c²(1+γ²)/(1-γ²)³，确立了三次逆间隙缩放规律。对于二次型跟随者，耦合与跟随者敏感度 R 通过 C = (I-D)R 相关联。当领导者为各向同性且跟随者映射为正……（原文摘要在此处截断）

    arXiv:2606.04031v3 Announce Type: replace-cross  Abstract: Leader-follower gradient iterations, including bilevel and two-time-scale methods, can exhibit large finite-time excursions despite asymptotic stability. For cascaded iterations with block matrix J = [[A,0],[C,D]], symmetric diagonal blocks of norm at most gamma < 1, and coupling norm at most c, we prove an exact, dimension-free minimax formula: at every horizon t, the worst-case amplification equals the spectral norm of [[gamma^t,0],[c t gamma^(t-1),gamma^t]], attained by an aligned two-dimensional quadratic. Consequently, the worst transient is on the order of 1 + c/(1-gamma), and eigenvalues alone cannot predict finite-time gain. Under martingale noise, the aligned system has stationary follower variance sigma^2 c^2 (1+gamma^2)/(1-gamma^2)^3, establishing cubic inverse-gap scaling. For quadratic followers, the coupling is linked to follower sensitivity R by C = (I-D)R. When the leader is isotropic and the follower map is pos
    
[^268]: 迭代纳什偏好优化的高效探索

    Efficient Exploration for Iterative Nash Preference Optimization

    [https://arxiv.org/abs/2606.01382](https://arxiv.org/abs/2606.01382)

    论文提出探索式纳什偏好优化（ENPO），通过SFT型正则化与对抗性探索机制，解决了迭代NLHF中隐式探索不足导致的KL正则参数指数级依赖问题，为在线迭代纳什学习提供了理论保证。

    

    偏好对齐是提升大语言模型（LLM）性能的核心，但基于奖励的建模方式在人类偏好具有非传递性时会受到限制。从人类反馈中进行纳什学习（NLHF）通过将对齐建模为偏好博弈并求解纳什均衡来解决这一局限。然而，可扩展NLHF的学习理论基础仍然有限：现有的遗憾保证依赖于显式的偏好模型估计和极小极大预言机，而更简单的迭代方法则缺乏此类保证。我们研究在线迭代NLHF，并发现探索是其中的关键障碍。首先，我们证明标准的迭代NLHF可能会对KL正则化参数的倒数产生指数级依赖，这表明通过策略更新实现的隐式探索可能是不充分的。随后，我们提出探索式纳什偏好优化（ENPO），该方法将SFT型正则化与对抗性……

    arXiv:2606.01382v2 Announce Type: replace-cross  Abstract: Preference alignment is central to improving large language models (LLMs), but reward-based formulations can be restrictive when human preferences are non-transitive. Nash learning from human feedback (NLHF) addresses this limitation by modeling alignment as a preference game and seeking a Nash equilibrium. However, the learning-theoretic foundations of scalable NLHF remain limited: existing regret guarantees rely on explicit preference-model estimation and minimax oracles, whereas simpler iterative methods lack such guarantees. We study online iterative NLHF and identify exploration as a key obstacle. First, we show that standard iterative NLHF can incur an exponential dependence on the inverse KL-regularization parameter, demonstrating that implicit exploration through policy updates can be insufficient. We then propose Exploratory Nash Preference Optimization (ENPO), which combines a SFT-type regularization with adversarial 
    
[^269]: SoundnessBench：你的AI科学家真的能分辨好研究想法与坏研究想法吗？

    SoundnessBench: Can Your AI Scientist Really Tell Good Research Ideas from Bad Ones?

    [https://arxiv.org/abs/2605.30329](https://arxiv.org/abs/2605.30329)

    提出了包含1,099个真实研究提案的SoundnessBench基准，评估发现12个前沿大语言模型普遍存在乐观偏见（平均假阳性率74.0%），无法可靠地在投入资源前甄别研究想法的方法论缺陷。

    

    自主AI研究智能体旨在通过自动化从假设生成到同行评审的整个研究流程来加速科学发现。然而，现有基准很少测试一个根本性的瓶颈：大语言模型能否在投入时间和计算资源**之前**判断一个研究想法的方法论可行性。如果没有可靠的“第一道关卡”过滤器，自主智能体可能会规模化推广有缺陷的方法论，而非加速有意义的科学进展。我们提出了SoundnessBench，这是一个精心构建的基准，包含1,099个真实研究提案，具有来源验证的提取方式、来自评审员的代理标签，以及对其中327个提案子集的仅基于提案内容的专家判断。通过对12个前沿大语言模型的评估，我们揭示了一种普遍存在的**乐观偏见**：在标准提示下，平均假阳性率高达74.0%。采用激进提示可以减少错误批准，但会显著增加错误拒绝。

    arXiv:2605.30329v2 Announce Type: replace  Abstract: Autonomous AI research agents aim to speed up scientific discovery by automating the research pipeline from hypothesis generation to peer review. However, existing benchmarks rarely test a fundamental bottleneck whether LLMs can judge the methodological viability of a research idea $\textbf{before}$ investing time and computational resources. Without a reliable ``first-gate'' filter, autonomous agents risk scaling flawed methodology rather than accelerating meaningful science. We introduce $\textbf{SoundnessBench}$, a curated benchmark of 1,099 real research proposals, with source verified extraction, reviewer-derived proxy labels, and proposal-only expert judgments on a subset of 327 proposals. Across 12 frontier LLMs, our evaluation reveals a pervasive $\textbf{optimism bias}$: standard prompting yields a $74.0\%$ mean false-positive rate. Aggressive prompting reduces false approvals but substantially increases false rejections. Ad
    
[^270]: 依赖实例的随机Lipschitz老虎机问题

    Instance-dependent Stochastic Lipschitz bandit

    [https://arxiv.org/abs/2605.29748](https://arxiv.org/abs/2605.29748)

    该论文提出了一种新的分析与算法，通过Lipschitz函数在其水平集上的次优性间隙积分来刻画遗憾，从而得到了真正依赖实例的遗憾界，能够适应水平集的局部增长结构而不仅仅是其渐近行为。

    

    我们研究Lipschitz老虎机问题，其中学习者通过带噪的点评估，在域 $\mathcal{X} \subset [0,1]^d$ 上顺序地最大化一个未知的Lipschitz函数 $f$。现有的遗憾界要么是最坏情况下的，其量级为 $\tilde{\Theta}(T^{(d+1)/(d+2)})$，要么是通过zooming维度 $d_z$ 实现自适应的，得到 $\tilde{\Theta}(T^{(d_z+1)/(d_z+2)})$。然而，这类基于zooming的保证只是部分依赖实例的，因为它们仅仅依赖于近最优水平集的渐近增长，无法捕捉 $f$ 更精细的结构性质。我们提供了一种分析和算法，通过 $f$ 在其水平集上的次优性间隙的积分来刻画遗憾。这产生了能够适应水平集局部增长的遗憾界，而不仅仅是其渐近行为。作为推论，当最大化子集合的维度为 $d^\star>0$ 时，我们得到了改进的遗憾界。

    arXiv:2605.29748v2 Announce Type: replace  Abstract: We study the Lipschitz bandit problem, where a learner sequentially maximizes an unknown Lipschitz function $f$ over a domain $\mathcal{X} \subset [0,1]^d$ using noisy pointwise evaluations. Existing regret bounds are either worst-case, scaling as $\tilde{\Theta} \left ( T^{d+1/d+2}\right )$, or adaptive via the zooming dimension $d_z$, yielding $\tilde{\Theta} \left ( T^{d_z+1/d_z+2}\right )$. However, such zooming-based guarantees are only partially instance-dependent, as they depend solely on the asymptotic growth of near-optimal level sets and fail to capture finer structural properties of $f$. We provide an analysis and an algorithm that characterizes the regret through integrals of the suboptimality gap of $f$ over its level sets. This yields regret bounds that adapt to the local growth of level sets, rather than only their asymptotic behavior. As a corollary, when the set of maximizers has dimension $d^\star>0$, we obtain impr
    
[^271]: 面向可泛化推荐的因果表征学习

    Causal Representation Learning for Generalisable Recommendation

    [https://arxiv.org/abs/2605.27043](https://arxiv.org/abs/2605.27043)

    该论文提出一种基于因果表征学习的推荐方法，通过信息论解耦准则和可优化的变分下界，从处理表征中剔除受混淆因子影响的非因果信息，从而解决训练分布与服务分布不一致导致的推荐模型泛化难题。

    

    在观测数据上训练的预测模型在部署时往往难以泛化，尤其是当训练数据本身是被优化系统运行所产生的结果时。推荐系统就是一个典型的例子：它们是在交互日志上进行训练的，而这些日志受到已部署策略、用户历史行为以及平台过滤机制的混淆。其结果是，训练分布与服务阶段所评分的候选分布并不一致，这一差距使得离线指标无法可靠地预测线上表现。我们提出一种受因果表征学习（CRL）启发的方法来解决该分布偏移问题。我们提出了一种信息论解耦准则，并证明其最优解能够从处理（treatment）表征中排除依赖于混淆因子的非因果信息。我们进一步推导出一个可计算的变分下界，使该准则能够基于有限的观测数据进行优化。我们方法的适用范围比……

    arXiv:2605.27043v2 Announce Type: replace  Abstract: Predictive models trained on observational data often fail to generalise when deployed, especially when the training data is a product of the system being optimised. Recommender systems are a canonical example: they are trained on interaction logs confounded by the deployed policy, past user behaviour, and platform filtering. As a result, the training distribution differs from the candidate distribution scored at serving time, a gap that makes offline metrics unreliable predictors of online performance. We address the distribution shift problem with a method motivated by causal representation learning (CRL). We propose an information-theoretic disentanglement criterion and prove that its optimum excludes confounder-dependent non-causal information from the treatment representation. We derive a tractable variational lower bound making the criterion optimisable from finite observational data. The scope of our method is narrower than th
    
[^272]: 高效基准测试本质上就是特征选择与多元回归

    Efficient Benchmarking Is Just Feature Selection and Multiple Regression

    [https://arxiv.org/abs/2605.25773](https://arxiv.org/abs/2605.25773)

    该论文将高效基准测试重新构建为带特征选择的多元回归问题，通过结合核岭回归与mRMR特征选择算法，在预测误差和排序相关性上显著改进了现有高效基准测试方法。

    

    arXiv:2605.25773v3 公告类型：replace-cross 摘要：高效基准测试技术旨在通过仅使用基准测试题目中的一个子集来预测完整基准测试分数，从而降低评估大语言模型（LLM）的计算成本。通过将这一问题重新构建为带特征选择的多元回归的一个实例，我们发现现有高效基准测试方法只需在预测阶段采用核岭回归（kernel ridge regression）即可得到大幅改进。此外，利用一种称为最小冗余最大相关性（mRMR）的信息论特征选择算法，我们可以通过选择对预测最有价值的题目子集来进一步提升这些方法的效果。除在数据极度匮乏的情形外，这些方法在涵盖二值与连续分数的多种基准测试上，均始终取得更小的预测误差（在MAE和RMSE两方面）以及预测分数与真实分数之间更高的排序相关性（在Spearman ρ和Kendall τ两方面）。

    arXiv:2605.25773v3 Announce Type: replace-cross  Abstract: Efficient benchmarking techniques aim to lower the computational cost of evaluating LLMs by predicting full benchmark scores using only a subset of a benchmark's questions. By reframing this problem as an instance of multiple regression with feature selection, we find that existing efficient benchmarking methods can be greatly improved by simply using kernel ridge regression at the prediction stage. Additionally, using an information-theoretic feature-selection algorithm called minimum redundancy maximum relevance (mRMR), we can further improve upon these methods by selecting question subsets that will be maximally useful for prediction. Except in very data-poor settings, these approaches consistently achieve smaller prediction errors (in both MAE and RMSE), and greater ranking correlation between predicted and true scores (in both Spearman $\rho$ and Kendall $\tau$) across a range of benchmarks using both binary and continuous
    
[^273]: 逼近近似注意力的 I/O 最优性

    Approaching I/O-optimality for Approximate Attention

    [https://arxiv.org/abs/2605.23751](https://arxiv.org/abs/2605.23751)

    本文提出了受近似注意力方法启发的 I/O 高效算法，将注意力计算的 I/O 复杂度对上下文长度 n 的依赖从二次降低到几乎线性，并给出了几乎匹配的算法特定下界，证明了其接近最优性。

    

    我们重新审视了大语言模型中注意力机制的 I/O 复杂度。给定查询-键-值矩阵 Q, K, V ∈ R^{n×d}，以及一台快速内存大小为 M 的机器，目标是以快慢内存之间最少的数据传输次数计算“注意力矩阵” A = softmax(QK^⊤/√d)V。文献中现有的方法，尤其是 FlashAttention 及其变体，其 I/O 开销关于上下文长度 n 呈二次依赖，而一个平凡的下界仅需 Ω(nd) 次 I/O 来读取输入并写出输出。在这项工作中，我们受 Alman 和 Song [NeurIPS'23] 最新的近似注意力方法启发，开发了 I/O 高效的算法，证明了对 n 的依赖可以降低到几乎线性。我们还提供了在不同参数区域下的（算法特定的）下界，它们几乎匹配相应的上界，表明该算法接近最优。

    arXiv:2605.23751v2 Announce Type: replace  Abstract: We revisit the I/O complexity of attention in large language models. Given query-key-value matrices $Q,K,V\in\mathbb{R}^{n\times d}$, and a machine with fast memory size $M$, the goal is to compute the "attention matrix" $A=\text{softmax}(Q K ^{\top}/\sqrt{d}) V$ with the minimal number of data transfers between fast and slow memory. Existing methods in the literature, most notably FlashAttention and its variants, incur an I/O cost that depends quadratically on the context length $n$, while a trivial lower bound only requires $\Omega(nd)$ I/O's to read the inputs and write the output. In this work we show that the dependence on $n$ can be reduced to almost-linear, by developing I/O-efficient algorithms inspired by the recent approximate attention method of Alman and Song [NeurIPS'23]. We also provide (algorithm-specific) lower bounds in different parameter regimes, which nearly match the corresponding upper bounds, indicating that th
    
[^274]: 基于遗憾的贝叶斯优化 $(\epsilon,\delta)$-最优停止准则

    Regret-Based $(\epsilon,\delta)$-optimal Stopping Criteria for Bayesian Optimization

    [https://arxiv.org/abs/2605.22561](https://arxiv.org/abs/2605.22561)

    本文为GP-UCB提出了可证明更紧的瞬时遗憾界，并据此设计了在终止时能以高概率$1-\delta$保证$\epsilon$-最优解的贝叶斯优化停止准则。

    

    贝叶斯优化（BO）是一种广泛使用的迭代式黑盒优化方法，其利用高斯过程（GP）代理模型。在实践中，BO 通常在耗尽固定的评估预算后终止，这可能带来不必要的成本，并且无法对解的质量提供最优性保证。近期在开发实用停止准则方面的研究已取得经验性进展，但理论上可靠的停止准则仍在探索之中。在本工作中，我们为 GP 置信上界（GP-UCB）在任意给定迭代时刻提出了可证明更紧的瞬时遗憾界。然后，我们基于这一更紧的界为 GP-UCB 提出了停止准则，该准则可确保在终止时以高概率 $1-\delta$ 获得 $\epsilon$-最优解。我们通过数值实验验证并展示了所提停止准则的有效性和高效性。

    arXiv:2605.22561v2 Announce Type: replace  Abstract: Bayesian optimization (BO) is a widely used iterative black-box optimization method that utilizes Gaussian process (GP) surrogate models. In practice, BO is typically terminated after a fixed evaluation budget is exhausted, which can incur unnecessary cost and provides no optimality guarantee on solution quality. Recent research in developing a practical stopping criterion has made empirical progress, yet a theoretically sound stopping criterion remains a work in progress. In this work, we present provably tighter instantaneous regret bounds for GP upper confidence bound (GP-UCB) at any given iteration. Then, we propose stopping criteria for GP-UCB based on this tighter bound that ensures an $\epsilon$-optimal solution with high probability $1-\delta$ upon termination. Numerical experiments are performed to validate and demonstrate the effectiveness and efficiency of our stopping criteria.
    
[^275]: 塑造先验：合成任务分布如何决定表格基础模型的质量

    Shaping the Prior: How Synthetic Task Distributions Determine Tabular Foundation Model Quality

    [https://arxiv.org/abs/2605.18971](https://arxiv.org/abs/2605.18971)

    提出O'Prior组合式真实感合成先验（包含分层SCM元生成器、真实感引擎、压力注入模块与防泄漏课程生成协议），在固定架构与算力的条件下证明合成任务分布的设计是决定表格基础模型质量的关键因素。

    

    是什么决定了表格基础模型的质量？与语言或视觉领域不同，表格基础模型的归纳偏置几乎完全来自合成预训练分布，但人们对这些分布的设计仍然知之甚少。标准的合成先验过于“良态”：它们忽略了决定部署鲁棒性的不规则性与失效模式。我们提出O'Prior——一个组合式真实感先验，由四个相互耦合的组件构建：一个覆盖多样函数族的分层SCM元生成器；一个涵盖异质边缘分布、缺失机制与目标变换的模块化真实感引擎；一个显式注入混杂因素与支持集-查询失配的压力模块；以及一个由课程机制管控、防泄漏的生成协议。为了将先验设计隔离为唯一的科学变量，我们在固定架构、优化器和计算预算的前提下，仅改变合成任务分布……

    arXiv:2605.18971v2 Announce Type: replace-cross  Abstract: What determines the quality of a tabular foundation model? Unlike language or vision, tabular foundation models acquire their inductive biases almost entirely from synthetic pretraining distributions, yet the design of these distributions remains poorly understood. Standard synthetic priors are too well-behaved: they omit the irregularities and failure modes that determine deployment robustness. We introduce O'Prior, a compositional realism prior built around four coupled components: a hierarchical SCM meta-generator spanning diverse functional families; a modular realism engine covering heterogeneous marginals, missingness, and target transforms; an explicit stress module injecting confounding and support-query mismatch; and a curriculum-governed, leakage-safe generation protocol. To isolate prior design as the scientific variable, we hold architecture, optimizer, and compute budget fixed and vary only the synthetic task distr
    
[^276]: 用于信用风险预测的表格基础模型：专门综述与实证比较

    Tabular Foundation Models for Credit Risk Prediction: A Dedicated Review and Empirical Comparison

    [https://arxiv.org/abs/2605.18147](https://arxiv.org/abs/2605.18147)

    本文系统综述并实证比较了表格基础模型在信用风险预测中的应用，指出其借助上下文学习利用大规模预训练先验知识，有望在小数据场景下超越梯度提升等传统模型并解决长期存在的挑战。

    

    预测模型是信用风险管理的核心，违约概率和损失估计的准确性直接影响贷款业务的盈利能力和金融系统的稳定性。数十年的基准测试研究已经巩固了现有技术水平：梯度提升模型通常代表最强的机器学习基准，而线性模型和逻辑回归模型仍然是重要的监管与行业参考模型。近期出现的表格数据基础模型为预测建模引入了一种新范式——它们不再将模型拟合到目标数据集上，而是通过上下文学习，将来自目标数据集的信息与通过在多样化（合成）表格数据集合上进行大规模预训练所获得的先验知识相结合。我们推测，获取目标数据之外的信息在小数据场景中是有益的，并且可能有助于解决长期存在的挑战……

    arXiv:2605.18147v3 Announce Type: replace  Abstract: Predictive models are central to credit risk management, where the accuracy of default-probability and loss estimates affects the profitability of lending and the stability of the financial system. Decades of benchmarking research have consolidated the state of the art, with gradient-boosting models often representing the strongest machine-learning benchmark, while linear and logistic models remain important regulatory and industry reference models. The recent advent of tabular data foundation models introduces a new paradigm for predictive modeling. Rather than fitting a model to a target dataset, they use in-context learning to integrate information from the target dataset with prior knowledge acquired through large-scale pretraining on diverse collections of (synthetic) tabular data. We conjecture that access to information beyond the target data is beneficial in small-data settings and may help address longstanding challenges, in
    
[^277]: 多智能体系统中的时间公平分配：从精确的交替度量到可扩展的协调代理指标

    Temporal Fair Division in Multi-Agent Systems: From Precise Alternation Metrics to Scalable Coordination Proxies

    [https://arxiv.org/abs/2605.14879](https://arxiv.org/abs/2605.14879)

    该论文提出了“轮转周期性”这一计算高效的公平性度量指标家族，通过等待时间规律性和访问频率均衡性两个互补维度，区分有序轮流与不规则访问模式，为多智能体系统中的时间公平分配提供了可扩展的协调评估方法。

    

    许多多智能体游戏和模拟环境中，多个独立的、通常具有学习能力的智能体反复争夺共享资源。传统的公平性度量方法只能评估资源总体上是否被平均分配，却无法区分有序的轮流访问与不规则的访问模式——尽管两者的累积结果相似，但后者会导致漫长且不可预测的等待时间。我们提出了轮转周期性，这是一系列计算高效的度量指标，结合了两个互补的测量维度：成功访问之间等待时间的规律性，以及各智能体之间访问频率的均衡性。我们在一个重复的阈值拥塞博弈——“蜂蜜罐博弈”中，将这些指标与更精细的交替度量系列进行对比评估。在该博弈中，二到十个强化学习智能体竞争独占访问一个共享的高价值资源。我们的实验表明，独立训练的Q学习智能体通常……（摘要在此处截断）

    arXiv:2605.14879v3 Announce Type: replace-cross  Abstract: Many multi-agent games and simulated environments involve several independent, often learning, agents repeatedly competing for a shared resource. Conventional fairness measures assess whether resources are shared equally overall, but cannot distinguish orderly turn-taking from irregular access patterns that produce long, unpredictable waiting times despite similar cumulative outcomes. We introduce Rotational Periodicity, a family of computationally efficient metrics combining two complementary measurements, the regularity of waiting times between successful accesses and the balance of access frequencies across agents. We evaluate them against a more detailed family of alternation metrics in a repeated threshold-congestion game, the Honey-Jar Game, in which two to ten reinforcement-learning agents compete for exclusive access to a shared high-value resource. Our experiments show that independently trained Q-learning agents often
    
[^278]: NodeSyn：面向AI评估的社会对齐合成数据

    NodeSynth: Socially Aligned Synthetic Data for AI Evaluation

    [https://arxiv.org/abs/2605.14381](https://arxiv.org/abs/2605.14381)

    NodeSyn通过将微调的分类体系生成器与可信文献来源和敏感属性相结合，生成反映真实社会背景的敏感领域评估数据，其诱发的模型失败率达到通用合成基线的3.1倍、真实人类查询的1.6倍，显著优于现有合成数据方法。

    

    生成式人工智能的最新进展为模型评估所需的大规模合成数据生成提供了便利。然而，若缺乏针对性的方法，这些数据集往往缺乏敏感领域所需的社会技术层面的细微差别。我们提出了NodeSyn，这是一种将经过微调的分类体系生成器（TaG）与可信的已发表文献来源以及提取的敏感属性相结合的方法，用于生成反映真实社会背景和细微危害的查询。在使用经人工验证的自动评估器，于医疗建议和自我伤害两个领域对十个开源及专有大语言模型进行评估时，NodeSyn在10个模型中的8个上于每个领域均排名第一。NodeSyn数据集引发的平均失败率是通用合成基线的3.1倍，是结构化合成基准的2.4倍，是真实人类撰写查询的1.6倍。消融研究证实，在控制结构混杂因素的情况下，我们细粒度的第3级分类体系…

    arXiv:2605.14381v3 Announce Type: replace-cross  Abstract: Recent advancements in generative AI facilitate large-scale synthetic data generation for model evaluation. However, without targeted approaches, these datasets often lack the sociotechnical nuance required for sensitive domains. We introduce NodeSyn, a methodology that pairs a fine-tuned taxonomy generator (TaG) with credible published sources and extracted sensitive attributes to generate queries reflecting authentic societal contexts and nuanced harms. Evaluated across ten open-weight and proprietary LLMs in Medical Advice and Self-Harm using human-validated autoraters, NodeSyn ranked first on 8 of 10 models in each domain. NodeSyn datasets induced average failure rates 3.1 times higher than generic synthetic baselines, 2.4 times higher than structured synthetic benchmarks, and 1.6 times higher than authentic human-authored queries. Ablation studies confirm that, controlling for structural confounds, our granular Level 3 tax
    
[^279]: 水印应被视为一种监控原语

    Watermarking Should Be Treated as a Monitoring Primitive

    [https://arxiv.org/abs/2605.13095](https://arxiv.org/abs/2605.13095)

    论文主张水印不仅是防御移除和伪造的安全技术，更应被视为一种监控原语——同样的水印信号可用于将生成内容追溯至具体生成实体，即使是零比特水印也能被内部观察者用于个体归因，外部观察者在特定配置下也能识别内容来源。

    

    水印是生成式模型提供商满足合成内容标记法律要求的主要手段。水印将一个隐藏信号嵌入到生成内容中，持有密钥的检测器随后可以恢复该信号。其安全性主要针对的是在单个输出中移除或伪造该信号的对手。我们主张，水印还应作为一种“监控原语”来进行评估和治理，因为同样的信号可以将输出与其生成实体关联起来。我们表明：（i）拥有检测器访问权限的内部观察者即使在零比特水印（即不编码任何身份信息的水印）下，只要每个实体持有自己的密钥，也能将输出归因于具体实体；（ii）在没有密钥或检测器的情况下，外部观察者在我们所测试的配置中也能识别出未见过的文本和图像输出的来源。这两种能力都依赖于评估中未报告的部署选择，即密钥粒度……（摘要在此处截断）

    arXiv:2605.13095v4 Announce Type: replace-cross  Abstract: Watermarking is a primary means for providers of generative models to meet legal requirements to mark synthetic content. A watermark embeds a hidden signal into generated content that a detector holding a secret key can later recover. Its security is evaluated primarily against adversaries who remove or forge this signal in individual outputs. We argue that watermarking should also be evaluated and governed as a \emph{monitoring primitive}, because the same signal can link outputs to the entity that generated them. We show that (i) internal observers with detector access attribute outputs to individual entities even under zero-bit watermarks, which encode no identity, when each entity holds its own key, and (ii) external observers without keys or detectors identify the source of unseen text and image outputs in the configurations we test. Both depend on deployment choices that evaluations do not report, namely how fine-grained 
    
[^280]: U-STS-LLM：一种用于流量预测与填补的统一时空引导大语言模型

    U-STS-LLM: A Unified Spatio-Temporal Steered Large Language Model for Traffic Prediction and Imputation

    [https://arxiv.org/abs/2605.11735](https://arxiv.org/abs/2605.11735)

    该论文提出U-STS-LLM统一框架，基于部分冻结的时空注意力LLM主干，利用动态时空注意力偏置生成器引导注意力，将蜂窝流量预测与缺失数据填补两大任务统一处理，解决了现有模型任务专用、时空建模解耦、级联误差累积以及过拟合等问题。

    

    准确的时空蜂窝流量分析是现代移动网络可靠运行的基础，它支撑着用于避免拥塞的负载预测，以及在传输错误情况下保证服务连续性的缺失数据填补。尽管预测与填补密切相关，但历史上它们一直作为两个独立的子领域发展。现有模型通常是任务专用的，将时间建模与空间建模解耦，并依赖会累积误差和延迟的级联设计。与此同时，将预训练大语言模型（LLM）适配到该领域时，往往对空间结构的引导不足，并且容易在有限数据上过拟合。为了弥合这些差距，我们提出了U-STS-LLM，一个构建在部分冻结的时空注意力（PFSTA）LLM主干之上的统一框架。一个动态时空注意力偏置生成器合成持久的函数图与瞬态节点状态来引导注意力，同时一个轻量级并行……（摘要不完整，原文在此处截断）

    arXiv:2605.11735v2 Announce Type: replace  Abstract: Accurate spatio-temporal cellular traffic analysis is fundamental to the reliable operation of modern mobile networks, as it underpins load forecasting for congestion avoidance and missing-data imputation for continuity despite transmission errors. While deeply connected, forecasting and imputation have historically evolved as separate sub-fields. Existing models are typically task-specific, decouple temporal and spatial modeling, and rely on cascaded designs that accumulate errors and latency. Meanwhile, adapting pre-trained LLMs for this domain often leaves spatial structure under-guided and tends to overfit on limited data. To bridge these gaps, we propose U-STS-LLM, a unified framework built on a partially frozen spatio-temporally attentive (PFSTA) LLM backbone. A Dynamic Spatio-Temporal Attention Bias Generator synthesizes a persistent functional graph with transient nodal states to steer attention, while a lightweight parallel 
    
[^281]: 吸引子-血管耦合理论：基于智能手机光电容积脉搏波图的无袖带血压估计的形式化基础与实证验证（符合AAMI标准）

    Attractor-Vascular Coupling Theory: Formal Grounding and Empirical Validation for AAMI-Standard Cuffless Blood Pressure Estimation from Smartphone Photoplethysmography

    [https://arxiv.org/abs/2605.10871](https://arxiv.org/abs/2605.10871)

    该论文提出吸引子-血管耦合理论（AVCT），从PPG吸引子几何结构出发构建可解释的血压预测框架，实现了符合AAMI标准的智能手机无袖带血压估计，校准后收缩压/舒张压平均绝对误差仅2.05/1.67 mmHg。

    

    我们提出了吸引子-血管耦合理论（AVCT），这是一个理论指导的框架，用于从光电容积脉搏波图（PPG）吸引子几何结构中推导可解释的血压（BP）预测因子。AVCT将血管动力学、PPG观测、延迟坐标重构和吸引子描述符连接在一个统一的框架内，以形成可检验的假设，而不预先假定某个特定低维特征集具有生理充分性。我们在两个互补的实验设置下，使用留一受试者交叉验证（LOSO-CV）评估了基于LightGBM的血压估计器。实验1包含来自BIDMC和VitalDB数据库的46名ICU/手术受试者，拥有心电图（ECG）、PPG及重复的参考血压测量数据。校准后的完整模型实现了收缩压/舒张压（SBP/DBP）平均绝对误差（MAE）为2.05/1.67 mmHg，而仅使用PPG吸引子特征的模型则为2.02/1.63 mmHg。实验2包括217名动态（门诊）……（原文摘要在此处截断）

    arXiv:2605.10871v3 Announce Type: replace-cross  Abstract: We present Attractor-Vascular Coupling Theory (AVCT), a theory-guided framework for deriving interpretable blood pressure (BP) predictors from photoplethysmography (PPG) attractor geometry. AVCT connects vascular dynamics, PPG observation, delay-coordinate reconstruction, and attractor descriptors within a unified framework for formulating testable hypotheses, without presupposing the physiological sufficiency of a particular low-dimensional feature set. We evaluate a LightGBM-based estimator using leave-one-subject-out cross-validation (LOSO-CV) across two complementary experimental settings. Experiment 1 comprises 46 ICU/surgical subjects from BIDMC and VitalDB, with ECG, PPG, and repeated reference BP measurements. The calibrated full model achieves systolic/diastolic BP (SBP/DBP) mean absolute errors (MAEs) of 2.05/1.67 mmHg, compared with 2.02/1.63 mmHg for the PPG-attractor-only model. Experiment 2 includes 217 ambulatory
    
[^282]: 广义Wasserstein流匹配：面向点云生成的嵌套耦合

    Generalized Wasserstein Flow Matching: Nested Couplings for Point Cloud Generation

    [https://arxiv.org/abs/2605.08424](https://arxiv.org/abs/2605.08424)

    该论文将流匹配推广到概率测度之上的概率测度（元测度）层面，证明只需可容许的嵌套外内耦合（无需最优传输计划或光滑流形）即可诱导有效的元测度流与非局部速度场，从而拓展了多样本流匹配的设计空间，并在点云生成中揭示了几何与成本的权衡。

    

    我们在“概率测度之上的概率测度”（即元测度）上构造流匹配。直接从Wasserstein-on-Wasserstein（WoW）连续性方程出发，我们证明可容许的嵌套外层与内层耦合能够诱导出有效的元测度流和非局部速度场。这种有效性既不需要最优传输计划，也不需要光滑流形。在实际应用中，这开启了多样本流匹配的设计空间：精确的Wasserstein计划以能量和轨迹直线性见长，而切片（sliced）与惰性线性（lazy linear）计划则提供了可扩展的替代方案。点云实验（包括受控的Bures-Wasserstein与欧氏小批量基线）展示了由此产生的几何与成本之间的权衡。

    arXiv:2605.08424v3 Announce Type: replace  Abstract: We formulate flow matching on probability measures over probability measures. Directly from the Wasserstein-on-Wasserstein (WoW) continuity equation, we show that admissible nested outer and inner couplings induce a valid metameasure flow and non-local velocity field. Validity requires neither optimal plans nor a smooth manifold. Practically, this opens up the design space of multisample flow matching: exact Wasserstein plans are distinguished by energy and straightness, while sliced and lazy linear plans yield scalable alternatives. Point-cloud experiments, including controlled Bures-Wasserstein and Euclidean minibatch baselines, demonstrate the resulting geometry-cost trade-off.
    
[^283]: POETS：通过计算高效的策略集成实现不确定性感知的LLM优化

    POETS: Uncertainty-Aware LLM Optimization via Compute-Efficient Policy Ensembles

    [https://arxiv.org/abs/2605.07775](https://arxiv.org/abs/2605.07775)

    POETS利用KL正则化策略会隐式编码奖励函数这一洞察，直接训练共享预训练权重的策略集成来捕获认知不确定性，从而以计算高效的方式将不确定性量化与LLM策略优化相结合，绕过了传统上需要先训练不确定性感知奖励模型再拟合策略的复杂嵌套流程。

    

    在黑盒优化和强化学习中，平衡探索与利用是一个核心挑战。我们提出了POETS（用于汤普森采样的策略集成，Policy Ensembles for Thompson Sampling），这是一个连接不确定性量化与策略优化的新颖框架。我们的方法基于这样一个洞察：使用Kullback-Leibler (KL)正则化训练的策略会隐式地编码潜在的奖励函数。基于这一洞察，POETS绕过了训练不确定性感知奖励模型并单独将策略拟合到该模型这一复杂且嵌套的流程。相反，我们直接训练一个策略集成，通过将隐式编码的奖励函数与在线的自举数据相匹配来捕获认知不确定性。为了克服集成大型语言模型（LLM）所带来的高昂计算与内存开销，POETS采用了一种高效的架构：该集成共享一个预训练……

    arXiv:2605.07775v2 Announce Type: replace-cross  Abstract: Balancing exploration with exploitation is a core challenge in black-box optimization and reinforcement learning. We introduce POETS ($\textbf{Po}$licy $\textbf{E}$nsembles for $\textbf{T}$hompson $\textbf{S}$ampling), a novel framework that bridges uncertainty quantification and policy optimization. Our approach is grounded in the insight that policies trained with Kullback-Leibler (KL) regularization implicitly encode an underlying reward function. Building on this, POETS bypasses the complex, nested process of training an uncertainty-aware reward model and separately fitting a policy to this model. Instead, we directly train a policy ensemble to capture epistemic uncertainty by matching implicitly encoded reward functions to online, bootstrapped data. To overcome the prohibitive compute and memory constraints of ensembling Large Language Models (LLMs), POETS utilizes an efficient architecture: the ensemble shares a pre-train
    
[^284]: Q-MMR：通过递归重加权与矩匹配进行离策略评估

    Q-MMR: Off-Policy Evaluation via Recursive Reweighting and Moment Matching

    [https://arxiv.org/abs/2605.06474](https://arxiv.org/abs/2605.06474)

    提出 Q-MMR 框架，通过自上而下的递归重加权和矩匹配学习每个数据点的权重来实现离策略评估，并在仅假设 Q 函数可实现性的条件下获得与维度无关的有限样本保证。

    

    我们提出了一种新颖的理论框架 Q-MMR，用于有限时域马尔可夫决策过程（MDP）中的离策略评估。Q-MMR 学习一组标量权重，每个数据点对应一个权重，使得重加权后的奖励能够近似目标策略下的期望回报。这些权重通过与价值函数判别器类进行矩匹配的目标，以自上而下的方式归纳地学习。值得注意的是，或许令人惊讶的是，在仅需 $Q^\pi$ 可实现性的条件下，即可为一般函数逼近建立数据相关的有限样本保证，且该边界与维度无关——也就是说，误差不依赖于函数类的统计复杂度。我们还与若干现有方法建立了联系，例如重要性采样和线性 FQE。进一步的理论分析为覆盖性这一离线强化学习中至关重要的概念的本质提供了新的认识。

    arXiv:2605.06474v3 Announce Type: replace-cross  Abstract: We present a novel theoretical framework, Q-MMR, for off-policy evaluation in finite-horizon MDPs. Q-MMR learns a set of scalar weights, one for each data point, such that the reweighted rewards approximate the expected return under the target policy. The weights are learned inductively in a top-down manner via a moment matching objective against a value-function discriminator class. Notably, and perhaps surprisingly, a data-dependent finite-sample guarantee for general function approximation can be established under only the realizability of $Q^\pi$, with a dimension-free bound -- that is, the error does not depend on the statistical complexity of the function class. We also establish connections to several existing methods, such as importance sampling and linear FQE. Further theoretical analyses shed new light on the nature of coverage, a concept of fundamental importance to offline RL.
    
[^285]: MoLS：面向大语言模型的信噪比引导的模块化学习率校准

    MoLS: SNR-Guided Module-Wise Learning-Rate Calibration for Large Language Models

    [https://arxiv.org/abs/2605.05794](https://arxiv.org/abs/2605.05794)

    MoLS 通过估计各模块的梯度信噪比并采用逆平方根缩放自动确定相对学习率乘数，以一次性校准取代了大语言模型训练中昂贵的手动模块学习率调参。

    

    大语言模型（LLM）包含功能上各异的模块，这些模块具有异构的梯度统计特性。尽管 Adam(W) 能在参数层面自适应地调整更新，但在各模块之间选择相对学习率往往依赖于成本高昂的手动调参。我们通过梯度信噪比（SNR）来研究这一问题。我们的分析识别了模块间 SNR 差异的结构性来源，并将 SNR 与 Adam 归一化平均梯度分量的衰减联系起来。基于这些发现，我们提出了基于信噪比的模块化学习率缩放方法 MoLS，这是一种轻量级方法，用一次性校准取代了手动乘数搜索。MoLS 在初始适应期结束后估计模块级的 SNR，并使用逆平方根缩放来确定相对学习率乘数。这些乘数在预热阶段被平滑引入，随后保持固定，同时全局学习率调度照常进行。（注：原摘要在此处截断）

    arXiv:2605.05794v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) contain functionally distinct modules with heterogeneous gradient statistics. Although Adam(W) adapts updates at the parameter level, selecting relative learning rates across modules often relies on costly manual tuning. We study this problem through gradient signal-to-noise ratio (SNR). Our analysis identifies structural sources of module-wise SNR differences and relates SNR to the attenuation of Adam's normalized mean-gradient component. Based on these findings, we propose \textbf{Module-wise Learning Rate Scaling via SNR (\name)}, a lightweight method that replaces manual multiplier search with one-shot calibration. MoLS estimates module-level SNRs after an initial adaptation period and uses inverse-square-root scaling to determine relative learning-rate multipliers. The multipliers are introduced smoothly during warmup and then held fixed while the global learning-rate schedule continues. Our th
    
[^286]: VLMShield：针对恶意提示的视觉语言模型高效且鲁棒的防御

    VLMShield: Efficient and Robust Defense of Vision-Language Models against Malicious Prompts

    [https://arxiv.org/abs/2604.06502](https://arxiv.org/abs/2604.06502)

    该论文提出MAFE多模态特征提取框架和轻量级安全检测器VLMShield，利用良性与恶意提示之间的分布差异，以即插即用方式高效且鲁棒地检测针对视觉语言模型的多模态恶意攻击。

    

    视觉语言模型（VLMs）由于在视觉整合过程中对齐能力减弱，面临来自恶意提示攻击的重大安全漏洞。现有防御方法在效率和鲁棒性方面存在不足。为了解决这些挑战，我们首先提出了多模态聚合特征提取框架，使CLIP能够处理长文本并将多模态信息融合为统一的表示。通过对MAFE提取特征的实证分析，我们发现良性提示与恶意提示之间存在截然不同的分布模式。基于这一发现，我们开发了VLMShield，一种轻量级安全检测器，作为一种即插即用的解决方案，能够高效识别多模态恶意攻击。大量实验表明，该方法在鲁棒性、效率和实用性等多个维度上均展现出卓越性能。我们希望通过这项工作为更安全的多模态AI部署铺平道路。代码已发布。

    arXiv:2604.06502v2 Announce Type: replace  Abstract: Vision-Language Models (VLMs) face significant safety vulnerabilities from malicious prompt attacks due to weakened alignment during visual integration. Existing defenses suffer from efficiency and robustness. To address these challenges, we first propose the Multimodal Aggregated Feature Extraction (MAFE) framework that enables CLIP to handle long text and fuse multimodal information into unified representations. Through empirical analysis of MAFE-extracted features, we discover distinct distributional patterns between benign and malicious prompts. Building upon this finding, we develop VLMShield, a lightweight safety detector that efficiently identifies multimodal malicious attacks as a plug-and-play solution. Extensive experiments demonstrate superior performance across multiple dimensions, including robustness, efficiency, and utility. Through our work, we hope to pave the way for more secure multimodal AI deployment. Code is ava
    
[^287]: 在高维事件序列中学习预测、发现与推理

    Learning to Predict, Discover, and Reason in High-Dimensional Event Sequences

    [https://arxiv.org/abs/2603.16313](https://arxiv.org/abs/2603.16313)

    本文提出将车辆诊断故障码序列视为一种“语言”，通过机器学习方法实现自动预测、错误模式发现与推理，以替代昂贵且易错的人工规则定义过程。

    

    arXiv:2603.16313v3 公告类型：替换。摘要：现代车辆中嵌入的电子控制单元（ECU）会产生大量称为诊断故障码（DTCs）的异步事件。这些离散事件形成复杂的时间序列，反映车辆子系统不断变化的健康状况。在汽车行业中，领域专家使用布尔规则手动将这些代码分组为更高级别的错误模式（EPs），以表征系统故障并确保安全。然而，随着车辆复杂性的增长，这一手动过程变得越来越昂贵、易出错且难以扩展。值得注意的是，现代车辆中独特DTC的数量与自然语言的词汇量处于同一数量级，通常数以万计。这一观察促使范式转变：将诊断序列视为一种可以被建模、预测并最终解释的语言。传统的统计方法无法捕捉其中的丰富性。

    arXiv:2603.16313v3 Announce Type: replace  Abstract: Electronic control units (ECUs) embedded within modern vehicles generate a large number of asynchronous events known as diagnostic trouble codes (DTCs). These discrete events form complex temporal sequences that reflect the evolving health of the vehicle's subsystems. In the automotive industry, domain experts manually group these codes into higher-level error patterns (EPs) using Boolean rules to characterize system faults and ensure safety. However, as vehicle complexity grows, this manual process becomes increasingly costly, error-prone, and difficult to scale. Notably, the number of unique DTCs in a modern vehicle is on the same order of magnitude as the vocabulary of a natural language, often numbering in the tens of thousands. This observation motivates a paradigm shift: treating diagnostic sequences as a language that can be modeled, predicted, and ultimately explained. Traditional statistical approaches fail to capture the ri
    
[^288]: 非平稳线性老虎机中最优臂识别问题的复杂度研究

    On the Complexity of Best-Arm Identification in Non-Stationary Linear Bandits

    [https://arxiv.org/abs/2603.10346](https://arxiv.org/abs/2603.10346)

    本文研究了非平稳线性老虎机中固定预算最优臂识别的复杂度，指出基于G-最优设计的经典复杂度度量过于悲观，因为其下界仅建立在标准基向量构成的臂集合上，掩盖了更丰富的臂集合结构可能带来的优势。

    

    我们研究非平稳线性老虎机中的固定预算最优臂识别（BAI）问题。具体而言，给定一个固定的时间预算 $T\in \mathbb{N}$、一个有限的臂集合 $\mathcal{X} \subset \mathbb{R}^d$，以及一个可能具有对抗性的未知参数序列 $\lbrace \theta_t\rbrace_{t=1}^{T}$（因此是非平稳的），学习者旨在以高概率识别出具有最大累积奖励的臂 $x_* = \arg\max_{x \in \mathcal{X}} x^\top\sum_{t=1}^T \theta_t$。在该设定下，众所周知，从G-最优设计中独立同分布地采样各臂可以得到 $\exp\left(-\Theta\left(T / H_{G}\right)\right)$ 的极小极大最优错误概率，其中 $H_{G}$ 与维度 $d$ 成正比。然而，这种复杂度度量过于悲观，因为它源自一个下界，而该下界仅考虑了由标准基向量构成的臂集合，从而掩盖了臂集合本身可能带来的任何潜在优势

    arXiv:2603.10346v2 Announce Type: replace  Abstract: We study the fixed-budget best-arm identification (BAI) problem in non-stationary linear bandits. Concretely, given a fixed time budget $T\in \mathbb{N}$, finite arm set $\mathcal{X} \subset \mathbb{R}^d$, and a potentially adversarial sequence of unknown parameters $\lbrace \theta_t\rbrace_{t=1}^{T}$ (hence non-stationary), a learner aims to identify the arm with the largest cumulative reward $x_* = \arg\max_{x \in \mathcal{X}} x^\top\sum_{t=1}^T \theta_t$ with high probability. In this setting, it is well-known that i.i.d. sampling arms from the G-optimal design yields a minimax-optimal error probability of $\exp\left(-\Theta\left(T / H_{G}\right)\right)$, where $H_{G}$ scales proportionally with the dimension $d$. However, this notion of complexity is overly pessimistic, as it is derived from a lower bound in which the arm set consists only of the standard basis vectors, thus masking any potential advantages arising from arm sets 
    
[^289]: MoE-SpAc：异构边缘场景下基于投机激活效用的高效MoE推理框架

    MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios

    [https://arxiv.org/abs/2603.09983](https://arxiv.org/abs/2603.09983)

    提出MoE-SpAc框架，将投机解码创新性地重新用作内存管理的前瞻传感器，通过投机效用估计器、异构工作负载均衡器和异步执行引擎，在统一的效用空间中协调专家预取与逐出，使异构边缘设备上的MoE推理TPS较SOTA基线提升42%。

    

    混合专家模型能够实现可扩展的性能，但在边缘设备上面临严重的内存限制。由于自回归专家激活具有动态性和低信息量特性，现有的卸载策略难以应对I/O瓶颈。本文提出将投机解码重新定位，不仅作为计算加速器，更作为内存管理的信息前瞻传感器，这一提议得到了我们理论分析和实证分析的支持。因此，我们提出MoE-SpAc，一个MoE推理框架，它集成了用于追踪专家需求的投机效用估计器、通过在线整数优化动态划分计算的异构工作负载均衡器，以及在相同效用空间中统一预取与逐出操作的异步执行引擎。在七个基准测试上的大量实验表明，MoE-SpAc相对于基于投机解码的SOTA基线，TPS提升了42%……

    arXiv:2603.09983v2 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) models enable scalable performance but face severe memory constraints on edge devices. Existing offloading strategies struggle with I/O bottlenecks due to the dynamic, low-information nature of autoregressive expert activation. In this paper, we propose to repurpose Speculative Decoding (SD) not merely as a compute accelerator, but as an informative lookahead sensor for memory management, supported by our theoretical and empirical analyses. Hence, we introduce MoE-SpAc, an MoE inference framework that integrates a Speculative Utility Estimator to track expert demand, a Heterogeneous Workload Balancer to dynamically partition computation via online integer optimization, and an Asynchronous Execution Engine to unify the prefetching and eviction in the same utility space. Extensive experiments on seven benchmarks demonstrate that MoE-SpAc achieves a 42% improvement in TPS over the SOTA SD-based baseline, a
    
[^290]: 异构智能体协同强化学习

    Heterogeneous Agent Collaborative Reinforcement Learning

    [https://arxiv.org/abs/2603.02604](https://arxiv.org/abs/2603.02604)

    提出新的RLVR问题HACRL及协同算法HACPO，使异构智能体在训练中共享经验证的轨迹以实现双向互学习、推理时独立运行，从而最大化样本利用率与跨智能体知识传递。

    

    我们提出了异构智能体协同强化学习，这是一个新的可验证奖励强化学习（RLVR）问题，旨在解决孤立多智能体在线策略优化的低效问题。HACRL 实现了协作优化与独立执行的结合：异构智能体在训练期间共享经验证的轨迹以相互提升，同时在推理阶段独立运行。与基于 LLM 的多智能体强化学习（MARL）不同，HACRL 不需要协调部署；与在线/离线策略蒸馏不同，它实现的是异构智能体之间的双向互学习，而非单向的同构教师到学生的知识传递。基于这一问题，我们提出了 HACPO，一种协同强化学习算法，能够进行有原则的轨迹共享，以最大化样本利用率和跨智能体的知识传递。为了缓解能力差异和策略分布（原文摘要在此处被截断）

    arXiv:2603.02604v3 Announce Type: replace  Abstract: We introduce Heterogeneous Agent Collaborative Reinforcement Learning (HACRL), a new Reinforcement Learning from Verifiable Reward (RLVR) problem that addresses the inefficiencies of isolated multi-agent on-policy optimization. HACRL enables collaborative optimization with independent execution: heterogeneous agents share verified rollouts during training to mutually improve, while operating independently at inference time. Unlike LLM-based multi-agent reinforcement learning (MARL), HACRL does not require coordinated deployment, and unlike on-/off-policy distillation, it enables bidirectional mutual learning among heterogeneous agents rather than one-directional homogeneous teacher-to-student transfer. Building on this problem, we propose HACPO, a collaborative RL algorithm that enables principled rollout sharing to maximize sample utilization and cross-agent knowledge transfer. To mitigate capability discrepancies and policy distrib
    
[^291]: 标准时间序列预测基准如何迁移到金融领域？一项关于风险调整绩效的广泛研究

    How do standard time series forecasting benchmarks translate to finance? A broad study of risk adjusted performance

    [https://arxiv.org/abs/2603.01820](https://arxiv.org/abs/2603.01820)

    该研究在15年期货数据上通过波动率目标夏普比率优化任务对16种深度学习预测架构进行基准测试，发现通用Transformer和SSM等主流“最先进”模型表现不及简单的AR(1)基准，而基于循环结构的简单混合模型（变量选择网络+LSTM夏普比率最高、xLSTM对交易成本最鲁棒）表现最佳，表明标准时间序列基准上的成功难以直接迁移到金融预测任务。

    

    由于信噪比低、非平稳性和重尾分布，预测金融资产收益率极具挑战性。我们在一个15年的期货日线数据集上，通过端到端的波动率目标夏普比率优化任务，对16种深度学习时间序列预测架构进行了基准测试，其中包括Transformer、状态空间模型（SSM）、xLSTM和循环混合模型。我们的结果挑战了文献中的主流假设：若干被广泛使用的“最先进”架构，包括通用的Transformer和SSM，在现实的金融预测任务中表现不及简单的AR(1)基准模型。表现最好的是采用循环结构的相对简单的混合方法：变量选择网络与LSTM的组合取得了最高的夏普比率，而xLSTM则展现出对交易成本最强的鲁棒性。这些发现表明，在标准时间序列基准上取得的成功并不能直接转化为金融领域的成功。

    arXiv:2603.01820v2 Announce Type: replace  Abstract: Forecasting financial asset returns is challenging due to low signal-to-noise ratios, non-stationarity and heavy-tailed distributions. We benchmark 16 deep learning architectures for time series forecasting, including Transformers, State Space Models (SSMs), xLSTMs and recurrent hybrids, on a 15-year daily futures dataset using an end-to-end volatility-targeted Sharpe ratio optimisation task. Our results challenge prevailing assumptions in the literature. Several widely used 'state-of-the-art' architectures, including generic Transformers and SSMs, underperformed a simple AR(1) benchmark in a realistic financial forecasting task. The strongest results came from comparatively simple hybrid approaches using recurrent structures: a Variable Selection Network combined with an LSTM achieved the highest Sharpe ratio, while xLSTM showed the greatest robustness to transaction costs. These findings suggest that success on standard time-series
    
[^292]: 半真陈述破坏基于相似度的检索

    Half-Truths Break Similarity-Based Retrieval

    [https://arxiv.org/abs/2602.23906](https://arxiv.org/abs/2602.23906)

    本文发现CLIP模型会被“半真半假”的错误描述欺骗（相似度反而升高），并提出CS-CLIP方法，通过将描述分解为实体和关系单元并构建最小编辑的干扰样本进行对比微调，从而增强模型对细节错误的识别能力。

    

    当在文本描述中添加一个额外细节时，如果该细节是错误的，图文相似度理应下降。我们研究表明，CLIP风格的双编码器常常违背这一直觉：在原本正确的描述后附加一个看似合理但不正确的对象或关系，反而可能提高相似度得分。我们将这类情况称为“半真陈述”。在COCO数据集上，CLIP仅有40.6%的情况下更偏好正确但较短的描述，而当添加的细节是关系时，这一比例降至32.9%。我们将这一脆弱性归因于对描述各部分的弱监督：对比训练只对齐完整句子，并未显式地强制单个实体和关系与图像准确对应。我们提出CS-CLIP（组件监督CLIP），该方法将描述分解为实体和关系单元，为每个单元构建最小编辑的干扰样本，并微调模型使其对正确单元的评分高于干扰样本，同时保持标准双编码器架构……

    arXiv:2602.23906v1 Announce Type: cross  Abstract: When a text description is extended with an additional detail, image-text similarity should drop if that detail is wrong. We show that CLIP-style dual encoders often violate this intuition: appending a plausible but incorrect object or relation to an otherwise correct description can increase the similarity score. We call such cases half-truths. On COCO, CLIP prefers the correct shorter description only 40.6% of the time, and performance drops to 32.9% when the added detail is a relation. We trace this vulnerability to weak supervision on caption parts: contrastive training aligns full sentences but does not explicitly enforce that individual entities and relations are grounded. We propose CS-CLIP (Component-Supervised CLIP), which decomposes captions into entity and relation units, constructs a minimally edited foil for each unit, and fine-tunes the model to score the correct unit above its foil while preserving standard dual-encoder 
    
[^293]: 各向同性激活函数实现神经元去个体化与自适应拓扑结构

    Isotropic Activation Functions Enable Deindividuated Neurons and Adaptive Topologies

    [https://arxiv.org/abs/2602.23405](https://arxiv.org/abs/2602.23405)

    该论文提出围绕选定等变性设计的各向同性激活函数，通过仿射映射的奇异值分解将网络层对角化，使网络能够以精确或渐近保持函数的方式自适应地增删神经元，从而实现可变拓扑结构。

    

    在保持函数不变的前提下改变神经元数量是困难的，因为神经网络的密集连接在各个单元之间造成了强烈的计算相互依赖性。逐元素激活函数，特别是其有限的等变性，被认为是实现这一目标的核心障碍。我们提出了围绕选定等变性设计的激活函数，使密集神经网络能够通过其仿射映射的奇异值分解进行自适应调整。这将各层对角化为一对一的有序连接，使得更容易评估单个连接对函数的影响。低影响的神经元可以被移除（神经退化），同时维持一个由大部分不活跃的“支架”神经元组成的阈值缓冲区（神经发生）。这种由对称性引导的对角化与结构变化优先保证函数不变性：神经发生能够精确地保持函数，而神经退化能够渐近地保持函数。

    arXiv:2602.23405v3 Announce Type: replace-cross  Abstract: Varying the number of neurons while preserving function is difficult because neural networks' dense connectivity creates strong computational interdependence between units. Elementwise activation functions, particularly their limited equivariance properties, are identified as a central hurdle to achieving this. We propose activation functions designed around chosen equivariances, enabling dense neural networks to adapt via singular value decomposition of their affine maps. This diagonalises layers into one-to-one, ordered connections, making it easier to assess the impact of individual connections on function. Low-impact neurons can be removed (neurodegeneration), and a thresholded buffer of largely inactive `scaffold' neurons is maintained (neurogenesis). These symmetry-led diagonalisation and structural changes prioritise function-invariance: neurogenesis exactly preserves function, neurodegeneration is asymptotically functio
    
[^294]: 在影响之下：评估与提升大语言模型在智能体任务中的警觉性

    Under the Influence: Evaluating and Improving LLM Vigilance in Agentic Tasks

    [https://arxiv.org/abs/2602.21262](https://arxiv.org/abs/2602.21262)

    该论文针对大语言模型智能体容易被外界影响的问题，提出在现实智能体任务中评估并提升其“警觉性”，即辨别何时应采纳善意建议、何时应拒绝恶意建议的能力。

    

    大语言模型（LLMs）正日益成为自主化助手，能够通过与软件数据库、互联网等开放世界领域交互来完成一系列复杂任务。这既带来了巨大的机遇，也伴随着巨大的风险：开放领域经常出现自主智能体可能被他人影响的机会，例如决定是否接受拉取请求，或是否采纳留言板上的建议来完成任务。至关重要的是，智能体必须能够辨别何时应将善意建议纳入其推理过程，何时应拒绝恶意建议。这涉及“警觉性”这一社交能力：即判断哪些信息应当采纳、哪些信息应当舍弃，从而实现有效沟通的能力。尽管警觉性在人类中已被充分研究，但LLM智能体表现出的警觉性程度却相对未知。本文在三个现实任务中考察了LLM智能体的警觉性：导航……

    arXiv:2602.21262v4 Announce Type: replace  Abstract: Large Language Models (LLMs) are becoming increasingly autonomous assistants, capable of accomplishing series of complex tasks by interacting with open world domains such as software databases and the internet. This presents great opportunity but also great risk: open domains routinely present opportunities for autonomous agents to be influenced by others, e.g. deciding whether to accept pull requests or whether to use advice on a message board in order to accomplish a task. Critically, agents must be capable of deciphering when to incorporate benevolent advice into their reasoning, and when to reject malicious advice. This involves the social capacity of vigilance: the ability to determine which information to use, and which to discard, for effective communication. While well-studied in humans, the extent to which LLM agents exhibit vigilance is relatively unknown. Here, we examine LLM agent vigilance in three realistic tasks: navig
    
[^295]: 为什么反直觉的似然行为在表格异常检测中很少出现？

    Why Does Counterintuitive Likelihood Behavior Rarely Emerge in Tabular Anomaly Detection?

    [https://arxiv.org/abs/2602.09593](https://arxiv.org/abs/2602.09593)

    本文提出了一个领域无关的反直觉似然现象形式化定义，并通过大规模实验与理论分析证明：与图像领域不同，反直觉的似然行为在表格异常检测中很少出现，且标准化流似然可实现有竞争力的异常检测性能。

    

    具有可处理似然的深度生成模型，以标准化流（normalizing flows）为代表，为基于似然评分的异常检测提供了有效基础。我们证明，与图像领域中深度生成模型经常为异常数据分配更高似然的情况不同，这种反直觉行为在表格数据环境中很少发生。我们首先引入了一个与领域无关的形式化框架，使得能够对这一反直觉现象进行一致的检测和评估，解决了以往工作中缺乏精确定义的问题。通过对47个表格数据集（与13个基线模型进行基准对比）以及来自ADBench的10个CV/NLP嵌入数据集的大量实验，我们表明该现象在一般表格数据中始终很少见，并且标准化流似然能够实现具有竞争力的异常检测性能。我们进一步提供了理论和实证分析，将这一差异归因于……

    arXiv:2602.09593v2 Announce Type: replace-cross  Abstract: Deep generative models with tractable likelihoods, exemplified by normalizing flows, offer an effective basis for anomaly detection through likelihood-based scoring. We demonstrate that, unlike in the image domain where deep generative models frequently assign higher likelihoods to anomalous data, such counterintuitive behavior occurs rarely in tabular settings. We first introduce a domain-agnostic formulation that enables consistent detection and evaluation of the counterintuitive phenomenon, addressing the lack of a precise definition in prior work. Through extensive experiments on 47 tabular datasets benchmarked against 13 baseline models and on 10 CV/NLP embedding datasets from ADBench, we show that the phenomenon is consistently rare in general tabular data and that normalizing flow likelihood achieves competitive anomaly detection performance. We further provide theoretical and empirical analyses attributing this disparit
    
[^296]: 缓解基于流的OOD检测中的似然悖论：一种熵视角

    Mitigating the Likelihood Paradox in Flow-based OOD Detection: An Entropic Perspective

    [https://arxiv.org/abs/2602.09581](https://arxiv.org/abs/2602.09581)

    提出免训练的语义比例熵操控方法SPEM，从熵的视角通过对与分布内记忆库相似度低的输入施加更强扰动，增大对数似然差的期望下界，从而缓解基于流的OOD检测中的似然悖论，使似然排序与人类直觉对齐。

    

    能够便捷计算输入似然的深度生成模型（包括归一化流）常常对分布外（OOD）输入赋予意外高的似然值。我们从熵的视角出发，提出语义比例熵操控（SPEM）来缓解这一悖论，这是一种无需训练的方法，对与分布内（ID）记忆库相似度较低的输入施加更强的扰动。我们推导出理论下界，证明这种熵差能够以有利于分布内数据的方式增大期望对数似然差，并且在标准基准上的实验表明，这种操控使似然排序与人类直觉重新对齐。令人惊讶的是，即使只给流模型提供视觉上无意义的噪声而非原始输入，这种重新对齐依然存在，我们从理论上分析了这一现象，刻画了其发生的条件。

    arXiv:2602.09581v2 Announce Type: replace-cross  Abstract: Deep generative models that can tractably compute input likelihoods, including normalizing flows, often assign unexpectedly high likelihoods to out-of-distribution (OOD) inputs. We mitigate this paradox from an entropic perspective with Semantic Proportional Entropy Manipulation (SPEM), a training-free procedure that applies stronger perturbations to inputs less similar to an in-distribution (ID) memory bank. We derive theoretical lower bounds showing that this entropy gap increases the expected log-likelihood difference in favor of the ID, and experiments on standard benchmarks show that this manipulation realigns the likelihood ordering with human intuition. Strikingly, this realignment persists even when the flow is given only visually meaningless noise instead of the original input, and we analyze this phenomenon theoretically to characterize the conditions under which it occurs.
    
[^297]: StealthRL：用于规避多种AI文本检测器的强化学习改写攻击

    StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors

    [https://arxiv.org/abs/2602.08934](https://arxiv.org/abs/2602.08934)

    本文提出StealthRL强化学习改写攻击框架，在保持语义保真度和文本可读性的前提下，将多种AI文本检测器的平均TPR@1%FPR降至0.024，且攻击可迁移至训练中未见的检测器，揭示了AI文本检测器普遍存在的鲁棒性缺陷。

    

    AI文本检测器越来越多地被应用于高风险场景，但它们对保持语义不变的对抗性改写的鲁棒性仍不明确。我们提出了StealthRL，一个利用自适应改写攻击对检测器进行压力测试的强化学习框架。StealthRL在强制保持语义保真度的同时，针对一个检测器集成训练改写策略，随后在完整的过滤MAGE测试集上评估其对未参与训练的检测器家族的迁移能力。在RoBERTa、Fast-DetectGPT、Binoculars和MAGE等检测器上，该攻击显著降低了检测性能，将平均TPR@1%FPR降至0.024，同时保留了可观的语义重叠度和流畅可读的文本。该攻击还能迁移到训练中未使用过的检测器，表明这反映的是更广泛的鲁棒性缺陷，而非对单一目标模型的过拟合。我们还通过检测器得分分析、重复生成稳定性检验和自动化（评估）对主要结果进行了补充。

    arXiv:2602.08934v3 Announce Type: replace-cross  Abstract: AI-text detectors are increasingly used in high-stakes settings, but their robustness to meaning-preserving adversarial rewriting remains unclear. We introduce StealthRL, a reinforcement learning framework for stress-testing detectors with adaptive paraphrase attacks. StealthRL trains a paraphrase policy against a detector ensemble while enforcing semantic fidelity, then evaluates transfer to held-out detector families on the full filtered MAGE test pool. Across RoBERTa, Fast-DetectGPT, Binoculars, and MAGE, the resulting attack sharply degrades detection performance, driving mean TPR@1%FPR down to 0.024 while retaining nontrivial semantic overlap and readable prose. The attack also transfers to detectors not used during training, indicating a broader robustness failure rather than overfitting to a single target model. We complement the main results with detector-score analysis, repeated-generation stability checks, and automat
    
[^298]: 克利福德柯尔莫哥洛夫-阿诺德网络

    Clifford Kolmogorov-Arnold Networks

    [https://arxiv.org/abs/2602.05977](https://arxiv.org/abs/2602.05977)

    提出了克利福德柯尔莫哥洛夫-阿诺德网络（ClKAN），通过随机化拟蒙特卡罗网格生成方法解决高维代数的指数级扩展问题，并引入新的批归一化策略处理可变域输入，实现了在任意克利福德代数空间中灵活高效的函数逼近。

    

    我们提出了克利福德柯尔莫哥洛夫-阿诺德网络，这是一种在任意克利福德代数空间中进行函数逼近的灵活而高效的架构。我们提出使用随机化拟蒙特卡罗网格生成方法，以解决与高维代数相关的指数级扩展问题。我们的ClKAN还引入了新的批归一化策略来处理可变域输入。ClKAN在科学发现和工程领域具有应用价值，并在合成任务和物理启发任务中得到了验证。

    arXiv:2602.05977v3 Announce Type: replace-cross  Abstract: We introduce Clifford Kolmogorov-Arnold Network (ClKAN), a flexible and efficient architecture for function approximation in arbitrary Clifford Algebra spaces. We propose the use of Randomized Quasi-Monte Carlo grid generation as a solution to the exponential scaling associated with higher-dimensional algebras. Our ClKAN also introduces new batch normalization strategies to deal with variable domain input. ClKAN finds application in scientific discovery and engineering, and is validated in synthetic and physics-inspired tasks.
    
[^299]: 特征空间平滑：深度表示的认证鲁棒性

    Feature-Space Smoothing: Certified Robustness of Deep Representations

    [https://arxiv.org/abs/2601.16200](https://arxiv.org/abs/2601.16200)

    该论文提出基于高斯随机平滑的特征空间平滑（FS）框架，在表示层面为深度模型提供干净与对抗特征间余弦相似度的认证下界，并引入即插即用的高斯平滑增强器（GSB）来提升编码器的高斯鲁棒性分数。

    

    arXiv:2601.16200v4 公告类型：替换 摘要：现代深度学习模型在各类应用中展现出强大的能力，但仍然容易受到恶意输入的攻击，这些恶意输入会通过特征空间失真引发错误的预测。为了解决这一问题，我们提出了特征空间平滑，这是一个基于高斯随机平滑构建的表示级认证框架。我们证明，FS 可以将给定的特征编码器转换为一个平滑变体，该变体在 l₂ 有界扰动下能够保证维持干净特征与对抗特征之间余弦相似度的认证下界。随后，我们确立了这个特征余弦相似度界（FCSB）可以扩展到基于余弦相似度度量的预测级认证，并且 FCSB 的值由编码器固有的高斯鲁棒性分数决定。基于这些见解，我们引入了高斯平滑增强器，这是一个即插即用模块，用于提升编码器的高斯鲁棒性。

    arXiv:2601.16200v4 Announce Type: replace  Abstract: Modern deep learning models exhibit strong capabilities across diverse applications, yet remain vulnerable to malicious inputs that induce erroneous predictions via feature-space distortion. To address that, we propose Feature-Space Smoothing (FS), a representation-level certification framework built upon Gaussian randomized smoothing. We show that FS converts a given feature encoder into a smoothed variant that is guaranteed to maintain a certified lower bound on the cosine similarity between clean and adversarial features under l_2-bounded perturbations. We then establish that this Feature Cosine Similarity Bound (FCSB) can be extended to the prediction-wise certification under the cosine similarity measure, and the value of FCSB is determined by the encoder intrinsic Gaussian robustness score. Building on those insights, we introduce the Gaussian Smoothness Booster (GSB), a plug-and-play module to improve the encoder Gaussian robu
    
[^300]: 论语言建模中Transformer的跑道级联

    On the Runway Cascade of Transformers for Language Modeling

    [https://arxiv.org/abs/2601.14522](https://arxiv.org/abs/2601.14522)

    提出并形式化“跑道级联”现象，解释因果Transformer中直接路径与间接路径（跑道）之间的信息传播失调如何导致冗余信息污染标记表示，并设计“跑道感知重连”机制来显式缓解这一问题。

    

    在仅解码器（因果）Transformer中，由因果掩码构建的计算图通过直接路径注意力以及由中间标记形成的间接路径来传递信息。我们将标记对之间的这些间接路径称为其“跑道”。我们认为，近期越来越多的研究所观察到的因果Transformer的某些失效模式，很可能被这两种信息传播模式之间的失调所加剧。我们将“跑道级联”形式化为一种现象：尽管注意力模式已被充分学习，这种失调仍会导致冗余和无关信息级联至标记表示中。作为解决方案，我们提出“跑道感知重连”，这是一种将跑道上下文更直接地融入每个标记的直接路径注意力的显式方法。该机制基于每个标记的跑道全景概要来重新连接其注意力模式，使其能够感知累积（后文截断）。

    arXiv:2601.14522v2 Announce Type: replace  Abstract: In decoder-only (causal) transformers, the computation graph created by causal masking routes information through both direct-path attention and indirect paths formed by intermediate tokens. We denote these indirect paths between token pairs as their runways. We argue that certain failure modes of causal transformers as observed by a growing body of recent works are likely exacerbated by a misalignment between these two information propagation modes. We formalize runway cascade as a phenomenon whereby this misalignment results in redundancies and irrelevant information cascading to token representations despite adequately learned attention patterns. As a solution, we propose runway-aware rewiring as a more explicit way of incorporating runway context directly into each token's direct-path attention. This mechanism re-wires the attention pattern for each token based on a summary of its runway landscape, enabling awareness of accumulat
    
[^301]: 注意力空间对比引导：面向大型视觉-语言模型的高效幻觉缓解方法

    Attention-space Contrastive Guidance for Efficient Hallucination Mitigation in LVLMs

    [https://arxiv.org/abs/2601.13707](https://arxiv.org/abs/2601.13707)

    提出注意力空间对比引导（ACG），一种无需训练的单次前向传播方法，通过在自注意力层构建图像条件与纯文本对比注意力路径并结合轻量级正交投影，高效缓解大型视觉-语言模型的幻觉问题。

    

    大型视觉-语言模型（LVLMs）中的幻觉通常在语言先验压制视觉证据时产生，导致物体误识别和与视觉不符的描述。我们将幻觉缓解问题构建为一种对比引导，将生成过程引导向视觉上有依据且语义上忠实于图像的文本。我们提出了注意力空间对比引导（ACG），这是一种无需训练、仅需单次前向传播的方法，直接在自注意力层中运作，而诱发幻觉的跨模态偏差正是在这些层中出现的。ACG 在单次前向传播中同时构建图像条件注意力路径和近似的纯文本注意力路径，从而能够在错误于输出层累积之前实现高效引导。由于这种基于掩码的替代方法可能引入近似偏差，我们进一步应用轻量级的正交投影来抑制与纯文本路径对齐的成分，从而产生更忠实于视觉的输出。

    arXiv:2601.13707v3 Announce Type: replace-cross  Abstract: Hallucinations in large vision--language models (LVLMs) often arise when language priors dominate over visual evidence, leading to object misidentification and visually inconsistent descriptions. We address this problem by framing hallucination mitigation as contrastive guidance that steers generation toward visually grounded and semantically faithful text. We propose Attention-space Contrastive Guidance (ACG), a training-free, single-pass method that operates directly in self-attention layers, where hallucination-inducing cross-modal biases emerge. ACG constructs both image-conditioned and approximate text-only attention paths within a single forward pass, enabling efficient guidance before errors accumulate at the output layer. Because this masking-based surrogate can introduce approximation bias, we further apply a lightweight orthogonal projection that suppresses components aligned with the text-only path, yielding a more v
    
[^302]: 混合量子-经典神经网络中量子的作用：一项实证评估

    The Role of Quantum in Hybrid Quantum-Classical Neural Networks: An Empirical Assessment

    [https://arxiv.org/abs/2601.04732](https://arxiv.org/abs/2601.04732)

    本文通过理想无噪声模拟下的严谨统计研究，在三个医学二分类任务上系统评估了常用混合量子-经典神经网络架构，以揭示量子组件对模型整体性能的实际贡献。

    

    量子机器学习已成为近期量子硬件的一个有前景的应用领域，特别是通过同时利用经典与量子处理的混合量子-经典模型。尽管已有众多混合架构被提出并在基准任务上成功演示，但关于量子组件对这些模型整体性能的具体贡献，仍是一个重要的未解问题。在本工作中，我们旨在通过在理想的、无噪声模拟环境下开展严谨的统计研究，阐明量子处理在混合量子-经典神经网络（HQNN）架构中的影响。我们在涵盖医学信号数据以及平面和体积图像的三个二分类医学任务上，系统地评估了一系列常用的HQNN架构。在此过程中，我们检验了可归因于经典与量子方面（例如……）的影响。

    arXiv:2601.04732v2 Announce Type: replace-cross  Abstract: Quantum machine learning has emerged as a promising application domain for near-term quantum hardware, particularly through hybrid quantum-classical models that leverage both classical and quantum processing. Although numerous hybrid architectures have been proposed and demonstrated successfully on benchmark tasks, a significant open question remains regarding the specific contribution of quantum components to the overall performance of these models. In this work, we aim to shed light on the impact of quantum processing within hybrid quantum-classical neural network architectures through a rigorous statistical study under ideal, noise-free simulation. We systematically assess a selection of commonly used HQNN architectures on three binary medical classification tasks covering medical signal data as well as planar and volumetric images. In doing so, we examine the influence attributable to classical and quantum aspects such as e
    
[^303]: Dementia-R1：基于非结构化临床笔记的强化预训练与推理，用于真实世界痴呆症预后预测

    Dementia-R1: Reinforced Pretraining and Reasoning from Unstructured Clinical Notes for Real-World Dementia Prognosis

    [https://arxiv.org/abs/2601.03018](https://arxiv.org/abs/2601.03018)

    提出了Dementia-R1框架，通过冷启动强化学习策略先预训练模型预测可验证的临床指标，再基于非结构化临床笔记进行痴呆症的纵向预后预测，实现了最佳整体性能。

    

    尽管大型语言模型（LLMs）在临床文本理解方面表现出强大的性能，但它们在痴呆症预后等纵向预测任务上仍然面临困难，这类任务需要对多次就诊中复杂、非单调的症状轨迹进行推理。标准的监督式训练缺乏对症状演变的显式标注，而直接的强化学习（RL）则受到稀疏二值奖励的阻碍。为了应对这一挑战，我们提出了Dementia-R1，一个基于强化学习的框架，用于从非结构化临床笔记中进行纵向痴呆症预后预测。我们的方法采用冷启动强化学习策略，预训练模型以预测从患者病史中提取的可验证临床指标，从而在确定最终临床状态之前增强模型对疾病进展进行推理的能力。大量实验表明，Dementia-R1在AMC真实世界非结构化临床数据集上取得了最佳的整体性能。

    arXiv:2601.03018v3 Announce Type: replace-cross  Abstract: While Large Language Models (LLMs) have shown strong performance on clinical text understanding, they struggle with longitudinal prediction tasks such as dementia prognosis, which require reasoning over complex, non-monotonic symptom trajectories across multiple visits. Standard supervised training lacks explicit annotations for symptom evolution, while direct Reinforcement Learning (RL) is hindered by sparse binary rewards. To address this challenge, we introduce Dementia-R1, an RL-based framework for longitudinal dementia prognosis from unstructured clinical notes. Our approach adopts a Cold-Start RL strategy that pre-trains the model to predict verifiable clinical indices extracted from patient histories, enhancing the capability to reason about disease progression before determining the final clinical status. Extensive experiments show that Dementia-R1 achieves the best overall performance on the AMC real-world unstructured
    
[^304]: 面向金属合金性能预测的多任务学习：负迁移与缓解策略的实证研究

    Multi-Task Learning for Metal Alloy Property Prediction: An Empirical Study of Negative Transfer and Mitigation Strategies

    [https://arxiv.org/abs/2512.22740](https://arxiv.org/abs/2512.22740)

    本研究基于5.4万样本的金属合金数据集实证揭示了多任务学习中因任务函数形式不匹配导致的梯度错位负迁移现象——它损害电阻率和硬度的回归性能却提升非晶形成能力的分类召回率，并提出PCGrad以及标签分布平滑结合梯度归一化等有效缓解策略。

    

    材料科学中的多任务学习（MTL）依赖于这样一个假设：物理上相关的属性共享可学习的表示。我们使用一个包含54,028个样本、呈现极端任务级不平衡的金属合金数据集对该假设提出了挑战。我们的结果揭示了一个显著的两极分化现象：MTL显著降低了电阻率和硬度的回归性能，但提升了非晶形成能力的分类召回率。我们将这种差异追溯到函数形式的不匹配——例如电阻率的多项式依赖性与硬度的复杂交互作用——这些不匹配在优化过程中导致了严重的梯度错位。通过评估深度不平衡回归技术，我们发现投影冲突梯度（PCGrad）能够恢复单任务性能，而将标签分布平滑与梯度归一化相结合则实现了最佳的总体平衡。因此，我们提出了一个战略性框架：利用……（摘要原文在此处截断）

    arXiv:2512.22740v4 Announce Type: replace  Abstract: Multi-task learning (MTL) in materials science relies on the assumption that physically related properties share learnable representations. We challenge this assumption using a 54,028-sample metal alloy dataset exhibiting extreme task-level imbalance. Our results reveal a striking dichotomy: MTL significantly degrades regression performance for resistivity and hardness but improves classification recall for amorphous-forming ability. We trace this divergence to mismatched functional forms--such as resistivity's polynomial dependence versus hardness's complex interactions--which cause severe gradient misalignment during optimization. Evaluating Deep Imbalanced Regression techniques, we find that projecting conflicting gradients (PCGrad) recovers single-task performance, while combining label distribution smoothing with gradient normalization achieves the best overall balance. Consequently, we propose a strategic framework: utilize ind
    
[^305]: MedInsightBench：通过多模态医学数据中的多步骤洞察发现评估医学分析智能体

    MedInsightBench: Evaluating Medical Analytics Agents Through Multi-Step Insight Discovery in Multimodal Medical Data

    [https://arxiv.org/abs/2512.13297](https://arxiv.org/abs/2512.13297)

    提出首个用于评估大型多模态模型和智能体框架在多模态医学图像数据上进行多步骤医学洞察发现能力的基准MedInsightBench，包含332个精心标注的医学案例，并揭示现有模型表现仍然有限。

    

    在医学数据分析中，从复杂的多模态数据集中提取深层洞察，对于改善患者护理、提高诊断准确性和优化医疗运营至关重要。然而，目前缺乏专门设计用于评估大型多模态模型（LMMs）发现医学洞察能力的高质量数据集。在本文中，我们提出了MedInsightBench，这是首个包含332个精心筛选的医学案例的基准数据集，每个案例均标注了经过缜密设计的洞察。该基准旨在评估LMMs和智能体框架分析多模态医学图像数据的能力，包括提出相关问题、解读复杂发现，以及综合生成可操作的洞察与建议。我们的分析表明，现有的LMMs在MedInsightBench上的表现较为有限，这主要归因于它们在提取多步骤洞察方面存在困难。

    arXiv:2512.13297v2 Announce Type: replace  Abstract: In medical data analysis, extracting deep insights from complex, multi-modal datasets is essential for improving patient care, increasing diagnostic accuracy, and optimizing healthcare operations. However, there is currently a lack of high-quality datasets specifically designed to evaluate the ability of large multi-modal models (LMMs) to discover medical insights. In this paper, we introduce MedInsightBench, the first benchmark that comprises 332 carefully curated medical cases, each annotated with thoughtfully designed insights. This benchmark is intended to evaluate the ability of LMMs and agent frameworks to analyze multi-modal medical image data, including posing relevant questions, interpreting complex findings, and synthesizing actionable insights and recommendations. Our analysis indicates that existing LMMs exhibit limited performance on MedInsightBench, which is primarily attributed to their challenges in extracting multi-s
    
[^306]: 演示：生成式AI基于用户偏好辅助放射治疗计划设计

    Demo: Generative AI helps Radiotherapy Planning with User Preference

    [https://arxiv.org/abs/2512.08996](https://arxiv.org/abs/2512.08996)

    本文提出一种仅依据用户自定义偏好即可预测三维剂量分布的生成式模型，使计划制定者能够个性化权衡危及器官与靶区之间的取舍，并在适应性上超越Varian RapidPlan。

    

    放射治疗计划是一个高度复杂的过程，在不同机构和不同计划制定者之间往往存在显著差异。现有的大多数用于三维剂量预测的深度学习方法在训练时依赖参考计划作为真值（ground truth），这可能会无意中使模型偏向特定的计划风格或机构偏好。在本研究中，我们提出了一种新颖的生成式模型，仅根据用户自定义的偏好风格来预测三维剂量分布。这些可定制的偏好使计划制定者能够优先考虑危及器官（OARs）与计划靶区（PTVs）之间的特定权衡，提供了更大的灵活性和个性化。我们的方法旨在与临床治疗计划系统无缝集成，帮助用户高效地生成高质量计划。对比评估表明，我们的方法在适应性方面可以超越Varian RapidPlan模型。

    arXiv:2512.08996v2 Announce Type: replace-cross  Abstract: Radiotherapy planning is a highly complex process that often varies significantly across institutions and individual planners. Most existing deep learning approaches for 3D dose prediction rely on reference plans as ground truth during training, which can inadvertently bias models toward specific planning styles or institutional preferences. In this study, we introduce a novel generative model that predicts 3D dose distributions based solely on user-defined preference flavors. These customizable preferences enable planners to prioritize specific trade-offs between organs-at-risk (OARs) and planning target volumes (PTVs), offering greater flexibility and personalization. Designed for seamless integration with clinical treatment planning systems, our approach assists users in generating high-quality plans efficiently. Comparative evaluations demonstrate that our method can surpasses the Varian RapidPlan model in both adaptability
    
[^307]: 软符号化的Kolmogorov-Arnold网络

    Softly Symbolifying Kolmogorov-Arnold Networks

    [https://arxiv.org/abs/2512.07875](https://arxiv.org/abs/2512.07875)

    提出软符号化的Kolmogorov-Arnold网络（S2KAN），通过将符号基元直接融入训练，利用可微分的门控稀疏化机制与最小描述长度目标，在保留稠密样条灵活逼近能力的同时发现可解释的激活函数。

    

    Kolmogorov-Arnold网络（KANs）为可解释机器学习提供了一条有前景的路径：其可学习的激活函数可以被单独研究，同时能够共同准确地拟合复杂数据。然而在实践中，训练得到的激活函数往往缺乏*符号保真度*，会学习到与可解释形式没有任何有意义对应关系的病态分解。我们提出软符号化的Kolmogorov-Arnold网络（S2KAN），它将符号基元直接集成到训练过程中。每个激活函数从符号项与稠密项组成的字典中提取，并通过可学习的门控机制对表示进行稀疏化。至关重要的是，这种稀疏化是可微分的，能够实现端到端优化，并采用了受最小描述长度启发的优化目标。S2KAN能够在保留稠密样条以实现灵活逼近的同时，发现可解释的激活函数。我们引入了一种对符号……（原文摘要在此处截断）

    arXiv:2512.07875v2 Announce Type: replace-cross  Abstract: Kolmogorov--Arnold Networks (KANs) offer a promising path toward interpretable machine learning: their learnable activations can be studied individually while collectively fitting complex data accurately. In practice, however, trained activations often lack *symbolic fidelity*, learning pathological decompositions with no meaningful correspondence to interpretable forms. We propose Softly Symbolified Kolmogorov--Arnold Networks (S2KAN), which integrate symbolic primitives directly into training. Each activation draws from a dictionary of symbolic and dense terms, with learnable gates that sparsify the representation. Crucially, this sparsification is differentiable, enabling end-to-end optimization, and uses an optimization objective inspired by minimum description length. S2KAN can discover interpretable activation functions while retaining dense splines for flexible approximation. We introduce a quantitative measure of symbol
    
[^308]: 迭代GRPO：借助单轮RLHF实现多轮强化学习的批在线策略迭代

    Iterative GRPO: Batch-Online Policy Iteration for Multi-Turn RL via Single-Turn RLHF

    [https://arxiv.org/abs/2511.21638](https://arxiv.org/abs/2511.21638)

    提出Iterative GRPO方法，利用批在线部署模式收集的交互数据，将经典的近似策略迭代算法与单轮RLHF相结合，从而无需在训练循环中接入真实或模拟用户即可实现多轮强化学习。

    

    arXiv:2511.21638v2 公告类型：替换。摘要：实际的LLM智能体通常运行在多轮对话场景中，其成功与否只有在整个交互过程结束后才能确定。大多数多轮RL方法通过同策略 rollout 进行训练，但与单轮RLHF不同，策略无法独自产生完整轨迹，因为在智能体的每一轮之后，外部环境必须做出响应。对于对话式智能体而言，这个环境就是用户，但真实用户通常无法嵌入训练循环中，而忠实地构建模拟用户也十分困难。另一方面，现实世界的部署很少是完全在线或完全离线的。常见的生产模式被称为“批在线”，即先部署当前策略收集一批交互数据，然后在该批数据上重新训练并重新部署。我们证明，这种批在线设置为应用经典的近似策略迭代算法创造了机会。我们的核心观察是，运行标准的token级…

    arXiv:2511.21638v2 Announce Type: replace  Abstract: Practical LLM agents often operate over multi-turn conversations where success is determined only after the full interaction ends. Most multi-turn RL methods train via on-policy rollouts, but unlike in single-turn RLHF, the policy cannot produce a trajectory alone, since an external environment must respond after each agent turn. For conversational agents, this environment is a user, but real users are generally unavailable inside the training loop and simulated users are difficult to build faithfully. Separately, real-world deployment is rarely fully online or fully offline. The common production pattern is called "batch online," where the current policy is deployed to collect a batch of interaction data, then retrained on that batch and redeployed. We show that this batch-online setting creates an opportunity for applying the classical approximate policy iteration algorithm. Our central observation is that running standard token-le
    
[^309]: BrainRotViT：用于3D结构磁共振成像脑衰老可解释建模的Transformer-ResNet混合模型

    BrainRotViT: Transformer-ResNet Hybrid for Explainable Modeling of Brain Aging from 3D sMRI

    [https://arxiv.org/abs/2511.15188](https://arxiv.org/abs/2511.15188)

    提出了一种两阶段2.5D混合框架BrainRotViT，将预训练视觉Transformer与轻量级残差CNN相结合，从结构MRI中实现准确且可解释的脑年龄估计，验证集平均绝对误差仅3.43年。

    

    从结构磁共振成像（sMRI）准确估计脑年龄，为研究健康衰老和神经退行性疾病提供了有价值的生物标志物。我们提出了BrainRotViT（基于训练视觉Transformer的脑部ResNet），这是一个两阶段、基于切片的（2.5D）框架，在结合CNN与Transformer优势的同时近似体积上下文信息。一个视觉Transformer首先在一个辅助的年龄-性别复合分类任务上进行预训练，随后被冻结。其逐切片嵌入被堆叠为160×768的特征矩阵，并作为伪图像由一个轻量级2D残差CNN回归器进行处理，受试者性别在最终预测层被纳入。该模型在来自11个多站点数据集的5,108个样本上进行训练和验证，并在包含948个样本的四个独立留出队列上进行测试，BrainRotViT取得了验证集平均绝对误差（MAE）3.43年的结果（Pearson r = 0.98，R² = 0.96，Spearman ρ = 0.97）……

    arXiv:2511.15188v3 Announce Type: replace-cross  Abstract: Accurate brain-age estimation from structural MRI provides a valuable biomarker for studying healthy aging and neurodegenerative disease. We propose BrainRotViT (Brain ResNet over trained Vision Transformer), a two-stage, slice-based (2.5D) framework that approximates volumetric context while combining the strengths of CNNs and transformers. A Vision Transformer is pretrained on an auxiliary age-sex composite classification task and subsequently frozen. Its per-slice embeddings are stacked into a 160 x 768 feature matrix and processed as a pseudo-image by a lightweight 2D residual CNN regressor, with subject sex incorporated at the final prediction layer. Trained and validated on 5,108 samples from 11 multi-site datasets and tested on four independent held-out cohorts comprising 948 samples, BrainRotViT achieves a validation mean absolute error (MAE) of 3.43 years (Pearson r = 0.98, $R^2$ = 0.96, Spearman $\rho$ = 0.97) and cro
    
[^310]: 大语言模型强化学习中Clip-Low增加熵而Clip-High降低熵

    Clip-Low Increases Entropy and Clip-High Decreases Entropy in Reinforcement Learning of Large Language Models

    [https://arxiv.org/abs/2509.26114](https://arxiv.org/abs/2509.26114)

    该研究揭示了PPO/GRPO中的裁剪机制是RLVR中一个被忽视的混杂因素——clip-low会增加熵而clip-high会降低熵，且在标准参数下clip-high占主导地位，导致即使奖励完全随机，模型熵也会下降并引发熵坍缩。

    

    强化学习与可验证奖励（RLVR）近来已成为提升大语言模型（LLM）推理能力的主要方法。然而，RLVR容易出现熵坍缩问题，即LLM快速收敛到近乎确定性的形式，从而阻碍长时间强化学习训练中的探索与进步。在本工作中，我们揭示了PPO和GRPO中的裁剪机制会对熵产生偏差。通过理论和实证分析，我们证明clip-low会增加熵，而clip-high会降低熵。此外，在标准裁剪参数下，clip-high的影响占主导地位，导致即使向强化学习算法提供纯随机奖励，整体熵仍然会下降。我们的发现突出了RLVR中一个被忽视的混杂因素：与奖励信号无关，裁剪机制会影响熵，进而影响推理行为。

    arXiv:2509.26114v2 Announce Type: replace  Abstract: Reinforcement learning with verifiable rewards (RLVR) has recently emerged as the leading approach for enhancing the reasoning capabilities of large language models (LLMs). However, RLVR is prone to entropy collapse, where the LLM quickly converges to a near-deterministic form, hindering exploration and progress during prolonged RL training. In this work, we reveal that the clipping mechanism in PPO and GRPO induces biases on entropy. Through theoretical and empirical analyses, we show that clip-low increases entropy, while clip-high decreases it. Further, under standard clipping parameters, the effect of clip-high dominates, resulting in an overall entropy reduction even when purely random rewards are provided to the RL algorithm. Our findings highlight an overlooked confounding factor in RLVR: independent of the reward signal, the clipping mechanism influences entropy, which in turn affects the reasoning behavior. Furthermore, our 
    
[^311]: 动态关系启动机制改进多元时间序列中的Transformer

    Dynamic Relational Priming Improves Transformer in Multivariate Time Series

    [https://arxiv.org/abs/2509.12196](https://arxiv.org/abs/2509.12196)

    本文提出动态关系启动注意力机制（prime attention），通过为每个token对交互定制动态表示，显著改进了Transformer在多元时间序列建模中捕捉异构通道间依赖关系的能力。

    

    标准Transformer中的注意力机制采用静态的token表示，这些表示在每一层的所有成对计算中保持不变。这限制了其表征与每个token对交互中潜在多样关系动态之间的对齐。尽管标准注意力在关系相对同质的领域表现出色，但其静态关系学习难以捕捉多元时间序列（MTS）数据中多样且异构的通道间依赖关系——在多元时间序列中，同一系统内不同的通道对交互可能由完全不同的物理定律或时间动态所支配。为了使注意力机制更好地与这类领域现象对齐，我们提出了带动态关系启动的注意力机制（prime attention）。与标准注意力中每个token在所有成对交互中呈现相同表示不同，prime attention会为每个（token对交互）定制表示。

    arXiv:2509.12196v3 Announce Type: replace-cross  Abstract: Standard attention mechanisms in transformers employ static token representations that remain unchanged across all pair-wise computations in each layer. This limits their representational alignment with the potentially diverse relational dynamics of each token-pair interaction. While they excel in domains with relatively homogeneous relationships, standard attention's static relational learning struggles to capture the diverse, heterogeneous inter-channel dependencies of multivariate time series (MTS) data--where different channel-pair interactions within a single system may be governed by entirely different physical laws or temporal dynamics. To better align the attention mechanism for such domain phenomena, we propose attention with dynamic relational priming (prime attention). Unlike standard attention where each token presents an identical representation across all of its pair-wise interactions, prime attention tailors each
    
[^312]: AU-Harness：一个用于高效统一评估音频大语言模型的开源工具包

    AU-Harness: An Open-Source Toolkit for Efficient and Unified Evaluation of AUdio-LLMs

    [https://arxiv.org/abs/2509.08031](https://arxiv.org/abs/2509.08031)

    AU-Harness是一个开源的音频大语言模型统一评估框架，通过高效流水线将评估处理时间减少77%，并解决了现有工具在多轮对话支持不足和框架缺乏标准化方面的关键局限。

    

    大音频语言模型（LALMs）正在快速发展，但由于评估工具包效率低下且缺乏标准化，限制了公平比较和系统性评估，对其进行评估仍然具有挑战性。现有的评估框架存在三个关键局限性：（1）缓慢且低效的处理流程，成为大规模研究的瓶颈；（2）对多轮对话的支持不足，导致关于LALMs中跨轮上下文整合以及长对话中性能动态变化的基本问题尚未得到解答；（3）缺乏能够跟上LALMs和音频基准快速增长步伐的统一且可扩展的评估框架。为了解决这些问题，我们推出了AU-Harness，一个高效且全面的LALMs评估框架。我们的实证研究表明，AU-Harness相比现有评估工具包将评估处理时间减少了77%，使...

    arXiv:2509.08031v4 Announce Type: replace-cross  Abstract: Large Audio Language Models (LALMs) are rapidly advancing, but evaluating them remains challenging due to inefficient and nonstandardized toolkits that limit fair comparison and systematic assessment. Existing evaluation frameworks exhibit three critical limitations: (1) slow and inefficient processing pipeline that bottlenecks large-scale studies, (2) inadequate multi-turn dialogue support, leaving fundamental questions about cross-turn context integration and performance dynamics over extended conversations in LALMs unanswered; and (3) the absence of unified and scalable evaluation framework capable of keeping pace with the rapid growth of both LALMs and audio benchmarks. To address these issues, we introduce AU-Harness, an efficient and comprehensive evaluation framework for LALMs. Our empirical studies demonstrate that AU-Harness achieves 77% reduction in evaluation processing time over existing evaluation toolkits, enablin
    
[^313]: 从决斗老虎机到情景马尔可夫决策过程：对抗性偏好反馈下的学习

    From Dueling Bandits to Episodic MDPs: Learning with Adversarial Preference Feedback

    [https://arxiv.org/abs/2507.11706](https://arxiv.org/abs/2507.11706)

    本文将决斗老虎机扩展到偏好反馈会影响状态转移的情景MDP设定，在对抗性偏好变化下建立了极小极大遗憾界 $\tilde{\Theta}(\min\{HT,(H^3SK)^{1/3}T^{2/3}\})$，并给出了达到该界的算法。

    

    在决斗老虎机问题中，学习者在每一轮比较两个臂（选项），并且只能观察到哪一个更受偏好。本文探讨当被比较的选项对还会影响下一个状态时问题会发生什么变化，例如在推荐界面中，展示给用户的选项对会影响用户的下一个情境。我们将该问题形式化为一个基于状态的偏好MDP（马尔可夫决策过程），即一个情景式的表格型MDP，其中学习者在每个访问的状态选择一对臂，只观察到两者之间带有噪声的比较结果，并转移到可能依赖于该选项对的下一个状态。偏好可能在不同的情景之间以对抗方式变化，选项对的损失通过Borda分数定义。我们证明了在转移概率已知的情况下，该问题的极小极大遗憾为 $\tilde{\Theta}(\min\{HT,(H^3SK)^{1/3}T^{2/3}\})$，其中 $H$、$S$、$K$ 和 $T$ 分别是每个情景的步数、状态数、臂数和情景数，并且该速率可以通过基于在线学习的算法达到。

    arXiv:2507.11706v2 Announce Type: replace-cross  Abstract: In dueling bandits, the learner compares two arms in each round and observes only which one is preferred. This paper asks what changes when the compared pair also affects the next state, as in a recommendation interface where the pair shown to a user affects the user's next context. We formulate this problem as a state-wise preference-based MDP, an episodic tabular MDP in which the learner selects a pair of arms at each visited state, observes only a noisy comparison between them, and moves to a next state that may depend on the pair. The preferences may change adversarially across episodes, and the loss of a pair is defined through Borda scores. We show that the minimax regret of this problem under known transitions is $\tilde{\Theta}(\min\{HT,(H^3SK)^{1/3}T^{2/3}\})$, where $H$, $S$, $K$, and $T$ are the numbers of steps per episode, states, arms, and episodes, and that this rate is achieved by an algorithm based on online li
    
[^314]: 面向国际空间站轨迹优化的深度学习热启动方法

    Deep Learning Warm Starts for Trajectory Optimization on the International Space Station

    [https://arxiv.org/abs/2505.05588](https://arxiv.org/abs/2505.05588)

    本研究在国际空间站上首次实现在轨演示，利用神经网络热启动加速Astrobee自由飞行机器人的轨迹优化，通过学习顺序凸规划问题的结构来实现满足安全约束的星载实时轨迹生成。

    

    轨迹优化是现代机器人自主性的基石，它使系统能够在满足安全性和物理约束的前提下实时计算轨迹与控制指令。然而，由于其繁重的计算需求超出了大多数飞行计算机的能力，轨迹优化在航天应用中的使用一直非常有限。在这项工作中，我们展示了首次在太空中进行的在轨演示：利用基于机器学习的热启动方法，为国际空间站（ISS）上的Astrobee自由飞行机器人加速轨迹优化。我们提出了一种数据驱动的最优控制方法，训练神经网络来学习使用顺序凸规划（SCP）求解的轨迹生成问题的结构。在星载设备上，训练好的神经网络预测轨迹生成问题的解，并依靠SCP求解器来强制执行系统的安全约束。

    arXiv:2505.05588v4 Announce Type: replace-cross  Abstract: Trajectory optimization is a cornerstone of modern robot autonomy, enabling systems to compute trajectories and controls in real-time while respecting safety and physical constraints. However, it has seen limited usage in spaceflight applications due to its heavy computational demands that exceed the capability of most flight computers. In this work, we provide results on the first in-space demonstration of using machine learning-based warm starts for accelerating trajectory optimization for the Astrobee free-flying robot onboard the International Space Station (ISS). We formulate a data-driven optimal control approach that trains a neural network to learn the structure of the trajectory generation problem being solved using sequential convex programming (SCP). Onboard, this trained neural network predicts solutions for the trajectory generation problem and relies on using the SCP solver to enforce safety constraints for the sy
    
[^315]: 灾难性过拟合、熵差距与参与率：用于快速对抗训练的无噪声 $l^p$ 范数解决方案

    Catastrophic Overfitting, Entropy Gap and Participation Ratio: A Noiseless $l^p$ Norm Solution for Fast Adversarial Training

    [https://arxiv.org/abs/2505.02360](https://arxiv.org/abs/2505.02360)

    本文提出一种纯粹通过控制 $l^p$ 训练范数来缓解快速对抗训练中灾难性过拟合的无噪声新方案，并揭示灾难性过拟合源于高度集中的梯度与激进范数约束的相互作用。

    

    对抗训练是鲁棒深度学习的基石，但快速梯度符号法（FGSM）等快速方法常常遭受灾难性过拟合（CO）的困扰，即模型对单步攻击具有鲁棒性，但在面对多步攻击变体时会失效。虽然现有解决方案依赖于噪声注入、正则化或梯度裁剪，我们提出了一种新颖的解决方案，纯粹通过控制 $l^p$ 训练范数来缓解CO。我们的研究源于一个实证观察：CO在 $l^{\infty}$ 范数下比在 $l^2$ 范数下更为普遍。利用这一洞察，我们开发了一个将广义 $l^p$ 攻击建模为不动点问题的框架，并设计了 $l^p$-FGSM 攻击以理解从 $l^2$ 到 $l^{\infty}$ 的转换机制。由此得出我们的核心洞察：当高度集中的梯度（即信息局限于少数维度）与激进的范数约束相互作用时，CO就会出现。（注：原文摘要在此处截断）

    arXiv:2505.02360v3 Announce Type: replace-cross  Abstract: Adversarial training is a cornerstone of robust deep learning, but fast methods like the Fast Gradient Sign Method (FGSM) often suffer from Catastrophic Overfitting (CO), where models become robust to single-step attacks but fail against multi-step variants. While existing solutions rely on noise injection, regularization, or gradient clipping, we propose a novel solution that purely controls the $l^p$ training norm to mitigate CO. Our study is motivated by the empirical observation that CO is more prevalent under the $l^{\infty}$ norm than the $l^2$ norm. Leveraging this insight, we develop a framework for generalized $l^p$ attack as a fixed point problem and craft $l^p$-FGSM attacks to understand the transition mechanics from $l^2$ to $l^{\infty}$. This leads to our core insight: CO emerges when highly concentrated gradients, where information localizes in few dimensions, interact with aggressive norm constraints. By quantify
    
[^316]: 面向量子态分类的量子感知Transformer模型

    Quantum-aware Transformer model for state classification

    [https://arxiv.org/abs/2502.21055](https://arxiv.org/abs/2502.21055)

    提出了一种量子感知的Transformer模型，通过对量子态厄米矩阵表示进行掩码式无监督预训练来学习量子密度矩阵的结构特性，从而实现两体量子态（涵盖可分离态、Werner态、一般纠缠态和最大纠缠态）的数据驱动纠缠分类。

    

    纠缠是量子力学的基本特征，在量子信息处理中发挥着至关重要的作用。然而，对纠缠态进行分类，尤其是在混合态情形下，仍然是一个具有挑战性的问题，特别是随着系统维数的增加。在这项工作中，我们聚焦于两体量子态，提出了一种基于Transformer神经网络的数据驱动的纠缠分类方法。我们的数据集由多样化的两体态构成，包括纯可分离态、Werner纠缠态、一般纠缠态以及最大纠缠态。我们通过掩蔽量子态向量化厄米矩阵表示中的元素，以无监督方式对Transformer进行预训练，使模型能够学习量子密度矩阵的结构特性。这一方法使模型能够将纠缠特性泛化到不同类别的量子态上。在训练完成后……（摘要在此处截断）

    arXiv:2502.21055v2 Announce Type: replace-cross  Abstract: Entanglement is a fundamental feature of quantum mechanics, playing a crucial role in quantum information processing. However, classifying entangled states, particularly in the mixed-state regime, remains a challenging problem, especially as system dimensions increase. In this work, we focus on bipartite quantum states and present a data-driven approach to entanglement classification using transformer-based neural networks. Our dataset consists of a diverse set of bipartite states, including pure separable states, Werner entangled states, general entangled states, and maximally entangled states. We pretrain the transformer in an unsupervised fashion by masking elements of vectorized Hermitian matrix representations of quantum states, allowing the model to learn structural properties of quantum density matrices. This approach enables the model to generalize entanglement characteristics across different classes of states. Once tr
    
[^317]: 随机逆问题：稳定性、正则化与Wasserstein梯度流

    Stochastic Inverse Problem: stability, regularization and Wasserstein gradient flow

    [https://arxiv.org/abs/2410.00229](https://arxiv.org/abs/2410.00229)

    本文将逆问题推广到随机情形，在概率空间中借助测度输运理论研究了直接反演、带正则化的变分方法以及Wasserstein梯度流优化，并揭示了度量选择对稳定性和解的性质的关键影响。

    

    物理或生物科学中的逆问题通常涉及恢复一个随机的未知参数。所寻求的目标是未知参数的概率分布，该分布产生的数据与测量结果相一致。因此，这些问题自然地被构建为随机逆问题。本文从三个方面探讨这一问题：直接反演、带正则化的变分形式，以及通过梯度流进行优化，并与确定性逆问题进行类比。与确定性情形的一个关键区别在于我们所工作的空间。在这里，我们在概率空间而非欧几里得空间或Sobolev空间中进行研究，这使得测度输运理论的工具成为研究的必要手段。我们的研究结果表明，度量的选择——无论是在损失函数的设计中还是在优化过程中——对稳定性和问题的性质都有显著影响。

    arXiv:2410.00229v2 Announce Type: replace  Abstract: Inverse problems in physical or biological sciences often involve recovering an unknown parameter that is random. The sought-after quantity is a probability distribution of the unknown parameter, that produces data that aligns with measurements. Consequently, these problems are naturally framed as stochastic inverse problems. In this paper, we explore three aspects of this problem: direct inversion, variational formulation with regularization, and optimization via gradient flows, drawing parallels with deterministic inverse problems. A key difference from the deterministic case is the space in which we operate. Here, we work within a probability space rather than Euclidean or Sobolev spaces, making tools from measure transport theory necessary for the study. Our findings reveal that the choice of metric---both in the design of the loss function and in the optimization process---significantly impacts the stability and properties of th
    
[^318]: 量化近似正交循环神经网络

    Quantized Approximately Orthogonal Recurrent Neural Networks

    [https://arxiv.org/abs/2402.04012](https://arxiv.org/abs/2402.04012)

    本文提出了量化近似正交循环神经网络（QORNNs）来解决正交循环神经网络（ORNNs）中参数过多的问题，采用一种后训练量化策略和三种融入正交约束和量化权重的量化感知训练算法，取得了与s相似的结果。

    

    正交循环神经网络（ORNN）是一种吸引人的选择，用于学习涉及具有长期依赖性的时间序列的任务，由于其简单性和计算稳定性。然而，这些网络通常需要大量参数才能表现良好，这在功率受限的环境（如紧凑设备）中可能是阻碍因素。解决这个问题的一种方法是神经网络量化。构建这样的网络仍然是一个待解决的问题，其固有的不稳定性是被认可的。在本文中，我们探讨了ORNN中的循环和输入权重矩阵的量化，导致了量化近似正交RNN（QORNN）。我们研究了一种后训练量化（PTQ）策略和三种融入正交约束和量化权重的量化感知训练（QAT）算法。实证结果证明了使用QAT的优势。最高效的模型实现了与s类似的结果。

    Orthogonal recurrent neural networks (ORNNs) are an appealing option for learning tasks involving time series with long-term dependencies, thanks to their simplicity and computational stability. However, these networks often require a substantial number of parameters to perform well, which can be prohibitive in power-constrained environments, such as compact devices. One approach to address this issue is neural network quantization. The construction of such networks remains an open problem, acknowledged for its inherent instability.In this paper, we explore the quantization of the recurrent and input weight matrices in ORNNs, leading to Quantized approximately Orthogonal RNNs (QORNNs). We investigate one post-training quantization (PTQ) strategy and three quantization-aware training (QAT) algorithms that incorporate orthogonal constraints and quantized weights. Empirical results demonstrate the advantages of employing QAT over PTQ. The most efficient model achieves results similar to s
    
[^319]: FC-PINO：基于傅里叶延拓的高精度物理信息神经算子

    FC-PINO: High Precision Physics-Informed Neural Operators via Fourier Continuation

    [https://arxiv.org/abs/2211.15960](https://arxiv.org/abs/2211.15960)

    FC-PINO 通过傅里叶延拓技术克服了谱微分固有的周期性假设限制，将物理信息神经算子的高精度、高效率求解能力扩展到了非周期性偏微分方程问题。

    

    物理信息神经算子（PINO）是一种机器学习范式，在学习偏微分方程（PDE）的解方面已展现出有前景的结果。它利用傅里叶神经算子在函数空间中学习解算子，并在训练过程中利用物理损失来惩罚对已知物理定律的偏离。谱微分提供了一种为物理损失高效计算导数的方法，但它本质上假设函数具有周期性。当应用于非周期函数时，这一假设可能导致显著误差，包括区域边界附近的吉布斯现象，从而降低函数表示和导数计算的精度。为了克服这一限制，我们提出了 FC-PINO（基于傅里叶延拓的物理信息神经算子）架构，该架构将 PINO 和谱微分的精度与效率扩展到了非周期情形。

    arXiv:2211.15960v4 Announce Type: replace  Abstract: The physics-informed neural operator (PINO) is a machine learning paradigm that has demonstrated promising results for learning solutions to partial differential equations (PDEs). It leverages the Fourier Neural Operator to learn solution operators in function spaces and leverages physics losses during training to penalize deviations from known physics laws. Spectral differentiation provides an efficient way to compute derivatives for the physics losses, but it inherently assumes periodicity. When applied to non-periodic functions, this assumption can lead to significant errors, including Gibbs phenomena near domain boundaries which degrade the accuracy of both function representations and derivative computations. To overcome this limitation, we introduce the FC-PINO (Fourier-Continuation-based Physics-Informed Neural Operator) architecture which extends the accuracy and efficiency of PINO and spectral differentiation to non-periodic
    
[^320]: 基于数据与条件联合分布的最大似然方法：利用条件流模型求解不适定问题

    Maximum Likelihood on the Joint (Data, Condition) Distribution for Solving Ill-Posed Problems with Conditional Flow Models

    [https://arxiv.org/abs/2208.11782](https://arxiv.org/abs/2208.11782)

    提出了一种计算低成本的最大似然代理方法来训练条件流模型，无需显式条件分布、辅助网络或额外损失项，即可自然地求解一类不适定逆问题。

    

    在缺乏条件信息的情况下，归一化流模型使用最大似然进行训练。以往对条件化设置的扩展放弃了简单的最大似然方法，因为为了训练而获取数据与条件的完整联合分布（或等价地，获取训练数据集中每个点的真实条件分布）是不可能的。然而，在适用于一大类有用问题的特殊条件下，可以获得一个简单且计算成本低廉的代理方法，其全局最小值与条件似然相同。使用该代理方法训练的条件流模型不需要显式了解条件分布，不需要辅助网络或其他特定架构，也不需要在最大似然之外的额外损失项。它们为求解一类不适定逆问题提供了一种自然且强大的方法。我演示了这些特性……（摘要在此处截断）

    arXiv:2208.11782v2 Announce Type: replace  Abstract: In the absence of conditioning information, normalizing flow models are trained using maximum likelihood. Previous extensions to conditional settings have abandoned the simple maximum-likelihood approach, as obtaining the full joint distribution of data and conditions for training purposes (or equivalently, the true conditional distribution for each point in the training dataset) is impossible. However, under special conditions that nevertheless apply to a wide class of useful problems, it is possible to obtain a simple, computationally cheap proxy with the same global minimum as the conditional likelihood. Conditional flow models trained using this proxy do not require explicit knowledge of the conditional distribution, auxiliary networks or other specific architecture, or additional loss terms beyond maximum likelihood. They provide a natural and powerful way to solve a class of ill-posed inverse problems. I demonstrate these prope
    
[^321]: 从超维向量中学习：关于超维编码的调查

    Learning from Hypervectors: A Survey on Hypervector Encoding. (arXiv:2308.00685v1 [cs.LG])

    [http://arxiv.org/abs/2308.00685](http://arxiv.org/abs/2308.00685)

    本调查研究聚焦于超维计算系统的输入和超维向量的生成，探讨了各种方法的限制、挑战和潜在好处。

    

    超维计算是一种模拟大脑结构，提供强大而高效的处理和学习模型的新兴计算范式。在超维计算中，数据被编码为长向量，称为超维向量，通常长度为1K到10K。文献提供了几种编码技术来生成正交或相关的超维向量，这取决于预期的应用。现有文献调查通常集中在超维计算系统的整体方面，包括系统输入、主要计算和最终输出。然而，本研究采取了更具体的方法。它聚焦于超维计算系统的输入和超维向量的生成，直接影响超维编码过程。本调查汇集了不同研究中关于超维向量生成的各种方法，并探讨了它们的限制、挑战和潜在的好处。通过对本调查的全面探索，读者将获得一些关于超维编码的重要见解。

    Hyperdimensional computing (HDC) is an emerging computing paradigm that imitates the brain's structure to offer a powerful and efficient processing and learning model. In HDC, the data are encoded with long vectors, called hypervectors, typically with a length of 1K to 10K. The literature provides several encoding techniques to generate orthogonal or correlated hypervectors, depending on the intended application. The existing surveys in the literature often focus on the overall aspects of HDC systems, including system inputs, primary computations, and final outputs. However, this study takes a more specific approach. It zeroes in on the HDC system input and the generation of hypervectors, directly influencing the hypervector encoding process. This survey brings together various methods for hypervector generation from different studies and explores the limitations, challenges, and potential benefits they entail. Through a comprehensive exploration of this survey, readers will acquire a 
    

