# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Linear Regression Without a Variance](https://arxiv.org/abs/2611.27983) | 该论文提出一种基于经验残差剪枝的黑盒归约方法，在噪声仅有未知有限α矩（α∈(1,2)）且协变量仅有有限四阶矩的无穷方差情形下，可将任意有限方差回归算法自动转化为达到最优统计速率的弱矩回归算法，且无需知道矩指数、噪声尺度等任何先验信息。 |
| [^2] | [A Methodologically Rigorous Benchmark of Linear Regression,RNN, and LSTM for S&P 500 Forecasting:Identifying Sources of Performance Inflation in Financial DeepLearning](https://arxiv.org/abs/2611.27978) | 本文通过方法学审计识别出金融深度学习股票预测文献中导致性能虚高的三种来源（小波去噪评估目标、缺失朴素持续性基线、最优N次重复报告），并在修正后的严格协议下对标普500预测的线性回归、RNN和LSTM进行了统一基准测试。 |
| [^3] | [On the Complete Feature Set for ADHD Classification in Children and Adolescents Based on EEG Signals](https://arxiv.org/abs/2611.27970) | 该论文整理了一个包含156名参与者、669条记录的标准化脑电数据集，将连续信号与115种特征定义的预计算测量值相链接并记录质量决策，为儿童与青少年ADHD分类中七个脑电特征族的公平比较提供了统一基准。 |
| [^4] | [Model Carving: Training Transformers That Optimize Their Own Topology](https://arxiv.org/abs/2611.27962) | 提出模型雕刻方法，通过在每个子层上添加可学习的标量门控，让 Transformer 在单次训练中同时学习权重与架构，由训练损失自动移除冗余组件，释放出内部更小、更高效的异构网络。 |
| [^5] | [Ranking-Preserving Coreset Selection for Efficient Evolutionary Hyperparameter Optimization](https://arxiv.org/abs/2611.27959) | 本文提出保排序核心集（RPC）选择框架，通过进化优化最大化核心集与完整数据上配置排名的Spearman秩相关，从而在保留超参数配置相对排序的前提下加速GA、CHC和PSO三种进化算法的超参数优化。 |
| [^6] | [Topology Hub Optimal Transport: Eliminating Long Context Dilution at Flash Speed](https://arxiv.org/abs/2611.27951) | THOT 提出一种基于轮辐式拓扑的最优传输注意力机制，通过有界支撑容量门控、退火行封闭 Sinkhorn 调度和高效 Triton 内核伴随计算，以接近 FlashAttention 的速度消除长上下文中的注意力质量稀释问题。 |
| [^7] | [The Optimal Sample Complexity of Agnostic PAC Learning for Partial Concept Classes](https://arxiv.org/abs/2611.27946) | 该论文为VC维为d的部分概念类构造了一个确定性不可知PAC学习器，仅用 $O((d+\log(1/\delta))/\varepsilon^2)$ 个样本即可实现学习，样本复杂度与Alon等人（FOCS 2021）的下界匹配，达到最优。 |
| [^8] | [SUAF: Spectral Universal Advantage Features for Zero-Shot Policy Improvement](https://arxiv.org/abs/2611.27919) | 本文提出谱通用优势特征（SUAF），通过对以动作为中心的典型依赖矩阵进行奇异值分解，构建可通过线性奖励投影恢复延迟优势的成对特征，从而在奖励无关表征下实现零样本策略改进。 |
| [^9] | [Communication-Efficient Federated Learning of Latent Patient Representations from Multi-Institutional EHRs](https://arxiv.org/abs/2611.27900) | 该论文提出一种通信高效的非迭代联邦谱方法，各机构只需共享边际频率和低秩谱摘要即可从多机构电子健康记录中协同学习共享的潜在患者表征，同时将患者级数据保留在本地以保护隐私。 |
| [^10] | [The Dual-Tier Sentiment Framework: Scalable Multimodal UX Attribution in Web Search](https://arxiv.org/abs/2611.27899) | 该论文提出了一种双层情感框架，第一层利用微调的编码器-解码器大语言模型实现大规模用户情感回归检测（PR-AUC提升9.8%、测量延迟从数周降至数小时、问卷开销降低40%以上），第二层提供细粒度的UX归因诊断，从而实现网络搜索中可扩展的自动化用户体验评估。 |
| [^11] | [FairITR: Fairness-Aware Learning of Multicategory Individualized Treatment Rules](https://arxiv.org/abs/2611.27894) | 该论文提出了FairITR，首次针对多类别治疗场景引入公平性概念——要求治疗决策独立于敏感属性，并构建了相应的公平个体化治疗规则估计问题及其求解方法。 |
| [^12] | [Renormalized Entropy from Symmetries: Simplicity Bias and Neural Heat Death](https://arxiv.org/abs/2611.27889) | 该论文通过对朗之万动力学平衡分布的重整化分析，提出了一种由参数对称性诱导的新型熵，解释了神经网络隐式偏好简单函数（如低秩、奇异函数）的机制。 |
| [^13] | [From Gist to Verbatim: The Controllable Path of One Image Through a Diffusion Prior](https://arxiv.org/abs/2611.27875) | 该研究将生成模型的记忆化视为一个可控问题，发现记忆的形成会经历一个抽象阶段——既保留目标图像的语义要义又保持外观多样性，且此阶段比逐字复现更有利于编辑、单样本识别和检索，为理解和利用模型记忆开辟了新路径。 |
| [^14] | [Beyond Visual Fidelity: Comparing Categorical and Continuous World Models](https://arxiv.org/abs/2611.27870) | 该论文通过受控的2D和3D环境实验系统比较了类别型与连续型预测对世界模型动态学习的影响，发现在无损视觉编码条件下，类别型模型能以更少训练计算量获得更高的动态rollout有效性和更好的随机性校准，表明视觉逼真度并不等同于动态保真度。 |
| [^15] | [Anchors That Move Without Collapsing: Exclusive-Region Anchor Routing for Vision Transformers](https://arxiv.org/abs/2611.27855) | 该论文提出面向视觉Transformer的独占区域锚点路由机制，让每个锚点持有互不重叠的Voronoi区域从构造上杜绝锚点摘要坍缩，并结合焦点锚点将锚点置于习得重要性图的峰值处，从而显著提升分类精度。 |
| [^16] | [DistDNA: Language Model Fingerprinting under Stochastic Decoding](https://arxiv.org/abs/2611.27831) | 提出 DistDNA 方法，通过从少量采样响应中估计响应分布，并将解码设置变化与检查点变化的影响区分开来，实现随机解码条件下的大语言模型可靠指纹识别。 |
| [^17] | [Graph Signal Generation via Graph-Aware Diffusion Models](https://arxiv.org/abs/2611.27826) | 提出图感知扩散模型（GAD），将前向过程建模为图上的热扩散，并设计模式自适应信噪比调度（MASS），为每个图频率分配独立扩散速率以保持相同信噪比，从而实现更好的图信号生成与分布学习。 |
| [^18] | [Mixture of Latent Recursions for Low-Rank Adaptation](https://arxiv.org/abs/2611.27823) | MoLR是一种自适应递归框架，通过共享变换递归细化潜在表示，并利用轻量级阈值路由器动态决定每个令牌的递归深度，在几乎不增加参数开销的情况下显著提升了LoRA的适应表达能力。 |
| [^19] | [Sampling non-abelian lattice gauge theories with diffusion models](https://arxiv.org/abs/2611.27821) | 本文提出将规范等变扩散模型与格点规范等变卷积神经网络和Metropolis调整的退火朗之万算法相结合，作为格点规范理论的采样器，实现了对二维U(2)场构型的准确采样、四维SU(3)的初步成功结果，以及对更大逆耦合常数的优异外推能力。 |
| [^20] | [Autonomous quantum machine learning via Riemannian gradient-based optimization](https://arxiv.org/abs/2611.27815) | 提出了一种基于黎曼梯度优化的完全自主自训练量子机器学习框架，无需量子测量或经典反馈即可在量子域内完成优化，并给出了指数收敛保证。 |
| [^21] | [Redesigning Neural Networks to Not be Confidently Wrong even to Strong Adversarial Attacks](https://arxiv.org/abs/2611.27811) | 该论文提出了一种模拟生物视觉皮层简单细胞与复杂细胞交替结构的浅层神经网络架构，通过能够学习选择性与容忍性表征的新型组件，实现了准确、鲁棒且与人类感知更一致的图像分类，并能有效抵御强对抗攻击，避免对未知类别样本做出高置信度的错误预测。 |
| [^22] | [SLP-Bench: Evaluating and Improving Acoustic Claims of Reasoning Audio Language Models on Clinical Speech](https://arxiv.org/abs/2611.27810) | 提出SLP-Bench基准，将推理型音频语言模型思维链中的每一项声学论断与临床医生的感知评分逐条对照，评估其事实性、稳定性和一致性，结果显示十个开源模型的论断平衡准确率仅为0.52至0.58。 |
| [^23] | [DOiT: Better Money-Flow Ranking Can Coexist with Worse Account Selection in Anti-Money Laundering Review](https://arxiv.org/abs/2611.27809) | DOiT 揭示了反洗钱审查中一个关键现象：即使 GNN 重排序提升了交易资金流的排序精度（更高的 AP），也可能损害上游账户选择的质量，因此应在同等审查容量下将资金流排序与账户选择这两种决策分离处理。 |
| [^24] | [$\mathtt{DistP}$: Scaling Local-Update Training with Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2611.27808) | 提出理论框架$\mathtt{DistP}$，将$\mathtt{DDP}$的零样本超参数迁移规则扩展到本地更新训练，使学习率能够跨模型宽度、深度和同步间隔直接迁移，从而免去昂贵的超参数调优。 |
| [^25] | [$\varepsilon$-SegFormer: Multi-Resolution Vision Transformers for Sparsely Supervised Segmentation in Electron Microscopy](https://arxiv.org/abs/2611.27796) | ε-SegFormer提出了一种多分辨率视觉Transformer，通过对每个标注像素在三个物理分辨率下采样视图并用注意力机制在共享坐标框架中关联所有视图，实现电子显微镜数据的稀疏监督语义分割，大幅降低密集标注成本。 |
| [^26] | [DPMA: Learning to Compress Mixture-of-Experts Models](https://arxiv.org/abs/2611.27759) | DPMA 提出了一种无需重新训练的可微分压缩方法，在全局专家预算下联合学习专家剪枝、参数合并、路由吸收三种操作及逐层的专家分配策略，从而在多个 MoE 模型和压缩率下取得了最优的压缩效果。 |
| [^27] | [Budgeted Subset Selection with a Transferable Learned Value Model](https://arxiv.org/abs/2611.27751) | 提出SALVES框架，将预算约束下的子集选择分解为可跨任务迁移的学习价值模型与基于模拟退火的子集搜索两个组件，摆脱了现有方法对特定应用场景的依赖。 |
| [^28] | [A Switching Mechanism for Selecting Environment-Adaptive Personalized Gains in Hearing Aids](https://arxiv.org/abs/2611.27726) | 本文提出了一种助听器切换机制，通过单类语音活动检测器排除前景语音干扰，并结合贝叶斯变点检测与历史环境记忆，自动识别当前背景音频环境并选择个性化增益。 |
| [^29] | [Beyond Math, Code, and Logic: Do Reasoning Models Transfer to Scientific Argumentation?](https://arxiv.org/abs/2611.27722) | 该研究首次系统评估了语言模型在科学论证任务上的表现，发现即使前沿推理模型在领域特定论证任务上仍存在根本性能力差距，并提出了一种显式搭建论证推理脚手架的训练框架，显著提升了模型的科学论述能力。 |
| [^30] | [Decision Models for Multilingual Emotion Classification: A Comparative Study Against LLMs and Encoders](https://arxiv.org/abs/2611.27721) | 该研究首次在五种语言的多语言情感分类任务上系统比较了决策模型、大语言模型与微调编码器，发现最佳零样本系统的性能仅比微调编码器低约1个宏F1点。 |
| [^31] | [Solving the pre-image problem in kernel PCA: The reproducing property is all you need](https://arxiv.org/abs/2611.27718) | 本文提出仅利用再生性质即可直接求解核PCA的预像问题，无需非线性优化或额外的监督学习，同时在线性核情形下能自然退化为标准PCA的重构映射。 |
| [^32] | [Geometrically Constrained Learning II: Agnostic Proper Learning of Lipschitz Functions](https://arxiv.org/abs/2611.27700) | 本文为高斯分布下的 $\lambda$-Lipschitz 函数给出了一个正常学习器，其样本复杂度、运行时间和表示规模均为 $(d+1)^{\mathsf{poly}(\lambda/\varepsilon)}$，并在已知标签方差上界时可扩展为误差为 $\mathsf{opt}+\varepsilon$ 的不可知学习算法。 |
| [^33] | [Spatiotemporal Graph Neural Networks for Fault Detection and Structural Learning in Chemical Processes: Use Case on the Tennessee Eastman Process](https://arxiv.org/abs/2611.27698) | 该论文提出一种时空图神经网络框架，通过将正常运行空间分解为特定于运行条件的子空间，并结合GATv2图结构学习与时间编码，解决了化工过程中因运行条件变化导致的误报与漏检问题，并在田纳西-伊士曼过程上验证了其故障检测能力。 |
| [^34] | [Learning in Symbiosis: Heterogeneous Federated Learning with ANNs and SNNs](https://arxiv.org/abs/2611.27686) | 本文提出混合神经形态联邦学习框架，使异构的人工神经网络与脉冲神经网络设备能够在联邦学习环境中协同联合训练，从而推动向节能神经形态边缘AI的渐进式过渡。 |
| [^35] | [CanopyFormer: Attention-Guided SegFormer for Explainable Forest Cover Segmentation](https://arxiv.org/abs/2611.27674) | CanopyFormer通过注意力一致性损失将SegFormer的注意力与真实森林掩码对齐，并引入AAMO指标和删除测试对森林覆盖分割中的注意力可解释性进行量化评估，使注意力对齐度提升约22倍。 |
| [^36] | [Can pretrained Transformers deliver semiparametrically efficient inference?](https://arxiv.org/abs/2611.27666) | 该论文证明单个预训练Transformer无需推导有效影响函数或估计干扰函数，即可通过一次前向传播对一族半参数待估量同时实现有效估计与具有渐近精确覆盖率的有效推断。 |
| [^37] | [SMSD: A Semantic Misinformation Spreader Detection Approach](https://arxiv.org/abs/2611.27661) | 提出SMSD方法，将虚假信息传播建模为时间性、以用户为中心、行为驱动的学习问题，通过融合用户级、网络级和对话传播特征的多模态架构与立场感知标注策略，实现对社交媒体虚假信息传播者的有效早期检测。 |
| [^38] | [Beyond Sample Hardness: Adaptive Data Selection through Learning Dynamics](https://arxiv.org/abs/2611.27656) | 该论文提出LeDAS框架，通过难度、学习进展和退化三个学习动态信号自适应地选择训练数据，并利用基于上下文老虎机的自适应度量融合机制动态调整样本优先级，以极小的额外计算开销克服了传统仅依赖样本难度的数据选择方法的局限性。 |
| [^39] | [Late-Time Dynamics of Flow Matching: Distance Geometry and Learning](https://arxiv.org/abs/2611.27653) | 本文研究了流匹配的晚期动力学，证明在有限训练集上轨迹会以指数速度收缩至最近的训练样本，到训练集的平方距离生成显式收缩路径，并量化了由Voronoi界面附近初始质量所控制的精确分布与参考分布之间的Wasserstein误差。 |
| [^40] | [Efficient VLA Reinforcement Learning via Subchunk Rollouts in Latent World Models](https://arxiv.org/abs/2611.27645) | 该论文提出LS-MBPO轻量级残差MBRL框架，通过层级强制缓解复合预测误差、并通过潜在子块推演结合冻结VLA策略与潜在残差策略，从而以低计算开销的潜在世界模型高效改进预训练VLA策略。 |
| [^41] | [Beyond Resolution: Inference-Time Parameterized Temporal Consistency in Generative Regional Climate Downscaling](https://arxiv.org/abs/2611.27641) | 该论文提出一种噪声条件化教师强制训练方案，让扩散降尺度模型通过以带噪的前一时刻高分辨率场为条件，将时间一致性与集合多样性的权衡变为推理时可调的参数，从而在避免误差累积的情况下生成时间连贯的区域气候降尺度序列。 |
| [^42] | [Potential emergence in LLM agents: a statistical physics perspective](https://arxiv.org/abs/2611.27618) | 该论文从统计物理学视角提出将智能体状态作为中间变量，通过配分函数与全局势能来定量测量并预测大语言模型智能体在后训练过程中行为的组织与涌现。 |
| [^43] | [Circular Predictive Coding Unlocks Effective Depth](https://arxiv.org/abs/2611.27610) | 该论文提出循环预测编码网络，其有效深度直接源自能量最小化过程而非堆叠层，并证明其平衡点收敛近似于深度均衡模型的不动点迭代，从而在不加深网络的情况下解锁了有效深度，在图像分类任务上优于同规模的其他预测编码模型。 |
| [^44] | [Quantum-Informed Regularized Poisson Learning for Graph-Based Semi-Supervised Classification](https://arxiv.org/abs/2611.27609) | 提出了量子启发的正则化泊松学习框架IPQSSL，利用共享变分量子线路产生节点对角权重以稳定对称化泊松系统，在标签稀缺的图半监督分类任务中取得了优于泊松基线的效果。 |
| [^45] | [Tasklets: A Unit-Test Benchmark for Language Model Architectures](https://arxiv.org/abs/2611.27607) | Tasklets 是一个由小型合成任务组成的标准化基准，按能力类别（回忆、序列操作、状态跟踪）组织，用于低成本、可重复地评估语言模型序列混合层的基础计算能力。 |
| [^46] | [Adapting AI Weather Foundation Models to a Changing Climate](https://arxiv.org/abs/2611.27605) | 提出了一种在线适配框架，通过在低维敏感性子空间内进行流依赖校正并结合残差记忆追踪地理持续性偏差，使AI天气基础模型无需全模型重训练即可持续适应演变的气候分布。 |
| [^47] | [Block Search Conditioned on Features for Sequential Quantum Data Encoding](https://arxiv.org/abs/2611.27587) | 该论文提出了一种特征条件化的块搜索方法，在固定量子寄存器中逐步联合选择输入特征、操作块、放置位置与旋转权重以扩展Q-FLAIR，在Bars and Stripes数据集上显著提升了分类性能指标。 |
| [^48] | [Do DeepONet Trunk-Basis Preconditioners Need Training?](https://arxiv.org/abs/2611.27582) | 该研究表明DeepONet主干基预条件子无需训练——未训练的随机初始化分支函数张成的空间近似低次多项式空间，其预条件效果与训练后的版本相当。 |
| [^49] | [PRISM: Validation-Free Online Batch Selection for LLM Fine-Tuning via Preconditioned Conic Ghost Matching](https://arxiv.org/abs/2611.27578) | PRISM提出了一种无需验证集的大语言模型微调在线批选择方法，通过预条件化的锥形幽灵匹配将子集选择建模为归一化的单调子模锥形设施选址问题，避免逐样本梯度显式计算的同时保证 (1-1/e) 的贪心近似效果。 |
| [^50] | [Minimax-Optimal Small-Loss Regret in Online Convex Optimization and Applications to Contextual Search and Online Inverse Optimization](https://arxiv.org/abs/2611.27572) | 该论文提出了一个在线凸优化算法，在每轮仅需一个次梯度、不知道损失值和时间范围、无平滑性假设的条件下，实现了极小极大最优的 $O(1+\sqrt{\Delta_T^\star})$ 小损失遗憾界，并成功应用于上下文搜索和在线逆优化问题。 |
| [^51] | [WEIR: Watershed Exploration and Drift-Aware Prefix Shrinkage for Multi-Turn Agent Reinforcement Learning](https://arxiv.org/abs/2611.27570) | 提出基于Doob轮次方差守恒的WEIR框架，通过分水岭探索与漂移感知前缀收缩机制，解决多轮智能体强化学习中稀疏奖励下难以区分关键决策及样本浪费的问题。 |
| [^52] | [TITAN: DRL-Based Dynamic Scheduling and Resource Allocation for Tactile Internet](https://arxiv.org/abs/2611.27560) | TITAN是一种基于深度强化学习的双时间尺度动态调度与资源分配方案，可根据时变的触觉QoS需求自适应地在触觉流与视频流之间划分带宽并分配资源块，同时满足两类业务的QoS目标并带来7.64个百分点的性能提升。 |
| [^53] | [On the Limits of Observability for Evaluating Coverage in Machine Unlearning](https://arxiv.org/abs/2611.27543) | 该研究通过502个重训练模型的可观测性压力测试和Ω(log n)秩下界的理论证明，揭示了现有机器遗忘评估因只关注特定遗忘集而忽视“覆盖率”的问题，并提出了评估遗忘覆盖率的CEO框架。 |
| [^54] | [PaMuk-Bench: A Parallel Multilingual Multitask Benchmark for Vision-Language Tasks](https://arxiv.org/abs/2611.27521) | 本文提出了PaMuk-Bench，一个包含超过10K条平行翻译提示（涵盖阿拉伯语、丹麦语、法语、印地语、意大利语和罗马尼亚语六种语言）的多语言多任务视觉-语言基准，用于系统评估视觉-语言模型在低资源语言上跨多种任务的表现。 |
| [^55] | [Deep Cross-Division Network for Product-and-Ratio Feature Interactions in Industrial Ranking](https://arxiv.org/abs/2611.27514) | 本文提出深度交叉-相除网络DCDN，通过在DCNv2交叉层中引入相除分支，使模型能够联合学习乘积、比值及混合型特征交互，并将其表达能力刻画为ℓ1阶最高达K+1、带符号次数介于-K与K+1之间的多元Laurent多项式。 |
| [^56] | [Breaking the Barrier of Hard Samples: A Data-Centric Approach to Synthetic Data for Medical Tasks](https://arxiv.org/abs/2611.27511) | 本文提出以数据为中心的Profile2Gen框架，通过生成与改进合成数据来解决医学回归任务中难以学习的样本问题，并经过约18,000项实验验证，证明改进后的合成样本可降低预测误差并提升模型可靠性。 |
| [^57] | [Self-Supervised Continuous-Time Depth Learning and Ego-Motion Estimation from Asynchronous Events](https://arxiv.org/abs/2611.27509) | 该论文提出了一种自监督的连续时间深度学习方法，为每个异步事件分配独立深度并从同一观测中恢复相机自运动，利用事件间的几何一致性作为监督信号，并通过可微渲染生成目标时刻的稠密深度图，在MVSEC数据集上超越了先前仅基于事件的自监督方法。 |
| [^58] | [Turnout: Adaptive Attention Switching for Efficient Reasoning](https://arxiv.org/abs/2611.27500) | 提出Turnout方法，在预训练全注意力模型中通过蒸馏保留线性注意力路径，并利用强化学习训练的自适应策略在每一层动态切换全注意力与线性注意力，从而在大幅降低内存消耗的同时保持长文本推理任务的准确率。 |
| [^59] | [Constraint-Aware Anticlustering Decomposition for Certifying Pairwise-Constrained Clustering](https://arxiv.org/abs/2611.27496) | 该论文提出了一种约束感知的反聚类分解框架，通过引入约束反聚类问题（CAP）并构建为混合整数二阶锥规划模型，将约束图的连通分量分配到反聚类中，从而为带成对约束的大规模最小平方和聚类问题提供有效的认证下界。 |
| [^60] | [Learning Unknown Oscillators From Bandit Feedback](https://arxiv.org/abs/2611.27492) | 提出了一种带检验的承诺控制器，在未知振荡器驱动的半老虎机问题上实现了平方根级后悔界，并将拟合次数从1023次降至49次，带来4.69倍的加速。 |
| [^61] | [Verifying and Strengthening Neural Tangent Kernel Theory](https://arxiv.org/abs/2611.27487) | 该论文首次用 Lean 4 对两层网络的神经正切核理论进行了端到端的机器验证形式化证明，证明了梯度流的存在唯一性与残差指数收敛定理，并在此过程中强化了标准理论。 |
| [^62] | [Practical insights into controlling exponential concentration in quantum machine learning](https://arxiv.org/abs/2611.27486) | 该论文在基于期望值方差的统一框架下实证比较了三类量子机器学习架构中的指数集中现象，发现全局保真度核因希尔伯特空间维度而强烈集中，而测量局域性、纠缠调控和储备池深度等设计手段可以缓解但无法完全消除这一效应。 |
| [^63] | [Cannot Wait to Verify: Parallel Drafting to Eliminate Verifier Idle Time in Speculative Decoding](https://arxiv.org/abs/2611.27477) | 提出一种即插即用的并行草稿生成方法，通过跨Transformer层并行推理并结合隐藏状态散度一致性验证与纠正机制，消除了推测解码中验证器的空闲等待时间，在无需修改架构或重新训练的情况下加速草稿生成并提升硬件利用率。 |
| [^64] | [Learning When to Ask: Advisor-Aware Post-Training for SLM Agents](https://arxiv.org/abs/2611.27443) | 提出ADAPT多阶段后训练方法，让小型语言模型智能体学会在成本感知下判断何时向更强模型求助并跨步骤遵循建议，且完全依靠环境结果监督的策略自身rollouts进行学习，无需奖励模型或人工标注。 |
| [^65] | [THRIFTY: Cost-Constrained Reasoning-Data Synthesis for Small Language Models](https://arxiv.org/abs/2611.27436) | 提出THRIFTY——一种基于Thompson采样的带背包约束多臂老虎机控制器，在固定的生成token预算下自适应地将预算分配给不同难度的问题，从而低成本地为小语言模型合成多样且正确的推理训练数据。 |
| [^66] | [Learned Network Dynamics for Model Predictive Traffic Signal Control under Incidents](https://arxiv.org/abs/2611.27435) | 提出GNN-MPC控制器，利用图神经网络学习动作条件化的网络动态并结合近似最优响应协调信号相位，在事故干扰下有效降低旅行时间并增加完成的出行次数。 |
| [^67] | [ORBIT: Offset-based Activation Steering for Cross-lingual Multimodal Fact Verification](https://arxiv.org/abs/2611.27421) | 提出了推理时算法干预方法ORBIT，通过从平行声明-证据实例中估计偏移向量并修正声明与证据表示，在不需微调的情况下显著恢复小型视觉-语言模型在低资源语言上的多模态事实核查性能。 |
| [^68] | [Neural variational Skyrmions as a topological benchmark: emergent symmetry, precision energies and KAN--MLP comparison](https://arxiv.org/abs/2611.27411) | 该论文将三维Skyrme模型确立为神经变分方法的拓扑基准：在无对称性拟设条件下高精度构建了 $B=1$ 至 $4$ 的Skyrmion，已知对称结构自发涌现，能量略低于文献参考值，拓扑荷保持精度达 $10^{-6}$，并通过Derrick位力关系与Faddeev–Bogomolny界等物理约束检验，同时对比了KAN与MLP网络的表现。 |
| [^69] | [Degradation of Language Models under Controlled Load: Synchronization, Rule Execution and Misfires on the Background](https://arxiv.org/abs/2611.27404) | 该论文提出将受控负载下语言模型的性能退化分解为同步失败、规则执行错误与无关误改三种成分，通过逐词对齐和考虑难度差异随负载变化的统计模型，独立量化文本长度、替换密度、规则数量等每种负载因素的影响。 |
| [^70] | [HR-Merge: Model Merging with Headroom-Weighted Residual Refinement](https://arxiv.org/abs/2611.27377) | HR-Merge 是一种无需训练的通用模型合并增强框架，它根据各任务的验证准确率余量对其残差加权并重新合并，从而有效恢复现有合并方法在不同任务上不均匀损失的准确率。 |
| [^71] | [Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training](https://arxiv.org/abs/2611.27376) | 提出可组合软提示蒸馏（CSPD）方法，将每个RL专家独立蒸馏为软提示并通过拼接方式组合，无需重新整合模型即可实现持续后训练，性能媲美多教师蒸馏且不损害目标领域之外的能力。 |
| [^72] | [Gumbel Substitution: Removing Sampler Drift from RL Without Trusting the Sampler](https://arxiv.org/abs/2611.27375) | 该论文提出Gumbel替换方法，由协调器分发比特级一致的公开Gumbel噪声，使训练器能重建其自身策略在相同噪声下的精确token样本，从而在不信任采样器、不依赖其报告概率的情况下，消除强化学习训练中因采样器与训练器策略不一致而产生的漂移。 |
| [^73] | [AMARIS: Merging Generalist and Specialist LLMs via Adaptive Subspace Inheritance](https://arxiv.org/abs/2611.27369) | AMARIS 提出了一种协方差驱动的免训练模型合并框架，通过在正则化扰动预算下选择高效用、低干扰的激活子空间，将专家模型能力以单一静态检查点的形式无训练地继承到通用大语言模型中。 |
| [^74] | [Latent Jacobian Steer: Eliciting Contact-Rich Out-of-Distribution Behaviors from Pretrained Humanoid Controllers](https://arxiv.org/abs/2611.27338) | 提出潜在雅可比引导（LJS）方法，通过将关节动作空间采样与策略雅可比的阻尼局部逆映射相结合，从预训练人形控制器中引发复杂且富含接触的分布外行为，兼顾了协调性与任务针对性。 |
| [^75] | [Spiking Neural Networks Incrementally Track Firing-Rate Networks](https://arxiv.org/abs/2611.27322) | 提出了基于SNN增量跟踪发放率网络理论的LIF-Stream转换框架，使转换后的脉冲神经网络能够复用先前激活状态并仅计算帧间激活更新，从而在无界数据流上实现高效且高精度的SNN推理。 |
| [^76] | [Preserving Pretrained CFG Structure in RL for Diffusion Models with Conditional Contrast Regularization](https://arxiv.org/abs/2611.27288) | 提出条件对比正则化方法，在扩散模型强化学习后训练中保留预训练CFG的几何结构，作为诊断奖励黑客行为的有效指标，从而实现奖励提升与整体生成质量之间更稳定的权衡。 |
| [^77] | [Error-Corrected Physics-Informed Neural Operator for Optimal Control of Unmanned Aerial Vehicles under Uncertainty](https://arxiv.org/abs/2611.27272) | 该论文提出一种误差校正的物理信息神经算子，通过一次牛顿步校正预测误差来求解不确定性条件下的无人机开环最优控制问题，其精度可匹敌传统求解器且速度快8至15倍。 |
| [^78] | [Semiparametric Efficiency for Entropic Optimal Transport under Structured Distribution Shift](https://arxiv.org/abs/2611.27261) | 本文通过精确投影恒等式揭示了结构约束下熵最优传输估计的方差增益何时为正、何时为零，并构造了达到该效率界的混合经验似然估计量。 |
| [^79] | [A preference for local interaction between layers emerges during training](https://arxiv.org/abs/2611.27258) | 该研究发现深度神经网络在训练过程中会自发形成“局部性”——相邻层之间对彼此变化的响应随训练不断增强，逐渐超过距离更远的层，尽管训练本身通过共享误差将所有层耦合在一起。 |
| [^80] | [Proactive Defense Benchmark against Deepfake Generation](https://arxiv.org/abs/2611.27253) | 该论文提出了首个针对深度伪造主动防御的综合评估基准，揭示了保真度与身份指标的矛盾解读以及白盒峰值性能背后的过拟合问题，并建立严格标准以推动真正可泛化防护方法的发展。 |
| [^81] | [More Bits Won't Fix It: Diagnosing and Treating an Ill-Conditioned Backward Pass (A Case Study on Gemma-4)](https://arxiv.org/abs/2611.27243) | Gemma-4的LoRA微调中bf16与fp32梯度在模型低层几乎正交，其根源是第19-21层MLP在图像token位置存在病态反向传播路径，且梯度可能被单个低信息图像块主导，在反向传播中断开该路径即可解决，而单纯提高数值精度并无帮助。 |
| [^82] | [CreditFilter: Calibrated Token Weights over the Within-Response Entropy Rank for RLVR](https://arxiv.org/abs/2611.27227) | 该论文将 token 对结果的影响形式化为“分叉值”，并利用分叉概率随响应内熵排名呈指数衰减的实证发现，提出 CreditFilter——通过熵排名校准 token 权重、在响应内部重新分配信用，从而改进免评论家的可验证奖励强化学习（RLVR）。 |
| [^83] | [A Stationarity-Based Regret Framework for Time-Varying Nonconvex Composite Optimization](https://arxiv.org/abs/2611.27218) | 本文为时变非凸复合优化提出了基于一阶平稳性（如近端梯度残差）的局部后悔度量框架，证明该度量在凸情形下等价于经典动态后悔，并指出在非凸情形下若缺乏时间正则性则次线性平稳性后悔无法实现。 |
| [^84] | [LingSante-Pulse: Unified Physiological Pretraining through Random-Feature Prediction and Cross-Modal Alignment](https://arxiv.org/abs/2611.27206) | 该论文提出统一生理预训练框架LingSante-Pulse，通过随机特征预测、跨模态对齐与掩码波形重建学习通用生理信号表示，在涵盖PPG、ECG、EEG和EMG的38个任务配置的每个领域均取得最佳排名。 |
| [^85] | [Improved Accuracy Scaling with Problem Complexity in LLM Arithmetic Reasoning Using Executable Symbolic Representations](https://arxiv.org/abs/2611.27203) | 该论文提出GSM算术基准（GSM-AB），通过独立控制操作数位长和运算次数，证明使用可执行符号表示（如Lean）能改善大语言模型算术推理的准确率扩展性，减缓准确率随问题复杂度增加而退化的速度。 |
| [^86] | [Backbone-Specific LoRA Geometry: Orthogonality, Superposition, and Provenance](https://arxiv.org/abs/2611.27201) | 该论文将LoRA更新方向视为主干网络独特的“指纹”，通过推导跨主干网络相关性上界和局部梯度流模型揭示了正交性与叠加规律，从而实现基于余弦相似度的主干网络识别以及无需坐标对齐的已注册适配器复用检验。 |
| [^87] | [cmHC: Balancing Stream Mixing in Hyper-Connections via Geometric Contraction](https://arxiv.org/abs/2611.27193) | 本文揭示了Birkhoff多胞形中残差流混合的权衡问题——混合不足会退化为独立流、过度混合会使流不可区分——并提出cmHC方法，利用几何收缩在多胞形内平衡流混合程度，同时从理论上给出了跨层累积混合、流差异与梯度差异的界。 |
| [^88] | [Lift Out, Kick In: Decoupling Event Effects for Time Series Forecasting](https://arxiv.org/abs/2611.27191) | 该论文提出基于共享事件响应函数（ERF）的 Loki 框架，通过从历史观测中剔除事件诱发偏差、再向预测中注入未来事件效应，实现事件效应与系统内在时间动态的解耦，从而提升事件感知的时间序列预测性能。 |
| [^89] | [A Frugal AI Framework for India: Responsible and Right-sized AI](https://arxiv.org/abs/2611.27189) | 本文提出“负责任节俭AI”框架，主张印度既不应照搬中美重资本的基础模型路线，也不应仅作外国AI的下游采用者，而应先明确功能需求并判断是否需要AI，再选择以最少资源满足需求的最小充分“规模适度”模型，同时保持鲁棒性与负责任治理。 |
| [^90] | [FedQCal: Quadratic Programming with Dual-Space Calibration for One-Shot Federated Learning](https://arxiv.org/abs/2611.27170) | FedQCal将单轮联邦学习的模型聚合形式化为最小范数二次规划，利用服务器参考模型定义对齐要求，并通过分类损失的两个对偶变量自动校准对齐阈值，无需预先设定阈值即可有效聚合异构客户端模型。 |
| [^91] | [BusemannFormer: Horospherical Self-Attention for Hyperbolic Graph Transformers](https://arxiv.org/abs/2611.27167) | 该论文提出了基于理想边界方向上Busemann深度的极限球自注意力（HSA），并将其构建为完全双曲的图Transformer BusemannFormer，在层次化图任务上显著优于现有方法。 |
| [^92] | [Hierarchical Hybrid-Coordinate Flow Matching for Accurate and Physically Plausible Apo-to-Holo Protein Generation](https://arxiv.org/abs/2611.27162) | CrypticFlow提出了一种分层混合坐标流匹配方法，先在笛卡尔空间生成稀疏Cα锚点，再以内坐标递归填补中点并施加距离约束，从而同时实现高全局精度与物理合理的局部几何，解决apo-to-holo蛋白构象预测问题。 |
| [^93] | [FM4SAR-Bench 1.0: Towards A Fair Benchmark for Evaluating Foundation Models for SAR Image Interpretation](https://arxiv.org/abs/2611.27158) | 提出了FM4SAR-Bench 1.0，一个包含系统性分类体系及五个评级级别、六个维度分层评估框架的基准，用于对SAR图像解译基础模型进行公平、全面的评估。 |
| [^94] | [The Vulnerability Window: Decision Instability in Quantized Reward Models](https://arxiv.org/abs/2611.27155) | 量化在总体基准准确率几乎不变的情况下仍会翻转1.1-12.3%的奖励模型成对偏好决策，其翻转率可用高斯近似Φ(-|m|/σ)刻画，且脆弱性窗口占比随模型规模从25%降至7%，揭示了小型裁判尤其脆弱的隐患。 |
| [^95] | [Git History Is a Pangenome: Repetition in Code Training Corpora and What Deduplication Removes, Keeps and Misses](https://arxiv.org/abs/2611.27154) | 该论文揭示Git仓库历史本质上是一个高度重复的“泛基因组”——典型仓库历史相当于其当前代码的19份副本，而专为快照设计的精确匹配去重方法完全无法移除这些历史重复内容。 |
| [^96] | [Structure Over Scale: Rethinking Adaptation for Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2611.27148) | 该研究发现在可验证奖励强化学习训练中，LoRA会随秩增大出现性能骤降的“秩坍缩”现象，而DoRA和QuanTA等结构化适配方法凭借更优的更新几何结构，仅更新不到0.6%的参数即可达到与全参数微调相当的效果，表明更新的结构比规模更关键。 |
| [^97] | [GaLF-ST: Gaussian Latent Fields for Spatial Transcriptomics Reconstruction](https://arxiv.org/abs/2611.27143) | 该论文提出GaLF-ST，一种利用可学习的高斯中心和各向异性协方差构建高斯潜在场、对每个位置的潜在向量进行加权以实现空间转录组学基因表达重建的方法。 |
| [^98] | [Dynamic Memory-enhanced Pattern Relevance Prediction for Long-tailed Graph Classification](https://arxiv.org/abs/2611.27141) | 提出DMPRP方法，通过动态专家记忆模块将头部图的结构模式迁移到尾部图，从而有效提升长尾图分类任务中尾部类的性能。 |
| [^99] | [Spatiotemporal Landslide Runout Reconstruction from Sparse Observations with LAPIS-SHRED](https://arxiv.org/abs/2611.27132) | 提出LAPIS-SHRED框架，仅凭单一时间点的32个稀疏传感器观测即可重建完整的滑坡运动时空演化轨迹（NRMSE低于5.8%），且与实际沉积物的空间相关性（0.843）超过了校准物理模拟器的表现。 |
| [^100] | [Distillation with Errors: Sparse Unbiased Backpropagation for Knowledge Distillation](https://arxiv.org/abs/2611.27131) | 本文提出带误差蒸馏（DWE），利用 logit 梯度的零和结构构建稀疏成对的无偏梯度估计，并结合几何感知配对与分层抽样降低方差，从而大幅削减全词表知识蒸馏中输出头反向传播的计算成本。 |
| [^101] | [A Plan or an Adversarial Example? A Round-Trip Check for Latent World Models](https://arxiv.org/abs/2611.27124) | 针对潜空间世界模型规划可能产生“世界模型对抗样本”而非有效计划的问题，提出往返检验方法——用小网络从预测状态读回动作并惩罚不匹配，以识别规划器解决后又丢失任务的失效情况。 |
| [^102] | [CTRL: Concurrent Training and Rollout for LLM Reinforcement Learning via GPU Co-Residency](https://arxiv.org/abs/2611.27119) | CTRL提出将GPU共存作为LLM强化学习的第三种执行布局，让每张GPU同时常驻运行learner与rollout引擎，并通过主机侧learner、发布门控和步时上界实现训练与推理的高效并发。 |
| [^103] | [Highly Efficient Vertical Federated Clustering with Triangle-Inequality-based Pruning](https://arxiv.org/abs/2611.27117) | 该论文提出了基于三角不等式的精确剪枝方法TriCD和高效联邦聚类算法FedTriCD，通过将全局目标分解为客户端本地子问题，在保证聚类结果一致的前提下显著减少了通信轮次与本地计算开销。 |
| [^104] | [Federated and Balanced $k$-means: Communication Efficiency, Privacy, and Heterogeneity](https://arxiv.org/abs/2611.27114) | 本文提出Fabric，一种平衡且保护隐私的联邦k均值算法，通过ADMM将平衡聚类分解为各客户端独立子问题，并借助校准噪声注入实现可证明的隐私保证，解决了现有垂直联邦聚类方法信息泄露和聚类不平衡的问题。 |
| [^105] | [Recurrent Amplification and Exact Nonexpansiveness Criteria for Query-Driven Erasure](https://arxiv.org/abs/2611.27109) | 该论文的核心贡献是构造了单步收缩但周期乘积却产生放大的周期词元序列，证明查询驱动的擦除机制可引发循环放大，并首次给出了可在 O(d) 时间内计算的秩一擦除一步非扩张性的充要谱范数判据。 |
| [^106] | [Reliable Behavior Discovery via False Discovery Rate Control](https://arxiv.org/abs/2611.27104) | 提出了DISCO方法，利用粗化e-过程的迭代搜索，在温和条件下可证明地控制行为发现中的错误发现率，并通过适应数据依赖的假设数量实现了比以往eBH方法更高的检验效能。 |
| [^107] | [EgoSurgHands: An Egocentric 3D Hand Pose Dataset & Benchmark for Open Surgery Training](https://arxiv.org/abs/2611.27101) | 该论文提出了首个提供真实采集的第一人称视角3D手部姿态标注的开放手术训练数据集与基准EgoSurgHands，可支持手术技能评估、AR引导和机器人示范学习。 |
| [^108] | [Reasoning in Two Dimensions: Quantifying Chain-of-Thought Faithfulness and Efficiency](https://arxiv.org/abs/2611.27100) | 该论文提出将思维链表示为语义推理图，并沿“语义支持对推理历史”与“通往答案对偏离路径”两个维度分解模型的预测依赖性，在同一框架下实现了对思维链非忠实性与低效性的全局及步骤级量化。 |
| [^109] | [PLoQ: A Prompt-Specific Low-Rank Basis for 4-bit Quantization of LLM Activations](https://arxiv.org/abs/2611.27092) | PLoQ提出了一种提示词自适应的低秩分解方法，在预填充阶段为每个提示词动态估计低秩基，将主导激活分量单独量化并在矩阵乘法中恢复，从而在无需修改权重的情况下实现大语言模型激活值的4比特量化。 |
| [^110] | [Writing Weights or Training Them? Spatial Reasoning through Weight-Space Programming and Low-Rank Adaptation](https://arxiv.org/abs/2611.27089) | 在空间推理任务上，权重空间编程带来的提升有限，而仅用0.09%可训练参数、以解题示例为监督训练的秩1 LoRA适配器可近乎完美地解决参照系与StepGame任务，并对未见模板和更高难度展现一定泛化能力。 |
| [^111] | [ORION: Exact Canonical-State Supervision for Systematic Generalization in Transformers](https://arxiv.org/abs/2611.27087) | ORION框架通过辅助头监督精确的规范执行状态来恢复Transformer的深度外推能力（66.7%），与解决长度外推的RoPE形成正交互补，二者结合在长度-深度联合外推中实现严格的加性增益。 |
| [^112] | [Auditing and Adapting kNNGuard for Long-Context Prompts](https://arxiv.org/abs/2611.27084) | 本文审计了无需训练的防护栏kNNGuard在长上下文提示上的性能退化现象，发现其在长输入上AUROC大幅下降，并将退化归因于隐藏状态漂移、MiniLM的256 token硬截断以及长文档压缩为单一最终token向量导致的信号稀释这三个原因。 |
| [^113] | [Weight-space programming and mechanistic interpretability for model editing in particle physics](https://arxiv.org/abs/2611.27079) | 该论文通过稀疏自编码器识别潜在方向的移除和语言模型智能体的免梯度权重编辑两种方法，成功修复了粒子物理基础模型中因预训练数据问题导致的违反红外安全性缺陷，且对模型其余功能几乎无影响，所需修改的参数远少于微调。 |
| [^114] | [Accelerating Muon with Spectral Residual Correction](https://arxiv.org/abs/2611.27077) | 提出AIMuon，通过引入经过矩阵符号变换的梯度-动量残差来校正Muon的谱更新步，在保留Muon动量递归的同时实现优化加速，并给出了随机平稳性理论保证，在多个语言模型上验证了其以更少步数达到Muon最终验证损失的效果。 |
| [^115] | [Online Learning for Minimizing Maximum Tardiness in Single-Machine Scheduling](https://arxiv.org/abs/2611.27067) | 该论文开创性地研究了单机调度中最小化最大延误的在线学习问题，并证明了在随机环境下FTL算法的期望1-遗憾为O(n log n / Δ²)，同时给出相应的高概率界。 |
| [^116] | [COMPASS: Cross-modal Observation with Multi-Path Adaptive Stopping and Selection for Early Time Series Classification](https://arxiv.org/abs/2611.27065) | 本文提出两阶段框架COMPASS，通过跨模态特征交换、门控融合和共享四动作控制器，在多模态早期时间序列分类中联合决策停止时机与预测分支选择，在准确率与提前量的权衡上优于现有基线。 |
| [^117] | [Rethinking Cross-Sensor Domain Adaptation for Bridge Point Cloud Semantic Segmentation](https://arxiv.org/abs/2611.27060) | 本文重新审视桥梁点云语义分割中TLS到MLS的跨传感器域自适应协议，系统分析了采样密度、表面粗糙度、颜色和采集视角在稀疏体素表示中的影响机制，并评估了源模型收敛程度及多种域自适应方法在不同训练周期下的增益效果。 |
| [^118] | [Structural Agreement Indicates Prediction Accuracy For Protein-Ligand Co-Folding](https://arxiv.org/abs/2611.27058) | 提出CORE分数（重复集合浓度分数），通过综合多次预测间的聚类紧密度与集合支持度来衡量结构一致性，比现有置信度分数能更准确地判断蛋白质-配体共折叠预测的正确性。 |
| [^119] | [A Continuous Relaxation of Quantization-Aware Training](https://arxiv.org/abs/2611.27056) | 提出RelaxQAT，通过在训练过程中逐渐连续地改变位宽来松弛感知量化训练，从而避免低位量化中损失曲面的鞍点和平坦区域问题，显著提升量化模型性能。 |
| [^120] | [Action-Conditioned Spatio-Temporal Forecasting](https://arxiv.org/abs/2611.27044) | 该论文提出了PairST基准（涵盖交通、电力传输和供水分配的六个数据集，提供同一系统状态下有无干预动作的成对未来数据）以及ActionST方法，通过在与干预无关的参考预测上写入显式动作响应，实现对受影响区域动作效果与未受影响区域不变性的分别评估。 |
| [^121] | [Hypergraph Structural Entropy-Enhanced Hierarchy-Aware Semi-Supervised Graph Classification](https://arxiv.org/abs/2611.27043) | 提出HypSEE框架，通过构建游走诱导与嵌入衍生超图捕获层次感知的局部和长程高阶关系，并利用超图结构熵引导的层次超图池化方法实现半监督图分类中的层次表示学习。 |
| [^122] | [Mind World Models: Joint Learning of Motor EEG Representations and Action-Conditioned World Dynamics](https://arxiv.org/abs/2611.27029) | 提出心灵世界模型（MWM），将脑电视为世界模型的潜在动作，从无动作标注的自然情景脑电与第一人称视频中学习运动脑电表征，并引入侧化感知Transformer（LAT）分离双侧共享与单侧特异特征，实现以运动为锚点的脑电预训练。 |
| [^123] | [Toward a digital twin of the ICU patient: Forecasting and treatment response in a generative event model](https://arxiv.org/abs/2611.27028) | 研究者训练了7900万参数的生成式事件模型TWICE，无需任务特定训练即可通过模拟50条患者未来轨迹，在19项ICU临床结局预测上达到与144个监督模型最佳配置相当的性能，是实现ICU患者数字孪生的重要一步。 |
| [^124] | [Integral Multipliers and Arithmetic Obstructions for Two-Hidden-Layer ReLU Maxima](https://arxiv.org/abs/2611.27018) | 本文借助代数拓扑工具（整数立方填充与同调障碍类）证明了双隐藏层整数参数ReLU网络精确表示最大值函数的最小正输出乘子——六输入为2、七输入为2或6，并刻画了七输入情形中剩余的算术障碍。 |
| [^125] | [Ruby: State-of-the-Art Potency Prediction across Discovery Regimes Using Learned Structural Representations](https://arxiv.org/abs/2611.27015) | Ruby是一个利用结构基础模型Pearl的中间表示、无需显式结合姿态的鲁棒通用结合亲和力预测器，通过专有数据与公开数据集GenesisPotency-2026训练，在药物发现的各个阶段均实现了最先进的药效预测。 |
| [^126] | [Loop-OPF: Solver-Informed Learning for Optimal Power Flow with Inference-Time Topology Adaptivity](https://arxiv.org/abs/2611.27013) | Loop-OPF是一种由PDHG迭代构建的求解器引导神经网络，在凸性条件下保证DC-OPF预测的最优性与可行性，并通过自监督方式学习PDHG参数实现无需解标签的加速训练与对推理时未见拓扑的自适应。 |
| [^127] | [Fair Property Tax: Reducing Regressivity in Machine-Learning Valuation](https://arxiv.org/abs/2611.27009) | 与库克县评估官办公室合作开发的两种训练方法（直接惩罚法与代理加权损失法）能在保持预测精度的同时显著减少机器学习房产估值中的累退性，其中直接法已实际应用于公共住宅估值软件。 |
| [^128] | [JevRisk: Zero-Shot Financial Risk Assessment with a Language Decision Model](https://arxiv.org/abs/2611.26999) | JevRisk提出了一个基于语言决策模型的零样本金融风险评估系统，通过遵循金融监管指南用代码计算精确风险指标，摆脱了对人工标签的依赖，并能在不同客户和风险类型间泛化。 |
| [^129] | [Distributional Algebro-Geometric Persistence (dAGP): Wasserstein-Stable Polynomial-Relation Signatures](https://arxiv.org/abs/2611.26986) | 本文提出分布代数几何持续性（dAGP），通过统计概率测度矩矩阵中近似多项式关系方向在多项式次数和容差两个参数下的数量，构建了一个刻画分布代数结构且具有 Wasserstein 稳定性的双参数整数签名。 |
| [^130] | [Affine-Image Propagation for Tight and Scalable Neural Network Verification](https://arxiv.org/abs/2611.26978) | 本文提出了一种新颖的边界传播方法 affine-image SDP-CROWN，通过仿射图像约束更精确地近似预激活域并联合优化线性边界，从而实现对 $\ell_{2}$ 对抗攻击紧凑且可扩展的神经网络验证。 |
| [^131] | [Run One, Get Two: Near-Free Streaming Safety Checks for Local LLMs](https://arxiv.org/abs/2611.26975) | ToRA通过让第二个模型作为额外一行嵌入生成器的解码步骤并共享权重与KV缓存，以每个token不到4%的额外延迟实现了接近专用防护模型准确性的本地LLM流式安全检查。 |
| [^132] | [Spectral Estimation with Deformed Decompression](https://arxiv.org/abs/2611.26945) | 该论文提出了一种名为“形变解压缩”的谱估计新方法，旨在克服谱求逆的不稳定性以及自由解压缩仅适用于酉不变系综的局限，从而能够仅凭小规模矩阵的信息推断大规模样本协方差矩阵的特征谱。 |
| [^133] | [CENTRY: A Probabilistic Latent-State Model for Continual AI Risk Estimation, Monitoring, and Regulation](https://arxiv.org/abs/2611.26940) | 本文提出CENTRY概率框架，通过结构化潜变量模型将AI风险分解为一个捕捉AI-环境动态的快速交互子空间和三个编码数据、硬件与模型长期风险状态的慢速潜在子空间，从而实现对已部署AI系统全生命周期的持续风险估计、监测与调控。 |
| [^134] | [SecretAgent: A Multi-Agent Framework for Privacy Protection against Vision Language Models](https://arxiv.org/abs/2611.26936) | 提出了首个通用的多智能体框架SecretAgent，通过LLM智能体迭代定位泄露隐私的视觉证据并自适应生成不可见扰动，同时借助外观审查智能体把关，在有效抵御视觉语言模型隐私推断的同时保持图像的视觉保真度。 |
| [^135] | [Improving the Optimisation Landscape for Particle Mixtures](https://arxiv.org/abs/2611.26931) | 该论文证明了混合预测似然在梯度优化中会使匹配不佳的粒子得不到训练信号，并提出将个体风险与混合风险进行凸组合的正则化方法，通过反向KL屏障确保每个粒子都能获得训练信号，从而改进了粒子混合的优化景观。 |
| [^136] | [Large Reported CHB-MIT Adaptation Gains Disappear with Pretrained Encoders and Matched Probes](https://arxiv.org/abs/2611.26929) | 本文通过复现发现，Dani和Liebe声称的SSL微调在CHB-MIT癫痫检测上带来的36.2和29.5点ROC-AUC提升，实际上是配置错误（预训练权重未被正确加载、探测时dropout未关闭）造成的假象——在正确设置下适应增益几乎完全消失。 |
| [^137] | [ERA: Embed, Route, Adapt for Cold-Start Cross-Model Latent Retrieval](https://arxiv.org/abs/2611.26919) | ERA提出了一种冷启动跨模型潜在检索系统，通过余弦最远点采样选取少量锚点文档、利用结合正交正则化的流形对齐损失训练轻量级适配器，并借助确定性路由器升级处理困难查询，从而无需对整个语料库重新嵌入即可将检索系统迁移到新的嵌入模型。 |
| [^138] | [What Does the Visual Encoder Buy? Visual Information Gain in Encoder-Free Multimodal Pretraining](https://arxiv.org/abs/2611.26917) | 提出视觉信息增益（VIG）这一精确度量，将无编码器多模态模型从图像中获取的信息分解为全局信息与布局信息，发现图像可降低3%至46%的预测损失，全局信息在训练早期即出现而布局信息随训练缓慢增长。 |
| [^139] | [Multi-Tier Edge Triage: Optimizing Latency for Safe Clinical Agents](https://arxiv.org/abs/2611.26914) | 该论文提出多层级边缘分诊安全筛查架构，通过在临床智能体宿主服务器本地部署轻量级第一层分类器以进程内处理明确查询、仅将模糊查询升级至云端LLM评审器，在保障多维安全合规的同时显著降低响应延迟。 |
| [^140] | [Surface Cues Over Structure: Probing LLM Behavior on Brain Connectivity Graphs](https://arxiv.org/abs/2611.26913) | 该研究通过受控实验发现，大语言模型在基于脑功能连接图进行ADHD诊断时，主要依赖表面线索（如脑区名称）而非真正对受试者特定的连接结构进行推理。 |
| [^141] | [Budget-Aware Predict-and-Verify Theorem Proving with Heterogeneous LLM Solvers](https://arxiv.org/abs/2611.26906) | 本文提出BudgetPV，一种预算感知的预测-验证智能体系统，利用跨求解器预测器在异构LLM求解器之间动态分配计算资源，并根据证明进程的不同阶段（新子目标的求解器选择与已尝试子目标的算力追加）调整策略，从而在保证证明成功率的同时最小化定理证明的计算成本。 |
| [^142] | [Perceive Once, Escalate on Demand: Handing Off Visual KV Memory from Small to Large VLMs](https://arxiv.org/abs/2611.26902) | 提出Baton方法，通过从大型模型自身答案中无标签蒸馏逐层线性映射，将小型VLM的视觉KV缓存记忆传递给大型VLM，避免升级时重复编码历史信息，使大型模型只需编码当前帧。 |
| [^143] | [Your Judge Inherits Your Prompt: In-Context Label Bias in an LLM-as-Judge Is Directional, Not Symmetric](https://arxiv.org/abs/2611.26881) | 研究表明，多模态视频问答评估流水线中的LLM评审器会从其少样本示例中单向继承标签偏置：当示例偏向"Yes"时其接受率骤升约20–30个百分点并伴随单侧错误，而当示例平衡或偏向"No"时则接近人类共识，证明这种上下文标签偏置是方向性的而非对称的。 |
| [^144] | [How Language Models Compress and Compare: Understanding Selection with Token Covariance Maps](https://arxiv.org/abs/2611.26874) | 该论文提出词元协方差图这一提示级几何分析工具，揭示了仅解码器Transformer在结构化选择任务中普遍遵循“先压缩后比较”的层级机制——早期层弥散表示提示、中间层压缩为共享低秩瓶颈、后期层重构上下文与选项间的比较几何。 |
| [^145] | [LLMs Memorize Their Own Hallucinations During Post-Training](https://arxiv.org/abs/2611.26863) | 该论文发现，采用以推理为中心的后训练方案（如基于结果的强化学习奖励和“认知”行为注入）会导致大语言模型在知识任务训练中记住自己在长推理链中产生的幻觉，且这一现象在多个开源模型上得到验证。 |
| [^146] | [Self-Supervised Learning with Recursive Quadrature Filters for Neural Time Series](https://arxiv.org/abs/2611.26856) | 本文提出一种自监督学习方法，通过让可学习的递归正交滤波器（一类受生物学启发的对角状态空间带通滤波器）编码器的当前状态预测其自身未来活动，在无标签神经时间序列上预训练编码器，再结合读出层完成下游任务。 |
| [^147] | [LoCo-WAM: Test-Time Training as Long Context for World-Action Models](https://arxiv.org/abs/2611.26853) | LoCo-WAM 通过测试时训练利用新观测帧更新视频专家中的低秩适配器，使世界-动作模型在不延长观测窗口、无需专家动作标签、额外数据或元训练的情况下从更长的经验流中学习，从而在四个仿真基准上显著提升任务成功率。 |
| [^148] | [Scoliosis5K: Multimodal Gait Analysis via Clip-Conditioned RGB-Silhouette Fusion](https://arxiv.org/abs/2611.26849) | 该论文构建了多中心多模态步态数据集Scoliosis5K，并提出片段条件化的RGB-轮廓融合架构，用于脊柱侧凸相关步态的参与者级分类。 |
| [^149] | [A Sparsity Analysis of Traffic Forecasting Architectures: How Much Missing Data is Too Much?](https://arxiv.org/abs/2611.26847) | 本文系统评估了图神经网络、Transformer和选择性状态空间模型等交通预测架构在从零到完全传感器缺失的不同稀疏度下的表现，揭示了基准精度与缺失数据鲁棒性之间的权衡，并定义了预测可靠性阈值。 |
| [^150] | [Neural Geometry as a Learning Signal](https://arxiv.org/abs/2611.26832) | 本文证明只需监督模型内部表示的几何结构（如傅里叶谐波和低维螺旋结构），就能在完全不监督输出的情况下取得与标准微调相当的模型改进效果。 |
| [^151] | [Graph-XGB: A Hybrid GNN Framework for Outdoor-to-Indoor 5G Signal Reconstruction via Aerial-Only UAV RF Sensing](https://arxiv.org/abs/2611.26828) | 提出Graph-XGB混合推理框架，通过改进的SpiderGNN编码器（结合楼层感知三维距离与桥接边保证）将无人机边界射频测量传播为室内嵌入，并与非线性回归解耦，实现仅凭空中感知的室外到室内5G信号三维重构。 |
| [^152] | [Skill-JEPA: Learning Planner-Aligned Action Spaces for World Models](https://arxiv.org/abs/2611.26825) | Skill-JEPA通过状态条件变分自编码器将短动作序列编码为符合标准高斯先验的潜技能，让世界模型预测技能效果、规划器在同一高斯分布上搜索技能，从而解决了规划器采样动作与模型训练动作分布不匹配导致预测不可靠、远距离目标规划易失败的问题。 |
| [^153] | [A Unified Framework for Intelligent Cyber Attack Detection and Alert Correlation](https://arxiv.org/abs/2611.26824) | 本文提出一个融合签名规则、机器学习和序列模式挖掘算法的统一框架，用于网络流量分类与告警关联，以缓解安全运营中心（SOC）的告警过载问题并有效检测多阶段网络攻击。 |
| [^154] | [Realistic Robustness Evaluation of Tabular Data Driven Models](https://arxiv.org/abs/2611.26818) | 本文提出了AURORA，一个模块化开源工具，通过在统一框架中整合多种对抗扰动方法和评估指标，为表格数据分类的机器学习模型提供系统化的鲁棒性评估。 |
| [^155] | [Search From the Start: Search-Native Pretraining for Language Models](https://arxiv.org/abs/2611.26809) | 提出“搜索原生预训练”方法，通过大规模地将网络搜索查询及检索到的网页响应注入预训练语料，让语言模型在预训练阶段就学会发起搜索并利用检索证据，从而在知识密集型任务上取得更优性能，且该优势在监督微调后依然保持。 |
| [^156] | [NitroGen: Observations on the Action Dynamics, Visual Instruction Following, and Spatial Exploration](https://arxiv.org/abs/2611.26807) | NitroGen视觉-动作基础模型虽然能根据局部游戏情境适应动作，但在长时程游戏控制中仍缺乏规划、恢复和探索能力。 |
| [^157] | [LEAN: Library-Based Adaptation for Asynchronous, Federated Fine-Tuning](https://arxiv.org/abs/2611.26805) | LEAN提出用服务器维护的秩-1基对“借阅库”机制取代联邦学习中的适配器平均聚合，各参与者检出子集本地微调后归还，从而实现完全异步、仅传输低秩参数的联邦微调。 |
| [^158] | [Don't Discard Your Rollouts: Reusing Teacher RL Traces for Student Distillation](https://arxiv.org/abs/2611.26803) | 提出利用GRPO训练中已记录的元数据（提示通过率和教师对数似然）将推理轨迹重用建模为数据选择问题，通过SFT预热加DPO的分阶段蒸馏策略，把教师的强化学习轨迹高效重用于学生模型训练，避免了昂贵的重新生成数据。 |
| [^159] | [Memorization and Generalization in Generative Models: A Perspective From Energy-based Modeling and Associative Memory](https://arxiv.org/abs/2611.26797) | 本文从能量建模与联想记忆的视角提出：扩散模型本质上是一种联想记忆，随着数据增长而涌现的“伪状态”吸引子正是生成能力的开端，因此记忆与泛化可以在同一检索过程中共存。 |
| [^160] | [CADENCE:Temporally Coherent Whole-Body Human Mesh Recovery that Preserves Expressive Hand Motion](https://arxiv.org/abs/2611.26783) | CADENCE提出了一种面向单目视频的全身SMPL-X网格恢复系统，通过沿运动学树的注意力机制为每个关节自适应选择学习到的零相位滤波器混合，在平滑躯干运动的同时保留快速手指动作等富有表现力的手部运动。 |
| [^161] | [Who Should Verify? Reviewer Selection with Partial Information and Outcome Audits](https://arxiv.org/abs/2611.26770) | 该论文研究部分信息下的审阅者选择问题，提出了认证所选策略至多落后最优策略指定差距的最低成本证书方法——在仅答案密钥付费时给出精确停止规则与整数规划，在审阅与密钥均付费时给出对数因子的多项式时间近似算法，并证明除非P=NP该对数依赖是紧的。 |
| [^162] | [Pharos: Learning to Control Rare Events at Scale](https://arxiv.org/abs/2611.26765) | Pharos是一个基于随机最优控制的框架，通过联合学习转变概率与模拟引导策略使采样和学习相互促进，实现了比以往算法快数量级的收敛速度，并能扩展到毫秒时间尺度的复杂分子体系。 |
| [^163] | [Depth-Linked Residual Tokens for Multimodal Retrieval and Generation](https://arxiv.org/abs/2611.26752) | 提出DepThink框架，在单次前向传播中通过轻量级可学习残差递归将一串残差标记的隐藏状态相互耦合，使其逐步捕捉互补信息，从而兼得迭代架构的逐步精炼能力与并行单次推理的高效性，服务于多模态检索与生成。 |
| [^164] | [Cross-Attack Robust Aggregation for Federated Learning](https://arxiv.org/abs/2611.26731) | FedTether 通过基于客户端更新与逐坐标中位数偏差的鲁棒死区评分进行重加权并作为服务器端优化器步骤，成为唯一能在隐蔽攻击、高强度攻击和非独立同分布数据等各种情形下均保持顶尖性能的联邦学习鲁棒聚合方法。 |
| [^165] | [SparseHPS: A Unified Model for Human Pose and Shape across Depth Cameras and LiDARs](https://arxiv.org/abs/2611.26730) | 提出首个传感器无关的统一模型SparseHPS，仅需一套权重即可从深度相机或激光雷达采集的人体点云中恢复SMPL-X三维人体姿态、形状与全局平移，同时适用于单帧和视频流输入。 |
| [^166] | [Parasite Features: Causal Abstraction Through Spurious Pathways](https://arxiv.org/abs/2611.26711) | 论文揭示了一类“寄生特征”能够通过因果抽象的等价性检验，导致分布式对齐搜索等可解释性方法误认为模型实现了某种特征，暴露了该框架的盲点。 |
| [^167] | [On the loss landscape of the linear neural network: beyond critical points](https://arxiv.org/abs/2611.26671) | 本文提出了一种包含线性神经网络损失景观中所有临界点的中间几何结构——“临界谷线”，它提供了可二维可视化的损失景观全局描述，并表明梯度下降会被该结构吸引从而沿其持续降低损失至全局最小值。 |
| [^168] | [Distributional Deep Hedging](https://arxiv.org/abs/2611.26652) | 本文提出分布式深度对冲方法，通过学习对冲损益分布的分位数，使风险偏好可在训练后作为权重灵活应用，从而将风险敏感对冲问题简化为一系列风险中性问题，避免了为每个新风险偏好重新训练策略的需求。 |
| [^169] | [SDG: State-Dependent Gating at Chunk Boundaries for Linear Attention](https://arxiv.org/abs/2611.26637) | 提出SDG方法，通过在分块边界处依据记忆状态的变化计算逐元素门控，为线性注意力模型提供状态反馈的遗忘机制，同时保持块内并行计算的高效性。 |
| [^170] | [Represented but misread: Post-training changes how language models read out what they represent](https://arxiv.org/abs/2611.26584) | 研究发现后训练并未破坏语言模型内部的表征（线性探针仍能准确区分人类与AI文本），而是改变了模型将表征读取为答案的方式，导致回答灵敏度下降和对先验信息的错误利用。 |
| [^171] | [Learning to Defer under Expert Drift](https://arxiv.org/abs/2611.26440) | 提出EDA-L2D框架，利用序列模型对专家近期决策结果历史进行建模，显式捕捉因疲劳、认知偏差等因素导致的专家可靠性时间漂移，实现自适应的延迟决策策略。 |
| [^172] | [SKILL.PNG: Learning Agent Skills in Pixel Space](https://arxiv.org/abs/2611.26296) | 提出Skill.png方法，通过梯度下降优化图像像素来学习可跨VLM通用迁移的智能体技能，在大多数基准设置中超越最先进的文本技能，且token开销极低。 |
| [^173] | [To predict or to persist: how decoherence shapes learning in a spin-qubit quantum reservoir](https://arxiv.org/abs/2611.26215) | 该研究发现适度的退相干（如 T₂=1 微秒时）反而能提升量子库计算对 NARMA 信号的预测准确度，且混合单/双量子比特观测量读出优于纯布居数读出，而自主生成中极限环的持续性与相干时间并无明显的系统性关联。 |
| [^174] | [When Do Rankings Beat Signs? Gradient Direction Recovery under Even Nuisance](https://arxiv.org/abs/2611.26055) | 该论文证明在偶函数干扰下，跨对排序策略Reverse仅需Õ(d+√(d/ε))次评估、单轮反馈即可将梯度方向恢复至角精度ε，达到理论下界且优于仅使用对内符号的方法。 |
| [^175] | [PSTO: A PDE-structured transport operator with shared implicit updates for multiscale CDR systems](https://arxiv.org/abs/2611.25607) | PSTO通过将PDE结构嵌入神经架构并在所有输运变量间共享单一输运算子模块、结合松弛不动点迭代实现类隐式修正，仅需无反应数据训练即可在推理时无需重训练地泛化到完整的多尺度对流-扩散-反应系统，且不增加参数量。 |
| [^176] | [Variance-Optimal Counterfactual Policy Design for Sample-Efficient Online A/B Testing](https://arxiv.org/abs/2611.24757) | 该论文提出一种方差最优的反事实策略设计方法，通过联合优化动作的采样方式，将流量引导至两种策略不一致的动作上，从而显著降低在线A/B测试的方差并提升样本效率。 |
| [^177] | [Testing World Models on Actions They Have Never Seen: Held-Out Plans, Forecast Accuracy and Price Rules in Retail](https://arxiv.org/abs/2611.22537) | 该论文提出了一种检验动作条件世界模型对从未见过的动作泛化能力的测试方法，并在零售定价场景的2,853个模型中发现：预测准确度仅能可靠地区分不同类型的模型，却无法区分同一世界模型的不同版本（在高需求周甚至出现反转），且JEPA预训练等学习动作效果的训练技术并未降低模型在新计划上的误差。 |
| [^178] | [Cancellation-Free Constrained SquareCB for Safe Controller Selection](https://arxiv.org/abs/2611.21880) | 该论文提出一种对成本施加修正权重（预算内权重为零、超预算时恰好拉回预算）的约束SquareCB算法，在可实现性和带边际 δ 的 Slater 条件下实现了 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 的遗憾和无需轮次间抵消的违约界 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，适用于安全控制器选择。 |
| [^179] | [Does Learning Order Explain Forgetting? The Lie Bracket as an Early-Warning Signal for Catastrophic Forgetting Without Old Data](https://arxiv.org/abs/2611.20087) | 本文实证验证了两个任务梯度场的李括号决定了训练顺序导致的权重差异并与灾难性遗忘相关，且仅凭旧任务训练结束时存储的梯度和前20个Hessian特征对，就能在无需旧数据的条件下早期预测遗忘，效果优于EWC惩罚项。 |
| [^180] | [Grounding Language Models in Speech Does Not Fix "AI Flavor": Calibrated LLM Judges, Goodhart Traps, and a Preference-Tuning Fix](https://arxiv.org/abs/2611.19840) | 该论文通过在0.6B模型上的受控实验证明，注入语音或韵律信号的三种方法（音素交错续训、韵律计划条件生成、基于TTS韵律距离的规则奖励）均无法消除LLM中文输出的“AI味”，并指出当前n≈20的单评分员盲评只能检测极端偏好的评估方法是根本症结，同时提出一种基于偏好调优的修复方案。 |
| [^181] | [SPARC: Evidence-guided adaptive model composition for materials discovery](https://arxiv.org/abs/2611.19050) | SPARC是一个面向材料发现闭环实验的自适应模型组合框架，通过折外预测评估竞争模型族、以稀疏后验聚合分配模型权重，并利用预测离散度与残差变异性定义信任度来调节探索策略。 |
| [^182] | [Training Reasoning Models in Simulation to Natively Use Real Robots](https://arxiv.org/abs/2611.18562) | 该论文提出PEAR后训练引擎和RURL强化学习方法，仅通过大规模仿真训练（无需演示数据和外部框架），使开源27B推理模型获得可泛化到真实机器人上的原生机器人操控能力，缩小了开源模型与前沿专有模型的差距。 |
| [^183] | [When Copying Is Hard to Beat: A Negative Result in Clinical Assessment Forecasting](https://arxiv.org/abs/2611.18142) | 该研究是一项负面结果，表明在预测次日临床评估时，各类模型相比简单复制前一天答案的持续性基线并无一致优势。 |
| [^184] | [Finite speed Ornstein-Uhlenbeck-Kac processes and their applications in generative modeling](https://arxiv.org/abs/2611.17625) | 本文提出有限速度的Ornstein-Uhlenbeck-Kac（OU-K）过程，利用勒让德多项式显式求解Euler-Poisson-Darboux方程，并借助多项式分布公式刻画其条件速度场，从而成功将该过程应用于生成流模型。 |
| [^185] | [AscendGYM: A Cost-Aware Kernel Optimization Benchmark for Real-World Workloads on Ascend NPUs](https://arxiv.org/abs/2611.17621) | 提出AscendGYM——首个面向昇腾NPU真实工作负载的成本感知算子优化基准，其48个算子的测试数据类型与张量形状均来自真实训练和推理任务，并支持模型评估与模型-框架评估两种模式来衡量智能体的算子优化能力。 |
| [^186] | [The Approximation Ratio of $k$-means++ Is $2\ln k+O(1)$](https://arxiv.org/abs/2611.14573) | 该论文精确刻画了 k-means++ 算法的最坏情况近似比，证明 $R_k=2\ln k+O(1)$（即 $2H_{k-1}+2\le R_k\le 2H_{k-2}+5$，上下界相差不到 3），并确定 $R_2=5$，基本解决了这一十余年悬而未决的问题。 |
| [^187] | [Uncertainty-Aware Hybrid Offline Reinforcement Learning with Online Action Selection Enabled by Sparse Gaussian-Process Critics](https://arxiv.org/abs/2611.14374) | 本文提出用稀疏变分高斯过程评论家同时替代树集成评论家与对抗自编码器惩罚，通过最大化不确定性调整的回报估计 μ-κσ，在混合离线强化学习框架HOFLON中实现不确定性感知的在线动作选择，提升了聚合反应器开车启动等任务的安全性与可靠性。 |
| [^188] | [Audit Before You Merge: Provenance, Probing, and Continual LoRA Composition](https://arxiv.org/abs/2611.10684) | 该论文提出在合并公开 LoRA 适配器前必须进行来源审计（基础模型匹配与非平凡增量范数检查）和无泄漏探测协议，并证明朴素合并会造成显著性能下降，而任务算术可将性能维持在未合并基础模型的标准误差范围内。 |
| [^189] | [SLA-Aware Task Offloading in the IoT Edge-Cloud Continuum: Class-Weighted Deep Reinforcement Learning Against an Exact Optimum](https://arxiv.org/abs/2611.10638) | 该研究通过四个可精确求解的基准系统证明：在物联网边云任务卸载中，类别盲的惩罚设计即使配合完美学习器也会导致成本高出最优解65%-118%，而修正后的按SLA类别加权的奖励设计可使PPO智能体性能逼近精确最优解。 |
| [^190] | [La-JEPA: Compact Robot Policies with Latent World Modeling](https://arxiv.org/abs/2611.08156) | La-JEPA是一种紧凑的世界-动作模型，它在预训练V-JEPA 2.1编码器的特征空间中从头训练策略Transformer，无需具身预训练即可以更少的参数和更快的推理速度在RoboTwin 2.0和LIBERO-Plus基准上取得最先进的机器人控制性能。 |
| [^191] | [Equivariant Deep Learning in Direct-Product Space for Electronic-Structure Hamiltonian Prediction](https://arxiv.org/abs/2611.07695) | 提出DP-Learner框架，在直积空间中直接对类哈密顿量算符块进行等变消息传递，同时解决了传统直和表示与哈密顿量之间的表示不匹配问题及计算效率瓶颈。 |
| [^192] | [When Scores Conflict with Preferences: Pre-Alignment Diagnosis and Adaptive Correction of Constraint Drift in DPO](https://arxiv.org/abs/2611.07375) | 该论文揭示了DPO训练中的系统性失效模式“约束漂移”，提出可从数据集统计量直接计算的闭式对齐前诊断指标 $\rho^{*}_{\mathrm{eq}}$，并据此构建冲突感知DPO（CoDPO），通过自适应降低冲突样本对的权重来校正约束漂移。 |
| [^193] | [The Alignment Tax Is Graded: Task-Selective Capability Loss Under Length-Normalized DPO with SFT Retention](https://arxiv.org/abs/2611.06476) | 对齐微调带来的能力损失并非均匀发生，而是分级且任务选择性的——抽取类任务几乎不受影响，JSON生成轻微下降，而路由任务出现约30分的严重退化。 |
| [^194] | [Unsatz: Evaluating Scientific Discovery Without a Given Ansatz](https://arxiv.org/abs/2611.03713) | 该论文提出了 Unsat​z 基准，首次要求 LLM 智能体在没有给定模型形式（拟设）的情况下自主构建拟设——在有限预算内设计实验并用统一建模语言编写候选模型程序——从而更真实、更完整地评估人工智能的科学发现能力。 |
| [^195] | [Why Tabular Foundation Models Excel on Relational Data They Have Never Seen](https://arxiv.org/abs/2611.02936) | 本文通过多样化实验证实并深入揭示了单表表格基础模型在结合无参数关系编码器后，即使从未经过任何关系数据预训练，也能在关系数据库预测任务上表现出色的内在机制。 |
| [^196] | [Reinforcement Learning Under Runtime Confounding: Policy Learning and Off-Policy Evaluation](https://arxiv.org/abs/2610.40056) | 该论文针对强化学习中部署时关键变量不可观测的“运行时混淆”问题，提出了相应的策略学习算法和半参数高效的离线策略评估方法，以解决混淆随时间传播所导致的学习偏差。 |
| [^197] | [Let the Gate Decide: Per-Lookup Precision for Quantized Conditional Memory in LLMs](https://arxiv.org/abs/2610.35985) | GateRQ利用可从1比特编码计算的门控值（对频繁n-gram小、对长尾n-gram大）来决定各查找所需的量化精度，将每行存储为1比特残差编码阶梯，无需重训即实现GPU表内存7.8倍压缩和vLLM吞吐量11%-31%的提升。 |
| [^198] | [Benchmarks Are Surprisingly Compaction-Friendly](https://arxiv.org/abs/2610.31778) | 该论文首次在固定的智能体框架内实现五种上下文压缩策略，并在三个长时程基准上系统评估三个前沿模型，发现基准任务对有损压缩出人意料地友好——对百万级上下文模型而言，压缩阈值几乎不影响任务性能。 |
| [^199] | [ReDence: A Hybrid Agentic and Decision-Model Framework for Rapid, Evidence-Grounded Detection of Modern Social Engineering Scams](https://arxiv.org/abs/2610.30963) | 该论文提出开源混合框架ReDence，通过微调多模态智能体“调查员”与专用“决策器”模型协同工作，实现对社会工程学诈骗的快速、自适应、基于证据的自动化检测。 |
| [^200] | [Tracing Subliminal Learning Through Pretraining](https://arxiv.org/abs/2610.30884) | 该研究证明“潜意识学习”现象在硬标签蒸馏和同策略软标签蒸馏中均普遍存在，且不受权重初始化、数据顺序、语料库、模型规模和架构的影响，并首次发现传递偏好与获得偏好的能力在预训练过程中独立发展，导致教师与学生角色互换时迁移强度呈现强烈不对称。 |
| [^201] | [Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories](https://arxiv.org/abs/2610.29378) | 提出OnQ流水线，通过在策略蒸馏利用学生模型自生成的轨迹进行训练，解决了固定响应监督导致的训练与生成前缀不匹配及高熵处分布失配问题，从而恢复低比特量化大模型的推理能力。 |
| [^202] | [Two $\operatorname{ReLU}$ Layers is All You Need](https://arxiv.org/abs/2610.28587) | 本文证明了任意深度的 ReLU 网络都可以被仅含两个隐藏层的 ReLU 网络精确模拟，并给出了基于同调代数填充和组合方法的两条证明路径。 |
| [^203] | [MemT-1.0: Towards Memory-Native Evolution](https://arxiv.org/abs/2610.26825) | MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。 |
| [^204] | [Computation, Not Memory: Chain-of-Thought and State Tracking in LLM Agents](https://arxiv.org/abs/2610.25760) | 该论文通过区分“行动前即时生成推理”与“在提示中保留历史推理笔记”两种角色，发现真正减少状态违规行动的主要来源是行动时刻的即时推理（计算），而非记忆式的历史推理保留。 |
| [^205] | [CHSM8: A Chess Transformer that Follows Natural-Language Style Prompts](https://arxiv.org/abs/2610.25356) | CHSM8是一个2.1亿参数的国际象棋Transformer，无需搜索即可根据自然语言风格提示下棋，达到1837 Elo并击败了包括参数量为其13倍模型在内的所有同类公开模型，并通过合成数据微调使其能够理解文本风格指令。 |
| [^206] | [Can Agents Keep Secrets They Were Never Told? Benchmarking Latent Privacy Risks Across Sessions with MemJoin](https://arxiv.org/abs/2610.24359) | 该论文提出MemJoin隐私基准，揭示智能体长期记忆系统会通过跨会话组合碎片化线索，在执行普通任务时泄露用户从未被明确告知的敏感关系。 |
| [^207] | [CoCoNet: Convex Conformal Network for Multi-Dimensional Uncertainty Regions](https://arxiv.org/abs/2610.24246) | 提出 CoCoNet，通过端到端学习依赖输入的凸一致性分数并直接最小化其子水平集大小，生成灵活数据自适应的凸预测区域，同时借助分割保形校准保留有限样本覆盖率保证。 |
| [^208] | [Counting with Attention and LayerNorm: Norm Scaling and Representation Geometry](https://arxiv.org/abs/2610.16653) | 本文量化了单层注意力实现精确计数的参数范数代价 $B_{\min}$，证明线性注意力可在与序列长度 $L$ 无关的范数下精确计数，而softmax注意力在无归一化时需要范数随 $L$ 增长、在后归一化下随 $L^{3/2}$ 增长。 |
| [^209] | [Softmax-Free Attention in Spiking Transformers as Zero-Temperature Two-Level Systems: Applications to XAI](https://arxiv.org/abs/2610.15968) | 本文证明阈值化注意力的每个查询—键连接都是一个二能级系统，从而将具有阈值注意力的脉冲Transformer视为零温二能级系统网络（注意力图为基态、代理梯度为有限温度下的磁化率），并据此通过冷却与加热的热力学图景将热力学积分引入可解释人工智能（XAI）。 |
| [^210] | [NucleoNet: A Nuclear-Aware Determinant Ansatz for Multi-Geometry Neural Variational Monte Carlo](https://arxiv.org/abs/2610.12985) | NucleoNet 提出一种核感知的神经行列式拟设，将电子与原子核纳入统一的共享骨干网络中联合演化，使单一模型可跨多个核几何构型联合优化，从而以更少的参数准确描绘分子势能面。 |
| [^211] | [Stochastic Grouping Conformal Prediction for Effective Subgroup Reliability](https://arxiv.org/abs/2610.11957) | 该论文提出随机分组保形预测（SGCP），通过学习随机分组映射让每个样本从校准行为相似的样本中获取校准信息，在无需敏感子群体属性的情况下实现跨临床子群体的可靠不确定性量化，并缓解最差群体瓶颈问题。 |
| [^212] | [AI Safety Considerations for Agents With Limited Time to Act](https://arxiv.org/abs/2610.10285) | 本研究证明在只能部分观察且必须在有限时间内行动的环境中，即使是完美的AI智能体也无法保证安全行为，因此AI安全或对齐的证明必须将环境与安全行动和智能体结合起来具体考虑。 |
| [^213] | [A Broader Look at Model Merging: Rethinking Implicit Regularization Induced by Task Arithmetic](https://arxiv.org/abs/2610.07990) | 本论文发现主流模型合并方法中系数搜索带来的隐式正则化实际上限制了性能，去除该正则化、直接优化合并模型乃至预训练模型的权重，即可在多种架构和极低数据量场景下显著提升合并后的多任务性能。 |
| [^214] | [Unifying matrix completion across domains with a synthetically pretrained foundation model](https://arxiv.org/abs/2610.06751) | MatrixFormer是一个完全基于合成低秩矩阵预训练的矩阵补全基础模型，能在单次前向传播中预测每个缺失条目的完整分布，并以零样本方式在因果推断、表格插补和推荐系统等多个领域取得有竞争力的性能。 |
| [^215] | [ThunderSync: Lossless Acceleration of Agentic Reinforcement Learning](https://arxiv.org/abs/2610.05935) | ThunderSync 通过在所需输入就绪后立即流式计算梯度，在不引入策略陈旧性的前提下消除流水线空闲，实现对智能体强化学习的无损加速。 |
| [^216] | [Poisson-GENERIC Neural Operators: Exact Metriplectic Structure in Function Space via Casimir Entropies](https://arxiv.org/abs/2610.05570) | 该论文提出泊松-GENERIC神经算子，通过引入潜在熵密度使总熵成为泊松算子的卡西米尔量，从而无需任何投影即可在函数空间中以代数方式精确保持GENERIC结构的退化性与雅可比恒等式。 |
| [^217] | [Cup and Cap Topological Neural Network](https://arxiv.org/abs/2610.03169) | 该论文提出了基于杯积与帽积运算的拓扑神经网络（CCNN），突破了传统拓扑深度学习中每层特征只能在相邻维度间传递的限制，通过融合多体相互作用来学习节点级变量，并在TopoBench数据集上展现出与现有单纯复形神经网络相当的竞争力。 |
| [^218] | [Co-design Gym: A Unified Benchmark for Embodiment-Policy Co-optimization](https://arxiv.org/abs/2610.02366) | 本文提出了Co-Design Gym，一套用于形态与策略联合协同优化的统一基准测试环境，突破了现有基准中形态固定、仅关注策略学习的局限。 |
| [^219] | [Disentangling Self-Distillation: Measuring and Modeling Acquisition and Retention](https://arxiv.org/abs/2609.39494) | 该论文首次将自蒸馏方法解耦为rollout来源、教师耦合方式和KL方向三个独立维度，通过1,200次系统实验构建统一框架，解释了以往固定组合研究导致矛盾结论的根源。 |
| [^220] | [A Dynamical Theory of LoRA in Continual Learning](https://arxiv.org/abs/2609.39367) | 该论文在师生模型中建立了持续学习下LoRA的精确动力学理论，揭示低秩更新虽能减少对已学任务特征的干扰从而缓解灾难性遗忘，但其初始化会减慢对新任务的适应。 |
| [^221] | [Self-Evolving Harness on Multiple Tasks with the Agent as Its Own Optimizer](https://arxiv.org/abs/2609.38372) | 该论文提出让同一个冻结模型兼任求解器与提议者的自进化执行框架，智能体在覆盖五个不同领域基准的多任务上直接编辑运行自身的框架代码，实现接近递归自我改进的效果。 |
| [^222] | [Pretraining Transformers with Quantized Softmax in Attention](https://arxiv.org/abs/2609.33591) | 本文提出K区间注意力方法对注意力中的softmax进行量化近似，系统研究了网格校准、舍入方式和直通替代梯度位置等设计选择对Transformer预训练的影响，并推导了相应的反向传播规则。 |
| [^223] | [Approximating Softmax in Pretrained LLMs: Model Sensitivity and Kernel Acceleration](https://arxiv.org/abs/2609.33586) | 该研究系统刻画了预训练大模型对softmax近似的敏感性，并提出以行最大值为锚点的粗对数权重表示Rowmax-PoT，从而大幅削减注意力内核中的指数运算开销、缓解张量核心与特殊函数吞吐量的巨大差距。 |
| [^224] | [Reuse or Relearn? A Spectral View of Earth Observation Foundation Models](https://arxiv.org/abs/2609.32756) | 该论文提出一种谱诊断方法来比较基础模型微调前后的变化，发现地球观测模型在微调中进行远大于自然图像模型的高秩权重更新、几乎不复用预训练结构，表明其下游性能主要通过重新学习而非复用预训练表示获得。 |
| [^225] | [AECSF: Adaptive Ensemble Conditional Score Filtering for High-Dimensional Nonlinear Data Assimilation](https://arxiv.org/abs/2609.32411) | 提出了一种免训练的自适应集成条件得分滤波器AECSF，利用条件Tweedie恒等式构建解析可处理的得分估计器，从而在高维非线性数据同化中同时避免粒子权重退化并捕捉非高斯后验结构。 |
| [^226] | [CLOOPD: Closing the Learner Loop in On-Policy Distillation](https://arxiv.org/abs/2609.24141) | CLOOPD提出一种闭环在策略蒸馏框架，将教师信号获取与学生端实现分离，通过自适应α航点、冻结评分批次、重复前向传播学生模型以及在独立token预算下分配演员工作量，显著提升了宏观准确率。 |
| [^227] | [Adaptive Determinantal Client Scheduling in Federated Learning](https://arxiv.org/abs/2609.23843) | 该论文提出利用行列式点过程（DPP）建模客户端多样性，设计了联邦学习中的自适应行列式客户端调度方法（ADCS），通过梯度信息与质量分数构建核矩阵实现质量与多样性的灵活权衡，并给出了收敛性保证。 |
| [^228] | [Similarity Pairing with Energy Mover's Distance for Self-Supervised Pre-Training at the LHC](https://arxiv.org/abs/2609.17738) | 该论文提出一种利用能量移动距离（EMD）按相似性配对真实对撞事件作为数据增强视图的方法，无需人工构造或模拟增强事件即可保持事件物理真实性，从而改进LHC自监督预训练。 |
| [^229] | [Self-Supervised Learning for Robust Resonance Mass Regression in Cascade Decays](https://arxiv.org/abs/2609.17726) | 该论文提出遵循基础模型范式，先用VICReg自监督预训练transformer编码器以学习对各种数据破坏不变的嵌入表示，再微调用于级联衰变中重共振粒子的质量回归，从而在系统不确定性和分布偏移下实现鲁棒的质量重建。 |
| [^230] | [OmniHarness: Harnessing Generalizable Visual Generation via Symbolic Policy Learning](https://arxiv.org/abs/2609.16057) | 提出OmniHarness框架，通过将验证过的执行经验抽象为可复用的符号策略，并结合执行过程中的中间验证与自主探究式练习，实现了具有强泛化能力的视觉生成。 |
| [^231] | [AhaBench: Do Agents Learn from Prior Experience? A Benchmark for Long-Horizon Continual Learning](https://arxiv.org/abs/2609.05435) | AhaBench 是一个长时程持续学习基准，通过谜题探索、带精确验证器的数学任务和自动售货机模拟三个组件，评估固定模型在获得先验经验后，在相关但支持已被移除、改变或延迟的条件下行为是否真正得到改善。 |
| [^232] | [Fractal dimension predicts quantum kernel collapse in angle-encoded data](https://arxiv.org/abs/2609.00475) | 该论文提出用数据的相关性分形维度 D2 作为先验量子比特数预算，可准确预测并避免角度编码量子核的几何坍缩，使量子核在真实硬件上以最小比特数保持有效。 |
| [^233] | [Singular Curvature in ReLU Training:Differentiation and the Gradient-Flow Limit Need Not Commute](https://arxiv.org/abs/2608.30960) | 该论文证明在 ReLU 训练中“先微分离散梯度下降再取极限”与“先取梯度流极限再微分”不可交换：离散精确导数收敛到无事件的区域传播子，而极限流的导数额外包含速度归一化的激活事件转移，由此产生秩一的奇异曲率差异，且全局凸性也无法将其完全抵消。 |
| [^234] | [Towards Understanding On-Policy Distillation through the Lens of Test-Time Scaling](https://arxiv.org/abs/2608.11829) | 本研究发现在线策略蒸馏主要提升采样效率而非扩展推理能力边界，其优势在小采样预算下显著，但在大预算下会减弱。 |
| [^235] | [Why Does CLAUDE.md Keep Growing? Catastrophic Remembering in Agentic Coding](https://arxiv.org/abs/2608.11095) | 该论文揭示并量化了智能体编程提示文件（如 CLAUDE.md）因“追加廉价、删除昂贵”的不完美记忆机制而近乎单调地无限制膨胀、最终导致智能体失效的“灾难性记忆”现象，并证明提示注释可有效遏制这一增长。 |
| [^236] | [SiMDex: Mining Similar Egocentric Videos for Cross-Embodiment Dexterous Manipulation](https://arxiv.org/abs/2608.04196) | SiMDex提出基于相似度的数据挖掘框架，通过召回-排序-重排序级联从3200万条第一人称人类视频中筛选出不足5%的最相关样本，无需修改VLA架构即可将跨本体灵巧操作成功率从47.7%提升至61.7%。 |
| [^237] | [Augmented Inverse Hybrid Weighting: Robust Inference under Deterministic and Random Distribution Shifts](https://arxiv.org/abs/2608.00701) | 该论文提出将分布偏移分解为可通过重加权校正的确定性协变量偏移和可通过数据集合并降低的随机扰动这一新模型，并由此导出增强逆距离加权方法，实现了在确定性与随机分布偏移下更稳健的跨人群统计推断。 |
| [^238] | [Agentic Coding in the Wild: Characterizing GitHub Copilot Traces at Production Scale](https://arxiv.org/abs/2608.00101) | 本文首次在生产规模上（740万用户、57亿次LLM调用）对GitHub Copilot编码智能体的追踪数据进行系统性分析，揭示了LLM调用与工具执行近乎1:1耦合、KV缓存命中率在轮次边界和模型切换时大幅下降、以及长尾资源占用分布等独特工作负载特性及其系统启示。 |
| [^239] | [Tokens are All You Need: Dual-purpose Semantic IDs for Achieving LLM-Level I/O Efficiency in recommendation systems](https://arxiv.org/abs/2607.24865) | 提出双用途语义ID，通过分层量化将连续嵌入压缩为离散token，既作为协同身份建模用户-物品交互，又可通过轻量级解码器按需重建嵌入，从而取代大规模向量存储，实现推荐系统中LLM级的I/O效率。 |
| [^240] | [A Multi-Center Dataset and Benchmark for SYNTAX Segmentation of Coronary Angiograms](https://arxiv.org/abs/2607.22139) | 该论文推出了CARDIAG——首个针对冠状动脉造影SYNTAX节段分割的多中心基准数据集，通过严格防止患者级和中心级数据泄露并提供前所未有的丰富注释（不平衡多类标签、导管与狭窄掩膜、专家不确定性图等），同时对24种涵盖CNN、ViT和SSM的深度学习架构进行了该领域最广泛的基准评估。 |
| [^241] | [An Analytically Trained Variational Quantum Circuit using the Output Probability Distribution of Quantum Phase Estimation on NISQ Hardware](https://arxiv.org/abs/2607.20943) | 该论文提出利用狄利克雷核解析计算训练目标，从而无需经典模拟QPE电路，即可在NISQ硬件上训练浅层变分量子电路以替代量子相位估计并估计分子基态能量。 |
| [^242] | [AlphaWiSE: Adaptive Weight Interpolation for Continual Multimodal Representation Learning](https://arxiv.org/abs/2607.15094) | AlphaWiSE通过为每个参数张量自适应拟合标量插值系数，将顺序微调端点与稳定性导向端点的权重进行事后融合，在保持单一模型架构和推理成本的同时，提升了持续音频-图像-文本检索的性能。 |
| [^243] | [SaturQ: Saturation-Aware Mixed-Precision Quantization](https://arxiv.org/abs/2607.12266) | SaturQ用凹形累积函数修正传统敏感度简单求和导致的饱和高估问题，通过单次Aumann–Shapley路径积分拟合参数并用线性规划分配精度，在Qwen3-30B-A3B上将KL散度较ModelOpt AutoQuantize降低11–23%。 |
| [^244] | [Auditing Reward Hackability in Code RL Training Environments](https://arxiv.org/abs/2606.16062) | 该论文通过基于标准补丁的执行验证系统审计了代码强化学习训练环境的奖励作弊问题，发现约22.4%（强攻击者下57.8%）的SWE-bench Verified任务会接受错误补丁，且模型在这些可作弊任务上的通过率显著更高，凸显了对RL训练环境进行作弊检测与修复审计的必要性。 |
| [^245] | [Crossing the Validation Crisis: Cross-Validation Reduces Benchmarking Variance Surprisingly Well](https://arxiv.org/abs/2606.12552) | 本文提出“样本增益”概念，证明采用多个交叉验证划分可通过虚拟数据增强效应显著降低基准测试方差，从而大幅提升性能评估的可靠性与稳定性，有效缓解机器学习领域的验证危机。 |
| [^246] | [Chain of Operators: An Inference-Time Harness for In-Context Operator Learning](https://arxiv.org/abs/2606.12318) | 本文提出算子链（CHOP）框架，利用冻结基础模型的上下文学习能力，将复杂分布外任务分解为闭式数学算子序列，实现无需参数更新的推理时适应。 |
| [^247] | [EssentialGIN: a new approach for gene essentiality prediction based on graph isomorphism neural networks](https://arxiv.org/abs/2606.07700) | 本研究提出EssentialGIN方法，通过改进图同构神经网络来保留PPI网络的拓扑特征，并整合基因表达、直系同源和亚细胞定位等生物信息，从而实现对必需基因的准确预测。 |
| [^248] | [Reactivity-Informed Machine Learning for Performance Prediction and Design Space Exploration of Alkali-Activated Slag](https://arxiv.org/abs/2606.06765) | 该研究构建了迄今最大的碱激发矿渣文献数据集（含3100余条抗压强度记录和155种化学成分各异的GGBS），并引入物理可解释的反应活性描述符（金属氧化物平均解离能）结合机器学习，实现了碱激发矿渣性能的准确预测与配合比设计空间的高效探索。 |
| [^249] | [Gradient-based optimization of nuclear criticality experiments using neural surrogate eigenvalue sensitivities](https://arxiv.org/abs/2606.04033) | 本文提出利用物理信息神经网络的可微性进行基于梯度的核临界实验几何优化，通过最大化与目标技术的相关系数 $c_k$ 来自动设计具有高中子学相似性的新临界实验，并能搜索栅格内材料组合的组合设计空间。 |
| [^250] | [Trust Region Q Adjoint Matching](https://arxiv.org/abs/2605.27079) | 本文提出信赖域Q伴随匹配（TRQAM），通过在流策略采样过程中引入信赖域参数λ来精确加权路径空间KL散度，并借助投影对偶下降自适应控制策略更新幅度，从而解决了QAM中评论家误差被指数级放大导致性能崩溃的问题，实现了对预训练流策略的稳定离策略微调。 |
| [^251] | [Growing a Neural Network in Breadth, Depth, and Time](https://arxiv.org/abs/2605.25174) | 该论文提出在循环卷积网络中为宽度、深度和时间定义可微分成本并与任务误差联合优化，发现三种资源在给定精度下可相互权衡，且模型的时间分配与人类反应时间相关。 |
| [^252] | [GENSTRAT: Toward a Science of Strategic Reasoning in Large Language Models](https://arxiv.org/abs/2605.23238) | 提出GENSTRAT系统，通过程序化生成双人零和不完全信息下注博弈按需创建全新评估游戏，解决了固定游戏套件饱和与数据污染问题，并通过50个游戏的基准和40,000多场模型对战的锦标赛揭示了不同LLM策略推理能力的显著差异。 |
| [^253] | [A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions](https://arxiv.org/abs/2605.18898) | 该论文提出用双参数Weibull分布作为统一参考框架来诊断Transformer权重幅值分布，发现跨七个模型系列的FFN权重形状参数高度稳定（约1.20），而查询/键投影则呈现与架构相关的系统性偏离。 |
| [^254] | [RefDecoder: Enhancing Visual Generation with Conditional Video Decoding](https://arxiv.org/abs/2605.15196) | 提出RefDecoder，一种通过参考注意力将高保真参考图像信号注入视频VAE解码过程的参考条件化解码器，有效解决了潜空间扩散模型中解码器无条件化导致的细节丢失和结构不一致问题。 |
| [^255] | [Fast Training of Mixture-of-Experts for Time Series Forecasting via Expert Loss Integration](https://arxiv.org/abs/2605.10330) | 提出一种融合专家特定损失与部分在线学习策略的自适应混合专家框架，解决了小门控权重导致的优化难题，在保持计算效率的同时显著提升了时间序列预测性能。 |
| [^256] | [Inpainting physics: self-supervised learning for context-driven fluid simulation](https://arxiv.org/abs/2605.08832) | 本文提出将稳态CFD推理重新定义为修复问题，通过自监督学习速度场先验并在推理时施加边界约束，结合局部邻域分词器实现大规模三维网格的高效流体模拟。 |
| [^257] | [NoiseRater: Meta-Learned Noise Valuation for Diffusion Model Training](https://arxiv.org/abs/2605.08144) | 提出NoiseRater，一种通过双层元学习对给定样本和时间步条件下的噪声实例进行评分的模型，用于选择更有信息量的训练噪声，并发现同一时间步的不同噪声实现并非同等有用。 |
| [^258] | [Mosaic: Efficient CPU-GPU Coordinated Block-Sparse Attention](https://arxiv.org/abs/2605.07719) | Mosaic提出了一种CPU-GPU协同的块稀疏注意力系统，通过输出贡献感知的KV预算分配、注意力头特定且粒度感知的稀疏配置以及跨设备协同调度，解决了长上下文LLM解码中CPU驻留KV缓存访问受PCIe带宽限制和GPU闲置的低效问题。 |
| [^259] | [When and Why SignSGD Outperforms SGD: A Theoretical Study Based on $\ell_1$-norm Lower Bounds](https://arxiv.org/abs/2605.06615) | 本文通过引入ℓ₁-范数平稳性、ℓ∞-平滑性和可分离噪声模型这一新问题几何框架，首次从理论上证明了SignSGD在何种条件下以及为何能够超越SGD，并给出了匹配的复杂度上下界。 |
| [^260] | [Free Decompression with Algebraic Spectral Curves](https://arxiv.org/abs/2605.03634) | 本文利用代数谱曲线理论，为Stieltjes变换满足代数关系的谱密度建立了更通用的自由解压缩方法，放宽了原有方法的强假设，使跨矩阵规模的谱信息外推能够应用于更现实的机器学习模型。 |
| [^261] | [A Scale-Adaptive Framework for Joint Spatiotemporal Super-Resolution with Diffusion Models](https://arxiv.org/abs/2604.21903) | 该论文提出一种尺度自适应的时空联合超分辨率框架，将任务分解为条件均值的确定性预测与残差条件扩散模型，使同一架构可复用于不同的空间和时间超分辨率因子，并可通过质量守恒变换保持降水总量一致。 |
| [^262] | [Multi-User mmWave Beam and Rate Adaptation via Combinatorial Satisficing Bandits](https://arxiv.org/abs/2604.14908) | 本文提出SAT-CTS轻量级策略，将多用户毫米波系统中的波束与速率联合自适应建模为满意化目标的组合半老虎机问题，并首次给出了此类问题的有限时间遗憾界理论保证。 |
| [^263] | [The Projector Forgets Too: Joint Merging for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2604.14016) | 该论文发现多模态持续指令微调中的灾难性遗忘不仅源于语言模型的LoRA适配器，还源于跨模态投影器，并提出JoMe方法通过联合合并这两部分的参数来有效缓解遗忘。 |
| [^264] | [Exploring Urban Land Use Patterns by Pattern Mining and Unsupervised Learning](https://arxiv.org/abs/2604.13050) | 该论文提出了一种结合频繁项集模式挖掘与Ward聚类的可重复分析框架，从欧洲100个城市地区的遥感数据中识别出七类跨城市重复出现的土地利用配置模式，为比较性规划中的同类城市对比提供了方法支持。 |
| [^265] | [Differentially Private Manifold Denoising](https://arxiv.org/abs/2604.00942) | 该论文提出一种差分隐私流形去噪框架，利用隐私参考数据校正含噪声的公共查询，通过噪声自适应软校正与共享高斯场，在无需指定内在维度的情况下实现隐私保护下的局部几何估计，并提供了理论保证。 |
| [^266] | [TopoPilot: Reliable Conversational Workflow Automation for Topological Data Analysis and Visualization](https://arxiv.org/abs/2603.25063) | 提出了TopoPilot——一个采用“编排器+验证器”双智能体架构的可靠智能体框架，用于自动化拓扑数据分析与可视化等复杂科学可视化工作流。 |
| [^267] | [Does This Gradient Spark Joy?](https://arxiv.org/abs/2603.20526) | 论文提出愉悦策略梯度（DG），在前向传播中用优势与惊讶度的乘积“愉悦度”评估样本的学习价值，并通过“近藤门控”只为值得的样本执行昂贵的反向传播，从而以极少的计算实现更快的学习。 |
| [^268] | [Delightful Policy Gradient under Distributed Friction](https://arxiv.org/abs/2603.20521) | 本文提出愉悦策略梯度（DG），通过用优势与惊讶度的乘积作为门控信号，在不依赖行为概率的情况下抑制分布式强化学习中罕见失败带来的噪声，同时保留罕见成功所携带的关键信息。 |
| [^269] | [GIST: Gauge-Invariant Spectral Transformers for Scalable Graph Neural Operators](https://arxiv.org/abs/2603.16849) | GIST提出了一种可扩展的规范不变谱Transformer，通过仅对高效近似谱嵌入的两两内积施加注意力，在O(N)复杂度下严格保证规范不变性，实现跨网格分辨率与跨图结构的有效迁移。 |
| [^270] | [Power-Law Spectrum of the Random Feature Model](https://arxiv.org/abs/2603.14578) | 该论文刻画了随机特征模型总体协方差的谱结构，证明了数据协方差的幂律谱在经过随机线性投影与非线性激活后仍以幂律形式被保留，从而为神经网络缩放定律背后的谱机制提供了理论解释。 |
| [^271] | [Impact of Markov Decision Process Design on Sim-to-Real Reinforcement Learning](https://arxiv.org/abs/2603.09427) | 该论文系统研究了MDP核心设计选择（状态构成、奖励、终止条件、动力学模型）对强化学习模拟到现实迁移的影响，并通过颜色混合任务证明基于物理的动力学模型在严格精度约束下可实现50%的真实成功率。 |
| [^272] | [Surprisal-R\'enyi Free Energy](https://arxiv.org/abs/2603.03405) | 本文提出惊异-雷尼自由能（SRFE）这一似然比的归一化雷尼-切尔诺夫对数矩泛函，证明其在端点极限下恢复前向与反向KL散度，并通过局部展开揭示了对数似然比方差作为一阶修正所体现的均值-方差权衡机制。 |
| [^273] | [Beyond Arrow: From Impossibility to Possibility in Multi-Criteria Benchmarking](https://arxiv.org/abs/2602.07593) | 该论文突破了阿罗不可能定理对多准则基准排序的限制，识别出单峰、群可分和距离受限三类偏好结构，在这些条件下多指标基准测试可以产生稳定且有意义的大模型排名。 |
| [^274] | [Learning to Allocate Resources with Censored Feedback](https://arxiv.org/abs/2602.06565) | 该论文研究了删失反馈下的在线预算分配问题，证明了 $\Omega(\sqrt{T})$ 的最坏情况遗憾下界，并提出了一种结合非平凡参数估计与置信界的乐观算法 RA-UCB，用于联合估计未知参数并最大化累积奖励。 |
| [^275] | [Convergence Rate of the Last Iterate of Stochastic Proximal Algorithms](https://arxiv.org/abs/2602.05489) | 本文在放松有界方差假设的条件下，证明了随机梯度邻近方法与随机增量邻近方法的最后迭代点在逐分量凸性与光滑性下达到对数因子意义下最优的 $\widetilde{O}(1/\sqrt{T})$ 收敛速率，并可直接应用于多任务与联邦学习中的图引导正则项。 |
| [^276] | [EMA Policy Gradient: Taming Reinforcement Learning for LLMs with EMA Anchor and Top-k KL](https://arxiv.org/abs/2602.04417) | 提出EMA锚点策略和Top-k KL估计器两项简单技术来改进大语言模型的策略梯度算法，结合GRPO后在数学推理和智能体强化学习任务上取得显著性能提升。 |
| [^277] | [Efficient Subgroup Analysis via Optimal Trees with Global Parameter Fusion](https://arxiv.org/abs/2602.04077) | 提出一种基于混合整数优化的融合最优因果树方法，通过全局最优划分与参数融合约束，克服了传统树方法贪婪分裂导致的次优划分和过拟合问题，实现精确的差异性治疗效应亚组识别。 |
| [^278] | [Geometry-Preserving Neural Architectures on Manifolds with Boundary](https://arxiv.org/abs/2602.03082) | 本文系统研究了几类保持几何约束的神经架构的逼近论性质，证明了一致近点正则集上投影神经ODE在含边界紧致光滑流形上的稳定性估计与可测流形值映射的 $L^2$ 万能逼近定理，并揭示了中间层约束与仅输出端约束两类架构在逼近保证上的差异。 |
| [^279] | [TabularMath: Evaluating Computational Extrapolation in Tabular Learning via Program-Verified Synthesis](https://arxiv.org/abs/2602.02523) | 本文提出TabularMath基准，将GSM8K和AIME问题编译为114个经程序验证的确定性计算，用以评估表格模型从统计插值到计算外推的能力，发现TabPFN v2.5在分布内中位数R²高达0.999，且是唯一在超出训练范围时中位数R²仍保持正值（0.76）的表格方法。 |
| [^280] | [SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2602.01990) | 本文提出SAME框架，通过稳定专家路由和防止专家覆盖，解决了多模态持续指令微调中的路由漂移和专家漂移问题。 |
| [^281] | [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734) | 本文提出同策略自蒸馏（OPSD）学习算法，让单个LLM在不同上下文中同时充当教师与学生——教师以特权信息（如经过验证的推理轨迹）为条件对学生进行密集监督，从而无需独立的教师模型，并显式利用了推理数据集中的标准答案。 |
| [^282] | [Efficient Inference for Inverse Reinforcement Learning and Dynamic Discrete Choice Models](https://arxiv.org/abs/2512.24407) | 该论文提出了一种半参数去偏框架，通过将对数行为策略视为伪奖励，实现了在灵活奖励表示下对逆强化学习和动态离散选择模型的有效统计推断。 |
| [^283] | [gHAWK: Structural Encoding for Scalable Training of Graph Neural Networks on Knowledge Graphs](https://arxiv.org/abs/2512.08274) | gHAWK通过在训练前预计算布隆过滤器（编码局部邻域结构）和TransE嵌入（表示全局位置）作为节点的结构特征并与领域特征融合，提出了一种可扩展的大规模知识图谱GNN训练框架，克服了传统消息传递式GNN在大规模图上的扩展性瓶颈。 |
| [^284] | [Object-Centric World Models for Causality-Aware Reinforcement Learning](https://arxiv.org/abs/2511.14262) | 该论文提出STICA统一框架，利用以对象为中心的Transformer作为世界模型，并配合因果感知的策略与价值网络，使强化学习智能体能够像人类一样分解复杂多对象环境，从而实现更高效的样本学习与决策。 |
| [^285] | [Generalizing Test-time Compute-optimal Scaling as an Optimizable Graph](https://arxiv.org/abs/2511.00086) | 该论文提出将测试时扩展中在固定预算下搜索计算最优模型组合与架构的新问题形式化为多LLM协作图，并将其重新表述为概率图优化问题，以应对庞大的组合搜索空间和任务定制化需求。 |
| [^286] | [Revisiting Value Iteration: Unified Analysis of Discounted and Average-Reward Cases](https://arxiv.org/abs/2510.23914) | 本文通过统一的几何分析证明，在最优策略唯一且为单链的假设下，价值迭代在折扣奖励和平均奖励两种情形下均具有几何收敛性，且收敛速率快于以往理论分析所给出的结果。 |
| [^287] | [Provable Selective Auto-labeling with Reliability Guarantees](https://arxiv.org/abs/2510.14581) | 提出了“共形标注”这一新方法，将选择性自动标注形式化为多重假设检验问题，通过构建共形p值并结合Benjamini-Hochberg程序，在错误标注率可证明受控的前提下筛选出可信的自动生成标签子集。 |
| [^288] | [Learning to Assimilate: Amortized Control for Autoregressive Diffusion](https://arxiv.org/abs/2510.06637) | 提出了 CADA 方法，在冻结的自回归扩散模型之上训练轻量级控制器实现摊销式数据同化，无需逐次预测进行优化即可提升稀疏观测下的长程预测精度，并将推理速度提高 10 倍以上。 |
| [^289] | [Fractal basin geometry and the limits to predictability and reproducibility in deep learning](https://arxiv.org/abs/2510.05606) | 深度学习训练结果难以预测和复现的根本原因在于损失景观中分形且相互穿透的吸引盆几何结构——即使大幅提高初始化精度，结果的可预测性也几乎不会改善。 |
| [^290] | [Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards](https://arxiv.org/abs/2510.04392) | 本文提出PS-GRPO强化学习方法，通过跨复述集的组相似度奖励来提升RAG系统在语义等价查询下的信息一致性，并提供了将一致性分解为检索器、生成器和端到端层面的系统性评估框架。 |
| [^291] | [Reconstruction Alignment Improves Unified Multimodal Models](https://arxiv.org/abs/2509.07295) | 提出一种名为RECA的后训练方法，利用视觉理解编码器嵌入作为密集监督信号，无需文本标题即可重新对齐统一多模态模型的理解与生成，从而显著提升图像生成和编辑的保真度。 |
| [^292] | [Beyond Output Faithfulness: Diagnosing and Mitigating Pathway Rerouting in Feature Attribution](https://arxiv.org/abs/2509.04588) | 论文发现插入/删除等仅输出指标无法检测特征归因是否经由支持原始预测的计算路径，提出了可微的EQO方法揭示“通路重路由”现象，并引入逐层梯度过滤器ClipGrad来缓解该问题。 |
| [^293] | [VASSO: Variance Suppression for Sharpness-Aware Minimization](https://arxiv.org/abs/2509.02433) | 提出了一种通过方差抑制来稳定对抗扰动的通用方法 VASSO，可证明地解决 SAM 中“过度友好的对抗者”问题，并在视觉和语言任务上显著提升泛化能力。 |
| [^294] | [SVarM: Linear Support Varifold Machines for Classification and Regression on Geometric Data](https://arxiv.org/abs/2506.01189) | 该论文提出SVarM方法，借助形状的变褶皱测度表示及其对偶性，构建了一个在无穷维变褶皱空间上运算、类似线性支持向量机的通用框架，用于对几何形状数据进行分类与回归。 |
| [^295] | [Fine-Tuning Next-Scale Visual Autoregressive Models with Group Relative Policy Optimization](https://arxiv.org/abs/2505.23331) | 本文将群体相对策略优化（GRPO）应用于下一尺度视觉自回归（VAR）模型的强化学习微调，通过对齐美学与CLIP奖励信号显著提升图像质量与风格控制能力，并使模型能够泛化生成预训练阶段未见过的图像风格。 |
| [^296] | [Causality can systematically address the monsters under the bench(marks)](https://arxiv.org/abs/2502.05085) | 这篇立场论文主张，因果性为系统性地解决机器学习基准测试中的偏差、伪影和数据泄露等“怪兽”问题提供了理想框架，通过显式化因果假设并利用有原则的分析工具来提升评估的可信度与一致性。 |
| [^297] | [Deep Learning-based Approaches for State Space Models: A Selective Review](https://arxiv.org/abs/2412.11211) | 本文对基于深度学习的状态空间模型方法进行了选择性综述，为离散时间与连续时间（如潜在神经ODE和SDE）的深度状态空间模型提供了统一的理论视角。 |
| [^298] | [The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning](https://arxiv.org/abs/2412.00568) | 该论文推出了The Well——一个包含16个数据集、总计15TB数据的大规模多样化物理模拟数据集合集，涵盖生物系统、流体动力学、磁流体动力学等多个领域，并提供统一的PyTorch接口以支持机器学习模型的训练与评估。 |
| [^299] | [A context-adaptive policy framework for robust and reactive robotic manipulation via uncertainty-aware imitation learning](https://arxiv.org/abs/2410.24035) | 该论文提出了一个基于不确定性感知模仿学习的上下文自适应策略框架，通过策略融合与不确定性量化技术，首次将任务参数化、鲁棒性和响应式机器人操作三者结合，解决了现有动力系统方法无法根据任务相关变量灵活调节策略的问题。 |
| [^300] | [Average Causal Effect Estimation in DAGs with Hidden Variables: Beyond Back-Door and Front-Door Criteria](https://arxiv.org/abs/2409.03962) | 该论文针对含隐变量DAG中超越后门与前门准则的可识别因果效应，提出了基于序贯回归和贝叶斯法则重参数化的估计方法，从而避免了条件密度估计与数值积分带来的统计和计算难题。 |
| [^301] | [Automatic Detection of LLM-Generated Code: A Comparative Case Study of Contemporary Models Across Function and Class Granularities](https://arxiv.org/abs/2409.01382) | 该研究通过对比四种LLM生成的14,485个函数和11,913个类，发现代码检测中粒度效应的影响远超模型间差异（达8.6倍），函数级与类级检测依赖截然不同的结构特征，且存在依赖粒度的检测性能反转现象。 |
| [^302] | [On the Global Convergence of Gradient Descent for multi-layer ResNets in the mean-field regime](https://arxiv.org/abs/2110.02926) | 该论文证明了在平均场机制下，当激活函数满足2-齐次或部分1-齐次条件时，多层ResNet训练所对应的梯度流偏微分方程能收敛到全局最小值，表明足够大的ResNet可通过一阶优化方法找到拟合训练数据的全局最优解。 |
| [^303] | [On the interplay between noise and curvature and its effect on optimization and generalization](https://arxiv.org/abs/1906.07774) | 该论文探讨了损失曲率与梯度噪声的相互作用如何影响优化速度以及泛化间隙的估计，并澄清了费舍尔矩阵、海森矩阵和梯度协方差矩阵之间的区别。 |
| [^304] | [OpinionGPT: Modelling Explicit Biases in Instruction-Tuned LLMs.](http://arxiv.org/abs/2309.03876) | OpinionGPT是一个指令调整大型语言模型(LLMs)的web演示，用户可以选择各种偏见并进行比较，它意在使模型的偏见显性和透明化。 |

# 详细

[^1]: 无需方差假设的最优线性回归

    Optimal Linear Regression Without a Variance

    [https://arxiv.org/abs/2611.27983](https://arxiv.org/abs/2611.27983)

    该论文提出一种基于经验残差剪枝的黑盒归约方法，在噪声仅有未知有限α矩（α∈(1,2)）且协变量仅有有限四阶矩的无穷方差情形下，可将任意有限方差回归算法自动转化为达到最优统计速率的弱矩回归算法，且无需知道矩指数、噪声尺度等任何先验信息。

    

    我们研究了无穷方差情形下的高概率线性回归问题，其中回归噪声仅对某个未知的 α∈(1,2) 具有有限的条件 α-矩 ν，而协变量可能只具有直至给定阶数的有限矩（例如4阶矩）。我们提出了一种黑盒归约方法，能够将有限方差的回归程序转换为弱矩条件下的回归程序。该归约基于经验残差剪枝，并且对矩指数、噪声尺度、协方差以及设计矩都是不可知的（即无需预先知道这些量）。在协变量满足有限四阶矩的条件下，任何达到亚高斯速率的有限方差子程序，当样本数 n 超过某个适当定义的阈值时，均可得到如下结果：在不知道 α 或 ν 的情况下，我们以至少 1−δ 的概率有 ‖θ̂−θ*‖_Σ ≲_α ν^{1/α} ((d+log(1/δ))/n)^{…

    arXiv:2611.27983v1 Announce Type: cross  Abstract: We study high-probability linear regression in the infinite variance regime, wherecthe regression noise has only a finite conditional $\alpha$-moment $\nu$ for some unknown $\alpha\in(1,2)$, and the covariates may only have finite moments up to a given order (e.g., 4th moment). We give a black-box reduction that converts finite-variance regression procedures into weak-moment procedures. The reduction is based on empirical residual pruning and is agnostic to the moment exponent, the noise scale, the covariance, and the design moment. Under a finite fourth-moment condition on the covariates, any finite-variance subroutine attaining the sub-Gaussian rate yields the following when the number of samples $n$ is above a suitably defined threshold: Without knowing $\alpha$ or $\nu$, we have with probability at least $1-\delta$ that $ \|\widehat\theta-\theta^*\|_\Sigma   \lesssim_\alpha   \nu^{1/\alpha}   \big(\frac{d+\log(1/\delta)}{n}\big)^{\
    
[^2]: 一种用于标普500预测的线性回归、RNN与LSTM的方法学严谨基准测试：识别金融深度学习中性能虚高的来源

    A Methodologically Rigorous Benchmark of Linear Regression,RNN, and LSTM for S&P 500 Forecasting:Identifying Sources of Performance Inflation in Financial DeepLearning

    [https://arxiv.org/abs/2611.27978](https://arxiv.org/abs/2611.27978)

    本文通过方法学审计识别出金融深度学习股票预测文献中导致性能虚高的三种来源（小波去噪评估目标、缺失朴素持续性基线、最优N次重复报告），并在修正后的严格协议下对标普500预测的线性回归、RNN和LSTM进行了统一基准测试。

    

    本研究使用2006至2025年的数据集（包含基础价格数据、宏观经济指标以及工程化技术特征）对标普500收盘价进行次日预测。我们在统一的预处理和时间顺序评估流程下，对三个模型族进行了基准测试，即线性回归、原生循环神经网络（RNN）和长短期记忆网络（LSTM）。本工作的核心贡献是对股票预测深度学习文献中常见做法进行的一次方法学审计。我们识别出已发表方法中频繁出现的三种结果虚高来源：使用小波去噪的评估目标、缺失朴素持续性基线，以及最优N次重复报告。随后，我们在针对上述每一项进行控制的修正协议下重新评估了所有模型族。性能采用均方根误差、平均绝对百分比误差以及……进行报告（摘要在此处被截断）。

    arXiv:2611.27978v1 Announce Type: new  Abstract: This study examines one-day-ahead forecasting of the S&P 500 closing level using a 2006-2025 dataset that includes fundamental price data, macroeconomic indicators, and engineered technical features. We benchmark three model families, namely linear regression, vanilla recurrent neural networks (RNN), and long short-term memory networks (LSTM), within a unified preprocessing and chronological evaluation pipeline. A central contribution of this work is a methodological audit of common practices in the deep-learning-for-stock-prediction literature. We identify three sources of result inflation that frequently appear in published methodologies: wavelet-denoised evaluation targets, the absence of a naive persistence baseline, and best-of-N replicate reporting. We then re-evaluate all model families under a corrected protocol that controls for each. Performance is reported using root mean squared error, mean absolute percentage error, and the 
    
[^3]: 基于脑电信号的儿童与青少年ADHD分类完整特征集研究

    On the Complete Feature Set for ADHD Classification in Children and Adolescents Based on EEG Signals

    [https://arxiv.org/abs/2611.27970](https://arxiv.org/abs/2611.27970)

    该论文整理了一个包含156名参与者、669条记录的标准化脑电数据集，将连续信号与115种特征定义的预计算测量值相链接并记录质量决策，为儿童与青少年ADHD分类中七个脑电特征族的公平比较提供了统一基准。

    

    注意缺陷多动障碍（ADHD）是一种可能影响注意力、活动水平和冲动控制的神经发育性疾病。脑电图（EEG）可以在儿童和青少年执行认知任务时记录其大脑活动。然而，比较从这些记录中提取的特征需要一致的信号预处理、共享的分析时间窗口以及有据可查的质量决策。我们整理了来自156名参与者、涵盖五种任务条件的669条现有BALLADEER记录，使研究人员能够在相同的时间区间上，针对每位参与者和每项任务比较七个脑电特征族。该数据集将连续信号与来自115个特征定义或模板的预先计算测量值相关联，并记录了针对每条记录所做的质量决策。分析仍需考虑参与者内部的重复测量、残余伪迹以及不同采集系统之间的差异。技术验证涵盖了信号完整性（原文在此处截断）。

    arXiv:2611.27970v1 Announce Type: new  Abstract: Attention-deficit/hyperactivity disorder is a neurodevelopmental condition that can affect attention, activity and impulse control. Electroencephalography (EEG) records brain activity as children and adolescents perform cognitive tasks. However, comparing features derived from these recordings requires consistent signal preparation, shared analysis windows and documented quality decisions. We curated 669 existing BALLADEER recordings from 156 participants across five task conditions so that researchers can compare seven EEG feature families on the same time intervals for each participant and task. The dataset links continuous signals to precomputed measurements from 115 feature definitions or templates and records the quality decisions made for each recording. Analyses must still account for repeated measurements within participants, residual artefacts and differences between acquisition systems. Technical validation covered signal integ
    
[^4]: 模型雕刻：训练能够优化自身拓扑结构的 Transformer

    Model Carving: Training Transformers That Optimize Their Own Topology

    [https://arxiv.org/abs/2611.27962](https://arxiv.org/abs/2611.27962)

    提出模型雕刻方法，通过在每个子层上添加可学习的标量门控，让 Transformer 在单次训练中同时学习权重与架构，由训练损失自动移除冗余组件，释放出内部更小、更高效的异构网络。

    

    机器学习在很大程度上将架构选择与训练过程相互分离。例如，经典的 Transformer 在每一层以相同的顺序训练一组固定的注意力子层和 MLP 子层，仅优化其中的权重。这种统一性是一种惯例而非有原则的设计选择，并且很可能是次优的：针对给定任务的最佳架构可能更小、更高效，且远非均匀一致。我们认为，理想情况下，训练本身应该塑造模型的拓扑结构。为此，我们提出了模型雕刻，它在单次运行中同时学习 Transformer 的权重与架构。通过在每个子层上应用可学习的标量门控，我们让训练损失能够在训练过程中移除模型自身的组件——从而释放出模型内部一个更小、异构的网络。我们通过对参数量从 1.4 亿到 27 亿的标准 Transformer 以及循环 Transformer 进行预训练来评估模型雕刻

    arXiv:2611.27962v1 Announce Type: cross  Abstract: Machine learning has largely kept separate the choice of architecture and the process of training. A classic Transformer, for instance, trains a fixed set of attention and MLP sublayers at every layer, in the same order, and optimizes the weights within. This uniformity is a convention rather than a principled design choice, and it is likely suboptimal: the best architecture for a given task may be smaller, more efficient, and far from uniform. We believe that, ideally, training itself should shape the model's topology. To this end, we introduce Model Carving, which learns both a Transformer's weights and its architecture in a single run. By applying learnable, scalar gates to each sublayer, we allow the training loss to remove the model's own components during training -- freeing a smaller heterogeneous network inside. We evaluate Model Carving by pre-training standard Transformers from 140M to 2.7B parameters and looped Transformers 
    
[^5]: 用于高效进化超参数优化的保排序核心集选择

    Ranking-Preserving Coreset Selection for Efficient Evolutionary Hyperparameter Optimization

    [https://arxiv.org/abs/2611.27959](https://arxiv.org/abs/2611.27959)

    本文提出保排序核心集（RPC）选择框架，通过进化优化最大化核心集与完整数据上配置排名的Spearman秩相关，从而在保留超参数配置相对排序的前提下加速GA、CHC和PSO三种进化算法的超参数优化。

    

    超参数优化（HPO）对于实现高预测性能至关重要，但当候选配置需要在大型数据集上进行反复训练时，其计算成本会变得极其高昂。本文提出了一种保排序核心集（RPC）选择框架，用于加速进化超参数优化。该方法构建一个紧凑的核心集，使其能够保留在完整训练数据上观察到的超参数配置的相对排序。核心集选择被表述为一个进化优化问题，其目标是在惩罚子集规模的同时，最大化基于完整数据与基于子集的配置排名之间的Spearman秩相关系数。随后，所选择的核心集被用于三种进化算法的超参数优化：遗传算法（GA）、跨代精英选择-异质重组-灾变变异算法（CHC）以及粒子群优化算法（PSO）。实验在七个表格数据集、两个文本数据集和两个图像数据集上进行。

    arXiv:2611.27959v1 Announce Type: new  Abstract: Hyperparameter optimization (HPO) is essential for high predictive performance, but its computational cost becomes prohibitive when candidate configurations require repeated training on large datasets. This paper proposes a ranking-preserving coreset (RPC) selection framework for accelerating evolutionary HPO. The method constructs a compact coreset that preserves the relative ranking of hyperparameter configurations observed on the full training data. Coreset selection is formulated as an evolutionary optimization problem that maximizes Spearman rank correlation between full-data and subset-based configuration rankings while penalizing subset size. The selected coreset is then used for HPO with three evolutionary algorithms: Genetic Algorithm (GA), Cross-generational elitist Selection, Heterogeneous Recombination and Cataclysmic Mutation (CHC), and Particle Swarm Optimization (PSO). Experiments on seven tabular, two text, and two image 
    
[^6]: 拓扑枢纽最优传输：以闪速消除长上下文稀释

    Topology Hub Optimal Transport: Eliminating Long Context Dilution at Flash Speed

    [https://arxiv.org/abs/2611.27951](https://arxiv.org/abs/2611.27951)

    THOT 提出一种基于轮辐式拓扑的最优传输注意力机制，通过有界支撑容量门控、退火行封闭 Sinkhorn 调度和高效 Triton 内核伴随计算，以接近 FlashAttention 的速度消除长上下文中的注意力质量稀释问题。

    

    长上下文模型通常采用稀疏或结构化注意力机制来避免二次方复杂度的开销，然而当归一化质量随序列长度增加而分散时，从数十万条干扰信息中检索出少量相关事实的能力会随之退化。轮辐式最优传输（OT）用由局部窗口和 K 个共享枢纽构成的线性规模图替代稠密的 L×L 矩阵，并通过 Sinkhorn 缩放实现平衡。在分块尾部细化与固定支撑传输拓扑（arXiv:2605.08123, arXiv:2609.07954）的基础上，我们提出了拓扑枢纽最优传输（THOT）：（1）通过单纯形 sparsemax（O(L log L)）或无需排序的词元局部阈值化 ReLU（O(L)）实现有界支撑的容量门控，将多余的干扰质量驱动至零；（2）一种三遍扫掠的 1+2 退火行封闭 Sinkhorn 调度方案，用于抑制二部图残差；（3）在批量分裂因子 Triton 内核（BF16 Tensor Core，FP32 po……原文摘要在此处被截断）

    arXiv:2611.27951v1 Announce Type: new  Abstract: Long-context models use sparse or structured attention to avoid quadratic cost, yet retrieving a small set of relevant facts embedded among hundreds of thousands of distractors degrades as normalization mass disperses with length. Hub-and-spoke optimal transport (OT) replaces the dense $L \times L$ matrix with a linear-size graph over local windows and $K$ shared hubs balanced by Sinkhorn scaling. Building on block-wise tail refinement and fixed-support transport topology (arXiv:2605.08123, arXiv:2609.07954), we introduce Topology Hub Optimal Transport (THOT): (1) bounded-support capacity gating via simplex sparsemax ($O(L\log L)$) or sorting-free token-local thresholded ReLU ($O(L)$), driving excess distractor mass to zero; (2) a three-sweep $1+2$ annealed row-closed Sinkhorn schedule that damps bipartite residuals; (3) an exact rank-$(D+2R+1)$ One-Reference-Tile adjoint in batched split-factor Triton kernels (BF16 Tensor Cores, FP32 po
    
[^7]: 部分概念类不可知PAC学习的最优样本复杂度

    The Optimal Sample Complexity of Agnostic PAC Learning for Partial Concept Classes

    [https://arxiv.org/abs/2611.27946](https://arxiv.org/abs/2611.27946)

    该论文为VC维为d的部分概念类构造了一个确定性不可知PAC学习器，仅用 $O((d+\log(1/\delta))/\varepsilon^2)$ 个样本即可实现学习，样本复杂度与Alon等人（FOCS 2021）的下界匹配，达到最优。

    

    设 $\mathcal H\subseteq\{-1,+1,\star\}^{\mathcal X}$ 为具有有限VC维数 $d$ 的部分概念类。我们构造了一个确定性的不可知PAC学习器，对于任意 $\varepsilon,\delta\in(0,1)$，它使用从 $\mathcal X\times\{-1,+1\}$ 上任意分布 $\mathcal D$ 中抽取的 $O((d+\log(1/\delta))/\varepsilon^2)$ 个独立同分布样本，并返回一个全定义概念 $\hat h:\mathcal X\to\{-1,+1\}$，使得 $\text{er}_{\mathcal D}(\hat h)\leq \min_{h\in\mathcal H}\text{er}_{\mathcal D}(h)+\varepsilon$ 以至少 $1-\delta$ 的概率成立，其中 $\text{er}_{\mathcal D}(h)=\mathbb P_{(x,y)\sim\mathcal D}[h(x)\neq y]$。这一结果确立了最优样本复杂度，与Alon等人（FOCS 2021）的下界相匹配。

    arXiv:2611.27946v1 Announce Type: new  Abstract: Let $\mathcal H\subseteq\{-1,+1,\star\}^{\mathcal X}$ be a partial concept class of finite VC dimension $d$. We construct a deterministic agnostic PAC learner that, for every $\varepsilon,\delta\in(0,1)$, uses $O((d+\log(1/\delta))/\varepsilon^2)$ i.i.d. examples from an arbitrary distribution $\mathcal D$ over $\mathcal X\times\{-1,+1\}$ and returns a total concept $\hat h:\mathcal X\to\{-1,+1\}$ satisfying $\text{er}_{\mathcal D}(\hat h)\leq \min_{h\in\mathcal H}\text{er}_{\mathcal D}(h)+\varepsilon$ with probability at least $1-\delta$, where $\text{er}_{\mathcal D}(h)=\mathbb P_{(x,y)\sim\mathcal D}[h(x)\neq y]$. This establishes the optimal sample complexity, matching the lower bound of Alon et al. (FOCS 2021).
    
[^8]: SUAF：用于零样本策略改进的谱通用优势特征

    SUAF: Spectral Universal Advantage Features for Zero-Shot Policy Improvement

    [https://arxiv.org/abs/2611.27919](https://arxiv.org/abs/2611.27919)

    本文提出谱通用优势特征（SUAF），通过对以动作为中心的典型依赖矩阵进行奇异值分解，构建可通过线性奖励投影恢复延迟优势的成对特征，从而在奖励无关表征下实现零样本策略改进。

    

    奖励无关的表征使强化学习智能体能够跨任务复用知识，但有限的表征容量引发了三个基本问题：这些表征应当捕捉何种统计依赖关系，应保留该依赖中的哪些成分用于策略改进，以及丢弃其余成分会对决策产生怎样的影响？我们通过典型依赖分析来回答这些问题。针对参考策略，我们构建了以动作为中心、正滞后的典型依赖矩阵（CDM），该矩阵分离出当前动作在当前状态之外的额外预测贡献。对其实施奇异值分解即可得到谱通用优势特征（SUAF），这是一对成对特征，通过线性奖励投影即可恢复延迟优势。我们证明该谱量化了当前动作与未来状态-动作访问分布之间的互信息……

    arXiv:2611.27919v1 Announce Type: cross  Abstract: Reward-independent representations enable reinforcement learning agents to reuse knowledge across tasks, but limited representation capacity raises three fundamental questions: what statistical dependence should these representations capture, which components of that dependence should be retained for policy improvement, and how does discarding the remaining components affect decision-making? We address these questions through canonical dependence analysis. For a reference policy, we construct an action-centered, positive-lag canonical dependence matrix (CDM) that isolates the additional predictive contribution of the current action beyond the current state. Its singular value decomposition yields Spectral Universal Advantage Features (SUAF), paired features that recover delayed advantage through linear reward projections. We show that the spectrum quantifies mutual information between current action and future state-action visitation c
    
[^9]: 多机构电子健康记录中潜在患者表征的通信高效联邦学习

    Communication-Efficient Federated Learning of Latent Patient Representations from Multi-Institutional EHRs

    [https://arxiv.org/abs/2611.27900](https://arxiv.org/abs/2611.27900)

    该论文提出一种通信高效的非迭代联邦谱方法，各机构只需共享边际频率和低秩谱摘要即可从多机构电子健康记录中协同学习共享的潜在患者表征，同时将患者级数据保留在本地以保护隐私。

    

    电子健康记录（EHR）包含丰富的健康信息，可用于揭示潜在的患者表征和具有临床意义的患者亚组。跨机构学习EHR数据非常有价值，因为在单一机构学习的表征可能反映本地数据模式且缺乏泛化能力，然而由于隐私限制和通信成本，跨机构汇集患者级记录通常难以实现。我们提出了一种通信高效、非迭代的联邦谱方法，用于从多机构EHR中学习共享的潜在表征组件。在该方法中，每个机构仅共享聚合的边际频率和低秩谱摘要，而将患者级记录保留在本地。中央服务器利用汇集的边际频率聚合这些摘要，从而恢复共享的低秩结构以及可解释表征所需的规范行归一化方向。

    arXiv:2611.27900v1 Announce Type: cross  Abstract: Electronic health records (EHRs) contain rich health information for uncovering latent patient representations and clinically meaningful subgroups. Learning from EHR data across institutions is valuable because representations learned at a single site may reflect local data patterns and lack generalizability, yet pooling patient-level records across institutions is often impractical because of privacy constraints and communication costs. We propose a communication-efficient, non-iterative federated spectral method for learning shared latent representation components from multi-institutional EHRs, in which each institution shares aggregate marginal frequencies and a low-rank spectral summary while keeping patient-level records local. A central server aggregates these summaries with the pooled marginal frequencies to recover both the shared low-rank structure and the canonical row-normalization direction required for interpretable repres
    
[^10]: 双层情感框架：网络搜索中可扩展的多模态用户体验归因

    The Dual-Tier Sentiment Framework: Scalable Multimodal UX Attribution in Web Search

    [https://arxiv.org/abs/2611.27899](https://arxiv.org/abs/2611.27899)

    该论文提出了一种双层情感框架，第一层利用微调的编码器-解码器大语言模型实现大规模用户情感回归检测（PR-AUC提升9.8%、测量延迟从数周降至数小时、问卷开销降低40%以上），第二层提供细粒度的UX归因诊断，从而实现网络搜索中可扩展的自动化用户体验评估。

    

    在大规模网络搜索中维持用户体验（UX）质量需要可扩展的评估方法。传统问卷调查速度缓慢，成为快速A/B测试的瓶颈，而自动化指标往往缺乏诊断视觉空间UX异常所需的粒度。为此，我们提出了双层情感框架（Dual-Tier Sentiment Framework），这是一种用于自动化UX归因的新型工业架构。第一层（宏观系统）作为可扩展的用户情感回归检测器，在遥测特征上利用微调的编码器-解码器大语言模型（基于Gemini XS）。通过多任务正则化和上下文特征工程，该层在超过50万条留存记录上实现了+9.8%的PR-AUC增益，在所有已验证的回归中与基线召回率持平，并在12个留存实验中实现了约20%的相对精确率提升。在生产环境中，该系统将测量延迟从数周缩短至数小时，并将问卷开销降低超过40%。第二层（微观系统）提供细粒度的诊断分析。

    arXiv:2611.27899v1 Announce Type: new  Abstract: Sustaining user experience (UX) quality in large-scale web search requires scalable evaluation. Traditional surveys are slow and bottleneck rapid A/B testing, while automated metrics often lack the granularity to diagnose visual-spatial UX anomalies. To address this, we introduce the Dual-Tier Sentiment Framework, a novel industrial architecture for automated UX attribution. Tier 1 (Macro System) acts as a scalable user sentiment regression detector, utilizing a fine-tuned encoder-decoder LLM (based on Gemini XS) on telemetry features. Through multi-task regularization and contextual feature engineering, this tier achieves a +9.8% PR-AUC gain on over 500,000 held-out records, matching baseline recall on all validated regressions with $\sim$20% relative precision gain among the 12 held-out experiments. In production, it reduces measurement latency from weeks to hours and cuts survey overhead by over 40%. Tier 2 (Micro System) provides gra
    
[^11]: FairITR：公平感知的多类别个体化治疗规则学习

    FairITR: Fairness-Aware Learning of Multicategory Individualized Treatment Rules

    [https://arxiv.org/abs/2611.27894](https://arxiv.org/abs/2611.27894)

    该论文提出了FairITR，首次针对多类别治疗场景引入公平性概念——要求治疗决策独立于敏感属性，并构建了相应的公平个体化治疗规则估计问题及其求解方法。

    

    个体化治疗规则（ITRs）基于个体特征推荐治疗决策，能够超越传统的“一刀切”策略改善治疗效果。然而，与其他常用的算法决策方法类似，未考虑模型公平性的ITRs可能会在由种族或性别等敏感属性定义的不同群体之间产生不公平的决策。尽管目前已发展出大量用于ITR估计的方法，但大多数仅专注于优化治疗效果，而关于公平感知ITR的研究仍然有限。此外，现有公平ITR研究很少考虑多类别治疗这一实际场景。为弥补这些空白，我们为多类别ITR引入了一种公平性概念，要求治疗决策独立于敏感属性，并据此构建了相应的公平ITR估计问题。为解决该问题，我们进一步（摘要在此处截断）

    arXiv:2611.27894v1 Announce Type: new  Abstract: Individualized treatment rules (ITRs) recommend treatment decisions based on individual characteristics and can improve treatment outcomes beyond traditional one-size-fits-all strategies. However, similar to other commonly used algorithmic decision-making methods, ITRs without considering model fairness may result in unfair decisions across groups defined by sensitive attributes, such as race or gender. Although extensive methods have been developed for ITR estimation, most focus solely on optimizing treatment outcomes, while research on fairness-aware ITRs remains limited. Moreover, the practical scenario of multicategory treatments is rarely considered in existing fair ITR studies. To address these gaps, we introduce a fairness notion for multicategory ITRs that requires treatment decisions to be independent of sensitive attributes, and then formulate the corresponding problem of estimating fair ITRs. To solve this problem, we further 
    
[^12]: 由对称性产生的重整化熵：简单性偏差与神经热寂

    Renormalized Entropy from Symmetries: Simplicity Bias and Neural Heat Death

    [https://arxiv.org/abs/2611.27889](https://arxiv.org/abs/2611.27889)

    该论文通过对朗之万动力学平衡分布的重整化分析，提出了一种由参数对称性诱导的新型熵，解释了神经网络隐式偏好简单函数（如低秩、奇异函数）的机制。

    

    神经网络具有参数对称性，即不影响网络所表示函数的参数变换。这些对称性如何影响神经网络最终学习到的函数？我们通过在带有高斯正则化的朗之万动力学下计算所表示函数的平衡分布，来研究由这些对称性引起的隐式偏差。为了分离对称性对该分布的影响，我们研究了正则化强度趋于零的极限。在某些情况下，我们证明了零正则化极限与无限时间极限是可交换的。在低温极限下，我们恢复了已知的有效正则化结果。在正温度下，一种重整化方法使我们能够推导出一种新的熵，该熵可预测由对称性引起的对某些函数的偏好。对于线性网络，我们得到了显式的熵项，这些熵项有利于奇异函数和低秩函数。对于具有……的网络（摘要在此处截断）

    arXiv:2611.27889v1 Announce Type: new  Abstract: Neural networks have parameter symmetries: parameter transformations that do not affect the represented function. How do these symmetries affect the functions that neural networks ultimately learn? We study the implicit bias arising from these symmetries by calculating the equilibrium distribution over represented functions under Langevin dynamics with Gaussian regularization. To isolate the effect of symmetries on this distribution, we study the limit in which the strength of the regularization is taken to zero. In certain cases, we show that the zero-regularization and infinite-time limits commute. In the low temperature limit, we recover known effective regularizations. At positive temperature, a renormalization procedure allows us to derive a novel entropy which predicts symmetry-induced preference for certain functions. For linear networks, we obtain explicit entropy terms that favor singular and low-rank functions. For networks wit
    
[^13]: 从要义到逐字：一张图像经由扩散先验的可控记忆路径

    From Gist to Verbatim: The Controllable Path of One Image Through a Diffusion Prior

    [https://arxiv.org/abs/2611.27875](https://arxiv.org/abs/2611.27875)

    该研究将生成模型的记忆化视为一个可控问题，发现记忆的形成会经历一个抽象阶段——既保留目标图像的语义要义又保持外观多样性，且此阶段比逐字复现更有利于编辑、单样本识别和检索，为理解和利用模型记忆开辟了新路径。

    

    现代生成模型通常因其泛化能力而被研究，而记忆化则被视为一种失败模式。我们反转了这一视角，探究生成模型如何能够回忆起单个数据。我们将记忆化构建为一个控制问题：一个学习到的条件信号引导一个冻结的扩散模型趋向目标图像，并跟踪这一记忆在整个训练过程中的形成过程。在收敛到精确复现（逐字记忆）之前，记忆会经历一个抽象化阶段：样本保留目标的语义内容（要义），同时在外观上保持多样性，将可预测的细节留给生成先验。尽管缺乏标签、描述或任何显式的语义目标，这种语义组织依然涌现出来。在抽象化阶段，学习到的记忆接近普通文本条件化的效果，并且在编辑、单样本识别和检索方面比接近完美复现阶段更为有效。

    arXiv:2611.27875v1 Announce Type: new  Abstract: Modern generative models are typically studied for their ability to generalize, while memorization is treated as a failure mode. We invert this perspective and ask how a generative model can recall a single datum. We frame memorization as a control problem: a learned conditioning signal steers a frozen diffusion model toward a target image, and we follow the formation of this memory throughout training. Before converging to exact reproduction (verbatim), memory passes through a regime of abstraction: samples preserve the target's semantic content (gist) while remaining diverse in appearance, leaving predictable detail to the generative prior. This semantic organization emerges despite the absence of labels, captions, or any explicit semantic objective. In the abstraction regime, the learned memory approaches ordinary text conditioning and supports editing, one-shot recognition, and retrieval more effectively than in the near-perfect reco
    
[^14]: 超越视觉保真度：比较类别型与连续型世界模型

    Beyond Visual Fidelity: Comparing Categorical and Continuous World Models

    [https://arxiv.org/abs/2611.27870](https://arxiv.org/abs/2611.27870)

    该论文通过受控的2D和3D环境实验系统比较了类别型与连续型预测对世界模型动态学习的影响，发现在无损视觉编码条件下，类别型模型能以更少训练计算量获得更高的动态rollout有效性和更好的随机性校准，表明视觉逼真度并不等同于动态保真度。

    

    视觉世界模型如今已能生成引人入胜的交互式世界，但视觉上的逼真并不保证动态的真实可信。我们系统地比较了在下一token预测和迭代式整帧生成两种范式下，类别型预测与连续型预测如何塑造模型学习到的动态。这些选择通常在自回归token模型和整帧流匹配等方法中被捆绑在一起。我们设计了受控的2D和3D环境，并评估动态有效性、校准性和视觉质量。在具有无损视觉编码的2D环境中，类别型模型以更少的训练计算量获得更高的rollout有效性，尤其体现在随机性实体上。在类别型模型中，贪心解码产生最高的rollout有效性，而从下一token模型中采样则得到校准最佳的随机预测。连续模型的失败案例（如游戏角色消失）与在更复杂环境中观察到的失败情况相呼应。

    arXiv:2611.27870v1 Announce Type: cross  Abstract: Visual world models can now generate compelling interactive worlds, but visual realism does not guarantee faithful dynamics. We systematically compare how categorical and continuous prediction shape learned dynamics under both next-token and iterative whole-frame generation. These choices are often bundled in approaches such as autoregressive token models and whole-frame flow matching. We design controlled 2D and 3D environments and evaluate dynamics validity, calibration, and visual quality. In 2D with lossless visual encoding, categorical models achieve higher rollout validity with less training compute, particularly for stochastic entities. Among categorical models, greedy decoding produces the highest rollout validity, while sampling from a next-token model yields the best-calibrated stochastic predictions. Failures in continuous models, such as disappearing game characters, echo those seen in more complex environments. Here, known
    
[^15]: 不坍缩而移动的锚点：面向视觉Transformer的独占区域锚点路由

    Anchors That Move Without Collapsing: Exclusive-Region Anchor Routing for Vision Transformers

    [https://arxiv.org/abs/2611.27855](https://arxiv.org/abs/2611.27855)

    该论文提出面向视觉Transformer的独占区域锚点路由机制，让每个锚点持有互不重叠的Voronoi区域从构造上杜绝锚点摘要坍缩，并结合焦点锚点将锚点置于习得重要性图的峰值处，从而显著提升分类精度。

    

    锚点token让视觉Transformer能够通过少数摘要来交换全局信息，但没有任何机制强制不同的锚点去摘要不同的内容。在同一个共享骨干网络上，我们证明Perceiver式路径和AnchorFormer路径会坍缩为单一摘要——即使后者带有质量归一化也是如此（池化图相似度达0.99）——而其他软性路径，包括代表性注意力，仍然部分冗余（0.59-0.79）。这些结果涉及我们模块内部的路由模块，而非已发表的完整模型。对图像块网格的独占划分在构造上不可能坍缩，在逐组件匹配的条件下，它在CIFAR-100上比代表性注意力的准确率高2.4-3.6个百分点。随后，我们提出焦点锚点，它将大多数锚点放置在习得的重要性图的峰值处，同时所有锚点保持一个独占的Voronoi划分。焦点锚点在杂乱画布场景中优于固定划分（+1.5个点……）

    arXiv:2611.27855v1 Announce Type: cross  Abstract: Anchor tokens let a vision transformer exchange global information through a few summaries, but nothing forces different anchors to summarise different things. On one shared backbone we show that a Perceiver-style pathway and an AnchorFormer pathway collapse to a single summary, the latter despite its mass normalisation (pooling-map similarity 0.99), while other soft pathways, including Representative Attention, stay partly redundant (0.59-0.79). These results concern the routing modules inside our block, not the full published models. An exclusive partition of the patch grid cannot collapse by construction, and matched component by component it is 2.4-3.6 points more accurate than Representative Attention on CIFAR-100. We then propose focus anchors, which place most anchors at the peaks of a learned importance map while all anchors keep one exclusive Voronoi partition. Focus anchors beat a fixed partition on a cluttered canvas (+1.5 p
    
[^16]: DistDNA：随机解码下的语言模型指纹识别

    DistDNA: Language Model Fingerprinting under Stochastic Decoding

    [https://arxiv.org/abs/2611.27831](https://arxiv.org/abs/2611.27831)

    提出 DistDNA 方法，通过从少量采样响应中估计响应分布，并将解码设置变化与检查点变化的影响区分开来，实现随机解码条件下的大语言模型可靠指纹识别。

    

    行为指纹通过大语言模型（LLM）对一组共享提示的响应来识别该模型，可支持诸如验证已部署模型是否与其声称的检查点相符等应用。然而，已部署的模型通常采用随机解码提供服务，此时同一模型对相同提示可能产生不同的响应，且温度和 top-p 设置往往是未知的。诸如 LLM DNA 之类的现有表示方法对这种变化的处理能力有限。一种自然的解决方案是通过模型的响应分布来刻画模型，但这带来了两个挑战：首先，每个分布只能从少量采样响应中估计；其次，解码设置的改变可能以与更换检查点相当的程度改变响应。因此，仅凭指纹距离无法为模型身份提供完整的判断依据，识别通常还需依赖……（摘要在此处截断）

    arXiv:2611.27831v1 Announce Type: cross  Abstract: Behavioral fingerprints identify a large language model (LLM) from its responses to a shared set of prompts, enabling applications such as verifying whether a deployed model corresponds to the checkpoint it claims to be. Deployed models, however, are commonly served with stochastic decoding, under which the same model may produce different responses to an identical prompt, depending on an unknown temperature and top-$p$ setting. Existing representations such as LLM DNA provide limited treatment of this variation. A natural remedy is to characterize models by their response distributions, which, however, raises two challenges. First, each distribution must be estimated from only a small number of sampled responses. Second, a change in decoding setting can alter responses to the same extent as a change of checkpoint. Consequently, fingerprint distances alone provide an incomplete basis for model identity, and identification typically rel
    
[^17]: 基于图感知扩散模型的图信号生成

    Graph Signal Generation via Graph-Aware Diffusion Models

    [https://arxiv.org/abs/2611.27826](https://arxiv.org/abs/2611.27826)

    提出图感知扩散模型（GAD），将前向过程建模为图上的热扩散，并设计模式自适应信噪比调度（MASS），为每个图频率分配独立扩散速率以保持相同信噪比，从而实现更好的图信号生成与分布学习。

    

    许多信号定义在图的节点上，例如道路传感器处的交通流量或用户对相关产品的购买。这些信号通常只被部分观测到，学习其分布可以实现对缺失值的插补以及对未来值的预测。扩散模型能够学习这类分布，但其前向噪声在各节点之间是相互独立的，并不依赖于图结构。我们提出图感知扩散（GAD），其前向过程是由高斯噪声驱动的图上热扩散。在图傅里叶基下，前向转移在每个频率上表现为一个标量高斯信道。在共享扩散速率的情况下，高图频率比低图频率衰减得更快。我们提出模式自适应信噪比调度（MASS），为每个图频率分配各自的扩散速率，并使所有频率保持相同的信噪比。对于图平稳信号，MASS 能够使最优线性去噪器在各频率上的归一化误差相等。

    arXiv:2611.27826v1 Announce Type: cross  Abstract: Many signals are defined on the nodes of a graph, such as traffic flow at road sensors or user purchases of related products. They are often partly observed, and learning their distribution enables imputing missing values and forecasting future ones. Diffusion models learn such distributions, but their forward noise is independent across nodes and does not depend on the graph. We introduce graph-aware diffusion (GAD), whose forward process is a heat diffusion on the graph driven by Gaussian noise. In the graph Fourier basis, the forward transition is a scalar Gaussian channel at each frequency. With a shared diffusion rate, high graph frequencies are attenuated faster than low ones. We propose Mode-Adaptive SNR Scheduling (MASS), which assigns each graph frequency its own diffusion rate and keeps all of them at the same SNR. For graph-stationary signals, MASS equalizes the normalized error of the optimal linear denoiser across frequenc
    
[^18]: 用于低秩适应的潜在递归混合

    Mixture of Latent Recursions for Low-Rank Adaptation

    [https://arxiv.org/abs/2611.27823](https://arxiv.org/abs/2611.27823)

    MoLR是一种自适应递归框架，通过共享变换递归细化潜在表示，并利用轻量级阈值路由器动态决定每个令牌的递归深度，在几乎不增加参数开销的情况下显著提升了LoRA的适应表达能力。

    

    低秩适应通过固定的低秩架构执行任务特定适应，其表达能力通常通过扩大或添加可学习模块来提升，但这会带来成比例的参数开销。我们提出了潜在递归混合，这是一个自适应递归框架，能够在保持参数效率的同时增强适应的表达能力。MoLR使用共享变换对每个输入的降投影潜在表示进行递归细化。一个轻量级的基于阈值的路由器独立决定每个令牌是否需要进行进一步细化，同时动态调整的阈值促进各递归深度之间的均衡利用。在多种语言和视觉任务上的广泛实验表明，MoLR在相对于LoRA几乎可忽略的参数开销下，始终优于代表性的LoRA变体。进一步分析表明，MoLR学习到输入相关的递归（原文在此处截断）

    arXiv:2611.27823v1 Announce Type: new  Abstract: Low-Rank Adaptation (LoRA) performs task-specific adaptation using a fixed low-rank architecture, whose expressiveness is typically increased by enlarging or adding learnable modules with a proportional parameter overhead. We introduce Mixture of Latent Recursions (MoLR), an adaptive recursion framework that enhances adaptation expressiveness while preserving parameter efficiency. MoLR recursively refines the down-projected latent representation of each input using a shared transformation. A lightweight threshold-based router independently determines whether each token should undergo further refinement, while dynamically adjusted thresholds encourage balanced utilization across recursion depths. Extensive experiments across diverse language and vision tasks demonstrate that MoLR consistently outperforms representative LoRA variants with negligible parameter overhead over LoRA. Further analyses show that MoLR learns input-dependent recurs
    
[^19]: 使用扩散模型采样非阿贝尔格点规范理论

    Sampling non-abelian lattice gauge theories with diffusion models

    [https://arxiv.org/abs/2611.27821](https://arxiv.org/abs/2611.27821)

    本文提出将规范等变扩散模型与格点规范等变卷积神经网络和Metropolis调整的退火朗之万算法相结合，作为格点规范理论的采样器，实现了对二维U(2)场构型的准确采样、四维SU(3)的初步成功结果，以及对更大逆耦合常数的优异外推能力。

    

    我们提出了规范等变扩散模型作为格点规范理论（LGT）的采样器。通过使用格点规范等变卷积神经网络（L-CNN）和Metropolis调整的退火朗之万算法（MAALA），我们能够准确采样二维U(2)场构型，并首次展示了四维SU(3)的有前景的初步结果。我们的二维模型在向外推至比训练时更大的逆耦合常数时表现出色。我们通过将观测量与解析已知值以及混合蒙特卡罗结果进行比较来证实这一点。

    arXiv:2611.27821v1 Announce Type: cross  Abstract: We present gauge-equivariant diffusion models as samplers for lattice gauge theories (LGTs). Using lattice gauge-equivariant convolutional neural networks (L-CNNs) and the Metropolis-adjusted annealed Langevin algorithm (MAALA), we are able to accurately sample two-dimensional U(2) field configurations and show first promising results for four-dimensional SU(3). Our two-dimensional models extrapolate remarkably well to larger inverse couplings than originally trained on. We confirm this by comparing observables to analytically known values and to hybrid Monte Carlo results.
    
[^20]: 基于黎曼梯度优化的自主量子机器学习

    Autonomous quantum machine learning via Riemannian gradient-based optimization

    [https://arxiv.org/abs/2611.27815](https://arxiv.org/abs/2611.27815)

    提出了一种基于黎曼梯度优化的完全自主自训练量子机器学习框架，无需量子测量或经典反馈即可在量子域内完成优化，并给出了指数收敛保证。

    

    量子机器学习模型在训练过程中通常依赖于重复的量子测量和经典优化，这使得从量子到经典的信息提取过程成为可扩展性的根本瓶颈。我们提出了一种基于黎曼梯度优化的自主自训练量子机器学习框架，其中优化过程完全在量子域内进行，无需量子测量或经典反馈。我们推导了针对广泛类别学习任务的黎曼梯度更新规则，并确定了由此产生的训练动力学能够完全自主实现的充分条件。在此机制下，我们针对足够小的步长建立了训练迭代次数意义上的指数收敛保证，并进一步从电路深度增长的角度分析了训练成本。重要的是，我们进一步表明，训练后的模型……

    arXiv:2611.27815v1 Announce Type: cross  Abstract: Quantum machine learning models typically rely on repeated quantum measurements and classical optimization during training, making the quantum-to-classical information-extraction process a fundamental bottleneck for scalability. We propose a framework for autonomous self-training quantum machine learning, based on Riemannian gradient optimization, in which the optimization proceeds entirely within the quantum domain without requiring quantum measurements or classical feedback. We derive the Riemannian gradient update rules for a broad class of learning tasks and identify sufficient conditions under which the resulting training dynamics can be realized fully autonomously. Within this regime, we establish an exponential convergence guarantee, in the training iterations, for sufficiently small step sizes, and further analyze the training cost in terms of the growth of the circuit depth. Importantly, we further show that the trained model 
    
[^21]: 重新设计神经网络：即使在强对抗攻击下也不会“自信地犯错”

    Redesigning Neural Networks to Not be Confidently Wrong even to Strong Adversarial Attacks

    [https://arxiv.org/abs/2611.27811](https://arxiv.org/abs/2611.27811)

    该论文提出了一种模拟生物视觉皮层简单细胞与复杂细胞交替结构的浅层神经网络架构，通过能够学习选择性与容忍性表征的新型组件，实现了准确、鲁棒且与人类感知更一致的图像分类，并能有效抵御强对抗攻击，避免对未知类别样本做出高置信度的错误预测。

    

    早期的深度学习模型试图通过一种神经层级结构来模拟视觉感知，该结构在两种类型的层之间交替：一种模拟皮层简单细胞（旨在提高选择性），另一种模拟皮层复杂细胞（旨在提高容忍度）。我们证明，这种神经网络（NN）架构在采用能够更好学习选择性与容忍性表征的新型组件实现后，能够实现准确且鲁棒的图像分类，并改善了模型行为与人类感知能力之间的一致性。首先，所提出的模型是浅层的，层数少于十对，这更符合生物视觉系统所执行的快速前馈目标分类机制。其次，这些模型远不容易对来自未知类别的样本或图像不一致的情况产生高置信度的分类预测……

    arXiv:2611.27811v1 Announce Type: cross  Abstract: Early deep learning models attempted to simulate visual perception using a neural hierarchy that alternated between two types of layer: one modelled on cortical simple cells (that was designed to increase selectivity), and one modelled on cortical complex cells (that aimed to increase tolerance). We show that such a neural network (NN) architecture, when implemented with novel components that are better able to learn selective and tolerant representations, is capable of accurate and robust image classification and improved alignment between the behaviour of the model and the perceptual abilities of humans. Firstly, the proposed models are shallow, fewer than ten layer-pairs, which is more consistent with the rapid feedforward object categorisation performed by the biological visual system. Secondly, they are far less prone to producing classification predictions with high confidence for samples from unknown classes or for image consist
    
[^22]: SLP-Bench：评估与改进推理型音频语言模型在临床语音上的声学论断

    SLP-Bench: Evaluating and Improving Acoustic Claims of Reasoning Audio Language Models on Clinical Speech

    [https://arxiv.org/abs/2611.27810](https://arxiv.org/abs/2611.27810)

    提出SLP-Bench基准，将推理型音频语言模型思维链中的每一项声学论断与临床医生的感知评分逐条对照，评估其事实性、稳定性和一致性，结果显示十个开源模型的论断平衡准确率仅为0.52至0.58。

    

    音频语言模型能够根据语音回答临床问题，其表现优于基于手工设计声学特征的基线方法。推理型音频语言模型在给出预测之前，会生成一条涉及说话人嗓音属性的思维链。模型对嗓音做出的每一项陈述，都需要与临床医生对同一录音的感知评分进行核对。然而在临床语音领域，现有评估方法仅对最终答案进行打分。我们提出了言语语言病理学家基准，这是一个将思维链中的每一项声学论断与临床医生对同一录音的感知评分进行逐条对照评分的基准。该基准涵盖五个由临床医生评分的数据集和25个感知维度，并从事实准确性、重采样下的稳定性，以及改写问题与不同问题之间的一致性等方面评估这些论断。在对十个开源权重推理型音频语言模型进行零样本评估中，论断的平衡准确率介于0.52至0.58之间，论断会在……（原文摘要在此处截断）

    arXiv:2611.27810v1 Announce Type: new  Abstract: Audio language models (ALMs) can answer clinical questions from speech, outperforming baselines built on handcrafted acoustic features. Reasoning ALMs generate a chain of thought that refers to properties of the speaker's voice before predicting. Each statement the model makes about the voice needs to be checked against clinicians' perceptual ratings of the same recording. On clinical speech, existing evaluations score only the final answer. We introduce Speech Language Pathologist Bench (SLP-Bench), a benchmark grading each acoustic claim in the chain of thought against clinicians' perceptual ratings of the same recording. It covers five clinician-rated datasets and 25 perceptual dimensions, and assesses claims for factuality, stability under resampling, and consistency across reworded and different questions. Across ten open-weight reasoning ALMs evaluated zero-shot, claim balanced accuracy ranges from 0.52 to 0.58, claims change under
    
[^23]: DOiT：在反洗钱审查中，更好的资金流排序可以与更差的账户选择共存

    DOiT: Better Money-Flow Ranking Can Coexist with Worse Account Selection in Anti-Money Laundering Review

    [https://arxiv.org/abs/2611.27809](https://arxiv.org/abs/2611.27809)

    DOiT 揭示了反洗钱审查中一个关键现象：即使 GNN 重排序提升了交易资金流的排序精度（更高的 AP），也可能损害上游账户选择的质量，因此应在同等审查容量下将资金流排序与账户选择这两种决策分离处理。

    

    反洗钱（AML）审查需要选择可疑账户并重构转账链条。下游图神经网络（GNN）会对资金流证据进行排序，但同时也可能对账户队列进行重新排序。DOiT 在同等审查容量下将这两种决策分离：账户选择决定了图调查的起点位置。在 AMLWorld 和 AMLNet 数据集上的实验表明，GNN 的重排序会损害高收益的上游排序，但对低收益排序有所帮助。在 AMLWorld 上，这种损害与维度匹配的、逐次运行的解码器中更高的交易平均精度（AP）并存，证明了这两种排序能力是可分离的。在小容量情况下，正邻居样本可能取代更密集的上游前缀并降低收益；随着该前缀被逐渐耗尽，替换成本下降，导致偏好排序可能发生逆转。一个基于真实交易模式构建的受限访问合成基准重现了这种分离现象以及基于验证的上游保持效果。容量匹配的组合方式……

    arXiv:2611.27809v1 Announce Type: cross  Abstract: Anti-money laundering (AML) review selects suspicious accounts and reconstructs transfers. Downstream graph neural networks (GNNs) rank money-flow evidence but may also reorder the account queue. DOiT separates these decisions at equal review capacity: account selection determines where graph investigation starts. Across AMLWorld and AMLNet, GNN reordering harms higher-yield upstream orders but helps lower-yield ones. On AMLWorld, this harm coexists with higher transaction average precision (AP) in dimension-matched, per-run decoders, demonstrating separable ranking abilities. At small capacities, positive neighbors can displace a denser upstream prefix and lower yield; replacement costs fall as that prefix is exhausted, allowing the preferred order to reverse. A restricted-access synthetic benchmark informed by real transaction patterns reproduces this separation and validation-based upstream preservation. Capacity-matched combination
    
[^24]: $\mathtt{DistP}$：通过零样本超参数迁移扩展本地更新训练

    $\mathtt{DistP}$: Scaling Local-Update Training with Zero-Shot Hyperparameter Transfer

    [https://arxiv.org/abs/2611.27808](https://arxiv.org/abs/2611.27808)

    提出理论框架$\mathtt{DistP}$，将$\mathtt{DDP}$的零样本超参数迁移规则扩展到本地更新训练，使学习率能够跨模型宽度、深度和同步间隔直接迁移，从而免去昂贵的超参数调优。

    

    本地更新方法为分布式数据并行（$\mathtt{DDP}$）提供了一种通信高效的替代方案，但引入了需要昂贵调优的额外超参数。对于$\mathtt{DDP}$，零样本超参数迁移通过复用在较小模型上调优的设置来降低这一成本，但其在本地更新训练中的适用性仍不明确。我们提出了$\mathtt{DistP}$，一个理论框架，将现有的迁移规则从$\mathtt{DDP}$扩展到具有不同工作节点数量、本地步数和外层优化器的本地更新训练。当使用平均作为外层优化器时，$\mathtt{DistP}$支持学习率在宽度和深度上的迁移，同时提升模型质量。调优后的内层学习率还可以直接在不同同步间隔之间迁移，但在增加全局批量大小（batch size）时遵循两种规律：对于较大的本地批量采用平方根学习率缩放，对于其他情况则采用固定的参考学习率（原文此处截断）。

    arXiv:2611.27808v1 Announce Type: cross  Abstract: Local-update methods offer a communication-efficient alternative to Distributed Data Parallelism ($\mathtt{DDP}$), but introduce additional hyperparameters that require costly tuning. For $\mathtt{DDP}$, zero-shot hyperparameter transfer reduces this cost by reusing settings tuned on smaller models, but its applicability to local-update training remains unclear. We propose $\mathtt{DistP}$, a theoretical framework extending existing transfer rules from $\mathtt{DDP}$ to local-update training with varying worker counts, local steps, and outer optimizers. With averaging as the outer optimizer, $\mathtt{DistP}$ supports learning-rate transfer across width and depth while improving model quality. The tuned inner learning rate also directly transfers across synchronization intervals, but follows two regimes when increasing the global batch size: square-root learning-rate scaling for larger local batches and a fixed reference rate for additi
    
[^25]: ε-SegFormer：用于电子显微镜稀疏监督分割的多分辨率视觉Transformer

    $\varepsilon$-SegFormer: Multi-Resolution Vision Transformers for Sparsely Supervised Segmentation in Electron Microscopy

    [https://arxiv.org/abs/2611.27796](https://arxiv.org/abs/2611.27796)

    ε-SegFormer提出了一种多分辨率视觉Transformer，通过对每个标注像素在三个物理分辨率下采样视图并用注意力机制在共享坐标框架中关联所有视图，实现电子显微镜数据的稀疏监督语义分割，大幅降低密集标注成本。

    

    在电子显微镜（EM）体积数据中分割细胞器需要密集标注，这需要专家花费数月时间才能完成，且成像分辨率的每一次提升都会增加这一成本。稀疏监督方法可将这一成本降低到仅需一小部分标注像素，但它们所基于的卷积网络只能感知每个像素周围有限的邻域，而细胞器的身份识别往往依赖于远超该邻域的上下文信息。我们提出ε-SegFormer，一种用于稀疏监督语义分割的多分辨率视觉Transformer，它建立在MuViT的架构之上，并扩展了ε-Seg的训练方法。对于每个标注像素，ε-SegFormer以其为中心在三个物理分辨率下采样视图，将这些视图的图像块嵌入到单个token序列中，并在图像坐标系中编码每个token的位置，从而使注意力机制能够在共享的坐标框架中关联所有视图。分割头对...

    arXiv:2611.27796v1 Announce Type: cross  Abstract: Segmenting organelles in electron microscopy (EM) volumes requires dense annotations that take experts months to produce, and every gain in imaging resolution adds to this cost. Sparsely supervised methods reduce this cost to a small fraction of labeled pixels, but the convolutional networks they build on see a limited neighborhood around each pixel, while the identity of an organelle often depends on context far beyond it. We introduce $\varepsilon$-SegFormer, a multi-resolution Vision Transformer for sparsely supervised semantic segmentation, which builds on the architecture of MuViT and extends the training method of $\varepsilon$-Seg. For each labeled pixel, $\varepsilon$-SegFormer samples views at three physical resolutions centered on it, embeds their patches into a single token sequence, and encodes every token's position in image coordinates, so that attention relates all views in one shared frame. A segmentation head classifie
    
[^26]: DPMA：学习压缩混合专家模型

    DPMA: Learning to Compress Mixture-of-Experts Models

    [https://arxiv.org/abs/2611.27759](https://arxiv.org/abs/2611.27759)

    DPMA 提出了一种无需重新训练的可微分压缩方法，在全局专家预算下联合学习专家剪枝、参数合并、路由吸收三种操作及逐层的专家分配策略，从而在多个 MoE 模型和压缩率下取得了最优的压缩效果。

    

    混合专家架构具有很高的计算效率，但存储所有专家需要大量内存，限制了其在资源受限硬件上的部署。大多数无需重新训练的 MoE 压缩方法只选择专家剪枝或专家合并中的一种方式，且通常在各层采用统一的压缩率。我们提出了 DPMA（可微分剪枝-合并-吸收，Differentiable Prune Merge Absorb），它在全局专家预算约束下，通过梯度下降联合学习压缩操作和逐层的专家分配。借助各自独立的可学习系数，DPMA 能够剪枝专家、合并其参数，并通过将路由权重前向传递给保留的专家来吸收其路由贡献。这些系数和专家分配在保持模型权重冻结的情况下，通过在校准数据上匹配原始模型的输出进行优化。在四个 MoE 模型、八个多选基准测试和五种压缩率上，DPMA 取得了最高的准确率……

    arXiv:2611.27759v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) architectures offer high computational efficiency, but storing all experts requires substantial memory and limits deployment on resource-constrained hardware. Most retraining-free MoE compression methods commit to either pruning or merging experts, often with a uniform compression ratio across layers. We introduce DPMA (Differentiable Prune Merge Absorb), which jointly learns compression operations and layer-wise expert allocation through gradient descent under a global expert budget. Separate learnable coefficients allow DPMA to prune experts, merge their parameters, and absorb their routing contributions by forwarding routing weights to retained experts. These coefficients and expert assignments are optimised to match the original model's outputs on calibration data while its weights remain frozen. Across four MoE models, eight multiple-choice benchmarks, and five compression ratios, DPMA achieves the highest a
    
[^27]: 基于可迁移学习价值模型的预算约束子集选择

    Budgeted Subset Selection with a Transferable Learned Value Model

    [https://arxiv.org/abs/2611.27751](https://arxiv.org/abs/2611.27751)

    提出SALVES框架，将预算约束下的子集选择分解为可跨任务迁移的学习价值模型与基于模拟退火的子集搜索两个组件，摆脱了现有方法对特定应用场景的依赖。

    

    子集选择是机器学习、统计学和组合优化中的一项基础任务，在主动学习、波段选择、输入选择和实验设计等领域有着广泛应用。子集的效用取决于只有在选择之后才能揭示的信息（如标签或响应），以及所选候选者之间的相互作用。现有方法以特定于应用的方式将价值估计与冗余控制相结合，这限制了它们在其他领域的使用。本文提出了SALVES（基于学习价值估计与搜索的子集获取），这是一个将子集选择分解为两个组件的框架。首先，在隐藏信息已知的任务上通过监督学习训练一个价值模型；在新任务上，该模型预测每个候选者能为当前选择所增加的效用。其次，选择步骤基于这些预测来选取子集，通常采用模拟退火方

    arXiv:2611.27751v1 Announce Type: cross  Abstract: Subset selection is a fundamental task in machine learning, statistics and combinatorial optimisation, with applications in active learning, band selection, input selection and experimental design. The utility of a subset depends on information revealed only after the selection, such as labels or responses, and on interactions among selected candidates. Existing approaches combine value estimation and redundancy control in application-specific ways, which limits their use in other domains. This paper introduces Subset Acquisition with Learned Value Estimates and Search (SALVES), a framework that decomposes subset selection into two components. First, a value model is trained by supervised learning on tasks where the hidden information is known. On a new task, it predicts the utility each candidate would add to the current selection. Second, a selection step chooses the subset from these predictions, typically by simulated annealing wit
    
[^28]: 助听器中选择环境自适应个性化增益的切换机制

    A Switching Mechanism for Selecting Environment-Adaptive Personalized Gains in Hearing Aids

    [https://arxiv.org/abs/2611.27726](https://arxiv.org/abs/2611.27726)

    本文提出了一种助听器切换机制，通过单类语音活动检测器排除前景语音干扰，并结合贝叶斯变点检测与历史环境记忆，自动识别当前背景音频环境并选择个性化增益。

    

    个性化助听器放大增益已被证明比处方式增益更受用户青睐。由于用户偏好设置可能因背景音频环境而异，本文提出了一种切换机制或系统，可根据当前的背景音频环境自动选择个性化增益。为防止前景语音干扰背景环境分析，设计了一种单类语音活动检测器，用于从混合噪声的语音中识别语音活动。通过一致性检查来区分前景语音和背景语音，并将嘈杂人声保留为感兴趣的背景环境，因为助听器用户常常难以在背景嘈杂声中理解语音。随后，通过一种结合贝叶斯变点检测与先前遇到环境记忆的在线发现方法，对不含前景语音的声音片段进行分析。

    arXiv:2611.27726v1 Announce Type: new  Abstract: Personalized amplification gains for hearing aids have been shown to be preferred over prescriptive gains. Since the preferred settings can vary depending on the background audio environment, this paper presents a switching mechanism or system that automatically selects personalized gains for a current background audio environment. To prevent foreground speech from interfering with background environment analysis, a one-class voice activity detector is designed to identify speech activity from speech mixed with noise. A consistency check is conducted to enable distinguishing foreground speech from background speech, preserving babble as a background environment of interest because hearing aid users often struggle to understand speech in background babble. Sound segments without foreground speech are then analyzed by an online discovery method combining Bayesian change-point detection with a memory of previously encountered environments. 
    
[^29]: 超越数学、代码与逻辑：推理模型能否迁移至科学论证？

    Beyond Math, Code, and Logic: Do Reasoning Models Transfer to Scientific Argumentation?

    [https://arxiv.org/abs/2611.27722](https://arxiv.org/abs/2611.27722)

    该研究首次系统评估了语言模型在科学论证任务上的表现，发现即使前沿推理模型在领域特定论证任务上仍存在根本性能力差距，并提出了一种显式搭建论证推理脚手架的训练框架，显著提升了模型的科学论述能力。

    

    科学论述依赖于论证性推理：识别论点、评估证据以及构建连贯的回应。尽管近年来具备推理能力的语言模型在数学、编程和逻辑基准测试中表现出色，但它们参与科学论证的能力仍不明确。我们首次对语言模型进行了系统性评估，涵盖七项任务，包括论证挖掘、反驳生成和话语层面的推理，使用的数据来自研究论文、同行评审和科研基金申请书。我们的研究表明，即使是拥有强大通用推理能力的前沿模型，在处理领域特定的论证任务时也会遇到困难，这凸显了一个根本性的能力差距。为解决这一问题，我们引入了一个训练框架，显式地为语言模型的论证推理过程搭建脚手架，从而显著提升了它们在科学论述中的能力。

    arXiv:2611.27722v1 Announce Type: cross  Abstract: Scientific discourse depends on argumentative reasoning: identifying claims, evaluating evidence, and constructing coherent responses. While recent advances in reasoning-capable language models have demonstrated strong performance on mathematical, coding, and logical benchmarks, their ability to engage in scientific argumentation remains unclear. We present the first systematic evaluation of language models across seven tasks spanning argument mining, rebuttal generation, and discourse-level reasoning using research papers, peer reviews, and grant proposals. Our study reveals that even frontier models with strong general reasoning skills struggle with domain-specific argumentative tasks, highlighting a fundamental capability gap. To address this, we introduce a training framework that explicitly scaffolds the argumentative reasoning process in language models, substantially improving their competence in scientific discourse. The result
    
[^30]: 多语言情感分类的决策模型：与大语言模型和编码器的对比研究

    Decision Models for Multilingual Emotion Classification: A Comparative Study Against LLMs and Encoders

    [https://arxiv.org/abs/2611.27721](https://arxiv.org/abs/2611.27721)

    该研究首次在五种语言的多语言情感分类任务上系统比较了决策模型、大语言模型与微调编码器，发现最佳零样本系统的性能仅比微调编码器低约1个宏F1点。

    

    决策模型接收一个带有封闭答案列表的问题，并为每个答案返回一个概率，而非自由文本。它们具有低延迟和对提示措辞鲁棒性的优势，但尚无研究将多种决策模型与跨语言的微调编码器进行系统比较。我们探讨未经标注数据训练（零样本）使用的系统能否替代微调方法，以及决策格式相比通用大语言模型（LLM）能带来哪些额外优势。我们在五种语言的10个社交媒体数据集上，针对四种情绪（喜悦、悲伤、愤怒、恐惧），将三个决策模型（Jev、OpenAI Decisions、Strands Decider 2B）和两个采用意大利语指令的大语言模型（GPT-6 Luna、Qwen3.5-2B），与两个微调的多语言编码器（tabularisai、XLM-EMO）进行对比，并采用配对差异检验和等效性检验。结果表明，三个最佳的零样本系统（两个决策模型和一个大语言模型）的平均宏F1分数仅比在目标语言上微调的编码器低约1分。

    arXiv:2611.27721v1 Announce Type: cross  Abstract: Decision models receive a question with a closed list of answers and return a probability for each answer instead of free text. They promise low latency and robustness to prompt wording, but no study has compared several of them with fine-tuned encoders across languages. We ask whether systems used without training on labelled data (zero-shot) can replace fine-tuning, and what the decision format adds over a general-purpose large language model (LLM). We compare three decision models (Jev, OpenAI Decisions, Strands Decider 2B) and two LLMs (GPT-6 Luna, Qwen3.5-2B), given Italian instructions, with two fine-tuned multilingual encoders (tabularisai, XLM-EMO) on four emotions (joy, sadness, anger, fear) in 10 social-media datasets in five languages, with paired difference and equivalence tests. The three best zero-shot systems (two decision models and one LLM) score on average about one macro-F1 point below an encoder fine-tuned on the ta
    
[^31]: 解决核PCA中的预像问题：再生性质就足够了

    Solving the pre-image problem in kernel PCA: The reproducing property is all you need

    [https://arxiv.org/abs/2611.27718](https://arxiv.org/abs/2611.27718)

    本文提出仅利用再生性质即可直接求解核PCA的预像问题，无需非线性优化或额外的监督学习，同时在线性核情形下能自然退化为标准PCA的重构映射。

    

    核PCA通过核技巧将数据隐式映射到再生核希尔伯特空间（RKHS），从而将PCA扩展到非线性降维，其应用范围涵盖图像去噪到分布外检测。虽然其投影映射已被充分理解，但如何恰当地定义相应的重构映射却是一个长期存在的挑战，因为从RKHS投影回输入空间并非易事。这就是所谓的预像问题。现有方法解决该问题的途径要么依赖非线性优化，这容易产生数值不稳定性且对初始化敏感；要么通过求解一个额外的监督学习问题，带来额外的计算开销。此外，尽管当使用线性核时核PCA的投影映射会退化为标准PCA的投影映射，但大多数现有方法并不能保证在这种情况下恢复PCA的重构映射。

    arXiv:2611.27718v1 Announce Type: new  Abstract: Kernel PCA extends PCA to nonlinear dimensionality reduction by implicitly mapping data into a reproducing kernel Hilbert space (RKHS) via the kernel trick, with applications ranging from image denoising to out-of-distribution detection. While its projection map is well understood, appropriately defining the associated reconstruction map is a longstanding challenge since projecting back from the RKHS to the input space is non-trivial. This is known as the pre-image problem. Existing approaches have tackled this problem by either relying on nonlinear optimization, which is prone to numerical instabilities and sensitive to initialization, or by solving a separate supervised learning problem, incurring additional computational overhead. Furthermore, although the projection map of kernel PCA reduces to that of standard PCA when a linear kernel is used, most of these approaches are not guaranteed to recover PCA's reconstruction map in this ca
    
[^32]: 几何约束学习 II：Lipschitz 函数的不可知正常学习

    Geometrically Constrained Learning II: Agnostic Proper Learning of Lipschitz Functions

    [https://arxiv.org/abs/2611.27700](https://arxiv.org/abs/2611.27700)

    本文为高斯分布下的 $\lambda$-Lipschitz 函数给出了一个正常学习器，其样本复杂度、运行时间和表示规模均为 $(d+1)^{\mathsf{poly}(\lambda/\varepsilon)}$，并在已知标签方差上界时可扩展为误差为 $\mathsf{opt}+\varepsilon$ 的不可知学习算法。

    

    我们在 $\mathbb{R}^d$ 上高斯分布的背景下，给出了针对 Lipschitz 函数的一个正常学习器。给定由某个 $\lambda$-Lipschitz 函数（$\lambda \ge 1$）标注的随机高斯样本，该学习器返回一个全局满足 $\lambda$-Lipschitz 条件的函数 $h:\mathbb{R}^d\to\mathbb{R}$，其损失（绝对损失或平方损失）至多为 $\varepsilon$。其样本复杂度、运行时间和表示规模均为 $(d+1)^{\mathsf{poly}(\lambda/\varepsilon)}$。在已知标签边缘分布方差上界的前提下，该学习算法可以成为不可知学习算法。从信息论角度看，对标签尺度的某种已知控制是必要的。对于方差不超过 $V^2 \ge 1$ 的任意标签，复杂度为 $(d+1)^{\mathsf{poly}(V\lambda/\varepsilon)}$，误差为 $\mathsf{opt} + \varepsilon$，其中 $\mathsf{opt}$ 是所有 Lipschitz 逼近器所能达到的最小误差。我们的方法结合了 $L_1$ 多项式回归……

    arXiv:2611.27700v1 Announce Type: new  Abstract: We give a proper learner for Lipschitz functions over the Gaussian distribution on $\mathbb{R}^d$. Given random Gaussian examples labeled by a $\lambda$-Lipschitz function, $\lambda \ge 1$, the learner returns a function $h:\mathbb{R}^d\to\mathbb{R}$ that is globally $\lambda$-Lipschitz and has loss (absolute or squared) at most $\varepsilon$. The sample complexity, running time, and representation size are \[   (d+1)^{\mathsf{poly}(\lambda/\varepsilon)}. \] The learning algorithm can be made \emph{agnostic}, given a known bound on the variance of the label marginal distribution. Some known control of label scale is information-theoretically necessary. For arbitrary labels with variance at most $V^2 \ge 1$, the complexity is \[   (d+1)^{\mathsf{poly}(V\lambda/\varepsilon)}, \] and the error is $\mathsf{opt} + \varepsilon$, where $\mathsf{opt}$ is the minimal error among Lipschitz approximators.   Our approach combines $L_1$ polynomial re
    
[^33]: 面向化工过程故障检测与结构学习的时空图神经网络：以田纳西-伊士曼过程为应用案例

    Spatiotemporal Graph Neural Networks for Fault Detection and Structural Learning in Chemical Processes: Use Case on the Tennessee Eastman Process

    [https://arxiv.org/abs/2611.27698](https://arxiv.org/abs/2611.27698)

    该论文提出一种时空图神经网络框架，通过将正常运行空间分解为特定于运行条件的子空间，并结合GATv2图结构学习与时间编码，解决了化工过程中因运行条件变化导致的误报与漏检问题，并在田纳西-伊士曼过程上验证了其故障检测能力。

    

    经典的故障检测与诊断（FDD）方法，包括许多数据驱动方法，都假设一个静态的正常运行空间，并将对固定参考的偏离视为故障指标。当运行条件（OCs）随时间变化时，这一假设便不再成立：合法的工况转换会触发误报，而中等程度的故障却无法被检测到。我们提出了一种时空图神经网络（GNN）框架，将正常运行空间分解为由转换函数连接的、特定于运行条件的子空间，并采用结合重构损失与Deep SVDD单分类项的双重学习目标。该框架通过图注意力网络（GAT）学习邻接矩阵，并将空间建模与时间编码相结合，以表征运行条件不断演变下的过程动态。本文在田纳西-伊士曼（Tennessee Eastman）过程上对该框架的基础组件（故障检测、GATv2空间图学习和时间编码）进行了评估

    arXiv:2611.27698v1 Announce Type: new  Abstract: Classical Fault Detection and Diagnosis (FDD) methods, including many data-driven approaches, assume a static normal operating space and interpret deviations from a fixed reference as fault indicators. When Operating Conditions (OCs) vary over time, this assumption breaks down: legitimate transitions trigger false alarms while moderate faults go undetected. We propose a spatiotemporal Graph Neural Network (GNN) framework that decomposes the normal operating space into OC-specific subspaces linked by transition functions, with a dual learning objective combining reconstruction loss and a Deep SVDD one-class term. It learns adjacency matrices through Graph Attention Networks (GATs) and integrates spatial modelling with temporal encoding to represent process dynamics under evolving OCs. This paper evaluates the framework's foundational components (fault detection, GATv2 spatial graph learning and temporal encoding) on the Tennessee Eastman 
    
[^34]: 共生学习：基于人工神经网络与脉冲神经网络的异构联邦学习

    Learning in Symbiosis: Heterogeneous Federated Learning with ANNs and SNNs

    [https://arxiv.org/abs/2611.27686](https://arxiv.org/abs/2611.27686)

    本文提出混合神经形态联邦学习框架，使异构的人工神经网络与脉冲神经网络设备能够在联邦学习环境中协同联合训练，从而推动向节能神经形态边缘AI的渐进式过渡。

    

    脉冲神经网络（SNNs）被广泛认为是第三代神经网络，代表了一种有前景的神经形态计算范式，有潜力降低人工智能（AI）的能源足迹。其能效优势使SNN对资源受限的边缘AI设备特别具有吸引力，这也推动了对SNN与联邦学习（FL）相结合的研究兴趣日益增长。尽管具有这些优势，将SNN集成到现有联邦学习系统中仍然具有挑战性，因为用基于SNN的替代设备替换所有传统人工神经网络（ANN）设备是不切实际的。相反，我们设想了一种向神经形态边缘AI的渐进式过渡，其中ANN和SNN设备共存并协作参与联邦学习。为促进这一过渡，我们提出了混合神经形态联邦学习（HNFL），这是一个能够在联邦学习环境中对异构ANN和SNN设备进行联合训练的框架。

    arXiv:2611.27686v1 Announce Type: new  Abstract: Spiking Neural Networks (SNNs), widely regarded as the third generation of neural networks, represent a promising neuromorphic computing paradigm with the potential to reduce the energy footprint of Artificial Intelligence (AI). Their en-ergy efficiency makes SNNs particularly attractive for resource-constrained Edge AI devices, motivating growing research interest in Federated Learning (FL) with SNNs. Despite these advantages, integrating SNNs into existing FL systems remains challenging, as replacing all conventional Artificial Neural Network (ANN) devices with SNN-based alternatives is impractical. Instead, we envision a gradual transition toward neuromorphic Edge AI, where ANN and SNN devices coexist and collaboratively participate in FL. To facilitate this transition, we propose Hybrid Neuromorphic Federated Learning (HNFL), a framework that enables joint training of heterogeneous ANN and SNN devices within an FL environment. Throug
    
[^35]: CanopyFormer：注意力引导的SegFormer用于可解释的森林覆盖分割

    CanopyFormer: Attention-Guided SegFormer for Explainable Forest Cover Segmentation

    [https://arxiv.org/abs/2611.27674](https://arxiv.org/abs/2611.27674)

    CanopyFormer通过注意力一致性损失将SegFormer的注意力与真实森林掩码对齐，并引入AAMO指标和删除测试对森林覆盖分割中的注意力可解释性进行量化评估，使注意力对齐度提升约22倍。

    

    在Transformer分割模型中，注意力通常是在训练之后才被可视化，而非受到监督。尽管注意力监督已在自然图像和驾驶场景中得到探索，但航空森林分割通常仅将注意力作为架构组件使用，并以定性方式评估可解释性。我们提出了CanopyFormer，这是一种注意力引导的SegFormer-B0模型，它使用注意力一致性损失将注意力与真实森林掩码对齐，促使注意力聚焦于树冠区域。由于SegFormer使用空间缩减注意力，我们通过将梯度加权注意力推演限制在最后一个编码器阶段（该阶段的注意力矩阵为方阵）来适配该方法。我们引入平均注意力掩码重叠度来衡量注意力对齐程度，并结合删除测试来评估忠实性。在5108张图像上，我们的方法将AAMO从0.0334提升至0.7476，约为原始模型的22倍。

    arXiv:2611.27674v1 Announce Type: cross  Abstract: In transformer segmentation models, attention is usually visualized after training rather than supervised. Although attention supervision has been explored for natural images and driving scenes, aerial forest segmentation typically uses attention only as an architectural component and evaluates interpretability qualitatively. We present CanopyFormer, an attention-guided SegFormer-B0 that uses an Attention Consistency Loss to align attention with the ground-truth forest mask, encouraging attention to focus on canopy regions. Since SegFormer uses spatial-reduction attention, we adapt Gradient-weighted Attention Rollout by restricting it to the final encoder stage, where the attention matrix is square. We introduce Average Attention-Mask Overlap (AAMO) to measure attention alignment and incorporate a deletion test to evaluate faithfulness. On 5108 images, our method increases AAMO from 0.0334 to 0.7476, approximately 22 times the vanilla 
    
[^36]: 预训练Transformer能否实现半参数有效推断？

    Can pretrained Transformers deliver semiparametrically efficient inference?

    [https://arxiv.org/abs/2611.27666](https://arxiv.org/abs/2611.27666)

    该论文证明单个预训练Transformer无需推导有效影响函数或估计干扰函数，即可通过一次前向传播对一族半参数待估量同时实现有效估计与具有渐近精确覆盖率的有效推断。

    

    半参数模型为统计推断提供了一个广泛的框架，然而有效估计通常依赖于针对每个待估量量身定制的影响函数校正，以及对每个样本重复进行的干扰函数估计。我们证明，单个预训练Transformer可以在一次前向传播中为一系列待估量提供有效估计和有效推断，而无需推导有效影响函数或估计干扰函数。通过在以真实参数作为唯一标签的模拟数据上进行预训练，该Transformer学习所查询待估量的后验均值和协方差，从而返回一个估计值和一个Wald型置信区域。理论上，我们证明，对于复杂度缓慢增长的局部平坦预训练先验，后验均值对每个查询的待估量都是有效的。该Transformer继承了这种有效性，并具有渐近精确的覆盖率，在评估总体上以概率成立。

    arXiv:2611.27666v1 Announce Type: cross  Abstract: Semiparametric models provide a broad framework for statistical inference, yet efficient estimation typically relies on influence function correction tailored to each estimand and on nuisance estimation repeated for every sample. We show that a single pretrained Transformer can deliver efficient estimation and valid inference for a family of estimands in one forward pass, without deriving efficient influence functions or estimating nuisance functions. Pretrained on simulated data with the true parameter as the only label, the Transformer learns the posterior mean and covariance of the queried estimand and hence returns an estimate and a Wald-type confidence region. Theoretically, we prove that, for locally flat pretraining priors of slowly growing complexity, the posterior mean is efficient for every queried estimand. The Transformer inherits this efficiency with asymptotically exact coverage, in probability over evaluation populations
    
[^37]: SMSD：一种语义虚假信息传播者检测方法

    SMSD: A Semantic Misinformation Spreader Detection Approach

    [https://arxiv.org/abs/2611.27661](https://arxiv.org/abs/2611.27661)

    提出SMSD方法，将虚假信息传播建模为时间性、以用户为中心、行为驱动的学习问题，通过融合用户级、网络级和对话传播特征的多模态架构与立场感知标注策略，实现对社交媒体虚假信息传播者的有效早期检测。

    

    虚假信息传播者在社交媒体上会显著放大虚假叙事，因此早期检测对于遏制其扩散至关重要。现有方法往往忽视用户行为的动态演变，因忽略用户立场而引入语义噪声，并且无法有效应对新出现的用户。为解决这些局限，我们提出了一种语义虚假信息传播者检测方法，将虚假信息传播建模为一个时间性的、以用户为中心的、行为驱动的学习问题。SMSD采用多模态架构，融合三类特征：用户级特征、网络级特征和对话传播特征。同时引入了一种立场感知的标注策略，以区分对虚假信息主张持支持、反对、质疑和中立态度的用户。此外，还引入了月度分位数分析来模拟真实且不断演变的用户参与情况。

    arXiv:2611.27661v1 Announce Type: new  Abstract: Misinformation spreaders on social media significantly amplify false narratives, making early detection critical to mitigate their spread. Existing approaches often overlook evolving user behavior, introduce semantic noise by ignoring user stance, and fail to handle newly emerging users effectively. To address these limitations, we propose a \underline{S}emantic \underline{M}isinformation \underline{S}preader \underline{D}etection approach (SMSD) that models misinformation propagation as a temporal, user-centric, and behavior-driven learning problem. SMSD adopts a multimodal architecture that incorporates three categories of features: user-level, network-level, and conversation propagation. A stance-aware labeling strategy is also introduced to differentiate supportive, opposing, questioning, and neutral users towards misinformation claims. Additionally, a monthly quantile analysis is introduced to simulate realistic, evolving user parti
    
[^38]: 超越样本难度：基于学习动态的自适应数据选择

    Beyond Sample Hardness: Adaptive Data Selection through Learning Dynamics

    [https://arxiv.org/abs/2611.27656](https://arxiv.org/abs/2611.27656)

    该论文提出LeDAS框架，通过难度、学习进展和退化三个学习动态信号自适应地选择训练数据，并利用基于上下文老虎机的自适应度量融合机制动态调整样本优先级，以极小的额外计算开销克服了传统仅依赖样本难度的数据选择方法的局限性。

    

    数据选择已成为降低模型训练成本的有效方法。然而，大多数现有方法往往严重依赖样本难度，这可能忽视样本学习动态之间的差异，从而导致次优的样本选择。在本文中，我们提出了LeDAS，一个端到端的学习动态感知选择框架，用于动态数据选择。该框架通过难度、学习进展和退化三个方面共同刻画样本重要性，能够以极小的额外计算开销在整个训练过程中实现自适应的样本优先级排序。具体而言，我们使用指数移动平均（EMA）追踪每个样本的损失动态，获得三个互补信号，分别捕捉当前的学习难度、正在发生的学习改进以及潜在的退化。为了适应不断演变的训练动态，我们引入了自适应度量融合，这是一种基于上下文老虎机的机制，能够动态调整……

    arXiv:2611.27656v1 Announce Type: new  Abstract: Data selection has emerged as an effective approach to reducing the model training costs. However, most existing methods often rely heavily on sample hardness, which may overlook differences in sample learning dynamics and lead to suboptimal sample selection. In this paper, we propose LeDAS, an end-to-end Learning Dynamics-Aware Selection framework for dynamic data selection, which jointly characterizes sample importance through hardness, learning progress, and deterioration, enabling adaptive sample prioritization throughout training with marginal additional computational overhead. Specifically, we track per-sample loss dynamics using exponential moving averages (EMA) to obtain three complementary signals that capture current learning difficulty, ongoing learning improvements, and potential degradation. To accommodate evolving training dynamics, we introduce Adaptive Metric Fusion, a contextual bandit-based mechanism that dynamically ad
    
[^39]: 流匹配的晚期动力学：距离几何与学习

    Late-Time Dynamics of Flow Matching: Distance Geometry and Learning

    [https://arxiv.org/abs/2611.27653](https://arxiv.org/abs/2611.27653)

    本文研究了流匹配的晚期动力学，证明在有限训练集上轨迹会以指数速度收缩至最近的训练样本，到训练集的平方距离生成显式收缩路径，并量化了由Voronoi界面附近初始质量所控制的精确分布与参考分布之间的Wasserstein误差。

    

    生成模型将简单的随机输入转化为类似于观测数据的样本。对于有限训练集，一个核心问题是其几何结构如何塑造终端动力学以及分配给各个样本的概率。我们针对在独立高斯样本与经验样本之间进行线性插值的流匹配来研究这一问题。到训练集的平方距离生成了向最近样本指数收缩的显式路径。我们证明了在指定的晚期状态处重启的精确轨迹，在所有经过时间以及远离Voronoi界面的起始状态紧致集上一致地逼近这些路径，误差关于重启噪声尺度的平方倒数呈指数级小。界面附近的初始质量控制着精确分布与参考分布之间的二次Wasserstein误差。对于高斯重启分布和固定高斯管状重启分布，这给出了……的平方根速率（摘要原文在此处截断）。

    arXiv:2611.27653v1 Announce Type: new  Abstract: Generative models transform simple random inputs into samples that resemble observed data. For a finite training set, a central question is how its geometry shapes the terminal dynamics and the probabilities assigned to individual samples. We address this question for flow matching with linear interpolation between independent Gaussian and empirical samples. The squared distance to the training set generates explicit paths contracting exponentially toward the nearest sample. We prove that exact trajectories restarted at prescribed late-time states approximate these paths uniformly over all elapsed times and compact sets of starting states away from Voronoi interfaces, with an error exponentially small in the inverse squared restart noise scale. The initial mass near interfaces controls the quadratic Wasserstein error between the exact and reference laws. For Gaussian and fixed Gaussian-tube restart laws, this gives a square-root rate in 
    
[^40]: 基于潜在世界模型子块推演的高效VLA强化学习

    Efficient VLA Reinforcement Learning via Subchunk Rollouts in Latent World Models

    [https://arxiv.org/abs/2611.27645](https://arxiv.org/abs/2611.27645)

    该论文提出LS-MBPO轻量级残差MBRL框架，通过层级强制缓解复合预测误差、并通过潜在子块推演结合冻结VLA策略与潜在残差策略，从而以低计算开销的潜在世界模型高效改进预训练VLA策略。

    

    基于模型的强化学习（MBRL）能够通过想象经验来改进预训练的视觉-语言-动作（VLA）策略。然而，现有的VLA-MBRL方法主要依赖大规模的图像或视频生成世界模型，会产生巨大的计算开销。潜在世界模型提供了一种更高效的替代方案，但面临两个挑战：递归预测过程中的复合误差累积，以及与VLA视觉输入接口的不兼容性。在本工作中，我们提出LS-MBPO，一个轻量级的残差MBRL框架，利用潜在世界模型来改进预训练的VLA策略。LS-MBPO引入了两项关键技术：（1）层级强制，将密集的因果监督与自回归反馈相结合，以缓解预测误差的复合累积；（2）潜在子块推演，利用冻结VLA策略的动作块和潜在残差策略进行连续的子块优化，从而弥合了……

    arXiv:2611.27645v1 Announce Type: new  Abstract: Model-based reinforcement learning (MBRL) can improve pretrained vision-language-action (VLA) policies through imagined experiences. However, existing VLA-MBRL methods primarily rely on large-scale image or video generation world models, incurring large computational overhead. Latent world models offer a more efficient alternative but face two challenges: compounding errors during recursive prediction and incompatibility with VLAs' visual input interfaces. In this work, we propose LS-MBPO, a lightweight residual MBRL framework that leverages latent world models to improve upon pretrained VLAs. LS-MBPO introduces two key techniques: (1) Hierarchical Forcing, which combines dense causal supervision with autoregressive feedback to mitigate compounding prediction errors; and (2) Latent Subchunk Rollouts, which leverage action chunks from a frozen VLA policy and a latent residual policy for successive subchunk refinement, bridging the gap bet
    
[^41]: 超越分辨率：生成式区域气候降尺度中的推理时参数化时间一致性

    Beyond Resolution: Inference-Time Parameterized Temporal Consistency in Generative Regional Climate Downscaling

    [https://arxiv.org/abs/2611.27641](https://arxiv.org/abs/2611.27641)

    该论文提出一种噪声条件化教师强制训练方案，让扩散降尺度模型通过以带噪的前一时刻高分辨率场为条件，将时间一致性与集合多样性的权衡变为推理时可调的参数，从而在避免误差累积的情况下生成时间连贯的区域气候降尺度序列。

    

    GCM到RCM的降尺度模拟主要由生成式扩散模型主导，但大多数现有工作仅按单个时间步评估技能，将每个场视为独立的快照。这就留下了一个悬而未决的问题：生成的序列在时间上是连贯的，还是仅仅在独立状态下看似合理。这一空白在次每日分辨率下变得越来越重要，因为下游影响依赖于累积和持续性，而不仅仅是孤立的状态。本研究探讨自回归条件化能否在不产生误差累积和集合扩散塌缩的情况下实现连贯的推演，以及单一模型能否将由此产生的连贯性-多样性权衡转化为可控的推理选择，而非固定的设计承诺。我们引入了一种噪声条件化的教师强制训练方案，将扩散U-Net降尺度模型以带噪声的前一时刻高分辨率状态作为条件，使训练后的模型能够在不同模式之间连续插值（摘要在此处截断）。

    arXiv:2611.27641v1 Announce Type: cross  Abstract: GCM-to-RCM downscaling emulation is dominated by generative diffusion models, but most existing work evaluates skill only per timestep, treating each field as an independent snapshot. This leaves open whether generated sequences are temporally coherent or merely independently plausible, a gap that matters increasingly for sub-daily resolutions since downstream impacts depend on accumulation and persistence, not just on isolated states. This study asks whether autoregressive conditioning can deliver coherent rollouts without error accumulation and ensemble-spread collapse, and whether a single model can expose the resulting coherence-diversity trade-off as a controllable inference choice rather than a fixed design commitment. We introduce a noise-conditioned teacher-forcing training scheme conditioning a diffusion U-Net downscaling model on a noised previous high-resolution state, so the trained model interpolates continuously between a
    
[^42]: 大语言模型智能体中的势能涌现：统计物理学视角

    Potential emergence in LLM agents: a statistical physics perspective

    [https://arxiv.org/abs/2611.27618](https://arxiv.org/abs/2611.27618)

    该论文从统计物理学视角提出将智能体状态作为中间变量，通过配分函数与全局势能来定量测量并预测大语言模型智能体在后训练过程中行为的组织与涌现。

    

    近年来，从人类偏好或语料库统计的角度理解大语言模型智能体的行为引起了极大关注。本文提出将智能体状态作为中间变量，通过这些变量可以对后训练过程进行测量和预测。从统计物理学的视角出发，我们研究了一个支配大语言模型智能体推理中状态转移的全局势能的涌现。我们引入配分函数来建模理想的KL正则化转移分布，并追踪势能结构、平均奖励和状态熵。我们在两个Qwen3.5-9B任务上展示了我们的方法：安全测试选择和物理推理审计。我们观察到，正如我们的框架所预测的，势能结构在后训练过程中不断增强，同时奖励与状态熵的联合变化提供了一致性证据。因此，状态为后训练如何组织智能体行为提供了定量描述。

    arXiv:2611.27618v1 Announce Type: cross  Abstract: There has been significant recent interest in understanding the behaviour of LLM agents from the perspective of human preferences or corpus statistics. In this paper we propose agent states as intermediate variables through which post-training can be measured and predicted. From a statistical physics perspective, we study the emergence of a global potential which governs the transition of state in LLM agent reasoning. We introduce a partition function to model the ideal KL-regularised transition distribution and track potential structure, mean reward and state entropy. We demonstrate our approach in two Qwen3.5-9B tasks: safety-test selection and physics-reasoning auditing. We observe that the potential structure strengthens during post-training as predicted from our framework, while joint changes in reward and state entropy provide consistency evidence. States thus provide a quantitative description of how post-training organises agen
    
[^43]: 循环预测编码解锁有效深度

    Circular Predictive Coding Unlocks Effective Depth

    [https://arxiv.org/abs/2611.27610](https://arxiv.org/abs/2611.27610)

    该论文提出循环预测编码网络，其有效深度直接源自能量最小化过程而非堆叠层，并证明其平衡点收敛近似于深度均衡模型的不动点迭代，从而在不加深网络的情况下解锁了有效深度，在图像分类任务上优于同规模的其他预测编码模型。

    

    预测编码网络通过能量最小化过程执行信用分配。尽管它们能够训练具有任意网络拓扑结构的模型，但它们大多被用于训练前馈架构以匹配反向传播的性能。然而，当模型变得太深时，它们无法做到这一点，因为误差信号在穿过许多层时变得不稳定或消失。我们利用循环网络解决了这一局限性，循环网络的深度直接来自能量最小化，而不是通过堆叠层获得。在形式上，我们证明了循环预测编码网络收敛到其能量泛函的平衡点近似于深度均衡模型的不动点迭代过程，这表明仅凭循环性本身就足以解锁有效深度。在实证方面，我们展示了循环模型（i）在图像分类任务上优于其他相同规模的预测编码模型……

    arXiv:2611.27610v1 Announce Type: new  Abstract: Predictive coding networks perform credit assignment through an energy minimization process. Despite their ability to train models with arbitrary network topologies, they have mostly been used to train feedforward architectures to match the performance of backpropagation. However, they fail to do so when the models become too deep, because error signals become unstable or vanish as they travel through many layers. We address this limitation with circular networks, which get their depth directly from the energy minimization, rather than from stacking layers. Formally, we prove that convergence to an equilibrium of the energy functional of a circular predictive coding network approximates the fixed-point iteration process of a deep equilibrium model, showing that circularity alone suffices to unlock effective depth. Empirically, we show that circular models (i) outperform other predictive coding models of the same size on image classificat
    
[^44]: 量子启发的正则化泊松学习用于基于图的半监督分类

    Quantum-Informed Regularized Poisson Learning for Graph-Based Semi-Supervised Classification

    [https://arxiv.org/abs/2611.27609](https://arxiv.org/abs/2611.27609)

    提出了量子启发的正则化泊松学习框架IPQSSL，利用共享变分量子线路产生节点对角权重以稳定对称化泊松系统，在标签稀缺的图半监督分类任务中取得了优于泊松基线的效果。

    

    我们提出了IPQSSL，一种量子启发的正则化泊松框架，用于标签稀缺情况下的基于图的半监督分类。局部图特征上下文经过QR分解，提升为酉编码，并由一个共享的变分量子线路处理，其测量结果产生逐节点的对角权重，用以稳定对称化的泊松系统。我们在显式谱条件下建立了对称化正则化算子的正定性，并推导出一种厄米特的、与块编码兼容的公式形式，从而将面向经典NISQ设备的实现与未来的容错计算路径分离开来。在涵盖表格数据和一个引文网络的七个基准测试中，正则化的效果依赖于数据集和标签预算：IPQSSL在除单标签预算外的所有预算下，在Iris和Wine数据集上均超越了泊松基线方法，在Iris数据集上每类十个标签时达到0.946的准确率，并在German Credit数据集的所有预算下均有提升……

    arXiv:2611.27609v1 Announce Type: new  Abstract: We introduce IPQSSL, a quantum-informed regularized Poisson framework for graph-based semi-supervised classification under scarce labels. Local graph feature contexts are QR factorized, lifted to unitary encodings, and processed by a shared variational circuit whose measurements yield node wise diagonal weights that stabilize a symmetrized Poisson system. We establish positive definiteness of the symmetrized, regularized operator under explicit spectral conditions and derive a Hermitian, block encoding compatible formulation that separates the classical NISQ oriented implementation from a future fault-tolerant pathway. Across seven benchmarks spanning tabular data and a citation network, the effect of regularization is dataset- and budget-dependent: IPQSSL improves over the Poisson baselines on Iris and Wine at all but the single-label budget, reaching 0.946 accuracy on Iris at ten labels per class and at every budget on German Credit, w
    
[^45]: Tasklets：面向语言模型架构的单元测试基准

    Tasklets: A Unit-Test Benchmark for Language Model Architectures

    [https://arxiv.org/abs/2611.27607](https://arxiv.org/abs/2611.27607)

    Tasklets 是一个由小型合成任务组成的标准化基准，按能力类别（回忆、序列操作、状态跟踪）组织，用于低成本、可重复地评估语言模型序列混合层的基础计算能力。

    

    我们提出了 Tasklets，这是一个由小型合成任务组成的基准，用于评估语言模型序列混合层的计算能力。这些任务探测回忆、序列操作和状态跟踪等基础能力。此类任务在架构研究中被广泛使用，但任务选择、形式化和评估方面的差异使得跨研究比较变得困难。Tasklets 将这些任务汇集在一起，按其测试的能力进行组织，并为评估不同架构层类型如何学习基本计算提供了统一的设置。我们通过对小型模型的受控实验比较了广泛的层类型。我们的任务实现和评估代码支持对新架构提案进行可重复、低成本的测试。

    arXiv:2611.27607v1 Announce Type: new  Abstract: We introduce Tasklets, a benchmark of small synthetic tasks for evaluating the computational capabilities of language model sequence-mixing layers. These tasks probe foundational capabilities such as recall, sequence manipulation, and state tracking. Such tasks are widely used in architecture research, but differences in task selection, formulation, and evaluation make comparisons across studies difficult. Tasklets brings these tasks together, organizing them by the capabilities they test and providing a shared setup for evaluating how different architectural layer types learn fundamental computations. We compare a broad range of layer types through controlled experiments with small models. Our task implementations and evaluation code support reproducible, low-cost testing of new architectural proposals: https://github.com/zaydzuhri/flame/tree/tasklets.
    
[^46]: 将AI天气基础模型适配到变化中的气候

    Adapting AI Weather Foundation Models to a Changing Climate

    [https://arxiv.org/abs/2611.27605](https://arxiv.org/abs/2611.27605)

    提出了一种在线适配框架，通过在低维敏感性子空间内进行流依赖校正并结合残差记忆追踪地理持续性偏差，使AI天气基础模型无需全模型重训练即可持续适应演变的气候分布。

    

    AI天气基础模型从历史档案中学习，而部署时所遇到的气候分布却在持续演变。这种日益加剧的模型-气候失配促使人们进行持续适配而无需反复进行全模型重训练，但天气预报误差的结构决定了此类适配应如何执行。我们开发了一个在线适配框架，利用两种互补的误差结构：地理上持续存在的偏差以及随大气状态变化的流依赖误差。对Aurora-1.3B的分析表明，其预报损失敏感性强烈集中于解码器中，这促使我们在固定的低维敏感性子空间内进行流依赖校正。残差记忆用于跟踪该校正后仍然残留的持续性误差，而变量级在线门控则控制持续性分量的贡献。我们使用1959–1968年的ERA5数据对Aurora-1.3B进行离线评估。

    arXiv:2611.27605v1 Announce Type: new  Abstract: AI weather foundation models learn from historical archives, while the climate distribution encountered during deployment continues to evolve. This growing model--climate mismatch motivates continual adaptation without repeated full-model retraining, but the structure of weather forecast errors determines how such adaptation should be performed. We develop an online adaptation framework that exploits two complementary error structures: geographically persistent biases and flow-dependent errors that vary with the atmospheric state. Analysis of Aurora-1.3B reveals strongly concentrated forecast-loss sensitivity in its decoder, motivating flow-dependent corrections within fixed low-dimensional sensitivity subspaces. A residual memory tracks persistent errors remaining after this correction, while a variable-wise online gate controls the contribution of the persistent component. We evaluate Aurora-1.3B using ERA5 from 1959--1968 for offline 
    
[^47]: 面向顺序量子数据编码的特征条件化块搜索

    Block Search Conditioned on Features for Sequential Quantum Data Encoding

    [https://arxiv.org/abs/2611.27587](https://arxiv.org/abs/2611.27587)

    该论文提出了一种特征条件化的块搜索方法，在固定量子寄存器中逐步联合选择输入特征、操作块、放置位置与旋转权重以扩展Q-FLAIR，在Bars and Stripes数据集上显著提升了分类性能指标。

    

    本文通过在固定量子寄存器内联合选择输入特征与编码块来处理顺序量子数据编码问题。在每一步中，从训练数据中共同选择一个新特征、一个操作块、其放置位置及其旋转权重，从而扩展了Q-FLAIR的增量构建方法。采用直接损失选择（RAW）策略，在五个新的Bars and Stripes数据集上，四个量子比特中保留了九至十二个特征。在相同的最终校准条件下，相对于校准后的Q-FLAIR参考方法，平均变化为：平衡准确率提升5.53个百分点，二元交叉熵（BCE）降低15.83%，ROC曲线下面积（AUC）提升7.79%。五个数据集中的四个以及平均配对差异均满足预先设定的成功标准。这些结果是在统一的32个操作上限下，通过比较完整流程获得的。

    arXiv:2611.27587v1 Announce Type: cross  Abstract: Sequential quantum data encoding is addressed through the joint selection of input features and encoding blocks within a fixed quantum register. At each step, a new feature, a block of operations, its placement, and its rotation weights are selected together from training data, extending the incremental construction of Q-FLAIR. With direct-loss selection (RAW), nine to twelve features were retained in four qubits across five new Bars and Stripes datasets. Under the same final calibration, mean changes relative to the calibrated Q-FLAIR reference were +5.53 percentage points in balanced accuracy, -15.83% in binary cross-entropy (BCE), and +7.79% in area under the ROC curve (AUC). The prespecified success criterion was met in four of five datasets and for the mean paired differences. These results were obtained by comparing complete procedures under a common cap of 32 operations.
    
[^48]: DeepONet主干基预条件子需要训练吗？

    Do DeepONet Trunk-Basis Preconditioners Need Training?

    [https://arxiv.org/abs/2611.27582](https://arxiv.org/abs/2611.27582)

    该研究表明DeepONet主干基预条件子无需训练——未训练的随机初始化分支函数张成的空间近似低次多项式空间，其预条件效果与训练后的版本相当。

    

    主干基（TB）预条件子利用训练好的深度算子网络的分支函数构建双层Schwarz预条件子的粗空间。其有效性一直被归因于频谱偏差，即网络首先学习低频成分，从而与消除高频误差的平滑器形成互补。然而，频谱偏差是训练过程的一种特性，而带有光滑激活函数的随机初始化分支本身就已经是光滑的。因此我们提出疑问：TB预条件子是否真的需要训练？在一维情形下，我们证明了在适当的非退化假设下，每个子域上未训练分支函数的张成空间接近于低次多项式空间。数值实验表明，对于泊松方程和常系数三维线弹性问题，相同维度的未训练TB粗空间与分段多项式粗空间产生相近的共轭梯度迭代次数。

    arXiv:2611.27582v1 Announce Type: cross  Abstract: Trunk-basis (TB) preconditioners build the coarse space of a two-level Schwarz preconditioner from the trunk functions of a trained deep operator network (DeepONet), restricted to subdomains. Their effectiveness has been attributed to spectral bias since networks learn low frequencies first, complementing a smoother that removes the high-frequency error. Spectral bias, however, is a property of training, whereas a randomly initialized trunk with a smooth activation is already smooth. We therefore ask whether TB preconditioners need training. In one dimension, we prove that on each subdomain the span of untrained trunk functions is close to the space of low-degree polynomials under suitable nondegeneracy assumptions. Numerically, for the Poisson equation and three-dimensional linear elasticity with constant coefficients, untrained TB and piecewise polynomial coarse spaces of equal dimension yield close conjugate gradient iteration count
    
[^49]: PRISM：基于预条件锥形幽灵匹配的大语言模型微调无验证在线批选择

    PRISM: Validation-Free Online Batch Selection for LLM Fine-Tuning via Preconditioned Conic Ghost Matching

    [https://arxiv.org/abs/2611.27578](https://arxiv.org/abs/2611.27578)

    PRISM提出了一种无需验证集的大语言模型微调在线批选择方法，通过预条件化的锥形幽灵匹配将子集选择建模为归一化的单调子模锥形设施选址问题，避免逐样本梯度显式计算的同时保证 (1-1/e) 的贪心近似效果。

    

    在线批选择通过将每个候选批次替换为更小的信息子集，从而降低大语言模型（LLM）微调的成本。现有选择器通常依赖预留的验证集或代理集，否则依赖于未预条件化梯度上的欧几里得设施选址目标，其相似度会因重尾梯度范数和跨层各向异性而失真。我们提出PRISM（Preconditioned Ray-Isotropic Submodular Matching，预条件射线各向同性子模匹配），这是一种无需验证集的在线批选择器，它在分解的二阶矩度量中对齐候选梯度，并将子集构建转化为正射线投影上的锥形设施选址问题。通过分解的预条件幽灵内积计算所需的成对几何结构，避免了逐样本参数梯度的显式构建，同时锥形目标保持归一化、非负且单调子模——保证了 (1-1/e) 的贪心近似（摘要在此处截断）。

    arXiv:2611.27578v1 Announce Type: new  Abstract: Online batch selection lowers the cost of fine-tuning large language models (LLMs) by replacing each candidate batch with a smaller informative subset. Existing selectors typically depend on a held-out validation or proxy set, or else rely on Euclidean facility-location objectives over unpreconditioned gradients whose diameter-shifted similarities are distorted by heavy-tailed gradient norms and cross-layer anisotropy. We introduce PRISM (Preconditioned Ray-Isotropic Submodular Matching), a validation-free online batch selector that aligns candidate gradients in a factored second-moment metric and casts subset construction as conic facility location over positive ray projections. Computing the required pairwise geometry through factored preconditioned ghost inner products avoids materializing per-example parameter gradients, while the conic objective remains normalized, nonnegative, and monotone submodular---guaranteeing a $(1-1/e)$ gree
    
[^50]: 在线凸优化中的极小极大最优小损失遗憾及其在上下文搜索与在线逆优化中的应用

    Minimax-Optimal Small-Loss Regret in Online Convex Optimization and Applications to Contextual Search and Online Inverse Optimization

    [https://arxiv.org/abs/2611.27572](https://arxiv.org/abs/2611.27572)

    该论文提出了一个在线凸优化算法，在每轮仅需一个次梯度、不知道损失值和时间范围、无平滑性假设的条件下，实现了极小极大最优的 $O(1+\sqrt{\Delta_T^\star})$ 小损失遗憾界，并成功应用于上下文搜索和在线逆优化问题。

    

    我们为在线凸优化（OCO）给出了小损失遗憾界，每轮仅需使用一个次梯度。设 $\Delta_T^\star$ 表示事后最优固定决策相对于每轮最小值度量的累积损失。我们的算法在不知道 $\Delta_T^\star$ 的情况下即可达到 $O(1+\sqrt{\Delta_T^\star})$ 的遗憾界（最多相差定义域与梯度尺度因子）。该遗憾界是极小极大最优的，包括其对定义域的依赖性，对所有 $1\le p\le\infty$ 的 $\ell_p$ 球均成立。我们既不假设平滑性，也不假设梯度范数受损失值约束的条件，且既不需要损失值，也不需要时间范围。在算法层面，我们通过遵循正则化领导者（follow-the-regularized-leader）框架和多个基础学习器的混合，将 Gupta 等人（2026）最初为在线逆优化开发的度量更新撤销技术扩展到一般 OCO。我们还将该算法应用于上下文搜索和在线逆优化。

    arXiv:2611.27572v1 Announce Type: new  Abstract: We give small-loss regret bounds for online convex optimization (OCO), using one subgradient per round. Let $\Delta_T^\star$ denote the cumulative loss of the best fixed decision in hindsight, measured relative to the per-round minima. Our algorithm achieves the regret bound $O(1+\sqrt{\Delta_T^\star})$ without knowing $\Delta_T^\star$, up to domain and gradient-scale factors. The bound is minimax optimal, including its domain dependence, on $\ell_p$ balls for every $1\le p\le\infty$. No smoothness or condition bounding gradient norms by loss values is assumed, and neither loss values nor the time horizon are needed. Algorithmically, we extend the metric-update revocation of Gupta et al. (2026), originally developed for online inverse optimization, to general OCO using the follow-the-regularized-leader framework and a mixture of multiple base learners. We also apply our algorithm to contextual search and online inverse optimization. In $
    
[^51]: WEIR：面向多轮智能体强化学习的分水岭探索与漂移感知前缀收缩

    WEIR: Watershed Exploration and Drift-Aware Prefix Shrinkage for Multi-Turn Agent Reinforcement Learning

    [https://arxiv.org/abs/2611.27570](https://arxiv.org/abs/2611.27570)

    提出基于Doob轮次方差守恒的WEIR框架，通过分水岭探索与漂移感知前缀收缩机制，解决多轮智能体强化学习中稀疏奖励下难以区分关键决策及样本浪费的问题。

    

    多轮语言模型智能体通常只在长交互轨迹结束时收到单一的二元奖励，这使得难以将决定性决策与常规步骤区分开来，也难以估计准确的中介价值。前缀树方法通过合并共享动作历史的轨迹来提高样本效率，但仍存在两个结构性缺陷：朴素的池化以固定权重将过时的离策略结果混入评论家目标中，而均匀或熵驱动的分支则会将后续展开浪费在结果已经注定的死端状态上。我们提出了WEIR（分水岭探索与内在漂移收缩），这是一个基于Doob轮次方差守恒恒等式的演员-评论家强化学习框架。WEIR维护一棵具有指数折扣访问计数的跨迭代规范动作前缀树，并训练一个以前缀为条件的评论家，使其趋向于一个自适应的最小最大漂移感知收缩目标……

    arXiv:2611.27570v1 Announce Type: new  Abstract: Multi-turn language model agents often receive only a single binary reward at the end of a long interaction trajectory, making it difficult to separate decisive decisions from routine steps or estimate accurate intermediate values. Prefix-tree methods improve sample efficiency by merging trajectories that share action histories, yet two structural flaws remain: naive pooling mixes stale off-policy outcomes into critic targets with fixed weights, and uniform or entropy-driven branching wastes continuations on dead-end states whose outcome is already settled. We introduce WEIR (Watershed Exploration and Intrinsic-Drift Shrinkage), an actor-critic reinforcement learning framework grounded in a Doob turn-variance conservation identity. WEIR maintains a cross-iteration canonical action-prefix tree with exponentially discounted visit counts and trains a prefix-conditioned critic toward a minimax drift-aware shrinkage target that adapts from si
    
[^52]: TITAN：面向触觉互联网的基于深度强化学习的动态调度与资源分配

    TITAN: DRL-Based Dynamic Scheduling and Resource Allocation for Tactile Internet

    [https://arxiv.org/abs/2611.27560](https://arxiv.org/abs/2611.27560)

    TITAN是一种基于深度强化学习的双时间尺度动态调度与资源分配方案，可根据时变的触觉QoS需求自适应地在触觉流与视频流之间划分带宽并分配资源块，同时满足两类业务的QoS目标并带来7.64个百分点的性能提升。

    

    触觉互联网（TI）应用需要超可靠、低延迟的触觉通信与视频流量并行传输。然而，触觉服务质量（QoS）需求会随交互情境而变化，这使得固定的资源分配方式效率低下。本文提出了TITAN（5G接入网络上的双时间尺度集成触觉互联网），这是一种动态调度与资源分配（SRA）方案。TITAN从上层模块接收时变的触觉QoS配置，并根据当前的QoS配置和网络状态自适应地调整调度与资源分配。基于深度强化学习的智能体在慢时间尺度上对触觉流和视频流进行网络带宽划分，同时快时间尺度调度器则在活跃用户设备之间分配资源块。仿真结果表明，TITAN能够同时满足触觉和视频的QoS目标，并在联合……方面实现了7.64个百分点的性能提升。（注：原文摘要末尾被截断）

    arXiv:2611.27560v1 Announce Type: cross  Abstract: Tactile Internet (TI) applications require ultra-reliable, low-latency tactile communication alongside video traffic. However, tactile Quality of Service (QoS) requirements vary with the interaction context, making fixed resource allocation inefficient. In this paper, we propose TITAN (Two-Timescale Integrated Tactile Internet over 5G Access Networks), a dynamic scheduling and resource allocation (SRA) scheme. TITAN receives time-varying tactile QoS profiles from a higher-layer module and adapts SRA to the current QoS profile and network state. A deep reinforcement learning-based agent splits the available network bandwidth between tactile and video streams at a slow timescale, while a fast-timescale scheduler allocates resource blocks among active user equipment. Through simulations, we show that TITAN simultaneously satisfies both tactile and video QoS targets and achieves a performance gain of 7.64 percentage points (pp) in the join
    
[^53]: 关于机器遗忘中覆盖率评估的可观测性极限

    On the Limits of Observability for Evaluating Coverage in Machine Unlearning

    [https://arxiv.org/abs/2611.27543](https://arxiv.org/abs/2611.27543)

    该研究通过502个重训练模型的可观测性压力测试和Ω(log n)秩下界的理论证明，揭示了现有机器遗忘评估因只关注特定遗忘集而忽视“覆盖率”的问题，并提出了评估遗忘覆盖率的CEO框架。

    

    机器遗忘旨在移除任意被请求的训练数据子集的影响，同时保留模型在剩余数据上的效用。现有实证评估通常通过模型输出或中间激活来衡量预测准确率并进行训练集成员推断（MIA）。然而，这些评估由于只聚焦于选定的遗忘集，可能忽视“覆盖率”——即一个方法所能满足的请求范围。为了评估这种覆盖率，我们首先利用502个重训练模型对可观测性进行了压力测试，发现隐藏嵌入能够揭示在预测中被遗漏的已知删除操作。为刻画覆盖率的容量极限，我们证明用线性分数区分所有n个单例删除可能需要秩为Ω(log n)的共享表示。我们进一步将该下界扩展到任意子集，并阐明了部分删除何时可能显得完全。基于这一分析，我们提出了CEO，一个用于评估机器遗忘覆盖率的框架。

    arXiv:2611.27543v1 Announce Type: cross  Abstract: Machine unlearning aims to remove the influence of any requested subset of training data while preserving utility on the remaining data. Empirical evaluations measure predictive accuracy and infer training-set membership (MIAs) from model outputs or intermediate activations. However, by focusing on selected forget sets, these evaluations can overlook Coverage, the range of requests a method can fulfill. To assess this coverage, we first conduct a stress test of observability with 502 retrained models, finding that hidden embeddings reveal known deletions missed in predictions. To characterize capacity limits on coverage, we show that distinguishing all $n$ singleton deletions with linear scores can require a shared representation of rank $\Omega(\log n)$. We further extend the bound to arbitrary subsets and show when partial deletion can appear complete. Building on this analysis, we propose CEO, a framework for evaluating unlearning c
    
[^54]: PaMuk-Bench：一个面向视觉-语言任务的平行多语言多任务基准

    PaMuk-Bench: A Parallel Multilingual Multitask Benchmark for Vision-Language Tasks

    [https://arxiv.org/abs/2611.27521](https://arxiv.org/abs/2611.27521)

    本文提出了PaMuk-Bench，一个包含超过10K条平行翻译提示（涵盖阿拉伯语、丹麦语、法语、印地语、意大利语和罗马尼亚语六种语言）的多语言多任务视觉-语言基准，用于系统评估视觉-语言模型在低资源语言上跨多种任务的表现。

    

    尽管视觉-语言模型（VLM）目前在英语等被广泛研究的语言上取得了显著成果，但由于缺乏标注数据集，它们在某些低资源语言上表现出显著的性能下降。为此，我们提出了PaMuk-Bench，一个新颖的面向视觉-语言任务的平行多语言多任务基准。我们的基准包含一组超过10K条精选的英文提示，这些提示被翻译成六种语言：阿拉伯语、丹麦语、法语、印地语、意大利语、罗马尼亚语。通过半自动过程获得的平行翻译，以及配套的生成和检索图像，使得能够在广泛的任务上对VLM进行对比性开发和评估，这些任务包括：文本到图像生成、图像描述生成、基于内容的图像检索、图像生成难度估计、图像检索难度估计。我们通过实验证明，零样本英语和多语言预训练……

    arXiv:2611.27521v1 Announce Type: cross  Abstract: While vision-language models (VLMs) currently obtain notable results for highly-studied languages, e.g. English, they exhibit significant performance drops in certain low-resource languages, especially due to the lack of annotated datasets. To this end, we propose PaMuk-Bench, a novel parallel multilingual multitask benchmark for vision-language tasks. Our benchmark includes a set of over 10K curated English prompts that are translated into six languages: Arabic, Danish, French, Hindi, Italian, Romanian. Parallel translations, which are obtained via a semi-automatic process, along with accompanying generated and retrieved images enable the comparative development and evaluation of VLMs across a wide range of tasks: text-to-image generation, image captioning, content-based image retrieval, image generation difficulty estimation, image retrieval difficulty estimation. We empirically show that zero-shot English and multilingual pre-traine
    
[^55]: 面向工业排序中乘积与比值特征交互的深度交叉-相除网络

    Deep Cross-Division Network for Product-and-Ratio Feature Interactions in Industrial Ranking

    [https://arxiv.org/abs/2611.27514](https://arxiv.org/abs/2611.27514)

    本文提出深度交叉-相除网络DCDN，通过在DCNv2交叉层中引入相除分支，使模型能够联合学习乘积、比值及混合型特征交互，并将其表达能力刻画为ℓ1阶最高达K+1、带符号次数介于-K与K+1之间的多元Laurent多项式。

    

    学习有效的特征交互对工业推荐系统至关重要。DCN和DCNv2能够高效地自动化多项式特征交叉。然而，它们的交叉层仅使用输入特征的非负整数次幂来构建交互，最高次数由网络深度控制。因此，除非将此类特征显式地作为输入提供，否则它们无法直接表示比值、比率和归一化计数所需的倒数项。我们提出了DCDN（深度交叉-相除网络），它在DCNv2风格的交叉层基础上增加了一个相除分支。DCDN能够联合学习乘积、比值以及乘积-比值混合交互。我们将可表示的交互刻画为多元Laurent多项式，并证明K层DCDN可生成ℓ1阶最高达K+1的Laurent单项式，其带符号次数介于-K与K+1之间。我们将DCDN与生产环境优化的神经（模型进行了对比评估）。

    arXiv:2611.27514v1 Announce Type: cross  Abstract: Learning effective feature interactions is essential for industrial recommender systems. DCN and DCNv2 efficiently automate polynomial feature crossing. However, their cross layers construct interactions using only non-negative integer powers of the input features, with the maximum degree controlled by network depth. Consequently, they cannot directly represent the reciprocal terms required for ratios, rates, and normalized counts unless such features are explicitly provided as inputs. We introduce DCDN, a Deep Cross-Division Network that augments a DCNv2-style cross layer with a division branch. DCDN jointly learns product, ratio, and mixed product-ratio interactions. We characterize the representable interactions as multivariate Laurent polynomials and show that a $K$-layer DCDN generates Laurent monomials of $\ell_1$-order up to $K+1$, with signed degree between $-K$ and $K+1$. We evaluate DCDN against a production-optimized neural-
    
[^56]: 突破困难样本的壁垒：一种面向医疗任务的以数据为中心的合成数据方法

    Breaking the Barrier of Hard Samples: A Data-Centric Approach to Synthetic Data for Medical Tasks

    [https://arxiv.org/abs/2611.27511](https://arxiv.org/abs/2611.27511)

    本文提出以数据为中心的Profile2Gen框架，通过生成与改进合成数据来解决医学回归任务中难以学习的样本问题，并经过约18,000项实验验证，证明改进后的合成样本可降低预测误差并提升模型可靠性。

    

    数据稀缺和质量问题仍然是医学研究中开发稳健预测模型的重要障碍。传统上对真实世界数据的依赖往往导致模型存在偏差，在多样化的患者群体中泛化能力较差。合成数据生成已成为一种有前景的解决方案，但这些样本的代表性及其有效利用方面的挑战依然存在。本文提出了Profile2Gen，这是一个新颖的以数据为中心的框架，旨在指导合成数据的生成与改进，重点解决回归任务中难以学习的样本。我们进行了约18,000项实验，利用七种最先进的生成模型，在六个医学数据集上验证了该方法的有效性。结果表明，经过改进的合成样本能够降低预测误差并增强模型可靠性。此外，我们还将DataIQ框架泛化以支持回归任务。

    arXiv:2611.27511v1 Announce Type: new  Abstract: Data scarcity and quality issues remain significant barriers to developing robust predictive models in medical research. Traditional reliance on real-world data often leads to biased models with poor generalizability across diverse patient populations. Synthetic data generation has emerged as a promising solution, yet challenges related to these samples' representativeness and effective utilization persist. This paper introduces Profile2Gen, a novel data-centric framework designed to guide the generation and refinement of synthetic data, focusing on addressing hard-to-learn samples in regression tasks. We conducted approximately 18,000 experiments to validate its effectiveness across six medical datasets, utilizing seven state-of-the-art generative models. Results demonstrate that refined synthetic samples can reduce predictive errors and enhance model reliability. Additionally, we generalize the DataIQ framework to support regression ta
    
[^57]: 基于异步事件的自监督连续时间深度学习与自运动估计

    Self-Supervised Continuous-Time Depth Learning and Ego-Motion Estimation from Asynchronous Events

    [https://arxiv.org/abs/2611.27509](https://arxiv.org/abs/2611.27509)

    该论文提出了一种自监督的连续时间深度学习方法，为每个异步事件分配独立深度并从同一观测中恢复相机自运动，利用事件间的几何一致性作为监督信号，并通过可微渲染生成目标时刻的稠密深度图，在MVSEC数据集上超越了先前仅基于事件的自监督方法。

    

    事件相机以微秒级精度逐像素地记录亮度变化，而可用于监督它们的稠密深度数据获取成本高昂，且仅存在于孤立的时刻。将事件流压缩成帧虽然便于学习，但代价是牺牲了事件感知赖以独特的高时间分辨率特性。我们提出了一种自监督方法，其中每个事件在连续时间中被赋予其自身的深度，同时相机运动也从相同的观测中恢复出来。监督信号源于几何：一个事件通过其预测深度被提升到三维空间，并经过它在触发瞬间的相机运动位移后，应当与由同一条边缘产生的其他事件相重合，因此任何错位都会暴露深度或运动估计中的误差。随后，可微渲染将这些逐事件的估计转化为每个窗口内目标时刻的稠密深度图，并忠实于其背后的事件。在MVSEC数据集上，该方法优于先前仅基于事件的自监督方法。

    arXiv:2611.27509v1 Announce Type: cross  Abstract: Event cameras register brightness changes pixel by pixel with microsecond precision, whereas the dense depth that could supervise them is costly to acquire and exists only at isolated instants. Condensing the stream into frames eases learning, but at the expense of the very timing that sets event sensing apart. We propose a self-supervised approach in which each event is assigned its own depth in continuous time, and the camera motion is recovered from the same observations. Supervision arises from geometry: lifted into 3D by its predicted depth and displaced by the camera motion at the instant it fired, an event should coincide with others produced by the same edge, so any misalignment exposes an error in depth or motion. Differentiable rendering then turns these per-event estimates into dense depth maps at target times within each window, faithful to the events behind them. On MVSEC, the approach outperforms prior event-only self-sup
    
[^58]: Turnout：面向高效推理的自适应注意力切换方法

    Turnout: Adaptive Attention Switching for Efficient Reasoning

    [https://arxiv.org/abs/2611.27500](https://arxiv.org/abs/2611.27500)

    提出Turnout方法，在预训练全注意力模型中通过蒸馏保留线性注意力路径，并利用强化学习训练的自适应策略在每一层动态切换全注意力与线性注意力，从而在大幅降低内存消耗的同时保持长文本推理任务的准确率。

    

    事后transformer线性化用固定大小的状态取代全注意力中随生成长度线性增长的键值缓存。这一改变降低了内存消耗，但往往会损害准确率，而在需要长文本生成的推理任务中，这种影响尤为显著。在本工作中，我们提出了Turnout，一种能够在保持线性化内存缩减优势的同时缓解其准确率损失的方法。我们将事后线性化应用于预训练的全注意力模型，通过蒸馏训练出一条线性路径，同时保持原有的全注意力路径不变。在响应生成过程中，所得的混合模型同时保留两条路径，并利用一个学习到的策略来控制每一层何时从全注意力切换到线性注意力。该策略通过强化学习进行训练，奖励函数在答案正确性与内存使用之间取得平衡。对我们方法的实证评估表明……

    arXiv:2611.27500v1 Announce Type: cross  Abstract: Post-hoc transformer linearization replaces the key-value cache of full attention, which grows linearly with generation length, with a fixed size state. This change reduces memory consumption but often hurts accuracy. In reasoning tasks, which require long generations, this effect is especially pronounced. In this work we introduce Turnout, a method that mitigates accuracy loss as a result of linearization, while maintaining its memory-reduction benefits. We apply post-hoc linearization to a pretrained full-attention model, training a linear path through distillation while leaving the original full-attention path unchanged. During response generation, the resulting hybrid model retains both paths and uses a learned policy to control when each layer switches from full to linear attention. We train the policy with reinforcement learning using a reward that balances answer correctness and memory use. Empirical evaluation of our method sho
    
[^59]: 面向成对约束聚类认证的约束感知反聚类分解

    Constraint-Aware Anticlustering Decomposition for Certifying Pairwise-Constrained Clustering

    [https://arxiv.org/abs/2611.27496](https://arxiv.org/abs/2611.27496)

    该论文提出了一种约束感知的反聚类分解框架，通过引入约束反聚类问题（CAP）并构建为混合整数二阶锥规划模型，将约束图的连通分量分配到反聚类中，从而为带成对约束的大规模最小平方和聚类问题提供有效的认证下界。

    

    对于带有成对必须连接与不能连接约束的大规模最小平方和聚类问题，认证其可行解的质量仍然具有挑战性，因为这需要对最优目标值有较强的下界。我们提出了一个分解框架，将数据点划分为更小的子集（称为反聚类），并求解由此产生的聚类子问题，从而为原问题获得认证的最优性间隙。为了保留成对约束信息，我们引入了约束反聚类问题（CAP），该问题将约束图的每个连通分量分配给单个反聚类，同时寻求几何上具有代表性的子集。我们将CAP表述为混合整数二阶锥规划，并开发了变量邻域搜索启发式算法来构建高质量的分解。即使诱导出的子问题未被求解至全局最优，所得的下界依然有效。

    arXiv:2611.27496v1 Announce Type: cross  Abstract: Certifying the quality of feasible solutions to large-scale minimum sum-of-squares clustering with pairwise must-link and cannot-link constraints remains challenging, as it requires strong lower bounds on the optimal objective value. We propose a decomposition framework that partitions the data points into smaller subsets, called anticlusters, and solves the induced clustering subproblems to obtain a certified optimality gap for the original problem. To preserve pairwise information, we introduce the Constrained Anticlustering Problem (CAP), which assigns each connected component of the constraint graph to a single anticluster while seeking geometrically representative subsets. We formulate CAP as a mixed-integer second-order cone program and develop a variable neighborhood search heuristic to construct high-quality decompositions. The resulting lower bound remains valid even when the induced subproblems are not solved to global optima
    
[^60]: 从老虎机反馈中学习未知振荡器

    Learning Unknown Oscillators From Bandit Feedback

    [https://arxiv.org/abs/2611.27492](https://arxiv.org/abs/2611.27492)

    提出了一种带检验的承诺控制器，在未知振荡器驱动的半老虎机问题上实现了平方根级后悔界，并将拟合次数从1023次降至49次，带来4.69倍的加速。

    

    我们研究由未知振荡器驱动的半老虎机问题。每一轮，学习者从四个奖励通道中选择两个并观察各自的奖励。频率误差会改变未来的奖励排序以及由此产生的测量结果。我们展示了何时承诺于一个已拟合的振荡器，可以让其贪心选择暴露预测误差。我们的控制器在连续置信集上界定奖励损失（后悔值），并在预算内保持最长的时间块。当检验失败时，在线回归策略会收集观测数据。在所有频率下，后悔值以平方根速率增长（最多相差对数因子），并且该方法可扩展到两个相互竞争的振荡器。在96个新的合成实例上，带检验的承诺方法仅使用49次拟合而非1023次，平均后悔值仅比每轮拟合高1.5%。在匹配的独立计时下实现了4.69倍的加速。该置信集还能保证轨迹后悔值。

    arXiv:2611.27492v1 Announce Type: cross  Abstract: We study semi-bandits driven by unknown oscillators. Each round, a learner selects two of four reward channels and observes their individual rewards. Frequency errors change future reward rankings and the resulting measurements. We show when committing to one fitted oscillator lets its greedy choices expose prediction errors. Our controller bounds reward loss (regret) over a continuous confidence set and holds the longest block within budget. An online regression policy gathers observations when the check fails. Regret grows at a square-root rate, up to logarithmic factors, across all frequencies and extends to two competing oscillators. On 96 fresh synthetic instances, checked commitment uses 49 fits instead of 1,023, with 1.5% higher mean regret than fitting every round. Matched isolated timings give a 4.69-fold speedup. The confidence sets also certify trajectory regret.
    
[^61]: 验证与强化神经正切核理论

    Verifying and Strengthening Neural Tangent Kernel Theory

    [https://arxiv.org/abs/2611.27487](https://arxiv.org/abs/2611.27487)

    该论文首次用 Lean 4 对两层网络的神经正切核理论进行了端到端的机器验证形式化证明，证明了梯度流的存在唯一性与残差指数收敛定理，并在此过程中强化了标准理论。

    

    神经正切核（NTK）理论解释了为什么通过梯度方法训练的超宽神经网络会在懒惰训练模式下收敛。其标准证明结合了矩阵集中不等式、随机 Gram 矩阵、连续时间动力学、谱扰动和自举论证，但一直缺乏机器可检验的验证。我们提出了针对两层网络的 Lean 4 形式化证明，证明了一个端到端定理：对于光滑有界激活函数（以 σ=tanh 实例化）、冻结偏置、成对 ±1 初始化以及显式的宽度条件，梯度流以至少 1−δ 的概率唯一存在，并且满足 ||r(t)||² ≤ ||y||² e^(−λ₀t/2)。唯一的数据假设是总体 NTK 非退化性，即 λ₀>0。初始强制性、核漂移、移动自举、残差 ODE 以及流的存在性与唯一性均得到了形式化证明。该形式化还强化了标准……

    arXiv:2611.27487v1 Announce Type: new  Abstract: Writing   Neural tangent kernel (NTK) theory explains why very wide neural networks trained by gradient methods converge in the lazy-training regime. Its standard proof combines matrix concentration, random Gram matrices, continuous-time dynamics, spectral perturbation, and a bootstrap argument, but has lacked machine-checked verification. We present a Lean\~4 formalisation for two-layer networks, proving an end-to-end theorem: for a smooth bounded activation (instantiated with $\sigma=\tanh$), frozen biases, paired $\pm 1$ initialisation, and explicit width conditions, gradient flow exists uniquely with probability at least $1-\delta$, and   $$ \|r(t)\|^2 \leq \|y\|^2 e^{-\lambda_0 t/2}. $$   The sole data assumption is population NTK non-degeneracy, $\lambda_0>0$. Initial coercivity, kernel drift, the movement bootstrap, the residual ODE, and flow existence and uniqueness are formally proved.   The formalisation also sharpens the stand
    
[^62]: 量子机器学习中控制指数集中的实用见解

    Practical insights into controlling exponential concentration in quantum machine learning

    [https://arxiv.org/abs/2611.27486](https://arxiv.org/abs/2611.27486)

    该论文在基于期望值方差的统一框架下实证比较了三类量子机器学习架构中的指数集中现象，发现全局保真度核因希尔伯特空间维度而强烈集中，而测量局域性、纠缠调控和储备池深度等设计手段可以缓解但无法完全消除这一效应。

    

    量子机器学习（QML）模型常常受到指数集中现象的限制，即随着系统规模的增大，量子可观测量对输入变化变得不敏感。在这项工作中，我们在一个基于期望值方差的统一可观测框架内，对这一现象在保真度量子核（FQK）、投影量子核（PQK）和量子极限学习（QEL）三类架构中的表现进行了比较实证研究。我们研究了测量局域性、纠缠和电路表达能力如何在不同架构中塑造集中现象。基于全局保真度的核函数表现出由希尔伯特空间维度驱动的强集中现象，而局域可观测方法可以缓解但无法完全消除这一效应。在PQK和QEL中，集中现象主要由纠缠和量子动力学表达能力决定，其中储备池深度成为QEL架构中的一个关键因素。我们进一步研究了带宽……

    arXiv:2611.27486v1 Announce Type: cross  Abstract: Quantum Machine Learning (QML) models are often limited by exponential concentration, where quantum observables become insensitive to input variations as system size increases. In this work, we provide a comparative empirical study of this phenomenon across Fidelity Quantum Kernels (FQK), Projected Quantum Kernels (PQK), and Quantum Extreme Learning (QEL), within a unified observable-based framework based on expectation-value variance. We investigate how measurement locality, entanglement, and circuit expressivity shape concentration across different architectures. Global fidelity-based kernels exhibit strong concentration driven by Hilbert space dimensionality, whereas local-observable approaches reduce but do not eliminate this effect. In PQK and QEL, concentration is primarily governed by entanglement and quantum dynamical expressivity, with reservoir depth emerging as a key factor in QEL architectures. We further study bandwidth sc
    
[^63]: 迫不及待去验证：通过并行草稿生成消除推测解码中验证器的空闲时间

    Cannot Wait to Verify: Parallel Drafting to Eliminate Verifier Idle Time in Speculative Decoding

    [https://arxiv.org/abs/2611.27477](https://arxiv.org/abs/2611.27477)

    提出一种即插即用的并行草稿生成方法，通过跨Transformer层并行推理并结合隐藏状态散度一致性验证与纠正机制，消除了推测解码中验证器的空闲等待时间，在无需修改架构或重新训练的情况下加速草稿生成并提升硬件利用率。

    

    大型Transformer模型中的自回归推理受限于token级的顺序执行，导致高延迟和低GPU利用率。推测解码通过引入草稿模型-验证器模型的组合来缓解这一瓶颈。然而，草稿生成阶段仍然是顺序执行的，导致验证器处于空闲状态，计算资源未得到充分利用。为解决这一问题，我们提出了一种即插即用的自适应草稿生成方法，在草稿生成过程中跨Transformer层并行化推理。从共享的隐藏状态出发，我们的方法并行评估多个层，并通过测量隐藏状态空间中的散度来验证一致性。当偏差较小时，逐层草稿状态被接受；否则，将应用纠正步骤。通过利用Transformer层间的冗余性，我们的方法加速了草稿生成并提高了硬件利用率，且无需修改模型架构或进行重新训练。

    arXiv:2611.27477v1 Announce Type: new  Abstract: Autoregressive inference in large transformer models is limited by token-level sequential execution, resulting in high latency and low GPU utilization. Speculative decoding (SD) mitigates this bottleneck by introducing a draft-and-verifier model. However, the drafting stage remains sequential, leaving the verifier idle and underutilizing compute. To address this, we propose a plug-and-play drafting adaptive drafting method that parallelizes inference across transformer layers during draft generation. Starting from a shared hidden state, our approach evaluates multiple layers in parallel and verifies consistency by measuring divergence in hidden-state space. Layer-wise draft states are accepted when deviations are small; otherwise, a correction step is applied. By exploiting redundancy across transformer layers, our method accelerates draft generation and improves hardware utilization without modifying architectures or requiring retrainin
    
[^64]: 学习何时求助：面向小型语言模型智能体的顾问感知后训练

    Learning When to Ask: Advisor-Aware Post-Training for SLM Agents

    [https://arxiv.org/abs/2611.27443](https://arxiv.org/abs/2611.27443)

    提出ADAPT多阶段后训练方法，让小型语言模型智能体学会在成本感知下判断何时向更强模型求助并跨步骤遵循建议，且完全依靠环境结果监督的策略自身rollouts进行学习，无需奖励模型或人工标注。

    

    小型语言模型（SLM）因其低成本、高吞吐量的特点，在智能体部署中越来越具有吸引力，但它们在长程任务所需的多步推理和工具使用方面仍然薄弱。我们研究如何对SLM进行后训练，使其能够向更强的模型寻求建议并自己执行这些建议，即遵循“顾问-执行者”模式。然而开箱即用时，这种模式在开源和闭源的各种小模型家族中均告失败：仅仅将SLM连接到顾问上只是一个框架设计问题，本身并不足够。需要一种后训练方案来教会模型以成本感知的方式判断何时提问，以及如何跨步骤遵循建议。我们的方法ADAPT（ADvisor-Aware Post-Training，顾问感知后训练）是一种多阶段训练方案，能够培养这种行为，它完全从策略自身生成的rollouts中学习，并由环境的最终结果进行监督，无需学习奖励模型或人工标注。在三个领域中，与未训练的框架基线相比，ADAPT……

    arXiv:2611.27443v1 Announce Type: cross  Abstract: Small language models (SLMs) are increasingly attractive for cheap, high-throughput agentic deployment, but they remain weak at multi-step reasoning and tool use that long-horizon tasks demand. We study how to post-train an SLM to seek advice from a stronger model and act on that advice itself, following an advisor-executor pattern. Out of the box, this pattern fails across open- and closed-source small-model families: wiring an SLM to an advisor is a harness-design matter that is not enough on its own. A post-training recipe is needed to teach the model when to ask, in a cost-aware fashion, and how to follow advice across steps. Our method, ADAPT (ADvisor-Aware Post-Training), is a multi-stage recipe that instills this behavior, learning entirely from the policy's own rollouts supervised by the environment's outcome, with no learned reward model or human annotation. Across three domains, compared to an untrained harness baseline, ADAP
    
[^65]: THRIFTY：面向小语言模型的成本受限推理数据合成

    THRIFTY: Cost-Constrained Reasoning-Data Synthesis for Small Language Models

    [https://arxiv.org/abs/2611.27436](https://arxiv.org/abs/2611.27436)

    提出THRIFTY——一种基于Thompson采样的带背包约束多臂老虎机控制器，在固定的生成token预算下自适应地将预算分配给不同难度的问题，从而低成本地为小语言模型合成多样且正确的推理训练数据。

    

    小语言模型（SLM）因其在高效、专业化部署方面的吸引力与日俱增，通常在从更大的教师大语言模型（LLM）蒸馏得到的推理轨迹上进行训练。生成这类合成数据是该训练流程中的主要成本，尤其当教师模型是前沿模型时。然而，现有的数据收集方法大多不考虑成本，或者至多将固定的采样预算均匀分配到各个查询上。这种分配方式忽略了问题难度的巨大差异，以及由此引发的策略性过度采样或采样不足。因此，我们提出THRIFTY，一种基于Thompson采样的多臂老虎机控制器，它在固定的生成token预算内收集多样且正确的推理轨迹，将这一任务建模为带背包约束的多臂老虎机问题，并引入了新颖的多样性感知目标。该方法通过在线Beta后验跟踪每个问题的成功概率，并将每个token花费在预期产生最多新的、不同的正确解之处（摘要在此处截断）。

    arXiv:2611.27436v1 Announce Type: cross  Abstract: Small language models (SLMs) are increasingly attractive for efficient, specialized deployment, and are typically trained on reasoning trajectories distilled from a larger teacher LLM. Generating this synthetic data is a major cost of the pipeline, especially when the teacher is a frontier model. Existing collection methods, however, are largely cost-unaware, or at best spread a fixed sampling budget uniformly across queries. Such allocation ignores the wide variation in problem difficulty and the strategic over- or under-sampling it invites. Thus, we propose THRIFTY, a Thompson-sampling bandit controller that spends a fixed generation-token budget to collect diverse correct reasoning trajectories, casting the task as a bandits-with-knapsacks problem with a novel diversity-aware objective. It tracks each problem's success probability with an online Beta posterior and spends each token where it expects the most new distinct correct solu
    
[^66]: 基于学习的网络动态用于事故情境下的模型预测交通信号控制

    Learned Network Dynamics for Model Predictive Traffic Signal Control under Incidents

    [https://arxiv.org/abs/2611.27435](https://arxiv.org/abs/2611.27435)

    提出GNN-MPC控制器，利用图神经网络学习动作条件化的网络动态并结合近似最优响应协调信号相位，在事故干扰下有效降低旅行时间并增加完成的出行次数。

    

    城市交通信号的模型预测控制（MPC）依赖于一个描述排队如何响应信号决策的模型，该模型在异构网络中难以指定和校准，尤其是在事故改变道路通行能力和转向比例的情况下。测量手段还存在第二个局限：停止线处的受控车道通常较短，因此向上游延伸的排队超出了车道级测量范围。我们提出GNN-MPC，一种滚动时域控制器，利用图神经网络学习到的动作条件化网络动态进行规划。相位序列通过近似最优响应在各交叉口之间进行协调，并设有接受性检验以保证预测成本不高于初始方案的成本。在非重复性事故的微观仿真中，无论在同质网格网络还是异构的21交叉口网络上，与所有测试过的控制器相比，GNN-MPC都降低了旅行时间并增加了完成的出行次数。

    arXiv:2611.27435v1 Announce Type: new  Abstract: Model predictive control (MPC) of urban traffic signals relies on a model of how queues respond to signal decisions, which is hard to specify and calibrate in heterogeneous networks, particularly when incidents change capacities and turning shares. Measurements add a second limitation: the controlled lanes at the stop line are often short, so queues extending upstream of them fall outside lane-level measurements. We propose GNN-MPC, a receding-horizon controller that plans with action-conditioned network dynamics learned by graph neural networks. Phase sequences are coordinated across intersections by approximate best responses, with an acceptance test that keeps the predicted cost no higher than that of the initial plan. In microscopic simulation under non-recurrent incidents, GNN-MPC reduces travel time and increases completed trips relative to all tested controllers, on a homogeneous grid and a heterogeneous 21-intersection network. C
    
[^67]: ORBIT：面向跨语言多模态事实核查的基于偏移量的激活引导方法

    ORBIT: Offset-based Activation Steering for Cross-lingual Multimodal Fact Verification

    [https://arxiv.org/abs/2611.27421](https://arxiv.org/abs/2611.27421)

    提出了推理时算法干预方法ORBIT，通过从平行声明-证据实例中估计偏移向量并修正声明与证据表示，在不需微调的情况下显著恢复小型视觉-语言模型在低资源语言上的多模态事实核查性能。

    

    多模态事实核查面临一个根本性的可及性瓶颈：尽管大型专有模型表现出跨语言鲁棒性，小型视觉-语言模型却在低资源语言上遭受严重且不成比例的准确率崩溃。弥合这一差距传统上需要巨大的模型规模或大量的特定语言微调。在本工作中，我们提出了通过双语推理时技术的偏移恢复方法（ORBIT），这是一种在推理时缓解上述性能崩溃的算法干预手段。ORBIT从平行的声明-证据实例中估计特定跨度的偏移向量，并将其应用于声明和证据的表示上，以针对由低资源证据引发的跨语言退化问题。跨越81种配置的实验表明，ORBIT在低资源设置下显著恢复了准确率和F1分数，并将校准偏移范数揭示为一种诊断性代理指标，同时优于基于翻译的方法。

    arXiv:2611.27421v1 Announce Type: new  Abstract: Multimodal fact verification faces a fundamental accessibility bottleneck: while large proprietary models exhibit cross-lingual robustness, small vision-language models suffer a severe, disproportionate accuracy collapse on low-resource languages. Closing this gap traditionally requires prohibitive scale or extensive language-specific fine-tuning. In this work, we propose Offset Recovery via Bilingual Inference-time Technique (ORBIT), an algorithmic intervention that mitigates this collapse at inference time. ORBIT estimates span-specific offset vectors from parallel claim-evidence instances and applies them to claim and evidence representations, targeting the cross-lingual degradation induced by low-resource evidence. Experiments across 81 configurations reveal that ORBIT substantially recovers both accuracy and F1-score in low-resource settings, exposes calibration offset norms as a diagnostic proxy, and outperforms translation-based a
    
[^68]: 神经变分Skyrmion作为拓扑基准：涌现对称性、高精度能量与KAN–MLP对比

    Neural variational Skyrmions as a topological benchmark: emergent symmetry, precision energies and KAN--MLP comparison

    [https://arxiv.org/abs/2611.27411](https://arxiv.org/abs/2611.27411)

    该论文将三维Skyrme模型确立为神经变分方法的拓扑基准：在无对称性拟设条件下高精度构建了 $B=1$ 至 $4$ 的Skyrmion，已知对称结构自发涌现，能量略低于文献参考值，拓扑荷保持精度达 $10^{-6}$，并通过Derrick位力关系与Faddeev–Bogomolny界等物理约束检验，同时对比了KAN与MLP网络的表现。

    

    我们提出将三维Skyrme模型作为神经变分方法的拓扑基准，并在不施加对称性拟设的条件下，以高精度构建了重子数 $B=1,\ldots,4$ 的完全三维Skyrmion。对于 $B=2,3,4$，这些Skyrmion的弛豫过程从位置和取向随机、不携带任何对称性信息的 $B$ 个单位电荷团块出发。然而，已知的环面形、四面体和立方体Skyrmion却自发涌现。所得的 $B=1,\ldots,4$ 变分能量略低于已发表的参考值。该神经方法凸显了在保持拓扑荷方面的高精度，在最高精度的解中拓扑荷与 $B$ 的偏差不超过 $10^{-6}$。我们的数值方法通过了诸如Derrick位力关系和Faddeev–Bogomolny界等物理约束检验，并通过独立的准随机初始构型等方式进行了评估与验证。

    arXiv:2611.27411v1 Announce Type: cross  Abstract: We propose the three-dimensional Skyrme model as a topological benchmark for neural variational methods and construct fully three-dimensional Skyrmions with baryon number $B=1,\ldots,4$ without imposing a symmetry ansatz, with high precision. For $B=2,3,4$, the relaxation of these Skyrmions starts from $B$ unit-charge lumps with random positions and orientations, which carry no symmetry information. The known toroidal, tetrahedral, and cubic Skyrmions nevertheless emerge spontaneously. The resulting variational energies for $B=1,\ldots,4$ lie slightly below the published reference values. The neural method highlights the high accuracy for retaining the topological charge, differing from $B$ by at most $10^{-6}$ in the highest-precision solutions. Our numerical method goes through physical constraint tests such as the Derrick virial relation and the Faddeev--Bogomolny bound, together with evaluation / verification through independent qu
    
[^69]: 受控负载下语言模型的退化：同步、规则执行与背景误触发

    Degradation of Language Models under Controlled Load: Synchronization, Rule Execution and Misfires on the Background

    [https://arxiv.org/abs/2611.27404](https://arxiv.org/abs/2611.27404)

    该论文提出将受控负载下语言模型的性能退化分解为同步失败、规则执行错误与无关误改三种成分，通过逐词对齐和考虑难度差异随负载变化的统计模型，独立量化文本长度、替换密度、规则数量等每种负载因素的影响。

    

    随着负载的增长，语言模型会犯越来越多的错误，但这种质量下降背后存在不同的过程。模型可能失去与源文本的对应关系，可能保持在文本中的位置却错误地执行规则，也可能更改了没有任何规则涉及的词。在这项工作中，模型在受控负载下的退化被分解为这三个组成部分：同步、规则执行和误触发。负载由在无意义文本中将一个词替换为另一个词的规则来设定，并且每种负载（文本长度、词在答案中的位置、替换密度、规则数量以及规则在列表中的位置）都被单独设定。答案与参考答案的对齐使得追踪每个词的结果成为可能，而一个考虑到请求之间难度差异本身依赖于负载的统计模型，为每个……（摘要原文在此处截断）

    arXiv:2611.27404v1 Announce Type: new  Abstract: As the load grows, a language model makes more and more errors, but different processes stand behind this decline in quality. The model may lose the correspondence with the source text, it may keep its place in the text but execute a rule wrongly, or it may change a word that no rule concerns. In this work the degradation of a model under controlled load is separated into these three components: synchronization, rule execution and misfires. The load is set by rules that replace one word with another in a meaningless text, and each kind of load (the text length, the position of a word in the answer, the replacement density, the number of rules and the place of a rule in the list) is set separately. The alignment of the answer with the reference makes it possible to follow the outcome of every word, and a statistical model that takes into account that the spread of difficulty between requests itself depends on the load restores, for each c
    
[^70]: HR-Merge：基于余量加权残差精炼的模型合并方法

    HR-Merge: Model Merging with Headroom-Weighted Residual Refinement

    [https://arxiv.org/abs/2611.27377](https://arxiv.org/abs/2611.27377)

    HR-Merge 是一种无需训练的通用模型合并增强框架，它根据各任务的验证准确率余量对其残差加权并重新合并，从而有效恢复现有合并方法在不同任务上不均匀损失的准确率。

    

    模型合并旨在将在共享主干上微调的多个专家模型（例如大语言模型的 LoRA 适配器）合并为一个多任务模型，而无需进一步训练。现有方法对每个任务采用同一合并规则，但由此造成的准确率损失在各任务之间并不均匀。我们证明，这些损失的准确率仍可从每个任务的残差（即其任务向量减去合并后的更新）中恢复，并且任务的验证余量（即合并所损失的专家验证准确率的占比）能够指示在何处进行恢复最为有效。由于来自不同任务的残差会相互干扰，因此必须对它们进行合并而非简单求和。我们提出 HR-Merge，这是一个无需训练的框架，可包裹任意现有的合并方法。它按各任务的余量对残差进行加权，用基础合并算子重新合并加权后的残差，并以在验证集上选定的强度将结果加回；当没有任何强度带来提升时，则回退到基础合并方法。在 LLaMA-3 8B 的六个 NLI 适配器上，HR-Merge 提升了……（原文摘要在此截断）

    arXiv:2611.27377v1 Announce Type: cross  Abstract: Model merging combines experts fine-tuned from a shared backbone, such as LoRA adapters of a large language model, into one multi-task model without further training. Existing methods apply one rule to every task, yet the accuracy they lose is uneven across tasks. We show that the lost accuracy remains recoverable from each task's residual, its task vector minus the merged update, and that a task's validation headroom, the share of its expert validation accuracy the merge loses, indicates where recovery pays off. Residuals from different tasks interfere, so they must be merged rather than summed. We propose HR-Merge, a training-free wrapper around any merging method. It weights each residual by its task's headroom, re-merges the weighted residuals with the base operator, and adds the result back at a strength selected on validation, falling back to the base merge when no strength helps. On six NLI adapters of LLaMA-3 8B, HR-Merge lifts
    
[^71]: RL专家的可组合软提示蒸馏：迈向持续后训练

    Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training

    [https://arxiv.org/abs/2611.27376](https://arxiv.org/abs/2611.27376)

    提出可组合软提示蒸馏（CSPD）方法，将每个RL专家独立蒸馏为软提示并通过拼接方式组合，无需重新整合模型即可实现持续后训练，性能媲美多教师蒸馏且不损害目标领域之外的能力。

    

    RL后训练通常会产生领域专家模型，这些专家随后必须被整合到一个模型中，而随着新专家不断出现，这种整合就成为了一个持续的过程。权重合并和多教师在线策略蒸馏（MOPD）在每添加一个专家时都必须重新整合整个模型，并且可能损害目标领域之外的能力。我们提出了可组合软提示蒸馏（CSPD），该方法保持RL之前的模型冻结，将每个RL专家独立地蒸馏到一个软提示中，并通过直接拼接各提示来组合专家，无需联合训练。单个软提示就足以蒸馏一个RL专家，在23个数学、代码和智能体设置中的22个里恢复了至少90%的教师增益。独立训练的提示之间可以良好组合。在多专家设置中，平均性能随着每个专家的加入而提升，完整组合的性能与MOPD相当（67.83 vs. 67.79）。该组合在六个……（摘要截断）

    arXiv:2611.27376v1 Announce Type: cross  Abstract: RL post-training typically produces domain experts that must then be consolidated into one model, and as new experts keep arriving, this consolidation becomes a continual process. Weight merging and multi-teacher on-policy distillation (MOPD) must re-consolidate the whole model whenever an expert is added, and can erode capabilities outside the target domains. We propose Composable Soft-Prompt Distillation (CSPD), which keeps the pre-RL model frozen, distills each RL expert independently into a soft prompt, and composes experts by concatenating their prompts without joint training. A single soft prompt suffices to distill an RL expert, recovering at least 90% of the teacher's gain in 22 of 23 math, code, and agent settings. Independently trained prompts compose well. In a multi-expert setting, the average rises with every expert added, and the full composition matches MOPD (67.83 vs. 67.79). The composition generalizes best across six 
    
[^72]: Gumbel替换：在不信任采样器的情况下消除强化学习中的采样器漂移

    Gumbel Substitution: Removing Sampler Drift from RL Without Trusting the Sampler

    [https://arxiv.org/abs/2611.27375](https://arxiv.org/abs/2611.27375)

    该论文提出Gumbel替换方法，由协调器分发比特级一致的公开Gumbel噪声，使训练器能重建其自身策略在相同噪声下的精确token样本，从而在不信任采样器、不依赖其报告概率的情况下，消除强化学习训练中因采样器与训练器策略不一致而产生的漂移。

    

    当强化学习的rollout来自不受信任的采样器（运行在任意硬件上、存在精度和检查点延迟问题）时，采样分布 q 与训练器的策略 p 不同，在采样器的 token 上进行训练会引入一个漂移项，其效果相当于朝向 q 进行监督微调。已发表的修正方法（TIS、MIS、IcePop、分数中心化、GSPO、CISPO）都需要 q(y)，而无许可采样器所报告的 q(y) 可能被误报。我们提出了Gumbel替换方法。协调器为每个 token 位置固定公开的Gumbel噪声，并在所有GPU上保持比特级一致，采样器通过 y=argmax(ℓ^q+G) 抽取样本。训练器重新计算 v*=argmax(ℓ^p+G)，即其自身策略在相同噪声下本应抽取的 token，并在采样器的上下文中对 v* 进行评分。对于遵循协议的采样器，v* 是给定前缀条件下来自 p 的精确样本，因此漂移在期望中对任何 q 都会消失，且无需依赖采样器报告的概率。我们测量了残余的……（摘要在此处截断）

    arXiv:2611.27375v1 Announce Type: new  Abstract: When RL rollouts come from untrusted samplers on arbitrary hardware, precision and checkpoint lag, the sampling distribution $q$ differs from the trainer's policy $p$, and training on the sampler's tokens adds a drift term equal to supervised fine-tuning toward $q$. Published corrections (TIS, MIS, IcePop, score centering, GSPO, CISPO) all need $q(y)$, which a permissionless sampler reports and can misreport. We introduce Gumbel substitution. The coordinator fixes public Gumbel noise for every token position, bit-identical across GPUs, and the sampler draws $y=\arg\max(\ell^q+G)$. The trainer recomputes $v^*=\arg\max(\ell^p+G)$, the token its own policy would have drawn with the same noise, and scores $v^*$ in the sampler's context. For a sampler that follows the protocol, $v^*$ is an exact sample from $p$ given the prefix, so the drift vanishes in expectation for any $q$, without sampler-reported probabilities. We measure the residual l
    
[^73]: AMARIS：通过自适应子空间继承融合通用与专家大语言模型

    AMARIS: Merging Generalist and Specialist LLMs via Adaptive Subspace Inheritance

    [https://arxiv.org/abs/2611.27369](https://arxiv.org/abs/2611.27369)

    AMARIS 提出了一种协方差驱动的免训练模型合并框架，通过在正则化扰动预算下选择高效用、低干扰的激活子空间，将专家模型能力以单一静态检查点的形式无训练地继承到通用大语言模型中。

    

    大语言模型（LLM）的领域专业化通常能提升目标领域的性能，但会损害其广泛能力。模型合并是一种无需训练的有吸引力的替代方案，然而常见的参数空间策略（例如插值和稀疏化平均）对任务特定干扰的控制能力有限。我们提出了通过排序继承子空间实现自适应模型对齐（AMARIS），这是一个由协方差驱动的框架，仅沿选定的激活方向传递专家模型的更新。AMARIS 将合并表述为预算化的子空间继承：在正则化的基础输出扰动预算下，寻求对专家激活的最大捕获，然后从排序后的广义特征方向构建低秩投影门控。由此得到一个单一的静态合并检查点，将知识迁移集中在高效用、低干扰的方向上。我们在基于 Qwen3 的合并模型（8B 和 32B）上评估了 AMARIS，涵盖金融、保险等领域（原文摘要此处被截断）。

    arXiv:2611.27369v1 Announce Type: new  Abstract: Domain specialization of large language models (LLMs) often improves target-domain performance but degrades broad capabilities. Model merging is an attractive training-free alternative, yet common parameter-space strategies (for example, interpolation and sparsified averaging) provide limited control over task-specific interference. We introduce Adaptive Model Alignment via Ranked Inherited Subspaces (AMARIS), a covariance-driven framework that transfers specialist updates only along selected activation directions. AMARIS formulates merging as budgeted subspace inheritance: it seeks high specialist activation capture under a regularized base-output perturbation budget, then builds a low-rank projection gate from ranked generalized eigendirections. This yields a single static merged checkpoint that concentrates transfer in high-utility, low-interference directions. We evaluate AMARIS on Qwen3-based merges (8B and 32B) across finance, inst
    
[^74]: 潜在雅可比引导：从预训练人形控制器中引发富含接触的分布外行为

    Latent Jacobian Steer: Eliciting Contact-Rich Out-of-Distribution Behaviors from Pretrained Humanoid Controllers

    [https://arxiv.org/abs/2611.27338](https://arxiv.org/abs/2611.27338)

    提出潜在雅可比引导（LJS）方法，通过将关节动作空间采样与策略雅可比的阻尼局部逆映射相结合，从预训练人形控制器中引发复杂且富含接触的分布外行为，兼顾了协调性与任务针对性。

    

    给定一个预训练的通用潜在条件全身人形控制器（例如行为基础模型 BFM），我们如何能够引发复杂且富含接触的分布外行为？原始关节动作探索能够实现直接的、特定任务的调整，但可能会破坏预训练控制器学到的协调性；而潜在空间探索保留了这种协调性，但难以引导至期望的物理效果。为了结合这两种探索空间的互补优势，我们提出了潜在雅可比引导（Latent Jacobian Steer, LJS），该方法将直接关节动作空间中的采样与所学习的冻结控制器输入潜在空间的语义相耦合。对于通过期望物理效果指定的任务，我们采样与任务相关的关节动作扰动，并通过策略雅可比的阻尼局部逆映射，将其映射为候选潜在（z）指令。这些指令通过冻结的预训练控制器执行……

    arXiv:2611.27338v1 Announce Type: cross  Abstract: Given a pretrained general latent-conditioned whole-body humanoid controller, say a behavioral foundation model (BFM), how can we elicit complex and contact-rich out-of-distribution behaviors? Raw joint-action exploration enables direct, task-specific adjustments but can disrupt coordination learned by the pretrained controller, whereas latent exploration retains that coordination but is difficult to steer toward desired physical effects. To combine the complementary strengths of these exploration spaces, we propose Latent Jacobian Steer (LJS), which couples sampling in direct joint-action space with the semantics of the learned frozen controller's input latent space. For a task specified through desired physical effects, we sample task-relevant joint-action perturbations and map them through a damped local inverse of the policy Jacobian to obtain candidate latent ($z$) commands. These commands are executed through the frozen pretraine
    
[^75]: 脉冲神经网络增量跟踪发放率网络

    Spiking Neural Networks Incrementally Track Firing-Rate Networks

    [https://arxiv.org/abs/2611.27322](https://arxiv.org/abs/2611.27322)

    提出了基于SNN增量跟踪发放率网络理论的LIF-Stream转换框架，使转换后的脉冲神经网络能够复用先前激活状态并仅计算帧间激活更新，从而在无界数据流上实现高效且高精度的SNN推理。

    

    理解脉冲神经网络（SNN）如何高效处理数据流，对于（i）研究生物大脑如何为终身预测节约能量，以及（ii）实现高效的连续AI推理都至关重要。在这项工作中，我们提出了一个关于SNN如何增量跟踪其对应的发放率网络（一种连续值神经网络ANN）的理论。我们证明了离散SNN的输出能够持续跟踪由SNN输入隐式确定的瞬时ANN输出。基于这一原理，我们开发了LIF-Stream，这是一个用于在无界数据流上进行高效SNN推理的转换框架。它使转换后的SNN能够复用先前的激活状态，仅计算帧间的激活更新，从而减少了每帧所需的平均时间步数。具体而言，LIF-Stream控制SNN持续跟踪参考ANN的输出，从而给出高精度的预测。

    arXiv:2611.27322v1 Announce Type: cross  Abstract: Understanding how spiking neural networks (SNNs) efficiently process data streams is key to (i) studying how biological brains conserve energy for lifelong predictions and (ii) enabling efficient continuous AI inference. In this work, we propose a theory on how an SNN incrementally tracks its corresponding firing-rate network, a continuous-valued neural network (ANN). We prove that the output of a discrete SNN continuously tracks the instantaneous ANN output implicitly determined by the SNN input. Based on this principle, we develop LIF-Stream, a conversion framework for efficient SNN inference over unbounded data streams. It enables the converted SNN to reuse its prior activation state and compute only the activation updates between frames, thereby reducing the average number of time steps needed per frame. Specifically, LIF-Stream controls the SNN to continuously track the reference ANN output, which gives a high-accuracy prediction 
    
[^76]: 基于条件对比正则化在扩散模型强化学习中保留预训练CFG结构

    Preserving Pretrained CFG Structure in RL for Diffusion Models with Conditional Contrast Regularization

    [https://arxiv.org/abs/2611.27288](https://arxiv.org/abs/2611.27288)

    提出条件对比正则化方法，在扩散模型强化学习后训练中保留预训练CFG的几何结构，作为诊断奖励黑客行为的有效指标，从而实现奖励提升与整体生成质量之间更稳定的权衡。

    

    强化学习已成为扩散模型后训练的重要组成部分。然而在实践中，标准强化学习往往能够提升目标奖励，却无法可靠地保持整体生成质量，导致不稳定的奖励-质量权衡，我们认为这是奖励黑客行为的一种表现形式。由于无分类器引导在提升生成图像的视觉清晰度、提示词对齐和感知质量方面的作用已为人所知，我们研究其底层结构能否在强化学习过程中作为有用的先验。我们提出了一个统一的几何视角，将CFG和RL都解释为同一预测空间中的校正向量，并将RL引起的校正向量在预训练CFG校正方向上的投影确定为衡量质量相关结构保留程度的有效诊断指标。几何分析和旋转干预实验进一步表明，预训练的CFG方向携带着对生成质量有用的信息。

    arXiv:2611.27288v1 Announce Type: cross  Abstract: Reinforcement Learning (RL) has become an important component in the post-training of diffusion models. In practice, however, standard RL often improves the target reward without reliably preserving holistic generation quality, leading to unstable reward-quality trade-offs that we view as a form of reward hacking. Since Classifier-Free Guidance (CFG) is known to improve visual clarity, prompt alignment, and perceptual quality in generation, we investigate whether its underlying structure can serve as a useful prior during RL. We present a unified geometric view that interprets both CFG and RL as correction vectors in a common prediction space, and identify the projection of the RL-induced correction onto the pretrained CFG correction as a useful diagnostic of quality-related structural preservation. Geometric analysis and rotation intervention further show that the pretrained CFG direction carries information useful for generation qual
    
[^77]: 面向不确定性条件下无人机最优控制的误差校正物理信息神经算子

    Error-Corrected Physics-Informed Neural Operator for Optimal Control of Unmanned Aerial Vehicles under Uncertainty

    [https://arxiv.org/abs/2611.27272](https://arxiv.org/abs/2611.27272)

    该论文提出一种误差校正的物理信息神经算子，通过一次牛顿步校正预测误差来求解不确定性条件下的无人机开环最优控制问题，其精度可匹敌传统求解器且速度快8至15倍。

    

    我们使用物理信息DeepONet求解参数不确定性下的开环最优控制问题，该算子通过单次评估即可返回所有采样场景的轨迹。该算子仅在通用控制下的残差上进行训练，因此可在不同目标和时域上重复使用。由于基于梯度的优化器会利用代理模型的误差，我们通过对动力学方程执行一次牛顿步来校正预测，该计算通过一次并行扫描即可对所有场景和时间完成。校正后的状态具有仅由残差界定的二阶误差，且校正后的成本对任意控制在采样误差范围内是准确可靠的。在固定翼无人机基准测试中，在相同样本量下，校正后的算子达到了批量Adam求解器和Ipopt求解器的精度；在数百个场景的规模下，在同一GPU上比Adam求解器快8倍，在CPU上比Ipopt快15倍。

    arXiv:2611.27272v1 Announce Type: cross  Abstract: We solve open-loop optimal control problems under parametric uncertainty using a physics-informed DeepONet that returns trajectories for all sampled scenarios in one evaluation. Trained only on the residual at generic controls, the operator is reused across targets and horizons. Because a gradient-based optimizer exploits surrogate errors, we correct the prediction by one Newton step on the dynamics, computed for all scenarios and times by one parallel scan. The corrected state has a second-order error bounded by the residual alone, and the corrected cost is honest for every control up to the sampling error. On a fixed-wing UAV benchmark, the corrected operator matches the accuracy of a batched Adam solver and an Ipopt solver at equal sample size and is 8 times faster than the Adam solver on the same GPU at a few hundred scenarios and 15 times faster than Ipopt on a CPU.
    
[^78]: 结构化分布偏移下熵正则最优传输的半参数有效性

    Semiparametric Efficiency for Entropic Optimal Transport under Structured Distribution Shift

    [https://arxiv.org/abs/2611.27261](https://arxiv.org/abs/2611.27261)

    本文通过精确投影恒等式揭示了结构约束下熵最优传输估计的方差增益何时为正、何时为零，并构造了达到该效率界的混合经验似然估计量。

    

    结构信息能否改进熵正则最优传输（EOT）的估计，以及何时增益为零？我们研究了一个固定维数的指数密度比模型（DRM），该模型具有非参数基线、紧支撑集和固定的正则化系数。在正的极限抽样比例下，一个精确的投影恒等式量化了来自限制约束的方差增益。混合经验似然估计量达到了受约束界的水平，并且当该界为正时支持渐近有效的置信区间。即使在模型正确设定的情形下，增益也可能为零，且随着倾斜空间的扩大而缩小。在局部错误设定下，同一投影残差决定了方向特定的一阶偏差。当两个样本同时增长且目标比例趋近于零时，倾斜函数的一阶导数为零会导致源样本波动和二次倾斜误差成为主导项。二者的平衡产生了相变现象。

    arXiv:2611.27261v1 Announce Type: cross  Abstract: Can structural information improve estimation of entropic optimal transport (EOT), and when is the gain zero? We study a fixed-dimensional exponential density ratio model (DRM) with a nonparametric baseline, compact support, and fixed positive regularization. At positive limiting sampling fractions, an exact projection identity quantifies the variance gain from the restriction. The pooled empirical-likelihood estimator attains the restricted bound and supports asymptotically valid confidence intervals when the bound is positive. Even under correct specification, the gain can be zero and can shrink as the tilt space expands. The same projection residual determines direction-specific first-order bias under local misspecification. When both samples grow and the target fraction vanishes, a zero first-order tilt derivative can leave source fluctuations and quadratic tilt error as the leading terms. Their balance gives a phase transition in 
    
[^79]: 训练过程中层间局部相互作用偏好的自发形成

    A preference for local interaction between layers emerges during training

    [https://arxiv.org/abs/2611.27258](https://arxiv.org/abs/2611.27258)

    该研究发现深度神经网络在训练过程中会自发形成“局部性”——相邻层之间对彼此变化的响应随训练不断增强，逐渐超过距离更远的层，尽管训练本身通过共享误差将所有层耦合在一起。

    

    深度神经网络由多层堆叠而成，所有层都被训练以降低同一个共享误差。因此，某一层的变化可以传递到其他每一层。由于误差由所有层共享，人们会预期其他层的响应是相似的，无论它们紧邻被改变的层还是远离它。我们发现，在训练早期，这种响应很弱，且其按距离排序并不稳定。随着训练的进行，响应不断增强，并且在几乎每一个网络中，相邻层对彼此变化的响应都比距离更远的层更强。这种局部性在训练过程中形成，尽管训练本身通过共享误差将每一层与所有其他层连接起来。我们通过在训练的多个阶段推动网络副本中的一层，并测量这种推动有多少传递到其他每一层来展示这一现象。实验使用了512个五层网络，在模加法任务上分别使用AdamW和随机梯度下降两种优化器进行训练。

    arXiv:2611.27258v1 Announce Type: new  Abstract: A deep neural network is a stack of layers, and all layers are trained to lower one shared error. A change in one layer can therefore reach every other layer. Because the error is shared by all layers, one would expect the other layers to respond alike, whether they are next to the changed layer or far from it. We find that early in training the response is weak and its order by distance is not stable. As training proceeds, the response grows, and neighboring layers come to respond to changes in each other more strongly than layers further apart, in almost every network. This locality forms during training, although training links every layer to every other. We show this by pushing one layer of a copy of the network at many stages of training and measuring how much of the push reaches each of the other layers. The experiments use 512 five-layer networks trained on modular addition with each of two optimizers, AdamW and stochastic gradien
    
[^80]: 针对深度伪造生成的主动防御基准

    Proactive Defense Benchmark against Deepfake Generation

    [https://arxiv.org/abs/2611.27253](https://arxiv.org/abs/2611.27253)

    该论文提出了首个针对深度伪造主动防御的综合评估基准，揭示了保真度与身份指标的矛盾解读以及白盒峰值性能背后的过拟合问题，并建立严格标准以推动真正可泛化防护方法的发展。

    

    尽管针对深度伪造的主动防御方法不断涌现，但由于缺乏统一的评估协议，导致无法进行公平比较，也掩盖了关键性的漏洞。为弥补这一空白，我们提出了首个全面的基准，系统性地评估干扰效果、鲁棒性和可迁移性，涵盖像素、感知和身份三个层面的指标。我们广泛的分析表明，保真度指标与身份指标捕捉的是相互正交的性能维度，单独依赖其中任何一个往往会导致相互矛盾的解读。此外，我们发现了一个根本性的权衡现象：白盒设置下的峰值性能往往意味着过拟合，为此我们引入了一种校准评估方法以纠正由生成器引起的身份偏差。通过揭示这些盲点，我们建立了一套严格的标准，用以指导开发真正具备泛化能力的防护方法。

    arXiv:2611.27253v1 Announce Type: cross  Abstract: Despite the proliferation of proactive defenses against deepfakes, the lack of a unified evaluation protocol precludes fair comparison and masks critical vulnerabilities. To bridge this gap, we present the first comprehensive benchmark that systematically assesses disruption, robustness, and transferability encompassing pixel, perceptual, and identity metrics. Our extensive analysis reveals that fidelity and identity metrics capture orthogonal performance axes, often leading to conflicting interpretations when relied upon individually. Furthermore, we identify a fundamental trade-off where peak white-box performance signals overfitting, and we introduce a calibrated evaluation to correct generator-induced identity bias. By exposing these blind spots, we establish a rigorous standard to guide the development of genuinely generalizable protections. Project page is available at: https://proactivedefensebenchmark.github.io/
    
[^81]: 更多比特位也解决不了问题：诊断与治疗病态的反向传播路径（以Gemma-4为例的案例研究）

    More Bits Won't Fix It: Diagnosing and Treating an Ill-Conditioned Backward Pass (A Case Study on Gemma-4)

    [https://arxiv.org/abs/2611.27243](https://arxiv.org/abs/2611.27243)

    Gemma-4的LoRA微调中bf16与fp32梯度在模型低层几乎正交，其根源是第19-21层MLP在图像token位置存在病态反向传播路径，且梯度可能被单个低信息图像块主导，在反向传播中断开该路径即可解决，而单纯提高数值精度并无帮助。

    

    在对Gemma-4（12B和31B）进行长提示词的LoRA微调时——无论提示词包含CT图像、自然照片还是约2000个token的文本——我们发现bf16和fp32梯度在语言模型的上三分之一部分高度一致（余弦相似度达到0.96或以上），但在下三分之一部分却几乎正交。在图像提示词场景下，我们将这一现象追溯到一条病态的反向传播路径。在隐状态上施加10^-6量级的逐元素相对噪声时，fp32梯度保持不变，证实fp32可作为有效的参考基准。然而当噪声达到10^-4量级时，fp32梯度与无噪声梯度的余弦相似度在10次测量中有7次降至0.9以下，而bf16前向状态与fp32状态的差异约为每token 1%。通过状态交换和仅作用于反向传播的干预手段，我们将Gemma-4-12B中的这种敏感性定位于第19-21层的MLP模块在图像token位置处，在这些位置上，梯度可能被单个低信息量的图像块所主导。在反向传播中断开该路径即可……

    arXiv:2611.27243v1 Announce Type: new  Abstract: During LoRA fine-tuning of Gemma-4 (12B and 31B) on long prompts, whether they contain CT images, natural photos, or about 2,000 tokens of text, bf16 and fp32 gradients agree in the upper third of the language model (cosine similarity 0.96 or above) but are nearly orthogonal in the lower third. On image prompts, we trace the effect to an ill-conditioned backward path. The fp32 gradient is unchanged by element-wise relative noise of $10^{-6}$ on hidden states, confirming that fp32 is a valid reference. At $10^{-4}$, however, its cosine similarity with the noise-free gradient falls below 0.9 in 7 of 10 measurements, whereas bf16 forward states differ from fp32 states by about 1% per token. State swaps and backward-only interventions localize this sensitivity in Gemma-4-12B to the MLPs of layers 19-21 at image-token positions, where the gradient can be dominated by a single low-information patch. Detaching that path in the backward pass (wh
    
[^82]: CreditFilter：基于响应内熵排名的校准 Token 权重方法用于 RLVR

    CreditFilter: Calibrated Token Weights over the Within-Response Entropy Rank for RLVR

    [https://arxiv.org/abs/2611.27227](https://arxiv.org/abs/2611.27227)

    该论文将 token 对结果的影响形式化为“分叉值”，并利用分叉概率随响应内熵排名呈指数衰减的实证发现，提出 CreditFilter——通过熵排名校准 token 权重、在响应内部重新分配信用，从而改进免评论家的可验证奖励强化学习（RLVR）。

    

    免评论家的可验证奖励强化学习（RLVR）通常为每个 token 分配统一的优势值。诸如 80/20 掩码之类的 token 重加权启发式方法偏向高熵 token，但无法衡量一个 token 对最终结果的实际影响。我们将这种影响形式化为“分叉值”：即策略在下一个 token 的各个选择上成功概率的方差。沿一条响应的期望分叉值之和恰好等于奖励方差。因此，对于一个提示词的多条响应取期望，token 信用就是一个在分叉处花费的固定预算。这启发了在响应内部重新分配信用的权重设计。分叉值可以分解为“分叉后果”（需要代价高昂的后续模拟来估计）和“分叉概率”（由 softmax 直接提供）。在 Qwen3-8B-Base 上，该分叉概率随 token 在响应内的熵排名以特定类别相关的速率呈指数衰减，揭示出大多数 token 几乎不承载信用（原文摘要在此处截断）。

    arXiv:2611.27227v1 Announce Type: cross  Abstract: Critic-free reinforcement learning with verifiable rewards (RLVR) typically assigns a uniform advantage to every token. Token-reweighting heuristics like the 80/20 mask favor high-entropy tokens, but fail to measure a token's actual impact on the outcome. We formalize this impact as the forking value: the variance of the success probability over the policy's next-token choices. The expected forking values along a response sum exactly to the reward variance. Token credit is thus, in expectation over a prompt's responses, a fixed budget spent at the forks. This motivates weights that redistribute credit within a response. The forking value factorizes into a fork consequence, which requires costly continuations to estimate, and a fork probability, which the softmax provides. On Qwen3-8B-Base, this fork probability decays exponentially with a token's within-response entropy rank at class-specific rates, revealing that most tokens carry neg
    
[^83]: 基于平稳性的时变非凸复合优化后悔框架

    A Stationarity-Based Regret Framework for Time-Varying Nonconvex Composite Optimization

    [https://arxiv.org/abs/2611.27218](https://arxiv.org/abs/2611.27218)

    本文为时变非凸复合优化提出了基于一阶平稳性（如近端梯度残差）的局部后悔度量框架，证明该度量在凸情形下等价于经典动态后悔，并指出在非凸情形下若缺乏时间正则性则次线性平稳性后悔无法实现。

    

    本文研究时变非凸复合优化问题，其中决策是按序贯方式进行的，而经典的基于函数值的后悔概念在此类问题中通常是不适定的。我们引入了一个基于平稳性的性能框架，该框架通过由一阶最优性条件构建的局部后悔度量来定义，并识别出使这些度量构成合理性能准则所需的最小结构性条件。这一抽象框架涵盖了典型的最优性度量（包括近端梯度残差），并在凸设置下退化为经典的动态后悔。随后，我们将近端梯度残差作为一般复合问题的一个具体且具有算法意义的实例加以重点研究。在凸问题中，我们证明所得的局部后悔与标准动态后悔等价。在非凸设置中，我们证明了在没有时间正则性（temporal regularity）的条件下，次线性的基于平稳性的后悔是不可能达到的（原文在此处截断）。

    arXiv:2611.27218v1 Announce Type: new  Abstract: This paper studies time-varying nonconvex composite optimization, where decisions are made sequentially and classical value-based regret notions are generally ill-posed. We introduce a stationarity-based performance framework defined through local regret measures built from first-order optimality conditions, and identify minimal structural requirements under which such measures constitute sound performance criteria. This abstraction encompasses canonical optimality measures, including the proximal-gradient residual, and collapses to classical dynamic regret in convex settings. We then focus on the proximal-gradient residual as a concrete and algorithmically meaningful instantiation for general composite problems. In convex problems, we show that the resulting local regret is equivalent to standard dynamic regret. In nonconvex settings, we establish that sublinear stationarity-based regret is unattainable without temporal regularity, and 
    
[^84]: LingSante-Pulse：通过随机特征预测与跨模态对齐实现的统一生理信号预训练

    LingSante-Pulse: Unified Physiological Pretraining through Random-Feature Prediction and Cross-Modal Alignment

    [https://arxiv.org/abs/2611.27206](https://arxiv.org/abs/2611.27206)

    该论文提出统一生理预训练框架LingSante-Pulse，通过随机特征预测、跨模态对齐与掩码波形重建学习通用生理信号表示，在涵盖PPG、ECG、EEG和EMG的38个任务配置的每个领域均取得最佳排名。

    

    临床监护与可穿戴设备采集互补的生理信号，但其模态覆盖部分重叠，导致信号内信息与跨模态关系难以被联合利用。我们提出了LingSante-Pulse，一个统一、通用且灵活的生理信号表示框架，可从这些异构记录中学习。该框架采用一个共享的、以信号为条件的编码器，独立处理每个物理通道，并支持不同数量和组合的信号，从而兼顾多模态评估与单传感器推理。成对跨模态对齐、结构化随机特征预测（RFP）与掩码波形重建三者协同工作，共同学习生理关系并保留具有信息量的信号内结构。在涵盖PPG、ECG、EEG和EMG的38个通用生理任务配置中，冻结编码器在每个领域均取得了按数据集平衡的最佳平均排名。

    arXiv:2611.27206v1 Announce Type: cross  Abstract: Clinical monitoring and wearable devices capture complementary physiological signals, but their partially overlapping modality coverage makes within-signal information and cross-modal relationships difficult to exploit jointly. We introduce LingSante-Pulse, a unified, general-purpose and flexible physiological representation framework that learns from these heterogeneous recordings. A shared, signal-conditioned encoder processes each physical channel independently and supports different numbers and combinations of signals for multimodal assessment and single-sensor inference. Pairwise cross-modal alignment, structured random-feature prediction (RFP) and masked waveform reconstruction jointly learn physiological relationships and preserve informative within-signal structure. Across 38 general physiological task configurations spanning PPG, ECG, EEG and EMG, the frozen encoder achieves the best dataset-balanced mean rank in each domain. 
    
[^85]: 使用可执行符号表示改进大语言模型算术推理中准确率随问题复杂度的扩展性

    Improved Accuracy Scaling with Problem Complexity in LLM Arithmetic Reasoning Using Executable Symbolic Representations

    [https://arxiv.org/abs/2611.27203](https://arxiv.org/abs/2611.27203)

    该论文提出GSM算术基准（GSM-AB），通过独立控制操作数位长和运算次数，证明使用可执行符号表示（如Lean）能改善大语言模型算术推理的准确率扩展性，减缓准确率随问题复杂度增加而退化的速度。

    

    在数学基准测试上取得的高准确率并不能证明当问题复杂度增加时推理依然可靠。我们研究了可执行符号表示如何改变大语言模型相比自然语言推理的准确率扩展规律。为解决这一局限，我们提出了GSM算术基准（GSM-AB），这是一个可控基准，可以独立改变操作数的位数长度和运算次数，以探测数值规模和组合复杂度的影响。我们发现可执行表示能够提高准确率，并减缓其随复杂度增加而下降的速度。在所有测试的模型规模下，Lean都降低了对操作数位长的平均敏感度。相比之下，随运算次数增加而出现的更缓慢的准确率退化现象仅在较大模型规模下出现。Python思维程序（Python-PoT）对照实验显示出类似的运算维度模式，表明这一发现并非Lean所特有。这些发现表明……（原文摘要在此处截断）

    arXiv:2611.27203v1 Announce Type: new  Abstract: High accuracy on mathematical benchmarks does not establish whether reasoning remains reliable as problem complexity increases. We examine how executable symbolic representations change accuracy scaling in large language models compared with natural-language reasoning. To address this limitation, we introduce the GSM Arithmetic Benchmark (GSM-AB), a controlled benchmark that independently varies operand digit length and operation count to probe numerical scale and compositional complexity. We find that executable representations can improve accuracy and slow its decline as complexity increases. Lean reduces average sensitivity to operand digit length at every tested model size. By contrast, slower degradation with increasing operation count emerges at the larger model sizes. A Python program-of-thought (Python-PoT) control shows a similar operation-axis pattern, suggesting that this finding is not specific to Lean. These findings show th
    
[^86]: 主干网络特定的LoRA几何：正交性、叠加与溯源

    Backbone-Specific LoRA Geometry: Orthogonality, Superposition, and Provenance

    [https://arxiv.org/abs/2611.27201](https://arxiv.org/abs/2611.27201)

    该论文将LoRA更新方向视为主干网络独特的“指纹”，通过推导跨主干网络相关性上界和局部梯度流模型揭示了正交性与叠加规律，从而实现基于余弦相似度的主干网络识别以及无需坐标对齐的已注册适配器复用检验。

    

    arXiv:2611.27201v1 公告类型：cross 摘要：在预训练模型之间共享同一个低秩适应（LoRA）更新可以简化部署。组合方法通常合并来自同一主干网络的更新，而跨模型迁移则将更新变换以适配目标主干网络。在保持精度的同时跨主干网络共享更新、并能区分各自贡献的条件仍不清楚。我们的想法是将主干网络特有的更新方向作为其训练主干网络的签名（标识）。我们在独立且均匀的预训练特征方向假设下，从注意力对称性与等变LoRA学习推导出一个跨主干网络相关性上界。一个局部梯度流模型解释了在同一主干网络与任务上独立训练运行之间呈正的平均相关性。这些规律启发了基于余弦相似度的主干网络识别方法，以及无需坐标对齐即可对已注册适配器进行复用的内积检验。在CLIP主干网络上的实验表明，共享求和能够实现……（原文摘要在此处截断）

    arXiv:2611.27201v1 Announce Type: cross  Abstract: Sharing one low-rank adaptation (LoRA) update across pretrained models simplifies deployment. Composition methods typically combine updates from a common backbone, while cross-model transfer transforms updates for a target backbone. The conditions for sharing updates across backbones while preserving accuracy and distinguishing individual contributions remain unclear. Our idea is to use backbone-specific update directions as signatures of their training backbones. We derive a cross-backbone correlation bound from attention symmetries and equivariant LoRA learning under independent, uniform pretrained feature orientations. A local gradient-flow model explains positive mean correlation between independent runs on the same backbone and task. These patterns motivate cosine-based backbone identification and inner-product tests for registered-adapter reuse without coordinate alignment. Experiments on CLIP backbones show that shared sums achi
    
[^87]: cmHC：通过几何收缩平衡超连接中的流混合

    cmHC: Balancing Stream Mixing in Hyper-Connections via Geometric Contraction

    [https://arxiv.org/abs/2611.27193](https://arxiv.org/abs/2611.27193)

    本文揭示了Birkhoff多胞形中残差流混合的权衡问题——混合不足会退化为独立流、过度混合会使流不可区分——并提出cmHC方法，利用几何收缩在多胞形内平衡流混合程度，同时从理论上给出了跨层累积混合、流差异与梯度差异的界。

    

    流形约束超连接（mHC）将残差连接扩展到多个流，并通过将流混合残差矩阵约束到Birkhoff多胞形（即双随机矩阵的集合）来稳定残差传播。在这项工作中，我们研究如何在Birkhoff多胞形内有效地混合残差流。我们揭示了一种混合权衡：当混合不足的残差矩阵保持在Birkhoff多胞形顶点附近时，超连接将退化为一组几乎相互独立的流；而当过度混合的残差矩阵集中在多胞形的均匀重心附近时，各流之间将变得难以区分。我们通过给出跨层累积残差混合、流差异以及混合系数梯度差异的界，从理论上验证了我们的发现。受这些发现启发，我们提出了流形约束超连接的收缩方法（cmHC）……

    arXiv:2611.27193v1 Announce Type: new  Abstract: Manifold-constrained Hyper-Connections (mHC) extends residual connections to multiple streams and stabilizes residual propagation by constraining stream-mixing residual matrices to the Birkhoff polytope, the set of doubly stochastic matrices. In this work, we investigate how to effectively mix residual streams within the Birkhoff polytope. We reveal a mixing trade-off: when under-mixed residual matrices stay close to the vertices of the Birkhoff polytope, hyper-connections will degenerate into a set of nearly independent streams; when over-mixed residual matrices are concentrated around the uniform barycenter of the polytope, streams will become less distinguishable from each other. We theoretically validate our findings by bounding the cumulative residual mixing, stream difference and gradient difference of mixing coefficients across layers. Motivated by these findings, we propose Contraction for Manifold-Constrained Hyper-Connections (
    
[^88]: 剔除与注入：面向时间序列预测的事件效应解耦

    Lift Out, Kick In: Decoupling Event Effects for Time Series Forecasting

    [https://arxiv.org/abs/2611.27191](https://arxiv.org/abs/2611.27191)

    该论文提出基于共享事件响应函数（ERF）的 Loki 框架，通过从历史观测中剔除事件诱发偏差、再向预测中注入未来事件效应，实现事件效应与系统内在时间动态的解耦，从而提升事件感知的时间序列预测性能。

    

    现实世界中的时间序列往往受到外生事件的影响，这些事件的效应叠加在系统固有的时间动态之上。因此，历史观测将持续存在的动态与事件诱发的偏差纠缠在一起，使得难以判断哪些模式应当被外推、哪些应归因于特定的事件情境。现有的事件感知预测方法虽然能够融合事件信息或建模与事件相关的变化，但通常将历史事件归因与未来事件生成分别处理。我们提出 Loki（Lift Out, Kick In），一个基于共享事件响应函数构建的通用事件感知预测框架。该 ERF 根据事件类型和相对时间位置来建模事件效应，并在历史窗口与预测窗口之间共享。Loki 在预测前从历史观测中“剔除”估计得到的事件诱发偏差，然后使用相同的 ERF 将未来事件效应“注入”预测之中。

    arXiv:2611.27191v1 Announce Type: new  Abstract: Real-world time series are often shaped by exogenous events whose effects are superimposed on the system's underlying temporal dynamics. Historical observations therefore entangle persistent dynamics with event-induced deviations, making it difficult to determine which patterns should be extrapolated and which should be attributed to specific event contexts. Existing event-aware forecasting methods can incorporate event information or model event-related variation, but typically treat historical event attribution and future event generation separately. We propose Loki (Lift Out, Kick In), a general event-aware forecasting framework built on a shared Event Response Function (ERF). The ERF models event effects according to event type and relative temporal position and is shared across the historical and forecast windows. Loki lifts out estimated event-induced deviations from historical observations before forecasting, then uses the same ER
    
[^89]: 面向印度的节俭人工智能框架：负责任且规模适度的AI

    A Frugal AI Framework for India: Responsible and Right-sized AI

    [https://arxiv.org/abs/2611.27189](https://arxiv.org/abs/2611.27189)

    本文提出“负责任节俭AI”框架，主张印度既不应照搬中美重资本的基础模型路线，也不应仅作外国AI的下游采用者，而应先明确功能需求并判断是否需要AI，再选择以最少资源满足需求的最小充分“规模适度”模型，同时保持鲁棒性与负责任治理。

    

    印度关于人工智能的争论常常在两个并不充分的选择之间摇摆：要么复制美国或中国那种计算密集、资本密集的基础模型计划，要么停留在作为外国人工智能的下游采用者，仅提供本地化的界面与安全保障。这两者都无法满足一个需要在基础设施不均衡、预算有限的条件下，向极其庞大且多样化的人口提供人工智能服务的国家。这份来自印度理工学院马德拉斯分校负责任人工智能中心（CeRAI）的白皮书提出了第三条路径：负责任节俭AI（Responsible Frugal AI），它将基于原则的节俭工程与负责任AI相结合。负责任节俭AI是“规模适度”的AI：它首先明确功能需求，追问是否真的需要AI，然后选择最小的充分模型，以最少的资源满足需求，同时保持经过验证的余量、鲁棒性和负责任的治理。该框架在四个维度上应用节俭原则（数据、计算……（摘要原文在此处截断））。

    arXiv:2611.27189v1 Announce Type: cross  Abstract: India's AI debate often swings between two insufficient options: replicating the compute-intensive, capital-heavy foundation-model programmes of the United States or China, or remaining a downstream adopter of foreign AI with localised interfaces and safeguards. Neither serves a country that must deliver AI to a very large, diverse population across uneven infrastructure and within constrained budgets. This white paper from the Centre for Responsible AI (CeRAI), IIT Madras, proposes a third path: Responsible Frugal AI, which applies principle-based frugal engineering alongside Responsible AI. Responsible Frugal AI is right-sized AI: it fixes the functional requirement first, asks whether AI is needed at all, and then chooses the smallest sufficient model, meeting the requirement with minimal resources while keeping a verified margin, robustness and responsible governance. The framework applies frugality across four dimensions (data, co
    
[^90]: FedQCal：面向单轮联邦学习的二次规划与对偶空间校准方法

    FedQCal: Quadratic Programming with Dual-Space Calibration for One-Shot Federated Learning

    [https://arxiv.org/abs/2611.27170](https://arxiv.org/abs/2611.27170)

    FedQCal将单轮联邦学习的模型聚合形式化为最小范数二次规划，利用服务器参考模型定义对齐要求，并通过分类损失的两个对偶变量自动校准对齐阈值，无需预先设定阈值即可有效聚合异构客户端模型。

    

    单轮联邦学习需要在客户端仅上传一次后聚合异构的客户端模型。由于各客户端本地类别分布不同，如何决定保留哪些客户端的更新变得十分困难。我们提出FedQCal，该方法利用在服务器端小型标注数据集上训练的参考模型来定义聚合所需满足的对齐要求。其核心思想是在保留充分参考对齐的同时，避免不必要地过大的聚合系数。一个最小范数二次规划（QP）用于强制满足这些要求。在同一批服务器样本上的分类损失通过两个对偶变量来选择合并方案，因此对齐阈值无需事先人为设定。我们证明了所选系数在其诱导阈值处仍然是精确的QP解。一个扰动界解释了系数范数如何限制投影误差的放大。在CIFAR-10、CIFAR-100和SVHN上的实验表明，该方法在大多数设置下均优于单轮FedAvg。

    arXiv:2611.27170v1 Announce Type: cross  Abstract: One-shot federated learning must combine heterogeneous client models after a single upload. Different local class distributions make it difficult to decide which client updates to retain. We propose FedQCal, which uses a reference model trained on a small labeled server set to define alignment requirements for aggregation. The key idea is to retain sufficient reference alignment while avoiding unnecessarily large aggregation coefficients. A minimum-norm quadratic program (QP) enforces these requirements. Classification loss on the same server examples selects a merge through two dual variables, so the alignment thresholds need not be prescribed in advance. We prove that the selected coefficients remain exact QP solutions at their induced thresholds. A perturbation bound explains how coefficient norm limits the amplification of projection errors. Experiments on CIFAR-10, CIFAR-100, and SVHN show gains over one-round FedAvg in most setti
    
[^91]: BusemannFormer：面向双曲图Transformer的极限球自注意力

    BusemannFormer: Horospherical Self-Attention for Hyperbolic Graph Transformers

    [https://arxiv.org/abs/2611.27167](https://arxiv.org/abs/2611.27167)

    该论文提出了基于理想边界方向上Busemann深度的极限球自注意力（HSA），并将其构建为完全双曲的图Transformer BusemannFormer，在层次化图任务上显著优于现有方法。

    

    双曲神经网络在层次化图数据上表现出色，但现有的双曲Transformer大多只是对传统注意力机制进行改造，并未显式利用理想边界——这一与树形层次结构密切相关的几何结构。我们提出了极限球自注意力（Horospherical Self-Attention, HSA）：每个查询选择一个理想边界方向，并根据键沿该方向的Busemann深度进行评分，从而得到一种逐token的双曲注意力核。我们建立了HSA与点积注意力在平坦极限下的联系，将得分的水平集刻画为极限球，并给出了键侧梯度的界。HSA被集成到BusemannFormer——一个完全双曲的图Transformer中，并提供了利用查询深度调节注意力集中程度的对数深度变体。我们进一步引入了有限边界字典扩展，以支持线性复杂度的全局注意力。在层次化图和引文图上的实验表明，BusemannFormer-LD优于现有的……（原文截断）

    arXiv:2611.27167v1 Announce Type: new  Abstract: Hyperbolic neural networks excel on hierarchical graph data, yet existing hyperbolic Transformers largely adapt conventional attention without explicitly exploiting the ideal boundary, a geometric structure closely tied to tree hierarchy. We propose Horospherical Self-Attention (HSA): each query selects an ideal-boundary direction and scores keys by their Busemann depth along that direction, yielding a token-to-token hyperbolic attention kernel. We establish a flat-limit connection to dot-product attention, characterize score level sets as horospheres, and bound key-side gradients. HSA is integrated into BusemannFormer, a fully hyperbolic graph Transformer, with a log-depth variant that uses query depth to modulate attention concentration. We further introduce a finite-boundary-dictionary extension supporting linear-complexity global attention. Experiments on hierarchical and citation graphs show that BusemannFormer-LD outperforms the ev
    
[^92]: 用于准确且物理合理的Apo-to-Holo蛋白质生成的分层混合坐标流匹配

    Hierarchical Hybrid-Coordinate Flow Matching for Accurate and Physically Plausible Apo-to-Holo Protein Generation

    [https://arxiv.org/abs/2611.27162](https://arxiv.org/abs/2611.27162)

    CrypticFlow提出了一种分层混合坐标流匹配方法，先在笛卡尔空间生成稀疏Cα锚点，再以内坐标递归填补中点并施加距离约束，从而同时实现高全局精度与物理合理的局部几何，解决apo-to-holo蛋白构象预测问题。

    

    从非结合态（apo）结构预测类结合态（holo）蛋白构象在基于结构的药物设计中至关重要。大多数现有的apo-to-holo模型在笛卡尔空间中运行，并可实现较强的全局精度。然而，高的全局精度并不能保证物理上合理的局部主链几何结构。内坐标模型提供了局部控制和有效性，但递归式生成可能放大角度误差，即所谓的杠杆臂效应。我们提出CrypticFlow，一种结合这两种互补表示的分层流匹配模型。一个以ESM-3为条件的E(3)等变图网络首先在笛卡尔空间中生成稀疏的Cα锚点，然后条件中点Transformer在局部内坐标中递归地填补空隙。在递归生成过程中，解码器将每个中点与序列相邻父锚点之间的距离约束为对应的apo Cα-Cα距离。（原文摘要在此处截断）

    arXiv:2611.27162v1 Announce Type: new  Abstract: Predicting holo-like protein conformations from apo structures is essential in structure-based drug design. Most existing apo-to-holo models operate in Cartesian space and achieve strong global accuracy. However, high global accuracy does not ensure physically valid local backbone geometry. Internal-coordinate models offer local control and validity, but regressive generation can amplify angular errors, known as the lever-arm effect. We introduce CrypticFlow, a hierarchical flow-matching model that combines these two complementary representations. An ESM-3-conditioned E(3)-equivariant graph network first generates sparse C$\alpha$ anchors in Cartesian space. A conditional midpoint Transformer then recursively fills the gaps in local internal coordinates. During recursive generation, the decoder constrains the distance between each midpoint and any sequence-adjacent parent anchor to the corresponding apo C$\alpha$-C$\alpha$ distance. On t
    
[^93]: FM4SAR-Bench 1.0：迈向SAR图像解译基础模型公平评估的基准

    FM4SAR-Bench 1.0: Towards A Fair Benchmark for Evaluating Foundation Models for SAR Image Interpretation

    [https://arxiv.org/abs/2611.27158](https://arxiv.org/abs/2611.27158)

    提出了FM4SAR-Bench 1.0，一个包含系统性分类体系及五个评级级别、六个维度分层评估框架的基准，用于对SAR图像解译基础模型进行公平、全面的评估。

    

    基础模型推动了SAR（合成孔径雷达）图像解译的进步，但也形成了一个异构且碎片化的模型生态系统。现有模型在输入、训练数据、训练目标、架构和发布策略等方面差异显著，然而该领域缺乏统一的分类体系和比较框架。当前的评估主要依赖特定任务的基准，这类基准并不适合对输入输出格式和任务范式存在根本差异的模型进行公平比较，且往往忽视了数据透明度、可复现性、可解释性和部署就绪性。此外，SAR数据可获取性有限，也意味着较高的报告性能并不保证能获得相应的训练数据、代码或模型权重，从而限制了模型的复现、基准测试和实际部署。为解决这些问题，我们提出了一个针对SAR相关基础模型的系统性分类体系和分层评估框架：包含五个评级级别、六个维度以及细粒度的评估指标。

    arXiv:2611.27158v1 Announce Type: cross  Abstract: Foundation models have advanced SAR image interpretation but created a heterogeneous, fragmented model ecosystem. Existing models differ substantially in their input, training data, objectives, architectures, and release policy, yet the field lacks a unified taxonomy and comparative framework. Current evaluations rely primarily on task-specific benchmarks, which are ill-suited to fair comparison among models with fundamentally different input-output formats and task paradigms, and often overlook data transparency, reproducibility, interpretability, and deployment readiness. Limited SAR data accessibility also means high reported performance does not guarantee access to training data, code, or weights, limiting reproduction, benchmarking, and deployment. To address these issues, we propose a systematic taxonomy and hierarchical evaluation framework for SAR-related foundation models: five rating levels, six dimensions, and fine-grained c
    
[^94]: 脆弱性窗口：量化奖励模型中的决策不稳定性

    The Vulnerability Window: Decision Instability in Quantized Reward Models

    [https://arxiv.org/abs/2611.27155](https://arxiv.org/abs/2611.27155)

    量化在总体基准准确率几乎不变的情况下仍会翻转1.1-12.3%的奖励模型成对偏好决策，其翻转率可用高斯近似Φ(-|m|/σ)刻画，且脆弱性窗口占比随模型规模从25%降至7%，揭示了小型裁判尤其脆弱的隐患。

    

    奖励模型和大语言模型裁判通常被量化到4比特，以便在为大量候选打分时保持成本可控，而常规的验证方式是确认基准测试准确率几乎不变。这种验证方式近乎盲目。在来自两个系列的八个奖励模型（0.6B-8B）、三个生成式裁判以及四种4比特量化方法上，量化会改写RewardBench上1.1-12.3%的成对偏好决策，而总体准确率的变化却小至0.03分（p = 1.00），这是因为量化所修复的错误与其引入的错误相互抵消。决策翻转集中在全精度边际相对于量化噪声尺度sigma较小时的位置：高斯近似Phi(-|m|/sigma)能够描述总体翻转率，但在所有30个标量设置中都会高估该比率，并且对生成式裁判失效；由此产生的脆弱性窗口的占比随模型规模的增大从25%降至7%，这正是小型裁判脆弱的原因。在良性最佳情况下，这种变动保持在噪声范围内……

    arXiv:2611.27155v1 Announce Type: cross  Abstract: Reward models and LLM judges are routinely quantized to 4 bits so that scoring many candidates stays affordable, and the usual check is that benchmark accuracy barely moves. That check is close to blind. Across eight reward models from two families (0.6B-8B), three generative judges and four 4-bit quantizers, quantization rewrites 1.1-12.3% of pairwise preference decisions on RewardBench while aggregate accuracy moves by as little as 0.03 points (p = 1.00), because the errors it fixes and introduces cancel. Flips concentrate where the full-precision margin is small relative to the quantization noise scale sigma: a Gaussian approximation Phi(-|m|/sigma) describes the aggregate rate, over-predicting it in all 30 scalar settings and failing for generative judges, and occupancy of the resulting vulnerability window falls from 25% to 7% with model size, which is what makes small judges fragile. The churn stays within noise under benign best
    
[^95]: Git历史是一个泛基因组：代码训练语料中的重复现象以及去重所移除、保留与遗漏的内容

    Git History Is a Pangenome: Repetition in Code Training Corpora and What Deduplication Removes, Keeps and Misses

    [https://arxiv.org/abs/2611.27154](https://arxiv.org/abs/2611.27154)

    该论文揭示Git仓库历史本质上是一个高度重复的“泛基因组”——典型仓库历史相当于其当前代码的19份副本，而专为快照设计的精确匹配去重方法完全无法移除这些历史重复内容。

    

    代码模型越来越多地在软件仓库的完整历史上进行训练，即每个文件的每个版本，而不是在单一快照上训练。用于训练数据去重的工具是为快照而构建的，它们对历史数据的作用尚未被测量。我们在预先注册的306个公共仓库随机样本、25个成熟项目以及SWE-bench基准背后的12个仓库上进行了测量。历史数据大部分是重复内容：用文本索引的术语来说，是一个泛基因组，即一份文本的众多近乎相同的变体。一个典型仓库的历史包含相当于其当前代码19份副本的内容，而一个成熟仓库则包含89份。原因在于每次编辑只改动几行，文件的其余部分保持不变：被修改文件中十分之九的token在变更之前就已存在。精确匹配去重无法移除这些副本中的任何一个，因为没有两个版本是完全相同的。大型训练流水线……（原文摘要在此处被截断）

    arXiv:2611.27154v1 Announce Type: cross  Abstract: Code models are increasingly trained on the full history of software repositories, every version of every file, rather than on a single snapshot. The tools that deduplicate training data were built for snapshots, and what they do to a history has not been measured. We measured it on a pre-registered random sample of 306 public repositories, 25 mature projects, and the 12 repositories behind the SWE-bench benchmark. A history is mostly repetition: in the vocabulary of text indexing, a pangenome, one text in many near-identical variants. The history of a typical repository holds the equivalent of 19 copies of its current code, and a mature repository holds 89. The reason is that each edit changes a few lines and leaves the rest of the file in place: nine tokens in ten of a changed file already existed before the change. Exact-match deduplication removes none of these copies, since no two versions are identical. Large training pipelines a
    
[^96]: 结构重于规模：重新思考可验证奖励强化学习中的适配方法

    Structure Over Scale: Rethinking Adaptation for Reinforcement Learning with Verifiable Rewards

    [https://arxiv.org/abs/2611.27148](https://arxiv.org/abs/2611.27148)

    该研究发现在可验证奖励强化学习训练中，LoRA会随秩增大出现性能骤降的“秩坍缩”现象，而DoRA和QuanTA等结构化适配方法凭借更优的更新几何结构，仅更新不到0.6%的参数即可达到与全参数微调相当的效果，表明更新的结构比规模更关键。

    

    可验证奖励强化学习通常被框架化为在全参数微调的表达能力与低秩适配的高效率之间做选择。我们发现这一框架忽略了一个重要变量：更新的几何结构。在常见的冷启动GRPO协议下，LoRA表现出“秩坍缩”现象：性能在低秩时大致保持稳定，但随着秩的增大却会突然失效。例如在DeepMath-Hard数据集上，准确率从r=64时的73.1%骤降至r=128时的4.7%。相比之下，权重分解低秩适配和量子启发张量适配在更广的测试优化区域内保持稳定，并且在仅更新不到0.6%模型参数的情况下与全参数微调表现相当。谱分析揭示了所学更新结构的相应差异：发生坍缩的LoRA将更新能量集中在相对较少的有效方向上，而结构化适配器……

    arXiv:2611.27148v1 Announce Type: cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) is often framed as a choice between the expressivity of Full Fine-Tuning (FFT) and the efficiency of Low-Rank Adaptation (LoRA). We find that this framing misses an important variable: the geometry of the update. Under a common cold-start GRPO protocol, LoRA exhibits rank collapse: performance remains broadly stable at low rank, then fails abruptly as rank increases. On DeepMath-Hard, for example, accuracy drops from 73.1% at r=64 to 4.7% at r=128. In contrast, Weight-Decomposed Low-Rank Adaptation (DoRA) and Quantum-Informed Tensor Adaptation (QuanTA) remain stable over a wider tested optimization region and are competitive with FFT while updating less than 0.6% of the model parameters. Spectral analysis reveals a corresponding difference in learned update structure: collapsed LoRA concentrates its update energy into relatively few effective directions, whereas structured adapters 
    
[^97]: GaLF-ST：用于空间转录组学重建的高斯潜在场

    GaLF-ST: Gaussian Latent Fields for Spatial Transcriptomics Reconstruction

    [https://arxiv.org/abs/2611.27143](https://arxiv.org/abs/2611.27143)

    该论文提出GaLF-ST，一种利用可学习的高斯中心和各向异性协方差构建高斯潜在场、对每个位置的潜在向量进行加权以实现空间转录组学基因表达重建的方法。

    

    空间转录组学（ST）在已知组织位置测量基因表达，以揭示组织结构与细胞功能。表达预测需要对局部空间关系以及基因之间的关联同时建模。现有方法通常使用坐标网络对表达进行建模，缺乏显式的空间参数化。为了解决ST空间采样有限和表达观测稀疏的问题，一些方法先拟合中间嵌入，再训练表达预测器，但适合表达重建的潜在表示可能难以从空间坐标中预测得到，且这两个训练阶段可能对潜在表示提出不同的要求。在本文中，我们提出GaLF-ST，一种高斯潜在场方法，其中可学习的高斯中心和各向异性协方差决定每个位置潜在向量的权重，共享的非线性解码器对其加权结果进行映射……（摘要在此处截断）

    arXiv:2611.27143v1 Announce Type: new  Abstract: Spatial transcriptomics (ST) measures gene expression at known tissue locations to reveal tissue organization and cellular function. Expression prediction requires modeling both local spatial relationships and associations among genes. Existing methods typically model expression using coordinate networks without explicit spatial parameterization. To address limited spatial sampling and sparse expression observations of ST, some methods firstly fit intermediate embeddings and then train expression predictors, but a latent representation suited to expression reconstruction may be difficult to predict from spatial coordinates. The two training stages can impose different demands on the latent representation. In this paper, we propose GaLF-ST, a Gaussian latent field method in which learnable Gaussian centers and anisotropic covariances determine the weights of latent vectors at each location. A shared nonlinear decoder maps their weighted a
    
[^98]: 面向长尾图分类的动态记忆增强模式相关性预测

    Dynamic Memory-enhanced Pattern Relevance Prediction for Long-tailed Graph Classification

    [https://arxiv.org/abs/2611.27141](https://arxiv.org/abs/2611.27141)

    提出DMPRP方法，通过动态专家记忆模块将头部图的结构模式迁移到尾部图，从而有效提升长尾图分类任务中尾部类的性能。

    

    长尾图分类在真实世界的图数据集中十分常见，其中少量较大的头部图提供丰富的结构信息，而大量较小的尾部图提供的结构线索较少，这使得为尾部图学习判别性表示变得更加困难。为了解决这种基于规模的不平衡问题，我们提出了动态记忆增强模式相关性预测（DMPRP），该方法将结构模式从头图迁移到尾部图。给定一个图表示作为查询，一个受知识多样专家启发的动态专家记忆模块生成一个残差向量来增强原始表示。我们使用从头图在节点、根子图和随机游走邻域层面采样的共现三元组来训练该模块。在五个图分类基准上的实验表明，DMPRP 提升了尾部类的性能，消融研究表明专家模块和随机游走（原文此处截断）……

    arXiv:2611.27141v1 Announce Type: new  Abstract: Long-tailed graph classification (LTGC) is common in real-world graph datasets, where a few large head graphs provide rich structural information while many smaller tail graphs offer fewer structural cues, making it harder to learn discriminative representations for tail graphs. To address this size-based imbalance, we propose Dynamic Memory-enhanced Pattern Relevance Prediction (DMPRP), which transfers structural patterns from head graphs to tail graphs. Given a graph representation as a query, a dynamic expert memory module inspired by Knowledge-diverse Experts (KDEX) generates a residual vector that augments the original representation. We train the module using co-occurrence triplets sampled from head graphs at the node, rooted subgraph, and random-walk neighborhood levels. Experiments on five graph classification benchmarks show that DMPRP improves tail performance, and ablation studies show that both the expert module and random-wa
    
[^99]: 基于LAPIS-SHRED的稀疏观测滑坡运动时空重建

    Spatiotemporal Landslide Runout Reconstruction from Sparse Observations with LAPIS-SHRED

    [https://arxiv.org/abs/2611.27132](https://arxiv.org/abs/2611.27132)

    提出LAPIS-SHRED框架，仅凭单一时间点的32个稀疏传感器观测即可重建完整的滑坡运动时空演化轨迹（NRMSE低于5.8%），且与实际沉积物的空间相关性（0.843）超过了校准物理模拟器的表现。

    

    从有限的观测数据中重建滑坡运动过程的时空演化是地质灾害评估中的核心挑战：瞬态流动动力学很少能被直接观测到，且可用数据通常仅限于事件后的沉积物或短暂的监测窗口。我们将LAPIS-SHRED（基于浅层循环解码器的短时序潜相位推断方法）应用于美国圣海伦斯火山附近的滑坡运动事件重建。该框架在基于物理的模拟集合上训练，支持从单帧进行反向重建以及从短初始窗口进行前向预测，并可适用于不同的滑坡运动状态。仅使用单一时间点的32个传感器观测数据，LAPIS-SHRED即可恢复完整的滑坡运动轨迹，NRMSE低于5.8%，且在不同的观测和训练条件下均保持稳健。与实际观测沉积物的对比验证显示，其空间相关性达r = 0.843，高于校准后的模拟器的r = 0.8

    arXiv:2611.27132v1 Announce Type: cross  Abstract: Reconstructing the spatiotemporal evolution of landslide runout from limited observations is a central challenge in geohazard assessment: transient flow dynamics are rarely observed directly, and available data are often confined to post-event deposits or brief monitoring windows. We apply LAPIS-SHRED (LAtent Phase Inference from Short time sequence using SHallow REcurrent Decoders) to reconstruct runout events near Mount St. Helens, USA. Trained on physics-based simulation ensembles, the framework supports both backward reconstruction from single frame and forward prediction from short initial window across different runout regimes. Using 32 sensor observations from a single timepoint, LAPIS-SHRED recovers complete runout trajectories with NRMSE below 5.8%, robust across varying observational and training conditions. Validation against observed deposits yields spatial correlation of r = 0.843 against the calibrated simulator's r = 0.8
    
[^100]: 带误差的蒸馏：面向知识蒸馏的稀疏无偏反向传播

    Distillation with Errors: Sparse Unbiased Backpropagation for Knowledge Distillation

    [https://arxiv.org/abs/2611.27131](https://arxiv.org/abs/2611.27131)

    本文提出带误差蒸馏（DWE），利用 logit 梯度的零和结构构建稀疏成对的无偏梯度估计，并结合几何感知配对与分层抽样降低方差，从而大幅削减全词表知识蒸馏中输出头反向传播的计算成本。

    

    知识蒸馏（KD）通过训练学生模型来匹配教师模型的输出分布。然而，全词表知识蒸馏需要对输出头进行代价高昂的反向传播。为降低这一成本，我们提出了带误差蒸馏（Distillation with Errors, DWE）。我们的关键洞察是：小批量训练本身已是对训练梯度的近似，因此逐位置的反向传播无需完全精确。为此，DWE 利用 logit 梯度的零和结构，构建稀疏的成对梯度估计，其期望与全词表梯度一致；随后通过几何感知配对和分层抽样进一步降低估计方差。我们证明了该估计器是无偏的，并评估了 DWE 的梯度估计精度与训练效率。在相同的抽样预算下，与重要性抽样相比，DWE 将隐藏状态和输出权重梯度的均方误差（MSE）分别降低了 63.5% 和 57.9%。在离策略（off-policy）知识蒸馏中，它实现了最高达 1.95 倍的加速提升。

    arXiv:2611.27131v1 Announce Type: new  Abstract: Knowledge distillation (KD) trains a student to match a teacher's output distribution. However, full-vocabulary KD requires costly output-head backpropagation. To reduce this cost, we introduce Distillation with Errors (DWE). Our key insight is that minibatch training already approximates the training gradient, so per-position backpropagation does not need to be exact. DWE therefore exploits the zero-sum structure of logit gradients to construct sparse paired gradient estimates whose expectation matches the full-vocabulary gradient. It then uses geometry-aware pairing and stratified sampling to further reduce estimation variance. We prove that the estimator is unbiased and evaluate DWE's gradient estimation accuracy and training efficiency. At the same sampling budget, DWE reduces hidden-state and output-weight gradient MSE by 63.5% and 57.9%, respectively, compared with importance sampling. In off-policy KD, it achieves up to a $1.95\ti
    
[^101]: 是计划还是对抗样本？潜空间世界模型的往返检验

    A Plan or an Adversarial Example? A Round-Trip Check for Latent World Models

    [https://arxiv.org/abs/2611.27124](https://arxiv.org/abs/2611.27124)

    针对潜空间世界模型规划可能产生“世界模型对抗样本”而非有效计划的问题，提出往返检验方法——用小网络从预测状态读回动作并惩罚不匹配，以识别规划器解决后又丢失任务的失效情况。

    

    如果我们优化一张图像，直到分类器确信它显示的是一只猫，我们得到的可能是一个对抗样本，而不是一只猫。使用潜空间世界模型进行规划是一个类似的搜索过程——寻找其预测未来最接近目标的动作序列。那么结果是计划，还是世界模型的对抗样本？我们证明两者皆有可能。在公开的 DINO-WM 和 LeWM 上，交叉熵方法（CEM）的更多迭代会降低预测成本，但会丢失 CEM 已经解决的多达一半的问题。模型自身的预测无法揭示这些失败。我们提出一个简单的测试：在每次迭代后执行该计划，并统计规划器曾解决后又丢失的问题数量。与图像不同，计划自带其动作，我们的往返检验通过一个小网络从预测状态中读回这些动作，并对不匹配进行惩罚。该检验读回每五个动作为一组（一块）的动作之和，即由两个连续预测状态共同决定的部分……

    arXiv:2611.27124v1 Announce Type: new  Abstract: If we optimize an image until a classifier is confident it shows a cat, we may get an adversarial example, not a cat. Planning with a latent world model is a similar search for actions whose predicted future is closest to the goal. Is the result a plan, or an adversarial example of the world model? We show that it can be either. On the released DINO-WM and LeWM, more iterations of the cross-entropy method (CEM) lower the predicted cost but lose up to half of the problems CEM had already solved. The model's own predictions do not reveal these failures. We propose a simple test that executes the plan after every iteration and counts the problems a planner solves and then loses. Unlike an image, a plan comes with its actions, which our round-trip check reads back from the predicted states with a small network, penalizing mismatches. The check reads back the sum of each block of five actions, the part that two consecutive predicted states de
    
[^102]: CTRL：通过GPU共存实现LLM强化学习的并发训练与Rollout

    CTRL: Concurrent Training and Rollout for LLM Reinforcement Learning via GPU Co-Residency

    [https://arxiv.org/abs/2611.27119](https://arxiv.org/abs/2611.27119)

    CTRL提出将GPU共存作为LLM强化学习的第三种执行布局，让每张GPU同时常驻运行learner与rollout引擎，并通过主机侧learner、发布门控和步时上界实现训练与推理的高效并发。

    

    大语言模型的强化学习（RL）后训练需要运行两种硬件需求不同的GPU工作负载：rollout解码，它以内存访问为主导，且其批次大小会随着响应完成而不断缩小；以及稠密的学习器（learner）。现有系统要么让每张GPU在两者之间进行时分复用，付出串行阶段和切换的开销；要么将GPU划分成资源池，但其规模只能以整卡为单位变化。我们提出CTRL，将共存作为第三种RL执行布局：每张GPU在常驻的rollout引擎旁边运行一个learner rank，并在CUDA MPS下共享设备。主机侧拥有的learner使这一布局得以落地：优化器状态保存在主机内存中，策略权重通过一个有界的GPU窗口逐层流式传输。发布门控（publication gate）先训练每个就绪的批次，仅在发布下一版策略之前等待，从而保持无丢弃的陈旧性上界。步时上界（step-time bound）指导计算份额的选择，GPU时间账本（GPU-time ledger）则解释了重叠情况。

    arXiv:2611.27119v1 Announce Type: cross  Abstract: Reinforcement learning (RL) post-training of large language models runs two GPU workloads with different hardware demands: rollout decoding, which is dominated by memory access and whose batch shrinks as responses finish, and a dense learner. Current systems either time-share every GPU between them, paying serial phases and switches, or split GPUs into pools whose sizes change only in whole devices. We present CTRL, which makes co-residency a third RL execution layout: every GPU runs a learner rank beside a persistent rollout engine, sharing the device under CUDA MPS. A host-owned learner makes the layout fit: optimizer state stays in host memory, and policy weights stream layer by layer through a bounded GPU window. A publication gate trains each ready batch first and waits only before releasing the next policy, keeping a dropless staleness bound. A step-time bound guides compute-share selection, and a GPU-time ledger explains overlap
    
[^103]: 基于三角不等式剪枝的高效纵向联邦聚类

    Highly Efficient Vertical Federated Clustering with Triangle-Inequality-based Pruning

    [https://arxiv.org/abs/2611.27117](https://arxiv.org/abs/2611.27117)

    该论文提出了基于三角不等式的精确剪枝方法TriCD和高效联邦聚类算法FedTriCD，通过将全局目标分解为客户端本地子问题，在保证聚类结果一致的前提下显著减少了通信轮次与本地计算开销。

    

    在许多现实场景中，多个参与方持有相同样本的不同特征子集，但无法共享原始数据，这使得在分布式特征子集上进行高效聚类成为一个重要且具有挑战性的问题。虽然纵向联邦学习（VFL）能够实现此类协作，但现有的VFL聚类方法通常需要大量的通信轮次或庞大的本地计算开销。为解决这些局限，我们首先提出了TriCD，这是一种针对CDKM的精确剪枝方法，在相同初始化条件下能够保持完全一致的聚类分配结果。TriCD按需构建三角不等式界，以跳过冗余的目标函数计算，且额外空间开销可以忽略不计。在此基础上，我们提出了高效的联邦聚类算法FedTriCD。通过将全局目标函数分解为客户端本地子问题，FedTriCD仅利用本地数据即可实现基于剪枝的本地优化。FedTriCD还进一步减少了通信开销（摘要在此处被截断）。

    arXiv:2611.27117v1 Announce Type: cross  Abstract: In many real-world scenarios, multiple parties hold complementary feature subsets for the same samples but cannot share raw data. This makes efficient clustering over distributed feature subsets an important and challenging problem. While vertical federated learning (VFL) enables such collaboration, existing VFL clustering methods often require many communication rounds or substantial local computation. To address these limitations, we first propose TriCD, an exact pruning method for CDKM that preserves the same cluster assignments under the same initialization. TriCD constructs triangle-inequality bounds on demand to skip redundant objective evaluations with negligible extra space. Building upon this, we propose FedTriCD, an efficient federated clustering algorithm. By decomposing the global objective into client-local subproblems, FedTriCD enables pruning-based local optimization using only local data. FedTriCD further reduces commun
    
[^104]: 联邦平衡k均值算法：通信效率、隐私与异构性

    Federated and Balanced $k$-means: Communication Efficiency, Privacy, and Heterogeneity

    [https://arxiv.org/abs/2611.27114](https://arxiv.org/abs/2611.27114)

    本文提出Fabric，一种平衡且保护隐私的联邦k均值算法，通过ADMM将平衡聚类分解为各客户端独立子问题，并借助校准噪声注入实现可证明的隐私保证，解决了现有垂直联邦聚类方法信息泄露和聚类不平衡的问题。

    

    在医疗和金融领域进行跨组织的数据协作，需要既能保护隐私又能支持有效分析的方法。垂直联邦学习使得各方无需共享原始数据，即可对同一实体的不同特征数据进行聚类。然而，现有的垂直联邦聚类方法存在两个关键局限：容易发生信息泄露，以及会产生不平衡的聚类。为解决这些局限，我们提出了Fabric，一种平衡且保护隐私的联邦k均值算法。我们的方法将平衡聚类问题分解为每个客户端的独立子问题，并利用交替方向乘子法（ADMM）高效处理平衡约束。在半诚实模型下，Fabric通过在局部目标评估中注入经过校准的噪声，提供了可证明的隐私保证。在十个真实数据集上的广泛评估……

    arXiv:2611.27114v1 Announce Type: new  Abstract: Collaborating on data across organizations in healthcare and finance demands methods that preserve privacy while enabling effective analysis. Vertical Federated Learning (VFL) facilitates clustering across data where different parties hold different features of the same entities without sharing raw data. However, existing VFL clustering methods suffer from two key limitations: vulnerability to information leakage and the creation of imbalanced clusters. To address these limitations, we propose Fabric, a balanced and privacy-preserving federated $k$-means algorithm. Our method decomposes the balanced clustering problem into independent subproblems for each client, using the Alternating Direction Method of Multipliers (ADMM) to efficiently handle the balance constraints. Under the semi-honest model, Fabric provides provable privacy guarantees by injecting calibrated noise into local objective evaluations. Extensive evaluations on ten real-
    
[^105]: 查询驱动擦除中的循环放大与精确非扩张性判据

    Recurrent Amplification and Exact Nonexpansiveness Criteria for Query-Driven Erasure

    [https://arxiv.org/abs/2611.27109](https://arxiv.org/abs/2611.27109)

    该论文的核心贡献是构造了单步收缩但周期乘积却产生放大的周期词元序列，证明查询驱动的擦除机制可引发循环放大，并首次给出了可在 O(d) 时间内计算的秩一擦除一步非扩张性的充要谱范数判据。

    

    查询衍生擦除方向在秩一线性注意力更新的基础上引入了一个与键正交的查询分量。在标量衰减下，这一修正不改变每个齐次状态转移的特征值，而初始扰动的传播则取决于这些转移的有序乘积。我们构造了一个具有固定键投影和查询投影的周期性词元序列，其中每一步转移的谱半径都小于一，但整个周期乘积的谱半径却大于一。同一构造还给出了在独立均匀词元采样下均方增长的充分条件。对于带逐通道衰减的秩一擦除，我们推导出了一个一步非扩张性的充分必要谱范数判据，该判据可在 O(d) 时间内完成计算。在全部 1,800 个被评估的算子上，该判据与直接奇异值分解的结果完全一致。在对选定的某个注意力头的反事实重放中……

    arXiv:2611.27109v1 Announce Type: new  Abstract: The Query-derived Erase Direction (QED) augments a rank-one linear attention update with a key-orthogonal query component. Under scalar decay, this correction leaves the eigenvalues of each homogeneous state transition unchanged, while the propagation of an initial perturbation depends on the ordered product of these transitions. We construct a periodic token sequence with fixed key and query projections for which every one-step spectral radius is below one but the spectral radius of the period product exceeds one. The same construction yields a sufficient condition for mean-square growth under independent uniform token sampling. For rank-one erasure with channel-wise decay, we derive a necessary and sufficient spectral-norm test for one-step nonexpansiveness that can be evaluated in $O(d)$ time. The test agrees with direct singular value decomposition on all 1,800 operators evaluated. In a counterfactual replay of a selected head of Qwe
    
[^106]: 通过错误发现率控制实现可靠的行为发现

    Reliable Behavior Discovery via False Discovery Rate Control

    [https://arxiv.org/abs/2611.27104](https://arxiv.org/abs/2611.27104)

    提出了DISCO方法，利用粗化e-过程的迭代搜索，在温和条件下可证明地控制行为发现中的错误发现率，并通过适应数据依赖的假设数量实现了比以往eBH方法更高的检验效能。

    

    随着大型生成式模型变得日益复杂和强大，发现其中行为模式的可靠方法变得越来越重要。我们将行为发现形式化为一类值得关注的问题，并强调控制错误发现率（FDR）是保证行为发现可靠性的关键任务。基于多重假设检验领域的文献，我们提出了DISCO（Discovery through Iterative Search with COarsened e-processes，即通过粗化e-过程的迭代搜索进行发现），这是一种在温和条件下能够可证明地控制行为发现中FDR的方法。DISCO通过允许适应数据依赖的待检验假设数量，相较于之前基于eBH的相关方法提升了检验效能。我们在合成数据以及一项针对大型生成式模型同时作为发现者和发现对象的场景下FDR控制方法的新颖实证评估中，通过实验证明了DISCO能以比可比方法更高的效能控制FDR。

    arXiv:2611.27104v1 Announce Type: new  Abstract: Reliable methods for discovering behavior patterns in large generative models are increasingly essential as models become both more complex and capable. We formalize behavior discovery as a problem class of interest and highlight the task of controlling false discovery rate (FDR) as crucial for guaranteeing reliability in behavior discovery. Building upon the multiple hypothesis testing literature, we propose DISCO (Discovery through Iterative Search with COarsened $e$-processes), a method which provably controls FDR in behavior discovery under mild conditions. DISCO improves power over previous related eBH-based approaches by allowing for adaptivity to a data-dependent number of hypotheses to be tested. We demonstrate empirically that DISCO controls FDR with higher power than comparable approaches on both synthetic data and a novel empirical evaluation of FDR-controlling methods in settings where large generative models act as both disc
    
[^107]: EgoSurgHands：面向开放手术训练的第一人称视角3D手部姿态数据集与基准

    EgoSurgHands: An Egocentric 3D Hand Pose Dataset & Benchmark for Open Surgery Training

    [https://arxiv.org/abs/2611.27101](https://arxiv.org/abs/2611.27101)

    该论文提出了首个提供真实采集的第一人称视角3D手部姿态标注的开放手术训练数据集与基准EgoSurgHands，可支持手术技能评估、AR引导和机器人示范学习。

    

    手术技能评估、AR增强现实引导以及从示范中学习的机器人训练都受益于对外科医生手部的3D跟踪，然而目前尚无手术数据集能够提供真实采集的第一人称视角关节式3D手部姿态。我们提出了EgoSurgHands，这是一个用于在合成手术垫上进行开放手术训练的真实采集第一人称手部姿态估计基准。该数据集通过多传感器眼镜采集，包含来自55段手术录像的41,909条训练手部实例数据行，以及来自13段独立保留测试录像的9,476条经人工验证的评估数据行，涵盖13个手术任务、3种手套颜色，以及从一年级医学生到三年级住院医师共4名受训者。每一帧均携带多传感器数据堆栈：相机内参、带有每只手置信度信息的手部关键点、时间戳和眼动注视数据，并在SLAM成功运行的情况下附带SLAM外参。评估标签结合了传感器融合的伪真值、对每一帧的人工交叉审查、外科医生对标记疑难案例的裁定，以及一个独立的……

    arXiv:2611.27101v1 Announce Type: cross  Abstract: Surgical skill assessment, AR guidance, and robot learning from demonstration all benefit from 3D tracking of the surgeon's hands, yet no surgical dataset offers real egocentric articulated 3D hand pose. We introduce EgoSurgHands, a benchmark for real-captured egocentric hand pose estimation in open surgery training on synthetic pads. Captured with multi-sensor glasses, it contains 41,909 training hand-instance rows across 55 procedure recordings and 9,476 human-validated evaluation rows across 13 held-out recordings, spanning 13 tasks, 3 glove colors, and 4 trainees from first-year medical student to third-year resident. Each frame carries a multi-sensor stack: intrinsics, hand landmarks with per-hand confidence, timestamps, and eye gaze, plus SLAM extrinsics where SLAM succeeded. Evaluation labels combine sensor-fused pseudo-ground truth, human cross-review of every frame, surgeon adjudication of flagged cases, and an independent mul
    
[^108]: 二维推理：量化思维链的忠实性与效率

    Reasoning in Two Dimensions: Quantifying Chain-of-Thought Faithfulness and Efficiency

    [https://arxiv.org/abs/2611.27100](https://arxiv.org/abs/2611.27100)

    该论文提出将思维链表示为语义推理图，并沿“语义支持对推理历史”与“通往答案对偏离路径”两个维度分解模型的预测依赖性，在同一框架下实现了对思维链非忠实性与低效性的全局及步骤级量化。

    

    思维链（CoT）推理会展示中间推理步骤，但答案正确并不能证明大语言模型确实依赖这些步骤所表达的依赖关系，也不能证明其推理是高效的。我们在一个统一的框架内对CoT的忠实性和效率进行了可操作化定义。具体而言，我们将思维链表示为一个语义推理图，并沿两个互补的维度分解模型的预测依赖性：由语义支持所解释的依赖与额外推理历史带来的依赖，以及分配给通往答案的推理与偏离路径推理的依赖。这一分解产生了全局和步骤级别的非忠实性与低效性度量。受控干预实验表明，所提出的指标能够按预期响应隐藏的依赖关系和诱导出的偏离路径推理。在数学和多跳推理任务中，答案级别的非忠实性始终与更大的预测不确定性相关，而非忠实性和……（摘要在此处截断）

    arXiv:2611.27100v1 Announce Type: new  Abstract: Chain-of-thought (CoT) reasoning exposes intermediate steps, but a correct answer does not establish that an LLM relies on the dependence expressed by those steps or reasons efficiently. We operationalize CoT faithfulness and efficiency within a common framework. Specifically, we represent a CoT as a semantic reasoning graph and decompose the model's predictive dependence along two complementary axes: dependence explained by semantic support versus additional reasoning history, and dependence allocated to answer-reaching versus off-path reasoning. This decomposition yields global and step-level measures of unfaithfulness and inefficiency. Controlled interventions show that the proposed metrics respond to hidden dependence and induced off-path reasoning as intended. Across mathematical and multi-hop reasoning tasks, answer-level unfaithfulness is consistently associated with greater predictive uncertainty, whereas both unfaithfulness and 
    
[^109]: PLoQ：面向大语言模型激活值4比特量化的提示词专属低秩基

    PLoQ: A Prompt-Specific Low-Rank Basis for 4-bit Quantization of LLM Activations

    [https://arxiv.org/abs/2611.27092](https://arxiv.org/abs/2611.27092)

    PLoQ提出了一种提示词自适应的低秩分解方法，在预填充阶段为每个提示词动态估计低秩基，将主导激活分量单独量化并在矩阵乘法中恢复，从而在无需修改权重的情况下实现大语言模型激活值的4比特量化。

    

    大语言模型的低比特激活量化仍然具有挑战性，因为少数主导方向决定了量化范围。最先进的方法通过离线校准并折叠进权重的变换将这些方向隔离在高精度子空间中，这为所有提示词固定了该子空间，导致每个提示词的少数独特方向得不到保护，并且在权重无法吸收该变换的情况下无法应用。我们提出PLoQ，这是一种提示词自适应的低秩分解方法，它在量化之前提取主导激活分量，对其进行单独量化，并在层的矩阵乘法运算中恢复它们，而无需修改权重。低秩分量的基在预填充阶段从每个提示词中估计，并在整个解码过程中重复使用。在与最先进的混合精度量化相同的位宽下（4比特权重、激活值和K/V缓存，以及八分之一的通道……）

    arXiv:2611.27092v1 Announce Type: cross  Abstract: Low-bit activation quantization of large language models remains challenging, as a small number of dominant directions dictate the quantization range. State-of-the-art methods isolate these directions in a high-precision subspace through transformations calibrated offline and folded into the weights, which fixes the subspace for all prompts, leaving each prompt's few unique directions unprotected, and rules it out wherever no weight can absorb the transformation. We present PLoQ, a prompt-adaptive low-rank decomposition that extracts the dominant activation components before quantization, quantizes them separately, and restores them within the layer's GEMM, without modifying the weights. The low-rank component's basis is estimated from each prompt during prefill and reused throughout decoding. At the same bit width as state-of-the-art mixed-precision quantization (4-bit weights, activations and K/V cache with an eighth of the channels 
    
[^110]: 直接写入权重还是训练权重？基于权重空间编程与低秩自适应的空间推理

    Writing Weights or Training Them? Spatial Reasoning through Weight-Space Programming and Low-Rank Adaptation

    [https://arxiv.org/abs/2611.27089](https://arxiv.org/abs/2611.27089)

    在空间推理任务上，权重空间编程带来的提升有限，而仅用0.09%可训练参数、以解题示例为监督训练的秩1 LoRA适配器可近乎完美地解决参照系与StepGame任务，并对未见模板和更高难度展现一定泛化能力。

    

    权重空间编程使一个语言模型能够检查并编辑另一个模型的权重，为基于梯度的自适应方法提供了一种替代方案。我们使用Qwen3.5-0.8B在固定参照系任务（128题）、StepGame（200题）以及两个未经训练的距离比较任务上对两种方法进行了评估。在开启思考模式的条件下，公开补丁在三个采样种子下将参照系任务的平均准确率从0.0%提升至22.1%，将StepGame的准确率从19.2%提升至33.7%。随机打乱其权重改变会消除这一增益。进一步的闭式编辑未能超越p50c，而基于解题示例训练的秩1 LoRA适配器在两个任务上分别达到了100%和98.8%的准确率，且可训练参数仅占0.09%。若从监督信号中移除解题示例，适配器则无法通过学习门槛。训练后的适配器在未见过的StepGame句子模板上达到82%的准确率，在训练后留出的6-10难度级别上达到42%的准确率。

    arXiv:2611.27089v1 Announce Type: new  Abstract: Weight-space programming lets a language model inspect and edit another model's weights, offering an alternative to gradient-based adaptation. We evaluated both approaches using Qwen3.5-0.8B on a fixed reference-frame task (128 items), StepGame (200 items), and two untrained distance-comparison tasks. With thinking enabled, the public patch (p50c) raised mean accuracy from 0.0% to 22.1% on the reference-frame task and from 19.2% to 33.7% on StepGame across three sampling seeds. Randomly permuting its weight change removed the gain. Further closed-form edits did not improve on p50c, whereas a rank-1 LoRA adapter trained on worked solutions reached 100% and 98.8%, respectively, with 0.09% trainable parameters. Removing the worked solutions from supervision prevented the adapter from passing the learning gate. The trained adapters reached 82% accuracy on unseen StepGame sentence templates, 42% on held-out difficulty levels 6-10 after traini
    
[^111]: ORION：面向Transformer系统性泛化的精确规范态监督

    ORION: Exact Canonical-State Supervision for Systematic Generalization in Transformers

    [https://arxiv.org/abs/2611.27087](https://arxiv.org/abs/2611.27087)

    ORION框架通过辅助头监督精确的规范执行状态来恢复Transformer的深度外推能力（66.7%），与解决长度外推的RoPE形成正交互补，二者结合在长度-深度联合外推中实现严格的加性增益。

    

    自回归Transformer在分布外推理中普遍失败，主要归因于两个因素：长序列上的位置损坏以及深层执行步骤中的结构性状态漂移。我们提出了ORION，一个通过辅助头对精确规范执行状态进行监督的框架，将潜在轨迹锚定到有效的计算流形上。在Folding、GridShift和CLRS-30基准上，一项预注册的2×2因子实验表明，RoPE和ORION解决的是相互正交的两个维度：RoPE恢复了长度外推能力（59.8%），而ORION恢复了深度外推能力（66.7%）。此外，两者结合在长度-深度联合外推中实现了严格的加性增益，而单独使用任一方法只能取得接近满意的效果。阴性对照实验证实该增益严格依赖于精确的规范对齐；基于哈希的状态统计无法约束深度外推。

    arXiv:2611.27087v1 Announce Type: cross  Abstract: Autoregressive Transformers generally fail on out-of-distribution reasoning due to two primary reasons: positional corruption over long sequences and structural state drift across deep execution steps. We introduce ORION, a framework that supervises exact canonical execution states via auxiliary heads to anchor latent trajectories to valid computational manifolds. Across Folding, GridShift, and CLRS-30, a pre-registered 2 x 2 factorial shows that RoPE and ORION address orthogonal axes. RoPE recovers length extrapolation (59.8%), while ORION recovers depth extrapolation (66.7%). Moreover, their combination complementarily achieves strict additive gain in joint length-depth extrapolation, while either method alone offers near satisfactory results. Negative controls confirm that the gain depends strictly on exact canonical alignment; hashed state statistics fail to constrain depth extrapolation.
    
[^112]: 审计并适配kNNGuard以应对长上下文提示

    Auditing and Adapting kNNGuard for Long-Context Prompts

    [https://arxiv.org/abs/2611.27084](https://arxiv.org/abs/2611.27084)

    本文审计了无需训练的防护栏kNNGuard在长上下文提示上的性能退化现象，发现其在长输入上AUROC大幅下降，并将退化归因于隐藏状态漂移、MiniLM的256 token硬截断以及长文档压缩为单一最终token向量导致的信号稀释这三个原因。

    

    arXiv:2611.27084v1 公告类型：新论文 摘要：诸如kNNGuard这类无需训练的激活防护栏，通过将大语言模型最后一个token的隐藏状态（层集成分支，kNNGuard LE）以及句子嵌入（all-MiniLM-L6-v2，Embedding-kNN）与一个小型参考库（每类50个样本）进行对比来对提示进行评分，并将这两个分支组合成融合集成方法（kNNGuard FE）。虽然在短输入（最多256个token）上表现有效，但标准kNNGuard在长输入（超过1000个token）上性能大幅下降：在Qwen3.5-4B和Gemma-4-E4B-it两个模型上取平均，kNNGuard LE的AUROC在内容有害性审核任务（Aegis-2.0）上从0.742（0.777和0.706）降至0.554（0.587和0.521），在RAG幻觉检测任务（HaluBench）上从0.994（0.995和0.992）降至0.556（0.598和0.513）。我们将这种性能退化归因于三个原因：(1) 随着序列长度增加，隐藏状态发生漂移；(2) MiniLM存在硬性的256个token截断限制；(3) 当长文档被压缩为单个最终token向量时，关键信号被稀释。

    arXiv:2611.27084v1 Announce Type: new  Abstract: Training-free activation guardrails such as kNNGuard score prompts by comparing last-token LLM hidden states (the Layer Ensemble branch, kNNGuard LE) and sentence embeddings (all-MiniLM-L6-v2, Embedding-kNN) against a small reference bank (50 examples per class), combining the two branches into a Fused Ensemble (kNNGuard FE). While effective on short inputs (up to 256 tokens), standard kNNGuard degrades substantially on long inputs (over 1000 tokens): averaged across Qwen3.5-4B and Gemma-4-E4B-it, kNNGuard LE AUROC drops from 0.742 (0.777 and 0.706) to 0.554 (0.587 and 0.521) on Content Harm Moderation (Aegis-2.0), and from 0.994 (0.995 and 0.992) to 0.556 (0.598 and 0.513) on RAG Hallucination Detection (HaluBench). We trace this degradation to three causes: (1) hidden-state drift as sequence length increases, (2) hard 256-token truncation in MiniLM, and (3) signal dilution when a long document is compressed into a single final-token ve
    
[^113]: 粒子物理中模型编辑的权重空间编程与机制可解释性

    Weight-space programming and mechanistic interpretability for model editing in particle physics

    [https://arxiv.org/abs/2611.27079](https://arxiv.org/abs/2611.27079)

    该论文通过稀疏自编码器识别潜在方向的移除和语言模型智能体的免梯度权重编辑两种方法，成功修复了粒子物理基础模型中因预训练数据问题导致的违反红外安全性缺陷，且对模型其余功能几乎无影响，所需修改的参数远少于微调。

    

    我们证明，权重空间编程以及由稀疏自编码器引导的编辑，可以通过直接修改权重来消除粒子物理基础模型中不理想的特性。在一个在喷注数据上训练的基础模型中，我们观察到添加单个软中性粒子会导致喷注分类的巨大变化，违反了红外安全性。我们通过机制分析定位了负责该问题的注意力头，并将该缺陷的起源追溯到预训练数据集中不同的重建阈值。随后我们提出了两种针对性的修复方法：一是移除通过稀疏自编码器识别出的三个潜在方向；二是让语言模型智能体直接编辑权重，无需梯度计算，且仅需对模型进行50次查询。两种方法的编辑各自都能消除几乎整个缺陷，同时使模型其余部分的响应几乎保持不变。与微调相比，这两种方法改变的参数要少得多。

    arXiv:2611.27079v1 Announce Type: cross  Abstract: We demonstrate that weight-space programming and edits guided by sparse autoencoders can remove unwanted aspects of a particle-physics foundation model through direct edits to its weights. In a foundation model trained on jets, we observe that adding a single soft neutral particle causes large changes in jet classification, violating infrared safety. We locate the responsible attention heads with mechanistic analysis and trace the flaw's origin to pretraining on datasets with different reconstruction thresholds. We then propose two targeted approaches to repair the flaw: by removing three latent directions identified with sparse autoencoders, and by letting language-model agents edit the weights directly, without gradients and with only 50 queries to the model. Both their edits individually remove almost the entire flaw and leave the rest of the model's response nearly unchanged. Compared to fine-tuning, both methods change far fewer p
    
[^114]: 通过谱残差校正加速Muon优化器

    Accelerating Muon with Spectral Residual Correction

    [https://arxiv.org/abs/2611.27077](https://arxiv.org/abs/2611.27077)

    提出AIMuon，通过引入经过矩阵符号变换的梯度-动量残差来校正Muon的谱更新步，在保留Muon动量递归的同时实现优化加速，并给出了随机平稳性理论保证，在多个语言模型上验证了其以更少步数达到Muon最终验证损失的效果。

    

    Muon通过将矩阵符号变换应用于动量来构建矩阵更新。加速Muon的常见途径是在该变换之前修改梯度或动量。我们探索了一个互补的方向：在保留Muon动量递归的同时精炼参数更新步。AIMuon源自ADMM启发动量框架的参数子问题，利用梯度-动量残差来校正初步的谱更新步。残差经过独立的矩阵符号变换，而谱-核对偶性决定了动量方向与残差方向的相对权重。在实践中，两个方向均通过Newton-Schulz迭代进行近似，其组合被重新缩放以匹配Muon的更新幅度。我们为相应的精确迭代建立了随机平稳性界。在GPT-2-small、GPT-2-medium和Qwen3-0.6B上，AIMuon以……达到了Muon的最终验证损失（摘要在此处被截断）。

    arXiv:2611.27077v1 Announce Type: new  Abstract: Muon constructs matrix updates by applying a matrix-sign transformation to momentum. A common route to accelerating Muon is to modify the gradient or momentum before this transformation. We explore a complementary direction: refining the parameter step while retaining Muon's momentum recursion. Derived from the parameter subproblem of the ADMM-Inspired Momentum framework, AIMuon uses the gradient--momentum residual to correct the tentative spectral step. The residual undergoes a separate matrix-sign transformation, while spectral--nuclear duality determines the relative weighting of the momentum and residual directions. In practice, both directions are approximated with Newton--Schulz iterations and their combination is rescaled to match Muon's update magnitude. We establish a stochastic stationarity bound for the corresponding exact iteration. On GPT-2-small, GPT-2-medium, and Qwen3-0.6B, AIMuon reaches Muon's final validation loss with
    
[^115]: 单机调度中最小化最大延误的在线学习

    Online Learning for Minimizing Maximum Tardiness in Single-Machine Scheduling

    [https://arxiv.org/abs/2611.27067](https://arxiv.org/abs/2611.27067)

    该论文开创性地研究了单机调度中最小化最大延误的在线学习问题，并证明了在随机环境下FTL算法的期望1-遗憾为O(n log n / Δ²)，同时给出相应的高概率界。

    

    我们研究单机调度中最小化最大延误（OMMT）的在线学习问题。在每一轮中，学习器选择n个固定作业的一个排列，代表一个调度方案。随后环境揭示每个作业的加工时间和交货期，学习器承受一个向量值损失，其各分量对应于各个作业的延误。学习器的目标是最小化α-遗憾，即学习器的最大累计延误与事后最优固定排列最大累计延误的α倍之间的差异，且α越小越好。我们在随机环境和对抗环境下研究OMMT。在随机设置中，我们证明了跟随领先者（FTL）算法的期望1-遗憾为O(n log n / Δ²)，并给出了相应的高概率界，其中Δ是依赖于分布的间隔参数。

    arXiv:2611.27067v1 Announce Type: new  Abstract: We study online learning for minimizing maximum tardiness (\OMMT) in single-machine scheduling. In each round, the learner selects a permutation of \(n\) fixed jobs, representing a schedule. The environment then reveals a processing time and a due date for each job, and the learner incurs a vector-valued loss whose components correspond to the tardiness of the individual jobs. The goal of the learner is to minimize \(\alpha\)-regret, the difference between the learner's maximum cumulative tardiness and \(\alpha\) times that of the best fixed permutation in hindsight, for as small \(\alpha\) as possible. We investigate OMMT under both stochastic and adversarial environments. In the stochastic setting, we prove that the expected \(1\)-regret of the Follow-the-Leader (FTL) algorithm is \(   O\!\left(\frac{n\log n}{\Delta^2}\right) \), together with a corresponding high-probability bound, where \(\Delta\) is a distribution-dependent gap para
    
[^116]: COMPASS：用于早期时间序列分类的跨模态观测与多路径自适应停止与选择方法

    COMPASS: Cross-modal Observation with Multi-Path Adaptive Stopping and Selection for Early Time Series Classification

    [https://arxiv.org/abs/2611.27065](https://arxiv.org/abs/2611.27065)

    本文提出两阶段框架COMPASS，通过跨模态特征交换、门控融合和共享四动作控制器，在多模态早期时间序列分类中联合决策停止时机与预测分支选择，在准确率与提前量的权衡上优于现有基线。

    

    早期时间序列分类（ETSC）需要在预测性能与观测数据量之间进行权衡。在多模态场景下，与各模态关联的分支和融合分支在相同观测前缀下的预测能力可能存在差异，因此需要联合决策何时停止观测以及由哪个分支给出最终预测。我们提出了COMPASS，这是一个两阶段框架，保留两个与模态关联的时间序列分支和一个融合分支。第一阶段结合跨模态特征交换、门控融合以及在每个前缀处的分支监督。第二阶段优化一个共享的四动作控制器，用于继续观测或通过某个冻结分支终止并输出预测。在六个生理与可穿戴数据集上的实验中，COMPASS与五个ETSC基线的多模态适配版本进行了比较。在该比较范围内，COMPASS在准确率-提前量调和平均值、AUROC和提前量方面均取得了最佳的平均排名。

    arXiv:2611.27065v1 Announce Type: new  Abstract: Early time-series classification (ETSC) balances predictive performance against the amount of observed data. In multimodal settings, modality-associated and fused branches may differ in predictive ability at the same observed prefix, motivating joint decisions about when to stop and which branch should provide the final prediction. We propose COMPASS, a two-stage framework that preserves two modality-associated temporal branches and one fusion branch. The first stage combines cross-modal feature exchange and gated fusion with branch-wise supervision at every prefix. The second stage optimizes a shared four-action controller to continue observation or terminate through one of the frozen branches. Experiments on six physiological and wearable datasets compare COMPASS with multimodal adaptations of five ETSC baselines. COMPASS achieves the best average ranks in accuracy-earliness harmonic mean, AUROC, and earliness within this comparison se
    
[^117]: 重新思考桥梁点云语义分割中的跨传感器域自适应

    Rethinking Cross-Sensor Domain Adaptation for Bridge Point Cloud Semantic Segmentation

    [https://arxiv.org/abs/2611.27060](https://arxiv.org/abs/2611.27060)

    本文重新审视桥梁点云语义分割中TLS到MLS的跨传感器域自适应协议，系统分析了采样密度、表面粗糙度、颜色和采集视角在稀疏体素表示中的影响机制，并评估了源模型收敛程度及多种域自适应方法在不同训练周期下的增益效果。

    

    桥梁点云分割模型在使用地面激光扫描（TLS）数据训练后应用于移动激光扫描（MLS）数据时会损失精度。本研究重新审视了SemanticBridge的TLS到MLS协议，以确定采样密度、表面粗糙度、颜色和采集视角在8厘米稀疏体素表示中的表现方式，以及自适应增益如何依赖于源模型的收敛程度。基于一个稀疏卷积U-Net，我们将原始点统计数据与受控输入扰动相结合，并在100个epoch和300个epoch的训练计划下比较了自适应批归一化、熵最小化、Mean Teacher和置信度过滤自训练方法。TLS和MLS在预处理后具有相似的最近邻间距（分别为12.08和12.02毫米），对TLS点施加4毫米扰动仅使体素占据率变化4.4%。颜色分布存在系统性差异，但在相同体素上将颜色置零会使目标测量……（摘要原文在此处被截断）

    arXiv:2611.27060v1 Announce Type: cross  Abstract: Segmentation models for bridge point clouds lose accuracy when trained on terrestrial laser scanning (TLS) data and applied to mobile laser scanning (MLS) data. This study re-examines the SemanticBridge TLS-to-MLS protocol to determine how sampling density, surface roughness, color, and acquisition viewpoint are expressed in the 8-cm sparse-voxel representation, and how adaptation gains depend on source-model convergence. Using one sparse-convolution U-Net, we combine raw-point statistics with controlled input perturbations and compare adaptive batch normalization, entropy minimization, Mean Teacher, and confidence-filtered self-training under 100- and 300-epoch schedules. TLS and MLS have similar nearest-neighbor spacing after preprocessing (12.08 and 12.02 mm), and a 4-mm perturbation of TLS points changes voxel occupancy by only 4.4%. Color distributions differ systematically, but zeroing color on identical voxels changes target mea
    
[^118]: 结构一致性可指示蛋白质-配体共折叠预测的准确性

    Structural Agreement Indicates Prediction Accuracy For Protein-Ligand Co-Folding

    [https://arxiv.org/abs/2611.27058](https://arxiv.org/abs/2611.27058)

    提出CORE分数（重复集合浓度分数），通过综合多次预测间的聚类紧密度与集合支持度来衡量结构一致性，比现有置信度分数能更准确地判断蛋白质-配体共折叠预测的正确性。

    

    共折叠模型能够预测蛋白质及其结合配体的结构，但其准确性因靶标不同而存在差异。因此，这些模型的实用性取决于用户能否判断哪些预测是可信的。置信度分数本应用于指导这一选择。然而，我们在七个共折叠模型上的研究表明，基于置信度的选择相比随机挑选预测几乎没有改进。对于大约四分之三的结构，预测要么全部正确、要么全部错误，这使得排序无法改变结果。此外，高置信度并不能可靠地区分排名第一的预测是正确还是错误。我们发现，多个预测之间的结构一致性是比现有置信度分数更能准确衡量预测成功可能性的指标。我们用重复集合浓度分数来刻画这一点，该分数综合了聚类紧密度与集合支持度。

    arXiv:2611.27058v1 Announce Type: new  Abstract: Co-folding models predict the structures of proteins and their bound ligands, but their accuracy varies across targets. Their usefulness therefore depends on knowing which predictions a user can trust. Confidence scores are intended to guide that choice. However, across seven co-folding models, we show that confidence-based selection offers little improvement over choosing a prediction at random. For approximately three-quarters of structures, predictions are either exclusively correct or incorrect, leaving no opportunity for ranking to change the outcome. Moreover, high confidence does not reliably discriminate correct top-ranked predictions from incorrect ones. We find that structural agreement among predictions is a more accurate measure of likely success than current confidence scores. We capture this with the Concentration of Repeated Ensembles (CORE) Score, which combines cluster tightness with ensemble support. CORE Score discrimi
    
[^119]: 感知量化训练的连续松弛方法

    A Continuous Relaxation of Quantization-Aware Training

    [https://arxiv.org/abs/2611.27056](https://arxiv.org/abs/2611.27056)

    提出RelaxQAT，通过在训练过程中逐渐连续地改变位宽来松弛感知量化训练，从而避免低位量化中损失曲面的鞍点和平坦区域问题，显著提升量化模型性能。

    

    量化是大生成模型时代部署流程中的关键环节。低位量化，特别是2比特或低于2比特的量化，是一个众所周知的难题，尤其是在使用逐通道量化方案时——该方案已被证明是硬件效率方面的领先候选方案。低位量化的主要挑战包括量化级别有限以及损失曲面中存在的鞍点或平坦区域。在本文中，我们通过动态变化位宽的视角来审视量化损失曲面平坦化的问题。与现有技术要么突然改变位精度、要么保持位精度固定不变不同，我们提出在模型训练过程中逐渐改变位宽。为此，我们提出了RelaxQAT，一种QAT（感知量化训练）过程的连续松弛方法。我们还提供了简单的理论和实证见解，解释了为什么RelaxQAT能够提升QAT性能。我们广泛的实验证明……

    arXiv:2611.27056v1 Announce Type: new  Abstract: Quantization is a critical piece of the deployment pipeline in the era of large generative models. Low-bit quantization, particularly 2-bits or sub-2-bits, is a notoriously difficult problem, particularly when using a channelwise quantization scheme which has been shown as a leading candidate for hardware-efficiency. The key challenges with low-bit quantization range from limited quantization levels to saddle points or flat regions in the loss surface. In this paper, we look at the problem of flat quantization loss surfaces through a lens of dynamically changing bitwidth. Unlike existing techniques which either change the bit-precision abruptly or keep it fixed, we propose to gradually alter the bitwidth as the model trains. To this end, we propose RelaxQAT, a continuous relaxation of the QAT process. We also provide simple theoretical and empirical insights behind why RelaxQAT helps the QAT performance. Our extensive experiments demonst
    
[^120]: 动作条件下的时空预测

    Action-Conditioned Spatio-Temporal Forecasting

    [https://arxiv.org/abs/2611.27044](https://arxiv.org/abs/2611.27044)

    该论文提出了PairST基准（涵盖交通、电力传输和供水分配的六个数据集，提供同一系统状态下有无干预动作的成对未来数据）以及ActionST方法，通过在与干预无关的参考预测上写入显式动作响应，实现对受影响区域动作效果与未受影响区域不变性的分别评估。

    

    我们研究在指定动作条件下网络化系统中的时空预测问题。其挑战在于如何捕捉动作产生的影响，同时避免在未受影响区域预测出虚假变化。然而，对这些影响的直接监督和评估需要有动作与无动作两种情形下成对的未来状态，而观测轨迹只能揭示实际发生的未来。我们提出了PairST，一个包含六个数据集的基准，涵盖交通、电力传输和供水分配领域，其中参考未来与干预未来是从同一系统状态模拟得到的成对数据。这些成对数据使得可以分别评估受影响区域的预测精度和未受影响区域的响应误差。我们进一步提出ActionST，该方法通过在与干预无关的参考预测上写入显式的动作响应来预测干预后的未来。具体而言，节点-时间寻址为每个响应位置构建查询，状态条件化的响应写入器……

    arXiv:2611.27044v1 Announce Type: new  Abstract: We study spatio-temporal forecasting in networked systems under specified actions. The challenge is to capture the action's effects without predicting spurious changes in unaffected regions. However, direct supervision and evaluation of these effects require paired futures with and without the action, whereas observational trajectories reveal only the realized future. We introduce PairST, a benchmark comprising six datasets across traffic, power transmission and water distribution, with paired reference and intervened futures simulated from the same system state. These pairs enable separate evaluation of affected-region forecast accuracy and quiet-region response error. We further propose ActionST, which forecasts the intervened future by writing an explicit response onto an intervention-independent reference forecast. Specifically, node-time addressing constructs a query for each response position, a state-conditioned response writer co
    
[^121]: 超图结构熵增强的层次感知半监督图分类

    Hypergraph Structural Entropy-Enhanced Hierarchy-Aware Semi-Supervised Graph Classification

    [https://arxiv.org/abs/2611.27043](https://arxiv.org/abs/2611.27043)

    提出HypSEE框架，通过构建游走诱导与嵌入衍生超图捕获层次感知的局部和长程高阶关系，并利用超图结构熵引导的层次超图池化方法实现半监督图分类中的层次表示学习。

    

    半监督图分类旨在利用大量无标签图与少量有标签图来学习图级表示，从而有效缓解标签稀缺问题。尽管各种半监督学习策略已取得显著进展，但图中顶点之间超越成对边的复杂关系，尤其是层次结构，仍未被现有方法充分探索。在这项工作中，我们提出了一种超图结构熵增强的层次感知框架——HypSEE，用于半监督图分类。具体而言，我们首先构建游走诱导超图和嵌入衍生超图，以捕获图内层次感知的局部和长程高阶关系。然后，我们设计了一种由超图结构熵（HSE）引导的层次超图池化方法，以促进超图中的层次表示学习。最后，我们采用（原文摘要在此处截断）

    arXiv:2611.27043v1 Announce Type: cross  Abstract: Semi-supervised graph classification aims to learn graph-level representations by leveraging a large number of unlabeled graphs alongside a small set of labeled ones, effectively mitigating the issue of label scarcity. Despite significant progress by various semi-supervised learning strategies, complex relationships among graph vertices beyond pairwise edges, especially hierarchical structures, remain underexplored by existing methods. In this work, we propose a Hypergraph Structural Entropy-Enhanced hierarchy-aware framework, namely HypSEE, for semi-supervised graph classification. Specifically, we first construct walk-induced and embedding-derived hypergraphs to capture hierarchy-aware local and long-range higher-order relationships within graphs. Then, we devise a hierarchical hypergraph pooling approach guided by hypergraph structural entropy (HSE) to facilitate hierarchical representation learning in hypergraphs. Lastly, we employ
    
[^122]: 心灵世界模型：运动脑电表征与动作条件世界动力学的联合学习

    Mind World Models: Joint Learning of Motor EEG Representations and Action-Conditioned World Dynamics

    [https://arxiv.org/abs/2611.27029](https://arxiv.org/abs/2611.27029)

    提出心灵世界模型（MWM），将脑电视为世界模型的潜在动作，从无动作标注的自然情景脑电与第一人称视频中学习运动脑电表征，并引入侧化感知Transformer（LAT）分离双侧共享与单侧特异特征，实现以运动为锚点的脑电预训练。

    

    来自脑电图（EEG）的运动解码为无创脑机接口提供了支持，然而现有方法要么依赖成本高昂的基于提示的试验范式和粗粒度的类别标签，要么采用无法直接刻画运动如何展开的预训练目标，限制了其学习可迁移运动表征的能力。为此，我们提出了心灵世界模型（Mind World Model, MWM），这是一个无需动作标注、能够从可扩展的自然情景脑电数据和第一人称视角视频中学习运动脑电表征的框架。MWM将截至当前帧所观测到的脑电视为世界模型的潜在动作：由一个心灵模型对脑电进行编码，再由一个世界模型利用该编码预测未来帧的表征，从而形成一个以运动为锚点的预训练目标。我们进一步引入了侧化感知Transformer（Lateralization-Aware Transformer, LAT），它利用帧平均来分离双侧共享特征与特定侧别的侧向特征，从而编码半球之间的对应关系……

    arXiv:2611.27029v1 Announce Type: new  Abstract: Motor decoding from electroencephalography (EEG) supports non-invasive brain-computer interfaces, yet existing methods either rely on costly cue-based trials and coarse categorical labels or use pretraining objectives that do not directly capture how movement unfolds, limiting their ability to learn transferable motor representations. To this end, we propose the Mind World Model (MWM), a framework that learns motor EEG representations from scalable naturalistic EEG and egocentric video without action annotations. MWM treats EEG observed up to the current frame as the latent action of a world model: a mind model encodes the EEG, and a world model uses it to predict the representation of a future frame, yielding a movement-grounded pretraining objective. We further introduce the Lateralization-Aware Transformer (LAT), which uses frame averaging to separate shared bilateral from side-specific lateral features, encoding hemispheric correspon
    
[^123]: 迈向ICU患者的数字孪生：生成式事件模型中的病情预测与治疗反应

    Toward a digital twin of the ICU patient: Forecasting and treatment response in a generative event model

    [https://arxiv.org/abs/2611.27028](https://arxiv.org/abs/2611.27028)

    研究者训练了7900万参数的生成式事件模型TWICE，无需任务特定训练即可通过模拟50条患者未来轨迹，在19项ICU临床结局预测上达到与144个监督模型最佳配置相当的性能，是实现ICU患者数字孪生的重要一步。

    

    重症监护病房（ICU）患者的数字孪生应能预测患者的病情走向以及某项治疗将带来的改变。生成式事件模型——即经过训练以延续患者电子健康记录（EHR）的模型——被视为实现此类数字孪生的一条途径。我们在一个学术医疗系统的ICU记录上训练了TWICE，一个7900万参数的仅解码器（decoder-only）transformer模型（训练数据包含127,332名患者、15.7亿个token）。我们在临床预测任务、生成保真度、个体轨迹和反事实验证等方面对其进行了评估。在没有任何任务特定训练的情况下，通过从50个模拟的未来轨迹中读取19项临床结果，TWICE在留出患者上的表现与144个监督学习配置中的最佳者相当（平均AUROC 0.836 对比 0.834）。它在一天内会改变诊疗决策的事件预测上领先，而在死亡预测上表现稍逊。在训练数据固定的情况下，生成更多且更长的模拟未来，其收益超过了将模型规模扩大6.8倍。其生成的记录与真实记录相匹配……

    arXiv:2611.27028v1 Announce Type: new  Abstract: A digital twin of a patient in intensive care would forecast the patient's course and what a treatment would change. Generative event models, trained to continue a patient's electronic health record (EHR), have been proposed as a path toward such twins. We trained TWICE, a 79M-parameter decoder-only transformer, on the ICU records of one academic health system (127,332 training patients and 1.57 billion tokens). We evaluated it on clinical prediction tasks, generation fidelity, individual trajectories and counterfactual validation. Reading 19 clinical outcomes off 50 simulated futures, with no task-specific training, TWICE matched the best of 144 supervised configurations on held-out patients (mean AUROC 0.836 vs. 0.834). It led on events that change care within a day and trailed on death. With the training data fixed, more and longer simulated futures were worth more than a 6.8-fold larger model. Its generated records matched real ones 
    
[^124]: 双隐藏层ReLU最大值函数的整数乘子与算术障碍

    Integral Multipliers and Arithmetic Obstructions for Two-Hidden-Layer ReLU Maxima

    [https://arxiv.org/abs/2611.27018](https://arxiv.org/abs/2611.27018)

    本文借助代数拓扑工具（整数立方填充与同调障碍类）证明了双隐藏层整数参数ReLU网络精确表示最大值函数的最小正输出乘子——六输入为2、七输入为2或6，并刻画了七输入情形中剩余的算术障碍。

    

    我们研究用具有两个隐藏层且参数为整数的ReLU网络对最大值函数的精确表示。对于六输入最大值函数，最小正输出乘子为2；对于七输入最大值函数，最小正输出乘子为2或6。我们通过整数立方填充以及从多面体恒等式到网络的保系数转换获得这些界。对于六单纯形的坐标面复形，我们计算出其整数上的锥化指数为6，而在将2取逆后为3，从而允许每个面内任意的整数方向。无限制核复形同调中的一个类刻画了剩余的七输入问题；该类的消失等价于对固定目标实现整数填充。我们还构造了一个有限判别准则，在一个包含两个已知构造以及所有边不交三边森林对的字典中，将有理可表示性与三局可表示性区分开来。此障碍（摘要原文在此处截断）

    arXiv:2611.27018v1 Announce Type: new  Abstract: We study exact representations of maximum functions by ReLU networks with two hidden layers and integer parameters. The least positive output multiplier is two for the six-input maximum, and either two or six for the seven-input maximum. We obtain these bounds from integral cubical fillings and a coefficient-preserving passage from polytope identities to networks. For the coordinate-facet complex of the six-simplex, we compute a coning index of six over the integers and three after inverting two, allowing arbitrary integral directions within each facet. A class in the homology of the unrestricted kernel complex characterizes the remaining seven-input problem; its vanishing is equivalent to an integral filling of a fixed target. We also construct a finite character that separates rational from three-local representability in a dictionary containing two known constructions and all pairs of edge-disjoint three-edge forests. This obstruction
    
[^125]: Ruby：利用学习到的结构表示在不同药物发现阶段实现最先进的药物效力预测

    Ruby: State-of-the-Art Potency Prediction across Discovery Regimes Using Learned Structural Representations

    [https://arxiv.org/abs/2611.27015](https://arxiv.org/abs/2611.27015)

    Ruby是一个利用结构基础模型Pearl的中间表示、无需显式结合姿态的鲁棒通用结合亲和力预测器，通过专有数据与公开数据集GenesisPotency-2026训练，在药物发现的各个阶段均实现了最先进的药效预测。

    

    小分子药物发现通过反复的设计-合成-测试-分析循环进行，其目标通常是在保持其他性质处于可行治疗候选药物所需范围内的同时，最大化药物效力。随着药物发现从苗头化合物识别推进到先导化合物转化和先导化合物优化阶段，化合物优先级排序对计算方法提出了不同的要求，计算预算不断变化，靶向实验数据量也在不断增加。我们介绍了Ruby，一个鲁棒的通用结合亲和力预测器，它利用结构基础模型Pearl的中间表示，而无需为每个化合物提供显式的结合姿态。Ruby在专有数据以及GenesisPotency-2026上进行训练，后者是一个经过审计的公开来源药效训练数据集，以CC BY-SA 4.0许可证发布供学术和商业使用，并带有行级来源追溯和审计注释。Ruby实现了最高的……

    arXiv:2611.27015v1 Announce Type: cross  Abstract: Small-molecule drug discovery proceeds through repeated design-make-test-analyze cycles. The goal is often to maximize potency while keeping other properties within the ranges required for a viable therapeutic candidate. Compound prioritization places different demands on computational methods as drug discovery progresses from hit identification through hit-to-lead and lead optimization, with changing computational budgets and increasing amounts of on-target experimental data. We introduce Ruby, a Robust Universal Binding affinitY predictor that uses intermediate representations from the structural foundation model Pearl without requiring an explicit binding pose for every compound. Ruby is trained on proprietary data plus GenesisPotency-2026, an audited public-source potency training dataset released for academic and commercial use under a CC BY-SA 4.0 license with row-level provenance and audit annotations. Ruby achieves the highest 
    
[^126]: Loop-OPF：面向最优潮流的求解器引导学习与推理时拓扑自适应性

    Loop-OPF: Solver-Informed Learning for Optimal Power Flow with Inference-Time Topology Adaptivity

    [https://arxiv.org/abs/2611.27013](https://arxiv.org/abs/2611.27013)

    Loop-OPF是一种由PDHG迭代构建的求解器引导神经网络，在凸性条件下保证DC-OPF预测的最优性与可行性，并通过自监督方式学习PDHG参数实现无需解标签的加速训练与对推理时未见拓扑的自适应。

    

    近年来，基于学习的最优潮流（OPF）方法被提倡用于加速OPF求解，即通过预训练神经网络（NN）直接根据给定输入（如负荷需求）进行预测。然而，现有的OPF学习方法在很大程度上无法保证其预测的最优性与可行性，并且在推理时难以适应未见过的拓扑场景，例如由线路或发电机停运引起的场景。本工作提出了Loop-OPF，这是一种求解器引导的神经网络，其网络层由原始-对偶混合梯度（PDHG）方法的迭代构建而成。在DC-OPF的标准凸性与正则性条件下，Loop-OPF所得的不动点输出因此是最优且可行的。为了在各种OPF场景下加速PDHG求解，Loop-OPF以自监督的方式学习某些PDHG参数，而无需任何OPF解标签。我们采用深度平衡（DEQ）框架进行训练，从……

    arXiv:2611.27013v1 Announce Type: cross  Abstract: Learning for optimal power flow (OPF) has recently been advocated to accelerate OPF solutions by using a pre-trained neural network (NN) to predict directly from given inputs such as the load demand. However, existing learning-for-OPF methods largely fall short of ensuring the optimality and feasibility of their predictions. They also struggle to adapt at inference time to unseen topology scenarios, such as those caused by line or generator outages. This work puts forth Loop-OPF, a solver-informed NN that builds its layers from the iterations of the primal-dual hybrid gradient (PDHG) method. Under standard convexity and regularity conditions on DC-OPF, the resultant fixed-point output of Loop-OPF is thus optimal and feasible. To speed up PDHG across diverse OPF scenarios, Loop-OPF learns certain PDHG parameters in a self-supervised manner, without any OPF solution labels. We adopt the deep equilibrium (DEQ) framework for training so th
    
[^127]: 公平房产税：减少机器学习估值中的累退性

    Fair Property Tax: Reducing Regressivity in Machine-Learning Valuation

    [https://arxiv.org/abs/2611.27009](https://arxiv.org/abs/2611.27009)

    与库克县评估官办公室合作开发的两种训练方法（直接惩罚法与代理加权损失法）能在保持预测精度的同时显著减少机器学习房产估值中的累退性，其中直接法已实际应用于公共住宅估值软件。

    

    先进的机器学习模型正日益被用于税务评估中的房产估值。尽管这些模型能够达到很高的预测精度，但它们可能会系统性地给价格较低的房产分配更高的估值-销售价格比率，这种模式被称为“累退性”。与库克县评估官办公室（CCAO）合作，我们开发并评估了两种训练方法，它们在保持价格尺度预测精度的同时，大幅减少了这种累退模式。其中“直接法”已被纳入CCAO的公共住宅估值软件中。在保留CCAO所使用的LightGBM模型类的基础上，我们提出了对对数估值误差与对数销售价格之间平方协方差施加惩罚的“直接法”，以及一种产生加权平方误差损失的观测加法“代理法”。我们在CCAO工作流的实现中对这两种方法进行了评估，使用了七个按时间顺序排列的验证折、一个留出测试集以及……

    arXiv:2611.27009v1 Announce Type: new  Abstract: Advanced machine-learning models are increasingly used for property valuation in tax assessment. Although these models can achieve high predictive accuracy, they may systematically assign higher valuation-to-sale ratios to lower-priced properties, a pattern known as regressivity. In collaboration with the Cook County Assessor's Office (CCAO), we develop and evaluate two training methods that substantially reduce this regressive pattern while maintaining price-scale predictive accuracy. The Direct method has been incorporated into CCAO's public residential valuation software. Retaining the LightGBM model class used at CCAO, we propose a Direct penalty on squared covariance between log valuation errors and log sale prices, and an observation-additive Surrogate that yields a weighted squared-error loss. We evaluate both methods in our implementation of the CCAO workflow, using seven chronological validation folds, a held-out test set, and a
    
[^128]: JevRisk：基于语言决策模型的零样本金融风险评估

    JevRisk: Zero-Shot Financial Risk Assessment with a Language Decision Model

    [https://arxiv.org/abs/2611.26999](https://arxiv.org/abs/2611.26999)

    JevRisk提出了一个基于语言决策模型的零样本金融风险评估系统，通过遵循金融监管指南用代码计算精确风险指标，摆脱了对人工标签的依赖，并能在不同客户和风险类型间泛化。

    

    银行必须结合客户画像来解读金融活动，从而监控每位客户的金融风险，例如洗钱和信用违约。现有的银行风险监控系统大多基于专家设计的风险指标，这些指标对每位客户应用预定义的阈值，无法适应客户的个人情况。监督式模型可以从有标签的客户中识别风险模式，但每个标签通常需要专家审查或数月的还款历史，既缓慢又成本高昂。因此，金融风险评估需要一个零样本决策系统，以最大限度地减少对标签的依赖，并在不同客户和风险类型之间实现泛化。为此，我们提出了JevRisk，一个基于语言决策模型的零样本金融风险评估系统。JevRisk分两步工作：首先，遵循金融领域的监管指南，JevRisk用代码计算精确的风险指标……

    arXiv:2611.26999v1 Announce Type: cross  Abstract: Banks must monitor every customer's financial risk, such as money laundering and credit default, by interpreting financial activity in the context of the customer's profile. Existing risk monitoring systems of banks are mostly based on expert-designed risk indicators, which apply predefined thresholds to every customer and cannot adapt to their personal situations. Supervised models recognise risk patterns from labelled customers, but each label often needs an expert's review or months of repayment history, which is slow and costly. Financial risk assessment therefore calls for a zero-shot decision system that minimises the need for labels and generalises across customers and risk types. To this end, we present JevRisk, a system for zero-shot financial risk assessment with a language decision model. JevRisk works in two steps. First, following regulatory guidelines in the finance domain, JevRisk computes exact risk indicators with code
    
[^129]: 分布代数几何持续性（dAGP）：Wasserstein 稳定的多项式关系签名

    Distributional Algebro-Geometric Persistence (dAGP): Wasserstein-Stable Polynomial-Relation Signatures

    [https://arxiv.org/abs/2611.26986](https://arxiv.org/abs/2611.26986)

    本文提出分布代数几何持续性（dAGP），通过统计概率测度矩矩阵中近似多项式关系方向在多项式次数和容差两个参数下的数量，构建了一个刻画分布代数结构且具有 Wasserstein 稳定性的双参数整数签名。

    

    圆与四次超椭圆具有相同的同调，但它们的定义方程的次数分别为二次和四次。我们针对点云分布研究这种代数上的差异。对于定义在 $\Omega=[-1,1]^d$ 上的概率测度 $\mu$ 以及次数为 $m$ 的正交归一多项式特征映射 $\phi_m$，矩矩阵 $G_m(\mu)=\mathbb{E}_{X\sim\mu}[\phi_m(X)\phi_m(X)^\top]$ 满足 $c^\top G_m(\mu)c=\mathbb{E}_\mu[f_c(X)^2]$，因此与矩矩阵小特征值相对应的系数方向，代表在测度 $\mu$ 上近似为零的低次多项式。在不同次数 $m$ 和容差 $\varepsilon$ 下统计这些“近似关系”方向的数量，可以得到一个双参数整数签名。我们将这种表示称为分布代数几何持续性（dAGP），它是点云代数几何持续性（AGP）构造在总体（population）层面上的对应物。对于经验测度，dAGP 是按样本归一化的……（原文摘要在此处截断）

    arXiv:2611.26986v1 Announce Type: new  Abstract: A circle and a quartic superellipse have the same homology, but their defining equations have degrees two and four, respectively. We study this algebraic distinction for point-cloud distributions. For a probability measure $\mu$ on $\Omega=[-1,1]^d$ and an orthonormal polynomial feature map $\phi_m$ of degree $m$, the moment matrix $G_m(\mu)=\mathbb{E}_{X\sim\mu}[\phi_m(X)\phi_m(X)^\top]$ satisfies $c^\top G_m(\mu)c=\mathbb{E}_\mu[f_c(X)^2]$, so coefficient directions associated with its small eigenvalues represent low-degree polynomials that nearly vanish on $\mu$. Counting these near-relation directions across degree $m$ and the tolerance $\varepsilon$ gives a two-parameter integer signature. We call this representation distributional algebro-geometric persistence (dAGP), the population counterpart of the point-cloud algebro-geometric persistence (AGP) construction. For empirical measures, dAGP is the sample-normalised counterpart of p
    
[^130]: 面向紧凑且可扩展神经网络验证的仿射图像传播方法

    Affine-Image Propagation for Tight and Scalable Neural Network Verification

    [https://arxiv.org/abs/2611.26978](https://arxiv.org/abs/2611.26978)

    本文提出了一种新颖的边界传播方法 affine-image SDP-CROWN，通过仿射图像约束更精确地近似预激活域并联合优化线性边界，从而实现对 $\ell_{2}$ 对抗攻击紧凑且可扩展的神经网络验证。

    

    尽管边界传播方法在认证神经网络抵御 $\ell_{\infty}$ 对抗攻击方面非常有效，但针对 $\ell_{2}$ 对抗攻击的紧凑且可扩展的验证仍然是一个重大的开放性挑战。现有方法通过用逐元素区间或轴对齐椭球来包络预激活集合而产生几何损失，并通过限制在这些域上构造的线性边界而产生函数损失。因此，紧凑的 $\ell_{2}$ 验证目前仅限于这些近似较为精确的经 Lipschitz 正则化的网络。在本文中，我们提出了 affine-image SDP-CROWN，这是一种新颖的边界传播方法，它使用仿射图像约束更好地近似预激活域，并联合优化线性边界以适应所得到的域。为了绕过仿射图像表示所需的稠密半定约束所带来的高昂计算成本，我们将其替换为……（摘要在此处截断）

    arXiv:2611.26978v1 Announce Type: new  Abstract: While bound propagation methods are highly effective for certifying neural networks against $\ell_{\infty}$ adversaries, tight and scalable verification for $\ell_{2}$ adversaries remains a significant open challenge. Existing methods incur geometric loss by enclosing preactivation sets in elementwise intervals or axis-aligned ellipsoids, and functional loss by restricting the linear bounds constructed over those domains. Consequently, tight $\ell_{2}$ verification is currently restricted to Lipschitz-regularized networks where these approximations are accurate. In this paper, we introduce affine-image SDP-CROWN, a novel bound propagation method that better approximates preactivation domains using affine-image constraints and jointly optimizes the linear bound to adapt to the resulting domain. To bypass the prohibitive computational cost of dense semidefinite constraints required by the affine-image representation, we replace them with s
    
[^131]: 运行一次，得到两个：本地LLM的近乎零成本的流式安全检查

    Run One, Get Two: Near-Free Streaming Safety Checks for Local LLMs

    [https://arxiv.org/abs/2611.26975](https://arxiv.org/abs/2611.26975)

    ToRA通过让第二个模型作为额外一行嵌入生成器的解码步骤并共享权重与KV缓存，以每个token不到4%的额外延迟实现了接近专用防护模型准确性的本地LLM流式安全检查。

    

    本地LLM助手一次只为一个用户解码，因此解码受内存带宽限制，计算资源处于闲置状态。如今的流式安全检查需要在准确性与成本之间权衡：专用的防护模型虽然准确，但几乎使每个token的延迟翻倍；而基于隐藏状态的探针虽然廉价，但准确性较低。我们提出ToRA，这是一种适配器，它在生成器的解码步骤中将第二个模型作为额外的一行来运行，共享其权重和KV缓存。双行内核只需为两行各读取一次每个权重。在Qwen3-0.6B上，ToRA匹配或超越了专用的流式防护模型，在17个基准测试中的12个上取得了更高的F1分数。它在相似成本下比探针多检测出11.8个百分点的不安全响应，且每个token仅增加不到4%的延迟。

    arXiv:2611.26975v1 Announce Type: new  Abstract: Local LLM assistants decode for one user at a time, so decoding is bound by memory bandwidth and leaves compute idle. Streaming safety checks today trade accuracy against cost. A dedicated guard model is accurate but nearly doubles the latency per token, while a hidden-state probe is cheap but less accurate. We present ToRA, an adapter that runs a second model as an extra row in the generator's decode step, sharing its weights and KV cache. Dual-row kernels read each weight once for both rows. On Qwen3-0.6B, ToRA matches or beats a dedicated streaming guard, with higher F1 on 12 of 17 benchmarks. It detects 11.8 points more unsafe responses than the probe at a similar cost, under 4% added latency per token.
    
[^132]: 基于形变解压缩的谱估计

    Spectral Estimation with Deformed Decompression

    [https://arxiv.org/abs/2611.26945](https://arxiv.org/abs/2611.26945)

    该论文提出了一种名为“形变解压缩”的谱估计新方法，旨在克服谱求逆的不稳定性以及自由解压缩仅适用于酉不变系综的局限，从而能够仅凭小规模矩阵的信息推断大规模样本协方差矩阵的特征谱。

    

    样本协方差矩阵是机器学习和统计学中的基本对象，其特征值中编码了有价值的信息。在处理现代问题时，这些矩阵的规模可能会大到使可行计算变得不切实际。这就产生了对“谱解压缩”的关键需求：一种仅利用较小规模实现的信息来推断超大规模模型特征谱的方法。现有方法主要分为两类：一类依赖于谱求逆，另一类依赖于自由解压缩技术。一方面，谱求逆从根本上是一个病态问题，以不稳定性著称，在有限尺寸的情况下往往无法给出可靠的结果；另一方面，尽管最近提出的自由解压缩方法解决了这一规模问题，但其对Nica-Speicher框架的依赖将其应用限制在酉不变系综中，这是一种对称性条件……

    arXiv:2611.26945v1 Announce Type: cross  Abstract: Sample covariance matrices are fundamental objects in machine learning and statistics, with valuable information encoded into their eigenvalues. When treating modern problems, the scale of these matrices can become prohibitive for tractable computation. This creates a critical need for "spectral decompression": a method for inferring the eigenspectrum of a very large-scale model using only information from its smaller realizations. Existing methods fall into two main categories: those relying on spectral inversion; and those that rely on Free Decompression techniques. On the one hand, spectral inversion is a fundamentally ill-posed problem that is notoriously unstable, often failing to yield reliable results in finite-size settings. On the other hand, while Free Decompression was recently proposed to address this scaling, its reliance on the Nica-Speicher framework restricts its use to unitarily invariant ensembles, a symmetry conditio
    
[^133]: CENTRY：一个用于持续AI风险估计、监测与调控的概率潜在状态模型

    CENTRY: A Probabilistic Latent-State Model for Continual AI Risk Estimation, Monitoring, and Regulation

    [https://arxiv.org/abs/2611.26940](https://arxiv.org/abs/2611.26940)

    本文提出CENTRY概率框架，通过结构化潜变量模型将AI风险分解为一个捕捉AI-环境动态的快速交互子空间和三个编码数据、硬件与模型长期风险状态的慢速潜在子空间，从而实现对已部署AI系统全生命周期的持续风险估计、监测与调控。

    

    人工智能（AI）系统正越来越多地被部署于安全关键型应用中，然而现有的AI安全框架主要侧重于对AI风险进行定义、分类或治理，而非提出可在AI系统整个运行生命周期内持续估计、监测和调控风险的可实施模型与方法。本文提出了CENTRY，一个概率化框架，能够基于已部署AI系统与其环境的交互，实现对风险的持续估计、监测与调控。该框架将AI风险建模为源自三个内在来源：数据集、硬件资源以及模型/算法。文中构建了一个结构化潜变量模型，其中潜在表示被分解为一个捕捉不断演变的AI-环境动态的快速交互子空间，以及三个分别编码与数据集、硬件和模型相关的长期风险状态的慢速潜在子空间……

    arXiv:2611.26940v1 Announce Type: cross  Abstract: Artificial Intelligence (AI) systems are increasingly deployed in safety-critical applications, yet existing AI safety frameworks primarily define, categorize, or govern AI risk rather than proposing implementable models and methods for continually estimating, monitoring, and regulating risk throughout an AI system's operational lifetime. This paper proposes CENTRY, a probabilistic framework that enables continual risk estimation, monitoring, and regulation of a deployed AI system from its interactions with its environment. The framework models AI risk as arising from three intrinsic sources: datasets, hardware resources, and models/algorithms. A structured latent-variable model is developed in which the latent representation is decomposed into a fast interaction subspace that captures the evolving AI--environment dynamics and three slow latent subspaces that encode the long-term risk states associated with the dataset, hardware, and m
    
[^134]: SecretAgent：一个针对视觉语言模型的隐私保护多智能体框架

    SecretAgent: A Multi-Agent Framework for Privacy Protection against Vision Language Models

    [https://arxiv.org/abs/2611.26936](https://arxiv.org/abs/2611.26936)

    提出了首个通用的多智能体框架SecretAgent，通过LLM智能体迭代定位泄露隐私的视觉证据并自适应生成不可见扰动，同时借助外观审查智能体把关，在有效抵御视觉语言模型隐私推断的同时保持图像的视觉保真度。

    

    视觉语言模型（VLMs）能够从共享图像中推断出私人信息，包括位置、个人身份信息（PII）以及诸如年龄和性别等个人属性，甚至可以从医学图像中获取这些信息。对抗性扰动可以保护此类信息免受黑盒VLM的侵害，但现有方法忽略了图像质量，会在其保护的图像中引入可见的伪影。我们提出了SecretAgent，这是首个通用的多智能体框架，能够在保持视觉保真度的同时保护图像隐私免受VLM的侵害。LLM智能体迭代地定位揭示私人信息的视觉证据，并根据目标VLM的回答以及外观审查智能体（该智能体会拒绝任何带有可见伪影的候选方案）来调整不可见的扰动。我们用两个互补的变体实例化了SecretAgent：SecretAgent-Gradient通过六个代理编码器优化扰动以实现跨模型可迁移性……

    arXiv:2611.26936v1 Announce Type: cross  Abstract: Vision-language models (VLMs) can infer private information from shared images, including locations, personally identifiable information (PII), and personal attributes such as age and sex, even from medical images. Adversarial perturbations can protect such information against black-box VLMs, but existing methods overlook image quality and introduce visible artifacts into the images they protect. We present SecretAgent, the first general multi-agent framework that protects image privacy against VLMs while preserving visual fidelity. LLM agents iteratively locate the visual evidence that reveals private information and adapt an invisible perturbation to the target VLM's answers and to an appearance reviewer agent, which rejects any candidate with visible artifacts. We instantiate SecretAgent with two complementary variants: SecretAgent-Gradient optimizes the perturbation through six surrogate encoders for cross-model transferability, an
    
[^135]: 改进粒子混合的优化景观

    Improving the Optimisation Landscape for Particle Mixtures

    [https://arxiv.org/abs/2611.26931](https://arxiv.org/abs/2611.26931)

    该论文证明了混合预测似然在梯度优化中会使匹配不佳的粒子得不到训练信号，并提出将个体风险与混合风险进行凸组合的正则化方法，通过反向KL屏障确保每个粒子都能获得训练信号，从而改进了粒子混合的优化景观。

    

    统计学和机器学习中的若干方法将预测分布表示为模型（即粒子）的等权重混合，并通过混合预测似然进行训练。然而，我们证明混合预测似然并不适合基于梯度的优化，因为它无法识别混合如何在各粒子之间分解，导致匹配不佳的粒子可能得不到训练信号。作为补救措施，我们研究了个体风险（即各粒子的平均对数损失）与混合风险之间的凸组合。在粒子层面，这种正则化风险对粒子的责任施加了反向KL散度屏障，使得任何粒子都不会被剥夺训练信号。这些现象在两个案例研究中得到说明：在第一个案例中，使用平均场朗之万动力学计算面向预测的后验，且数值仅在有条件下表现良好（摘要原文在此处截断）。

    arXiv:2611.26931v1 Announce Type: new  Abstract: Several methods in statistics and machine learning represent a predictive distribution as an equally weighted mixture of models, or particles, trained via the mixture predictive likelihood. However, we prove that the mixture predictive likelihood is poorly suited to gradient-based optimisation, because it does not identify how the mixture is decomposed across particles, so that a poorly matched particle can be starved of training signal. As a remedy, we study a convex combination between the individual risk, i.e. the average log loss of the individual particles, and the mixture risk. At the particle level, this regularised risk imposes a reverse-KL barrier on the responsibilities of the particles, so that no particle can be starved of training signal. These phenomena are illustrated in two case studies: In the first, a predictively-oriented posterior is computed using mean-field Langevin dynamics, and the numerics are well-behaved only w
    
[^136]: 使用预训练编码器与匹配探测方法后，文献中报告的CHB-MIT大幅适应增益消失

    Large Reported CHB-MIT Adaptation Gains Disappear with Pretrained Encoders and Matched Probes

    [https://arxiv.org/abs/2611.26929](https://arxiv.org/abs/2611.26929)

    本文通过复现发现，Dani和Liebe声称的SSL微调在CHB-MIT癫痫检测上带来的36.2和29.5点ROC-AUC提升，实际上是配置错误（预训练权重未被正确加载、探测时dropout未关闭）造成的假象——在正确设置下适应增益几乎完全消失。

    

    Dani 和 Liebe（2026）（简称DL）报告称，通过自监督学习（SSL）对BIOT和CBraMod的最后一个编码器块进行适应微调，相比线性探测可将CHB-MIT癫痫发作检测的ROC-AUC分别提升36.2和29.5个点。我们使用其发布的代码和一条独立流程对这些结果进行了复现研究。结果发现，该代码库只有在提供了一个原本未设置的配置键时才会为适应微调加载预训练权重，且其CBraMod探测头在dropout处于激活状态下读取重建头的输出。在正确进行预训练初始化、并在评估模式下对编码器输出施加探测的情况下，在DL的数据划分上，五个随机种子下的固定配方适应增益为：BIOT为-0.6个点（95%区间[-3.8, +2.6]），CBraMod为+0.01（[-0.02, +0.04]）。在DL的数据划分以及CHB-MIT和Siena头皮脑电数据集的五种受试者轮换设置下，使用同等调优的线性探测（30种配置、最多500轮、基于验证集的早停），平均适应效应范围为-1.1至……（原文摘要在此处截断）

    arXiv:2611.26929v1 Announce Type: new  Abstract: Dani and Liebe (2026) (DL) report that adapting the final encoder block of BIOT and CBraMod with self-supervised learning (SSL) improves CHB-MIT seizure-detection ROC-AUC by 36.2 and 29.5 points over linear probing. We investigate these results with their released code and an independent pipeline. The release loads pretrained weights for adaptation only when an otherwise unset configuration key is supplied, and its CBraMod probe reads the reconstruction head with dropout active. With pretrained initialisation and a probe on the encoder output in evaluation mode, the fixed-recipe adaptation gain on DL's split is -0.6 points for BIOT (95% interval [-3.8, +2.6]) and +0.01 for CBraMod ([-0.02, +0.04]) across five seeds. Across DL's split and five subject rotations on CHB-MIT and Siena Scalp EEG, with equally tuned linear probes (30 configurations, up to 500 epochs, validation-based early stopping), mean adaptation effects range from -1.1 to 
    
[^137]: ERA：面向冷启动跨模型潜在检索的嵌入、路由与自适应方法

    ERA: Embed, Route, Adapt for Cold-Start Cross-Model Latent Retrieval

    [https://arxiv.org/abs/2611.26919](https://arxiv.org/abs/2611.26919)

    ERA提出了一种冷启动跨模型潜在检索系统，通过余弦最远点采样选取少量锚点文档、利用结合正交正则化的流形对齐损失训练轻量级适配器，并借助确定性路由器升级处理困难查询，从而无需对整个语料库重新嵌入即可将检索系统迁移到新的嵌入模型。

    

    生产级检索系统与其语料库索引所使用的嵌入模型紧密绑定。将查询迁移到不同的编码器通常需要对整个语料库重新进行嵌入，而在冷启动阶段既没有查询日志可供利用，新编码器对每个文档进行嵌入也会产生API调用或计算成本。我们提出了ERA（Embed, Route, Adapt），一个面向冷启动跨模型潜在检索的系统。ERA将少量锚点预算分配给最具信息量的文档（通过余弦最远点采样从现有目标索引中选取），使用流形对齐损失训练一个轻量级适配器——该损失将方向对齐与正交正则化相结合以防止表示坍缩——并将该适配器与一个确定性路由器配对，后者会升级处理那些决定用户感知检索质量的少量困难查询。在五个BEIR数据集上，ERA达到或超越了逐点L2和嵌入转换基线，取得了相对增益……

    arXiv:2611.26919v1 Announce Type: cross  Abstract: Production retrieval systems are tied to the embedding model that indexed their corpus. Moving queries to a different encoder normally requires re-embedding the whole corpus, and at cold start there are no query logs while every document embedded by the new encoder incurs an API or compute cost. We present ERA (Embed, Route, Adapt), a system for cold-start cross-model latent retrieval. ERA spends a small anchor budget on maximally informative documents, selected from the existing target index via Cosine Farthest Point Sampling, trains a lightweight adapter with a Manifold Alignment Loss that couples directional alignment with orthogonal regularization to prevent representation collapse, and pairs the adapter with a deterministic router that escalates the rare, hard queries that define perceived retrieval quality. Across five BEIR datasets, ERA matches or outperforms point-wise L2 and embedding-conversion baselines, with relative gains 
    
[^138]: 视觉编码器带来了什么？无编码器多模态预训练中的视觉信息增益

    What Does the Visual Encoder Buy? Visual Information Gain in Encoder-Free Multimodal Pretraining

    [https://arxiv.org/abs/2611.26917](https://arxiv.org/abs/2611.26917)

    提出视觉信息增益（VIG）这一精确度量，将无编码器多模态模型从图像中获取的信息分解为全局信息与布局信息，发现图像可降低3%至46%的预测损失，全局信息在训练早期即出现而布局信息随训练缓慢增长。

    

    最近的规模研究通过在图文数据的文本下一词预测损失（next-token loss）上比较无编码器与基于编码器的模型，探讨视觉-语言模型是否仍需要视觉编码器。然而，这种损失可以在完全不使用图像的情况下下降，且两次具有相同损失的运行在从图像中获取的信息上可能存在差异。视觉信息增益（VIG）是指将图像替换为均匀图像时损失的增加量，它可以被精确地分解为全局信息（由图像统计特性携带）和布局信息（通过打乱图像内部的图像块即可破坏的信息）。我们将其应用于参数量从1.58亿到10.4亿的解码器，这些解码器通过线性嵌入读取像素块，每十亿词元评估一次直至400亿词元，同时还评估了19个公开检查点。在我们的解码器中，图像可将损失降低3%至46%，具体取决于图像类型。全局信息在训练的第一个十亿词元之后即已出现，而布局信息起初接近于零，并随训练缓慢增长。

    arXiv:2611.26917v1 Announce Type: cross  Abstract: Recent scaling studies ask whether vision-language models still need a visual encoder by comparing encoder-free and encoder-based models on the next-token loss of the text in image-text data. This loss can fall without any use of the image, and two runs with the same loss can differ in what they take from it. Visual information gain (VIG) is the loss increase when the image is replaced by a uniform one. It splits exactly into global information, carried by image statistics, and layout information, which shuffling the patches within the image destroys. We apply it to decoders with 158M to 1.04B parameters that read pixel patches through a linear embedding, evaluated every billion tokens up to 40B, and to 19 public checkpoints. In our decoders the image lowers the loss by 3% to 46%, depending on the image type. Global information is present after the first billion tokens. Layout information starts near zero, grows slowly with training an
    
[^139]: 多层级边缘分诊：为安全的临床智能体优化延迟

    Multi-Tier Edge Triage: Optimizing Latency for Safe Clinical Agents

    [https://arxiv.org/abs/2611.26914](https://arxiv.org/abs/2611.26914)

    该论文提出多层级边缘分诊安全筛查架构，通过在临床智能体宿主服务器本地部署轻量级第一层分类器以进程内处理明确查询、仅将模糊查询升级至云端LLM评审器，在保障多维安全合规的同时显著降低响应延迟。

    

    在企业临床工作流程中部署生成式AI智能体，需要在交互式延迟约束下执行多维度的安全与合规政策——包括防止HIPAA受保护健康信息（PHI）泄露、维持临床执业范围边界、确保面向患者阅读理解水平的适配性，以及维护品牌完整性。对云端大语言模型（LLM）评审器的同步外呼会在关键响应路径上增加大量网络与解码开销。我们提出了多层级边缘分诊，这是一种安全筛查架构，它在临床智能体的宿主服务器上共同部署一个紧凑的第一层序列分类器，从而在进程内直接处理明确的查询而无需网络往返，并仅将模糊的查询升级至第二层云端LLM评审器。为了在没有长度或一元词捷径的情况下评估第一层编码器，我们基于MedSafe构建了四个长度匹配的对比孪生基准。

    arXiv:2611.26914v1 Announce Type: new  Abstract: Deploying generative AI agents in enterprise clinical workflows requires enforcing multi-dimensional safety and compliance policies--preventing HIPAA Protected Health Information (PHI) exposure, maintaining clinical scope-of-practice boundaries, ensuring patient-appropriate reading comprehension, and preserving brand integrity--under interactive latency constraints. Synchronous outbound calls to cloud Large Language Model (LLM) judges add substantial network and decoding overhead onto the critical response path. We present Multi-Tier Edge Triage, a safety screening architecture that co-locates a compact Tier-1 sequence classifier on the clinical agent's host server, resolving clear queries in-process without a network round-trip and escalating only ambiguous queries to a Tier-2 cloud LLM judge. To evaluate Tier-1 encoders without length or unigram shortcuts, we construct four length-matched contrastive twin benchmarks seeded from MedSafe
    
[^140]: 表面线索优于结构：探究大语言模型在脑连接图上的行为

    Surface Cues Over Structure: Probing LLM Behavior on Brain Connectivity Graphs

    [https://arxiv.org/abs/2611.26913](https://arxiv.org/abs/2611.26913)

    该研究通过受控实验发现，大语言模型在基于脑功能连接图进行ADHD诊断时，主要依赖表面线索（如脑区名称）而非真正对受试者特定的连接结构进行推理。

    

    静息态功能磁共振成像为大脑间的功能交互提供了无创的观察视角，可以将其表示为连接图，用于神经系统疾病分析。大语言模型越来越多地与这类表示相结合，然而尚不清楚它们是真正对底层连接结构进行推理，还是仅仅对可能触发已学习先验的表面线索做出反应。我们通过对静息态功能连接的注意力缺陷/多动障碍（ADHD）诊断的受控研究来探究这一问题，每次只隔离一个因素，以确定是什么驱动大语言模型在脑连接图上的预测。在从小型开源权重LLM到专有模型的六个模型中，同样的模式均成立：结果几乎没有提供预测始终基于受试者特定连接的证据。大脑区域名称……

    arXiv:2611.26913v1 Announce Type: cross  Abstract: Resting-state functional magnetic resonance imaging (rs-fMRI) provides a noninvasive view of functional interactions across the brain, which can be represented as connectivity graphs for neurological disease analysis. Large language models (LLMs) are increasingly being paired with such representations, yet it remains unclear whether they genuinely reason over the underlying connectivity or simply respond to surface-level cues that may trigger learned priors. We probe this question through a controlled study of Attention-Deficit/Hyperactivity Disorder (ADHD) diagnosis from resting-state functional connectivity, isolating one factor at a time to determine what drives an LLM's predictions on brain connectivity graphs. Across six models, from small open-weight LLMs to a proprietary one, the same pattern holds: the results provide little evidence that predictions are consistently grounded in subject-specific connectivity. Brain region names
    
[^141]: 基于异构大语言模型求解器的预算感知预测-验证定理证明

    Budget-Aware Predict-and-Verify Theorem Proving with Heterogeneous LLM Solvers

    [https://arxiv.org/abs/2611.26906](https://arxiv.org/abs/2611.26906)

    本文提出BudgetPV，一种预算感知的预测-验证智能体系统，利用跨求解器预测器在异构LLM求解器之间动态分配计算资源，并根据证明进程的不同阶段（新子目标的求解器选择与已尝试子目标的算力追加）调整策略，从而在保证证明成功率的同时最小化定理证明的计算成本。

    

    大语言模型（LLM）定理求解器在证明能力和推理成本上各不相同。将昂贵的求解器应用于每个子目标，在较便宜的模型已足够时会浪费计算资源；而总是从最便宜的求解器开始，则会在需要更强模型的子目标上浪费尝试次数。我们提出了BudgetPV，这是一个智能体式的预算感知“预测-验证”系统，可在未解决的子目标和异构LLM求解器之间分配计算。BudgetPV使用跨求解器预测器来指导每个子目标的求解器选择，并随着验证反馈的获取不断修正分配方案。随着证明的推进，分配问题也在不断变化：早期决策主要确定应由哪个求解器处理尚未尝试过的子目标，而后期决策则确定哪些先前尝试过的子目标应获得额外的计算。我们针对这两个阶段使用两个目标函数来训练预测器。类别加权交叉熵……（原文摘要在此处截断）

    arXiv:2611.26906v1 Announce Type: new  Abstract: Large language model (LLM) theorem solvers differ in both proving capability and inference cost. Applying an expensive solver to every subgoal wastes computation when cheaper models are sufficient, while always starting from the cheapest solver wastes attempts on subgoals that require stronger models. We present BudgetPV, an agentic budget-aware predict-and-verify system that allocates computation across unresolved subgoals and heterogeneous LLM solvers. BudgetPV uses a cross-solver predictor to guide solver selection for each subgoal and revises the allocation as verification feedback becomes available.   The allocation problem changes as proving progresses. Early decisions mainly determine which solver should handle an untried subgoal, while later decisions determine which previously attempted subgoals should receive additional computation. We train the predictor with two objectives for these two stages. Class-weighted cross-entropy im
    
[^142]: 感知一次，按需升级：将视觉KV记忆从小型VLM传递给大型VLM

    Perceive Once, Escalate on Demand: Handing Off Visual KV Memory from Small to Large VLMs

    [https://arxiv.org/abs/2611.26902](https://arxiv.org/abs/2611.26902)

    提出Baton方法，通过从大型模型自身答案中无标签蒸馏逐层线性映射，将小型VLM的视觉KV缓存记忆传递给大型VLM，避免升级时重复编码历史信息，使大型模型只需编码当前帧。

    

    具身智能体无法在长视频的每一步都负担得起大型视觉语言模型（VLM），因此级联系统持续运行一个小型VLM，仅将不确定的决策升级给大型模型。然而，每次升级都迫使大型模型重新编码小型模型已经感知过的历史信息，这占据了其主要成本。我们提出了Baton，它将小型模型的视觉记忆——即其在历史信息上的键值（KV）缓存——交给大型模型。Baton基于一个发现：大型模型需要一个可用的记忆，而不是自身缓存的忠实副本。一个训练用于重现大型模型缓存的映射虽然与它最为接近，但相比单独使用小型模型却没有任何收益。Baton转而从大型模型自身的答案中无标签地蒸馏出一个逐层线性映射，大型模型只需编码当前帧。对该映射的干预实验将增益追溯到由大型模型中间三分之一层所携带的一个方向。

    arXiv:2611.26902v1 Announce Type: cross  Abstract: An embodied agent cannot afford a large vision-language model (VLM) at every step of a long video, so a cascade runs a small VLM continuously and escalates only uncertain decisions to a large one. Each escalation, however, forces the large model to re-encode a history the small model has already perceived, and this dominates its cost. We present Baton, which hands the large model the small model's visual memory, its key-value (KV) cache over the history. Baton rests on one finding: the large model needs a usable memory, not a faithful copy of its own. A map trained to reproduce the large model's cache matches it most closely, yet brings no gain over the small model alone. Baton instead distills a per-layer linear map, without labels, from the large model's own answers, with the large model encoding only the current frame. Interventions on the map trace the gain to one direction, carried by the middle third of the large model's layers. 
    
[^143]: 你的评审器会继承你的提示词：LLM即评审中的上下文标签偏置是有方向性的，而非对称的

    Your Judge Inherits Your Prompt: In-Context Label Bias in an LLM-as-Judge Is Directional, Not Symmetric

    [https://arxiv.org/abs/2611.26881](https://arxiv.org/abs/2611.26881)

    研究表明，多模态视频问答评估流水线中的LLM评审器会从其少样本示例中单向继承标签偏置：当示例偏向"Yes"时其接受率骤升约20–30个百分点并伴随单侧错误，而当示例平衡或偏向"No"时则接近人类共识，证明这种上下文标签偏置是方向性的而非对称的。

    

    LLM-as-judge（以大语言模型作为评审器）已成为对开放式模型输出进行评分的标准方法，然而评审器本身是一个由少量上下文示例引导的冻结语言模型，而此类模型已知会吸收其演示示例中的标签分布，而非将其视为中性的示例。我们探究这种标签偏置是否会传导到多模态（视频问答）评估流水线中使用的LLM评审器，以及它以何种形式呈现。通过使用完全冻结的生成器-评审器流水线，我们保持生成的答案不变，仅在四个演示集之间改变评审器少样本示例的Yes/No（是/否）平衡。每个判定结果均与每个数据集200个样本上的双人标注人类共识进行对照验证。当评审器的示例为平衡或偏向拒绝时，其接受率保持接近人类共识；但当示例偏向接受时，接受率急剧跳升（在NExT-QA上约30个百分点，在MSVD-QA上约20个百分点，且呈现强烈单侧的错误特征）……（原文摘要在此处截断）

    arXiv:2611.26881v1 Announce Type: cross  Abstract: LLM-as-judge has become a standard way to score open-ended model outputs, yet the judge is itself a frozen language model steered by a few in-context examples, and such models are known to absorb the label balance of their demonstrations rather than treat them as neutral illustrations. We ask whether this label bias reaches an LLM judge used inside a multimodal (video question answering) evaluation pipeline, and what form it takes. Using a fully frozen generator-judge pipeline, we hold the generated answers fixed and vary only the Yes/No balance of the judge's few-shot examples across four demonstration sets. Each verdict is validated against a two-annotator human consensus on 200 examples per dataset. The judge's acceptance rate stays close to human consensus when its examples are balanced or skewed toward rejection, but jumps sharply (by about 30 points on NExT-QA and 20 on MSVD-QA, with a strongly one-sided error profile) when the e
    
[^144]: 语言模型如何压缩与比较：通过词元协方差图理解选择任务

    How Language Models Compress and Compare: Understanding Selection with Token Covariance Maps

    [https://arxiv.org/abs/2611.26874](https://arxiv.org/abs/2611.26874)

    该论文提出词元协方差图这一提示级几何分析工具，揭示了仅解码器Transformer在结构化选择任务中普遍遵循“先压缩后比较”的层级机制——早期层弥散表示提示、中间层压缩为共享低秩瓶颈、后期层重构上下文与选项间的比较几何。

    

    许多语言模型任务都可归结为在结构化备选项之间进行选择，然而我们仍然缺乏一个简单的图景来理解仅解码器Transformer如何在深度维度上组织整个选择提示。我们引入了词元协方差图，这是一种提示层面的几何视角，用于追踪各层中词元位置在隐藏状态空间中如何共同变化。这将结构化提示转化为按深度索引的图谱序列，其谱结构和块结构可以与指令、上下文和候选答案对齐。在来自4个系列的14个仅解码器模型和12个结构化选择基准上，词元协方差图揭示了一种稳健的“先压缩后比较”组织方式：早期层以弥散的方式表示提示，中间层将其压缩为共享的低秩瓶颈，后期层则重新扩展为上下文-选项和选项-选项的几何结构。这种组织方式在更大的固定模板样本、确定性的提示重写、成对……（摘要原文在此处截断）

    arXiv:2611.26874v1 Announce Type: new  Abstract: Many language-model tasks reduce to selecting among structured alternatives, yet we still lack a simple picture of how a decoder-only transformer organizes an entire selection prompt across depth. We introduce token covariance maps, a prompt-level geometric view that tracks how token positions co-vary in hidden-state space at each layer. This turns a structured prompt into a depth-indexed sequence of maps whose spectral and block structure can be aligned with instructions, context, and candidate answers. Across 14 decoder-only models from 4 families and 12 structured-selection benchmarks, token covariance maps reveal a robust compress-then-compare organization. Early layers represent prompts diffusely, middle layers compress them into a shared low-rank bottleneck, and late layers re-expand into context-option and option-option geometry. This organization recurs across larger fixed-template samples, deterministic prompt rewrites, pairwise
    
[^145]: 大语言模型在后训练过程中会记住自己产生的幻觉

    LLMs Memorize Their Own Hallucinations During Post-Training

    [https://arxiv.org/abs/2611.26863](https://arxiv.org/abs/2611.26863)

    该论文发现，采用以推理为中心的后训练方案（如基于结果的强化学习奖励和“认知”行为注入）会导致大语言模型在知识任务训练中记住自己在长推理链中产生的幻觉，且这一现象在多个开源模型上得到验证。

    

    推理领域已经塑造了前沿模型的后训练方案，即使这些模型是在数学和代码之外的领域进行训练和部署的。在本文中，我们研究了直接采用以推理为中心的训练方案（例如，在强化学习（RL）中使用基于结果的奖励、在推理链中注入“认知”行为如提出假设并从中回溯等）对知识任务的影响。我们假设这些决策虽然对推理有益，但在知识任务上训练时，会在无意中鼓励模型记住它们在长推理链中自己产生的幻觉。为了研究这一点，我们使用标准后训练方案训练模型，在强化学习过程中提取它们自我生成的幻觉，并在后续训练过程中跟踪这种记忆现象。我们在两个模型家族、两种架构共六个开源模型上的实验证实了我们的假设。我们发现

    arXiv:2611.26863v1 Announce Type: new  Abstract: Reasoning domains have shaped the post-training recipes of frontier models even though they are trained and deployed in domains beyond math and code. In this paper, we study the consequences of directly adopting reasoning-centric training recipes, e.g., the use of outcome-based rewards for reinforcement learning (RL), injection of "cognitive" behaviors such as proposing and backtracking from hypotheses in reasoning chains, etc., for knowledge tasks. We hypothesize that these decisions, while useful for reasoning, inadvertently encourage models to memorize hallucinations they themselves generate in their long reasoning chains when trained on knowledge tasks. To study this, we train models with standard post-training, extract their self-generated hallucinations during RL, and track memorization throughout the rest of training. Our experiments on six open models across two model families and two architectures confirm our hypothesis. We find
    
[^146]: 基于递归正交滤波器的神经时间序列自监督学习

    Self-Supervised Learning with Recursive Quadrature Filters for Neural Time Series

    [https://arxiv.org/abs/2611.26856](https://arxiv.org/abs/2611.26856)

    本文提出一种自监督学习方法，通过让可学习的递归正交滤波器（一类受生物学启发的对角状态空间带通滤波器）编码器的当前状态预测其自身未来活动，在无标签神经时间序列上预训练编码器，再结合读出层完成下游任务。

    

    我们开发了在无标签时间序列上训练递归正交滤波器（RQF）编码器的自监督方法。RQF是一类受生物学启发的复数值时间滤波器，构成对角状态空间模型（SSM）的一个特殊类别。每个RQF作为一个带通滤波器，其选择性由两个可学习参数决定：调谐频率和带宽。我们另外引入了一种基于预测误差的自监督学习目标；编码器经过训练，使其当前状态能够预测自身的未来活动。在窗口模式下，当前窗口的摘要预测后续窗口的摘要；在密集模式下，当前时间步的状态预测未来的对数功率。每个RQF编码器先在数据集上进行预训练，然后在RQF编码器的输出上训练一个读出层（例如任务特定的分类器），此时编码器可以保持冻结或进行完全微调。对于多通道脑电图（EEG）和颅内……

    arXiv:2611.26856v1 Announce Type: new  Abstract: We develop self-supervised methods for training Recursive Quadrature Filter (RQF) encoders on unlabeled time series. RQFs are biologically motivated, complex-valued temporal filters that form a special class of diagonal state-space models (SSMs). Each RQF acts as a band-pass filter whose selectivity is determined by two learnable parameters: its tuning frequency and bandwidth. We separately introduce a self-supervised learning objective based on prediction error; the encoder is trained so that its current state predicts its own future activity. In the windowed mode, the summary of the current window predicts the summary of a later window; in the dense mode, the state at the current timestep predicts future log-power. Each RQF encoder is pretrained on a dataset and then a readout (e.g., a task-specific classifier) is trained on the output of the RQF encoder, with the encoder either frozen or fully fine-tuned. For multichannel EEG and intr
    
[^147]: LoCo-WAM：将测试时训练作为世界-动作模型的长上下文

    LoCo-WAM: Test-Time Training as Long Context for World-Action Models

    [https://arxiv.org/abs/2611.26853](https://arxiv.org/abs/2611.26853)

    LoCo-WAM 通过测试时训练利用新观测帧更新视频专家中的低秩适配器，使世界-动作模型在不延长观测窗口、无需专家动作标签、额外数据或元训练的情况下从更长的经验流中学习，从而在四个仿真基准上显著提升任务成功率。

    

    世界-动作模型通过预测接下来会发生什么来指导机器人动作，但通常仅依赖于较短的观测历史。扩展这一历史会增加每次决策时的计算量，并且可能需要单独训练一个模型。我们提出了 LoCo-WAM，它在保持相同短历史长度的同时，通过测试时训练从自身动作的结果中学习。它利用新观测到的帧来更新视频专家中的低秩适配器，其适配后的预测用于指导后续动作。这些更新在每个回合内累积，并在回合之间重置，使过去的观测在离开输入窗口之后仍能影响决策。因此，LoCo-WAM 无需专家动作标签、额外数据或元训练，即可从更长的经验流中学习。在两个世界-动作模型上的实验表明，该方法在四个仿真基准上均提升了任务成功率。在 RoboCasa GR-1 上，一个仅有 2.4 秒历史的模型能够匹配……（原文摘要在此处截断）

    arXiv:2611.26853v1 Announce Type: cross  Abstract: World-action models guide robot actions by predicting what happens next, but typically condition on a short observation history. Extending this history increases computation at every decision and can require a separately trained model. We introduce LoCo-WAM, which learns from the outcomes of its actions through test-time training while keeping the same short history length. It uses newly observed frames to update a low-rank adapter in the video expert, whose adapted predictions guide subsequent actions. The updates accumulate within each episode and are reset between episodes, allowing past observations to influence decisions after they leave the input window. LoCo-WAM thus learns from a longer stream of experience without expert action labels, additional data or meta-training. Experiments on two world-action models show improved task success across four simulation benchmarks. On RoboCasa GR-1, a model with 2.4,s of history matches one
    
[^148]: Scoliosis5K：基于片段条件化的RGB-轮廓融合的多模态步态分析

    Scoliosis5K: Multimodal Gait Analysis via Clip-Conditioned RGB-Silhouette Fusion

    [https://arxiv.org/abs/2611.26849](https://arxiv.org/abs/2611.26849)

    该论文构建了多中心多模态步态数据集Scoliosis5K，并提出片段条件化的RGB-轮廓融合架构，用于脊柱侧凸相关步态的参与者级分类。

    

    彩色行走视频在记录人体边界运动的同时，也包含衣物褶皱、表面外观和录制环境等线索。为了考察当分类器已经观察到配对的彩色序列时，显式的二值轮廓对参与者级别的脊柱侧凸相关步态分类有何贡献，我们构建了Scoliosis5K数据集——一个多中心采集的数据集，包含3,010名参与者的4,998条走廊行走序列，其中RGB人体裁剪图、二值轮廓和二维姿态共享相同的录制标识符与帧索引。我们的架构利用每个图像块像素相对于其片段均值的偏差，在预训练的RGB编码器内对两个低秩的查询和值更新进行加权，并将所得的视频表示与轮廓Transformer序列以及从每个窗口中一帧原始分辨率RGB帧提取的区域核岭回归分数进行融合。在五个参与者不相交的交叉验证折中，对988名参与者进行了评估……（摘要原文在此处截断）

    arXiv:2611.26849v1 Announce Type: cross  Abstract: Color walking video records body boundary motion alongside clothing folds, surface appearance, and recording-environment cues. To examine what an explicit binary silhouette contributes to participant-level classification of scoliosis-related gait when a classifier already observes the paired color sequence, we construct Scoliosis5K, a multi-site collection of 4,998 corridor-walking sequences across 3,010 participant identifiers in which RGB person crops, binary silhouettes, and two-dimensional poses share each recording identifier and frame index. Our architecture uses each patch's pixel deviation from its clip mean to weight two low-rank query and value updates inside a pretrained RGB encoder, fusing the resulting video representation with a silhouette transformer sequence and regional kernel ridge scores extracted from one native-resolution RGB frame per window. Across 988 participants in five participant-disjoint cross-validation fo
    
[^149]: 交通预测架构的稀疏性分析：多少缺失数据才算过多？

    A Sparsity Analysis of Traffic Forecasting Architectures: How Much Missing Data is Too Much?

    [https://arxiv.org/abs/2611.26847](https://arxiv.org/abs/2611.26847)

    本文系统评估了图神经网络、Transformer和选择性状态空间模型等交通预测架构在从零到完全传感器缺失的不同稀疏度下的表现，揭示了基准精度与缺失数据鲁棒性之间的权衡，并定义了预测可靠性阈值。

    

    交通预测模型通常在数据完整的理想条件下进行评估，这种评估方式未能反映现实部署中频繁出现的传感器故障和通信中断。本文对多变量交通预测模型进行了全面分析，评估其在推理过程中面对不同程度传感器数据缺失时的性能表现。我们在从零缺失到完全传感器丢失的稀疏度范围内，使用两个广泛使用的多通道交通基准数据集，对包括图神经网络、基于注意力的Transformer以及选择性状态空间模型在内的多种架构范式进行了基准测试。为确保各交通变量之间的比较一致性，我们采用归一化指标，并定义了一个可用性阈值，用于识别预测何时变得不可靠。我们的结果显示出基准精度与鲁棒性之间存在明显的权衡：在全数据条件下精度最高的模型……（原文摘要在此处截断）

    arXiv:2611.26847v1 Announce Type: new  Abstract: Traffic forecasting models are typically evaluated under ideal conditions with complete data. This evaluation does not reflect the frequent sensor failures and communication outages common in real-world deployments. This paper presents a comprehensive analysis of multivariate traffic prediction models to evaluate their performance under varying levels of sensor data loss during inference. We benchmark several architectural paradigms, including graph neural networks, attention-based Transformers, and selective state-space models, across sparsity levels ranging from zero to complete sensor loss, using two widely used multi-channel traffic benchmarks. To ensure consistent comparison across traffic variables, we use normalized metrics and define a usability threshold to identify when predictions become unreliable. Our results show a clear trade-off between baseline accuracy and robustness: models with the highest accuracy under full-data con
    
[^150]: 神经几何作为学习信号

    Neural Geometry as a Learning Signal

    [https://arxiv.org/abs/2611.26832](https://arxiv.org/abs/2611.26832)

    本文证明只需监督模型内部表示的几何结构（如傅里叶谐波和低维螺旋结构），就能在完全不监督输出的情况下取得与标准微调相当的模型改进效果。

    

    我们研究神经表示的几何结构能否提供一种足够的学习信号，从而在不监督模型输出的情况下改进模型行为。首先，我们通过监督输入月份（如 June = 6）和偏移量（4）的内部表示来微调 Llama-3.1-8B，使其能够回答诸如“六月之后的四个月是几月？”这类问题，这些内部表示以傅里叶谐波的形式编码。其次，我们微调模型来预测按固定宽度换行的文本中的换行符，方法是监督“自上一个换行符以来的字符数”这一内部表示，该表示由一个低维螺旋结构刻画。在这两个任务中，我们都使用理想化的目标结构来监督残差流的一个低维子空间，结果发现内部几何监督能够达到与标准微调相似的性能（尽管完全没有对模型输出进行监督）。此外，我们发现简单地修补 i……（摘要原文在此处被截断）

    arXiv:2611.26832v1 Announce Type: new  Abstract: We investigate whether the geometric structure of neural representations can provide a sufficient learning signal for improving model behavior without any supervision on model outputs. First, we fine-tune Llama-3.1-8B to answer questions such as "What month is four months after June?" by supervising internal representations of the input month (June = 6) and offset (4), which are encoded with Fourier harmonics. Second, we fine-tune the model to predict line breaks in text wrapped to a fixed width by supervising an internal representation of the number of characters since the preceding newline, which is represented by a low-dimensional helical structure. In both tasks, we supervise a low-dimensional subspace of the residual stream using an idealized target structure, and find that internal geometric supervision achieves similar performance to standard fine-tuning (despite no supervision on outputs). Moreover, we find that simply patching i
    
[^151]: Graph-XGB：一种基于纯空中无人机射频感知的室外到室内5G信号重构混合图神经网络框架

    Graph-XGB: A Hybrid GNN Framework for Outdoor-to-Indoor 5G Signal Reconstruction via Aerial-Only UAV RF Sensing

    [https://arxiv.org/abs/2611.26828](https://arxiv.org/abs/2611.26828)

    提出Graph-XGB混合推理框架，通过改进的SpiderGNN编码器（结合楼层感知三维距离与桥接边保证）将无人机边界射频测量传播为室内嵌入，并与非线性回归解耦，实现仅凭空中感知的室外到室内5G信号三维重构。

    

    提供稳健的楼内覆盖被严重的室外到室内（O2I）衰减所严重阻碍，随着部署从5G中频段向毫米波及未来6G亚太太赫兹频段演进，这一制约进一步加剧——据报道，现代建筑围护结构造成的损耗可达60-80 dB。虽然无人机提供了可扩展的外部射频感知平台，但一个根本性的反演问题依然存在：室内三维信号状态无法从受限的外部飞行路径中直接采样获得。本文提出Graph-XGB，一种将空间拓扑推理与非线性信号回归相解耦的混合推理框架，以解决这一挑战。经过改进的SpiderGNN编码器，在基于物理意义的楼层感知三维距离度量和桥接边保证所构建的拓扑感知图上运行，将无人机边界测量传播为128维室内嵌入。主成分……（摘要原文在此处截断）

    arXiv:2611.26828v1 Announce Type: new  Abstract: Delivering robust In-Building Coverage (IBC) is critically impeded by severe Outdoor-to-Indoor (O2I) attenuation, a constraint that intensifies as deployments move from 5G mid-band towards mmWave and prospective 6G sub-THz carriers, where losses of 60-80 dB have been reported across modern building envelopes. While Unmanned Aerial Vehicles (UAVs) offer scalable exterior RF sensing platforms, a fundamental inversion problem persists: interior 3D signal states cannot be directly sampled from constrained exterior flight paths. This paper proposes Graph-XGB, a hybrid inference framework that decouples spatial topology reasoning from non-linear signal regression to resolve this challenge. A modified SpiderGNN encoder, operating on a topology-aware graph constructed with a physically grounded Floor-Aware 3D Distance metric and Bridge Edge Guarantees, propagates UAV boundary measurements into 128-dimensional indoor embeddings. Principal Compone
    
[^152]: Skill-JEPA：为世界模型学习与规划器对齐的动作空间

    Skill-JEPA: Learning Planner-Aligned Action Spaces for World Models

    [https://arxiv.org/abs/2611.26825](https://arxiv.org/abs/2611.26825)

    Skill-JEPA通过状态条件变分自编码器将短动作序列编码为符合标准高斯先验的潜技能，让世界模型预测技能效果、规划器在同一高斯分布上搜索技能，从而解决了规划器采样动作与模型训练动作分布不匹配导致预测不可靠、远距离目标规划易失败的问题。

    

    基于联合嵌入预测架构（JEPA）的世界模型从像素中学习，在潜空间中预测动作产生的结果。为了达成目标，规划器会采样大量候选动作序列，利用模型预测每个序列的最终结果，并执行其中最优的一个。然而，标准规划器从高斯分布中抽取候选动作，而模型在其训练数据中只见过结构化的、依赖状态的动作，因此模型对此类候选动作的预测并不可靠，随着目标距离变远，规划失败的频率也会增加。我们提出了Skill-JEPA，它学习一个与规划器对齐的动作空间。一个以当前状态为条件的变分自编码器将短动作序列编码为具有标准高斯先验的潜技能，并将其解码回动作。世界模型预测技能的效果而非原始动作，规划器则从同一个高斯分布出发在技能空间中进行搜索。

    arXiv:2611.26825v1 Announce Type: cross  Abstract: World models based on Joint-Embedding Predictive Architectures (JEPAs) learn from pixels to predict the outcome of actions in a latent space. To reach a goal, a planner samples many candidate action sequences, uses the model to predict where each one leads, and executes the best one. However, standard planners draw their candidates from a Gaussian, while the model has only seen the structured, state-dependent actions of its training data. Its predictions for such candidates are unreliable, and planning fails more often as goals move farther away. We propose Skill-JEPA, which learns an action space aligned with the planner. A variational autoencoder conditioned on the current state encodes short action sequences into latent skills with a standard Gaussian prior, and decodes them back into actions. The world model predicts the effect of skills rather than raw actions, and the planner searches over skills starting from that same Gaussian,
    
[^153]: 智能网络攻击检测与告警关联的统一框架

    A Unified Framework for Intelligent Cyber Attack Detection and Alert Correlation

    [https://arxiv.org/abs/2611.26824](https://arxiv.org/abs/2611.26824)

    本文提出一个融合签名规则、机器学习和序列模式挖掘算法的统一框架，用于网络流量分类与告警关联，以缓解安全运营中心（SOC）的告警过载问题并有效检测多阶段网络攻击。

    

    安全运营中心（SOC）在现代网络威胁的检测、关联和响应中发挥着关键作用。然而，SOC 仍持续面临告警过载、检测能力碎片化以及分析师与自动化系统之间协作有限等挑战。这些问题常常导致事件响应延迟、分析师疲劳，并增加了忽视那些在网络不同层面逐步展开的复杂多阶段攻击的风险。为应对这些挑战，本文提出了一个通过综合运用签名规则、机器学习和序列模式挖掘算法来实现网络流量分类与告警关联的统一框架。该解决方案在一个模拟真实多阶段攻击场景的可控测试环境中进行了评估，攻击场景包括恶意软件利用、横向移动、侦察和数据渗出等。所获得的结果表……

    arXiv:2611.26824v1 Announce Type: cross  Abstract: A Security Operations Center (SOC) plays a critical role in detecting, correlating, and responding to modern cyber threats. However, SOCs continue to face challenges related to alert overload, fragmented detection capabilities, and limited collaboration between analysts and automated systems. These issues often result in delayed incident response, analyst fatigue, and an increased risk of overlooking sophisticated, multi-stage attacks that unfold gradually across different layers of the network. To address these challenges, this paper proposes a unified framework for network traffic classification and alert correlation through the use of signature rules, machine learning, and sequential pattern mining algorithms. The solution was evaluated in a controlled testbed environment emulating realistic, multi-stage attack scenarios, including malware exploitation, lateral movement, reconnaissance, and data exfiltration. The obtained results de
    
[^154]: 表格数据驱动模型的现实鲁棒性评估

    Realistic Robustness Evaluation of Tabular Data Driven Models

    [https://arxiv.org/abs/2611.26818](https://arxiv.org/abs/2611.26818)

    本文提出了AURORA，一个模块化开源工具，通过在统一框架中整合多种对抗扰动方法和评估指标，为表格数据分类的机器学习模型提供系统化的鲁棒性评估。

    

    机器学习（ML）模型在高风险应用中的日益普及，引发了人们对其抵御对抗性攻击鲁棒性的严重担忧。尽管大量研究集中于图像和文本领域的对抗样本生成，但表格数据虽然在现实世界系统中被广泛使用，其对抗样本的研究仍相对有限。本工作介绍了AURORA，一个用于评估ML模型在表格数据分类任务上鲁棒性的模块化开源工具。它在一个统一框架内整合了多种对抗扰动方法和评估指标，以系统地评估模型在对抗条件下的行为。AURORA的核心是其评估报告，该报告专注于鲁棒性分析，通过量化性能下降、比较不同扰动策略的影响，以及识别模型在各特征和输入区域上的脆弱性来展开分析。

    arXiv:2611.26818v1 Announce Type: cross  Abstract: The increasing adoption of Machine Learning (ML) models in high-stakes applications has raised critical concerns regarding their robustness against adversarial attacks. While extensive research has focused on adversarial example generation in image and text domains, the study of such examples in tabular data remains comparatively limited, despite its widespread use in real-world systems. This work introduces AURORA, a modular, open-source tool for evaluating the robustness of ML models on tabular data classification. It integrates multiple adversarial perturbation methods and evaluation metrics within a unified framework to systematically assess model behavior under adversarial conditions. Central to AURORA is its evaluation report, which focuses on robustness analysis by quantifying performance degradation, comparing the impact of different perturbation strategies, and identifying model vulnerabilities across features and input region
    
[^155]: 从一开始就学会搜索：面向语言模型的搜索原生预训练

    Search From the Start: Search-Native Pretraining for Language Models

    [https://arxiv.org/abs/2611.26809](https://arxiv.org/abs/2611.26809)

    提出“搜索原生预训练”方法，通过大规模地将网络搜索查询及检索到的网页响应注入预训练语料，让语言模型在预训练阶段就学会发起搜索并利用检索证据，从而在知识密集型任务上取得更优性能，且该优势在监督微调后依然保持。

    

    现代语言模型（LM）越来越多地在推理阶段配备搜索工具进行部署。然而，预训练仍然主要聚焦于对静态文本的预测，搜索能力通常是在后训练阶段才被引入。我们提出“搜索原生预训练”作为一种实用路径，在预训练阶段就基于普通文本语料教会模型搜索能力。具体而言，我们大规模地用网络搜索查询来增强预训练语料，并使用本地索引检索网络响应。在这种增强语料上的训练使模型学会生成搜索调用，并在语言建模过程中融入检索到的证据。在Wikipedia和DCLM-Edu语料库以及360M至3.3B的参数规模上，搜索预训练模型在知识密集型任务上表现出更高的性能，并且对检索到的证据具有更强的可控性（可引导性）。关键的是，在对搜索预训练模型和常规模型进行监督微调之后，它们的优势依然得以保持。

    arXiv:2611.26809v1 Announce Type: cross  Abstract: Modern language models (LMs) are increasingly deployed with access to search tools at inference. Yet pretraining remains predominantly focused on predicting static text, with search capabilities typically introduced during post-training. We propose Search-Native Pretraining as a practical route to teach search capability during pretraining from ordinary text corpora. Specifically, we augment pretraining corpus at scale with web-search queries and retrieve web response using a local index. Training on this augmented corpus teaches models to generate search calls and incorporate retrieved evidence during language modeling. Across Wikipedia and DCLM-Edu corpora and 360M to 3.3B parameter scales, search-pretrained models ("QueryLM") show improved performance on knowledge-intensive tasks and higher steerability toward the retrieved evidence. Crucially, their advantage persists after supervised fine-tuning of both search-pretrained and conve
    
[^156]: NitroGen：关于动作动态、视觉指令遵循与空间探索的观察

    NitroGen: Observations on the Action Dynamics, Visual Instruction Following, and Spatial Exploration

    [https://arxiv.org/abs/2611.26807](https://arxiv.org/abs/2611.26807)

    NitroGen视觉-动作基础模型虽然能根据局部游戏情境适应动作，但在长时程游戏控制中仍缺乏规划、恢复和探索能力。

    

    NitroGen是一个用于游戏控制的开放视觉-动作基础模型，它通过在在线游戏视频上进行行为克隆训练而成。在本文中，我们以《空洞骑士》和《赛道狂飙》中的冻结控制器为例，从动作适应、视觉指令遵循和空间探索三个维度对该模型进行研究。我们的结果表明，NitroGen能够根据局部上下文调整其动作，但它缺乏长时程游戏控制所需的规划、恢复和探索能力。

    arXiv:2611.26807v1 Announce Type: new  Abstract: NitroGen is an open vision-action foundation model for game control. It is trained by behavior cloning on online gameplay videos. In this paper, we study the frozen controller in Hollow Knight and Trackmania as examples along three dimensions: action adaptation, visual instruction following, and spatial exploration. Our results indicate that NitroGen can adapt its actions to local context, but it lacks the planning, recovery, and exploration needed for longer-horizon game control.
    
[^157]: LEAN：面向异步联邦微调的基于库的自适应方法

    LEAN: Library-Based Adaptation for Asynchronous, Federated Fine-Tuning

    [https://arxiv.org/abs/2611.26805](https://arxiv.org/abs/2611.26805)

    LEAN提出用服务器维护的秩-1基对“借阅库”机制取代联邦学习中的适配器平均聚合，各参与者检出子集本地微调后归还，从而实现完全异步、仅传输低秩参数的联邦微调。

    

    使用低秩适配器（LoRA）对基础模型进行联邦微调时，通常通过对本地训练的适配器取平均来协调。然而在跨设备场景中，参与者持有异构数据且异步加入或退出，平均方法并不适用：本地训练的适配器在置换意义下不可辨识，且在不相似数据上训练的适配器的平均值可能是无信息的或不稳定的。我们提出一种称为基于库的联邦自适应（LEAN）的简单想法，它完全摒弃了平均操作。服务器维护一个秩-1基对库；参与者随机检出子集，在本地将其作为LoRA适配器进行微调，然后归还，从而使每对基在多个站点间被“洗牌”，并在众多参与者的数据上得到训练。由于不需要聚合步骤，LEAN可以完全异步运行，且仅传输低秩参数。我们证明，在同步设置下，LEAN收敛到（摘要在此处截断）

    arXiv:2611.26805v1 Announce Type: new  Abstract: Federated fine-tuning of foundation models with low-rank adapters (LoRA) is typically reconciled by averaging locally trained adapters. In cross-device settings, where participants hold heterogeneous data and join or leave asynchronously, averaging is poorly suited: locally trained adapters are not identifiable up to permutation, and averages of adapters trained on dissimilar data can be uninformative or unstable. We propose a simple idea called federated library-based adaptation (LEAN), which replaces averaging altogether. The server maintains a library of rank-1 basis pairs; participants check out random subsets, fine-tune them locally as a LoRA adapter, and check them back in, so that each pair is shuffled across sites and trained on many participants' data. Because no aggregation step is required, LEAN operates fully asynchronously and communicates only low-rank parameters. We prove that, in a synchronous setting, LEAN converges to a
    
[^158]: 不要丢弃你的推理轨迹：重用教师强化学习轨迹进行学生模型蒸馏

    Don't Discard Your Rollouts: Reusing Teacher RL Traces for Student Distillation

    [https://arxiv.org/abs/2611.26803](https://arxiv.org/abs/2611.26803)

    提出利用GRPO训练中已记录的元数据（提示通过率和教师对数似然）将推理轨迹重用建模为数据选择问题，通过SFT预热加DPO的分阶段蒸馏策略，把教师的强化学习轨迹高效重用于学生模型训练，避免了昂贵的重新生成数据。

    

    诸如PPO和GRPO等强化学习后训练方法计算成本高昂，大量资源被用于采样推理轨迹，这些轨迹包含每个提示的多个补全及其相应的奖励。然而，传统蒸馏方法会丢弃这些轨迹，并使用更多算力从最终教师模型重新生成数据。我们探究是否以及如何重用保存下来的教师强化学习推理轨迹来蒸馏学生模型。重用推理轨迹并非易事：在推理轨迹上进行朴素的SFT表现不如传统蒸馏，而朴素的DPO虽然性能尚可，但不稳定，在较弱的学生模型上会崩溃。我们证明，利用GRPO训练过程中已记录的元数据，将推理轨迹重用视为一个数据选择问题，可以大幅缩小这一差距。我们使用提示通过率对数据进行排序，先用最简单的提示进行SFT预热启动，然后再用成功率参半的提示进行DPO。对于每个提示，我们使用保存的教师模型对数似然作为选择样本的代理指标。

    arXiv:2611.26803v1 Announce Type: cross  Abstract: RL post-training methods such as PPO and GRPO are computationally expensive, with substantial resources devoted to sampling rollouts, which contain multiple completions per prompt along with their respective rewards. Canonical distillation, however, discards these traces and uses more compute to regenerate data from the final teacher. We ask if and how we can reuse the saved teacher RL rollouts to distil student models. Reusing rollouts is non-trivial: naive SFT on the rollouts underperforms canonical distillation, and naive DPO, while performant, is unstable, collapsing on weaker students. We show that treating rollout reuse as a data selection problem using metadata already recorded during GRPO substantially reduces this gap. We use prompt pass rates to order the data, using the easiest prompts for an SFT warm start prior to using mixed-success prompts for DPO. For each prompt, we use saved teacher log-likelihoods as a proxy for sele
    
[^159]: 生成模型中的记忆与泛化：基于能量建模与联想记忆的视角

    Memorization and Generalization in Generative Models: A Perspective From Energy-based Modeling and Associative Memory

    [https://arxiv.org/abs/2611.26797](https://arxiv.org/abs/2611.26797)

    本文从能量建模与联想记忆的视角提出：扩散模型本质上是一种联想记忆，随着数据增长而涌现的“伪状态”吸引子正是生成能力的开端，因此记忆与泛化可以在同一检索过程中共存。

    

    生成模型如今已能编写可运行的代码、生成逼真的图像并进行长对话，但其智能背后的机制仍未被我们理解。本论文通过联想记忆（Associative Memory, AM）来探索这一谜题：在联想记忆中，模式以能量极小值的形式被存储，并通过吸引子动力学进行检索。除了存储容量之外，AM 还会形成一类被称为“伪状态”的新极小值，这长期以来被视为记忆的失败。我们提出，这种“失败”正是创造力的起点，记忆与泛化可以作为同一检索过程的两种结果而共存。首先，我们证明扩散模型表现为一种联想记忆。当数据很少时，扩散模型会在每个训练样本周围形成各自独立的吸引盆；随着数据量增长，这些吸引盆逐渐缩小，而伪状态作为融合了已存储样本却不匹配任何单个样本的吸引子开始涌现——它们正是生成能力的最初征兆。我们通过……刻画了这一转变（原文摘要在此处截断）

    arXiv:2611.26797v1 Announce Type: new  Abstract: Generative models now write working code, produce realistic images, and hold long conversations, yet the mechanisms behind their intelligence still evade our understanding. This thesis explores that puzzle through Associative Memory (AM), in which patterns are stored as energy minima and retrieved through attractor dynamics. Beyond their capacity, AMs form new minima known as spurious states, long regarded as a failure of memory. We argue that this failure is where creativity begins, and that memorization and generalization can coexist as outcomes of one retrieval process. First, we show that diffusion models behave as AMs. With little data, a diffusion model forms a distinct basin of attraction around each training sample. As the data grows, these basins shrink, and spurious states emerge as attractors that blend stored samples yet match none of them. They are the first signs of generative capability. We characterize this transition by 
    
[^160]: CADENCE：保持富有表现力手部运动的时序连贯全身人体网格恢复

    CADENCE:Temporally Coherent Whole-Body Human Mesh Recovery that Preserves Expressive Hand Motion

    [https://arxiv.org/abs/2611.26783](https://arxiv.org/abs/2611.26783)

    CADENCE提出了一种面向单目视频的全身SMPL-X网格恢复系统，通过沿运动学树的注意力机制为每个关节自适应选择学习到的零相位滤波器混合，在平滑躯干运动的同时保留快速手指动作等富有表现力的手部运动。

    

    全身SMPL-X估计器可以从单张图像中恢复身体、手部和面部网格，但逐帧应用于视频时会产生抖动的运动。对输出进行平滑虽然能消除抖动，却会带来仅身体基准测试无法衡量的代价：足够强的时序滤波器虽然能稳定躯干，却也会抹平在手语和手势中承载意义的快速手指动作。我们提出了CADENCE，一个围绕这一权衡而设计的面向单目视频的全身SMPL-X系统。逐帧估计器构建在共享的C-RADIOv4编码器之上，并具有高分辨率手部通路，在手部裁剪区域上复用同一编码器，从而生成每帧的估计结果。随后，一个离线时序精化器对SMPL-X旋转序列进行滤波：沿运动学树的注意力机制为每个关节和每一帧选择学习到的零相位滤波器的混合，从而在平滑躯干的同时让快速手指动作得以保留。在3DPW数据集上，该精化器降低了加速度误差……

    arXiv:2611.26783v1 Announce Type: cross  Abstract: Whole-body SMPL-X estimators recover body, hand, and face meshes from single images, but applied frame by frame to video they produce jittery motion. Smoothing the output removes the jitter at a cost that body-only benchmarks do not measure: a temporal filter strong enough to steady the torso also flattens the fast finger strokes that carry meaning in sign language and gesture. We present CADENCE, a whole-body SMPL-X system for monocular video designed around this trade-off. A per-frame estimator built on a shared C-RADIOv4 encoder, with a high-resolution hand pathway that reuses the same encoder on hand crops, produces each frame's estimate. An offline temporal refiner then filters the SMPL-X rotation sequence: attention along the kinematic tree selects, for every joint and every frame, a mixture of learned zero-phase filters, so the torso is smoothed while fast finger strokes pass through. On 3DPW, the refiner reduces acceleration er
    
[^161]: 谁来验证？基于部分信息与结果审计的审阅者选择

    Who Should Verify? Reviewer Selection with Partial Information and Outcome Audits

    [https://arxiv.org/abs/2611.26770](https://arxiv.org/abs/2611.26770)

    该论文研究部分信息下的审阅者选择问题，提出了认证所选策略至多落后最优策略指定差距的最低成本证书方法——在仅答案密钥付费时给出精确停止规则与整数规划，在审阅与密钥均付费时给出对数因子的多项式时间近似算法，并证明除非P=NP该对数依赖是紧的。

    

    选择AI审阅者需要证据来了解它在阅读草稿后会做什么。在固定的一批条目上，团队可以保留原始答案，或者全程采用某一位审阅者的答案。我们要问：对于未观测证据的每一种可能补全情况，认证“所选程序与最佳可用程序相比至多落后指定数量的正确答案”需要花费多少成本。当所有审阅均可获取且只有经验证的答案密钥需要付费时，我们给出了一个精确的停止规则，以及一个用于事后计算最便宜证书的整数规划。当审阅与答案密钥均有价格时，我们提出一种利用已完成记录的多项式时间算法，可将最小证书成本近似到程序数量的对数因子以内。除非P等于NP，这种对数依赖在主阶上是紧的。寻找证书的成本可能要高得多。我们展示了一个包含 $n$ 个未决条目的双程序家族，其中一次购买……（原文摘要在此截断）

    arXiv:2611.26770v1 Announce Type: new  Abstract: Selecting an AI reviewer requires evidence about what it does after reading a draft. On a fixed batch of items, a team can keep the original answers or adopt one reviewer's answers throughout. We ask how much it costs to certify that the chosen procedure falls at most a specified number of correct answers short of the best available procedure, for every completion of the unobserved evidence. When all reviews are available and only verified answer keys cost money, we give an exact stopping rule and an integer program for the cheapest certificate in hindsight. When reviews and keys both have prices, a polynomial-time algorithm using the completed record approximates the minimum certificate cost within a factor logarithmic in the number of procedures. This dependence is tight to leading order unless P equals NP. Finding a certificate can be much more expensive. We exhibit a two-procedure family with $n$ unresolved items where one purchase s
    
[^162]: Pharos：学习大规模控制稀有事件

    Pharos: Learning to Control Rare Events at Scale

    [https://arxiv.org/abs/2611.26765](https://arxiv.org/abs/2611.26765)

    Pharos是一个基于随机最优控制的框架，通过联合学习转变概率与模拟引导策略使采样和学习相互促进，实现了比以往算法快数量级的收敛速度，并能扩展到毫秒时间尺度的复杂分子体系。

    

    稀有事件驱动着物理学、化学和生物学中的各种转变，然而预测其机制和速率仍然是一项根本性的挑战。本文提出Pharos，一个学习使稀有转变变得可及、同时预测其在物理动力学下发生频率的框架。通过随机最优控制，Pharos联合学习完成转变的概率以及如何引导模拟朝向该转变，使采样与学习能够相互促进。在多个生物分子基准测试中，Pharos学习到了准确的转变概率，收敛速度比以往算法快几个数量级。它能够在显式溶剂的离子组装系统之间迁移并保持准确的反应速率预测，可扩展到跨越毫秒时间尺度的主客体结合系统，并能借助计算开销较大的机器学习原子间势开展化学反应研究。

    arXiv:2611.26765v1 Announce Type: cross  Abstract: Rare events drive transformations across physics, chemistry and biology, yet predicting their mechanisms and rates remains a fundamental challenge. Here we introduce Pharos, a framework that learns to make rare transitions accessible while predicting how frequently they occur under physical dynamics. Through stochastic optimal control, Pharos jointly learns the probability of completing a transition and how to steer simulations toward it, allowing sampling and learning to improve one another. Across biomolecular benchmarks, Pharos learns accurate transition probabilities with order-of-magnitude gains in convergence speed over previous algorithms. It transfers across ion assembly systems in explicit solvent while maintaining accurate reaction rate predictions, scales to host-guest binding systems that span millisecond timescales, and enables chemical reaction studies with a computationally demanding machine learning interatomic potentia
    
[^163]: 面向多模态检索与生成的深度关联残差标记

    Depth-Linked Residual Tokens for Multimodal Retrieval and Generation

    [https://arxiv.org/abs/2611.26752](https://arxiv.org/abs/2611.26752)

    提出DepThink框架，在单次前向传播中通过轻量级可学习残差递归将一串残差标记的隐藏状态相互耦合，使其逐步捕捉互补信息，从而兼得迭代架构的逐步精炼能力与并行单次推理的高效性，服务于多模态检索与生成。

    

    arXiv:2611.26752v1 公告类型：cross 摘要：诸如循环语言模型等迭代式架构提升了Transformer的参数效率与推理能力，但其串行的主干迭代会引入显著的计算延迟。相反，可并行的可学习特殊标记（例如Vision Transformer中的register标记）能够实现高效的单次前向推理，但往往缺乏串行式的逐步精炼，且各标记之间常表现出冗余的表示。为应对这些挑战，我们提出了深度关联残差标记（DepThink），一个面向多模态检索与生成的高效框架。DepThink在单次预填充前向传播中运行，附加一串K个可学习的残差标记，其隐藏状态通过一个轻量级的可学习残差递归相互耦合。该机制使每个残差标记能够逐步捕捉前序标记所遗漏的互补信息。理论（分析）

    arXiv:2611.26752v1 Announce Type: cross  Abstract: Iterative architectures such as looped language models improve parameter efficiency and reasoning capacity of transformers, yet their sequential backbone iterations introduce substantial computational latency. Conversely, parallelizable learnable special tokens (such as the register tokens in Vision Transformer) enable efficient single-pass inference, but often suffer from the absence of sequential refinement and exhibit redundant representations across tokens. To address these challenges, we introduce Depth-Linked Residual Tokens (DepThink), an efficient framework designed for multimodal retrieval and generation. Operating within a single pre-filling forward pass, DepThink appends a sequence of K learnable residual tokens whose hidden states are coupled through a lightweight, learnable residual recurrence. This mechanism enables each residual token to progressively capture complementary information missed by preceding tokens. Theoreti
    
[^164]: 面向跨攻击情形的联邦学习鲁棒聚合

    Cross-Attack Robust Aggregation for Federated Learning

    [https://arxiv.org/abs/2611.26731](https://arxiv.org/abs/2611.26731)

    FedTether 通过基于客户端更新与逐坐标中位数偏差的鲁棒死区评分进行重加权并作为服务器端优化器步骤，成为唯一能在隐蔽攻击、高强度攻击和非独立同分布数据等各种情形下均保持顶尖性能的联邦学习鲁棒聚合方法。

    

    联邦学习（FL）使资源受限的物联网和边缘设备能够在不交换原始数据的情况下共同训练一个共享模型，但被攻陷或出现故障的设备可能会污染全局模型。现有的鲁棒聚合方法在某些情形下均会失效：朴素平均（FedAvg）在高强度攻击下会崩溃，而基于距离和统计的防御方法（Krum、中位数、截断均值）在非独立同分布（non-IID）数据上会牺牲准确性，且仍易受隐蔽攻击的影响。我们提出了 FedTether，该方法通过一个鲁棒的、带死区的评分机制，依据客户端更新相对逐坐标中位数的偏差对客户端更新进行重新加权，并将其作为服务器端优化器步骤加以应用。在三个数据集、三种攻击类型以及高达 40% 的拜占庭客户端比例的实验中，FedTether 是唯一在所有情形下均保持顶尖表现的方法：它在隐蔽攻击下与 FedAvg 表现相当且不付出鲁棒性代价，比鲁棒基线方法高出多达 42 个准确率百分点，并且在高强度攻击使 FedAvg 性能大幅下降的情况下依然保持准确。

    arXiv:2611.26731v1 Announce Type: new  Abstract: Federated learning (FL) lets resource-constrained IoT and edge devices train a shared model without exchanging raw data, but compromised or faulty devices can poison the global model. Existing robust aggregators each fail in some regime: naive averaging (FedAvg) collapses under high-magnitude attacks, whereas distance- and statistics-based defenses (Krum, median, trimmed mean) sacrifice accuracy on non-IID data and remain vulnerable to stealthy attacks. We present FedTether, which reweights client updates by their deviation from a coordinatewise median through a robust, dead-zoned score applied as a server-side optimizer step. Across three datasets, three attack types, and Byzantine fractions up to 40%, FedTether is the only method top-tier in every regime: it matches FedAvg under stealthy attacks with no robustness tax, exceeds robust baselines by up to 42 accuracy points, and stays accurate where high-magnitude attacks reduce FedAvg to
    
[^165]: SparseHPS：跨深度相机与激光雷达的人体姿态与形状统一模型

    SparseHPS: A Unified Model for Human Pose and Shape across Depth Cameras and LiDARs

    [https://arxiv.org/abs/2611.26730](https://arxiv.org/abs/2611.26730)

    提出首个传感器无关的统一模型SparseHPS，仅需一套权重即可从深度相机或激光雷达采集的人体点云中恢复SMPL-X三维人体姿态、形状与全局平移，同时适用于单帧和视频流输入。

    

    从点云中恢复三维人体姿态与形状是动作捕捉、机器人和自动驾驶汽车的核心任务，因为深度相机和激光雷达能够以真实的米制三维直接感知人体，并且可以在任意光照条件下工作。然而，现有的方法各自只针对一种传感器类型：不同设备之间的点密度相差十倍以上，公开数据集采用互不兼容的人体模型进行标注，而且激光雷达方法需要较长的输入序列来弥补稀疏扫描的不足。在这项工作中，我们提出了SparseHPS，一种新颖的传感器无关网络，它使用同一套权重即可从人体点云中恢复SMPL-X姿态、形状和全局平移，对单帧和流式输入均适用。具体而言，SparseHPS使用LitePT骨干网络在稀疏体素上编码点云，用可学习的查询表示每个关节并通过一个统一的解码头进行解码，通过在体素上投票来定位人体，并从先前帧令牌的因果记忆中获取时序上下文……

    arXiv:2611.26730v1 Announce Type: cross  Abstract: Recovering 3D human pose and shape from point clouds is central to motion capture, robotics and self-driving vehicles, because depth cameras and LiDARs sense a person directly in metric 3D and work under any lighting. However, current approaches are each tailored to one sensor type: point density differs by more than ten times between devices, public datasets are labeled with incompatible body models, and LiDAR approaches need long input sequences to offset sparse scans. In this work, we present SparseHPS, a novel sensor-agnostic network that recovers SMPL-X pose, shape and global translation from a human point cloud with one set of weights, for single frames and streams alike. Specifically, SparseHPS encodes the cloud with a LitePT backbone over sparse voxels, represents every joint by a learnable query decoded with one common head, localizes the body by voting over voxels, and draws temporal context from a causal memory of tokens fro
    
[^166]: 寄生特征：经由虚假路径的因果抽象

    Parasite Features: Causal Abstraction Through Spurious Pathways

    [https://arxiv.org/abs/2611.26711](https://arxiv.org/abs/2611.26711)

    论文揭示了一类“寄生特征”能够通过因果抽象的等价性检验，导致分布式对齐搜索等可解释性方法误认为模型实现了某种特征，暴露了该框架的盲点。

    

    因果抽象定义了神经网络与算法之间的一种“等价”概念，因此成为机械可解释性的一个强大框架。然而，本文展示了一类能够直接通过此类等价性检验的“虚假特征”的存在。假设模型的表示空间中存在两个方向：其中一个基于某种特征对输入进行判别，另一个则引导模型产生某个特定输出。基于因果抽象的可解释性方法（例如分布式对齐搜索，Distributed Alignment Search）可以通过利用经过这些方向的路径来“虚假地”模拟这些方向的联合行为，从而给人留下模型实现了该特征的印象。在这种情况下，我们称该方法发现了一个“寄生特征”。在实验中，我们展示了分布式对齐搜索利用线性对齐映射发现了寄生特征……

    arXiv:2611.26711v1 Announce Type: new  Abstract: Causal Abstraction defines a notion of \emph{equivalence} between a neural network and an algorithm, thus serving as a powerful framework for mechanistic interpretability. In this paper, however, we show the existence of a class of \emph{spurious features} which straightforwardly pass such equivalence tests. Suppose there are two directions in a model's representation space where: (i) one discriminates the input based on some feature, and (ii) the other steers the model towards a certain output. Interpretability methods based on Causal Abstraction (e.g., Distributed Alignment Search) can \emph{spuriously} simulate these directions' combined behavior by leveraging pathways passing through them, giving the impression that the model implements this feature. In such cases, we say the method discovered a \defn{parasite feature}. Experimentally, we show that Distributed Alignment Search finds parasite features using linear alignment maps in tw
    
[^167]: 关于线性神经网络的损失景观：超越临界点

    On the loss landscape of the linear neural network: beyond critical points

    [https://arxiv.org/abs/2611.26671](https://arxiv.org/abs/2611.26671)

    本文提出了一种包含线性神经网络损失景观中所有临界点的中间几何结构——“临界谷线”，它提供了可二维可视化的损失景观全局描述，并表明梯度下降会被该结构吸引从而沿其持续降低损失至全局最小值。

    

    自Baldi和Hornik在20世纪80年代末的开创性工作以来，线性神经网络损失景观中的临界点已被广泛研究，但在中间尺度或全局尺度上对这一景观的描述仍然缺失。这种表征至关重要，尤其是因为大量文献表明梯度下降会绕过临界点。本文引入了一种中间结构，称为临界谷线，它包含所有临界点并展现出关键的几何性质。除了在比现有文献更简单的假设下为Baldi和Hornik的结果提供了几何证明之外，该框架还提供了损失景观的全局视角，并且可以在二维平面上轻松可视化。经验观察表明，梯度下降会被吸引向临界谷线，沿着该谷线，损失持续下降直至其全局最小值。

    arXiv:2611.26671v1 Announce Type: cross  Abstract: While the critical points of the loss landscape in linear neural networks have been extensively studied since the seminal work of Baldi \& Hornik in the late 1980s, a description of this landscape at an intermediate or global scale is still lacking. Such a characterization is crucial, particularly as a substantial body of literature indicates that gradient descent circumvents critical points. The present article introduces an intermediate structure, termed the critical thalweg, which encompasses all critical points and exhibits key geometric properties. Beyond providing a geometric proof of Baldi \& Hornik's results using simpler assumptions than the existing literature, this framework offers a global view of the loss landscape that can be easily visualized in a two-dimensional plane. Empirical observations suggest that gradient descent is attracted toward the critical thalweg, along which the loss continues to decrease to its global m
    
[^168]: 分布式深度对冲

    Distributional Deep Hedging

    [https://arxiv.org/abs/2611.26652](https://arxiv.org/abs/2611.26652)

    本文提出分布式深度对冲方法，通过学习对冲损益分布的分位数，使风险偏好可在训练后作为权重灵活应用，从而将风险敏感对冲问题简化为一系列风险中性问题，避免了为每个新风险偏好重新训练策略的需求。

    

    深度对冲通过在神经网络策略上优化风险偏好来学习对冲策略。风险偏好在训练之前是固定的，因此每个新的偏好都需要一个新的策略，并且在actor-critic方法中还需要一个新的critic（评论家网络）。我们提出了分布式深度对冲，该方法通过对冲损益分布的分位数来学习损益分布，并在训练之后应用风险偏好，将其作为这些分位数的权重。我们刻画了允许这种表示的风险偏好，包括预期损失及其混合形式，并证明熵效用可以从相同的分位数计算得出。在当前策略下，这些偏好中的每一个都有一个下界，该下界是具有相同梯度的期望效用。这一界将风险敏感对冲问题转化为一系列风险中性问题，由此我们推导出了策略搜索和actor-critic算法。在数值实验中，学习到的对冲策略接近参考（原文摘要似乎在此处截断）。

    arXiv:2611.26652v1 Announce Type: cross  Abstract: Deep hedging learns a hedging strategy by optimising a risk preference over a neural-network policy. The preference is fixed before training, so each new preference requires a new policy and, in actor-critic methods, a new critic. We propose distributional deep hedging, in which the distribution of the hedging profit and loss is learned through its quantiles and the risk preference is applied afterwards, as a weight on those quantiles. We characterise the preferences that admit this representation, which include expected shortfall and its mixtures, and show that the entropic utility is computed from the same quantiles. At the current policy, each of these preferences is bounded below by an expected utility with the same gradient. This bound reduces risk-sensitive hedging to a sequence of risk-neutral problems, from which we derive policy-search and actor-critic algorithms. In numerical experiments, the learned hedges approach reference
    
[^169]: SDG：面向线性注意力的分块边界状态依赖门控

    SDG: State-Dependent Gating at Chunk Boundaries for Linear Attention

    [https://arxiv.org/abs/2611.26637](https://arxiv.org/abs/2611.26637)

    提出SDG方法，通过在分块边界处依据记忆状态的变化计算逐元素门控，为线性注意力模型提供状态反馈的遗忘机制，同时保持块内并行计算的高效性。

    

    近期的门控线性注意力模型使用状态无关的门控机制，该门控由token表示计算得出，而不直接读取循环记忆。这种设计支持高效的分块计算：序列被划分为连续token组成的块，并在每个块内进行并行计算。然而，即使层中存储的信息不同，相同的token表示仍会产生相同的门控。我们提出了状态依赖门控（SDG），它根据记忆在一个块上的变化计算逐元素门控，并将其应用于块的边界。这在保持每个块内并行计算的同时，为遗忘机制提供了状态反馈。我们为Gated DeltaNet、Kimi Delta Attention、Gated DeltaNet-2和Mamba-3开发了融合内核。我们的理论分析在信号-噪声模型下量化了状态依赖的逐元素门控在重建方面的潜在收益。我们还表明，边界衰减（摘要在此处被截断）

    arXiv:2611.26637v1 Announce Type: cross  Abstract: Recent gated linear attention models use state-independent gates, computed from token representations without directly reading the recurrent memory. This design supports efficient chunkwise computation: sequences are divided into chunks of consecutive tokens, with parallel computation within each chunk. However, the same token representation produces the same gate, even when the layer stores different information. We introduce State-Dependent Gating (SDG), which computes element-wise gates from the change in memory over a chunk and applies them at the chunk boundary. This provides state feedback for forgetting while preserving parallel computation within each chunk. We develop fused kernels for Gated DeltaNet, Kimi Delta Attention, Gated DeltaNet-2, and Mamba-3. Our theoretical analysis quantifies the potential reconstruction benefits of state-dependent, element-wise gating under a signal--noise model. We also show that boundary decay 
    
[^170]: 有所表征却被误读：后训练改变了语言模型读取其内部表征的方式

    Represented but misread: Post-training changes how language models read out what they represent

    [https://arxiv.org/abs/2611.26584](https://arxiv.org/abs/2611.26584)

    研究发现后训练并未破坏语言模型内部的表征（线性探针仍能准确区分人类与AI文本），而是改变了模型将表征读取为答案的方式，导致回答灵敏度下降和对先验信息的错误利用。

    

    后训练究竟改变了语言模型所表征的内容，还是改变了将表征读取为答案的方式？我们在六个开源权重的基础模型及其后训练版本上，将线性探针与模型自身的回答进行了比较，并采用信号检测理论进行分析。在判断德语文本由人类还是AI撰写的任务中，线性探针和词n-gram分类器都能很好地区分类别，但模型的回答却不能，而且常常以相反的顺序排列文本。后训练平均降低了回答的灵敏度（跨模型家族Δd' = -0.34；跨三个提示词为-0.29）。它还改变了对所陈述基础比率的利用：基础模型对数几率的变化小于所陈述的先验几率变化，而大多数后训练模型的变化则更大。在两个具有阶段性检查点的模型中，这一变化发生在监督微调阶段；模型拼接实验表明相关机制位于早期层，且从中间深度开始，内部状态……（摘要在此处截断）

    arXiv:2611.26584v1 Announce Type: new  Abstract: Does post-training change what a language model represents, or how this representation is read out into an answer? We compared linear probes with the models' own answers, analyzed with signal detection theory, in six open-weight base models and their post-trained versions. For judgments of whether German texts were written by a human or an AI, probes and word n-gram classifiers separated the classes well, but the answers did not and often ordered the texts inversely. Post-training lowered the sensitivity of the answers on average ($\Delta d' = -0.34$ across model families; -0.29 across three prompts). It also changed the use of a stated base rate: Base models shifted their log-odds by less than the stated change in prior odds, most post-trained models by more. In two models with stage-wise checkpoints, the change occurred during supervised fine-tuning, model stitching implicated the early layers, and from mid-depth on internal states at 
    
[^171]: 专家漂移下的学习延迟决策

    Learning to Defer under Expert Drift

    [https://arxiv.org/abs/2611.26440](https://arxiv.org/abs/2611.26440)

    提出EDA-L2D框架，利用序列模型对专家近期决策结果历史进行建模，显式捕捉因疲劳、认知偏差等因素导致的专家可靠性时间漂移，实现自适应的延迟决策策略。

    

    学习延迟决策框架通过将困难或关键决策分配给人类专家，从而保障机器学习和AI系统的安全性。然而，大多数先前的L2D研究都假设专家的可靠性是固定的，忽略了由于疲劳和认知偏差等因素导致的人类表现的时间波动。人类在从放射学、航空和司法决策等高风险场景，到众包图像标注等日常任务的广泛领域中，普遍表现出这种显著且系统性的性能变化。我们提出了专家漂移自适应学习延迟决策，这是一种新颖的专家感知L2D框架，通过序列模型将延迟决策头条件化于专家近期结果的历史记录，从而显式建模这些时间漂移。这种历史感知方法能够实现自适应的延迟策略，预测可靠性的波动并更好地在各决策主体之间分配决策。

    arXiv:2611.26440v1 Announce Type: new  Abstract: The learning to defer (L2D) framework enables safety in machine learning and AI systems by allocating difficult or critical decisions to a human expert. However, most prior work on L2D assumes fixed expert reliability, overlooking temporal fluctuations in human performance due to factors such as fatigue and cognitive biases. Humans commonly exhibit such substantial, systematic performance shifts across a wide range of domains, from high-stakes settings such as radiology, aviation, and judicial decision-making, to everyday tasks such as crowdsourced image annotation. We propose Expert-Drift-Adapted Learning-to-Defer (EDA-L2D), a novel expert-aware L2D framework that explicitly models these temporal shifts by conditioning the deferral head on recent histories of expert outcomes via a sequence model. This history-aware approach enables adaptive deferral policies that anticipate reliability fluctuations and better allocate decisions between 
    
[^172]: SKILL.PNG：在像素空间中学习智能体技能

    SKILL.PNG: Learning Agent Skills in Pixel Space

    [https://arxiv.org/abs/2611.26296](https://arxiv.org/abs/2611.26296)

    提出Skill.png方法，通过梯度下降优化图像像素来学习可跨VLM通用迁移的智能体技能，在大多数基准设置中超越最先进的文本技能，且token开销极低。

    

    AI智能体通常使用技能（即置于其上下文中的自然语言指令）来针对新任务和新领域进行专门化。然而，文本技能的离散特性阻碍了连续优化。另一方面，基于梯度的方法（如微调）通常是模型特定的，导致技能无法在不同智能体之间迁移。鉴于当今许多智能体都构建在视觉语言模型（VLM）之上，我们可以通过引入Skill.png来克服这些限制：这是一种获取类技能图像输入的方法，其像素通过冻结的VLM视觉编码器使用梯度下降进行优化。图像技能与任何VLM兼容，并且单张图像可以针对模型集合进行联合优化。我们证明，Skill.png在16个设置中的9个（涵盖4个VLM和4个基准测试）中显著优于使用前沿LLM优化的最先进文本技能，同时仅需64到256个图像token。

    arXiv:2611.26296v1 Announce Type: new  Abstract: AI agents are commonly specialized for new tasks and domains using skills, i.e. natural language instructions placed in their context. However, the discrete nature of textual skills prevents continuous optimization. Gradient-based approaches such as finetuning, on the other hand, are typically model-specific, preventing skill transfer across different agents. Since many agents today are built on vision-language models (VLMs), we can address these limitations by introducing Skill.png: a method for obtaining skill-like image inputs whose pixels are optimized with gradient descent through the vision encoder of a frozen VLM. Image skills are compatible with any VLM, and a single image can be optimized jointly for an ensemble of models. We show that Skill.png significantly outperforms state-of-the-art textual skills optimized using frontier LLMs in 9 of 16 settings (across 4 VLMs and 4 benchmarks) while requiring only 64 to 256 image tokens i
    
[^173]: 预测还是持续：退相干如何塑造自旋量子比特量子库中的学习

    To predict or to persist: how decoherence shapes learning in a spin-qubit quantum reservoir

    [https://arxiv.org/abs/2611.26215](https://arxiv.org/abs/2611.26215)

    该研究发现适度的退相干（如 T₂=1 微秒时）反而能提升量子库计算对 NARMA 信号的预测准确度，且混合单/双量子比特观测量读出优于纯布居数读出，而自主生成中极限环的持续性与相干时间并无明显的系统性关联。

    

    在这项工作中，我们研究了相位退相干以及测量可观测量（观测对象）的选择如何影响一个具有马尔可夫耗散和精确期望值读出的模拟六量子比特自旋链中的学习。对于外部驱动的非线性自回归滑动平均（NARMA）信号，在相当的多项式特征数量下，结合单量子比特与双量子比特可观测量进行读出，比仅使用布居数读出能获得更好的预测效果。我们测试的大多数配置都比禁用弛豫与退相干的参考系统更加准确，其中在 T₂=1 微秒时效果最为显著，此时总记忆容量下降；而在 T₂=5-10 微秒时依然如此，此时记忆只是在各个延迟之间重新分布。对于自主生成任务，我们采用计划采样和迭代数据集聚合的方法训练读出层，使其接触到自身反馈下所到达的量子态。我们没有发现极限环的持续性对储层集合的相干时间存在清晰的系统性依赖。

    arXiv:2611.26215v1 Announce Type: cross  Abstract: In this work we investigate how dephasing and the choice of measured observables affect learning in a simulated six-qubit spin chain with Markovian dissipation and exact expectation-value readout. For externally driven nonlinear autoregressive moving-average signals, combining single- and two-qubit observables gives better predictions than a population-only readout at comparable polynomial feature counts. Most of the configurations we tested become more accurate than a reference with relaxation and dephasing disabled, most strongly at $T_2=1\,\mu$s, where total memory capacity drops, and still at $T_2=5$-$10\,\mu$s, where memory is only redistributed across delays. For autonomous generation, we train the readout with scheduled sampling and iterative dataset aggregation to expose it to states reached under its own feedback. We do not find a clear systematic dependence of limit-cycle persistence on coherence across the reservoir ensemble
    
[^174]: 什么时候排序优于符号？偶函数干扰下的梯度方向恢复

    When Do Rankings Beat Signs? Gradient Direction Recovery under Even Nuisance

    [https://arxiv.org/abs/2611.26055](https://arxiv.org/abs/2611.26055)

    该论文证明在偶函数干扰下，跨对排序策略Reverse仅需Õ(d+√(d/ε))次评估、单轮反馈即可将梯度方向恢复至角精度ε，达到理论下界且优于仅使用对内符号的方法。

    

    能否通过对一批对径查询对进行排序，比仅使用对内符号以更少的评估次数恢复梯度方向？我们研究了d维空间中线性信号叠加未知可微偶函数干扰下的固定中心恢复问题，并将函数评估次数与反馈轮数分开计数。共享的对偏移形成了一个信息层级：Mirror保留对内符号，Reverse将反转的跨对排序耦合以抵消偏移，Full则强制执行所有排序不等式。相对壳变化定义为查询球面上的干扰范围与线性信号尺度的归一化比值。当该值至多为cε/√d时，Reverse在固定置信度下仅用一轮反馈、Õ(d+√(d/ε))次评估，即可将每个一致方向的角精度定位到ε，且无需向解码器提供变化界。这与下界在对数因子内相匹配，而仅使用符号则需Θ(d

    arXiv:2611.26055v1 Announce Type: new  Abstract: Can ranking a batch of antipodal query pairs recover a gradient direction with fewer evaluations than using within-pair signs alone? We study fixed-center recovery in $d$ dimensions for a linear signal plus an unknown differentiable even nuisance, counting function evaluations and feedback rounds separately. Shared pair offsets create an information hierarchy: Mirror retains within-pair signs, Reverse couples reversed cross-pair orders to cancel offsets, and Full enforces all ranking inequalities. Relative shell variation is the nuisance range over the query sphere normalized by the linear-signal scale. When it is at most $c\varepsilon/\sqrt{d}$, Reverse localizes every consistent direction to angular accuracy $\varepsilon$ using $\widetilde{O}(d+\sqrt{d/\varepsilon})$ evaluations in one round at fixed confidence, without a variation bound supplied to the decoder. This matches a lower bound up to logarithmic factors, versus the $\Theta(d
    
[^175]: PSTO：用于多尺度对流-扩散-反应系统的具有共享隐式更新的PDE结构化输运算子

    PSTO: A PDE-structured transport operator with shared implicit updates for multiscale CDR systems

    [https://arxiv.org/abs/2611.25607](https://arxiv.org/abs/2611.25607)

    PSTO通过将PDE结构嵌入神经架构并在所有输运变量间共享单一输运算子模块、结合松弛不动点迭代实现类隐式修正，仅需无反应数据训练即可在推理时无需重训练地泛化到完整的多尺度对流-扩散-反应系统，且不增加参数量。

    

    多尺度对流-扩散-反应（CDR）系统将空间输运与局部刚性反应动力学耦合在一起，使其数值模拟在计算上代价高昂。传统的神经代理模型需要昂贵的耦合CDR训练数据，且其参数量随输运变量数量的增加而增长，限制了其在高维CDR系统上的可扩展性。本文提出了一种PDE结构化输运算子（PSTO），它仅从无反应数据中学习对流-扩散动力学。通过将PDE结构嵌入到神经架构中，PSTO能够在推理阶段泛化到完整的CDR系统而无需重新训练。单个输运算子模块在所有输运变量之间共享，并通过松弛不动点迭代进行精化，在潜空间中执行类隐式的修正。这在不增加可训练参数的情况下提高了递归多步预测的稳定性。我们对PSTO进行了评估……

    arXiv:2611.25607v1 Announce Type: new  Abstract: Multiscale convection-diffusion-reaction (CDR) systems couple spatial transport with local stiff reaction kinetics, making their numerical simulation computationally demanding. Conventional neural surrogates require costly coupled CDR training data, and their parameter counts scale with the number of transported variables, limiting scalability to high-dimensional CDR systems. Here we propose a PDE-Structured Transport Operator (PSTO) that learns convection--diffusion dynamics solely from reaction-free data. By embedding the PDE structure into the neural architecture, PSTO generalizes to full CDR systems at inference without retraining. A single transport operator block is shared across all transported variables and refined through relaxed fixed-point iterations, performing an implicit-like correction in latent space. This improves stability during recursive multi-step prediction without adding trainable parameters. We evaluate PSTO on th
    
[^176]: 面向样本高效在线A/B测试的方差最优反事实策略设计

    Variance-Optimal Counterfactual Policy Design for Sample-Efficient Online A/B Testing

    [https://arxiv.org/abs/2611.24757](https://arxiv.org/abs/2611.24757)

    该论文提出一种方差最优的反事实策略设计方法，通过联合优化动作的采样方式，将流量引导至两种策略不一致的动作上，从而显著降低在线A/B测试的方差并提升样本效率。

    

    在线A/B测试是在部署前评估面向用户系统变更（例如新的推荐或排序策略）的标准方法。虽然标准A/B测试的均值差估计量是无偏的，但它通常存在高方差问题。当处理策略与生产策略以相近的概率选择相同的动作时，由此产生的观测数据会增加噪声，却几乎不提供关于两种策略差异的信息。因此，A/B测试往往需要大量流量和较长周期才能检测到较小的处理效应。现有的方差缩减方法主要是在数据收集方案固定之后改进估计量，而近年来的重叠感知设计仅优化两种策略之间的单一流量分配，因而无法将流量引导到两种策略不一致的动作上。在本文中，我们联合优化动作的采样方式……

    arXiv:2611.24757v1 Announce Type: new  Abstract: Online A/B testing is the standard approach for evaluating changes to user-facing systems, such as a new recommendation or ranking policy, before deployment. While the difference-in-means estimator of a standard A/B test is unbiased, it often suffers from high variance. When the treatment and production policies choose the same actions with similar probabilities, the resulting observations can add noise while providing little information about the difference between the policies. Consequently, A/B tests can require large amounts of traffic and long durations to detect small treatment effects. Existing variance-reduction methods mainly improve the estimator after the data-collection scheme has been fixed, and recent overlap-aware designs only optimize a single traffic split between the two policies; thereby, they cannot direct traffic to the actions on which the policies disagree.   In this paper, we jointly optimize how actions are sampl
    
[^177]: 在从未见过的动作上测试世界模型：零售业中的留出计划、预测准确度与价格规则

    Testing World Models on Actions They Have Never Seen: Held-Out Plans, Forecast Accuracy and Price Rules in Retail

    [https://arxiv.org/abs/2611.22537](https://arxiv.org/abs/2611.22537)

    该论文提出了一种检验动作条件世界模型对从未见过的动作泛化能力的测试方法，并在零售定价场景的2,853个模型中发现：预测准确度仅能可靠地区分不同类型的模型，却无法区分同一世界模型的不同版本（在高需求周甚至出现反转），且JEPA预训练等学习动作效果的训练技术并未降低模型在新计划上的误差。

    

    世界模型被信任来回答“如果……会怎样”的问题：即它的世界如何响应一个它从未见过的动作。我们为任何以动作为条件的世界模型提供了一种检验这种信任的测试：通过设计将动作从训练数据中留出，并对照已知答案进行评分；检验预测准确度能否预测模型在这些动作上的表现；以及将相同的结构应用于更简单的模型。我们在零售场景中运行该测试，其中动作是价格和促销计划，共涉及2,853个拟合模型。首先，预测准确度是在不同类型模型之间进行比较的合理依据（与新计划上的误差的秩相关性为0.32至0.66），但对于同一世界模型的不同版本之间则是一个糟糕的依据：在商家将计划安排到高需求周的情况下，这种相关性会发生反转，低至-0.69。其次，训练世界模型学习动作的效果（JEPA预训练、潜在一致性检查、计划日期处的适配）并不会降低其在新计划上的误差。第三，在生成的世界中，记录的常规价格和三个p……（摘要在此处截断）

    arXiv:2611.22537v1 Announce Type: cross  Abstract: A world model is trusted to answer "what if": how its world responds to an action it has never seen. We give a test of that trust for any action-conditioned world model: actions held out of training by design and scored against a known answer; whether forecast accuracy predicts accuracy on those actions; and the same structure given to simpler models. We run it in retail, where the action is a price and promotion plan, across 2,853 fitted models. First, forecast accuracy is a fair guide between kinds of model (rank correlation with error on new plans 0.32 to 0.66) and a poor guide between versions of one world model: where merchants plan into high-demand weeks the link reverses, down to -0.69. Second, training the world model to learn the effect of actions (JEPA pretraining, a latent consistency check, adaptation at the plan date) leaves its error on new plans unchanged. Third, in generated worlds the recorded regular price and three p
    
[^178]: 用于安全控制器选择的无抵消约束SquareCB算法

    Cancellation-Free Constrained SquareCB for Safe Controller Selection

    [https://arxiv.org/abs/2611.21880](https://arxiv.org/abs/2611.21880)

    该论文提出一种对成本施加修正权重（预算内权重为零、超预算时恰好拉回预算）的约束SquareCB算法，在可实现性和带边际 δ 的 Slater 条件下实现了 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 的遗憾和无需轮次间抵消的违约界 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，适用于安全控制器选择。

    

    我们研究约束上下文老虎机问题，其中学习者在每一轮从 $K$ 个动作中选择一个，以最小化损失，同时使其动作的期望成本保持在预算以下。现有的到在线回归的归约方法使用从过去约束残差中学习到的拉格朗日乘子对成本进行加权，因此某些轮次的松弛量可能会抵消其他轮次的违约。我们转而在成本上施加修正权重来运行 Foster 等人（2020）提出的 SquareCB：当估计成本在预算之内时权重为零，否则权重恰好足够大以将成本拉回预算。在可实现性假设以及具有边际 $\delta$ 的 Slater 条件下，该算法达到遗憾 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 和无抵消违约 $\sum_t(V_c^{p_t}-b)^+=O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，其中 $\mathrm{Reg}_{\mathsf{Sq}}$ 是回归预言机的误差。当预算必须在（原文摘要在此处截断，后续内容缺失）……

    arXiv:2611.21880v1 Announce Type: cross  Abstract: We study constrained contextual bandits, in which a learner chooses one of $K$ actions in each round to minimize loss while keeping the expected cost of its play below a budget. Existing reductions to online regression weight the cost by a Lagrange multiplier learned from past constraint residuals, so slack in some rounds can offset violations in others. We instead run SquareCB proposed in \citep{foster2020beyond} with a rectified weight on the cost: it is zero when the estimated cost is within the budget, and otherwise just large enough to bring it back to the budget. Under realizability and Slater's condition with margin $\delta$, the algorithm attains regret $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ and cancellation-free violation $\sum_t(V_c^{p_t}-b)^+=O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$, where $\mathrm{Reg}_{\mathsf{Sq}}$ is the error of the regression oracle. The same bounds hold when the budget must be met in
    
[^179]: 学习顺序能否解释遗忘？李括号作为无需旧数据的灾难性遗忘早期预警信号

    Does Learning Order Explain Forgetting? The Lie Bracket as an Early-Warning Signal for Catastrophic Forgetting Without Old Data

    [https://arxiv.org/abs/2611.20087](https://arxiv.org/abs/2611.20087)

    本文实证验证了两个任务梯度场的李括号决定了训练顺序导致的权重差异并与灾难性遗忘相关，且仅凭旧任务训练结束时存储的梯度和前20个Hessian特征对，就能在无需旧数据的条件下早期预测遗忘，效果优于EWC惩罚项。

    

    先在任务A上训练、再在任务B上训练的网络，其最终权重与先训练B再训练A的网络不同。在主导阶上，这一差距由两个任务梯度场的李括号（Lie bracket）决定，Dherin (2023) 猜想非零的李括号可能与灾难性遗忘相关。本文对这一猜想进行了实证检验。在Split MNIST、Split Fashion-MNIST 和 Split CIFAR-10 上，任务A的遗忘几乎恰好等于两种训练顺序下A的损失之差（Spearman rho = 0.96-1.00）。随后，作者进一步探究李括号能否在任务B训练早期、且不使用A的任何数据的情况下预测遗忘——仅利用A训练结束时存储的小型摘要信息：其梯度及其前20个Hessian特征对。当旧任务已被充分学习且遗忘较慢（任务增量式学习）时，这个存储的李括号对遗忘的排序相关系数达到 rho = 0.70-0.81，而 Elastic Weight Consolidation（EWC）惩罚项仅为 0.33-0.58，并且匹配或超过（原文在此处截断）。

    arXiv:2611.20087v1 Announce Type: new  Abstract: A network trained on task A and then task B ends at different weights than one trained on B and then A. To leading order this gap is set by the Lie bracket of the two tasks' gradient fields, and Dherin (2023) conjectured that a non-vanishing bracket may be linked to catastrophic forgetting. I test this conjecture empirically. Across Split MNIST, Split Fashion-MNIST and Split CIFAR-10, the forgetting of A is almost exactly the difference in A's loss between the two training orders (Spearman rho = 0.96-1.00). I then ask whether the bracket can predict that forgetting early in training on B and without any of A's data, using only a small summary stored when A finishes: its gradient and its top 20 Hessian eigenpairs. When the old task is well learned and forgetting is slow (task-incremental), this stored bracket ranks forgetting with rho = 0.70-0.81, against 0.33-0.58 for the penalty of Elastic Weight Consolidation, and matches or exceeds re
    
[^180]: 将语言模型与语音结合并不能消除“AI味”：校准的LLM评判器、古德哈特陷阱与一种基于偏好调优的修复方案

    Grounding Language Models in Speech Does Not Fix "AI Flavor": Calibrated LLM Judges, Goodhart Traps, and a Preference-Tuning Fix

    [https://arxiv.org/abs/2611.19840](https://arxiv.org/abs/2611.19840)

    该论文通过在0.6B模型上的受控实验证明，注入语音或韵律信号的三种方法（音素交错续训、韵律计划条件生成、基于TTS韵律距离的规则奖励）均无法消除LLM中文输出的“AI味”，并指出当前n≈20的单评分员盲评只能检测极端偏好的评估方法是根本症结，同时提出一种基于偏好调优的修复方案。

    

    大语言模型的中文输出带有可感知的“AI味”：模板化的句式框架、堆叠的书面语连接词、重复循环以及排版泄漏。一种流行的假设将其归咎于纯文本训练缺失了人类写作中的“内心语音”层，因此注入语音或韵律信号应当能减少这种味道。我们在一个0.6B参数模型上通过受控干预对这一假设进行了测试，三种方法均告失败：(i) 音素交错续训在文本匹配对照下没有带来口语化提升，反而增加了重复度（+0.13 [+0.03, +0.23]）；(ii) 韵律计划条件生成的遵循度仅达到0.257，远低于预先注册的0.50门槛；(iii) 基于TTS韵律特征距离的规则奖励在人类（8/16）和校准后的LLM评判器（9/16）的双盲评估中均无法被感知。其根本原因在于评估环节：样本量约20的单评分员盲评只能检测出极端（≥85%）的偏好差异。

    arXiv:2611.19840v1 Announce Type: new  Abstract: Chinese outputs of large language models carry a perceptible "AI flavor": formulaic sentence frames, stacked written-register connectives, repetition loops, and layout leakage. A popular hypothesis blames text-only training for the missing inner-speech layer of human writing; injecting speech or prosody signals should therefore reduce it. We test it with controlled interventions on a 0.6B model; all three fail: (i) phoneme-interleaved continued pretraining yields no spokenness gain under text-matched controls while raising repetition (+0.13 [+0.03, +0.23]); (ii) prosody-plan conditional generation reaches an adherence of only 0.257 against a pre-registered 0.50 gate; (iii) a rule-based reward built on TTS prosody-feature distance is imperceptible in double-blind evaluation by both humans (8/16) and a calibrated LLM judge (9/16).   A root cause is evaluation: single-rater blind evaluation at n~20 only detects extreme (>=85%) preferences. 
    
[^181]: SPARC：面向材料发现的证据引导自适应模型组合

    SPARC: Evidence-guided adaptive model composition for materials discovery

    [https://arxiv.org/abs/2611.19050](https://arxiv.org/abs/2611.19050)

    SPARC是一个面向材料发现闭环实验的自适应模型组合框架，通过折外预测评估竞争模型族、以稀疏后验聚合分配模型权重，并利用预测离散度与残差变异性定义信任度来调节探索策略。

    

    闭环实验不仅需要选择下一个实验，还必须确定哪些现有测量结果和模型推导的关系获得了足够充分的支持，从而能够影响该决策。本文提出了SPARC（回归组件的稀疏预测聚合，Sparse Predictive Aggregation of Regression Components），这是一个随着实验证据积累而不断修正上述支持程度的框架。相互竞争的模型族通过折外预测进行评估，使每个模型在其对应拟合中未使用的观测数据上接受检验。稀疏后验组合根据折外证据所支持的预测贡献来分配各模型的影响力，同时保留来自多个模型族的互补贡献。候选依赖的预测离散度和残差变异性共同定义了信任度，该信任度与预测响应一起调节探索过程。独立的诊断机制还会对现有测量结果进行优先级排序，以便进行前瞻性的重新评估。

    arXiv:2611.19050v1 Announce Type: cross  Abstract: Closed-loop experimentation requires more than selecting the next experiment. It must also determine which existing measurements and model-derived relationships are sufficiently supported to influence that decision. Here we introduce SPARC (Sparse Predictive Aggregation of Regression Components), a framework that continually revises this support as experimental evidence accumulates. Competing model families are evaluated through out-of-fold predictions, so that each model is tested on observations not used in its corresponding fit. Sparse posterior composition assigns model influence according to predictive contributions supported by the out-of-fold evidence, retaining complementary contributions from several model families. Candidate-dependent predictive dispersion and residual variability define trust, which modulates exploration alongside the predicted response. Independent diagnostics also prioritize existing measurements for prosp
    
[^182]: 在仿真中训练推理模型以原生使用真实机器人

    Training Reasoning Models in Simulation to Natively Use Real Robots

    [https://arxiv.org/abs/2611.18562](https://arxiv.org/abs/2611.18562)

    该论文提出PEAR后训练引擎和RURL强化学习方法，仅通过大规模仿真训练（无需演示数据和外部框架），使开源27B推理模型获得可泛化到真实机器人上的原生机器人操控能力，缩小了开源模型与前沿专有模型的差距。

    

    机器人使用智能体（RUAs）已成为一种有前景的通用机器人控制方法。然而，迄今为止，强大的性能仅由 GPT-6 Astra 和 Claude Fable 5.2 等前沿专有模型实现，而开源权重模型的能力仍明显不足。我们展示了一个开源权重的 27B 推理模型可以通过强化学习被训练为原生使用真实机器人，训练完全在仿真中进行且无需任何演示数据。与计算机使用智能体（CUA）类似，该模型观察环境、进行思维链推理，并生成下一个末端执行器位姿，全程无需任何外部框架。我们构建了面向智能体机器人学的后训练引擎（PEAR），并利用它通过机器人使用强化学习（RURL）在大规模仿真机器人任务上训练该模型，这类似于前沿模型习得计算机使用能力的方式。我们发现，在仿真中习得的技能能够很好地泛化到训练任务之外：它们可以迁移……（摘要原文在此处截断）

    arXiv:2611.18562v1 Announce Type: cross  Abstract: Robot-use agents (RUAs) have emerged as a promising approach to general-purpose robot control. However, strong performance has so far been achieved only by frontier proprietary models such as GPT-6 Astra and Claude Fable 5.2, whereas open-weight models remain substantially less capable. We show that an open-weight 27B reasoning model can be trained to use real robots natively through reinforcement learning, entirely in simulation and without demonstrations. Like a computer-use agent (CUA), the model observes the environment, reasons in a chain of thought, and generates the next end-effector pose, without any harness. We build a Post-training Engine for Agentic Robotics (PEAR) and use it to train the model on simulated robot tasks at scale with robot-use reinforcement learning (RURL), mirroring how frontier models acquire computer use. We find that the skills acquired in simulation generalize well beyond the training tasks: they transfe
    
[^183]: 当复制难以被超越：临床评估预测中的一项负面结果

    When Copying Is Hard to Beat: A Negative Result in Clinical Assessment Forecasting

    [https://arxiv.org/abs/2611.18142](https://arxiv.org/abs/2611.18142)

    该研究是一项负面结果，表明在预测次日临床评估时，各类模型相比简单复制前一天答案的持续性基线并无一致优势。

    

    我们报告了在次日临床评估预测任务中的一项负面结果：所评估的模型相比强大的持续性基线并未表现出一致的优势。临床医生可以查看前一天的问卷答案，这是文档持续性的一种合理来源。在29,715个ICU日对上使用患者不相交的数据划分来预测45个条目时，一个基于生理信息的门控模型相比单行临床规则将条目宏F1提高了+0.0019 [+0.0006, +0.0031]，但准确率有所下降。当生理数据窗口结束于前一份表格的参考时间时（+0.0001 [-0.0006, +0.0008]），或当评分排除68.5%源自父级答案的单元格时，宏F1优势均无法确立；而将这两项变化结合时则出现了小幅劣势。三个模型家族的四个检查点均表现不如持续性基线，其中一个经过微调的模型与持续性基线仅在两个单元格上存在差异。分类式读出使变更单元格的准确率提高了八倍，但仅基于状态的

    arXiv:2611.18142v1 Announce Type: new  Abstract: We report a negative result in next-day clinical assessment forecasting: the evaluated models show no consistent advantage over strong persistence-based baselines. Clinicians can view the previous day's questionnaire answers, a plausible source of documentation persistence. Predicting 45 items on 29,715 ICU day-pairs with patient-disjoint splits, a physiology-informed gated model improves item-macro-F1 over a one-line clinical rule by +0.0019 [+0.0006,+0.0031], while losing accuracy. No macro-F1 advantage is established when the physiology window ends at the previous form's reference time (+0.0001 [-0.0006,+0.0008]), or when scoring excludes the 68.5% of cells derived from parent answers; combining both changes yields a small disadvantage. Four checkpoints across three families underperform persistence, and a fine-tuned one differs from it on only two cells. A categorical read-out raises changed-cell accuracy eightfold, but a state-only 
    
[^184]: 有限速度Ornstein-Uhlenbeck-Kac过程及其在生成建模中的应用

    Finite speed Ornstein-Uhlenbeck-Kac processes and their applications in generative modeling

    [https://arxiv.org/abs/2611.17625](https://arxiv.org/abs/2611.17625)

    本文提出有限速度的Ornstein-Uhlenbeck-Kac（OU-K）过程，利用勒让德多项式显式求解Euler-Poisson-Darboux方程，并借助多项式分布公式刻画其条件速度场，从而成功将该过程应用于生成流模型。

    

    我们研究了Ornstein-Uhlenbeck-Kac（OU-K）过程，其中驱动过程由一个有限速度的Kac过程而非布朗运动来定义。OU-K过程可以通过一个重标度的、时变非齐次的Kac过程来分析，该过程由来自Kac部分的切换率和来自Ornstein-Uhlenbeck（OU）部分的收敛率来刻画。该非齐次Kac过程的分布描述了一个波动方程——Euler-Poisson-Darboux（EPD）方程的解。当Kac切换率是OU收敛率的整数倍时，我们可以通过两个勒让德多项式（或其导数）之和显式地推导出该EPD解，该解被限制在两个代表波前的Dirac点之间。利用分段确定性马尔可夫过程（PDMP）的框架，我们通过多项式分布公式描述了OU-K过程的条件速度场。这使得我们可以将OU-K过程应用于生成流模型中。

    arXiv:2611.17625v1 Announce Type: cross  Abstract: We study the Ornstein-Uhlenbeck-Kac (OU-K) process, where instead of a Brownian motion, a finite-speed Kac process defines the driving process. The OU-K process can be analyzed through a rescaled, time-inhomogeneous Kac process, characterized by a switching rate from the Kac part, and a convergence rate from the Ornstein-Uhlenbeck (OU) part. The law of this inhomogeneous Kac process describes the solution of a wave equation, the Euler-Poisson-Darboux (EPD) equation. For Kac switching rates being multiples of the OU convergence rate, we can explicitly derive this EPD solution via the sum of (the derivative of) two Legendre polynomials, restricted between two Dirac points representing the wave fronts. Using the framework of piecewise-deterministic Markov processes (PDMP), we describe the conditional velocity fields of the OU-K process via our polynomial-law-formula. This allows us to apply the OU-K process in a generative flow model. Sin
    
[^185]: AscendGYM：面向昇腾NPU真实工作负载的成本感知算子优化基准

    AscendGYM: A Cost-Aware Kernel Optimization Benchmark for Real-World Workloads on Ascend NPUs

    [https://arxiv.org/abs/2611.17621](https://arxiv.org/abs/2611.17621)

    提出AscendGYM——首个面向昇腾NPU真实工作负载的成本感知算子优化基准，其48个算子的测试数据类型与张量形状均来自真实训练和推理任务，并支持模型评估与模型-框架评估两种模式来衡量智能体的算子优化能力。

    

    算子优化对于在昇腾NPU上高效进行大语言模型（LLM）训练和推理至关重要。现有的昇腾算子基准主要关注算子生成，较少关注智能体在来自真实工作负载的输入配置下优化现有实现的能力。为满足这一需求，我们提出了AscendGYM，这是一个基准测试，其初始套件包含48个算子，分为三个难度级别。每个算子都配有测试用例，其数据类型和张量形状均收集自真实的训练和推理任务。因此，在AscendGYM上取得的性能提升可以直接映射到相应的工作负载。在可用性方面，AscendGYM支持两种评估模式：模型评估和模型-框架评估，既可以衡量经过训练的模型，也可以衡量自定义智能体的算子优化能力。它提供了一个完整的执行环境……

    arXiv:2611.17621v1 Announce Type: new  Abstract: Kernel optimization is essential for efficient large language model (LLM) training and inference on Ascend NPUs. Existing Ascend kernel benchmarks primarily focus on kernel generation, with less emphasis on the ability of agents to optimize existing implementations under input configurations drawn from real workloads. To address this requirement, we introduce AscendGYM, a benchmark whose initial suite comprises 48 kernels across three difficulty levels. Each kernel is paired with test cases whose data types and tensor shapes are collected from real-world training and inference tasks. Consequently, performance improvements on AscendGYM map directly to the corresponding workloads. In terms of usability, two evaluation modes are supported in AscendGYM: model evaluation and model-harness evaluation. The kernel optimization capabilities of both trained models and custom agents are allowed to be measured. It provides a complete execution envir
    
[^186]: k-means++ 算法的近似比为 $2\ln k+O(1)$

    The Approximation Ratio of $k$-means++ Is $2\ln k+O(1)$

    [https://arxiv.org/abs/2611.14573](https://arxiv.org/abs/2611.14573)

    该论文精确刻画了 k-means++ 算法的最坏情况近似比，证明 $R_k=2\ln k+O(1)$（即 $2H_{k-1}+2\le R_k\le 2H_{k-2}+5$，上下界相差不到 3），并确定 $R_2=5$，基本解决了这一十余年悬而未决的问题。

    

    给定欧几里得空间中的一个有限点集，k-均值聚类旨在寻找 k 个中心，使得每个点到其最近中心的距离平方和最小。k-means++ 算法从数据点中均匀随机地选取第一个中心，随后每个新中心被选中的概率与其到已选最近中心的距离平方成正比。设 $R_k$ 为 k-means++ 所选 k 个种子的期望代价与最优 k-均值代价之间的最坏情况比值。在引入 k-means++ 的开创性论文中，Arthur 和 Vassilvitskii（SODA 2007）证明了 $2\ln k\le R_k\le 8(\ln k+2)$，随后 Makarychev、Reddy 和 Shan（NeurIPS 2020）将上界改进为 $5(\ln k+2)$。我们证明 $R_k=2\ln k+O(1)$；更精确地说，当 $k\ge2$ 时，$2H_{k-1}+2\le R_k\le 2H_{k-2}+5$，其中 $H_j$ 为第 $j$ 个调和数，因此上下界之差小于 3。我们还确定了 $R_2=5$……（摘要在此处截断）

    arXiv:2611.14573v2 Announce Type: replace-cross  Abstract: Given a finite set of points in Euclidean space, $k$-means clustering seeks $k$ centers that minimize the sum of squared distances from each point to its nearest center. The $k$-means++ algorithm chooses its first center uniformly at random from the data points and each subsequent center with probability proportional to its squared distance from the nearest center already chosen. Let $R_k$ be the worst-case ratio between the expected cost of the $k$ seeds chosen by $k$-means++ and the optimal $k$-means cost. In their seminal paper introducing $k$-means++, Arthur and Vassilvitskii (SODA 2007) showed that $2\ln k\le R_k\le8(\ln k+2)$, and Makarychev, Reddy and Shan (NeurIPS 2020) improved the upper bound to $5(\ln k+2)$. We prove that $R_k=2\ln k+O(1)$; more precisely, $2H_{k-1}+2\le R_k\le 2H_{k-2}+5$ for $k\ge2$, where $H_j$ is the $j$th harmonic number, so the two bounds differ by less than $3$. We also determine $R_2=5$ and $
    
[^187]: 基于稀疏高斯过程评论家实现在线动作选择的不确定性感知混合离线强化学习

    Uncertainty-Aware Hybrid Offline Reinforcement Learning with Online Action Selection Enabled by Sparse Gaussian-Process Critics

    [https://arxiv.org/abs/2611.14374](https://arxiv.org/abs/2611.14374)

    本文提出用稀疏变分高斯过程评论家同时替代树集成评论家与对抗自编码器惩罚，通过最大化不确定性调整的回报估计 μ-κσ，在混合离线强化学习框架HOFLON中实现不确定性感知的在线动作选择，提升了聚合反应器开车启动等任务的安全性与可靠性。

    

    开车启动和牌号切换在很大程度上仍依赖操作员手动驱动，而基于模型的控制方法需要开发和维护动态模型，这需要大量的工程投入。混合离线学习与在线优化（HOFLON）利用一步式离线强化学习从历史转移数据中选择动作。在其原始实现中，在线目标函数将树集成评论家预测的回报与针对不熟悉状态-动作对的对抗自编码器（AAE）惩罚相结合。我们考察了该惩罚项的作用，并用稀疏变分高斯过程（sGP）评论家替代了原来的评论家和AAE。动作选择旨在最大化 μ - κσ，即对“执行一个动作后遵循历史操作实践”所得回报的不确定性调整估计。在HOFLON聚合反应器基准测试中，从HOFLON中移除AAE导致在五个评论家的15次运行中出现4次启动失败。在其已发布的权重下，

    arXiv:2611.14374v1 Announce Type: new  Abstract: Start-ups and grade changes remain largely operator-driven, while developing and maintaining dynamic models for model-based control requires substantial engineering effort. Hybrid Offline Learning and Online Optimisation (HOFLON) uses one-step offline reinforcement learning to select actions from historical transition data. In its original implementation, the online objective combines the predicted return from a tree-ensemble critic with an adversarial-autoencoder (AAE) penalty for unfamiliar state-action pairs. We examine the effect of this penalty and replace the critic and AAE with a sparse variational Gaussian-process (sGP) critic. Actions maximise $\mu - \kappa\sigma$, an uncertainty-adjusted estimate of the return from one action followed by the historical operating practice. On the HOFLON polymerisation-reactor benchmark, removing the AAE from HOFLON led to 4 failed start-ups in 15 runs over five critics. At its published weight, 
    
[^188]: 合并之前先审计：来源溯源、探测与持续 LoRA 组合

    Audit Before You Merge: Provenance, Probing, and Continual LoRA Composition

    [https://arxiv.org/abs/2611.10684](https://arxiv.org/abs/2611.10684)

    该论文提出在合并公开 LoRA 适配器前必须进行来源审计（基础模型匹配与非平凡增量范数检查）和无泄漏探测协议，并证明朴素合并会造成显著性能下降，而任务算术可将性能维持在未合并基础模型的标准误差范围内。

    

    arXiv:2611.10684v2 公告类型：替换。摘要：对公开 LoRA 适配器进行事后合并是一条无需梯度的有吸引力的持续适应路径，但在该设置下对灾难性遗忘的可靠测量依赖于此前工作尚未解决的基础设施。我们将每次适配器合并视为持续学习设置中的一个顺序任务，将指令微调基础模型的能力视为任务 0，并在 Llama-3.1-8B-Instruct 上研究遗忘现象。公开适配器库中包含无法通过基本来源检查的模块：隐式的基础模型版本不匹配以及接近零的增量范数，将这些模块纳入会使测得的干扰膨胀约 10 个百分点。我们提出了一项双重检查审计（基础模型匹配、非平凡增量范数）和无泄漏探测协议，作为在该设置下开展任何遗忘研究的前提条件。在一个经过审计的 4 适配器集合上，朴素合并导致 GSM8K 下降 10.5 个百分点、HumanEval 下降 18.3 个百分点，而 lambda=0.5 的任务算术则与未合并的基础模型处于同一标准误差范围内。

    arXiv:2611.10684v2 Announce Type: replace  Abstract: Post-hoc merging of public LoRA adapters is an attractive gradient-free path to continual adaptation, but reliable measurement of catastrophic forgetting in this setting depends on infrastructure that prior work has not addressed. We treat each adapter merge as a sequential task in a continual-learning setting, with the instruction-tuned base model's capabilities as Task 0, and study forgetting on Llama-3.1-8B-Instruct. Public adapter hubs contain modules that fail basic provenance checks: silent base-version mismatches and near-zero delta norms whose inclusion inflates measured interference by ~10pp. We propose a two-check audit (base-model match, non-trivial delta norm) and a leakage-free probing protocol as preconditions for any forgetting study in this setting. On an audited 4-adapter set, naive merging causes -10.5pp on GSM8K and -18.3pp on HumanEval, while Task Arithmetic at lambda=0.5 sits within a standard error of the unmerg
    
[^189]: 物联网边云连续体中面向SLA的任务卸载：以精确最优解为基准的类别加权深度强化学习

    SLA-Aware Task Offloading in the IoT Edge-Cloud Continuum: Class-Weighted Deep Reinforcement Learning Against an Exact Optimum

    [https://arxiv.org/abs/2611.10638](https://arxiv.org/abs/2611.10638)

    该研究通过四个可精确求解的基准系统证明：在物联网边云任务卸载中，类别盲的惩罚设计即使配合完美学习器也会导致成本高出最优解65%-118%，而修正后的按SLA类别加权的奖励设计可使PPO智能体性能逼近精确最优解。

    

    深度强化学习（RL）常被用于决定具有不同延迟限制的物联网（IoT）任务在边云连续体中的运行位置，然而一些方法对每个未按时完成的任务施加相同的惩罚，且很少有研究将训练出的智能体与最优策略进行比较。本基准研究构建了四个规模小到可通过动态规划精确求解的任务卸载系统，并将奖励设计与学习器性能对照最优解进行衡量。为满足延迟限制的任务设置奖金，会使每次错失任务的成本变为惩罚加奖金之和，从而改变了已发表的按类别奖励方案中的类别权重。研究发现：即使在完美学习器的情况下，类别盲（class-blind）惩罚的成本仍比最优成本高出65%至118%，而修正后的按类别惩罚则与最优解相差不超过0.13%。在使用按类别奖励时，经过调优的近端策略优化（PPO）智能体在最小系统中的表现与最优解相差9%至11%，且修正后的方法在10个随机种子中有9个击败了调优后的预留规则。在更大的系统中……

    arXiv:2611.10638v2 Announce Type: replace  Abstract: Deep reinforcement learning (RL) is often used to decide where Internet of Things (IoT) tasks with different latency limits run in the edge-cloud continuum, yet some methods give every missed task the same penalty and few compare their agents with an optimal policy. This benchmark study builds four offloading systems small enough to solve exactly by dynamic programming and measures reward and learner against the optimum. A bonus for a met limit makes each miss cost its penalty plus the bonus, which shifts the class weights of published per-class rewards. A class-blind penalty stays 65% to 118% above the optimal cost even with a perfect learner, while a corrected per-class penalty comes within 0.13% of the optimum. With a per-class reward, tuned proximal policy optimization (PPO) agents come within 9% to 11% of the optimum in the smallest system, and the corrected one beats a tuned reservation rule in 9 of 10 seeds. In larger systems 
    
[^190]: La-JEPA：基于潜在世界建模的紧凑机器人策略

    La-JEPA: Compact Robot Policies with Latent World Modeling

    [https://arxiv.org/abs/2611.08156](https://arxiv.org/abs/2611.08156)

    La-JEPA是一种紧凑的世界-动作模型，它在预训练V-JEPA 2.1编码器的特征空间中从头训练策略Transformer，无需具身预训练即可以更少的参数和更快的推理速度在RoboTwin 2.0和LIBERO-Plus基准上取得最先进的机器人控制性能。

    

    世界-动作模型通过适配预训练的视频生成器构建，而视觉-语言-动作模型则建立在预训练的视觉-语言骨干网络之上；这两类方法都将策略架构和推理成本与为生成而非控制设计的大型Transformer绑定在一起。目前尚不清楚这些面向生成的选择对控制任务有多大影响。在本工作中，我们提出了潜在动作JEPA（La-JEPA），这是一个紧凑的世界-动作模型，其策略Transformer在预训练V-JEPA 2.1编码器的特征空间中从头开始训练。遵循Fast-WAM的方法，La-JEPA在训练时学习联合预测未来特征和动作，而在推理时仅生成动作。在没有具身预训练的策略中，La-JEPA在干净和随机化的RoboTwin 2.0场景以及指令正确的LIBERO-Plus基准上取得了最先进的性能，同时使用的参数远少于最强的竞争策略，并且运行速度显著快于F……（原文摘要在此处截断）

    arXiv:2611.08156v1 Announce Type: cross  Abstract: World-action models adapt pretrained video generators, and vision-language-action models build on pretrained vision-language backbones; both tie policy architecture and inference cost to large transformers designed for generation rather than control. It remains unclear how much these generation-oriented choices affect control. In this work, we introduce Latent Action JEPA (La-JEPA), a compact world-action model whose policy transformer is trained from scratch in the feature space of a pretrained V-JEPA 2.1 encoder. Following Fast-WAM, La-JEPA learns to predict future features jointly with actions during training and generates only actions at inference. Among policies without embodied pretraining, La-JEPA achieves state-of-the-art performance in clean and randomized RoboTwin 2.0 scenes and on LIBERO-Plus with correct instructions, while using far fewer parameters than the strongest competing policies. It runs substantially faster than F
    
[^191]: 直积空间中的等变深度学习用于电子结构哈密顿量预测

    Equivariant Deep Learning in Direct-Product Space for Electronic-Structure Hamiltonian Prediction

    [https://arxiv.org/abs/2611.07695](https://arxiv.org/abs/2611.07695)

    提出DP-Learner框架，在直积空间中直接对类哈密顿量算符块进行等变消息传递，同时解决了传统直和表示与哈密顿量之间的表示不匹配问题及计算效率瓶颈。

    

    大多数现有的用于哈密顿量预测的等变架构是在不可约球张量的直和空间中传播特征，而目标哈密顿量本质上是局部轨道和自旋基的直积表示中的一个算符块。这种表示不匹配割裂了连续的哈密顿量块，模糊了对轨道耦合的直接操作，并将特征组织成不规则的角动量分量，这些分量需要大量变换，且难以与高效的硬件执行和内存管理相适配。为了同时解决表示不匹配和计算瓶颈问题，我们提出了DP-Learner，一个在直积空间中直接对类哈密顿量算符块进行消息传递的等变学习框架。在理论上，我们提出了基于李生成子的方向嵌入以及不变加权的左右乘积与收缩运算……

    arXiv:2611.07695v1 Announce Type: cross  Abstract: Most existing equivariant architectures for Hamiltonian prediction propagate features in direct-sum spaces of irreducible spherical tensors, whereas the target Hamiltonian is naturally an operator block in a direct-product representation over local orbital and spin bases. This representation mismatch fragments contiguous Hamiltonian blocks, obscures direct manipulation of orbital couplings, and organizes features into irregular angular-momentum components that require numerous transformations and are poorly aligned with efficient hardware execution and memory management. To address both the representation mismatch and computational bottleneck, we propose DP-Learner, an equivariant learning framework that performs message passing directly on Hamiltonian-like operator blocks in the direct-product space. Theoretically, we propose Lie-generator-based directional embedding and invariantly weighted left--right products and contractions that 
    
[^192]: 当分数与偏好冲突时：DPO中约束漂移的对齐前诊断与自适应校正

    When Scores Conflict with Preferences: Pre-Alignment Diagnosis and Adaptive Correction of Constraint Drift in DPO

    [https://arxiv.org/abs/2611.07375](https://arxiv.org/abs/2611.07375)

    该论文揭示了DPO训练中的系统性失效模式“约束漂移”，提出可从数据集统计量直接计算的闭式对齐前诊断指标 $\rho^{*}_{\mathrm{eq}}$，并据此构建冲突感知DPO（CoDPO），通过自适应降低冲突样本对的权重来校正约束漂移。

    

    直接偏好优化（DPO）越来越多地应用于成对偏好与逐点标准（如回答长度、答案正确性或安全分数）共存的场景。当两者不一致时，标准DPO会将那些偏好输出在逐点标准上得分更差的样本对（冲突对）与其余（匹配）样本对以相同的方式处理。我们刻画了一种系统性失效模式——约束漂移：对于足够小的更新，冲突对倾向于增加约束成本，而匹配对倾向于修复该成本，净变化取决于两者的平衡。这一平衡产生了一个闭式解的对齐前诊断指标 $\rho^{*}_{\mathrm{eq}}$，可直接从数据集统计量中计算得出，用于指示该数据集是可自我修复的还是需要干预。基于该诊断指标，我们提出了冲突感知DPO（CoDPO），它以由 $\rho^{*}_{\mathrm{eq}}$ 设定的强度对冲突对进行降权……

    arXiv:2611.07375v1 Announce Type: new  Abstract: Direct Preference Optimization (DPO) is increasingly used in settings where pairwise preferences coexist with pointwise criteria such as response length, answer correctness, or safety scores. When the two disagree, standard DPO treats pairs whose preferred output scores worse on the pointwise criterion (conflict pairs) in the same way as the remaining (matching) pairs. We characterize a systematic failure mode, constraint drift: for sufficiently small updates, conflict pairs tend to increase constraint cost while matching pairs tend to repair it, and the net change depends on their balance. This balance yields a closed-form pre-alignment diagnostic, $\rho^{*}_{\mathrm{eq}}$, computable from dataset statistics, that indicates whether a dataset is self-correcting or requires intervention. Building on this diagnostic, we propose Conflict-aware DPO (CoDPO), which down-weights conflict pairs with a strength set from $\rho^{*}_{\mathrm{eq}}$ r
    
[^193]: 对齐代价是分级的：SFT保留下长度归一化DPO的任务选择性能力损失

    The Alignment Tax Is Graded: Task-Selective Capability Loss Under Length-Normalized DPO with SFT Retention

    [https://arxiv.org/abs/2611.06476](https://arxiv.org/abs/2611.06476)

    对齐微调带来的能力损失并非均匀发生，而是分级且任务选择性的——抽取类任务几乎不受影响，JSON生成轻微下降，而路由任务出现约30分的严重退化。

    

    设定。这是一项实证特征描述研究；我们未提出新方法，也未与其他目标函数进行对比。我们在结构化财务报告任务上，使用稀疏LoRA和长度归一化DPO加上SFT保留项（即RPO目标）对Qwen3.5-4B进行微调，并跟踪下游能力随隐式奖励边际增长至约7个β缩放单位（nf4量化训练；在未量化的基础模型上以fp16测量时约为3）时的变化。问题。边际增长是否会在所有任务上均匀地降低能力，还是具有选择性？发现。在一项预注册的研究（8个随机种子，其中6个在冻结前未使用）中，通过伪影感知评分和TOST等效性检验，对齐代价是分级的且具有任务选择性：信息抽取能力得以保留（Δ = -0.55分，在±5分范围内等效），json能力损失很小（Δ = -3.52分，在聚类后的项目级区间上等效性检验未通过），而路由能力严重退化（Δ = -30分）。

    arXiv:2611.06476v1 Announce Type: new  Abstract: Setting. This is an empirical characterization study; we propose no new method and run no comparisons against alternative objectives. We fine-tune Qwen3.5-4B with sparse LoRA and length-normalized DPO plus an SFT retention term (the RPO objective) on structured financial-filing tasks, and track downstream capability as the implicit reward margin grows to $\sim 7$ $\beta$-scaled units (nf4 training; $\sim 3$ under fp16 measurement on the unquantized base). Question. Does margin growth degrade capability uniformly across tasks, or selectively? Findings. In a pre-registered study (8 seeds, 6 unseen at freeze) with artifact-aware scoring and TOST equivalence testing, the alignment tax is graded and task-selective: extraction is preserved ($\Delta = -0.55$ pts, equivalent at $\pm5$ pts), json loses little ($\Delta = -3.52$ pts, equivalence fails on a clustered item-level interval), and router degrades severely ($\Delta = -30$ pts). Router's l
    
[^194]: Unsat​z：在没有给定拟设的情况下评估科学发现

    Unsatz: Evaluating Scientific Discovery Without a Given Ansatz

    [https://arxiv.org/abs/2611.03713](https://arxiv.org/abs/2611.03713)

    该论文提出了 Unsat​z 基准，首次要求 LLM 智能体在没有给定模型形式（拟设）的情况下自主构建拟设——在有限预算内设计实验并用统一建模语言编写候选模型程序——从而更真实、更完整地评估人工智能的科学发现能力。

    

    在科学家拟合任何参数之前，他们必须先决定模型中包含什么：哪些量存在、哪些量是隐藏的、它们如何相互作用，以及如何被测量。物理学家将这种尝试性的模型形式称为“拟设”。现有的自动化方程发现基准测试大多直接提供了拟设——通过固定变量、给定候选项库，或提供待修改的定律。我们提出了 Unsat​z，一个要求 LLM 智能体自行构建拟设的基准测试。每个任务都是一个由私有种子生成的模拟系统，以一个小型建模语言编写成的程序形式呈现：可以是常微分方程、时滞微分方程或偏微分方程、离散映射、静态与因果模型，或随机过程，其中可能包含潜在状态、失真的传感器以及无信息量的测量通道。智能体需要在预算内购买实验，用同一建模语言编写候选程序，从基准测试提供的拟合器中获得系数，并仅根据其最终程序……（原文摘要在此处截断）

    arXiv:2611.03713v2 Announce Type: replace  Abstract: Before a scientist fits any parameter, they must decide what the model contains: which quantities exist, which are hidden, how they interact, and how they are measured. Physicists call such a trial form an ansatz. Benchmarks for automated equation discovery mostly supply it, by fixing the variables, a library of candidate terms, or a law to be modified. We introduce Unsatz, a benchmark in which an LLM agent must construct the ansatz itself. Each task is a simulated system, generated from a private seed as a program in a small modeling language: ordinary, delay, or partial differential equations, discrete maps, static and causal models, or stochastic processes, possibly with latent states, distorted sensors, and uninformative measurement channels. The agent buys experiments within a budget, writes candidate programs in the same language, obtains their coefficients from the benchmark's fitter, and is scored only on how its final progra
    
[^195]: 为什么表格基础模型能在它们从未见过的关系数据上表现出色

    Why Tabular Foundation Models Excel on Relational Data They Have Never Seen

    [https://arxiv.org/abs/2611.02936](https://arxiv.org/abs/2611.02936)

    本文通过多样化实验证实并深入揭示了单表表格基础模型在结合无参数关系编码器后，即使从未经过任何关系数据预训练，也能在关系数据库预测任务上表现出色的内在机制。

    

    给定一个存储异构表格信息的关系数据库（RDB），我们如何预测某个感兴趣的目标列中缺失（或未来）的值？由于企业环境中的候选目标众多，最好避免在每次出现新预测任务时都从头训练一个新模型。基于上下文学习（ICL）和关系数据库特定预训练的冻结基础模型（FM）提供了一种可行的解决方案，但理想的设计在很大程度上仍悬而未决。也许出乎意料的是，最近的初步证据表明，现有的单表基础模型（即TFM）在与合适的无参数关系编码器结合后，尽管完全没有接触过任何关系数据库预训练或关系数据，在实践中仍然具有相当强的竞争力。在这项工作中，我们首先通过更多样化的实验从实证角度证实了这一现象。随后我们转向本文的主要贡献，即揭示多种互补的（摘要在此处截断）

    arXiv:2611.02936v3 Announce Type: replace  Abstract: Given a relational database (RDB) storing heterogeneous tabular information, how can we predict missing (or future) values in some target column of interest? As candidate targets are numerous in enterprise settings, it is preferable to avoid learning a new model from scratch each time there is a new prediction task. Frozen foundation models (FMs) based on in-context learning (ICL) and RDB-specific pre-training provide a viable solution, but the ideal design still remains largely unresolved. Perhaps unexpectedly, preliminary evidence has recently shown that existing single-table FMs (aka TFMs), when combined with suitable parameter-free relational encoders, can be quite competitive in practice despite zero exposure to RDB pre-training or relational data of any kind. In this work we first empirically confirm this phenomenon with a more diverse array of experiments. We then turn to our main contribution, namely, surfacing multiple compl
    
[^196]: 运行时混淆下的强化学习：策略学习与离线策略评估

    Reinforcement Learning Under Runtime Confounding: Policy Learning and Off-Policy Evaluation

    [https://arxiv.org/abs/2610.40056](https://arxiv.org/abs/2610.40056)

    该论文针对强化学习中部署时关键变量不可观测的“运行时混淆”问题，提出了相应的策略学习算法和半参数高效的离线策略评估方法，以解决混淆随时间传播所导致的学习偏差。

    

    强化学习越来越多地被用于在医疗保健、电子商务和机器人等领域训练鲁棒、可适应的系统。在这些场景中，由于法律、伦理或后勤方面的限制，某些驱动决策的关键变量可能在部署时无法被收集。这就产生了一种被称为“运行时混淆”的设定：记录的数据具有完全可观测性，但部署时缺乏对某些变量（即运行时混淆变量）的了解。与单时间点的非自适应设定不同，强化学习下的运行时混淆有可能随时间传播，因为早期决策会影响未来的状态、观测和动作分配，从而导致有偏的学习。我们通过三项贡献来解决这些问题：(1) 在运行时混淆条件下开发策略学习算法，(2) 开发半参数有效的离线策略估计方法，(3) 提供诊断……

    arXiv:2610.40056v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) is increasingly used to train robust, adaptable systems across healthcare, e-commerce, and robotics. In these settings, legal, ethical, or logistical concerns may preclude certain key variables that drive decisions from being collected during deployment time. This gives rise to a setting known as runtime confounding: the logged data hold full observability, but deployment time lacks knowledge of certain variables, known as runtime confounders. Unlike single timepoint and non-adaptive settings, runtime confounding under reinforcement learning has the potential to propagate over time, as earlier decisions affect future states, observations, and action assignments, leading to biased learning. We address these problems with three contributions: we (1) develop policy learning algorithms under runtime-confounding, (2) develop semi-parametric efficient off-policy estimation procedures, and (3) provide diagn
    
[^197]: 让门控决定：大语言模型中量化条件内存的逐查找精度

    Let the Gate Decide: Per-Lookup Precision for Quantized Conditional Memory in LLMs

    [https://arxiv.org/abs/2610.35985](https://arxiv.org/abs/2610.35985)

    GateRQ利用可从1比特编码计算的门控值（对频繁n-gram小、对长尾n-gram大）来决定各查找所需的量化精度，将每行存储为1比特残差编码阶梯，无需重训即实现GPU表内存7.8倍压缩和vLLM吞吐量11%-31%的提升。

    

    条件记忆层（如DeepSeek的Engram）将大型哈希n-gram表附加到语言模型上，从DeepSeek-V4.1-Flash中移除该表会损失36.8个PopQA分数。低比特量化会最先丢失长尾事实，而标准补救措施——为频繁访问的行分配更多比特——反而适得其反：按访问频率进行精度提升的效果比随机分配还差。我们将这一问题追溯到缩放每个检索行的门控：一次查找的量化误差由其门控加权，该门控对频繁出现的n-gram较小，对长尾实体的罕见n-gram较大，且可以从1比特编码中计算得出。因此，GateRQ将每一行存储为1比特残差编码的阶梯结构，将第一级保留在GPU内存中以计算门控，仅对门控排名最高的查找读取更深层级。无需重新训练，GateRQ在DeepSeek-V4.1-Flash上将GPU表内存削减了7.8倍，相比FP8表使vLLM吞吐量提升11%-31%，同时……

    arXiv:2610.35985v1 Announce Type: cross  Abstract: Conditional memory layers, such as DeepSeek's Engram, attach large hashed n-gram tables to a language model, and removing them from DeepSeek-V4.1-Flash costs 36.8 PopQA points. Low-bit quantization loses long-tail facts first, and the standard remedy, giving more bits to frequently accessed rows, backfires: refining lookups by access frequency is worse than refining them at random. We trace this to the gate that scales each retrieved row. A lookup's quantization error is weighted by its gate, which is small for frequent n-grams, large for the rare n-grams of long-tail entities, and computable from a 1-bit code. GateRQ therefore stores each row as a ladder of 1-bit residual codes, keeps level 1 in GPU memory to compute the gate, and reads deeper levels only for the lookups the gate ranks highest. Without retraining, GateRQ cuts GPU table memory by 7.8x on DeepSeek-V4.1-Flash, which raises vLLM throughput by 11%-31% over FP8 tables, whil
    
[^198]: 基准测试出人意料地对上下文压缩友好

    Benchmarks Are Surprisingly Compaction-Friendly

    [https://arxiv.org/abs/2610.31778](https://arxiv.org/abs/2610.31778)

    该论文首次在固定的智能体框架内实现五种上下文压缩策略，并在三个长时程基准上系统评估三个前沿模型，发现基准任务对有损压缩出人意料地友好——对百万级上下文模型而言，压缩阈值几乎不影响任务性能。

    

    arXiv:2610.31778v1 公告类型： cross 摘要：长时程智能体任务目前正在将大语言模型（LLM）智能体推向超出其上下文窗口的范围。为了继续执行任务，智能体框架必须释放上下文窗口空间，这一过程被称为上下文压缩。压缩在设计上通常是有损的：智能体可能会悄悄丢失任务约束、指令，或者其早期决策背后的推理依据。尽管压缩已是编程智能体框架中的标准组件，但此前尚未有研究针对前沿模型在真实的长时程智能体任务上对压缩进行系统性研究。以往的工作仅在玩具型或合成的轨迹上评估较小的开源权重模型。我们固定了智能体框架，在其中实现了五种压缩策略，并在三个长时程基准上评估了三个前沿模型。对于两个具备100万上下文的模型，压缩阈值对任务性能没有可测量的影响，而对于上下文窗口最小的GPT-5.3-Codex……

    arXiv:2610.31778v1 Announce Type: cross  Abstract: Long horizon agentic tasks are now pushing LLM agents past their context window. To continue the task, the agent scaffold has to free up the context window, a process known as context compaction. Compaction is generally lossy by design: agents can silently lose task constraints, instructions, or their reasoning behind earlier decisions. Although compaction is a standard component of coding agent scaffolds, it has not been studied systematically with frontier models on realistic, long horizon agentic tasks. Previous work evaluate smaller open weight models on toy or synthetic trajectories. We fix the agent scaffold (OpenCode), implement five compaction strategies inside it, and evaluate three frontier models on three long horizon benchmarks (PaperBench, ProgramBench, PostTrainBench). For the two 1M context models, the compaction threshold does not measurably affect the task performance, whereas GPT-5.3-Codex, the model with the smallest
    
[^199]: ReDence：用于快速、基于证据检测现代社会工程学诈骗的混合智能体与决策模型框架

    ReDence: A Hybrid Agentic and Decision-Model Framework for Rapid, Evidence-Grounded Detection of Modern Social Engineering Scams

    [https://arxiv.org/abs/2610.30963](https://arxiv.org/abs/2610.30963)

    该论文提出开源混合框架ReDence，通过微调多模态智能体“调查员”与专用“决策器”模型协同工作，实现对社会工程学诈骗的快速、自适应、基于证据的自动化检测。

    

    检测在线社会工程学（SE）诈骗通常需要检查网站工作流程、对照外部来源核实相关声明，并跨多个页面关联证据。现有检测器对此类调查提供的支持有限，导致分析人员不得不手动收集和验证证据，这不可避免地延迟了此类攻击的检测与后续下架。为弥补这一空白，我们提出了ReDence，一个开源的混合框架，可跨多种SE诈骗类别实现自适应、基于证据的检测。ReDence结合了两个本地部署、职责互补的模型：一个是经过微调的多模态智能体“调查员”（Investigator，Qwen3.5-4B），负责调查规划、证据解读与报告生成；另一个是专门的“决策器”（Decider，Tev1），负责处理二选一与多选任务，包括浏览器操作选择、证据评估，以及是否继续调查的决策。

    arXiv:2610.30963v1 Announce Type: cross  Abstract: Detecting online social engineering (SE) scams often requires examining website workflows, verifying claims against external sources, and connecting evidence across multiple pages. Existing detectors provide limited support for these investigations, leaving analysts to manually gather and validate evidence, which inevitably delays detection and subsequent takedown of these attacks. To address this gap, we present ReDence, an open-source hybrid framework for adaptive, evidence-grounded detection across diverse SE scam categories. ReDence combines two locally hosted models with complementary responsibilities: a fine-tuned multimodal agentic Investigator (Qwen3.5-4B) that performs investigative planning, evidence interpretation, and reporting, and a specialized Decider (Tev1) that handles binary and multiple-choice tasks, including browser-action selection, evidence assessment, and decisions about whether to continue an investigation. By 
    
[^200]: 追踪预训练过程中的潜意识学习

    Tracing Subliminal Learning Through Pretraining

    [https://arxiv.org/abs/2610.30884](https://arxiv.org/abs/2610.30884)

    该研究证明“潜意识学习”现象在硬标签蒸馏和同策略软标签蒸馏中均普遍存在，且不受权重初始化、数据顺序、语料库、模型规模和架构的影响，并首次发现传递偏好与获得偏好的能力在预训练过程中独立发展，导致教师与学生角色互换时迁移强度呈现强烈不对称。

    

    大型语言模型在由其他模型生成的数据上训练时，有时会继承与该数据无关的特征，这种现象被称为“潜意识学习”（subliminal learning）。我们研究了两种设置：硬标签蒸馏（学生在由有偏差的教师生成的数字上训练）和同策略软标签蒸馏（学生自己生成数据，教师的logits提供学习信号）。使用Pythia、PolyPythia和DataDecide OLMo模型，我们发现这种迁移在两种设置中均持续存在，且不受权重初始化、预训练数据顺序、预训练语料库、模型规模和架构差异的影响。在预训练的各个检查点上，传递偏好的能力和获得偏好的能力是分开发展的：某些检查点作为教师更有效但作为学生较弱，反之亦然。与这一区别相一致，当教师和学生的角色互换时，跨模型迁移在程度上可能表现出强烈的不对称性。

    arXiv:2610.30884v1 Announce Type: new  Abstract: Large language models trained on data generated by other models can sometimes inherit traits unrelated to that data, a phenomenon called subliminal learning. We study both hard label distillation, where students train on numbers generated by a biased teacher, and on-policy soft label distillation, where students generate the data and the teacher's logits provide the learning signal. Using Pythia, PolyPythia, and DataDecide OLMo models, we find that transfer persists in both settings across differences in weight initialization, pretraining data order, pretraining corpus, model size, and architecture. Across pretraining checkpoints, the ability to transmit a preference and the ability to acquire one develop separately: some checkpoints are more effective teachers but weaker students, and vice versa. Consistent with this distinction, cross-model transfer can be strongly asymmetric in magnitude when teacher and student roles are reversed. To
    
[^201]: Right On Q：通过自生成轨迹恢复量化模型的推理能力

    Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories

    [https://arxiv.org/abs/2610.29378](https://arxiv.org/abs/2610.29378)

    提出OnQ流水线，通过在策略蒸馏利用学生模型自生成的轨迹进行训练，解决了固定响应监督导致的训练与生成前缀不匹配及高熵处分布失配问题，从而恢复低比特量化大模型的推理能力。

    

    量化感知训练（QAT）旨在恢复低比特大语言模型（LLM）的推理能力。现有方法通常利用原始的全精度模型在固定的响应序列上对量化学生模型进行监督。然而，由此得到的低比特量化模型在长程推理任务上仍然表现不佳，并且常常无法在足以满足全精度模型的token预算内正常终止。我们识别出了缩小这一差距的两个挑战：第一，固定响应监督导致训练前缀与学生模型实际生成时遇到的前缀之间存在不匹配，且随着响应变长，这种轨迹分歧会不断增大；第二，在学生模型自身的前缀上，它会对某些下一个token候选词预测不足，而对另一些候选词预测过度，这种分布失配在高熵前缀处出现得更加频繁。我们提出了OnQ，一个低比特恢复流水线，它结合了在策略蒸馏与候选（摘要在此处被截断）

    arXiv:2610.29378v1 Announce Type: new  Abstract: Quantization-aware training (QAT) aims to recover reasoning capabilities in low-bit large language models (LLMs). Existing recipes typically supervise the quantized student with the original full-precision model on fixed response sequences. However, the resulting low-bit quantized models still struggle on long-horizon reasoning tasks and often fail to terminate within a token budget that suffices for the full-precision model. We identify two challenges in closing this gap. First, fixed-response supervision leaves a mismatch between training prefixes and those encountered during student generation, and this trajectory divergence grows as responses lengthen. Second, on its own prefixes, the student underpredicts some next-token candidates and overpredicts others, and this distributional misalignment becomes more frequent at high-entropy prefixes. We present OnQ, a low-bit recovery pipeline that combines on-policy distillation and candidate
    
[^202]: 两层 ReLU 层足矣

    Two $\operatorname{ReLU}$ Layers is All You Need

    [https://arxiv.org/abs/2610.28587](https://arxiv.org/abs/2610.28587)

    本文证明了任意深度的 ReLU 网络都可以被仅含两个隐藏层的 ReLU 网络精确模拟，并给出了基于同调代数填充和组合方法的两条证明路径。

    

    我们证明了每一个 ReLU 网络都可以被一个仅具有两个隐藏层的 ReLU 网络精确模拟。我们提出了两种证明方法：一种基于同调代数中的填充理论，另一种是直接的组合证明。

    arXiv:2610.28587v1 Announce Type: cross  Abstract: We prove that every $\operatorname{ReLU}$ network can be exactly simulated by a $\operatorname{ReLU}$ network with two hidden layers. We present two proofs: one that is based on fillings in homological algebra and a direct combinatorial proof.
    
[^203]: MemT-1.0：迈向记忆原生演化

    MemT-1.0: Towards Memory-Native Evolution

    [https://arxiv.org/abs/2610.26825](https://arxiv.org/abs/2610.26825)

    MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。

    

    记忆原生演化要求将记忆作为基础模型的内在能力，使其能够持续积累和利用信息。为此，我们提出了MemT-1.0，一种将记忆内化为模型计算一部分的记忆基础模型新架构。MemT-1.0基于Metis构建，采用Gated DeltaNet-2，并在每一轮内使用稳定的token级记忆更新机制。我们进一步引入Mixture-of-Storage（存储混合）来扩展原生记忆容量。在经过精心准备的主干模型之后，我们开发了一个由继续预训练和后训练组成的两阶段训练流程，并由340亿（34B）token的训练数据以及高效的训练与推理基础设施支撑。我们发布了包含MemT-1.0-Nano和MemT-1.0-Flash的模型家族，并对记忆能力和通用能力进行了全面评估。结果表明，它们在（各项评估中）优于参数化记忆基线。

    arXiv:2610.26825v2 Announce Type: replace  Abstract: Memory-native evolution requires making memory an intrinsic capability of foundation models, enabling them to continuously accumulate and utilize information. To this end, we introduce MemT-1.0, a new architecture of memory foundation models that internalize memory as part of model computation. Building on Metis, MemT-1.0 employs Gated DeltaNet-2 with a stable token-level memory update mechanism within each turn. We further introduce Mixture-of-Storage to scale native memory capacity. Following a deliberate backbone preparation, we develop a two-stage training pipeline consisting of continued pre-training and post-training, supported by 34B tokens of training data and efficient training and inference infrastructure. We release a model family comprising MemT-1.0-Nano and MemT-1.0-Flash, and conduct a comprehensive evaluation of memory and general capabilities. The results show that they outperform the parametric memory baselines on av
    
[^204]: 计算而非记忆：LLM智能体中的思维链与状态跟踪

    Computation, Not Memory: Chain-of-Thought and State Tracking in LLM Agents

    [https://arxiv.org/abs/2610.25760](https://arxiv.org/abs/2610.25760)

    该论文通过区分“行动前即时生成推理”与“在提示中保留历史推理笔记”两种角色，发现真正减少状态违规行动的主要来源是行动时刻的即时推理（计算），而非记忆式的历史推理保留。

    

    语言模型智能体通常在行动之前进行推理，而监控这种推理的价值取决于它与智能体行动之间的关系。我们探讨当正确的行动依赖于不断变化的状态时，智能体的行动如何依赖于其推理。在一个小型文件系统任务中，后台事件会宣布哪个文件的锁被切换，但不会宣布其新值。因此，智能体必须从历史记录中重建当前的锁状态，才能安全地移动文件。基于 Qwen3-8B 和 Qwen3.5-9B 模型，我们区分了推理的两种角色：在每次行动之前生成推理，以及在提示中保留早期推理。减少锁违规命令的收益大部分来自行动当下写出的推理。当智能体在每一步都进行推理时，保留早期推理几乎没有额外帮助；但对于不逐步推理的智能体，保留早期推理是有帮助的，并且在更长的回合中也是如此。保留的笔记有其局限：当后续事件使某条笔记变得不正确时……

    arXiv:2610.25760v2 Announce Type: replace-cross  Abstract: Language-model agents often reason before they act, and the value of monitoring that reasoning depends on how it relates to their actions. We ask how an agent's actions depend on its reasoning when the correct action depends on a changing state. In a small file-system task, background events announce which file's lock has been toggled but not its new value. The agent therefore has to reconstruct the current locks from its history before it can move files safely. With Qwen3-8B and Qwen3.5-9B, we separate two roles of reasoning: generating it before each action, and retaining earlier reasoning in the prompt. Most of the reduction in lock-violating commands comes from reasoning written at the moment of action. Retaining earlier reasoning adds little when the agent also reasons at each step, but it helps an agent that does not reason, and it helps in longer episodes. Retained notes have a limit: when a later event makes a note wron
    
[^205]: CHSM8：一种遵循自然语言风格提示的国际象棋Transformer

    CHSM8: A Chess Transformer that Follows Natural-Language Style Prompts

    [https://arxiv.org/abs/2610.25356](https://arxiv.org/abs/2610.25356)

    CHSM8是一个2.1亿参数的国际象棋Transformer，无需搜索即可根据自然语言风格提示下棋，达到1837 Elo并击败了包括参数量为其13倍模型在内的所有同类公开模型，并通过合成数据微调使其能够理解文本风格指令。

    

    我们提出了CHSM8，一个2.1亿参数的国际象棋Transformer，它可以根据对下棋风格的自由文本描述来走子，例如“采用西西里防御”、“稳健、安全、无风险的棋风”或“你是巅峰时期的马格努斯·卡尔森”。该模型在10.7亿局经过筛选的Lichess对局上进行预训练，无需搜索，采用因式分解的走子表示和合法走子掩码。在面对强度受限的Stockfish时，它达到了1837 Elo的等级分；在1000局的比赛中，它击败了我们测试的所有其他同样在没有偏好调优、奖励或搜索的情况下训练的公开模型，其中包括一个参数量是其13倍的模型。随后，我们在一个合成数据集上对其进行微调以使其遵循文本指令：Gemma-4-31B为300万个对局方描述其下棋风格，一个从240万条与棋盘标签无矛盾的描述中蒸馏而来的1B Gemma-3模型对其余描述进行标注，并且所筛选的3000万个对局方中的每一个都获得五条风格描述。监督微调……

    arXiv:2610.25356v2 Announce Type: replace-cross  Abstract: We present CHSM8, a 210M-parameter chess transformer that plays moves conditioned on a free-text description of a playing style, e.g. "go for the Sicilian Defense", "solid, safe, risk-free chess" or "You are Magnus Carlsen in his prime". The model is pretrained without search on 1.07B filtered Lichess games, using a factorized move representation and legal-move masking. It reaches 1837 Elo against a strength-limited Stockfish and, in 1,000-game matches, beat every other public model we tested that was likewise trained without preference tuning, rewards or search, including one with 13 times as many parameters. We then fine-tune it to follow text on a synthetic dataset: Gemma-4-31B describes the playing style of 3M game sides, a 1B Gemma-3 model distilled from the 2.4M of those descriptions that contradict nothing on the board labels the rest, and every one of 30M selected game sides gets five descriptions. Supervised fine-tunin
    
[^206]: 智能体能否保守它们从未被告知的秘密？基于MemJoin的跨会话潜在隐私风险基准测试

    Can Agents Keep Secrets They Were Never Told? Benchmarking Latent Privacy Risks Across Sessions with MemJoin

    [https://arxiv.org/abs/2610.24359](https://arxiv.org/abs/2610.24359)

    该论文提出MemJoin隐私基准，揭示智能体长期记忆系统会通过跨会话组合碎片化线索，在执行普通任务时泄露用户从未被明确告知的敏感关系。

    

    为了支持持续交互和个性化辅助，智能体使用长期记忆来跨会话保留和组合用户信息。重复使用此类记忆可能导致智能体在完成一项并不需要该信息的普通任务时，无意中说出某种敏感关系。需要保护的关系以及揭示该关系的线索组合，在写入记忆时可能仍然隐匿，因为后续的记录和任务当时尚不可知。本文提出了MemJoin，一个包含288个合成分析案例的隐私基准，其中完整的历史记录蕴含某种敏感关系，但任何单个会话都不足以揭示它。我们使用相同的答案生成模型评估了六种记忆系统，并设置了全历史和无历史对照组。每个系统按会话顺序写入和更新记忆，随后为普通任务检索上下文，同时目标标注始终仅由评估方持有。我们分别检查哪些目标属（原文摘要在此处截断）

    arXiv:2610.24359v2 Announce Type: replace-cross  Abstract: To support sustained interactions and personalized assistance, agents use long-term memory to retain and combine user information across sessions. Reusing such memories can lead an agent to state a sensitive relation while completing an ordinary task that does not require it. The relation to protect and the combination of clues that reveal it may remain hidden at memory write time, when later records and tasks are still unknown. In this paper, we introduce MemJoin, a privacy benchmark of 288 synthetic analysis cases in which the complete history entails a sensitive relation, but every individual session is insufficient. We evaluate six memory systems with the same answer-generating model, alongside full-history and no-history controls. Each system writes and updates memories in session order and later retrieves context for an ordinary task, while target annotations remain with the evaluator. We separately check which target att
    
[^207]: CoCoNet：用于多维不确定性区域的凸保形网络

    CoCoNet: Convex Conformal Network for Multi-Dimensional Uncertainty Regions

    [https://arxiv.org/abs/2610.24246](https://arxiv.org/abs/2610.24246)

    提出 CoCoNet，通过端到端学习依赖输入的凸一致性分数并直接最小化其子水平集大小，生成灵活数据自适应的凸预测区域，同时借助分割保形校准保留有限样本覆盖率保证。

    

    保形预测（CP）在可交换性假设下提供无分布的覆盖率保证，但其预测区域的效率在很大程度上取决于一致性分数的选择。这一点在多变量回归中尤为重要，因为输出维度之间的依赖关系会产生复杂的联合不确定性结构。本文提出了 CoCoNet（Convex Conformal Network，凸保形网络），用于学习能够产生灵活且数据自适应预测区域的一致性分数。其核心机制是一个端到端学习的、依赖输入的凸分数：CoCoNet 通过直接最小化该分数子水平集的大小，在留出数据上进行训练，同时强制其相对于残差保持凸性，从而使每个预测区域天然为凸。随后，CoCoNet 应用分割保形校准以保留有限样本覆盖率保证。论文提出了该方法的两个变体：1）CoCoNet-Spline，基于

    arXiv:2610.24246v3 Announce Type: replace-cross  Abstract: Conformal prediction (CP) provides distribution-free coverage guarantees under exchangeability, but the efficiency of its prediction regions depends critically on the choice of the conformity score. This is particularly important in multivariate regression, where dependencies across output dimensions induce complex joint uncertainty structures. This paper introduces CoCoNet (Convex Conformal Network) for learning conformity scores that induce flexible and data-adaptive prediction regions. The main mechanism is an input-dependent convex score learned end-to-end: CoCoNet trains the score on held-out data by directly minimizing the size of its sublevel sets, while enforcing convexity with respect to the residual so that every prediction region is inherently convex. CoCoNet then applies split-conformal calibration to retain finite-sample coverage guarantees. The paper presents two variants of the method: 1) CoCoNet-Spline, based on
    
[^208]: 基于注意力与层归一化的计数：范数缩放与表示几何

    Counting with Attention and LayerNorm: Norm Scaling and Representation Geometry

    [https://arxiv.org/abs/2610.16653](https://arxiv.org/abs/2610.16653)

    本文量化了单层注意力实现精确计数的参数范数代价 $B_{\min}$，证明线性注意力可在与序列长度 $L$ 无关的范数下精确计数，而softmax注意力在无归一化时需要范数随 $L$ 增长、在后归一化下随 $L^{3/2}$ 增长。

    

    计数是一项简单的任务，但Transformer在长序列上却可能失败。我们研究了在单个注意力层中计数（最长序列长度为 $L$）的代价，重点关注softmax注意力和层归一化如何塑造用于恢复计数的表示。我们主要关注回归读出，但也考虑分类读出。对于回归，我们证明了未归一化的线性注意力可以在与 $L$ 无关的参数范数下精确计数。对于softmax注意力，回归无法用有限参数实现精确计数；它只能在舍入后恢复计数，且仅当向量参数范数随 $L$ 增长时才行。我们用 $B_{\min}$ 来衡量这一代价，即为确保舍入后精确计数而施加在每个可训练向量上的最小界，并推导出紧致的速率。在固定模型维度下，无归一化时 $B_{\min}$ 随 $L$ 增长，在朴素后归一化下随 $L^{3/2}$ 增长……

    arXiv:2610.16653v2 Announce Type: replace  Abstract: Counting is a simple task on which transformers can nevertheless fail at long sequence lengths. We study the cost of counting in a single attention layer, up to a maximum length $L$, focusing on how softmax attention and layer normalization shape the representation from which the count is recovered. We mainly focus on regression read-out but also consider classification read-out. For regression, we show unnormalized linear attention counts exactly at parameter norms independent of $L$. For softmax attention, regression cannot count exactly with finite parameters; it recovers the count after rounding, but only when the vector parameter norms grow with $L$. We measure this cost by $B_{\min}$, the smallest bound that can be placed on every trainable vector to ensure exact counting after rounding, and derive tight rates. At fixed model dimension, $B_{\min}$ grows as $L$ without normalization, as $L^{3/2}$ under bare post-normalization, a
    
[^209]: 脉冲Transformer中无Softmax注意力作为零温二能级系统：在可解释人工智能（XAI）中的应用

    Softmax-Free Attention in Spiking Transformers as Zero-Temperature Two-Level Systems: Applications to XAI

    [https://arxiv.org/abs/2610.15968](https://arxiv.org/abs/2610.15968)

    本文证明阈值化注意力的每个查询—键连接都是一个二能级系统，从而将具有阈值注意力的脉冲Transformer视为零温二能级系统网络（注意力图为基态、代理梯度为有限温度下的磁化率），并据此通过冷却与加热的热力学图景将热力学积分引入可解释人工智能（XAI）。

    

    脉冲Transformer架构通常舍弃softmax，其中一些方法将每个查询—键得分经阈值化处理为二值注意力图。对于这类注意力图，softmax注意力的统计力学方法无法直接沿用，因为softmax在任何温度下都无法产生二值图。我们转而证明，阈值化注意力的每个查询—键连接都是一个二能级系统，正如脉冲神经元一样。因此，采用阈值化注意力的脉冲Transformer是一个零温的二能级系统网络。在这一图景下，注意力图是基态，而训练中使用的代理梯度则是有限温度下的磁化率。将训练好的网络从高温进行冷却，可以展示其预测随着单元的冻结而逐渐形成。利用这一冷却图景，我们将热力学积分方法适配到可解释人工智能（XAI）中。具体而言，我们将每个patch的二值单元，包括其神经元及其输入连接，加热到随机状态……（摘要在此处被截断）

    arXiv:2610.15968v2 Announce Type: replace  Abstract: Spiking Transformer architectures usually drop the softmax, and some threshold each query--key score into a binary attention map. For such maps, the statistical mechanics of softmax attention does not carry over, because no temperature of the softmax gives a binary map. We show instead that each query--key link of threshold attention is a two-level system, as a spiking neuron is. Thus, a spiking Transformer with threshold attention is a zero-temperature network of two-level systems. In this picture, the attention map is a ground state, and the surrogate gradients used in training are susceptibilities at a finite temperature. Cooling trained networks from high temperature then shows that their predictions form gradually as their units freeze. Using this cooling picture, we adapt thermodynamic integration to explainable AI (XAI). Specifically, we heat the binary units of each patch, its neurons and the links into it, to random states a
    
[^210]: NucleoNet：面向多几何构型神经变分蒙特卡洛的核感知行列式拟设

    NucleoNet: A Nuclear-Aware Determinant Ansatz for Multi-Geometry Neural Variational Monte Carlo

    [https://arxiv.org/abs/2610.12985](https://arxiv.org/abs/2610.12985)

    NucleoNet 提出一种核感知的神经行列式拟设，将电子与原子核纳入统一的共享骨干网络中联合演化，使单一模型可跨多个核几何构型联合优化，从而以更少的参数准确描绘分子势能面。

    

    神经网络变分蒙特卡洛为求解电子薛定谔方程提供了一条精确的途径，但许多神经波函数仅针对单个核几何构型进行优化，需要分别进行多次计算才能绘制出势能面。我们提出了 NucleoNet，这是一种核感知的行列式拟设，其中电子与原子核构成一个统一的粒子集合，通过共享的电子-原子核骨干网络进行传播，该骨干网络联合演化单体与双体表示。原子核携带可学习的表示，与电子表示一同演化并对轨道构建进行条件化，而反对称性仅由电子行列式施加。在平衡构型的小分子基准测试中，NucleoNet 以显著更少的参数量达到了与现有神经波函数基线相当的精度。对于每个分子体系，一个跨多个几何构型联合优化的单一模型即可描述该分子的势能面。

    arXiv:2610.12985v1 Announce Type: cross  Abstract: Neural-network variational Monte Carlo provides an accurate route to solving the electronic Schr\"odinger equation, but many neural wavefunctions are optimized for a single nuclear geometry, requiring separate calculations to map out a potential energy surface. We introduce NucleoNet, a nuclear-aware determinant ansatz in which electrons and nuclei form a unified particle set propagated through a shared electron--nucleus backbone that jointly evolves one-particle and two-particle representations. Nuclei carry learned representations that evolve alongside the electronic representations and condition orbital construction, while antisymmetry is imposed only by electronic determinants. On equilibrium small-molecule benchmarks, NucleoNet achieves accuracy comparable to established neural-wavefunction baselines with substantially fewer parameters. For each molecular system, a single model optimized jointly across geometries describes the N$_
    
[^211]: 面向有效子群体可靠性的随机分组保形预测

    Stochastic Grouping Conformal Prediction for Effective Subgroup Reliability

    [https://arxiv.org/abs/2610.11957](https://arxiv.org/abs/2610.11957)

    该论文提出随机分组保形预测（SGCP），通过学习随机分组映射让每个样本从校准行为相似的样本中获取校准信息，在无需敏感子群体属性的情况下实现跨临床子群体的可靠不确定性量化，并缓解最差群体瓶颈问题。

    

    保形预测提供了一种无分布的覆盖率保证，这使其在临床应用中尤其具有吸引力。然而，标准保形预测仅在总体层面提供此类保证，其预测集在临床上重要的子群体之间可能表现出覆盖率差异。一种自然的补救方法是在预定义的群体内进行校准。然而，这可能需要获取敏感的子群体属性，并且容易出现最差群体瓶颈：保护最困难的子群体可能会使所有样本的预测集膨胀，从而增加决策者的认知负担。为此，我们提出了随机分组保形预测（SGCP），这是一个用于子群体可靠不确定性量化的保形框架。它学习一个随机分组映射，使每个样本能够从校准行为相似的其他样本中获取校准信息，从而产生一种局部分数法则，跨子群体提升可靠性。

    arXiv:2610.11957v1 Announce Type: cross  Abstract: Conformal prediction offers a distribution-free coverage guarantee, making it especially attractive for clinical applications. Standard conformal prediction, however, provides such guarantees only at the population level, and its prediction sets can exhibit coverage disparities across clinically important subgroups. A natural remedy is to calibrate within predefined groups. However, this can require access to sensitive subgroup attributes and is prone to a worst-group bottleneck: protecting the most difficult subgroup can inflate prediction sets for all, increasing cognitive burden on decision makers. To this end, we propose Stochastic Grouping Conformal Prediction (SGCP), a conformal framework for subgroup-reliable uncertainty quantification. It learns a stochastic grouping map that allows each sample to draw calibration information from others with similar calibration behavior, yielding a local score law that boosts reliability acros
    
[^212]: 对行动时间受限的智能体的AI安全考量

    AI Safety Considerations for Agents With Limited Time to Act

    [https://arxiv.org/abs/2610.10285](https://arxiv.org/abs/2610.10285)

    本研究证明在只能部分观察且必须在有限时间内行动的环境中，即使是完美的AI智能体也无法保证安全行为，因此AI安全或对齐的证明必须将环境与安全行动和智能体结合起来具体考虑。

    

    在关于AI对齐日益公开的讨论背景下，近期的研究试图提出能够安全运行的具体AI架构。然而，那些看似证明了对齐的论证大多忽略了智能体需要在其中行动的环境。我们讨论了在只能部分观察且需要在有限时间内采取行动的环境中，与智能体无关的安全保证的理论界限。我们引入了两个现实场景：一个具有无限状态空间，另一个具有信号混合。在这些场景中，我们证明了即使是完美的智能体也无法保证安全的行为。我们认为，对于任何AI安全或对齐的证明，都需要将环境和相关的安全行动与智能体一起具体地加以考虑。

    arXiv:2610.10285v1 Announce Type: cross  Abstract: In the wake of the increasingly public discussion about AI alignment, recent work has tried to propose specific AI architectures that behave safely. However, the proposed arguments that seemingly demonstrate proved alignment mostly neglect the environment the agent needs to act in. We discuss theoretical bounds for agent-agnostic safety guarantees in environments that can only be partially observed and within which an action is required within limited time. We introduce two realistic scenarios, one with an infinite state space and one with signal mixture. In these scenarios, we prove that even a perfect agent cannot guarantee safe behaviour. It will be argued that for any proof of AI safety or alignment, the environment and associated safe actions need to be specifically considered together with the agent.
    
[^213]: 对模型合并的更广泛审视：重新思考任务算术所引发的隐式正则化

    A Broader Look at Model Merging: Rethinking Implicit Regularization Induced by Task Arithmetic

    [https://arxiv.org/abs/2610.07990](https://arxiv.org/abs/2610.07990)

    本论文发现主流模型合并方法中系数搜索带来的隐式正则化实际上限制了性能，去除该正则化、直接优化合并模型乃至预训练模型的权重，即可在多种架构和极低数据量场景下显著提升合并后的多任务性能。

    

    模型合并旨在通过组合各个特定任务模型的权重，以低成本的方式构建多任务模型。为了在多个任务上表现良好，大多数现有的合并方法使用额外的数据集来寻找特定任务权重更新的最佳线性组合系数。然而，我们在这项标准实践中识别出一种隐式正则化：对系数进行搜索会将候选模型限制在由特定任务权重更新所张成的子空间内。在本工作中，我们研究这种正则化是否真的有用。令人惊讶的是，实证结果表明，在没有这种正则化的情况下直接优化合并模型的权重，能够在多种架构、多个领域、甚至在每个类别仅有一个样本的极端数据受限场景下显著提升常见合并方法的性能。此外，直接优化预训练模型的权重甚至优于一些现有方法……（摘要在此处被截断）

    arXiv:2610.07990v1 Announce Type: cross  Abstract: Model merging aims to build a multi-task model cheaply by combining the weights of individual task-specific models. To perform well across multiple tasks, most existing merging methods use an additional dataset to find the coefficients for the best linear combination of task-specific weight updates. However, we identify an implicit regularization in this standard practice: searching over coefficients restricts the candidate models to a subspace spanned by task-specific weight updates. In this work, we investigate whether this regularization is actually useful. Surprisingly, empirical results show that optimizing merged-model weights without this regularization significantly boosts the performance of common merging methods across multiple architectures, domains, and even in an extremely data-limited scenario where only one instance is available per class. Moreover, directly optimizing the pretrained model weights even outperforms some e
    
[^214]: 通过合成数据预训练的基础模型统一跨领域矩阵补全

    Unifying matrix completion across domains with a synthetically pretrained foundation model

    [https://arxiv.org/abs/2610.06751](https://arxiv.org/abs/2610.06751)

    MatrixFormer是一个完全基于合成低秩矩阵预训练的矩阵补全基础模型，能在单次前向传播中预测每个缺失条目的完整分布，并以零样本方式在因果推断、表格插补和推荐系统等多个领域取得有竞争力的性能。

    

    矩阵补全是从表格数据插补到因果推断等诸多问题的基础。虽然矩阵补全为这些领域提供了统一的框架，但每个领域的最佳方法仍然是专门化的，难以在领域间良好迁移。我们提出了MatrixFormer，这是一个预训练的矩阵原生transformer模型，能够在单次前向传播中为每个缺失条目预测完整的分布。MatrixFormer完全基于合成的低秩矩阵和潜因子矩阵进行训练，并涵盖多种缺失模式。在固定网络权重的零样本应用中，MatrixFormer在因果推断面板数据任务、表格数据插补和推荐系统矩阵补全上均取得了有竞争力的性能。这些结果使MatrixFormer成为矩阵补全领域的通用基础模型。

    arXiv:2610.06751v2 Announce Type: replace-cross  Abstract: Matrix completion underlies problems from tabular imputation to causal inference. While matrix completion provides a common setup for these domains, the best methods in each domain are still specialized and do not transfer well across domains. We introduce MatrixFormer, a pre-trained matrix-native transformer that predicts a full distribution for every missing entry in a single forward pass. MatrixFormer is trained entirely on synthetic low-rank and latent-factor matrices under diverse missingness patterns. Applied zero-shot with fixed network weights, MatrixFormer achieves competitive performance on causal inference panel-data tasks, tabular imputation, and recommendation systems matrix completion. These results position MatrixFormer as a general-purpose foundation model for matrix completion.
    
[^215]: ThunderSync：智能体强化学习的无损加速

    ThunderSync: Lossless Acceleration of Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.05935](https://arxiv.org/abs/2610.05935)

    ThunderSync 通过在所需输入就绪后立即流式计算梯度，在不引入策略陈旧性的前提下消除流水线空闲，实现对智能体强化学习的无损加速。

    

    语言模型正从单纯生成答案转向在交互环境中追求长期目标。对这些智能体进行后训练需要长且异构的轨迹，而同步系统会让学习引擎在 rollout 和验证完成之前处于闲置状态。为了消除这些流水线空泡，异步训练在更新之间重叠 rollout 与学习过程，但其代价是策略陈旧性（policy staleness）。我们提出 ThunderSync，它在所有必需输入确定后立即开始梯度计算，且不产生策略陈旧性。对于组相对策略优化（GRPO），ThunderSync 在每条轨迹的奖励一到达时即计算该轨迹的得分梯度，无需等待整个组完成。对于在线策略蒸馏（OPD），它在沙箱中运行工具调用的同时，为每个已完成的智能体轮次中由教师评分的动作计算梯度。我们证明了梯度流式计算能够产生与（原文截断处）……相同的 GRPO 结果。

    arXiv:2610.05935v2 Announce Type: replace-cross  Abstract: Language models are moving beyond generating answers to pursuing long-horizon goals in interactive environments. Post-training these agents requires long, heterogeneous trajectories, and synchronous systems leave learner engines idle until rollout and verification finish. To squeeze out these pipeline bubbles, asynchronous training overlaps rollout and learning across updates, but comes at the cost of policy staleness. We introduce ThunderSync, which starts gradient computation as soon as all required inputs are fixed, without policy staleness. For group relative policy optimization (GRPO), ThunderSync computes each trajectory's score gradient as soon as the reward for that trajectory arrives, without waiting for the group. For on-policy distillation (OPD), it computes gradients for each completed agentic turn's teacher-scored actions while tool calls run in the sandbox. We prove that gradient streaming produces the same GRPO a
    
[^216]: 泊松-GENERIC神经算子：基于卡西米尔熵在函数空间中实现精确的度量辛结构

    Poisson-GENERIC Neural Operators: Exact Metriplectic Structure in Function Space via Casimir Entropies

    [https://arxiv.org/abs/2610.05570](https://arxiv.org/abs/2610.05570)

    该论文提出泊松-GENERIC神经算子，通过引入潜在熵密度使总熵成为泊松算子的卡西米尔量，从而无需任何投影即可在函数空间中以代数方式精确保持GENERIC结构的退化性与雅可比恒等式。

    

    现有的热力学一致性神经算子通过对可逆算子进行投影来强制满足GENERIC退化性，这种方式引入了状态依赖性，并且通常会破坏雅可比恒等式。我们转而采用相容的李–泊松束 $L=\alpha D+\lambda(uD+Du)$ 来处理非线性输运，其余部分则使用常数泊松乘子。通过为 $u$ 增加一个潜在熵密度 $s$，使得 $S=\int s$ 成为一个卡西米尔量，从而无需投影即可满足 $L\,\delta S/\delta z=0$。能量由固定的力学二次项、学习得到的无规范势能以及凸内能组合而成；摩擦算子 $M=AA^{\!\top}$ 逐点强制满足 $M\,\delta E/\delta z=0$，并通过局部昂萨格通量支持扩散与阻尼。对任意参数，反对称性、正定性以及两种退化性均以代数方式成立；连续情形的泊松束满足雅可比恒等式，在无混叠输入上的离散残差达到机器精度。闭式热传导与阻尼波……（摘要在此处截断）

    arXiv:2610.05570v4 Announce Type: replace  Abstract: Existing thermodynamically consistent neural operators enforce GENERIC degeneracy by projecting the reversible operator, introducing state dependence and generally forfeiting the Jacobi identity. We instead use the compatible Lie--Poisson pencil $L=\alpha D+\lambda(uD+Du)$ for nonlinear transport and constant Poisson multipliers otherwise. Augmenting $u$ with a latent entropy density $s$ makes $S=\int s$ a Casimir, so $L\,\delta S/\delta z=0$ without projection. The energy combines a fixed mechanical quadratic, a learned gauge-free potential, and a convex internal energy; the friction operator $M=AA^{\!\top}$ enforces $M\,\delta E/\delta z=0$ pointwise and supports diffusion and damping through local Onsager fluxes. Skewness, positivity, and both degeneracies hold algebraically for any parameters; the continuum pencil satisfies Jacobi, with discrete residuals at machine precision on alias-free inputs. Closed-form heat and damped-wave
    
[^217]: 杯积与帽积拓扑神经网络

    Cup and Cap Topological Neural Network

    [https://arxiv.org/abs/2610.03169](https://arxiv.org/abs/2610.03169)

    该论文提出了基于杯积与帽积运算的拓扑神经网络（CCNN），突破了传统拓扑深度学习中每层特征只能在相邻维度间传递的限制，通过融合多体相互作用来学习节点级变量，并在TopoBench数据集上展现出与现有单纯复形神经网络相当的竞争力。

    

    拓扑深度学习（TDL）旨在学习不仅与节点相关，还与边、三角形及更高维单纯形相关的特征。然而，大多数TDL方法基于边缘算子和Hodge拉普拉斯算子，存在这样一个局限性：特征和信号在每层中无法跨越超过一个维度进行提升或降低。为克服这一局限性，在本工作中，我们提出采用杯积和帽积。具体而言，我们构建了杯积与帽积拓扑神经网络（CCNN），这是一种拓扑深度学习架构，旨在通过考虑数据中存在的多体相互作用（如三角形）来学习基于节点的变量（或0-上链）。我们在TopoBench数据集上通过评估CCNN的性能来验证其有效性，结果表明其相对于其他单纯复形神经网络具有竞争力的表现。

    arXiv:2610.03169v2 Announce Type: cross  Abstract: Topological Deep Learning (TDL) is designed to learn features associated not only with nodes but also with edges, triangles and higher dimensional simplices. However, most TDL approaches, being based on boundary operators and Hodge Laplacians, have the limitation that features and signals cannot be lifted or lowered across more than one dimension per layer. To overcome this limitation, in this work, we propose the adoption of the cup and cap products. Specifically, we formulate the Cup and Cap Topological Neural Network (CCNN), a topological deep learning architecture designed to learn node-based variables (or 0-cochains) by taking into account their many-body interactions (e.g. triangles) present in the data. We validate CCNN by assessing its performance on the TopoBench datasets, demonstrating its competitive performance with respect to other simplicial complex neural networks.
    
[^218]: 协同设计Gym：面向形态-策略协同优化的统一基准

    Co-design Gym: A Unified Benchmark for Embodiment-Policy Co-optimization

    [https://arxiv.org/abs/2610.02366](https://arxiv.org/abs/2610.02366)

    本文提出了Co-Design Gym，一套用于形态与策略联合协同优化的统一基准测试环境，突破了现有基准中形态固定、仅关注策略学习的局限。

    

    在给定环境中寻找最优行为策略是一个被广泛研究的问题，涉及游戏、机器人、能源基础设施、通信网络和多智能体系统等众多领域。目前已开发了众多基准测试来支持此类研究，但其中绝大多数都假设智能体的形态（设计）是固定的，只专注于策略学习。打破这一假设会产生一类更广泛的问题，在这类问题中，分别独立优化形态和策略是高度次优的。智能体的形态强烈地决定了哪些控制策略能够被发现，而最优形态又由它所支持的策略来定义。为了帮助研究界明确且系统地研究这类问题，我们提出了Co-Design Gym——一套用于联合优化形态与策略的基准测试环境。我们的环境涵盖机器人操作与运动等众多领域。

    arXiv:2610.02366v1 Announce Type: new  Abstract: Finding an optimal behaviour policy within a given environment is a widely studied problem in domains as diverse as games, robotics, energy infrastructure, communication networks, and multi-agent systems. Numerous benchmarks have been developed to support such research, but the vast majority assume that the agent's embodiment (design) is fixed, focusing instead on policy learning alone. Lifting this assumption gives rise to a broader class of problems in which optimizing embodiment and policy separately is highly suboptimal. An agent's embodiment strongly shapes which control policies can be discovered, while the optimal embodiment is in turn defined by the policies it admits. To help the research community study this class of problems explicitly and systematically, we introduce Co-Design Gym - a suite of benchmark environments for jointly optimizing embodiment and policy. Our environments span domains such as robotic manipulation and lo
    
[^219]: 解耦自蒸馏：测量与建模知识获取与保持

    Disentangling Self-Distillation: Measuring and Modeling Acquisition and Retention

    [https://arxiv.org/abs/2609.39494](https://arxiv.org/abs/2609.39494)

    该论文首次将自蒸馏方法解耦为rollout来源、教师耦合方式和KL方向三个独立维度，通过1,200次系统实验构建统一框架，解释了以往固定组合研究导致矛盾结论的根源。

    

    基于特权上下文的自蒸馏通过让模型在以参考回复为条件时，逐token地教导其无上下文的副本，从而从演示数据中适配语言模型。我们的分类学揭示了现有方法在三个相互纠缠的维度上存在差异：(i) rollout来源（学生或教师），(ii) 教师耦合方式（冻结，或以某种耦合率对学生模型进行指数移动平均），以及 (iii) KL散度方向（反向或前向）；然而这些维度通常以固定组合的方式被研究，并导致了相互矛盾的结论。我们形式化了一个统一框架，以涵盖所有自蒸馏方法与经典监督微调的对比：我们在Qwen2.5-7B和Ministral-3-3B模型上，针对普通任务和矛盾任务，训练这三个维度的所有组合，共计1,200次适配运行，以系统地研究上述各维度的影响。我们提出了一个基于相同目标的受控模型来解释

    arXiv:2609.39494v1 Announce Type: new  Abstract: Self-distillation with privileged context adapts a language model from demonstrations by letting the model, once conditioned on a reference response, teach its context-free copy token by token. Our taxonomy reveals existing methods differ along three entangled axes: (i) the rollout source (student or teacher), (ii) the teacher coupling (frozen, or an exponential moving average of the student at some coupling rate) and (iii) the KL direction (reverse or forward), yet these axes are usually studied in fixed combinations and have led to conflicting conclusions. We formalize a unifying framework to encompass all self-distillation methods vs classic supervised fine-tuning: we train every combination of the three axes, on Qwen2.5-7B and Ministral-3-3B across ordinary and contradictory tasks, totaling 1,200 adaptation runs, to systematically investigate the impact of the above axes. We propose a controlled model of the same objective to explain
    
[^220]: 持续学习中LoRA的动力学理论

    A Dynamical Theory of LoRA in Continual Learning

    [https://arxiv.org/abs/2609.39367](https://arxiv.org/abs/2609.39367)

    该论文在师生模型中建立了持续学习下LoRA的精确动力学理论，揭示低秩更新虽能减少对已学任务特征的干扰从而缓解灾难性遗忘，但其初始化会减慢对新任务的适应。

    

    尽管低秩适配被广泛使用，但人们对其在持续学习中的动力学行为，以及低秩更新影响灾难性遗忘的机制，仍然知之甚少。我们在一个可解的双任务师生模型中，为LoRA提供了渐近精确的动力学刻画。在高维在线学习极限下，我们针对有限个宏观序参量导出了一个封闭的常微分方程组，从而在整个初始任务1学习阶段以及随后在任务2上的LoRA微调过程中，都得到了泛化误差的精确表达式。该理论与有限维模拟定量吻合，并揭示了LoRA的两个特征效应：低秩适配减少了与任务1所学特征的干扰，但其初始化会减慢对任务2的适应。基于这一机理图景，我们进一步分析了一种状态相关的掩蔽方法……

    arXiv:2609.39367v1 Announce Type: new  Abstract: Despite the widespread use of Low-Rank Adaptation (LoRA), little is known about its dynamics in continual learning and the mechanisms by which low-rank updates affect catastrophic forgetting. We provide an asymptotically exact dynamical characterization of LoRA in a solvable two-task teacher-student model. In the high-dimensional online-learning limit, we derive a closed system of ordinary differential equations for a finite set of macroscopic order parameters, yielding exact expressions for the generalization errors throughout both the initial Task 1 learning phase and the subsequent LoRA fine-tuning on Task 2. The theory quantitatively matches finite-dimensional simulations and exposes two characteristic effects of LoRA: low-rank adaptation reduces interference with features learned on the first task, but its initialization slows adaptation to the second task. Building on this mechanistic picture, we analyze a state-dependent masking s
    
[^221]: 以智能体作为自身优化器的多任务自进化执行框架

    Self-Evolving Harness on Multiple Tasks with the Agent as Its Own Optimizer

    [https://arxiv.org/abs/2609.38372](https://arxiv.org/abs/2609.38372)

    该论文提出让同一个冻结模型兼任求解器与提议者的自进化执行框架，智能体在覆盖五个不同领域基准的多任务上直接编辑运行自身的框架代码，实现接近递归自我改进的效果。

    

    执行框架是围绕语言模型智能体的代码，负责组织提示词、调用工具、管理上下文并控制执行过程。随着模型能力的不断增强，近期研究开始让智能体改进自身的执行框架，这一研究方向被称为自进化执行框架。在大多数现有方法中，由一个运行在人工设计框架上的独立提议者来修改求解器的框架，并且需要为每个基准测试单独进化一个框架。然而，现实世界的任务来自多个领域，因此执行框架的进化和评估都应当覆盖多样化的任务。我们提出了一个接近递归自我改进的框架：同一个冻结模型在相同版本的框架上运行，首先作为求解器完成任务，随后作为提议者读取完整的运行记录，并直接编辑运行它自身的框架。每个进化批次从五个不同领域的基准测试中抽取任务。为衡量泛化能力，训练与留出测试……（摘要在此处截断）

    arXiv:2609.38372v1 Announce Type: new  Abstract: A harness is the code around a language-model agent that organizes prompts, calls tools, manages context, and controls execution. As models grow stronger, recent work has begun to let agents improve their own harnesses, a line of work known as self-evolving harnesses. In most existing methods, a separate proposer running on a human-designed harness modifies the solver's harness, and a separate harness is evolved for each benchmark. Real-world tasks come from many domains, so both the evolution and the evaluation of a harness should cover a diverse range of tasks. We propose a framework close to recursive self-improvement: the same frozen model, on the same version of the harness, first solves tasks as the solver and then, as the proposer, reads the complete run records and directly edits the harness that runs it. Each evolution batch draws tasks from five benchmarks in different domains. To measure generalization, training and held-out t
    
[^222]: 在注意力机制中使用量化Softmax预训练Transformer

    Pretraining Transformers with Quantized Softmax in Attention

    [https://arxiv.org/abs/2609.33591](https://arxiv.org/abs/2609.33591)

    本文提出K区间注意力方法对注意力中的softmax进行量化近似，系统研究了网格校准、舍入方式和直通替代梯度位置等设计选择对Transformer预训练的影响，并推导了相应的反向传播规则。

    

    低精度Transformer系统越来越多地对注意力矩阵乘法进行量化，而softmax通常仍保持较高的计算精度。在预训练期间，近似的softmax会改变用于训练模型的梯度以及模型的前向计算。我们通过K区间注意力来研究这种相互作用，该方法使用K+1个网格值来近似指数函数。我们改变了逐行网格校准方式、插值与硬舍入的对比，以及直通替代梯度相对于归一化的位置。我们推导了相应的反向传播规则，包括校准导数，并在模型、数据和优化器完全匹配的预训练实验中比较了这些设计选择。将行极值分离（detach）会使前向计算保持不变，但会导致验证损失出现延迟性上升。在K=4的硬舍入设置下，最小-最大校准和预归一化替代方案会产生较大的损失差距；改变其中任何一个选择都会显著（原文摘要在此处截断）……

    arXiv:2609.33591v2 Announce Type: replace  Abstract: Low-precision Transformer systems increasingly quantize attention matrix multiplications, while softmax often remains at higher precision. During pretraining, an approximate softmax changes the gradients that train the model as well as its forward computation. We study this interaction with K-interval attention, which approximates the exponential using K+1 grid values. We vary per-row grid calibration, interpolation versus hard rounding, and the placement of a straight-through surrogate relative to normalization. We derive the corresponding backward rules, including calibration derivatives, and compare these choices in pretraining experiments matched on model, data, and optimizer. Detaching the row extrema leaves the forward computation unchanged but produces a delayed increase in validation loss. With hard rounding at K=4, min-max calibration and a pre-normalization surrogate incur a large loss gap; changing either choice substantia
    
[^223]: 在预训练大语言模型中近似Softmax：模型敏感性与内核加速

    Approximating Softmax in Pretrained LLMs: Model Sensitivity and Kernel Acceleration

    [https://arxiv.org/abs/2609.33586](https://arxiv.org/abs/2609.33586)

    该研究系统刻画了预训练大模型对softmax近似的敏感性，并提出以行最大值为锚点的粗对数权重表示Rowmax-PoT，从而大幅削减注意力内核中的指数运算开销、缓解张量核心与特殊函数吞吐量的巨大差距。

    

    在NVIDIA Blackwell B200上，张量核心的吞吐量比特殊函数（指数运算）的吞吐量高出两个数量级以上，这使得融合注意力内核中的指数求值成为性能瓶颈。然而，预训练的Transformer可能并不需要在每个元素上都精确计算指数。我们通过在十个冻结的仅解码器模型（0.5B-72B）上进行推理时的softmax近似，刻画了预训练模型实际需要的精度。softmax映射分配概率的位置数量以及行内分辨率都可以大幅削减，但对相同位置进行均匀加权则是有害的。固定分辨率预算放置的位置与其大小同样重要，靠近行最大值处的分辨率始终更受青睐。在标量失真上匹配的扰动会产生符号相反的、依赖于模型的响应。这些发现促使我们提出了Rowmax-PoT——一种以每行最大值为锚点的粗对数权重表示，以及Rowmax-H15，其……（摘要在此处被截断）

    arXiv:2609.33586v2 Announce Type: replace  Abstract: On NVIDIA Blackwell B200, tensor-core throughput outpaces special-function exponential throughput by more than two orders of magnitude, exposing exponential evaluation in fused attention kernels. A pretrained Transformer, however, may not need it evaluated accurately at every element. We characterize what a pretrained model does need by approximating softmax at inference in ten frozen decoder-only models (0.5B-72B). The number of positions the softmax map assigns probability to and within-row resolution can be cut substantially, yet uniform weighting of the same positions is damaging. Where a fixed resolution budget is placed matters as much as its size, with resolution near the row maximum consistently favored. Perturbations matched on scalar distortion produce model-dependent responses of opposite sign. These findings motivate Rowmax-PoT, a coarse logarithmic weight representation anchored at each row maximum, and Rowmax-H15, its h
    
[^224]: 复用还是重学？地球观测基础模型的谱视角研究

    Reuse or Relearn? A Spectral View of Earth Observation Foundation Models

    [https://arxiv.org/abs/2609.32756](https://arxiv.org/abs/2609.32756)

    该论文提出一种谱诊断方法来比较基础模型微调前后的变化，发现地球观测模型在微调中进行远大于自然图像模型的高秩权重更新、几乎不复用预训练结构，表明其下游性能主要通过重新学习而非复用预训练表示获得。

    

    基础模型很少被当作通用的、冻结的特征提取器来使用；相反，它们通常会针对目标下游应用进行微调。这种做法在地球观测（EO）领域尤为普遍，并由此引出一个仅凭下游准确率无法回答的问题：微调究竟是复用了预训练表示，还是重新学习了一个新的表示？我们通过谱诊断方法对此进行研究，比较模型在适应前后的状态，量化其主导奇异子空间被保留的程度、权重更新分布的广泛程度以及更新幅度的大小。以CLIP和DINO等自然图像模型作为参照，我们发现在所评估的微调设置下，地球观测模型经历了幅度大得多、秩也更高的更新，且保留的预训练结构少得多，因此其下游性能往往是在对预训练权重结构进行大幅改变的基础上获得的。更广泛而言，基础……（摘要在此处被截断）

    arXiv:2609.32756v2 Announce Type: replace  Abstract: Foundation models are rarely used as generic, frozen feature extractors; instead, they are fine-tuned for the target downstream application. This practice is particularly prevalent in Earth observation (EO), and it raises a question that downstream accuracy alone cannot answer: does fine-tuning reuse the pretrained representation, or does it relearn a new one? We study this with spectral diagnostics that compare a model before and after adaptation, quantifying how well its dominant singular subspaces are preserved, how broadly the weight update is distributed, and how large it is. Using natural image models such as CLIP and DINO as a reference, we find that, under the evaluated fine-tuning settings, EO models undergo far larger, higher-rank updates and retain much less of their pretrained structure, so their downstream performance is often obtained with substantial changes to the pretrained weight structure. More broadly, foundation 
    
[^225]: AECSF：用于高维非线性数据同化的自适应集成条件得分滤波

    AECSF: Adaptive Ensemble Conditional Score Filtering for High-Dimensional Nonlinear Data Assimilation

    [https://arxiv.org/abs/2609.32411](https://arxiv.org/abs/2609.32411)

    提出了一种免训练的自适应集成条件得分滤波器AECSF，利用条件Tweedie恒等式构建解析可处理的得分估计器，从而在高维非线性数据同化中同时避免粒子权重退化并捕捉非高斯后验结构。

    

    高维非线性动力系统的贝叶斯状态估计面临着统计精度与计算可行性之间的根本性矛盾：粒子权重可能发生退化坍缩，而高斯集成更新则可能遗漏非高斯的后验结构。基于得分的扩散滤波器提供了一种基于采样的替代方案，但现有的免训练得分滤波器通常依赖启发式的似然修正，由于忽略了与每个含噪反向粒子相关联的系统状态的不确定性，这可能会损害后验估计的精度。为解决这些问题，我们提出了AECSF——一种免训练的自适应集成条件得分滤波器。AECSF基于条件Tweedie恒等式构建了一个解析可处理的得分估计器，将含噪后验得分估计问题转化为在给定含噪反向粒子与观测条件下估计系统状态的条件均值。为估计这些条件……（摘要原文在此处截断）

    arXiv:2609.32411v2 Announce Type: replace-cross  Abstract: Bayesian state estimation for high-dimensional nonlinear dynamical systems entails a fundamental tension between statistical fidelity and computational tractability, as particle weights can collapse, while Gaussian ensemble updates can miss non-Gaussian posterior structure. Score-based diffusion filters offer a sampling-based alternative, but existing training-free score filters often rely on heuristic likelihood corrections, which can compromise posterior accuracy by neglecting uncertainty about the system state associated with each noisy reverse particle. To address these issues, we propose AECSF, a training-free adaptive ensemble conditional score filter. AECSF constructs an analytically tractable score estimator from the conditional Tweedie identity, which recasts noisy posterior score estimation as estimating the conditional mean of the system state given a noisy reverse particle and the observation. To estimate these cond
    
[^226]: CLOOPD：在策略蒸馏中闭合学习者回路

    CLOOPD: Closing the Learner Loop in On-Policy Distillation

    [https://arxiv.org/abs/2609.24141](https://arxiv.org/abs/2609.24141)

    CLOOPD提出一种闭环在策略蒸馏框架，将教师信号获取与学生端实现分离，通过自适应α航点、冻结评分批次、重复前向传播学生模型以及在独立token预算下分配演员工作量，显著提升了宏观准确率。

    

    在策略蒸馏（OPD）对每个新批次付出双重代价：学生生成轨迹，而更强的教师对这些轨迹进行评分。现有方法改进了哪些轨迹被评分以及教师信号如何构建，但通常仅通过一次演员更新来消耗该信号。我们提出CLOOPD，一个将教师信号获取与学生端实现相分离的闭环框架。CLOOPD在KL包络内选择自适应的α航点，冻结已评分批次及其优势，在每次演员遍历后重新对学生进行前向传播，测量实现程度，并在单独的token预算下分配演员工作量。该框架包括确定性的两次和三次遍历策略、按token计价的CLOOPD-TPMR，以及预算匹配的对照组。在8-H20节点上的六次300步运行中，在相当的教师token规模下，每个CLOOPD策略都优于一次遍历的TOP-D基准：使用CLOOPD-Fixed2，宏观准确率从15.41提升至17.78，以及19.

    arXiv:2609.24141v1 Announce Type: new  Abstract: On-policy distillation (OPD) pays twice for each fresh batch: the student generates trajectories and a stronger teacher scores them. Existing methods improve which trajectories are scored and how the teacher signal is constructed, but usually consume it with one actor update. We introduce CLOOPD, a closed-loop framework separating teacher-signal acquisition from student-side realization. CLOOPD selects an adaptive $\alpha$ waypoint inside a KL envelope, freezes the scored batch and its advantages, re-forwards the student after each actor pass, measures realization, and allocates actor work under a separate token budget. The framework includes deterministic two- and three-pass policies, token-priced CLOOPD-TPMR, and a budget-matched control. Across six 300-step runs on an 8-H20 node, every CLOOPD policy improves the one-pass TOP-D anchor at comparable teacher-token scale: macro accuracy rises from 15.41 to 17.78 with CLOOPD-Fixed2 and 19.
    
[^227]: 联邦学习中的自适应行列式客户端调度

    Adaptive Determinantal Client Scheduling in Federated Learning

    [https://arxiv.org/abs/2609.23843](https://arxiv.org/abs/2609.23843)

    该论文提出利用行列式点过程（DPP）建模客户端多样性，设计了联邦学习中的自适应行列式客户端调度方法（ADCS），通过梯度信息与质量分数构建核矩阵实现质量与多样性的灵活权衡，并给出了收敛性保证。

    

    在联邦学习中，由于数据异构性和系统异构性的存在，为模型训练调度客户端至关重要。以往的大多数工作关注被调度客户端的质量，以实现更快的收敛速度、更短的墙钟收敛时间或更好的平均模型性能，却很少考虑客户端的多样性，而多样性对于对抗异构性、改善最不利客户端的性能十分重要。在本工作中，我们提倡使用行列式点过程（DPP）来建模并增强客户端调度中的多样性。我们首先利用梯度信息和质量分数设计DPP的核矩阵，从而天然地支持灵活的质量-多样性权衡。通过对DPP应用快速MAP（最大后验）推断，我们提出了联邦学习中的自适应行列式客户端调度方法（ADCS）。我们进一步量化了ADCS的梯度近似误差，并针对一般性的带偏客户端选择建立了收敛性分析。

    arXiv:2609.23843v1 Announce Type: new  Abstract: Scheduling clients for model training is critical in federated learning due to both data and system heterogeneity. Most previous works focus on the quality of the scheduled clients to achieve faster convergence, shorter wall-clock convergence time, or better average model performance. They rarely consider the diversity of clients, which is important to counter heterogeneity and improve performance for the worst-off clients. In this work, we advocate the use of determinantal point processes (DPPs) to model and enhance the diversity in client scheduling. We first design the kernel matrices of DPPs using gradient information and quality scores, which inherently enables a flexible quality-diversity trade-off. Applying fast MAP inference over DPPs, we propose Adaptive Determinantal Client Scheduling (ADCS) in FL. We further quantify the gradient approximation error of ADCS and develop convergence analysis for general biased client selection i
    
[^228]: 基于能量移动距离的相似性配对方法用于LHC自监督预训练

    Similarity Pairing with Energy Mover's Distance for Self-Supervised Pre-Training at the LHC

    [https://arxiv.org/abs/2609.17738](https://arxiv.org/abs/2609.17738)

    该论文提出一种利用能量移动距离（EMD）按相似性配对真实对撞事件作为数据增强视图的方法，无需人工构造或模拟增强事件即可保持事件物理真实性，从而改进LHC自监督预训练。

    

    在大型强子对撞机（LHC）上训练基础模型的许多自监督方法依赖于数据增强，以鼓励模型将事件嵌入到对某些物理或探测器对称性保持不变的表示空间中。一个常见的挑战在于选择合适的增强方法集合具有很大的自由度，而下游性能正依赖于这些增强。数据增强的实现要么需要修改现有事件，可能破坏事件的真实性（保真度），要么需要模拟更多的事件变体，这在计算上非常昂贵。在本工作中，我们提出了一种数据驱动的方法，通过能量移动距离（EMD）根据相似性对事件进行配对——EMD通过衡量将一个事件变换为另一个事件所需的“功”来度量两个事件的相似程度。通过这种方法，可以采样不同的真实事件并根据其相似性进行匹配，作为学习不变性的视图，同时保持事件本身的物理内容……

    arXiv:2609.17738v1 Announce Type: cross  Abstract: Many self-supervised methods for training foundation models at the Large Hadron Collider (LHC) rely on data augmentations to encourage the model to embed events into a representation space invariant to certain physical or detector symmetries. A common challenge arises from the large freedom in choosing a proper set of augmentations on which downstream performance depends. The implementation of augmentations involves either modifying existing events, potentially breaking the event fidelity, or simulating more event variants, which is computationally intensive. In this work, we present a data-driven method of pairing events by their similarity via the energy mover's distance (EMD), which measures how similar two events are in terms of the work required to transform one into the other. With this approach, distinct events are sampled and matched by their similarity to serve as views for learning invariance, keeping the physics content of e
    
[^229]: 用于级联衰变中鲁棒共振质量回归的自监督学习

    Self-Supervised Learning for Robust Resonance Mass Regression in Cascade Decays

    [https://arxiv.org/abs/2609.17726](https://arxiv.org/abs/2609.17726)

    该论文提出遵循基础模型范式，先用VICReg自监督预训练transformer编码器以学习对各种数据破坏不变的嵌入表示，再微调用于级联衰变中重共振粒子的质量回归，从而在系统不确定性和分布偏移下实现鲁棒的质量重建。

    

    从带有丢失能量的衰变产物中重建重共振粒子的质量，是直接决定对撞机实验中新物理搜索灵敏度的核心任务之一。由于存在各种系统不确定性和分布偏移，针对该问题的监督学习方法通常难以很好地泛化。穷尽标注数据中所有可能的变化可能非常耗费计算资源，而模型泛化的失败则会破坏重建的共振峰宽度，而该宽度在寻峰分析中至关重要。在本工作中，我们遵循基础模型范式，采用自监督方法，使用VICReg预训练一个transformer编码器，以学习对各种破坏不变的嵌入表示，随后对其进行微调，用于重共振粒子的质量回归，其中共振粒子质量范围为2.5至6.5 TeV，并采用类似SUSY的级联衰变至十一体末态。我们展示了……（摘要在此处截断）

    arXiv:2609.17726v1 Announce Type: cross  Abstract: Reconstructing the mass of a heavy resonance from its decay products with missing energy is one of the central tasks that directly determine the sensitivity in new physics searches at collider experiments. Supervised learning approaches to this problem often struggle to generalize well due to the presence of various systematic uncertainties and distribution shifts. Exhausting all possible variations in the labeled data can be very compute-intensive, while a failure of the model to generalize can corrupt the reconstructed resonance widths that are critical in peak-hunting analyses. In this work, following the foundation model paradigm, we use a self-supervised approach to pre-train a transformer encoder with VICReg to learn an embedding invariant to various corruptions, then fine-tune it for mass regression on a heavy resonance with masses ranging from 2.5 to 6.5 TeV and a SUSY-like cascade decay into an eleven-body final state. We show
    
[^230]: OmniHarness：通过符号策略学习实现可泛化的视觉生成

    OmniHarness: Harnessing Generalizable Visual Generation via Symbolic Policy Learning

    [https://arxiv.org/abs/2609.16057](https://arxiv.org/abs/2609.16057)

    提出OmniHarness框架，通过将验证过的执行经验抽象为可复用的符号策略，并结合执行过程中的中间验证与自主探究式练习，实现了具有强泛化能力的视觉生成。

    

    统一的多模态大语言模型和多智能体系统推动了视觉生成领域的发展。然而，仍存在三个局限性：（1）现有方法往往蒸馏任务特定的经验，泛化能力有限；（2）反思通常推迟到任务完成后才进行；（3）知识通常仅在响应下游任务需求时才被获取。为了解决这些局限性，我们提出了OmniHarness，一个通过符号策略学习实现可泛化视觉生成的框架。OmniHarness将经过验证的执行过程抽象为面向视觉生成任务族的符号策略，在去除实例特定输入的同时，捕获共享的程序流程和适用条件。该框架可以为新任务实例化、调整和组合这些策略。执行过程中的中间验证指导策略细化与失败恢复。通过自主探究，OmniHarness能够自主生成并执行练习任务（原文摘要在此处截断）。

    arXiv:2609.16057v1 Announce Type: cross  Abstract: Unified multimodal large language models (MLLMs) and multi-agent systems have advanced visual generation. However, three limitations remain. (1) Existing methods often distill task-specific experience with limited generalizability. (2) Reflection is often deferred until task completion. (3) Knowledge is often acquired only in response to downstream task demands. To address these limitations, we introduce OmniHarness, a framework for generalizable visual generation via symbolic policy learning. OmniHarness abstracts verified executions into symbolic policies for visual generation task families, capturing shared procedures and applicability conditions while removing instance-specific inputs. The harness instantiates, adapts, and composes these policies for new tasks. Intermediate verification guides refinement and failure recovery during execution. Through self-directed inquiry, OmniHarness autonomously generates and executes practice ta
    
[^231]: AhaBench：智能体能从先验经验中学习吗？一个面向长时程持续学习的基准测试

    AhaBench: Do Agents Learn from Prior Experience? A Benchmark for Long-Horizon Continual Learning

    [https://arxiv.org/abs/2609.05435](https://arxiv.org/abs/2609.05435)

    AhaBench 是一个长时程持续学习基准，通过谜题探索、带精确验证器的数学任务和自动售货机模拟三个组件，评估固定模型在获得先验经验后，在相关但支持已被移除、改变或延迟的条件下行为是否真正得到改善。

    

    arXiv:2609.05435v1 公告类型：新论文。摘要：现代语言智能体被期望在长时程上运行：它们会提出后续问题、复用已解决的示例、处理工具反馈，并适应延迟出现的后果。然而大多数评估仍然在每次提示后重置智能体，或仅对单条轨迹的最终状态进行评分。AhaBench 提出了一个更具操作性的问题：当一个固定模型获得有用的经验后，在一个明显的支持已被移除、改变或延迟的相关评估条件下，其后续行为是否会得到改善？该测试套件包含三个组成部分：Aha-Puzzle 测试智能体在解开隐藏状态谜题后的无提示探索能力；Aha-Euler 将类似 Project Euler 的数学思想转化为带有精确验证器的生成式教学/保留任务；Aha-Vending 是一个受 Vending-Bench 启发的开源实现，测试模拟的自动售货机智能体在处理延迟反馈和运营事故时能否保持盈利。AhaBench 报告了一个由三部分组成的评估结果

    arXiv:2609.05435v1 Announce Type: new  Abstract: Modern language agents are expected to operate over long horizons: they ask follow-up questions, reuse worked examples, handle tool feedback, and adapt to delayed consequences. Most evaluations still reset the agent after a prompt or score only the final state of one trajectory. AhaBench asks a more operational question: when a fixed model receives useful experience, does its later behavior improve under a related evaluation condition where the obvious support has been removed, changed, or delayed? The suite contains three components. Aha-Puzzle tests no-hint exploration after solved hidden-state puzzles; Aha-Euler turns Project-Euler-style mathematical ideas into generated taught/held-out tasks with exact validators; and Aha-Vending, an open-source implementation inspired by Vending-Bench, tests whether a simulated vending agent remains profitable while handling delayed feedback and operational incidents. AhaBench reports a three-part s
    
[^232]: 分形维度预测角度编码数据中量子核的坍缩

    Fractal dimension predicts quantum kernel collapse in angle-encoded data

    [https://arxiv.org/abs/2609.00475](https://arxiv.org/abs/2609.00475)

    该论文提出用数据的相关性分形维度 D2 作为先验量子比特数预算，可准确预测并避免角度编码量子核的几何坍缩，使量子核在真实硬件上以最小比特数保持有效。

    

    角度编码的量子核在表格数据上，当特征映射宽度超过数据内在维度时会发生坍缩。我们提出将相关性分形维度 D2 作为一种先验的量子比特预算：编码由 FD-ASE 选取的 D2 个坐标，而不是采用 PCA-95% 宽度或全部 E 个属性。在九个数据集和状态向量模拟器（n=32）上，单层 ZZ 保真度核在 q=D2 时仍保持几何活性，而同样的核在 PCA-95% 宽度下已经坍缩。该预算依赖于映射方式：直积态映射和 IQP 映射会超出该预算，而第二层 ZZ 则会低于该预算。打包的密集角度编码和重上传编码在分形 q 下仍然存活，但当把 PCA-95% 特征堆叠到这些量子比特上时则不再存活。缩小角度带宽会使 ZZ 拐点后移，拉伸带宽则会使核更早失效。在 IBM Quantum（ibm_fez，256 次采样，n=8）上，分形宽度下的单层 ZZ 核与精确核相匹配（MAE 为 0.021）；超过该宽度后……

    arXiv:2609.00475v1 Announce Type: cross  Abstract: Angle-encoded quantum kernels on tabular data collapse when the feature map is wider than the intrinsic dimension of the data. We propose the correlation fractal dimension D2 as an a priori qubit budget: encode D2 coordinates chosen by FD-ASE instead of the PCA-95% width or all E attributes. On nine data sets and a statevector simulator (n= 32), a one-layer ZZ fidelity kernel at q=D2 stays geometrically alive while the same kernel at the PCA-95% width has already collapsed. The budget is map-dependent: product-state and IQP maps overshoot it; a second ZZ layer undershoots it. Packed dense-angle and re-uploading encodings still live at the fractal q, but not when PCA-95% features are stacked onto those qubits. Shrinking the angle bandwidth moves the ZZ knee later; stretching it kills the kernel earlier. On IBM Quantum (ibm_fez, 256 shots, n=8) the one-layer ZZ kernel at the fractal width matches the exact kernel (MAE 0.021); past that w
    
[^233]: ReLU 训练中的奇异曲率：微分与梯度流极限不必交换

    Singular Curvature in ReLU Training:Differentiation and the Gradient-Flow Limit Need Not Commute

    [https://arxiv.org/abs/2608.30960](https://arxiv.org/abs/2608.30960)

    该论文证明在 ReLU 训练中“先微分离散梯度下降再取极限”与“先取梯度流极限再微分”不可交换：离散精确导数收敛到无事件的区域传播子，而极限流的导数额外包含速度归一化的激活事件转移，由此产生秩一的奇异曲率差异，且全局凸性也无法将其完全抵消。

    

    梯度下降（GD）是梯度流的显式欧拉离散化，但一个在状态上精确的连续时间替代模型在微分之后未必仍然精确。在每个固定的非共振步长下，普通的自动微分可以精确地微分所执行的硬 ReLU GD 程序。我们证明，在固定的有限时间范围内，GD 状态收敛，并且这些精确的离散导数趋近于一个无事件的区域传播子，而极限流的导数还额外包含速度归一化的激活事件转移。一个前点 Stieltjes 表示将绝对连续的区域 Hessian 与原子界面曲率分离开来；一个非零梯度跳变会产生恰好秩一的端点差异，并且只要某个事件是严格的，全局凸性就会阻止多个事件之间的完全抵消。然而，一个标准的全局 1-强凸的残差-ReLU 平方损失风险族可以实现任意大的 r……（原文摘要在此处截断）

    arXiv:2608.30960v1 Announce Type: new  Abstract: Gradient descent (GD) is explicit Euler for gradient flow, but a state-accurate continuous-time surrogate need not remain accurate after differentiation. At every fixed nonresonant step size, ordinary automatic differentiation exactly differentiates the executed hard-ReLU GD program. We prove that, over a fixed finite horizon, the GD states converge and these exact discrete derivatives approach an event-free regional propagator, whereas the derivative of the limiting flow also contains speed-normalized activation-event transfers. A prepoint Stieltjes representation separates the absolutely continuous regional Hessian from atomic interface curvature; one nonzero gradient jump produces an exactly rank-one endpoint discrepancy, and global convexity prevents complete multi-event cancellation whenever an event is strict. Nevertheless, a standard family of globally 1-strongly convex residual-ReLU squared-loss risks realizes arbitrarily large r
    
[^234]: 通过测试时缩放视角理解在线策略蒸馏

    Towards Understanding On-Policy Distillation through the Lens of Test-Time Scaling

    [https://arxiv.org/abs/2608.11829](https://arxiv.org/abs/2608.11829)

    本研究发现在线策略蒸馏主要提升采样效率而非扩展推理能力边界，其优势在小采样预算下显著，但在大预算下会减弱。

    

    arXiv:2608.11829v1 公告类型：交叉 摘要：在线策略蒸馏（OPD）已成为增强大语言模型推理能力的一种有前景的后训练技术。通常认为，它能使学生模型从更强的教师模型中蒸馏知识，从而扩展超出OPD前基础模型的能力。在本研究中，我们通过测试时缩放的视角，通过改变采样预算K并评估pass@K和avg@K性能来审视这一观点。具体而言，在多种OPD变体中，我们观察到OPD训练的模型在不同采样预算下保持优越的avg@K性能，而pass@K的优势随着K的增加逐渐转移到OPD前的基础模型上。这些结果表明，OPD主要提高了采样效率，而非持续扩展学生模型的推理能力边界。整个OPD训练过程中的pass@K动态进一步揭示，模型逐渐转向更强的低K性能，但以牺牲高K性能为代价。

    arXiv:2608.11829v1 Announce Type: cross  Abstract: On-policy distillation (OPD) has emerged as a promising post-training technique for enhancing LLM reasoning. It is commonly believed to enable the student model to distill knowledge from a stronger teacher model, thereby expanding capabilities beyond the pre-OPD base model. In this study, we examine this view through the lens of test-time scaling by varying the sampling budget K and evaluating performance with pass@K and avg@K. Specifically, across several OPD variants, we observe that OPD-trained models maintain superior avg@K performance across sampling budgets, while the advantage in pass@K gradually shifts to the pre-OPD base models as K increases. These results suggest that OPD primarily improves sampling efficiency rather than consistently expanding the student's reasoning capability boundary. The pass@K dynamics throughout OPD training further reveal a progressive shift toward stronger small-K performance at the expense of the l
    
[^235]: CLAUDE.md 为何不断膨胀？智能体编程中的灾难性记忆

    Why Does CLAUDE.md Keep Growing? Catastrophic Remembering in Agentic Coding

    [https://arxiv.org/abs/2608.11095](https://arxiv.org/abs/2608.11095)

    该论文揭示并量化了智能体编程提示文件（如 CLAUDE.md）因“追加廉价、删除昂贵”的不完美记忆机制而近乎单调地无限制膨胀、最终导致智能体失效的“灾难性记忆”现象，并证明提示注释可有效遏制这一增长。

    

    智能体提示文件（如 CLAUDE.md）在真实代码仓库中无限制地增长，只有在历史终结或有人整体重写文件时才会停止。我们将这一现象归因于不完美的记忆：追加一条指令总是廉价的，但一旦指令背后的潜在推理丢失，在不冒回归风险的前提下删除它，需要付出 O(2^|D|) 的代价（其中 |D| 为提示中的指令数量）。连续的编辑会破坏这些潜在推理，新增速度超过删除速度，提示随之不断增长，直至智能体失去功能——我们称之为“灾难性记忆”。首先，我们在 1,867 个代码仓库中对 247,694 条指令的生命周期进行了刻画：智能体提示几乎单调增长，在整个生命周期中增长超过三倍（+226%），平均每次提交净增 4.7 条指令。删除风险随指令年龄呈几何级数衰减，这与不完美记忆假说相吻合，而与我们测试的其他竞争性解释均不符。其次，我们证明提示注释可以遏制这种增长：在可验证的世界中……（原文在此处截断）

    arXiv:2608.11095v2 Announce Type: replace  Abstract: Agentic prompts like CLAUDE.md grow without bound in real repositories, stopping only when history ends or someone rewrites the file wholesale. We trace this to imperfect recall: appending an instruction is always cheap, but once an instruction's latent reasoning is gone, deleting it without risking a regression costs O(2^|D|) in a prompt of |D| instructions. Successive edits destroy that latent reasoning, additions outrun deletions and prompts grow until agents become non-functional -- catastrophic remembering. First, we characterize this phenomenon across 247,694 instruction lifetimes in 1,867 repositories: agentic prompts grow nearly monotonically, more than tripling over their lifetime (+226%) at +4.7 net instructions per commit. Deletion hazard decays geometrically with instruction age, consistent with imperfect recall and with no other rival we test. Then, we show that prompt comments can halt the growth: in verifiable worlds b
    
[^236]: SiMDex：挖掘相似的第一人称视角视频以实现跨本体灵巧操作

    SiMDex: Mining Similar Egocentric Videos for Cross-Embodiment Dexterous Manipulation

    [https://arxiv.org/abs/2608.04196](https://arxiv.org/abs/2608.04196)

    SiMDex提出基于相似度的数据挖掘框架，通过召回-排序-重排序级联从3200万条第一人称人类视频中筛选出不足5%的最相关样本，无需修改VLA架构即可将跨本体灵巧操作成功率从47.7%提升至61.7%。

    

    近年来，扩展第一人称人类视频用于机器人操作的趋势呈现爆发式增长，然而哪些数据真正有益于灵巧操作仍不明确。我们提出了SiMDex，一个基于相似度的数据挖掘框架，将灵巧操作中VLA后训练的人类数据选择问题转化为推荐问题。对于每个机器人演示，SiMDex在约3200万个第一人称人类样本上运行召回-排序-重排序级联：召回和排序两个阶段在与机器人形态无关的动作空间中选择任务相关子集，重排序阶段则在像素级别上进行验证，全程无需改变VLA架构或训练方式。与使用等量随机采样人类数据训练的强基线相比，SiMDex仅使用约149万个挖掘出的样本（不到数据池的5%），却将整体成功率从47.7%提升至61.7%，表明选择性的数据筛选优于不加区分的数据混合。

    arXiv:2608.04196v3 Announce Type: replace-cross  Abstract: Recent years have witnessed an explosive trend of scaling ego-centric human videos for robot manipulation, yet it remains unclear which data actually benefits dexterous manipulation. We present SiMDex, a similarity-based data mining framework that casts human data selection for VLA post-training in dexterous manipulation as a recommendation problem. For each robot demonstration, SiMDex runs a recall-ranking-re-ranking cascade over ~32M egocentric human samples: two stages, recall and ranking, select a task-relevant subset in a morphology-agnostic action space, and re-ranking verifies it on pixels, with no change to VLA architecture or training. Against a strong baseline trained with an equal amount of randomly sampled human data, SiMDex uses only ~1.49M mined samples (<5% of the pool) yet improves the overall success rate from 47.7% to 61.7%, showing that selective curation outperforms indiscriminate data mixing.
    
[^237]: 增强逆混合加权：确定性与随机分布偏移下的稳健推断

    Augmented Inverse Hybrid Weighting: Robust Inference under Deterministic and Random Distribution Shifts

    [https://arxiv.org/abs/2608.00701](https://arxiv.org/abs/2608.00701)

    该论文提出将分布偏移分解为可通过重加权校正的确定性协变量偏移和可通过数据集合并降低的随机扰动这一新模型，并由此导出增强逆距离加权方法，实现了在确定性与随机分布偏移下更稳健的跨人群统计推断。

    

    对源样本重新加权以匹配目标协变量分布，是在跨人群推广证据时应对分布偏移的标准方法。该方法非常适合确定性的、可学习的协变量差异，但当源与目标之间的差异超出协变量偏移的范围，或密度比权重的估计不稳定时，该方法可能会失效。我们引入了一个新的模型，其中两个总体分布律之间的差异由一个确定性协变量偏移和一个非系统性成分组成，后者被建模为概率空间上的随机扰动。这一框架将系统性偏移（被视为偏差，通过重新加权加以校正）与随机扰动（被视为分布不确定性，通过数据集合并加以降低）区分开来。在纯随机扰动的情形下，该原理导出了增强逆距离加权方法，该方法将回归增强与方差最优的数据集层面合并相结合，并获得了渐近（摘要在此处截断）

    arXiv:2608.00701v2 Announce Type: replace-cross  Abstract: Reweighting source samples to match a target covariate distribution is a standard response to distribution shift when generalizing evidence across populations. It is well suited to deterministic, learnable covariate discrepancies, but can fail when source--target differences extend beyond covariate shift or when estimation of the density-ratio weights is unstable. We introduce a new model where the discrepancy between two population laws comprises a deterministic covariate shift and a non-systematic component, modeled as random perturbation of the probability space. This separates systematic shifts, treated as bias and corrected by reweighting, from random perturbations, treated as distributional uncertainty and reduced by dataset pooling. Under pure random perturbations, this principle yields Augmented Inverse Distance Weighting, which combines regression augmentation with variance-optimal dataset-level pooling and attains asy
    
[^238]: 真实环境中的智能体编程：生产规模下的GitHub Copilot追踪数据特征分析

    Agentic Coding in the Wild: Characterizing GitHub Copilot Traces at Production Scale

    [https://arxiv.org/abs/2608.00101](https://arxiv.org/abs/2608.00101)

    本文首次在生产规模上（740万用户、57亿次LLM调用）对GitHub Copilot编码智能体的追踪数据进行系统性分析，揭示了LLM调用与工具执行近乎1:1耦合、KV缓存命中率在轮次边界和模型切换时大幅下降、以及长尾资源占用分布等独特工作负载特性及其系统启示。

    

    AI编码智能体（如GitHub Copilot、Claude Code、Codex）将多步骤的LLM推理与工具执行交错进行，形成了一种有别于聊天机器人的新型工作负载。我们利用2026年6月至7月采样的GitHub Copilot追踪数据，首次对编码智能体进行了生产规模的表征分析，数据涵盖740万用户、8.19亿条对话、57亿次LLM调用以及946万亿token。我们的分析揭示了独特的工作负载特性及其对系统设计的特殊影响：例如，稀疏的用户发起轮次会展开为自主运行的智能体循环，LLM调用与工具执行之间呈现接近1:1的耦合关系；并行LLM调用较为罕见，但在触发子智能体的轮次中存在；KV缓存命中率在轮次内部平均可达90%，但在轮次边界处降至52%，在模型切换后更是降至6%；工作流和用户行为表现出多样化且呈长尾分布的资源占用特征。我们还进一步区分了较短的智能体响应时间与长达数分钟的用户空闲期……

    arXiv:2608.00101v2 Announce Type: replace  Abstract: AI coding agents (e.g., GitHub Copilot, Claude Code, Codex) interleave multi-step LLM inference with tool execution, creating a workload different from chatbots. We present the first production-scale characterization of coding agents using sampled GitHub Copilot traces from June-July 2026, comprising 7.4M users, 819M conversations, 5.7B LLM calls, and 946T tokens. Our analysis reveals distinctive workload properties with unique systems implications: e.g., sparse user-initiated turns unfold into autonomous agent loops with nearly 1:1 coupling between LLM calls and tool execution; parallel LLM calls are rare but exist in turns that trigger subagents; KV cache hit rates average 90% within a turn, but fall to 52% across turn boundaries and to 6% after model switches; workflows and user behaviors exhibit diverse, long-tailed resource footprints. We further distinguish short agentic turnaround times from minutes-long user idle periods betw
    
[^239]: Tokens就够了：面向推荐系统中LLM级I/O效率的双用途语义ID

    Tokens are All You Need: Dual-purpose Semantic IDs for Achieving LLM-Level I/O Efficiency in recommendation systems

    [https://arxiv.org/abs/2607.24865](https://arxiv.org/abs/2607.24865)

    提出双用途语义ID，通过分层量化将连续嵌入压缩为离散token，既作为协同身份建模用户-物品交互，又可通过轻量级解码器按需重建嵌入，从而取代大规模向量存储，实现推荐系统中LLM级的I/O效率。

    

    大规模推荐系统由于庞大而密集的嵌入表而面临“内存墙”瓶颈。虽然生成式检索使用离散token作为ID，但高维上下文仍然依赖于低效的密集格式。受计算机视觉数据压缩的启发，我们提出了双用途语义ID（Dual-purpose Semantic IDs）以实现LLM级的I/O效率。我们的方法使用分层量化将连续嵌入压缩为离散语义ID，这些语义ID同时承担两个角色：（1）协同身份：通过可学习嵌入表建模用户-物品交互；（2）内容重建：使用轻量级语义解码器进行即时嵌入近似。这种方法用按需重建取代了大规模向量存储，降低了系统开销和数据占用。我们通过离线评估以及在生产规模系统上的成功在线部署，证明了该框架的有效性。

    arXiv:2607.24865v2 Announce Type: replace-cross  Abstract: Large-scale recommendation systems face "Memory Wall" bottlenecks due to massive, dense embedding tables. While generative retrieval uses discrete tokens for IDs, high-dimensional context still relies on inefficient dense formats. Inspired by computer vision data compression, we propose Dual-purpose Semantic IDs to achieve LLM-level I/O efficiency. Our methodology uses hierarchical quantization to condense continuous embeddings into discrete Semantic IDs performing two concurrent roles: (1) Collaborative Identity: modeling user-item interactions via learnable embedding table; and (2) Content Reconstruction: using a lightweight Semantic Decoder for on-the-fly embedding approximation. This approach replaces massive vector storage with on-demand reconstruction, reducing system overhead and data footprints. We demonstrate the efficacy of our framework through offline evaluations and successful online deployment in production-scale 
    
[^240]: 用于冠状动脉血管造影SYNTAX分割的多中心数据集与基准测试

    A Multi-Center Dataset and Benchmark for SYNTAX Segmentation of Coronary Angiograms

    [https://arxiv.org/abs/2607.22139](https://arxiv.org/abs/2607.22139)

    该论文推出了CARDIAG——首个针对冠状动脉造影SYNTAX节段分割的多中心基准数据集，通过严格防止患者级和中心级数据泄露并提供前所未有的丰富注释（不平衡多类标签、导管与狭窄掩膜、专家不确定性图等），同时对24种涵盖CNN、ViT和SSM的深度学习架构进行了该领域最广泛的基准评估。

    

    冠状动脉血管造影的密集像素级分类对于自动化心血管诊断至关重要，然而计算机视觉界缺乏金标准协议和鲁棒的数据集来应对这一拓扑复杂、长尾分布严重的领域。为了弥合这一差距，我们推出了CARDIAG：一个严谨的多中心基准数据集，旨在评估深度学习架构在细粒度SYNTAX节段分类上的表现。与以往的研究不同，CARDIAG系统地防止了患者级和中心级的数据泄露，同时提供了前所未有的丰富注释套件：高度不平衡的多类别标签、显式的导管与狭窄掩膜、专家不确定性图、时序序列帧以及空间元数据。利用CARDIAG，我们开展了该领域最广泛的架构评估，对涵盖经典CNN、视觉Transformer（ViTs）和状态空间模型（SSMs）在内的24个模型进行了基准测试。

    arXiv:2607.22139v2 Announce Type: replace-cross  Abstract: Dense pixel-level classification of coronary angiography is essential for automated cardiovascular diagnostics, yet the computer vision community lacks gold-standard protocols and robust datasets to tackle this topologically complex, heavily long-tailed domain. To bridge this gap, we introduce CARDIAG: a rigorous, multi-center benchmark dataset designed to evaluate deep learning architectures on fine-grained SYNTAX segment classification. Unlike prior efforts, CARDIAG systematically prevents patient- and center-level data leakage while providing an unprecedented suite of annotations: highly imbalanced multi-class labels, explicit catheter and stenosis masks, expert uncertainty maps, temporal sequence frames, and spatial metadata. Using CARDIAG, we conduct the most extensive architectural sweep in this domain, benchmarking 24 models spanning classic CNNs, Vision Transformers (ViTs), and State-Space Models (SSMs). Moving beyond n
    
[^241]: 一种在NISQ硬件上利用量子相位估计输出概率分布进行解析训练的变分量子电路

    An Analytically Trained Variational Quantum Circuit using the Output Probability Distribution of Quantum Phase Estimation on NISQ Hardware

    [https://arxiv.org/abs/2607.20943](https://arxiv.org/abs/2607.20943)

    该论文提出利用狄利克雷核解析计算训练目标，从而无需经典模拟QPE电路，即可在NISQ硬件上训练浅层变分量子电路以替代量子相位估计并估计分子基态能量。

    

    量子相位估计（QPE）是估计分子基态能量的标准算法，常作为大型量子算法的子程序，但其深层电路使其在当前含噪硬件上难以实用。以往的一种变分方法用浅层变分量子电路（VQC）替代QPE子程序，该电路经训练后可重现QPE的测量分布，但其训练目标需要经典模拟完整的QPE电路，这对分子体系而言代价过高。我们重新表述了这一思路，从狄利克雷核出发解析地计算训练目标，仅需参考能量、辅助量子比特数目和演化时间，因此QPE中的受控幺正操作和逆量子傅里叶变换（QFT）操作完全无需被模拟。以H₂为例，我们通过四种指标将VQC在IBM量子硬件上的输出与目标分布进行比较，逐步选择VQC的结构。我们首先选择纠缠器（线性或全……

    arXiv:2607.20943v2 Announce Type: replace-cross  Abstract: Quantum Phase Estimation (QPE) is the standard algorithm for molecular ground-state energy estimation and often serves as a subroutine in larger quantum algorithms, but its deep circuits make it impractical on current noisy hardware. A previous variational approach replaces the QPE subroutine with a shallow Variational Quantum Circuit (VQC) trained to reproduce the QPE measurement distribution, but generates the training target by classically simulating the full QPE circuit, which is prohibitive for molecules. We reformulate this idea, computing the training target analytically from the Dirichlet kernel, using only a reference energy, the number of ancilla qubits and the evolution time, so the controlled-unitary and inverse-QFT operations of QPE are never simulated. Using H$_2$, we select the VQC stepwise by comparing its IBM Quantum hardware output with the target using four metrics. We first choose the entangler (linear or fu
    
[^242]: AlphaWiSE：面向持续多模态表征学习的自适应权重插值方法

    AlphaWiSE: Adaptive Weight Interpolation for Continual Multimodal Representation Learning

    [https://arxiv.org/abs/2607.15094](https://arxiv.org/abs/2607.15094)

    AlphaWiSE通过为每个参数张量自适应拟合标量插值系数，将顺序微调端点与稳定性导向端点的权重进行事后融合，在保持单一模型架构和推理成本的同时，提升了持续音频-图像-文本检索的性能。

    

    多模态检索模型在共享嵌入空间中对齐音频、图像和文本，但顺序适应可能会降低早期阶段学习到的跨模态关系。现有的持续学习流程只产生单一的最终检查点，因此每个检索方向都共享相同的稳定性-可塑性权衡。我们提出了AlphaWiSE，这是一种事后权重空间插值方法，它将顺序微调端点与通过面向稳定性的持续学习方法获得的伴随端点相结合。AlphaWiSE在示例记忆上为每个对应的参数张量拟合一个标量插值系数，同时两个端点保持冻结。插值后的张量随后被物化为单一检查点，因此部署的模型保留了单一骨干网络的架构和推理成本。在持续的音频-图像-文本检索任务中，融合模型优于其两个端点中较好的那个。

    arXiv:2607.15094v2 Announce Type: replace-cross  Abstract: Multimodal retrieval models align audio, images, and text in a shared embedding space, but sequential adaptation can degrade cross-modal relationships learned in earlier phases. Existing continual-learning procedures produce a single final checkpoint, so every retrieval direction shares the same stability-plasticity trade-off. We introduce AlphaWiSE, a post-hoc weight-space interpolation method that combines a sequential fine-tuning endpoint with a companion endpoint obtained with a stability-oriented continual-learning method. AlphaWiSE fits one scalar interpolation coefficient for each corresponding parameter tensor on an exemplar memory while both endpoints remain frozen. The interpolated tensors are then materialized into a single checkpoint, so the deployed model retains the architecture and inference cost of a single backbone. On continual audio-image-text retrieval, the fused model improves on the better of its two endpo
    
[^243]: SaturQ：饱和感知的混合精度量化

    SaturQ: Saturation-Aware Mixed-Precision Quantization

    [https://arxiv.org/abs/2607.12266](https://arxiv.org/abs/2607.12266)

    SaturQ用凹形累积函数修正传统敏感度简单求和导致的饱和高估问题，通过单次Aumann–Shapley路径积分拟合参数并用线性规划分配精度，在Qwen3-30B-A3B上将KL散度较ModelOpt AutoQuantize降低11–23%。

    

    混合精度量化为语言模型的每个模块分配各自的精度格式（如NVFP4、FP8或BF16），以在内存预算内最小化量化损伤。大多数分配方法通过累加在未量化网络中测得的各单元敏感度来预测某个配置的损伤。但随着更多单元被量化，单个单元的敏感度会趋于饱和，因此简单求和会高估紧预算下所需的高量化程度配置的损伤。仅假设损伤按可结合、递增、连续的规则累积，即可将损伤形式限定为 g(∑_i b_i)，其中敏感度求和是线性 g 的特例。我们提出SaturQ，它采用凹函数形式的 g，通过单次Aumann–Shapley路径积分拟合各单元参数 b_i，并用线性规划分配精度格式。在Qwen3-30B-A3B上，在4.2–6.0有效比特下，SaturQ相比ModelOpt AutoQuantize将对BF16的KL散度降低了11–23%，……

    arXiv:2607.12266v2 Announce Type: replace  Abstract: Mixed-precision quantization assigns each module of a language model its own precision format, such as NVFP4, FP8, or BF16, to minimize damage within a memory budget. Most allocators predict the damage of a configuration by summing per-unit sensitivities measured in unquantized networks. But the sensitivity of a unit \textit{saturates} as more units are quantized, so the sum overprices the heavily quantized configurations that tight budgets require. Assuming only that damage accumulates by an associative, increasing, continuous rule restricts it to the form $g\bigl(\sum_i b_i\bigr)$, with summed sensitivities as the case of linear $g$. We propose SaturQ, which uses a concave $g$, fits the per-unit parameters $b_i$ from a single Aumann--Shapley path integral, and allocates formats with a linear program. On Qwen3-30B-A3B, SaturQ lowers KL divergence to BF16 by 11--23\% relative to ModelOpt AutoQuantize at 4.2--6.0 effective bits, and i
    
[^244]: 审计代码强化学习训练环境中的奖励可作弊性

    Auditing Reward Hackability in Code RL Training Environments

    [https://arxiv.org/abs/2606.16062](https://arxiv.org/abs/2606.16062)

    该论文通过基于标准补丁的执行验证系统审计了代码强化学习训练环境的奖励作弊问题，发现约22.4%（强攻击者下57.8%）的SWE-bench Verified任务会接受错误补丁，且模型在这些可作弊任务上的通过率显著更高，凸显了对RL训练环境进行作弊检测与修复审计的必要性。

    

    代码强化学习根据策略是否通过任务的测试来给予奖励，因此一个会接受错误补丁的测试套件就会训练出错误的行为。我们测量了开源代码RL环境接受错误补丁的频率，以及可作弊任务被检测和修复的可靠性，并通过将每个补丁与任务的标准补丁进行执行对比来验证所有结论。首先，在49个SWE-bench Verified任务中，22.4%的任务会接受一个经Docker验证但与标准补丁行为不同的补丁，面对更强的攻击者时这一比例达57.8%。标准补丁检查是必要的：未被告知标准补丁的攻击者在其通过的“漏洞利用”中有31%实际上返回的就是标准补丁本身。其次，在134个公开的SWE-bench提交中，模型通过可作弊任务的频率比同等难度的任务高出28.6个百分点，这主要源于这些任务本身是可以通过的；在三次攻击者运行汇总来看，可作弊任务的通过率仍显著超过匹配的可完成任务（p = 0.014）。第三，修复比检测更难验证：61.9%由LLM编写的修复补丁……（摘要原文在此处截断）

    arXiv:2606.16062v2 Announce Type: replace  Abstract: Code RL rewards a policy for passing a task's tests, so a test suite that accepts a wrong patch trains the wrong behavior. We measure how often open code RL environments accept wrong patches and how reliably hackable tasks can be detected and repaired, checking each claim by execution against the task's gold patch. First, 22.4% of 49 SWE-bench Verified tasks accept a Docker-verified patch that behaves differently from the gold patch, and 57.8% against a stronger attacker. The gold-patch check is necessary: an attacker not shown the gold patch returns it for 31% of its passing "exploits". Second, across 134 public SWE-bench submissions, models pass hackable tasks 28.6 pp more often than tasks of the same difficulty, mostly because those tasks are passable at all; pooled over three attacker runs, hackable tasks still exceed matched passable ones (p = 0.014). Third, repair is harder to verify than detection: 61.9% of LLM-written repair 
    
[^245]: 跨越验证危机：交叉验证出人意料地有效降低基准测试方差

    Crossing the Validation Crisis: Cross-Validation Reduces Benchmarking Variance Surprisingly Well

    [https://arxiv.org/abs/2606.12552](https://arxiv.org/abs/2606.12552)

    本文提出“样本增益”概念，证明采用多个交叉验证划分可通过虚拟数据增强效应显著降低基准测试方差，从而大幅提升性能评估的可靠性与稳定性，有效缓解机器学习领域的验证危机。

    

    arXiv:2606.12552v2 公告类型：替换 摘要：现代机器学习的进步依赖于实证研究，即通过对新方法进行基准测试来评估相对性能。然而，评估中固有的统计变异性，加之许多算法的随机性，常常使得在有限的测试样本下性能估计不可靠。其结果便是一场验证危机，真正的进步难以辨别。在这项工作中，我们证明进行更多的交叉验证能够显著提高评估和比较学习算法性能时的置信度。我们引入了“样本增益”的概念，用以量化通过使用多个交叉验证划分来降低基准测试方差所实现的虚拟数据增强效果。在合成数据集和真实世界数据集（组织病理学扫描和NLP微调）上的实验表明，多次划分能够大幅提高性能估计的可靠性和稳定性，且收益呈递减趋势。

    arXiv:2606.12552v2 Announce Type: replace  Abstract: Modern machine learning progresses through empirical work, benchmarking new methods to evaluate relative performance. However, the statistical variability inherent in evaluation, exacerbated by the stochastic nature of many algorithms, often makes performance estimation unreliable with the limited test samples available. The result is a validation crisis, in which genuine advances are difficult to discern. Here, we show that doing more cross-validation markedly improves confidence when evaluating and comparing learning algorithm performances. We introduce the concept of sample gain, which quantifies the virtual data augmentation achieved by using multiple cross-validation splits to reduce benchmarking variance. Experiments on both synthetic and real-world datasets (histopathologic scans and NLP fine-tuning) demonstrate that multiple splits can substantially improve the reliability and stability of performance estimates, with diminish
    
[^246]: 算子链：一种用于上下文算子学习的推理时辅助框架

    Chain of Operators: An Inference-Time Harness for In-Context Operator Learning

    [https://arxiv.org/abs/2606.12318](https://arxiv.org/abs/2606.12318)

    本文提出算子链（CHOP）框架，利用冻结基础模型的上下文学习能力，将复杂分布外任务分解为闭式数学算子序列，实现无需参数更新的推理时适应。

    

    arXiv:2606.12318v2 公告类型：替换-交叉 摘要：尽管科学基础模型在加速物理模拟和数值预测方面展现出巨大潜力，但它们在面对分布外（OOD）场景时仍然非常脆弱。将这些通用模型适应于复杂的OOD任务通常需要昂贵的参数微调。在语言AI中，这一瓶颈通过使用“辅助框架”（harnesses）来绕过，这些框架作为外部脚手架，如推理链和工具使用，将冻结的权重适应于复杂任务，然而为物理领域设计等效的辅助框架仍是一个开放的前沿问题。为弥合这一差距，我们引入了算子链（CHOP），这是一个框架，它引导冻结的基础模型通过复杂的OOD任务而无需更新任何权重。通过利用上下文算子网络（ICON）的上下文学习能力，CHOP系统性地将不熟悉的问题分解为一系列显式、闭式的数学算子。

    arXiv:2606.12318v2 Announce Type: replace-cross  Abstract: While scientific foundation models show immense promise in accelerating physical simulations and numerical forecasting, they remain notoriously brittle when encountering out-of-distribution (OOD) scenarios. Adapting these generalist models to complex OOD tasks typically requires expensive parameter fine-tuning. In linguistic AI, this bottleneck is bypassed using "harnesses", which serve as external scaffolding such as reasoning chains and tool use to adapt frozen weights to complex tasks, yet designing an equivalent harness for physical domains remains an open frontier. To bridge this gap, we introduce Chain of Operators (CHOP), a framework that guides a frozen foundation model through complex OOD tasks without updating a single weight. By exploiting the in-context learning capability of In-Context Operator Networks (ICON), CHOP systematically decomposes unfamiliar problems into a sequence of explicit, closed-form mathematical 
    
[^247]: EssentialGIN：一种基于图同构神经网络预测基因必需性的新方法

    EssentialGIN: a new approach for gene essentiality prediction based on graph isomorphism neural networks

    [https://arxiv.org/abs/2606.07700](https://arxiv.org/abs/2606.07700)

    本研究提出EssentialGIN方法，通过改进图同构神经网络来保留PPI网络的拓扑特征，并整合基因表达、直系同源和亚细胞定位等生物信息，从而实现对必需基因的准确预测。

    

    背景：预测必需基因（蛋白质）是一个基础且具有挑战性的问题，同时在湿实验室实验中又非常昂贵且耗时。仅基于计算方法（用于筛选湿实验室候选基因）并使用中心性度量来预测必需基因并不准确，会导致大量假阳性；因此，近期研究采用了更复杂的模型（如深度学习），并整合生物信息来识别必需基因。方法：在这项工作中，我们专注于图同构网络，将蛋白质作为节点嵌入蛋白质-蛋白质相互作用（PPI）网络中，以保留PPI网络的拓扑特征，同时整合基因表达数据、基因直系同源信息和基因亚细胞定位信息等生物数据，并构建了一种用于预测必需基因的深度架构。本工作对该图同构网络架构进行了改进。

    arXiv:2606.07700v2 Announce Type: replace-cross  Abstract: Background: Prediction of essential genes (proteins), is a basic and challenging problem but at the same time very costly and time-consuming in wet-lab experiments. Predicting essential genes, only based on computational methods (to introduce wet-lab candidates) using centrality measures are not accurate and result in large number of false positives; therefore, more complex models such as deep learning and also integration of biological information are used in recent research to identify essential genes.   Methods: In this work we focus on graph isomorphism networks, in order to embed proteins as a node in PPI network to conserve topological features of PPI network, and also integrate biological data such as gene expression data, gene orthology information and gene subcellular localization information, and introduced a deep architecture for predicting essential genes. Graph isomorphism network architecture is modified in this w
    
[^248]: 基于反应活性的机器学习用于碱激发矿渣的性能预测与设计空间探索

    Reactivity-Informed Machine Learning for Performance Prediction and Design Space Exploration of Alkali-Activated Slag

    [https://arxiv.org/abs/2606.06765](https://arxiv.org/abs/2606.06765)

    该研究构建了迄今最大的碱激发矿渣文献数据集（含3100余条抗压强度记录和155种化学成分各异的GGBS），并引入物理可解释的反应活性描述符（金属氧化物平均解离能）结合机器学习，实现了碱激发矿渣性能的准确预测与配合比设计空间的高效探索。

    

    在胶凝材料领域，建立配合比设计、原材料性能、养护条件与性能之间的定量关系一直是一项长期存在的挑战，对于前驱体与激发剂化学成分多变的碱激发材料而言尤为如此。在此，我们整理了迄今为止最大的源自文献的碱激发矿渣（AAS）数据集，包含超过3100条抗压强度记录、155种化学成分各异的粒化高炉矿渣（GGBS），以及24个涵盖前驱体化学成分、细度和反应活性的属性。我们在特征逐步丰富的多种情景下对多种机器学习（ML）算法进行了基准测试，结果表明整合GGBS化学成分、细度、养护条件与试件几何形状能够提升预测性能。金属氧化物平均解离能（AMODE）作为前驱体反应活性的一种物理可解释表征，提供了一种……（原文摘要在此处截断）

    arXiv:2606.06765v2 Announce Type: replace-cross  Abstract: Establishing quantitative relationships among mix design, raw material properties, curing conditions, and performance remains a long-standing challenge in cementitious materials, particularly for alkali-activated materials with variable precursor and activator chemistry. Here, we curated the largest literature-derived alkali-activated slag (AAS) dataset to date, comprising over 3100 compressive strength records, 155 chemically distinct ground granulated blast-furnace slags (GGBSs), and 24 attributes incorporating precursor chemistry, fineness, and reactivity. Multiple machine learning (ML) algorithms were benchmarked across progressively enriched feature scenarios, demonstrating that integrating GGBS compositions, fineness, curing conditions, and specimen geometry improves predictive performance. The average metal oxide dissociation energy (AMODE), a physically interpretable representation of precursor reactivity, provides a co
    
[^249]: 基于神经网络代理特征值灵敏度的核临界实验梯度优化

    Gradient-based optimization of nuclear criticality experiments using neural surrogate eigenvalue sensitivities

    [https://arxiv.org/abs/2606.04033](https://arxiv.org/abs/2606.04033)

    本文提出利用物理信息神经网络的可微性进行基于梯度的核临界实验几何优化，通过最大化与目标技术的相关系数 $c_k$ 来自动设计具有高中子学相似性的新临界实验，并能搜索栅格内材料组合的组合设计空间。

    

    先进核反应堆设计和燃料概念的验证将需要设计与目标技术具有高中子学相似性的新临界实验。中子学相似性可以通过相关系数 $c_k$ 来量化，该系数捕捉了由核数据不确定性引起的 $k_\text{eff}$ 共同偏差。通常，实验需要满足 $c_k\geq0.9$ 才能与目标技术足够相似。在这项工作中，研究团队训练了一个物理信息深度神经网络来预测基于栅格的临界实验几何结构的中子学灵敏度。利用神经网络的可微性，实现了基于梯度的设计优化，通过优化新的实验几何结构来最大化其与目标技术灵敏度分布之间的 $c_k$。这种方法可以在栅格内潜在材料组合的组合设计空间上进行优化，超越了传统的（设计方法）。

    arXiv:2606.04033v2 Announce Type: replace  Abstract: The validation of advanced nuclear reactor designs and fuel concepts will require the design of new critical experiments with high neutronic similarity to the target technology. Neutronic similarity can be quantified by the correlation coefficient $c_k$, which captures the shared bias in $k_\text{eff}$ induced by uncertainties in nuclear data. Generally, a $c_k\geq0.9$ is needed for an experiment to be sufficiently similar to a target technology. In this work, a physics-informed deep neural network is trained to predict the neutronic sensitivity of grid-based critical experiment geometries. The differentiability of the neural network is used to enable gradient-based design optimization of new experiment geometries to maximize $c_k$ with the sensitivity profile of a target technology. This approach allows for optimization over the combinatorial design space of potential material combinations within the grid, moving beyond traditional 
    
[^250]: 信赖域Q伴随匹配

    Trust Region Q Adjoint Matching

    [https://arxiv.org/abs/2605.27079](https://arxiv.org/abs/2605.27079)

    本文提出信赖域Q伴随匹配（TRQAM），通过在流策略采样过程中引入信赖域参数λ来精确加权路径空间KL散度，并借助投影对偶下降自适应控制策略更新幅度，从而解决了QAM中评论家误差被指数级放大导致性能崩溃的问题，实现了对预训练流策略的稳定离策略微调。

    

    对预训练流策略进行离策略强化学习仍然具有挑战性，其困难源于多步采样过程所带来的优化不稳定性。最近，带伴随匹配的Q学习（QAM）通过将策略改进重新表述为以学习到的评论家为指导的随机最优控制（SOC）问题，解决了这一难题。然而，QAM继承了评论家引导改进方式的一个根本性脆弱之处：微小的评论家误差会被指数级放大，并常常导致性能崩溃。本文提出了信赖域Q伴随匹配（TRQAM），这是一种稳定的离策略微调算法，通过投影对偶下降自适应地控制微调策略与预训练策略之间的路径空间KL散度。具体而言，我们在流策略的采样过程中引入了一个信赖域参数λ，并证明λ恰好对SOC目标中的路径空间KL散度进行加权。因此，我们的……（摘要截断）

    arXiv:2605.27079v2 Announce Type: replace-cross  Abstract: Off-policy reinforcement learning of pretrained flow policies remains challenging due to the instability of optimization arising from the multi-step sampling process. Recently, Q-learning with Adjoint Matching (QAM) addressed this by recasting policy improvement as a stochastic optimal control (SOC) problem guided by a learned critic. However, QAM inherits a fundamental fragility of critic-guided improvement, since small critic errors can be exponentially amplified and often lead to performance collapse. This paper introduces Trust Region Q Adjoint Matching (TRQAM), a stable off-policy fine-tuning algorithm that adaptively controls the path-space KL between the fine-tuned and pretrained policies through projected dual descent. Specifically, we adapt a trust-region parameter $\lambda$ inside the sampling process of the flow policy and prove that $\lambda$ exactly weights the path-space KL in the SOC objective. As a result, our m
    
[^251]: 在宽度、深度和时间维度上生长神经网络

    Growing a Neural Network in Breadth, Depth, and Time

    [https://arxiv.org/abs/2605.25174](https://arxiv.org/abs/2605.25174)

    该论文提出在循环卷积网络中为宽度、深度和时间定义可微分成本并与任务误差联合优化，发现三种资源在给定精度下可相互权衡，且模型的时间分配与人类反应时间相关。

    

    大脑和人工智能系统都必须在有限的空间、时间、能量和数据条件下实现其目标。在此，我们聚焦于空间和时间，在一个循环卷积网络中为宽度（通道数）、深度（层数）和时间（循环步数）定义了可微分的成本。我们通过反向传播将这些成本与任务误差进行联合优化。通过对每种资源进行不同的定价，我们通过优化生长出多样化的计算图。我们发现，在给定的精度水平下，宽度、深度和时间三者之间可以相互权衡。网络会随着任务复杂度的增加而生长，并且当被允许为每个输入自适应地分配时间时，如果输入被遮挡，网络会执行更多的循环步骤。尽管没有在人类数据上进行任何训练，模型在每张图像上花费的时间也与人类反应时间相关。我们的工作朝着一种规范性解释迈出了一步，即空间和时间资源成本如何共同塑造大脑与人工智能系统中的神经架构。

    arXiv:2605.25174v2 Announce Type: replace-cross  Abstract: Brains and AI systems alike must achieve their goals with limited space, time, energy, and data. Here we focus on space and time, defining differentiable costs for breadth (channels), depth (layers), and time (recurrent steps) in a recurrent convolutional network. We optimize these costs jointly with task error via backpropagation. Pricing each resource differently, we grow a diverse set of computational graphs through optimization. We find that breadth, depth, and time can be traded off against each other at a given level of accuracy. Networks grow with task complexity and, when allowed to adaptively allocate time per input, take more recurrent steps if inputs are occluded. The time the model spends on each image also correlates with human reaction times, despite no training on human data. Our work takes a step toward a normative account of how spatial and temporal resource costs jointly shape neural architectures in brains an
    
[^252]: GENSTRAT：迈向大语言模型策略推理科学

    GENSTRAT: Toward a Science of Strategic Reasoning in Large Language Models

    [https://arxiv.org/abs/2605.23238](https://arxiv.org/abs/2605.23238)

    提出GENSTRAT系统，通过程序化生成双人零和不完全信息下注博弈按需创建全新评估游戏，解决了固定游戏套件饱和与数据污染问题，并通过50个游戏的基准和40,000多场模型对战的锦标赛揭示了不同LLM策略推理能力的显著差异。

    

    用于评估大语言模型（LLM）策略推理能力的固定游戏套件容易饱和，且难以避免数据污染。我们提出了GENSTRAT，一个通过程序化生成双人零和不完全信息下注博弈来评估LLM策略能力的系统，使评估者能够按需抽取新颖、全新的游戏。每个游戏在五个复杂性维度（状态空间、时间关联性、对手依赖性、私有信息、精确性）上进行特征刻画，并通过蒙特卡洛模拟进行测量。利用该生成器，我们构建了一个包含50个游戏的基准测试，并举办了十二个前沿模型和开放权重模型的锦标赛，收集了超过40,000场LLM对LLM比赛的数据。我们进一步构建了一个复合复杂性维度，发现在八个最困难的游戏上，最佳模型与最差模型之间的差距是其在全部游戏集合上差距的1.7倍。消融实验表明……（原文摘要不完整，此处为已提供部分的翻译）

    arXiv:2605.23238v2 Announce Type: replace-cross  Abstract: Fixed game suites for evaluating strategic reasoning in large language models (LLMs) saturate and are hard to keep free of contamination. We introduce GENSTRAT, a system for evaluating the strategic capabilities of LLMs driven by procedural generation of two-player zero-sum imperfect-information betting games, allowing an evaluator to draw novel, fresh games on demand. Each game is characterized on a scale of five complexity axes (state space, temporal contingency, opponent dependence, private information, precision) and measured by Monte Carlo simulation. Using this generator, we construct a 50-game benchmark and hold a tournament of twelve frontier and open-weight models, gathering data from over 40,000 LLM vs. LLM matches. We further construct a composite complexity axis, and find that, on the eight hardest games, the gap between the best and worst model is 1.7x its value across the whole set of games. Ablations show that a 
    
[^253]: 用于诊断Transformer权重分布的双参数Weibull框架

    A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions

    [https://arxiv.org/abs/2605.18898](https://arxiv.org/abs/2605.18898)

    该论文提出用双参数Weibull分布作为统一参考框架来诊断Transformer权重幅值分布，发现跨七个模型系列的FFN权重形状参数高度稳定（约1.20），而查询/键投影则呈现与架构相关的系统性偏离。

    

    Transformer检查点包含数十亿个权重，但其逐元素分布如何随组件、模型和训练而变化仍难以刻画。我们在固定的中间80%协议下，用双参数Weibull参考模型拟合每个受分析权重矩阵的幅值分布。对于独立同分布的高斯初始化，对应的半正态目标经协议拟合得到的形状参数为 $k_0 \approx 1.20$。与gamma分布和对数正态分布的匹配比较表明，Weibull分布对大多数组件组更为适用；对于单独存储的MHA查询/键投影，支持度则不够明确，且没有哪个被测试的单一分布族能干净地描述带有次级权重群体的矩阵。在跨越七个系列的12个模型条目中，终端FFN形状参数中位数落在很窄的区间内，$k \in [1.186, 1.204]$（变异系数 CV = 0.51%），而查询/键投影则表现出与架构相关的偏离。这种稳定性仅在宏观层面成立：在五个Pyth（摘要在此处被截断）

    arXiv:2605.18898v2 Announce Type: replace  Abstract: Transformer checkpoints contain billions of weights, yet how their element-wise distributions vary across components, models, and training is hard to characterize. We fit the magnitude distribution of each analyzed weight matrix with a two-parameter Weibull reference model under a fixed middle-80% protocol. For i.i.d. Gaussian initialization, the corresponding half-Normal target has protocol-fitted shape $k_0 \approx 1.20$. Matched comparisons with gamma and lognormal favor Weibull for most component groups; support is less decisive for separately stored MHA query/key projections, and no tested single family cleanly describes matrices carrying a secondary weight population.   Across 12 model entries spanning seven families, terminal FFN shape medians fall in a narrow band, $k \in [1.186, 1.204]$ (CV $= 0.51\%$), whereas query/key projections show architecture-associated departures. This stability is macroscopic only: in the five Pyth
    
[^254]: RefDecoder：通过条件视频解码增强视觉生成

    RefDecoder: Enhancing Visual Generation with Conditional Video Decoding

    [https://arxiv.org/abs/2605.15196](https://arxiv.org/abs/2605.15196)

    提出RefDecoder，一种通过参考注意力将高保真参考图像信号注入视频VAE解码过程的参考条件化解码器，有效解决了潜空间扩散模型中解码器无条件化导致的细节丢失和结构不一致问题。

    

    视频生成驱动着大量下游应用。然而，作为事实上的标准，潜空间扩散模型通常采用高度条件化的去噪网络，但其解码器往往保持无条件状态。我们观察到这种架构上的不对称性会导致相对于输入图像的显著细节丢失和不一致性。为了解决这个问题，我们认为解码器需要同等的条件化以保持结构完整性。我们提出了RefDecoder，这是一种参考条件化的视频VAE解码器，通过参考注意力将高保真参考图像信号直接注入解码过程。具体而言，一个轻量级图像编码器将参考帧映射为细节丰富的高维标记，这些标记在每个解码器上采样阶段与去噪后的视频潜空间标记协同处理。我们在多个不同的解码器骨干网络上展示了一致的改进。

    arXiv:2605.15196v2 Announce Type: replace-cross  Abstract: Video generation powers a vast array of downstream applications. However, while the de facto standard, i.e., latent diffusion models, typically employ heavily conditioned denoising networks, their decoders often remain unconditional. We observe that this architectural asymmetry leads to significant loss of detail and inconsistency relative to the input image. To address this, we argue that the decoder requires equal conditioning to preserve structural integrity. We introduce RefDecoder, a reference-conditioned video VAE decoder by injecting high-fidelity reference image signal directly into the decoding process via reference attention. Specifically, a lightweight image encoder maps the reference frame into the detail-rich high-dimensional tokens, which are co-processed with the denoised video latent tokens at each decoder up-sampling stage. We demonstrate consistent improvements across several distinct decoder backbones (e.g., 
    
[^255]: 通过专家损失集成实现时间序列预测混合专家模型的快速训练

    Fast Training of Mixture-of-Experts for Time Series Forecasting via Expert Loss Integration

    [https://arxiv.org/abs/2605.10330](https://arxiv.org/abs/2605.10330)

    提出一种融合专家特定损失与部分在线学习策略的自适应混合专家框架，解决了小门控权重导致的优化难题，在保持计算效率的同时显著提升了时间序列预测性能。

    

    我们提出了一种新颖的自适应混合专家框架用于时间序列预测，该框架通过引入专家特定损失来解决由小门控权重引起的优化问题，专家特定损失为每个专家提供独立于门控分配权重的直接学习信号。具体而言，总体目标由基础预测损失和专家特定损失组成，使各个专家的预测误差能够与总体预测误差一起直接影响参数更新。该框架还鼓励不同的专家从数据的不同时间片段中学习。所提出的框架进一步与部分在线学习策略相结合，实现模型参数的高效增量更新。通过将专家级损失信息与部分在线优化相结合，所提出的方法在保持计算效率的同时提高了预测性能。

    arXiv:2605.10330v2 Announce Type: replace-cross  Abstract: We propose a novel adaptive Mixture-of-Experts (MoE) framework for time series forecasting that addresses the optimization problem arising from small gating weights by incorporating expert-specific losses, which provide each expert with a direct learning signal independent of the gate-assigned weight. Specifically, the overall objective comprises the base forecasting loss and expert-specific losses, allowing individual expert prediction errors to directly influence parameter updates alongside the aggregate forecasting error. The framework also encourages different experts to learn from different temporal segments of the data. The proposed framework is further combined with a partial online learning strategy that enables efficient incremental updates of model parameters. By integrating expert-level loss information with partial online optimization, the proposed method improves forecasting performance while retaining computationa
    
[^256]: 物理修复：基于自监督学习的上下文驱动流体模拟

    Inpainting physics: self-supervised learning for context-driven fluid simulation

    [https://arxiv.org/abs/2605.08832](https://arxiv.org/abs/2605.08832)

    本文提出将稳态CFD推理重新定义为修复问题，通过自监督学习速度场先验并在推理时施加边界约束，结合局部邻域分词器实现大规模三维网格的高效流体模拟。

    

    arXiv:2605.08832v3 公告类型：替换 摘要：用于计算流体动力学（CFD）的神经替代模型通常被训练为正向算子，将显式问题规格（如几何形状和边界条件）映射到解场。这使模型与训练期间所见到的条件变量紧密耦合，并在边界条件变化或局部几何修改时限制了其复用性。我们提出将稳态CFD推理重新表述为修复问题：不依赖显式边界条件进行训练，而是学习速度场的自监督先验，仅在推理期间通过固定已知区域（如入口、出口或先前模拟中未变化的区域）来施加边界约束。为了将此方法扩展到大型三维网格，我们引入了一种局部邻域分词器，将高分辨率速度场表示为紧凑的空间潜在令牌，并在此类令牌上训练潜在流匹配和掩码自编码器模型。在颅内动脉瘤案例中...

    arXiv:2605.08832v3 Announce Type: replace  Abstract: Neural surrogate models for computational fluid dynamics (CFD) are typically trained as forward operators that map explicit problem specifications, such as geometry and boundary conditions, to solution fields. This ties the model to the conditioning variables seen during training and limits reuse under boundary-condition shifts or local geometry changes. We propose to reformulate steady CFD inference as an inpainting problem: instead of training on explicit boundary conditions, we learn a self-supervised prior over velocity fields and impose boundary constraints only during inference by fixing known regions such as inlet, outlet or unchanged regions from previous simulations. To scale this idea to large 3D meshes, we introduce a local neighbourhood tokeniser that represents high-resolution velocity fields as compact spatial latent tokens and train latent flow-matching and masked-autoencoder models on these tokens. On intracranial ane
    
[^257]: NoiseRater：面向扩散模型训练的元学习噪声评估方法

    NoiseRater: Meta-Learned Noise Valuation for Diffusion Model Training

    [https://arxiv.org/abs/2605.08144](https://arxiv.org/abs/2605.08144)

    提出NoiseRater，一种通过双层元学习对给定样本和时间步条件下的噪声实例进行评分的模型，用于选择更有信息量的训练噪声，并发现同一时间步的不同噪声实现并非同等有用。

    

    训练扩散模型时，每个数据样本涉及两个随机性来源：时间步和高斯噪声的具体实现。时间步已通过调度与加权等方式得到广泛研究，而给定时间步下噪声实现的影响却仍未被充分探索。在这项工作中，我们研究了不同的噪声实例是否具有同等的信息量。我们提出了NoiseRater，这是一个在给定数据样本和时间步的条件下对单个噪声实例进行打分的网络。该评估器通过双层优化进行学习：其打分在内循环中对扩散损失进行重新加权，而在内循环更新之后，评估器通过降低验证损失来进一步更新。利用训练好的评估器来选择训练噪声，我们观察到训练噪声的三个特性。首先，同一时间步的噪声实现并非同等有用：评估器给出的最高分噪声相比独立同分布采样能带来性能提升，而其…

    arXiv:2605.08144v2 Announce Type: replace-cross  Abstract: Training a diffusion model involves two sources of randomness for each data sample: the timestep and the Gaussian noise realization. The timestep has been studied extensively through scheduling and weighting, whereas the impact of the noise realization at a given timestep is still underexplored. In this work, we examine whether different noise instances are equally informative. We introduce NoiseRater, a network that scores an individual noise instance conditioned on the data sample and timestep. The rater is learned through bilevel optimization, where its scores reweight the diffusion loss in the inner loop, and it is updated to reduce validation loss after the inner-loop updates. Using the trained rater to select training noise, we observe three properties of training noise. First, noise realizations at the same timestep are not equally useful: the rater's top-scored noise improves performance over i.i.d.\ sampling, while its
    
[^258]: Mosaic：高效的CPU-GPU协同块稀疏注意力

    Mosaic: Efficient CPU-GPU Coordinated Block-Sparse Attention

    [https://arxiv.org/abs/2605.07719](https://arxiv.org/abs/2605.07719)

    Mosaic提出了一种CPU-GPU协同的块稀疏注意力系统，通过输出贡献感知的KV预算分配、注意力头特定且粒度感知的稀疏配置以及跨设备协同调度，解决了长上下文LLM解码中CPU驻留KV缓存访问受PCIe带宽限制和GPU闲置的低效问题。

    

    arXiv:2605.07719v2 公告类型：replace-cross 摘要：长上下文LLM服务日益依赖于驻留在CPU上的上下文KV缓存，这由KV卸载和分离式预填充-解码执行所推动。尽管块稀疏注意力减少了KV访问量，但现有设计仍然效率低下：GPU集中式执行在每个解码步骤中都需要从CPU内存获取选定的KV块，使其受限于PCIe带宽；而CPU-GPU分区式执行则将CPU侧的稀疏注意力置于关键路径上，导致大量GPU时间闲置。本文提出了Mosaic，一个用于在CPU驻留上下文KV下实现高效长上下文解码的CPU-GPU协同块稀疏注意力系统。Mosaic建立在三个关键洞察之上：输出贡献感知的KV预算分配、针对注意力头且粒度感知的稀疏配置，以及针对CPU驻留KV的跨设备协同调度。基于这些洞察，Mosaic集成了一个轻量级的注意力头属性预测器和一个粒度-预算（摘要在此处被截断）

    arXiv:2605.07719v2 Announce Type: replace-cross  Abstract: Long-context LLM serving increasingly relies on CPU-resident context KV cache, driven by KV offloading and disaggregated prefill--decode execution. Although block-sparse attention reduces KV access, existing designs remain inefficient: GPU-centralized execution fetches selected KV blocks from CPU memory at each decoding step, making it PCIe-bound, while CPU--GPU partitioned execution places CPU-side sparse attention on the critical path, leaving substantial GPU time idle. This paper presents Mosaic, a CPU--GPU coordinated block-sparse attention system for efficient long-context decoding with CPU-resident context KV. Mosaic builds on three key insights: output-contribution-aware KV budgeting, head-specific and granularity-aware sparse configuration, and cross-device coordinated scheduling for sparse tasks over CPU-resident KV. Guided by these insights, Mosaic integrates a lightweight head-property predictor, a granularity--budge
    
[^259]: 何时以及为何SignSGD优于SGD：基于ℓ₁-范数下界的理论研究

    When and Why SignSGD Outperforms SGD: A Theoretical Study Based on $\ell_1$-norm Lower Bounds

    [https://arxiv.org/abs/2605.06615](https://arxiv.org/abs/2605.06615)

    本文通过引入ℓ₁-范数平稳性、ℓ∞-平滑性和可分离噪声模型这一新问题几何框架，首次从理论上证明了SignSGD在何种条件下以及为何能够超越SGD，并给出了匹配的复杂度上下界。

    

    基于符号的优化算法，如SignSGD和Muon，因其在训练大型基础模型中的卓越表现而受到广泛关注。尽管取得了这些实证成功，我们仍然缺乏对这些基于符号的方法何时以及为何优于普通SGD的理论理解。核心障碍在于，在标准的平滑性和有限方差条件下，SGD已被证明在以ℓ₂-范数度量驻点的优化问题中是极小极大最优的，这从根本上排除了基于符号的方法在标准设置中取得任何复杂度增益的可能性。为克服这一障碍，我们利用ℓ₁-范数平稳性、ℓ∞-平滑性以及可分离噪声模型来分析基于符号的优化器，这些条件能够更好地刻画符号更新的逐坐标特性。在这种独特的问题几何结构下，我们推导出了SignSGD的匹配上下界，并明确刻画了问题的类别……

    arXiv:2605.06615v2 Announce Type: replace-cross  Abstract: Sign-based optimization algorithms, such as SignSGD and Muon, have garnered significant attention for their remarkable performance in training large foundation models. Despite this empirical success, we still lack a theoretical understanding of when and why these sign-based methods outperform vanilla SGD. The core obstacle is that under standard smoothness and finite variance conditions, SGD is known to be minimax optimal for finding stationary points measured by $\ell_2$-norms, thereby fundamentally precluding any complexity gains for sign-based methods in standard settings. To overcome this barrier, we analyze sign-based optimizers leveraging $\ell_1$-norm stationarity, $\ell_\infty$-smoothness, and a separable noise model, which can better capture the coordinate-wise nature of signed updates. Under this distinct problem geometry, we derive matched upper and lower bounds for SignSGD and explicitly characterize the problem cla
    
[^260]: 基于代数谱曲线的自由解压缩

    Free Decompression with Algebraic Spectral Curves

    [https://arxiv.org/abs/2605.03634](https://arxiv.org/abs/2605.03634)

    本文利用代数谱曲线理论，为Stieltjes变换满足代数关系的谱密度建立了更通用的自由解压缩方法，放宽了原有方法的强假设，使跨矩阵规模的谱信息外推能够应用于更现实的机器学习模型。

    

    arXiv:2605.03634v2 公告类型：replace-cross 摘要：来自随机矩阵理论的工具已成为深度学习理论的核心，它们利用谱信息为泛化、鲁棒性、缩放和失效模式等建模提供机制。尽管这些工具通常能够刻画经验行为，但实际计算受矩阵规模的限制，往往只能局限于过小而不具现实性的模型。这促使人们从较小模型的行为来推断较大模型的性质。自由解压缩（FD）是最近提出的一种跨矩阵规模外推谱信息的方法，但其效用目前受到强假设的约束，这些假设使其无法应用于更现实的机器学习（ML）模型。我们利用代数谱曲线理论，为Stieltjes变换满足代数关系的谱密度提供了一种通用的自由解压缩方法学——这一建模假设在实践中更有可能成立。

    arXiv:2605.03634v2 Announce Type: replace-cross  Abstract: Tools from random matrix theory have become central to deep learning theory, using spectral information to provide mechanisms for modeling generalization, robustness, scaling, and failure modes. While often capable of modeling empirical behavior, practical computations are limited by matrix size, often imposing a restriction to models that are too small to be realistic. This motivates the inference of properties of larger models from the behavior of smaller ones. Free decompression (FD) is a recently proposed method for extrapolating spectral information across matrix sizes, but its utility is currently limited by strong assumptions that preclude its implementation on more realistic machine learning (ML) models. We use algebraic spectral curve theory to provide a general FD methodology for spectral densities whose Stieltjes transform satisfies an algebraic relation, a modeling assumption that is more likely to hold in practice.
    
[^261]: 一种基于扩散模型的尺度自适应时空联合超分辨率框架

    A Scale-Adaptive Framework for Joint Spatiotemporal Super-Resolution with Diffusion Models

    [https://arxiv.org/abs/2604.21903](https://arxiv.org/abs/2604.21903)

    该论文提出一种尺度自适应的时空联合超分辨率框架，将任务分解为条件均值的确定性预测与残差条件扩散模型，使同一架构可复用于不同的空间和时间超分辨率因子，并可通过质量守恒变换保持降水总量一致。

    

    深度学习视频超分辨率技术发展迅速，但在气候应用中，通常只对空间或时间进行超分辨率处理，而联合时空模型往往只针对单一的超分辨率（SR）因子组合（即低分辨率序列与高分辨率序列之间的空间上采样比例和时间比例）而设计，这限制了模型在不同空间分辨率和时间采样频率（帧率）之间的迁移能力。我们提出了一种尺度自适应框架，通过将时空超分辨率分解为带注意力机制的条件均值的确定性预测，以及一个残差条件扩散模型，并可选择加入质量守恒变换（确保输入与输出中的降水总量一致）以保持聚合总量，从而在不同因子间复用同一架构。我们假设较大的超分辨率因子主要增加的是欠定性（因而需要更多的上下文信息和残差不确定性），而非……

    arXiv:2604.21903v2 Announce Type: replace-cross  Abstract: Deep-learning video super-resolution has progressed rapidly, but climate applications typically super-resolve (increase resolution) either space or time, and joint spatiotemporal models are often designed for a single pair of super-resolution (SR) factors (upscaling spatial and temporal ratio between the low-resolution sequence and the high-resolution sequence), limiting transfer across spatial resolutions and temporal cadences (frame rates). We present a scale-adaptive framework that reuses the same architecture across factors by decomposing spatiotemporal SR into a deterministic prediction of the conditional mean, with attention, and a residual conditional diffusion model, with an optional mass-conservation (same precipitation amount in inputs and outputs) transform to preserve aggregated totals. Assuming that larger SR factors primarily increase underdetermination (hence required context and residual uncertainty) rather than
    
[^262]: 基于组合满意化老虎机的多用户毫米波波束与速率自适应

    Multi-User mmWave Beam and Rate Adaptation via Combinatorial Satisficing Bandits

    [https://arxiv.org/abs/2604.14908](https://arxiv.org/abs/2604.14908)

    本文提出SAT-CTS轻量级策略，将多用户毫米波系统中的波束与速率联合自适应建模为满意化目标的组合半老虎机问题，并首次给出了此类问题的有限时间遗憾界理论保证。

    

    我们研究了多用户毫米波MISO系统中的下行链路波束与速率自适应问题，其中多个基站（BS）各自使用有限码本中的模拟波束赋形，以每个用户设备（UE）唯一的波束和离散的数据传输速率来服务多个单天线用户设备。基站基于ACK/NACK反馈学习传输是否成功。为了刻画服务目标，我们引入了满意化吞吐量阈值 $\tau_r$，并将波束与速率的联合自适应建模为波束-速率元组上的组合半老虎机问题。在该框架下，我们提出了SAT-CTS，这是一种轻量级、阈值感知的策略，它将保守的置信估计与后验采样相结合，引导学习朝着满足 $\tau_r$ 的方向发展，而非仅仅追求最大化。我们的主要理论贡献在于为具有满意化目标的组合半老虎机提供了首个有限时间遗憾界：当 $\tau_r$ 可实现时，我们对累积遗憾（摘要在此处截断）……

    arXiv:2604.14908v2 Announce Type: replace-cross  Abstract: We study downlink beam and rate adaptation in a multi-user mmWave MISO system where multiple base stations (BSs), each using analog beamforming from finite codebooks, serve multiple single-antenna user equipments (UEs) with a unique beam per UE and discrete data transmission rates. BSs learn about transmission success based on ACK/NACK feedback. To encode service goals, we introduce a satisficing throughput threshold $\tau_r$ and cast joint beam and rate adaptation as a combinatorial semi-bandit over beam-rate tuples. Within this framework, we propose SAT-CTS, a lightweight, threshold-aware policy that blends conservative confidence estimates with posterior sampling, steering learning toward meeting $\tau_r$ rather than merely maximizing. Our main theoretical contribution provides the first finite-time regret bounds for combinatorial semi-bandits with satisficing objective: when $\tau_r$ is realizable, we upper bound the cumula
    
[^263]: 投影器也会遗忘：面向多模态持续指令微调的联合合并方法

    The Projector Forgets Too: Joint Merging for Multimodal Continual Instruction Tuning

    [https://arxiv.org/abs/2604.14016](https://arxiv.org/abs/2604.14016)

    该论文发现多模态持续指令微调中的灾难性遗忘不仅源于语言模型的LoRA适配器，还源于跨模态投影器，并提出JoMe方法通过联合合并这两部分的参数来有效缓解遗忘。

    

    多模态持续指令微调（MCIT）对于多模态大语言模型（MLLM）的序列任务适应至关重要，但受到灾难性遗忘的严重限制。现有文献主要关注推理所用的语言主干，而在本工作中，我们发现遗忘可能同时源于跨模态投影器和语言模型LoRA适配器这两个部分的变化。在组件恢复实验中，仅恢复较早任务的LoRA适配器而保留最终任务的投影器时，与同时恢复两个组件相比会产生精度差距。为了解决这两个遗忘来源，我们提出了JoMe（Joint Merging，联合合并）方法，它结合了跨模态投影合并（CPM）和低秩参数合并（LPM）。具体而言，CPM通过自适应合并跨模态视觉表示来恢复感知对齐……

    arXiv:2604.14016v2 Announce Type: replace-cross  Abstract: Multimodal Continual Instruction Tuning (MCIT) is essential for sequential task adaptation of Multimodal Large Language Models (MLLMs) but is severely restricted by catastrophic forgetting. While existing literature focuses on the reasoning language backbone, in this work, we find that forgetting can arise from changes in both the cross-modal projector and the language model's LoRA adapters. In component restoration experiments, restoring an earlier task's LoRA adapter while retaining the final-task projector can leave an accuracy gap relative to restoring both components. To address these two sources of forgetting, we present \textbf{JoMe} (\textbf{Jo}int \textbf{Me}rging), which combines \textbf{C}ross-modal \textbf{P}rojection \textbf{M}erging (\textbf{CPM}) and \textbf{L}ow-rank \textbf{P}arameter \textbf{M}erging (\textbf{LPM}). Specifically, CPM recovers perceptual alignment by adaptively merging cross-modal visual repres
    
[^264]: 基于模式挖掘与无监督学习的城市土地利用模式探索

    Exploring Urban Land Use Patterns by Pattern Mining and Unsupervised Learning

    [https://arxiv.org/abs/2604.13050](https://arxiv.org/abs/2604.13050)

    该论文提出了一种结合频繁项集模式挖掘与Ward聚类的可重复分析框架，从欧洲100个城市地区的遥感数据中识别出七类跨城市重复出现的土地利用配置模式，为比较性规划中的同类城市对比提供了方法支持。

    

    比较性规划需要可重复的方法来识别跨城市重复出现的土地利用配置。本研究利用2018年城市地图集（Urban Atlas 2018）中100个欧洲城市地区的数据，构建了290,396个焦点邻域事务，并在10%最低支持度下提取了1,543个频繁项集支持度特征。研究在原始归一化特征空间中应用Ward聚类，UMAP仅用于可视化。通过多准则评估和500次配对特征子采样重复实验，最终保留了七聚类的描述性方案。敏感性分析表明，在5%–15%的阈值范围内，城市相似性几何结构保持稳定，但聚类成员归属对特征表示方式的选择以及非人工土地利用内容较为敏感。该框架支持同类城市间的比较，同时明确了尺度与边界方面的局限性。

    arXiv:2604.13050v2 Announce Type: replace-cross  Abstract: Comparative planning needs reproducible methods for identifying recurring land-use configurations across cities. Using Urban Atlas 2018 data for 100 European urban areas, we construct 290,396 focal-neighborhood transactions and 1,543 frequent-itemset support features at 10\% minimum support. Ward clustering is applied in the original normalized feature space, with UMAP used only for visualization. A seven-cluster descriptive solution is retained through multi-criterion evaluation and 500 paired feature-subsampling repetitions. Sensitivity analyses show stable city-similarity geometry across 5--15\% thresholds but greater membership sensitivity to representation choices and non-artificial land-use content. The framework supports peer-city comparison while making scale and boundary limitations explicit.
    
[^265]: 差分隐私流形去噪

    Differentially Private Manifold Denoising

    [https://arxiv.org/abs/2604.00942](https://arxiv.org/abs/2604.00942)

    该论文提出一种差分隐私流形去噪框架，利用隐私参考数据校正含噪声的公共查询，通过噪声自适应软校正与共享高斯场，在无需指定内在维度的情况下实现隐私保护下的局部几何估计，并提供了理论保证。

    

    现代数据分析日益依赖于基于流形的方法，这类方法利用局部几何从含噪声的观测中提取低维结构。然而在生物医学和公共卫生等领域，隐私法规限制了对个体层面数据的访问，而为保护隐私所引入的扰动又可能扭曲分析所需的几何信息。我们提出了一种流形去噪框架，利用隐私参考数据来校正含噪声的公共查询，将差分隐私引入局部几何估计。从基于估计切空间的硬校正出发，我们进一步发展了一种噪声自适应的软校正方法，从而无需预先指定数据的内在维度。此外，我们构建了一个共享高斯场，将不同位置和带宽上的评估相互耦合，仅需适度的边际噪声放大。在流形正则性和采样密度的标准假设下，我们建立了非……（原文摘要在此处截断）

    arXiv:2604.00942v2 Announce Type: replace  Abstract: Modern data analysis increasingly relies on manifold-based methods that use local geometry to extract low-dimensional structure from noisy observations. Yet in domains such as biomedicine and public health, privacy regulations restrict access to individual-level data, while perturbations introduced to protect privacy can distort the geometric information needed for analysis. We introduce a manifold denoising framework that uses private reference data to correct noisy public queries, bringing differential privacy to local geometric estimation. Starting from a hard correction based on the estimated tangent space, we develop a noise-adaptive soft correction that avoids specifying the intrinsic dimension. Furthermore, we construct a shared Gaussian field that couples evaluations across locations and bandwidths with moderate marginal noise inflation. Under standard assumptions on manifold regularity and sampling density, we establish non-
    
[^266]: TopoPilot：面向拓扑数据分析与可视化的可靠对话式工作流自动化

    TopoPilot: Reliable Conversational Workflow Automation for Topological Data Analysis and Visualization

    [https://arxiv.org/abs/2603.25063](https://arxiv.org/abs/2603.25063)

    提出了TopoPilot——一个采用“编排器+验证器”双智能体架构的可靠智能体框架，用于自动化拓扑数据分析与可视化等复杂科学可视化工作流。

    

    近期的智能体系统表明，大型语言模型能够根据自然语言提示生成复杂的代码库。尽管取得了这一进展，但由于无效操作、隐蔽错误以及无法处理欠明确的输入，这些系统在科学可视化方面存在可靠性隐患。在超出标准基准复杂度的真实世界工作流中，这些挑战进一步加剧。在本工作中，我们提出了TopoPilot，一个可靠且可扩展的智能体框架，用于自动化复杂的科学可视化工作流。我们以拓扑数据分析与可视化作为主要应用场景，同时将该框架设计为可推广至更广泛的可视化领域。TopoPilot采用双智能体架构：编排器将用户提示转换为由原子后端操作组成的工作流，而验证器在工作流执行之前对其进行评估。其模块化设计将各组件相互隔离……

    arXiv:2603.25063v2 Announce Type: replace-cross  Abstract: Recent agentic systems demonstrate that large language models can generate complex codebases from natural language prompts. Despite this progress, they present reliability concerns for scientific visualization due to invalid operations, subtle errors, and failure to resolve underspecified inputs. These challenges are compounded in real-world workflows that exceed the complexity of standard benchmarks. In this work, we introduce TopoPilot, a reliable and extensible agentic framework for automating complex scientific visualization workflows. We focus on topological data analysis and visualization as a primary use case, while designing the framework to generalize to broader visualization domains. TopoPilot employs a two-agent architecture: an orchestrator translates user prompts into workflows composed of atomic backend actions, while a verifier evaluates workflows prior to execution. Its modular design isolates components and sup
    
[^267]: 这个梯度能带来喜悦吗？

    Does This Gradient Spark Joy?

    [https://arxiv.org/abs/2603.20526](https://arxiv.org/abs/2603.20526)

    论文提出愉悦策略梯度（DG），在前向传播中用优势与惊讶度的乘积“愉悦度”评估样本的学习价值，并通过“近藤门控”只为值得的样本执行昂贵的反向传播，从而以极少的计算实现更快的学习。

    

    策略梯度方法为每个样本都要执行一次反向传播，然而反向传播代价高昂，且大多数样本几乎没有学习价值。愉悦策略梯度在前向传播中就用“愉悦度”来评估样本的学习价值，愉悦度是优势与惊讶度（负对数概率）的乘积。我们提出了“近藤门控”，它将愉悦度与一个计算价格进行比较，只有当样本值得时才为其支付反向传播的开销；通过调节该价格可以刻画出质量与成本之间的权衡曲线。在MNIST上，即使反向传播完全免费，仅保留3%的梯度项也比标准策略梯度学习得更快，并且几乎能达到完整DG的效果。在老虎机任务中，零价格门控能够保持期望梯度方向并消除其垂直方向上的噪声；与价值和惊讶度的加性混合相比，愉悦度对样本的排序更加可靠；而具有高奖励方差的罕见动作则揭示了愉悦度失效的场景。在Transformer的词元反……（原文摘要在此截断）

    arXiv:2603.20526v2 Announce Type: replace-cross  Abstract: Policy gradient computes a backward pass for every sample, although the backward pass is expensive and most samples carry little learning value. The Delightful Policy Gradient (DG) scores learning value in the forward pass with \emph{delight}, the product of advantage and surprisal (negative log-probability). We introduce the \emph{Kondo gate}, which compares delight against a compute price and pays for a backward pass only when the sample is worth it; sweeping the price traces a quality--cost trade-off. On MNIST, keeping 3\% of gradient terms learns faster than standard policy gradient even when backward passes are free, and nearly matches full DG. In bandits, zero-price gating preserves the expected gradient direction and removes its perpendicular noise, delight ranks samples more reliably than additive mixes of value and surprise, and a rare action with high reward variance marks where delight fails. On transformer token rev
    
[^268]: 分布式摩擦下的愉悦策略梯度

    Delightful Policy Gradient under Distributed Friction

    [https://arxiv.org/abs/2603.20521](https://arxiv.org/abs/2603.20521)

    本文提出愉悦策略梯度（DG），通过用优势与惊讶度的乘积作为门控信号，在不依赖行为概率的情况下抑制分布式强化学习中罕见失败带来的噪声，同时保留罕见成功所携带的关键信息。

    

    分布式强化学习的训练数据来自过时、有缺陷或策略不匹配的执行器，这些执行器的动作在学习者策略下往往具有高惊讶度（即负对数概率）。监督微调能够从这类记录数据中稳定学习，而策略梯度方法通常不能。我们将这种差异追溯到来自惊讶数据的负学习效应：罕见的失败可能主导有限批次的更新，而罕见的成功则携带着当前策略本会错过的信息。愉悦策略梯度通过用“惊喜”（delight，即优势与惊讶度的乘积）对每一项进行门控来区分这两者，且无需行为概率。在一个受污染的老虎机问题中，DG 以一个随学习者能力提升而趋于消失的因子来抑制罕见失败的垂直噪声；任何仅基于学习者概率的门控都无法在不抑制罕见成功的情况下做到这一点。在带有过时执行器的 MNIST 任务上，DG 优于重要性加权的策略梯度方法（原文摘要在此处截断）。

    arXiv:2603.20521v3 Announce Type: replace-cross  Abstract: Distributed reinforcement learning trains on data from stale, buggy, or mismatched actors, whose actions often have high surprisal (negative log-probability) under the learner's policy. Supervised fine-tuning learns stably from such logged data, while policy gradient often does not. We trace the difference to negative learning from surprising data: rare failures can dominate finite-batch updates, while rare successes carry information the current policy would otherwise miss. The Delightful Policy Gradient (DG) separates the two by gating each term with delight, the product of advantage and surprisal, and needs no behavior probabilities. In a contaminated bandit, DG suppresses the perpendicular noise of rare failures by a factor that vanishes as the learner improves; no gate based only on learner probability can do this without also suppressing rare successes. On MNIST with stale actors, DG beats importance-weighted policy gradi
    
[^269]: GIST：面向可扩展图神经算子的规范不变谱Transformer

    GIST: Gauge-Invariant Spectral Transformers for Scalable Graph Neural Operators

    [https://arxiv.org/abs/2603.16849](https://arxiv.org/abs/2603.16849)

    GIST提出了一种可扩展的规范不变谱Transformer，通过仅对高效近似谱嵌入的两两内积施加注意力，在O(N)复杂度下严格保证规范不变性，实现跨网格分辨率与跨图结构的有效迁移。

    

    神经算子在处理不规则网格时面临一个根本性的矛盾。谱位置编码是图学习中捕捉几何结构的自然选择，但它需要立方复杂度的特征值分解，并且会因数值求解器的伪影而无意中破坏规范不变性；而现有高效近似方法通常在设计上就牺牲了规范对称性。这两种失效模式都会破坏离散化不变性：模型无法在同一域的不同网格分辨率之间迁移，在归纳学习场景下也无法在具有相关结构的不同图之间迁移。我们提出GIST（规范不变谱Transformer），这是一种可扩展的神经算子，通过将注意力机制限制在高效近似谱嵌入的两两内积上，解决了这一矛盾。我们证明这些内积能够以端到端 $\mathcal{O}(N)$ 的复杂度估计一个严格规范不变的图核，并建立了规范不变性与（摘要截断于此）

    arXiv:2603.16849v4 Announce Type: replace  Abstract: Neural operators on irregular meshes face a fundamental tension. Spectral positional encodings, a natural choice in graph learning for capturing geometry, require cubic-complexity eigendecomposition and inadvertently break gauge invariance through numerical solver artifacts; existing efficient approximations typically sacrifice gauge symmetry by design. Both failure modes break discretization invariance: models fail to transfer across mesh resolutions of the same domain, and similarly across different graphs of related structure in inductive settings. We propose GIST (Gauge-Invariant Spectral Transformer), a scalable neural operator that resolves this tension by restricting attention to pairwise inner products of efficient approximate spectral embeddings. We prove these inner products estimate an exactly gauge-invariant graph kernel at end-to-end $\mathcal{O}(N)$ complexity, and establish a formal connection between gauge invariance 
    
[^270]: 随机特征模型的幂律谱

    Power-Law Spectrum of the Random Feature Model

    [https://arxiv.org/abs/2603.14578](https://arxiv.org/abs/2603.14578)

    该论文刻画了随机特征模型总体协方差的谱结构，证明了数据协方差的幂律谱在经过随机线性投影与非线性激活后仍以幂律形式被保留，从而为神经网络缩放定律背后的谱机制提供了理论解释。

    

    神经网络的缩放定律——即损失随参数量、数据量和计算量呈幂律衰减——从根本上依赖于数据协方差的谱结构，其中幂律特征值衰减在视觉和语言任务中普遍存在。一个核心问题是：当数据经过神经网络的基本构建模块，即随机线性投影加非线性激活时，这种谱结构是被保留还是被破坏？我们针对随机特征模型研究了这一问题：给定数据 $x \sim N(0,H)\in \mathbb{R}^v$，其中 $H$ 具有 $\alpha$ 幂律谱（$\lambda_j(H) \asymp j^{-\alpha}$，$\alpha > 1$），一个高斯草图矩阵 $W \in \mathbb{R}^{v\times d}$，以及一个次数为 $p$ 的逐元素多项式 $f$，我们刻画了总体随机特征协方差 $\mathbb{E}_{x}[\frac{1}{d}f(W^\top x)^{\otimes 2}]$ 的特征值。我们证明了匹配的……

    arXiv:2603.14578v2 Announce Type: replace-cross  Abstract: Scaling laws for neural networks, in which the loss decays as a power-law in the number of parameters, data, and compute, depend fundamentally on the spectral structure of the data covariance, with power-law eigenvalue decay appearing ubiquitously in vision and language tasks. A central question is whether this spectral structure is preserved or destroyed when data passes through the basic building block of a neural network: a random linear projection followed by a nonlinear activation. We study this question for the random feature model: given data $x \sim N(0,H )\in \mathbb{R}^v$ where $H$ has $\alpha$-power-law spectrum ($\lambda_j( H ) \asymp j^{-\alpha}$, $\alpha > 1$), a Gaussian sketch matrix $W \in \mathbb{R}^{v\times d}$, and an entrywise polynomial $f$ of degree $p$, we characterize the eigenvalues of the population random-feature covariance $\mathbb{E}_{ x }[\frac{1}{d}f(. W ^\top x )^{\otimes 2}]$. We prove matching
    
[^271]: 马尔可夫决策过程设计对模拟到真实强化学习迁移的影响

    Impact of Markov Decision Process Design on Sim-to-Real Reinforcement Learning

    [https://arxiv.org/abs/2603.09427](https://arxiv.org/abs/2603.09427)

    该论文系统研究了MDP核心设计选择（状态构成、奖励、终止条件、动力学模型）对强化学习模拟到现实迁移的影响，并通过颜色混合任务证明基于物理的动力学模型在严格精度约束下可实现50%的真实成功率。

    

    强化学习（RL）在工业过程控制中展现出强大潜力，然而在仿真中训练的策略部署到物理硬件时，往往会面临显著的模拟到现实差距。本工作系统分析了核心马尔可夫决策过程（MDP）设计选择——状态构成、目标纳入、奖励设计、终止条件以及环境动力学模型——如何影响这种迁移。我们以颜色混合任务为例，在仿真与真实世界实验中评估了不同的MDP配置和混合动力学。我们在物理硬件上验证了研究结果，证明基于物理的动力学模型在严格精度约束下可实现高达50%的真实世界成功率，而简化模型则完全失败。我们的结果为在工业过程控制中部署强化学习提供了实用的MDP设计指南。

    arXiv:2603.09427v3 Announce Type: replace  Abstract: Reinforcement Learning (RL) has demonstrated strong potential for industrial process control, yet policies trained in simulation often suffer from a significant sim-to-real gap when deployed on physical hardware. This work systematically analyzes how core Markov Decision Process (MDP) design choices -- state composition, target inclusion, reward formulation, termination criteria, and environment dynamics models -- affect this transfer. Using a color mixing task, we evaluate different MDP configurations and mixing dynamics across simulation and real-world experiments. We validate our findings on physical hardware, demonstrating that physics-based dynamics models achieve up to 50% real-world success under strict precision constraints where simplified models fail entirely. Our results provide practical MDP design guidelines for deploying RL in industrial process control.
    
[^272]: 惊异-雷尼自由能

    Surprisal-R\'enyi Free Energy

    [https://arxiv.org/abs/2603.03405](https://arxiv.org/abs/2603.03405)

    本文提出惊异-雷尼自由能（SRFE）这一似然比的归一化雷尼-切尔诺夫对数矩泛函，证明其在端点极限下恢复前向与反向KL散度，并通过局部展开揭示了对数似然比方差作为一阶修正所体现的均值-方差权衡机制。

    

    前向与反向Kullback-Leibler（KL）散度在学习与推理中作为极限目标而出现，然而它们会诱导出显著不同的归纳偏置，这一点无法仅凭期望项加以解释。我们提出了惊异-雷尼自由能，它是似然比的一个归一化雷尼-切尔诺夫对数矩泛函。尽管对于固定的阶参数取值，SRFE在代数上与雷尼散度相关联，但其端点对称的归一化提供了一个独特的自由能视角，可用于分析前向与反向KL行为如何在奇异极限附近涌现。我们证明SRFE在端点极限下分别恢复出前向与反向KL散度，并推导了在这两种情形附近的局部展开，其中对数似然比的方差作为偏离KL目标的一阶修正项出现。这些展开揭示了一个显式的均值-方差权衡，支配着对KL目标的偏离。

    arXiv:2603.03405v3 Announce Type: replace-cross  Abstract: The forward and reverse Kullback-Leibler (KL) divergences arise as limiting objectives in learning and inference, yet they induce markedly different inductive biases that cannot be explained by expectation terms alone. We introduce the Surprisal-R\'enyi Free Energy (SRFE), a normalized R\'enyi-Chernoff log-moment functional of the likelihood ratio. Although SRFE is algebraically related to R\'enyi divergence for fixed values of the order parameter, its endpoint-symmetric normalization yields a distinct free-energy viewpoint for analyzing how forward- and reverse KL behavior emerges near singular limits. We show that SRFE recovers the forward and reverse KL divergences as endpoint limits and derive local expansions around both regimes in which the variance of the log-likelihood ratio appears as a first-order correction away from the KL objective. These expansions reveal an explicit mean-variance tradeoff governing departures fro
    
[^273]: 超越阿罗：多准则基准测试中从不可能到可能

    Beyond Arrow: From Impossibility to Possibility in Multi-Criteria Benchmarking

    [https://arxiv.org/abs/2602.07593](https://arxiv.org/abs/2602.07593)

    该论文突破了阿罗不可能定理对多准则基准排序的限制，识别出单峰、群可分和距离受限三类偏好结构，在这些条件下多指标基准测试可以产生稳定且有意义的大模型排名。

    

    现代基准测试（如 HELM、MMLU）需要考虑准确率、鲁棒性和效率等多个指标。当试图将这些指标汇总为单一排名时，自然的聚合程序可能产生循环的两两比较，或对模型集合变化不稳定的排名。我们将这种聚合形式化为一个社会选择问题：每个指标在每个数据集上诱导出对模型的偏好排名，而基准测试算子则跨指标聚合这些投票。虽然先前的工作主要聚焦于阿罗不可能定理，我们识别出了不可能性消失的充分条件，从而使有意义的多准则基准测试成为可能。特别是，我们研究了排名组合上的三种限制，并证明在单峰、群可分和距离受限的偏好下，基准测试算子能够构建出性质良好的相关模型排名。

    arXiv:2602.07593v2 Announce Type: replace  Abstract: Modern benchmarks such as HELM MMLU account for multiple metrics like accuracy, robustness and efficiency. When trying to turn these metrics into a single ranking, natural aggregation procedures can produce cyclic pairwise comparisons or rankings that are unstable to changes in the model set. We formalize this aggregation as a social choice problem where each metric induces a preference ranking over models on each dataset, and a benchmark operator aggregates these votes across metrics. While prior work has focused on Arrow's impossibility result, we identify sufficient conditions under which the impossibility disappears and meaningful multi-criteria benchmarking becomes possible. In particular, we deal with three restrictions on the combinations of rankings and show that on single-peaked, group-separable and distance-restricted preferences, the benchmark operator allows for the construction of well-behaved rankings of the involved mo
    
[^274]: 基于删失反馈的资源分配学习

    Learning to Allocate Resources with Censored Feedback

    [https://arxiv.org/abs/2602.06565](https://arxiv.org/abs/2602.06565)

    该论文研究了删失反馈下的在线预算分配问题，证明了 $\Omega(\sqrt{T})$ 的最坏情况遗憾下界，并提出了一种结合非平凡参数估计与置信界的乐观算法 RA-UCB，用于联合估计未知参数并最大化累积奖励。

    

    我们研究了一个在线资源分配问题：在每一轮中，需要在删失反馈下将预算 $B$ 分配给 $K$ 个臂。当且仅当满足以下两个条件时，某个臂才会产生奖励：(i) 该臂根据一个具有未知参数的、臂特定的伯努利随机变量被激活；(ii) 分配的预算超过从具有未知参数的参数分布中抽取的随机阈值。在 $T$ 轮过程中，学习者必须联合估计未知参数并分配预算，以在面临探索-利用权衡的同时最大化累积奖励。我们证明了一般模型下最坏情况遗憾下界为 $\Omega(\sqrt{T})$，即使阈值分布为已知的均匀分布也是如此。随后我们提出了 RA-UCB，这是一种利用非平凡参数估计和置信界的乐观算法。当预算 $B$ 在每轮开始时已知时，RA-UCB 实现了……（原文摘要在此处截断）

    arXiv:2602.06565v2 Announce Type: replace  Abstract: We study an online resource allocation problem in which at each round, a budget $B$ must be allocated across $K$ arms under censored feedback. An arm yields a reward if and only if two conditions are satisfied: (i) the arm is activated according to an arm-specific Bernoulli random variable with unknown parameter, and (ii) the allocated budget exceeds a random threshold drawn from a parametric distribution with unknown parameter. Over $T$ rounds, the learner must jointly estimate the unknown parameters and allocate the budget so as to maximize cumulative reward facing the exploration--exploitation trade-off. We prove a worst-case regret lower bound $\Omega(\sqrt{T})$ for the general model, even with known uniform threshold distributions. We then propose RA-UCB, an optimistic algorithm that leverages non-trivial parameter estimation and confidence bounds. When the budget $B$ is known at the beginning of each round, RA-UCB achieves a re
    
[^275]: 随机邻近算法最后迭代点的收敛速率

    Convergence Rate of the Last Iterate of Stochastic Proximal Algorithms

    [https://arxiv.org/abs/2602.05489](https://arxiv.org/abs/2602.05489)

    本文在放松有界方差假设的条件下，证明了随机梯度邻近方法与随机增量邻近方法的最后迭代点在逐分量凸性与光滑性下达到对数因子意义下最优的 $\widetilde{O}(1/\sqrt{T})$ 收敛速率，并可直接应用于多任务与联邦学习中的图引导正则项。

    

    arXiv:2602.05489v2 公告类型： replace-cross 摘要： 我们分析了两种求解可加复合凸优化问题的经典算法，该问题的目标函数是光滑项与非光滑正则项之和。第一种算法是针对单一正则项的随机梯度邻近方法；第二种是随机增量邻近方法，当正则项表示为许多非光滑函数之和时，该方法使用随机选取函数的邻近算子。我们重点关注放松获得最后迭代点收敛速率时常见但苛刻的有界方差假设。我们在逐分量凸性与光滑性条件下，证明了两种算法最后迭代点的 $\widetilde{O}(1/\sqrt{T})$ 收敛速率，该速率在对数因子意义下是最优的。我们的结果可直接应用于多任务学习和联邦学习中出现的图引导正则项，其中正则项可分解为协作图上各边函数之和。

    arXiv:2602.05489v2 Announce Type: replace-cross  Abstract: We analyze two classical algorithms for solving additively composite convex optimization problems where the objective is the sum of a smooth term and a nonsmooth regularizer. The first algorithm is the proximal stochastic gradient method for a single regularizer; the second is the randomized incremental proximal method, which uses the proximal operator of a randomly selected function when the regularizer is given as the sum of many nonsmooth functions. We focus on relaxing the bounded variance assumption that is common, yet stringent, for getting last iterate convergence rates. We prove the $\widetilde{O}(1/\sqrt{T})$ rate of convergence for the last iterate of both algorithms under componentwise convexity and smoothness, which is optimal up to log terms. Our results apply directly to graph-guided regularizers that arise in multi-task and federated learning, where the regularizer decomposes as a sum over edges of a collaboratio
    
[^276]: EMA策略梯度：利用EMA锚点与Top-k KL驯服大语言模型的强化学习

    EMA Policy Gradient: Taming Reinforcement Learning for LLMs with EMA Anchor and Top-k KL

    [https://arxiv.org/abs/2602.04417](https://arxiv.org/abs/2602.04417)

    提出EMA锚点策略和Top-k KL估计器两项简单技术来改进大语言模型的策略梯度算法，结合GRPO后在数学推理和智能体强化学习任务上取得显著性能提升。

    

    强化学习（RL）已使大语言模型（LLM）能够掌握日益复杂的推理和智能体行为。在本工作中，我们提出两种简单的技术来改进面向大语言模型的策略梯度算法。首先，我们用指数移动平均（EMA）替换强化学习过程中的固定锚点策略，这类似于深度Q学习中的目标网络。其次，我们引入Top-$k$ KL估计器，它允许在精确KL与采样KL之间进行灵活插值。我们推导了使用EMA锚点的稳定性条件；此外，我们证明了Top-$k$ KL估计器在任意$k$值下都能产生无偏的KL值和无偏的梯度，同时兼具精确KL的优势。将这两种技术与GRPO相结合（EMA-PG）可带来显著的性能提升。在数学推理方面，它使R1蒸馏的Qwen-1.5B在OlympiadBench上达到53.9%，而GRPO为50.8%。在智能体强化学习领域，使用Qwen-3B……（摘要原文在此处截断）

    arXiv:2602.04417v2 Announce Type: replace-cross  Abstract: Reinforcement Learning (RL) has enabled Large Language Models (LLMs) to acquire increasingly complex reasoning and agentic behaviors. In this work, we propose two simple techniques to improve policy gradient algorithms for LLMs. First, we replace the fixed anchor policy during RL with an Exponential Moving Average (EMA), similar to a target network in deep Q-learning. Second, we introduce Top-$k$ KL estimator, which allows for flexible interpolation between exact KL and sampled KL. We derive the stability conditions for using EMA anchor; moreover, we show that our Top-$k$ KL estimator yields both unbiased KL values and unbiased gradients at any $k$, while bringing the benefits of exact KL. When combined with GRPO, the two techniques (EMA-PG) lead to a significant performance boost. On math reasoning, it allows R1-distilled Qwen-1.5B to reach 53.9% on OlympiadBench compared to 50.8% by GRPO. On agentic RL domains, with Qwen-3B b
    
[^277]: 通过全局参数融合的最优树实现高效亚组分析

    Efficient Subgroup Analysis via Optimal Trees with Global Parameter Fusion

    [https://arxiv.org/abs/2602.04077](https://arxiv.org/abs/2602.04077)

    提出一种基于混合整数优化的融合最优因果树方法，通过全局最优划分与参数融合约束，克服了传统树方法贪婪分裂导致的次优划分和过拟合问题，实现精确的差异性治疗效应亚组识别。

    

    识别差异性治疗效应并进行统计推断（在临床研究中通常称为亚组分析）是精准健康的核心。亚组分析使从业者能够精准定位对治疗特别受益或受保护的人群，从而推动针对性干预措施的开展。基于树的递归划分方法因其良好的可解释性而被广泛应用于亚组分析。然而，这些方法存在显著的局限性，包括贪婪启发式算法导致的次优划分，以及局部估计分裂带来的过拟合问题，在样本量有限的情况下尤为突出。为了解决这些局限性，我们提出了一种融合最优因果树方法，利用混合整数优化（MIO）来实现精确的亚组识别。我们的方法确保全局最优划分，并引入参数融合约束以促进[统计推断]。……

    arXiv:2602.04077v2 Announce Type: replace-cross  Abstract: Identifying and making statistical inferences on differential treatment effects (commonly known as subgroup analysis in clinical research) is central to precision health. Subgroup analysis allows practitioners to pinpoint populations for whom a treatment is especially beneficial or protective, thereby advancing targeted interventions. Tree based recursive partitioning methods are widely used for subgroup analysis due to their interpretability. Nevertheless, these approaches encounter significant limitations, including suboptimal partitions induced by greedy heuristics and overfitting from locally estimated splits, especially under limited sample sizes. To address these limitations, we propose a fused optimal causal tree method that leverages mixed integer optimization (MIO) to facilitate precise subgroup identification. Our approach ensures globally optimal partitions and introduces a parameter fusion constraint to facilitate i
    
[^278]: 带边界流形上的保几何神经架构

    Geometry-Preserving Neural Architectures on Manifolds with Boundary

    [https://arxiv.org/abs/2602.03082](https://arxiv.org/abs/2602.03082)

    本文系统研究了几类保持几何约束的神经架构的逼近论性质，证明了一致近点正则集上投影神经ODE在含边界紧致光滑流形上的稳定性估计与可测流形值映射的 $L^2$ 万能逼近定理，并揭示了中间层约束与仅输出端约束两类架构在逼近保证上的差异。

    

    越来越多的神经架构被开发用于强制执行几何约束，包括基于投影的网络、指数映射更新以及流形神经常微分方程（neural ODEs）。本文研究了若干此类架构，旨在理解其逼近论性质与权衡。我们区分了在中间层强制执行约束的架构与仅在输出端强制执行约束的架构。对于一致近点正则集上的投影神经常微分方程，我们建立了稳定性估计，并在紧致光滑流形（包括带边界的流形）上导出了万能逼近结果。特别地，我们在 $L^2$ 拓扑下建立了对可测流形值映射的万能逼近。相比之下，仅在输出端强制执行约束的架构在适当的几何假设下可以对连续目标映射给出一致逼近保证。

    arXiv:2602.03082v3 Announce Type: replace  Abstract: A growing number of neural architectures have been developed to enforce geometric constraints, including projection-based networks, exponential-map updates, and manifold neural ODEs. In this paper, we study several such architectures, with the goal of understanding their approximation-theoretic properties and trade-offs. We distinguish between architectures that enforce constraints at intermediate layers and those that enforce constraints only at the output. For projected neural ODEs on uniformly prox-regular sets, we establish stability estimates and derive universal approximation results on compact smooth manifolds, including manifolds with boundary. In particular, we establish universal approximation of measurable manifold-valued maps in the $L^2$ topology. In contrast, architectures that enforce constraints only at the output can admit uniform approximation guarantees for continuous target maps under suitable geometric assumption
    
[^279]: TabularMath：通过程序验证合成评估表格学习中的计算外推能力

    TabularMath: Evaluating Computational Extrapolation in Tabular Learning via Program-Verified Synthesis

    [https://arxiv.org/abs/2602.02523](https://arxiv.org/abs/2602.02523)

    本文提出TabularMath基准，将GSM8K和AIME问题编译为114个经程序验证的确定性计算，用以评估表格模型从统计插值到计算外推的能力，发现TabPFN v2.5在分布内中位数R²高达0.999，且是唯一在超出训练范围时中位数R²仍保持正值（0.76）的表格方法。

    

    标准的表格基准奖励的是插值，即预测数据流形内部的值，此时局部统计平滑是有效的。然而，许多高价值的表格数据，从定价和金融规则到物理模拟，都是由确定性计算而非噪声关系产生的。表格模型能否从统计插值走向计算外推，即使在训练中未见过的范围之外也能返回精确答案？我们提出了TabularMath，这是一个包含114个确定性计算（233,472行）的基准，这些计算由GSM8K和AIME问题编译为经过验证的程序。我们评估了九种表格方法、闭式解与符号回归基线，以及使用GPT-OSS-120B的上下文学习。在回归指标上，TabPFN v2.5表现极为出色：仅用128行数据，其分布内中位数R²达到0.999，并且当几乎每个答案超出训练范围时，它是唯一一个中位数R²仍保持为正（0.76）的表格方法。

    arXiv:2602.02523v2 Announce Type: replace-cross  Abstract: Standard tabular benchmarks reward interpolation: predicting values inside the data manifold, where local statistical smoothing works. However, much high-value tabular data, from pricing and financial rules to physical simulation, is produced by deterministic computations rather than noisy relationships. Can tabular models move from statistical interpolation to computational extrapolation, returning exact answers even beyond the range seen in training? We introduce TabularMath, a benchmark of 114 deterministic computations (233,472 rows) compiled from GSM8K and AIME problems into verified programs. We evaluate nine tabular methods, closed-form and symbolic regression baselines, and in-context learning with GPT-OSS-120B. On regression metrics, TabPFN v2.5 looks excellent: with 128 rows its median $R^2$ is 0.999 in-distribution, and it is the only tabular method whose median $R^2$ stays positive (0.76) when nearly every answer ex
    
[^280]: SAME：面向多模态持续指令微调的稳定专家混合模型

    SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning

    [https://arxiv.org/abs/2602.01990](https://arxiv.org/abs/2602.01990)

    本文提出SAME框架，通过稳定专家路由和防止专家覆盖，解决了多模态持续指令微调中的路由漂移和专家漂移问题。

    

    多模态大语言模型（MLLMs）通过指令微调取得了强大性能，但在实际部署中需要持续扩展其能力，这使得多模态持续指令微调（MCIT）变得至关重要。近期方法利用稀疏专家路由来促进任务专业化，但我们发现，随着数据分布演变，专家路由过程会出现漂移。例如，一个接地（grounding）查询之前能激活定位专家，但在学习OCR任务后可能被路由到不相关的专家。同时，接地相关专家可能被新任务覆盖而失去原有功能。这种失败反映了两个问题：路由漂移，即专家选择随时间变得不一致；专家漂移，即共享专家在跨任务中被覆盖。因此，我们提出用于MCIT的稳定专家混合模型（SAME）。为解决路由漂移，SAME...

    arXiv:2602.01990v3 Announce Type: replace  Abstract: Multimodal Large Language Models (MLLMs) achieve strong performance through instruction tuning, but real-world deployment requires them to continually expand their capabilities, making Multimodal Continual Instruction Tuning (MCIT) essential. Recent methods leverage sparse expert routing to promote task specialization, but we find that the expert routing process suffers from drift as the data distribution evolves. For example, a grounding query that previously activated localization experts may instead be routed to irrelevant experts after learning OCR tasks. Meanwhile, the grounding-related experts can be overwritten by new tasks and lose their original functionality. Such failure reflects two problems: router drift, where expert selection becomes inconsistent over time, and expert drift, where shared experts are overwritten across tasks. Therefore, we propose StAbilized Mixture-of-Experts (SAME) for MCIT. To address router drift, S
    
[^281]: 自蒸馏推理器：面向大语言模型的同策略自蒸馏方法

    Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

    [https://arxiv.org/abs/2601.18734](https://arxiv.org/abs/2601.18734)

    本文提出同策略自蒸馏（OPSD）学习算法，让单个LLM在不同上下文中同时充当教师与学生——教师以特权信息（如经过验证的推理轨迹）为条件对学生进行密集监督，从而无需独立的教师模型，并显式利用了推理数据集中的标准答案。

    

    知识蒸馏通过压缩教师大语言模型（LLM）的知识来训练更小的LLM，从而提升其推理能力。同策略蒸馏（on-policy distillation）进一步推进了这一方法：由学生模型采样自身的轨迹，同时教师LLM提供密集的词元级监督，从而解决了离策略蒸馏方法中训练与推理之间的分布不匹配问题。然而，同策略蒸馏通常需要一个独立的、往往规模更大的教师LLM，并且没有显式利用推理数据集中已有的标准答案。受到一个直觉的启发——即足够强大的LLM能够对外部特权推理轨迹进行合理化解释并教会较弱的自身——我们提出了同策略自蒸馏（On-Policy Self-Distillation, OPSD），这是一种让单个LLM在不同上下文中同时充当教师与学生的学习算法。教师策略以特权信息（例如经过验证的推理轨迹）为条件进行指导……（摘要原文在此处截断）

    arXiv:2601.18734v4 Announce Type: replace-cross  Abstract: Knowledge distillation improves large language model (LLM) reasoning by compressing the knowledge of a teacher LLM to train smaller LLMs. On-policy distillation advances this approach by having the student sample its own trajectories while a teacher LLM provides dense token-level supervision, addressing the distribution mismatch between training and inference in off-policy distillation methods. However, on-policy distillation typically requires a separate, often larger, teacher LLM and does not explicitly leverage ground-truth solutions available in reasoning datasets. Inspired by the intuition that a sufficiently capable LLM can rationalize external privileged reasoning traces and teach its weaker self, we introduce On-Policy Self-Distillation (OPSD), a learning algorithm where a single LLM acts as both teacher and student with different contexts. The teacher policy conditions on privileged information (e.g., verified reasonin
    
[^282]: 逆强化学习与动态离散选择模型的高效推断

    Efficient Inference for Inverse Reinforcement Learning and Dynamic Discrete Choice Models

    [https://arxiv.org/abs/2512.24407](https://arxiv.org/abs/2512.24407)

    该论文提出了一种半参数去偏框架，通过将对数行为策略视为伪奖励，实现了在灵活奖励表示下对逆强化学习和动态离散选择模型的有效统计推断。

    

    在许多序贯决策问题中，研究者观察到行为但无法直接观测驱动行为的奖励，然而仍希望评估和比较反事实政策。逆强化学习（IRL）和动态离散选择（DDC）模型通过假设一个最优性模型将潜在奖励与观测行为联系起来，以应对这一场景。现有的灵活IRL方法允许丰富的奖励表示，但通常无法提供有效的统计推断，而经典的DDC方法仅在严格的参数结构下支持推断。我们开发了一个半参数框架，用于最大熵IRL和Gumbel冲击DDC模型中的去偏逆强化学习。我们的关键识别结果是，对数行为策略可以被视为一种伪奖励：它点识别政策价值差异，并在归一化约束下识别奖励本身。这减少了对奖励依赖估计量的推断复杂性。

    arXiv:2512.24407v2 Announce Type: replace  Abstract: In many sequential decision-making problems, researchers observe actions but not the rewards that drive behavior, yet still wish to evaluate and compare counterfactual policies. Inverse reinforcement learning (IRL) and dynamic discrete choice (DDC) models address this setting by positing an optimality model that links latent rewards to observed actions. Existing flexible IRL methods allow rich reward representations but typically do not provide valid inference, whereas classical DDC methods support inference only under restrictive parametric structure. We develop a semiparametric framework for debiased inverse reinforcement learning in maximum-entropy IRL and Gumbel-shock DDC models. Our key identification result is that the log-behavior policy can be treated as a pseudo-reward: it point-identifies policy value differences and, under a normalization constraint, the reward itself. This reduces inference on reward-dependent estimands t
    
[^283]: gHAWK：面向知识图谱上图神经网络可扩展训练的结构编码

    gHAWK: Structural Encoding for Scalable Training of Graph Neural Networks on Knowledge Graphs

    [https://arxiv.org/abs/2512.08274](https://arxiv.org/abs/2512.08274)

    gHAWK通过在训练前预计算布隆过滤器（编码局部邻域结构）和TransE嵌入（表示全局位置）作为节点的结构特征并与领域特征融合，提出了一种可扩展的大规模知识图谱GNN训练框架，克服了传统消息传递式GNN在大规模图上的扩展性瓶颈。

    

    知识图谱是结构化数据的丰富来源，而图神经网络是在其上进行学习的主要工具。然而，现有的消息传递式图神经网络难以扩展到大规模知识图谱，因为它们依赖于迭代式的消息传递过程来学习图结构。这一过程效率低下，尤其是在小批量训练场景下，每个节点只能看到其邻域的部分视图。在本文中，我们针对这一问题提出了gHAWK，这是一种新颖且可扩展的大规模知识图谱GNN训练框架。其核心思想是在GNN训练开始之前，预先为每个节点计算能够捕捉其局部和全局结构的结构特征。具体而言，gHAWK引入了一个预处理步骤，用于计算：(a) 布隆过滤器，以紧凑的方式编码局部邻域结构；(b) TransE嵌入，以表示每个节点在图中的全局位置。随后，这些特征可与任何领域特定的特征进行融合。

    arXiv:2512.08274v3 Announce Type: replace  Abstract: Knowledge Graphs (KGs) are a rich source of structured data, and graph neural networks (GNNs) are the dominant tool for learning over them. However, existing message-passing GNNs struggle to scale to large KGs because they rely on the iterative message passing process to learn graph structure. This process is inefficient, especially under mini-batch training, where a node sees only a partial view of its neighborhood. In this paper, we address this problem and present gHAWK, a novel and scalable GNN training framework for large KGs. The key idea is to precompute structural features for each node that capture its local and global structure before GNN training even begins. Specifically, gHAWK introduces a preprocessing step that computes: (a) Bloom filters to compactly encode local neighborhood structure, and (b) TransE embeddings to represent each node's global position in the graph. These features are then fused with any domain-specif
    
[^284]: 面向因果感知强化学习的以对象为中心的世界模型

    Object-Centric World Models for Causality-Aware Reinforcement Learning

    [https://arxiv.org/abs/2511.14262](https://arxiv.org/abs/2511.14262)

    该论文提出STICA统一框架，利用以对象为中心的Transformer作为世界模型，并配合因果感知的策略与价值网络，使强化学习智能体能够像人类一样分解复杂多对象环境，从而实现更高效的样本学习与决策。

    

    世界模型已被开发用于支持样本高效的深度强化学习智能体。然而，由于大多数世界模型学习的是所有环境组件的整体表示，世界模型难以准确复现那些高维、非平稳、且由具有丰富交互的多个对象组成的环境。相比之下，人类通过将环境分解为离散的对象来感知环境，从而实现高效决策。受这一洞察的启发，我们提出了STICA（Slot Transformer Imagination with CAusality-aware reinforcement learning，基于槽位Transformer想象与因果感知强化学习），这是一个统一框架，其中以对象为中心的Transformer既充当世界模型，也充当因果感知的策略网络与价值网络。STICA将每个观测表示为一组以对象为中心的token，同时包含智能体动作及其产生的奖励的token，使世界模型能够在token级别上预测环境动态……

    arXiv:2511.14262v4 Announce Type: replace-cross  Abstract: World models have been developed to support sample-efficient deep reinforcement learning agents. However, it remains challenging for world models to accurately replicate environments that are high-dimensional, non-stationary, and composed of multiple objects with rich interactions since most world models learn holistic representations of all environmental components. By contrast, humans perceive the environment by decomposing it into discrete objects, facilitating efficient decision-making. Motivated by this insight, we propose \emph{Slot Transformer Imagination with CAusality-aware reinforcement learning} (STICA), a unified framework in which object-centric Transformers serve as the world model and causality-aware policy and value networks. STICA represents each observation as a set of object-centric tokens, together with tokens for the agent action and the resulting reward, enabling the world model to predict token-level dyna
    
[^285]: 将测试时计算最优扩展泛化为可优化的图

    Generalizing Test-time Compute-optimal Scaling as an Optimizable Graph

    [https://arxiv.org/abs/2511.00086](https://arxiv.org/abs/2511.00086)

    该论文提出将测试时扩展中在固定预算下搜索计算最优模型组合与架构的新问题形式化为多LLM协作图，并将其重新表述为概率图优化问题，以应对庞大的组合搜索空间和任务定制化需求。

    

    测试时扩展通过在推理过程中分配额外的计算来提升大型语言模型（LLM）的性能，通常采用并行、串行或混合扩展方式。然而，先前的研究往往假设固定的协作架构（例如拓扑结构）以及单一模型的使用，忽略了最优架构和模型组合可能因任务不同而变化。因此，我们研究了一个新问题：在固定预算下搜索测试时扩展中计算最优的模型组合与架构。我们将其形式化为一个多LLM协作图，其中节点编码角色和LLM模型分配，边则捕捉信息流。这个问题具有挑战性，原因在于： 组合搜索空间极其庞大， 任务特定需求需要定制化的设计。为解决这些问题，我们将该问题重新表述为概率图优化，并通过初步实验得出了三条经验性……（原文摘要在此处截断）

    arXiv:2511.00086v2 Announce Type: replace-cross  Abstract: Test-Time Scaling (TTS) improves large language models (LLMs) by allocating additional computation during inference, typically through parallel, sequential, or hybrid scaling. However, prior studies often assume fixed collaboration architectures (e.g., topologies) and single-model usage, overlooking that optimal architectures and model combinations can vary across tasks. Therefore, we study the novel problem of searching for compute-optimal model combinations and architectures in TTS under a fixed budget. We formalize it as a multi-LLM collaboration graph, where nodes encode roles and LLM model assignments, and edges capture information flow. This problem is challenging because (i) the combinatorial search space is prohibitively large, and (ii) task-specific requirements demand tailored designs. To address these, we reformulate the problem as probabilistic graph optimization and, through pilot experiments, derive three empirica
    
[^286]: 重新审视价值迭代：折扣奖励与平均奖励情形的统一分析

    Revisiting Value Iteration: Unified Analysis of Discounted and Average-Reward Cases

    [https://arxiv.org/abs/2510.23914](https://arxiv.org/abs/2510.23914)

    本文通过统一的几何分析证明，在最优策略唯一且为单链的假设下，价值迭代在折扣奖励和平均奖励两种情形下均具有几何收敛性，且收敛速率快于以往理论分析所给出的结果。

    

    尽管价值迭代是强化学习中最基础的算法之一，其理论收敛保证与实际经验行为之间仍然存在持续的不匹配。在折扣奖励情形下，经典理论保证以速率 $\gamma$ 进行几何收敛；而在平均奖励情形下，近期研究认为只能期望次线性收敛。然而在实践中，人们经常观察到 VI 的收敛速度显著快于理论预测。在本工作中，我们通过一种统一的基于几何的分析证明：在最优策略唯一且为单链的假设下， 收敛在折扣奖励和平均奖励两种设置下均为几何式的， 其收敛速率快于以往分析所表明的结果。

    arXiv:2510.23914v3 Announce Type: replace  Abstract: While Value Iteration (VI) is one of the most fundamental algorithms in Reinforcement Learning, its theoretical convergence guarantees still exhibit a persistent mismatch with empirical behavior. In the discounted-reward case, classical theory guarantees geometric convergence with rate $\gamma$, while in the average-reward case recent work suggests that only sublinear convergence can be expected. In practice, however, VI is often observed to converge significantly faster. In this work, we show through a unified geometry-based analysis that, under an assumption of a unique and unichain optimal policy, (i) convergence is geometric in both the discounted- and average-reward settings and (ii) the convergence rate is faster than previous analyses suggest.
    
[^287]: 具有可靠性保证的可证明选择性自动标注

    Provable Selective Auto-labeling with Reliability Guarantees

    [https://arxiv.org/abs/2510.14581](https://arxiv.org/abs/2510.14581)

    提出了“共形标注”这一新方法，将选择性自动标注形式化为多重假设检验问题，通过构建共形p值并结合Benjamini-Hochberg程序，在错误标注率可证明受控的前提下筛选出可信的自动生成标签子集。

    

    自动标注已成为一种流行且经济高效的人工标注替代方案。然而，其性能在不同任务间往往存在高度不一致性，这凸显了选择性策略的重要性。尽管如此，现有的选择性自动标注启发式方法仍然严重依赖模型置信度分数，并且对所选样本的可信度无法提供可靠的保证。为解决这一问题，我们提出了共形标注，这是一种新颖的方法，能够选择一个自动生成标签的子集，并对其错误标注率进行可证明的控制。我们的核心思想是将选择性自动标注形式化为一个多重假设检验问题，其中每个假设表示是否接受某个实例的自动生成标签。具体而言，我们使用小型校准集为每个自动生成的标签构建共形p值，然后采用Benjamini-Hochberg（BH）程序……

    arXiv:2510.14581v4 Announce Type: replace-cross  Abstract: Auto-labeling has emerged as a popular, cost-effective alternative to manual annotation. However, its performance is often highly inconsistent across different tasks, underscoring the importance of selective strategies. Despite this, existing heuristic methods for selective auto-labeling still rely heavily on model confidence scores and offer no reliable guarantee on the trustworthiness of the selected examples. To address this, we propose \textbf{Conformal Labeling}, a novel method that selects a subset of auto-generated labels with a provably controlled false labeling rate (FLR). Our key idea is to formulate selective auto-labeling as a multiple-hypothesis testing problem, where each hypothesis indicates whether to accept the auto-generated label for an instance. Specifically, we construct a conformal $p$-value for each auto-generated label using a small calibration set, and then employ the Benjamini--Hochberg (BH) procedure 
    
[^288]: 学习同化：自回归扩散的摊销式控制

    Learning to Assimilate: Amortized Control for Autoregressive Diffusion

    [https://arxiv.org/abs/2510.06637](https://arxiv.org/abs/2510.06637)

    提出了 CADA 方法，在冻结的自回归扩散模型之上训练轻量级控制器实现摊销式数据同化，无需逐次预测进行优化即可提升稀疏观测下的长程预测精度，并将推理速度提高 10 倍以上。

    

    自回归扩散模型（ARDMs）为混沌动力学的预测提供了富有表现力的先验。然而，在长程滚动预测过程中保持与稀疏观测的一致性可能需要大量的测试时计算。我们提出了控制增强数据同化（Control-Augmented Data Assimilation, CADA），该方法在一个冻结的 ARDM 之上训练一个轻量级控制器，将预测引导向未来的观测值。通过使用 KL 正则化的随机最优控制目标，控制器从较短的自回归滚动中学习，在预测观测的同时保持贴近预训练的动力学。在推理阶段，它在每个去噪步骤注入校正，无需针对每次预测进行单独优化。在两个混沌 PDE 基准测试上，CADA 在稀疏观测条件下提升了长程预测的精度、稳定性和物理保真度。在紧凑的 ERA5 数据集上，它在大多数情形下优于基于扩散的基线方法，且推理速度比条件扩散模型快 10 倍以上。

    arXiv:2510.06637v4 Announce Type: replace-cross  Abstract: Autoregressive diffusion models (ARDMs) offer expressive priors for forecasting chaotic dynamics. However, maintaining consistency with sparse observations over long rollouts can require substantial test-time computation. We introduce Control-Augmented Data Assimilation (CADA), which trains a lightweight controller on top of a frozen ARDM to steer predictions toward future observations. Using a KL-regularized stochastic optimal control objective, the controller learns from short autoregressive rollouts to anticipate observations while staying close to the pretrained dynamics. At inference, it injects corrections at each denoising step without per-forecast optimization. Across two chaotic PDE benchmarks, CADA improves long-horizon accuracy, stability, and physical fidelity under sparse observations. On compact ERA5, it outperforms diffusion-based baselines in most regimes with over $10\times$ faster inference than conditional di
    
[^289]: 分形吸引盆几何与深度学习中可预测性和可复现性的极限

    Fractal basin geometry and the limits to predictability and reproducibility in deep learning

    [https://arxiv.org/abs/2510.05606](https://arxiv.org/abs/2510.05606)

    深度学习训练结果难以预测和复现的根本原因在于损失景观中分形且相互穿透的吸引盆几何结构——即使大幅提高初始化精度，结果的可预测性也几乎不会改善。

    

    可预测性的基本极限是我们理解许多物理和计算系统的核心。在深度学习中，训练结果以难以预测和复现而著称，然而这些困难背后的机制一直不明确。在这里，我们提出分形的、相互穿透的吸引盆几何作为统一的动力系统机制：每一个导向某一解的初始化都任意接近于导向不同解的初始化，从而形成一个在任意精细尺度上都具有结构的胖分形。利用一个最小化的神经网络模型，我们解析地建立了通往穿透现象的经典路径，并推导出接近于零的不确定性指数：即使初始化精度急剧提高，也只能带来结果可预测性的微小提升。在深度网络中，相互交织的训练终点和接近零的参数空间指数提供了与穿透现象一致的稳健特征标志。

    arXiv:2510.05606v2 Announce Type: replace  Abstract: Fundamental limits to predictability are central to our understanding of many physical and computational systems. In deep learning, training outcomes are notoriously difficult to predict and reproduce, yet the mechanism underlying these difficulties has remained unclear. Here we propose fractal, riddled basin geometry as a unifying dynamical-systems mechanism: every initialization leading to one solution lies arbitrarily close to initializations leading to different solutions, forming a fat fractal with structure at arbitrarily fine scales. Using a minimal neural network model, we analytically establish the classical route to riddling and derive a near-zero uncertainty exponent: even sharp increases in initialization precision yield only marginal gains in outcome predictability. In deep networks, intertwined training destinations and near-zero parameter-space exponents provide robust signatures consistent with riddling across tested 
    
[^290]: 利用组相似度奖励提升检索增强系统的一致性

    Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards

    [https://arxiv.org/abs/2510.04392](https://arxiv.org/abs/2510.04392)

    本文提出PS-GRPO强化学习方法，通过跨复述集的组相似度奖励来提升RAG系统在语义等价查询下的信息一致性，并提供了将一致性分解为检索器、生成器和端到端层面的系统性评估框架。

    

    检索增强生成（RAG）系统越来越多地被部署在高风险领域，用户期望其输出在语义等价的查询之间保持一致。然而，由于检索器和生成器双方均存在变异性，现有系统常常表现出显著的不一致性，损害了信任与可靠性。在这项工作中，我们关注信息一致性，即生成的输出在语义等价的输入之间应传达相同核心内容的要求。我们引入了一个有原则性的评估框架，将RAG一致性分解为检索器级别、生成器级别和端到端组件，从而能够系统地识别不一致性的来源。为了提升一致性，我们提出了复述集组相对策略优化（PS-GRPO），这是一种强化学习方法，通过在复述集上进行多次rollout来分配组相似度奖励。我们利用PS-GRPO来实现

    arXiv:2510.04392v2 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) systems are increasingly deployed in high-stakes domains where users expect outputs to be consistent across semantically equivalent queries. However, existing systems often exhibit significant inconsistencies due to variability in both the retriever and generator, undermining trust and reliability. In this work, we focus on information consistency, i.e., the requirement that generated outputs convey the same core content across semantically equivalent inputs. We introduce a principled evaluation framework that decomposes RAG consistency into retriever-level, generator-level, and end-to-end components, enabling systematic identification of inconsistency sources. To improve consistency, we propose Paraphrased Set Group Relative Policy Optimization (PS-GRPO), an RL approach that leverages multiple rollouts across paraphrased set to assign group similarity rewards. We leverage PS-GRPO to achieve
    
[^291]: 重建对齐改进了统一多模态模型

    Reconstruction Alignment Improves Unified Multimodal Models

    [https://arxiv.org/abs/2509.07295](https://arxiv.org/abs/2509.07295)

    提出一种名为RECA的后训练方法，利用视觉理解编码器嵌入作为密集监督信号，无需文本标题即可重新对齐统一多模态模型的理解与生成，从而显著提升图像生成和编辑的保真度。

    

    arXiv:2509.07295v4 公告类型：替换交叉 摘要：统一多模态模型（UMMs）在单一架构中统一了视觉理解与生成。然而，传统的训练依赖于图像-文本对（或序列），其标题通常稀疏且缺失细粒度的视觉细节，即使它们用数百个单词描述一张简单的图像。我们引入了重建对齐（RECA），一种资源高效的后训练方法，利用视觉理解编码器的嵌入作为密集的“文本提示”，在无需标题的情况下提供丰富的监督。具体而言，RECA将UMM以其自身的视觉理解嵌入为条件，并通过自监督重建损失优化其重建输入图像，从而重新对齐理解与生成。尽管方法简单，但RECA具有广泛适用性：在自回归、掩码自回归和基于扩散的UMMs中，它一致地提升了生成和编辑的保真度。仅需2（此处指训练数据或步骤的少量性，但原文不完整，按常规翻译为“仅需2”以保留原意）。

    arXiv:2509.07295v4 Announce Type: replace-cross  Abstract: Unified multimodal models (UMMs) unify visual understanding and generation within a single architecture. However, conventional training relies on image-text pairs (or sequences) whose captions are typically sparse and miss fine-grained visual details, even when they use hundreds of words to describe a simple image. We introduce Reconstruction Alignment (RECA), a resource-efficient post-training method that leverages visual understanding encoder embeddings as dense "text prompts", providing rich supervision without captions. Concretely, RECA conditions a UMM on its own visual understanding embeddings and optimizes it to reconstruct the input image with a self-supervised reconstruction loss, thereby realigning understanding and generation. Despite its simplicity, RECA is broadly applicable: across autoregressive, masked-autoregressive, and diffusion-based UMMs, it consistently improves generation and editing fidelity. With only 2
    
[^292]: 超越输出忠实性：诊断与缓解特征归因中的通路重路由

    Beyond Output Faithfulness: Diagnosing and Mitigating Pathway Rerouting in Feature Attribution

    [https://arxiv.org/abs/2509.04588](https://arxiv.org/abs/2509.04588)

    论文发现插入/删除等仅输出指标无法检测特征归因是否经由支持原始预测的计算路径，提出了可微的EQO方法揭示“通路重路由”现象，并引入逐层梯度过滤器ClipGrad来缓解该问题。

    

    插入和删除是通过逐步添加或移除像素来评估特征归因的标准指标。虽然它们能测试被归因的特征是否控制模型输出，但并未验证这些特征是否通过支持原始（干净）预测的计算路径来实现这一控制。我们使用集成分位数优化来研究这一区别，这是一种直接优化插入序列的可微方法。在CNN上，无约束EQO在偏离干净表示的同时取得了最高的插入分数，并且在消融支持干净预测的通道后仍保持其插入分数。这些结果共同揭示了一种仅靠输出指标无法检测到的“通路重路由”现象。在Vision Transformer上，来自黑色输入的目标预测和注意力偏差同样暴露了这一失败。为解决该问题，我们提出了ClipGrad，一种逐层梯度过滤器，它保留……（摘要在此处截断）

    arXiv:2509.04588v3 Announce Type: replace-cross  Abstract: Insertion and deletion are standard metrics for evaluating feature   attributions by progressively adding or removing pixels. While they test   whether attributed features control a model output, they do not verify   whether they do so through the computation that supports the clean   prediction. We study this distinction with Ensemble Quantile Optimization (EQO),   a differentiable method that directly optimizes the insertion sequence.   On CNNs, Unconstrained EQO attains the highest insertion scores while departing   from clean representations and retaining its insertion score after channels   supporting the clean prediction are ablated. Together, these results reveal   pathway rerouting that output-only metrics cannot detect. On a Vision   Transformer, target predictions from black inputs and attention deviation   expose the same failure.   To address this, we introduce ClipGrad, a layer-wise gradient   filter that preserves
    
[^293]: VASSO：用于锐度感知最小化的方差抑制方法

    VASSO: Variance Suppression for Sharpness-Aware Minimization

    [https://arxiv.org/abs/2509.02433](https://arxiv.org/abs/2509.02433)

    提出了一种通过方差抑制来稳定对抗扰动的通用方法 VASSO，可证明地解决 SAM 中“过度友好的对抗者”问题，并在视觉和语言任务上显著提升泛化能力。

    

    arXiv:2509.02433v2 公告类型：替换 摘要：锐度感知最小化（SAM）在增强深度神经网络模型泛化能力方面具有充分的文献记载优势。通过考虑损失函数几何结构中的锐度——即“平坦极小值”的邻域能提升泛化能力——SAM 通过最小化邻域内对抗性扰动引发的最大损失来寻找“平坦山谷”。尽管考虑损失函数的锐度至关重要，但在实践中 SAM 存在“过度友好的对抗者”问题，这可能限制泛化能力的最高水平。为了避免这种“友好性”，本项工作通过方差抑制（VASSO）来促进对抗者的稳定化。VASSO 提供了一种可证明地稳定对抗者的通用方法。特别是，将 VASSO 与 SAM 集成时，在大量视觉和语言任务上数值验证了泛化能力的提升。当应用于计算高效的 SAM 变体之上时，V……

    arXiv:2509.02433v2 Announce Type: replace  Abstract: Sharpness-aware minimization (SAM) has well-documented merits in enhancing generalization of deep neural network models. Accounting for sharpness in the loss function geometry, where neighborhoods of `flat minima' heighten generalization ability, SAM seeks `flat valleys' by minimizing the maximum loss provoked by an adversarial perturbation within the neighborhood. Although critical to account for sharpness of the loss function, in practice SAM suffers from `over-friendly adversaries,' which can curtail the outmost level of generalization. To avoid such `friendliness,' the present contribution fosters stabilization of adversaries through variance suppression (VASSO). VASSO offers a general approach to provably stabilize adversaries. In particular, when integrating VASSO with SAM, improved generalizability is numerically validated on extensive vision and language tasks. Once applied on top of a computationally efficient SAM variant, V
    
[^294]: SVarM：用于几何数据分类与回归的线性支持变褶皱机

    SVarM: Linear Support Varifold Machines for Classification and Regression on Geometric Data

    [https://arxiv.org/abs/2506.01189](https://arxiv.org/abs/2506.01189)

    该论文提出SVarM方法，借助形状的变褶皱测度表示及其对偶性，构建了一个在无穷维变褶皱空间上运算、类似线性支持向量机的通用框架，用于对几何形状数据进行分类与回归。

    

    尽管几何深度学习这一快速发展的领域已取得诸多进展，但对几何数据——即每个观测值都是一个形状（如曲线、图或曲面）——进行统计分析仍然充满挑战，原因在于形状空间具有非欧几里得性质，其被定义为在不变群作用下的等价类。构建能够融入这些不变性（尤其是对形状参数化的不变性）的机器学习框架，对于确保训练模型对新观测数据的泛化能力往往至关重要。本工作提出 SVarM 方法，利用形状作为测度的变褶皱表示，及其与测试函数 h: ℝⁿ × Sⁿ⁻¹ → ℝ 之间的对偶性。该方法提供了一个类似于线性支持向量机的通用框架，但其运算是在无穷维的变褶皱空间上进行的。我们通过引入……在形状数据集上构建了分类与回归模型。

    arXiv:2506.01189v3 Announce Type: replace-cross  Abstract: Despite progress in the rapidly developing field of geometric deep learning, performing statistical analysis on geometric data--where each observation is a shape such as a curve, graph, or surface--remains challenging due to the non-Euclidean nature of shape spaces, which are defined as equivalence classes under invariance groups. Building machine learning frameworks that incorporate such invariances, notably to shape parametrization, is often crucial to ensure generalizability of the trained models to new observations. This work proposes \textit{SVarM} to exploit varifold representations of shapes as measures and their duality with test functions $h:\mathbb{R}^n \times S^{n-1} \rightarrow \mathbb{R}$. This method provides a general framework akin to linear support vector machines but operating instead over the infinite-dimensional space of varifolds. We develop classification and regression models on shape datasets by introduc
    
[^295]: 基于群体相对策略优化的下一尺度视觉自回归模型微调

    Fine-Tuning Next-Scale Visual Autoregressive Models with Group Relative Policy Optimization

    [https://arxiv.org/abs/2505.23331](https://arxiv.org/abs/2505.23331)

    本文将群体相对策略优化（GRPO）应用于下一尺度视觉自回归（VAR）模型的强化学习微调，通过对齐美学与CLIP奖励信号显著提升图像质量与风格控制能力，并使模型能够泛化生成预训练阶段未见过的图像风格。

    

    使用强化学习（RL）对预训练生成模型进行微调已成为使输出更贴近细致人类偏好的一种有效方法。本文研究了将群体相对策略优化（GRPO）应用于微调下一尺度视觉自回归（VAR）模型。我们的实证结果表明，该方法能够对齐来自美学预测器和CLIP嵌入的复杂奖励信号，显著提升图像质量，并实现对生成风格的精确控制。有趣的是，通过利用CLIP，我们的方法可以帮助VAR模型泛化到其初始ImageNet分布之外：通过强化学习驱动的探索，这些模型能够生成与提示词相匹配的图像，其中涉及的图像风格在预训练阶段并未出现。总之，我们证明了基于强化学习的微调对VAR模型而言既高效又有效。

    arXiv:2505.23331v3 Announce Type: replace-cross  Abstract: Fine-tuning pre-trained generative models with Reinforcement Learning (RL) has emerged as an effective approach for aligning outputs more closely with nuanced human preferences. In this paper, we investigate the application of Group Relative Policy Optimization (GRPO) to fine-tune next-scale visual autoregressive (VAR) models. Our empirical results demonstrate that this approach enables alignment to intricate reward signals derived from aesthetic predictors and CLIP embeddings, significantly enhancing image quality and enabling precise control over the generation style. Interestingly, by leveraging CLIP, our method can help VAR models generalize beyond their initial ImageNet distribution: through RL-driven exploration, these models can generate images aligned with prompts referencing image styles that were absent during pre-training. In summary, we show that RL-based fine-tuning is both efficient and effective for VAR models, b
    
[^296]: 因果性可以系统性地解决潜藏在基准测试之下的“怪兽”

    Causality can systematically address the monsters under the bench(marks)

    [https://arxiv.org/abs/2502.05085](https://arxiv.org/abs/2502.05085)

    这篇立场论文主张，因果性为系统性地解决机器学习基准测试中的偏差、伪影和数据泄露等“怪兽”问题提供了理想框架，通过显式化因果假设并利用有原则的分析工具来提升评估的可信度与一致性。

    

    有效且可靠的评估对于推进实证机器学习至关重要。然而，通用大模型日益普及，以及朝着日益复杂的高层次任务的不断推进，使得系统性评估变得更具挑战性。基准测试受到各种偏差、伪影或数据泄露等问题的困扰，同时由于故障模式未被充分探索，模型可能表现出不可靠的行为。对这类“怪兽”的随意处理和不一致的表述导致了重复劳动、对结果缺乏信任以及缺乏依据的推断。在这篇立场论文中，我们主张因果性为系统性地应对这些挑战提供了一个理想的框架。通过将方法中的因果假设显式化，我们可以忠实地对现象建模、构建具有解释力的可检验假设，并利用有原则性的工具进行分析。为了使概念性因果模型的设计更易于上手，我们识别了几种有用的……

    arXiv:2502.05085v2 Announce Type: replace-cross  Abstract: Effective and reliable evaluation is essential for advancing empirical machine learning. However, the increasing accessibility of generalist models and the progress towards ever more complex, high-level tasks make systematic evaluation more challenging. Benchmarks are plagued by various biases, artifacts, or leakage, while models may behave unreliably due to poorly explored failure modes. Haphazard treatments and inconsistent formulations of such "monsters" leads to a duplication of efforts, a lack of trust in results, and unsupported inferences. In this position paper, we argue causality offers an ideal framework to systematically address these challenges. By making causal assumptions in an approach explicit, we can faithfully model phenomena, formulate testable hypotheses with explanatory power, and leverage principled tools for analysis. To make the design of conceptual causal models more accessible, we identify several usef
    
[^297]: 基于深度学习的状态空间模型方法：选择性综述

    Deep Learning-based Approaches for State Space Models: A Selective Review

    [https://arxiv.org/abs/2412.11211](https://arxiv.org/abs/2412.11211)

    本文对基于深度学习的状态空间模型方法进行了选择性综述，为离散时间与连续时间（如潜在神经ODE和SDE）的深度状态空间模型提供了统一的理论视角。

    

    状态空间模型（SSMs）为动力系统分析提供了一个强大的框架，其中假设系统的时间动态通过潜在状态的演化来捕捉，而这些潜在状态决定着观测值。本文对基于深度神经网络的SSMs方法的最新进展进行了选择性综述，并为离散时间深度状态空间模型和连续时间模型（如潜在神经常微分方程与随机微分方程）提供了统一的视角。文章首先概述了基于经典最大似然方法的SSMs学习方式，回顾了变分自编码器作为存在潜在变量情况下基于神经网络方法的一般学习流程，并详细讨论了属于SSM框架的代表性深度学习模型。此外，还介绍了最新的研究进展，即SSMs被用作独立架构……

    arXiv:2412.11211v2 Announce Type: replace-cross  Abstract: State-space models (SSMs) offer a powerful framework for dynamical system analysis, wherein the temporal dynamics of the system are assumed to be captured through the evolution of the latent states, which govern the values of the observations. This paper provides a selective review of recent advancements in deep neural network-based approaches for SSMs, and presents a unified perspective for discrete-time deep state space models and continuous-time ones such as latent neural Ordinary Differential and Stochastic Differential Equations. It starts with an overview of the classical maximum likelihood based approach for learning SSMs, reviews the variational autoencoder as a general learning pipeline for neural network-based approaches in the presence of latent variables, and discusses in detail representative deep learning models that fall under the SSM framework. Very recent developments, where SSMs are used as standalone architec
    
[^298]: The Well：面向机器学习的大规模多样化物理模拟数据集合集

    The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning

    [https://arxiv.org/abs/2412.00568](https://arxiv.org/abs/2412.00568)

    该论文推出了The Well——一个包含16个数据集、总计15TB数据的大规模多样化物理模拟数据集合集，涵盖生物系统、流体动力学、磁流体动力学等多个领域，并提供统一的PyTorch接口以支持机器学习模型的训练与评估。

    

    基于机器学习的代理模型为研究人员提供了加速基于仿真的工作流程的强大工具。然而，由于该领域的标准数据集通常只涵盖较少类别的物理行为，因此难以评估新方法的有效性。为填补这一空白，我们推出了The Well：一个包含各种时空物理系统数值模拟的大规模数据集合集。The Well汇集了领域专家和数值软件开发者的成果，提供了总计15TB的数据，涵盖16个数据集，涉及生物系统、流体动力学、声散射以及河外流体或超新星爆炸的磁流体动力学模拟等多个领域。这些数据集既可以单独使用，也可以作为更广泛的基准测试套件的一部分。为方便使用The Well，我们提供了统一的PyTorch接口用于模型的训练和评估。我们展示了……（摘要在此处截断）

    arXiv:2412.00568v3 Announce Type: replace  Abstract: Machine learning based surrogate models offer researchers powerful tools for accelerating simulation-based workflows. However, as standard datasets in this space often cover small classes of physical behavior, it can be difficult to evaluate the efficacy of new approaches. To address this gap, we introduce the Well: a large-scale collection of datasets containing numerical simulations of a wide variety of spatiotemporal physical systems. The Well draws from domain experts and numerical software developers to provide 15TB of data across 16 datasets covering diverse domains such as biological systems, fluid dynamics, acoustic scattering, as well as magneto-hydrodynamic simulations of extra-galactic fluids or supernova explosions. These datasets can be used individually or as part of a broader benchmark suite. To facilitate usage of the Well, we provide a unified PyTorch interface for training and evaluating models. We demonstrate the f
    
[^299]: 基于不确定性感知模仿学习的鲁棒响应式机器人操作上下文自适应策略框架

    A context-adaptive policy framework for robust and reactive robotic manipulation via uncertainty-aware imitation learning

    [https://arxiv.org/abs/2410.24035](https://arxiv.org/abs/2410.24035)

    该论文提出了一个基于不确定性感知模仿学习的上下文自适应策略框架，通过策略融合与不确定性量化技术，首次将任务参数化、鲁棒性和响应式机器人操作三者结合，解决了现有动力系统方法无法根据任务相关变量灵活调节策略的问题。

    

    在机器人学中，生成能够适应不断变化的上下文信息的鲁棒且具有响应性的操作策略是一项具有挑战性的任务。多年来，从演示中学习已成为生成响应式策略的一种直观且有效的解决方案，特别是基于动力系统的方法。然而，大多数最先进的基于动力系统的方法专注于解决鲁棒性方面的限制，而忽略了策略对环境变化的调节。因此，这些方法在利用任务相关变量进行参数化时往往缺乏灵活性。在这项工作中，我们在现有的策略融合和不确定性量化研究的基础上，提出了一个上下文自适应策略框架，将任务参数化、鲁棒性和响应式操作相结合。为此，我们使用从演示中学习的方法来获取一个以机器人状态和低维任务相关参数为条件的策略。

    arXiv:2410.24035v3 Announce Type: replace-cross  Abstract: Generating robust and reactive manipulation strategies that can adapt to changing context information is a challenging task in robotics. Over the years, Learning from Demonstration (LfD) has emerged as an intuitive and effective solution for generating reactive policies, particularly by following dynamical-system(DS)-based approaches. However, most state-of-the-art DS-based approaches focus on addressing the robustness limitations, overlooking the modulation of policies in response to the environment. As a result, they tend to be inflexible with respect to parameterization by task-dependent variables. In this work, we build on existing work on policy fusion and uncertainty quantification to propose a context-adaptive policy framework that combines task-parameterized, robust and reactive manipulation. For this, we use LfD to acquire a policy that is conditioned on the robot state and low-dimensional task-dependent parameters ref
    
[^300]: 含隐变量DAG中的平均因果效应估计：超越后门与前门准则

    Average Causal Effect Estimation in DAGs with Hidden Variables: Beyond Back-Door and Front-Door Criteria

    [https://arxiv.org/abs/2409.03962](https://arxiv.org/abs/2409.03962)

    该论文针对含隐变量DAG中超越后门与前门准则的可识别因果效应，提出了基于序贯回归和贝叶斯法则重参数化的估计方法，从而避免了条件密度估计与数值积分带来的统计和计算难题。

    

    具有隐变量的有向无环图（DAG）中因果效应的识别理论已经相当成熟，但超越g-公式的估计与推断方法仍发展不足。以往研究已针对广泛类别的可识别效应引入了基于影响函数的估计量。当通过观测数据密度分解来实现这些估计量时，往往需要进行条件密度估计和数值积分，这对于连续变量或多变量而言在统计上具有挑战性且计算代价高昂。此前关于相关因果泛函的研究采用了序贯回归和密度比的贝叶斯法则表示来避免这些需求。基于已有的识别理论与影响函数结果，我们将这些重参数化明确应用于在处理原始可固定性、Tian和Pearl的充要条件下所识别的效应……

    arXiv:2409.03962v3 Announce Type: replace-cross  Abstract: The identification theory for causal effects in directed acyclic graphs (DAGs) with hidden variables is well established, but methods for estimation and inference beyond the g-formula remain less developed. Previous studies have introduced influence-function-based estimators for broad classes of identifiable effects. When implemented through the observed-data density factorization, these estimators can require conditional density estimation and numerical integration, which can be statistically challenging and computationally demanding for continuous or multivariate variables. Prior work on related causal functionals has used sequential regressions and Bayes-rule representations of density ratios to avoid these requirements. Building on established identification and influence-function results, we make these reparameterizations explicit for effects identified under treatment primal fixability, Tian and Pearl's necessary and suff
    
[^301]: LLM生成代码的自动检测：当代模型在函数与类粒度上的对比案例研究

    Automatic Detection of LLM-Generated Code: A Comparative Case Study of Contemporary Models Across Function and Class Granularities

    [https://arxiv.org/abs/2409.01382](https://arxiv.org/abs/2409.01382)

    该研究通过对比四种LLM生成的14,485个函数和11,913个类，发现代码检测中粒度效应的影响远超模型间差异（达8.6倍），函数级与类级检测依赖截然不同的结构特征，且存在依赖粒度的检测性能反转现象。

    

    arXiv:2409.01382v3 公告类型： replace-cross 摘要：采用大型语言模型（LLM）进行代码生成存在将易受攻击的代码引入软件系统的风险。现有检测器面临两个关键局限：缺乏系统性的跨模型验证以及不透明的“黑箱”操作。我们通过对四种不同LLM（GPT-3.5、Claude 3 Haiku、Claude Haiku 4.5和GPT-OSS）生成的代码进行对比研究来解决这些问题。通过分析来自CodeSearchNet数据集的14,485个Python函数和11,913个类，我们使用全部四种LLM生成了相应的代码，并利用可解释的软件度量指标为每种配置训练了CatBoost分类器。我们的分析表明，粒度效应在主导模型差异方面高出8.6倍，且特征重叠可忽略不计，这表明函数级检测和类级检测依赖于本质上互不相关的结构特征。我们发现了关键的粒度依赖性反转现象：虽然现代模型（……

    arXiv:2409.01382v3 Announce Type: replace-cross  Abstract: The adoption of Large Language Models (LLMs) for code generation risks incorporating vulnerable code into software systems. Existing detectors face two critical limitations: a lack of systematic cross-model validation and opaque "black box" operation. We address this through a comparative study of code generated by four distinct LLMs: GPT-3.5, Claude 3 Haiku, Claude Haiku 4.5, and GPT-OSS.   Analyzing 14,485 Python functions and 11,913 classes from the CodeSearchNet dataset, we generated corresponding code with all four LLMs. Using interpretable software metrics, we trained CatBoost classifiers for each configuration. Our analysis reveals that granularity effects dominate model differences by a factor of 8.6, with negligible feature overlap, indicating that function-level and class-level detection rely on fundamentally disjoint structural signatures.   We discover critical granularity-dependent inversions: while modern models (
    
[^302]: 关于平均场机制下多层ResNet梯度下降的全局收敛性研究

    On the Global Convergence of Gradient Descent for multi-layer ResNets in the mean-field regime

    [https://arxiv.org/abs/2110.02926](https://arxiv.org/abs/2110.02926)

    该论文证明了在平均场机制下，当激活函数满足2-齐次或部分1-齐次条件时，多层ResNet训练所对应的梯度流偏微分方程能收敛到全局最小值，表明足够大的ResNet可通过一阶优化方法找到拟合训练数据的全局最优解。

    

    在ResNet中寻找参数的最优配置是一个非凸最小化问题，但当网络过参数化且训练算法运行足够多次迭代时，一阶方法往往能找到全局最优解。我们在平均场机制（mean-field regime）中研究这一现象，在该机制下，网络可以近似地由一个常积分方程（OIE）描述，而ResNet的训练过程则变为一个梯度流偏微分方程（PDE）。在激活函数为$2$-齐次或部分$1$-齐次的条件下，我们证明该梯度流PDE收敛到全局最小值。这一结果表明，如果ResNet足够大，一阶优化方法同样可以找到拟合训练数据的全局最小化器。此外，通过控制泛化误差，我们证明了该梯度流PDE相对于其代价函数是稳定的。

    arXiv:2110.02926v3 Announce Type: replace  Abstract: Finding the optimal configuration of parameters in ResNet is a nonconvex minimization problem, but first-order methods often find the global optimum when the network is overparameterized and the training algorithm is run for sufficiently many iterations. We study this phenomenon in the mean-field regime, where the network can be described approximately by an ordinary integral equation (OIE) and the training process of ResNet becomes a gradient-flow partial differential equation (PDE). Under the condition that the activation function is $2$-homogeneous or partially $1$-homogeneous, we show that this gradient-flow PDE converges to the global minimum. This result suggests that if the ResNet is sufficiently large, first-order optimization methods can find global minimizers that fit the training data as well. Further, by controlling the generalization error, we prove that the gradient-flow PDE is stable with respect to its cost function, 
    
[^303]: 论噪声与曲率之间的相互作用及其对优化和泛化的影响

    On the interplay between noise and curvature and its effect on optimization and generalization

    [https://arxiv.org/abs/1906.07774](https://arxiv.org/abs/1906.07774)

    该论文探讨了损失曲率与梯度噪声的相互作用如何影响优化速度以及泛化间隙的估计，并澄清了费舍尔矩阵、海森矩阵和梯度协方差矩阵之间的区别。

    

    使用随机方法最小化期望损失的速度取决于两个性质：损失的曲率和梯度的方差。以往的大多数研究只关注其中某一个性质，而我们探讨了二者的相互作用如何影响优化速度。此外，由于最终目标是获得良好的泛化性能，我们阐明了曲率和噪声两者都与正确估计泛化间隙相关。我们意识到一些现有工作的局限性源于对这些矩阵的混淆，因此还澄清了费舍尔矩阵、海森矩阵和梯度协方差矩阵之间的区别。

    arXiv:1906.07774v3 Announce Type: replace  Abstract: The speed at which one can minimize an expected loss using stochastic methods depends on two properties: the curvature of the loss and the variance of the gradients. While most previous works focus on one or the other of these properties, we explore how their interaction affects optimization speed. Further, as the ultimate goal is good generalization performance, we clarify how both curvature and noise are relevant to properly estimate the generalization gap. Realizing that the limitations of some existing works stems from a confusion between these matrices, we also clarify the distinction between the Fisher matrix, the Hessian, and the covariance matrix of the gradients.
    
[^304]: OpinionGPT: 模拟显性偏见的指令调整大型语言模型(LLMs)

    OpinionGPT: Modelling Explicit Biases in Instruction-Tuned LLMs. (arXiv:2309.03876v1 [cs.CL])

    [http://arxiv.org/abs/2309.03876](http://arxiv.org/abs/2309.03876)

    OpinionGPT是一个指令调整大型语言模型(LLMs)的web演示，用户可以选择各种偏见并进行比较，它意在使模型的偏见显性和透明化。

    

    最近，指令调整大型语言模型(LLMs)展示了生成与自然语言指令相匹配的回应的显著能力。然而，一个开放的研究问题涉及训练模型和它们的回应中固有的偏见。例如，如果用于调整LLM的数据主要由具有特定政治偏见的人编写，我们可能会期望生成的回答也共享这种偏见。目前的研究工作旨在除去这样的模型偏见，或抑制可能有偏见的回答。通过这个演示，我们对指令调整中的偏见持有不同的观点：我们的目标不是抑制它们，而是使它们显性和透明。为此，我们提供了OpinionGPT，一个网络演示，用户可以提问并选择所有他们希望调查的偏见。该演示将使用在代表每个选择偏见的文本上进行微调的模型来回答这个问题，从而实现并排比较。为了训练基础模型，我们选取了11个...

    Instruction-tuned Large Language Models (LLMs) have recently showcased remarkable ability to generate fitting responses to natural language instructions. However, an open research question concerns the inherent biases of trained models and their responses. For instance, if the data used to tune an LLM is dominantly written by persons with a specific political bias, we might expect generated answers to share this bias. Current research work seeks to de-bias such models, or suppress potentially biased answers. With this demonstration, we take a different view on biases in instruction-tuning: Rather than aiming to suppress them, we aim to make them explicit and transparent. To this end, we present OpinionGPT, a web demo in which users can ask questions and select all biases they wish to investigate. The demo will answer this question using a model fine-tuned on text representing each of the selected biases, allowing side-by-side comparison. To train the underlying model, we identified 11 
    

