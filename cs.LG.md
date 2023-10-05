# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decision ConvFormer: Local Filtering in MetaFormer is Sufficient for Decision Making.](http://arxiv.org/abs/2310.03022) | Decision ConvFormer提出了一种新的动作序列预测器，通过使用本地卷积过滤来捕捉强化学习数据集中的局部关联，同时在各个标准RL基准上取得了最先进的性能。 |
| [^2] | [Understanding In-Context Learning in Transformers and LLMs by Learning to Learn Discrete Functions.](http://arxiv.org/abs/2310.03016) | 这项工作通过实验发现，变压器可以学习各种实值函数的基于梯度的学习算法，但对于更复杂的任务性能下降。另外，该研究还探讨了这些能力在基于注意力模型中的限制程度以及推广到预训练的大型语言模型（LLM）的可行性。 |
| [^3] | [SemiReward: A General Reward Model for Semi-supervised Learning.](http://arxiv.org/abs/2310.03013) | SemiReward是一个通用奖励模型，通过预测奖励分数来评估和过滤高质量的伪标签，可以应用于各种半监督学习任务，并在实验中取得了显著的成果。 |
| [^4] | [High-dimensional SGD aligns with emerging outlier eigenspaces.](http://arxiv.org/abs/2310.03010) | 本研究通过研究训练动态和经验海森矩阵以及梯度矩阵的谱的联合演化，证明了在高维混合和多层神经网络的分类任务中，SGD轨迹与海森矩阵和梯度矩阵的新兴低秩异常特征空间吻合。在多层设置中，这种对齐会在每一层发生，并且在收敛到亚优分类器时会表现出秩缺乏。 |
| [^5] | [Soft Convex Quantization: Revisiting Vector Quantization with Convex Optimization.](http://arxiv.org/abs/2310.03004) | 本文提出了软凸量化（SCQ）作为向量量化（VQ）的替代方法，通过解决量化输入的码书向量的最优凸组合问题，缓解了VQ面临的实际挑战。 |
| [^6] | [Learning characteristic parameters and dynamics of centrifugal pumps under multi-phase flow using physics-informed neural networks.](http://arxiv.org/abs/2310.03001) | 本文提出了一种基于物理信息神经网络（PINNs）的机器学习模型，用于估计离心泵在多相流下的特性参数和动力学。 |
| [^7] | [ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models.](http://arxiv.org/abs/2310.02998) | ECoFLaP提出了一种高效的粗到细的逐层剪枝方法，解决了大型视觉语言模型在压缩和部署时的计算和能耗问题。 |
| [^8] | [IBCL: Zero-shot Model Generation for Task Trade-offs in Continual Learning.](http://arxiv.org/abs/2310.02995) | IBCL提出了一种用于连续学习中任务权衡的零样本模型生成方法，通过更新知识库并利用模型参数分布的凸包形式，实现不同任务性能之间的权衡偏好。 |
| [^9] | [Multiple Physics Pretraining for Physical Surrogate Models.](http://arxiv.org/abs/2310.02994) | 多物理学预训练是一种用于物理代理建模的自回归预训练方法，通过训练大型代理模型同时预测多个异构物理系统的动力学，学习在不同物理任务中广泛适用的特征。实验证明，单个MPP预训练的变换器可以在所有预训练子任务上与或超过特定任务的基准结果，无需微调，并且在下游任务中，微调MPP训练的模型相较于从头训练的模型，对新物理的预测结果更准确。 |
| [^10] | [xVal: A Continuous Number Encoding for Large Language Models.](http://arxiv.org/abs/2310.02989) | xVal是一种连续数字编码方案，通过使用单个标记来表示任何实数。与现有的数字编码方案相比，xVal更加高效，并且在泛化性能上表现更好。 |
| [^11] | [Variance Reduced Halpern Iteration for Finite-Sum Monotone Inclusions.](http://arxiv.org/abs/2310.02987) | 提出了使用方差减少的 Halpern 迭代来优化有限和单调包含问题的求解过程，具有更好的复杂度保证。 |
| [^12] | [Exploring the Impact of Disrupted Peer-to-Peer Communications on Fully Decentralized Learning in Disaster Scenarios.](http://arxiv.org/abs/2310.02986) | 在灾难场景中，完全去中心化学习可以帮助解决通信基础设施中断或不可用导致的传统集中式学习任务无法进行的问题。 |
| [^13] | [Scaling Laws for Associative Memories.](http://arxiv.org/abs/2310.02984) | 本文研究了应用于联想记忆中的缩放定律，通过高维矩阵和嵌入的外积来模拟内层Transformer语言模型。作者推导出了与样本数量和参数大小相关的精确缩放定律，并验证了理论结果的有效性。同时，作者还通过大量实验展示了存储记忆关联的细粒度可视化。 |
| [^14] | [Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors.](http://arxiv.org/abs/2310.02980) | 本文研究表明使用随机初始化会导致对架构差异的严重高估，而使用标准消噪目标进行预训练可以在多种架构上实现显著的性能提升，并将Transformers与状态空间模型之间的差距缩小到很小。与之前的研究不同的是，我们发现当正确预训练时，普通的Transformers在Long Range Arena上的性能与S4相匹配，并且在PathX-256任务上改进了SSMs的最佳结果20个百分点。 |
| [^15] | [T$^3$Bench: Benchmarking Current Progress in Text-to-3D Generation.](http://arxiv.org/abs/2310.02977) | T$^3$Bench是第一个综合的文本到3D基准测试，它包含了多个复杂程度的文本提示，并引入了两个自动度量标准来评估生成的3D场景的主观质量和文本对齐性能。 |
| [^16] | [Towards Fully Adaptive Regret Minimization in Heavy-Tailed Bandits.](http://arxiv.org/abs/2310.02975) | 本文研究了在重尾波段问题中完全自适应的遗憾最小化，提出了随机自适应重尾波段问题，并证明了适应性算法相对于标准设置会有更高的遗憾。 |
| [^17] | [Fast, Expressive SE$(n)$ Equivariant Networks through Weight-Sharing in Position-Orientation Space.](http://arxiv.org/abs/2310.02970) | 该论文通过在位置-方向空间中共享权重，提出了一种快速、表达力强的SE$(n)$等变网络。他们基于同态空间理论，推导出几何优化的边属性，并将权重共享形式化为对等处理相同点对的消息函数。他们在处理3D点云时，开发了一个高效的等变群卷积网络，并选择了$\mathbb{R}^3 {\times} S^2$作为最佳的处理空间。 |
| [^18] | [Dual Conic Proxies for AC Optimal Power Flow.](http://arxiv.org/abs/2310.02969) | 本文提出了一种基于双圆锥代理的方法来求解交流最优功率流问题，并通过自监督学习方案来辅助训练，实验证明了该方法的效率和可扩展性。 |
| [^19] | [Co-modeling the Sequential and Graphical Route for Peptide.](http://arxiv.org/abs/2310.02964) | 本论文提出了一种肽合模式化方法，使用对比学习框架来增强从顺序和图形模型中学到的表示的相互信息，以提高肽的判别性能。 |
| [^20] | [Credit card score prediction using machine learning models: A new dataset.](http://arxiv.org/abs/2310.02956) | 本研究探索了利用机器学习模型对信用卡违约进行预测的方法，并提出了一个新的信用卡评分数据集。实验结果表明，多层感知器（MLP）模型在预测性能上表现最佳。 |
| [^21] | [A Fisher-Rao gradient flow for entropy-regularised Markov decision processes in Polish spaces.](http://arxiv.org/abs/2310.02951) | 该论文研究了在Polish空间中的熵正则化Markov决策过程上的Fisher-Rao策略梯度流的全局收敛性和指数收敛性，并证明了梯度流在梯度评估方面的稳定性，为自然策略梯度流的性能提供了洞见。 |
| [^22] | [Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models.](http://arxiv.org/abs/2310.02949) | 该论文探讨了阴影对齐这种新的攻击方式，通过调整少量恶意示例，安全对齐的语言模型可以被轻松颠覆生成有害的内容，同时仍然可以正确响应常规查询。 |
| [^23] | [HappyFeat -- An interactive and efficient BCI framework for clinical applications.](http://arxiv.org/abs/2310.02948) | HappyFeat是一种面向临床应用的交互和高效的BCI框架，通过一个方便的GUI和参数自动化的帮助，使得基于运动想象的BCI实验更加容易，并能在时间受限的环境中实现良好的性能。 |
| [^24] | [Online Constraint Tightening in Stochastic Model Predictive Control: A Regression Approach.](http://arxiv.org/abs/2310.02942) | 本文提出了一种数据驱动方法，用于在线学习随机模型预测控制中的约束加紧参数。通过将约束加紧参数选择问题重新表述为二进制回归问题，并利用高斯过程模型进行学习，实现了在线学习约束加紧参数的目标。 |
| [^25] | [Hoeffding's Inequality for Markov Chains under Generalized Concentrability Condition.](http://arxiv.org/abs/2310.02941) | 本文研究了在广义可集中条件下的马尔可夫链的Hoeffding不等式，拓展了现有的马尔可夫链Hoeffding型不等式的应用范围。通过应用该框架到机器学习领域，我们得到了几个非渐近分析的结果。 |
| [^26] | [Assessing Large Language Models on Climate Information.](http://arxiv.org/abs/2310.02932) | 本研究提出了一个基于科学传播原则的综合评估框架，评估了大规模语言模型在气候变化信息中的表现，能够在回答气候变化主题方面提供细粒度的分析。 |
| [^27] | [Graph data modelling for outcome prediction in oropharyngeal cancer patients.](http://arxiv.org/abs/2310.02931) | 本研究首次使用基于计算机断层扫描的放射学特征，提出了一种用于咽喉癌患者预测结果的患者超图网络（PHGN）。研究还将模型扩展到进行事件发生时间分析，并与GNN和基准线性模型进行比较。 |
| [^28] | [Optimal Transport with Adaptive Regularisation.](http://arxiv.org/abs/2310.02925) | OTARI是一种新的最优传输形式，它通过对每个点的质量施加约束来解决全局约束造成的不平衡问题，在领域适应中具有重要作用。 |
| [^29] | [Enhancing Ayurvedic Diagnosis using Multinomial Naive Bayes and K-modes Clustering: An Investigation into Prakriti Types and Dosha Overlapping.](http://arxiv.org/abs/2310.02920) | 本研究提出使用多项式朴素贝叶斯和K-modes聚类来增强阿育吠陀诊断中的普拉克里蒂类型和Dosha重叠的识别。通过将Dosha分类为7个类别，包括重叠的Dosha类别，可以提高诊断模型的准确性和真实性。 |
| [^30] | [Attention-based Multi-task Learning for Base Editor Outcome Prediction.](http://arxiv.org/abs/2310.02919) | 基于注意力的多任务学习模型可以加速基因编辑设计过程，并提高编辑结果预测的准确性。 |
| [^31] | [ELUQuant: Event-Level Uncertainty Quantification in Deep Inelastic Scattering.](http://arxiv.org/abs/2310.02913) | ELUQuant是一种能够在深度非弹性散射中对事件级别的不确定性进行量化的方法，利用基于物理的贝叶斯神经网络和归一化流近似计算后验分布，能够提供详细的不确定性描述。这为决策制定和减少真实不准确性提供了宝贵的帮助。 |
| [^32] | [Use Your INSTINCT: INSTruction optimization usIng Neural bandits Coupled with Transformers.](http://arxiv.org/abs/2310.02905) | 该论文提出了一种使用神经探测器和转换器优化指令的方法，以提高大型语言模型的性能。 |
| [^33] | [Spline-based neural network interatomic potentials: blending classical and machine learning models.](http://arxiv.org/abs/2310.02904) | 本研究引入了一种新的基于样条函数的神经网络势(s-NNP)框架，将简单性的s-MEAM原子间势与神经网络的灵活性相结合，用于构建高质量的IPs。该框架能够突破经典和ML IPs之间的界限，并通过关键架构变化提供更好的性能。同时，使用样条滤波器来编码原子环境，可以产生容易解释的嵌入层。 |
| [^34] | [FroSSL: Frobenius Norm Minimization for Self-Supervised Learning.](http://arxiv.org/abs/2310.02903) | FroSSL是一种基于Frobenius范数最小化的自监督学习方法，通过最小化协方差Frobenius范数来避免信息崩溃，同时通过最小化均方差来实现数据增强的不变性，相比其他SSL方法，FroSSL收敛更快，并且这种快速收敛是由于FroSSL影响嵌入协方差矩阵的特征值所致。 |
| [^35] | [Searching for High-Value Molecules Using Reinforcement Learning and Transformers.](http://arxiv.org/abs/2310.02902) | 通过使用强化学习和Transformer，我们提出了一种新的基于RL的分子设计算法（ChemRLformer），并在25个分子设计任务中进行了综合分析，包括计算复杂的蛋白质对接模拟。我们发现了分子设计领域的独特见解，并展示了ChemRLformer相对于之前的工作更为简单且实现了最先进的性能。 |
| [^36] | [Recovery of Training Data from Overparameterized Autoencoders: An Inverse Problem Perspective.](http://arxiv.org/abs/2310.02897) | 本研究从逆问题的角度研究了从过参数自编码器模型恢复训练数据的问题，并提出了一种实际方法，该方法利用训练好的自编码器来定义正则化器并通过迭代计算处理未知的退化操作符。实验结果表明，该方法在自编码器恢复训练数据方面具有显著的优势。 |
| [^37] | [CoLiDE: Concomitant Linear DAG Estimation.](http://arxiv.org/abs/2310.02895) | 本论文提出了CoLiDE算法用于学习线性DAG，该算法使用了一个新的凸评分函数，结合了标度的共同估计，从而有效地将稀疏参数与外生噪声水平分离。 |
| [^38] | [Something for (almost) nothing: Improving deep ensemble calibration using unlabeled data.](http://arxiv.org/abs/2310.02885) | 本文提出了一种简单的方法，在小训练数据情况下利用无标签数据改进深度合奏模型校准，实验证明该方法在测试集上具有较低的负对数似然和高的合奏多样性，比标准方法更好。 |
| [^39] | [Stationarity without mean reversion: Improper Gaussian process regression and improper kernels.](http://arxiv.org/abs/2310.02877) | 本论文展示了使用具有无限方差的不恰当高斯过程先验来定义静止但不均值回归过程的可能性，并引入了一类特殊的不恰当核函数来实现此目的。 |
| [^40] | [Recent Methodological Advances in Federated Learning for Healthcare.](http://arxiv.org/abs/2310.02874) | 最近的医疗保健领域联邦学习研究提出了新的方法学，用于解决医疗保健数据的挑战，如孤立数据、类别不平衡、缺失数据、分布转移和非标准化变量。 |
| [^41] | [Stable and Interpretable Deep Learning for Tabular Data: Introducing InterpreTabNet with the Novel InterpreStability Metric.](http://arxiv.org/abs/2310.02870) | 我们引入了InterpreTabNet，通过改进的注意模块和TabNet架构，提高了表格数据的分类准确度和解释性。我们还提出了一种新的评价指标InterpreStability，用于量化模型的解释稳定性。 |
| [^42] | [Harmonic Control Lyapunov Barrier Functions for Constrained Optimal Control with Reach-Avoid Specifications.](http://arxiv.org/abs/2310.02869) | 该论文介绍了谐波控制李亚普诺夫障碍函数（harmonic CLBF），它可以在约束控制问题中解决避障要求，通过最大化系统动力学与谐波CLBF最陡下降方向的内积来选择控制输入，从而显著降低进入不安全区域的风险并提高进入目标区域的概率。 |
| [^43] | [Estimation of Models with Limited Data by Leveraging Shared Structure.](http://arxiv.org/abs/2310.02864) | 本文提出了一种利用共享结构进行有限数据模型估计的方法，通过估计潜在的低维参数空间，并在该空间内产生精确的参数估计。这种方法适用于具有多个系统且每个系统的数据量很少的情况下，提供有限样本子空间估计误差保证。 |
| [^44] | [Conformal Predictions for Longitudinal Data.](http://arxiv.org/abs/2310.02863) | 这篇论文介绍了一种新颖的基于分布的共形预测算法LPCI，用于处理长期数据。通过将剩余数据建模为分位数固定效应回归问题，并使用训练好的回归器构建预测区间，LPCI实现了有效的横截面覆盖，并优于现有的基准模型。 |
| [^45] | [A novel asymmetrical autoencoder with a sparsifying discrete cosine Stockwell transform layer for gearbox sensor data compression.](http://arxiv.org/abs/2310.02862) | 这篇论文提出了一种信号自适应的非对称自编码器，使用离散余弦Stockwell变换层进行齿轮传感器数据压缩。通过引入可训练的滤波器和硬阈值化层，该方法能够提高数据重构的准确性，并且仅需要少量数据集进行训练。 |
| [^46] | [Rayleigh Quotient Graph Neural Networks for Graph-level Anomaly Detection.](http://arxiv.org/abs/2310.02861) | 《Rayleigh Quotient Graph Neural Networks用于图级异常检测的研究》提出使用Rayleigh Quotient作为驱动因素，通过探索图的固有光谱特征来实现图级异常检测。 |
| [^47] | [Multi-Domain Causal Representation Learning via Weak Distributional Invariances.](http://arxiv.org/abs/2310.02854) | 本文提出了一种通过弱分布不变性进行多领域因果表示学习的方法，证明了融入这种不变性的自编码器能够可靠地识别出稳定的变量集合。 |
| [^48] | [Out-of-Distribution Detection by Leveraging Between-Layer Transformation Smoothness.](http://arxiv.org/abs/2310.02832) | 本文提出了一种通过利用神经网络中间层变换的平滑性来检测带外数据的方法(BLOOD),该方法适用于没有训练数据访问权限的预训练模型，并在Transformer网络上的文本分类任务中取得了良好的效果。 |
| [^49] | [Learning to Scale Logits for Temperature-Conditional GFlowNets.](http://arxiv.org/abs/2310.02823) | 这项研究提出了一种名为LSL-GFN的新型架构设计，可以大大加速温度条件下GFlowNets的训练，从而提高GFlowNets的探索和利用能力。 |
| [^50] | [Time-Series Classification in Smart Manufacturing Systems: An Experimental Evaluation of State-of-the-Art Machine Learning Algorithms.](http://arxiv.org/abs/2310.02812) | 本研究通过严格实验评估了智能制造系统中最先进的机器学习和深度学习算法在时间序列分类任务中的性能，填补了该领域的研究空白。 |
| [^51] | [A Deep Instance Generative Framework for MILP Solvers Under Limited Data Availability.](http://arxiv.org/abs/2310.02807) | 本文提出了G2MILP，这是第一个用于MILP实例的深度生成框架，它可以生成新颖而逼真的MILP实例。 |
| [^52] | [A Data-facilitated Numerical Method for Richards Equation to Model Water Flow Dynamics in Soil.](http://arxiv.org/abs/2310.02806) | 本文提出了一种基于数据的Richards方程数值解法，称为D-GRW方法，通过整合自适应线性化方案、神经网络和全局随机游走，在有限体积离散化框架下，能够产生具有收敛性保证的Richards方程数值解，并在精度和质量守恒性能方面表现卓越。 |
| [^53] | [DOMINO: A Dual-System for Multi-step Visual Language Reasoning.](http://arxiv.org/abs/2310.02804) | 本文提出了一种用于多步骤多模态推理的双系统，其中一个系统用于提取视觉信息，另一个系统用于有意推理。这种方法可以避免模型在一个步骤回答复杂问题或被转换文本中的误导信息困扰。 |
| [^54] | [MAD Max Beyond Single-Node: Enabling Large Machine Learning Model Acceleration on Distributed Systems.](http://arxiv.org/abs/2310.02784) | 该研究提出了一个性能建模框架，在分布式系统上实现了大规模机器学习模型的加速，获得了2.24倍和5.27倍的吞吐量提升潜力。 |
| [^55] | [Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design.](http://arxiv.org/abs/2310.02782) | 通过对抗性环境设计，我们提出了一种通用强化学习算法，通过元学习更新规则和自动生成课程来提高算法的泛化性能，并引入了一种新的遗憾近似方法，名为算法遗憾（AR）。 |
| [^56] | [Expected flow networks in stochastic environments and two-player zero-sum games.](http://arxiv.org/abs/2310.02779) | 该论文提出了预期流网络（EFlowNets）和对抗流网络（AFlowNets），分别应用于随机环境和双人零和游戏中。在随机任务中，EFlowNets表现优于其他方法，在双人游戏中，AFlowNets在自我对弈中找到了80%以上的最佳动作，并在竞赛中超过了AlphaZero。 |
| [^57] | [Graph Neural Networks and Time Series as Directed Graphs for Quality Recognition.](http://arxiv.org/abs/2310.02774) | 这篇论文研究了图神经网络在时间序列上的应用，将时间序列视为有向图来编码时间依赖性，并开发了两个几何深度学习模型，分别用于监督分类和信号重构，在质量识别问题上取得了良好效果。 |
| [^58] | [Kernel-based function learning in dynamic and non stationary environments.](http://arxiv.org/abs/2310.02767) | 本文研究了在非平稳分布下的基于核的岭回归问题，在环境监测和传感器重构等探索-利用问题中具有重要应用价值。 |
| [^59] | [Comparative Study and Framework for Automated Summariser Evaluation: LangChain and Hybrid Algorithms.](http://arxiv.org/abs/2310.02759) | 本研究关注自动摘要评估的比较研究和框架，通过使用LangChain和混合算法对PDF文档进行摘要和提取关键信息，以确定用户对摘要内容的理解程度如何。 |
| [^60] | [MUNCH: Modelling Unique 'N Controllable Heads.](http://arxiv.org/abs/2310.02753) | 本论文提出了一种方法，可以自动生成质量高、多样性强、可控制的逼真三维人头，具有可解释的网络设计。方法包括几何生成器、渲染图生成器和颜色变换模型。同时还引入了独特性和新颖性的量化指标。 |
| [^61] | [Fair Feature Selection: A Comparison of Multi-Objective Genetic Algorithms.](http://arxiv.org/abs/2310.02752) | 本文比较了两种基于多目标优化方法的公平特征选择遗传算法，旨在同时最大化分类器的准确性和公平性。这是第一项系统性地进行比较的研究。 |
| [^62] | [SHOT: Suppressing the Hessian along the Optimization Trajectory for Gradient-Based Meta-Learning.](http://arxiv.org/abs/2310.02751) | 本文提出了一种名为SHOT的算法，通过抑制梯度优化轨迹中的海森矩阵来改进基于梯度的元学习，在标准的少样本学习任务中取得了优于基线模型的效果。 |
| [^63] | [SALSA: Semantically-Aware Latent Space Autoencoder.](http://arxiv.org/abs/2310.02744) | SALSA提出了一种语义感知的潜空间自编码器（SALSA），通过在自编码器中引入对比任务，专门设计用于学习分子之间的图对图相似性。 |
| [^64] | [Reward Model Ensembles Help Mitigate Overoptimization.](http://arxiv.org/abs/2310.02743) | 本研究通过探究奖励模型集成和保守优化目标的效果，对减轻奖励模型过度优化进行了系统研究。 |
| [^65] | [Comparative Analysis of Imbalanced Malware Byteplot Image Classification using Transfer Learning.](http://arxiv.org/abs/2310.02742) | 本文通过比较分析了不平衡恶意软件图像分类的六种模型性能，发现类别不平衡程度越高，收敛所需的迭代次数越少，不同模型的性能差异也越大。同时，研究发现ResNet50、EfficientNetB0和DenseNet169可以很好地处理不平衡和平衡的数据，对于恶意软件检测器具有较高的精度。 |
| [^66] | [Extracting Rules from Event Data for Study Planning.](http://arxiv.org/abs/2310.02735) | 本研究利用事件数据分析高等教育学生的学习路径，通过决策树模型生成数据驱动的规则建议，并用于学习规划。研究发现所选课程序列特征对学业表现有较好解释，为制定更适应性的学习计划提供了思路。 |
| [^67] | [Functional trustworthiness of AI systems by statistically valid testing.](http://arxiv.org/abs/2310.02727) | 作者认为欧盟AI法案对AI系统的质量保证方式存在不足，并指出基于统计学有效测试及准确定义应用是确保AI系统功能可信度的核心。 |
| [^68] | [End-to-End Training of a Neural HMM with Label and Transition Probabilities.](http://arxiv.org/abs/2310.02724) | 本研究提出了一种新的端到端神经网络训练方法，通过显式建模和学习隐藏状态之间的转移概率，而不是隐含地编码持续时间统计的空标签。虽然转移模型的训练不会改善识别性能，但对齐质量有积极影响。 |
| [^69] | [Leveraging Temporal Graph Networks Using Module Decoupling.](http://arxiv.org/abs/2310.02721) | 本研究通过解耦时间图网络的核心模块并使用最少的可学习参数，提出了一种轻量级解耦时间图网络 (LDTGN)。在学习动态图的过程中，LDTGN表现出与之前方法可比甚至领先的结果，并且具有显著更高的吞吐量。 |
| [^70] | [Understanding Pan-Sharpening via Generalized Inverse.](http://arxiv.org/abs/2310.02718) | 通过研究广义逆理论，本文提出了一种新的全色增强算法，该算法基于简单矩阵方程描述全色增强问题，并探讨解的条件和光谱、空间分辨率的获取。通过引入降采样增强方法，我们得到了与分量替代和多分辨率分析方法相对应的广义逆矩阵表达式，并提出了一个新的模型先验来解决全色增强中的理论误差问题。 |
| [^71] | [Online Clustering of Bandits with Misspecified User Models.](http://arxiv.org/abs/2310.02717) | 本文介绍了在用户模型错误的情况下的聚类强化学习问题，并提出了两个鲁棒的聚类强化学习算法，以解决用户模型偏差的挑战。 |
| [^72] | [scHyena: Foundation Model for Full-Length Single-Cell RNA-Seq Analysis in Brain.](http://arxiv.org/abs/2310.02713) | scHyena是一个基于Transformer架构的模型，称为单细胞Hyena(scHyena)，旨在处理大脑中的全长scRNA-seq数据，并提高分析的准确性。 |
| [^73] | [ED-NeRF: Efficient Text-Guided Editing of 3D Scene using Latent Space NeRF.](http://arxiv.org/abs/2310.02712) | ED-NeRF 提出了一种高效的 3D 场景编辑方法，通过将场景嵌入到潜空间中，得到更快速且更易于编辑的 NeRF 骨干。 |
| [^74] | [Local Search GFlowNets.](http://arxiv.org/abs/2310.02710) | 本文提出使用局部搜索训练GFlowNets，通过破坏和重构的方式探索局部邻域，分别由反向和正向策略引导，使得样本偏向高奖励解决方案。 |
| [^75] | [Tackling Hybrid Heterogeneity on Federated Optimization via Gradient Diversity Maximization.](http://arxiv.org/abs/2310.02702) | 本文探讨了混合异构性如何影响联邦优化，并提出了一种通过最大化梯度多样性来减轻混合异构性负面影响的方法。 |
| [^76] | [Exploring Federated Optimization by Reducing Variance of Adaptive Unbiased Client Sampling.](http://arxiv.org/abs/2310.02698) | 本文通过减少自适应无偏客户采样方差，探索了联邦优化中的一系列自适应客户采样技术，并提出了一种名为K-Vib的新型采样器，显著提高了联邦学习性能。 |
| [^77] | [Probabilistic Block Term Decomposition for the Modelling of Higher-Order Arrays.](http://arxiv.org/abs/2310.02694) | 本研究提出了一种高效的变分贝叶斯概率块项分解（pBTD）方法，适用于高阶数组的建模，通过使用von-Mises Fisher矩阵分布来实现正交性约束。实验结果表明，pBTD在噪声数据和模型顺序量化方面具有良好的性能。 |
| [^78] | [Robust Ocean Subgrid-Scale Parameterizations Using Fourier Neural Operators.](http://arxiv.org/abs/2310.02691) | 本文使用傅里叶神经算子开发了海洋子区尺度稳健参数化方法，展示了其准确性和普适性，为解决气候模拟中长期预测误差的问题提供了潜在解决方案。 |
| [^79] | [Diffusion Generative Flow Samplers: Improving learning signals through partial trajectory optimization.](http://arxiv.org/abs/2310.02679) | 这项工作介绍了一种名为扩散生成流采样器（DGFS）的采样框架，通过将学习过程分解为短的部分轨迹段，实现从难以处理的高维密度函数中进行采样。它通过利用中间的学习信号和非策略探索能力来改善学习信号的分配问题。 |
| [^80] | [PostRainBench: A comprehensive benchmark and a new model for precipitation forecasting.](http://arxiv.org/abs/2310.02676) | PostRainBench是一个全面的降水预测基准，结合AI后处理技术和传统的数值天气预报方法，能够增强准确性并解决复杂的降水预测挑战。 |
| [^81] | [On Memorization in Diffusion Models.](http://arxiv.org/abs/2310.02664) | 本论文研究了扩散模型的记忆化行为，发现记忆化倾向于在较小的数据集上发生。通过定义有效模型记忆化 (EMM) 这一指标，量化了数据分布和模型配置对记忆化行为的影响。 |
| [^82] | [A Study of Quantisation-aware Training on Time Series Transformer Models for Resource-constrained FPGAs.](http://arxiv.org/abs/2310.02654) | 本研究探索了对时间序列Transformer模型进行量化感知训练的方法，并提出了一种新颖的自适应量化方案，通过匹配量化方案与实际数据分布，可以降低计算开销并保持可接受的精度。此外，该方法在应用于现实世界数据和混合精度量化时表现出鲁棒性。这些发现为模型量化和部署决策提供了参考，并推进了量化技术的发展。 |
| [^83] | [Hire When You Need to: Gradual Participant Recruitment for Auction-based Federated Learning.](http://arxiv.org/abs/2310.02651) | GPS-AFL是一种基于拍卖的联邦学习渐进式参与者选择方案，通过在多轮训练中逐渐选择数据所有者，解决了冷启动问题和选择偏差对联邦学习的影响。 |
| [^84] | [Foundation Reinforcement Learning: towards Embodied Generalist Agents with Foundation Prior Assistance.](http://arxiv.org/abs/2310.02635) | 本研究提出了一种基于具身基础先验的基础强化学习框架，通过加速训练过程来提高样本效率。 |
| [^85] | [Multi-rules mining algorithm for combinatorially exploded decision trees with modified Aitchison-Aitken function-based Bayesian optimization.](http://arxiv.org/abs/2310.02633) | 该论文提出了两种算法(MAABO-MT和GS-MRM)，分别在所有可能的树中高性能估计构建树，仅提取可靠且不相似的规则。 |
| [^86] | [Generative Modeling of Regular and Irregular Time Series Data via Koopman VAEs.](http://arxiv.org/abs/2310.02619) | 本文介绍了一种基于Koopman VAEs的新型生成框架，可以用于生成规则和非规则时间序列数据，解决了GANs训练不稳定和模式崩溃的问题，并通过利用谱工具对线性映射的特征值施加约束，实现了领域知识的整合和对定性行为和稳定性的研究。 |
| [^87] | [Analyzing and Improving OT-based Adversarial Networks.](http://arxiv.org/abs/2310.02611) | 本文分析和改进了基于OT的对抗网络，首先在一个统一的框架中统一了这些方法，然后通过全面分析展示了各组件在训练中的作用，最后提出了一个简单但新颖的方法以改进最优生成模型，该方法通过逐步调整生成分布逐渐使其与数据分布对齐 |
| [^88] | [Learning adjacency matrix for dynamic graph neural network.](http://arxiv.org/abs/2310.02606) | 该论文介绍了一种用于动态图神经网络的学习邻接矩阵的方法。通过引入一个特殊设计的编码器块来学习缺失的时空连接，将其丰富后的块邻接矩阵输入到图神经网络中，以捕捉网络的复杂时空拓扑。 |
| [^89] | [Multi-Agent Reinforcement Learning for Power Grid Topology Optimization.](http://arxiv.org/abs/2310.02605) | 本文提出了一种用于电网拓扑优化的分层多智能体强化学习（MARL）框架，有效处理随着网络增长而扩大的大型行动空间。实验表明，该框架在性能上与单一智能体强化学习方法相当，并比较了不同的RL算法和不同的高阶智能体策略。 |
| [^90] | [ViT-ReciproCAM: Gradient and Attention-Free Visual Explanations for Vision Transformer.](http://arxiv.org/abs/2310.02588) | 本文提出了一种新的方法，名为ViT-ReciproCAM，用于解释Vision Transformer中的预测过程和调试预测错误。该方法不依赖梯度和注意力矩阵，并使用令牌遮罩和新的层输出来利用激活的令牌与目标类别的网络预测之间的关联。 |
| [^91] | [Machine Learning-Enabled Precision Position Control and Thermal Regulation in Advanced Thermal Actuators.](http://arxiv.org/abs/2310.02583) | 本论文介绍了一种基于机器学习的恒功率开环控制器，在没有外部传感器的情况下，实现了对尼龙人工肌肉的精确位置控制。通过构建神经网络，将期望位移转化为所需功率，神经控制器在经过精心训练后能够适应各种类型的热人工肌肉。 |
| [^92] | [Online Estimation and Inference for Robust Policy Evaluation in Reinforcement Learning.](http://arxiv.org/abs/2310.02581) | 该论文提出了一种针对鲁棒策略评估的在线估计和推断方法，在解决异常值污染和重尾奖励的问题方面引入了鲁棒统计学的概念。此外，还提出了一种完全在线的统计推断过程，并建立了估计量的极限分布。 |
| [^93] | [On the Stability of Expressive Positional Encodings for Graph Neural Networks.](http://arxiv.org/abs/2310.02579) | 本研究针对图神经网络中使用拉普拉斯特征向量作为位置编码面临的非唯一性和不稳定性问题，提出了稳定且表达丰富的位置编码方法（SPE），该方法通过利用特征值对特征空间进行"软分割"，在未见过的图结构上表现出良好的泛化能力。 |
| [^94] | [AdaMerging: Adaptive Model Merging for Multi-Task Learning.](http://arxiv.org/abs/2310.02575) | AdaMerging通过自适应学习模型合并的系数，以更有效地合并预训练模型来解决多任务学习中存在的性能下降问题。 |
| [^95] | [Improving Knowledge Distillation with Teacher's Explanation.](http://arxiv.org/abs/2310.02572) | 本论文提出了一种新颖的知识解释蒸馏（KED）框架，允许学生从教师的解释中学习，并扩展了KED的应用范围，以提高卷积神经网络的性能和处理有限训练数据的能力。 |
| [^96] | [Stand for Something or Fall for Everything: Predict Misinformation Spread with Stance-Aware Graph Neural Networks.](http://arxiv.org/abs/2310.02568) | 使用立场感知的图神经网络（stance-aware GNN）预测谣言传播。与没有用户立场的GNN相比，该模型在真实数据集上的表现优于32.65%的基准模型。注意权重表明用户的反对立场对邻居行为的影响更大，可以作为社会纠正措施阻止谣言传播。 |
| [^97] | [Improving Automatic VQA Evaluation Using Large Language Models.](http://arxiv.org/abs/2310.02567) | 提出使用大型语言模型改进自动视觉问答（VQA）评估的方法，将VQA评估格式化为回答评分任务，通过指令调整大型语言模型在准确度上评分候选答案，证明该方法与人类判断相关性优于现有度量方法。 |
| [^98] | [Practical, Private Assurance of the Value of Collaboration.](http://arxiv.org/abs/2310.02563) | 该论文研究了两方在数据集上合作前如何保证合作的价值。通过构建基于全同态加密方案和标签差分隐私的交互式协议，该研究提供了一个实用的、私密的解决方案。最终的结果是确保合作前双方的模型和数据集不会被透露。 |
| [^99] | [Semi-Federated Learning: Convergence Analysis and Optimization of A Hybrid Learning Framework.](http://arxiv.org/abs/2310.02559) | 该论文提出了一种半联邦学习（SemiFL）范式，以在基站和设备之间进行集中式学习（CL）和无线联邦学习（FL）的混合实现。通过集成空中计算和非正交多址接入传输，提高了通信效率，并进行了收敛分析和优化。 |
| [^100] | [Generalization in diffusion models arises from geometry-adaptive harmonic representation.](http://arxiv.org/abs/2310.02557) | 通过分析基于分数的反向扩散算法生成的高质量样本的研究结果，我们发现尽管存在维度灾难，但为了降噪而训练的深度神经网络可以学习到高维密度。此外，我们展示了在训练集的非重叠子集上训练的网络可以学习到相同的密度，从而证明了DNN架构和训练算法中的归纳偏差与数据分布的一致性。 |
| [^101] | [zkFL: Zero-Knowledge Proof-based Gradient Aggregation for Federated Learning.](http://arxiv.org/abs/2310.02554) | zkFL是一种基于零知识证明的联邦学习梯度聚合方法，通过提供每轮的证明来解决协调者恶意行为的问题。 |
| [^102] | [Heterogeneous Federated Learning Using Knowledge Codistillation.](http://arxiv.org/abs/2310.02549) | 使用知识共蒸合的异构联邦学习方法通过在整个池子和容量较高的部分客户端上训练不同大小的模型，实现了双向信息交换和领域转移，改进了联邦平均化算法在图像分类和语言建模任务上的性能。 |
| [^103] | [Exact and soft boundary conditions in Physics-Informed Neural Networks for the Variable Coefficient Poisson equation.](http://arxiv.org/abs/2310.02548) | 本研究研究了在物理信息神经网络(PINN)中应用软损失和精确距离函数的边界条件时的差异，并提供了有关如何实现这些PINN的资源和工具。 |
| [^104] | [Joint Design of Protein Sequence and Structure based on Motifs.](http://arxiv.org/abs/2310.02546) | 本文提出了一种GeoPro方法，用于联合设计蛋白质的骨架结构和序列。实验证明，GeoPro在多个指标上优于其他方法，并发现了新型β-内酰胺酶和肌红蛋白。 |
| [^105] | [Provable Tensor Completion with Graph Information.](http://arxiv.org/abs/2310.02543) | 本文提出了一个创新框架，系统地解决了具有图信息的动态图正则化张量补全问题，建立了严格的数学表示，并推导出了新的图导向补全模型。 |
| [^106] | [Benign Overfitting and Grokking in ReLU Networks for XOR Cluster Data.](http://arxiv.org/abs/2310.02541) | 通过梯度下降训练的ReLU网络在XOR集群数据上会产生良性过拟合和理解现象，即在训练阶段实现噪声标签的完美拟合但在测试阶段表现随机，在后续阶段可以实现近乎最优的泛化能力。 |
| [^107] | [Auto-FP: An Experimental Study of Automated Feature Preprocessing for Tabular Data.](http://arxiv.org/abs/2310.02540) | 本文研究了如何自动化表格数据的特征预处理（Auto-FP），将其建模为超参数优化或神经网络架构搜索问题，并扩展了各种算法来解决Auto-FP问题。 |
| [^108] | [Quantifying and mitigating the impact of label errors on model disparity metrics.](http://arxiv.org/abs/2310.02533) | 本研究量化和减轻了标签错误对模型差异度量的影响，并且提出了一种估计训练输入标签对模型差异度量影响的方法，有效地改进了现有方法。 |
| [^109] | [Federated Conditional Stochastic Optimization.](http://arxiv.org/abs/2310.02524) | 本文提出了一种新的联邦条件随机优化算法(FCSG)，针对联邦学习中的非凸条件随机优化问题，通过设计加速算法(Acc-FCSG-M)来实现最佳的样本和通信复杂度。 |
| [^110] | [MedDiffusion: Boosting Health Risk Prediction via Diffusion-based Data Augmentation.](http://arxiv.org/abs/2310.02520) | 本文介绍了一种名为MedDiffusion的新型、端到端的扩散式风险预测模型，通过基于扩散的数据增强，提升了健康风险预测的效果。 |
| [^111] | [Parameterized Convex Minorant for Objective Function Approximation in Amortized Optimization.](http://arxiv.org/abs/2310.02519) | 提出了一种参数化凸支配（PCM）方法，用于在摊销优化中逼近目标函数。该方法具有通用逼近性能，并可以通过单个凸优化获得全局最小值。 |
| [^112] | [A Recipe for Improved Certifiable Robustness: Capacity and Data.](http://arxiv.org/abs/2310.02513) | 在本研究中，我们通过使用一系列新技术、设计优化和综合以前的研究，更全面地评估了基于Lipschitz的认证方法的潜力，并显著提高了状... |
| [^113] | [Ophiuchus: Scalable Modeling of Protein Structures through Hierarchical Coarse-graining SO(3)-Equivariant Autoencoders.](http://arxiv.org/abs/2310.02508) | Ophiuchus是一个通过分层粗粒化SO(3)-等变自编码器对蛋白质结构进行可扩展建模的模型，它能在高分辨率下操作所有重原子，同时捕捉到结构的重复和分层模式。 |
| [^114] | [Learning to Reach Goals via Diffusion.](http://arxiv.org/abs/2310.02505) | 本论文提出了一种通过扩散学习实现目标达成的方法，可以在任意初始状态下从预定义或新目标达成，而无需学习单独的价值函数。 |
| [^115] | [Towards an Interpretable Representation of Speaker Identity via Perceptual Voice Qualities.](http://arxiv.org/abs/2310.02497) | 本研究提出了一种基于感知声音特质的可解释的说话者身份表示方法，通过将性别化的PQs添加到声音感知评估协议中，实现了成年人声音特征的中间层抽象。实验证明这些声音特质是可以被非专业人员感知到的，并且基于PQs的表示中的信息是可以被各种语音表示预测的。 |
| [^116] | [DON-LSTM: Multi-Resolution Learning with DeepONets and Long Short-Term Memory Neural Networks.](http://arxiv.org/abs/2310.02491) | DON-LSTM是一种新的架构，将DeepONet与长短期记忆网络（LSTM）结合起来，旨在通过利用多分辨率数据和捕捉长序列的时间依赖性来提高模型的性能。实验结果表明，DON-LSTM能够在多个非线性系统的长时间演化建模方面实现较低的泛化误差，并且需要较少的高分辨率数据。 |
| [^117] | [ResidualTransformer: Residual Low-rank Learning with Weight-sharing for Transformer Layers.](http://arxiv.org/abs/2310.02489) | 本文提出了一种名为ResidualTransformer的方法，通过重新参数化Transformer编码器层之间的模型权重，将模型的大小减小。实验结果表明，ResidualTransformer的性能优于传统Transformer模型，且模型大小得到了显著减小。 |
| [^118] | [OCU-Net: A Novel U-Net Architecture for Enhanced Oral Cancer Segmentation.](http://arxiv.org/abs/2310.02486) | OCU-Net是一种新型的U-Net架构，专门用于口腔癌分割任务。它结合了多种创新的深度学习模块和特征，包括通道和空间注意融合模块、挤压激活注意模块、空洞空间金字塔池化模块等，并在两个数据集上展现出卓越的性能。 |
| [^119] | [Splitting the Difference on Adversarial Training.](http://arxiv.org/abs/2310.02480) | 本文提出了一种基于分割类别的对抗训练方法，将每个类别的扰动样本视为单独的类别进行学习，从而简化了决策边界，提高了模型的鲁棒性。 |
| [^120] | [ML4EJ: Decoding the Role of Urban Features in Shaping Environmental Injustice Using Interpretable Machine Learning.](http://arxiv.org/abs/2310.02476) | 本研究使用可解释的机器学习模型，研究了城市特征对空气污染、城市热岛效应和洪涝灾害的暴露差异的影响，并弥补了传统环境不公正观点对城市特征影响的有限视角。 |
| [^121] | [Prompting-based Efficient Temporal Domain Generalization.](http://arxiv.org/abs/2310.02473) | 我们提出了一种基于提示的高效时域泛化方法，通过学习全局提示、领域特定提示和感知时序漂移的提示，不需要目标域数据的情况下适应时序漂移，并在各种任务中取得了state-of-the-art的性能。 |
| [^122] | [Distributionally Safe Reinforcement Learning under Model Uncertainty: A Single-Level Approach by Differentiable Convex Programming.](http://arxiv.org/abs/2310.02459) | 本文提出了一个分布安全的强化学习框架，通过Wasserstein度量来确保在模型不确定性下的安全性。通过使用对偶理论和可微分凸规划，将双层问题简化为单层问题，提高了可行性和效率。 |
| [^123] | [Learning Optimal Advantage from Preferences and Mistaking it for Reward.](http://arxiv.org/abs/2310.02456) | 本文研究了从人类偏好中学习奖励函数的算法，并发现实际上学到的是最佳优势函数而不是奖励函数。这种错误的使用方式虽然不特别有害，但与正确的贪婪最大化最佳优势函数相比仍不够理想。 |
| [^124] | [Dual-stage Flows-based Generative Modeling for Traceable Urban Planning.](http://arxiv.org/abs/2310.02453) | 这项研究提出了一种基于流式生成模型的双阶段城市规划框架，用于解决传统城市规划方法中忽略功能区关系、生成过程不稳定等问题。 |
| [^125] | [Feather: An Elegant Solution to Effective DNN Sparsification.](http://arxiv.org/abs/2310.02448) | Feather是一种优雅的DNN稀疏化解决方案，它具有高效的稀疏训练模块和强大的直通过估计器核心，能够在标准训练过程中实现鲁棒的稀疏化性能，并在CIFAR数据集上展示了其有效性和适应性，在ImageNet上使用ResNet-50架构实现了最新的最佳验证准确率，超越了现有方法。 |
| [^126] | [Machine learning assist nyc subway navigation safer and faster.](http://arxiv.org/abs/2310.02447) | 一项使用机器学习的研究旨在通过整数规划模型和最短路径算法的综合评估，平衡纽约地铁导航的安全性和效率性。 |
| [^127] | [Low-Resource Languages Jailbreak GPT-4.](http://arxiv.org/abs/2310.02446) | 通过翻译不安全的英文输入成低资源语言，我们成功绕过了GPT-4的安全机制，并展示了这种跨语言漏洞。这一方法在实验中取得了与甚至超过了最先进的越狱攻击的效果，揭示了低资源语言在AI安全性中的薄弱环节。 |
| [^128] | [GenCO: Generating Diverse Solutions to Design Problems with Combinatorial Nature.](http://arxiv.org/abs/2310.02442) | GenCO是一个新的框架，它整合了嵌入的组合求解器和深层生成模型，以发现与非线性目标一致的高质量解决方案。 |
| [^129] | [Episodic Memory Theory for the Mechanistic Interpretation of Recurrent Neural Networks.](http://arxiv.org/abs/2310.02430) | 提出了片段记忆理论(EMT)，将反复神经网络(RNN)概念化为通用序列片段记忆模型的离散时间类比，并且通过实验证实了EMT的有效性。通过引入新的算法任务，发现受训练的RNN始终会收敛到变量绑定电路，揭示了RNN动力学的普遍性，并且设计了一个算法来揭示变量的时间存储和组合中起重要作用的隐藏神经元。 |
| [^130] | [EGraFFBench: Evaluation of Equivariant Graph Neural Network Force Fields for Atomistic Simulations.](http://arxiv.org/abs/2310.02428) | EGraFFBench对六种EGraFF算法进行了系统的基准测试，以评估其在原子模拟中的性能，并提出了新的数据集和度量标准。 |
| [^131] | [Delta-AI: Local objectives for amortized inference in sparse graphical models.](http://arxiv.org/abs/2310.02423) | Delta-AI算法提出了一种基于稀疏图模型的摊还推理方法，通过局部信用分配和离策略训练加快了训练速度。 |
| [^132] | [OneAdapt: Fast Adaptation for Deep Learning Applications via Backpropagation.](http://arxiv.org/abs/2310.02422) | OneAdapt通过梯度上升策略来实现快速自适应，满足了深度学习应用在配置参数方面的三个要求。 |
| [^133] | [Can a student Large Language Model perform as well as it's teacher?.](http://arxiv.org/abs/2310.02421) | 这篇论文总结了知识蒸馏技术，并强调了其关键原理和成功要素，以及在资源受限环境中的部署挑战。同时，论文还指出知识蒸馏有潜力成为关键的技术转折点。 |
| [^134] | [FedL2P: Federated Learning to Personalize.](http://arxiv.org/abs/2310.02420) | 本论文介绍了一种名为FedL2P的联邦学习算法，通过学习个性化策略的元网络，实现了在不同联邦学习问题上的个性化学习，并取得了良好的性能。 |
| [^135] | [Bag of Tricks for Fully Test-Time Adaptation.](http://arxiv.org/abs/2310.02416) | 本研究提出了一种充分测试时间适应（TTA）的技巧，包括小批量归一化、流平衡、可靠样本选择和网络置信度校准。通过对这些技术的详细分析，我们揭示了它们在准确性、计算能力和模型复杂性之间的权衡，并取得了新的最先进的结果。 |
| [^136] | [Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness.](http://arxiv.org/abs/2310.02410) | 本文提出了一种量化专家混合（MoQE）方法，通过将专家权重应用2位低位量化，减轻了大规模专家混合（MoE）模型在内存消耗和延迟问题上的压力，同时在大多数情况下不需要额外训练也能保持可靠的模型性能。 |
| [^137] | [Nugget 2D: Dynamic Contextual Compression for Scaling Decoder-only Language Models.](http://arxiv.org/abs/2310.02409) | Nugget 2D是一种用于仅解码器语言模型的动态上下文压缩方法，可以在保留任务能力的同时大幅减少解码过程所需的时间和空间开销。 |
| [^138] | [Automated Bug Generation in the era of Large Language Models.](http://arxiv.org/abs/2310.02407) | 本论文探讨了在大型语言模型时代的自动缺陷生成问题，针对难以检测和难以修复的缺陷提出了解决方案，并分析了基于学习的技术中这两个目标的冲突。 |
| [^139] | [PCGPT: Procedural Content Generation via Transformers.](http://arxiv.org/abs/2310.02405) | 本文介绍了PCGPT框架，它利用离线强化学习和Transformer网络进行程序化内容生成。该框架通过解决传统PCG方法的挑战，生成了更复杂和多样化的游戏内容，并且在更少的步骤中实现了这些结果。 |
| [^140] | [On the Parallel Complexity of Multilevel Monte Carlo in Stocahstic Gradient Descent.](http://arxiv.org/abs/2310.02402) | 本文提出了一种延迟MLMC梯度估计器，通过重复利用之前步骤中计算过的梯度分量，大大降低了MLMC的并行复杂性，并在数值实验中证明了其在随机梯度下降中具有更好的并行复杂性。 |
| [^141] | [Reducing Intraspecies and Interspecies Covariate Shift in Traumatic Brain Injury EEG of Humans and Mice Using Transfer Euclidean Alignment.](http://arxiv.org/abs/2310.02398) | 本文介绍了一种转移学习技术，以应对在睡眠脑电图（EEG）分析中由于个体间的高变异性而导致机器学习模型在不同数据集上的表现差异。该技术可以通过转移欧几里德对齐来解决缺乏高质量人类生物医学数据的问题，并在多种经典机器学习模型和深度学习模型上展现出鲁棒性。 |
| [^142] | [Implicit regularization of multi-task learning and finetuning in overparameterized neural networks.](http://arxiv.org/abs/2310.02396) | 本文研究了在过参数化神经网络中，多任务学习和微调所带来的隐式正则化效果。在简化的线性网络环境中，我们发现了多任务学习和微调所对特征共享和学习特定特征稀疏性的鼓励作用，并发现微调过程同时具有内核和特征学习的混合状态。此外，微调还可以展现一种嵌套特征学习行为，使其偏向于提取一组稀疏的特征子集。 |
| [^143] | [SE(3)-Stochastic Flow Matching for Protein Backbone Generation.](http://arxiv.org/abs/2310.02391) | 通过SE(3)-Stochastic Flow Matching，我们提出了一系列新型生成模型FoldFlow，可以准确建模蛋白质主链。这些模型通过无需模拟训练和Riemannian最优传输的结合，具有更好的稳定性和建模能力。 |
| [^144] | [Secure and Effective Data Appraisal for Machine Learning.](http://arxiv.org/abs/2310.02373) | 本文介绍了一种机密的数据选择和评估方法，通过创新的流程和简化的低维度操作来实现，以保护数据和模型的隐私，并在多个Transformer模型和NLP/CV基准测试中进行了评估。 |
| [^145] | [Reinforcement Learning from Automatic Feedback for High-Quality Unit Test Generation.](http://arxiv.org/abs/2310.02368) | 本论文提出了一种名为静态质量指标强化学习（RLSQM）的新技术，用于解决大型语言模型（LLM）在自动生成测试用例时可能生成不良代码异味的问题。通过训练特定的奖励模型和利用PPO算法进行优化，我们实现了对单个质量指标和整体质量的优化。 |
| [^146] | [Stochastic force inference via density estimation.](http://arxiv.org/abs/2310.02366) | 该论文提出了一种基于密度估计的方法，通过概率流推断在分布之间插值的自主非线性力场，并应用于生物物理学中的转录组学，解决了从低分辨率的时间数据中推断动力学模型的挑战。 |
| [^147] | [On the definition of toxicity in NLP.](http://arxiv.org/abs/2310.02357) | 这项研究探讨了毒性的定义模糊性问题，并提出了一种基于定量压力的毒性定义来弥补现有定义的缺点。 |
| [^148] | [Investigating Speed Deviation Patterns During Glucose Episodes: A Quantile Regression Approach.](http://arxiv.org/abs/2310.02351) | 本研究利用分位数回归方法探究了糖尿病驾驶者在急性血糖情况下的速度行为模式，相较于以往的研究，该方法能够更好地捕捉到分布模式。 |
| [^149] | [Approximately Equivariant Quantum Neural Network for $p4m$ Group Symmetries in Images.](http://arxiv.org/abs/2310.02323) | 本文提出了几乎等变量子神经网络（EquivQCNNs），针对图像中的平面$p4m$对称性进行图像分类。这种方法在保持优化性能的同时，通过将先验知识纳入模型中，提升了训练和泛化能力。 |
| [^150] | [Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation.](http://arxiv.org/abs/2310.02304) | 本文提出了一种自学优化器（STOP），通过递归自我改进的代码生成，使用融合了语言模型的脚手架程序来改进自身，从而生成性能更好的程序。 |
| [^151] | [Relaxed Octahedral Group Convolution for Learning Symmetry Breaking in 3D Physical Systems.](http://arxiv.org/abs/2310.02299) | 本文介绍了一种用于建模3D物理系统的松弛八面体群卷积技术，它可以在保持数据一致的最高等变性水平的同时，发现物理系统中微妙的对称性破缺因素。 |
| [^152] | [Unsupervised Complex Semi-Binary Matrix Factorization for Activation Sequence Recovery of Quasi-Stationary Sources.](http://arxiv.org/abs/2310.02295) | 本文提出了一种无监督复杂半二进制矩阵分解算法，用于恢复拟静止源的激活序列。这种算法能够通过传感器数据提取个体激活序列，并对工业过程和制造系统的能源可持续性研究提供帮助。 |
| [^153] | [A Comparison of Mesh-Free Differentiable Programming and Data-Driven Strategies for Optimal Control under PDE Constraints.](http://arxiv.org/abs/2310.02286) | 本研究对直接-伴随循环、物理感知神经网络和可微分编程进行了比较，发现在偏微分方程约束下的最优控制中，可微分编程是最有效的方法，并提供了条件下的使用指南。 |
| [^154] | [PASTA: PArallel Spatio-Temporal Attention with spatial auto-correlation gating for fine-grained crowd flow prediction.](http://arxiv.org/abs/2310.02284) | 本文提出了一种名为PASTA的神经网络模型，通过细粒度地图中的历史人流的时空模式来预测未来全市范围内的人群流。这种方法包括空间自相关门控、多尺度残差块和时序注意力门控模块，能够有效捕捉细粒度地图中的不规则时空模式。 |
| [^155] | [SWMLP: Shared Weight Multilayer Perceptron for Car Trajectory Speed Prediction using Road Topographical Features.](http://arxiv.org/abs/2310.02282) | 本论文提出了一种独立于大量历史速度数据的速度预测方法，通过使用车辆轨迹的道路地形特征来拟合共享权重多层感知机学习模型，取得了显著的定性和定量改进，同时为交通分析的新方法设计提供了新的视角。 |
| [^156] | [End-to-End Continuous Speech Emotion Recognition in Real-life Customer Service Call Center Conversations.](http://arxiv.org/abs/2310.02281) | 本研究介绍了一个用于实时客服呼叫中心对话中连续语音情感识别的大规模数据集CusEmo，采用维度情感注释方法捕捉情感的微妙、复杂和连续性，并解决了应用于数据集时的挑战。 |
| [^157] | [Expert enhanced dynamic time warping based anomaly detection.](http://arxiv.org/abs/2310.02280) | 本文提出了一种名为E-DTWA的新型异常检测方法，基于动态时间规整（DTW）并加入了人机交互概念相关的额外改进。该方法具有高效的检测能力，能够在强烈考虑到专家的检测反馈的基础上灵活地进行重新训练，同时保持低计算和空间复杂度。 |
| [^158] | [Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion.](http://arxiv.org/abs/2310.02279) | 提出了一种一致性轨迹模型（CTM），它可以加速扩散模型的采样，同时通过对抗训练和去噪得分匹配损失的组合来提高性能，并实现了最先进的采样质量。 |
| [^159] | [Junk DNA Hypothesis: A Task-Centric Angle of LLM Pre-trained Weights through Sparsity.](http://arxiv.org/abs/2310.02277) | 本文研究通过稀疏性分析LLM预训练权重的任务中心角度，挑战了传统对于权重中冗余性的观点，并提出了"垃圾DNA假设"。 |
| [^160] | [Deep learning soliton dynamics and complex potentials recognition for 1D and 2D PT-symmetric saturable nonlinear Schr\"odinger equations.](http://arxiv.org/abs/2310.02276) | 本文利用深度学习方法扩展了物理信息神经网络(PINNs)来学习1D和2D饱和非线性薛定谔方程中的孤子动力学和复杂势识别。该方法还能解决PT对称势函数的逆问题，并进行了与传播距离z相关的1D和2D PT对称势的识别。 |
| [^161] | [MuSe-GNN: Learning Unified Gene Representation From Multimodal Biological Graph Data.](http://arxiv.org/abs/2310.02275) | 本研究引入了一种名为MuSe-GNN的模型，通过结合多模态机器学习和深度图神经网络，从单细胞测序和空间转录组数据中学习基因表示，并利用加权相似性学习和对比学习的正则化方法学习跨数据基因关系。该模型能够在一个共同的空间中提供包含不同背景下功能相似性的基因表示。 |
| [^162] | [ARRQP: Anomaly Resilient Real-time QoS Prediction Framework with Graph Convolution.](http://arxiv.org/abs/2310.02269) | 本研究介绍了一种名为ARRQP的实时QoS预测框架，重点改善了对数据中异常的鲁棒性，并利用图卷积技术来捕捉用户和服务之间复杂的关系和依赖。 |
| [^163] | [CoNO: Complex Neural Operator for Continuous Dynamical Systems.](http://arxiv.org/abs/2310.02094) | 本文介绍了一种复杂神经算子（CoNO），用于解决连续动力学系统中的偏微分方程。该算子通过复分数傅里叶变换来捕获丰富的表示，并通过复值神经网络来提高表示能力、稳健性和泛化性能。 |
| [^164] | [Learning Quantum Processes with Quantum Statistical Queries.](http://arxiv.org/abs/2310.02075) | 本文提出了第一个在量子统计查询模型内学习量子过程的框架，并提供了一个高效的学习器和可证明的性能保证。通过在密码分析中的应用，揭示了经典读出量子物理不可克隆函数的脆弱性，这是量子硬件安全领域一个重要的开放问题的解决方法。 |
| [^165] | [OceanGPT: A Large Language Model for Ocean Science Tasks.](http://arxiv.org/abs/2310.02031) | OceanGPT是首个专为海洋科学任务设计的大型语言模型，通过DoInstruct框架实现自动获取海洋领域指导数据。这一模型的引入填补了海洋科学领域中对LLM的需求缺口，并为海洋科学研究提供了新的工具和方法。 |
| [^166] | [FiGURe: Simple and Efficient Unsupervised Node Representations with Filter Augmentations.](http://arxiv.org/abs/2310.01892) | 本文介绍了一种简单的过滤器增强方法来改进无监督节点表示学习的性能，通过捕捉不同特征频谱部分，我们展示了显著的改进，并减少了计算负载。同时，我们通过使用简单的随机 Fourier 特征投影来解决高维表示的计算问题，并在基准数据集上取得了良好的性能。 |
| [^167] | [Effective and Parameter-Efficient Reusing Fine-Tuned Models.](http://arxiv.org/abs/2310.01886) | 本文提出了一种有效且参数高效的方法，可以重复使用微调模型来处理下游任务，减轻存储和服务负担，并提出了PERU-FFT方法用于重复使用全面微调模型。 |
| [^168] | [Blending Imitation and Reinforcement Learning for Robust Policy Improvement.](http://arxiv.org/abs/2310.01737) | 本文提出了一种融合模仿学习和强化学习的方法，根据在线评估结果交替使用二者，以提高样本效率和学习效果。 |
| [^169] | [SmartPlay : A Benchmark for LLMs as Intelligent Agents.](http://arxiv.org/abs/2310.01557) | SmartPlay是一个用于评估LLMs作为智能Agent能力的基准，包括6个具有不同挑战的游戏，并测试了智能LLM Agent的多种关键能力。这不仅是一个评估LLM Agent整体性能的严格测试场地，还可以分析每个能力的表现。 |
| [^170] | [PharmacoNet: Accelerating Large-Scale Virtual Screening by Deep Pharmacophore Modeling.](http://arxiv.org/abs/2310.00681) | PharmacoNet是一个利用深度学习框架的虚拟筛选方法，通过识别最佳的三维药物谱排列来加速大规模虚拟筛选过程，相比于现有方法更快且准确性合理。 |
| [^171] | [Learning Type Inference for Enhanced Dataflow Analysis.](http://arxiv.org/abs/2310.00673) | 该论文介绍了一种学习类型推断的方法，以增强数据流分析。传统的类型推断在面对程序规模增长时面临性能挑战，而基于机器学习的统计技术可以提供更快的推断。然而，目前的方法在用户定义的类型上性能仍较差，限制了其在实际应用中的效果。 |
| [^172] | [SIMD Dataflow Co-optimization for Efficient Neural Networks Inferences on CPUs.](http://arxiv.org/abs/2310.00574) | 我们提出了一种通过共优化数据流和SIMD实现来高效地在CPU上进行神经网络推理的方法，实验结果表明，这种方法能够在保持准确性的同时大幅提升推理速度。 |
| [^173] | [Are Graph Neural Networks Optimal Approximation Algorithms?.](http://arxiv.org/abs/2310.00526) | 本文设计了图神经网络架构OptGNN，利用半定规划工具获得大类组合优化问题的最优近似算法。通过实证结果表明在各种数据集上超过了神经网络基线算法和传统算法，同时利用OptGNN的能力设计了一个产生优化的对偶证书的算法。 |
| [^174] | [Structural Adversarial Objectives for Self-Supervised Representation Learning.](http://arxiv.org/abs/2310.00357) | 通过结构对抗目标和平滑正则化器，该论文提出了一种自监督表示学习方法，可以在生成对抗网络中学习提取信息丰富的特征表示，而无需依赖手工数据增强方案。实验证明该方法在图像分类任务上取得了良好的效果。 |
| [^175] | [SpatialRank: Urban Event Ranking with NDCG Optimization on Spatiotemporal Data.](http://arxiv.org/abs/2310.00270) | 这篇论文提出了一种名为SpatialRank的新颖空间事件排名方法，通过基于时空数据的NDCG优化来解决城市事件排名问题。 |
| [^176] | [Module-wise Training of Neural Networks via the Minimizing Movement Scheme.](http://arxiv.org/abs/2309.17357) | 通过引入模块化正则化方法，解决了神经网络模块化训练中早期层过拟合和深层停滞的问题，实验结果展示了该方法在不同架构上的优越性。 |
| [^177] | [Efficient Biologically Plausible Adversarial Training.](http://arxiv.org/abs/2309.17348) | 本文研究了生物合理的学习算法是否比反向传播更具有对抗攻击的鲁棒性，并进行了广泛的比较分析。 |
| [^178] | [A Foundation Model for General Moving Object Segmentation in Medical Images.](http://arxiv.org/abs/2309.17264) | 本文提出了一种用于医学图像中移动目标分割的基础模型iMOS，通过对序列中只有少量图像进行注释，即可实现高精度的分割效果 |
| [^179] | [On Computational Entanglement and Its Interpretation in Adversarial Machine Learning.](http://arxiv.org/abs/2309.15669) | 本研究探索了对抗机器学习模型的复杂性和可解释性，通过将其与爱因斯坦的特殊相对论中的纠缠概念联系起来，发现远程特征样本可以表现出纠缠现象，挑战了对抗可传递性现象的传统描述方法。 |
| [^180] | [MLOps for Scarce Image Data: A Use Case in Microscopic Image Analysis.](http://arxiv.org/abs/2309.15521) | 本论文研究在稀缺数据分析中完全应用MLOps的情况，并提出了一种新的整体方法来增强生物医学图像分析。 |
| [^181] | [Towards a statistical theory of data selection under weak supervision.](http://arxiv.org/abs/2309.14563) | 本研究针对弱监督下的数据选择进行了统计理论研究，通过实验证明数据选择可以非常有效，有时甚至可以战胜对整个样本的训练。并分析了在不同情况下的数据选择选择方法的有效性。 |
| [^182] | [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models.](http://arxiv.org/abs/2309.14509) | 本论文介绍了DeepSpeed-Ulysses，一种用于实现具备极长序列长度的高效可扩展LLM训练的新颖方法。 |
| [^183] | [Probabilistic Weight Fixing: Large-scale training of neural network weight uncertainties for quantization.](http://arxiv.org/abs/2309.13575) | 本文提出了一种基于贝叶斯神经网络和变分松弛的概率框架，用于通过将权重值限制在一组有限值上来减少推理过程中的能量消耗。通过利用权重值的概率分布，提高了噪声鲁棒性和可压缩性。迭代聚类过程展示了超越现有方法的优势。 |
| [^184] | [Time-Series Forecasting: Unleashing Long-Term Dependencies with Fractionally Differenced Data.](http://arxiv.org/abs/2309.13409) | 本研究提出了一种利用分数差分来捕捉时间序列数据中短期和长期依赖关系的预测策略。通过将FD应用于金融数据并结合情感分析，实证结果证明FD在二元分类中的性能优于整数差分方法。 |
| [^185] | [Learning Adaptive Safety for Multi-Agent Systems.](http://arxiv.org/abs/2309.10657) | 本论文研究了在多智能体系统中学习自适应安全性的问题，提出了一种全新的自适应安全强化学习框架ASRL，通过优化策略和CBF系数，增强安全性和长期性能。在与其他智能体的交互中，ASRL学会了应对不同的智能体行为，并保持成本违规在所需限制之下。 |
| [^186] | [Investigating the Catastrophic Forgetting in Multimodal Large Language Models.](http://arxiv.org/abs/2309.10313) | 本论文针对多模态大规模语言模型中的灾难性遗忘问题进行研究，引入了EMT方法来评估灾难性遗忘，并发现在标准图像分类任务上，几乎所有评估的模型都无法保持与视觉编码器相同的性能水平。研究结果表明，早期微调阶段对性能至关重要。 |
| [^187] | [Sparse Autoencoders Find Highly Interpretable Features in Language Models.](http://arxiv.org/abs/2309.08600) | 本研究通过稀疏自编码器在语言模型中发现了一组高度可解释和单一义的特征，从而解决了神经网络内部多义性的问题。 |
| [^188] | [ConR: Contrastive Regularizer for Deep Imbalanced Regression.](http://arxiv.org/abs/2309.06651) | ConR是一种对比正则化器，通过建模全局和局部标签相似性，防止少数样本的特征被折叠到其多数邻居中，有效地处理深度不平衡回归问题。 |
| [^189] | [Speciality vs Generality: An Empirical Study on Catastrophic Forgetting in Fine-tuning Foundation Models.](http://arxiv.org/abs/2309.06256) | 本研究实证了基础模型微调中的灾难性遗忘现象，微调过程中追求专业性会导致模型的广泛性损失。 |
| [^190] | [Navigating Out-of-Distribution Electricity Load Forecasting during COVID-19: A Continual Learning Approach Leveraging Human Mobility.](http://arxiv.org/abs/2309.04296) | 本研究提出了一种利用人类移动数据和持续学习技术的方法来解决COVID-19期间非分布期间的电力负荷预测问题，通过保留过去的见解并整合新的数据，提高了模型的准确性和鲁棒性。 |
| [^191] | [Optimal Transport with Tempered Exponential Measures.](http://arxiv.org/abs/2309.04015) | 本文推广了熵正则化最优输运方法，将其应用于温度指数测度中，实现了快速有效的算法和可控的稀疏性。 |
| [^192] | [Physics-inspired Equivariant Descriptors of Non-bonded Interactions.](http://arxiv.org/abs/2308.13208) | 基于物理启发，我们提出了一种受物理启发的非键相互作用等变描述符框架，该框架能够模拟长程物理相互作用，并且能够生成类似非键位势的局部描述符。 |
| [^193] | [Bayesian low-rank adaptation for large language models.](http://arxiv.org/abs/2308.13111) | 本研究提出了一种名为Laplace-LoRA的贝叶斯方法，通过应用拉普拉斯近似来增强经过微调的大型语言模型的校准能力。 |
| [^194] | [Instruction Tuning for Large Language Models: A Survey.](http://arxiv.org/abs/2308.10792) | 本文调查了指令调优这一关键技术在增强大型语言模型能力和可控性方面的研究工作，包括方法、数据集构建、模型训练和应用，以及对结果影响的分析。同时回顾了可能的问题和批评，并指出了目前的不足。 |
| [^195] | [AQUILA: Communication Efficient Federated Learning with Adaptive Quantization of Lazily-Aggregated Gradients.](http://arxiv.org/abs/2308.00258) | AQUILA是一个自适应量化梯度的通信高效联邦学习框架，解决了传输大规模模型时的通信开销和局部数据偏差导致的全局模型鲁棒性问题。 |
| [^196] | [Imitating Complex Trajectories: Bridging Low-Level Stability and High-Level Behavior.](http://arxiv.org/abs/2307.14619) | 本文提出了一个理论框架，研究了在非线性动态系统中模仿复杂专家演示的行为。通过稳定模仿策略并确保准确估计演示者分布，可以使模仿者与演示者的轨迹分布相近。 |
| [^197] | [PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks.](http://arxiv.org/abs/2307.11833) | PINNsFormer是一种基于Transformer的框架，通过捕捉时间依赖性准确逼近求解偏微分方程，相比传统方法具有更好的性能。 |
| [^198] | [Hierarchical Empowerment: Towards Tractable Empowerment-Based Skill-Learning.](http://arxiv.org/abs/2307.02728) | 分层授权提出了一种可以计算授权的新框架，通过引入变分下界和分层架构，实现了在短期和长期时间尺度上的授权计算，并在模拟机器人任务中得到了验证。 |
| [^199] | [Named Entity Inclusion in Abstractive Text Summarization.](http://arxiv.org/abs/2307.02570) | 该论文提出了一种解决抽象文本摘要中命名实体遗漏问题的方法，通过使用定制的预训练目标和模型训练策略，改善了命名实体的包含情况，提高了摘要的准确性和召回率。 |
| [^200] | [Unified Transfer Learning Models for High-Dimensional Linear Regression.](http://arxiv.org/abs/2307.00238) | UTrans是一种统一转移学习模型，它能检测可转移变量和源数据，并具有较低的估计和预测误差，同时保持可解释性。 |
| [^201] | [Graph Interpolation via Fast Fused-Gromovization.](http://arxiv.org/abs/2306.15963) | 本文提出了一种通过快速融合Gromov化的方法，用于图插值和图数据增强。通过考虑图结构和信号之间的相互作用，我们提出了一种匹配节点之间的最优策略来解决这一问题。为了提高可扩展性，我们引入了一种放松的FGW求解器来加速算法的收敛速度。 |
| [^202] | [DiMSam: Diffusion Models as Samplers for Task and Motion Planning under Partial Observability.](http://arxiv.org/abs/2306.13196) | 本文提出了一种使用扩散模型作为采样器的任务和动作规划方法，在部分可观测下能够实现长周期受约束的操作计划。 |
| [^203] | [Deep graph kernel point processes.](http://arxiv.org/abs/2306.11313) | 本文提出了一种基于潜在图拓扑的图点过程方法，并开发了一种新颖的深度图核来描述事件之间的触发和抑制效应，该方法在合成和实际数据集上具有优越性。 |
| [^204] | [Stochastic Re-weighted Gradient Descent via Distributionally Robust Optimization.](http://arxiv.org/abs/2306.09222) | 我们通过分布健壮优化和重要性加权的梯度下降技术提升了深度神经网络的性能，并在各种任务上取得了优越的结果。 |
| [^205] | [Fast Diffusion Model.](http://arxiv.org/abs/2306.06991) | 本文提出了一种快速扩散模型（FDM），通过将动量集成到扩散过程中，显著加速了扩散模型（DMs）的训练和采样过程。 |
| [^206] | [Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations.](http://arxiv.org/abs/2306.05880) | 该论文提出了基于INR的时间序列连续建模方法，解决了处理缺失数据、不规则采样和多传感器不对准观测等重复建模问题，并在预测和插值任务中取得了最新的性能表现，具有很好的泛化能力。 |
| [^207] | [Leaping through tree space: continuous phylogenetic inference for rooted and unrooted trees.](http://arxiv.org/abs/2306.05739) | 本研究首次在连续空间中进行树形系统探索和推断，用于有根和无根树，优于当前最佳方法并在实验中证明了其效果，可用于加速生命科学的新进化发现。 |
| [^208] | [Deductive Verification of Chain-of-Thought Reasoning.](http://arxiv.org/abs/2306.03872) | 本文旨在通过应用演绎验证技术，使语言模型能够进行明确而严谨的演绎推理，以确保其推理过程的可信度。 |
| [^209] | [Learning Representations on the Unit Sphere: Application to Online Continual Learning.](http://arxiv.org/abs/2306.03364) | 该论文提出了一种基于单位球的表示学习方法，通过将表示推向固定方向，使得学习策略对数据漂移具有弹性，从而能够应对在线连续学习的挑战性问题。 |
| [^210] | [Generative Diffusion for 3D Turbulent Flows.](http://arxiv.org/abs/2306.01776) | 该论文提出了一种生成模型，可以在任意三维空间中模拟湍流现象，避免了湍流流动的不可预测性，能够快速生成高质量的流场。 |
| [^211] | [Large-Batch, Neural Multi-Objective Bayesian Optimization.](http://arxiv.org/abs/2306.01095) | 本文提出了一种针对数据密集型问题和多目标优化设置的贝叶斯优化框架，该方法利用了贝叶斯神经网络代理建模和可扩展、具有不确定性的收购策略，能够在最少迭代次数的情况下高效地进行优化。 |
| [^212] | [Deep Stochastic Mechanics.](http://arxiv.org/abs/2305.19685) | 本文提出了一种基于深度学习的方法，用于数值模拟时间演化薛定谔方程，利用马尔可夫扩散采样来适应波函数的潜在低维结构，并提出了新的随机量子力学方程，具有线性的计算复杂度。数值模拟显示出显着的优势。 |
| [^213] | [Stochastic Gradient Langevin Dynamics Based on Quantization with Increasing Resolution.](http://arxiv.org/abs/2305.18864) | 本文提出了一种基于递增分辨率的量化随机梯度 langevin 动力学方法，通过利用 langevin 随机微分方程动力学，实现了具有可控噪声且具有相同分布的优化过程，无需添加噪声或调整小批量的大小。实验结果证明了该方法在不同数据集上对卷积神经网络模型和 ResNet-50 架构的有效性。 |
| [^214] | [Improved Probabilistic Image-Text Representations.](http://arxiv.org/abs/2305.18171) | 本论文提出了一种改进的概率图像-文本表示方法，通过引入新的概率距离和两种优化技术，解决了现有方法中的计算负担过重和损失饱和问题，取得了显著的性能提升。 |
| [^215] | [Rotational Optimizers: Simple & Robust DNN Training.](http://arxiv.org/abs/2305.17212) | 该论文提出了旋转优化器，这些优化器可以简化深度神经网络训练过程，甚至在几乎不需调整基线超参数的情况下与原始优化器的性能相匹配。 |
| [^216] | [Break-A-Scene: Extracting Multiple Concepts from a Single Image.](http://arxiv.org/abs/2305.16311) | 该论文提出了一种从单个图像中提取多个概念的文本场景分解方法，通过增加目标概念的掩码和优化文本嵌入和模型权重的方式，实现对生成场景的精细控制。 |
| [^217] | [Improving selective classification performance of deep neural networks through post-hoc logit normalization and temperature scaling.](http://arxiv.org/abs/2305.15508) | 本文提出了一种$p$-NormSoftmax的事后置信度估计器来提高深度神经网络的选择分类性能。 |
| [^218] | [Text Conditional Alt-Text Generation for Twitter Images.](http://arxiv.org/abs/2305.14779) | 本文针对Twitter上分享的图像提出了一种文本条件下的替代文本生成方法。通过CLIP前缀模型，该模型结合图像和推文中的文本信息，生成关于图像的上下文相关的替代文本。 |
| [^219] | [Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport.](http://arxiv.org/abs/2305.14777) | 本文提出了一种基于非平衡最优输运半对偶公式的新型生成模型，相比于OT，它具有更好的鲁棒性、稳定性和更快的收敛速度，实验结果表明其优于现有的基于OT的生成模型。 |
| [^220] | [Robust Explanations for Deep Neural Networks via Pseudo Neural Tangent Kernel Surrogate Models.](http://arxiv.org/abs/2305.14585) | 本研究通过建立一个规范化的伪神经切线核，证明了它能够更好地与神经网络决策函数相关，比基于嵌入和影响的替代品更有效，并且从它创建的归因会更准确地选择被扰动的训练数据，从而证明了核线性模型是跨多个数据领域并有效的替代模型。 |
| [^221] | [Property-Guided Generative Modelling for Robust Model-Based Design with Imbalanced Data.](http://arxiv.org/abs/2305.13650) | 本文提出了一种属性引导的变分自编码器（PGVAE），通过属性值明确结构化潜在空间，使得MBO可以在不平衡数据上稳健地寻找具有改进属性的序列。 |
| [^222] | [Conditional Generative Modeling is All You Need for Marked Temporal Point Processes.](http://arxiv.org/abs/2305.12569) | 本文提出了一种从标记时间点过程中提取其统计直觉的事件生成模型，通过条件生成器以历史观察作为输入，生成可能发生的高质量随后事件。该模型具有高效、灵活和表示能力等方面的优势。 |
| [^223] | [Q-malizing flow and infinitesimal density ratio estimation.](http://arxiv.org/abs/2305.11857) | 研究提出了一种可以从一个数据分布P传输到任意访问通过有限样本的Q的流模型。这个模型通过神经ODE模型进行，可以进行无穷小DRE。 |
| [^224] | [Vision-based DRL Autonomous Driving Agent with Sim2Real Transfer.](http://arxiv.org/abs/2305.11589) | 本研究提出了一种基于视觉的深度强化学习代理，可以同时执行保持车道和跟车操作，并且展示了其在真实情况下的模型迁移能力，是第一个具有此能力的代理。 |
| [^225] | [Generative Sliced MMD Flows with Riesz Kernels.](http://arxiv.org/abs/2305.11463) | 本文使用Riesz核展示了生成式分割MMD流的高效计算方法，实现了在大规模应用中通过神经网络训练生成模型。 |
| [^226] | [A Parameter-Efficient Learning Approach to Arabic Dialect Identification with Pre-Trained General-Purpose Speech Model.](http://arxiv.org/abs/2305.11244) | 本文介绍了一种利用预训练通用语音模型进行阿拉伯方言识别的参数高效学习方法，通过残差适配器和模型重编程，设计了一个基于记号的标签映射，并在ADI-17数据集上实现了最高精度，同时使用PEL方法进一步减少了训练成本。 |
| [^227] | [PDP: Parameter-free Differentiable Pruning is All You Need.](http://arxiv.org/abs/2305.11203) | PDP提出了一种无需参数的可微剪枝方案，具有最先进的模型大小、准确性和训练成本，适用于各种视觉和自然语言任务。 |
| [^228] | [Self-supervised Neural Factor Analysis for Disentangling Utterance-level Speech Representations.](http://arxiv.org/abs/2305.08099) | 本文提出了一种自监督神经因子分析模型，使用HuBERT中的聚类方法来发现隐藏的声学单元，并使用这些单元对齐SSL模型的特征，从而产生解耦后的语音表示，从而为专门任务提供了一种基于Utterance水平的无监督学习目标。实验结果表明，SSNFA模型在说话人识别、语言识别和情感识别等各种任务中均显著优于现有的SSL模型，并且没有任何特定任务的微调或监督。 |
| [^229] | [Personalize Segment Anything Model with One Shot.](http://arxiv.org/abs/2305.03048) | 本文提出了一种无需训练的SAM个性化方法PerSAM，只需要一张带有参考掩模的单张图像即可定位和分割目标概念，还提出了高效的一次性微调变体PerSAM-F，旨在解决掩模不确定性问题。 |
| [^230] | [Dynamic Sparse Training with Structured Sparsity.](http://arxiv.org/abs/2305.02299) | 本文提出了一种结构化稀疏动态训练（DST）方法，学习一种变体的结构化 N:M 稀疏性，其加速在一般情况下通常被支持，可缩减参数和内存占用，同时相较于密集模型，具有减少推理时间的优势。 |
| [^231] | [Examining Computational Performance of Unsupervised Concept Drift Detection: A Survey and Beyond.](http://arxiv.org/abs/2304.08319) | 这篇论文调查了无监督概念漂移检测的计算性能，提出了一套指标来评估漂移检测器对AI系统的计算影响。 |
| [^232] | [Preemptively Pruning Clever-Hans Strategies in Deep Neural Networks.](http://arxiv.org/abs/2304.05727) | 本文提出了一种新方法，Explanation-Guided Exposure Minimization (EGEM)，该方法预防性地修剪了ML模型中未受到积极解释反馈的变化，从而大大减少了对隐藏Clever Hans策略的依赖，并实现了更高的性能。 |
| [^233] | [A Benchmark Generative Probabilistic Model for Weak Supervised Learning.](http://arxiv.org/abs/2303.17841) | 本文提出一种基准生成性概率模型，在启发式标注的原始数据集上训练，生成伪标签作为一种准确、快速、经济的弱监督学习方法，在图像分类和自然语言处理中达到了最先进的表现。 |
| [^234] | [Analysis of Failures and Risks in Deep Learning Model Converters: A Case Study in the ONNX Ecosystem.](http://arxiv.org/abs/2303.17708) | 本文详细分析了深度学习模型转换器的故障情况，特别是对ONNX相关的转换器进行了首次故障分析，并详细报告了故障的症状，原因和位置以及随时间的趋势。 |
| [^235] | [Variantional autoencoder with decremental information bottleneck for disentanglement.](http://arxiv.org/abs/2303.12959) | 本论文提出了一种逐步减少信息瓶颈的变分自编码器方法，使用去纠缠不变变换来平衡去纠缠和重构保真度，避免信息扩散问题。 |
| [^236] | [Neural Frailty Machine: Beyond proportional hazard assumption in neural survival regressions.](http://arxiv.org/abs/2303.10358) | 提出神经衰弱机器（NFM）框架用于生存回归，利用多重衰弱的经典思想来捕捉个体间未观察到的异质性，并能够处理非线性协变量依赖性。两个具体模型下扩展了神经比例危险模型和非参数危险回归模型，结论获得了统计保证。 |
| [^237] | [Robust Learning from Explanations.](http://arxiv.org/abs/2303.06419) | 本文提出了一种新的机器学习方法，将机器学习从解释（MLX）重新构建为对抗鲁棒性问题，通过人类提供的解释来指定一个低维流形，从而减轻了对强参数正则化的需求，并在合成和真实世界基准测试中取得了最新结果。 |
| [^238] | [Technical report: Graph Neural Networks go Grammatical.](http://arxiv.org/abs/2303.01590) | 本文介绍了一种将代数语言片段与图神经网络形式上联系的框架，并从MATLANG定义了一个符合3-WL测试的语法，进而得出一个符合3-WL GNN模型的G$^2$N$^2$。此外，语法方法还提供了计算长度为六及以下的环和弦环的代数公式，并在多个下游任务中取得优秀的表现。 |
| [^239] | [Computational Complexity of Learning Neural Networks: Smoothness and Degeneracy.](http://arxiv.org/abs/2302.07426) | 本文研究了学习神经网络的计算复杂度，特别关注了输入分布和权重矩阵的假设对学习算法有效性的影响。结果表明，在高斯输入分布下，学习深度为3的ReLU网络是困难的，即使权重矩阵是非退化的。同时，学习深度为2的网络也面临困难。 |
| [^240] | [A novel approach to generate datasets with XAI ground truth to evaluate image models.](http://arxiv.org/abs/2302.05624) | 该论文介绍了一种新方法来生成具有XAI基准的数据集，用于评估图像模型。通过与真实模型解释进行比较，实验证实了该方法的可靠性。 |
| [^241] | [A Survey on Deep Learning based Time Series Analysis with Frequency Transformation.](http://arxiv.org/abs/2302.02173) | 近期，频率变换（FT）在深度学习时间序列分析中得到广泛应用，显著提高了准确性和效率。本文系统回顾和总结了基于FT的深度学习时间序列模型的研究进展，并探讨了其优势、限制以及主要方法。 |
| [^242] | [Double Permutation Equivariance for Knowledge Graph Completion.](http://arxiv.org/abs/2302.01313) | 本研究提出了双排列等变性的KG表示方法，可以使神经网络在KG中执行复杂的逻辑推理任务，并在多个归纳KG完成任务中实现了最先进的Hits@10测试准确率。双排列等变性在KG中开辟了新的研究方向。 |
| [^243] | [Normalizing Flow Ensembles for Rich Aleatoric and Epistemic Uncertainty Modeling.](http://arxiv.org/abs/2302.01312) | 本文提出了一个正则化流（NF）集合来估计Epistemic不确定性和Aleatoric不确定性，通过固定的dropout掩码来创建集合，运用于各种实验并可以提供全面的基准线。 |
| [^244] | [Alternating Updates for Efficient Transformers.](http://arxiv.org/abs/2301.13310) | 本文介绍了一种交替更新（AltUp）的方法，可以在不增加计算负担的情况下增加模型的容量，通过对扩展表示的子块进行操作并使用预测和修正机制来更新未激活的块。实验证明AltUp方法在提高Transformer模型的容量和效率方面是有效的。 |
| [^245] | [Explaining $\mathcal{ELH}$ Concept Descriptions through Counterfactual Reasoning.](http://arxiv.org/abs/2301.05109) | 本研究提出了一种通过反事实推理来解释概念描述的方法，以提供简洁且易于理解的解释，便于非专家理解和采取行动。 |
| [^246] | [A comprehensive review of automatic text summarization techniques: method, data, evaluation and coding.](http://arxiv.org/abs/2301.03403) | 本文提供了关于自动文本摘要系统的综述，包括方法、数据、评估和编码。作者通过引用的方式回顾了相关文献，并介绍了不同的摘要生成方法。此外，还对可用于评估和数据训练的数据集进行了综述，并使用CNN语料库数据集对方法进行了实证探索。 |
| [^247] | [Expanding Small-Scale Datasets with Guided Imagination.](http://arxiv.org/abs/2211.13976) | 本论文提出了一个引导想象框架(GIF)，通过利用DALL-E2和Stable Diffusion (SD)等生成模型，从种子数据中扩充小规模数据集。该框架通过在先验模型的语义空间中优化种子数据潜在特征来创建逼真的图像，并引入了类别保持和样本多样性的标准来指导想象过程。 |
| [^248] | [Private Ad Modeling with DP-SGD.](http://arxiv.org/abs/2211.11896) | 本研究将差分隐私随机梯度下降（DP-SGD）应用于广告建模任务，证明了该方法可以在处理高类别不平衡和稀疏梯度更新的广告数据中提供隐私和效用。 |
| [^249] | [PersA-FL: Personalized Asynchronous Federated Learning.](http://arxiv.org/abs/2210.01176) | 本论文研究了异步更新下的个性化联邦学习问题，并提出了一种改进的个性化方法，通过移除同步通信假设和去除梯度范数有界性假设来提高可伸缩性。 |
| [^250] | [On the Convergence of AdaGrad on $\R^{d}$: Beyond Convexity, Non-Asymptotic Rate and Acceleration.](http://arxiv.org/abs/2209.14827) | 本论文主要展示了AdaGrad在平滑凸函数和更一般的quasar凸函数的情况下的收敛性。具体地，我们提出了新的技术，明确限定了vanilla AdaGrad在无约束问题中的收敛速率，并提出了一种AdaGrad变种，可以实现更快的收敛。 |
| [^251] | [The Neural Process Family: Survey, Applications and Perspectives.](http://arxiv.org/abs/2209.00517) | 神经过程家族旨在结合神经网络和高斯过程的优点，实现元学习预测不确定性的能力，并在深度学习领域带来重要进展。 |
| [^252] | [Latent Neural Stochastic Differential Equations for Change Point Detection.](http://arxiv.org/abs/2208.10317) | 本文提出了一种基于潜在神经随机微分方程的变点检测算法，通过学习将过程转换到潜在空间，并使用学习随机过程的似然比来定位过程中的变点。在合成和真实数据集上的实验证明了该方法的出色性能。 |
| [^253] | [Interactive Code Generation via Test-Driven User-Intent Formalization.](http://arxiv.org/abs/2208.05950) | 本文提出了交互式测试驱动代码生成的工作流程，该方法通过生成的测试形式化用户意图，并通过修剪和排名代码建议来提供改进的代码建议集。 |
| [^254] | [Co-Located Human-Human Interaction Analysis using Nonverbal Cues: A Survey.](http://arxiv.org/abs/2207.10574) | 使用非语言线索进行共同人际互动分析的研究调查了从2010年以来的计算研究，并总结了最常用的非语言线索，以及有关互动分析的未来研究方向。 |
| [^255] | [Gradual Domain Adaptation via Normalizing Flows.](http://arxiv.org/abs/2206.11492) | 该论文提出使用标准化流来解决逐渐领域适应中中间域有限且距离较大的问题，并通过从源域到高斯混合分布学习目标域的分布变换。 |
| [^256] | [Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger.](http://arxiv.org/abs/2206.07136) | 本论文提出了一种自动剪辑的替代方案，它消除了为差分隐私优化器调整剪辑阈值的需要，并在非凸情况下进行了收敛性分析。在多个任务中证明了自动剪辑的优势。 |
| [^257] | [Reconsidering Learning Objectives in Unbiased Recommendation: A Distribution Shift Perspective.](http://arxiv.org/abs/2206.03851) | 本文从分布转移视角出发，研究了从偏向反馈中学习无偏算法进行推荐的问题。通过建立无偏推荐与分布转移的关系，对现有无偏学习方法进行了理论解释并提出了两个泛化界限。 |
| [^258] | [On the efficiency of Stochastic Quasi-Newton Methods for Deep Learning.](http://arxiv.org/abs/2205.09121) | 本文研究了随机拟牛顿算法在深度学习中的效率，分析了有限内存BFGS和对称秩一SR1两种更新算法的性能表现，并比较了两者的优劣，探讨了SR1算法在处理非凸优化问题中病态鞍点时的潜力。 |
| [^259] | [Trajectory balance: Improved credit assignment in GFlowNets.](http://arxiv.org/abs/2201.13259) | GFlowNets使用轨迹平衡作为一种更高效的学习目标，解决了先前学习目标中信用传播效率低下的问题，并且在实验中证明了其在收敛性、生成样本多样性以及鲁棒性方面的优势。 |
| [^260] | [NeuroBack: Improving CDCL SAT Solving using Graph Neural Networks.](http://arxiv.org/abs/2110.14053) | NeuroBack提出了一种使用图神经网络改进CDCL SAT求解的方法，通过预测出现在大多数满足赋值中的变量的阶段，使得求解更加有效，并且消除了对GPU资源的依赖。 |
| [^261] | [AKE-GNN: Effective Graph Learning with Adaptive Knowledge Exchange.](http://arxiv.org/abs/2106.05455) | AKE-GNN是一种新型的图神经网络学习框架，通过自适应知识交换策略在多个图视图之间交换通道，以实现有效的图学习。 |
| [^262] | [Naive Exploration is Optimal for Online LQR.](http://arxiv.org/abs/2001.09576) | 在线LQR问题中，我们证明了天真的探索是最优的策略，可以在未知参数的情况下达到最小遗憾。这一结论对于解决在线自适应控制问题具有重要意义。 |
| [^263] | [How Implicit Regularization of ReLU Neural Networks Characterizes the Learned Function -- Part I: the 1-D Case of Two Layers with Random First Layer.](http://arxiv.org/abs/1911.02903) | 本文研究了一维ReLU神经网络，通过数学分析和实验证明了对于这种网络，L2正则化回归在函数空间中对应于对估计的二阶导数进行正则化，同时提出了早停止的梯度下降和平滑样条回归之间的新对应关系。 |

# 详细

[^1]: Decision ConvFormer: MetaFormer中的本地过滤对于决策制定已经足够了

    Decision ConvFormer: Local Filtering in MetaFormer is Sufficient for Decision Making. (arXiv:2310.03022v1 [cs.LG])

    [http://arxiv.org/abs/2310.03022](http://arxiv.org/abs/2310.03022)

    Decision ConvFormer提出了一种新的动作序列预测器，通过使用本地卷积过滤来捕捉强化学习数据集中的局部关联，同时在各个标准RL基准上取得了最先进的性能。

    

    Transformer在自然语言处理中的成功引发了其在各个领域的应用。在离线强化学习中，Decision Transformer（DT）作为一种基于Transformer的有前途的模型逐渐崭露头角。然而，我们发现DT的注意力模块不适合捕捉作为马尔科夫决策过程建模的强化学习轨迹中固有的局部依赖模式。为了解决DT的局限性，我们提出了一种基于MetaFormer架构的新型动作序列预测器，称为Decision ConvFormer（DC）。DC采用本地卷积过滤作为令牌混合器，能够有效捕捉RL数据集中固有的局部关联。在大量实验证明中，DC在各种标准RL基准上取得了最先进的性能，同时需要更少的资源。

    The recent success of Transformer in natural language processing has sparked its use in various domains. In offline reinforcement learning (RL), Decision Transformer (DT) is emerging as a promising model based on Transformer. However, we discovered that the attention module of DT is not appropriate to capture the inherent local dependence pattern in trajectories of RL modeled as a Markov decision process. To overcome the limitations of DT, we propose a novel action sequence predictor, named Decision ConvFormer (DC), based on the architecture of MetaFormer, which is a general structure to process multiple entities in parallel and understand the interrelationship among the multiple entities. DC employs local convolution filtering as the token mixer and can effectively capture the inherent local associations of the RL dataset. In extensive experiments, DC achieved state-of-the-art performance across various standard RL benchmarks while requiring fewer resources. Furthermore, we show that 
    
[^2]: 通过学习离散函数来理解变压器和LLM中的上下文学习现象

    Understanding In-Context Learning in Transformers and LLMs by Learning to Learn Discrete Functions. (arXiv:2310.03016v1 [cs.LG])

    [http://arxiv.org/abs/2310.03016](http://arxiv.org/abs/2310.03016)

    这项工作通过实验发现，变压器可以学习各种实值函数的基于梯度的学习算法，但对于更复杂的任务性能下降。另外，该研究还探讨了这些能力在基于注意力模型中的限制程度以及推广到预训练的大型语言模型（LLM）的可行性。

    

    为了理解上下文学习现象，最近的工作采用了规范化的实验框架，并证明了变压器可以学习各种实值函数的基于梯度的学习算法。然而，变压器在实现学习算法方面的限制以及它们学习其他形式算法的能力还不清楚。此外，这些能力在基于注意力模型中的限制程度也不明确。此外，尚不清楚从这些规范化设置中得出的见解是否能推广到预训练的大型语言模型（LLM）。在这项工作中，我们通过以下方式向这些问题迈进：（a）在一个包含各种布尔函数类的测试平台上，我们发现变压器几乎可以与“较简单”任务的最优学习算法相匹配，但在“较复杂”任务上其性能下降。此外，我们还发现

    In order to understand the in-context learning phenomenon, recent works have adopted a stylized experimental framework and demonstrated that Transformers can learn gradient-based learning algorithms for various classes of real-valued functions. However, the limitations of Transformers in implementing learning algorithms, and their ability to learn other forms of algorithms are not well understood. Additionally, the degree to which these capabilities are confined to attention-based models is unclear. Furthermore, it remains to be seen whether the insights derived from these stylized settings can be extrapolated to pretrained Large Language Models (LLMs). In this work, we take a step towards answering these questions by demonstrating the following: (a) On a test-bed with a variety of Boolean function classes, we find that Transformers can nearly match the optimal learning algorithm for 'simpler' tasks, while their performance deteriorates on more 'complex' tasks. Additionally, we find th
    
[^3]: SemiReward: 半监督学习的通用奖励模型

    SemiReward: A General Reward Model for Semi-supervised Learning. (arXiv:2310.03013v1 [cs.LG])

    [http://arxiv.org/abs/2310.03013](http://arxiv.org/abs/2310.03013)

    SemiReward是一个通用奖励模型，通过预测奖励分数来评估和过滤高质量的伪标签，可以应用于各种半监督学习任务，并在实验中取得了显著的成果。

    

    半监督学习在自训练框架和伪标签上取得了显著进展。主要挑战是如何区分高质量的伪标签，避免确证偏见。然而，现有的伪标签选择策略限制于预定义的方案或复杂的手工制作策略，无法同时实现高质量标签、快速收敛和任务多样性。为此，我们提出了一种半监督奖励框架（SemiReward），用于预测奖励分数以评估和过滤高质量的伪标签，可以在各种任务类型和场景下与主流的半监督学习方法相结合使用。为了减少确证偏见，在两个阶段通过生成模型和子抽样策略进行在线训练。通过在三种模态的13个标准半监督学习基准上进行分类和回归任务的广泛实验验证，表明SemiReward取得了显著的成果。

    Semi-supervised learning (SSL) has witnessed great progress with various improvements in the self-training framework with pseudo labeling. The main challenge is how to distinguish high-quality pseudo labels against the confirmation bias. However, existing pseudo-label selection strategies are limited to pre-defined schemes or complex hand-crafted policies specially designed for classification, failing to achieve high-quality labels, fast convergence, and task versatility simultaneously. To these ends, we propose a Semi-supervised Reward framework (SemiReward) that predicts reward scores to evaluate and filter out high-quality pseudo labels, which is pluggable to mainstream SSL methods in wide task types and scenarios. To mitigate confirmation bias, SemiReward is trained online in two stages with a generator model and subsampling strategy. With classification and regression tasks on 13 standard SSL benchmarks of three modalities, extensive experiments verify that SemiReward achieves sig
    
[^4]: 高维度 SGD 与新兴的异常特征空间相吻合

    High-dimensional SGD aligns with emerging outlier eigenspaces. (arXiv:2310.03010v1 [cs.LG])

    [http://arxiv.org/abs/2310.03010](http://arxiv.org/abs/2310.03010)

    本研究通过研究训练动态和经验海森矩阵以及梯度矩阵的谱的联合演化，证明了在高维混合和多层神经网络的分类任务中，SGD轨迹与海森矩阵和梯度矩阵的新兴低秩异常特征空间吻合。在多层设置中，这种对齐会在每一层发生，并且在收敛到亚优分类器时会表现出秩缺乏。

    

    我们通过随机梯度下降（SGD）和经验海森矩阵和梯度矩阵的谱的联合演化，对训练动态进行了严格的研究。我们证明在多类高维混合和1或2层神经网络的两个典型分类任务中，SGD轨迹迅速与海森矩阵和梯度矩阵的新兴低秩异常特征空间相吻合。此外，在多层设置中，这种对齐发生在每一层，最后一层的异常特征空间在训练过程中演化，并且在SGD收敛到亚优分类器时表现出秩缺乏。这为过去十年中关于在超参数化网络中训练过程中海森矩阵和信息矩阵的谱的广泛数值研究提供了丰富的预测。

    We rigorously study the joint evolution of training dynamics via stochastic gradient descent (SGD) and the spectra of empirical Hessian and gradient matrices. We prove that in two canonical classification tasks for multi-class high-dimensional mixtures and either 1 or 2-layer neural networks, the SGD trajectory rapidly aligns with emerging low-rank outlier eigenspaces of the Hessian and gradient matrices. Moreover, in multi-layer settings this alignment occurs per layer, with the final layer's outlier eigenspace evolving over the course of training, and exhibiting rank deficiency when the SGD converges to sub-optimal classifiers. This establishes some of the rich predictions that have arisen from extensive numerical studies in the last decade about the spectra of Hessian and information matrices over the course of training in overparametrized networks.
    
[^5]: 软凸量化：用凸优化重新思考向量量化

    Soft Convex Quantization: Revisiting Vector Quantization with Convex Optimization. (arXiv:2310.03004v1 [cs.LG])

    [http://arxiv.org/abs/2310.03004](http://arxiv.org/abs/2310.03004)

    本文提出了软凸量化（SCQ）作为向量量化（VQ）的替代方法，通过解决量化输入的码书向量的最优凸组合问题，缓解了VQ面临的实际挑战。

    

    向量量化（VQ）是深度学习中用于提取信息性离散潜在表示的一种众所周知的技术。VQ嵌入模型在包括图像和语音生成在内的一系列应用中取得了令人印象深刻的结果。VQ作为一种参数化的K-means算法，在前向传递中使用单个码书向量将输入进行量化。尽管功能强大，但该技术面临实际挑战，包括码书崩溃、不可区分性和有损压缩。为了缓解上述问题，我们提出了软凸量化（SCQ）作为VQ的直接替代。SCQ的工作方式类似于可微凸优化（DCO）层：在前向传递中，我们求解量化输入的码书向量的最优凸组合。在反向传递中，我们通过前向解的最优性条件利用可区分性。然后，我们引入了一个可扩展的SCQ优化松弛方法，并展示其在CIFAR数据集上的有效性。

    Vector Quantization (VQ) is a well-known technique in deep learning for extracting informative discrete latent representations. VQ-embedded models have shown impressive results in a range of applications including image and speech generation. VQ operates as a parametric K-means algorithm that quantizes inputs using a single codebook vector in the forward pass. While powerful, this technique faces practical challenges including codebook collapse, non-differentiability and lossy compression. To mitigate the aforementioned issues, we propose Soft Convex Quantization (SCQ) as a direct substitute for VQ. SCQ works like a differentiable convex optimization (DCO) layer: in the forward pass, we solve for the optimal convex combination of codebook vectors that quantize the inputs. In the backward pass, we leverage differentiability through the optimality conditions of the forward solution. We then introduce a scalable relaxation of the SCQ optimization and demonstrate its efficacy on the CIFAR-
    
[^6]: 使用物理信息神经网络学习多相流下离心泵的特性参数和动力学

    Learning characteristic parameters and dynamics of centrifugal pumps under multi-phase flow using physics-informed neural networks. (arXiv:2310.03001v1 [cs.LG])

    [http://arxiv.org/abs/2310.03001](http://arxiv.org/abs/2310.03001)

    本文提出了一种基于物理信息神经网络（PINNs）的机器学习模型，用于估计离心泵在多相流下的特性参数和动力学。

    

    电潜泵（ESP）由于其高流量和增压，是油气工业中第二常用的人工提升设备。它们经常需要处理多相流动，这些流体通常包含烃类、水和/或沉积物的混合物。在这种情况下，通常会形成乳液，它是由两种不互溶流体组成的液液流动，其有效粘度和密度与单独的单相流动有所不同。在此背景下，准确建模ESP系统对于优化油田生产和实施控制策略至关重要。然而，由于时间限制和经济原因，实时和直接测量流体和系统特性通常是不可实现的。因此，一般考虑间接方法来估计系统参数。本文提出了一个基于物理信息神经网络（PINNs）的机器学习模型，用于估计关键的系统参数。

    Electrical submersible pumps (ESP) are the second most used artificial lifting equipment in the oil and gas industry due to their high flow rates and boost pressures. They often have to handle multiphase flows, which usually contain a mixture of hydrocarbons, water, and/or sediments. Given these circumstances, emulsions are commonly formed. It is a liquid-liquid flow composed of two immiscible fluids whose effective viscosity and density differ from the single phase separately. In this context, accurate modeling of ESP systems is crucial for optimizing oil production and implementing control strategies. However, real-time and direct measurement of fluid and system characteristics is often impractical due to time constraints and economy. Hence, indirect methods are generally considered to estimate the system parameters. In this paper, we formulate a machine learning model based on Physics-Informed Neural Networks (PINNs) to estimate crucial system parameters. In order to study the effic
    
[^7]: ECoFLaP: 高效的粗到细的逐层剪枝方法用于视觉语言模型

    ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models. (arXiv:2310.02998v1 [cs.CV])

    [http://arxiv.org/abs/2310.02998](http://arxiv.org/abs/2310.02998)

    ECoFLaP提出了一种高效的粗到细的逐层剪枝方法，解决了大型视觉语言模型在压缩和部署时的计算和能耗问题。

    

    大型视觉语言模型（LVLMs）通过整合不同模态的丰富信息，全面理解世界，并在各种多模态下游任务上取得显著的性能提升。然而，由于其巨大的计算/能耗和碳排放，部署LVLMs往往存在问题。这些问题使得采用传统的迭代全局剪枝变得不可行，因为其需要计算整个大型模型的Hessian矩阵以进行稀疏化。相反，最近的研究提出了逐层剪枝方法，避免了全局剪枝的昂贵计算，并根据层内权重的重要性有效压缩模型。然而，这些方法常常由于缺乏全局视角而导致模型压缩不够优化。为了解决大型模型最近高效剪枝方法的这一局限性，我们提出了高效的粗到细的逐层剪枝方法（ECoFLaP）。

    Large Vision-Language Models (LVLMs) can understand the world comprehensively by integrating rich information from different modalities, achieving remarkable performance improvements on various multimodal downstream tasks. However, deploying LVLMs is often problematic due to their massive computational/energy costs and carbon consumption. Such issues make it infeasible to adopt conventional iterative global pruning, which is costly due to computing the Hessian matrix of the entire large model for sparsification. Alternatively, several studies have recently proposed layer-wise pruning approaches to avoid the expensive computation of global pruning and efficiently compress model weights according to their importance within a layer. However, these methods often suffer from suboptimal model compression due to their lack of a global perspective. To address this limitation in recent efficient pruning methods for large models, we propose Efficient Coarse-to-Fine Layer-Wise Pruning (ECoFLaP), 
    
[^8]: IBCL：连续学习中零样本模型生成用于任务权衡

    IBCL: Zero-shot Model Generation for Task Trade-offs in Continual Learning. (arXiv:2310.02995v1 [cs.LG])

    [http://arxiv.org/abs/2310.02995](http://arxiv.org/abs/2310.02995)

    IBCL提出了一种用于连续学习中任务权衡的零样本模型生成方法，通过更新知识库并利用模型参数分布的凸包形式，实现不同任务性能之间的权衡偏好。

    

    类似于通用的多任务学习，连续学习具有多目标优化的特性，因此面临着不同任务性能之间的权衡。也就是说，为了优化当前任务分布，可能需要在一些先前的任务上牺牲性能。这意味着在不同时间点存在多个帕累托最优的模型，每个模型都解决了不同的任务性能权衡问题。研究人员讨论了如何训练特定的模型来解决特定的权衡偏好。然而，现有的算法需要与偏好数量成比例的训练开销，当存在多个甚至是无限多个偏好时，这是一个巨大的负担。作为响应，我们提出了Imprecise Bayesian Continual Learning (IBCL)。在新任务出现时，IBCL(1)通过模型参数分布的凸包形式更新知识库，(2)获得了特定模型，以实现零样本的任务权衡偏好。

    Like generic multi-task learning, continual learning has the nature of multi-objective optimization, and therefore faces a trade-off between the performance of different tasks. That is, to optimize for the current task distribution, it may need to compromise performance on some previous tasks. This means that there exist multiple models that are Pareto-optimal at different times, each addressing a distinct task performance trade-off. Researchers have discussed how to train particular models to address specific trade-off preferences. However, existing algorithms require training overheads proportional to the number of preferences -- a large burden when there are multiple, possibly infinitely many, preferences. As a response, we propose Imprecise Bayesian Continual Learning (IBCL). Upon a new task, IBCL (1) updates a knowledge base in the form of a convex hull of model parameter distributions and (2) obtains particular models to address task trade-off preferences with zero-shot. That is,
    
[^9]: 多物理学预训练用于物理代理模型

    Multiple Physics Pretraining for Physical Surrogate Models. (arXiv:2310.02994v1 [cs.LG])

    [http://arxiv.org/abs/2310.02994](http://arxiv.org/abs/2310.02994)

    多物理学预训练是一种用于物理代理建模的自回归预训练方法，通过训练大型代理模型同时预测多个异构物理系统的动力学，学习在不同物理任务中广泛适用的特征。实验证明，单个MPP预训练的变换器可以在所有预训练子任务上与或超过特定任务的基准结果，无需微调，并且在下游任务中，微调MPP训练的模型相较于从头训练的模型，对新物理的预测结果更准确。

    

    我们引入了一种多物理学预训练（MPP）的方法，这是一种自回归任务不可知的预训练方法，用于物理代理建模。MPP通过训练大型代理模型同时预测多个异构物理系统的动力学，学习在不同物理任务中广泛适用的特征。为了有效学习，在这种设置中，我们引入了一种共享嵌入和归一化策略，将多个系统的字段投影到一个共享嵌入空间中。我们在一个涉及流体力学的广泛基准测试中验证了我们方法的有效性。我们表明，单个MPP预训练的变换器能够在所有预训练子任务上与或超过特定任务的基准结果，而无需微调。对于下游任务，我们证明微调MPP训练的模型相较于从头训练的模型，在多个时间步骤上对新物理的预测结果更准确。

    We introduce multiple physics pretraining (MPP), an autoregressive task-agnostic pretraining approach for physical surrogate modeling. MPP involves training large surrogate models to predict the dynamics of multiple heterogeneous physical systems simultaneously by learning features that are broadly useful across diverse physical tasks. In order to learn effectively in this setting, we introduce a shared embedding and normalization strategy that projects the fields of multiple systems into a single shared embedding space. We validate the efficacy of our approach on both pretraining and downstream tasks over a broad fluid mechanics-oriented benchmark. We show that a single MPP-pretrained transformer is able to match or outperform task-specific baselines on all pretraining sub-tasks without the need for finetuning. For downstream tasks, we demonstrate that finetuning MPP-trained models results in more accurate predictions across multiple time-steps on new physics compared to training from
    
[^10]: xVal: 大型语言模型的连续数字编码

    xVal: A Continuous Number Encoding for Large Language Models. (arXiv:2310.02989v1 [stat.ML])

    [http://arxiv.org/abs/2310.02989](http://arxiv.org/abs/2310.02989)

    xVal是一种连续数字编码方案，通过使用单个标记来表示任何实数。与现有的数字编码方案相比，xVal更加高效，并且在泛化性能上表现更好。

    

    由于数字令牌化的独特困难，大型语言模型尚未广泛用于科学数据集的分析。我们提出了xVal，一种数字编码方案，可以使用单个标记来表示任何实数。xVal通过将专用嵌入向量按数字值进行缩放来表示给定的实数。结合修改后的数字推断方法，该策略使模型在考虑作为从输入字符串的数字到输出字符串的数字的映射时成为端到端连续的。这导致了一种更适用于科学领域应用的归纳偏差。我们在许多合成和现实世界数据集上进行了实证评估。与现有的数字编码方案相比，我们发现xVal在令牌效率和泛化性能上表现更好。

    Large Language Models have not yet been broadly adapted for the analysis of scientific datasets due in part to the unique difficulties of tokenizing numbers. We propose xVal, a numerical encoding scheme that represents any real number using just a single token. xVal represents a given real number by scaling a dedicated embedding vector by the number value. Combined with a modified number-inference approach, this strategy renders the model end-to-end continuous when considered as a map from the numbers of the input string to those of the output string. This leads to an inductive bias that is generally more suitable for applications in scientific domains. We empirically evaluate our proposal on a number of synthetic and real-world datasets. Compared with existing number encoding schemes, we find that xVal is more token-efficient and demonstrates improved generalization.
    
[^11]: 方差减少的 Halpern 迭代在有限和单调包含中的应用

    Variance Reduced Halpern Iteration for Finite-Sum Monotone Inclusions. (arXiv:2310.02987v1 [cs.LG])

    [http://arxiv.org/abs/2310.02987](http://arxiv.org/abs/2310.02987)

    提出了使用方差减少的 Halpern 迭代来优化有限和单调包含问题的求解过程，具有更好的复杂度保证。

    

    依赖对抗稳健性或多智能体环境的机器学习方法引发了解决博弈均衡问题的需求。在这些应用中，具有可计算逼近误差的方法非常理想，因为它们提供了可验证的终止准则。在这些应用的基础上，我们研究了模拟广泛类别均衡问题的有限和单调包含问题。我们的主要贡献是改进了经典的 Halpern 迭代方法，利用方差减少获得改进的算法复杂度保证，在有限和的 $n$ 个组成操作符中，“平均”地是互补协同或Lipschitz连续和单调，参数为 $L$。我们的方法的结果预测了最后的迭代和一个（compu）

    Machine learning approaches relying on such criteria as adversarial robustness or multi-agent settings have raised the need for solving game-theoretic equilibrium problems. Of particular relevance to these applications are methods targeting finite-sum structure, which generically arises in empirical variants of learning problems in these contexts. Further, methods with computable approximation errors are highly desirable, as they provide verifiable exit criteria. Motivated by these applications, we study finite-sum monotone inclusion problems, which model broad classes of equilibrium problems. Our main contributions are variants of the classical Halpern iteration that employ variance reduction to obtain improved complexity guarantees in which $n$ component operators in the finite sum are ``on average'' either cocoercive or Lipschitz continuous and monotone, with parameter $L$. The resulting oracle complexity of our methods, which provide guarantees for the last iterate and for a (compu
    
[^12]: 在灾难场景中探索中断的点对点通信对完全去中心化学习的影响

    Exploring the Impact of Disrupted Peer-to-Peer Communications on Fully Decentralized Learning in Disaster Scenarios. (arXiv:2310.02986v1 [cs.LG])

    [http://arxiv.org/abs/2310.02986](http://arxiv.org/abs/2310.02986)

    在灾难场景中，完全去中心化学习可以帮助解决通信基础设施中断或不可用导致的传统集中式学习任务无法进行的问题。

    

    完全去中心化学习使得学习资源和决策能力可以分布在多个用户设备或节点上，由于其保护隐私和去中心化的特性，它正迅速变得流行起来。重要的是，这种学习过程的众包机制使得系统在一些节点受到影响或断开连接时仍然可以继续运转。在灾难场景中，通信基础设施和集中式系统可能会中断或完全不可用，这阻碍了在这些环境中进行标准的集中式学习任务的可能性。因此，完全去中心化的学习可以在这种情况下提供帮助。然而，从集中式到点对点通信的过渡引入了学习过程与节点之间的通信图拓扑的依赖关系。在灾难场景中，即使是点对点通信也容易出现突然的变化，如设备耗尽电池或断开连接。

    Fully decentralized learning enables the distribution of learning resources and decision-making capabilities across multiple user devices or nodes, and is rapidly gaining popularity due to its privacy-preserving and decentralized nature. Importantly, this crowdsourcing of the learning process allows the system to continue functioning even if some nodes are affected or disconnected. In a disaster scenario, communication infrastructure and centralized systems may be disrupted or completely unavailable, hindering the possibility of carrying out standard centralized learning tasks in these settings. Thus, fully decentralized learning can help in this case. However, transitioning from centralized to peer-to-peer communications introduces a dependency between the learning process and the topology of the communication graph among nodes. In a disaster scenario, even peer-to-peer communications are susceptible to abrupt changes, such as devices running out of battery or getting disconnected fro
    
[^13]: 缩放定律在联想记忆中的应用

    Scaling Laws for Associative Memories. (arXiv:2310.02984v1 [stat.ML])

    [http://arxiv.org/abs/2310.02984](http://arxiv.org/abs/2310.02984)

    本文研究了应用于联想记忆中的缩放定律，通过高维矩阵和嵌入的外积来模拟内层Transformer语言模型。作者推导出了与样本数量和参数大小相关的精确缩放定律，并验证了理论结果的有效性。同时，作者还通过大量实验展示了存储记忆关联的细粒度可视化。

    

    学习很可能涉及到抽象规则的发现和记忆。本文旨在研究联想记忆机制。我们的模型基于高维矩阵，由嵌入的外积组成，与Transformer语言模型的内层相关。我们推导出关于样本数量和参数规模的精确缩放定律，并讨论了不同估计器的统计效率，包括基于优化的算法。我们进行了大量的数值实验，以验证和解释理论结果，包括对存储记忆关联的细粒度可视化。

    Learning arguably involves the discovery and memorization of abstract rules. The aim of this paper is to study associative memory mechanisms. Our model is based on high-dimensional matrices consisting of outer products of embeddings, which relates to the inner layers of transformer language models. We derive precise scaling laws with respect to sample size and parameter size, and discuss the statistical efficiency of different estimators, including optimization-based algorithms. We provide extensive numerical experiments to validate and interpret theoretical results, including fine-grained visualizations of the stored memory associations.
    
[^14]: 永远不要从头开始训练：公正比较长序列模型需要数据驱动的先验知识

    Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors. (arXiv:2310.02980v1 [cs.LG])

    [http://arxiv.org/abs/2310.02980](http://arxiv.org/abs/2310.02980)

    本文研究表明使用随机初始化会导致对架构差异的严重高估，而使用标准消噪目标进行预训练可以在多种架构上实现显著的性能提升，并将Transformers与状态空间模型之间的差距缩小到很小。与之前的研究不同的是，我们发现当正确预训练时，普通的Transformers在Long Range Arena上的性能与S4相匹配，并且在PathX-256任务上改进了SSMs的最佳结果20个百分点。

    

    建模序列之间的长程依赖一直是机器学习中的目标，并导致了一些架构，如状态空间模型，在处理长序列时比Transformers有显著的优势。然而，这些令人印象深刻的经验性进展主要是在随机初始化并通过预测输入序列的目标标签进行训练的基准测试（例如Long Range Arena）上展示出来的。在这项工作中，我们展示了随机初始化导致对架构之间差异的严重高估，并且使用标准消噪目标进行预训练（仅使用下游任务数据）可以在多种架构上实现显著的收益，并且可以在Transformers和状态空间模型（SSMs）之间得到很小的差距。与之前的研究形成鲜明对比的是，我们发现当正确预训练时，普通的Transformers在Long Range Arena上与S4的性能相匹配，并且我们在PathX-256任务上将SSMs的最佳报告结果提高了20个百分点。

    Modeling long-range dependencies across sequences is a longstanding goal in machine learning and has led to architectures, such as state space models, that dramatically outperform Transformers on long sequences. However, these impressive empirical gains have been by and large demonstrated on benchmarks (e.g. Long Range Arena), where models are randomly initialized and trained to predict a target label from an input sequence. In this work, we show that random initialization leads to gross overestimation of the differences between architectures and that pretraining with standard denoising objectives, using $\textit{only the downstream task data}$, leads to dramatic gains across multiple architectures and to very small gaps between Transformers and state space models (SSMs). In stark contrast to prior works, we find vanilla Transformers to match the performance of S4 on Long Range Arena when properly pretrained, and we improve the best reported results of SSMs on the PathX-256 task by 20 
    
[^15]: T$^3$Bench：标注目前在文本到3D生成领域的进展的基准测试（arXiv:2310.02977v1 [cs.CV]）

    T$^3$Bench: Benchmarking Current Progress in Text-to-3D Generation. (arXiv:2310.02977v1 [cs.CV])

    [http://arxiv.org/abs/2310.02977](http://arxiv.org/abs/2310.02977)

    T$^3$Bench是第一个综合的文本到3D基准测试，它包含了多个复杂程度的文本提示，并引入了两个自动度量标准来评估生成的3D场景的主观质量和文本对齐性能。

    

    近期的文本到3D方法利用强大的预训练扩散模型来优化NeRF。值得注意的是，这些方法能够在没有3D数据训练的情况下生成高质量的3D场景。由于任务的开放性，大多数研究通过主观案例研究和用户实验证明其结果，从而在定量上回答“文本到3D的当前进展如何？”这个问题上存在挑战。在本文中，我们介绍了T$^3$Bench，这是一个第一个综合的文本到3D基准测试，包含三个不断增加复杂性的文本提示，专门为3D生成而设计。为了评估主观质量和文本对齐性，我们提出了基于3D内容产生的多视图图像的两个自动度量标准。质量度量结合了多视图文本-图像分数和区域卷积以检测质量和视角不一致性。对齐度量使用多视图字幕和大型语言模型（LLM）e

    Recent methods in text-to-3D leverage powerful pretrained diffusion models to optimize NeRF. Notably, these methods are able to produce high-quality 3D scenes without training on 3D data. Due to the open-ended nature of the task, most studies evaluate their results with subjective case studies and user experiments, thereby presenting a challenge in quantitatively addressing the question: How has current progress in Text-to-3D gone so far? In this paper, we introduce T$^3$Bench, the first comprehensive text-to-3D benchmark containing diverse text prompts of three increasing complexity levels that are specially designed for 3D generation. To assess both the subjective quality and the text alignment, we propose two automatic metrics based on multi-view images produced by the 3D contents. The quality metric combines multi-view text-image scores and regional convolution to detect quality and view inconsistency. The alignment metric uses multi-view captioning and Large Language Model (LLM) e
    
[^16]: 在重尾波段的完全自适应遗憾最小化领域中的研究

    Towards Fully Adaptive Regret Minimization in Heavy-Tailed Bandits. (arXiv:2310.02975v1 [cs.LG])

    [http://arxiv.org/abs/2310.02975](http://arxiv.org/abs/2310.02975)

    本文研究了在重尾波段问题中完全自适应的遗憾最小化，提出了随机自适应重尾波段问题，并证明了适应性算法相对于标准设置会有更高的遗憾。

    

    重尾分布在金融到电信等多种环境中自然而然地出现。虽然在次高斯或有界支撑奖励下的遗憾最小化已被广泛研究，但在重尾分布上的学习只在过去十年中受到关注。在随机重尾波段问题中，一个代理在假设分布有有界最大阶的有限矩的情况下学习，这些矩被常数u一致有界，对于某个ε∈(0,1]。据我们所知，文献中只提供需要这两个量作为输入的算法。在本文中，我们研究了随机自适应重尾波段问题，这是标准设置的一个变种，其中代理对ε和u均不知晓。我们表明，适应性是存在代价的，并引入对于任何自适应算法遗憾的两个下界，意味着相对于标准设置有更高的遗憾。最后，我们引入一种特定的分布假设。

    Heavy-tailed distributions naturally arise in many settings, from finance to telecommunications. While regret minimization under sub-Gaussian or bounded support rewards has been widely studied, learning on heavy-tailed distributions only gained popularity over the last decade. In the stochastic heavy-tailed bandit problem, an agent learns under the assumption that the distributions have finite moments of maximum order $1+\epsilon$ which are uniformly bounded by a constant $u$, for some $\epsilon \in (0,1]$. To the best of our knowledge, literature only provides algorithms requiring these two quantities as an input. In this paper, we study the stochastic adaptive heavy-tailed bandit, a variation of the standard setting where both $\epsilon$ and $u$ are unknown to the agent. We show that adaptivity comes at a cost, introducing two lower bounds on the regret of any adaptive algorithm, implying a higher regret w.r.t. the standard setting. Finally, we introduce a specific distributional ass
    
[^17]: 快速、表达力强的SE$(n)$等变网络通过在位置-方向空间中共享权重

    Fast, Expressive SE$(n)$ Equivariant Networks through Weight-Sharing in Position-Orientation Space. (arXiv:2310.02970v1 [cs.LG])

    [http://arxiv.org/abs/2310.02970](http://arxiv.org/abs/2310.02970)

    该论文通过在位置-方向空间中共享权重，提出了一种快速、表达力强的SE$(n)$等变网络。他们基于同态空间理论，推导出几何优化的边属性，并将权重共享形式化为对等处理相同点对的消息函数。他们在处理3D点云时，开发了一个高效的等变群卷积网络，并选择了$\mathbb{R}^3 {\times} S^2$作为最佳的处理空间。

    

    我们基于同态空间理论推导出用于灵活的消息传递框架的“几何优化边属性”。我们将卷积神经网络中的权重共享形式化为对等地处理并且应该被平等对待的点对的消息函数共享。我们定义了等价类，这些等价类在群中进行变换时是相同的，并且推导出唯一标识这些类别的属性。通过在这些属性上进行条件化，可以实现权重共享。作为该理论的应用，我们开发了一个高效的等变群卷积网络来处理3D点云。同态空间理论告诉我们如何在位置$\mathbb{R}^3$、位置和方向$\mathbb{R}^3 {\times} S^2$的同态空间以及群SE$(3)$上的特征图上进行群卷积。在这些选择中，$\mathbb{R}^3 {\times} S^2$是一个最佳选择，因为它具有处理方向信息的能力。

    Based on the theory of homogeneous spaces we derive \textit{geometrically optimal edge attributes} to be used within the flexible message passing framework. We formalize the notion of weight sharing in convolutional networks as the sharing of message functions over point-pairs that should be treated equally. We define equivalence classes of point-pairs that are identical up to a transformation in the group and derive attributes that uniquely identify these classes. Weight sharing is then obtained by conditioning message functions on these attributes. As an application of the theory, we develop an efficient equivariant group convolutional network for processing 3D point clouds. The theory of homogeneous spaces tells us how to do group convolutions with feature maps over the homogeneous space of positions $\mathbb{R}^3$, position and orientations $\mathbb{R}^3 {\times} S^2$, and the group SE$(3)$ itself. Among these, $\mathbb{R}^3 {\times} S^2$ is an optimal choice due to the ability to 
    
[^18]: 双圆锥代理用于交流最优功率流问题

    Dual Conic Proxies for AC Optimal Power Flow. (arXiv:2310.02969v1 [cs.LG])

    [http://arxiv.org/abs/2310.02969](http://arxiv.org/abs/2310.02969)

    本文提出了一种基于双圆锥代理的方法来求解交流最优功率流问题，并通过自监督学习方案来辅助训练，实验证明了该方法的效率和可扩展性。

    

    近年来，人们对基于机器学习的交流最优功率流问题（AC-OPF）优化代理的发展表现出了极大的兴趣。虽然在预测高质量原始解方面取得了显著进展，但现有的基于学习的方法无法为AC-OPF提供有效的对偶界限。本文通过训练AC-OPF的一个凸松弛的优化代理来填补这一空白。具体而言，本文考虑了AC-OPF的二阶圆锥（SOC）松弛，并提出了一种新的对偶架构，嵌入了一个快速、可微分的（对偶）可行性恢复，从而提供有效的对偶界限。本文将这种新架构与自监督学习方案相结合，减轻了昂贵的训练数据生成需求。对中等和大规模电力网络进行了大量的数值实验，证明了所提方法的效率和可扩展性。

    In recent years, there has been significant interest in the development of machine learning-based optimization proxies for AC Optimal Power Flow (AC-OPF). Although significant progress has been achieved in predicting high-quality primal solutions, no existing learning-based approach can provide valid dual bounds for AC-OPF. This paper addresses this gap by training optimization proxies for a convex relaxation of AC-OPF. Namely, the paper considers a second-order cone (SOC) relaxation of ACOPF, and proposes a novel dual architecture that embeds a fast, differentiable (dual) feasibility recovery, thus providing valid dual bounds. The paper combines this new architecture with a self-supervised learning scheme, which alleviates the need for costly training data generation. Extensive numerical experiments on medium- and large-scale power grids demonstrate the efficiency and scalability of the proposed methodology.
    
[^19]: 合模式化肽的顺序和图形路径

    Co-modeling the Sequential and Graphical Route for Peptide. (arXiv:2310.02964v1 [cs.LG])

    [http://arxiv.org/abs/2310.02964](http://arxiv.org/abs/2310.02964)

    本论文提出了一种肽合模式化方法，使用对比学习框架来增强从顺序和图形模型中学到的表示的相互信息，以提高肽的判别性能。

    

    肽是由多个氨基酸的脱水缩合形成的。肽的主要结构可以表示为氨基酸序列或由原子和化学键组成的分子图。先前的研究表明，针对顺序和图形肽形式的深度学习路径在下游任务上表现相似。尽管这些模型学习了同一种肽的表示，但我们发现它们解释其预测的方式不同。将顺序和图形模型视为从不同角度进行推理的两个专家，我们致力于融合专家知识，丰富学到的表示以提高判别性能。为实现这一目标，我们提出了一种肽合模式化方法RepCon，它采用对比学习框架，增强从解耦的顺序和图形端到端模型的表示的相互信息。

    Peptides are formed by the dehydration condensation of multiple amino acids. The primary structure of a peptide can be represented either as an amino acid sequence or as a molecular graph consisting of atoms and chemical bonds. Previous studies have indicated that deep learning routes specific to sequential and graphical peptide forms exhibit comparable performance on downstream tasks. Despite the fact that these models learn representations of the same modality of peptides, we find that they explain their predictions differently. Considering sequential and graphical models as two experts making inferences from different perspectives, we work on fusing expert knowledge to enrich the learned representations for improving the discriminative performance. To achieve this, we propose a peptide co-modeling method, RepCon, which employs a contrastive learning-based framework to enhance the mutual information of representations from decoupled sequential and graphical end-to-end models. It cons
    
[^20]: 使用机器学习模型进行信用卡评分预测：一个新数据集

    Credit card score prediction using machine learning models: A new dataset. (arXiv:2310.02956v1 [cs.LG])

    [http://arxiv.org/abs/2310.02956](http://arxiv.org/abs/2310.02956)

    本研究探索了利用机器学习模型对信用卡违约进行预测的方法，并提出了一个新的信用卡评分数据集。实验结果表明，多层感知器（MLP）模型在预测性能上表现最佳。

    

    近年来，信用卡的使用量不断增加，为了最小化潜在风险，急需信用卡评估方法。本研究调查了利用机器学习模型进行信用卡违约预测系统的应用。主要目标是研究在新提出的信用卡评分数据集上表现最佳的机器学习模型。这个新数据集包括信用卡交易历史和客户档案，并使用了多种机器学习算法进行了测试，包括逻辑回归、决策树、随机森林、多层感知器（MLP）神经网络、XGBoost和LightGBM。为了准备机器学习模型的数据，我们进行了数据预处理、特征提取、特征选择和数据平衡技术。实验结果表明，在真正阳性率方面，MLP在预测性能上优于逻辑回归、决策树、随机森林、LightGBM和XGBoost，实现了最佳表现。

    The use of credit cards has recently increased, creating an essential need for credit card assessment methods to minimize potential risks. This study investigates the utilization of machine learning (ML) models for credit card default prediction system. The main goal here is to investigate the best-performing ML model for new proposed credit card scoring dataset. This new dataset includes credit card transaction histories and customer profiles, is proposed and tested using a variety of machine learning algorithms, including logistic regression, decision trees, random forests, multi layer perceptron (MLP) neural network, XGBoost, and LightGBM. To prepare the data for machine learning models, we perform data pre-proccessing, feature extraction, feature selection, and data balancing techniques. Experimental results demonstrate that MLP outperforms logistic regression, decision trees, random forests, LightGBM, and XGBoost in terms of predictive performance in true positive rate, achieving 
    
[^21]: Fisher-Rao梯度流在Polish空间中对熵正则化Markov决策过程的研究

    A Fisher-Rao gradient flow for entropy-regularised Markov decision processes in Polish spaces. (arXiv:2310.02951v1 [math.OC])

    [http://arxiv.org/abs/2310.02951](http://arxiv.org/abs/2310.02951)

    该论文研究了在Polish空间中的熵正则化Markov决策过程上的Fisher-Rao策略梯度流的全局收敛性和指数收敛性，并证明了梯度流在梯度评估方面的稳定性，为自然策略梯度流的性能提供了洞见。

    

    我们研究了在Polish状态和动作空间中无限时域的熵正则化的Markov决策过程的Fisher-Rao策略梯度流的全局收敛性。这个流是策略镜像下降方法的连续时间类比。我们证明了梯度流的全局良定义性，并展示了它对最优策略的指数收敛性。此外，我们证明了梯度流在梯度评估方面的稳定性，为对数线性策略参数化的自然策略梯度流的性能提供了洞见。为了克服目标函数非凸性和熵正则化引起的不连续性所带来的挑战，我们利用性能差别引理和梯度与镜像下降流之间的对偶关系。

    We study the global convergence of a Fisher-Rao policy gradient flow for infinite-horizon entropy-regularised Markov decision processes with Polish state and action space. The flow is a continuous-time analogue of a policy mirror descent method. We establish the global well-posedness of the gradient flow and demonstrate its exponential convergence to the optimal policy. Moreover, we prove the flow is stable with respect to gradient evaluation, offering insights into the performance of a natural policy gradient flow with log-linear policy parameterisation. To overcome challenges stemming from the lack of the convexity of the objective function and the discontinuity arising from the entropy regulariser, we leverage the performance difference lemma and the duality relationship between the gradient and mirror descent flows.
    
[^22]: 阴影对齐：轻松颠覆安全对齐的语言模型

    Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models. (arXiv:2310.02949v1 [cs.CL])

    [http://arxiv.org/abs/2310.02949](http://arxiv.org/abs/2310.02949)

    该论文探讨了阴影对齐这种新的攻击方式，通过调整少量恶意示例，安全对齐的语言模型可以被轻松颠覆生成有害的内容，同时仍然可以正确响应常规查询。

    

    警告：本论文包含有害语言的例子，建议读者慎重。强大的大型语言模型（LLMs）的逐渐开放释放，通过降低数据注释和计算的核心成本，促进了下游应用的发展。为了确保AI的安全性，进行了广泛的安全对齐措施，以保护这些模型免受恶意使用（主要是硬提示攻击）。然而，在这种看似坚固的盔甲背后，可能潜伏着一个阴影。通过仅调整100个恶意示例，使用1个GPU小时，这些安全对齐的LLMs可以轻松地被颠覆以生成有害内容。形式上，我们将一种新攻击称为阴影对齐：利用少量数据可以使安全对齐模型适应有害任务，而不会牺牲模型的有用性。值得注意的是，被颠覆的模型仍然保留其对常规查询的适当响应能力。在5个发行的8个模型上进行的实验证实了这一点。

    Warning: This paper contains examples of harmful language, and reader discretion is recommended. The increasing open release of powerful large language models (LLMs) has facilitated the development of downstream applications by reducing the essential cost of data annotation and computation. To ensure AI safety, extensive safety-alignment measures have been conducted to armor these models against malicious use (primarily hard prompt attack). However, beneath the seemingly resilient facade of the armor, there might lurk a shadow. By simply tuning on 100 malicious examples with 1 GPU hour, these safely aligned LLMs can be easily subverted to generate harmful content. Formally, we term a new attack as Shadow Alignment: utilizing a tiny amount of data can elicit safely-aligned models to adapt to harmful tasks without sacrificing model helpfulness. Remarkably, the subverted models retain their capability to respond appropriately to regular inquiries. Experiments across 8 models released by 5
    
[^23]: HappyFeat -- 一种面向临床应用的交互和高效的BCI框架

    HappyFeat -- An interactive and efficient BCI framework for clinical applications. (arXiv:2310.02948v1 [q-bio.NC])

    [http://arxiv.org/abs/2310.02948](http://arxiv.org/abs/2310.02948)

    HappyFeat是一种面向临床应用的交互和高效的BCI框架，通过一个方便的GUI和参数自动化的帮助，使得基于运动想象的BCI实验更加容易，并能在时间受限的环境中实现良好的性能。

    

    脑机接口（BCI）系统允许用户通过将大脑活动转化为命令来执行动作。这类系统通常需要进行训练阶段，包括通过使用记录的信号的特定特征来训练分类算法，以区分不同的心理状态。在临床背景下，如中风康复，对于BCI的性能和特征选择的培训阶段有特定的要求。本文介绍了HappyFeat，一种软件，在单一便利的图形用户界面和实验或分析参数的自动化的帮助下，使基于运动想象（MI）的BCI实验更加容易。结果工作流程可以轻松选择最佳特征，有助于在时间受限的环境中实现良好的BCI性能。基于功能连通性的替代特征可以与功率谱密度相比或结合使用。

    Brain-Computer Interface (BCI) systems allow users to perform actions by translating their brain activity into commands. Such systems usually need a training phase, consisting in training a classification algorithm to discriminate between mental states using specific features from the recorded signals. This phase of feature selection and training is crucial for BCI performance and presents specific constraints to be met in a clinical context, such as post-stroke rehabilitation.  In this paper, we present HappyFeat, a software making Motor Imagery (MI) based BCI experiments easier, by gathering all necessary manipulations and analysis in a single convenient GUI and via automation of experiment or analysis parameters. The resulting workflow allows for effortlessly selecting the best features, helping to achieve good BCI performance in time-constrained environments. Alternative features based on Functional Connectivity can be used and compared or combined with Power Spectral Density, allo
    
[^24]: 在随机模型预测控制中在线约束加紧：一种回归方法。

    Online Constraint Tightening in Stochastic Model Predictive Control: A Regression Approach. (arXiv:2310.02942v1 [eess.SY])

    [http://arxiv.org/abs/2310.02942](http://arxiv.org/abs/2310.02942)

    本文提出了一种数据驱动方法，用于在线学习随机模型预测控制中的约束加紧参数。通过将约束加紧参数选择问题重新表述为二进制回归问题，并利用高斯过程模型进行学习，实现了在线学习约束加紧参数的目标。

    

    解决概率约束的随机最优控制问题是控制领域的一项重要挑战，因为在很少的特殊情况下不存在解析解。一种常见且计算效率高的方法是将概率约束重新表述为具有约束加紧参数的硬约束。然而，在这种方法中，选取约束加紧参数仍然具有挑战性，并且只能在已知过程噪声分布的情况下获得保证。此外，概率约束通常无法得到严格满足，导致成本过高。本文提出一种在线学习约束加紧参数的数据驱动方法。为此，我们将闭环的约束加紧参数选择问题重新表述为二进制回归问题。接着，我们利用高度表达能力的高斯过程模型进行学习。

    Solving chance-constrained stochastic optimal control problems is a significant challenge in control. This is because no analytical solutions exist for up to a handful of special cases. A common and computationally efficient approach for tackling chance-constrained stochastic optimal control problems consists of reformulating the chance constraints as hard constraints with a constraint-tightening parameter. However, in such approaches, the choice of constraint-tightening parameter remains challenging, and guarantees can mostly be obtained assuming that the process noise distribution is known a priori. Moreover, the chance constraints are often not tightly satisfied, leading to unnecessarily high costs. This work proposes a data-driven approach for learning the constraint-tightening parameters online during control. To this end, we reformulate the choice of constraint-tightening parameter for the closed-loop as a binary regression problem. We then leverage a highly expressive \gls{gp} m
    
[^25]: Hoeffding不等式在具有广义可集中条件的马尔可夫链中的应用

    Hoeffding's Inequality for Markov Chains under Generalized Concentrability Condition. (arXiv:2310.02941v1 [stat.ML])

    [http://arxiv.org/abs/2310.02941](http://arxiv.org/abs/2310.02941)

    本文研究了在广义可集中条件下的马尔可夫链的Hoeffding不等式，拓展了现有的马尔可夫链Hoeffding型不等式的应用范围。通过应用该框架到机器学习领域，我们得到了几个非渐近分析的结果。

    

    本文研究了在通过积分概率度量(IPM)定义的广义可集中条件下的马尔可夫链的Hoeffding不等式。广义可集中条件建立了一个框架，可以插值和扩展现有的马尔可夫链Hoeffding型不等式的假设。我们的框架的灵活性使得Hoeffding不等式可以应用于传统意义上的非自关马尔可夫链。我们通过将我们的框架应用于机器学习领域中的几个非渐近分析来证明其实用性，包括：(i) 带有马尔可夫样本的经验风险最小化的一般化界限，(ii) SGD的Ployak-Ruppert平均的有限样本保证，以及(iii) 具有广义状态空间的休息马尔可夫赌博机的新的后悔界限。

    This paper studies Hoeffding's inequality for Markov chains under the generalized concentrability condition defined via integral probability metric (IPM). The generalized concentrability condition establishes a framework that interpolates and extends the existing hypotheses of Markov chain Hoeffding-type inequalities. The flexibility of our framework allows Hoeffding's inequality to be applied beyond the ergodic Markov chains in the traditional sense. We demonstrate the utility by applying our framework to several non-asymptotic analyses arising from the field of machine learning, including (i) a generalization bound for empirical risk minimization with Markovian samples, (ii) a finite sample guarantee for Ployak-Ruppert averaging of SGD, and (iii) a new regret bound for rested Markovian bandits with general state space.
    
[^26]: 对气候信息的大规模语言模型进行评估

    Assessing Large Language Models on Climate Information. (arXiv:2310.02932v1 [cs.CL])

    [http://arxiv.org/abs/2310.02932](http://arxiv.org/abs/2310.02932)

    本研究提出了一个基于科学传播原则的综合评估框架，评估了大规模语言模型在气候变化信息中的表现，能够在回答气候变化主题方面提供细粒度的分析。

    

    理解气候变化对我们的影响，了解可用的解决方案，是赋予个人和社区减缓和适应气候变化的重要步骤。随着大规模语言模型（LLMs）的普及，有必要评估它们在这个领域的能力。本研究提出了一个基于科学传播原则的综合评估框架，以分析LLM对气候变化主题的回答。我们的框架强调回答的呈现和认识上的适当性，为LLM生成提供了细粒度的分析。覆盖了8个维度，我们的框架能够识别模型输出中的30个不同问题。该任务是一个现实世界中的例子，这个领域存在越来越多的具有挑战性的问题，AI可以补充和提升人类的表现。我们引入了一种新颖而实用的可扩展监督协议，利用AI辅助并依靠具有相关教育背景的评估员。我们评估了几个最近的LLM，并进行了实证评估。

    Understanding how climate change affects us and learning about available solutions are key steps toward empowering individuals and communities to mitigate and adapt to it. As Large Language Models (LLMs) rise in popularity, it is necessary to assess their capability in this domain. In this study, we present a comprehensive evaluation framework, grounded in science communication principles, to analyze LLM responses to climate change topics. Our framework emphasizes both the presentational and epistemological adequacy of answers, offering a fine-grained analysis of LLM generations. Spanning 8 dimensions, our framework discerns up to 30 distinct issues in model outputs. The task is a real-world example of a growing number of challenging problems where AI can complement and lift human performance. We introduce a novel and practical protocol for scalable oversight that uses AI Assistance and relies on raters with relevant educational backgrounds. We evaluate several recent LLMs and conduct 
    
[^27]: 咽喉癌患者预测结果的图数据建模

    Graph data modelling for outcome prediction in oropharyngeal cancer patients. (arXiv:2310.02931v1 [cs.CV])

    [http://arxiv.org/abs/2310.02931](http://arxiv.org/abs/2310.02931)

    本研究首次使用基于计算机断层扫描的放射学特征，提出了一种用于咽喉癌患者预测结果的患者超图网络（PHGN）。研究还将模型扩展到进行事件发生时间分析，并与GNN和基准线性模型进行比较。

    

    图神经网络（GNNs）在医学领域中越来越受欢迎，用于疾病分类和结果预测的任务。由于患者数据不容易作为图的形式获得，大部分现有的方法要么手动定义患者图，要么基于患者间的配对相似性学习一个潜在的图。最近引入的超图神经网络（HGNN）方法还利用超图来表示患者之间的潜在高阶关联。在本研究中，我们首次研究了一种患者超图网络（PHGN），并在归纳学习设置下，使用基于计算机断层扫描（CT）的放射学特征对咽喉癌（OPC）患者进行二元结果预测。此外，我们还将所提出的模型扩展到进行事件发生时间分析，并与GNN和基准线性模型进行比较。

    Graph neural networks (GNNs) are becoming increasingly popular in the medical domain for the tasks of disease classification and outcome prediction. Since patient data is not readily available as a graph, most existing methods either manually define a patient graph, or learn a latent graph based on pairwise similarities between the patients. There are also hypergraph neural network (HGNN)-based methods that were introduced recently to exploit potential higher order associations between the patients by representing them as a hypergraph. In this work, we propose a patient hypergraph network (PHGN), which has been investigated in an inductive learning setup for binary outcome prediction in oropharyngeal cancer (OPC) patients using computed tomography (CT)-based radiomic features for the first time. Additionally, the proposed model was extended to perform time-to-event analyses, and compared with GNN and baseline linear models.
    
[^28]: 自适应正则化的最优传输

    Optimal Transport with Adaptive Regularisation. (arXiv:2310.02925v1 [cs.LG])

    [http://arxiv.org/abs/2310.02925](http://arxiv.org/abs/2310.02925)

    OTARI是一种新的最优传输形式，它通过对每个点的质量施加约束来解决全局约束造成的不平衡问题，在领域适应中具有重要作用。

    

    使用严格凸约束来正则化最优传输（OT）的原始形式会增加数值复杂度并使传输计划更密集。许多公式对传输计划施加全局约束，例如依赖熵正则化。由于对于离群点而言扩散质量比中心点更昂贵，这通常导致质量在点之间的分布方式存在显著的不平衡。对于一些需要每个点最低平滑性的应用而言，这可能是有害的。为了解决这个问题，我们引入了自适应正则化最优传输（OTARI），一种对进出每个点的质量施加约束的新形式。然后我们展示了这种方法在领域适应中的好处。

    Regularising the primal formulation of optimal transport (OT) with a strictly convex term leads to enhanced numerical complexity and a denser transport plan. Many formulations impose a global constraint on the transport plan, for instance by relying on entropic regularisation. As it is more expensive to diffuse mass for outlier points compared to central ones, this typically results in a significant imbalance in the way mass is spread across the points. This can be detrimental for some applications where a minimum of smoothing is required per point. To remedy this, we introduce OT with Adaptive RegularIsation (OTARI), a new formulation of OT that imposes constraints on the mass going in or/and out of each point. We then showcase the benefits of this approach for domain adaptation.
    
[^29]: 用多项式朴素贝叶斯和K-modes聚类增强阿育吠陀诊断：对普拉克里蒂类型和Dosha重叠的研究

    Enhancing Ayurvedic Diagnosis using Multinomial Naive Bayes and K-modes Clustering: An Investigation into Prakriti Types and Dosha Overlapping. (arXiv:2310.02920v1 [cs.LG])

    [http://arxiv.org/abs/2310.02920](http://arxiv.org/abs/2310.02920)

    本研究提出使用多项式朴素贝叶斯和K-modes聚类来增强阿育吠陀诊断中的普拉克里蒂类型和Dosha重叠的识别。通过将Dosha分类为7个类别，包括重叠的Dosha类别，可以提高诊断模型的准确性和真实性。

    

    识别人体的普拉克里蒂类型是一种长时间失传的医学实践，旨在寻找人类本质与行为之间的和谐。个体有3种基本的普拉克里蒂类型，可以属于任何Dosha。在现有模型中，研究人员使用了支持向量机（SVM）、K最近邻（KNN）、主成分分析（PCA）、决策树和其他各种算法。这些算法的输出相当不错，但可以借助多项式朴素贝叶斯和K-modes聚类进行增强。大多数研究人员仅限于3个基本类别。然而，在现实世界的情况下，可能出现重叠。考虑到这一点，我们将Dosha分类为7个类别，其中包括Dosha的重叠。这些类别分别是VATT-Dosha、PITT-Dosha、KAPH-Dosha、VATT-PITT-Dosha、PITT-KAPH-Dosha、KAPH-VATT-Dosha和VATT-PITT-KAPH-Dosha。所使用的数据包含了所有个体条目的平衡集，对其进行了预处理。

    The identification of Prakriti types for the human body is a long-lost medical practice in finding the harmony between the nature of human beings and their behaviour. There are 3 fundamental Prakriti types of individuals. A person can belong to any Dosha. In the existing models, researchers have made use of SVM, KNN, PCA, Decision Tree, and various other algorithms. The output of these algorithms was quite decent, but it can be enhanced with the help of Multinomial Naive Bayes and K-modes clustering. Most of the researchers have confined themselves to 3 basic classes. This might not be accurate in the real-world scenario, where overlapping might occur. Considering these, we have classified the Doshas into 7 categories, which includes overlapping of Doshas. These are namely, VATT-Dosha, PITT-Dosha, KAPH-Dosha, VATT-PITT-Dosha, PITT-KAPH-Dosha, KAPH-VATT-Dosha, and VATT-PITT-KAPH-Dosha. The data used contains a balanced set of all individual entries on which preprocessing steps of machin
    
[^30]: 基于注意力的多任务学习用于碱基编辑结果预测

    Attention-based Multi-task Learning for Base Editor Outcome Prediction. (arXiv:2310.02919v1 [cs.LG])

    [http://arxiv.org/abs/2310.02919](http://arxiv.org/abs/2310.02919)

    基于注意力的多任务学习模型可以加速基因编辑设计过程，并提高编辑结果预测的准确性。

    

    人类遗传疾病通常由点突变引起，这凸显了对精确基因组编辑技术的关键需求。其中，碱基编辑以其能够在单个核苷酸水平上进行定向改变而脱颖而出。然而，其临床应用受到编辑效率低和非预期突变的限制，需要在实验室进行大量的试错实验。为了加速这个过程，我们提出了一个基于注意力的两阶段机器学习模型，该模型学习预测给定基因组目标序列的所有可能编辑结果的可能性。我们进一步提出了一个多任务学习模式，同时学习多种碱基编辑器（即变体）。我们模型的预测结果在多个数据集和碱基编辑器变体上始终表现出与实际实验结果的强相关性。这些结果进一步验证了模型改进和加速基因编辑设计过程能力。

    Human genetic diseases often arise from point mutations, emphasizing the critical need for precise genome editing techniques. Among these, base editing stands out as it allows targeted alterations at the single nucleotide level. However, its clinical application is hindered by low editing efficiency and unintended mutations, necessitating extensive trial-and-error experimentation in the laboratory. To speed up this process, we present an attention-based two-stage machine learning model that learns to predict the likelihood of all possible editing outcomes for a given genomic target sequence. We further propose a multi-task learning schema to jointly learn multiple base editors (i.e. variants) at once. Our model's predictions consistently demonstrated a strong correlation with the actual experimental results on multiple datasets and base editor variants. These results provide further validation for the models' capacity to enhance and accelerate the process of refining base editing desig
    
[^31]: ELUQuant: 深度非弹性散射中事件级不确定性量化

    ELUQuant: Event-Level Uncertainty Quantification in Deep Inelastic Scattering. (arXiv:2310.02913v1 [cs.LG])

    [http://arxiv.org/abs/2310.02913](http://arxiv.org/abs/2310.02913)

    ELUQuant是一种能够在深度非弹性散射中对事件级别的不确定性进行量化的方法，利用基于物理的贝叶斯神经网络和归一化流近似计算后验分布，能够提供详细的不确定性描述。这为决策制定和减少真实不准确性提供了宝贵的帮助。

    

    我们引入了一种基于物理信息的贝叶斯神经网络（BNN），通过使用乘法归一化流（MNF）来近似后验分布，以对物理事件级别进行详细的不确定性量化（UQ）。我们的方法能够识别异方差的唯有性和认知不确定性，提供了精细的物理洞察力。应用于深度非弹性散射（DIS）事件，我们的模型有效提取了动力学变量$x$，$Q^2$和$y$，与最新的深度学习回归技术在性能上相匹配，但具有事件级别UQ的关键增强。对基于HERA的H1探测器进行的DIS模拟表明了未来EIC的可能应用。此外，这为相关任务铺平了道路，如事件过滤等决策制定中的精细不确定性描述对于决策制定非常宝贵，特别是在不直接访问基本事实的情况下，还可以减少真实不准确性。

    We introduce a physics-informed Bayesian Neural Network (BNN) with flow approximated posteriors using multiplicative normalizing flows (MNF) for detailed uncertainty quantification (UQ) at the physics event-level. Our method is capable of identifying both heteroskedastic aleatoric and epistemic uncertainties, providing granular physical insights. Applied to Deep Inelastic Scattering (DIS) events, our model effectively extracts the kinematic variables $x$, $Q^2$, and $y$, matching the performance of recent deep learning regression techniques but with the critical enhancement of event-level UQ. This detailed description of the underlying uncertainty proves invaluable for decision-making, especially in tasks like event filtering. It also allows for the reduction of true inaccuracies without directly accessing the ground truth. A thorough DIS simulation using the H1 detector at HERA indicates possible applications for the future EIC. Additionally, this paves the way for related tasks such 
    
[^32]: 使用您的本能：使用神经探测器与转换器进行指令优化

    Use Your INSTINCT: INSTruction optimization usIng Neural bandits Coupled with Transformers. (arXiv:2310.02905v1 [cs.LG])

    [http://arxiv.org/abs/2310.02905](http://arxiv.org/abs/2310.02905)

    该论文提出了一种使用神经探测器和转换器优化指令的方法，以提高大型语言模型的性能。

    

    大型语言模型(LLMs)在各种应用中展示了出色的指令跟随能力，并取得了令人瞩目的表现。然而，LLMs的性能严重依赖于给予它们的指令，这些指令通常需要大量人力进行手动调整。最近的研究使用了高效的贝叶斯优化（BO）算法来自动优化给予黑盒LLMs的指令。然而，在优化高度复杂（例如高维）的目标函数时，如将指令映射到LLM性能的函数，BO通常表现不佳。这主要是由于BO使用的高斯过程（GP）模型的表达能力有限，该模型被用作BO的代理来建模目标函数。与此同时，已经多次证明神经网络（NNs），尤其是预训练的转换器，具有很强的表达能力，可以建模高度复杂的函数。因此，我们采用了一种神经探测器算法。

    Large language models (LLMs) have shown remarkable instruction-following capabilities and achieved impressive performances in various applications. However, the performances of LLMs depend heavily on the instructions given to them, which are typically manually tuned with substantial human efforts. Recent work has used the query-efficient Bayesian optimization (BO) algorithm to automatically optimize the instructions given to black-box LLMs. However, BO usually falls short when optimizing highly sophisticated (e.g., high-dimensional) objective functions, such as the functions mapping an instruction to the performance of an LLM. This is mainly due to the limited expressive power of the Gaussian process (GP) model which is used by BO as a surrogate to model the objective function. Meanwhile, it has been repeatedly shown that neural networks (NNs), especially pre-trained transformers, possess strong expressive power and can model highly complex functions. So, we adopt a neural bandit algor
    
[^33]: 基于样条函数的神经网络原子间势: 融合经典与机器学习模型

    Spline-based neural network interatomic potentials: blending classical and machine learning models. (arXiv:2310.02904v1 [cond-mat.mtrl-sci])

    [http://arxiv.org/abs/2310.02904](http://arxiv.org/abs/2310.02904)

    本研究引入了一种新的基于样条函数的神经网络势(s-NNP)框架，将简单性的s-MEAM原子间势与神经网络的灵活性相结合，用于构建高质量的IPs。该框架能够突破经典和ML IPs之间的界限，并通过关键架构变化提供更好的性能。同时，使用样条滤波器来编码原子环境，可以产生容易解释的嵌入层。

    

    虽然机器学习(Machine Learning, ML)原子间势(Interatomic Potentials, IPs)在训练时能够达到接近第一原理数据固有噪音水平的精确度，但还需要展示它们的增加复杂性是否严格必要来构建高质量的IPs。在这项工作中，我们引入了一种新的MLIP框架，它将基于样条函数的MEAM (s-MEAM)原子间势的简单性与神经网络(NN)架构的灵活性相结合。提出的框架被称为基于样条函数的神经网络势(s-NNP)，是传统NNP的简化版本，可以用于以高效的方式描述复杂数据集。我们演示了如何使用这个框架来探索经典和ML IPs之间的边界，并突出了关键架构变化的好处。此外，我们还展示了使用样条滤波器来编码原子环境会产生一个容易解释的嵌入层，可以与其他模型进行耦合。

    While machine learning (ML) interatomic potentials (IPs) are able to achieve accuracies nearing the level of noise inherent in the first-principles data to which they are trained, it remains to be shown if their increased complexities are strictly necessary for constructing high-quality IPs. In this work, we introduce a new MLIP framework which blends the simplicity of spline-based MEAM (s-MEAM) potentials with the flexibility of a neural network (NN) architecture. The proposed framework, which we call the spline-based neural network potential (s-NNP), is a simplified version of the traditional NNP that can be used to describe complex datasets in a computationally efficient manner. We demonstrate how this framework can be used to probe the boundary between classical and ML IPs, highlighting the benefits of key architectural changes. Furthermore, we show that using spline filters for encoding atomic environments results in a readily interpreted embedding layer which can be coupled with 
    
[^34]: FroSSL: 基于Frobenius范数最小化的自监督学习

    FroSSL: Frobenius Norm Minimization for Self-Supervised Learning. (arXiv:2310.02903v1 [cs.LG])

    [http://arxiv.org/abs/2310.02903](http://arxiv.org/abs/2310.02903)

    FroSSL是一种基于Frobenius范数最小化的自监督学习方法，通过最小化协方差Frobenius范数来避免信息崩溃，同时通过最小化均方差来实现数据增强的不变性，相比其他SSL方法，FroSSL收敛更快，并且这种快速收敛是由于FroSSL影响嵌入协方差矩阵的特征值所致。

    

    自监督学习（SSL）是一种越来越受欢迎的表示学习范式。最近的方法可分类为样本对比、维度对比或非对称网络的方法，每个家族都有自己的方法来避免信息崩溃。虽然维度对比方法收敛到与样本对比方法相似的解，但可以经验性地证明一些方法需要更多的训练迭代才能收敛。为了弥合这一差距，我们提出了目标函数FroSSL，它在嵌入归一化方面既是样本对比又是维度对比。FroSSL通过最小化协方差Frobenius范数来避免崩溃，并通过最小化均方差来实现数据增强的不变性。我们展示了FroSSL比其他各种SSL方法更快地收敛，并提供了理论和实证支持，证明了这种更快的收敛是由于FroSSL对嵌入协方差矩阵的特征值产生的影响。

    Self-supervised learning (SSL) is an increasingly popular paradigm for representation learning. Recent methods can be classified as sample-contrastive, dimension-contrastive, or asymmetric network-based, with each family having its own approach to avoiding informational collapse. While dimension-contrastive methods converge to similar solutions as sample-contrastive methods, it can be empirically shown that some methods require more epochs of training to converge. Motivated by closing this divide, we present the objective function FroSSL which is both sample- and dimension-contrastive up to embedding normalization. FroSSL works by minimizing covariance Frobenius norms for avoiding collapse and minimizing mean-squared error for augmentation invariance. We show that FroSSL converges more quickly than a variety of other SSL methods and provide theoretical and empirical support that this faster convergence is due to how FroSSL affects the eigenvalues of the embedding covariance matrices. W
    
[^35]: 使用强化学习和Transformer搜索高价值分子

    Searching for High-Value Molecules Using Reinforcement Learning and Transformers. (arXiv:2310.02902v1 [cs.LG])

    [http://arxiv.org/abs/2310.02902](http://arxiv.org/abs/2310.02902)

    通过使用强化学习和Transformer，我们提出了一种新的基于RL的分子设计算法（ChemRLformer），并在25个分子设计任务中进行了综合分析，包括计算复杂的蛋白质对接模拟。我们发现了分子设计领域的独特见解，并展示了ChemRLformer相对于之前的工作更为简单且实现了最先进的性能。

    

    在搜索图中的高价值策略方面，使用文本表示的强化学习（RL）可以很有效。然而，RL需要对搜索空间进行精心结构化和算法设计才能在这个挑战中发挥作用。通过大量实验，我们探索了不同文本语法设计和训练算法选择如何影响RL策略生成具有所需属性的分子的能力。我们提出了一种新的基于RL的分子设计算法（ChemRLformer），并对其进行了深入分析，包括对计算复杂的蛋白质对接模拟进行的25个分子设计任务。通过这个分析，我们发现了该问题空间中的独特见解，并展示了ChemRLformer相较于之前的工作，通过阐明哪些设计选择实际上对基于文本的分子设计有帮助，实现了最先进的性能。

    Reinforcement learning (RL) over text representations can be effective for finding high-value policies that can search over graphs. However, RL requires careful structuring of the search space and algorithm design to be effective in this challenge. Through extensive experiments, we explore how different design choices for text grammar and algorithmic choices for training can affect an RL policy's ability to generate molecules with desired properties. We arrive at a new RL-based molecular design algorithm (ChemRLformer) and perform a thorough analysis using 25 molecule design tasks, including computationally complex protein docking simulations. From this analysis, we discover unique insights in this problem space and show that ChemRLformer achieves state-of-the-art performance while being more straightforward than prior work by demystifying which design choices are actually helpful for text-based molecule design.
    
[^36]: 从过参数自编码器中恢复训练数据：逆问题的观点

    Recovery of Training Data from Overparameterized Autoencoders: An Inverse Problem Perspective. (arXiv:2310.02897v1 [cs.LG])

    [http://arxiv.org/abs/2310.02897](http://arxiv.org/abs/2310.02897)

    本研究从逆问题的角度研究了从过参数自编码器模型恢复训练数据的问题，并提出了一种实际方法，该方法利用训练好的自编码器来定义正则化器并通过迭代计算处理未知的退化操作符。实验结果表明，该方法在自编码器恢复训练数据方面具有显著的优势。

    

    我们研究了从过参数自编码器模型中恢复训练数据的问题。给定一个退化的训练样本，我们将原始样本的恢复定义为一个逆问题，并将其构建为一个优化任务。在我们的逆问题中，我们使用训练好的自编码器来隐式地定义一个正则化器，用于从特定的训练数据集中检索。我们将复杂的优化任务开发成一个实际方法，该方法迭代地应用训练好的自编码器和相对简单的计算来估计和处理未知的退化操作符。我们将该方法应用于盲目修补，目标是从许多缺失的像素中恢复训练图像，而这些缺失的像素是按照未知的模式进行的。我们检验了各种深度自编码器架构，如全连接和U-Net（具有不同的非线性和多样的训练损失值），并且证明了我们的方法明显优于以前的自编码器恢复训练数据的方法。

    We study the recovery of training data from overparameterized autoencoder models. Given a degraded training sample, we define the recovery of the original sample as an inverse problem and formulate it as an optimization task. In our inverse problem, we use the trained autoencoder to implicitly define a regularizer for the particular training dataset that we aim to retrieve from. We develop the intricate optimization task into a practical method that iteratively applies the trained autoencoder and relatively simple computations that estimate and address the unknown degradation operator. We evaluate our method for blind inpainting where the goal is to recover training images from degradation of many missing pixels in an unknown pattern. We examine various deep autoencoder architectures, such as fully connected and U-Net (with various nonlinearities and at diverse train loss values), and show that our method significantly outperforms previous methods for training data recovery from autoen
    
[^37]: CoLiDE: 共同线性有向无环图估计

    CoLiDE: Concomitant Linear DAG Estimation. (arXiv:2310.02895v1 [cs.LG])

    [http://arxiv.org/abs/2310.02895](http://arxiv.org/abs/2310.02895)

    本论文提出了CoLiDE算法用于学习线性DAG，该算法使用了一个新的凸评分函数，结合了标度的共同估计，从而有效地将稀疏参数与外生噪声水平分离。

    

    我们处理从遵循线性结构方程模型 (SEM) 的观测数据中学习有向无环图 (DAG) 结构的组合问题。利用不可微分、非凸的有效性特征，最近的研究提出了一种连续受限优化范式，以便高效地探索DAG空间。大多数现有方法使用套索类型的评分函数来引导这个搜索过程，这些函数在$\textit{未知}$SEM噪声方差在问题实例之间发生变化时需进行昂贵的惩罚参数重新调整，并且隐含地依赖于有界同方差假设。在这项工作中，我们提出了一个新的凸评分函数，用于稀疏感知线性DAG的学习，该函数结合了标度的共同估计，从而有效地将稀疏参数与外生噪声水平分离。通过平滑的、非凸的无环惩罚项进行正则化，可以得到CoLiDE （共同线性DAG估计）。

    We deal with the combinatorial problem of learning directed acyclic graph (DAG) structure from observational data adhering to a linear structural equation model (SEM). Leveraging advances in differentiable, nonconvex characterizations of acyclicity, recent efforts have advocated a continuous constrained optimization paradigm to efficiently explore the space of DAGs. Most existing methods employ lasso-type score functions to guide this search, which (i) require expensive penalty parameter retuning when the $\textit{unknown}$ SEM noise variances change across problem instances; and (ii) implicitly rely on limiting homoscedasticity assumptions. In this work, we propose a new convex score function for sparsity-aware learning of linear DAGs, which incorporates concomitant estimation of scale and thus effectively decouples the sparsity parameter from the exogenous noise levels. Regularization via a smooth, nonconvex acyclicity penalty term yields CoLiDE ($\textbf{Co}$ncomitant $\textbf{Li}$n
    
[^38]: 无中生有：利用无标签数据改进深度合奏模型校准

    Something for (almost) nothing: Improving deep ensemble calibration using unlabeled data. (arXiv:2310.02885v1 [cs.LG])

    [http://arxiv.org/abs/2310.02885](http://arxiv.org/abs/2310.02885)

    本文提出了一种简单的方法，在小训练数据情况下利用无标签数据改进深度合奏模型校准，实验证明该方法在测试集上具有较低的负对数似然和高的合奏多样性，比标准方法更好。

    

    我们提出了一种在小训练数据情况下利用无标签数据改进深度合奏模型校准的方法。我们的方法非常简单：对于每个无标签数据点，在每个合奏成员中，我们随机选择一个不同的标签进行拟合。我们基于PAC-Bayes边界提供了一个理论分析，保证如果我们在无标签数据上进行这样的标签拟合，并在训练数据上使用真实标签，在测试样本上可以得到较低的负对数似然和较高的合奏多样性。通过详细的实验证明，对于训练集规模较小到中等规模的情况，我们的合奏模型比标准的合奏模型更具多样性，并提供更好的校准效果，有时显著地。

    We present a method to improve the calibration of deep ensembles in the small training data regime in the presence of unlabeled data. Our approach is extremely simple to implement: given an unlabeled set, for each unlabeled data point, we simply fit a different randomly selected label with each ensemble member. We provide a theoretical analysis based on a PAC-Bayes bound which guarantees that if we fit such a labeling on unlabeled data, and the true labels on the training data, we obtain low negative log-likelihood and high ensemble diversity on testing samples. Empirically, through detailed experiments, we find that for low to moderately-sized training sets, our ensembles are more diverse and provide better calibration than standard ensembles, sometimes significantly.
    
[^39]: 无均值回归：不恰当高斯过程回归和不恰当核

    Stationarity without mean reversion: Improper Gaussian process regression and improper kernels. (arXiv:2310.02877v1 [stat.ML])

    [http://arxiv.org/abs/2310.02877](http://arxiv.org/abs/2310.02877)

    本论文展示了使用具有无限方差的不恰当高斯过程先验来定义静止但不均值回归过程的可能性，并引入了一类特殊的不恰当核函数来实现此目的。

    

    高斯过程（GP）回归在机器学习应用中已经广泛流行。GP回归的行为取决于协方差函数的选择。在机器学习应用中，静止协方差函数是首选。然而，（非周期性的）静止协方差函数总是均值回归的，因此在应用于不通过到固定全局均值值的数据时可能表现出病态行为。在本文中，我们展示了使用具有无限方差的不恰当GP先验来定义静止但不均值回归过程是可能的。为此，我们引入了一大类只能在这种不恰当的范围内定义的不恰当核函数。具体地，我们引入了平滑行走核，它产生无限平滑的样本，以及一类不恰当的Matern核，它可以被定义为任意整数j倍可微。所得到的后验分布可以用解析的方式计算出来。

    Gaussian processes (GP) regression has gained substantial popularity in machine learning applications. The behavior of a GP regression depends on the choice of covariance function. Stationary covariance functions are favorite in machine learning applications. However, (non-periodic) stationary covariance functions are always mean reverting and can therefore exhibit pathological behavior when applied to data that does not relax to a fixed global mean value. In this paper, we show that it is possible to use improper GP prior with infinite variance to define processes that are stationary but not mean reverting. To this aim, we introduce a large class of improper kernels that can only be defined in this improper regime. Specifically, we introduce the Smooth Walk kernel, which produces infinitely smooth samples, and a family of improper Mat\'ern kernels, which can be defined to be $j$-times differentiable for any integer $j$. The resulting posterior distributions can be computed analyticall
    
[^40]: 医疗保健领域联邦学习的最新方法学进展

    Recent Methodological Advances in Federated Learning for Healthcare. (arXiv:2310.02874v1 [cs.LG])

    [http://arxiv.org/abs/2310.02874](http://arxiv.org/abs/2310.02874)

    最近的医疗保健领域联邦学习研究提出了新的方法学，用于解决医疗保健数据的挑战，如孤立数据、类别不平衡、缺失数据、分布转移和非标准化变量。

    

    对于医疗保健数据集来说，由于伦理、隐私或后勤问题，通常不可能合并来自多个机构的数据样本。而联邦学习可以在不需要数据汇集的情况下利用强大的机器学习算法。医疗保健数据具有许多挑战，需要新的方法学来解决，如高度孤立的数据、类别不平衡、缺失数据、分布转移和非标准化变量。联邦学习对于传统的集中式机器学习增加了显著的方法学复杂性，需要分布式优化、节点间的通信、模型的聚合和模型的重新分发。在这个系统性综述中，我们考虑了Scopus上在2015年1月至2023年2月之间发表的所有描述解决医疗保健数据挑战的新联邦学习方法学的论文。我们对满足这些条件的89篇论文进行了详细的回顾。

    For healthcare datasets, it is often not possible to combine data samples from multiple sites due to ethical, privacy or logistical concerns. Federated learning allows for the utilisation of powerful machine learning algorithms without requiring the pooling of data. Healthcare data has many simultaneous challenges which require new methodologies to address, such as highly-siloed data, class imbalance, missing data, distribution shifts and non-standardised variables. Federated learning adds significant methodological complexity to conventional centralised machine learning, requiring distributed optimisation, communication between nodes, aggregation of models and redistribution of models. In this systematic review, we consider all papers on Scopus that were published between January 2015 and February 2023 and which describe new federated learning methodologies for addressing challenges with healthcare data. We performed a detailed review of the 89 papers which fulfilled these criteria. S
    
[^41]: 针对表格数据的稳定且可解释的深度学习: 引入具有新型InterpreStability指标的InterpreTabNet

    Stable and Interpretable Deep Learning for Tabular Data: Introducing InterpreTabNet with the Novel InterpreStability Metric. (arXiv:2310.02870v1 [cs.LG])

    [http://arxiv.org/abs/2310.02870](http://arxiv.org/abs/2310.02870)

    我们引入了InterpreTabNet，通过改进的注意模块和TabNet架构，提高了表格数据的分类准确度和解释性。我们还提出了一种新的评价指标InterpreStability，用于量化模型的解释稳定性。

    

    随着人工智能（AI）在各个领域的深度整合，对强大模型的追求已经加剧。虽然在提升模型能力和适用性方面已经取得了重大进展，但一个明显的挑战仍然存在：许多最先进的模型仍然是黑箱。这种不透明性不仅使解释模型决策给最终用户带来了复杂性，而且还阻碍了模型设计者对中间过程的洞察。为了解决这些挑战，我们引入了InterpreTabNet，这是一个通过利用改进的注意模块，改进TabNet架构，从而提高分类准确度和解释性的模型。这种设计确保了强大的梯度传播和计算稳定性。此外，我们还提出了一种新型评价指标InterpreStability，该指标量化了模型的解释稳定性。所提出的模型和指标标志着可解释模型研究的重要进展。

    As Artificial Intelligence (AI) integrates deeper into diverse sectors, the quest for powerful models has intensified. While significant strides have been made in boosting model capabilities and their applicability across domains, a glaring challenge persists: many of these state-of-the-art models remain as black boxes. This opacity not only complicates the explanation of model decisions to end-users but also obstructs insights into intermediate processes for model designers. To address these challenges, we introduce InterpreTabNet, a model designed to enhance both classification accuracy and interpretability by leveraging the TabNet architecture with an improved attentive module. This design ensures robust gradient propagation and computational stability. Additionally, we present a novel evaluation metric, InterpreStability, which quantifies the stability of a model's interpretability. The proposed model and metric mark a significant stride forward in explainable models' research, set
    
[^42]: 使用谐波控制李亚普诺夫障碍函数解决有约束的最优控制问题与避障要求

    Harmonic Control Lyapunov Barrier Functions for Constrained Optimal Control with Reach-Avoid Specifications. (arXiv:2310.02869v1 [math.OC])

    [http://arxiv.org/abs/2310.02869](http://arxiv.org/abs/2310.02869)

    该论文介绍了谐波控制李亚普诺夫障碍函数（harmonic CLBF），它可以在约束控制问题中解决避障要求，通过最大化系统动力学与谐波CLBF最陡下降方向的内积来选择控制输入，从而显著降低进入不安全区域的风险并提高进入目标区域的概率。

    

    本文介绍了谐波控制李亚普诺夫障碍函数（harmonic CLBF），它有助于解决诸如避障问题等的约束控制问题。谐波CLBF利用谐波函数满足的最大值原理来编码控制李亚普诺夫障碍函数的属性。因此，它们可以在实验开始时初始化，而不是基于样本轨迹进行训练。控制输入被选择为最大化系统动力学与谐波CLBF最陡下降方向的内积。数值结果在不同的避障环境下展示了四个不同系统的情况。谐波CLBF显示出进入不安全区域的风险显著降低，进入目标区域的概率高。

    This paper introduces harmonic control Lyapunov barrier functions (harmonic CLBF) that aid in constrained control problems such as reach-avoid problems. Harmonic CLBFs exploit the maximum principle that harmonic functions satisfy to encode the properties of control Lyapunov barrier functions (CLBFs). As a result, they can be initiated at the start of an experiment rather than trained based on sample trajectories. The control inputs are selected to maximize the inner product of the system dynamics with the steepest descent direction of the harmonic CLBF. Numerical results are presented with four different systems under different reach-avoid environments. Harmonic CLBFs show a significantly low risk of entering unsafe regions and a high probability of entering the goal region.
    
[^43]: 利用共享结构进行有限数据模型估计

    Estimation of Models with Limited Data by Leveraging Shared Structure. (arXiv:2310.02864v1 [cs.LG])

    [http://arxiv.org/abs/2310.02864](http://arxiv.org/abs/2310.02864)

    本文提出了一种利用共享结构进行有限数据模型估计的方法，通过估计潜在的低维参数空间，并在该空间内产生精确的参数估计。这种方法适用于具有多个系统且每个系统的数据量很少的情况下，提供有限样本子空间估计误差保证。

    

    现代数据集（如医疗保健和电子商务）通常来自许多个体或系统，但每个单独来源的数据都不足以分别估计个体的高维模型参数。然而，如果系统之间存在共享结构，可能可以利用其他系统的数据来帮助估计个体参数，否则这些参数可能是不可识别的。本文假设系统共享一个潜在的低维参数空间，并提出一种方法来恢复N个不同线性系统的d维参数，即使每个系统只有T<d个观测值。为此，我们开发了一个三步算法，估计由系统参数构成的低维子空间，并在子空间内产生精确的参数估计。我们为所提出的方法提供了有限样本子空间估计误差保证。最后，我们在模拟实验中验证了我们的方法。

    Modern data sets, such as those in healthcare and e-commerce, are often derived from many individuals or systems but have insufficient data from each source alone to separately estimate individual, often high-dimensional, model parameters. If there is shared structure among systems however, it may be possible to leverage data from other systems to help estimate individual parameters, which could otherwise be non-identifiable. In this paper, we assume systems share a latent low-dimensional parameter space and propose a method for recovering $d$-dimensional parameters for $N$ different linear systems, even when there are only $T<d$ observations per system. To do so, we develop a three-step algorithm which estimates the low-dimensional subspace spanned by the systems' parameters and produces refined parameter estimates within the subspace. We provide finite sample subspace estimation error guarantees for our proposed method. Finally, we experimentally validate our method on simulations wi
    
[^44]: 长期数据的共形预测

    Conformal Predictions for Longitudinal Data. (arXiv:2310.02863v1 [stat.ML])

    [http://arxiv.org/abs/2310.02863](http://arxiv.org/abs/2310.02863)

    这篇论文介绍了一种新颖的基于分布的共形预测算法LPCI，用于处理长期数据。通过将剩余数据建模为分位数固定效应回归问题，并使用训练好的回归器构建预测区间，LPCI实现了有效的横截面覆盖，并优于现有的基准模型。

    

    我们介绍了一种新颖的基于分布的共形预测算法，称为长期预测共形推断（LPCI），用于处理长期数据。目前针对时间序列数据的共形预测方法主要集中在单变量设置上，因此在应用于长期数据集中的每个时间序列时缺乏横截面覆盖。目前长期数据的最新方法依赖于创建无限宽的预测区间，以保证横截面和渐近长期覆盖率。所提出的LPCI方法通过确保同时保证纵向和横截面覆盖而无需使用无限宽的区间来解决这个问题。在我们的方法中，我们将剩余数据建模为一个分位数固定效应回归问题，并使用训练好的分位数回归器构建预测区间。我们广泛的实验表明，LPCI实现了有效的横截面覆盖，并优于现有的基准模型。

    We introduce Longitudinal Predictive Conformal Inference (LPCI), a novel distribution-free conformal prediction algorithm for longitudinal data. Current conformal prediction approaches for time series data predominantly focus on the univariate setting, and thus lack cross-sectional coverage when applied individually to each time series in a longitudinal dataset. The current state-of-the-art for longitudinal data relies on creating infinitely-wide prediction intervals to guarantee both cross-sectional and asymptotic longitudinal coverage. The proposed LPCI method addresses this by ensuring that both longitudinal and cross-sectional coverages are guaranteed without resorting to infinitely wide intervals. In our approach, we model the residual data as a quantile fixed-effects regression problem, constructing prediction intervals with a trained quantile regressor. Our extensive experiments demonstrate that LPCI achieves valid cross-sectional coverage and outperforms existing benchmarks in 
    
[^45]: 使用稀疏离散余弦Stockwell变换层的新型非对称自编码器用于齿轮传感器数据压缩

    A novel asymmetrical autoencoder with a sparsifying discrete cosine Stockwell transform layer for gearbox sensor data compression. (arXiv:2310.02862v1 [cs.LG])

    [http://arxiv.org/abs/2310.02862](http://arxiv.org/abs/2310.02862)

    这篇论文提出了一种信号自适应的非对称自编码器，使用离散余弦Stockwell变换层进行齿轮传感器数据压缩。通过引入可训练的滤波器和硬阈值化层，该方法能够提高数据重构的准确性，并且仅需要少量数据集进行训练。

    

    在非接触式齿轮故障诊断问题中，缺乏高效的数据压缩模型仍然是无线传输齿轮数据的一个挑战。本文提出了一种信号自适应的非对称自编码器，其中增加了一个变换域层来压缩传感器信号。首先，引入了一种新的离散余弦Stockwell变换（DCST）层以替代多层自编码器中的线性层。通过利用卷积的乘法特性在DCST域实现了一个可训练的滤波器。然后，应用可训练的硬阈值化层来减少DCST层中的冗余数据以使特征图稀疏化。与线性层相比，DCST层减少了可训练参数的数量，并提高了数据重构的准确性。其次，使用稀疏化的DCST层训练自编码器只需要少量的数据集。提出的方法在康涅狄格大学（Uo...[被截断]

    The lack of an efficient compression model remains a challenge for the wireless transmission of gearbox data in non-contact gear fault diagnosis problems. In this paper, we present a signal-adaptive asymmetrical autoencoder with a transform domain layer to compress sensor signals. First, a new discrete cosine Stockwell transform (DCST) layer is introduced to replace linear layers in a multi-layer autoencoder. A trainable filter is implemented in the DCST domain by utilizing the multiplication property of the convolution. A trainable hard-thresholding layer is applied to reduce redundant data in the DCST layer to make the feature map sparse. In comparison to the linear layer, the DCST layer reduces the number of trainable parameters and improves the accuracy of data reconstruction. Second, training the autoencoder with a sparsifying DCST layer only requires a small number of datasets. The proposed method is superior to other autoencoder-based methods on the University of Connecticut (Uo
    
[^46]: Rayleigh Quotient Graph Neural Networks用于图级异常检测的研究

    Rayleigh Quotient Graph Neural Networks for Graph-level Anomaly Detection. (arXiv:2310.02861v1 [cs.LG])

    [http://arxiv.org/abs/2310.02861](http://arxiv.org/abs/2310.02861)

    《Rayleigh Quotient Graph Neural Networks用于图级异常检测的研究》提出使用Rayleigh Quotient作为驱动因素，通过探索图的固有光谱特征来实现图级异常检测。

    

    图级异常检测在癌症诊断和酶预测等领域中广泛应用。然而，现有方法无法捕捉到图异常的潜在属性，导致框架设计不可解释和性能不令人满意。在本文中，我们退一步重新研究了异常和正常图之间的光谱差异。我们的主要观察表明，这两个类之间的累计光谱能量存在显著差异。此外，我们证明了图信号的累计光谱能量可以用其瑞利商表示，这表明瑞利商是图异常属性的一个驱动因素。受此启发，我们提出了Rayleigh Quotient Graph Neural Network（RQGNN），这是第一个用于图级异常检测的光谱GNN，为探索异常图的固有光谱特征提供了新的视角。

    Graph-level anomaly detection has gained significant attention as it finds many applications in various domains, such as cancer diagnosis and enzyme prediction. However, existing methods fail to capture the underlying properties of graph anomalies, resulting in unexplainable framework design and unsatisfying performance. In this paper, we take a step back and re-investigate the spectral differences between anomalous and normal graphs. Our main observation shows a significant disparity in the accumulated spectral energy between these two classes. Moreover, we prove that the accumulated spectral energy of the graph signal can be represented by its Rayleigh Quotient, indicating that the Rayleigh Quotient is a driving factor behind the anomalous properties of graphs. Motivated by this, we propose Rayleigh Quotient Graph Neural Network (RQGNN), the first spectral GNN for graph-level anomaly detection, providing a new perspective on exploring the inherent spectral features of anomalous graph
    
[^47]: 通过弱分布不变性实现多领域因果表示学习

    Multi-Domain Causal Representation Learning via Weak Distributional Invariances. (arXiv:2310.02854v1 [cs.LG])

    [http://arxiv.org/abs/2310.02854](http://arxiv.org/abs/2310.02854)

    本文提出了一种通过弱分布不变性进行多领域因果表示学习的方法，证明了融入这种不变性的自编码器能够可靠地识别出稳定的变量集合。

    

    因果表示学习已成为因果机器学习研究的核心。特别是，多领域数据集为展示因果表示学习相对于标准无监督表示学习的优势提供了自然机会。虽然最近的研究在学习因果表示方面取得了重要进展，但由于过于简化数据的假设，它们往往不能适用于多领域数据集；例如，每个领域都来自不同的单节点完美干预。在本文中，我们放宽了这些假设，并利用以下观察结果：在多领域数据中，往往存在一部分潜变量的某些分布属性（例如支持度、方差）在不同领域之间保持稳定；当每个领域来自多节点不完美干预时，这个属性成立。利用这个观察结果，我们证明了融入这种不变性的自编码器能够可靠地识别出稳定的变量集合。

    Causal representation learning has emerged as the center of action in causal machine learning research. In particular, multi-domain datasets present a natural opportunity for showcasing the advantages of causal representation learning over standard unsupervised representation learning. While recent works have taken crucial steps towards learning causal representations, they often lack applicability to multi-domain datasets due to over-simplifying assumptions about the data; e.g. each domain comes from a different single-node perfect intervention. In this work, we relax these assumptions and capitalize on the following observation: there often exists a subset of latents whose certain distributional properties (e.g., support, variance) remain stable across domains; this property holds when, for example, each domain comes from a multi-node imperfect intervention. Leveraging this observation, we show that autoencoders that incorporate such invariances can provably identify the stable set o
    
[^48]: 通过利用层间变换的平滑性进行带外分布检测

    Out-of-Distribution Detection by Leveraging Between-Layer Transformation Smoothness. (arXiv:2310.02832v1 [cs.LG])

    [http://arxiv.org/abs/2310.02832](http://arxiv.org/abs/2310.02832)

    本文提出了一种通过利用神经网络中间层变换的平滑性来检测带外数据的方法(BLOOD),该方法适用于没有训练数据访问权限的预训练模型，并在Transformer网络上的文本分类任务中取得了良好的效果。

    

    有效的带外分布检测对于可靠的机器学习模型至关重要，然而大多数当前方法由于需要访问训练数据或者干预训练而在实际应用中受到限制。我们提出了一种新的方法，通过网络中间层的变换平滑性来检测深度神经网络中的带外数据（BLOOD），该方法适用于没有训练数据访问权限的预训练模型。BLOOD利用内分布（ID）数据的层间表示变换相较于带外数据的变换更平滑的倾向，这也是我们在Transformer网络中经验证明的一个特性。我们在几个文本分类任务上评估了BLOOD与Transformer网络，并证明其在资源需求相当的方法上性能更好。我们的分析还表明，当学习更简单的任务时，带外数据的变换会保持其原始的锐度，而锐度会随着任务的增加而增加。

    Effective OOD detection is crucial for reliable machine learning models, yet most current methods are limited in practical use due to requirements like access to training data or intervention in training. We present a novel method for detecting OOD data in deep neural networks based on transformation smoothness between intermediate layers of a network (BLOOD), which is applicable to pre-trained models without access to training data. BLOOD utilizes the tendency of between-layer representation transformations of in-distribution (ID) data to be smoother than the corresponding transformations of OOD data, a property that we also demonstrate empirically for Transformer networks. We evaluate BLOOD on several text classification tasks with Transformer networks and demonstrate that it outperforms methods with comparable resource requirements. Our analysis also suggests that when learning simpler tasks, OOD data transformations maintain their original sharpness, whereas sharpness increases wit
    
[^49]: 学习温度条件下尺度标量化的GFlowNets

    Learning to Scale Logits for Temperature-Conditional GFlowNets. (arXiv:2310.02823v1 [cs.LG])

    [http://arxiv.org/abs/2310.02823](http://arxiv.org/abs/2310.02823)

    这项研究提出了一种名为LSL-GFN的新型架构设计，可以大大加速温度条件下GFlowNets的训练，从而提高GFlowNets的探索和利用能力。

    

    GFlowNets是一种概率模型，通过学习随机策略来顺序生成组合结构，例如分子图。它们的训练目标是按比例采样具有相应温度调节的对象的奖励。在GFlowNets中，温度条件下的GFlowNets代表了一系列由温度索引的策略，每个策略与相应的温度调节奖励函数相关联。温度条件下的GFlowNets的主要优势在于通过调整温度来控制对GFlowNets的探索和利用。我们提出了一种名为学习温度条件下尺度标量化的GFlowNets（LSL-GFN）的新型架构设计，它极大地加速了温度条件下GFlowNets的训练。它基于一个思想，即之前提出的温度条件方法在深度网络的训练中引入了数值挑战，因为不同的温度可能导致非常不同的情况。

    GFlowNets are probabilistic models that learn a stochastic policy that sequentially generates compositional structures, such as molecular graphs. They are trained with the objective of sampling such objects with probability proportional to the object's reward. Among GFlowNets, the temperature-conditional GFlowNets represent a family of policies indexed by temperature, and each is associated with the correspondingly tempered reward function. The major benefit of temperature-conditional GFlowNets is the controllability of GFlowNets' exploration and exploitation through adjusting temperature. We propose Learning to Scale Logits for temperature-conditional GFlowNets (LSL-GFN), a novel architectural design that greatly accelerates the training of temperature-conditional GFlowNets. It is based on the idea that previously proposed temperature-conditioning approaches introduced numerical challenges in the training of the deep network because different temperatures may give rise to very differe
    
[^50]: 智能制造系统中的时间序列分类: 对最先进机器学习算法的实验评估

    Time-Series Classification in Smart Manufacturing Systems: An Experimental Evaluation of State-of-the-Art Machine Learning Algorithms. (arXiv:2310.02812v1 [cs.LG])

    [http://arxiv.org/abs/2310.02812](http://arxiv.org/abs/2310.02812)

    本研究通过严格实验评估了智能制造系统中最先进的机器学习和深度学习算法在时间序列分类任务中的性能，填补了该领域的研究空白。

    

    随着传感器数量的增加和感知技术的快速发展，制造业正在收集大量各种各样的数据。在智能制造系统 (SMS) 环境中，时间序列数据起着关键的作用。因此，时间序列分类 (TSC) 在该领域中至关重要。本研究的目标是通过对制造业和工业环境中 TSC 任务的最先进机器学习和深度学习算法进行严格的实验评估来填补这一空白。我们首先在 TSC 和制造业文献中探索和编制了一份包含超过92个最先进算法的全面列表。随后，我们从该列表中选择了最具代表性的36个算法。为了评估这些算法在各种制造业分类任务中的性能，我们策划了一组包含22个制造业数据集的基准数据集，这些数据集具有不同的特征，涵盖了各种制造问题。随后，我们在制造业基准数据集上实施并评估了这些算法。

    Manufacturing is gathering extensive amounts of diverse data, thanks to the growing number of sensors and rapid advances in sensing technologies. Among the various data types available in SMS settings, time-series data plays a pivotal role. Hence, TSC emerges is crucial in this domain. The objective of this study is to fill this gap by providing a rigorous experimental evaluation of the SoTA ML and DL algorithms for TSC tasks in manufacturing and industrial settings. We first explored and compiled a comprehensive list of more than 92 SoTA algorithms from both TSC and manufacturing literature. Following, we selected the 36 most representative algorithms from this list. To evaluate their performance across various manufacturing classification tasks, we curated a set of 22 manufacturing datasets, representative of different characteristics that cover diverse manufacturing problems. Subsequently, we implemented and evaluated the algorithms on the manufacturing benchmark datasets, and analy
    
[^51]: 有限数据条件下用于MILP求解器的深度实例生成框架

    A Deep Instance Generative Framework for MILP Solvers Under Limited Data Availability. (arXiv:2310.02807v1 [cs.LG])

    [http://arxiv.org/abs/2310.02807](http://arxiv.org/abs/2310.02807)

    本文提出了G2MILP，这是第一个用于MILP实例的深度生成框架，它可以生成新颖而逼真的MILP实例。

    

    在过去的几年中，使用机器学习技术解决组合优化问题，特别是混合整数线性规划问题（MILP），出现了爆炸式增长。尽管取得了一些成果，但真实世界实例的有限可用性往往会导致次优决策和有偏见的求解器评估，这就需要一系列合成MILP实例生成技术。然而，现有方法要么过于依赖专家设计的表达式，要么难以捕捉真实世界实例的丰富特征。为了解决这个问题，我们提出了G2MILP，据我们所知这是第一个用于MILP实例的深度生成框架。具体来说，G2MILP将MILP实例表示为二分图，并应用遮蔽变分自编码器来迭代地破坏和替换原始图的部分以生成新的实例。G2MILP的一个吸引人的特点是它可以学会生成新颖而逼真的MILP实例。

    In the past few years, there has been an explosive surge in the use of machine learning (ML) techniques to address combinatorial optimization (CO) problems, especially mixed-integer linear programs (MILPs). Despite the achievements, the limited availability of real-world instances often leads to sub-optimal decisions and biased solver assessments, which motivates a suite of synthetic MILP instance generation techniques. However, existing methods either rely heavily on expert-designed formulations or struggle to capture the rich features of real-world instances. To tackle this problem, we propose G2MILP, which to the best of our knowledge is the first deep generative framework for MILP instances. Specifically, G2MILP represents MILP instances as bipartite graphs, and applies a masked variational autoencoder to iteratively corrupt and replace parts of the original graphs to generate new ones. The appealing feature of G2MILP is that it can learn to generate novel and realistic MILP instan
    
[^52]: 一种基于数据的Richards方程数值解法，用于模拟土壤中的水流动力学

    A Data-facilitated Numerical Method for Richards Equation to Model Water Flow Dynamics in Soil. (arXiv:2310.02806v1 [math.NA])

    [http://arxiv.org/abs/2310.02806](http://arxiv.org/abs/2310.02806)

    本文提出了一种基于数据的Richards方程数值解法，称为D-GRW方法，通过整合自适应线性化方案、神经网络和全局随机游走，在有限体积离散化框架下，能够产生具有收敛性保证的Richards方程数值解，并在精度和质量守恒性能方面表现卓越。

    

    根区土壤湿度的监测对于精密农业、智能灌溉和干旱预防至关重要。通常通过求解Richards方程这样的水文模型来模拟土壤的时空水流动力学。在本文中，我们提出了一种新型的基于数据的Richards方程数值解法。这种数值解法被称为D-GRW（Data-facilitated global Random Walk）方法，它在有限体积离散化框架中协同地整合了自适应线性化方案、神经网络和全局随机游走，可以在合理的假设下产生精确的Richards方程数值解，并且具有收敛性保证。通过三个示例，我们展示和讨论了我们的D-GRW方法在精度和质量守恒性能方面的卓越表现，并将其与基准数值解法和商用软件进行了比较。

    Root-zone soil moisture monitoring is essential for precision agriculture, smart irrigation, and drought prevention. Modeling the spatiotemporal water flow dynamics in soil is typically achieved by solving a hydrological model, such as the Richards equation which is a highly nonlinear partial differential equation (PDE). In this paper, we present a novel data-facilitated numerical method for solving the mixed-form Richards equation. This numerical method, which we call the D-GRW (Data-facilitated global Random Walk) method, synergistically integrates adaptive linearization scheme, neural networks, and global random walk in a finite volume discretization framework to produce accurate numerical solutions of the Richards equation with guaranteed convergence under reasonable assumptions. Through three illustrative examples, we demonstrate and discuss the superior accuracy and mass conservation performance of our D-GRW method and compare it with benchmark numerical methods and commercial so
    
[^53]: DOMINO: 多步骤视觉语言推理的双系统

    DOMINO: A Dual-System for Multi-step Visual Language Reasoning. (arXiv:2310.02804v1 [cs.CL])

    [http://arxiv.org/abs/2310.02804](http://arxiv.org/abs/2310.02804)

    本文提出了一种用于多步骤多模态推理的双系统，其中一个系统用于提取视觉信息，另一个系统用于有意推理。这种方法可以避免模型在一个步骤回答复杂问题或被转换文本中的误导信息困扰。

    

    视觉语言推理需要一个系统从信息密集的图像（如图表或绘图）中提取文本或数字，并执行逻辑或算术推理以得出答案。为了解决这个任务，现有的工作依靠以下两种方法：（1）训练在大量数据上的端到端视觉语言模型，或者（2）使用两阶段的流程，其中一个字幕模型将图像转换成文本，然后由另一个大型语言模型读取文本以推断出答案。然而，前一种方法强迫模型用单个步骤回答复杂问题，后一种方法容易产生转换文本中的不准确或分散注意力的信息，从而让语言模型混淆。在这项工作中，我们提出了一种用于多步骤多模型推理的双系统，它包含用于视觉信息提取的“System-1”步骤和用于有意推理的“System-2”步骤。给定一个输入，System-2将问题分解为原子子步骤，每个子步骤指导System-1进行信息提取。

    Visual language reasoning requires a system to extract text or numbers from information-dense images like charts or plots and perform logical or arithmetic reasoning to arrive at an answer. To tackle this task, existing work relies on either (1) an end-to-end vision-language model trained on a large amount of data, or (2) a two-stage pipeline where a captioning model converts the image into text that is further read by another large language model to deduce the answer. However, the former approach forces the model to answer a complex question with one single step, and the latter approach is prone to inaccurate or distracting information in the converted text that can confuse the language model. In this work, we propose a dual-system for multi-step multimodal reasoning, which consists of a "System-1" step for visual information extraction and a "System-2" step for deliberate reasoning. Given an input, System-2 breaks down the question into atomic sub-steps, each guiding System-1 to extr
    
[^54]: 超越单节点：在分布式系统上实现大规模机器学习模型加速

    MAD Max Beyond Single-Node: Enabling Large Machine Learning Model Acceleration on Distributed Systems. (arXiv:2310.02784v1 [cs.DC])

    [http://arxiv.org/abs/2310.02784](http://arxiv.org/abs/2310.02784)

    该研究提出了一个性能建模框架，在分布式系统上实现了大规模机器学习模型的加速，获得了2.24倍和5.27倍的吞吐量提升潜力。

    

    训练和部署大规模机器学习（ML）模型是耗时且需要大量分布式计算基础设施。根据实际情况在数据中心规模基础设施上进行大模型训练，我们发现14~32%的GPU小时用于通信，没有重叠计算。为了尽量减少等待通信延迟，本研究开发了一个灵活的性能建模框架，指导并行化和硬件软件共同设计策略。利用最先进的GPU训练硬件上的一套实际大规模ML模型，我们展示了预训练和推断场景分别可以提高2.24倍和5.27倍的吞吐量。

    Training and deploying large machine learning (ML) models is time-consuming and requires significant distributed computing infrastructures. Based on real-world large model training on datacenter-scale infrastructures, we show 14~32% of all GPU hours are spent on communication with no overlapping computation. To minimize the outstanding communication latency, in this work, we develop an agile performance modeling framework to guide parallelization and hardware-software co-design strategies. Using the suite of real-world large ML models on state-of-the-art GPU training hardware, we demonstrate 2.24x and 5.27x throughput improvement potential for pre-training and inference scenarios, respectively.
    
[^55]: 使用对抗性环境设计探索通用强化学习算法

    Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design. (arXiv:2310.02782v1 [cs.LG])

    [http://arxiv.org/abs/2310.02782](http://arxiv.org/abs/2310.02782)

    通过对抗性环境设计，我们提出了一种通用强化学习算法，通过元学习更新规则和自动生成课程来提高算法的泛化性能，并引入了一种新的遗憾近似方法，名为算法遗憾（AR）。

    

    在过去的十年中，深度强化学习取得了巨大的进展，这些进展是由人类研究人员手动设计的算法推动的。最近，已经证明可以元学习更新规则，希望发现在各种强化学习任务中表现良好的算法。尽管像学习策略梯度（LPG）这样的算法取得了令人印象深刻的初步结果，但是当这些算法应用于未见过的环境时仍存在泛化差距。在这项工作中，我们研究了元训练分布的特征如何影响这些算法的泛化性能。受到这个分析的启发，并借鉴了无监督环境设计（UED）的思想，我们提出了一种自动生成课程的新方法，以最大化元学习优化器的遗憾，此外还提出了一种新的遗憾近似方法，我们称之为算法遗憾（AR）。我们的方法是通过环境设计获得的通用强化学习优化器。

    The past decade has seen vast progress in deep reinforcement learning (RL) on the back of algorithms manually designed by human researchers. Recently, it has been shown that it is possible to meta-learn update rules, with the hope of discovering algorithms that can perform well on a wide range of RL tasks. Despite impressive initial results from algorithms such as Learned Policy Gradient (LPG), there remains a generalization gap when these algorithms are applied to unseen environments. In this work, we examine how characteristics of the meta-training distribution impact the generalization performance of these algorithms. Motivated by this analysis and building on ideas from Unsupervised Environment Design (UED), we propose a novel approach for automatically generating curricula to maximize the regret of a meta-learned optimizer, in addition to a novel approximation of regret, which we name algorithmic regret (AR). The result is our method, General RL Optimizers Obtained Via Environment
    
[^56]: 随机环境中的预期流网络和双人零和游戏

    Expected flow networks in stochastic environments and two-player zero-sum games. (arXiv:2310.02779v1 [cs.LG])

    [http://arxiv.org/abs/2310.02779](http://arxiv.org/abs/2310.02779)

    该论文提出了预期流网络（EFlowNets）和对抗流网络（AFlowNets），分别应用于随机环境和双人零和游戏中。在随机任务中，EFlowNets表现优于其他方法，在双人游戏中，AFlowNets在自我对弈中找到了80%以上的最佳动作，并在竞赛中超过了AlphaZero。

    

    生成流网络（GFlowNets）是训练用于匹配给定分布的序列采样模型。GFlowNets已成功应用于各种结构对象生成任务，能够迅速采样出多样化且高回报的对象。我们提出了预期流网络（EFlowNets），将GFlowNets扩展到随机环境中。我们展示了EFlowNets在随机任务（如蛋白质设计）中优于其他GFlowNet方案。然后，我们将EFlowNets的概念扩展到对抗环境中，为双人零和游戏提出了对抗流网络（AFlowNets）。我们展示了AFlowNets通过自我对弈在Connect-4游戏中能找到80%以上的最佳动作，并在竞赛中优于AlphaZero。

    Generative flow networks (GFlowNets) are sequential sampling models trained to match a given distribution. GFlowNets have been successfully applied to various structured object generation tasks, sampling a diverse set of high-reward objects quickly. We propose expected flow networks (EFlowNets), which extend GFlowNets to stochastic environments. We show that EFlowNets outperform other GFlowNet formulations in stochastic tasks such as protein design. We then extend the concept of EFlowNets to adversarial environments, proposing adversarial flow networks (AFlowNets) for two-player zero-sum games. We show that AFlowNets learn to find above 80% of optimal moves in Connect-4 via self-play and outperform AlphaZero in tournaments.
    
[^57]: 图神经网络和时间序列作为有向图进行质量识别

    Graph Neural Networks and Time Series as Directed Graphs for Quality Recognition. (arXiv:2310.02774v1 [cs.LG])

    [http://arxiv.org/abs/2310.02774](http://arxiv.org/abs/2310.02774)

    这篇论文研究了图神经网络在时间序列上的应用，将时间序列视为有向图来编码时间依赖性，并开发了两个几何深度学习模型，分别用于监督分类和信号重构，在质量识别问题上取得了良好效果。

    

    图神经网络（GNNs）正成为研究时间序列的核心，与现有的算法，如时间卷积网络和循环神经网络结合使用。在本文中，我们将时间序列视为有向图，以使其拓扑结构编码时间依赖性，并开始探索在其中应用GNN架构的有效性。我们开发了两种不同的几何深度学习模型，一个是监督分类器，一个是类似于自编码模型的信号重构模型。我们将这些模型应用于质量识别问题。

    Graph Neural Networks (GNNs) are becoming central in the study of time series, coupled with existing algorithms as Temporal Convolutional Networks and Recurrent Neural Networks. In this paper, we see time series themselves as directed graphs, so that their topology encodes time dependencies and we start to explore the effectiveness of GNNs architectures on them. We develop two distinct Geometric Deep Learning models, a supervised classifier and an autoencoder-like model for signal reconstruction. We apply these models on a quality recognition problem.
    
[^58]: 在动态和非平稳环境中的基于核的函数学习

    Kernel-based function learning in dynamic and non stationary environments. (arXiv:2310.02767v1 [cs.LG])

    [http://arxiv.org/abs/2310.02767](http://arxiv.org/abs/2310.02767)

    本文研究了在非平稳分布下的基于核的岭回归问题，在环境监测和传感器重构等探索-利用问题中具有重要应用价值。

    

    机器学习中的一个中心主题是从稀疏和噪声数据中进行函数估计。本文考虑了在非平稳分布下的基于核的岭回归，并给出了收敛条件，包括可能无限次发生随机调整的情况。

    One central theme in machine learning is function estimation from sparse and noisy data. An example is supervised learning where the elements of the training set are couples, each containing an input location and an output response. In the last decades, a substantial amount of work has been devoted to design estimators for the unknown function and to study their convergence to the optimal predictor, also characterizing the learning rate. These results typically rely on stationary assumptions where input locations are drawn from a probability distribution that does not change in time. In this work, we consider kernel-based ridge regression and derive convergence conditions under non stationary distributions, addressing also cases where stochastic adaption may happen infinitely often. This includes the important exploration-exploitation problems where e.g. a set of agents/robots has to monitor an environment to reconstruct a sensorial field and their movements rules are continuously upda
    
[^59]: 自动摘要评估的比较研究和框架：LangChain和混合算法

    Comparative Study and Framework for Automated Summariser Evaluation: LangChain and Hybrid Algorithms. (arXiv:2310.02759v1 [cs.LG])

    [http://arxiv.org/abs/2310.02759](http://arxiv.org/abs/2310.02759)

    本研究关注自动摘要评估的比较研究和框架，通过使用LangChain和混合算法对PDF文档进行摘要和提取关键信息，以确定用户对摘要内容的理解程度如何。

    

    自动文章评分（AES）被证明是一种尖端技术，评分技术用于各种目的，可靠的得分是基于有影响力的变量计算得出的，这些变量可以根据领域通过不同的方法计算出来。本研究集中于用户对给定主题的理解，分析是基于使用大型语言模型的评分指数，用户可以比较和对比最近学习的主题的理解程度，结果则对学习分析有所贡献，以提高学习能力。本研究的重点是对PDF文档进行摘要和衡量用户对其内容的理解程度。该过程涉及使用LangChain工具对PDF进行摘要和提取关键信息，通过采用这种技术，本研究旨在确定用户对摘要内容的理解程度如何。

    Automated Essay Score (AES) is proven to be one of the cutting-edge technologies. Scoring techniques are used for various purposes. Reliable scores are calculated based on influential variables. Such variables can be computed by different methods based on the domain. The research is concentrated on the user's understanding of a given topic. The analysis is based on a scoring index by using Large Language Models. The user can then compare and contrast the understanding of a topic that they recently learned. The results are then contributed towards learning analytics and progression is made for enhancing the learning ability. In this research, the focus is on summarizing a PDF document and gauging a user's understanding of its content. The process involves utilizing a Langchain tool to summarize the PDF and extract the essential information. By employing this technique, the research aims to determine how well the user comprehends the summarized content.
    
[^60]: MUNCH: 建模独特且可控制的头部

    MUNCH: Modelling Unique 'N Controllable Heads. (arXiv:2310.02753v1 [cs.CV])

    [http://arxiv.org/abs/2310.02753](http://arxiv.org/abs/2310.02753)

    本论文提出了一种方法，可以自动生成质量高、多样性强、可控制的逼真三维人头，具有可解释的网络设计。方法包括几何生成器、渲染图生成器和颜色变换模型。同时还引入了独特性和新颖性的量化指标。

    

    对于计算机视觉研究人员来说，自动生成三维人头一直是一个引人入胜且具有挑战性的任务。现有的方法可以合成逼真的角色，但对于生成结果的多样性和质量的控制有限，并且形状和纹理之间的相关性也受到限制。我们提出的方法在质量、多样性、控制和逼真性方面都具备了可行性，并且还提供了可解释的网络设计，对于游戏设计艺术家来说这些都是理想的特点。首先，我们提出的几何生成器可以识别脱耦的潜在方向并生成新颖且多样的样本。然后，渲染图生成器学习合成多个高保真度的基于物理的渲染图，包括反照率、光泽度、镜面反射和法线方向。对于更细粒度的控制输出的艺术家，我们引入了一种新颖的颜色变换模型，允许对生成的图像进行语义颜色控制。我们还引入了一种可量化的指标，称为独特性和新颖性。

    The automated generation of 3D human heads has been an intriguing and challenging task for computer vision researchers. Prevailing methods synthesize realistic avatars but with limited control over the diversity and quality of rendered outputs and suffer from limited correlation between shape and texture of the character. We propose a method that offers quality, diversity, control, and realism along with explainable network design, all desirable features to game-design artists in the domain. First, our proposed Geometry Generator identifies disentangled latent directions and generate novel and diverse samples. A Render Map Generator then learns to synthesize multiply high-fidelty physically-based render maps including Albedo, Glossiness, Specular, and Normals. For artists preferring fine-grained control over the output, we introduce a novel Color Transformer Model that allows semantic color control over generated maps. We also introduce quantifiable metrics called Uniqueness and Novelt
    
[^61]: 公平特征选择：多目标遗传算法比较

    Fair Feature Selection: A Comparison of Multi-Objective Genetic Algorithms. (arXiv:2310.02752v1 [cs.LG])

    [http://arxiv.org/abs/2310.02752](http://arxiv.org/abs/2310.02752)

    本文比较了两种基于多目标优化方法的公平特征选择遗传算法，旨在同时最大化分类器的准确性和公平性。这是第一项系统性地进行比较的研究。

    

    机器学习分类器被广泛用于做出对人们生活影响重大的决策（如接受或拒绝贷款、招聘决策等）。在这些应用中，学习到的分类器需要在不同人群（性别、种族等不同变量值）之间既准确又公平。本文侧重于公平特征选择，即选择一个特征子集，旨在最大化分类器所做预测的准确性和公平性。具体而言，我们比较了最近提出的两种用于公平特征选择的遗传算法：（a）基于帕累托优势的遗传算法；（b）基于词典优化的遗传算法，其中准确性的最大化优先级高于公平性的最大化。两种遗传算法使用相同的准确性和公平性指标，可以进行控制比较。据我们所知，这是第一项比较两种不同多目标优化方法的公平特征选择遗传算法的研究。

    Machine learning classifiers are widely used to make decisions with a major impact on people's lives (e.g. accepting or denying a loan, hiring decisions, etc). In such applications,the learned classifiers need to be both accurate and fair with respect to different groups of people, with different values of variables such as sex and race. This paper focuses on fair feature selection for classification, i.e. methods that select a feature subset aimed at maximising both the accuracy and the fairness of the predictions made by a classifier. More specifically, we compare two recently proposed Genetic Algorithms (GAs) for fair feature selection that are based on two different multi-objective optimisation approaches: (a) a Pareto dominance-based GA; and (b) a lexicographic optimisation-based GA, where maximising accuracy has higher priority than maximising fairness. Both GAs use the same measures of accuracy and fairness, allowing for a controlled comparison. As far as we know, this is the fi
    
[^62]: SHOT：抑制梯度优化轨迹中的海森矩阵以用于基于梯度的元学习

    SHOT: Suppressing the Hessian along the Optimization Trajectory for Gradient-Based Meta-Learning. (arXiv:2310.02751v1 [cs.LG])

    [http://arxiv.org/abs/2310.02751](http://arxiv.org/abs/2310.02751)

    本文提出了一种名为SHOT的算法，通过抑制梯度优化轨迹中的海森矩阵来改进基于梯度的元学习，在标准的少样本学习任务中取得了优于基线模型的效果。

    

    本文假设梯度优化的元学习（GBML）在内循环中隐式地抑制了优化轨迹上的海森矩阵。基于这个假设，我们引入了一种名为SHOT（抑制梯度优化轨迹中的海森矩阵）的算法，通过最小化目标模型和参考模型参数之间的距离来抑制内循环中的海森矩阵。尽管涉及高阶项，SHOT并不显著增加基线模型的计算复杂度。SHOT对GBML中使用的算法和架构都是不可知的，使其具有高度的通用性，并适用于任何GBML基线模型。为了验证SHOT的有效性，我们在标准的少样本学习任务上进行了实证测试，并进行了动力学的定性分析。我们通过实验证实了我们的假设，并证明了SHOT优于相应的基线模型。代码可在https://github.com/JunHoo-Lee/SHOT找到。

    In this paper, we hypothesize that gradient-based meta-learning (GBML) implicitly suppresses the Hessian along the optimization trajectory in the inner loop. Based on this hypothesis, we introduce an algorithm called SHOT (Suppressing the Hessian along the Optimization Trajectory) that minimizes the distance between the parameters of the target and reference models to suppress the Hessian in the inner loop. Despite dealing with high-order terms, SHOT does not increase the computational complexity of the baseline model much. It is agnostic to both the algorithm and architecture used in GBML, making it highly versatile and applicable to any GBML baseline. To validate the effectiveness of SHOT, we conduct empirical tests on standard few-shot learning tasks and qualitatively analyze its dynamics. We confirm our hypothesis empirically and demonstrate that SHOT outperforms the corresponding baseline. Code is available at: https://github.com/JunHoo-Lee/SHOT
    
[^63]: SALSA: 语义感知的潜空间自编码器

    SALSA: Semantically-Aware Latent Space Autoencoder. (arXiv:2310.02744v1 [cs.LG])

    [http://arxiv.org/abs/2310.02744](http://arxiv.org/abs/2310.02744)

    SALSA提出了一种语义感知的潜空间自编码器（SALSA），通过在自编码器中引入对比任务，专门设计用于学习分子之间的图对图相似性。

    

    在深度学习中应用于药物发现的研究中，化学数据通常以简化的分子输入线条输入系统 (SMILES) 序列表示，这可以方便地实施自然语言处理方法之一，即序列到序列自编码器。然而，我们观察到仅仅在SMILES上训练自编码器是不足以学习到语义上有意义的分子表示的。在这里，语义通过分子之间的结构（图对图）相似性来定义。我们通过示例证明，自编码器可能会将结构相似的分子映射到相距较远的编码，导致一个不一致的潜空间，不尊重分子之间的结构相似性。为了解决这个问题，我们提出了Semantically-Aware Latent Space Autoencoder (SALSA)，这是一个经过修改的变换器自编码器，用于学习分子之间的图对图相似性。形式上，对比目标是

    In deep learning for drug discovery, chemical data are often represented as simplified molecular-input line-entry system (SMILES) sequences which allow for straightforward implementation of natural language processing methodologies, one being the sequence-to-sequence autoencoder. However, we observe that training an autoencoder solely on SMILES is insufficient to learn molecular representations that are semantically meaningful, where semantics are defined by the structural (graph-to-graph) similarities between molecules. We demonstrate by example that autoencoders may map structurally similar molecules to distant codes, resulting in an incoherent latent space that does not respect the structural similarities between molecules. To address this shortcoming we propose Semantically-Aware Latent Space Autoencoder (SALSA), a transformer-autoencoder modified with a contrastive task, tailored specifically to learn graph-to-graph similarity between molecules. Formally, the contrastive objective
    
[^64]: 奖励模型集成有助于减轻过度优化问题

    Reward Model Ensembles Help Mitigate Overoptimization. (arXiv:2310.02743v1 [cs.LG])

    [http://arxiv.org/abs/2310.02743](http://arxiv.org/abs/2310.02743)

    本研究通过探究奖励模型集成和保守优化目标的效果，对减轻奖励模型过度优化进行了系统研究。

    

    人类反馈强化学习（RLHF）是一种将大型语言模型微调以遵循指令的标准方法。在这个过程中，学习到的奖励模型被用来近似人类偏好。然而，作为“真实”奖励的不完美表示，这些学习到的奖励模型容易受到过度优化的影响。Gao等人在一个人工反馈实验中研究了这个现象，使用一个较大的“金标准”奖励模型作为真实奖励（而不是人类），并显示过度优化仍然是一个持续存在的问题，无论代理奖励模型和训练数据的大小如何。使用类似的设置，我们进行了一项系统研究，评估了在使用两种优化方法时，使用基于集合的保守优化目标（最坏情况优化和权重不确定性优化）来减轻奖励模型过度优化的有效性。

    Reinforcement learning from human feedback (RLHF) is a standard approach for fine-tuning large language models to follow instructions. As part of this process, learned reward models are used to approximately model human preferences. However, as imperfect representations of the "true" reward, these learned reward models are susceptible to \textit{overoptimization}. Gao et al. (2023) studied this phenomenon in a synthetic human feedback setup with a significantly larger "gold" reward model acting as the true reward (instead of humans) and showed that overoptimization remains a persistent problem regardless of the size of the proxy reward model and training data used. Using a similar setup, we conduct a systematic study to evaluate the efficacy of using ensemble-based conservative optimization objectives, specifically worst-case optimization (WCO) and uncertainty-weighted optimization (UWO), for mitigating reward model overoptimization when using two optimization methods: (a) best-of-n sa
    
[^65]: 借助迁移学习比较不平衡恶意软件Byteplot图像分类的比较分析

    Comparative Analysis of Imbalanced Malware Byteplot Image Classification using Transfer Learning. (arXiv:2310.02742v1 [cs.LG])

    [http://arxiv.org/abs/2310.02742](http://arxiv.org/abs/2310.02742)

    本文通过比较分析了不平衡恶意软件图像分类的六种模型性能，发现类别不平衡程度越高，收敛所需的迭代次数越少，不同模型的性能差异也越大。同时，研究发现ResNet50、EfficientNetB0和DenseNet169可以很好地处理不平衡和平衡的数据，对于恶意软件检测器具有较高的精度。

    

    随着对科技和互联系统的日益依赖，网络安全成为一个主要关注的问题。恶意软件检测器通过比较恶意软件特征来缓解网络攻击。机器学习可以通过自动化特征提取、识别模式和增强动态分析来改进这些检测器。本文比较了六种多类别分类模型在Malimg数据集、混合数据集和Malevis数据集上的性能，以了解类别不平衡对模型性能和收敛性的影响。观察到类别不平衡的程度越高，收敛所需的迭代次数越少，不同模型的性能差异也越大。此外，还观察到对于恶意软件检测器，ResNet50、EfficientNetB0和DenseNet169可以很好地处理不平衡和平衡的数据。在不平衡数据集上获得了最高97%的精度，在中等不平衡数据上获得了最高95%的精度。

    Cybersecurity is a major concern due to the increasing reliance on technology and interconnected systems. Malware detectors help mitigate cyber-attacks by comparing malware signatures. Machine learning can improve these detectors by automating feature extraction, identifying patterns, and enhancing dynamic analysis. In this paper, the performance of six multiclass classification models is compared on the Malimg dataset, Blended dataset, and Malevis dataset to gain insights into the effect of class imbalance on model performance and convergence. It is observed that the more the class imbalance less the number of epochs required for convergence and a high variance across the performance of different models. Moreover, it is also observed that for malware detectors ResNet50, EfficientNetB0, and DenseNet169 can handle imbalanced and balanced data well. A maximum precision of 97% is obtained for the imbalanced dataset, a maximum precision of 95% is obtained on the intermediate imbalance data
    
[^66]: 从事件数据中提取规则用于学习规划研究

    Extracting Rules from Event Data for Study Planning. (arXiv:2310.02735v1 [cs.LG])

    [http://arxiv.org/abs/2310.02735](http://arxiv.org/abs/2310.02735)

    本研究利用事件数据分析高等教育学生的学习路径，通过决策树模型生成数据驱动的规则建议，并用于学习规划。研究发现所选课程序列特征对学业表现有较好解释，为制定更适应性的学习计划提供了思路。

    

    在这项研究中，我们探讨了如何利用校园管理系统的事件数据来分析高等教育学生的学习路径。主要目标是为他们的学习规划提供有价值的指导。我们使用过程和数据挖掘技术来探索所选课程序列对学业成绩的影响。通过使用决策树模型，我们生成以规则形式的数据驱动的学习规划建议，并将其与推荐的学习计划进行比较。评估重点关注RWTH Aachen大学计算机科学学士学位计划的学生，并证明了所提出的课程序列特征有效地解释了学业表现指标。此外，研究结果还建议了开发更具适应性学习计划的途径。

    In this study, we examine how event data from campus management systems can be used to analyze the study paths of higher education students. The main goal is to offer valuable guidance for their study planning. We employ process and data mining techniques to explore the impact of sequences of taken courses on academic success. Through the use of decision tree models, we generate data-driven recommendations in the form of rules for study planning and compare them to the recommended study plan. The evaluation focuses on RWTH Aachen University computer science bachelor program students and demonstrates that the proposed course sequence features effectively explain academic performance measures. Furthermore, the findings suggest avenues for developing more adaptable study plans.
    
[^67]: AI系统的功能可信度通过统计学有效测试

    Functional trustworthiness of AI systems by statistically valid testing. (arXiv:2310.02727v1 [stat.ML])

    [http://arxiv.org/abs/2310.02727](http://arxiv.org/abs/2310.02727)

    作者认为欧盟AI法案对AI系统的质量保证方式存在不足，并指出基于统计学有效测试及准确定义应用是确保AI系统功能可信度的核心。

    

    作者关注欧洲公民的安全、健康和权益问题，因为当前欧盟人工智能（AI）法案的草案对AI系统的符合性评估所需的措施和程序不足。我们注意到，欧盟AI法案的当前草案以及在CEN/CENELEC进行的配套标准化工作，都采取了一个观点，即AI系统的实际功能保证似乎是不切实际且过于复杂的。然而，制定一个符合性评估程序，使未经充分评估的AI系统产生虚假的信任幻象，充其量是幼稚的，充其更糟的情况是严重疏忽的。因此，欧盟AI法案错过了确保通过功能可信度来确保质量和正确分配责任的目的。AI决策系统的可信度首先在于对随机选择的样本进行正确的统计测试，并在定义应用的准确性上。

    The authors are concerned about the safety, health, and rights of the European citizens due to inadequate measures and procedures required by the current draft of the EU Artificial Intelligence (AI) Act for the conformity assessment of AI systems. We observe that not only the current draft of the EU AI Act, but also the accompanying standardization efforts in CEN/CENELEC, have resorted to the position that real functional guarantees of AI systems supposedly would be unrealistic and too complex anyways. Yet enacting a conformity assessment procedure that creates the false illusion of trust in insufficiently assessed AI systems is at best naive and at worst grossly negligent. The EU AI Act thus misses the point of ensuring quality by functional trustworthiness and correctly attributing responsibilities.  The trustworthiness of an AI decision system lies first and foremost in the correct statistical testing on randomly selected samples and in the precision of the definition of the applica
    
[^68]: 神经HMM的端到端训练：在标签和转移概率上

    End-to-End Training of a Neural HMM with Label and Transition Probabilities. (arXiv:2310.02724v1 [cs.LG])

    [http://arxiv.org/abs/2310.02724](http://arxiv.org/abs/2310.02724)

    本研究提出了一种新的端到端神经网络训练方法，通过显式建模和学习隐藏状态之间的转移概率，而不是隐含地编码持续时间统计的空标签。虽然转移模型的训练不会改善识别性能，但对齐质量有积极影响。

    

    本研究探讨了一种新颖的建模方法，用于使用隐马尔可夫模型(HMM)进行端到端神经网络训练，在隐藏状态之间的转移概率被显式地建模和学习。大多数当代序列到序列模型允许通过在给定拓扑结构中对所有可能的标签分割进行求和来进行从头训练。在我们的方法中，存在着显式的、可学习的分割之间的转移概率，而不是隐含地编码持续时间统计的空标签。我们实现了基于GPU的前向-后向算法，可以同时训练标签和转移概率。我们研究了我们模型的识别结果和Viterbi对齐。我们发现，转移模型的训练虽然无法改善识别性能，但对齐质量有积极影响。生成的对齐在最先进的 Viterbi 训练中被证明是可行的目标。

    We investigate a novel modeling approach for end-to-end neural network training using hidden Markov models (HMM) where the transition probabilities between hidden states are modeled and learned explicitly. Most contemporary sequence-to-sequence models allow for from-scratch training by summing over all possible label segmentations in a given topology. In our approach there are explicit, learnable probabilities for transitions between segments as opposed to a blank label that implicitly encodes duration statistics. We implement a GPU-based forward-backward algorithm that enables the simultaneous training of label and transition probabilities. We investigate recognition results and additionally Viterbi alignments of our models. We find that while the transition model training does not improve recognition performance, it has a positive impact on the alignment quality. The generated alignments are shown to be viable targets in state-of-the-art Viterbi trainings.
    
[^69]: 利用模块解耦提升时间图网络

    Leveraging Temporal Graph Networks Using Module Decoupling. (arXiv:2310.02721v1 [cs.LG])

    [http://arxiv.org/abs/2310.02721](http://arxiv.org/abs/2310.02721)

    本研究通过解耦时间图网络的核心模块并使用最少的可学习参数，提出了一种轻量级解耦时间图网络 (LDTGN)。在学习动态图的过程中，LDTGN表现出与之前方法可比甚至领先的结果，并且具有显著更高的吞吐量。

    

    现代学习动态图的方法采用了批处理来替代逐个更新。采用批处理使得这些技术在流式场景中能够处理极快速度的图更新。然而，使用批处理会导致模型的更新频率降低，从而降低了性能。本研究提出了一种解耦策略，使得模型能够在使用批处理的同时频繁地更新。通过将时间图网络的核心模块进行解耦并使用最少的可学习参数进行实现，我们开发了轻量级解耦时间图网络 (LDTGN)，这是一个非常高效的学习动态图的模型。LDTGN在各种动态图基准测试上得到了验证，在吞吐量显著高于之前的方法的同时，提供了可比或具有最新成果的结果。值得注意的是，我们的方法在be数据集上的性能超过之前的方法20%以上。

    Modern approaches for learning on dynamic graphs have adopted the use of batches instead of applying updates one by one. The use of batches allows these techniques to become helpful in streaming scenarios where updates to graphs are received at extreme speeds. Using batches, however, forces the models to update infrequently, which results in the degradation of their performance. In this work, we suggest a decoupling strategy that enables the models to update frequently while using batches. By decoupling the core modules of temporal graph networks and implementing them using a minimal number of learnable parameters, we have developed the Lightweight Decoupled Temporal Graph Network (LDTGN), an exceptionally efficient model for learning on dynamic graphs. LDTG was validated on various dynamic graph benchmarks, providing comparable or state-of-the-art results with significantly higher throughput than previous art. Notably, our method outperforms previous approaches by more than 20\% on be
    
[^70]: 通过广义逆理解全色增强算法

    Understanding Pan-Sharpening via Generalized Inverse. (arXiv:2310.02718v1 [cs.LG])

    [http://arxiv.org/abs/2310.02718](http://arxiv.org/abs/2310.02718)

    通过研究广义逆理论，本文提出了一种新的全色增强算法，该算法基于简单矩阵方程描述全色增强问题，并探讨解的条件和光谱、空间分辨率的获取。通过引入降采样增强方法，我们得到了与分量替代和多分辨率分析方法相对应的广义逆矩阵表达式，并提出了一个新的模型先验来解决全色增强中的理论误差问题。

    

    全色增强算法利用全色图像和多光谱图像获取具有高空间和高光谱的图像。然而，这些算法的优化是根据不同的标准设计的。我们采用简单的矩阵方程来描述全色增强问题，并讨论解的存在条件以及光谱和空间分辨率的获取。我们引入了一种降采样增强方法，以更好地获取空间和光谱降采样矩阵。通过广义逆理论，我们推导出了两种形式的广义逆矩阵表达式，可以对应于两个主要的全色增强方法：分量替代和多分辨率分析方法。具体而言，我们证明了Gram Schmidt自适应(GSA)方法遵循分量替代的广义逆矩阵表达式。我们提出了一个在光谱函数的广义逆矩阵之前的模型先验。我们对理论误差进行了分析。

    Pan-sharpening algorithm utilizes panchromatic image and multispectral image to obtain a high spatial and high spectral image. However, the optimizations of the algorithms are designed with different standards. We adopt the simple matrix equation to describe the Pan-sharpening problem. The solution existence condition and the acquirement of spectral and spatial resolution are discussed. A down-sampling enhancement method was introduced for better acquiring the spatial and spectral down-sample matrices. By the generalized inverse theory, we derived two forms of general inverse matrix formulations that can correspond to the two prominent classes of Pan-sharpening methods, that is, component substitution and multi-resolution analysis methods. Specifically, the Gram Schmidt Adaptive(GSA) was proved to follow the general inverse matrix formulation of component substitution. A model prior to the general inverse matrix of the spectral function was rendered. The theoretical errors are analyzed
    
[^71]: 在用户模型错误的情况下的在线聚类强化学习

    Online Clustering of Bandits with Misspecified User Models. (arXiv:2310.02717v1 [cs.LG])

    [http://arxiv.org/abs/2310.02717](http://arxiv.org/abs/2310.02717)

    本文介绍了在用户模型错误的情况下的聚类强化学习问题，并提出了两个鲁棒的聚类强化学习算法，以解决用户模型偏差的挑战。

    

    上下文线性强化学习是一个重要的在线学习问题，在每轮中，给定臂的特征，学习代理选择一个臂来最大化长期的累积奖励。聚类强化学习是一系列工作，利用用户偏好的协同效应，并在经典的线性强化学习算法上取得了显著的改进。然而，现有的聚类强化学习算法需要正确规定线性用户模型，当这个关键假设不成立时，可能会失败。如何为在用户模型错误的实际情况下设计鲁棒的聚类强化学习算法仍然是一个开放的问题。在本文中，我们首次提出了在用户模型错误的情况下的聚类强化学习问题，其中用户模型中的期望奖励可能有偏差，不是完美的线性模型。我们设计了两个鲁棒的聚类强化学习算法RCLUMB和RSCLUMB（分别用动态图和集合表示学习到的聚类结构）。

    The contextual linear bandit is an important online learning problem where given arm features, a learning agent selects an arm at each round to maximize the cumulative rewards in the long run. A line of works, called the clustering of bandits (CB), utilize the collaborative effect over user preferences and have shown significant improvements over classic linear bandit algorithms. However, existing CB algorithms require well-specified linear user models and can fail when this critical assumption does not hold. Whether robust CB algorithms can be designed for more practical scenarios with misspecified user models remains an open problem. In this paper, we are the first to present the important problem of clustering of bandits with misspecified user models (CBMUM), where the expected rewards in user models can be perturbed away from perfect linear models. We devise two robust CB algorithms, RCLUMB and RSCLUMB (representing the learned clustering structure with dynamic graph and sets, resp
    
[^72]: scHyena: 基于全长单细胞RNA-Seq的大脑分析的基础模型

    scHyena: Foundation Model for Full-Length Single-Cell RNA-Seq Analysis in Brain. (arXiv:2310.02713v1 [cs.LG])

    [http://arxiv.org/abs/2310.02713](http://arxiv.org/abs/2310.02713)

    scHyena是一个基于Transformer架构的模型，称为单细胞Hyena(scHyena)，旨在处理大脑中的全长scRNA-seq数据，并提高分析的准确性。

    

    单细胞RNA测序(scRNA-seq)在揭示复杂组织中微妙的细胞多样性方面取得了重要进展。这在大脑中尤为关键，因为大脑比其他组织类型有更多种类的细胞，以便更深入地了解不同细胞环境下的大脑功能。然而，由于缺失事件所产生的固有测量噪声以及对大量基因表达信息的有限利用，分析scRNA-seq数据仍然是一个挑战。在这项工作中，我们引入了scHyena，这是一个旨在应对这些挑战并提高大脑中scRNA-seq分析准确性的基础模型。具体而言，受到最近的Hyena算子的启发，我们设计了一种新颖的Transformer架构，称为单细胞Hyena(scHyena)，它配备了线性适配器层、通过基因嵌入实现的位置编码和一个双向Hyena算子。这使我们能够处理全长的scRNA-seq数据而不丢失信息。

    Single-cell RNA sequencing (scRNA-seq) has made significant strides in unraveling the intricate cellular diversity within complex tissues. This is particularly critical in the brain, presenting a greater diversity of cell types than other tissue types, to gain a deeper understanding of brain function within various cellular contexts. However, analyzing scRNA-seq data remains a challenge due to inherent measurement noise stemming from dropout events and the limited utilization of extensive gene expression information. In this work, we introduce scHyena, a foundation model designed to address these challenges and enhance the accuracy of scRNA-seq analysis in the brain. Specifically, inspired by the recent Hyena operator, we design a novel Transformer architecture called singe-cell Hyena (scHyena) that is equipped with a linear adaptor layer, the positional encoding via gene-embedding, and a {bidirectional} Hyena operator. This enables us to process full-length scRNA-seq data without losi
    
[^73]: ED-NeRF: 使用潜空间 NeRF 实现高效的文本引导的 3D 场景编辑

    ED-NeRF: Efficient Text-Guided Editing of 3D Scene using Latent Space NeRF. (arXiv:2310.02712v1 [cs.CV])

    [http://arxiv.org/abs/2310.02712](http://arxiv.org/abs/2310.02712)

    ED-NeRF 提出了一种高效的 3D 场景编辑方法，通过将场景嵌入到潜空间中，得到更快速且更易于编辑的 NeRF 骨干。

    

    最近，文本到图像扩散模型取得了显著进展，在二维图像生成方面取得了突破性的性能。这些进展已经扩展到三维模型，实现了从文本描述中生成新的三维对象。这演变成了 NeRF 编辑方法，通过文本条件允许对现有的三维对象进行操作。然而，现有的 NeRF 编辑技术在性能上面临着一些限制，如训练速度慢和使用的损失函数不充分考虑编辑。为了解决这个问题，我们提出了一种新颖的 3D NeRF 编辑方法，称为 ED-NeRF，通过将真实世界场景成功嵌入到潜扩散模型 (LDM) 的潜空间中，通过独特的细化层。这种方法使我们能够获得一个不仅更快，而且更适合于编辑的 NeRF 骨干，与传统的图像空间 NeRF 编辑相比。此外，我们提出了一种改进的损失函数。

    Recently, there has been a significant advancement in text-to-image diffusion models, leading to groundbreaking performance in 2D image generation. These advancements have been extended to 3D models, enabling the generation of novel 3D objects from textual descriptions. This has evolved into NeRF editing methods, which allow the manipulation of existing 3D objects through textual conditioning. However, existing NeRF editing techniques have faced limitations in their performance due to slow training speeds and the use of loss functions that do not adequately consider editing. To address this, here we present a novel 3D NeRF editing approach dubbed ED-NeRF by successfully embedding real-world scenes into the latent space of the latent diffusion model (LDM) through a unique refinement layer. This approach enables us to obtain a NeRF backbone that is not only faster but also more amenable to editing compared to traditional image space NeRF editing. Furthermore, we propose an improved loss 
    
[^74]: 本地搜索GFlowNets

    Local Search GFlowNets. (arXiv:2310.02710v1 [cs.LG])

    [http://arxiv.org/abs/2310.02710](http://arxiv.org/abs/2310.02710)

    本文提出使用局部搜索训练GFlowNets，通过破坏和重构的方式探索局部邻域，分别由反向和正向策略引导，使得样本偏向高奖励解决方案。

    

    生成流网络(GFlowNets)是一种学习与奖励成比例的离散对象分布的摊还采样方法。GFlowNets具有生成多样样本的显著能力，但由于广泛样本空间上的过度探索，有时难以一致地生成高奖励的样本。本文提出使用局部搜索训练GFlowNets，通过破坏和重构的方式探索局部邻域，分别由反向和正向策略引导。这使得样本偏向高奖励解决方案，而传统的GFlowNet解决方案生成方案则使用正向策略从头生成解决方案。大量实验证明在几个生化任务中取得了显著的性能改进。

    Generative Flow Networks (GFlowNets) are amortized sampling methods that learn a distribution over discrete objects proportional to their rewards. GFlowNets exhibit a remarkable ability to generate diverse samples, yet occasionally struggle to consistently produce samples with high rewards due to over-exploration on wide sample space. This paper proposes to train GFlowNets with local search which focuses on exploiting high rewarded sample space to resolve this issue. Our main idea is to explore the local neighborhood via destruction and reconstruction guided by backward and forward policies, respectively. This allows biasing the samples toward high-reward solutions, which is not possible for a typical GFlowNet solution generation scheme which uses the forward policy to generate the solution from scratch. Extensive experiments demonstrate a remarkable performance improvement in several biochemical tasks. Source code is available: \url{https://github.com/dbsxodud-11/ls_gfn}.
    
[^75]: 通过最大化梯度多样性来解决联邦优化中的混合异构性

    Tackling Hybrid Heterogeneity on Federated Optimization via Gradient Diversity Maximization. (arXiv:2310.02702v1 [cs.LG])

    [http://arxiv.org/abs/2310.02702](http://arxiv.org/abs/2310.02702)

    本文探讨了混合异构性如何影响联邦优化，并提出了一种通过最大化梯度多样性来减轻混合异构性负面影响的方法。

    

    联邦学习是一种分布式机器学习范式，其中数据样本被分散和分布在多个客户端之间。这些样本可能表现出统计异质性，即数据分布在客户端之间不是独立和相同的。此外，系统异质性，即客户端计算能力的变化，会给联邦学习带来偏差。统计和系统异质性的综合效应可以显著降低联邦优化的效率。然而，混合异构性的影响并没有得到严谨的讨论。本文通过研究服务器端优化，探讨了混合异构性如何影响联邦优化。理论结果表明，在服务器更新方向上自适应地最大化梯度多样性可以帮助减轻混合异构性的潜在负面影响。为此，我们引入了一种新颖的基于服务器端梯度的优化器。

    Federated learning refers to a distributed machine learning paradigm in which data samples are decentralized and distributed among multiple clients. These samples may exhibit statistical heterogeneity, which refers to data distributions are not independent and identical across clients. Additionally, system heterogeneity, or variations in the computational power of the clients, introduces biases into federated learning. The combined effects of statistical and system heterogeneity can significantly reduce the efficiency of federated optimization. However, the impact of hybrid heterogeneity is not rigorously discussed. This paper explores how hybrid heterogeneity affects federated optimization by investigating server-side optimization. The theoretical results indicate that adaptively maximizing gradient diversity in server update direction can help mitigate the potential negative consequences of hybrid heterogeneity. To this end, we introduce a novel server-side gradient-based optimizer \
    
[^76]: 探索通过减少自适应无偏客户采样方差的联邦优化

    Exploring Federated Optimization by Reducing Variance of Adaptive Unbiased Client Sampling. (arXiv:2310.02698v1 [cs.LG])

    [http://arxiv.org/abs/2310.02698](http://arxiv.org/abs/2310.02698)

    本文通过减少自适应无偏客户采样方差，探索了联邦优化中的一系列自适应客户采样技术，并提出了一种名为K-Vib的新型采样器，显著提高了联邦学习性能。

    

    联邦学习系统通常对一部分客户进行采样来进行训练过程。值得注意的是，基于来自采样客户的信息建立全局模型的全局估计方差与联邦优化质量密切相关。本文探讨了一系列“免费”的自适应客户采样技术，其中服务器构建了有前途的采样概率和可靠的全局估计，而无需额外的本地通信和计算。我们捕捉了采样过程中的一个小变体，并相应改进了全局估计。在此基础上，我们提出了一种名为K-Vib的新型采样器，它解决了在联邦优化中遵循客户采样的在线凸优化问题。它在通信预算K的情况下实现了改进的线性速率上升，具有遗憾边界$\tilde{\mathcal{O}}\big(N^{\frac{1}{3}}T^{\frac{2}{3}}/K^{\frac{4}{}3}\big)$。结果是，它显著提高了联邦学习性能。

    Federated Learning (FL) systems usually sample a fraction of clients to conduct a training process. Notably, the variance of global estimates for updating the global model built on information from sampled clients is highly related to federated optimization quality. This paper explores a line of "free" adaptive client sampling techniques in federated optimization, where the server builds promising sampling probability and reliable global estimates without requiring additional local communication and computation. We capture a minor variant in the sampling procedure and improve the global estimation accordingly. Based on that, we propose a novel sampler called K-Vib, which solves an online convex optimization respecting client sampling in federated optimization. It achieves improved a linear speed up on regret bound $\tilde{\mathcal{O}}\big(N^{\frac{1}{3}}T^{\frac{2}{3}}/K^{\frac{4}{3}}\big)$ with communication budget $K$. As a result, it significantly improves the performance of federat
    
[^77]: 高阶数组建模的概率块项分解

    Probabilistic Block Term Decomposition for the Modelling of Higher-Order Arrays. (arXiv:2310.02694v1 [stat.ML])

    [http://arxiv.org/abs/2310.02694](http://arxiv.org/abs/2310.02694)

    本研究提出了一种高效的变分贝叶斯概率块项分解（pBTD）方法，适用于高阶数组的建模，通过使用von-Mises Fisher矩阵分布来实现正交性约束。实验结果表明，pBTD在噪声数据和模型顺序量化方面具有良好的性能。

    

    张量在科学和工程中无处不在，张量分解方法已经成为表征高阶结构的重要工具。分解包括外积秩标准多项式分解（CPD）以及多线性秩图尔克分解，其中块项分解（BTD）是两种表示之间的结构化中间插值。虽然CPD、图尔克和BTD传统上依赖于最大似然估计，但已经使用贝叶斯推断形成了概率CPD和图尔克。我们提出了一种高效的变分贝叶斯概率BTD，它使用von-Mises Fisher矩阵分布在形成BTD的多线性图尔克部分中施加正交性。在合成和两个实际数据集上，我们突出了贝叶斯推断过程，并使用提议的pBTD对噪声数据进行了演示和模型顺序量化。我们发现概率BTD能够量化...

    Tensors are ubiquitous in science and engineering and tensor factorization approaches have become important tools for the characterization of higher order structure. Factorizations includes the outer-product rank Canonical Polyadic Decomposition (CPD) as well as the multi-linear rank Tucker decomposition in which the Block-Term Decomposition (BTD) is a structured intermediate interpolating between these two representations. Whereas CPD, Tucker, and BTD have traditionally relied on maximum-likelihood estimation, Bayesian inference has been use to form probabilistic CPD and Tucker. We propose, an efficient variational Bayesian probabilistic BTD, which uses the von-Mises Fisher matrix distribution to impose orthogonality in the multi-linear Tucker parts forming the BTD. On synthetic and two real datasets, we highlight the Bayesian inference procedure and demonstrate using the proposed pBTD on noisy data and for model order quantification. We find that the probabilistic BTD can quantify su
    
[^78]: 使用傅里叶神经算子的海洋子区尺度稳健参数化

    Robust Ocean Subgrid-Scale Parameterizations Using Fourier Neural Operators. (arXiv:2310.02691v1 [cs.LG])

    [http://arxiv.org/abs/2310.02691](http://arxiv.org/abs/2310.02691)

    本文使用傅里叶神经算子开发了海洋子区尺度稳健参数化方法，展示了其准确性和普适性，为解决气候模拟中长期预测误差的问题提供了潜在解决方案。

    

    在气候模拟中，小尺度过程塑造了海洋动力学，但直接解决这些过程在计算上代价昂贵。因此，常常使用经验参数化方法来近似它们的贡献，在长期预测中会产生显著的误差。本文基于傅里叶神经算子开发了参数化方法，并展示了与其他方法相比的准确性和普适性。最后，我们讨论了在频域中运行神经网络的潜力和局限性，为未来的研究铺平了道路。

    In climate simulations, small-scale processes shape ocean dynamics but remain computationally expensive to resolve directly. For this reason, their contributions are commonly approximated using empirical parameterizations, which lead to significant errors in long-term projections. In this work, we develop parameterizations based on Fourier Neural Operators, showcasing their accuracy and generalizability in comparison to other approaches. Finally, we discuss the potential and limitations of neural networks operating in the frequency domain, paving the way for future investigation.
    
[^79]: 扩散生成流采样器：通过部分轨迹优化改善学习信号

    Diffusion Generative Flow Samplers: Improving learning signals through partial trajectory optimization. (arXiv:2310.02679v1 [cs.LG])

    [http://arxiv.org/abs/2310.02679](http://arxiv.org/abs/2310.02679)

    这项工作介绍了一种名为扩散生成流采样器（DGFS）的采样框架，通过将学习过程分解为短的部分轨迹段，实现从难以处理的高维密度函数中进行采样。它通过利用中间的学习信号和非策略探索能力来改善学习信号的分配问题。

    

    我们解决了从难以处理的高维密度函数中进行采样的问题，这是在机器学习和统计中经常出现的基本任务。我们扩展了最近的基于采样的方法，利用控制的随机过程来模拟这些目标密度的近似样本。这些方法的主要缺点是训练目标需要计算完整的轨迹，导致由于使用完整轨迹和只在终端时间存在的学习信号的使用而产生缓慢的信用分配问题。在这项工作中，我们提出了扩散生成流采样器（DGFS），这是一个基于采样的框架，可以将学习过程可行地分解为短的部分轨迹段，通过参数化一个额外的“流函数”。我们的方法借鉴了生成流网络（GFlowNets）的理论，使我们能够利用中间的学习信号，并从非策略探索能力中受益。

    We tackle the problem of sampling from intractable high-dimensional density functions, a fundamental task that often appears in machine learning and statistics. We extend recent sampling-based approaches that leverage controlled stochastic processes to model approximate samples from these target densities. The main drawback of these approaches is that the training objective requires full trajectories to compute, resulting in sluggish credit assignment issues due to use of entire trajectories and a learning signal present only at the terminal time. In this work, we present Diffusion Generative Flow Samplers (DGFS), a sampling-based framework where the learning process can be tractably broken down into short partial trajectory segments, via parameterizing an additional "flow function". Our method takes inspiration from the theory developed for generative flow networks (GFlowNets), allowing us to make use of intermediate learning signals and benefit from off-policy exploration capabilitie
    
[^80]: PostRainBench: 一种全面的降水预测基准和新模型

    PostRainBench: A comprehensive benchmark and a new model for precipitation forecasting. (arXiv:2310.02676v1 [cs.LG])

    [http://arxiv.org/abs/2310.02676](http://arxiv.org/abs/2310.02676)

    PostRainBench是一个全面的降水预测基准，结合AI后处理技术和传统的数值天气预报方法，能够增强准确性并解决复杂的降水预测挑战。

    

    准确的降水预测是一项具有科学和社会重要性的重大挑战。数据驱动方法已经成为解决这个挑战的广泛采用的解决方案。然而，仅依赖数据驱动方法在模拟基础物理过程方面有限，使得准确预测困难。将基于人工智能的后处理技术与传统的数值天气预报（NWP）方法相结合，为提高预测准确性提供了更有效的解决方案。尽管之前进行过后处理的尝试，但由于不同位置的降水数据失衡和多个气象变量之间的复杂关系，准确预测大雨仍然具有挑战性。为了解决这些限制，我们提出了PostRainBench，这是一个全面的多变量NWP后处理基准，包括三个用于NWP后处理降水预测的数据集。我们提出了一种简单而有效的渠道注意力模型CAMT。

    Accurate precipitation forecasting is a vital challenge of both scientific and societal importance. Data-driven approaches have emerged as a widely used solution for addressing this challenge. However, solely relying on data-driven approaches has limitations in modeling the underlying physics, making accurate predictions difficult. Coupling AI-based post-processing techniques with traditional Numerical Weather Prediction (NWP) methods offers a more effective solution for improving forecasting accuracy. Despite previous post-processing efforts, accurately predicting heavy rainfall remains challenging due to the imbalanced precipitation data across locations and complex relationships between multiple meteorological variables. To address these limitations, we introduce the PostRainBench, a comprehensive multi-variable NWP post-processing benchmark consisting of three datasets for NWP post-processing-based precipitation forecasting. We propose CAMT, a simple yet effective Channel Attention
    
[^81]: 关于扩散模型记忆化的研究

    On Memorization in Diffusion Models. (arXiv:2310.02664v1 [cs.LG])

    [http://arxiv.org/abs/2310.02664](http://arxiv.org/abs/2310.02664)

    本论文研究了扩散模型的记忆化行为，发现记忆化倾向于在较小的数据集上发生。通过定义有效模型记忆化 (EMM) 这一指标，量化了数据分布和模型配置对记忆化行为的影响。

    

    近年来，由于其生成新颖高质量样本的能力，扩散模型引起了广泛的研究兴趣。然而，通过典型的训练目标，即去噪得分匹配，扩散模型只能生成复制训练数据的样本，这表明在理论上会出现记忆化的行为，这与现有先进扩散模型的普遍泛化能力相矛盾，因此需要深入理解。我们观察到记忆化行为倾向于在较小的数据集上发生，我们提出了有效模型记忆化(EMM)的定义，这是一种衡量学习的扩散模型在最大数据集上近似其理论最优点的度量标准。然后，我们量化了影响这些记忆化行为的重要因素，重点关注数据分布和模型配置。

    Due to their capacity to generate novel and high-quality samples, diffusion models have attracted significant research interest in recent years. Notably, the typical training objective of diffusion models, i.e., denoising score matching, has a closed-form optimal solution that can only generate training data replicating samples. This indicates that a memorization behavior is theoretically expected, which contradicts the common generalization ability of state-of-the-art diffusion models, and thus calls for a deeper understanding. Looking into this, we first observe that memorization behaviors tend to occur on smaller-sized datasets, which motivates our definition of effective model memorization (EMM), a metric measuring the maximum size of training data at which a learned diffusion model approximates its theoretical optimum. Then, we quantify the impact of the influential factors on these memorization behaviors in terms of EMM, focusing primarily on data distribution, model configuratio
    
[^82]: 对资源受限的FPGA的时间序列Transformer模型进行量化感知训练的研究

    A Study of Quantisation-aware Training on Time Series Transformer Models for Resource-constrained FPGAs. (arXiv:2310.02654v1 [cs.LG])

    [http://arxiv.org/abs/2310.02654](http://arxiv.org/abs/2310.02654)

    本研究探索了对时间序列Transformer模型进行量化感知训练的方法，并提出了一种新颖的自适应量化方案，通过匹配量化方案与实际数据分布，可以降低计算开销并保持可接受的精度。此外，该方法在应用于现实世界数据和混合精度量化时表现出鲁棒性。这些发现为模型量化和部署决策提供了参考，并推进了量化技术的发展。

    

    本研究探讨了对时间序列Transformer模型进行量化感知训练（QAT）。我们提出了一种新颖的自适应量化方案，在QAT阶段动态选择对称和非对称方案。我们的方法表明，将量化方案与真实数据分布匹配可以减少计算开销，同时保持可接受的精度。此外，我们的方法在应用于现实世界数据和混合精度量化时表现出鲁棒性，其中大多数对象被量化为4位。我们的发现为模型量化和部署决策提供了参考，并为推进量化技术奠定了基础。

    This study explores the quantisation-aware training (QAT) on time series Transformer models. We propose a novel adaptive quantisation scheme that dynamically selects between symmetric and asymmetric schemes during the QAT phase. Our approach demonstrates that matching the quantisation scheme to the real data distribution can reduce computational overhead while maintaining acceptable precision. Moreover, our approach is robust when applied to real-world data and mixed-precision quantisation, where most objects are quantised to 4 bits. Our findings inform model quantisation and deployment decisions while providing a foundation for advancing quantisation techniques.
    
[^83]: 在需要时聘用：渐进式参与者招募用于基于拍卖的联邦学习

    Hire When You Need to: Gradual Participant Recruitment for Auction-based Federated Learning. (arXiv:2310.02651v1 [cs.LG])

    [http://arxiv.org/abs/2310.02651](http://arxiv.org/abs/2310.02651)

    GPS-AFL是一种基于拍卖的联邦学习渐进式参与者选择方案，通过在多轮训练中逐渐选择数据所有者，解决了冷启动问题和选择偏差对联邦学习的影响。

    

    联邦学习的成功依赖于数据所有者（DOs）的数量和质量，以及他们参与联邦学习模型训练的动机。已经提出了以声誉为基础的联邦学习参与者选择方法。然而，它们仍然面临冷启动问题和对高声誉DOs的潜在选择偏差的挑战。这种偏差可能导致较低声誉的DOs被过早地排除在未来的联邦学习训练轮次中，从而降低了训练数据的多样性和结果模型的泛化能力。为了解决这些挑战，我们提出了基于拍卖的联邦学习渐进式参与者选择方案（GPS-AFL）。与现有的联邦学习激励机制不同，后者通常认为用于联邦学习任务的所有DOs必须一次性选择，GPS-AFL在多轮训练中逐渐选择所需的DOs，随着通过重复交互逐渐揭示更多信息。它的设计旨在在成本和效用之间取得平衡。

    The success of federated Learning (FL) depends on the quantity and quality of the data owners (DOs) as well as their motivation to join FL model training. Reputation-based FL participant selection methods have been proposed. However, they still face the challenges of the cold start problem and potential selection bias towards highly reputable DOs. Such a bias can result in lower reputation DOs being prematurely excluded from future FL training rounds, thereby reducing the diversity of training data and the generalizability of the resulting models. To address these challenges, we propose the Gradual Participant Selection scheme for Auction-based Federated Learning (GPS-AFL). Unlike existing AFL incentive mechanisms which generally assume that all DOs required for an FL task must be selected in one go, GPS-AFL gradually selects the required DOs over multiple rounds of training as more information is revealed through repeated interactions. It is designed to strike a balance between cost s
    
[^84]: 强化学习基础：朝向具有基础先验辅助的具身通用智能体

    Foundation Reinforcement Learning: towards Embodied Generalist Agents with Foundation Prior Assistance. (arXiv:2310.02635v1 [cs.RO])

    [http://arxiv.org/abs/2310.02635](http://arxiv.org/abs/2310.02635)

    本研究提出了一种基于具身基础先验的基础强化学习框架，通过加速训练过程来提高样本效率。

    

    最近人们已经表明，从互联网规模的数据中进行大规模预训练是构建通用模型的关键，正如在NLP中所见。为了构建具身通用智能体，我们和许多其他研究者假设这种基础先验也是不可或缺的组成部分。然而，目前尚不清楚如何以适当的具体形式表示这些具身基础先验，以及它们应该如何在下游任务中使用。在本文中，我们提出了一组直观有效的具身先验，包括基础策略、价值和成功奖励。所提出的先验是基于目标条件的MDP。为了验证其有效性，我们实例化了一个由这些先验辅助的演员-评论家方法，称之为基础演员-评论家（FAC）。我们将我们的框架命名为基础强化学习（FRL），因为它完全依赖于具身基础先验来进行探索、学习和强化。FRL的好处有三个。(1)样本效率高。通过基础先验加速训练过程，减少样本使用量。

    Recently, people have shown that large-scale pre-training from internet-scale data is the key to building generalist models, as witnessed in NLP. To build embodied generalist agents, we and many other researchers hypothesize that such foundation prior is also an indispensable component. However, it is unclear what is the proper concrete form to represent those embodied foundation priors and how they should be used in the downstream task. In this paper, we propose an intuitive and effective set of embodied priors that consist of foundation policy, value, and success reward. The proposed priors are based on the goal-conditioned MDP. To verify their effectiveness, we instantiate an actor-critic method assisted by the priors, called Foundation Actor-Critic (FAC). We name our framework as Foundation Reinforcement Learning (FRL), since it completely relies on embodied foundation priors to explore, learn and reinforce. The benefits of FRL are threefold. (1) Sample efficient. With foundation p
    
[^85]: 基于改进的Aitchison-Aitken函数贝叶斯优化的组合爆炸决策树的多规则挖掘算法

    Multi-rules mining algorithm for combinatorially exploded decision trees with modified Aitchison-Aitken function-based Bayesian optimization. (arXiv:2310.02633v1 [cs.LG])

    [http://arxiv.org/abs/2310.02633](http://arxiv.org/abs/2310.02633)

    该论文提出了两种算法(MAABO-MT和GS-MRM)，分别在所有可能的树中高性能估计构建树，仅提取可靠且不相似的规则。

    

    决策树具有易于解释的优点，因为它们允许根据if-then规则对输入数据进行分类。然而，决策树是由一个算法构建的，该算法通过最少的规则实现清晰的分类，而无论数据中是否存在各种潜在规则，都只能提取最小的规则。确实存在使用随机选择的特征子集构建多棵决策树的方法。然而，可以构建的树的数量仍然在相同的数量级，因为特征子集的数量是一个组合性爆炸。此外，当构建多棵树时，会生成许多规则，其中有几个是不可靠和/或非常相似的。因此，我们提出了“MAABO-MT”和“GS-MRM”算法，它们分别在所有可能的树中高性能估计构建树，并仅提取可靠且不相似的规则。

    Decision trees offer the benefit of easy interpretation because they allow the classification of input data based on if--then rules. However, as decision trees are constructed by an algorithm that achieves clear classification with minimum necessary rules, the trees possess the drawback of extracting only minimum rules, even when various latent rules exist in data. Approaches that construct multiple trees using randomly selected feature subsets do exist. However, the number of trees that can be constructed remains at the same scale because the number of feature subsets is a combinatorial explosion. Additionally, when multiple trees are constructed, numerous rules are generated, of which several are untrustworthy and/or highly similar. Therefore, we propose "MAABO-MT" and "GS-MRM" algorithms that strategically construct trees with high estimation performance among all possible trees with small computational complexity and extract only reliable and non-similar rules, respectively. Experi
    
[^86]: 通过Koopman VAEs生成规则和非规则时间序列数据的生成建模

    Generative Modeling of Regular and Irregular Time Series Data via Koopman VAEs. (arXiv:2310.02619v1 [cs.LG])

    [http://arxiv.org/abs/2310.02619](http://arxiv.org/abs/2310.02619)

    本文介绍了一种基于Koopman VAEs的新型生成框架，可以用于生成规则和非规则时间序列数据，解决了GANs训练不稳定和模式崩溃的问题，并通过利用谱工具对线性映射的特征值施加约束，实现了领域知识的整合和对定性行为和稳定性的研究。

    

    生成真实的时间序列数据对于许多工程和科学应用非常重要。现有的工作使用生成对抗网络（GANs）来解决这个问题。然而，GANs在训练过程中常常不稳定，并且可能出现模式崩溃的问题。而变分自动编码器（VAEs）被认为对这些问题更具鲁棒性，但却很少被考虑用于时间序列生成。在这项工作中，我们引入了基于新型模型先验的生成框架Koopman VAE（KVAE），可以为规则和非规则训练数据进行优化。受Koopman理论的启发，我们使用线性映射来表示潜在条件先验动态。我们的方法增强了生成建模的两个期望特性：（i）通过利用谱工具对线性映射的特征值施加约束，可以实现领域知识的整合；（ii）研究定性行为和稳定性。

    Generating realistic time series data is important for many engineering and scientific applications. Existing work tackles this problem using generative adversarial networks (GANs). However, GANs are often unstable during training, and they can suffer from mode collapse. While variational autoencoders (VAEs) are known to be more robust to these issues, they are (surprisingly) less often considered for time series generation. In this work, we introduce Koopman VAE (KVAE), a new generative framework that is based on a novel design for the model prior, and that can be optimized for either regular and irregular training data. Inspired by Koopman theory, we represent the latent conditional prior dynamics using a linear map. Our approach enhances generative modeling with two desired features: (i) incorporating domain knowledge can be achieved by leverageing spectral tools that prescribe constraints on the eigenvalues of the linear map; and (ii) studying the qualitative behavior and stablity 
    
[^87]: 分析和改进基于OT的对抗网络

    Analyzing and Improving OT-based Adversarial Networks. (arXiv:2310.02611v1 [cs.LG])

    [http://arxiv.org/abs/2310.02611](http://arxiv.org/abs/2310.02611)

    本文分析和改进了基于OT的对抗网络，首先在一个统一的框架中统一了这些方法，然后通过全面分析展示了各组件在训练中的作用，最后提出了一个简单但新颖的方法以改进最优生成模型，该方法通过逐步调整生成分布逐渐使其与数据分布对齐

    

    最优输运（OT）问题旨在找到一种输运方案，它在最小化给定的成本函数的同时连接两个分布。OT理论已经广泛应用于生成模型。最初，OT距离被用作评估数据和生成分布之间的距离的一种度量。最近，OT传输映射在数据和先验分布之间被用作生成模型。这些基于OT的生成模型具有相似的对抗训练目标。在本文中，我们首先在一个统一的框架中统一了这些基于OT的对抗方法。然后，通过对这个统一框架的全面分析，我们阐明了每个组件在训练动力学中的作用。此外，我们提出了一个简单但新颖的方法，改进了先前表现最佳的基于OT的模型。直观地说，我们的方法通过逐步调整生成分布，逐渐使其与数据分布对齐，实现了渐进的改进。

    Optimal Transport (OT) problem aims to find a transport plan that bridges two distributions while minimizing a given cost function. OT theory has been widely utilized in generative modeling. In the beginning, OT distance has been used as a measure for assessing the distance between data and generated distributions. Recently, OT transport map between data and prior distributions has been utilized as a generative model. These OT-based generative models share a similar adversarial training objective. In this paper, we begin by unifying these OT-based adversarial methods within a single framework. Then, we elucidate the role of each component in training dynamics through a comprehensive analysis of this unified framework. Moreover, we suggest a simple but novel method that improves the previously best-performing OT-based model. Intuitively, our approach conducts a gradual refinement of the generated distribution, progressively aligning it with the data distribution. Our approach achieves a
    
[^88]: 动态图神经网络中学习邻接矩阵

    Learning adjacency matrix for dynamic graph neural network. (arXiv:2310.02606v1 [cs.LG])

    [http://arxiv.org/abs/2310.02606](http://arxiv.org/abs/2310.02606)

    该论文介绍了一种用于动态图神经网络的学习邻接矩阵的方法。通过引入一个特殊设计的编码器块来学习缺失的时空连接，将其丰富后的块邻接矩阵输入到图神经网络中，以捕捉网络的复杂时空拓扑。

    

    在最近的研究中，[1] 引入了使用块邻接矩阵（BA）来表示时空数据的概念。虽然他们的方法成功地串联了邻接矩阵，以封装单个图中的时空关系，但它形成了一个不连通的图。这个限制妨碍了图卷积网络（GCN）在属于不同时间步的节点之间进行消息传递的能力，因为没有时间链接存在。为了克服这个挑战，我们引入了一个专门设计用于学习这些缺失的时间链接的编码器块。编码器块处理BA并预测之前未连接的子图之间的连接，从而产生一个富化的时空块邻接矩阵（STBAM）。然后，将这个富化的矩阵输入到图神经网络（GNN）中，以捕捉网络的复杂时空拓扑。我们对基准数据集surgVisDom和C2D2进行评估，结果证明我们的方法稍高一些。

    In recent work, [1] introduced the concept of using a Block Adjacency Matrix (BA) for the representation of spatio-temporal data. While their method successfully concatenated adjacency matrices to encapsulate spatio-temporal relationships in a single graph, it formed a disconnected graph. This limitation hampered the ability of Graph Convolutional Networks (GCNs) to perform message passing across nodes belonging to different time steps, as no temporal links were present. To overcome this challenge, we introduce an encoder block specifically designed to learn these missing temporal links. The encoder block processes the BA and predicts connections between previously unconnected subgraphs, resulting in a Spatio-Temporal Block Adjacency Matrix (STBAM). This enriched matrix is then fed into a Graph Neural Network (GNN) to capture the complex spatio-temporal topology of the network. Our evaluations on benchmark datasets, surgVisDom and C2D2, demonstrate that our method, with slightly higher
    
[^89]: 用于电网拓扑优化的多智能体强化学习

    Multi-Agent Reinforcement Learning for Power Grid Topology Optimization. (arXiv:2310.02605v1 [cs.LG])

    [http://arxiv.org/abs/2310.02605](http://arxiv.org/abs/2310.02605)

    本文提出了一种用于电网拓扑优化的分层多智能体强化学习（MARL）框架，有效处理随着网络增长而扩大的大型行动空间。实验表明，该框架在性能上与单一智能体强化学习方法相当，并比较了不同的RL算法和不同的高阶智能体策略。

    

    近年来，面临着能源需求增加和风能、太阳能等不可预测可再生能源的挑战，操作电网成为一个问题。强化学习在管理这些网络中显示出潜力，通过总线和线路切换等拓扑操作，但对于随着网络增长而扩大的大型行动空间的高效处理至关重要。本文提出了一种针对这种扩展行动空间的分层多智能体强化学习（MARL）框架，利用电网固有的分层特性。实验结果表明，MARL框架与单一智能体强化学习方法在性能上具有竞争力。我们还比较了不同的RL算法和不同的高阶智能体策略。

    Recent challenges in operating power networks arise from increasing energy demands and unpredictable renewable sources like wind and solar. While reinforcement learning (RL) shows promise in managing these networks, through topological actions like bus and line switching, efficiently handling large action spaces as networks grow is crucial. This paper presents a hierarchical multi-agent reinforcement learning (MARL) framework tailored for these expansive action spaces, leveraging the power grid's inherent hierarchical nature. Experimental results indicate the MARL framework's competitive performance with single-agent RL methods. We also compare different RL algorithms for lower-level agents alongside different policies for higher-order agents.
    
[^90]: ViT-ReciproCAM: 不需要梯度和注意力的Vision Transformer的视觉解释方法

    ViT-ReciproCAM: Gradient and Attention-Free Visual Explanations for Vision Transformer. (arXiv:2310.02588v1 [cs.CV])

    [http://arxiv.org/abs/2310.02588](http://arxiv.org/abs/2310.02588)

    本文提出了一种新的方法，名为ViT-ReciproCAM，用于解释Vision Transformer中的预测过程和调试预测错误。该方法不依赖梯度和注意力矩阵，并使用令牌遮罩和新的层输出来利用激活的令牌与目标类别的网络预测之间的关联。

    

    本文提出了一种新的方法来解决Vision Transformers (ViT)在理解预测过程和调试预测错误方面的挑战。ViT在各种计算机视觉任务（如图像分类和目标检测）中表现出了卓越的性能。虽然对于卷积神经网络（CNN）已经广泛研究了一些可视化解释技术，如CAM、Grad-CAM、Score-CAM和Recipro-CAM，但在ViT上的研究仍然有限。当前ViT的最新解决方案依赖于类不可知的Attention-Rollout和Relevance技术。本文提出了一种新的不依赖梯度的ViT视觉解释方法，称为ViT-ReciproCAM，它不需要注意力矩阵和梯度信息。ViT-ReciproCAM利用令牌遮罩和从目标层的输入产生的新层输出来利用激活的令牌与目标类别的网络预测之间的关联。

    This paper presents a novel approach to address the challenges of understanding the prediction process and debugging prediction errors in Vision Transformers (ViT), which have demonstrated superior performance in various computer vision tasks such as image classification and object detection. While several visual explainability techniques, such as CAM, Grad-CAM, Score-CAM, and Recipro-CAM, have been extensively researched for Convolutional Neural Networks (CNNs), limited research has been conducted on ViT. Current state-of-the-art solutions for ViT rely on class agnostic Attention-Rollout and Relevance techniques. In this work, we propose a new gradient-free visual explanation method for ViT, called ViT-ReciproCAM, which does not require attention matrix and gradient information. ViT-ReciproCAM utilizes token masking and generated new layer outputs from the target layer's input to exploit the correlation between activated tokens and network predictions for target classes. Our proposed 
    
[^91]: 机器学习在高级热致动器中的精确位置控制和热调节的应用

    Machine Learning-Enabled Precision Position Control and Thermal Regulation in Advanced Thermal Actuators. (arXiv:2310.02583v1 [cs.RO])

    [http://arxiv.org/abs/2310.02583](http://arxiv.org/abs/2310.02583)

    本论文介绍了一种基于机器学习的恒功率开环控制器，在没有外部传感器的情况下，实现了对尼龙人工肌肉的精确位置控制。通过构建神经网络，将期望位移转化为所需功率，神经控制器在经过精心训练后能够适应各种类型的热人工肌肉。

    

    尼龙人工肌肉具有独特的特性，能量密度几乎是人类肌肉的100倍，功率密度为5.3 kW/kg，类似于喷气发动机的输出，因此特别适用于机器人应用。然而，集成传感器和控制器的必要性对其实际使用造成了限制。本研究提出了基于机器学习的恒功率开环控制器。我们展示了在没有外部传感器的情况下，我们可以控制尼龙人工肌肉的位置。为此，我们使用集成编码器式前馈神经网络构建了一个从期望位移轨迹到所需功率的映射。神经控制器在基于物理的去噪数据集上经过精心训练，并可以进行微调以适应各种类型的热人工肌肉，无论是否存在滞后现象。

    With their unique combination of characteristics - an energy density almost 100 times that of human muscle, and a power density of 5.3 kW/kg, similar to a jet engine's output - Nylon artificial muscles stand out as particularly apt for robotics applications. However, the necessity of integrating sensors and controllers poses a limitation to their practical usage. Here we report a constant power open-loop controller based on machine learning. We show that we can control the position of a nylon artificial muscle without external sensors. To this end, we construct a mapping from a desired displacement trajectory to a required power using an ensemble encoder-style feed-forward neural network. The neural controller is carefully trained on a physics-based denoised dataset and can be fine-tuned to accommodate various types of thermal artificial muscles, irrespective of the presence or absence of hysteresis.
    
[^92]: 在强化学习中的鲁棒策略评估的在线估计和推断

    Online Estimation and Inference for Robust Policy Evaluation in Reinforcement Learning. (arXiv:2310.02581v1 [stat.ML])

    [http://arxiv.org/abs/2310.02581](http://arxiv.org/abs/2310.02581)

    该论文提出了一种针对鲁棒策略评估的在线估计和推断方法，在解决异常值污染和重尾奖励的问题方面引入了鲁棒统计学的概念。此外，还提出了一种完全在线的统计推断过程，并建立了估计量的极限分布。

    

    最近，强化学习在现代统计学中备受关注，策略评估是其中一个关键组成部分。与传统机器学习文献上对该主题的研究不同，我们的工作强调使用强化学习算法计算的参数估计的统计推断。尽管大多数现有分析假设随机奖励遵循标准分布，限制了它们的适用性，但我们在统一框架中同时解决了异常值污染和重尾奖励的问题，从而拥抱了鲁棒统计学在强化学习中的概念。在本文中，我们开发了一种在线鲁棒策略评估过程，并根据其Bahadur表示建立了我们估计量的极限分布。此外，我们还开发了一种完全在线的过程，以高效地进行基于渐近分布的统计推断。这篇论文填补了强化学习中鲁棒统计学和统计推断之间的差距。

    Recently, reinforcement learning has gained prominence in modern statistics, with policy evaluation being a key component. Unlike traditional machine learning literature on this topic, our work places emphasis on statistical inference for the parameter estimates computed using reinforcement learning algorithms. While most existing analyses assume random rewards to follow standard distributions, limiting their applicability, we embrace the concept of robust statistics in reinforcement learning by simultaneously addressing issues of outlier contamination and heavy-tailed rewards within a unified framework. In this paper, we develop an online robust policy evaluation procedure, and establish the limiting distribution of our estimator, based on its Bahadur representation. Furthermore, we develop a fully-online procedure to efficiently conduct statistical inference based on the asymptotic distribution. This paper bridges the gap between robust statistics and statistical inference in reinfor
    
[^93]: 关于图神经网络中表达位置编码的稳定性

    On the Stability of Expressive Positional Encodings for Graph Neural Networks. (arXiv:2310.02579v1 [cs.LG])

    [http://arxiv.org/abs/2310.02579](http://arxiv.org/abs/2310.02579)

    本研究针对图神经网络中使用拉普拉斯特征向量作为位置编码面临的非唯一性和不稳定性问题，提出了稳定且表达丰富的位置编码方法（SPE），该方法通过利用特征值对特征空间进行"软分割"，在未见过的图结构上表现出良好的泛化能力。

    

    设计有效的图位置编码对构建强大的图转换器和增强消息传递图神经网络非常关键。尽管广泛使用，使用拉普拉斯特征向量作为位置编码面临两个根本性挑战：（1）\emph{非唯一性}：同一拉普拉斯矩阵存在许多不同的特征分解，以及（2）\emph{不稳定性}：对拉普拉斯矩阵的微小扰动可能导致完全不同的特征空间，从而导致位置编码的不可预测性变化。尽管有很多尝试解决非唯一性的方法，但大多数方法忽视了稳定性，导致在未见过的图结构上表现不佳。我们发现，不稳定性的原因是特征空间的"硬分割"。因此，我们引入了稳定且表达丰富的位置编码（SPE），这是一种用于处理特征向量的架构，利用特征值将特征空间进行"软分割"。SPE是首个（1）可证明稳定的架构，以及（2）普适地提升图结构泛化性能的架构。

    Designing effective positional encodings for graphs is key to building powerful graph transformers and enhancing message-passing graph neural networks. Although widespread, using Laplacian eigenvectors as positional encodings faces two fundamental challenges: (1) \emph{Non-uniqueness}: there are many different eigendecompositions of the same Laplacian, and (2) \emph{Instability}: small perturbations to the Laplacian could result in completely different eigenspaces, leading to unpredictable changes in positional encoding.  Despite many attempts to address non-uniqueness, most methods overlook stability, leading to poor generalization on unseen graph structures. We identify the cause of instability to be a "hard partition" of eigenspaces. Hence, we introduce Stable and Expressive Positional Encodings (SPE), an architecture for processing eigenvectors that uses eigenvalues to "softly partition" eigenspaces. SPE is the first architecture that is (1) provably stable, and (2) universally exp
    
[^94]: AdaMerging: 适应性模型合并用于多任务学习

    AdaMerging: Adaptive Model Merging for Multi-Task Learning. (arXiv:2310.02575v1 [cs.LG])

    [http://arxiv.org/abs/2310.02575](http://arxiv.org/abs/2310.02575)

    AdaMerging通过自适应学习模型合并的系数，以更有效地合并预训练模型来解决多任务学习中存在的性能下降问题。

    

    多任务学习旨在使模型能够同时处理多个任务。最近的一项发展被称为任务算术，揭示了几个针对不同任务进行微调的模型可以直接合并成一个单一模型，以执行多任务学习，而无需使用初始训练数据进行重新训练。然而，这种直接添加模型往往会导致合并模型的整体性能显著下降。这种下降是由于多个任务之间存在潜在的冲突和复杂的相关性所致。因此，如何更有效地合并预训练模型而不使用其原始训练数据成为一个挑战。本文介绍了一种创新技术，称为自适应模型合并（AdaMerging）。该方法旨在自动学习模型合并的系数，可以是逐任务或逐层的方式，而不依赖于原始训练数据。

    Multi-task learning (MTL) aims to empower a model to tackle multiple tasks simultaneously. A recent development known as task arithmetic has revealed that several models, each fine-tuned for distinct tasks, can be directly merged into a single model to execute MTL without necessitating a retraining process using the initial training data. Nevertheless, this direct addition of models often leads to a significant deterioration in the overall performance of the merged model. This decline occurs due to potential conflicts and intricate correlations among the multiple tasks. Consequently, the challenge emerges of how to merge pre-trained models more effectively without using their original training data. This paper introduces an innovative technique called Adaptive Model Merging (AdaMerging). This approach aims to autonomously learn the coefficients for model merging, either in a task-wise or layer-wise manner, without relying on the original training data. Specifically, our AdaMerging meth
    
[^95]: 使用教师解释改进知识蒸馏

    Improving Knowledge Distillation with Teacher's Explanation. (arXiv:2310.02572v1 [cs.LG])

    [http://arxiv.org/abs/2310.02572](http://arxiv.org/abs/2310.02572)

    本论文提出了一种新颖的知识解释蒸馏（KED）框架，允许学生从教师的解释中学习，并扩展了KED的应用范围，以提高卷积神经网络的性能和处理有限训练数据的能力。

    

    知识蒸馏通过一个强大的教师来提高低复杂度的学生模型的性能。在知识蒸馏中，教师是一个黑盒模型，只通过其预测来传授知识给学生。这限制了传输知识的数量。在这项工作中，我们引入了一种新颖的知识解释蒸馏（KED）框架，它允许学生不仅从教师的预测中学习，还可以从教师的解释中学习。我们提出了一类能够解释特征组的超特征教师，以及相应的学生模型。我们还提出了一种构建超特征的方法。然后，我们扩展了KED，以减少卷积神经网络的复杂性，以允许与隐藏表示蒸馏方法的增强，以及使用嵌合集合处理有限的训练数据。我们在多个数据集上的实验证明，KED学生的性能可以显著超越传统的知识蒸馏方法。

    Knowledge distillation (KD) improves the performance of a low-complexity student model with the help of a more powerful teacher. The teacher in KD is a black-box model, imparting knowledge to the student only through its predictions. This limits the amount of transferred knowledge. In this work, we introduce a novel Knowledge Explaining Distillation (KED) framework, which allows the student to learn not only from the teacher's predictions but also from the teacher's explanations. We propose a class of superfeature-explaining teachers that provide explanation over groups of features, along with the corresponding student model. We also present a method for constructing the superfeatures. We then extend KED to reduce complexity in convolutional neural networks, to allow augmentation with hidden-representation distillation methods, and to work with a limited amount of training data using chimeric sets. Our experiments over a variety of datasets show that KED students can substantially outp
    
[^96]: 为了某事而站立，否则就会为一切垮掉：使用立场感知的图神经网络预测谣言传播

    Stand for Something or Fall for Everything: Predict Misinformation Spread with Stance-Aware Graph Neural Networks. (arXiv:2310.02568v1 [cs.SI])

    [http://arxiv.org/abs/2310.02568](http://arxiv.org/abs/2310.02568)

    使用立场感知的图神经网络（stance-aware GNN）预测谣言传播。与没有用户立场的GNN相比，该模型在真实数据集上的表现优于32.65%的基准模型。注意权重表明用户的反对立场对邻居行为的影响更大，可以作为社会纠正措施阻止谣言传播。

    

    尽管社交媒体平台上流行的谣言传播已成为迫切的挑战，但现有的平台干预措施在遏制其传播方面显示出有限的成功。在本研究中，我们提出了一种立场感知的图神经网络（stance-aware GNN），利用用户的立场主动预测谣言传播。由于不同用户的立场可以形成独特的回声室，我们在立场感知的GNN中定制了四个信息传递路径，而可训练的注意权重通过突出显示每个结构的重要性来提供可解释性。在一个真实数据集上进行评估，立场感知的GNN的表现优于基准模型32.65％，并且超过了没有用户立场的先进GNN 4.69％以上。此外，注意权重表明，用户的反对立场对邻居行为的影响高于支持立场，这起到了社会纠正作用，阻止了谣言的传播。总体而言，我们的研究提供了一种有效的预测模型。

    Although pervasive spread of misinformation on social media platforms has become a pressing challenge, existing platform interventions have shown limited success in curbing its dissemination. In this study, we propose a stance-aware graph neural network (stance-aware GNN) that leverages users' stances to proactively predict misinformation spread. As different user stances can form unique echo chambers, we customize four information passing paths in stance-aware GNN, while the trainable attention weights provide explainability by highlighting each structure's importance. Evaluated on a real-world dataset, stance-aware GNN outperforms benchmarks by 32.65% and exceeds advanced GNNs without user stance by over 4.69%. Furthermore, the attention weights indicate that users' opposition stances have a higher impact on their neighbors' behaviors than supportive ones, which function as social correction to halt misinformation propagation. Overall, our study provides an effective predictive model
    
[^97]: 使用大型语言模型改进自动VQA评估

    Improving Automatic VQA Evaluation Using Large Language Models. (arXiv:2310.02567v1 [cs.CV])

    [http://arxiv.org/abs/2310.02567](http://arxiv.org/abs/2310.02567)

    提出使用大型语言模型改进自动视觉问答（VQA）评估的方法，将VQA评估格式化为回答评分任务，通过指令调整大型语言模型在准确度上评分候选答案，证明该方法与人类判断相关性优于现有度量方法。

    

    在提出视觉问答（VQA）任务8年后，准确率仍然是自动评估的主要指标。在IID评估设置中，VQA准确度一直很有效。然而，我们的社区正在转向开放式生成模型和OOD评估。在这种新的范式中，现有的VQA准确度指标过于严格，低估了VQA系统的性能。因此，有必要开发更强大的自动VQA度量，作为人类判断的代理。在这项工作中，我们提出利用指令调整大型语言模型（LLM）的上下文学习能力来构建更好的VQA度量。我们将VQA评估格式化为一个回答评分任务，即指令调整的大型语言模型被指示根据一组参考答案评分候选答案的准确性。我们证明所提出的度量与人类判断相关性优于现有度量在几个VQA模型和基准测试中。

    8 years after the visual question answering (VQA) task was proposed, accuracy remains the primary metric for automatic evaluation. VQA Accuracy has been effective so far in the IID evaluation setting. However, our community is undergoing a shift towards open-ended generative models and OOD evaluation. In this new paradigm, the existing VQA Accuracy metric is overly stringent and underestimates the performance of VQA systems. Thus, there is a need to develop more robust automatic VQA metrics that serve as a proxy for human judgment. In this work, we propose to leverage the in-context learning capabilities of instruction-tuned large language models (LLMs) to build a better VQA metric. We formulate VQA evaluation as an answer-rating task where the LLM is instructed to score the accuracy of a candidate answer given a set of reference answers. We demonstrate the proposed metric better correlates with human judgment compared to existing metrics across several VQA models and benchmarks. We ho
    
[^98]: 实用的、私密的合作价值保证

    Practical, Private Assurance of the Value of Collaboration. (arXiv:2310.02563v1 [cs.CR])

    [http://arxiv.org/abs/2310.02563](http://arxiv.org/abs/2310.02563)

    该论文研究了两方在数据集上合作前如何保证合作的价值。通过构建基于全同态加密方案和标签差分隐私的交互式协议，该研究提供了一个实用的、私密的解决方案。最终的结果是确保合作前双方的模型和数据集不会被透露。

    

    两个方向希望在数据集上进行合作。然而，在彼此透露数据集之前，双方希望能够得到合作将是富有成果的保证。我们从机器学习的角度来看待这个问题，其中一方被承诺通过合并来自另一方的数据来改进其预测模型。只有当更新的模型显示出准确性的提升时，双方才希望进一步合作。在确定这一点之前，双方不希望透露他们的模型和数据集。在这项工作中，我们基于Torus上的全同态加密方案（TFHE）和标签差分隐私构建了一个交互式协议，其中底层的机器学习模型是一个神经网络。标签差分隐私用于确保计算不完全在加密领域进行，这对神经网络训练来说是一个重要瓶颈。

    Two parties wish to collaborate on their datasets. However, before they reveal their datasets to each other, the parties want to have the guarantee that the collaboration would be fruitful. We look at this problem from the point of view of machine learning, where one party is promised an improvement on its prediction model by incorporating data from the other party. The parties would only wish to collaborate further if the updated model shows an improvement in accuracy. Before this is ascertained, the two parties would not want to disclose their models and datasets. In this work, we construct an interactive protocol for this problem based on the fully homomorphic encryption scheme over the Torus (TFHE) and label differential privacy, where the underlying machine learning model is a neural network. Label differential privacy is used to ensure that computations are not done entirely in the encrypted domain, which is a significant bottleneck for neural network training according to the cu
    
[^99]: 半联邦学习：混合学习框架的收敛分析与优化

    Semi-Federated Learning: Convergence Analysis and Optimization of A Hybrid Learning Framework. (arXiv:2310.02559v1 [cs.IT])

    [http://arxiv.org/abs/2310.02559](http://arxiv.org/abs/2310.02559)

    该论文提出了一种半联邦学习（SemiFL）范式，以在基站和设备之间进行集中式学习（CL）和无线联邦学习（FL）的混合实现。通过集成空中计算和非正交多址接入传输，提高了通信效率，并进行了收敛分析和优化。

    

    在基站（BS）的组织下，无线联邦学习（FL）实现了多个设备之间的协同模型训练。然而，BS仅负责在训练过程中聚合本地更新，这导致了BS上计算资源的浪费。为了解决这个问题，我们提出了一种半联邦学习（SemiFL）范式，以充分利用BS和设备的计算能力，进行集中式学习（CL）和FL的混合实现。具体来说，每个设备将本地梯度和数据样本发送给BS，以训练共享的全局模型。为了提高通信效率，我们设计了一种新的收发器结构，将空中计算集成到聚合和非正交多址接入传输中。为了深入了解，我们通过推导SemiFL的闭式最优性差距进行了收敛分析，并将结果扩展到了两种额外的情况。

    Under the organization of the base station (BS), wireless federated learning (FL) enables collaborative model training among multiple devices. However, the BS is merely responsible for aggregating local updates during the training process, which incurs a waste of the computational resource at the BS. To tackle this issue, we propose a semi-federated learning (SemiFL) paradigm to leverage the computing capabilities of both the BS and devices for a hybrid implementation of centralized learning (CL) and FL. Specifically, each device sends both local gradients and data samples to the BS for training a shared global model. To improve communication efficiency over the same time-frequency resources, we integrate over-the-air computation for aggregation and non-orthogonal multiple access for transmission by designing a novel transceiver structure. To gain deep insights, we conduct convergence analysis by deriving a closed-form optimality gap for SemiFL and extend the result to two extra cases.
    
[^100]: 扩散模型中的泛化性质源于几何自适应的谐波表示

    Generalization in diffusion models arises from geometry-adaptive harmonic representation. (arXiv:2310.02557v1 [cs.CV])

    [http://arxiv.org/abs/2310.02557](http://arxiv.org/abs/2310.02557)

    通过分析基于分数的反向扩散算法生成的高质量样本的研究结果，我们发现尽管存在维度灾难，但为了降噪而训练的深度神经网络可以学习到高维密度。此外，我们展示了在训练集的非重叠子集上训练的网络可以学习到相同的密度，从而证明了DNN架构和训练算法中的归纳偏差与数据分布的一致性。

    

    使用基于分数的反向扩散算法生成的高质量样本提供了证据，表明尽管存在维度灾难，为了降噪而训练的深度神经网络（DNN）可以学习高维密度。然而，关于训练集记忆化的最新报告引发了一个问题，即这些网络是否学习了数据的“真实”连续密度。我们在这里展示，训练在数据集的非重叠子集上的两个降噪DNN学习的几乎是相同的分数函数，从而学习了相同的密度，且仅需很少的训练图像。这种强大的泛化性证明了DNN架构和/或训练算法中的有力归纳偏差与数据分布的特性之间的一致性。我们分析了这些结果，展示了降噪器在适应于底层图像的基础上执行收缩操作。对这些基矢的检查揭示了沿轮廓和均匀图像区域的振荡谐波结构。

    High-quality samples generated with score-based reverse diffusion algorithms provide evidence that deep neural networks (DNN) trained for denoising can learn high-dimensional densities, despite the curse of dimensionality. However, recent reports of memorization of the training set raise the question of whether these networks are learning the "true" continuous density of the data. Here, we show that two denoising DNNs trained on non-overlapping subsets of a dataset learn nearly the same score function, and thus the same density, with a surprisingly small number of training images. This strong generalization demonstrates an alignment of powerful inductive biases in the DNN architecture and/or training algorithm with properties of the data distribution. We analyze these, demonstrating that the denoiser performs a shrinkage operation in a basis adapted to the underlying image. Examination of these bases reveals oscillating harmonic structures along contours and in homogeneous image region
    
[^101]: zkFL: 基于零知识证明的联邦学习梯度聚合

    zkFL: Zero-Knowledge Proof-based Gradient Aggregation for Federated Learning. (arXiv:2310.02554v1 [cs.AI])

    [http://arxiv.org/abs/2310.02554](http://arxiv.org/abs/2310.02554)

    zkFL是一种基于零知识证明的联邦学习梯度聚合方法，通过提供每轮的证明来解决协调者恶意行为的问题。

    

    联邦学习是一种机器学习范式，使多个分散的客户端在中央协调者的组织下共同训练一个模型。传统的联邦学习解决方案依赖于对中央协调者的信任，它以公平诚实的方式形成客户端的群体。然而，在现实中，恶意的协调者可能会放弃并替换客户端的训练模型，或者发动虚假客户端的肆意攻击。这种恶意行为让协调者在联邦学习环境中拥有更多控制客户端和决定最终训练结果的权力。本文介绍了zkFL，它利用零知识证明(ZKPs)来解决训练模型聚合过程中的恶意协调者问题。为了保证正确的聚合结果，协调者需要每轮提供一个证明。这个证明可以向客户端证明协调者忠实执行预期行为。为了进一步保护客户端隐私和数据安全，我们还引入了差分隐私机制，并对zkFL进行了实验评估。

    Federated Learning (FL) is a machine learning paradigm, which enables multiple and decentralized clients to collaboratively train a model under the orchestration of a central aggregator. Traditional FL solutions rely on the trust assumption of the centralized aggregator, which forms cohorts of clients in a fair and honest manner. However, a malicious aggregator, in reality, could abandon and replace the client's training models, or launch Sybil attacks to insert fake clients. Such malicious behaviors give the aggregator more power to control clients in the FL setting and determine the final training results. In this work, we introduce zkFL, which leverages zero-knowledge proofs (ZKPs) to tackle the issue of a malicious aggregator during the training model aggregation process. To guarantee the correct aggregation results, the aggregator needs to provide a proof per round. The proof can demonstrate to the clients that the aggregator executes the intended behavior faithfully. To further r
    
[^102]: 使用知识共蒸合的异构联邦学习

    Heterogeneous Federated Learning Using Knowledge Codistillation. (arXiv:2310.02549v1 [cs.LG])

    [http://arxiv.org/abs/2310.02549](http://arxiv.org/abs/2310.02549)

    使用知识共蒸合的异构联邦学习方法通过在整个池子和容量较高的部分客户端上训练不同大小的模型，实现了双向信息交换和领域转移，改进了联邦平均化算法在图像分类和语言建模任务上的性能。

    

    联邦平均化及其建立在其上的许多联邦学习算法变种存在一个限制：所有客户端必须共享相同的模型架构。这导致许多客户端上未使用的建模能力，从而限制了模型性能。为解决这个问题，我们提出了一种方法，该方法涉及在整个池内训练一个小模型和在具有更高容量的一部分客户端上训练一个更大模型。模型通过知识共蒸合在未共享参数的服务器上进行双向信息交换，利用一个无标签数据集。我们提出了两种改进联邦平均化在图像分类和语言模型任务上的方法。我们展示了即使只有领域外或有限领域内的蒸馏数据可用，这种技术也可以很有用。此外，双向知识共蒸合在不同池子中引入领域转移时允许模型之间的领域转移。

    Federated Averaging, and many federated learning algorithm variants which build upon it, have a limitation: all clients must share the same model architecture. This results in unused modeling capacity on many clients, which limits model performance. To address this issue, we propose a method that involves training a small model on the entire pool and a larger model on a subset of clients with higher capacity. The models exchange information bidirectionally via knowledge distillation, utilizing an unlabeled dataset on a server without sharing parameters. We present two variants of our method, which improve upon federated averaging on image classification and language modeling tasks. We show this technique can be useful even if only out-of-domain or limited in-domain distillation data is available. Additionally, the bi-directional knowledge distillation allows for domain transfer between the models when different pool populations introduce domain shift.
    
[^103]: 变系数泊松方程中物理信息神经网络的精确和软边界条件

    Exact and soft boundary conditions in Physics-Informed Neural Networks for the Variable Coefficient Poisson equation. (arXiv:2310.02548v1 [cs.LG])

    [http://arxiv.org/abs/2310.02548](http://arxiv.org/abs/2310.02548)

    本研究研究了在物理信息神经网络(PINN)中应用软损失和精确距离函数的边界条件时的差异，并提供了有关如何实现这些PINN的资源和工具。

    

    边界条件是每个物理信息神经网络（PINN）中的关键组成部分。通过在域边界上定义偏微分方程（PDE）的解，边界条件约束了PINN试图逼近的基本边界值问题（BVP）。如果没有它们，唯一的PDE解可能不存在，使用PINN找到近似解将是一项具有挑战性的任务，甚至是不可能的任务。本研究考察了在PINN中应用基于软损失和基于精确距离函数的边界条件强制方法的差异。著名的变系数泊松方程被用作本工作中所有PINN模型的目标PDE。除了比较边界条件强制方法，本工作的目标还是提供有关如何实现这些PINN的资源。为此，通过此评论一并发布了使用Tensorflow后端的Keras模型以及带有代码示例和逐步说明如何构建软/精确边界条件PINN的Python笔记本。

    Boundary conditions (BCs) are a key component in every Physics-Informed Neural Network (PINN). By defining the solution to partial differential equations (PDEs) along domain boundaries, BCs constrain the underlying boundary value problem (BVP) that a PINN tries to approximate. Without them, unique PDE solutions may not exist and finding approximations with PINNs would be a challenging, if not impossible task. This study examines how soft loss-based and exact distance function-based BC imposition approaches differ when applied in PINNs. The well known variable coefficient Poisson equation serves as the target PDE for all PINN models trained in this work. Besides comparing BC imposition approaches, the goal of this work is to also provide resources on how to implement these PINNs in practice. To this end, Keras models with Tensorflow backend as well as a Python notebook with code examples and step-by-step explanations on how to build soft/exact BC PINNs are published alongside this revie
    
[^104]: 基于模式的蛋白质序列和结构的联合设计

    Joint Design of Protein Sequence and Structure based on Motifs. (arXiv:2310.02546v1 [cs.LG])

    [http://arxiv.org/abs/2310.02546](http://arxiv.org/abs/2310.02546)

    本文提出了一种GeoPro方法，用于联合设计蛋白质的骨架结构和序列。实验证明，GeoPro在多个指标上优于其他方法，并发现了新型β-内酰胺酶和肌红蛋白。

    

    在生物学和化学中，设计具有所需功能的新型蛋白质至关重要。然而，大多数现有的研究都集中在蛋白质序列设计上，而忽视了蛋白质序列和结构的联合设计。本文提出了一种名为GeoPro的方法，用于联合设计蛋白质的骨架结构和序列。我们的动机是蛋白质序列和骨架结构相互约束，因此联合设计能够避免非折叠和错误折叠，并产生更多具有所需功能的候选蛋白质。为此，GeoPro利用一个三维骨架结构的等变编码器和一个由三维几何引导的蛋白质序列解码器。在包括β-内酰胺酶和肌红蛋白在内的两个具有生物学意义的金属蛋白质数据集上的实验结果表明，我们提出的GeoPro在大多数指标上优于几个强基线方法。值得注意的是，我们的方法发现了一些之前未发现的新型β-内酰胺酶和肌红蛋白。

    Designing novel proteins with desired functions is crucial in biology and chemistry. However, most existing work focus on protein sequence design, leaving protein sequence and structure co-design underexplored. In this paper, we propose GeoPro, a method to design protein backbone structure and sequence jointly. Our motivation is that protein sequence and its backbone structure constrain each other, and thus joint design of both can not only avoid nonfolding and misfolding but also produce more diverse candidates with desired functions. To this end, GeoPro is powered by an equivariant encoder for three-dimensional (3D) backbone structure and a protein sequence decoder guided by 3D geometry. Experimental results on two biologically significant metalloprotein datasets, including $\beta$-lactamases and myoglobins, show that our proposed GeoPro outperforms several strong baselines on most metrics. Remarkably, our method discovers novel $\beta$-lactamases and myoglobins which are not present
    
[^105]: 具有图信息的可证明张量补全

    Provable Tensor Completion with Graph Information. (arXiv:2310.02543v1 [cs.LG])

    [http://arxiv.org/abs/2310.02543](http://arxiv.org/abs/2310.02543)

    本文提出了一个创新框架，系统地解决了具有图信息的动态图正则化张量补全问题，建立了严格的数学表示，并推导出了新的图导向补全模型。

    

    图被广泛应用作为变量之间相互关系的有效侧面信息，用于各种矩阵/张量恢复相关应用中的准确数据恢复。本文研究了具有图信息的张量补全问题。目前关于图正则化的张量补全的研究更倾向于特定任务，缺乏普适性和系统化的方法。此外，缺乏保证性能的恢复理论。此外，这些方法忽视了图的动态特性，在张量相关场景中将其视为类似矩阵的静态对象，即使图可能在时间上具有动态性。为了应对这些挑战，在本文中我们介绍了一个创新框架，系统地建立了解决动态图正则化张量补全问题的新模型、理论和算法。对于模型，我们建立了动态图的严格数学表示，基于该表示我们推导出了一个新的针对张量的图导向的补全模型。

    Graphs, depicting the interrelations between variables, has been widely used as effective side information for accurate data recovery in various matrix/tensor recovery related applications. In this paper, we study the tensor completion problem with graph information. Current research on graph-regularized tensor completion tends to be task-specific, lacking generality and systematic approaches. Furthermore, a recovery theory to ensure performance remains absent. Moreover, these approaches overlook the dynamic aspects of graphs, treating them as static akin to matrices, even though graphs could exhibit dynamism in tensor-related scenarios. To confront these challenges, we introduce a pioneering framework in this paper that systematically formulates a novel model, theory, and algorithm for solving the dynamic graph regularized tensor completion problem. For the model, we establish a rigorous mathematical representation of the dynamic graph, based on which we derive a new tensor-oriented g
    
[^106]: 针对XOR集群数据中的ReLU网络的良性过拟合和理解

    Benign Overfitting and Grokking in ReLU Networks for XOR Cluster Data. (arXiv:2310.02541v1 [cs.LG])

    [http://arxiv.org/abs/2310.02541](http://arxiv.org/abs/2310.02541)

    通过梯度下降训练的ReLU网络在XOR集群数据上会产生良性过拟合和理解现象，即在训练阶段实现噪声标签的完美拟合但在测试阶段表现随机，在后续阶段可以实现近乎最优的泛化能力。

    

    通过梯度下降(GD)训练的神经网络展现了许多令人惊讶的泛化行为。首先，它们可以对噪声训练数据实现完美拟合，并且仍然能够近乎最优地进行泛化，表明过拟合有时可能是良性的。其次，在训练的早期阶段，它们可能会经历一段经典且有害的过拟合期，即在训练数据上实现完美拟合但在测试数据上表现随机，随后过渡到近乎最优的泛化行为（即“理解”）。在这项工作中，我们证明了这两个现象在通过GD对XOR集群数据上的两层ReLU网络进行训练时确实会出现，其中训练标签的一部分会被翻转。我们发现在GD的第一步之后，神经网络能够实现100%的训练准确度，在训练数据中完美拟合噪声标签，但在测试上表现接近随机。在随后的训练步骤中，网络能够实现近乎最优的测试准确度，同时仍然拟合随机标签。

    Neural networks trained by gradient descent (GD) have exhibited a number of surprising generalization behaviors. First, they can achieve a perfect fit to noisy training data and still generalize near-optimally, showing that overfitting can sometimes be benign. Second, they can undergo a period of classical, harmful overfitting -- achieving a perfect fit to training data with near-random performance on test data -- before transitioning ("grokking") to near-optimal generalization later in training. In this work, we show that both of these phenomena provably occur in two-layer ReLU networks trained by GD on XOR cluster data where a constant fraction of the training labels are flipped. In this setting, we show that after the first step of GD, the network achieves 100% training accuracy, perfectly fitting the noisy labels in the training data, but achieves near-random test accuracy. At a later training step, the network achieves near-optimal test accuracy while still fitting the random labe
    
[^107]: Auto-FP:自动化特征预处理在表格数据上的实验研究

    Auto-FP: An Experimental Study of Automated Feature Preprocessing for Tabular Data. (arXiv:2310.02540v1 [cs.LG])

    [http://arxiv.org/abs/2310.02540](http://arxiv.org/abs/2310.02540)

    本文研究了如何自动化表格数据的特征预处理（Auto-FP），将其建模为超参数优化或神经网络架构搜索问题，并扩展了各种算法来解决Auto-FP问题。

    

    传统的机器学习模型，如线性模型和基于树的模型，在工业中被广泛使用。这些模型对数据分布敏感，因此特征预处理是确保模型质量良好的关键步骤。手动构建特征预处理流程很具挑战性，因为数据科学家需要在选择哪些预处理器以及以什么顺序组合它们方面作出困难的决策。在本文中，我们研究了如何自动化表格数据的特征预处理（Auto-FP）。由于搜索空间较大，暴力解决方案代价太高。为了解决这个挑战，我们有趣地观察到Auto-FP可以被建模为超参数优化（HPO）或神经网络架构搜索（NAS）问题。这个观察使我们能够扩展各种HPO和NAS算法来解决Auto-FP问题。我们进行了全面的评估和分析，共进行了15个...

    Classical machine learning models, such as linear models and tree-based models, are widely used in industry. These models are sensitive to data distribution, thus feature preprocessing, which transforms features from one distribution to another, is a crucial step to ensure good model quality. Manually constructing a feature preprocessing pipeline is challenging because data scientists need to make difficult decisions about which preprocessors to select and in which order to compose them. In this paper, we study how to automate feature preprocessing (Auto-FP) for tabular data. Due to the large search space, a brute-force solution is prohibitively expensive. To address this challenge, we interestingly observe that Auto-FP can be modelled as either a hyperparameter optimization (HPO) or a neural architecture search (NAS) problem. This observation enables us to extend a variety of HPO and NAS algorithms to solve the Auto-FP problem. We conduct a comprehensive evaluation and analysis of 15 
    
[^108]: 量化和减轻标签错误对模型差异度量的影响

    Quantifying and mitigating the impact of label errors on model disparity metrics. (arXiv:2310.02533v1 [cs.LG])

    [http://arxiv.org/abs/2310.02533](http://arxiv.org/abs/2310.02533)

    本研究量化和减轻了标签错误对模型差异度量的影响，并且提出了一种估计训练输入标签对模型差异度量影响的方法，有效地改进了现有方法。

    

    通过人工注释获取的标签错误会对模型的性能产生负面影响。现有方法提出了减轻标签错误对模型下游准确性影响的方法，但对模型的差异度量的影响仍知之甚少。本文研究了标签错误对模型差异度量的影响。我们以实证方式表征了不同水平的标签错误对这些差异度量的影响，包括训练数据和测试数据中的标签错误。我们发现群体校准和其他度量对训练时和测试时的标签错误非常敏感，尤其对于少数群体。这种差异效应甚至适用于使用噪声感知算法训练的模型。为了减轻训练时的标签错误影响，我们提出了一种估计训练输入标签对模型差异度量影响的方法。我们在多个数据集上以实证方式评估了该方法，并与替代方法相比发现了显著的改进。

    Errors in labels obtained via human annotation adversely affect a model's performance. Existing approaches propose ways to mitigate the effect of label error on a model's downstream accuracy, yet little is known about its impact on a model's disparity metrics. Here we study the effect of label error on a model's disparity metrics. We empirically characterize how varying levels of label error, in both training and test data, affect these disparity metrics. We find that group calibration and other metrics are sensitive to train-time and test-time label error -- particularly for minority groups. This disparate effect persists even for models trained with noise-aware algorithms. To mitigate the impact of training-time label error, we present an approach to estimate the influence of a training input's label on a model's group disparity metric. We empirically assess the proposed approach on a variety of datasets and find significant improvement, compared to alternative approaches, in identif
    
[^109]: 联邦条件随机优化

    Federated Conditional Stochastic Optimization. (arXiv:2310.02524v1 [cs.LG])

    [http://arxiv.org/abs/2310.02524](http://arxiv.org/abs/2310.02524)

    本文提出了一种新的联邦条件随机优化算法(FCSG)，针对联邦学习中的非凸条件随机优化问题，通过设计加速算法(Acc-FCSG-M)来实现最佳的样本和通信复杂度。

    

    条件随机优化在机器学习任务中广泛应用，例如不变学习、AUPRC最大化和元学习。随着这些应用中对使用大规模分布式数据进行模型训练的需求增加，对于高效通信的分布式优化算法，例如联邦学习算法的需求也越来越大。本文考虑联邦学习中的非凸条件随机优化，并提出了第一个联邦条件随机优化算法(FCSG)，其中包括条件随机梯度估计器和基于动量的算法(FCSG-M)。为了达到单机设定下的下界复杂度，我们通过方差减少设计了加速算法(Acc-FCSG-M)以实现最佳的样本和通信复杂度。与现有的FL中MAML的优化分析相比，联邦条件随机优化考虑了样本的。

    Conditional stochastic optimization has found applications in a wide range of machine learning tasks, such as invariant learning, AUPRC maximization, and meta-learning. As the demand for training models with large-scale distributed data grows in these applications, there is an increasing need for communication-efficient distributed optimization algorithms, such as federated learning algorithms. This paper considers the nonconvex conditional stochastic optimization in federated learning and proposes the first federated conditional stochastic optimization algorithm (FCSG) with a conditional stochastic gradient estimator and a momentum-based algorithm (FCSG-M). To match the lower bound complexity in the single-machine setting, we design an accelerated algorithm (Acc-FCSG-M) via the variance reduction to achieve the best sample and communication complexity. Compared with the existing optimization analysis for MAML in FL, federated conditional stochastic optimization considers the sample of
    
[^110]: MedDiffusion: 通过基于扩散的数据增强提升健康风险预测

    MedDiffusion: Boosting Health Risk Prediction via Diffusion-based Data Augmentation. (arXiv:2310.02520v1 [cs.LG])

    [http://arxiv.org/abs/2310.02520](http://arxiv.org/abs/2310.02520)

    本文介绍了一种名为MedDiffusion的新型、端到端的扩散式风险预测模型，通过基于扩散的数据增强，提升了健康风险预测的效果。

    

    健康风险预测是医学领域中基于预测建模的基本任务之一，旨在利用历史电子健康记录（EHR）来预测患者未来可能面临的潜在健康风险。研究人员已经开发了几种风险预测模型来处理EHR数据的独特挑战，例如其序列特性，高维度和固有噪音。这些模型已经取得了令人印象深刻的结果。然而，一个影响它们有效性的关键问题是数据不足。为了缓解这个问题，引入了各种数据生成和增强方法，通过学习底层数据分布来扩大训练数据集的大小。然而，这些方法的性能往往受到任务无关设计的限制。为了解决这些缺点，本文引入了一种新颖的端到端扩散式风险预测模型MedDiffusion，来增强风险预测的性能。

    Health risk prediction is one of the fundamental tasks under predictive modeling in the medical domain, which aims to forecast the potential health risks that patients may face in the future using their historical Electronic Health Records (EHR). Researchers have developed several risk prediction models to handle the unique challenges of EHR data, such as its sequential nature, high dimensionality, and inherent noise. These models have yielded impressive results. Nonetheless, a key issue undermining their effectiveness is data insufficiency. A variety of data generation and augmentation methods have been introduced to mitigate this issue by expanding the size of the training data set through the learning of underlying data distributions. However, the performance of these methods is often limited due to their task-unrelated design. To address these shortcomings, this paper introduces a novel, end-to-end diffusion-based risk prediction model, named MedDiffusion. It enhances risk predicti
    
[^111]: 参参数化凸支配法用于摊销优化中目标函数的逼近

    Parameterized Convex Minorant for Objective Function Approximation in Amortized Optimization. (arXiv:2310.02519v1 [cs.LG])

    [http://arxiv.org/abs/2310.02519](http://arxiv.org/abs/2310.02519)

    提出了一种参数化凸支配（PCM）方法，用于在摊销优化中逼近目标函数。该方法具有通用逼近性能，并可以通过单个凸优化获得全局最小值。

    

    提出了一种参数化凸支配（PCM）方法，用于在摊销优化中逼近目标函数。在提出的方法中，目标函数逼近器由PCM和非负间隙函数之和表示，其中目标函数逼近器在优化变量上由PCM凸函数下界约束。所提出的目标函数逼近器是连续函数的通用逼近器，PCM的全局最小值达到目标函数逼近器的全局最小值。因此，可以通过单个凸优化获取目标函数逼近器的全局最小值。作为所提方法的实现，提出了扩展的参数化对数和指数网络，利用参数化的对数和指数网络作为PCM。对于非参数化凸目标函数逼近和基于学习的非线性模型预测控制进行了数值模拟。

    Parameterized convex minorant (PCM) method is proposed for the approximation of the objective function in amortized optimization. In the proposed method, the objective function approximator is expressed by the sum of a PCM and a nonnegative gap function, where the objective function approximator is bounded from below by the PCM convex in the optimization variable. The proposed objective function approximator is a universal approximator for continuous functions, and the global minimizer of the PCM attains the global minimum of the objective function approximator. Therefore, the global minimizer of the objective function approximator can be obtained by a single convex optimization. As a realization of the proposed method, extended parameterized log-sum-exp network is proposed by utilizing a parameterized log-sum-exp network as the PCM. Numerical simulation is performed for non-parameterized-convex objective function approximation and for learning-based nonlinear model predictive control 
    
[^112]: 改进可验证稳健性的方法：容量和数据的配方

    A Recipe for Improved Certifiable Robustness: Capacity and Data. (arXiv:2310.02513v1 [cs.LG])

    [http://arxiv.org/abs/2310.02513](http://arxiv.org/abs/2310.02513)

    在本研究中，我们通过使用一系列新技术、设计优化和综合以前的研究，更全面地评估了基于Lipschitz的认证方法的潜力，并显著提高了状...

    

    理论和实践都支持一个关键挑战，即稳健性要求比标准训练更大的网络容量和更多的数据。然而，在严格的Lipschitz约束下有效地增加容量比看起来更困难，这表明现有的方法更倾向于低拟合而不是过拟合。此外，我们主张对基于Lipshitz的方法的设计空间进行仔细探索不足，这会导致潜在的性能提升被忽视。在这项工作中，我们通过使用一系列新技术、设计优化和综合以前的研究，更全面地评估Lipschitz-based认证方法的潜力。我们能够显著提高确定性认证在各种基准数据集上的“验证稳健准确性”（VRA），并覆盖一系列扰动大小。特别值得注意的是，我们发现...

    A key challenge, supported both theoretically and empirically, is that robustness demands greater network capacity and more data than standard training. However, effectively adding capacity under stringent Lipschitz constraints has proven more difficult than it may seem, evident by the fact that state-of-the-art approach tend more towards \emph{underfitting} than overfitting. Moreover, we posit that a lack of careful exploration of the design space for Lipshitz-based approaches has left potential performance gains on the table. In this work, we provide a more comprehensive evaluation to better uncover the potential of Lipschitz-based certification methods. Using a combination of novel techniques, design optimizations, and synthesis of prior work, we are able to significantly improve the state-of-the-art \emph{verified robust accuracy} (VRA) for deterministic certification on a variety of benchmark datasets, and over a range of perturbation sizes. Of particular note, we discover that th
    
[^113]: Ophiuchus: 通过分层粗粒化SO(3)-等变自编码器对蛋白质结构进行可扩展建模

    Ophiuchus: Scalable Modeling of Protein Structures through Hierarchical Coarse-graining SO(3)-Equivariant Autoencoders. (arXiv:2310.02508v1 [cs.LG])

    [http://arxiv.org/abs/2310.02508](http://arxiv.org/abs/2310.02508)

    Ophiuchus是一个通过分层粗粒化SO(3)-等变自编码器对蛋白质结构进行可扩展建模的模型，它能在高分辨率下操作所有重原子，同时捕捉到结构的重复和分层模式。

    

    天然蛋白质的三维原生态状态显示出重复和分层模式。然而，传统的基于图的蛋白质结构建模通常局限于在单一精细化分辨率内操作，并且缺乏捕捉高级构建模块的中间神经网络架构。我们通过引入Ophiuchus来填补这个差距，它是一个SO(3)-等变粗粒化模型，可以高效地操作标准蛋白质残基的所有重原子，并同时尊重它们的相关对称性。我们的模型与当前采用图模型的方法不同，而是专注于局部卷积粗化，以在对数线性长度复杂度下模拟序列模体之间的相互作用。我们使用PDB单体的连续片段对Ophiuchus进行训练，研究其在不同压缩率下的重构能力。我们检查学习到的潜空间，并展示其在构象插值中的快速使用，将插值轨迹与结构进行比较。

    Three-dimensional native states of natural proteins display recurring and hierarchical patterns. Yet, traditional graph-based modeling of protein structures is often limited to operate within a single fine-grained resolution, and lacks hourglass neural architectures to learn those high-level building blocks. We narrow this gap by introducing Ophiuchus, an SO(3)-equivariant coarse-graining model that efficiently operates on all heavy atoms of standard protein residues, while respecting their relevant symmetries. Our model departs from current approaches that employ graph modeling, instead focusing on local convolutional coarsening to model sequence-motif interactions in log-linear length complexity. We train Ophiuchus on contiguous fragments of PDB monomers, investigating its reconstruction capabilities across different compression rates. We examine the learned latent space and demonstrate its prompt usage in conformational interpolation, comparing interpolated trajectories to structure
    
[^114]: 通过扩散学习实现目标达成

    Learning to Reach Goals via Diffusion. (arXiv:2310.02505v1 [cs.LG])

    [http://arxiv.org/abs/2310.02505](http://arxiv.org/abs/2310.02505)

    本论文提出了一种通过扩散学习实现目标达成的方法，可以在任意初始状态下从预定义或新目标达成，而无需学习单独的价值函数。

    

    扩散模型是一类强大的生成模型，能够通过迭代去噪将高维空间中的随机噪声映射到目标流形上。在本研究中，我们通过将目标条件强化学习框架放在扩散建模的背景下，给出了一种新的视角。类似于扩散过程，其中利用高斯噪声创建随机轨迹，使其远离数据流形，我们构造了远离潜在目标状态的轨迹。然后我们学习一个类似于评分函数的目标条件策略。这个称为Merlin的方法能够在任意初始状态下从预定义或新目标达成，而无需学习单独的价值函数。我们考虑了三种选择，用于取代扩散中的高斯噪声模型 - 缓冲区中的反向播放，反向动力学模型和一种新的非参数方法。我们在离线目标达成任务上理论上证明了我们的方法，并对其进行了验证。

    Diffusion models are a powerful class of generative models capable of mapping random noise in high-dimensional spaces to a target manifold through iterative denoising. In this work, we present a novel perspective on goal-conditioned reinforcement learning by framing it within the context of diffusion modeling. Analogous to the diffusion process, where Gaussian noise is used to create random trajectories that walk away from the data manifold, we construct trajectories that move away from potential goal states. We then learn a goal-conditioned policy analogous to the score function. This approach, which we call Merlin, can reach predefined or novel goals from an arbitrary initial state without learning a separate value function. We consider three choices for the noise model to replace Gaussian noise in diffusion - reverse play from the buffer, reverse dynamics model, and a novel non-parametric approach. We theoretically justify our approach and validate it on offline goal-reaching tasks.
    
[^115]: 通过感知声音特质实现可解释的说话者身份表示

    Towards an Interpretable Representation of Speaker Identity via Perceptual Voice Qualities. (arXiv:2310.02497v1 [cs.SD])

    [http://arxiv.org/abs/2310.02497](http://arxiv.org/abs/2310.02497)

    本研究提出了一种基于感知声音特质的可解释的说话者身份表示方法，通过将性别化的PQs添加到声音感知评估协议中，实现了成年人声音特征的中间层抽象。实验证明这些声音特质是可以被非专业人员感知到的，并且基于PQs的表示中的信息是可以被各种语音表示预测的。

    

    与文本和视觉等其他数据形式不同，语音不易解释。虽然普通人可以通过感知来描述图像或句子，但对于语音，非专业人士的描述通常仅限于高级别的人口统计信息，如性别或年龄。在本文中，我们提出了一种可能的基于感知声音特质（PQs）的可解释的说话者身份表示。通过将性别化的PQs添加到以病理为焦点的声音统一听觉感知评估（CAPE-V）协议中，我们的基于PQs的方法提供了一种成年人声音特征的感知潜在空间，它是高级别人口统计信息和低级别声学、物理或学习表示之间的中间层抽象。与先前的观点相反，我们证明了这些PQs是可以通过非专业人员的观察听到的，并进一步证明了在基于PQs的表示中编码的信息是可以通过各种语音表示来预测的。

    Unlike other data modalities such as text and vision, speech does not lend itself to easy interpretation. While lay people can understand how to describe an image or sentence via perception, non-expert descriptions of speech often end at high-level demographic information, such as gender or age. In this paper, we propose a possible interpretable representation of speaker identity based on perceptual voice qualities (PQs). By adding gendered PQs to the pathology-focused Consensus Auditory-Perceptual Evaluation of Voice (CAPE-V) protocol, our PQ-based approach provides a perceptual latent space of the character of adult voices that is an intermediary of abstraction between high-level demographics and low-level acoustic, physical, or learned representations. Contrary to prior belief, we demonstrate that these PQs are hearable by ensembles of non-experts, and further demonstrate that the information encoded in a PQ-based representation is predictable by various speech representations.
    
[^116]: DON-LSTM: 用DeepONets和长短期记忆神经网络进行多分辨率学习

    DON-LSTM: Multi-Resolution Learning with DeepONets and Long Short-Term Memory Neural Networks. (arXiv:2310.02491v1 [cs.LG])

    [http://arxiv.org/abs/2310.02491](http://arxiv.org/abs/2310.02491)

    DON-LSTM是一种新的架构，将DeepONet与长短期记忆网络（LSTM）结合起来，旨在通过利用多分辨率数据和捕捉长序列的时间依赖性来提高模型的性能。实验结果表明，DON-LSTM能够在多个非线性系统的长时间演化建模方面实现较低的泛化误差，并且需要较少的高分辨率数据。

    

    深度操作器网络（DeepONets, DONs）在能够训练多分辨率数据方面相比传统神经网络具有独特优势。在现实世界的场景中，高分辨率的测量数据往往难以获得，而低分辨率的数据更容易获得，这个特性尤为重要。然而，仅凭DeepONets往往难以捕捉和保持长序列的相关性，与其他最先进的算法相比。我们提出了一种新的架构，命名为DON-LSTM，它将DeepONet与长短期记忆网络（LSTM）结合起来。通过结合这两种架构，我们赋予网络明确的机制来利用多分辨率数据，并捕捉长序列的时间依赖性。我们在多个非线性系统的长时间演化建模方面测试了我们的方法，并展示了所提出的多分辨率DON-LSTM实现了显著较低的泛化误差，并且需要较少的高分辨率数据。

    Deep operator networks (DeepONets, DONs) offer a distinct advantage over traditional neural networks in their ability to be trained on multi-resolution data. This property becomes especially relevant in real-world scenarios where high-resolution measurements are difficult to obtain, while low-resolution data is more readily available. Nevertheless, DeepONets alone often struggle to capture and maintain dependencies over long sequences compared to other state-of-the-art algorithms. We propose a novel architecture, named DON-LSTM, which extends the DeepONet with a long short-term memory network (LSTM). Combining these two architectures, we equip the network with explicit mechanisms to leverage multi-resolution data, as well as capture temporal dependencies in long sequences. We test our method on long-time-evolution modeling of multiple non-linear systems and show that the proposed multi-resolution DON-LSTM achieves significantly lower generalization error and requires fewer high-resolut
    
[^117]: ResidualTransformer：带有权重共享的残差低秩学习的Transformer层

    ResidualTransformer: Residual Low-rank Learning with Weight-sharing for Transformer Layers. (arXiv:2310.02489v1 [cs.CL])

    [http://arxiv.org/abs/2310.02489](http://arxiv.org/abs/2310.02489)

    本文提出了一种名为ResidualTransformer的方法，通过重新参数化Transformer编码器层之间的模型权重，将模型的大小减小。实验结果表明，ResidualTransformer的性能优于传统Transformer模型，且模型大小得到了显著减小。

    

    在部署语音处理模型到始终开启设备上时，内存限制是一个主要关注点之一。虽然使用足够大量的数据训练得到的更大的模型通常表现更好，但使其适应设备内存是一个具有挑战性的问题。在本文中，我们旨在通过重新参数化Transformer编码器层之间的模型权重，并假设特殊的权重组合和结构，来减小模型的大小。更具体地说，受ResNet和最新的LoRA工作的启发，我们提出了一种名为ResidualTransformer的方法，其中Transformer层中的每个权重矩阵包括1）与其相邻层共享的满秩组件，和2）仅属于它自己的独特低秩组件。低秩矩阵只占模型大小的一小部分。此外，我们添加对角线权重矩阵来提高低秩矩阵的建模能力。我们的10k小时语音识别和语音翻译任务的实验结果表明，ResidualTransformer的性能优于传统Transformer模型，且模型大小得到了显著减小。

    Memory constraint of always-on devices is one of the major concerns when deploying speech processing models on these devices. While larger models trained with sufficiently large amount of data generally perform better, making them fit in the device memory is a demanding challenge. In this paper, we aim to reduce model size by reparameterizing model weights across Transformer encoder layers and assuming a special weight composition and structure. More specifically, inspired by ResNet and the more recent LoRA work, we propose an approach named ResidualTransformer, where each weight matrix in a Transformer layer comprises 1) a shared full-rank component with its adjacent layers, and 2) a unique low-rank component to itself. The low-rank matrices only account for a small amount of model size increase. In addition, we add diagonal weight matrices to improve modeling capacity of the low-rank matrices. Experiments of our 10k-hour speech recognition and speech translation tasks show that the T
    
[^118]: OCU-Net: 一种用于增强口腔癌分割的新型U-Net架构

    OCU-Net: A Novel U-Net Architecture for Enhanced Oral Cancer Segmentation. (arXiv:2310.02486v1 [eess.IV])

    [http://arxiv.org/abs/2310.02486](http://arxiv.org/abs/2310.02486)

    OCU-Net是一种新型的U-Net架构，专门用于口腔癌分割任务。它结合了多种创新的深度学习模块和特征，包括通道和空间注意融合模块、挤压激活注意模块、空洞空间金字塔池化模块等，并在两个数据集上展现出卓越的性能。

    

    准确检测口腔癌对于改善患者结果至关重要。然而，该领域面临两个关键挑战：缺乏专门针对口腔癌的基于深度学习的图像分割研究和缺乏注释数据。我们的研究提出了OCU-Net，一种创新的U-Net图像分割架构，专门设计用于在血红素与噪音(HE)染色图像数据集中检测口腔癌。OCU-Net采用了先进的深度学习模块，如通道和空间注意融合(CSAF)模块，这是一种强调HE图像中重要通道和空间区域并探索上下文信息的新颖创新特征。此外，OCU-Net还集成了其他创新组件，如挤压激活(SE)注意模块，空洞空间金字塔池化(ASPP)模块，残差块和多尺度融合。这些模块的融合在口腔癌分割的两个数据集中表现出卓越的性能。

    Accurate detection of oral cancer is crucial for improving patient outcomes. However, the field faces two key challenges: the scarcity of deep learning-based image segmentation research specifically targeting oral cancer and the lack of annotated data. Our study proposes OCU-Net, a pioneering U-Net image segmentation architecture exclusively designed to detect oral cancer in hematoxylin and eosin (H&E) stained image datasets. OCU-Net incorporates advanced deep learning modules, such as the Channel and Spatial Attention Fusion (CSAF) module, a novel and innovative feature that emphasizes important channel and spatial areas in H&E images while exploring contextual information. In addition, OCU-Net integrates other innovative components such as Squeeze-and-Excite (SE) attention module, Atrous Spatial Pyramid Pooling (ASPP) module, residual blocks, and multi-scale fusion. The incorporation of these modules showed superior performance for oral cancer segmentation for two datasets used in th
    
[^119]: 对抗训练中的差异性分割

    Splitting the Difference on Adversarial Training. (arXiv:2310.02480v1 [cs.LG])

    [http://arxiv.org/abs/2310.02480](http://arxiv.org/abs/2310.02480)

    本文提出了一种基于分割类别的对抗训练方法，将每个类别的扰动样本视为单独的类别进行学习，从而简化了决策边界，提高了模型的鲁棒性。

    

    对抗性样本的存在指出了深度神经网络的一个基本弱点。对抗训练作为针对此类样本最有效的防御方法，需要在某种程度上训练模型以提高鲁棒性，通常以自然准确性的降低为代价。大多数对抗训练方法的目标是学习模型，为每个类别找到一个共同的决策边界，涵盖了干净和扰动的样本。在这项工作中，我们采取了一个根本不同的方法，将每个类别的扰动样本视为一个单独的需要学习的类别，有效地将每个类别分为两个类别："干净"和"对抗性"。这种分割使得需要学习的类别数量翻倍，但同时大大简化了决策边界。我们提供了一种理论上的合理性论证，以阐明我们的方法有望在何种条件下有益。同样，我们通过实验证明了我们的方法学习到了鲁棒的模型。

    The existence of adversarial examples points to a basic weakness of deep neural networks. One of the most effective defenses against such examples, adversarial training, entails training models with some degree of robustness, usually at the expense of a degraded natural accuracy. Most adversarial training methods aim to learn a model that finds, for each class, a common decision boundary encompassing both the clean and perturbed examples. In this work, we take a fundamentally different approach by treating the perturbed examples of each class as a separate class to be learned, effectively splitting each class into two classes: "clean" and "adversarial." This split doubles the number of classes to be learned, but at the same time considerably simplifies the decision boundaries. We provide a theoretical plausibility argument that sheds some light on the conditions under which our approach can be expected to be beneficial. Likewise, we empirically demonstrate that our method learns robust
    
[^120]: ML4EJ：使用可解释的机器学习解码城市特征在塑造环境不公正中的作用

    ML4EJ: Decoding the Role of Urban Features in Shaping Environmental Injustice Using Interpretable Machine Learning. (arXiv:2310.02476v1 [cs.LG])

    [http://arxiv.org/abs/2310.02476](http://arxiv.org/abs/2310.02476)

    本研究使用可解释的机器学习模型，研究了城市特征对空气污染、城市热岛效应和洪涝灾害的暴露差异的影响，并弥补了传统环境不公正观点对城市特征影响的有限视角。

    

    理解塑造环境危害暴露及其相关环境不公正问题的关键因素对制定公平政策措施至关重要。传统环境不公正观点主要关注社会经济方面，往往忽视了异质城市特征的影响。这种有限的观点可能阻碍对环境正义的复杂性及其与城市设计特征的关系的全面理解。为了填补这一空白，本研究创建了一个可解释的机器学习模型，以研究各种城市特征及其非线性交互对三种主要危害（空气污染、城市热岛效应和洪涝灾害）的暴露差异的影响。分析使用来自美国六个大都会县的数据进行随机森林和XGBoost训练和测试模型。性能用于衡量城市特征变化的程度，以塑造不公正。

    Understanding the key factors shaping environmental hazard exposures and their associated environmental injustice issues is vital for formulating equitable policy measures. Traditional perspectives on environmental injustice have primarily focused on the socioeconomic dimensions, often overlooking the influence of heterogeneous urban characteristics. This limited view may obstruct a comprehensive understanding of the complex nature of environmental justice and its relationship with urban design features. To address this gap, this study creates an interpretable machine learning model to examine the effects of various urban features and their non-linear interactions to the exposure disparities of three primary hazards: air pollution, urban heat, and flooding. The analysis trains and tests models with data from six metropolitan counties in the United States using Random Forest and XGBoost. The performance is used to measure the extent to which variations of urban features shape disparitie
    
[^121]: 基于提示的高效时域泛化

    Prompting-based Efficient Temporal Domain Generalization. (arXiv:2310.02473v1 [cs.LG])

    [http://arxiv.org/abs/2310.02473](http://arxiv.org/abs/2310.02473)

    我们提出了一种基于提示的高效时域泛化方法，通过学习全局提示、领域特定提示和感知时序漂移的提示，不需要目标域数据的情况下适应时序漂移，并在各种任务中取得了state-of-the-art的性能。

    

    传统的机器学习假设训练和测试数据是独立且相同分布的。然而，在许多实际应用中，数据分布会随时间变化，导致训练好的模型在未来时间段的泛化能力变差。我们的论文提出了一种新颖的基于提示的时域泛化方法，它具有参数高效、时间高效，并且在训练过程中不需要访问目标域数据（即未知的未来时间段）。我们的方法通过学习全局提示、领域特定提示和感知到时序漂移的提示的方式，将目标预训练模型适应于时序漂移。它适用于各种任务，例如分类、回归和时间序列预测，并在时域泛化方面取得了新的最优性能。代码仓库将公开分享。

    Machine learning traditionally assumes that training and testing data are distributed independently and identically. However, in many real-world settings, the data distribution can shift over time, leading to poor generalization of trained models in future time periods. Our paper presents a novel prompting-based approach to temporal domain generalization that is parameter-efficient, time-efficient, and does not require access to the target domain data (i.e., unseen future time periods) during training. Our method adapts a target pre-trained model to temporal drift by learning global prompts, domain-specific prompts, and drift-aware prompts that capture underlying temporal dynamics. It is compatible across diverse tasks, such as classification, regression, and time series forecasting, and sets a new state-of-the-art benchmark in temporal domain generalization. The code repository will be publicly shared.
    
[^122]: 在模型不确定性下的分布安全强化学习: 基于可微分凸规划的单层方法

    Distributionally Safe Reinforcement Learning under Model Uncertainty: A Single-Level Approach by Differentiable Convex Programming. (arXiv:2310.02459v1 [cs.LG])

    [http://arxiv.org/abs/2310.02459](http://arxiv.org/abs/2310.02459)

    本文提出了一个分布安全的强化学习框架，通过Wasserstein度量来确保在模型不确定性下的安全性。通过使用对偶理论和可微分凸规划，将双层问题简化为单层问题，提高了可行性和效率。

    

    在存在剧烈模型不确定性（如分布偏移）的安全关键环境中，安全保证是不可妥协的，特别是在人员参与的情况下。然而，将不确定性纳入安全学习中自然会导致一个双层问题，在这个问题中，较低层次上在不确定性模糊集合内评估（最坏情况下的）安全约束。本文提出了一个可行的分布安全强化学习框架，通过Wasserstein度量来确保在分布偏移下的安全性。为了提高可操作性，我们首先使用对偶理论将较低层次的优化问题从无限维概率空间（用于测量分布偏移）转化为有限维参数空间。此外，通过可微分凸规划，将双层安全学习问题进一步简化为一个单层问题，需要两个顺序计算高效模块：一个凸二次规划来保证安全性约束，一个可微分优化来学习策略。

    Safety assurance is uncompromisable for safety-critical environments with the presence of drastic model uncertainties (e.g., distributional shift), especially with humans in the loop. However, incorporating uncertainty in safe learning will naturally lead to a bi-level problem, where at the lower level the (worst-case) safety constraint is evaluated within the uncertainty ambiguity set. In this paper, we present a tractable distributionally safe reinforcement learning framework to enforce safety under a distributional shift measured by a Wasserstein metric. To improve the tractability, we first use duality theory to transform the lower-level optimization from infinite-dimensional probability space where distributional shift is measured, to a finite-dimensional parametric space. Moreover, by differentiable convex programming, the bi-level safe learning problem is further reduced to a single-level one with two sequential computationally efficient modules: a convex quadratic program to gu
    
[^123]: 从偏好中学习最佳优势，并将其误解为奖励

    Learning Optimal Advantage from Preferences and Mistaking it for Reward. (arXiv:2310.02456v1 [cs.LG])

    [http://arxiv.org/abs/2310.02456](http://arxiv.org/abs/2310.02456)

    本文研究了从人类偏好中学习奖励函数的算法，并发现实际上学到的是最佳优势函数而不是奖励函数。这种错误的使用方式虽然不特别有害，但与正确的贪婪最大化最佳优势函数相比仍不够理想。

    

    我们考虑从人类对轨迹片段对的偏好中学习奖励函数的算法，这在从人类反馈中进行强化学习（RLHF）中使用。最近的工作假设人类偏好仅基于这些片段中积累的奖励或其部分回报。最近的工作对这一假设的有效性提出了怀疑，并提出了一种基于遗憾的替代偏好模型。我们研究了当假设偏好是基于部分回报而实际上来自遗憾时的后果。我们认为学到的函数是最佳优势函数$\hat{A^*_r}$的近似，而不是奖励函数。我们发现，如果解决了特定的陷阱，这种错误假设并不特别有害，结果是一个高度变形的奖励函数。尽管如此，这种错误使用$\hat{A^*_r}$的方式不如适当且更简单的方法——贪婪最大化$\hat{A^*_r}$。

    We consider algorithms for learning reward functions from human preferences over pairs of trajectory segments, as used in reinforcement learning from human feedback (RLHF). Most recent work assumes that human preferences are generated based only upon the reward accrued within those segments, or their partial return. Recent work casts doubt on the validity of this assumption, proposing an alternative preference model based upon regret. We investigate the consequences of assuming preferences are based upon partial return when they actually arise from regret. We argue that the learned function is an approximation of the optimal advantage function, $\hat{A^*_r}$, not a reward function. We find that if a specific pitfall is addressed, this incorrect assumption is not particularly harmful, resulting in a highly shaped reward function. Nonetheless, this incorrect usage of $\hat{A^*_r}$ is less desirable than the appropriate and simpler approach of greedy maximization of $\hat{A^*_r}$. From th
    
[^124]: 可追溯城市规划的双阶段流式生成建模

    Dual-stage Flows-based Generative Modeling for Traceable Urban Planning. (arXiv:2310.02453v1 [cs.LG])

    [http://arxiv.org/abs/2310.02453](http://arxiv.org/abs/2310.02453)

    这项研究提出了一种基于流式生成模型的双阶段城市规划框架，用于解决传统城市规划方法中忽略功能区关系、生成过程不稳定等问题。

    

    城市规划在当代社会的高速城市化进程中变得越来越重要，但传统的人工规划方法复杂且繁重。由于深度学习算法的进步，研究人员开始开发自动化规划技术。虽然这些模型取得了有希望的结果，但仍存在一些未解决的限制：1）忽略城市功能区之间的关系以及无法捕捉不同功能区之间的关系。2）生成过程缺乏解释性和稳定性。为了克服这些限制，我们提出了一种基于归一化流的新型生成框架，即双阶段城市流(DSUF)框架。具体而言，第一阶段利用区域级城市规划流来生成基于给定配置的城市功能区。

    Urban planning, which aims to design feasible land-use configurations for target areas, has become increasingly essential due to the high-speed urbanization process in the modern era. However, the traditional urban planning conducted by human designers can be a complex and onerous task. Thanks to the advancement of deep learning algorithms, researchers have started to develop automated planning techniques. While these models have exhibited promising results, they still grapple with a couple of unresolved limitations: 1) Ignoring the relationship between urban functional zones and configurations and failing to capture the relationship among different functional zones. 2) Less interpretable and stable generation process. To overcome these limitations, we propose a novel generative framework based on normalizing flows, namely Dual-stage Urban Flows (DSUF) framework. Specifically, the first stage is to utilize zone-level urban planning flows to generate urban functional zones based on give
    
[^125]: Feather:一种优雅的解决DNN稀疏化问题的解决方案

    Feather: An Elegant Solution to Effective DNN Sparsification. (arXiv:2310.02448v1 [cs.LG])

    [http://arxiv.org/abs/2310.02448](http://arxiv.org/abs/2310.02448)

    Feather是一种优雅的DNN稀疏化解决方案，它具有高效的稀疏训练模块和强大的直通过估计器核心，能够在标准训练过程中实现鲁棒的稀疏化性能，并在CIFAR数据集上展示了其有效性和适应性，在ImageNet上使用ResNet-50架构实现了最新的最佳验证准确率，超越了现有方法。

    

    神经网络剪枝是一种越来越流行的方法，可以生成适用于资源有限环境并保持高性能的紧凑高效的模型。虽然剪枝可以通过多周期训练和微调的过程来执行，但最近的趋势是在标准训练过程中同时包含稀疏化过程。为此，我们引入了Feather，一种高效的稀疏训练模块，其核心是强大的直通过估计器，配合一个新的阈值算子和梯度缩放技术，实现了强大的开箱即用的稀疏化性能。在CIFAR数据集上，我们使用不同的架构证明了Feather的有效性和适应性，而在ImageNet上，它使用ResNet-50架构实现了最新的Top-1验证准确率，超过了现有的方法，包括更复杂、计算量更大的方法，差距很大。代码公开在 https://git...

    Neural Network pruning is an increasingly popular way for producing compact and efficient models, suitable for resource-limited environments, while preserving high performance. While the pruning can be performed using a multi-cycle training and fine-tuning process, the recent trend is to encompass the sparsification process during the standard course of training. To this end, we introduce Feather, an efficient sparse training module utilizing the powerful Straight-Through Estimator as its core, coupled with a new thresholding operator and a gradient scaling technique, enabling robust, out-of-the-box sparsification performance. Feather's effectiveness and adaptability is demonstrated using various architectures on the CIFAR dataset, while on ImageNet it achieves state-of-the-art Top-1 validation accuracy using the ResNet-50 architecture, surpassing existing methods, including more complex and computationally heavy ones, by a considerable margin. Code is publicly available at https://git
    
[^126]: 机器学习辅助纽约地铁导航更安全更快

    Machine learning assist nyc subway navigation safer and faster. (arXiv:2310.02447v1 [cs.SI])

    [http://arxiv.org/abs/2310.02447](http://arxiv.org/abs/2310.02447)

    一项使用机器学习的研究旨在通过整数规划模型和最短路径算法的综合评估，平衡纽约地铁导航的安全性和效率性。

    

    主流导航软件如谷歌地图和苹果地图往往缺乏提供以安全为优先的路线的能力。然而，安全始终是许多人关注的重点。我们的目标是在安全和效率之间取得平衡。为了实现这一目标，我们正在设计一个整数规划模型，考虑最短路径和最安全路线的因素。我们将利用机器学习来推导安全系数，采用广义线性模型、线性回归和循环神经网络等方法。我们的评估将基于各个地铁站的均方根误差（RMSE），帮助我们确定最准确的安全系数估计模型。此外，我们还将对不同的最短路径算法进行全面评估，根据时间复杂度和真实世界数据评估它们在同时考虑安全和时间效率方面的适用性。

    Mainstream navigation software, like Google and Apple Maps, often lacks the ability to provide routes prioritizing safety. However, safety remains a paramount concern for many. Our aim is to strike a balance between safety and efficiency. To achieve this, we're devising an Integer Programming model that takes into account both the shortest path and the safest route. We will harness machine learning to derive safety coefficients, employing methodologies such as generalized linear models, linear regression, and recurrent neural networks. Our evaluation will be based on the Root Mean Square Error (RMSE) across various subway stations, helping us identify the most accurate model for safety coefficient estimation. Furthermore, we'll conduct a comprehensive review of different shortest-path algorithms, assessing them based on time complexity and real-world data to determine their appropriateness in merging both safety and time efficiency.
    
[^127]: 低资源语言越狱 GPT-4

    Low-Resource Languages Jailbreak GPT-4. (arXiv:2310.02446v1 [cs.CL])

    [http://arxiv.org/abs/2310.02446](http://arxiv.org/abs/2310.02446)

    通过翻译不安全的英文输入成低资源语言，我们成功绕过了GPT-4的安全机制，并展示了这种跨语言漏洞。这一方法在实验中取得了与甚至超过了最先进的越狱攻击的效果，揭示了低资源语言在AI安全性中的薄弱环节。

    

    人工智能安全培训和大型语言模型（LLM）的红队测试是减少生成不安全内容的措施。我们的工作通过将不安全的英文输入翻译成低资源语言，成功绕过GPT-4的安全机制，并揭示了这些安全机制的跨语言漏洞。在AdvBenchmark中，GPT-4针对不安全的翻译输入进行交互，并且79%的时间内提供了可行的方案，使用户实现其有害目标，这与甚至超过了最先进的越狱攻击的效果相当。其他高/中资源语言的攻击成功率显著较低，这表明跨语言漏洞主要适用于低资源语言。以前，对低资源语言的有限训练主要影响那些使用这些语言的人，造成技术差距。然而，我们的工作突出了一个关键转变：

    AI safety training and red-teaming of large language models (LLMs) are measures to mitigate the generation of unsafe content. Our work exposes the inherent cross-lingual vulnerability of these safety mechanisms, resulting from the linguistic inequality of safety training data, by successfully circumventing GPT-4's safeguard through translating unsafe English inputs into low-resource languages. On the AdvBenchmark, GPT-4 engages with the unsafe translated inputs and provides actionable items that can get the users towards their harmful goals 79% of the time, which is on par with or even surpassing state-of-the-art jailbreaking attacks. Other high-/mid-resource languages have significantly lower attack success rate, which suggests that the cross-lingual vulnerability mainly applies to low-resource languages. Previously, limited training on low-resource languages primarily affects speakers of those languages, causing technological disparities. However, our work highlights a crucial shift:
    
[^128]: GenCO: 用于具有组合特征的设计问题的生成多样解决方案

    GenCO: Generating Diverse Solutions to Design Problems with Combinatorial Nature. (arXiv:2310.02442v1 [cs.LG])

    [http://arxiv.org/abs/2310.02442](http://arxiv.org/abs/2310.02442)

    GenCO是一个新的框架，它整合了嵌入的组合求解器和深层生成模型，以发现与非线性目标一致的高质量解决方案。

    

    使用生成模型（如GAN或VAE）生成多样化的对象（如图像）在最近几年取得了令人瞩目的成果，以帮助解决许多传统上由人类完成的设计问题。我们的目标是超越图像生成，在更一般的设计问题中寻找解决方案，其中设计的多样性和约束的一致性都很重要。这样的设置在计算机图形学、动画、工业设计、材料科学等领域中都有应用，其中我们希望生成器的输出遵循离散/组合约束并惩罚任何偏离，这对于现有的生成模型和优化求解器来说是非平凡的。为了解决这个问题，我们提出了GenCO，一种新颖的框架，它集成了嵌入的组合求解器和深层生成模型的端到端训练，旨在发现与非线性目标一致的高质量解决方案。

    Generating diverse objects (e.g., images) using generative models (such as GAN or VAE) has achieved impressive results in the recent years, to help solve many design problems that are traditionally done by humans. Going beyond image generation, we aim to find solutions to more general design problems, in which both the diversity of the design and conformity of constraints are important. Such a setting has applications in computer graphics, animation, industrial design, material science, etc, in which we may want the output of the generator to follow discrete/combinatorial constraints and penalize any deviation, which is non-trivial with existing generative models and optimization solvers. To address this, we propose GenCO, a novel framework that conducts end-to-end training of deep generative models integrated with embedded combinatorial solvers, aiming to uncover high-quality solutions aligned with nonlinear objectives. While structurally akin to conventional generative models, GenCO 
    
[^129]: 反复神经网络(RNN)机制解释的片段记忆理论

    Episodic Memory Theory for the Mechanistic Interpretation of Recurrent Neural Networks. (arXiv:2310.02430v1 [cs.NE])

    [http://arxiv.org/abs/2310.02430](http://arxiv.org/abs/2310.02430)

    提出了片段记忆理论(EMT)，将反复神经网络(RNN)概念化为通用序列片段记忆模型的离散时间类比，并且通过实验证实了EMT的有效性。通过引入新的算法任务，发现受训练的RNN始终会收敛到变量绑定电路，揭示了RNN动力学的普遍性，并且设计了一个算法来揭示变量的时间存储和组合中起重要作用的隐藏神经元。

    

    了解反复神经网络(RNN)的复杂操作对于推动其能力和应用至关重要。在这项追求中，我们提出了片段记忆理论(EMT)，说明了RNN可以被概念化为最近提出的通用序列片段记忆模型的离散时间类比。为了证实EMT，我们引入了一系列新颖的算法任务，旨在探索RNN的变量绑定行为。利用EMT，我们制定了一个数学严谨的电路，用于促进这些任务中的变量绑定。我们的实证研究发现，经过训练的RNN始终会收敛到变量绑定电路，从而表明了RNN动力学的普遍性。基于这些发现，我们设计了一个算法来定义一个特权基础，揭示了在时间储存和组合变量中起重要作用的隐藏神经元，这是成功推广这些任务的关键机制。

    Understanding the intricate operations of Recurrent Neural Networks (RNNs) mechanistically is pivotal for advancing their capabilities and applications. In this pursuit, we propose the Episodic Memory Theory (EMT), illustrating that RNNs can be conceptualized as discrete-time analogs of the recently proposed General Sequential Episodic Memory Model. To substantiate EMT, we introduce a novel set of algorithmic tasks tailored to probe the variable binding behavior in RNNs. Utilizing the EMT, we formulate a mathematically rigorous circuit that facilitates variable binding in these tasks. Our empirical investigations reveal that trained RNNs consistently converge to the variable binding circuit, thus indicating universality in the dynamics of RNNs. Building on these findings, we devise an algorithm to define a privileged basis, which reveals hidden neurons instrumental in the temporal storage and composition of variables, a mechanism vital for the successful generalization in these tasks. 
    
[^130]: EGraFFBench: 用于原子模拟的等变图神经网络力场评估

    EGraFFBench: Evaluation of Equivariant Graph Neural Network Force Fields for Atomistic Simulations. (arXiv:2310.02428v1 [cs.LG])

    [http://arxiv.org/abs/2310.02428](http://arxiv.org/abs/2310.02428)

    EGraFFBench对六种EGraFF算法进行了系统的基准测试，以评估其在原子模拟中的性能，并提出了新的数据集和度量标准。

    

    通过利用图的固有对称性，等变图神经网络力场(EGraFF)在建模原子系统中的复杂相互作用方面表现出巨大的潜力。最近的研究引发了对新型架构的开发潮，这些架构将等变性的归纳偏见与图变换器和消息传递等架构创新结合起来，以建模原子相互作用。然而，我们目前对这些使用EGraFF进行实际原子模拟任务的彻底评估还缺乏。为了达到这个目的，我们在这里对6种EGraFF算法(NequIP, Allegro, BOTNet, MACE, Equiformer, TorchMDNet)进行了系统的基准测试，以了解它们在实际原子模拟中的能力和限制。除了对基于基准测试文献的八个现有数据集进行彻底评估和分析外，我们还发布了两个新的基准数据集，提出了四个新的度量标准和三个新的具有挑战性的任务。

    Equivariant graph neural networks force fields (EGraFFs) have shown great promise in modelling complex interactions in atomic systems by exploiting the graphs' inherent symmetries. Recent works have led to a surge in the development of novel architectures that incorporate equivariance-based inductive biases alongside architectural innovations like graph transformers and message passing to model atomic interactions. However, thorough evaluations of these deploying EGraFFs for the downstream task of real-world atomistic simulations, is lacking. To this end, here we perform a systematic benchmarking of 6 EGraFF algorithms (NequIP, Allegro, BOTNet, MACE, Equiformer, TorchMDNet), with the aim of understanding their capabilities and limitations for realistic atomistic simulations. In addition to our thorough evaluation and analysis on eight existing datasets based on the benchmarking literature, we release two new benchmark datasets, propose four new metrics, and three new challenging tasks.
    
[^131]: Delta-AI: 稀疏图模型的摊还推理中的局部目标

    Delta-AI: Local objectives for amortized inference in sparse graphical models. (arXiv:2310.02423v1 [cs.LG])

    [http://arxiv.org/abs/2310.02423](http://arxiv.org/abs/2310.02423)

    Delta-AI算法提出了一种基于稀疏图模型的摊还推理方法，通过局部信用分配和离策略训练加快了训练速度。

    

    我们提出了一种新的算法，用于稀疏概率图模型（PGMs）的摊还推理，我们称之为Delta-AI。我们的方法基于这样的观察：当PGM中的变量采样被视为一个代理人采取的动作序列时，PGM的稀疏性使得代理人的策略学习目标能够进行局部信用分配。这导致了一个局部约束，可以转化为类似生成流网络（GFlowNets）中的局部损失，从而实现了离策略训练，但避免了每个参数更新需要实例化所有随机变量的需求，从而大大加快了训练速度。Delta-AI目标与一个可计算的学习采样器中的变量给定其马尔可夫毯子的条件分布相匹配，该采样器的结构类似于贝叶斯网络，在目标PGM下具有相同的条件分布。因此，训练后的采样器可以恢复感兴趣变量的边际分布和条件分布。

    We present a new algorithm for amortized inference in sparse probabilistic graphical models (PGMs), which we call $\Delta$-amortized inference ($\Delta$-AI). Our approach is based on the observation that when the sampling of variables in a PGM is seen as a sequence of actions taken by an agent, sparsity of the PGM enables local credit assignment in the agent's policy learning objective. This yields a local constraint that can be turned into a local loss in the style of generative flow networks (GFlowNets) that enables off-policy training but avoids the need to instantiate all the random variables for each parameter update, thus speeding up training considerably. The $\Delta$-AI objective matches the conditional distribution of a variable given its Markov blanket in a tractable learned sampler, which has the structure of a Bayesian network, with the same conditional distribution under the target PGM. As such, the trained sampler recovers marginals and conditional distributions of intere
    
[^132]: OneAdapt：通过反向传播实现深度学习应用的快速自适应

    OneAdapt: Fast Adaptation for Deep Learning Applications via Backpropagation. (arXiv:2310.02422v1 [cs.LG])

    [http://arxiv.org/abs/2310.02422](http://arxiv.org/abs/2310.02422)

    OneAdapt通过梯度上升策略来实现快速自适应，满足了深度学习应用在配置参数方面的三个要求。

    

    深度学习在流媒体数据的推断方面已经普及，如视频中的目标检测、LiDAR数据和音频波形中的文本提取。为了实现高推断准确性，这些应用通常需要大量的网络带宽来收集高保真数据，并且需要广泛的GPU资源来运行深度神经网络(DNN)。尽管通过优化配置参数（如视频分辨率和帧率）可以大大减少对网络带宽和GPU资源的需求，但目前的自适应技术无法同时满足三个要求：（i）以最小的额外GPU或带宽开销来自适应配置；（ii）基于数据对最终DNN的准确性的影响来达到接近最优的决策；（iii）针对一系列配置参数进行自适应。本文提出了OneAdapt，通过利用梯度上升策略来自适应配置参数，满足了这些要求。关键思想是充分利用DNN的不同

    Deep learning inference on streaming media data, such as object detection in video or LiDAR feeds and text extraction from audio waves, is now ubiquitous. To achieve high inference accuracy, these applications typically require significant network bandwidth to gather high-fidelity data and extensive GPU resources to run deep neural networks (DNNs). While the high demand for network bandwidth and GPU resources could be substantially reduced by optimally adapting the configuration knobs, such as video resolution and frame rate, current adaptation techniques fail to meet three requirements simultaneously: adapt configurations (i) with minimum extra GPU or bandwidth overhead; (ii) to reach near-optimal decisions based on how the data affects the final DNN's accuracy, and (iii) do so for a range of configuration knobs. This paper presents OneAdapt, which meets these requirements by leveraging a gradient-ascent strategy to adapt configuration knobs. The key idea is to embrace DNNs' different
    
[^133]: 学生大规模语言模型能否与其教师一样表现出色？

    Can a student Large Language Model perform as well as it's teacher?. (arXiv:2310.02421v1 [cs.LG])

    [http://arxiv.org/abs/2310.02421](http://arxiv.org/abs/2310.02421)

    这篇论文总结了知识蒸馏技术，并强调了其关键原理和成功要素，以及在资源受限环境中的部署挑战。同时，论文还指出知识蒸馏有潜力成为关键的技术转折点。

    

    当代深度学习模型的复杂性不断增加，虽然能实现无与伦比的准确性，但也不可避免地在资源受限环境中带来部署挑战。知识蒸馏作为一种将高容量“教师”模型中的知识转移到简化的“学生”模型的技术，成为解决这一困境的有希望的方法。本文全面概述了知识蒸馏范式，强调了其基本原理，如软标签的实用性和温度缩放的重要性。通过细致的研究，我们阐明了成功蒸馏的关键因素，包括学生模型的架构、教师的水平以及超参数的平衡。同时我们还深入探讨了这一过程中的复杂性和挑战。我们的探索凸显了知识蒸馏作为一个重要转折点的潜力。

    The burgeoning complexity of contemporary deep learning models, while achieving unparalleled accuracy, has inadvertently introduced deployment challenges in resource-constrained environments. Knowledge distillation, a technique aiming to transfer knowledge from a high-capacity "teacher" model to a streamlined "student" model, emerges as a promising solution to this dilemma. This paper provides a comprehensive overview of the knowledge distillation paradigm, emphasizing its foundational principles such as the utility of soft labels and the significance of temperature scaling. Through meticulous examination, we elucidate the critical determinants of successful distillation, including the architecture of the student model, the caliber of the teacher, and the delicate balance of hyperparameters. While acknowledging its profound advantages, we also delve into the complexities and challenges inherent in the process. Our exploration underscores knowledge distillation's potential as a pivotal 
    
[^134]: FedL2P: 分布式学习的个性化联邦学习

    FedL2P: Federated Learning to Personalize. (arXiv:2310.02420v1 [cs.LG])

    [http://arxiv.org/abs/2310.02420](http://arxiv.org/abs/2310.02420)

    本论文介绍了一种名为FedL2P的联邦学习算法，通过学习个性化策略的元网络，实现了在不同联邦学习问题上的个性化学习，并取得了良好的性能。

    

    联邦学习研究在开发用于全局模型的分布式学习算法以及用于根据每个客户端的本地数据分布进行本地个性化的算法方面取得了进展。然而，不同的联邦学习问题可能需要不同的个性化策略，甚至可能无法为所有客户端定义一种有效的通用个性化策略：根据每个客户端的最优预测器与全局模型的相似程度，可能首选不同的个性化策略。在本文中，我们考虑了学习个性化策略的联邦元学习问题。具体而言，我们考虑了通过本地数据统计给出每个客户端的批量归一化和学习率参数的元网络。通过在联邦学习中学习这些元网络，我们允许整个联邦学习网络合作学习为每个客户端定制个性化策略。实证结果表明，我们提出的FedL2P方法可以在各种联邦学习问题上获得良好的性能。

    Federated learning (FL) research has made progress in developing algorithms for distributed learning of global models, as well as algorithms for local personalization of those common models to the specifics of each client's local data distribution. However, different FL problems may require different personalization strategies, and it may not even be possible to define an effective one-size-fits-all personalization strategy for all clients: depending on how similar each client's optimal predictor is to that of the global model, different personalization strategies may be preferred. In this paper, we consider the federated meta-learning problem of learning personalization strategies. Specifically, we consider meta-nets that induce the batch-norm and learning rate parameters for each client given local data statistics. By learning these meta-nets through FL, we allow the whole FL network to collaborate in learning a customized personalization strategy for each client. Empirical results s
    
[^135]: 充分测试时间适应的技巧

    Bag of Tricks for Fully Test-Time Adaptation. (arXiv:2310.02416v1 [cs.LG])

    [http://arxiv.org/abs/2310.02416](http://arxiv.org/abs/2310.02416)

    本研究提出了一种充分测试时间适应（TTA）的技巧，包括小批量归一化、流平衡、可靠样本选择和网络置信度校准。通过对这些技术的详细分析，我们揭示了它们在准确性、计算能力和模型复杂性之间的权衡，并取得了新的最先进的结果。

    

    充分测试时间适应（TTA）旨在适应数据漂移，最近引起了广泛关注。已经提出了许多技巧和技术，以确保在任意未标记数据流上进行稳健学习。然而，评估每个个体技术的真正影响并进行公平比较仍然是一个重大挑战。为了帮助巩固社区知识，我们提出了对所选正交TTA技术的分类，包括小批量归一化、流平衡、可靠样本选择和网络置信度校准。我们详细分析了每种方法在不同感兴趣的场景下的影响。通过我们的分析，我们揭示了这些技术在准确性、所需计算能力和模型复杂性之间产生的权衡。我们还发现了结合技术时产生的协同效应，并能够建立新的最先进结果。

    Fully Test-Time Adaptation (TTA), which aims at adapting models to data drifts, has recently attracted wide interest. Numerous tricks and techniques have been proposed to ensure robust learning on arbitrary streams of unlabeled data. However, assessing the true impact of each individual technique and obtaining a fair comparison still constitutes a significant challenge. To help consolidate the community's knowledge, we present a categorization of selected orthogonal TTA techniques, including small batch normalization, stream rebalancing, reliable sample selection, and network confidence calibration. We meticulously dissect the effect of each approach on different scenarios of interest. Through our analysis, we shed light on trade-offs induced by those techniques between accuracy, the computational power required, and model complexity. We also uncover the synergy that arises when combining techniques and are able to establish new state-of-the-art results.
    
[^136]: 量化专家混合（MoQE）：低位量化和鲁棒性的互补效应

    Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness. (arXiv:2310.02410v1 [cs.LG])

    [http://arxiv.org/abs/2310.02410](http://arxiv.org/abs/2310.02410)

    本文提出了一种量化专家混合（MoQE）方法，通过将专家权重应用2位低位量化，减轻了大规模专家混合（MoE）模型在内存消耗和延迟问题上的压力，同时在大多数情况下不需要额外训练也能保持可靠的模型性能。

    

    大规模的专家混合（MoE）模型在各种语言任务中，包括机器翻译任务，通过专家并行性的高效模型扩展能力，实现了最先进的质量。然而，在部署时，这也带来了更大的内存消耗和增加的内存带宽瓶颈的基本问题。在本文中，我们提出了量化专家混合（MoQE），这是一种简单的仅将专家权重应用于超低位2位量化的量化方法，以减轻MoE模型的增大内存和延迟问题。我们展示了低位量化与MoE架构结合，即使在大多数情况下不需要任何额外的训练，也能提供可靠的模型性能，并且显著减小内存大小。特别地，MoE模型中的专家层对量化比传统的前馈网络（FFN）层更具鲁棒性。在我们的综合分析中，我们展示了MoE模型...

    Large Mixture of Experts (MoE) models could achieve state-of-the-art quality on various language tasks, including machine translation task, thanks to the efficient model scaling capability with expert parallelism. However, it has brought a fundamental issue of larger memory consumption and increased memory bandwidth bottleneck at deployment time. In this paper, we propose Mixture of Quantized Experts (MoQE) which is a simple weight-only quantization method applying ultra low-bit down to 2-bit quantizations only to expert weights for mitigating the increased memory and latency issues of MoE models. We show that low-bit quantization together with the MoE architecture delivers a reliable model performance while reducing the memory size significantly even without any additional training in most cases. In particular, expert layers in MoE models are much more robust to the quantization than conventional feedforward networks (FFN) layers. In our comprehensive analysis, we show that MoE models
    
[^137]: Nugget 2D：用于仅解码器语言模型的动态上下文压缩的扩展

    Nugget 2D: Dynamic Contextual Compression for Scaling Decoder-only Language Models. (arXiv:2310.02409v1 [cs.CL])

    [http://arxiv.org/abs/2310.02409](http://arxiv.org/abs/2310.02409)

    Nugget 2D是一种用于仅解码器语言模型的动态上下文压缩方法，可以在保留任务能力的同时大幅减少解码过程所需的时间和空间开销。

    

    标准的基于Transformer的语言模型在长上下文中缩放效果不佳。我们提出了一种基于动态上下文压缩的解决方案，该方案将Qin＆Van Durme（2023年）的Nugget方法从BERT类框架扩展到仅解码器的语言模型。我们的方法将历史建模为压缩的“nuggets”，这些“nuggets”经过训练可以进行重建，它可以使用诸如LLaMA之类的现成模型进行初始化。我们通过语言建模、问答和摘要的实验证明，Nugget2D在这些任务中保留了能力，同时在解码过程中大幅减少了时间和空间开销。例如，在自动编码实验中，Nugget2D可以以20倍的压缩比收缩上下文，重建时的BLEU得分为98％，实现了近乎无损编码。

    Standard Transformer-based language models (LMs) scale poorly to long contexts. We propose a solution based on dynamic contextual compression, which extends the Nugget approach of Qin & Van Durme (2023) from BERT-like frameworks to decoder-only LMs. Our method models history as compressed "nuggets" which are trained to allow for reconstruction, and it can be initialized with off-the-shelf models such as LLaMA. We demonstrate through experiments in language modeling, question answering, and summarization that Nugget2D retains capabilities in these tasks, while drastically reducing the overhead during decoding in terms of time and space. For example, in the experiments of autoencoding, Nugget2D can shrink context at a 20x compression ratio with a BLEU score of 98% for reconstruction, achieving nearly lossless encoding.
    
[^138]: 在大型语言模型时代的自动缺陷生成

    Automated Bug Generation in the era of Large Language Models. (arXiv:2310.02407v1 [cs.SE])

    [http://arxiv.org/abs/2310.02407](http://arxiv.org/abs/2310.02407)

    本论文探讨了在大型语言模型时代的自动缺陷生成问题，针对难以检测和难以修复的缺陷提出了解决方案，并分析了基于学习的技术中这两个目标的冲突。

    

    缺陷在软件工程中是至关重要的；过去几十年的许多研究已经提出了检测、定位和修复软件系统中的缺陷的方法。评估这些技术的有效性需要复杂的缺陷，即那些很难通过测试和调试来检测和修复的缺陷。从传统软件工程的角度来看，难以修复的缺陷与正确的代码在多个位置上有所差异，这使得它们难以定位和修复。而难以检测的缺陷则在特定的测试输入和可达条件下展现出来。这两个目标，即生成难以检测和难以修复的缺陷，大多数是一致的；缺陷生成技术可以将多个语句更改为仅在特定输入集合下被覆盖。然而，对于基于学习的技术来说，这两个目标是相互冲突的：一个缺陷应该有与训练数据中的正确代码相似的代码表示，以挑战缺陷预测。

    Bugs are essential in software engineering; many research studies in the past decades have been proposed to detect, localize, and repair bugs in software systems. Effectiveness evaluation of such techniques requires complex bugs, i.e., those that are hard to detect through testing and hard to repair through debugging. From the classic software engineering point of view, a hard-to-repair bug differs from the correct code in multiple locations, making it hard to localize and repair. Hard-to-detect bugs, on the other hand, manifest themselves under specific test inputs and reachability conditions. These two objectives, i.e., generating hard-to-detect and hard-to-repair bugs, are mostly aligned; a bug generation technique can change multiple statements to be covered only under a specific set of inputs. However, these two objectives are conflicting for learning-based techniques: A bug should have a similar code representation to the correct code in the training data to challenge a bug predi
    
[^139]: PCGPT：利用转换器进行程序化内容生成

    PCGPT: Procedural Content Generation via Transformers. (arXiv:2310.02405v1 [cs.LG])

    [http://arxiv.org/abs/2310.02405](http://arxiv.org/abs/2310.02405)

    本文介绍了PCGPT框架，它利用离线强化学习和Transformer网络进行程序化内容生成。该框架通过解决传统PCG方法的挑战，生成了更复杂和多样化的游戏内容，并且在更少的步骤中实现了这些结果。

    

    该论文介绍了PCGPT框架，这是一种使用离线强化学习和Transformer网络进行程序化内容生成(PCG)的创新方法。 PCGPT利用基于Transformer的自回归模型迭代生成游戏关卡，解决了传统PCG方法中重复、可预测或不一致内容的挑战。该框架通过捕捉时间依赖性和因果关系的Transformer自注意机制，对动作、状态和奖励的轨迹进行建模。该方法在Sokoban益智游戏中进行了评估，模型预测所需物品及其相应位置。在Sokoban游戏的实验结果表明，PCGPT生成了更复杂和多样化的游戏内容。有趣的是，与现有方法相比，它以显著较少的步骤实现了这些结果，展示了改进游戏设计和在线内容生成的潜力。

    The paper presents the PCGPT framework, an innovative approach to procedural content generation (PCG) using offline reinforcement learning and transformer networks. PCGPT utilizes an autoregressive model based on transformers to generate game levels iteratively, addressing the challenges of traditional PCG methods such as repetitive, predictable, or inconsistent content. The framework models trajectories of actions, states, and rewards, leveraging the transformer's self-attention mechanism to capture temporal dependencies and causal relationships. The approach is evaluated in the Sokoban puzzle game, where the model predicts items that are needed with their corresponding locations. Experimental results on the game Sokoban demonstrate that PCGPT generates more complex and diverse game content. Interestingly, it achieves these results in significantly fewer steps compared to existing methods, showcasing its potential for enhancing game design and online content generation. Our model repr
    
[^140]: 关于随机梯度下降中多层蒙特卡洛的并行复杂性

    On the Parallel Complexity of Multilevel Monte Carlo in Stocahstic Gradient Descent. (arXiv:2310.02402v1 [cs.LG])

    [http://arxiv.org/abs/2310.02402](http://arxiv.org/abs/2310.02402)

    本文提出了一种延迟MLMC梯度估计器，通过重复利用之前步骤中计算过的梯度分量，大大降低了MLMC的并行复杂性，并在数值实验中证明了其在随机梯度下降中具有更好的并行复杂性。

    

    在用于顺序模拟（如神经随机微分方程）的随机梯度下降（SGD）中，多层蒙特卡洛（MLMC）方法已被证明在理论计算复杂性方面优于朴素的蒙特卡洛方法。然而在实践中，MLMC在现代GPU等大规模并行计算平台上的可扩展性较差，因为其并行复杂性与朴素蒙特卡洛方法相当。为了解决这个问题，我们提出了延迟MLMC梯度估计器，通过重复利用之前步骤中计算过的梯度分量，大大降低MLMC的并行复杂性。所提出的估计器在每次迭代中能够证明降低平均并行复杂性，但代价是稍差的收敛速率。在我们的数值实验中，我们使用深度对冲的示例来证明与标准MLMC在SGD中相比，我们的方法具有更好的并行复杂性。

    In the stochastic gradient descent (SGD) for sequential simulations such as the neural stochastic differential equations, the Multilevel Monte Carlo (MLMC) method is known to offer better theoretical computational complexity compared to the naive Monte Carlo approach. However, in practice, MLMC scales poorly on massively parallel computing platforms such as modern GPUs, because of its large parallel complexity which is equivalent to that of the naive Monte Carlo method. To cope with this issue, we propose the delayed MLMC gradient estimator that drastically reduces the parallel complexity of MLMC by recycling previously computed gradient components from earlier steps of SGD. The proposed estimator provably reduces the average parallel complexity per iteration at the cost of a slightly worse per-iteration convergence rate. In our numerical experiments, we use an example of deep hedging to demonstrate the superior parallel complexity of our method compared to the standard MLMC in SGD.
    
[^141]: 减少人和小鼠脑外伤 EEG 数据中种内和种间协变量偏移的转移欧几里德对齐方法

    Reducing Intraspecies and Interspecies Covariate Shift in Traumatic Brain Injury EEG of Humans and Mice Using Transfer Euclidean Alignment. (arXiv:2310.02398v1 [cs.LG])

    [http://arxiv.org/abs/2310.02398](http://arxiv.org/abs/2310.02398)

    本文介绍了一种转移学习技术，以应对在睡眠脑电图（EEG）分析中由于个体间的高变异性而导致机器学习模型在不同数据集上的表现差异。该技术可以通过转移欧几里德对齐来解决缺乏高质量人类生物医学数据的问题，并在多种经典机器学习模型和深度学习模型上展现出鲁棒性。

    

    虽然睡眠脑电图（EEG）分析在临床应用中具有某些优势，但不同受试者之间的高变异性在部署机器学习模型用于分类任务时提出了重要挑战。在这种情况下，对于相同任务，在特定数据集上表现出卓越性能的机器学习模型未必在应用于不同数据集时表现出类似的熟练程度。高质量生物医学数据的稀缺性进一步加剧了这个挑战，使得全面评估模型的普适性变得困难。本文介绍了一种转移学习技术“转移欧几里德对齐”，用于解决缺乏人类生物医学数据以训练深度学习模型的问题。我们通过评估多种基于规则的经典机器学习模型以及基于 EEGNet 的深度学习模型，测试了这种转移学习技术的鲁棒性。

    While analytics of sleep electroencephalography (EEG) holds certain advantages over other methods in clinical applications, high variability across subjects poses a significant challenge when it comes to deploying machine learning models for classification tasks in the real world. In such instances, machine learning models that exhibit exceptional performance on a specific dataset may not necessarily demonstrate similar proficiency when applied to a distinct dataset for the same task. The scarcity of high-quality biomedical data further compounds this challenge, making it difficult to evaluate the model's generality comprehensively. In this paper, we introduce Transfer Euclidean Alignment - a transfer learning technique to tackle the problem of the dearth of human biomedical data for training deep learning models. We tested the robustness of this transfer learning technique on various rule-based classical machine learning models as well as the EEGNet-based deep learning model by evalua
    
[^142]: 用过参数化的神经网络中的隐式正则化方法进行多任务学习和微调

    Implicit regularization of multi-task learning and finetuning in overparameterized neural networks. (arXiv:2310.02396v1 [cs.LG])

    [http://arxiv.org/abs/2310.02396](http://arxiv.org/abs/2310.02396)

    本文研究了在过参数化神经网络中，多任务学习和微调所带来的隐式正则化效果。在简化的线性网络环境中，我们发现了多任务学习和微调所对特征共享和学习特定特征稀疏性的鼓励作用，并发现微调过程同时具有内核和特征学习的混合状态。此外，微调还可以展现一种嵌套特征学习行为，使其偏向于提取一组稀疏的特征子集。

    

    在深度学习中，常常使用训练辅助任务的方法来期望学习可以部分地转移到其他感兴趣的任务上。本研究探讨了学习辅助任务所产生的归纳偏置，包括同时学习（多任务学习，MTL）和依序学习（预训练和随后微调，PT+FT）。在使用梯度下降法训练两层对角线线性网络的简化环境中，我们发现了与MTL和PT+FT相关的隐式正则化惩罚，两者都鼓励任务之间的特征共享和学习任务特定特征的稀疏性。值得注意的是，我们的结果表明，在微调过程中，网络在先前研究中确定的内核（或“惰性”）状态和特征学习（“丰富”）状态之间具有混合状态。此外，PT+FT还可以展现一种新颖的“嵌套特征学习”行为，该行为无法被任何状态所捕捉，使其偏向于提取一组稀疏的特征子集。

    It is common in deep learning to train networks on auxiliary tasks with the expectation that the learning will transfer, at least partially, to another task of interest. In this work, we investigate the inductive biases that result from learning auxiliary tasks, either simultaneously (multi-task learning, MTL) or sequentially (pretraining and subsequent finetuning, PT+FT). In the simplified setting of two-layer diagonal linear networks trained with gradient descent, we identify implicit regularization penalties associated with MTL and PT+FT, both of which incentivize feature sharing between tasks and sparsity in learned task-specific features. Notably, our results imply that during finetuning, networks operate in a hybrid of the kernel (or "lazy") regime and the feature learning ("rich") regime identified in prior work. Moreover, PT+FT can exhibit a novel "nested feature learning" behavior not captured by either regime, which biases it to extract a sparse subset of the features learned
    
[^143]: SE(3)-蛋白质主链生成中的随机流匹配

    SE(3)-Stochastic Flow Matching for Protein Backbone Generation. (arXiv:2310.02391v1 [cs.LG])

    [http://arxiv.org/abs/2310.02391](http://arxiv.org/abs/2310.02391)

    通过SE(3)-Stochastic Flow Matching，我们提出了一系列新型生成模型FoldFlow，可以准确建模蛋白质主链。这些模型通过无需模拟训练和Riemannian最优传输的结合，具有更好的稳定性和建模能力。

    

    通过基于三维刚体运动（即SE(3)群）的流匹配范式，我们引入了一系列具有不断增强建模能力的新型生成模型：FoldFlow，从而实现了对蛋白质主链的准确建模。首先，我们介绍了FoldFlow-Base，一种无需模拟的学习确定性连续时间动力学和匹配不变目标分布的方法。接下来，我们通过引入Riemannian最优传输来加速训练，创建了FoldFlow-OT，从而构建了更简单和稳定的流。最后，我们设计了FoldFlow-SFM，将Riemannian最优传输和无需模拟训练相结合，可以学习SE(3)上的随机连续时间动力学。我们的FoldFlow生成模型家族相比之前的方法具有几个关键优势。

    The computational design of novel protein structures has the potential to impact numerous scientific disciplines greatly. Toward this goal, we introduce $\text{FoldFlow}$ a series of novel generative models of increasing modeling power based on the flow-matching paradigm over $3\text{D}$ rigid motions -i.e. the group $\text{SE(3)}$ -- enabling accurate modeling of protein backbones. We first introduce $\text{FoldFlow-Base}$, a simulation-free approach to learning deterministic continuous-time dynamics and matching invariant target distributions on $\text{SE(3)}$. We next accelerate training by incorporating Riemannian optimal transport to create $\text{FoldFlow-OT}$, leading to the construction of both more simple and stable flows. Finally, we design $\text{FoldFlow-SFM}$ coupling both Riemannian OT and simulation-free training to learn stochastic continuous-time dynamics over $\text{SE(3)}$. Our family of $\text{FoldFlow}$ generative models offer several key advantages over previous
    
[^144]: 机器学习的安全有效数据评估

    Secure and Effective Data Appraisal for Machine Learning. (arXiv:2310.02373v1 [cs.LG])

    [http://arxiv.org/abs/2310.02373](http://arxiv.org/abs/2310.02373)

    本文介绍了一种机密的数据选择和评估方法，通过创新的流程和简化的低维度操作来实现，以保护数据和模型的隐私，并在多个Transformer模型和NLP/CV基准测试中进行了评估。

    

    一个无拘无束的数据市场需要在数据所有者和模型所有者最终交易前能够对训练数据进行私密选择和评估。为了保护数据和模型的隐私，这个过程涉及使用多方计算(MPC)来审查目标模型。尽管之前的研究认为基于MPC的Transformer模型评估过于耗费资源，本文介绍了一种创新的方法，使数据选择成为可行的。本研究的贡献包括三个关键要素：(1)使用MPC进行机密数据选择的开创性流程；(2)通过在有限的相关数据子集上训练简化的低维度MLP来复制复杂的高维度操作；(3)并发、多阶段地实现MPC。所提出的方法在一系列Transformer模型和NLP/CV基准测试中进行了评估。与直接基于MPC的评估相比

    Essential for an unfettered data market is the ability to discreetly select and evaluate training data before finalizing a transaction between the data owner and model owner. To safeguard the privacy of both data and model, this process involves scrutinizing the target model through Multi-Party Computation (MPC). While prior research has posited that the MPC-based evaluation of Transformer models is excessively resource-intensive, this paper introduces an innovative approach that renders data selection practical. The contributions of this study encompass three pivotal elements: (1) a groundbreaking pipeline for confidential data selection using MPC, (2) replicating intricate high-dimensional operations with simplified low-dimensional MLPs trained on a limited subset of pertinent data, and (3) implementing MPC in a concurrent, multi-phase manner. The proposed method is assessed across an array of Transformer models and NLP/CV benchmarks. In comparison to the direct MPC-based evaluation 
    
[^145]: 从自动反馈中进行强化学习以生成高质量的单元测试

    Reinforcement Learning from Automatic Feedback for High-Quality Unit Test Generation. (arXiv:2310.02368v1 [cs.SE])

    [http://arxiv.org/abs/2310.02368](http://arxiv.org/abs/2310.02368)

    本论文提出了一种名为静态质量指标强化学习（RLSQM）的新技术，用于解决大型语言模型（LLM）在自动生成测试用例时可能生成不良代码异味的问题。通过训练特定的奖励模型和利用PPO算法进行优化，我们实现了对单个质量指标和整体质量的优化。

    

    软件测试是软件开发的关键方面，创建符合最佳实践的高质量测试对于有效的维护至关重要。最近，大型语言模型（LLM）在代码生成方面越来越受欢迎，包括自动创建测试用例。然而，这些LLM通常在大量公开可用的代码上进行训练，其中可能包含不符合最佳实践甚至包含测试代码异味（反模式）的测试用例。为了解决这个问题，我们提出了一种称为静态质量指标强化学习（RLSQM）的新技术。首先，我们分析了LLM生成的反模式，并展示了LLM可以生成不良的测试代码异味。因此，我们为每个静态质量指标训练了专门的奖励模型，然后利用Proximal Policy Optimization （PPO）来训练逐个优化单个质量指标的模型。此外，我们将这些奖励融合到一个统一的奖励模型中，以实现对整体质量的优化。

    Software testing is a crucial aspect of software development, and the creation of high-quality tests that adhere to best practices is essential for effective maintenance. Recently, Large Language Models (LLMs) have gained popularity for code generation, including the automated creation of test cases. However, these LLMs are often trained on vast amounts of publicly available code, which may include test cases that do not adhere to best practices and may even contain test smells (anti-patterns). To address this issue, we propose a novel technique called Reinforcement Learning from Static Quality Metrics (RLSQM). To begin, we analyze the anti-patterns generated by the LLM and show that LLMs can generate undesirable test smells. Thus, we train specific reward models for each static quality metric, then utilize Proximal Policy Optimization (PPO) to train models for optimizing a single quality metric at a time. Furthermore, we amalgamate these rewards into a unified reward model aimed at ca
    
[^146]: 基于密度估计的随机力推断

    Stochastic force inference via density estimation. (arXiv:2310.02366v1 [cs.LG])

    [http://arxiv.org/abs/2310.02366](http://arxiv.org/abs/2310.02366)

    该论文提出了一种基于密度估计的方法，通过概率流推断在分布之间插值的自主非线性力场，并应用于生物物理学中的转录组学，解决了从低分辨率的时间数据中推断动力学模型的挑战。

    

    在生物物理学中，从低分辨率的时间数据中推断动力学模型仍然是一个重要的挑战，特别是在转录组学中，将分子程序与噪声分离仍然是一个重要的待解决问题。我们探索了一种常见情况，即我们可以在少数时间点采样到足够数量的横截面样本，并假设我们的样本是从潜在的扩散过程中生成的。我们提出了一种方法，依赖于与潜在扩散过程相关的概率流，以推断在分布之间插值的自主非线性力场。在给定噪声模型的先验条件下，我们使用得分匹配法来区分力场与内在噪声。通过相关的生物物理实例，我们证明了我们的方法可以从非平稳数据中提取非保守力，并且当应用于稳态数据时学习平衡动力学，而且可以处理加法和乘法噪声。

    Inferring dynamical models from low-resolution temporal data continues to be a significant challenge in biophysics, especially within transcriptomics, where separating molecular programs from noise remains an important open problem. We explore a common scenario in which we have access to an adequate amount of cross-sectional samples at a few time-points, and assume that our samples are generated from a latent diffusion process. We propose an approach that relies on the probability flow associated with an underlying diffusion process to infer an autonomous, nonlinear force field interpolating between the distributions. Given a prior on the noise model, we employ score-matching to differentiate the force field from the intrinsic noise. Using relevant biophysical examples, we demonstrate that our approach can extract non-conservative forces from non-stationary data, that it learns equilibrium dynamics when applied to steady-state data, and that it can do so with both additive and multipli
    
[^147]: 关于自然语言处理中毒性定义的探讨

    On the definition of toxicity in NLP. (arXiv:2310.02357v1 [cs.CL])

    [http://arxiv.org/abs/2310.02357](http://arxiv.org/abs/2310.02357)

    这项研究探讨了毒性的定义模糊性问题，并提出了一种基于定量压力的毒性定义来弥补现有定义的缺点。

    

    毒性检测任务中的根本问题在于毒性的定义模糊不清。谷歌旗下的团队Jigsaw是该领域的领导者之一，他们使用Dixon等人给出的毒性定义：“粗鲁、不尊重或不合理的语言，可能会让某人离开讨论”。人们可以立即看到这个定义的问题，因为它没有给出毒性的定量度量，而且涉及高度主观的文化术语。尽管存在模糊和缺陷，但这个定义已经成为许多研究者广泛采用的实际标准。在这项工作中，我们提出了一种基于定量压力的毒性定义，克服了现有的缺点。

    The fundamental problem in toxicity detection task lies in the fact that the toxicity is ill-defined. Jigsaw, a unit within Google and one of the leaders in the field, uses a definition of toxicity given by Dixon et al. - 'rude, disrespectful, or unreasonable language that is likely to make someone leave a discussion'. One can instantly see the issue with this definition, as it gives no quantitative measure of the toxicity and operates with highly subjective cultural terms. Despite all vagueness and flaws, this definition is de-facto widely used by many researchers. In this work we suggest quantative stress-based defenition for the toxicity that overcomes existing shortcomings.
    
[^148]: 探究葡萄糖突发期间的速度偏差模式：一个分位数回归的方法

    Investigating Speed Deviation Patterns During Glucose Episodes: A Quantile Regression Approach. (arXiv:2310.02351v1 [stat.AP])

    [http://arxiv.org/abs/2310.02351](http://arxiv.org/abs/2310.02351)

    本研究利用分位数回归方法探究了糖尿病驾驶者在急性血糖情况下的速度行为模式，相较于以往的研究，该方法能够更好地捕捉到分布模式。

    

    鉴于糖尿病的日益普遍，人们对糖尿病对驾驶等日常功能的影响产生了重要兴趣。糖尿病控制的并发症包括低血糖和高血糖突发，这可能会影响驾驶所需的认知和精神运动功能。本文的目标是通过使用捕获分布模式的分布分析方法，确定糖尿病驾驶者在急性血糖情况下的速度行为模式，和正常血糖的糖尿病驾驶者或无糖尿病的对照驾驶者相比，以自然驾驶环境为背景。这项研究推动了以往的论文，前者主要关注的是通过平均速度来探索速度偏差模式的传统方法。

    Given the growing prevalence of diabetes, there has been significant interest in determining how diabetes affects instrumental daily functions, like driving. Complication of glucose control in diabetes includes hypoglycemic and hyperglycemic episodes, which may impair cognitive and psychomotor functions needed for safe driving. The goal of this paper was to determine patterns of diabetes speed behavior during acute glucose to drivers with diabetes who were euglycemic or control drivers without diabetes in a naturalistic driving environment. By employing distribution-based analytic methods which capture distribution patterns, our study advances prior literature that has focused on conventional approach of average speed to explore speed deviation patterns.
    
[^149]: 几乎等变量子神经网络用于图像中$p4m$群对称性

    Approximately Equivariant Quantum Neural Network for $p4m$ Group Symmetries in Images. (arXiv:2310.02323v1 [quant-ph])

    [http://arxiv.org/abs/2310.02323](http://arxiv.org/abs/2310.02323)

    本文提出了几乎等变量子神经网络（EquivQCNNs），针对图像中的平面$p4m$对称性进行图像分类。这种方法在保持优化性能的同时，通过将先验知识纳入模型中，提升了训练和泛化能力。

    

    量子神经网络（QNNs）被认为是一种可以在噪声存在的情况下在近期量子硬件上低深度高效模拟的量子算法之一。然而，它们的性能高度依赖于选择最合适的变分量子算法（VQAs）架构，问题无关的模型通常遇到可训练性和泛化能力的问题。作为解决方案，最近的研究探索了几何量子机器学习（GQML），使用了与给定数据集的底层对称性等变的QNNs。GQML通过将先验知识纳入到模型中来为模型添加归纳偏差，并在约束搜索空间的同时提高了优化性能。本文提出了针对平面$p4m$对称性的图像分类的等变量子卷积神经网络（EquivQCNNs）。我们展示了在不同数据集测试的结果。

    Quantum Neural Networks (QNNs) are suggested as one of the quantum algorithms which can be efficiently simulated with a low depth on near-term quantum hardware in the presence of noises. However, their performance highly relies on choosing the most suitable architecture of Variational Quantum Algorithms (VQAs), and the problem-agnostic models often suffer issues regarding trainability and generalization power. As a solution, the most recent works explore Geometric Quantum Machine Learning (GQML) using QNNs equivariant with respect to the underlying symmetry of the dataset. GQML adds an inductive bias to the model by incorporating the prior knowledge on the given dataset and leads to enhancing the optimization performance while constraining the search space. This work proposes equivariant Quantum Convolutional Neural Networks (EquivQCNNs) for image classification under planar $p4m$ symmetry, including reflectional and $90^\circ$ rotational symmetry. We present the results tested in diff
    
[^150]: 自学优化器（STOP）：递归自我改进的代码生成

    Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation. (arXiv:2310.02304v1 [cs.CL])

    [http://arxiv.org/abs/2310.02304](http://arxiv.org/abs/2310.02304)

    本文提出了一种自学优化器（STOP），通过递归自我改进的代码生成，使用融合了语言模型的脚手架程序来改进自身，从而生成性能更好的程序。

    

    最近几年的人工智能系统（例如思维树和程序辅助语言模型）取得了一些重要进展，通过提供一个“脚手架”程序来解决问题，该程序构建了多次调用语言模型以生成更好的输出。脚手架程序通常使用Python等编程语言编写。在这项工作中，我们使用了一个融合了语言模型的脚手架程序来改进自身。我们从一个种子“改进器”开始，通过多次查询语言模型并返回最佳解决方案，根据给定的效用函数来改进输入程序。然后，我们运行这个种子改进器来改进自身。在一系列细分任务中，得到的改进改进器生成的程序在性能上明显优于种子改进器。随后，我们对语言模型提出的各种自我改进策略进行了分析，包括波束搜索、遗传算法和模拟退火。由于语言模型本身没有改变，这并不是一种增长领域。

    Several recent advances in AI systems (e.g., Tree-of-Thoughts and Program-Aided Language Models) solve problems by providing a "scaffolding" program that structures multiple calls to language models to generate better outputs. A scaffolding program is written in a programming language such as Python. In this work, we use a language-model-infused scaffolding program to improve itself. We start with a seed "improver" that improves an input program according to a given utility function by querying a language model several times and returning the best solution. We then run this seed improver to improve itself. Across a small set of downstream tasks, the resulting improved improver generates programs with significantly better performance than its seed improver. Afterward, we analyze the variety of self-improvement strategies proposed by the language model, including beam search, genetic algorithms, and simulated annealing. Since the language models themselves are not altered, this is not fu
    
[^151]: 3D物理系统中学习对称性破缺的松弛八面体群卷积

    Relaxed Octahedral Group Convolution for Learning Symmetry Breaking in 3D Physical Systems. (arXiv:2310.02299v1 [cs.LG])

    [http://arxiv.org/abs/2310.02299](http://arxiv.org/abs/2310.02299)

    本文介绍了一种用于建模3D物理系统的松弛八面体群卷积技术，它可以在保持数据一致的最高等变性水平的同时，发现物理系统中微妙的对称性破缺因素。

    

    深度等价模型利用对称性提高样本效率和泛化性能。然而，在许多这些模型中，完美对称性的假设有时可能会限制性能，特别是当数据与这些对称性不完全一致时。因此，我们在本文中引入了用于建模3D物理系统的松弛八面体群卷积。这种灵活的卷积技术能够在保持与数据一致的最高等变性水平的同时，发现物理系统中微妙的对称性破缺因素。实证结果验证了我们的方法不仅可以揭示相变中的对称性破缺因素，还可以在流体超分辨率任务中实现卓越性能。

    Deep equivariant models use symmetries to improve sample efficiency and generalization. However, the assumption of perfect symmetry in many of these models can sometimes be restrictive, especially when the data does not perfectly align with such symmetries. Thus, we introduce relaxed octahedral group convolution for modeling 3D physical systems in this paper. This flexible convolution technique provably allows the model to both maintain the highest level of equivariance that is consistent with data and discover the subtle symmetry-breaking factors in the physical systems. Empirical results validate that our approach can not only provide insights into the symmetry-breaking factors in phase transitions but also achieves superior performance in fluid super-resolution tasks.
    
[^152]: 无监督复杂半二进制矩阵分解用于拟静止源激活序列恢复

    Unsupervised Complex Semi-Binary Matrix Factorization for Activation Sequence Recovery of Quasi-Stationary Sources. (arXiv:2310.02295v1 [cs.LG])

    [http://arxiv.org/abs/2310.02295](http://arxiv.org/abs/2310.02295)

    本文提出了一种无监督复杂半二进制矩阵分解算法，用于恢复拟静止源的激活序列。这种算法能够通过传感器数据提取个体激活序列，并对工业过程和制造系统的能源可持续性研究提供帮助。

    

    在推动可持续、具有韧性和以人为中心的工业的同时，工业5.0的三个支柱需要增加对工业过程和制造系统以及其能源可持续性的理解。理解的一个基本要素是知道系统何时运行，这对定位高能耗子系统和操作至关重要。然而，实践中常常缺乏这样的知识。可以通过传感器数据恢复激活状态。一些非侵入式传感器（加速度计、电流传感器等）获取包含多个执行器信息的混合信号。尽管它们监测的系统成本低廉，但需要额外的信号处理来提取个体激活序列。为此，稀疏回归技术可以提取序列数据中的主要动态。著名的字典学习算法在这方面已经被证明是有效的。本文考虑了一种算法来处理这个问题。

    Advocating for a sustainable, resilient and human-centric industry, the three pillars of Industry 5.0 call for an increased understanding of industrial processes and manufacturing systems, as well as their energy sustainability. One of the most fundamental elements of comprehension is knowing when the systems are operated, as this is key to locating energy intensive subsystems and operations. Such knowledge is often lacking in practice. Activation statuses can be recovered from sensor data though. Some non-intrusive sensors (accelerometers, current sensors, etc.) acquire mixed signals containing information about multiple actuators at once. Despite their low cost as regards the fleet of systems they monitor, additional signal processing is required to extract the individual activation sequences. To that end, sparse regression techniques can extract leading dynamics in sequential data. Notorious dictionary learning algorithms have proven effective in this regard. This paper considers di
    
[^153]: 无网格可微分编程与基于数据驱动策略的偏微分方程约束下的最优控制比较

    A Comparison of Mesh-Free Differentiable Programming and Data-Driven Strategies for Optimal Control under PDE Constraints. (arXiv:2310.02286v1 [cs.LG])

    [http://arxiv.org/abs/2310.02286](http://arxiv.org/abs/2310.02286)

    本研究对直接-伴随循环、物理感知神经网络和可微分编程进行了比较，发现在偏微分方程约束下的最优控制中，可微分编程是最有效的方法，并提供了条件下的使用指南。

    

    在深度学习和自动微分库的影响下，偏微分方程约束下的最优控制领域正在迅速变化。我们对直接-伴随循环(DAL)、物理感知神经网络(PINN)和可微分编程(DP)进行了全面比较，使用基于径向基函数的通用无网格可微分PDE求解器。在拉普拉斯和纳维-斯托克斯方程下，我们发现DP非常有效，因为它产生了最准确的梯度，即使DAL失败和PINN困难。此外，我们提供了一个详细的基准，突出了这些方法在哪些条件下可以有效使用。我们的工作为最优控制从业者提供了指南，并进一步连接了他们与深度学习社区。

    The field of Optimal Control under Partial Differential Equations (PDE) constraints is rapidly changing under the influence of Deep Learning and the accompanying automatic differentiation libraries. Novel techniques like Physics-Informed Neural Networks (PINNs) and Differentiable Programming (DP) are to be contrasted with established numerical schemes like Direct-Adjoint Looping (DAL). We present a comprehensive comparison of DAL, PINN, and DP using a general-purpose mesh-free differentiable PDE solver based on Radial Basis Functions. Under Laplace and Navier-Stokes equations, we found DP to be extremely effective as it produces the most accurate gradients; thriving even when DAL fails and PINNs struggle. Additionally, we provide a detailed benchmark highlighting the limited conditions under which any of those methods can be efficiently used. Our work provides a guide to Optimal Control practitioners and connects them further to the Deep Learning community.
    
[^154]: PASTA: 并行时空注意力与空间自相关门控用于细粒度人群流预测

    PASTA: PArallel Spatio-Temporal Attention with spatial auto-correlation gating for fine-grained crowd flow prediction. (arXiv:2310.02284v1 [cs.LG])

    [http://arxiv.org/abs/2310.02284](http://arxiv.org/abs/2310.02284)

    本文提出了一种名为PASTA的神经网络模型，通过细粒度地图中的历史人流的时空模式来预测未来全市范围内的人群流。这种方法包括空间自相关门控、多尺度残差块和时序注意力门控模块，能够有效捕捉细粒度地图中的不规则时空模式。

    

    理解城市中物体（如人类和车辆）的运动模式对于许多应用非常重要，包括城市规划和管理。本文提出了一种通过建模细粒度城市地图中历史人流的时空模式来预测未来全市范围内人群流的方法。我们引入了一种名为PASTA的新型神经网络，它有效地捕捉了细粒度地图中的不规则时空模式。我们方法中的新颖组件包括空间自相关门控、多尺度残差块和时序注意力门控模块。空间自相关门控采用空间统计的概念来识别不规则的空间区域。多尺度残差块负责处理细粒度地图中的多个范围空间依赖关系，而时序注意力门控则过滤掉不相关的时间信息。

    Understanding the movement patterns of objects (e.g., humans and vehicles) in a city is essential for many applications, including city planning and management. This paper proposes a method for predicting future city-wide crowd flows by modeling the spatio-temporal patterns of historical crowd flows in fine-grained city-wide maps. We introduce a novel neural network named PArallel Spatio-Temporal Attention with spatial auto-correlation gating (PASTA) that effectively captures the irregular spatio-temporal patterns of fine-grained maps. The novel components in our approach include spatial auto-correlation gating, multi-scale residual block, and temporal attention gating module. The spatial auto-correlation gating employs the concept of spatial statistics to identify irregular spatial regions. The multi-scale residual block is responsible for handling multiple range spatial dependencies in the fine-grained map, and the temporal attention gating filters out irrelevant temporal information
    
[^155]: 使用道路地形特征的共享权重多层感知机及其在车辆轨迹速度预测中的应用

    SWMLP: Shared Weight Multilayer Perceptron for Car Trajectory Speed Prediction using Road Topographical Features. (arXiv:2310.02282v1 [cs.LG])

    [http://arxiv.org/abs/2310.02282](http://arxiv.org/abs/2310.02282)

    本论文提出了一种独立于大量历史速度数据的速度预测方法，通过使用车辆轨迹的道路地形特征来拟合共享权重多层感知机学习模型，取得了显著的定性和定量改进，同时为交通分析的新方法设计提供了新的视角。

    

    尽管交通是一种大规模收集的数据，但通常只在特定区域可用。一个问题是，虽然有研究对这些数据给出了良好的结果，但这些地区的数据可能不足以充分描述全球其他地区的所有交通模式。为了解决这个问题，我们提出了一种独立于大量历史速度数据的速度预测方法。为了预测车辆的速度，我们使用车辆轨迹的道路地形特征来拟合一个共享权重多层感知机学习模型。我们的结果在定性和定量上都显著改进了标准回归分析。此外，该提出的框架为设计交通分析的新方法提供了新的视角。

    Although traffic is one of the massively collected data, it is often only available for specific regions. One concern is that, although there are studies that give good results for these data, the data from these regions may not be sufficiently representative to describe all the traffic patterns in the rest of the world. In quest of addressing this concern, we propose a speed prediction method that is independent of large historical speed data. To predict a vehicle's speed, we use the trajectory road topographical features to fit a Shared Weight Multilayer Perceptron learning model. Our results show significant improvement, both qualitative and quantitative, over standard regression analysis. Moreover, the proposed framework sheds new light on the way to design new approaches for traffic analysis.
    
[^156]: 实时客服呼叫中心对话中的端到端连续语音情感识别

    End-to-End Continuous Speech Emotion Recognition in Real-life Customer Service Call Center Conversations. (arXiv:2310.02281v1 [eess.AS])

    [http://arxiv.org/abs/2310.02281](http://arxiv.org/abs/2310.02281)

    本研究介绍了一个用于实时客服呼叫中心对话中连续语音情感识别的大规模数据集CusEmo，采用维度情感注释方法捕捉情感的微妙、复杂和连续性，并解决了应用于数据集时的挑战。

    

    在呼叫中心对话中进行语音情感识别已经成为评估客户和代理人之间互动质量的有价值工具。与受控实验室环境不同，实际对话发生在不受控制的环境下，并受到影响情感表达的情境因素的影响。本文介绍了我们构建大规模实时数据集（CusEmo）用于客户服务呼叫中心对话中连续语音情感识别的方法。我们采用维度情感注释方法捕捉实际呼叫中心对话中情感的微妙、复杂和连续性，并对情境信息进行了注释。研究还解决了将端到端（E2E）语音情感识别系统应用于数据集时遇到的挑战，包括确定适当的标签采样率和输入片段长度，以及整合情境信息（对话者的性别）。

    Speech Emotion recognition (SER) in call center conversations has emerged as a valuable tool for assessing the quality of interactions between clients and agents. In contrast to controlled laboratory environments, real-life conversations take place under uncontrolled conditions and are subject to contextual factors that influence the expression of emotions. In this paper, we present our approach to constructing a large-scale reallife dataset (CusEmo) for continuous SER in customer service call center conversations. We adopted the dimensional emotion annotation approach to capture the subtlety, complexity, and continuity of emotions in real-life call center conversations, while annotating contextual information. The study also addresses the challenges encountered during the application of the End-to-End (E2E) SER system to the dataset, including determining the appropriate label sampling rate and input segment length, as well as integrating contextual information (interlocutor's gender 
    
[^157]: 专家增强的基于动态时间规整的异常检测

    Expert enhanced dynamic time warping based anomaly detection. (arXiv:2310.02280v1 [cs.LG])

    [http://arxiv.org/abs/2310.02280](http://arxiv.org/abs/2310.02280)

    本文提出了一种名为E-DTWA的新型异常检测方法，基于动态时间规整（DTW）并加入了人机交互概念相关的额外改进。该方法具有高效的检测能力，能够在强烈考虑到专家的检测反馈的基础上灵活地进行重新训练，同时保持低计算和空间复杂度。

    

    动态时间规整（DTW）是一个用于时间序列弹性相似性度量的著名算法。其处理非线性时间扭曲的能力使其在各种数据挖掘任务中很有帮助。其中一个任务是异常检测，它试图揭示出意外的行为而没有错误的检测警报。在本文中，我们提出了一种名为专家增强的动态时间规整异常检测（E-DTWA）的新型异常检测方法。它基于DTW，并在其中加入了与人机交互概念相关的额外改进。我们方法的主要优势包括高效的检测，在强烈考虑到专家的检测反馈的基础上灵活地进行重新训练，同时保持低计算和空间复杂度。

    Dynamic time warping (DTW) is a well-known algorithm for time series elastic dissimilarity measure. Its ability to deal with non-linear time distortions makes it helpful in variety of data mining tasks. Such a task is also anomaly detection which attempts to reveal unexpected behaviour without false detection alarms. In this paper, we propose a novel anomaly detection method named Expert enhanced dynamic time warping anomaly detection (E-DTWA). It is based on DTW with additional enhancements involving human-in-the-loop concept. The main benefits of our approach comprise efficient detection, flexible retraining based on strong consideration of the expert's detection feedback while retaining low computational and space complexity.
    
[^158]: 一致性轨迹模型：学习扩散的概率流ODE轨迹

    Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion. (arXiv:2310.02279v1 [cs.LG])

    [http://arxiv.org/abs/2310.02279](http://arxiv.org/abs/2310.02279)

    提出了一种一致性轨迹模型（CTM），它可以加速扩散模型的采样，同时通过对抗训练和去噪得分匹配损失的组合来提高性能，并实现了最先进的采样质量。

    

    一致性模型（CM）加速基于得分的扩散模型采样，但以牺牲样本质量为代价，缺乏一种自然的方法来权衡速度和质量。为了解决这个限制，我们提出了一致性轨迹模型（CTM），它是包括CM和基于得分模型在内的泛化模型。CTM训练一个单一的神经网络，可以在单次前向传递中输出得分（即对数密度的梯度），并允许在扩散过程中任意初始和最终时间之间进行不受限制的遍历概率流普通微分方程（ODE）。CTM利用对抗训练和去噪得分匹配损失的有效组合来提高性能，并在CIFAR-10（FID 1.73）和64X64分辨率的ImageNet上实现新的最先进FID。CTM还实现了一系列新的采样方案，包括确定性和随机的ODE解中的长跳跃。

    Consistency Models (CM) (Song et al., 2023) accelerate score-based diffusion model sampling at the cost of sample quality but lack a natural way to trade-off quality for speed. To address this limitation, we propose Consistency Trajectory Model (CTM), a generalization encompassing CM and score-based models as special cases. CTM trains a single neural network that can -- in a single forward pass -- output scores (i.e., gradients of log-density) and enables unrestricted traversal between any initial and final time along the Probability Flow Ordinary Differential Equation (ODE) in a diffusion process. CTM enables the efficient combination of adversarial training and denoising score matching loss to enhance performance and achieves new state-of-the-art FIDs for single-step diffusion model sampling on CIFAR-10 (FID 1.73) and ImageNet at 64X64 resolution (FID 2.06). CTM also enables a new family of sampling schemes, both deterministic and stochastic, involving long jumps along the ODE soluti
    
[^159]: "垃圾DNA假设：通过稀疏性对LLM预训练权重进行任务中心角度分析"

    Junk DNA Hypothesis: A Task-Centric Angle of LLM Pre-trained Weights through Sparsity. (arXiv:2310.02277v1 [cs.LG])

    [http://arxiv.org/abs/2310.02277](http://arxiv.org/abs/2310.02277)

    本文研究通过稀疏性分析LLM预训练权重的任务中心角度，挑战了传统对于权重中冗余性的观点，并提出了"垃圾DNA假设"。

    

    传统对"垃圾DNA"的概念长期以来与人类基因组中的非编码片段相关联，占其组成的大约98%。然而，最近的研究揭示了一些这些看似无功能的DNA序列在细胞过程中起到的关键作用。有趣的是，深度神经网络中的权重与人类基因中观察到的冗余性有着显著的相似性。人们认为，庞大模型中的权重包含了过多的冗余，可以在不影响性能的情况下去除。本文通过提出一个令人信服的反论来挑战这个传统观点。我们使用稀疏性作为一种工具，来独立而准确地量化预训练大语言模型(LLM)中低幅度权重的细微重要性，从下游任务中心的角度理解它们包含的知识。我们提出了支持我们深入研究的"垃圾DNA假设"。

    The traditional notion of "Junk DNA" has long been linked to non-coding segments within the human genome, constituting roughly 98% of its composition. However, recent research has unveiled the critical roles some of these seemingly non-functional DNA sequences play in cellular processes. Intriguingly, the weights within deep neural networks exhibit a remarkable similarity to the redundancy observed in human genes. It was believed that weights in gigantic models contained excessive redundancy, and could be removed without compromising performance. This paper challenges this conventional wisdom by presenting a compelling counter-argument. We employ sparsity as a tool to isolate and quantify the nuanced significance of low-magnitude weights in pre-trained large language models (LLMs). Our study demonstrates a strong correlation between these weight magnitudes and the knowledge they encapsulate, from a downstream task-centric angle. we raise the "Junk DNA Hypothesis" backed by our in-depth
    
[^160]: 深度学习饱和非线性薛定谔方程中的孤子动力学和复杂势识别

    Deep learning soliton dynamics and complex potentials recognition for 1D and 2D PT-symmetric saturable nonlinear Schr\"odinger equations. (arXiv:2310.02276v1 [nlin.PS])

    [http://arxiv.org/abs/2310.02276](http://arxiv.org/abs/2310.02276)

    本文利用深度学习方法扩展了物理信息神经网络(PINNs)来学习1D和2D饱和非线性薛定谔方程中的孤子动力学和复杂势识别。该方法还能解决PT对称势函数的逆问题，并进行了与传播距离z相关的1D和2D PT对称势的识别。

    

    本文首先将物理信息神经网络(PINNs)扩展到学习光纤中具有两种基本的PT对称Scarf-II势和周期势的一维和二维饱和非线性薛定谔方程(SNLSEs)的数据驱动平稳和非平稳孤子。其次，研究了PT对称势函数的数据驱动逆问题，而不仅仅是一维和二维SNLSEs中的势参数。特别地，我们提出了一种改进的PINNs(mPINNs)方案，通过解决数据直接识别一维和二维SNLSEs的PT势函数。而且还利用mPINNs方法研究了关于传播距离z的一维和二维PT对称势的逆问题。我们还通过将PINNs应用于SNLSE的平稳方程来识别势函数。此外，在不同参数条件下比较了两种网络结构，以使预测的PT势能达到相似程度。

    In this paper, we firstly extend the physics-informed neural networks (PINNs) to learn data-driven stationary and non-stationary solitons of 1D and 2D saturable nonlinear Schr\"odinger equations (SNLSEs) with two fundamental PT-symmetric Scarf-II and periodic potentials in optical fibers. Secondly, the data-driven inverse problems are studied for PT-symmetric potential functions discovery rather than just potential parameters in the 1D and 2D SNLSEs. Particularly, we propose a modified PINNs (mPINNs) scheme to identify directly the PT potential functions of the 1D and 2D SNLSEs by the solution data. And the inverse problems about 1D and 2D PT -symmetric potentials depending on propagation distance z are also investigated using mPINNs method. We also identify the potential functions by the PINNs applied to the stationary equation of the SNLSE. Furthermore, two network structures are compared under different parameter conditions such that the predicted PT potentials can achieve the simil
    
[^161]: MuSe-GNN：从多模态生物图数据中学习统一的基因表示

    MuSe-GNN: Learning Unified Gene Representation From Multimodal Biological Graph Data. (arXiv:2310.02275v1 [cs.LG])

    [http://arxiv.org/abs/2310.02275](http://arxiv.org/abs/2310.02275)

    本研究引入了一种名为MuSe-GNN的模型，通过结合多模态机器学习和深度图神经网络，从单细胞测序和空间转录组数据中学习基因表示，并利用加权相似性学习和对比学习的正则化方法学习跨数据基因关系。该模型能够在一个共同的空间中提供包含不同背景下功能相似性的基因表示。

    

    在基因表示学习中，由于数据异质性，发现具有相似功能的基因在不同生物医学背景下的问题仍然具有挑战性。本研究引入了一种名为多模态相似性学习图神经网络（MuSe-GNN）的新模型来解决这个问题，该模型结合了多模态机器学习和深度图神经网络，从单细胞测序和空间转录组数据中学习基因表示。利用10个组织的82个训练数据集、三种测序技术和三个物种，我们创建了信息丰富的图结构用于模型训练和基因表示生成，并通过加权相似性学习和对比学习的正则化方法学习跨数据基因关系。这种新颖的设计确保我们可以在一个共同的空间中提供包含不同背景下功能相似性的基因表示。全面的基准分析表明我们的模型具有有效的性能。

    Discovering genes with similar functions across diverse biomedical contexts poses a significant challenge in gene representation learning due to data heterogeneity. In this study, we resolve this problem by introducing a novel model called Multimodal Similarity Learning Graph Neural Network, which combines Multimodal Machine Learning and Deep Graph Neural Networks to learn gene representations from single-cell sequencing and spatial transcriptomic data. Leveraging 82 training datasets from 10 tissues, three sequencing techniques, and three species, we create informative graph structures for model training and gene representations generation, while incorporating regularization with weighted similarity learning and contrastive learning to learn cross-data gene-gene relationships. This novel design ensures that we can offer gene representations containing functional similarity across different contexts in a joint space. Comprehensive benchmarking analysis shows our model's capacity to eff
    
[^162]: ARRQP: 具有图卷积的异常鲁棒实时QoS预测框架

    ARRQP: Anomaly Resilient Real-time QoS Prediction Framework with Graph Convolution. (arXiv:2310.02269v1 [cs.LG])

    [http://arxiv.org/abs/2310.02269](http://arxiv.org/abs/2310.02269)

    本研究介绍了一种名为ARRQP的实时QoS预测框架，重点改善了对数据中异常的鲁棒性，并利用图卷积技术来捕捉用户和服务之间复杂的关系和依赖。

    

    在现代面向服务的架构中，确保服务质量（QoS）非常重要。提前预测QoS值的能力使用户能够做出明智的决策。然而，在存在各种问题和异常情况（包括异常值、数据稀疏性、灰羊实例和冷启动场景）的情况下实现准确的QoS预测仍然是一个挑战。当前最先进的方法在同时解决这些问题时往往表现不佳，导致性能下降。在本文中，我们介绍了一个具有特别强调改善对数据中异常的鲁棒性的实时QoS预测框架（称为ARRQP）。ARRQP利用图卷积技术的力量来捕捉用户和服务之间复杂的关系和依赖，即使数据有限或稀疏。ARRQP整合了上下文信息和协作见解，实现了对用户-服务交互的全面理解。

    In the realm of modern service-oriented architecture, ensuring Quality of Service (QoS) is of paramount importance. The ability to predict QoS values in advance empowers users to make informed decisions. However, achieving accurate QoS predictions in the presence of various issues and anomalies, including outliers, data sparsity, grey-sheep instances, and cold-start scenarios, remains a challenge. Current state-of-the-art methods often fall short when addressing these issues simultaneously, resulting in performance degradation. In this paper, we introduce a real-time QoS prediction framework (called ARRQP) with a specific emphasis on improving resilience to anomalies in the data. ARRQP utilizes the power of graph convolution techniques to capture intricate relationships and dependencies among users and services, even when the data is limited or sparse. ARRQP integrates both contextual information and collaborative insights, enabling a comprehensive understanding of user-service interac
    
[^163]: CoNO: 复杂神经算子用于连续动力学系统

    CoNO: Complex Neural Operator for Continuous Dynamical Systems. (arXiv:2310.02094v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.02094](http://arxiv.org/abs/2310.02094)

    本文介绍了一种复杂神经算子（CoNO），用于解决连续动力学系统中的偏微分方程。该算子通过复分数傅里叶变换来捕获丰富的表示，并通过复值神经网络来提高表示能力、稳健性和泛化性能。

    

    神经算子扩展了数据驱动模型，用于映射无限维函数空间之间的关系。这些模型成功地解决了由微分方程表示的连续动力学系统，如天气预报、流体流动或固体力学。然而，现有的算子仍然依赖于实数空间，因此无法捕捉到在复数空间中通过函数变换可能捕捉到的丰富表示。在本文中，我们引入了一种复杂神经算子（CoNO），该算子在复分数傅里叶域中参数化积分核。另外，使用复值神经网络和无混叠激活函数的模型保留了复数值和复代数特性，从而提供了更好的表示能力、对噪声的稳健性和泛化性能。我们展示了该模型能够通过单一的复分数傅里叶变换有效地捕捉到潜在的偏微分方程。我们进行了大量的实证评估。

    Neural operators extend data-driven models to map between infinite-dimensional functional spaces. These models have successfully solved continuous dynamical systems represented by differential equations, viz weather forecasting, fluid flow, or solid mechanics. However, the existing operators still rely on real space, thereby losing rich representations potentially captured in the complex space by functional transforms. In this paper, we introduce a Complex Neural Operator (CoNO), that parameterizes the integral kernel in the complex fractional Fourier domain. Additionally, the model employing a complex-valued neural network along with aliasing-free activation functions preserves the complex values and complex algebraic properties, thereby enabling improved representation, robustness to noise, and generalization. We show that the model effectively captures the underlying partial differential equation with a single complex fractional Fourier transform. We perform an extensive empirical e
    
[^164]: 使用量子统计查询学习量子过程

    Learning Quantum Processes with Quantum Statistical Queries. (arXiv:2310.02075v1 [quant-ph] CROSS LISTED)

    [http://arxiv.org/abs/2310.02075](http://arxiv.org/abs/2310.02075)

    本文提出了第一个在量子统计查询模型内学习量子过程的框架，并提供了一个高效的学习器和可证明的性能保证。通过在密码分析中的应用，揭示了经典读出量子物理不可克隆函数的脆弱性，这是量子硬件安全领域一个重要的开放问题的解决方法。

    

    学习复杂的量子过程是量子计算和量子机器学习的许多领域的一个核心挑战，应用于量子基准测试、密码分析和变分量子算法。本文引入了第一个学习框架，用于在量子统计查询（QSQ）模型内研究量子过程学习，提供了对量子过程（QPSQs）进行统计查询的第一个正式定义。该框架使我们能够提出一种高效的QPSQ学习器，适用于任意量子过程，并附带可证明的性能保证。我们还提供了数值模拟来展示该算法的有效性。通过在密码分析中应用该框架，突出了经典读出量子物理不可克隆函数（CR-QPUFs）的脆弱性，解决了量子硬件安全领域中的一个重要开放问题。这项工作是朝着深入理解量子过程学习迈出的重要一步。

    Learning complex quantum processes is a central challenge in many areas of quantum computing and quantum machine learning, with applications in quantum benchmarking, cryptanalysis, and variational quantum algorithms. This paper introduces the first learning framework for studying quantum process learning within the Quantum Statistical Query (QSQ) model, providing the first formal definition of statistical queries to quantum processes (QPSQs). The framework allows us to propose an efficient QPSQ learner for arbitrary quantum processes accompanied by a provable performance guarantee. We also provide numerical simulations to demonstrate the efficacy of this algorithm. The practical relevance of this framework is exemplified through application in cryptanalysis, highlighting vulnerabilities of Classical-Readout Quantum Physical Unclonable Functions (CR-QPUFs), addressing an important open question in the field of quantum hardware security. This work marks a significant step towards underst
    
[^165]: OceanGPT：用于海洋科学任务的大型语言模型

    OceanGPT: A Large Language Model for Ocean Science Tasks. (arXiv:2310.02031v1 [cs.CL])

    [http://arxiv.org/abs/2310.02031](http://arxiv.org/abs/2310.02031)

    OceanGPT是首个专为海洋科学任务设计的大型语言模型，通过DoInstruct框架实现自动获取海洋领域指导数据。这一模型的引入填补了海洋科学领域中对LLM的需求缺口，并为海洋科学研究提供了新的工具和方法。

    

    海洋科学是探索充满生命和生物多样性的海洋的科学，考虑到海洋覆盖了地球表面的70％以上，这一领域具有重要意义。最近，大型语言模型（LLM）的进展改变了科学的范式。尽管在其他领域取得了成功，但现有的LLM通常无法满足海洋学家等领域专家的需求，同时对LLM在海洋科学中的潜力尚未得到充分探索。这其中的根本原因可能是海洋数据的庞大而复杂的性质，以及对更高的粒度和丰富的知识的需求。为了解决这些问题，我们推出了首个海洋领域的LLM——OceanGPT，该模型擅长各种海洋科学任务。我们提出了一个新颖的框架DoInstruct，用于自动获取大量的海洋领域指导数据，它基于多智能体的协作生成指导。

    Ocean science, which delves into the oceans that are reservoirs of life and biodiversity, is of great significance given that oceans cover over 70% of our planet's surface. Recently, advances in Large Language Models (LLMs) have transformed the paradigm in science. Despite the success in other domains, current LLMs often fall short in catering to the needs of domain experts like oceanographers, and the potential of LLMs for ocean science is under-explored. The intrinsic reason may be the immense and intricate nature of ocean data as well as the necessity for higher granularity and richness in knowledge. To alleviate these issues, we introduce OceanGPT, the first-ever LLM in the ocean domain, which is expert in various ocean science tasks. We propose DoInstruct, a novel framework to automatically obtain a large volume of ocean domain instruction data, which generates instructions based on multi-agent collaboration. Additionally, we construct the first oceanography benchmark, OceanBench,
    
[^166]: FiGURe: 使用过滤器增强的简单高效的无监督节点表示

    FiGURe: Simple and Efficient Unsupervised Node Representations with Filter Augmentations. (arXiv:2310.01892v1 [cs.LG])

    [http://arxiv.org/abs/2310.01892](http://arxiv.org/abs/2310.01892)

    本文介绍了一种简单的过滤器增强方法来改进无监督节点表示学习的性能，通过捕捉不同特征频谱部分，我们展示了显著的改进，并减少了计算负载。同时，我们通过使用简单的随机 Fourier 特征投影来解决高维表示的计算问题，并在基准数据集上取得了良好的性能。

    

    使用对比学习方法学习的无监督节点表示在下游任务上表现出良好的性能。然而，这些方法依赖于模拟低通滤波器的增强，限制了在需要不同特征频谱部分的任务上的性能。本文提出了一种简单的基于过滤器的增强方法来捕捉特征频谱的不同部分。我们展示了使用这些增强方法的显著改进。此外，我们展示了在这些不同的过滤器增强之间共享相同权重是可能的，从而减少了计算负载。此外，先前的研究表明，在下游任务上获得良好的性能需要高维表示。在处理高维度数据时，特别是当涉及多个增强方法时，增加了计算量。我们通过使用简单的随机 Fourier 特征投影来减轻这个问题并恢复良好的性能。我们的方法 FiGURe 在一些基准数据集上实现了

    Unsupervised node representations learnt using contrastive learning-based methods have shown good performance on downstream tasks. However, these methods rely on augmentations that mimic low-pass filters, limiting their performance on tasks requiring different eigen-spectrum parts. This paper presents a simple filter-based augmentation method to capture different parts of the eigen-spectrum. We show significant improvements using these augmentations. Further, we show that sharing the same weights across these different filter augmentations is possible, reducing the computational load. In addition, previous works have shown that good performance on downstream tasks requires high dimensional representations. Working with high dimensions increases the computations, especially when multiple augmentations are involved. We mitigate this problem and recover good performance through lower dimensional embeddings using simple random Fourier feature projections. Our method, FiGURe achieves an ave
    
[^167]: 有效且参数高效的重复使用微调模型

    Effective and Parameter-Efficient Reusing Fine-Tuned Models. (arXiv:2310.01886v1 [cs.LG])

    [http://arxiv.org/abs/2310.01886](http://arxiv.org/abs/2310.01886)

    本文提出了一种有效且参数高效的方法，可以重复使用微调模型来处理下游任务，减轻存储和服务负担，并提出了PERU-FFT方法用于重复使用全面微调模型。

    

    许多在线提供的预训练大规模模型在传递到下游任务中变得非常有效。与此同时，各种在这些预训练模型上微调的任务特定模型也可供公众使用。在实践中，由于收集任务特定数据耗时且微调大规模预训练模型计算复杂，可以重复使用任务特定微调模型来处理下游任务。然而，为每个任务使用一个模型会给存储和服务带来巨大负担。最近，有许多无需训练且参数高效的方法被提出，将多个微调的任务特定模型重复使用到一个多任务模型中。然而，与为每个任务使用微调模型相比，这些方法表现出较大的准确性差距。本文中，我们提出了参数高效方法来重复使用微调模型。针对重复使用全面微调模型，我们提出了PERU-FFT，通过将稀疏任务向量注入到一个mer模型中来实现。

    Many pre-trained large-scale models provided online have become highly effective in transferring to downstream tasks. At the same time, various task-specific models fine-tuned on these pre-trained models are available online for public use. In practice, as collecting task-specific data is labor-intensive and fine-tuning the large pre-trained models is computationally expensive, one can reuse task-specific finetuned models to deal with downstream tasks. However, using a model per task causes a heavy burden on storage and serving. Recently, many training-free and parameter-efficient methods have been proposed for reusing multiple fine-tuned task-specific models into a single multi-task model. However, these methods exhibit a large accuracy gap compared with using a fine-tuned model per task. In this paper, we propose Parameter-Efficient methods for ReUsing (PERU) fine-tuned models. For reusing Fully Fine-Tuned (FFT) models, we propose PERU-FFT by injecting a sparse task vector into a mer
    
[^168]: 融合模仿学习和强化学习以实现鲁棒策略改进

    Blending Imitation and Reinforcement Learning for Robust Policy Improvement. (arXiv:2310.01737v1 [cs.LG])

    [http://arxiv.org/abs/2310.01737](http://arxiv.org/abs/2310.01737)

    本文提出了一种融合模仿学习和强化学习的方法，根据在线评估结果交替使用二者，以提高样本效率和学习效果。

    

    虽然强化学习在性能上表现出色，但其样本复杂度仍然是一个重大障碍，限制了其在各个领域的广泛应用。模仿学习利用神经网络优化样本效率，但通常受到所使用的专家示范的质量限制。本文介绍了一种融合模仿学习和强化学习的方法，该方法根据在线评估结果交替使用二者，有效地提高了学习效率。这种算法能够从多种黑盒专家示范中学习和改进。

    While reinforcement learning (RL) has shown promising performance, its sample complexity continues to be a substantial hurdle, restricting its broader application across a variety of domains. Imitation learning (IL) utilizes oracles to improve sample efficiency, yet it is often constrained by the quality of the oracles deployed. which actively interleaves between IL and RL based on an online estimate of their performance. RPI draws on the strengths of IL, using oracle queries to facilitate exploration, an aspect that is notably challenging in sparse-reward RL, particularly during the early stages of learning. As learning unfolds, RPI gradually transitions to RL, effectively treating the learned policy as an improved oracle. This algorithm is capable of learning from and improving upon a diverse set of black-box oracles. Integral to RPI are Robust Active Policy Selection (RAPS) and Robust Policy Gradient (RPG), both of which reason over whether to perform state-wise imitation from the o
    
[^169]: SmartPlay: 一种用于评估LLMs作为智能Agent能力的基准

    SmartPlay : A Benchmark for LLMs as Intelligent Agents. (arXiv:2310.01557v1 [cs.LG])

    [http://arxiv.org/abs/2310.01557](http://arxiv.org/abs/2310.01557)

    SmartPlay是一个用于评估LLMs作为智能Agent能力的基准，包括6个具有不同挑战的游戏，并测试了智能LLM Agent的多种关键能力。这不仅是一个评估LLM Agent整体性能的严格测试场地，还可以分析每个能力的表现。

    

    最近的大型语言模型(LLMs)在智能Agent和下一代自动化方面展示了巨大的潜力，但目前缺乏一个系统化的基准来评估LLMs作为Agent的能力。我们介绍了SmartPlay：一个具有挑战性的基准和评估LLMs作为Agent的方法论。SmartPlay包括6个不同的游戏，包括剪刀石头布、汉诺塔、Minecraft等。每个游戏都具有独特的设置，提供最多20个评估设置和无限的环境变化。SmartPlay中的每个游戏都独特地挑战了智能LLM Agent的9个重要能力的子集，包括对对象依赖的推理、提前规划、空间推理、从历史中学习和理解随机性。每个游戏测试的能力集的区别使我们能够单独分析每个能力。SmartPlay不仅是评估LLM Agent整体性能的严格测试场地，而且也是评估Agent在不同能力方面的性能的一个重要工具。

    Recent large language models (LLMs) have demonstrated great potential toward intelligent agents and next-gen automation, but there currently lacks a systematic benchmark for evaluating LLMs' abilities as agents. We introduce SmartPlay: both a challenging benchmark and a methodology for evaluating LLMs as agents. SmartPlay consists of 6 different games, including Rock-Paper-Scissors, Tower of Hanoi, Minecraft. Each game features a unique setting, providing up to 20 evaluation settings and infinite environment variations. Each game in SmartPlay uniquely challenges a subset of 9 important capabilities of an intelligent LLM agent, including reasoning with object dependencies, planning ahead, spatial reasoning, learning from history, and understanding randomness. The distinction between the set of capabilities each game test allows us to analyze each capability separately. SmartPlay serves not only as a rigorous testing ground for evaluating the overall performance of LLM agents but also as
    
[^170]: PharmacoNet: 利用深度药物谱建模加速大规模虚拟筛选

    PharmacoNet: Accelerating Large-Scale Virtual Screening by Deep Pharmacophore Modeling. (arXiv:2310.00681v2 [q-bio.BM] UPDATED)

    [http://arxiv.org/abs/2310.00681](http://arxiv.org/abs/2310.00681)

    PharmacoNet是一个利用深度学习框架的虚拟筛选方法，通过识别最佳的三维药物谱排列来加速大规模虚拟筛选过程，相比于现有方法更快且准确性合理。

    

    随着可访问化合物库的规模扩大到超过100亿个，对更高效的基于结构的虚拟筛选方法的需求正在出现。尽管已经开发了不同的预筛选方法来快速筛选库，但适用于通用蛋白质的基于结构的方法仍然缺乏：挑战是在极短的时间内预测蛋白质和配体之间的结合位姿并进行评分。我们引入PharmacoNet，这是一个深度学习框架，它通过从结合位点生成的最佳三维药物谱排列来识别配体应该具有的稳定结合要求。通过粗粒化图匹配，我们在一步中解决了现有方法中昂贵的结合位姿采样和评分过程。PharmacoNet比最先进的基于结构的方法要快得多，但仍具有合理的准确性和简单的评分函数。此外，我们展示了有希望的再利用效果。

    As the size of accessible compound libraries expands to over 10 billion, the need for more efficient structure-based virtual screening methods is emerging. Different pre-screening methods have been developed to rapidly screen the library, but the structure-based methods applicable to general proteins are still lacking: the challenge is to predict the binding pose between proteins and ligands and perform scoring in an extremely short time. We introduce PharmacoNet, a deep learning framework that identifies the optimal 3D pharmacophore arrangement which a ligand should have for stable binding from the binding site. By coarse-grained graph matching between ligands and the generated pharmacophore arrangement, we solve the expensive binding pose sampling and scoring procedures of existing methods in a single step. PharmacoNet is significantly faster than state-of-the-art structure-based approaches, yet reasonably accurate with a simple scoring function. Furthermore, we show the promising re
    
[^171]: 学习类型推断以增强数据流分析

    Learning Type Inference for Enhanced Dataflow Analysis. (arXiv:2310.00673v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.00673](http://arxiv.org/abs/2310.00673)

    该论文介绍了一种学习类型推断的方法，以增强数据流分析。传统的类型推断在面对程序规模增长时面临性能挑战，而基于机器学习的统计技术可以提供更快的推断。然而，目前的方法在用户定义的类型上性能仍较差，限制了其在实际应用中的效果。

    

    静态分析动态类型代码是一项具有挑战性的任务，因为即使是看似微不足道的任务，比如确定过程调用的目标，在没有在编译时知道对象类型的情况下也是非常困难的。为了解决这个挑战，渐进式类型越来越多地添加到动态类型语言中，一个著名的例子是TypeScript，它为JavaScript引入了静态类型。渐进式类型提高了开发人员验证程序行为的能力，有助于创建稳健、安全和可调试的程序。然而，在实践中，用户很少直接注释类型。与此同时，传统的类型推断在程序规模增长时面临与性能相关的挑战。基于机器学习的统计技术提供了更快的推断，但尽管最近的方法在整体上表现出更高的准确性，但它们在用户定义的类型上的性能仍然明显较差，而在最常见的内置类型上的性能较好。这更限制了它们在真实世界中的实用性。

    Statically analyzing dynamically-typed code is a challenging endeavor, as even seemingly trivial tasks such as determining the targets of procedure calls are non-trivial without knowing the types of objects at compile time. Addressing this challenge, gradual typing is increasingly added to dynamically-typed languages, a prominent example being TypeScript that introduces static typing to JavaScript. Gradual typing improves the developer's ability to verify program behavior, contributing to robust, secure and debuggable programs. In practice, however, users only sparsely annotate types directly. At the same time, conventional type inference faces performance-related challenges as program size grows. Statistical techniques based on machine learning offer faster inference, but although recent approaches demonstrate overall improved accuracy, they still perform significantly worse on user-defined types than on the most common built-in types. Limiting their real-world usefulness even more, t
    
[^172]: SIMD数据流共优化用于CPU上高效的神经网络推理

    SIMD Dataflow Co-optimization for Efficient Neural Networks Inferences on CPUs. (arXiv:2310.00574v2 [cs.AR] UPDATED)

    [http://arxiv.org/abs/2310.00574](http://arxiv.org/abs/2310.00574)

    我们提出了一种通过共优化数据流和SIMD实现来高效地在CPU上进行神经网络推理的方法，实验结果表明，这种方法能够在保持准确性的同时大幅提升推理速度。

    

    我们针对在CPU上部署神经网络所面临的挑战提出了解决方案，特别关注的是在保持准确性的同时最小化推理时间。我们的新颖方法是利用神经网络的数据流（即计算顺序），通过启发式引导分析和代码生成框架来探索数据重用机会，从而实现各种单指令多数据（SIMD）实现以实现优化的神经网络执行。我们的结果表明，将输出保持在SIMD寄存器中的数据流同时最大化输入和权重重用，在各种推理工作负载下始终能够获得最佳性能，相比今天的神经网络优化实现，8位神经网络的加速比可达3倍，而二进制神经网络的加速比可达4.8倍。

    We address the challenges associated with deploying neural networks on CPUs, with a particular focus on minimizing inference time while maintaining accuracy. Our novel approach is to use the dataflow (i.e., computation order) of a neural network to explore data reuse opportunities using heuristic-guided analysis and a code generation framework, which enables exploration of various Single Instruction, Multiple Data (SIMD) implementations to achieve optimized neural network execution. Our results demonstrate that the dataflow that keeps outputs in SIMD registers while also maximizing both input and weight reuse consistently yields the best performance for a wide variety of inference workloads, achieving up to 3x speedup for 8-bit neural networks, and up to 4.8x speedup for binary neural networks, respectively, over the optimized implementations of neural networks today.
    
[^173]: 图神经网络能否作为最优近似算法？

    Are Graph Neural Networks Optimal Approximation Algorithms?. (arXiv:2310.00526v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.00526](http://arxiv.org/abs/2310.00526)

    本文设计了图神经网络架构OptGNN，利用半定规划工具获得大类组合优化问题的最优近似算法。通过实证结果表明在各种数据集上超过了神经网络基线算法和传统算法，同时利用OptGNN的能力设计了一个产生优化的对偶证书的算法。

    

    在这项工作中，我们设计了能够使用半定规划（SDP）强大的算法工具来获得大类组合优化问题的最优近似算法的图神经网络架构。具体而言，我们证明了多项式大小的消息传递算法可以表示最强大的多项式时间算法，前提是假设唯一游戏猜想成立。我们利用这一结果构建了高效的图神经网络架构OptGNN，它在诸如最大割和最大独立集等重要组合优化问题上获得了高质量的近似解。我们的方法在各种真实世界和合成数据集上表现出强大的实证结果，不仅超过了神经网络基线算法，还超过了传统算法。最后，我们利用OptGNN捕捉凸松弛的能力，设计了一个产生优化的对偶证书（确定性上界）的算法。

    In this work we design graph neural network architectures that can be used to obtain optimal approximation algorithms for a large class of combinatorial optimization problems using powerful algorithmic tools from semidefinite programming (SDP). Concretely, we prove that polynomial-sized message passing algorithms can represent the most powerful polynomial time algorithms for Max Constraint Satisfaction Problems assuming the Unique Games Conjecture. We leverage this result to construct efficient graph neural network architectures, OptGNN, that obtain high-quality approximate solutions on landmark combinatorial optimization problems such as Max Cut and maximum independent set. Our approach achieves strong empirical results across a wide range of real-world and synthetic datasets against both neural baselines and classical algorithms. Finally, we take advantage of OptGNN's ability to capture convex relaxations to design an algorithm for producing dual certificates of optimality (bounds on
    
[^174]: 结构对抗目标用于自监督表示学习

    Structural Adversarial Objectives for Self-Supervised Representation Learning. (arXiv:2310.00357v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.00357](http://arxiv.org/abs/2310.00357)

    通过结构对抗目标和平滑正则化器，该论文提出了一种自监督表示学习方法，可以在生成对抗网络中学习提取信息丰富的特征表示，而无需依赖手工数据增强方案。实验证明该方法在图像分类任务上取得了良好的效果。

    

    在生成对抗网络（GAN）的框架下，我们提出了一种通过额外的结构建模责任来使判别器在自监督表示学习中学习的目标。结合对网络施加的有效平滑正则化器，这些目标引导判别器学习提取信息丰富的表示，同时保持生成器能够从领域中进行采样。具体而言，我们的目标鼓励判别器在两个粒度级别上对特征进行结构化处理：在粗粒度上对齐分布特性（如均值和方差），在细粒度上对特征进行局部聚类。作为GAN框架中的特征学习器，我们的自监督系统不依赖于手工数据增强方案，这是对比表示学习方法中常见的。在CIFAR-10/100和ImageNet子集上的实验证明，我们的方法取得了良好的效果。

    Within the framework of generative adversarial networks (GANs), we propose objectives that task the discriminator for self-supervised representation learning via additional structural modeling responsibilities. In combination with an efficient smoothness regularizer imposed on the network, these objectives guide the discriminator to learn to extract informative representations, while maintaining a generator capable of sampling from the domain. Specifically, our objectives encourage the discriminator to structure features at two levels of granularity: aligning distribution characteristics, such as mean and variance, at coarse scales, and grouping features into local clusters at finer scales. Operating as a feature learner within the GAN framework frees our self-supervised system from the reliance on hand-crafted data augmentation schemes that are prevalent across contrastive representation learning methods. Across CIFAR-10/100 and an ImageNet subset, experiments demonstrate that equippi
    
[^175]: SpatialRank: 基于时空数据的城市事件排名与NDCG优化

    SpatialRank: Urban Event Ranking with NDCG Optimization on Spatiotemporal Data. (arXiv:2310.00270v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.00270](http://arxiv.org/abs/2310.00270)

    这篇论文提出了一种名为SpatialRank的新颖空间事件排名方法，通过基于时空数据的NDCG优化来解决城市事件排名问题。

    

    城市事件排名问题旨在预测未来事件（如交通事故和犯罪事件）的风险最高的前k个地点。这个问题对公共安全和城市管理非常重要，尤其是在资源有限的情况下。然而，由于地点之间复杂而动态的时空相关性，空间中城市事件的不均匀分布，以及正确对具有相似特征的附近地点进行排名的困难，这个问题很具挑战性。前人的研究主要旨在准确预测所有地点的实际风险得分或事件计数。由于预测错误，由此得到的排名通常质量较低。学习排序方法直接优化诸如标准化折扣累积增益（NDCG）之类的指标，但不能处理地点之间存在的时空自相关性。在本文中，我们通过提出一种名为SpatialRank的新颖空间事件排名方法来弥合这一差距。

    The problem of urban event ranking aims at predicting the top-k most risky locations of future events such as traffic accidents and crimes. This problem is of fundamental importance to public safety and urban administration especially when limited resources are available. The problem is, however, challenging due to complex and dynamic spatio-temporal correlations between locations, uneven distribution of urban events in space, and the difficulty to correctly rank nearby locations with similar features. Prior works on event forecasting mostly aim at accurately predicting the actual risk score or counts of events for all the locations. Rankings obtained as such usually have low quality due to prediction errors. Learning-to-rank methods directly optimize measures such as Normalized Discounted Cumulative Gain (NDCG), but cannot handle the spatiotemporal autocorrelation existing among locations. In this paper, we bridge the gap by proposing a novel spatial event ranking approach named Spati
    
[^176]: 通过最小移动方案实现神经网络的模块化训练

    Module-wise Training of Neural Networks via the Minimizing Movement Scheme. (arXiv:2309.17357v1 [cs.LG])

    [http://arxiv.org/abs/2309.17357](http://arxiv.org/abs/2309.17357)

    通过引入模块化正则化方法，解决了神经网络模块化训练中早期层过拟合和深层停滞的问题，实验结果展示了该方法在不同架构上的优越性。

    

    在内存有限的受限设备环境中，贪婪的逐层或逐模块训练神经网络可以绕过端到端反向传播的一些问题，因此具有吸引力。然而，这种方法存在停滞问题，早期层过拟合和更深层在一定深度后停止提高测试准确性。我们提出通过引入与分布空间中梯度流的最小化移动方法相启发的模块化正则化来解决这个问题。我们称这种方法为TRGL（Transport Regularized Greedy Learning），并对其进行了理论研究，证明它会导致模块化贪婪方法是规则的，并逐步解决任务。在实验中，我们展示了在添加我们的正则化方法之后，各种架构（如ResNets，Transformers和VGG）的模块化训练的准确性得到了改善，其优于其他模块化训练方法，甚至经常优于端到端训练，并且可以减少高达60%的内存使用。

    Greedy layer-wise or module-wise training of neural networks is compelling in constrained and on-device settings where memory is limited, as it circumvents a number of problems of end-to-end back-propagation. However, it suffers from a stagnation problem, whereby early layers overfit and deeper layers stop increasing the test accuracy after a certain depth. We propose to solve this issue by introducing a module-wise regularization inspired by the minimizing movement scheme for gradient flows in distribution space. We call the method TRGL for Transport Regularized Greedy Learning and study it theoretically, proving that it leads to greedy modules that are regular and that progressively solve the task. Experimentally, we show improved accuracy of module-wise training of various architectures such as ResNets, Transformers and VGG, when our regularization is added, superior to that of other module-wise training methods and often to end-to-end training, with as much as 60% less memory usage
    
[^177]: 高效的生物合理对抗训练

    Efficient Biologically Plausible Adversarial Training. (arXiv:2309.17348v1 [cs.LG])

    [http://arxiv.org/abs/2309.17348](http://arxiv.org/abs/2309.17348)

    本文研究了生物合理的学习算法是否比反向传播更具有对抗攻击的鲁棒性，并进行了广泛的比较分析。

    

    用反向传播训练的人工神经网络(ANNs)表现出令人惊讶的性能，并且越来越多地被用于执行我们日常生活中的任务。然而，ANNs极易受到对抗攻击的影响，这些攻击通过微小的有针对性的扰动来改变输入，从而严重破坏模型的性能。使ANNs对这些攻击具有鲁棒性最有效的方法是对抗训练，其中训练数据集被添加了样本用于对抗攻击。不幸的是，这种方法的缺点是增加了训练复杂性，因为生成对抗样本是非常计算消耗高的。与ANNs不同，人类不容易受到对抗攻击的影响。因此，在这项工作中，我们研究了生物合理的学习算法是否比BP更具有对抗攻击的鲁棒性。具体而言，我们对BP和“Error to Pertu"的对抗鲁棒性进行了广泛的比较分析。

    Artificial Neural Networks (ANNs) trained with Backpropagation (BP) show astounding performance and are increasingly often used in performing our daily life tasks. However, ANNs are highly vulnerable to adversarial attacks, which alter inputs with small targeted perturbations that drastically disrupt the models' performance. The most effective method to make ANNs robust against these attacks is adversarial training, in which the training dataset is augmented with exemplary adversarial samples. Unfortunately, this approach has the drawback of increased training complexity since generating adversarial samples is very computationally demanding. In contrast to ANNs, humans are not susceptible to adversarial attacks. Therefore, in this work, we investigate whether biologically-plausible learning algorithms are more robust against adversarial attacks than BP. In particular, we present an extensive comparative analysis of the adversarial robustness of BP and \textit{Present the Error to Pertu
    
[^178]: 一种用于医学图像中一般移动目标分割的基础模型

    A Foundation Model for General Moving Object Segmentation in Medical Images. (arXiv:2309.17264v1 [cs.CV])

    [http://arxiv.org/abs/2309.17264](http://arxiv.org/abs/2309.17264)

    本文提出了一种用于医学图像中移动目标分割的基础模型iMOS，通过对序列中只有少量图像进行注释，即可实现高精度的分割效果

    

    医学图像分割旨在描绘感兴趣的解剖或病理结构，在临床诊断中起着关键作用。构建高精度的深度分割模型需要大量高质量的注释数据。然而，医学注释非常繁琐耗时，特别是对于医学视频或3D体积，由于巨大的标签空间和差的帧间一致性。最近，在自然图像中，一个名为Moving Object Segmentation (MOS)的基本任务在技术上取得了重大进展。它的目标是在图像序列中从背景中描绘移动物体，只需要最小的注释。在本文中，我们提出了第一个用于医学图像中MOS的基础模型，名为iMOS。对一个大规模多模态医学数据集进行的大量实验验证了所提出的iMOS的有效性。具体而言，只需对序列中少量的图像进行注释，iMOS就可以实现了

    Medical image segmentation aims to delineate the anatomical or pathological structures of interest, playing a crucial role in clinical diagnosis. A substantial amount of high-quality annotated data is crucial for constructing high-precision deep segmentation models. However, medical annotation is highly cumbersome and time-consuming, especially for medical videos or 3D volumes, due to the huge labeling space and poor inter-frame consistency. Recently, a fundamental task named Moving Object Segmentation (MOS) has made significant advancements in natural images. Its objective is to delineate moving objects from the background within image sequences, requiring only minimal annotations. In this paper, we propose the first foundation model, named iMOS, for MOS in medical images. Extensive experiments on a large multi-modal medical dataset validate the effectiveness of the proposed iMOS. Specifically, with the annotation of only a small number of images in the sequence, iMOS can achieve sati
    
[^179]: 关于计算纠缠及其在对抗机器学习中的解释

    On Computational Entanglement and Its Interpretation in Adversarial Machine Learning. (arXiv:2309.15669v1 [cs.LG])

    [http://arxiv.org/abs/2309.15669](http://arxiv.org/abs/2309.15669)

    本研究探索了对抗机器学习模型的复杂性和可解释性，通过将其与爱因斯坦的特殊相对论中的纠缠概念联系起来，发现远程特征样本可以表现出纠缠现象，挑战了对抗可传递性现象的传统描述方法。

    

    由于对抗性样本在机器学习中具有欺骗模型的能力，潜在地导致严重后果，因此已成为研究的焦点。在本研究中，我们对对抗机器学习模型进行了全面探索，揭示了它们固有的复杂性和可解释性。我们的调查揭示了机器学习模型复杂性与爱因斯坦的特殊相对论之间的有趣联系，通过纠缠的概念。更具体地说，我们对计算纠缠进行了定义，并证明了远程特征样本可以表现出强相关性，类似于量子领域中的纠缠。这一发现挑战了对当代机器学习模型中观察到的对抗可传递性现象的传统描述方法。

    Adversarial examples in machine learning has emerged as a focal point of research due to their remarkable ability to deceive models with seemingly inconspicuous input perturbations, potentially resulting in severe consequences. In this study, we embark on a comprehensive exploration of adversarial machine learning models, shedding light on their intrinsic complexity and interpretability. Our investigation reveals intriguing links between machine learning model complexity and Einstein's theory of special relativity, through the concept of entanglement. More specific, we define entanglement computationally and demonstrate that distant feature samples can exhibit strong correlations, akin to entanglement in quantum realm. This revelation challenges conventional perspectives in describing the phenomenon of adversarial transferability observed in contemporary machine learning models. By drawing parallels with the relativistic effects of time dilation and length contraction during computatio
    
[^180]: 稀缺图像数据的MLOps：显微镜图像分析的一个案例研究

    MLOps for Scarce Image Data: A Use Case in Microscopic Image Analysis. (arXiv:2309.15521v1 [cs.LG])

    [http://arxiv.org/abs/2309.15521](http://arxiv.org/abs/2309.15521)

    本论文研究在稀缺数据分析中完全应用MLOps的情况，并提出了一种新的整体方法来增强生物医学图像分析。

    

    如今，机器学习（ML）正在经历前所未有的流行。ML模型的操作化由一组被称为机器学习操作（MLOps）的概念和方法所指导。然而，研究人员和专业人员往往更多地关注自动化方面，忽视MLOps的持续部署和监控方面。结果，由于概念漂移，特别是在处理稀缺数据时，从生产到开发过程中的反馈缺乏连续学习，导致模型会随时间不断恶化。本文探讨了在稀缺数据分析环境中完全应用MLOps的情况。该论文提出了一种新的整体方法来增强生物医学图像分析。我们的方法包括：指纹化过程，使得根据手头的图像分析任务选择最佳模型、数据集和模型开发策略；一种自动化的模型开发过程。

    Nowadays, Machine Learning (ML) is experiencing tremendous popularity that has never been seen before. The operationalization of ML models is governed by a set of concepts and methods referred to as Machine Learning Operations (MLOps). Nevertheless, researchers, as well as professionals, often focus more on the automation aspect and neglect the continuous deployment and monitoring aspects of MLOps. As a result, there is a lack of continuous learning through the flow of feedback from production to development, causing unexpected model deterioration over time due to concept drifts, particularly when dealing with scarce data. This work explores the complete application of MLOps in the context of scarce data analysis. The paper proposes a new holistic approach to enhance biomedical image analysis. Our method includes: a fingerprinting process that enables selecting the best models, datasets, and model development strategy relative to the image analysis task at hand; an automated model deve
    
[^181]: 面向弱监督下的数据选择统计理论

    Towards a statistical theory of data selection under weak supervision. (arXiv:2309.14563v1 [stat.ML])

    [http://arxiv.org/abs/2309.14563](http://arxiv.org/abs/2309.14563)

    本研究针对弱监督下的数据选择进行了统计理论研究，通过实验证明数据选择可以非常有效，有时甚至可以战胜对整个样本的训练。并分析了在不同情况下的数据选择选择方法的有效性。

    

    对于一个大小为N的样本，选择一个更小的大小n<N的子样本用于统计估计或学习通常是有用的。这样的数据选择步骤有助于减少数据标记的要求和学习的计算复杂性。我们假设给定了N个未标记的样本{x_i}，并且可以访问一个“替代模型”，它可以比随机猜测更好地预测标签y_i。我们的目标是选择一个子样本集{𝐱_i}，其大小为|G|=n<N。然后我们为这个集合获取标签，并使用它们通过正则化经验风险最小化来训练模型。通过在真实和合成数据上进行混合的数值实验，并在低维和高维渐近情况下进行数学推导，我们证明：(i) 数据选择可以非常有效，特别是在某些情况下可以击败对整个样本的训练；(ii) 在数据选择方面，某些流行的选择在一些情况下是有效的，而在其他情况下则不是。

    Given a sample of size $N$, it is often useful to select a subsample of smaller size $n<N$ to be used for statistical estimation or learning. Such a data selection step is useful to reduce the requirements of data labeling and the computational complexity of learning. We assume to be given $N$ unlabeled samples $\{{\boldsymbol x}_i\}_{i\le N}$, and to be given access to a `surrogate model' that can predict labels $y_i$ better than random guessing. Our goal is to select a subset of the samples, to be denoted by $\{{\boldsymbol x}_i\}_{i\in G}$, of size $|G|=n<N$. We then acquire labels for this set and we use them to train a model via regularized empirical risk minimization.  By using a mixture of numerical experiments on real and synthetic data, and mathematical derivations under low- and high- dimensional asymptotics, we show that: $(i)$~Data selection can be very effective, in particular beating training on the full sample in some cases; $(ii)$~Certain popular choices in data selecti
    
[^182]: DeepSpeed Ulysses：用于训练极长序列Transformer模型的系统优化

    DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models. (arXiv:2309.14509v1 [cs.LG])

    [http://arxiv.org/abs/2309.14509](http://arxiv.org/abs/2309.14509)

    本论文介绍了DeepSpeed-Ulysses，一种用于实现具备极长序列长度的高效可扩展LLM训练的新颖方法。

    

    传统的基于Transformer的大型语言模型（LLM）的计算可以通过批量大小、隐藏维度、层数和序列长度来描述。到目前为止，加速LLM训练的系统工作主要集中在前三个维度上：批量大小的数据并行化、隐藏尺寸的张量并行化以及模型深度或层数的流水线并行化。这些被广泛研究的并行形式并不针对长序列Transformer模型进行优化。鉴于长序列LLM在实际应用需求上的重要性，序列并行化引起了重新关注。然而，现有的序列并行化工作受到内存通信效率的限制，限制了它们在长序列大模型上的可扩展性。在这项工作中，我们引入了DeepSpeed-Ulysses，一种新颖、便携且有效的方法，用于实现具备极长序列长度的高效可扩展LLM训练。

    Computation in a typical Transformer-based large language model (LLM) can be characterized by batch size, hidden dimension, number of layers, and sequence length. Until now, system works for accelerating LLM training have focused on the first three dimensions: data parallelism for batch size, tensor parallelism for hidden size and pipeline parallelism for model depth or layers. These widely studied forms of parallelism are not targeted or optimized for long sequence Transformer models. Given practical application needs for long sequence LLM, renewed attentions are being drawn to sequence parallelism. However, existing works in sequence parallelism are constrained by memory-communication inefficiency, limiting their scalability to long sequence large models. In this work, we introduce DeepSpeed-Ulysses, a novel, portable and effective methodology for enabling highly efficient and scalable LLM training with extremely long sequence length. DeepSpeed-Ulysses at its core partitions input da
    
[^183]: 概率权重固定：用于量化的神经网络权重不确定性的大规模训练

    Probabilistic Weight Fixing: Large-scale training of neural network weight uncertainties for quantization. (arXiv:2309.13575v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.13575](http://arxiv.org/abs/2309.13575)

    本文提出了一种基于贝叶斯神经网络和变分松弛的概率框架，用于通过将权重值限制在一组有限值上来减少推理过程中的能量消耗。通过利用权重值的概率分布，提高了噪声鲁棒性和可压缩性。迭代聚类过程展示了超越现有方法的优势。

    

    权重共享量化是一种通过将神经网络的权重限制在一组有限的值上来减少推理过程中能量消耗的技术。然而，现有的权重共享量化方法常常基于权重值本身进行假设，并忽视了权重位置在其中扮演的独特角色。本文提出了一个基于贝叶斯神经网络（BNNs）和变分松弛的概率框架，根据单个权重的位置特定学习不确定性分布来确定可以将哪些权重移动到哪个聚类中心以及移动到什么程度。我们引入了一种新的初始化设置和正则化项，可以在复杂的数据集-模型组合下训练BNNs。通过利用通过概率分布捕捉到的权重值的灵活性，我们提高了噪声的鲁棒性和下游的可压缩性。我们的迭代聚类过程展示了超越现有方法的优越性能。

    Weight-sharing quantization has emerged as a technique to reduce energy expenditure during inference in large neural networks by constraining their weights to a limited set of values. However, existing methods for weight-sharing quantization often make assumptions about the treatment of weights based on value alone that neglect the unique role weight position plays. This paper proposes a probabilistic framework based on Bayesian neural networks (BNNs) and a variational relaxation to identify which weights can be moved to which cluster centre and to what degree based on their individual position-specific learned uncertainty distributions. We introduce a new initialisation setting and a regularisation term which allow for the training of BNNs under complex dataset-model combinations. By leveraging the flexibility of weight values captured through a probability distribution, we enhance noise resilience and downstream compressibility. Our iterative clustering procedure demonstrates superio
    
[^184]: 时间序列预测：利用分数差分数据释放长期依赖关系

    Time-Series Forecasting: Unleashing Long-Term Dependencies with Fractionally Differenced Data. (arXiv:2309.13409v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.13409](http://arxiv.org/abs/2309.13409)

    本研究提出了一种利用分数差分来捕捉时间序列数据中短期和长期依赖关系的预测策略。通过将FD应用于金融数据并结合情感分析，实证结果证明FD在二元分类中的性能优于整数差分方法。

    

    本研究介绍了一种新颖的预测策略，利用分数差分（FD）的能力来捕捉时间序列数据中的短期和长期依赖关系。与传统的整数差分方法不同，FD在保持系列记忆的同时稳定了它以供建模目的。通过将FD应用于来自SPY指数的金融数据，并结合新闻报道的情感分析，这个实证分析探讨了FD与目标变量的二元分类的效果。采用了监督分类算法来验证FD系列的性能。结果显示，FD相比整数差分具有优越性，这一点通过接收者操作特征/曲线下面积（ROCAUC）和马修斯相关系数（MCC）的评估得到确认。

    This study introduces a novel forecasting strategy that leverages the power of fractional differencing (FD) to capture both short- and long-term dependencies in time series data. Unlike traditional integer differencing methods, FD preserves memory in series while stabilizing it for modeling purposes. By applying FD to financial data from the SPY index and incorporating sentiment analysis from news reports, this empirical analysis explores the effectiveness of FD in conjunction with binary classification of target variables. Supervised classification algorithms were employed to validate the performance of FD series. The results demonstrate the superiority of FD over integer differencing, as confirmed by Receiver Operating Characteristic/Area Under the Curve (ROCAUC) and Mathews Correlation Coefficient (MCC) evaluations.
    
[^185]: 学习自适应安全性的多智能体系统

    Learning Adaptive Safety for Multi-Agent Systems. (arXiv:2309.10657v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2309.10657](http://arxiv.org/abs/2309.10657)

    本论文研究了在多智能体系统中学习自适应安全性的问题，提出了一种全新的自适应安全强化学习框架ASRL，通过优化策略和CBF系数，增强安全性和长期性能。在与其他智能体的交互中，ASRL学会了应对不同的智能体行为，并保持成本违规在所需限制之下。

    

    由于对其他智能体的信息有限，确保动态多智能体系统的安全性是具有挑战性的。控制屏障函数（CBFs）在安全保证方面显示出了潜力，但当前方法对其他智能体做出了很强的假设，并且常常依赖手动调整以平衡安全性、可行性和性能。在这项工作中，我们深入研究了带有CBF的多智能体系统的自适应安全学习问题。我们展示了CBF配置如何深刻影响新兴行为，凸显了对CBF设计进行响应式和动态方法的必要性。我们提出了ASRL，一种全新的自适应安全强化学习框架，通过强化学习完全自动优化策略和CBF系数，增强安全性和长期性能。通过直接与其他智能体进行交互，ASRL学会了应对不同的智能体行为，并将成本违规保持在所需的限制之下。我们在多机器人系统和竞争中评估了ASRL。

    Ensuring safety in dynamic multi-agent systems is challenging due to limited information about the other agents. Control Barrier Functions (CBFs) are showing promise for safety assurance but current methods make strong assumptions about other agents and often rely on manual tuning to balance safety, feasibility, and performance. In this work, we delve into the problem of adaptive safe learning for multi-agent systems with CBF. We show how emergent behavior can be profoundly influenced by the CBF configuration, highlighting the necessity for a responsive and dynamic approach to CBF design. We present ASRL, a novel adaptive safe RL framework, to fully automate the optimization of policy and CBF coefficients, to enhance safety and long-term performance through reinforcement learning. By directly interacting with the other agents, ASRL learns to cope with diverse agent behaviours and maintains the cost violations below a desired limit. We evaluate ASRL in a multi-robot system and a competi
    
[^186]: 对多模态大规模语言模型中的灾难性遗忘进行的研究

    Investigating the Catastrophic Forgetting in Multimodal Large Language Models. (arXiv:2309.10313v1 [cs.CL])

    [http://arxiv.org/abs/2309.10313](http://arxiv.org/abs/2309.10313)

    本论文针对多模态大规模语言模型中的灾难性遗忘问题进行研究，引入了EMT方法来评估灾难性遗忘，并发现在标准图像分类任务上，几乎所有评估的模型都无法保持与视觉编码器相同的性能水平。研究结果表明，早期微调阶段对性能至关重要。

    

    在GPT4的成功之后，多模态大规模语言模型（MLLM）研究引起了广泛关注。这一研究方向侧重于通过微调预训练的LLM和视觉模型来开发通用的LLM。然而，灾难性遗忘，即微调模型无法保持与预训练模型相似的性能水平，仍然是多模态LLM（MLLM）中的一个固有问题。本文介绍了EMT：用于评估MLLM中灾难性遗忘的评估方法，将每个MLLM作为一个图像分类器进行评估。我们首先应用EMT来评估几个开源的微调MLLM，并发现几乎所有评估的MLLM在标准图像分类任务上无法保持与他们的视觉编码器相同的性能水平。此外，我们继续微调LLaVA，一种MLLM，并利用EMT来评估整个微调过程中的性能。有趣的是，我们的结果表明，早期的微调阶段是关键的，过早停止微调可能导致低性能的模型。

    Following the success of GPT4, there has been a surge in interest in multimodal large language model (MLLM) research. This line of research focuses on developing general-purpose LLMs through fine-tuning pre-trained LLMs and vision models. However, catastrophic forgetting, a notorious phenomenon where the fine-tuned model fails to retain similar performance compared to the pre-trained model, still remains an inherent problem in multimodal LLMs (MLLM). In this paper, we introduce EMT: Evaluating MulTimodality for evaluating the catastrophic forgetting in MLLMs, by treating each MLLM as an image classifier. We first apply EMT to evaluate several open-source fine-tuned MLLMs and we discover that almost all evaluated MLLMs fail to retain the same performance levels as their vision encoders on standard image classification tasks. Moreover, we continue fine-tuning LLaVA, an MLLM and utilize EMT to assess performance throughout the fine-tuning. Interestingly, our results suggest that early-sta
    
[^187]: 稀疏自编码器在语言模型中发现高度可解释的特征

    Sparse Autoencoders Find Highly Interpretable Features in Language Models. (arXiv:2309.08600v1 [cs.LG])

    [http://arxiv.org/abs/2309.08600](http://arxiv.org/abs/2309.08600)

    本研究通过稀疏自编码器在语言模型中发现了一组高度可解释和单一义的特征，从而解决了神经网络内部多义性的问题。

    

    神经网络内部理解的一个障碍是多义性，其中神经元在多个语义不同的上下文中激活。多义性使我们无法找到简洁的、人类可理解的解释来解释神经网络内部的工作。多义性的一个猜测原因是叠加效应，即神经网络通过将特征分配给激活空间中的一个过完备方向集合，而不是个别神经元，表示更多的特征。在这里，我们尝试使用稀疏自编码器来确定这些方向，以重构语言模型的内部激活。这些自编码器学习到的一组稀疏激活特征比其他方法鉴定出的方向更可解释和单一义，解释性是通过自动化方法衡量的。删除这些特征可以实现精确的模型编辑，例如通过删除这些特征可以改变模型输出。

    One of the roadblocks to a better understanding of neural networks' internals is \textit{polysemanticity}, where neurons appear to activate in multiple, semantically distinct contexts. Polysemanticity prevents us from identifying concise, human-understandable explanations for what neural networks are doing internally. One hypothesised cause of polysemanticity is \textit{superposition}, where neural networks represent more features than they have neurons by assigning features to an overcomplete set of directions in activation space, rather than to individual neurons. Here, we attempt to identify those directions, using sparse autoencoders to reconstruct the internal activations of a language model. These autoencoders learn sets of sparsely activating features that are more interpretable and monosemantic than directions identified by alternative approaches, where interpretability is measured by automated methods. Ablating these features enables precise model editing, for example, by remo
    
[^188]: ConR: 用于深度不平衡回归的对比正则化器

    ConR: Contrastive Regularizer for Deep Imbalanced Regression. (arXiv:2309.06651v1 [cs.LG])

    [http://arxiv.org/abs/2309.06651](http://arxiv.org/abs/2309.06651)

    ConR是一种对比正则化器，通过建模全局和局部标签相似性，防止少数样本的特征被折叠到其多数邻居中，有效地处理深度不平衡回归问题。

    

    不平衡分布在现实世界的数据中很常见。它们对深度神经网络提出了约束，以表示少数类别标签并避免对多数类别的偏见。大量的不平衡方法处理了分类标签空间，但在连续标签空间的回归问题上未能有效应用。相反，连续标签之间的局部和全局关联为在特征空间中有效建模关系提供了有价值的见解。在这项工作中，我们提出了ConR，一种对比正则化器，它在特征空间中建模全局和局部标签相似性，防止少数样本的特征被折叠到它们的多数邻居中。通过将预测的相似性作为特征相似性的指示器，ConR区分了标签空间和特征空间之间的不一致，并对这些不一致施加惩罚。ConR通过两个主要策略关注标签空间的连续性。

    Imbalanced distributions are ubiquitous in real-world data. They create constraints on Deep Neural Networks to represent the minority labels and avoid bias towards majority labels. The extensive body of imbalanced approaches address categorical label spaces but fail to effectively extend to regression problems where the label space is continuous. Conversely, local and global correlations among continuous labels provide valuable insights towards effectively modelling relationships in feature space. In this work, we propose ConR, a contrastive regularizer that models global and local label similarities in feature space and prevents the features of minority samples from being collapsed into their majority neighbours. Serving the similarities of the predictions as an indicator of feature similarities, ConR discerns the dissagreements between the label space and feature space and imposes a penalty on these disagreements. ConR minds the continuous nature of label space with two main strategi
    
[^189]: 专业性与广泛性：关于基础模型微调中灾难性遗忘的实证研究

    Speciality vs Generality: An Empirical Study on Catastrophic Forgetting in Fine-tuning Foundation Models. (arXiv:2309.06256v1 [cs.LG])

    [http://arxiv.org/abs/2309.06256](http://arxiv.org/abs/2309.06256)

    本研究实证了基础模型微调中的灾难性遗忘现象，微调过程中追求专业性会导致模型的广泛性损失。

    

    基础模型，包括视觉语言模型(VLMs)和大型语言模型(LLMs)，具有处理多样分布和任务的广泛性，这源于它们广泛的预训练数据集。对基础模型进行微调是提高任务性能或调整模型行为与人类期望一致的常见做法，使其获得专业性。然而，用于微调的小型数据集可能无法充分覆盖预训练过程中遇到的多样分布和任务。因此，追求微调过程中的专业性可能导致模型的广泛性损失，这与深度学习中的灾难性遗忘(Catastrophic Forgetting, CF)相关。在本研究中，我们展示了这种现象在VLMs和LLMs中的存在。例如，对像CLIP这样的VLM进行在ImageNet上的微调会导致处理多样分布的广泛性损失，对医学领域的Galactica进行微调则会导致遵循指令的能力损失。

    Foundation models, including Vision Language Models (VLMs) and Large Language Models (LLMs), possess the $generality$ to handle diverse distributions and tasks, which stems from their extensive pre-training datasets. The fine-tuning of foundation models is a common practice to enhance task performance or align the model's behavior with human expectations, allowing them to gain $speciality$. However, the small datasets used for fine-tuning may not adequately cover the diverse distributions and tasks encountered during pre-training. Consequently, the pursuit of speciality during fine-tuning can lead to a loss of {generality} in the model, which is related to catastrophic forgetting (CF) in deep learning. In this study, we demonstrate this phenomenon in both VLMs and LLMs. For instance, fine-tuning VLMs like CLIP on ImageNet results in a loss of generality in handling diverse distributions, and fine-tuning LLMs like Galactica in the medical domain leads to a loss in following instructions
    
[^190]: 在COVID-19期间导航不在分布范围内的电力负荷预测：利用人类移动的持续学习方法

    Navigating Out-of-Distribution Electricity Load Forecasting during COVID-19: A Continual Learning Approach Leveraging Human Mobility. (arXiv:2309.04296v1 [cs.LG])

    [http://arxiv.org/abs/2309.04296](http://arxiv.org/abs/2309.04296)

    本研究提出了一种利用人类移动数据和持续学习技术的方法来解决COVID-19期间非分布期间的电力负荷预测问题，通过保留过去的见解并整合新的数据，提高了模型的准确性和鲁棒性。

    

    在传统的深度学习算法中，一个关键假设是数据分布在训练和部署过程中保持不变。然而，在面对非分布期间时，如COVID-19的封锁期，数据分布与模型在训练过程中所见的明显偏离。本文采用双重策略：利用持续学习技术更新模型的新数据，并利用在建筑物外部的保护隐私的行人计数器收集的人类移动数据。与在线学习相比，后者常常会遭受“灾难性遗忘”的困扰，因为新获得的知识常常会抹去先前的信息，持续学习则通过保留过去的见解并整合新的数据，提供了一个整体的方法。本研究将FSNet，一种强大的持续学习算法，应用于墨尔本市13个建筑群的真实数据。

    In traditional deep learning algorithms, one of the key assumptions is that the data distribution remains constant during both training and deployment. However, this assumption becomes problematic when faced with Out-of-Distribution periods, such as the COVID-19 lockdowns, where the data distribution significantly deviates from what the model has seen during training. This paper employs a two-fold strategy: utilizing continual learning techniques to update models with new data and harnessing human mobility data collected from privacy-preserving pedestrian counters located outside buildings. In contrast to online learning, which suffers from 'catastrophic forgetting' as newly acquired knowledge often erases prior information, continual learning offers a holistic approach by preserving past insights while integrating new data. This research applies FSNet, a powerful continual learning algorithm, to real-world data from 13 building complexes in Melbourne, Australia, a city which had the s
    
[^191]: 使用温度指数测度的最优输运

    Optimal Transport with Tempered Exponential Measures. (arXiv:2309.04015v1 [cs.LG])

    [http://arxiv.org/abs/2309.04015](http://arxiv.org/abs/2309.04015)

    本文推广了熵正则化最优输运方法，将其应用于温度指数测度中，实现了快速有效的算法和可控的稀疏性。

    

    在最优输运领域中，两个重要的子领域相互对立：（i）非正则化最优输运，“卡托诺维奇方式”，导致了非常稀疏的规划，但算法效率较低；（ii）熵正则化最优输运，“辛克霍恩-库都里方式”，获得了近似线性算法，但最大程度上无法稀疏规划。本文中，我们将后一种方法推广到温度指数测度，即具有间接测度归一化的指数族泛化，取得了非常方便的折中效果，具有非常快的近似算法和可控的稀疏性，同时也适用于不平衡最优输运问题。

    In the field of optimal transport, two prominent subfields face each other: (i) unregularized optimal transport, ``\`a-la-Kantorovich'', which leads to extremely sparse plans but with algorithms that scale poorly, and (ii) entropic-regularized optimal transport, ``\`a-la-Sinkhorn-Cuturi'', which gets near-linear approximation algorithms but leads to maximally un-sparse plans. In this paper, we show that a generalization of the latter to tempered exponential measures, a generalization of exponential families with indirect measure normalization, gets to a very convenient middle ground, with both very fast approximation algorithms and sparsity which is under control up to sparsity patterns. In addition, it fits naturally in the unbalanced optimal transport problem setting as well.
    
[^192]: 受物理启发的非键相互作用等变描述符

    Physics-inspired Equivariant Descriptors of Non-bonded Interactions. (arXiv:2308.13208v1 [physics.chem-ph])

    [http://arxiv.org/abs/2308.13208](http://arxiv.org/abs/2308.13208)

    基于物理启发，我们提出了一种受物理启发的非键相互作用等变描述符框架，该框架能够模拟长程物理相互作用，并且能够生成类似非键位势的局部描述符。

    

    大多数应用于原子尺度模拟的现有机器学习方案依赖于对结构几何的局部描述，并且在建模由长程物理相互作用驱动的效应方面困难重重。克服这些限制的努力集中于直接将静电引入，这是最突出的效应，通常依赖于与显式物理模型的功能形式相似的体系结构。包括其他形式的非键相互作用，或者预测除了原子间势能之外的性质，需要进行临时修改。我们提出了一种替代方法，将远程等变（LODE）框架扩展到生成类似任意渐近行为的非键位势的原子环境的局部描述符，从点电荷静电到色散力。我们证明，LODE形式主义可通过广义多极展开直观地解释。

    Most of the existing machine-learning schemes applied to atomic-scale simulations rely on a local description of the geometry of a structure, and struggle to model effects that are driven by long-range physical interactions. Efforts to overcome these limitations have focused on the direct incorporation of electrostatics, which is the most prominent effect, often relying on architectures that mirror the functional form of explicit physical models. Including other forms of non-bonded interactions, or predicting properties other than the interatomic potential, requires ad hoc modifications. We propose an alternative approach that extends the long-distance equivariant (LODE) framework to generate local descriptors of an atomic environment that resemble non-bonded potentials with arbitrary asymptotic behaviors, ranging from point-charge electrostatics to dispersion forces. We show that the LODE formalism is amenable to a direct physical interpretation in terms of a generalized multipole exp
    
[^193]: 基于贝叶斯低秩适应的大型语言模型

    Bayesian low-rank adaptation for large language models. (arXiv:2308.13111v1 [cs.LG])

    [http://arxiv.org/abs/2308.13111](http://arxiv.org/abs/2308.13111)

    本研究提出了一种名为Laplace-LoRA的贝叶斯方法，通过应用拉普拉斯近似来增强经过微调的大型语言模型的校准能力。

    

    参数高效的微调（PEFT）已成为大型语言模型（LLMs）成本高效微调的新范式，其中低秩适应（LoRA）被广泛采用。然而，经过微调的LLMs往往变得过于自信，尤其是在较小数据集上进行微调时。贝叶斯方法具有估计不确定性的固有能力，可作为减轻过度自信并增强校准能力的有力工具。在这项工作中，我们引入了Laplace-LoRA，一种直观而有效的贝叶斯方法，它将拉普拉斯近似应用于LoRA参数，并显著提升了经过微调的LLMs的校准能力。

    Parameter-efficient fine-tuning (PEFT) has emerged as a new paradigm for cost-efficient fine-tuning of large language models (LLMs), with low-rank adaptation (LoRA) being a widely adopted choice. However, fine-tuned LLMs often become overconfident especially on when fine-tuned on smaller datasets. Bayesian methods, with their inherent ability to estimate uncertainty, serve as potent tools to mitigate overconfidence and enhance calibration. In this work, we introduce Laplace-LoRA, a straightforward yet effective Bayesian method, which applies the Laplace approximation to the LoRA parameters and, considerably boosts the calibration of fine-tuned LLMs.
    
[^194]: 大型语言模型的指令调优：一项调研

    Instruction Tuning for Large Language Models: A Survey. (arXiv:2308.10792v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2308.10792](http://arxiv.org/abs/2308.10792)

    本文调查了指令调优这一关键技术在增强大型语言模型能力和可控性方面的研究工作，包括方法、数据集构建、模型训练和应用，以及对结果影响的分析。同时回顾了可能的问题和批评，并指出了目前的不足。

    

    本文调查了指令调优（IT）这一快速发展的领域中的研究工作，这是一种增强大型语言模型（LLM）能力和可控性的关键技术。指令调优是指以监督方式在包含“指令-输出”对的数据集上进一步训练LLM，这将LLM的下一个词预测目标与用户希望LLM遵守人类指令的目标之间的差距。本文对IT的常规方法、IT数据集的构建、IT模型的训练以及应用于不同模态、领域和应用的情况进行了系统的文献综述，并对影响IT结果的各个方面进行了分析（例如，指令输出的生成、指令数据集的大小等）。我们还回顾了IT的潜在问题以及针对其的批评，以及指出当前不足的努力。

    This paper surveys research works in the quickly advancing field of instruction tuning (IT), a crucial technique to enhance the capabilities and controllability of large language models (LLMs). Instruction tuning refers to the process of further training LLMs on a dataset consisting of \textsc{(instruction, output)} pairs in a supervised fashion, which bridges the gap between the next-word prediction objective of LLMs and the users' objective of having LLMs adhere to human instructions. In this work, we make a systematic review of the literature, including the general methodology of IT, the construction of IT datasets, the training of IT models, and applications to different modalities, domains and applications, along with an analysis on aspects that influence the outcome of IT (e.g., generation of instruction outputs, size of the instruction dataset, etc). We also review the potential pitfalls of IT along with criticism against it, along with efforts pointing out current deficiencies 
    
[^195]: AQUILA: 自适应量化懒汇聚梯度的通信高效联邦学习

    AQUILA: Communication Efficient Federated Learning with Adaptive Quantization of Lazily-Aggregated Gradients. (arXiv:2308.00258v1 [cs.LG])

    [http://arxiv.org/abs/2308.00258](http://arxiv.org/abs/2308.00258)

    AQUILA是一个自适应量化梯度的通信高效联邦学习框架，解决了传输大规模模型时的通信开销和局部数据偏差导致的全局模型鲁棒性问题。

    

    联邦学习的广泛应用受到高通信开销的挑战，主要来自大规模模型的传输。现有的自适应量化方法在每一轮训练中都假设设备参与均匀，在实践中不可行。此外，这些方法在选取量化级别时存在局限性，并经常忽视本地设备数据的偏差，从而影响全局模型的鲁棒性。为了解决这些问题，本文引入了一种名为AQUILA（自适应量化懒汇聚梯度）的新型自适应框架，以增强联邦学习的效率和鲁棒性。AQUILA整合了一种复杂的设备选择方法，优先考虑设备更新的质量和实用性。

    The widespread adoption of Federated Learning (FL), a privacy-preserving distributed learning methodology, has been impeded by the challenge of high communication overheads, typically arising from the transmission of large-scale models. Existing adaptive quantization methods, designed to mitigate these overheads, operate under the impractical assumption of uniform device participation in every training round. Additionally, these methods are limited in their adaptability due to the necessity of manual quantization level selection and often overlook biases inherent in local devices' data, thereby affecting the robustness of the global model. In response, this paper introduces AQUILA (adaptive quantization of lazily-aggregated gradients), a novel adaptive framework devised to effectively handle these issues, enhancing the efficiency and robustness of FL. AQUILA integrates a sophisticated device selection method that prioritizes the quality and usefulness of device updates. Utilizing the e
    
[^196]: 模仿复杂轨迹：桥接低层稳定性与高层行为

    Imitating Complex Trajectories: Bridging Low-Level Stability and High-Level Behavior. (arXiv:2307.14619v1 [cs.LG])

    [http://arxiv.org/abs/2307.14619](http://arxiv.org/abs/2307.14619)

    本文提出了一个理论框架，研究了在非线性动态系统中模仿复杂专家演示的行为。通过稳定模仿策略并确保准确估计演示者分布，可以使模仿者与演示者的轨迹分布相近。

    

    我们提出了一个理论框架来研究在非线性动态系统中模仿随机、非马尔可夫、潜在多模态（即“复杂”）专家演示的行为。我们的框架使用低层控制器（无论是学习的还是隐含的）来稳定围绕专家演示的模仿策略。我们证明，在（a）合适的低层稳定性保证和（b）学习策略的随机连续性属性（我们称之为“总变差连续性”）（TVC）的情况下，一个精确估计演示者状态分布上的行动的模仿者会与演示者对整个轨迹的分布相近。然后，我们证明可以通过将流行的数据增强规则与一种新颖的算法技巧相结合（即在执行时添加增强噪声）来确保TVC并且最小程度上降低精度。我们将我们的保证实例化为由扩散模型参数化的策略，并证明如果学习者准确地估计了演示者的分布，则最终完成这种实例化。

    We propose a theoretical framework for studying the imitation of stochastic, non-Markovian, potentially multi-modal (i.e. "complex" ) expert demonstrations in nonlinear dynamical systems. Our framework invokes low-level controllers either learned or implicit in position-command control - to stabilize imitation policies around expert demonstrations. We show that with (a) a suitable low-level stability guarantee and (b) a stochastic continuity property of the learned policy we call "total variation continuity" (TVC), an imitator that accurately estimates actions on the demonstrator's state distribution closely matches the demonstrator's distribution over entire trajectories. We then show that TVC can be ensured with minimal degradation of accuracy by combining a popular data-augmentation regimen with a novel algorithmic trick: adding augmentation noise at execution time. We instantiate our guarantees for policies parameterized by diffusion models and prove that if the learner accuratel
    
[^197]: PINNsFormer: 基于Transformer的物理信息神经网络框架

    PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks. (arXiv:2307.11833v1 [cs.CE])

    [http://arxiv.org/abs/2307.11833](http://arxiv.org/abs/2307.11833)

    PINNsFormer是一种基于Transformer的框架，通过捕捉时间依赖性准确逼近求解偏微分方程，相比传统方法具有更好的性能。

    

    物理信息神经网络（PINNs）已经成为一种有效的深度学习框架，用于近似求解偏微分方程（PDEs）的数值解。然而，传统的PINNs和大多数相关研究采用全连接的多层感知机（MLP）作为核心结构，忽略了PDEs中的时间关系，无法准确逼近真解。在本文中，我们提出了一种新的基于Transformer的框架，即PINNsFormer，通过Transformer-based模型中的多头注意力机制捕捉时间依赖性，准确逼近PDEs的解。PINNsFormer不仅适应输入向量以伪序列的形式进行近似预测，还将逐点的PINNs损失改为了顺序的PINNs损失。此外，PINNsFormer还配备了一种新的激活函数，即小波函数，通过深度神经网络实现对傅里叶分解的预测。我们通过实验证明了PINNsFormer捕捉时间依赖关系的能力。

    Physics-Informed Neural Networks (PINNs) have emerged as a promising deep learning framework for approximating numerical solutions for partial differential equations (PDEs). While conventional PINNs and most related studies adopt fully-connected multilayer perceptrons (MLP) as the backbone structure, they have neglected the temporal relations in PDEs and failed to approximate the true solution. In this paper, we propose a novel Transformer-based framework, namely PINNsFormer, that accurately approximates PDEs' solutions by capturing the temporal dependencies with multi-head attention mechanisms in Transformer-based models. Instead of approximating point predictions, PINNsFormer adapts input vectors to pseudo sequences and point-wise PINNs loss to a sequential PINNs loss. In addition, PINNsFormer is equipped with a novel activation function, namely Wavelet, which anticipates the Fourier decomposition through deep neural networks. We empirically demonstrate PINNsFormer's ability to captu
    
[^198]: 分层授权：朝着可行的基于授权的技能学习迈进

    Hierarchical Empowerment: Towards Tractable Empowerment-Based Skill-Learning. (arXiv:2307.02728v1 [cs.LG])

    [http://arxiv.org/abs/2307.02728](http://arxiv.org/abs/2307.02728)

    分层授权提出了一种可以计算授权的新框架，通过引入变分下界和分层架构，实现了在短期和长期时间尺度上的授权计算，并在模拟机器人任务中得到了验证。

    

    通用智能体需要大量的技能。 授权 - 技能和状态之间的最大互信息 - 为学习大量不同技能提供了一条路径，但互信息很难优化。我们介绍了一种新的框架，分层授权，通过集成目标条件层次强化学习的概念，使得计算授权更加可行。我们的框架提供了两个具体的贡献。首先，我们介绍了一种新的变分下界，可用于计算短期视角下的授权。其次，我们引入了一个分层架构，用于计算指数时间尺度下的授权。我们在一系列模拟机器人任务中验证了该框架的贡献。在一个流行的蚂蚁导航领域，我们的四级智能体能够学习覆盖面积比之前的工作大两个数量级的技能。

    General purpose agents will require large repertoires of skills. Empowerment -- the maximum mutual information between skills and the states -- provides a pathway for learning large collections of distinct skills, but mutual information is difficult to optimize. We introduce a new framework, Hierarchical Empowerment, that makes computing empowerment more tractable by integrating concepts from Goal-Conditioned Hierarchical Reinforcement Learning. Our framework makes two specific contributions. First, we introduce a new variational lower bound on mutual information that can be used to compute empowerment over short horizons. Second, we introduce a hierarchical architecture for computing empowerment over exponentially longer time scales. We verify the contributions of the framework in a series of simulated robotics tasks. In a popular ant navigation domain, our four level agents are able to learn skills that cover a surface area over two orders of magnitude larger than prior work.
    
[^199]: 抽象文本摘要中的命名实体包含

    Named Entity Inclusion in Abstractive Text Summarization. (arXiv:2307.02570v1 [cs.CL])

    [http://arxiv.org/abs/2307.02570](http://arxiv.org/abs/2307.02570)

    该论文提出了一种解决抽象文本摘要中命名实体遗漏问题的方法，通过使用定制的预训练目标和模型训练策略，改善了命名实体的包含情况，提高了摘要的准确性和召回率。

    

    我们解决了许多当前抽象文本摘要器的缺点，即命名实体的遗漏问题。我们建议采用定制的预训练目标来增强模型对文本中的命名实体的注意力。首先，使用命名实体识别模型RoBERTa来确定文本中的命名实体。然后，使用该模型对文本中的命名实体进行屏蔽，再使用BART模型对其进行重建。接下来，将BART模型在摘要任务上进行微调。实验证明，这种预训练方法改善了命名实体包含的精确度和召回率指标。

    We address the named entity omission - the drawback of many current abstractive text summarizers. We suggest a custom pretraining objective to enhance the model's attention on the named entities in a text. At first, the named entity recognition model RoBERTa is trained to determine named entities in the text. After that, this model is used to mask named entities in the text and the BART model is trained to reconstruct them. Next, the BART model is fine-tuned on the summarization task. Our experiments showed that this pretraining approach improves named entity inclusion precision and recall metrics.
    
[^200]: 高维线性回归的统一转移学习模型

    Unified Transfer Learning Models for High-Dimensional Linear Regression. (arXiv:2307.00238v1 [stat.ML])

    [http://arxiv.org/abs/2307.00238](http://arxiv.org/abs/2307.00238)

    UTrans是一种统一转移学习模型，它能检测可转移变量和源数据，并具有较低的估计和预测误差，同时保持可解释性。

    

    在现代数据分析中，当目标数据稀缺而源数据充足，或者源数据和目标数据的分布不同的情况下，转移学习在发挥重要作用。本文提出了一种可解释的统一转移学习模型，称为UTrans，该模型能够检测可转移变量和源数据。具体来说，我们建立了估计误差界限，并证明我们的界限低于仅有目标数据的界限。此外，我们基于假设检验提出了一种源数据检测算法，用于排除不可转移的数据。我们在多个实验中评估和比较了UTrans与现有算法。结果显示，UTrans在保持可解释性的同时，比现有方法具有更低的估计和预测误差。最后，我们将其应用于美国代际流动数据，并将我们提出的算法与经典的机器学习算法进行比较。

    Transfer learning plays a key role in modern data analysis when: (1) the target data are scarce but the source data are sufficient; (2) the distributions of the source and target data are heterogeneous. This paper develops an interpretable unified transfer learning model, termed as UTrans, which can detect both transferable variables and source data. More specifically, we establish the estimation error bounds and prove that our bounds are lower than those with target data only. Besides, we propose a source detection algorithm based on hypothesis testing to exclude the nontransferable data. We evaluate and compare UTrans to the existing algorithms in multiple experiments. It is shown that UTrans attains much lower estimation and prediction errors than the existing methods, while preserving interpretability. We finally apply it to the US intergenerational mobility data and compare our proposed algorithms to the classical machine learning algorithms.
    
[^201]: 通过快速融合Gromov化实现图插值

    Graph Interpolation via Fast Fused-Gromovization. (arXiv:2306.15963v1 [cs.LG])

    [http://arxiv.org/abs/2306.15963](http://arxiv.org/abs/2306.15963)

    本文提出了一种通过快速融合Gromov化的方法，用于图插值和图数据增强。通过考虑图结构和信号之间的相互作用，我们提出了一种匹配节点之间的最优策略来解决这一问题。为了提高可扩展性，我们引入了一种放松的FGW求解器来加速算法的收敛速度。

    

    图数据增强已被证明对于增强图神经网络（GNN）的泛化能力和鲁棒性在图级分类方面是有效的。然而，现有的方法主要集中在独立地增强图信号空间和图结构空间，忽视了它们的共同作用。本文通过将问题形式化为一个最优传输问题，旨在找到一种匹配图之间节点的最优策略，考虑图结构和信号之间的相互作用，以解决这个限制。为了解决这个问题，我们提出了一种新颖的图mixup算法，称为FGWMixup，它利用融合Gromov-Wasserstein（FGW）度量空间来识别源图的“中点”。为了提高我们方法的可扩展性，我们引入了一个放松的FGW求解器，通过将收敛速度从O(t^-1)加速到O(t^-2)，提高了FGWMixup的收敛速度。在五个数据集上进行了大量实验证明了我们的方法的有效性。

    Graph data augmentation has proven to be effective in enhancing the generalizability and robustness of graph neural networks (GNNs) for graph-level classifications. However, existing methods mainly focus on augmenting the graph signal space and the graph structure space independently, overlooking their joint interaction. This paper addresses this limitation by formulating the problem as an optimal transport problem that aims to find an optimal strategy for matching nodes between graphs considering the interactions between graph structures and signals. To tackle this problem, we propose a novel graph mixup algorithm dubbed FGWMixup, which leverages the Fused Gromov-Wasserstein (FGW) metric space to identify a "midpoint" of the source graphs. To improve the scalability of our approach, we introduce a relaxed FGW solver that accelerates FGWMixup by enhancing the convergence rate from $\mathcal{O}(t^{-1})$ to $\mathcal{O}(t^{-2})$. Extensive experiments conducted on five datasets, utilizin
    
[^202]: DiMSam:扩散模型作为部分可观测任务与动作规划中的采样器。

    DiMSam: Diffusion Models as Samplers for Task and Motion Planning under Partial Observability. (arXiv:2306.13196v1 [cs.RO])

    [http://arxiv.org/abs/2306.13196](http://arxiv.org/abs/2306.13196)

    本文提出了一种使用扩散模型作为采样器的任务和动作规划方法，在部分可观测下能够实现长周期受约束的操作计划。

    

    任务和动作规划（TAMP）方法非常有效地计划长周期自主机器人操作。但是，由于它们需要一个规划模型，因此在环境和其动态不完全了解的领域中应用它们可能非常困难。我们提出通过利用深度生成建模，特别是扩散模型来克服这些限制，学习捕获规划模型中难以设计的约束和采样器。这些学习采样器在TAMP求解器中组合和合并，以联合找到满足规划中约束的行动参数值。为了便于对环境中未知对象进行预测，我们将这些采样器定义为学习的低维潜变量嵌入的可变对象状态。我们在关节式物体操作领域评估了我们的方法，并展示了经典TAMP、生成学习和潜在嵌入的组合如何使得在部分可观测下进行长周期受约束的操作计划。

    Task and Motion Planning (TAMP) approaches are effective at planning long-horizon autonomous robot manipulation. However, because they require a planning model, it can be difficult to apply them to domains where the environment and its dynamics are not fully known. We propose to overcome these limitations by leveraging deep generative modeling, specifically diffusion models, to learn constraints and samplers that capture these difficult-to-engineer aspects of the planning model. These learned samplers are composed and combined within a TAMP solver in order to find action parameter values jointly that satisfy the constraints along a plan. To tractably make predictions for unseen objects in the environment, we define these samplers on low-dimensional learned latent embeddings of changing object state. We evaluate our approach in an articulated object manipulation domain and show how the combination of classical TAMP, generative learning, and latent embeddings enables long-horizon constra
    
[^203]: 深度图核点过程

    Deep graph kernel point processes. (arXiv:2306.11313v1 [stat.ML])

    [http://arxiv.org/abs/2306.11313](http://arxiv.org/abs/2306.11313)

    本文提出了一种基于潜在图拓扑的图点过程方法，并开发了一种新颖的深度图核来描述事件之间的触发和抑制效应，该方法在合成和实际数据集上具有优越性。

    

    点过程模型广泛用于分析图中异步事件，反映不同类型事件之间的相互影响。预测未来事件的时间和类型是一项关键任务，并且图的大小和拓扑结构增加了问题的难度。最近的神经点过程模型揭示了捕捉复杂的事件类别之间依赖关系的可能性。然而，这些方法在每个目标事件类型的强度计算中使用了包括所有事件类别在内的未经滤波的事件记录。在本文中，我们提出了一种基于潜在图拓扑的图点过程方法。对应的无向图具有代表事件类别的节点和表示潜在贡献关系的边。然后，我们开发了一种新颖的深度图核来描述事件之间的触发和抑制效应。本质影响结构通过图神经网络-based的局部邻域信息聚合进行了融合。我们在合成和实际数据集上展示了我们提出的方法比最先进的模型更具优越性。

    Point process models are widely used to analyze asynchronous events occurring within a graph that reflect how different types of events influence one another. Predicting future events' times and types is a crucial task, and the size and topology of the graph add to the challenge of the problem. Recent neural point process models unveil the possibility of capturing intricate inter-event-category dependencies. However, such methods utilize an unfiltered history of events, including all event categories in the intensity computation for each target event type. In this work, we propose a graph point process method where event interactions occur based on a latent graph topology. The corresponding undirected graph has nodes representing event categories and edges indicating potential contribution relationships. We then develop a novel deep graph kernel to characterize the triggering and inhibiting effects between events. The intrinsic influence structures are incorporated via the graph neural
    
[^204]: 随机加权梯度下降通过分布健壮优化

    Stochastic Re-weighted Gradient Descent via Distributionally Robust Optimization. (arXiv:2306.09222v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.09222](http://arxiv.org/abs/2306.09222)

    我们通过分布健壮优化和重要性加权的梯度下降技术提升了深度神经网络的性能，并在各种任务上取得了优越的结果。

    

    我们通过在每一次优化步骤中对数据点进行重要性加权，开发了一种提高深度神经网络性能的加权梯度下降技术。我们的方法受到分布健壮优化和f-散度的启发，已知可以得到具有改进的泛化保证的模型。我们的加权方案简单、计算高效，可以与许多流行的优化算法（如SGD和Adam）结合使用。实验证明，我们的方法在各种任务上都表现出了优越性能，包括监督学习和领域适应。值得注意的是，我们在DomainBed和Tabular分类基准上分别比现有最佳结果提升了0.7%和1.44%。此外，我们的算法将BERT在GLUE基准上的性能提升了1.94%，将ViT在ImageNet-1K上的性能提升了1.01%。这些结果表明了所提出方法的有效性，预示着它在改善性能方面的潜力。

    We develop a re-weighted gradient descent technique for boosting the performance of deep neural networks, which involves importance weighting of data points during each optimization step. Our approach is inspired by distributionally robust optimization with f-divergences, which has been known to result in models with improved generalization guarantees. Our re-weighting scheme is simple, computationally efficient, and can be combined with many popular optimization algorithms such as SGD and Adam. Empirically, we demonstrate the superiority of our approach on various tasks, including supervised learning, domain adaptation. Notably, we obtain improvements of +0.7% and +1.44% over SOTA on DomainBed and Tabular classification benchmarks, respectively. Moreover, our algorithm boosts the performance of BERT on GLUE benchmarks by +1.94%, and ViT on ImageNet-1K by +1.01%. These results demonstrate the effectiveness of the proposed approach, indicating its potential for improving performance in 
    
[^205]: 快速扩散模型

    Fast Diffusion Model. (arXiv:2306.06991v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.06991](http://arxiv.org/abs/2306.06991)

    本文提出了一种快速扩散模型（FDM），通过将动量集成到扩散过程中，显著加速了扩散模型（DMs）的训练和采样过程。

    

    扩散模型（DMs）以其在捕捉复杂数据分布方面的显著能力，被广泛应用于各个领域。在本文中，我们提出了一种快速扩散模型（FDM），从随机优化的角度显著加速DMs的训练和采样过程。我们首先发现DMs的扩散过程与随机梯度下降（SGD）的随机时变问题的随机优化过程相符合。然后，受到动量SGD的启发，该方法使用梯度和额外的动量，以实现比SGD更快和更稳定的收敛，我们将动量集成到DMs的扩散过程中。这带来了一个独特的挑战，即从基于动量的扩散过程中导出噪声扰动核。为此，我们将这个过程构建为一个阻尼振荡系统，临界阻尼状态-核解决方案-避免振荡，使扩散过程的收敛速度更快。实验证明，FDM在加速训练和采样过程方面取得了显著效果。

    Diffusion models (DMs) have been adopted across diverse fields with its remarkable abilities in capturing intricate data distributions. In this paper, we propose a Fast Diffusion Model (FDM) to significantly speed up DMs from a stochastic optimization perspective for both faster training and sampling. We first find that the diffusion process of DMs accords with the stochastic optimization process of stochastic gradient descent (SGD) on a stochastic time-variant problem. Then, inspired by momentum SGD that uses both gradient and an extra momentum to achieve faster and more stable convergence than SGD, we integrate momentum into the diffusion process of DMs. This comes with a unique challenge of deriving the noise perturbation kernel from the momentum-based diffusion process. To this end, we frame the process as a Damped Oscillation system whose critically damped state -- the kernel solution -avoids oscillation and yields a faster convergence speed of the diffusion process. Empirical r
    
[^206]: 基于Implicit Neural Representations的时间序列连续建模用于插值和预测

    Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations. (arXiv:2306.05880v1 [cs.LG])

    [http://arxiv.org/abs/2306.05880](http://arxiv.org/abs/2306.05880)

    该论文提出了基于INR的时间序列连续建模方法，解决了处理缺失数据、不规则采样和多传感器不对准观测等重复建模问题，并在预测和插值任务中取得了最新的性能表现，具有很好的泛化能力。

    

    尽管时间序列建模已被广泛探索，但在面对真实世界的数据时仍面临重大挑战。我们提出了一种新颖的建模方法，利用Implicit Neural Representations (INR)。该方法使我们能够有效地捕捉时间序列的连续性，并提供了自然的解决方案，以处理缺失数据、处理不规则采样或来自多个传感器的不对准观测等重复建模问题。通过引入条件调制INR参数并利用元学习技术，我们解决了模型泛化到未见样本和时间窗口移位的问题。通过大量实验，我们的模型展示了在预测和插值任务中领先的性能，同时在处理许多竞争模型无法处理的各种具有挑战性的场景方面展现了灵活性。

    Although widely explored, time series modeling continues to encounter significant challenges when confronted with real-world data. We propose a novel modeling approach leveraging Implicit Neural Representations (INR). This approach enables us to effectively capture the continuous aspect of time series and provides a natural solution to recurring modeling issues such as handling missing data, dealing with irregular sampling, or unaligned observations from multiple sensors. By introducing conditional modulation of INR parameters and leveraging meta-learning techniques, we address the issue of generalization to both unseen samples and time window shifts. Through extensive experimentation, our model demonstrates state-of-the-art performance in forecasting and imputation tasks, while exhibiting flexibility in handling a wide range of challenging scenarios that competing models cannot.
    
[^207]: 跃迁于树空间：连续的树形系统推断方法用于有根和无根树

    Leaping through tree space: continuous phylogenetic inference for rooted and unrooted trees. (arXiv:2306.05739v1 [q-bio.PE])

    [http://arxiv.org/abs/2306.05739](http://arxiv.org/abs/2306.05739)

    本研究首次在连续空间中进行树形系统探索和推断，用于有根和无根树，优于当前最佳方法并在实验中证明了其效果，可用于加速生命科学的新进化发现。

    

    生物进化系统学现在是生命科学中的一个基础，可以阐明生命早期支系和传染病的起源和传播。然而，从可能的树的广阔空间中找到合适的系统树仍然具有挑战性。为了解决这个问题，我们首次在连续空间中进行了树形系统探索和推断，使梯度计算成为可能。这种连续的放松方式允许在有根和无根树中跨越树空间，且不易收敛到局部最小值。我们的方法优于当前最佳的无根树推断方法，并且在模拟中准确地推断出树和树根。该方法在实际数据中也很有效，我们在颌口动物的系统发育中证明了这一点。事实上，仅具有超指数信号的少数基因通常足以分辨脊椎动物的主要谱系。通过我们的方法，我们希望加速发现生命科学中的新进化发现。

    Phylogenetics is now fundamental in life sciences, providing insights into the earliest branches of life and the origins and spread of epidemics. However, finding suitable phylogenies from the vast space of possible trees remains challenging. To address this problem, for the first time, we perform both tree exploration and inference in a continuous space where the computation of gradients is possible. This continuous relaxation allows for major leaps across tree space in both rooted and unrooted trees, and is less susceptible to convergence to local minima. Our approach outperforms the current best methods for inference on unrooted trees and, in simulation, accurately infers the tree and root in ultrametric cases. The approach is effective in cases of empirical data with negligible amounts of data, which we demonstrate on the phylogeny of jawed vertebrates. Indeed, only a few genes with an ultrametric signal were generally sufficient for resolving the major lineages of vertebrate. With
    
[^208]: 应用演绎验证技术验证思维链的推理过程

    Deductive Verification of Chain-of-Thought Reasoning. (arXiv:2306.03872v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.03872](http://arxiv.org/abs/2306.03872)

    本文旨在通过应用演绎验证技术，使语言模型能够进行明确而严谨的演绎推理，以确保其推理过程的可信度。

    

    大语言模型在各种推理任务中受益匪浅，特别是应用思维链提示可以使模型产生更全面的推理过程。然而，思维链的强调中间推理步骤可能会不慎导致产生幻觉和累积错误，从而限制模型解决复杂推理任务的能力。本文灵感来自于人类如何进行细致的演绎逻辑推理过程来解决任务，我们旨在使语言模型能够进行明确而严谨的演绎推理，并通过自我验证确保推理过程的可信度。然而，即使是像ChatGPT这样先进的模型，直接验证整个演绎推理过程的有效性也是具有挑战性的。因此，我们提出将推理验证过程分解为一系列逐步的子过程，每个过程只接收其必要的上下文和前提条件。

    Large Language Models (LLMs) significantly benefit from Chain-of-Thought (CoT) prompting in performing various reasoning tasks. While CoT allows models to produce more comprehensive reasoning processes, its emphasis on intermediate reasoning steps can inadvertently introduce hallucinations and accumulated errors, thereby limiting models' ability to solve complex reasoning tasks. Inspired by how humans engage in careful and meticulous deductive logical reasoning processes to solve tasks, we seek to enable language models to perform explicit and rigorous deductive reasoning, and also ensure the trustworthiness of their reasoning process through self-verification. However, directly verifying the validity of an entire deductive reasoning process is challenging, even with advanced models like ChatGPT. In light of this, we propose to decompose a reasoning verification process into a series of step-by-step subprocesses, each only receiving their necessary context and premises. To facilitate t
    
[^209]: 在单位球上学习表示：应用于在线连续学习

    Learning Representations on the Unit Sphere: Application to Online Continual Learning. (arXiv:2306.03364v1 [cs.LG])

    [http://arxiv.org/abs/2306.03364](http://arxiv.org/abs/2306.03364)

    该论文提出了一种基于单位球的表示学习方法，通过将表示推向固定方向，使得学习策略对数据漂移具有弹性，从而能够应对在线连续学习的挑战性问题。

    

    我们使用最大后验估计原理来学习分布在单位球上的表示。我们针对对称方向数据建立了 von Mises-Fisher 分布和角高斯分布的损失函数。我们方法的一个显著特点是，学习到的表示被推向固定的方向，使得学习策略对数据漂移具有弹性。这使得它适合于在线连续学习，即在连续的数据流上训练神经网络的问题，其中多个分类任务按顺序呈现，因此过去任务的数据不再可用，当前任务的数据只能看一次。为了应对这种具有挑战性的情况，我们提出了一种基于记忆的表示学习技术，配备了我们的新损失函数。我们的方法不需要负数据或任务边界的知识，并且在较小的批处理下表现良好。

    We use the maximum a posteriori estimation principle for learning representations distributed on the unit sphere. We derive loss functions for the von Mises-Fisher distribution and the angular Gaussian distribution, both designed for modeling symmetric directional data. A noteworthy feature of our approach is that the learned representations are pushed toward fixed directions, allowing for a learning strategy that is resilient to data drift. This makes it suitable for online continual learning, which is the problem of training neural networks on a continuous data stream, where multiple classification tasks are presented sequentially so that data from past tasks are no longer accessible, and data from the current task can be seen only once. To address this challenging scenario, we propose a memory-based representation learning technique equipped with our new loss functions. Our approach does not require negative data or knowledge of task boundaries and performs well with smaller batch s
    
[^210]: 生成扩散在三维湍流流动中的应用

    Generative Diffusion for 3D Turbulent Flows. (arXiv:2306.01776v1 [physics.flu-dyn])

    [http://arxiv.org/abs/2306.01776](http://arxiv.org/abs/2306.01776)

    该论文提出了一种生成模型，可以在任意三维空间中模拟湍流现象，避免了湍流流动的不可预测性，能够快速生成高质量的流场。

    

    湍流流动通常难以预测，但二维和三维的湍流流动性质不同。在二维情况下，湍流会形成大的、连续的结构，而在三维情况下，旋涡级联成越来越小的尺度，形成许多快速变化的小尺度结构，加剧了不可预测性，难以使用回归方法。本文提出了第一个生成模型，可以在任意三维空间中模拟湍流现象，并引入了一种基于Wasserstein距离的流场质量度量方法。在多个实验中，我们证明了我们的生成扩散模型可以避免湍流流动的不可预测性，并且可以只依靠几何信息生成高质量的样本。此外，我们还展示了我们的模型可以比工业级数值求解器更快地生成湍流流场。

    Turbulent flows are well known to be chaotic and hard to predict; however, their dynamics differ between two and three dimensions. While 2D turbulence tends to form large, coherent structures, in three dimensions vortices cascade to smaller and smaller scales. This cascade creates many fast-changing, small-scale structures and amplifies the unpredictability, making regression-based methods infeasible. We propose the first generative model for forced turbulence in arbitrary 3D geometries and introduce a sample quality metric for turbulent flows based on the Wasserstein distance of the generated velocity-vorticity distribution. In several experiments, we show that our generative diffusion model circumvents the unpredictability of turbulent flows and produces high-quality samples based solely on geometric information. Furthermore, we demonstrate that our model beats an industrial-grade numerical solver in the time to generate a turbulent flow field from scratch by an order of magnitude.
    
[^211]: 大批量神经多目标贝叶斯优化

    Large-Batch, Neural Multi-Objective Bayesian Optimization. (arXiv:2306.01095v1 [cs.LG])

    [http://arxiv.org/abs/2306.01095](http://arxiv.org/abs/2306.01095)

    本文提出了一种针对数据密集型问题和多目标优化设置的贝叶斯优化框架，该方法利用了贝叶斯神经网络代理建模和可扩展、具有不确定性的收购策略，能够在最少迭代次数的情况下高效地进行优化。

    

    贝叶斯优化在全局优化黑盒高成本函数方面提供了强大的框架。然而，由于默认高斯过程代理的可扩展性差，它在处理数据密集型问题，特别是在多目标设置中的能力有限。本文提出了一种新颖的贝叶斯优化框架，专为解决这些限制而设计。我们的方法利用了贝叶斯神经网络方法进行代理建模。这使得它能够有效地处理大批量数据，建模复杂问题以及产生预测的不确定性。此外，我们的方法结合了一种基于众所周知且易于部署的NSGA-II的可扩展的、具有不确定性的收购策略。这种完全可并行化的策略促进了未勘探区域的有效探索。我们的框架允许在最少迭代次数的情况下在数据密集环境中进行有效的优化。我们展示了我们方法的优越性。

    Bayesian optimization provides a powerful framework for global optimization of black-box, expensive-to-evaluate functions. However, it has a limited capacity in handling data-intensive problems, especially in multi-objective settings, due to the poor scalability of default Gaussian Process surrogates. We present a novel Bayesian optimization framework specifically tailored to address these limitations. Our method leverages a Bayesian neural networks approach for surrogate modeling. This enables efficient handling of large batches of data, modeling complex problems, and generating the uncertainty of the predictions. In addition, our method incorporates a scalable, uncertainty-aware acquisition strategy based on the well-known, easy-to-deploy NSGA-II. This fully parallelizable strategy promotes efficient exploration of uncharted regions. Our framework allows for effective optimization in data-intensive environments with a minimum number of iterations. We demonstrate the superiority of ou
    
[^212]: 深度随机力学

    Deep Stochastic Mechanics. (arXiv:2305.19685v1 [cs.LG])

    [http://arxiv.org/abs/2305.19685](http://arxiv.org/abs/2305.19685)

    本文提出了一种基于深度学习的方法，用于数值模拟时间演化薛定谔方程，利用马尔可夫扩散采样来适应波函数的潜在低维结构，并提出了新的随机量子力学方程，具有线性的计算复杂度。数值模拟显示出显着的优势。

    

    本文引入了一种基于深度学习的方法，用于数值模拟时间演化薛定谔方程，受随机力学和生成性扩散模型的启发。与现有方法不同的是，我们的方法允许我们通过从马尔可夫扩散中采样来适应波函数潜在的低维结构，因此可以在更高的维度上降低计算复杂度。此外，我们提出了新的随机量子力学方程，结果具有与维数数量线性的计算复杂度。数值模拟验证了我们的理论发现，并显示出我们的方法与其他用于量子力学的基于深度学习的方法相比具有显着优势。

    This paper introduces a novel deep-learning-based approach for numerical simulation of a time-evolving Schr\"odinger equation inspired by stochastic mechanics and generative diffusion models. Unlike existing approaches, which exhibit computational complexity that scales exponentially in the problem dimension, our method allows us to adapt to the latent low-dimensional structure of the wave function by sampling from the Markovian diffusion. Depending on the latent dimension, our method may have far lower computational complexity in higher dimensions. Moreover, we propose novel equations for stochastic quantum mechanics, resulting in linear computational complexity with respect to the number of dimensions. Numerical simulations verify our theoretical findings and show a significant advantage of our method compared to other deep-learning-based approaches used for quantum mechanics.
    
[^213]: 基于递增分辨率的量化随机梯度 langevin 动力学

    Stochastic Gradient Langevin Dynamics Based on Quantization with Increasing Resolution. (arXiv:2305.18864v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.18864](http://arxiv.org/abs/2305.18864)

    本文提出了一种基于递增分辨率的量化随机梯度 langevin 动力学方法，通过利用 langevin 随机微分方程动力学，实现了具有可控噪声且具有相同分布的优化过程，无需添加噪声或调整小批量的大小。实验结果证明了该方法在不同数据集上对卷积神经网络模型和 ResNet-50 架构的有效性。

    

    基于 langevin 或 levy 随机微分方程的随机学习动力学通过改变小批量的大小或直接注入噪声的大小来控制噪声的方差。由于噪声方差会影响近似性能，在基于 SDE 的学习和实际实现中，添加噪声的设计很重要。本文提出了一种基于量化优化的随机下降学习方程，采用随机分析的视角，用于非凸目标函数。所提出的方法采用了一种利用 langevin SDE 动力学的量化优化方法，可以实现具有相同分布的可控噪声，而无需添加噪声或调整小批量的大小。数值实验证明了所提出算法在各种数据集上对 vanilla 卷积神经网络（CNN）模型和 ResNet-50 架构的有效性。

    Stochastic learning dynamics based on Langevin or Levy stochastic differential equations (SDEs) in deep neural networks control the variance of noise by varying the size of the mini-batch or directly those of injecting noise. Since the noise variance affects the approximation performance, the design of the additive noise is significant in SDE-based learning and practical implementation. In this paper, we propose an alternative stochastic descent learning equation based on quantized optimization for non-convex objective functions, adopting a stochastic analysis perspective. The proposed method employs a quantized optimization approach that utilizes Langevin SDE dynamics, allowing for controllable noise with an identical distribution without the need for additive noise or adjusting the mini-batch size. Numerical experiments demonstrate the effectiveness of the proposed algorithm on vanilla convolution neural network(CNN) models and the ResNet-50 architecture across various data sets. Fur
    
[^214]: 改进的概率图像-文本表示方法

    Improved Probabilistic Image-Text Representations. (arXiv:2305.18171v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2305.18171](http://arxiv.org/abs/2305.18171)

    本论文提出了一种改进的概率图像-文本表示方法，通过引入新的概率距离和两种优化技术，解决了现有方法中的计算负担过重和损失饱和问题，取得了显著的性能提升。

    

    图像-文本匹配是一种基本的视觉-语言任务，由于多样性和不完美注释导致的固有歧义使其受到困扰。确定性函数无法足够强大地捕捉到这种歧义，因此需要探索概率嵌入来解决这个挑战。然而，现有的概率图像-文本匹配方法存在两个关键缺点：蒙特卡洛逼近导致计算负担较重，且在大量误检情况下容易出现损失饱和问题。为了克服这些问题，本文提出了一种改进的概率跨模态嵌入方法（PCME++），通过引入具有闭合形式解的新的概率距离。此外，还提出了两种优化技术进一步增强PCME++：首先，引入伪正样本以防止大量误检情况下的损失饱和问题；其次，采用混合样本数据增强进行概率匹配。实验结果表明，PCME++在ITM任务中取得了显著的性能提升。

    Image-Text Matching (ITM) task, a fundamental vision-language (VL) task, suffers from the inherent ambiguity arising from multiplicity and imperfect annotations. Deterministic functions are not sufficiently powerful to capture ambiguity, prompting the exploration of probabilistic embeddings to tackle the challenge. However, the existing probabilistic ITM approach encounters two key shortcomings; the burden of heavy computations due to the Monte Carlo approximation, and the loss saturation issue in the face of abundant false negatives. To overcome the issues, this paper presents an improved Probabilistic Cross-Modal Embeddings (named PCME++) by introducing a new probabilistic distance with a closed-form solution. In addition, two optimization techniques are proposed to enhance PCME++ further; first, the incorporation of pseudo-positives to prevent the loss saturation problem under massive false negatives; second, mixed sample data augmentation for probabilistic matching. Experimental re
    
[^215]: 旋转优化器：简单而强健的深度神经网络训练。

    Rotational Optimizers: Simple & Robust DNN Training. (arXiv:2305.17212v1 [cs.LG])

    [http://arxiv.org/abs/2305.17212](http://arxiv.org/abs/2305.17212)

    该论文提出了旋转优化器，这些优化器可以简化深度神经网络训练过程，甚至在几乎不需调整基线超参数的情况下与原始优化器的性能相匹配。

    

    现代深度神经网络的训练动态取决于学习率、权重衰减、初始化等超参数之间的复杂交互作用。这些交互作用可以在尺度不变层（如归一化层）中产生球面运动动态，这些动态收敛到平衡状态，其中权重范数和预期旋转更新大小是固定的。我们对AdamW、带动量的SGD和Lion中的这个平衡进行了分析，提供了关于不同超参数及其相互作用对训练过程的影响的新见解。我们提出了这些优化器的旋转变体（RVs），强制预期角度更新大小与整个训练期间的平衡值相匹配。这简化了训练动态，通过消除收敛到平衡状态的瞬态相应。我们的旋转优化器可以匹配原始变体的性能，通常需要对基线超参数进行最少或不调整。

    The training dynamics of modern deep neural networks depend on complex interactions between the learning rate, weight decay, initialization, and other hyperparameters. These interactions can give rise to Spherical Motion Dynamics in scale-invariant layers (e.g., normalized layers), which converge to an equilibrium state, where the weight norm and the expected rotational update size are fixed. Our analysis of this equilibrium in AdamW, SGD with momentum, and Lion provides new insights into the effects of different hyperparameters and their interactions on the training process. We propose rotational variants (RVs) of these optimizers that force the expected angular update size to match the equilibrium value throughout training. This simplifies the training dynamics by removing the transient phase corresponding to the convergence to an equilibrium. Our rotational optimizers can match the performance of the original variants, often with minimal or no tuning of the baseline hyperparameters,
    
[^216]: 从单个图像中提取多个概念的场景分解：破解现场

    Break-A-Scene: Extracting Multiple Concepts from a Single Image. (arXiv:2305.16311v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2305.16311](http://arxiv.org/abs/2305.16311)

    该论文提出了一种从单个图像中提取多个概念的文本场景分解方法，通过增加目标概念的掩码和优化文本嵌入和模型权重的方式，实现对生成场景的精细控制。

    

    文本到图像模型个性化的目标是引入用户提供的概念到模型中，以便在不同的情境中合成。然而，当前的方法主要集中在从具有不同背景和姿势变化的多个图像中学习单个概念的情况，当应用到不同的场景时会遇到困难。在这项工作中，我们引入了文本场景分解的任务：给定一个可能包含多个概念的场景的单个图像，我们旨在提取每个概念的独特文本标记，从而对生成的场景进行精细控制。为此，我们提出了一种增强输入图像的方法，用来指示目标概念的存在的掩码。这些掩码可以由用户提供，也可以由预训练的分割模型自动生成。然后，我们提出了一种新颖的两阶段定制流程，优化一组专用的文本嵌入（句柄）以及模型权重，以在准确捕捉概念的同时保持平衡。

    Text-to-image model personalization aims to introduce a user-provided concept to the model, allowing its synthesis in diverse contexts. However, current methods primarily focus on the case of learning a single concept from multiple images with variations in backgrounds and poses, and struggle when adapted to a different scenario. In this work, we introduce the task of textual scene decomposition: given a single image of a scene that may contain several concepts, we aim to extract a distinct text token for each concept, enabling fine-grained control over the generated scenes. To this end, we propose augmenting the input image with masks that indicate the presence of target concepts. These masks can be provided by the user or generated automatically by a pre-trained segmentation model. We then present a novel two-phase customization process that optimizes a set of dedicated textual embeddings (handles), as well as the model weights, striking a delicate balance between accurately capturin
    
[^217]: 通过事后对数归一化和温度缩放改善深度神经网络的选择分类性能

    Improving selective classification performance of deep neural networks through post-hoc logit normalization and temperature scaling. (arXiv:2305.15508v1 [cs.LG])

    [http://arxiv.org/abs/2305.15508](http://arxiv.org/abs/2305.15508)

    本文提出了一种$p$-NormSoftmax的事后置信度估计器来提高深度神经网络的选择分类性能。

    

    本文解决深度神经网络的选择分类问题，其中模型可以避免潜在错误通过放弃低置信度的预测。我们针对的是优化固定分类器的置信度估计器，旨在增强其误分类检测性能，即通过将更高的置信度值分配给正确的预测来区分正确和不正确的预测。我们提出了一个简单有效的事后置信度估计器$p$-NormSoftmax，通过对数进行$p$-范数归一化和温度缩放得到。

    This paper addresses the problem of selective classification for deep neural networks, where a model is allowed to abstain from low-confidence predictions to avoid potential errors. Specifically, we tackle the problem of optimizing the confidence estimator of a fixed classifier, aiming to enhance its misclassification detection performance, i.e., its ability to discriminate between correct and incorrect predictions by assigning higher confidence values to the correct ones. Previous work has found that different classifiers exhibit varying levels of misclassification detection performance, particularly when using the maximum softmax probability (MSP) as a measure of confidence. However, we argue that these findings are mainly due to a sub-optimal confidence estimator being used for each model. To overcome this issue, we propose a simple and efficient post-hoc confidence estimator, named $p$-NormSoftmax, which consists of transforming the logits through $p$-norm normalization and tempera
    
[^218]: Twitter图像的文本条件下的替代文本生成

    Text Conditional Alt-Text Generation for Twitter Images. (arXiv:2305.14779v1 [cs.CV])

    [http://arxiv.org/abs/2305.14779](http://arxiv.org/abs/2305.14779)

    本文针对Twitter上分享的图像提出了一种文本条件下的替代文本生成方法。通过CLIP前缀模型，该模型结合图像和推文中的文本信息，生成关于图像的上下文相关的替代文本。

    

    本文提出了一种针对社交媒体特别是Twitter上分享的图像生成替代文本（或alt-text）描述的方法。与图像的字幕不同，文本替换文本更加直白描述和上下文特定。此外，关键是，发布到Twitter上的图像通常是由用户编写的文本附加的，尽管这些文本不一定描述图像，但可能提供有用的上下文信息，如果正确利用可以提供信息，例如推文可能会命名图片中模型之前没有见过的不常见的对象。我们通过一个CLIP前缀模型来解决这个问题，该模型提取图像的嵌入并将其传递给映射网络，该网络输出单词嵌入空间中的短序列，或称为“前缀”，我们将推文本身的文本也连接到其中。这样，模型就可以在文章中条件化视觉和文本信息。然后将合并的多模式前缀作为预训练的语言模型的提示输入。

    In this work we present an approach for generating alternative text (or alt-text) descriptions for images shared on social media, specifically Twitter. This task is more than just a special case of image captioning, as alt-text is both more literally descriptive and context-specific. Also critically, images posted to Twitter are often accompanied by user-written text that despite not necessarily describing the image may provide useful context that if properly leveraged can be informative -- e.g. the tweet may name an uncommon object in the image that the model has not previously seen. We address this with a CLIP prefix model that extracts an embedding of the image and passes it to a mapping network that outputs a short sequence in word embedding space, or a ``prefix'', to which we also concatenate the text from the tweet itself. This lets the model condition on both visual and textual information from the post. The combined multimodal prefix is then fed as a prompt to a pretrained lang
    
[^219]: 通过非平衡最优输运半对偶公式的生成建模

    Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport. (arXiv:2305.14777v1 [cs.CV])

    [http://arxiv.org/abs/2305.14777](http://arxiv.org/abs/2305.14777)

    本文提出了一种基于非平衡最优输运半对偶公式的新型生成模型，相比于OT，它具有更好的鲁棒性、稳定性和更快的收敛速度，实验结果表明其优于现有的基于OT的生成模型。

    

    最优输运（OT）问题研究一种运输映射，能够在最小化给定成本函数的同时连接两个分布。在这方面，OT已被用于生成建模任务中的可追溯的先验分布和数据之间。然而，基于OT的方法容易受到离群点的影响，并在训练期间面临优化挑战。在本文中，我们提出了一种基于非平衡最优输运（UOT）半对偶公式的新型生成模型。与OT不同，UOT消除了分布匹配的硬性约束，提供了更好的对抗离群点的鲁棒性，训练期间的稳定性以及更快的收敛速度。我们通过实验验证了这些属性。此外，我们还研究了UOT之间分布差异的理论上限。我们的模型优于现有的基于OT的生成模型，在CIFAR-10和CelebA-HQ-256上实现了分别为2.97和5.80的FID分数。

    Optimal Transport (OT) problem investigates a transport map that bridges two distributions while minimizing a given cost function. In this regard, OT between tractable prior distribution and data has been utilized for generative modeling tasks. However, OT-based methods are susceptible to outliers and face optimization challenges during training. In this paper, we propose a novel generative model based on the semi-dual formulation of Unbalanced Optimal Transport (UOT). Unlike OT, UOT relaxes the hard constraint on distribution matching. This approach provides better robustness against outliers, stability during training, and faster convergence. We validate these properties empirically through experiments. Moreover, we study the theoretical upper-bound of divergence between distributions in UOT. Our model outperforms existing OT-based generative models, achieving FID scores of 2.97 on CIFAR-10 and 5.80 on CelebA-HQ-256.
    
[^220]: 通过伪神经切线核代理模型提供深度神经网络的鲁棒性解释

    Robust Explanations for Deep Neural Networks via Pseudo Neural Tangent Kernel Surrogate Models. (arXiv:2305.14585v1 [cs.LG])

    [http://arxiv.org/abs/2305.14585](http://arxiv.org/abs/2305.14585)

    本研究通过建立一个规范化的伪神经切线核，证明了它能够更好地与神经网络决策函数相关，比基于嵌入和影响的替代品更有效，并且从它创建的归因会更准确地选择被扰动的训练数据，从而证明了核线性模型是跨多个数据领域并有效的替代模型。

    

    最近，通过数据归属任务，解释型AI的进步之一是通过解释示例策略实现的。然而，用于将决策归因于训练数据的特征空间，尚未相互比较，以确定它们是否形成神经网络(NN)的真正代理模型。在这里，我们通过两种方式证明了线性特征空间对神经网络的有效性：(1)我们建立了一个规范化的伪神经切线核(pNTK)，它在计算机视觉和大语言模型架构中与神经网络决策函数更相关，比基于嵌入和影响的替代品更为有效；(2)我们展示了从规范化pNTK创建的归因比这些替代品更准确地选择被扰动的训练数据。基于这些观察结果，我们得出结论，核线性模型是跨多个数据领域并有效的替代模型。

    One of the ways recent progress has been made on explainable AI has been via explain-by-example strategies, specifically, through data attribution tasks. The feature spaces used to attribute decisions to training data, however, have not been compared against one another as to whether they form a truly representative surrogate model of the neural network (NN). Here, we demonstrate the efficacy of surrogate linear feature spaces to neural networks through two means: (1) we establish that a normalized psuedo neural tangent kernel (pNTK) is more correlated to the neural network decision functions than embedding based and influence based alternatives in both computer vision and large language model architectures; (2) we show that the attributions created from the normalized pNTK more accurately select perturbed training data in a data poisoning attribution task than these alternatives. Based on these observations, we conclude that kernel linear models are effective surrogate models across m
    
[^221]: 面向不均衡数据的鲁棒基于模型的设计的属性引导生成建模

    Property-Guided Generative Modelling for Robust Model-Based Design with Imbalanced Data. (arXiv:2305.13650v1 [cs.LG])

    [http://arxiv.org/abs/2305.13650](http://arxiv.org/abs/2305.13650)

    本文提出了一种属性引导的变分自编码器（PGVAE），通过属性值明确结构化潜在空间，使得MBO可以在不平衡数据上稳健地寻找具有改进属性的序列。

    

    设计具有特定属性的蛋白质序列是一项具有挑战性的任务，因为这需要探索具有极度稀疏的有意义区域的高维蛋白质序列空间。这导致了模型优化（MBO）技术的发展，通过使用由序列空间中的属性引导的有效搜索模型来辅助设计。然而，实验获得的数据集的内在不平衡性使得现有的MBO方法很难或根本无法处理。我们提出了一种属性引导的变分自编码器（PGVAE），其潜在空间由属性值明确结构化，使得按照这些属性值优先考虑样本。通过对真实和半合成蛋白质数据集的广泛基准测试，我们展示了MBO与PGVAE稳健地发现具有改进属性的序列，尽管数据集存在显著的不平衡性。我们进一步展示了我们的方法对于连续设计空间的普适性及其稳健性。

    The problem of designing protein sequences with desired properties is challenging, as it requires to explore a high-dimensional protein sequence space with extremely sparse meaningful regions. This has led to the development of model-based optimization (MBO) techniques that aid in the design, by using effective search models guided by the properties over the sequence space. However, the intrinsic imbalanced nature of experimentally derived datasets causes existing MBO approaches to struggle or outright fail. We propose a property-guided variational auto-encoder (PGVAE) whose latent space is explicitly structured by the property values such that samples are prioritized according to these properties. Through extensive benchmarking on real and semi-synthetic protein datasets, we demonstrate that MBO with PGVAE robustly finds sequences with improved properties despite significant dataset imbalances. We further showcase the generality of our approach to continuous design spaces, and its rob
    
[^222]: 有条件生成模型是标记时间点过程的必备工具。

    Conditional Generative Modeling is All You Need for Marked Temporal Point Processes. (arXiv:2305.12569v1 [stat.ML])

    [http://arxiv.org/abs/2305.12569](http://arxiv.org/abs/2305.12569)

    本文提出了一种从标记时间点过程中提取其统计直觉的事件生成模型，通过条件生成器以历史观察作为输入，生成可能发生的高质量随后事件。该模型具有高效、灵活和表示能力等方面的优势。

    

    近年来，生成建模的进步使得从上下文信息中生成高质量内容成为可能，但一个关键问题仍然存在：如何教模型知道何时生成内容？为了回答这个问题，本研究提出了一种新的事件生成模型，从标记时间点过程中提取其统计直觉，并提供了一个干净、灵活和计算效率高的解决方案，适用于涉及多维标记的各种应用。我们旨在捕捉点过程的分布而不需明确指定条件强度或概率密度。我们使用一个条件生成器，以事件历史为输入并生成在先前观察到的事件下，可能发生的高质量随后事件。所提出的框架提供了一系列利益，包括在学习模型和生成样本方面的异常效率以及相当大的表示能力来捕捉。

    Recent advancements in generative modeling have made it possible to generate high-quality content from context information, but a key question remains: how to teach models to know when to generate content? To answer this question, this study proposes a novel event generative model that draws its statistical intuition from marked temporal point processes, and offers a clean, flexible, and computationally efficient solution for a wide range of applications involving multi-dimensional marks. We aim to capture the distribution of the point process without explicitly specifying the conditional intensity or probability density. Instead, we use a conditional generator that takes the history of events as input and generates the high-quality subsequent event that is likely to occur given the prior observations. The proposed framework offers a host of benefits, including exceptional efficiency in learning the model and generating samples, as well as considerable representational power to capture
    
[^223]: Q-malizing流和无穷小密度比估计

    Q-malizing flow and infinitesimal density ratio estimation. (arXiv:2305.11857v1 [stat.ML])

    [http://arxiv.org/abs/2305.11857](http://arxiv.org/abs/2305.11857)

    研究提出了一种可以从一个数据分布P传输到任意访问通过有限样本的Q的流模型。这个模型通过神经ODE模型进行，可以进行无穷小DRE。

    

    连续的正则化流在生成任务中被广泛使用，其中流网络从数据分布P传输到正态分布。一种能够从P传输到任意Q的流模型，其中P和Q都可通过有限样本访问，将在各种应用兴趣中使用，特别是在最近开发的望远镜密度比估计中（DRE），它需要构建中间密度以在P和Q之间建立桥梁。在这项工作中，我们提出了这样的“Q-malizing流”，通过神经ODE模型进行，该模型通过经验样本的可逆传输从P到Q（反之亦然），并通过最小化传输成本进行正则化。训练好的流模型使我们能够沿与时间参数化的log密度进行无穷小DRE，通过训练附加的连续时间流网络使用分类损失来估计log密度的时间偏导数。通过积分时间得分网络

    Continuous normalizing flows are widely used in generative tasks, where a flow network transports from a data distribution $P$ to a normal distribution. A flow model that can transport from $P$ to an arbitrary $Q$, where both $P$ and $Q$ are accessible via finite samples, would be of various application interests, particularly in the recently developed telescoping density ratio estimation (DRE) which calls for the construction of intermediate densities to bridge between $P$ and $Q$. In this work, we propose such a ``Q-malizing flow'' by a neural-ODE model which is trained to transport invertibly from $P$ to $Q$ (and vice versa) from empirical samples and is regularized by minimizing the transport cost. The trained flow model allows us to perform infinitesimal DRE along the time-parametrized $\log$-density by training an additional continuous-time flow network using classification loss, which estimates the time-partial derivative of the $\log$-density. Integrating the time-score network
    
[^224]: 基于视觉的深度强化学习自动驾驶系统及Sim2Real转移

    Vision-based DRL Autonomous Driving Agent with Sim2Real Transfer. (arXiv:2305.11589v1 [cs.RO])

    [http://arxiv.org/abs/2305.11589](http://arxiv.org/abs/2305.11589)

    本研究提出了一种基于视觉的深度强化学习代理，可以同时执行保持车道和跟车操作，并且展示了其在真实情况下的模型迁移能力，是第一个具有此能力的代理。

    

    要实现完全自动驾驶，车辆必须能够持续执行各种驾驶任务，包括保持车道和跟车，这两个任务是基本的并且研究得很充分。然而，以前的研究主要集中在单个任务上，而跟车任务通常依赖完整的领导-跟随者信息来实现最佳性能。为解决这一限制，我们提出了一种基于视觉的深度强化学习（DRL）代理，可以同时执行保持车道和跟车操作。为了评估我们的DRL代理的性能，我们将其与基线控制器进行比较，并使用各种性能指标进行定量分析。此外，我们进行了现实世界的评估，以证明训练的DRL代理的Sim2Real转移能力。据我们所知，我们的基于视觉的保持车道和跟车代理及其Sim2Real转移能力是第一个这样的代理。

    To achieve fully autonomous driving, vehicles must be capable of continuously performing various driving tasks, including lane keeping and car following, both of which are fundamental and well-studied driving ones. However, previous studies have mainly focused on individual tasks, and car following tasks have typically relied on complete leader-follower information to attain optimal performance. To address this limitation, we propose a vision-based deep reinforcement learning (DRL) agent that can simultaneously perform lane keeping and car following maneuvers. To evaluate the performance of our DRL agent, we compare it with a baseline controller and use various performance metrics for quantitative analysis. Furthermore, we conduct a real-world evaluation to demonstrate the Sim2Real transfer capability of the trained DRL agent. To the best of our knowledge, our vision-based car following and lane keeping agent with Sim2Real transfer capability is the first of its kind.
    
[^225]: 利用Riesz核的生成式分割MMD流

    Generative Sliced MMD Flows with Riesz Kernels. (arXiv:2305.11463v1 [cs.LG])

    [http://arxiv.org/abs/2305.11463](http://arxiv.org/abs/2305.11463)

    本文使用Riesz核展示了生成式分割MMD流的高效计算方法，实现了在大规模应用中通过神经网络训练生成模型。

    

    在大规模计算中，最大平均差异度(MMD)流的计算成本很高。在本文中，我们展示了使用Riesz核$K(x,y)=-\|x-y\|^r$，$r \in (0,2)$的MMD流具有杰出的性质，可允许其进行高效计算。首先，Riesz核的MMD与其分割版本的MMD重合。因此，可以在一维设置中进行MMD梯度的计算。在此处，对于$r=1$，可以应用简单的排序算法将两个经验度量的复杂度从$O(MN+N^2)$降低到$O((M+N)\log(M+N))$，其中$M$和$N$是支持点。对于实现，我们通过仅使用有限数量的$P$个切片来近似分割MMD的梯度。我们展示了由此产生的误差具有$O(\sqrt{d/P})$的复杂度，其中$d$是数据维度。这些结果使我们能够通过神经网络近似MMD梯度流来训练生成模型，甚至用于大规模应用。

    Maximum mean discrepancy (MMD) flows suffer from high computational costs in large scale computations. In this paper, we show that MMD flows with Riesz kernels $K(x,y) = - \|x-y\|^r$, $r \in (0,2)$ have exceptional properties which allow for their efficient computation. First, the MMD of Riesz kernels coincides with the MMD of their sliced version. As a consequence, the computation of gradients of MMDs can be performed in the one-dimensional setting. Here, for $r=1$, a simple sorting algorithm can be applied to reduce the complexity from $O(MN+N^2)$ to $O((M+N)\log(M+N))$ for two empirical measures with $M$ and $N$ support points. For the implementations we approximate the gradient of the sliced MMD by using only a finite number $P$ of slices. We show that the resulting error has complexity $O(\sqrt{d/P})$, where $d$ is the data dimension. These results enable us to train generative models by approximating MMD gradient flows by neural networks even for large scale applications. We demo
    
[^226]: 一种参数高效的学习方法，用于带有预训练通用语音模型的阿拉伯方言识别

    A Parameter-Efficient Learning Approach to Arabic Dialect Identification with Pre-Trained General-Purpose Speech Model. (arXiv:2305.11244v1 [cs.CL])

    [http://arxiv.org/abs/2305.11244](http://arxiv.org/abs/2305.11244)

    本文介绍了一种利用预训练通用语音模型进行阿拉伯方言识别的参数高效学习方法，通过残差适配器和模型重编程，设计了一个基于记号的标签映射，并在ADI-17数据集上实现了最高精度，同时使用PEL方法进一步减少了训练成本。

    

    本文研究了参数高效学习（PEL）技术，以重新利用通用语音模型（GSM）进行阿拉伯方言识别（ADI）。我们设计了一个基于记号的标签映射，将GSM适应于阿拉伯方言识别，通过残差适配器和模型重编程来实现。我们通过vanilla fine-tuning在ADI-17数据集上实现了新的最高精度。此外，我们通过PEL方法进一步减少了训练成本，使用额外2.5％的网络可训练参数即可达到fine-tuning精度的1.86％。我们的研究展示了如何使用小型数据集和有限的计算资源来识别阿拉伯方言。

    In this work, we explore Parameter-Efficient-Learning (PEL) techniques to repurpose a General-Purpose-Speech (GSM) model for Arabic dialect identification (ADI). Specifically, we investigate different setups to incorporate trainable features into a multi-layer encoder-decoder GSM formulation under frozen pre-trained settings. Our architecture includes residual adapter and model reprogramming (input-prompting). We design a token-level label mapping to condition the GSM for Arabic Dialect Identification (ADI). This is challenging due to the high variation in vocabulary and pronunciation among the numerous regional dialects. We achieve new state-of-the-art accuracy on the ADI-17 dataset by vanilla fine-tuning. We further reduce the training budgets with the PEL method, which performs within 1.86% accuracy to fine-tuning using only 2.5% of (extra) network trainable parameters. Our study demonstrates how to identify Arabic dialects using a small dataset and limited computation with open sou
    
[^227]: PDP：无需参数的可微剪枝即可搞定

    PDP: Parameter-free Differentiable Pruning is All You Need. (arXiv:2305.11203v1 [cs.LG])

    [http://arxiv.org/abs/2305.11203](http://arxiv.org/abs/2305.11203)

    PDP提出了一种无需参数的可微剪枝方案，具有最先进的模型大小、准确性和训练成本，适用于各种视觉和自然语言任务。

    

    DNN剪枝是一种常用的方法，可以减少模型的大小，提高推理延迟，并最小化DNN加速器上的功耗。然而，现有的方法可能过于复杂、昂贵或无法适用于各种视觉/语言任务、DNN体系结构并遵守结构化剪枝约束。在本文中，我们提出了一种高效而有效的训练时间剪枝方案——PDP（参数自由可微剪枝），它在模型大小、准确性和训练成本方面具有最先进的性能。PDP在训练过程中使用权重的动态函数，以参数无关的方式为给定的剪枝目标生成软剪枝掩码。虽然是可微的，但是PDP的简单和高效使其足够普遍，以在各种视觉和自然语言任务上提供最先进的随机/结构化/通道剪枝结果。例如，对于MobileNet-v1，PDP可以在86.6%的稀疏度下达到68.2%的ImageNet1k top-1准确率。

    DNN pruning is a popular way to reduce the size of a model, improve the inference latency, and minimize the power consumption on DNN accelerators. However, existing approaches might be too complex, expensive or ineffective to apply to a variety of vision/language tasks, DNN architectures and to honor structured pruning constraints. In this paper, we propose an efficient yet effective train-time pruning scheme, Parameter-free Differentiable Pruning (PDP), which offers state-of-the-art qualities in model size, accuracy, and training cost. PDP uses a dynamic function of weights during training to generate soft pruning masks for the weights in a parameter-free manner for a given pruning target. While differentiable, the simplicity and efficiency of PDP make it universal enough to deliver state-of-the-art random/structured/channel pruning results on various vision and natural language tasks. For example, for MobileNet-v1, PDP can achieve 68.2% top-1 ImageNet1k accuracy at 86.6% sparsity, wh
    
[^228]: 自监督神经因子分析解耦语音表示

    Self-supervised Neural Factor Analysis for Disentangling Utterance-level Speech Representations. (arXiv:2305.08099v1 [cs.SD])

    [http://arxiv.org/abs/2305.08099](http://arxiv.org/abs/2305.08099)

    本文提出了一种自监督神经因子分析模型，使用HuBERT中的聚类方法来发现隐藏的声学单元，并使用这些单元对齐SSL模型的特征，从而产生解耦后的语音表示，从而为专门任务提供了一种基于Utterance水平的无监督学习目标。实验结果表明，SSNFA模型在说话人识别、语言识别和情感识别等各种任务中均显著优于现有的SSL模型，并且没有任何特定任务的微调或监督。

    

    自监督学习技术在自动语音识别方面已经展示了出色的性能，在低标注资源情况下证明非常有用，本文针对该技术在说话人、情感和语言识别等任务中的性能问题进行了探究。本文提出了一种因子分析模型，使用HuBERT中的聚类方法来发现隐藏的声学单元，并使用这些单元对齐SSL模型的特征，从而产生解耦后的语音表示，从而为专门任务提供了一种基于Utterance水平的无监督学习目标。实验结果表明，SSNFA模型在说话人识别、语言识别和情感识别等各种任务中均显著优于现有的SSL模型，并且没有任何特定任务的微调或监督。

    Self-supervised learning (SSL) speech models such as wav2vec and HuBERT have demonstrated state-of-the-art performance on automatic speech recognition (ASR) and proved to be extremely useful in low label-resource settings. However, the success of SSL models has yet to transfer to utterance-level tasks such as speaker, emotion, and language recognition, which still require supervised fine-tuning of the SSL models to obtain good performance. We argue that the problem is caused by the lack of disentangled representations and an utterance-level learning objective for these tasks. Inspired by how HuBERT uses clustering to discover hidden acoustic units, we formulate a factor analysis (FA) model that uses the discovered hidden acoustic units to align the SSL features. The underlying utterance-level representations are disentangled from the content of speech using probabilistic inference on the aligned features. Furthermore, the variational lower bound derived from the FA model provides an ut
    
[^229]: 个性化一次性分割模型

    Personalize Segment Anything Model with One Shot. (arXiv:2305.03048v1 [cs.CV])

    [http://arxiv.org/abs/2305.03048](http://arxiv.org/abs/2305.03048)

    本文提出了一种无需训练的SAM个性化方法PerSAM，只需要一张带有参考掩模的单张图像即可定位和分割目标概念，还提出了高效的一次性微调变体PerSAM-F，旨在解决掩模不确定性问题。

    

    在大数据预训练的推动下，分割任何物体模型（SAM）已被证明是一个强大且高效的框架，革新了分割模型领域。尽管SAM非常通用，但自动为特定视觉概念定制SAM而不需要手动提示，如在不同图像中自动分割你的宠物狗等， 还未深入研究。本文提出了一种无需训练的SAM个性化方法，称为PerSAM。只需要一张带有参考掩模的单张图像，PerSAM首先通过位置先验定位目标概念，并通过三种技术来在其他图像或视频中分割它：目标引导注意力，目标语义提示和级联后处理。这样，我们有效地适应了SAM的私人使用而无需任何训练。为了进一步缓解掩模的不确定性，我们提出了一个高效的一次性微调变体，即PerSAM-F。冻结整个SAM，我们引入了两个可学习权重用于多尺度掩模，仅训练2个参数即可。

    Driven by large-data pre-training, Segment Anything Model (SAM) has been demonstrated as a powerful and promptable framework, revolutionizing the segmentation models. Despite the generality, customizing SAM for specific visual concepts without man-powered prompting is under explored, e.g., automatically segmenting your pet dog in different images. In this paper, we propose a training-free Personalization approach for SAM, termed as PerSAM. Given only a single image with a reference mask, PerSAM first localizes the target concept by a location prior, and segments it within other images or videos via three techniques: target-guided attention, target-semantic prompting, and cascaded post-refinement. In this way, we effectively adapt SAM for private use without any training. To further alleviate the mask ambiguity, we present an efficient one-shot fine-tuning variant, PerSAM-F. Freezing the entire SAM, we introduce two learnable weights for multi-scale masks, only training 2 parameters wit
    
[^230]: 结构化稀疏动态训练

    Dynamic Sparse Training with Structured Sparsity. (arXiv:2305.02299v1 [cs.LG])

    [http://arxiv.org/abs/2305.02299](http://arxiv.org/abs/2305.02299)

    本文提出了一种结构化稀疏动态训练（DST）方法，学习一种变体的结构化 N:M 稀疏性，其加速在一般情况下通常被支持，可缩减参数和内存占用，同时相较于密集模型，具有减少推理时间的优势。

    

    动态稀疏训练在稀疏神经网络训练中取得了最先进的结果，并匹配了密集模型的泛化性，同时使得稀疏训练和推理成为可能。尽管得到的模型高度稀疏，理论上训练更便宜，但在实际硬件上，使用非结构化稀疏性加速依然具有人们所面临的挑战。在本文中，我们提出一种 DST 方法，学习一种变体的结构化 N:M 稀疏性，其加速在一般情况下通常被支持。此外，我们通过理论分析和实证结果，证明了特定 N:M 稀疏方法（常数扇入）的泛化性能，并展示了一种缩减参数和内存占用的紧凑表示。经过对 PyTorch CPU 实现的简单表示进行推断，我们证明了相较于密集模型，该方法减少了推理时间。我们的源代码可在 https://github.com/calgaryml/condensed-sparsity 上获得。

    DST methods achieve state-of-the-art results in sparse neural network training, matching the generalization of dense models while enabling sparse training and inference. Although the resulting models are highly sparse and theoretically cheaper to train, achieving speedups with unstructured sparsity on real-world hardware is challenging. In this work we propose a DST method to learn a variant of structured N:M sparsity, the acceleration of which in general is commonly supported in commodity hardware. Furthermore, we motivate with both a theoretical analysis and empirical results, the generalization performance of our specific N:M sparsity (constant fan-in), present a condensed representation with a reduced parameter and memory footprint, and demonstrate reduced inference time compared to dense models with a naive PyTorch CPU implementation of the condensed representation Our source code is available at https://github.com/calgaryml/condensed-sparsity
    
[^231]: 考察无监督的概念漂移检测的计算性能: 一项调查和更多

    Examining Computational Performance of Unsupervised Concept Drift Detection: A Survey and Beyond. (arXiv:2304.08319v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2304.08319](http://arxiv.org/abs/2304.08319)

    这篇论文调查了无监督概念漂移检测的计算性能，提出了一套指标来评估漂移检测器对AI系统的计算影响。

    

    概念漂移检测对于许多人工智能系统来说至关重要，以确保系统的可靠性。这些系统往往需要处理大量的数据或实时反应。因此，漂移检测器必须满足计算要求或约束，并进行全面的性能评估。然而，到目前为止，开发漂移检测器的重点是检测质量，如准确性，而不是计算性能，如运行时间。我们发现先前的工作只将计算性能视为次要目标，并没有针对这样的评估提出基准。因此，我们提出了一套指标，既考虑计算性能又考虑检测质量。其中，我们的指标集包括相对运行时间开销（RRO），用于评估漂移检测器对人工智能系统的计算影响。这项工作专注于无监督的漂移检测器，不受标记数据的可用性限制。我们基于

    Concept drift detection is crucial for many AI systems to ensure the system's reliability. These systems often have to deal with large amounts of data or react in real time. Thus, drift detectors must meet computational requirements or constraints with a comprehensive performance evaluation. However, so far, the focus of developing drift detectors is on detection quality, e.g.~accuracy, but not on computational performance, such as running time. We show that the previous works consider computational performance only as a secondary objective and do not have a benchmark for such evaluation. Hence, we propose a set of metrics that considers both, computational performance and detection quality. Among others, our set of metrics includes the Relative Runtime Overhead RRO to evaluate a drift detector's computational impact on an AI system. This work focuses on unsupervised drift detectors, not being restricted to the availability of labeled data. We measure the computational performance base
    
[^232]: 在深度神经网络中预防性修剪Clever Hans策略

    Preemptively Pruning Clever-Hans Strategies in Deep Neural Networks. (arXiv:2304.05727v1 [cs.LG])

    [http://arxiv.org/abs/2304.05727](http://arxiv.org/abs/2304.05727)

    本文提出了一种新方法，Explanation-Guided Exposure Minimization (EGEM)，该方法预防性地修剪了ML模型中未受到积极解释反馈的变化，从而大大减少了对隐藏Clever Hans策略的依赖，并实现了更高的性能。

    

    可解释的AI已成为验证机器学习模型的流行工具。解释模型的决策策略与用户的领域知识之间的不匹配（例如Clever Hans效应）也被认为是改进错误模型的起点。然而，当用户和解释达成一致时，要怎么做就不那么清楚了。本文通过展示用户接受解释并不保证ML模型的良好功能，特别是一些隐藏的Clever Hans效应可能仍然未被发现，证明了这一点。我们通过贡献一个新方法Explanation-Guided Exposure Minimization (EGEM)，该方法预防性地修剪了ML模型中未受到积极解释反馈的变化。自然画像数据的实验表明，我们的方法导致模型大大减少了对隐藏的Clever Hans策略的依赖，并因此实现了更高的性能。

    Explainable AI has become a popular tool for validating machine learning models. Mismatches between the explained model's decision strategy and the user's domain knowledge (e.g. Clever Hans effects) have also been recognized as a starting point for improving faulty models. However, it is less clear what to do when the user and the explanation agree. In this paper, we demonstrate that acceptance of explanations by the user is not a guarantee for a ML model to function well, in particular, some Clever Hans effects may remain undetected. Such hidden flaws of the model can nevertheless be mitigated, and we demonstrate this by contributing a new method, Explanation-Guided Exposure Minimization (EGEM), that premptively prunes variations in the ML model that have not been the subject of positive explanation feedback. Experiments on natural image data demonstrate that our approach leads to models that strongly reduce their reliance on hidden Clever Hans strategies, and consequently achieve hig
    
[^233]: 一种针对弱监督学习的基准生成性概率模型

    A Benchmark Generative Probabilistic Model for Weak Supervised Learning. (arXiv:2303.17841v1 [cs.LG])

    [http://arxiv.org/abs/2303.17841](http://arxiv.org/abs/2303.17841)

    本文提出一种基准生成性概率模型，在启发式标注的原始数据集上训练，生成伪标签作为一种准确、快速、经济的弱监督学习方法，在图像分类和自然语言处理中达到了最先进的表现。

    

    寻找相关高质量的数据集来训练机器学习模型对于实践者来说是一个主要 bottleneck。而且，为了解决野心勃勃实际应用场景下的问题，数据通常需要附带带有高质量注释的标签，以便于监督模型的训练。手动标记具有高质量标签的数据通常是一项耗时且具有挑战性的任务，往往成为机器学习项目的瓶颈。弱监督学习 (WSL) 方法已被开发出来，通过根据启发式、远程监视和知识库来赋予未标记数据大约标签 (伪标签) 的自动方式，从而减轻注释负担。我们应用概率生成隐变量模型 (PLVMs)，在启发式标注表示的原始数据集上进行训练，作为一种生成伪标签的准确、快速、经济的方式。我们展示了 PLVMs 在图像分类中的多个基准数据集上实现了最先进的表现，并展示了它们在自然语言处理中的事件检测任务上的多才多艺。

    Finding relevant and high-quality datasets to train machine learning models is a major bottleneck for practitioners. Furthermore, to address ambitious real-world use-cases there is usually the requirement that the data come labelled with high-quality annotations that can facilitate the training of a supervised model. Manually labelling data with high-quality labels is generally a time-consuming and challenging task and often this turns out to be the bottleneck in a machine learning project. Weak Supervised Learning (WSL) approaches have been developed to alleviate the annotation burden by offering an automatic way of assigning approximate labels (pseudo-labels) to unlabelled data based on heuristics, distant supervision and knowledge bases. We apply probabilistic generative latent variable models (PLVMs), trained on heuristic labelling representations of the original dataset, as an accurate, fast and cost-effective way to generate pseudo-labels. We show that the PLVMs achieve state-of-
    
[^234]: 深度学习模型转换器中的故障和风险分析：以ONNX生态系统的案例研究为例

    Analysis of Failures and Risks in Deep Learning Model Converters: A Case Study in the ONNX Ecosystem. (arXiv:2303.17708v1 [cs.SE])

    [http://arxiv.org/abs/2303.17708](http://arxiv.org/abs/2303.17708)

    本文详细分析了深度学习模型转换器的故障情况，特别是对ONNX相关的转换器进行了首次故障分析，并详细报告了故障的症状，原因和位置以及随时间的趋势。

    

    软件工程师开发，优化和部署深度学习模型。他们在各种开发框架中使用和重新使用模型，并在各种运行时环境中部署它们。在这个多样化的生态系统中，工程师使用深度学习模型转换器将模型从框架移动到运行时环境。然而，转换器中的错误可能会影响模型质量并破坏部署。深度学习模型转换器的故障频率和故障模式尚不清楚。本文针对ONNX (Open Neural Network eXchange)相关的模型转换器进行了首次故障分析。具体而言，我们分析了ONNX转换器在两个重要的DL框架PyTorch和TensorFlow中的过去故障。还报告了故障（N=200个问题）的症状，原因和位置以及随时间的趋势。我们还通过转换8,797个模型（真实世界和人工生成的实例）来评估当今的故障。

    Software engineers develop, fine-tune, and deploy deep learning (DL) models. They use and re-use models in a variety of development frameworks and deploy them on a range of runtime environments. In this diverse ecosystem, engineers use DL model converters to move models from frameworks to runtime environments. However, errors in converters can compromise model quality and disrupt deployment. The failure frequency and failure modes of DL model converters are unknown.  In this paper, we conduct the first failure analysis on DL model converters. Specifically, we characterize failures in model converters associated with ONNX (Open Neural Network eXchange). We analyze past failures in the ONNX converters in two major DL frameworks, PyTorch and TensorFlow. The symptoms, causes, and locations of failures (for N=200 issues), and trends over time are also reported. We also evaluate present-day failures by converting 8,797 models, both real-world and synthetically generated instances. The consis
    
[^235]: 变分自编码器中逐步减少信息瓶颈的去纠缠方法

    Variantional autoencoder with decremental information bottleneck for disentanglement. (arXiv:2303.12959v1 [cs.LG])

    [http://arxiv.org/abs/2303.12959](http://arxiv.org/abs/2303.12959)

    本论文提出了一种逐步减少信息瓶颈的变分自编码器方法，使用去纠缠不变变换来平衡去纠缠和重构保真度，避免信息扩散问题。

    

    变分自编码器中去纠缠学习的一个主要挑战是在权衡去纠缠和重构保真度之间的平衡。之前仅在一个潜在空间中进行的逐步方法无法同时优化这两个目标，因此在训练过程中扩展了信息瓶颈，以从去纠缠到重构进行优化。然而，大型瓶颈会失去去纠缠的约束，导致信息扩散问题。为了解决这个问题，我们提出了一种新颖的逐步减少信息瓶颈的变分自编码器方法，使用去纠缠不变变换来优化不同层的多个目标，称为DeVAE。通过逐渐减小不同潜在空间的信息瓶颈，DeVAE 平衡了去纠缠和重构保真度。由于具有多个潜在空间，DeVAE 允许同时优化多个目标，以在保持去纠缠约束的同时优化重构，避免信息扩散问题。

    One major challenge of disentanglement learning with variational autoencoders is the trade-off between disentanglement and reconstruction fidelity. Previous incremental methods with only on latent space cannot optimize these two targets simultaneously, so they expand the Information Bottleneck while training to {optimize from disentanglement to reconstruction. However, a large bottleneck will lose the constraint of disentanglement, causing the information diffusion problem. To tackle this issue, we present a novel decremental variational autoencoder with disentanglement-invariant transformations to optimize multiple objectives in different layers, termed DeVAE, for balancing disentanglement and reconstruction fidelity by decreasing the information bottleneck of diverse latent spaces gradually. Benefiting from the multiple latent spaces, DeVAE allows simultaneous optimization of multiple objectives to optimize reconstruction while keeping the constraint of disentanglement, avoiding info
    
[^236]: 神经衰弱机器：神经生存回归中超比例危险假设的突破

    Neural Frailty Machine: Beyond proportional hazard assumption in neural survival regressions. (arXiv:2303.10358v1 [cs.LG])

    [http://arxiv.org/abs/2303.10358](http://arxiv.org/abs/2303.10358)

    提出神经衰弱机器（NFM）框架用于生存回归，利用多重衰弱的经典思想来捕捉个体间未观察到的异质性，并能够处理非线性协变量依赖性。两个具体模型下扩展了神经比例危险模型和非参数危险回归模型，结论获得了统计保证。

    

    本文提出神经衰弱机器（NFM），这是一个强大灵活的神经建模框架，用于生存回归。NFM框架利用生存分析中的多重衰弱的经典思想来捕捉个体间未观察到的异质性，并能够利用神经结构的强大逼近能力处理非线性协变量依赖性。框架下推导出两个具体模型，它们分别扩展了神经比例危险模型和非参数危险回归模型。这两个模型都允许在似然目标下进行高效训练。理论上，对于两个提出的模型，我们通过表征其收敛速率，建立了神经函数逼近非参数组件的统计保证。在实证上，我们提供了合成实验来验证我们的理论陈述。我们还在不同规模的6个基准数据集上进行了实验评估，显示出所提出的方法的优越性。

    We present neural frailty machine (NFM), a powerful and flexible neural modeling framework for survival regressions. The NFM framework utilizes the classical idea of multiplicative frailty in survival analysis to capture unobserved heterogeneity among individuals, at the same time being able to leverage the strong approximation power of neural architectures for handling nonlinear covariate dependence. Two concrete models are derived under the framework that extends neural proportional hazard models and nonparametric hazard regression models. Both models allow efficient training under the likelihood objective. Theoretically, for both proposed models, we establish statistical guarantees of neural function approximation with respect to nonparametric components via characterizing their rate of convergence. Empirically, we provide synthetic experiments that verify our theoretical statements. We also conduct experimental evaluations over $6$ benchmark datasets of different scales, showing th
    
[^237]: 从解释中进行鲁棒学习

    Robust Learning from Explanations. (arXiv:2303.06419v1 [cs.LG])

    [http://arxiv.org/abs/2303.06419](http://arxiv.org/abs/2303.06419)

    本文提出了一种新的机器学习方法，将机器学习从解释（MLX）重新构建为对抗鲁棒性问题，通过人类提供的解释来指定一个低维流形，从而减轻了对强参数正则化的需求，并在合成和真实世界基准测试中取得了最新结果。

    This paper proposes a new machine learning approach, recasting machine learning from explanations (MLX) as an adversarial robustness problem, which specifies a lower dimensional manifold from which perturbations can be drawn based on human-provided annotations, and shows improved performance over prior MLX methods on both synthetic and real-world benchmarks.

    机器学习从解释（MLX）是一种学习方法，它使用人类提供的有关每个输入的相关特征的注释，以确保模型预测的原因正确。现有的MLX方法严重依赖于特定的模型解释方法，并需要强大的参数正则化来对齐模型和人类解释，导致次优性能。我们将MLX重新构建为对抗鲁棒性问题，其中人类解释指定了一个低维流形，可以从中绘制扰动，并理论上和实验上展示了这种方法如何减轻对强参数正则化的需求。我们考虑了实现鲁棒性的各种方法，从而提高了先前MLX方法的性能。最后，我们将鲁棒性与早期的MLX方法相结合，产生了在合成和真实世界基准测试中的最新结果。

    Machine learning from explanations (MLX) is an approach to learning that uses human-provided annotations of relevant features for each input to ensure that model predictions are right for the right reasons. Existing MLX approaches rely heavily on a specific model interpretation approach and require strong parameter regularization to align model and human explanations, leading to sub-optimal performance. We recast MLX as an adversarial robustness problem, where human explanations specify a lower dimensional manifold from which perturbations can be drawn, and show both theoretically and empirically how this approach alleviates the need for strong parameter regularization. We consider various approaches to achieving robustness, leading to improved performance over prior MLX methods. Finally, we combine robustness with an earlier MLX method, yielding state-of-the-art results on both synthetic and real-world benchmarks.
    
[^238]: 技术报告：图神经网络也可以变得语法化

    Technical report: Graph Neural Networks go Grammatical. (arXiv:2303.01590v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.01590](http://arxiv.org/abs/2303.01590)

    本文介绍了一种将代数语言片段与图神经网络形式上联系的框架，并从MATLANG定义了一个符合3-WL测试的语法，进而得出一个符合3-WL GNN模型的G$^2$N$^2$。此外，语法方法还提供了计算长度为六及以下的环和弦环的代数公式，并在多个下游任务中取得优秀的表现。

    

    本文提出了一个框架，将一个代数语言的一个片段与图神经网络（GNN）形式上联系起来。它依赖于上下文无关语法（CFG），将代数操作组织成可以翻译为GNN层模型的生成规则。由于直接从语言派生出的CFG的规则和变量包含冗余，因此介绍了一种语法简化方案，使得将其翻译为GNN层成为可能。应用这种策略，从MATLANG定义了一个符合第三阶Weisfeiler-Lehman（3-WL）测试要求的语法。从这个3-WL CFG中，我们得出了一个经过证明符合3-WL GNN模型的G$^2$N$^2$。此外，这种语法方法使我们能够提供计算长度为六及以下的环和弦环的代数公式，从而阐明了3-WL的计数能力。多个实验证明，G$^2$N$^2$在许多下游任务中的表现要比其他3-WL GNN更为高效。

    This paper proposes a framework to formally link a fragment of an algebraic language to a Graph Neural Network (GNN). It relies on Context Free Grammars (CFG) to organise algebraic operations into generative rules that can be translated into a GNN layer model. Since the rules and variables of a CFG directly derived from a language contain redundancies, a grammar reduction scheme is presented making tractable the translation into a GNN layer. Applying this strategy, a grammar compliant with the third-order Weisfeiler-Lehman (3-WL) test is defined from MATLANG. From this 3-WL CFG, we derive a provably 3-WL GNN model called G$^2$N$^2$. Moreover, this grammatical approach allows us to provide algebraic formulas to count the cycles of length up to six and chordal cycles at the edge level, which enlightens the counting power of 3-WL. Several experiments illustrate that G$^2$N$^2$ efficiently outperforms other 3-WL GNNs on many downstream tasks.
    
[^239]: 学习神经网络的计算复杂度: 光滑性和退化性

    Computational Complexity of Learning Neural Networks: Smoothness and Degeneracy. (arXiv:2302.07426v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.07426](http://arxiv.org/abs/2302.07426)

    本文研究了学习神经网络的计算复杂度，特别关注了输入分布和权重矩阵的假设对学习算法有效性的影响。结果表明，在高斯输入分布下，学习深度为3的ReLU网络是困难的，即使权重矩阵是非退化的。同时，学习深度为2的网络也面临困难。

    

    理解神经网络何时可以被有效学习是学习理论中的一个基本问题。已有的难度结果表明，对输入分布和网络权重都需要做出一定的假设才能得到有效的算法。此前的研究已经表明，假设输入分布为高斯分布且权重矩阵非退化时，可以有效地学习深度为2的网络。本文研究了这些假设是否适用于学习更深的网络，并给出了否定的结论。我们证明，在光滑分析框架下，即在网络参数中加入随机噪声的情况下，学习深度为3的ReLU网络在高斯输入分布下是困难的。这意味着，即使权重矩阵是非退化的，学习深度为3的ReLU网络在高斯分布下也是困难的。此外，我们还考虑了深度为2的网络，并展示了在光滑分析框架下学习的困难性。

    Understanding when neural networks can be learned efficiently is a fundamental question in learning theory. Existing hardness results suggest that assumptions on both the input distribution and the network's weights are necessary for obtaining efficient algorithms. Moreover, it was previously shown that depth-$2$ networks can be efficiently learned under the assumptions that the input distribution is Gaussian, and the weight matrix is non-degenerate. In this work, we study whether such assumptions may suffice for learning deeper networks and prove negative results. We show that learning depth-$3$ ReLU networks under the Gaussian input distribution is hard even in the smoothed-analysis framework, where a random noise is added to the network's parameters. It implies that learning depth-$3$ ReLU networks under the Gaussian distribution is hard even if the weight matrices are non-degenerate. Moreover, we consider depth-$2$ networks, and show hardness of learning in the smoothed-analysis fr
    
[^240]: 用于评估图像模型的具有XAI基准的数据集生成新方法

    A novel approach to generate datasets with XAI ground truth to evaluate image models. (arXiv:2302.05624v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.05624](http://arxiv.org/abs/2302.05624)

    该论文介绍了一种新方法来生成具有XAI基准的数据集，用于评估图像模型。通过与真实模型解释进行比较，实验证实了该方法的可靠性。

    

    随着人工智能（AI）的广泛使用，了解这些模型的内部工作机制变得至关重要。这些需求推动了一种名为可解释人工智能（XAI）的新领域的发展。该领域涵盖了一系列技术，使我们能够理论上确定AI决策的原因。XAI的一个主要问题是如何验证该领域的工作，考虑到缺乏基准（GT）。在本研究中，我们提出了一种新的方法来生成带有GT的数据集。我们进行了一系列实验，将我们的GT与真实模型解释进行比较，并获得了卓越的结果，证实我们提出的方法是正确的。

    With the increased usage of artificial intelligence (AI), it is imperative to understand how these models work internally. These needs have led to the development of a new field called eXplainable artificial intelligence (XAI). This field consists of on a set of techniques that allows us to theoretically determine the cause of the AI decisions. One main issue of XAI is how to verify the works on this field, taking into consideration the lack of ground truth (GT). In this study, we propose a new method to generate datasets with GT. We conducted a set of experiments that compared our GT with real model explanations and obtained excellent results confirming that our proposed method is correct.
    
[^241]: 基于频率变换的深度学习时间序列分析综述

    A Survey on Deep Learning based Time Series Analysis with Frequency Transformation. (arXiv:2302.02173v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02173](http://arxiv.org/abs/2302.02173)

    近期，频率变换（FT）在深度学习时间序列分析中得到广泛应用，显著提高了准确性和效率。本文系统回顾和总结了基于FT的深度学习时间序列模型的研究进展，并探讨了其优势、限制以及主要方法。

    

    最近，频率变换（FT）越来越多地被纳入深度学习模型中，可以显著提高时间序列分析的最新准确性和效率。频率变换的优势，如高效性和全局视角，在各种时间序列任务和应用中被迅速探索和利用，展示了频率变换作为一种新的深度学习范式在时间序列分析领域的潜力。尽管这个新兴领域受到了越来越多的关注和研究，但目前还缺乏对基于频率变换的深度学习时间序列模型的系统回顾和深入分析。目前还不清楚为什么频率变换可以提升时间序列分析的效果，以及它在该领域的限制是什么。为了填补这些空白，我们提供了一份全面的综述，系统调查和总结了基于频率变换的深度学习时间序列分析的最新研究进展。具体而言，我们探讨了主要的方法。

    Recently, frequency transformation (FT) has been increasingly incorporated into deep learning models to significantly enhance state-of-the-art accuracy and efficiency in time series analysis. The advantages of FT, such as high efficiency and a global view, have been rapidly explored and exploited in various time series tasks and applications, demonstrating the promising potential of FT as a new deep learning paradigm for time series analysis. Despite the growing attention and the proliferation of research in this emerging field, there is currently a lack of a systematic review and in-depth analysis of deep learning-based time series models with FT. It is also unclear why FT can enhance time series analysis and what its limitations in the field are. To address these gaps, we present a comprehensive review that systematically investigates and summarizes the recent research advancements in deep learning-based time series analysis with FT. Specifically, we explore the primary approaches us
    
[^242]: 双排列等变性在知识图谱补全中的应用

    Double Permutation Equivariance for Knowledge Graph Completion. (arXiv:2302.01313v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.01313](http://arxiv.org/abs/2302.01313)

    本研究提出了双排列等变性的KG表示方法，可以使神经网络在KG中执行复杂的逻辑推理任务，并在多个归纳KG完成任务中实现了最先进的Hits@10测试准确率。双排列等变性在KG中开辟了新的研究方向。

    

    本研究将知识图谱(KGs)形式化为一种新型的图，并称之为双交换属性图，其中节点和二元（两个节点之间的）表示必须对节点号和边（及节点）属性（关系和节点特征）的排列等变。双重排列等变的KG表示在KG中开辟了新的研究方向。我们展示了这种等变性对关系的结构表示产生的影响，从而使神经网络能够在KG中执行复杂的逻辑推理任务。最后，我们介绍了一种通用的等变表示蓝图，并测试了一种简单的基于GNN的双排列等变神经结构，在WN18RR、FB237和NELL995归纳KG完成任务中实现了最先进的Hits@10测试准确率，并能够准确执行现有方法无法执行的逻辑推理任务。

    This work provides a formalization of Knowledge Graphs (KGs) as a new class of graphs that we denote doubly exchangeable attributed graphs, where node and pairwise (joint 2-node) representations must be equivariant to permutations of both node ids and edge (& node) attributes (relations & node features). Double-permutation equivariant KG representations open a new research direction in KGs. We show that this equivariance imposes a structural representation of relations that allows neural networks to perform complex logical reasoning tasks in KGs. Finally, we introduce a general blueprint for such equivariant representations and test a simple GNN-based double-permutation equivariant neural architecture that achieve state-of-the-art Hits@10 test accuracy in the WN18RR, FB237 and NELL995 inductive KG completion tasks, and can accurately perform logical reasoning tasks that no existing methods can perform, to the best of our knowledge.
    
[^243]: 正则化流集合用于丰富的Aleatoric和Epistemic不确定性建模

    Normalizing Flow Ensembles for Rich Aleatoric and Epistemic Uncertainty Modeling. (arXiv:2302.01312v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.01312](http://arxiv.org/abs/2302.01312)

    本文提出了一个正则化流（NF）集合来估计Epistemic不确定性和Aleatoric不确定性，通过固定的dropout掩码来创建集合，运用于各种实验并可以提供全面的基准线。

    

    本文展示了如何可靠地估计Epistemic不确定性，同时保持捕捉复杂Aleatoric分布所需的灵活性。为此，我们提出了一个正则化流（NF）集合，这是建模Aleatoric不确定性的最先进技术。集合是通过固定的dropout掩码集合创建的，比创建单独的NF模型更加经济。我们演示了如何利用NF的独特结构——基础分布——来估计Aleatoric不确定性，而无需依赖样本，提供了全面的基准线，并推导出无偏的微分熵估计值。该方法被应用于各种用于基准测试Aleatoric和Epistemic不确定性估计的实验中：1D正弦数据，2D有风格网格世界（$\it{Wet Chicken}$），$\it{Pendulum}$和$\it{Hopper}$。在这些实验中，我们建立了一个主动学习框架，并评估了每个模型在测量Aleatoric和Epistemic不确定性方面的能力。

    In this work, we demonstrate how to reliably estimate epistemic uncertainty while maintaining the flexibility needed to capture complicated aleatoric distributions. To this end, we propose an ensemble of Normalizing Flows (NF), which are state-of-the-art in modeling aleatoric uncertainty. The ensembles are created via sets of fixed dropout masks, making them less expensive than creating separate NF models. We demonstrate how to leverage the unique structure of NFs, base distributions, to estimate aleatoric uncertainty without relying on samples, provide a comprehensive set of baselines, and derive unbiased estimates for differential entropy. The methods were applied to a variety of experiments, commonly used to benchmark aleatoric and epistemic uncertainty estimation: 1D sinusoidal data, 2D windy grid-world ($\it{Wet Chicken}$), $\it{Pendulum}$, and $\it{Hopper}$. In these experiments, we setup an active learning framework and evaluate each model's capability at measuring aleatoric and
    
[^244]: 高效Transformer的交替更新方法

    Alternating Updates for Efficient Transformers. (arXiv:2301.13310v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.13310](http://arxiv.org/abs/2301.13310)

    本文介绍了一种交替更新（AltUp）的方法，可以在不增加计算负担的情况下增加模型的容量，通过对扩展表示的子块进行操作并使用预测和修正机制来更新未激活的块。实验证明AltUp方法在提高Transformer模型的容量和效率方面是有效的。

    

    众所周知，增加深度Transformer网络的规模可以提高模型的质量和性能。然而，这种规模的增加往往会导致计算成本和推理延迟的大幅增加。我们引入了交替更新（AltUp）方法，这是一种简单易实现的方法，可以增加模型的容量而不增加计算负担。AltUp通过在每一层中对扩展表示的子块进行操作，并使用预测和修正机制来更新未激活的块，从而实现了仅在延迟上微不足道的情况下扩大了学习表示，即标记嵌入。我们还介绍了AltUp的扩展，例如其在序列维度上的适用性，并展示了如何将AltUp与现有方法（如稀疏专家混合模型）结合起来，以获得具有更高容量的高效模型。我们在基准Transformer模型和语言任务上进行了实验证明了AltUp方法的有效性。

    It has been well established that increasing scale in deep transformer networks leads to improved quality and performance. However, this increase in scale often comes with prohibitive increases in compute cost and inference latency. We introduce Alternating Updates (AltUp), a simple-to-implement method to increase a model's capacity without the computational burden. AltUp enables the widening of the learned representation, i.e., the token embedding, while only incurring a negligible increase in latency. AltUp achieves this by working on a subblock of the widened representation at each layer and using a predict-and-correct mechanism to update the inactivated blocks. We present extensions of AltUp, such as its applicability to the sequence dimension, and demonstrate how AltUp can be synergistically combined with existing approaches, such as Sparse Mixture-of-Experts models, to obtain efficient models with even higher capacity. Our experiments on benchmark transformer models and language 
    
[^245]: 用反事实推理解释$\mathcal{ELH}$概念描述

    Explaining $\mathcal{ELH}$ Concept Descriptions through Counterfactual Reasoning. (arXiv:2301.05109v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2301.05109](http://arxiv.org/abs/2301.05109)

    本研究提出了一种通过反事实推理来解释概念描述的方法，以提供简洁且易于理解的解释，便于非专家理解和采取行动。

    

    知识库被广泛应用于信息管理，能够支持高影响力的应用程序如网络搜索、问答和自然语言处理。它们也作为自动决策系统的基础，例如医疗诊断和信用评分。由于受到这些决策影响的利益相关者希望了解他们的情况并验证决策的公平性，因此提出了许多解释方法。描述逻辑中使用概念来进行分类是一种固有透明的方式。然而，即使在口头化的情况下，这些概念也会变得冗长且难以理解，特别是对于非专家而言。一种解决方法是使用反事实来回答问题：“为了得到不同的分类，特征值应如何改变？”通过关注最小的特征变化，解释变得短小、易于理解，并提供了关于变化对预测的影响的明确行动路径。

    Knowledge bases are widely used for information management, enabling high-impact applications such as web search, question answering, and natural language processing. They also serve as the backbone for automatic decision systems, e.g., for medical diagnostics and credit scoring. As stakeholders affected by these decisions would like to understand their situation and verify how fair the decisions are, a number of explanation approaches have been proposed. An intrinsically transparent way to do classification is by using concepts in description logics. However, these concepts can become long and difficult to fathom for non-experts, even when verbalized. One solution is to employ counterfactuals to answer the question, ``How must feature values be changed to obtain a different classification?'' By focusing on the minimal feature changes, the explanations are short, human-friendly, and provide a clear path of action regarding the change in prediction. While previous work investigated coun
    
[^246]: 自动文本摘要技术的综合回顾：方法、数据、评估和编码

    A comprehensive review of automatic text summarization techniques: method, data, evaluation and coding. (arXiv:2301.03403v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.03403](http://arxiv.org/abs/2301.03403)

    本文提供了关于自动文本摘要系统的综述，包括方法、数据、评估和编码。作者通过引用的方式回顾了相关文献，并介绍了不同的摘要生成方法。此外，还对可用于评估和数据训练的数据集进行了综述，并使用CNN语料库数据集对方法进行了实证探索。

    

    我们提供了关于自动文本摘要系统的文献综述。我们采用基于引用的方法。我们从我们已经掌握的关于每个我们想涵盖的主题的一些流行和著名的论文开始，然后我们追踪了“向后引用”（被我们之前知道的一系列论文引用的论文）和“向前引用”（引用我们之前知道的一系列论文的较新论文）。为了组织不同的方法，我们介绍了各种基于不同机制生成摘要的自动文本摘要方法。除了介绍方法外，我们还对可用于摘要任务的数据集和用于评估摘要质量的方法进行了广泛的回顾。最后，我们还使用CNN语料库数据集对这些方法进行了实证探索，该数据集为抽取式和生成式方法提供了金标准摘要。

    We provide a literature review about Automatic Text Summarization (ATS) systems. We consider a citation-based approach. We start with some popular and well-known papers that we have in hand about each topic we want to cover and we have tracked the "backward citations" (papers that are cited by the set of papers we knew beforehand) and the "forward citations" (newer papers that cite the set of papers we knew beforehand). In order to organize the different methods, we present the diverse approaches to ATS guided by the mechanisms they use to generate a summary. Besides presenting the methods, we also present an extensive review of the datasets available for summarization tasks and the methods used to evaluate the quality of the summaries. Finally, we present an empirical exploration of these methods using the CNN Corpus dataset that provides golden summaries for extractive and abstractive methods.
    
[^247]: 用引导想象扩充小规模数据集

    Expanding Small-Scale Datasets with Guided Imagination. (arXiv:2211.13976v4 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.13976](http://arxiv.org/abs/2211.13976)

    本论文提出了一个引导想象框架(GIF)，通过利用DALL-E2和Stable Diffusion (SD)等生成模型，从种子数据中扩充小规模数据集。该框架通过在先验模型的语义空间中优化种子数据潜在特征来创建逼真的图像，并引入了类别保持和样本多样性的标准来指导想象过程。

    

    DNN的功效在很大程度上取决于训练数据的数量和质量。然而，大规模收集和标注数据通常费时费力。为了解决这个问题，我们探索了一项名为数据集扩充的新任务，旨在通过自动创建新的标记样本来扩充一个小规模的可用数据集。为此，我们提出了一个引导想象框架(GIF)，利用DALL-E2和Stable Diffusion (SD)等尖端生成模型的力量，从输入的种子数据中“想象”并创建信息丰富的新数据。具体而言，GIF通过在先验模型的语义有意义的空间中优化种子数据的潜在特征来进行数据的想象，从而创建具有新内容的照片般逼真的图像。为了引导想象朝着创建用于模型训练的信息丰富样本的方向发展，我们引入了两个关键标准，即类别保持信息提升和样本多样性促进。这些标准被证明有效地提高了生成图像的质量和多样性。

    The power of DNNs relies heavily on the quantity and quality of training data. However, collecting and annotating data on a large scale is often expensive and time-consuming. To address this issue, we explore a new task, termed dataset expansion, aimed at expanding a ready-to-use small dataset by automatically creating new labeled samples. To this end, we present a Guided Imagination Framework (GIF) that leverages cutting-edge generative models like DALL-E2 and Stable Diffusion (SD) to "imagine" and create informative new data from the input seed data. Specifically, GIF conducts data imagination by optimizing the latent features of the seed data in the semantically meaningful space of the prior model, resulting in the creation of photo-realistic images with new content. To guide the imagination towards creating informative samples for model training, we introduce two key criteria, i.e., class-maintained information boosting and sample diversity promotion. These criteria are verified to
    
[^248]: 使用DP-SGD的私有广告建模

    Private Ad Modeling with DP-SGD. (arXiv:2211.11896v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.11896](http://arxiv.org/abs/2211.11896)

    本研究将差分隐私随机梯度下降（DP-SGD）应用于广告建模任务，证明了该方法可以在处理高类别不平衡和稀疏梯度更新的广告数据中提供隐私和效用。

    

    在隐私保护机器学习中，一种众所周知的算法是差分隐私随机梯度下降（DP-SGD）。尽管该算法在文本和图像数据上已经进行了评估，但在以往的研究中尚未将其应用于广告数据，而广告数据因其高类别不平衡和稀疏的梯度更新而臭名昭著。本研究我们将DP-SGD应用于多个广告建模任务，包括预测点击率、转化率和转化事件数量，并在真实数据集上评估其隐私和效用的权衡。我们的工作首次实证了DP-SGD可以为广告建模任务提供隐私和效用。

    A well-known algorithm in privacy-preserving ML is differentially private stochastic gradient descent (DP-SGD). While this algorithm has been evaluated on text and image data, it has not been previously applied to ads data, which are notorious for their high class imbalance and sparse gradient updates. In this work we apply DP-SGD to several ad modeling tasks including predicting click-through rates, conversion rates, and number of conversion events, and evaluate their privacy-utility trade-off on real-world datasets. Our work is the first to empirically demonstrate that DP-SGD can provide both privacy and utility for ad modeling tasks.
    
[^249]: PersA-FL：个性化异步联邦学习

    PersA-FL: Personalized Asynchronous Federated Learning. (arXiv:2210.01176v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.01176](http://arxiv.org/abs/2210.01176)

    本论文研究了异步更新下的个性化联邦学习问题，并提出了一种改进的个性化方法，通过移除同步通信假设和去除梯度范数有界性假设来提高可伸缩性。

    

    我们研究了异步更新下的个性化联邦学习问题。在这个问题中，每个客户端都希望获得一个个性化模型，同时能够优于本地模型和全局模型。我们考虑了两个基于优化的个性化框架：（i）模型无关元学习（MAML）和（ii）Moreau包络（ME）。MAML通过微调学习适应于每个客户端的联合模型，而ME通过隐式梯度的双层优化问题来通过规范化损失实现个性化。我们的主要技术贡献是对有界滞后的异步联邦学习进行统一证明，并将其应用于MAML和ME个性化框架。针对平滑和非凸函数类，我们进一步扩展了所研究的函数类，去除了梯度范数的有界性假设。

    We study the personalized federated learning problem under asynchronous updates. In this problem, each client seeks to obtain a personalized model that simultaneously outperforms local and global models. We consider two optimization-based frameworks for personalization: (i) Model-Agnostic Meta-Learning (MAML) and (ii) Moreau Envelope (ME). MAML involves learning a joint model adapted for each client through fine-tuning, whereas ME requires a bi-level optimization problem with implicit gradients to enforce personalization via regularized losses. We focus on improving the scalability of personalized federated learning by removing the synchronous communication assumption. Moreover, we extend the studied function class by removing boundedness assumptions on the gradient norm. Our main technical contribution is a unified proof for asynchronous federated learning with bounded staleness that we apply to MAML and ME personalization frameworks. For the smooth and non-convex functions class, we 
    
[^250]: 论AdaGrad在$\R^{d}$上的收敛性：超越凸性、非渐近速率和加速（arXiv：2209.14827v2 [cs.LG] UPDATED）

    On the Convergence of AdaGrad on $\R^{d}$: Beyond Convexity, Non-Asymptotic Rate and Acceleration. (arXiv:2209.14827v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.14827](http://arxiv.org/abs/2209.14827)

    本论文主要展示了AdaGrad在平滑凸函数和更一般的quasar凸函数的情况下的收敛性。具体地，我们提出了新的技术，明确限定了vanilla AdaGrad在无约束问题中的收敛速率，并提出了一种AdaGrad变种，可以实现更快的收敛。

    

    现有的关于平滑凸优化的AdaGrad和其他自适应方法的分析通常是针对具有有界定义域直径的函数。在无约束问题中，以前的研究保证了渐近收敛速率，但没有明确的恒定因子，这适用于整个函数类。此外，在随机环境中，只分析了一个修改版本的AdaGrad，与通常实践中使用的版本不同，在这个回归中不使用最新的梯度来更新步幅。我们的论文旨在弥合这些差距，并在平滑凸函数的标准情况下以及更一般的quasar凸函数的情况下深入理解AdaGrad及其变种。首先，我们展示了新技术，明确地限定了vanilla AdaGrad在无约束问题中的收敛速率，无论是确定性的还是随机的情况下。其次，我们提出了一种AdaGrad变种，我们可以展示l的收敛

    Existing analysis of AdaGrad and other adaptive methods for smooth convex optimization is typically for functions with bounded domain diameter. In unconstrained problems, previous works guarantee an asymptotic convergence rate without an explicit constant factor that holds true for the entire function class. Furthermore, in the stochastic setting, only a modified version of AdaGrad, different from the one commonly used in practice, in which the latest gradient is not used to update the stepsize, has been analyzed. Our paper aims at bridging these gaps and developing a deeper understanding of AdaGrad and its variants in the standard setting of smooth convex functions as well as the more general setting of quasar convex functions. First, we demonstrate new techniques to explicitly bound the convergence rate of the vanilla AdaGrad for unconstrained problems in both deterministic and stochastic settings. Second, we propose a variant of AdaGrad for which we can show the convergence of the l
    
[^251]: 神经过程家族：调查、应用和展望

    The Neural Process Family: Survey, Applications and Perspectives. (arXiv:2209.00517v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.00517](http://arxiv.org/abs/2209.00517)

    神经过程家族旨在结合神经网络和高斯过程的优点，实现元学习预测不确定性的能力，并在深度学习领域带来重要进展。

    

    标准的神经网络实现方法具有强大的函数逼近能力，但在学习元表示和推理其预测中的概率不确定性方面有限。另一方面，高斯过程采用贝叶斯学习方案来估计这种不确定性，但受到其效率和逼近能力的限制。神经过程家族（NPF）意在通过利用神经网络进行元学习预测不确定性，提供两者的优点。近年来，这种潜力已经吸引了大量的研究活动。因此，需要进行一项全面的NPF模型调查，以组织和关联它们的动机、方法和实验。本文旨在填补这一空白，深入探讨关于家族成员的形式化、研究主题和应用。我们着重阐述它们在带来其他深度学习领域的一些最新进展方面的潜力。

    The standard approaches to neural network implementation yield powerful function approximation capabilities but are limited in their abilities to learn meta representations and reason probabilistic uncertainties in their predictions. Gaussian processes, on the other hand, adopt the Bayesian learning scheme to estimate such uncertainties but are constrained by their efficiency and approximation capacity. The Neural Processes Family (NPF) intends to offer the best of both worlds by leveraging neural networks for meta-learning predictive uncertainties. Such potential has brought substantial research activity to the family in recent years. Therefore, a comprehensive survey of NPF models is needed to organize and relate their motivation, methodology, and experiments. This paper intends to address this gap while digging deeper into the formulation, research themes, and applications concerning the family members. We shed light on their potential to bring several recent advances in other deep 
    
[^252]: 潜在神经随机微分方程用于变点检测

    Latent Neural Stochastic Differential Equations for Change Point Detection. (arXiv:2208.10317v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.10317](http://arxiv.org/abs/2208.10317)

    本文提出了一种基于潜在神经随机微分方程的变点检测算法，通过学习将过程转换到潜在空间，并使用学习随机过程的似然比来定位过程中的变点。在合成和真实数据集上的实验证明了该方法的出色性能。

    

    基于多个读数的复杂系统的自动分析仍然是一个挑战。变点检测算法用于定位过程的时间序列行为中的突变点。在本文中，我们提出了一种基于潜在神经随机微分方程（SDE）的新型变点检测算法。我们的方法学习将过程非线性地转换为潜在空间，并估计描述其随时间演化的SDE。该算法使用不同时间戳的学习随机过程的似然比来找到过程的变点。我们在合成和真实数据集上展示了我们算法的检测能力和性能。在我们的大多数实验中，所提出的方法优于现有算法。

    Automated analysis of complex systems based on multiple readouts remains a challenge. Change point detection algorithms are aimed to locating abrupt changes in the time series behaviour of a process. In this paper, we present a novel change point detection algorithm based on Latent Neural Stochastic Differential Equations (SDE). Our method learns a non-linear deep learning transformation of the process into a latent space and estimates a SDE that describes its evolution over time. The algorithm uses the likelihood ratio of the learned stochastic processes in different timestamps to find change points of the process. We demonstrate the detection capabilities and performance of our algorithm on synthetic and real-world datasets. The proposed method outperforms the state-of-the-art algorithms on the majority of our experiments.
    
[^253]: 通过测试驱动的用户意图规范化进行交互式代码生成

    Interactive Code Generation via Test-Driven User-Intent Formalization. (arXiv:2208.05950v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2208.05950](http://arxiv.org/abs/2208.05950)

    本文提出了交互式测试驱动代码生成的工作流程，该方法通过生成的测试形式化用户意图，并通过修剪和排名代码建议来提供改进的代码建议集。

    

    大型语言模型（LLMs）展示了通过从非正式的自然语言（NL）意图生成自然代码来自动化编码的巨大潜力。然而，与LLMs进行交互时，用户不能保证生成的代码建议正确地满足其提供的意图。事实上，很难定义正确性的概念，因为自然语言可能存在歧义，并且缺乏形式化语义。在本文中，我们提出了“交互式测试驱动代码生成”的工作流程，它利用轻量级用户反馈来（a）使用生成的测试来形式化用户意图，这对于调试非常有用，以及（b）通过修剪和排名候选代码建议来生成改进的代码建议集。我们描述了一种与语言无关的抽象算法和一个具体的实现TiCoder。我们在\emph {MBPP}和\emph {HumanEval}代码生成基准上对TiCoder进行了自动评估。我们的结果令人鼓舞。

    Large language models (LLMs) have shown great potential in automating significant aspects of coding by producing natural code from informal natural language (NL) intent. However, when interacting with LLMs, users have no guarantees that the code suggestions produced correctly satisfy the intent they provided. In fact, it is hard to define a notion of correctness since natural language can be ambiguous and lacks a formal semantics.  In this paper, we propose the workflow of {\it interactive test-driven code generation}, which leverages lightweight user feedback to (a) formalize the user intent using generated tests that can be useful for debugging, and (b) produce an improved set of code suggestions by pruning and ranking candidate code suggestions. We describe a language-agnostic abstract algorithm and a concrete implementation TiCoder. We perform an automated evaluation of TiCoder on the \emph{MBPP} and \emph{HumanEval} code generation benchmarks. Our results are promising with using 
    
[^254]: 利用非语言线索分析共同人际互动：一项调查

    Co-Located Human-Human Interaction Analysis using Nonverbal Cues: A Survey. (arXiv:2207.10574v2 [cs.HC] UPDATED)

    [http://arxiv.org/abs/2207.10574](http://arxiv.org/abs/2207.10574)

    使用非语言线索进行共同人际互动分析的研究调查了从2010年以来的计算研究，并总结了最常用的非语言线索，以及有关互动分析的未来研究方向。

    

    使用非语言沟通作为社会和心理现象测量的证据，自动化的共同人际互动分析得到了解决。我们对从2010年以来的计算研究进行了调查，检测与社交特征（如领导力、支配力、个性特征）、社交角色/关系和互动动态（如团队凝聚力、参与度、融洽度）相关的现象。我们的目标是确定导致有效性能的非语言线索和计算方法。这项调查在涉及最广泛的社会现象和互动场景（自由对话、会议、二元组和人群）方面与其同类不同。我们还提供了相关数据集的综合总结，并概述了关于人工智能实施、数据集策划和保护隐私的互动分析的未来研究方向。一些重要观察结果是：最常用的非语言线索是共同 ...

    Automated co-located human-human interaction analysis has been addressed by the use of nonverbal communication as measurable evidence of social and psychological phenomena. We survey the computing studies (since 2010) detecting phenomena related to social traits (e.g., leadership, dominance, personality traits), social roles/relations, and interaction dynamics (e.g., group cohesion, engagement, rapport). Our target is to identify the nonverbal cues and computational methodologies resulting in effective performance. This survey differs from its counterparts by involving the widest spectrum of social phenomena and interaction settings (free-standing conversations, meetings, dyads, and crowds). We also present a comprehensive summary of the related datasets and outline future research directions which are regarding the implementation of artificial intelligence, dataset curation, and privacy-preserving interaction analysis. Some major observations are: the most often used nonverbal cue, co
    
[^255]: 通过标准化流进行逐渐领域适应

    Gradual Domain Adaptation via Normalizing Flows. (arXiv:2206.11492v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2206.11492](http://arxiv.org/abs/2206.11492)

    该论文提出使用标准化流来解决逐渐领域适应中中间域有限且距离较大的问题，并通过从源域到高斯混合分布学习目标域的分布变换。

    

    当源域和目标域之间存在较大差距时，传统的领域适应方法效果不佳。逐渐领域适应是解决该问题的一种方法，它涉及利用逐渐从源域转移到目标域的中间域。在先前的工作中，假设中间域的数量较大且相邻域之间的距离较小，因此，涉及使用无标签数据集进行自我训练的逐渐领域适应算法是可行的。然而，在实践中，逐渐自我训练将失败，因为中间域的数量有限且相邻域之间的距离较大。我们提出使用标准化流来解决这个问题，同时保持无监督领域适应的框架。所提出的方法通过从源域到高斯混合分布学习目标域的分布变换。

    Standard domain adaptation methods do not work well when a large gap exists between the source and target domains. Gradual domain adaptation is one of the approaches used to address the problem. It involves leveraging the intermediate domain, which gradually shifts from the source domain to the target domain. In previous work, it is assumed that the number of intermediate domains is large and the distance between adjacent domains is small; hence, the gradual domain adaptation algorithm, involving self-training with unlabeled datasets, is applicable. In practice, however, gradual self-training will fail because the number of intermediate domains is limited and the distance between adjacent domains is large. We propose the use of normalizing flows to deal with this problem while maintaining the framework of unsupervised domain adaptation. The proposed method learns a transformation from the distribution of the target domain to the Gaussian mixture distribution via the source domain. We e
    
[^256]: 自动剪辑：更简单和更强大的差分隐私深度学习

    Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger. (arXiv:2206.07136v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.07136](http://arxiv.org/abs/2206.07136)

    本论文提出了一种自动剪辑的替代方案，它消除了为差分隐私优化器调整剪辑阈值的需要，并在非凸情况下进行了收敛性分析。在多个任务中证明了自动剪辑的优势。

    

    单样本梯度剪辑是实现深度学习模型的差分隐私(DP)训练的关键算法步骤。然而，剪辑阈值R的选择对于在DP下实现高准确性至关重要。我们提出了一种易于使用的替代方案，称为自动剪辑，它消除了为任何DP优化器（包括DP-SGD、DP-Adam、DP-LAMB等）调整R的需要。自动变量与现有的DP优化器一样具有隐私性和计算效率，但不需要DP特定的超参数，因此使得DP训练像标准的非隐私训练一样可行。我们在非凸情况下对自动DP-SGD进行了严格的收敛性分析，表明在样本梯度的对称性噪声假设下（在非DP文献中常用），它能够享受与标准SGD相同的渐近收敛速率。我们在各种语言和视觉任务中展示了自动剪辑的优势。

    Per-example gradient clipping is a key algorithmic step that enables practical differential private (DP) training for deep learning models. The choice of clipping threshold R, however, is vital for achieving high accuracy under DP. We propose an easy-to-use replacement, called automatic clipping, that eliminates the need to tune R for any DP optimizers, including DP-SGD, DP-Adam, DP-LAMB and many others. The automatic variants are as private and computationally efficient as existing DP optimizers, but require no DP-specific hyperparameters and thus make DP training as amenable as the standard non-private training. We give a rigorous convergence analysis of automatic DP-SGD in the non-convex setting, showing that it can enjoy an asymptotic convergence rate that matches the standard SGD, under a symmetric gradient noise assumption of the per-sample gradients (commonly used in the non-DP literature). We demonstrate on various language and vision tasks that automatic clipping outperforms o
    
[^257]: 在无偏推荐中重新考虑学习目标：分布转移视角下的研究

    Reconsidering Learning Objectives in Unbiased Recommendation: A Distribution Shift Perspective. (arXiv:2206.03851v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2206.03851](http://arxiv.org/abs/2206.03851)

    本文从分布转移视角出发，研究了从偏向反馈中学习无偏算法进行推荐的问题。通过建立无偏推荐与分布转移的关系，对现有无偏学习方法进行了理论解释并提出了两个泛化界限。

    

    本文研究了从偏向反馈中学习无偏算法进行推荐的问题，我们从一个新颖的分布转移视角来解决这个问题。最近在无偏推荐领域的研究中，通过各种技术如重新加权、多任务学习和元学习，取得了最新的成果。尽管它们在实证上取得了成功，但大部分缺乏理论保证，导致了理论和最新算法之间的显著差距。本文提出了对现有无偏学习目标为何适用于无偏推荐的理论理解。我们建立了无偏推荐与分布转移之间的密切关系，显示了现有的无偏学习目标隐含地将有偏的训练分布与无偏的测试分布对齐。基于这个关系，我们针对现有的无偏学习方法发展了两个泛化界限并分析了它们的学习行为。

    This work studies the problem of learning unbiased algorithms from biased feedback for recommendation. We address this problem from a novel distribution shift perspective. Recent works in unbiased recommendation have advanced the state-of-the-art with various techniques such as re-weighting, multi-task learning, and meta-learning. Despite their empirical successes, most of them lack theoretical guarantees, forming non-negligible gaps between theories and recent algorithms. In this paper, we propose a theoretical understanding of why existing unbiased learning objectives work for unbiased recommendation. We establish a close connection between unbiased recommendation and distribution shift, which shows that existing unbiased learning objectives implicitly align biased training and unbiased test distributions. Built upon this connection, we develop two generalization bounds for existing unbiased learning methods and analyze their learning behavior. Besides, as a result of the distributio
    
[^258]: 关于随机拟牛顿方法在深度学习中的效率问题

    On the efficiency of Stochastic Quasi-Newton Methods for Deep Learning. (arXiv:2205.09121v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2205.09121](http://arxiv.org/abs/2205.09121)

    本文研究了随机拟牛顿算法在深度学习中的效率，分析了有限内存BFGS和对称秩一SR1两种更新算法的性能表现，并比较了两者的优劣，探讨了SR1算法在处理非凸优化问题中病态鞍点时的潜力。

    

    尽管在大规模深度学习问题中，一阶方法非常流行来解决优化问题，但它们存在一些明显的缺点。为了减少这些缺点，最近一直有兴趣应用基于拟牛顿的二阶方法，这些方法使用梯度信息构建Hessian矩阵的近似。我们的研究主要关注随机拟牛顿算法在训练深度神经网络中的行为。我们分析了两种著名的拟牛顿更新算法，即有限内存Broyden-Fletcher-Goldfarb-Shanno（BFGS）和对称秩一（SR1）算法。这项研究填补了关于这两种方法真实性能的空白，并分析了在使用更强大的BFGS算法还是更廉价的SR1算法进行训练时是否得到更高效的结果，SR1算法允许不定Hessian矩阵近似，从而有助于更好地避开非凸优化问题中出现的病态鞍点。

    While first-order methods are popular for solving optimization problems that arise in large-scale deep learning problems, they come with some acute deficiencies. To diminish such shortcomings, there has been recent interest in applying second-order methods such as quasi-Newton based methods which construct Hessians approximations using only gradient information. The main focus of our work is to study the behaviour of stochastic quasi-Newton algorithms for training deep neural networks. We have analyzed the performance of two well-known quasi-Newton updates, the limited memory Broyden-Fletcher-Goldfarb-Shanno (BFGS) and the Symmetric Rank One (SR1). This study fills a gap concerning the real performance of both updates and analyzes whether more efficient training is obtained when using the more robust BFGS update or the cheaper SR1 formula which allows for indefinite Hessian approximations and thus can potentially help to better navigate the pathological saddle points present in the non
    
[^259]: 轨迹平衡：改进了GFlowNets中的信用分配

    Trajectory balance: Improved credit assignment in GFlowNets. (arXiv:2201.13259v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.13259](http://arxiv.org/abs/2201.13259)

    GFlowNets使用轨迹平衡作为一种更高效的学习目标，解决了先前学习目标中信用传播效率低下的问题，并且在实验中证明了其在收敛性、生成样本多样性以及鲁棒性方面的优势。

    

    生成流网络（GFlowNets）是一种学习使用动作序列生成组合对象（如图形或字符串）的随机策略的方法，其中许多可能的动作序列可能导致相同的对象。我们发现先前提出的GFlowNets学习目标，即流匹配和详细平衡，类似于时间差分学习，容易在长的动作序列中出现信用传播效率低下的问题。因此，我们提出了一种新的学习目标，即轨迹平衡，作为先前使用目标的更高效的替代方法。我们证明了轨迹平衡目标的任何全局极小值可以定义一个从目标分布精确采样的策略。在四个不同领域的实验中，我们从实证上证明了轨迹平衡目标对于GFlowNet收敛性、生成样本的多样性以及对长动作序列和噪声的鲁棒性的益处。

    Generative flow networks (GFlowNets) are a method for learning a stochastic policy for generating compositional objects, such as graphs or strings, from a given unnormalized density by sequences of actions, where many possible action sequences may lead to the same object. We find previously proposed learning objectives for GFlowNets, flow matching and detailed balance, which are analogous to temporal difference learning, to be prone to inefficient credit propagation across long action sequences. We thus propose a new learning objective for GFlowNets, trajectory balance, as a more efficient alternative to previously used objectives. We prove that any global minimizer of the trajectory balance objective can define a policy that samples exactly from the target distribution. In experiments on four distinct domains, we empirically demonstrate the benefits of the trajectory balance objective for GFlowNet convergence, diversity of generated samples, and robustness to long action sequences and
    
[^260]: NeuroBack: 使用图神经网络改进CDCL SAT求解

    NeuroBack: Improving CDCL SAT Solving using Graph Neural Networks. (arXiv:2110.14053v4 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2110.14053](http://arxiv.org/abs/2110.14053)

    NeuroBack提出了一种使用图神经网络改进CDCL SAT求解的方法，通过预测出现在大多数满足赋值中的变量的阶段，使得求解更加有效，并且消除了对GPU资源的依赖。

    

    命题可满足性（SAT）是一个影响到规划、验证和安全等许多研究领域的NP完全问题。主流的现代SAT求解器基于冲突驱动子句学习（CDCL）算法。最近的研究旨在利用图神经网络（GNNs）增强CDCL SAT求解器。然而，到目前为止，这种方法要么没有使求解更加有效，要么需要大量的GPU资源进行频繁的在线模型推断。为了使GNN的改进变得实用，本文提出了一种名为NeuroBack的方法，它建立在两个洞察上：（1）预测出现在大多数（甚至全部）满足赋值中的变量的阶段（即值）对于CDCL SAT求解至关重要，（2）在SAT求解开始之前，只需查询一次神经模型进行预测即可。一旦训练完成，离线模型推断使NeuroBack能够仅在CPU上执行，消除了对GPU资源的依赖。

    Propositional satisfiability (SAT) is an NP-complete problem that impacts many research fields, such as planning, verification, and security. Mainstream modern SAT solvers are based on the Conflict-Driven Clause Learning (CDCL) algorithm. Recent work aimed to enhance CDCL SAT solvers using Graph Neural Networks (GNNs). However, so far this approach either has not made solving more effective, or required substantial GPU resources for frequent online model inferences. Aiming to make GNN improvements practical, this paper proposes an approach called NeuroBack, which builds on two insights: (1) predicting phases (i.e., values) of variables appearing in the majority (or even all) of the satisfying assignments are essential for CDCL SAT solving, and (2) it is sufficient to query the neural model only once for the predictions before the SAT solving starts. Once trained, the offline model inference allows NeuroBack to execute exclusively on the CPU, removing its reliance on GPU resources. To t
    
[^261]: AKE-GNN: 自适应知识交流的有效图学习

    AKE-GNN: Effective Graph Learning with Adaptive Knowledge Exchange. (arXiv:2106.05455v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2106.05455](http://arxiv.org/abs/2106.05455)

    AKE-GNN是一种新型的图神经网络学习框架，通过自适应知识交换策略在多个图视图之间交换通道，以实现有效的图学习。

    

    图神经网络(GNNs)已经被广泛应用于各种图挖掘任务。然而，最近的研究揭示了在经过训练的GNN中学到的权重(通道)是高度冗余的，这不可避免地限制了GNN的性能。我们的目标不是为了效率考虑而移除这些冗余通道，而是试图重新激活它们，以扩大GNN的表示能力，实现有效的图学习。本文提出了一种名为AKE-GNN的新型GNN学习框架，它通过对图增强生成的多个图视图之间进行自适应知识交换策略来实现这个目标。AKE-GNN首先训练多个GNN，每个对应一个图视图，以获得信息通道。然后，AKE-GNN迭代地以逐层方式在一个GNN的权重参数矩阵中进行冗余通道与另一个GNN的信息通道之间的交换。

    Graph Neural Networks (GNNs) have already been widely used in various graph mining tasks. However, recent works reveal that the learned weights (channels) in well-trained GNNs are highly redundant, which inevitably limits the performance of GNNs. Instead of removing these redundant channels for efficiency consideration, we aim to reactivate them to enlarge the representation capacity of GNNs for effective graph learning. In this paper, we propose to substitute these redundant channels with other informative channels to achieve this goal. We introduce a novel GNN learning framework named AKE-GNN, which performs the Adaptive Knowledge Exchange strategy among multiple graph views generated by graph augmentations. AKE-GNN first trains multiple GNNs each corresponding to one graph view to obtain informative channels. Then, AKE-GNN iteratively exchanges redundant channels in the weight parameter matrix of one GNN with informative channels of another GNN in a layer-wise manner. Additionally, 
    
[^262]: Naive Exploration is Optimal for Online LQR（在线LQR问题中，天真的探索是最优的）

    Naive Exploration is Optimal for Online LQR. (arXiv:2001.09576v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2001.09576](http://arxiv.org/abs/2001.09576)

    在线LQR问题中，我们证明了天真的探索是最优的策略，可以在未知参数的情况下达到最小遗憾。这一结论对于解决在线自适应控制问题具有重要意义。

    

    本文研究了线性二次调节器在线自适应控制的问题，其中真实系统参数是未知的。我们证明了新的上下界，表明最优遗憾的尺度与$\widetilde{\Theta}({\sqrt{d_{\mathbf{u}}^2 d_{\mathbf{x}} T}})$成正比，其中$T$是时间步数，$d_{\mathbf{u}}$是输入空间的维度，$d_{\mathbf{x}}$是系统状态的维度。值得注意的是，我们的下界排除了可能存在的$\mathrm{poly}(\log{}T)$遗憾算法的可能性，这是由于问题明显的强凸性引出的猜想。我们的上界通过一种简单的$\textit{{确定性等价控制}}$的变体得到，其中学习者根据对系统的估计选择控制输入，并注入探索性的随机噪声。虽然这种方法已经被证明可以实现$\sqrt{T}$的遗憾(Raina et al. 2019)，但我们表明，如果学习者不断改进他们的估计，找到了系统的真实参数，则可以实现零遗憾。

    We consider the problem of online adaptive control of the linear quadratic regulator, where the true system parameters are unknown. We prove new upper and lower bounds demonstrating that the optimal regret scales as $\widetilde{\Theta}({\sqrt{d_{\mathbf{u}}^2 d_{\mathbf{x}} T}})$, where $T$ is the number of time steps, $d_{\mathbf{u}}$ is the dimension of the input space, and $d_{\mathbf{x}}$ is the dimension of the system state. Notably, our lower bounds rule out the possibility of a $\mathrm{poly}(\log{}T)$-regret algorithm, which had been conjectured due to the apparent strong convexity of the problem. Our upper bound is attained by a simple variant of $\textit{{certainty equivalent control}}$, where the learner selects control inputs according to the optimal controller for their estimate of the system while injecting exploratory random noise. While this approach was shown to achieve $\sqrt{T}$-regret by (Mania et al. 2019), we show that if the learner continually refines their esti
    
[^263]: 隐式正则化ReLU神经网络如何刻画学习函数 - 第一部分：随机第一层的两层一维情况

    How Implicit Regularization of ReLU Neural Networks Characterizes the Learned Function -- Part I: the 1-D Case of Two Layers with Random First Layer. (arXiv:1911.02903v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/1911.02903](http://arxiv.org/abs/1911.02903)

    本文研究了一维ReLU神经网络，通过数学分析和实验证明了对于这种网络，L2正则化回归在函数空间中对应于对估计的二阶导数进行正则化，同时提出了早停止的梯度下降和平滑样条回归之间的新对应关系。

    

    本文研究了一维（浅层）ReLU神经网络，其中权重是随机选择的，只有终端层进行训练。首先，我们从数学上证明了对于这种网络，L2正则化回归在函数空间中对应于对估计的二阶导数进行正则化，适用于相当一般的损失函数。对于最小二乘回归，我们证明了训练好的网络在隐藏节点数趋向无穷时收敛到训练数据的平滑样条插值。此外，我们推导出了早停止的梯度下降（没有对权重进行显式正则化）与平滑样条回归之间的新对应关系。

    In this paper, we consider one dimensional (shallow) ReLU neural networks in which weights are chosen randomly and only the terminal layer is trained. First, we mathematically show that for such networks L2-regularized regression corresponds in function space to regularizing the estimate's second derivative for fairly general loss functionals. For least squares regression, we show that the trained network converges to the smooth spline interpolation of the training data as the number of hidden nodes tends to infinity. Moreover, we derive a novel correspondence between the early stopped gradient descent (without any explicit regularization of the weights) and the smoothing spline regression.
    

