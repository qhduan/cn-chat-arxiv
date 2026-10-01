# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Global Sensitivity Indices from Observational Data: A Metamodel-Based Approach](https://arxiv.org/abs/2609.40342) | 该论文提出MM-GSA方法，利用监督学习构建元模型，从无法重复实验的观测数据中进行全局敏感性分析，结合模型无关的一阶Sobol'指数估计量与新的基于触发器的结构指数来量化输入相关性，并证明了两者的相合性及结构指数的变量选择性质。 |
| [^2] | [Signal Processing over Product DAGs: Causal Shifts and Filters](https://arxiv.org/abs/2609.40275) | 本文提出了一种新的有向无环图（DAG）乘积算子，使得乘积DAG上的结构方程模型、傅里叶模态、因果平移和滤波器在两个图因子之间均可分离，从而为双域线性因果结构构建了高效的信号处理框架。 |
| [^3] | [Distribution Matching Distillation for Continuous Diffusion Language Models](https://arxiv.org/abs/2609.40235) | 提出了两种分布匹配蒸馏方法Simplex-DMD和Reinforce-DMD，通过利用学生模型的概率性词元输出，将连续扩散语言模型的网络评估次数大幅降低至仅需4次即可实现高质量文本生成。 |
| [^4] | [Cheap to Draw, Expensive to Trust: Certifying Test-Time Scaling Curves](https://arxiv.org/abs/2609.40190) | 该论文推导了统计认证整条测试时扩展曲线的极小极大采样成本，并证明利用“基准是固定题目列表、方差主要来自题目之间”这一结构，可以避免为错误的不确定性买单，从而大幅降低同时认证所有预算所需的生成样本量。 |
| [^5] | [Partial identification with entropy regularized optimal transport](https://arxiv.org/abs/2609.40156) | 该论文将统计部分识别中的优化问题统一表述为路径空间上的最优传输问题，通过熵正则化将其转化为可用Sinkhorn迭代高效求解的多边际熵正则化最优传输问题，并建立了正则化值向最锐利边界的收敛性及插件估计量的一致性收敛速率。 |
| [^6] | [Proximal Balancing for Causal Effect Estimation under Unmeasured Confounding](https://arxiv.org/abs/2609.40051) | 提出邻近平衡方法，将经典协变量平衡思想扩展到仅能通过代理变量观测的混杂因素，无需指定代理角色、求解逆问题或依赖潜变量模型，即可实现未测量混杂下的因果效应估计。 |
| [^7] | [Gromov-Wasserstein Distillation for Inductive Multi-View Embedding](https://arxiv.org/abs/2609.40047) | 该论文提出了一种基于重心蒸馏的归纳式框架，通过GW-MDS教师模型与神经学生网络的蒸馏，实现了对未见样本的显式多视图嵌入映射，并避免了推理时额外的GW优化计算。 |
| [^8] | [Efficient Active Auditing of Multi-Group Fairness with Bias Probes](https://arxiv.org/abs/2609.40034) | 提出偏差探针框架，通过有针对性的自适应查询高效审计多群体公平性，在保持模型机密性的同时揭示数据分布中驱动偏差的结构。 |
| [^9] | [Amortized Bayesian Inference on Multilevel Models of Arbitrary Structure](https://arxiv.org/abs/2609.40024) | 提出了一种基于图操作（图扩展与图反转）的通用摊销贝叶斯推断方法，能够自动为任意结构的多层模型推导有效的后验因式分解及匹配的神经网络架构，在完整保留生成模型依赖假设的同时，将训练后的推断简化为近即时的前向计算。 |
| [^10] | [BayesNDE: Bayesian Generative Modeling for Neural Density Estimation](https://arxiv.org/abs/2609.39843) | 提出了一种基于贝叶斯生成建模的神经密度估计器BayesNDE，无需可逆网络或雅可比行列式计算，通过为每个观测推断样本特定的潜在后验构建自适应提案并结合桥接采样来估计密度，在密度估计精度和异常检测性能上均超越了现有最先进方法。 |
| [^11] | [Estimation of the Label-Noise Transition Matrix with Performance Guarantees via Selective Classification](https://arxiv.org/abs/2609.39829) | 该论文提出了一种基于单侧选择性分类的标签噪声转移矩阵估计新方法，绕过了脆弱的类后验概率估计，提供了有限样本性能保证，并利用灵活的二元分类学习方法及有效算法加以实现。 |
| [^12] | [Probabilistic Adversarial Training](https://arxiv.org/abs/2609.39798) | 本文提出概率对抗训练，从两个分布重叠的概率视角出发，证明基于KL散度的下界可作为概率鲁棒性的可处理代理目标，最大化该下界即可恢复出对抗训练的缩放形式。 |
| [^13] | [Amortized ratio-estimation importance sampling and localized simulation-based calibration for intractable likelihoods](https://arxiv.org/abs/2609.39712) | 该论文提出了针对难解似然模型的顺序化后验估计方法与局部化模拟校准技术，利用局部化条件密度代理和摊销化的似然-证据比率估计器，以低计算成本实现准确且经过校准的贝叶斯推断。 |
| [^14] | [BAM! Bayesian Anything Model: a foundation model for generative computational imaging](https://arxiv.org/abs/2609.39660) | 本文提出仅含3600万参数的轻量级基础模型BAM，通过将RAM骨干升级为条件流映射，使仪器物理特性可在推理时灵活指定，从而实现少步数的物理感知后验采样，并能零样本或仅需极少微调即可泛化到未见过的数据和任务。 |
| [^15] | [Mitigating Representation Gaps in Amortized Bayesian Inference with Auxiliary Supervision](https://arxiv.org/abs/2609.39525) | 提出一种通用的辅助监督方法，通过对神经网络内部表示施加引导损失来缓解摊销贝叶斯推断中的表示差距，在数据充足时加快收敛，在数据稀缺时提升推断性能。 |
| [^16] | [The Geometry of Randomized Smoothing on Feasible Sets](https://arxiv.org/abs/2609.39497) | 该论文将可行性/置信度过滤下的随机平滑认证问题分解为几何与认证两部分，证明凸保留集能完整保持高斯比较，一般集合需要对保留分布的几何控制（否则无法认证任何正半径），而联合保留与标签概率总能给出有效认证。 |
| [^17] | [CAMOS: Coupled Oscillatory State-Space Model for Multimodal Clinical Time-Series](https://arxiv.org/abs/2609.39484) | 本文证明了线性状态空间模型用掩码处理缺失模态存在表征局限——无法建模模态联合存在或缺失的交互作用，并提出CAMOS模型，通过可用性门控的耦合二阶振荡器使转移算子本身随模态可用性变化，从而更好地建模多模态临床时间序列。 |
| [^18] | [Distributionally robust linear regression through the lens of adversarial training](https://arxiv.org/abs/2609.39449) | 本文提出以 Wasserstein 分布鲁棒优化框架研究线性回归，统一了平方根 Lasso 与对抗线性回归这两个特例，并证明了该一般方法具有样本内误差界、对噪声水平的枢轴性以及解的等价性等重要性质。 |
| [^19] | [Raw-Routed Mixture of Adapters: A Causal Intervention for Routing Collapse in Time Series Foundation Models](https://arxiv.org/abs/2609.39445) | 该论文揭示了实例归一化会剥离路由器区分数据模式所需的统计信息、从而导致时间序列基础模型中混合专家发生路由坍缩，并据此提出原始路由的适配器混合作为因果干预方案，其信号比指标可在训练前预测数据集的脆弱性（Spearman ρ = -0.88）。 |
| [^20] | [Principal Component Regression Dominates all Monotone Spectral Filters for Linear Regression](https://arxiv.org/abs/2609.39440) | 本文证明经过最优调参的主成分回归（PCR）在逐实例有限样本风险意义下优于包括梯度下降和岭回归在内的所有单调谱滤波器，从而在单调滤波器类中达到最优且可容许。 |
| [^21] | [Awakening of the Buddha: Subspace Learning During Population-Loss Plateaus](https://arxiv.org/abs/2609.39408) | 该论文证明了神经网络在总体损失几乎不变的高损失平台期内仍能显著学习到更具预测性的表示——教师子空间与 AGOP 前导特征空间的最小对齐度相比初始化至少提升 1/2，且固定系数预算下的最小重拟合 MSE 下降超过 0.399。 |
| [^22] | [Towards Optimal Inventory Control under Censored Demand: A Biased Sample-Average Approximation Approach](https://arxiv.org/abs/2609.39397) | 该论文针对删失需求下的多周期缺货损失库存控制问题，提出了基于新成本分解与覆盖条件的有偏样本平均近似（SAA）统一框架，通过设计上偏和下偏两种算法，分别在离线场景下实现近乎最优的样本复杂度、在在线场景下实现近乎最优的遗憾界。 |
| [^23] | [A Dynamical Theory of LoRA in Continual Learning](https://arxiv.org/abs/2609.39367) | 该论文在师生模型中建立了持续学习下LoRA的精确动力学理论，揭示低秩更新虽能减少对已学任务特征的干扰从而缓解灾难性遗忘，但其初始化会减慢对新任务的适应。 |
| [^24] | [A Unified Dual Method for Matching Problems](https://arxiv.org/abs/2609.39339) | 本文基于对偶理论提出了一个统一框架，将可分解为凸函数之差（DC）的匹配目标重构为隐式配准问题，并将其应用于Gromov-Wasserstein及其非平衡形式和变体等二次匹配问题，同时提供了广泛的收敛性保证。 |
| [^25] | [Discrete Score Matching Enables Causal Discovery from Count Data](https://arxiv.org/abs/2609.39326) | 该论文提出条件曲率分数（CCS）与对角线外曲率分数（OCS），将基于分数匹配的因果发现方法从连续数据推广到计数数据，并证明零CCS可精确刻画半参数广义线性模型的条件形式，从而实现从计数数据中恢复因果有向无环图。 |
| [^26] | [Minimax Additive Regression under Unknown Dependent Designs](https://arxiv.org/abs/2609.39212) | 该论文研究了维度随样本量增长、设计分布可能非乘积且边缘密度未知情形下的可加回归问题，通过 Riesz 基构造与阈值化最小二乘估计，建立了已知或未知边缘密度下相互匹配的极小极大预测误差界，并揭示了密度正则性何时影响最优速率。 |
| [^27] | [Asymptotic Properties of Support Vector Machines in High-Dimension, Low-Sample-Size Settings under a Spiked Model](https://arxiv.org/abs/2609.39173) | 在尖峰模型下，HDLSS数据的几何表示不再成立，支持向量机不具有相合性（误分类率不趋于零），而偏差校正SVM也因偏差项本身需要修改而无法取得更优性能。 |
| [^28] | [Steepest Guidance: A Practical and Principled Approach to Inference-Time Alignment of Flow and Diffusion-based Models](https://arxiv.org/abs/2609.39091) | 本文提出“最速引导”框架，将流模型和扩散模型的推理时对齐建模为概率测度空间中的序贯优化问题，通过最大化目标的局部改进来规避Doob h-变换最优引导估计的困难，提供了一种兼具理论支撑与实用性的推理时对齐方法。 |
| [^29] | [A Rank Graduation metric for Algorithmic fairness](https://arxiv.org/abs/2609.39025) | 本文提出一个基于秩的公平性评估框架（RGF 及其积分度量 AURGF），通过模型预测误差的分布来衡量算法公平性，将公平性评估与预测精度和可解释性相联系，并配有推断检验和特征移除程序以识别导致不公平的因素。 |
| [^30] | [Minimax rates for learning spectral Barron functions by deep ReLU neural networks](https://arxiv.org/abs/2609.39020) | 本文为深度ReLU神经网络逼近和学习谱Barron函数建立了新的逼近界，并证明了相应的极小极大最优学习速率 $n^{-\frac{d+2s}{2d+2s}}$。 |
| [^31] | [On the Relaxation of Conditional Independence Assumption for Image Segmentation](https://arxiv.org/abs/2609.38930) | 该论文提出用空间局部化依赖（SLD）结构替代RankSEG方法中的条件独立性假设，并借助倒数矩近似与不动点优化策略克服计算瓶颈，从而有效捕获局部标签相关性并提升模糊、低对比度场景下的图像分割性能。 |
| [^32] | [Flow Matching under Noisy Latent Structure: Beyond Exact Low-Dimensional Support](https://arxiv.org/abs/2609.38918) | 论文研究了噪声潜在生成模型下的流匹配，证明了其样本复杂度指数由潜在维度而非环境维度决定，突破了传统理论要求精确低维支撑的限制。 |
| [^33] | [Warm-starting PDE solvers with any-dimensional machine learning](https://arxiv.org/abs/2609.38916) | 提出了基于偏微分方程与初始数据对称性的数学条件，使低维训练的机器学习PDE求解器能够零样本直接求解高维PDE，并在数据不满足对称性时提供有原则的热启动策略，从而在热传导方程、Burgers方程和可压缩Navier-Stokes方程上均提升了性能。 |
| [^34] | [Amortized Data Borrowing with Exchangeability-Aware Neural Posterior Estimation](https://arxiv.org/abs/2609.38902) | 本文提出一种可交换性感知的摊销式神经后验估计方法，通过在模拟的当前/外部数据集对上预训练单个网络，实现一次前向传播即可快速获得当前研究的近似后验，为传统依赖手工先验和MCMC的贝叶斯动态借入提供了高效且可泛化的替代方案。 |
| [^35] | [Sharp Statistical Rates for Asynchronous TD Learning with Markovian Data](https://arxiv.org/abs/2609.38880) | 该论文为基于单条马尔可夫轨迹的表格型TD学习建立了精确的末次迭代样本复杂度界 $\widetilde O\left(\frac{H^3}{\mu_{\min}\varepsilon^2}+\frac{t_{\operatorname{mix}}}{\mu_{\min}}\right)$，且该界对常数步长与递减步长调度均成立。 |
| [^36] | [Optimal Allocation and Volume under Surface](https://arxiv.org/abs/2609.38875) | 本文提出基于Aumann期望表示与Minkowski混合体积的框架来估计ROC曲面下体积（VUS），并开发了其双重/去偏机器学习估计量及推断方法，还可用于分组可行误差分析和基尼系数的推广。 |
| [^37] | [Optimal VC Dimension of Contrastive Learning with Margin](https://arxiv.org/abs/2609.38834) | 该论文解决了Alon等人提出的开放问题，确定了带间隔对比学习的最优VC维界。 |
| [^38] | [Understanding Off- vs On-Policy Distillation: A Tale of Distinct Training Objectives](https://arxiv.org/abs/2609.38666) | 该论文从理论上揭示了在策略与离策略蒸馏分别对应反向KL与前向KL散度下的不同聚合目标（几何聚合与算术混合），由此解释了在策略蒸馏在减少遗忘方面的优势及其脆弱性。 |
| [^39] | [Bandits with Multiple Optimal Arms: Minimax Regret and Non-Adaptivit](https://arxiv.org/abs/2609.38659) | 该论文针对具有多个最优臂的多臂老虎机问题，通过对子采样算法的更精细分析建立了近乎极小极大最优的遗憾界 $\tilde{O}(\frac{K-A}{\sqrt{KA}}\sqrt{T})$，给出匹配下界，并证明了解最优臂数量对于达到近乎最优遗憾是必要的。 |
| [^40] | [Adaptive mixture variational inference for spike-and-slab regression](https://arxiv.org/abs/2609.38656) | 该论文提出了一种针对尖峰-厚板先验回归的自适应混合变分推断方法，通过在变量包含指示和活跃系数上直接最小化反向KL散度来捕捉相关变量选择的联合不确定性，并给出了后验收缩性、选择一致性和Bernstein–von Mises近似等理论保证。 |
| [^41] | [Learning-Enabled Estimation: Tight Characterizations under Sample Selection Biases](https://arxiv.org/abs/2609.38608) | 该论文首次为存在样本选择偏差时的回归学习问题提供了完整的紧致刻画，确立了选择机制函数形式所需的最小必要假设条件。 |
| [^42] | [Towards Universal Wasserstein Barycenters through Flow Matching](https://arxiv.org/abs/2609.38547) | 提出BaryFM流匹配模型，实现对Wasserstein单纯形上任意权重重心的通用近似，一次训练后即可通过常微分方程从任意重心测度采样，并在领域自适应、贝叶斯后验聚合等多个下游任务中取得最佳平均表现。 |
| [^43] | [Generative sequence modeling for infinite memory processes via predictive states](https://arxiv.org/abs/2609.38524) | 本文提出一种基于预测状态的生成式序列建模新方法，能够处理具有无穷记忆的随机过程，当过去历史可被压缩为低维充分统计量时实现快速收敛，且估计问题的统计复杂度仅由预测状态空间的内在维数决定。 |
| [^44] | [ShamAN-Q: Shampoo Augmented NanoQuant for Sub-1-bit LLM Weights](https://arxiv.org/abs/2609.38521) | ShamAN-Q用Shampoo式的密集曲率度量（对经验Fisher信息矩阵进行Kronecker拟合并构建马氏重建损失）替代NanoQuant的对角重建几何，实现了大语言模型的低于1比特训练后量化。 |
| [^45] | [Grokking through the Lens of Minimum-Norm Interpolation](https://arxiv.org/abs/2609.38453) | 该论文建立了一个统计理论，证明在高维过参数化无噪声回归中存在零—一泛化定律，并构造了一族凸范数，使插值解在保持零训练误差的同时从平凡解过渡到信号的精确恢复，从而从正则化几何与信号稀疏性的角度为Grokking的延迟泛化现象提供了定量解释。 |
| [^46] | [Local polynomial density ratio estimation](https://arxiv.org/abs/2609.38412) | 提出了一种新的局部多项式密度比估计器，在任意光滑度的 Hölder 类上达到无需额外对数因子的逐点极小化极大最优速率，仅需假设密度比本身光滑而无需单个密度光滑，在支撑集边界点上依然有效，并附带可用于分类任务的集中不等式。 |
| [^47] | [Advantage of Sample Complexity in Quantum PAC Learning Requires Inverse Access to State-Preparation Unitaries](https://arxiv.org/abs/2609.38403) | 本文证明了在量子PAC学习中，查询复杂度对精度参数依赖的改进优势本质上必须依赖于对态制备酉算子逆的访问，而仅有前向访问的酉算子无法带来任何样本复杂度上的优势。 |
| [^48] | [PTED: A multi-dimensional two-sample test for scientific inference and generative machine learning](https://arxiv.org/abs/2609.38388) | 本文提出了基于能量距离的置换检验PTED——一种Python实现的多维精确双样本检验方法，由于统计量仅依赖成对距离，它可适用于高维数据、任意可定义距离的数据类型及不平衡样本，并通过近似公式实现线性扩展。 |
| [^49] | [Learning to Plan from Random Exploration](https://arxiv.org/abs/2609.38383) | 该论文提出一种仅从随机探索数据中、通过时程条件化的能量模型与噪声对比估计来学习时间关系的方法，从而无需动作或奖励标签、无需策略改进训练即可实现长程规划。 |
| [^50] | [Lower Bounds for Linear-Oracle Online Learning](https://arxiv.org/abs/2609.38375) | 本文证明了Weibel等人的猜想，即每轮常数次线性最小化无法在一般凸集上改进在线Frank-Wolfe的$T^{3/4}$遗憾率，并将该下界扩展到仅使用预言机模型中的所有确定性学习者。 |
| [^51] | [Acceleration of Diffusion Language Model through Discrete Average Generator](https://arxiv.org/abs/2609.38364) | 提出离散平均生成器，将MeanFlow扩展至连续时间马尔可夫链，通过自洽恒等式训练目标实现扩散语言模型的高效少步生成加速。 |
| [^52] | [Robust LassoNet: Enhancing Feature Selection in Neural Networks via Robust Loss Functions](https://arxiv.org/abs/2609.38263) | 本文提出Robust LassoNet，通过在LassoNet中引入Huber、Cauchy、Tukey双平方等鲁棒损失函数，在保留原始优化框架的同时，显著提升了神经网络在数据污染情况下的特征选择准确性与预测性能。 |
| [^53] | [Conformal Factuality Control for Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2609.38222) | 该研究将主张级保形事实性控制应用于多跳检索增强生成，证明在六种模型-数据集配置中，保形过滤能将保留主张获完全支持的回复比例从无过滤时的55.60%-76.03%稳定提升至95%目标下的95.80%-97.20%。 |
| [^54] | [Identifiability Guarantees for Drivers and Dynamics of Delayed Physical Systems](https://arxiv.org/abs/2609.37944) | 本文提出一种有理论支撑的方法，证明在宽松假设下随机时滞微分方程的结构驱动项与漂移项是可辨识的，并在驱动项可辨识性与动力学物理一致性基准上优于现有方法。 |
| [^55] | [A Mesoscopic View of Transformer Weights Through Row and Column Scale Fields](https://arxiv.org/abs/2609.35852) | 提出以“行与列尺度场”作为Transformer权重矩阵的介观级精确表示，揭示平衡化后核心幅值分布在不同模型规模与初始化下高度相似，且尺度场在共享功能通道的投影间对齐、查询/键分布随重新分配的RoPE频率变化。 |
| [^56] | [Global Convergence of Third-Order Langevin Dynamics for Non-Convex Optimization via Simulated Annealing](https://arxiv.org/abs/2609.28611) | 该论文证明了在模拟退火框架下，采用固定摩擦与递减噪声的三阶朗之万动力学在非凸优化中可依概率收敛到全局最小值，并给出了离散化格式保持该收敛速率的充分步长条件。 |
| [^57] | [From Prediction to Explainable Provider Behavior Profiles for Fraud, Waste, and Abuse Review](https://arxiv.org/abs/2609.28477) | 该研究提出将欺诈、浪费与滥用（FWA）审查从预测建模转向可解释的提供者行为画像，通过将账单收入分解为提供者规模与诊疗项目构成的乘积，从而解释提供者行为变化的原因。 |
| [^58] | [Stochastic Flow Map for Count Data](https://arxiv.org/abs/2609.23290) | Count Flow Map 是一种直接在计数空间中学习有限时间随机转移的生成模型，通过泊松出生与二项死亡机制保持非负整数计数，实现一步或少数几步的高效计数数据生成。 |
| [^59] | [How Many Posterior Samples? Calibrated Stopping for Adaptive Sensing](https://arxiv.org/abs/2609.21813) | 论文揭示了自适应感知中基于票数份额阈值的即插即用停止规则并不提供置信度保证，并提出经校准的固定样本规则、有限视界序贯规则以及精确截断方法，以在规定错误声明概率下回答“需要多少后验样本”这一停止问题。 |
| [^60] | [ChorusTIC: Training-Free Multivariate Time Series Classification via Chorus In-Context Learning](https://arxiv.org/abs/2608.24033) | ChorusTIC提出了一种无需训练的分类原生基础模型，通过情节一致的随机子通道拼接和双轴编码器，在异构通道配置下实现多变量时间序列的上下文分类，无需目标任务参数更新。 |
| [^61] | [Zero-Flow Two-Sample Tests](https://arxiv.org/abs/2607.21542) | 提出零流双样本检验（ZF2ST），通过可学习速度场的时间反演反对称性刻画分布相等性，利用见证函数的变分表示直接最大化检验功效，同时在留出样本上检验以保证第一类错误控制。 |
| [^62] | [Certifying Residual Architectures from Their Primitives: A Sharp Stability Threshold](https://arxiv.org/abs/2607.14576) | 该论文提出一种在训练之前即可从残差块基本构件出发、以深度和浮点格式的显式函数形式认证残差架构训练稳定性的方法，通过前向幂律增长界与后向Lipschitz梯度界给出了状态能否达到最大有限浮点值的锐利稳定性阈值。 |
| [^63] | [From Dataset Spectral Geometry to Network Weights: A Geometry-Aware Initialization for Sigmoidal MLPs in Image Classification](https://arxiv.org/abs/2606.28444) | 提出一种几何感知的初始化方法，通过对各类别训练样本进行SVD谱分析，将类别几何结构编码为成对的Sigmoid板状门控并直接编译进单隐层Sigmoid型MLP的权重中，从而实现基于数据几何的图像分类网络初始化。 |
| [^64] | [Flow Annealing Posterior Sampling for Function-Space Regression and Inverse Problems](https://arxiv.org/abs/2606.22346) | 提出了首个统一随机过程回归与偏微分方程逆问题的函数空间后验采样框架 FLAPS，其利用预训练流匹配先验与低秩朗之万校正，从稀疏含噪观测中生成连贯且不确定性校准良好的后验样本，显著优于现有方法。 |
| [^65] | [Closing the Approximation Gap in Simulation-free Latent SDEs](https://arxiv.org/abs/2606.16138) | 本文揭示现有免模拟变分推断算法虽高效，但其参数化将近似后验限制在SDE的一个子集中而产生近似差距，并提出方法弥合这一差距，兼得效率与表达能力。 |
| [^66] | [Generalization in Nonlinear Least Squares via Learned Feature Geometry](https://arxiv.org/abs/2606.08799) | 该论文利用平均算法稳定性为岭正则化非线性最小二乘模型建立泛化误差界，核心创新是引入一个由训练后参数处的经验雅可比Gram矩阵与残差曲率项构成的数据依赖有效维度，并通过梯度特征的覆盖复杂度对其进行控制，从而使泛化保证取决于学习到的特征几何（如流形内在维度）而非参数数量。 |
| [^67] | [Tighter Regret Bounds for Contextual Action-Set Reinforcement Learning](https://arxiv.org/abs/2605.15692) | 本文将MVP算法扩展到具有片段依赖可行动作集的上下文强化学习框架，建立了对抗上下文下 $\widetilde{O}(\sqrt{SAH^3K\log L})$ 的极小极大遗憾界，并由此导出随机上下文下的 $\widetilde{O}(\sqrt{SAH^3K})$ 遗憾界与 $\widetilde{O}(SAH^3/\epsilon^2)$ 的样本复杂度保证。 |
| [^68] | [Time-adaptive infinite-dimensional Gaussian process regression on manifolds](https://arxiv.org/abs/2603.21144) | 本文基于经验贝叶斯方法提出了流形上泛函高斯过程回归的新框架，通过拉普拉斯-贝尔特拉米算子特征函数对应的时变角度谱及时间自适应截断方案实现了降维与有效预测。 |
| [^69] | [Diffusion-Augmented Markov Decision Processes for Maximum Entropy Reinforcement Learning](https://arxiv.org/abs/2512.02019) | 该论文提出扩散增强马尔可夫决策过程（DA-MDPs），将反向扩散的每一步去噪转移视为独立的强化学习决策，并通过数据处理不等式推导出可分解的反向KL散度上界，为将最大熵强化学习算法推广到扩散策略提供了通用的理论框架。 |
| [^70] | [Beyond Uncertainty Sets: Leveraging Optimal Transport to Extend Conformal Predictive Distributions to Multivariate Settings](https://arxiv.org/abs/2511.15146) | 本文通过对最优传输分位数区域进行共形化，首次将共形预测分布扩展到多元/向量值分数场景，并恢复了有限样本、无分布假设的覆盖率保证。 |
| [^71] | [Sequential Bayesian Evaluation of Large Language Model Behavior](https://arxiv.org/abs/2511.10661) | 本文提出一种序贯贝叶斯评估框架，通过量化LLM随机性带来的评估不确定性，并自适应地优先选择下一个最值得评估的基准提示词，从而实现更具成本效益的大语言模型行为评估。 |
| [^72] | [Transferable Generative Models Bridge Femtosecond to Nanosecond Time-Step Molecular Dynamics](https://arxiv.org/abs/2510.07589) | 该研究提出一种可迁移的深度生成建模框架，在保持物理真实性的同时将分子动力学采样加速四个数量级，并能跨化学组成和体系规模泛化，从而实现对此前难以企及的平衡系综与慢速弛豫动力学过程的定量表征。 |
| [^73] | [Effects of Structural Allocation of Geometric Task Diversity in Linear Meta-Learning Models](https://arxiv.org/abs/2509.18349) | 本研究证明了元学习性能不仅取决于任务参数的整体几何多样性，还取决于这种多样性相对于底层低维结构的分配方式，从而深化了对任务多样性影响元学习机制的理解。 |
| [^74] | [Differentiable Expectation-Maximisation and Applications to Gaussian Mixture Model Optimal Transport](https://arxiv.org/abs/2509.02109) | 本文提出了多种使EM算法可微分的方法，并将其应用于计算高斯混合模型间的混合Wasserstein距离，使MW2可作为可微分损失用于成像和机器学习任务，并提供了新颖的稳定性理论保证。 |
| [^75] | [Barycentric subspace analysis of network-valued data](https://arxiv.org/abs/2507.23559) | 本文提出重心子空间分析（BSA）方法，用样本点而非向量来生成降维子空间，从而提升了对无标注网络值数据进行探索性分析时的可解释性。 |
| [^76] | [Conformalized Regression for Continuous Bounded Outcomes](https://arxiv.org/abs/2507.14023) | 本文在变换回归模型框架下为连续有界结果提出保形预测区间方法，通过基于模型残差的分位数-残差非一致性分数，同时处理数据的异方差性和边界附近的不对称性，并连接了归一化保形预测与分布保形预测。 |
| [^77] | [Bringing Generative Learning to Representation Learning: Self-Supervised Transfer Learning as Distribution Matching](https://arxiv.org/abs/2502.14424) | 本文提出将表示学习重新定义为分布匹配，通过匹配显式几何参考分布来学习增强不变的编码器，从而实现自监督迁移学习，并证明了其理论保证和实际效果。 |
| [^78] | [Decentralized Projection-free Online Upper-Linearizable Optimization with Applications to DR-Submodular Optimization](https://arxiv.org/abs/2501.18183) | 该论文提出了一个去中心化无投影在线优化框架，将无投影方法推广到上线性化函数（涵盖DR-次模函数），在任意参数0≤θ≤1下实现O(T^{1-θ/2})遗憾值、O(T^{θ})通信复杂度和O(T^{2θ})次预言机调用，并首次给出一般凸约束下单调及非单调上凹优化的结果，且将结论扩展至零阶、半老虎机和老虎机反馈。 |
| [^79] | [Mining Causality: AI-Assisted Search for Instrumental Variables](https://arxiv.org/abs/2409.14202) | 本文提出利用大语言模型通过叙事和反事实推理自动搜索新颖有效的工具变量，大幅加速因果推断中工具变量的发现过程。 |
| [^80] | [Causal Inference in Possibly Nonlinear Factor Models](https://arxiv.org/abs/2008.13651) | 本文针对含有噪声测量混杂因素的处理效应模型提出了一种因果推断方法，利用结合K近邻匹配与主成分分析的局部主子空间逼近程序以及双重稳健得分函数，在代理变量与潜在混杂因素之间存在未知非线性因子结构的情况下实现了对多种因果参数的有效估计。 |
| [^81] | [Constrained Classification and Policy Learning.](http://arxiv.org/abs/2106.12886) | 研究了受限分类和策略学习中替代损失程序的一致性和适用性。 |

# 详细

[^1]: 从观测数据中学习全局敏感性指数：一种基于元模型的方法

    Learning Global Sensitivity Indices from Observational Data: A Metamodel-Based Approach

    [https://arxiv.org/abs/2609.40342](https://arxiv.org/abs/2609.40342)

    该论文提出MM-GSA方法，利用监督学习构建元模型，从无法重复实验的观测数据中进行全局敏感性分析，结合模型无关的一阶Sobol'指数估计量与新的基于触发器的结构指数来量化输入相关性，并证明了两者的相合性及结构指数的变量选择性质。

    

    经典的基于方差的全局敏感性分析（GSA）假设输入-输出机制可以在设计的采样方案下被重复评估，当仅有给定的观测样本可用时，这一假设是无法实现的。我们提出MM-GSA，一种基于元模型、从观测数据出发进行GSA的方法，其中利用监督学习来逼近系统性的输入-输出关系。MM-GSA结合了两种互补的输入相关性视角：一个是模型无关的一阶Sobol'指数估计量，用于量化某个输入对系统性响应变异性的贡献；另一个是新的基于触发器的结构指数，用于量化预测性能如何依赖于某个预测变量在不同备选预测变量子集中的可用性。我们在输入独立的条件下建立了两个估计量的一致性，以及结构指数的变量选择性质。蒙特卡洛实验和NHANES数据应用（摘要在此处被截断）。

    arXiv:2609.40342v1 Announce Type: cross  Abstract: Classical variance-based Global Sensitivity Analysis (GSA) assumes that the input--output mechanism can be repeatedly evaluated under a designed sampling scheme, which is infeasible when only a given sample of observations is available. We propose MM--GSA, a metamodel-based approach to GSA from observational data, in which supervised learning approximates the systematic input--output relationship. MM--GSA combines two complementary perspectives on input relevance: a model-agnostic estimator of the first-order Sobol' index, quantifying the contribution of an input to the variability of the systematic response, and a new trigger-based structural index, quantifying how predictive performance depends on the availability of a predictor across alternative predictor subsets. We establish consistency for both estimators and a variable-selection property for the structural index under input independence. Monte Carlo experiments and an NHANES ap
    
[^2]: 乘积有向无环图上的信号处理：因果平移与滤波器

    Signal Processing over Product DAGs: Causal Shifts and Filters

    [https://arxiv.org/abs/2609.40275](https://arxiv.org/abs/2609.40275)

    本文提出了一种新的有向无环图（DAG）乘积算子，使得乘积DAG上的结构方程模型、傅里叶模态、因果平移和滤波器在两个图因子之间均可分离，从而为双域线性因果结构构建了高效的信号处理框架。

    

    我们开发了一个信号处理框架，用于处理由两个有向无环图（DAG）的乘积所索引、并由线性结构方程模型（SEM）描述的信号。这种设置出现在（线性）因果关系沿两个域起作用的情形中，例如组件与制造阶段之间，或基因与实验条件之间。若忽略底层图的可分解性及其固有的双轴因果结构，则在进行傅里叶分析时需要对一个加权传递闭包矩阵求逆，而该矩阵的规模是两个因子规模的乘积。我们注意到标准的图乘积无法产生可分解的传递闭包，因此引入了一种新的DAG乘积，在该乘积下可分离性得以成立。我们在顶点域中对该新算子进行了论证，并证明它使得乘积DAG上的结构方程模型（SEM）、傅里叶模态、因果平移以及滤波器在其构成图因子之间均可分离。

    arXiv:2609.40275v1 Announce Type: new  Abstract: We develop a signal processing framework for signals indexed by the product of two directed acyclic graphs (DAGs) and described by a linear structural equation model (SEM). Such a setup arises whenever (linear) causal relations act along two domains, as in component versus manufacturing stage or gene versus experimental condition. Disregarding the factorization of the underlying graph and the native two-axis causal structure requires inverting a weighted transitive closure matrix whose size is the product of the two factor sizes for Fourier analysis. Recognizing that standard graph products fail to yield factorizable transitive closures, we introduce a new DAG product under which separability holds. We motivate the new operator in the vertex domain, and show that it also renders the SEM, the Fourier modes, the causal shifts, and the filters on the product DAG separable across its constituent graph factors.
    
[^3]: 面向连续扩散语言模型的分布匹配蒸馏

    Distribution Matching Distillation for Continuous Diffusion Language Models

    [https://arxiv.org/abs/2609.40235](https://arxiv.org/abs/2609.40235)

    提出了两种分布匹配蒸馏方法Simplex-DMD和Reinforce-DMD，通过利用学生模型的概率性词元输出，将连续扩散语言模型的网络评估次数大幅降低至仅需4次即可实现高质量文本生成。

    

    连续扩散语言模型可以并行生成所有词元，但高质量的生成仍可能需要数百次网络评估（NFEs）。我们研究了如何通过分布蒸馏来降低这一成本，并利用学生模型的概率性词元输出。我们的统一公式将学生模型的输出参数化与由此产生的梯度估计器联系起来，从而得到两种采用相同学生架构和反向KL匹配目标的方法：Simplex-DMD使用连续词元松弛和路径梯度，而Reinforce-DMD使用类别采样以及带有可学习密度比的REINFORCE方法。我们为多步生成开发了这两种方法，并研究了与每种参数化相关的训练和采样选择。在OpenWebText数据集上，对于1,024个词元的序列，Simplex-DMD仅用4次NFEs就在5.44 nats的一元熵下实现了45.6的生成困惑度，相比基线降低了49%。

    arXiv:2609.40235v1 Announce Type: cross  Abstract: Continuous diffusion language models generate all tokens in parallel, yet high-quality generation can still require hundreds of network evaluations (NFEs). We study how distributional distillation can reduce this cost by exploiting the student's probabilistic token outputs. Our unified formulation connects the student's output parameterization to the resulting gradient estimators and yields two methods with the same student architecture and reverse-KL matching objective: Simplex-DMD uses continuous token relaxations and pathwise gradients, while Reinforce-DMD uses categorical sampling and REINFORCE with a learned density ratio. We develop both methods for multi-step generation and investigate the training and sampling choices associated with each parameterization. On OpenWebText, for sequences of 1,024 tokens, Simplex-DMD achieves a generative perplexity of 45.6 at a unigram entropy of 5.44 nats in just 4 NFEs, a 49% reduction relative
    
[^4]: 绘制廉价，信任昂贵：为测试时扩展曲线提供统计认证

    Cheap to Draw, Expensive to Trust: Certifying Test-Time Scaling Curves

    [https://arxiv.org/abs/2609.40190](https://arxiv.org/abs/2609.40190)

    该论文推导了统计认证整条测试时扩展曲线的极小极大采样成本，并证明利用“基准是固定题目列表、方差主要来自题目之间”这一结构，可以避免为错误的不确定性买单，从而大幅降低同时认证所有预算所需的生成样本量。

    

    采样多个答案并保留验证器评分最高的那一个，是在测试时换取准确率的最简单方法之一。其效果通常以缩放曲线的形式报告：即准确率随采样答案数量 $k$ 变化的曲线。这条曲线画起来便宜，信起来却很贵。从曲线上读出的预算是在查看所有数据点之后才选定的，因此只有一条能同时覆盖所有预算的置信带才能保护这一选择；在一个包含100道题的基准上，固定的精确二项式设计需要生成192,000个答案，才能以95%的置信度将64个预算认证到 ±1/32 的精度。而这些成本的大部分，实际上为错误的不确定性买了单。基准测试是一个固定的问题列表；在预算为64时，所选答案正确性的方差约四分之三来自题目之间的差异，而一个会重新审视每道题的审计无需为此支付代价。我们推导出了认证整条曲线的极小极大成本（在对数因子意义下）。该成本由三部分组成：校准验证器评分分布的尾部……（摘要原文在此处截断）

    arXiv:2609.40190v1 Announce Type: cross  Abstract: Sampling several answers and keeping the one a verifier scores highest is one of the simplest ways to buy accuracy at test time. Its effect is reported as a scaling curve: accuracy against the number $k$ of sampled answers. The curve is cheap to draw and expensive to trust. A budget read off it is chosen after looking at every point, so only a band that covers all budgets at once protects the choice, and on a 100-question benchmark a fixed exact-binomial design needs 192,000 generated answers to certify 64 budgets to within $\pm1/32$ at 95%. Most of that cost pays for the wrong uncertainty. A benchmark is a fixed list of questions; at budget 64, about three quarters of the variance of a selected answer's correctness lies between questions, and an audit that revisits every question need not pay for it. We derive the minimax cost of certifying the whole curve, up to logarithmic factors. It has three parts: calibrating the tail of the sco
    
[^5]: 熵正则化最优传输下的部分识别

    Partial identification with entropy regularized optimal transport

    [https://arxiv.org/abs/2609.40156](https://arxiv.org/abs/2609.40156)

    该论文将统计部分识别中的优化问题统一表述为路径空间上的最优传输问题，通过熵正则化将其转化为可用Sinkhorn迭代高效求解的多边际熵正则化最优传输问题，并建立了正则化值向最锐利边界的收敛性及插件估计量的一致性收敛速率。

    

    在许多统计场景中，可用的数据和所维持的假设不足以唯一识别所关心的模型参数。在这类情况下，人们只能识别保证包含真实参数的集合，这些集合通常通过线性规划来刻画，即在观测数据相容的模型上进行优化。这些规划在优化变量和约束数量上都可能是无限维的。我们提供了一种统一的方式来刻画并求解此类优化问题：将其表述为路径空间上的最优传输问题。这使得我们能够用熵惩罚对问题进行正则化，将其重新表述为多边际熵正则化最优传输问题，从而可以通过Sinkhorn迭代高效求解。此外，该框架使我们能够建立正则化后的值向最锐利边界的收敛性，推导出插件估计量的一致性收敛速率，并获得渐近性质（原文摘要在此处截断）。

    arXiv:2609.40156v1 Announce Type: new  Abstract: In many statistical settings, the available data and maintained assumptions do not suffice to uniquely identify the model parameters of interest. In such cases, one can only identify sets which are guaranteed to contain the true parameters. These are often characterized through linear programs that optimize over models compatible with the observed data. These programs can be infinite-dimensional in the optimizer and the number of constraints. We provide a unified way to characterize and solve such optimization problems by phrasing them as optimal transport problems on path spaces. This allows us to regularize the problem with an entropy penalty, recasting it as a multi-marginal entropic optimal transport problem, which can be solved efficiently via Sinkhorn iterations. In addition, it allows us to establish convergence of the regularized value to the sharpest bound, derive consistency rates for a plug-in estimator, and obtain asymptotic 
    
[^6]: 面向未测量混杂下因果效应估计的邻近平衡方法

    Proximal Balancing for Causal Effect Estimation under Unmeasured Confounding

    [https://arxiv.org/abs/2609.40051](https://arxiv.org/abs/2609.40051)

    提出邻近平衡方法，将经典协变量平衡思想扩展到仅能通过代理变量观测的混杂因素，无需指定代理角色、求解逆问题或依赖潜变量模型，即可实现未测量混杂下的因果效应估计。

    

    从观测数据中估计因果效应是科学与政策领域的核心问题，但当混杂因素未被测量时，因果效应无法被识别。邻近因果推断通过未测量混杂因素的代理变量来应对这一问题。然而，现有的基于代理的方法要么指定代理角色并求解一个逆问题，该问题是不适定的，且在高维代理变量下难以估计；要么使用潜变量模型，该模型假设学习到的潜变量与隐藏的混杂因素相匹配，一旦不匹配就会产生偏差。为了应对这些挑战，我们提出了邻近平衡方法。它将经典的协变量平衡思想推广到只能通过代理变量观测的混杂因素：它学习协变量和代理变量的一个低维摘要表示，使处理组之间具有可比性，然后对该摘要进行调整。该方法不需要指定代理角色、求解逆问题或使用潜变量模型。我们给出（原文摘要在此处截断）

    arXiv:2609.40051v1 Announce Type: new  Abstract: Estimating causal effects from observational data is central to science and policy, but the effects are not identified when confounders are unmeasured. Proximal causal inference addresses this problem with proxies of the unmeasured confounders. However, existing proxy-based approaches either designate proxy roles and solve an inverse problem, which is ill-posed and hard to estimate with high-dimensional proxies, or use a latent-variable model, which assumes that the learned latent variable matches the hidden confounder and leaves bias when it does not. To address these challenges, we introduce proximal balancing. It carries the classical idea of covariate balancing to confounders that are observed only through proxies: it learns a low-dimensional summary of the covariates and proxies that makes the treatment groups comparable, and then adjusts for this summary. It needs no designated proxy roles, inverse problem, or latent model. We give
    
[^7]: 面向归纳式多视图嵌入的Gromov-Wasserstein蒸馏

    Gromov-Wasserstein Distillation for Inductive Multi-View Embedding

    [https://arxiv.org/abs/2609.40047](https://arxiv.org/abs/2609.40047)

    该论文提出了一种基于重心蒸馏的归纳式框架，通过GW-MDS教师模型与神经学生网络的蒸馏，实现了对未见样本的显式多视图嵌入映射，并避免了推理时额外的GW优化计算。

    

    Gromov-Wasserstein多维缩放（GW-MDS）能够从关系数据中学习低维表示，但该方法仍是转导式的，无法为未见样本提供显式的映射。我们提出了一种基于重心蒸馏的归纳式框架。GW-MDS教师模型从训练数据中学习潜在支撑集和最优传输计划，随后重心投影将所得的耦合转换为与样本对齐的目标。接着，一个神经学生网络学习显式的样本外映射，从而避免了推理阶段额外的关系矩阵构建和GW优化。我们将该方法在单视图数据上进行形式化，并通过由具有视图特定编码器的多视图学生网络学习的一致性目标和选择投影目标，将其扩展至Mean-GWMDS和Multi-GWMDS教师模型。我们还研究了一种仅使用GW目标训练的直接神经基线方法。在合成数据和真实数据上使用欧氏距离、地理（摘要在此处被截断）……的实验

    arXiv:2609.40047v1 Announce Type: cross  Abstract: Gromov-Wasserstein multidimensional scaling (GW-MDS) learns low-dimensional representations from relational data but remains transductive, providing no explicit mapping for unseen samples. We introduce an inductive framework based on barycentric distillation. A GW-MDS teacher learns a latent support and an optimal transport plan from the training data, and barycentric projection converts the resulting coupling into sample-aligned targets. A neural student then learns an explicit out-of-sample mapping, avoiding additional relational-matrix construction and GW optimization at inference. We formulate the approach for single-view data and extend it to Mean-GWMDS and Multi-GWMDS teachers through consensus and selected-projection targets learned by a multi-view student with view-specific encoders. We also investigate a direct neural baseline trained solely with a GW objective. Experiments on synthetic and real-world data using Euclidean, geo
    
[^8]: 基于偏差探针的多群体公平性高效主动审计

    Efficient Active Auditing of Multi-Group Fairness with Bias Probes

    [https://arxiv.org/abs/2609.40034](https://arxiv.org/abs/2609.40034)

    提出偏差探针框架，通过有针对性的自适应查询高效审计多群体公平性，在保持模型机密性的同时揭示数据分布中驱动偏差的结构。

    

    过去十年中，机器学习（ML）一直在双重目标下进行训练：通过经验风险最小化（ERM）最小化预测误差，同时控制不公平偏差。然而在实践中，公平性感知的训练相对于标准ERM往往只能带来有限的改进，这使得可靠的事后审计变得至关重要。现有的黑盒模型审计方法要么依赖于模型重构（使系统面临提取攻击的风险），要么直接估计公平性指标，无法深入洞察数据分布中哪些区域驱动了偏差。更根本的是，针对特定属性的审计——旨在在不重构模型的情况下仅提取针对性的公平性信息——仍然缺乏深入理解。在本工作中，我们引入了偏差探针框架，该框架能够进行有针对性和自适应的查询，在揭示偏差结构的同时保持模型的机密性。基于该框架，我们提出了ALe

    arXiv:2609.40034v1 Announce Type: cross  Abstract: Over the past decade, Machine Learning (ML) has been trained under dual objectives: minimizing prediction error via Empirical Risk Minimization (ERM) while controlling unfairness bias. In practice, however, fairness-aware training often yields limited improvements over standard ERM, making reliable post hoc auditing essential. Existing auditing approaches for black-box models either rely on model reconstruction --exposing systems to extraction attacks-- or directly estimate fairness metrics, offering limited insight into which regions of the data distribution drive bias. More fundamentally, property-specific auditing --aimed at extracting only targeted fairness information without reconstructing the model-- remains poorly understood. In this work, we introduce the bias probe framework, which enables targeted and adaptive querying to reveal bias structure while preserving model confidentiality. Building on this framework, we propose ALe
    
[^9]: 任意结构多层模型的摊销贝叶斯推断

    Amortized Bayesian Inference on Multilevel Models of Arbitrary Structure

    [https://arxiv.org/abs/2609.40024](https://arxiv.org/abs/2609.40024)

    提出了一种基于图操作（图扩展与图反转）的通用摊销贝叶斯推断方法，能够自动为任意结构的多层模型推导有效的后验因式分解及匹配的神经网络架构，在完整保留生成模型依赖假设的同时，将训练后的推断简化为近即时的前向计算。

    

    我们开发了一种针对任意结构多层模型的摊销贝叶斯推断通用方法。给定一个以有向无环图形式指定的生成模型，该方法可自动推导出联合后验的有效因式分解以及与之匹配的神经网络架构。其中的关键步骤——图扩展与图反转——会生成一个反向图，该反向图决定了推断网络的堆叠与条件化方式，从而得到能够对组数以及每组内观测数量进行摊销的因式分解。与那些通过简化依赖结构来加速学习或推断的方法不同，我们的方法完整保留了生成模型的所有条件独立性与可交换性假设。在三个案例研究中，该方法在参数量超过6,500的模型上与金标准采样器结果高度吻合，且一旦训练完成，推断即可简化为近乎瞬时的一次前向传播。

    arXiv:2609.40024v1 Announce Type: new  Abstract: We develop a general method for amortized Bayesian inference on multilevel models of arbitrary structure. Given a generative model specified as a directed acyclic graph, our method automatically derives valid factorizations of the joint posterior and matching neural network architectures. The key steps, graph expansion and graph inversion, yield an inverse graph that determines how inference networks are stacked and conditioned, producing factorizations that amortize over the number of groups and the number of observations within each group. Unlike approaches that simplify the dependency structure to speed up learning or inference, our method preserves all conditional independence and exchangeability assumptions of the generative model. Across three case studies, it closely matches gold-standard samplers on models with more than 6,500 parameters while reducing inference to a near-instant forward pass once trained.
    
[^10]: BayesNDE：用于神经密度估计的贝叶斯生成建模

    BayesNDE: Bayesian Generative Modeling for Neural Density Estimation

    [https://arxiv.org/abs/2609.39843](https://arxiv.org/abs/2609.39843)

    提出了一种基于贝叶斯生成建模的神经密度估计器BayesNDE，无需可逆网络或雅可比行列式计算，通过为每个观测推断样本特定的潜在后验构建自适应提案并结合桥接采样来估计密度，在密度估计精度和异常检测性能上均超越了现有最先进方法。

    

    密度估计是统计学和机器学习中的一个基本问题。在这项工作中，我们提出了BayesNDE，一种基于贝叶斯生成建模的神经密度估计器。BayesNDE学习一个贝叶斯生成模型，并在无需可逆网络或雅可比行列式计算的情况下评估其密度。对于每个观测值，它推断出一个样本特定的潜在后验分布，以构建自适应提案分布，将计算集中在对该观测密度贡献最大的区域。随后，桥接采样将该提案分布的样本与独立采样的后验样本相结合来估计密度。在非线性和多峰合成数据集上的实验表明，与最先进的神经密度估计器相比，该方法改进了密度值的估计并更好地恢复了密度结构。在真实世界数据集上的应用进一步证明了其在异常检测方面的提升。这些结果共同凸显了BayesNDE作为一种有效密度估计方法的价值。

    arXiv:2609.39843v1 Announce Type: cross  Abstract: Density estimation is a fundamental problem in statistics and machine learning. In this work, we introduce BayesNDE, a neural density estimator based on Bayesian generative modeling. BayesNDE learns a Bayesian generative model and evaluates its density without requiring invertible networks or Jacobian-determinant computation. For each observation, it infers a sample-specific latent posterior to construct an adaptive proposal that focuses computation on regions contributing most to its density. Bridge sampling then combines samples from this proposal with separate posterior samples to estimate the density. Experiments on nonlinear and multimodal synthetic datasets show improved estimation of density values and better recovery of the density structure compared to the state-of-the-art neural density estimators. Applications to real-world datasets further demonstrate improved anomaly detection. Together, these results highlight BayesNDE as
    
[^11]: 基于选择性分类的带性能保证的标签噪声转移矩阵估计

    Estimation of the Label-Noise Transition Matrix with Performance Guarantees via Selective Classification

    [https://arxiv.org/abs/2609.39829](https://arxiv.org/abs/2609.39829)

    该论文提出了一种基于单侧选择性分类的标签噪声转移矩阵估计新方法，绕过了脆弱的类后验概率估计，提供了有限样本性能保证，并利用灵活的二元分类学习方法及有效算法加以实现。

    

    现代机器学习高度依赖大规模数据集，但大规模获取高质量标注往往成本高昂。因此，从带噪声标签的数据中学习已变得十分普遍，这使得准确估计标签噪声转移矩阵变得至关重要。然而，现有的转移矩阵估计方法依赖于脆弱的类后验概率估计，且无法提供有限样本性能保证。在这项工作中，我们提出了一种基于单侧选择性分类来估计转移矩阵的新方法。该方法绕过了类后验概率估计，提供了有限样本性能保证，并利用了灵活的二元分类学习方法。此外，我们引入了实现该方法论的有效算法，并给出了其精细的有限样本性能界。

    arXiv:2609.39829v1 Announce Type: new  Abstract: Modern machine learning depends heavily on massive datasets, but obtaining high-quality annotations at scale is often expensive. As a result, learning from noisily-labeled data has become common, making accurate estimation of the label-noise transition matrix crucial. However, existing transition matrix estimators rely on the fragile estimation of class-posteriors and do not provide finite-sample performance guarantees. In this work, we propose a novel methodology to estimate the transition matrix based on one-sided selective classification. This approach bypasses class-posterior estimation, provides finite-sample performance guarantees, and leverages flexible learning methods for binary classification. Moreover, we introduce effective algorithms to implement the proposed methodology and provide their refined finite-sample performance bounds.
    
[^12]: 概率对抗训练

    Probabilistic Adversarial Training

    [https://arxiv.org/abs/2609.39798](https://arxiv.org/abs/2609.39798)

    本文提出概率对抗训练，从两个分布重叠的概率视角出发，证明基于KL散度的下界可作为概率鲁棒性的可处理代理目标，最大化该下界即可恢复出对抗训练的缩放形式。

    

    基于一个概率视角——即对抗样本源于基于距离的分布 $p_{\mathrm{dis}}$ 与由受害分类器诱导的分布 $p_{\mathrm{vic}}$ 之间的重叠——我们从一个简单的直觉出发：当这两个分布被相互推开时，它们的重叠变小，对抗样本就变得更难生成，从而提升鲁棒性。这一直觉自然地启发了基于KL散度的鲁棒性目标。我们随后证明，$\mathrm{KL}(p_{\mathrm{dis}}\|p_{\mathrm{vic}})-\log Z_{\mathrm{vic}}$ 是概率鲁棒性（PR）的一个下界，其中 $Z_{\mathrm{vic}}$ 表示 $p_{\mathrm{vic}}$ 的归一化常数。由于概率鲁棒性通常难以直接计算，最大化这个基于KL散度的下界为提升概率鲁棒性提供了一个可处理的代理目标。我们进一步证明，该目标可恢复出对抗训练的一种缩放形式，从而提供了一个概率视角的解释……

    arXiv:2609.39798v1 Announce Type: cross  Abstract: Building on a probabilistic perspective in which adversarial examples arise from the overlap between a distance-based distribution $p_{\mathrm{dis}}$ and a victim-classifier-induced distribution $p_{\mathrm{vic}}$, we start from a simple intuition: adversarial examples become harder to generate when these two distributions are pushed apart, as their overlap becomes smaller, thereby increasing robustness. This intuition naturally motivates a KL-based robustness objective. We then prove that $\mathrm{KL}(p_{\mathrm{dis}}\|p_{\mathrm{vic}})-\log Z_{\mathrm{vic}}$ is a lower bound on probabilistic robustness (PR), where $Z_{\mathrm{vic}}$ denotes the normalizing constant of $p_{\mathrm{vic}}$. Since PR is generally intractable to compute directly, maximizing this KL-based lower bound provides a tractable surrogate objective for improving PR. We further show that this objective recovers a scaled form of adversarial training, offering a prob
    
[^13]: 针对难解似然的摊销比率估计重要性采样与局部化基于模拟的校准

    Amortized ratio-estimation importance sampling and localized simulation-based calibration for intractable likelihoods

    [https://arxiv.org/abs/2609.39712](https://arxiv.org/abs/2609.39712)

    该论文提出了针对难解似然模型的顺序化后验估计方法与局部化模拟校准技术，利用局部化条件密度代理和摊销化的似然-证据比率估计器，以低计算成本实现准确且经过校准的贝叶斯推断。

    

    我们考虑对具有难解似然但具备可操作前向模拟的模型参数进行基于模拟的贝叶斯推断。在专家混合高斯代理模型的基础上，作为第一个贡献，我们开发了一种用于后验估计的顺序化程序，其中使用逐步局部化的、由数据引导的条件密度近似作为提议分布。随后，通过重要性采样步骤对最终的后验分布代理进行校正，该步骤引入了一个局部摊销化的似然-证据比率估计器。第二个贡献是局部化的基于模拟的校准。局部化SBC在刻意设置得比可用后验更宽的邻域上进行校准探测，且额外计算成本很低。局部化SBC无需为每个模拟的伪观测数据重复运行完整的推断流程，而是拟合局部代理模型和比率估

    arXiv:2609.39712v1 Announce Type: cross  Abstract: We consider simulation-based Bayesian inference (SBI) for the parameters of models with intractable likelihoods but tractable forward simulation. Building on Gaussian mixtures-of-experts surrogates, as a first contribution we develop a sequential procedure for posterior estimation in which progressively localized, data informed, conditional density approximations are used as proposal distributions. A final surrogate of the posterior distribution is then corrected using an importance sampling step that introduces a locally amortized likelihood-to-evidence ratio estimator. A second contribution is localized simulation-based calibration (SBC). Localized SBC probe calibration over neighborhoods that are deliberately broader than the available posterior, at low additional computational cost. Rather than repeatedly running the full inference procedure for each simulated pseudo-observation, localized SBC fits the local surrogate and ratio est
    
[^14]: BAM！贝叶斯万物模型：面向生成式计算成像的基础模型

    BAM! Bayesian Anything Model: a foundation model for generative computational imaging

    [https://arxiv.org/abs/2609.39660](https://arxiv.org/abs/2609.39660)

    本文提出仅含3600万参数的轻量级基础模型BAM，通过将RAM骨干升级为条件流映射，使仪器物理特性可在推理时灵活指定，从而实现少步数的物理感知后验采样，并能零样本或仅需极少微调即可泛化到未见过的数据和任务。

    

    生成式模型正在变革贝叶斯计算成像，但该领域仍缺乏具备物理感知能力的基础模型。目前的实践分为两大阵营：大型基础图像模型被作为即插即用的先验，配合零样本的近似似然引导，这会引入显著的偏差和巨大的计算成本；而物理感知的生成式模型虽然避免了这种偏差，但每个模型都被绑定于特定的数据集、任务和仪器。我们提出了BAM（贝叶斯万物模型），这是一个轻量级的基础模型，用于少步数的物理感知后验采样，能够稳健地泛化到未见过的数据和任务，实现零样本或仅需极少微调即可使用。BAM将算子条件化的Reconstruct Anything Model（RAM）骨干网络（Terris等人）升级为条件流映射，使仪器物理特性在推理时指定，而非在训练时固定。BAM仅有3600万参数，并在大型图像语料库上进行联合预训练。

    arXiv:2609.39660v1 Announce Type: cross  Abstract: Generative models are transforming Bayesian computational imaging, yet the field still lacks physics-aware foundation models. Current practice falls into two camps. Large foundation image models are deployed as plug-and-play priors with zero-shot approximate likelihood guidance, which introduces significant bias and computational cost. Physics-aware generative models avoid this bias, but each is tied to a specific dataset, task and instrument. We introduce BAM (Bayesian Anything Model), a lightweight foundation model for few-step, physics-aware posterior sampling that generalises robustly to unseen data and tasks, zero-shot or with minimal finetuning. BAM upgrades the operator-conditioned Reconstruct Anything Model (RAM) backbone (Terris et al.) into a conditional flow map, so instrument physics is specified at inference time rather than fixed during training. BAM has just 36M parameters and is pre-trained jointly on large image corpor
    
[^15]: 通过辅助监督缓解摊销贝叶斯推断中的表示差距

    Mitigating Representation Gaps in Amortized Bayesian Inference with Auxiliary Supervision

    [https://arxiv.org/abs/2609.39525](https://arxiv.org/abs/2609.39525)

    提出一种通用的辅助监督方法，通过对神经网络内部表示施加引导损失来缓解摊销贝叶斯推断中的表示差距，在数据充足时加快收敛，在数据稀缺时提升推断性能。

    

    将贝叶斯推断转化为以摊销后验为目标的神经网络优化问题是很有吸引力的，因为它可以扩展到原本难以处理的统计模型，并且在预付训练成本之后，能够对新数据集提供近乎即时的推断。尽管理论上保证在理想收敛条件下推断的忠实性，但实际的摊销推断仍然需要反复尝试不同的架构和优化选择，并最终在有限的仿真、算力和时间预算下进行“满意解”取舍。因此，即使是表现最好的解决方案也可能保留本可避免的表示差距，而这通常需要针对具体问题的修复方法。在此，我们提出了一种通用的替代方案，通过对内部表示施加辅助引导损失来改善训练动态。具体而言，我们展示了这种引导方法如何在训练数据充足时带来更快的收敛速度，并在数据稀缺时带来更好的性能。我们形式化地定义了表示……（摘要在此处截断）

    arXiv:2609.39525v1 Announce Type: new  Abstract: Casting Bayesian inference as a neural network optimization problem targeting an amortized posterior is attractive, as it extends to otherwise intractable statistical models and offers near instantaneous inference for new datasets after prepaying the training cost. Although theory guarantees faithfulness under ideal convergence, practical amortized inference still requires iterating over architectures and optimization choices and ultimately ``satisficing'' under finite simulation, compute, and time budgets. Even the best-performing solution may thus retain avoidable representation gaps that typically require problem-specific fixes. Here, we propose a generic alternative which improves training dynamics with auxiliary guidance losses applied to internal representations. Specifically, we show how such guidance leads to faster convergence when training data is abundant and to better performance when it is scarce. We formalize representation
    
[^16]: 可行集上随机平滑的几何学

    The Geometry of Randomized Smoothing on Feasible Sets

    [https://arxiv.org/abs/2609.39497](https://arxiv.org/abs/2609.39497)

    该论文将可行性/置信度过滤下的随机平滑认证问题分解为几何与认证两部分，证明凸保留集能完整保持高斯比较，一般集合需要对保留分布的几何控制（否则无法认证任何正半径），而联合保留与标签概率总能给出有效认证。

    

    随机平滑在以某点为中心的高斯噪声中心移动时，认证固定输出事件的概率。可行性或置信度过滤仅在被保留的候选提议中报告标签概率，从而形成一个比值。该比值的分子是固定的高斯事件概率质量，而分母是被保留的概率，且会随中心的变化而变化。因此，将此比值代入普通的平滑公式，可能会认证出一个包含决策边界的球。我们将该问题分解为几何问题和认证问题两部分。几何问题决定了条件化何时保持高斯比较关系：凸的保留集可保持完整的比较关系，而一般集合则要求在中心移动时对保留后分布施加几何控制。在没有这种控制的情况下，条件概率无法给出任何正的通用半径认证。而联合保留与标签概率则总能对相同的过滤后预测给出有效的认证证书。

    arXiv:2609.39497v1 Announce Type: cross  Abstract: Randomized smoothing certifies the probability of a fixed output event as the center of Gaussian noise moves. Feasibility or confidence filtering reports label probabilities only among retained proposals, producing a ratio. Its numerator is a fixed Gaussian event mass, while its denominator is the probability of retention and can change with the center. Substituting this ratio into the ordinary smoothing formula can therefore certify a ball that contains a decision boundary. We separate the problem into a geometric question and a certification question. Geometry determines when conditioning preserves Gaussian comparisons. Convex retained sets preserve the full comparison, while general sets require geometric control of the retained law as the center moves. Without such control, conditional probabilities imply no positive universal radius. Joint retention-and-label probabilities always yield a valid certificate for the same filtered pre
    
[^17]: CAMOS：面向多模态临床时间序列的耦合振荡状态空间模型

    CAMOS: Coupled Oscillatory State-Space Model for Multimodal Clinical Time-Series

    [https://arxiv.org/abs/2609.39484](https://arxiv.org/abs/2609.39484)

    本文证明了线性状态空间模型用掩码处理缺失模态存在表征局限——无法建模模态联合存在或缺失的交互作用，并提出CAMOS模型，通过可用性门控的耦合二阶振荡器使转移算子本身随模态可用性变化，从而更好地建模多模态临床时间序列。

    

    纵向临床队列具有多模态、非规则采样且普遍不完整的特点：在ADNI数据集中，正电子发射断层扫描（PET）和脑脊液检测在大约一半的访视中缺失。线性状态空间模型能够优雅地处理非规则采样，但其通过掩码输入来处理缺失模态，而转移算子保持不变。我们证明这是一种表征能力上的局限：对于任何转移算子不依赖于可用性模式的线性状态空间层，其潜在状态都是可用性指示的加性函数，因此这类层无法表示两种模态同时存在或同时缺失之间的交互作用。我们提出CAMOS，为每个模态配备一组二阶振荡器，通过一个位于微分方程内部并由可用性门控的矩阵实现耦合，从而使转移算子本身成为哪些测量可用的函数。

    arXiv:2609.39484v1 Announce Type: cross  Abstract: Longitudinal clinical cohorts are multimodal, irregularly sampled and pervasively incomplete: in ADNI, positron emission tomography and cerebrospinal fluid assays are absent from roughly half of all visits. Linear state-space models handle irregular sampling gracefully but treat a missing modality by masking the input, leaving the transition operator untouched. We prove that this is a representational limitation: the latent state of any linear state-space layer whose transition operator does not depend on the availability pattern is an additive function of the availability indicators, so no such layer can represent an interaction between two modalities being jointly present or jointly absent. We propose CAMOS, which gives each modality a bank of second-order oscillators coupled through a matrix that sits inside the differential equation and is gated by availability, so the transition operator itself becomes a function of which measurem
    
[^18]: 通过对抗训练视角研究分布鲁棒线性回归

    Distributionally robust linear regression through the lens of adversarial training

    [https://arxiv.org/abs/2609.39449](https://arxiv.org/abs/2609.39449)

    本文提出以 Wasserstein 分布鲁棒优化框架研究线性回归，统一了平方根 Lasso 与对抗线性回归这两个特例，并证明了该一般方法具有样本内误差界、对噪声水平的枢轴性以及解的等价性等重要性质。

    

    arXiv:2609.39449v1 公告类型：new 摘要：分布鲁棒优化（DRO）研究在潜在概率分布存在不确定性情况下的参数估计问题，并已成为分析鲁棒性与泛化能力的一个有原则的框架。特别地，以 Wasserstein 距离刻画分布不确定性的 Wasserstein DRO 推广了多种流行的正则化方法。本文研究 Wasserstein DRO 线性回归，将平方根 Lasso（square-root Lasso）与对抗线性回归统一为该方法的重要特例。我们证明了这两个特例的许多性质可以延续到这一一般方法上。具体而言，我们展示了：(i) 确定性与非渐近的样本内误差界，一般情形为 $O(n^{-1/2})$，在设计矩阵与稀疏性条件下为 $O(n^{-1})$；(ii) 对噪声水平不敏感，即所谓的枢轴性；(iii) 小模糊集与大模糊集情形下解的等价性。关键的证明步骤是将该方法重新表述为……（摘要原文在此处被截断）

    arXiv:2609.39449v1 Announce Type: new  Abstract: Distributionally robust optimization (DRO) studies parameter estimation under uncertainty in the underlying probability distribution and has emerged as a principled framework for analyzing robustness and generalization. In particular, Wasserstein DRO, with distributional uncertainty induced by the Wasserstein distance, generalizes several popular regularizers. This paper studies Wasserstein DRO linear regression, unifying square-root Lasso and adversarial linear regression as important special cases. We prove that many properties of these two special cases carry over to this general method. In particular, we show (i) deterministic and non-asymptotic in-sample error bounds $O(n^{-1/2})$ in general and $O(n^{-1})$ under design matrix and sparsity conditions; (ii) insensitivity to the noise level, also known as the pivotal property; and (iii) solution equivalences for small and large ambiguity sets. The key proof step is to recast the metho
    
[^19]: 原始路由的适配器混合：时间序列基础模型中路由坍缩的因果干预

    Raw-Routed Mixture of Adapters: A Causal Intervention for Routing Collapse in Time Series Foundation Models

    [https://arxiv.org/abs/2609.39445](https://arxiv.org/abs/2609.39445)

    该论文揭示了实例归一化会剥离路由器区分数据模式所需的统计信息、从而导致时间序列基础模型中混合专家发生路由坍缩，并据此提出原始路由的适配器混合作为因果干预方案，其信号比指标可在训练前预测数据集的脆弱性（Spearman ρ = -0.88）。

    

    时间序列基础模型（TSFM）通常通过在冻结的主干网络上附加单一可训练头部来适应新数据，这种“一刀切”的设置难以充分拟合异质的数据模式。用混合专家替换该头部是标准的升级方案，但在采用实例归一化的主干网络（TSFM 的主流设计类型）上却会失效：路由熵坍缩为零，单个专家吸收所有输入，我们将这种失效称为归一化诱导的路由坍缩。标准的 MoE 补救机制无法修复该问题，因为其根源在于路由器的输入而非其优化过程：编码器前的归一化剥离了路由器区分不同模式所需的统计信息。互信息分解使这一分析变得精确，并导出一个信号比指标，在训练前计算即可预测数据集的脆弱性（Spearman ρ = -0.88）。八项因果控制实验（包括在视觉模态上的重复验证）将实例归一化隔离为根本原因。

    arXiv:2609.39445v1 Announce Type: cross  Abstract: Time series foundation models (TSFMs) commonly adapt to new data by attaching a single trainable head to a frozen backbone, a one-size-fits-all setup that underfits heterogeneous regimes. Replacing the head with a mixture of experts is the standard upgrade, but on instance-normalized backbones (the dominant TSFM design class) it fails: routing entropy collapses to zero and one expert absorbs every input, a failure we call normalization-induced routing collapse. Standard MoE rescue mechanisms do not repair it, because the cause is in the router's input, not its optimization. Pre-encoder normalization strips the statistics a router would need to tell regimes apart. A mutual-information decomposition makes this precise and yields a signal-ratio that, computed before training, predicts dataset vulnerability (Spearman $\rho = -0.88$). Eight causal controls, including a vision-modality replication, isolate instance normalization as the cause
    
[^20]: 主成分回归在线性回归中优于所有单调谱滤波器

    Principal Component Regression Dominates all Monotone Spectral Filters for Linear Regression

    [https://arxiv.org/abs/2609.39440](https://arxiv.org/abs/2609.39440)

    本文证明经过最优调参的主成分回归（PCR）在逐实例有限样本风险意义下优于包括梯度下降和岭回归在内的所有单调谱滤波器，从而在单调滤波器类中达到最优且可容许。

    

    我们比较了线性回归中单调谱滤波器的逐实例有限样本风险，这是一类广泛的估计器，包括主成分回归（PCR）、梯度下降（GD）和岭回归。我们证明PCR优于所有单调谱滤波器：与任何此类滤波器相比，经过最优调参的PCR的风险在所有问题上都不会大过对方一个常数因子。此外，如果该滤波器与阶梯函数分离（例如GD和岭回归），则这种优势是强的：存在某些问题实例，使得PCR的风险在样本量依赖性上比它们小一个多项式因子。我们的比较结果表明PCR在单调滤波器中是最优的，因而是可容许的，这显著扩展了Wu等人（2026）关于GD强优于岭回归的结果。从技术角度来看，我们为一般谱滤波器建立了新的上界和下界，当特化到岭回归等情形时这些界在逐实例意义上是紧的。

    arXiv:2609.39440v1 Announce Type: new  Abstract: We compare the instance-wise, finite-sample risks of monotone spectral filters for linear regression, a broad class of estimators including principal component regression (PCR), gradient descent (GD), and ridge regression. We show that PCR dominates all monotone spectral filters: compared to any such filter, the risk of optimally tuned PCR is no bigger by a constant factor for all problems. Furthermore, the dominance is strong if the filter is separated from step functions (e.g., GD and ridge): there exist problem instances for which the risk of PCR is smaller by a polynomial factor in sample size dependence. Our comparison results show that PCR is optimal and thus admissible among monotone filters, significantly extending Wu et al. (2026)'s result that GD strongly dominates ridge. From a technical perspective, we establish new upper and lower bounds for general spectral filters, which are instance-wise sharp when specialized to ridge or
    
[^21]: 佛陀的觉醒：总体损失平台期的子空间学习

    Awakening of the Buddha: Subspace Learning During Population-Loss Plateaus

    [https://arxiv.org/abs/2609.39408](https://arxiv.org/abs/2609.39408)

    该论文证明了神经网络在总体损失几乎不变的高损失平台期内仍能显著学习到更具预测性的表示——教师子空间与 AGOP 前导特征空间的最小对齐度相比初始化至少提升 1/2，且固定系数预算下的最小重拟合 MSE 下降超过 0.399。

    

    总体损失（population loss）可以保持几乎不变，而神经网络却在学习预测能力显著更强的表示。我们针对在高斯输入上训练的两层 ReLU 和 leaky-ReLU 网络，通过对所有参数进行同步的固定步长总体梯度下降，建立了这种分离现象。对于其链接函数为 $H^1(\gamma)$ 中高斯阻尼三次函数的正混合的结构化加性教师模型，我们给出了显式条件，在这些条件下，小的独立同分布高斯初始化可产生高概率保证：在高损失平台期的某个检查点处，秩为 $r$ 的教师子空间与预测器平均梯度外积（AGOP）的前 $r$ 维特征空间之间的最小对齐度相比初始化至少增加 $1/2$，且在系数预算不变的情况下，最小重拟合 MSE 相比初始化下降超过 $0.399$。同一条训练轨迹随后达到的训练损失低于……（原文摘要在此处被截断）

    arXiv:2609.39408v1 Announce Type: cross  Abstract: Population loss can remain nearly constant while a neural network learns a substantially more predictive representation. We establish this separation for two-layer ReLU and leaky-ReLU networks trained on Gaussian inputs by simultaneous fixed-step population gradient descent on all parameters. For structured additive teachers whose links are positive mixtures of Gaussian-damped cubics in $H^1(\gamma)$, we give explicit conditions under which small IID Gaussian initialization yields a high-probability guarantee: at a checkpoint during a high-loss plateau, minimum alignment between the rank-$r$ teacher subspace and the leading $r$-dimensional eigenspace of the predictor's average gradient outer product (AGOP) increases by at least $1/2$, and the minimum refit MSE under unchanged coefficient budgets decreases by more than $0.399$, both relative to initialization. The same trajectory subsequently attains a trained loss below every value in 
    
[^22]: 面向删失需求下的最优库存控制：一种有偏样本平均近似方法

    Towards Optimal Inventory Control under Censored Demand: A Biased Sample-Average Approximation Approach

    [https://arxiv.org/abs/2609.39397](https://arxiv.org/abs/2609.39397)

    该论文针对删失需求下的多周期缺货损失库存控制问题，提出了基于新成本分解与覆盖条件的有偏样本平均近似（SAA）统一框架，通过设计上偏和下偏两种算法，分别在离线场景下实现近乎最优的样本复杂度、在在线场景下实现近乎最优的遗憾界。

    

    我们研究删失需求下数据驱动的多周期缺货损失（lost-sales）库存控制问题，其中缺货仅能揭示需求超过了当时的库存水平。我们开发了一个统一的、基于模型的从删失数据中进行策略学习的框架，该框架建立在针对基本库存策略（base-stock policies）的新成本分解以及有偏样本平均近似（SAA）方法之上。该成本分解使我们能够提出一个新的覆盖条件（coverage condition），在该条件下删失观测包含足够的信息，可实现样本高效的策略学习。在该覆盖条件的指导下，我们设计了两种有偏SAA算法：一种是上偏算法，在离线覆盖条件下实现近乎最优的样本复杂度；另一种是下偏算法，能够主动生成所需的覆盖条件，并在在线情形下实现近乎最优的遗憾界。更广泛地说，这种有偏SAA方法为在删失反馈下实施悲观与乐观原则提供了一般性原理，这可能……

    arXiv:2609.39397v1 Announce Type: new  Abstract: We study data-driven multi-period lost-sales inventory control under censored demand, where a stockout reveals only that demand exceeded the stocking level. We develop a unified, model-based framework for policy learning from censored data, built on a new cost decomposition for base-stock policies and a biased sample-average approximation (SAA) approach. The cost decomposition allows us to propose a new coverage condition under which censored observations are informative enough for sample-efficient policy learning. Guided by this coverage condition, we design two biased SAA algorithms: an upper-biased one that achieves near-optimal sample complexity under the offline coverage condition, and a lower-biased one that actively generates the required coverage and achieves near-optimal regret online. More broadly, this biased SAA approach provides a general principle for implementing pessimism and optimism under censored feedback, which may be
    
[^23]: 持续学习中LoRA的动力学理论

    A Dynamical Theory of LoRA in Continual Learning

    [https://arxiv.org/abs/2609.39367](https://arxiv.org/abs/2609.39367)

    该论文在师生模型中建立了持续学习下LoRA的精确动力学理论，揭示低秩更新虽能减少对已学任务特征的干扰从而缓解灾难性遗忘，但其初始化会减慢对新任务的适应。

    

    尽管低秩适配被广泛使用，但人们对其在持续学习中的动力学行为，以及低秩更新影响灾难性遗忘的机制，仍然知之甚少。我们在一个可解的双任务师生模型中，为LoRA提供了渐近精确的动力学刻画。在高维在线学习极限下，我们针对有限个宏观序参量导出了一个封闭的常微分方程组，从而在整个初始任务1学习阶段以及随后在任务2上的LoRA微调过程中，都得到了泛化误差的精确表达式。该理论与有限维模拟定量吻合，并揭示了LoRA的两个特征效应：低秩适配减少了与任务1所学特征的干扰，但其初始化会减慢对任务2的适应。基于这一机理图景，我们进一步分析了一种状态相关的掩蔽方法……

    arXiv:2609.39367v1 Announce Type: new  Abstract: Despite the widespread use of Low-Rank Adaptation (LoRA), little is known about its dynamics in continual learning and the mechanisms by which low-rank updates affect catastrophic forgetting. We provide an asymptotically exact dynamical characterization of LoRA in a solvable two-task teacher-student model. In the high-dimensional online-learning limit, we derive a closed system of ordinary differential equations for a finite set of macroscopic order parameters, yielding exact expressions for the generalization errors throughout both the initial Task 1 learning phase and the subsequent LoRA fine-tuning on Task 2. The theory quantitatively matches finite-dimensional simulations and exposes two characteristic effects of LoRA: low-rank adaptation reduces interference with features learned on the first task, but its initialization slows adaptation to the second task. Building on this mechanistic picture, we analyze a state-dependent masking s
    
[^24]: 匹配问题的统一对偶方法

    A Unified Dual Method for Matching Problems

    [https://arxiv.org/abs/2609.39339](https://arxiv.org/abs/2609.39339)

    本文基于对偶理论提出了一个统一框架，将可分解为凸函数之差（DC）的匹配目标重构为隐式配准问题，并将其应用于Gromov-Wasserstein及其非平衡形式和变体等二次匹配问题，同时提供了广泛的收敛性保证。

    

    匹配问题在数据科学中无处不在，因为它们能够实现结构化对象与分布之间的对齐。尽管现有的求解器通常针对特定的匹配形式而定制，我们基于对偶理论将一大类此类问题统一在一个共同的数学与优化框架中。在理论方面，我们证明了可分解为凸函数之差（DC）的匹配目标可以重新表述为隐式配准问题。这一联系将匹配与另一类已被广泛研究的目标函数联系起来，并产生了一种易于采用自然优化策略的对偶形式。随后，我们将这些发现应用于具有DC分解的二次匹配（QM）问题，并为其提供了广泛的收敛性保证。我们的框架适用于Gromov-Wasserstein（GW）及其非平衡形式和若干变体，这些是日益流行的QM问题。在数值实验中，我们

    arXiv:2609.39339v1 Announce Type: cross  Abstract: Matching problems are ubiquitous in data science as they enable the alignment of structured objects and distributions. While existing solvers are often tailored to specific matching formulations, we unify a broad class of such problems within a common mathematical and optimization framework based on duality theory. Theoretically, we demonstrate that matching objectives decomposable as a difference of convex (DC) functions can be recast as implicit registration problems. This connection links matching to another well-studied class of objectives and yields a dual formulation amenable to natural optimization strategies. We then apply these findings to quadratic matching (QM) problems, which admit DC decompositions and for which we provide extensive convergence guarantees. Our framework applies to Gromov-Wasserstein (GW), as well as its unbalanced formulation and several variants, which are increasingly popular QM problems. Numerically, we
    
[^25]: 离散分数匹配使基于计数数据的因果发现成为可能

    Discrete Score Matching Enables Causal Discovery from Count Data

    [https://arxiv.org/abs/2609.39326](https://arxiv.org/abs/2609.39326)

    该论文提出条件曲率分数（CCS）与对角线外曲率分数（OCS），将基于分数匹配的因果发现方法从连续数据推广到计数数据，并证明零CCS可精确刻画半参数广义线性模型的条件形式，从而实现从计数数据中恢复因果有向无环图。

    

    计数数据对基于分数匹配的因果发现方法构成了挑战：导数不可用，而简单地用有限差分来替代导数通常不足以完成因果发现。我们通过以节点的取值为条件，推广了SCORE的常曲率准则（Rolland et al., 2022），得到了用于变量排序的条件曲率分数（CCS）。我们还通过对角线外曲率分数（OCS）扩展了基于曲率的父节点恢复方法，使得有向无环图（DAG）的恢复成为可能，这两种分数对于连续数据由分数函数构造，对于计数数据则由具体分数构造。在双变量情形下，零CCS精确刻画了一种半参数广义线性模型（GLM）条件形式，其中条件分布族无需事先指定，这与经典GLM不同。对于满足我们正则性条件的双变量半参数GLM DAG，规范参数的非线性是必要且充分的（原文此处截断）。

    arXiv:2609.39326v1 Announce Type: new  Abstract: Count data pose a challenge for score-matching-based causal discovery: derivatives are unavailable, and simply replacing them with finite differences does not generally suffice for causal discovery. We generalize SCORE's constant-curvature criterion (Rolland et al., 2022) by conditioning on the node's value, yielding the conditional curvature score (CCS) for ordering. We also extend curvature-based parent recovery through the off-diagonal curvature score (OCS), enabling directed acyclic graph (DAG) recovery with both scores constructed from score functions for continuous data and concrete scores for counts. In the bivariate setting, zero CCS exactly characterizes a semiparametric generalized linear model (GLM) conditional form in which the conditional family need not be specified in advance, unlike in classical GLMs. For bivariate semiparametric GLM DAGs under our regularity condition, canonical-parameter nonlinearity is necessary and su
    
[^26]: 未知相关设计下的极小极大可加回归

    Minimax Additive Regression under Unknown Dependent Designs

    [https://arxiv.org/abs/2609.39212](https://arxiv.org/abs/2609.39212)

    该论文研究了维度随样本量增长、设计分布可能非乘积且边缘密度未知情形下的可加回归问题，通过 Riesz 基构造与阈值化最小二乘估计，建立了已知或未知边缘密度下相互匹配的极小极大预测误差界，并揭示了密度正则性何时影响最优速率。

    

    我们研究了在 $[0,1]^d$ 上可能非乘积的随机设计下的可加回归问题，并允许维度 $d$ 随样本量 $n$ 增长。我们引入了耦合的光滑性类，分别控制边缘密度的正则性和密度加权的可加分量。为处理变量间的相关性，我们针对函数型方差分析（ANOVA）模型采用了 Riesz 基构造，并在联合密度的一致界条件下，建立了常数与维度无关的相容性界。我们构造了阈值化最小二乘估计量，并在适当的维度增长条件下，针对已知或未知边缘密度的预测问题建立了相互匹配的极小极大上下界。当边缘密度的光滑性至少不低于加权分量时，未知密度情形可以达到已知密度情形的极小极大速率；当密度不够光滑时，其正则性决定了极小极大速率。

    arXiv:2609.39212v1 Announce Type: new  Abstract: We study additive regression under a potentially non-product random design on $[0,1]^d$, allowing the dimension $d$ to grow with the sample size $n$. We introduce coupled smoothness classes that separately control the regularity of the marginal densities and the density-weighted additive components. To handle dependence, we adapt a Riesz-basis construction for functional ANOVA models and establish compatibility bounds with constants independent of the dimension under uniform bounds on the joint density. We construct thresholded least-squares estimators and establish matching minimax upper and lower bounds for prediction with known or unknown marginal densities, under suitable dimension-growth conditions. When the marginal densities are at least as smooth as the weighted components, the unknown-density problem attains the known-density minimax rate. When the densities are less smooth, their regularity determines the minimax rate over the 
    
[^27]: 尖峰模型下高维小样本设置中支持向量机的渐近性质

    Asymptotic Properties of Support Vector Machines in High-Dimension, Low-Sample-Size Settings under a Spiked Model

    [https://arxiv.org/abs/2609.39173](https://arxiv.org/abs/2609.39173)

    在尖峰模型下，HDLSS数据的几何表示不再成立，支持向量机不具有相合性（误分类率不趋于零），而偏差校正SVM也因偏差项本身需要修改而无法取得更优性能。

    

    本文研究尖峰模型下高维小样本（HDLSS）设置中支持向量机（SVM）的渐近性质。现有的HDLSS背景下SVM理论依赖于HDLSS数据的几何表示，该表示要求协方差矩阵的特征值不占主导地位。我们首先证明在尖峰模型下这种几何表示并不成立。我们证明HDLSS数据的Gram矩阵依分布收敛于一个随机矩阵，也就是说，HDLSS数据收敛到有限维空间中的随机构型，该空间的维度由尖峰的个数决定。我们证明SVM的误分类率不会趋于零，即SVM不满足相合性。我们还证明了偏差校正SVM（BC-SVM）在该设置下无法给出更优的性能，因为偏差项本身需要被修改……

    arXiv:2609.39173v1 Announce Type: new  Abstract: In this paper, we consider asymptotic properties of the support vector machine (SVM) in high-dimension, low-sample-size (HDLSS) settings under a spiked model. The existing theory of the SVM in the HDLSS context relies on the geometric representation of HDLSS data, which requires that the eigenvalues of the covariance matrices are not dominant. We first show that the geometric representation does not hold under the spiked model. We show that the Gram matrix of HDLSS data converges in distribution to a random matrix, namely, the HDLSS data converge to a random configuration in a finite-dimensional space whose dimension is given by the number of the spikes. We show that the misclassification rates of the SVM do not tend to zero, that is, the SVM does not hold the consistency property. We also show that the bias-corrected SVM (BC-SVM) does not give preferable performance in this setting because the bias term itself should be modified. In ord
    
[^28]: 最速引导：一种实用且有原则的流模型与扩散模型推理时对齐方法

    Steepest Guidance: A Practical and Principled Approach to Inference-Time Alignment of Flow and Diffusion-based Models

    [https://arxiv.org/abs/2609.39091](https://arxiv.org/abs/2609.39091)

    本文提出“最速引导”框架，将流模型和扩散模型的推理时对齐建模为概率测度空间中的序贯优化问题，通过最大化目标的局部改进来规避Doob h-变换最优引导估计的困难，提供了一种兼具理论支撑与实用性的推理时对齐方法。

    

    流模型和基于扩散的模型的推理时对齐对于实现灵活的生成建模至关重要。从理论上讲，Doob的$h$-变换为这一问题提供了优雅的解决方案，大多数现有方法都基于这一原理。然而在实践中，在推理时估计由Doob的$h$-变换导出的最优引导是具有挑战性的。为了解决这个问题，我们将推理时对齐视为概率测度空间中的序贯优化问题，并提出了一种名为“最速引导”的新型框架，该框架基于最大化目标的局部改进这一原则。我们对所提出的方法进行了理论分析，并通过大量实验证明了其有效性。

    arXiv:2609.39091v1 Announce Type: new  Abstract: Inference-time alignment of flow and diffusion-based models is critical for achieving flexible generative modeling. Theoretically, Doob's $h$-transform provides an elegant solution to this problem, and most existing methods are based on this principle. However, in practice, estimating the optimal guidance derived from Doob's $h$-transform at inference time is challenging. To deal with this issue, we regard inference-time alignment as a sequential optimization problem in the space of probability measures and propose a novel framework called *Steepest Guidance*, based on the principle of maximizing local improvement in the objective. We provide a theoretical analysis of the proposed method and demonstrate its effectiveness through extensive experiments.
    
[^29]: 一种用于算法公平性的秩毕业度量

    A Rank Graduation metric for Algorithmic fairness

    [https://arxiv.org/abs/2609.39025](https://arxiv.org/abs/2609.39025)

    本文提出一个基于秩的公平性评估框架（RGF 及其积分度量 AURGF），通过模型预测误差的分布来衡量算法公平性，将公平性评估与预测精度和可解释性相联系，并配有推断检验和特征移除程序以识别导致不公平的因素。

    

    arXiv:2609.39025v1 公告类型：交叉 摘要：在影响个体的算法决策（如信用评分）中，公平性评估通常依赖于在总体群体层面计算的均等性（parity）度量。此类度量可能无法揭示哪些个体遭受了不公平待遇，也无法揭示哪些解释性因素导致了不公平。在本文中，我们提出了一个基于秩的框架，通过模型预测误差的分布来评估公平性，从而将公平性评估与预测精度和可解释性联系起来。该框架结合了秩毕业公平性度量（Rank Graduation Fairness, RGF）、其积分度量 AURGF、中心化 Cramér–von Mises 置换检验，以及用于公平性可解释性的特征移除程序。我们使用逻辑回归、随机森林、梯度提升和多层感知机对该方法进行了评估。模拟研究表明，受保护群体的不平衡可能逆转描述性公平性比较的结论，而所提出的推断程序……

    arXiv:2609.39025v1 Announce Type: cross  Abstract: Fairness assessment in algorithmic decisions that affect individuals, such as credit scoring, often relies on parity measures calculated at the aggregate group level. Such measures may not reveal which individuals experience unfairness or which explanatory factors contribute to it. In this paper, we propose a rank-based framework that evaluates fairness through the distribution of model prediction errors, thereby linking fairness assessment with predictive accuracy and explainability. The framework combines Rank Graduation Fairness (RGF), its integrated measure AURGF, a centered Cramer--von Mises permutation test, and a feature removal procedure for fairness explainability.   We evaluate the methodology using logistic regression, random forest, gradient boosting, and a multilayer perceptron. The simulation study shows that protected-group imbalance can reverse descriptive fairness comparisons, whereas the proposed inferential procedure
    
[^30]: 深度ReLU神经网络学习谱Barron函数的极小极大速率

    Minimax rates for learning spectral Barron functions by deep ReLU neural networks

    [https://arxiv.org/abs/2609.39020](https://arxiv.org/abs/2609.39020)

    本文为深度ReLU神经网络逼近和学习谱Barron函数建立了新的逼近界，并证明了相应的极小极大最优学习速率 $n^{-\frac{d+2s}{2d+2s}}$。

    

    我们研究深度神经网络对谱Barron函数的逼近与学习能力。近期的研究表明，这些函数类可以被浅层神经网络高效逼近，而不会受到维度灾难的影响。我们通过为采用ReLU激活函数的深度网络提供新的逼近界，并建立学习这些函数类的极小极大速率，对这些已有结果进行了补充。具体而言，我们证明了光滑性指数为 $s>0$ 的 $d$ 维谱Barron函数可以被深度ReLU神经网络以 $\widetilde{\mathcal{O}} (S^{-\frac{1}{2}-\frac{s}{d}})$ 的逼近速率进行逼近，其中 $S$ 表示网络中非零参数的数量。利用这一逼近结果，我们进一步证明了深度ReLU神经网络在 $n$ 个训练样本下能够以 $n^{-\frac{d+2s}{2d+2s}}$ 的快速速率学习谱Barron函数。最后，我们证明这一收敛速率（摘要在此处截断）

    arXiv:2609.39020v1 Announce Type: new  Abstract: We study how well deep neural networks approximate and learn spectral Barron functions. Recent studies have shown that these function classes can be efficiently approximated by shallow neural networks without suffering from the curse of dimensionality. We complement these results by providing new approximation bounds for deep networks with ReLU activation and establishing the minimax rates for learning these function classes. Specifically, we show that $d$-dimensional spectral Barron functions with smoothness index $s>0$ can be approximated by deep ReLU neural networks with approximation rate $\widetilde{\mathcal{O}} (S^{-\frac{1}{2}-\frac{s}{d}})$, where $S$ denotes the number of nonzero parameters in the network. Using this approximation result, we further show that deep ReLU neural networks can learn spectral Barron functions in a fast rate $n^{-\frac{d+2s}{2d+2s}}$ with $n$ training samples. Finally, we prove that this convergence ra
    
[^31]: 关于图像分割中条件独立性假设的放宽

    On the Relaxation of Conditional Independence Assumption for Image Segmentation

    [https://arxiv.org/abs/2609.38930](https://arxiv.org/abs/2609.38930)

    该论文提出用空间局部化依赖（SLD）结构替代RankSEG方法中的条件独立性假设，并借助倒数矩近似与不动点优化策略克服计算瓶颈，从而有效捕获局部标签相关性并提升模糊、低对比度场景下的图像分割性能。

    

    在语义分割中，最近一类名为RankSEG的方法在推理阶段直接优化Dice/IoU分数，在不修改模型训练的情况下提高了与评估指标的一致性。尽管RankSEG在理论和实证上都取得了成功，但它依赖于限制性较强的条件独立性假设（CIA），该假设忽略了关键的标签相关性，因此在模糊或低对比度场景下会导致性能下降。然而，考虑完全的标签依赖在计算上是不可行的，需要 $\mathcal{O}(d^3)$ 的时间复杂度。为了解决这一问题，我们用空间局部化依赖（SLD）结构替代CIA，该结构在保持依赖模型可处理性的同时捕获局部标签相关性。我们进一步通过倒数矩近似并结合一种新颖的不动点优化策略克服了剩余的计算瓶颈，从而消除了穷举搜索。所提出的算法实现了高……（原文摘要至此截断）

    arXiv:2609.38930v1 Announce Type: cross  Abstract: In semantic segmentation, a recent line of RankSEG methods directly optimizes Dice/IoU scores at inference time, improving alignment with evaluation metrics without modifying model training. Despite its theoretical and empirical success, RankSEG relies on the restrictive Conditional Independence Assumption (CIA), which ignores crucial label correlations and therefore degrades performance in ambiguous or low-contrast scenarios. However, accounting for full label dependence is computationally prohibitive, requiring $\mathcal{O}(d^3)$ time. To address this, we replace the CIA with a Spatially Localized Dependence (SLD) structure that captures local label correlations while keeping the dependence model tractable. We further overcome the remaining computational bottleneck via a Reciprocal Moment Approximation coupled with a novel fixed-point optimization strategy that eliminates exhaustive search. The proposed algorithm achieves a highly pr
    
[^32]: 噪声潜在结构下的流匹配：超越精确低维支撑

    Flow Matching under Noisy Latent Structure: Beyond Exact Low-Dimensional Support

    [https://arxiv.org/abs/2609.38918](https://arxiv.org/abs/2609.38918)

    论文研究了噪声潜在生成模型下的流匹配，证明了其样本复杂度指数由潜在维度而非环境维度决定，突破了传统理论要求精确低维支撑的限制。

    

    流匹配（FM）学习一个速度场，其ODE将简单的源分布传输到目标分布。现有的有限样本理论主要处理环境空间正则性或精确支撑在低维集合上的数据。我们研究噪声潜在生成模型下的线性流匹配，其中低维Hölder映射被非退化的环境高斯噪声扰动，因此目标律尽管具有潜在结构，但仍是全维度的。我们构造了一个空间正则的ReLU速度类，并建立了非渐近高概率近似和估计界，其主要样本量指数由潜在维度而非环境维度决定，同时保持环境依赖和噪声依赖的显式性。固定的正目标噪声使得插值在整个时间区间上保持非退化。同样的空间正则性将通过传输ODE传播学习到的速度误差，产生相应的

    arXiv:2609.38918v1 Announce Type: cross  Abstract: Flow Matching (FM) learns a velocity field whose ODE transports a simple source distribution to a target law. Existing finite-sample theory largely treats ambient-space regularity or data supported exactly on low-dimensional sets. We study linear FM under a noisy latent-generator model, where a low-dimensional H\"older map is perturbed by nondegenerate ambient Gaussian noise, so the target law is full-dimensional despite its latent structure. We construct a spatially regular ReLU velocity class and establish non-asymptotic high-probability approximation and estimation bounds whose leading sample-size exponent is governed by the latent dimension rather than the ambient dimension, with ambient and noise dependence kept explicit. Fixed positive target noise keeps the interpolation nondegenerate over the full time interval. The same spatial regularity propagates the learned velocity error through the transport ODE, yielding a corresponding
    
[^33]: 使用任意维度机器学习对偏微分方程求解器进行热启动

    Warm-starting PDE solvers with any-dimensional machine learning

    [https://arxiv.org/abs/2609.38916](https://arxiv.org/abs/2609.38916)

    提出了基于偏微分方程与初始数据对称性的数学条件，使低维训练的机器学习PDE求解器能够零样本直接求解高维PDE，并在数据不满足对称性时提供有原则的热启动策略，从而在热传导方程、Burgers方程和可压缩Navier-Stokes方程上均提升了性能。

    

    任意维度机器学习模型，例如图神经网络（GNN），可以在不同大小和维度的输入上进行自然的训练与评估。受GNN可迁移性文献的启发，我们给出了数学条件，使得基于学习的偏微分方程（PDE）求解器可以在低维度上训练，并以零样本（zero-shot）方式直接应用于求解更高维度的PDE。这些条件基于偏微分方程和初始数据中的对称性。当方程满足对称性而数据不满足时——这是许多源自物理学的PDE所面临的情况——我们证明我们的理论提供了一种有原则的方法，可以用低维PDE求解器为高维PDE进行热启动。我们将该方法应用于热传导方程、Burgers方程以及可压缩Navier-Stokes方程，在零样本和常规训练模式下均提升了求解性能。

    arXiv:2609.38916v1 Announce Type: new  Abstract: Any-dimensional machine learning models, such as graph neural networks (GNNs), can be naturally trained and evaluated on inputs of different sizes and dimensions. Inspired by the GNN transferability literature, we show mathematical conditions under which a partial differential equation (PDE) learning-based solver can be trained in small dimensions and directly applied to solve a higher dimensional PDE in a zero-shot fashion. These conditions are based on symmetries in both the partial differential equation and the initial data. When the equations satisfy the symmetries but the data does not, which is the case for many PDEs arising from physics, we show that our theory gives a principled way of warm-starting low-dimensional PDE solvers for higher dimensional PDEs. We apply this method on the heat equation, Burgers' equation, and the compressible Navier--Stokes equations, improving the performance in both zero-shot and typical training reg
    
[^34]: 基于可交换性感知神经后验估计的摊销式数据借入

    Amortized Data Borrowing with Exchangeability-Aware Neural Posterior Estimation

    [https://arxiv.org/abs/2609.38902](https://arxiv.org/abs/2609.38902)

    本文提出一种可交换性感知的摊销式神经后验估计方法，通过在模拟的当前/外部数据集对上预训练单个网络，实现一次前向传播即可快速获得当前研究的近似后验，为传统依赖手工先验和MCMC的贝叶斯动态借入提供了高效且可泛化的替代方案。

    

    在药物研发中，受试者入组缓慢、随访成本高昂，而密切相关的试验数据或真实世界数据往往已经存在，因此利用外部或历史队列来扩充小规模的同期研究颇具吸引力。贝叶斯动态借入（BDB）为自适应地控制外部数据的影响提供了一个有原则的框架，但其经典实现通常依赖于手工指定的先验和基于MCMC的推断，计算开销大且难以泛化。在这项工作中，我们研究了摊销式神经后验估计（NPE）作为一种灵活的替代方案。我们在模拟的“当前/外部”数据集对上预训练单个网络，这些数据对涵盖了协变量偏移、结果漂移以及联合不可交换性等情形；随后该网络仅需一次前向传播，即可为标量形式的当前研究目标返回近似后验分布。通过模拟研究，我们发现NPE在结果漂移和联合失配的情形下最为有用。

    arXiv:2609.38902v1 Announce Type: cross  Abstract: Augmenting small concurrent studies with external or historical cohorts is attractive in drug development, where enrollment is slow, follow-up is expensive, and closely related trial or real-world data are often already available. Bayesian dynamic borrowing (BDB) provides a principled framework for adaptively controlling the influence of external data, but classical implementations often depend on hand-specified priors and MCMC-based inference, which can be computationally expensive and not generalizable. In this work, we study amortized neural posterior estimation (NPE) as a flexible alternative. A single network is pretrained on simulated current/external dataset pairs spanning covariate shift, outcome drift, and joint non-exchangeability, and then returns an approximate posterior for a scalar current-study target in a single forward pass. Through simulation studies, we find that NPE is most useful under outcome drift and joint misma
    
[^35]: 马尔可夫数据下异步TD（时序差分）学习的精确统计速率

    Sharp Statistical Rates for Asynchronous TD Learning with Markovian Data

    [https://arxiv.org/abs/2609.38880](https://arxiv.org/abs/2609.38880)

    该论文为基于单条马尔可夫轨迹的表格型TD学习建立了精确的末次迭代样本复杂度界 $\widetilde O\left(\frac{H^3}{\mu_{\min}\varepsilon^2}+\frac{t_{\operatorname{mix}}}{\mu_{\min}}\right)$，且该界对常数步长与递减步长调度均成立。

    

    我们研究了基于有限马尔可夫奖励过程单条轨迹的标准表格型时序差分（TD）学习的末次迭代（last iterate）。对于折扣因子 $\gamma$，记 $H=(1-\gamma)^{-1}$，并令 $\mu_{\min}$ 和 $t_{\operatorname{mix}}$ 分别表示最小平稳概率和全变差混合时间。我们证明，对于 $0<\varepsilon\leq1$，末次迭代TD以高概率达到至多 $\varepsilon$ 的sup范数误差，所需转移步数为 $\widetilde O\left(\frac{H^3}{\mu_{\min}\varepsilon^2}+\frac{t_{\operatorname{mix}}}{\mu_{\min}}\right)$。该速率既适用于为目标精度所选的常数步长，也适用于与目标精度和终止时间无关的递减步长调度。后者在超过某个显式瞬态阈值之后的所有时刻上提供了一致的同时保证。统计项保留了同步TD中三次方（立方）的有效视界依赖关系，而附加的混合过渡项则是不可省略的。

    arXiv:2609.38880v1 Announce Type: new  Abstract: We study the last iterate of standard tabular temporal-difference (TD) learning from a single trajectory of a finite Markov reward process. For discount factor $\gamma$, write $H=(1-\gamma)^{-1}$, and let $\mu_{\min}$ and $t_{\operatorname{mix}}$ denote the minimum stationary probability and total-variation mixing time. We prove that last-iterate TD achieves sup-norm error at most $\varepsilon$ with high probability using$\widetilde O\left(   \frac{H^3}{\mu_{\min}\varepsilon^2}   +\frac{t_{\operatorname{mix}}}{\mu_{\min}} \right)$ transitions, for $0<\varepsilon\leq1$. This rate holds both for a constant step size selected for the target accuracy and for a decreasing schedule independent of the target accuracy and terminal time. The latter gives a simultaneous guarantee over all times beyond an explicit transient threshold. The statistical term retains the cubic effective-horizon dependence of synchronous TD, and the additive mixing tran
    
[^36]: 曲面下的最优配置与体积

    Optimal Allocation and Volume under Surface

    [https://arxiv.org/abs/2609.38875](https://arxiv.org/abs/2609.38875)

    本文提出基于Aumann期望表示与Minkowski混合体积的框架来估计ROC曲面下体积（VUS），并开发了其双重/去偏机器学习估计量及推断方法，还可用于分组可行误差分析和基尼系数的推广。

    

    本文开发了一个用于对临界函数集合的投影集合体积进行估计和推断的框架，特别关注最优受试者工作特征（ROC）曲面下方的凸体。具体而言，我们提出了一种体积计算方法，该方法首先使用Aumann期望表示，然后应用Minkowski混合体积。利用这一框架，我们证明了ROC曲面下的总体体积（VUS）与对称U统计量核的期望成正比。随后，我们提出了VUS的双重/去偏机器学习估计量，推导了其渐近性质，并开发了一种推断程序。该框架的进一步应用包括对预定义分组之间可行误差集合的分析，以及用于衡量不平等程度的基尼系数的自然推广。

    arXiv:2609.38875v1 Announce Type: new  Abstract: This paper develops a framework for estimation and inference on the volumes of sets that are projections of critical function sets, focusing particularly on the convex body beneath the optimal receiver operating characteristic (ROC) surface. Specifically, we propose a volume calculation method that first uses an Aumann expectation representation and then applies Minkowski mixed volumes. Using this framework, we show that the population volume under the ROC surface (VUS) is proportional to the expectation of a symmetric U-statistic kernel. We then propose a double/debiased machine learning estimator of the VUS, derive its asymptotic properties, and develop an inference procedure. Further applications of this framework include an analysis of the feasible error set across pre-defined groups and a natural generalization of the Gini coefficient for measuring inequality.
    
[^37]: 带间隔对比学习的最优VC维

    Optimal VC Dimension of Contrastive Learning with Margin

    [https://arxiv.org/abs/2609.38834](https://arxiv.org/abs/2609.38834)

    该论文解决了Alon等人提出的开放问题，确定了带间隔对比学习的最优VC维界。

    

    对比学习是一种成功的范式，它从“锚点-正例-负例”三元组$(i,j^{+},k^{-})$的集合中学习$d$维几何表示，这些三元组表明“物品$i$更接近$j$而非$k$”。尽管取得了成功，理解为什么对比学习能够产生高质量\textit{泛化}的表示——超越PAC学习通常给出的悲观预测——仍然是一个核心问题。最近，Alon等人证明了，对于$n$点数据集的$d$维欧几里得表示的PAC学习，$\Theta(\min(nd, n^2))$个三元组既是必要的也是充分的，同时他们提出了一个开放问题：在更贴近现实的“带间隔对比学习”设置下，他们的VC维界是否可以得到改进。对于间隔参数$\alpha > 0$，三元组$(i,j^{+},k^{-})_{\alpha}$被嵌入$\phi:[n]\rightarrow \mathbb{R}$……（摘要原文在此处截断）

    arXiv:2609.38834v1 Announce Type: cross  Abstract: Contrastive learning is a successful paradigm for learning $d$-dimensional geometric representations from a collection of ``anchor--positive--negative'' triplets $(i,j^{+},k^{-})$, indicating that ``item $i$ is closer to $j$ than to $k$.'' Despite its success, understanding why contrastive learning leads to representations of high \textit{generalization} quality---beyond the often pessimistic predictions from PAC-learning---remains a central question. Recently, \citet*{alon2024optimal} proved that, for PAC-learning $d$-dimensional Euclidean representations of $n$-point datasets, $\Theta(\min(nd, n^2))$ triplets are necessary and sufficient, while they posed as an open question whether their VC dimension bounds for the more realistic setting of \textit{contrastive learning with a margin} can be improved. For a margin parameter $\alpha >0$, a triplet $(i,j^{+},k^{-})_{\alpha}$ is satisfied by the embedding $\phi:[n]\rightarrow \mathbb{R}
    
[^38]: 理解离策略与在策略蒸馏：不同训练目标的对比研究

    Understanding Off- vs On-Policy Distillation: A Tale of Distinct Training Objectives

    [https://arxiv.org/abs/2609.38666](https://arxiv.org/abs/2609.38666)

    该论文从理论上揭示了在策略与离策略蒸馏分别对应反向KL与前向KL散度下的不同聚合目标（几何聚合与算术混合），由此解释了在策略蒸馏在减少遗忘方面的优势及其脆弱性。

    

    在策略蒸馏（OPD）通过教师对学生生成的响应给出反馈来学习，与监督微调（SFT）相比，其在减少遗忘方面展现出前景。然而，其优势与脆弱性尚未被充分理解。我们研究了从多个教师进行序贯蒸馏的场景，其中学生最小化其与各教师之间的平均散度。前向Kullback–Leibler（KL）散度会产生加权的算术混合目标，而反向KL散度则产生归一化的加权几何聚合目标。我们开发了分别适用于离策略和在策略反馈、用于学习这些目标的算法，在表格化设定中建立了对数遗憾界，并将分析扩展至函数逼近情形。通过分析这些聚合目标，我们识别出了有助于解释在策略蒸馏的优势与脆弱性的机制。相对于前向KL散度，反向KL散度能够更好地保留高置信度专家的偏好（原文摘要此处截断）。

    arXiv:2609.38666v1 Announce Type: cross  Abstract: On-policy distillation (OPD) learns from teacher feedback on student-generated responses and has shown promise in reducing forgetting relative to supervised fine-tuning (SFT). However, its benefits and fragility remain incompletely understood. We study sequential distillation from multiple teachers, where the student minimizes its average divergence from the teachers. Forward Kullback--Leibler (KL) divergence yields a weighted arithmetic mixture, while reverse KL yields a normalized weighted geometric aggregate. We develop algorithms that learn these targets under off-policy and on-policy feedback, respectively, establishing logarithmic regret bounds in the tabular setting and extending the analysis to function approximation. By analyzing these aggregation targets, we identify mechanisms that help explain both the benefits and fragility of OPD. Relative to forward KL, reverse KL can better retain a confident expert's preferences under 
    
[^39]: 具有多个最优臂的多臂老虎机：极小极大遗憾与非自适应性

    Bandits with Multiple Optimal Arms: Minimax Regret and Non-Adaptivit

    [https://arxiv.org/abs/2609.38659](https://arxiv.org/abs/2609.38659)

    该论文针对具有多个最优臂的多臂老虎机问题，通过对子采样算法的更精细分析建立了近乎极小极大最优的遗憾界 $\tilde{O}(\frac{K-A}{\sqrt{KA}}\sqrt{T})$，给出匹配下界，并证明了解最优臂数量对于达到近乎最优遗憾是必要的。

    

    我们研究具有多个最优臂的多臂老虎机问题，其动机在于许多实际的决策问题往往允许多个正确答案。对于具有 $A$ 个最优臂的 $K$ 臂老虎机，我们首先对先前的子采样算法（De Heide et al., 2021; Zhu and Nowak, 2020）进行了更精细的分析，建立了 $\tilde{O}\Big(\frac{K-A}{\sqrt{KA}}\sqrt{T}\Big)$ 的极小极大遗憾，其中 $T$ 是总交互次数，$\tilde{O}(\cdot)$ 省略了所有常数和对数因子，改进了此前 $\tilde{O}(\sqrt{KT/A})$ 的遗憾界。随后我们给出了在对数因子意义下与之匹配的下界，表明我们所建立的速率几乎达到了极小极大最优。我们进一步证明，在 $\tilde{O}(1)$ 因子意义下掌握最优臂的数量 $A$ 是实现近乎最优遗憾的必要条件，因为针对某一最优臂数量设计的近乎最优算法，在最优臂数量较少时必然会产生远大于最优遗憾的损失。

    arXiv:2609.38659v1 Announce Type: cross  Abstract: We study multi-armed bandits (MAB) with multiple optimal arms, motivated by the fact that many practical decision making problems admit multiple correct answers. For $K$-armed bandits with $A$ optimal arms, we first provide a sharper analysis of previous sub-sampling algorithms (De Heide et al., 2021; Zhu and Nowak, 2020), establishing a $\tilde{O}\Big(\frac{K-A}{\sqrt{KA}}\sqrt{T} \Big)$ minimax regret, where $T$ is the total number of interactions and $\tilde O(\cdot)$ drops all constant and logarithmic factors, improving the previous $\tilde{O}(\sqrt{KT/A})$ regret. We then provide a matching lower bound up to logarithmic factors, indicating that our established rate is nearly minimax-optimal. We further show that the knowledge of $A$ up to $\tilde{O}(1)$ factors is necessary to achieve near-optimal regret, as near-optimal algorithms for one number of optimal arms must incur substantially larger regret than optimal regret for a smal
    
[^40]: 尖峰-厚板回归的自适应混合变分推断

    Adaptive mixture variational inference for spike-and-slab regression

    [https://arxiv.org/abs/2609.38656](https://arxiv.org/abs/2609.38656)

    该论文提出了一种针对尖峰-厚板先验回归的自适应混合变分推断方法，通过在变量包含指示和活跃系数上直接最小化反向KL散度来捕捉相关变量选择的联合不确定性，并给出了后验收缩性、选择一致性和Bernstein–von Mises近似等理论保证。

    

    相关的预测变量可能支持多个预测效果相近的竞争性稀疏解释，这使得变量选择的联合不确定性难以用均值场近似来刻画。我们针对具有点质量尖峰-厚板先验的高斯回归，开发了一种面向乘积分布混合的自适应拟合方法。该方法直接在变量包含指示变量和活跃系数上最小化反向Kullback-Leibler散度，并随着混合成分的增长联合优化成分参数与权重。这避免了在独立增广方法下对未使用的潜在系数施加额外的散度惩罚。我们的分析将近似精度与混合成分数量、支撑集的覆盖范围以及支撑集内部的依赖结构联系起来，并在先验、后验集中性和变分误差的明确条件下，建立了后验收缩性、变量选择一致性以及Bernstein–von Mises近似。在全部250个模拟数据集上……（原文摘要在此处截断）

    arXiv:2609.38656v1 Announce Type: cross  Abstract: Correlated predictors can support competing sparse explanations with similar predictions, making joint uncertainty about variable inclusion difficult to capture with mean-field approximations. We develop an adaptive fitting procedure for mixtures of product distributions in Gaussian regression with a point-mass spike-and-slab prior. It minimizes reverse Kullback-Leibler divergence directly on inclusion indicators and active coefficients, jointly refining component parameters and weights as the mixture grows. This avoids an additional divergence penalty on unused latent coefficients under independent augmentation. Our analysis relates approximation accuracy to mixture size, support coverage and dependence within supports, and establishes contraction, selection consistency and a Bernstein-von Mises approximation under explicit conditions on the prior, posterior concentration and variational error. On all 250 simulated datasets with exact
    
[^41]: 基于学习的估计：样本选择偏差下的紧致刻画

    Learning-Enabled Estimation: Tight Characterizations under Sample Selection Biases

    [https://arxiv.org/abs/2609.38608](https://arxiv.org/abs/2609.38608)

    该论文首次为存在样本选择偏差时的回归学习问题提供了完整的紧致刻画，确立了选择机制函数形式所需的最小必要假设条件。

    

    我们何时能够从有偏样本中学习？我们研究这样一种回归问题：结果变量只有在通过同时依赖于协变量和结果自身的选择过滤器之后才能被观测到，这一挑战普遍存在于存在患者脱落的临床试验、存在自我选择的劳动力市场以及存在策略性进入的拍卖等场景中。忽视这种选择会产生具有系统性偏差的结论，并带来现实世界层面的后果。这一挑战在计量经济学和统计学中有着悠久的历史，始于Heckman开创性的两阶段模型，随后出现了众多推广工作。尽管这些工作为可识别性提供了各种充分条件，但对于何时能够进行此类回归的完整刻画始终未能实现。在本工作中，我们给出了存在样本选择偏差时回归何时可行的完整刻画。我们的结果确立了对选择机制函数形式所需的最小假设条件……

    arXiv:2609.38608v1 Announce Type: cross  Abstract: When can we learn from biased samples? We study regression when outcomes are observed only after passing through selection filters that depend on both covariates and outcomes themselves, a ubiquitous challenge spanning clinical trials with patient dropout, labor markets with self-selection, and auctions with strategic entry. Ignoring such selection yields systematically biased conclusions with real-world consequences. This challenge has a long history in econometrics and statistics, starting with Heckman's seminal two-stage model and followed by numerous generalizations. While these works provide various sufficient conditions for identification, a complete characterization of when such regression is possible has remained elusive.   In this work, we provide a characterization for when regression is possible in the presence of sample selection bias. Our results establish the minimal assumptions required on the functional forms of selecti
    
[^42]: 通过流匹配迈向通用Wasserstein重心

    Towards Universal Wasserstein Barycenters through Flow Matching

    [https://arxiv.org/abs/2609.38547](https://arxiv.org/abs/2609.38547)

    提出BaryFM流匹配模型，实现对Wasserstein单纯形上任意权重重心的通用近似，一次训练后即可通过常微分方程从任意重心测度采样，并在领域自适应、贝叶斯后验聚合等多个下游任务中取得最佳平均表现。

    

    在概率度量下定义概率测度的加权平均是概率机器学习中的一种核心工具。在Wasserstein度量下，这种加权平均被称为“Wasserstein重心”。尽管大多数方法仅针对固定权重向量计算重心，但对单纯形上整个重心家族（我们称之为“Wasserstein单纯形”）进行近似的研究仍然不足。我们将这一问题称为“通用重心近似”，并提出BaryFM——一个流匹配模型，它将边缘测度传输到Wasserstein单纯形中的任意重心。训练完成后，该网络可以通过常微分方程从Wasserstein单纯形中的测度中采样。我们在4个下游任务上验证了该方法：领域自适应、泛化、贝叶斯后验聚合和算法公平性。BaryFM在15种竞争方法中取得了最佳平均排名。

    arXiv:2609.38547v1 Announce Type: cross  Abstract: Defining a weighted mean over probability measures under probability metrics is a central tool in probabilistic machine learning. Under the Wasserstein metric, these are called \emph{Wasserstein barycenters}. While most approaches compute barycenters for a fixed weight vector, approximating the whole family of barycenters over the simplex, which we call the \emph{Wasserstein simplex}, remains underexplored. We refer to this problem as \emph{Universal Barycenter Approximation}, and propose \texttt{BaryFM}, a flow matching model transporting the marginal measures into any barycenter in the Wasserstein simplex. Once trained, the network can draw samples from measures in the Wasserstein simplex through an ordinary differential equation. We validate our method on 4 downstream tasks: domain adaptation, generalization, Bayesian posterior aggregation and algorithmic fairness. \texttt{BaryFM} achieves the best average rank among 15 competing me
    
[^43]: 基于预测状态的无穷记忆过程生成式序列建模

    Generative sequence modeling for infinite memory processes via predictive states

    [https://arxiv.org/abs/2609.38524](https://arxiv.org/abs/2609.38524)

    本文提出一种基于预测状态的生成式序列建模新方法，能够处理具有无穷记忆的随机过程，当过去历史可被压缩为低维充分统计量时实现快速收敛，且估计问题的统计复杂度仅由预测状态空间的内在维数决定。

    

    我们考虑估计多元随机过程的一步预测条件分布。许多现有方法依赖于有限范围记忆、稀疏性或可加性等假设，这些假设对于具有长程非线性交互作用的过程可能并不适用。然而，在没有这些结构性假设的情况下，由于维数灾难，非参数估计面临巨大挑战。为了应对这一挑战，我们提出了一种基于过程预测状态的新估计方法，该方法可以处理可能具有无穷范围记忆的过程。我们证明了当过去的历史信息能够被压缩成一个足以预测未来的低维统计量时，我们的估计器能够达到快速收敛速度。具体而言，我们证明了该估计问题的统计复杂度由预测状态空间的内在维数决定。我们为基于该方法的一种具体实现建立了理论保证（摘要在此处被截断）。

    arXiv:2609.38524v1 Announce Type: new  Abstract: We consider estimating the one-step-ahead conditional distribution of a multivariate stochastic process. Many existing approaches rely on assumptions such as finite-range memory, sparsity, or additivity, which can be poorly suited to processes with long-range nonlinear interactions. However, without such structural assumptions, nonparametric estimation is challenging due to the curse of dimensionality. To address this challenge, we introduce a new estimation approach based on the predictive states of a process, possibly with infinite-range memory. We show that our estimator achieves fast convergence rates when the past history can be compressed into a low-dimensional statistic that is sufficient for predicting the future. Specifically, we show that the statistical complexity of the estimation problem is determined by the intrinsic dimension of the predictive state space. We establish guarantees for an instantiation of our method based on
    
[^44]: ShamAN-Q：用于低于1比特大语言模型权重的Shampoo增强NanoQuant

    ShamAN-Q: Shampoo Augmented NanoQuant for Sub-1-bit LLM Weights

    [https://arxiv.org/abs/2609.38521](https://arxiv.org/abs/2609.38521)

    ShamAN-Q用Shampoo式的密集曲率度量（对经验Fisher信息矩阵进行Kronecker拟合并构建马氏重建损失）替代NanoQuant的对角重建几何，实现了大语言模型的低于1比特训练后量化。

    

    我们提出了ShamAN-Q，这是一种低于1比特的训练后量化方法，它通过将NanoQuant的对角重建几何替换为可处理的密集曲率度量来扩展NanoQuant，并采用了由Shampoo优化器推广的通用范式。对于每个线性权重，ShamAN-Q通过Kullback-Leibler最小化，将克罗内克积拟合到小型校准集的经验Fisher信息矩阵上，并由此构建马氏重建损失。NanoQuant的连续ADMM更新由此变为Sylvester方程的解，而其离散投影和部署格式保持不变。由于曲率对于给定权重集是局部的，ShamAN-Q在每层分解之前立即重新测量该层的输入曲率统计量，并定期在部分量化后的模型上刷新所有统计量。ShamAN-Q还将NanoQuant的统一秩重新分配到各层……

    arXiv:2609.38521v1 Announce Type: cross  Abstract: We introduce ShamAN-Q, a sub-1-bit post-training quantization method that extends NanoQuant by replacing each its diagonal reconstruction geometry with a tractable dense curvature metric, using a general paradigm popularized by the Shampoo optimizer. For each linear weight, ShamAN-Q fits a Kronecker product to the empirical Fisher information matrix of a small calibration set by Kullback--Leibler minimization, forming a Mahalanobis reconstruction loss from the result. The continuous ADMM updates from NanoQuant become solutions to Sylvester equations, while its discrete projection and deployment format remain unchanged. Because the curvature is local to a given set of weights, ShamAN-Q re-measures the input curvature statistic for each layer immediately before layer factorization, periodically refreshing all statistics on the partially quantized model. ShamAN-Q also redistributes the uniform rank from NanoQuant across layers at the same
    
[^45]: 从最小范数插值的视角解读Grokking（顿悟）现象

    Grokking through the Lens of Minimum-Norm Interpolation

    [https://arxiv.org/abs/2609.38453](https://arxiv.org/abs/2609.38453)

    该论文建立了一个统计理论，证明在高维过参数化无噪声回归中存在零—一泛化定律，并构造了一族凸范数，使插值解在保持零训练误差的同时从平凡解过渡到信号的精确恢复，从而从正则化几何与信号稀疏性的角度为Grokking的延迟泛化现象提供了定量解释。

    

    Grokking现象表明，拟合训练数据与学习到潜在信号可能发生在截然不同的阶段。然而，现有理论对于这种延迟泛化如何依赖于归纳偏置和信号结构，所能提供的定量洞见十分有限。我们的工作通过建立一个统计理论来填补这一空白，该理论刻画了正则化几何与信号稀疏性如何共同支配插值附近的泛化行为。特别地，我们聚焦于高维回归这一典型场景，识别出了促进稀疏性的正则化使精确插值比近似拟合准确得多的运行区间。在强过参数化的无噪声问题中，我们证明了一个零—一泛化定律，并构造了一族凸范数，使得其对应的插值解在保持训练误差为 $0$ 的同时，从全零预测器的平凡风险过渡到对信号的精确恢复。此外，当特征……（原文在此处截断）

    arXiv:2609.38453v1 Announce Type: cross  Abstract: Grokking shows that fitting the training data and learning the underlying signal can occur at very different stages. However, existing theories offer limited quantitative insight into how this delayed generalization depends on inductive bias and signal structure. Our work addresses the gap by developing a statistical theory that characterizes how regularization geometry and signal sparsity govern generalization near interpolation. In particular, we focus on the prototypical setting of high-dimensional regression and identify regimes in which sparsity-promoting regularization makes exact interpolation much more accurate than approximate fitting. In strongly overparameterized noiseless problems, we prove a zero--one generalization law and construct a family of convex norms whose interpolators transition from the trivial risk of the all-zero predictor to exact recovery, while keeping the training error equal to $0$. Furthermore, when feat
    
[^46]: 局部多项式密度比估计

    Local polynomial density ratio estimation

    [https://arxiv.org/abs/2609.38412](https://arxiv.org/abs/2609.38412)

    提出了一种新的局部多项式密度比估计器，在任意光滑度的 Hölder 类上达到无需额外对数因子的逐点极小化极大最优速率，仅需假设密度比本身光滑而无需单个密度光滑，在支撑集边界点上依然有效，并附带可用于分类任务的集中不等式。

    

    我们提出了一种新颖的局部多项式估计器，用于估计两个 $d$ 维密度函数 $f$ 和 $g$ 的比值 $r=f/g$，其中我们拥有来自这两个分布的独立样本。该估计器被证明在任意光滑度指标的 Hölder 类上达到逐点极小化极大最优速率，且不包含额外的对数因子，同时只需假设比值 $r$ 本身具有光滑性，而无需假设 $f$ 或 $g$ 具有光滑性。在对待定边界施加温和几何假设的条件下，我们的分析对于与 $g$ 相关的分布支撑集边界上的点仍然有效。我们还推导了该估计器的集中不等式，这在分类等应用中可能非常有用，并给出了上确界范数下的收敛速率，其中上确界也涵盖了边界支撑点。获得该速率的光滑度类足够大，以至于单个密度函数 $f$ 和 $g$ 在该类上无法被一致地均匀估计。

    arXiv:2609.38412v1 Announce Type: cross  Abstract: We propose a novel local-polynomial estimator of the ratio $r=f/g$ of two $d$-dimensional densities $f$ and $g$, from which independent samples are available. The estimator is shown to achieve pointwise minimax optimal rates over H\"older classes of arbitrary smoothness index without additional logarithmic factors, and with smoothness being only assumed of $r$ but not of $f$ nor $g$. Our analysis remains valid for points on the boundary of the support of the distribution associated to $g$ under a mild geometric assumption on the (unknown) boundary. We also derive a concentration inequality for the estimator, which can be useful in applications to classification, and give a rate in the supremum norm, where the supremum is also taken over boundary support points. The smoothness class over which the rate is obtained is sufficiently large that the individual densities $f$ and $g$ cannot be consistently estimated uniformly over this class. 
    
[^47]: 量子PAC学习中的样本复杂度优势需要对态制备酉算子的逆的访问

    Advantage of Sample Complexity in Quantum PAC Learning Requires Inverse Access to State-Preparation Unitaries

    [https://arxiv.org/abs/2609.38403](https://arxiv.org/abs/2609.38403)

    本文证明了在量子PAC学习中，查询复杂度对精度参数依赖的改进优势本质上必须依赖于对态制备酉算子逆的访问，而仅有前向访问的酉算子无法带来任何样本复杂度上的优势。

    

    量子计算能否减少学习一个预测规则所需从未知概率分布中采样的数据量，是量子机器学习中的一个基本问题。量子PAC学习通过量子数据来研究这一问题，其中量子数据是一种量子态，其振幅的平方编码了用于采样经典学习数据的未知分布。仅拥有此类量子数据的副本时，最优的最坏情况样本复杂度渐近地与经典PAC学习相匹配。相比之下，同时访问该量子态的态制备酉算子及其逆算子，可以在可实现学习中改善查询复杂度对精度参数的依赖关系。然而，仅前向访问（即只访问酉算子本身而不访问其逆）是否能够带来这样的改进，一直尚不明确。在本工作中，通过对所有兼容的态制备酉算子及其有限环境维度取最坏情况，我们证明了仅前向访问的最优查询复杂度无法获得该优势，即样本复杂度的提升本质上依赖于对态制备酉算子逆的访问。

    arXiv:2609.38403v1 Announce Type: cross  Abstract: Whether quantum computation can reduce the amount of data sampled from an unknown probability distribution required to learn a prediction rule is a fundamental question in quantum machine learning. Quantum PAC learning studies this question using quantum data as a quantum state whose squared amplitudes encode the unknown distribution from which classical learning data are sampled. With only copies of such quantum data, the optimal worst-case sample complexity asymptotically matches that of classical PAC learning. In contrast, access to both a state-preparation unitary for this state and its inverse can improve the query-complexity dependence on the accuracy parameter in realizable learning. However, it has remained unclear whether forward-only access allows such an improvement.   In this work, taking the worst case over compatible state-preparation unitaries and their finite ambient dimensions, we show that the optimal forward-only que
    
[^48]: PTED：一种用于科学推断与生成式机器学习的多维双样本检验

    PTED: A multi-dimensional two-sample test for scientific inference and generative machine learning

    [https://arxiv.org/abs/2609.38388](https://arxiv.org/abs/2609.38388)

    本文提出了基于能量距离的置换检验PTED——一种Python实现的多维精确双样本检验方法，由于统计量仅依赖成对距离，它可适用于高维数据、任意可定义距离的数据类型及不平衡样本，并通过近似公式实现线性扩展。

    

    双样本检验在统计推断和生成式建模中应用广泛，但由于缺乏一种易于使用、能够在多维空间中运行的检验方法，用户常常只能退而求其次，依赖启发式方法和肉眼观察。本文提出了基于能量距离的置换检验，这是Székely和Rizzo [2004] 所描述的一种强大双样本检验的Python实现。该检验统计量为能量距离——一种定义在概率分布上的度量，它具有由成对距离构建的天然样本估计量。通过对能量距离运行置换检验，可得到一个精确的双样本检验。由于该统计量仅通过成对距离依赖于数据，因此该检验适用于高维数据、学习到的特征表示、大样本、小样本或不平衡样本，以及任何可以定义距离的数据类型。对该公式的近似使得PTED能够随维度数量（与样本数量）线性扩展。

    arXiv:2609.38388v1 Announce Type: cross  Abstract: Two-sample tests are widely applicable in inference and generative modelling, yet users frequently fall back on heuristics and visual inspection due to lack of an accessible test that operates in multiple dimensions. I present Permutation Test using the Energy Distance (PTED), a Python implementation of a powerful two-sample test described in Sz\'ekely and Rizzo [2004]. The test statistic is the energy distance, a metric on probability distributions that admits a natural sample estimator built from pairwise distances. A permutation test is run on energy distances to produce an exact two-sample test. Because the statistic depends on the data only through pairwise distances, the test applies in high dimensions, on learned feature representations, at large or small or imbalanced sample sizes, and to any data type on which a distance can be defined. An approximation to the formula allows PTED to scale linearly with both number of dimension
    
[^49]: 从随机探索中学习规划

    Learning to Plan from Random Exploration

    [https://arxiv.org/abs/2609.38383](https://arxiv.org/abs/2609.38383)

    该论文提出一种仅从随机探索数据中、通过时程条件化的能量模型与噪声对比估计来学习时间关系的方法，从而无需动作或奖励标签、无需策略改进训练即可实现长程规划。

    

    随机探索能够在目标被指定之前揭示环境如何被遍历。这些经验能否在无需策略改进训练的情况下支持长程规划？我们的随机游走分析解释了时间关系所包含的信息：在扩散极限下，短时程揭示测地线几何，而更长的时程则揭示区域之间的连通性，直到混合过程抹去这些区别。我们使用一个条件能量模型来学习这些时间关系，该模型通过时程条件化的嵌入来估计时间对数密度比。该模型仅使用观测对、通过噪声对比估计进行训练，无需动作或奖励标签。规划器在向目标移动的过程中，会在不同时程上查询这些学到的关系。在测试阶段，一个独立的局部动力学模型预测候选动作的结果，而时间模型通过选择或聚合估计的改进量来评估这些动作向目标推进的程度。

    arXiv:2609.38383v1 Announce Type: cross  Abstract: Random exploration reveals how an environment can be traversed before a goal is specified. Can this experience support long-range planning without policy-improvement training? Our random-walk analysis explains what temporal relations contain: short horizons reveal geodesic geometry in the diffusion limit, while longer horizons reveal connectivity between regions before mixing removes these distinctions. We learn these relations with a conditional energy-based model that estimates temporal log-density ratios through horizon-conditioned embeddings. The model is trained on observation pairs by noise-contrastive estimation, without action or reward labels. The planner queries these learned relations at different horizons as it moves toward the goal. At test time, a separate local dynamics model predicts candidate action outcomes, and the temporal model evaluates their progress toward the goal by selecting or aggregating estimated improveme
    
[^50]: 线性预言机在线学习的下界

    Lower Bounds for Linear-Oracle Online Learning

    [https://arxiv.org/abs/2609.38375](https://arxiv.org/abs/2609.38375)

    本文证明了Weibel等人的猜想，即每轮常数次线性最小化无法在一般凸集上改进在线Frank-Wolfe的$T^{3/4}$遗憾率，并将该下界扩展到仅使用预言机模型中的所有确定性学习者。

    

    在一般凸集上，每轮进行常数次线性最小化能否改进在线Frank-Wolfe的$T^{3/4}$遗憾率？Weibel等人猜想固定系数的方法无法做到。我们证明了他们的猜想，并将该下界扩展到“仅预言机”模型中的所有确定性学习者。学习者获得一个初始可行点和直径上界，并且必须在其预言机回复所一致的所有域上保持可行。对于$T$轮、每次决策之间至多$b$次调用、直径上界$D$和梯度范数上界$L$，我们构造了一个维度$d=2b(T-1)+1$的实例，其遗憾值至少为$2^{-1/4}LDb^{-1/4}T^{3/4}$。对手在博弈开始前固定域、初始点、确定性平局规则和线性损失。顶点构成一条路径，使得决策前可用的每个点当前损失为零，而最后一个顶点在每一轮上都具有负损失。对于常数$b$，该结果……

    arXiv:2609.38375v1 Announce Type: new  Abstract: Can a constant number of linear minimizations per round improve on the $T^{3/4}$ regret rate of online Frank-Wolfe on general convex sets? Weibel et al. conjectured that fixed-coefficient methods cannot. We prove their conjecture and extend the lower bound to every deterministic learner in an oracle-only model. The learner receives an initial feasible point and a diameter bound, and must remain feasible on every domain consistent with its oracle replies. For $T$ rounds, at most $b$ calls between decisions, diameter bound $D$, and gradient norm bound $L$, we construct an instance in dimension $d=2b(T-1)+1$ with regret at least $2^{-1/4}LDb^{-1/4}T^{3/4}$. The adversary fixes the domain, initial point, deterministic tie rule and linear losses before play. The vertices form a path on which every point available before a decision has zero current loss, while the final vertex has negative loss on every round. For constant $b$, the result matc
    
[^51]: 通过离散平均生成器加速扩散语言模型

    Acceleration of Diffusion Language Model through Discrete Average Generator

    [https://arxiv.org/abs/2609.38364](https://arxiv.org/abs/2609.38364)

    提出离散平均生成器，将MeanFlow扩展至连续时间马尔可夫链，通过自洽恒等式训练目标实现扩散语言模型的高效少步生成加速。

    

    离散扩散模型和流匹配已成为离散状态空间上生成建模的强大框架，然而高效的少步生成仍然是一个根本性挑战。在本工作中，我们提出了离散平均生成器，这是MeanFlow向连续时间马尔可夫链（CTMCs）的一个原则性扩展。类似于MeanFlow在连续空间中定义时间区间上的平均速度场，我们将平均生成器定义为转移核在时间区间上的归一化增量。我们证明该平均生成器满足一个自洽恒等式，这为我们的训练目标奠定了基础。我们进一步开发了与扩散语言模型标准训练范式相一致、同时保持所得目标可处理的训练策略。当投影到逐坐标边缘分布时，该自洽恒等式具有闭式表达式。

    arXiv:2609.38364v1 Announce Type: cross  Abstract: Discrete diffusion models and flow matching have emerged as powerful frameworks for generative modeling over discrete state spaces, yet efficient few-step generation remains a fundamental challenge. In this work, we introduce the Discrete Average Generator, a principled extension of MeanFlow to Continuous-Time Markov Chains (CTMCs). Analogously to how MeanFlow defines an average velocity field over a time interval in continuous spaces, we define an average generator as the normalized increment of the transition kernel over a time interval. We show that this average generator satisfies a self-consistency identity, which provides the foundation for our training objective. We further develop training strategies that align with the standard training paradigm of diffusion language models while keeping the resulting objective tractable. When projected onto per-coordinate marginals, the self-consistency identity admits a closed-form expressio
    
[^52]: 鲁棒LassoNet：通过鲁棒损失函数增强神经网络中的特征选择

    Robust LassoNet: Enhancing Feature Selection in Neural Networks via Robust Loss Functions

    [https://arxiv.org/abs/2609.38263](https://arxiv.org/abs/2609.38263)

    本文提出Robust LassoNet，通过在LassoNet中引入Huber、Cauchy、Tukey双平方等鲁棒损失函数，在保留原始优化框架的同时，显著提升了神经网络在数据污染情况下的特征选择准确性与预测性能。

    

    神经网络中的特征选择仍然是一个具有挑战性的问题，尤其是在存在噪声或被污染数据的情况下。LassoNet是近期提出的一种应对该问题的方法，它将神经网络与分层稀疏性约束相结合，能够同时进行预测和变量选择。然而，其标准形式依赖于均方误差（MSE）损失，而MSE损失对异常值高度敏感是众所周知的。在本文中，我们提出了Robust LassoNet，这是LassoNet的一种扩展，它引入了鲁棒损失函数，如Huber损失、Cauchy损失、Tukey双平方损失和非负Garrote损失，以减轻极端观测值的影响。所提出的方法在保留原始优化框架的同时，提高了数据被污染情况下的稳定性。通过在合成数据集和真实数据集上的实验，我们证明了鲁棒LassoNet在预测性能和特征选择准确性方面均有显著提升。

    arXiv:2609.38263v1 Announce Type: new  Abstract: Feature selection in neural networks remains a challenging problem, particularly in the presence of noisy or contaminated data. LassoNet is a recent approach that addresses this issue by combining neural networks with hierarchical sparsity constraints, enabling simultaneous prediction and variable selection. However, its standard formulation relies on the mean squared error (MSE) loss, which is known to be highly sensitive to outliers. In this paper, we present Robust LassoNet, an extension of LassoNet that incorporates robust loss functions, such as Huber, Cauchy, Tukey's bisquare, and Nonnegative Garrote, to mitigate the effect of extreme observations. The proposed approach preserves the original optimization framework while improving stability under data contamination. Through experiments on synthetic and real datasets, we show that robust LassoNet significantly improves both predictive performance and feature selection accuracy in th
    
[^53]: 面向多跳检索增强生成的保形事实性控制

    Conformal Factuality Control for Multi-Hop Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.38222](https://arxiv.org/abs/2609.38222)

    该研究将主张级保形事实性控制应用于多跳检索增强生成，证明在六种模型-数据集配置中，保形过滤能将保留主张获完全支持的回复比例从无过滤时的55.60%-76.03%稳定提升至95%目标下的95.80%-97.20%。

    

    检索增强生成（RAG）可以将大语言模型建立在外部证据之上，但检索到的上下文并不能保证生成的主张在事实上得到支持。这一问题在多跳RAG中尤为突出，因为其检索和推理需要经过多个相互依赖的阶段。我们研究了此前为RAG开发的主张级保形事实性控制（conformal factuality control）在这一设置中是否依然有效。我们将分割保形主张过滤（split-conformal claim filtering）应用于多跳RAG，并在HotpotQA、Natural Questions和TriviaQA数据集上使用Llama 3.1 8B和GPT-4o-mini进行评估，同时开展了单跳参考实验。在全部六种多跳模型-数据集配置中，越来越严格的保形目标始终提升了其保留主张获得完全支持的回复比例。在95%目标下，该比例达到95.80%至97.20%，而未过滤时仅为55.60%-76.03%。然而，这一改进（摘要原文在此处截断）……

    arXiv:2609.38222v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) can ground large language models in external evidence, but retrieved context does not guarantee that generated claims are factually supported. This problem is especially relevant in multi-hop RAG, where retrieval and reasoning proceed through multiple dependent stages. We study whether claim-level conformal factuality control, previously developed for RAG, remains effective in this setting. We apply split-conformal claim filtering to multi-hop RAG and evaluate it on HotpotQA, Natural Questions, and TriviaQA using Llama 3.1 8B and GPT-4o-mini, together with a single-hop reference experiment. Across all six multi-hop model-dataset configurations, increasingly stringent conformal targets consistently increase the fraction of responses whose retained claims are fully supported. At the 95% target, this rate ranges from 95.80% to 97.20%, compared with 55.60%-76.03% without filtering. However, the improvemen
    
[^54]: 时滞物理系统驱动项与动力学的可辨识性保证

    Identifiability Guarantees for Drivers and Dynamics of Delayed Physical Systems

    [https://arxiv.org/abs/2609.37944](https://arxiv.org/abs/2609.37944)

    本文提出一种有理论支撑的方法，证明在宽松假设下随机时滞微分方程的结构驱动项与漂移项是可辨识的，并在驱动项可辨识性与动力学物理一致性基准上优于现有方法。

    

    目前已有大量方法被提出，包括物理信息神经网络（功能强大但不保证动力学的可辨识性）、符号回归（需要一组预先计算好的操作）以及因果发现（更具原则性，但通常依赖于物理系统可能违反的强假设）。在本工作中，我们开发了一种有理论支撑的方法，并证明在一组宽松的假设下，随机时滞微分方程的结构驱动项和漂移项是可辨识的。我们的方法在驱动项可辨识性基准测试中优于其他方法，并在第二个用于评估所学动力学物理一致性的基准测试中也表现更佳。

    arXiv:2609.37944v1 Announce Type: cross  Abstract: A wide range of methods have been proposed, including physics-informed neural networks, which are powerful but do not guarantee identifiability of the dynamics, symbolic regression, which requires a set of precomputed operations, and causal discovery, which is more principled but usually relies on strong assumptions that physical systems may violate. In this work, we develop a theory-grounded method and prove that under a set of permissive assumptions, the structural drivers and drift of stochastic delayed differential equations are identifiable. Our method outperforms others on a benchmark for driver identifiability, and on a second benchmark to evaluate physical consistency of the learned dynamics.
    
[^55]: 通过行与列尺度场观察Transformer权重的介观视角

    A Mesoscopic View of Transformer Weights Through Row and Column Scale Fields

    [https://arxiv.org/abs/2609.35852](https://arxiv.org/abs/2609.35852)

    提出以“行与列尺度场”作为Transformer权重矩阵的介观级精确表示，揭示平衡化后核心幅值分布在不同模型规模与初始化下高度相似，且尺度场在共享功能通道的投影间对齐、查询/键分布随重新分配的RoPE频率变化。

    

    Transformer权重的汇总统计量掩盖了幅值在功能通道上的分布方式，而单个权重数量过于庞大，无法直接逐一比较。我们研究介于两者之间的介观层面：行与列尺度场，即权重矩阵在其各通道上以中位数为中心的对数均方根（log-RMS）分布；它与一个全局尺度以及一个完整的平衡核心一起，可以精确地表示整个权重矩阵。在四个规模的公开Pythia检查点以及来自三种初始化家族的受控实验中，平衡化揭示了相似的可测核心幅值分布。一种混合桥接模型——其形式在分析之前固定、仅拟合其系数——在受控网格的留出运行和数据分支上预测了汇总形状相对场宽度的偏离。带索引的尺度场保留了进一步的结构：它们在共享同一功能通道的投影之间相互对齐，并且查询/键的分布遵循重新分配的RoPE频率，而非固定的矩阵位置……

    arXiv:2609.35852v1 Announce Type: new  Abstract: Pooled statistics of Transformer weights obscure how magnitude is distributed across functional channels, while individual weights are too numerous to compare directly. We study the mesoscopic level between them: row and column scale fields, the median-centred log-RMS profiles of a weight matrix over its channels, which together with a global scale and a full balanced core represent the matrix exactly. Across public Pythia checkpoints at four sizes and controlled runs from three initialization families, balancing reveals similar measured core magnitude profiles. A mixture bridge, with its form fixed before the analysis and its coefficients fitted, predicts the pooled-shape departure from field width on held-out runs and data arms of the controlled grid. The indexed fields retain further structure: they align across projections that share a functional channel, and query/key profiles follow reassigned RoPE frequencies rather than fixed mat
    
[^56]: 基于模拟退火的三阶朗之万动力学在非凸优化中的全局收敛性

    Global Convergence of Third-Order Langevin Dynamics for Non-Convex Optimization via Simulated Annealing

    [https://arxiv.org/abs/2609.28611](https://arxiv.org/abs/2609.28611)

    该论文证明了在模拟退火框架下，采用固定摩擦与递减噪声的三阶朗之万动力学在非凸优化中可依概率收敛到全局最小值，并给出了离散化格式保持该收敛速率的充分步长条件。

    

    我们研究了在固定摩擦力与递减噪声的模拟退火框架下，三阶朗之万动力学用于非凸优化的全局收敛性保证。一个显式的三块扭曲熵结构将耗散从含噪的辅助变量传递到整个状态空间。在耗散性、正则性以及低温泛函不等式假设下，对数冷却调度使目标值以势垒控制的动力学速率依概率收敛到全局最小值。对于精确力积分和中点三阶段离散化格式，多项式递减的步长可在物理时间尺度上保持该收敛速率。三次局部端点估计给出了比现有冻结力动力学结果更宽松的充分步长条件。与单梯度UBU积分器的比较表明，在相同的强耦合分析框架下，其中心化随机局部误差会带来更小的充分（步长条件要求）。

    arXiv:2609.28611v1 Announce Type: cross  Abstract: We study global convergence guarantees of third-order Langevin dynamics for non-convex optimization via simulated annealing with fixed friction and decreasing noise. An explicit three-block distorted entropy transfers dissipation from the noisy auxiliary variable to the full state. Under dissipativity, regularity, and low-temperature functional-inequality assumptions, logarithmic cooling drives the objective values to the global minimum in probability at the barrier-controlled kinetic rate. For the exact-force-integral and midpoint three-stage discretizations, polynomially decreasing steps preserve this rate on the physical time scale. The cubic local endpoint estimate gives a less restrictive sufficient step-size condition than the available frozen-force kinetic result. A comparison with the one-gradient UBU integrator shows how its centered stochastic local error leads, under the same strong-coupling analysis, to a smaller sufficient
    
[^57]: 从预测到可解释的医疗服务提供者行为画像：面向欺诈、浪费与滥用审查

    From Prediction to Explainable Provider Behavior Profiles for Fraud, Waste, and Abuse Review

    [https://arxiv.org/abs/2609.28477](https://arxiv.org/abs/2609.28477)

    该研究提出将欺诈、浪费与滥用（FWA）审查从预测建模转向可解释的提供者行为画像，通过将账单收入分解为提供者规模与诊疗项目构成的乘积，从而解释提供者行为变化的原因。

    

    理赔数据可以显示医疗服务提供者的行为发生了变化，但仅凭数据本身无法解释原因。欺诈、浪费与滥用（FWA）审查需要识别重要的行为、定位驱动这些行为的诊疗代码和资金，并检验合理的解释。一种常见的替代方法——预测建模——通过标记偏离预期使用量预测的异常来发现问题，但预测的价值有限，除非它能够超越简单的持续性预测，并解释偏差为何重要。在我们的季度提供者-诊疗项目数据中，最新观测值已捕获了大部分可预测的变化，而增加模型结构几乎无法提升准确性。残差将增长、服务线变化、代码维护以及不完整的观测与潜在的可疑行为混为一谈，使得单点预测并不完整。因此，我们将提供者审查重新表述为一个描述性表示问题：账单收入 y = s × p，其中 s 衡量提供者规模，p 描述诊疗项目构成。

    arXiv:2609.28477v1 Announce Type: cross  Abstract: Claims data can show that provider behavior changed but cannot by itself explain why. FWA (fraud, waste, and abuse) review requires identifying material behavior, locating the codes and dollars driving it, and testing plausible explanations. A common alternative, predictive modeling, flags deviations from an expected-utilization forecast -- but a forecast has limited value unless it beats simple persistence and explains why a deviation matters. In our quarterly provider-procedure data, the latest observation captures most forecastable variation, and added model structure adds little accuracy. Residuals conflate growth, service-line shifts, code maintenance, and incomplete observation with potentially concerning behavior, making point forecasts incomplete.   We instead formulate provider review as a descriptive representation problem: billed revenue y = s * p, where s measures provider scale and p describes procedure composition. The pr
    
[^58]: 面向计数数据的随机流映射

    Stochastic Flow Map for Count Data

    [https://arxiv.org/abs/2609.23290](https://arxiv.org/abs/2609.23290)

    Count Flow Map 是一种直接在计数空间中学习有限时间随机转移的生成模型，通过泊松出生与二项死亡机制保持非负整数计数，实现一步或少数几步的高效计数数据生成。

    

    高维计数数据在科学应用中十分常见，但大多数扩散模型和流模型是为连续数据或分类数据设计的，且生成过程通常需要多次顺序模型评估。我们提出了 Count Flow Map，这是一种直接在计数空间中学习有限时间转移的生成模型，可实现一步或少数几步生成。我们的模型直接学习有限时间区间上的随机转移，利用泊松出生和二项死亡过程来保持非负整数计数，且无需预先定义最大值。这些转移模型经过训练以匹配底层的局部生灭动力学，并在不同步长之间保持一致性。我们刻画了局部动力学与有限时间转移一致性之间的联系，并推导了生成误差的界。在包括高维、高计数设置在内的多个模拟中验证 Count Flow Map 之后，我们将其应用于单细胞药物（响应）……

    arXiv:2609.23290v1 Announce Type: cross  Abstract: High-dimensional count data are common in scientific applications, but most diffusion and flow models are designed for continuous or categorical data, and generation often requires many sequential model evaluations. We propose Count Flow Map, a generative model that learns finite-time transitions directly in count space for one- or few-step generation. Our model directly learns stochastic transitions over finite time intervals, using Poisson births and Binomial deaths to preserve nonnegative integer counts without a predefined maximum. These transition models are trained to match the underlying local birth--death dynamics and to maintain consistency across step sizes. We characterize the connection between local dynamics and finite-time transition consistency and derive a bound on the generation error. After validating Count Flow Map in several simulations, including a high-dimensional, high-count setting, we apply it to single-cell dr
    
[^59]: 需要多少后验样本？面向自适应感知的校准停止准则

    How Many Posterior Samples? Calibrated Stopping for Adaptive Sensing

    [https://arxiv.org/abs/2609.21813](https://arxiv.org/abs/2609.21813)

    论文揭示了自适应感知中基于票数份额阈值的即插即用停止规则并不提供置信度保证，并提出经校准的固定样本规则、有限视界序贯规则以及精确截断方法，以在规定错误声明概率下回答“需要多少后验样本”这一停止问题。

    

    在面向分类的自适应感知中，后验样本刻画了当前测量状态下的不确定性，并可发挥两种作用：它们可以引导下一次感知方向，同时其类别标签为候选类别提供“票数”，并决定是否应继续感知。我们聚焦于将这些票数转化为最终判决的停止层，而不修改后验采样器或感知方向。一种自然的即插即用规则在观测到的票数份额超过某个阈值时做出判决。我们证明该阈值本身并不构成置信度保证：当潜在票数质量恰好等于该阈值时，即插即用规则约有半数时间会做出判决。作为替代方案，我们将固定样本量规则和有限视界序贯规则校准到规定的错误声明概率，并研究精确截断方法，即在固定样本池规则的最终判决已被确定时提前停止。随后我们推导了单轮判决（摘要在此处被截断）

    arXiv:2609.21813v1 Announce Type: new  Abstract: In classification-oriented adaptive sensing, posterior samples characterize uncertainty at the current measurement state and can serve two roles: they may guide the next sensing direction, while their class labels provide votes for the candidate classes and determine whether sensing should continue. We focus on the stopping layer that turns these votes into a declaration, without modifying the posterior sampler or sensing directions. A natural plug-in rule declares when the observed vote share exceeds a threshold. We show that this threshold is not itself a confidence guarantee: when the underlying vote mass equals the threshold, the plug-in rule declares about half the time. As alternatives, we calibrate a fixed-sample rule and a finite-horizon sequential rule to a prescribed false-declaration probability, and study exact curtailment, which stops a fixed-pool rule once its final verdict is forced. We then derive how one-round declaratio
    
[^60]: ChorusTIC：通过合唱上下文学习实现无需训练的多变量时间序列分类

    ChorusTIC: Training-Free Multivariate Time Series Classification via Chorus In-Context Learning

    [https://arxiv.org/abs/2608.24033](https://arxiv.org/abs/2608.24033)

    ChorusTIC提出了一种无需训练的分类原生基础模型，通过情节一致的随机子通道拼接和双轴编码器，在异构通道配置下实现多变量时间序列的上下文分类，无需目标任务参数更新。

    

    时间序列分类支撑着医疗保健、传感和工业监控等应用。尽管时间序列基础模型支持预测和可迁移表示学习，但分类通常仍需要在每个目标数据集上拟合特定任务的分类器，而多变量输入的各个通道往往被独立编码。我们引入了ChorusTIC，一种分类原生基础模型，用于在异构通道配置下进行上下文分类，无需目标任务参数更新。ChorusTIC结合了情节一致的随机子通道槽拼接与共享双轴编码器，以建模时间与跨通道交互，并将可变通道配置映射为与原始通道数无关的固定宽度表示。然后，它使用上下文分布校准特征轴，并通过泄漏保护机制预测查询标签。

    arXiv:2608.24033v1 Announce Type: cross  Abstract: Time series classification underpins applications in healthcare, sensing, and industrial monitoring. Although time series foundation models support forecasting and transferable representation learning, classification still typically requires fitting a task-specific classifier on each target dataset, while individual channels of multivariate inputs are often encoded independently. We introduce ChorusTIC, a classification-native foundation model for in-context classification across heterogeneous channel configurations without target-task parameter updates. ChorusTIC combines episode-consistent Random Subchannel Slot Concatenation with a shared dual-axis encoder to model temporal and cross-channel interactions and map variable channel configurations into a fixed-width representation independent of the original channel count. It then calibrates feature axes using context-derived distributions and predicts query labels through leakage-prote
    
[^61]: 零流双样本检验

    Zero-Flow Two-Sample Tests

    [https://arxiv.org/abs/2607.21542](https://arxiv.org/abs/2607.21542)

    提出零流双样本检验（ZF2ST），通过可学习速度场的时间反演反对称性刻画分布相等性，利用见证函数的变分表示直接最大化检验功效，同时在留出样本上检验以保证第一类错误控制。

    

    受现代基于流的生成模型在建模复杂数据方面的成功启发，我们通过基于流的方法的视角来研究双样本检验问题。我们提出了零流双样本检验（ZF2ST），该检验建立在零流准则之上，该准则通过可学习速度场的时间反演反对称性来刻画分布相等性。我们扩展了这一准则，并进一步发展了零流差异，这是一种能够控制Wasserstein距离的识别性差异度量，并推导出以见证函数表达的变分表示形式。这一表示自然地引出了基于见证函数的检验方法，其检验功效由信噪比（SNR）决定，从而可以通过直接最大化功效来学习见证函数。ZF2ST在一个数据划分上学习见证函数，并在留出样本上进行检验，从而保持第一类错误的控制，并具有简单的渐近零分布。实验表明，ZF2ST…

    arXiv:2607.21542v2 Announce Type: replace-cross  Abstract: Motivated by the success of modern flow-based generative models in modeling complex data, we study two-sample testing through the lens of flow-based methods. We propose the Zero-Flow Two-Sample Test (ZF2ST), built on the zero-flow criterion, which characterizes distributional equality through a time-reversal antisymmetry of a learnable velocity field. We extend this criterion and further develop the Zero-Flow Discrepancy, an identifying discrepancy that controls the Wasserstein distance, and derive a variational representation in terms of a witness function. This representation naturally leads to a witness-based test whose power is governed by the signal-to-noise ratio (SNR), allowing direct power maximization for witness learning. ZF2ST learns the witness on one data split and performs testing on held-out samples, thereby maintaining Type-I error control and admitting a simple asymptotic null distribution. Experimentally, ZF2S
    
[^62]: 从残差架构的基本构件进行稳定性认证：一个锐利的稳定性阈值

    Certifying Residual Architectures from Their Primitives: A Sharp Stability Threshold

    [https://arxiv.org/abs/2607.14576](https://arxiv.org/abs/2607.14576)

    该论文提出一种在训练之前即可从残差块基本构件出发、以深度和浮点格式的显式函数形式认证残差架构训练稳定性的方法，通过前向幂律增长界与后向Lipschitz梯度界给出了状态能否达到最大有限浮点值的锐利稳定性阈值。

    

    深度残差架构能否稳定训练通常只能通过实际训练来确定，这种方式代价高昂，且只能回答所测试的特定架构、深度和浮点格式下的稳定性问题。在实践中，稳定性依赖于关于归一化放置位置及其类型选择的启发式规则，这些规则虽得到实验支持，但缺乏统一的原则。我们证明，稳定性可以在训练之前得到认证，即直接从残差块的架构基本构件出发，表示为深度和浮点格式的显式函数。该认证包含两个要素：（a）幂律增长界 $|v(x)|\le c|x|^q+b$，其指数 $q$ 通过一种指数算术由残差块的基本构件计算得出（前向）；（b）由残差块在可达状态上的 Lipschitz 条件导出的梯度界（反向）。该认证判定了状态何时能够达到最大的有限浮点值 $M$：当 $q\le（原文在此处截断）

    arXiv:2607.14576v2 Announce Type: replace-cross  Abstract: Whether a deep residual architecture trains stably is usually determined by training it, which is expensive and answers the question only for the architecture, depth, and floating-point format tested. In practice, stability is secured by heuristics for where to place normalization and which type to use, supported by experiments but lacking a common principle. We show that stability can instead be certified before training, directly from the architectural primitives of a residual block, as an explicit function of depth and floating-point format. The certificate has two elements: (a) a power-law growth bound, $|v(x)|\le c|x|^q+b$, whose exponent $q$ is computed from the block's primitives by an arithmetic of exponents (forward); and (b) gradient bounds from a Lipschitz condition on the block over reachable states (backward). The certificate determines when the state can reach the largest finite floating-point value $M$: for $q\le
    
[^63]: 从数据集谱几何到网络权重：一种用于图像分类的Sigmoid型多层感知机的几何感知初始化方法

    From Dataset Spectral Geometry to Network Weights: A Geometry-Aware Initialization for Sigmoidal MLPs in Image Classification

    [https://arxiv.org/abs/2606.28444](https://arxiv.org/abs/2606.28444)

    提出一种几何感知的初始化方法，通过对各类别训练样本进行SVD谱分析，将类别几何结构编码为成对的Sigmoid板状门控并直接编译进单隐层Sigmoid型MLP的权重中，从而实现基于数据几何的图像分类网络初始化。

    

    经典的通用逼近定理（UAT）确立了Sigmoid型多层感知机的表达能力，但它们并未说明权重应如何初始化。我们研究了一种有监督的、数据依赖的、几何感知的单隐层Sigmoid型MLP初始化方法，该方法将带标签的类别几何结构编译到网络权重中。这一构建始于这样一个思想：Sigmoid单元可以充当平滑的半空间门控。对于每个类别，我们将训练样本以其均值为中心，应用SVD来估计主方向和谱尺度，通过能量阈值选择需要保留的方向，并用一对Sigmoid板状门控来表示每个保留的方向。随后，这些类别特定的门控被拼接成一个直接从训练集初始化的共享隐藏层。我们还提出了一种SVD-马氏距离子空间分类器作为非神经网络的几何参考，用以检验所估计的谱……（摘要在此处截断）

    arXiv:2606.28444v2 Announce Type: replace-cross  Abstract: Classical universal approximation theorems (UAT) establish the expressive power of sigmoidal multilayer perceptrons, but they do not specify how the weights should be initialized. We study a supervised, data-dependent, geometry-aware initialization for one-hidden-layer sigmoidal MLPs that compiles labeled class geometry into network weights. The construction starts from the idea that sigmoid units can act as smooth half-space gates. For each class, we center the training samples at their mean, apply SVD to estimate principal directions and spectral scales, select retained directions by an energy threshold, and represent each retained direction by a pair of sigmoid slab gates. These class-specific gates are then concatenated into a shared hidden layer initialized directly from the training set. We also formulate a SVD-Mahalanobis subspace classifier as a non-neural geometric reference, which tests whether the estimated spectral 
    
[^64]: 面向函数空间回归与逆问题的流退火后验采样

    Flow Annealing Posterior Sampling for Function-Space Regression and Inverse Problems

    [https://arxiv.org/abs/2606.22346](https://arxiv.org/abs/2606.22346)

    提出了首个统一随机过程回归与偏微分方程逆问题的函数空间后验采样框架 FLAPS，其利用预训练流匹配先验与低秩朗之万校正，从稀疏含噪观测中生成连贯且不确定性校准良好的后验样本，显著优于现有方法。

    

    随机过程的原理化回归是一个长期存在的挑战，并与科学逆问题有着深刻的联系。我们提出了流退火后验采样（FLAPS），据我们所知，这是首个将随机过程回归与偏微分方程（PDE）逆问题统一起来的函数空间后验采样框架。FLAPS 基于预训练的函数空间流匹配先验构建，能够从稀疏且含噪的观测中进行似然引导的后验推断，支持可变的查询离散化，并避免了显式的先验密度评估。其朗之万校正采用低秩协方差预条件子，以利用不同离散化之间占主导地位的函数空间相关性。在高斯与非高斯随机过程回归基准测试以及多样化的偏微分方程逆问题上，FLAPS 生成了连贯的后验样本并具有良好的不确定性量化校准，显著优于现有的函数空间方法。

    arXiv:2606.22346v2 Announce Type: replace-cross  Abstract: Principled regression for stochastic processes is a long-standing challenge with deep connections to scientific inverse problems. We introduce Flow Annealing Posterior Sampling (FLAPS), to our knowledge the first function-space posterior sampling framework that unifies stochastic-process regression and PDE inverse problems. Built on pretrained function-space flow-matching priors, FLAPS enables likelihood-guided posterior inference from sparse and noisy observations, supports variable query discretizations, and avoids explicit prior-density evaluation. Its Langevin correction uses a low-rank covariance preconditioner to exploit dominant function-space correlations across discretizations. Across Gaussian and non-Gaussian stochastic-process regression benchmarks and diverse PDE inverse problems, FLAPS produces coherent posterior samples with well-calibrated uncertainty quantification, significantly outperforming existing functiona
    
[^65]: 弥合免模拟潜变量随机微分方程中的近似差距

    Closing the Approximation Gap in Simulation-free Latent SDEs

    [https://arxiv.org/abs/2606.16138](https://arxiv.org/abs/2606.16138)

    本文揭示现有免模拟变分推断算法虽高效，但其参数化将近似后验限制在SDE的一个子集中而产生近似差距，并提出方法弥合这一差距，兼得效率与表达能力。

    

    从含噪观测中恢复动力系统是神经科学和物理学等科学领域中反复出现的挑战。潜变量随机微分方程（SDEs）通过将系统建模为一个不可观测的隐状态来解决这一问题，该隐状态依据一个可学习的SDE演化并生成观测数据。变分推断（VI）为拟合潜变量SDE提供了一个易于处理的优化目标。传统VI算法通过在时间离散化上进行数值模拟来评估该目标，以保真度换取计算成本。最近出现的一类算法——免模拟VI——通过用瞬时边缘分布而非漂移项来参数化后验，规避了这种权衡。在这项工作中，我们证明了现有免模拟VI算法的效率是有代价的：其参数化方式将近似后验限制在基于模拟的方法可用的SDE范围的一个子集内，从而降低了（推断质量）。

    arXiv:2606.16138v2 Announce Type: replace  Abstract: Recovering dynamical systems from noisy observations is a recurring challenge across scientific domains, including neuroscience and physics. Latent stochastic differential equations (SDEs) address this by modeling the system as an unobserved state that evolves according to a learnable SDE and generates the observations. Variational inference (VI) provides a tractable objective for fitting latent SDEs. Traditional VI algorithms evaluate this objective by numerical simulation over a time discretization, trading fidelity for computational cost. A recent class of algorithms, simulation-free VI, sidesteps this tradeoff by parameterizing the posterior through its instantaneous marginals rather than its drift. In this work, we show that the efficiency of existing simulation-free VI algorithms comes at a price: their parameterizations restrict the approximate posterior to a subset of the SDEs available to simulation-based methods, degrading 
    
[^66]: 基于学习特征几何的非线性最小二乘泛化性研究

    Generalization in Nonlinear Least Squares via Learned Feature Geometry

    [https://arxiv.org/abs/2606.08799](https://arxiv.org/abs/2606.08799)

    该论文利用平均算法稳定性为岭正则化非线性最小二乘模型建立泛化误差界，核心创新是引入一个由训练后参数处的经验雅可比Gram矩阵与残差曲率项构成的数据依赖有效维度，并通过梯度特征的覆盖复杂度对其进行控制，从而使泛化保证取决于学习到的特征几何（如流形内在维度）而非参数数量。

    

    我们通过平均算法稳定性研究岭正则化非线性最小二乘模型的泛化性，针对局部极小值点推导了误差界。该误差界用一个数据依赖的有效维度来刻画，该有效维度通过经验雅可比Gram矩阵和残差曲率项，反映了训练后参数处梯度模型的几何结构。在线性情形下，曲率项消失，该结果恢复为雅可比核协方差的经典有效维度，但与神经正切核（NTK）分析中通常在初始化处评估不同，这里是在训练后的模型处进行评估。我们进一步通过梯度特征的覆盖复杂度来界定这一有效维度，从而得到依赖于学习到的几何结构而非参数数量的泛化保证。特别地，对于流形支撑的数据和分段Lipschitz雅可比矩阵，误差界随内在维度缩放；而对于单隐层ReLU网络……

    arXiv:2606.08799v3 Announce Type: replace  Abstract: We study the generalization of ridge-regularized nonlinear least-squares models via on-average algorithmic stability, deriving error bounds for local minimizers in terms of a data-dependent effective dimension that reflects the geometry of the gradient model at the trained parameters, through the empirical Jacobian Gram matrix and a residual-curvature term. In the linear case, where the curvature term vanishes, this recovers the classical effective dimension of the Jacobian kernel covariance, but evaluated at the trained model rather than at initialization as is typical in neural tangent kernel analyses. We further bound this effective dimension via covering complexity of the gradient features, leading to guarantees that depend on learned geometry rather than parameter count. In particular, for manifold-supported data and piecewise Lipschitz Jacobians, the bounds scale with intrinsic dimension, while for one-hidden-layer ReLU network
    
[^67]: 上下文动作集强化学习的更紧遗憾界

    Tighter Regret Bounds for Contextual Action-Set Reinforcement Learning

    [https://arxiv.org/abs/2605.15692](https://arxiv.org/abs/2605.15692)

    本文将MVP算法扩展到具有片段依赖可行动作集的上下文强化学习框架，建立了对抗上下文下 $\widetilde{O}(\sqrt{SAH^3K\log L})$ 的极小极大遗憾界，并由此导出随机上下文下的 $\widetilde{O}(\sqrt{SAH^3K})$ 遗憾界与 $\widetilde{O}(SAH^3/\epsilon^2)$ 的样本复杂度保证。

    

    我们研究片段式（episodic）强化学习问题，其中奖励函数和转移函数固定，但每个片段依赖的可行动作集会在片段开始时被观测到。性能通过相对逐片段最优值的累积遗憾来衡量，即 $\sum_{k=1}^K [V^{*,M^k} - V^{\pi^k,M^k}]$，其中 $M^k$ 表示第 k 个片段中的动作上下文。我们证明 MVP 算法可以自然地扩展到该框架，并具有强大的理论保证。特别地，我们针对对抗性上下文建立了 $\widetilde{O}(\sqrt{SAH^3K\log L})$ 的极小极大遗憾界，其中 L 表示可能的上下文数量。该结果意味着在随机上下文情形下遗憾界为 $\widetilde{O}(\sqrt{SAH^3K})$。我们进一步将随机遗憾保证转化为固定上下文分布下 $\widetilde{O}(SAH^3/\epsilon^2)$ 的样本复杂度界。此外，我们还推导了一个依赖间隙（gap-dependent）的……

    arXiv:2605.15692v2 Announce Type: replace-cross  Abstract: We study episodic reinforcement learning with fixed reward and transition functions, but with episode-dependent admissible action sets that are observed at the start of each episode. Performance is measured by cumulative regret against the episode-wise optimal value, $\sum_{k=1}^K [V^{*,M^k} - V^{\pi^k,M^k}]$, where $M^k$ represents the action context in the $k$-th episode. We show that the MVP algorithm naturally extends to this framework and enjoys strong theoretical guarantees. In particular, we establish a minimax regret bound of $\widetilde{O}(\sqrt{SAH^3K\log L})$ for adversarial contexts, where $L$ denotes the number of possible contexts. This result implies a regret bound of $\widetilde{O}(\sqrt{SAH^3K})$ for stochastic contexts. We further translate the stochastic regret guarantee into a sample complexity bound of $\widetilde{O}(SAH^3/\epsilon^2)$ for a fixed context distribution.   In addition, we derive a gap-depende
    
[^68]: 流形上的时间自适应无穷维高斯过程回归

    Time-adaptive infinite-dimensional Gaussian process regression on manifolds

    [https://arxiv.org/abs/2603.21144](https://arxiv.org/abs/2603.21144)

    本文基于经验贝叶斯方法提出了流形上泛函高斯过程回归的新框架，通过拉普拉斯-贝尔特拉米算子特征函数对应的时变角度谱及时间自适应截断方案实现了降维与有效预测。

    

    本文在时空随机场的背景下，基于经验贝叶斯方法，提出了流形上泛函高斯过程回归的一种新表述。我们应用可分希尔伯特空间中紧高斯测度的理论框架，并利用协方差核在流形等距变换群作用下的不变性。随后借助流形上拉普拉斯-贝尔特拉米算子的特征函数，通过特征函数方法将这些测度与一维高斯测度的无穷乘积建立了等同关系。其中涉及的随时间变化的角度谱构成了该回归方法实现中降维的关键工具，并采用了一种取决于泛函样本容量的合适截断方案。仿真研究和合成数据应用展示了所提出的泛函回归预测器的性能。

    arXiv:2603.21144v2 Announce Type: replace  Abstract: This paper proposes a new formulation of functional Gaussian Process regression on manifolds, based on an Empirical Bayes approach, in the spatiotemporal random field context. We apply the machinery of tight Gaussian measures in separable Hilbert spaces, exploiting the invariance property of covariance kernels under the group of isometries of the manifold. The identification via characteristic function of these measures with the infinite product of one-dimensional Gaussian measures is then obtained, in terms of the eigenfunctions of the Laplace-Beltrami operator on the manifold. The involved time-varying angular spectrum constitutes the key tool for dimension reduction in the implementation of this regression approach, adopting a suitable truncation scheme depending on the functional sample size. The simulation study and synthetic data application illustrate the performance of the proposed functional regression predictor.
    
[^69]: 用于最大熵强化学习的扩散增强马尔可夫决策过程

    Diffusion-Augmented Markov Decision Processes for Maximum Entropy Reinforcement Learning

    [https://arxiv.org/abs/2512.02019](https://arxiv.org/abs/2512.02019)

    该论文提出扩散增强马尔可夫决策过程（DA-MDPs），将反向扩散的每一步去噪转移视为独立的强化学习决策，并通过数据处理不等式推导出可分解的反向KL散度上界，为将最大熵强化学习算法推广到扩散策略提供了通用的理论框架。

    

    扩散模型为从复杂的非归一化分布中进行采样提供了一个富有表现力的框架。在这项工作中，我们通过引入扩散增强马尔可夫决策过程（DA-MDPs），将最大熵强化学习（ME-RL）扩展到基于扩散的策略。DA-MDPs 将每个反向扩散转移解释为一个独立的强化学习决策，而只有最终去噪后的动作才会在环境中执行。我们的 DA-MDPs 源于基于最大熵强化学习变分推断形式化表述的原则性推导。通过用中间扩散变量增强策略轨迹和目标轨迹，我们借助数据处理不等式得到了一个可处理的反向 KL 散度上界。该上界在各去噪转移之间可分解，从而产生了软奖励、价值函数和局部策略目标的扩散增强变体。这为将最大熵强化学习算法适配到扩散策略提供了一个通用框架。

    arXiv:2512.02019v4 Announce Type: replace-cross  Abstract: Diffusion models provide an expressive framework for sampling from complex, unnormalized distributions. In this work, we extend Maximum Entropy Reinforcement Learning (ME-RL) to diffusion-based policies by introducing Diffusion-Augmented Markov Decision Processes (DA-MDPs). DA-MDPs interpret each reverse-diffusion transition as an individual reinforcement-learning decision, while only the final denoised action is executed in the environment. Our DA-MDPs follow from a principled derivation based on the variational-inference formulation of ME-RL. By augmenting policy and target trajectories with intermediate diffusion variables, we obtain a tractable reverse-KL upper bound via the data-processing inequality. This bound decomposes across denoising transitions, yielding diffusion-augmented variants of soft rewards, value functions, and local policy objectives. This provides a general framework for adapting ME-RL algorithms to diffu
    
[^70]: 超越不确定性集合：利用最优传输将共形预测分布扩展至多元场景

    Beyond Uncertainty Sets: Leveraging Optimal Transport to Extend Conformal Predictive Distributions to Multivariate Settings

    [https://arxiv.org/abs/2511.15146](https://arxiv.org/abs/2511.15146)

    本文通过对最优传输分位数区域进行共形化，首次将共形预测分布扩展到多元/向量值分数场景，并恢复了有限样本、无分布假设的覆盖率保证。

    

    共形预测（CP）为模型输出构建具有有限样本覆盖率保证的不确定性集合。然而，只有当分数是标量值时，对分数进行排序才是直接可行的，这限制了CP只能应用于实值分数或临时性的一维降维方法。向量值分数在多输出回归和模型聚合中自然产生，例如集成学习中的每个预测器都提供各自的分数。最优传输（OT）定义了向量秩和由中心向外的多元分位数区域，但通常只具有渐近的覆盖率保证。将从校准数据中学习到的固定传输映射应用于新样本点会引入不受控的近似误差。我们通过对向量值的最优传输分位数区域进行共形化处理，恢复了有限样本、无分布假设的覆盖率保证。每个候选点的秩被定义为传输“增广了该候选分数的校准分数集合”，从而保持了有效性所需的对称性。本文……

    arXiv:2511.15146v2 Announce Type: replace  Abstract: Conformal prediction (CP) constructs uncertainty sets for model outputs with finite-sample coverage guarantees. Yet ranking scores is straightforward only when they are scalar-valued, limiting CP to real-valued scores or ad-hoc one-dimensional reductions. Vector-valued scores arise naturally in multi-output regression and model aggregation, where each predictor in an ensemble provides its own score. Optimal transport (OT) defines vector ranks and center-outward multivariate quantile regions, though generally with asymptotic coverage guarantees. Applying a fixed transport map learned from calibration data to a new point introduces an uncontrolled approximation error. We restore finite-sample, distribution-free coverage by conformalizing vector-valued OT quantile regions. Each candidate's rank is defined by transporting the calibration scores augmented with that candidate's score, preserving the symmetry needed for validity. This appea
    
[^71]: 大语言模型行为的序贯贝叶斯评估

    Sequential Bayesian Evaluation of Large Language Model Behavior

    [https://arxiv.org/abs/2511.10661](https://arxiv.org/abs/2511.10661)

    本文提出一种序贯贝叶斯评估框架，通过量化LLM随机性带来的评估不确定性，并自适应地优先选择下一个最值得评估的基准提示词，从而实现更具成本效益的大语言模型行为评估。

    

    评估基于大语言模型（LLM）的系统的特性正变得日益重要。此类评估通常依赖于向LLM提供的一组精心筛选的基准提示词（prompt）集合，其中每个提示词的输出可能被赋予二元或有序的分数，随后将各提示词的分数汇总作为总结性评估。在本文中，我们开发了一种贝叶斯方法，用于量化此类评估指标中由于基于LLM系统的随机性而产生的不确定性——即同一提示词在重复运行时可能表现出不同的结果。我们的框架自然地引出了一种序贯评估方法，即利用贝叶斯模型优先选择基准中接下来应使用哪些提示词，从而实现更具成本效益的LLM评估。我们通过四个案例研究展示了该方法：交互式对话中的LLM成对偏好评估（MT-Bench）、……

    arXiv:2511.10661v2 Announce Type: replace  Abstract: It is increasingly important to evaluate the characteristics of systems based on large language models (LLMs). Evaluations in this context often rely on a curated benchmark set of input prompts provided to the LLM, where the output for each prompt may be assigned a binary or ordinal score and the aggregation of scores across prompts is then used as a summary evaluation. In this paper, we develop a Bayesian approach for quantifying the uncertainty that arises in such evaluation metrics as a result of the stochasticity of the LLM-based systems; the same prompt may exhibit different outcomes on repeated runs. Our framework leads naturally to a sequential evaluation, in which we leverage the Bayesian model to preferentially select which prompts in the benchmark to use next, enabling more cost-effective LLM evaluations. We demonstrate this approach through four case studies: pairwise LLM preferences in interactive dialogue (MT-Bench), ref
    
[^72]: 可迁移生成模型架起飞秒到纳秒时间步长分子动力学的桥梁

    Transferable Generative Models Bridge Femtosecond to Nanosecond Time-Step Molecular Dynamics

    [https://arxiv.org/abs/2510.07589](https://arxiv.org/abs/2510.07589)

    该研究提出一种可迁移的深度生成建模框架，在保持物理真实性的同时将分子动力学采样加速四个数量级，并能跨化学组成和体系规模泛化，从而实现对此前难以企及的平衡系综与慢速弛豫动力学过程的定量表征。

    

    理解分子结构、动力学与反应活性需要跨越相距悬殊的时间尺度上的过程。传统分子动力学模拟可提供原子级分辨率，但其飞秒级时间步长限制了对决定化学功能的慢速构象变化与弛豫过程的探索。在此，我们提出了一种深度生成建模框架，能够在保持物理真实性的同时，将分子动力学采样加速四个数量级。将该框架应用于小有机分子和多肽，可对平衡系综与动力学弛豫过程进行定量表征，而这些过程此前只能通过代价高昂的暴力模拟才能获得。重要的是，该方法能够在不同化学组成和体系规模之间泛化，可外推到比训练集更大的多肽体系，并能捕捉具有化学意义的转变过程……

    arXiv:2510.07589v2 Announce Type: replace-cross  Abstract: Understanding molecular structure, dynamics, and reactivity requires bridging processes that occur across widely separated time scales. Conventional molecular dynamics simulations provide atomistic resolution, but their femtosecond time steps limit access to the slow conformational changes and relaxation processes that govern chemical function. Here, we introduce a deep generative modeling framework that accelerates sampling of molecular dynamics by four orders of magnitude while retaining physical realism. Applied to small organic molecules and peptides, the approach enables quantitative characterization of equilibrium ensembles and dynamical relaxation processes that were previously only accessible by costly brute-force simulation. Importantly, the method generalizes across chemical composition and system size, extrapolating to peptides larger than those used for training, and captures chemically meaningful transitions on ext
    
[^73]: 线性元学习模型中几何任务多样性的结构性分配的影响

    Effects of Structural Allocation of Geometric Task Diversity in Linear Meta-Learning Models

    [https://arxiv.org/abs/2509.18349](https://arxiv.org/abs/2509.18349)

    本研究证明了元学习性能不仅取决于任务参数的整体几何多样性，还取决于这种多样性相对于底层低维结构的分配方式，从而深化了对任务多样性影响元学习机制的理解。

    

    元学习旨在利用相关任务之间的信息，在仅有少量标注观测数据（即“少样本”学习）的情况下，提高对新任务未标注数据的预测能力。人们通常认为，增加任务多样性可以通过提供跨任务的更丰富信息来增强元学习效果。然而，Kumar等人（2022）最近的研究表明，通过任务表示的整体几何散布来量化的任务多样性增加，实际上可能会在一系列模型和数据集上降低元学习的预测性能。在这项工作中，我们基于这一观察进一步证明，元学习性能不仅受任务参数整体几何变异性的影响，还受这种变异性相对于潜在低维结构的分配方式的影响。与Pimonova等人（2025）类似，我们将特定于任务的回归效应分解为具有结构信息的部分……

    arXiv:2509.18349v4 Announce Type: replace  Abstract: Meta-learning aims to leverage information across related tasks to improve prediction on unlabeled data for new tasks when only a small number of labeled observations are available ("few-shot" learning). Increased task diversity is often believed to enhance meta-learning by providing richer information across tasks. However, recent work by Kumar et al. (2022) shows that increasing task diversity, quantified through the overall geometric spread of task representations, can in fact degrade meta-learning prediction performance across a range of models and datasets. In this work, we build on this observation by showing that meta-learning performance is affected not only by the overall geometric variability of task parameters, but also by how this variability is allocated relative to an underlying low-dimensional structure. Similar to Pimonova et al. (2025), we decompose task-specific regression effects into a structurally informative com
    
[^74]: 可微分期望最大化算法及其在高斯混合模型最优传输中的应用

    Differentiable Expectation-Maximisation and Applications to Gaussian Mixture Model Optimal Transport

    [https://arxiv.org/abs/2509.02109](https://arxiv.org/abs/2509.02109)

    本文提出了多种使EM算法可微分的方法，并将其应用于计算高斯混合模型间的混合Wasserstein距离，使MW2可作为可微分损失用于成像和机器学习任务，并提供了新颖的稳定性理论保证。

    

    期望最大化（EM）算法是统计学和机器学习中的核心工具，被广泛应用于高斯混合模型（GMM）等潜变量模型。尽管EM算法无处不在，但它通常被视为一个不可微分的黑箱，这阻碍了其集成到需要端到端梯度传播的现代学习流水线中。在本工作中，我们提出并比较了多种针对EM算法的微分策略，从完全自动微分到近似方法，并评估了它们的精度和计算效率。作为一项关键应用，我们利用这种可微分EM算法来计算高斯混合模型之间的混合Wasserstein距离 $\mathrm{MW}_2$，使 $\mathrm{MW}_2$ 能够作为可微分损失函数用于成像和机器学习任务中。作为对 $\mathrm{MW}_2$ 实际应用的补充，我们提出了一种新颖的稳定性结果，为其提供了理论依据。

    arXiv:2509.02109v3 Announce Type: replace-cross  Abstract: The Expectation-Maximisation (EM) algorithm is a central tool in statistics and machine learning, widely used for latent-variable models such as Gaussian Mixture Models (GMMs). Despite its ubiquity, EM is typically treated as a non-differentiable black box, preventing its integration into modern learning pipelines where end-to-end gradient propagation is essential. In this work, we present and compare several differentiation strategies for EM, from full automatic differentiation to approximate methods, assessing their accuracy and computational efficiency. As a key application, we leverage this differentiable EM in the computation of the Mixture Wasserstein distance $\mathrm{MW}_2$ between GMMs, allowing $\mathrm{MW}_2$ to be used as a differentiable loss in imaging and machine learning tasks. To complement our practical use of $\mathrm{MW}_2$, we contribute a novel stability result which provides theoretical justification for 
    
[^75]: 网络值数据的重心子空间分析

    Barycentric subspace analysis of network-valued data

    [https://arxiv.org/abs/2507.23559](https://arxiv.org/abs/2507.23559)

    本文提出重心子空间分析（BSA）方法，用样本点而非向量来生成降维子空间，从而提升了对无标注网络值数据进行探索性分析时的可解释性。

    

    某些数据天然地可以用网络或加权图来建模，例如生物网络或流动性网络。当数据集中各网络之间没有规范的节点标注时，我们将其称为无标注网络。本文聚焦于对这类数据进行探索性分析的问题。更具体地说，我们着手解决如何解释由降维方法构建的特征子空间这一难题。现有大多数面向网络值数据的方法都源自主成分分析（PCA），因此依赖于由一组向量生成的子空间，我们将这一点视为可解释性方面的主要局限。与此不同，我们提出了一种称为重心子空间分析（BSA）的方法，该方法依赖于由一组点生成的子空间，在实践中我们选择这些点为样本。为了给 BSA 提供一个计算上可行的框架，我们引入了一种新颖的嵌入方法……

    arXiv:2507.23559v2 Announce Type: replace-cross  Abstract: Certain data are naturally modeled by networks or weighted graphs, be they biological networks or mobility networks. When there is no canonical labeling of the nodes across the dataset, we talk about unlabeled networks. In this paper, we focus on the question of exploratory analysis of this type of data. More specifically, we address the issue of interpreting the feature subspace constructed by dimensionality reduction methods. Most existing methods for network-valued data are derived from principal component analysis (PCA) and therefore rely on subspaces generated by a set of vectors, which we identify as a major limitation in terms of interpretability. Instead, we propose to implement the method called barycentric subspace analysis (BSA), which relies on subspaces generated by a set of points, which we choose, in practice, to be samples. In order to provide a computationally feasible framework for BSA, we introduce a novel em
    
[^76]: 针对连续有界结果的保形化回归

    Conformalized Regression for Continuous Bounded Outcomes

    [https://arxiv.org/abs/2507.14023](https://arxiv.org/abs/2507.14023)

    本文在变换回归模型框架下为连续有界结果提出保形预测区间方法，通过基于模型残差的分位数-残差非一致性分数，同时处理数据的异方差性和边界附近的不对称性，并连接了归一化保形预测与分布保形预测。

    

    在统计学和机器学习应用中，例如比率和比例的分析，经常会遇到具有连续有界结果的回归问题。在这一设定下，一个核心挑战是在新的协变量取值处预测响应变量。现有文献大多集中于点预测，或基于渐近近似的区间预测。我们在变换回归模型的框架内，为有界结果开发了保形预测区间，其中涵盖了贝塔回归和logit-正态回归等广泛使用的模型。我们基于与模型相一致的残差构建了非一致性分数，并识别出一种特别适合有界结果的分位数-残差分数，该分数连接了归一化保形预测与分布保形预测。这一分数既考虑了此类数据中固有的异方差性，也考虑了在接近边界处所出现的不对称性。

    arXiv:2507.14023v3 Announce Type: replace  Abstract: Regression problems with continuous bounded outcomes frequently arise in statistical and machine learning applications, such as the analysis of rates and proportions. A central challenge in this setting is predicting the response at a new covariate value. Most of the existing literature has focused either on point prediction or on interval prediction based on asymptotic approximations. We develop conformal prediction intervals for bounded outcomes within the framework of transformation regression models, encompassing widely used models such as beta regression and logit-normal regression. We construct non-conformity scores based on model-aligned residuals and identify a quantile-residual score that is particularly well suited to bounded outcomes, bridging normalized conformal prediction and distributional conformal prediction. This score accounts for both the heteroscedasticity inherent in such data and the asymmetry that emerges near
    
[^77]: 将生成式学习引入表示学习：作为分布匹配的自监督迁移学习

    Bringing Generative Learning to Representation Learning: Self-Supervised Transfer Learning as Distribution Matching

    [https://arxiv.org/abs/2502.14424](https://arxiv.org/abs/2502.14424)

    本文提出将表示学习重新定义为分布匹配，通过匹配显式几何参考分布来学习增强不变的编码器，从而实现自监督迁移学习，并证明了其理论保证和实际效果。

    

    arXiv:2502.14424v3 公告类型：替换交叉 摘要：大多数自监督学习目标旨在防止表示坍缩，但未明确目标表示规律。我们将表示学习形式化为分布匹配（DM），学习一个增强不变的编码器，其诱导的分布规律与一个明确的几何参考匹配。参考规律指定了学习到的表示分布应呈现的形式，而单独选择的差异度量则用于衡量与该目标的偏差；此处我们使用马氏距离。DM框架揭示了一个方向性反转：生成式学习将可处理的参考映射到数据，而表示学习则将数据映射到设计的参考规律。我们将总体目标与类中心分离和分类误差联系起来，并证明了非渐近神经筛保证。模拟和图像基准测试显示了流形校正、细粒度结构和跨标签空间的迁移能力。

    arXiv:2502.14424v3 Announce Type: replace-cross  Abstract: Most self-supervised learning objectives defend against collapse but leave the target representation law unspecified. We formulate representation learning as Distribution Matching (DM), learning an augmentation-invariant encoder whose induced law matches an explicit geometric reference. The reference law specifies what the learned representation distribution should look like, whereas a separately chosen discrepancy determines how deviations from this target are measured; here we use Mallows distance. The DM framework reveals a directional inverse: generative learning maps a tractable reference to data, whereas representation learning maps data to a designed reference law. We connect the population objective to class-centre separation and classification error and prove a non-asymptotic neural-sieve guarantee. Simulations and image benchmarks show manifold rectification, fine-grained structure and transfer across label spaces.
    
[^78]: 去中心化无投影在线上线性化优化及其在DR-次模优化中的应用

    Decentralized Projection-free Online Upper-Linearizable Optimization with Applications to DR-Submodular Optimization

    [https://arxiv.org/abs/2501.18183](https://arxiv.org/abs/2501.18183)

    该论文提出了一个去中心化无投影在线优化框架，将无投影方法推广到上线性化函数（涵盖DR-次模函数），在任意参数0≤θ≤1下实现O(T^{1-θ/2})遗憾值、O(T^{θ})通信复杂度和O(T^{2θ})次预言机调用，并首次给出一般凸约束下单调及非单调上凹优化的结果，且将结论扩展至零阶、半老虎机和老虎机反馈。

    

    我们提出了一种新的去中心化无投影优化框架，将无投影方法扩展到更广泛的一类上线性化函数。我们的方法结合了去中心化优化技术与上线性化函数框架的灵活性，有效地推广了传统的DR-次模函数优化。对于任意0≤θ≤1，在去中心化上线性化函数优化中，我们获得了O(T^{1-θ/2})的遗憾值，通信复杂度为O(T^{θ})，线性优化预言机调用次数为O(T^{2θ})。这一方法首次给出了一般凸约束下单调上凹优化和非单调上凹优化的结果。此外，上述针对一阶反馈的结果还被扩展到零阶、半老虎机和老虎机反馈情形。

    arXiv:2501.18183v3 Announce Type: replace-cross  Abstract: We introduce a novel framework for decentralized projection-free optimization, extending projection-free methods to a broader class of upper-linearizable functions. Our approach leverages decentralized optimization techniques with the flexibility of upper-linearizable function frameworks, effectively generalizing traditional DR-submodular function optimization. We obtain the regret of $O(T^{1-\theta/2})$ with communication complexity of $O(T^{\theta})$ and number of linear optimization oracle calls of $O(T^{2\theta})$ for decentralized upper-linearizable function optimization, for any $0\le \theta \le 1$. This approach allows for the first results for monotone up-concave optimization with general convex constraints and non-monotone up-concave optimization with general convex constraints. Further, the above results for first order feedback are extended to zeroth order, semi-bandit, and bandit feedback.
    
[^79]: 挖掘因果关系：AI辅助搜索工具变量

    Mining Causality: AI-Assisted Search for Instrumental Variables

    [https://arxiv.org/abs/2409.14202](https://arxiv.org/abs/2409.14202)

    本文提出利用大语言模型通过叙事和反事实推理自动搜索新颖有效的工具变量，大幅加速因果推断中工具变量的发现过程。

    

    工具变量（IVs）方法是因果推断中最主要的实证策略之一。寻找工具变量是一个启发式和创造性的过程，而论证其有效性——尤其是排他性约束——在很大程度上依赖于修辞论证。我们提出使用大型语言模型（LLMs）通过叙述和反事实推理来搜索新的工具变量，类似于人类研究者的思考方式。然而，其显著的差异在于，LLMs可以极大地加速这一过程，并探索极其庞大的搜索空间。我们引入了一个发现流程，采用两步提示和角色扮演提示策略来搜索潜在新颖且有效的工具变量。我们认为，这些策略模拟了经济主体和社会行为者的内生决策过程，并将语言模型置于真实世界的场景中，从而掩盖了工具变量发现任务本身。我们将该方法应用于经济学三个经典领域：教育回报……

    arXiv:2409.14202v4 Announce Type: replace  Abstract: The instrumental variables (IVs) method is a leading empirical strategy for causal inference. Finding IVs is a heuristic and creative process, and justifying their validity---especially exclusion restrictions---is largely rhetorical. We propose using large language models (LLMs) to search for new IVs through narratives and counterfactual reasoning, similar to how a human researcher would. The stark difference, however, is that LLMs can dramatically accelerate this process and explore an extremely large search space. We introduce a discovery pipeline that searches for potentially novel and valid IVs using two-step and role-playing prompting strategies. We contend that these strategies simulate the endogenous decision-making of economic agents and social actors and ground language models in real-world scenarios, thereby masking the IV discovery task itself. We apply our method to three canonical areas in economics: returns to schooling
    
[^80]: 可能非线性因子模型中的因果推断

    Causal Inference in Possibly Nonlinear Factor Models

    [https://arxiv.org/abs/2008.13651](https://arxiv.org/abs/2008.13651)

    本文针对含有噪声测量混杂因素的处理效应模型提出了一种因果推断方法，利用结合K近邻匹配与主成分分析的局部主子空间逼近程序以及双重稳健得分函数，在代理变量与潜在混杂因素之间存在未知非线性因子结构的情况下实现了对多种因果参数的有效估计。

    

    本文针对带有噪声测量混杂因素的处理效应模型开发了一种因果推断方法。其关键特征是可以获得大量含噪声的代理变量，这些代理变量通过一种未知的、可能非线性的因子结构与潜在的混杂因素相关联。该方法的主要构建模块是一种结合了K近邻匹配与主成分分析的局部主子空间逼近程序。基于双重稳健的得分函数，构建了包括平均处理效应和反事实分布在内的多种因果参数的估计量，并建立了这些估计量的大样本性质。这些结果要求所收集的代理变量集合能够联合地提供关于与结果和处理相关的潜在混杂因素的信息，同时允许部分代理变量不携带信息、其身份未知，以及测量误差与结果或处理相关。

    arXiv:2008.13651v4 Announce Type: replace  Abstract: This paper develops a causal inference method for treatment effects models with noisily measured confounders. The key feature is that a large number of noisy proxies are available and linked with the underlying latent confounders through an unknown, possibly nonlinear factor structure. The main building block is a local principal subspace approximation procedure that combines K-nearest-neighbor matching and principal component analysis. Estimators of many causal parameters, including average treatment effects and counterfactual distributions, are constructed based on doubly-robust score functions, and their large-sample properties are established. These results require the collection of proxies to be jointly informative about the latent confounders relevant to the outcome and treatment, while allowing some proxies to be uninformative, their identities to be unknown, and measurement errors to be correlated with the outcome or treatmen
    
[^81]: 受限分类和策略学习

    Constrained Classification and Policy Learning. (arXiv:2106.12886v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2106.12886](http://arxiv.org/abs/2106.12886)

    研究了受限分类和策略学习中替代损失程序的一致性和适用性。

    

    现代机器学习方法对于分类问题使用了一些替代损失技术，如AdaBoost、支持向量机和深度神经网络，以绕过最小化经验分类风险的计算复杂性。这些技术在因果策略学习问题中也很有用，因为个性化治疗规则的估计可以被视为一种加权（成本敏感）分类问题。Zhang（2004年）和Bartlett等人（2006年）研究的替代损失方法的一致性关键依赖于正确规范的假设，即指定的分类器集合足够丰富，包含一个最佳分类器。然而，当分类器集合受到可解释性或公平性的限制时，这个假设较不可靠，这导致在这种次佳情景下替代损失方法的适用性未知。本文研究了在受限类集合条件下的替代损失程序的一致性。

    Modern machine learning approaches to classification, including AdaBoost, support vector machines, and deep neural networks, utilize surrogate loss techniques to circumvent the computational complexity of minimizing empirical classification risk. These techniques are also useful for causal policy learning problems, since estimation of individualized treatment rules can be cast as a weighted (cost-sensitive) classification problem. Consistency of the surrogate loss approaches studied in Zhang (2004) and Bartlett et al. (2006) crucially relies on the assumption of correct specification, meaning that the specified set of classifiers is rich enough to contain a first-best classifier. This assumption is, however, less credible when the set of classifiers is constrained by interpretability or fairness, leaving the applicability of surrogate loss based algorithms unknown in such second-best scenarios. This paper studies consistency of surrogate loss procedures under a constrained set of class
    

