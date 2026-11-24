# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AlphaDefect: Expert-Level AI for Slay the Spire 2](https://arxiv.org/abs/2611.31076) | AlphaDefect通过PPO训练单一神经网络控制全部对局决策，在《杀戮尖塔2》最高难度下取得62-80%的胜率，远超人类玩家11-20%的水平。 |
| [^2] | [RoFE: Rotated Frozen Embeddings for Low-Data Multimodal Tabular Learning](https://arxiv.org/abs/2611.31049) | 该论文揭示了表格基础模型（PFN）中按列标准化会消除PCA压缩嵌入的方差、导致标注数据稀缺时PCA接口失效甚至不如线性分类器的失败机制，并提出RoFE（旋转冻结嵌入）作为低数据多模态表格学习的更优接口。 |
| [^3] | [Revealing Geometry under Strong Noise: Spectral Embeddings and Clustering via Random Geometric Graphs](https://arxiv.org/abs/2611.31047) | 本文提出一种先对成对距离去噪再构建随机几何图进行谱嵌入与聚类的新方法，即使在强噪声和群体中心重合（如同心形状）的情况下也能有效揭示数据的潜在低维几何结构并区分不同群体。 |
| [^4] | [Prediction-Powered Regression via Calibration: Beyond the Supervised Minimax Benchmark](https://arxiv.org/abs/2611.30976) | 该论文提出基于校准的加权预测驱动回归方法 wPPRC，通过用标注数据校准伪标签并进行加权回归，在理论上证明其最坏情形风险可以超越有监督极小极大基准，从而体现大量伪标签带来的额外价值。 |
| [^5] | [$K$-Midranks: A robust clustering algorithm for heavy-tailed data with applications to spectroscopy and finance](https://arxiv.org/abs/2611.30953) | 本文提出了一种名为 $K$-中位秩的鲁棒聚类算法，通过将观测值替换为逐坐标秩来降低重尾数据中极端值的影响，在柯西型重尾数据上优于传统聚类方法。 |
| [^6] | [Aggregation Rules Induced by KL-Regularised Preference Dynamics on Networks](https://arxiv.org/abs/2611.30941) | 论文证明网络中多个智能体进行KL正则化偏好对齐的迭代动态会诱导出一种加权聚合规则，权重由各智能体在网络中的长期注意力份额除以其温度参数决定，并在参考项或挑战者被聚合时分别导出基数型聚合与策略分化等结果。 |
| [^7] | [A theory of inductive node-level generalization from a single large graph](https://arxiv.org/abs/2611.30937) | 本文提出了一个新理论框架，用于刻画图机器学习算法在仅从单个大图学习时进行归纳式节点级任务的泛化能力。 |
| [^8] | [Fast and Consistent Structure Learning in Graphical Models via Approximate Cross-Validation](https://arxiv.org/abs/2611.30822) | 本文提出基于近似交叉验证的可扩展图结构学习方法，解决了交叉验证在结构学习中计算不可行且理论不一致的问题，并在温和的信号强度假设下实现了选择一致性。 |
| [^9] | [Precise Asymptotics of Non-Convex M-Estimators](https://arxiv.org/abs/2611.30798) | 该论文突破了现有状态演化理论要求近邻算子Lipschitz连续的限制，通过修正Onsager项以处理非凸损失函数或正则化项导致的近邻算子不连续问题，从而建立了非凸M-估计量在高维广义线性模型中的精确渐近理论。 |
| [^10] | [MTL-MHOL: Predicting Online Conversions under Delayed Feedback and Data Sparsity](https://arxiv.org/abs/2611.30648) | 本文提出模型无关的MTL-MHOL框架，通过时间分桶处理延迟反馈并结合辅助任务的多任务学习来缓解数据稀疏，在CVR预测中达到或超越所有基准模型。 |
| [^11] | [What Environment Invariance Identifies: Minimal Invariant Sets in Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/2611.30609) | 本文将不变因果预测的不变性准则提升到再生核希尔伯特空间中条件均值嵌入的算子层面，精确证明了不变集恰好是包含 PA(Y) 且不含 Y 后代的集合，从而确立 PA(Y) 是唯一的极小不变集，并给出有限样本下恢复所需的样本量刻画。 |
| [^12] | [SCAN: Neural Additive Networks for Scalable High-Order Component Selection and Estimation](https://arxiv.org/abs/2611.30452) | SCAN通过让每个基函数自动学习组合哪些特征，无需预先枚举交互项即可联合选择与估计不同阶数的成分，并具备万能逼近和极小极大最优速率等理论保证。 |
| [^13] | [Does coverage explain core-set active learning? Mean, max, and the choice of distance](https://arxiv.org/abs/2611.30450) | 本文证明核心集主动学习的覆盖泛化界对任意不相似度度量和任何不使用样本外标签的选择规则均成立，且在标注点标签确定时无需零损失假设、去除了类别数因子 C，仅依赖池到标注样本的平均距离与单元纯度。 |
| [^14] | [Spectral Regularization Beyond Ridge Regression](https://arxiv.org/abs/2611.30440) | 本文提出包含岭回归与迭代Tikhonov正则化的双参数谱正则化族“包容岭回归”，精确刻画其有限样本风险，证明神谕调优的岭回归在该族中通常次优，并给出高阶估计量与指数极限滤波器严格优于岭回归的明确条件。 |
| [^15] | [Tensor Recovery with Heterogeneous Mode Structures](https://arxiv.org/abs/2611.30435) | 提出了一个针对异质模式结构的统一凸张量恢复框架，对平滑模式采用局部核加权、对其他模式分别施加低秩、组稀疏、图拉普拉斯或自回归正则化，配合共识式 HM-ADMM 算法求解，并给出了 oracle 不等式与高概率误差界的理论保证。 |
| [^16] | [Low-Rank Tensor Representations for Efficient Multimarginal Optimal Transport](https://arxiv.org/abs/2611.30376) | 该论文提出低秩多边际最优传输（LRMOT），通过将传输计划限制为有界多重线性秩的张量，大幅降低了多边际最优传输的计算与内存开销，同时保持较低的传输成本。 |
| [^17] | [Boosting zero-shot learning through neuro-symbolic integration](https://arxiv.org/abs/2611.30358) | 本文提出模糊逻辑原型网络（FLPN），利用逻辑张量网络框架将逻辑公理形式的背景知识注入训练过程，通过优化神经符号损失在视觉-语义嵌入空间中进行原型匹配，从而提升零样本学习能力。 |
| [^18] | [How long must an honest confidence interval be after causal discovery? Benign and malignant weak edges](https://arxiv.org/abs/2611.30256) | 该论文通过 Meek 闭包将弱边分为“良性”与“恶性”两类，刻画了因果发现后诚实置信区间相对已知图最优区间的最坏长度膨胀代价：良性边处代价有界（当跨模型常数 K=0 时甚至等于 1，即诚实性免费），而一旦存在恶性边，代价便达到 √n 量级。 |
| [^19] | [Operator-Valued Q-Learning: Compositional Bellman Recursions for Order-Sensitive Decision Making](https://arxiv.org/abs/2611.30255) | 该论文提出算子值Q学习，将Q值表示为矩阵算子，通过保持不可交换动作算子有序乘积的贝尔曼递归，实现对顺序敏感决策问题的精确价值估计，并给出压缩性保证与几乎必然收敛的异步算子TD学习方案。 |
| [^20] | [Revisiting Continual Learning in Language Models: A Three-Interface Diagnosis of Stability and Plasticity](https://arxiv.org/abs/2611.30151) | 该论文提出应通过已训练头部、冻结特征探针和无分类器主干诊断这三种评估接口来定位持续学习遗忘的发生位置（主干表征退化还是读出头失配），并主张BWT与FWT是依赖于接口的量而非单一的模型级属性。 |
| [^21] | [Sparse Hodge Preference Optimization: Fine-Tuning Language Models toward Alignment beyond Transitive Preferences](https://arxiv.org/abs/2611.30135) | 本文提出稀疏霍奇偏好优化（SHPO），借助组合霍奇分解理论为DPO引入稀疏反对称旋度残差，使其能够捕捉人类偏好数据中的非传递性与循环结构，并证明其严格扩大了DPO的假设类。 |
| [^22] | [PsyDrift: Healthy-Reference Latent Diffusion for Anomaly Scoring in Autism Spectrum Disorder](https://arxiv.org/abs/2611.30050) | PsyDrift通过仅从健康对照数据学习的潜在扩散模型，将功能连接表征与去噪重建的偏离转化为连续异常评分，在ABIDE-I全部20个站点跨站点验证中实现自闭症检测平均AUROC达0.798。 |
| [^23] | [The response field approach to path integral control](https://arxiv.org/abs/2611.30044) | 该论文开创性地将非平衡统计物理中的响应场形式体系引入路径积分最优控制，建立了响应场与随机最大值原理中后向伴随过程的对应关系，并证明最优控制由初始时刻响应场的平均值给出。 |
| [^24] | [Sequential Selection against an Adaptive Opponent](https://arxiv.org/abs/2611.30006) | 本文提出SPAR规则，通过与自适应对手的序贯练习对局累积证据并在私有随机阈值处停止，从而在极少误选已失效策略的前提下选出满足目标胜率的策略。 |
| [^25] | [Training Dynamics of Large State-Transition Models](https://arxiv.org/abs/2611.30005) | 本文研究了在嵌入维度小于词表规模的条件下，从随机初始化出发联合训练词元嵌入与模型参数的训练动力学，揭示了大型状态转移模型实现准确下一词元预测的条件。 |
| [^26] | [Statistically Optimal and Near Linear Time Hypothesis Selection](https://arxiv.org/abs/2611.30003) | 本文提出了一种假设选择学习器，在期望意义下近线性运行时间、仅需 $O(\log(n/\delta)/\epsilon^2)$ 个样本即可达到统计最优的因子2近似，同时去除了Bousquet等人（FOCS 2021）因子2界中的对数因子。 |
| [^27] | [Wavelet determinantal samplers: Optimal approximation rates adapting to regularity](https://arxiv.org/abs/2611.29992) | 该论文构造了基于小波的行列式点过程采样器，在流形与几何图上对线性统计量实现了 n^{1-2s/m} 阶的方差缩减，且该逼近速率能够自适应于观测量的 Hölder/Sobolev 正则性。 |
| [^28] | [Sharp Asymptotic Theory of Maximum Likelihood Estimation for Gaussian Processes with a Matern Kernel](https://arxiv.org/abs/2611.29841) | 本文为Matérn核高斯过程参数的联合极大似然估计建立了精确的渐近理论，刻画了不同观测域维度下可估计的参数及其收敛速度，弥补了从参数可辨识性到估计量渐近性质之间的理论空白。 |
| [^29] | [Tabular Foundation Models for Weighted Regression and Classification](https://arxiv.org/abs/2611.29789) | 提出了 WeightedPFN，通过加权注意力、归一化加权损失以及基于缺失数据问题的合成加权任务预训练，将表格基础模型 TabPFN-3.5 适配到加权回归与分类中，在真实数据集上表现优异。 |
| [^30] | [Constrained Diffusion from Binary Evaluations via Learned Doob h-Functions](https://arxiv.org/abs/2611.29779) | 该论文提出基于 Doob h-变换的约束扩散生成框架，仅用二元样本评估学习“含噪样本去噪后满足约束的概率”这一单一函数，无需可微测量算子、成对数据或逐步 rollout，即可在冻结基础模型上实现带逆概率因子的精确约束引导。 |
| [^31] | [Selecting Stable Sampling Dynamics for Compositional Diffusion: A Feynman-Kac Weight Reduction](https://arxiv.org/abs/2611.29764) | 该论文在Feynman-Kac加权扩散框架中确定λ=1/B（B为各专家权重之和）这一特殊取值，此时瞬时对数权重中的聚合平方分数项恰好完全抵消，从而为组合扩散采样选出最稳定的提议动力学。 |
| [^32] | [probly: Uncertainty-Aware Machine Learning](https://arxiv.org/abs/2611.29762) | probly是一个模块化Python包，通过轻量级转换将现有模型（支持PyTorch、scikit-learn和Flax）变为不确定性感知模型，并在统一框架内集成二阶分布、信度集合和保形预测等多种不确定性表示与量化方法。 |
| [^33] | [Grammar Sparsity Sets How Transformers Learn Random Languages](https://arxiv.org/abs/2611.29692) | 语法稀疏性同时决定了Transformer学习随机语言的顺序和速度——稠密语法分层学习且经历随词表大小变宽的平台期（$t\propto V$ 到 $t\propto V^3$），而稀疏语法则一次性快速习得（$t\propto\sqrt V$）。 |
| [^34] | [How Multi-Timescale Memory Cools Optimization: Theory and Evidence from AdEMAMix](https://arxiv.org/abs/2611.29588) | 该论文揭示多时间尺度优化器记忆通过改变参数范数来控制参数旋转速度，证明AdEMAMix中独立加权的慢速动量本质上充当一种内生的退火调度，从而解释了其在语言模型预训练中带来收益的机制。 |
| [^35] | [From Learning Rates to Rotational Schedules: Finite-Horizon Theory on the Sphere](https://arxiv.org/abs/2611.29579) | 该论文将学习率调度设计重新表述为球面上的旋转调度问题，推导了任意旋转调度的最后一次迭代收敛保证，证明最优调度当时域增长时趋近于线性衰减，为大语言模型预训练实践提供了有用的理论指导。 |
| [^36] | [Gated Payoff-Weighted Net Profit Maximization For Online Portfolio Selection With Transaction Costs](https://arxiv.org/abs/2611.29552) | 针对AOLMA预测高估实际收益的问题，本文提出一种门控机制来估计预测被实现的程度，并将其作为资产特定的成本阈值应用于净利润最大化规则，再通过收益加权基金组合多个门控NPM专家，从而在存在交易成本的条件下实现更可靠的在线投资组合选择。 |
| [^37] | [GeoFusion: An Uncertain Machine Learning Framework for Sensor Fusion](https://arxiv.org/abs/2611.29510) | GeoFusion提出一个三阶段不确定性机器学习框架，将GPS观测建模为不确定性对象（火山函数与山峰函数）并按位置事件聚类，从而利用多部智能手机的协作传感器融合提升定位精度。 |
| [^38] | [Optimal Labelling Designs for Sequential Decisions on Language Model Updates](https://arxiv.org/abs/2611.29416) | 该论文提出SOLD方法，通过结合廉价的LLM裁判评估与少量人工标注，序贯地选择提示并决定停止时机，在保证错误概率可控的前提下最快地判断语言模型更新后准确率的变化。 |
| [^39] | [A comparison of knot point screening algorithms for highly adaptive lasso](https://arxiv.org/abs/2611.29270) | 本文比较了高度自适应lasso（HAL）中不同的结点筛选算法，旨在保持其快速收敛率的同时提升该非参数回归估计量的计算效率。 |
| [^40] | [Fuzzy e-Means: Fuzzy c-Means on Large Data with Moving-Average Mini-Batch Updates](https://arxiv.org/abs/2611.29241) | 本文提出模糊e均值（e-FCM），将e均值的移动平均小批量优化器迁移到模糊c均值上，并通过先硬分配初始化、再模糊更新的两阶段策略解决了空簇误删和局部极小值问题，实现了大数据上的高效模糊聚类。 |
| [^41] | [Deep Negative Binomial Autoregression for Sparse Spatio-Temporal Counts: Estimation and Predictive Distribution Bounds](https://arxiv.org/abs/2611.29207) | 该论文提出深度负二项自回归模型DeepNB-STAR用于稀疏、过度离散的时空计数数据，首次在无混合性假设且允许站点间任意依赖的条件下建立了Hellinger风险的oracle不等式、几何绝对正则性以及h步预测分布误差的理论界，并在日本监测数据上取得了更优的预测性能。 |
| [^42] | [Why flattening the P-value histogram works: a theory of the standard-deviation optimization in tensor-decomposition-based unsupervised feature extraction](https://arxiv.org/abs/2611.29201) | 该论文首次为基于张量分解的无监督特征提取中“拉平P值直方图以优化标准差”这一启发式方法提供了理论证明，表明背景基因的分量近似服从标准差为√d/s的正态分布，且信号基因即使占全部基因的正比例也能与背景分离。 |
| [^43] | [Sub-Gaussian HDLSS Principal Component Analysis: High-Probability Theory and Numerical Evidence for NRM, CDM, and Block Noise Reduction](https://arxiv.org/abs/2611.29058) | 本文在独立因子次高斯模型下为HDLSS主成分分析建立了基于得分矩阵、体移与高概率余项的高概率分解理论，揭示了序贯降噪对应含后续尖峰减除的得分预言机而分块校正可避免该减除的机制，并解释了交叉数据矩阵方法中样本分裂的收益与代价，同时以千次重复模拟加以验证。 |
| [^44] | [Mirror Descent at the Edge of Stability](https://arxiv.org/abs/2611.29043) | 该论文提出“镜像锐度”概念并推导对偶中心流ODE，首次将梯度下降的稳定性边缘现象研究推广到非欧几里得的镜像下降设定，并实验发现镜像锐度徘徊在 $2/\eta$ 阈值稍上方。 |
| [^45] | [A Novel Approach to Clustering Shape Data Using Iteratively Weighted Vectors with k-Means](https://arxiv.org/abs/2611.28880) | 该论文提出在迭代k-means算法中加入逐元素权重更新步骤以提升形状数据的聚类性能，并将权重估计归结为最小二乘问题，几乎不增加额外计算成本。 |
| [^46] | [How many mistakes does it take to learn an interval? Exact Littlestone dimensions of intervals and their unions](https://arxiv.org/abs/2611.28727) | 本文精确确定了离散区间与区间并集的 Littlestone 维数，证明强迫 d 次错误所需的最小域规模遵循一个三段式规律，并发现常见的“等待正样本后二分搜索端点”学习算法并非最优。 |
| [^47] | [Robust Forest-based Estimation of Heterogeneous Treatment Effect Under High-dimensional Confounding](https://arxiv.org/abs/2611.28662) | 本文提出正交Huber随机森林（OHRF）框架，通过融合Huber稳健估计、Neyman正交性与森林自适应加权，在高维混淆、非线性异质性和重尾极端结果的阿尔茨海默病观察数据中稳健地估计异质性处理效应。 |
| [^48] | [Stein Guidance without Gradients: Discretization Uniform Diffusion Inference Time Scaling beyond Particle SMC](https://arxiv.org/abs/2611.28654) | 提出Stein引导方法，将无导数引导与全路径Metropolis-Hastings校正相结合，每步仅需两次查询即可在期望意义下近似梯度引导，并证明误差随MH更新次数超多项式衰减且常数不随引导深度增长，突破了粒子SMC需粒子数随引导深度线性增长的限制。 |
| [^49] | [Nearly Minimax-Optimal Sample Complexity of Natural Policy Gradient with a Matching Information-Theoretic Lower Bound](https://arxiv.org/abs/2611.28628) | 本文证明了基于样本的自然策略梯度（NPG）在表格型有限时域MDP中达到接近极小极大最优的样本复杂度 $\widetilde O(SAH^4/\varepsilon^2)$，并给出了与之匹配的信息论下界，回答了直接策略优化能否达到统计极限这一基本问题。 |
| [^50] | [Neural Schr\"{o}dinger Bridge via Yasue Action with Applications to Generative Modeling](https://arxiv.org/abs/2611.28623) | 该论文提出了一种基于Yasue作用泛函鞍点公式的神经求解器，通过学习常微分方程的最优速度（而非随机微分方程的漂移）来求解薛定谔桥问题，且仅需源和目标样本即可运行，因此特别适用于高维生成建模任务。 |
| [^51] | [The Curved Exponential Family of Cylindrical Distributions: m-Projection, Curvature and a Test of the von Mises-Fisher Component](https://arxiv.org/abs/2611.28598) | 本文研究推广 Mardia-Sutton 模型的圆柱数据弯曲指数族，给出了极大似然估计存在唯一的充要条件及其作为 m-投影的显式构造，并证明该估计残差在 Fisher 度量下的平方长度恰为经典的 von Mises-Fisher 对 Fisher-Bingham 方向的得分检验。 |
| [^52] | [Tight Mistake-Abstention Tradeoffs for Sequential Prediction under Adversarial Injections](https://arxiv.org/abs/2611.28579) | 本文提出“加权打散势”及其变体，为对抗注入下的序列预测问题给出了弥合上下界差距的算法，在不同设定下基本紧致地刻画了错误-弃权权衡，并在分布已知时达到了此前猜想的 O(d log T) 错误界。 |
| [^53] | [Complex band-limited maximum likelihood: estimating the wave behind detected quanta](https://arxiv.org/abs/2611.28572) | 该论文提出复数带限最大似然方法（complex BLML），通过在物理允许的谱约束下最大化探测位置的似然，直接从粒子探测位置估计包含相位的完整波函数 $\psi$，且仅需一个特征值即可认证全局最优。 |
| [^54] | [When and Where to Transfer: Learning Transferable Historical Contexts for Short-History Multivariate Time Series Forecasting](https://arxiv.org/abs/2611.28567) | 提出分层上下文迁移框架HiCT，通过共享上下文编码器和交叉注意力自适应地加权数据源群体与历史上下文，确定迁移的时机与来源，以改进短历史多变量时间序列预测，并为其提供了理论上的目标超额风险界。 |
| [^55] | [A hybrid Graph-based Machine Learning Model for predicting Delivery Times in Supply Chain Logistics](https://arxiv.org/abs/2611.28467) | 该论文提出了一种融合图论、机器学习与马尔可夫链随机方法的混合建模框架，利用历史与实时异构数据（交通、天气、供应商备货状态等）预测供应链中供应商与制造商之间入站物流的交付延误。 |
| [^56] | [Guidance-Induced Activation of Score Estimation Errors in Diffusion Models](https://arxiv.org/abs/2611.28451) | 该论文揭示了CFG中的“引导诱导误差激活”机制：通过加权Helmholtz-Hodge分解，识别并量化了原本对参考密度无瞬时影响的螺线型score误差分量如何因引导密度与参考密度的差异而影响引导采样的边缘分布演化。 |
| [^57] | [Targeted Learning for Externally Controlled Trials and Beyond](https://arxiv.org/abs/2611.28421) | 本文针对外部对照试验提出了基于平衡权重的靶向学习框架，给出有效影响函数与靶向最大似然估计量（TMLE）及其实用算法，并将该框架扩展到辅以外部协变量的随机对照试验等更多场景。 |
| [^58] | [Data-Dependent Probability Estimation via Sample Amplification](https://arxiv.org/abs/2611.28399) | 该论文将样本扩增与基于在线优化的分布学习相结合，提出了一种多项式时间估计器，证明在选择规则已知的条件下仅需 Õ(√d) 个样本即可对数据相关事件实现常数均方误差估计，从而解决了 Feldman 等人提出的开放问题。 |
| [^59] | [Beyond Gaussian Calibration: Local Inference in Decentralized SGD](https://arxiv.org/abs/2611.28370) | 该论文揭示去中心化SGD中客户端局部端点虽整体高斯覆盖率接近标称但双尾严重失准，并给出一个可由图与步长计算的精确噪声滤波判据，证明在特定网络上极限分布是非高斯局部场与独立高斯共识项之和。 |
| [^60] | [Directional Bias in Quantile SGD and Score-Inversion Inference under Dependence](https://arxiv.org/abs/2611.28367) | 该论文揭示了数据的序列依赖性会在分位数SGD中产生一种得分自协方差无法检测的方向性偏差（反转数据流可使主导偏差变化三倍），并将其分解为滞后特定交互作用项与曲率项，同时证明延迟阈值可逐滞后消除交互作用而不损失观测信息与主导方差。 |
| [^61] | [Which Loans, Not How Many: Exactly Level-Invariant, Anytime-Valid Monitoring of Deployed Risk Models](https://arxiv.org/abs/2611.28342) | 本文提出一种精确水平不变且随时有效（anytime-valid）的已部署风险模型校准监控方法，通过以月度计数条件化消除截距影响，从而将贷款数量的变化（如再融资浪潮）与评分-结果关系本身的变化区分开来。 |
| [^62] | [Variable Importance Recovery in Tabular Foundation Models: A Simulation Study](https://arxiv.org/abs/2611.28241) | 该论文通过蒙特卡洛模拟研究发现，表格基础模型在恢复总体层面变量重要性方面与成熟的机器学习基线方法相当，且往往更为准确。 |
| [^63] | [Learning the Projection onto the Data Support: Distribution Learning that Avoids Memorization](https://arxiv.org/abs/2611.28234) | 提出了一种通过学习数据支撑集上最近点投影来实现流形上分布学习的新框架，该框架能自动适应内在维度并支持多维度流形并集，且其优化目标从原理上排除了通过记忆训练样本实现的解，从而避免了扩散模型等生成模型中的记忆化问题。 |
| [^64] | [Rollout Scheduling for Chaotic Neural ODEs: Coverage Gains, Computational Cost, and Predictive Accuracy](https://arxiv.org/abs/2611.28192) | 本文提出了一种面向混沌系统神经ODE训练的rollout长度自适应调度方法，在不稳定性预算约束下最大化轨迹数据覆盖并给出覆盖增益的理论界，实验表明其在相同计算成本下优于固定调度并提升预测精度。 |
| [^65] | [The Value of Labels in Data Selection for Least Squares](https://arxiv.org/abs/2611.28169) | 该论文量化了最小二乘数据选择中“看到标签”的价值：仅用特征的最优选择者最坏情况超额损失率为 $d\kappa/n$，而利用标签可将主导常数改善至 $d\kappa/2$，即在低维（$d\le4$）情形下标签恰好带来两倍的收益。 |
| [^66] | [Statistical Inference with Ill-Conditioned Random Weight Neural Network Sieves](https://arxiv.org/abs/2611.26410) | 本文证明了随机权重神经网络筛在维数增大时必然出现病态性（最小特征值消失），并在显式处理该病态性的基础上建立了相应的逼近速率、估计与统计推断理论。 |
| [^67] | [Diffusion Models are Provably Efficient Priors for Sparse Recovery](https://arxiv.org/abs/2611.24602) | 本文针对压缩感知中的稀疏恢复问题，首次证明基于预训练扩散模型先验的似然引导退火朗之万采样器能够在多项式时间内以近乎最优的样本复杂度可证明地恢复稀疏信号，为扩散模型作为逆问题先验提供了高效的端到端理论保证。 |
| [^68] | [Fold Weighting for Cross-Validation Under Continuous Distributional Drift](https://arxiv.org/abs/2611.23184) | 该论文提出在连续分布漂移下对交叉验证的各折进行加权，证明了对权重最小化风险界是一个凸优化问题，且将权重向近期折偏移可以有效降低由漂移导致的部署风险，同时需权衡验证噪声的增加。 |
| [^69] | [On numerical issues of divergences between nearly equal distributions](https://arxiv.org/abs/2611.18511) | 该论文揭示了当类别分布几乎相等时，KL散度、JSD散度及熵等标准散度运算存在与softmax类似的数值精度问题，并提出了数值稳定的重新表述方案及经过属性测试验证的实现。 |
| [^70] | [Robbing Peter to Pay Paul: Cross-Attribute Externalities of Fairness Interventions](https://arxiv.org/abs/2611.14717) | 本文揭示了针对某一敏感属性的公平性干预可能会加剧其他敏感属性（甚至包括不可观测属性）的不公平性，并利用Wasserstein重心差异刻画这种跨属性外部性，提出了能同时捕捉边际与联合公平性变化的分布框架。 |
| [^71] | [Katz Kernel Based Stratification of Breast Cancer Patients](https://arxiv.org/abs/2611.14540) | 本文提出了一种基于Katz核的图核方法，通过比较患者特异性的信号通路网络结构来实现乳腺癌患者分层，从而识别具有临床意义的患者亚群并助力新型生物标志物的发现。 |
| [^72] | [Bayesian Tensor Train Kernel Machines with Automatic Model Complexity Inference](https://arxiv.org/abs/2611.07129) | 本文提出了贝叶斯张量链核机器（BTT-KM），通过在TT核心上施加分层ARD先验并结合平均场变分推断，实现了从数据中自动推断TT秩与特征维度的闭式贝叶斯学习，且计算复杂度与确定性方法相同。 |
| [^73] | [The Price of Scale in Proactive Memory Recall](https://arxiv.org/abs/2610.34507) | 该论文证明主动记忆召回的门控决策本质上是一个随记忆库规模 N 增长的多重假设检验问题——单条记忆的假阳性率按 -ln(1-α)/N 衰减，导致命中率随 N 呈幂次下降，从理论与实证上解释了记忆库越大、主动记忆触发越容易失效的现象。 |
| [^74] | [SABLE: Scalable Affine Bayesian Latent-space Exploration](https://arxiv.org/abs/2610.31495) | SABLE 提出了一种贝叶斯仿射代理优化方法，通过将查询分解为由期望改进（EI）优化的低维跨度内分量和由标量范数支配的跨度外分量，在评估预算远小于潜空间维度的高维潜空间中实现了可扩展的贝叶斯优化。 |
| [^75] | [Which Model for Which Role? Cost-Efficient Pareto Recommendation for LLM Agentic Workflows](https://arxiv.org/abs/2610.26676) | 提出CC-Gittins算法，通过利用评估成本与部署成本之间的耦合关系，高效地为LLM智能体工作流中的不同角色推荐质量与成本权衡下的Pareto最优模型分配方案 |
| [^76] | [Trust-Region Optimization for Smooth Potential-Interaction Energies in Wasserstein Space](https://arxiv.org/abs/2610.08883) | 该论文提出了Wasserstein空间上光滑势-相互作用能量的信赖域优化方法，通过推前曲线上的二次模型、$L^2(\rho)$ 步长半径以及带显式自伴二阶变分算子的Steihaug-Toint子求解器，在温和条件下证明了目标函数单调不增且Wasserstein梯度范数收敛于零。 |
| [^77] | [FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training](https://arxiv.org/abs/2610.00385) | FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。 |
| [^78] | [Boolean threshold functions, neuron capacity, and memory retrieval](https://arxiv.org/abs/2609.29756) | 本文精确计数了n维布尔阈值函数的数目，将单个阈值神经元的容量确定为 $n^2-\log_2(n!)+1+O(n^{-99})$ 比特，把经典误差项从 $O(n)$ 改进到 $O(n^{-99})$，并为神经网络无虚假记忆检索的容量问题提供了数学基础。 |
| [^79] | [Efficient Cost-Aware LLM Evaluation via Bayesian Bandit Gittins Indices](https://arxiv.org/abs/2609.25645) | 提出GittinsEval，一种基于贝叶斯最优Gittins策略的成本感知LLM评估方法，通过轻量级在线更新高效决定下一个待评估配置及停止时机，在多个基准上以更低的评估成本取得有竞争力的性能。 |
| [^80] | [Matching Multi-Loop Complexities with a Single Loop: Optimal Optimization Stationarity and Best-Known Game Stationarity in Nonconvex--Concave Minimax Optimization](https://arxiv.org/abs/2609.17973) | 该论文提出了一种结合投影外梯度更新、对偶动量和移动近端中心的单循环投影阻尼外梯度算法，在非凸-凹极小极大优化中以单循环方法匹配了多循环方法的复杂度，同时实现了最优的优化平稳性保证和已知最佳的博弈平稳性保证。 |
| [^81] | [A complete characterization of sequential testability and change detectability in i.i.d. models](https://arxiv.org/abs/2609.05752) | 该论文首次对i.i.d.复合模型中的序列检验与变化检测给出了完整刻画：可检验性当且仅当备择假设可被可数族有限块事件分离，而可数族有界块赌注能否恢复内在增长率则决定了最优e-过程及最优期望样本量系数1/Γ(Q)能否同时达到。 |
| [^82] | [Exact Algebraic Computation of Learning Coefficients for Two-Dimensional Singular Models](https://arxiv.org/abs/2608.20183) | 本文提出了首个确定性算法，能精确计算二维奇异模型的局部RLCT，并给出了复杂度上界和广泛适用性证明。 |
| [^83] | [Benchmarking Quantum Machine Learning for Power-System Attack Detection: Evaluation Choices Decide the Outcome Before the Models Do](https://arxiv.org/abs/2608.15617) | 该论文通过系统性的基准测试揭示，电力系统攻击检测中量子机器学习与经典模型的比较结果高度依赖于评估协议中的八项关键选择，这些选择在模型运行前就决定了结论走向。 |
| [^84] | [Online Inference for Quantile Temporal Difference Learning in Distributional Reinforcement Learning](https://arxiv.org/abs/2608.12973) | 本文提出了一种基于随机缩放的在线推断方法，通过泛函中心极限定理支持，在不存储完整轨迹的情况下对分布强化学习中的QTD进行统计推断，显著减少了内存需求。 |
| [^85] | [Handling Missing Data in Probabilistic Regression Trees](https://arxiv.org/abs/2608.06195) | 本文提出三种在概率回归树构建过程中直接处理缺失预测变量值的方法，无需单独的插补步骤，并能在任意缺失模式下保持概率守恒和边缘兼容性等基本概率特性。 |
| [^86] | [The Loss Does Not See the Basis, but Adam Does](https://arxiv.org/abs/2608.05136) | 论文证明了旋转等变性是梯度下降在过参数化问题中产生低秩偏置的关键机制，而 Adam 因破坏了旋转对称性而不具备该偏置，并提出了一种基于范数的判别测试来预测各优化器的解的分组行为。 |
| [^87] | [Convergence of Large-Stepsize SGD for the Cross-entropy Loss via Stochastic Lyapunov Stability](https://arxiv.org/abs/2606.30930) | 本文借助随机李雅普诺夫稳定性理论证明，大步长SGD在多分类交叉熵损失下（含线性分类器和两层神经网络）虽会在“稳定边缘”振荡与稳定下降之间交替，但迭代点总能在固定次数内回归稳定，从而实现最佳迭代意义上的收敛。 |
| [^88] | [Conformal Selection under Prevalence Shift: FDR Bounds and Finite Calibration](https://arxiv.org/abs/2606.22336) | 该文揭示了源总体与测试总体之间的流行率偏移会破坏cfBH方法的FDR控制，并给出大样本下的最坏情形FDR界（如名义10%可升至近40%），同时证明在非目标得分分布一致的条件下，仅用带标签非目标样本校准可摆脱对非目标比例的限制。 |
| [^89] | [Proxy-Based Approximation of Shapley and Banzhaf Interactions](https://arxiv.org/abs/2605.22738) | ProxySHAP将树代理模型的高样本效率与情景残差校正相结合，实现了Shapley与Banzhaf交互作用估计在速度与精度上的兼得，并首次提出多项式时间的树集成精确交互指数计算方法。 |
| [^90] | [Sliced-Regularized Optimal Transport](https://arxiv.org/abs/2604.23944) | 本文提出切片正则化最优传输（SROT），首次以切片最优传输计划作为参考先验来正则化经典最优传输，在保持可扩展性且稀疏情形下复杂度更低的同时，能在相同正则化水平下比熵正则化OT更准确地逼近精确最优传输计划，并且所得传输计划优于参考的切片最优传输计划本身。 |
| [^91] | [Well-Conditioned Oblivious Perturbations in Linear Space](https://arxiv.org/abs/2604.23193) | 该论文提出了一种仅需 O(n) 个 O(log n) 比特随机数的无关扰动方法，能将任意确定性矩阵的条件数降至 O(n)（效果与高斯扰动相当），从而仅需 O(n log n) 个随机比特即可在 O(n²log(n)/ε) 时间内求解 n×n 线性方程组至 ε 后向误差，推动了矩阵扰动的去随机化。 |
| [^92] | [Rare Event Analysis via Stochastic Optimal Control](https://arxiv.org/abs/2604.13213) | 该论文提出了一种基于随机最优控制的committor函数估计新框架，将committor学习建立在有上限的命中时间轨迹之上，为稀有事件分析提供了有原则且严谨的方法。 |
| [^93] | [Dataset Distillation Efficiently Encodes Low-Dimensional Representations from Gradient-Based Learning of Non-Linear Tasks](https://arxiv.org/abs/2603.14830) | 本文首次从理论上证明，在两层神经网络的梯度训练中，数据集蒸馏算法能将多指标模型等非线性任务的低维结构高效编码进蒸馏数据，且所需存储复杂度仅为 $\tilde{\Theta}(r^2d+L)$。 |
| [^94] | [A Trust-Region Interior-Point Stochastic Sequential Quadratic Programming Method](https://arxiv.org/abs/2603.10230) | 本文提出一种信赖域内点随机序贯二次规划方法，利用满足自适应精度条件的随机估计和内点障碍策略求解随机目标加确定性非线性约束的优化问题，并证明了KKT残差几乎必然收敛。 |
| [^95] | [On the Convergence of Multicalibration Gradient Boosting](https://arxiv.org/abs/2602.06773) | 本文首次为多校准梯度提升算法提供了理论收敛保证，证明预测更新以 O(1/√T) 的速率衰减，并在平方误差损失及弱学习器平滑性假设下进一步提升至线性收敛。 |
| [^96] | [Variance Reduction Based Experience Replay for Policy Optimization](https://arxiv.org/abs/2602.05379) | 本文提出方差缩减经验回放（VRER）框架，通过选择性重用信息样本降低策略梯度方差，并配套理论分析，提升策略优化样本效率。 |
| [^97] | [More Bang for the Buck: Improving the Inference of Large Language Models at a Fixed Budget using Reset and Discard (ReD)](https://arxiv.org/abs/2601.21522) | 本文提出重置与丢弃（ReD）查询方法，克服了pass@k幂律行为导致的边际收益递减问题，在固定预算下显著提升大语言模型回答独特问题的数量（coverage@cost），并能定量预测节省的尝试次数。 |
| [^98] | [Contextual Distributionally Robust Optimization with Causal and Continuous Structure](https://arxiv.org/abs/2601.11016) | 该论文提出了融合因果与连续结构的情境化分布鲁棒优化框架，通过引入因果 Sinkhorn 散度构建模糊集并给出强对偶形式，同时设计了可微、光滑且可解释的软回归森林决策规则来求解最优策略。 |
| [^99] | [Stochastic Deep Learning: A Probabilistic Framework for Modeling Uncertainty in Structured Temporal Data](https://arxiv.org/abs/2601.05227) | 提出SLDI框架，将伊藤随机微分方程嵌入变分自编码器的潜空间，通过神经网络参数化漂移与扩散项并对伴随状态进行共同参数化，实现对结构化时序数据不确定性的连续时间灵活建模与量化。 |
| [^100] | [New Lower Bounds for Stochastic Non-Convex Optimization through Divergence Decomposition](https://arxiv.org/abs/2502.14060) | 该论文通过将优化问题重新表述为函数识别问题并运用散度分解技术，为满足准凸性、二次增长和受限割线不等式等性质的非凸函数类的随机一阶优化建立了新的且几乎紧的带噪声梯度查询复杂度下界。 |
| [^101] | [A Unified Comparative Study with Generalized Conformity Scores for Multi-Output Conformal Regression](https://arxiv.org/abs/2501.10533) | 本文对多输出保形回归中的九种保形方法进行了统一比较研究，并提出两类由单变量方法推广而来的新型一致性分数，它们在保持精确有限样本边际覆盖率的同时实现渐近条件覆盖率。 |
| [^102] | [Optimal Rate of Kernel Regression in Large Dimensions.](http://arxiv.org/abs/2309.04268) | 该论文提出了一种针对大维度数据的核回归的最优比率，通过使用Mendelson复杂性和度量熵来刻画其上界和最小化下界。此外，研究还发现最优比率随着维度与样本大小关系的变化呈现出多次下降的行为。 |

# 详细

[^1]: AlphaDefect：《杀戮尖塔2》的专家级AI

    AlphaDefect: Expert-Level AI for Slay the Spire 2

    [https://arxiv.org/abs/2611.31076](https://arxiv.org/abs/2611.31076)

    AlphaDefect通过PPO训练单一神经网络控制全部对局决策，在《杀戮尖塔2》最高难度下取得62-80%的胜率，远超人类玩家11-20%的水平。

    

    《杀戮尖塔2》(STS2) 是一款单人卡组构建游戏，需要在高度随机性下进行长期规划。我们提出了AlphaDefect，一个面向STS2的专家级AI系统：对于每个角色，仅用一个通过近端策略优化(PPO)训练的网络来完成单次对局中的所有决策，涵盖战斗内与战斗外的各个环节。在最高难度“飞升10”下，基于2026年11月时的两个游戏版本以及训练中从未玩过的测试种子，AlphaDefect在六个角色上使用贪婪策略时赢得62-77%的对局，使用测试时搜索时赢得68-80%的对局（每个角色经过13亿至67亿次决策的训练）；而在相同难度下，各水平人类玩家上传的对局胜率仅为11-20%。这些网络将游戏状态读取为实体记录，通过学习到的嵌入以及来自游戏数据的属性来识别每张卡牌、遗物和敌人，并结合由游戏引擎计算出的公共信息后果。消融实验显示……

    arXiv:2611.31076v1 Announce Type: new  Abstract: Slay the Spire 2 (STS2) is a single-player deck-building game that requires long-term planning under heavy randomness. We present AlphaDefect, an expert-level AI system for STS2: for each character, one network trained with proximal policy optimization (PPO) makes every decision of a run, in combat and outside it. At the highest difficulty, Ascension 10, on two versions of the game current in November 2026 and on test seeds not played in training, AlphaDefect wins 62-77% of runs across six characters with greedy play and 68-80% with test-time search, after 1.3 to 6.7 billion decisions of training per character; runs uploaded by human players of all levels are won 11-20% of the time at the same difficulty. The networks read the game as entity records that identify each card, relic and enemy through a learned embedding and attributes from the game data, together with consequences of public information computed by the engine. Ablations show
    
[^2]: RoFE：面向低数据多模态表格学习的旋转冻结嵌入

    RoFE: Rotated Frozen Embeddings for Low-Data Multimodal Tabular Learning

    [https://arxiv.org/abs/2611.31049](https://arxiv.org/abs/2611.31049)

    该论文揭示了表格基础模型（PFN）中按列标准化会消除PCA压缩嵌入的方差、导致标注数据稀缺时PCA接口失效甚至不如线性分类器的失败机制，并提出RoFE（旋转冻结嵌入）作为低数据多模态表格学习的更优接口。

    

    多模态表格学习将表格数据与图像和文本相结合，通常应用于标注数据稀缺的场景。诸如先验数据拟合网络（PFN）等表格基础模型通过上下文学习提供了一种有前景的解决方案，无需针对特定任务进行训练。然而，现代图像和文本编码器生成的嵌入具有数百甚至数千个维度，远远超过PFN预训练期间所遇到的特征数量。现有方法使用主成分分析（PCA）压缩冻结嵌入来弥合这一差距。令人惊讶的是，我们发现当标注样本稀缺时，这种PCA接口的表现甚至可能不如简单的线性分类器，尽管PFN本就是为小数据集设计的。我们揭示了其潜在的失败机制：PFN的按列标准化会消除PCA得分的方差。当PCA保留了标注样本的所有方向时，样本会塌缩成一个正则单纯形……

    arXiv:2611.31049v1 Announce Type: new  Abstract: Multimodal tabular learning combines tabular data with images and text, often in settings where labeled data are scarce. Tabular foundation models such as prior-data fitted networks (PFNs) offer a promising solution via in-context learning without task-specific training. However, modern image and text encoders produce embeddings with hundreds or thousands of dimensions, far exceeding the number of features encountered during PFN pretraining. Existing approaches bridge this gap using principal component analysis (PCA) to compress frozen embeddings. Surprisingly, we find that this PCA interface can underperform simple linear classifiers when labeled examples are scarce, even though PFNs are designed for small datasets. We identify the underlying failure mechanism: the PFN's per-column standardization removes the variances of the PCA scores. When the PCA keeps every direction of the labeled examples, the examples collapse into a regular sim
    
[^3]: 强噪声下的几何揭示：基于随机几何图的谱嵌入与聚类

    Revealing Geometry under Strong Noise: Spectral Embeddings and Clustering via Random Geometric Graphs

    [https://arxiv.org/abs/2611.31047](https://arxiv.org/abs/2611.31047)

    本文提出一种先对成对距离去噪再构建随机几何图进行谱嵌入与聚类的新方法，即使在强噪声和群体中心重合（如同心形状）的情况下也能有效揭示数据的潜在低维几何结构并区分不同群体。

    

    现代大规模数据集通常包含有用的低维非线性结构，但这些结构往往被噪声所掩盖。谱嵌入算法依赖于观测之间有意义的局部关系，然而跨越多个坐标累积的噪声会使观测到的距离变得不可靠。当数据群体具有弯曲或同心形状、无法通过其中心加以区分时，这也使聚类问题变得更加复杂。在本文中，我们提出了一种新方法，在使用成对距离构建谱嵌入和识别社区之前对其进行去噪。我们的方法使用交叉拟合投影来估计底层信号之间的距离，从观测样本中选取邻域半径，并构建一个随机几何图，其谱坐标代表了潜在的几何结构。通过连接邻近的点，这种几何图即使在各群体中心重合的情况下也能区分不同的群体。一种新颖的递归（recu...）方法...

    arXiv:2611.31047v1 Announce Type: new  Abstract: Modern massive datasets often contain useful low-dimensional nonlinear structures that are hidden by noise. Spectral embedding algorithms rely on meaningful local relationships between observations, but noise accumulated across many coordinates can make the observed distances unreliable. This also complicates clustering when groups have curved or concentric shapes and cannot be distinguished by their centers. In this paper, we develop a new method that denoises pairwise distances before using them to construct spectral embeddings and identify communities. Our approach uses cross-fitted projections to estimate distances between the underlying signals, selects a neighborhood radius from the observed sample, and builds a random geometric graph whose spectral coordinates represent the latent geometric structure. By connecting nearby points, such a geometric graph can distinguish different groups even when their centers coincide. A novel recu
    
[^4]: 基于校准的预测驱动回归：超越有监督极小极大基准

    Prediction-Powered Regression via Calibration: Beyond the Supervised Minimax Benchmark

    [https://arxiv.org/abs/2611.30976](https://arxiv.org/abs/2611.30976)

    该论文提出基于校准的加权预测驱动回归方法 wPPRC，通过用标注数据校准伪标签并进行加权回归，在理论上证明其最坏情形风险可以超越有监督极小极大基准，从而体现大量伪标签带来的额外价值。

    

    我们研究标注数据有限、而携带可能不准确伪标签的观测数据大量存在情形下的非参数回归问题。我们提出的方法——基于校准的加权预测驱动回归，首先利用标注数据对伪标签进行校准，然后使用观测响应和校准后的响应通过加权回归来拟合预测器。在 Hölder 类上，我们为截断 ReLU 估计量推导了非渐近超额风险上界，并为相应的回归问题建立了极小极大下界。在适当条件下，上界与下界在对数因子范围内匹配。此外，在同一联合分布类上，wPPRC 能够取得比四种基线观测结构下的极小极大风险更低阶的最坏情形风险，从而展示了额外伪标签的价值。我们还构造了一个验证选择估计量，利用独立的带标注验证样本来进行选择。

    arXiv:2611.30976v1 Announce Type: cross  Abstract: We study nonparametric regression with limited labeled data and abundant observations carrying potentially inaccurate pseudo-labels. Our proposed method, weighted prediction-powered regression via calibration (wPPRC), first calibrates pseudo-labels using labeled data, then fits a predictor by weighted regression using observed and calibrated responses. Over H\"older classes, we derive nonasymptotic excess-risk bounds for clipped ReLU estimators and establish a minimax lower bound for the corresponding regression problem. Under suitable conditions, the upper and lower bounds match up to logarithmic factors. Moreover, wPPRC can achieve a worst-case risk of smaller order than the minimax risks under four baseline observation structures on the same joint distribution class, demonstrating the value of additional pseudo-labels. We also construct a validation-selected (VS) estimator using an independent labeled validation sample to choose the
    
[^5]: $K$-中位秩：一种针对重尾数据的鲁棒聚类算法及其在光谱学和金融中的应用

    $K$-Midranks: A robust clustering algorithm for heavy-tailed data with applications to spectroscopy and finance

    [https://arxiv.org/abs/2611.30953](https://arxiv.org/abs/2611.30953)

    本文提出了一种名为 $K$-中位秩的鲁棒聚类算法，通过将观测值替换为逐坐标秩来降低重尾数据中极端值的影响，在柯西型重尾数据上优于传统聚类方法。

    

    对于依赖原始观测值之间距离或特定分布假设的方法而言，对包含极端观测值和重尾分布的数据进行聚类可能具有挑战性。我们提出了 $K$-中位秩（$K$-midranks），这是一种聚类方法，它将观测值替换为逐坐标的秩，并使用类似于 $K$-均值的程序对变换后的数据进行划分。这种变换降低了极端值的影响，同时保留了每个变量内观测值的排序信息。我们通过模拟研究将 $K$-中位秩与 $K$-均值、$K$-中心点以及基于高斯分布、学生t分布、柯西分布和广义双曲分布的混合模型进行了比较，并提供了一些理论性质。$K$-中位秩在柯西平移数据上通常优于竞争方法。相比之下，$K$-均值和基于模型的方法在高斯数据以及三自由度的学生t数据上表现更好。我们展示了该方法在光谱学和金融领域的应用。

    arXiv:2611.30953v1 Announce Type: cross  Abstract: Clustering data with extreme observations and heavy tails can be challenging for methods that rely on distances between raw observations or specific distributional assumptions. We introduce $K$-midranks, a clustering method that replaces observations with coordinatewise ranks and partitions the transformed data using a procedure analogous to $K$-means. This transformation reduces the influence of extreme values while preserving the ordering of observations within each variable. We evaluate $K$-midranks against $K$-means, $K$-medoids, and mixture models based on Gaussian, Student's t, Cauchy, and generalized hyperbolic distributions through simulation studies, and provide some theoretical properties. $K$-midranks generally outperforms the competing methods on Cauchy-shifted data. In contrast, $K$-means and model-based methods perform better on Gaussian data and Student's t data with three degrees of freedom. We demonstrate the method's 
    
[^6]: 网络上KL正则化偏好动态所诱导的聚合规则

    Aggregation Rules Induced by KL-Regularised Preference Dynamics on Networks

    [https://arxiv.org/abs/2611.30941](https://arxiv.org/abs/2611.30941)

    论文证明网络中多个智能体进行KL正则化偏好对齐的迭代动态会诱导出一种加权聚合规则，权重由各智能体在网络中的长期注意力份额除以其温度参数决定，并在参考项或挑战者被聚合时分别导出基数型聚合与策略分化等结果。

    

    我们研究了一类智能体，它们各自将模型与自己的偏好数据进行对齐，同时观察其他智能体，并探讨由此产生的策略实现了哪种聚合规则。每一轮中，每个智能体执行一步KL正则化更新。该步骤涉及一个参考项——智能体因偏离它而受到惩罚——以及一个用于与自身响应进行比较的挑战者，这两者中的每一个既可以是智能体自身的，也可以是其所观察智能体的平均值；且注意力不必是相互的。对参考项进行聚合使该规则成为集体规则。此时智能体 $v$ 拥有权重 $\pi_v/\beta_v$，即其在网络长期注意力中所占份额除以其温度参数。此外，如果挑战者是固定的，则智能体会集中于最大化 $\sum_v(\pi_v/\beta_v)r_v(y)$ 的响应，其中 $r_v$ 是智能体 $v$ 的奖励，且该规则是基数型（cardinal）的。如果参考项固定而挑战者被聚合，则各智能体的策略在不动点处通常各不相同，任何加权的……

    arXiv:2611.30941v1 Announce Type: new  Abstract: We study agents that each align a model to their own preference data while observing one another, and ask which aggregation rule the resulting policies implement. Each round, an agent takes one KL-regularised step. The step involves a reference that the agent is penalised for leaving and a challenger against which its responses are compared, and each of the two may be the agent's own or an average over the agents it observes; attention need not be reciprocal. Aggregating the reference makes the rule collective. Agent $v$ then has the weight $\pi_v/\beta_v$, its share of the network's long-run attention divided by its temperature. If, in addition, the challenger is fixed, the agents concentrate on the response maximising $\sum_v(\pi_v/\beta_v)r_v(y)$, where $r_v$ is agent $v$'s reward, and the rule is cardinal. If the reference is fixed and the challenger aggregated, the agents' policies at a fixed point generally differ, and any weighted
    
[^7]: 从单个大图进行归纳式节点级泛化的理论

    A theory of inductive node-level generalization from a single large graph

    [https://arxiv.org/abs/2611.30937](https://arxiv.org/abs/2611.30937)

    本文提出了一个新理论框架，用于刻画图机器学习算法在仅从单个大图学习时进行归纳式节点级任务的泛化能力。

    

    在图机器学习（GML）中，泛化的概念根据任务层级和数据集属性有着根本的不同。例如，图级任务符合传统的独立同分布样本的机器学习框架，通常涉及相对较小的图。相反，节点级任务涉及相互依赖的节点，且往往只存在于少数几个大图中。现有研究通常考虑直推式设置，即训练数据和测试数据共存于同一个图中；或者对于归纳学习，试图恢复某种概率意义上的“独立性”，例如仍然假设存在大量独立的图或子图。在这项工作中，我们开发了一个新颖的框架，用于表达图机器学习算法在从单个大图学习时执行归纳式节点级任务的泛化能力，该框架借鉴了图神经网络可迁移性方面的文献。特别是，我们通过重新审视……强调了其与经典机器学习的相似之处。

    arXiv:2611.30937v1 Announce Type: cross  Abstract: In Graph Machine Learning (GML), the notion of generalization fundamentally varies depending on the task level and the properties of the dataset. For instance, graph-level tasks fit into the traditional ML framework of i.i.d. samples, with relatively small graphs. On the contrary, node-level tasks involve dependent nodes, often in a few large graphs. Existing studies often consider the transductive settings, where training and test data co-exist in the same graph, or, for inductive learning, try to recover some kind of probabilistic ``independence'', e.g. by still assuming a large number of independent graphs or subgraphs.   In this work, we develop a novel framework to express the generalization capabilities of Graph ML algorithms for inductive node-level tasks when learning from a single large graph, taking inspiration from the literature on GNN transferability. In particular, we emphasize the similarities with classical ML by revisi
    
[^8]: 通过近似交叉验证实现图模型中快速且一致的结构学习

    Fast and Consistent Structure Learning in Graphical Models via Approximate Cross-Validation

    [https://arxiv.org/abs/2611.30822](https://arxiv.org/abs/2611.30822)

    本文提出基于近似交叉验证的可扩展图结构学习方法，解决了交叉验证在结构学习中计算不可行且理论不一致的问题，并在温和的信号强度假设下实现了选择一致性。

    

    高维稀疏图模型的结构学习既需要计算上的可扩展性，也需要选择一致性。经典准则如（扩展）贝叶斯信息准则可以提供一致性，但依赖于模型特定的似然函数和复杂度校准。相比之下，交叉验证避免了这种解析校准，因此在图学习中颇具吸引力。然而，交叉验证在结构学习中计算上不可行，且理论上不一致。这在建模的一致性与灵活性之间造成了张力。在本工作中，我们通过开发一种基于近似交叉验证的可扩展图恢复方法来化解这一矛盾。我们从理论和实证两方面展示了该方法在学习大型图模型方面的有效性，并在温和的信号强度假设下建立了选择一致性。

    arXiv:2611.30822v1 Announce Type: cross  Abstract: Structure learning for sparse graphical models in high dimensions requires both computational scalability and selection consistency. Classical criteria such as (Extended) Bayesian Information Criterion can provide consistency, but rely on model-specific likelihoods and complexity calibration. In comparison, cross-validation avoids such analytic calibration and is therefore attractive for graph learning. However, cross-validation is computationally infeasible and theoretically inconsistent in structure learning. This creates a tension between consistency and flexibility in modeling. In this work, we resolve the tension by developing a scalable graph-recovery procedure built on approximate cross-validation. We demonstrate its effectiveness in learning large graphical models both theoretically and empirically, and establish selection consistency under mild signal strength assumptions.
    
[^9]: 非凸M-估计量的精确渐近性

    Precise Asymptotics of Non-Convex M-Estimators

    [https://arxiv.org/abs/2611.30798](https://arxiv.org/abs/2611.30798)

    该论文突破了现有状态演化理论要求近邻算子Lipschitz连续的限制，通过修正Onsager项以处理非凸损失函数或正则化项导致的近邻算子不连续问题，从而建立了非凸M-估计量在高维广义线性模型中的精确渐近理论。

    

    M-估计量在高维情形下的精确渐近理论目前主要局限于凸或轻度非凸的公式形式，然而非凸的损失函数和正则化项在实践中被广泛使用。我们研究了广义线性模型（GLM）中的M-估计问题，其中损失函数或正则化项可以是非凸的。在比例渐近区域下，对于旋转不变的设计矩阵，我们通过向量近似消息传递（VAMP）来分析该问题，其迭代过程由一个称为状态演化的低维确定性递归来描述。现有的状态演化证明要求损失函数和正则化项的近邻算子满足Lipschitz条件，但非凸的损失函数或正则化项可能具有不连续的近邻算子，此时VAMP中通常的Onsager项便会遗漏不连续点带来的贡献。假设近邻算子是局部（摘要原文在此处被截断）的特殊函数……

    arXiv:2611.30798v1 Announce Type: cross  Abstract: Precise asymptotic theory for M-estimators in high-dimensional settings is largely limited to convex or mildly non-convex formulations, yet non-convex loss functions as well as regularizers are widely used in practice. We study M-estimation in generalized linear models (GLMs) where the loss function or the regularizer can be non-convex. Under rotationally invariant design matrices in the proportional asymptotic regime, we analyze the problem through Vector Approximate Message Passing (VAMP), whose iterates are described by a low-dimensional deterministic recursion called State Evolution (SE). Existing proofs of SE require the proximal operators of the loss function and the regularizer to be Lipschitz, but a non-convex loss or regularizer can have a discontinuous proximal operator, and the usual Onsager terms in VAMP then miss the contribution of the discontinuities. Assuming that the proximal operators are special functions of locally 
    
[^10]: MTL-MHOL：在延迟反馈和数据稀疏条件下的在线转化预测

    MTL-MHOL: Predicting Online Conversions under Delayed Feedback and Data Sparsity

    [https://arxiv.org/abs/2611.30648](https://arxiv.org/abs/2611.30648)

    本文提出模型无关的MTL-MHOL框架，通过时间分桶处理延迟反馈并结合辅助任务的多任务学习来缓解数据稀疏，在CVR预测中达到或超越所有基准模型。

    

    本文提出了一种与模型无关的多任务学习多头在线学习（MTL-MHOL）框架，用于转化率（CVR）预测。现有方法通常孤立地处理CVR预测中的关键挑战（如延迟反馈和数据稀疏），或缺乏灵活性和实际适用性。MTL-MHOL采用时间分桶方法来处理延迟反馈，并结合辅助任务的多任务学习来缓解数据稀疏问题。我们在一家公司的专有数据集和Criteo的公开数据集上对该框架进行了评估。MTL-MHOL在负对数似然（NLL）和相对交叉熵（RCE）指标上达到或超越了所有基准模型，并且在使用MLP骨干网络时能正确捕捉数据中的时间趋势，同时在使用DeepFM骨干网络时也能保持强劲性能。特别是，MTL-MHOL达到了先进的延迟反馈方法的性能水平。

    arXiv:2611.30648v1 Announce Type: new  Abstract: This paper proposes the model-agnostic Multi-Task Learning Multi-Head Online Learning (MTL-MHOL) framework for conversion rate (CVR) prediction. Existing approaches typically address key challenges in CVR prediction, such as delayed feedback and data sparsity, in isolation or lack flexibility and practical applicability. MTL-MHOL adopts a time bucketing approach to account for delayed feedback and combines it with multi-task learning of an auxiliary task to mitigate data sparsity. We evaluate the framework on proprietary datasets from a private company and on a public dataset from Criteo. MTL-MHOL matches or outperforms all benchmark models in terms of Negative Log-Likelihood (NLL) and Relative Cross Entropy (RCE), and it correctly captures temporal trends in the data using an MLP backbone, while maintaining strong performance with a DeepFM backbone. In particular, MTL-MHOL matches the performance of the advanced delayed-feedback method 
    
[^11]: 环境不变性究竟识别了什么：再生核希尔伯特空间中的极小不变集

    What Environment Invariance Identifies: Minimal Invariant Sets in Reproducing Kernel Hilbert Spaces

    [https://arxiv.org/abs/2611.30609](https://arxiv.org/abs/2611.30609)

    本文将不变因果预测的不变性准则提升到再生核希尔伯特空间中条件均值嵌入的算子层面，精确证明了不变集恰好是包含 PA(Y) 且不含 Y 后代的集合，从而确立 PA(Y) 是唯一的极小不变集，并给出有限样本下恢复所需的样本量刻画。

    

    不变因果预测通过搜索使目标变量 Y 的条件分布在各环境间保持稳定的协变量集合，来识别 Y 的直接原因。我们在算子层面表述该准则：当给定 X_S 时 Y 的条件均值嵌入在各环境间一致时，称集合 S 是不变的，并精确刻画了由此产生的集合族：在对环境的两个条件下（我们将其分开陈述，因为它们会各自独立失效），不变集合恰好是那些包含 PA(Y) 且不包含 Y 的任何后代的集合。由此得出两个结论：其一，不变性并不能唯一识别 PA(Y)，因为每一个不含 Y 后代的超集同样是不变的；其二，PA(Y) 是唯一的极小不变集。在有限样本情形下，恢复极小不变集要求非不变集合与不变性之间被一个间隔（margin）分隔开，且所需样本量分解为一个与周围维度呈对数关系的筛选项和一个继承项……

    arXiv:2611.30609v1 Announce Type: new  Abstract: Invariant causal prediction identifies the direct causes of a target $Y$ by searching for covariate sets whose conditional law of $Y$ is stable across environments. We state that criterion at the level of operators, calling a set $S$ invariant when the conditional mean embeddings of $Y$ given $X_S$ agree across environments, and characterise the resulting family exactly: under two conditions on the environments, which we state separately because they fail independently, the invariant sets are precisely those containing $PA(Y)$ and no descendant of $Y$. Two consequences follow. Invariance does not identify $PA(Y)$ uniquely, since every descendant-free superset is invariant as well; and $PA(Y)$ is the unique minimal invariant set. In finite samples, recovery requires the non-invariant sets to be separated from invariance by a margin, and the sample size splits into a screening term logarithmic in the ambient dimension and a term inherited 
    
[^12]: SCAN：用于可扩展高阶成分选择与估计的神经加性网络

    SCAN: Neural Additive Networks for Scalable High-Order Component Selection and Estimation

    [https://arxiv.org/abs/2611.30452](https://arxiv.org/abs/2611.30452)

    SCAN通过让每个基函数自动学习组合哪些特征，无需预先枚举交互项即可联合选择与估计不同阶数的成分，并具备万能逼近和极小极大最优速率等理论保证。

    

    函数方差分析（ANOVA）模型将预测表示为主效应与交互效应之和，但现有的神经函数ANOVA模型需要预先指定交互项，这迫使研究者在遗漏高阶结构与枚举大量交互项的成本之间进行权衡。我们提出了SCAN（Scalable Component Additive Networks，可扩展成分加性网络），它学习每个基函数组合哪些特征，从而无需枚举候选即可联合地选择和估计不同阶数的成分。学习到的成分索引集形成一个小型候选集合，可在其上高效地进行事后识别。一个全局成分选择参数在基函数之间共享信息，截断可忽略的成分选择权重即可得到训练后可识别的稀疏模型。我们建立了万能逼近性、（在相差一个对数因子的意义下的）极小极大最优速率以及成分选择……

    arXiv:2611.30452v1 Announce Type: cross  Abstract: Functional ANOVA models express a prediction as a sum of main effects and interactions, but existing neural functional ANOVA models require the interactions to be specified in advance, which forces a trade-off between missing higher-order structure and the cost of enumerating many interactions. We propose SCAN (Scalable Component Additive Networks), which learns which features each basis function combines, so that components of different orders are selected and estimated jointly without enumerating candidates. The learned component index sets form a small candidate collection on which post-hoc identification is carried out efficiently. A global component-selection parameter shares information across basis functions, and truncating negligible component-selection weights gives a sparse model that is identified after training. We establish universal approximation, a minimax optimal rate up to a logarithmic factor, and component selection 
    
[^13]: 覆盖能否解释核心集主动学习？均值、最大值与距离的选择

    Does coverage explain core-set active learning? Mean, max, and the choice of distance

    [https://arxiv.org/abs/2611.30450](https://arxiv.org/abs/2611.30450)

    本文证明核心集主动学习的覆盖泛化界对任意不相似度度量和任何不使用样本外标签的选择规则均成立，且在标注点标签确定时无需零损失假设、去除了类别数因子 C，仅依赖池到标注样本的平均距离与单元纯度。

    

    核心集选择将批量主动学习视为一个覆盖问题，即待标注的点使数据池的覆盖半径最小化。其核心论据是一个泛化界，其中该半径控制着数据池损失与已标注样本损失之间的差距。我们考察了这一界所要求的前提条件及其在实践中的含义。我们首先在证明所用的标注点处零条件风险假设下陈述该界，并证明此时它对任意不相似度度量以及任何不使用样本外标签的选择规则均成立。该证明逐点进行，并涉及两个平均值：数据池到已标注样本的平均距离，以及与标注点相关联的单元的纯度。在标注点标签确定的情况下，保留这两个量便可得到数据池损失的界，无需零损失假设，并且去掉了原常数中类别数 C 这一因子。

    arXiv:2611.30450v1 Announce Type: new  Abstract: Core-set selection casts batch active learning as a covering problem, in which the points to label minimise the covering radius of the pool. Its main argument is a generalization bound in which this radius controls the gap between the loss on the pool and on the labelled sample. We examine what this bound requires and what it says in practice. We first state it with the assumption that its proof uses at the labelled points, a zero conditional risk, and show that it then holds for any dissimilarity and any selection rule that uses no label outside the sample. The proof works point by point and involves two averages, the mean distance of the pool to the labelled sample and the purity of the cells attached to the labelled points. Keeping them gives, when labels are deterministic at the labelled points, a bound on the pool loss with no zero-loss assumption and without the factor C, the number of classes, of the original constant. We then com
    
[^14]: 超越岭回归的谱正则化

    Spectral Regularization Beyond Ridge Regression

    [https://arxiv.org/abs/2611.30440](https://arxiv.org/abs/2611.30440)

    本文提出包含岭回归与迭代Tikhonov正则化的双参数谱正则化族“包容岭回归”，精确刻画其有限样本风险，证明神谕调优的岭回归在该族中通常次优，并给出高阶估计量与指数极限滤波器严格优于岭回归的明确条件。

    

    我们研究了包容岭回归，这是一个双参数谱正则化族，其中同时包含岭回归和迭代Tikhonov正则化。我们推导了其精确的有限样本系数风险，并建立了在岭回归基线附近增大或减小额外滤波器形状参数以局部降低风险的充分必要条件。我们证明，经神谕调优的岭回归在IR族中通常是次优的。在惩罚强度与滤波器阶数同时发散的情形下，我们证明当二者的比值收敛于一个正的有限常数时，恰好存在非退化的谱极限，从而得到连续的指数滤波器。沿着固定有效尺度的路径，我们还得到了精确的风险差异比较，并给出了明确的充分条件，使得高阶估计量与指数极限滤波器均严格优于经神谕调优的岭回归。在此区域内，IR逐渐变得……

    arXiv:2611.30440v1 Announce Type: cross  Abstract: We study Inclusive Ridge (IR), a two-parameter spectral regularization family that contains ridge regression as well as iterated Tikhonov regularization. We derive its exact finite-sample coefficient risk and establish necessary and sufficient conditions for increasing or decreasing the additional filter-shape parameter around the ridge baseline to locally lower risk. We show that oracle-tuned ridge is generically suboptimal within the IR family. Under joint divergence of the penalty and filter order, we show that a nondegenerate spectral limit exists precisely when their ratio converges to a positive finite constant, yielding a continuous exponential filter. Along the fixed effective-scale path, we also obtain an exact risk-difference comparison and explicit sufficient conditions under which both higher-order estimators and the exponential limiting filter strictly outperform oracle-tuned ridge. In this regime, IR becomes progressively
    
[^15]: 具有异质模式结构的张量恢复

    Tensor Recovery with Heterogeneous Mode Structures

    [https://arxiv.org/abs/2611.30435](https://arxiv.org/abs/2611.30435)

    提出了一个针对异质模式结构的统一凸张量恢复框架，对平滑模式采用局部核加权、对其他模式分别施加低秩、组稀疏、图拉普拉斯或自回归正则化，配合共识式 HM-ADMM 算法求解，并给出了 oracle 不等式与高概率误差界的理论保证。

    

    张量恢复方法通常对各模式施加统一的结构假设，即使不同模式可能表现出不同形式的正则性。我们提出了一个统一的凸优化框架，根据各模式自身的结构对其进行区别处理：平滑变化的模式构成一个索引域，通过局部核加权加以处理；而保留的模式则可以接受低秩、组稀疏、图拉普拉斯或自回归正则化，这些正则化可以以不同且可能相互重叠的组合方式施加。该框架也涵盖了不带局部平滑的普通张量恢复。我们提出一种共识式 HM-ADMM 算法，将激活的惩罚项分解为易于处理的近端更新和线性系统更新。对于总体二次估计量，我们在独立迹回归假设下推导出确定性的 oracle 不等式和高概率误差界。这些界将随机误差、秩与支撑集复杂度、平滑偏差以及二次失配分离开来，并且还识别了……

    arXiv:2611.30435v1 Announce Type: new  Abstract: Tensor recovery methods often impose a common structural assumption across modes, even when different modes may exhibit distinct forms of regularity. We propose a unified convex framework that treats each mode according to its structure. Smoothly varying modes form an indexing domain and are handled by local kernel weighting, whereas retained modes may receive low-rank, group-sparse, graph-Laplacian, or autoregressive regularization in distinct and potentially overlapping combinations. The framework also covers ordinary tensor recovery without local smoothing. A consensus HM-ADMM algorithm separates the active penalties into tractable proximal and linear-system updates. For the population-quadratic estimator, we derive deterministic oracle inequalities and high-probability bounds under independent trace regression. The bounds disentangle stochastic error, rank and support complexity, smoothing bias, and quadratic mismatch. They also iden
    
[^16]: 用于高效多边际最优传输的低秩张量表示

    Low-Rank Tensor Representations for Efficient Multimarginal Optimal Transport

    [https://arxiv.org/abs/2611.30376](https://arxiv.org/abs/2611.30376)

    该论文提出低秩多边际最优传输（LRMOT），通过将传输计划限制为有界多重线性秩的张量，大幅降低了多边际最优传输的计算与内存开销，同时保持较低的传输成本。

    

    多边际最优传输（MOT）随着边际数量的增加，其传输计划由高维张量表示，导致计算和内存开销呈指数级增长。我们提出了低秩多边际最优传输（LRMOT），它将传输计划限制为具有有界多重线性秩的张量，为多边际耦合提供了紧凑且结构化的表示。这种结构使得我们能够设计出内存占用低的高效算法。我们为所得的低秩公式建立了近似界，并通过实验证明所提出的算法在大幅减少计算时间的同时，能够获得传输成本较低的解。

    arXiv:2611.30376v1 Announce Type: new  Abstract: Multimarginal Optimal Transport (MOT) suffers from exponential computational and memory costs in the number of marginals, as transport plans are represented by high-dimensional tensors. We introduce Low-Rank Multimarginal Optimal Transport (LRMOT), which restricts transport plans to tensors with bounded multilinear rank, providing a compact and structured representation of multimarginal couplings. This structure enables efficient algorithms with a low memory footprint. We establish approximation bounds for the resulting low-rank formulations and demonstrate experimentally that the proposed algorithms substantially reduce computational time while achieving solutions with low transport cost.
    
[^17]: 通过神经符号整合增强零样本学习

    Boosting zero-shot learning through neuro-symbolic integration

    [https://arxiv.org/abs/2611.30358](https://arxiv.org/abs/2611.30358)

    本文提出模糊逻辑原型网络（FLPN），利用逻辑张量网络框架将逻辑公理形式的背景知识注入训练过程，通过优化神经符号损失在视觉-语义嵌入空间中进行原型匹配，从而提升零样本学习能力。

    

    零样本学习（ZSL）旨在训练深度神经网络，使其能够从概念的语义描述出发，识别来自未见类别的物体。神经符号整合是指一类将符号知识表示与推理同深度神经网络的学习能力相结合的技术。然而，迄今为止，很少有研究探索如何在训练过程中利用神经符号技术注入先验知识以提升零样本学习能力。本文提出了模糊逻辑原型网络（FLPN），它将分类任务表述为视觉-语义嵌入空间中的原型匹配，并通过优化神经符号损失进行训练。具体而言，FLPN利用逻辑张量网络（LTN）框架，通过将一阶逻辑语言落地为实张量之间的可微分运算，以逻辑公理的形式纳入背景知识。

    arXiv:2611.30358v1 Announce Type: cross  Abstract: Zero-shot learning (ZSL) aims to train deep neural networks to recognize objects from unseen classes, starting from a semantic description of the concepts. Neuro-symbolic (NeSy) integration refers to a class of techniques that incorporate symbolic knowledge representation and reasoning with the learning capabilities of deep neural networks. However, to date, few studies have explored how to leverage NeSy techniques to inject prior knowledge during the training process to boost ZSL capabilities. Here, we present Fuzzy Logic Prototypical Network (FLPN) that formulates the classification task as prototype matching in a visual-semantic embedding space, which is trained by optimizing a NeSy loss. Specifically, FLPN exploits the Logic Tensor Network (LTN) framework to incorporate background knowledge in the form of logical axioms by grounding a first-order logic language as differentiable operations between real tensors. This prior knowledge
    
[^18]: 因果发现之后，诚实的置信区间必须有多长？良性与恶性弱边

    How long must an honest confidence interval be after causal discovery? Benign and malignant weak edges

    [https://arxiv.org/abs/2611.30256](https://arxiv.org/abs/2611.30256)

    该论文通过 Meek 闭包将弱边分为“良性”与“恶性”两类，刻画了因果发现后诚实置信区间相对已知图最优区间的最坏长度膨胀代价：良性边处代价有界（当跨模型常数 K=0 时甚至等于 1，即诚实性免费），而一旦存在恶性边，代价便达到 √n 量级。

    

    在因果发现之后，要在整个模型类上一致有效的总效应置信区间，必须为结构不确定性付出代价。我们以最坏情况下期望区间长度与已知图时有效区间长度之比来度量这一代价，研究对象为具有已知邻接关系超集和 k-三角忠实性假设的线性高斯模型。一旦存在一条可能“恶性”的边，全局代价便达到 √n 量级，因此我们在局部尺度上考察这一代价，即在一条边缺失的通用分布附近。我们的主要结果通过 Meek 闭包对这条弱边进行分类：若删除该边后，效应在所得的 MPDAG 中仍可识别，且各马尔可夫等价的解释彼此一致，则该边是“良性”的，此时局部代价有界；否则，当 k 较小时，代价为 √n 量级。在良性边处，代价为 R+K 的量级，其中 R 是有效长度的比值，K 是跨模型常数；当且仅当 K=0 时（这通常可由图结构判定），该代价等于 1，即“诚实性是免费的”。

    arXiv:2611.30256v1 Announce Type: cross  Abstract: After causal discovery, a confidence interval for a total effect valid uniformly over the model class must pay for structural uncertainty. We measure this price by the worst-case ratio of expected length to that of the efficient interval knowing the graph, for linear Gaussian models with a known superset of the adjacencies and k-triangle faithfulness. The global price is of order sqrt(n) once one edge can be malignant, so we study it locally, near a generic distribution with one edge absent. Our main result classifies this weak edge by Meek closures. It is benign if deleting it leaves the effect identifiable in the resulting MPDAG and Markov equivalent explanations agree; the local price is then bounded, and otherwise, for small k, of order sqrt(n). At benign edges the price is of order R+K, with R a ratio of efficient lengths and K a cross-model constant. It equals one, so that honesty is free, exactly when K = 0, generically decided 
    
[^19]: 算子值Q学习：面向顺序敏感决策的组合式贝尔曼递归

    Operator-Valued Q-Learning: Compositional Bellman Recursions for Order-Sensitive Decision Making

    [https://arxiv.org/abs/2611.30255](https://arxiv.org/abs/2611.30255)

    该论文提出算子值Q学习，将Q值表示为矩阵算子，通过保持不可交换动作算子有序乘积的贝尔曼递归，实现对顺序敏感决策问题的精确价值估计，并给出压缩性保证与几乎必然收敛的异步算子TD学习方案。

    

    在量子脉冲序列、航天器姿态控制以及手内物体重定向等任务中，动作以不可交换算子的形式作用于隐藏的物理状态，因此一个计划的结果取决于其动作的执行顺序。我们让价值函数承载这一结构：每个 Q(s,a) 是一个 N×N 算子，贝尔曼回溯将动作的对偶映射作用于后续价值，即 Q(s,a) = K_a^*(R(s,a) + γ E V(s'))。沿一个计划展开时，该递归构建动作算子的有序乘积，而任意潜在状态处的价值是一个线性迹读出。该递归对于不观测潜在状态的策略是精确的，回溯在酉动力学和正交动力学下是压缩的，并且一种异步算子时序差分（TD）方案可以以几乎必然收敛的方式学习它。经典价值存在于格结构中，因此动作上的最大值存在；而自伴算子构成一个反格（Kadison），不存在这样的最大值。算子化的 log-sum-exp 仍能为每个……给出一个算子（摘要原文在此处截断）。

    arXiv:2611.30255v1 Announce Type: new  Abstract: In quantum pulse sequences, spacecraft attitude control and in-hand reorientation, actions act on a hidden physical state as non-commuting operators, so the outcome of a plan depends on the order of its actions. We let the value function carry this structure: each Q(s,a) is an N x N operator, and the Bellman backup applies the action's dual map to the continuation value, Q(s,a) = K_a^*(R(s,a) + gamma E V(s')). Unrolled along a plan, the recursion builds the ordered product of the actions' operators, and the value at any latent state is a linear trace readout. It is exact for policies that do not observe the latent, the backup contracts for unitary and orthogonal dynamics, and an asynchronous operator TD scheme learns it with almost-sure convergence. Classical values live in a lattice, so a maximum over actions exists; self-adjoint operators form an anti-lattice (Kadison) and have none. An operator log-sum-exp still gives one operator per
    
[^20]: 重新审视语言模型中的持续学习：稳定性与可塑性的三接口诊断

    Revisiting Continual Learning in Language Models: A Three-Interface Diagnosis of Stability and Plasticity

    [https://arxiv.org/abs/2611.30151](https://arxiv.org/abs/2611.30151)

    该论文提出应通过已训练头部、冻结特征探针和无分类器主干诊断这三种评估接口来定位持续学习遗忘的发生位置（主干表征退化还是读出头失配），并主张BWT与FWT是依赖于接口的量而非单一的模型级属性。

    

    语言模型中的持续学习通常在增量学习（IL）设置下进行研究，即模型学习一系列任务，并通过其最终的分类器或语言模型（LM）头进行评估。然而，这一接口无法揭示遗忘发生在何处：准确率的下降可能反映主干网络中的表征退化、读出头中的失配，或两者兼有。因此，我们认为后向迁移（BWT）和前向迁移（FWT）应被解释为依赖于接口的量，而非单一的模型级属性。我们研究了三种评估接口：与主干一起按顺序更新的已训练头部；在冻结的主干特征上新训练的探针；以及无分类器的主干诊断。它们分别衡量部署性能、在新读出下的可恢复性，以及表征可分性。在针对编码器和解码器模型的类增量实验中……

    arXiv:2611.30151v1 Announce Type: cross  Abstract: Continual learning in language models is commonly studied in the incremental learning (IL) setting, where a model learns a sequence of tasks and is evaluated through its final classifier or language-model (LM) head. This interface, however, does not indicate where forgetting occurs: a drop in accuracy may reflect representation degradation in the backbone, readout mismatch in the head, or both. We therefore argue that backward transfer (BWT) and forward transfer (FWT) should be interpreted as interface-dependent quantities rather than as single modellevel properties. We study three evaluation interfaces: the trained head, which is updated sequentially with the backbone; a probe newly trained on frozen backbone features; and a classifier-free backbone diagnostic. They measure deployed performance, recoverability under a new readout, and representation separability, respectively. Across class-incremental experiments with encoder and deco
    
[^21]: 稀疏霍奇偏好优化：微调语言模型以实现对超越传递性偏好的对齐

    Sparse Hodge Preference Optimization: Fine-Tuning Language Models toward Alignment beyond Transitive Preferences

    [https://arxiv.org/abs/2611.30135](https://arxiv.org/abs/2611.30135)

    本文提出稀疏霍奇偏好优化（SHPO），借助组合霍奇分解理论为DPO引入稀疏反对称旋度残差，使其能够捕捉人类偏好数据中的非传递性与循环结构，并证明其严格扩大了DPO的假设类。

    

    直接偏好优化的广泛应用，反映了其在利用成对偏好数据微调大型语言模型以符合人类期望方面的巨大成功。它直接从人类偏好中优化策略，无需显式的奖励建模，也避免了基于强化学习方法所带来的复杂性。然而，经典的DPO依赖于Bradley-Terry采样模型，无法有效捕捉聚合的人类偏好数据中的非传递性或循环结构。在本文中，我们提出了稀疏霍奇偏好优化，这是一种偏好优化算法，通过受组合霍奇分解启发的稀疏反对称旋度残差来增强DPO。我们证明了在响应表示满足一些温和正则性条件的情形下，SHPO在包含至少一个三角形的比较图上严格扩大了DPO的假设类。我们进一步……（原文摘要在此处截断）

    arXiv:2611.30135v1 Announce Type: cross  Abstract: The growing adoption of Direct Preference Optimization (DPO), which leverages pairwise preference data, reflects its great success in fine-tuning large language models to align with human expectations. It directly optimizes the policy from human preferences without explicit reward modeling or the complexities of reinforcement learning-based approaches. However, the classic DPO relies on the Bradley--Terry sampling model and cannot effectively capture intransitivity or cyclic structures in aggregated human preference data. In this article, we propose Sparse Hodge Preference Optimization (SHPO), a preference optimization algorithm that augments DPO with a sparse antisymmetric curl residual motivated by the combinatorial Hodge decomposition. We prove that SHPO strictly enlarges DPO's hypothesis class on comparison graphs containing at least one triangle, under some mild regularity conditions on the response representations. We further est
    
[^22]: PsyDrift：用于自闭症谱系障碍异常评分的健康参照潜在扩散模型

    PsyDrift: Healthy-Reference Latent Diffusion for Anomaly Scoring in Autism Spectrum Disorder

    [https://arxiv.org/abs/2611.30050](https://arxiv.org/abs/2611.30050)

    PsyDrift通过仅从健康对照数据学习的潜在扩散模型，将功能连接表征与去噪重建的偏离转化为连续异常评分，在ABIDE-I全部20个站点跨站点验证中实现自闭症检测平均AUROC达0.798。

    

    个体异质性和采集站点变异性使基于神经影像学的自闭症谱系障碍（ASD）表征变得复杂。我们提出了PsyDrift，将ASD/健康对照监督训练的功能连接表征与健康参照潜在扩散相结合。编码器生成128维表征，去噪扩散概率模型（DDPM）仅从健康训练潜变量中学习。与扩散去噪重建之间的距离提供了一个连续的偏离分数。我们使用五种随机种子的留一站点交叉验证，评估了来自ABIDE-I全部20个站点的1,102名受试者。分类器和DDPM分别达到平均站点AUROC为0.798±0.003和0.718±0.013；不确定度表示种子标准差。在相同表征上匹配的健康参照评分器达到0.68-0.71；扩散模型的适度数值优势的统计显著性仍未确立（原文摘要在此处截断）。

    arXiv:2611.30050v1 Announce Type: new  Abstract: Individual heterogeneity and acquisition-site variability complicate neuroimaging-based characterization of autism spectrum disorder (ASD). We propose PsyDrift, combining ASD/healthy-control-supervised functional-connectivity representations with healthy-reference latent diffusion. An encoder produces 128-dimensional representations, and a denoising diffusion probabilistic model (DDPM) learns exclusively from healthy training latents. Distance from a diffusion-denoised reconstruction provides a continuous deviation score. We evaluate 1,102 subjects from all 20 ABIDE-I sites using five-seed leave-one-site-out cross-validation. The classifier and DDPM achieve mean-site AUROCs of $0.798 \pm 0.003$ and $0.718 \pm 0.013$, respectively; uncertainties denote seed standard deviations. Matched healthy-reference scorers on the same representation achieve 0.68-0.71; the statistical significance of diffusion's modest numerical advantage remains unes
    
[^23]: 路径积分控制的响应场方法

    The response field approach to path integral control

    [https://arxiv.org/abs/2611.30044](https://arxiv.org/abs/2611.30044)

    该论文开创性地将非平衡统计物理中的响应场形式体系引入路径积分最优控制，建立了响应场与随机最大值原理中后向伴随过程的对应关系，并证明最优控制由初始时刻响应场的平均值给出。

    

    本文在非平衡场论的响应场形式体系框架下研究非线性路径积分控制问题。这项工作具有探索性，其基础是建立后向伴随过程与非平衡统计物理中响应场之间的对应关系，而这些性质此前尚未在最优控制的背景下被系统阐述。研究发现，响应场在随机最大值原理下扮演着与后向伴随过程类似的角色。特别地，最优代价函数在初始时刻的梯度（与最优控制成正比）由初始时刻响应场的平均值给出。与非平衡动力学的情形不同，在控制背景下响应函数被发现是非因果的。本文进一步研究了路径积分控制的线性二次型情形，证实其相当于自由理论，即求解该控制问题……

    arXiv:2611.30044v1 Announce Type: cross  Abstract: This paper investigates non-linear path integral control problems within the response field formalism of nonequilibrium field theory. The aim of this work is exploratory, based on the identification of correspondences between backward adjoint processes and the response field of non-equilibrium statistical physics, whose properties have not been set out in an optimal control setting. The response field is found to play an analogous role to the backward adjoint process under the stochastic maximum principle. In particular, the gradient of the optimal cost-to-go at the initial time, which is proportional to the optimal control, is given by the average of the response field at the initial time. In contrast to the case of non-equilibrium dynamics, the response function is found to be acausal in the control setting. The linear-quadratic case of path integral control is studied, confirming it as the free theory, meaning that solving the contr
    
[^24]: 针对自适应对手的序贯策略选择

    Sequential Selection against an Adaptive Opponent

    [https://arxiv.org/abs/2611.30006](https://arxiv.org/abs/2611.30006)

    本文提出SPAR规则，通过与自适应对手的序贯练习对局累积证据并在私有随机阈值处停止，从而在极少误选已失效策略的前提下选出满足目标胜率的策略。

    

    通过在与对手的对抗测试中选择策略，这一做法在游戏AI、语言模型红队测试以及AI智能体安全评估中十分常见。在许多此类场景中，对手会根据其观察到的信息进行自适应调整，因此一个在实践中持续获胜的策略可能在真正投入使用时已经失效。受这些场景启发，我们研究了一个玩家在与自适应对手进行练习对局的过程中尝试各种策略、决定何时停止，然后为下一局对局选择一个策略或选择放弃的问题。其目标是在极少选中已低于目标胜率的策略的前提下，选出一个满足目标胜率的策略。针对单一策略的情形，我们提出了序贯私有累积超额规则，该规则通过带有上限的似然比来累积证据，并在此证据达到一个私有抽取的均匀阈值时选择该策略。我们证明了停止时间必须包含这样的 p…

    arXiv:2611.30006v1 Announce Type: cross  Abstract: Selecting a strategy by testing it against an opponent is common in game-playing AI, red-teaming of language models, and safety evaluations of AI agents. In many of these settings, the opponent adapts to what it observes, so a strategy that keeps winning in practice may already have stopped working by the time it is used. Motivated by these settings, we study a player who tries strategies in practice games against an adaptive opponent, decides when to stop, and then either selects one strategy for the next game or abstains. The goal is to select a strategy that meets a target win probability while rarely selecting one that has already fallen below it. For a single strategy, we propose the Sequential Private Accumulated-excess Rule (SPAR), which accumulates evidence through a capped likelihood ratio and selects the strategy once this evidence reaches a privately drawn uniform threshold. We show that the stopping time must involve such p
    
[^25]: 大型状态转移模型的训练动力学

    Training Dynamics of Large State-Transition Models

    [https://arxiv.org/abs/2611.30005](https://arxiv.org/abs/2611.30005)

    本文研究了在嵌入维度小于词表规模的条件下，从随机初始化出发联合训练词元嵌入与模型参数的训练动力学，揭示了大型状态转移模型实现准确下一词元预测的条件。

    

    下一词元预测依赖于学习得到的词元嵌入，其维度 $d$ 远小于词表大小 $V$。现有的训练动力学分析要么在 $d\ge V$ 的条件下固定嵌入不动，要么仅在受限设置中训练嵌入。当 $d$ 小于 $V$ 时，通过基于梯度的方法从随机初始化联合训练嵌入能否实现准确预测，目前仍不清楚。

    arXiv:2611.30005v1 Announce Type: new  Abstract: Next-token prediction relies on learned token embeddings whose dimension $d$ is much smaller than the vocabulary size $V$. Existing analyses of training dynamics either fix the embeddings with $d\ge V$ or train them only in restricted settings. It remains unclear whether jointly training the embeddings by gradient-based methods from random initialization leads to accurate prediction when $d
    
[^26]: 统计最优且近线性时间的假设选择

    Statistically Optimal and Near Linear Time Hypothesis Selection

    [https://arxiv.org/abs/2611.30003](https://arxiv.org/abs/2611.30003)

    本文提出了一种假设选择学习器，在期望意义下近线性运行时间、仅需 $O(\log(n/\delta)/\epsilon^2)$ 个样本即可达到统计最优的因子2近似，同时去除了Bousquet等人（FOCS 2021）因子2界中的对数因子。

    

    在假设选择问题中，给定可数域 $\mathcal{X}$ 上的 $n$ 个分布 $q_1,\dots,q_n$，以及来自未知分布 $p$ 的样本。目标是找到一个分布 $q$，使得 $\mathrm{TV}(p,q)\le\alpha\min_i\mathrm{TV}(p,q_i)+\epsilon$。对于使用与 $|\mathcal{X}|$ 无关的来自 $p$ 的样本量的算法而言，因子 $\alpha=2$ 是最优的，且通常只有非真学习器才能达到该因子。我们给出了一个学习器，它以至少 $1-\delta$ 的概率，使用 $O(\log(n/\delta)/\epsilon^2)$ 个来自 $p$ 的样本达到 $\alpha=2$。这与 Yatracos 因子-$3$ 估计器的样本复杂度相匹配，并去除了 Bousquet 等人（FOCS 2021）因子-$2$ 界中的对数因子。该学习器仅通过采样和逐点比较来访问 $q_i$。我们用预言机复杂度（即预言机调用次数）来衡量其运行时间。在期望意义下，它进行 $O((n\epsilon^{-2}+\epsilon^{-4}\log(1/\ldots$ 次预言机调用。

    arXiv:2611.30003v1 Announce Type: cross  Abstract: In hypothesis selection we are given $n$ distributions $q_1,\dots,q_n$ on a countable domain $\mathcal{X}$ and samples from an unknown distribution $p$. The goal is a distribution $q$ with $\mathrm{TV}(p,q)\le\alpha\min_i\mathrm{TV}(p,q_i)+\epsilon$. The factor $\alpha=2$ is optimal for algorithms using a number of samples from $p$ that is independent of $|\mathcal{X}|$, and in general only improper learners achieve it. We give a learner that achieves $\alpha=2$ with $O(\log(n/\delta)/\epsilon^2)$ samples of $p$, with probability at least $1-\delta$. This matches the sample complexity of Yatracos's factor-$3$ estimator, and it removes the logarithmic factors in the factor-$2$ bound of Bousquet et al. (FOCS 2021). The learner accesses the $q_i$ only through sampling and pointwise comparisons. We measure its running time by its oracle complexity, the number of oracle calls. In expectation it makes $O((n\epsilon^{-2}+\epsilon^{-4}\log(1/\
    
[^27]: 小波行列式采样器：自适应于正则性的最优逼近速率

    Wavelet determinantal samplers: Optimal approximation rates adapting to regularity

    [https://arxiv.org/abs/2611.29992](https://arxiv.org/abs/2611.29992)

    该论文构造了基于小波的行列式点过程采样器，在流形与几何图上对线性统计量实现了 n^{1-2s/m} 阶的方差缩减，且该逼近速率能够自适应于观测量的 Hölder/Sobolev 正则性。

    

    我们在流形与几何图上构造了基于小波的行列式采样器，其逼近速率能够自适应于观测量的正则性。在维度为 $m$ 的黎曼流形上，我们的小波行列式点过程（DPP）对于 $s$-Hölder 观测量和 $H^s$ 观测量的线性统计量实现了 $n^{1-2s/m}$ 阶的方差缩减，其中 $0 < s < m/2$。

    arXiv:2611.29992v1 Announce Type: cross  Abstract: We construct wavelet-based determinantal samplers on manifolds and geometric graphs whose approximation rates adapt to the regularity of the observable. On a Riemannian manifold of dimension $m$, our wavelet DPPs satisfy variance reduction of order $n^{1-2s/m}$ for linear statistics of both $s$-H\"older and $H^s$ observables, where $0
    
[^28]: 基于Matérn核的高斯过程极大似然估计的精确渐近理论

    Sharp Asymptotic Theory of Maximum Likelihood Estimation for Gaussian Processes with a Matern Kernel

    [https://arxiv.org/abs/2611.29841](https://arxiv.org/abs/2611.29841)

    本文为Matérn核高斯过程参数的联合极大似然估计建立了精确的渐近理论，刻画了不同观测域维度下可估计的参数及其收敛速度，弥补了从参数可辨识性到估计量渐近性质之间的理论空白。

    

    高斯过程（GP）通过协方差函数（即核函数）为建模复杂的依赖结构提供了一个灵活的框架，核函数的参数决定了潜在过程的变异性、相关性和光滑性。Matérn核因其在建模不同光滑度方面的灵活性而成为最流行的选择之一。尽管Matérn核参数在固定域渐近下的可辨识性已被完全理解，但仅有可辨识性并不能确立实践中常用的极大似然估计量（MLE）的一致性或渐近行为。本文通过为Matérn核参数的联合极大似然估计建立精确的渐近理论来弥补这一空白。由于参数的可辨识性关键取决于观测域的维度，我们刻画了在不同维度下可估计的参数，并建立了它们的收敛速度……

    arXiv:2611.29841v1 Announce Type: cross  Abstract: Gaussian processes (GPs) provide a flexible framework for modeling complex dependence structures through a covariance function, or kernel, whose parameters determine the variability, correlation, and smoothness of the underlying process. The Mat\'ern kernel is among the most popular choices due to its flexibility in modeling different degrees of smoothness. Although the identifiability of Mat\'ern kernel parameters under fixed-domain asymptotics is completely understood, identifiability alone does not establish the consistency or asymptotic behavior of the maximum likelihood estimators (MLEs) routinely used in practice. In this paper, we address this gap by developing a sharp asymptotic theory for the joint MLE of Mat\'ern kernel parameters. Since parameter identifiability depends critically on the dimension of the observation domain, we characterize the estimable parameters and establish their convergence rates across different dimens
    
[^29]: 用于加权回归与分类的表格基础模型

    Tabular Foundation Models for Weighted Regression and Classification

    [https://arxiv.org/abs/2611.29789](https://arxiv.org/abs/2611.29789)

    提出了 WeightedPFN，通过加权注意力、归一化加权损失以及基于缺失数据问题的合成加权任务预训练，将表格基础模型 TabPFN-3.5 适配到加权回归与分类中，在真实数据集上表现优异。

    

    加权回归与分类的应用范围广泛，涵盖缺失数据分析、因果推断、保险定价以及粒子物理学等领域。现有的加权方法必须在每个数据集上进行拟合与调优，这对于灵活的非线性学习器而言代价高昂。表格基础模型可以避免这一成本：Transformer 在众多模拟数据集上仅预训练一次，随后通过上下文学习在新数据集上进行预测，将观测数据作为上下文使用，无需进一步拟合。近期诸如 TabPFN-3.5 等模型已超越经过调优的传统方法，但它们无法利用样本权重。我们开发了一套通用流程，通过加权注意力机制、归一化加权损失，以及基于缺失数据问题生成的合成加权预测任务的预训练，将此类模型适配到加权预测中。将该流程应用于 TabPFN-3.5，我们得到了 WeightedPFN，它在真实数据集上表现强劲，在缺失数据问题和分布外场景中均有出色表现。

    arXiv:2611.29789v1 Announce Type: cross  Abstract: Weighted regression and classification have applications ranging from missing-data analysis and causal inference to insurance pricing and particle physics. Existing weighted methods must be fitted and tuned on each dataset, which is costly for flexible nonlinear learners. Tabular foundation models avoid this cost. A Transformer is pretrained once on many simulated datasets and then predicts on a new dataset by in-context learning, using its observations as context without further fitting. Recent models such as TabPFN-3.5 surpass tuned conventional methods, but they cannot use sample weights. We develop a general pipeline that adapts such models to weighted prediction through weighted attention, a normalized weighted loss, and pretraining on synthetic weighted prediction tasks from missing-data problems. Applied to TabPFN-3.5, it yields WeightedPFN, which performs strongly on real datasets, both in missing-data problems and in out-of-di
    
[^30]: 通过学习的 Doob h-函数从二元评估实现约束扩散生成

    Constrained Diffusion from Binary Evaluations via Learned Doob h-Functions

    [https://arxiv.org/abs/2611.29779](https://arxiv.org/abs/2611.29779)

    该论文提出基于 Doob h-变换的约束扩散生成框架，仅用二元样本评估学习“含噪样本去噪后满足约束的概率”这一单一函数，无需可微测量算子、成对数据或逐步 rollout，即可在冻结基础模型上实现带逆概率因子的精确约束引导。

    

    对于扩散模型而言，在只能在生成完成的样本上进行检验的约束条件下进行条件生成（例如分子性质过滤器）仍然具有挑战性：大多数引导方法需要可微的测量算子、成对的条件数据，或者在每一步都要对基础模型进行 rollout 采样。我们提出了一个基于 Doob h-变换的框架，将条件生成简化为学习单一的函数——即一个含噪样本去噪后落入约束集的概率。我们仅通过二元（binary）的干净样本评估来学习该函数，对前向加噪的标签进行平方误差回归；基础模型保持冻结，每个约束只需训练一个轻量级输出头，而含噪状态下的分类器引导则作为该框架的特例自然出现。精确的引导步骤在约束信号可证明消失之处会自然衰减，并且携带一个跨样本逆概率因子——这是归一化引导所丢弃的：随调度（schedule）……（摘要此处截断）

    arXiv:2611.29779v1 Announce Type: new  Abstract: Conditional generation under constraints that can only be checked on a finished sample, such as a molecular property filter, remains challenging for diffusion models: most guidance methods require differentiable measurement operators, paired conditional data, or rollouts of the base model at every step. We present a framework grounded in the Doob $h$-transform that reduces conditional generation to learning a single function, the probability that a noisy sample denoises into the constraint set. We learn this function from binary clean-sample evaluations alone, by squared-error regression on forward-noised labels; the base model is frozen, one lightweight head is trained per constraint, and noisy-state classifier guidance arises as a special case. The exact guidance step fades where the constraint signal provably vanishes and carries a factor of the inverse probability across samples, which normalized guidance discards: with the schedule 
    
[^31]: 面向组合扩散的稳定采样动力学选择：一种Feynman-Kac权重约简方法

    Selecting Stable Sampling Dynamics for Compositional Diffusion: A Feynman-Kac Weight Reduction

    [https://arxiv.org/abs/2611.29764](https://arxiv.org/abs/2611.29764)

    该论文在Feynman-Kac加权扩散框架中确定λ=1/B（B为各专家权重之和）这一特殊取值，此时瞬时对数权重中的聚合平方分数项恰好完全抵消，从而为组合扩散采样选出最稳定的提议动力学。

    

    给定K个分别独立训练的扩散模型，如何在不联合重新训练专家模型的情况下，从其终端分布的β_k加权乘积中进行采样？现有的Feynman-Kac构造提供了许多具有正确目标边缘分布的加权随机微分方程（SDE），随之而来的关键问题是如何从中选择有用的提议动力学。我们研究了一族显式的加权扩散过程，它们实现相同的“中间专家乘积”轨迹。标量参数λ控制提议漂移跟随组合专家分数的强度，而额外的朗之万分量可以独立变化。记B=Σ_{k=1}^{K}β_k>0，我们确定了特殊取值λ=1/B。在该取值下，瞬时Feynman-Kac对数权重速率中的聚合平方分数项恰好完全抵消。对于正的专家指数，剩余的分数分歧贡献项……（摘要原文在此处被截断）

    arXiv:2611.29764v1 Announce Type: cross  Abstract: Given $K$ separately trained diffusion models, how should one sample from a $\beta_k$-weighted product of their terminal laws without jointly retraining the experts? Existing Feynman--Kac constructions provide many weighted stochastic differential equations (SDEs) with the correct target marginal. The consequential question is how to select useful proposal dynamics among them. We study an explicit family of weighted diffusions that realizes the same product-of-intermediate-experts trajectory. A scalar parameter $\lambda$ controls how strongly the proposal drift follows the composed expert score, while an additional Langevin component can be varied independently. Writing $B=\sum_{k=1}^{K}\beta_k>0$, we identify the distinguished choice $\lambda=1/B$. At this value, the aggregate squared-score term in the instantaneous Feynman--Kac log-weight rate cancels exactly. For positive expert exponents, the remaining score-disagreement contributi
    
[^32]: probly：不确定性感知的机器学习

    probly: Uncertainty-Aware Machine Learning

    [https://arxiv.org/abs/2611.29762](https://arxiv.org/abs/2611.29762)

    probly是一个模块化Python包，通过轻量级转换将现有模型（支持PyTorch、scikit-learn和Flax）变为不确定性感知模型，并在统一框架内集成二阶分布、信度集合和保形预测等多种不确定性表示与量化方法。

    

    随着机器学习系统越来越多地部署于实际应用中，如何表示和量化不确定性已从方法论上的次要问题转变为核心关注点。然而在实践中，让模型具备不确定性感知能力仍然出奇地困难：相关工具往往分散在各个库中，每个库都绑定于特定的框架和特定的不确定性建模方法，而表示、量化和评估方式的选择通常在缺乏指导的情况下留给用户自行决定。在本文中，我们提出了probly，一个以单一模块化框架解决这些问题的Python包。probly提供：(i) 轻量级转换，可将现有模型转变为具有不确定性感知能力的模型，目前支持PyTorch、scikit-learn和Flax；(ii) 多种不确定性表示方式，包括二阶分布、信度集合（credal sets）和保形预测（conformal prediction）；(iii) 相应的量…（摘要在此处截断）

    arXiv:2611.29762v1 Announce Type: new  Abstract: As machine learning systems are increasingly deployed in real-world applications, the question of how to represent and quantify uncertainty has moved from a methodological side issue to a central concern. In practice, however, making a model uncertainty-aware is still surprisingly difficult: relevant tools tend to be scattered across libraries, each tied to a particular framework and a particular approach to modeling uncertainty, and the choice of representation, quantification, and evaluation is typically left to the user without much guidance. In this paper, we present probly, a Python package that addresses these issues in a single, modular framework. probly offers (i) lightweight transformations that turn existing models into uncertainty-aware ones, currently supporting PyTorch, scikit-learn, and Flax, (ii) several representations, including second-order distributions, credal sets, and conformal prediction, (iii) the corresponding qu
    
[^33]: 语法稀疏性决定Transformer学习随机语言的方式

    Grammar Sparsity Sets How Transformers Learn Random Languages

    [https://arxiv.org/abs/2611.29692](https://arxiv.org/abs/2611.29692)

    语法稀疏性同时决定了Transformer学习随机语言的顺序和速度——稠密语法分层学习且经历随词表大小变宽的平台期（$t\propto V$ 到 $t\propto V^3$），而稀疏语法则一次性快速习得（$t\propto\sqrt V$）。

    

    我们研究了在随机语言模型上训练的Transformer的语言习得过程：随机语言模型是一种合成的分层语法，具有随机的产生式规则，其稀疏性由单一参数控制，且其条件熵可以被精确计算。在深度为二的语法上，我们发现稀疏性决定了学习的顺序和速度。稠密语法的 learned 一次一层：token 在 $t\propto V$ 步后可以从其兄弟节点预测，而只有在 $t\propto V^{3}$ 步后才能从完整上下文中预测，因此这两个学习阶段被一个随 $V$ 增大而变宽的平台期分隔开来。稀疏语法则是一次性全部学习，在 $t\propto\sqrt V$ 步内完成，其标度指数甚至比稠密语法的第一阶段还要小。指数 $1$ 和 $1/2$ 在深度为一的情形中就已经出现，我们对使用 Adam 训练的线性模型推导出了这两个指数。

    arXiv:2611.29692v1 Announce Type: new  Abstract: We study language acquisition by transformers trained on the Random Language Model: a synthetic hierarchical grammar with random production rules, whose sparsity is controlled by a single parameter and whose conditional entropies are known exactly. On depth-two grammars, we find that sparsity sets both the order and the speed of learning. Dense grammars are learnt one level at a time: a token is predicted from its sibling after $t\propto V$ steps and from the full context only after $t\propto V^{3}$, so the two stages are separated by a plateau that widens with $V$. Sparse grammars are learnt all at once, at $t\propto\sqrt V$, with a smaller exponent than even the first stage of a dense one. The exponents $1$ and $1/2$ already appear at depth one, where we derive them for a linear model trained with Adam.
    
[^34]: 多时间尺度记忆如何“冷却”优化：来自AdEMAMix的理论与证据

    How Multi-Timescale Memory Cools Optimization: Theory and Evidence from AdEMAMix

    [https://arxiv.org/abs/2611.29588](https://arxiv.org/abs/2611.29588)

    该论文揭示多时间尺度优化器记忆通过改变参数范数来控制参数旋转速度，证明AdEMAMix中独立加权的慢速动量本质上充当一种内生的退火调度，从而解释了其在语言模型预训练中带来收益的机制。

    

    在AdamW中加入多个动量时间尺度可以改进大语言模型的预训练，但驱动这些收益的机制仍知之甚少。我们识别出多时间尺度优化器记忆与归一化网络几何结构之间的相互作用：记忆会改变参数范数，从而控制参数旋转的快慢。从解耦权重衰减下的精确范数动力学出发，我们证明参数范数取决于更新的时间相关性，而权重衰减控制这些相关性影响优化的时间长度。我们展示了独立加权的慢速动量如何能显著增大参数范数，同时对瞬时更新的贡献微乎其微，因此充当了一种内生的退火调度。我们在AdEMAMix的语言模型预训练中研究了这一机制。受控干预实验表明，这种调度机制具有明确的实际意义。

    arXiv:2611.29588v1 Announce Type: cross  Abstract: Augmenting AdamW with multiple momentum timescales improves large language model pretraining, yet the mechanism driving these gains remains poorly understood. We identify an interaction between multi-timescale optimizer memory and the geometry of normalized networks: memory changes parameter norms and thereby controls how fast parameters rotate. Starting from exact norm dynamics under decoupled weight decay, we show that the parameter norm depends on the temporal correlation of the updates, with weight decay controlling how long these correlations influence optimization. We show how an independently weighted slow momentum can substantially increase the norm of parameters, while contributing negligible power to the instantaneous update, therefore acting as an endogenous annealing schedule. We study this mechanism in AdEMAMix for language-model pretraining. Controlled interventions show that this scheduling mechanism has clear practical 
    
[^35]: 从学习率到旋转调度：球面上的有限时域理论

    From Learning Rates to Rotational Schedules: Finite-Horizon Theory on the Sphere

    [https://arxiv.org/abs/2611.29579](https://arxiv.org/abs/2611.29579)

    该论文将学习率调度设计重新表述为球面上的旋转调度问题，推导了任意旋转调度的最后一次迭代收敛保证，证明最优调度当时域增长时趋近于线性衰减，为大语言模型预训练实践提供了有用的理论指导。

    

    在有限的训练时域内，步长应该如何调度？余弦调度和预热-稳定-衰减（WSD）调度在大规模预训练中被广泛采用，然而它们对优化的影响还依赖于参数范数。归一化可以使参数组具有尺度不变性，从而使得损失仅依赖于参数的方向，而参数的范数仍然会影响参数旋转的速度。在本工作中，我们直接在旋转运动的空间中表述调度设计。在测地凸的球面设置下，我们为任意旋转调度推导了最后一次迭代的收敛保证，该保证将调度的整体规模与其在时间上的分布分离开来。我们推导出了最小化该边界的唯一调度，并证明随着训练时域的增长，该调度趋近于线性衰减。在实证方面，我们的旋转解释和理想化的球面分析为实践中大语言模型的预训练提供了有用的预测。

    arXiv:2611.29579v1 Announce Type: cross  Abstract: How should step sizes be scheduled across a finite training horizon? Cosine and warmup-stable-decay (WSD) schedules are widely adopted in large-scale pretraining, yet their effect on optimization also depends on parameter norms. Normalization can make parameter groups scale invariant, so that the loss depends only on their direction, while their norms still affect how fast the parameters rotate. In this work, we formulate schedule design directly in the space of rotational motion. Working within a geodesically convex spherical setting, we derive a last-iterate convergence guarantee for arbitrary rotational schedules that separates their overall scale from their distribution over time. We derive the unique schedule that minimizes this bound and prove that it approaches linear decay as the horizon grows. Empirically, our rotational interpretation and idealized spherical analysis provide useful predictions for LLM pretraining in practice,
    
[^36]: 面向含交易成本的在线投资组合选择的门控收益加权净利润最大化

    Gated Payoff-Weighted Net Profit Maximization For Online Portfolio Selection With Transaction Costs

    [https://arxiv.org/abs/2611.29552](https://arxiv.org/abs/2611.29552)

    针对AOLMA预测高估实际收益的问题，本文提出一种门控机制来估计预测被实现的程度，并将其作为资产特定的成本阈值应用于净利润最大化规则，再通过收益加权基金组合多个门控NPM专家，从而在存在交易成本的条件下实现更可靠的在线投资组合选择。

    

    本文研究在线投资组合选择（OLPS）问题。将净利润最大化（NPM）作用于自适应在线移动平均（AOLMA）的策略，即投资组合选择中的AOLNPM策略，在预测收益超过交易成本时进行交易。从本质上讲，该方法严重依赖于经过校准的预测。事实上，我们证明AOLMA预测会高估实际实现的相对收益。因此，我们设计了一个门控机制来调整AOLMA预测的使用强度；对于每种资产，它通过与中性预测（即不变）对比的预测似然，学习近期分配给AOLMA的混合比例。因此，该比例可以估计预测被实现的程度。随后，我们将由此产生的预测输入到NPM投资组合规则中，其中门控充当针对特定资产的成本阈值。收益加权基金按固定衰减因子以其财富比例组合多个门控NPM专家，且无需学习……

    arXiv:2611.29552v1 Announce Type: new  Abstract: In this paper, we study online portfolio selection (OLPS) problems. The net profit maximization (NPM) acting upon Adaptive Online Moving Average (AOLMA), namely the AOLNPM strategy in portfolio selection, trades when forecast gain exceeds the transaction cost. In essence, this method relies heavily on calibrated forecasts. Indeed, we show that AOLMA forecasts overstate the realized relative returns. We therefore design a gate that adjusts how strongly an AOLMA forecast is used; for each asset, it learns the recent mixture proportion assigned to AOLMA from predictive likelihoods against a neutral forecast (no change). The proportion therefore estimates the extent to which the forecast is realized. We then feed the resulting forecast into the NPM portfolio rule, where the gate acts as an asset-specific cost threshold. A payoff-weighted fund combines gated NPM experts with fixed decay factors in proportion to their wealth and needs no learn
    
[^37]: GeoFusion：一个面向传感器融合的不确定性机器学习框架

    GeoFusion: An Uncertain Machine Learning Framework for Sensor Fusion

    [https://arxiv.org/abs/2611.29510](https://arxiv.org/abs/2611.29510)

    GeoFusion提出一个三阶段不确定性机器学习框架，将GPS观测建模为不确定性对象（火山函数与山峰函数）并按位置事件聚类，从而利用多部智能手机的协作传感器融合提升定位精度。

    

    高定位精度是许多全球定位系统（GPS）应用中的关键需求，然而利用价格低廉的智能手机传感器实现高精度定位仍然是一个悬而未决的挑战。现有方法将每个GPS观测视为一个绝对点，丢弃了测量条件中固有的不确定性，也未能发掘多部手机观测同一位置时的协作潜力。在本工作中，我们提出了GeoFusion，一个用于协作传感器融合的三阶段不确定性机器学习框架。在第一阶段，每个GPS观测通过两种提出的模型之一被建模为不确定性对象：基于卫星几何的火山函数，以及基于真实智能手机数据统计结构经验推导的山峰函数。在第二阶段，使用划分式和基于图的算法按位置事件对不确定性对象进行聚类。在第三阶段，聚类（原文摘要在此处截断）

    arXiv:2611.29510v1 Announce Type: new  Abstract: High positioning precision is a critical requirement in many Global Positioning System (GPS) applications, yet achieving it with affordable smartphone sensors remains an open challenge. Existing approaches treat each GPS observation as an absolute point, discarding the uncertainty inherent in the measurement conditions and leaving the cooperative potential of multiple phones observing the same location untapped. In this work, we propose GeoFusion, a three-stage uncertain machine learning framework for cooperative sensor fusion. In the first stage, each GPS observation is modeled as an uncertain object using one of two proposed models: the volcano function, grounded in satellite geometry, and the empirically derived mountain function, built from the statistical structure of real smartphone data. In the second stage, uncertain objects are clustered by location event using partitional and graph-based algorithms. In the third stage, the clus
    
[^38]: 语言模型更新序贯决策的最优标注设计

    Optimal Labelling Designs for Sequential Decisions on Language Model Updates

    [https://arxiv.org/abs/2611.29416](https://arxiv.org/abs/2611.29416)

    该论文提出SOLD方法，通过结合廉价的LLM裁判评估与少量人工标注，序贯地选择提示并决定停止时机，在保证错误概率可控的前提下最快地判断语言模型更新后准确率的变化。

    

    当大型语言模型（LLM）更新时，其开发者必须判断模型的准确率是提升了、下降了，还是保持在容忍范围之内。LLM裁判可以低成本地对每个提示进行评估，但评估并不完全准确，而人工标注则成本高昂。我们提出了带最优标注设计的序贯决策方法（SOLD），该方法每次标注一个提示，利用LLM裁判来校正准确率变化的估计并选择提示，一旦证据足够确凿即停止标注。只要标注停止时做出决策，错误决策的概率不会超过预设水平。常用的Neyman分配旨在最小化估计的方差，因此很少会标注那些裁判非常有把握的提示；而当这些提示的结果仍可能远低于裁判的预测时，该方法可能需要多得多的抽样次数。我们转而推导出能够最快收集证据的标注设计，该设计以至少与……成比例的概率标注每个提示（摘要在此处被截断）。

    arXiv:2611.29416v1 Announce Type: cross  Abstract: When a large language model (LLM) is updated, its developer must decide whether accuracy improved, worsened or stayed within a tolerance. An LLM judge can assess every prompt cheaply but imperfectly, while human labels are costly. We propose Sequential Decisions with Optimal Labelling Designs (SOLD), which labels one prompt at a time, uses the judge to correct the estimate of the change in accuracy and to choose the prompts, and stops once the evidence is conclusive. A wrong decision has probability at most a prespecified level whenever labelling stops. The commonly used Neyman allocation minimises the variance of the estimate and therefore rarely labels prompts that the judge is confident about. It can need far more draws when the outcomes of these prompts could still fall far below the judge's prediction. We instead derive the design that gathers evidence fastest, which labels each prompt with probability at least proportional to how
    
[^39]: 高度自适应Lasso的结点筛选算法比较

    A comparison of knot point screening algorithms for highly adaptive lasso

    [https://arxiv.org/abs/2611.29270](https://arxiv.org/abs/2611.29270)

    本文比较了高度自适应lasso（HAL）中不同的结点筛选算法，旨在保持其快速收敛率的同时提升该非参数回归估计量的计算效率。

    

    高度自适应lasso（HAL）是一种非参数回归估计量，其收敛速度为 $n^{-1/3}(\log n)^{d/2}$，其中数据维度 $d$ 仅出现在样本量对数的幂次中。这一快速收敛速度对许多因果推断任务至关重要，因为在这些任务中，有效估计通常要求干扰参数以特定的速度被估计。HAL 能够达到这一收敛速度，并且当与半参数有效估计量（例如通过目标最大似然估计（TMLE）构建的估计量）相结合时，可用于开发有效估计量。除因果推断外，HAL 还适用于广泛的监督机器学习任务。HAL 的实施过程包括首先对原始设计矩阵 $X$ 进行基展开，然后计算 lasso 解路径，并使用交叉验证来选择系数的最优 $L_1$ 范数。对于所有协变量均为连续变量且没有截断的数据集……

    arXiv:2611.29270v1 Announce Type: cross  Abstract: Highly adaptive lasso (HAL) is a nonparametric regression estimator with a rate of convergence $n^{-1/3}(\log n)^{d/2}$, where the data dimension $d$ only appears in the power of the log of sample size. This fast rate is crucial for many causal inference tasks, where efficient estimation typically requires nuisance parameters to be estimated at a specific rate. HAL achieves this rate and, when paired with semiparametrically efficient estimators -- such as those constructed using targeted maximum likelihood estimation (TMLE) -- can be used to develop efficient estimators. Beyond causal inference, HAL is applicable to a wide range of supervised machine learning tasks. The procedure for HAL involves first performing a basis expansion on the original design matrix $X$ and then computing a lasso solution path. Cross-validation is used to select the optimal $L_1$-norm for the coefficients. For datasets with all continuous covariates and no t
    
[^40]: 模糊e均值：基于移动平均小批量更新的大数据模糊c均值

    Fuzzy e-Means: Fuzzy c-Means on Large Data with Moving-Average Mini-Batch Updates

    [https://arxiv.org/abs/2611.29241](https://arxiv.org/abs/2611.29241)

    本文提出模糊e均值（e-FCM），将e均值的移动平均小批量优化器迁移到模糊c均值上，并通过先硬分配初始化、再模糊更新的两阶段策略解决了空簇误删和局部极小值问题，实现了大数据上的高效模糊聚类。

    

    模糊c均值（FCM）为每个数据点赋予其在每个聚类中的隶属度。其标准求解器——交替优化（AO）——每次迭代需要对数据进行完整遍历，且往往需要多次迭代；而专为大数据设计的方法要么收敛到较差的目标函数值，要么仍需数十次数据遍历。我们采用e均值的优化器来运行FCM。e均值是近期提出的一种小批量k均值方法，它对每个聚类中的点数和点的坐标总和保持移动平均，并通过缩放定律设定批量大小和步长。该优化器之所以能够迁移，是因为FCM的质心同样是一个加权总和除以总权重。但这种直接迁移会在两个方面失效：e均值中重新安置近乎空聚类的规则会将小而远的群体的质心也移除；此外，从随机点出发的FCM可能陷入较差的局部极小值。因此，模糊e均值（e-FCM）先执行一遍基于硬分配的e均值，随后运行模糊的移动平均遍历。

    arXiv:2611.29241v1 Announce Type: new  Abstract: Fuzzy c-means (FCM) gives every point a degree of membership in every cluster. Its standard solver, alternating optimization (AO), needs a full pass over the data per iteration and often many iterations, while methods built for large data stop at a worse objective or still need tens of passes. We run FCM with the optimizer of e-means, a recent mini-batch k-means that keeps moving averages of the number and the sum of the points in each cluster and sets its batch size and step size by scaling laws. The optimizer carries over because an FCM centroid is also a weighted sum divided by a total weight, but this direct transfer fails in two ways: the e-means rule that relocates nearly empty clusters also removes the centroids of small, distant groups, and FCM started from random points can settle in poor local minima. Fuzzy e-means (e-FCM) therefore starts with one pass of e-means with hard assignments and then runs fuzzy moving-average passes 
    
[^41]: 用于稀疏时空计数的深度负二项自回归：估计与预测分布界

    Deep Negative Binomial Autoregression for Sparse Spatio-Temporal Counts: Estimation and Predictive Distribution Bounds

    [https://arxiv.org/abs/2611.29207](https://arxiv.org/abs/2611.29207)

    该论文提出深度负二项自回归模型DeepNB-STAR用于稀疏、过度离散的时空计数数据，首次在无混合性假设且允许站点间任意依赖的条件下建立了Hellinger风险的oracle不等式、几何绝对正则性以及h步预测分布误差的理论界，并在日本监测数据上取得了更优的预测性能。

    

    我们研究稀疏、过度离散计数面板数据的估计与概率预测问题，提出DeepNB-STAR——一种负二项自回归模型，其对数均值与对数尺度参数是深度ReLU网络的输出，该网络以过去观测的时间性摘要和邻域摘要作为输入，并通过极大似然进行拟合。我们证明了沿观测轨迹的Hellinger风险的oracle不等式，无需混合性假设，其收敛速率由复合光滑性决定，并对某个Hölder子类给出了匹配的下界；同时证明了模型具有几何绝对正则性，以及h步预测分布误差的界，其阶为h^{1/2}乘以估计风险的平方根再加上蒙特卡洛误差项。所有结果均适用于依赖于协变量的尺度参数，且估计界在站点之间任意相关的情况下仍然成立，并以时间点数量作为样本量。在四个日本传染病监测面板数据上，DeepNB-STAR的加权区间分数比对照方法低0.6%至1.0%（原文此处截断）。

    arXiv:2611.29207v1 Announce Type: cross  Abstract: We study estimation and probabilistic forecasting for sparse, overdispersed count panels using DeepNB-STAR, a negative binomial autoregression whose log-mean and log-size are outputs of a deep ReLU network of temporal and neighboring summaries of the past, fitted by maximum likelihood. We prove an oracle inequality for the Hellinger risk along the observed trajectory without mixing assumptions, rates governed by compositional smoothness with a matching lower bound for a H\"older subclass, geometric absolute regularity, and bounds on the error of h-step predictive distributions of order h^{1/2} times the root estimation risk plus a Monte Carlo term. All results hold for a size that depends on the covariates, and the estimation bounds hold under arbitrary dependence between sites, with the number of time points as sample size. On four Japanese surveillance panels, DeepNB-STAR has a weighted interval score 0.6% to 1.0% below that of a thr
    
[^42]: 为什么拉平P值直方图有效：基于张量分解的无监督特征提取中标准差优化的理论

    Why flattening the P-value histogram works: a theory of the standard-deviation optimization in tensor-decomposition-based unsupervised feature extraction

    [https://arxiv.org/abs/2611.29201](https://arxiv.org/abs/2611.29201)

    该论文首次为基于张量分解的无监督特征提取中“拉平P值直方图以优化标准差”这一启发式方法提供了理论证明，表明背景基因的分量近似服从标准差为√d/s的正态分布，且信号基因即使占全部基因的正比例也能与背景分离。

    

    背景：基于张量分解的无监督特征提取（TD-based UFE）通过在零假设正态分布下将P值归因于基因奇异向量的各个分量来筛选基因。该分布的标准差σ是未知的，其选取方式是使未被选中的基因的P值直方图尽可能平坦。这种启发式方法在实践中效果良好，但一直缺乏理论依据。结果：我们针对包含信号、平均表达和噪声的模型证明：高阶奇异值分解具有能够重现实验设计的分量；其基因奇异向量中属于背景基因的分量近似服从标准差为σ₀ = √d/s的正态分布（其中d为噪声方差，s为奇异值）；并且即使信号基因构成全部基因的一个正比例，它们也能与背景基因分离。

    arXiv:2611.29201v1 Announce Type: cross  Abstract: Background: Tensor-decomposition-based unsupervised feature extraction (TD-based UFE) selects genes by attributing P-values to the components of a gene singular vector under a null normal distribution. The standard deviation sigma of this distribution is unknown and is chosen so that the histogram of the P-values of the genes that are not selected becomes as flat as possible. This heuristic has worked well in practice but has had no theoretical justification. Results: We prove, for a model of signal, mean expression and noise, that the higher-order singular value decomposition has a component that reproduces the experimental design, that the components of its gene singular vector belonging to background genes are approximately normal with standard deviation sigma_0 = sqrt(d)/s, where d is the noise variance and s the singular value, and that signal genes separate from the background even when they form a positive fraction of all genes.
    
[^43]: 次高斯HDLSS主成分分析：NRM、CDM与分块噪声降噪的高概率理论与数值证据

    Sub-Gaussian HDLSS Principal Component Analysis: High-Probability Theory and Numerical Evidence for NRM, CDM, and Block Noise Reduction

    [https://arxiv.org/abs/2611.29058](https://arxiv.org/abs/2611.29058)

    本文在独立因子次高斯模型下为HDLSS主成分分析建立了基于得分矩阵、体移与高概率余项的高概率分解理论，揭示了序贯降噪对应含后续尖峰减除的得分预言机而分块校正可避免该减除的机制，并解释了交叉数据矩阵方法中样本分裂的收益与代价，同时以千次重复模拟加以验证。

    

    我们在独立因子次高斯模型下研究维数 d 远大于样本量 n 时的主成分分析。将对偶样本协方差矩阵分解为一个有限秩得分矩阵、一个各向同性体移以及一个由体迹 τ₀、Frobenius 尺度 q₀ 和最大噪声特征值 b₀ = ||Λ₀||_op 所控制的高概率余项。该分解将固定得分情形下的 HDLSS 极限与 n 同步增长的联合极限区分开来。理论表明，序贯噪声降噪通常瞄准的是包含后续尖峰减除项的得分预言机，而在信号秩已知时，预言机分块校正可以避免这一减除。与之平行的交叉数据矩阵界则解释了样本分裂带来的收益与代价。每个配置重复 1000 次的可复现高斯因子模拟证实了得分预言机逼近、多尖峰减除以及特征值之间的分离。

    arXiv:2611.29058v1 Announce Type: cross  Abstract: We study principal component analysis when the dimension d is much larger than the sample size n under an independent-factor sub-Gaussian model. The dual sample covariance is decomposed into a finite-rank score matrix, an isotropic bulk shift, and a high-probability remainder governed by the bulk trace \tau_0, Frobenius scale q_0, and largest noise eigenvalue b_0 = ||\Lambda_0||_{op}. This decomposition separates fixed-score HDLSS limits from joint limits in which n also grows. It shows that sequential noise reduction generally targets a score oracle containing a later-spike subtraction, while an oracle block correction avoids that subtraction when the signal rank is known. A parallel cross-data-matrix bound explains the benefit and cost of sample splitting. Reproducible Gaussian-factor simulations with 1,000 replications per configuration confirm the score-oracle approximation, the multi-spike subtraction, the separation between eigen
    
[^44]: 稳定性边缘处的镜像下降

    Mirror Descent at the Edge of Stability

    [https://arxiv.org/abs/2611.29043](https://arxiv.org/abs/2611.29043)

    该论文提出“镜像锐度”概念并推导对偶中心流ODE，首次将梯度下降的稳定性边缘现象研究推广到非欧几里得的镜像下降设定，并实验发现镜像锐度徘徊在 $2/\eta$ 阈值稍上方。

    

    在许多深度学习场景中，梯度下降已被证明运行在稳定性边缘——即损失函数Hessian矩阵的最大特征值徘徊在 $2/\eta$ 附近的区域。我们在镜像下降（梯度下降的一种推广）中研究这一现象。由于镜像下降是非欧几里得的，现有的对稳定性边缘阶段的解释并不能直接推广到这一设定。我们推广了现有的锐度（sharpness）概念，引入了“镜像锐度”，用于衡量损失相对于镜像势函数的曲率。在实验上，我们观察到镜像锐度徘徊在 $2/\eta$ 阈值稍上方。此外，我们推导了镜像下降的对偶中心流ODE，以连续时间方式建模稳定性边缘发生时的迭代过程。我们的ODE为理解将“锐度为何稳定在 $2/\eta$”的现有论证从梯度下降推广到完全一般的镜像下降所面临的困难提供了洞察。

    arXiv:2611.29043v1 Announce Type: new  Abstract: In many deep learning settings, gradient descent has been shown to operate at the edge of stability -- a regime where the largest eigenvalue of the loss Hessian hovers around $2/\eta$. We study this phenomenon in mirror descent, a generalization of gradient descent. As mirror descent is non-Euclidean, existing explanations of the edge of stability phase do not extend to this setting. We generalize the existing notion of sharpness to introduce mirror sharpness, which measures the curvature of the loss with respect to the mirror potential. Empirically, we observe that the mirror sharpness hovers slightly above the $2/\eta$ threshold. Furthermore, we derive a dual central flow ODE for mirror descent that models iterates in continuous time at the onset of edge of stability. Our ODE provides insights into the difficulties of extending existing arguments for why sharpness stabilizes at $2/\eta$ from gradient descent to mirror descent in full g
    
[^45]: 一种利用迭代加权向量结合k-均值对形状数据进行聚类的新方法

    A Novel Approach to Clustering Shape Data Using Iteratively Weighted Vectors with k-Means

    [https://arxiv.org/abs/2611.28880](https://arxiv.org/abs/2611.28880)

    该论文提出在迭代k-means算法中加入逐元素权重更新步骤以提升形状数据的聚类性能，并将权重估计归结为最小二乘问题，几乎不增加额外计算成本。

    

    拓扑数据分析提供了一系列用于分析具有形状结构的数据（包括点云、图和图像）的方法。拓扑向量化领域的最新进展使得形状数据能够嵌入到欧几里得空间中，从而便于进行比较和机器学习任务。能够有效捕捉拓扑结构的向量摘要对于形状分类以及聚类等无监督学习任务尤为有用。我们研究了利用向量化拓扑数据通过k-means进行聚类的方法，该算法通过最小化平方误差损失函数将观测值划分为固定数量的组。通过在迭代k-means算法中引入逐元素的权重更新步骤，我们在模拟数据集和真实世界的形状数据集上都提升了聚类性能。我们证明了这些权重的估计可以归结为一个最小二乘问题，只需极少的额外计算量。

    arXiv:2611.28880v1 Announce Type: cross  Abstract: Topological data analysis provides a collection of methods for analyzing data with shape structure, including point clouds, graphs, and images. Recent advances in topological vectorization have enabled the embedding of shape data into Euclidean spaces, facilitating comparisons and machine learning tasks. Vector summaries that effectively capture topological structures are particularly useful for shape classification and unsupervised learning tasks such as clustering. We investigate the use of vectorized topological data for clustering through k-means, which partitions observations into a fixed number of groups by minimizing a squared-error loss function. By incorporating an element-wise weight-update step into the iterative k-means algorithm, we improve clustering performance on both simulated and real-world shape datasets. We show that estimating these weights reduces to a least-squares problem, requiring minimal additional computatio
    
[^46]: 学习一个区间需要多少次错误？区间及其并集的精确 Littlestone 维数

    How many mistakes does it take to learn an interval? Exact Littlestone dimensions of intervals and their unions

    [https://arxiv.org/abs/2611.28727](https://arxiv.org/abs/2611.28727)

    本文精确确定了离散区间与区间并集的 Littlestone 维数，证明强迫 d 次错误所需的最小域规模遵循一个三段式规律，并发现常见的“等待正样本后二分搜索端点”学习算法并非最优。

    

    假设类的 Littlestone 维数等于在线学习者在面对自适应对手时所能保证的最少犯错次数。对于 m 个点上的离散区间以及 j 个区间的并集，此前人们只知道其增长阶，而我们给出了精确值。对每个切换预算 k（当 k 为奇数时两端的标签被固定），能强迫 d 次错误的最小规模遵循一个三段式规律：当 d≤k 时为 d+1，随后为 2d−k+1，而在 d 超过 ⌈3k/2⌉+k−1 之后，规模每翻倍恰好增加 k 次错误。m 个点上 j 个区间的并集对应于 k=2j 且规模为 m+1，而单个区间给出阈值序列 1,2,3,5,7,10,14,20,…。等待正样本出现再对两个端点做二分搜索的自然学习算法并非最优：在 10^5 以下的所有域规模中，有 50,847 个规模可使其被迫多犯一次错误。上界来自间隙势函数与块配对学习器（原文摘要在此处截断）。

    arXiv:2611.28727v1 Announce Type: new  Abstract: The Littlestone dimension of a hypothesis class equals the smallest number of mistakes an online learner can guarantee against an adaptive adversary. For discrete intervals and unions of $j$ intervals on $m$ points, only its order of growth was known. We determine it exactly. For every switch budget $k$ (with both end labels fixed when $k$ is odd), the smallest size on which $d$ mistakes can be forced follows a three-regime law: $d+1$ while $d\le k$, then $2d-k+1$, and beyond $d=\lceil 3k/2\rceil+k-1$ doubling the size adds exactly $k$ mistakes. Unions of $j$ intervals on $m$ points correspond to $k=2j$ and size $m+1$, and a single interval gives the thresholds $1,2,3,5,7,10,14,20,\dots$. The natural learner that waits for a positive example and binary-searches both endpoints is not optimal: it can be forced into one extra mistake for 50,847 of the domain sizes below $10^5$. Upper bounds come from a gap potential and a block-pairing lear
    
[^47]: 高维混淆因素下基于森林的异质性处理效应稳健估计

    Robust Forest-based Estimation of Heterogeneous Treatment Effect Under High-dimensional Confounding

    [https://arxiv.org/abs/2611.28662](https://arxiv.org/abs/2611.28662)

    本文提出正交Huber随机森林（OHRF）框架，通过融合Huber稳健估计、Neyman正交性与森林自适应加权，在高维混淆、非线性异质性和重尾极端结果的阿尔茨海默病观察数据中稳健地估计异质性处理效应。

    

    理解阿尔茨海默病患者对治疗的不同反应对于个性化治疗评估十分重要。利用阿尔茨海默病神经影像学倡议（ADNI）的观察性数据，我们研究了多奈哌齐剂量与12个月认知表现之间的关联如何随诊断阶段和海马体体积的变化而变化。该分析带来了若干统计挑战，包括潜在的非线性处理效应异质性、高维混淆因素，以及含有极端观测值的重尾结果。为了应对这些挑战，我们开发了一个正交Huber随机森林框架，该方法结合了基于Huber的稳健估计、Neyman正交性和基于森林的自适应加权。所提出的方法能够处理复杂的处理效应异质性，同时降低对干扰参数估计误差和极端结果的敏感性。我们建立了其一致性（原文摘要在此处截断）。

    arXiv:2611.28662v1 Announce Type: cross  Abstract: Understanding how patients with Alzheimer's disease respond differently to treatment is important for individualized therapeutic assessment. Using observational data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), we investigate how the association between donepezil dosage and 12-month cognitive performance varies across diagnostic stages and hippocampal volumes. This analysis presents several statistical challenges, including potentially nonlinear treatment effect heterogeneity, high-dimensional confounding, and heavy-tailed outcomes with extreme observations. To address these challenges, we develop an Orthogonal Huber Random Forest (OHRF) framework that combines Huber-based robust estimation, Neyman orthogonality, and forest-based adaptive weighting. The proposed method accommodates complex treatment effect heterogeneity while reducing sensitivity to nuisance estimation errors and extreme outcomes. We establish its consi
    
[^48]: 无梯度Stein引导：超越粒子SMC的离散化一致扩散推理时间缩放

    Stein Guidance without Gradients: Discretization Uniform Diffusion Inference Time Scaling beyond Particle SMC

    [https://arxiv.org/abs/2611.28654](https://arxiv.org/abs/2611.28654)

    提出Stein引导方法，将无导数引导与全路径Metropolis-Hastings校正相结合，每步仅需两次查询即可在期望意义下近似梯度引导，并证明误差随MH更新次数超多项式衰减且常数不随引导深度增长，突破了粒子SMC需粒子数随引导深度线性增长的限制。

    

    奖励引导的扩散模型需要对由近似价值函数引入的偏差进行有效引导与校正。一个布朗运动示例表明，每一步都执行完全多项式重采样的序贯蒙特卡洛（SMC）方法可能需要粒子数量随引导深度线性增长才能保持精度。我们提出了Stein引导方法，它将无导数引导与全路径独立性Metropolis-Hastings校正相结合。通过从中心化高斯候选中进行价值加权选择，可在期望意义下近似梯度引导，每步仅需少至两次查询。中心化处理稳定了局部归一化因子的估计，且同样的查询还能产生用于校正的显式路径权重。在一致的价值与空间正则性界限下，我们证明了有限网格总变差误差随MH更新次数呈超多项式衰减，且相关常数不随引导深度增长。当价值函数趋近精确时，误差前因子趋于零。

    arXiv:2611.28654v1 Announce Type: new  Abstract: Reward-guided diffusion requires effective steering and correction of the bias introduced by approximate value functions. A Brownian example shows that SMC with full multinomial resampling at every step can require particle counts to grow linearly with guidance depth to maintain accuracy. We propose Stein Guidance, which combines derivative-free guidance with whole-path independence Metropolis-Hastings correction. Value-weighted selection from centered Gaussian candidates approximates gradient guidance in expectation, using as few as two queries per step. Centering stabilizes the local normalizer estimates, and the same queries yield an explicit path weight for correction. Under uniform value and spatial regularity bounds, we prove superpolynomial decay of finite-grid total variation error in the number of MH updates, with constants that do not grow with guidance depth. The error prefactor tends to zero as the value functions approach ex
    
[^49]: 自然策略梯度的近极小极大最优样本复杂度及匹配的信息论下界

    Nearly Minimax-Optimal Sample Complexity of Natural Policy Gradient with a Matching Information-Theoretic Lower Bound

    [https://arxiv.org/abs/2611.28628](https://arxiv.org/abs/2611.28628)

    本文证明了基于样本的自然策略梯度（NPG）在表格型有限时域MDP中达到接近极小极大最优的样本复杂度 $\widetilde O(SAH^4/\varepsilon^2)$，并给出了与之匹配的信息论下界，回答了直接策略优化能否达到统计极限这一基本问题。

    

    强化学习中的一个基本问题是：学习一个近似最优策略需要多少采样转移，以及直接策略优化方法能否达到这一统计极限。为回答这一问题，我们研究了自然策略梯度（NPG）这一被广泛使用的策略优化方法。我们考虑表格型有限时域马尔可夫决策过程（MDP），其奖励已知、转移核依赖时域长度且未知，并通过生成模型进行访问。我们提出的基于样本的NPG算法通过反向扫描更新策略，且不需要估计转移概率。在三种不同的批大小与步长调度下，我们建立了最后迭代的保证，样本复杂度为 $\widetilde O(SAH^4/\varepsilon^2)$，其中 $S$、$A$ 和 $H$ 分别表示状态数、动作数和时域长度。在此设定下，依赖时域的转移核贡献了一个 $H$ 的因子

    arXiv:2611.28628v1 Announce Type: new  Abstract: A fundamental question in reinforcement learning is how many sampled transitions are needed to learn a near-optimal policy, and whether direct policy optimization can attain this statistical limit. To answer this question, we consider natural policy gradient (NPG), a widely used policy optimization method. We study tabular finite-horizon Markov decision processes (MDPs) with known rewards and unknown horizon-dependent transition kernels, accessed through a generative model. Our sample-based NPG algorithms update the policy by backward sweeps and do not estimate transition probabilities. Under three different batch-size and step size schedules, we establish last-iterate guarantees with sample complexity $\widetilde O(SAH^4/\varepsilon^2)$, where $S$, $A$, and $H$ denote the number of states, the number of actions, and the horizon length, respectively. In this setting, horizon-dependent transition kernels contribute a factor of $H$ to our 
    
[^50]: 基于Yasue作用的神经薛定谔桥及其在生成建模中的应用

    Neural Schr\"{o}dinger Bridge via Yasue Action with Applications to Generative Modeling

    [https://arxiv.org/abs/2611.28623](https://arxiv.org/abs/2611.28623)

    该论文提出了一种基于Yasue作用泛函鞍点公式的神经求解器，通过学习常微分方程的最优速度（而非随机微分方程的漂移）来求解薛定谔桥问题，且仅需源和目标样本即可运行，因此特别适用于高维生成建模任务。

    

    我们提出了一种新颖的神经求解器，用于推断薛定谔桥问题的连续状态演化。该求解器建立在涉及Yasue作用泛函的鞍点公式化之上。这使得我们能够通过学习常微分方程的最优速度来实现连续状态演化，而无需学习随机微分方程的最优漂移。所提出的求解器仅使用源样本和目标样本，而不需要源分布和目标分布本身，这使其特别适合生成建模中出现的高维应用。我们通过在低维和高维空间中、使用真实和合成数据集的多个数值实验，证明了所提出求解器的有效性。

    arXiv:2611.28623v1 Announce Type: cross  Abstract: We propose a novel neural solver for inferring the continuous state evolution of the Schr\"{o}dinger bridge problem. The proposed solver builds on a saddle point formulation involving the Yasue action functional. This allows us to realize the continuous state evolution by learning the optimal velocity for an ordinary differential equation, as opposed to learning the optimal drift for a stochastic differential equation. The proposed solver only uses the source and the target samples, and not the source and target distributions per se, which makes it particularly suitable for high dimensional applications arising in generative modeling. We demonstrate the effectiveness of the proposed solver using several numerical experiments in low and high dimensions, using both real and synthetic datasets.
    
[^51]: 圆柱分布的弯曲指数族：m-投影、曲率与von Mises-Fisher分量的检验

    The Curved Exponential Family of Cylindrical Distributions: m-Projection, Curvature and a Test of the von Mises-Fisher Component

    [https://arxiv.org/abs/2611.28598](https://arxiv.org/abs/2611.28598)

    本文研究推广 Mardia-Sutton 模型的圆柱数据弯曲指数族，给出了极大似然估计存在唯一的充要条件及其作为 m-投影的显式构造，并证明该估计残差在 Fisher 度量下的平方长度恰为经典的 von Mises-Fisher 对 Fisher-Bingham 方向的得分检验。

    

    圆柱型数据由 R^d 中的一个向量与 S^(m-1) 上的一个方向配对组成。我们研究了一族分布，其中方向服从 von Mises-Fisher 分布，而在给定方向的条件下，向量服从高斯分布且其均值是方向的仿射函数；这推广了 Mardia-Sutton 模型。该分布族是具有 Fisher-Bingham 方向的正规分布族内部的一个弯曲指数族。我们给出了极大似然估计量存在且唯一的充分必要条件。该估计量除需求解一个标量方程的根之外均为显式表达式，并且它是矩匹配成员的 m-投影（即 Kullback-Leibler 投影）。该估计量未能匹配的每个矩都可以表示为单一残差的函数——即方向二阶矩的样本值与拟合值之间的差距。该差距在 Fisher 度量下的平方长度正是经典的 von Mises-Fisher 对 Fisher-Bingham 方向的得分检验，该检验源于 Cox 在圆上的工作，并由 Mardia、Holmes 和 Kent 加以推广。

    arXiv:2611.28598v1 Announce Type: cross  Abstract: Cylindrical data pair a vector in R^d with a direction in S^(m-1). We study a family in which the direction is von Mises-Fisher and, given the direction, the vector is Gaussian with mean affine in it; this generalizes the Mardia-Sutton model. The family is a curved exponential family inside a regular family with Fisher-Bingham directions. We state necessary and sufficient conditions for the maximum-likelihood estimator to exist and be unique. The estimator is explicit except for one scalar root and is the m-projection (Kullback-Leibler projection) of the moment-matching member. Every moment it does not match is a function of one residual, the gap between the sample and fitted second moments of the direction. The squared length of this gap in the Fisher metric is the classical score test of von Mises-Fisher against Fisher-Bingham directions, due to Cox on the circle and extended by Mardia, Holmes and Kent. Specific here are this reading
    
[^52]: 对抗注入下序列预测的错误-弃权紧密权衡

    Tight Mistake-Abstention Tradeoffs for Sequential Prediction under Adversarial Injections

    [https://arxiv.org/abs/2611.28579](https://arxiv.org/abs/2611.28579)

    本文提出“加权打散势”及其变体，为对抗注入下的序列预测问题给出了弥合上下界差距的算法，在不同设定下基本紧致地刻画了错误-弃权权衡，并在分布已知时达到了此前猜想的 O(d log T) 错误界。

    

    我们研究了由 Goel 等人（2023）提出的序列预测问题：对手可以在来自干净分布 μ 的独立同分布数据流中注入任意多的损坏实例，而学习器在当前实例确实被注入时可以无偿地选择弃权、不作预测。当 μ 未知时，Yu 和 Blanchard（2026）证明了在有限 VC 维的类别和遗忘型（oblivious）对手下，错误数与错误弃权之间存在多项式式权衡，但其权衡格局中上下界之间仍存在显著差距。在本文中，我们提出了加权打散势，并基于该势的若干变体，给出了弥合差距的算法，从本质上刻画了多种感兴趣设定下的权衡关系：当 μ 已知时，我们的算法改进了 Goel 等人（2023）的结果，并达到了他们所猜想 O(d log T) 的错误界；当 μ 未……（原文摘要到此截断）

    arXiv:2611.28579v1 Announce Type: new  Abstract: We study the sequential prediction problem introduced by Goel et al. (2023), where an adversary can inject arbitrarily many corrupted instances in an i.i.d. stream from a clean distribution $\mu$, while the learner is allowed to abstain from making a prediction at no cost if the current instance was indeed injected. When $\mu$ is unknown, Yu and Blanchard (2026) show a polynomial tradeoff between mistakes and false abstentions for classes with finite VC dimension and oblivious adversaries, but still leave open a significant gap between upper and lower bounds in their tradeoff landscape. In this paper, we propose the weighted shattering potential, and based on variants of this potential, provide gap-closing algorithms that essentially characterize the tradeoffs under various settings of interest: When $\mu$ is known, our algorithm improves over Goel et al. (2023) and achieves their conjectured $O(d\log T)$ mistake bound; When $\mu$ is unk
    
[^53]: 复数带限最大似然：估计探测到的量子背后的波

    Complex band-limited maximum likelihood: estimating the wave behind detected quanta

    [https://arxiv.org/abs/2611.28572](https://arxiv.org/abs/2611.28572)

    该论文提出复数带限最大似然方法（complex BLML），通过在物理允许的谱约束下最大化探测位置的似然，直接从粒子探测位置估计包含相位的完整波函数 $\psi$，且仅需一个特征值即可认证全局最优。

    

    探测到的光子和电子落在由 $|\psi|^2$ 决定的位置上，然而波动方程所传播的是包含相位的 $\psi$，而产生这些粒子的量子态往往只有部分是已知的。密度估计器可以直接从探测位置恢复 $|\psi|^2$，无需借助任何波动方程，但它会丢弃相位，从而丢失量子态本身。我们提出复数带限最大似然，通过在物理所允许的强度范围内最大化探测位置的似然来直接估计 $\psi$ 本身：即谱位于给定集合（例如某个频带、孔径的形状，或部分相干源的若干模式）中的复函数的模平方。仅需一个特征值即可认证全局最大值。在模拟的双缝远场实验中，我们量化了每种先验信息的价值：频带先验在达到给定精度时所需的光子数比核密度估计少七倍，并且已经……（摘要原文在此处截断）

    arXiv:2611.28572v1 Announce Type: cross  Abstract: Detected photons and electrons land at positions drawn from $|\psi|^2$, yet it is $\psi$, phase included, that a wave equation propagates, and the state that produced the particles is often known only in part. A density estimator can recover $|\psi|^2$ directly from the positions, without any wave equation, but it discards the phase, and with it the state. We introduce complex band-limited maximum likelihood (complex BLML), which estimates $\psi$ itself by maximising the likelihood of the detected positions over exactly the intensities the physics allows: squared moduli of complex functions whose spectrum lies in a given set, such as a band, the shape of the aperture, or several modes for a partially coherent source. A single eigenvalue can certify a global maximum. On simulated two-slit far fields we measure what each prior is worth: a band reaches a given accuracy with seven times fewer photons than a kernel density estimate and alre
    
[^54]: 何时与何处迁移：为短历史多变量时间序列预测学习可迁移的历史上下文

    When and Where to Transfer: Learning Transferable Historical Contexts for Short-History Multivariate Time Series Forecasting

    [https://arxiv.org/abs/2611.28567](https://arxiv.org/abs/2611.28567)

    提出分层上下文迁移框架HiCT，通过共享上下文编码器和交叉注意力自适应地加权数据源群体与历史上下文，确定迁移的时机与来源，以改进短历史多变量时间序列预测，并为其提供了理论上的目标超额风险界。

    

    arXiv:2611.28567v1 公告类型：cross 摘要：当目标历史数据较短时，相关数据源可以改进多变量时间序列预测，但其有用性因数据源群体和历史时期的不同而有所差异。一个关键挑战是从数据源中识别出有信息量的历史上下文。本文提出了分层上下文迁移，这是一个自适应地为数据源群体及其历史上下文分配权重的框架，从而确定在“何处”和“何时”进行迁移。该框架使用共享上下文编码器和交叉注意力机制来学习数据源群体和历史上下文的权重。这些权重引导正则化的迁移估计，随后进行可选的针对目标的校正。本文为上下文加权的迁移估计器建立了目标超额风险界。实验表明，该方法在四个多变量预测数据集上展现出有竞争力的预测性能，并且相对于所评估的基线方法具有更优的精度-运行时间权衡。

    arXiv:2611.28567v1 Announce Type: cross  Abstract: Related sources can improve multivariate time series forecasting when the target history is short, but their usefulness varies across source groups and historical periods. A key challenge is to identify informative historical contexts from the sources. This paper proposes Hierarchical Context Transfer (HiCT), a framework that adaptively weights source groups and their historical contexts to determine both \emph{where} and \emph{when} to transfer. A shared context encoder and cross-attention are used to learn weights over source groups and historical contexts. These weights guide regularized transfer estimation, followed by an optional target-specific correction. This paper establishes a target excess-risk bound for the context-weighted transfer estimator. Experiments demonstrate competitive predictive performance across four multivariate forecasting datasets and a favorable accuracy--runtime trade-off against the evaluated baselines. S
    
[^55]: 一种基于图的混合机器学习模型，用于预测供应链物流中的交付时间

    A hybrid Graph-based Machine Learning Model for predicting Delivery Times in Supply Chain Logistics

    [https://arxiv.org/abs/2611.28467](https://arxiv.org/abs/2611.28467)

    该论文提出了一种融合图论、机器学习与马尔可夫链随机方法的混合建模框架，利用历史与实时异构数据（交通、天气、供应商备货状态等）预测供应链中供应商与制造商之间入站物流的交付延误。

    

    全球供应链日益容易受到外部事件（如自然灾害和地缘政治危机）以及内部运营低效（包括供应商延迟和订单发货瓶颈）所引发的干扰。本工作在欧盟“地平线欧洲”M4ESTRO项目框架内开展，提出了一种结合图论与机器学习的混合建模方法来表示供应链物流运营，重点预测工业供应商与制造实体之间入站物流的延误。订单履行的各个阶段通过一个形式化的数学框架进行建模，通过定义基于历史数据和实时数据（来源于交通状况、天气条件以及供应商发货准备情况等异构数据源）计算得出的干扰指标。在基于图的供应链表示之上，采用基于马尔可夫链的随机方法来预测延误。

    arXiv:2611.28467v1 Announce Type: cross  Abstract: Global supply chains are increasingly vulnerable to disruptions arising from external events, such as natural disasters and geopolitical crises, and from internal operational inefficiencies, including supplier delays and bottlenecks in order shipping. This work, conducted within the Horizon Europe project M4ESTRO, introduces a hybrid modelling approach combining graph theory and machine learning to represent supply chain logistics operations, with a focus on predicting delays in inbound flows between industrial suppliers and manufacturing entities. The various stages of order fulfilment are modelled through a formal mathematical framework, by defining disruption indicators computed from both historical and real-time data drawn from heterogeneous sources, including traffic and weather conditions and supplier dispatch readiness. Building on a graph-based representation of the supply chain, stochastic methods based on Markov chains are ap
    
[^56]: 扩散模型中引导诱导的分数估计误差激活

    Guidance-Induced Activation of Score Estimation Errors in Diffusion Models

    [https://arxiv.org/abs/2611.28451](https://arxiv.org/abs/2611.28451)

    该论文揭示了CFG中的“引导诱导误差激活”机制：通过加权Helmholtz-Hodge分解，识别并量化了原本对参考密度无瞬时影响的螺线型score误差分量如何因引导密度与参考密度的差异而影响引导采样的边缘分布演化。

    

    无分类器引导（Classifier-free guidance, CFG）通过组合条件与无条件分数估计来引导扩散采样。每个分数误差的加权螺线分量对其相应参考密度的演化没有瞬时影响。然而，由于CFG中的引导密度通常与两个参考密度均不相同，这些螺线分量的组合可能会影响引导边缘分布的演化。我们将这一机制称为“引导诱导的误差激活”。利用加权Helmholtz-Hodge分解，我们识别了被激活的误差场，量化了其大小，并刻画了其消失的条件。在固定噪声时间下，激活的误差场随引导强度连续变化，其弱一阶变化包含两项：一项来自误差的重新加权，另一项来自引导密度的变化。对于有限的分数误差，ODE特征线方法与SDE的Feynman-Kac公式将误差激活…（摘要截断）

    arXiv:2611.28451v1 Announce Type: cross  Abstract: Classifier-free guidance (CFG) combines conditional and unconditional score estimates to steer diffusion sampling. The weighted-solenoidal component of each score error has no instantaneous effect on the evolution of its corresponding reference density. Because the guided density in CFG generally differs from both reference densities, the combination of these solenoidal components can affect the evolution of the guided marginals. We call this mechanism guidance-induced error activation. Using a weighted Helmholtz-Hodge decomposition, we identify the activated error field, quantify its magnitude, and characterize when it vanishes. At fixed noise time, the activated error field varies continuously with guidance strength. Its weak first-order variation has two terms, one from reweighting the errors and one from the change in the guided density. For finite score errors, ODE characteristics and an SDE Feynman-Kac formula connect error-activ
    
[^57]: 外部对照试验及更广泛场景的靶向学习

    Targeted Learning for Externally Controlled Trials and Beyond

    [https://arxiv.org/abs/2611.28421](https://arxiv.org/abs/2611.28421)

    本文针对外部对照试验提出了基于平衡权重的靶向学习框架，给出有效影响函数与靶向最大似然估计量（TMLE）及其实用算法，并将该框架扩展到辅以外部协变量的随机对照试验等更多场景。

    

    本文聚焦于外部对照试验（ECT），即由单臂临床试验辅以外部对照臂（ECA）构成的试验设计。外部对照臂的潜在数据来源可大致分为三类：(i) 历史临床试验的对照组，(ii) 同期队列研究，以及 (iii) 外部真实世界数据（RWD）。本文考虑了一类通过平衡权重定义的广泛的估计对象（estimands）。针对该类中给定的估计对象，我们讨论了有效影响函数（EIF）与靶向最大似然估计量（TMLE），并给出了实用的实现算法。在靶向学习框架内，TMLE 是一种替代估计量，具有多项理想的理论性质和良好的有限样本表现。最后，所提出的框架被扩展到另外两种场景：(1) 辅以外部协变量信息的随机对照试验（RCT）……

    arXiv:2611.28421v1 Announce Type: cross  Abstract: This paper focuses on externally controlled trials (ECTs), which consist of a single-arm clinical trial supplemented by an external control arm (ECA). Potential data sources for the ECA can be broadly classified into three categories: (i) the control arm of a historical clinical trial, (ii) a concurrent cohort study, and (iii) external real-world data (RWD). In this paper, we consider a broad class of estimands defined through balancing weights. For a given estimand in the class, we discuss the efficient influence function (EIF) and the targeted maximum likelihood estimator (TMLE), together with practical implementation algorithms. Within the targeted learning framework, TMLE is a substitution estimator that enjoys several desirable theoretical properties and favorable finite-sample performance. Finally, the proposed framework is extended to two additional settings: (1) randomized controlled trials (RCTs) augmented with external covari
    
[^58]: 基于样本扩增的数据相关概率估计

    Data-Dependent Probability Estimation via Sample Amplification

    [https://arxiv.org/abs/2611.28399](https://arxiv.org/abs/2611.28399)

    该论文将样本扩增与基于在线优化的分布学习相结合，提出了一种多项式时间估计器，证明在选择规则已知的条件下仅需 Õ(√d) 个样本即可对数据相关事件实现常数均方误差估计，从而解决了 Feldman 等人提出的开放问题。

    

    arXiv:2611.28399v1 公告类型：新论文 摘要：我们能否在看到数据之后再选择一个事件并估计其概率，而无需使用任何新的样本？经验概率可能会产生极大的误导，而在大小为 d 的域上学习整个分布以达到常数精度需要 Θ(d) 个样本。我们证明，在选择事件的规则已知的条件下，Õ(√d) 个样本就足以达到常数均方误差。更一般地，我们的估计器能够处理任意取值于 [0,1] 的查询，并在可以对选择规则进行求值的情况下以多项式时间运行。这一样本复杂度在相差对数因子的意义上与已知下界相匹配，并解决了 Feldman、Kale、Raman、Talwar 和 Tewari（arXiv:2610.41244）提出的开放问题。我们的估计框架依赖于样本扩增（arXiv:1904.12053 和 arXiv:2201.04315）与基于在线优化的分布学习的一种新组合。我们的估计器达到了均方误差 O(d^{3/4……

    arXiv:2611.28399v1 Announce Type: new  Abstract: Can we estimate the probability of an event chosen after seeing the data, without using any fresh samples? Its empirical probability can be arbitrarily misleading, while learning the entire distribution to constant accuracy requires $\Theta(d)$ samples on a domain of size $d$. We show that $\widetilde{O}(\sqrt{d})$ samples suffice for constant mean squared error, provided the rule choosing the event is known. More generally, our estimator handles arbitrary $[0,1]$-valued queries and runs in polynomial time given evaluation access to the rule. This sample complexity matches the known lower bound up to logarithmic factors and resolves the open question posed by Feldman, Kale, Raman, Talwar, and Tewari (arXiv:2610.41244).   Our estimation framework relies on a new combination of sample amplification (arXiv:1904.12053 and arXiv:2201.04315) with distribution learning via online optimization. Our estimator achieves mean squared error $O(d^{3/4
    
[^59]: 超越高斯校准：去中心化SGD中的局部推断

    Beyond Gaussian Calibration: Local Inference in Decentralized SGD

    [https://arxiv.org/abs/2611.28370](https://arxiv.org/abs/2611.28370)

    该论文揭示去中心化SGD中客户端局部端点虽整体高斯覆盖率接近标称但双尾严重失准，并给出一个可由图与步长计算的精确噪声滤波判据，证明在特定网络上极限分布是非高斯局部场与独立高斯共识项之和。

    

    去中心化随机梯度下降通过局部梯度步与邻居平均来最小化一个共同目标。其统计理论大多关注网络平均值，然而客户端实际上是基于自身的迭代值采取行动的，且诸如告警之类的单侧决策仅使用局部误差的单侧尾部。我们证明：客户端在恒定步长下的端点虽然整体覆盖率可接近标称高斯水平，但其两个尾部却严重失准。一个精确的噪声滤波表示解释了其中原因：一个可由图结构与步长计算出的量决定了端点是否渐近服从高斯分布；当客户端数量超过步长的倒数时，在具有可和返回概率的网络（例如维度至少为三的惰性环面以及扩张图）上，邻近客户端的少数近期噪声项会保留局部方差的一个固定份额。在这些环面上，极限分布是一个非高斯局部场加上一个独立的高斯共识项，并且没有任何基于有限……构建的参考分布（摘要原文在此处截断）。

    arXiv:2611.28370v1 Announce Type: cross  Abstract: Decentralized SGD minimizes a common objective by local gradient steps and neighbor averaging. Its statistical theory mostly concerns network averages, yet clients act on their own iterates, and one-sided decisions such as alarms use one tail of the local error. We show that a client's constant-step endpoint can have nearly nominal Gaussian coverage while both tails are badly miscalibrated. An exact noise-filter representation explains why: a quantity computable from the graph and stepsize decides whether the endpoint is asymptotically Gaussian, and when the number of clients outgrows the inverse stepsize on networks with summable return probabilities, such as lazy tori of dimension at least three and expanders, a few recent noise terms at nearby clients keep a fixed share of local variance. On these tori the limit is a non-Gaussian local field plus an independent Gaussian consensus term, and no reference distribution built from finite
    
[^60]: 依赖条件下的分位数SGD中的方向性偏差与得分反转推断

    Directional Bias in Quantile SGD and Score-Inversion Inference under Dependence

    [https://arxiv.org/abs/2611.28367](https://arxiv.org/abs/2611.28367)

    该论文揭示了数据的序列依赖性会在分位数SGD中产生一种得分自协方差无法检测的方向性偏差（反转数据流可使主导偏差变化三倍），并将其分解为滞后特定交互作用项与曲率项，同时证明延迟阈值可逐滞后消除交互作用而不损失观测信息与主导方差。

    

    分位数随机梯度下降（SGD）通过将每个观测值与一个自适应阈值进行比较来追踪分位数，与之密切相关的更新方法被用于校准在线共形集合并驱动分位数时序差分学习。这些方法的输入存在序列依赖性，然而现有的理论保证大多假设数据相互独立。我们证明依赖性会产生一种得分自协方差无法检测到的方向性偏差：将平稳数据流反转后，所有得分自协方差保持不变，但主导偏差却可能变化三倍。在有限记忆和递减步长的条件下，该偏差可分解为滞后特定的阈值与当前观测值之间的交互作用项以及一个曲率项；通过延迟阈值，可以逐个滞后地消除这些交互作用，同时保留全部观测值和主导方差。在步长接近 $t^{-1/2}$ 时（在线推断中常用的设置），该偏差与标准误相当，并会使基于SGD的区间（摘要在此处被截断）……

    arXiv:2611.28367v1 Announce Type: cross  Abstract: Quantile stochastic gradient descent (SGD) tracks a quantile by comparing each observation with an adaptive threshold, and closely related updates calibrate online conformal sets and drive quantile temporal-difference learning. Their inputs are serially dependent, yet existing guarantees largely assume independence. We show that dependence creates a directional bias that score autocovariances cannot detect: reversing a stationary stream leaves every score autocovariance unchanged but can change the leading bias threefold. For finite memory and decreasing stepsizes, the bias splits into lag-specific interactions between the threshold and the current observation and a curvature term; delaying the threshold removes the interactions one lag at a time while keeping every observation and the leading variance. At stepsizes near $t^{-1/2}$, as used for online inference, the bias is comparable to the standard error and makes an SGD-based interv
    
[^61]: 哪些贷款，而非多少笔：已部署风险模型的精确水平不变、随时有效的监控

    Which Loans, Not How Many: Exactly Level-Invariant, Anytime-Valid Monitoring of Deployed Risk Models

    [https://arxiv.org/abs/2611.28342](https://arxiv.org/abs/2611.28342)

    本文提出一种精确水平不变且随时有效（anytime-valid）的已部署风险模型校准监控方法，通过以月度计数条件化消除截距影响，从而将贷款数量的变化（如再融资浪潮）与评分-结果关系本身的变化区分开来。

    

    再融资浪潮可以在不改变评分与提前还款之间关系的前提下，改变抵押贷款提前还款的数量。在逻辑斯蒂模型中，以每月计数为条件，可以将截距从“哪些贷款提前还款”的似然中消除。我们的固定起点斜率乘积用于检验校准斜率的下界；在阈值为20时，其在原假设下曾经越界的概率至多为5%。重启的水平和方法用于跟踪对截距容差的偏离。置信序列用于估计恒定的斜率，而重启-交集规则则检测该斜率发生的变化。连续混合分布避免了离散先验带来的网格外宽度下限问题。在Freddie Mac贷款数据上，固定斜率重采样在开发集中重现了维持模型月度AUC方差的78%，在留出集中重现了70%（95%区间分别为57–100%和48–100%）。从全新的留出集起点出发，重启的水平和方法在两条路径上均发生越界，而固定起点的斜率乘积则保持……

    arXiv:2611.28342v1 Announce Type: new  Abstract: A refinancing wave can change the number of mortgage prepayments without changing the relationship between scores and prepayment. In a logistic model, conditioning on the monthly count removes the intercept from the likelihood of which loans prepay. Our fixed-start slope product tests a lower bound on the calibration slope. At threshold 20 its probability of ever crossing under the null is at most 5%. The restarted level sum tracks departures from intercept tolerance. A confidence sequence estimates a constant slope, and the restart-intersection rule detects changes in that slope. A continuous mixture avoids the off-grid width floor of a discrete prior. On Freddie Mac loans, fixed-slope resampling reproduces 78% of maintained-model monthly AUC variance in development and 70% in the holdout (95% intervals 57-100% and 48-100%). From fresh holdout starts, the restarted level sum crosses on both paths, and the fixed-start slope product stays
    
[^62]: 表格基础模型中的变量重要性恢复：一项模拟研究

    Variable Importance Recovery in Tabular Foundation Models: A Simulation Study

    [https://arxiv.org/abs/2611.28241](https://arxiv.org/abs/2611.28241)

    该论文通过蒙特卡洛模拟研究发现，表格基础模型在恢复总体层面变量重要性方面与成熟的机器学习基线方法相当，且往往更为准确。

    

    在机器学习中，识别哪些变量贡献了预测信息可能与预测准确性同等重要。表格基础模型在许多中小型表格数据集上的预测性能已经达到或超过经过调优的梯度提升树。然而，它们在恢复总体层面变量重要性方面的能力却鲜受关注。我们设计并开展了一项蒙特卡洛模拟研究，以比较表格基础模型与成熟机器学习基线方法在变量重要性恢复方面的表现。我们的框架区分了多种总体估计目标，将每一个估计目标与相应的模型无关的重要性估计器配对，并从排序、支撑集和相对量级三个方面评估恢复效果。我们的结果表明，表格基础模型在恢复总体变量重要性方面总体上与成熟的机器学习基线方法具有竞争力，且往往更为准确。

    arXiv:2611.28241v1 Announce Type: cross  Abstract: In machine learning, identifying which variables contribute predictive information can be as important as predictive accuracy. Tabular foundation models have matched or exceeded the predictive performance of tuned gradient boosting trees on many small- and medium-sized tabular datasets. Yet their ability to recover population-level variable importance has received little attention. We design and conduct a Monte Carlo simulation study to compare variable importance recovery across tabular foundation models and established machine learning baselines. Our framework distinguishes multiple population estimands, pairs each one with an aligned model-agnostic importance estimator, and evaluates recovery in terms of ranking, support, and relative magnitude. Our results show that tabular foundation models are generally competitive with established machine-learning baselines in recovering population variable importance, and often more accurate. W
    
[^63]: 学习数据支撑集上的投影：避免记忆化的分布学习

    Learning the Projection onto the Data Support: Distribution Learning that Avoids Memorization

    [https://arxiv.org/abs/2611.28234](https://arxiv.org/abs/2611.28234)

    提出了一种通过学习数据支撑集上最近点投影来实现流形上分布学习的新框架，该框架能自动适应内在维度并支持多维度流形并集，且其优化目标从原理上排除了通过记忆训练样本实现的解，从而避免了扩散模型等生成模型中的记忆化问题。

    

    我们提出了一种学习流形上支撑分布的框架，该框架基于估计到数据支撑集的最近点投影。具体而言，我们在一类函数上定义了一个目标泛函，其最小化点可以恢复该投影。所提出的框架能够自动适应数据的内在维度，并支持数据分布在具有不同维度的多个流形的并集上。所得到的目标函数可以使用标准神经网络架构进行优化。我们给出了总体层面的理论结果，证明了流形投影的恢复性，并证明了有限样本下的 Wasserstein 泛化界。此外，我们在一系列设置中对该方法进行了实证验证。与扩散模型等广泛使用的生成模型不同——这些模型的目标可以通过记忆训练样本来最小化——我们的公式化排除了这种记忆化行为。我们还引入了一种直接的记忆化度量来验证这一点。

    arXiv:2611.28234v1 Announce Type: new  Abstract: We introduce a framework for learning distributions supported on manifolds based on estimating the nearest-point projection onto the data support. Specifically, we define an objective functional over a class of functions whose minimizer recovers this projection. The proposed framework automatically adapts to the intrinsic dimension and allows for data supported on unions of manifolds with varying dimensions. The resulting objective can be optimized using standard neural network architectures. We present population-level results establishing recovery of the manifold projection and prove finite-sample Wasserstein generalization bounds. Furthermore, we validate the approach empirically across a range of settings. In contrast to widely used generative models such as diffusion models, whose objectives can be minimized by memorizing training samples, our formulation excludes such behavior. We also introduce a direct memorization metric to veri
    
[^64]: 混沌神经常微分方程的Rollout调度：覆盖增益、计算成本与预测精度

    Rollout Scheduling for Chaotic Neural ODEs: Coverage Gains, Computational Cost, and Predictive Accuracy

    [https://arxiv.org/abs/2611.28192](https://arxiv.org/abs/2611.28192)

    本文提出了一种面向混沌系统神经ODE训练的rollout长度自适应调度方法，在不稳定性预算约束下最大化轨迹数据覆盖并给出覆盖增益的理论界，实验表明其在相同计算成本下优于固定调度并提升预测精度。

    

    arXiv:2611.28192v1 公告类型：交叉。在混沌系统上训练神经常微分方程存在困难，因为模型参数的微小变化会导致预测差异迅速增长。Rollout（推演轨迹）是指模型所预测出的轨迹，若仅依据单一的全局Lyapunov时间来选择其长度，就会忽略局部不稳定性的差异。本文研究训练中的rollout调度问题，即对每个采样状态使用一段连续的rollout进行训练，探讨根据局部不稳定性自适应调整其长度能否改进预测。作者找出了在不稳定性预算（即对任何单个rollout可累积不稳定性总量的限制）约束下能使用最多轨迹数据的调度方案，并在给定假设下，对该方案带来的覆盖增益以及不稳定性估计存在误差时所损失的覆盖给出了理论界定。数值验证阐明了该理论。在Lorenz-63、Lorenz-96和Kuramoto--Sivashinsky系统上的训练实验在相同计算成本下比较了自适应调度与固定调度（原文摘要在此处被截断）。

    arXiv:2611.28192v1 Announce Type: cross  Abstract: Training Neural Ordinary Differential Equations (Neural ODEs) on chaotic systems is difficult because small changes in model parameters can lead to rapidly growing differences in predictions. A rollout is a trajectory predicted by the model. Choosing its length from a single global Lyapunov time overlooks differences in local instability. We study rollout scheduling for training on one contiguous rollout per sampled state, asking whether adapting its length to local instability improves prediction. We find the schedule that uses the most trajectory data under an instability budget, a limit on how much instability any one rollout may accumulate. Under stated assumptions, we bound the resulting coverage gain and the coverage lost when the instability estimates are in error. Numerical checks illustrate the theory. Training experiments on Lorenz-63, Lorenz-96, and Kuramoto--Sivashinsky compare adaptive and constant schedules at the same in
    
[^65]: 最小二乘数据选择中标签的价值

    The Value of Labels in Data Selection for Least Squares

    [https://arxiv.org/abs/2611.28169](https://arxiv.org/abs/2611.28169)

    该论文量化了最小二乘数据选择中“看到标签”的价值：仅用特征的最优选择者最坏情况超额损失率为 $d\kappa/n$，而利用标签可将主导常数改善至 $d\kappa/2$，即在低维（$d\le4$）情形下标签恰好带来两倍的收益。

    

    假设一个选择者面对一个有限的带标签数据集，并从中选取 $n$ 行（允许重复）。随后对被选中的行拟合普通最小二乘（不加权），并以最优损失为单位衡量该拟合在整个数据集上的超额损失。在没有进一步假设的情况下，该比率不会随 $n$ 衰减。我们研究了具有 $d$ 个特征、且各行归一化杠杆值至多为 $\kappa$ 的情形，并确定或界定了其主导常数。必须在看到标签之前就确定其选择规则的选择者，无论怎样利用特征信息，其最优最坏情况比率在一阶意义下均为 $d\kappa/n$。而对于标量标签，能够看到标签的选择者可以取得介于 $d\kappa/(2n)$ 与 $d\kappa\Lambda_d/n$ 之间的值，其中 $\Lambda_d<1$ 是均值选择的舍入常数。当 $d\le4$ 时，该常数为 $d\kappa/2$，因此在这种情形下标签恰好价值两倍；而对于单一特征的情形……（原文截断）

    arXiv:2611.28169v1 Announce Type: cross  Abstract: Suppose a selector sees a finite labelled dataset and chooses $n$ of its rows, with repetition. Ordinary least squares is then fitted to the rows that were chosen, without weights, and the excess loss of the fit on the full dataset is measured in units of the optimal loss. Without further assumptions this ratio does not decay with $n$. We examine the problem for $d$ features and rows of normalized leverage at most $\kappa$, and determine or bound the leading constants. A selector that must establish its rule before seeing the labels has an optimal worst-case ratio of $d\kappa/n$ to leading order, whatever use it makes of the features. With scalar labels, a selector that sees the labels achieves a value between $d\kappa/(2n)$ and $d\kappa\Lambda_d/n$, where $\Lambda_d<1$ is the rounding constant of mean selection. For $d\le4$, the constant is then $d\kappa/2$, so that labels are worth exactly a factor of two there, and for one feature a
    
[^66]: 基于病态随机权重神经网络筛的统计推断

    Statistical Inference with Ill-Conditioned Random Weight Neural Network Sieves

    [https://arxiv.org/abs/2611.26410](https://arxiv.org/abs/2611.26410)

    本文证明了随机权重神经网络筛在维数增大时必然出现病态性（最小特征值消失），并在显式处理该病态性的基础上建立了相应的逼近速率、估计与统计推断理论。

    

    随机权重神经网络（RWNN）是一类浅层网络，其隐藏层参数为随机生成并保持固定，仅对输出系数进行估计。在给定采样参数的条件下，RWNN回归可视为一个具有随机逼近空间的线性筛问题。我们证明，随着维数的增长，这类随机筛无法始终保持一致的良好条件性。对于满足多项式覆盖条件与Lipschitz条件的参数化特征族，我们证明了总体协方差矩阵的特征值具有由参数空间覆盖几何结构决定的多项式上界，从而导致最小特征值趋于消失。我们发展了显式处理这种病态性的逼近、估计与推断理论。在β-Hölder源条件下，我们同时获得了快于蒙特卡洛速率的L²逼近速率，以及基于……（原文摘要在此处截断）的新L^∞逼近界。

    arXiv:2611.26410v1 Announce Type: cross  Abstract: Random weight neural networks (RWNNs) are shallow networks whose hidden-layer parameters are randomly generated and then kept fixed, while only the output coefficients are estimated. Conditional on the sampled parameters, RWNN regression is a linear sieve problem with random approximation space. We show that such random sieves cannot remain uniformly well-conditioned as their dimension grows. For parametrized feature families satisfying polynomial covering and Lipschitz conditions, the eigenvalues of the population covariance matrix are proven to have polynomial upper bounds determined by the covering geometry of the parameter space, leading to a vanishing smallest eigenvalue. We develop approximation, estimation, and inference theory that explicitly treats the ill-conditioning. Under a $\beta$-H\"older source condition, we obtain both faster-than-Monte-Carlo $L^2$ approximation rates and new $L^\infty$ approximation bounds based on st
    
[^67]: 扩散模型是稀疏恢复的可证明高效先验

    Diffusion Models are Provably Efficient Priors for Sparse Recovery

    [https://arxiv.org/abs/2611.24602](https://arxiv.org/abs/2611.24602)

    本文针对压缩感知中的稀疏恢复问题，首次证明基于预训练扩散模型先验的似然引导退火朗之万采样器能够在多项式时间内以近乎最优的样本复杂度可证明地恢复稀疏信号，为扩散模型作为逆问题先验提供了高效的端到端理论保证。

    

    扩散模型凭借其捕捉复杂数据分布并从中生成样本的能力，彻底改变了生成建模领域。这一成功激发了人们浓厚的兴趣，即将预训练的扩散模型用作成像与感知中高度病态逆问题的表达性先验。尽管经验性能出色，但这种方法的理论理解仍然有限：现有保证往往是渐近性的，或在最坏情况下需要难以承受的计算成本。在本文中，我们通过研究从少量线性测量中进行稀疏恢复这一压缩感知中的经典问题来填补这一空白。我们证明，给定一个准确刻画促进稀疏性的伯努利-高斯先验的扩散模型，一个由似然引导的退火朗之万采样器能够在多项式时间内、以近乎最优的样本复杂度，可证明地恢复稀疏信号，且适用于广泛的随机和确定性传感矩阵族。

    arXiv:2611.24602v1 Announce Type: cross  Abstract: Diffusion models have transformed generative modeling through their ability to capture and generate samples from complex data distributions. This success has spurred substantial interest in using pretrained diffusion models as expressive priors for highly ill-posed inverse problems in imaging and sensing. Despite strong empirical performance, theoretical understanding of this approach remains limited: existing guarantees are often asymptotic or entail prohibitive computational costs in the worst case. In this paper, we address this gap by studying sparse recovery from a small number of linear measurements, a classical problem in compressed sensing. We show that, given an accurate diffusion model capturing a sparsity-promoting Bernoulli-Gaussian prior, a likelihood-guided annealed Langevin sampler provably recovers the sparse signal in polynomial time with near-optimal sample complexity for a broad family of random and deterministic sen
    
[^68]: 连续分布漂移下交叉验证的折加权方法

    Fold Weighting for Cross-Validation Under Continuous Distributional Drift

    [https://arxiv.org/abs/2611.23184](https://arxiv.org/abs/2611.23184)

    该论文提出在连续分布漂移下对交叉验证的各折进行加权，证明了对权重最小化风险界是一个凸优化问题，且将权重向近期折偏移可以有效降低由漂移导致的部署风险，同时需权衡验证噪声的增加。

    

    交叉验证对所有折赋予相同权重，因此在分布漂移存在时，最旧的折与最新的折具有同等的重要性。我们通过全变差距离下的赫尔德（Hölder）条件对漂移进行建模，并给出了滑动窗口交叉验证所选配置在部署时的超额风险的有限样本界。每个界都分解为漂移项、估计项和验证噪声项。在使用等权重时，随着数据记录的增长，漂移项会趋于饱和。当折的权重预先固定时，漂移项变为加权平均值，并随着权重向部署时刻移动而缩小。噪声项随着权重集中而增大，并随批量大小的增加而缩小。由于漂移项关于权重是线性的，而噪声项是凸的，因此在该风险界上对权重进行最小化是一个凸优化问题。在美国收入数据和航班延误数据上的应用展示了这一权衡的两个方面。当最优预测器的变化持续存在且每个时期……（摘要原文在此处截断）

    arXiv:2611.23184v1 Announce Type: cross  Abstract: Cross-validation weights all folds equally, so under drift the oldest folds count as much as the newest. We model drift by a H\"older condition in total variation distance and give finite-sample bounds on the excess deployment risk of the configuration that sliding-window cross-validation selects. Each bound splits into drift, estimation, and validation-noise terms. With equal weights the drift terms saturate as the record grows. With fold weights fixed in advance, the drift terms become weighted averages, which shrink as weight moves toward deployment. The noise term grows as the weights concentrate and shrinks with the batch size. The drift terms are linear in the weights and the noise term is convex, so minimizing the bound over the weights is a convex problem. Applications to U.S. earnings and flight delays show both sides of this trade-off. Weight on recent folds pays off when a change in the best predictor persists and each perio
    
[^69]: 关于几乎相等分布之间散度的数值问题

    On numerical issues of divergences between nearly equal distributions

    [https://arxiv.org/abs/2611.18511](https://arxiv.org/abs/2611.18511)

    该论文揭示了当类别分布几乎相等时，KL散度、JSD散度及熵等标准散度运算存在与softmax类似的数值精度问题，并提出了数值稳定的重新表述方案及经过属性测试验证的实现。

    

    Barron等人（2026年）最近发现了softmax和软交叉熵损失中的数值问题。我们证明在类别分布之间的标准散度运算中也存在类似的问题，这包括正向和反向Kullback-Leibler（KL）散度、广义Jensen-Shannon散度（JSD），以及蒙特卡洛KL估计和熵。当分布几乎相等时，舍入和抵消问题会影响前向计算和梯度计算。这些问题可能影响在优化目标中使用散度的场景，例如知识蒸馏、强化学习和近端方法。我们提出了数值稳定的重新表述方案，并为每种情况提供了示例实现，并通过基于属性的测试进行了验证。

    arXiv:2611.18511v1 Announce Type: new  Abstract: Barron et al. (2026) recently identified numerical issues in the softmax and soft cross-entropy loss. We show that similar issues are present in standard divergence operations between categorical distributions. This includes the forward and reverse Kullback-Leibler (KL) divergence and generalized Jensen-Shannon divergence (JSD), as well as Monte Carlo KL estimates and entropy. When the distributions are nearly equal, rounding and cancellation issues affect the forward and gradient computations. These issues can affect settings using divergences in optimization objectives, such as knowledge distillation, reinforcement learning, and proximal methods. We propose numerically stable reformulations and provide example implementations for each setting, validated by property-based tests.
    
[^70]: 拆东墙补西墙：公平性干预的跨属性外部性

    Robbing Peter to Pay Paul: Cross-Attribute Externalities of Fairness Interventions

    [https://arxiv.org/abs/2611.14717](https://arxiv.org/abs/2611.14717)

    本文揭示了针对某一敏感属性的公平性干预可能会加剧其他敏感属性（甚至包括不可观测属性）的不公平性，并利用Wasserstein重心差异刻画这种跨属性外部性，提出了能同时捕捉边际与联合公平性变化的分布框架。

    

    公平性干预通常针对二分类任务中与单一敏感属性相关的差异。然而，当一个数据主体关联多个敏感属性时，针对某一属性提升公平性并不一定会改善所有属性上的公平性：它可能会加剧与其他敏感属性相关的差异，包括从业者无法获取的属性。我们将这些非预期的效应称为公平性的跨属性外部性。我们研究了当一个最初不公平的分类或回归模型针对另一个属性进行修正时，某一敏感属性上的公平性如何变化。我们使用Wasserstein重心差异来刻画这些外部性，并开发了一个分布框架，能够同时捕捉边际公平性度量和联合公平性度量的变化。这凸显了传统公平性[度量方法]的一个局限性。

    arXiv:2611.14717v2 Announce Type: replace-cross  Abstract: Fairness interventions often targets disparities associated with a single sensitive attribute in binary classification tasks. However, when multiple sensitive attributes are associated with a data subject, improving fairness with respect to one attribute does not necessarily improve it across all attributes: it may exacerbate disparities associated with other sensitive attributes, including those that are unavailable to the practitioner. We refer to these unintended effects as cross-attribute externalities of fairness. We study how fairness with respect to one sensitive attribute varies when an initially unfair classification or regression model is corrected with respect to another attribute. We characterize these externalities using Wasserstein barycentric disparities and develop a distributional framework that captures changes in both marginal and joint fairness measures. This highlights a limitation of conventional fairness 
    
[^71]: 基于Katz核的乳腺癌患者分层方法

    Katz Kernel Based Stratification of Breast Cancer Patients

    [https://arxiv.org/abs/2611.14540](https://arxiv.org/abs/2611.14540)

    本文提出了一种基于Katz核的图核方法，通过比较患者特异性的信号通路网络结构来实现乳腺癌患者分层，从而识别具有临床意义的患者亚群并助力新型生物标志物的发现。

    

    将机制性信号网络整合到临床研究中已成为精准肿瘤学的基石。通过捕捉细胞内复杂的生化相互作用，这些网络提供了一个超越简单基因组突变的功能性疾病进展图谱。因此，开发患者特异性信号模型具有重要意义，能够更精细地表示个体疾病谱。在这种背景下，人们日益需要稳健的方法来比较基于图的通路结构，以识别患者队列之间的异同。这种比较对于识别具有临床意义的亚群和发现新型生物标志物至关重要。标准的图核方法虽然在数学上严谨，但往往过于通用而难以实现这一目的，它们经常无法捕捉基因信号传导所特有的特征。

    arXiv:2611.14540v2 Announce Type: replace-cross  Abstract: The integration of mechanistic signaling networks into clinical research has become a cornerstone of precision oncology. By capturing the complex biochemical interactions within a cell, these networks provide a functional map of disease progression that goes beyond simple genomic mutations. Consequently, the development of specific signaling models of patients has significant relevance, allowing for a more granular representation of individual disease profiles. In this setting, there is a growing need for robust methodologies capable of comparing graph based pathway structures to identify similarities and differences across patient cohorts. Such comparisons are essential for the identification of clinically meaningful subgroups, and the discovery of novel biomarkers. Standard graph kernels, while mathematically rigorous, often remain too generic for this purpose. They frequently fail to capture features specific to gene signali
    
[^72]: 具有自动模型复杂度推断的贝叶斯张量链核机器

    Bayesian Tensor Train Kernel Machines with Automatic Model Complexity Inference

    [https://arxiv.org/abs/2611.07129](https://arxiv.org/abs/2611.07129)

    本文提出了贝叶斯张量链核机器（BTT-KM），通过在TT核心上施加分层ARD先验并结合平均场变分推断，实现了从数据中自动推断TT秩与特征维度的闭式贝叶斯学习，且计算复杂度与确定性方法相同。

    

    张量网络（TN）核机器通过使用低秩张量网络来降低计算量和内存需求，从而加速监督学习。然而，大多数基于张量网络的方法是确定性的，其秩和特征维度需要手动调整，而现有的贝叶斯张量网络核机器仅限于CANDECOMP/PARAFAC分解（CPD），其单一秩由所有模式共享。我们将该框架扩展到张量链（TT）分解，提出了贝叶斯张量链核机器（BTT-KM），其在TT核心上的分层ARD类型先验可以从数据中自动推断TT秩和特征维度。由于每个TT核心仅通过其相邻核心与链上其他核心耦合，其闭式后验更新需要沿链传播所有其他核心的贡献。平均场变分推断产生了一个贝叶斯交替最小二乘算法，其计算复杂度与确定性对应算法相同，并具有闭式预测分布。

    arXiv:2611.07129v1 Announce Type: cross  Abstract: Tensor Network (TN) kernel machines accelerate supervised learning by using low-rank TNs to reduce computation and memory. However, most TN-based methods are deterministic with manually tuned ranks and feature dimensions, while the existing Bayesian TN kernel machine is restricted to the CANDECOMP/PARAFAC decomposition (CPD), whose single rank is shared by all modes. We extend this framework to the tensor train (TT) and propose Bayesian Tensor Train Kernel Machines (BTT-KM), whose hierarchical ARD-type priors on the TT cores infer the TT ranks and feature dimensions from data. Since each TT core is coupled to the others only through its neighbors, its closed-form posterior update requires propagating the contributions of all other cores along the chain. Mean-field variational inference yields a Bayesian alternating least squares algorithm with the same computational complexity as its deterministic counterpart and a closed-form predicti
    
[^73]: 主动记忆召回中规模的代价

    The Price of Scale in Proactive Memory Recall

    [https://arxiv.org/abs/2610.34507](https://arxiv.org/abs/2610.34507)

    该论文证明主动记忆召回的门控决策本质上是一个随记忆库规模 N 增长的多重假设检验问题——单条记忆的假阳性率按 -ln(1-α)/N 衰减，导致命中率随 N 呈幂次下降，从理论与实证上解释了记忆库越大、主动记忆触发越容易失效的现象。

    

    一个无需被询问便能主动提起记忆的长期助手，必须在每轮交互中判断其记忆库中是否有内容值得打断用户。我们证明这一决策本质上是一个多重假设检验问题，其难度随记忆库规模 N 的增长而加剧。在 LongMemEval 与 LoCoMo 的流式回放实验中，一个在 N=8 时调校至 5% 误报率的相似度阈值，在 N=2048 时误报率飙升至 87–100%，该现象在各类嵌入模型以及显式与隐式线索中均成立；LLM 判别器虽起点更低，也呈现同样的增长趋势。我们证明，一个误报率为 α 的门控机制，其单条记忆的假阳性率约为 -ln(1-α)/N，这将把其命中率锁定在单次比较 ROC 曲线的最左端，且在正则变化条件下，命中率随 N 以幂次规律衰减并带有显式常数。由此得出的预测与逐 N oracle 阈值的命中率相差在一个百分点以内。该分析解释了为何词汇与语义通道……（原文摘要在此处截断）

    arXiv:2610.34507v1 Announce Type: cross  Abstract: A long-lived assistant that brings up memories without being asked must decide at every turn whether anything in its store is worth an interruption. We show that this decision is a multiple-testing problem whose difficulty grows with the store size $N$. On streaming replays of LongMemEval and LoCoMo, a similarity threshold tuned to 5% false alarms at $N=8$ reaches 87-100% at $N=2048$, across embedding models and for explicit and implicit cues; LLM judges show the same growth from a lower base. We prove that a gate with false-alarm rate $\alpha$ operates at a per-item false-positive rate of about $-\ln(1-\alpha)/N$, which pins its hit rate to the far left of the single-comparison ROC curve and, under regular variation, makes it decay as a power of $N$ with an explicit constant. The resulting prediction matches the hit rate of a per-$N$ oracle threshold to within one percentage point. The analysis explains why lexical and semantic channe
    
[^74]: SABLE：可扩展的仿射贝叶斯潜空间探索

    SABLE: Scalable Affine Bayesian Latent-space Exploration

    [https://arxiv.org/abs/2610.31495](https://arxiv.org/abs/2610.31495)

    SABLE 提出了一种贝叶斯仿射代理优化方法，通过将查询分解为由期望改进（EI）优化的低维跨度内分量和由标量范数支配的跨度外分量，在评估预算远小于潜空间维度的高维潜空间中实现了可扩展的贝叶斯优化。

    

    诸如工程设计、蛋白质等复杂对象往往难以优化，因为它们的设计空间呈现高维结构，且目标评估代价高昂。生成模型提供了连续的潜空间表示，但其潜空间可能具有数千至数万维，形成了潜空间维度 D 远超评估预算的情形。我们提出 SABLE，一种针对各向同性分布且预算有限的高维潜空间的贝叶斯优化方法。SABLE 使用贝叶斯仿射代理模型进行预测与采集，并将高阶结构视为代理模型的偏差。在 N 次观测之后，斜率更新的维度至多为 N-1，从而将每个查询分解为一个至多 N 维的跨度内分量（通过期望改进 EI 进行优化）和一个由标量范数支配的跨度外分量。我们表明……

    arXiv:2610.31495v1 Announce Type: new  Abstract: Complex objects such as engineering designs and proteins are often difficult to optimise because their design spaces exhibit high-dimensional structure and objective evaluations are expensive. Generative models provide continuous latent representations, but their latent spaces can have thousands to tens of thousands of dimensions, creating a regime where the latent dimension $D$ greatly exceeds the evaluation budget. We introduce SABLE, a Bayesian optimisation method for high-dimensional latent spaces with an isotropic distribution and limited budget. SABLE uses a Bayesian affine surrogate for prediction and acquisition, treating higher-order structure as surrogate discrepancy. After $N$ observations, the slope update has dimension at most $N-1$, yielding a decomposition of each query into an in-span component optimised by expected improvement (EI) in at most $N$ dimensions and an off-span component governed by a scalar norm. We show tha
    
[^75]: 哪个模型适合哪个角色？面向LLM智能体工作流的成本高效Pareto推荐

    Which Model for Which Role? Cost-Efficient Pareto Recommendation for LLM Agentic Workflows

    [https://arxiv.org/abs/2610.26676](https://arxiv.org/abs/2610.26676)

    提出CC-Gittins算法，通过利用评估成本与部署成本之间的耦合关系，高效地为LLM智能体工作流中的不同角色推荐质量与成本权衡下的Pareto最优模型分配方案

    

    LLM智能体工作流日益为不同的专业角色采用不同的模型。用户可能在任务质量与部署成本之间偏好不同的权衡，有时还会考虑延迟。寻找良好的模型分配会产生与评估多个候选者不同的额外搜索成本。我们提出了成本耦合Gittins算法（CC-Gittins），这是面向Pareto推荐的单目标成本感知Gittins搜索的多目标扩展。该方法利用了评估成本与部署成本之间的耦合关系：部署成本更高的分配方案，其评估成本也更高。搜索过程在追求更高质量和更低部署成本之间交替进行，复用每次评估来同时为两个搜索方向提供信息。每一批次之后，CC-Gittins会推荐在质量与部署成本的不确定性调整估计下不被支配的分配方案。我们在来自具有两到五个角色的智能体工作流的八个质量-成本矩阵上评估了CC-Gittins

    arXiv:2610.26676v3 Announce Type: replace-cross  Abstract: LLM agentic workflows increasingly employ different models for specialized roles. Users may prefer different trade-offs between task quality and deployment cost, sometimes also considering latency. Finding good assignments incurs a separate search cost from evaluating multiple candidates. We introduce Cost-Coupled Gittins (CC-Gittins), a multi-objective extension of single-objective cost-aware Gittins search for Pareto recommendation. It exploits the coupling between evaluation and deployment costs: assignments that cost more to deploy also cost more to evaluate. The search alternates between seeking higher quality and lower deployment cost, reusing each evaluation to inform both searches. After each batch, CC-Gittins recommends assignments that are nondominated under uncertainty-adjusted estimates of quality and deployment cost. We evaluate CC-Gittins on eight quality--cost matrices from workflows with two to five roles and up
    
[^76]: Wasserstein空间中光滑势-相互作用能量的信赖域优化

    Trust-Region Optimization for Smooth Potential-Interaction Energies in Wasserstein Space

    [https://arxiv.org/abs/2610.08883](https://arxiv.org/abs/2610.08883)

    该论文提出了Wasserstein空间上光滑势-相互作用能量的信赖域优化方法，通过推前曲线上的二次模型、$L^2(\rho)$ 步长半径以及带显式自伴二阶变分算子的Steihaug-Toint子求解器，在温和条件下证明了目标函数单调不增且Wasserstein梯度范数收敛于零。

    

    寻找相互作用粒子的低能量构型以及逼近概率分布，都会导致在Wasserstein空间中最小化势-相互作用能量。这些能量可能是非凸的，因此在利用二阶信息的同时控制局部近似的可靠性变得十分重要。我们研究了在具有有限二阶矩的概率测度的Wasserstein空间上，光滑势-相互作用能量的信赖域优化。该方法使用沿推前曲线的二次模型、$L^2(\rho)$ 步长半径，以及带有显式自伴二阶变分算子的Steihaug-Toint子求解器。比值检验决定步骤的接受与否并指导半径更新。在能量存在下界、且势函数与相互作用核的Hessian全局有界的条件下，我们证明了目标函数单调不增、Wasserstein梯度范数收敛于零，并（得到ε-驻点，摘要原文在此处截断）。

    arXiv:2610.08883v1 Announce Type: cross  Abstract: Finding low-energy configurations of interacting particles and approximating probability distributions lead to the minimization of potential-interaction energies in Wasserstein space. These energies can be nonconvex, making it important to exploit second-order information while controlling the reliability of local approximations. We study trust-region optimization of smooth potential-interaction energies on the Wasserstein space of probability measures with finite second moment. The method uses a quadratic model along pushforward curves, an $L^2(\rho)$ step radius, and a Steihaug-Toint subsolver with an explicit self-adjoint second-variation operator. A ratio test determines acceptance and guides the radius update. Under a lower energy bound and globally bounded Hessians of the potential and interaction kernel, we prove that the objective is nonincreasing, the Wasserstein-gradient norms converge to zero, and an $\varepsilon$-stationary
    
[^77]: FAER：面向语言模型后训练的可审计、效用对齐的轨迹重放

    FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training

    [https://arxiv.org/abs/2610.00385](https://arxiv.org/abs/2610.00385)

    FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。

    

    重放选择器通常依据格式反馈、置信度、新鲜度或响应长度对缓存轨迹进行排序，然而缓存层面的正确性与下游学习器的效用是两个不同的目标。我们形式化了这种“选择到学习”的差距，并提出FAER作为一个可审计的全轨迹重放框架。其中免训练的固定选择器作为协议基线；FAER-UTILITY则是在互不相交的校准块上拟合的学习器感知选择器。归一化梯度对齐被报告为基线方法，而一次性的优化器感知虚拟更新提供了幅度感知的效用面。审计契约会在评估标签加入之前冻结观测字段和重放轨迹。在GSM8K数据集上使用Qwen2.5-1.5B-Instruct模型，匹配学习器的研究显示固定选择器在128次更新下质量达到0.6329，相比之下均匀采样为0.5482，格式反馈为0.6037。仅使用元数据的交叉拟合校准达到0.6...

    arXiv:2610.00385v1 Announce Type: cross  Abstract: Replay selectors often rank cached trajectories by format feedback, confidence, freshness, or response length, although cache-level correctness and downstream learner utility are distinct objectives. We formalize this selection-to-learning gap and introduce FAER as an auditable full-trajectory replay framework. Its training-free fixed selector is a protocol baseline; FAER-UTILITY is the learner-aware selector fitted on disjoint calibration blocks. The normalized gradient alignment is reported as a baseline, while a disposable optimizer-aware virtual update supplies a magnitude-aware utility surface. The audit contract freezes observed fields and replay traces before evaluation labels are joined. On GSM8K with Qwen2.5-1.5B-Instruct, the matched learner study reports quality 0.6329 for the fixed selector, compared with 0.5482 for uniform and 0.6037 for format-feedback under 128 updates. Metadata-only cross-fitted calibration reaches $0.6
    
[^78]: 布尔阈值函数、神经元容量与记忆检索

    Boolean threshold functions, neuron capacity, and memory retrieval

    [https://arxiv.org/abs/2609.29756](https://arxiv.org/abs/2609.29756)

    本文精确计数了n维布尔阈值函数的数目，将单个阈值神经元的容量确定为 $n^2-\log_2(n!)+1+O(n^{-99})$ 比特，把经典误差项从 $O(n)$ 改进到 $O(n^{-99})$，并为神经网络无虚假记忆检索的容量问题提供了数学基础。

    

    单个神经元能记住多少信息？神经网络在不产生虚假记忆的前提下能检索多少条记忆？这些问题都与一个基本问题相关：形如 $f(x)=\operatorname{sgn}(a_0+\langle a,x\rangle)$（$x\in\{-1,1\}^n$）的布尔阈值函数究竟有多少个？本文证明，不同布尔阈值函数的数目 $T_n$ 满足 $T_n=2\binom{2^n-1}{n}\bigl(1+O(n^{-99})\bigr)$。等价地，单个阈值神经元的容量为 $n^2-\log_2(n!)+1+O(n^{-99})$ 比特，这将 Kahn–Komlós–Szemerédi 结果中的 $O(n)$ 误差项改进为 $O(n^{-99})$。为证明这一点，我们证明了：当 $1\le r\le n-1$ 且 $v_1,\ldots,v_r$ 从 $\{-1,1\}^n$ 中随机选取时，$\mathbb P\{\langle v_1,\ldots,v_r\rangle\cap\{-1,1\}^n =\{\pm v_1,\ldots,\pm v_r\} \} =1-O(n^{-99})$。在用于记忆检索的 Kanter–Sompolinsky 哈密顿量的背景下，这一结果表明……

    arXiv:2609.29756v1 Announce Type: cross  Abstract: How much information can a single neuron remember? How many memories can neural networks retrieve without creating false memories? These questions are related to a basic question: how many Boolean threshold functions $f(x)=\operatorname{sgn}(a_0+\langle a,x\rangle)$, $x\in\{-1,1\}^n$, are there? In this paper, we show that the number $T_n$ of distinct Boolean threshold functions is \[ T_n=2\binom{2^n-1}{n}\bigl(1+O(n^{-99})\bigr). \] Equivalently, the capacity of a single threshold neuron is $n^2-\log_2(n!)+1+O(n^{-99})$ bits, improving the $O(n)$ error term in the result of Kahn--Koml\'os--Szemer\'edi to $O(n^{-99})$. To prove this, we show that, for $1\le r\le n-1$, and $v_1,\ldots,v_r$ are chosen at random from $\{-1,1\}^n$, \[ \mathbb P\!\left\{ \langle v_1,\ldots,v_r\rangle\cap\{-1,1\}^n =\{\pm v_1,\ldots,\pm v_r\} \right\} =1-O(n^{-99}). \] In the context of the Kanter--Sompolinsky Hamiltonian for memory retrieval, this identifie
    
[^79]: 基于贝叶斯多臂老虎机Gittins指数的高效成本感知LLM评估

    Efficient Cost-Aware LLM Evaluation via Bayesian Bandit Gittins Indices

    [https://arxiv.org/abs/2609.25645](https://arxiv.org/abs/2609.25645)

    提出GittinsEval，一种基于贝叶斯最优Gittins策略的成本感知LLM评估方法，通过轻量级在线更新高效决定下一个待评估配置及停止时机，在多个基准上以更低的评估成本取得有竞争力的性能。

    

    在每个基准测试项目上穷尽评估每个候选LLM配置以找出高性能配置的成本十分高昂。我们将配置选择问题形式化为一个成本感知的贝叶斯多臂老虎机问题，并提出GittinsEval，该方法借鉴贝叶斯最优的Gittins策略来决定下一步评估哪个配置以及何时停止。我们通过一个随时推荐规则对该策略进行了扩展，使其能够同时覆盖完全评估和部分评估的配置，并使用LCB风格的分数来考虑后验不确定性。GittinsEval在计算上非常高效，只需在离线预计算之后进行轻量级的在线更新。在GSM8K、PIQA、AlpacaEval和MMLU响应矩阵上，GittinsEval始终具有竞争力，在大样本基准上相比配置级贝叶斯优化取得了尤为显著的提升，在大候选任务上相比不考虑成本的bandit基线也表现更优。至关重要的是，GittinsEval通常能够以（原文在此处截断）……

    arXiv:2609.25645v1 Announce Type: cross  Abstract: Exhaustively evaluating every candidate LLM configuration on every benchmark item to identify a high-performing one is costly. We formulate configuration selection as a cost-aware Bayesian bandit problem and propose GittinsEval, which draws on the Bayesian-optimal Gittins policy to determine which configuration to evaluate next and when to stop. We extend the policy with an anytime recommendation rule over both fully and partially evaluated configurations, using an LCB-style score to account for posterior uncertainty. GittinsEval is computationally efficient, requiring only lightweight online updates after offline precomputation. Across GSM8K, PIQA, AlpacaEval, and MMLU response matrices, GittinsEval is consistently competitive, with particularly strong gains over configuration-level Bayesian optimization on large-example benchmarks and over cost-unaware bandit baselines on large-candidate tasks. Crucially, GittinsEval often attains ne
    
[^80]: 单循环匹配多循环复杂度：非凸-凹极小极大优化中的最优优化平稳性与已知最佳博弈平稳性

    Matching Multi-Loop Complexities with a Single Loop: Optimal Optimization Stationarity and Best-Known Game Stationarity in Nonconvex--Concave Minimax Optimization

    [https://arxiv.org/abs/2609.17973](https://arxiv.org/abs/2609.17973)

    该论文提出了一种结合投影外梯度更新、对偶动量和移动近端中心的单循环投影阻尼外梯度算法，在非凸-凹极小极大优化中以单循环方法匹配了多循环方法的复杂度，同时实现了最优的优化平稳性保证和已知最佳的博弈平稳性保证。

    

    我们为光滑非凸-凹极小极大优化问题引入了一个新的单循环算法框架。由此得到的投影阻尼外梯度方法结合了投影外梯度更新、对偶动量和移动近端中心。在优化平稳性和博弈平稳性两种准则下，我们的方法在单循环一阶方法中达到了已知的最佳复杂度。对于优化平稳性，我们的方法达到了 $O(L^2D_Y\bar\Delta_0\varepsilon^{-3})$ 的梯度复杂度，其中 $L$ 是梯度Lipschitz常数，$D_Y$ 是对偶可行集直径的界，$\bar\Delta_0$ 是一个涉及值函数间隙和初始梯度的初始化量。此外，通过引入固定中心预热阶段，复杂度可以改进为 $O(L^2D_Y\Delta_\phi\varepsilon^{-3})$，外加一个可忽略的低阶可加代价，其中 $\Delta_\phi:=\phi(x_0)-\inf_x\phi(x)$。我们进一步

    arXiv:2609.17973v1 Announce Type: cross  Abstract: We introduce a new single-loop algorithmic framework for smooth nonconvex--concave minimax optimization. The resulting projected damped extragradient method combines projected extragradient updates, dual momentum, and a moving proximal center. Under both the optimization-stationarity and game-stationarity criteria, our method achieves the best-known complexity among single-loop first-order methods. For optimization stationarity, our method achieves a gradient complexity of $O(L^2D_Y\bar\Delta_0\varepsilon^{-3})$, where $L$ is the gradient Lipschitz constant, $D_Y$ bounds the diameter of the dual feasible set, and $\bar\Delta_0$ is an initialization quantity involving the value-function gap and the initial gradients. Moreover, by incorporating a fixed-center warm-up phase, the complexity can be improved to $O(L^2D_Y\Delta_\phi\varepsilon^{-3})$, up to an additive lower-order cost, where $\Delta_\phi:=\phi(x_0)-\inf_x\phi(x)$. We further
    
[^81]: i.i.d.模型中序列可检验性与变化可检测性的完整刻画

    A complete characterization of sequential testability and change detectability in i.i.d. models

    [https://arxiv.org/abs/2609.05752](https://arxiv.org/abs/2609.05752)

    该论文首次对i.i.d.复合模型中的序列检验与变化检测给出了完整刻画：可检验性当且仅当备择假设可被可数族有限块事件分离，而可数族有界块赌注能否恢复内在增长率则决定了最优e-过程及最优期望样本量系数1/Γ(Q)能否同时达到。

    

    我们在i.i.d.复合假设检验问题中给出了序列检验与变化检测的定性和定量充要条件。当且仅当备择假设能被一族可数的有限块事件与原假设分离时，才存在对所有备择假设功效为1的水平为α的检验。等价条件可通过随机化检验、有界得分、e-过程、缩减过滤的检验上鞅，以及具有正有限块弱*凸包分离的可数覆盖来表述。该构造给出了逐点期望样本量 O_Q(log(1/α))。在定量层面，存在一个e-过程在每个备择假设处达到内在逐点增长率 Γ(Q)，当且仅当一族可数的有界块赌注能够恢复这些增长率。同样的条件刻画了能否同时达到最优的期望样本量系数 1/Γ(Q)；以及任意可实现的下增长率（的刻画）。

    arXiv:2609.05752v2 Announce Type: replace-cross  Abstract: We give qualitative and quantitative necessary and sufficient conditions for sequential testing and change detection in an i.i.d. composite problem. A level-\(\alpha\) test with power one against every alternative exists if and only if the alternatives are separated from the null by a countable family of finite-block events. Equivalent conditions use randomized tests, bounded scores, e-processes, reduced-filtration test supermartingales, and a countable cover with positive finite-block weak-$*$ convex-hull separation. The construction gives pointwise expected sample size \(O_Q(\log(1/\alpha))\). Quantitatively, one e-process attains the intrinsic pointwise growth rate \(\Gamma(Q)\) at every alternative if and only if a countable family of bounded block bets recovers these rates. The same condition characterizes simultaneous attainment of the sharp expected-sample-size coefficient \(1/\Gamma(Q)\); arbitrary attainable lower rate
    
[^82]: 二维奇异模型学习系数的精确代数计算

    Exact Algebraic Computation of Learning Coefficients for Two-Dimensional Singular Models

    [https://arxiv.org/abs/2608.20183](https://arxiv.org/abs/2608.20183)

    本文提出了首个确定性算法，能精确计算二维奇异模型的局部RLCT，并给出了复杂度上界和广泛适用性证明。

    

    经典信息准则如贝叶斯信息准则（BIC）依赖于正则性假设，这些假设在奇异模型中失效，导致在深度学习等场景下的模型选择不正确。广泛适用的贝叶斯信息准则（WBIC）依赖于局部学习系数$\lambda$，在解析情况下，该系数与模型的Kullback-Leibler散度的局部实对数规范阈值（RLCT）一致，以捕捉正确的边际似然渐近性。学习系数的精确计算仅限于特殊情况，通常只能使用基于采样的估计方法。我们提出了第一个确定性算法，能够精确计算任何二维模型的局部RLCT，其中模型的Kullback-Leibler距离与多项式接触等价，我们推导出其复杂性上界，并展示了其在广泛模型类别中的有效性，附有应用示例。

    arXiv:2608.20183v1 Announce Type: new  Abstract: Classical information criteria such as the Bayesian Information Criterion (BIC) rely on regularity assumptions that break down for singular models, leading to incorrect model selection in settings such as deep learning. The Widely Applicable Bayesian Information Criterion (WBIC) relies on local learning coefficients $\lambda$, which in the analytic case coincides with local Real Log Canonical Thresholds (RLCT) of the Kullback-Leibler divergence of the model, to capture correct marginal likelihood asymptotics. Exact computation of the learning coefficients has been limited to special cases, and only sampling-based estimation methods are generally applicable. We present the first deterministic algorithm that computes local RLCTs exactly for any two-dimensional model whose Kullback-Leibler distance is contact equivalent to a polynomial, derive a bound on its complexity, and demonstrate its effectiveness for a broad class of models, with app
    
[^83]: 量子机器学习用于电力系统攻击检测的基准测试：评估选择在模型之前就决定了结果

    Benchmarking Quantum Machine Learning for Power-System Attack Detection: Evaluation Choices Decide the Outcome Before the Models Do

    [https://arxiv.org/abs/2608.15617](https://arxiv.org/abs/2608.15617)

    该论文通过系统性的基准测试揭示，电力系统攻击检测中量子机器学习与经典模型的比较结果高度依赖于评估协议中的八项关键选择，这些选择在模型运行前就决定了结论走向。

    

    arXiv:2608.15617v1 公告类型：新 摘要：针对电力系统网络攻击的机器学习检测器本身也是攻击面，而量子机器学习已被提议用于此目的。我们在公共电力系统攻击数据（密西西比州立大学/ORNL）上，将保真度核支持向量机和变分分类器与六种调优后的经典模型进行基准测试，涵盖白盒、迁移、基于决策的黑盒和投毒攻击。我们的主要发现是方法论层面的：基准测试的答案在模型之前就由评估者的选择决定了。评估协议中的六项选择和基准测试本身调优中的两项选择，每一项都在固定模型下逆转或移动了一个结论。最大的是数据分割：行级协议得分为0.905宏F1，而保留整个源文件时得分为0.594；在封顶匹配维度设置下，量子部分在随机噪声范围内，经典部分高出0.024。保真度核在受到攻击前看起来最稳健。

    arXiv:2608.15617v1 Announce Type: new  Abstract: Machine-learning detectors for power-system cyberattacks are themselves attack surfaces, and quantum machine learning has been proposed for them. We benchmark fidelity-kernel SVMs and variational classifiers against six tuned classical models on public power-system attack data (Mississippi State/ORNL), across white-box, transfer, decision-based black-box, and poisoning attacks. Our headline finding is methodological: the benchmark's answers are set by the evaluator's choices before the models. Eight choices -- six in the evaluation protocol, two in the tuning the benchmark itself runs -- each reversed or moved a conclusion at fixed models. The largest is the split: the row-level protocol scores 0.905 macro-F1 where holding whole source files out leaves 0.594, and in the capped matched-dimensionality regime the quantum arm sits within noise of chance with the classical arm 0.024 above it. A fidelity kernel looks most robust until attacked
    
[^84]: 分布强化学习中分位数时序差分学习的在线推断

    Online Inference for Quantile Temporal Difference Learning in Distributional Reinforcement Learning

    [https://arxiv.org/abs/2608.12973](https://arxiv.org/abs/2608.12973)

    本文提出了一种基于随机缩放的在线推断方法，通过泛函中心极限定理支持，在不存储完整轨迹的情况下对分布强化学习中的QTD进行统计推断，显著减少了内存需求。

    

    本文研究了分布强化学习中分位数时序差分学习（QTD）的统计推断方法。假设可以访问生成模型，我们首先为同步和异步QTD建立了泛函中心极限定理，表明QTD的平均迭代弱收敛于一个重新缩放的布朗运动。接着，我们提供了在线推断方法。基于随机缩放，该推断过程利用整个QTD路径上的信息构建了一个渐近枢轴统计量进行推断。同时，所提出的统计量可以在线计算，无需存储QTD迭代的完整轨迹。这大大降低了内存需求，并实现了分布强化学习中的高效统计推断。

    arXiv:2608.12973v1 Announce Type: cross  Abstract: In this paper, we study how to perform statistical inference for quantile temporal difference learning (QTD) in distributional reinforcement learning. Assuming access to a generative model, we first establish functional central limit theorems for both synchronous and asynchronous QTD, which show that the averaged iterates of QTD converge weakly to a rescaled Brownian motion. We next provide online inference methods. Based on random scaling, the inference procedure constructs an asymptotically pivotal statistic for inference by using the information along the whole QTD path. Meanwhile, the proposed statistic can be computed online without storing the entire trajectory of QTD iterates. This substantially reduces the memory requirement and enables efficient statistical inference in distributional reinforcement learning.
    
[^85]: 概率回归树中缺失数据的处理

    Handling Missing Data in Probabilistic Regression Trees

    [https://arxiv.org/abs/2608.06195](https://arxiv.org/abs/2608.06195)

    本文提出三种在概率回归树构建过程中直接处理缺失预测变量值的方法，无需单独的插补步骤，并能在任意缺失模式下保持概率守恒和边缘兼容性等基本概率特性。

    

    概率回归树是经典回归树的一种平滑替代方案，在完整数据情况下具有一致性，通过概率化的分裂分配产生连续的预测。本文扩展了这一框架，提出了三种方法，可以在树构建过程中直接处理缺失的预测变量值，而无需单独的插补步骤。这些修改被定义为在任意缺失协变量值模式下，保留原始方法的基本概率特性，包括概率守恒和边缘兼容性。所提出的方法在几个呈现不同程度缺失的真实数据集上进行了评估，并与经典回归树进行了比较。此外，一项半合成研究（其中缺失值以完全随机、随机和非随机方式注入）将所提出的策略与先插补后拟合的方法进行了比较。

    arXiv:2608.06195v2 Announce Type: replace-cross  Abstract: Probabilistic regression trees are a smooth alternative to classical regression trees, consistent in the complete-data case, producing continuous predictions through probabilistic split assignments. This paper extends this framework by proposing three approaches to accommodate missing predictor values directly during tree construction, without a separate imputation step. These modifications are defined to preserve the fundamental probabilistic properties of the original methodology, including probability conservation and marginal compatibility, under arbitrary patterns of missing covariate values. The proposed methods are evaluated on several real-world datasets exhibiting different levels of missingness and are compared with classical regression trees. In addition, a semi-synthetic study, in which missing values are injected completely at random, at random and not at random, compares the proposed strategies with impute-then-fi
    
[^86]: 损失函数看不到基，但 Adam 能看到

    The Loss Does Not See the Basis, but Adam Does

    [https://arxiv.org/abs/2608.05136](https://arxiv.org/abs/2608.05136)

    论文证明了旋转等变性是梯度下降在过参数化问题中产生低秩偏置的关键机制，而 Adam 因破坏了旋转对称性而不具备该偏置，并提出了一种基于范数的判别测试来预测各优化器的解的分组行为。

    

    不同的优化算法在同一个问题上可以找到截然不同的解。当初始化接近于零时，梯度下降（GD）在过参数化问题中表现出低秩偏置，而 Adam 则不然。GD 具有旋转等变性：对分解模型 $W = UV^\top$ 的两个因子用正交矩阵进行旋转，损失函数保持不变，GD 的解也随之旋转。相比之下，Adam 不具备这一性质。我们证明，任何仅使用当前梯度且满足旋转对称性的算法，都等价于一个仅依赖于梯度 Gram 矩阵的左预条件子。由此得到一个基于范数的判别测试：Muon 中的谱范数通过该测试，而 Adam 的 $\ell_\infty$ 范数则不通过。在欠定矩阵感知问题上的实证结果显示，九种优化器的分组与该测试的预测一致。在坐标间共享 Adam 的二阶矩可以逐渐恢复低秩偏置，并且随着步长衰减，其解逐渐趋近 GD 的解。

    arXiv:2608.05136v2 Announce Type: replace  Abstract: Different optimization algorithms can find very different solutions to the same problem. Gradient descent (GD) exhibits a low-rank bias in overparameterized problems when initialized near zero, while Adam does not. GD is rotationally equivariant: rotating both factors of a factored model $W = UV^\top$ by an orthogonal matrix leaves the loss unchanged, and GD's solution rotates accordingly. In contrast, Adam is not. We show that any algorithm using only the current gradient and respecting rotational symmetry amounts to a left-preconditioner depending only on the gradient's Gram matrix. This gives a norm-based test, which passes for the spectral norm in Muon and fails for Adam's $\ell_\infty$ norm. Empirically, nine optimizers fall into groups consistent with the test on underdetermined matrix sensing. Sharing Adam's second moment across coordinates gradually recovers a low-rank bias, and its solutions approach GD's as the stepsize dec
    
[^87]: 通过随机李雅普诺夫稳定性分析大步长随机梯度下降在交叉熵损失下的收敛性

    Convergence of Large-Stepsize SGD for the Cross-entropy Loss via Stochastic Lyapunov Stability

    [https://arxiv.org/abs/2606.30930](https://arxiv.org/abs/2606.30930)

    本文借助随机李雅普诺夫稳定性理论证明，大步长SGD在多分类交叉熵损失下（含线性分类器和两层神经网络）虽会在“稳定边缘”振荡与稳定下降之间交替，但迭代点总能在固定次数内回归稳定，从而实现最佳迭代意义上的收敛。

    

    arXiv:2606.30930v2 公告类型：替换 摘要：现代深度学习已被证明能够以远超经典优化理论所能解释的学习率运行。以往对这一现象的大多数分析都集中在确定性梯度下降上，而随机设置在很大程度上尚未被探索。在这项工作中，我们为应用于多分类交叉熵损失的随机梯度下降（SGD）提供了精确的收敛保证，涵盖了线性分类器和两层神经网络。我们表明，SGD的随机性可能导致其动力学在由曲率驱动的振荡所主导的“稳定边缘”区域与期望损失以可控速率下降的稳定区域之间交替。尽管如此，我们证明了SGD能够稳定动力学，确保迭代点在固定的迭代次数内回归稳定，从而即使在学习率很大的情况下也能实现最佳迭代意义上的收敛。实验验证了我们的理论结果。

    arXiv:2606.30930v2 Announce Type: replace  Abstract: Modern deep learning has been shown to operate using learning rates far larger than those justified by classical optimization theory. Most prior analyses of this phenomenon focus on deterministic gradient descent, leaving the stochastic setting largely unexplored. In this work, we provide sharp convergence guarantees for Stochastic Gradient Descent (SGD) applied to the multiclass cross-entropy loss, for both linear classifiers and two-layer neural networks. We show that the stochasticity of SGD may cause the dynamics to alternate between an edge-of-stability regime that is dominated by curvature-driven oscillations, and a stable regime in which the expected loss decreases at a controlled rate. Despite that, we prove that SGD stabilizes the dynamics, ensuring that the iterates return to stability in a fixed number of iterations and allowing convergence in the best-iterate sense even with large learning rates. Experiments validate our 
    
[^88]: 流行率偏移下的共形选择：FDR界与有限校准

    Conformal Selection under Prevalence Shift: FDR Bounds and Finite Calibration

    [https://arxiv.org/abs/2606.22336](https://arxiv.org/abs/2606.22336)

    该文揭示了源总体与测试总体之间的流行率偏移会破坏cfBH方法的FDR控制，并给出大样本下的最坏情形FDR界（如名义10%可升至近40%），同时证明在非目标得分分布一致的条件下，仅用带标签非目标样本校准可摆脱对非目标比例的限制。

    

    共形化的Benjamini–Hochberg方法（cfBH）利用带标签的校准数据来筛选候选对象，并在适当假设下控制错误发现率（FDR）。我们研究了源总体与测试总体之间的差异如何影响FDR与功效。当非目标样本的得分在两个总体中服从相同分布时，已有结果保证：只要非目标样本在源总体中的占比不低于其在测试总体中的占比，截断版cfBH即可实现FDR控制。当这一流行率条件不成立时，FDR可能超过名义水平。我们确定了大样本情形下FDR的最坏情况：例如，当源总体中非目标占比为20%、而测试总体中为80%时，10%的名义水平可能导致FDR接近40%。仅使用带标签的非目标样本进行校准，在其得分分布仍然一致的条件下，可以避免对非目标比例的这一限制。然而，任何非空的选择结果都必须包含一个最小数量的……（摘要原文在此处截断）

    arXiv:2606.22336v2 Announce Type: replace-cross  Abstract: Conformalized Benjamini--Hochberg (cfBH) uses labeled calibration data to select candidates while controlling the false discovery rate (FDR) under suitable assumptions. We examine how differences between source and test populations affect FDR and power. When non-target scores have the same distribution in both populations, existing results guarantee FDR control for clipped cfBH if non-targets are at least as common in the source population as in the test population. When this prevalence condition fails, FDR can exceed the nominal level. We determine the worst-case FDR in large samples. For example, a nominal level of 10\% can yield FDR approaching 40\% when the source population has 20\% non-targets and the test population has 80\%. Calibrating only against labeled non-targets avoids this restriction on their proportions, provided their score distributions still agree. However, every nonempty selection must contain a minimum nu
    
[^89]: 基于代理模型的Shapley与Banzhaf交互作用近似

    Proxy-Based Approximation of Shapley and Banzhaf Interactions

    [https://arxiv.org/abs/2605.22738](https://arxiv.org/abs/2605.22738)

    ProxySHAP将树代理模型的高样本效率与情景残差校正相结合，实现了Shapley与Banzhaf交互作用估计在速度与精度上的兼得，并首次提出多项式时间的树集成精确交互指数计算方法。

    

    Shapley与Banzhaf交互作用能够刻画现代机器学习应用中固有的复杂动态。然而，现有的高阶交互作用估计器在速度与精度之间难以兼顾。为克服这一局限，我们提出了ProxySHAP。ProxySHAP将基于树的代理模型的高样本效率与情景残差校正相结合，从而恢复一致性。在理论层面，我们推导出了干预式TreeSHAP的多项式时间泛化方法，可为树集成模型计算精确的交互作用指数，从而规避了以往方法中指数级树深度的依赖。此外，我们对残差调整策略进行了正式分析，刻画了最大样本复用在何种具体条件下能够校正代理偏差，同时其方差不会随交互作用规模呈指数级增长。大量基准测试表明，ProxySHAP树立了新的最先进水平。

    arXiv:2605.22738v3 Announce Type: replace-cross  Abstract: Shapley and Banzhaf interactions capture the complex dynamics inherent in modern machine learning applications. However, current estimators for these higher-order interactions trade off between speed and accuracy. To overcome this limitation, we introduce ProxySHAP. ProxySHAP reconciles the high sample efficiency of tree-based proxy models with a situational residual correction recovering consistency. On a theoretical level, we derive a polynomial-time generalization of interventional TreeSHAP that computes exact interaction indices for tree ensembles, thereby bypassing exponential tree-depth dependencies in prior methods. Furthermore, we formally analyze the residual adjustment strategy, characterizing the specific conditions under which Maximum Sample Reuse (MSR) corrects proxy bias without its variance scaling exponentially with interaction size. Extensive benchmarking demonstrates that ProxySHAP sets a new state-of-the-art 
    
[^90]: 切片正则化最优传输

    Sliced-Regularized Optimal Transport

    [https://arxiv.org/abs/2604.23944](https://arxiv.org/abs/2604.23944)

    本文提出切片正则化最优传输（SROT），首次以切片最优传输计划作为参考先验来正则化经典最优传输，在保持可扩展性且稀疏情形下复杂度更低的同时，能在相同正则化水平下比熵正则化OT更准确地逼近精确最优传输计划，并且所得传输计划优于参考的切片最优传输计划本身。

    

    我们提出了一种新的正则化最优传输（OT）公式，称为切片正则化最优传输（SROT）。与熵正则化最优传输（EOT）将传输计划向独立耦合方向正则化不同，SROT将传输计划向切片最优传输（SOT）计划方向正则化。据我们所知，SROT是首个利用SOT计划的一种形式作为参考来改进经典最优传输的方法。我们给出了SROT的正式定义，推导了其对偶形式，并提供了SROT的后贝叶斯解释。随后，我们开发了一种Sinkhorn风格的高效计算算法。该算法保留了EOT的可扩展性，并且当SOT计划稀疏时，能够实现更低的每次迭代复杂度。通过将可扩展的SOT计划作为先验引入，在相同正则化水平下，SROT比EOT能够更准确地逼近精确的最优传输计划。此外，所得的传输计划还优于作为参考的SOT计划本身。

    arXiv:2604.23944v5 Announce Type: replace-cross  Abstract: We propose a new regularized optimal transport (OT) formulation, termed sliced-regularized optimal transport (SROT). Unlike entropic OT (EOT), which regularizes the transport plan toward an independent coupling, SROT regularizes it toward a sliced OT (SOT) plan. To the best of our knowledge, SROT is the first approach to leverage a version of SOT plan as a reference to improve classical OT. We provide a formal definition of SROT, derive its dual formulation, and provide a post-Bayesian interpretation of SROT. We then develop a Sinkhorn-style algorithm for efficient computation. It retains the scalability of EOT and, when the SOT plan is sparse, achieves even lower per-iteration complexity. By incorporating a scalable SOT plan as a prior, SROT yields more accurate approximations of the exact OT plan than EOT under the same level of regularization. Moreover, the resulting transport plan improves upon the reference SOT plan itself
    
[^91]: 线性空间中具有良好条件数的无关扰动

    Well-Conditioned Oblivious Perturbations in Linear Space

    [https://arxiv.org/abs/2604.23193](https://arxiv.org/abs/2604.23193)

    该论文提出了一种仅需 O(n) 个 O(log n) 比特随机数的无关扰动方法，能将任意确定性矩阵的条件数降至 O(n)（效果与高斯扰动相当），从而仅需 O(n log n) 个随机比特即可在 O(n²log(n)/ε) 时间内求解 n×n 线性方程组至 ε 后向误差，推动了矩阵扰动的去随机化。

    

    用小的高斯噪声扰动确定性的 n 维矩阵是算法平滑分析 [Spielman and Teng, JACM 2004] 的基石，因为它能将输入的条件数降低到 O(n)，从而随之降低许多矩阵算法的复杂度。然而，在实际算法部署中，这类扰动由于需要生成和存储 n² 个高斯随机变量而代价高昂。我们提出了一种只需生成和存储 O(n) 个、精度为 O(log n) 比特的随机数的扰动方法，它能将任意确定性矩阵的条件数降低到 O(n)，与高斯扰动的效果相当。我们的结果朝着数值线性代数中矩阵扰动的去随机化迈进了一步；特别地，它意味着我们可以仅使用 O(n log n) 个随机比特，在 O(n²log(n)/ε) 时间内将 n×n 线性方程组求解至 ε 的后向误差。

    arXiv:2604.23193v2 Announce Type: replace-cross  Abstract: Perturbing a deterministic $n$-dimensional matrix with small Gaussian noise is a cornerstone of smoothed analysis of algorithms [Spielman and Teng, JACM 2004], as it reduces the condition number of the input to $O(n)$, and with it the complexity of many matrix algorithms. However, when deployed algorithmically, these perturbations are expensive due to the cost of generating and storing $n^2$ Gaussian random variables. We propose a perturbation that requires generating and storing $O(n)$ random numbers in $O(\log n)$ bits of precision, and reduces the condition number of any deterministic matrix to $O(n)$, matching Gaussian perturbations. Our result takes a step towards derandomizing matrix perturbations in numerical linear algebra, and in particular, implies that we can solve an $n\times n$ linear system to within $\varepsilon$ backward error in $O(n^2\log(n)/\varepsilon)$ time using only $O(n\log n)$ random bits.   In our cons
    
[^92]: 基于随机最优控制的稀有事件分析

    Rare Event Analysis via Stochastic Optimal Control

    [https://arxiv.org/abs/2604.13213](https://arxiv.org/abs/2604.13213)

    该论文提出了一种基于随机最优控制的committor函数估计新框架，将committor学习建立在有上限的命中时间轨迹之上，为稀有事件分析提供了有原则且严谨的方法。

    

    稀有事件，从生物分子构象变化、相变到化学反应，是许多物理系统的核心，但由于无偏模拟很少能产生这些事件，因此极难通过计算进行研究。跃迁路径理论为分析此类事件提供了一个严格的统计框架：它刻画了两个亚稳态（反应物与产物）之间的反应性轨迹集合，其核心对象——committor函数（即系统下一步到达产物而非反应物的概率）——编码了关键的动力学与热力学信息。我们提出了一个基于随机最优控制的committor估计框架，采用了一种有原则的表述方式，将committor学习建立在有上限的命中时间轨迹之上。在该框架中，committor定义了一个反馈控制，其大小与committor对数的梯度成正比，用于引导轨迹……

    arXiv:2604.13213v4 Announce Type: replace-cross  Abstract: Rare events, from biomolecular conformational changes and phase transitions to chemical reactions, are central to many physical systems, yet extremely difficult to study computationally because unbiased simulations seldom produce them. Transition Path Theory provides a rigorous statistical framework for analyzing such events: it characterizes the ensemble of reactive trajectories between two metastable states (reactant and product), and its central object, the committor function (the probability that the system next reaches the product rather than the reactant), encodes essential kinetic and thermodynamic information. We introduce a framework for committor estimation that builds on stochastic optimal control, with a principled formulation that grounds committor learning on capped hitting-time trajectories. In this framework the committor defines a feedback control, proportional to the gradient of its logarithm, that steers traj
    
[^93]: 数据集蒸馏通过非线性任务的基于梯度学习高效编码低维表示

    Dataset Distillation Efficiently Encodes Low-Dimensional Representations from Gradient-Based Learning of Non-Linear Tasks

    [https://arxiv.org/abs/2603.14830](https://arxiv.org/abs/2603.14830)

    本文首次从理论上证明，在两层神经网络的梯度训练中，数据集蒸馏算法能将多指标模型等非线性任务的低维结构高效编码进蒸馏数据，且所需存储复杂度仅为 $\tilde{\Theta}(r^2d+L)$。

    

    数据集蒸馏是一种感知训练过程的数据压缩技术，作为降低优化与数据存储成本的有效工具，近来受到越来越多的关注。然而，相关进展在很大程度上仍是经验性的。从训练过程中提取任务相关信息并将这些信息高效编码到合成数据点中的底层机制至今仍不明确。本文对应用于宽度为 $L$ 的两层神经网络基于梯度训练的数据集蒸馏实用算法进行了理论分析。通过聚焦于一种称为多指标模型的非线性任务结构，我们证明了问题的低维结构会被高效地编码到所得的蒸馏数据中。该蒸馏数据集以 $\tilde{\Theta}(r^2d+L)$ 的所需存储复杂度复现出一个具有高泛化能力的模型，其中 $d$ 和 $r$ 分别为输入维度和内在维度。

    arXiv:2603.14830v4 Announce Type: replace  Abstract: Dataset distillation, a training-aware data compression technique, has recently attracted increasing attention as an effective tool for mitigating costs of optimization and data storage. However, progress remains largely empirical. Mechanisms underlying the extraction of task-relevant information from the training process and the efficient encoding of such information into synthetic data points remain elusive. In this paper, we theoretically analyze practical algorithms of dataset distillation applied to the gradient-based training of two-layer neural networks with width $L$. By focusing on a non-linear task structure called multi-index model, we prove that the low-dimensional structure of the problem is efficiently encoded into the resulting distilled data. This dataset reproduces a model with high generalization ability for a required memory complexity of $\tilde{\Theta}$$(r^2d+L)$, where $d$ and $r$ are the input and intrinsic dim
    
[^94]: 信赖域内点随机序贯二次规划方法

    A Trust-Region Interior-Point Stochastic Sequential Quadratic Programming Method

    [https://arxiv.org/abs/2603.10230](https://arxiv.org/abs/2603.10230)

    本文提出一种信赖域内点随机序贯二次规划方法，利用满足自适应精度条件的随机估计和内点障碍策略求解随机目标加确定性非线性约束的优化问题，并证明了KKT残差几乎必然收敛。

    

    本文提出了一种信赖域内点随机序贯二次规划（TR-IP-SSQP）方法，用于求解目标函数随机、等式与不等式约束为确定性非线性的优化问题。在该设定下，目标函数及其梯度的精确求值不可获得，但可以构造其随机估计。特别地，在每次迭代中，我们的方法构建随机预言机，其目标值与梯度的估计以固定概率满足适当的自适应精度条件。为处理不等式约束，我们采用内点法（IPM），其中障碍参数取自一个预先设定的递减序列，并且一旦通过随机平稳性检验，障碍参数就会被减小。在标准假设下，我们证明了在障碍参数减小的迭代点上，KKT残差几乎必然趋于零。

    arXiv:2603.10230v2 Announce Type: replace-cross  Abstract: In this paper, we propose a trust-region interior-point stochastic sequential quadratic programming (TR-IP-SSQP) method for solving optimization problems with a stochastic objective and deterministic nonlinear equality and inequality constraints. In this setting, exact evaluations of the objective function and its gradient are unavailable, but their stochastic estimates can be constructed. In particular, at each iteration our method builds stochastic oracles whose estimates of the objective value and gradient satisfy proper adaptive accuracy conditions with a fixed probability. To handle inequality constraints, we adopt an interior-point method (IPM), in which the barrier parameter is taken from a prescribed decreasing sequence and is reduced once a stochastic stationarity test is passed. Under standard assumptions, we show that, almost surely, the KKT residual vanishes along the iterations at which the barrier parameter is red
    
[^95]: 论多校准梯度提升的收敛性

    On the Convergence of Multicalibration Gradient Boosting

    [https://arxiv.org/abs/2602.06773](https://arxiv.org/abs/2602.06773)

    本文首次为多校准梯度提升算法提供了理论收敛保证，证明预测更新以 O(1/√T) 的速率衰减，并在平方误差损失及弱学习器平滑性假设下进一步提升至线性收敛。

    

    多校准梯度提升近来作为一种可扩展的方法出现，它能够在实证上产生近似多校准的预测器，并已在网络规模的系统中部署应用。尽管取得了这一实证成功，但其收敛性质尚未得到充分理解。本文为多校准梯度提升算法提供了计算保证。我们证明连续预测更新的幅度以 $O(1/\sqrt{T})$ 的速率衰减（在 $T$ 轮上取最小值），这意味着逐轮的经验多校准误差也具有相同的收敛速率界。对于平方误差损失，在对弱学习器附加平滑性假设的条件下，该速率可提升至线性收敛，且收敛结论可扩展到松弛版本和自适应重缩放变体。将该经验界与一致收敛论证相结合，我们还在附加条件下对对数损失给出了总体多校准误差的界。

    arXiv:2602.06773v3 Announce Type: replace  Abstract: Multicalibration gradient boosting has recently emerged as a scalable method that empirically produces approximately multicalibrated predictors and has been deployed at web scale. Despite this empirical success, its convergence properties are not well understood. In this paper, we provide computational guarantees for multicalibration gradient boosting algorithms. We show that the magnitude of successive prediction updates decays at $O(1/\sqrt{T})$ (in the minimum over $T$ rounds), which implies the same convergence rate bound for the empirical multicalibration error over rounds. For the squared-error loss, this rate improves to linear convergence under additional smoothness assumptions on the weak learners, and convergence extends to relaxed and adaptive rescaling variants. Combining the empirical bound with uniform convergence arguments, we also bound the population multicalibration error for the log-loss under additional conditions
    
[^96]: 基于方差缩减的经验回放用于策略优化

    Variance Reduction Based Experience Replay for Policy Optimization

    [https://arxiv.org/abs/2602.05379](https://arxiv.org/abs/2602.05379)

    本文提出方差缩减经验回放（VRER）框架，通过选择性重用信息样本降低策略梯度方差，并配套理论分析，提升策略优化样本效率。

    

    arXiv:2602.05379v2 公告类型：替换交叉 摘要：针对复杂随机系统的有效强化学习（RL）需要利用历史数据来提高样本效率并加速策略优化。然而，经典的经验回放将所有过去的观测均匀对待，未能考虑它们对学习的贡献差异。为解决这一局限性，我们提出了方差缩减经验回放（VRER），这是一个原则性框架，选择性地重用信息丰富的样本以减少策略梯度估计的方差。VRER是算法无关的，可以与现有策略优化方法集成，产生样本高效的离策略算法——带VRER的策略梯度（PG-VRER）。为提供严格的理论保证，我们开发了一个新的经验回放分析框架，该框架显式考虑了马尔可夫动力学和行为策略交互引起的依赖关系。利用该框架，我们建立了...

    arXiv:2602.05379v2 Announce Type: replace-cross  Abstract: Effective reinforcement learning (RL) for complex stochastic systems requires leveraging historical data to improve sample efficiency and accelerate policy optimization. However, classical experience replay treats all past observations uniformly and fails to account for their varying contributions to learning. To address this limitation, we propose Variance Reduction Experience Replay (VRER), a principled framework that selectively reuses informative samples to reduce the variance of policy gradient estimates. VRER is algorithm-agnostic and can be integrated with existing policy optimization methods, yielding the sample-efficient off-policy algorithm, Policy Gradient with VRER (PG-VRER). To provide rigorous theoretical guarantees, we develop a novel analysis framework for experience replay that explicitly accounts for dependencies induced by Markovian dynamics and behavior-policy interactions. Using this framework, we establish
    
[^97]: 物有所值：使用重置与丢弃（ReD）方法在固定预算下提升大语言模型的推理能力

    More Bang for the Buck: Improving the Inference of Large Language Models at a Fixed Budget using Reset and Discard (ReD)

    [https://arxiv.org/abs/2601.21522](https://arxiv.org/abs/2601.21522)

    本文提出重置与丢弃（ReD）查询方法，克服了pass@k幂律行为导致的边际收益递减问题，在固定预算下显著提升大语言模型回答独特问题的数量（coverage@cost），并能定量预测节省的尝试次数。

    

    大语言模型（LLM）在可验证任务上的性能通常用pass@k来衡量，即在k次尝试中至少一次正确回答问题的概率。在固定预算下，更合适的指标是coverage@cost，即作为总尝试次数函数的已正确回答的独特问题平均数量。我们建立了这两个指标之间的联系，并证明在经验上观察到的pass@k幂律行为会导致coverage@cost的次线性增长（即边际收益递减）。为了解决这个问题，我们提出了重置与丢弃（ReD），这是一种大语言模型查询方法，无论pass@k的具体形式如何，都能在给定预算下提高coverage@cost。此外，在给定pass@k的情况下，我们可以定量预测使用ReD所能节省的总尝试次数。如果模型的pass@k不可用，ReD可以推断其幂律指数。我们在三个大语言模型上进行了实验，涵盖编程（HumanEval）、数学（GSM8K）和推理（MMLU-Pro）任务。

    arXiv:2601.21522v3 Announce Type: replace-cross  Abstract: The performance of large language models (LLMs) on verifiable tasks is usually measured by pass@k, the probability of answering a question correctly at least once in k trials. At a fixed budget, a more suitable metric is coverage@cost, the average number of unique questions answered as a function of the total number of attempts. We connect the two metrics and show that the empirically-observed power-law behavior in pass@k leads to a sublinear growth of the coverage@cost (diminishing returns). To solve this problem, we propose Reset-and-Discard (ReD), a query method of LLMs that increases coverage@cost for a given budget, regardless of the pass@k form. Moreover, given a pass@k, we can quantitatively predict the savings in the total number of attempts using ReD. If pass@k is not available for the model, ReD can infer its power-law exponent. Experiments on three LLMs across coding (HumanEval), math (GSM8K), and reasoning (MMLU-Pro
    
[^98]: 具有因果与连续结构的情境化分布鲁棒优化

    Contextual Distributionally Robust Optimization with Causal and Continuous Structure

    [https://arxiv.org/abs/2601.11016](https://arxiv.org/abs/2601.11016)

    该论文提出了融合因果与连续结构的情境化分布鲁棒优化框架，通过引入因果 Sinkhorn 散度构建模糊集并给出强对偶形式，同时设计了可微、光滑且可解释的软回归森林决策规则来求解最优策略。

    

    我们提出了一个考虑底层分布的因果结构与连续结构的情境化分布鲁棒优化（DRO）框架，并开发了可解释且易处理的决策规则。我们首先引入因果 Sinkhorn 散度（CSD），这是一种熵正则化的因果 Wasserstein 距离，在保持因果一致性的同时鼓励连续的传输计划。随后，我们基于 CSD 构建了模糊集，建立了情境化 DRO 模型，称为因果 Sinkhorn DRO（Causal-SDRO），并推导出其强对偶重构形式，其中最坏情况分布被刻画为 Gibbs 分布的混合。为了获得（无限维的）最优策略，我们提出了软回归森林（SRF）决策规则：它既保留了经典决策树的可解释性，又是完全参数化、可微且 Lipschitz 光滑的，从而能够实现内在的解释性。

    arXiv:2601.11016v4 Announce Type: replace-cross  Abstract: WWe propose a framework for contextual distributionally robust optimization (DRO) that considers the causal and continuous structure of the underlying distribution, and we develop interpretable and tractable decision rules. We first introduce the causal Sinkhorn discrepancy (CSD), an entropy-regularized causal Wasserstein distance that encourages continuous transport plans while preserving causal consistency. We then formulate a contextual DRO model with a CSD-based ambiguity set, termed Causal Sinkhorn DRO (Causal-SDRO), and derive its strong dual reformulation, where the worst-case distribution is characterized as a mixture of Gibbs distributions. To obtain an (infinite-dimensional) optimal policy, we propose a soft regression forest (SRF) decision rule: it preserves the interpretability of classical decision trees while being fully parametric, differentiable, and Lipschitz-smooth, enabling intrinsic interpretation from both 
    
[^99]: 随机深度学习：一种用于建模结构化时序数据不确定性的概率框架

    Stochastic Deep Learning: A Probabilistic Framework for Modeling Uncertainty in Structured Temporal Data

    [https://arxiv.org/abs/2601.05227](https://arxiv.org/abs/2601.05227)

    提出SLDI框架，将伊藤随机微分方程嵌入变分自编码器的潜空间，通过神经网络参数化漂移与扩散项并对伴随状态进行共同参数化，实现对结构化时序数据不确定性的连续时间灵活建模与量化。

    

    我提出了一个新颖的框架，将随机微分方程（SDE）与深度生成模型相结合，以改进涉及结构化和时序数据的机器学习应用中的不确定性量化。这种方法被称为随机潜变量微分推断，它将伊藤随机微分方程嵌入到变分自编码器的潜空间中，从而在保持有原则的数学基础的同时，实现对不确定性的灵活的连续时间建模。SDE的漂移项和扩散项由神经网络进行参数化，实现了数据驱动的推断，并将经典时间序列模型推广到能够处理不规则采样和复杂动态结构的情形。一个核心的理论贡献是将伴随状态与专用神经网络进行共同参数化，形成一个耦合的前向-后向系统，不仅能捕捉潜变量的演化，还能捕捉梯度动态。我引入了一个

    arXiv:2601.05227v2 Announce Type: replace-cross  Abstract: I propose a novel framework that integrates stochastic differential equations (SDEs) with deep generative models to improve uncertainty quantification in machine learning applications involving structured and temporal data. This approach, termed Stochastic Latent Differential Inference (SLDI), embeds an It\^o SDE in the latent space of a variational autoencoder, allowing for flexible, continuous-time modeling of uncertainty while preserving a principled mathematical foundation. The drift and diffusion terms of the SDE are parameterized by neural networks, enabling data-driven inference and generalizing classical time series models to handle irregular sampling and complex dynamic structure.   A central theoretical contribution is the co-parameterization of the adjoint state with a dedicated neural network, forming a coupled forward-backward system that captures not only latent evolution but also gradient dynamics. I introduce a 
    
[^100]: 通过散度分解得到随机非凸优化的新下界

    New Lower Bounds for Stochastic Non-Convex Optimization through Divergence Decomposition

    [https://arxiv.org/abs/2502.14060](https://arxiv.org/abs/2502.14060)

    该论文通过将优化问题重新表述为函数识别问题并运用散度分解技术，为满足准凸性、二次增长和受限割线不等式等性质的非凸函数类的随机一阶优化建立了新的且几乎紧的带噪声梯度查询复杂度下界。

    

    我们研究了在一系列非凸情形下一阶随机优化的基本极限，这些情形包括满足准凸性（QC）、二次增长（QG）以及受限割线不等式（RSI）的L-光滑函数。尽管标准算法在确定性情形下的收敛性质已被充分理解，但针对随机情形（即仅能获得无偏且有噪声的梯度）的研究结果却少得多。我们对最小化这些函数类别所需的带噪声梯度查询次数建立了新的下界，并证明这些下界在刻画每个函数类别的所有相关参数上都是紧的（仅相差一个对数因子）。我们的方法将优化任务重新表述为一个函数识别问题，并利用散度分解的论证构造出一个具有挑战性的函数子类，从而得到精确的下界。此外，我们还提出了一种适用于一维情形的专用算法。

    arXiv:2502.14060v3 Announce Type: replace-cross  Abstract: We study fundamental limits of first-order stochastic optimization in a range of nonconvex settings, including L-smooth functions satisfying Quasar-Convexity (QC), Quadratic Growth (QG), and Restricted Secant Inequalities (RSI). While the convergence properties of standard algorithms are well-understood in deterministic regimes, significantly fewer results address the stochastic case, where only unbiased and noisy gradients are available. We establish new lower bounds on the number of noisy gradient queries to minimize these classes of functions, also showing that they are tight (up to a logarithmic factor) in all the relevant quantities characterizing each class. Our approach reformulates the optimization task as a function identification problem, leveraging divergence decomposition arguments to construct a challenging subclass that leads to sharp lower bounds. Furthermore, we present a specialized algorithm in the one-dimensi
    
[^101]: 面向多输出保形回归的基于广义一致性分数的统一比较研究

    A Unified Comparative Study with Generalized Conformity Scores for Multi-Output Conformal Regression

    [https://arxiv.org/abs/2501.10533](https://arxiv.org/abs/2501.10533)

    本文对多输出保形回归中的九种保形方法进行了统一比较研究，并提出两类由单变量方法推广而来的新型一致性分数，它们在保持精确有限样本边际覆盖率的同时实现渐近条件覆盖率。

    

    保形预测提供了一个强大的框架，用于构建具有有限样本覆盖率保证的无分布预测区域。尽管保形预测在单变量场景中已被广泛研究，但将其扩展到多输出问题会带来额外的挑战，包括复杂的输出依赖性和高昂的计算成本，目前仍相对缺乏探索。在这项工作中，我们在同一框架内对九种采用不同多变量基础模型的保形方法进行了统一的比较研究，以构建多变量预测区域。该研究突出了这些方法的关键特性，同时也探讨了它们之间的联系。此外，我们为多输出回归引入了两类新颖的一致性分数，它们是对单变量对应方法的推广。这些分数在保持精确的有限样本边际覆盖率的同时，确保了渐近条件覆盖率。其中一类与……

    arXiv:2501.10533v3 Announce Type: replace-cross  Abstract: Conformal prediction provides a powerful framework for constructing distribution-free prediction regions with finite-sample coverage guarantees. While extensively studied in univariate settings, its extension to multi-output problems presents additional challenges, including complex output dependencies and high computational costs, and remains relatively underexplored. In this work, we present a unified comparative study of nine conformal methods with different multivariate base models for constructing multivariate prediction regions within the same framework. This study highlights their key properties while also exploring the connections between them. Additionally, we introduce two novel classes of conformity scores for multi-output regression that generalize their univariate counterparts. These scores ensure asymptotic conditional coverage while maintaining exact finite-sample marginal coverage. One class is compatible with a
    
[^102]: 大维度情况下核回归的最优比率

    Optimal Rate of Kernel Regression in Large Dimensions. (arXiv:2309.04268v1 [stat.ML])

    [http://arxiv.org/abs/2309.04268](http://arxiv.org/abs/2309.04268)

    该论文提出了一种针对大维度数据的核回归的最优比率，通过使用Mendelson复杂性和度量熵来刻画其上界和最小化下界。此外，研究还发现最优比率随着维度与样本大小关系的变化呈现出多次下降的行为。

    

    我们对大维度数据（样本大小$n$与样本维度$d$的关系为多项式，即$n\asymp d^{\gamma}$，其中$\gamma>0$）的核回归进行了研究。我们首先通过Mendelson复杂性$\varepsilon_{n}^{2}$和度量熵$\bar{\varepsilon}_{n}^{2}$来建立一个通用工具，用于刻画大维度数据的核回归的上界和最小化下界。当目标函数属于与$\mathbb{S}^{d}$上定义的（一般）内积模型相关联的RKHS时，我们利用这个新工具来展示核回归的过量风险的最小化率是$n^{-1/2}$，当$n\asymp d^{\gamma}$，其中$\gamma=2, 4, 6, 8, \cdots$。然后我们进一步确定了对于所有$\gamma>0$，核回归过量风险的最优比率，并发现随着$\gamma$的变化，最优比率的曲线展现出几个新现象，包括多次下降行为。

    We perform a study on kernel regression for large-dimensional data (where the sample size $n$ is polynomially depending on the dimension $d$ of the samples, i.e., $n\asymp d^{\gamma}$ for some $\gamma >0$ ). We first build a general tool to characterize the upper bound and the minimax lower bound of kernel regression for large dimensional data through the Mendelson complexity $\varepsilon_{n}^{2}$ and the metric entropy $\bar{\varepsilon}_{n}^{2}$ respectively. When the target function falls into the RKHS associated with a (general) inner product model defined on $\mathbb{S}^{d}$, we utilize the new tool to show that the minimax rate of the excess risk of kernel regression is $n^{-1/2}$ when $n\asymp d^{\gamma}$ for $\gamma =2, 4, 6, 8, \cdots$. We then further determine the optimal rate of the excess risk of kernel regression for all the $\gamma>0$ and find that the curve of optimal rate varying along $\gamma$ exhibits several new phenomena including the {\it multiple descent behavior
    

