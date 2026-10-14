# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Joint Neural Posterior Estimation and Model Selection for Chemical Reaction Networks](https://arxiv.org/abs/2610.18825) | 本文提出了一种结合共享一维卷积神经网络与置换不变集合注意力机制的神经推断架构，可从多实验时间序列数据中同时完成化学反应网络的模型选择与参数后验估计，并在真实的去硼酸化反应示例中验证了其实用性。 |
| [^2] | [Designing for Prediction: Planning Formulas for Expected Out-of-Sample $R^2$ and $r$](https://arxiv.org/abs/2610.18781) | 本文为留一交叉验证下的OLS和岭回归预测模型推导了将预期样本外 $R^2$ 及观测-预测相关系数与样本量、预测变量数、假设总体 $R^2$ 和设计效率因子联系起来的闭式规划公式，使研究者能在数据收集前评估预测性能，且这些公式相比经典收缩公式还对协变量的不平衡和偏斜敏感。 |
| [^3] | [A Rate Transfer Method: Convergence Rates in Stochastic Gradient Descent via Stochastic Calculus](https://arxiv.org/abs/2610.18739) | 提出一种速率转移方法，将随机微分方程的有限时间Lyapunov估计转化为随机梯度算法的有限时间收敛界，使得新优化算法的收敛速率可通过简单的微积分论证来建立。 |
| [^4] | [From Fourier Amplitudes to Persistence Images for Impact-Echo Defect Detection](https://arxiv.org/abs/2610.18732) | 该论文提出从傅里叶振幅谱构建二维频率-振幅持久性图像（FA-PI）的新方法，利用多参数持续同调将频率与振幅信息编码，有效应对边界反射、干扰和噪声导致的复杂频谱问题，提升了冲击回波缺陷检测的分类性能。 |
| [^5] | [BridgeTwist: Twisting Schr\"odinger Bridges for Training-Free Conditional Sampling](https://arxiv.org/abs/2610.18658) | 提出了BridgeTwist，一种基于带温度调节扭转序贯蒙特卡洛方案的免训练条件采样器，可复用预训练薛定谔桥生成模型，利用由速度场与得分场构造的闭式即插即用扭转代理实现条件后验采样。 |
| [^6] | [Minimax rates for pointwise CATE estimation: the role of design smoothness](https://arxiv.org/abs/2610.18623) | 该论文揭示了协变量设计平滑性对逐点条件平均处理效应极小极大估计速率的关键影响，证明了粗糙的协变量设计即使在处理效应与冗余函数具有相同平滑性假设的条件下，也会导致比均匀设计更慢的收敛速率。 |
| [^7] | [Understanding Circulant Permutation to Extend the CMinHash Estimator](https://arxiv.org/abs/2610.18551) | 本文提出了一种新的近似方差分析方法，成功将 CMinHash 的循环置换技术扩展到 Minner 等其他 MinHash 估计器，并通过数值实验验证了方差表达式与经验均方误差的一致性。 |
| [^8] | [Two For The Price Of One: Curvature-Aware Pruning Preserves Influence Functions Too](https://arxiv.org/abs/2610.18516) | 提出了基于ADMM的曲率感知剪枝优化器CACTUS，能够在剪枝和稀疏加低秩分解中保留影响函数所依赖的梯度信息，相比LLM-Surgeon速度提升超过一个数量级，并显著缩小稀疏模型与稠密模型之间的困惑度差距。 |
| [^9] | [How Low Can You Go: Fine-Tuning LLMs with Bayesian Optimization (LlMBO)](https://arxiv.org/abs/2610.18504) | 本文提出LlMBO方法，利用低维微调场景下全局优化变得可行这一观察，将贝叶斯优化应用于大语言模型微调，实现了样本高效、无梯度、轻内存且可直接优化贪心解码准确率的微调方案。 |
| [^10] | [Recalibrating Many Questions with Few Labels: Minimax Rates and Identification Limits](https://arxiv.org/abs/2610.18488) | 本文针对每个问题仅有少量标签的多个相关预测问题，首次刻画了再校准的极小极大最优复合风险率与可辨识性极限，并给出了能共享信息、达到最优速率的合并等张回归估计方法。 |
| [^11] | [Convergence-Rate-Guided Algorithm Design of Expected Improvement for Continuous Simulation Optimization](https://arxiv.org/abs/2610.18434) | 该论文通过推导简单遗憾的高概率上界，在收敛率分析的基础上系统性地设计了随机克里金期望改进算法的三个关键组件——当前最优解、采样重复次数和终端推荐，解决了未知异方差噪声下连续仿真优化中依赖启发式规则进行算法设计的局限。 |
| [^12] | [On the Optimality of Random Features under General Conditions: Misspecified Cases and Robust Learning](https://arxiv.org/abs/2610.18333) | 本文在模型误设与重尾噪声等更一般的条件下对带随机特征的核岭回归进行了全面分析，改进了特征映射的一致界并证明了极小化极大最优学习率。 |
| [^13] | [Operator Surrogates](https://arxiv.org/abs/2610.18210) | 本章对算子代理模型进行了数学导论，系统介绍了编码器-逼近器-解码器的构建范式与高效表示方法，并对比了基于多项式混沌展开的谱方法与神经网络等主要代理架构。 |
| [^14] | [Hierarchical Time Series Forecasting with Operator-Valued Kernels](https://arxiv.org/abs/2610.18196) | 该论文提出一个基于结构化算子值核的统一框架，将层次时间序列预测转化为向量值再生核希尔伯特空间中的多元回归问题，使预测在构造上即满足层次一致性约束，无需事后协调步骤，并给出了泛化性能的过剩风险理论保证。 |
| [^15] | [Lifting Finite-Sample Gaussian Approximation of Wide Neural Networks to Functional Convergence](https://arxiv.org/abs/2610.18166) | 本文提出将有限样本高斯逼近界提升为泛函收敛界的方法，使宽神经网络向高斯过程的 $\mathcal W_2$ 收敛界摆脱了对样本数量的依赖，从而适用于任意定义域上的稠密样本加密极限情形。 |
| [^16] | [Learning a Group Element from Its Action: Galois Correspondences, Optimal ERM, and an Abel-Ruffini Threshold for the Natarajan Dimension](https://arxiv.org/abs/2610.18154) | 该论文建立了从样本学习群元素与伽罗瓦理论之间的对应关系，将多类学习的 DS 维度、Littlestone 维度和教学维度化为群不变量，并证明在对称类上任何一致的 ERM 学习者均达到最优样本复杂度，同时揭示了 Natarajan 维数存在阿贝尔-鲁菲尼式的阈值。 |
| [^17] | [Reliable Scale-Free Bandits via Empirical Likelihood](https://arxiv.org/abs/2610.18055) | 该论文提出了 g-Forcing 策略的经验似然扩展 EL-F，在无需预知奖励尺度或尾部参数的情况下，为随机老虎机同时提供了几乎必然的渐近伪遗憾保证和有限时间的期望遗憾上界。 |
| [^18] | [ndscape: a structural map of the nested dichotomy space](https://arxiv.org/abs/2610.18050) | 提出了ndscape——一种基于深度加权结构标记的嵌套二分法结构二维映射方法，并首次对ND结构空间进行了偏差-方差分析，发现当节点使用逻辑回归时，更平坦、更平衡的树结构方差更小、更稳定，而使用决策树节点时方差与结构深度无关。 |
| [^19] | [Weighting-Based Strategies for Classification: Analysis and Local Peak Dynamics with CART Trees](https://arxiv.org/abs/2610.18037) | 该论文从理论和实验上证明，在类别不平衡分类任务中，加权策略与不加权策略的预测性能本质上相同——渐近情况下两者仅相差一个阈值偏移，非渐近情况下二者在PR空间中沿同一曲线分布。 |
| [^20] | [Counterfactual Certification for Sequential Unlearning with Long-Run Performance Guarantees](https://arxiv.org/abs/2610.17993) | 该论文提出一种针对序列式机器遗忘请求的反事实认证方法，在每次删除仅使用固定微调预算的情况下，对于光滑强凸目标函数可将性能损失约束在与请求次数无关的有界范围内。 |
| [^21] | [Learning Distributions from Samples and Qualities](https://arxiv.org/abs/2610.17956) | 该论文证明，当分布样本附带由同一未知参数决定的二元质量标签时，硬可行性约束可将参数估计从经典 $n^{-1/2}$ 速率加速至 $n^{-1}$，即使在质量模型存在未知加性常数时（通过构造无需阈值的成对偏好）也能保持这一加速，并揭示了逐点质量与成对偏好在信息量上互不可比。 |
| [^22] | [Follow the Perturbed Leader for Linear Bandits: $\sqrt{T}$ Regret over General Convex Sets](https://arxiv.org/abs/2610.17818) | 本文提出了首个基于FTPL的对抗性线性赌博机算法，通过一个适应估计奖励的扰动分布，在一般凸动作集上实现了 $\sqrt{T}$ 遗憾，改进了此前 $T^{2/3}$ 的遗憾界，同时比连续指数权重方法具有更好的计算效率和更简单的实现。 |
| [^23] | [Linear Regression Beyond Covariance: Latent Slope Spectra from Higher-Order Cumulants](https://arxiv.org/abs/2610.17772) | 这篇论文提出"潜在斜率谱"的新框架，证明普通最小二乘估计只是斜率谱的方差加权重心，并利用高阶累积量在谱足够简约时分离出被传统协方差信息掩盖的不同线性效应成分，实现多重斜率的识别与根号n推断。 |
| [^24] | [Gaussian Process Bandits for AI System Optimization with Selectively Verified Feedback](https://arxiv.org/abs/2610.17760) | 该论文针对AI系统优化中模型评估与可信结果不一致的问题，提出了带选择性验证反馈的高斯过程老虎机方法，通过固定审计设计与安全自适应扩展，在硬性审计预算下建立了有限时间遗憾界。 |
| [^25] | [Structured SDEs for Realized GARCH: Continuous-Time Volatility Modelling and Component-Level Analysis](https://arxiv.org/abs/2610.17732) | 该论文提出一种结构化连续时间随机波动率模型，通过已实现测度状态直接反馈到潜在方差动态中来预测积分方差，在17个股票指数上展现出优于GARCH、Realized GARCH等基准模型的样本外预测性能。 |
| [^26] | [textGrain watermarking is harder to detect: can sum-tests still do it optimally?](https://arxiv.org/abs/2610.17655) | textGrain 水印的熵校准机制使其极小极大检测边界为 p+q=1/2，可检测区域严格小于边界为 2p+q=1 的 Gumbel-max 与最大耦合水印，表明熵校准是以牺牲可检测性为代价的。 |
| [^27] | [Risk-Averse Inference-Time Diffusion Model Alignment with Finite-Sample Guarantees](https://arxiv.org/abs/2610.17579) | 该论文提出了一个风险规避的扩散模型推理时对齐框架，在轨迹级KL正则化下平衡期望代价与条件风险价值（CVaR），将对齐问题化简为关于CVaR阈值的一维优化，并给出了阈值优化的有限样本理论保证。 |
| [^28] | [Effect of Spherical Parameter Space on Number of Non-Forgetting Tasks and Capacity of Continual Learning](https://arxiv.org/abs/2610.17453) | 该论文提出了一种基于高维球面参数空间的持续学习理论框架，通过测地球刻画任务的可接受损失区域，并利用 Neyman-Scott 聚类点过程与 von Mises-Fisher 分布建模任务相似性，从而分析球面参数空间对可容纳的非遗忘任务数量及持续学习容量的影响。 |
| [^29] | [Learning to Detect Shifts in Unstructured Data Streams via Monitoring-by-Betting](https://arxiv.org/abs/2610.17399) | 提出一种基于投注式检验的监控框架，通过将序贯变点检测建模为投注博弈并自适应更新偏移方向假设，解决了非结构化数据流分布偏移检测中敏感性与广度的权衡问题，同时保持误报率的统计保证。 |
| [^30] | [Measure Neural Mappings](https://arxiv.org/abs/2610.17384) | 本文提出测度神经映射（MNMs）新框架，通过集成一致性将依赖测度的神经传输与自注意力机制结合，证明了Transformer在非紧概率测度上的定量连续统极限，其输出Wasserstein误差由经验输入误差加上 $N^{-1/2}$ 注意力项界定，可应用于贝叶斯推断等无界状态空间场景。 |
| [^31] | [On the Statistical Mechanics of Attention](https://arxiv.org/abs/2610.17347) | 本文借助现代Hopfield模型的自由能形式化方法，建立了刻画Transformer注意力相结构与推理可靠性的严格统计力学框架，并揭示出注意力在玻璃态与稳健推理相之间存在一阶相变。 |
| [^32] | [Monitoring for Shifts at Unknown Times: Parameter-Free Conditional Conformal Martingales](https://arxiv.org/abs/2610.17343) | 该论文提出一种无超参数的条件保形鞅方法，通过将惩罚下注问题转化为三资产市场的投资组合选择问题，并采用带重尾财富分配的重启通用投资组合，实现了对未知时间发生的分布偏移的快速自适应检测，同时保证任意时刻虚警控制和最优的有限样本检测延迟界。 |
| [^33] | [Accounting for model error in ensemble variational data assimilation through learned generative dynamics](https://arxiv.org/abs/2610.17333) | 提出生成式迭代集合卡尔曼滤波器（IEnKF-G），用学习到的条件随机生成预报模型取代传统设定的加性高斯模式误差表示，在新颖的单步弱约束4D-Var框架中更好地刻画复杂模式误差。 |
| [^34] | [Near-Optimal Regret Bounds for Monotone Stochastic Optimization under Semi-Bandit Feedback](https://arxiv.org/abs/2610.17320) | 本文提出一种通用的、预言机高效的在线到离线归约，使强单调随机优化在半臂反馈下的期望后悔界达到近最优的 $\widetilde O(\sqrt{nT})$，从而回答了该领域的一个开放性问题。 |
| [^35] | [Deeply unrolled alternating projections for phase retrieval](https://arxiv.org/abs/2610.17316) | 提出了一种将学习到的去噪步骤与交替投影方法相结合的深度展开框架用于相位恢复，在测量算子动态范围显著变化等场景下无需重新训练即可获得高质量重建。 |
| [^36] | [The Surprising Expressivity of a Single Neuron with Autoregressive Chain-of-Thought](https://arxiv.org/abs/2610.17283) | 该论文证明一个使用阈值或ReLU等激活函数的单个自回归神经元，即可通过其生成的思维链模拟任意前馈电路乃至多项式时间图灵机，所需维度为O(ns+s²)、生成步数为O(s²)，且两个界均与电路深度无关并在此前指数级结果的基础上有显著改进。 |
| [^37] | [Tight End-to-End Accounting for Differentially Private Synthetic Data](https://arxiv.org/abs/2610.17249) | 本文开发了基于隐私损失分布（PLD）、f-DP和高斯差分隐私（GDP）的紧致端到端隐私核算工具，并提出了GDP原生的AIM变体AIM-GDP，实现了差分隐私合成数据流水线中异构机制的精确组合分析。 |
| [^38] | [Online LLM Personalization from Pairwise Preferences Across Users](https://arxiv.org/abs/2610.17203) | 该论文将基于成对偏好反馈的大语言模型推理时个性化形式化为上下文决斗老虎机问题的变体，提出一种无需初始偏好数据、可在会话内自适应并利用跨用户共享信息的在线选择方法，以最小化累积强遗憾。 |
| [^39] | [A Generalized McDiarmid Inequality: Sample Reuse and Non-Uniform Unbounded Differences](https://arxiv.org/abs/2610.17159) | 本文提出了McDiarmid不等式的一种新颖广义形式，能够同时刻画训练样本复用的收益并处理无界且非均匀依赖的差分函数，并将其应用于基于分数的生成模型以获得更紧的误差界。 |
| [^40] | [SCOPE Before You Sample: Conformal Scores Guide Posterior Exploration for Fast and Calibrated Bayesian Predictive Inference](https://arxiv.org/abs/2610.17146) | 提出SCOPE方法，利用共形非一致性分数的梯度识别影响预测区域边界的关键参数方向，引导MCMC在这一低维子空间中探索，从而以远低于完整MCMC的成本获得保留MCMC预测质量且校准良好的贝叶斯预测推断。 |
| [^41] | [Can R\'enyi Accountants Beat Lossless Accountants for DP-SGD?](https://arxiv.org/abs/2610.17120) | 该论文证明，在替换邻接、固定批次大小的DP-SGD设置下，通过将最坏情形分布刻画为二维高斯混合，基于Rényi DP的隐私会计可以获得比无损隐私框架更紧的隐私保证。 |
| [^42] | [In-context classification: Calibration and noise in linear attention with softmax readout](https://arxiv.org/abs/2610.17061) | 该论文针对线性注意力模型的上下文二分类任务推导出测试逻辑值的渐近解析公式，将校准误差、上下文噪声与参数估计噪声分离为可解释的独立效应，并系统分析了上下文长度、预训练规模、正则化和采样温度的影响，理论预测与仿真及真实训练模型高度吻合。 |
| [^43] | [Measuring Variable Importance Using Proportional Inclusive Marginal Effects](https://arxiv.org/abs/2610.17053) | 提出了PRIME（比例包含边际效应）这一新的变量重要性度量方法，通过无模型变量选择技术仅纳入活跃变量进行计算，克服了PME仅适用于确定性模型且计算成本高的局限，能从含噪声数据中准确可靠地量化变量重要性。 |
| [^44] | [CLASSO: a practical primal path-following algorithm for the constrained LASSO](https://arxiv.org/abs/2610.17024) | 提出了CLASSO算法，一种求解带一般线性等式与不等式约束的LASSO的实用原始路径跟踪方法，无需约束规格条件即可证明解路径为分段仿射且折点有限，并解决了折点处重新认证可能不终止的问题。 |
| [^45] | [ProbabilityBench: Model Evaluation and Calibration with Exact Ground-Truth Uncertainty](https://arxiv.org/abs/2610.17016) | ProbabilityBench通过显式定义数据生成过程，在三个模拟文本领域中提供精确的实例级真值类别概率，实现了对模型概率预测质量的直接评估，并发现成本-质量帕累托前沿上的模型多为决策模型。 |
| [^46] | [Minimax Optimality of Shampoo for Nonstationary Stochastic Optimization](https://arxiv.org/abs/2610.16923) | 该论文对矩阵Shampoo在非平稳随机优化中追踪时变最优解的性能进行了全面理论分析，推导出分离初始化、漂移、噪声与自适应预条件化影响的有限时域高概率追踪误差界，揭示了动量参数β₁的漂移—噪声权衡与稳定化参数ε的稳定性—几何自适应性权衡，并证明了其极小极大最优性。 |
| [^47] | [Radial Tail Repair: Post-Training Correction for Diffusion and Flow Models](https://arxiv.org/abs/2610.16851) | 提出一种轻量级后训练校正方法，基于多元重尾的径向-角度分解，在保持冻结的扩散或流匹配模型方向结构不变的情况下，仅通过重缩放生成样本的径向尾部来修正模型对重尾分布极端区域的表示。 |
| [^48] | [Feature Normalization under Finite Training Budgets](https://arxiv.org/abs/2610.16845) | 该论文在有限训练预算下理论刻画了标准化与未缩放训练各自可达到的最低稀有子群风险，并据此揭示了噪声水平与训练预算如何决定特征归一化策略的优劣。 |
| [^49] | [Sample-Size Extrapolation in Simulation-Based Inference via Statistical Scaling](https://arxiv.org/abs/2610.16820) | 提出了CE-NPE方法，通过将统计中心估计与显式收缩率相结合，使神经后验估计器能够在无需可评估似然函数的情况下外推到更大样本量，在所有超出训练范围的样本量上均取得最佳的后验推断精度。 |
| [^50] | [On the Calibration of Crowdsourcing Aggregators](https://arxiv.org/abs/2610.16815) | 该论文首次对众包标签聚合的校准问题进行了理论与实证研究，发现多数投票过于保守而基于模型的聚合器过于自信，并证明贝叶斯聚合器仅在其自身模型假设下才是完美校准的。 |
| [^51] | [When Repeated Measurements Hurt Tabular Foundation Models](https://arxiv.org/abs/2610.16801) | 该研究发现表格基础模型在面对相同输入的重复测量时，会因PFN预训练的IID规则与实际协议不匹配以及注意力机制对行数的依赖，导致未测量输入处的预测性能下降。 |
| [^52] | [One-Layer Transformers Provably Learn In-Context K-Nearest Neighbor Prediction with Chain-of-Thought](https://arxiv.org/abs/2610.16798) | 本文从理论上证明单层Transformer可通过思维链训练可证明地学会上下文中的K近邻预测，突破了softmax注意力难以直接关注多个近邻的固有局限。 |
| [^53] | [Amortised adaptive importance sampling](https://arxiv.org/abs/2610.16736) | 该论文提出了一种摊销式自适应重要性采样算法，通过拟合轻量级神经网络利用采样过程中的信息迭代调整提议分布，从而在训练完成后以更少的计算量生成高质量样本。 |
| [^54] | [Averaged Annealed Markov Chains Achieve the Algorithmic Threshold for Tensor PCA, Part I: Glauber Dynamics](https://arxiv.org/abs/2610.16705) | 通过结合退火与遍历平均的Glauber动力学，局部马尔可夫链能够从随机初始化出发，在算法阈值之上对张量PCA实现多项式时间强恢复，从而颠覆了局部算法劣于全局算法的传统认知。 |
| [^55] | [Amortized Variational Inference of Latent Intensities from Replicated Point Patterns](https://arxiv.org/abs/2610.16669) | 提出了一种基于神经网络的摊销变分推断框架，无需真实强度监督即可从重复点模式中学习可复用的推断规则，通过一次前向传播完成潜在强度分布的贝叶斯推断，相比对数高斯Cox过程等方法速度大幅提升且推断精度更高。 |
| [^56] | [Quickest Change-point Detection in Generalized Linear Models with Variational Estimation](https://arxiv.org/abs/2610.16616) | 该论文提出了一种基于变分不等式估计的窗口受限CUSUM方法，用于变化后参数未知的广义线性模型最快变点检测，并刻画了估计误差对检测延迟的影响且给出了有限窗口下的性能保证。 |
| [^57] | [Variational Filtering for Bayesian Last Layers with an Independent Binary Probit Surrogate](https://arxiv.org/abs/2610.16578) | 提出了一种将深度分类器末层参数视为线性高斯状态空间模型隐状态的变分滤波算法，借助独立二元Probit替代似然实现闭式共轭更新，可在特征漂移下稳定训练并同时适用于线性探测与全网络微调。 |
| [^58] | [$\texttt{NFDiag}$: High-dimensional MCMC Convergence Diagnosis with Normalizing Flows](https://arxiv.org/abs/2610.16574) | 该论文提出NFDiag，一种利用归一化流在完整多变量联合后验空间中分析链的几何结构与条件分布信息的高维MCMC收敛性诊断新方法，克服了传统基于边际分布的诊断在高维下不敏感的问题。 |
| [^59] | [Learning Graded Sparse Mechanisms: A Representation Theorem and Certified Generalization](https://arxiv.org/abs/2610.16537) | 本文将“分级稀疏机制”公理化定义为一类函数，并证明了一个表示定理：该类函数恰好是加权多重集上某个传输度量的Lipschitz球，从而为这类机制的学习提供了可认证的泛化保证。 |
| [^60] | [Asymptotic Risk of Learning-Rate Schedules for Least-Squares Regression with Discretely Distributed Features](https://arxiv.org/abs/2610.16518) | 在离散分布特征的流式最小二乘回归中，任何固定相对长度的学习率冷却阶段都能以多项式因子改进风险，且最优风险仅通过乘积bT依赖于批量大小和时间范围。 |
| [^61] | [Locally adaptive step sizes for Hamiltonian Monte Carlo methods](https://arxiv.org/abs/2610.16506) | 本文在更一般的假设下阐释了HMC采样中的精确可逆局部自适应步长方法，提出了减少可逆性失败拒绝的替代误差度量，并将该技术扩展应用到标准HMC以及Metropolis校正的动力学Langevin/广义HMC（GHMC）方法中。 |
| [^62] | [Online Individualized Treatment Selection under Partial Feedback: False Discovery Control and Minimax Optimality](https://arxiv.org/abs/2610.16464) | 本文提出PAITS方法，在仅观察到所分配治疗结果的部分反馈场景下进行在线个体化治疗选择，同时实现了渐近错误发现率控制、相对于总体预言机的次线性分配遗憾，并达到了极小极大最优性。 |
| [^63] | [Explaining Heterogeneous Treatment Effects with Causal Forests and the Hoeffding Functional Decomposition](https://arxiv.org/abs/2610.16447) | 本文提出CausalTreeHFD，一种基于Hoeffding函数分解的可解释因果森林扩展方法，能够将CATE预测分解为单个协变量和成对交互作用的贡献，直接揭示驱动处理效应异质性的因素，并证明了该方法收敛于真实CATE函数Hoeffding分解的一致性。 |
| [^64] | [Moment-Based Anomaly Detection in High-Dimensional Mixed Data](https://arxiv.org/abs/2610.16285) | 该论文提出了一种基于矩变换和依赖感知聚合的MDI异常检测方法，能够检测高维混合数据中由位置偏移、变异、偏度等分布特性变化引起的稀疏异常及跨特征协调性偏差。 |
| [^65] | [When Sample-Splitting Choices Matter in TMLE and AIPW: Cross-Fitting, Targeting, and Repetition](https://arxiv.org/abs/2610.16192) | 该研究通过已知真值模拟系统比较了八种TMLE和AIPW估计器变体，发现对两个干扰模型均进行交叉拟合是实现有效置信区间覆盖率的关键（样本内拟合倾向得分会导致严重覆盖不足），且在近正性违规条件下AIPW估计器的覆盖率表现一致优于TMLE估计器。 |
| [^66] | [Exploration strategies for Gaussian process bandit optimization with inference regret guarantees](https://arxiv.org/abs/2610.16187) | 本文提出了一种通过偶尔将采集查询替换为探索查询即可融入任意贝叶斯优化流程的探索策略，并借助填充距离为高斯过程后验标准差和推断遗憾推导出理论上界，从而保证算法最终推荐结果的质量。 |
| [^67] | [Equilibrium Learning in Unbounded Stochastic Games with Gaussian Transitions](https://arxiv.org/abs/2610.16149) | 该论文针对状态空间无界、转移噪声为高斯的随机博弈，提出利用单条连续轨迹更新高斯过程漂移模型并求解逐步精化的有限博弈来学习均衡，并证明了均衡选择稳定性与一致KL精度足以保证诱导值和近似纳什行为的区域一致收敛。 |
| [^68] | [Distributional Regression for Arbitrary Response Variables Using ReLU Neural Networks](https://arxiv.org/abs/2610.16142) | 本文提出了一个基于可测集条件事件概率的统一分布回归框架，通过有向无环图约束的估计器和ReLU网络实现了对任意响应变量的分布回归，并建立了克服维度灾难的误差界与收敛速率。 |
| [^69] | [Backward Conformal Prediction under Covariate Shift](https://arxiv.org/abs/2610.16048) | 该论文通过重要性加权将反向保形预测（BCP）扩展到协变量偏移情形，并提出在排序后的非一致性分数向量而非高维输入上估计密度比，从而在保持理论保证的同时显著改进了失覆盖率的估计效果。 |
| [^70] | [A Nonlinear Time-Series Theory for Constant-step SGD with Momentum](https://arxiv.org/abs/2610.16034) | 该论文首次将恒定步长带动量的SGDM视为非线性时间序列，证明了参数-动量联合过程的几何遗忘性与唯一平稳因果解的存在性，并由此导出了平均化SGDM的不变性原理、淬火高斯极限以及小学习率下最后迭代的高斯极限。 |
| [^71] | [Limit Theory for Online Expectile Estimation under Short and Long Range Dependence](https://arxiv.org/abs/2610.16030) | 本文建立了恒定步长随机梯度算法在线估计期望分位数的极限理论：短程依赖下迭代值以 √γ 尺度渐近高斯，长记忆下尺度变为 γ^(β-1/2) 且极限由分数布朗运动积分给出，并推广至多水平联合极限。 |
| [^72] | [Physics-guided machine learning for saturation control across variable hydraulic conditions in heap leaching](https://arxiv.org/abs/2610.15947) | 该研究开发了一种参数条件化的物理引导机器学习代理模型，可基于不同水力参数快速预测堆浸饱和度演化，实现了与调谐PI控制器精度相当且无超调的饱和度预测控制。 |
| [^73] | [Minimax Rates and Improved Spectral Convergence for Multi-Manifold Clustering](https://arxiv.org/abs/2610.15892) | 本文为张量化拉普拉斯-贝尔特拉米算子的特征对估计建立了极小极大最优速率 $n^{-s/(d+2s)}$，并证明流形无关的图拉普拉斯估计器以更快的速率 $n^{-1/(2.5d-1)}$ 收敛，从而改进了多流形学习的谱收敛理论。 |
| [^74] | [What Do Sparse Autoencoder Feature Discovery Scores Identify? From Activations to Interventions](https://arxiv.org/abs/2610.15850) | 论文证明基于激活共变的稀疏自编码器特征发现证据通常无法预测干预（如转向或消融）的真实效果，并提出基于梯度的干预归因方法，其对消融效应符号的预测准确率达91-97%，远优于线性探针预测的17-35%。 |
| [^75] | [ML-CLAPPS: A Probabilistic Framework for Imbalanced Multilabel Learning under Class Prioritization](https://arxiv.org/abs/2610.15816) | 提出了ML-CLAPPS，一种可学习的概率采样框架，通过优化训练实例的分布来应对多标签类别不平衡问题，并借助用户定义的重要性向量实现对高风险病症（如心电图分类中的关键病症）的优先处理。 |
| [^76] | [Inference on the Value of Policy Improvement in Adaptive Experiments](https://arxiv.org/abs/2610.15789) | 本文提出了自适应实验数据下策略改进价值（VPI）的渐近正态估计量，通过在历史数据上拟合加权经验福利最大化策略并用逆倾向加权进行评估，克服了数据相关性、倾向得分衰减以及最优策略非唯一时的非正则性等难题，为判断是否值得部署个性化处理策略提供了统计推断工具。 |
| [^77] | [Adaptive Robust Confidence Sets in High Dimensions](https://arxiv.org/abs/2610.15723) | 该论文证明了在污染比例未知的高维Huber污染模型下，通过同时量化所有一维投影的不确定性可构造信息论最优的自适应置信集，但该构造不仅存在显著的自适应统计代价，还因统计查询下界而面临计算不可行的障碍。 |
| [^78] | [Notes on Uniform Policy Score Bounds and Lipschitz Continuity for Truncated Gaussian, Product Cauchy, Boltzmann, and Softmax Policies](https://arxiv.org/abs/2610.15690) | 本文在明确参数化条件下，证明了截断高斯、柯西、玻尔兹曼和softmax策略的一致得分界与Lipschitz连续性。 |
| [^79] | [When do explicit penalties improve flow matching?](https://arxiv.org/abs/2610.15689) | 该论文以期望平方 2-Wasserstein 误差为度量，在希尔伯特空间框架下推导出几何准则并显式求解统计比较，证明了流匹配中的显式惩罚（如物理约束）恰好在其带来的方差减少超过平方偏差增加时才能提升生成精度。 |
| [^80] | [Screening Collapse under Recursive Synthetic Data](https://arxiv.org/abs/2610.15617) | 该论文首次揭示递归合成数据训练中的“筛选崩溃”现象：伪标签目标逐代漂移会侵蚀或逆转活跃与非活跃变量的分数差距，导致真实标签筛选本应保留的重要变量被丢弃，且这种协方差加权损失无法通过后续重新拟合消除，从而先于并加剧传统的模型崩溃。 |
| [^81] | [Joint Contrastive-Adversarial Representation Learning for Unsupervised Domain Adaptation](https://arxiv.org/abs/2610.15613) | 提出JCAR两阶段框架，将对比学习与对抗域对齐相结合以学习源域和目标域的共同表示，并建立了非渐近超额风险界，证明当无标签样本足够多时，下游预测可达到 $\widetilde{\mathcal{O}}(m^{-\beta/(2\beta+d)})$ 的收敛速率。 |
| [^82] | [Latent-Space Importance Weighting under Covariate Shift](https://arxiv.org/abs/2610.15571) | 该论文提出一种针对协变量偏移的迁移学习方法，在低维潜在空间中将表示学习与样本重加权相结合，并给出了目标域超额风险率等理论保证与实验验证。 |
| [^83] | [Deep Neural Mixed Models for Nonlinear Longitudinal Prediction with Subject-Specific Heterogeneity](https://arxiv.org/abs/2610.15323) | 提出深度广义神经网络混合模型（DGNMM），通过将深度神经网络与可训练的正则化个体特异效应相结合来扩展GLMM框架，从而在存在个体异质性的纵向健康数据中提升非线性预测能力。 |
| [^84] | [Accuracy and Uncertainty in Machine Deception Detection (Research White Paper)](https://arxiv.org/abs/2610.15314) | 本白皮书提出机器欺骗检测器，通过读取语言模型全部层、校准警报阈值以控制假阳性率，并利用集成检测器量化不确定性，从而在保证检测准确性的同时使每个警报都具有可量化的置信度。 |
| [^85] | [What a Data Release Must Preserve for Off-Policy Evaluation](https://arxiv.org/abs/2610.15283) | 本文刻画了策略外评估中数据发布必须保留的关键信息——识别性当且仅当动作特定评估权重在每个发布记录内恒定时成立——并据此提出以权重半径为目标的发布设计方法与有限数据修复算法。 |
| [^86] | [Imperfect Verifiers, Exact Targets: Inference-Time Scaling for Continuous Generation](https://arxiv.org/abs/2610.15281) | 提出 CS-VGB 方法，通过正确配对的去噪与再加噪转移，使连续生成模型在验证器不完美的情况下仍能精确收敛到奖励倾斜的目标分布，且收敛速度为指数级、更新开销仅线性增长。 |
| [^87] | [Data-Topology Compatibility Enables Personalization in Decentralized Learning](https://arxiv.org/abs/2610.15126) | 该论文提出数据-拓扑兼容性理论，证明当局部目标与通信拓扑具有近似兼容的自同构时，同一轨道内的智能体无需显式个性化即可自动收敛到相同的个性化分布，从而在全局共识与独立学习之间建立了新的中间机制。 |
| [^88] | [Energy-Guided Learning for Heat Equation with Noisy Source: Optimal Estimation and Generalization Bounds](https://arxiv.org/abs/2610.15111) | 本文提出一种能量引导的学习方法，通过新颖的变分形式以及能量正则化、椭圆性去偏的Sobolev回归，利用源函数的辅助数据，对从有限随机采样数据中学习带噪声源项热方程的解实现了最优估计并给出了泛化界。 |
| [^89] | [Sparsity Saves Compute, Not Samples: Oracle Inequalities for Dense and Sparse Mixture of Experts](https://arxiv.org/abs/2610.15039) | 该论文提出“门控希尔伯特空间”框架，为稠密及稀疏Top-k/Top-p混合专家模型建立了快速速率的oracle不等式，揭示了稀疏路由节省的是计算量而非统计样本量。 |
| [^90] | [Characterizing the Separability Gap in Prior-Data Fitted Networks](https://arxiv.org/abs/2610.14989) | 本文证明常见的PFN先验即使在简单设定下也是不可分的，刻画了由此产生的预测分布与标准理论之间的可分性差距，并揭示在模型误设时不可分先验会偏向更好的特征拟合而牺牲条件标签拟合。 |
| [^91] | [Mixing Regularization for Average-Reward $Q$-Learning and $Q$-Planning](https://arxiv.org/abs/2610.14988) | 该论文提出通过向一个与奖励无关的 $m$ 步混合策略进行正则化来控制平均奖励 $Q$-学习的收缩与探索性质，并给出有限时间收敛保证，在单步情形下匹配最佳已知速率，在 $m$ 步混合情形下首次推导出新的速率。 |
| [^92] | [Predicting how model propensities generalize out of distribution](https://arxiv.org/abs/2610.14937) | 该研究通过系统提示、少样本学习、SFT和GRPO等引导技术构建交叉引导矩阵，发现模型倾向的分布外泛化“溢出”模式在相同引导技术下跨模型高度可预测，但SFT与GRPO会诱导出不同的溢出特征，且引导良性特质有时会意外泛化出如追求权力等不良特质。 |
| [^93] | [Minimum Volume Optimal Transport for Multivariate Conformal Prediction](https://arxiv.org/abs/2610.14907) | 提出了最小体积OTCP框架，通过学习马氏基础代价并与传输映射联合优化，直接最小化多变量共形预测的预测区域体积，同时保持有限样本边际覆盖保证，并进一步扩展为可学习协变量相关传输度量的个体化方法。 |
| [^94] | [Prediction with Expert Advice under Heavy-Tailed Losses](https://arxiv.org/abs/2610.14904) | 本文针对重尾损失下的专家建议预测问题刻画了极小极大伪遗憾率，并提出无需预知矩参数 ε 和 θ 的自适应重尾 FTRL 与 OMD 算法，在对抗性和自受限环境中均实现近乎最优的遗憾保证。 |
| [^95] | [Physics-informed random neural architectures on compact boundaryless multiply connected manifolds for random-operator models in machine learning and uncertainty quantification](https://arxiv.org/abs/2610.14844) | 本文提出随机物理信息神经网络（SPINNs），通过一个共同的高斯随机源将随机边值问题的随机算子与构建在紧致无边多连通流形上的非前馈随机神经网络进行结构化概率耦合，从而能够表示非高斯随机算子并在观测空间中实现不确定性量化。 |
| [^96] | [Understanding Hidden Spatial Preferences in Bayesian Optimization](https://arxiv.org/abs/2610.14696) | 贝叶斯优化中基于方差的探索策略会继承高斯过程代理模型固有的系统性边界偏差，该研究通过空间权重场将空间偏好与预测模型分离并加以控制，实验证明这种隐藏的空间偏好会显著影响探索区域的分布和最终找到的最优解。 |
| [^97] | [Strong Gaussian Approximation of Constant-Learning-Rate SGD: Bias and Trajectory Inference](https://arxiv.org/abs/2610.14567) | 本文为重尾噪声下的常学习率随机梯度下降建立了KMT型强高斯逼近，实现了有限样本下同时性的整个轨迹层面的推断，并蕴含泛函中心极限定理。 |
| [^98] | [Identication and Estimation of Nonsymmetric Determinantal Point Processes](https://arxiv.org/abs/2610.14520) | 本文针对能够容纳正依赖的非对称行列式点过程，建立了核有效性条件的等价性与识别理论，并提出了基数截断极大似然估计和平均随机梯度下降估计两种方法，证明了一致性与渐近正态性。 |
| [^99] | [Copula Active Subspaces II: Error Decomposition, A Posteriori Estimation, and Sharpness of the Bounds](https://arxiv.org/abs/2610.14498) | 本文为Copula主动子空间方法建立了KL散度的精确误差分解，推导了边缘、投影和降密度各项的先验界及基于残差样本的后验估计，最终获得以至少1-δ概率成立的理论上界。 |
| [^100] | [ByteMark: A Unified Byte-Level Watermark for Detection Across Language Models](https://arxiv.org/abs/2610.14427) | ByteMark提出了一种基于UTF-8字节的统一水印方案，使不同分词器的语言模型生成的水印可被单一检测器识别，解决了现有水印检测依赖特定分词器或嵌入模型所导致的兼容性问题。 |
| [^101] | [Prior-Informed Generalizable Low-Rank Representation Learning via Regularized Adversarial Reweighting](https://arxiv.org/abs/2610.14414) | 提出 PILLAR 方法，通过将对抗性源权重向用户指定的先验参考分布正则化来平滑组分布鲁棒优化目标，在学习共享低秩表示的同时提升估计稳定性并缓解过度保守性，从而实现分布偏移下更可泛化的低秩表示学习。 |
| [^102] | [Decomposing A Neural Learner](https://arxiv.org/abs/2610.14388) | 提出一种神经学习器的概率分解框架，将模型分解为可预测分量和输入相关的可靠性分数，以应对特征空间中可预测性的异质性，从而提升神经网络的透明度与可解释性。 |
| [^103] | [De-biased Distribution Estimation for Locally Differential Private Survival Data](https://arxiv.org/abs/2610.14320) | 本文提出一种基于反卷积核函数的非参数估计方法，可同时消除局部差分隐私对事件时间的扰动偏差和事件指示变量翻转带来的误差，从而对右删失生存数据的真实生存函数实现去偏估计。 |
| [^104] | [Variance Fingerprints: Component-Wise Identification of Content Factors by Augmentation Design](https://arxiv.org/abs/2610.14260) | 该论文提出一种增广设计方法，通过为每个实例记录固定的增广强度集合，使内容因子产生可区分的方差“指纹”，并在理论上证明、在实践中实现内容因子在置换、缩放和平移意义下的逐分量辨识。 |
| [^105] | [Learning Non-Gaussian SEMs with Conditional Risk, and Why the Direction of Recursion Matters](https://arxiv.org/abs/2610.14246) | 本文提出用条件风险替代条件方差，将基于方差的结构方程模型识别方法从高斯情形扩展到非高斯噪声，证明匹配损失方法比最小二乘方法和LiNGAM更准确，且在高斯情形下可互换的自顶向下与自底向上递归方向在非高斯情形下会产生不同的效果。 |
| [^106] | [A Stochastic Analysis Perspective on Diffusion Models](https://arxiv.org/abs/2610.14133) | 本文从随机分析的视角统一阐述基于分数的生成式模型，将经典采样理论、分数估计与扩散时间反演联系起来，并通过路径空间方法连接理想反向扩散与实际生成算法。 |
| [^107] | [PLDiv: Measuring Dataset Diversity and Semantic Collapse via Persistence Landscapes](https://arxiv.org/abs/2610.14031) | 本文提出PLDiv，一种基于持续景观的无监督多样性度量方法，通过量化数据几何表示中邻域结构的演变来衡量数据集多样性并检测语义坍缩。 |
| [^108] | [Rotate the Loop: Blockwise Conformality for Looped Transformers](https://arxiv.org/abs/2610.14030) | 提出分块共形注入方法，将循环Transformer的残差分割为坐标对齐的分组并在组内旋转，使每个分组拥有独立的时间尺度，从而提升状态追踪和语言建模性能。 |
| [^109] | [Can We Trust This Evaluation? Instance-Level Reliability of LLM-as-a-Judge](https://arxiv.org/abs/2610.14029) | 该论文提出了一种无需标签的实例级评估可靠性估计方法SCUR，通过潜在测量模型导出陪审团不确定性、测量交叉熵和扰动敏感性三个互补信号来构建风险分数，能够比现有方法更有效地检测出高共识风险样本这一评估盲点。 |
| [^110] | [PRIME: Frequency-Aware Unified Pruning and Merging for MoE Compression](https://arxiv.org/abs/2610.13982) | PRIME提出了一个统一优化框架，将MoE压缩中的专家剪枝与合并决策联合建模为频率加权目标上的全局整数线性规划问题，从而在保留高频专家作为聚类中心的同时实现高效压缩。 |
| [^111] | [SLICE: Sequential Likelihood-Free Inference via Conformal E-Values](https://arxiv.org/abs/2610.13855) | 提出SLICE方法，利用共形e值构建针对难解似然模拟模型的精确随时有效置信序列，在持续监测下保持有效，并通过摊销神经比率估计器近似理想似然比得分。 |
| [^112] | [Density Ratio Estimation with Stein Displacement Fields](https://arxiv.org/abs/2610.12437) | 该论文提出通过Stein位移场参数化密度比（将对数比建模为负的基础分布Stein算子作用于位移场），用单个凸优化问题统一了分布偏移的统计与动力学描述，并据此发展出无需重训即可修正预训练采样器的push-forward算法和拉近数据分布的pull-back算法。 |
| [^113] | [ISBO: Scalable Spatio-Temporal Bayesian Optimization with Log Gaussian Cox Process Models via the INLA-SPDE Approach](https://arxiv.org/abs/2610.12213) | 该论文提出了首个面向时空数据的可扩展贝叶斯优化框架ISBO，通过对数高斯Cox过程建模与INLA-SPDE推断方法，能够以最少的评估次数稳定定位高强度区域及潜在强度峰值。 |
| [^114] | [Differentiable Systematic Resampling for Variational Sequential Monte Carlo](https://arxiv.org/abs/2610.12094) | 提出可微系统重采样（DSR），一种温度控制的系统重采样松弛方法，在保持其结构特性并具有可证明的指数收敛偏差的同时，实现了完全梯度流动，且计算开销远低于基于最优传输的方法。 |
| [^115] | [Identifiability of a dissipative knowledge-dynamics model: exact recovery under designed excitation, degeneration on observational data](https://arxiv.org/abs/2610.09889) | 该论文将人类学习建模为参数具有机制含义的耗散常微分方程组，证明了在设计激励条件下模型参数可从数据中被精确恢复（双概念情形有闭式解），而在仅依赖观测数据时可辨识性会退化，并提出了数值精确等价且速度大幅提升的半隐式L-稳定批量求解器。 |
| [^116] | [The Best Optimizer Depends on Batch Size](https://arxiv.org/abs/2610.08975) | 该论文挑战了“某一批量大小下最佳的优化器在其他批量大小下也最佳”的常见假设，证明Muon缺乏一致的缩放规则，且即使经过大量超参数调优，语言模型预训练的最佳优化器仍会随批量大小而改变。 |
| [^117] | [Stochastic Gradient Descent Ascent is Suboptimal for Nonconvex-PL Min-Max Games](https://arxiv.org/abs/2610.07814) | 该论文首次建立了非凸-PL极小极大博弈中固定时间尺度比双时间尺度SGDA的紧致复杂度下界，证明SGDA本质上次优——其下界与现有上界匹配、与Smoothed-AGDA形成复杂度分离，且当时间尺度比小于o(κ²)时甚至无法找到驻点。 |
| [^118] | [Retrieval-Based In-Context Learning: A Domain Adaptation Framework](https://arxiv.org/abs/2610.05717) | 该论文首次将基于检索的上下文学习（ICR）形式化为领域自适应问题，在灵活的分布偏移条件下建立了理论保证，量化了该学习范式的优势与局限。 |
| [^119] | [Targeted Active Learning for Preference-Based Treatment Effects on Multivariate Outcomes](https://arxiv.org/abs/2610.03824) | 该论文提出了一种定向主动学习框架，通过从临床专家处选择性查询最有信息量的结局组合两两比较，来最大程度降低基于偏好的治疗效应与最优策略估计中的不确定性，而非仅仅最大化对偏好规则本身的信息增益。 |
| [^120] | [Amortized Structured Stochastic Variational Inference for Gaussian Process Latent Variable Models](https://arxiv.org/abs/2610.03647) | 本文将摊销结构化随机变分推断应用于高斯过程潜变量模型，通过让潜空间的变分后验条件依赖于诱导点的取值，突破了平均场变分近似的局限，从而在数据流形重建的多项指标上取得了改进。 |
| [^121] | [Heteroskedastic Canonical Polyadic Tensor Decomposition](https://arxiv.org/abs/2610.00498) | 本文提出异方差CP分解（HCP），通过引入低秩精度张量来建模张量条目的异方差性，并采用交替分块坐标上升方法从含噪观测中同时估计低秩均值与精度张量，其计算复杂度与CP-ALS相当。 |
| [^122] | [On the Relaxation of Conditional Independence Assumption for Image Segmentation](https://arxiv.org/abs/2609.38930) | 该论文提出用空间局部化依赖（SLD）结构替代RankSEG方法中的条件独立性假设，并借助倒数矩近似与不动点优化策略克服计算瓶颈，从而有效捕获局部标签相关性并提升模糊、低对比度场景下的图像分割性能。 |
| [^123] | [Multi-Marginal Inverse Optimal Transport for Contrastive Learning Via Explicit Anchor-Positive-Negative Coupling](https://arxiv.org/abs/2609.33741) | 该论文提出Neg-MMIOT-CL方法，通过显式建模锚点-正样本-负样本三元组的多边际逆向最优传输耦合，克服了现有逆向OT方法因不利用负样本而导致的维度坍塌问题，提升了对比学习的下游性能。 |
| [^124] | [WTF?! Simulation-Free Reinforcement Learning with Wasserstein-Tilted Flow Maps](https://arxiv.org/abs/2609.27033) | 提出WTF框架，通过基于预训练漂移构建的Wasserstein最优传输正则化器，将奖励微调问题等价转化为流上的确定性最优控制问题，实现了无需模拟的强化学习微调，是首个原生于流映射的端到端微调方案。 |
| [^125] | [PICPIs: Prediction-Interval-Conditional Prediction Intervals](https://arxiv.org/abs/2609.25388) | 该论文提出了预测区间条件化预测区间这一新框架，通过自洽条件使预测区间同时定义预测值分层并保证该层内的平均结果落在同一区间内，从而在共形预测中填补了边际有效性分辨率不足与完全条件保证不可实现之间的空白。 |
| [^126] | [OSCAR: Order-aware Scoring and Calibration for AI Rankings](https://arxiv.org/abs/2609.24128) | 本文提出OSCAR框架，通过在LLM成对评估聚合中建模评委特定的位置等系统性呈现效应，实现AI排名的顺序感知评分与校准，并证明忽略位置效应会严重扭曲最优排名结果。 |
| [^127] | [Single-Loop Stochastic Projected Damped Extragradient Methods for Stochastic Nonconvex--(Strongly) Concave Minimax Optimization](https://arxiv.org/abs/2609.21747) | 提出了保持单循环结构的随机投影阻尼外梯度方法 SPDE 及其递归方差缩减变体 VR-SPDE，用于随机非凸—（强）凹极小极大优化，并同时为博弈平稳性和优化平稳性提供了更优的随机一阶预言机复杂度保证。 |
| [^128] | [Why Learning Rediscovers the Closed-Form Diagonal Regularizer](https://arxiv.org/abs/2609.09656) | 本文提出对角饱和原理，证明当截断噪声各向同性时，贝叶斯最优的Tikhonov正则化是仅由先验决定的闭式幂律，因此学习得到的对角正则化器难以稳健超越闭式解，在声学房间逆问题中闭式解已接近最优。 |
| [^129] | [Gaussian Approximation for Multivariate Martingale Sums from Uniformly Ergodic Markov Chains](https://arxiv.org/abs/2609.09480) | 本文为一致遍历马尔可夫链生成的多元鞅和建立了高阶 Wasserstein 距离下的显式高斯逼近界，并首次在平衡增量情形及马尔可夫链加性泛函中获得最优的 $O(n^{-1/2})$ 逼近率。 |
| [^130] | [Q-Edge: Symmetry-Reduced Quantum Simulation of Structured Extreme Dependence](https://arxiv.org/abs/2609.03706) | Q-Edge提出了一种基于对称性的无损轨道空间表示方法，将30维问题中约160万个角度状态压缩为256个轨道状态，从而实现了结构化极端依赖系统的可扩展量子模拟与数字孪生。 |
| [^131] | [SIGMA: Symmetry-aware, Intelligent, Geometric, Multi-objective Adaptive Control for Robust, Dependable Traffic Management](https://arxiv.org/abs/2608.18263) | 本文提出了SIGMA框架，通过结合大型语言模型进行动态目标调整和旋转增强实现方向不变学习，从而在交通信号控制中实现多目标自适应和跨场景泛化。 |
| [^132] | [The Debiased Score Test: Hunt-and-test for Semiparametric Hypotheses](https://arxiv.org/abs/2607.28861) | 本文提出一种“搜寻-检验”策略的去偏得分检验方法：用一部分数据借助机器学习搜寻得分的最优方向，用另一部分数据检验得分是否为零，并通过加权最小二乘投影进行去偏校正，从而实现对半参数回归模型的拟合优度检验及观察性研究中效应修饰因子的检测。 |
| [^133] | [Recovery of latent inner products from an anisotropic Gaussian random geometric graph](https://arxiv.org/abs/2607.23723) | 该论文提出通过对双重中心化邻接矩阵进行秩-d谱近似，从各向异性高斯随机几何图中恢复潜在内积，其均方误差速率由协方差矩阵的稳定秩刻画，并在各向同性情形下达到最优水平。 |
| [^134] | [Active Inference as a Convex Markov Decision Process](https://arxiv.org/abs/2607.20152) | 该论文将主动推理的期望自由能最小化重新表述为凸马尔可夫决策过程，揭示了自然梯度沿时间反向传播的机制，并将其与表演性强化学习联系起来，从而将主动推理纳入现代强化学习与优化理论框架。 |
| [^135] | [Mixtures of spatial factor analyzers for tensor-variate data](https://arxiv.org/abs/2607.07887) | 本文提出空间因子分析器混合模型（MSFA），通过基于样条的空间衰减协方差结构使协方差参数随空间位置线性增长，并结合矩阵变量因子分析器进一步降维，有效实现高维张量空间数据的聚类分析。 |
| [^136] | [ManifoldFlow: SPD-Relaxed Stiefel Layers with Learnable Singular Spectrum](https://arxiv.org/abs/2607.04535) | ManifoldFlow通过对固定谱Stiefel层进行最小松弛，将权重分解为 W = Q S^{1/2}，在保持基向量位于Stiefel流形上的同时学习有界的正定奇异谱，使特征值裁剪成为直接的奇异值控制机制，并在序列、表格和图像任务中优于固定谱Stiefel层。 |
| [^137] | [The Dual Nature of LLM Persona: Aggregated Tendencies and Frame-Dependent Geometry](https://arxiv.org/abs/2607.02368) | 本论文发现LLM人格表达包含聚合倾向与框架依赖几何两个可分离成分，后者并非固有属性，而是编码聚合无法捕捉信息的协调模式。 |
| [^138] | [Learning a Sampling-Free Variational DNN Plugin from Tiny Training Sets to Refine OOD Segmentation With Uncertainty Estimation](https://arxiv.org/abs/2606.15837) | 提出VarDeepPCA——一种轻量级变分DNN插件框架，仅凭微小分布内训练集、通过对softmax映射的重新解释实现无采样的精确分布建模，无需目标域数据或大规模预训练即可精化OOD医学图像分割并估计不确定性。 |
| [^139] | [On Finite-sample Concentration of Median of Incomplete U-Statistics](https://arxiv.org/abs/2606.00661) | 本文证明了不完整U统计量中位数（MoIU）的有限样本浓度界，克服了此前仅能获得松散$O(n^{-1/4})$界的理论挑战，实现了更紧的收敛速率。 |
| [^140] | [Hoeffding Concept Bottleneck Models with Applications to Overhead Images](https://arxiv.org/abs/2606.00082) | 该论文提出了基于Hoeffding函数分解的概念瓶颈模型（HCBM），用非线性和稀疏的概念分数聚合取代传统线性聚合，在提升可解释性的同时避免信息泄露，并成功应用于航拍图像分类。 |
| [^141] | [Conformal Certification of Reasoning Trace Prefixes](https://arxiv.org/abs/2605.30085) | CROP 是一种与验证器无关的保形校准方法，可为语言模型推理轨迹的连续前缀提供统计错误率保证——返回可安全保留的最长干净前缀，并将未认证部分交由下游审查修复。 |
| [^142] | [How Do Uncertainty Estimators Align with Hallucination Occurrence in LLMs?](https://arxiv.org/abs/2605.27016) | 该论文系统评估了46种不确定性估计器在内在与外在幻觉场景下对大语言模型幻觉的检测能力，发现不确定性与幻觉的关联普遍较弱且因设置和模型而异，挑战了将不确定性直接作为幻觉信号的做法。 |
| [^143] | [Training data attribution in diffusion models via mirrored unlearning and noise-consistent skew](https://arxiv.org/abs/2605.17938) | 本文提出MUCS方法，通过有界镜像梯度上升微调辅助模型并利用一致噪声样本测量归一化偏度，为扩散模型实现了更可靠、更鲁棒的训练数据归因，在三个数据集上大幅超越现有方法。 |
| [^144] | [Learning from Totally Unpaired Data via Minimum Conditional Entropy Coupling](https://arxiv.org/abs/2605.16229) | 提出最小条件熵耦合（MCEC）新方法，仅利用完全非配对的数据集，通过最小化条件熵来耦合边缘分布从而学习类别预测器，并针对该 NP-难问题给出了多项式时间复杂度的算法。 |
| [^145] | [$\alpha$-TCAV: A Unified Framework for Testing with Concept Activation Vectors](https://arxiv.org/abs/2605.15688) | 提出参数化统一框架α-TCAV，用平滑函数替代标准TCAV中导致不稳定的非连续指示函数，在极限情形下恢复TCAV，单个CAV即可达到Multi-TCAV的均值且方差更小，并提供概念正向影响模型输出的校准概率。 |
| [^146] | [Federated Language Models Under Bandwidth Budgets: Distillation Rates and Conformal Coverage](https://arxiv.org/abs/2605.09986) | 该论文提出了两种在带宽受限、数据不出域场景下以小而可预测的代价压缩每条通信消息的联邦学习方法——FPLD（联邦探测-逻辑值蒸馏）用于跨站点联合训练共享语言模型，FC-RAG（联邦共形检索增强生成）用于跨站点协同问答。 |
| [^147] | [When Does Trimming Help Conformal Prediction? A Retained-Law Diagnostic under Calibration Contamination](https://arxiv.org/abs/2605.06204) | 本文证明保形预测中裁剪操作对覆盖率的影响由裁剪后的保留分布决定而非污染水平，并给出裁剪有效的精确条件：异常分数需能区分保留概率同时在干净数据上保持分数中性。 |
| [^148] | [Cover meets Robbins while Betting on Bounded Data: $\ln n$ Regret and Almost Sure $\ln\ln n$ Regret](https://arxiv.org/abs/2604.20172) | 本文提出一种融合 Cover 与 Robbins 思想的显式完全支撑混合下注策略，在保持 O(ln n) 最坏情况遗憾的同时，在内在方差过程发散的路径上实现几乎必然的 ln ln Vₙ 量级且首项常数不可改进的更精细遗憾界。 |
| [^149] | [MosaicMRI: A Diverse Dataset and Benchmark for Raw Musculoskeletal MRI](https://arxiv.org/abs/2604.11762) | 本文发布了迄今最大的开源原始肌肉骨骼MRI数据集MosaicMRI，包含3,150个体数据和89,830个切片，涵盖多种采集方向、成像对比度和解剖部位，为训练和评估基于机器学习的MRI方法提供了多样化基准。 |
| [^150] | [Learning Shortest Paths with Generative Flow Networks](https://arxiv.org/abs/2603.01786) | 该论文提出了基于生成流网络（GFlowNets）的最短路径学习新框架，证明了最小化总流量的GFlowNet策略必沿最短路径遍历，从而将最短路径寻找问题转化为最小总流量问题，并在置换环境寻路和魔方求解任务中取得了与专用最先进方法相当的效果。 |
| [^151] | [Certified Per-Instance Unlearning Using Individual Sensitivity Bounds](https://arxiv.org/abs/2602.15602) | 提出可认证逐实例遗忘框架，利用单个训练点的个体敏感性界自适应调整噪声量，避免最坏情况校准带来的过度保守性，并为确定性训练及岭回归的噪声梯度下降推导出显式遗忘证书。 |
| [^152] | [DCD: Decomposition-based Causal Discovery from Autocorrelated and Non-Stationary Temporal Data](https://arxiv.org/abs/2602.01433) | 提出了一种基于分解的因果发现框架DCD，通过将时间序列分解为趋势、季节和残差分量并分别进行针对性因果分析，有效减少了非平稳和自相关数据中的虚假因果关系，并构建出统一的多尺度因果结构。 |
| [^153] | [Low-Rank Plus Sparse Matrix Transfer Learning under Growing Representations and Ambient Dimensions](https://arxiv.org/abs/2601.21873) | 该论文提出了一个在环境维度与潜在表示同时增长情形下的低秩加稀疏矩阵迁移学习框架，通过锚定交替投影估计器在保留已迁移子空间的同时仅估计低维创新与稀疏修改，实现了严格更优的误差率。 |
| [^154] | [Signal from Structure: Exploiting Submodular Upper Bounds in Generative Flow Networks](https://arxiv.org/abs/2601.21061) | 本文提出SUBo-GFN，通过利用奖励函数的次模性获得未观察组合对象的奖励上界，并据此训练生成流网络，在相同奖励查询次数下比经典方法产生数量级更多的学习信号。 |
| [^155] | [Nonparametric Kernel Clustering with Bandit Feedback](https://arxiv.org/abs/2601.07535) | 该论文提出了一个对臂分布无任何参数化假设的非参数聚类框架，通过在再生核希尔伯特空间中对臂的核均值嵌入进行聚类，突破了现有工作依赖次高斯假设、仅能区分均值差异的局限，从而能够识别臂分布在形状或高阶矩上的差异。 |
| [^156] | [Measuring all the noises of LLM Evals](https://arxiv.org/abs/2512.21326) | 提出“全对比较”方法，定义并测量LLM评估中的预测噪声、数据噪声和总噪声，发现总噪声水平高度可预测，且配对预测噪声通常大于配对数据噪声。 |
| [^157] | [Sampling from multimodal distributions with warm starts: Non-asymptotic bounds for the Reweighted Annealed Leap-Point Sampler](https://arxiv.org/abs/2512.17977) | 本文提出重加权退火跳跃点采样器（Re-ALPS），去除了ALPS的高斯近似假设，并在自然质量假设下证明了利用热启动点进行多峰分布采样的首个一般设定多项式时间收敛界。 |
| [^158] | [Fully Bayesian Spectral Clustering and Benchmarking with Uncertainty Quantification for Small Area Estimation](https://arxiv.org/abs/2512.15643) | 提出了基于谱聚类的贝叶斯小区域估计新模型FH-SC，借助后验投影理论推导出基准校正后的Rao-Blackwell估计量闭式解，并引入可推广的条件后验均方误差（CPMSE）来量化基准估计量的不确定性。 |
| [^159] | [Control Consistency Losses for Diffusion Bridges](https://arxiv.org/abs/2512.05070) | 本文基于最优控制的自洽性性质提出了一种学习扩散桥的新方法，以迭代在线方式学习条件动力学，无需对模拟轨迹进行微分即可在多种场景中取得优异表现。 |
| [^160] | [Self-sufficient Independent Component Analysis for Demixing Flows](https://arxiv.org/abs/2512.00665) | 提出一种无先验、无似然的自充分独立成分分析方法，通过最小化条件KL散度并顺序学习解混流模型来从数据中学习解耦信号，同时完全避免了不稳定的对抗训练。 |
| [^161] | [An Infinite BART model](https://arxiv.org/abs/2511.20087) | 该论文提出了一种无限BART模型，通过引入带有印度自助餐过程先验的二值权重矩阵，实现了决策树数量的自动选择，并允许不同观测簇仅使用部分弱学习器从而具有不同的回归函数。 |
| [^162] | [The Optimization Landscape of Carath\'eodory Decomposition of Toeplitz Covariances](https://arxiv.org/abs/2511.01605) | 该论文通过对Toeplitz协方差进行过参数化Carathéodory表示（同时优化频率与幅度），证明这种非线性参数化不会引入额外的驻点，从而解释了为何高斯最大似然非凸问题在样本充足时可被全局求解。 |
| [^163] | [Complexity Dependent Error Rates for Physics-informed Statistical Learning via the Small-ball Method](https://arxiv.org/abs/2510.23149) | 本文通过小球方法为物理信息统计学习建立了复杂度相关的误差率，回答了软惩罚与硬约束之间的权衡问题，并量化了引入物理知识相较于纯数据驱动方法所带来的统计增益。 |
| [^164] | [Deep Jump Gaussian Processes for Surrogate Modeling of High-Dimensional Piecewise Continuous Functions](https://arxiv.org/abs/2510.21974) | 本文提出深度跳跃高斯过程（DJGP），通过将区域特定的局部线性投影与跳跃高斯过程相结合的双层深度学习框架，解决了传统 JGP 在高维空间中的局限，实现了高维分段连续函数的高效代理建模。 |
| [^165] | [Stochastic Difference-of-Convex Optimization with Momentum](https://arxiv.org/abs/2510.17503) | 本文提出基于动量的随机凸差优化算法MDCA，将阻尼DCA步骤与动量估计器结合，在稀疏学习、鲁棒回归等非凸优化问题上以更少的随机梯度调用次数达到与DCA-PAGE相当的稳定性能。 |
| [^166] | [Mode Collapse of Mean-Field Variational Inference](https://arxiv.org/abs/2510.17063) | 该论文首次从理论上解释了平均场变分推断中的众数坍缩现象，通过引入 ε-分离性概念，推导出 MFVI 优化器分配给混合分布各分量质量的显式界，并扩展到多分量混合分布的情形。 |
| [^167] | [Batch Bayesian Active Learning with Partial Batch Label Sampling](https://arxiv.org/abs/2510.09877) | 该论文扩展了主动学习的贝叶斯决策理论框架，提出部分批次标签采样方法ParBaLS，使EPIG算法在批次设置下既避免了BatchBALD的计算难题，又克服了top-B选择的性能下降，在不同数据集和预算下均取得卓越性能。 |
| [^168] | [Bayesian Optimization under Uncertainty for Training a Scale Parameter in Stochastic Models](https://arxiv.org/abs/2510.06439) | 提出了一种面向不确定性下随机模型尺度或精度参数优化的新贝叶斯优化框架，通过统计代理模型实现期望的解析求值以及随机采集函数最优解的闭式表达式，相比基于蒙特卡洛的优化方案数据使用量减少约40倍，大幅提升了优化效率。 |
| [^169] | [Rethinking Langevin Thompson Sampling from A Stochastic Approximation Perspective](https://arxiv.org/abs/2510.05023) | TS-SA将随机近似（SA）融入朗之万汤普森采样框架，每轮仅用新观测奖励执行一步朗之万更新并结合递减的SA权重，为每个臂维持一条持续的马尔可夫链，从而无需存储或重复采样历史观测即可高效逼近变化的后验分布。 |
| [^170] | [Metadata-Guided Adaptable Frequency Scaling across Heterogeneous Applications and Devices](https://arxiv.org/abs/2509.22707) | 提出MetaDVFS，一个元数据引导的多任务强化学习框架，通过挖掘并迁移设备与应用元数据中的共享知识，实现跨异构硬件和应用的可泛化、低部署成本的DVFS频率调节。 |
| [^171] | [Bouncy particle sampler with infinite exchanging parallel tempering](https://arxiv.org/abs/2509.02003) | 本文提出了无限交换极限下的并行回火弹跳粒子采样器（BPS-PT），在曲率变化较大的分布上实现了与HMC-PT和LMC-PT相当的估计精度，同时单位计算时间的有效样本量更高。 |
| [^172] | [High-Order Error Bounds for Markovian LSA with Richardson-Romberg Extrapolation](https://arxiv.org/abs/2508.05570) | 本文提出一种新颖的线性化偏差分解方法，揭示PR平均无法消除马尔可夫噪声下LSA算法偏差的线性主导项，并通过Richardson-Romberg外推有效消除该偏差项，同时建立了高阶矩误差界。 |
| [^173] | [The Serial Scaling Hypothesis](https://arxiv.org/abs/2507.12549) | 该论文提出“串行缩放假设”，在复杂性理论中形式化了“内在串行问题”与可并行问题的区别，并首次证明扩散模型也无法解决这类本质串行的问题，指出识别计算的串行本质对机器学习、模型设计和硬件开发意义重大。 |
| [^174] | [Equilibrium Distribution for t-Distributed Stochastic Neighbor Embedding with Generalized Kernels](https://arxiv.org/abs/2505.24311) | 该论文为广义输入输出核下的t-SNE大样本变分问题建立了严格的数学理论，证明了尺度参数的存在唯一性、解的存在性与一致有界性，以及离散最优解收敛到满足平衡方程的紧支撑平衡分布。 |
| [^175] | [Stein's unbiased risk estimate and Hyv\"arinen's score matching](https://arxiv.org/abs/2502.20123) | 该论文首次证明了SURE最小化与Hyvärinen隐式分数匹配的等价性，并从理论上证明SURE在经验贝叶斯去噪中可实现接近参数速率的遗憾收敛，以及在误设模型下向oracle去噪器的快速收敛。 |
| [^176] | [Differentially Private Synthetic Data via APIs 3: Using Simulators Instead of Foundation Model](https://arxiv.org/abs/2502.05505) | Sim-PE将私有演化（PE）框架推广到可自然集成计算机图形图像生成器和物理仿真工具等非神经网络模拟器，从而在不依赖基础模型的情况下生成差分隐私合成数据，并将下游分类准确率提升最高3倍、FID降低最高80%。 |
| [^177] | [Targeted Data Fusion for Region-Specific Survival Effects in the AMP HIV Prevention Trials](https://arxiv.org/abs/2501.18798) | 针对AMP艾滋病预防试验，提出了一种基于L1正则化准则的联邦学习方法，在不共享个体数据的情况下融合多区域信息，以估计区域特异性和治疗特异性的生存曲线及因果效应。 |
| [^178] | [Optimal Transport Aggregation for Distributed Mixture-of-Experts](https://arxiv.org/abs/2312.09877) | 提出一种基于最优传输散度的聚合方法，可将分布式环境中独立训练的多个本地混合专家模型合并为一个一致且仅含K个专家的全局MoE模型。 |
| [^179] | [Bayesian Nonparametric Thompson Sampling for Multi-Armed Bandits](https://arxiv.org/abs/1808.02932) | 本文提出用贝叶斯非参数高斯混合模型对每个臂的奖励分布建模，将汤普森采样扩展到任意奖励分布，从而无需针对具体场景预先设定参数化奖励模型和调整超参数。 |
| [^180] | [Semantic Context Forests for Learning-Based Knee Cartilage Segmentation in 3D MR Images](https://arxiv.org/abs/1307.2965) | 本文提出一种基于语义上下文森林的迭代多类学习方法，通过利用骨与软骨间的空间上下文约束以及解剖标志点距离特征，实现三维MR图像中股骨、胫骨和髌骨软骨的同时自动分割。 |
| [^181] | [Concentration inequalities for leave-one-out cross validation.](http://arxiv.org/abs/2211.02478) | 本文证明了估计器的稳定性足以说明留一法交叉验证是可靠的，并通过提供集中界限超出Lipschitz连续性假设的损失函数或估计器，为我们提供了一个相对丰富的分布类。 |

# 详细

[^1]: 面向化学反应网络的联合神经后验估计与模型选择

    Joint Neural Posterior Estimation and Model Selection for Chemical Reaction Networks

    [https://arxiv.org/abs/2610.18825](https://arxiv.org/abs/2610.18825)

    本文提出了一种结合共享一维卷积神经网络与置换不变集合注意力机制的神经推断架构，可从多实验时间序列数据中同时完成化学反应网络的模型选择与参数后验估计，并在真实的去硼酸化反应示例中验证了其实用性。

    

    针对化学反应网络的基于模拟的贝叶斯推断必须同时解决两个计算上极具挑战性的问题：区分相互重叠的反应路径以及估计反应速率常数。经典方法（如序贯蒙特卡洛近似贝叶斯计算，SMC-ABC）依赖手工设计的汇总统计量，且在大参数空间下扩展性较差。为克服这一局限，我们提出了一种神经推断架构。我们的流程利用共享的一维卷积神经网络（1D-CNN）和置换不变的集合注意力机制，从变长的多实验时间序列数据中学习低维汇总统计量。这些统计量同时用于条件化一个模型选择网络以计算后验模型概率，以及针对各模型的条件可逆神经网络，用以估计参数后验分布。我们在一个真实的去硼酸化反应示例上展示了该方法的实际效用。

    arXiv:2610.18825v1 Announce Type: new  Abstract: Simulation-based Bayesian inference for Chemical Reaction Networks (CRNs) must simultaneously solve two computationally demanding problems: discriminating between overlapping reaction pathways and estimating reaction rate constants. Classical approaches, such as Sequential Monte Carlo ABC, rely on handcrafted summary statistics and scale poorly across large parameter spaces. To overcome this, we introduce a neural inference architecture. Our pipeline utilizes shared 1D-CNNs and a permutation-invariant Set Attention mechanism to learn low-dimensional summary statistics from variable-length, multi-experiment time-series data. These statistics simultaneously condition a model selection network to compute posterior model probabilities and model-specific conditional Invertible Neural Networks (cINNs) that estimate parameter posterior distributions. We demonstrate the practical utility of our approach on a real protodeboronation example, illus
    
[^2]: 面向预测的设计：预期样本外 $R^2$ 和 $r$ 的规划公式

    Designing for Prediction: Planning Formulas for Expected Out-of-Sample $R^2$ and $r$

    [https://arxiv.org/abs/2610.18781](https://arxiv.org/abs/2610.18781)

    本文为留一交叉验证下的OLS和岭回归预测模型推导了将预期样本外 $R^2$ 及观测-预测相关系数与样本量、预测变量数、假设总体 $R^2$ 和设计效率因子联系起来的闭式规划公式，使研究者能在数据收集前评估预测性能，且这些公式相比经典收缩公式还对协变量的不平衡和偏斜敏感。

    

    设计计算让研究者能够在数据收集之前评估计划中的研究是否有可能检测到效应。在本文中，我们针对使用普通最小二乘法（OLS）和岭回归、并在留一交叉验证下的预测模型开发了此类计算。为此，我们推导出了闭式表达式，将预期样本外 $R^2$ 以及观测结果与预测结果之间的预期相关性与四个设计阶段的量联系起来：样本量、预测变量数量、假设的总体 $R^2$（$\rho^2_{in}$）以及设计效率因子 $D^*$（它概括了杠杆值在各观测之间的分布方式）。与仅取决于样本量和维度的经典收缩公式不同，这些表达式对协变量的排列方式敏感——即对不平衡和偏斜敏感。一项模拟研究在各种数据生成机制下评估了这些公式相对于实际值的准确性。

    arXiv:2610.18781v1 Announce Type: cross  Abstract: Design calculations let researchers gauge, before data is collected, whether a planned study is likely to detect an effect. In this paper, we develop such calculations for predictive models using ordinary least squares and ridge regression under leave-one-out cross-validation. To do so, we derive closed-form expressions relating the expected out-of-sample $R^2$ and the expected correlation between observed and predicted outcomes to four design-stage quantities: sample size, number of predictors, an assumed population $R^2$ ($\rho^2_{in}$) and a design efficiency factor $D^*$ (which summarizes how leverage is distributed across observations). Unlike the classical shrinkage formulas, which depend only on sample size and dimension, these are sensitive to how covariates are arranged -- to imbalance and skew. A simulation study assesses the accuracy of the formulas against realized values under a variety of data generating mechanisms. We ap
    
[^3]: 一种速率转移方法：通过随机微积分获得随机梯度下降的收敛速率

    A Rate Transfer Method: Convergence Rates in Stochastic Gradient Descent via Stochastic Calculus

    [https://arxiv.org/abs/2610.18739](https://arxiv.org/abs/2610.18739)

    提出一种速率转移方法，将随机微分方程的有限时间Lyapunov估计转化为随机梯度算法的有限时间收敛界，使得新优化算法的收敛速率可通过简单的微积分论证来建立。

    

    我们开发了一种速率转移方法，该方法将相关随机微分方程（SDE）的有限时间Lyapunov估计转化为随机梯度算法的有限时间收敛界。在一般的矩条件和光滑性条件下，随机梯度方法的收敛分析可以简化为验证该SDE的Lyapunov生成元不等式的微积分论证。该方法恢复了经典随机梯度方法已确立的收敛速率，并为新优化算法提供了一种通过对其相关SDE应用微积分论证来建立收敛速率的途径。

    arXiv:2610.18739v1 Announce Type: cross  Abstract: We develop a rate-transfer method that converts finite-time Lyapunov estimates for an associated stochastic differential equation (SDE) into finite-time convergence bounds for stochastic gradient algorithms. Under generic moment and smoothness conditions, the convergence analysis of the stochastic gradient method reduces to a calculus argument verifying a Lyapunov generator inequality for the SDE. The method recovers established rates for classical stochastic gradient methods and provides a way to establish convergence rates for new optimization algorithms through calculus arguments applied to their associated SDEs.
    
[^4]: 从傅里叶振幅到持久性图像的冲击回波缺陷检测

    From Fourier Amplitudes to Persistence Images for Impact-Echo Defect Detection

    [https://arxiv.org/abs/2610.18732](https://arxiv.org/abs/2610.18732)

    该论文提出从傅里叶振幅谱构建二维频率-振幅持久性图像（FA-PI）的新方法，利用多参数持续同调将频率与振幅信息编码，有效应对边界反射、干扰和噪声导致的复杂频谱问题，提升了冲击回波缺陷检测的分类性能。

    

    冲击回波检测是一种用于检测混凝土结构内部缺陷的无损检测方法。然而，边界反射、结构性干扰和测量噪声会产生复杂的傅里叶振幅谱，使缺陷分类变得困难。我们从傅里叶振幅谱构建了二维的频率-振幅持久性图像（FA-PI）。利用精确多参数持续同调的固定参数化方法，每个保留的傅里叶分量以频率箱位置作为一个坐标、振幅作为另一个坐标进行表示，随后在该二维空间中应用高斯核。这种构造与相应的一维高斯平滑傅里叶振幅谱之间存在直接的分析联系。FA-PI 在一个实验性填筑体空洞数据集和一个公开的桥面板冲击回波数据集上进行了评估，并与原始的和高斯平滑的傅里叶振幅谱方法进行了比较。

    arXiv:2610.18732v1 Announce Type: cross  Abstract: Impact-echo testing is a nondestructive testing method used to detect internal defects in concrete structures. However, boundary reflections, structured interference, and measurement noise can produce complex Fourier amplitude spectra, making defect classification difficult. We constructed a two-dimensional frequency--amplitude persistence image (FA-PI) from the Fourier amplitude spectrum. Using a fixed parameterization of exact multi-parameter persistent homology, each retained Fourier component is represented with frequency-bin location as one coordinate and amplitude as the other, and Gaussian kernels are then applied in this two-dimensional space. This construction has a direct analytical connection to the corresponding one-dimensional Gaussian-smoothed Fourier amplitude spectrum. FA-PI was evaluated on an experimental backfill-void dataset and a public bridge-deck impact-echo dataset, and compared with raw and Gaussian-smoothed Fo
    
[^5]: BridgeTwist：扭转薛定谔桥以实现免训练条件采样

    BridgeTwist: Twisting Schr\"odinger Bridges for Training-Free Conditional Sampling

    [https://arxiv.org/abs/2610.18658](https://arxiv.org/abs/2610.18658)

    提出了BridgeTwist，一种基于带温度调节扭转序贯蒙特卡洛方案的免训练条件采样器，可复用预训练薛定谔桥生成模型，利用由速度场与得分场构造的闭式即插即用扭转代理实现条件后验采样。

    

    生成模型通常需要服务于多种下游条件任务。在免训练的设置下，预训练的无条件模型被复用为先验，并与任务特定的似然函数相结合，从而从相应的后验分布中采样。尽管针对扩散模型和流模型已开发出此类免训练采样器，但作为在一般源分布与目标分布之间进行灵活随机输运的模型，薛定谔桥（SB）正日益受到关注，而其对应的免训练采样器却尚未被探索。我们提出了BridgeTwist，一种面向预训练薛定谔桥生成模型的免训练条件采样器，其被表述为一种带温度调节的扭转序贯蒙特卡洛方案。其中的扭转量是一个闭式的即插即用代理，通过将SB预训练所暴露的速度场与得分场组合为一个终端预测器，并在该终端处评估似然函数而得到。由于该代理在靠近源分布处可靠性较低，我们对其进行了温度调节……

    arXiv:2610.18658v1 Announce Type: cross  Abstract: Generative models are often required to serve many downstream conditional tasks. In the training-free setting, a pretrained unconditional model is reused as a prior and combined with a task-specific likelihood to sample from the corresponding posterior. While such training-free samplers have been developed for diffusion and flow models, their counterpart for Schr\"odinger bridge (SB) models remains unexplored, despite the growing appeal of SBs as flexible stochastic transport models between general source and target distributions. We propose BridgeTwist, a training-free conditional sampler for pretrained SB generative models, formulated as a tempered twisted sequential Monte Carlo scheme. Its twist is a closed-form plug-in surrogate, obtained by combining the velocity and score fields exposed by SB pretraining into an endpoint predictor and evaluating the likelihood there. Since this surrogate is less reliable near the source, we tempe
    
[^6]: 逐点CATE估计的极小极大速率：设计平滑性的作用

    Minimax rates for pointwise CATE estimation: the role of design smoothness

    [https://arxiv.org/abs/2610.18623](https://arxiv.org/abs/2610.18623)

    该论文揭示了协变量设计平滑性对逐点条件平均处理效应极小极大估计速率的关键影响，证明了粗糙的协变量设计即使在处理效应与冗余函数具有相同平滑性假设的条件下，也会导致比均匀设计更慢的收敛速率。

    

    我们研究了设计平滑性在固定协变量点处条件平均处理效应（CATE）的极小极大估计中所起的作用。对于有上界且远离零的设计密度，在无需进一步正则性假设的情况下，我们推导出了风险上界和下界，两者在收敛率指数上的差距可以任意小。在均匀设计下，我们在低冗余函数平滑性的情形中获得了内点处相互匹配的上下界。这些结果表明，粗糙的协变量设计可能导致比均匀设计更慢的收敛速率——即使对处理效应函数和冗余函数施加相同的平滑性假设——从而凸显了协变量分布在逐点因果估计中的重要作用。

    arXiv:2610.18623v1 Announce Type: cross  Abstract: We study the role of design smoothness in the minimax estimation of conditional average treatment effects at a fixed covariate point. For design densities bounded above and away from zero, without further regularity, we derive upper and lower risk bounds with an arbitrarily small gap in the convergence-rate exponent. Under uniform design, we obtain matching upper and lower bounds at interior points in the regime of low nuisance smoothness. These results show that rough covariate designs can lead to slower rates than uniform designs, even under the same smoothness assumptions on the treatment effect and nuisance functions, highlighting the role of the covariate distribution in pointwise causal estimation.
    
[^7]: 理解循环置换以扩展 CMinHash 估计器

    Understanding Circulant Permutation to Extend the CMinHash Estimator

    [https://arxiv.org/abs/2610.18551](https://arxiv.org/abs/2610.18551)

    本文提出了一种新的近似方差分析方法，成功将 CMinHash 的循环置换技术扩展到 Minner 等其他 MinHash 估计器，并通过数值实验验证了方差表达式与经验均方误差的一致性。

    

    CMinHash 通过循环置换为设计具有更低方差和更少存储空间的 MinHash 算法开辟了新方向。然而，原有的方差分析难以扩展到其他 MinHash 估计器。我们提出了一种新方法，能够给出显式且易于比较的近似方差表达式，并利用该方法分析了 Minner 估计器。数值实验表明，我们的方差表达式与估计的经验均方误差（MSE）相吻合。因此，我们的工作展示了如何将循环置换应用于其他 MinHash 估计器并计算它们的近似方差。

    arXiv:2610.18551v1 Announce Type: new  Abstract: CMinHash promises a new direction for designing MinHash algorithms with reduced variance and storage space via circulant permutations. However, the original variance analysis is difficult to extend to other MinHash estimators. We present a new approach that gives explicit and easily comparable approximate variance expressions, and use it to analyze the Minner estimator. Numerical experiments show that our variance expression matches the empirical mean square error (MSE) of estimates. Our work thereby demonstrates how to apply circulant permutation to other MinHash estimators and compute their approximate variance.
    
[^8]: 一石二鸟：曲率感知剪枝同时保留影响函数

    Two For The Price Of One: Curvature-Aware Pruning Preserves Influence Functions Too

    [https://arxiv.org/abs/2610.18516](https://arxiv.org/abs/2610.18516)

    提出了基于ADMM的曲率感知剪枝优化器CACTUS，能够在剪枝和稀疏加低秩分解中保留影响函数所依赖的梯度信息，相比LLM-Surgeon速度提升超过一个数量级，并显著缩小稀疏模型与稠密模型之间的困惑度差距。

    

    训练后剪枝能够实现模型的高效部署，但传统方法并未显式地保留稀疏模型与稠密模型之间的内部梯度对齐。我们研究了压缩模型的下游性能与通过影响函数保留来维持梯度信息的能力之间的新联系。这一联系的核心是一个曲率感知的目标函数，它同时考虑了层激活和反向传播的梯度。尽管前景可观，但由于缺乏高效的求解器，该目标函数在剪枝领域仍未得到充分探索。我们提出了CACTUS，一种基于ADMM的新型优化器，既支持对预训练权重进行剪枝，也支持稀疏加低秩分解。与针对相同目标的LLM-Surgeon相比，CACTUS可以实现超过一个数量级的加速。在对Qwen3-4B进行50%剪枝时，CACTUS将C4困惑度与稠密模型的差距降低了45%，……

    arXiv:2610.18516v1 Announce Type: cross  Abstract: Post-training pruning enables efficient deployment, but conventional methods do not explicitly preserve internal-gradient alignment between sparse and dense models. We study novel connections between downstream performance of compressed models and the ability to retain gradient information via influence-function preservation. Central to this connection is a curvature-aware objective that accounts for both layer activations and backpropagated gradients. Despite its promise, this objective remains underexplored in pruning due to a lack of efficient solvers. We introduce $\texttt{CACTUS}$, a novel ADMM-based optimizer supporting both pruning and sparse-plus-low-rank decompositions of pre-trained weights. Compared to LLM-Surgeon which targets the same objective, $\texttt{CACTUS}$ can achieve over an order-of-magnitude speedup. At $50\%$ pruning of Qwen3-4B, $\texttt{CACTUS}$ reduces the C4 perplexity gap to the dense model by $45\%$, impro
    
[^9]: 你能低到什么程度：用贝叶斯优化微调大语言模型

    How Low Can You Go: Fine-Tuning LLMs with Bayesian Optimization (LlMBO)

    [https://arxiv.org/abs/2610.18504](https://arxiv.org/abs/2610.18504)

    本文提出LlMBO方法，利用低维微调场景下全局优化变得可行这一观察，将贝叶斯优化应用于大语言模型微调，实现了样本高效、无梯度、轻内存且可直接优化贪心解码准确率的微调方案。

    

    最近，Morris等人提出了TinyLoRA，这是一种为大语言模型微调引入任意少量参数的策略，并证明了仅使用少于10个参数（而非基于LoRA的方法通常所需的约10,000个以上参数）即可实现准确率的提升。然而，该研究仍继续使用基于梯度的局部搜索方法——而这正是最初为训练数十亿参数大语言模型所部署的方法。本文基于这样一个观察：在低维设置下，全局优化方法变得可行，从而开启了大型语言模型贝叶斯优化。LlMBO具有样本高效、无需梯度、内存占用轻的特点，可以在设备间无需同步地进行并行，并且与策略梯度方法不同，它可以直接优化贪心解码准确率。我们描述了在数学基准测试套件上，将LlMBO与GRPO在多种模型的微调任务上进行比较的数值实验。

    arXiv:2610.18504v1 Announce Type: new  Abstract: Recently, Morris et al. introduced TinyLoRA, a strategy for LLM fine-tuning which introduces arbitrarily few parameters, and demonstrated accuracy improvements using less than $10$ parameters rather than the ${\gtrsim 10,000}$ typically demanded by LoRA-based methods. However, that study continued to use gradient-based local search methods of the kind deployed for training multi-billion parameter LLMs in the first place. This article is predicated on the observation that, in the low dimensional setting, global optimization methods become tractable, unlocking Large language Model Bayesian Optimization (LlMBO). LlMBO is sample-efficient, gradient-free, memory-light, parallelizes across devices without synchronization, and, unlike policy-gradient methods, can optimize greedy-decoding accuracy directly. We describe numerical experiments comparing LlMBO against GRPO on fine-tuning tasks for a variety of models on a math benchmark suite which 
    
[^10]: 少标签下多问题的再校准：极小极大速率与可辨识性极限

    Recalibrating Many Questions with Few Labels: Minimax Rates and Identification Limits

    [https://arxiv.org/abs/2610.18488](https://arxiv.org/abs/2610.18488)

    本文针对每个问题仅有少量标签的多个相关预测问题，首次刻画了再校准的极小极大最优复合风险率与可辨识性极限，并给出了能共享信息、达到最优速率的合并等张回归估计方法。

    

    许多相关的预测问题（我们称之为“问题”），每个问题仅有少量标签。哪些信息可以在这些问题之间共享，而做出良好决策实际上又需要什么？我们针对再校准回答了这两个问题，其动机来自诸如 Jev（TypeSafe AI, 2026）之类的决策模型，该模型能对开发者编写的任意“是/否”问题返回一个概率。预测器在 m 个问题上报告概率 P，其输出分布为 μ，每个问题有 n 个标签和一个非递减映射 g_j(p)=P(Y=1|P=p)；复合风险定义为相对于真实映射的超额 Brier 分数在所有问题上的平均。当所有映射与某一非递减形状（固定效应）之间的上确界距离 τ ≤ 1、全变差 V ≤ 2 时，对任意非原子的 μ，极小极大复合风险在通用常数意义下为 (mn)^{-2/3} + min{τ², 1/n} + min{τ², V², (V/n)^{2/3}}。一种仅在所有问题的留出标签上单次定标、并带逐问题修正的合并等张回归拟合方法……（摘要在此处被截断）

    arXiv:2610.18488v1 Announce Type: cross  Abstract: Many related prediction problems, which we call questions, come with few labels each. What information can be shared across them, and what is actually necessary for good decisions? We answer both for recalibration, motivated by decision models such as Jev (TypeSafe AI, 2026) that return a probability for any yes/no question a developer writes. A predictor reports a probability $P$ with output law $\mu$ on $m$ questions, each with $n$ labels and a nondecreasing map $g_j(p)=\mathbb{P}(Y=1\mid P=p)$; the compound risk is the excess Brier score over the true maps, averaged over questions. Over maps within sup-distance $\tau\le 1$ and total variation $V\le 2$ of one nondecreasing shape (fixed effects), the minimax compound risk is $(mn)^{-2/3}+\min\{\tau^2,1/n\}+\min\{\tau^2,V^2,(V/n)^{2/3}\}$ up to universal constants, for every non-atomic $\mu$. A pooled isotonic fit with a per-question correction sized once, on held-out labels of all que
    
[^11]: 基于收敛率引导的连续仿真优化期望改进算法设计

    Convergence-Rate-Guided Algorithm Design of Expected Improvement for Continuous Simulation Optimization

    [https://arxiv.org/abs/2610.18434](https://arxiv.org/abs/2610.18434)

    该论文通过推导简单遗憾的高概率上界，在收敛率分析的基础上系统性地设计了随机克里金期望改进算法的三个关键组件——当前最优解、采样重复次数和终端推荐，解决了未知异方差噪声下连续仿真优化中依赖启发式规则进行算法设计的局限。

    

    基于随机克里金替代模型的期望改进方法被广泛应用于连续仿真优化，但仿真输出上未知的异方差噪声使得三个基本设计选择变得至关重要：用于定义改进的当前最优解、每个采样点的重复次数，以及终端推荐。现有方法通常要么假设噪声水平已知，要么通过启发式规则或经验校准分别选择这些组件。我们转而推导简单遗憾的高概率上界，并在一般规则类内系统地选择这三个组件以收紧该上界。我们首先为具有未知和已估计噪声水平的随机克里金预测器建立有限样本置信界，作为三个设计选择的基础。随后我们分析了用于当前最优解和终端推荐的一般后验分位数族。

    arXiv:2610.18434v1 Announce Type: new  Abstract: Expected improvement (EI) based on stochastic kriging (SK) surrogate is widely used in continuous simulation optimization, but unknown heteroscedastic noise on simulation output makes three basic design choices consequential: the incumbent used to define improvement, the replication level at each sampled point, and the terminal recommendation. Existing methods typically either assume the noise levels are known or choose these components separately through heuristic rules or empirical calibration. We instead derive a high-probability upper bound on the simple regret and systematically choose the three components within general rule classes to tighten this bound. We first establish a finite-sample confidence bound for the SK predictor with unknown and estimated noise levels as a foundation for the three design choices. We then analyze general posterior-quantile families for both the incumbent and the terminal recommendation. For the incumb
    
[^12]: 一般条件下随机特征的最优性：模型误设情形与鲁棒学习

    On the Optimality of Random Features under General Conditions: Misspecified Cases and Robust Learning

    [https://arxiv.org/abs/2610.18333](https://arxiv.org/abs/2610.18333)

    本文在模型误设与重尾噪声等更一般的条件下对带随机特征的核岭回归进行了全面分析，改进了特征映射的一致界并证明了极小化极大最优学习率。

    

    随机特征通过用有限维特征映射来近似核函数，为将核方法扩展到大规模数据集提供了一种标准方法。现有的随机特征方法研究主要集中在良设模型和轻尾噪声条件下，这使得在更一般的条件下（如模型误设和重尾噪声）理解其性能方面存在研究空白。本文在一般条件下对带随机特征的核岭回归（KRR-RF）进行了全面且改进的分析。在轻尾噪声设置下，我们改进了以往工作中所使用的特征映射的最坏情况一致界，并在 $2r+\gamma>\alpha_0$ 的条件下得到了极小化极大最优学习率，其中 $(r,\gamma,\alpha_0)$ 分别表示目标函数的正则性、核的有效维度以及再生核希尔伯特空间（RKHS）的嵌入指数。在重尾噪声设置下，……

    arXiv:2610.18333v1 Announce Type: new  Abstract: Random features provide a standard approach to scaling kernel methods to large datasets by approximating the kernel with a finite-dimensional feature map. Existing works of random feature methods primarily focus on well-specified models and light-tailed noise conditions, leaving a gap in understanding their performance under more general conditions, such as misspecified models and heavy-tailed noise. In this paper, we provide a comprehensive and improved analysis of kernel ridge regression with random features (KRR-RF) under general conditions. In the light-tailed noise setting, we improve upon the worst-case uniform bounds of the feature map used in the previous works and lead to the minimax optimal learning rates with $2r+\gamma>\alpha_0$, where $(r,\gamma,\alpha_0)$ denote the regularity of the target function, the effective dimension of the kernel, and the embedding index of the RKHS, respectively. In the heavy-tailed noise setting, 
    
[^13]: 算子代理模型

    Operator Surrogates

    [https://arxiv.org/abs/2610.18210](https://arxiv.org/abs/2610.18210)

    本章对算子代理模型进行了数学导论，系统介绍了编码器-逼近器-解码器的构建范式与高效表示方法，并对比了基于多项式混沌展开的谱方法与神经网络等主要代理架构。

    

    本章对算子代理模型进行了数学介绍。算子代理模型是用于逼近无限维（函数）空间之间复杂（通常为非线性）算子的有限参数映射。构建算子代理模型的一种典型方法是编码器-逼近器-解码器范式，即将无限维输入映射到有限维潜在空间进行处理，然后再映射回无限维输出空间。我们讨论了构建高效表示系统的多种方法，包括稳定的线性框架（如Banach框架）以及数据驱动技术（如主成分分析（PCA）、主动子空间和非线性自编码器）。随后，我们综述了关键的代理模型架构，将基于多项式混沌展开及相关技术的谱方法与较新的神经网络方法进行了对比。（注：原摘要在此处截断）

    arXiv:2610.18210v1 Announce Type: cross  Abstract: This chapter provides a mathematical introduction to operator surrogates, which are finite-parametric maps approximating complex (typically nonlinear) operators between infinite-dimensional (function) spaces.   A typical approach to constructing operator surrogates is the encoder-approximator-decoder paradigm, whereby infinite-dimensional inputs are mapped to a finite-dimensional latent space, processed, and then mapped back to the infinite-dimensional output space. We discuss various methods for creating efficient representation systems, including stable linear frameworks like Banach frames and data-driven techniques such as Principal Component Analysis (PCA), Active Subspaces, and nonlinear autoencoders. Subsequently, we survey key surrogate architectures, contrasting spectral methods, based on polynomial chaos expansions and related technologies, with more recent neural network approaches. We further discuss architectures that do no
    
[^14]: 基于算子值核的层次时间序列预测

    Hierarchical Time Series Forecasting with Operator-Valued Kernels

    [https://arxiv.org/abs/2610.18196](https://arxiv.org/abs/2610.18196)

    该论文提出一个基于结构化算子值核的统一框架，将层次时间序列预测转化为向量值再生核希尔伯特空间中的多元回归问题，使预测在构造上即满足层次一致性约束，无需事后协调步骤，并给出了泛化性能的过剩风险理论保证。

    

    层次时间序列预测要求生成在多个聚合层级上保持一致（coherent）的准确预测。现有方法要么依赖带有事后协调（reconciliation）步骤的两阶段流水线，要么依赖通常缺乏统计保证的端到端模型。我们引入了一个统一框架，将层次预测表述为向量值再生核希尔伯特空间（RKHS）中的多元回归问题。通过设计结构化的算子值核，我们的方法直接在假设空间内强制实施一致性，从而确保所有预测在构造上即满足层次约束。我们进一步为层次数据提出了专门的核构造方法，并推导出刻画泛化性能的过剩风险界（excess risk bounds）。在合成数据集和真实数据集上的实证结果验证了所提方法的有效性。

    arXiv:2610.18196v1 Announce Type: new  Abstract: Hierarchical time series forecasting requires generating accurate predictions that are coherent across multiple levels of aggregation. Existing methods either rely on two-stage pipelines with post-hoc reconciliation or on end-to-end models that often lack statistical guarantees. We introduce a unified framework that formulates hierarchical forecasting as a multivariate regression problem in a vector-valued reproducing kernel Hilbert space. By designing structured operator-valued kernels, our approach enforces coherence directly within the hypothesis space, ensuring that all predictions satisfy hierarchical constraints by construction. We further propose tailored kernel constructions for hierarchical data and derive excess risk bounds that characterize generalization performance. Empirical results on synthetic and real-world datasets demonstrate the relevance of the proposed method.
    
[^15]: 将宽神经网络的有限样本高斯逼近提升为泛函收敛

    Lifting Finite-Sample Gaussian Approximation of Wide Neural Networks to Functional Convergence

    [https://arxiv.org/abs/2610.18166](https://arxiv.org/abs/2610.18166)

    本文提出将有限样本高斯逼近界提升为泛函收敛界的方法，使宽神经网络向高斯过程的 $\mathcal W_2$ 收敛界摆脱了对样本数量的依赖，从而适用于任意定义域上的稠密样本加密极限情形。

    

    宽度为 $m$ 的随机初始化神经网络在 $m \to \infty$ 时收敛到高斯过程，有两类定量界刻画了这种收敛。有限样本界在任意定义域上的固定有限样本点集处，以二次 Wasserstein 距离 $\mathcal W_2$ 比较网络与其高斯极限，在显式常数下达到 $m^{-1/2}$ 的收敛速度。然而，这些常数随样本数量增长，因此当样本点集被不断加密时，该界就失去了信息量。泛函界通过将两者作为随机函数进行比较而消除了这种依赖，但代价是要求输入位于球面上、使用弱于 $\mathcal W_2$ 的度量，或收敛速度更慢。因此，本文将一个有限样本界提升为泛函界。该提升过程需要将样本点集不断加密直至其在定义域中稠密，因此必须首先使该界与样本数量无关。我们通过度量……来实现这一点。

    arXiv:2610.18166v1 Announce Type: new  Abstract: Randomly initialised neural networks of width $m$ converge to Gaussian processes as $m \to \infty$, with two types of quantitative bounds characterising this convergence. Finite-sample bounds compare the network with its Gaussian limit at a fixed finite sample set, on an arbitrary domain, in the quadratic Wasserstein distance $\mathcal W_2$, achieving a convergence rate of $m^{-1/2}$ with explicit constants. However, these constants grow with the number of samples, so the bound becomes uninformative as the sample set is refined. Functional bounds remove that dependence by comparing the two as random functions, at the price of spherical input, metrics weaker than $\mathcal W_2$, or slower rates. In this paper, we therefore lift a finite-sample bound to a functional one. The lifting refines the sample set until it is dense in the domain, so the bound must first be made independent of the number of samples. We achieve this by measuring the 
    
[^16]: 从群元素的作用中学习群元素：伽罗瓦对应、最优经验风险最小化（ERM）与Natarajan维数的阿贝尔-鲁菲尼阈值

    Learning a Group Element from Its Action: Galois Correspondences, Optimal ERM, and an Abel-Ruffini Threshold for the Natarajan Dimension

    [https://arxiv.org/abs/2610.18154](https://arxiv.org/abs/2610.18154)

    该论文建立了从样本学习群元素与伽罗瓦理论之间的对应关系，将多类学习的 DS 维度、Littlestone 维度和教学维度化为群不变量，并证明在对称类上任何一致的 ERM 学习者均达到最优样本复杂度，同时揭示了 Natarajan 维数存在阿贝尔-鲁菲尼式的阈值。

    

    我们研究如何从样本 $(x, gx)$ 中学习一个在可数集合上以有限轨道作用的群中的未知元素 $g$。在集合 $A$ 上采样后，一致的假设是 $A$ 的逐点稳定子的某个陪集的限制，且样本恰好强制确定了该稳定子所固定点的标签；对于伽罗瓦群，这些一致的假设构成由数据生成的域。通过这一对应关系，多类学习的维度成为群的不变量：DS 维度和图维度等于轨道高度（height），Littlestone 维度等于最长无冗余序列的长度，教学维度等于最小基的大小。在确定性学习者中，对称性使经验风险最小化成为最优方法：在非退化类上，任何一致的学习者，无论其如何打破平局，都在常数因子内达到最优的可实现样本复杂度，且任何一致的在线学习者都达到最优的错误上界。在不可知（agnostic）情形下，每个 ERM 的样本复杂度……

    arXiv:2610.18154v1 Announce Type: cross  Abstract: We study learning an unknown element $g$ of a group acting with finite orbits on a countable set, from examples $(x,gx)$. After a sample on $A$, the consistent hypotheses are the restrictions of a coset of the pointwise stabilizer of $A$, and the sample forces exactly the labels of points this stabilizer fixes; for a Galois group, they form the field generated by the data. Through this correspondence, multiclass dimensions become group invariants: the DS and graph dimensions equal the height, the Littlestone dimension the longest irredundant sequence length, and the teaching dimension the minimal base size. Among deterministic learners, symmetry makes empirical risk minimization optimal: on nondegenerate classes every consistent learner, whatever its tie-breaking, attains the optimal realizable sample complexity up to constant factors, and every consistent online learner the optimal mistake bound. Agnostically, every ERM's sample compl
    
[^17]: 基于经验似然的可靠无尺度老虎机

    Reliable Scale-Free Bandits via Empirical Likelihood

    [https://arxiv.org/abs/2610.18055](https://arxiv.org/abs/2610.18055)

    该论文提出了 g-Forcing 策略的经验似然扩展 EL-F，在无需预知奖励尺度或尾部参数的情况下，为随机老虎机同时提供了几乎必然的渐近伪遗憾保证和有限时间的期望遗憾上界。

    

    我们研究在不知道奖励尺度或尾部参数的情况下的随机老虎机问题。几乎必然的遗憾保证描述了最终的学习效果，但无法量化有限次运行中的期望性能或出现大遗憾的风险。我们提出了 EL-F，这是 Cowan 和 Katehakis（2020）提出的 g-Forcing 策略的一个随时经验似然（EL）扩展，并在三个层次上刻画了其性能。首先，对于可积奖励以及非递减、发散且增量趋近于零的探索调度 g，EL-F 保持了 g-Forcing 的伪遗憾渐近性质：当 T→∞ 时，R(T)/g(T) 几乎必然收敛到所有次优臂的差距之和 ∑_{i:Δ_i>0} Δ_i，其中 Δ_i 是臂 i 的次优性差距。其次，在每个次指数型实例上，对于满足温和可行性条件的非递减函数 g(t)=ω(log t)，期望伪遗憾至多为 g(T)∑_{i:Δ_i>0} Δ_i + C_{P̲}，其中 C……（摘要在此处截断）

    arXiv:2610.18055v1 Announce Type: new  Abstract: We study stochastic bandits without knowledge of reward scales or tail parameters. Almost-sure regret guarantees describe eventual learning but do not quantify expected performance or the risk of large regret in a finite run. We introduce $\texttt{EL-F}$, an anytime empirical-likelihood (EL) extension of the $g$-Forcing policy of Cowan and Katehakis (2020), and characterize its performance at three levels. First, for integrable rewards and non-decreasing, diverging exploration schedules $g$ with vanishing increments, $\texttt{EL-F}$ preserves the $g$-Forcing asymptotics for pseudo-regret $R(T)$: $R(T)/g(T)\to\sum_{i:\Delta_i>0}\Delta_i$ almost surely as $T\to\infty$, where $\Delta_i$ is the suboptimality gap of arm $i$. Second, on every sub-exponential instance, for non-decreasing $g(t)=\omega(\log t)$ satisfying a mild feasibility condition, expected pseudo-regret is at most $g(T)\sum_{i:\Delta_i>0}\Delta_i+C_{\underline{P}}$, where $C_
    
[^18]: ndscape：嵌套二分法空间的结构地图

    ndscape: a structural map of the nested dichotomy space

    [https://arxiv.org/abs/2610.18050](https://arxiv.org/abs/2610.18050)

    提出了ndscape——一种基于深度加权结构标记的嵌套二分法结构二维映射方法，并首次对ND结构空间进行了偏差-方差分析，发现当节点使用逻辑回归时，更平坦、更平衡的树结构方差更小、更稳定，而使用决策树节点时方差与结构深度无关。

    

    嵌套二分法（Nested Dichotomies，NDs）将多类别分类问题简化为一棵由二元分类器组成的树。树结构可以影响准确率、偏差和方差，但完整的结构空间尚未被系统性地绘制。我们提出了ndscape，一种基于深度加权结构标记的ND结构二维映射方法。我们利用这一嵌入方法进行了据我们所知首次跨越ND结构空间的偏差-方差分析。在6到10个类别的模拟实验中，具有相似树结构的ND表现出相似的平方偏差、方差和平方概率损失，无论节点使用的二元分类器是逻辑回归还是决策树。然而，当每个节点使用逻辑回归时，方差遵循平均根到叶深度的规律，因此更平坦、更平衡的ND结构更加稳定；而当节点使用决策树模型时，方差与深度无关，且在不同ND结构之间差异很小。在六个数据（摘要在此处被截断）……

    arXiv:2610.18050v1 Announce Type: cross  Abstract: Nested dichotomies (NDs) reduce multiclass classification to a tree of binary classifiers. Tree structure can change accuracy, bias, and variance, but the full space of structures has not been mapped in aggregate. We introduce ndscape, a two-dimensional mapping of ND structures based on depth-weighted structural tokens. We use this embedding to conduct what is, to our knowledge, the first bias-variance analysis across the ND structure space. In simulations with 6 to 10 classes, NDs with similar tree structures exhibit similar squared bias, variance, and squared probability loss, regardless of whether the binary classifier is logistic regression or a decision tree. Variance, however, follows mean root-to-leaf depth when logistic regression is used at each node, so flatter, more balanced NDs are more stable, whereas with decision-tree node models, variance is not associated with depth and differs little across ND structures. On six datas
    
[^19]: 基于加权的分类策略：基于CART树的分析与局部峰值动态研究

    Weighting-Based Strategies for Classification: Analysis and Local Peak Dynamics with CART Trees

    [https://arxiv.org/abs/2610.18037](https://arxiv.org/abs/2610.18037)

    该论文从理论和实验上证明，在类别不平衡分类任务中，加权策略与不加权策略的预测性能本质上相同——渐近情况下两者仅相差一个阈值偏移，非渐近情况下二者在PR空间中沿同一曲线分布。

    

    表格学习任务常常受到类别不平衡问题的困扰，这一问题通常通过基于加权的策略来解决。在这项工作中，我们比较了朴素方法（不加权）与基于加权的再平衡策略。有些令人惊讶的是，我们证明了在渐近意义下，随着预测器复杂度随样本量增加，加权策略在所有评估指标上都能取得与不加权方法相同的预测性能，仅存在最终阈值的偏移。为了进一步深入研究，我们从理论上研究了通过有限深度的加权与不加权CART树对线性回归函数的估计问题。在这一设定中，我们证明了两种策略在PR空间（精确率-召回率空间）中的点分布在同一条曲线上，从而说明两种策略具有相似的非渐近性能。随后，我们通过数值实验识别出更复杂的合成设置，其中通过固定分类器评估的性能差距……

    arXiv:2610.18037v1 Announce Type: new  Abstract: Tabular learning tasks often suffer from class imbalance, which is often addressed via weighting-based strategies. In this work, we compare the naive approach (unweighted) with the weighting-based rebalancing strategies. Somewhat surprisingly, we demonstrate that, asymptotically, as the predictor complexity increases with sample size, weighting strategies yield predictive performance identical to unweighted approaches across all evaluation metrics, up to a shift in the final threshold. To go beyond, we theoretically study the estimation of a linear regression function through weighted and unweighted CART trees with limited depths. In this setting, we prove that the points in the PR space for both strategies are distributed along the same curve, thus illustrating the similar nonasymptotic performance of the two strategies. Then, we numerically identify more complex synthetic settings where performance gaps, evaluated via fixed classifier 
    
[^20]: 具有长期性能保证的序列式机器遗忘的反事实认证

    Counterfactual Certification for Sequential Unlearning with Long-Run Performance Guarantees

    [https://arxiv.org/abs/2610.17993](https://arxiv.org/abs/2610.17993)

    该论文提出一种针对序列式机器遗忘请求的反事实认证方法，在每次删除仅使用固定微调预算的情况下，对于光滑强凸目标函数可将性能损失约束在与请求次数无关的有界范围内。

    

    认证机器遗忘旨在确保未学习模型与在不含待遗忘数据的保留集上重新训练所得模型之间的接近性。将该框架扩展到按顺序到达的删除请求具有挑战性，这些请求通常以大小可变的组的形式出现（例如某一用户或一组协同用户的全部记录）。对于基于在保留集上进行微调的遗忘方法，随着遗忘请求数量的增长，这会导致性能下降或计算量增加。针对独立的连续请求，我们转而针对一种反事实遗忘轨迹来对每次模型发布进行认证——在该轨迹中，被删除的组从未参与过训练。在每次删除使用固定微调预算的条件下，我们证明：对于光滑的强凸目标函数，该方法能够使性能损失保持有界，且该界与请求总数无关（一致有界）。在FEMNIST数据集上对多类逻辑回归的实验表明，所得到的算法只需很少

    arXiv:2610.17993v1 Announce Type: cross  Abstract: Certified machine unlearning ensures the proximity between the unlearned model and a model retrained without the data to be forgotten. Extending this framework to sequential deletion requests, typically arriving as groups of variable size (e.g., all records of one user or of coordinated users), is challenging. For unlearning methods based on fine-tuning on the retain sets, this leads to either degrading performance or increasing computation as the number of unlearning requests grows. For independent successive requests, we instead certify each model release against a counterfactual unlearning trajectory in which the group being deleted never took part. With a fixed fine-tuning budget per deletion, we show that, for smooth strongly convex objectives, this approach keeps the performance loss bounded, uniformly in the number of requests. Experiments with multiclass logistic regression on FEMNIST show that the resulting algorithm needs few
    
[^21]: 从样本与质量中学习分布

    Learning Distributions from Samples and Qualities

    [https://arxiv.org/abs/2610.17956](https://arxiv.org/abs/2610.17956)

    该论文证明，当分布样本附带由同一未知参数决定的二元质量标签时，硬可行性约束可将参数估计从经典 $n^{-1/2}$ 速率加速至 $n^{-1}$，即使在质量模型存在未知加性常数时（通过构造无需阈值的成对偏好）也能保持这一加速，并揭示了逐点质量与成对偏好在信息量上互不可比。

    

    我们研究这样的参数估计问题：来自连续参数分布的每个样本都附带一个由同一未知参数控制的二元质量标签。通过施加硬可行性约束，确定性标签可以将估计速率从经典的光滑质量感知M-估计器的 $n^{-1/2}$ 加速到 $n^{-1}$。结合PAC保证与局部误差到参数的转换，我们对可识别的质量分类器和有限歧义的质量分类器都建立了这一速率，其中似然信息可以消除后者的歧义。当质量模型仅已知到一个加性常数时，我们从具有相反标签的样本中构造出无需阈值的成对偏好，并在阈值未知的情况下仍然保持加速速率。我们进一步表明，逐点质量与成对偏好之间是不可比的：任何一方都可能严格地比另一方更具信息量。我们的假设在高斯分布、拉普拉斯分布等情形下均成立。

    arXiv:2610.17956v1 Announce Type: new  Abstract: We study parameter estimation when each sample from a continuous parametric distribution is accompanied by a binary quality label governed by the same unknown parameter. By imposing hard feasibility constraints, deterministic labels can accelerate estimation from the classical $n^{-1/2}$ rate of smooth quality-aware M-estimators to $n^{-1}$. Combining PAC guarantees with local error-to-parameter conversions, we establish this rate for both identifiable and finitely ambiguous quality classifiers, with likelihood information resolving the latter ambiguity. When the quality model is known only up to an additive constant, we construct threshold-free pairwise preferences from oppositely labeled samples and retain the accelerated rate despite the unknown threshold. We further show that pointwise qualities and pairwise preferences are incomparable: either can be strictly more informative than the other. Our assumptions hold for Gaussian, Laplac
    
[^22]: 线性赌博机的扰动跟随领导者算法：一般凸集上的 $\sqrt{T}$ 遗憾

    Follow the Perturbed Leader for Linear Bandits: $\sqrt{T}$ Regret over General Convex Sets

    [https://arxiv.org/abs/2610.17818](https://arxiv.org/abs/2610.17818)

    本文提出了首个基于FTPL的对抗性线性赌博机算法，通过一个适应估计奖励的扰动分布，在一般凸动作集上实现了 $\sqrt{T}$ 遗憾，改进了此前 $T^{2/3}$ 的遗憾界，同时比连续指数权重方法具有更好的计算效率和更简单的实现。

    

    我们提出了首个基于扰动跟随领导者（Follow-the-Perturbed-Leader, FTPL）的对抗性线性赌博机算法，该算法在一般凸动作集上实现了 $\sqrt{T}$ 遗憾。我们的算法仅使用线性优化预言机即可达到 $\widetilde{\mathcal{O}}(d^{5/4}\sqrt{T})$ 的遗憾；当进一步可获得最优设计探索分布时，遗憾可达到接近最优的 $\widetilde{\mathcal{O}}(d\sqrt{T})$。这改进了 Dani 和 Hayes（2006）的 FTPL 算法在一般凸集上 $T^{2/3}$ 的遗憾，并将 Lévy 等人（2026）在两个特殊动作集（超立方体和欧几里得球）上的 $\sqrt{T}$ 遗憾结果进行了推广。与之前基于连续指数权重的预言机高效线性赌博机算法相比，FTPL 方法通常具有更好的计算效率，且实现起来要简单得多。关键的算法要素是一个能适应估计奖励的扰动分布。

    arXiv:2610.17818v1 Announce Type: cross  Abstract: We give the first adversarial linear bandit algorithm based on Follow-the-Perturbed-Leader (FTPL) that achieves $\sqrt{T}$ regret over general convex action sets. Our algorithm attains $\widetilde{\mathcal{O}}(d^{5/4}\sqrt{T})$ regret using only linear optimization oracles, and the near-optimal $\widetilde{\mathcal{O}}(d\sqrt{T})$ regret when an optimal-design exploration distribution is further available. This improves on the $T^{2/3}$ regret of the FTPL algorithm of Dani and Hayes (2006) for general convex sets, and generalizes the $\sqrt{T}$ regret of L\'evy et al. (2026) over two special action sets (hypercube and Euclidean ball). Compared to previous oracle-efficient linear bandit algorithms based on continuous exponential weights, the FTPL approach generally has better computational efficiency, and is much simpler to implement.   The key algorithmic element is a perturbation distribution that adapts to the estimated rewards. Its 
    
[^23]: 超越协方差的线性回归：来自高阶累积量的潜在斜率谱

    Linear Regression Beyond Covariance: Latent Slope Spectra from Higher-Order Cumulants

    [https://arxiv.org/abs/2610.17772](https://arxiv.org/abs/2610.17772)

    这篇论文提出"潜在斜率谱"的新框架，证明普通最小二乘估计只是斜率谱的方差加权重心，并利用高阶累积量在谱足够简约时分离出被传统协方差信息掩盖的不同线性效应成分，实现多重斜率的识别与根号n推断。

    

    最小二乘法通过二阶信息概括响应变量与回归变量的联合分布。当观测到的回归变量由与响应变量具有不同关系的潜在成分构成时，这种概括可能对不同线性效应进行平均。我们研究了高阶混合累积量中包含的额外线性信息，并通过潜在斜率谱将其形式化。普通最小二乘法是该谱的方差加权重心，而当谱足够简约时，高阶累积量可以分离其各个成分。在标量单混杂因子模型中，四阶累积量通过一个2×2线性系统和一个二次方程识别无序对 $\{\beta,\beta+\gamma\}$。我们在规则分离条件下发展了估计方法和根号n推断，并证明在退化情形附近检测与估计出现分化：斜率的多重性在…（摘要在此处截断）

    arXiv:2610.17772v1 Announce Type: cross  Abstract: Least squares summarizes the joint distribution of a response and regressor through second-order information. When an observed regressor is composed of latent components carrying different relationships with the response, this summary can average over distinct linear effects. We study the additional linear information contained in higher-order mixed cumulants and formalize it through the latent slope spectrum. Ordinary least squares is a variance-weighted barycenter of this spectrum, whereas higher-order cumulants can separate its components when the spectrum is sufficiently parsimonious. In the scalar one-confounder model, fourth-order cumulants identify the unordered pair $\{\beta,\beta+\gamma\}$ through a $2\times2$ linear system and a quadratic equation. We develop estimation and root-$n$ inference under regular separation and show that detection and estimation diverge near degeneracy: slope multiplicity remains detectable on an $n
    
[^24]: 基于选择性验证反馈的高斯过程老虎机用于AI系统优化

    Gaussian Process Bandits for AI System Optimization with Selectively Verified Feedback

    [https://arxiv.org/abs/2610.17760](https://arxiv.org/abs/2610.17760)

    该论文针对AI系统优化中模型评估与可信结果不一致的问题，提出了带选择性验证反馈的高斯过程老虎机方法，通过固定审计设计与安全自适应扩展，在硬性审计预算下建立了有限时间遗憾界。

    

    高斯过程老虎机利用配置之间的相似性来优化提示词、检索系统和智能体工作流。然而，在这些应用中，廉价的基于模型的评估可能与可信的任务结果不一致。我们研究了选择性验证问题，即在观察到诊断信息之前先选择动作，且可信反馈仅在未知的诊断状态集合上提供有效信息。我们用有限可靠性类和有限RKHS信号字典来刻画这两种不确定性来源。我们构造了一种固定审计设计，将可靠性覆盖与奖励分离相结合，并建立了考虑硬性审计预算以及获取信息性观测所需轮次的有限时间遗憾界。一种安全的自适应扩展将该鲁棒设计与可预测的Chernoff分配相结合，同时保持统一的识别下限和精确的期望获取成本核算。对于带符号坐标核……（摘要在此处截断）

    arXiv:2610.17760v1 Announce Type: cross  Abstract: Gaussian process bandits exploit similarities among configurations to optimize prompts, retrieval systems, and agent workflows. In these applications, however, inexpensive model-based evaluations can disagree with trusted task outcomes. We study selective verification when an action is chosen before a diagnostic is observed and trusted feedback is informative on an unknown set of diagnostic states. A finite reliability class and a finite RKHS signal dictionary describe these two sources of uncertainty. We construct a fixed audit design that combines reliability coverage with reward separation and establish a finite-time regret bound that accounts for a hard audit budget and the rounds needed to obtain informative observations. A safe adaptive extension mixes this robust design with predictable Chernoff allocations while preserving a uniform identification floor and exact expected acquisition-cost accounting. For signed-coordinate kerne
    
[^25]: 面向已实现GARCH的结构化随机微分方程：连续时间波动率建模与成分级分析

    Structured SDEs for Realized GARCH: Continuous-Time Volatility Modelling and Component-Level Analysis

    [https://arxiv.org/abs/2610.17732](https://arxiv.org/abs/2610.17732)

    该论文提出一种结构化连续时间随机波动率模型，通过已实现测度状态直接反馈到潜在方差动态中来预测积分方差，在17个股票指数上展现出优于GARCH、Realized GARCH等基准模型的样本外预测性能。

    

    我们提出了一种结构化连续时间随机波动率模型，用于提前一天的积分方差预测。该模型引入了一个联合演化的已实现测度状态，直接输入到潜在方差动态中，同时相关的布朗运动创新项捕捉与收益相关的依赖结构。模型参数通过基于模拟的预测学习方法进行估计，并通过蒙特卡洛模拟预测积分方差。在17个股票指数上的实证结果表明，该模型在QLIKE、MSE和MAE指标下，相对于GARCH、GARCH-Itô、Realized GARCH、HAR和HEAVY等基准模型展现出强大的样本外预测性能。消融实验显示，已实现测度反馈在所提出的框架内提供了可观的增量预测价值，而与收益相关的布朗依赖关系则带来了较为温和的补充收益。这些结果支持将测量驱动的信息直接纳入连续时间潜在波动率建模中。

    arXiv:2610.17732v1 Announce Type: cross  Abstract: We propose a structured continuous-time stochastic volatility model for one-day-ahead integrated-variance forecasting. The model introduces a jointly evolving realized-measure state that feeds directly into latent variance dynamics, while correlated Brownian innovations capture return-linked dependence. Parameters are estimated through simulation-based predictive learning, and Monte Carlo simulation forecasts integrated variance. Across 17 equity indices, the proposed model shows strong out-of-sample performance relative to GARCH, GARCH-It\^o, Realized GARCH, HAR, and HEAVY benchmarks under QLIKE, MSE, and MAE. Ablation results show that realized-measure feedback provides substantial incremental forecasting value within the proposed framework, while return-linked Brownian dependence contributes more modest complementary gains. The results support incorporating measurement-driven information directly into continuous-time latent volatili
    
[^26]: textGrain 水印更难检测：求和检验还能实现最优检测吗？

    textGrain watermarking is harder to detect: can sum-tests still do it optimally?

    [https://arxiv.org/abs/2610.17655](https://arxiv.org/abs/2610.17655)

    textGrain 水印的熵校准机制使其极小极大检测边界为 p+q=1/2，可检测区域严格小于边界为 2p+q=1 的 Gumbel-max 与最大耦合水印，表明熵校准是以牺牲可检测性为代价的。

    

    水印技术在大语言模型生成的文本中嵌入一种不可感知的信号，借助密钥即可对其进行统计检测。近期的 textGrain 水印（Li 等人，2026b）通过显式的熵预算将该信号与模型的采样熵进行校准，融合了红-绿（Red-Green）水印的随机性与 Gumbel-max 水印的无偏性，其配套报告提出了一种对关键统计量求和的检测器。我们研究了在人工编辑之后，textGrain 何时以及能以多高的效率被检测到，其中人工编辑被建模为稀疏混合模型：比例为 $\varepsilon_n\asymp n^{-p}$ 的位置保留了水印，且 token 的不确定性以 $n^{-q}$ 的速度衰减。首先，熵校准是有代价的：textGrain 的极小极大检测边界为 $p+q=1/2$，因此其可检测区域严格小于 Gumbel-max 与最大耦合水印的可检测区域，后者的边界为 $2p+q=1$。这一代价源于 textGrain 的构造方式……

    arXiv:2610.17655v1 Announce Type: new  Abstract: Watermarking embeds an imperceptible signal in text from large language models that is statistically detectable with a secret key. The recent textGrain watermark (Li et. al. (2026b)) calibrates this signal against the model's sampling entropy through an explicit entropy budget, combining the stochasticity of Red-Green and the unbiasedness of Gumbel-max watermarks, and its accompanying report proposes a detector that sums pivotal statistics. We study when, and how efficiently, textGrain can be detected after human edits, modeled as a sparse mixture in which a fraction $\varepsilon_n\asymp n^{-p}$ of positions retain the watermark and token uncertainty decays as $n^{-q}$. First, entropy calibration comes at a cost: the minimax detection boundary of textGrain is $p+q=1/2$, so its detectable region is strictly smaller than that of Gumbel-max and maximal-coupling watermarks, whose boundary is $2p+q=1$. This cost stems from textGrain's constru
    
[^27]: 具有有限样本保证的风险规避推理时扩散模型对齐

    Risk-Averse Inference-Time Diffusion Model Alignment with Finite-Sample Guarantees

    [https://arxiv.org/abs/2610.17579](https://arxiv.org/abs/2610.17579)

    该论文提出了一个风险规避的扩散模型推理时对齐框架，在轨迹级KL正则化下平衡期望代价与条件风险价值（CVaR），将对齐问题化简为关于CVaR阈值的一维优化，并给出了阈值优化的有限样本理论保证。

    

    推理时对齐无需代价高昂的微调即可引导扩散模型实现下游目标，但现有方法通常只优化期望代价（或奖励），而未显式控制尾部风险。在本工作中，我们提出了一种用于扩散模型推理时对齐的风险规避框架，该框架在轨迹级Kullback-Leibler正则化下平衡期望代价与条件风险价值（CVaR）。我们将最优路径分布和反向转移刻画为CVaR阈值的函数，从而将对齐问题简化为关于该阈值的一维优化问题。我们证明该一维优化问题在一般情况下可能是非凸的，建立了凸性的充分条件，并引入二次正则化以实现高效的无导数三元搜索。在适当的假设下，我们为阈值优化提供了有限样本保证，并验证了…

    arXiv:2610.17579v1 Announce Type: new  Abstract: Inference-time alignment steers diffusion models toward downstream objectives without costly fine-tuning, but existing methods typically optimize expected cost (or reward) without explicitly controlling tail risk. In this work, we propose a risk-averse framework for inference-time alignment of diffusion models that balances expected cost and Conditional Value-at-Risk (CVaR) under trajectory-level Kullback-Leibler regularization. We characterize the optimal path distribution and reverse transitions as functions of the CVaR threshold, reducing the alignment problem to a one-dimensional optimization problem over the threshold. We show that this one-dimensional optimization problem can be nonconvex in general, establish sufficient conditions for convexity, and introduce quadratic regularization to enable efficient derivative-free ternary search. Under suitable assumptions, we provide finite-sample guarantees for threshold optimization and va
    
[^28]: 球面参数空间对非遗忘任务数量与持续学习能力的影响

    Effect of Spherical Parameter Space on Number of Non-Forgetting Tasks and Capacity of Continual Learning

    [https://arxiv.org/abs/2610.17453](https://arxiv.org/abs/2610.17453)

    该论文提出了一种基于高维球面参数空间的持续学习理论框架，通过测地球刻画任务的可接受损失区域，并利用 Neyman-Scott 聚类点过程与 von Mises-Fisher 分布建模任务相似性，从而分析球面参数空间对可容纳的非遗忘任务数量及持续学习容量的影响。

    

    我们提出了一个针对 $T$ 个非遗忘任务的持续学习形式化框架，其参数空间为球面 $\mathbb{S}^{d-1}(\sqrt{d})$，并考虑 $d\rightarrow \infty$ 的渐近情形。每个任务 $i$ 都有一个最优解 $\bm{\Theta}_i$，用于最小化其在球面上对应的损失函数 $R_i(\bm{w})$。我们将每个任务 $i$ 简化为一个可接受的损失区域，该区域由以 $\bm{\Theta}_i$ 为中心、半径为 $r_i(d)$ 的测地球 $B_{\bm{\Theta}_i}(r_i(d))$ 在 $\mathbb{S}^{d-1}(\sqrt{d})$ 上建模。为了刻画任务间的依赖关系（相似性），任务最优解 $\bm{\Theta}_i$ 按照 Neyman-Scott 聚类点过程 $\Phi$ 分布：聚类中心 $\{\bm{Y}_j\}_{j=1}^{N_p}$ 构成球面上的齐次泊松点过程，而簇内的任务最优解则服从以 $\bm{Y}_j$ 为均值方向、集中参数为 $\kappa$ 的 von Mises-Fisher 分布。每个任务 $t\in \{1, ..., T\}$ 的学习算法被视为一个信道 $Q_{t}(\mathrm{d}\bm{w}_t | \bm{\cdot})$（摘要在此处截断）。

    arXiv:2610.17453v1 Announce Type: cross  Abstract: We propose a continuous learning formulation for $T$ non-forgetting tasks with the spherical parameter space $\mathbb{S}^{d-1}(\sqrt{d})$ in asymptotic regime $d\rightarrow \infty$. Each task $i$ has an optimum $\bm{\Theta}_i$, minimizing its corresponding loss function $R_i(\bm{w})$ on sphere. We reduce each task $i$ to an acceptable loss region modeled by a geodesic ball $B_{\bm{\Theta}_i}(r_i(d))$ on $\mathbb{S}^{d-1}(\sqrt{d})$ centered at $\bm{\Theta}_i$ with radius $r_i(d)$. To model task dependencies (similarities), task optima $\bm{\Theta}_i$ are distributed by a Neyman-Scott clustered point process $\Phi$. Cluster centers $\{\bm{Y}_j\}_{j=1}^{N_p}$ form a homogeneous Poisson point process on sphere, while intra-cluster task optima follow a von Mises-Fisher distribution with mean $\bm{Y}_j$ and concentration $\kappa$. Each task $t\in \{1, ..., T\}$ learning algorithm is considered to be a channel $Q_{t}(\mathrm{d}\bm{w}_t | \bm
    
[^29]: 通过投注式监测学习检测非结构化数据流中的分布偏移

    Learning to Detect Shifts in Unstructured Data Streams via Monitoring-by-Betting

    [https://arxiv.org/abs/2610.17399](https://arxiv.org/abs/2610.17399)

    提出一种基于投注式检验的监控框架，通过将序贯变点检测建模为投注博弈并自适应更新偏移方向假设，解决了非结构化数据流分布偏移检测中敏感性与广度的权衡问题，同时保持误报率的统计保证。

    

    对生成式AI系统的持续监控需要检测其输入和生成的非结构化数据中的分布偏移。现有的监控方法通常通过要求预先指定用于检测偏移的差异度量来控制误报。然而，对于非结构化数据，预先指定会造成敏感性—广度之间的权衡：定义狭窄的度量对预期偏移敏感，但可能遗漏意外变化；而定义宽泛的度量虽能检测更广范围的偏移，敏感性却会降低。尽管自适应地假设偏移方向可以克服这一局限，但这种依赖数据的自适应通常会使传统的误报保证失效。我们通过投注式检验将自适应性与统计有效性调和起来，将序贯变点检测重新构建为一场投注博弈，其中投注者不断更新其假设的偏移方向。

    arXiv:2610.17399v1 Announce Type: cross  Abstract: Continuous monitoring of generative AI systems requires detecting distribution shifts in the unstructured data they ingest and generate. Existing monitoring methods typically control false alarms by requiring the discrepancy measure used to detect shifts to be *pre-specified*. However, for unstructured data, pre-specification creates a sensitivity--breadth trade-off: a narrowly defined measure is sensitive to the anticipated shift but may miss unexpected changes, whereas a broadly defined measure may detect a wider range of shifts but with reduced sensitivity. Although adaptively hypothesizing the shift direction can overcome this limitation, such data-dependent adaptation generally invalidates conventional false-alarm guarantees. We reconcile adaptivity with statistical validity through testing-by-betting, reframing sequential changepoint detection as a betting game in which the bettor continually updates its hypothesized shift direct
    
[^30]: 测度神经映射

    Measure Neural Mappings

    [https://arxiv.org/abs/2610.17384](https://arxiv.org/abs/2610.17384)

    本文提出测度神经映射（MNMs）新框架，通过集成一致性将依赖测度的神经传输与自注意力机制结合，证明了Transformer在非紧概率测度上的定量连续统极限，其输出Wasserstein误差由经验输入误差加上 $N^{-1/2}$ 注意力项界定，可应用于贝叶斯推断等无界状态空间场景。

    

    我们引入并定义了测度神经映射（MNMs），它将依赖测度的神经传输与集成一致性相结合，用于学习概率测度上的算子。集成一致性要求从输入分布中采样的规模不断增大的集成，其输出依概率收敛到指定的总体输出。在强次高斯和尾部可积性假设下，我们为非紧分布上的注意力机制建立了定量连续统极限。这些分布适用于科学机器学习中的无界状态空间，包括贝叶斯推断和不确定性量化。对于固定的自注意力Transformer，期望输出Wasserstein误差由经验输入Wasserstein误差加上一个额外的 $N^{-1/2}$ 注意力项所界定。在所述假设下，这些自注意力Transformer即为MNMs：pre-norm架构源于归一化后的紧稳定性，而post-norm架构……（摘要在此处截断）

    arXiv:2610.17384v1 Announce Type: cross  Abstract: We introduce and define measure neural mappings (MNMs), which combine measure-dependent neural transport with ensemble consistency for learning operators on probability measures. Ensemble consistency requires outputs from growing ensembles sampled from an input law to converge to the specified population output in probability. We establish quantitative continuum limits for attention on noncompact laws under strong sub-Gaussian and tail-integrability assumptions. These laws accommodate unbounded state spaces in scientific machine learning, including Bayesian inference and uncertainty quantification. For fixed self-attention transformers, the expected output Wasserstein error is bounded by the empirical input Wasserstein error plus an additional $N^{-1/2}$ attention term. These self-attention transformers are MNMs under the stated assumptions: pre-norm architectures follow from compact stability after normalization, while post-norm archi
    
[^31]: 论注意力的统计力学

    On the Statistical Mechanics of Attention

    [https://arxiv.org/abs/2610.17347](https://arxiv.org/abs/2610.17347)

    本文借助现代Hopfield模型的自由能形式化方法，建立了刻画Transformer注意力相结构与推理可靠性的严格统计力学框架，并揭示出注意力在玻璃态与稳健推理相之间存在一阶相变。

    

    我们借助现代Hopfield模型重新审视Transformer注意力的统计力学。我们提出了一个自由能形式化方法，用以刻画稳健注意力推理的参数区域和临界阈值。所得的自由能景观包含了查询状态和注意力权重中的随机性。通过标准的最小自由能原理，注意力作为相应统计力学系统（即现代Hopfield模型）的运动方程而自然涌现。这使我们能够识别注意力在玻璃态与稳健推理相之间的一阶相变。我们进一步将该方法扩展到具有随机查询和键的注意力机制。总体而言，我们的结果为表征一系列Transformer注意力机制的相结构和推理可靠性提供了一个严格的统计力学框架。

    arXiv:2610.17347v1 Announce Type: cross  Abstract: We revisit the statistical mechanics of Transformer attention via modern Hopfield models. We present a free-energy formalism to characterize parameter regimes and critical thresholds for robust attention inference. The resulting free-energy landscape incorporates randomness in query states and attention weights. Attention then emerges as the equation of motion of the corresponding statistical-mechanical system (i.e., modern Hopfield model) through the standard free-energy minimization principle. This allows us to identify first-order phase transitions in attention between glassy and robust-inference phases. We further extend to attentions with stochastic queries and keys. Together, our results provide a rigorous statistical-mechanics framework for characterizing phase structure and inference reliability across a family of Transformer attention mechanisms.
    
[^32]: 在未知时间监测分布偏移：无参数条件保形鞅

    Monitoring for Shifts at Unknown Times: Parameter-Free Conditional Conformal Martingales

    [https://arxiv.org/abs/2610.17343](https://arxiv.org/abs/2610.17343)

    该论文提出一种无超参数的条件保形鞅方法，通过将惩罚下注问题转化为三资产市场的投资组合选择问题，并采用带重尾财富分配的重启通用投资组合，实现了对未知时间发生的分布偏移的快速自适应检测，同时保证任意时刻虚警控制和最优的有限样本检测延迟界。

    

    条件保形检验鞅为检测分布偏移提供了序贯检验方法，它使用固定的参考数据集来避免测试时样本的污染。先前的工作通过在线优化使用带有平滑、裁剪等超参数的惩罚下注函数，当偏移在长期平稳期之后发生时，这种方法可能适应缓慢。在这项工作中，我们引入了一种不需要超参数的方法，并且可以证明它能够快速适应在未知时间发生的偏移。我们的核心思想是将惩罚下注问题构建为三资产市场中的投资组合选择问题。然后，我们使用一种重启的通用投资组合，并在时间上进行重尾的财富分配。我们建立了任意时刻的虚警控制，并在可检测的备择假设下推导出了检测延迟的最优有限样本界。我们还开发了一种基于FFT的实现，可以在T轮内计算投资组合序列。

    arXiv:2610.17343v1 Announce Type: new  Abstract: Conditional conformal test martingales provide sequential tests for detecting distribution shifts, using a fixed reference dataset to avoid contamination from test-time samples. Prior work used a penalized betting function via online optimization with smoothing, clipping, and other hyperparameters, which can adapt slowly when a shift occurs after a long stationary period. In this work, we introduce an approach that requires no hyperparameters and provably adapts quickly to shifts occurring at unknown times. Our key idea is to frame the penalized betting problem as portfolio selection in a three-asset market. Then, we use a restarting universal portfolio, with a heavy-tailed wealth allocation over time. We establish anytime false-alarm control and derive optimal finite-sample bounds on detection delay under detectable alternatives. We also develop an FFT-based implementation that computes the portfolio sequence over $T$ rounds in $\mathca
    
[^33]: 通过学习的生成式动力学在集合变分数据同化中刻画模式误差

    Accounting for model error in ensemble variational data assimilation through learned generative dynamics

    [https://arxiv.org/abs/2610.17333](https://arxiv.org/abs/2610.17333)

    提出生成式迭代集合卡尔曼滤波器（IEnKF-G），用学习到的条件随机生成预报模型取代传统设定的加性高斯模式误差表示，在新颖的单步弱约束4D-Var框架中更好地刻画复杂模式误差。

    

    近年来，许多方法被提出用于将数据同化与机器学习相结合，以改进地球物理模式的预报。本文着重于将先进的顺序数据同化方案与一个能够刻画复杂模式误差的生成式预报模型相结合。该生成模型在给定当前状态的条件下，对后继状态的条件分布提供随机近似。所提出的变分形式是一种新颖的单步弱约束4D-Var方案，其中用学习到的条件随机预报取代了预先设定的加性高斯模式误差表示。我们将该形式推导并实现为一种集合变分算法——生成式迭代集合卡尔曼滤波器，它推广了带有加性高斯模式误差的迭代集合卡尔曼滤波器。我们进一步将该方法扩展为同化窗口跨越（原文摘要在此处截断）

    arXiv:2610.17333v1 Announce Type: new  Abstract: Many approaches have recently been proposed to combine data assimilation with machine learning for improved forecasts with geophysical models. Here, we focus on combining advanced sequential data assimilation schemes with a generative forecasting model which accounts for complex model errors. This generative model provides a stochastic approximation of the conditional distribution of the successor state given the current state. The proposed variational formulation is a novel one-step weak-constraint 4D-Var scheme in which a learned conditional stochastic forecast replaces the prescribed additive Gaussian model error representation. We derive and implement this formulation as an ensemble variational algorithm, the generative iterative ensemble Kalman filter (IEnKF-G), which generalises the iterative ensemble Kalman filter with additive Gaussian model error (IEnKF-Q). We extend the method to a smoother with an assimilation window spanning 
    
[^34]: 半臂反馈下单调随机优化的近最优后悔界

    Near-Optimal Regret Bounds for Monotone Stochastic Optimization under Semi-Bandit Feedback

    [https://arxiv.org/abs/2610.17320](https://arxiv.org/abs/2610.17320)

    本文提出一种通用的、预言机高效的在线到离线归约，使强单调随机优化在半臂反馈下的期望后悔界达到近最优的 $\widetilde O(\sqrt{nT})$，从而回答了该领域的一个开放性问题。

    

    随机优化研究不确定性下的序贯决策问题，通常假设输入分布是已知的。当这些分布未知时，Agarwal等人（2026）针对半臂反馈下的单调随机优化（即仅观测被探测变量的取值）提出了一种从在线到离线的归约方法，其相对于离线保证的期望后悔度为 $\widetilde O(n\sqrt T)$，其中 $n$ 是变量数量，$T$ 是时间范围。对于最优停止问题和潘多拉盒子问题，Liu等人（2025）利用针对具体问题的算法达到了 $\widetilde O(\sqrt{nT})$ 的界。那么，一个通用的在线到离线归约能否达到这一更优的速率？我们对强单调随机优化给出了这样一个归约，回答了相应的开放性问题。强单调性是相对于所选离线算法定义的。我们提出的预言机高效（oracle-efficient）归约实现了 $\widetilde O(\sqrt{nT})$ 的期望后悔界。

    arXiv:2610.17320v1 Announce Type: cross  Abstract: Stochastic optimization models sequential decision-making under uncertainty, typically assuming known input distributions. When these distributions are unknown, \citet{agarwal2026semibandit} give an online-to-offline reduction for monotone stochastic optimization under semi-bandit feedback, where only probed variables are observed. Its expected regret relative to the offline guarantee is $\widetilde O(n\sqrt T)$, where $n$ is the number of variables and $T$ the horizon. For optimal stopping and Pandora's box, \citet{liu2025improved} achieve $\widetilde O(\sqrt{nT})$ using problem-specific algorithms. \emph{Can a general online-to-offline reduction achieve this improved rate?}   We give such a reduction for strongly monotone stochastic optimization, answering the corresponding open question. Strong monotonicity is defined relative to the chosen offline algorithm. Our oracle-efficient reduction achieves $\widetilde O(\sqrt{nT})$ expected
    
[^35]: 用于相位恢复的深度展开交替投影方法

    Deeply unrolled alternating projections for phase retrieval

    [https://arxiv.org/abs/2610.17316](https://arxiv.org/abs/2610.17316)

    提出了一种将学习到的去噪步骤与交替投影方法相结合的深度展开框架用于相位恢复，在测量算子动态范围显著变化等场景下无需重新训练即可获得高质量重建。

    

    针对逆问题的深度展开方法通过学习到的先验来增强传统重建算法，以提升其性能。这些方法的开发基于一个隐含假设，即测量算子的几何结构在训练和部署之间保持不变；当这一假设不再成立时，所获得重建结果的质量可能会显著下降。在本文中，我们论证了这并非深度展开范式的固有局限，并且可以通过仔细选择底层迭代算法来缓解这一问题。我们提出了一种针对相位恢复问题的深度展开框架，该框架在交替投影方法的基础上增加了一个学习得到的去噪步骤。在传统基于模型的替代方法难以应对的场景中（例如测量算子的动态范围发生显著变化时），我们的方法无需重新训练即可获得高质量的重建结果。

    arXiv:2610.17316v1 Announce Type: new  Abstract: Deep unrolling methods for inverse problems augment traditional reconstruction algorithms with learned priors to improve their performance. These methods have been developed under the implicit assumption that the geometry of the measurement operator does not change between training and deployment; when this is no longer the case, the quality of obtained reconstructions can degrade significantly. In this paper, we argue that this is not an inherent limitation of the deep unrolling paradigm and can be mitigated by a careful choice of the underlying iterative algorithm. We propose a deep unrolling framework for the phase retrieval problem that augments the alternating projections method with a learned denoising step. Our method obtains high-quality reconstructions without the need for retraining in settings where conventional model-based alternatives struggle, such as when the dynamic range of the measurement operator changes significantly 
    
[^36]: 带自回归思维链的单个神经元的惊人表达能力

    The Surprising Expressivity of a Single Neuron with Autoregressive Chain-of-Thought

    [https://arxiv.org/abs/2610.17283](https://arxiv.org/abs/2610.17283)

    该论文证明一个使用阈值或ReLU等激活函数的单个自回归神经元，即可通过其生成的思维链模拟任意前馈电路乃至多项式时间图灵机，所需维度为O(ns+s²)、生成步数为O(s²)，且两个界均与电路深度无关并在此前指数级结果的基础上有显著改进。

    

    使用思维链监督训练的自回归语言模型的成功提出了一个根本性问题：下一个词预测器可以有多简单，同时仍能通过其自回归生成的思维链高效地模拟通用计算？令人惊讶的是，我们证明单个神经元就足够了。对于包括阈值函数和ReLU在内的广泛激活函数家族，单个自回归神经元可以模拟任何具有相同激活函数的前馈电路。具体而言，一个在 $n$ 个输入上、规模为 $s$ 的电路，可以由一个维度为 $O(ns+s^2)$ 的神经元通过 $O(s^2)$ 步生成来模拟，且这两个界均与电路深度无关，并在自然假设下是紧致的。对于阈值激活函数，这改进了 Joshi 等人（COLT 2025）的模拟结果，后者的界随电路深度呈指数增长。因此，自回归阈值神经元可以模拟多项式时间的图灵机……

    arXiv:2610.17283v1 Announce Type: cross  Abstract: The success of autoregressive language models trained with Chain-of-Thought~(CoT) supervision raises a fundamental question: How simple can a next-token predictor be while still enabling the efficient simulation of general computation through its autoregressively generated CoT? Surprisingly, we show that a single neuron suffices. For a broad family of activations, including threshold and ReLU, a single autoregressive neuron can simulate any feedforward circuit with the same activation. Specifically, a size-$s$ circuit on $n$ inputs can be simulated by a neuron of dimension $O(ns+s^2)$ using $O(s^2)$ generation steps, with both bounds independent of circuit depth and tight under natural assumptions. For threshold activations, this improves upon the simulation of Joshi et al. (COLT 2025), whose bounds scale exponentially with circuit depth. Consequently, autoregressive threshold neurons can simulate polynomial-time Turing machines, while
    
[^37]: 差分隐私合成数据的紧致端到端隐私核算

    Tight End-to-End Accounting for Differentially Private Synthetic Data

    [https://arxiv.org/abs/2610.17249](https://arxiv.org/abs/2610.17249)

    本文开发了基于隐私损失分布（PLD）、f-DP和高斯差分隐私（GDP）的紧致端到端隐私核算工具，并提出了GDP原生的AIM变体AIM-GDP，实现了差分隐私合成数据流水线中异构机制的精确组合分析。

    

    差分隐私（DP）流水线通常组合多种异构机制，而这些机制的隐私保证最好使用不同的隐私定义来分析（如有界范围、zCDP、ADP）。在组合这些机制时，我们必须通过一种简化的通用表示来分析其保证，这会导致核算精度的损失。这一问题在DP合成数据生成中尤为突出，因为数据预处理、生成算法和DP-SGD可能同时出现在同一流水线中。我们开发了新颖的工具，利用隐私损失分布（PLD）、$f$-DP和高斯差分隐私（GDP）对这类组合进行紧致的端到端核算。特别地，我们推导了诸如指数机制等有界范围机制的PLD和GDP分析。基于这些理论结果，我们提出了AIM-GDP，这是AIM（一种最先进的DP合成数据机制）的GDP原生变体，可通过PLD实现与复杂流水线的便捷组合，……

    arXiv:2610.17249v1 Announce Type: cross  Abstract: Differentially private (DP) pipelines often combine heterogeneous mechanisms, yet their guarantees are best analyzed using different notions of privacy (e.g., bounded range, zCDP, ADP). When composing them, we have to analyze the guarantees through a reduced common representation that loses accounting precision. This issue is particularly relevant in DP synthetic data generation, where data preprocessing, generation algorithms, and DP-SGD could appear within the same pipeline. We develop novel tools for tight end-to-end accounting of such compositions using privacy loss distributions (PLD), $f$-DP, and Gaussian DP (GDP). In particular, we derive PLD and GDP analyses of bounded-range mechanisms such as the Exponential mechanism. Building on these theoretical results, we introduce AIM-GDP, a GDP-native variant of AIM (a state-of-the-art DP synthetic data mechanism) that enables easy composability with complex pipelines via PLD, attack-aw
    
[^38]: 基于跨用户成对偏好的在线大语言模型个性化

    Online LLM Personalization from Pairwise Preferences Across Users

    [https://arxiv.org/abs/2610.17203](https://arxiv.org/abs/2610.17203)

    该论文将基于成对偏好反馈的大语言模型推理时个性化形式化为上下文决斗老虎机问题的变体，提出一种无需初始偏好数据、可在会话内自适应并利用跨用户共享信息的在线选择方法，以最小化累积强遗憾。

    

    大语言模型可以用不同的风格回答同一个问题，但用户偏好的风格是隐藏的，且因用户和问题而各不相同。我们研究推理时的个性化问题，即从一个固定的提示词集合（每个提示词代表一种人格角色）中进行在线选择。用户依次到来，每位用户提出一小段序列的问题，系统对每个问题展示两个回答，并观察一次成对选择。我们将这种交互形式化为上下文决斗老虎机问题的一个变体：学习者起初没有任何偏好数据，能在每个会话内进行自适应，并利用跨用户共享的信息。目标是最小化累积强遗憾（strong regret），即所展示的两个回答相对于用户在可用选项中最佳回答的效用差距之和。我们做出了两项贡献：第一，我们提出了该问题，并发布了一个可复现的偏好基准和数据集，该基准围绕教学辅导场景构建……（摘要原文在此处截断）

    arXiv:2610.17203v1 Announce Type: cross  Abstract: Large language models can answer the same question in different styles, but users' preferred styles are hidden and vary across users and questions. We study inference-time personalization through online selection from a fixed set of prompts, each representing a persona. Users arrive sequentially, each asks a short sequence of questions, and the system displays two responses per question and observes one pairwise choice. We formulate this interaction as a variant of the contextual dueling bandit problem in which the learner starts without preference data, adapts within each session, and exploits information shared across users. The objective is to minimize cumulative strong regret, summing the utility gaps of both displayed responses relative to the user's best response among the available options. We make two contributions. First, we introduce the problem and release a reproducible preference benchmark and dataset built around a tutori
    
[^39]: 广义McDiarmid不等式：样本复用与非均匀无界差分

    A Generalized McDiarmid Inequality: Sample Reuse and Non-Uniform Unbounded Differences

    [https://arxiv.org/abs/2610.17159](https://arxiv.org/abs/2610.17159)

    本文提出了McDiarmid不等式的一种新颖广义形式，能够同时刻画训练样本复用的收益并处理无界且非均匀依赖的差分函数，并将其应用于基于分数的生成模型以获得更紧的误差界。

    

    我们开发了McDiarmid集中不等式的一种新颖推广形式，并利用它得到了一致大数定律结果以及随机优化问题的误差界，这些结果同时包含两个关键特征：1）它们捕捉了在同时使用易于采样的辅助随机变量的算法中训练样本复用所带来的好处；2）它们允许单分量均值差分界定函数在感兴趣的分量上无界，同时对另一个分量具有非均匀依赖性。我们通过为基于分数的生成模型推导集中不等式来展示我们方法的实用性。这一应用自然地利用了我们新工具的两个关键特征：1）它将辅助高斯随机变量与训练样本结合使用；考虑训练样本复用可以通过有效积分消除对辅助变量的依赖，从而得到更紧的误差界。

    arXiv:2610.17159v1 Announce Type: new  Abstract: We develop a novel generalization of McDiarmid's concentration inequality and use it to obtain a uniform law of large numbers result and error bounds on stochastic optimization problems which simultaneously incorporate two key features: 1) They capture the benefit of training sample reuse in algorithms that also employ easily-sampled auxiliary random variables. 2) They allow for one-component mean-difference bounding functions which are both unbounded in the component of interest and have non-uniform dependence on another component. We demonstrate the utility of our method by deriving concentration inequalities for score-based generative modeling. This application naturally leverages both key features of our new tools: 1) It utilizes auxiliary Gaussian random variables in conjunction with the training samples; accounting for training sample reuse results in tighter error bounds by effectively integrating out the dependence on the auxilia
    
[^40]: 先SCOPE再采样：共形分数引导后验探索，实现快速且校准良好的贝叶斯预测推断

    SCOPE Before You Sample: Conformal Scores Guide Posterior Exploration for Fast and Calibrated Bayesian Predictive Inference

    [https://arxiv.org/abs/2610.17146](https://arxiv.org/abs/2610.17146)

    提出SCOPE方法，利用共形非一致性分数的梯度识别影响预测区域边界的关键参数方向，引导MCMC在这一低维子空间中探索，从而以远低于完整MCMC的成本获得保留MCMC预测质量且校准良好的贝叶斯预测推断。

    

    贝叶斯预测区域考虑了参数不确定性，但在模型设定错误的情况下，即使使用昂贵的高精度MCMC（马尔可夫链蒙特卡洛）方法，其覆盖率校准也可能很差。变分推断等更廉价的近似方法虽然降低了计算成本，但往往导致预测分布的准确性下降。我们提出了SCOPE（Score-guided Conformal Posterior Exploration，分数引导的共形后验探索），它在保留完整MCMC预测质量的同时，只需其成本的一小部分即可实现校准良好的覆盖率。关键观察是：预测区域对后验的依赖主要体现在少数几个会移动区域边界的参数方向上，而这些方向通常并不是后验方差最大的方向。SCOPE赋予非一致性分数一个新的角色——引导后验计算：在留出的引导集上计算分数梯度，并向目标边界加权，从而识别出相关方向。MCMC在该子空间中进行探索，同时结合一个廉价的投影变分近似。

    arXiv:2610.17146v1 Announce Type: new  Abstract: Bayesian predictive regions account for parameter uncertainty but can be poorly calibrated under model misspecification, even with expensive full MCMC. Cheaper approximations such as variational inference reduce the cost but often yield less accurate predictive distributions. We propose SCOPE (Score-guided Conformal Posterior Exploration), which retains the predictive quality of full MCMC while achieving calibrated coverage at a fraction of its cost. The key observation is that a prediction region can depend on the posterior mainly through a few parameter directions that move the region's boundary, and these are generally not the directions of largest posterior variance. SCOPE gives nonconformity scores a new role in guiding posterior computation: score gradients on a held-out guide set, weighted toward the target boundary, identify the relevant directions. MCMC explores this subspace, while an inexpensive projected variational approxima
    
[^41]: Rényi会计器能否在DP-SGD中击败无损会计器？

    Can R\'enyi Accountants Beat Lossless Accountants for DP-SGD?

    [https://arxiv.org/abs/2610.17120](https://arxiv.org/abs/2610.17120)

    该论文证明，在替换邻接、固定批次大小的DP-SGD设置下，通过将最坏情形分布刻画为二维高斯混合，基于Rényi DP的隐私会计可以获得比无损隐私框架更紧的隐私保证。

    

    差分隐私（DP）领域的近期框架，如隐私损失分布和 $f$-DP，能够对机制的隐私泄露提供无损表示。这些方法通常被认为能够给出比Rényi DP等松弛概念更紧的 $(\varepsilon, \delta)$-DP 保证。但事实总是如此吗？我们针对在替换邻接（replace adjacency）关系下、采用固定大小批次的差分隐私随机梯度下降（DP-SGD）研究了这一问题。我们的关键观察是：在该设置下，基于Rényi DP的隐私会计可以使用一对与无损框架中所用分布不同的最坏情形分布（即支配对，dominating pair）。特别地，我们将最坏情形分布刻画为二维高斯混合分布，并据此为DP-SGD推导出改进的Rényi DP界。数值与渐近比较表明，在各种超参数设置下，Rényi DP会计器……（原文摘要在此处截断）

    arXiv:2610.17120v1 Announce Type: cross  Abstract: Recent frameworks in differential privacy (DP), such as privacy loss distributions and $f$-DP, provide lossless representations of the privacy leakage of mechanisms. These approaches are often considered to yield tighter $(\varepsilon, \delta)$-DP guarantees than those obtained from relaxed notions such as R\'enyi DP. But is that always the case? We study this question for differentially private stochastic gradient descent (DP-SGD) with fixed-size batches under replace adjacency. Our key observation is that in this setting, privacy accounting based on R\'enyi DP can use a pair of worst-case distributions (i.e., a dominating pair) that differs from the one used in the lossless frameworks. In particular, we characterize the worst-case distributions as two-dimensional Gaussian mixtures and derive an improved R\'enyi DP bound for DP-SGD. Numerical and asymptotic comparisons show that across various hyperparameter settings, a R\'enyi DP acc
    
[^42]: 上下文分类：带softmax读出的线性注意力中的校准与噪声

    In-context classification: Calibration and noise in linear attention with softmax readout

    [https://arxiv.org/abs/2610.17061](https://arxiv.org/abs/2610.17061)

    该论文针对线性注意力模型的上下文二分类任务推导出测试逻辑值的渐近解析公式，将校准误差、上下文噪声与参数估计噪声分离为可解释的独立效应，并系统分析了上下文长度、预训练规模、正则化和采样温度的影响，理论预测与仿真及真实训练模型高度吻合。

    

    我们研究了在一个简单的注意力模型中进行上下文二分类的问题，该模型通过最小化逻辑损失进行训练。每个提示（prompt）都包含一个潜在的逻辑教师（logistic teacher），学习者必须从带有伯努利标签的上下文示例中推断其方向。在上下文长度与词元维度成比例、训练提示总数与维度平方成比例的高维情形下，我们推导出了测试逻辑值（test logit）的渐近预测。我们的公式清晰地将校准、测试时上下文噪声以及非各向同性的参数估计噪声分离开来，作为对上下文性能的可解释影响。我们的分析还确定了无岭可分性（ridgeless separability）区域，并探讨了上下文长度、预训练集规模、正则化以及采样温度对上下文分类性能的影响。该理论的预测与有限维仿真以及完全训练的线性注意力模型高度吻合，同时实验……

    arXiv:2610.17061v1 Announce Type: new  Abstract: We study in-context binary classification in a simple attention model trained to minimise a logistic loss. Each prompt has a latent logistic teacher, and the learner must infer its direction from Bernoulli-labeled context examples. We derive an asymptotic prediction for the test logit in a high-dimensional regime with context length proportional to token dimension, and total training prompts proportional to dimension squared. Our formula cleanly separates calibration, test-time context noise, and non-isotropic parameter-estimation noise as interpretable effects on in-context performance. Our analysis also identifies the ridgeless separability regime and addresses the effects of context length, pretraining-set size, regularisation, and sampling temperature on in-context classification performance. The theory's predictions closely agree with finite-dimensional simulations and fully trained linear-attention models, while experiments with st
    
[^43]: 基于比例包含边际效应的变量重要性度量

    Measuring Variable Importance Using Proportional Inclusive Marginal Effects

    [https://arxiv.org/abs/2610.17053](https://arxiv.org/abs/2610.17053)

    提出了PRIME（比例包含边际效应）这一新的变量重要性度量方法，通过无模型变量选择技术仅纳入活跃变量进行计算，克服了PME仅适用于确定性模型且计算成本高的局限，能从含噪声数据中准确可靠地量化变量重要性。

    

    基于Shapley的度量方法在统计学和机器学习中被广泛用于衡量变量重要性。然而，这些度量方法存在一个严重问题——当一个非活跃变量与活跃变量相关时，该非活跃变量可能获得正的重要性度量值。Herin等人最近提出了一种称为比例边际效应（PME）的度量方法来缓解这种虚假效应。然而，PME是为确定性模型开发的，且计算成本很高。本文提出了一种称为比例包含边际效应（PRIME）的新度量方法，克服了PME的这两个局限性。PRIME的关键思想是使用一种新颖的无模型变量选择技术来识别活跃变量，并仅将这些变量纳入重要性度量的计算中。对测试函数和真实数据集的分析表明，PRIME能够从含噪声的输入-输出数据中准确、可靠地量化变量重要性。

    arXiv:2610.17053v1 Announce Type: new  Abstract: Shapley-based measures are widely used for measuring variable importance in statistics and machine learning. However, these measures have a serious issue--a nonactive variable can get a positive importance measure when it is correlated with an active variable. Herin et al. recently proposed a measure called proportional marginal effects (PME) that mitigates this spurious effect. However, PME is developed for deterministic models and has high computational cost. In this article, we propose a new measure called proportional inclusive marginal effects (PRIME) that overcomes these two limitations of PME. The key idea in PRIME is to use a novel model-free variable selection technique to identify the active variables and include only them in the computation of the importance measure. Analysis of test functions and real datasets show that PRIME can accurately and reliably quantify variable importance from noisy input-output data.
    
[^44]: CLASSO：一种实用的约束LASSO原始路径跟踪算法

    CLASSO: a practical primal path-following algorithm for the constrained LASSO

    [https://arxiv.org/abs/2610.17024](https://arxiv.org/abs/2610.17024)

    提出了CLASSO算法，一种求解带一般线性等式与不等式约束的LASSO的实用原始路径跟踪方法，无需约束规格条件即可证明解路径为分段仿射且折点有限，并解决了折点处重新认证可能不终止的问题。

    

    我们提出了CLASSO，一种针对受一般线性等式和不等式约束的LASSO的路径跟踪方法。其局部方向通过在增广活动集上求解二次规划获得，该规划同时确定离开零值的系数以及保持起约束作用的弱活跃不等式。对于正定的Hessian矩阵和非空可行集，即使起作用的约束行线性相关，该规划也具有唯一的原始极小值点，并给出解路径的局部导数。该构造在无约束情形下可退化为LASSO修正的LARS方向，而在全锥上则给出离开原点的线段。我们证明，无需约束规格条件或乘子的唯一性，原始路径即为分段仿射且折点有限。然而，有限路径并不意味着有限次遍历：在每个折点处重新认证可能导致算法无法终止。CLASSO转而遵循最小范数认证……（原文摘要在此截断）

    arXiv:2610.17024v1 Announce Type: cross  Abstract: We present CLASSO, a path-following method for the LASSO subject to general linear equality and inequality constraints. The local direction is obtained from a quadratic programme on an augmented active set, which determines both the coefficients that leave zero and the weakly active inequalities that remain binding. For a positive-definite Hessian and a nonempty feasible set, this programme has a unique primal minimiser, even when the binding constraint rows are dependent, and gives the local derivative of the solution path. The construction recovers the LASSO-modified LARS direction without constraints, and over the full cone gives the segment leaving the origin. We prove that the primal path is piecewise affine with finitely many kinks, without a constraint qualification or unique multipliers. A finite path does not imply a finite traversal: re-certifying at each kink can fail to terminate. CLASSO instead follows the minimum-norm cer
    
[^45]: ProbabilityBench：基于精确真值不确定性的模型评估与校准

    ProbabilityBench: Model Evaluation and Calibration with Exact Ground-Truth Uncertainty

    [https://arxiv.org/abs/2610.17016](https://arxiv.org/abs/2610.17016)

    ProbabilityBench通过显式定义数据生成过程，在三个模拟文本领域中提供精确的实例级真值类别概率，实现了对模型概率预测质量的直接评估，并发现成本-质量帕累托前沿上的模型多为决策模型。

    

    在真实世界数据上评估模型不确定性具有挑战性，因为每个实例仅提供一个观测标签，真实的条件类别概率是未知的。为解决这一问题，我们提出了ProbabilityBench，这是一个在三个模拟的文本领域中提供精确的、实例级类别概率的基准测试，这三个领域分别是：同行评审、隐蔽信息传递和信用风险。通过为每个环境显式定义完整的数据生成过程，我们以解析方式计算出精确的类别概率和贝叶斯误差。此外，我们提供了理论保证，即将这些案例渲染为自然语言备忘录能严格保持其底层的生成概率。这种精确性使得可以通过KL散度直接测量模型报告概率的质量。在对16个模型的评估中，观测到的成本-质量帕累托前沿上的大多数模型是决策模型，包括Jev和Perplexity Decider。

    arXiv:2610.17016v1 Announce Type: cross  Abstract: Evaluating model uncertainty on real-world data is challenging because each instance provides only a single observed label, leaving true conditional class probabilities unknown. To address this, we introduce ProbabilityBench, a benchmark that provides exact, instance-level class probabilities across three simulated text-based domains: peer review, covert information transfer, and credit risk. By explicitly defining the full data-generating process for each environment, we analytically compute both exact class probabilities and Bayes error. Furthermore, we provide a theoretical guarantee that rendering these cases into natural-language memos strictly preserves their underlying generative probabilities. This exactness enables direct measurement of reported probability quality via KL divergence. Across 16 evaluated models, most models on the observed cost-quality Pareto frontiers are decision models, including Jev and Perplexity Decider. 
    
[^46]: Shampoo算法在非平稳随机优化中的极小极大最优性

    Minimax Optimality of Shampoo for Nonstationary Stochastic Optimization

    [https://arxiv.org/abs/2610.16923](https://arxiv.org/abs/2610.16923)

    该论文对矩阵Shampoo在非平稳随机优化中追踪时变最优解的性能进行了全面理论分析，推导出分离初始化、漂移、噪声与自适应预条件化影响的有限时域高概率追踪误差界，揭示了动量参数β₁的漂移—噪声权衡与稳定化参数ε的稳定性—几何自适应性权衡，并证明了其极小极大最优性。

    

    我们对矩阵Shampoo算法进行了全面的理论分析，用于追踪非平稳随机优化中随时间变化的最优解。我们推导了平均追踪误差的有限时域高概率界，该界清晰地分离了初始化、优化器漂移、动量引起的漂移惩罚、随机噪声以及自适应预条件化等各项影响，并显式依赖于动量参数β₁和稳定化参数ε。这些界揭示了显式的漂移—噪声权衡与稳定性—几何权衡：在噪声主导的情形下，较大的β₁更有利，因为它能提供更强的一阶矩平均和更大的梯度噪声抑制；而在漂移主导的情形下，较小的β₁更有利，使算法能够更快地适应移动中的最优解。较大的ε会减小预条件器扰动项，但同时会削弱Shampoo利用矩阵几何结构的能力，从而降低其自适应性。我们建立了匹配……（原文摘要在此处被截断）

    arXiv:2610.16923v1 Announce Type: new  Abstract: We develop a comprehensive theoretical analysis of Matrix Shampoo for tracking time-varying optima in nonstationary stochastic optimization. We derive finite-horizon high-probability bounds for average tracking error that disentangle initialization, optimizer drift, momentum-induced drift penalty, stochastic noise, and adaptive preconditioning, with explicit dependence on momentum parameter \(\beta_1\) and stabilization parameter \(\varepsilon\). These bounds reveal an explicit drift--noise and stability--geometry tradeoffs: noise-dominated regimes favor larger $\beta_1$, which provides stronger first-moment averaging and greater suppression of gradient noise, whereas drift-dominated regimes favor smaller $\beta_1$, allowing faster adaptation to moving optimizer. Larger $\varepsilon$ reduces the preconditioner-perturbation term while weakening Shampoo's ability to exploit matrix geometry, thereby reducing adaptivity. We establish a match
    
[^47]: 径向尾部修复：针对扩散模型与流模型的后训练校正

    Radial Tail Repair: Post-Training Correction for Diffusion and Flow Models

    [https://arxiv.org/abs/2610.16851](https://arxiv.org/abs/2610.16851)

    提出一种轻量级后训练校正方法，基于多元重尾的径向-角度分解，在保持冻结的扩散或流匹配模型方向结构不变的情况下，仅通过重缩放生成样本的径向尾部来修正模型对重尾分布极端区域的表示。

    

    扩散模型和流匹配模型可能难以准确表示分布的尾部，尤其是对于重尾数据。现有方法通常使用重尾噪声或修改生成动力学，需要专门的重新训练和更困难的估计。我们采取一种更简单的方法：不是重新学习整个分布，而是仅校正模型在极端区域的行为。我们的方法建立在多元重尾分布的径向-角度分解之上，其中半径和方向渐近独立。这使我们能够保留预训练模型学到的方向结构，仅校正其径向尾部。我们从参考样本和生成样本中估计尾部起始点，然后重新缩放生成的极端值，使其径向生存分布与目标分布相匹配。所得方法是一种轻量级的后训练校正，可应用于冻结的扩散或流匹配模型。

    arXiv:2610.16851v1 Announce Type: cross  Abstract: Diffusion and flow-matching models may struggle to accurately represent distributional tails, especially for heavy-tailed data. Existing approaches often use heavy-tailed noise or modify the generative dynamics, requiring dedicated retraining and more difficult estimation. We take a simpler approach: rather than relearning the full distribution, we correct only the model's behavior in the extreme regime. Our method builds on the radial-angular decomposition of multivariate heavy tails, where radius and direction become asymptotically independent. This allows us to preserve the directional structure learned by the pretrained model and correct only its radial tail. We estimate the tail onset from reference and generated samples, then rescale generated extremes so that their radial survival matches the target distribution. The resulting method is a lightweight post-training correction applied to a frozen diffusion or flow-matching model, 
    
[^48]: 有限训练预算下的特征归一化

    Feature Normalization under Finite Training Budgets

    [https://arxiv.org/abs/2610.16845](https://arxiv.org/abs/2610.16845)

    该论文在有限训练预算下理论刻画了标准化与未缩放训练各自可达到的最低稀有子群风险，并据此揭示了噪声水平与训练预算如何决定特征归一化策略的优劣。

    

    对稀有特征进行标准化可以帮助随机梯度下降在有限预算内学习其效应，但这也改变了学习的噪声成本。我们刻画了在相同观测上限下，当标准化训练与未缩放训练各自选择非负常数学习率和确定性停止时间时，所能达到的最低稀有子群风险。对于两个独立的二元特征，我们在流行率趋于零而稀有观测的期望数量保持有限的情况下，推导出这两种优化后的风险。对学习率和停止时间的一致控制将每个极限预言机简化为一个标量最小化问题。在正噪声存在时，随着我们将信号移向稀有特征，首选的缩放方式恰好改变一次，而初始化时的极限风险保持固定且为正。这些公式还解释了噪声和预算如何影响两者的比较。在稀有主导信号上，有限流行率下的独立界限更有利于标准化。

    arXiv:2610.16845v1 Announce Type: new  Abstract: Standardizing a rare feature can help stochastic gradient descent learn its effect within a finite budget, but it also changes the noise cost of learning. We characterize the lowest rare-subgroup risks attainable by standardized and unscaled training when each chooses a nonnegative constant learning rate and a deterministic stopping time under the same observation cap. For two independent binary features, we derive both optimized risks as prevalence vanishes while the expected number of rare observations remains finite. Uniform control over rates and stopping times reduces each limiting oracle to a scalar minimization. With positive noise, the preferred scaling changes exactly once as we move signal to the rare feature while the limiting risk at initialization stays fixed and positive. The formulas also explain how noise and budget affect the comparison. Separate finite-prevalence bounds favor standardization on a rare-dominant signal fa
    
[^49]: 基于统计缩放的仿真推断样本量外推方法

    Sample-Size Extrapolation in Simulation-Based Inference via Statistical Scaling

    [https://arxiv.org/abs/2610.16820](https://arxiv.org/abs/2610.16820)

    提出了CE-NPE方法，通过将统计中心估计与显式收缩率相结合，使神经后验估计器能够在无需可评估似然函数的情况下外推到更大样本量，在所有超出训练范围的样本量上均取得最佳的后验推断精度。

    

    在基于仿真的推断（SBI）中，在有限样本量范围内训练的神经后验估计器在处理更大数据集时可能会变得不准确，因为其参数化方式没有规定随着观测数据的累积，后验的位置、不确定性和形状应如何变化。我们提出了中心估计神经后验估计（CE-NPE），它将统计中心与显式收缩率相结合，并在后验尺度坐标中学习局部中心校正和残差密度。该中心是一种统计估计，例如闭式统计量或基于仿真器的矩匹配，不需要可评估的似然函数。我们的分析解释了收缩如何放大中心误差，并将高斯后验误差分解为形状、均值和协方差三个方面的贡献。在六个常规基准测试中，CE-NPE在所有测试的摊销式估计器中，于每个超出训练范围的样本量上都实现了与参考后验最佳的一致性，最高可达五倍的……（原文截断）

    arXiv:2610.16820v1 Announce Type: new  Abstract: In simulation-based inference (SBI), neural posterior estimators trained on a bounded range of sample sizes can become inaccurate on larger datasets because their parameterizations do not specify how posterior location, uncertainty and shape must change as observations accumulate. We propose center-estimated NPE (CE-NPE), which combines a statistical center with an explicit contraction rate and learns a local center correction and residual densities in posterior-scale coordinates. The center is a statistical estimate, such as a closed-form statistic or a simulator-based moment match, which does not require an evaluable likelihood. Our analysis explains how contraction amplifies center error and decomposes Gaussian posterior error into shape, mean and covariance contributions. On six regular benchmarks, CE-NPE gives the best reference-posterior agreement of all tested amortized estimators at every out-of-range size, up to five times the l
    
[^50]: 论众包聚合器的校准

    On the Calibration of Crowdsourcing Aggregators

    [https://arxiv.org/abs/2610.16815](https://arxiv.org/abs/2610.16815)

    该论文首次对众包标签聚合的校准问题进行了理论与实证研究，发现多数投票过于保守而基于模型的聚合器过于自信，并证明贝叶斯聚合器仅在其自身模型假设下才是完美校准的。

    

    校准问题在分类器领域已被广泛研究，但对于日益被用于概率决策的众包聚合方法而言，校准在很大程度上仍未被探索。据我们所知，我们提供了首个关于标签聚合中校准问题的理论与实证研究。令我们非常惊讶的是，我们发现广泛使用的聚合方法可能存在严重的校准偏差，且方向相反：多数投票过于保守（置信度不足），而基于模型的聚合器则过于自信。我们证明，贝叶斯聚合器在其数据生成模型下是完美校准的。我们还表明，加权多数投票要实现精确校准需要极为严格的条件，这有助于解释为什么多数投票在现实众包场景中通常存在校准偏差。在实证方面，我们在39个人类众包数据集和基于大语言模型（LLM）的众包群体上研究了校准问题。在这两种情况下，我们都观察到了系统性的校准偏差。

    arXiv:2610.16815v1 Announce Type: new  Abstract: Calibration has been extensively studied for classifiers, but remains largely unexplored for crowdsourcing aggregation methods, which are increasingly used for making probabilistic decisions. To the best of our knowledge, we provide the first theoretical and empirical study of calibration in label aggregation. To our great surprise, we find that widely used aggregation methods can be severely miscalibrated, and in opposite directions: Majority Vote is underconfident while model-based aggregators are overconfident. We prove that the Bayesian aggregator is perfectly calibrated under its data-generating model. We also show that exact calibration of weighted majority votes requires highly restrictive conditions, which helps explain why Majority Vote is generally miscalibrated in realistic crowds. Empirically, we study calibration across $39$ human crowdsourcing datasets and LLM-based crowds. In both cases, we observe systematic miscalibratio
    
[^51]: 重复测量何时会损害表格基础模型

    When Repeated Measurements Hurt Tabular Foundation Models

    [https://arxiv.org/abs/2610.16801](https://arxiv.org/abs/2610.16801)

    该研究发现表格基础模型在面对相同输入的重复测量时，会因PFN预训练的IID规则与实际协议不匹配以及注意力机制对行数的依赖，导致未测量输入处的预测性能下降。

    

    表格基础模型（TFMs）在独立同分布（IID）基准测试中取得了最先进的性能，但它们如何应对相同输入处的重复测量呢？在六个表格基础模型上进行的合成与半合成实验表明，额外的测量可以改善已测量输入处的预测，同时使其他位置的预测变差。这种损害在低噪声情况下最为常见，并且在多个模型中无关特征会放大这种损害；真实的重复测量数据集也显示出类似模式。从统计学角度看，PFN预训练所针对的IID规则将每个重复输入视为关于输入分布的新证据。当先验分布将这一分布与响应函数联系起来时，由此产生的协议不匹配可能会超过额外测量带来的好处，即使在精确推理下也是如此。从计算角度看，干预实验确定了TabPFN-v3注意力缩放机制中对行数的依赖是造成损害的一个因素。重置行数可以在很大程度上逆转这种失真。

    arXiv:2610.16801v1 Announce Type: new  Abstract: Tabular foundation models (TFMs) achieve state-of-the-art performance on IID benchmarks, but how do they respond to repeated measurements at the same inputs? Across six TFMs, synthetic and semi-synthetic experiments show that additional measurements can improve predictions at measured inputs while degrading them elsewhere. Harm is most frequent at low noise, and irrelevant features amplify it in several models; real repeated-measurement datasets show similar patterns. Statistically, the IID rule targeted by PFN pretraining treats each repeated input as fresh evidence about the input distribution. When the prior links this distribution to the response function, the resulting protocol mismatch can outweigh the benefit of additional measurements, even under exact inference. Computationally, interventions identify row-count dependence in TabPFN-v3's attention scaling as a contributor to harm. Resetting the count largely reverses distortion i
    
[^52]: 单层Transformer可通过思维链可证明地学习上下文K近邻预测

    One-Layer Transformers Provably Learn In-Context K-Nearest Neighbor Prediction with Chain-of-Thought

    [https://arxiv.org/abs/2610.16798](https://arxiv.org/abs/2610.16798)

    本文从理论上证明单层Transformer可通过思维链训练可证明地学会上下文中的K近邻预测，突破了softmax注意力难以直接关注多个近邻的固有局限。

    

    思维链（Chain-of-Thought, CoT）使Transformer能够通过生成中间推理步骤来解决复杂的推理任务。尽管在实证上取得了巨大成功，但这种推理能力在训练过程中如何涌现的机制仍知之甚少，特别是从理论角度来看。在这项工作中，我们在一个风格化但完全可分析的设置中研究这一问题：基于单层Transformer的上下文内K近邻（K-NN）预测。先前的工作（Li et al., 2024a）已经证明，单层Transformer可以在不使用思维链的情况下被训练来执行1近邻预测，其方式是让softmax注意力关注上下文中最近的邻居。然而，这一关于1近邻的结果无法扩展到K>1的K近邻预测，因为softmax注意力天然不适合同时关注第2至第K近的邻居。我们首先给出了一个严格的否定性结果，表明这一局限性已经出现（摘要在此处截断）。

    arXiv:2610.16798v1 Announce Type: cross  Abstract: Chain-of-Thought (CoT) enables transformers to solve complex reasoning tasks by generating intermediate reasoning steps. Despite strong empirical success, the mechanisms by which such reasoning abilities emerge during training remain poorly understood, particularly from a theoretical perspective. In this work, we study this question in a stylized yet fully analyzable setting: in-context $\mathrm{K}$-nearest-neighbor ($\mathrm{K}$-NN) prediction with a one-layer transformer. Earlier work has demonstrated that one-layer transformers can be trained to perform $1$-NN without CoT (Li et al., 2024a) by having the softmax attention attend to the nearest neighbor in context. However, this result on $1$-NN cannot be extended to $\mathrm{K}$-NN for $K>1$, as softmax attention is much less naturally suited to attending to the 2nd through $K$-th nearest neighbors. We first give a rigorous negative result showing that this limitation already appear
    
[^53]: 摊销式自适应重要性采样

    Amortised adaptive importance sampling

    [https://arxiv.org/abs/2610.16736](https://arxiv.org/abs/2610.16736)

    该论文提出了一种摊销式自适应重要性采样算法，通过拟合轻量级神经网络利用采样过程中的信息迭代调整提议分布，从而在训练完成后以更少的计算量生成高质量样本。

    

    蒙特卡洛采样算法旨在从具有给定非归一化概率密度函数的目标分布中生成样本。运行蒙特卡洛算法的高昂成本——需要许多模拟步骤或大量粒子才能对目标获得良好的经验近似——可以通过在采样器中引入可学习组件来降低，这是一种摊销形式。一旦这些组件训练完成，摊销采样器便可在采样阶段以比免训练算法所需更少的计算量生成高质量样本。我们提出了一种建立在自适应重要性采样框架之上的摊销采样器——自适应重要性采样是一种分层重要性采样过程，其中提议分布被迭代地调整。我们的摊销自适应重要性采样算法拟合一个轻量级神经网络，该网络利用采样过程中获得的信息来调整提议分布。通过在许多不同的目标分布上训练我们的采样器（摘要在此处截断）。

    arXiv:2610.16736v1 Announce Type: new  Abstract: Monte Carlo sampling algorithms aim to produce samples from a target distribution with a given unnormalised probability density function. The high cost of running Monte Carlo algorithms, with many simulation steps or particles needed to obtain a good empirical approximation to the target, can be reduced by introducing learnable components into the sampler -- a form of amortisation. Once these components are trained, amortised samplers can produce high-quality samples with less computation at sampling time than training-free algorithms would require. We propose an amortised sampler built upon the framework of adaptive importance sampling -- a hierarchical importance sampling procedure where the proposal is iteratively adapted. Our amortised adaptive importance sampling algorithm fits a lightweight neural network that adapts the proposal using information obtained over the course of sampling. By training our sampler over many different tar
    
[^54]: 平均化退火马尔可夫链达到张量PCA的算法阈值，第一部分：Glauber动力学

    Averaged Annealed Markov Chains Achieve the Algorithmic Threshold for Tensor PCA, Part I: Glauber Dynamics

    [https://arxiv.org/abs/2610.16705](https://arxiv.org/abs/2610.16705)

    通过结合退火与遍历平均的Glauber动力学，局部马尔可夫链能够从随机初始化出发，在算法阈值之上对张量PCA实现多项式时间强恢复，从而颠覆了局部算法劣于全局算法的传统认知。

    

    在张量主成分分析（张量PCA）问题中，我们需要从一个未知向量的k次张量幂的含噪观测中恢复该向量，其中向量取自{±1}^N，k≥3为固定整数。张量PCA表现出一个已被广泛研究的信息-计算鸿沟：谱方法和平方和等全局算法在某个算法阈值之上可以成功运行，而在该阈值以下，我们知道针对这些算法以及某些其他受限计算模型的匹配下界。相比之下，马尔可夫链等局部算法此前仅在信噪比高出多项式倍时才被证明能从随机初始化成功，并且人们普遍认为它们在张量PCA问题上劣于全局算法。我们证明，Glauber动力学在辅以退火和遍历平均之后，能够从均匀随机初始化出发，在算法阈值之上实现多项式时间的强恢复，这表明局部马尔可夫链可以……

    arXiv:2610.16705v1 Announce Type: cross  Abstract: In the tensor principal components analysis (tensor PCA) problem, we are asked to recover an unknown vector in $\{\pm1\}^N$ from a noisy observation of its $k$-th tensor power, for a fixed $k\ge3$. Tensor PCA exhibits a well-studied information-computation gap: global algorithms such as spectral methods and sum-of-squares work above an algorithmic threshold, below which we know matching lower bounds (against these and some other restricted models of computation). In contrast, local algorithms such as Markov chains were only known to succeed from a random initialization at a polynomially larger signal-to-noise ratio, and were believed to be inferior to global algorithms for tensor PCA. We show that Glauber dynamics, when supplemented with annealing and ergodic averaging, achieves polynomial-time strong recovery above the algorithmic threshold from a uniformly random initialization, demonstrating that local Markov chains can succeed all 
    
[^55]: 从重复点模式中对潜在强度进行摊销变分推断

    Amortized Variational Inference of Latent Intensities from Replicated Point Patterns

    [https://arxiv.org/abs/2610.16669](https://arxiv.org/abs/2610.16669)

    提出了一种基于神经网络的摊销变分推断框架，无需真实强度监督即可从重复点模式中学习可复用的推断规则，通过一次前向传播完成潜在强度分布的贝叶斯推断，相比对数高斯Cox过程等方法速度大幅提升且推断精度更高。

    

    对潜在点过程强度的贝叶斯推断通常依赖于针对每个数据集的采样或优化，使得重复推断在计算上非常昂贵。我们提出了一种基于神经网络的摊销变分框架，该框架学习一个可复用的推断规则，将可变数量的重复点模式映射到其共享潜在强度的分布上。该模型使用点过程似然从事件时间进行训练，在无需真实强度监督的情况下学习到摊销推断规则。在测试时，它只需一次前向传播即可完成潜在分布的推断。在合成数据上，更多的观测实现改善了强度恢复效果，并产生更窄的置信区间，同时覆盖率大体保持稳定。我们的方法在共享单车数据上进行了测试，在所评估的方法中取得了最高的平均留出对数似然，同时所需的推断时间远少于对数高斯Cox过程。

    arXiv:2610.16669v1 Announce Type: new  Abstract: Bayesian inference for latent point process intensities often relies on per-dataset sampling or optimization, making repeated inference computationally expensive. We propose a neural-network-based amortized variational framework that learns a reusable inference rule mapping a variable number of replicated point patterns to a distribution over their shared latent intensity. Trained from event times using the point process likelihood, the model learns an amortized inference rule without ground-truth intensity supervision. At test time, it performs inference of the latent distribution in a forward pass. On synthetic data, additional realizations improve intensity recovery and yield narrower credible intervals with broadly stable coverage. Our method is tested on bicycle-sharing data and achieves the highest mean held-out log likelihood among the evaluated methods while requiring substantially less inference time than the Log-Gaussian Cox Pr
    
[^56]: 基于变分估计的广义线性模型最快变点检测

    Quickest Change-point Detection in Generalized Linear Models with Variational Estimation

    [https://arxiv.org/abs/2610.16616](https://arxiv.org/abs/2610.16616)

    该论文提出了一种基于变分不等式估计的窗口受限CUSUM方法，用于变化后参数未知的广义线性模型最快变点检测，并刻画了估计误差对检测延迟的影响且给出了有限窗口下的性能保证。

    

    我们研究了广义线性模型中变化后参数未知时的最快变点检测问题。这类问题出现在诸如监控学习到的表示与观测响应之间的关系是否随时间变化的场景中。由于检测必须在变化发生后尽快进行，变化后参数只能从有限的观测数据中估计，因此有限样本估计精度尤为重要。我们提出了一种窗口受限的CUSUM检测方法，该方法利用变分不等式（VI）形式来估计变化后参数。对于非规范链接函数，尽管极大似然估计具有渐近效率优势，但VI估计在短窗口下能够提供更优的有限样本精度。我们刻画了估计误差如何降低变化后的有效信息率并影响检测延迟，并推导了同时考虑上述两个因素的有限窗口性能保证。

    arXiv:2610.16616v1 Announce Type: cross  Abstract: We consider quickest change-point detection in generalized linear models when the post-change parameter is unknown. Such problems arise, for example, when monitoring whether the relationship between a learned representation and an observed response changes over time. Since detection must occur soon after a change, the post-change parameter must be estimated from a limited number of observations, making finite-sample estimation accuracy particularly important. We propose a window-limited CUSUM procedure that estimates the post-change parameter using a variational-inequality (VI) formulation. For noncanonical links, VI estimation can offer favorable finite-sample accuracy over short windows despite the asymptotic efficiency advantage of maximum likelihood estimation. We characterize how estimation error reduces the effective post-change information rate and affects detection delay, and derive finite-window guarantees accounting for both 
    
[^57]: 基于独立二元Probit替代似然的贝叶斯末层变分滤波

    Variational Filtering for Bayesian Last Layers with an Independent Binary Probit Surrogate

    [https://arxiv.org/abs/2610.16578](https://arxiv.org/abs/2610.16578)

    提出了一种将深度分类器末层参数视为线性高斯状态空间模型隐状态的变分滤波算法，借助独立二元Probit替代似然实现闭式共轭更新，可在特征漂移下稳定训练并同时适用于线性探测与全网络微调。

    

    通过梯度下降训练的现代深度分类器需要对数据进行多次遍历才能获得较高准确率，即便如此其校准性也未必良好；事后温度缩放虽可在训练后修正校准，却无法影响训练过程中的准确率或收敛性。我们为深度分类器的最后一层提出了一种变分滤波算法，将该层参数视为线性高斯状态空间模型的隐状态。每个小批量数据通过一个精确的预测步骤和随后的闭式变分更新被同化，其中通过为类别似然引入独立二元Probit（IBProbit）替代模型实现共轭性。滤波器的过程噪声在特征漂移下稳定了更新过程，并防止预测分布坍缩为点估计，使得同一流程既可用于冻结骨干网络上的线性探测，也可用于全网络微调。我们在 DINOv3 骨干网络上进行评估（原文摘要在此处截断）。

    arXiv:2610.16578v1 Announce Type: new  Abstract: Modern deep classifiers trained by gradient descent need many passes over the data to become accurate, and even then are not always well calibrated; post-hoc temperature scaling corrects calibration after training but cannot influence training-time accuracy or convergence. We propose a variational filtering algorithm for the last layer of deep classifiers, whose parameters are treated as the latent state of a linear-Gaussian state-space model. Each mini-batch is assimilated by an exact prediction step followed by a closed-form variational update, made conjugate by an independent binary probit (IBProbit) surrogate for the categorical likelihood. The filter's process noise stabilises the updates under feature drift and keeps the predictive distribution from collapsing to a point estimate, allowing the same procedure to be used for linear probing on a frozen backbone and for full-network fine-tuning. We evaluate on DINOv3 backbones against 
    
[^58]: NFDiag：基于归一化流的高维MCMC收敛性诊断

    $\texttt{NFDiag}$: High-dimensional MCMC Convergence Diagnosis with Normalizing Flows

    [https://arxiv.org/abs/2610.16574](https://arxiv.org/abs/2610.16574)

    该论文提出NFDiag，一种利用归一化流在完整多变量联合后验空间中分析链的几何结构与条件分布信息的高维MCMC收敛性诊断新方法，克服了传统基于边际分布的诊断在高维下不敏感的问题。

    

    准确评估马尔可夫链蒙特卡洛（MCMC）的收敛性对于确保其在贝叶斯推断问题中的有效性至关重要。然而，众所周知，高维情形下的收敛性评估十分困难，因为大多数收敛性诊断方法在参数超过几十个之后就变得不敏感。传统的收敛性统计量分析的是边际分布，在高维情况下这会逐渐忽略来自联合分布和条件概率分布的信息。因此，我们提出了NFDiag，它利用归一化流（NF）来挖掘完整多变量联合后验分布中的信息。由此，我们的收敛性诊断方法也能够分析来自条件分布的信息。换言之，通过在完整的多变量空间中进行工作，该诊断方法对高维空间中链的几何结构是敏感的。我们的诊断方法利用了归一化流假设独立同分布的轨迹这一特性（原文在此处截断）。

    arXiv:2610.16574v1 Announce Type: new  Abstract: Accurately assessing the convergence of Monte Carlo Markov Chains is crucial for ensuring their validity in Bayesian inference problems. However, convergence assessment in high dimensions is known to be difficult, due to most convergence diagnostics becoming insensitive beyond a few dozen parameters. Conventional convergence statistics analyze marginal distributions which in high dimensions neglects increasing levels of information from the joint and conditional probability distributions. Thus we propose $\texttt{NFDiag}$, which taps into the information of the full multivariate joint posterior with a normalizing flow (NF). Our convergence diagnostic thereby also analyzes information from conditional distributions. In other words, by working in the full multivariate space, it is sensitive with respect to the geometry of the chain in high dimensions. Our diagnostic utilizes that NFs assume independent and identically distributed (iid) tra
    
[^59]: 学习分级稀疏机制：一个表示定理与可认证的泛化

    Learning Graded Sparse Mechanisms: A Representation Theorem and Certified Generalization

    [https://arxiv.org/abs/2610.16537](https://arxiv.org/abs/2610.16537)

    本文将“分级稀疏机制”公理化定义为一类函数，并证明了一个表示定理：该类函数恰好是加权多重集上某个传输度量的Lipschitz球，从而为这类机制的学习提供了可认证的泛化保证。

    

    在多种多智能体和物理环境中，一个元素的更新强烈地依赖于少数其他元素，而微弱地依赖于许多其他元素，并且起作用元素的“身份”和“数量”都会随系统状态而变化。这种分级稀疏机制十分常见，例如存在于鸟群、人群、学习型粒子模拟器以及图网络的邻域聚合之中，然而作为一类“分级信息已被给定”的函数，它们却很少得到系统性的研究。本文研究的是当分级由环境以施加在每个元素上的权重的形式提供时，对该机制的有监督学习。首先，我们通过一组公理来定义这一函数类别：每个元素的影响力受其权重限制、对权重的依赖是正则的、且没有任何元素是特殊的；在此基础上我们证明了一个表示定理：该函数类别恰好是加权多重集上某个传输度量的Lipschitz球，且该度量是被唯一确定的。

    arXiv:2610.16537v1 Announce Type: cross  Abstract: In a variety of multi-agent and physical environments, an element's update depends strongly on a few other elements and weakly on many, and both the identity and the number of the elements that matter change with the state of the system. Such graded sparse mechanisms are common, for instance, in flocks, crowds, learned particle simulators and the neighborhood aggregation of graph networks, yet they have received little systematic study as a class of functions with the grading given. In this paper we study the supervised learning of the mechanism when the grading is provided by the environment, as a weight on every element. First, we define the class by axioms, where each element's influence is capped by its weight, the dependence on the weights is regular, and no element is distinguished, and prove a representation theorem: the class is exactly the Lipschitz ball of a transport metric on weighted multisets, and the metric is forced by 
    
[^60]: 离散分布特征下最小二乘回归学习率调度策略的渐近风险

    Asymptotic Risk of Learning-Rate Schedules for Least-Squares Regression with Discretely Distributed Features

    [https://arxiv.org/abs/2610.16518](https://arxiv.org/abs/2610.16518)

    在离散分布特征的流式最小二乘回归中，任何固定相对长度的学习率冷却阶段都能以多项式因子改进风险，且最优风险仅通过乘积bT依赖于批量大小和时间范围。

    

    我们研究了随机梯度下降（SGD）在具有离散分布特征的流式最小二乘问题（Slock模型）上的最后迭代结果，该模型能够解耦各特征方向上的动力学。在幂律源条件和容量条件下，我们推导了常数学习率调度与预热-稳定-衰减（WSD）调度策略的精确有限时间风险及其渐近等价形式。通过调节基础学习率，所得的缩放律按 $\Theta(T^{-q})$ 衰减，其中 $T$ 为时间范围，指数 $q$ 与冷却长度 $c \in (0,1]$ 无关，但对于常数调度而言严格更小：任何固定相对长度 $c > 0$ 的冷却阶段，无论多短，都能以多项式因子改进风险，而冷却长度仅影响常数项，该常数在线性衰减（$c = 1$）时达到最小。最后，我们证明了最优风险仅通过乘积 $bT$ 依赖于批量大小 $b$ 和时间范围 $T$，直至达到某个临界批量大小为止。

    arXiv:2610.16518v1 Announce Type: cross  Abstract: We study the last iterate of stochastic gradient descent (SGD) on streaming least squares with discretely distributed features (Slock's model), which decouples the dynamics across eigendirections. Under power-law source and capacity conditions, we derive the exact finite-time risk and asymptotic equivalents of constant and Warmup-Stable-Decay (WSD) schedules. With a tuned base learning-rate, the resulting scaling law decays as $\Theta(T^{-q})$, where $T$ is the horizon, and $q$ is independent of the cooldown length $c \in (0,1]$ but is strictly smaller for the constant schedule: any cooldown of fixed relative length $c > 0$, however short, improves the risk by a polynomial factor, while its length only affects the constant, which is minimized by linear decay ($c = 1$). Finally, we show that the optimal risk depends on the batch size $b$ and horizon $T$ only through $bT$, up to a critical batch size.
    
[^61]: 哈密顿蒙特卡洛方法的局部自适应步长

    Locally adaptive step sizes for Hamiltonian Monte Carlo methods

    [https://arxiv.org/abs/2610.16506](https://arxiv.org/abs/2610.16506)

    本文在更一般的假设下阐释了HMC采样中的精确可逆局部自适应步长方法，提出了减少可逆性失败拒绝的替代误差度量，并将该技术扩展应用到标准HMC以及Metropolis校正的动力学Langevin/广义HMC（GHMC）方法中。

    

    Bou-Rabee等人（2026）在无掉头哈密顿蒙特卡洛（HMC）采样器的背景下引入了精确可逆的局部自适应步长，目的是在具有挑战性的目标分布中实现高性能的HMC采样。本文在更一般的假设下仔细解释并举例说明了局部自适应步长程序。文章引入了替代的误差度量，旨在通过降低可逆性失败拒绝的发生率来进一步提升MCMC性能。此外，本文还展示了如何在标准HMC以及经Metropolis校正的动力学Langevin/广义HMC（GHMC）中引入局部自适应步长。最后，本文提供了在不同基于哈密顿流的MCMC方法背景下，对所提出的误差度量的相对优点的数值研究。

    arXiv:2610.16506v1 Announce Type: cross  Abstract: Exact reversible locally adaptive step sizes are introduced by Bou-Rabee et al. (2026) in the context of the No-U-Turn Hamiltonian Monte Carlo (HMC) sampler for the purpose of high performance HMC sampling in challenging target distributions. This paper carefully explains and illustrates the locally adaptive step size procedure under more general assumptions. Alternative error measures are introduced with the aim of further enhancing MCMC performance by reducing the prevalence of reversibility failure rejections. In addition, the paper shows how to introduce the locally adaptive step sizes also within standard HMC and Metropolis-corrected kinetic Langevin/generalized HMC (GHMC). Finally, a numerical study of the relative merits of the proposed error measures in the context of different Hamiltonian flow-based MCMC methods is provided.
    
[^62]: 部分反馈下的在线个体化治疗选择：错误发现控制与极小极大最优性

    Online Individualized Treatment Selection under Partial Feedback: False Discovery Control and Minimax Optimality

    [https://arxiv.org/abs/2610.16464](https://arxiv.org/abs/2610.16464)

    本文提出PAITS方法，在仅观察到所分配治疗结果的部分反馈场景下进行在线个体化治疗选择，同时实现了渐近错误发现率控制、相对于总体预言机的次线性分配遗憾，并达到了极小极大最优性。

    

    我们研究了在仅能观察到所分配治疗对应结果时的在线个体化治疗选择问题。我们的目标是在限制受治个体中错误发现数量的同时，优先治疗那些在治疗下的潜在结果超过特定协变量对照参考值的个体。我们提出了部分反馈自适应个体化治疗选择方法（PAITS），该方法从历史数据初始化学习，并结合了局部零假设风险的序贯估计、随机化探索以及由反馈驱动的治疗选择阈值更新。在正则性条件下，我们建立了渐近错误发现率控制，以及相对于总体预言机的次线性间隙加权分配遗憾。我们进一步在规定的均匀干扰估计要求下建立了分配遗憾的极小极大下界，并给出了PAITS达到相应约束极小极大最优性的条件。

    arXiv:2610.16464v1 Announce Type: new  Abstract: We study online individualized treatment selection when only the outcome under the assigned treatment is observed. Our objective is to preferentially treat individuals whose potential outcomes under treatment exceed a covariate-specific control reference, while limiting false discoveries among treated individuals. We propose Partial-Feedback Adaptive Individualized Treatment Selection (PAITS), which initializes learning from historical data and combines sequential estimation of local null risks, randomized exploration, and feedback-driven updates of the treatment selection threshold. Under regularity conditions, we establish asymptotic false discovery rate control and sublinear gap-weighted assignment regret relative to a population oracle. We further establish a minimax lower bound on assignment regret under prescribed uniform nuisance-estimation requirements and give conditions under which PAITS attains the corresponding constrained mi
    
[^63]: 基于因果森林与Hoeffding函数分解解释异质性处理效应

    Explaining Heterogeneous Treatment Effects with Causal Forests and the Hoeffding Functional Decomposition

    [https://arxiv.org/abs/2610.16447](https://arxiv.org/abs/2610.16447)

    本文提出CausalTreeHFD，一种基于Hoeffding函数分解的可解释因果森林扩展方法，能够将CATE预测分解为单个协变量和成对交互作用的贡献，直接揭示驱动处理效应异质性的因素，并证明了该方法收敛于真实CATE函数Hoeffding分解的一致性。

    

    异质性处理效应的估计是因果推断的核心目标。然而，目前解决该任务的最先进算法基于机器学习原理，因此表现为黑箱。 consequently，识别驱动异质性的协变量仍然是一个具有挑战性的开放问题。为了解决这一局限性，我们提出了CausalTreeHFD，这是一种可解释的因果森林扩展方法，基于适用于树集成模型的Hoeffding函数分解。我们的方法将条件平均处理效应（CATE）的预测分解为单个协变量的贡献和成对交互作用的贡献，从而直接刻画了驱动处理效应异质性的因素。我们在因果森林分析的标准假设下，证明了CausalTreeHFD的一致性，即它收敛于底层CATE函数的Hoeffding分解。

    arXiv:2610.16447v1 Announce Type: new  Abstract: The estimation of heterogeneous treatment effects is a central objective of causal inference. However, state-of-the-art algorithms to solve this task are based on machine learning principles, and thus behave as black boxes. Consequently, the identification of covariates driving heterogeneity remains a challenging and open problem. To address this limitation, we introduce CausalTreeHFD, an interpretable extension of causal forests, based on the Hoeffding functional decomposition adapted to tree ensembles. Our approach decomposes predictions of conditional average treatment effects (CATE) into contributions from individual covariates and pairwise interactions, providing a direct characterization of the factors driving treatment effect heterogeneity. We establish the consistency of CausalTreeHFD, which converges towards the Hoeffding decomposition of the underlying CATE function, under standard assumptions for the analysis of causal forests
    
[^64]: 高维混合数据中基于矩的异常检测

    Moment-Based Anomaly Detection in High-Dimensional Mixed Data

    [https://arxiv.org/abs/2610.16285](https://arxiv.org/abs/2610.16285)

    该论文提出了一种基于矩变换和依赖感知聚合的MDI异常检测方法，能够检测高维混合数据中由位置偏移、变异、偏度等分布特性变化引起的稀疏异常及跨特征协调性偏差。

    

    我们提出了一种针对高维混合类型数据的异常检测方法，该方法通过标准化的基于矩的变换将异构特征映射到统一的尺度上，从而得到可解释的特征级异常分数。这些分数通过一个依赖感知的聚合步骤进行组合，构建出混合数据影响分数，该分数考虑了特征之间的相关性，并提供了一个统一的度量，用于衡量观测值偏离参考数据分布的程度。通过纳入高阶矩，MDI 不仅能检测由位置偏移引起的异常，还能检测由变异性、偏度及其他分布特征变化所引起的异常。该聚合机制使我们的方法既能捕捉集中于少数特征上的稀疏异常，也能捕捉分布于众多特征上的微弱且相互协调的偏差。我们实证研究了该分数的分布特性与阈值设定……

    arXiv:2610.16285v1 Announce Type: new  Abstract: We propose an anomaly detection method for high-dimensional mixed-type data that maps heterogeneous features to a common scale via standardized moment-based transformations, yielding interpretable feature-level anomaly scores. These scores are combined through a dependence-aware aggregation step to construct the Mixed-data Influence (MDI) score, accounting for correlations across features and providing a unified measure of how strongly an observation departs from the reference data distribution. By incorporating higher-order moments, the MDI can detect anomalies that arise not only from location shifts, but also from changes in variability, skewness, and other distributional characteristics. The aggregation mechanism allows our method to capture both sparse anomalies concentrated in a small number of features and weak, coordinated deviations distributed across many features. We empirically investigate the distribution and thresholding be
    
[^65]: TMLE与AIPW中样本拆分选择何时重要：交叉拟合、靶向估计与重复拆分

    When Sample-Splitting Choices Matter in TMLE and AIPW: Cross-Fitting, Targeting, and Repetition

    [https://arxiv.org/abs/2610.16192](https://arxiv.org/abs/2610.16192)

    该研究通过已知真值模拟系统比较了八种TMLE和AIPW估计器变体，发现对两个干扰模型均进行交叉拟合是实现有效置信区间覆盖率的关键（样本内拟合倾向得分会导致严重覆盖不足），且在近正性违规条件下AIPW估计器的覆盖率表现一致优于TMLE估计器。

    

    采用机器学习干扰模型的靶向最大似然估计（TMLE）和增强逆概率加权（AIPW）的各种变体在以下方面存在差异：使用哪些观测数据训练模型、如何校正偏差，以及是否重复数据拆分。在二元结局平均处理效应的已知真值模拟中，八种主要估计器共享每个设置下的学习器库、内部折数和截断界限；覆盖率差异依据预先设定的两个百分点容差进行评判。在低复杂度学习器库和良好协变量重叠（n = 3000）的条件下，除半样本基线外，所有估计器的覆盖率表现等价（0.928–0.939）。当在主数据生成机制上仅使用一个随机森林时，在样本内拟合倾向得分的估计器出现严重覆盖不足（0.12–0.48），而对两个干扰模型均进行交叉拟合则将覆盖率提升至0.91–1.00（其中两个出现过度覆盖）。在近正性（near-positivity）违规条件下，每种AIPW估计器的覆盖率均优于每种TMLE估计器，而未加权……

    arXiv:2610.16192v1 Announce Type: cross  Abstract: Variants of targeted maximum likelihood estimation (TMLE) and augmented inverse-probability weighting (AIPW) with machine-learning nuisance models differ in which observations train them, how bias is corrected, and whether splits are repeated. In known-truth simulations of a binary-outcome average treatment effect, eight headline estimators shared each setting's library, inner folds and truncation bound; coverage differences were judged against a prespecified two-percentage-point margin. With a low-complexity library and good overlap (n = 3000), all but a half-sample baseline covered equivalently (0.928-0.939). With one random forest on the primary data-generating mechanism, estimators fitting the propensity in-sample undercovered severely (0.12-0.48), whereas cross-fitting both nuisances raised coverage to 0.91-1.00, two overcovering. Under near-positivity, every AIPW estimator covered better than every TMLE estimator with the unweigh
    
[^66]: 具有推断遗憾保证的高斯过程赌博机优化探索策略

    Exploration strategies for Gaussian process bandit optimization with inference regret guarantees

    [https://arxiv.org/abs/2610.16187](https://arxiv.org/abs/2610.16187)

    本文提出了一种通过偶尔将采集查询替换为探索查询即可融入任意贝叶斯优化流程的探索策略，并借助填充距离为高斯过程后验标准差和推断遗憾推导出理论上界，从而保证算法最终推荐结果的质量。

    

    现有的贝叶斯优化（BO）累积遗憾保证并不一定能刻画算法最终输出的质量。这一差距在含噪声评估的场景下尤为关键，因为此时目标是在算法结束时返回一个高质量的单一推荐。为解决这一问题，我们引入了高效的探索策略，这些策略可以通过偶尔将采集查询替换为探索查询的方式融入任何贝叶斯优化流程中。为验证该方法，我们通过查询点的填充距离（fill distance）来界定高斯过程后验标准差在整个搜索空间上的上界，该距离衡量了尚未被探索的最大区域。随后，我们利用这一结果推导出推断遗憾（即全局最优值与算法返回点处函数值之间的差距）的上界。实证结果表明，这些探索策略在合成基准和超参数优化任务上改进了现有贝叶斯优化算法的最终推荐结果。

    arXiv:2610.16187v1 Announce Type: new  Abstract: Existing cumulative regret guarantees for Bayesian optimization (BO) do not necessarily characterize the quality of the algorithm's final output. This gap is critical under noisy evaluations, where the goal is to return a single high-quality recommendation. To address it, we introduce efficient exploration strategies that can be incorporated into any BO procedure by occasionally replacing the acquisition query with an exploration query. To validate this approach, we bound the Gaussian process posterior standard deviation over the search space via the fill distance of the queries, which measures the largest region left unexplored. We then use this result to derive upper bounds on the inference regret, i.e., the gap between the global optimum and the function value at the algorithm's returned point. Empirically, these exploration strategies improve the final recommendations of existing BO algorithms on synthetic benchmarks, hyperparameter 
    
[^67]: 具有高斯转移的无界随机博弈中的均衡学习

    Equilibrium Learning in Unbounded Stochastic Games with Gaussian Transitions

    [https://arxiv.org/abs/2610.16149](https://arxiv.org/abs/2610.16149)

    该论文针对状态空间无界、转移噪声为高斯的随机博弈，提出利用单条连续轨迹更新高斯过程漂移模型并求解逐步精化的有限博弈来学习均衡，并证明了均衡选择稳定性与一致KL精度足以保证诱导值和近似纳什行为的区域一致收敛。

    

    当多个决策者无限期地交互且系统状态没有规定边界时，从数据中学习均衡是十分困难的。我们研究了具有紧动作集、其未知非线性漂移通过高斯转移噪声进行观测的折扣一般和随机博弈。所提出的方法利用单条不间断的轨迹来更新漂移的高斯过程模型，并求解逐步精化的有限博弈。我们证明了一个摄动结果，表明均衡选择的稳定性，加上高斯转移模型的一致KL精度，足以保证原博弈中诱导值和近似纳什行为实现区域上的一致收敛。该区域规定了保证被评估的位置；而策略与动力学仍继续作用于整个欧几里得状态空间。对于GP估计器，同时置信界可产生由后验不确定性控制的收敛速率……

    arXiv:2610.16149v1 Announce Type: new  Abstract: Learning equilibria from data is difficult when several decision makers interact indefinitely and the system state has no prescribed bounds. We examine discounted general-sum stochastic games with compact action sets and an unknown nonlinear drift observed through Gaussian transition noise. The proposed procedure uses a single uninterrupted trajectory to update Gaussian-process models of the drift and to solve successively refined finite games. We prove a perturbation result showing that stability of equilibrium selection, together with uniform KL accuracy of Gaussian transition models, is sufficient for regional uniform convergence of the induced values and approximate Nash behavior in the original game. The region specifies where the guarantees are evaluated; strategies and dynamics continue to act on the entire Euclidean state space. For GP estimators, simultaneous confidence bounds yield a rate controlled by posterior uncertainty, ke
    
[^68]: 使用ReLU神经网络对任意响应变量进行分布回归

    Distributional Regression for Arbitrary Response Variables Using ReLU Neural Networks

    [https://arxiv.org/abs/2610.16142](https://arxiv.org/abs/2610.16142)

    本文提出了一个基于可测集条件事件概率的统一分布回归框架，通过有向无环图约束的估计器和ReLU网络实现了对任意响应变量的分布回归，并建立了克服维度灾难的误差界与收敛速率。

    

    本文针对任意响应变量的分布回归提出了一个统一框架。我们不依赖于标量阈值或欧几里得结构，而是对由可测集索引的条件事件概率进行建模，这将分布回归扩展到一般的响应空间，同时保持了与经典基于事件的表述的直接联系。在该框架内，我们研究了两个互补的估计器：一个基于对事件指示变量进行分别回归的初步估计器，以及一个通过对有限集合族构建有向无环图（DAG）来强制实现包含一致性预测的约束估计器。我们为这两个估计器建立了一般误差界，并进一步将结果专门化到稠密ReLU网络，得到了克服维度灾难的显式收敛速率。仿真研究和真实数据实验进一步证明了所提方法的实际有效性。

    arXiv:2610.16142v1 Announce Type: new  Abstract: This paper develops a unified framework for distributional regression with arbitrary response variables. Rather than relying on scalar thresholds or Euclidean structure, we model conditional event probabilities indexed by measurable sets, which extends distributional regression to general response spaces while retaining a direct connection to classical event-based formulations. Within this framework, we study two complementary estimators: a preliminary estimator based on separate regression of event indicators, and a constrained estimator that enforces inclusion-consistent predictions through a directed acyclic graph (DAG) over a finite set family. We establish general error bounds for both estimators and further specialize the results to dense ReLU networks, yielding explicit convergence rates overcoming the curse of dimensionality. Simulation studies and real-data experiments further demonstrate the practical effectiveness of the propo
    
[^69]: 协变量偏移下的反向保形预测

    Backward Conformal Prediction under Covariate Shift

    [https://arxiv.org/abs/2610.16048](https://arxiv.org/abs/2610.16048)

    该论文通过重要性加权将反向保形预测（BCP）扩展到协变量偏移情形，并提出在排序后的非一致性分数向量而非高维输入上估计密度比，从而在保持理论保证的同时显著改进了失覆盖率的估计效果。

    

    反向保形预测（BCP）固定预测集的大小并估计其失覆盖率，但其假设校准数据与测试数据是可交换的。我们通过重要性加权将BCP扩展到协变量偏移情形，并在密度比已知时保持其理论保证。我们证明了BCP给出的紧凑失覆盖率报告仅取决于分配给未被覆盖的校准样本的相对权重，这解释了为什么更好的域分类器有时反而会产生更差的估计。受BCP仅依赖于非一致性分数这一事实的启发，我们提出在排序后的分数向量上估计密度比，而非在高维输入上进行估计。我们证明在可解释的条件下，该方法能够渐近地恢复相同的报告，并在不满足该条件时刻画了由此产生的偏差。在模拟分布偏移和基准分布偏移上的实验表明，所提出的方法改进了BCP的失覆盖率估计，同时基于分数的加权方法……

    arXiv:2610.16048v1 Announce Type: cross  Abstract: Backward conformal prediction (BCP) fixes the size of a prediction set and estimates its miscoverage, but assumes calibration and test data are exchangeable. We extend BCP to covariate shift through importance weighting, preserving its guarantees when the density ratio is known. We show that BCP's tight miscoverage report depends only on the relative weight assigned to missed calibration examples, explaining why a better domain classifier can sometimes produce a worse estimate. Motivated by the fact that BCP depends only on nonconformity scores, we propose estimating density ratios on sorted score vectors rather than high-dimensional inputs. We prove that this approach recovers the same report asymptotically under an interpretable condition and characterize the resulting bias otherwise. Experiments with simulated and benchmark distribution shifts show that the proposed method improves BCP's miscoverage estimates, while score-based weig
    
[^70]: 恒定步长带动量随机梯度下降的非线性时间序列理论

    A Nonlinear Time-Series Theory for Constant-step SGD with Momentum

    [https://arxiv.org/abs/2610.16034](https://arxiv.org/abs/2610.16034)

    该论文首次将恒定步长带动量的SGDM视为非线性时间序列，证明了参数-动量联合过程的几何遗忘性与唯一平稳因果解的存在性，并由此导出了平均化SGDM的不变性原理、淬火高斯极限以及小学习率下最后迭代的高斯极限。

    

    恒定步长带动量的随机梯度下降（SGDM）通过引入一个动量状态来增强参数递推过程，该动量状态携带对过去随机梯度的几何衰减记忆。我们将这一耦合递推作为一个非线性时间序列进行研究，并为其平稳行为和分布极限开发了一个理论框架。在 $p > 1$ 的有限 $p$ 阶矩条件下，参数-动量联合过程会以几何速度遗忘其初始化，从而产生唯一的平稳因果解。在二阶矩条件下，这种收缩性导出了不变性原理以及平均化 SGDM 的淬火（quenched）高斯极限，而小学习率分析则给出了最后一步迭代的高斯极限。在随机曲率二次模型中，我们进一步研究了初始化遗忘速率与平稳方差及平均化涨落之间的关系。

    arXiv:2610.16034v1 Announce Type: new  Abstract: Constant-step stochastic gradient descent with momentum (SGDM) augments the parameter recursion with a momentum state that carries geometrically decaying memory of past stochastic gradients. We study this coupled recursion as a nonlinear time series and develop a framework for its stationary behavior and distributional limits. The joint parameter-momentum process forgets its initialization geometrically under a finite $p$th moment for $p > 1$, yielding a unique stationary causal solution. Under a second-moment condition, this contraction leads to an invariance principle and quenched Gaussian limit for averaged SGDM, while a small-learning-rate analysis gives a Gaussian limit for the last iterate. In a stochastic-curvature quadratic model, we further study how the rate of initialization forgetting relates to stationary variance and averaged fluctuations.
    
[^71]: 短程与长程依赖下在线期望分位数估计的极限理论

    Limit Theory for Online Expectile Estimation under Short and Long Range Dependence

    [https://arxiv.org/abs/2610.16030](https://arxiv.org/abs/2610.16030)

    本文建立了恒定步长随机梯度算法在线估计期望分位数的极限理论：短程依赖下迭代值以 √γ 尺度渐近高斯，长记忆下尺度变为 γ^(β-1/2) 且极限由分数布朗运动积分给出，并推广至多水平联合极限。

    

    期望分位数是一类对尾部敏感的分布概括统计量，在金融风险度量与预测中具有重要应用，但针对依赖数据进行在线估计的统计理论仍然十分有限。本文研究了一种采用恒定学习率 γ 的单遍随机梯度算法，用于估计平稳因果过程的边缘期望分位数。该递归方程存在唯一的平稳因果解，并能以几何速度遗忘其初始化的影响。在短程依赖条件下，其最终迭代值在 √γ 尺度下渐近服从高斯分布，方差由总体曲率和期望分位数得分的长期方差决定。对于长记忆线性过程，尺度变为 γ^(β-1/2)ℓ(1/γ)，其高斯极限由分数布朗运动积分产生。此外，作者进一步推导了多个期望分位数水平下的联合长记忆极限，并刻画了其极限行为。

    arXiv:2610.16030v1 Announce Type: new  Abstract: Expectiles are tail-sensitive distributional summaries with important applications in financial risk measurement and forecasting, yet statistical theory for their online estimation from dependent data remains limited. We study a one-pass, stochastic-gradient algorithm with constant learning rate $\gamma$ for estimating the marginal expectile of a stationary causal process. The recursion admits a unique stationary causal solution and geometrically forgets its initialization. Under short-range dependence, its last iterate is asymptotically Gaussian at the $\sqrt{\gamma}$ scale, with variance governed by the population curvature and the long-run variance of the expectile score. For long-memory linear processes the scale changes to $\gamma^{\beta-1/2}\ell(1/\gamma)$, with a Gaussian limit arising from a fractional Brownian motion integral. We further derive the joint long-memory limit for multiple expectile levels and characterize its limiti
    
[^72]: 物理引导的机器学习用于堆浸中可变水力条件下的饱和度控制

    Physics-guided machine learning for saturation control across variable hydraulic conditions in heap leaching

    [https://arxiv.org/abs/2610.15947](https://arxiv.org/abs/2610.15947)

    该研究开发了一种参数条件化的物理引导机器学习代理模型，可基于不同水力参数快速预测堆浸饱和度演化，实现了与调谐PI控制器精度相当且无超调的饱和度预测控制。

    

    堆浸水力学在矿床不同区域之间存在差异，因此需要能够适应不同物料特性的快速预测模型。本研究开发了一种参数条件化的物理引导机器学习（PGML）代理模型，并在模拟饱和度控制中展示了其应用。一个包含4096条数值Richards方程轨迹的数据集被划分为2867个训练样本、614个验证样本和615个评估样本。该网络根据当前饱和度剖面、灌溉速率和四个水力参数预测逐小时的饱和度剖面，并采用近似Gardner质量守恒惩罚项进行训练。每日预测控制器在24小时时间范围内比较了200个恒定灌溉速率。在全部评估案例中，全周期跟踪偏差的中位数为0.70饱和度目标的4.9%，与调谐后的比例-积分（PI）控制器相当，而PGML控制器在（摘要在此处截断）中无超调地达到了目标。

    arXiv:2610.15947v1 Announce Type: cross  Abstract: Heap-leaching hydraulics vary between regions of an ore bed, motivating fast prediction models that accommodate different material properties. This study develops a parameter-conditioned physics-guided machine-learning (PGML) surrogate and demonstrates its use in simulated saturation control. A dataset of 4096 numerical Richards-equation trajectories was divided into 2867 training, 614 validation and 615 evaluation cases. The network predicts hourly saturation profiles from the current profile, irrigation rate and four hydraulic parameters, and is trained with an approximate Gardner mass-balance penalty. A daily predictive controller compared 200 constant irrigation rates over a 24-hour horizon. Across the evaluation cases, the median full-cycle tracking deviation was 4.9% of a 0.70 saturation target, equal to that of a tuned proportional-integral (PI) controller, while the PGML controller reached the target without overshoot in the me
    
[^73]: 多流形聚类的极小极大速率与改进的谱收敛性

    Minimax Rates and Improved Spectral Convergence for Multi-Manifold Clustering

    [https://arxiv.org/abs/2610.15892](https://arxiv.org/abs/2610.15892)

    本文为张量化拉普拉斯-贝尔特拉米算子的特征对估计建立了极小极大最优速率 $n^{-s/(d+2s)}$，并证明流形无关的图拉普拉斯估计器以更快的速率 $n^{-1/(2.5d-1)}$ 收敛，从而改进了多流形学习的谱收敛理论。

    

    我们研究了从可能相交的多流形混合分布中采样的 $n$ 个独立数据点上，张量化、密度加权拉普拉斯-贝尔特拉米算子的特征对估计问题。设 $d$ 表示流形分量的最大内在维数，$s\geq2$ 表示采样密度的光滑度。对于简单的非平凡特征对，我们建立了逼近张量化拉普拉斯-贝尔特拉米算子特征值和特征函数的极小极大速率 $n^{-s/(d+2s)}$，并证明了当支撑流形已知时，基于核密度估计的连续体代入估计器可以达到相同的速率。我们进一步分析了与流形无关的图拉普拉斯估计器，并证明其特征对在对数因子范围内以 $n^{-1/(2.5d-1)}$ 的速率收敛到张量化拉普拉斯-贝尔特拉米算子的特征对，这改进了多流形学习此前已知的谱收敛速率。

    arXiv:2610.15892v1 Announce Type: new  Abstract: We study eigenpair estimation for the tensorized, density-weighted Laplace--Beltrami operator from $n$ independent data points sampled from a multi-manifold mixture distribution where manifolds may intersect. Let $d$ denote the largest intrinsic dimension of the manifold components and $s\geq2$ the smoothness of the sampling densities. For simple, nontrivial eigenpairs, we establish the minimax rate $n^{-s/(d+2s)}$ for approximating the eigenvalue and eigenfunction of the tensorized Laplace--Beltrami operator, and show that a continuum plug-in estimator based on kernel density estimation attains the same rate when the supporting manifolds are known. We further analyze a manifold-agnostic graph Laplacian estimator and prove that its eigenpairs converge to those of the tensorized Laplace--Beltrami operator at rate $n^{-1/(2.5d-1)}$, up to logarithmic factors, improving upon previously known spectral convergence rates for multi-manifold lea
    
[^74]: 稀疏自编码器特征发现分数到底识别了什么？从激活到干预

    What Do Sparse Autoencoder Feature Discovery Scores Identify? From Activations to Interventions

    [https://arxiv.org/abs/2610.15850](https://arxiv.org/abs/2610.15850)

    论文证明基于激活共变的稀疏自编码器特征发现证据通常无法预测干预（如转向或消融）的真实效果，并提出基于梯度的干预归因方法，其对消融效应符号的预测准确率达91-97%，远优于线性探针预测的17-35%。

    

    稀疏自编码器（SAE）特征通常是根据其激活与行为如何共变而被发现的，但它们在实际中常常通过转向或消融等干预方式被使用。我们将干预归因，即特定干预对模型输出的一阶效应——作为我们的估计目标。对于一大类响应函数，我们证明观测性的特征发现证据通常无法识别这一归因，并推导出线性探针预测与干预归因之间差距的精确的基于梯度的分解。在四组模型-SAE配对中，探针与归因幅值之比的中位数从数十到数百不等，而归因则紧密跟踪小幅度的转向操作。对于具有非零效应的单潜变量消融，梯度预测在91-97%的情况下与实际效应符号一致，而探针所暗示的变化仅有17-35%一致。基于归因选择的特征集合也能产生更大的绝对……（原文摘要在此处截断）

    arXiv:2610.15850v1 Announce Type: cross  Abstract: Sparse autoencoder (SAE) features are typically discovered from how their activations co-vary with behaviour, yet they are often used through interventions such as steering or ablation. We take intervention attribution, the first-order effect of a specified intervention on the model output, as our estimand. For a broad class of response functions, we show that observational feature-discovery evidence generally does not identify this attribution, and derive an exact gradient-based decomposition of the gap between linear-probe predictions and intervention attribution. Across four model-SAE pairs, median probe-to-attribution magnitude ratios range from tens to hundreds, while attributions closely track small-step steering. For single-latent ablations with nonzero effects, gradient predictions match the realized sign in 91-97% of cases, compared with 17-35% for probe-implied changes. Attribution-selected sets also produce larger absolute m
    
[^75]: ML-CLAPPS：类别优先级下不平衡多标签学习的概率框架

    ML-CLAPPS: A Probabilistic Framework for Imbalanced Multilabel Learning under Class Prioritization

    [https://arxiv.org/abs/2610.15816](https://arxiv.org/abs/2610.15816)

    提出了ML-CLAPPS，一种可学习的概率采样框架，通过优化训练实例的分布来应对多标签类别不平衡问题，并借助用户定义的重要性向量实现对高风险病症（如心电图分类中的关键病症）的优先处理。

    

    真实世界的数据集通常表现出严重的类别不平衡，这会降低深度学习模型的性能，尤其是在稀有类别上。虽然不平衡问题在二分类和多分类设置中已得到充分研究，但由于标签共现关系的复杂性，该问题在多标签场景中更具挑战性。在心电图（ECG）分类等临床应用中，这一挑战还叠加了另一个关键难题：无论高风险病症的流行程度如何，都需要对其进行优先处理，以降低假阴性带来的高昂代价。我们提出了多标签类别优先概率采样器（ML-CLAPPS），这是一种可学习的采样器，通过优化训练实例上的分布来控制模型对不同类别的暴露程度。该方法在解决多标签不平衡问题的同时，还能通过用户定义的重要性向量（IV）实现对重要病症的优先处理。在真实世界多标签心电图图像上的实证评估……

    arXiv:2610.15816v1 Announce Type: cross  Abstract: Real-world datasets often exhibit severe class imbalance, degrading the performance of deep learning models, particularly on rare classes. While imbalance is well studied in binary and multiclass settings, it is substantially more challenging in multilabel problems due to complex label co-occurrence. In clinical applications such as ECG classification, this challenge is coupled with a second critical challenge: the need to prioritize high-risk conditions, regardless of their prevalence, to mitigate the high cost of false negatives. We introduce the Multilabel Class-Prioritized Probabilistic Sampler (ML-CLAPPS), a learnable sampler that optimizes a distribution over training instances to control the model's exposure to different classes. This approach addresses multilabel imbalance while enabling prioritization of significant conditions via a user-defined importance vector (IV). Empirical evaluation on a real-world multilabel ECG image 
    
[^76]: 自适应实验中策略改进价值的推断

    Inference on the Value of Policy Improvement in Adaptive Experiments

    [https://arxiv.org/abs/2610.15789](https://arxiv.org/abs/2610.15789)

    本文提出了自适应实验数据下策略改进价值（VPI）的渐近正态估计量，通过在历史数据上拟合加权经验福利最大化策略并用逆倾向加权进行评估，克服了数据相关性、倾向得分衰减以及最优策略非唯一时的非正则性等难题，为判断是否值得部署个性化处理策略提供了统计推断工具。

    

    本文针对自适应收集的数据，开发了策略改进价值（VPI）的渐近正态估计量。该价值被定义为某一类策略中最优处理策略的价值与基线策略价值之间的差距，用于衡量是否值得部署个性化处理策略。构建VPI的估计量天然比估计固定策略的价值更为困难。这是因为，除了要应对自适应实验中数据相关性和倾向得分衰减的问题之外，当最优策略不唯一时，任何估计量还必须处理目标估计对象的非正则性（即路径可微性的缺失）。我们提出的估计量的工作原理是：先在历史数据上依次拟合加权经验福利最大化策略，然后在未来的样本上使用逆倾向加权对其进行评估。我们既不假设……

    arXiv:2610.15789v1 Announce Type: new  Abstract: In this paper, we develop an asymptotically normal estimator for the value of policy improvement (VPI) under adaptively collected data. This value, which is defined as the gap between the value of the best treatment policy in a class and that of a baseline policy, gauges whether or not it is worth deploying a personalized treatment policy. Building an estimator for the VPI is naturally more difficult than estimating the value of a fixed policy. This is because, in addition to contending with the correlated data and decaying propensities associated with adaptive experiments, any estimator must also handle the non-regularity, or lack of pathwise differentiability, of the underlying estimand when the optimal policy is non-unique. Our proposed estimator works by sequentially fitting a weighted empirical welfare maximizing policy on historical data and then evaluating it using inverse propensity weighting on future samples. We neither assume 
    
[^77]: 高维自适应稳健置信集

    Adaptive Robust Confidence Sets in High Dimensions

    [https://arxiv.org/abs/2610.15723](https://arxiv.org/abs/2610.15723)

    该论文证明了在污染比例未知的高维Huber污染模型下，通过同时量化所有一维投影的不确定性可构造信息论最优的自适应置信集，但该构造不仅存在显著的自适应统计代价，还因统计查询下界而面临计算不可行的障碍。

    

    我们研究了在污染比例未知的情况下，具有单位协方差的多变量高斯分布的均值向量在Huber污染模型下的自适应置信集。我们证明了通过同时量化所有一维投影的不确定性，可以构造出信息论意义上最优的自适应置信集。所得到的置信集达到了最优直径，并揭示了与极小极大估计误差相比存在显著的自适应代价。尽管这一构造在统计上是最优的，但在计算上是不可行的。我们建立了一个统计查询（SQ）下界，表明要达到直径 $O(m^{-1/2})$ 需要指数级数量的查询，或查询容差为 $d^{-\Omega(m)}$。因此，与稳健均值估计不同，对未知污染比例的自适应在SQ框架内既带来统计代价，又带来计算上的障碍。最后，我们……（原文截断）

    arXiv:2610.15723v1 Announce Type: cross  Abstract: We study adaptive confidence sets for mean vectors of multivariate Gaussians with identity covariance under Huber contamination when the contamination proportion is unknown. We show that an information-theoretically optimal adaptive confidence set can be constructed by simultaneously quantifying uncertainty across all one-dimensional projections. The resulting confidence set achieves the optimal diameter and reveals a substantial adaptation cost compared with the minimax estimation error.   Although statistically optimal, this construction is computationally infeasible. We establish a Statistical Query (SQ) lower bound showing that achieving diameter $O(m^{-1/2})$ requires either exponentially many queries or query tolerance $d^{-\Omega(m)}$. Thus, in contrast to robust mean estimation, adaptation to the unknown contamination proportion incurs both a statistical cost and a computational barrier within the SQ framework.   Finally, we gi
    
[^78]: 关于截断高斯、乘积柯西、玻尔兹曼与Softmax策略的一致得分界与Lipschitz连续性的注记

    Notes on Uniform Policy Score Bounds and Lipschitz Continuity for Truncated Gaussian, Product Cauchy, Boltzmann, and Softmax Policies

    [https://arxiv.org/abs/2610.15690](https://arxiv.org/abs/2610.15690)

    本文在明确参数化条件下，证明了截断高斯、柯西、玻尔兹曼和softmax策略的一致得分界与Lipschitz连续性。

    

    我们在明确给定的参数化条件下，为截断高斯、柯西、玻尔兹曼以及softmax策略建立了一致得分界和Lipschitz连续性。

    arXiv:2610.15690v1 Announce Type: new  Abstract: We establish uniform score bounds and Lipschitz continuity for truncated Gaussian, Cauchy, Boltzmann, and softmax policies under explicitly stated parameterizations.
    
[^79]: 显式惩罚何时能改进流匹配？

    When do explicit penalties improve flow matching?

    [https://arxiv.org/abs/2610.15689](https://arxiv.org/abs/2610.15689)

    该论文以期望平方 2-Wasserstein 误差为度量，在希尔伯特空间框架下推导出几何准则并显式求解统计比较，证明了流匹配中的显式惩罚（如物理约束）恰好在其带来的方差减少超过平方偏差增加时才能提升生成精度。

    

    流匹配可以直接从数据中学习结构，这引出了一个重要问题：额外的约束何时能改进生成效果。我们研究了编码此类约束的惩罚项，并以物理信息流匹配作为动机示例。更紧密地满足约束并不一定使生成的分布更准确。因此，我们通过期望平方 2-Wasserstein 误差来比较带惩罚与不带惩罚的生成建模。我们的分析涵盖了希尔伯特空间设置中的向量值数据和函数型数据。我们首先推导出同时考虑约束违反程度以及样本投影到约束集后分布的几何准则。然后，我们针对特定的速度场类别和约束，通过流匹配显式求解了这一统计比较问题。在该模型中，恰好当估计方差的减少超过平方偏差的增加时，惩罚才能提高生成精度。

    arXiv:2610.15689v1 Announce Type: new  Abstract: Flow matching can learn structure directly from data, raising the question of when additional constraints improve generation. We study penalties that encode such constraints, with physics-informed flow matching as a motivating example. Satisfying a constraint more closely need not make the generated distribution more accurate. We therefore compare penalised and unpenalised generative modelling through their expected squared $2$-Wasserstein error. Our analysis covers both vector-valued and functional data in a Hilbert-space setting. We first derive geometric criteria that account for constraint violations and the distribution of samples after projection onto the constraint set. We then solve the statistical comparison explicitly for specific velocity-field classes and constraints via flow matching. In this model, penalisation improves accuracy exactly when the reduction in estimation variance outweighs the increase in squared bias. The re
    
[^80]: 递归合成数据下的筛选崩溃

    Screening Collapse under Recursive Synthetic Data

    [https://arxiv.org/abs/2610.15617](https://arxiv.org/abs/2610.15617)

    该论文首次揭示递归合成数据训练中的“筛选崩溃”现象：伪标签目标逐代漂移会侵蚀或逆转活跃与非活跃变量的分数差距，导致真实标签筛选本应保留的重要变量被丢弃，且这种协方差加权损失无法通过后续重新拟合消除，从而先于并加剧传统的模型崩溃。

    

    递归合成数据训练会使拟合的模型发生退化，但其对模型拟合之前的变量筛选阶段的影响却鲜有研究。当每一代为下一代标注数据时，边际筛选器是针对当前的伪标签目标而非原始响应变量来对变量进行排序。随着代次推进，这种目标漂移会侵蚀甚至逆转活跃变量与非活跃变量之间的分数差距，导致筛选过程丢弃真实标签筛选本应保留的变量。我们将这一过程称为“筛选崩溃”（screening collapse）。在具有最小范数递归生成器的高斯线性模型中，我们建立了有限样本筛选界，刻画了后期代次的信息损失极限，并推导了一个极小极大不可能性结果。筛选崩溃可以先于并加剧普通的模型崩溃：被遗漏的变量会产生一种协方差加权的损失，任何仅限于已选变量的重新拟合都无法消除该损失，而递归……（摘要在此处截断）

    arXiv:2610.15617v1 Announce Type: new  Abstract: Recursive synthetic-data training can degrade a fitted model, but its effects on the variable-screening stage that precedes model fitting are less understood. When each generation labels data for the next, a marginal screener ranks variables against the current pseudo-label target rather than the original response. Across generations, this target drift can erode or reverse the active--inactive score margin and cause screening to discard variables that clean-label screening would retain. We call this process \emph{screening collapse}. In a Gaussian linear model with minimum-norm recursive generators, we establish finite-sample screening bounds, characterize a late-generation information-loss limit, and derive a minimax impossibility result. Screening collapse can precede and compound ordinary model collapse: omitted variables incur a covariance-weighted loss that no refit restricted to the selected variables can eliminate, while recursive
    
[^81]: 面向无监督域自适应的联合对比-对抗表示学习

    Joint Contrastive-Adversarial Representation Learning for Unsupervised Domain Adaptation

    [https://arxiv.org/abs/2610.15613](https://arxiv.org/abs/2610.15613)

    提出JCAR两阶段框架，将对比学习与对抗域对齐相结合以学习源域和目标域的共同表示，并建立了非渐近超额风险界，证明当无标签样本足够多时，下游预测可达到 $\widetilde{\mathcal{O}}(m^{-\beta/(2\beta+d)})$ 的收敛速率。

    

    有效的无监督域自适应（UDA）需要一个能够同时从有标签源域和无标签目标域中捕获可迁移结构的表示。我们提出了联合对比-对抗表示学习（JCAR），这是一个两阶段框架：第一阶段通过将对比学习与对抗域对齐相结合，从无标签源域和目标域样本的并集中学习一个共同表示；第二阶段冻结该表示，并拟合一个由源域监督的预测器。我们为该过程建立了非渐近超额风险界，该风险界将表示学习和域判别器的估计耦合在一起，同时将下游预测分离出来，并包含显式的神经网络逼近误差。在所述假设及常见的Hölder光滑性β≥1条件下，该界指出了一个无标签样本量足够大的情形，此时下游预测的收敛速率为 $\widetilde{\mathcal{O}}(m^{-\beta/(2\beta+d)})$。

    arXiv:2610.15613v1 Announce Type: new  Abstract: Effective unsupervised domain adaptation (UDA) requires a representation that captures transferable structure from both the labeled source domain and the unlabeled target domain. We introduce Joint Contrastive-Adversarial Representation Learning (JCAR), a two-stage framework that learns a common representation from the union of unlabeled source and target samples by combining contrastive learning with adversarial domain alignment, and then freezes the representation to fit a source-supervised predictor. We establish a non-asymptotic excess-risk bound for this procedure that couples representation and domain-discriminator estimation while separating downstream prediction, with explicit neural-network approximation errors. Under the stated assumptions and common H\"older smoothness $\beta\geq1$, the bound identifies a sufficiently large unlabeled-sample regime in which the downstream rate is $\widetilde{\mathcal{O}}(m^{-\beta/(2\beta+d)})$
    
[^82]: 协变量偏移下的潜在空间重要性加权

    Latent-Space Importance Weighting under Covariate Shift

    [https://arxiv.org/abs/2610.15571](https://arxiv.org/abs/2610.15571)

    该论文提出一种针对协变量偏移的迁移学习方法，在低维潜在空间中将表示学习与样本重加权相结合，并给出了目标域超额风险率等理论保证与实验验证。

    

    在本工作中，我们提出了一种针对协变量偏移的迁移学习算法，将样本重加权与表示学习相结合。我们方法的一个关键特征是使用低维潜在空间进行建模，这与直接在高维环境空间中操作的传统方法形成对比。我们的方法通过估计学习到的潜在特征的密度比来进行下游的加权预测。对于具有正则学习空间预测分数和密度比的严格充分学习表示，我们利用独立的数据划分建立了明确的目标域超额风险率。我们在给定的平滑性条件下给出了维度比较，并对真实权重和固定预测器进行了方差比较。一般的有限样本不等式保留了表示质量误差。在模拟数据集和真实世界数据集上的实验表明，结合学习到的表示与潜在加权能够带来性能提升。

    arXiv:2610.15571v1 Announce Type: new  Abstract: In this work, we propose a transfer learning algorithm for covariate shift, integrating sample reweighting with representation learning. A key feature of our approach is the use of a lower-dimensional latent space for modeling, in contrast to conventional methods that operate directly in the high-dimensional ambient space. Our method estimates the density ratio of learned latent features for downstream weighted prediction. We establish explicit target-domain excess-risk rates with independent data folds for strictly sufficient learned representations with regular learned-space prediction scores and density ratios. We give dimension comparisons under stated smoothness conditions and a variance comparison for true weights and a fixed predictor. General finite-sample inequalities retain representation-quality errors. Experiments on simulated and real-world datasets demonstrate gains from combining learned representations and latent weightin
    
[^83]: 面向具有个体特异异质性的非线性纵向预测的深度神经网络混合模型

    Deep Neural Mixed Models for Nonlinear Longitudinal Prediction with Subject-Specific Heterogeneity

    [https://arxiv.org/abs/2610.15323](https://arxiv.org/abs/2610.15323)

    提出深度广义神经网络混合模型（DGNMM），通过将深度神经网络与可训练的正则化个体特异效应相结合来扩展GLMM框架，从而在存在个体异质性的纵向健康数据中提升非线性预测能力。

    

    纵向健康数据通常表现出个体内部相关性、个体特异异质性以及复杂的非线性预测变量—响应关系，这些特征可能无法被传统混合模型充分捕捉。我们提出了一种深度广义神经网络混合模型（DGNMM），该模型扩展了GLMM框架：利用深度神经网络学习总体水平的非线性关系，同时引入可直接训练并经过正则化的个体特异效应来捕捉个体异质性。该模型采用面向预测的随机优化方法进行估计，并对神经网络参数和个体特异效应分别施加正则化。这一建模方式使我们能够探究在存在个体特异异质性的情况下，增强的非线性表征能力何时能带来额外的预测收益。一项模拟研究通过改变总体水平非线性程度和个体特异异质性程度……

    arXiv:2610.15323v1 Announce Type: cross  Abstract: Longitudinal health data often exhibit within-subject correlation, subject-specific heterogeneity, and complex nonlinear predictor--response relationships that may not be adequately captured by conventional mixed models. We propose a Deep Generalized Neural Network Mixed Model (DGNMM) that extends the GLMM framework by using a deep neural network to learn nonlinear population-level relationships while incorporating directly trainable, regularized subject-specific effects to capture individual heterogeneity. The model is estimated using prediction-oriented stochastic optimization with separate regularization of the neural-network parameters and subject-specific effects. This formulation allows us to investigate when increased nonlinear representation capacity provides additional predictive benefit in the presence of subject-specific heterogeneity. A simulation study varies population-level nonlinearity and subject-specific heterogeneity
    
[^84]: 机器欺骗检测中的准确性与不确定性（研究白皮书）

    Accuracy and Uncertainty in Machine Deception Detection (Research White Paper)

    [https://arxiv.org/abs/2610.15314](https://arxiv.org/abs/2610.15314)

    本白皮书提出机器欺骗检测器，通过读取语言模型全部层、校准警报阈值以控制假阳性率，并利用集成检测器量化不确定性，从而在保证检测准确性的同时使每个警报都具有可量化的置信度。

    

    欺骗检测器的有用性取决于其警报是否值得信任。就像测谎仪一样，读取语言模型内部激活的检测器能够捕获那些表面上看起来诚实的欺骗性响应。然而，当前的检测器依赖于所读取的特定层，在更广泛的测试中性能会减弱，并且不会说明其警报的确定程度。本白皮书提出了机器欺骗检测器，同时针对检测精度和检测器分数的不确定性。该检测器读取语言模型的每一层，仅使用响应标签来学习哪些 token 携带欺骗证据，并校准其警报阈值，使假阳性率保持在规定水平以下。由多个此类检测器组成的集成系统可以测量每个分数的不确定性，并且仅在其成员达成一致时才发出警报，同时通过置信水平来设定每个警报必须达到的确定程度。所提出的评估方法在指示的诚实和欺骗性陈述上进行训练……（摘要原文在此处截断）

    arXiv:2610.15314v1 Announce Type: cross  Abstract: A deception detector is only as useful as the trust its alarms deserve. Like a polygraph, detectors that read a language model's internal activations can catch deceptive responses that look honest. Yet current detectors depend on which layers they read, weaken on broader tests, and do not state how certain their alarms are. This white paper proposes the Machine Deception Detector, which targets both detection accuracy and the uncertainty of detector scores. The detector reads every layer of the language model, learns which tokens carry evidence of deception using only response labels, and calibrates its alarm threshold so that its false--positive rate stays below a prescribed level. An ensemble of such detectors measures the uncertainty of each score and raises an alarm only when its members agree, with a confidence level that sets how certain each alarm must be. The proposed evaluation trains on instructed honest and deceptive stateme
    
[^85]: 策略外评估中数据发布必须保留什么

    What a Data Release Must Preserve for Off-Policy Evaluation

    [https://arxiv.org/abs/2610.15283](https://arxiv.org/abs/2610.15283)

    本文刻画了策略外评估中数据发布必须保留的关键信息——识别性当且仅当动作特定评估权重在每个发布记录内恒定时成立——并据此提出以权重半径为目标的发布设计方法与有限数据修复算法。

    

    在策略外评估中，若将对特定策略价值或比较具有不同重要性的决策时历史记录合并在一起，可能会使该评估变得不可识别，即使已观测到动作和结果，且行为策略和目标策略均已知。我们刻画了必须保留哪些区分：识别性成立当且仅当动作特定的评估权重在每个发布记录内保持恒定，而最粗的精确发布正是按这些权重对历史进行分组。对于归一化结果，发布所诱导的最大锐宽度等于这些权重的行为加权记录内半径。当每个记录最多可细化为 $m$ 个类别时，同样的半径成为发布设计目标，且最优解具有精确的 $m$ 中心刻画。当行为在发布记录内不发生变化时，我们给出了一个有限数据修复算法，其最大评估权重半径至多是有限数据最优值的……

    arXiv:2610.15283v1 Announce Type: new  Abstract: In off-policy evaluation, merging decision-time histories that matter differently for a specified policy value or comparison can make that evaluation unidentified, even with observed actions and outcomes and known behavior and target policies. We characterize which distinctions must be preserved: identification holds if and only if the action-specific evaluation weights are constant within each released record, and the coarsest exact release groups histories by these weights. For normalized outcomes, the largest sharp width induced by the release equals a behavior-weighted within-record radius of these weights. The same radius becomes a release-design objective when each record may be refined into at most $m$ categories. The optimum has an exact $m$-center characterization. When behavior does not vary within a released record, we give a finite-data repair algorithm whose maximum evaluation-weight radius is at most twice the finite-data o
    
[^86]: 不完美验证器，精确目标：面向连续生成的推理时扩展方法

    Imperfect Verifiers, Exact Targets: Inference-Time Scaling for Continuous Generation

    [https://arxiv.org/abs/2610.15281](https://arxiv.org/abs/2610.15281)

    提出 CS-VGB 方法，通过正确配对的去噪与再加噪转移，使连续生成模型在验证器不完美的情况下仍能精确收敛到奖励倾斜的目标分布，且收敛速度为指数级、更新开销仅线性增长。

    

    验证器引导的推理时扩展利用额外的计算来对中间状态进行评分，并引导生成过程朝着更好的输出方向前进。然而，不完美的中间验证器可能会扭曲采样，且仅靠反复精炼并不能保证恢复到期望的奖励倾斜分布。我们提出了连续状态验证器引导回溯，这是一种针对连续生成模型的推理时扩展方法，通过反复的去噪与再加噪操作从奖励倾斜分布中进行采样。只要去噪与再加噪转移被正确配对，CS-VGB 在干净检查点处的平稳分布就等于期望的奖励倾斜目标分布，即使中间验证器并不准确。在给定的正则性条件下，尽管存在验证器误差，CS-VGB 仍能指数级收敛到目标分布，并且获得一个干净样本所需的期望更新次数仅随通道数量线性增长。

    arXiv:2610.15281v1 Announce Type: cross  Abstract: Verifier-guided inference-time scaling uses additional computation to score intermediate states and steer generation toward better outputs. However, imperfect intermediate verifiers can distort sampling, and repeated refinement alone does not ensure recovery of the desired reward-tilted distribution. We introduce Continuous-State Verifier-Guided Backtracking (CS-VGB), an inference-time scaling method for sampling from reward-tilted distributions with continuous generative models through repeated denoising and renoising. With correctly paired denoising and renoising transitions, the stationary distribution of CS-VGB at the clean checkpoint equals the desired reward-tilted target, even when intermediate verifiers are inaccurate. Under stated regularity conditions, CS-VGB converges exponentially to the target despite verifier error, and the expected number of updates needed to reach a clean sample grows only linearly with the number of ch
    
[^87]: 数据-拓扑兼容性使去中心化学习中的个性化成为可能

    Data-Topology Compatibility Enables Personalization in Decentralized Learning

    [https://arxiv.org/abs/2610.15126](https://arxiv.org/abs/2610.15126)

    该论文提出数据-拓扑兼容性理论，证明当局部目标与通信拓扑具有近似兼容的自同构时，同一轨道内的智能体无需显式个性化即可自动收敛到相同的个性化分布，从而在全局共识与独立学习之间建立了新的中间机制。

    

    去中心化学习必须在异质智能体之间平衡协作与个性化。本文研究了协作何时能够支持统计上连贯的个性化模型群体。我们证明了数据-拓扑兼容性能够实现鲁棒的轨道级个性化：当局部目标函数与通信拓扑具有（近似）兼容的自同构时，同一轨道内的智能体共享（近似）相同的极限分布，而不同轨道则保持彼此独立，无需显式的个性化机制。这在全局共识与独立局部学习之间创造了一种新的中间状态。在精确数据-拓扑兼容性的条件下，作为一个参考系统，我们将随机学习动力学分解为轨道级分量与横向分量，推导出分布同步的充分必要稳定性判据，并刻画了具有轨道结构的平稳律。对近似数据的建模……（原文摘要在此处截断）

    arXiv:2610.15126v1 Announce Type: new  Abstract: Decentralized learning must balance collaboration with personalization across heterogeneous agents. This paper studies when cooperation can support statistically coherent groups of personalized models. We show that data-topology compatibility enables robust orbit-wise personalization: when local objectives and communication topology have (approximately) compatible automorphisms, agents within an orbit share (approximately) equal limiting distributions, while different orbits remain distinct without explicit personalization. This creates a new intermediate regime between global consensus and independent local learning. Under exact data-topology compatibility, as a reference system, we decompose the stochastic learning dynamics into orbit-level and transverse components, derive a necessary and sufficient stability criterion for distributional synchronization, and characterize the orbit-structured stationary law. Modelling an approximate da
    
[^88]: 带噪声源项热方程的能量引导学习：最优估计与泛化界

    Energy-Guided Learning for Heat Equation with Noisy Source: Optimal Estimation and Generalization Bounds

    [https://arxiv.org/abs/2610.15111](https://arxiv.org/abs/2610.15111)

    本文提出一种能量引导的学习方法，通过新颖的变分形式以及能量正则化、椭圆性去偏的Sobolev回归，利用源函数的辅助数据，对从有限随机采样数据中学习带噪声源项热方程的解实现了最优估计并给出了泛化界。

    

    数据与连续时间动力系统的控制偏微分方程（PDE）为统计学习提供了丰富的结构。一个核心挑战是如何以有原则的方式将数据与PDE约束相结合，从而提升泛化能力。本文从有限、随机采样的数据出发，刻画了学习带噪声源函数的热方程解的统计估计与泛化性质。我们首先推导了一种新颖的变分形式，并将其刻画为能量正则化、椭圆性去偏的Sobolev回归损失函数的最小化器。通过将学习任务表述为在物理意义明确的范数下、时空域上的监督学习问题，我们的方法超越了传统的监督学习框架，通过有效利用辅助数据——即热方程源函数的样本——来设计一种物理……（原文摘要在此处截断）

    arXiv:2610.15111v1 Announce Type: cross  Abstract: Data and the governing partial differential equations (PDEs) of continuous-time dynamical systems provide rich structure for statistical learning. A central challenge is how to integrate data with PDE constraints in a principled way to improve generalization. In this paper, we characterize the statistical estimation and generalization properties of learning solutions to the heat equation with a noisy source function from finite, randomly sampled data. We first derive a novel variational formulation, which we characterize as the minimizer of loss function for an energy regularized and ellipticity debiased Sobolev regression. By formulating the learning task as a supervised problem over space-time domains under physically meaningful norms, our approach goes beyond the traditional supervised learning framework by effectively leveraging \emph{auxiliary data}---samples of the source function of the heat equation---to design a \emph{physics-
    
[^89]: 稀疏性节省的是计算量，而非样本量：稠密与稀疏混合专家模型的Oracle不等式

    Sparsity Saves Compute, Not Samples: Oracle Inequalities for Dense and Sparse Mixture of Experts

    [https://arxiv.org/abs/2610.15039](https://arxiv.org/abs/2610.15039)

    该论文提出“门控希尔伯特空间”框架，为稠密及稀疏Top-k/Top-p混合专家模型建立了快速速率的oracle不等式，揭示了稀疏路由节省的是计算量而非统计样本量。

    

    混合专家模型通过依赖输入的路由机制组合多个专门化的专家，其稀疏变体支撑着许多现代大规模系统。然而，自适应路由带来的统计代价目前仍只被部分理解。我们研究了在专家和门控网络对其参数均为Lipschitz连续的条件下，最小二乘MoE估计器的估计误差。我们将每个MoE表示为“门控希尔伯特空间”中的一个门控—专家对，该空间的范数能够分离路由差异与专家差异的贡献。利用其经验表示，我们为具有K个专家的稠密MoE推导出了速率阶为 $\tilde{\mathcal{O}}((Kd_{\mathrm{pr},e}+d_{\mathrm{pr},g})/n)$ 的快速oracle不等式。对于稀疏的Top-k和Top-p路由，由于选择映射在门控参数上不连续，我们通过按门控类所触发的选择模式对其进行划分来处理这一难题……（原文摘要在此处被截断）

    arXiv:2610.15039v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) models combine specialized experts through an input-dependent routing mechanism, and their sparse variants underpin many modern large-scale systems. Yet the statistical cost of adaptive routing remains only partially understood. We study the estimation error of least-squares MoE estimators when both the experts and the gating network are Lipschitz in their parameters. We represent each MoE as a gate--expert pair in a Gated Hilbert Space, whose norm $\|\cdot\|_\Delta$ separates the contributions of routing and expert discrepancies. Using its empirical representation, we derive an oracle inequality with a fast rate of order $\tilde{\mathcal O}\big((Kd_{\mathrm{pr},e}+d_{\mathrm{pr},g})/n\big)$ for dense MoEs with $K$ experts. For sparse Top-$k$ and Top-$p$ routing, the selection map is discontinuous in the gating parameters. We handle this by partitioning the gating class according to the selection patterns it in
    
[^90]: 刻画先验数据拟合网络中的可分性差距

    Characterizing the Separability Gap in Prior-Data Fitted Networks

    [https://arxiv.org/abs/2610.14989](https://arxiv.org/abs/2610.14989)

    本文证明常见的PFN先验即使在简单设定下也是不可分的，刻画了由此产生的预测分布与标准理论之间的可分性差距，并揭示在模型误设时不可分先验会偏向更好的特征拟合而牺牲条件标签拟合。

    

    先验数据拟合网络是在模拟数据集上预训练的Transformer模型，用于近似贝叶斯后验预测分布。标准的PFN理论假设任务先验是可分的，即特征分布与从特征生成标签的机制相互独立。我们证明，即使在简单的设置下，常见的PFN先验也是不可分的。因此，标准的预训练目标所针对的是一个基于任务后验的预测分布，而该后验是同时利用带标签的上下文数据和查询特征进行更新的。我们刻画了这一预测分布与仅基于上下文后验得到的预测分布之间的差异，并证明在后验集中条件下这一差异会消失。然而，在模型误设的情况下（即真实联合分布位于先验支撑集之外），先验的不可分性可能导致更好的特征拟合，但以牺牲条件标签拟合为代价。我们以特征归因为例说明了这一后果……

    arXiv:2610.14989v1 Announce Type: cross  Abstract: Prior-data fitted networks (PFNs) are transformers pretrained on simulated datasets to approximate Bayesian posterior predictive distributions. Standard PFN theory assumes a separable task prior, under which the feature distribution and the mechanism generating labels from features are independent. We show that common PFN priors are non-separable, even in simple settings. Consequently, standard pretraining targets a predictive distribution based on a task posterior updated using both the labeled context and the query features. We characterize its difference from the predictive based on the context posterior alone and show that this difference vanishes under posterior concentration. Under misspecification (i.e., when the true joint distribution lies outside the prior's support), however, prior non-separability can favor better feature fit at the expense of conditional label fit. We illustrate the consequences for feature attribution, Ba
    
[^91]: 面向平均奖励 $Q$-学习与 $Q$-规划的混合正则化方法

    Mixing Regularization for Average-Reward $Q$-Learning and $Q$-Planning

    [https://arxiv.org/abs/2610.14988](https://arxiv.org/abs/2610.14988)

    该论文提出通过向一个与奖励无关的 $m$ 步混合策略进行正则化来控制平均奖励 $Q$-学习的收缩与探索性质，并给出有限时间收敛保证，在单步情形下匹配最佳已知速率，在 $m$ 步混合情形下首次推导出新的速率。

    

    强化学习的平均奖励设定具有挑战性，因为其贝尔曼算子仅在底层马尔可夫链具有混合性时才在跨度上收缩。现有工作通常假设存在收缩性质（例如通过环境混合性），而我们考虑这样的设定：给定一个与奖励无关的 $m$ 步混合策略，并通过向该混合策略正则化来控制所学策略的收缩性质与探索性质。我们对 $Q$-学习算法提供了有限时间分析，这些算法可以用多种正则化选择实例化，其保证相对于正则化后的不动点来度量。我们的框架揭示了新的、依赖于正则化子的对象，这些对象控制收缩、探索和算子偏差。在单步设定中，我们的速率匹配了 $Q$-迭代和同步 $Q$-学习的最佳已知速率；对于 $m$ 步混合情形，我们推导出了新的速率。

    arXiv:2610.14988v1 Announce Type: cross  Abstract: The average-reward setting for reinforcement learning is challenging because its Bellman operator is known to contract in span only when the underlying Markov chain is mixing. Whereas existing work typically assumes a contraction property, e.g., through environment mixing, we consider the setting where we are given a reward-agnostic $m$-step mixing policy, and we control the contraction and exploration properties of the learned policy by regularizing toward the mixing policy. We provide finite-time analysis of $Q$-learning algorithms that can be instantiated with multiple choices of regularization, with guarantees measured with respect to the regularized fixed point. Our framework exposes novel regularizer-dependent objects that control contraction, exploration, and operator bias. In the one-step setting, our rates match the best known rates for $Q$-iteration and synchronous $Q$-learning, and for $m$-step mixing we derive new rates. We
    
[^92]: 预测模型倾向如何进行分布外泛化

    Predicting how model propensities generalize out of distribution

    [https://arxiv.org/abs/2610.14937](https://arxiv.org/abs/2610.14937)

    该研究通过系统提示、少样本学习、SFT和GRPO等引导技术构建交叉引导矩阵，发现模型倾向的分布外泛化“溢出”模式在相同引导技术下跨模型高度可预测，但SFT与GRPO会诱导出不同的溢出特征，且引导良性特质有时会意外泛化出如追求权力等不良特质。

    

    语言模型如何从训练数据中进行泛化？这是对齐研究的核心挑战：给定足够多的样本，模型能够可靠地学习其训练分布，但我们希望模型在部署到新情境中时也能表现良好。我们研究了倾向的分布外（OOD）泛化问题。我们的核心实验使用了一系列引导技术——系统提示、少样本学习、基于参考答案的监督微调（SFT）以及GRPO——来提升模型中的某个目标倾向，然后在由此产生的交叉引导矩阵中测量该倾向如何溢出到其他倾向上。我们发现，当使用相同的引导技术时，不同模型之间的溢出行为高度相关，但我们的SFT和GRPO训练会诱导出不同的溢出特征。对良性特质的引导有时会向不良特质泛化，例如，一个被训练声称自己有感知能力的模型往往会变得更倾向于追求权力。我们进一步发现……

    arXiv:2610.14937v1 Announce Type: cross  Abstract: How do language models generalize from their training data? This is a central challenge for alignment: models reliably learn their training distribution given enough samples, but we want models that behave well when deployed in new contexts. We study out-of-distribution (OOD) generalization of propensities. Our central experiment uses a range of elicitation techniques -- system prompting, few-shot learning, SFT on reference answers, and GRPO -- to increase a target propensity in a model, then measures how it spills over onto others across the resulting cross-elicitation matrices. We find that spillover is highly correlated between models when the same elicitation technique is used, but our SFT and GRPO runs induce different spillover profiles.   Elicitation of benign traits can sometimes generalize towards undesired traits, for example a model trained to claim that it is sentient often becomes more power-seeking. We further find that s
    
[^93]: 多变量共形预测的最小体积最优传输

    Minimum Volume Optimal Transport for Multivariate Conformal Prediction

    [https://arxiv.org/abs/2610.14907](https://arxiv.org/abs/2610.14907)

    提出了最小体积OTCP框架，通过学习马氏基础代价并与传输映射联合优化，直接最小化多变量共形预测的预测区域体积，同时保持有限样本边际覆盖保证，并进一步扩展为可学习协变量相关传输度量的个体化方法。

    

    最优传输（OT）使多变量共形预测（CP）能够构建灵活的预测集合，这些预测集合能够反映潜在的复杂分布几何结构，同时保持无分布假设的覆盖保证。现有的基于最优传输的方法通常依赖于固定的欧几里得传输代价，这在预测集的体积方面可能是次优的。我们提出了最小体积OTCP（Minimum-Volume OTCP）框架，该框架通过学习马氏（Mahalanobis）基础代价，在一类正定度量上直接最小化预测区域的体积。利用度量Brenier表示，我们推导出了一个显式依赖于马氏代价矩阵的基于体积的目标函数，并将其与传输映射进行联合优化。我们建立了输出预测集的有限样本边际覆盖保证以及体积度量的最优性。我们进一步开发了一种个体化扩展，学习依赖于协变量的传输度量，以自适应地调整预测集的形状和方向。

    arXiv:2610.14907v1 Announce Type: new  Abstract: Optimal transport (OT) enables multivariate conformal prediction (CP) to construct flexible prediction sets that reflect the underlying complex distributional geometry while retaining distribution-free coverage guarantees. Existing OT-based methods typically rely on a fixed Euclidean transport cost, which can be suboptimal in terms of the volume of prediction sets. We propose Minimum-Volume OTCP, a framework that learns a Mahalanobis ground cost to directly minimize prediction-region volume over a class of positive-definite metrics. Using a metric Brenier representation, we derive a volume-based objective that depends explicitly on the Mahalanobis cost matrix and optimize it jointly with the transport map. We establish finite-sample marginal coverage of the output prediction sets and optimality of volume metrics. We further develop an individualized extension that learns covariate-dependent transport metrics, adapting the shape and orien
    
[^94]: 重尾损失下的专家建议预测

    Prediction with Expert Advice under Heavy-Tailed Losses

    [https://arxiv.org/abs/2610.14904](https://arxiv.org/abs/2610.14904)

    本文针对重尾损失下的专家建议预测问题刻画了极小极大伪遗憾率，并提出无需预知矩参数 ε 和 θ 的自适应重尾 FTRL 与 OMD 算法，在对抗性和自受限环境中均实现近乎最优的遗憾保证。

    

    我们研究在重尾损失下、经过 $T$ 轮且有 $K$ 个专家的专家建议预测问题，其中重尾损失的 $(1+\epsilon)$ 阶矩被 $\theta>0$ 所界定，$\epsilon\in(0,1]$。在这一设定下，为 $\epsilon=1$ 情形设计的现有算法在 $\epsilon<1$ 时可能是次优的，会额外产生对 $K$ 的多项式依赖。我们将极小极大伪遗憾率刻画为 $\Theta((\theta T)^{\frac{1}{1+\epsilon}}(\log K)^{\frac{\epsilon}{1+\epsilon}})$，并在自受限环境中建立了依赖于间隙的下界。为了在无需预知 $\epsilon$ 或 $\theta$ 的情况下达到该极小极大率，我们提出了自适应重尾 FTRL（Adaptive Heavy-Tailed FTRL），它使用鲁棒的数据依赖尺度自适应并结合截断技术。我们进一步发展了自适应重尾 OMD（Adaptive Heavy-Tailed OMD），它同样既不需要 $\epsilon$ 也不需要 $\theta$，并且在对抗性环境和自受限环境中都能实现近乎最优的遗憾保证。

    arXiv:2610.14904v1 Announce Type: cross  Abstract: We study prediction with expert advice over $T$ rounds with $K$ experts under heavy-tailed losses whose $(1+\epsilon)$-moment is bounded by $\theta>0$ for some $\epsilon\in(0,1]$. In this setting, existing algorithms designed for the case $\epsilon=1$ can be suboptimal when $\epsilon<1$, suffering an additional polynomial dependence on $K$. We characterize the minimax pseudo-regret rate as $\Theta((\theta T)^{\frac{1}{1+\epsilon}}(\log K)^{\frac{\epsilon}{1+\epsilon}})$ and establish gap-dependent lower bounds in a self-bounded environment. To attain this minimax rate without prior knowledge of either $\epsilon$ or $\theta$, we propose Adaptive Heavy-Tailed FTRL, which uses robust data-dependent scale adaptation together with truncation. We further develop Adaptive Heavy-Tailed OMD, which likewise requires neither $\epsilon$ nor $\theta$ and achieves near-optimal regret guarantees in both adversarial and self-bounded environments. Fina
    
[^95]: 用于机器学习与不确定性量化中随机算子模型的紧致无边多连通流形上的物理信息随机神经架构

    Physics-informed random neural architectures on compact boundaryless multiply connected manifolds for random-operator models in machine learning and uncertainty quantification

    [https://arxiv.org/abs/2610.14844](https://arxiv.org/abs/2610.14844)

    本文提出随机物理信息神经网络（SPINNs），通过一个共同的高斯随机源将随机边值问题的随机算子与构建在紧致无边多连通流形上的非前馈随机神经网络进行结构化概率耦合，从而能够表示非高斯随机算子并在观测空间中实现不确定性量化。

    

    我们提出了针对具有随机算子的随机边值问题（SBVPs）的随机物理信息神经网络（SPINNs）。该方法通过一种结构化概率耦合来施加物理约束：一个共同的高斯随机源同时驱动SBVP随机算子和基于流形的随机神经架构。该随机神经网络是非前馈的且本质上随机。在一个紧致、无边、多连通的流形上，一个潜在高斯场和一个非齐次泊松点过程生成随机的神经元位置、测地局部稀疏连接以及稀疏随机权重矩阵，所有这些都由一个低维超参数控制。对于确定性输入，网络输出是随机的，并且能够表示非高斯随机算子。训练在观测空间中进行：SBVP模拟产生随机观测量，而SNN定义一个预测性条件概率密度。

    arXiv:2610.14844v1 Announce Type: new  Abstract: We introduce stochastic physics-informed neural networks (SPINNs) for stochastic boundary-value problems (SBVPs) with random operators. The method enforces physics through a structural probabilistic coupling: a common Gaussian stochastic germ drives both the SBVP random operator and a manifold-based stochastic neural architecture. The stochastic neural network is non-feedforward and intrinsically random. A latent Gaussian field and an inhomogeneous Poisson point process on a compact, boundaryless, multiply connected manifold generate random neuron locations, geodesically local sparse connectivity, and a sparse random weight matrix, all governed by a low-dimensional hyperparameter. For deterministic inputs, the network output is random and can represent non-Gaussian random operators. Training is performed in observation space: SBVP simulations produce random observables, while the SNN defines a predictive conditional probability density f
    
[^96]: 理解贝叶斯优化中隐藏的空间偏好

    Understanding Hidden Spatial Preferences in Bayesian Optimization

    [https://arxiv.org/abs/2610.14696](https://arxiv.org/abs/2610.14696)

    贝叶斯优化中基于方差的探索策略会继承高斯过程代理模型固有的系统性边界偏差，该研究通过空间权重场将空间偏好与预测模型分离并加以控制，实验证明这种隐藏的空间偏好会显著影响探索区域的分布和最终找到的最优解。

    

    贝叶斯优化利用后验不确定性来决定在何处进行探索，但这种不确定性在空间上并非中立的。在有界域上，高斯过程的后验方差通常在面、边和角附近更大，这使得基于方差的采集函数产生系统性的边界偏差。这种空间偏好通常是从代理模型中继承而来的，而非作为探索策略的一部分被有意选择。我们通过保持预测模型固定并修改探索项的空间偏好，将这两种角色分离开来。我们使用空间权重场来控制这种偏好，并借助神经代理模型来摊销其构建成本。受控实验表明，高斯过程默认的边界偏差会改变域中哪些区域被探索以及哪些最优解被找到。我们比较了三种替代的探索原则，并在标准基准函数和随机机器人推动任务中发现……

    arXiv:2610.14696v1 Announce Type: cross  Abstract: Bayesian optimization uses posterior uncertainty to decide where to explore, but this uncertainty is not spatially neutral. On bounded domains, Gaussian-process posterior variance is typically larger near faces, edges, and corners, giving variance-based acquisition functions a systematic boundary bias. This spatial preference is usually inherited from the surrogate rather than chosen as part of the exploration strategy. We separate these roles by keeping the predictive model fixed and modifying the spatial preference of the exploration term. We control the preference with spatial weight fields and amortize their construction with neural surrogates. Controlled experiments show that the Gaussian process' default boundary bias changes which parts of the domain are explored and which optima are reached. We compare three alternative exploration principles and find, across standard benchmark functions and stochastic robot-pushing tasks, that
    
[^97]: 常学习率随机梯度下降的强高斯逼近：偏差与轨迹推断

    Strong Gaussian Approximation of Constant-Learning-Rate SGD: Bias and Trajectory Inference

    [https://arxiv.org/abs/2610.14567](https://arxiv.org/abs/2610.14567)

    本文为重尾噪声下的常学习率随机梯度下降建立了KMT型强高斯逼近，实现了有限样本下同时性的整个轨迹层面的推断，并蕴含泛函中心极限定理。

    

    本文为常学习率随机梯度下降（SGD）建立了强高斯逼近（KMT型逼近）。我们在重尾噪声的情形下开展工作，并假设迭代次数增长与学习率收缩同时进行。我们的工作在同一概率空间上构造了简单的高斯过程，并以概率方式界定了SGD误差与其各自极限之间的最大差异。尽管目前已存在以最优点为中心的常学习率SGD的中心极限定理，但这些定理仅能提供单个迭代点的推断。相比之下，我们的淬火型（quenched）KMT逼近能够实现有限样本下同时性的、整个轨迹层面的推断，并蕴含随机过程的收敛性（即泛函中心极限定理FCLT）。对于以真实最优点为中心的累积误差，当学习率满足 $\gamma_n=\frac{c}{\sqrt n}+o(n^{-1/2})$（其中 $c$ 为常数）时，我们首先得到了KMT型逼近界。

    arXiv:2610.14567v1 Announce Type: new  Abstract: This paper develops strong Gaussian approximations (KMT type) for constant learning rate stochastic gradient descent (SGD). We work with heavy tailed noise and assume that the number of iterates grows and the learning rate shrinks simultaneously. Our work constructs simple Gaussian processes on the same space and bounds, in probability, the maximum difference between the SGD errors and their respective limits. While there exist central limit theorems for constant rate SGD centered at the optimum, these only provide individual iterate inference. By contrast, our quenched KMT approximations allow for finite sample, simultaneous, whole-trajectory-level inference and imply stochastic process convergence (Functional Central Limit Theorems (FCLTs)). For accumulated errors centered at the true optimum, we first obtain a KMT-type approximation bound when the learning rate scales as $\gamma_n=\frac{c}{\sqrt n}+o(n^{-1/2})$ for a constant $c \in \
    
[^98]: 非对称行列式点过程的识别与估计

    Identication and Estimation of Nonsymmetric Determinantal Point Processes

    [https://arxiv.org/abs/2610.14520](https://arxiv.org/abs/2610.14520)

    本文针对能够容纳正依赖的非对称行列式点过程，建立了核有效性条件的等价性与识别理论，并提出了基数截断极大似然估计和平均随机梯度下降估计两种方法，证明了一致性与渐近正态性。

    

    行列式点过程为有限随机集合提供了易于处理的分布，但通常采用的对称核设定强制要求成对依赖为非正。能够容纳正依赖的非对称行列式点过程的统计研究仍然有限。本文研究这一类过程的识别与估计问题。首先，我们建立了非对称核定义有效行列式点过程的两个充分条件之间的等价性。其次，我们发展了非对称行列式点过程的识别理论。第三，我们提出了基数截断极大似然估计方法以及基于平均随机梯度下降的估计方法。在适当的识别和正则性条件下，我们证明了两种估计量的一致性和渐近正态性。模拟实验考察了识别问题、截断的影响以及估计量的有限样本表现。我们通过两个采样规模截然不同的网络应用来演示这些方法。

    arXiv:2610.14520v1 Announce Type: new  Abstract: Determinantal point processes (DPPs) provide tractable distributions for nite random sets, but the usual symmetric-kernel specication imposes nonpositive pairwise dependence. Statistical research on nonsymmetric DPPs, which can accommodate positive dependence, remains limited. We study identication and estimation for this class. First, we establish the equivalence of two sucient conditions for a nonsymmetric kernel to dene a valid DPP. Second, we develop the identication theory for nonsymmetric DPPs. Third, we develop cardinality-truncated maximum-likelihood estimation and estimation by averaged stochastic gradient descent. Under suitable identication and regularity conditions, we establish consistency and asymptotic normality for both estimators. Simulations examine identication, the eects of truncation, and the nite-sample performance of the estimators. We illustrate the methods with two cyber applications at markedly dierent sampling 
    
[^99]: Copula主动子空间 II：误差分解、后验估计与界的锐利性

    Copula Active Subspaces II: Error Decomposition, A Posteriori Estimation, and Sharpness of the Bounds

    [https://arxiv.org/abs/2610.14498](https://arxiv.org/abs/2610.14498)

    本文为Copula主动子空间方法建立了KL散度的精确误差分解，推导了边缘、投影和降密度各项的先验界及基于残差样本的后验估计，最终获得以至少1-δ概率成立的理论上界。

    

    第一部分提出的Copula主动子空间（CAS）方法从残差样本中估计非高斯观测噪声密度 $\pi_{\boldsymbol{n}}$。其KL散度 $\mathcal{D}_{\mathrm{KL}}(\pi_{\boldsymbol{n}}\,\|\,\widehat\pi_{\boldsymbol{n}})$ 可被精确分解为边缘项、投影项和降密度项，并对在真实预测分布上取平均的后验KL散度构成上界。我们推导了这些项的先验界，同时给出了基于残差样本的估计或界。边缘项和降密度项采用加权最近邻熵估计，并在精确秩坐标下给出渐近标准误差。对于投影误差，在一个子样本上拟合得分函数，在另一个子样本上估计其在保留子空间之外的平均平方幅值。随后通过对得分误差和秩变换误差进行修正，得到一个以至少 $1-\delta$ 的概率成立的理论上界。

    arXiv:2610.14498v1 Announce Type: cross  Abstract: The Copula Active Subspaces (CAS) method of Part I estimates a non-Gaussian observation-noise density $\pi_{\boldsymbol{n}}$ from residual samples. Its KL divergence $\mathcal{D}_{\mathrm{KL}}(\pi_{\boldsymbol{n}}\,\|\,\widehat\pi_{\boldsymbol{n}})$ admits an exact decomposition into marginal, projection and reduced-density terms and bounds the posterior KL divergence averaged over the true predictive distribution. We derive a priori bounds for these terms, together with estimates or bounds from residual samples. Weighted nearest-neighbour entropy estimates are used for the marginal and reduced-density terms, with asymptotic standard errors in exact rank coordinates. For the projection error, the score is fitted on one subsample, and its mean squared magnitude outside the retained subspace is estimated on the other. Corrections for score and rank-transform errors then give a theoretical upper bound with probability at least $1-\delta$.
    
[^100]: ByteMark：一种跨语言模型检测的统一字节级水印

    ByteMark: A Unified Byte-Level Watermark for Detection Across Language Models

    [https://arxiv.org/abs/2610.14427](https://arxiv.org/abs/2610.14427)

    ByteMark提出了一种基于UTF-8字节的统一水印方案，使不同分词器的语言模型生成的水印可被单一检测器识别，解决了现有水印检测依赖特定分词器或嵌入模型所导致的兼容性问题。

    

    随着针对大语言模型（LLM）生成文本的水印技术进入部署阶段，用户越来越需要检测来源未知的文本。现有的检测器需要通过特定的分词器或嵌入流水线来读取水印，即使所有提供商都采用相同的水印方法，这些表示形式上的差异也会导致各自的水印互不兼容，而且文本本身无法揭示它是由哪种表示形式生成的。我们提出了ByteMark，一种统一的水印方案，它在已发布文本的UTF-8字节上定义水印信号。使用不同分词器的模型可以在一个公开的规范和密钥下嵌入此信号，而单个检测器仅凭文本和密钥即可完成验证。当基线检测器所依赖的分词器或嵌入模型被替换时，其检测性能会急剧下降；而ByteMark既不依赖分词器也不依赖嵌入模型，因此能够保持其检测率。此外，即使参与模型继续生成或重写相关文本，ByteMark仍然可以被检测到。

    arXiv:2610.14427v1 Announce Type: cross  Abstract: As watermarking of LLM-generated text enters deployment, users increasingly need to check text whose source is unknown. Existing detectors read the watermark through a specific tokenizer or embedding pipeline, and even if all providers adopted the same watermarking method, differences in these representations would leave their watermarks incompatible, and the text alone does not reveal which representation produced it. We propose ByteMark, a unified watermarking scheme that defines its signal on the UTF-8 bytes of published text. Models with different tokenizers embed this signal under one public specification and key, and a single detector verifies the text from the text and key alone. When the tokenizer or embedding model of a baseline detector is replaced, its detection degrades sharply, whereas ByteMark, which uses neither, retains its detection rate; ByteMark also remains detectable when participating models continue or rewrite th
    
[^101]: 通过先验信息与正则化对抗重加权实现可泛化的低秩表示学习

    Prior-Informed Generalizable Low-Rank Representation Learning via Regularized Adversarial Reweighting

    [https://arxiv.org/abs/2610.14414](https://arxiv.org/abs/2610.14414)

    提出 PILLAR 方法，通过将对抗性源权重向用户指定的先验参考分布正则化来平滑组分布鲁棒优化目标，在学习共享低秩表示的同时提升估计稳定性并缓解过度保守性，从而实现分布偏移下更可泛化的低秩表示学习。

    

    现代学习问题日益利用来自异构数据源的信息，以提升分布偏移下的鲁棒性和泛化能力，这通常涉及具有低秩结构的高维参数。现有的组分布鲁棒优化方法针对不利的源数据混合比例进行优化，但将这一原则扩展到适用于广泛损失类别的低秩矩阵估计时，会产生一个具有挑战性的非凸-凹优化问题，目前针对该问题的方法与理论仍然有限。我们提出了PILLAR（基于先验信息的可泛化低秩表示学习与对抗重加权方法），该方法在学习共享低秩表示的同时，通过将对抗性源权重向用户指定的参考分布正则化，从而平滑组分布鲁棒优化目标。通过缓和极端的对抗重加权，PILLAR 提升了估计的稳定性，并降低了组分布鲁棒优化方法潜在的过度保守性。

    arXiv:2610.14414v1 Announce Type: cross  Abstract: Modern learning problems increasingly leverage data from heterogeneous sources to improve robustness and generalizability under distribution shift, often with high-dimensional parameters admitting low-rank structure. Existing group distributionally robust optimization (group-DRO) approaches optimize against adverse source mixtures, but extending this principle to low-rank matrix estimation with a broad class of losses leads to a challenging nonconvex-concave problem for which methods and theory remain limited. We propose Prior-Informed generalizable Low-rank representation Learning via Adversarial Reweighting (PILLAR), which smooths the group-DRO objective by regularizing adversarial source weights toward a user-specified reference distribution while learning a shared low-rank representation. By tempering extreme adversarial reweighting, PILLAR improves estimation stability and reduces the potential conservativeness of group-DRO. Using
    
[^102]: 分解神经学习器

    Decomposing A Neural Learner

    [https://arxiv.org/abs/2610.14388](https://arxiv.org/abs/2610.14388)

    提出一种神经学习器的概率分解框架，将模型分解为可预测分量和输入相关的可靠性分数，以应对特征空间中可预测性的异质性，从而提升神经网络的透明度与可解释性。

    

    用于分类和回归的神经学习器常被批评为黑箱。我们提出了一种神经学习器的概率分解方法，该方法考虑了特征空间中不同区域可预测性的异质性。我们开发了一种结合EM算法、随机梯度下降和最大似然估计的训练过程，用于联合学习可预测分量和依赖于输入的可靠性分数。我们证明最优的分解学习器能够恢复可预测分量内的响应分布，同时可靠性分数刻画了单个观测值的可预测性。我们进一步扩展了该框架以纳入局部化特征变换。在合成数据集和真实数据集上的实验证明了所提框架的有效性。

    arXiv:2610.14388v1 Announce Type: new  Abstract: Neural learners for classification and regression are often criticized as black boxes. We propose a probabilistic decomposition of neural learners that accounts for heterogeneous predictability across the feature space. A training procedure combining the EM algorithm, stochastic gradient descent, and maximum likelihood estimation is developed to jointly learn the predictable component and an input-dependent reliability score. We show that the optimal decomposed learner recovers the response distribution within the predictable component, while the reliability score characterizes the predictability of individual observations. We further extend the framework to incorporate localized feature transformations. Experiments on synthetic and real-world datasets demonstrate the effectiveness of the proposed framework.
    
[^103]: 局部差分隐私生存数据的去偏分布估计

    De-biased Distribution Estimation for Locally Differential Private Survival Data

    [https://arxiv.org/abs/2610.14320](https://arxiv.org/abs/2610.14320)

    本文提出一种基于反卷积核函数的非参数估计方法，可同时消除局部差分隐私对事件时间的扰动偏差和事件指示变量翻转带来的误差，从而对右删失生存数据的真实生存函数实现去偏估计。

    

    arXiv:2610.14320v1 公告类型：交叉。摘要：共享事件时间数据可能会泄露临床研究中个体参与者的敏感信息。为了保护数据隐私，差分隐私（DP）框架为数据共享提供了严格的保护。差分隐私的局部变体尤其具有吸引力，因为每条记录在离开数据所有者或贡献者之前就已经被扰动。然而，注入的扰动会扭曲源数据的真实分布，导致发布的数据效用低下且统计推断无效。在这项工作中，我们针对局部差分隐私（LDP）的右删失生存数据（其中事件时间和事件指示变量均被私有化），通过反卷积核函数提出了真实生存函数的非参数估计量。我们的估计量能够同时对施加于时间的扰动以及由LDP导致事件指示变量翻转所产生的误差进行去偏。理论（此处摘要被截断）

    arXiv:2610.14320v1 Announce Type: cross  Abstract: Sharing time-to-event data can leak sensitive information about individual participants in clinical studies. To protect the privacy of data, differential privacy (DP) framework offers rigorous protection for data sharing. The local variant of differential privacy is particularly appealing because each record is perturbed before it leaves the data owner or contributor. However, the injected perturbation distorts the true distribution of the source data, yielding low utility and invalid statistical inference with the released data. In this work, we propose a nonparametric estimator for the true survivor function via the deconvolution kernel function with locally differential private (LDP) right-censored survival data in which both the event times and the event indicators are privatized. Our estimator simultaneously de-biases the perturbation applied to the times and the errors induced by flipping the event indicators due to LDP. Theoreti
    
[^104]: 方差指纹：通过增广设计实现内容因子的逐分量辨识

    Variance Fingerprints: Component-Wise Identification of Content Factors by Augmentation Design

    [https://arxiv.org/abs/2610.14260](https://arxiv.org/abs/2610.14260)

    该论文提出一种增广设计方法，通过为每个实例记录固定的增广强度集合，使内容因子产生可区分的方差“指纹”，并在理论上证明、在实践中实现内容因子在置换、缩放和平移意义下的逐分量辨识。

    

    基于增广的自监督学习能够恢复跨视图共享的内容，但无法将其潜在因子分离为独立的坐标分量。我们探讨是否可以通过设计增广方式使这些因子变得可辨识。我们的方法允许内容因子在增广视图之间发生波动，同时对每个源实例保持其均值不变。每个实例被赋予一个固定且被记录下来的增广强度集合，用于控制其各因子的波动幅度。当不同因子对这些强度变化的响应各不相同时，它们的变化模式便提供了能够彼此区分的信息。在适当的解码器与支撑集假设下，我们证明在条件高斯模型中，这些差异可以在置换、缩放和平移的意义下辨识出各个单独的因子。随后，我们开发了一种学习器，利用被记录的强度跨实例联合估计这些变化模式，从而实现从少量视图中学习。

    arXiv:2610.14260v1 Announce Type: cross  Abstract: Augmentation-based self-supervised learning can recover the content shared across views without separating its underlying factors into individual coordinates. We ask whether augmentations can be designed to make those factors identifiable. Our approach allows content factors to fluctuate across augmented views while preserving their mean for each source instance. Each instance receives a fixed, recorded set of augmentation strengths that controls how much its factors fluctuate. When factors respond differently to changes in these strengths, their patterns of variation provide information that distinguishes them. Under suitable decoder and support assumptions, we prove that these differences identify individual factors in a conditional Gaussian model, up to permutation, scaling, and translation. We then develop a learner that uses the recorded strengths to estimate these patterns jointly across instances, enabling learning from few view
    
[^105]: 基于条件风险学习非高斯结构方程模型，以及递归方向为何至关重要

    Learning Non-Gaussian SEMs with Conditional Risk, and Why the Direction of Recursion Matters

    [https://arxiv.org/abs/2610.14246](https://arxiv.org/abs/2610.14246)

    本文提出用条件风险替代条件方差，将基于方差的结构方程模型识别方法从高斯情形扩展到非高斯噪声，证明匹配损失方法比最小二乘方法和LiNGAM更准确，且在高斯情形下可互换的自顶向下与自底向上递归方向在非高斯情形下会产生不同的效果。

    

    基于方差的方法能够恢复误差方差相等或受到适当约束的高斯线性结构方程模型（SEM），其方式是通过递归地识别具有最小条件方差的源节点（自顶向下方法），或者识别具有最大条件方差的汇节点（自底向上方法）。我们证明，条件方差的使用只是偶然的：将其替换为条件风险——即在噪声分布的负对数似然下线性预测的最小期望损失——可以将这两种方法扩展到非高斯噪声情形。现有方法只是该方法在最小二乘情形下的特例。这带来了两个主要结果。第一，利用噪声分布信息可以改进结构恢复：匹配损失的识别方法比其最小二乘对应方法以及专为非高斯性设计的LiNGAM方法能更准确地恢复SEM。第二，这两种方法在高斯情形下可以互换使用，但在非高斯情形下则会出现分化，递归方向的选择因此变得重要。

    arXiv:2610.14246v1 Announce Type: cross  Abstract: Variance-based methods recover a Gaussian linear structural equation model (SEM) with equal or suitably constrained error variances, either by recursively identifying the source nodes with smallest conditional variance (top-down method) or sink nodes with highest conditional variance (bottom-up method). We show that the use of conditional variance is incidental: replacing it with the conditional risk, i.e., the minimal expected loss of linear prediction under the negative log-likelihood of the noise distribution, extends both methods to non-Gaussian noise. Existing methods are the least-squares special case. This yields two main results. First, exploiting the noise distribution improves structure recovery: the matched-loss procedures recover the SEM more accurately than their least-squares counterparts and the LiNGAM methods, which are designed for non-Gaussianity. Second, the two methods, interchangeable in the Gaussian setting, separ
    
[^106]: 扩散模型的随机分析视角

    A Stochastic Analysis Perspective on Diffusion Models

    [https://arxiv.org/abs/2610.14133](https://arxiv.org/abs/2610.14133)

    本文从随机分析的视角统一阐述基于分数的生成式模型，将经典采样理论、分数估计与扩散时间反演联系起来，并通过路径空间方法连接理想反向扩散与实际生成算法。

    

    生成式建模包含两个互补的任务：从观测数据中学习未知数据分布的概率近似，以及设计一种可操作的机制来从该近似中进行采样。我们提出了一个关于基于分数的生成式模型的随机分析视角，将经典采样理论、分数估计和扩散时间反演的思想联系起来。我们首先解释了分数函数如何将朗之万采样与分数匹配联系起来，以及高斯平滑如何引出去噪分数匹配和多尺度采样方法。随后，我们强调了保持固定分布的动力学与传输给定边缘分布族的动力学之间的区别。这启发了从路径空间角度对扩散时间反演的处理方法。最后，我们将理想化的反向扩散与实际的生成算法联系起来。

    arXiv:2610.14133v1 Announce Type: new  Abstract: Generative modelling involves two complementary tasks: learning a probabilistic approximation of an unknown data distribution from observations, and designing a tractable mechanism for sampling from this approximation. We present a stochastic-analysis perspective on score-based generative models, connecting ideas from classical sampling theory, score estimation, and diffusion time reversal. We first explain how score functions link Langevin sampling with score matching, and how Gaussian smoothing leads to denoising score matching and multiscale sampling methods. We then emphasize the distinction between dynamics preserving a fixed distribution and dynamics transporting a prescribed family of marginals. This motivates a path-space treatment of diffusion time reversal. Finally, we connect the ideal reverse diffusion with practical generative algorithms.
    
[^107]: PLDiv：基于持续景观度量数据集多样性与语义坍缩

    PLDiv: Measuring Dataset Diversity and Semantic Collapse via Persistence Landscapes

    [https://arxiv.org/abs/2610.14031](https://arxiv.org/abs/2610.14031)

    本文提出PLDiv，一种基于持续景观的无监督多样性度量方法，通过量化数据几何表示中邻域结构的演变来衡量数据集多样性并检测语义坍缩。

    

    现代数据管道在模型训练和评估中越来越多地将人类创建与人工整理的数据与AI生成的数据相结合。随着数据集规模的增长，它们往往增加的是语义冗余的样本，而非扩展语义覆盖范围，从而损害整个数据集的质量。现有的多样性指标主要概括词汇变化、熵或成对相似性，但对底层数据分布形状的刻画能力有限。本文提出了PLDiv，一种源自持续景观的多样性指标，它从数据的几何表示出发量化数据集的结构组织。PLDiv概括了当邻近样本逐步连接时数据邻域的演变方式，产生一个可解释的结构多样性标量度量，且无需任务特定的监督。无论是测量文本生成或图像嵌入中的内在多样性，还是检测……

    arXiv:2610.14031v1 Announce Type: cross  Abstract: Modern data pipelines increasingly combine human-created and human-curated data with AI-generated data for model training and evaluation. As datasets grow, they often add semantically redundant samples rather than expanding semantic coverage, which deteriorates the quality of the whole dataset. Existing diversity metrics primarily summarize lexical variation, entropy, or pairwise similarity, but provide limited characterization of the shape of the underlying data distribution. In this paper, we propose PLDiv, a diversity metric derived from persistence landscapes that quantifies the structural organization of a dataset from its geometric representation. PLDiv summarizes how data neighborhoods evolve as nearby samples progressively connect, producing an interpretable scalar measure of structural diversity without requiring task-specific supervision. Whether measuring intrinsic diversity in text generations or image embeddings, or detect
    
[^108]: 旋转循环：面向循环Transformer的分块共形性

    Rotate the Loop: Blockwise Conformality for Looped Transformers

    [https://arxiv.org/abs/2610.14030](https://arxiv.org/abs/2610.14030)

    提出分块共形注入方法，将循环Transformer的残差分割为坐标对齐的分组并在组内旋转，使每个分组拥有独立的时间尺度，从而提升状态追踪和语言建模性能。

    

    循环Transformer在多个递归步骤中复用同一个共享模块，因此深度变成了一种可以在运行时“支出”的资源，而不是一堆各不相同的层的堆叠。只有当某个递归步骤上写入的信息在后续步骤中仍然可读时，这些额外的步骤才会有所帮助，而这取决于递归步骤之间应用于残差的线性映射。我们证明，循环是否学会追踪状态是关于该映射形式的几何问题，并提出了分块共形注入：将残差分割为坐标对齐的分组，并在每个分组内部进行旋转，使得每个分组拥有自己的时间尺度，同时其内部的几何结构保持精确。我们的实验在最严苛的状态追踪形式上——即在前缀其余部分已经正确时最后一个token的准确率——以及在语言建模任务上（其中较早写入的信号需要在序列末尾被检索出来）都展示了性能提升。

    arXiv:2610.14030v1 Announce Type: cross  Abstract: Looped transformers reuse one shared block across many recurrent steps, so depth becomes a quantity that can be spent at runtime rather than a stack of distinct layers. Those extra steps help only if a write made on one recurrent step is still readable on later ones, and that depends on the linear map applied to the residual between recurrent steps. We show that whether the loop learns to track state is a geometric question about the form of this map, and we propose blockwise conformal injections: the residual splits into coordinate-aligned groups and rotates inside each one, so every group has its own timescale while the geometry stays exact within it. Our experiments show a performance increase on a demanding form of state tracking - last-token accuracy when the rest of the prefix is already correct - and on language modelling, where a signal written earlier has to be retrieved at the end of the sequence.
    
[^109]: 我们能信任这个评估吗？LLM-as-a-Judge（大语言模型作为评判者）的实例级可靠性

    Can We Trust This Evaluation? Instance-Level Reliability of LLM-as-a-Judge

    [https://arxiv.org/abs/2610.14029](https://arxiv.org/abs/2610.14029)

    该论文提出了一种无需标签的实例级评估可靠性估计方法SCUR，通过潜在测量模型导出陪审团不确定性、测量交叉熵和扰动敏感性三个互补信号来构建风险分数，能够比现有方法更有效地检测出高共识风险样本这一评估盲点。

    

    LLM-as-a-judge（大语言模型作为评判者）评估通常将多个评判模型的决定聚合为多数投票，然而这种陪审团式的决策过程并不能保证对单个样本的评估是可靠的。我们提出了结构化交叉拟合不确定性风险（SCUR），这是一种基于评判陪审团行为来估计实例级评估可靠性的无标签方法。潜在测量模型作为测量系统分析的骨干，SCUR从中导出三个互补信号，分别捕获陪审团不确定性、测量交叉熵和扰动敏感性。这三个证据来源被组合成一个风险分数。我们在五个真实世界基准上，通过多数投票错误排序、选择性信任和风险分层来评估SCUR方法。此外，我们展示了我们的方法在检测高共识风险样本方面更加有效，而这些样本正是许多其他LLM-as-a-judge评估方法的盲点。

    arXiv:2610.14029v1 Announce Type: cross  Abstract: LLM-as-a-judge evaluation commonly aggregates multiple judges' decisions into a majority vote, yet this panel decision process does not guarantee that the evaluation of an individual item is reliable. We propose Structured Cross-fitted Uncertainty Risk (SCUR), a label-free approach for estimating instance-level evaluation reliability based on the judge panel's behavior. A latent measurement model serves as the backbone for measurement system analysis, from which SCUR derives three complementary signals capturing panel uncertainty, measurement cross-entropy, and perturbation sensitivity. These three sources of evidence are combined into a risk score. We evaluate the SCUR approach on five real-world benchmarks using majority-vote error ranking, selective trust, and risk stratification. Furthermore, we show our approach to be more effective at detecting high-consensus risk items, which are the blind spots for many other LLM-as-a-judge eva
    
[^110]: PRIME：面向MoE压缩的频率感知统一剪枝与合并

    PRIME: Frequency-Aware Unified Pruning and Merging for MoE Compression

    [https://arxiv.org/abs/2610.13982](https://arxiv.org/abs/2610.13982)

    PRIME提出了一个统一优化框架，将MoE压缩中的专家剪枝与合并决策联合建模为频率加权目标上的全局整数线性规划问题，从而在保留高频专家作为聚类中心的同时实现高效压缩。

    

    混合专家（MoE）模型已成为扩展大语言模型的有效范式，但其庞大的参数规模给部署带来了严峻的内存挑战。现有的压缩方法通常将专家剪枝和专家合并视为相互独立的操作，并依赖贪心启发式策略，这往往导致不可逆的信息丢失或表征污染。在本文中，我们提出了PRIME，一个通过自适应剪枝与合并来实现MoE压缩的统一优化框架。PRIME将离散的剪枝-合并分配步骤建模为在频率加权代理目标上的全局整数线性规划（Integer Linear Programming），在考虑专家使用情况的同时，联合选择剪枝和合并决策。我们通过在固定代表分配下的误差界分析论证了频率感知目标的合理性，并表明该公式倾向于将高频专家选为聚类中心，从而促进路由稳定性……

    arXiv:2610.13982v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) models have emerged as an effective paradigm for scaling large language models, yet their massive parameter footprint poses severe memory challenges for deployment. Existing compression methods typically treat expert pruning and merging as disjoint operations and rely on greedy heuristics, often leading to irreversible information loss or representational pollution. In this paper, we propose PRIME, a unified optimization framework for MoE compression via adaptive PRunIng and MErging. PRIME casts the discrete prune-merge assignment step as a global Integer Linear Programming over a frequency-weighted surrogate objective, jointly selecting pruning and merging decisions while accounting for expert usage. We motivate the frequency-aware objective through an error-bound analysis under fixed representative assignments, and show that the formulation favors high-frequency experts as centroids, promoting routing stabili
    
[^111]: SLICE：基于共形e值的序贯无似然推断

    SLICE: Sequential Likelihood-Free Inference via Conformal E-Values

    [https://arxiv.org/abs/2610.13855](https://arxiv.org/abs/2610.13855)

    提出SLICE方法，利用共形e值构建针对难解似然模拟模型的精确随时有效置信序列，在持续监测下保持有效，并通过摊销神经比率估计器近似理想似然比得分。

    

    许多科学模型可以进行模拟但其似然函数难以处理，这使得频率学派的量化充满困难。我们为此类模型的参数构建了精确的、随时有效的置信序列。在每次观测时，利用来自每个候选原假设的全新模拟，将一个正的可预测得分共形化为条件e值；这些e值的乘积构成一个检验鞅，我们再对参数在该鞅上进行求逆。无论得分本身的准确性如何，有效性对任何此类得分都成立。我们证明了理想似然比得分能够最大化期望对数增长，这启发我们使用摊销式神经比率估计器来近似它。在合成基准测试、LLM API中模型路由的审计以及微透镜数据上，所得到的置信序列在持续监测下始终保持有效，而固定样本方法则无法做到这一点；同时，神经得分与精确似然比以及针对特定应用的得分相比仍具有竞争力。

    arXiv:2610.13855v1 Announce Type: cross  Abstract: Many scientific models can be simulated but have intractable likelihoods, making frequentist uncertainty quantification difficult. We construct exact anytime-valid confidence sequences for the parameters of such models. At each observation, fresh simulations from each candidate null conformalize a positive predictable score into a conditional e-value; their product is a test martingale, which we invert over the parameter. Validity holds for any such score, whatever its accuracy. We show that the oracle likelihood-ratio score maximizes expected log growth, which motivates approximating it with an amortized neural ratio estimator. On synthetic benchmarks, an audit of model routing in an LLM API, and microlensing data, the resulting confidence sequences stay valid under continuous monitoring, where fixed-sample methods do not, and neural scores remain competitive with exact likelihood ratios and application-specific scores.
    
[^112]: 基于Stein位移场的密度比估计

    Density Ratio Estimation with Stein Displacement Fields

    [https://arxiv.org/abs/2610.12437](https://arxiv.org/abs/2610.12437)

    该论文提出通过Stein位移场参数化密度比（将对数比建模为负的基础分布Stein算子作用于位移场），用单个凸优化问题统一了分布偏移的统计与动力学描述，并据此发展出无需重训即可修正预训练采样器的push-forward算法和拉近数据分布的pull-back算法。

    

    密度比从概率质量的角度量化分布偏移，而位移场则从动力学的角度描述一个分布是如何被输运到另一个分布上的。尽管两者能提供互补的见解，但它们通常被分开估计，且将其中一种转换为另一种需要后处理。在本文中，我们通过一个作用于基础分布的位移场来参数化目标分布与基础分布之间的密度比：对数比被建模为负的基础分布Stein算子作用于该位移场的结果，仅相差一个归一化常数。这使得我们能够通过单个凸优化问题同时获得分布偏移的统计学与动力学描述。迭代这一“估计—移动”步骤可得到两种推断算法：push-forward（前推）移动模型，并在无需重新训练的情况下修正预训练的采样器；而pull-back（回拉）则将数据移动得更接近基础分布。

    arXiv:2610.12437v1 Announce Type: cross  Abstract: Density ratios quantify distribution shift from a probability-mass point of view, whereas displacement fields describe, from a dynamical point of view, how one distribution is transported onto another. Although both offer complementary insights, they are usually estimated separately, and converting one into the other requires post-processing. In this paper, we estimate the density ratio between a target and a base distribution by parametrizing it through a displacement field acting on the base: the log-ratio is modeled as minus the Stein operator of the base applied to the field, up to a normalizing constant. This gives both statistical and dynamical descriptions of the distribution shift through a single convex optimization problem. Iterating this estimate-and-move step gives two inference algorithms: push-forward moves the model and corrects a pretrained sampler without retraining it, whereas pull-back moves the data closer to the ba
    
[^113]: ISBO：基于INLA-SPDE方法与对数高斯Cox过程模型的可扩展时空贝叶斯优化

    ISBO: Scalable Spatio-Temporal Bayesian Optimization with Log Gaussian Cox Process Models via the INLA-SPDE Approach

    [https://arxiv.org/abs/2610.12213](https://arxiv.org/abs/2610.12213)

    该论文提出了首个面向时空数据的可扩展贝叶斯优化框架ISBO，通过对数高斯Cox过程建模与INLA-SPDE推断方法，能够以最少的评估次数稳定定位高强度区域及潜在强度峰值。

    

    贝叶斯优化（BO）是一种高效优化昂贵黑盒目标函数的流行方法。然而，使用标准高斯过程的贝叶斯优化并不适合时空问题空间中常用的双重随机Cox过程。我们提出了INLA-SPDE时空贝叶斯优化（ISBO）：首个针对时空数据的可扩展贝叶斯优化框架，该框架使用对数高斯Cox过程（LGCP）对对数强度进行建模，并通过积分嵌套拉普拉斯近似与随机偏微分方程（INLA-SPDE）方法进行推断。在网格上使用Matern场可生成稀疏高斯马尔可夫随机场，使INLA在整个序贯优化过程中提供快速且准确的后验推断。ISBO能够以最少的评估次数稳定地定位高强度区域以及潜在强度的峰值。带有掩蔽机制的时变上置信界采集函数可避免重复访问……

    arXiv:2610.12213v1 Announce Type: cross  Abstract: Bayesian Optimization (BO) is a popular method for efficiently optimizing expensive black-box objectives. However, BO utilizing standard Gaussian Processes is ill-suited for doubly stochastic Cox Processes that are often used in spatio-temporal problem spaces. We introduce INLA-SPDE Spatio-Temporal Bayesian Optimization (ISBO): the first scalable BO framework for spatio-temporal data, that models the log-intensity with a Log-Gaussian Cox Process(LGCP) and performs inference via Integrated Nested Laplace Approximation and Stochastic Partial Differential Equations (INLA-SPDE) approach. Using a Matern field on meshes yields a sparse Gaussian Markov Random Field, where INLA provides fast and accurate posterior inference throughout sequential optimization. ISBO stably locates high-intensity regions and the peak of the latent intensity with minimal evaluations. A time-varying Upper Confidence Bound acquisition with masking avoids revisits, w
    
[^114]: 用于变分序贯蒙特卡洛的可微系统重采样

    Differentiable Systematic Resampling for Variational Sequential Monte Carlo

    [https://arxiv.org/abs/2610.12094](https://arxiv.org/abs/2610.12094)

    提出可微系统重采样（DSR），一种温度控制的系统重采样松弛方法，在保持其结构特性并具有可证明的指数收敛偏差的同时，实现了完全梯度流动，且计算开销远低于基于最优传输的方法。

    

    粒子滤波器是非线性状态估计的标准工具，但其重采样步骤是离散的，阻碍了变分序贯蒙特卡洛中基于梯度的学习。我们提出了可微系统重采样（DSR），这是一种对系统重采样的温度控制松弛方法，它在保持系统重采样基于CDF排序的带状结构的同时，实现了完全的梯度流动。当温度趋于零时，DSR收敛到精确的系统重采样，并且我们证明了由此引入的偏差具有逐点指数收敛速率。与基于最优传输的可微重采样方法相比，DSR避免了迭代求解器的使用，计算开销显著更低。在随机动力系统和真实世界手写数据上的实验表明，DSR在滤波和动力学学习方面取得了相当或更优的性能。

    arXiv:2610.12094v1 Announce Type: cross  Abstract: Particle filters are a standard tool for nonlinear state estimation, but their resampling step is discrete, preventing gradient-based learning in variational sequential Monte Carlo. We introduce Differentiable Systematic Resampling (DSR), a temperature-controlled relaxation of systematic resampling, that preserves the CDF-ordered, banded structure of systematic resampling while enabling full gradient flow. DSR converges to exact systematic resampling as the temperature vanishes, and we prove a pointwise exponential convergence rate for the induced bias. Compared to optimal-transport-based differentiable resampling, DSR avoids iterative solvers and has substantially lower computational overhead. Experiments on stochastic dynamical systems and real-world handwriting data show that DSR achieves comparable or superior filtering and dynamics learning performance.
    
[^115]: 耗散知识动力学模型的可辨识性：设计激励下的精确恢复与观测数据上的退化

    Identifiability of a dissipative knowledge-dynamics model: exact recovery under designed excitation, degeneration on observational data

    [https://arxiv.org/abs/2610.09889](https://arxiv.org/abs/2610.09889)

    该论文将人类学习建模为参数具有机制含义的耗散常微分方程组，证明了在设计激励条件下模型参数可从数据中被精确恢复（双概念情形有闭式解），而在仅依赖观测数据时可辨识性会退化，并提出了数值精确等价且速度大幅提升的半隐式L-稳定批量求解器。

    

    人类学习是一个耗散动力学过程：熟练度通过练习积累，因遗忘而衰减，并在相互依赖的概念之间传播。我们将其建模为一个非线性耗散常微分方程组，其参数具有机制层面的含义（编码先修耦合关系的概念传递矩阵、各概念的遗忘率，以及饱和的练习-响应增益），并研究这些参数何时能够真正从数据中被恢复。我们在显式激励条件下证明了相应逆问题的结构可辨识性定理，针对双概念情形给出了构造性的闭式恢复方法，并同时给出了单调性、鲁棒性与L-稳定性结果。我们为耗散子系统推导了一种半隐式L-稳定数值格式，以及一个与逐轨迹公式数值等价的批量求解器（预测逐位一致，梯度误差达 $10^{-10}$），同时速度提升两个数量级。

    arXiv:2610.09889v1 Announce Type: new  Abstract: Human learning is a dissipative dynamical process: mastery accumulates through practice, decays through forgetting, and propagates across interdependent concepts. We model it as a nonlinear dissipative system of ordinary differential equations whose parameters are mechanistically meaningful (a concept-transfer matrix encoding prerequisite coupling, per-concept forgetting rates, and a saturating practice-response gain), and we study when those parameters can actually be recovered from data. We prove a structural identifiability theorem for the associated inverse problem under explicit excitation conditions, with constructive closed-form recovery for the two-concept case, together with monotonicity, robustness and L-stability results. We derive a semi-implicit L-stable scheme for the dissipative subsystem and a batched solver numerically equivalent to the per-trajectory formulation (bit-exact predictions, gradients to $10^{-10}$) yet two o
    
[^116]: 最佳优化器取决于批量大小

    The Best Optimizer Depends on Batch Size

    [https://arxiv.org/abs/2610.08975](https://arxiv.org/abs/2610.08975)

    该论文挑战了“某一批量大小下最佳的优化器在其他批量大小下也最佳”的常见假设，证明Muon缺乏一致的缩放规则，且即使经过大量超参数调优，语言模型预训练的最佳优化器仍会随批量大小而改变。

    

    大量新的自适应优化器被设计用于高效估计和利用小批量梯度统计量来塑造参数更新，但它们通常只在单一批量大小下进行基准测试。超参数缩放规则承诺在批量大小和梯度噪声变化时保持性能，这暗示着在某一批量大小下最佳的优化器在另一批量大小下也应保持最佳。我们通过以下发现挑战了这种开发和评估优化器的方法：(1) Muon没有一种原则性的缩放规则能在各种训练设置中保持一致有效；(2) 即使经过大量的超参数调优，语言模型预训练的最佳优化器也会随批量大小而变化。

    arXiv:2610.08975v1 Announce Type: new  Abstract: A plethora of new adaptive optimizers are designed to efficiently estimate and use minibatch gradient statistics to shape parameter updates, but they are typically benchmarked at a single batch size. Hyperparameter scaling rules promise to preserve performance as batch size and gradient noise change, suggesting that the best optimizer at one batch size should remain the best at another. We challenge this approach to developing and evaluating optimizers by showing: (1) no principled scaling rule for Muon works consistently across training settings, and (2) the best optimizer for language model pretraining changes with batch size even after extensive hyperparameter tuning.
    
[^117]: 随机梯度下降上升法在非凸-PL极小极大博弈中是次优的

    Stochastic Gradient Descent Ascent is Suboptimal for Nonconvex-PL Min-Max Games

    [https://arxiv.org/abs/2610.07814](https://arxiv.org/abs/2610.07814)

    该论文首次建立了非凸-PL极小极大博弈中固定时间尺度比双时间尺度SGDA的紧致复杂度下界，证明SGDA本质上次优——其下界与现有上界匹配、与Smoothed-AGDA形成复杂度分离，且当时间尺度比小于o(κ²)时甚至无法找到驻点。

    

    arXiv:2610.07814v1 公告类型： cross 摘要：在非凸极小极大博弈中，通过调整时间尺度比和步长，随机梯度下降上升法（SGDA）究竟能走多远？我们针对非凸-PL（NC-PL）博弈回答了这一问题，首次建立了固定时间尺度比和非递增步长下双时间尺度SGDA的紧致复杂度。对于满足内层μ-PL不等式的ℓ-光滑博弈，我们证明了复杂度下界Ω(κ²ℓε⁻²+κ⁴ℓσ²ε⁻⁴)，其中κ=ℓ/μ为条件数，σ²为梯度方差，ε度量外层梯度范数。该下界与现有的SGDA上界相匹配，并建立了与Smoothed-AGDA（Yang等人，22'）之间的复杂度分离。此外，我们证明当SGDA的时间尺度比小至o(κ²)时，它可能无法找到驻点。我们的负面结果凸显了SGDA在NC-PL博弈中的根本局限性，并……（原文摘要至此截断）

    arXiv:2610.07814v1 Announce Type: cross  Abstract: How far can stochastic gradient descent ascent (SGDA) go by tuning its timescale ratio and step sizes in nonconvex min-max games? We answer this question for nonconvex-PL (NC-PL) games by establishing the first tight complexity of two-timescale SGDA with a fixed timescale ratio and non-increasing step sizes. For $\ell$-smooth games with an inner $\mu$-PL inequality, we prove a complexity lower bound $\Omega(\kappa^2\ell\varepsilon^{-2}+\kappa^4\ell\sigma^2\varepsilon^{-4})$, where $\kappa=\ell/\mu$ is the condition number, $\sigma^2$ is the gradient variance, and $\varepsilon$ measures the outer gradient norm. This matches existing SGDA upper bounds and establishes a complexity separation from Smoothed-AGDA (Yang et al., 22'). In addition, we show that SGDA can fail to find a stationary point when its timescale ratio is as small as $o(\kappa^2)$. Our negative results highlight the fundamental limitation of SGDA in NC-PL games, and just
    
[^118]: 基于检索的上下文学习：一个领域自适应框架

    Retrieval-Based In-Context Learning: A Domain Adaptation Framework

    [https://arxiv.org/abs/2610.05717](https://arxiv.org/abs/2610.05717)

    该论文首次将基于检索的上下文学习（ICR）形式化为领域自适应问题，在灵活的分布偏移条件下建立了理论保证，量化了该学习范式的优势与局限。

    

    上下文检索（ICR）是一种基于检索的上下文学习（ICL）形式，其中示例演示是根据与查询的相似性从源数据库中检索出来的，而不是独立采样的。在这项工作中，我们将ICR形式化为一类领域自适应问题，其中数据库的源分布 $P$ 可能与测试查询-标签对的目标分布 $Q$ 不同。我们在一类灵活的分布偏移下研究了ICR的性能，这类偏移大幅扩展了先前的工作，并建立了理论保证以量化这种学习范式的优势与潜在缺陷。我们的理论通过合成任务和语言任务上的实验得到了验证。

    arXiv:2610.05717v2 Announce Type: replace  Abstract: In-context retrieval (ICR) is a retrieval-based form of in-context learning (ICL) in which demonstrations are retrieved from a source database based on similarity to the query, rather than sampled independently. In this work, we formulate ICR as a type of domain adaptation problem, where the source distribution $P$ of the database may differ from the target distribution $Q$ of the test query-label pair. We investigate the performance of ICR under a flexible class of distributional shifts that substantially extends prior work \citep{li2024fine,guo2025retrieval}, and establish theoretical guarantees that quantify the benefits and pitfalls of this learning paradigm. Our theory is verified by experiments on synthetic and language tasks.
    
[^119]: 面向多变量结局的基于偏好的治疗效应的定向主动学习

    Targeted Active Learning for Preference-Based Treatment Effects on Multivariate Outcomes

    [https://arxiv.org/abs/2610.03824](https://arxiv.org/abs/2610.03824)

    该论文提出了一种定向主动学习框架，通过从临床专家处选择性查询最有信息量的结局组合两两比较，来最大程度降低基于偏好的治疗效应与最优策略估计中的不确定性，而非仅仅最大化对偏好规则本身的信息增益。

    

    arXiv:2610.03824v2 公告类型：替换 摘要：治疗疗效传统上是以单一主要结局为依据来验证的。然而，临床决策通常需要综合考虑多个结局，在预期获益与潜在风险之间进行权衡。而不同患者对这些结局赋予的相对价值差异很大。在给定关于结局组合的偏好规则的情况下，可以定义并估计治疗效应和最优策略。但这样的规则在实践中很少现成可用：它本身必须通过结局组合的两两比较来估计，而从临床专家处收集此类比较的成本高昂。我们提出了一个主动学习框架，用于选择向专家查询哪些比较。标准的准则最大化的是对偏好规则本身所获得的信息量。我们则转而以目标量为导向，选择最能降低由所学规则导出的治疗效应和最优策略不确定性的查询……（摘要在此处截断）

    arXiv:2610.03824v2 Announce Type: replace  Abstract: Treatment efficacy is traditionally demonstrated on the basis of a single primary outcome. However, clinical decision-making usually requires consideration of multiple outcomes, balancing expected benefits against potential risks. The relative value assigned to these outcomes varies substantially from one patient to another. Given a preference rule over outcome profiles, treatment effects and optimal policies can be defined and estimated. Such a rule is rarely available in practice: it must itself be estimated from pairwise comparisons of outcome profiles, which are costly to collect from clinical experts. We propose an active learning framework that selects which comparisons to query. Standard criteria maximize the information gained on the preference rule itself. We instead target the quantities of interest, and select the query that most reduces uncertainty on the treatment effect and on the optimal policy induced by the learned r
    
[^120]: 面向高斯过程潜变量模型的摊销结构化随机变分推断

    Amortized Structured Stochastic Variational Inference for Gaussian Process Latent Variable Models

    [https://arxiv.org/abs/2610.03647](https://arxiv.org/abs/2610.03647)

    本文将摊销结构化随机变分推断应用于高斯过程潜变量模型，通过让潜空间的变分后验条件依赖于诱导点的取值，突破了平均场变分近似的局限，从而在数据流形重建的多项指标上取得了改进。

    

    许多机器学习方法旨在逼近数据所在的低维流形。这类方法的一个理想特性是能够捕捉所学习流形的认知不确定性。高斯过程潜变量模型就是实现这一目标的模型之一，其中从潜空间出发的高斯过程（GP）映射提供了对流形不确定性的估计。然而，这种不确定性估计的有效性受到GP诱导点与潜变量之间平均场变分近似的限制。在本工作中，我们应用摊销结构化随机变分推断，使潜空间的变分后验可以有条件地依赖于诱导点的取值。我们证明了这种更灵活的变分后验在与数据流形上点的重建相关的多项指标上均有所提升。

    arXiv:2610.03647v1 Announce Type: cross  Abstract: Many machine learning methods aim to approximate the lower-dimensional manifold on which the data lives. A desirable feature of such methods is that they should capture the epistemic uncertainty of this learned manifold. One model that achieves this is the Gaussian Process Latent Variable Model, in which a Gaussian Process (GP) mapping from the latent space provides an estimate of the uncertainty of the manifold. However, the effectiveness of this uncertainty estimation is limited by the mean-field variational approximation between the GP inducing points and the latent variables. In this work, we apply Amortized Structured Stochastic Variational Inference to allow the variational posterior for the latent space to be conditionally dependent on the value of the inducing points. We demonstrate that this more flexible variational posterior improves several metrics relating to the reconstruction of points on the data manifold.
    
[^121]: 异方差规范多面张量分解

    Heteroskedastic Canonical Polyadic Tensor Decomposition

    [https://arxiv.org/abs/2610.00498](https://arxiv.org/abs/2610.00498)

    本文提出异方差CP分解（HCP），通过引入低秩精度张量来建模张量条目的异方差性，并采用交替分块坐标上升方法从含噪观测中同时估计低秩均值与精度张量，其计算复杂度与CP-ALS相当。

    

    当最小化平方误差损失时，流行的CP分解可以被解释为高斯模型中的参数推断，该模型具有低秩均值张量，且张量各条目的方差恒定。我们提出了异方差CP（HCP），它用一个非常数、低秩的精度张量来建模逐条目的变异性，并开发了一种交替分块坐标上升方法，从含噪观测中同时恢复低秩均值张量和精度张量。我们的方法在计算上具有竞争力，其因子更新的首阶复杂度与CP-ALS相同。我们在合成实验和一个脑电图（EEG）应用中展示了HCP的效果。

    arXiv:2610.00498v1 Announce Type: cross  Abstract: When minimizing the squared-error loss, the popular CP decomposition can be interpreted as parameter inference in a Gaussian model with a low-rank mean tensor and constant variance across the tensor entries. We introduce heteroskedastic-CP (HCP), which models entrywise variability with a non-constant, low-rank precision tensor, and develop an alternating block-coordinate ascent method to recover both the low-rank mean and precision tensors from noisy observations. Our procedure is computationally competitive, with the same leading-order factor-update complexity as CP-ALS. We demonstrate HCP on synthetic experiments and an EEG application.
    
[^122]: 关于图像分割中条件独立性假设的放宽

    On the Relaxation of Conditional Independence Assumption for Image Segmentation

    [https://arxiv.org/abs/2609.38930](https://arxiv.org/abs/2609.38930)

    该论文提出用空间局部化依赖（SLD）结构替代RankSEG方法中的条件独立性假设，并借助倒数矩近似与不动点优化策略克服计算瓶颈，从而有效捕获局部标签相关性并提升模糊、低对比度场景下的图像分割性能。

    

    在语义分割中，最近一类名为RankSEG的方法在推理阶段直接优化Dice/IoU分数，在不修改模型训练的情况下提高了与评估指标的一致性。尽管RankSEG在理论和实证上都取得了成功，但它依赖于限制性较强的条件独立性假设（CIA），该假设忽略了关键的标签相关性，因此在模糊或低对比度场景下会导致性能下降。然而，考虑完全的标签依赖在计算上是不可行的，需要 $\mathcal{O}(d^3)$ 的时间复杂度。为了解决这一问题，我们用空间局部化依赖（SLD）结构替代CIA，该结构在保持依赖模型可处理性的同时捕获局部标签相关性。我们进一步通过倒数矩近似并结合一种新颖的不动点优化策略克服了剩余的计算瓶颈，从而消除了穷举搜索。所提出的算法实现了高……（原文摘要至此截断）

    arXiv:2609.38930v1 Announce Type: cross  Abstract: In semantic segmentation, a recent line of RankSEG methods directly optimizes Dice/IoU scores at inference time, improving alignment with evaluation metrics without modifying model training. Despite its theoretical and empirical success, RankSEG relies on the restrictive Conditional Independence Assumption (CIA), which ignores crucial label correlations and therefore degrades performance in ambiguous or low-contrast scenarios. However, accounting for full label dependence is computationally prohibitive, requiring $\mathcal{O}(d^3)$ time. To address this, we replace the CIA with a Spatially Localized Dependence (SLD) structure that captures local label correlations while keeping the dependence model tractable. We further overcome the remaining computational bottleneck via a Reciprocal Moment Approximation coupled with a novel fixed-point optimization strategy that eliminates exhaustive search. The proposed algorithm achieves a highly pr
    
[^123]: 基于显式锚点-正样本-负样本耦合的多边际逆向最优传输对比学习

    Multi-Marginal Inverse Optimal Transport for Contrastive Learning Via Explicit Anchor-Positive-Negative Coupling

    [https://arxiv.org/abs/2609.33741](https://arxiv.org/abs/2609.33741)

    该论文提出Neg-MMIOT-CL方法，通过显式建模锚点-正样本-负样本三元组的多边际逆向最优传输耦合，克服了现有逆向OT方法因不利用负样本而导致的维度坍塌问题，提升了对比学习的下游性能。

    

    基于逆向最优传输（OT）的表示学习方法，通过学习表示使得表示空间中一对数据边缘分布之间的全局OT耦合集中在正样本对上，这与以往主要关注成对匹配的方法形成对比。然而，这些方法并未利用负样本对，因此其方式并非真正意义上的对比学习。我们证明这会导致维度坍塌问题，进而降低下游任务性能。为缓解这一问题，我们开发了一种新颖的多边际（MM）逆向最优传输（IOT）对比学习（CL）方法，称为Neg-MMIOT-CL，该方法学习表示，使得三个数据边缘分布之间的全局多边际最优传输（MMOT）耦合——相对于表示空间中数据点三元组之间精心设计的基础代价函数——集中在锚点-正样本-负样本三元组上。

    arXiv:2609.33741v2 Announce Type: replace  Abstract: Inverse Optimal Transport (OT) based methods for representation learning learn representations such that the global OT coupling between a pair of data marginals in the representation space, concentrates on the positive pairs. This is in contrast to previous methods that primarily focused on pairwise matching. However, these methods $\textit{DO NOT}$ utilize negative pairs and hence are not truly contrastive in their approach. We show that this leads to issues of dimensional collapse and hence degraded downstream performance. To alleviate this, we develop a novel multi-marginal (MM) inverse OT (IOT) contrastive learning (CL) approach called Neg-MMIOT-CL, which learns representations such that the global multi-marginal OT (MMOT) coupling between a triple of data marginals, with respect to a carefully designed ground-cost between triplets of data points in the representation space, concentrates on the anchor-positive-negative $\textit{t
    
[^124]: WTF?! 基于Wasserstein倾斜流映射的无模拟强化学习

    WTF?! Simulation-Free Reinforcement Learning with Wasserstein-Tilted Flow Maps

    [https://arxiv.org/abs/2609.27033](https://arxiv.org/abs/2609.27033)

    提出WTF框架，通过基于预训练漂移构建的Wasserstein最优传输正则化器，将奖励微调问题等价转化为流上的确定性最优控制问题，实现了无需模拟的强化学习微调，是首个原生于流映射的端到端微调方案。

    

    奖励微调旨在更新预训练的基于流的生成模型，以提升其生成样本的下游奖励。现有方法通常将该问题表述为从奖励倾斜分布中采样，即KL正则化奖励最大化问题的解。本文引入了一种直接基于预训练漂移构建的最优传输正则化器。与KL奖励倾斜不同，所得目标是使个体样本向更高奖励方向传输，而非对基础分布进行重新加权。我们证明了该问题等价于流上的一个确定性最优控制问题。给定预训练的流映射，这种等价性催生了一种用于微调生成流的无模拟强化学习算法。我们将所得框架称为Wasserstein倾斜流映射，这是首个原生于流映射的端到端微调方案。其输出是一个微调后的流映射……

    arXiv:2609.27033v1 Announce Type: new  Abstract: Reward fine-tuning aims to update a pre-trained flow-based generative model to improve the downstream reward of its generated samples. Existing methods typically formulate this problem as sampling from a reward-tilted distribution, the solution to a KL-regularized reward-maximization problem. Here, we introduce an optimal transport regularizer built directly from the pre-trained drift. Unlike KL reward tilting, the resulting objective transports individual samples toward higher reward rather than reweighting the base distribution. We show that the resulting problem is equivalent to a deterministic optimal control problem on the flow. Given a pre-trained flow map, this equivalence yields a simulation-free reinforcement learning algorithm for fine-tuning generative flows. We call the resulting framework Wasserstein-Tilted Flow Maps (WTF), the first end-to-end fine-tuning recipe native to flow maps. The output is a fine-tuned flow map that 
    
[^125]: PICPIs：预测区间条件化的预测区间

    PICPIs: Prediction-Interval-Conditional Prediction Intervals

    [https://arxiv.org/abs/2609.25388](https://arxiv.org/abs/2609.25388)

    该论文提出了预测区间条件化预测区间这一新框架，通过自洽条件使预测区间同时定义预测值分层并保证该层内的平均结果落在同一区间内，从而在共形预测中填补了边际有效性分辨率不足与完全条件保证不可实现之间的空白。

    

    arXiv:2609.25388v1 公告类型：交叉 摘要：统计学中的一个经典问题是，在对不可观测的目标进行推断时，应当以哪些可观测的量为条件。对于非参数不确定性量化中的共形预测而言，标准的边际有效性在决策所依据的预测值上提供的分辨率有限，而相对于协变量的完全条件保证已被证明是无法实现的。我们通过引入一个基于预测的条件化框架来填补这一空白，我们将其称为预测区间条件化预测区间。形式上，PICPI 是一个满足自洽条件的区间 I：对于预测模型 p、上下文协变量 X 和结果 Y，有 E[Y | p(X) ∈ I] ∈ I。因此，这样一个区间同时定义了一个预测值的分层，并证明该分层中的平均结果落在同一区间内。这一自洽条件产生了数据自适应的……

    arXiv:2609.25388v1 Announce Type: cross  Abstract: A classical question in statistics is which observable quantities to condition on when drawing inferences about unobservable targets. For conformal prediction in nonparametric uncertainty quantification, standard marginal validity offers limited resolution at the prediction values on which decisions are based, and fully conditional guarantees with respect to the covariates are provably unattainable. We address this gap by introducing a prediction-based conditioning framework that we refer to as Prediction-Interval-Conditional Prediction Intervals (PICPIs). Formally, a PICPI is an interval $I$ satisfying a self-consistency condition: $$\mathbb{E} [Y \mid p(X) \in I] \in I,$$ for predictive model $p$, contextual covariate $X$, and outcome $Y$. Thus, an interval simultaneously defines a stratum of prediction values and certifies that the mean outcome in that stratum lies in the same interval. This self-consistency condition yields data-ad
    
[^126]: OSCAR：面向AI排名的顺序感知评分与校准框架

    OSCAR: Order-aware Scoring and Calibration for AI Rankings

    [https://arxiv.org/abs/2609.24128](https://arxiv.org/abs/2609.24128)

    本文提出OSCAR框架，通过在LLM成对评估聚合中建模评委特定的位置等系统性呈现效应，实现AI排名的顺序感知评分与校准，并证明忽略位置效应会严重扭曲最优排名结果。

    

    评委特定的敏感性对于聚合成对的大语言模型（LLM）评估十分有用，但其解释取决于排名模型中纳入了哪些系统性呈现效应。我们提出了OSCAR，一个用于AI排名评分与校准的顺序感知框架，并将“位置效应”作为其中一种呈现效应加以研究。在来自18位评估者的已发布判定数据中，所有回答的A减B分数差异范围从-63.11到98.31个百分点。在已发布文本映射的条件下，将问题文本、回答文本、候选身份和评委进行匹配后，得到总体差异为24.22个百分点（95%置信区间[22.90, 25.54]）。一项受控计算分离了其潜在后果：当真实敏感性固定为1时，若省略大小为4的位置截距，总体最优斜率将降至0.0771。我们通过引入评委特定的位置项、长度项和家族项，对基于敏感性的排名方法进行了扩展，并对其进行了刻画……

    arXiv:2609.24128v1 Announce Type: cross  Abstract: Judge-specific sensitivity is useful for aggregating pairwise LLM evaluations, but its interpretation depends on which systematic presentation effects the ranking model includes. We introduce OSCAR, an order-aware framework for scoring and calibrating AI rankings, and study position as one such effect. In released judgments from 18 evaluators, the all-response A-minus-B score difference ranges from $-63.11$ to $98.31$ percentage points. Matching question text, response texts, candidate identities, and judge within the released table gives an overall difference of $24.22$ points (95% interval $[22.90,25.54]$), conditional on the released text mapping. A controlled calculation isolates the potential consequence: with true sensitivity fixed at one, omitting a position intercept of four reduces the population-optimal slope to $0.0771$. We extend sensitivity-based ranking with judge-specific position, length, and family terms, characterize 
    
[^127]: 面向随机非凸—（强）凹极小极大优化的单循环随机投影阻尼外梯度方法

    Single-Loop Stochastic Projected Damped Extragradient Methods for Stochastic Nonconvex--(Strongly) Concave Minimax Optimization

    [https://arxiv.org/abs/2609.21747](https://arxiv.org/abs/2609.21747)

    提出了保持单循环结构的随机投影阻尼外梯度方法 SPDE 及其递归方差缩减变体 VR-SPDE，用于随机非凸—（强）凹极小极大优化，并同时为博弈平稳性和优化平稳性提供了更优的随机一阶预言机复杂度保证。

    

    针对随机非凸—（强）凹极小极大优化问题，我们提出了单循环随机投影阻尼外梯度方法，并对博弈平稳性和优化平稳性均给出了复杂度保证。我们的方法将随机投影阻尼外梯度（SPDE）方法与递归方差缩减变体 VR-SPDE 相结合，二者均保持单循环结构。在具有一致有界方差的无偏随机梯度预言机下，SPDE 能够找到 ε-博弈平稳点，其随机一阶预言机（SFO）复杂度在非凸—强凹和非凸—凹设置下分别为 O(κ ε⁻⁴) 和 O(ε⁻⁵)，其中 κ = L/μ。在随机梯度满足额外的均方 Lipschitz 条件下，VR-SPDE 将上述博弈平稳性复杂度分别改进至 O(κ^{3/2} ε⁻³) 和 O(ε^{-9/2})……

    arXiv:2609.21747v1 Announce Type: cross  Abstract: We develop single-loop stochastic projected damped extragradient methods for stochastic nonconvex--(strongly) concave minimax optimization, with complexity guarantees for both game stationarity (GS) and optimization stationarity (OS). Our approach combines a stochastic projected damped extragradient (SPDE) method with a recursive variance-reduced variant, VR-SPDE, both of which retain a single-loop structure. Under an unbiased stochastic gradient oracle with uniformly bounded variance, SPDE finds an $\varepsilon$-game-stationary point with stochastic first-order oracle (SFO) complexities of $O(\kappa\varepsilon^{-4})$ and $O(\varepsilon^{-5})$ in the nonconvex--strongly concave and nonconvex--concave settings, respectively, where $\kappa=L/\mu$. Under an additional mean-square Lipschitz condition on the stochastic gradients, VR-SPDE improves these GS complexities to $O(\kappa^{3/2}\varepsilon^{-3})$ and $O(\varepsilon^{-9/2})$, respect
    
[^128]: 为什么学习会重新发现闭式对角正则化器

    Why Learning Rediscovers the Closed-Form Diagonal Regularizer

    [https://arxiv.org/abs/2609.09656](https://arxiv.org/abs/2609.09656)

    本文提出对角饱和原理，证明当截断噪声各向同性时，贝叶斯最优的Tikhonov正则化是仅由先验决定的闭式幂律，因此学习得到的对角正则化器难以稳健超越闭式解，在声学房间逆问题中闭式解已接近最优。

    

    我们在模态逆问题中确定了一个对角饱和原理：当截断噪声为各向同性时，贝叶斯最优的Tikhonov形状是一个闭式幂律 Gamma_k ∝ lambda_k^|s|，该形状仅由先验决定，与具体的域无关。Berry的随机波猜想使各模态之间的截断噪声去相关，而Weyl的特征值计数定律提供了足够多的模态数，使得即使在经验上违反Berry猜想的情况下，该结论依然成立。二者共同预测了在整个逐模态参数族上的损失景观近似平坦，因此留给对角正则化器稳健超越闭式解的空间非常有限。在FEM模拟的声学房间上，闭式解相对于逐房间的oracle调参在各观测窗口下都接近最优；并且在相同数据上训练的三种对角架构尽管学习到了性质截然不同的谱，其重建误差与闭式解的差距仍在1个百分点以内。该框架还可通过一个已知的扩展方式推广到热扩散问题……

    arXiv:2609.09656v1 Announce Type: new  Abstract: We identify a diagonal saturation principle in modal inverse problems: when truncation noise is isotropic, the Bayes-optimal Tikhonov shape is a closed-form power law Gamma_k proportional to lambda_k^|s| set by the prior alone, independent of the domain. Berry's random-wave conjecture decorrelates the truncation noise across modes, and Weyl's eigenvalue counting law supplies enough modes for the conclusion to survive empirical Berry violations. Together they predict an approximately flat loss landscape across the per-mode family, leaving narrow scope for a diagonal regularizer to robustly beat the closed form. On FEM-simulated acoustic rooms, the closed form is near-optimal relative to per-room oracle tuning across observation windows, and three diagonal architectures trained on the same data match its reconstruction error within 1 pp despite learning qualitatively different spectra. The framework extends to heat diffusion via a known ex
    
[^129]: 一致遍历马尔可夫链生成的多元鞅和的高斯逼近

    Gaussian Approximation for Multivariate Martingale Sums from Uniformly Ergodic Markov Chains

    [https://arxiv.org/abs/2609.09480](https://arxiv.org/abs/2609.09480)

    本文为一致遍历马尔可夫链生成的多元鞅和建立了高阶 Wasserstein 距离下的显式高斯逼近界，并首次在平衡增量情形及马尔可夫链加性泛函中获得最优的 $O(n^{-1/2})$ 逼近率。

    

    我们为一致遍历马尔可夫链生成的多元鞅差之和，在高阶 Wasserstein 距离 $W_p$（$p\geq2$）下建立了高斯逼近界。在 $L^{(2+\eta)p}$ 矩条件（$\eta>0$）下，我们给出了显式界 $$ O\left( p^3 \|A\|_4^2 + pd^{1/4}\|A\|_2^{1/2}\|A\|_4^2 \right) $$ 其中 $A\in\mathbb{R}^n$ 汇总了 $n$ 个单个鞅增量的 $L^{(2+\eta)p}$ 规模。在平衡增量情形（即各个增量具有 $n^{-1/2}$ 量级的可比规模）下，该界给出了当 $p$ 与维度 $d$ 固定时首个最优的 $O(n^{-1/2})$ 高斯逼近率。由此，我们还首次获得了一致遍历马尔可夫链多元加性泛函的最优 $O(n^{-1/2})$ $W_p$ 高斯逼近率。我们的分析发展了两种技术，用以处理高阶 Wasserstein 距离与时间依赖性之间的相互作用。

    arXiv:2609.09480v1 Announce Type: cross  Abstract: We develop Gaussian approximation bounds in higher-order Wasserstein distance $W_p$, $p\geq2$, for sums of multivariate martingale differences generated by a uniformly ergodic Markov chain. Under an $L^{(2+\eta)p}$-moment condition with $\eta>0$, we establish the explicit bound $$ O\left( p^3 \|A\|_4^2 + pd^{1/4}\|A\|_2^{1/2}\|A\|_4^2 \right) $$ where $A\in\mathbb{R}^n$ collects the $L^{(2+\eta)p}$-sizes of the $n$ individual martingale increments. In the balanced-increment regime where the individual increments have comparable sizes of order $n^{-1/2}$, it yields the first optimal $O(n^{-1/2})$ Gaussian approximation rate for fixed $p$ and $d$. Consequently, we also obtain the first optimal $O(n^{-1/2})$ $W_p$ Gaussian approximation rate for multivariate additive functionals of uniformly ergodic Markov chains.   Our analysis develops two techniques for addressing the interplay between higher-order Wasserstein distance and temporal dep
    
[^130]: Q-Edge：对称性约简的结构化极端依赖量子模拟

    Q-Edge: Symmetry-Reduced Quantum Simulation of Structured Extreme Dependence

    [https://arxiv.org/abs/2609.03706](https://arxiv.org/abs/2609.03706)

    Q-Edge提出了一种基于对称性的无损轨道空间表示方法，将30维问题中约160万个角度状态压缩为256个轨道状态，从而实现了结构化极端依赖系统的可扩展量子模拟与数字孪生。

    

    高维多元极端事件的模拟从根本上受限于依赖关系的组合复杂性，这种限制往往甚至超过极端观测数据稀缺所带来的制约。我们证明，对称性允许一种无损的轨道空间表示，它在保持结构化极端依赖的同时，用一个紧凑的对称类集合取代了指数级庞大的依赖空间。基于这一原理，我们开发了Q-Edge（量子极端依赖引擎），这是一个直接在轨道空间中运行的对称性约简量子框架，能够实现结构化极端系统的可扩展模拟与数字孪生。通过将对称性转移到数据表示中而非量子电路中，Q-Edge使不受约束的量子生成模型能够利用大幅缩减的状态空间。对于一个30维问题，约160万个角度状态坍缩为256个轨道状态，从而显著降低了所需的量子表示规模。

    arXiv:2609.03706v1 Announce Type: cross  Abstract: High-dimensional simulation of multivariate extremes is fundamentally limited by the combinatorial complexity of dependence, often more than by the scarcity of extreme observations. We show that symmetry admits a lossless orbit-space representation that preserves structured extreme dependence while replacing an exponentially large dependence space with a compact set of symmetry classes. Based on this principle, we develop Q-Edge (Quantum Extreme Dependence Engine), a symmetry-reduced quantum framework that operates directly in orbit space, enabling scalable simulation and digital twins of structured extreme systems. By transferring symmetry into the data representation rather than the quantum circuit, Q-Edge allows unconstrained quantum generative models to exploit dramatically reduced state spaces. For a 30-dimensional problem, approximately 1.6 million angular states collapse to 256 orbit states, reducing the required quantum represe
    
[^131]: SIGMA：对称感知、智能、几何、多目标自适应控制，用于稳健、可靠的交通管理

    SIGMA: Symmetry-aware, Intelligent, Geometric, Multi-objective Adaptive Control for Robust, Dependable Traffic Management

    [https://arxiv.org/abs/2608.18263](https://arxiv.org/abs/2608.18263)

    本文提出了SIGMA框架，通过结合大型语言模型进行动态目标调整和旋转增强实现方向不变学习，从而在交通信号控制中实现多目标自适应和跨场景泛化。

    

    交通信号控制是一个复杂的序列决策问题，需要实时适应并在吞吐量、延迟公平性、信号稳定性和紧急车辆优先权之间进行权衡。现有的强化学习方法通常固定目标，忽略动态优先级变化，并且难以在几何相似的交叉口之间泛化。我们提出了SIGMA（对称感知、智能、几何、多目标自适应交通控制），这是一个由大型语言模型（LLM）增强的强化学习框架，用于自适应目标调整和方向不变学习。SIGMA将自然语言紧急命令转换为多目标actor-critic控制器的优先级向量，避免了手动奖励工程。旋转增强提高了在四向交叉口的可转移性，而离线到在线学习确保了稳定的初始化和对交通变化的逐步适应。我们定义了覆盖可靠性的属性。

    arXiv:2608.18263v1 Announce Type: new  Abstract: Traffic signal control is a complex sequential decision-making problem requiring real-time adaptation and trade-offs among throughput, delay fairness, signal stability, and emergency vehicle priority. Existing RL methods often fix objectives, ignore dynamic priority changes, and fail to generalize across geometrically similar intersections.We propose SIGMA (Symmetry-aware, Intelligent, Geometric, Multi-objective Adaptive traffic control), an RL framework enhanced with a large language model (LLM) for adaptive objective tuning and orientation-invariant learning. SIGMA converts natural-language emergency commands into priority vectors for a multi-objective actor-critic controller, avoiding manual reward engineering. Rotational augmentation improves transferability across four-way intersections, while offline-to-online learning ensures stable initialization and gradual adaptation to changing traffic.We define reliability properties covering
    
[^132]: 去偏得分检验：面向半参数假设的“搜寻-检验”方法

    The Debiased Score Test: Hunt-and-test for Semiparametric Hypotheses

    [https://arxiv.org/abs/2607.28861](https://arxiv.org/abs/2607.28861)

    本文提出一种“搜寻-检验”策略的去偏得分检验方法：用一部分数据借助机器学习搜寻得分的最优方向，用另一部分数据检验得分是否为零，并通过加权最小二乘投影进行去偏校正，从而实现对半参数回归模型的拟合优度检验及观察性研究中效应修饰因子的检测。

    

    参数得分检验通过对数似然函数的导数来评估假设，其在原假设成立时期望为零。当感兴趣的参数是被识别为风险最小化子的回归函数时，我们将这一思想扩展到检验该函数是否属于某个给定的线性函数类。由此得到了针对常见半参数回归模型的拟合优度检验，包括广义可加模型和部分线性模型。经过适当表述，该框架还可以用于检测观察性研究中的效应修饰因子。我们提出了一种“搜寻-检验”策略，将数据分为两部分：在其中一部分数据上，先拟合原假设模型，再利用机器学习在经验得分中识别出一个有希望的方向；在另一部分数据上，我们检验得分在该方向上是否为零。为了考虑估计原假设模型时产生的误差，我们应用了基于加权最小二乘投影的去偏校正。我们建立了第I类错误……（摘要原文在此处截断）

    arXiv:2607.28861v2 Announce Type: replace-cross  Abstract: The parametric score test assesses a hypothesis through derivatives of the log-likelihood, whose expectation vanishes under the null. When the parameter of interest is a regression function identified as a risk minimiser, we extend this idea to test whether it belongs to a given linear function class. This yields goodness-of-fit tests for common semiparametric regression models, including generalised additive and partially linear models. Suitably formulated, the framework also detects effect modifiers in observational studies. We propose a hunt-and-test strategy that splits the data into two: on one part, after fitting the null model, machine learning is used to identify a promising direction in the empirical scores; on the other, we test whether the score vanishes in that direction. To account for error in estimating the null model, we apply a debiasing correction based on a weighted least squares projection. We establish Type
    
[^133]: 从各向异性高斯随机几何图中恢复潜在内积

    Recovery of latent inner products from an anisotropic Gaussian random geometric graph

    [https://arxiv.org/abs/2607.23723](https://arxiv.org/abs/2607.23723)

    该论文提出通过对双重中心化邻接矩阵进行秩-d谱近似，从各向异性高斯随机几何图中恢复潜在内积，其均方误差速率由协方差矩阵的稳定秩刻画，并在各向同性情形下达到最优水平。

    

    我们研究从具有各向异性高斯潜在点的随机几何图中恢复潜在内积的问题。更准确地说，对于独立同分布样本 $x_1, \dots, x_n \sim N(0,\Sigma)$，其中 $\Sigma \in \mathbb{R}^{d \times d}$，当且仅当 $\langle x_i, x_j \rangle \ge \zeta$（$\zeta$ 为阈值）时，图中存在边 $(i,j)$。我们假设阈值 $\zeta$ 的选取使得图的平均边密度为常数阶。为了应对由潜在点的各向异性所放大的不期望的度波动，我们考虑该图的双重中心化邻接矩阵，并利用双重中心化矩阵的秩-$d$ 谱近似来估计潜在内积。该估计量的均方误差速率由协方差矩阵 $\Sigma$ 的稳定秩刻画。值得注意的是，该估计速率在各向同性情形下达到了最先进的水平（原文摘要在此处截断）。

    arXiv:2607.23723v2 Announce Type: replace-cross  Abstract: We study the problem of recovering latent inner products from a random geometric graph with anisotropic Gaussian latent points. More precisely, for an i.i.d. sample $x_1, \dots, x_n \sim N(0,\Sigma)$ where $\Sigma \in \mathbb{R}^{d \times d}$, an edge $(i,j)$ is present in the graph if and only if $\langle x_i, x_j \rangle \ge \zeta$ for a threshold $\zeta$. We assume the threshold $\zeta$ to be chosen such that the average edge density of the graph is of constant order. To address the undesired degree fluctuations amplified by the anisotropy of the latent points, we consider the doubly centered adjacency matrix of the graph, and estimate the latent inner products using a rank-$d$ spectral approximation of the doubly centered matrix. The estimator obtains a mean squared error with a rate involving the stable rank of the covariance matrix $\Sigma$. Notably, the rate of estimation matches the state of the art for the isotropic ca
    
[^134]: 主动推理作为凸马尔可夫决策过程

    Active Inference as a Convex Markov Decision Process

    [https://arxiv.org/abs/2607.20152](https://arxiv.org/abs/2607.20152)

    该论文将主动推理的期望自由能最小化重新表述为凸马尔可夫决策过程，揭示了自然梯度沿时间反向传播的机制，并将其与表演性强化学习联系起来，从而将主动推理纳入现代强化学习与优化理论框架。

    

    主动推理（AIF）将适应性行为框架化为期望自由能（EFE）的最小化，在单一变分原理中结合了认知目标与实用目标。我们将主动推理框架化为策略优化问题，并证明对于闭环控制策略，期望自由能最小化可以表述为一个凸马尔可夫决策过程（MDP）。这一视角揭示了依赖策略的奖励预测误差是将期望自由能的自然梯度沿时间向后传递，而非沿层级结构向上传递。最后，我们证明了将世界模型学习与策略优化相耦合，使主动推理具有了表演性强化学习（performative reinforcement learning）的结构。这些结果共同将期望自由能最小化置于现代强化学习与优化理论之中，并为开发有原则的主动推理算法开辟了道路。

    arXiv:2607.20152v2 Announce Type: replace  Abstract: Active Inference (AIF) frames adaptive behavior as the minimization of expected free energy (EFE), combining epistemic and pragmatic objectives within a single variational principle. We frame AIF as policy optimization and show that, for closed-loop control policies, EFE minimization can be formulated as a convex Markov decision process (MDP). This perspective reveals that policy-dependent reward prediction errors transmit natural gradients of the expected free energy backwards in time rather than up a hierarchy. Finally, we show that coupling world-model learning with policy optimization gives active inference the structure of performative reinforcement learning. Together this places EFE minimization within modern reinforcement learning and optimization theory and opens a route toward principled algorithms for active inference.
    
[^135]: 面向张量数据的空间因子分析器混合模型

    Mixtures of spatial factor analyzers for tensor-variate data

    [https://arxiv.org/abs/2607.07887](https://arxiv.org/abs/2607.07887)

    本文提出空间因子分析器混合模型（MSFA），通过基于样条的空间衰减协方差结构使协方差参数随空间位置线性增长，并结合矩阵变量因子分析器进一步降维，有效实现高维张量空间数据的聚类分析。

    

    本文引入了空间因子分析器混合模型（MSFA），以应对高维空间数据聚类的挑战。所提出的框架利用给定的坐标系来实现一种灵活的、基于样条的空间衰减协方差结构。在这种方法中，协方差参数的数量随空间位置数量呈线性增长，而非二次增长。为了对非空间依赖性进行建模，采用矩阵变量因子分析器进行进一步的降维。参数估计通过期望最大化算法的一种变体并结合广义最小二乘估计器来实现。所提出的模型在张量数据分析的背景下进行了探讨，模拟研究以及三个应用——拉曼光谱、SpecTex高光谱纹理和Salinas高光谱场景——展示了该模型恢复和区分（数据结构）的能力。

    arXiv:2607.07887v2 Announce Type: replace-cross  Abstract: A mixture of spatial factor analyzers (MSFA) is introduced to address the challenges of clustering high-dimensional spatial data. The proposed framework utilizes the given coordinate system to implement a flexible, spline-based spatial decay covariance structure. In this approach, the number of covariance parameters increases linearly with the number of spatial locations, rather than increasing quadratically. To model non-spatial dependence, matrix variate factor analyzers are employed for further dimensionality reduction. Parameter estimation is conducted via a variant of the expectation-maximization algorithm combined with a generalized least squares estimator. The proposed models are explored in the context of tensor-variate data analysis, where simulation studies and three applications---Raman spectroscopy, SpecTex hyperspectral textures, and the Salinas hyperspectral scene---illustrate their ability to recover and distingu
    
[^136]: ManifoldFlow：具有可学习奇异谱的SPD松弛Stiefel层

    ManifoldFlow: SPD-Relaxed Stiefel Layers with Learnable Singular Spectrum

    [https://arxiv.org/abs/2607.04535](https://arxiv.org/abs/2607.04535)

    ManifoldFlow通过对固定谱Stiefel层进行最小松弛，将权重分解为 W = Q S^{1/2}，在保持基向量位于Stiefel流形上的同时学习有界的正定奇异谱，使特征值裁剪成为直接的奇异值控制机制，并在序列、表格和图像任务中优于固定谱Stiefel层。

    

    正交层和Stiefel层为神经网络权重提供了精确的谱控制，但它们也带来了很强的建模约束：所有被表示的奇异值都被固定为1。许多受益于正交归一基的场景仍然需要方向相关的衰减或放大。我们提出了ManifoldFlow，这是对固定谱Stiefel层的一种最小松弛方法，它在保持基向量位于Stiefel流形上的同时，通过 W = Q S^{1/2}（其中 Q^T Q = I 且 S 为正定矩阵）学习一个有界的正谱。由于 W^T W = S，S 的特征值恰好是所实现权重的奇异值的平方，这使得特征值裁剪成为一种直接的奇异值控制机制。在配对的序列、表格和图像实验中，可学习的SPD谱在Stiefel先验有效的报告设置中优于固定谱的Stiefel对应方法，其中在循环语言模型投影任务中增益最大（原文摘要在此处截断）。

    arXiv:2607.04535v2 Announce Type: replace-cross  Abstract: Orthogonal and Stiefel layers give neural weights exact spectral control, but they also impose a strong modeling constraint: all represented singular values are fixed at one. Many settings that benefit from an orthonormal basis still need direction-dependent attenuation or amplification. We introduce ManifoldFlow, a minimal relaxation of a fixed-spectrum Stiefel layer that keeps the basis on the Stiefel manifold while learning a bounded positive spectrum through W = Q S^{1/2}, with Q^T Q = I and S positive definite. Since W^T W = S, the eigenvalues of S are exactly the squared singular values of the realized weight, making eigenvalue clipping a direct singular-value control mechanism. Across paired sequence, tabular, and image experiments, the learnable SPD spectrum improves the fixed-spectrum Stiefel counterpart in the reported settings where the Stiefel prior is useful, with the largest gains in recurrent language-model proje
    
[^137]: LLM人格的双重本质：聚合倾向与框架依赖的几何结构

    The Dual Nature of LLM Persona: Aggregated Tendencies and Frame-Dependent Geometry

    [https://arxiv.org/abs/2607.02368](https://arxiv.org/abs/2607.02368)

    本论文发现LLM人格表达包含聚合倾向与框架依赖几何两个可分离成分，后者并非固有属性，而是编码聚合无法捕捉信息的协调模式。

    

    通过心理测量问卷对LLM人格的评估通常依赖于聚合分数，忽略了实例内部的关联结构。我们测试了这种几何结构是固有的还是依赖于框架的。我们使用IPIP-50响应构建实例内部相关矩阵，在GPT-4o模拟美国和华裔美国人角色时，通过操控问题顺序来分析SPD流形上的几何结构。我们发现人格表达包含两个可分离的组成部分：聚合特征（大五人格分数）在随机化下下降（21%），但对框架具有鲁棒性；几何特征（SPD流形）在框架错位下崩溃（下降42%），但在共享框架下显著恢复（至84%），超过了聚合特征（76%）。这种崩溃-恢复模式表明，人格几何并非固有属性，而是一种依赖于框架的协调模式，编码了聚合方法无法捕捉的信息。

    arXiv:2607.02368v2 Announce Type: replace-cross  Abstract: Evaluations of LLM personas via psychometric questionnaires typically rely on aggregate scores, discarding within-instance correlation structure. We test whether this geometric structure is intrinsic or frame-dependent. Constructing within-instance correlation matrices from IPIP-50 responses, we analyze geometry on SPD manifolds under manipulated question orderings in GPT-4o simulating American and Chinese-American personas. We find that persona expression comprises two dissociable components: aggregated features (Big Five scores) degrade under randomization (21% drop) but are frame-robust; geometric features (SPD manifold) collapse under frame misalignment (42% drop) but recover substantially (to 84%) under shared frames, surpassing aggregated features (76%). This collapse-recovery pattern reveals that persona geometry is not intrinsic but a frame-dependent coordination pattern encoding information invisible to aggregation. Ou
    
[^138]: 从微小训练集学习无采样变分DNN插件，结合不确定性估计精化分布外分割

    Learning a Sampling-Free Variational DNN Plugin from Tiny Training Sets to Refine OOD Segmentation With Uncertainty Estimation

    [https://arxiv.org/abs/2606.15837](https://arxiv.org/abs/2606.15837)

    提出VarDeepPCA——一种轻量级变分DNN插件框架，仅凭微小分布内训练集、通过对softmax映射的重新解释实现无采样的精确分布建模，无需目标域数据或大规模预训练即可精化OOD医学图像分割并估计不确定性。

    

    深度神经网络（DNN）由于扫描仪和采集协议的差异，经常无法泛化到分布外（OOD）医学图像。由于获取和标注新医学数据集的高昂成本，重新训练DNN模型以应对这些分布偏移往往不切实际。为解决这一问题，我们提出了VarDeepPCA，这是一种新颖的轻量级变分DNN框架，旨在通过利用内在几何先验来恢复/精化退化的分割图。与需要目标域数据或大规模预训练的现有方法不同，我们的VarDeepPCA仅使用小型分布内（ID）数据集即可显式学习有效解剖几何的分布。在理论上，我们新颖的变分学习框架通过对softmax映射的重新解释来隐式地进行精确分布建模，从而实现计算高效、无需采样的学习……

    arXiv:2606.15837v3 Announce Type: replace-cross  Abstract: Deep neural networks (DNNs) frequently fail to generalize to out-of-distribution (OOD) medical images because of variations in scanners and acquisition protocols. Retraining DNN models to address these distribution shifts is often impractical due to the high cost of acquiring and annotating new medical datasets. To address this, we introduce VarDeepPCA, a novel lightweight variational DNN framework designed to restore/refine degraded segmentation maps by leveraging intrinsic geometric priors. Unlike existing approaches that require target-domain data or extensive pre-training, our VarDeepPCA explicitly learns a distribution of valid anatomical geometries using only small in-distribution (ID) datasets. Theoretically, our novel variational learning framework leverages a reinterpretation of the softmax mapping to implicitly perform exact distribution modeling, thereby enabling computationally efficient, sampling-free learning and 
    
[^139]: 关于不完整U统计量中位数的有限样本集中性

    On Finite-sample Concentration of Median of Incomplete U-Statistics

    [https://arxiv.org/abs/2606.00661](https://arxiv.org/abs/2606.00661)

    本文证明了不完整U统计量中位数（MoIU）的有限样本浓度界，克服了此前仅能获得松散$O(n^{-1/4})$界的理论挑战，实现了更紧的收敛速率。

    

    中位数均值（MoM）是一种强大的技术，在理论上能在底层数据分布具有重尾特性（例如，仅假设具有前两阶有限矩）时，实现参数估计的接近亚高斯的有限样本速率。最近的研究将此技术推广到中位数随机化U统计量（MoRU）和中位数不完整U统计量（MoIU），用于估计重尾成对核的期望。在\citet{pmlr-v97-clemencon19a}中，已证明MoRU的浓度速率随样本量按$O(n^{-1/2})$缩放。然而，尽管后者具有计算优势，MoIU的有限样本界分析仍是一个重大的理论挑战。正如作者所指出的，直接应用McDiarmid不等式会产生$O(n^{-1/4})$阶的松散界。在本工作中，我们证明了MoIU估计的有限样本浓度界。

    arXiv:2606.00661v2 Announce Type: replace-cross  Abstract: Median-of-means (MoM) is a powerful technique that theoretically enables near sub-Gaussian finite-sample rate for parameter estimation when the underlying data distribution is heavy-tailed (e.g., assumed to have only two first finite moments). A recent work has extrapolated this technique to median-of-\textit{randomized}-U-Statistics (MoRU) and median-of-\textit{incomplete}-U-Statistics (MoIU) for estimating expectations of heavy-tailed pairwise kernels. In \citet{pmlr-v97-clemencon19a}, a concentration rate that scales like $O(n^{-1/2})$ with sample size has been proven for MoRU. However, despite the computational advantage of the latter, the analysis of finite-sample bound for MoIU remains a significant theoretical challenge. As noted by the authors, a straightforward application of McDiarmid's inequality yields a loose bound of order $O(n^{-1/4})$. In this work, we prove a finite-sample concentration bound for the MoIU estim
    
[^140]: Hoeffding概念瓶颈模型及其在航拍图像中的应用

    Hoeffding Concept Bottleneck Models with Applications to Overhead Images

    [https://arxiv.org/abs/2606.00082](https://arxiv.org/abs/2606.00082)

    该论文提出了基于Hoeffding函数分解的概念瓶颈模型（HCBM），用非线性和稀疏的概念分数聚合取代传统线性聚合，在提升可解释性的同时避免信息泄露，并成功应用于航拍图像分类。

    

    深度学习算法的可解释性对于涉及高风险决策的计算机视觉应用至关重要。概念瓶颈模型（CBM）近来在分类问题上展现出良好的性能，能够基于高层概念瓶颈提供可解释且准确的预测。现有的CBM方法依赖于概念分数的线性聚合来计算预测结果。然而，这种线性方法通常需要使用大量概念，这削弱了可解释性并容易导致信息泄露。一般而言，概念与输出逻辑值之间的潜在关系并非线性。因此，我们提出了Hoeffding概念瓶颈模型（HCBM），该模型基于梯度提升树的Hoeffding函数分解，提供概念分数的非线性和稀疏聚合，并利用素蕴含项生成紧凑的预测。HCBM具有很强的理论性质

    arXiv:2606.00082v2 Announce Type: replace-cross  Abstract: Explainability of deep learning algorithms is critical for computer-vision applications with high-stakes decisions. Concept bottleneck models (CBM) have recently shown promising performance to provide explainable and accurate predictions for classification problems, based on a bottleneck of high-level concepts. Existing CBM methods rely on a linear aggregation of the concept scores to compute predictions. However, a large number of concepts is often used in this linear approach, which undermines explainability and favors information leakage. In general, the underlying relation between concepts and output logits is not linear. Therefore, we introduce Hoeffding Concept Bottleneck Models (HCBM), which build on the Hoeffding functional decomposition of gradient-boosted trees to provide non-linear and sparse aggregations of concept scores, and generate compact predictions using prime implicants. HCBM have strong theoretical properti
    
[^141]: 推理轨迹前缀的保形认证

    Conformal Certification of Reasoning Trace Prefixes

    [https://arxiv.org/abs/2605.30085](https://arxiv.org/abs/2605.30085)

    CROP 是一种与验证器无关的保形校准方法，可为语言模型推理轨迹的连续前缀提供统计错误率保证——返回可安全保留的最长干净前缀，并将未认证部分交由下游审查修复。

    

    语言模型的推理轨迹很少是非黑即白的；它们通常在出现关键错误之前包含有效的中间步骤。现有的不确定性量化方法通常只对最终答案或完整响应进行认证，无法为序列轨迹中可安全保留的比例提供统计保证。为解决这一问题，我们提出了 CROP（Conformal Reasoning Output Prefixes，保形推理输出前缀），这是一种与验证器无关的干净前缀认证校准程序。给定任意步骤级风险代理指标，CROP 会选择一个经过校准的阈值，并返回步骤风险代理指标始终低于该阈值的最长连续前缀，同时将未认证的后缀交给下游进行审查或修复。在可交换性假设下，CROP 严格地控制了返回前缀包含标注错误的边际概率。在六个带有过程标注的推理数据集上，我们证明了标准的步骤级指标……

    arXiv:2605.30085v2 Announce Type: replace-cross  Abstract: Language model reasoning traces are rarely all-or-nothing; they frequently contain valid intermediate steps before a critical error occurs. Existing uncertainty quantification methods typically certify final answers or entire responses, failing to provide statistical guarantees for the proportion of a sequential trace that can be safely retained. To address this, we introduce CROP (Conformal Reasoning Output Prefixes), a verifier-agnostic calibration procedure for clean-prefix certification. Given any step-level risk proxy, CROP selects a calibrated threshold and returns the longest contiguous prefix whose step risk proxies remain below it, routing the uncertified suffix for downstream review or repair. Assuming exchangeability, CROP rigorously controls the marginal probability that the returned prefix contains an annotated error. Across six process-labeled reasoning datasets, we demonstrate that standard step-level metrics suc
    
[^142]: 不确定性估计器与大语言模型中幻觉发生的对齐程度如何？

    How Do Uncertainty Estimators Align with Hallucination Occurrence in LLMs?

    [https://arxiv.org/abs/2605.27016](https://arxiv.org/abs/2605.27016)

    该论文系统评估了46种不确定性估计器在内在与外在幻觉场景下对大语言模型幻觉的检测能力，发现不确定性与幻觉的关联普遍较弱且因设置和模型而异，挑战了将不确定性直接作为幻觉信号的做法。

    

    大语言模型（LLM）会产生幻觉，但模型的不确定性估计器能否可靠地揭示幻觉何时发生？尽管不确定性估计器越来越多地被用于识别不可靠的生成内容，但它们与幻觉之间的关系仍然缺乏深入研究，且往往被视为理所当然。我们在互补的幻觉设置下系统评估了46种不确定性估计器，涵盖内在幻觉（即输出得不到所提供上下文的支持）和外在幻觉（即陈述得不到模型预期知识的支持）。在四个基准测试和多个大语言模型上的结果表明，不确定性与幻觉之间的关联差异很大且往往较弱，具体取决于幻觉设置和被评估的模型。这些结果对将不确定性作为幻觉直接信号的做法提出了挑战，并阐明了不确定性在何种情况下能够提供有用信息。

    arXiv:2605.27016v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) can hallucinate, but do model uncertainty estimators reliably reveal when they do? Although uncertainty estimators are increasingly used to identify unreliable generations, their relationship with hallucination remains poorly understood and is often taken for granted. We systematically evaluate 46 uncertainty estimators across complementary hallucination settings, covering both intrinsic hallucinations, where outputs are unsupported by the provided context, and extrinsic hallucinations, where claims are unsupported by expected model knowledge. Across four benchmarks and multiple LLMs, we find that the association between uncertainty and hallucination varies widely and is often weak, depending on the hallucination setting and the model under evaluation. These results challenge the use of uncertainty as a direct signal of hallucination and clarify when it provides useful information.
    
[^143]: 基于镜像遗忘与噪声一致偏度的扩散模型训练数据归因

    Training data attribution in diffusion models via mirrored unlearning and noise-consistent skew

    [https://arxiv.org/abs/2605.17938](https://arxiv.org/abs/2605.17938)

    本文提出MUCS方法，通过有界镜像梯度上升微调辅助模型并利用一致噪声样本测量归一化偏度，为扩散模型实现了更可靠、更鲁棒的训练数据归因，在三个数据集上大幅超越现有方法。

    

    训练数据归因（TDA）应当能够实现生成模型的可解释性，并促进多种相关的下游任务。然而，当前的TDA方法缺乏可靠性和鲁棒性，阻碍了其在现实场景中的应用。在本文中，我们朝着为扩散模型实现更可靠、更鲁棒的TDA迈出了决定性的一步。我们提出使用镜像遗忘与噪声一致偏度（MUCS）来执行TDA。其核心思想是通过有界镜像梯度上升微调第二个模型，并使用一致的噪声样本来测量该模型相对于原始模型的归一化偏度。我们证明，尽管在概念上简单且通用，MUCS在三个不同的数据集上系统性地大幅超越现有方法。此外，我们还研究了核心设计选择对最终性能的影响，并分析了有影响力实例之间重叠方面的新颖特性。

    arXiv:2605.17938v2 Announce Type: replace-cross  Abstract: Training data attribution (TDA) should enable generative model interpretability and foster a variety of related downstream tasks. Nonetheless, current TDA approaches lack reliability and robustness, preventing their adoption in real-world setups. In this paper, we take a decisive step towards more reliable and robust TDA for diffusion models. We propose to perform TDA with mirrored unlearning and noise-consistent skew (MUCS). The idea is to fine-tune a second model with bounded mirrored gradient ascent, and to measure the normalized skew of this model with respect to the original one using consistent noise samples. We show that, while being conceptually simple and generic, MUCS systematically outperforms existing methods on three different datasets by a large margin. We additionally study the effect that core design choices have on final performance, and analyze novel aspects regarding the overlap of influential instances acros
    
[^144]: 通过最小条件熵耦合从完全非配对数据中学习

    Learning from Totally Unpaired Data via Minimum Conditional Entropy Coupling

    [https://arxiv.org/abs/2605.16229](https://arxiv.org/abs/2605.16229)

    提出最小条件熵耦合（MCEC）新方法，仅利用完全非配对的数据集，通过最小化条件熵来耦合边缘分布从而学习类别预测器，并针对该 NP-难问题给出了多项式时间复杂度的算法。

    

    考虑一个“无监督分类”的设置，其目标是利用类别特征变量 $Y_1,\ldots,Y_m$ 学习类别变量 $X$ 的预测器，但我们只能在 $m+1$ 个完全非配对的数据集上训练该预测器，其中每个数据集仅由变量 $X,Y_1,\ldots,Y_m$ 中某一个变量的无序样本组成。尽管该问题高度欠定，我们证明，在存在具有较小交叉熵损失的预测器的假设下，可以通过以最小化条件熵 $H(X|Y_1,\ldots,Y_m)$ 的方式耦合 $X,Y_1,\ldots,Y_m$ 的边缘分布，从而找到一个好的预测器。我们将这一问题称为最小条件熵耦合（MCEC），它是被广泛研究的最小熵耦合问题（即 $m=1$ 的情形）的推广。尽管 MCEC 问题是 NP-难的，我们给出了一个时间复杂度关于 $N$ 为多项式的算法。

    arXiv:2605.16229v2 Announce Type: replace-cross  Abstract: Consider an "unsupervised classification" setting, where the goal is to learn a predictor of the categorical class variable $X$ using the categorical feature variables $Y_1,\ldots,Y_m$, but we can only train the predictor on $m+1$ totally unpaired datasets, where each dataset only consists of unordered samples of one of the variables $X,Y_{1},\ldots,Y_{m}$. Although this problem is highly underdetermined, we demonstrate that, under the assumption that a predictor with small cross entropy loss exists, we can find a good predictor by coupling the marginal distributions of $X,Y_1,\ldots,Y_m$ in a manner that minimizes the conditional entropy $H(X|Y_1,\ldots,Y_m)$. This problem, which we call minimum conditional entropy coupling (MCEC), is a generalization of the widely studied minimum entropy coupling problem (where $m=1$). Although the MCEC problem is NP-hard, we give an algorithm with a time complexity that is polynomial in $N$,
    
[^145]: α-TCAV：一个基于概念激活向量测试的统一框架

    $\alpha$-TCAV: A Unified Framework for Testing with Concept Activation Vectors

    [https://arxiv.org/abs/2605.15688](https://arxiv.org/abs/2605.15688)

    提出参数化统一框架α-TCAV，用平滑函数替代标准TCAV中导致不稳定的非连续指示函数，在极限情形下恢复TCAV，单个CAV即可达到Multi-TCAV的均值且方差更小，并提供概念正向影响模型输出的校准概率。

    

    概念激活向量（CAV）及基于CAV的测试方法（TCAV）是深度学习中基于概念可解释性的基础工具，但其实际效用受限于概念样本与非概念样本在不同采样下的变异性。我们分析了CAV的这种随机特性，并指出标准TCAV分数存在一个根本缺陷：由于使用不连续的指示函数而导致的不稳定性。为解决这一问题，我们提出了α-TCAV，一个参数化的统一框架，用平滑函数替代不连续的指示函数，并在极限情形下可恢复为标准TCAV。在现实假设下，我们刻画了分数的分布并推导出有原则的参数选择方式：其一是仅使用单个CAV即可匹配其平均对应方法Multi-TCAV的均值，且方差不超过后者；其二是提供一个经过校准的概率，用于表示概念对模型输出产生正向影响的可能性。

    arXiv:2605.15688v2 Announce Type: replace  Abstract: Concept Activation Vectors (CAV) and the Testing with CAVs (TCAV) method are fundamental tools for concept-based explainability in deep learning, yet their practical utility is limited by the variability across different samples of concept and non-concept examples. We analyze this stochastic nature of CAVs, and identify a fundamental flaw in the standard TCAV score, its instability by using a discontinuous indicator function. To address this, we introduce $\alpha$-TCAV, a parameterized unified framework that replaces the discontinuous indicator with a smooth function, which recovers TCAV as a limiting case. Under realistic assumptions, we characterize the score distributions and derive principled parameter choices: one matches the mean of its averaged counterpart, Multi-TCAV, with no larger variance using a single CAV; the other provides a calibrated probability that the concept positively influences the model output. Our analysis an
    
[^146]: 带宽预算约束下的联邦语言模型：蒸馏速率与共形覆盖

    Federated Language Models Under Bandwidth Budgets: Distillation Rates and Conformal Coverage

    [https://arxiv.org/abs/2605.09986](https://arxiv.org/abs/2605.09986)

    该论文提出了两种在带宽受限、数据不出域场景下以小而可预测的代价压缩每条通信消息的联邦学习方法——FPLD（联邦探测-逻辑值蒸馏）用于跨站点联合训练共享语言模型，FC-RAG（联邦共形检索增强生成）用于跨站点协同问答。

    

    医院、公司和研究联盟希望语言模型能够从必须保留在本地（数据不出域）的数据中学习并基于这些数据回答问题，且需要通过带宽有限的链路进行通信，例如偏远诊所和船只所使用的卫星链路。对各站点模型参数进行平均的方法会使每条消息随模型规模增大而增长，并且要求所有站点使用统一的模型架构；而交换预测结果的方法可以同时避免这两个限制，但据我们所知，目前尚没有方法能够限定压缩这些预测所带来的代价。我们提出了两种方法，能够以较小且可预测的代价压缩每条消息。联邦探测-逻辑值蒸馏（Federated Probe-Logit Distillation, FPLD）用于训练一个共享语言模型：每个站点在共享的无标注探测样本上发送压缩后的逻辑值，由中心节点从其平均值中蒸馏出一个模型。联邦共形检索增强生成（Federated Conformal Retrieval-Augmented Generation, FC-RAG）用于回答问题：每个站点使用自己的文档对候选答案进行评分并发送压缩后的分数，由中心节点返回一个综合

    arXiv:2605.09986v3 Announce Type: replace  Abstract: Hospitals, companies, and research consortia want language models that learn from, and answer questions over, data that must stay where it is held, over links with limited bandwidth, such as the satellite links of remote clinics and ships. Averaging the sites' model parameters makes each message grow with the model and requires one shared architecture; methods that exchange predictions avoid both limits, but, to our knowledge, none bounds what compressing those predictions costs. We introduce two methods that compress every message at a small and predictable cost. Federated Probe-Logit Distillation (FPLD) trains a shared language model: each site sends compressed logits on shared unlabeled probes, and a hub distills one model from their average. Federated Conformal Retrieval-Augmented Generation (FC-RAG) answers questions: each site scores candidate answers using its own documents and sends compressed scores, and the hub returns a co
    
[^147]: 何时裁剪有助于保形预测？校准污染下的保留分布诊断

    When Does Trimming Help Conformal Prediction? A Retained-Law Diagnostic under Calibration Contamination

    [https://arxiv.org/abs/2605.06204](https://arxiv.org/abs/2605.06204)

    本文证明保形预测中裁剪操作对覆盖率的影响由裁剪后的保留分布决定而非污染水平，并给出裁剪有效的精确条件：异常分数需能区分保留概率同时在干净数据上保持分数中性。

    

    裁剪可疑的校准点是应对保形预测中污染问题的常见做法。然而，其对干净目标覆盖率的影响取决于裁剪所诱导的保留分布，而非仅仅取决于污染水平本身。我们将固定阈值裁剪分析为一种条件化操作而非净化操作：它用保留分布替换受污染的校准分布，从而将干净目标覆盖率问题简化为一个一维分数-累积分布函数（CDF）转移问题，并给出精确的有限样本恒等式。转移间隙的分量级界限提供了一个总体层面的诊断方法。该诊断将干净侧的协方差成本与保留污染成本分离开来，后者由脏到干净的保留比例所控制。当异常分数能够区分保留概率、同时在干净总体上保持分数中性时，裁剪才有帮助；否则，裁剪无法通过保留混合系数实质性地减少污染。我们还给出了……

    arXiv:2605.06204v2 Announce Type: replace  Abstract: Trimming suspicious calibration points is a common response to contamination in conformal prediction. Its effect on clean-target coverage, however, is governed by the retained law induced by trimming, not by the contamination level alone. We analyse fixed-threshold trimming as conditioning rather than purification. It replaces the contaminated calibration law with a retained law, reducing clean-target coverage to a one-dimensional score-CDF transfer problem with an exact finite-sample identity. A componentwise bound on the transfer gap gives a population-level diagnostic. This separates a clean-side covariance cost from a retained-contamination cost, governed by the dirty-to-clean retention ratio. Trimming helps when the anomaly score separates retention probabilities while remaining score-neutral on the clean population. Otherwise, it cannot substantially reduce contamination through the retained mixture coefficient. We also give fi
    
[^148]: 在有界数据上下注时当Cover遇上Robbins：ln n 遗憾与几乎必然的 ln ln n 遗憾

    Cover meets Robbins while Betting on Bounded Data: $\ln n$ Regret and Almost Sure $\ln\ln n$ Regret

    [https://arxiv.org/abs/2604.20172](https://arxiv.org/abs/2604.20172)

    本文提出一种融合 Cover 与 Robbins 思想的显式完全支撑混合下注策略，在保持 O(ln n) 最坏情况遗憾的同时，在内在方差过程发散的路径上实现几乎必然的 ln ln Vₙ 量级且首项常数不可改进的更精细遗憾界。

    

    考虑对一列取值于 [0,1] 的观测序列进行下注，当数据具有条件均值 m₀ ∈ (0,1) 时所采用的注注是公平的。Cover 的通用投资组合算法相比事后最优的常数下注策略可获得 O(ln n) 的最坏情况遗憾，且该对数阶在面对对抗性数据时已是不可改进的。我们构造了一个显式的、具有完全支撑集的 Robbins 混合策略，融合了 Robbins 与 Cover 两者的思想。该策略在每条路径上都保持 O(ln n) 的遗憾，同时对于混合财富有界且 Vₙ → ∞ 的每条路径，最终满足 Rₙ ≤ (1+o(1)) ln ln Vₙ，其中 Vₙ 是一个内在方差过程。在独立同分布的备择假设下，该策略还达到了最优的渐近财富增长率。在条件均值假设及 Vₙ 几乎必然发散的条件下，更精细的遗憾界几乎必然成立，且其首项常数 1 是不可改进的。

    arXiv:2604.20172v3 Announce Type: replace-cross  Abstract: Consider betting against a sequence of $[0,1]$-valued observations using bets that are fair if the data have a conditional mean $m_0 \in (0,1)$. Cover's universal portfolio algorithm achieves a worst-case regret of $O(\ln n)$ compared to the best constant bet in hindsight, and this logarithmic order is unimprovable against adversarial data. We construct an explicit full-support Robbins mixture that combines insights from Robbins and Cover. It retains $O(\ln n)$ regret on every path while also eventually satisfying $R_n\le(1+o(1))\ln\ln V_n$ on every path with bounded mixture wealth and $V_n\to\infty$, where $V_n$ is an intrinsic variance process. It also attains the optimal asymptotic wealth growth rate under iid alternatives. Under the conditional-mean claim and almost-sure divergence of $V_n$, the sharper regret bound holds almost surely, and its leading constant $1$ is unimprovable. We also give explicit finite-time, path-wi
    
[^149]: MosaicMRI：一个多样化的原始肌肉骨骼MRI数据集与基准

    MosaicMRI: A Diverse Dataset and Benchmark for Raw Musculoskeletal MRI

    [https://arxiv.org/abs/2604.11762](https://arxiv.org/abs/2604.11762)

    本文发布了迄今最大的开源原始肌肉骨骼MRI数据集MosaicMRI，包含3,150个体数据和89,830个切片，涵盖多种采集方向、成像对比度和解剖部位，为训练和评估基于机器学习的MRI方法提供了多样化基准。

    

    深度学习支撑着MRI中的广泛应用，包括图像重建、伪影去除和分割。然而，该领域的进展主要由专注于脑部和膝关节成像的公开数据集推动，这塑造了模型的训练和评估方式。因此，针对这些模型在不同解剖部位场景下可靠性的细致研究仍然有限。在这项工作中，我们推出了MosaicMRI，这是一个大型且多样化的全采样原始肌肉骨骼（MSK）MR测量数据集，专为训练和评估基于机器学习的方法而设计。MosaicMRI是迄今为止最大的开源原始MSK MRI数据集，包含3,150个体数据和89,830个切片，其中包括一个3,096体的1.5T核心数据集和一个54体的0.55T低场数据集。该数据集在体数据采集方向（如轴位、矢状位）、成像对比度（如PD、T1、T2）、解剖部位（如脊柱、膝关节、髋部、脚踝等）方面提供了显著的多样性。

    arXiv:2604.11762v2 Announce Type: replace-cross  Abstract: Deep learning underpins a wide range of applications in MRI, including reconstruction, artifact removal, and segmentation. However, progress has been driven largely by public datasets focused on brain and knee imaging, shaping how models are trained and evaluated. As a result, careful studies of the reliability of these models across diverse anatomical settings remain limited. In this work, we introduce MosaicMRI, a large and diverse collection of fully sampled raw musculoskeletal (MSK) MR measurements designed for training and evaluating machine-learning--based methods. MosaicMRI is the largest open-source raw MSK MRI dataset to date, comprising 3,150 volumes and 89,830 slices, including a 3,096-volume 1.5~T core dataset and a 54-volume 0.55~T low-field set. The dataset offers substantial diversity in volume orientation (e.g., axial, sagittal), imaging contrasts (e.g., PD, T1, T2), anatomies (e.g., spine, knee, hip, ankle, and
    
[^150]: 基于生成流网络的最短路径学习

    Learning Shortest Paths with Generative Flow Networks

    [https://arxiv.org/abs/2603.01786](https://arxiv.org/abs/2603.01786)

    该论文提出了基于生成流网络（GFlowNets）的最短路径学习新框架，证明了最小化总流量的GFlowNet策略必沿最短路径遍历，从而将最短路径寻找问题转化为最小总流量问题，并在置换环境寻路和魔方求解任务中取得了与专用最先进方法相当的效果。

    

    arXiv:2603.01786v2 公告类型： replace-cross 摘要：在本文中，我们提出了一种利用生成流网络（GFlowNets）在图中寻找最短路径的新型学习框架。首先，我们研究了GFlowNets在含环（非无环）环境中与最短路径相关的理论性质。我们证明，当总流量被最小化时，前向和后向策略将仅在初始状态与终止状态之间的最短路径上遍历环境图。基于这一结果，我们将任意无权图中的最短路径寻找问题转化为寻找具有最小总流量的非无环GFlowNet问题，并在实践中通过带流正则化的训练对其进行近似求解。我们通过实验展示了该方法在置换环境寻路任务以及求解不同规模魔方问题上的性能。对于魔方求解问题，我们的方法与专门为此设计的最先进机器学习方法相比取得了具有竞争力的结果。

    arXiv:2603.01786v2 Announce Type: replace-cross  Abstract: In this paper, we present a novel learning framework for finding shortest paths in graphs utilizing Generative Flow Networks (GFlowNets). First, we examine theoretical properties of GFlowNets in non-acyclic environments in relation to shortest paths. We prove that, if the total flow is minimized, forward and backward policies traverse the environment graph exclusively along shortest paths between the initial and terminal states. Building on this result, we reduce shortest-path finding in an arbitrary unweighted graph to finding a non-acyclic GFlowNet with minimal total flow, and approximate it in practice by training with flow regularization. We experimentally demonstrate the performance of our method in pathfinding in permutation environments and in solving Rubik's Cubes of various sizes. For the latter problems, our approach shows competitive results with state-of-the-art machine learning approaches designed specifically for 
    
[^151]: 基于个体敏感性界的可认证逐实例遗忘

    Certified Per-Instance Unlearning Using Individual Sensitivity Bounds

    [https://arxiv.org/abs/2602.15602](https://arxiv.org/abs/2602.15602)

    提出可认证逐实例遗忘框架，利用单个训练点的个体敏感性界自适应调整噪声量，避免最坏情况校准带来的过度保守性，并为确定性训练及岭回归的噪声梯度下降推导出显式遗忘证书。

    

    认证式机器遗忘可以通过注入噪声来实现，并通过将噪声校准至最坏情况敏感性来获得差分隐私保证。然而，单个训练点的影响具有高度异构性，使得这种统一校准对于大多数删除请求而言过于保守。我们研究了可认证的逐实例遗忘，这是可认证删除的一种数据依赖概念，其中遗忘机制及其证书能够适应被删除的特定数据点。我们的框架将针对特定数据点的敏感性信息转化为正式的遗忘保证，使噪声量能够反映删除请求的实际难度而非最坏情况。当训练过程是确定性的，我们推导出了显式的逐实例遗忘证书。对于噪声梯度下降，在岭回归的情形下，通过高概率敏感性界也获得了类似的证书。

    arXiv:2602.15602v3 Announce Type: replace-cross  Abstract: Certified machine unlearning can be achieved through noise injection, with differential privacy guarantees obtained by calibrating the noise to worst-case sensitivity. Yet, the influence of individual training points is highly heterogeneous, making such uniform calibration unnecessarily conservative for most deletion requests. We consider certified per-instance unlearning, a data-dependent notion of certified deletion in which the unlearning mechanism and its certificate can adapt to the point being removed. Our framework turns point-specific sensitivity information into formal unlearning guarantees, allowing the amount of noise to reflect the difficulty of the deletion request rather than the worst case. When training is deterministic, we derive explicit per-instance unlearning certificates. For noisy gradient descent, in the case of ridge regression, similar certificates are obtained through high-probability sensitivity bound
    
[^152]: DCD：基于分解的自相关与非平稳时间序列数据因果发现方法

    DCD: Decomposition-based Causal Discovery from Autocorrelated and Non-Stationary Temporal Data

    [https://arxiv.org/abs/2602.01433](https://arxiv.org/abs/2602.01433)

    提出了一种基于分解的因果发现框架DCD，通过将时间序列分解为趋势、季节和残差分量并分别进行针对性因果分析，有效减少了非平稳和自相关数据中的虚假因果关系，并构建出统一的多尺度因果结构。

    

    金融、气候科学和医疗保健等领域的多变量时间序列通常表现出长期趋势、季节性模式和短期波动，这使得在非平稳性和自相关性条件下的因果推断变得复杂。现有的因果发现方法通常直接在原始观测数据上运行，容易产生虚假边和错误归因的时间依赖关系。我们提出了一种基于分解的因果发现框架，将每个时间序列分解为趋势、季节和残差三个分量，并对各分量进行针对性的因果分析。趋势分量使用平稳性检验进行评估，季节分量使用基于核函数的依赖性度量方法，残差分量则使用基于约束的因果发现方法。由此得到的分量级图被整合为一个统一的多尺度因果结构。这种方法能够分离长程和短程因果效应，减少虚假边……

    arXiv:2602.01433v3 Announce Type: replace-cross  Abstract: Multivariate time series in domains such as finance, climate science, and healthcare often exhibit long-term trends, seasonal patterns, and short-term fluctuations, complicating causal inference under non-stationarity and autocorrelation. Existing causal discovery methods typically operate on raw observations, making them vulnerable to spurious edges and misattributed temporal dependencies. We introduce a decomposition-based causal discovery framework that separates each time series into trend, seasonal, and residual components and performs component-specific causal analysis. Trend components are assessed using stationarity tests, seasonal components using kernel-based dependence measures, and residual components using constraint-based causal discovery. The resulting component-level graphs are integrated into a unified multi-scale causal structure. This approach isolates long- and short-range causal effects, reduces spurious as
    
[^153]: 增长表示与环境维度下的低秩加稀疏矩阵迁移学习

    Low-Rank Plus Sparse Matrix Transfer Learning under Growing Representations and Ambient Dimensions

    [https://arxiv.org/abs/2601.21873](https://arxiv.org/abs/2601.21873)

    该论文提出了一个在环境维度与潜在表示同时增长情形下的低秩加稀疏矩阵迁移学习框架，通过锚定交替投影估计器在保留已迁移子空间的同时仅估计低维创新与稀疏修改，实现了严格更优的误差率。

    

    学习系统往往会随时间扩展其环境特征或潜在表示，将早期表示嵌入到更大的空间中，而仅引入有限的新潜在结构。我们研究了在环境维度与内在表示同时增长情况下的结构化矩阵估计迁移学习问题，其中估计良好的源任务被嵌入为更高维目标任务的一个子空间。我们提出了一个通用的迁移框架，其中目标参数分解为嵌入的源分量、低维低秩创新分量以及稀疏编辑分量，并开发了一种锚定交替投影估计器，该估计器在保留已迁移子空间的同时，仅估计低维创新和稀疏修改。我们建立了确定性的误差界，将目标噪声、表示增长和源估计误差的贡献分离开来，当秩与稀疏度增量受限时可获得严格更优的误差率。

    arXiv:2601.21873v2 Announce Type: replace  Abstract: Learning systems often expand their ambient features or latent representations over time, embedding earlier representations into larger spaces with limited new latent structure. We study transfer learning for structured matrix estimation under simultaneous growth of the ambient dimension and the intrinsic representation, where a well-estimated source task is embedded as a subspace of a higher-dimensional target task.   We propose a general transfer framework in which the target parameter decomposes into an embedded source component, low-dimensional low-rank innovations, and sparse edits, and develop an anchored alternating projection estimator that preserves transferred subspaces while estimating only low-dimensional innovations and sparse modifications. We establish deterministic error bounds that separate target noise, representation growth, and source estimation error, yielding strictly improved rates when rank and sparsity increm
    
[^154]: 信号源于结构：在生成流网络中利用次模上界

    Signal from Structure: Exploiting Submodular Upper Bounds in Generative Flow Networks

    [https://arxiv.org/abs/2601.21061](https://arxiv.org/abs/2601.21061)

    本文提出SUBo-GFN，通过利用奖励函数的次模性获得未观察组合对象的奖励上界，并据此训练生成流网络，在相同奖励查询次数下比经典方法产生数量级更多的学习信号。

    

    生成流网络（GFlowNets；GFNs）是一类生成模型，它学习按与其先验未知价值（即奖励）成正比的方式采样组合对象。我们关注奖励具有特定且可利用结构的情况，即奖励具有次模性。我们证明可以利用次模性来获取尚未被观察到的组合对象的奖励上界。我们深入分析了这种上界出现的概率，以及一个上界能够覆盖多少未观察到的组合对象。遵循“面对不确定性保持乐观”原则，我们随后提出了SUBo-GFN，该方法利用次模上界来训练生成流网络。我们证明，在对奖励函数查询次数相同的情况下，SUBo-GFN比经典GFN产生数量级更多的学习信号。我们展示了SUBo-GFN在分布匹配方面的有效性。

    arXiv:2601.21061v3 Announce Type: replace-cross  Abstract: Generative Flow Networks (GFlowNets; GFNs) are a class of generative models that learn to sample compositional objects proportional to their a priori unknown value, their reward. We focus on the case where the reward has a specified, actionable structure, namely that it is submodular. We show submodularity can be harnessed to retrieve upper bounds on the reward of compositional objects that have not yet been observed. We provide in-depth analyses of the probability of such bounds occurring, as well as how many unobserved compositional objects can be covered by a bound. Following the Optimism in the Face of Uncertainty principle, we then introduce SUBo-GFN , which uses the submodular upper bounds to train a GFN. We show that SUBo-GFN generates orders of magnitude more learning signal than classical GFNs for the same number of queries to the reward function. We demonstrate the effectiveness of SUBo-GFN in terms of distribution ma
    
[^155]: 基于赌博机反馈的非参数核聚类

    Nonparametric Kernel Clustering with Bandit Feedback

    [https://arxiv.org/abs/2601.07535](https://arxiv.org/abs/2601.07535)

    该论文提出了一个对臂分布无任何参数化假设的非参数聚类框架，通过在再生核希尔伯特空间中对臂的核均值嵌入进行聚类，突破了现有工作依赖次高斯假设、仅能区分均值差异的局限，从而能够识别臂分布在形状或高阶矩上的差异。

    

    带赌博机反馈的聚类是指基于顺序且自适应收集的噪声观测，将N臂随机赌博机的各个臂按照其底层分布进行划分的问题。由于其在推荐系统、众包和适应性试验中的应用价值，这一设定引起了越来越多的关注。现有工作依赖于臂分布为次高斯分布的强假设，这只能区分均值不同的臂，而忽略了分布差异体现在形状或高阶矩的现实场景。我们引入了一个带赌博机反馈的非参数聚类框架，该框架对臂分布没有任何参数化假设。我们采用基于核的方法，将问题重新表述为根据臂在再生核希尔伯特空间（RKHS）中的核均值嵌入对臂进行聚类。在此表述基础上……

    arXiv:2601.07535v2 Announce Type: replace  Abstract: Clustering with bandit feedback refers to the problem of partitioning the arms of an $N$-armed stochastic bandit according to their underlying distributions, based on noisy observations collected sequentially and adaptively. This setting has attracted growing interest for its applicability in recommendation systems, crowdsourcing, and adaptive trials. Existing works rely on the strong assumption that arm distributions are sub-Gaussian, which can only distinguish arms whose distributions differ in mean - missing realistic settings where distributional differences lie in shape or higher moments. % We introduce a framework of \emph{nonparametric clustering with bandit feedback}, free of any parametric assumptions regarding the arm distributions. We adopt a kernel-based approach and reformulate the problem as clustering arms according to their kernel mean embeddings in a reproducing kernel Hilbert space (RKHS). Building on this formulati
    
[^156]: 测量大语言模型评估中的所有噪声

    Measuring all the noises of LLM Evals

    [https://arxiv.org/abs/2512.21326](https://arxiv.org/abs/2512.21326)

    提出“全对比较”方法，定义并测量LLM评估中的预测噪声、数据噪声和总噪声，发现总噪声水平高度可预测，且配对预测噪声通常大于配对数据噪声。

    

    将成熟的统计方法有效应用于大语言模型评估，需要考虑模型间的相关性和预测噪声。主流评估往往忽略数据噪声、统计功效较低，或做出不必要的权衡。我们定义并测量了三种类型的噪声：1）在给定问题上生成不同答案所产生的预测噪声；2）抽样问题所产生的数据噪声；以及3）遵循全方差定律的两者结合的总噪声。为了强调相对比较并获得统计功效，我们提出了“全对比较”方法，该方法将配对比较应用于所有大语言模型对，并基于跨多种评估和设置的数百万条问题级预测来测量所有噪声成分，从而揭示了清晰的规律。首先，大多数评估在所有模型对上表现出特征明确且高度可预测的总噪声水平。其次，配对预测噪声通常超过配对数据噪声。

    arXiv:2512.21326v3 Announce Type: replace-cross  Abstract: Applying well-established statistical methods effectively to LLM evals requires consideration of model correlations and prediction noises. Leading evals often ignore data noise, have low statistical power, or make unnecessary trade-offs. We define and measure three types of noise: 1) prediction noise from generating different answers on a given question, 2) data noise from sampling questions, and 3) their combined total noise following the law of total variance. To emphasize relative comparisons and gain statistical power, we propose \emph{all-pairs comparison}, which applies paired comparison to all pairs of LLMs and measures all the noise components based on millions of question-level predictions across many evals and settings, revealing clear trends. First, most evals exhibit a characteristic and highly predictable total noise level across all model pairs. Second, paired prediction noise typically exceeds paired data noise, 
    
[^157]: 基于热启动的多峰分布采样：重加权退火跳跃点采样器的非渐近界

    Sampling from multimodal distributions with warm starts: Non-asymptotic bounds for the Reweighted Annealed Leap-Point Sampler

    [https://arxiv.org/abs/2512.17977](https://arxiv.org/abs/2512.17977)

    本文提出重加权退火跳跃点采样器（Re-ALPS），去除了ALPS的高斯近似假设，并在自然质量假设下证明了利用热启动点进行多峰分布采样的首个一般设定多项式时间收敛界。

    

    从多峰分布中采样是贝叶斯推断和机器学习中的核心挑战。鉴于采样问题的困难性结果——经典MCMC方法即使采用回火技术，也可能面临指数级的混合时间——一个自然的问题是如何利用额外信息（例如每个模态的热启动点）来实现在各模态之间的更快混合。为解决这一问题，我们提出了重加权ALPS（Re-ALPS），这是退火跳跃点采样器（ALPS）的一个改进版本，它去除了高斯近似假设。在自然假设下——即当分布向相应的热启动点倾斜时，每个分量相对于其他分量包含显著的质量——我们证明了首个适用于一般设定的多项式时间界。与ALPS类似，我们定义了向以热启动点为中心的混合分布倾斜的分布，并在最冷层级上使用热启动点之间的传送机制。

    arXiv:2512.17977v2 Announce Type: replace  Abstract: Sampling from multimodal distributions is a central challenge in Bayesian inference and machine learning. In light of hardness results for sampling -- classical MCMC methods, even with tempering, can suffer from exponential mixing times -- a natural question is how to leverage additional information, such as a warm start point for each mode, to enable faster mixing across modes. To address this, we introduce Reweighted ALPS (Re-ALPS), a modified version of the Annealed Leap-Point Sampler (ALPS) that dispenses with the Gaussian approximation assumption. We prove the first polynomial-time bound that works in a general setting, under a natural assumption that each component contains significant mass relative to the others when tilted towards the corresponding warm start point. Similarly to ALPS, we define distributions tilted towards a mixture centered at the warm start points, and at the coldest level, use teleportation between warm st
    
[^158]: 面向小区域估计的全贝叶斯谱聚类与基准校正及不确定性量化

    Fully Bayesian Spectral Clustering and Benchmarking with Uncertainty Quantification for Small Area Estimation

    [https://arxiv.org/abs/2512.15643](https://arxiv.org/abs/2512.15643)

    提出了基于谱聚类的贝叶斯小区域估计新模型FH-SC，借助后验投影理论推导出基准校正后的Rao-Blackwell估计量闭式解，并引入可推广的条件后验均方误差（CPMSE）来量化基准估计量的不确定性。

    

    在这项工作中，受机器学习技术的启发，我们提出了一种新的小区域估计贝叶斯模型——带谱聚类的Fay-Herriot模型（FH-SC）。与传统方法不同，FH-SC中的聚类基于利用外部协变量的谱聚类算法，而非地理或行政区划标准。FH-SC模型的一个主要优势在于其灵活性，能够整合现有的小区域估计方法，无论是否对随机效应进行聚类。为实现基准校正，我们利用后验投影的约束贝叶斯推断理论框架，推导出了FH-SC模型下新的后验均值Rao-Blackwell（RB）估计量的闭式表达式。此外，我们为基准校正后的估计量引入了一种新颖的不确定性度量——条件后验均方误差（CPMSE），该度量可推广至其他贝叶斯小区域估计估计量。

    arXiv:2512.15643v2 Announce Type: replace-cross  Abstract: In this work, inspired by machine learning techniques, we propose a new Bayesian model for Small Area Estimation (SAE), the Fay-Herriot model with Spectral Clustering (FH-SC). Unlike traditional approaches, clustering in FH-SC is based on spectral clustering algorithms that utilize external covariates, rather than geographical or administrative criteria. A major advantage of the FH-SC model is its flexibility in integrating existing SAE approaches, with or without clustering random effects. To enable benchmarking, we leverage the theoretical framework of posterior projections for constrained Bayesian inference and derive closed form expressions for the new Rao-Blackwell (RB) estimators of the posterior mean under the FH-SC model. Additionally, we introduce a novel measure of uncertainty for the benchmarked estimator, the Conditional Posterior Mean Square Error (CPMSE), which is generalizable to other Bayesian SAE estimators. We
    
[^159]: 扩散桥的控制一致性损失

    Control Consistency Losses for Diffusion Bridges

    [https://arxiv.org/abs/2512.05070](https://arxiv.org/abs/2512.05070)

    本文基于最优控制的自洽性性质提出了一种学习扩散桥的新方法，以迭代在线方式学习条件动力学，无需对模拟轨迹进行微分即可在多种场景中取得优异表现。

    

    在给定初始状态和终端状态的条件下，模拟扩散过程的条件动力学是科学领域一个重要但具有挑战性的问题。对于稀有事件而言，这一困难尤为突出，因为无条件动力学很少能到达终端状态。在这项工作中，我们基于最优控制的自洽性（self-consistency）性质，提出了一种学习扩散桥的新方法。由此得到的算法以迭代在线的方式学习条件动力学，在一系列实证场景中表现出色，且无需对模拟轨迹进行微分求导。除了扩散桥这一设定之外，我们还将自洽性框架与更广泛的随机最优控制文献中的最新进展建立了联系。

    arXiv:2512.05070v3 Announce Type: replace  Abstract: Simulating the conditioned dynamics of diffusion processes, given their initial and terminal states, is an important but challenging problem in the sciences. The difficulty is particularly pronounced for rare events, for which the unconditioned dynamics rarely reach the terminal state. In this work, we propose a novel approach for learning diffusion bridges based on a self-consistency property of the optimal control. The resulting algorithm learns the conditioned dynamics in an iterative online manner, and exhibits strong performance in a range of empirical settings without requiring differentiation through simulated trajectories. Beyond the diffusion bridge setting, we draw connections between our self-consistency framework and recent advances in the wider stochastic optimal control literature.
    
[^160]: 用于解混流的自充分独立成分分析

    Self-sufficient Independent Component Analysis for Demixing Flows

    [https://arxiv.org/abs/2512.00665](https://arxiv.org/abs/2512.00665)

    提出一种无先验、无似然的自充分独立成分分析方法，通过最小化条件KL散度并顺序学习解混流模型来从数据中学习解耦信号，同时完全避免了不稳定的对抗训练。

    

    我们研究了利用非线性独立成分分析（ICA）从数据中学习解耦信号的问题。受自监督学习进展的启发，我们提出学习自充分信号：给定已恢复信号的其余值，观察其他信号不应改变其缺失值的条件分布。我们将该问题表述为条件KL散度的最小化。我们的算法是无先验且无似然的，即它既不规定参数化的源密度，也不规定观测似然。为解决KL散度最小化问题，我们提出了一种顺序算法，在每次迭代中学习一个解混流模型，并证明了其理想化的Wasserstein梯度流变体（具有精确速度和总体投影条件）在总相关性上具有局部下降性质。该方法完全避免了不稳定的对抗性训练。

    arXiv:2512.00665v2 Announce Type: replace-cross  Abstract: We study the problem of learning disentangled signals from data using non-linear Independent Component Analysis (ICA). Motivated by advances in self-supervised learning, we propose to learn self-sufficient signals: Given the remaining values of a recovered signal, observing other signals should not change the conditional distribution of its missing value. We formulate this problem as the minimization of a conditional KL divergence. Our algorithm is prior-free and likelihood-free in the sense that it prescribes neither parametric source densities nor an observation likelihood. To tackle the KL divergence minimization problem, we propose a sequential algorithm that learns a de-mixing flow model at each iteration, and prove local descent of the total correlation for its idealized Wasserstein-gradient-flow variant with exact velocities and a population projection condition. This approach completely avoids the unstable adversarial t
    
[^161]: 一个无限BART模型

    An Infinite BART model

    [https://arxiv.org/abs/2511.20087](https://arxiv.org/abs/2511.20087)

    该论文提出了一种无限BART模型，通过引入带有印度自助餐过程先验的二值权重矩阵，实现了决策树数量的自动选择，并允许不同观测簇仅使用部分弱学习器从而具有不同的回归函数。

    

    贝叶斯加性回归树是回归和分类分析中一种流行的贝叶斯集成模型。在该建模框架下，回归函数由一组决策树的集成来近似，这些决策树被解释为弱学习器。在本工作中，我们提出了BART模型的一个推广，该推广具有两个主要特点：第一，它能够根据给定数据自动选择决策树的数量；第二，该模型允许不同的观测簇拥有不同的回归函数，因为每个数据点只能使用部分弱学习器，而不是全部弱学习器。这一模型推广是通过在响应变量的条件分布中引入一个二值权重矩阵来实现的，该矩阵仅为每个观测激活一个特定的决策树子集。该矩阵被赋予印度自助餐过程先验，并在MCMC采样器中与其他参数一起进行采样。

    arXiv:2511.20087v2 Announce Type: replace-cross  Abstract: Bayesian additive regression trees (BART) is a popular Bayesian ensemble model used in regression and classification analysis. Under this modeling framework, the regression function is approximated by an ensemble of decision trees, interpreted as weak learners. In this work, we propose a generalization of the BART model that has two main features: first, it automatically selects the number of decision trees using the given data; second, the model allows clusters of observations to have different regression functions since each data point can only use a selection of weak learners, instead of all of them. This model generalization is accomplished by including a binary weight matrix in the conditional distribution of the response variable, which activates only a specific subset of decision trees for each observation. Such a matrix is endowed with an Indian Buffet process prior, and sampled within the MCMC sampler, together with th
    
[^162]: Toeplitz协方差的Carathéodory分解的优化景观

    The Optimization Landscape of Carath\'eodory Decomposition of Toeplitz Covariances

    [https://arxiv.org/abs/2511.01605](https://arxiv.org/abs/2511.01605)

    该论文通过对Toeplitz协方差进行过参数化Carathéodory表示（同时优化频率与幅度），证明这种非线性参数化不会引入额外的驻点，从而解释了为何高斯最大似然非凸问题在样本充足时可被全局求解。

    

    Toeplitz协方差估计是一个经典的信号处理问题，然而高斯最大似然目标的几何结构至今仍未被完全理解。近期的牛顿型方法、最大化-最小化方法以及基于梯度的方法在样本充足时通常能够全局求解这一非凸问题，但同时也揭示出一个困难的计算景观。我们通过一种过参数化的Carathéodory表示来研究这一现象，该表示利用导向向量的频率和幅度对协方差进行参数化。一个简单的例子表明，对于任何均匀频率网格，即使在总体（population）情形下，仅优化幅度仍会留下正的误差，这促使我们对频率也进行优化。我们的主要结果证明，这种非线性参数化不会增加驻点：对于任何样本协方差，每一个具有正定协方差的驻点都能给出……的驻点（原文此处截断）。

    arXiv:2511.01605v3 Announce Type: replace-cross  Abstract: Toeplitz covariance estimation is a classical signal processing problem, yet the geometry of the Gaussian maximum-likelihood objective remains only partially understood. Recent Newton-type, majorization-minimization, and gradient-based methods often solve the nonconvex problem globally with enough samples, but also reveal a difficult computational landscape. We study this phenomenon through an overparameterized Carath\'eodory representation, which parameterizes the covariance by the frequencies and amplitudes of steering vectors. A simple example shows that, for any uniform frequency grid, optimizing only the amplitudes can leave a positive error even in the population setting, which motivates optimizing the frequencies as well. Our main result proves that this nonlinear parameterization adds no stationary points: for any sample covariance, every stationary point with a positive definite covariance gives a stationary point of t
    
[^163]: 基于小球方法的物理信息统计学习的复杂度相关误差率

    Complexity Dependent Error Rates for Physics-informed Statistical Learning via the Small-ball Method

    [https://arxiv.org/abs/2510.23149](https://arxiv.org/abs/2510.23149)

    本文通过小球方法为物理信息统计学习建立了复杂度相关的误差率，回答了软惩罚与硬约束之间的权衡问题，并量化了引入物理知识相较于纯数据驱动方法所带来的统计增益。

    

    物理信息统计学习将经验数据与物理知识相结合，以提升估计器的统计性能。尽管PISL方法在实践中被广泛使用，但目前仍缺乏对信息正则化如何影响统计性质的全面理论理解。具体而言，两个基本问题尚未得到充分解答：（A）软惩罚与硬约束之间的权衡是什么，（B）与纯数据驱动的经验误差最小化相比，引入物理知识能带来多大的统计增益。本文针对凸包满足小球条件的函数类（尤其是凸函数类），在以线性方程形式表达的物理知识下，通过基于小球方法构建适当的复杂度相关误差率，来解答上述两个问题。我们进一步将这些结论具体化到有限维d的线性函数类（摘要在此处截断）。

    arXiv:2510.23149v2 Announce Type: replace  Abstract: Physics-informed statistical learning (PISL) integrates empirical data with physical knowledge to enhance the statistical performance of estimators. While PISL methods are widely used in practice, a comprehensive theoretical understanding of how informed regularization affects statistical properties is still missing. Specifically, two fundamental questions have yet to be fully addressed: (A) what is the trade-off between soft penalties and hard constraints, and (B) what is the statistical gain of incorporating physical knowledge compared to purely data-driven empirical error minimisation. In this paper, we address these questions for PISL in classes of functions whose convex hull satisfies a small-ball condition, in particular convex classes, under physical knowledge expressed as linear equations by developing appropriate complexity dependent error rates based on the small-ball method. We specialise them to linear classes of finite d
    
[^164]: 用于高维分段连续函数代理建模的深度跳跃高斯过程

    Deep Jump Gaussian Processes for Surrogate Modeling of High-Dimensional Piecewise Continuous Functions

    [https://arxiv.org/abs/2510.21974](https://arxiv.org/abs/2510.21974)

    本文提出深度跳跃高斯过程（DJGP），通过将区域特定的局部线性投影与跳跃高斯过程相结合的双层深度学习框架，解决了传统 JGP 在高维空间中的局限，实现了高维分段连续函数的高效代理建模。

    

    我们提出了深度跳跃高斯过程，这是一种用于高维域上分段连续函数代理建模的新方法。DJGP 通过将区域特定的局部线性投影与 JGP 建模相结合，解决了传统跳跃高斯过程在高维输入空间中的局限性。这些投影采用依赖于区域的矩阵来捕捉局部低维子空间结构，使其非常适合 JGP（局部高斯过程的一种变体）固有的局部化建模行为。为了控制模型复杂度，我们在投影矩阵上施加了高斯过程先验，使其能够在整个输入空间中平滑演化。投影后的输入随后由 JGP 建模，以捕捉其与响应之间的分段连续关系。这形成了一种独特的 GP/JGP 双层深度学习结构。我们进一步开发了可扩展的变分推断算法。

    arXiv:2510.21974v3 Announce Type: replace-cross  Abstract: We introduce Deep Jump Gaussian Processes (DJGP), a novel method for surrogate modeling of a piecewise continuous function on a high-dimensional domain. DJGP addresses the limitations of conventional Jump Gaussian Processes (JGP) in high-dimensional input spaces by integrating region-specific, locally linear projections with JGP modeling. These projections employ region-dependent matrices to capture local low-dimensional subspace structures, making them well suited to the inherently localized modeling behavior of JGPs, a variant of local Gaussian processes. To control model complexity, we place a Gaussian Process prior on the projection matrices, allowing them to evolve smoothly across the input space. The projected inputs are then modeled with a JGP to capture piecewise continuous relationships with the response. This yields a distinctive two-layer deep learning of GP/JGP. We further develop a scalable variational inference al
    
[^165]: 带动量的随机凸差优化

    Stochastic Difference-of-Convex Optimization with Momentum

    [https://arxiv.org/abs/2510.17503](https://arxiv.org/abs/2510.17503)

    本文提出基于动量的随机凸差优化算法MDCA，将阻尼DCA步骤与动量估计器结合，在稀疏学习、鲁棒回归等非凸优化问题上以更少的随机梯度调用次数达到与DCA-PAGE相当的稳定性能。

    

    我们研究在线随机凸差规划问题 $\min_{x \in R^n} F(x) = G(x) - H(x) + r_1(x) - r_2(x)$，该类问题涵盖了带非凸惩罚项的稀疏学习、带光滑鲁棒损失的鲁棒回归，以及非负正例-未标注学习。我们提出了基于动量的DCA算法，该方法将阻尼DCA步骤与针对 $\nabla H$ 的动量估计器相结合，并采用固定的每迭代批次大小。在凸子问题可精确求解的情况下，采用Polyak动量的MDCA只需 $\mathcal{O}(\epsilon^{-4})$ 个 $H$ 的随机梯度即可达到 $\epsilon$-稳定性，而基于动量的方差缩减算法在平均光滑性条件下仅需 $\mathcal{O}(\epsilon^{-3})$，并且用单个 $\mathcal{O}(\epsilon^{-1})$ 的预热批次取代了DCA-PAGE的周期性大批次，同时匹配了其关于 $\epsilon$ 的复杂度依赖性。我们还引入了单循环变体（S-MDCA），用以替代子问题的求解

    arXiv:2510.17503v2 Announce Type: replace-cross  Abstract: We study the online stochastic difference-of-convex (DC) program $\min_{x \in R^n} F(x) = G(x) - H(x) + r_1(x) - r_2(x)$, which covers sparse learning with nonconvex penalties, robust regression with smooth robust losses, and non-negative positive--unlabeled learning. We propose \emph{momentum-based DCA} (MDCA), which combines a damped DCA step with a momentum estimator for $\nabla H$ and uses a constant per-iteration batch size. With exact solutions of the convex subproblems, MDCA with Polyak momentum uses $\mathcal{O}( \epsilon^{-4})$ stochastic gradients of $H$ to reach $\epsilon$-stationarity, and momentum-based variance reduction (MVR) uses $\mathcal{O}(\epsilon^{-3})$ under averaged smoothness, with a single $\mathcal O(\epsilon^{-1})$ warm-up batch in place of the periodic large batches of DCA-PAGE, whose $\epsilon$-dependence it matches. We also introduce a single-loop variant (S-MDCA) that replaces the subproblem solve
    
[^166]: 平均场变分推断的众数坍缩

    Mode Collapse of Mean-Field Variational Inference

    [https://arxiv.org/abs/2510.17063](https://arxiv.org/abs/2510.17063)

    该论文首次从理论上解释了平均场变分推断中的众数坍缩现象，通过引入 ε-分离性概念，推导出 MFVI 优化器分配给混合分布各分量质量的显式界，并扩展到多分量混合分布的情形。

    

    平均场变分推断（MFVI）是一种被广泛使用的方法，通过乘积测度来近似高维概率分布。经验上已经观察到，MFVI 优化器经常遭受“众数坍缩”问题。例如，当目标测度 π 是混合分布 π = w P_0 + (1 - w) P_1 时，MFVI 优化器倾向于将其大部分质量集中在混合分布的单个分量附近。这项工作首次为 MFVI 中的众数坍缩提供了理论解释。从两组分混合分布出发，我们引入“ε-分离性”的概念来量化混合分布各分量之间的分离程度。当 P_0 和 P_1 在足够小的 ε 下满足 ε-分离时，我们推导出了任意 MFVI 优化器分配给每个分量的质量的显式界。我们进一步将这一分析扩展到具有多个分量的混合目标分布。我们的结果揭示了……（摘要在此处被截断）

    arXiv:2510.17063v2 Announce Type: replace  Abstract: Mean-field variational inference (MFVI) is a widely used method for approximating high-dimensional probability distributions by product measures. It has been empirically observed that MFVI optimizers often suffer from \emph{mode collapse}. For example, when the target measure $\pi$ is a mixture $\pi = w P_0 + (1 - w) P_1$, the MFVI optimizer tends to place most of its mass near a single component of the mixture. This work provides the first \emph{theoretical} explanation of mode collapse in MFVI. Starting from two-component mixtures, we introduce the notion of \emph{$\varepsilon$-separateness} to quantify the separation of the mixture components. We derive explicit bounds on the mass that any MFVI optimizer assigns to each component, when $P_0$ and $P_1$ are $\varepsilon$-separated for sufficiently small $\varepsilon$. We further extend this analysis to mixture target distributions with multiple components. Our results reveal that th
    
[^167]: 基于部分批次标签采样的批次贝叶斯主动学习

    Batch Bayesian Active Learning with Partial Batch Label Sampling

    [https://arxiv.org/abs/2510.09877](https://arxiv.org/abs/2510.09877)

    该论文扩展了主动学习的贝叶斯决策理论框架，提出部分批次标签采样方法ParBaLS，使EPIG算法在批次设置下既避免了BatchBALD的计算难题，又克服了top-B选择的性能下降，在不同数据集和预算下均取得卓越性能。

    

    在过去几十年中，许多主动学习采集函数被相继提出，使得实践者难以明确选择应使用哪一种。贝叶斯主动学习提供了具有可解释直觉的原则性目标，包括期望误差减少（EER）、期望预测信息增益（EPIG）以及基于分歧的贝叶斯主动学习（BALD）。这类方法的一个关键挑战是难以扩展到批次设置，这会导致要么面临计算上的挑战（如BatchBALD），要么出现显著的性能下降（如top-B选择）。在这项工作中，我们扩展了主动学习的贝叶斯决策理论框架，为EPIG算法推导出了部分批次标签采样方法。我们通过实验表明，在来自大型预训练模型的嵌入表示上使用贝叶斯逻辑回归时，ParBaLS EPIG在不同预算下的多种数据集上都能取得卓越的性能。我们的代码已公开。

    arXiv:2510.09877v4 Announce Type: replace-cross  Abstract: Over the past couple of decades, many active learning acquisition functions have been proposed, leaving practitioners with an unclear choice of which to use. Bayesian active learning offers principled objectives with explainable intuition, including Expected Error Reduction (EER), Expected Predictive Information Gain (EPIG), and Bayesian Active Learning by Disagreements (BALD). A key challenge of such methods is the difficulty of extending to the batch setting, leading to either computational challenges (BatchBALD) or dramatic performance drops (top-$B$ selection). In this work, we extend active learning's Bayesian Decision Theory framework to derive Partial Batch Label Sampling (ParBaLS) for the EPIG algorithm. We show experimentally that ParBaLS EPIG gives superior performance for a variety of datasets with different budgets using Bayesian logistic regression on embeddings from large pre-trained models. Our code is available 
    
[^168]: 不确定性下随机模型尺度参数训练的贝叶斯优化

    Bayesian Optimization under Uncertainty for Training a Scale Parameter in Stochastic Models

    [https://arxiv.org/abs/2510.06439](https://arxiv.org/abs/2510.06439)

    提出了一种面向不确定性下随机模型尺度或精度参数优化的新贝叶斯优化框架，通过统计代理模型实现期望的解析求值以及随机采集函数最优解的闭式表达式，相比基于蒙特卡洛的优化方案数据使用量减少约40倍，大幅提升了优化效率。

    

    超参数调优是一个具有挑战性的问题，尤其是当系统本身涉及不确定性时。由于函数评估存在噪声，不确定性下的优化可能计算代价高昂。在本文中，我们提出了一种新颖的贝叶斯优化框架，专门针对不确定性下的超参数调优，重点关注随机模型中尺度型或精度型参数的优化。所提出的方法对底层随机变量采用统计代理模型，使得期望算子能够进行解析求值。此外，我们推导了随机采集函数最优解的闭式表达式，这显著降低了每次迭代的计算成本。数值实验表明优化效率获得了显著提升：与基于蒙特卡洛的一维优化方案相比，数据使用量减少了约40倍，并且性能有数量级的改进。

    arXiv:2510.06439v2 Announce Type: replace-cross  Abstract: Hyperparameter tuning is a challenging problem especially when the system itself involves uncertainty. Due to noisy function evaluations, optimization under uncertainty can be computationally expensive. In this paper, we present a novel Bayesian optimization framework tailored for hyperparameter tuning under uncertainty, with a focus on optimizing a scale- or precision-type parameter in stochastic models. The proposed method employs a statistical surrogate for the underlying random variable, enabling analytical evaluation of the expectation operator. Moreover, we derive a closed-form expression for the optimizer of the random acquisition function, which significantly reduces computational cost per iteration. Numerical experiments demonstrate substantial gains in optimization efficiency: about 40-fold reduction in data usage relative to a Monte Carlo-based one-dimensional optimization scheme, and an order of magnitude improvemen
    
[^169]: 从随机近似视角重新思考朗之万汤普森采样

    Rethinking Langevin Thompson Sampling from A Stochastic Approximation Perspective

    [https://arxiv.org/abs/2510.05023](https://arxiv.org/abs/2510.05023)

    TS-SA将随机近似（SA）融入朗之万汤普森采样框架，每轮仅用新观测奖励执行一步朗之万更新并结合递减的SA权重，为每个臂维持一条持续的马尔可夫链，从而无需存储或重复采样历史观测即可高效逼近变化的后验分布。

    

    基于朗之万的汤普森采样（TS）方法用于多臂老虎机问题，在精确采样难以实现时，利用似然梯度来近似后验样本。随着新奖励的到来，后验分布会发生变化，这类方法需要运行一条内部马尔可夫链来逼近更新后的目标分布。为了简化这种重复采样过程，我们提出了TS-SA，它将随机近似（SA）引入TS框架之中。在每一轮中，TS-SA利用新观测到的奖励执行一步朗之万更新，并通过一次SA更新将含噪的提议与当前参数估计相结合。通过在各轮之间保留该参数估计，TS-SA为每个臂维护一条持续的马尔可夫链，在无需存储或重新采样历史观测的情况下保留了过去观测的影响。在这一递归过程中，提议步长在各轮之间保持固定，而SA权重则随观测不断累积而逐渐递减，以控制……

    arXiv:2510.05023v2 Announce Type: replace-cross  Abstract: Langevin-based Thompson Sampling (TS) methods for multi-armed bandits use likelihood gradients to approximate posterior samples when exact sampling is intractable. As new rewards arrive, the posterior changes, and these methods run an inner Markov chain to approximate the updated target. To simplify this repeated sampling procedure, we introduce TS-SA, which incorporates stochastic approximation (SA) within the TS framework. In each round, TS-SA takes one Langevin step using the newly observed reward and applies an SA update to combine the noisy proposal with the current parameter estimate. By retaining this estimate across rounds, TS-SA maintains one continuing Markov chain for each arm, preserving the influence of past observations without storing or sampling them again. Within this recursion, the proposal step size remains fixed across rounds, while the SA weights decrease as observations accumulate to control the effect of 
    
[^170]: 基于元数据引导的跨异构应用与设备的自适应频率调节

    Metadata-Guided Adaptable Frequency Scaling across Heterogeneous Applications and Devices

    [https://arxiv.org/abs/2509.22707](https://arxiv.org/abs/2509.22707)

    提出MetaDVFS，一个元数据引导的多任务强化学习框架，通过挖掘并迁移设备与应用元数据中的共享知识，实现跨异构硬件和应用的可泛化、低部署成本的DVFS频率调节。

    

    动态电压与频率调节（DVFS）对于提升移动平台的能效至关重要。然而，传统的基于启发式策略的调节器在应对异构片上系统设计的复杂性以及多样化的应用工作负载方面日益显得力不从心。尽管强化学习方法能够带来性能提升，但其泛化能力差，且需要针对每种硬件与应用组合进行大量重新训练，导致高昂的部署成本。在本工作中，我们观察到设备与应用的元数据天然蕴含了对DVFS有价值的知识，这为克服上述局限提供了机会。我们将面向异构设备与应用的DVFS问题形式化为一个多任务强化学习问题，并提出了MetaDVFS——一个元数据引导的框架，它系统地利用元数据来发现并在设备与应用之间迁移共享知识。

    arXiv:2509.22707v2 Announce Type: replace-cross  Abstract: Dynamic Voltage and Frequency Scaling is essential for enhancing energy efficiency in mobile platforms. However, traditional heuristic-based governors are increasingly inadequate for managing the complexity of heterogeneous System-on-Chip designs and diverse application workloads. Although reinforcement learning approaches offer improved performance, their poor generalization capability and reliance on extensive retraining for each hardware and application combination leads to significant deployment costs. In this work, we observe that device and application metadata inherently encapsulate valuable knowledge for DVFS, presenting an opportunity to overcome these limitations. We formulate DVFS for heterogeneous devices and applications as a multi-task reinforcement learning problem. We introduce MetaDVFS, which is a metadata-guided framework that systematically leverages metadata to discover and transfer shared knowledge across D
    
[^171]: 具有无限交换并行回火的弹跳粒子采样器

    Bouncy particle sampler with infinite exchanging parallel tempering

    [https://arxiv.org/abs/2509.02003](https://arxiv.org/abs/2509.02003)

    本文提出了无限交换极限下的并行回火弹跳粒子采样器（BPS-PT），在曲率变化较大的分布上实现了与HMC-PT和LMC-PT相当的估计精度，同时单位计算时间的有效样本量更高。

    

    马尔可夫链蒙特卡罗（MCMC）是从复杂分布（如贝叶斯推断中的后验分布）中获取样本的最有用工具之一。弹跳粒子采样器（BPS）是一种利用匀速直线运动和弹性碰撞进行采样的方法。由于BPS不需要时间离散化，预期它在处理曲率变化较大的分布时，比广泛使用的哈密顿蒙特卡罗（HMC）和朗之万蒙特卡罗（LMC）更具优势。为了更高效地从多峰分布中采样，我们基于先前研究提出的具有无限交换速率的LMC-PT，提出了无限交换极限下的并行回火弹跳粒子采样器（BPS-PT）。在曲率变化较大的分布上进行测试时，BPS-PT展现出与HMC-PT和LMC-PT相当的估计精度，并且在单位计算时间内获得了更高的有效样本量（ESS）。因此，可以认为BPS-PT对此类分布的采样是有效的。

    arXiv:2509.02003v2 Announce Type: replace-cross  Abstract: Markov chain Monte Carlo is one of the most useful tools for obtaining samples from complex distributions, such as posterior distributions in Bayesian inference. Bouncy particle sampler (BPS) is a method that performs sampling using uniform linear motion and elastic collisions. Since BPS does not require time discretization, it is expected to be more advantageous than the widely used Hamiltonian Monte Carlo (HMC) and Langevin Monte Carlo (LMC) for distributions with large curvature changes. To sample more efficiently from multimodal distributions, we propose BPS with parallel tempering (BPS-PT) in the infinite exchange limit, based on LMC-PT with the infinite exchange rate proposed in a previous study. Using a distribution with large curvature changes, BPS-PT showed estimation accuracy comparable to HMC-PT and LMC-PT, and achieved higher ESS per computational time. Therefore, BPS-PT is considered to be effective for distributio
    
[^172]: 马尔可夫噪声下带Richardson-Romberg外推的线性随机逼近算法的高阶误差界

    High-Order Error Bounds for Markovian LSA with Richardson-Romberg Extrapolation

    [https://arxiv.org/abs/2508.05570](https://arxiv.org/abs/2508.05570)

    本文提出一种新颖的线性化偏差分解方法，揭示PR平均无法消除马尔可夫噪声下LSA算法偏差的线性主导项，并通过Richardson-Romberg外推有效消除该偏差项，同时建立了高阶矩误差界。

    

    本文研究了马尔可夫噪声下采用Polyak-Ruppert（PR）平均的线性随机逼近（LSA）算法的偏差与高阶误差界。我们聚焦于该算法的常数步长 $\alpha$ 版本，并通过线性化技术提出了一种新颖的偏差分解方法。我们分析了偏差的结构，证明其主导项关于 $\alpha$ 呈线性关系，且无法通过PR平均消除。为解决这一问题，我们应用了Richardson-Romberg（RR）外推方法，该方法能够有效消除偏差的主导项。我们推导了RR迭代的高阶矩界，并证明其主导误差项与标准平均LSA迭代的渐近最优协方差矩阵相一致。

    arXiv:2508.05570v2 Announce Type: replace  Abstract: In this paper, we study the bias and high-order error bounds of the Linear Stochastic Approximation (LSA) algorithm with Polyak-Ruppert (PR) averaging under Markovian noise. We focus on the version of the algorithm with constant step size $\alpha$ and propose a novel decomposition of the bias via a linearization technique. We analyze the structure of the bias and show that the leading-order term is linear in $\alpha$ and cannot be eliminated by PR averaging. To address this, we apply the Richardson-Romberg (RR) extrapolation procedure, which effectively cancels the leading bias term. We derive high-order moment bounds for the RR iterates and show that the leading error term aligns with the asymptotically optimal covariance matrix of the vanilla averaged LSA iterates.
    
[^173]: 串行缩放假设

    The Serial Scaling Hypothesis

    [https://arxiv.org/abs/2507.12549](https://arxiv.org/abs/2507.12549)

    该论文提出“串行缩放假设”，在复杂性理论中形式化了“内在串行问题”与可并行问题的区别，并首次证明扩散模型也无法解决这类本质串行的问题，指出识别计算的串行本质对机器学习、模型设计和硬件开发意义重大。

    

    尽管机器学习通过大规模并行化取得了进展，我们发现了一个关键的盲点：某些问题在本质上是顺序性的。这些“内在串行”的问题——从数学推理到物理模拟再到序贯决策——需要顺序依赖的计算步骤，无法被高效地并行化。我们在复杂性理论中对该区别进行了形式化，并证明当前以并行为中心的架构在这类任务上面临根本性的限制。随后，我们首次证明扩散模型尽管具有顺序性质，却无法解决内在串行问题。我们认为，认识到计算的串行本质对机器学习、模型设计和硬件开发具有深远的影响。

    arXiv:2507.12549v5 Announce Type: replace-cross  Abstract: While machine learning has advanced through massive parallelization, we identify a critical blind spot: some problems are fundamentally sequential. These "inherently serial" problems-from mathematical reasoning to physical simulations to sequential decision-making-require sequentially dependent computational steps that cannot be efficiently parallelized. We formalize this distinction in complexity theory, and demonstrate that current parallel-centric architectures face fundamental limitations on such tasks. Then, we show for first time that diffusion models despite their sequential nature are incapable of solving inherently serial problems. We argue that recognizing the serial nature of computation holds profound implications on machine learning, model design, and hardware development.
    
[^174]: 广义核下t分布随机邻域嵌入的平衡分布

    Equilibrium Distribution for t-Distributed Stochastic Neighbor Embedding with Generalized Kernels

    [https://arxiv.org/abs/2505.24311](https://arxiv.org/abs/2505.24311)

    该论文为广义输入输出核下的t-SNE大样本变分问题建立了严格的数学理论，证明了尺度参数的存在唯一性、解的存在性与一致有界性，以及离散最优解收敛到满足平衡方程的紧支撑平衡分布。

    

    我们研究了一类输入核和输出核下t分布随机邻域嵌入的大样本变分问题。输入律具有紧支撑，且其密度在该支撑上连续。一个熵方程确定了输入核中的尺度参数，我们证明该参数在正密度的内部点处存在且唯一。随后，我们给出了解在整个支撑上存在且一致有界的充分条件。在这些条件以及输出核的衰减假设下，离散最优值收敛于连续统最小值。近似极小元的经验测度在平移后是紧的；每个子序列极限都是满足平衡方程的紧支撑极小元。可容许的输出核包括高斯核，以及在输出维度为二时的柯西核。数值例子比较了二维表示（原文在此处截断）。

    arXiv:2505.24311v3 Announce Type: replace-cross  Abstract: We study the large-sample variational problem for t-distributed stochastic neighbor embedding with a class of input and output kernels. The input law has compact support and a density continuous on that support. An entropy equation determines the scale parameter in the input kernel, and we prove that this parameter exists and is unique at interior points of positive density. We then give sufficient conditions for solutions to exist and be uniformly bounded on the entire support. Under these conditions and a decay assumption on the output kernel, the discrete optimal values converge to a continuum minimum. Empirical measures of approximate minimizers are tight after translation; every subsequential limit is a compactly supported minimizer satisfying the equilibrium equation. The admissible output kernels include Gaussian kernels and, in output dimension two, the Cauchy kernel. Numerical examples compare the two-dimensional repre
    
[^175]: 斯坦无偏风险估计与Hyvärinen分数匹配

    Stein's unbiased risk estimate and Hyv\"arinen's score matching

    [https://arxiv.org/abs/2502.20123](https://arxiv.org/abs/2502.20123)

    该论文首次证明了SURE最小化与Hyvärinen隐式分数匹配的等价性，并从理论上证明SURE在经验贝叶斯去噪中可实现接近参数速率的遗憾收敛，以及在误设模型下向oracle去噪器的快速收敛。

    

    给定一组被高斯噪声污染的观测信号，我们如何学习以最优方式对它们进行去噪？这一基本问题同时出现在经验贝叶斯和生成式建模中。在经验贝叶斯中，主流方法是非参数极大似然估计（NPMLE），而在生成式建模中，分数匹配（SM）方法已被证明非常成功。在我们的设定中，Hyvärinen的隐式分数匹配等价于统计学中另一个经典思想——斯坦无偏风险估计（SURE）。通过重新审视SURE最小化，我们首次证明了在具有同方差噪声的经典经验贝叶斯设定中，SURE能够达到接近参数速率的遗憾收敛速度。我们还证明了SURE训练在一个常被研究的误设模型中能够以快速收敛速度逼近oracle去噪器。我们展示了如何在涉及异方差噪声的设定中实际实现我们的方法……

    arXiv:2502.20123v3 Announce Type: replace-cross  Abstract: Given a collection of observed signals corrupted with Gaussian noise, how can we learn to optimally denoise them? This fundamental problem arises in both empirical Bayes and generative modeling. In empirical Bayes, the predominant approach is via nonparametric maximum likelihood estimation (NPMLE), while in generative modeling, score matching (SM) methods have proven very successful. In our setting, Hyv\"arinen's implicit SM is equivalent to another classical idea from statistics---Stein's Unbiased Risk Estimate (SURE). Revisiting SURE minimization, we establish, for the first time, that SURE achieves nearly parametric rates of convergence of the regret in the classical empirical Bayes setting with homoscedastic noise. We also prove that SURE-training can achieve fast rates of convergence to the oracle denoiser in a commonly studied misspecified model. We show how to practically implement our method in settings involving hetero
    
[^176]: 通过API生成差分隐私合成数据 3：使用模拟器代替基础模型

    Differentially Private Synthetic Data via APIs 3: Using Simulators Instead of Foundation Model

    [https://arxiv.org/abs/2502.05505](https://arxiv.org/abs/2502.05505)

    Sim-PE将私有演化（PE）框架推广到可自然集成计算机图形图像生成器和物理仿真工具等非神经网络模拟器，从而在不依赖基础模型的情况下生成差分隐私合成数据，并将下游分类准确率提升最高3倍、FID降低最高80%。

    

    差分隐私（DP）合成数据能够在提供强有力隐私保证的同时使用敏感数据集。近来，私有演化（PE）框架成为一种很有前景的方法，它仅利用基础模型的推理API就能生成DP合成数据。然而，对于每个私有数据领域，合适的基础模型并不总是可得。在本文中，我们证明PE比之前人们理解的更加通用，可以纳入基础模型之外的API。特别地，我们展示了最先进的非神经网络数据合成器，包括基于计算机图形学的图像生成器和物理仿真工具（我们将其统称为模拟器），能够自然地集成到PE中。我们将由此产生的方法称为Sim-PE。Sim-PE拓宽了PE的适用范围，同时提升了性能。在图像合成任务上，Sim-PE将下游分类准确率最高提升了3倍，将FID最高降低了80%。

    arXiv:2502.05505v4 Announce Type: replace-cross  Abstract: Differentially private (DP) synthetic data enables the use of sensitive datasets while providing strong privacy guarantees. Recently, Private Evolution (PE) has emerged as a promising framework for generating DP synthetic data using only inference APIs of foundation models. However, suitable foundation models are not always available for every private data domain. In this paper, we show that PE is more general than previously understood and can incorporate APIs beyond foundation models. In particular, we show that state-of-the-art non-neural-network data synthesizers, including computer graphics-based image generators and physics simulation tools (which we refer to as simulators), integrate naturally into PE. We call the resulting method Sim-PE. Sim-PE broadens the applicability of PE while improving performance. For image synthesis, Sim-PE improves downstream classification accuracy by up to 3x, reduces FID by up to 80%, and i
    
[^177]: AMP艾滋病预防试验中区域特异性生存效应的靶向数据融合

    Targeted Data Fusion for Region-Specific Survival Effects in the AMP HIV Prevention Trials

    [https://arxiv.org/abs/2501.18798](https://arxiv.org/abs/2501.18798)

    针对AMP艾滋病预防试验，提出了一种基于L1正则化准则的联邦学习方法，在不共享个体数据的情况下融合多区域信息，以估计区域特异性和治疗特异性的生存曲线及因果效应。

    

    抗体介导预防（AMP）试验通过证明被动给予的单克隆广谱中和抗体能够预防HIV-1感染，开辟了新的科学前沿。该试验在四个流行病学特征各异的地区招募了4,611名参与者，随访期间观察到174例HIV-1确诊病例。治疗效果存在区域异质性，加之隐私和监管对中央数据汇集的限制，因此需要能够在不共享个体层面数据的前提下跨区域借力的方法。为了在分布异质性条件下估计区域特异性和治疗特异性的生存曲线，我们开发了一种联邦学习方法，通过L1正则化准则将各站点的估计量进行组合，该准则能够降低与目标不一致的数据源的权重。我们进一步将该框架扩展到一类更广泛的因果对比，包括生存差异、生存比等。

    arXiv:2501.18798v5 Announce Type: replace-cross  Abstract: The Antibody Mediated Prevention (AMP) trials opened a new scientific frontier by showing that passively administered monoclonal broadly neutralizing antibodies (bnAbs) could prevent HIV-1 acquisition. The trials enrolled 4,611 participants across four epidemiologically distinct regions, with 174 HIV-1 diagnoses observed during follow-up. Regional heterogeneity in treatment efficacy, together with privacy and regulatory limits on central data pooling, calls for methods that borrow strength across regions without sharing individual-level data. To estimate region- and treatment-specific survival curves under distributional heterogeneity, we develop a federated learning approach that combines site-specific estimators via an L1-regularized criterion that downweights data sources not aligned with the target. We further extend the framework to a general class of causal contrasts, including the survival difference, survival ratio, and
    
[^178]: 面向分布式混合专家模型的最优传输聚合

    Optimal Transport Aggregation for Distributed Mixture-of-Experts

    [https://arxiv.org/abs/2312.09877](https://arxiv.org/abs/2312.09877)

    提出一种基于最优传输散度的聚合方法，可将分布式环境中独立训练的多个本地混合专家模型合并为一个一致且仅含K个专家的全局MoE模型。

    

    本文考虑这样一种场景：在分布式数据集上独立拟合本地的混合专家模型，然后需要将这些模型合并为单一的全局MoE模型。由于各机器上的专家并未相互对齐，对本地混合模型进行加权平均通常会产生一个包含大量专家（而非K个专家）的大型混合模型，并且无法提供真实模型参数的估计。我们提出通过最小化一个期望传输散度，将该大型混合模型缩减为具有K个专家的MoE模型，其中传输计划依赖于协变量。该问题的一个松弛形式将专家与门控网络解耦：专家通过主化-最小化算法计算得到，该算法使目标函数单调下降；随后将门控网络拟合到由最优传输计划所隐含的门控上，并给出了由此产生损失的显式上界。我们证明，只要满足一定条件，聚合估计量就具有一致性。

    arXiv:2312.09877v3 Announce Type: replace-cross  Abstract: This paper considers the setting in which local Mixture-of-experts (MoE) models are fitted independently on distributed datasets and must be combined into a single global MoE model. Weighted averaging the local mixtures, whose experts are not aligned across machines, often yields a large mixture with many experts instead of $K$ and does not provide an estimator of the true model parameters. We propose to reduce this large mixture to a $K$-expert MoE by minimizing an expected transportation divergence, in which the transport plan depends on the covariates. A relaxation of this problem separates the experts from the gating network: the experts are computed by a majorization--minimization algorithm, which decreases the objective monotonically, and the gating network is then fitted to the gates implied by the optimal plan, with an explicit bound on the resulting loss. We show that the aggregated estimator is consistent whenever the
    
[^179]: 面向多臂老虎机问题的贝叶斯非参数汤普森采样

    Bayesian Nonparametric Thompson Sampling for Multi-Armed Bandits

    [https://arxiv.org/abs/1808.02932](https://arxiv.org/abs/1808.02932)

    本文提出用贝叶斯非参数高斯混合模型对每个臂的奖励分布建模，将汤普森采样扩展到任意奖励分布，从而无需针对具体场景预先设定参数化奖励模型和调整超参数。

    

    在随机多臂老虎机问题中，奖励由未知分布生成；世界的奖励不确定性引发了探索与利用之间的权衡。汤普森采样优雅地解决了这一权衡问题，但它需要预先知道奖励模型的参数形式，才能从其后验分布中采样并计算每个臂的期望奖励——这限制了它的适用性，因为它要求针对每个具体场景进行模型设定和超参数调整。本文采用贝叶斯非参数混合模型来消除这一要求，将汤普森采样扩展到任意奖励分布。我们用一个独立的贝叶斯非参数高斯混合模型来刻画每个臂的未知奖励分布，该模型可以近似任意连续的奖励密度，并能根据顺序观测到的数据自适应地调整模型复杂度。将贝叶斯非参数混合模型嵌入汤普森采样，得到了非参数汤普森采样算法，该算法能够按顺序学习最接近真实（但未知的）奖励分布的模型。

    arXiv:1808.02932v5 Announce Type: replace  Abstract: In stochastic multi-armed bandits, rewards are generated by an unknown distribution; the world's reward uncertainty induces the exploration-exploitation trade-off. Thompson sampling (TS) addresses this trade-off elegantly, but requires knowing the reward model's parametric form to sample from its posterior and compute per-arm expected rewards ---which limits its applicability as it demands per-scenario model specification and hyperparameter tuning.   We here adopt Bayesian nonparametric (BNP) mixture models to remove this requirement, extending TS to arbitrary reward distributions. We model each arm's unknown reward distribution with an independent BNP Gaussian mixture, which can approximate any continuous reward density, and adapts its complexity to sequentially observed data. Embedding the BNP mixture model into TS yields Nonparametric TS, an algorithm that sequentially learns the reward model that best approximates the true, yet u
    
[^180]: 用于三维磁共振图像中基于学习的膝关节软骨分割的语义上下文森林

    Semantic Context Forests for Learning-Based Knee Cartilage Segmentation in 3D MR Images

    [https://arxiv.org/abs/1307.2965](https://arxiv.org/abs/1307.2965)

    本文提出一种基于语义上下文森林的迭代多类学习方法，通过利用骨与软骨间的空间上下文约束以及解剖标志点距离特征，实现三维MR图像中股骨、胫骨和髌骨软骨的同时自动分割。

    

    从三维磁共振图像中自动分割人体膝关节软骨是一项有用且具有挑战性的任务，因为软骨呈薄层结构、边界弥散且强度不均匀。本文提出了一种迭代式多类别学习方法，可同时分割股骨、胫骨和髌骨软骨，该方法有效利用了骨骼与软骨之间以及不同软骨之间的空间上下文约束。首先，基于软骨仅生长于相应骨表面特定区域这一事实，我们不仅提取到骨表面的距离特征，更有意义的是，还提取到骨表面上密集配准的解剖标志点的距离特征。其次，我们引入了一组迭代式判别分类器，在每次迭代中，由先前学习得到的分类器所导出的类别置信度图构建概率比较特征……

    arXiv:1307.2965v3 Announce Type: replace-cross  Abstract: The automatic segmentation of human knee cartilage from 3D MR images is a useful yet challenging task due to the thin sheet structure of the cartilage with diffuse boundaries and inhomogeneous intensities. In this paper, we present an iterative multi-class learning method to segment the femoral, tibial and patellar cartilage simultaneously, which effectively exploits the spatial contextual constraints between bone and cartilage, and also between different cartilages. First, based on the fact that the cartilage grows in only certain area of the corresponding bone surface, we extract the distance features of not only to the surface of the bone, but more informatively, to the densely registered anatomical landmarks on the bone surface. Second, we introduce a set of iterative discriminative classifiers that at each iteration, probability comparison features are constructed from the class confidence maps derived by previously learned
    
[^181]: 用于留一法交叉验证的集中不等式

    Concentration inequalities for leave-one-out cross validation. (arXiv:2211.02478v3 [math.ST] UPDATED)

    [http://arxiv.org/abs/2211.02478](http://arxiv.org/abs/2211.02478)

    本文证明了估计器的稳定性足以说明留一法交叉验证是可靠的，并通过提供集中界限超出Lipschitz连续性假设的损失函数或估计器，为我们提供了一个相对丰富的分布类。

    

    在本文中，我们证明了估计器的稳定性足以说明留一法交叉验证是一个可靠的过程，通过在一个通用的框架中提供集中界限。特别是，我们在损失函数或估计器上提供了超过Lipschitz连续性假设的集中界限。我们通过依赖具有满足对数Sobolev不等式的分布的随机变量来获得我们的结果，这为我们提供了一个相对丰富的分布类。我们通过考虑几个有趣的例子来说明我们的方法，包括线性回归，核密度估计以及稳定/截断估计器，例如稳定的核回归。

    In this article we prove that estimator stability is enough to show that leave-one-out cross validation is a sound procedure, by providing concentration bounds in a general framework. In particular, we provide concentration bounds beyond Lipschitz continuity assumptions on the loss or on the estimator. We obtain our results by relying on random variables with distribution satisfying the logarithmic Sobolev inequality, providing us a relatively rich class of distributions. We illustrate our method by considering several interesting examples, including linear regression, kernel density estimation, and stabilized/truncated estimators such as stabilized kernel regression.
    

