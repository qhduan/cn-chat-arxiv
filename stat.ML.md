# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Why Forget-Only Unlearning Needs Memorization](https://arxiv.org/abs/2610.10519) | 本文证明仅遗忘式机器遗忘（只用训练模型和待遗忘样本、无保留数据）并非总是可行，其可行性取决于学习方法，且算法必须对训练数据进行足够记忆才能处理任意删除请求。 |
| [^2] | [Oracle-Efficient and Parameter-Free Agnostic Smoothed Online Learning](https://arxiv.org/abs/2610.10499) | 该论文提出了不可知平滑在线学习领域首个神谕高效且无参数的算法，同时摆脱了对基础测度采样访问能力和完美预测标签这两个限制性假设的依赖。 |
| [^3] | [Best Arm Identification for Bandits with Shifting Means](https://arxiv.org/abs/2610.10488) | 该论文针对均值会对抗性漂移但奖励差距保持稳定的新型老虎机环境，提出了重要性加权算法ISM，证明了传统基于广义似然比检验的算法（如Track-and-Stop）在此环境下会失效，而ISM能保持δ-正确性并获得良好的样本复杂度保证。 |
| [^4] | [Two-Level Softmax Sampling Done Right: Correcting Bias from Size Imbalance and Dispersion](https://arxiv.org/abs/2610.10483) | 本文揭示了双层softmax采样因忽略簇规模不均衡和簇内相似度离散度而产生的系统性采样偏差，并提出了S-2LS和SD-2LS两种修正方法，以几乎零额外计算开销实现了更优的softmax近似采样。 |
| [^5] | [Derivative Gaussian Processes on a Two-Direction Budget](https://arxiv.org/abs/2610.10428) | 本文提出一种每个观测梯度仅需两个方向的导数高斯过程，在Vecchia近似下将 $md$ 个梯度坐标压缩为至多 $2m$ 个方向导数，使每个预测目标的计算代价降至 $\mathcal{O}(m^3)$，并给出了近似误差的理论界。 |
| [^6] | [Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds](https://arxiv.org/abs/2610.10411) | 本文将投机解码建模为马尔可夫奖励过程，提出直接最小化期望解码轮数（EDR）的训练目标，以优化并行投机草稿模型的全局解码效率。 |
| [^7] | [Safe Meta-Policy Design with Risk Control](https://arxiv.org/abs/2610.10393) | 该论文提出一种带风险控制的离线元策略设计方法，在更新退化风险预算约束下通过动态规划规划模型更新时机，并揭示策略改进的信噪比是决定更新频率与风险分配的关键因素。 |
| [^8] | [Koopman Observers for Diffusion Acceleration: Correcting Feature Forecasts with Shallow Measurements](https://arxiv.org/abs/2610.10366) | 提出一种观测修正的库普曼框架，在不改变模型参数的前提下，利用即时计算的浅层特征观测来修正深层特征的库普曼预测，并配合周期性完整评估刷新观测器，从而加速冻结扩散模型的采样。 |
| [^9] | [Dataset Pruning from First Principles: A Label-Free Linear Programming Approach](https://arxiv.org/abs/2610.10347) | 提出了一种从第一性原理推导的数据集剪枝方法，将无偏子集选择表述为方差最小化的线性规划问题，无需标签和几何邻近假设即可选出具有代表性的训练子集。 |
| [^10] | [Data Reuse in Non-Stationary Learning](https://arxiv.org/abs/2610.10340) | 提出了暴露上限复用（ECR）类算法，通过结合在线变化检测、兼容性测试和污染控制来安全复用历史数据，使非平稳在线学习的遗憾值随不同取值数量而非变化次数增长，类似于偏差-方差权衡。 |
| [^11] | [Revisiting Explainable AI through Model-Independent Concept Dictionaries](https://arxiv.org/abs/2610.10301) | 提出DictXAI方法，通过在输入域中用包含可解释含义的词典定义概念，将输入的稀疏编码与模型预测归因到具体词典元素，从而实现跨模型、架构无关且可操作的可解释AI解释。 |
| [^12] | [Shared Gaussianization: What Gaussian Regularizers Certify About Contrastive Learning, and What They Miss](https://arxiv.org/abs/2610.10299) | 本文提出共享高斯化（SG）检验，证明该高斯正则化器能以紧的、与维度无关的平方根速率上界总体 InfoNCE 的超出量，并通过单次检验同时检测视图的失配与非均匀性。 |
| [^13] | [Neural Sampling with Reweighted Normalizing Flows via the Wasserstein--Fisher--Rao JKO Scheme](https://arxiv.org/abs/2610.10278) | 提出了一种基于WFR JKO格式的神经采样算法，首次证明其在任意固定步长下、无需对数凹性等结构性假设即可指数收敛到目标分布，并利用重加权归一化流对其输运与反应分量进行神经参数化实现。 |
| [^14] | [Finite-Sample Approximation of Hessian-Guided Perturbed Wasserstein Gradient Flows](https://arxiv.org/abs/2610.10218) | 该论文证明了海森引导扰动Wasserstein梯度流的有限粒子逼近在增长时间尺度上的高概率追踪界，关键在于沿参考路径累积的曲率——负曲率会放大误差而正曲率可抑制误差，从而刻画了暂时不稳定仍可精确追踪的有利情形。 |
| [^15] | [RoBART: Bayesian Additive Regression Trees with Tree-Specific Rotations](https://arxiv.org/abs/2610.10214) | RoBART通过为每棵树分配特定的Givens旋转来改进贝叶斯可加回归树，使其能高效逼近与预测变量轴不对齐的边界，并对具有各向异性Hölder光滑性的可加函数证明了后验收缩率。 |
| [^16] | [Computations of the slice genus and the unknotting number of links via machine learning](https://arxiv.org/abs/2610.10206) | 该论文利用强化学习和贝叶斯优化，为链环的切片亏格、解结数等难以算法计算的不变量求出新上界，并结合已知下界在许多情形下得到新的精确值，还重现了解结数的非可加性反例。 |
| [^17] | [Broadly Applicable Approximate MCMC for Switching Stochastic Differential Equations Using Uniformization and Time-Conditioned Factorized Neural Likelihood Estimation](https://arxiv.org/abs/2610.10194) | 该论文提出了一种结合均匀化与时间条件化因子分解神经似然估计的近似MCMC采样器，突破了现有方法在噪声观测、状态维度、漂移形式和扩散项等方面的限制，实现了对切换随机微分方程广泛适用的贝叶斯推断。 |
| [^18] | [Universal Local Error and Realized Amplification for the First-Order EDM Predictor](https://arxiv.org/abs/2610.10190) | 该论文证明了一阶EDM扩散采样器的单步局部离散化误差具有与数据分布无关的普适二次上界，并通过在高噪声水平利用显式收缩准则、在低噪声水平引入可远小于最坏Lipschitz常数的实际放大效应，最终获得了O(e^{Λ_K}/K)的全局离散化误差保证。 |
| [^19] | [Kinetic Langevin Meets Split Gibbs: Accelerated Posterior Sampling for Imaging Inverse Problems with Diffusion Priors](https://arxiv.org/abs/2610.10187) | 该论文提出RED-KLwSGS方法，将欠阻尼（动力学）朗之万扩散与分裂吉布斯采样框架结合，利用单次去噪得分驱动辅助变量更新，在与Langevin-within-SGS相同的每次迭代成本下实现成像逆问题后验采样的加速，并给出了强对数凹先验下连续和离散时间的非渐近Wasserstein-2收敛保证。 |
| [^20] | [Pre-training of Bayesian Optimization Algorithm through Bayesian Optimization](https://arxiv.org/abs/2610.10186) | 该论文提出了一种通过在高斯过程样本路径上运行贝叶斯优化来最小化期望累积遗憾，从而利用另一个贝叶斯优化过程自动预训练贝叶斯优化算法参数的框架。 |
| [^21] | [Conformal Prediction for Spatially Dependent Data via Sequential Whitening](https://arxiv.org/abs/2610.10168) | 该论文提出一种通过对校准残差进行顺序条件化（顺序白化）的保形预测方法，解决了空间相关数据下可交换性假设失效、且仅由校准残差可预测的空间变异残留导致区间效率下降的问题，在正确的工作协方差与椭圆残差分布下实现精确的有限样本覆盖率，并可借助最近邻近似扩展到大型网络。 |
| [^22] | [m-Set Adversarial Bandits with Winner Feedback](https://arxiv.org/abs/2610.10128) | 本文研究了不同效用和反馈模型下m集对抗性多臂老虎机的遗憾界，其主要技术贡献是遗憾值的信息论下界，揭示了环境设置中的细微变化会对学习速率产生巨大影响。 |
| [^23] | [Towards Calibrated Probabilistic Forecasts for Events of Interest via Outcome-Conditional Recalibration](https://arxiv.org/abs/2610.10076) | 本文提出了一种简单易实现的事后重校准方法——结果条件重校准，能够在用户定义的结果空间区域（如极端事件）上对概率预测进行重校准，从而确保决策者最关注的事件也能获得校准良好的预测。 |
| [^24] | [Transition Path Sampling Using Koopman Operators and Exit-Time Optimal Control](https://arxiv.org/abs/2610.10054) | 提出一种基于Koopman算子的转移路径采样新方法，利用其线性性质在无需转移路径数据的情况下识别亚稳态集合并估计committor函数，同时将TPS表述为退出时间最优随机控制问题，从而解决了现有神经网络方法的计算开销与性能保证问题。 |
| [^25] | [Gaussian Equivalence for Multi-Head Self-Attention](https://arxiv.org/abs/2610.10033) | 利用随机矩阵理论建立了多头自注意力的高斯等价性，证明用缩放分数加高斯噪声替代softmax注意力可保持中心化输出的极限谱定律，从而分离了头分配与投影宽度的影响。 |
| [^26] | [Controlling Dependence in Implicit Generative Models via Spread Mutual Information](https://arxiv.org/abs/2610.10021) | 提出扩散互信息（SMI），通过对生成变量施加扩散核并跨噪声级别对互信息进行加权积分，克服了隐式生成模型中奇异分布缺少得分函数及密度比估计重叠性差的难题，实现对统计依赖的有效控制。 |
| [^27] | [Extreme Binary Classification: Extreme Value Theory for Extreme Constraint on False Negative](https://arxiv.org/abs/2610.09984) | 本文提出“极端二分类”新问题，并基于极值理论设计了阈值自适应方法与基于置换检验的特征选择程序，使分类器的假负类率以快于 $1/N_1$ 的速率趋近于零，实验表现优于最先进方法。 |
| [^28] | [Possibilistic Radial Transport for Approximate IM Inference](https://arxiv.org/abs/2610.09956) | 提出一种可能性径向传输方法，将参数的可能性轮廓值编码到源点半径中，并结合深度学习算法实现高效的近似可能性推断模型推断，使覆盖率评估、功效分析和新数据预测检验变得切实可行。 |
| [^29] | [Expected Sample Complexity in Multi-Armed Bandits](https://arxiv.org/abs/2610.09929) | 本文提出了期望近似正确（ACE）新框架来研究多臂老虎机的期望样本复杂度，证明ACE保证蕴含几乎必然收敛到最优期望奖励，并揭示了确定性算法无法获得良好ACE界这一特性，同时针对次优水平ε已知与未知两种情形分析了随机算法。 |
| [^30] | [Eigenvalues of the Hessian in Deep Learning: The Origin of Symmetry and Its Breaking](https://arxiv.org/abs/2610.09919) | 本文提出，深度学习中训练模型Hessian特征值呈现的“零附近大块+孤立离群值”的谱结构，源于相对一个隐藏的高度对称参考构型的对称破缺——该参考构型的Hessian具有权重对称性之外的不变性，其对称破缺产生了观测到的谱层级结构。 |
| [^31] | [Outperformance Inverse Optimization: Learning Objective Functions that Outperform Agent Decisions](https://arxiv.org/abs/2610.09890) | 本文提出“超越性逆优化”新范式，不再复现智能体的次优决策，而是学习能诱导出在各分量上都优于观测决策的最优解的目标函数权重，并配套给出了适用于混合整数线性规划的预言机损失函数、梯度与DC优化算法以及泛化误差理论保证。 |
| [^32] | [Identifiability of a dissipative knowledge-dynamics model: exact recovery under designed excitation, degeneration on observational data](https://arxiv.org/abs/2610.09889) | 该论文将人类学习建模为参数具有机制含义的耗散常微分方程组，证明了在设计激励条件下模型参数可从数据中被精确恢复（双概念情形有闭式解），而在仅依赖观测数据时可辨识性会退化，并提出了数值精确等价且速度大幅提升的半隐式L-稳定批量求解器。 |
| [^33] | [Sparsifying Stochasticity, Not Capacity: Partial Stochasticity via Deep Weight Factorization of Prior Scales](https://arxiv.org/abs/2610.09886) | 该论文提出通过对先验尺度进行深度权重因子分解来学习贝叶斯神经网络中哪些参数应保持随机性，使正则化稀疏化随机性而非模型容量，并提供了可线性时间检验的通用条件密度逼近证书，同时证明常见的采样-优化混合方案是II型最大后验目标的随机近似。 |
| [^34] | [AdaPS-LiNGAM: Adaptive Predecessor Selection for Linear Non-Gaussian Acyclic Models under Small-Sample Settings](https://arxiv.org/abs/2610.09782) | 本文揭示了DirectLiNGAM在变量数超过样本量时残差化必然退化的结构性局限，并提出利用由图结构决定的“活动边界”子集进行自适应前驱选择的AdaPS-LiNGAM方法，以实现小样本情形下可靠的因果发现。 |
| [^35] | [Fluctuations of Nonlinear Observables in Mean Field Neural Network Training](https://arxiv.org/abs/2610.09768) | 本文通过在加权Sobolev空间中应用仅需普通Fréchet可微性的泛函Delta方法（无需Lions导数），证明了均场神经网络训练中的涨落会传播到有限维非线性观测量，并建立了相应的中心极限定理与显式协方差表示。 |
| [^36] | [Leaner Transformers Can Easily Learn to Cluster](https://arxiv.org/abs/2610.09760) | 本文提出了一种嵌入维度仅需 d+⌈log₂k⌉ 但表达能力不变的更精简Transformer来执行k均值聚类的Lloyd算法，并系统刻画了训练Transformer学习聚类算法时影响收敛性与泛化能力的关键因素。 |
| [^37] | [EntroPrefill: Renyi-Guided Context Pruning with Conditional Stability Guarantees for Retrieval-Augmented Generation](https://arxiv.org/abs/2610.09757) | 该论文提出 EntroPrefill，一种由 Renyi 熵引导、带显式注意力质量约束的预填充中期上下文剪枝方法，为检索增强生成提供了可计算的 token 删除上界、自适应剪枝层下依然有效的有限样本观测保证，以及带有显式 Lipschitz 常数的条件 Transformer 扰动稳定性界。 |
| [^38] | [Unbounded Characteristic and Universal Kernels](https://arxiv.org/abs/2610.09731) | 本文系统研究了无界核的特征性与万能性等表达能力概念，将针对有界核的成熟理论推广至无界核情形。 |
| [^39] | [The Silhouette Operator: Identifiability of Low-Rank Measures from One-Dimensional Projections](https://arxiv.org/abs/2610.09687) | 本文提出“轮廓算子”框架，证明了适当选取的 2k 个一维投影边缘分布足以唯一识别 $\mathbb{R}^2$ 上任何紧支撑的秩不超过 k 的符号测度，且该数量是最优的、投影方向不能任意选取。 |
| [^40] | [Gauss-Newton Accuracy and Indefinite Hessians: Uniform Coexistence in Low-Cost Sets](https://arxiv.org/abs/2610.09675) | 本文证明在岭正则化非线性最小二乘中，低成本集合内一致共存两种曲率状态：每个全局极小值点的海森矩阵相对误差低于 $(1+\sqrt{2})/8$，而同一集合中也存在海森矩阵不定、相对误差至少为 $15/8$ 的点，并给出逐点证书与尖锐的相对误差界。 |
| [^41] | [Cluster-Robust Prediction-Powered Inference](https://arxiv.org/abs/2610.09601) | 本文提出聚类稳健的PPI++方法，能够处理聚类内存在任意依赖性的部分标注数据，以闭式解形式提供标准误并构建渐近有效的置信区间，无需任何重抽样技术。 |
| [^42] | [Scalable Logistic Gaussian Process Density Regression with Kinetic Langevin Sampling](https://arxiv.org/abs/2610.09591) | 本文提出一种基于逻辑高斯过程的可扩展贝叶斯条件密度估计方法，通过在强对数凹后验上模拟动力学朗之万动力学进行直接采样，并利用Nyström特征支持具有依赖输入参数的非平稳核。 |
| [^43] | [Certified by Abstention: Distribution-Free Guarantees for Chain-of-Thought Verifiers at Small Calibration Budgets](https://arxiv.org/abs/2610.09541) | 该研究揭示“通过弃权实现有效性”现象——很少触发的验证证书虽形式上有效但每次触发时可能全部出错，并据此为小校准预算下的思维链验证器建立了无分布认证保证及其失效条件分析。 |
| [^44] | [Unpaired Canonical Correlation Analysis](https://arxiv.org/abs/2610.09530) | 提出UCCA方法，通过建立二次分配问题与CCA之间的理论联系，首次实现仅使用非配对数据学习最大化真实潜在配对相关性的线性投影。 |
| [^45] | [Reflected Anchored Langevin Algorithms](https://arxiv.org/abs/2610.09522) | 本文提出反射锚定朗之万动力学（RALD）及其蒙特卡洛算法 RALMC，通过光滑锚定参考势能与状态相关缩放因子，突破了传统方法要求对数密度可微的限制，实现了约束域上不可微目标分布的高效采样，并给出了显式的收敛界与迭代复杂度。 |
| [^46] | [Adjoint-Based Calibration and Optimal Control of Stochastic Multiscale Bioprocess Digital Twins](https://arxiv.org/abs/2610.09505) | 本文提出了一个基于伴随敏感性分析的偏差感知数字孪生校准与最优控制框架，通过拟似然估计、矩展开和正反向伴随方法量化校准不确定性对策略性能的传播影响，实现了多尺度生物过程的不确定性感知策略优化与自适应实验设计。 |
| [^47] | [DSReg: Provably Recovering Individual World Latents without Reconstruction](https://arxiv.org/abs/2610.09457) | 该论文提出“结构多样性”条件与依赖稀疏正则化方法DSReg，在无需重建、解码器或标签的情况下，可证明地恢复个体的世界潜在变量。 |
| [^48] | [Finite-Rank Logistic Gaussian Processes with Exact Likelihood for Conditional Density Estimation](https://arxiv.org/abs/2610.09452) | 提出ExFR-LGP方法，通过在规则网格上分段线性的有限秩高斯过程先验，使逻辑高斯过程的归一化常数具有闭式解，从而实现了条件密度估计的精确似然贝叶斯推断。 |
| [^49] | [Global Exponential Convergence of Two-Layer Linear Network Training](https://arxiv.org/abs/2610.09356) | 该论文证明了采用光滑PL预测损失训练的宽两层线性网络的全局指数收敛性，其动力学可精确刻画为神经元协方差的有限维Bures流，并给出显式收敛速率（初始协方差为σ²Id时至少为4σ²κ），且该速率在有限宽度采样下保持稳定。 |
| [^50] | [Benign Overfitting under Heterogeneous Input Fusion](https://arxiv.org/abs/2610.09340) | 该论文首次研究异构输入融合下的良性过拟合，发现回归中存在一个与截断阈值无关的全谱协方差证书，可保证良性性质在任意一致的联合协方差融合下得以保持，但这种保护是精确的——在证书范围之外，两个良性的边缘输入块融合后可能变得有害。 |
| [^51] | [The Geometry of Anisotropic Dilation for Optimal Regularization](https://arxiv.org/abs/2610.09310) | 本文提出“各向异性膨胀”这一保持方向的径向缩放变换，并证明其能以显式方式传递性地调控最优正则化中的径向统计量，从而通过数据变换实现对正则化子几何结构的完全控制与自适应适配。 |
| [^52] | [The Symbol of the Surrogate: Measuring Numerical Provenance in Neural PDE Solvers](https://arxiv.org/abs/2610.09255) | 该论文提出一种基于傅里叶符号的经验诊断方法，用于判定神经PDE代理模型究竟忠实于精确物理演化还是仅仅模仿训练数值求解器的离散化误差，并发现代理模型几乎完全复制（超过99.8%）了训练格式的振幅与相位误差特征。 |
| [^53] | [Efficient Best-of-N policy evaluation for inference-time alignment](https://arxiv.org/abs/2610.09250) | 本文提出了一种无需访问响应似然值的仅样本BoN策略评估框架，利用BoN的顺序统计结构将密度比转化为可由样本估计的得分排名概率，并开发了能跨候选预算高效重用共享辅助样本池的双重稳健估计器BoN-DR，在奖励模型误设下仍保证有效的渐近推断。 |
| [^54] | [Sketched Calibration for Conformal Prediction under Covariate Shift](https://arxiv.org/abs/2610.09208) | 该论文提出草图化校准方法，通过对协变量压缩后再进行加权共形预测，在不增加偏移相关校准代价的前提下校正协变量偏移，并给出了以“泄漏量”刻画的覆盖率保证。 |
| [^55] | [An Accuracy--Information Tradeoff for Loss-Difference Conditional Mutual Information](https://arxiv.org/abs/2610.09206) | 论文证明了精度与信息之间的权衡：在逻辑损失等光滑凸损失及幂次正则化条件下，任何以最优样本量达到低超额风险的正规学习器，其最坏情况损失差条件互信息必然达到 n 比特量级。 |
| [^56] | [Exact Dynamics and Finite-Sample Trajectory Recovery of Linear Recursive Feature Machines](https://arxiv.org/abs/2610.09196) | 本文将线性递归特征机与迭代重加权最小二乘的联系推广到岭正则化含噪多输出回归，并证明了其学习到的特征矩阵在每次迭代中都以 $O(\sqrt{d/n})$ 的误差速率逼近无限数据下的理想结果。 |
| [^57] | [Lower Bounds for Parallel Diffusion Sampling](https://arxiv.org/abs/2610.09166) | 本文首次建立了带近似分数的扩散采样的多项式并行轮数下界，证明了对 $R^d$ 中平滑近各向同性高斯混合的采样需要 $\widetilde{\Omega}(d^{1/3})$ 轮、对单位球内各向异性轴对齐盒子的均匀采样需要 $\Omega(d)$ 轮，且这些下界对每轮可进行多项式次查询的任意随机算法均成立。 |
| [^58] | [Relative Wasserstein Spatial Depth for Cluster Number Selection in Distributional Data](https://arxiv.org/abs/2610.09153) | 提出相对Wasserstein空间深度（RWSD）作为分布数据聚类中选择最优簇数的新准则，并与Wasserstein K-means和K-medians算法结合，在理论上证明了聚类中心和所选簇数的一致性。 |
| [^59] | [The Impact of Likelihood Tempering on the Limiting Predictive Moments of Variational Bayesian Linear Neural Networks](https://arxiv.org/abs/2610.09132) | 本文针对宽贝叶斯神经网络中的“先验主导”退化问题，推导了在温度调度T = τ/M^c下变分贝叶斯线性神经网络的极限预测分布，并揭示了似然温度调节与NNGP后验之间的关系。 |
| [^60] | [FedRSPO+: A Heterogeneity-aware Algorithm for Decision-focused Federated Learning](https://arxiv.org/abs/2610.09091) | 提出了异构感知的决策导向联邦学习框架FedRSPO+，其核心是基于通过投影平滑决策映射的正则化代理RSPO+，为决策误差和遗憾提供理论上界，从而解决联邦场景下下游目标与可行集异构性导致的训练不稳定问题。 |
| [^61] | [Covariate-dependent Joint Modeling of Multivariate Ordinal Preferences and Its Connections with Comparison Models](https://arxiv.org/abs/2610.09070) | 本文提出了一种协变量依赖的多元序数偏好联合建模方法，避免了传统方法单独处理各属性或将数据粗化为胜负比较所造成的信息损失，并建立了该方法与 Bradley–Terry、Plackett–Luce 等比较模型之间的联系。 |
| [^62] | [Learning Transition Kernels of Jump-Diffusion Processes with Conditional Diffusion Models](https://arxiv.org/abs/2610.09045) | 该论文提出用条件扩散模型学习时齐跳跃扩散过程的转移核，在理论上给出了条件分数估计误差和真实与生成路径分布间KL散度的非渐近界，并在合成与真实数据上验证了其在样本路径生成和概率预测任务中的有效性。 |
| [^63] | [Quadratic Weak-to-Strong Generalization in Random Feature Networks via Random Matrix Theory](https://arxiv.org/abs/2610.09044) | 本文利用随机矩阵理论证明，在两层随机特征网络中，由弱教师模型训练出的强学生模型误差为教师误差的平方，实现了二次级的弱到强泛化改进。 |
| [^64] | [Careful Judge: Safe and Efficient Human-AI Collaborative Decision Making](https://arxiv.org/abs/2610.09043) | CARE是一个端到端的人机协作决策框架，通过新颖的自适应校准模块在任何时刻保证风险控制，并持续从人工反馈中学习，以更少的人工查询实现更高的自动化水平。 |
| [^65] | [Graph-monotone entrywise guarantees for MLE and Rank Centrality on general comparison graphs](https://arxiv.org/abs/2610.09030) | 本文在Bradley--Terry--Luce模型下证明，MLE和Rank Centrality在任意固定比较图上均能达到以计数加权图代数连通度决定的逐项误差保证，且该保证随比较数据的增加单调改善。 |
| [^66] | [LASER: Latent Space Adjoint Matching for Support-Constrained Entropy-Regularized Offline RL](https://arxiv.org/abs/2610.08989) | LASER通过潜在空间伴随匹配实现熵正则化的潜在空间离线强化学习，既防止了策略坍缩成单一脆弱模式，又避免了时间反向传播，在40个不同数据质量的OGBench任务上取得了优异表现。 |
| [^67] | [The Best Optimizer Depends on Batch Size](https://arxiv.org/abs/2610.08975) | 该论文挑战了“某一批量大小下最佳的优化器在其他批量大小下也最佳”的常见假设，证明Muon缺乏一致的缩放规则，且即使经过大量超参数调优，语言模型预训练的最佳优化器仍会随批量大小而改变。 |
| [^68] | [Work While They Sleep: Exploiting Evaluation Latency for Fully Bayesian Optimization](https://arxiv.org/abs/2610.08969) | 该论文提出ELF-BO算法，巧妙利用贝叶斯优化中昂贵目标函数评估期间的等待时间，提前并行计算全贝叶斯代理模型，从而在不增加额外时间成本的情况下获得更好的不确定性估计和优化性能。 |
| [^69] | [Trust-Region Optimization for Smooth Potential-Interaction Energies in Wasserstein Space](https://arxiv.org/abs/2610.08883) | 该论文提出了Wasserstein空间上光滑势-相互作用能量的信赖域优化方法，通过推前曲线上的二次模型、$L^2(\rho)$ 步长半径以及带显式自伴二阶变分算子的Steihaug-Toint子求解器，在温和条件下证明了目标函数单调不增且Wasserstein梯度范数收敛于零。 |
| [^70] | [Slow Beats Fast at the Kesten-Stigum Threshold: Minimax, Fisher-Information and Belief-Propagation Characterizations of the Information-Computation Gap in Sparse Stochastic Block Models](https://arxiv.org/abs/2610.08872) | 该论文通过统计决策理论、Fisher信息和置信传播对稀疏随机块模型的Kesten-Stigum阈值给出三种刻画，证明在 q≥5 时阈值下方存在信息-计算差距：多项式低度算法渐近无法超越平凡风险，而指数时间算法却能成功。 |
| [^71] | [Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States](https://arxiv.org/abs/2610.08818) | 该论文提出GenST框架，将未观测节点状态预测（FUNS）重新定义为时空图上的条件生成任务，创新性地利用微调后的大语言模型从节点描述中提取语义特征作为语义桥梁，以弥补缺失的时空信号。 |
| [^72] | [DeepAJM: Deep Association Joint Model for Irregularly Sampled data](https://arxiv.org/abs/2610.07388) | 提出 DeepAJM——一种无需参数假设的深度联合模型，利用编码器-解码器架构学习不规则采样的时变协变量轨迹的潜在结构，并通过部分可解释的关联结构将其与生存结局关联，从而改进生存预测。 |
| [^73] | [Sample-Optimal Estimation of the Fr\'echet Inception Distance](https://arxiv.org/abs/2610.07114) | 该论文针对FID估计中的有限样本偏差问题，证明了插件估计器的紧致偏差与方差界并确立其平方级（d²）样本复杂度，同时将FID∞估计器推广到任意阶外推方法以实现去偏估计。 |
| [^74] | [The Signed Geometry of One-Shot Recourse: On-Path Validity and the Signed-Curvature Criterion](https://arxiv.org/abs/2609.36252) | 该论文证明单次解析式反事实补救能否一步成功由路径曲率 $\kappa=\hat g^\top\nabla^2 f(x)\hat g$ 的符号决定（非负则有效），给出仅凭分数与梯度的规则不可避免存在 $Kd_p^2/\|\nabla f(x)\|$ 量级过冲的下界，并证明在利普希茨曲率下于承诺点评估一次分数即可达到极小极大最优有效性。 |
| [^75] | [Control-Geometry Straightening for Sampling-Based Latent Planning](https://arxiv.org/abs/2609.35603) | 提出控制几何拉直（CGS）这一辅助损失，通过将动作间余弦相似度与潜在差异对齐来学习对规划器友好的表示，从而提升基于采样的潜在规划的优化效率，并给出相应理论保证。 |
| [^76] | [Ordinary Nonconvex SGD under Distance-Dependent Moments: Finite-Horizon Stationarity and Nagaev Bounds](https://arxiv.org/abs/2609.30499) | 本文证明，当条件矩允许噪声方差随迭代点距离增长时，普通单样本SGD无需任何修改即可达到与Blum–Gladyshev下界匹配的极小极大随机复杂度，并借助希尔伯特空间Fuk–Nagaev不等式给出高概率Nagaev型界。 |
| [^77] | [On Large-Scale Multiple Testing Over Networks: A Non-Asymptotic Approach](https://arxiv.org/abs/2609.14170) | 该论文发现分布式多重检验中有限样本FDR失控源于赢者诅咒偏差，并提出交叉拟合贪心聚合算法（CFGA），通过数据分割实现了网络上分布式多重检验在有限样本下的严格FDR控制。 |
| [^78] | [Generalized Score Matching for Parameter Estimation on Convex Domains](https://arxiv.org/abs/2609.11521) | 本文从最小概率流学习出发，构造性地推导出凸域上的广义分数匹配目标函数，统一了经典分数匹配与非负数据的域适配变体，并证明该目标是二阶正当局部评分规则，保证最小化时能恢复真实密度。 |
| [^79] | [Linear Independent Component Analysis via Optimal Transport](https://arxiv.org/abs/2607.14081) | 本文提出以数据线性投影到标准高斯分布的平方 Wasserstein 距离作为 ICA 对比函数，并证明该距离在投影恰好恢复出独立成分时达到最大、且与任何真实混合信号之间都存在显式间隔，从而为线性独立成分分析建立了一种基于最优传输的新方法。 |
| [^80] | [Non-asymptotic Convergence of Stochastic Gradient Descent in Score-based Generative Models](https://arxiv.org/abs/2607.04775) | 本文研究了基于分数的生成模型训练中随机梯度下降的非渐近收敛保证，针对一般分数参数化给出了显式依赖损失加权和时间采样分布的非凸优化界，并为过参数化两层 ReLU 网络建立了神经正切核分析。 |
| [^81] | [In-Context Residual Calibration for Uncertainty Quantification of Energy Time Series over Graphs](https://arxiv.org/abs/2606.31804) | 该论文提出一种上下文残差校准方法，针对现有共形预测难以捕捉能源系统复杂时空结构的缺陷，为图上能源时间序列提供更可靠的不确定性量化，以支持风险感知的能源运营决策。 |
| [^82] | [Operator Calculus for Population-Based Optimization: Modular Convergence and Finite-Population Guarantees](https://arxiv.org/abs/2606.14289) | 本文提出一种面向基于种群优化的算子微积分框架，使经过独立验证的更新规则效应可以模块化地组合，为收敛性分析提供可复用的构建模块，并给出有限评估预算下的收敛保证。 |
| [^83] | [Automatic, Debiased, and Invariant Counterfactual Generation under General Interventions](https://arxiv.org/abs/2606.07399) | ADIGen框架通过结合Riesz回归、因果不变性和正交统计学习，实现了通用干预下自动、去偏且不变的反事实生成，并提供了双重稳健的风险控制保证。 |
| [^84] | [A prism hierarchy of learning regimes in large linear autoencoders](https://arxiv.org/abs/2606.05335) | 本文提出用三棱柱的层级结构系统地刻画大型权重绑定线性自编码器的五个基本极端学习区间（大数据、小数据、平均场、窄潜层、自由），为此类非线性于权重的模型的学习动态提供了系统化的理论图景。 |
| [^85] | [The Metagame of Interpretability and Meta-Attributions](https://arxiv.org/abs/2605.06295) | 提出“元博弈”框架，将特征的归因值视为特征间的合作博弈并计算其Shapley值，从而得到方向性元归因，使任意基于梯度或注意力的归因方法都能泛化到二阶交互效应，并证明了元归因之和恰好等于其所解释的一阶归因。 |
| [^86] | [BONSAI: Bayesian Optimization with Natural Simplicity and Interpretability](https://arxiv.org/abs/2602.07144) | 提出了一种感知默认配置的贝叶斯优化策略BONSAI，它能在显式控制采集价值损失的前提下剪除对默认配置的低影响偏离，从而实现更简洁、可解释且易于审查的优化推荐。 |
| [^87] | [Multiparameter Uncertainty Mapping in Quantitative Molecular MRI using a Physics-Structured Variational Autoencoder (PS-VAE)](https://arxiv.org/abs/2602.03317) | 提出一种物理结构变分自编码器（PS-VAE），通过融合可微分自旋物理模拟器与自监督学习，实现定量分子MRI中体素级多参数后验分布的快速提取与不确定性量化。 |
| [^88] | [Order-Optimal Sample Complexity of Rectified Flows](https://arxiv.org/abs/2601.20250) | 本文证明了整流流模型在标准神经网络假设下可达到 $\tilde{O}(\varepsilon^{-2})$ 的最优阶样本复杂度，改进了流匹配模型已有的 $O(\varepsilon^{-4})$ 界并匹配均值估计的最优速率。 |
| [^89] | [Modified Loss of Momentum Gradient Descent: Fine-Grained Analysis](https://arxiv.org/abs/2509.08483) | 该论文证明当步长足够小时，重球动量梯度下降在指数吸引的不变流形上精确等价于带修正损失的普通梯度下降，能以任意有限阶精度刻画该修正损失，并在其无记忆近似的组合结构中发现了介于欧拉多项式与Narayana多项式之间的一类新的β多项式族。 |
| [^90] | [Causal Posterior Estimation](https://arxiv.org/abs/2505.21468) | 提出因果后验估计（CPE）方法，将模型图结构中的条件依赖关系直接硬编码进基于流匹配的神经网络架构，在似然函数难以计算的模拟器模型中实现高精度的贝叶斯后验推断。 |
| [^91] | [Allocation Stability and Wald Inference under Variance-Aware UCB](https://arxiv.org/abs/2412.08843) | 本文为双臂方差感知UCB策略给出了最优臂分配稳定性的尖锐判据，并证明即使最优臂计数不稳定，只要拉取次数与奖励方差的乘积依概率发散，臂均值线性组合的Wald统计量仍渐近服从标准正态分布，从而表明分配稳定性并非高斯推断的必要条件。 |
| [^92] | [Combining additivity and active subspaces for high-dimensional Gaussian process modeling](https://arxiv.org/abs/2402.03809) | 本论文的贡献是将可加性和主动子空间与多重真实度策略结合，解决了高维高斯过程建模中的维度灾难问题，并通过实验证明了这些优势。 |

# 详细

[^1]: 为什么仅遗忘式机器遗忘需要记忆

    Why Forget-Only Unlearning Needs Memorization

    [https://arxiv.org/abs/2610.10519](https://arxiv.org/abs/2610.10519)

    本文证明仅遗忘式机器遗忘（只用训练模型和待遗忘样本、无保留数据）并非总是可行，其可行性取决于学习方法，且算法必须对训练数据进行足够记忆才能处理任意删除请求。

    

    机器遗忘要求设计一种删除算法，其输出接近于在没有被选中遗忘样本的情况下从头重新训练的结果。在这项工作中，我们研究仅遗忘式机器遗忘，即删除算法只接收训练好的模型和需要遗忘的样本，而不保留任何数据或额外的训练信息。我们探讨仅遗忘式遗忘是否总是可行的。我们首先证明这取决于学习方法：不同的数据集可以产生相同的训练模型，但在删除相同样本后却需要非常不同的输出。利用这一观察，我们推导了遗忘算法匹配重新训练效果的精度下界，并在几种标准学习算法上进行了实例化。接着我们问，当仅遗忘式遗忘成功时，必须满足什么条件。为此，我们推导了算法为处理任意删除请求而必须对训练数据进行记忆的下界。对于简单的阈值学习……

    arXiv:2610.10519v1 Announce Type: new  Abstract: Machine unlearning asks for a deletion algorithm whose output is close to retraining from scratch without the selected forget examples. In this work, we study forget-only unlearning, where the deletion algorithm receives only the trained model and the examples to forget, with no retained data or extra training information. We ask whether forget-only unlearning is always possible. We first show that this depends on the learning method: different datasets can produce the same trained model but require very different outputs after the same examples are removed. Using this observation, we derive lower bounds on how accurately unlearning can match retraining and instantiate them for several standard learning algorithms. We then ask what must be true when forget-only unlearning succeeds. To this end, we derive lower bounds on what an algorithm must memorize about the training data to handle arbitrary deletion requests. For simple threshold lea
    
[^2]: 神谕高效且无参数的不可知平滑在线学习

    Oracle-Efficient and Parameter-Free Agnostic Smoothed Online Learning

    [https://arxiv.org/abs/2610.10499](https://arxiv.org/abs/2610.10499)

    该论文提出了不可知平滑在线学习领域首个神谕高效且无参数的算法，同时摆脱了对基础测度采样访问能力和完美预测标签这两个限制性假设的依赖。

    

    在线学习在许多领域都是一个有吸引力的框架，因为即使数据是相关的或被对抗性选择的，它也能实现有明确定义的学习。然而，这种通用性伴随着高昂的代价，带来了显著的统计和计算障碍。最近，平滑在线学习作为一个有前景的框架应运而生，它在完全对抗性和完全随机性两种设定之间进行衔接，其假设每个协变量的条件分布相对于某个固定基础测度 $\mu$ 的密度至多为 $1/\sigma$，并且已知该框架能够达到与经典学习相当的统计和计算保证，同时仍保留了在线学习的诸多灵活性。然而，现有的神谕高效算法要么需要（i）对基础测度 $\mu$ 的采样访问能力，要么需要（ii）标签能够被某个固定假设完美预测。这两个假设都限制了这些算法的适用性……

    arXiv:2610.10499v1 Announce Type: new  Abstract: Online learning is an attractive framework in many domains because it permits well-defined learning even when data are dependent or chosen adversarially. This generality, however, comes at a steep price, introducing significant statistical and computational barriers. Recently, smoothed online learning has emerged as a promising framework that interpolates between the fully adversarial and fully stochastic settings by assuming that the conditional law of each covariate has density at most $1/\sigma$ with respect to some fixed base measure $\mu$, and it is known to match the statistical and computational guarantees of classical learning while still allowing for much of the flexibility of online learning. However, existing oracle-efficient algorithms require either (i) sampling access to the base measure $\mu$ or (ii) labels that are perfectly predicted by a fixed hypothesis. Both assumptions limit the applicability of these algorithms, in 
    
[^3]: 均值漂移老虎机的最优臂识别问题

    Best Arm Identification for Bandits with Shifting Means

    [https://arxiv.org/abs/2610.10488](https://arxiv.org/abs/2610.10488)

    该论文针对均值会对抗性漂移但奖励差距保持稳定的新型老虎机环境，提出了重要性加权算法ISM，证明了传统基于广义似然比检验的算法（如Track-and-Stop）在此环境下会失效，而ISM能保持δ-正确性并获得良好的样本复杂度保证。

    

    我们研究了在具有一种新型对抗性扰动的随机环境中的最优臂识别问题，我们将这种扰动命名为“均值漂移”。在经典设定中，K个臂的平均奖励随时间保持稳定，而在均值漂移设定下，只有各臂平均奖励之间的差距Δ保持稳定，而它们的共同漂移量可能在每一轮被对抗性地决定。学习者的目标是在最小化样本复杂度的同时，以高概率识别出最优臂（固定置信度设定）。处理这种漂移需要新的工具：我们证明了采用广义似然比检验（GLRT）停止规则的算法，包括流行的Track-and-Stop算法，在时变漂移下会失效。因此，我们提出了针对均值漂移的重要性加权算法（ISM）。在假设均值以U为界且奖励服从σ²-次高斯分布的条件下，我们证明了ISM具有δ-正确性，并享有良好的样本复杂度保证。

    arXiv:2610.10488v1 Announce Type: cross  Abstract: We study the best arm identification problem in a stochastic environment with a novel form of adversarial perturbations, which we coin Shifting Means. While classically the mean rewards of the $K$ arms are stable in time, in Shifting Means only the gaps $\boldsymbol{\Delta}$ between mean rewards are stable, while their common shift may be determined adversarially in each round. The objective of the learner is to identify the best arm with high probability while minimizing sample complexity (the fixed confidence setting). Handling shifts requires new tools: we show that algorithms employing a Generalized Likelihood Ratio Test (GLRT) stopping rule, including the popular Track-and-Stop, fail under time-varying shifts. Instead, we propose Importance Weights for Shifting Means ($\mathsf{ISM}$). Assuming means bounded by $U$ and $\sigma^2$-sub-Gaussian rewards, we show $\mathsf{ISM}$ to be $\delta$-correct and to enjoy a sample complexity bo
    
[^4]: 正确的双层Softmax采样：修正来自簇规模不均衡与离散度的偏差

    Two-Level Softmax Sampling Done Right: Correcting Bias from Size Imbalance and Dispersion

    [https://arxiv.org/abs/2610.10483](https://arxiv.org/abs/2610.10483)

    本文揭示了双层softmax采样因忽略簇规模不均衡和簇内相似度离散度而产生的系统性采样偏差，并提出了S-2LS和SD-2LS两种修正方法，以几乎零额外计算开销实现了更优的softmax近似采样。

    

    从softmax分布中采样是机器学习中的一项基础操作，但其相对于项目数量的线性复杂度使得精确采样在大规模场景下难以实际应用。双层softmax（2LS）采样是一种流行的替代方案，可实现亚线性时间采样。该方法假设项目被划分为若干簇，2LS首先采样一个簇，然后在该簇内采样一个项目。在本文中，我们表明，尽管具有优势，2LS会引入系统性的不良采样偏差，这些偏差源于对簇的错误加权，即同时忽略了簇规模的不均衡性和簇内相似度的离散度。我们提出了两种采样方法：规模修正的2LS（S-2LS）以及规模与离散度修正的2LS（SD-2LS），它们修正了这些偏差，并以微乎其微甚至为零的额外计算开销提供了可证明更优的softmax近似。在五个大规模数据集上的深入实验验证了我们方法改进后的采样特性。

    arXiv:2610.10483v1 Announce Type: new  Abstract: Sampling from a softmax distribution is a fundamental operation in machine learning, but its linear complexity in the number of items makes exact sampling impractical at scale. Two-level softmax (2LS) sampling is a popular alternative enabling sublinear-time sampling. Assuming items are partitioned into clusters, 2LS first samples a cluster and then an item within it. In this paper, we show that, despite its advantages, 2LS introduces systematic and undesirable sampling biases, which arise from misweighting clusters by ignoring both cluster size imbalance and intra-cluster similarity dispersion. We propose two sampling methods, Size-Corrected 2LS (S-2LS) and Size- and Dispersion-Corrected 2LS (SD-2LS), which correct these biases and provide provably better softmax approximations with negligible to non-existent computational overhead. In-depth experiments on five large-scale datasets validate the improved sampling properties of our method
    
[^5]: 双方向预算下的导数高斯过程

    Derivative Gaussian Processes on a Two-Direction Budget

    [https://arxiv.org/abs/2610.10428](https://arxiv.org/abs/2610.10428)

    本文提出一种每个观测梯度仅需两个方向的导数高斯过程，在Vecchia近似下将 $md$ 个梯度坐标压缩为至多 $2m$ 个方向导数，使每个预测目标的计算代价降至 $\mathcal{O}(m^3)$，并给出了近似误差的理论界。

    

    梯度观测有望带来更精确的高斯过程（GP）代理模型，但引入梯度观测的代价长期以来一直阻碍着这一前景的实现。我们提出了一种导数高斯过程，其每个观测梯度的预算仅为两个方向。其中一个方向关注每个梯度对目标预测的直接贡献，另一个方向则通过与条件函数值的相关性来聚合其间接贡献。在Vecchia近似框架下，每次预测以 $d$ 维空间中 $m$ 个邻近输入为条件，该构造最多使用 $2m$ 个方向导数来表示其 $md$ 个梯度坐标，使每个预测目标的稠密分解代价为 $\mathcal{O}(m^3)$。对于一般的条件集，我们给出了相对于使用完整梯度的后验近似误差界，并刻画了误差较小或近似精确的条件。在仿真实验中，我们的方法与……（原文截断）

    arXiv:2610.10428v1 Announce Type: cross  Abstract: Gradient observations promise more accurate Gaussian process (GP) surrogates, but the cost of incorporating them has long stood in the way of realizing that promise. We propose a derivative GP with a budget of just two directions per observed gradient. One direction focuses on each gradient's direct contribution to target prediction, while the other aggregates its indirect contributions through correlations with the conditioning function values. Within a Vecchia approximation, where each prediction conditions on $m$ nearby inputs in $d$ dimensions, this construction represents their $md$ gradient coordinates using at most $2m$ directional derivatives, giving $\mathcal{O}(m^3)$ dense factorization cost per prediction target. For general conditioning sets, we bound the posterior approximation error relative to using full gradients and characterize when the error is small or the approximation is exact. In simulations, our method matches t
    
[^6]: 通过直接最小化期望解码轮数来训练并行投机草稿模型

    Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds

    [https://arxiv.org/abs/2610.10411](https://arxiv.org/abs/2610.10411)

    本文将投机解码建模为马尔可夫奖励过程，提出直接最小化期望解码轮数（EDR）的训练目标，以优化并行投机草稿模型的全局解码效率。

    

    投机解码通过使用低成本的草稿模型提出候选词元，再由完整规模的目标模型并行验证，从而加速大语言模型的推理。并行和半自回归（semi-AR）草稿模型通过单次前向传播提出整个词块来提高起草效率，但训练这类模型带来了新的困难：给定位置的草稿分布取决于解码轮从哪里开始，而每轮从哪里开始又取决于之前各轮接受词元的数量。现有的训练目标通常依赖于忽略这种跨轮耦合的块内局部替代目标，因此无法直接优化全局解码效率。在这项工作中，我们将投机解码表示为马尔可夫奖励过程，为训练和评估此类草稿模型建立了一个理论框架。这一表述产生了期望解码轮数（EDR）目标，该目标对局部拒绝进行加权……

    arXiv:2610.10411v1 Announce Type: cross  Abstract: Speculative decoding accelerates large language model inference by using a low-cost draft model to propose tokens that the full-size target model verifies in parallel. Parallel and semi-autoregressive (semi- AR) drafters improve drafting efficiency by proposing an entire block in a single forward pass, but training them raises a new difficulty: the draft distribution for a given position depends on where the decoding round starts, and where rounds start depends on how many tokens earlier rounds accepted. Existing training objectives typically rely on block-local surrogates that ignore this cross-round coupling, and therefore do not directly optimize the global decoding efficiency. In this work, we develop a theoretical framework for training and evaluating these drafters by representing speculative decoding as a Markov reward process. This formulation yields the Expected Decoding Rounds (EDR) objective, which weights local rejection co
    
[^7]: 基于风险控制的安全元策略设计

    Safe Meta-Policy Design with Risk Control

    [https://arxiv.org/abs/2610.10393](https://arxiv.org/abs/2610.10393)

    该论文提出一种带风险控制的离线元策略设计方法，在更新退化风险预算约束下通过动态规划规划模型更新时机，并揭示策略改进的信噪比是决定更新频率与风险分配的关键因素。

    

    随着新数据的到来，模型可以被重新训练，但部署每个新版本都存在用较差策略替换较好策略的风险。我们研究如何在未来候选模型被训练之前规划策略更新（即元策略），在改进的收益与性能退化的风险之间取得平衡。我们的离线元策略在“表现劣于被替换策略的更新次数的期望值受预算约束”的条件下，最大化期望累计价值。我们从历史学习轨迹中估计可能切换的价值与风险，将更新计划表示为有向无环图中的一条路径，并使用动态规划来选择更新计划。主阶分析表明，策略改进的信噪比是决定更新频率、等待时间和风险分配的关键因素：更清晰的改进支持更早、更频繁的更新，而更嘈杂的改进则需要更长的等待或更大的风险容忍度。

    arXiv:2610.10393v1 Announce Type: cross  Abstract: Models can be retrained as new data arrive, but deploying every new version risks replacing a good policy with a worse one. We study how to plan policy updates (i.e., meta-policy) before future candidates are trained, balancing the benefits of improvement against the risk of performance regression. Our offline meta-policy maximizes expected cumulative value subject to a budget on the expected number of updates that perform worse than the policies they replace. We estimate the value and risk of possible switches from historical learning trajectories, represent an update schedule as a path in a directed acyclic graph, and select a schedule using dynamic programming. A leading-order analysis identifies the signal-to-noise ratio of policy improvement as a key driver of update frequency, waiting times, and risk allocation: clearer improvements support earlier, more frequent updates, while noisier improvements call for longer waits or greate
    
[^8]: 用于扩散模型加速的库普曼观测器：利用浅层测量修正特征预测

    Koopman Observers for Diffusion Acceleration: Correcting Feature Forecasts with Shallow Measurements

    [https://arxiv.org/abs/2610.10366](https://arxiv.org/abs/2610.10366)

    提出一种观测修正的库普曼框架，在不改变模型参数的前提下，利用即时计算的浅层特征观测来修正深层特征的库普曼预测，并配合周期性完整评估刷新观测器，从而加速冻结扩散模型的采样。

    

    特征缓存通过用基于先前计算的激活值所做的预测来替代昂贵的网络评估，从而加速扩散采样。然而，仅基于过去特征的预测无法直接纳入当前去噪状态的变化。我们研究了廉价的、即时计算的浅层特征能否作为观测来修正这些预测。我们提出了一种用于加速冻结扩散模型的观测修正库普曼框架。利用校准轨迹，我们识别出有限维、随时间变化的库普曼近似，用以联合描述浅层与深层网络特征的增量。在加速采样过程中，这些算子预测昂贵深层特征的演化，而观测到的浅层特征的新息（innovation）则对预测状态进行修正。周期性的完整网络评估会刷新观测器，且所有生成模型参数保持不变。这一公式化……

    arXiv:2610.10366v1 Announce Type: new  Abstract: Feature caching accelerates diffusion sampling by replacing expensive network evaluations with predictions from previously computed activations. However, forecasts based only on past features cannot directly incorporate changes in the current denoising state. We investigate whether inexpensive, freshly computed features can serve as observations for correcting these predictions. We introduce an observation-corrected Koopman framework for accelerating frozen diffusion models. Using calibration trajectories, we identify finite-dimensional, time-dependent Koopman approximations that jointly describe the increments of shallow and deep network features. During accelerated sampling, these operators predict the evolution of expensive deep features, while innovations in the observed shallow features correct the predicted state. Periodic full evaluations refresh the observer, and all generative-model parameters remain unchanged. This formulation 
    
[^9]: 从第一性原理出发的数据集剪枝：一种无标签的线性规划方法

    Dataset Pruning from First Principles: A Label-Free Linear Programming Approach

    [https://arxiv.org/abs/2610.10347](https://arxiv.org/abs/2610.10347)

    提出了一种从第一性原理推导的数据集剪枝方法，将无偏子集选择表述为方差最小化的线性规划问题，无需标签和几何邻近假设即可选出具有代表性的训练子集。

    

    数据集剪枝旨在将大规模训练集缩减为一个具有代表性的子集，同时保持模型性能。现有的基于几何的方法通常假设嵌入空间中相邻的点具有相似的属性。我们没有强加这一假设，而是通过将无偏子集选择重新表述为方差最小化问题，从基本原理推导出几何选择准则。无偏性保证了未加权的子集均值在期望意义上能够恢复整个数据集的均值，包括在固定模型参数下的损失和梯度。具体而言，我们将一族无偏子集选择算法刻画为一个高维多面体。在此框架下，最小化期望采样方差是一个线性目标。在刚体运动下取平均的采样方差差异具有闭式的成对表达式。由于该多面体的维度很高，直接应用标准线性规划是不切实际的，我们转而使用……（摘要在此处截断）

    arXiv:2610.10347v1 Announce Type: cross  Abstract: Dataset pruning reduces a large training set to a representative subset while preserving model performance. Existing geometry-based methods typically assume that nearby points in embedding space share similar properties. Rather than imposing this assumption, we derive geometric selection criteria by reformulating unbiased subset selection as a variance minimization problem. Unbiasedness ensures that unweighted subset averages recover full-dataset averages in expectation, including losses and gradients at fixed model parameters. Specifically, we characterize a family of unbiased subset selection algorithms as a high-dimensional polytope. In this context, minimizing the expected sampling variance is a linear objective. Differences in sampling variance, averaged over rigid motions, admit closed-form pairwise expressions. Because the polytope has high dimension, directly applying standard linear programming is impractical. We instead use t
    
[^10]: 非平稳学习中的数据复用

    Data Reuse in Non-Stationary Learning

    [https://arxiv.org/abs/2610.10340](https://arxiv.org/abs/2610.10340)

    提出了暴露上限复用（ECR）类算法，通过结合在线变化检测、兼容性测试和污染控制来安全复用历史数据，使非平稳在线学习的遗憾值随不同取值数量而非变化次数增长，类似于偏差-方差权衡。

    

    我们研究非平稳环境下的在线学习问题，其目标是追踪一个在有限个重复出现的取值之间突然切换的未知参数。这种取值的重复性为审慎地复用历史观测数据以提升算法性能提供了可能。然而，底层信号不断变化的特性以及缺乏关于这些动态的信息，可能会限制“安全”复用数据的能力。在本文中，我们量化了这类问题中的一些基本权衡，并表明它们与经典的偏差-方差困境具有某种相似性。具体而言，我们提出了一类任意时刻可用的算法，称为暴露上限复用，该算法结合了在线变化检测、兼容性测试和“污染”控制。我们刻画了ECR的遗憾值随不同取值数量而非变化次数增长的情形，并推导出一个新颖的信息论下界。

    arXiv:2610.10340v1 Announce Type: cross  Abstract: We consider online learning in non-stationary environments, where the goal is to track an unknown parameter that switches abruptly between a finite set of recurring values. Recurrence opens the possibility of judiciously reusing past observations to improve algorithm performance. However, the changing nature of the underlying signal and lack of information on these dynamics may limit the ability to "safely" reuse data. In this paper we quantify some of the fundamental tradeoffs in this class of problems, and show that they bear a certain resemblance to the classical bias-variance dilemma. Specifically, we propose a class of anytime algorithms, dubbed Exposure-Capped Reuse (ECR), that combine online change detection, compatibility testing, and "contamination" control. We characterize the regime in which ECR's regret scales with the number of distinct values rather than the number of changes, and derive a novel information-theoretic lowe
    
[^11]: 通过模型无关的概念词典重新审视可解释AI

    Revisiting Explainable AI through Model-Independent Concept Dictionaries

    [https://arxiv.org/abs/2610.10301](https://arxiv.org/abs/2610.10301)

    提出DictXAI方法，通过在输入域中用包含可解释含义的词典定义概念，将输入的稀疏编码与模型预测归因到具体词典元素，从而实现跨模型、架构无关且可操作的可解释AI解释。

    

    现代AI应用依赖于日益复杂的模型。可解释AI（XAI）作为一套旨在提高模型透明度的技术应运而生。然而，现有的XAI方法通常假设输入特征本身是可解释的，或者依赖于难以刻画且高度依赖特定架构的中间内部抽象，这阻碍了这些方法在不同模型间的一致使用。为了解决这些局限性，我们提出了DictXAI，这是一种通过词典直接在输入域中定义概念的方法——词典是一个庞大的、可能过完备的预定义元素集合，每个元素都带有可解释的含义。在技术上，DictXAI首先计算输入的稀疏编码，然后将模型的预测归因于相关的词典元素。我们展示了DictXAI解释的可操作性，表明它们可以将AI故障（例如“Clever Hans”效应）直接归因于可识别的……

    arXiv:2610.10301v1 Announce Type: new  Abstract: Modern applications of AI rely on increasingly complex models. Explainable AI (XAI) has emerged as a set of techniques aimed at improving model transparency. However, existing XAI methods typically assume input features to be inherently interpretable, or they rely on intermediate internal abstractions that are difficult to characterize and highly architecture-specific, hindering consistent use across models. To address these limitations, we propose DictXAI, a method that defines concepts directly in the input domain via a dictionary---a large, potentially overcomplete set of predefined elements, each carrying an interpretable meaning. Technically, DictXAI first computes a sparse code of the input and then attributes the model's prediction to the associated dictionary elements. We demonstrate the actionable nature of DictXAI explanations, showing that they can attribute AI malfunctions (e.g., Clever Hans effects) directly to identifiable 
    
[^12]: 共享高斯化：高斯正则化器能为对比学习证明什么，又遗漏了什么

    Shared Gaussianization: What Gaussian Regularizers Certify About Contrastive Learning, and What They Miss

    [https://arxiv.org/abs/2610.10299](https://arxiv.org/abs/2610.10299)

    本文提出共享高斯化（SG）检验，证明该高斯正则化器能以紧的、与维度无关的平方根速率上界总体 InfoNCE 的超出量，并通过单次检验同时检测视图的失配与非均匀性。

    

    分布匹配正则化器（如 LeJEPA 中的 SIGReg）能为对比学习证明什么？我们研究共享高斯化（SG），这是一种对两个归一化视图的平均值进行的特征函数高斯性检验，并由独立的 $\chi_d$ 半径进行缩放。由于相互不一致的视图会缩短平均值，一次检验即可同时检测视图的失配与非均匀性。SG 恰好在总体 InfoNCE 的对齐且均匀的最小值点处消失，并且在边际相等的条件下，它将 InfoNCE 的超出量（excess）上界约束为 $4\cdot 3^{3/4}\beta$ 乘以 SG 损失的平方根，再加上一个与损失呈线性关系的项。该平方根速率与这一维度无关的常数都是紧的，且任何作用于视图对的平方平均嵌入距离都无法达到更快的速率。在具有显式对齐项的情况下，旋转不变的均匀性检验能给出线性界的充分必要条件是其谱支配 InfoNCE 核 $e^{\beta u^\top v}$ 的谱；SG 自身的检验满足该条件，而高斯核……（摘要原文在此处截断）

    arXiv:2610.10299v1 Announce Type: new  Abstract: What can a distribution-matching regularizer such as SIGReg in LeJEPA certify about contrastive learning? We study shared Gaussianization (SG), a characteristic-function Gaussianity test on the average of two normalized views, scaled by an independent $\chi_d$ radius. Because disagreeing views shorten the average, one test detects both misalignment and non-uniformity. SG vanishes exactly at the aligned, uniform minimizers of population InfoNCE, and under equal marginals it bounds the InfoNCE excess by $4\cdot 3^{3/4}\beta$ times the square root of the SG loss, plus a term linear in the loss. The square-root rate and this dimension-free constant are sharp, and no squared mean-embedding distance on view pairs achieves a faster rate. With an explicit alignment term, a rotation-invariant uniformity test gives a linear bound if and only if its spectrum dominates that of InfoNCE's kernel $e^{\beta u^\top v}$; SG's own test does, Gaussian kerne
    
[^13]: 基于Wasserstein-Fisher-Rao JKO格式的重加权归一化流神经采样

    Neural Sampling with Reweighted Normalizing Flows via the Wasserstein--Fisher--Rao JKO Scheme

    [https://arxiv.org/abs/2610.10278](https://arxiv.org/abs/2610.10278)

    提出了一种基于WFR JKO格式的神经采样算法，首次证明其在任意固定步长下、无需对数凹性等结构性假设即可指数收敛到目标分布，并利用重加权归一化流对其输运与反应分量进行神经参数化实现。

    

    我们提出了一种从由未归一化玻尔兹曼密度所指定的分布中进行采样的神经算法。我们的方法基于在Wasserstein-Fisher-Rao几何中对Kullback-Leibler散度应用Jordan-Kinderlehrer-Otto格式（WFR JKO格式）。我们的贡献有两个方面。首先，我们证明了对于任意固定的步长，精确的WFR JKO迭代在迭代次数趋于无穷时以指数速度收敛到目标分布。值得注意的是，这一结果无需对目标分布做任何结构性假设，例如对数凹性或对数Sobolev不等式。其次，我们开发了WFR JKO格式的神经实现，使用重加权归一化流对其输运和反应分量进行参数化。在具有挑战性的多峰目标分布上的数值实验表明了所提方法的良好性能。

    arXiv:2610.10278v1 Announce Type: cross  Abstract: We propose a neural algorithm for sampling from distributions specified by unnormalized Boltzmann densities. Our approach is based on the Jordan--Kinderlehrer--Otto scheme for the Kullback--Leibler divergence in the Wasserstein--Fisher--Rao geometry (WFR JKO scheme). Our contributions are twofold. First, we prove that, for any fixed step size, the exact WFR JKO iterates converge exponentially fast to the target as the number of iterations tends to infinity. Notably, this result requires no structural assumptions on the target, such as log-concavity or a logarithmic Sobolev inequality. Second, we develop a neural implementation of the WFR JKO scheme that parametrizes its transport and reaction components using reweighted normalizing flows. Numerical experiments on challenging multimodal targets demonstrate the promising performance of the proposed method.
    
[^14]: 海森引导扰动Wasserstein梯度流的有限样本逼近

    Finite-Sample Approximation of Hessian-Guided Perturbed Wasserstein Gradient Flows

    [https://arxiv.org/abs/2610.10218](https://arxiv.org/abs/2610.10218)

    该论文证明了海森引导扰动Wasserstein梯度流的有限粒子逼近在增长时间尺度上的高概率追踪界，关键在于沿参考路径累积的曲率——负曲率会放大误差而正曲率可抑制误差，从而刻画了暂时不稳定仍可精确追踪的有利情形。

    

    Wasserstein梯度流将梯度下降方法推广到了概率测度空间。其海森（Hessian）引导的扰动变体（PWGF）通过引入高斯扰动来帮助逃离非凸问题中的鞍点。我们研究了该流被有限个相互作用的粒子近似时，在随时间增长的时间尺度上何时仍能保持近似精度。我们的分析保留了沿种群驱动参考路径所累积的曲率信息：负曲率可能会放大近似误差，而随后出现的正曲率则可以抑制这些误差的影响。这刻画了一些有利的情形，即暂时的不稳定性与在增长时间尺度上的精确追踪可以并存。在正则性假设和预先设定的公共扰动调度下，我们对满足显式累积曲率条件的参考路径，在高概率事件上证明了粒子追踪界和目标值追踪界。为处理状态相关的高斯跳跃，我们构造了一个种群（摘要在此处截断）

    arXiv:2610.10218v1 Announce Type: new  Abstract: Wasserstein gradient flow extends gradient descent to probability measures. Its Hessian-guided perturbed variant (PWGF) adds Gaussian perturbations to escape saddle points in nonconvex problems. We investigate when its approximation by finitely many interacting particles remains accurate over growing time horizons. Our analysis retains the curvature accumulated along the population-driven reference path: negative curvature can amplify approximation errors, while subsequent positive curvature can damp their influence. This captures favorable scenarios in which temporary instability is compatible with accurate tracking over growing horizons. Under regularity assumptions and a prescribed common perturbation schedule, we prove particle and objective-value tracking bounds on a high-probability event for reference paths satisfying explicit conditions on accumulated curvature. To handle state-dependent Gaussian jumps, we construct a population-
    
[^15]: RoBART：具有树特定旋转的贝叶斯可加回归树

    RoBART: Bayesian Additive Regression Trees with Tree-Specific Rotations

    [https://arxiv.org/abs/2610.10214](https://arxiv.org/abs/2610.10214)

    RoBART通过为每棵树分配特定的Givens旋转来改进贝叶斯可加回归树，使其能高效逼近与预测变量轴不对齐的边界，并对具有各向异性Hölder光滑性的可加函数证明了后验收缩率。

    

    贝叶斯可加回归树（BART）在逼近与预测变量轴不对齐的边界时可能需要大量的分裂。RoBART为每棵树分配一个由其所有内部节点共享的旋转，从而在旋转后的坐标系中仍保持轴对齐分裂和常数叶节点的形式。我们通过Metropolis-Hastings方法联合提出Givens旋转序列以及在所得网格上的切分点，并在将叶节点均值积分掉的条件后验下建立了马氏链的可逆性。对于具有分量特定旋转和各向异性Hölder光滑性的可加函数，我们证明了在经验$L_2$距离下的后验收缩性以及噪声标准差的后验收缩性。在所述先验、设计和网格条件下，当预测变量、树和分量的数量固定且分量数不超过树数时，收敛速率为由各分量的光滑度和所使用的旋转坐标数量决定的分量速率之和。我们还建立了一个后验……

    arXiv:2610.10214v1 Announce Type: cross  Abstract: Bayesian additive regression trees (BART) can require many splits to approximate boundaries misaligned with the predictor axes. RoBART assigns each tree a rotation shared by all internal nodes, retaining axis-aligned splits in rotated coordinates and constant leaves. We jointly propose a Givens rotation sequence and cutpoints on the resulting grid by Metropolis-Hastings and establish reversibility with respect to the conditional posterior with leaf means integrated out. For additive functions with component-specific rotations and anisotropic H\"older smoothness, we prove posterior contraction in empirical $L_2$ distance and for the noise standard deviation. Under the stated prior, design, and grid conditions, with fixed numbers of predictors, trees, and components and no more components than trees, the rate is a sum of componentwise rates determined by smoothness and the number of rotated coordinates used. We also establish a posterior
    
[^16]: 通过机器学习计算链环的切片亏格与解结数

    Computations of the slice genus and the unknotting number of links via machine learning

    [https://arxiv.org/abs/2610.10206](https://arxiv.org/abs/2610.10206)

    该论文利用强化学习和贝叶斯优化，为链环的切片亏格、解结数等难以算法计算的不变量求出新上界，并结合已知下界在许多情形下得到新的精确值，还重现了解结数的非可加性反例。

    

    链环是光滑嵌入在 $S^3$ 中的圆的不相交并集。我们使用强化学习和贝叶斯优化，为几种尚不知道是否可以算法计算的链环不变量获得了新的上界：链环的切片亏格和解结数，以及代数可分裂链环的强切片亏格。我们还利用已知不变量计算了下界。通过结合上界与下界，我们在许多情况下得到了新的精确值。我们的解结智能体能够重现 Brittenham 和 Hermiller 提出的若干反例中解结数的非可加性，并在某些情况下找到了新的解结轨迹。

    arXiv:2610.10206v1 Announce Type: cross  Abstract: Links are disjoint unions of circles smoothly embedded in $S^3$. We use reinforcement learning and Bayesian optimisation to obtain new upper bounds on several link invariants that are not known to be algorithmically computable: the slice genus and the unknotting number for links, and the strong slice genus for algebraically split links. We also compute lower bounds using known invariants. Combining the upper and lower bounds, we obtain new exact values in many cases. Our unknotting agents can reproduce the non-additivity of the unknotting number for several counterexamples due to Brittenham and Hermiller, in some cases finding new unknotting trajectories.
    
[^17]: 基于均匀化与时间条件化因子分解神经似然估计的广泛适用的切换随机微分方程近似MCMC方法

    Broadly Applicable Approximate MCMC for Switching Stochastic Differential Equations Using Uniformization and Time-Conditioned Factorized Neural Likelihood Estimation

    [https://arxiv.org/abs/2610.10194](https://arxiv.org/abs/2610.10194)

    该论文提出了一种结合均匀化与时间条件化因子分解神经似然估计的近似MCMC采样器，突破了现有方法在噪声观测、状态维度、漂移形式和扩散项等方面的限制，实现了对切换随机微分方程广泛适用的贝叶斯推断。

    

    切换随机微分方程（SSDE）描述了参数随潜在状态过程（遵循连续时间马尔可夫链，CTMC）而切换的连续时间动力学。通过允许动力学在不同状态之间切换，SSDE能够表征异构系统行为，并已应用于众多领域。然而，SSDE的贝叶斯推断仍然十分困难，且现有的SSDE推断方法适用性有限，存在诸如无噪声观测、单变量状态、线性漂移或与状态无关的扩散等限制。在本研究中，我们提出了一种用于SSDE的近似马尔可夫链蒙特卡罗（MCMC）采样器，该方法结合了均匀化和因子分解神经似然估计（FNLE），后者是一种基于仿真的推断方法。均匀化为CTMC提供了精确的表示，但需要任意时间间隔上的SDE转移密度。我们通过训练时间条件化的模型来近似这些密度。（原文摘要在此处被截断）

    arXiv:2610.10194v1 Announce Type: cross  Abstract: Switching stochastic differential equations (SSDEs) describe continuous-time dynamics whose parameters switch according to a latent regime process that follows a continuous-time Markov chain (CTMC). By allowing dynamics to change between regimes, SSDEs represent heterogeneous system behavior and have been applied across diverse fields. However, Bayesian inference for SSDEs remains difficult, and existing SSDE inference methods have limited applicability, with restrictions such as noise-free observations, univariate states, linear drift, or state-independent diffusion. In this study, we propose an approximate Markov chain Monte Carlo sampler for SSDEs using uniformization and factorized neural likelihood estimation (FNLE), a simulation-based inference method. Uniformization provides an exact representation of the CTMC but requires SDE transition densities over arbitrary time intervals. We approximate these densities by training a time-c
    
[^18]: 一阶EDM预测器的普适局部误差与实际放大效应

    Universal Local Error and Realized Amplification for the First-Order EDM Predictor

    [https://arxiv.org/abs/2610.10190](https://arxiv.org/abs/2610.10190)

    该论文证明了一阶EDM扩散采样器的单步局部离散化误差具有与数据分布无关的普适二次上界，并通过在高噪声水平利用显式收缩准则、在低噪声水平引入可远小于最坏Lipschitz常数的实际放大效应，最终获得了O(e^{Λ_K}/K)的全局离散化误差保证。

    

    我们在2-Wasserstein距离下分析了Karras等人（2022年）提出的一阶确定性扩散采样器（称为EDM），将误差来源分为两部分：局部离散化误差及其被后续学习步骤放大的效应。我们证明局部误差存在一个普适上界：对于任何具有有限二阶矩的数据分布，单步离散化误差与步长呈二次方关系，且其显式常数不依赖于数据分布。相比之下，误差传播依赖于学习到的网络。在高噪声水平下，我们利用EDM的网络参数化推导出一个显式的收缩准则。在低噪声水平下，我们通过采样器所输运的分布上实际实现的放大来度量误差传播；这种实际放大可以远小于最坏情况下的Lipschitz常数。该分析得出了O(e^{Λ_K}/K)的全局离散化误差。

    arXiv:2610.10190v1 Announce Type: cross  Abstract: We analyze the first-order deterministic diffusion sampler of Karras et al. (2022), termed EDM, in 2-Wasserstein distance by separating two sources of error: local discretization error and its amplification by subsequent learned steps. We prove that local error admits a universal bound: for any data distribution with finite second moment, the one-step discretization error is quadratic in the step size, with an explicit constant that does not depend on the data distribution. Error propagation, in contrast, depends on the learned network. At high noise levels, we exploit the network parametrization of EDM to derive an explicit contraction criterion. At low noise levels, we measure propagation through the amplification realized on the distributions transported by the sampler; this realized amplification can be arbitrarily smaller than the worst-case Lipschitz constant. This analysis yields an $O(e^{\Lambda_K}/K)$ global discretization err
    
[^19]: 动力学朗之万遇见分裂吉布斯：基于扩散先验的成像逆问题加速后验采样

    Kinetic Langevin Meets Split Gibbs: Accelerated Posterior Sampling for Imaging Inverse Problems with Diffusion Priors

    [https://arxiv.org/abs/2610.10187](https://arxiv.org/abs/2610.10187)

    该论文提出RED-KLwSGS方法，将欠阻尼（动力学）朗之万扩散与分裂吉布斯采样框架结合，利用单次去噪得分驱动辅助变量更新，在与Langevin-within-SGS相同的每次迭代成本下实现成像逆问题后验采样的加速，并给出了强对数凹先验下连续和离散时间的非渐近Wasserstein-2收敛保证。

    

    分裂吉布斯采样（SGS）是贝叶斯成像逆问题中后验采样的一种流行框架。它通过引入辅助变量将高斯数据保真项与复杂先验解耦，使得数据变量可以精确更新，只有先验侧的条件分布难以采样。现有采样器以两种方式之一处理该条件分布：即插即用SGS在每次迭代中运行多步扩散去噪器，代价高昂且缺乏非渐近保证；Langevin-within-SGS采用廉价的过阻尼朗之万步，但需要大量迭代。我们提出RED-KLwSGS，该方法对数据变量保持精确的高斯更新，并使用由单次去噪得分驱动的欠阻尼（动力学）朗之万扩散来更新辅助变量，其每次迭代成本与Langevin-within-SGS相同。我们证明了对于强对数凹先验，该方法在连续和离散时间下均具有非渐近Wasserstein-2收敛性保证。

    arXiv:2610.10187v1 Announce Type: cross  Abstract: Split Gibbs sampling (SGS) is a popular framework for posterior sampling in Bayesian imaging inverse problems. It decouples a Gaussian data-fidelity term from a complex prior through an auxiliary variable, so the data variable is updated exactly and only the prior-side conditional is hard to sample. Existing samplers treat this conditional in one of two ways. Plug-and-play SGS runs a multi-step diffusion denoiser at every iteration, which is expensive and lacks non-asymptotic guarantees. Langevin-within-SGS takes cheap overdamped Langevin steps but needs many iterations. We propose RED-KLwSGS, which keeps the exact Gaussian update for the data variable and updates the auxiliary variable with underdamped (kinetic) Langevin diffusions driven by a one-shot denoising score, at the same per-iteration cost as Langevin-within-SGS. We prove non-asymptotic Wasserstein-2 convergence in continuous and discrete time for strongly log-concave priors
    
[^20]: 通过贝叶斯优化对贝叶斯优化算法进行预训练

    Pre-training of Bayesian Optimization Algorithm through Bayesian Optimization

    [https://arxiv.org/abs/2610.10186](https://arxiv.org/abs/2610.10186)

    该论文提出了一种通过在高斯过程样本路径上运行贝叶斯优化来最小化期望累积遗憾，从而利用另一个贝叶斯优化过程自动预训练贝叶斯优化算法参数的框架。

    

    贝叶斯优化（BO）作为昂贵黑箱优化问题的标准方法被广泛应用。然而，贝叶斯优化算法通常涉及必须事先指定的参数，其性能可能在很大程度上取决于这些参数的选择。我们提出了一个框架，利用从贝叶斯优化开始时已有信息推断出的高斯过程（GP）中抽取的样本路径来优化这些参数。我们使用累积遗憾作为贝叶斯优化算法的性能指标。通过在生成的样本路径上运行贝叶斯优化算法，我们可以获得给定参数配置下其期望累积遗憾的经验估计。优化该估计使我们能够识别出在当前可用信息条件下有望实现低累积遗憾的参数配置。由于该参数优化本身就是一个黑箱优化问题，我们采用另一个贝叶斯优化过程来求解它，我们将其称为……

    arXiv:2610.10186v1 Announce Type: new  Abstract: Bayesian optimization (BO) is widely used as a standard approach for expensive black-box optimization. However, BO algorithms often involve parameters that must be specified in advance, and their performance can strongly depend on these choices. We propose a framework for optimizing such parameters using sample paths drawn from a Gaussian process (GP) inferred from the information available at the start of BO. We use cumulative regret as the performance metric for a BO algorithm. By running the BO algorithm on the generated sample paths, we obtain an empirical estimate of its expected cumulative regret for a given parameter configuration. Optimizing this estimate allows us to identify parameter configurations that, given the currently available information, are expected to achieve low cumulative regret. Since this parameter optimization is itself a black-box optimization problem, we employ another BO procedure to solve it, which we refer
    
[^21]: 基于顺序白化的空间相关数据保形预测

    Conformal Prediction for Spatially Dependent Data via Sequential Whitening

    [https://arxiv.org/abs/2610.10168](https://arxiv.org/abs/2610.10168)

    该论文提出一种通过对校准残差进行顺序条件化（顺序白化）的保形预测方法，解决了空间相关数据下可交换性假设失效、且仅由校准残差可预测的空间变异残留导致区间效率下降的问题，在正确的工作协方差与椭圆残差分布下实现精确的有限样本覆盖率，并可借助最近邻近似扩展到大型网络。

    

    分割保形预测利用留出的（校准）数据上的预测误差来确定预测区间的宽度。当这些误差与目标位置的误差可交换时，它能保证无分布的有限样本覆盖率。然而在空间依赖和非随机采样几何结构下，这一假设可能失效。现有的空间方法使用拟合残差从校准误差和目标误差中去除空间变异中可预测的部分。但是，只有校准残差能够预测的那部分空间变异仍然保留在目标误差和校准误差之中，从而降低了预测区间的效率和稳定性。我们通过额外地对校准残差进行顺序条件化来解决这一问题，该方法借助最近邻近似可扩展到大规模网络。在工作协方差设定正确且残差服从椭圆分布定律的条件下，所得区间具有精确的有限样本覆盖率。

    arXiv:2610.10168v1 Announce Type: cross  Abstract: Split conformal prediction uses prediction errors on held-out (calibration) data to determine how wide the prediction intervals should be. It guarantees distribution-free finite-sample coverage when these errors and the error at the target site are exchangeable. This assumption may fail under spatial dependence and nonrandom sampling geometry. Existing spatial methods use fitting residuals to remove the predictable part of spatial variation from calibration and target errors. However, the spatial variation that only the calibration residuals can predict remains in both the target and calibration errors, reducing the efficiency and stability of the interval. We address this by additionally conditioning on the calibration residuals sequentially, which scales to large networks through nearest-neighbour approximations. Under a correct working covariance and an elliptical residual law, the resulting interval has exact finite-sample coverage
    
[^22]: 基于赢家反馈的m集对抗性多臂老虎机

    m-Set Adversarial Bandits with Winner Feedback

    [https://arxiv.org/abs/2610.10128](https://arxiv.org/abs/2610.10128)

    本文研究了不同效用和反馈模型下m集对抗性多臂老虎机的遗憾界，其主要技术贡献是遗憾值的信息论下界，揭示了环境设置中的细微变化会对学习速率产生巨大影响。

    

    我们针对不同效用函数（赢家奖励或奖励总和）和不同反馈模型（赢家索引、赢家奖励、奖励总和及其组合），给出了 $m$ 集对抗性多臂老虎机遗憾值的上界和下界。通过与组合老虎机和 MNL 老虎机的标准界进行比较，我们的结果揭示了环境设置中的细微变化如何对学习速率产生巨大影响。我们的主要技术贡献在于遗憾值的信息论下界。在合成数据上的实验证实了我们的理论分析。

    arXiv:2610.10128v1 Announce Type: new  Abstract: We show upper and lower bounds on the regret of $m$-set adversarial bandits for different utilities (winner reward or sum of rewards) and feedback models (winner index, winner reward, sum of rewards, and their combinations). By comparing to standard bounds for combinatorial and MNL bandits, our results reveal how subtle changes in the setting can have a dramatic impact on the learning rates. Our main technical contributions are the information-theoretic lower bounds on the regret. Experiments on synthetic data confirm our theoretical analyses.
    
[^23]: 通过结果条件重校准实现针对关注事件的校准概率预测

    Towards Calibrated Probabilistic Forecasts for Events of Interest via Outcome-Conditional Recalibration

    [https://arxiv.org/abs/2610.10076](https://arxiv.org/abs/2610.10076)

    本文提出了一种简单易实现的事后重校准方法——结果条件重校准，能够在用户定义的结果空间区域（如极端事件）上对概率预测进行重校准，从而确保决策者最关注的事件也能获得校准良好的预测。

    

    校准是概率预测能够有效支持决策制定的一项基本要求。尽管最先进的预测方法往往产生校准不佳的预测分布，但目前已有多种事后重校准方案被提出以生成校准良好的预测。然而，流行的重校准方案可能掩盖结果空间特定区域中存在的校准不佳问题。由于特定结果（例如极端事件）往往对决策制定最为重要，因此当评估仅限于这些结果时，概率预测应当是校准良好的。为此，本文提出了结果条件重校准，这是一种事后方法，可在用户自定义的结果空间区域上重新校准概率预测。该方法简单、易于实现，并且可应用于任意预测分布。其工作原理是将Kuleshov等人（2018）提出的分位数重校准方法应用于……（摘要在此处截断）

    arXiv:2610.10076v1 Announce Type: cross  Abstract: Calibration is an essential requirement for probabilistic predictions to be useful for decision making. While state-of-the-art prediction methods often yield miscalibrated predictive distributions, several post-hoc recalibration schemes have been proposed to generate calibrated predictions. However, popular recalibration schemes can conceal miscalibration in specific regions of the outcome space. Since particular outcomes, such as extreme events, often matter most for decision making, probabilistic predictions should be calibrated when evaluation is restricted to these outcomes. Hence, in this paper, we introduce outcome-conditional recalibration, a post-hoc method to recalibrate probabilistic predictions on user-defined regions of the outcome space. The method is simple, easy to implement, and can be applied to arbitrary predictive distributions. It works by applying the quantile recalibration approach of Kuleshov et al. (2018) to for
    
[^24]: 基于Koopman算子与退出时间最优控制的转移路径采样

    Transition Path Sampling Using Koopman Operators and Exit-Time Optimal Control

    [https://arxiv.org/abs/2610.10054](https://arxiv.org/abs/2610.10054)

    提出一种基于Koopman算子的转移路径采样新方法，利用其线性性质在无需转移路径数据的情况下识别亚稳态集合并估计committor函数，同时将TPS表述为退出时间最优随机控制问题，从而解决了现有神经网络方法的计算开销与性能保证问题。

    

    在亚稳态之间采样转移路径是动力系统理论（尤其是分子动力学）中的一个核心问题。其关键挑战在于分隔各亚稳态的高自由能势垒，使得态间转移极为罕见。近期基于机器学习的方法将转移路径采样（TPS）表述为固定时间范围内的最优随机控制（OSC）问题，并通过在仿真循环中训练的神经网络对漂移偏置进行参数化，这需要反复进行有偏的模拟推演。为解决此类模型的计算开销与性能保证问题，我们提出了一种基于Koopman算子的新方法。由于Koopman算子是线性的，其主导特征函数能够揭示亚稳态集合，并在无需任何转移路径信息的情况下给出committor函数的估计。此外，我们将TPS表述为直至退出时间的最优随机控制问题。我们的时间范围

    arXiv:2610.10054v1 Announce Type: cross  Abstract: Sampling transitions between metastable states is a central problem in dynamical systems theory and molecular dynamics in particular. A key challenge is the existence of high free-energy barriers that separate the states, making transitions extremely rare. Recent machine learning-based methods cast transition path sampling (TPS) as an optimal stochastic control (OSC) problem over a fixed time horizon, and parameterize the drift bias via a neural network trained by simulation-in-the-loop, requiring repeated biased rollouts. To address computational and performance guarantee issues of these models, we propose a new approach for the problem based on Koopman operators. Because Koopman operators are linear, their leading eigenfunctions reveal the metastable sets and provide an estimate of the committor function with no transition path information required. Furthermore, we formulate TPS as an OSC problem up to an exit time. Our time horizon 
    
[^25]: 多头自注意力的高斯等价性

    Gaussian Equivalence for Multi-Head Self-Attention

    [https://arxiv.org/abs/2610.10033](https://arxiv.org/abs/2610.10033)

    利用随机矩阵理论建立了多头自注意力的高斯等价性，证明用缩放分数加高斯噪声替代softmax注意力可保持中心化输出的极限谱定律，从而分离了头分配与投影宽度的影响。

    

    对多头自注意力的理论理解是研究现代神经网络的基础。利用随机矩阵理论，我们建立了多头自注意力的高斯等价性：用缩放后的分数加上高斯噪声替代softmax注意力，可以保持中心化输出的极限谱定律。这一等价性还涵盖了依赖于键的值投影和输出投影。所得到的定律分离了头分配和投影宽度的影响，并区分了保持谱特性的跨头共享与头内键-值依赖。

    arXiv:2610.10033v1 Announce Type: cross  Abstract: A theoretical understanding of multi-head self-attention is fundamental to the study of modern neural networks. Using random matrix theory, we establish Gaussian equivalence for multi-head self-attention: replacing softmax attention with rescaled scores plus Gaussian noise preserves the limiting spectral law of the centered output. This equivalence also covers value and output projections that depend on the keys. The resulting laws separate the effects of head allocation and projection widths, and distinguish spectrum-preserving across-head sharing from within-head key--value dependence.
    
[^26]: 通过扩散互信息控制隐式生成模型中的依赖关系

    Controlling Dependence in Implicit Generative Models via Spread Mutual Information

    [https://arxiv.org/abs/2610.10021](https://arxiv.org/abs/2610.10021)

    提出扩散互信息（SMI），通过对生成变量施加扩散核并跨噪声级别对互信息进行加权积分，克服了隐式生成模型中奇异分布缺少得分函数及密度比估计重叠性差的难题，实现对统计依赖的有效控制。

    

    互信息（MI）为隐式生成模型中抑制或鼓励统计依赖性提供了一个优化目标。然而，由于隐式模型的密度通常是难以处理的，直接评估互信息极具挑战性。一种补救方法是从条件得分与边缘得分之间的差异来估计生成器的梯度，而该得分差异又可以通过对通过分类学习得到的对数密度比进行求导来估计。然而，这种构造面临两个困难：(i) 奇异分布可能不具备所需的得分函数；(ii) 分布间重叠性差会阻碍密度比估计。因此，我们引入了扩散互信息，它是通过对生成变量施加一个共同的扩散核，从而获得的跨不同噪声级别的互信息的加权积分。高斯扩散可以产生平滑且严格为正的条件密度和边缘密度，从而将梯度构造扩展到……

    arXiv:2610.10021v1 Announce Type: cross  Abstract: Mutual information (MI) provides an objective for suppressing or encouraging statistical dependence in implicit generative models. However, direct MI evaluation is challenging in implicit models due to typically intractable densities. A remedy is estimating the generator gradient from the difference between conditional and marginal scores. This score difference can, in turn, be estimated by differentiating a log density ratio learned through classification. This construction nevertheless faces two difficulties: (i) singular distributions need not admit the required score functions, and (ii) poor overlap can hinder density-ratio estimation. We therefore introduce Spread Mutual Information (SMI), a weighted integral of MI across noise levels obtained by applying a common spreading kernel to the generated variable. Gaussian spreading yields smooth, strictly positive conditional and marginal densities, extending the gradient construction t
    
[^27]: 极端二分类：利用极值理论实现对假负类的极端约束

    Extreme Binary Classification: Extreme Value Theory for Extreme Constraint on False Negative

    [https://arxiv.org/abs/2610.09984](https://arxiv.org/abs/2610.09984)

    本文提出“极端二分类”新问题，并基于极值理论设计了阈值自适应方法与基于置换检验的特征选择程序，使分类器的假负类率以快于 $1/N_1$ 的速率趋近于零，实验表现优于最先进方法。

    

    尽管二分类是机器学习中研究最为广泛的问题之一，但以学习一个假负类率几乎为零的分类器为目标的这一情形在很大程度上仍未被探索。在本文中，我们提出了“极端二分类”问题，其目标是学习一个假负类率 $\alpha$ 受 $\epsilon_{N_1}=o_{N_1\to\infty}(1/N_1)$ 约束的分类器，其中 $N_1$ 表示训练集中正例样本的数量。为了解决这一问题，我们提出了一种基于极值理论推导的理论保证的阈值自适应方法，并结合一种基于对样本最大值进行置换检验的特征选择程序。在四个不同规模的真实数据集上的实验结果表明，我们的方法优于当前最先进的方法。此外，我们还通过……展示了该方法的可解释性。

    arXiv:2610.09984v1 Announce Type: cross  Abstract: While binary classification is one of the most extensively studied problems in machine learning,   the regime in which the goal is to learn a classifier with an almost zero false negative rate remains largely unexplored.   In this paper, we introduce the Extreme Binary Classification problem, where the objective is to learn a classifier whose false negative rate $\alpha$ is constrained by $\epsilon_{N_1}=o_{N_1\to\infty}(1/N_1)$, with $N_1$ denoting the number of positive examples in the training set.   To address this problem, we propose a threshold adaptation method theoretically grounded in guarantees derived from Extreme Value Theory, together with a feature selection procedure based on a permutation test applied to sample maxima.   Experimental results on four real-world datasets of varying sizes demonstrate that our approach compares favorably with state-of-the-art methods.   In addition, we illustrate its interpretability throug
    
[^28]: 用于近似推断模型（IM）推断的可能性径向传输

    Possibilistic Radial Transport for Approximate IM Inference

    [https://arxiv.org/abs/2610.09956](https://arxiv.org/abs/2610.09956)

    提出一种可能性径向传输方法，将参数的可能性轮廓值编码到源点半径中，并结合深度学习算法实现高效的近似可能性推断模型推断，使覆盖率评估、功效分析和新数据预测检验变得切实可行。

    

    arXiv:2610.09956v1 公告类型：交叉发布（cross）。摘要：在可能性推断模型（IM）框架下，在观察到数据之后再探索假设空间仍然有效，前提是显著性水平保持固定。其代价是计算量：每个合理性值都是可能性轮廓在假设上的上确界，而轮廓本身在每个被查询的参数值处都需要进行近似。我们提出了一种可能性径向传输方法，将参数的轮廓值隐藏在其源点的半径之中。当选择使壳层内熵最大化的传输时，对覆盖某一置信截断的参数进行采样就简化为截断半径的问题。我们提供了一种深度学习算法，在最大化每个壳层内熵的同时强制满足轮廓深度条件。我们的摊销方法使得对学习到的近似进行覆盖率和功效评估变得切实可行，同时也能对新数据集进行预测检验。我们还利用该采样器构建了Bel-Pl谱用于比较（摘要在此处截断）。

    arXiv:2610.09956v1 Announce Type: cross  Abstract: Probing the hypothesis space after seeing the data remains valid under possibilistic inferential models (IMs), provided the significance level stays fixed. The price is computation, as each plausibility is a supremum of the possibility contour over the hypothesis, and the contour itself is approximated at each queried parameter value. We propose a possibilistic radial transport, which hides the contour value of a parameter in the radius of its source point. When a transport that maximizes within-shell entropy is picked, sampling parameters covering a confidence cut becomes a matter of truncating the radius. We provide a deep learning algorithm that enforces the contour depth condition while maximizing the entropy within each shell. Our amortization makes coverage and power assessments of the learned approximation practical as well as predictive check of new datasets. We also use the sampler to construct a Bel-Pl spectrum for comparing 
    
[^29]: 多臂老虎机中的期望样本复杂度

    Expected Sample Complexity in Multi-Armed Bandits

    [https://arxiv.org/abs/2610.09929](https://arxiv.org/abs/2610.09929)

    本文提出了期望近似正确（ACE）新框架来研究多臂老虎机的期望样本复杂度，证明ACE保证蕴含几乎必然收敛到最优期望奖励，并揭示了确定性算法无法获得良好ACE界这一特性，同时针对次优水平ε已知与未知两种情形分析了随机算法。

    

    样本复杂度是序贯决策问题中广泛使用的一种指标，定义为智能体与环境交互过程中次优决策的次数。我们研究了随机多臂老虎机问题的样本复杂度，并引入了期望样本复杂度这一性能度量，在一个称为“期望近似正确”的新型框架中对其进行分析。我们证明了ACE保证意味着几乎必然收敛到最优期望奖励，这与其他框架中的高概率保证形成对比，同时我们还展示了如何将ACE保证转化为显式的期望遗憾界。我们进一步证明，与现有度量不同，确定性算法无法获得良好的ACE界，并在两种设置下分析了随机算法：当允许的次优水平 ε 对算法已知时，以及当其未知时。在前一种情况下，我们设计了一种……

    arXiv:2610.09929v1 Announce Type: new  Abstract: Sample complexity is a widely used metric in sequential decision-making problems, defined as the number of suboptimal decisions during the interaction between the agent and an environment. We study the sample complexity of stochastic multi-armed bandit problems and introduce the expected sample complexity performance measure, analyzing it in a novel framework called approximately correct in expectation (ACE). We show that ACE guarantees imply almost sure convergence to the optimal expected reward, in contrast to high-probability guarantees found in other frameworks, and also show how to convert ACE guarantees into explicit expected regret bounds. We further show that, in contrast to existing measures, deterministic algorithms cannot obtain favorable ACE bounds, and analyze stochastic algorithms in two settings: when the allowed suboptimality level $\epsilon$ is known to the algorithm and when it is unknown. In the former, we devise an ex
    
[^30]: 深度学习中Hessian矩阵的特征值：对称性的起源及其破缺

    Eigenvalues of the Hessian in Deep Learning: The Origin of Symmetry and Its Breaking

    [https://arxiv.org/abs/2610.09919](https://arxiv.org/abs/2610.09919)

    本文提出，深度学习中训练模型Hessian特征值呈现的“零附近大块+孤立离群值”的谱结构，源于相对一个隐藏的高度对称参考构型的对称破缺——该参考构型的Hessian具有权重对称性之外的不变性，其对称破缺产生了观测到的谱层级结构。

    

    深度学习中训练后模型的Hessian谱呈现出一种持续存在的模式：特征值组织成不同的簇，包括一个位于零附近的大块以及少数孤立的离群值。本文表明，当将原始设定理解为对附近一个原本隐藏的高度对称参考构型的偏离时，这些谱现象便获得了一个自然的解释。通过对架构、数据分布或参数度量等进行修改，可以揭示出一个邻近的参考构型，其Hessian展现出丰富的、无法由权重对称性所解释的不变性。在该参考构型中，对称性使得谱能够被精确描述，并迫使出现高维的核空间以及大重数的特征值。而回到原始构型则破坏了Hessian的对称性，从而产生了所观测到的簇与离群值的层级结构。该框架在相当一般的情形下被建立，并对……（摘要原文在此处截断）

    arXiv:2610.09919v1 Announce Type: new  Abstract: Hessian spectra at trained models in deep learning exhibit a persistent pattern: eigenvalues organize into distinct clusters, including a large bulk near zero and a few isolated outliers. This paper shows that a natural account of these spectral phenomena emerges when the original setting is understood as a departure from a nearby, otherwise hidden, highly symmetric reference.   Modifications, including changes to the architecture, data distribution, or parameter metric, expose a nearby reference configuration whose Hessian exhibits rich invariances-ones not accounted for by weight symmetries. There, symmetry enables a precise description of the spectra, forcing high-dimensional kernels and eigenvalues of large multiplicity. Returning to the original configuration breaks the Hessian symmetry and thereby produces the observed hierarchy of clusters and outliers.   The framework is developed in some generality, with a detailed analysis of t
    
[^31]: 超越性逆优化：学习优于智能体决策的目标函数

    Outperformance Inverse Optimization: Learning Objective Functions that Outperform Agent Decisions

    [https://arxiv.org/abs/2610.09890](https://arxiv.org/abs/2610.09890)

    本文提出“超越性逆优化”新范式，不再复现智能体的次优决策，而是学习能诱导出在各分量上都优于观测决策的最优解的目标函数权重，并配套给出了适用于混合整数线性规划的预言机损失函数、梯度与DC优化算法以及泛化误差理论保证。

    

    逆优化通过估计目标函数的权重来解释观测到的决策，使其被解释为最优解，并已在多个领域得到应用。对于混合整数线性规划（MILP），现有方法旨在将观测结果重现为最优解，因此当观测结果为次优时，只能学习到折中的权重。我们提出超越性逆优化，它转而寻求这样的权重：在每个状态下诱导出一个在各个分量上都优于观测行为的最优解。我们给出了一个仅需借助前向问题预言机即可计算的损失函数，因而适用于MILP，并提出了用于最小化该损失函数的基于梯度和DC（差分凸）优化算法。对于在所有观测点均诱导出唯一超越性最优解的权重，我们证明了在新状态下未能诱导出此类解的概率（即泛化误差）受一个与样本量成反比的量的约束（摘要在此处被截断）。

    arXiv:2610.09890v1 Announce Type: cross  Abstract: Inverse optimization estimates the weights of an objective function that explain observed decisions as optimal solutions, and is used in a variety of fields. For mixed-integer linear programs (MILPs), existing methods aim to reproduce the observations as optimal solutions, and thus learn compromise weights when the observations are suboptimal. We propose outperformance inverse optimization, which instead seeks weights that induce, at each state, an optimal solution outperforming the observed action in every component. We give a loss function that can be evaluated with forward-problem oracles alone and is thus applicable to MILPs, together with gradient-based and DC optimization algorithms for minimizing it. For weights inducing a unique outperforming optimal solution at all observations, we prove that the probability of failing to induce such a solution at a new state (the generalization error) is bounded by a quantity inversely propor
    
[^32]: 耗散知识动力学模型的可辨识性：设计激励下的精确恢复与观测数据上的退化

    Identifiability of a dissipative knowledge-dynamics model: exact recovery under designed excitation, degeneration on observational data

    [https://arxiv.org/abs/2610.09889](https://arxiv.org/abs/2610.09889)

    该论文将人类学习建模为参数具有机制含义的耗散常微分方程组，证明了在设计激励条件下模型参数可从数据中被精确恢复（双概念情形有闭式解），而在仅依赖观测数据时可辨识性会退化，并提出了数值精确等价且速度大幅提升的半隐式L-稳定批量求解器。

    

    人类学习是一个耗散动力学过程：熟练度通过练习积累，因遗忘而衰减，并在相互依赖的概念之间传播。我们将其建模为一个非线性耗散常微分方程组，其参数具有机制层面的含义（编码先修耦合关系的概念传递矩阵、各概念的遗忘率，以及饱和的练习-响应增益），并研究这些参数何时能够真正从数据中被恢复。我们在显式激励条件下证明了相应逆问题的结构可辨识性定理，针对双概念情形给出了构造性的闭式恢复方法，并同时给出了单调性、鲁棒性与L-稳定性结果。我们为耗散子系统推导了一种半隐式L-稳定数值格式，以及一个与逐轨迹公式数值等价的批量求解器（预测逐位一致，梯度误差达 $10^{-10}$），同时速度提升两个数量级。

    arXiv:2610.09889v1 Announce Type: new  Abstract: Human learning is a dissipative dynamical process: mastery accumulates through practice, decays through forgetting, and propagates across interdependent concepts. We model it as a nonlinear dissipative system of ordinary differential equations whose parameters are mechanistically meaningful (a concept-transfer matrix encoding prerequisite coupling, per-concept forgetting rates, and a saturating practice-response gain), and we study when those parameters can actually be recovered from data. We prove a structural identifiability theorem for the associated inverse problem under explicit excitation conditions, with constructive closed-form recovery for the two-concept case, together with monotonicity, robustness and L-stability results. We derive a semi-implicit L-stable scheme for the dissipative subsystem and a batched solver numerically equivalent to the per-trajectory formulation (bit-exact predictions, gradients to $10^{-10}$) yet two o
    
[^33]: 稀疏化随机性而非容量：通过先验尺度的深度权重因子分解实现部分随机性

    Sparsifying Stochasticity, Not Capacity: Partial Stochasticity via Deep Weight Factorization of Prior Scales

    [https://arxiv.org/abs/2610.09886](https://arxiv.org/abs/2610.09886)

    该论文提出通过对先验尺度进行深度权重因子分解来学习贝叶斯神经网络中哪些参数应保持随机性，使正则化稀疏化随机性而非模型容量，并提供了可线性时间检验的通用条件密度逼近证书，同时证明常见的采样-优化混合方案是II型最大后验目标的随机近似。

    

    贝叶斯神经网络不必完全随机也能成为通用的条件密度逼近器，但哪些参数应该是随机的仍是一个悬而未决的问题。我们通过对先验尺度（即参数先验的标准差）应用深度权重因子分解来学习这种划分，同时利用最大均值差异目标将函数先验拟合到高斯过程。先验尺度低于某个截断值的参数变为确定性的，并在推理过程中进行优化，因此该正则化器稀疏化的是随机性而非容量。我们给出了一个可在线性时间内检验的通用条件密度逼近证书，以及在证书失败时的最小修复方案。我们进一步证明，常见的混合方案——对部分参数进行采样而对其他参数进行优化——是针对某一类II型最大后验目标的随机近似，并且耦合的步长可能会留下追踪偏差。

    arXiv:2610.09886v1 Announce Type: cross  Abstract: Bayesian neural networks need not be fully stochastic to be universal conditional density approximators, but it remains open which parameters should be stochastic. We learn this split by applying deep weight factorization to the prior scales, which are the standard deviations of the parameter priors, while fitting the functional prior to a Gaussian process with a maximum mean discrepancy objective. A parameter whose prior scale falls below a cutoff becomes deterministic and is optimized during inference, so the regularizer sparsifies stochasticity rather than capacity. We give a certificate for universal conditional density approximation that is checkable in linear time, together with a minimal repair when it fails. We further show that the common hybrid scheme of sampling some parameters and optimizing the others is stochastic approximation for a type-II maximum a posteriori objective, and that coupled step sizes can leave a tracking 
    
[^34]: AdaPS-LiNGAM：小样本设定下线性非高斯无环模型的自适应前驱选择

    AdaPS-LiNGAM: Adaptive Predecessor Selection for Linear Non-Gaussian Acyclic Models under Small-Sample Settings

    [https://arxiv.org/abs/2610.09782](https://arxiv.org/abs/2610.09782)

    本文揭示了DirectLiNGAM在变量数超过样本量时残差化必然退化的结构性局限，并提出利用由图结构决定的“活动边界”子集进行自适应前驱选择的AdaPS-LiNGAM方法，以实现小样本情形下可靠的因果发现。

    

    当可用样本量相对于变量数量较小时，因果发现变得尤为具有挑战性。这一挑战同样存在于线性非高斯无环模型中，这是一个从观测数据中进行因果发现的可辨识框架。DirectLiNGAM通过依次识别外生变量并从剩余变量中去除其线性效应，来估计一个使原因排在结果之前的因果顺序。我们确立了这一过程的一个结构性局限：当变量数量超过样本量时，反复的残差化必然在完整因果顺序确定之前就变得退化。我们的分析进一步揭示，每个残差都可以仅利用因果顺序中先前已确定变量的一个由图结构决定的子集来重构，该子集被称为“活动边界”。这一结果启发了AdaPS-LiNGAM（自适应前驱选择）方法。

    arXiv:2610.09782v1 Announce Type: new  Abstract: Causal discovery becomes particularly challenging when the available sample size is small relative to the number of variables. This challenge also arises in the linear non-Gaussian acyclic model (LiNGAM), an identifiable framework for causal discovery from observational data. DirectLiNGAM estimates a causal order, which arranges variables so that causes precede their effects, by sequentially identifying an exogenous variable and removing its linear effect from the remaining variables. We establish a structural limitation of this procedure: when the number of variables exceeds the sample size, repeated residualization necessarily becomes degenerate before the full causal order can be determined. Our analysis further reveals that each residual can be reconstructed using only a graph-determined subset of variables already placed earlier in the causal order, termed the active boundary. This result motivates AdaPS-LiNGAM (Adaptive Predecessor
    
[^35]: 均场神经网络训练中非线性观测量的涨落

    Fluctuations of Nonlinear Observables in Mean Field Neural Network Training

    [https://arxiv.org/abs/2610.09768](https://arxiv.org/abs/2610.09768)

    本文通过在加权Sobolev空间中应用仅需普通Fréchet可微性的泛函Delta方法（无需Lions导数），证明了均场神经网络训练中的涨落会传播到有限维非线性观测量，并建立了相应的中心极限定理与显式协方差表示。

    

    均场极限通过参数经验分布的演化来描述宽神经网络的训练动力学。尽管泛函中心极限定理刻画了该分布的渐近涨落，但实际关心的量通常是参数分布的非线性观测量而非分布本身。在这项工作中，我们展示了这些均场涨落如何传播到由随机梯度下降训练的浅层神经网络的有限维非线性观测量上。我们在构建极限涨落过程的加权Sobolev空间中进行研究，在普通Fréchet可微性条件下应用泛函Delta方法，而无需对测度变量使用Lions导数。我们得到了这些观测量的中心极限定理，并在其微分的适当表示下，得到了显式的协方差……

    arXiv:2610.09768v1 Announce Type: new  Abstract: Mean field limits describe the training dynamics of wide neural networks through the evolution of the empirical distribution of their parameters. Although functional central limit theorems characterize the asymptotic fluctuations of this distribution, quantities of practical interest are typically nonlinear observables of the parameter distribution rather than the distribution itself. In this work, we show how these mean field fluctuations propagate to finite dimensional nonlinear observables for shallow neural networks trained by stochastic gradient descent. Working in the weighted Sobolev space in which the limiting fluctuation process is constructed, we apply a functional Delta method under ordinary Fr{\'e}chet differentiability, without requiring Lions derivatives with respect to the measure variable. We obtain a central limit theorem for the observables and, under a suitable representation of their differentials, an explicit covaria
    
[^36]: 更精简的Transformer可以轻松学会聚类

    Leaner Transformers Can Easily Learn to Cluster

    [https://arxiv.org/abs/2610.09760](https://arxiv.org/abs/2610.09760)

    本文提出了一种嵌入维度仅需 d+⌈log₂k⌉ 但表达能力不变的更精简Transformer来执行k均值聚类的Lloyd算法，并系统刻画了训练Transformer学习聚类算法时影响收敛性与泛化能力的关键因素。

    

    Transformer具备上下文学习能力，某些已知的学习算法可以在模型的前向传播过程中执行。近期的研究表明，Transformer可以精确执行Lloyd算法，对d维空间中的n个点进行k均值聚类，其嵌入维度为 d_emb = d+k（因此需要大小为(d+k)²的注意力投影矩阵）。在本工作中，我们在这一结果的基础上进行了以下拓展：首先，我们提出了一种表达能力相同但规模更小的Transformer，它能以嵌入维度 d_emb = (d + ⌈log₂ k⌉) 执行Lloyd算法。其次，我们在给定聚类任务分布的条件下训练这些Transformer学习聚类算法，并从理论上刻画和通过实验验证了影响基于随机梯度的学习算法收敛性和分布内泛化能力的因素。最后，我们探究了一般性的聚类能力（摘要此处被截断）。

    arXiv:2610.09760v1 Announce Type: new  Abstract: Transformers have in-context learning capabilities, where some known learning algorithms can be executed in the forward pass through the model. Recent work shows that transformers can exactly perform Lloyd's algorithm for $k$-means clustering with $n$ points in $d$ dimensions with an embedding size $d_{\textsf{emb}} = d+k$ (thus, requiring attention projection matrices of size $(d+k)^2$). In this work, we build upon this result in the following ways: First, we present an equally expressive but smaller transformer that executes Lloyd's algorithm with embedding size $d_{\textsf{emb}} = (d + \lceil \log_2 k \rceil)$. Next, we train these transformers to learn the clustering algorithms given a distribution of clustering tasks, and theoretically characterize and empirically validate the factors affecting the convergence and in-distribution generalization of learning algorithms based on stochastic gradients. Finally, we probe the general clust
    
[^37]: EntroPrefill：基于 Renyi 引导的上下文剪枝与条件稳定性保证的检索增强生成方法

    EntroPrefill: Renyi-Guided Context Pruning with Conditional Stability Guarantees for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.09757](https://arxiv.org/abs/2610.09757)

    该论文提出 EntroPrefill，一种由 Renyi 熵引导、带显式注意力质量约束的预填充中期上下文剪枝方法，为检索增强生成提供了可计算的 token 删除上界、自适应剪枝层下依然有效的有限样本观测保证，以及带有显式 Lipschitz 常数的条件 Transformer 扰动稳定性界。

    

    预填充（prefill）中期的剪枝可以减少 Transformer 更深层所需处理的序列长度，但仅凭注意力集中并不能证明被丢弃的上下文是可有可无的。我们将 EntroPrefill 构建为一个由 Renyi 熵引导的提议机制，并与对被丢弃注意力质量的显式约束相结合。通过隔离注意力汇聚点（sink）的正则化头池化，该方法在兼容分组查询注意力（GQA）的同时，揭示了头特化性与最差头覆盖率之间的定量权衡。我们推导了从混合分布到注意力头的删除包络——一个可计算的 token 可移除上界，以及一个在自适应选择剪枝层时依然有效的有限样本观测保证。随后，我们建立了带有显式充分 Lipschitz 常数的条件 Transformer 扰动界，以及一个关于首个 token 决策间隔的推论。一个反例表明，仅凭浅层观测无法推出对未条件化未来输出的无条件保证。系统分析区分……（原文在此截断）

    arXiv:2610.09757v1 Announce Type: new  Abstract: Mid-prefill pruning can reduce the sequence processed by deeper transformer layers, but attention concentration alone does not certify that discarded context is dispensable. We formulate EntroPrefill as a Renyi-guided proposal mechanism coupled to explicit constraints on discarded attention mass. Sink-isolated, regularized head pooling respects grouped-query attention while exposing a quantitative trade-off between specialization and worst-head coverage. We derive a mixture-to-head deletion envelope, a computable upper bound on feasible token removal, and a finite-sample observer guarantee that remains valid when the pruning layer is selected adaptively. We then establish a conditional transformer perturbation bound with explicit sufficient Lipschitz constants and a first-token decision-margin corollary. A counterexample shows why shallow observations alone cannot imply an unconditional future-output guarantee. The systems analysis disti
    
[^38]: 无界特征核与万能核

    Unbounded Characteristic and Universal Kernels

    [https://arxiv.org/abs/2610.09731](https://arxiv.org/abs/2610.09731)

    本文系统研究了无界核的特征性与万能性等表达能力概念，将针对有界核的成熟理论推广至无界核情形。

    

    核方法是机器学习与统计学中最强大的工具之一，拥有大量成功的应用。其巨大成功源于与每个核相关联的灵活函数类——即其再生核希尔伯特空间（RKHS）——这有助于统计分析，同时也源于其计算上的易处理性以及对众多领域的适用性。多种概念（例如特征性、$L_p$-万能性以及积分严格正定性）刻画了核及其RKHS的表达能力，并在理解核方法的统计性质方面发挥着关键作用；对于有界核而言，这些概念及其相互关系已被充分理解。尽管无界核在过去十年中受到了广泛关注（例如在构造基于核的差异度量和依赖性度量时，如最大均值差异（MMD）、希尔伯特-施密特独立性……

    arXiv:2610.09731v1 Announce Type: cross  Abstract: Kernel methods are among the most powerful tools in machine learning and statistics, with a large number of successful applications. Their immense success stems from the flexible function class associated to each kernel---its reproducing kernel Hilbert space (RKHS)---which facilitates statistical analysis, as well as from their computational tractability and applicability to many domains. Multiple notions (such as characteristic, $L_p$-universal, and integrally strictly positive definite) capture the expressivity of kernels and their RKHSs and play a key role in understanding the statistical properties of kernel methods; these concepts and their relations are well-understood for bounded kernels. Even though unbounded kernels have received significant attention over the past decade (for instance, in the construction of kernel-based discrepancy and dependence measures such as the maximum mean discrepancy, the Hilbert-Schmidt independence
    
[^39]: 轮廓算子：从一维投影中识别低秩测度的可辨识性

    The Silhouette Operator: Identifiability of Low-Rank Measures from One-Dimensional Projections

    [https://arxiv.org/abs/2610.09687](https://arxiv.org/abs/2610.09687)

    本文提出“轮廓算子”框架，证明了适当选取的 2k 个一维投影边缘分布足以唯一识别 $\mathbb{R}^2$ 上任何紧支撑的秩不超过 k 的符号测度，且该数量是最优的、投影方向不能任意选取。

    

    arXiv:2610.09687v1 公告类型：cross 摘要：结构化恢复现象，例如压缩感知中的受限等距性质，已经表明高维对象通常可以从极其低维的线性测量中重建。本工作为 $\mathbb{R}^2$ 上的低秩符号测度建立了一个类似的恢复框架，这里的低秩符号测度定义为可以表示为具有一维因子的乘积测度的有限和的测度。该框架基于一类线性算子，称为“轮廓算子”，它将一个测度映射到固定的有限个一维线性推前（pushforward）。主要结果表明：适当选取的 $2k$ 个投影边缘分布足以识别每一个紧支撑的秩 $\le k$ 的符号测度，且该数量是最优的，同时投影方向不能任意选取。该框架还通过建立充分的…（摘要原文在此处截断）扩展到了更高维的乘积测度之和的情形。

    arXiv:2610.09687v1 Announce Type: cross  Abstract: Structured recovery phenomena, such as restricted isometry properties in compressed sensing, have shown that high-dimensional objects can often be reconstructed from remarkably low-dimensional linear measurements. This work develops an analogous recovery framework for low-rank signed measures on $\mathbb{R}^2$, defined here as measures that can be expressed as finite sums of product measures with one-dimensional factors. The framework is based on linear operators, termed "silhouette operators," that map a measure to a fixed finite collection of one-dimensional linear pushforwards. The main results show that a suitably chosen collection of $2k$ projected marginals suffices to identify every compactly supported rank-$\le k$ signed measure, that this number is optimal, and that the projection directions cannot be chosen arbitrarily. The framework is also extended to higher-dimensional sums of product measures by establishing sufficient co
    
[^40]: 高斯-牛顿精度与不定海森矩阵：低成本集合中的一致共存

    Gauss-Newton Accuracy and Indefinite Hessians: Uniform Coexistence in Low-Cost Sets

    [https://arxiv.org/abs/2610.09675](https://arxiv.org/abs/2610.09675)

    本文证明在岭正则化非线性最小二乘中，低成本集合内一致共存两种曲率状态：每个全局极小值点的海森矩阵相对误差低于 $(1+\sqrt{2})/8$，而同一集合中也存在海森矩阵不定、相对误差至少为 $15/8$ 的点，并给出逐点证书与尖锐的相对误差界。

    

    我们研究岭正则化非线性最小二乘问题中高斯-牛顿曲率的精度。在水平集曲率幅值沿精确拟合截面保持局部正则性与持续性的条件下，我们证明了两种曲率状态的一致共存。全局极小值点存在，且每个全局极小值点的相对海森矩阵误差低于 $(1+\sqrt{2})/8$，而同一个低成本集合中还包含一个海森矩阵不定、相对误差至少为 $15/8$ 的点。一个正的岭上限对固定邻域内所有独立的中心和标签扰动、以及不超过该上限的每个正岭权重均适用。当岭权重趋于零时，这些邻域不会收缩。基于当前预测水平集的逐点证书控制海森修正的法向、混合和切向部分。当预测映射与岭参数变化时，我们在所述逐点类别上证明了尖锐的相对误差界。解析例子描述了……

    arXiv:2610.09675v1 Announce Type: new  Abstract: We study the accuracy of Gauss-Newton curvature in ridge-regularized nonlinear least squares. Under local regularity and persistence of level-set curvature magnitude along an exact-fit section, we prove uniform coexistence of two curvature regimes. Global minimizers exist, and every global minimizer has relative Hessian error below $(1+\sqrt2)/8$, while the same low-cost set contains a point with an indefinite Hessian and relative error at least $15/8$. One positive ridge cap works for all independent center and label perturbations in fixed neighborhoods and every positive ridge weight up to the cap. These neighborhoods do not shrink as the ridge weight tends to zero. A pointwise certificate based on the current prediction level set controls the normal, mixed, and tangent parts of the Hessian correction. We prove a sharp relative-error bound over the stated pointwise class when the prediction map and ridge vary. Analytic examples describ
    
[^41]: 聚类稳健的预测驱动推断

    Cluster-Robust Prediction-Powered Inference

    [https://arxiv.org/abs/2610.09601](https://arxiv.org/abs/2610.09601)

    本文提出聚类稳健的PPI++方法，能够处理聚类内存在任意依赖性的部分标注数据，以闭式解形式提供标准误并构建渐近有效的置信区间，无需任何重抽样技术。

    

    数据收集通常成本高昂或存在实际操作上的困难，这既限制了研究人员可以研究的问题，也限制了他们回答这些问题的精确程度。预测驱动推断（PPI）可以通过将标注数据与机器学习预测相结合，减少精确参数估计所需的数据量。然而，忽略聚类内部的依赖性会导致置信区间对真实参数的覆盖频率低于名义水平。我们提出了聚类稳健的PPI++（Cluster-Robust PPI++），它在独立聚类内部存在任意依赖性的情况下，提供闭式解的标准误和渐近有效的置信区间，且无需自助法（bootstrap）或重抽样。我们的核心贡献是能够处理部分标注的聚类，这是一种常见的实证场景，即聚类中同时包含已标注和未标注的单元。由于单元在聚类内部相互依赖，部分标注的聚类违反了PPI+的独立性假设。

    arXiv:2610.09601v1 Announce Type: cross  Abstract: Data collection is often costly or logistically demanding, limiting both the questions researchers can pursue and how precisely they can answer them. Prediction-powered inference (PPI) can reduce the amount of data needed for precise parameter estimation by combining labeled data with machine learning predictions. However, ignoring dependence within clusters can produce confidence intervals that cover the true parameter less often than their nominal rate. We introduce Cluster-Robust PPI++, which provides standard errors in closed form and asymptotically valid confidence intervals under arbitrary dependence within independent clusters, requiring no bootstrap or resampling. Our central contribution is to accommodate partially labeled clusters, a common empirical setting in which clusters contain both labeled and unlabeled units. As units are dependent within clusters, partially labeled clusters violate the independence assumption of PPI+
    
[^42]: 基于动力学朗之万采样的可扩展逻辑高斯过程密度回归

    Scalable Logistic Gaussian Process Density Regression with Kinetic Langevin Sampling

    [https://arxiv.org/abs/2610.09591](https://arxiv.org/abs/2610.09591)

    本文提出一种基于逻辑高斯过程的可扩展贝叶斯条件密度估计方法，通过在强对数凹后验上模拟动力学朗之万动力学进行直接采样，并利用Nyström特征支持具有依赖输入参数的非平稳核。

    

    条件密度估计旨在给出给定协变量下响应变量的完整分布，例如每星系的光度红移估计就需要此类方法。我们开发了一种基于逻辑高斯过程的可扩展贝叶斯估计器。对数条件密度具有可分离的协方差结构：沿响应方向使用Matérn核，通过圆上的截断傅里叶基表示；协变量方向使用由Nyström特征表示的核，可容纳具有依赖输入的幅度和长度尺度的非平稳核。不同于拉普拉斯近似或变分近似，我们直接对该有限特征模型的隐随机场进行采样。在给定超参数的情况下，其后验是强对数凹的，且Hessian矩阵一致有界，我们通过在Kronecker白化坐标系中模拟具有对称小批量分裂的动力学朗之万动力学来从中采样。边缘似然梯度通过费舍尔恒等式从后验期望中获得……

    arXiv:2610.09591v1 Announce Type: cross  Abstract: Conditional density estimation targets the full distribution of a response given covariates, as required, for example, for per-galaxy photometric redshifts. We develop a scalable Bayesian estimator based on the logistic Gaussian process. The log conditional density has a separable covariance: a Mat\'ern kernel along the response, represented in a truncated Fourier basis on a circle, and a covariate kernel represented by Nystr\"om features, which accommodate non-stationary kernels with input-dependent amplitudes and length scales. Instead of a Laplace or variational approximation, we sample the latent field of this finite-feature model. Given the hyperparameters, its posterior is strongly log-concave with a uniformly bounded Hessian, and we draw from it by simulating kinetic Langevin dynamics with symmetric minibatch splitting in Kronecker-whitened coordinates. Marginal-likelihood gradients follow from Fisher's identity as posterior exp
    
[^43]: 弃权式认证：小校准预算下思维链验证器的无分布保证

    Certified by Abstention: Distribution-Free Guarantees for Chain-of-Thought Verifiers at Small Calibration Budgets

    [https://arxiv.org/abs/2610.09541](https://arxiv.org/abs/2610.09541)

    该研究揭示“通过弃权实现有效性”现象——很少触发的验证证书虽形式上有效但每次触发时可能全部出错，并据此为小校准预算下的思维链验证器建立了无分布认证保证及其失效条件分析。

    

    预测思维链（CoT）轨迹是否正确的信号通常以AUC进行比较，但实际部署需要一个带有保证的阈值。我们研究了在几十到几百个标注问题这一现实校准预算下，无分布选择性保证能为CoT验证器提供什么，实验使用了七个开源模型、五种验证器信号和37,000条已评分轨迹。核心观察是“通过弃权实现有效性”：一个以概率 P_fire 发放证书的 (α,δ)-有效程序，仅能将已发放证书的失败概率约束在 δ/P_fire 以内，因此一个很少触发的证书可以在形式上“有效”，却在每次实际使用时都出错。在一个风险已知的模拟中，标准证书在至多0.3%的校准抽样中失败，但在其触发的抽样中失败率高达69%。认证下限以及Benjamini-Hochberg共形选择的格条件解释了为什么证书……（原文摘要在此处截断）

    arXiv:2610.09541v1 Announce Type: cross  Abstract: Signals that predict whether a chain-of-thought (CoT) trace is correct are compared by AUC, but deploying one requires a threshold with a guarantee. We ask what distribution-free selective guarantees deliver for CoT verifiers at realistic calibration budgets of tens to a few hundred labelled problems, using seven open models, five verifier signals and 37,000 graded traces. The central observation is validity by abstention: an $(\alpha,\delta)$-valid procedure that issues a certificate with probability $P_{\rm fire}$ bounds the failure probability of an issued certificate only by $\delta/P_{\rm fire}$, so a certificate that rarely fires can be valid and wrong every time it is used. In a simulation with known risk the standard certificate fails in at most 0.3% of calibration draws but in up to 69% of those in which it fires. A certification floor and a lattice condition for Benjamini-Hochberg conformal selection explain why certificates 
    
[^44]: 非配对典型相关分析

    Unpaired Canonical Correlation Analysis

    [https://arxiv.org/abs/2610.09530](https://arxiv.org/abs/2610.09530)

    提出UCCA方法，通过建立二次分配问题与CCA之间的理论联系，首次实现仅使用非配对数据学习最大化真实潜在配对相关性的线性投影。

    

    典型相关分析（CCA）是多视图共享空间学习的一种基础方法。然而，它对配对数据的严格依赖构成了重大限制，因为这类数据通常难以获取甚至完全不可得。在本文中，我们提出了非配对典型相关分析（UCCA），这是一种新颖的方法，它通过学习线性投影来最大化真实潜在配对的相关性，而无需在训练过程中访问任何配对样本。我们首先建立了将二次分配问题（QAP）与CCA联系起来的理论结果。利用这些理论见解，我们推导出一种仅从非配对数据中最大化相关性的实用方法。据我们所知，UCCA是首个在严格非配对设置下学习最大相关投影的方法。我们在真实世界的多模态数据集上验证了UCCA，证明其在恢复方面显著优于近期的非配对对齐基线方法。

    arXiv:2610.09530v1 Announce Type: cross  Abstract: Canonical Correlation Analysis (CCA) is a fundamental method for multiview shared space learning. However, its strict reliance on paired data poses a significant limitation, as such data is often difficult to obtain or entirely unavailable. In this paper, we present Unpaired CCA (UCCA), a novel method that learns linear projections to maximize the correlation of the true underlying pairing without access to any paired samples during training. We first establish theoretical results connecting the Quadratic Assignment Problem (QAP) to CCA. Leveraging these theoretical insights, we derive a practical method to maximize correlation exclusively from unpaired data. To the best of our knowledge, UCCA is the first approach to learn maximally correlated projections in a strictly unpaired setting. We validate UCCA on real-world multi-modal datasets, demonstrating that it significantly outperforms recent unpaired alignment baselines in recovering
    
[^45]: 反射锚定朗之万算法

    Reflected Anchored Langevin Algorithms

    [https://arxiv.org/abs/2610.09522](https://arxiv.org/abs/2610.09522)

    本文提出反射锚定朗之万动力学（RALD）及其蒙特卡洛算法 RALMC，通过光滑锚定参考势能与状态相关缩放因子，突破了传统方法要求对数密度可微的限制，实现了约束域上不可微目标分布的高效采样，并给出了显式的收敛界与迭代复杂度。

    

    机器学习中用于约束采样的朗之万算法（如基于反射朗之万动力学离散化的投影朗之万蒙特卡洛方法）通常要求对数密度可微，这限制了它们的应用范围。本文提出了反射锚定朗之万动力学（RALD），这是一种能够在约束域上收敛到不可微目标分布的反射扩散过程。该方法使用一个光滑的锚定参考势能，并将其反射朗之万动力学的漂移项与噪声协方差乘以相同的状态相关缩放因子。对该动力学采用带投影的 Euler-Maruyama 离散化，即可得到反射锚定朗之万蒙特卡洛（RALMC）算法。我们证明了 RALMC 在 2-Wasserstein 距离下到目标分布的显式收敛界与迭代复杂度。文中还提供了数值实验，用以验证理论预测并展示该算法的经验性能。

    arXiv:2610.09522v1 Announce Type: cross  Abstract: First order Langevin algorithms for constrained sampling in machine learning, such as projected Langevin Monte Carlo which are based on discretizations of reflected Langevin dynamics, require differentiable log densities that limits their applicability. This paper introduces reflected anchored Langevin dynamics (RALD), a reflected diffusion that converges to non-differentiable targets on constrained domains. The method uses a smooth anchored reference potential and multiplies the drift and noise covariance of its reflected Langevin dynamics by the same state dependent scaling factor. Its Euler-Maruyama discretization with projection gives reflected anchored Langevin Monte Carlo (RALMC) algorithm. We prove explicit convergence bounds and iteration complexity for RALMC in the 2-Wasserstein distance to the target distribution. Numerical experiments are provided to illustrate the theoretical predictions and the empirical performance of the
    
[^46]: 基于伴随方法的随机多尺度生物过程数字孪生校准与最优控制

    Adjoint-Based Calibration and Optimal Control of Stochastic Multiscale Bioprocess Digital Twins

    [https://arxiv.org/abs/2610.09505](https://arxiv.org/abs/2610.09505)

    本文提出了一个基于伴随敏感性分析的偏差感知数字孪生校准与最优控制框架，通过拟似然估计、矩展开和正反向伴随方法量化校准不确定性对策略性能的传播影响，实现了多尺度生物过程的不确定性感知策略优化与自适应实验设计。

    

    我们在生物系统体系（Bio-SoS）范式下，开发了一个面向多尺度生物过程模型的、具有偏差感知能力的数字孪生校准与控制框架。该数字孪生由随机微分方程（SDE）模型表示，并利用拟似然估计和伴随敏感性分析从稀疏的离散观测数据中进行校准。基于SDE生成算子的矩展开刻画了截断引起的参数偏差，而正反向伴随方法则量化了校准不确定性如何传播至价值函数和策略性能。由此得到的参数误差分布既支持面向策略的自适应实验设计，也支持通过二阶高斯平均目标实现的不确定性感知策略优化。我们刻画了所得探索准则的渐近行为，并推导了在优化策略下物理系统的性能表现。为了实现这些想法，我们开发了……（原文摘要在此处截断）

    arXiv:2610.09505v1 Announce Type: cross  Abstract: We develop a bias-aware digital-twin calibration and control framework for multiscale bioprocess models within a biological systems-of-systems (Bio-SoS) paradigm. The digital twin is represented by a stochastic differential equation (SDE) model and calibrated from sparse, discrete observations using quasi-likelihood estimation and adjoint sensitivity analysis. SDE generator-based moment expansions characterize truncation-induced parameter bias, while forward-backward adjoints quantify how calibration uncertainty propagates to value functions and policy performance. The resulting parameter-error distribution supports both policy-directed adaptive experimental design and uncertainty-aware policy optimization through a second-order Gaussian-averaged objective. We characterize the asymptotic behavior of the resulting exploration criterion and derive a physical-system performance under the optimized policy. To implement these ideas, we deve
    
[^47]: DSReg：无需重建即可可证明地恢复个体世界潜在变量

    DSReg: Provably Recovering Individual World Latents without Reconstruction

    [https://arxiv.org/abs/2610.09457](https://arxiv.org/abs/2610.09457)

    该论文提出“结构多样性”条件与依赖稀疏正则化方法DSReg，在无需重建、解码器或标签的情况下，可证明地恢复个体的世界潜在变量。

    

    从非线性ICA到字典学习和因果表征学习，恢复世界个体潜在变量的方法通常通过重建、辅助监督或分布不对称性（如非高斯性）将潜在变量锚定到观测上。而缺乏这些锚定机制的方法，包括联合嵌入预测架构（JEPAs），只能在线性变换的意义上识别潜在状态，因此个体潜在变量仍然是混合不可分的。我们弥合了这一差距：个体世界潜在变量可以在没有重建、没有解码器、没有标签的情况下被可证明地恢复。关键条件是结构多样性：不同的潜在变量会在观测上留下不同的依赖足迹，就像没有两片雪花是相同的一样。基于LeJEPA所提供的线性可辨识性，我们证明在结构多样性条件下，DSReg（依赖稀疏正则化）可以恢复个体世界潜在变量（精确到符号置换），而无需……

    arXiv:2610.09457v1 Announce Type: new  Abstract: Methods that recover individual latent variables of the world, from nonlinear ICA to dictionary learning and causal representation learning, anchor the latents to observations through reconstruction, auxiliary supervision, or distributional asymmetries such as non-Gaussianity. Methods without these anchors, including joint-embedding predictive architectures (JEPAs), identify the latent state only up to a linear transformation, so individual latents remain mixed. We close this gap: individual world latents can be provably recovered with no reconstruction, no decoder, and no labels. The key condition is Structural Diversity: different latents leave distinct dependency footprints on observations, just as no two snowflakes are alike. Building on the linear identifiability that LeJEPA provides, we prove that under Structural Diversity, DSReg (Dependency-Sparsity Regularization) recovers individual world latents up to signed permutation, witho
    
[^48]: 用于条件密度估计的具有精确似然的有限秩逻辑高斯过程

    Finite-Rank Logistic Gaussian Processes with Exact Likelihood for Conditional Density Estimation

    [https://arxiv.org/abs/2610.09452](https://arxiv.org/abs/2610.09452)

    提出ExFR-LGP方法，通过在规则网格上分段线性的有限秩高斯过程先验，使逻辑高斯过程的归一化常数具有闭式解，从而实现了条件密度估计的精确似然贝叶斯推断。

    

    条件密度估计描述响应变量的整个分布如何随协变量变化，在成像研究中还随空间位置变化。逻辑高斯过程（LGP）为此类密度提供了灵活的先验。然而，LGP的归一化常数没有闭式表达，因此现有方法只能近似或替换似然函数。我们提出了精确似然有限秩LGP（ExFR-LGP），该方法将对数密度写成两个二元函数之和：一个是响应变量与协变量的随位置变化的线性指标的函数，另一个是响应变量与位置的函数。每个函数均被赋予一个在规则网格上分段线性的有限秩高斯过程先验，使得归一化常数、条件均值和分位数均可获得闭式表示。在该先验下，后验样本通过吉布斯采样器从精确后验中抽取，该采样器通过椭……（原文截断）

    arXiv:2610.09452v1 Announce Type: cross  Abstract: Conditional density estimation describes how the entire distribution of a response changes with covariates, and in imaging studies also with location. Logistic Gaussian processes (LGP) give a flexible prior for such densities. However, the normalizing constant of an LGP has no closed form, so existing methods approximate or replace the likelihood. We propose the exact likelihood finite-rank LGP (ExFR-LGP), which writes the log density as the sum of two bivariate functions, one of the response and a location-varying linear index of the covariates, and one of the response and the location. Each function is assigned a finite-rank Gaussian process prior that is piecewise linear on a regular grid, enabling the normalizing constant, the conditional mean and the quantiles to enjoy closed form representations. Posterior samples are drawn from the exact posterior under this prior via a Gibbs sampler that updates the Gaussian components by ellip
    
[^49]: 两层线性网络训练的全局指数收敛性

    Global Exponential Convergence of Two-Layer Linear Network Training

    [https://arxiv.org/abs/2610.09356](https://arxiv.org/abs/2610.09356)

    该论文证明了采用光滑PL预测损失训练的宽两层线性网络的全局指数收敛性，其动力学可精确刻画为神经元协方差的有限维Bures流，并给出显式收敛速率（初始协方差为σ²Id时至少为4σ²κ），且该速率在有限宽度采样下保持稳定。

    

    我们在丰富缩放下证明了使用光滑Polyak-Lojasiewicz预测损失训练的宽两层线性网络的全局指数（线性）收敛性，并给出了显式速率。因子中的梯度流恰好以神经元法则协方差的有限维Bures流的形式闭合，其中预测器动力学由隐藏协方差块进行预条件处理。当初始协方差满足谱支撑间隙条件时，平均场守恒律为隐藏预条件块提供了一致的谱下界。该条件涵盖了正定性的情形，同时仍然允许奇异初始化。对于初始协方差 Σ₀ = σ²Id，损失以至少 4σ²κ 的线性速率收敛到全局最小值，其中 κ 是PL常数。我们建立了该速率在有限宽度采样下的稳定性，以及因子梯度的全局收敛性。

    arXiv:2610.09356v1 Announce Type: new  Abstract: We prove global exponential (linear) convergence with an explicit rate in the rich scaling for wide two-layer linear networks trained with smooth Polyak-Lojasiewicz predictor losses. Gradient flow in the factors closes exactly in terms of a finite-dimensional Bures flow of the neuron law covariance, in which the predictor dynamics are preconditioned by hidden covariance blocks. Mean-field conservation laws provide uniform spectral lower bounds on the hidden preconditioning blocks when the initial covariance satisfies a spectral support gap condition. This condition encompasses positive definiteness while still allowing for singular initializations. For an initial covariance $\Sigma_0 = \sigma^2 \mathrm{Id}$, the loss converges to the global minimum with linear rate at least $4\sigma^2\kappa$, where $\kappa$ is the PL constant. We establish stability of this rate under finite-width sampling, as well as global convergence of factor gradien
    
[^50]: 异构输入融合下的良性过拟合

    Benign Overfitting under Heterogeneous Input Fusion

    [https://arxiv.org/abs/2610.09340](https://arxiv.org/abs/2610.09340)

    该论文首次研究异构输入融合下的良性过拟合，发现回归中存在一个与截断阈值无关的全谱协方差证书，可保证良性性质在任意一致的联合协方差融合下得以保持，但这种保护是精确的——在证书范围之外，两个良性的边缘输入块融合后可能变得有害。

    

    良性过拟合在从单一高维输入学习时已被广泛研究，但其在异构输入融合下的行为在很大程度上仍未被探索。我们在异构高斯设计下研究最小范数线性插值，在保持总体任务不变的前提下，比较两个统计相关的输入块与其融合后的表现。对于回归问题，我们识别出一个全谱协方差证书，其渐近状态与截断阈值无关，并证明该证书能被每一个与两个边缘分布相一致的正半定联合协方差所保持。这种保护是精确的，但它并不能扩展到所有良性回归问题：在证书范围之外，两个良性的边缘分布融合后可能变得有害。对于单稀疏高斯分类，正则区域中的良性特征由存活的预测信号与干扰之间的平衡所刻画。

    arXiv:2610.09340v1 Announce Type: new  Abstract: Benign overfitting is extensively studied when learning from a single high-dimensional input, but its behavior under heterogeneous input fusion remains largely unexplored. We study this question for minimum-norm linear interpolation under a heterogeneous Gaussian design, comparing two statistically dependent input blocks with their fusion while holding the underlying population task fixed. For regression, we identify a full-spectrum covariance certificate whose asymptotic status is independent of the cutoff threshold and prove that it is preserved by every positive-semidefinite joint covariance consistent with the two marginals. This protection is sharp, yet it does not extend to all benign regression problems: outside the certified regime, two benign marginals can have a harmful fusion. For one-sparse Gaussian classification, benignity in the regular regime is characterized by the balance between surviving predictive signal and nuisance
    
[^51]: 面向最优正则化的各向异性膨胀几何学

    The Geometry of Anisotropic Dilation for Optimal Regularization

    [https://arxiv.org/abs/2610.09310](https://arxiv.org/abs/2610.09310)

    本文提出“各向异性膨胀”这一保持方向的径向缩放变换，并证明其能以显式方式传递性地调控最优正则化中的径向统计量，从而通过数据变换实现对正则化子几何结构的完全控制与自适应适配。

    

    数据驱动逆问题中的一个核心问题是：如何构造一个能够适应数据分布几何结构的正则化子。最近关于最优正则化的研究表明，在一个广泛的Gibbs类中，与分布 $P$ 最匹配的正则化子由一个单一的、依赖方向的径向汇总统计量 $\rho_P$ 所决定。这提示了一种通过变换数据来控制正则化子几何结构的途径。具体而言：哪些变换能够以简单、显式的方式作用于 $\rho_P$？这些变换能否改善所得变分问题的优化性质，或者将固定的基础正则化子适配到数据上？为了回答这些问题，我们引入了各向异性膨胀——一种保持方向的映射，它通过球面上的一个正函数轮廓，沿每个点的欧氏射线方向对其进行重新缩放。尽管形式简单，这一族映射在径向汇总统计量所构成的空间上具有传递作用：Gibbs类中的任何目标正则化子都可以通过该变换得到（原文在此处截断）。

    arXiv:2610.09310v1 Announce Type: cross  Abstract: A central question in data-driven inverse problems is how to construct a regularizer that adapts to the geometry of the data distribution. Recent work in optimal regularization shows that, within a broad Gibbs class, the regularizer best matched to a distribution $P$ is determined by a single, direction-dependent radial summary statistic $\rho_P$. This suggests a way to control regularizer geometry by transforming the data. In particular, which transformations act on $\rho_P$ in a simple, explicit way? Can they improve the optimization properties of the resulting variational problems or adapt a fixed base regularizer to data? To address these questions, we introduce anisotropic dilation, a direction-preserving map that rescales each point along its Euclidean ray by a positive profile on the sphere. Despite its simplicity, this family acts transitively on the space of radial summary statistics: any target regularizer in the Gibbs class 
    
[^52]: 代理模型之符：测量神经PDE求解器中的数值溯源

    The Symbol of the Surrogate: Measuring Numerical Provenance in Neural PDE Solvers

    [https://arxiv.org/abs/2610.09255](https://arxiv.org/abs/2610.09255)

    该论文提出一种基于傅里叶符号的经验诊断方法，用于判定神经PDE代理模型究竟忠实于精确物理演化还是仅仅模仿训练数值求解器的离散化误差，并发现代理模型几乎完全复制（超过99.8%）了训练格式的振幅与相位误差特征。

    

    神经PDE代理模型是在数值求解器的输出上训练的，而这些输出既包含物理演化，也包含求解器特有的离散化误差。由于代理模型又是使用同一求解器的保留轨迹进行评估的，标准基准无法区分模型是对精确演化的忠实还原，还是对数值格式的模仿。我们引入了一种经验性的傅里叶符号诊断方法，用单个傅里叶模式探测训练好的代理模型的线性化单步算子，并将其与精确演化和训练格式两种参考进行对比。为了解决架构固有的谱偏差问题，我们在具有正交的耗散与色散特征的格式上训练相同的网络，并比较它们学到的算子。在线性平流问题中，学习到的代理模型再现了训练格式的振幅和相位误差，其双格式差异达到了解析预测的完全模仿上限的99.8%以上。

    arXiv:2610.09255v1 Announce Type: new  Abstract: Neural PDE surrogates are trained on numerical solver outputs that contain both physical evolution and solver-specific discretization errors. Because surrogates are also evaluated against held-out trajectories from the same solver, standard benchmarks cannot distinguish fidelity to the exact evolution from imitation of the numerical scheme. We introduce an empirical Fourier-symbol diagnostic that probes a trained surrogate's linearized one-step operator with individual Fourier modes and compares it with both exact-evolution and training-scheme references. To address architectural spectral bias, we train identical networks on schemes with orthogonal dissipative and dispersive signatures and compare their learned operators. In linear advection, the learned surrogates reproduce the training schemes' amplitude and phase errors, with the twin-scheme difference reaching more than 99.8\% of the analytically predicted full-imitation ceiling. The
    
[^53]: 面向推理时对齐的高效Best-of-N策略评估

    Efficient Best-of-N policy evaluation for inference-time alignment

    [https://arxiv.org/abs/2610.09250](https://arxiv.org/abs/2610.09250)

    本文提出了一种无需访问响应似然值的仅样本BoN策略评估框架，利用BoN的顺序统计结构将密度比转化为可由样本估计的得分排名概率，并开发了能跨候选预算高效重用共享辅助样本池的双重稳健估计器BoN-DR，在奖励模型误设下仍保证有效的渐近推断。

    

    Best-of-N（BoN）是一种常见的推理时对齐方法，它从参考模型生成的N个样本中选择得分最高的响应。在仅有样本访问的条件下，从已记录数据中评估BoN策略极具挑战性，因为标准的离策略估计器需要依赖不可获得的响应似然值来计算密度比。在本文中，我们提出了一个仅需样本的框架，用于在无法访问这些似然值的情况下评估和选择BoN策略。我们证明，BoN的顺序统计结构使得所需的密度比可以通过得分排名概率来表达，而这些概率可以仅凭样本进行估计。随后，我们开发了一种BoN策略值的双重稳健估计器，它能够在候选预算之间高效地重用共享的辅助样本池。我们证明了即使在奖励估计器存在误设的情况下，该方法依然能够进行有效的渐近推断，并证明了BoN-DR估计器的有效性。由于更大的预算可以

    arXiv:2610.09250v1 Announce Type: new  Abstract: Best-of-N (BoN) is a common inference-time alignment method that selects the highest-scoring response among N samples from a reference model. Evaluating BoN policies from logged data is challenging under sample-only access because standard off-policy estimators require density ratios that depend on unavailable response likelihoods. In this paper, we propose a sample-only framework for evaluating and selecting BoN policies without access to these likelihoods. We show that the order-statistic structure of BoN allows the required density ratios to be expressed through score-rank probabilities that are estimable from samples alone. We then develop a doubly robust estimator of the BoN policy value (BoN-DR) that efficiently reuses a shared auxiliary sample pool across candidate budgets. We establish valid asymptotic inference even under reward estimator misspecification and prove the efficiency of our BoN-DR estimator. Since larger budgets can
    
[^54]: 协变量偏移下共形预测的草图化校准

    Sketched Calibration for Conformal Prediction under Covariate Shift

    [https://arxiv.org/abs/2610.09208](https://arxiv.org/abs/2610.09208)

    该论文提出草图化校准方法，通过对协变量压缩后再进行加权共形预测，在不增加偏移相关校准代价的前提下校正协变量偏移，并给出了以“泄漏量”刻画的覆盖率保证。

    

    加权共形预测通过使用目标协变量与源协变量之间的似然比对校准分数进行重新加权，从而校正协变量偏移。其代价随着两个协变量分布之间的卡方散度增长，通常随偏移规模呈指数级增长，而且其中很大一部分代价可能花费在那些不影响响应变量的偏移方向上。我们提出了草图化校准方法：使用压缩后协变量 Z=T(X) 的比值进行加权共形预测，该比值用于权重中，并可选地用于分数中。压缩绝不会增加与偏移相关的校准代价，且目标覆盖率至少为 1-α-Δ_T，其中泄漏量 Δ_T 衡量在给定 Z 的条件下，被丢弃的偏移中有多少会以响应变量分布或分数分布发生变化的形式重新出现。泄漏量是被丢弃的偏移与响应变量在草图纤维上的协方差；当草图充分或保留该偏移时，泄漏量消失。

    arXiv:2610.09208v1 Announce Type: cross  Abstract: Weighted conformal prediction corrects for covariate shift by reweighting calibration scores with the likelihood ratio between target and source covariates. Its cost grows with the chi-square divergence between the two covariate laws, typically exponentially in the size of the shift, and much of it can be paid for shift in directions that do not affect the response. We propose sketched calibration: weighted conformal prediction with the ratio of compressed covariates $Z=T(X)$, used in the weights and optionally in the score. Compression never increases the shift-dependent calibration cost, and target coverage is at least $1-\alpha-\Delta_T$, where the leakage $\Delta_T$ measures how much of the discarded shift reappears as a change in the law of the response, or of the score, given $Z$. The leakage is a covariance between the discarded shift and the response on the fibres of the sketch; it vanishes when the sketch is sufficient or reta
    
[^55]: 损失差条件互信息的精度—信息权衡

    An Accuracy--Information Tradeoff for Loss-Difference Conditional Mutual Information

    [https://arxiv.org/abs/2610.09206](https://arxiv.org/abs/2610.09206)

    论文证明了精度与信息之间的权衡：在逻辑损失等光滑凸损失及幂次正则化条件下，任何以最优样本量达到低超额风险的正规学习器，其最坏情况损失差条件互信息必然达到 n 比特量级。

    

    损失差条件互信息（ld-CMI）是泛化上界的超样本层次结构中最小的标准观测量：它衡量学习器的损失差在多大程度上泄露了它训练时使用的是每一对候选样本中的哪一个。已知精度会迫使信息进入模型；而数据处理不等式并不能将此类下界传递到损失上。我们通过对损失差的三个矩进行约束，证明了精度同样会迫使ld-CMI。对于使用在零点斜率非零的光滑凸损失（如逻辑损失）的线性预测器，加上曲率和增长均呈幂次 r≥2 的正则化器，在维度至少随 n 线性增长的缩放符号立方体上的乘积分布中，每个在最优样本量 n≍ε^(-2+2/r) 下于这些分布上期望超额风险至多为 ε 的正规学习器，其最坏情况ld-CMI达到 n 比特量级，且为 Θ(n/(1+(τ/φ…

    arXiv:2610.09206v1 Announce Type: new  Abstract: Loss-difference conditional mutual information (ld-CMI) uses the smallest of the standard observations in the supersample hierarchy of generalization bounds: it measures what a learner's loss differences reveal about which candidate of each pair it was trained on. Accuracy is known to force information into the model; data processing does not carry such lower bounds to losses. We show, by bounding three moments of the loss differences, that accuracy also forces ld-CMI. For linear predictors with a smooth convex loss of nonzero slope at zero, such as the logistic loss, plus a regularizer whose curvature and growth are both of power $r\ge2$, on product distributions over a scaled sign cube in dimension at least linear in $n$, every proper learner with expected excess risk at most $\varepsilon$ on these distributions at the optimal sample size $n\asymp\varepsilon^{-2+2/r}$ has worst-case ld-CMI of order $n$ bits, and $\Theta(n/(1+(\tau/\var
    
[^56]: 线性递归特征机的精确动力学与有限样本轨迹恢复

    Exact Dynamics and Finite-Sample Trajectory Recovery of Linear Recursive Feature Machines

    [https://arxiv.org/abs/2610.09196](https://arxiv.org/abs/2610.09196)

    本文将线性递归特征机与迭代重加权最小二乘的联系推广到岭正则化含噪多输出回归，并证明了其学习到的特征矩阵在每次迭代中都以 $O(\sqrt{d/n})$ 的误差速率逼近无限数据下的理想结果。

    

    递归特征机通过交替执行两个步骤来学习数据的表示：将预测器拟合到数据集上，以及利用平均梯度外积（AGOP）更新该预测器的特征。AGOP与神经网络中特征学习之间的联系，使得线性递归特征机（RFM）成为分析训练过程中表示如何演化的一个简单设定。本文研究了线性RFM在含噪多输出回归中的动力学与统计性质，其中输入数据为各向同性的亚高斯数据，目标由维度为 $d$ 的低秩教师矩阵生成。我们将线性RFM与迭代重加权最小二乘法之间已知的联系，从插值情形扩展到带噪声的岭正则化多输出回归。我们证明了学习到的特征矩阵在每一次迭代中都保持接近其无限数据下的理想对应物。具体而言，对于 $n$ 个样本，我们证明了特征矩阵的误差以 $O(\sqrt{d/n})$ 的速度衰减。（摘要原文在此处被截断）

    arXiv:2610.09196v1 Announce Type: new  Abstract: Recursive feature machines (RFMs) learn representations of data by alternating between fitting a predictor to a dataset and updating features of that predictor using the average gradient outer product (AGOP). Connections between AGOPs and feature learning in neural networks motivate linear RFMs as a simple setting for analyzing how representations evolve during training. Here, we study the dynamics and statistics of linear RFM in noisy multi-output regression with isotropic sub-Gaussian input data and targets generated by a low-rank teacher matrix of dimension $d$. We extend the known connection between linear RFM and iteratively reweighted least squares from the interpolating setting to ridge-regularized multi-output regression with noise. We show that the learned feature matrix remains close to its infinite-data ideal counterpart at every iteration. Namely, for $n$ samples, we show the error in the feature matrix decays as $O(\sqrt{d/n
    
[^57]: 并行扩散采样的下界

    Lower Bounds for Parallel Diffusion Sampling

    [https://arxiv.org/abs/2610.09166](https://arxiv.org/abs/2610.09166)

    本文首次建立了带近似分数的扩散采样的多项式并行轮数下界，证明了对 $R^d$ 中平滑近各向同性高斯混合的采样需要 $\widetilde{\Omega}(d^{1/3})$ 轮、对单位球内各向异性轴对齐盒子的均匀采样需要 $\Omega(d)$ 轮，且这些下界对每轮可进行多项式次查询的任意随机算法均成立。

    

    标准扩散采样器通过对学习到的分数函数进行反复求值来生成样本。并行采样方法试图通过以额外的求值为代价换取更少的顺序轮次，从而加速生成过程。这就引出了一个自然的问题：即使可以同时进行大量的分数查询，顺序依赖性仍然有多大的不可避免性？我们建立了使用近似分数的扩散采样的首个多项式并行轮数下界。具体而言，我们证明了：(1) 对 $R^d$ 中平滑、近各向同性高斯混合进行采样的 $\widetilde{\Omega}(d^{1/3})$ 轮下界；(2) 对包含在单位球内的各向异性轴对齐盒子进行均匀采样的 $\Omega(d)$ 轮下界。这两个下界对任意随机算法均成立，即这些算法每轮可以在任意位置和任意噪声水平上进行多项式次查询，且分数误差为逆多项式级别、总变差精度为常数级别。

    arXiv:2610.09166v1 Announce Type: cross  Abstract: Standard diffusion samplers generate samples through repeated evaluations of a learned score function. Parallel sampling methods seek to accelerate generation by trading additional evaluations for fewer sequential rounds. This raises the question of how much sequential dependence is unavoidable, even when many score queries can be made simultaneously.   We establish the first polynomial parallel-round lower bounds for diffusion sampling with approximate scores. Specifically, we prove (1) a $\widetilde{\Omega}(d^{1/3})$-round lower bound for sampling smooth, near-isotropic Gaussian mixtures in $R^d$, and (2) an $\Omega(d)$-round lower bound for uniform sampling from anisotropic axis-aligned boxes contained in the unit ball. Both bounds hold for arbitrary randomized algorithms making polynomially many queries per round at arbitrary locations and noise levels, with inverse-polynomial score error and constant total variation accuracy. The 
    
[^58]: 面向分布数据聚类数目选择的相对Wasserstein空间深度

    Relative Wasserstein Spatial Depth for Cluster Number Selection in Distributional Data

    [https://arxiv.org/abs/2610.09153](https://arxiv.org/abs/2610.09153)

    提出相对Wasserstein空间深度（RWSD）作为分布数据聚类中选择最优簇数的新准则，并与Wasserstein K-means和K-medians算法结合，在理论上证明了聚类中心和所选簇数的一致性。

    

    当代许多科学领域中的数据，如图像、媒体流、生物医学组学数据，天然地更适合建模为Wasserstein空间中的概率分布，而非欧几里得空间中的点。因此，针对分布的聚类方法需求很高，因为它们为相似对象的分组提供了探索性框架。在Wasserstein空间中对分布值数据进行聚类通常需要预先确定聚类的数目。我们提出相对Wasserstein空间深度（RWSD）作为最优聚类数目的选择准则，该准则将每个分布在自身所属簇中的深度与其在竞争簇中的深度进行比较。我们将该准则与Wasserstein K-means算法或所提出的K-medians算法相结合。在适当的假设下，我们建立了聚类中心以及所选最优聚类数目的一致性。我们还给出了两阶段抽样下的误差率，并刻画……（摘要原文在此处截断）

    arXiv:2610.09153v1 Announce Type: cross  Abstract: Contemporary data in many scientific domains, such as images, media streams, biomedical omics, are naturally modeled as probability distributions in Wasserstein space instead of points in Euclidean space. Consequently, clustering methods for distributions are in high demand, as they provide exploratory frameworks to group objects with similarity. Clustering distribution-valued data in Wasserstein space often requires choosing the number of clusters in advance. We propose Relative Wasserstein Spatial Depth (RWSD) as a selection criterion for optimal number of clusters. It compares the depth of each distribution in its assigned cluster with its depth in competing clusters. We pair this criterion with Wasserstein K-means or a proposed K-medians algorithm. Under suitable assumptions, we establish consistency of the cluster centers and of the selected optimal number of clusters. We also give an error rate under two-stage sampling and charac
    
[^59]: 似然温度调节对变分贝叶斯线性神经网络极限预测矩的影响

    The Impact of Likelihood Tempering on the Limiting Predictive Moments of Variational Bayesian Linear Neural Networks

    [https://arxiv.org/abs/2610.09132](https://arxiv.org/abs/2610.09132)

    本文针对宽贝叶斯神经网络中的“先验主导”退化问题，推导了在温度调度T = τ/M^c下变分贝叶斯线性神经网络的极限预测分布，并揭示了似然温度调节与NNGP后验之间的关系。

    

    在宽贝叶斯神经网络中，高斯平均场变分推断容易出现“先验主导”问题：证据下界（ELBO）的Kullback-Leibler（KL）正则化项压过期望对数似然项，随着网络宽度M的增长，变分预测分布坍缩为先验预测分布。通过将似然提升至1/T次幂（其中温度T < 1）来对似然进行温度调节，等价于将KL项缩放T倍。本文探讨T必须以多快的速度随M减小才能对抗这种退化现象，并在两项之间取得良好平衡。对于具有各向同性高斯先验的单隐层线性网络，我们推导了在形如T = τ/M^c（其中常数τ, c > 0）的调度方案下，当M → ∞时的极限预测分布，并将其与未温度调节的神经网络高斯过程（NNGP）后验——即精确后验的无限宽度极限——进行比较。我们的主要结果是……

    arXiv:2610.09132v1 Announce Type: cross  Abstract: In wide Bayesian neural networks, Gaussian mean-field variational inference is prone to "prior dominance": the Kullback-Leibler (KL) regularization term of the ELBO outweighs the expected log-likelihood, and the variational predictive distribution collapses to the prior predictive as the width $M$ grows. Tempering the likelihood, by raising it to the power $1/T$ for a temperature $T < 1$, is equivalent to scaling the KL term by $T$. We ask in this paper how fast $T$ must decrease with $M$ to counteract this degeneracy and strike a good balance between the two terms. For single-hidden-layer linear networks with isotropic Gaussian priors, we derive the limiting predictive distribution under schedules of the form $T = \tau/M^{c}$, with constants $\tau, c > 0$, as $M \to \infty$ and compare it with the untempered neural network Gaussian process (NNGP) posterior, the infinite-width limit of the exact posterior. Our main result is that the p
    
[^60]: FedRSPO+：一种异构感知的决策导向联邦学习算法

    FedRSPO+: A Heterogeneity-aware Algorithm for Decision-focused Federated Learning

    [https://arxiv.org/abs/2610.09091](https://arxiv.org/abs/2610.09091)

    提出了异构感知的决策导向联邦学习框架FedRSPO+，其核心是基于通过投影平滑决策映射的正则化代理RSPO+，为决策误差和遗憾提供理论上界，从而解决联邦场景下下游目标与可行集异构性导致的训练不稳定问题。

    

    决策导向学习（DFL）为下游优化训练预测模型，但现有方法大多假设数据是集中式的。在跨机构（cross-silo）场景中，联邦学习提供了一种自然的替代方案，然而标准联邦方法仅优化预测质量而非决策质量，并且没有处理下游目标或可行集中的异构性。这种异构性对决策导向学习尤其具有挑战性，因为多面体问题中的微小扰动可能导致最优决策发生不连续的变化，从而使客户端更新和聚合过程变得不稳定。我们提出了FedRSPO+，一个面向决策导向联邦学习的异构感知框架，它建立在RSPO+之上——一种通过投影来平滑决策映射的正则化“先预测后优化”代理方法。我们证明了RSPO+为正则化决策的决策误差和遗憾提供了上界，并且在精确正则化与一致的线性规划解选择条件下，对原始线性规划问题同样成立……

    arXiv:2610.09091v1 Announce Type: new  Abstract: Decision-focused learning (DFL) trains predictive models for downstream optimization, but existing methods largely assume centralized data. In cross-silo settings, federated learning offers a natural alternative, yet standard federated methods optimize prediction over decision quality and do not address heterogeneity in downstream objectives or feasible sets. This heterogeneity is especially challenging for DFL because small perturbations in polyhedral problems can cause discontinuous changes in optimal decisions, destabilizing client updates and aggregation. We propose FedRSPO+, a heterogeneity-aware framework for decision-focused federated learning, built on RSPO+, a regularized predict-then-optimize surrogate that smooths the decision map through projection. We show that RSPO+ upper bounds decision error and regret for the regularized decision and, under exact regularization and consistent LP solution selection, for the original LP de
    
[^61]: 协变量依赖的多元序数偏好联合建模及其与比较模型的联系

    Covariate-dependent Joint Modeling of Multivariate Ordinal Preferences and Its Connections with Comparison Models

    [https://arxiv.org/abs/2610.09070](https://arxiv.org/abs/2610.09070)

    本文提出了一种协变量依赖的多元序数偏好联合建模方法，避免了传统方法单独处理各属性或将数据粗化为胜负比较所造成的信息损失，并建立了该方法与 Bradley–Terry、Plackett–Luce 等比较模型之间的联系。

    

    arXiv:2610.09070v1 公告类型：交叉 摘要：带有协变量的多元序数数据在从语言模型与人类偏好对齐到推荐系统等各类问题中被广泛收集。例如，像 MovieLens 这样的数据集包含人类用户对多部电影以1至5量表给出的评分，以及用户的人口统计信息（如年龄或性别）。类似地，像 HelpSteer 这样的数据集收集人类对大语言模型回复的多个属性（如“有用性”或“冗长性”）在序数量表上的反馈，其协变量取决于大语言模型的提示—响应对。遗憾的是，这些数据的标准建模方法（a）对各个属性单独建模而非联合建模，并且（b）经常将数据转换为成对或列表式的胜负比较，以拟合诸如 Bradley–Terry 和 Plackett–Luce 等模型。这两种做法都会导致对实际观测数据的信息粗化，我们通过一种联合的协变量依赖的建模方法来解决这一问题……

    arXiv:2610.09070v1 Announce Type: cross  Abstract: Multivariate ordinal data along with covariates are commonly collected in problems ranging from alignment of language models with human preferences, as well as in recommender systems. For example, data sets such as MovieLens contain several movies rated on a scale 1--5 by human users, along with their demographic information such as age or gender. Similarly, data sets such as HelpSteer collect human feedback on several attributes such as "helpfulness" or "verbosity" of LLM response on an ordinal scale, with covariates depending on the LLM prompt--response pairs. Unfortunately, the standard approaches for modeling these data (a) look at the attributes individually rather than jointly, and (b) often convert the data into pairwise or list-wise win--loss comparisons for fitting models such as Bradley--Terry and Plackett--Luce. Both of these lead to a coarsening of what is actually observed, which we address via a joint covariate-dependent 
    
[^62]: 基于条件扩散模型学习跳跃扩散过程的转移核

    Learning Transition Kernels of Jump-Diffusion Processes with Conditional Diffusion Models

    [https://arxiv.org/abs/2610.09045](https://arxiv.org/abs/2610.09045)

    该论文提出用条件扩散模型学习时齐跳跃扩散过程的转移核，在理论上给出了条件分数估计误差和真实与生成路径分布间KL散度的非渐近界，并在合成与真实数据上验证了其在样本路径生成和概率预测任务中的有效性。

    

    我们研究了利用条件扩散模型学习时齐跳跃扩散过程转移核的问题，目标是从由N条独立轨迹组成的训练数据中生成新的样本路径，这些轨迹在高频离散时间网格上被观测。在理论方面，我们为条件分数估计误差以及真实离散观测路径分布与生成路径分布之间的KL散度建立了非渐近界。在数值方面，我们首先在合成数据上评估所提方法以验证理论发现，并将其性能与Gao等人（2025）的方法进行基准对比。随后我们将该方法应用于真实世界数据，并考察其在概率预测任务上的表现。

    arXiv:2610.09045v1 Announce Type: cross  Abstract: We study the problem of learning transition kernels for time-homogeneous jump-diffusion processes using conditional diffusion models, with the goal of generating new sample paths from training data consisting of N independent trajectories observed on a high-frequency discrete time grid. On the theoretical side, we establish non-asymptotic bounds for the conditional score estimation error and for the KL divergence between the laws of the true and generated discretely observed paths. On the numerical side, we first evaluate our method on synthetic data to assess the theoretical findings and benchmark its performance against the approach of Gao et al. (2025). We then apply our method to real-world data and investigate its performance on a probabilistic forecasting task.
    
[^63]: 基于随机矩阵理论的随机特征网络中的二次弱到强泛化

    Quadratic Weak-to-Strong Generalization in Random Feature Networks via Random Matrix Theory

    [https://arxiv.org/abs/2610.09044](https://arxiv.org/abs/2610.09044)

    本文利用随机矩阵理论证明，在两层随机特征网络中，由弱教师模型训练出的强学生模型误差为教师误差的平方，实现了二次级的弱到强泛化改进。

    

    弱到强泛化是指强学生模型在使用弱教师模型所产生的标签进行训练后，能够比教师模型具有更好泛化能力的现象。本文在两层随机特征网络中研究这一现象，其中模型的强度由其宽度决定。利用随机矩阵理论的工具，我们推导了最优训练的教师模型和通过梯度流训练的学生模型的总体误差的确定性等价形式。对于ReLU激活函数和纯球谐函数目标，我们在高斯普适性假设下获得了精确的渐近结果，展示出二次改进：学生误差按教师误差的平方比例缩放。这些结果达到了Medvedev等人（2025）给出的一般下界。我们还分析了学生在更一般的停止时间以及支持在多个谐波次数上的目标下的表现，刻画了弱到强泛化出现的区域。

    arXiv:2610.09044v1 Announce Type: cross  Abstract: Weak-to-strong generalization is the phenomenon where a strong student model trained with labels produced by a weak teacher model is able to generalize better than the teacher. In this paper, we study this phenomenon in two-layer random feature networks where the model strength is determined by its width. Using tools from random matrix theory, we derive deterministic equivalents for the population errors of an optimally trained teacher and a student trained with gradient flow. For ReLU activation and a pure spherical harmonic target, we obtain sharp asymptotics under a Gaussian universality assumption, showing a quadratic improvement: the student error scales as the square of the teacher error. These results attain the general lower bound of Medvedev at al (2025). We also analyze how the student behaves under more general stopping times and targets supported on multiple harmonic degrees, characterizing the regimes in which weak-to-stro
    
[^64]: 谨慎裁判：安全高效的人机协作决策

    Careful Judge: Safe and Efficient Human-AI Collaborative Decision Making

    [https://arxiv.org/abs/2610.09043](https://arxiv.org/abs/2610.09043)

    CARE是一个端到端的人机协作决策框架，通过新颖的自适应校准模块在任何时刻保证风险控制，并持续从人工反馈中学习，以更少的人工查询实现更高的自动化水平。

    

    在人机协作决策中，人工审核可以防止不安全的AI决策，但每一次人工判断的成本都很高。将AI弃权后的人工干预视为一次性的后备手段，会错失改进未来AI决策以实现更高自动化的机会；然而，AI若从选择性查询的人工反馈中进行自适应学习，又会破坏为旧模型校准的安全护栏。我们通过CARE——校准自适应纠正与升级——来应对这一挑战。CARE是一个端到端的流水线，将AI模型与人类审核员相结合，以保证安全且符合人类意图的决策，同时持续从人类反馈中学习，以更少的人工查询实现更高的自动化水平。CARE是有原则的、通用的、模块化的，可适用于任何黑盒AI模型。我们新颖的自适应校准模块可在任何时间步骤为任何纠正模块提供风险控制保证。我们还进一步展示了当AI模型（摘要内容不完整）……CARE如何提升查询效率。

    arXiv:2610.09043v1 Announce Type: cross  Abstract: In human-AI collaborative decision making, human review can prevent unsafe AI decisions, but each human judgment is costly. Treating human intervention after AI abstention as a one-off fallback misses the opportunity to improve future AI decisions for greater automation, yet AI adaptively learning from selectively queried human feedback breaks safety guardrails calibrated for old models. We approach this challenge with CARE---calibrated adaptive rectification and escalation---an end-to-end pipeline that combines AI models and human reviewers to guarantee safe, human-aligned decisions, while continuously learning from human feedback to achieve greater automation with fewer human queries. CARE is principled, general, modular, and works with any black-box AI model. Our novel adaptive calibration module guarantees risk control at every time step for any rectification module. We further show how CARE improves query efficiency when the AI mo
    
[^65]: 一般比较图上MLE与Rank Centrality的图单调逐项保证

    Graph-monotone entrywise guarantees for MLE and Rank Centrality on general comparison graphs

    [https://arxiv.org/abs/2610.09030](https://arxiv.org/abs/2610.09030)

    本文在Bradley--Terry--Luce模型下证明，MLE和Rank Centrality在任意固定比较图上均能达到以计数加权图代数连通度决定的逐项误差保证，且该保证随比较数据的增加单调改善。

    

    成对比较被广泛用于推断潜在分数和识别排名靠前的项目。尽管在均匀采样下已有精确的统计保证，但真实数据往往产生不规则的比较图，且各对之间的观测数量存在异质性。本文在Bradley--Terry--Luce模型下，以最少的假设研究任意固定的比较图。我们证明，标准最大似然估计量（MLE）和Rank Centrality均能以高概率达到 $1/\sqrt{\lambda_{\mathcal{D}}}$ 量级的逐项误差率（含对数因子和动态范围因子），其中 $\lambda_{\mathcal{D}}$ 是以比较计数加权的观测图的代数连通度。该保证是图单调的，因为当添加更多比较时 $\lambda_{\mathcal{D}}$ 不会减小。在异质采样下，即比较对以不相等的概率独立采样时，我们的保证……

    arXiv:2610.09030v1 Announce Type: cross  Abstract: Pairwise comparisons are widely used to infer latent scores and identify top-ranked items. Although sharp statistical guarantees are available under uniform sampling, real data often induce irregular comparison graphs with heterogeneous observation counts across pairs. In this paper, we study an arbitrary fixed comparison graph under the Bradley--Terry--Luce model with minimal assumptions. We prove that both the standard maximum likelihood estimator and Rank Centrality achieve a high-probability entrywise error rate of order $1/\sqrt{\lambda_{\mathcal{D}}}$ up to logarithmic and dynamic-range factors, where $\lambda_{\mathcal{D}}$ is the algebraic connectivity of the count-weighted observation graph. This guarantee is graph-monotone since $\lambda_{\mathcal{D}}$ cannot decrease when additional comparisons are added. Under heterogeneous sampling, where comparison pairs are sampled independently with unequal probabilities, our guarantee 
    
[^66]: LASER：用于支撑约束熵正则化离线强化学习的潜在空间伴随匹配

    LASER: Latent Space Adjoint Matching for Support-Constrained Entropy-Regularized Offline RL

    [https://arxiv.org/abs/2610.08989](https://arxiv.org/abs/2610.08989)

    LASER通过潜在空间伴随匹配实现熵正则化的潜在空间离线强化学习，既防止了策略坍缩成单一脆弱模式，又避免了时间反向传播，在40个不同数据质量的OGBench任务上取得了优异表现。

    

    虽然离线强化学习（RL）能够在无需昂贵的在线交互的情况下从静态数据集中进行策略优化，但其性能仍然受到执行分布外（OOD）动作风险的制约。近期的方法通过流匹配学习一个行为克隆策略，然后在其受约束的潜在空间内执行强化学习，从而缓解了这一问题。然而，朴素地优化潜在策略很容易导致策略坍缩成脆弱的单一模式，或利用学习到的评论家网络中尖锐的伪影。在这项工作中，我们发现熵正则化对于在潜在空间强化学习中应对这些挑战至关重要。我们提出了LASER，一种新颖的离线强化学习算法，它应用潜在空间伴随匹配来实现基于强表达能力的流策略的熵正则化潜在空间强化学习，同时避免了时间反向传播。通过在40个具有不同数据集质量的具有挑战性的OGBench任务上的全面实验，我们展示了（摘要在此处截断）

    arXiv:2610.08989v1 Announce Type: new  Abstract: While offline reinforcement learning (RL) enables policy optimization from static datasets without costly online interaction, it remains bottlenecked by the risk of executing out-of-distribution (OOD) actions. Recent approaches mitigate this by learning a behavior-cloning policy through flow matching and then performing RL within its constrained latent space. However, naively optimizing the latent policy can easily cause the policy to collapse into a brittle mode or exploit sharp artifacts of the learned critic. In this work, we find that entropy regularization is essential in latent-space RL for addressing these challenges. We introduce LASER, a novel offline RL algorithm that applies latent-space adjoint matching to achieve entropy-regularized latent-space RL with expressive flow policies while avoiding backpropagation through time. Through comprehensive experiments on 40 challenging OGBench tasks with varying dataset qualities, we sho
    
[^67]: 最佳优化器取决于批量大小

    The Best Optimizer Depends on Batch Size

    [https://arxiv.org/abs/2610.08975](https://arxiv.org/abs/2610.08975)

    该论文挑战了“某一批量大小下最佳的优化器在其他批量大小下也最佳”的常见假设，证明Muon缺乏一致的缩放规则，且即使经过大量超参数调优，语言模型预训练的最佳优化器仍会随批量大小而改变。

    

    大量新的自适应优化器被设计用于高效估计和利用小批量梯度统计量来塑造参数更新，但它们通常只在单一批量大小下进行基准测试。超参数缩放规则承诺在批量大小和梯度噪声变化时保持性能，这暗示着在某一批量大小下最佳的优化器在另一批量大小下也应保持最佳。我们通过以下发现挑战了这种开发和评估优化器的方法：(1) Muon没有一种原则性的缩放规则能在各种训练设置中保持一致有效；(2) 即使经过大量的超参数调优，语言模型预训练的最佳优化器也会随批量大小而变化。

    arXiv:2610.08975v1 Announce Type: new  Abstract: A plethora of new adaptive optimizers are designed to efficiently estimate and use minibatch gradient statistics to shape parameter updates, but they are typically benchmarked at a single batch size. Hyperparameter scaling rules promise to preserve performance as batch size and gradient noise change, suggesting that the best optimizer at one batch size should remain the best at another. We challenge this approach to developing and evaluating optimizers by showing: (1) no principled scaling rule for Muon works consistently across training settings, and (2) the best optimizer for language model pretraining changes with batch size even after extensive hyperparameter tuning.
    
[^68]: 趁它们沉睡时工作：利用评估延迟实现全贝叶斯优化

    Work While They Sleep: Exploiting Evaluation Latency for Fully Bayesian Optimization

    [https://arxiv.org/abs/2610.08969](https://arxiv.org/abs/2610.08969)

    该论文提出ELF-BO算法，巧妙利用贝叶斯优化中昂贵目标函数评估期间的等待时间，提前并行计算全贝叶斯代理模型，从而在不增加额外时间成本的情况下获得更好的不确定性估计和优化性能。

    

    黑盒优化问题在科学与工程领域无处不在，通常涉及代价高昂的目标函数。这种目标评估延迟在优化过程中带来两个后果：（i）目标函数评估主导了整个执行时间；（ii）样本高效的算法对于加速开发、避免资源浪费至关重要。贝叶斯优化（BO）方法是规划器为推荐下一个尝试点的事实上的标准选择。标准的BO采用点估计来拟合代理模型的超参数；而完全贝叶斯方法则通过模型平均来考虑超参数的不确定性，从而获得更好的不确定性估计——这在BO中普遍存在的低数据场景下非常有用。然而，该方法往往计算代价过高，因此很少被使用。在这项工作中，我们提出ELF-BO，一种利用目标函数评估延迟来提前启动全贝叶斯代理模型计算的算法。

    arXiv:2610.08969v1 Announce Type: new  Abstract: Black-box optimization problems are ubiquitous across science and engineering, often dealing with expensive objective functions. This objective latency has two consequences during optimization: (i) the objective evaluation dominates execution time, and (ii) sample-efficient algorithms are crucial to accelerate development and avoid wasting resources. Bayesian optimization (BO) methods are the \textit{de facto} choice of planners for suggesting the next point to try. Standard BO fits the surrogate model's hyperparameters with a point estimate. Alternatively, a fully Bayesian approach uses model averaging to account for uncertainty over the hyperparameters, leading to better uncertainty estimates---useful in the low-data regime that is pervasive in BO. However, it is often prohibitively expensive and thus rarely used. In this work, we propose ELF-BO, an algorithm that uses the objective evaluation latency to headstart the computation of th
    
[^69]: Wasserstein空间中光滑势-相互作用能量的信赖域优化

    Trust-Region Optimization for Smooth Potential-Interaction Energies in Wasserstein Space

    [https://arxiv.org/abs/2610.08883](https://arxiv.org/abs/2610.08883)

    该论文提出了Wasserstein空间上光滑势-相互作用能量的信赖域优化方法，通过推前曲线上的二次模型、$L^2(\rho)$ 步长半径以及带显式自伴二阶变分算子的Steihaug-Toint子求解器，在温和条件下证明了目标函数单调不增且Wasserstein梯度范数收敛于零。

    

    寻找相互作用粒子的低能量构型以及逼近概率分布，都会导致在Wasserstein空间中最小化势-相互作用能量。这些能量可能是非凸的，因此在利用二阶信息的同时控制局部近似的可靠性变得十分重要。我们研究了在具有有限二阶矩的概率测度的Wasserstein空间上，光滑势-相互作用能量的信赖域优化。该方法使用沿推前曲线的二次模型、$L^2(\rho)$ 步长半径，以及带有显式自伴二阶变分算子的Steihaug-Toint子求解器。比值检验决定步骤的接受与否并指导半径更新。在能量存在下界、且势函数与相互作用核的Hessian全局有界的条件下，我们证明了目标函数单调不增、Wasserstein梯度范数收敛于零，并（得到ε-驻点，摘要原文在此处截断）。

    arXiv:2610.08883v1 Announce Type: cross  Abstract: Finding low-energy configurations of interacting particles and approximating probability distributions lead to the minimization of potential-interaction energies in Wasserstein space. These energies can be nonconvex, making it important to exploit second-order information while controlling the reliability of local approximations. We study trust-region optimization of smooth potential-interaction energies on the Wasserstein space of probability measures with finite second moment. The method uses a quadratic model along pushforward curves, an $L^2(\rho)$ step radius, and a Steihaug-Toint subsolver with an explicit self-adjoint second-variation operator. A ratio test determines acceptance and guides the radius update. Under a lower energy bound and globally bounded Hessians of the potential and interaction kernel, we prove that the objective is nonincreasing, the Wasserstein-gradient norms converge to zero, and an $\varepsilon$-stationary
    
[^70]: 慢胜快于Kesten-Stigum阈值：稀疏随机块模型中信息-计算差距的极小极大、Fisher信息与置信传播刻画

    Slow Beats Fast at the Kesten-Stigum Threshold: Minimax, Fisher-Information and Belief-Propagation Characterizations of the Information-Computation Gap in Sparse Stochastic Block Models

    [https://arxiv.org/abs/2610.08872](https://arxiv.org/abs/2610.08872)

    该论文通过统计决策理论、Fisher信息和置信传播对稀疏随机块模型的Kesten-Stigum阈值给出三种刻画，证明在 q≥5 时阈值下方存在信息-计算差距：多项式低度算法渐近无法超越平凡风险，而指数时间算法却能成功。

    

    我们通过统计决策理论和Fisher信息，研究具有q个社区、平均度d和信号强度λ的稀疏对称随机块模型中的社区恢复问题，并得到Kesten-Stigum阈值 dλ²=1 及其下方信息-计算差距的三种刻画。首先，在任意给定的社区规模分布上，任何在对称平均和顶点重标记下封闭的规则类的极小极大风险等于其在均匀先验下的贝叶斯风险；后验均值是唯一的贝叶斯规则且是可容许的，而度数为D的多项式规则的贝叶斯风险等于平凡风险乘以 1-Corr_D²。结合已知的低度方法和信息论结果，这一差距被表述为一个最坏情形的命题：对于 q≥5，在阈值下方存在一个窗口区域，其中任何低度规则都无法渐近地超越平凡风险，而指数时间算法则可以在某些标签集上实现超越。

    arXiv:2610.08872v1 Announce Type: cross  Abstract: We study community recovery in the sparse symmetric stochastic block model with $q$ communities, average degree $d$ and signal strength $\lambda$ through statistical decision theory and Fisher information, and obtain three characterizations of the Kesten-Stigum threshold $d\lambda^2=1$ and of the information-computation gap below it. First, on each community-size profile the minimax risk of any class of rules closed under averaging and vertex relabeling equals its Bayes risk under the uniform prior; the posterior mean is the unique Bayes rule and is admissible, and the Bayes risk of degree-$D$ polynomial rules is the trivial risk times $1-\mathrm{Corr}_D^2$. Combined with known low-degree and information-theoretic results, this gives the gap as a worst-case statement: for $q\ge 5$ there is a window below the threshold in which no low-degree rule beats the trivial risk asymptotically, while an exponential-time rule does on a set of labe
    
[^71]: 只为FUNS：基于大语言模型引导的时空图节点生成方法用于预测未观测节点状态

    Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States

    [https://arxiv.org/abs/2610.08818](https://arxiv.org/abs/2610.08818)

    该论文提出GenST框架，将未观测节点状态预测（FUNS）重新定义为时空图上的条件生成任务，创新性地利用微调后的大语言模型从节点描述中提取语义特征作为语义桥梁，以弥补缺失的时空信号。

    

    时空预测是物流、城市规划和智能交通系统的基石。然而，受部署成本和维护资源的限制，传感器网络往往缺乏全面的空间覆盖，这使得预测未观测节点状态（FUNS）成为一项至关重要却又极具挑战性的任务。传统模型依赖历史观测数据，在遇到没有先前记录的节点时通常会表现失常。为解决这一问题，我们将该问题重新定义为时空图上的条件生成任务，并提出GenST框架，该框架引入大语言模型（LLMs）作为语义桥梁，利用经过微调的预训练LLM从节点描述（如功能分区和道路网络结构）中提取丰富的语义特征，以弥补缺失的时空信号。具体而言，我们设计了一个两阶段生成架构：时空变分自编码器（VAE）首先压缩……

    arXiv:2610.08818v1 Announce Type: cross  Abstract: Spatio-temporal forecasting is a cornerstone of logistics, urban planning, and intelligent transportation systems. However, constrained by deployment costs and maintenance resources, sensor networks often lack comprehensive spatial coverage, rendering Forecast Unobserved Node States (FUNS) a critical yet formidable challenge. Conventional models rely on historical observations and typically falter when encountering nodes without prior records. To address this, we redefine the problem as a conditional generation task on spatio-temporal graphs and propose GenST, a framework that introduces Large Language Models (LLMs) as a semantic bridge, leveraging a pre-trained LLM fine-tuned to extract rich semantic features from node descriptions, such as functional zones and road network structures, to compensate for missing spatio-temporal signals. Specifically, we design a two-stage generative architecture: a Spatio-Temporal VAE first compresses 
    
[^72]: DeepAJM：面向不规则采样数据的深度关联联合模型

    DeepAJM: Deep Association Joint Model for Irregularly Sampled data

    [https://arxiv.org/abs/2610.07388](https://arxiv.org/abs/2610.07388)

    提出 DeepAJM——一种无需参数假设的深度联合模型，利用编码器-解码器架构学习不规则采样的时变协变量轨迹的潜在结构，并通过部分可解释的关联结构将其与生存结局关联，从而改进生存预测。

    

    联合模型同时建模纵向结局与生存结局，利用患者纵向轨迹中的模式来改进生存结局的预测。然而，经典的参数化联合模型依赖于固定的参数假设，在模型误设和样本量较小的情况下容易产生偏差。我们提出了一种深度联合模型 DeepAJM，它不需要任何参数假设，同时保留了部分可解释的、针对每个纵向结局的关联结构。该联合模型采用编码器-解码器（序列到序列）架构来学习患者时变协变量轨迹中的潜在结构。模型通过一个学习得到的可解释关联结构将纵向过程与生存过程联系起来，其中解码器输出的每个纵向结果在贡献于（生存模型的）风险评分之前，会先由基线协变量进行重新调制……

    arXiv:2610.07388v1 Announce Type: cross  Abstract: Joint Models simultaneously model longitudinal and survival outcomes, leveraging patterns in patients' longitudinal trajectory to improve the prediction of survival outcomes. The classical parametric joint models, however, rely on fixed parametric assumptions, making them susceptible to bias under model misspecification and smaller sample sizes. We propose a deep joint model, DeepAJM, that does not require any parametric assumptions, while retaining a partially interpretable, per-longitudinal-outcome association structure. The joint model uses an encoder-decoder (sequence-to-sequence) architecture to learn the latent structure in patients' time-varying covariate trajectories. The model links the longitudinal processes to the survival processes through a learned interpretable association structure, in which each longitudinal output from the decoder gets remodulated by baseline covariates before it contributes to the risk scores from the
    
[^73]: Fréchet Inception 距离的样本最优估计

    Sample-Optimal Estimation of the Fr\'echet Inception Distance

    [https://arxiv.org/abs/2610.07114](https://arxiv.org/abs/2610.07114)

    该论文针对FID估计中的有限样本偏差问题，证明了插件估计器的紧致偏差与方差界并确立其平方级（d²）样本复杂度，同时将FID∞估计器推广到任意阶外推方法以实现去偏估计。

    

    Fréchet Inception 距离（FID）被广泛用于评估生成模型，但其经验插件估计器存在有限样本偏差 [BSAG18, CF20]。我们研究了在一个分布已知的情况下，估计具有有界均值距离和协方差的 $d$ 维高斯分布之间的 FID 至误差 $\epsilon$ 所需的样本复杂度 $n$。我们的贡献有三点：(1) 我们为经验插件估计器建立了紧致的有限样本偏差界 $\Theta(\frac{d^2}{n})$ 和方差界 $\Theta(\frac{d}{n} + \frac{d^2}{n^2})$，从而确立了 $\gtrsim d^2$ 的样本复杂度。(2) 为了对经验插件估计器进行去偏，我们将 [CF20] 的 ${\rm FID}_\infty$ 估计器推广到任意阶数 $k$ 的外推方法，并进一步在我们的框架下证明了任意 $k$ 阶外推的紧致偏差界 $\Theta(\frac{d^{k+2}}{n^{k+1}})$ 和方差界 $\Theta(\frac{d}{n} + \frac{d^2}{n^2})$。(3) 我们引入

    arXiv:2610.07114v1 Announce Type: new  Abstract: The Fr\'echet Inception Distance (FID) is widely used to evaluate generative models, but its empirical plug-in estimator suffers from finite-sample bias [BSAG18, CF20]. We study the sample complexity $n$ of estimating FID to error $\epsilon$ between $d$-dimensional Gaussians with bounded mean distance and covariances, when one distribution is known. Our contributions are threefold. (1) We establish tight finite-sample $\Theta(\frac{d^2}{n})$ bias and $\Theta(\frac{d}{n} + \frac {d^2} {n^2})$ variance bounds for the empirical plug-in estimator, establishing a $\gtrsim d^2$ sample complexity. (2) To debias the empirical plug-in estimator, we generalize the ${\rm FID}_\infty$ estimator of [CF20] to extrapolation methods of arbitrary order $k$. We further prove tight bias and variance bounds of $\Theta(\frac{d^{k + 2}}{n^{k + 1}})$ and $\Theta(\frac d n + \frac{d^2}{n^2})$ for any order-$k$ extrapolation under our framework. (3) We introduce
    
[^74]: 单次反事实补救的有符号几何：路径上有效性与带符号曲率判据

    The Signed Geometry of One-Shot Recourse: On-Path Validity and the Signed-Curvature Criterion

    [https://arxiv.org/abs/2609.36252](https://arxiv.org/abs/2609.36252)

    该论文证明单次解析式反事实补救能否一步成功由路径曲率 $\kappa=\hat g^\top\nabla^2 f(x)\hat g$ 的符号决定（非负则有效），给出仅凭分数与梯度的规则不可避免存在 $Kd_p^2/\|\nabla f(x)\|$ 量级过冲的下界，并证明在利普希茨曲率下于承诺点评估一次分数即可达到极小极大最优有效性。

    

    闭式（解析式）补救方法将一个被分类器拒绝的用户沿着分类器分数 $f$ 的单位梯度 $\hat g$ 移动承诺距离 $d_p=|f(x)|/\|\nabla f(x)\|$，在该处线性化后的分数恰好降为零。我们研究这一单步操作何时会成功，以及额外的模型查询能带来什么改变。在主阶近似下，该步恰好落在有利一侧当且仅当路径曲率 $\kappa=\hat g^\top\nabla^2 f(x)\,\hat g$ 非负。在80个浅层模型上，被拒用户中一步落在有利侧的比例与 $\kappa\ge0$ 的比例相关系数高达 $r=0.985$，但在 Fashion-MNIST 上前者平均比后者低8.2个百分点。任何仅使用分数值和梯度的规则，都不可能对所有路径曲率以 $K$ 为界的分数都保持有效，除非对其中某些分数产生 $Kd_p^2/\|\nabla f(x)\|$ 量级的过冲。当曲率还是利普希茨连续且步长较短时，在承诺点处对 $f$ 的一次评估即可达到极小极大（minimax）最优。

    arXiv:2609.36252v1 Announce Type: new  Abstract: Closed-form recourse moves a rejected user along the unit gradient $\hat g$ of the classifier score $f$ by the promised distance $d_p=|f(x)|/\|\nabla f(x)\|$, at which the linearized score reaches zero. We ask when this one-shot step succeeds and what additional model queries change. To leading order the step ends on the favorable side exactly when the path curvature $\kappa=\hat g^\top\nabla^2 f(x)\,\hat g$ is nonnegative. Across 80 shallow models, the fraction of rejected users whose step ends there and the fraction with $\kappa\ge0$ correlate at $r=0.985$, although on Fashion-MNIST the first falls below the second by 8.2 points on average. No rule that uses only the score value and gradient can be valid for every score with path curvature bounded by $K$ without overshooting some by order $Kd_p^2/\|\nabla f(x)\|$. When the curvature is also Lipschitz and the step is short, one evaluation of $f$ at the promised point attains the minimax
    
[^75]: 面向基于采样的潜在规划的控制几何拉直方法

    Control-Geometry Straightening for Sampling-Based Latent Planning

    [https://arxiv.org/abs/2609.35603](https://arxiv.org/abs/2609.35603)

    提出控制几何拉直（CGS）这一辅助损失，通过将动作间余弦相似度与潜在差异对齐来学习对规划器友好的表示，从而提升基于采样的潜在规划的优化效率，并给出相应理论保证。

    

    联合嵌入预测架构使基于潜在世界模型的规划成为可能，但仅有精确的转移预测并不能保证规划目标易于优化。我们提出了控制几何拉直（Control-Geometry Straightening, CGS），这是一种单一的辅助损失函数，通过直接拉直控制几何以实现采样高效的规划，从而学习对规划器友好的表示。CGS 仅利用来自像素-动作对的局部转移，将动作之间的成对余弦相似度与相应潜在差异之间的成对余弦相似度进行匹配。该损失可应用于各种世界模型架构，支持端到端学习的表示或预训练表示。在线性动力学条件下，我们的理论分析将该目标与时间维度的拉直以及整个规划范围内更均衡的终端代价曲率联系起来，为 MPPI 提供了有限预算保证，为 CEM 提供了局部收缩结果，并为梯度下降提供了收敛界。

    arXiv:2609.35603v2 Announce Type: replace  Abstract: Joint-embedding predictive architectures enable planning with latent world models, but accurate transition prediction alone does not ensure that the planning objective is easy to optimize. We introduce Control-Geometry Straightening (CGS), a single auxiliary loss that learns planner-friendly representations by directly straightening control geometry for sampling-efficient planning. CGS matches pairwise cosine similarities among actions to those among corresponding latent differences only using local transitions from pixel-action pairs. The loss can be applied across world-model architectures using end-to-end learned or pretrained representations. Under linear-dynamics, our theoretical analysis connects this objective to temporal straightening and more balanced terminal-cost curvature across the full planning horizon, yielding finite-budget guarantees for MPPI, local contraction results for CEM, and convergence bounds for gradient des
    
[^76]: 距离依赖矩条件下的普通非凸SGD：有限时域平稳性与Nagaev界

    Ordinary Nonconvex SGD under Distance-Dependent Moments: Finite-Horizon Stationarity and Nagaev Bounds

    [https://arxiv.org/abs/2609.30499](https://arxiv.org/abs/2609.30499)

    本文证明，当条件矩允许噪声方差随迭代点距离增长时，普通单样本SGD无需任何修改即可达到与Blum–Gladyshev下界匹配的极小极大随机复杂度，并借助希尔伯特空间Fuk–Nagaev不等式给出高概率Nagaev型界。

    

    统一的噪声矩界假设排除了那些变异性随迭代点位置增长的随机梯度。我们在距离依赖的条件矩假设下，研究针对光滑、下有界且可能非凸目标的普通单样本随机梯度下降。仅利用二阶矩条件，一个直接的“下降—位移”论证在使用依赖时域的步长时，给出了 $T^{-1/3}$ 的期望平均平方梯度平稳性。一个显式的预言机复杂度推论与已知的平滑Blum–Gladyshev（BG-0）下界相匹配，包括 $Lb_2\Delta^3\varepsilon^{-6}$ 和 $L\Delta\sigma^2\varepsilon^{-4}$ 两个随机项，其中 $\Delta$ 为初始目标间隙，$\sigma^2+b_2\|x-x_1\|^2$ 为方差的上界。因此，无需任何修改的SGD在这一二阶矩类别中即达到极小极大随机复杂度。对于 $p>2$，可预测局部化技术与希尔伯特空间上的Fuk–Nagaev不等式给出了一个高概率界，分离对数……（摘要在此处被截断）

    arXiv:2609.30499v1 Announce Type: new  Abstract: Uniform noise-moment bounds exclude stochastic gradients whose variability increases with the iterate. We study ordinary, single-sample stochastic gradient descent for smooth, lower-bounded, possibly nonconvex objectives under distance-dependent conditional moments. Under second moments alone, a direct descent--displacement argument yields $T^{-1/3}$ expected average squared-gradient stationarity with a horizon-dependent stepsize. An explicit oracle-complexity corollary matches the known smooth Blum--Gladyshev (BG-0) lower bound, including the $Lb_2\Delta^3\varepsilon^{-6}$ and $L\Delta\sigma^2\varepsilon^{-4}$ stochastic terms, where $\Delta$ is the initial objective gap and $\sigma^2+b_2\|x-x_1\|^2$ bounds the variance. Thus unchanged SGD attains the minimax stochastic complexity in this second-moment class. For $p>2$, predictable localization and a Hilbert-space Fuk--Nagaev inequality yield a high-probability bound separating logarith
    
[^77]: 关于网络上大规模多重检验的非渐近方法

    On Large-Scale Multiple Testing Over Networks: A Non-Asymptotic Approach

    [https://arxiv.org/abs/2609.14170](https://arxiv.org/abs/2609.14170)

    该论文发现分布式多重检验中有限样本FDR失控源于赢者诅咒偏差，并提出交叉拟合贪心聚合算法（CFGA），通过数据分割实现了网络上分布式多重检验在有限样本下的严格FDR控制。

    

    分布式多重检验要求网络中的N个站点在严格的通信预算下控制全局错误发现率（FDR）。Pournaderi和Xiang（2024）提出的贪心区间聚合算法渐近地解决了这一问题，但在有限样本下可能违反FDR≤α的约束。我们将这一违反追溯到被选密度统计量中的“赢者诅咒”偏差，在标准带宽ε≍m⁻¹/²下其精确阶为Θ(m⁻¹/⁴√(log m))，其中m是网络中p值的总数。交叉拟合贪心聚合算法（CFGA）通过在每个节点数据的一半上选择嵌套拒绝族，并在另一半上进行评分，从而消除这一偏差，在每节点零假设比例已知的情况下实现有限样本FDR≤α；一个膨胀变体在η=1/m的可忽略松弛下覆盖了插入法（plug-in）设定。BONuS-GA则通过计数型knockoff进行校准，掩盖一组合成的均匀零假设，使得每个……（原文摘要在此处截断）

    arXiv:2609.14170v1 Announce Type: cross  Abstract: Distributed multiple testing asks $N$ sites to control a global false discovery rate (FDR) under a tight communication budget. The greedy interval-aggregation algorithm of Pournaderi and Xiang (2024) solves this asymptotically but can violate $\mathrm{FDR}\le\alpha$ at finite samples. We trace the violation to a winner's-curse bias in the selected density statistics, of exact order $\Theta(m^{-1/4}\sqrt{\log m})$ at the standard bandwidth $\varepsilon\asymp m^{-1/2}$, with $m$ the total number of p-values in the network. Cross-Fit Greedy Aggregation (CFGA) eliminates the curse by selecting the nested rejection family on one half of each node's data and scoring it on the other, achieving finite-sample $\mathrm{FDR}\le\alpha$ when per-node null rates are known; an inflated variant covers the plug-in setting at a vanishing $\eta=1/m$ slack. BONuS-GA instead masks a bag of synthetic uniform nulls calibrated by counting knockoffs, so every 
    
[^78]: 凸域上参数估计的广义分数匹配

    Generalized Score Matching for Parameter Estimation on Convex Domains

    [https://arxiv.org/abs/2609.11521](https://arxiv.org/abs/2609.11521)

    本文从最小概率流学习出发，构造性地推导出凸域上的广义分数匹配目标函数，统一了经典分数匹配与非负数据的域适配变体，并证明该目标是二阶正当局部评分规则，保证最小化时能恢复真实密度。

    

    最大似然（ML）估计是学习概率模型的一种有原则且统计高效的方法。然而，对于非归一化模型，最大似然估计需要计算配分函数并对其进行求导，这在某些情况下可能并不可行。分数匹配提供了一种实际可行的替代方法，它通过以消除对归一化常数依赖的方式拟合分数，从而绕过了这一障碍。我们从最小概率流（MPF）学习出发，以构造性的方式推导出了 $\mathbb{R}^{d}$ 凸子集上的广义分数匹配目标函数，并展示了经典分数匹配以及适用于非负数据的域适配变体如何在该框架中自然产生。我们证明了所得到的目标函数是一个二阶的正当局部评分规则，这为最小化该目标函数时能够恢复真实密度提供了理论保证。

    arXiv:2609.11521v1 Announce Type: new  Abstract: Maximum likelihood (ML) estimation is a principled and statistically efficient approach for learning probabilistic models. However, for unnormalized models, ML estimation requires evaluating the partition function and differentiating through it, which may not always be tractable. Score matching provides a practically viable alternative that circumvents this obstacle by fitting the score in a way that eliminates dependence on the normalizing constant. We derive the generalized score matching objective on a convex subset of $\mathbb{R}^{d}$ constructively starting from Minimum Probability Flow (MPF) learning, and show how classical score matching as well as domain-adapted variants for non-negative data arise naturally within the proposed framework. We show that the resulting objective is a {\it proper local scoring rule} of second-order, which provides the theoretical guarantee that the true density is recovered when the objective is minim
    
[^79]: 基于最优传输的线性独立成分分析

    Linear Independent Component Analysis via Optimal Transport

    [https://arxiv.org/abs/2607.14081](https://arxiv.org/abs/2607.14081)

    本文提出以数据线性投影到标准高斯分布的平方 Wasserstein 距离作为 ICA 对比函数，并证明该距离在投影恰好恢复出独立成分时达到最大、且与任何真实混合信号之间都存在显式间隔，从而为线性独立成分分析建立了一种基于最优传输的新方法。

    

    线性独立成分分析（ICA）旨在从源信号的线性混合中恢复出联合独立的源信号。为实现这一目标，经典的ICA算法试图最大化非高斯性，通常以负熵来度量，而信息论将负熵与独立性联系在一起。由于精确的负熵优化是难以处理的，这些算法依赖于代理对比函数，例如四阶累积量和参数化对数似然。我们转而提出使用到标准高斯分布的平方 $L_2$-Wasserstein 距离作为 ICA 的对比函数。我们证明了当线性投影恢复出某个独立成分时，标准正态分布与数据线性投影之间的 Wasserstein 距离达到最大，并且在源信号满足一定正则性条件下，该最大值与每一个真实混合信号之间都由一个显式的间隔分隔开来。我们揭示了由此得到的估计量的优越性质：对于具有光滑密度的源信号……（原文摘要在此处被截断）

    arXiv:2607.14081v2 Announce Type: replace  Abstract: Linear Independent Component Analysis (ICA) recovers jointly independent source signals from their linear mixtures. To achieve this, classical ICA algorithms attempt to maximize non-Gaussianity, measured by negentropy, which is linked to independence by information theory. Because exact negentropy optimization is intractable, they rely on proxy contrast functions, such as fourth-order cumulants and parametric log-likelihoods. We propose instead to use the squared $L_2$-Wasserstein distance to a standard Gaussian as the ICA contrast. We show that the Wasserstein distance between a standard normal distribution and linear projections of the data is maximized when the projection recovers an independent component, and that under a regularity condition on the sources this maximum is separated from every genuine mixture by an explicit margin. We uncover the advantageous properties of the resulting estimator: for sources with a smooth densit
    
[^80]: 基于分数的生成模型中随机梯度下降的非渐近收敛性

    Non-asymptotic Convergence of Stochastic Gradient Descent in Score-based Generative Models

    [https://arxiv.org/abs/2607.04775](https://arxiv.org/abs/2607.04775)

    本文研究了基于分数的生成模型训练中随机梯度下降的非渐近收敛保证，针对一般分数参数化给出了显式依赖损失加权和时间采样分布的非凸优化界，并为过参数化两层 ReLU 网络建立了神经正切核分析。

    

    基于分数的生成模型在广泛的应用领域中取得了令人瞩目的数据生成性能。尽管其采样过程的统计特性已日益被充分理解，但其训练背后的优化动力学仍未得到充分探索。SGM 通常通过最小化加权去噪分数匹配目标进行训练，然而基于随机梯度的优化保证仍然有限。在本工作中，我们研究了随机梯度下降（SGD）在 SGM 中的应用，并在两个互补的场景中做出了贡献。对于一般的分数参数化，我们针对加权去噪分数匹配目标推导了 SGD 的非凸分析，明确揭示了所得优化界如何依赖于损失加权和时间采样分布。随后，我们考虑过参数化的两层 ReLU 网络，并开发了一种针对扩散模型的神经正切核分析……

    arXiv:2607.04775v2 Announce Type: replace-cross  Abstract: Score-based Generative Models (SGMs) have achieved impressive performance in data generation across a wide range of applications. While the statistical properties of their sampling procedures are increasingly well understood, the optimization dynamics underlying their training remain less explored. SGMs are typically trained by minimizing a weighted denoising score-matching objective, yet optimization guarantees with stochastic gradients remain limited. In this work, we study Stochastic Gradient Descent (SGD) for SGMs, contributing results in two complementary regimes. For general score parameterizations, we derive a non-convex analysis of SGD for the weighted denoising score-matching objective, making explicit how the resulting optimization bound depends on the loss weighting and time-sampling distribution. We then consider overparameterized two-layer ReLU networks and develop a Neural Tangent Kernel analysis tailored to diffu
    
[^81]: 面向图上能源时间序列不确定性量化的上下文残差校准方法

    In-Context Residual Calibration for Uncertainty Quantification of Energy Time Series over Graphs

    [https://arxiv.org/abs/2606.31804](https://arxiv.org/abs/2606.31804)

    该论文提出一种上下文残差校准方法，针对现有共形预测难以捕捉能源系统复杂时空结构的缺陷，为图上能源时间序列提供更可靠的不确定性量化，以支持风险感知的能源运营决策。

    

    精确的能源需求预测对于现代可持续能源系统的可靠运行和规划至关重要。时空图神经网络（STGNN）通过联合建模时间动态特性和互联能源节点之间的关系依赖性，近期在点预测方面取得了优异的表现。然而，在现实世界的能源系统中，仅有精确的点预测是不够的，运营者还需要可靠的不确定性估计，以支持风险感知决策、电网稳定性以及不确定性条件下的运营规划。共形预测在可交换性假设下为不确定性量化提供了一个有原则且与模型无关的框架，这使其对于安全关键的能源应用尤其具有吸引力。然而，现有的共形预测方法往往无法充分捕捉能源系统复杂的时空结构。为了解决这些问题……

    arXiv:2606.31804v2 Announce Type: replace  Abstract: Accurate energy demand forecasting is essential for the reliable operation and planning of modern sustainable energy systems. Spatial-temporal graph neural networks (STGNNs) have recently achieved strong performance in point forecasting by jointly modeling temporal dynamics and relational dependencies across interconnected energy nodes. However, in real-world energy systems, accurate point forecasts alone are insufficient, as operators also require reliable uncertainty estimates to support risk-aware decision-making, grid stability, and operational planning under uncertainty. Conformal prediction provides a principled and model-agnostic framework for uncertainty quantification under exchangeability assumptions, making it particularly attractive for safety-critical energy applications. However, existing conformal prediction approaches often fail to fully capture the complex spatial-temporal structure of energy systems. To address thes
    
[^82]: 面向基于种群优化的算子微积分：模块化收敛性与有限种群保证

    Operator Calculus for Population-Based Optimization: Modular Convergence and Finite-Population Guarantees

    [https://arxiv.org/abs/2606.14289](https://arxiv.org/abs/2606.14289)

    本文提出一种面向基于种群优化的算子微积分框架，使经过独立验证的更新规则效应可以模块化地组合，为收敛性分析提供可复用的构建模块，并给出有限评估预算下的收敛保证。

    

    基于种群的优化器将变异、选择和重组等更新规则组合在一起。当其中某条规则发生变化时，通常不清楚哪些收敛保证仍然成立，以及应如何评估新的组合。我们发展了一种算子微积分：算子即种群更新规则，而该微积分规定了如何将各自经过独立验证的效应进行组合。在明确的正则性和小步长条件下，由更新引起的主要变化可以相加，从而为收敛分析提供可复用的构建模块。该框架区分了找到并保留好的解、降低种群平均目标值以及使候选解集中于最优解附近这三类目标，并指出了获得有限评估预算保证所需的额外逼近条件。应用包括分布自适应、重组式演化和共识动力学，并验证了非凸情形。在……上进行的受控实验……

    arXiv:2606.14289v2 Announce Type: replace-cross  Abstract: Population-based optimizers combine update rules such as mutation, selection, and recombination. When one rule changes, it is often unclear which convergence guarantees survive or how the new combination should be assessed. We develop an operator calculus: an operator is a population-update rule, and the calculus specifies how separately checked effects can be combined. Under explicit regularity and small-step conditions, the leading changes caused by the updates add, yielding reusable building blocks for convergence analysis. The framework distinguishes finding and retaining a good solution, reducing the population's mean objective, and concentrating candidates near an optimizer, and identifies the extra approximation conditions needed for finite evaluation-budget guarantees. Applications include distribution adaptation, recombinative evolution, and consensus dynamics, with verified nonconvex cases. Controlled experiments on a
    
[^83]: 通用干预下自动、去偏且不变的反事实生成

    Automatic, Debiased, and Invariant Counterfactual Generation under General Interventions

    [https://arxiv.org/abs/2606.07399](https://arxiv.org/abs/2606.07399)

    ADIGen框架通过结合Riesz回归、因果不变性和正交统计学习，实现了通用干预下自动、去偏且不变的反事实生成，并提供了双重稳健的风险控制保证。

    

    反事实结果的生成模型在复杂干预下的决策支持方面具有巨大潜力，但现有方法受限于估计不稳定、跨环境泛化能力差以及因干扰模型误设而产生的偏差。我们提出了ADIGen框架，用于在通用干预（包括高维干预和结果）下实现自动、去偏且不变的反事实生成。ADIGen结合了Riesz回归以避免不稳定的密度比估计、因果不变性以改善分布偏移下的泛化能力，以及正交统计学习以获得针对干扰模型误设的双重稳健保证。我们提供了超额风险界，表明ADIGen在通用干预下控制反事实风险，具有乘积偏差干扰余项和跨环境的不变风险界。然后，我们将该框架扩展到多...

    arXiv:2606.07399v2 Announce Type: replace  Abstract: Generative models for counterfactual outcomes have great potential to support decision-making under complex interventions, but existing approaches are limited by unstable estimation, poor generalization across environments, and bias from nuisance model misspecification. We introduce ADIGen, a framework for automatic, debiased, and invariant counterfactual generation under general interventions, including high-dimensional interventions and outcomes. ADIGen combines Riesz regression to avoid unstable density-ratio estimation, causal invariance to improve generalization under distribution shift, and orthogonal statistical learning to obtain doubly robust guarantees against nuisance model misspecification. We provide excess-risk bounds showing that ADIGen controls counterfactual risk under general interventions, with a product-bias nuisance remainder and an invariant risk bound across environments. We then extend this framework to multip
    
[^84]: 大型线性自编码器中学习区间的棱镜层级结构

    A prism hierarchy of learning regimes in large linear autoencoders

    [https://arxiv.org/abs/2606.05335](https://arxiv.org/abs/2606.05335)

    本文提出用三棱柱的层级结构系统地刻画大型权重绑定线性自编码器的五个基本极端学习区间（大数据、小数据、平均场、窄潜层、自由），为此类非线性于权重的模型的学习动态提供了系统化的理论图景。

    

    机器学习模型的理论研究通常会考虑不同的极限区间，在这些区间中梯度下降的学习动态在理论上变得可处理。然而，对于特定类型的模型，能够系统地获得定性上不同的极端学习区间的整体图景是人们所期望的。本文为大型权重绑定线性自编码器提出了这样一幅图景，该模型由输入维度、潜在维度、初始化幅度和训练集大小来表征。该模型在权重上是非线性的，其梯度流不存在一般的理论解。我们证明，在形式损失展开层级的层面上，其极端区间自然地与一个三棱柱的各个面相关联。特别地，存在五个与棱柱的二维面相关联的基本极端区间：（1）大数据区间、（2）小数据区间、（3）平均场区间、（4）窄潜层区间，以及（5）自由区间。对于区间（1

    arXiv:2606.05335v2 Announce Type: replace  Abstract: Theoretical studies of machine learning models commonly consider different limiting regimes in which the learning dynamics of gradient descent becomes theoretically tractable. It is, however, desirable to have a systematically obtained picture of qualitatively different extreme learning regimes for a particular type of models. In this paper we propose such a picture for large weight-tied linear autoencoders characterized by input and latent dimensions, initialization magnitude, and training set size. This model is nonlinear in the weights and its gradient flow does not have a general theoretical solution. We show that at the level of the formal loss-expansion hierarchy, its extreme regimes are naturally associated with faces of a triangular prism. In particular, there are five basic extreme regimes associated with the 2-faces of the prism: (1) large-data, (2) small-data, (3) mean-field, (4) narrow-latent, and (5) free. For regimes (1
    
[^85]: 可解释性的元博弈与元归因

    The Metagame of Interpretability and Meta-Attributions

    [https://arxiv.org/abs/2605.06295](https://arxiv.org/abs/2605.06295)

    提出“元博弈”框架，将特征的归因值视为特征间的合作博弈并计算其Shapley值，从而得到方向性元归因，使任意基于梯度或注意力的归因方法都能泛化到二阶交互效应，并证明了元归因之和恰好等于其所解释的一阶归因。

    

    如何将任意的归因方法从第一性原理出发进行泛化，以捕获特征间的交互效应？我们用“元博弈”这一概念框架来回答这个问题，该框架用于量化模型解释中的二阶交互效应。我们将特征 i 的归因值 φ_i 视为其他特征之间的合作博弈，并计算其 Shapley 值，该值衡量特征 j 对特征 i 归因的影响程度，从而得到方向性元归因 φ_{j→i}。通过分解归因本身而非直接分解模型，元归因能够将任何基于梯度或注意力的方法扩展到交互效应，将基于移除的扰动方法与模型内部机制统一起来。在理论方面，我们证明了元归因之和等于其所解释的一阶归因，这是一种层级分解，而 Shapley 交互和积分 Hessian 方法实际上是隐式地执行了这种分解。在实证方面，我们展示了元（注：原文摘要在此处截断）

    arXiv:2605.06295v2 Announce Type: replace  Abstract: How can an arbitrary attribution method be generalized from first principles to capture interactions? We answer this with the metagame, a conceptual framework for quantifying second-order interaction effects of model explanations. We cast the attribution value $\phi_i$ of feature $i$ as a cooperative game among the other features and compute its Shapley value, which measures how much feature $j$ influences the attribution of $i$, yielding the directional meta-attribution $\varphi_{j \to i}$. By decomposing attribution itself rather than the model directly, meta-attributions extend any gradient- or attention-based method to interactions, uniting removal-based perturbations with model internals. Theoretically, we prove that meta-attributions sum to the first-order attribution they explain, a hierarchical decomposition that Shapley interactions and integrated Hessians turn out to perform implicitly. Empirically, we demonstrate that meta
    
[^86]: BONSAI：具有自然简洁性与可解释性的贝叶斯优化

    BONSAI: Bayesian Optimization with Natural Simplicity and Interpretability

    [https://arxiv.org/abs/2602.07144](https://arxiv.org/abs/2602.07144)

    提出了一种感知默认配置的贝叶斯优化策略BONSAI，它能在显式控制采集价值损失的前提下剪除对默认配置的低影响偏离，从而实现更简洁、可解释且易于审查的优化推荐。

    

    贝叶斯优化（BO）是一种流行的黑盒函数样本高效优化技术。在许多应用中，被调优的参数都伴随着经过精心设计的默认配置，实践者只在必要时才希望偏离默认值。然而，标准贝叶斯优化并不以最小化与默认配置的偏差为目标，在实践中常常将弱相关参数推向搜索空间的边界。这使得人们难以区分重要变更与虚假变更，并在优化目标遗漏相关运营考量时增加了审查推荐结果的负担。我们提出了BONSAI，这是一种感知默认配置的贝叶斯优化策略，它能够在显式控制采集函数价值损失的同时，剪除对默认配置影响较小的偏离。BONSAI兼容多种采集函数，包括期望改进和上置信界等。

    arXiv:2602.07144v3 Announce Type: replace  Abstract: Bayesian optimization (BO) is a popular technique for sample-efficient optimization of black-box functions. In many applications, the parameters being tuned come with a carefully engineered default configuration, and practitioners only want to deviate from this default when necessary. Standard BO, however, does not aim to minimize deviation from the default and, in practice, often pushes weakly relevant parameters to the boundary of the search space. This makes it difficult to distinguish between important and spurious changes and increases the burden of vetting recommendations when the optimization objective omits relevant operational considerations. We introduce BONSAI, a default-aware BO policy that prunes low-impact deviations from a default configuration while explicitly controlling the loss in acquisition value. BONSAI is compatible with a variety of acquisition functions, including expected improvement and upper confidence bou
    
[^87]: 使用物理结构变分自编码器（PS-VAE）实现定量分子磁共振成像中的多参数不确定性映射

    Multiparameter Uncertainty Mapping in Quantitative Molecular MRI using a Physics-Structured Variational Autoencoder (PS-VAE)

    [https://arxiv.org/abs/2602.03317](https://arxiv.org/abs/2602.03317)

    提出一种物理结构变分自编码器（PS-VAE），通过融合可微分自旋物理模拟器与自监督学习，实现定量分子MRI中体素级多参数后验分布的快速提取与不确定性量化。

    

    定量成像方法，如磁共振指纹成像（MRF），旨在通过从信号演化中估计生物物理组织参数来提取可解释的病理生物标志物。然而，此类逆问题中常用的模式匹配算法或神经网络往往缺乏有原则的不确定性量化，这限制了临床接受所必需的可信度与透明度。在此，我们提出了一种物理结构变分自编码器（PS-VAE），用于快速提取体素级的多参数后验分布。我们的方法将可微分自旋物理模拟器与自监督学习相结合，并提供了完整的协方差矩阵，以捕获潜在生物物理空间中的参数间相关性。该方法在多质子池化学交换饱和转移（CEST）与半固体磁化转移（MT）的分子MRF研究中得到了验证。

    arXiv:2602.03317v2 Announce Type: replace-cross  Abstract: Quantitative imaging methods, such as magnetic resonance fingerprinting (MRF), aim to extract interpretable pathology biomarkers by estimating biophysical tissue parameters from signal evolutions. However, the pattern-matching algorithms or neural networks used in such inverse problems often lack principled uncertainty quantification, which limits the trustworthiness and transparency, required for clinical acceptance. Here, we describe a physics-structured variational autoencoder (PS-VAE) designed for rapid extraction of voxelwise multi-parameter posterior distributions. Our approach integrates a differentiable spin physics simulator with self-supervised learning, and provides a full covariance that captures the inter-parameter correlations of the latent biophysical space. The method was validated in a multi-proton pool chemical exchange saturation transfer (CEST) and semisolid magnetization transfer (MT) molecular MRF study, a
    
[^88]: 整流流的最优阶样本复杂度

    Order-Optimal Sample Complexity of Rectified Flows

    [https://arxiv.org/abs/2601.20250](https://arxiv.org/abs/2601.20250)

    本文证明了整流流模型在标准神经网络假设下可达到 $\tilde{O}(\varepsilon^{-2})$ 的最优阶样本复杂度，改进了流匹配模型已有的 $O(\varepsilon^{-4})$ 界并匹配均值估计的最优速率。

    

    近年来，基于流的生成模型相比扩散模型展现出了更优的效率。本文研究整流流模型，该模型约束从基础分布到数据分布的传输轨迹为线性。这种结构性限制极大地加速了采样过程，通常仅需单步欧拉步即可实现高质量生成。在用于参数化速度场和数据分布的神经网络类的标准假设下，我们证明整流流可以达到 $\tilde{O}(\varepsilon^{-2})$ 的样本复杂度。这改进了流匹配模型已知的最佳 $O(\varepsilon^{-4})$ 界，并达到了均值估计的最优速率。我们的分析利用了整流流的特殊结构：由于模型是沿线性路径以平方损失进行训练的，相关的假设类具有严格可控的局部化 Rademacher 复杂度。

    arXiv:2601.20250v2 Announce Type: replace  Abstract: Recently, flow-based generative models have shown superior efficiency compared to diffusion models. In this paper, we study rectified flow models, which constrain transport trajectories to be linear from the base distribution to the data distribution. This structural restriction greatly accelerates sampling, often enabling high-quality generation with a single Euler step. Under standard assumptions on the neural network classes used to parameterize the velocity field and data distribution, we prove that rectified flows achieve sample complexity $\tilde{O}(\varepsilon^{-2})$. This improves on the best known $O(\varepsilon^{-4})$ bounds for flow matching model and matches the optimal rate for mean estimation. Our analysis exploits the particular structure of rectified flows: because the model is trained with a squared loss along linear paths, the associated hypothesis class admits a sharply controlled localized Rademacher complexity. T
    
[^89]: 动量梯度下降的修正损失：精细分析

    Modified Loss of Momentum Gradient Descent: Fine-Grained Analysis

    [https://arxiv.org/abs/2509.08483](https://arxiv.org/abs/2509.08483)

    该论文证明当步长足够小时，重球动量梯度下降在指数吸引的不变流形上精确等价于带修正损失的普通梯度下降，能以任意有限阶精度刻画该修正损失，并在其无记忆近似的组合结构中发现了介于欧拉多项式与Narayana多项式之间的一类新的β多项式族。

    

    我们分析了带有Polyak (1964) 重球动量（HB）的梯度下降算法，其固定动量超参数 β ∈ (0, 1) 提供了记忆的指数衰减。基于 Kovachki 和 Stuart (2021) 的工作，我们证明当步长 h 足够小时，该算法在指数吸引的不变流形上恰好等价于带有修正损失的普通梯度下降。尽管该修正损失不存在闭式表达式，我们对其进行了描述，对于任意有限阶 R 误差为 O(h^R)，并证明了全局（有限“时间”范围）轨迹逼近界 O(h^R)。随后，我们对 HB 无记忆近似背后的组合数学进行了精细分析，特别是发现了隐藏其中的一类丰富的关于 β 的多项式族，它们包含欧拉多项式和 Narayana 多项式，且在系数意义上介于二者之间。我们证明这些多项式……（摘要在此处被截断）

    arXiv:2509.08483v2 Announce Type: replace  Abstract: We analyze gradient descent with Polyak (1964) heavy-ball momentum (HB) whose fixed momentum hyperparameter $\beta \in (0, 1)$ provides exponential decay of memory. Building on Kovachki and Stuart (2021), we prove that on an exponentially attractive invariant manifold the algorithm is exactly plain gradient descent with a modified loss, provided that the step size $h$ is small enough. Although the modified loss does not admit a closed-form expression, we describe it up to $O(h^{\mathcal{R}})$-errors for arbitrary finite order $\mathcal{R}$, and prove global (finite "time" horizon) trajectory approximation bounds $O(h^{\mathcal{R}})$. We then conduct a fine-grained analysis of the combinatorics underlying the memoryless approximations of HB, in particular, finding a rich family of polynomials in $\beta$ hidden inside which include and lie coefficient-wise in between Eulerian and Narayana polynomials. We prove that these polynomials ar
    
[^90]: 因果后验估计

    Causal Posterior Estimation

    [https://arxiv.org/abs/2505.21468](https://arxiv.org/abs/2505.21468)

    提出因果后验估计（CPE）方法，将模型图结构中的条件依赖关系直接硬编码进基于流匹配的神经网络架构，在似然函数难以计算的模拟器模型中实现高精度的贝叶斯后验推断。

    

    我们提出了因果后验估计，这是一种用于模拟器模型贝叶斯推断的新方法，适用于似然函数难以求解或计算成本高昂、但根据给定参数值生成输出较为简单的场景。CPE利用流匹配来近似后验分布，同时将模型图结构所诱导的条件依赖关系直接融入神经网络架构中。通过大量实验，我们证明了将这些条件依赖关系硬编码到网络中（而非要求从数据中学习它们），使CPE能够实现高度精确的后验推断，其性能达到甚至超越当前最先进的基线方法。

    arXiv:2505.21468v2 Announce Type: replace  Abstract: We present Causal Posterior Estimation (CPE), a novel method for Bayesian inference in simulator models, where evaluating the likelihood function is intractable or computationally expensive, but generating outputs given parameter values is straightforward. CPE approximates the posterior distribution using flow matching while directly incorporating the conditional dependence structure induced by the model's graphical representation into the neural network architecture. Across extensive experiments, we demonstrate that hard-coding these conditional dependencies into the network, rather than requiring them to be learned from data, enables CPE to achieve highly accurate posterior inference that matches or outperforms state-of-the-art baselines.
    
[^91]: 方差感知UCB策略下的分配稳定性与Wald推断

    Allocation Stability and Wald Inference under Variance-Aware UCB

    [https://arxiv.org/abs/2412.08843](https://arxiv.org/abs/2412.08843)

    本文为双臂方差感知UCB策略给出了最优臂分配稳定性的尖锐判据，并证明即使最优臂计数不稳定，只要拉取次数与奖励方差的乘积依概率发散，臂均值线性组合的Wald统计量仍渐近服从标准正态分布，从而表明分配稳定性并非高斯推断的必要条件。

    

    分配稳定性常被用来论证基于老虎机数据进行高斯推断的合理性，但它何时才是必要的？本文针对双臂、固定时域、奖励分布有界且可能随时域变化的方差感知UCB策略研究了这一问题。我们找到了一个由奖励差距和方差刻画的尖锐判据，该判据决定了最优臂的拉取次数能否被具有消失相对误差的确定性量近似，而次优臂的计数总是稳定的。尽管最优臂计数可能不稳定，我们证明：只要每个臂的拉取次数与奖励方差的乘积依概率发散，对于任意固定的非零系数向量，臂均值线性组合的通常Wald统计量都具有标准正态极限。然而在同一条件下，这种高斯近似能否在所有确定性非零系数向量上一致成立，取决于……（摘要在此处截断）

    arXiv:2412.08843v3 Announce Type: replace-cross  Abstract: Allocation stability is often used to justify Gaussian inference from bandit data, but when is it necessary? In this paper, we address this question for a two-armed, fixed-horizon variance-aware UCB policy with bounded reward distributions that may vary with the horizon. We find a sharp criterion in terms of the reward gap and variances that determines whether the optimal-arm count admits a deterministic approximation with vanishing relative error, while the suboptimal-arm count is always stable. Despite the possible instability of the optimal-arm count, we show that the ordinary Wald statistic for a linear combination of the arm means has a standard normal limit for every fixed nonzero coefficient vector, provided the product of the pull count and reward variance diverges in probability for each arm. Under the same condition, however, this Gaussian approximation holds uniformly over deterministic nonzero coefficient vectors if
    
[^92]: 结合可加性和主动子空间用于高维高斯过程建模

    Combining additivity and active subspaces for high-dimensional Gaussian process modeling

    [https://arxiv.org/abs/2402.03809](https://arxiv.org/abs/2402.03809)

    本论文的贡献是将可加性和主动子空间与多重真实度策略结合，解决了高维高斯过程建模中的维度灾难问题，并通过实验证明了这些优势。

    

    高斯过程是一种被广泛接受的回归和分类技术，因其良好的预测准确性、分析可追溯性和内置的不确定性量化能力而倍受欢迎。然而，当变量数量增加时，它们受到维度灾难的困扰。这个挑战通常通过在问题中假设额外结构来解决，首选选项是可加性或低内在维度。我们在高维高斯过程建模中的贡献是将它们与多重真实度策略相结合，通过对合成函数和数据集进行实验证明了这些优势。

    Gaussian processes are a widely embraced technique for regression and classification due to their good prediction accuracy, analytical tractability and built-in capabilities for uncertainty quantification. However, they suffer from the curse of dimensionality whenever the number of variables increases. This challenge is generally addressed by assuming additional structure in theproblem, the preferred options being either additivity or low intrinsic dimensionality. Our contribution for high-dimensional Gaussian process modeling is to combine them with a multi-fidelity strategy, showcasing the advantages through experiments on synthetic functions and datasets.
    

