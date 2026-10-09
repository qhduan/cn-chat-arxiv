# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Density Ratio Estimation with Stein Displacement Fields](https://arxiv.org/abs/2610.12437) | 该论文提出通过Stein位移场参数化密度比（将对数比建模为负的基础分布Stein算子作用于位移场），用单个凸优化问题统一了分布偏移的统计与动力学描述，并据此发展出无需重训即可修正预训练采样器的push-forward算法和拉近数据分布的pull-back算法。 |
| [^2] | [Bilevel optimization for data-driven learning of Koopman embeddings using kernel-based autoencoders](https://arxiv.org/abs/2610.12370) | 本文提出了一种结合配置方法与双层优化的新方法EDMD-kDL，利用基于核的自编码器直接从数据中学习有限维Koopman嵌入，克服了传统EDMD需先验指定字典的局限，同时相比神经网络方法具有更好的可解释性和理论可分析性。 |
| [^3] | [Closing the Horizon Gap in Policy Optimization for Adversarial MDPs](https://arxiv.org/abs/2610.12362) | 该论文提出使用正则化Q函数在所有状态-动作对上联合控制局部更新稳定性，从而将对抗性MDP策略优化的遗憾界对视野H的依赖性改进到与基于占用测度的算法相当的水平。 |
| [^4] | [Prediction-Powered Data Fusion for Treatment Effect Estimation](https://arxiv.org/abs/2610.12332) | 提出了一种无需对观察性研究做特殊假设的数据融合框架，通过保持RCT估计的无偏性并从大型观察性研究中借力，显著提升平均处理效应（ATE）和条件平均处理效应（CATE）的估计精度。 |
| [^5] | [Composite Online-to-Nonconvex Conversion with Optimal Oracle Complexity](https://arxiv.org/abs/2610.12328) | 该论文通过为在线学习者设计新的损失函数，将在线到非凸转换框架扩展至复合优化场景，首次在一阶随机预言机访问下建立了复合非凸优化的最优复杂度保证。 |
| [^6] | [Testing Algebraic Complete Intersections](https://arxiv.org/abs/2610.12288) | 本文提出了一种显式有效的学习检验程序，利用正则多项式系统的定量几何估计，判断高维数据分布是否集中于规定维度、有界次数和有界条件数的实代数完全交集附近，要么返回几何复杂度受控的候选回归流形，要么在受控放宽的阈值下证明此类流形不存在。 |
| [^7] | [ISBO: Scalable Spatio-Temporal Bayesian Optimization with Log Gaussian Cox Process Models via the INLA-SPDE Approach](https://arxiv.org/abs/2610.12213) | 该论文提出了首个面向时空数据的可扩展贝叶斯优化框架ISBO，通过对数高斯Cox过程建模与INLA-SPDE推断方法，能够以最少的评估次数稳定定位高强度区域及潜在强度峰值。 |
| [^8] | [Verification with Transfer: Exact Information Frontiers and Their Price in Calls](https://arxiv.org/abs/2610.12211) | 本文从信息论角度为“借助相关源任务（迁移）来降低验证成本”这一策略精确定价：所需最小因果信息由列表率失真函数刻画，并把所有源调用放在首次验证之前不会破坏调用的硬上限，但交错安排可以无界地节省期望调用次数。 |
| [^9] | [Quickest Change Detection with Diffusion-Integrated Scores](https://arxiv.org/abs/2610.12200) | 提出了一种无需训练的扩散积分分数CUSUM（DI-SCUSUM）最快变化检测器，通过向样本添加高斯噪声并精确计算Hyvärinen分数来近似对数似然比，实现了指数级误报控制和有保证的一阶检测延迟界。 |
| [^10] | [Credal Machine Learning for Risk-Averse Decision Making](https://arxiv.org/abs/2610.12115) | 该论文提出用信度集（概率分布的集合）来表示预测中的认知不确定性，并结合一种新颖的决策规则，实现基于CVaR的可靠风险规避决策。 |
| [^11] | [Differentiable Systematic Resampling for Variational Sequential Monte Carlo](https://arxiv.org/abs/2610.12094) | 提出可微系统重采样（DSR），一种温度控制的系统重采样松弛方法，在保持其结构特性并具有可证明的指数收敛偏差的同时，实现了完全梯度流动，且计算开销远低于基于最优传输的方法。 |
| [^12] | [Exploiting Gradients in Bayesian Inference of Expensive Simulators](https://arxiv.org/abs/2610.12076) | 该论文提出在昂贵模拟器的贝叶斯推断中，利用模拟器输出对输入参数的梯度信息作为额外信号来指导基于贝叶斯优化的主动学习过程，从而提高有限模拟预算下的推断效率。 |
| [^13] | [Diffusion Removes Langevin's Conditioning Dependence: A Sharp Gaussian Analysis](https://arxiv.org/abs/2610.12052) | 本文在高斯情形下证明扩散模型的采样误差为 $O(\sqrt{d\lambda_{\max}}\log N/N)$，消除了经典朗之万类采样器中依赖条件数的 $\sqrt{\kappa}$ 因子，并通过精确的谱界与匹配的一阶渐近分析，严格解释了扩散模型优于传统基于分数采样器的理论原因。 |
| [^14] | [Efficient and Generalizable Archetypal Analysis for Discrete Data](https://arxiv.org/abs/2610.12035) | 提出了一种面向离散数据的基于似然的高效原型分析框架，支持伯努利、泊松和多项式观测模型，并引入交叉验证的预测似然准则来有原则地选择原型数量。 |
| [^15] | [Efficient quadratic entropy with distance sketches](https://arxiv.org/abs/2610.11976) | 本文提出了一种基于随机特征嵌入、投影和控制变量技术的可扩展二次熵近似方法，并在文献计量学应用中仅凭引用和文本特征揭示了论文、领域和机构的跨学科影响力。 |
| [^16] | [Score-Based Learning of Cluster DAGs from Interventions](https://arxiv.org/abs/2610.11947) | 提出首个基于评分的方法COARSE，利用干预数据识别聚类间的因果顺序并将边学习简化为局部搜索，从而在线性高斯假设下实现聚类DAG的学习。 |
| [^17] | [RobustLDS: Learning linear dynamical systems under adversarial corruptions](https://arxiv.org/abs/2610.11906) | 该论文提出了基于最小截断二乘法松弛与离群值组稀疏性的估计器，用于在对抗性污染下从单条轨迹学习线性动力系统，并通过非渐近误差界证明了其对离群值的鲁棒性。 |
| [^18] | [Learning structured linear dynamical systems from missing observations](https://arxiv.org/abs/2610.11869) | 本文提出一种基于偏差校正目标函数的估计器，用于在观测严重缺失的情况下学习凸集约束下的结构化线性动力系统，给出了依赖集合局部复杂度、轨迹长度和采样概率的非渐近误差界，并证明即使轨迹远短于无约束情形且采样概率趋于零时仍能有意义地恢复转移矩阵。 |
| [^19] | [Conditional Kernel Stein Discrepancy](https://arxiv.org/abs/2610.11863) | 提出了一个通过协变量空间上的算子值核将核斯坦因差异推广到条件设定的框架，用于在仅知道非归一化条件目标模型和联合分布样本的情况下量化条件拟合优度。 |
| [^20] | [Softmax Attention on Gaussian Mixtures: Linear When It Can, Selective When It Must](https://arxiv.org/abs/2610.11798) | 该论文通过研究softmax注意力在高斯混合分布上的无穷提示极限，证明softmax注意力既能像线性注意力一样有效解决线性任务，又能借助查询依赖的选择能力，以梯度方法学习到监督分类、去噪等具有潜在结构、多峰性和非线性依赖的统计任务的最优解。 |
| [^21] | [Optimal random quantisers for spherically symmetric distributions](https://arxiv.org/abs/2610.11772) | 该论文证明对于球对称目标分布，随机量化器的优化问题具有凸性，并据此发现均匀分布在适当半径球面上的随机量化器即使在中等样本量下也表现优异、被数值验证为全局最优，从而克服了Zador渐近理论中高维所需的天文数字级样本量问题。 |
| [^22] | [$\sigma$Transfer: Uncertainty Transfer from Small to Large Networks under $\mu\mathrm{P}$](https://arxiv.org/abs/2610.11668) | 该论文提出σTransfer方法，在μP参数化下通过重新缩放先验协方差，使拉普拉斯近似所需的先验精度可以从小模型零样本迁移到大模型，从而免去在大模型上的昂贵精度搜索，实测加速可达约5000倍。 |
| [^23] | [Minimax Gaussian Mechanisms for Continual Machine Unlearning](https://arxiv.org/abs/2610.11628) | 本文提出基于牛顿更新与高斯差分隐私的极小极大高斯机制，通过推导残差误差上界来校准噪声方差分配，使得顺序删除记录后发布的一系列模型在统计上与精确重训练难以区分，并最小化最坏情况下的噪声方差。 |
| [^24] | [Embedding-Bias in Conditional Independence Testing](https://arxiv.org/abs/2610.11584) | 该研究揭示了在条件独立性检验中用嵌入替代原始变量所引发的偏差问题，并证明对于残差相关性检验，只要嵌入遗漏的条件均值部分互不相关即可保证检验有效，否则偏差可精确量化为遗漏部分绝对相关性乘以两个偏R²值的几何平均数。 |
| [^25] | [Automated Detection of Match Phases in Football from Spatio-Temporal Tracking Data Using Graph Neural Networks](https://arxiv.org/abs/2610.11571) | 该论文提出了一种结合图神经网络与LSTM的框架，利用足球比赛的时空追踪数据逐秒自动分类七个比赛阶段，其性能超越XGBoost等所有基线模型，宏F1分数高出4.6%。 |
| [^26] | [LAIR-Net: Leaky Alignment-Impulse Residual Networks for Tabular Regression](https://arxiv.org/abs/2610.11538) | LAIR-Net通过泄漏残差过渡将浅层学习的锚点注入隐状态演化中，实现了对隐状态的目标感知控制，在23个表格回归基准数据集上超越了八个随机化网络和十二个传统模型，并在非线性目标结构可学习时收益最大。 |
| [^27] | [PSI-SINDy: Post-Selection Inference for Sparse Identification of Nonlinear Dynamics](https://arxiv.org/abs/2610.11486) | 本文提出PSI-SINDy，通过选择后推断方法为SINDy识别出的动力学项提供有效的假设检验和置信区间，从而消除选择偏差并量化所选动力学项的统计可靠性。 |
| [^28] | [Rare Gate Disagreements Can Limit Plasticity: When Gradient Flow Mispredicts Finite-Batch SGD](https://arxiv.org/abs/2610.11475) | 该论文证明总体梯度流可能在定性上错误预测有限批量SGD：在双神经元ReLU回归中，当预训练时间超过 $\log(b/\eta)$ 后，源于罕见门控分歧的机制使在线SGD在指数级长的时间范围内以高概率丧失可塑性、无法适应目标任务，而梯度流却只需线性时间即可恢复。 |
| [^29] | [Feature Space Adaptation for Effortless Gaussian Process Flows](https://arxiv.org/abs/2610.11459) | 该论文通过引入核近似和基于扩散工作量的边际似然估计方法，首次实现了 FlowGP 框架内的超参数自动优化，使其能够高效扩展到高分辨率域并处理非高斯条件推断任务。 |
| [^30] | [Beyond Distributional Fidelity: Causal-Penalized Diffusion for Synthetic Tabular Data](https://arxiv.org/abs/2610.11407) | 该论文首次将因果差异惩罚直接引入生成式表格扩散模型，提出因果惩罚化的 TabDDPM 训练框架，理论上证明高统计保真度不等于高因果保真度，并给出了因果正则化提升期望因果保真度的条件与实验验证。 |
| [^31] | [Sequential Conditional Independence Testing with Machine Learning Models](https://arxiv.org/abs/2610.11388) | 该论文通过将检验误差分解为零假设扩大误差、近似误差和估计误差，解释了直接检验可交换性的e-变量在实践中优于model-X框架下GRO e-变量的现象，并提出探索介于两者之间的中间零假设，从而将机器学习模型融入序贯条件独立性检验。 |
| [^32] | [From Geometry to Generalization: Why Row Normalization Can Beat Adam and Muon](https://arxiv.org/abs/2610.11309) | 该论文证明了在高维多分类任务中，行归一化凭借其类级欧几里得几何能够渐近保持总体决策边界方向，从而在总体精度上严格超越采用坐标级几何的Adam和采用谱几何的Muon等优化器。 |
| [^33] | [Regularized Small Area Estimation with Graph Laplacian Benchmarking priors](https://arxiv.org/abs/2610.11266) | 提出了一类新的贝叶斯基准化先验族（含基于图拉普拉斯的正则化版本），可在小区域估计中同时实现跨区域信息借用与对可靠总体数据的基准化校准，并针对退化先验设计了基于降参数化的定制MCMC算法。 |
| [^34] | [When Lower Reconstruction Loss Hurts: Distributionally Robust Refinement for Low-Bit LLM Quantization](https://arxiv.org/abs/2610.11226) | 本文发现更低的重构损失不一定带来更好的模型性能甚至可能有害，并提出分布鲁棒量化（DRQ）方法，通过在受约束的激活分布集合上最小化最坏情况重构损失来精炼量化权重编码，从而提升低比特大语言模型量化的效果。 |
| [^35] | [Tight Bounds for Equivalence Testing with Non-Adaptive Conditional Samples](https://arxiv.org/abs/2610.11145) | 本文针对非自适应条件样本模型下的等价性检验问题，给出了匹配的算法与下界，证明其查询复杂度为 $\tilde \Theta(\log n/\varepsilon^2)$，并由此揭示均匀性、恒等性和等价性检验的复杂度均为 $\tilde \Theta(\log n)$。 |
| [^36] | [Accelerating Non-Smooth and Heavy-Tailed Sampling](https://arxiv.org/abs/2610.11139) | 本文提出非可逆锚定朗之万动力学（NALD）与非可逆反射锚定朗之万动力学（NRALD），通过引入循环漂移项，在无需目标密度导数的情况下加速对欧氏空间及受限域上非光滑、重尾目标分布的采样。 |
| [^37] | [A General $\widetilde{\Omega}(\sqrt{T \gamma_T})$ Lower Bound for Kernel Bandits](https://arxiv.org/abs/2610.11082) | 本文针对紧域上非常数连续核函数的核赌博机问题，建立了通用的 $\Omega(\sqrt{T\gamma_T/\log T})$ 极小极大遗憾下界，并证明该下界中的对数因子在一般情况下不可避免，从而在非常一般的意义上确立了现有 $\sqrt{T\gamma_T}$ 上界的接近最优性。 |
| [^38] | [Decision-Sufficient Posterior Approximation](https://arxiv.org/abs/2610.11038) | 该论文提出“决策充分”的后验近似框架，通过在贝叶斯动作纤维上收缩KL散度精确刻画后验近似何时会改变下游决策，并利用目标遗憾Hessian与基线信息度量的广义特征值问题，按单位信息成本的遗憾后果对决策方向排序，从而找出跨越决策边界所需的最小信息后验形变。 |
| [^39] | [SPD-MetaFormer is what you need for small-data brain decoding](https://arxiv.org/abs/2610.10952) | 该研究发现基于SPD流形的注意力模型学到的注意力权重接近均匀、可被简单均匀权重替代而几乎不损失预测性能，据此表明架构结构比学习加权更关键，提出SPD-MetaFormer足以胜任小数据脑解码任务。 |
| [^40] | [Transformed Samplers with Variance Reduction](https://arxiv.org/abs/2610.10870) | 该论文提出通过学习双射（如归一化流）将MCMC采样器变换到潜空间，把简单参考分布上的泊松方程精确解推广到一般目标分布，从而获得显式控制变量以实现方差缩减。 |
| [^41] | [Conformal Prediction under Partial Verification](https://arxiv.org/abs/2610.10829) | 该论文提出了一种部分验证方法，通过刻画校准证书并在校准样本间协调验证，在产生与完整验证完全相同的预测集的同时，将验证成本降低15-82%。 |
| [^42] | [Calibrating Ambiguity Set via Diagnostic Transport for Distributionally Robust Optimization](https://arxiv.org/abs/2610.10793) | 本文提出诊断传输DRO（DT-DRO），利用留出校准数据和条件概率积分变换来诊断预测误差，自适应地调整模糊集的中心与几何结构，从而在保证决策风险可控的同时避免DRO决策过度保守。 |
| [^43] | [What can linear attention learn from nonlinear teachers in-context?](https://arxiv.org/abs/2610.10761) | 本文的核心创新是建立了“非线性-噪声等价性”理论：线性注意力在上下文学习中只提取目标函数的线性Hermite分量，剩余非线性结构等价于有效噪声，从而使线性理论的结论可以迁移到非线性任务。 |
| [^44] | [Explaining the Saliency Map Sparsity of Adversarially-Trained Neural Networks](https://arxiv.org/abs/2610.10666) | 本文首次为对抗训练神经网络梯度显著性图的稀疏性现象提供了理论解释，证明随着数据点和神经元数量增长，网络的最小化解收敛到具有最小梯度和Barron范数的贝叶斯分类器，从而自然产生了稀疏性。 |
| [^45] | [From Log-Odds to Shapley Values: An Explanatory Geometry for the Weighted Naive Bayes Classifier](https://arxiv.org/abs/2610.10642) | 该论文证明加权朴素贝叶斯分类器中基于对数几率的监督距离与解析Shapley值向量之间的ℓ1距离完全一致，从而为监督距离、局部解释与预测行为之间建立了形式化的联系。 |
| [^46] | [How Many Directions Must a Truncated Diffusion Sampler Retain? Matching Bounds Under Power-Law Spectra](https://arxiv.org/abs/2610.10640) | 针对幂律协方差谱数据，论文证明了截断扩散采样器所需保留方向数的匹配上下界，并揭示仅保留信号超过噪声水平方向的策略仍会产生不可忽略的总体截断误差。 |
| [^47] | [D-SLR: The Disjoint Row-Sparse plus Low-Rank Decomposition](https://arxiv.org/abs/2610.10636) | 本文提出D-SLR分解，一种截断SVD的闭式直接替代方法，它将矩阵行不相交地划分为逐字存储或低秩近似两类，在平方误差下以更少参数即可达到联合最优，且在相同代价下不会差于截断SVD。 |
| [^48] | [Exact SO(3)-Equivariant Isotropic Kernels for Rotation-Robust Neural Dynamics](https://arxiv.org/abs/2610.10626) | 本文提出不变量条件化各向同性核神经算子（IKNO），通过仅使用旋转不变标量与随数据共同旋转的向量方向来构建局部相互作用，使三维Navier–Stokes等向量值偏微分方程的神经代理模型实现数值精度上的精确SO(3)等变性，彻底消除无约束图模型无法根除的坐标依赖问题。 |
| [^49] | [JevForest: Path Voting for Budgeted Feature Acquisition](https://arxiv.org/abs/2610.10615) | 提出JevForest特征获取策略，通过聚合自助采样树的路径提议、以全局信息增益加权并用共享掩码分类器预测，在有限观测预算下自适应选择最值得查询的特征，但其实验效果在不同数据集上并不一致。 |
| [^50] | [The optimal information complexity of VC learning](https://arxiv.org/abs/2610.10600) | 本文通过构造一个eCMI为O(d)阶的随机化“5个基学习器多数投票”学习算法，首次经由CMI信息论分析框架恢复了VC类学习的最优PAC泛化保证。 |
| [^51] | [An Accuracy--Information Tradeoff for Loss-Difference Conditional Mutual Information](https://arxiv.org/abs/2610.09206) | 论文证明了精度与信息之间的权衡：在逻辑损失等光滑凸损失及幂次正则化条件下，任何以最优样本量达到低超额风险的正规学习器，其最坏情况损失差条件互信息必然达到 n 比特量级。 |
| [^52] | [Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States](https://arxiv.org/abs/2610.08818) | 该论文提出GenST框架，将未观测节点状态预测（FUNS）重新定义为时空图上的条件生成任务，创新性地利用微调后的大语言模型从节点描述中提取语义特征作为语义桥梁，以弥补缺失的时空信号。 |
| [^53] | [One-Shot Private Confidence Regions via Resampling](https://arxiv.org/abs/2610.08460) | 提出一个一次性构建差分隐私置信区域的简单框架，仅对最终重采样分位数加噪，使隐私代价在有放回（m-out-of-n）采样下仅为对数级、在无放回（子）采样下与重采样次数 B 无关，避免了以往方法中的 √B 因子，并给出了非渐近的高斯差分隐私与效用保证。 |
| [^54] | [SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics](https://arxiv.org/abs/2610.06513) | 提出SOL——一种基于固定Transformer隐藏状态经验测度的双切片Wasserstein距离的文本分布距离度量，当Transformer为单射时可证明其为真正的度量，为非自回归语言模型的分布拟合评估提供了稳定的样本级评估方案。 |
| [^55] | [Measuring Learned Monotone Temporal Aggregation at Matched Admissibility](https://arxiv.org/abs/2610.05196) | 该论文在对比双方单调可容许性完全匹配的前提下，用构造上单调的循环网络（EWMA与高水位标记的可学习变换）度量学习型时间聚合的价值，并通过函数回归揭示出一条“涵盖边界”——学习到的单调通道可复现几何加权可分手工统计量族。 |
| [^56] | [On the Tightness and Computational Tractability of Higher-Dimensional Confidence Sequences](https://arxiv.org/abs/2610.03727) | 该论文将一维基于下注的置信序列以三种方式提升到高维（加权Bonferroni区域、最大财富形式和组合区域），并针对最紧致但无封闭表达式解的组合区域提出了包围盒、ℓp-椭球及其交集等保持统计有效性的可计算外近似，从而在高维多元置信序列中兼顾了统计紧致性与计算可行性。 |
| [^57] | [Explicit Bounds on the Entropy of Piecewise H\"{o}lder Graphon Models](https://arxiv.org/abs/2608.26501) | 本文为分段Hölder图函数生成的随机图熵提供了显式收敛速率和定量界，取代了以往的渐近结果。 |
| [^58] | [K\"ahler landscapes for complex neural network descents and guarantees including a search and destroy of the Calabi-Yau manifold](https://arxiv.org/abs/2608.19584) | 本文提出了一种在复参数神经网络中使用Kähler信息度量和自然梯度下降的新方法，并针对Calabi-Yau流形上的不良曲率条件提供了理论保证，通过几何定义的全局势实现了搜索与消灭策略。 |
| [^59] | [Twoblock clustering trees with coskewness-based dimension reduction: recovering piecewise multivariate linear regimes](https://arxiv.org/abs/2607.20760) | 本文提出了一种基于协偏度最大化降维的双块聚类树（tbtree），这是一种高度可解释的多元回归决策树，其叶子节点为局部多元线性模型，能够有效恢复数据中的分段多元线性状态，并在保持可解释性的同时兼顾预测性能。 |
| [^60] | [Directly Optimizing Mean Demographic Parity for Nonlinear Regression](https://arxiv.org/abs/2607.05098) | 该论文提出DPVar（条件均值预测的方差）这一公平性度量，首次实现了非线性回归中平均人口均等准则的直接优化，克服了以往方法仅适用于线性预测器或低维敏感属性、且因过度约束而损害精度的局限。 |
| [^61] | [Minimax PAC Bounds for Learning in Exogenous Contextual MDPs](https://arxiv.org/abs/2606.25170) | 该论文提出了一个在查询已知前后分配采样预算的新型PAC学习框架，并针对带外生上下文的折扣马尔可夫决策过程中的策略评估、最优值估计和最优策略提取任务，给出了极小极大最优的样本复杂度界。 |
| [^62] | [Estimation of the sub-Gaussian Parameter](https://arxiv.org/abs/2606.06384) | 该论文研究零均值随机变量亚高斯参数（方差代理）的估计问题，证明其极小极大风险由刻画分布尾部行为的非增函数 $\delta_P$ 控制，并给出下界为 $r(\sqrt{\log n})+n^{-1/2}$、上界为 $r((\log n)^{1/2-\varepsilon})+n^{-1/2+\varepsilon}$ 的极小极大最优估计量。 |
| [^63] | [Memory by Design: Probabilistic Sequence Layers](https://arxiv.org/abs/2605.31163) | 本文提出了一种设计模型框架，通过贝叶斯滤波和协方差传播统一多种次二次递归序列层，并恢复协方差传播以增强记忆保留和检索。 |
| [^64] | [Concomitant DAG Learning: On the Roles of Noise Adaptivity, Sparsity, and Non-negativity](https://arxiv.org/abs/2605.23537) | 本教程综述了将DAG结构学习重新表述为邻接矩阵上连续、基于评分的估计问题的最新信号处理与优化进展，并阐述了噪声自适应性、稀疏性与非负性在其中的作用。 |
| [^65] | [Keeping Score: Adaptive, Tuning-Free Loss Weighting for Score-Augmented Neural Ratio Estimation](https://arxiv.org/abs/2605.12118) | 提出一种基于损失梯度的自适应、免调参算法来动态设置得分匹配损失的权重，以极小的额外开销提升得分增强神经比率估计代理模型的质量并大幅降低调参成本。 |
| [^66] | [Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes](https://arxiv.org/abs/2605.06152) | 本文证明深度神经网络长期训练中周期性的“弹弓机制”损失尖峰并非源于优化动力学本身，而是浮点精度极限所致——当模型进入高置信度阶段后，正确类别梯度因舍入误差变为零，打破跨类别梯度零和约束，引发分类器与特征间的系统性漂移和正反馈循环。 |
| [^67] | [Learning to Emulate Chaos: Adversarial Optimal Transport Regularization](https://arxiv.org/abs/2604.21097) | 提出对抗性最优传输正则化方法，能够仅从单一含噪轨迹中联合学习高质量的摘要统计量与物理一致的混沌动力学模拟器。 |
| [^68] | [3BASiL: An Algorithmic Framework for Sparse plus Low-Rank Compression of LLMs](https://arxiv.org/abs/2603.01376) | 该论文提出了3BASiL-TM，一种基于新颖三块ADMM算法的高效一次性后训练框架，通过带收敛保证的逐层重构误差最小化与跨Transformer层的联合精炼，实现大语言模型的稀疏加低秩压缩并显著缓解性能下降。 |
| [^69] | [V-ECE: Estimating General Expected Calibration Errors](https://arxiv.org/abs/2602.24230) | 该论文提出V-ECE方法，利用依赖预测的适当评分突破了以往方法仅能估计Bregman散度类校准误差的限制，实现了对包括$L_1$距离在内的一般凸散度（如$L_p$距离）校准误差在二分类和多分类场景下的可靠估计。 |
| [^70] | [Uncertainty Quantification in Federated Granger Causality Learning](https://arxiv.org/abs/2602.13004) | 本文针对客户端特征异构的联邦格兰杰因果学习场景，刻画了跨客户端依赖估计过程中的不确定性传播，并利用边特定的方差有效区分真实的跨客户端依赖关系与虚假的估计边。 |
| [^71] | [Handling Covariate Mismatch in Collaborative Linear Prediction](https://arxiv.org/abs/2602.02083) | 本文研究多中心协同线性预测中各中心记录不同协变量的“协变量不匹配”问题，提出了兼容联邦学习约束的估计方法——低维下基于逐分量聚合的代入估计量，高维下采用保持可交换性的“先插补后岭回归”策略。 |
| [^72] | [One Permutation Is All You Need: Fast, Deterministic Feature Importance and Model Stress-Testing](https://arxiv.org/abs/2512.13892) | 用单次最大-最小秩最优的确定性置换替代多次随机置换，可将特征重要性估计的计算复杂度从 O(B·n·p) 降至 O(n·p)，消除估计方差并保持或提升估计精度，并可扩展用于模型压力测试。 |
| [^73] | [Self-sufficient Independent Component Analysis for Demixing Flows](https://arxiv.org/abs/2512.00665) | 提出一种无先验、无似然的自充分独立成分分析方法，通过最小化条件KL散度并顺序学习解混流模型来从数据中学习解耦信号，同时完全避免了不稳定的对抗训练。 |
| [^74] | [Towards Scalable Meta-Learning of near-optimal Interpretable Models via Synthetic Model Generations](https://arxiv.org/abs/2511.04000) | 本文提出通过合成采样近最优决策树来生成大规模预训练数据的高效可扩展方法，使MetaTree transformer在决策树元学习上达到与真实数据或昂贵最优树预训练相当的性能，同时大幅降低计算成本。 |
| [^75] | [Deep reinforcement learning for optimal trading with partial information](https://arxiv.org/abs/2511.00190) | 该论文提出三种将循环神经网络与强化学习相结合的方法，用于求解具有状态切换参数的部分可观测环境下的最优交易问题，使交易者能够从可观测数据中推断并过滤市场的潜在状态信息。 |
| [^76] | [Convergence of graph Dirichlet energies and graph Laplacians on intersecting manifolds of varying dimensions](https://arxiv.org/abs/2509.24458) | 该论文证明了在多维度相交流形并集上，非归一化图狄利克雷能量渐近只能感知最高维流形内部的变化，而归一化图狄利克雷能量则收敛到能同时适应所有维度的张量化狄利克雷能量，从而为理解机器学习方法如何适应具有不同内在维度的数据提供了理论依据。 |
| [^77] | [The Sample Complexity of Membership Inference and Privacy Auditing](https://arxiv.org/abs/2508.19458) | 本文在高斯均值估计的基础设定下，研究了成员推断攻击的样本复杂度，即确定了成功实施攻击和隐私审计所需的最少参考样本数量。 |
| [^78] | [Conformal Data Contamination Tests for In-distribution Data Acquisition](https://arxiv.org/abs/2507.13835) | 本文提出了一种无分布假设的共形数据污染检验框架，仅需检查少量数据即可识别出对模型个性化最有价值的外部数据代理，从而在数据获取前提供质量保证。 |
| [^79] | [Towards Reasonable Concept Bottleneck Models](https://arxiv.org/abs/2506.05014) | 提出概念推理模型（CREAM），一种可在架构层面显式编码概念间关系与概念-任务关系、并能借助正则化旁路通道处理不完整概念集的概念瓶颈模型新框架，同时引入了与C→Y无关的可解释性评估指标。 |
| [^80] | [Equilibrium Distribution for t-Distributed Stochastic Neighbor Embedding with Generalized Kernels](https://arxiv.org/abs/2505.24311) | 该论文为广义输入输出核下的t-SNE大样本变分问题建立了严格的数学理论，证明了尺度参数的存在唯一性、解的存在性与一致有界性，以及离散最优解收敛到满足平衡方程的紧支撑平衡分布。 |
| [^81] | [A Survey on Archetypal Analysis](https://arxiv.org/abs/2504.12392) | 这是首篇关于原型分析（AA）的综述，系统介绍了其方法论、面临的非凸优化挑战、跨科学领域的广泛应用以及数据建模的最佳实践。 |
| [^82] | [Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](https://arxiv.org/abs/2502.20612) | 提出GloFND方法，通过为每个锚点动态学习阈值，在整个数据集范围内全局识别自监督对比学习中的假负样本，且每次迭代的计算成本与数据集规模无关。 |
| [^83] | [Networks with Finite VC Dimension: Pro and Contra](https://arxiv.org/abs/2502.02679) | 该论文证明有限的VC维虽有利于经验误差的一致收敛，却可能不利于函数逼近，并基于高维几何的测度集中性质证明，此类网络在处理大规模数据集时逼近误差与经验误差均几乎呈确定性行为。 |
| [^84] | [Graphons of Line Graphs](https://arxiv.org/abs/2409.01656) | 本文提出一种通过将稀疏图映射到其线图并利用“平方度性质”使稀疏图产生稠密线图的方法，从而可以应用稠密图极限理论来分析稀疏图，并实证证明能够区分原本都收敛到零图子的不同数量的星形图。 |

# 详细

[^1]: 基于Stein位移场的密度比估计

    Density Ratio Estimation with Stein Displacement Fields

    [https://arxiv.org/abs/2610.12437](https://arxiv.org/abs/2610.12437)

    该论文提出通过Stein位移场参数化密度比（将对数比建模为负的基础分布Stein算子作用于位移场），用单个凸优化问题统一了分布偏移的统计与动力学描述，并据此发展出无需重训即可修正预训练采样器的push-forward算法和拉近数据分布的pull-back算法。

    

    密度比从概率质量的角度量化分布偏移，而位移场则从动力学的角度描述一个分布是如何被输运到另一个分布上的。尽管两者能提供互补的见解，但它们通常被分开估计，且将其中一种转换为另一种需要后处理。在本文中，我们通过一个作用于基础分布的位移场来参数化目标分布与基础分布之间的密度比：对数比被建模为负的基础分布Stein算子作用于该位移场的结果，仅相差一个归一化常数。这使得我们能够通过单个凸优化问题同时获得分布偏移的统计学与动力学描述。迭代这一“估计—移动”步骤可得到两种推断算法：push-forward（前推）移动模型，并在无需重新训练的情况下修正预训练的采样器；而pull-back（回拉）则将数据移动得更接近基础分布。

    arXiv:2610.12437v1 Announce Type: cross  Abstract: Density ratios quantify distribution shift from a probability-mass point of view, whereas displacement fields describe, from a dynamical point of view, how one distribution is transported onto another. Although both offer complementary insights, they are usually estimated separately, and converting one into the other requires post-processing. In this paper, we estimate the density ratio between a target and a base distribution by parametrizing it through a displacement field acting on the base: the log-ratio is modeled as minus the Stein operator of the base applied to the field, up to a normalizing constant. This gives both statistical and dynamical descriptions of the distribution shift through a single convex optimization problem. Iterating this estimate-and-move step gives two inference algorithms: push-forward moves the model and corrects a pretrained sampler without retraining it, whereas pull-back moves the data closer to the ba
    
[^2]: 基于核方法自编码器的双层优化数据驱动Koopman嵌入学习

    Bilevel optimization for data-driven learning of Koopman embeddings using kernel-based autoencoders

    [https://arxiv.org/abs/2610.12370](https://arxiv.org/abs/2610.12370)

    本文提出了一种结合配置方法与双层优化的新方法EDMD-kDL，利用基于核的自编码器直接从数据中学习有限维Koopman嵌入，克服了传统EDMD需先验指定字典的局限，同时相比神经网络方法具有更好的可解释性和理论可分析性。

    

    Koopman算子理论为分析非线性动力系统提供了一个线性框架，并已成为数据驱动建模的主要工具。然而，一个核心挑战在于，由扩展动态模态分解（EDMD）等方法计算的有限维近似需要事先指定字典。近期的机器学习方法通过从数据中学习字典来解决这一局限，其中主要采用人工神经网络（ANN）自编码器架构。尽管核方法提供了一种更具可解释性且更易于理论分析的替代方案，但它们在该领域中受到的关注却很少。我们提出了基于核字典学习的扩展动态模态分解（EDMD-kDL），这是一种直接从数据中学习有限维Koopman嵌入的基于核的方法。该方法结合了配置方法和双层优化的思想。

    arXiv:2610.12370v1 Announce Type: new  Abstract: Koopman operator theory provides a linear framework for analyzing nonlinear dynamical systems and has become a major tool for data-driven modeling. A central challenge, however, is that finite-dimensional approximations computed by methods such as extended dynamic mode decomposition (EDMD) require the dictionary to be specified a priori. Recent machine-learning approaches address this limitation by learning the dictionary from data, predominantly using artificial neural network (ANN) autoencoder architectures. Although kernel methods offer an alternative with greater interpretability and tractability for theoretical analysis, they have received little attention in this setting. We introduce extended dynamic mode decomposition with kernel-based dictionary learning (EDMD-kDL), a kernel-based method for learning finite-dimensional Koopman embeddings directly from data. The method combines ideas from collocation methods and bilevel optimizat
    
[^3]: 缩小对抗性马尔可夫决策过程策略优化中的视野差距

    Closing the Horizon Gap in Policy Optimization for Adversarial MDPs

    [https://arxiv.org/abs/2610.12362](https://arxiv.org/abs/2610.12362)

    该论文提出使用正则化Q函数在所有状态-动作对上联合控制局部更新稳定性，从而将对抗性MDP策略优化的遗憾界对视野H的依赖性改进到与基于占用测度的算法相当的水平。

    

    我们研究具有对抗性损失和老虎机反馈的在线情景表格马尔可夫决策过程（MDP）的策略优化问题。策略优化在每个状态处对策略进行局部更新，避免了在占用测度多面体上进行优化的复杂性，但其现有的遗憾界比基于占用测度的算法大一个视野因子 $H$。我们通过使用正则化 $Q$ 函数来弥补这一差距，这使得我们能够在所有状态-动作对上联合控制局部更新的稳定性，而不是在每个状态处分别控制。由此得到的算法在转移已知的情况下达到 $\widetilde O(\sqrt{HS(H+A)T})$ 的高概率遗憾界，在转移未知的情况下达到 $\widetilde O(HS\sqrt{AT})$ 的高概率遗憾界，其中 $S$ 为状态数，$A$ 为动作数，$T$ 为回合数。这两个界都改善了现有策略优化界对视野的依赖性，且后者匹配了已知的最优结果。

    arXiv:2610.12362v1 Announce Type: new  Abstract: We consider policy optimization for online episodic tabular Markov decision processes (MDPs) with adversarial losses and bandit feedback. Policy optimization updates the policy locally at each state and avoids optimization over the occupancy-measure polytope, but its existing regret bounds are larger by a factor of the horizon $H$ than those of occupancy-measure-based algorithms. We close this gap by using regularized $Q$-functions, which allow us to control the stability of the local updates jointly over all state-action pairs rather than separately at each state. The resulting algorithm attains high-probability regret bounds of $\widetilde O(\sqrt{HS(H+A)T})$ for known transitions and $\widetilde O(HS\sqrt{AT})$ for unknown transitions, where $S$ is the number of states, $A$ the number of actions, and $T$ the number of episodes. Both bounds improve the horizon dependence of existing policy optimization bounds, and the latter matches th
    
[^4]: 面向处理效应估计的预测驱动数据融合方法

    Prediction-Powered Data Fusion for Treatment Effect Estimation

    [https://arxiv.org/abs/2610.12332](https://arxiv.org/abs/2610.12332)

    提出了一种无需对观察性研究做特殊假设的数据融合框架，通过保持RCT估计的无偏性并从大型观察性研究中借力，显著提升平均处理效应（ATE）和条件平均处理效应（CATE）的估计精度。

    

    随机对照试验（RCT）能够在无混杂的情况下识别处理效应，但样本量通常较小；而观察性研究（OBS）样本量大，但可能存在混杂。目前已有许多将小型RCT与大型OBS相结合的估计方法用于估计平均处理效应（ATE）和条件平均处理效应（CATE）。然而，现有的ATE估计方法要么对OBS施加额外假设，要么未能从OBS中充分借力。相比ATE，CATE的研究相对较少，现有的CATE方法要么假设OBS无混杂，要么依赖于混杂函数的模型，要么以引入偏差为代价来换取更低的方差。为此，我们提出了一个框架，在对OBS不做任何特殊假设的前提下融合OBS与RCT：既保持基于RCT估计的无偏性，又从大型OBS中借力以提升估计精度。基于这一原则，我们构建了具有闭式权重的ATE估计器AIPW-Fusion……

    arXiv:2610.12332v1 Announce Type: cross  Abstract: Randomized controlled trials (RCTs) identify treatment effects without confounding but are often small, whereas observational studies (OBS) are large but may be confounded. Many estimators combining a small RCT with a large OBS have been developed for the average treatment effect (ATE) and the conditional ATE (CATE). However, existing ATE estimators either make assumptions on the OBS or do not borrow enough power from them. The CATE has been studied less than the ATE. Existing CATE methods either assume the OBS are unconfounded, rely on a model of the confounding function, or accept bias in exchange for lower variance. We therefore propose a framework that, without special assumptions on the OBS, fuses the OBS and the RCT by preserving the unbiasedness of RCT-based estimation while borrowing power from the large OBS to boost precision. Applying this principle, we build an ATE estimator, AIPW-Fusion, with closed-form weights and confide
    
[^5]: 复合在线到非凸转换及最优预言机复杂度

    Composite Online-to-Nonconvex Conversion with Optimal Oracle Complexity

    [https://arxiv.org/abs/2610.12328](https://arxiv.org/abs/2610.12328)

    该论文通过为在线学习者设计新的损失函数，将在线到非凸转换框架扩展至复合优化场景，首次在一阶随机预言机访问下建立了复合非凸优化的最优复杂度保证。

    

    我们研究随机非光滑非凸复合优化问题，其中包含若干重要问题，如约束优化和神经网络的正则化训练。目标函数是一个可能非光滑的非凸Lipschitz函数与一个凸正则项之和，且该函数仅能通过随机梯度或函数值进行访问。我们的目标是找到一个满足针对复合目标设计的Goldstein型平稳性条件的点。据我们所知，在一阶访问下，目前尚无该设置的已知预言机复杂度界，而零阶访问下已有的复杂度结果也是次优的。为解决这一问题，我们采用在线到非凸转换框架，该框架通过在线学习者来选择更新方向，并且已知在非复合问题上能达到最优速率。我们通过为学习者引入新的损失函数，将该框架扩展到我们的复合场景中……

    arXiv:2610.12328v1 Announce Type: new  Abstract: We consider stochastic nonsmooth nonconvex composite optimization, which includes several important problems such as constrained optimization and the regularized training of neural networks. The objective is the sum of a possibly nonsmooth nonconvex Lipschitz function and a convex regularizer, and the function is accessed through stochastic gradients or function values. The goal is to find a point that satisfies a Goldstein-type stationarity condition designed for composite objectives. To our knowledge, no oracle complexity bound for this setting is known under first-order access, and existing complexities under zeroth-order access are suboptimal. To handle this issue, we employ the framework of online-to-nonconvex conversion, which chooses update directions by an online learner and is known to achieve optimal rates for noncomposite problems. We extend the framework to our composite scenario by introducing new losses for the learner, whi
    
[^6]: 测试代数完全交集

    Testing Algebraic Complete Intersections

    [https://arxiv.org/abs/2610.12288](https://arxiv.org/abs/2610.12288)

    本文提出了一种显式有效的学习检验程序，利用正则多项式系统的定量几何估计，判断高维数据分布是否集中于规定维度、有界次数和有界条件数的实代数完全交集附近，要么返回几何复杂度受控的候选回归流形，要么在受控放宽的阈值下证明此类流形不存在。

    

    给定来自一个可能处于高维实空间中的概率分布的独立同分布样本，我们研究检验该分布是否集中于一个具有规定维度、有界次数和有界条件数的实代数完全交集附近的问题。我们设计了一种显式且有效的学习程序，该程序或者在近似阈值受控放宽的条件下证明此类流形不存在，或者返回一个几何复杂度受控的候选回归流形。等价地说，该程序在此假设类内对流形假设进行检验。所提出的程序依赖于正则多项式系统的定量几何估计，这些估计归结为一个易于求解的辅助优化问题。随后，我们开发了一种数据驱动的算法来求解该辅助优化问题，并为其建立了显式的界。

    arXiv:2610.12288v1 Announce Type: new  Abstract: Given independent and identically distributed samples samples from a probability distribution in a potentially high-dimensional real space, we study the problem of testing whether the distribution is concentrated near a real algebraic complete intersection of prescribed dimension, bounded degree, and bounded condition number. We design an explicit and effective learning procedure which either certifies the nonexistence of such a manifold, up to a controlled relaxation of the approximation threshold, or returns a candidate regression manifold with controlled geometric complexity. Equivalently, the procedure tests the manifold hypothesis within this hypothesis class. The proposed procedure relies on quantitative geometric estimates for regular polynomial systems, which lead to a tractable auxiliary optimization problem. We then develop a data-driven algorithm to solve this auxiliary optimization problem, establishing explicit bounds on its
    
[^7]: ISBO：基于INLA-SPDE方法与对数高斯Cox过程模型的可扩展时空贝叶斯优化

    ISBO: Scalable Spatio-Temporal Bayesian Optimization with Log Gaussian Cox Process Models via the INLA-SPDE Approach

    [https://arxiv.org/abs/2610.12213](https://arxiv.org/abs/2610.12213)

    该论文提出了首个面向时空数据的可扩展贝叶斯优化框架ISBO，通过对数高斯Cox过程建模与INLA-SPDE推断方法，能够以最少的评估次数稳定定位高强度区域及潜在强度峰值。

    

    贝叶斯优化（BO）是一种高效优化昂贵黑盒目标函数的流行方法。然而，使用标准高斯过程的贝叶斯优化并不适合时空问题空间中常用的双重随机Cox过程。我们提出了INLA-SPDE时空贝叶斯优化（ISBO）：首个针对时空数据的可扩展贝叶斯优化框架，该框架使用对数高斯Cox过程（LGCP）对对数强度进行建模，并通过积分嵌套拉普拉斯近似与随机偏微分方程（INLA-SPDE）方法进行推断。在网格上使用Matern场可生成稀疏高斯马尔可夫随机场，使INLA在整个序贯优化过程中提供快速且准确的后验推断。ISBO能够以最少的评估次数稳定地定位高强度区域以及潜在强度的峰值。带有掩蔽机制的时变上置信界采集函数可避免重复访问……

    arXiv:2610.12213v1 Announce Type: cross  Abstract: Bayesian Optimization (BO) is a popular method for efficiently optimizing expensive black-box objectives. However, BO utilizing standard Gaussian Processes is ill-suited for doubly stochastic Cox Processes that are often used in spatio-temporal problem spaces. We introduce INLA-SPDE Spatio-Temporal Bayesian Optimization (ISBO): the first scalable BO framework for spatio-temporal data, that models the log-intensity with a Log-Gaussian Cox Process(LGCP) and performs inference via Integrated Nested Laplace Approximation and Stochastic Partial Differential Equations (INLA-SPDE) approach. Using a Matern field on meshes yields a sparse Gaussian Markov Random Field, where INLA provides fast and accurate posterior inference throughout sequential optimization. ISBO stably locates high-intensity regions and the peak of the latent intensity with minimal evaluations. A time-varying Upper Confidence Bound acquisition with masking avoids revisits, w
    
[^8]: 基于迁移的验证：精确信息前沿及其在调用次数上的代价

    Verification with Transfer: Exact Information Frontiers and Their Price in Calls

    [https://arxiv.org/abs/2610.12211](https://arxiv.org/abs/2610.12211)

    本文从信息论角度为“借助相关源任务（迁移）来降低验证成本”这一策略精确定价：所需最小因果信息由列表率失真函数刻画，并把所有源调用放在首次验证之前不会破坏调用的硬上限，但交错安排可以无界地节省期望调用次数。

    

    一个只接受或拒绝完整答案的验证器所能揭示的信息很少：在 $k$ 比特答案的均匀先验下，要达到零错误需要 $2^k-1$ 次验证。通常的补救方法是先解决相关的源任务，要么像课程学习那样全部先做，要么与验证交错进行。我们从信息和调用次数两个角度为这种补救方法定价。对于精确验证器，任何由源调用与 $n$ 次验证交错组成、并以概率 $s$ 成功的方案所需的最小因果信息，是一个列表率失真函数，且该值可由在任何验证之前的一次观察达到。它给出了二进制源调用期望次数的下界：设计良好的源在唯一答案情形下可在 $1+\log_25$ 次调用内达到该界，一般情况下可在一个对数项内达到，而任何附加常数都不足以做到。在精确验证器和固定源的设定下，将所有调用移到第一次验证之前不会破坏任何调用的硬上限，尽管交错安排可以无界地节省期望调用次数。

    arXiv:2610.12211v1 Announce Type: new  Abstract: A verifier that accepts or rejects whole answers reveals little: under a flat prior over $k$-bit answers, zero error needs $2^k-1$ verifications. The usual remedy is to solve related source tasks, either all first, as a curriculum does, or interleaved with verification. We price this remedy in information and in calls. With an exact verifier, the least causal information that any interleaving of source calls and $n$ verifications needs to succeed with probability $s$ is a list rate-distortion function, attained by one observation before any verification. It lower-bounds the expected number of binary source calls, which designed sources meet within $1+\log_25$ calls for unique answers and within a logarithmic term in general, where no additive constant suffices. With an exact verifier and fixed sources, moving every call before the first verification preserves all hard caps on calls, although interleaving can save unboundedly many expecte
    
[^9]: 基于扩散积分分数的最快变化检测

    Quickest Change Detection with Diffusion-Integrated Scores

    [https://arxiv.org/abs/2610.12200](https://arxiv.org/abs/2610.12200)

    提出了一种无需训练的扩散积分分数CUSUM（DI-SCUSUM）最快变化检测器，通过向样本添加高斯噪声并精确计算Hyvärinen分数来近似对数似然比，实现了指数级误报控制和有保证的一阶检测延迟界。

    

    经典的CUSUM方法依赖于底层分布的对数似然比，而这一般无法仅凭有限的变前与变后样本计算得到。我们提出了扩散积分分数CUSUM（DI-SCUSUM），这是一种无需训练的检测器。我们向样本添加高斯噪声以形成两个平滑的密度估计，并精确计算它们的Hyvärinen分数，而无需训练分数网络。对于每个新到的观测值，我们采样一个扩散时间，对观测值进行扰动，并将重要性加权的分数差作为DI-SCUSUM递归中的增量。在观测值遵循固定经验分布的假设下，变后的平均增量与从平滑后的变后经验分布到平滑后的变前经验分布的Kullback-Leibler（KL）散度成正比。我们建立了指数级的误报缩放特性以及一阶检测延迟界，对于固定阈值和增量……

    arXiv:2610.12200v1 Announce Type: cross  Abstract: Classical CUSUM relies on the log-likelihood ratio of the underlying distributions, which cannot generally be computed from finite pre- and post-change samples alone. We propose diffusion-integrated score CUSUM (DI-SCUSUM), a training-free detector. We add Gaussian noise to the samples to form two smooth density estimates and calculate their Hyv\"arinen scores exactly, without training a score network. For each incoming observation, we sample a diffusion time, perturb the observation, and use the importance-weighted score difference as an increment in the DI-SCUSUM recursion. Under the assumption that observations follow the fixed empirical distributions, the post-change mean increment is proportional to the Kullback-Leibler (KL) divergence from the smoothed post-change to the smoothed pre-change empirical distribution. We establish exponential false-alarm scaling and a first-order delay bound that, for a fixed threshold and increment 
    
[^10]: 面向风险规避决策的信度机器学习

    Credal Machine Learning for Risk-Averse Decision Making

    [https://arxiv.org/abs/2610.12115](https://arxiv.org/abs/2610.12115)

    该论文提出用信度集（概率分布的集合）来表示预测中的认知不确定性，并结合一种新颖的决策规则，实现基于CVaR的可靠风险规避决策。

    

    在许多机器学习应用中，有必要防范可能导致重大损失的最坏情形与预测。原则上，这可以通过训练风险规避的预测模型来实现，即最小化条件风险价值（CVaR）等损失函数，而不是依赖平均表现良好的模型。然而在实践中，由于学习者对真实损失分布（进而对真实CVaR）存在认知不确定性，这种风险规避方法的有效性会受到削弱。为实现可靠的风险规避，我们提出一种方法，将这种认知不确定性表示为信度集，即概率分布的集合。更具体地，我们开发了一个高效且可靠的学习器，以信度集的形式产生预测，并将其与一种新颖的决策规则相结合，该规则将每个信度集映射为用于CVaR最小化的单一预测分布。

    arXiv:2610.12115v1 Announce Type: new  Abstract: In many machine learning applications, it is necessary to guard against worst-case scenarios and predictions that could result in substantial losses. In principle, this can be achieved by training risk-averse predictive models that minimize loss functions such as conditional value-at-risk (CVaR), rather than relying on models that perform well on average. In practice, however, the effectiveness of this approach to risk aversion is undermined by the learner's uncertainty regarding the true loss distribution and, consequently, the true CVaR. To achieve reliable risk-aversion, we propose a method in which this (epistemic) uncertainty is represented in terms of credal sets, i.e., sets of probability distributions. More specifically, we develop an efficient yet reliable learner that produces predictions in the form of credal sets and combine it with a novel decision rule that maps each credal set to a single predictive distribution for CVaR m
    
[^11]: 用于变分序贯蒙特卡洛的可微系统重采样

    Differentiable Systematic Resampling for Variational Sequential Monte Carlo

    [https://arxiv.org/abs/2610.12094](https://arxiv.org/abs/2610.12094)

    提出可微系统重采样（DSR），一种温度控制的系统重采样松弛方法，在保持其结构特性并具有可证明的指数收敛偏差的同时，实现了完全梯度流动，且计算开销远低于基于最优传输的方法。

    

    粒子滤波器是非线性状态估计的标准工具，但其重采样步骤是离散的，阻碍了变分序贯蒙特卡洛中基于梯度的学习。我们提出了可微系统重采样（DSR），这是一种对系统重采样的温度控制松弛方法，它在保持系统重采样基于CDF排序的带状结构的同时，实现了完全的梯度流动。当温度趋于零时，DSR收敛到精确的系统重采样，并且我们证明了由此引入的偏差具有逐点指数收敛速率。与基于最优传输的可微重采样方法相比，DSR避免了迭代求解器的使用，计算开销显著更低。在随机动力系统和真实世界手写数据上的实验表明，DSR在滤波和动力学学习方面取得了相当或更优的性能。

    arXiv:2610.12094v1 Announce Type: cross  Abstract: Particle filters are a standard tool for nonlinear state estimation, but their resampling step is discrete, preventing gradient-based learning in variational sequential Monte Carlo. We introduce Differentiable Systematic Resampling (DSR), a temperature-controlled relaxation of systematic resampling, that preserves the CDF-ordered, banded structure of systematic resampling while enabling full gradient flow. DSR converges to exact systematic resampling as the temperature vanishes, and we prove a pointwise exponential convergence rate for the induced bias. Compared to optimal-transport-based differentiable resampling, DSR avoids iterative solvers and has substantially lower computational overhead. Experiments on stochastic dynamical systems and real-world handwriting data show that DSR achieves comparable or superior filtering and dynamics learning performance.
    
[^12]: 在昂贵模拟器的贝叶斯推断中利用梯度信息

    Exploiting Gradients in Bayesian Inference of Expensive Simulators

    [https://arxiv.org/abs/2610.12076](https://arxiv.org/abs/2610.12076)

    该论文提出在昂贵模拟器的贝叶斯推断中，利用模拟器输出对输入参数的梯度信息作为额外信号来指导基于贝叶斯优化的主动学习过程，从而提高有限模拟预算下的推断效率。

    

    基于微分方程的模拟器在科学和工程领域无处不在。它们常被用于基于模拟的推断中，根据对模拟器输出的真实世界观测来评估输入参数的后验分布。然而，当单次模拟器评估的计算成本很高时，推断就变得具有挑战性。在这种情况下，研究者已采用基于贝叶斯优化的主动学习方法，结合高斯过程代理模型，以从有限的模拟预算中最大化所获得的信息。近年来，模拟器输出关于输入参数的梯度变得越来越容易获得，但它们很少被用于推断。尽管我们只需要学习模拟器的输入-输出关系，梯度信息仍可以提供一个额外的有价值信号来引导主动学习过程。这在昂贵模拟器的情形下尤其值得关注。

    arXiv:2610.12076v1 Announce Type: new  Abstract: Simulators based on differential equations are ubiquitous in science and engineering. They are often used in simulation-based inference to evaluate the posterior distribution of the input parameters based on real-world observations of the simulator outputs. However, inference becomes challenging when individual simulator evaluations are computationally expensive. In such cases, a Bayesian optimization-based active learning approach with Gaussian process surrogate models has been used to maximize the information obtained from a limited simulation budget. Recently, gradients of simulator outputs with respect to input parameters have become increasingly available, yet they are rarely exploited for inference. Even though we only need to learn the simulator input-output relationship, gradient information can provide an additional valuable signal to guide the active learning procedure. This is of particular interest in the case of expensive si
    
[^13]: 扩散模型消除朗之万采样的条件数依赖：高斯情形下的精确分析

    Diffusion Removes Langevin's Conditioning Dependence: A Sharp Gaussian Analysis

    [https://arxiv.org/abs/2610.12052](https://arxiv.org/abs/2610.12052)

    本文在高斯情形下证明扩散模型的采样误差为 $O(\sqrt{d\lambda_{\max}}\log N/N)$，消除了经典朗之万类采样器中依赖条件数的 $\sqrt{\kappa}$ 因子，并通过精确的谱界与匹配的一阶渐近分析，严格解释了扩散模型优于传统基于分数采样器的理论原因。

    

    尽管扩散模型在实证上取得了巨大成功，但它们为何能突破经典基于分数的采样器的瓶颈仍不清楚。在本工作中，我们利用高斯分布来分离这一现象。我们针对优化超参数建立了2-Wasserstein收敛界，证明扩散过程能达到 $O(\sqrt{d\lambda_{\max}}\log N/N)$ 的采样误差，其中 $d$ 是维度，$N$ 是采样步数，$\lambda_{\max}$ 是目标协方差矩阵的最大特征值。相比之下，未校正朗之万动力学和欠阻尼朗之万动力学则额外多出 $\sqrt{\kappa}$ 因子，其中 $\kappa$ 是条件数。这些速率来自精确的谱界：我们通过在 $N\rightarrow\infty$ 时匹配的一阶渐近性证实了它们。我们的分析在高斯情形下严格刻画了依赖时间的分数轨迹如何在采样过程中消除条件数依赖。

    arXiv:2610.12052v1 Announce Type: cross  Abstract: Despite their empirical success, why diffusion models overcome the bottlenecks of classical score-based samplers remains unclear. In this work, we leverage Gaussian distributions to isolate this phenomenon. We establish 2-Wasserstein convergence bounds for optimized hyperparameters, showing that diffusion processes achieve a sampling error of $O(\sqrt{d\lambda_{\max}}\log N/N)$, where $d$ is the dimension, $N$ the number of sampling steps, and $\lambda_{\max}$ the largest eigenvalue of the target covariance matrix. Unadjusted and underdamped Langevin dynamics suffer from an additional $\sqrt\kappa$ factor, where $\kappa$ is the condition number. These rates follow from spectral bounds which are sharp: we confirm them via matching first-order asymptotics as $N\rightarrow\infty$. Our analysis provides a rigorous characterization, in the Gaussian setting, of how time-dependent score trajectories remove condition-number dependence during s
    
[^14]: 面向离散数据的高效且可泛化的原型分析

    Efficient and Generalizable Archetypal Analysis for Discrete Data

    [https://arxiv.org/abs/2610.12035](https://arxiv.org/abs/2610.12035)

    提出了一种面向离散数据的基于似然的高效原型分析框架，支持伯努利、泊松和多项式观测模型，并引入交叉验证的预测似然准则来有原则地选择原型数量。

    

    原型分析将观测数据表示为极端数据驱动轮廓的凸组合，从而为复杂数据集提供可解释的低维描述。经典的原型分析依赖于最小二乘目标函数，这并不适合诸如二值、计数和分类数据等离散观测数据。我们引入了一个高效的基于似然的原型分析框架，支持伯努利、泊松和多项式观测模型。我们的优化方案采用负对数似然的局部二次近似，通过序列最小优化（SMO）和活动集方法实现约束更新。通过在保持单纯形可行性的同时对活动集进行限制，提升了算法的可扩展性。我们进一步引入了交叉验证的预测似然准则来选择原型的数量，为重构误差启发式方法和基于稳定性的诊断方法提供了一种更有原则的替代方案。

    arXiv:2610.12035v1 Announce Type: cross  Abstract: Archetypal Analysis (AA) represents observations as convex combinations of extremal data-driven profiles, yielding interpretable low-dimensional descriptions of complex datasets. Classical AA relies on a least-squares objective, which is poorly suited to discrete observations such as binary, count, and categorical data. We introduce an efficient likelihood-based framework for AA supporting Bernoulli, Poisson, and multinomial observation models. Our optimization scheme employs local quadratic approximations of the negative log-likelihood, enabling constrained updates through sequential minimal optimization (SMO) and an active-set method. Scalability is improved by bounding the active set while preserving simplex feasibility. We further introduce a cross-validated predictive likelihood criterion for selecting the number of archetypes, providing a principled alternative to reconstruction-error heuristics and stability-based diagnostics. S
    
[^15]: 基于距离草图的高效二次熵计算

    Efficient quadratic entropy with distance sketches

    [https://arxiv.org/abs/2610.11976](https://arxiv.org/abs/2610.11976)

    本文提出了一种基于随机特征嵌入、投影和控制变量技术的可扩展二次熵近似方法，并在文献计量学应用中仅凭引用和文本特征揭示了论文、领域和机构的跨学科影响力。

    

    我们详细介绍了用于近似任意分布 $p$ 和负型常见距离 $d$ 下二次熵 $p^T d p$ 的可扩展方法。我们聚焦于欧几里得距离和球面测地距离两种情形，二者均在简单的框架内利用随机特征嵌入和投影来显著改善计算复杂度。通过摊销单次大型矩阵乘法以及控制变量技术，在 $d$ 保持不变而 $p$ 变化的场景下，该方法进一步实现了低内存占用和短运行时间的大规模计算。我们通过与直接配对采样方法的对比，以及在 Open Graph Benchmark 数据集上的文献计量学/科学计量学示例验证了该方法，仅凭引用和文本特征即可揭示跨学科影响力特别窄或特别宽的论文、领域和机构。

    arXiv:2610.11976v1 Announce Type: cross  Abstract: We detail scalable methods for approximating the quadratic entropy $p^T d p$ for arbitrary distributions $p$ and common distances $d$ of negative type. We focus on the Euclidean and spherical geodesic cases, which both use random feature embeddings and projections to dramatically improve computational complexity within a simple framework. Amortization of a single large matrix multiplication and control variates further enable computation at large scale with low memory and runtime in situations where $d$ is held constant while $p$ varies. We demonstrate this with a comparison against direct pair sampling and bibliometric/scientometric examples on Open Graph Benchmark datasets, revealing papers, fields, and institutions with both particularly narrow and broad interdisciplinary reach from their citations and text features alone.
    
[^16]: 基于评分的方法从干预中学习聚类DAG

    Score-Based Learning of Cluster DAGs from Interventions

    [https://arxiv.org/abs/2610.11947](https://arxiv.org/abs/2610.11947)

    提出首个基于评分的方法COARSE，利用干预数据识别聚类间的因果顺序并将边学习简化为局部搜索，从而在线性高斯假设下实现聚类DAG的学习。

    

    因果抽象的图方法将一个包含众多测量变量的低层因果有向无环图（DAG）转换为一个更小的高层DAG，其节点对原始变量进行聚类，其边总结了聚类之间的因果关�系。这样的聚类DAG更易于解释，但学习它们需要找到聚类并恢复它们之间的边。Madaleno等人（2026）以两个基于约束的阶段学习干预粗化（即将干预无法区分的变量合并而成的聚类DAG）：先学习聚类，再学习边。我们提出了COARSE，这是该任务中第一个基于评分的方法：它保留了两个阶段的结构，但在线性高斯假设下，将基于约束的边学习阶段替换为基于评分的阶段。我们证明干预本身可以识别聚类之间的因果顺序，并且边的学习可以简化为针对每个聚类的单一局部搜索。

    arXiv:2610.11947v1 Announce Type: cross  Abstract: Graphical approaches to causal abstraction transform a low-level causal directed acyclic graph (DAG) over many measured variables into a smaller, high-level DAG whose nodes cluster the original variables and whose edges summarize the causal relations between clusters. Such cluster DAGs are easier to interpret, but learning them requires finding the clusters and recovering the edges between them. Madaleno et al. (2026) learn the interventional coarsening (the cluster DAG that merges variables the interventions cannot distinguish) in two constraint-based phases: first the clusters, then the edges. We introduce COARSE, the first score-based method for this task: it keeps the two-phase structure but, under linear Gaussian assumptions, swaps the constraint-based edge phase for a score-based one. We show that the interventions themselves identify a causal order over the clusters, and learning the edges reduces to a single local search per cl
    
[^17]: RobustLDS：在对抗性污染下学习线性动力系统

    RobustLDS: Learning linear dynamical systems under adversarial corruptions

    [https://arxiv.org/abs/2610.11906](https://arxiv.org/abs/2610.11906)

    该论文提出了基于最小截断二乘法松弛与离群值组稀疏性的估计器，用于在对抗性污染下从单条轨迹学习线性动力系统，并通过非渐近误差界证明了其对离群值的鲁棒性。

    

    我们研究了从长度为 $T$ 的单条轨迹中、在对抗性污染下学习线性动力系统的问题。尽管线性动力系统的辨识本身已被广泛研究，但在对抗性污染下的鲁棒系统辨识问题却相对较少被探索。在这项工作中，我们研究了 $T$ 个观测值中有一部分被对抗性离群值污染的设定。我们提出了基于最小截断二乘法（least-trimmed squares）松弛的不同估计器，并配合一种交替最小化算法。此外，我们还提出了两个利用离群值组稀疏性的估计器（分别通过惩罚项和硬约束实现）。对于带组稀疏惩罚的估计器，我们推导了非渐近误差界，证明了其对离群值的鲁棒性。我们还通过实验证明所提出的估计器在实践中表现良好。

    arXiv:2610.11906v1 Announce Type: cross  Abstract: We consider the problem of learning linear dynamical systems under adversarial contamination from a single trajectory of length $T$. While identification of linear dynamical systems itself is well-studied, the problem of robust system identification under adversarial contamination is relatively less explored. In this work, we study the setting where a fraction of the $T$ observations are contaminated by adversarial outliers. We propose different estimators based on relaxations of least-trimmed squares along with an alternating minimization algorithm. Furthermore, we also propose two estimators which exploit the group-sparsity (through penalization/hard-constraints) of the outliers. For the estimator with group-sparse penalty, we derive non-asymptotic error bounds which establish its robustness to outliers. We also show empirically that the proposed estimators work well in practice.
    
[^18]: 从缺失观测中学习结构化线性动力系统

    Learning structured linear dynamical systems from missing observations

    [https://arxiv.org/abs/2610.11869](https://arxiv.org/abs/2610.11869)

    本文提出一种基于偏差校正目标函数的估计器，用于在观测严重缺失的情况下学习凸集约束下的结构化线性动力系统，给出了依赖集合局部复杂度、轨迹长度和采样概率的非渐近误差界，并证明即使轨迹远短于无约束情形且采样概率趋于零时仍能有意义地恢复转移矩阵。

    

    我们研究在凸集 $\mathcal{K}$ 上学习结构化线性动力系统的问题，其中每个时间点只有一小部分观测是可用的。我们提出了一种估计器，该估计器最小化一个经过偏差校正的、可能是非凸的目标函数。我们获得了统计误差的非渐近界，该界取决于 $\mathcal{K}$ 的局部复杂度、轨迹长度 $T$ 以及子采样概率 $p$。此外，我们还建立了投影梯度下降算法的收敛性。该一般理论被应用于以下三种情形：(i) $\mathcal{K}$ 是一个子空间；(ii) $\mathcal{K}$ 是双等张（bi-isotonic）矩阵集合；(iii) $\mathcal{K}$ 是行由采样 Lipschitz 函数构成的矩阵集合。我们证明，即使轨迹长度 $T$ 远小于无约束情形下所需的值，且子采样概率 $p = o(1)$，仍然能够对转移矩阵实现有意义的恢复。

    arXiv:2610.11869v1 Announce Type: cross  Abstract: We consider the problem of learning structured linear dynamical systems over convex sets $\mathcal{K}$, where only a small subset of the observations are available at each time point. An estimator which minimizes a bias-corrected, potentially non-convex objective function is proposed. Non-asymptotic bounds are obtained for the statistical error, which depend on the local complexity of $\mathcal{K}$, the trajectory length $T$, and the sub-sampling probability $p$. Convergence of the projected gradient descent algorithm is also established. The general theory is applied to settings where (i) $\mathcal{K}$ is a subspace, (ii) $\mathcal{K}$ is the set of bi-isotonic matrices, and (iii) $\mathcal{K}$ is the set of matrices whose rows are formed by sampling Lipschitz functions. We show meaningful recovery of the transition matrix is possible for values of $T$ much smaller than what is required in the unconstrained case, and for $p = o(1)$.
    
[^19]: 条件核斯坦因差异

    Conditional Kernel Stein Discrepancy

    [https://arxiv.org/abs/2610.11863](https://arxiv.org/abs/2610.11863)

    提出了一个通过协变量空间上的算子值核将核斯坦因差异推广到条件设定的框架，用于在仅知道非归一化条件目标模型和联合分布样本的情况下量化条件拟合优度。

    

    核斯坦因差异为比较分布提供了一种通用的工具。其主要应用之一是量化数据生成分布与给定目标分布之间的拟合优度。在本工作中，我们研究了与之相关的条件拟合优度量化问题：在仅给定一个（可能是非归一化的）条件目标模型、不掌握其协变量分布的信息、但拥有来自联合分布的样本的情况下，目标是评估样本的条件分布与目标分布的匹配程度。为解决这一设定，我们提出了一个框架，通过协变量空间上的算子值核，将无条件的KSD提升到条件设定中，超越了已知的欧几里得情形。我们证明了，当且仅当条件模型与真实条件分布在几乎所有协变量处一致时，我们所提出的统计量才会为零。

    arXiv:2610.11863v1 Announce Type: cross  Abstract: Kernel Stein discrepancies (KSDs) provide a versatile tool for comparing distributions. One of their main applications is in quantifying the goodness-of-fit (GoF) between a data-generating distribution and a prescribed target distribution. In this work, we study the related problem of conditional GoF quantification: given only a (possibly non-normalized) conditional target model, without information on the distribution of its covariates, and samples from a joint distribution, the goal is to assess how well the conditional distribution of the samples matches the target. To tackle this setting, we present a framework that allows lifting unconditional KSDs to the conditional setting through an operator-valued kernel on the covariate space, going beyond the known Euclidean case. We establish that our suggested statistic vanishes if and only if the conditional model and the true conditional distribution agree for almost all covariates and d
    
[^20]: 高斯混合分布上的Softmax注意力：能线性时则线性，需选择时方选择

    Softmax Attention on Gaussian Mixtures: Linear When It Can, Selective When It Must

    [https://arxiv.org/abs/2610.11798](https://arxiv.org/abs/2610.11798)

    该论文通过研究softmax注意力在高斯混合分布上的无穷提示极限，证明softmax注意力既能像线性注意力一样有效解决线性任务，又能借助查询依赖的选择能力，以梯度方法学习到监督分类、去噪等具有潜在结构、多峰性和非线性依赖的统计任务的最优解。

    

    Softmax注意力作为Transformer的核心，已展现出卓越的能力，然而其底层机制仍未被完全理解。近期的理论研究针对高斯提示展开，在无穷提示极限下，softmax注意力退化为一个线性映射，但这也消除了它区别于线性注意力的查询依赖选择能力。本研究探讨softmax注意力在高斯混合分布上的无穷提示极限，高斯混合分布在保留高斯数据可处理性的同时，引入了潜在结构、多峰性和非线性依赖。我们证明softmax注意力能够通过基于梯度的方法表示并学习一系列统计任务（包括监督分类和去噪）的最优解。我们的结果凸显了softmax注意力的两种互补能力：它能够像其更简单的线性对应方法一样有效地完成线性任务，同时还能利用查询依赖的选择能力来应对更复杂的分布。

    arXiv:2610.11798v1 Announce Type: cross  Abstract: Softmax attention, at the heart of Transformers, has demonstrated remarkable capabilities. Yet its underlying mechanisms remain only partially understood. Recent theoretical work studies Gaussian prompts, where the infinite-prompt limit reduces softmax attention to a linear map, but also removes the query-dependent selection that distinguishes it from linear attention. This work studies the infinite-prompt limit of softmax attention on Gaussian mixtures, which retain the tractability of Gaussian data while introducing latent structure, multimodality, and nonlinear dependencies. We show that softmax attention can represent and learn, via gradient-based methods, optimal solutions to a range of statistical tasks, including supervised classification and denoising. Our results highlight two complementary capabilities of softmax attention: it can recover linear tasks as effectively as its simpler linear counterpart, while also exploiting que
    
[^21]: 球对称分布的最优随机量化器

    Optimal random quantisers for spherically symmetric distributions

    [https://arxiv.org/abs/2610.11772](https://arxiv.org/abs/2610.11772)

    该论文证明对于球对称目标分布，随机量化器的优化问题具有凸性，并据此发现均匀分布在适当半径球面上的随机量化器即使在中等样本量下也表现优异、被数值验证为全局最优，从而克服了Zador渐近理论中高维所需的天文数字级样本量问题。

    

    Zador的著名定理是最优量化理论的基石：它既确定了 $\mathbb{R}^d$ 中最优 $n$ 点量化器经验分布的弱极限，也给出了相应的 $L_s$ 平均量化误差的衰减速率。然而，在高维情形下，观测到这种渐近行为需要天文数字般庞大的样本量。我们证明，对于球对称目标分布，在所有球对称分布上进行优化是一个凸问题，并推导出一个等价定理，该定理既刻画了全局最优性，又给出了一个构造性算法。我们表明，对于中等规模的 $n$，均匀分布在半径适当选取的球面上的随机量化器表现异常出色，并且在广泛的 $n$ 取值范围内，经数值验证在所有随机量化器中是最优的。它们的期望失真具有一个可以计算的显式积分表示。

    arXiv:2610.11772v1 Announce Type: cross  Abstract: Zador's celebrated theorem is a cornerstone of optimal quantisation: it establishes both the weak limit of the empirical distribution of an optimal $n$-point quantiser in $R^d$ and the decay rate of the associated $L_s$-mean quantisation error. In large dimension, however, observing this asymptotic behaviour requires an astronomically large sample size. We prove that, for spherically symmetric target distributions, optimisation over all spherically symmetric distributions is a convex problem and derive an equivalence theorem that both characterises global optimality and yields a constructive algorithm. We show that, for moderate $n$, random quantisers uniformly distributed on a sphere of suitably chosen radius $R$ perform exceptionally well and, over a broad range of values of $n$, are numerically certified to be optimal among all random quantisers. Their expected distortion has an explicit integral representation that can be evaluated
    
[^22]: σTransfer：基于μP（最大更新参数化）从小网络到大网络的不确定性迁移

    $\sigma$Transfer: Uncertainty Transfer from Small to Large Networks under $\mu\mathrm{P}$

    [https://arxiv.org/abs/2610.11668](https://arxiv.org/abs/2610.11668)

    该论文提出σTransfer方法，在μP参数化下通过重新缩放先验协方差，使拉普拉斯近似所需的先验精度可以从小模型零样本迁移到大模型，从而免去在大模型上的昂贵精度搜索，实测加速可达约5000倍。

    

    拉普拉斯近似中可靠的预测不确定性在很大程度上取决于先验精度，然而选择该精度需要进行后验扫描，对于拥有数十亿参数的神经网络而言，其代价高得令人望而却步。在最大更新参数化（μP）下，我们推导出一种先验协方差的重新缩放方法，使得所选精度随模型宽度的增长保持稳定。由此产生了σTransfer：我们在较小的模型上选择精度，然后将其零样本迁移到规模大得多的模型上，即完全无需在较大模型上搜索精度。我们在明确的条件下证明了先验核、后验协方差、所选精度以及基于后验得出的决策的收敛性，并在回归、图像分类和Transformer读取头等任务上验证了σTransfer。例如，从宽度12的模型迁移时，实测的精度扫描加速比可达约5000倍。

    arXiv:2610.11668v1 Announce Type: cross  Abstract: Reliable predictive uncertainty in Laplace approximations depends critically on the prior precision, yet selecting it requires a posterior sweep that is prohibitively expensive for neural networks with billions of parameters. Under the Maximal Update Parametrization ($\mu\mathrm{P}$), we derive a rescaling of the prior covariance that makes the selected precision stable as model width grows. This leads to $\sigma\mathrm{Transfer}$: we select the precision on a smaller model and zero-shot transfer it to the much larger model, i.e., without searching for the precision on the larger model at all. We show convergence of the prior kernel, posterior covariance, selected precision, and posterior-derived decisions under explicit conditions, and verify $\sigma\mathrm{Transfer}$ across regression, image classification, and Transformer readouts. For example, measured precision-sweep speedups reach $\sim 5000\times$ when transferring from width 12
    
[^23]: 持续机器遗忘的极小极大高斯机制

    Minimax Gaussian Mechanisms for Continual Machine Unlearning

    [https://arxiv.org/abs/2610.11628](https://arxiv.org/abs/2610.11628)

    本文提出基于牛顿更新与高斯差分隐私的极小极大高斯机制，通过推导残差误差上界来校准噪声方差分配，使得顺序删除记录后发布的一系列模型在统计上与精确重训练难以区分，并最小化最坏情况下的噪声方差。

    

    机器遗忘是指在记录被删除后更新已训练的模型，目标是在不重复完整训练过程的情况下，达到与精确重训练相同的效果。我们针对顺序删除请求，为牛顿更新开发了高斯机制。利用高斯差分隐私（GDP）及其自适应组合规则，我们证明了所发布模型的完整序列在统计上难以与对应的精确重训练区分开来。为了在经验风险最小化场景下校准这些机制，我们推导了牛顿近似相对于精确重训练的误差上界，以及该误差在每批删除之后变化幅度的上界。独立高斯噪声利用每次发布时完整残差的界进行校准，而高斯随机游走噪声则使用残差增量的更紧的界。这些界给出了在所得GDP认证下，使各次发布中最坏情况最大噪声方差最小化的噪声分配方案。

    arXiv:2610.11628v1 Announce Type: cross  Abstract: Machine unlearning updates a trained model after records are deleted, aiming to match exact retraining without repeating the full training procedure. We develop Gaussian mechanisms for Newton updates under sequential deletion requests. Using Gaussian differential privacy (GDP) and its adaptive composition rule, we show that the full sequence of released models is statistically difficult to distinguish from matched exact retraining. To calibrate these mechanisms for empirical risk minimization, we derive upper bounds on the error of the Newton approximation relative to exact retraining and on how this error changes after each deletion batch. Independent Gaussian noise is calibrated using bounds on the full residual at each release, whereas Gaussian random walk noise uses smaller bounds on residual increments. These bounds yield allocations minimizing the worst-case maximum noise variance across releases under the resulting GDP certifica
    
[^24]: 条件独立性检验中的嵌入偏差

    Embedding-Bias in Conditional Independence Testing

    [https://arxiv.org/abs/2610.11584](https://arxiv.org/abs/2610.11584)

    该研究揭示了在条件独立性检验中用嵌入替代原始变量所引发的偏差问题，并证明对于残差相关性检验，只要嵌入遗漏的条件均值部分互不相关即可保证检验有效，否则偏差可精确量化为遗漏部分绝对相关性乘以两个偏R²值的几何平均数。

    

    为了检验在给定文本或图像 Z 的条件下 X 和 Y 的条件独立性，一种做法是用嵌入 ψ(Z) 代替 Z 进行条件化。这种嵌入检验有效的前提是：在给定 ψ(Z) 的条件下 Z 与 X 或 Y 独立，而这一条件无法从数据中得到验证；当该条件不成立时，原假设下的拒绝概率甚至可能趋近于一。我们研究了这种失效现象，并证明若聚焦于特定形式的依赖关系，则可以放宽嵌入所需保留的信息要求。对于一种受广义协方差度量启发的残差相关性检验而言，其有效性只要求 E[X|Z] 与 E[Y|Z] 中被 E[X|ψ(Z)] 和 E[Y|ψ(Z)] 遗漏的部分互不相关。若不满足该条件，则可将被丢弃的信息视为遗漏变量。在原假设下，偏差等于遗漏部分之间的绝对相关系数乘以两个偏 R² 值的几何平均数。

    arXiv:2610.11584v1 Announce Type: cross  Abstract: To test conditional independence of $X$ and $Y$ given a text or an image $Z$, one conditions on an embedding $\psi(Z)$ in place of $Z$. The embedded test is valid if $Z$ is independent of $X$ or of $Y$ given $\psi(Z)$, which cannot be confirmed from data, and when this fails, the rejection probability under the null hypothesis can tend to one. We study this failure, and show that focusing on a specific form of dependence relaxes what the embedding must retain. For a residual correlation test inspired by the Generalised Covariance Measure, validity only requires that the parts of $\mathbb{E}[X \mid Z]$ and $\mathbb{E}[Y \mid Z]$ missed by $\mathbb{E}[X \mid \psi(Z)]$ and $\mathbb{E}[Y \mid \psi(Z)]$ are uncorrelated. Otherwise, we treat the discarded information as an omitted variable. Under the null hypothesis, the bias equals the absolute correlation of the missed parts times the geometric mean of two partial $R^2$ values. This identi
    
[^25]: 使用图神经网络从时空追踪数据中自动检测足球比赛阶段

    Automated Detection of Match Phases in Football from Spatio-Temporal Tracking Data Using Graph Neural Networks

    [https://arxiv.org/abs/2610.11571](https://arxiv.org/abs/2610.11571)

    该论文提出了一种结合图神经网络与LSTM的框架，利用足球比赛的时空追踪数据逐秒自动分类七个比赛阶段，其性能超越XGBoost等所有基线模型，宏F1分数高出4.6%。

    

    时空追踪数据为检测足球比赛中的复杂战术模式开辟了新的可能性，然而对多名球员的交互运动进行建模仍然充满挑战。本文提出了一个将图神经网络（GNN）与序列模型相结合的框架，以逐秒的方式在七类分类体系中对比赛阶段进行分类。比赛阶段分类具有战术意义，并且在203场比赛中可以获取基于规则的标签，这使其成为对邻接矩阵构建方式和消息传递层进行系统比较的合适试验平台，而这一问题在现有研究中受到的关注有限。我们选定的GNN-LSTM模型优于所有聚合特征基线模型，包括XGBoost和长短期记忆网络（LSTM），其中最强基线在宏平均F1分数上低4.6%。采用基于领域知识构建的德劳内三角剖分（Delaunay triangulation）的图表示方法，可近似传球路线……（摘要在此处被截断）

    arXiv:2610.11571v1 Announce Type: cross  Abstract: Spatio-temporal tracking data has opened new possibilities for detecting complex tactical patterns in football, yet modeling the interactive movements of multiple players remains challenging. This paper proposes a framework combining graph neural networks (GNNs) with a sequential model to classify match phases on a second-by-second basis across a seven-class taxonomy. Match phase classification is tactically meaningful, and the availability of rule-based labels across 203 matches makes it a suitable testbed for a systematic comparison of adjacency constructions and message-passing layers, a question that has received limited attention in existing research. Our selected GNN-LSTM model outperforms all aggregated-feature baselines, including XGBoost and a Long Short-Term Memory (LSTM) network, as the strongest baseline scores 4.6% lower in macro F1. Graph representations using a domain-informed Delaunay triangulation that approximates pas
    
[^26]: LAIR-Net：用于表格回归的泄漏对齐-脉冲残差网络

    LAIR-Net: Leaky Alignment-Impulse Residual Networks for Tabular Regression

    [https://arxiv.org/abs/2610.11538](https://arxiv.org/abs/2610.11538)

    LAIR-Net通过泄漏残差过渡将浅层学习的锚点注入隐状态演化中，实现了对隐状态的目标感知控制，在23个表格回归基准数据集上超越了八个随机化网络和十二个传统模型，并在非线性目标结构可学习时收益最大。

    

    深度随机化模型通过随机初始化固定隐层参数，仅学习闭式解的读出层，通常通过堆叠随机变换来增加深度，而对隐状态演化缺乏目标感知的控制。我们提出了LAIR-Net（泄漏对齐-脉冲残差网络），它通过泄漏残差过渡将一个浅层学习的锚点混合到每个隐状态中。我们推导了输入扰动敏感性的深度一致界，并通过受控仿真将相对于随机化基线的性能提升归因于锚点本身，而非递归结构或额外的模型容量。当非线性目标结构在可用噪声水平下可学习时，收益显现；而对于近似线性的目标或噪声占主导的情况，收益减弱。在23个基准数据集上，LAIR-Net在八个随机化网络和十二个传统模型中取得了最佳平均排名，其相对性能与非线性目标结构的可学习性相关联。

    arXiv:2610.11538v1 Announce Type: cross  Abstract: Deep randomized models fix hidden-layer parameters through random initialization and learn only closed-form readouts, typically adding depth by stacking random trans formations without target-aware control of hidden-state evolution. We propose LAIR Net, the Leaky Alignment-Impulse Residual Network, which mixes a shallow learned anchor into each hidden state through a leaky residual transition. We derive a depth uniform bound on input-perturbation sensitivity and use controlled simulations to attribute gains over a randomized baseline to the anchor rather than recursion or added capacity. Benefits emerge when a nonlinear target structure is learnable at the available noise level and diminish for nearly linear targets or dominant noise. Across 23 benchmark datasets, LAIR-Net achieves the best average rank among eight randomized networks and twelve conventional models, with relative performance associated with the same nonlinear-structure
    
[^27]: PSI-SINDy：面向非线性动力学稀疏识别的选择后推断

    PSI-SINDy: Post-Selection Inference for Sparse Identification of Nonlinear Dynamics

    [https://arxiv.org/abs/2610.11486](https://arxiv.org/abs/2610.11486)

    本文提出PSI-SINDy，通过选择后推断方法为SINDy识别出的动力学项提供有效的假设检验和置信区间，从而消除选择偏差并量化所选动力学项的统计可靠性。

    

    稀疏非线性动力学识别（SINDy）是一种数据驱动框架，它通过从预先指定的候选动力学项库中识别出一个稀疏子集，从时间序列数据中发现系统的主导动力学。在本工作中，我们开发了一个统计推断框架，通过假设检验和置信区间来量化SINDy所选动力学项的可靠性。一个关键困难在于，使用同一条含噪轨迹既进行动力学项选择又评估其统计显著性会引入选择偏差。选择后推断为解决此类偏差提供了一个有原则的框架，据此我们提出了PSI-SINDy，一种专为SINDy量身定制的选择后推断方法。由于SINDy涉及候选项中的测量误差以及响应与设计之间的共享噪声，直接应用现有的选择后推断技术具有挑战性。为解决这些挑战……（原文摘要在此处截断）

    arXiv:2610.11486v1 Announce Type: cross  Abstract: Sparse identification of nonlinear dynamics (SINDy) is a data-driven framework for discovering governing dynamics from time-series data by identifying a sparse subset of candidate dynamical terms from a prespecified library. In this work, we develop a statistical inference framework for quantifying the reliability of dynamical terms selected by SINDy through hypothesis tests and confidence intervals. A key difficulty is that using the same noisy trajectory for both selecting dynamical terms and assessing their statistical significance can introduce selection bias. Post-selection inference provides a principled framework for addressing such bias, and we propose PSI-SINDy, a post-selection inference method tailored to SINDy. Direct application of existing post-selection inference techniques is challenging because SINDy involves measurement error in the candidate terms and shared noise between the response and design. To address these cha
    
[^28]: 罕见的门控分歧可能限制可塑性：当梯度流错误预测有限批量SGD时

    Rare Gate Disagreements Can Limit Plasticity: When Gradient Flow Mispredicts Finite-Batch SGD

    [https://arxiv.org/abs/2610.11475](https://arxiv.org/abs/2610.11475)

    该论文证明总体梯度流可能在定性上错误预测有限批量SGD：在双神经元ReLU回归中，当预训练时间超过 $\log(b/\eta)$ 后，源于罕见门控分歧的机制使在线SGD在指数级长的时间范围内以高概率丧失可塑性、无法适应目标任务，而梯度流却只需线性时间即可恢复。

    

    总体梯度流是研究神经网络如何适应（包括预训练之后的适应）的常用工具。我们证明它可能在定性层面错误预测有限批量的随机梯度下降（SGD），并将这种偏差追溯到一种特定机制。在一个双单元ReLU回归中，源任务驱动两个神经元趋于正比例关系，而目标任务则奖励将它们分离。在源任务上训练时间 $T$ 之后，梯度流可在与 $T$ 呈线性关系的时间内于目标任务上恢复。而若两个阶段均采用批大小为 $b$、步长为 $\eta$ 的在线SGD，一旦 $T \gtrsim \log(b/\eta)$，它就会在 $e^{c/\eta}$ 量级的时间范围内以高概率失败，且该失败在一组显式构造的、高斯概率超过百分之一的初始化集合上一致成立。对于每个固定的 $T$，小步长SGD仍可恢复，因此这种失败需要小步长与长预训练的联合极限。在目标克隆处，总体不稳定性……

    arXiv:2610.11475v1 Announce Type: new  Abstract: Population gradient flow is a common tool for reasoning about how neural networks adapt, including after pretraining. We show that it can mispredict finite-batch stochastic gradient descent (SGD) qualitatively, and we trace the discrepancy to a specific mechanism. In a two-unit ReLU regression, a source task drives the two neurons toward positive proportionality and a target task rewards separating them. After source training for time $T$, gradient flow recovers on the target in time linear in $T$. Online SGD with batch size $b$ and step size $\eta$ in both phases instead fails with high probability throughout a horizon of order $e^{c/\eta}$ once $T \gtrsim \log(b/\eta)$, uniformly on an explicit set of initializations with Gaussian probability above one percent. For each fixed $T$, small-step SGD still recovers, so the failure requires the joint limit of small steps and long pretraining. At the target clone, the population instability i
    
[^29]: 特征空间自适应实现轻松的高斯过程流

    Feature Space Adaptation for Effortless Gaussian Process Flows

    [https://arxiv.org/abs/2610.11459](https://arxiv.org/abs/2610.11459)

    该论文通过引入核近似和基于扩散工作量的边际似然估计方法，首次实现了 FlowGP 框架内的超参数自动优化，使其能够高效扩展到高分辨率域并处理非高斯条件推断任务。

    

    在线性高斯范畴之外，从高斯过程（GP）进行条件采样是具有挑战性的。近期的方法如 FlowGP（Moss 等人，2026）能够以任意非线性和非高斯条件语句为条件，但代价相当大：需要昂贵的高维迭代扩散过程，且需要手动指定核超参数。在本文中，我们通过以下两点缓解了 FlowGP 的两个显著缺陷：（1）引入核近似方法，使其能够扩展到高分辨率域；（2）提出一种通过测量引导扩散过程趋向条件语句所需的工作量来获得边际似然的方法。我们首次实现了在 FlowGP 中进行超参数优化，并在以下任务上展示了我们的方法：基于区域汇总统计量的概率降尺度、不规则域上的偏微分方程（PDE）解推断，以及从非高斯卫星观测数据中恢复海平面异常场。

    arXiv:2610.11459v1 Announce Type: cross  Abstract: Outside the linear-Gaussian regime, conditional sampling from Gaussian processes (GPs) is challenging. Recent methods such as FlowGP (Moss et al., (2026)) can condition on arbitrary non-linear and non-Gaussian statements, but at considerable cost: an expensive iterative and high-dimensional diffusion that requires hand-specified kernel hyperparameters. In this paper, we alleviate two significant drawbacks of FlowGP by (1) introducing kernel approximations that enable scaling to high-resolution domains and (2) proposing a way to obtain the marginal likelihood by measuring the work needed to steer the diffusion towards conditioning statements. We enable, for the first time, hyperparameter optimisation within FlowGP and demonstrate our approach on probabilistic downscaling from areal summary statistics, PDE solution inference on irregular domains, and recovery of sea level anomaly fields from non-Gaussian satellite observations.
    
[^30]: 超越分布保真度：面向合成表格数据的因果惩罚扩散模型

    Beyond Distributional Fidelity: Causal-Penalized Diffusion for Synthetic Tabular Data

    [https://arxiv.org/abs/2610.11407](https://arxiv.org/abs/2610.11407)

    该论文首次将因果差异惩罚直接引入生成式表格扩散模型，提出因果惩罚化的 TabDDPM 训练框架，理论上证明高统计保真度不等于高因果保真度，并给出了因果正则化提升期望因果保真度的条件与实验验证。

    

    合成表格数据生成器通常以分布保真度为优化目标，但仅凭统计上的相似性并不能保证因果效应得以保留。本文研究了能否在完全生成式的表格模型中直接提升因果保真度。我们将“因果保真度”相对于目标估计量定义为：基于真实数据与合成数据所得到的推断分布之间的差异，并通过理论结果表明，高统计保真度通常并不意味着高因果保真度。随后，我们提出了一种因果保真度感知的训练框架，在生成目标函数中加入了因果差异惩罚项。该框架以因果惩罚化的 TabDDPM 为实例实现，并采用在策略得分函数估计器进行优化。我们进一步建立了因果正则化能够提升期望因果保真度的条件。在多种处理效应模拟及（摘要在此处截断）的实验中……

    arXiv:2610.11407v1 Announce Type: cross  Abstract: Synthetic tabular generators are commonly optimized for distributional fidelity, but statistical similarity alone does not guarantee preservation of causal effects. In this paper, we study whether causal fidelity can be improved directly within a fully generative tabular model. Causal Fidelity is defined with respect to a target estimand as the discrepancy between inferential distributions obtained from real and synthetic data, and theoretical results show that high statistical fidelity does not generally imply high causal fidelity. We then propose a causal-fidelity-aware training framework which adds a causal discrepancy penalty to the generative objective. The framework is instantiated with a causal-penalized TabDDPM and optimized using an on-policy score-function estimator. We further establish conditions under which causal regularization improves expected causal fidelity. Experiments across diverse treatment-effect simulations and 
    
[^31]: 基于机器学习模型的序贯条件独立性检验

    Sequential Conditional Independence Testing with Machine Learning Models

    [https://arxiv.org/abs/2610.11388](https://arxiv.org/abs/2610.11388)

    该论文通过将检验误差分解为零假设扩大误差、近似误差和估计误差，解释了直接检验可交换性的e-变量在实践中优于model-X框架下GRO e-变量的现象，并提出探索介于两者之间的中间零假设，从而将机器学习模型融入序贯条件独立性检验。

    

    条件独立性检验是科学发现中普遍存在的问题。广泛采用的model-X假设将建模负担从输出对输入的依赖性转移到输入内部的依赖关系上。在这一设定下，已有研究探讨了对数最优e-变量，但如何将机器学习模型融入其设计仍不明确。其他方法则直接检验可交换性，所得到的e-变量在理论上功效较低，但出人意料地在实践中功效更高。我们通过将误差分解为零假设扩大误差、近似误差和估计误差来解释这一现象。该分解表明，GRO e-变量估计可能被超越，原因在于其近似误差和估计误差更差；为此，我们探索了介于model-X条件独立性与可交换性之间的中间零假设，以降低这些误差。此外，model-X假设往往仅在一定程度上成立（原文摘要在此处截断）。

    arXiv:2610.11388v1 Announce Type: cross  Abstract: Conditional independence testing is a ubiquitous problem in scientific discovery. The widely employed model-X assumption shifts the modelling burden from the dependence of the output on the inputs to the dependencies within the inputs. Log-optimal e-variables have been studied in this setting, but it remains unclear how to incorporate machine learning models into their design. Other approaches test exchangeability directly, yielding an e-variable with lower power in theory but, surprisingly, higher power in practice. We explain this phenomenon by decomposing the error into null enlargement, approximation, and estimation error. The decomposition shows that GRO e-variable estimates can be beaten because of their worse approximation and estimation errors, and we explore intermediate null hypotheses between model-X conditional independence and exchangeability to reduce these errors. Moreover, the model-X assumption often only holds up to a
    
[^32]: 从几何到泛化：为什么行归一化能够胜过Adam和Muon

    From Geometry to Generalization: Why Row Normalization Can Beat Adam and Muon

    [https://arxiv.org/abs/2610.11309](https://arxiv.org/abs/2610.11309)

    该论文证明了在高维多分类任务中，行归一化凭借其类级欧几里得几何能够渐近保持总体决策边界方向，从而在总体精度上严格超越采用坐标级几何的Adam和采用谱几何的Muon等优化器。

    

    不同的优化器在拟合相同训练数据的同时，会选择出几何结构差异显著的分类器，但这种差异是否能被证明会影响总体性能仍不清楚。我们证明，在高维多分类任务中，按行归一化（row-wise normalization）可以获得严格高于全批量Adam（作为随机重排Adam的代理）以及精确SVD Muon的总体精度。在各向同性高斯云数据模型下，这一优势源于行归一化的类级欧几里得几何能够渐近地保持总体决策边界方向，而Adam的坐标级几何和Muon的谱几何则会引入不可忽略的失真。超越各向同性情形，对于在类均值上进行全批量训练、且类均值协方差与测试噪声协方差方向相互独立的情况，该优势依然成立；对于类均值指数小于1的幂律谱分布，即使在严重的（各向异性条件下）该优势同样保持……（摘要原文在此处截断）

    arXiv:2610.11309v1 Announce Type: cross  Abstract: Different optimizers can fit the same training data while selecting classifiers with substantially different geometries, but whether this difference provably affects population performance remains unclear. We show that row-wise normalization can achieve strictly higher population accuracy than full-batch Adam, a proxy for random-reshuffling Adam, and exact-SVD Muon in high-dimensional multiclass classification. Under an isotropic Gaussian-cloud data model, this advantage arises because row normalization's class-wise Euclidean geometry asymptotically preserves the population decision-boundary directions, whereas Adam's coordinate-wise geometry and Muon's spectral geometry introduce nonvanishing distortions. Beyond isotropy, the advantage persists for full-batch training on class means with independently oriented class-mean and test-noise covariances. It holds for power-law spectra with class-mean exponent below one, even under heavily a
    
[^33]: 基于图拉普拉斯基准化先验的正则化小区域估计

    Regularized Small Area Estimation with Graph Laplacian Benchmarking priors

    [https://arxiv.org/abs/2610.11266](https://arxiv.org/abs/2610.11266)

    提出了一类新的贝叶斯基准化先验族（含基于图拉普拉斯的正则化版本），可在小区域估计中同时实现跨区域信息借用与对可靠总体数据的基准化校准，并针对退化先验设计了基于降参数化的定制MCMC算法。

    

    小区域估计（SAE）通常既需要在不同区域之间借用信息，又需要将估计结果校准（基准化）到可靠的总体数据上。我们开发了一个贝叶斯框架，通过一类新的基准化先验同时实现这两个目标。该先验族由带基准约束的正则化问题导出，其中包括：一个基准化先验，它在不引入额外跨区域正则化的情况下融合基准化约束；以及单视图和多视图拉普拉斯基准化先验，它们通过基于外部协变量信息构建的区域相似性图拉普拉斯算子引入正则化。针对这些退化先验下的后验计算，我们基于约束空间的降参数化开发了定制的MCMC算法。我们通过基于数据的模拟评估了所提出的模型，并将该框架应用于估计平均家庭规模（AHS）……（摘要在此处截断）

    arXiv:2610.11266v1 Announce Type: cross  Abstract: Small area estimation (SAE) often requires both borrowing information across areas and benchmarking estimates to reliable aggregates. We develop a Bayesian framework that addresses these two objectives jointly through a new family of Benchmarking priors. The priors are induced by a benchmark-constrained regularization problem. The resulting family includes a Benchmarking Prior that incorporates the benchmarking restrictions without additional regularization across areas, and Single and Multi-View Laplacian Benchmarking Priors that introduce regularization through graph Laplacians constructed from area similarities based on external covariate information. For posterior computation under these degenerate priors, we develop tailored MCMC algorithms based on a reduced parameterization of the constraint space. We assess the proposed models using a data-based simulation and apply the framework to estimate Average Household Size (AHS) at the 
    
[^34]: 当更低的重构损失反而有害：面向低比特大语言模型量化的分布鲁棒精炼方法

    When Lower Reconstruction Loss Hurts: Distributionally Robust Refinement for Low-Bit LLM Quantization

    [https://arxiv.org/abs/2610.11226](https://arxiv.org/abs/2610.11226)

    本文发现更低的重构损失不一定带来更好的模型性能甚至可能有害，并提出分布鲁棒量化（DRQ）方法，通过在受约束的激活分布集合上最小化最坏情况重构损失来精炼量化权重编码，从而提升低比特大语言模型量化的效果。

    

    仅权重训练后量化（PTQ）在很大程度上依赖重构损失最小化来在低精度下保持模型质量。我们表明，通过最小化该损失所选择的权重不一定能在新任务上带来更好的模型性能。事实上，我们发现更低的重构损失甚至可能在相同的校准数据上降低模型性能。我们的分析进一步表明，在校准数据上重构损失更低的权重，当输入激活分布发生变化时，其损失可能高于其他权重。受这些观察和分析的启发，我们提出了分布鲁棒量化（DRQ），这是一种事后精炼过程，在受约束的输入激活分布集合上最小化最坏情况下的重构损失。DRQ在现有量化网格内对表示量化权重的整数编码进行精炼，同时保持量化参数和推理算子不变。

    arXiv:2610.11226v1 Announce Type: new  Abstract: Weight-only post-training quantization (PTQ) relies heavily on reconstruction loss minimization to preserve model quality at low precision. We show that the weights favored by minimizing this loss need not yield better model performance on new tasks. In fact, we find that lower reconstruction loss can even degrade model performance on the same calibration data. Our analysis further shows that weights with lower reconstruction loss on calibration data can have higher loss than other weights when the distribution of input activations changes. Motivated by these observations and our analysis, we propose Distributionally Robust Quantization (DRQ), a post-hoc refinement process that minimizes worst-case reconstruction loss over a constrained set of input activation distributions. DRQ refines the integer codes representing quantized weights within the existing quantization grid, keeping quantization parameters and inference operators unchanged
    
[^35]: 非自适应条件样本下等价性检验的紧致界

    Tight Bounds for Equivalence Testing with Non-Adaptive Conditional Samples

    [https://arxiv.org/abs/2610.11145](https://arxiv.org/abs/2610.11145)

    本文针对非自适应条件样本模型下的等价性检验问题，给出了匹配的算法与下界，证明其查询复杂度为 $\tilde \Theta(\log n/\varepsilon^2)$，并由此揭示均匀性、恒等性和等价性检验的复杂度均为 $\tilde \Theta(\log n)$。

    

    我们研究了具有非自适应条件样本访问权限的分布测试问题。具体而言，我们给出了等价性检验的紧致界，该问题旨在判定两个未知分布是相等的，还是在全变差距离下彼此相距至少 $\varepsilon$ 远。我们的算法和下界表明，解决该问题需要且仅需 $\tilde \Theta\left(\frac{\log n}{\varepsilon^2}\right)$ 次查询。这些结果表明，在非自适应条件样本模型下，均匀性检验、恒等性检验和等价性检验的复杂度均为 $\tilde \Theta(\log n)$。

    arXiv:2610.11145v1 Announce Type: cross  Abstract: We study distribution testing with access to non-adaptive conditional samples. Specifically, we give tight bounds for equivalence testing, determining whether two unknown distributions are equal to or $\varepsilon$-far from each other in total variation distance. Our algorithm and lower bound show that $\tilde \Theta\left(\frac{\log n}{\varepsilon^2}\right)$ queries are necessary and sufficient for this problem. These results demonstrate that the complexity of uniformity, identity, and equivalence testing with non-adaptive conditional samples are all $\tilde \Theta(\log n)$.
    
[^36]: 加速非光滑与重尾采样

    Accelerating Non-Smooth and Heavy-Tailed Sampling

    [https://arxiv.org/abs/2610.11139](https://arxiv.org/abs/2610.11139)

    本文提出非可逆锚定朗之万动力学（NALD）与非可逆反射锚定朗之万动力学（NRALD），通过引入循环漂移项，在无需目标密度导数的情况下加速对欧氏空间及受限域上非光滑、重尾目标分布的采样。

    

    锚定朗之万动力学（ALD）可用于非光滑采样场景，其中目标分布的密度可能不可微且呈重尾特性；反射锚定朗之万动力学（RALD）可以在受限域上对可能不可微的目标密度进行采样。本文提出并研究了非可逆锚定朗之万动力学（NALD），用于在欧几里得空间中对可能不可微且重尾的目标密度进行采样；以及非可逆反射锚定朗之万动力学（NRALD），用于在受限空间中对可能不可微的目标密度进行采样。我们的构造添加了一个由可能是状态依赖的无散度斜对称矩阵场和流势所产生的循环漂移。该构造无需目标密度的导数即可保持目标分布，具有随机时间变换表示，并且既适用于整个欧几里得空间，也适用于（受限空间）。

    arXiv:2610.11139v1 Announce Type: cross  Abstract: Anchored Langevin dynamics (ALD) is useful for non-smooth sampling where the density of the target distribution is possibly non-differentiable and heavy-tailed; reflected anchored Langevin dynamics (RALD) can sample possibly non-differentiable target density on a constrained domain. In this paper, we propose and study non-reversible anchored Langevin dynamics (NALD) for sampling possibly non-differentiable and heavy-tailed target density in the Euclidean space and the non-reversible reflected anchored Langevin dynamics (NRALD) for sampling possibly non-differentiable target density in the constrained space. Our construction adds a circulation drift generated by a possibly state-dependent divergence-free skew-symmetric matrix field and a stream potential. It preserves the target distribution without requiring derivatives of target density, admits a random-time-change representation, and applies both on the whole Euclidean space and on b
    
[^37]: 核赌博机问题的通用 $\widetilde{\Omega}(\sqrt{T \gamma_T})$ 下界

    A General $\widetilde{\Omega}(\sqrt{T \gamma_T})$ Lower Bound for Kernel Bandits

    [https://arxiv.org/abs/2610.11082](https://arxiv.org/abs/2610.11082)

    本文针对紧域上非常数连续核函数的核赌博机问题，建立了通用的 $\Omega(\sqrt{T\gamma_T/\log T})$ 极小极大遗憾下界，并证明该下界中的对数因子在一般情况下不可避免，从而在非常一般的意义上确立了现有 $\sqrt{T\gamma_T}$ 上界的接近最优性。

    

    核赌博机（kernel bandit）问题是指在有噪声反馈下顺序优化一个未知函数的问题，其中该函数在给定的再生核希尔伯特空间（RKHS）中具有有界范数。核赌博机遗憾分析中的一个核心量是最大信息增益 $\gamma_T$。特别地，现有的最佳上界在对数因子范围内以 $\sqrt{T\gamma_T}$ 的形式缩放，并且已经针对特定核函数（如平方指数核和Matérn核）推导出了几乎匹配的下界。然而，针对一般核函数的下界仍然缺失，这使得现有上界在何种一般性下接近最优尚不清楚。在本文中，我们针对紧域上的非常数连续核函数建立了一个通用的 $\Omega(\sqrt{T\gamma_T/\log T})$ 极小极大遗憾下界，从而在非常一般的意义上确立了上界的接近最优性（在对数因子范围内）。我们证明该下界中出现的对数因子在一般情况下是不可避免的，但……

    arXiv:2610.11082v1 Announce Type: cross  Abstract: The kernel bandit problem consists of sequentially optimizing an unknown function with noisy feedback, where the function has bounded norm in a given Reproducing Kernel Hilbert Space (RKHS). A central quantity in the regret analysis of kernel bandits is the maximum information gain $\gamma_T$. In particular, the best existing upper bounds scale as $\sqrt{T\gamma_T}$ up to log factors, and nearly-matching lower bounds have been derived for specific kernels such as squared exponential and Mat\'ern. However, lower bounds for general kernels are lacking, thus making it unclear in what generality the upper bounds are near-optimal. In this paper, we establish a general $\Omega(\sqrt{T\gamma_T/\log T})$ minimax regret lower bound for non-constant continuous kernels on compact domains, establishing near-optimality (within log factors) in a very general sense. We show that the log factor appearing in this bound is unavoidable in general, but th
    
[^38]: 决策充分的后验近似

    Decision-Sufficient Posterior Approximation

    [https://arxiv.org/abs/2610.11038](https://arxiv.org/abs/2610.11038)

    该论文提出“决策充分”的后验近似框架，通过在贝叶斯动作纤维上收缩KL散度精确刻画后验近似何时会改变下游决策，并利用目标遗憾Hessian与基线信息度量的广义特征值问题，按单位信息成本的遗憾后果对决策方向排序，从而找出跨越决策边界所需的最小信息后验形变。

    

    我们研究了要求后验近似保持特定下游决策问题时所产生的后果。目标后验 $P$ 与损失函数在动作空间上确定了一个遗憾几何结构，基线近似 $Q_0$ 确定了诱导动作改变所需的前向KL散度信息量，而受限的近似族 $\mathcal{Q}$ 则决定了哪些此类改变是可行的。通过将KL散度在贝叶斯动作纤维上进行收缩，可以得到到“决策充分”与“决策失败”的精确距离，以及到达决策边界任一侧的信息量最小的后验形变。在正则的有限维问题中，目标与基线构造具有二次局部极限：即目标遗憾Hessian矩阵 $G$ 与基线信息度量 $J_I$。它们的广义特征值问题 $Gv = \gamma J_I v$ 依据单位信息成本的遗憾后果对局部决策方向进行排序，并……（原文摘要在此处截断）。

    arXiv:2610.11038v1 Announce Type: cross  Abstract: We investigate the consequences of requiring a posterior approximation to preserve a specified downstream decision problem. A target posterior $P$ and loss determine a regret geometry on actions, a baseline approximation $Q_0$ determines the forward-Kullback-Leibler information required to induce action changes, and a restricted approximation family $\mathcal{Q}$ determines which such changes are available. Contracting KL divergence over Bayes-action fibers gives exact distances to decision adequacy and decision failure together with the least-informative posterior deformations that reach either side of the decision boundary. In regular finite-dimensional problems, the target and baseline constructions have quadratic local limits: a target regret Hessian $G$ and a baseline information metric $J_I$ . Their generalized eigenproblem $Gv = \gamma J_Iv$ orders local decision directions by regret consequence per unit information cost and ind
    
[^39]: 小数据脑解码只需 SPD-MetaFormer

    SPD-MetaFormer is what you need for small-data brain decoding

    [https://arxiv.org/abs/2610.10952](https://arxiv.org/abs/2610.10952)

    该研究发现基于SPD流形的注意力模型学到的注意力权重接近均匀、可被简单均匀权重替代而几乎不损失预测性能，据此表明架构结构比学习加权更关键，提出SPD-MetaFormer足以胜任小数据脑解码任务。

    

    脑信号解码极具挑战性，因为神经记录信号噪声大且因人而异，而带标签的数据往往十分有限。尽管如此，近年来在对称正定（SPD）流形上的注意力模型利用协方差和功能连接表示已取得了出色的性能，但其中学习到的词元加权所起的作用仍不清楚。我们研究了两种代表性架构——基于 log-Euclidean 几何的 MAtt 和基于广义 Bures–Wasserstein 几何的 GBWAtt——发现它们训练后学到的注意力权重仍然接近均匀分布。我们将这一现象归因于有界的相似度参数化：在原始的 softmax 缩放方式下，这种参数化限制了注意力权重之间的对比度。此外，无论是在训练还是评估阶段，用均匀权重替换学习到的权重，对平均预测性能几乎没有影响，同时还能保留各模型原有的聚合几何特性……

    arXiv:2610.10952v1 Announce Type: new  Abstract: Brain signal decoding is challenging because neural recordings are noisy and vary across individuals, while labeled data are often limited. Recent attention-based models on the symmetric positive definite (SPD) manifold have nevertheless achieved strong performance using covariance and connectivity representations, yet the contribution of learned token weighting remains unclear. We examine two representative architectures, MAtt (based on log-Euclidean geometry) and GBWAtt (based on generalized Bures--Wasserstein geometry), and find that their learned attention weights remain close to uniform after training. We relate this behavior to bounded similarity parameterizations that, under the original softmax scaling, limit attention-weight contrast. Moreover, replacing learned weights with uniform weights, throughout training and evaluation, has little effect on mean predictive performance while preserving each model's original aggregation geo
    
[^40]: 具有方差缩减的变换采样器

    Transformed Samplers with Variance Reduction

    [https://arxiv.org/abs/2610.10870](https://arxiv.org/abs/2610.10870)

    该论文提出通过学习双射（如归一化流）将MCMC采样器变换到潜空间，把简单参考分布上的泊松方程精确解推广到一般目标分布，从而获得显式控制变量以实现方差缩减。

    

    马尔可夫链蒙特卡罗（MCMC）方法是在复杂概率分布下计算期望的标准工具。控制变量可以降低估计结果的方差，但一个好的控制变量需要求解采样器的泊松方程，而该方程很少存在闭式解。当采样器的核在某个简单参考密度上具有已知的谱分解时，可以获得精确解。在我们的工作中，我们通过学习到的变量变换将这些解推广到一般的目标分布。我们训练一个双射（例如归一化流），使目标分布在潜空间中接近参考分布，并证明了马尔可夫核及其泊松解可以被任意双射所变换。在潜空间中运行此类采样器即可得到显式的控制变量，并且在该映射和目标分布的温和尾部条件下，估计量具有一致性。基于……的重要性采样（IS）

    arXiv:2610.10870v1 Announce Type: cross  Abstract: Markov chain Monte Carlo (MCMC) methods are the standard tool for computing expectations under complex probability distributions. Control variates reduce the variance of the resulting estimates, but a good control variate requires solving the Poisson equation of the sampler, which rarely admits a closed-form solution. Exact solutions are available when the sampler's kernel has a known spectral decomposition on a simple reference density. In our work, we extend these solutions to general targets through a learned change of variables. A bijection, such as a normalizing flow, is trained so that the target becomes close to the reference in a latent space, and we show that Markov kernels and their Poisson solutions are transformed by any bijection. Running such samplers in the latent space then yields explicit control variates, and the estimator is consistent under mild tail conditions on the map and target. Importance sampling (IS) from th
    
[^41]: 部分验证下的共形预测

    Conformal Prediction under Partial Verification

    [https://arxiv.org/abs/2610.10829](https://arxiv.org/abs/2610.10829)

    该论文提出了一种部分验证方法，通过刻画校准证书并在校准样本间协调验证，在产生与完整验证完全相同的预测集的同时，将验证成本降低15-82%。

    

    共形预测能够提供具有有限样本保证的预测集，但校准所需的标签验证可能代价高昂。我们开发了一种部分验证方法，其返回的预测集与完整验证的结果完全相同。我们刻画了校准证书——即足以确定共形阈值的已验证信息——并设计了一个在校准样本之间协调验证的流程。对于高覆盖率下的有限阈值，当按顺序检查候选样本时，该方法的验证成本低于最小证书成本的两倍。在检索、数学解答和配置评估等任务中，与逐个验证校准样本相比，该方法将验证成本降低了15-82%，同时产生完全相同的预测集。

    arXiv:2610.10829v1 Announce Type: cross  Abstract: Conformal prediction provides prediction sets with finite-sample guarantees, but the label verification required for calibration can be expensive. We develop a partial verification method that returns exactly the same prediction sets as complete verification. We characterize calibration certificates, the verified information sufficient to determine the conformal threshold, and design a procedure that coordinates verification across calibration examples. For finite thresholds at high coverage, its verification cost is less than twice the minimum certificate cost when candidates are checked in order. Across retrieval, mathematical solutions, and configuration evaluation, it reduces verification cost by 15-82% compared with verifying calibration examples one at a time, while producing identical prediction sets.
    
[^42]: 通过诊断性传输校准模糊集以用于分布鲁棒优化

    Calibrating Ambiguity Set via Diagnostic Transport for Distributionally Robust Optimization

    [https://arxiv.org/abs/2610.10793](https://arxiv.org/abs/2610.10793)

    本文提出诊断传输DRO（DT-DRO），利用留出校准数据和条件概率积分变换来诊断预测误差，自适应地调整模糊集的中心与几何结构，从而在保证决策风险可控的同时避免DRO决策过度保守。

    

    分布鲁棒优化（DRO）通过在模糊集上进行优化来保护决策免受分布不确定性的影响，但当模糊集的几何结构与问题不匹配时，往往需要设置较大的半径，从而导致决策过于保守。我们提出了诊断传输DRO（DT-DRO），它利用留出的校准数据使模糊集的几何结构适应观测到的预测误差。DT-DRO使用条件概率积分变换累积分布函数来诊断系统性的概率错配，并将该信息转化为结果层面的传输，从而联合调整模糊集的中心和基础成本。由此得到的公式具有计算上易于处理的对偶重构形式。在理论方面，我们推导了有效的模糊半径和决策风险保证，这些保证会随着估计误差和近似误差的消失而收紧，并证明DT-DRO能够消除由模型误设引起的无法消除的鲁棒性下限（摘要在此处被截断）。

    arXiv:2610.10793v1 Announce Type: cross  Abstract: Distributionally robust optimization (DRO) protects decisions against distributional uncertainty by optimizing over an ambiguity set, but poorly aligned set geometry can require large radii and yield overly conservative decisions. We introduce diagnostic-transport DRO (DT-DRO), which uses held-out calibration data to adapt the ambiguity-set geometry to observed predictive errors. DT-DRO uses the conditional probability integral transform cumulative distribution function to diagnose systematic probability misallocation and translates this information into an outcome-level transport that jointly adjusts the ambiguity-set center and ground cost. The resulting formulation admits a computationally tractable dual reformulation. Theoretically, we derive valid ambiguity radii and decision-risk guarantees that tighten as estimation and approximation errors vanish, and show that DT-DRO can eliminate the nonvanishing robustness floor caused by mo
    
[^43]: 线性注意力在上下文中能从非线性教师身上学到什么？

    What can linear attention learn from nonlinear teachers in-context?

    [https://arxiv.org/abs/2610.10761](https://arxiv.org/abs/2610.10761)

    本文的核心创新是建立了“非线性-噪声等价性”理论：线性注意力在上下文学习中只提取目标函数的线性Hermite分量，剩余非线性结构等价于有效噪声，从而使线性理论的结论可以迁移到非线性任务。

    

    线性注意力是理解Transformer中上下文学习（in-context learning）机制的一个可解析模型。针对线性回归任务，近期的渐近分析已刻画了其学习与泛化行为。我们将这一理论扩展到非线性单指标目标 $y=f(x^\top w)+\varepsilon$。我们的主要结果建立了非线性-噪声等价性：线性注意力只能提取 $f$ 的线性Hermite分量，而剩余的非线性结构则作为有效噪声贡献到泛化误差中。这一简化使得相应线性理论的结论能够迁移到非线性任务上。我们阐明了该结论对有限预训练数据场景，以及随任务多样性增加从任务记忆到任务泛化的转变所带来的启示。这些结果指出了简化的线性注意力模型的一个局限性，并为研究非线性上下文学习提供了一个可解析的起点。

    arXiv:2610.10761v1 Announce Type: cross  Abstract: Linear attention is a tractable model for understanding the mechanisms governing in-context learning in transformers. For linear regression tasks, recent asymptotic analyses have characterised its learning and generalisation behaviour. We extend this theory to nonlinear single-index targets, $y=f(x^\top w)+\varepsilon $. Our main result establishes a nonlinearity-noise equivalence: linear attention extracts only the linear Hermite component of $f$, while the remaining nonlinear structure contributes to the generalisation error as effective noise. This reduction allows results from the corresponding linear theory to be transferred to nonlinear tasks. We illustrate its implications for finite pretraining data and for the transition from task memorisation to task generalisation as task diversity increases. These results identify a limitation of the reduced linear-attention model and provide a tractable starting point for studying nonlinea
    
[^44]: 解释对抗训练神经网络的显著性图稀疏性

    Explaining the Saliency Map Sparsity of Adversarially-Trained Neural Networks

    [https://arxiv.org/abs/2610.10666](https://arxiv.org/abs/2610.10666)

    本文首次为对抗训练神经网络梯度显著性图的稀疏性现象提供了理论解释，证明随着数据点和神经元数量增长，网络的最小化解收敛到具有最小梯度和Barron范数的贝叶斯分类器，从而自然产生了稀疏性。

    

    理解深度神经网络为何做出某个特定预测，对于其安全部署至关重要。在计算机视觉领域，显著性图通过突出显示对预测最具影响力的图像区域，仍然是一种被广泛使用的解释形式。一个经验性观察是，对抗训练神经网络的梯度显著性图呈现出明显的稀疏性。本文针对两层ReLU网络对这一现象提出了理论解释。我们建立在已确立的等价性之上——即对抗训练等价于在经验风险最小化中加入权重衰减惩罚项以及一个额外的对抗全变差项（该等价性对某些损失函数成立）。随着数据点和神经元数量的增长，且正则化参数以适当的速率趋于零，我们证明了最小化解收敛到具有最小梯度和Barron范数的贝叶斯分类器。稀疏性的出现是因为，对于对抗训练……

    arXiv:2610.10666v1 Announce Type: new  Abstract: Understanding why deep neural networks make a given prediction is of great importance for their safe deployment. In computer vision, saliency maps, which highlight the image region most influential for a prediction, remain a widely-used form of explanation. An empirical observation is the apparent sparsity of gradient saliency maps of adversarially-trained neural networks. In this paper, we propose a theoretical explanation of this phenomenon for two-layer ReLU networks. We build on the established equivalence of adversarial training to the minimization of the empirical risk with weight-decay penalization and an added adversarial total variation term -- valid for certain loss functions. As the number of data points and neurons grows and the regularization parameters are sent to zero at appropriate rates, we prove that minimizers converge to a Bayes classifier with minimal gradient and Barron norm. Sparsity appears since for adversarial t
    
[^45]: 从对数几率到Shapley值：加权朴素贝叶斯分类器的解释性几何

    From Log-Odds to Shapley Values: An Explanatory Geometry for the Weighted Naive Bayes Classifier

    [https://arxiv.org/abs/2610.10642](https://arxiv.org/abs/2610.10642)

    该论文证明加权朴素贝叶斯分类器中基于对数几率的监督距离与解析Shapley值向量之间的ℓ1距离完全一致，从而为监督距离、局部解释与预测行为之间建立了形式化的联系。

    

    本文研究了如何从加权朴素贝叶斯分类器所诱导的监督表示出发，构建一个解释性空间。我们从基于条件对数似然的经典监督距离出发，引入了一种基于对数几率的判别性重构，该重构与分类决策的关系更为直接。随后我们证明，这种表示所诱导的距离恰好与解析Shapley值向量之间的 $\ell_1$ 距离完全一致，从而为模型所诱导的几何结构提供了形式化的解释性诠释。最后，我们使用 $k$ 近邻分类器，对从这些表示中导出的若干监督距离进行了实证比较。本工作从方法论视角出发，凸显了监督距离、局部解释与预测行为之间的紧密联系。

    arXiv:2610.10642v1 Announce Type: cross  Abstract: This paper studies the construction of an explanatory space for a weighted naive Bayes classifier from the supervised representation induced by the model. We start from the classical supervised distance based on conditional log-likelihoods and introduce a discriminative reformulation based on log-odds, which is more directly related to the classification decision. We then show that this representation induces a distance that exactly coincides with the $\ell_1$ distance between vectors of analytical Shapley values, thereby providing a formal explanatory interpretation of the geometry induced by the model. Finally, we empirically compare several supervised distances derived from these representations using a $k$-nearest neighbors classifier. This work highlights a close link between supervised distance, local explanation, and predictive behavior, from a primarily methodological perspective.
    
[^46]: 截断扩散采样器需要保留多少个方向？幂律谱下的匹配界

    How Many Directions Must a Truncated Diffusion Sampler Retain? Matching Bounds Under Power-Law Spectra

    [https://arxiv.org/abs/2610.10640](https://arxiv.org/abs/2610.10640)

    针对幂律协方差谱数据，论文证明了截断扩散采样器所需保留方向数的匹配上下界，并揭示仅保留信号超过噪声水平方向的策略仍会产生不可忽略的总体截断误差。

    

    扩散采样器可以通过仅生成选定的谱坐标、并用噪声填充其余方向来减少计算量。那么它们必须保留多少个方向？我们针对具有幂律协方差谱的数据研究了这一问题。对于高斯数据与平滑目标的比较，在环境维度足够大的前提下，我们证明了所需保留方向数的匹配界。截断误差取决于被省略方向的组合维纳增益，而与采样器在保留坐标上的精度无关。因此，仅保留信号强度超过输出噪声水平的方向仍可能留下不可忽略的误差：许多单独较弱的方向在总体上依然显著。将这一刻画与扩散收敛界相结合，我们得到了在精确分数条件下的充分采样步复杂度。上界还可推广到基于估计主成分的情形。

    arXiv:2610.10640v1 Announce Type: cross  Abstract: Diffusion samplers can reduce computation by generating selected spectral coordinates and filling the remaining directions with noise. How many directions must they retain? We study this question for data with power-law covariance spectra. For Gaussian data compared to a smoothed target, we prove matching bounds on the required number of retained directions, provided that the ambient dimension is sufficiently large. The truncation error depends on the combined Wiener gains of the omitted directions, regardless of the accuracy of the sampler on the retained coordinates. Keeping only directions whose signal exceeds the output noise level can therefore leave a non-vanishing error: many individually weak directions remain significant in aggregate. Combining this characterization with a diffusion convergence bound yields sufficient sampling-step complexity under exact scores. The upper bounds also extend to estimated principal components an
    
[^47]: D-SLR：不相交行稀疏加低秩分解

    D-SLR: The Disjoint Row-Sparse plus Low-Rank Decomposition

    [https://arxiv.org/abs/2610.10636](https://arxiv.org/abs/2610.10636)

    本文提出D-SLR分解，一种截断SVD的闭式直接替代方法，它将矩阵行不相交地划分为逐字存储或低秩近似两类，在平方误差下以更少参数即可达到联合最优，且在相同代价下不会差于截断SVD。

    

    用于重构的矩阵压缩目前仍默认采用截断SVD，即用单一低秩结构来近似数据。通常的做法是再添加一个重叠的行稀疏分量来进一步减小残差，但求解这一联合问题的方法往往需要迭代求解器和正则化参数调优。我们提出不相交行稀疏加低秩（D-SLR）分解，它是截断SVD的一种闭式直接替代方案，能够改进或恰好匹配截断SVD的效果。D-SLR将矩阵的每一行限制为要么被逐字存储，要么由低秩拟合来近似，绝不同时兼有。在平方误差准则下，这种限制没有任何代价：在每一个非平凡的秩和存储行数（形状）下，联合最优解都可以通过不相交的方式以更少的参数达到。当存储行数为零时，D-SLR退化为截断SVD，因此在相同代价下它永远不会表现更差。该算法对整个误差与参数之间的权衡进行评分，且解是c……（摘要在此处被截断）

    arXiv:2610.10636v1 Announce Type: new  Abstract: Compressing a matrix for reconstruction still defaults to the truncated SVD, approximating the data with a single low-rank structure. It is common to reduce the residual further by adding an overlapping row-sparse component, but methods that solve this joint problem often require iterative solvers and tuning of regularization parameters. We propose the Disjoint Row-Sparse plus Low-Rank (D-SLR) decomposition, a closed-form drop-in for the truncated SVD that improves or exactly matches it. D-SLR restricts rows to either being stored verbatim or approximated by the low-rank fit, never both. Under squared error this restriction costs nothing: the joint optimum is attainable disjointly with fewer parameters at every non-trivial rank and stored row count (shape). With zero stored rows D-SLR reduces to the truncated SVD, so it never does worse at equal cost. The algorithm scores the entire error-versus-parameters tradeoff, and the solution is c
    
[^48]: 面向旋转鲁棒神经动力学的精确SO(3)等变各向同性核算子

    Exact SO(3)-Equivariant Isotropic Kernels for Rotation-Robust Neural Dynamics

    [https://arxiv.org/abs/2610.10626](https://arxiv.org/abs/2610.10626)

    本文提出不变量条件化各向同性核神经算子（IKNO），通过仅使用旋转不变标量与随数据共同旋转的向量方向来构建局部相互作用，使三维Navier–Stokes等向量值偏微分方程的神经代理模型实现数值精度上的精确SO(3)等变性，彻底消除无约束图模型无法根除的坐标依赖问题。

    

    面向向量值偏微分方程的神经代理模型虽然能很好地拟合训练数据，但当同一物理状态在旋转后的坐标系中表达时，其预测结果却会发生变化。我们在不规则采样点上观测的三维Navier–Stokes动力学中研究了这种失效现象。我们提出了不变量条件化的各向同性核神经算子（IKNO），这是一种紧凑的图模型，它利用不受旋转影响的标量量和随数据一同旋转的向量方向来构建局部相互作用。因此，当对位置和速度进行旋转时，模型预测的速度变化会以完全相同的方式旋转。在一个在模型设计之后固定下来的保留测试集上，使用随机旋转的样本训练无约束图模型只能减少但不能消除其坐标依赖性。相比之下，IKNO在数值精度上保持一致，其预测精度可与通用的旋转感知张量场网络相媲美。

    arXiv:2610.10626v1 Announce Type: new  Abstract: Neural surrogates for vector-valued partial differential equations can fit training data yet change their predictions when the same physical state is expressed in a rotated coordinate frame. We study this failure on three-dimensional Navier--Stokes dynamics observed at irregularly placed points. We introduce the Invariant-Conditioned Isotropic Kernel Neural Operator (IKNO), a compact graph model that builds local interactions from scalar quantities unchanged by rotation and vector directions that rotate with the data. Consequently, rotating the positions and velocities rotates the predicted velocity change in exactly the same way. On a held-out test set fixed after model design, training unconstrained graph models on randomly rotated examples reduces but does not eliminate their coordinate dependence. In contrast, IKNO is consistent to numerical precision, matches the forecasting accuracy of a general rotation-aware Tensor Field Network 
    
[^49]: JevForest：面向预算受限特征获取的路径投票方法

    JevForest: Path Voting for Budgeted Feature Acquisition

    [https://arxiv.org/abs/2610.10615](https://arxiv.org/abs/2610.10615)

    提出JevForest特征获取策略，通过聚合自助采样树的路径提议、以全局信息增益加权并用共享掩码分类器预测，在有限观测预算下自适应选择最值得查询的特征，但其实验效果在不同数据集上并不一致。

    

    在有限的观测预算下，选择观测哪些信息是预测问题的核心。我们研究了JevForest，这是一种特征获取策略，它聚合来自自助采样树的路径依赖提议，根据全局训练信息增益对其进行加权，并利用共享的掩码分类器基于获取的特征值进行预测。一个在线实现通过该策略选择语义答案来查询Jev。在小型平衡的留出样本上，四问森林获取策略在AG News数据集（n=48）上达到0.729的准确率，而静态增益排序为0.667，随机排序为0.583。在TREC数据集（n=24）上，排序发生逆转：森林准确率为0.667，而静态增益排序为0.750，随机排序为0.833。一次性批量询问全部八个问题比四次顺序森林查询以更低的测量成本和延迟获得更高的准确率；直接Jev分类以更低的成本达到与批量查询相同的准确率。离线MiniBooNE实验……

    arXiv:2610.10615v1 Announce Type: cross  Abstract: Choosing which information to observe is central to prediction under limited observation budgets. We study JevForest, a feature acquisition policy that aggregates path-dependent proposals from bootstrapped trees, weights them by global training information gain, and predicts from the acquired values with a shared masked classifier. An online implementation queries Jev for semantic answers selected by this policy. On small balanced held-out samples, four-question forest acquisition achieves accuracy $0.729$ on AG News ($n=48$), compared with $0.667$ for a static gain ranking and $0.583$ for random ordering. On TREC ($n=24$), the ordering reverses: forest accuracy is $0.667$, compared with $0.750$ and $0.833$. Asking all eight questions in one batch yields higher accuracy at lower measured cost and latency than four sequential forest queries; direct Jev classification matches the batch accuracy while costing less. Offline MiniBooNE exper
    
[^50]: VC学习的最优信息复杂度

    The optimal information complexity of VC learning

    [https://arxiv.org/abs/2610.10600](https://arxiv.org/abs/2610.10600)

    本文通过构造一个eCMI为O(d)阶的随机化“5个基学习器多数投票”学习算法，首次经由CMI信息论分析框架恢复了VC类学习的最优PAC泛化保证。

    

    Steinke和Zakynthinou（2020）提出了条件互信息（CMI）框架，利用依赖于算法的信息论量来分析学习算法的信息复杂度。我们研究了其中一种量——评估条件互信息。一个有趣的问题是：能否通过基于CMI的依赖于算法的分析，恢复VC类的最优PAC保证？我们证明了这是可能的：在可实现情形下，我们构造了一个eCMI为O(d)阶的学习算法，从而恢复了该最优保证，其中d是概念类的VC维。特别地，我们的算法是一种随机化的“5个基学习器多数投票”算法，并具有最优的期望泛化保证。

    arXiv:2610.10600v1 Announce Type: cross  Abstract: Steinke and Zakynthinou(2020) introduces the Conditional Mutual Information (CMI) framework of analyzing the information complexity of learning algorithms based on algorithm-dependent information-theoretic quantities. We study one of these quantities, the evaluated Conditional Mutual Information (eCMI). It has been an interesting question whether the optimal PAC guarantee for VC classes can be recovered from the algorithm-dependent analyses via CMI. And we show that it is possible to recover this guarantee by constructing a learning algorithm whose eCMI is of order O(d) in the realizable case, where d is the VC-dimension of the concept class. Specially, our algorithm is a randomized Majority-of-5 base learners with optimal in-expectation generalization guarantee.
    
[^51]: 损失差条件互信息的精度—信息权衡

    An Accuracy--Information Tradeoff for Loss-Difference Conditional Mutual Information

    [https://arxiv.org/abs/2610.09206](https://arxiv.org/abs/2610.09206)

    论文证明了精度与信息之间的权衡：在逻辑损失等光滑凸损失及幂次正则化条件下，任何以最优样本量达到低超额风险的正规学习器，其最坏情况损失差条件互信息必然达到 n 比特量级。

    

    损失差条件互信息（ld-CMI）是泛化上界的超样本层次结构中最小的标准观测量：它衡量学习器的损失差在多大程度上泄露了它训练时使用的是每一对候选样本中的哪一个。已知精度会迫使信息进入模型；而数据处理不等式并不能将此类下界传递到损失上。我们通过对损失差的三个矩进行约束，证明了精度同样会迫使ld-CMI。对于使用在零点斜率非零的光滑凸损失（如逻辑损失）的线性预测器，加上曲率和增长均呈幂次 r≥2 的正则化器，在维度至少随 n 线性增长的缩放符号立方体上的乘积分布中，每个在最优样本量 n≍ε^(-2+2/r) 下于这些分布上期望超额风险至多为 ε 的正规学习器，其最坏情况ld-CMI达到 n 比特量级，且为 Θ(n/(1+(τ/φ…

    arXiv:2610.09206v1 Announce Type: new  Abstract: Loss-difference conditional mutual information (ld-CMI) uses the smallest of the standard observations in the supersample hierarchy of generalization bounds: it measures what a learner's loss differences reveal about which candidate of each pair it was trained on. Accuracy is known to force information into the model; data processing does not carry such lower bounds to losses. We show, by bounding three moments of the loss differences, that accuracy also forces ld-CMI. For linear predictors with a smooth convex loss of nonzero slope at zero, such as the logistic loss, plus a regularizer whose curvature and growth are both of power $r\ge2$, on product distributions over a scaled sign cube in dimension at least linear in $n$, every proper learner with expected excess risk at most $\varepsilon$ on these distributions at the optimal sample size $n\asymp\varepsilon^{-2+2/r}$ has worst-case ld-CMI of order $n$ bits, and $\Theta(n/(1+(\tau/\var
    
[^52]: 只为FUNS：基于大语言模型引导的时空图节点生成方法用于预测未观测节点状态

    Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States

    [https://arxiv.org/abs/2610.08818](https://arxiv.org/abs/2610.08818)

    该论文提出GenST框架，将未观测节点状态预测（FUNS）重新定义为时空图上的条件生成任务，创新性地利用微调后的大语言模型从节点描述中提取语义特征作为语义桥梁，以弥补缺失的时空信号。

    

    时空预测是物流、城市规划和智能交通系统的基石。然而，受部署成本和维护资源的限制，传感器网络往往缺乏全面的空间覆盖，这使得预测未观测节点状态（FUNS）成为一项至关重要却又极具挑战性的任务。传统模型依赖历史观测数据，在遇到没有先前记录的节点时通常会表现失常。为解决这一问题，我们将该问题重新定义为时空图上的条件生成任务，并提出GenST框架，该框架引入大语言模型（LLMs）作为语义桥梁，利用经过微调的预训练LLM从节点描述（如功能分区和道路网络结构）中提取丰富的语义特征，以弥补缺失的时空信号。具体而言，我们设计了一个两阶段生成架构：时空变分自编码器（VAE）首先压缩……

    arXiv:2610.08818v1 Announce Type: cross  Abstract: Spatio-temporal forecasting is a cornerstone of logistics, urban planning, and intelligent transportation systems. However, constrained by deployment costs and maintenance resources, sensor networks often lack comprehensive spatial coverage, rendering Forecast Unobserved Node States (FUNS) a critical yet formidable challenge. Conventional models rely on historical observations and typically falter when encountering nodes without prior records. To address this, we redefine the problem as a conditional generation task on spatio-temporal graphs and propose GenST, a framework that introduces Large Language Models (LLMs) as a semantic bridge, leveraging a pre-trained LLM fine-tuned to extract rich semantic features from node descriptions, such as functional zones and road network structures, to compensate for missing spatio-temporal signals. Specifically, we design a two-stage generative architecture: a Spatio-Temporal VAE first compresses 
    
[^53]: 通过重采样的一次性差分隐私置信区域

    One-Shot Private Confidence Regions via Resampling

    [https://arxiv.org/abs/2610.08460](https://arxiv.org/abs/2610.08460)

    提出一个一次性构建差分隐私置信区域的简单框架，仅对最终重采样分位数加噪，使隐私代价在有放回（m-out-of-n）采样下仅为对数级、在无放回（子）采样下与重采样次数 B 无关，避免了以往方法中的 √B 因子，并给出了非渐近的高斯差分隐私与效用保证。

    

    我们提出了一个简单的框架，用于“一次性”构建差分隐私置信区域，即仅对最终的重采样分位数添加噪声，而不是对每次重采样计算的估计器进行隐私化处理。我们的方法中隐私的代价在有放回采样（$m$-out-of-$n$ 采样）下仅与重采样次数 $B$ 呈对数关系，在无放回采样（子采样）下与 $B$ 无关，从而避免了以往工作中出现的 $\sqrt{B}$ 因子。我们为子采样和 $m$-out-of-$n$ 重采样提供了非渐近的高斯差分隐私（GDP）保证和效用保证，涵盖了具有较小全局敏感度的均值类估计器，以及具有可高效计算的光滑敏感度上界的估计器，包括分位数和退化U统计量。这使我们还能为退化U统计量获得隐私置信区域，其隐私误差要小得多。

    arXiv:2610.08460v1 Announce Type: new  Abstract: We propose a simple framework for constructing differentially private confidence regions \textit{in one shot}, i.e., by adding noise only to the final resampling quantile instead of privatizing the estimator computed on each resample. The cost of privacy of our procedure is only logarithmic in the number of resamples $B$ under with-replacement ($m$-out-of-$n$) sampling and independent of $B$ under without replacement sampling (subsampling), avoiding the $\sqrt{B}$ factor that arises in previous works. We provide nonasymptotic Gaussian Differential Privacy (GDP) and utility guarantees for both subsampling and $m$-out-of-$n$ resampling, covering mean-like estimators with small global sensitivity as well as estimators admitting efficiently computable smooth sensitivity bounds, including quantiles and degenerate U-statistics. This allows us to also obtain private confidence regions for degenerate U-statistics where the private error is much 
    
[^54]: SOL：利用双切片Wasserstein度量衡量文本分布之间的差距

    SOL: Measuring Gaps between Text Distributions by Double Sliced Wasserstein Metrics

    [https://arxiv.org/abs/2610.06513](https://arxiv.org/abs/2610.06513)

    提出SOL——一种基于固定Transformer隐藏状态经验测度的双切片Wasserstein距离的文本分布距离度量，当Transformer为单射时可证明其为真正的度量，为非自回归语言模型的分布拟合评估提供了稳定的样本级评估方案。

    

    评估文本生成需要衡量生成分布与数据分布的匹配程度。对于自回归模型，这通过困惑度来实现；而扩散模型和基于流的语言模型只能提供似然界，其紧密程度在不同模型家族之间存在差异。基于样本的替代方法（如结合熵的生成式困惑度）则未考虑分布拟合情况。我们提出了SOL，一种文本分布之间的距离度量。每个序列由其在固定Transformer下隐藏状态的经验测度来表示，并通过双切片Wasserstein距离来比较这些测度的分布。我们证明，当Transformer是单射时，SOL是一个真正的度量。实验表明，SOL能够检测分布性失败、恢复预期的模型趋势，并提供稳定的基于样本的估计。我们提出SOL以填补当前非自回归模型评估协议中的空白。

    arXiv:2610.06513v2 Announce Type: replace  Abstract: Evaluating text generation requires measuring how well the generated distribution matches the data distribution. For autoregressive models, this is done by the perplexity. Diffusion and flow-based language models can only provide a likelihood bound, whose tightness differs between model families. Sample-based substitutes such as generative perplexity with entropy do not consider the distribution fit. We propose SOL,   a distance between text distributions. Each sequence is represented by the empirical measure of its hidden states under a fixed transformer and the distributions of these measures are compared by the double sliced Wasserstein distance. We prove that SOL is a metric if the transformer is injective. Experiments show that SOL detects distributional failures, recovers expected model trends, and provides stable sample-based estimates. We put forward SOL to fill the gap in the current evaluation protocol used for non auto-reg
    
[^55]: 在可容许性匹配条件下度量学习型单调时间聚合

    Measuring Learned Monotone Temporal Aggregation at Matched Admissibility

    [https://arxiv.org/abs/2610.05196](https://arxiv.org/abs/2610.05196)

    该论文在对比双方单调可容许性完全匹配的前提下，用构造上单调的循环网络（EWMA与高水位标记的可学习变换）度量学习型时间聚合的价值，并通过函数回归揭示出一条“涵盖边界”——学习到的单调通道可复现几何加权可分手工统计量族。

    

    风险监管对评分施加方向性约束；我们采纳其严格的逐输入形式——即评分对每一个风险暴露输入都单调非递减——作为规范性承诺。已部署的流水线（由单调的手工聚合特征输入符号约束的梯度提升）通过结构组合已天然满足该性质，因此“约束 vs 无约束”式的比较实际上是在为现有系统免费享有的保证定价。与之不同，我们在对比双方保持可容许性完全一致，度量“学习聚合”本身的价值。我们的工具是一个循环网络，其状态由经典风险统计量构成（带可学习变换的指数加权移动平均与高水位标记），在构造上对每个输入以及每个MC-dropout样本均保持单调。通过函数回归得到的核心发现是一条“涵盖边界”：学习到的单调通道能够复现几何加权可分手工统计量族……（原文摘要在此处截断）

    arXiv:2610.05196v2 Announce Type: replace  Abstract: Risk regulation imposes directional constraints on scores; we adopt their strict per-input form -- the score monotone non-decreasing in every exposure input -- as a normative commitment. Deployed pipelines -- monotone hand-crafted aggregates feeding sign-constrained gradient boosting -- already satisfy it by composition, so constrained-versus-unconstrained comparisons price a guarantee the incumbent has for free. We instead hold admissibility fixed on both sides and measure what learning the aggregation is worth. Our instrument is a recurrent network whose state is classical risk statistics (an exponentially weighted moving average and a high-water mark with learned transforms), monotone by construction in every input and per MC-dropout sample. The central finding, by functional regression, is a subsumption boundary: a learned monotone channel reproduces the geometrically weighted separable family of hand-crafted statistics, one chan
    
[^56]: 论高维置信序列的紧致性与计算可行性

    On the Tightness and Computational Tractability of Higher-Dimensional Confidence Sequences

    [https://arxiv.org/abs/2610.03727](https://arxiv.org/abs/2610.03727)

    该论文将一维基于下注的置信序列以三种方式提升到高维（加权Bonferroni区域、最大财富形式和组合区域），并针对最紧致但无封闭表达式解的组合区域提出了包围盒、ℓp-椭球及其交集等保持统计有效性的可计算外近似，从而在高维多元置信序列中兼顾了统计紧致性与计算可行性。

    

    现代序贯监测问题通常涉及多个指标，即我们需要同时监测多个数据流，并在证据足够充分时采取行动。置信序列（CS）正是这类持续监测问题的天然工具。然而，对于有界向量均值，现有的多元置信序列要么统计上紧致但计算上难以处理，要么计算快速但结果保守。为解决这一问题，我们研究了将一维基于下注（betting-based）的置信序列提升到高维的三种方式：加权Bonferroni区域、与之等价的最大财富（max-wealth）形式，以及组合（portfolio）区域。其中组合区域通常紧致得多，尤其在更高维度下优势明显，但其边界以及体积等性质无法用封闭形式表达。为了使这种更紧致的构造具有实用性，我们提出了组合区域的三种可计算的外近似方法，且这些近似均保持统计有效性：包围盒、$\ell_p$-椭球，以及二者的交集。我们……

    arXiv:2610.03727v2 Announce Type: replace  Abstract: Modern sequential monitoring problems often involve multiple metrics, where we monitor several data streams simultaneously and may act once the evidence is strong enough. Confidence sequences (CSs) are a natural tool for such continuous monitoring. However, for bounded vector means, existing multivariate CSs are either tight but computationally intractable, or fast to compute but conservative. To address this, we study three lifts of one-dimensional betting-based CSs to higher dimensions: a weighted Bonferroni region, an equivalent max-wealth form, and a portfolio region. The portfolio is typically much tighter, especially in higher dimensions, but its boundary and properties such as volume are not available in closed form. To make this tighter construction usable, we propose tractable outer approximations of the portfolio region that preserve statistical validity: a bounding box, an $\ell_p$-ellipsoid, and their intersection. We pro
    
[^57]: 分段Hölder图模型熵的显式界

    Explicit Bounds on the Entropy of Piecewise H\"{o}lder Graphon Models

    [https://arxiv.org/abs/2608.26501](https://arxiv.org/abs/2608.26501)

    本文为分段Hölder图函数生成的随机图熵提供了显式收敛速率和定量界，取代了以往的渐近结果。

    

    我们研究了由分段Hölder连续图函数生成的随机图的熵。首先，我们给出了随着图规模增大，归一化熵收敛速率的一个结果。我们描述了证明的核心思想，详细证明见附录。基于此结果，我们进一步推导了随机分块模型和随机几何图模型中熵的定量界。这些界提供了显式公式，而非之前研究中的渐近陈述。

    arXiv:2608.26501v1 Announce Type: cross  Abstract: We study the entropy of random graphs generated by piecewise H\"{o}lder continuous graphons. We first present a result on the rate of convergence of the normalized entropy as the size of the graph grows. The core ideas of the proof are described, with the detailed proof provided in the appendix. From this result, we then derive quantitative bounds on the entropy for the stochastic block model and random geometric graph model. These bounds provide explicit formulae rather than asymptotic statements which have been found previously.
    
[^58]: 复杂神经网络下降的Kähler景观与包括Calabi-Yau流形搜索与消灭的保证

    K\"ahler landscapes for complex neural network descents and guarantees including a search and destroy of the Calabi-Yau manifold

    [https://arxiv.org/abs/2608.19584](https://arxiv.org/abs/2608.19584)

    本文提出了一种在复参数神经网络中使用Kähler信息度量和自然梯度下降的新方法，并针对Calabi-Yau流形上的不良曲率条件提供了理论保证，通过几何定义的全局势实现了搜索与消灭策略。

    

    我们研究复参数化网络的景观。我们的方法受参数的信息论流形视角以及经典优化保证的启发，尽管涉及复杂几何变体，如通过Dolbeault渐近。下降路径在交叉熵下承认Kähler信息度量，通过Wirtinger Hessian作用于对数似然势。我们关注一种下降更新规则，采用自然梯度下降，通过逆度量缩放的微分损失，使下降路径保持在全纯切丛中。我们强调Calabi-Yau信息流形，这些流形通过不良曲率条件的景观提供了理论保证。在Calabi-Yau度量下，特别是在非紧致设置中，具有全局势而非调用Calabi猜想的拓扑要求，一个楔入无处消失的全纯...

    arXiv:2608.19584v1 Announce Type: new  Abstract: We study landscapes for complex-parameterized networks. Our approach is motivated with an information-theoretic manifold perspective of the parameter and via classical optimization guarantees although of complex geometric variety such as through Dolbeault asymptotics. The descent path admits a K\"ahler information metric under a cross-entropy via the Wirtinger Hessian on the log-likelihood potential. We restrict attention to a descent update rule with natural gradient descent via a differentiated loss scaled by the inverse metric, so the descent path remains in the holomorphic tangent bundle. We emphasize Calabi-Yau information manifolds which profane theoretical guarantees via an ill-curvature-conditioned landscape. Under a Calabi-Yau metric, specifically in a non-compact setting with a global potential so defined geometrically rather than invoking the topological requirements of the Calabi conjecture, a wedged nowhere-vanishing holomor
    
[^59]: 基于协偏度降维的双块聚类树：恢复分段多元线性状态

    Twoblock clustering trees with coskewness-based dimension reduction: recovering piecewise multivariate linear regimes

    [https://arxiv.org/abs/2607.20760](https://arxiv.org/abs/2607.20760)

    本文提出了一种基于协偏度最大化降维的双块聚类树（tbtree），这是一种高度可解释的多元回归决策树，其叶子节点为局部多元线性模型，能够有效恢复数据中的分段多元线性状态，并在保持可解释性的同时兼顾预测性能。

    

    双块聚类树（tbtree）被引入作为一种针对多元响应的高度可解释回归树。双块树是一种确定性决策树，其叶子节点为局部多元线性模型，并在叶子局部模型和杂质度量中使用稠密或稀疏的双块降维。所得模型既计算高效又具有高度可解释性。该估计器的首要目标是提供数据的可解释的、与状态对齐的分段线性描述，同时通过分裂/叶子解耦机制将预测竞争力保持为约束条件。除了提出决策树估计器本身之外，本文还引入了一种基于最大化协偏度的双块降维空间估计器，这有助于识别数据中的非正态聚类。该树固有地产生一组局部线性模型，因此适合恢复数据中的分段多元线性状态。

    arXiv:2607.20760v2 Announce Type: replace-cross  Abstract: The twoblock clustering tree (\tbtree) is introduced as a highly interpretable regression tree for multivariate responses. Twoblock trees are deterministic decision trees that have local multivariate linear models as their leaves and use dense or sparse twoblock dimension reduction as local leaf models and in the impurity. The resulting models are both computationally efficient and can be highly interpretable. The estimator's primary aim is an interpretable, regime-aligned piecewise-linear description of the data, with predictive competitiveness retained as a constraint through a split/leaf decoupling. Beyond proposing the decision tree estimator itself, this paper also introduces an estimator for the twoblock dimension reduced space based on maximizing coskewness, which facilitates identification of non-normal clusters in the data. The tree inherently produces a set of local linear models and is therefore apt to recover piecew
    
[^60]: 面向非线性回归的平均人口均等性直接优化

    Directly Optimizing Mean Demographic Parity for Nonlinear Regression

    [https://arxiv.org/abs/2607.05098](https://arxiv.org/abs/2607.05098)

    该论文提出DPVar（条件均值预测的方差）这一公平性度量，首次实现了非线性回归中平均人口均等准则的直接优化，克服了以往方法仅适用于线性预测器或低维敏感属性、且因过度约束而损害精度的局限。

    

    我们关注一类回归场景，其公平性目标是使不同敏感属性取值下的平均预测相等，这一准则被称为平均人口均等。直接优化该准则十分困难，因为它依赖于一个未知且在训练过程中不断变化的条件均值。常见的依赖性惩罚和对抗方法并不估计该条件均值，而是将预测推向完全独立。这种更强的约束即使在平均预测已经相等的情况下也可能降低准确性。现有的条件均值方法仅限于线性预测器或低维敏感属性。我们通过DPVar实现了平均人口均等的直接优化，DPVar是一种公平性度量，定义为条件均值预测的方差。由于条件均值必须随预测器的变化而进行估计，优化DPVar会引出一个泛函双层优化问题。我们开发了……

    arXiv:2607.05098v2 Announce Type: replace  Abstract: We focus on regression settings where the fairness goal is to equalize average predictions across values of a sensitive attribute, a criterion known as mean demographic parity. Directly optimizing this criterion is difficult because it depends on a conditional mean that is unknown and changes during training. Common dependence penalties and adversarial methods do not estimate this conditional mean; instead, they push predictions toward full independence. This stronger constraint can reduce accuracy even when average predictions are already equal. Existing conditional-mean methods are limited to linear predictors or low-dimensional sensitive attributes. We enable direct optimization of mean demographic parity using DPVar, a fairness measure defined as the variance of the conditional mean prediction. Because the conditional mean must be estimated as the predictor changes, optimizing DPVar leads to a functional bilevel problem. We devel
    
[^61]: 外生上下文马尔可夫决策过程学习的极小极大PAC界

    Minimax PAC Bounds for Learning in Exogenous Contextual MDPs

    [https://arxiv.org/abs/2606.25170](https://arxiv.org/abs/2606.25170)

    该论文提出了一个在查询已知前后分配采样预算的新型PAC学习框架，并针对带外生上下文的折扣马尔可夫决策过程中的策略评估、最优值估计和最优策略提取任务，给出了极小极大最优的样本复杂度界。

    

    我们引入了一个PAC框架，其中学习者可以在决策之前和决策之时访问采样预言机。样本复杂度由一对 $(n,m)$ 来衡量，其中 $n$ 是在查询已知之前花费的学习预算，$m$ 是每个查询的额外采样预算。我们在带有外生独立同分布（i.i.d.）上下文的折扣马尔可夫决策过程中展示了该框架的相关性，这些上下文在行动之前被揭示。上下文可能影响奖励和转移，但不受智能体的控制。学习者可以对未知的上下文分布和转移核进行采样。我们研究了策略评估（PE）、最优值估计（BVE）和最优策略提取（BPE）三类任务。当奖励和转移已知时，一种方差缩减算法以样本复杂度 $(\widetilde O((1-\gamma)^{-3}\varepsilon^{-2}),0)$ 解决所有这三个任务，该复杂度在对数因子意义下是极小极大最优的。设 $\mathcal{X}$ 为受控状态……（摘要此处截断）

    arXiv:2606.25170v2 Announce Type: replace-cross  Abstract: We introduce a PAC framework in which the learner can access sampling oracles both before and at decision time. Sample complexity is measured by a pair $(n,m)$, where $n$ is the learning budget spent before a query is known and $m$ is the additional sampling budget per query. We demonstrate its relevance in discounted Markov decision processes with exogenous i.i.d.\ contexts revealed before acting. Contexts may affect both rewards and transitions but remain uncontrolled by the agent. The learner can sample the unknown context distribution and the transition kernel. We study policy evaluation (PE), best-value estimation (BVE), and best-policy extraction (BPE). When rewards and transitions are known, a variance-reduced algorithm solves all three tasks with sample complexity $\bigl(\widetilde O((1-\gamma)^{-3}\varepsilon^{-2}),0\bigr)$, which is minimax optimal up to logarithmic factors. Let $\mathcal{X}$ be the controlled state s
    
[^62]: 亚高斯参数的估计

    Estimation of the sub-Gaussian Parameter

    [https://arxiv.org/abs/2606.06384](https://arxiv.org/abs/2606.06384)

    该论文研究零均值随机变量亚高斯参数（方差代理）的估计问题，证明其极小极大风险由刻画分布尾部行为的非增函数 $\delta_P$ 控制，并给出下界为 $r(\sqrt{\log n})+n^{-1/2}$、上界为 $r((\log n)^{1/2-\varepsilon})+n^{-1/2+\varepsilon}$ 的极小极大最优估计量。

    

    零均值随机变量 $X$ 的亚高斯参数（也称为方差代理）定义为 $\xi^2_\star = \sup_{\lambda \in \mathbb{R}} L(\lambda)$，其中 $L(\lambda) = \frac{2}{\lambda^2} \log \mathbb{E} e^{\lambda X}$ 是加权累积量生成函数。我们研究 $\xi^2_\star$ 的估计问题，并证明其极小极大风险由一个非增函数 $\delta_P(C) = \sup_{|\lambda| \geq C} L(\lambda) - \sup_{|\lambda| \leq C} L(\lambda)$ 决定，该函数刻画了分布 $P$ 的尾部行为的影响。在满足 $\delta_P \leq r$（其中 $r$ 为非增函数）的分布类上，极小极大风险在乘法常数意义下，其下界为 $r(\sqrt{\log n}) + n^{-1/2}$，上界为 $r((\log n)^{1/2-\varepsilon}) + n^{-1/2 + \varepsilon}$（对任意 $\varepsilon > 0$）。我们用于达到该上界的估计量基于经验加权累积量生成函数的约束最大化而构造。

    arXiv:2606.06384v2 Announce Type: replace-cross  Abstract: The sub-Gaussian parameter (also called the variance proxy) of a mean-zero random variable $X$ is defined as $\xi^2_\star = \sup_{\lambda \in \mathbb{R}} L(\lambda)$ where $L(\lambda) = \frac{2}{\lambda^2} \log \mathbb{E} e^{\lambda X}$ is a weighted cumulant generating function. We study the estimation of $\xi^2_\star$ and prove that the minimax risk is governed by a non-increasing function $\delta_P(C) = \sup_{|\lambda| \geq C} L(\lambda) - \sup_{|\lambda| \leq C} L(\lambda)$ which captures the influence of the tail behavior of the distribution $P$. Over the class of distributions with $\delta_P \leq r$ for a non-increasing function $r$, the minimax risk is, up to a multiplicative constant, lower bounded by $r(\sqrt{\log n}) + n^{-1/2}$ and upper bounded by $r((\log n)^{1/2-\varepsilon}) + n^{-1/2 + \varepsilon}$ for any $\varepsilon > 0$. Our estimator for the upper bound is based on constrained maximization of the empirical
    
[^63]: 记忆设计：概率序列层

    Memory by Design: Probabilistic Sequence Layers

    [https://arxiv.org/abs/2605.31163](https://arxiv.org/abs/2605.31163)

    本文提出了一种设计模型框架，通过贝叶斯滤波和协方差传播统一多种次二次递归序列层，并恢复协方差传播以增强记忆保留和检索。

    

    arXiv:2605.31163v3 公告类型：交叉替换 摘要：我们引入了“设计模型框架”：一种从关于记忆的明确假设中推导高效循环序列映射的方法。设计模型通过精确贝叶斯滤波将证据写入记忆；查询相关的读取输出产生预测分布，其均值作为层输出。在我们的线性-高斯实例化中，“贝叶斯层”同时传播均值和协方差：协方差跟踪存储关联中的不确定性，将写入导向不确定方向，随着证据积累而衰减增益，并保留自信记忆。同一框架统一了多种次二次递归：线性注意力、GLA和Mamba-2/SSD在潜在输入设计模型下是精确滤波器，而DeltaNet及相关Delta规则模型是贝叶斯层设计模型的协方差重置简化。恢复协方差传播为检索提供闭式预测。

    arXiv:2605.31163v3 Announce Type: replace-cross  Abstract: We introduce the \emph{design-model framework}: a way to derive efficient recurrent sequence maps from explicit assumptions about memory. A design model writes evidence into memory by exact Bayesian filtering; a query- dependent readout produces a predictive distribution whose mean is the layer output. In our linear-Gaussian instantiation, the \emph{Bayesian Layer} propagates both a mean and a covariance: the covariance tracks uncertainty over stored associations, steering writes toward uncertain directions, attenuating gains as evidence accumulates, and preserving confident memories. The same framework unifies several sub-quadratic recurrences: linear attention, GLA, and Mamba-2/SSD are exact filters under a latent-input design model, whereas DeltaNet and related Delta-rule models are covariance-reset reductions of the Bayesian Layer's design model. Restoring covariance propagation yields closed-form predictions for retrieval 
    
[^64]: 伴随DAG学习：论噪声自适应性、稀疏性与非负性的作用

    Concomitant DAG Learning: On the Roles of Noise Adaptivity, Sparsity, and Non-negativity

    [https://arxiv.org/abs/2605.23537](https://arxiv.org/abs/2605.23537)

    本教程综述了将DAG结构学习重新表述为邻接矩阵上连续、基于评分的估计问题的最新信号处理与优化进展，并阐述了噪声自适应性、稀疏性与非负性在其中的作用。

    

    有向无环图（DAG）是使人们能够对复杂系统中的因果交互进行有原则推理的核心建模工具。然而，由于一组变量背后的因果结构往往是未知的，且干预实验可能不可行或在伦理上存在挑战，因此需要解决从观测数据中推断DAG的任务。然而，大多数经典的结构识别方法面临两个关键障碍：其一是强制无环性所带来的组合难题，这严重限制了可扩展性；其二是源于潜在混淆因素或异质噪声的可识别性难题。本教程概述了近年来信号处理与优化领域的进展，这些进展通过将DAG结构学习重新表述为邻接矩阵上的连续的、基于评分的估计问题，从而解决了上述问题。我们首先以教学方式介绍结构方程模型以及……

    arXiv:2605.23537v2 Announce Type: replace  Abstract: Directed acyclic graphs (DAGs) constitute a central modeling tool to enable principled reasoning about cause-effect interactions in complex systems. However, since the causal structure underlying a group of variables is often unknown and interventions may be infeasible or ethically challenging to implement, there is a need to address the task of inferring DAGs from observational data. However, most classical structure identification approaches face two key obstacles: the combinatorial challenge of enforcing acyclicity, which severely limits scalability, and identifiability challenges arising from latent confounding or heterogeneous noise. This tutorial offers an overview of recent signal processing and optimization advances that address these issues by recasting DAG structure learning as a continuous, score-based estimation problem over adjacency matrices. We begin with a didactic introduction to structural equation models and the fo
    
[^65]: 保持评分：面向得分增强神经比率估计的自适应、免调参损失加权方法

    Keeping Score: Adaptive, Tuning-Free Loss Weighting for Score-Augmented Neural Ratio Estimation

    [https://arxiv.org/abs/2605.12118](https://arxiv.org/abs/2605.12118)

    提出一种基于损失梯度的自适应、免调参算法来动态设置得分匹配损失的权重，以极小的额外开销提升得分增强神经比率估计代理模型的质量并大幅降低调参成本。

    

    随机过程模型的神经似然代理模型（例如神经比率估计）通常通过对模拟数据进行概率分类来训练，这迫使代理模型质量与训练成本之间做出权衡。对于可以获取精确得分 ∇_θ log p(x | θ) 的结构化模型，可以通过在交叉熵损失中增加得分匹配项，将该信息纳入训练过程。然而，两种损失的最优权重无法先验得知，手动选择权重需要昂贵的调参，从而削弱了计算成本上的节省。我们提出了一种自适应、免调参的算法，在训练过程中基于损失梯度来设置得分损失的权重，仅为标准分类器训练增加极小的开销。我们在涉及网络动力学和空间过程的案例研究中评估了该方法，证明其能以大幅降低的计算成本提升代理模型的质量。

    arXiv:2605.12118v3 Announce Type: replace-cross  Abstract: Neural likelihood surrogates (e.g., Neural Ratio Estimation) for stochastic process models are commonly trained via probabilistic classification on simulated data, which forces a tradeoff between surrogate quality and training costs. For structured models where the exact score $\nabla_\theta \log p(x \mid \theta)$ is available, this information can be incorporated into training by augmenting the cross-entropy loss with a score-matching term. However, the optimal weighting of the two losses is not known a priori, and selecting it by hand requires expensive tuning that undercuts the computational savings. We propose an adaptive, tuning-free algorithm that sets the score loss weights during training based on loss gradients, adding minimal overhead to standard classifier training. We evaluate our approach on case studies involving network dynamics and spatial processes, demonstrating that it improves surrogate quality at a drastica
    
[^66]: 顿悟还是故障？低精度如何驱动“弹弓机制”式损失尖峰

    Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes

    [https://arxiv.org/abs/2605.06152](https://arxiv.org/abs/2605.06152)

    本文证明深度神经网络长期训练中周期性的“弹弓机制”损失尖峰并非源于优化动力学本身，而是浮点精度极限所致——当模型进入高置信度阶段后，正确类别梯度因舍入误差变为零，打破跨类别梯度零和约束，引发分类器与特征间的系统性漂移和正反馈循环。

    

    深度神经网络在无正则化的长期训练过程中会表现出周期性的损失尖峰，这一现象被称为“弹弓机制”。现有工作通常将其归因于内在的优化动力学，但其触发机制仍不清楚。本文证明该现象是浮点算术精度极限的结果：当训练进入高置信度阶段后，正确类别 logit 与其他 logit 之间的差值可能超过吸收误差阈值。于是在反向传播过程中，正确类别的梯度被精确舍入为零，而错误类别的梯度仍保持非零。这打破了跨类别梯度的零和约束，并在分类器层的参数更新中引入了系统性漂移。我们证明该漂移与特征之间形成了正反馈回路，导致全局分类器均值与全局特征（摘要在此处截断）。

    arXiv:2605.06152v4 Announce Type: replace-cross  Abstract: Deep neural networks exhibit periodic loss spikes during unregularized long-term training, a phenomenon known as the "Slingshot Mechanism." Existing work usually attributes this to intrinsic optimization dynamics, but its triggering mechanism remains unclear. This paper proves that this phenomenon is a result of floating-point arithmetic precision limits. As training enters a high-confidence stage, the difference between the correct-class logit and the other logits may exceed the absorption-error threshold. Then during backpropagation, the gradient of the correct class is rounded exactly to zero, while the gradients of the incorrect classes remain nonzero. This breaks the zero-sum constraint of gradients across classes and introduces a systematic drift in the parameter update of the classifier layer. We prove that this drift forms a positive feedback loop with the feature, causing the global classifier mean and the global featu
    
[^67]: 学习模拟混沌：对抗性最优传输正则化

    Learning to Emulate Chaos: Adversarial Optimal Transport Regularization

    [https://arxiv.org/abs/2604.21097](https://arxiv.org/abs/2604.21097)

    提出对抗性最优传输正则化方法，能够仅从单一含噪轨迹中联合学习高质量的摘要统计量与物理一致的混沌动力学模拟器。

    

    混沌现象存在于许多复杂动力系统中，从天气到电网，但难以用机器学习模拟器等数据驱动方法进行准确建模。尽管模拟器是加速模拟求解和解决逆问题的有前景的工具，但它们在学习混沌动力学时仍然面临困难——对初始条件的敏感性使得精确的长期预测不可行，尤其是在数据含有噪声的情况下。近期的工作转而训练模拟器去匹配混沌吸引子的统计特性，但这些方法通常依赖于手工设计的摘要统计量，或需要大型、多样化的多环境数据集。在这项工作中，我们提出了一族对抗性最优传输目标函数，能够从单一含噪轨迹中联合学习高质量的摘要统计量以及物理一致的模拟器。我们对 Sinkhorn 散度公式（2-Wasserstein……

    arXiv:2604.21097v3 Announce Type: replace-cross  Abstract: Chaos arises in many complex dynamical systems, from weather to power grids, but is difficult to accurately model with data-driven methods such as machine learning emulators. While emulators are promising tools for accelerating simulations and solving inverse problems, they still struggle to learn chaotic dynamics, where sensitivity to initial conditions renders exact long-term forecasts infeasible, especially given noisy data. Recent work instead trains emulators to match the statistical properties of chaotic attractors, but these approaches often rely on handcrafted summary statistics or large, diverse multi-environment datasets. In this work, we propose a family of adversarial optimal transport objectives that can jointly learn high-quality summary statistics and a physically consistent emulator from a single noisy trajectory. We theoretically analyze and experimentally validate a Sinkhorn divergence formulation (2-Wasserste
    
[^68]: 3BASiL：一种用于大语言模型稀疏加低秩压缩的算法框架

    3BASiL: An Algorithmic Framework for Sparse plus Low-Rank Compression of LLMs

    [https://arxiv.org/abs/2603.01376](https://arxiv.org/abs/2603.01376)

    该论文提出了3BASiL-TM，一种基于新颖三块ADMM算法的高效一次性后训练框架，通过带收敛保证的逐层重构误差最小化与跨Transformer层的联合精炼，实现大语言模型的稀疏加低秩压缩并显著缓解性能下降。

    

    大语言模型（LLM）的稀疏加低秩（S+LR）分解已成为模型压缩领域一个有前景的方向，其目标是将预训练模型的权重分解为稀疏矩阵与低秩矩阵之和（W ≈ S + LR）。尽管近期取得了一定进展，现有方法相比稠密模型往往存在显著的性能下降。在这项工作中，我们提出了 3BASiL-TM，一种针对大语言模型稀疏加低秩分解的高效一次性后训练方法，以填补这一空白。我们的方法首先提出了一种新颖的三块交替方向乘子法（3-Block ADMM），称为 3BASiL，用于在具有收敛保证的前提下最小化逐层重构误差。随后，我们设计了一个高效的 Transformer 匹配（TM）精炼步骤，在 Transformer 各层之间联合优化稀疏分量和低秩分量。该步骤最小化……（原文摘要在此处截断）

    arXiv:2603.01376v2 Announce Type: replace  Abstract: Sparse plus Low-Rank $(\mathbf{S} + \mathbf{LR})$ decomposition of Large Language Models (LLMs) has emerged as a promising direction in model compression, aiming to decompose pre-trained model weights into a sum of sparse and low-rank matrices $(\mathbf{W} \approx \mathbf{S} + \mathbf{LR})$. Despite recent progress, existing methods often suffer from substantial performance degradation compared to dense models. In this work, we introduce 3BASiL-TM, an efficient one-shot post-training method for $(\mathbf{S} + \mathbf{LR})$ decomposition of LLMs that addresses this gap. Our approach first introduces a novel 3-Block Alternating Direction Method of Multipliers (ADMM) method, termed 3BASiL, to minimize the layer-wise reconstruction error with convergence guarantees. We then design an efficient transformer-matching (TM) refinement step that jointly optimizes the sparse and low-rank components across transformer layers. This step minimizes
    
[^69]: V-ECE：估计广义期望校准误差

    V-ECE: Estimating General Expected Calibration Errors

    [https://arxiv.org/abs/2602.24230](https://arxiv.org/abs/2602.24230)

    该论文提出V-ECE方法，利用依赖预测的适当评分突破了以往方法仅能估计Bregman散度类校准误差的限制，实现了对包括$L_1$距离在内的一般凸散度（如$L_p$距离）校准误差在二分类和多分类场景下的可靠估计。

    

    在概率分类中，校准误差（CE）衡量的是预测概率 $f(X)$ 与 $\mathbb{P}(Y|f(X))$（即该预测概率所对应的真实类别分布）之间的平均散度。尽管校准误差是一种有用的诊断工具，但它难以估计：流行的基于分箱的估计器往往不一致，且在超过两个类别时扩展性较差。最近的研究将校准误差重写为模型相对于其自身预测的最佳重校准所产生的超额风险，并用适当损失来度量。然而，这种方法只适用于基于 Bregman 散度的校准误差（如平方误差），而排除了更常用的基于 $L_1$ 距离的校准误差。我们证明，使用依赖预测的适当评分可以缓解这一限制，使我们能够估计具有一般凸散度的校准误差，包括在二分类和多分类设置下具有闭式损失形式的 $L_p$ 距离。为了估计超额风险，我们引入了一个模……（原文摘要在此处截断）

    arXiv:2602.24230v2 Announce Type: replace-cross  Abstract: In probabilistic classification, calibration error (CE) measures the average divergence of predicted probabilities $f(X)$ from $\mathbb{P}(Y|f(X))$, the true class distribution for that predicted probability. While being a useful diagnostic tool, it is hard to estimate: popular binning-based estimators are often inconsistent and scale poorly beyond two classes. Recent work rewrites the CE as the excess risk of a model compared to the best recalibration of its own predictions, measured with a proper loss. However, this only works for Bregman-divergence-based calibration errors like the squared error, excluding the more popular $L_1$-distance-based CE. We show that using prediction-dependent proper scores can alleviate this restriction, allowing us to estimate CEs with general convex divergences, including $L_p$ distances with closed-form losses in the binary and multiclass settings. To estimate the excess risk, we introduce a mo
    
[^70]: 联邦格兰杰因果学习中的不确定性量化

    Uncertainty Quantification in Federated Granger Causality Learning

    [https://arxiv.org/abs/2602.13004](https://arxiv.org/abs/2602.13004)

    本文针对客户端特征异构的联邦格兰杰因果学习场景，刻画了跨客户端依赖估计过程中的不确定性传播，并利用边特定的方差有效区分真实的跨客户端依赖关系与虚假的估计边。

    

    格兰杰因果性用于识别多元时间序列中的预测性依赖关系。在各方无法共享数据的分布式环境中，联邦因果学习使得联合分析成为可能。大多数联邦因果方法假设客户端观测到相同的特征，并以点估计的方式推断因果关系，缺乏正式的不确定性量化。这些假设在许多工业系统中并不成立——在工业系统中，客户端观测到不同的特征，且目标是要估计跨客户端的依赖关系（边）。这些依赖关系必须通过重复的客户端-服务器迭代来间接估计。来自客户端数据和模型参数的不确定性会在此过程中传播，使得仅凭点估计不足以评估跨客户端的边。本文刻画了这种不确定性传播过程，并利用边特定的方差来区分真实的跨客户端依赖关系与虚假的估计边。

    arXiv:2602.13004v3 Announce Type: replace  Abstract: Granger causality identifies predictive dependencies in multivariate time series. In distributed settings where parties cannot share data, federated causal learning enables joint analysis. Most federated causal methods assume that clients observe the same features and infer causal relationships as point estimates, with little formal uncertainty quantification. These assumptions do not hold in many industrial systems, where clients observe different features, and the objective is to estimate cross-client dependencies (edges). These dependencies must be estimated indirectly through repeated client-server iterations. Uncertainty from client data and model parameters propagates through this process, making point estimates alone insufficient for assessing cross-client edges. This paper characterizes this uncertainty propagation and uses edge-specific variances to distinguish genuine cross-client dependencies from spurious estimated edges.
    
[^71]: 处理协同线性预测中的协变量不匹配问题

    Handling Covariate Mismatch in Collaborative Linear Prediction

    [https://arxiv.org/abs/2602.02083](https://arxiv.org/abs/2602.02083)

    本文研究多中心协同线性预测中各中心记录不同协变量的“协变量不匹配”问题，提出了兼容联邦学习约束的估计方法——低维下基于逐分量聚合的代入估计量，高维下采用保持可交换性的“先插补后岭回归”策略。

    

    跨多个中心训练预测模型通常假设所有中心收集相同的协变量集合。然而在实践中，各中心可能记录观测对象的不同特征，我们将这种情形称为“协变量不匹配”。我们研究了这一具有挑战性设定下的线性预测问题，假设各中心服从中心级MCAR（完全随机缺失）缺失模式，并开发了即使特征集异质也能利用跨中心信息的估计量。在低维情形下，我们基于协方差和交叉矩估计的逐分量聚合，提出了针对预言线性预测器的代入估计量。在更高维度下，我们研究了一种“先插补后回归”策略：首先使用保持可交换性的插补程序补全缺失协变量，然后拟合岭正则化线性模型。所有提出的估计量均兼容联邦学习约束：个体层面数据保持（本地不出中心）。

    arXiv:2602.02083v2 Announce Type: replace-cross  Abstract: Training predictive models across multiple centers typically assumes that all centers collect the same set of covariates. In practice, however, they may record different features of their observations, a setting we refer to as covariate mismatch. We study linear prediction under this challenging setting, assuming center-wise MCAR missingness patterns, and develop estimators that exploit information across centers despite heterogeneous feature sets. In the low-dimensional regime, we propose a plug-in estimator of the oracle linear predictor based on component-wise aggregation of covariance and cross-moment estimates. In higher dimensions, we study an impute-then-regress strategy that first completes the missing covariates using an exchangeability-preserving imputation procedure and then fits a ridge-regularized linear model. All proposed estimators are compatible with federated learning constraints: individual-level data remain 
    
[^72]: 一次置换足矣：快速、确定性的特征重要性与模型压力测试

    One Permutation Is All You Need: Fast, Deterministic Feature Importance and Model Stress-Testing

    [https://arxiv.org/abs/2512.13892](https://arxiv.org/abs/2512.13892)

    用单次最大-最小秩最优的确定性置换替代多次随机置换，可将特征重要性估计的计算复杂度从 O(B·n·p) 降至 O(n·p)，消除估计方差并保持或提升估计精度，并可扩展用于模型压力测试。

    

    在机器学习模型中可靠地估计特征贡献对于透明度、算法公平性和监管合规至关重要。虽然置换特征重要性被广泛使用，但经典实现依赖于重复的蒙特卡洛洗牌，这引入了显著的计算开销和随机不稳定性。在本文中，我们证明用单一的、最大-最小秩最优的确定性置换替代 $B$ 次随机置换，能够在消除估计方差的同时保持或改善与真实重要性的相关性，并将复杂度从 $O(B \cdot n \cdot p)$ 降低至 $O(n \cdot p)$。在位置-尺度特征分布下，我们严格证明了尺度调整后线性回归系数的精确恢复，以及在凹型模型敏感度下改进的重要性估计。我们进一步沿两个互补维度扩展这一确定性框架。首先，系统特征

    arXiv:2512.13892v3 Announce Type: replace-cross  Abstract: Reliable estimation of feature contributions in machine learning models is essential for transparency, algorithmic fairness, and regulatory compliance. While permutation feature importance is widely used, classical implementations rely on repeated Monte Carlo shuffling, introducing significant computational overhead and stochastic instability. In this paper, we show that replacing $B$ random permutations with a single, max-min rank-optimal deterministic permutation maintains or improves correlation with ground-truth importance while eliminating estimation variance and reducing complexity from $O(B \cdot n \cdot p)$ to $O(n \cdot p)$. Under location-scale feature distributions, we formally prove exact recovery of scale-adjusted linear regression coefficients, alongside improved importance estimation under concave model sensitivity. We extend this deterministic framework along two complementary dimensions. First, Systemic Feature
    
[^73]: 用于解混流的自充分独立成分分析

    Self-sufficient Independent Component Analysis for Demixing Flows

    [https://arxiv.org/abs/2512.00665](https://arxiv.org/abs/2512.00665)

    提出一种无先验、无似然的自充分独立成分分析方法，通过最小化条件KL散度并顺序学习解混流模型来从数据中学习解耦信号，同时完全避免了不稳定的对抗训练。

    

    我们研究了利用非线性独立成分分析（ICA）从数据中学习解耦信号的问题。受自监督学习进展的启发，我们提出学习自充分信号：给定已恢复信号的其余值，观察其他信号不应改变其缺失值的条件分布。我们将该问题表述为条件KL散度的最小化。我们的算法是无先验且无似然的，即它既不规定参数化的源密度，也不规定观测似然。为解决KL散度最小化问题，我们提出了一种顺序算法，在每次迭代中学习一个解混流模型，并证明了其理想化的Wasserstein梯度流变体（具有精确速度和总体投影条件）在总相关性上具有局部下降性质。该方法完全避免了不稳定的对抗性训练。

    arXiv:2512.00665v2 Announce Type: replace-cross  Abstract: We study the problem of learning disentangled signals from data using non-linear Independent Component Analysis (ICA). Motivated by advances in self-supervised learning, we propose to learn self-sufficient signals: Given the remaining values of a recovered signal, observing other signals should not change the conditional distribution of its missing value. We formulate this problem as the minimization of a conditional KL divergence. Our algorithm is prior-free and likelihood-free in the sense that it prescribes neither parametric source densities nor an observation likelihood. To tackle the KL divergence minimization problem, we propose a sequential algorithm that learns a de-mixing flow model at each iteration, and prove local descent of the total correlation for its idealized Wasserstein-gradient-flow variant with exact velocities and a population projection condition. This approach completely avoids the unstable adversarial t
    
[^74]: 通过合成模型生成实现近最优可解释模型的可扩展元学习

    Towards Scalable Meta-Learning of near-optimal Interpretable Models via Synthetic Model Generations

    [https://arxiv.org/abs/2511.04000](https://arxiv.org/abs/2511.04000)

    本文提出通过合成采样近最优决策树来生成大规模预训练数据的高效可扩展方法，使MetaTree transformer在决策树元学习上达到与真实数据或昂贵最优树预训练相当的性能，同时大幅降低计算成本。

    

    决策树因其可解释性而广泛应用于金融和医疗等高风险领域。本工作提出了一种高效、可扩展的方法来生成合成预训练数据，从而实现决策树的元学习。我们的方法通过合成方式采样近最优的决策树，构建出大规模、贴近现实的数据集。借助MetaTree transformer架构，我们证明该方法所取得的性能可与在真实数据上预训练或使用计算成本高昂的最优决策树预训练相媲美。该策略显著降低了计算成本，提升了数据生成的灵活性，并为可解释决策树模型的可扩展、高效元学习铺平了道路。

    arXiv:2511.04000v2 Announce Type: replace-cross  Abstract: Decision trees are widely used in high-stakes fields like finance and healthcare due to their interpretability. This work introduces an efficient, scalable method for generating synthetic pre-training data to enable meta-learning of decision trees. Our approach samples near-optimal decision trees synthetically, creating large-scale, realistic datasets. Using the MetaTree transformer architecture, we demonstrate that this method achieves performance comparable to pre-training on real-world data or with computationally expensive optimal decision trees. This strategy significantly reduces computational costs, enhances data generation flexibility, and paves the way for scalable and efficient meta-learning of interpretable decision tree models.
    
[^75]: 部分信息下最优交易的深度强化学习方法

    Deep reinforcement learning for optimal trading with partial information

    [https://arxiv.org/abs/2511.00190](https://arxiv.org/abs/2511.00190)

    该论文提出三种将循环神经网络与强化学习相结合的方法，用于求解具有状态切换参数的部分可观测环境下的最优交易问题，使交易者能够从可观测数据中推断并过滤市场的潜在状态信息。

    

    强化学习（RL）在金融领域的应用（包括最优交易与执行）日益受到关注。然而，据我们所知，利用强化学习构建能够挖掘市场中潜在信息的最优交易策略这一问题，却鲜有人研究。在本文中，我们考虑了一个最优交易问题，其中交易信号遵循具有状态切换参数的Ornstein-Uhlenbeck过程。该问题自然地被表述为部分可观测马尔可夫决策问题，要求交易者直接从可观测过程的历史数据中推断潜在信息。我们将循环神经网络（RNN）与强化学习相结合，以同时解决该问题中的滤波与交易两个部分。更具体地，我们提出了三种不同的基于强化学习的方法，并在其基础上引入RNN对智能体所处交易环境的潜在状态进行滤波。

    arXiv:2511.00190v2 Announce Type: replace  Abstract: Reinforcement Learning (RL) has attracted increasing interest in financial applications, including optimal trading and execution. However, the use of RL for optimal trading strategies that exploit latent information in the market has been, to the best of our knowledge, subject to little attention. In this paper, we consider an optimal trading problem in which the trading signal follows an Ornstein-Uhlenbeck process with regime switching parameters. The problem is naturally formulated as a partially observable Markov decision problem, requiring a trader to infer latent information directly from the history of the observable process. We combine recurrent neural networks (RNN) with RL to address both the filtering and trading components of the problem. More specifically, we propose three distinct RL-based approaches, on top of which we incorporate RNN to filter the latent state of the environment where the agent is trading. The first, a
    
[^76]: 不同维度相交流形上图狄利克雷能量与图拉普拉斯算子的收敛性

    Convergence of graph Dirichlet energies and graph Laplacians on intersecting manifolds of varying dimensions

    [https://arxiv.org/abs/2509.24458](https://arxiv.org/abs/2509.24458)

    该论文证明了在多维度相交流形并集上，非归一化图狄利克雷能量渐近只能感知最高维流形内部的变化，而归一化图狄利克雷能量则收敛到能同时适应所有维度的张量化狄利克雷能量，从而为理解机器学习方法如何适应具有不同内在维度的数据提供了理论依据。

    

    我们研究了在可能具有不同维度的相交流形并集上，图狄利克雷能量的Γ-收敛性以及图拉普拉斯算子的谱收敛性。我们的研究源于机器学习中的问题，因为现实世界的数据通常由具有不同内在维度的部分或类别组成。一个重要的挑战是理解哪些机器学习方法能够适应这种多样的维度。我们研究了标准的非归一化图狄利克雷能量和归一化图狄利克雷能量。我们证明了非归一化能量及其相关的图拉普拉斯算子在渐近意义下只能感知最高维度流形内部的变化。另一方面，我们证明了归一化狄利克雷能量收敛到流形并集上的一个（张量化）狄利克雷能量，该能量能够同时适应所有维度。我们还建立了相应的谱收敛性，并给出了一些数值实验。

    arXiv:2509.24458v2 Announce Type: replace-cross  Abstract: We study $\Gamma$-convergence of graph Dirichlet energies and spectral convergence of graph Laplacians on unions of intersecting manifolds of potentially different dimensions. Our investigation is motivated by problems of machine learning, as real-world data often consist of parts or classes with different intrinsic dimensions. An important challenge is to understand which machine learning methods adapt to such varied dimensionalities. We investigate the standard unnormalized and the normalized graph Dirichlet energies. We show that the unnormalized energy and its associated graph Laplacian asymptotically only sees the variations within the manifold of the highest dimension. On the other hand, we prove that the normalized Dirichlet energy converges to a (tensorized) Dirichlet energy on the union of manifolds that adapts to all dimensions simultaneously. We also establish the related spectral convergence and present a few numeri
    
[^77]: 成员推断与隐私审计的样本复杂度

    The Sample Complexity of Membership Inference and Privacy Auditing

    [https://arxiv.org/abs/2508.19458](https://arxiv.org/abs/2508.19458)

    本文在高斯均值估计的基础设定下，研究了成员推断攻击的样本复杂度，即确定了成功实施攻击和隐私审计所需的最少参考样本数量。

    

    成员推断攻击通过获取学习算法的输出和一个目标个体，试图判断该个体是训练数据的成员，还是来自同一分布的独立样本。成功的成员推断攻击通常要求攻击者对训练数据所采样的分布具有一定的了解，而这种知识通常通过来自该分布的一组独立参考样本来体现。在本工作中，我们通过研究样本复杂度——即成功攻击所需的最少参考样本数量——来探究攻击者在成员推断中需要多少信息。我们在高斯均值估计这一基础设定中研究该问题：学习算法获得来自d维高斯分布 $\mathcal{N}(\mu,\Sigma)$ 的 $n$ 个样本，并尝试在一定的误差范围内估计 $\hat\mu$。

    arXiv:2508.19458v2 Announce Type: replace  Abstract: A membership-inference attack gets the output of a learning algorithm, and a target individual, and tries to determine whether this individual is a member of the training data or an independent sample from the same distribution. A successful membership-inference attack typically requires the attacker to have some knowledge about the distribution that the training data was sampled from, and this knowledge is often captured through a set of independent reference samples from that distribution. In this work we study how much information the attacker needs for membership inference by investigating the sample complexity-the minimum number of reference samples required-for a successful attack. We study this question in the fundamental setting of Gaussian mean estimation where the learning algorithm is given $n$ samples from a Gaussian distribution $\mathcal{N}(\mu,\Sigma)$ in $d$ dimensions, and tries to estimate $\hat\mu$ up to some error
    
[^78]: 面向分布内数据获取的共形数据污染检验

    Conformal Data Contamination Tests for In-distribution Data Acquisition

    [https://arxiv.org/abs/2507.13835](https://arxiv.org/abs/2507.13835)

    本文提出了一种无分布假设的共形数据污染检验框架，仅需检查少量数据即可识别出对模型个性化最有价值的外部数据代理，从而在数据获取前提供质量保证。

    

    在许多机器学习任务中，高质量数据的数量受限于数据所有者本地可获取的数据。高质量数据集可以通过与外部数据代理进行交易或共享来扩展。然而，外部数据可能被污染，或引入不良的样本多样性，从而降低个性化机器学习任务的性能，例如罕见疾病诊断或推荐系统。因此，数据购买者在获取数据之前需要质量保证。先前的工作主要依赖于对不同数据代理的数据分布假设，将质量检查推迟到事后步骤，这涉及成本高昂的数据估值流程。我们提出了一种无分布假设、具备污染感知能力的数据获取框架，该框架仅需检查少量数据，即可识别出其数据对模型个性化最有价值的外部数据代理。为实现这一目标，我们引入了新颖的双样本

    arXiv:2507.13835v2 Announce Type: replace-cross  Abstract: The amount of quality data in many machine learning tasks is limited to what is available locally to data owners. The set of quality data can be expanded through trading or sharing with external data agents. However, external data may be contaminated or introduce undesirable sample diversity which can degrade performance of personalized machine learning tasks, as in diagnosis of a rare disease or recommendation systems. Therefore, data buyers need quality guarantees prior to data acquisition. Previous works primarily rely on distributional assumptions about data from different agents, relegating quality checks to post-hoc steps involving costly data valuation procedures. We propose a distribution-free, contamination-aware data acquisition framework that, by inspecting only a small volume of data, identifies external data agents whose data is most valuable for model personalization. To achieve this, we introduce novel two-sample
    
[^79]: 迈向合理的概念瓶颈模型

    Towards Reasonable Concept Bottleneck Models

    [https://arxiv.org/abs/2506.05014](https://arxiv.org/abs/2506.05014)

    提出概念推理模型（CREAM），一种可在架构层面显式编码概念间关系与概念-任务关系、并能借助正则化旁路通道处理不完整概念集的概念瓶颈模型新框架，同时引入了与C→Y无关的可解释性评估指标。

    

    我们提出了一种新颖、灵活且高效的概念瓶颈模型设计框架，使从业者能够在模型进行预测时的推理过程中，显式地编码和扩展他们关于概念-概念（C-C）以及概念-任务（C→Y）关系的先验知识和信念。由此产生的概念推理模型在架构上编码了任意类型的C-C关系，例如互斥性、层次关联和/或相关性，以及可能稀疏的C→Y关系。此外，CREAM可以选择性地引入一个正则化的旁路通道来补充可能不完整的概念集，在取得有竞争力的任务性能的同时，促使预测基于概念进行。为了在此类设置下评估概念瓶颈模型，我们引入了一个与C→Y无关的度量指标，用以量化预测的可解释性。

    arXiv:2506.05014v3 Announce Type: replace-cross  Abstract: We propose a novel, flexible, and efficient framework for designing Concept Bottleneck Models (CBMs) that enables practitioners to explicitly encode and extend their prior knowledge and beliefs about the concept-concept ($C-C$) and concept-task ($C \to Y$) relationships within the model's reasoning when making predictions. The resulting $\textbf{C}$oncept $\textbf{REA}$soning $\textbf{M}$odels (CREAMs) architecturally encode arbitrary types of $C-C$ relationships such as mutual exclusivity, hierarchical associations, and/or correlations, as well as potentially sparse $C \to Y$ relationships. Moreover, CREAM can optionally incorporate a regularized side-channel to complement the potentially {incomplete concept sets}, achieving competitive task performance while encouraging predictions to be concept-grounded. To evaluate CBMs in such settings, we introduce a $C \to Y$ agnostic metric that quantifies interpretability when predicti
    
[^80]: 广义核下t分布随机邻域嵌入的平衡分布

    Equilibrium Distribution for t-Distributed Stochastic Neighbor Embedding with Generalized Kernels

    [https://arxiv.org/abs/2505.24311](https://arxiv.org/abs/2505.24311)

    该论文为广义输入输出核下的t-SNE大样本变分问题建立了严格的数学理论，证明了尺度参数的存在唯一性、解的存在性与一致有界性，以及离散最优解收敛到满足平衡方程的紧支撑平衡分布。

    

    我们研究了一类输入核和输出核下t分布随机邻域嵌入的大样本变分问题。输入律具有紧支撑，且其密度在该支撑上连续。一个熵方程确定了输入核中的尺度参数，我们证明该参数在正密度的内部点处存在且唯一。随后，我们给出了解在整个支撑上存在且一致有界的充分条件。在这些条件以及输出核的衰减假设下，离散最优值收敛于连续统最小值。近似极小元的经验测度在平移后是紧的；每个子序列极限都是满足平衡方程的紧支撑极小元。可容许的输出核包括高斯核，以及在输出维度为二时的柯西核。数值例子比较了二维表示（原文在此处截断）。

    arXiv:2505.24311v3 Announce Type: replace-cross  Abstract: We study the large-sample variational problem for t-distributed stochastic neighbor embedding with a class of input and output kernels. The input law has compact support and a density continuous on that support. An entropy equation determines the scale parameter in the input kernel, and we prove that this parameter exists and is unique at interior points of positive density. We then give sufficient conditions for solutions to exist and be uniformly bounded on the entire support. Under these conditions and a decay assumption on the output kernel, the discrete optimal values converge to a continuum minimum. Empirical measures of approximate minimizers are tight after translation; every subsequential limit is a compactly supported minimizer satisfying the equilibrium equation. The admissible output kernels include Gaussian kernels and, in output dimension two, the Cauchy kernel. Numerical examples compare the two-dimensional repre
    
[^81]: 原型分析综述

    A Survey on Archetypal Analysis

    [https://arxiv.org/abs/2504.12392](https://arxiv.org/abs/2504.12392)

    这是首篇关于原型分析（AA）的综述，系统介绍了其方法论、面临的非凸优化挑战、跨科学领域的广泛应用以及数据建模的最佳实践。

    

    原型分析（Archetypal Analysis, AA）最初由Adele Cutler和Leo Breiman于1994年提出，作为一种从观测数据中提取不同方面（即所谓的“原型”）的计算方法，每个观测记录被近似为这些原型的混合（即凸组合）。由此，AA为特征提取和降维提供了直接、可解释且可说明的表示方法，有助于理解高维数据的结构，并在各科学领域得到广泛应用。然而，AA也面临挑战，特别是其相关的优化问题是非凸的。这是首篇为研究人员和数据挖掘从业者提供AA所提供的方法论与机遇概览的综述，调研了AA在各科学领域的众多应用，以及使用AA进行数据建模的最佳实践。

    arXiv:2504.12392v3 Announce Type: replace-cross  Abstract: Archetypal analysis (AA) was originally proposed in 1994 by Adele Cutler and Leo Breiman as a computational procedure for extracting distinct aspects, so-called archetypes, from observations, with each observational record approximated as a mixture (i.e., convex combination) of these archetypes. AA thereby provides straightforward, interpretable, and explainable representations for feature extraction and dimensionality reduction, facilitating the understanding of the structure of high-dimensional data and enabling wide applications across the sciences. However, AA also faces challenges, particularly as the associated optimization problem is nonconvex. This is the first survey that provides researchers and data mining practitioners with an overview of the methodologies and opportunities that AA offers, surveying the many applications of AA across disparate fields of science, as well as best practices for modeling data with AA an
    
[^82]: 面向自监督对比学习的动态全局假负样本发现方法

    Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning

    [https://arxiv.org/abs/2502.20612](https://arxiv.org/abs/2502.20612)

    提出GloFND方法，通过为每个锚点动态学习阈值，在整个数据集范围内全局识别自监督对比学习中的假负样本，且每次迭代的计算成本与数据集规模无关。

    

    在自监督对比学习中，负样本对通常由一个锚点图像和从整个数据集中抽取的样本（排除锚点本身）构成。然而，这种方式可能构建出语义相似的负样本对，即所谓的“假负样本”，导致它们的嵌入向量被错误地推开。为了解决这一问题，我们提出了GloFND，一种基于优化的方法，能够在训练过程中动态地为每个锚点数据自动学习阈值，从而识别其假负样本。与以往用于假负样本发现的方法不同，我们的方法是在整个数据集范围内全局检测假负样本，而不是在小批次内进行局部检测。此外，其每次迭代的计算成本与数据集规模无关。在图像数据和图文数据上的实验结果验证了所提方法的有效性。我们的实现代码已开源。

    arXiv:2502.20612v2 Announce Type: cross  Abstract: In self-supervised contrastive learning, negative pairs are typically constructed using an anchor image and a sample drawn from the entire dataset, excluding the anchor. However, this approach can result in the creation of negative pairs with similar semantics, referred to as "false negatives", leading to their embeddings being falsely pushed apart. To address this issue, we introduce GloFND, an optimization-based approach that automatically learns on the fly the threshold for each anchor data to identify its false negatives during training. In contrast to previous methods for false negative discovery, our approach globally detects false negatives across the entire dataset rather than locally within the mini-batch. Moreover, its per-iteration computation cost remains independent of the dataset size. Experimental results on image and image-text data demonstrate the effectiveness of the proposed method. Our implementation is available at
    
[^83]: 具有有限VC维的网络：利与弊

    Networks with Finite VC Dimension: Pro and Contra

    [https://arxiv.org/abs/2502.02679](https://arxiv.org/abs/2502.02679)

    该论文证明有限的VC维虽有利于经验误差的一致收敛，却可能不利于函数逼近，并基于高维几何的测度集中性质证明，此类网络在处理大规模数据集时逼近误差与经验误差均几乎呈确定性行为。

    

    本文从高维几何与统计学习理论的角度，研究了利用神经网络对大规模数据集的分类器进行逼近与学习的问题。文章比较了网络输入-输出函数集合的VC维对逼近能力的影响，与其对基于数据样本学习的一致性的影响。结果表明，尽管有限的VC维对于经验误差的一致收敛是有利的，但对于逼近从某个概率分布（该分布建模了函数在特定应用中出现的可能性）中抽取的函数而言，却可能并不理想。基于高维几何的测度集中性质，论文证明：对于实现具有有限VC维的输入-输出函数集合的网络，在处理大规模数据集时，其逼近误差和经验误差均表现出近乎确定性的行为。

    arXiv:2502.02679v3 Announce Type: replace-cross  Abstract: Approximation and learning of classifiers of large data sets by neural networks in terms of high-dimensional geometry and statistical learning theory are investigated. The influence of the VC dimension of sets of input-output functions of networks on approximation capabilities is compared with its influence on consistency in learning from samples of data. It is shown that, whereas finite VC dimension is desirable for uniform convergence of empirical errors, it may not be desirable for approximation of functions drawn from a probability distribution modeling the likelihood that they occur in a given type of application. Based on the concentration-of-measure properties of high dimensional geometry, it is proven that both errors in approximation and empirical errors behave almost deterministically for networks implementing sets of input-output functions with finite VC dimensions in processing large data sets. Practical limitations
    
[^84]: 线图的图子（Graphons）

    Graphons of Line Graphs

    [https://arxiv.org/abs/2409.01656](https://arxiv.org/abs/2409.01656)

    本文提出一种通过将稀疏图映射到其线图并利用“平方度性质”使稀疏图产生稠密线图的方法，从而可以应用稠密图极限理论来分析稀疏图，并实证证明能够区分原本都收敛到零图子的不同数量的星形图。

    

    我们考虑从稀疏有限图序列的观测中估计图极限（称为图子，graphons）的问题。在本文中，我们展示了一种简单的方法，可以揭示一类稀疏图的性质。该方法将原始图映射到其线图。我们证明，满足一种特殊性质（我们称之为平方度性质，square-degree property）的图是稀疏的，但会产生稠密的线图。这使得我们能够利用稠密图图极限的已有结果来推导收敛性。特别地，星形图满足平方度性质，从而产生稠密的线图以及线图的非零图子。我们通过实证演示，可以利用相应线图的图子来区分不同数量的星（这些星本身是稀疏的）。而在原始图中，由于稀疏性，不同数量的星都会收敛到零图子。类似地，超线性优先连接……

    arXiv:2409.01656v4 Announce Type: replace-cross  Abstract: We consider the problem of estimating graph limits, known as graphons, from observations of sequences of sparse finite graphs. In this paper we show a simple method that can shed light on a subset of sparse graphs. The method involves mapping the original graphs to their line graphs. We show that graphs satisfying a particular property, which we call the square-degree property are sparse, but give rise to dense line graphs. This enables the use of results on graph limits of dense graphs to derive convergence. In particular, star graphs satisfy the square-degree property resulting in dense line graphs and non-zero graphons of line graphs. We demonstrate empirically that we can distinguish different numbers of stars (which are sparse) by the graphons of their corresponding line graphs. Whereas in the original graphs, the different number of stars all converge to the zero graphon due to sparsity. Similarly, superlinear preferentia
    

