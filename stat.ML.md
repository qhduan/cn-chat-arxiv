# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Conditional Generative Learning: Model and Error Analysis](https://rss.arxiv.org/abs/2402.01460) | 提出了一种基于ODE的深度生成方法，通过条件Follmer流来学习条件分布，通过离散化和深度神经网络实现高效转化。同时，通过Wasserstein距离的非渐近收敛速率，提供了第一个端到端误差分析，数值实验证明其在不同场景下的优越性。 |
| [^2] | [ENFR: Evidence Networks with Flow-Matched References for Amortized Simulation-Based Model Inference](https://arxiv.org/abs/2610.23825) | 本文提出ENFR方法，通过结合证据网络与流匹配参考技术，实现对贝叶斯证据的准确、快速且摊销式的估计，从而突破仅衡量证据比值的贝叶斯因子方法的局限，为高维拟合优度检验和模型误设检验提供直接解决方案。 |
| [^3] | [Learning without relearning physical symmetry: an SO(3)-equivariant convolutional network (CNN) on the sphere](https://arxiv.org/abs/2610.23821) | 本文提出了一种专为HEALPix球面像素化设计的SO(3)等变卷积神经网络，利用可操纵滤波器、平行移动和自旋加权特征实现可证明的精确等变性，使网络无需消耗训练数据来学习物理对称性。 |
| [^4] | [Task-aware Kernel Stein certificates for R\'{e}nyi differential privacy](https://arxiv.org/abs/2610.23798) | 本文提出基于护航分布的核Stein差异方法，通过设计聚焦相关隐私方向的核函数，为Rényi差分隐私提供可计算的隐私证书及面向任务的隐私保证，并给出了高斯与拉普拉斯机制的精确恒等式和指数机制的界。 |
| [^5] | [Parametric Resonance in Primal-Dual Learning for Slowly Non-Stationary Constrained MDPs](https://arxiv.org/abs/2610.23788) | 本文揭示了缓慢非平稳约束马尔可夫决策过程中原始-对偶学习的不稳定性并非源于非平稳性本身，而是源于约束估计与测量失配所引发的参数共振，并定量给出了可容忍失配随强迫带宽的平方根增长、随对偶-原始步长比的平方根缩减的规律。 |
| [^6] | [Mesoscopic learning dynamics of extensive-width neural networks](https://arxiv.org/abs/2610.23779) | 该论文通过刻画重叠量的分布，提出了一种介观描述方法，将广宽度两层神经网络学习任意非线性目标函数的动力学约化为少数集体变量的演化，从而克服了传统方法需要追踪发散数量序参量的难题。 |
| [^7] | [Exact Tail Laws for Nonlinear SGD with Infinite-Variance Gradients](https://arxiv.org/abs/2610.23750) | 该论文为无限方差梯度噪声下的非线性恒定步长 SGD 最后一次迭代建立了精确的一阶尾部概率律，证明其尾部由单个大梯度冲击经局部 Hessian 流传输所主导，并由此给出显式的投影与超额目标尾部及极端分位数。 |
| [^8] | [Equivalence testing and run-count design for stochastic machine-learning experiments](https://arxiv.org/abs/2610.23700) | 本文提出了一套基于两级（种子×样本）方差模型和TOST等效性检验的严格认证方案，并给出将预设容差与目标功效换算为所需运行次数的闭式公式，从而取代以往仅凭两三次运行无显著差异就宣称方法等效的不可靠做法。 |
| [^9] | [No-Regret Language Generation](https://arxiv.org/abs/2610.23641) | 本文提出无遗憾生成框架，用基于树的错误强制维度精确刻画生成任务的极小极大复杂度，揭示所有可容许生成率均可达成，并将其与在线预测联系起来，得到由Littlestone维度决定的错误界和遗憾界。 |
| [^10] | [Boundary Detection for High-Dimensional Point Clouds](https://arxiv.org/abs/2610.23590) | 本文提出两种适用于高维、非均匀密度且含噪点云数据的新型边界检测方法，可同时输出边界分数和法向量估计，并在肿瘤边界检测与LLM激活几何分析中得到验证。 |
| [^11] | [A general recipe for data-adaptive anytime-valid envelopes for iterative algorithms](https://arxiv.org/abs/2610.23570) | 该论文提出了一个通用方法，可为一大类迭代算法构建任意时刻有效且数据自适应的停止边界，将误差分析统一归结为对可预测加权噪声和满足时间一致集中性这一单一要求，克服了以往确定性包络过于保守、数据自适应包络只能逐例推导的局限。 |
| [^12] | [Beyond Expected Values: Risk-Sensitive Planning with Distributional Monte-Carlo Tree Search](https://arxiv.org/abs/2610.23565) | 该论文提出了CATSO和PATSO两种分布式蒙特卡洛树搜索算法，通过在Q边维护完整的经验分布（而非仅存储均值估计）并结合CVaR与乐观奖励进行动作选择，实现了针对尾部风险的风险敏感规划，并证明了n次rollout后根节点值误差的O(n^{-1/2})理论界。 |
| [^13] | [Causal Discovery beyond Conditional Independence: Identifiability, Algorithms, and Statistical Inference for Directed Acyclic and Cyclic Graphs](https://arxiv.org/abs/2610.23543) | 本综述系统梳理了超越条件独立性的现代因果发现方法，提出将识别来源与统计估计相分离的统一框架，涵盖分布不对称、条件矩约束和跨环境变异等手段，并讨论了潜在混淆与循环图下的可识别性、算法与统计推断。 |
| [^14] | [Adaptive conformal prediction on the Grassmannian](https://arxiv.org/abs/2610.23508) | 该论文提出了一个在格拉斯曼流形上构建局部自适应预测区域的保形预测框架，通过基不变的几何距离和学习到的难度估计器实现几何感知的非一致性度量，在保持有限样本覆盖保证的同时提升了不同难度下的覆盖均匀性。 |
| [^15] | [Towards optimal selection schedules for online active learning](https://arxiv.org/abs/2610.23477) | 该论文在高维在线主动学习设定下证明了最优样本查询策略的本质：贪婪最小化目标增量的短视策略等价于小间隔或中间间隔选择策略，即只标注间隔落在随时间不断收缩的区间内的样本，以在标注预算或“查询代价 Δ + 错误代价”的目标下最优。 |
| [^16] | [Minimax-optimal Scaling Laws in Ridge Regression via Transfer Learning](https://arxiv.org/abs/2610.23337) | 本文在高维岭回归迁移学习框架下建立了风险确定性等价，并证明在共享回归函数和可交换幂律协方差谱的条件下可实现极小极大最优标度律，且最优源正则化仅依赖源数据、具有跨任务普适性。 |
| [^17] | [The Loss or the Architecture? What Matters in Survival Analysis with Competing Risks](https://arxiv.org/abs/2610.23322) | 该论文将损失函数与网络架构解耦并独立评估，在20个表格和3个病理图像数据集上发现：在删失加权下直接建模累积发生率的损失函数比选择更好的骨干网络带来更大的收益。 |
| [^18] | [A Class of R\'enyi-based Concentration Metrics for Kolmogorov-Arnold Geometry in Multilayer Perceptrons](https://arxiv.org/abs/2610.23267) | 本文提出了一族基于Rényi熵的α-RC集中度度量，通过将多层感知机Kolmogorov-Arnold几何中雅可比子式的归一化幅值与均匀分布进行比较，克服了传统参与率指标因最大值依赖网络规模而无法跨架构解读集中度的局限性。 |
| [^19] | [Information-Geometric Blind Spots of Conformal Test Martingales](https://arxiv.org/abs/2610.23131) | 该论文揭示了共形检验鞅的信息几何盲区：凡保持符合性函数前推分布不变的分布偏移均无法被检测，并借助 A-等价类与 A-隐秘 I-投影给出该盲区的显式刻画与勾股分解，同时证明当符合性函数取 Neyman-Pearson 似然比时盲区消失。 |
| [^20] | [Distinguishability and Recovery for a Doubly Sparse Spiked Wigner Model](https://arxiv.org/abs/2610.23033) | 本文研究信号与噪声均稀疏的双稀疏尖峰Wigner模型，提出留一二次阈值化、符号同步及谱精炼等算法，在信噪比保持常数阶的条件下实现了稀疏秩一信号的支撑识别、精确恢复与强恢复，并给出了信号稀疏性、噪声稀疏性与计算成本之间的显式权衡刻画。 |
| [^21] | [Towards Hyperparameter Transfer for Differentially Private Optimization](https://arxiv.org/abs/2610.22976) | 本文提出DP-μP框架，将最大更新参数化（μP）扩展到差分隐私设置，通过信噪比条件与谱缩放公式为DP-SGD、DP-Adam和DP-Muon优化器提供了逐层学习率缩放规则，从而实现差分隐私训练中的超参数迁移。 |
| [^22] | [Conformal Policy Selection: When to Deploy AI Decisions for Targeted Intervention](https://arxiv.org/abs/2610.22930) | 该论文提出“共形策略选择”方法，将AI策略的针对性部署建模为多重检验问题，通过加权共形p值与两步选择-剪枝程序在控制错误发现率的前提下识别出能从AI策略中受益的个体，实现了有限样本FDR控制和双重稳健的渐近保证。 |
| [^23] | [Unit-level aggregation breaks coverage in causal foundation models](https://arxiv.org/abs/2610.22770) | 因果基础模型虽然在单元级估计上表现出色，但其可信区间在聚合到单元组（如全样本平均效应）时会严重退化，在ACIC 2016上全样本平均效应的区间仅有20%覆盖真实值。 |
| [^24] | [Optimal approximation of smooth functions by ReLU neural networks](https://arxiv.org/abs/2610.22693) | 本文证明了宽度为O(N)、深度为O(L)的ReLU网络对任意光滑度s和维度D下的Hölder函数类均能达到最优逼近误差$(N^2L^2\log N)^{-s/D}$，并由此导出无额外对数因子的极小极大回归速率。 |
| [^25] | [Is a Linear Probe Evidence of a Linear Representation?](https://arxiv.org/abs/2610.22655) | 该论文证明线性探针的高准确率不足以证明线性表示的存在——即使是随机特征也能实现线性读出——并提出反向预测性、谱集中度等三种非对称诊断方法，以区分真正的线性世界模型与核机器式的弯曲流形表示。 |
| [^26] | [Flat Loss, Evolving Features: Grokking in Modular Addition](https://arxiv.org/abs/2610.22593) | 该论文证明网络可以在训练交叉熵几乎平坦地保持正值的条件下继续学习：在模加法任务上，ReLU网络在保持100%训练准确率的同时隐藏特征持续演化、留出集准确率从零升至90%以上，并通过两个有限窗口定理从理论上证明了这种平坦损失与顿悟式泛化的共存性。 |
| [^27] | [Computing Fr\'echet on Unknown Submanifolds of Riemannian Manifolds](https://arxiv.org/abs/2610.22592) | 该论文提出一种基于图的迭代算法，在仅能访问未知黎曼流形的局部采样器和观测度量的条件下，自适应采样流形并近似测地线插值，从而实现了计算内在Fréchet均值的 $O(1/T)$ 期望目标间隙保证。 |
| [^28] | [Learning smooth dynamics via Kinematic Interpolants](https://arxiv.org/abs/2610.22519) | 该论文提出运动学插值子（KI）框架，将随机插值方法扩展至包含位置与速度测量的相空间多快照数据，从而无需模拟即可可扩展地高保真重建潜在的非线性动力学。 |
| [^29] | [Source-Aware Dynamic Pricing with Strategic Buyers](https://arxiv.org/abs/2610.22516) | 本文提出了SALE定价策略，利用已验证特征进行局部化价格探索并结合自报特征校正策略性操纵，从而解决高维半参数环境下存在策略性买家操纵的来源感知动态定价问题。 |
| [^30] | [Upper bounds on the number of causal pieces in spiking neural networks](https://arxiv.org/abs/2610.22484) | 本文首次为前馈单脉冲神经网络中因果片段的数量推导出两个互补的上界——一个基于多重伯努利数、适用于广泛脉冲神经元模型的组合界，以及一个基于超平面排列的界，从而填补了脉冲神经网络因果片段数量极限研究的空白。 |
| [^31] | [Causal Integrated Gradients: Scalable Feature Attribution Along Causal Paths](https://arxiv.org/abs/2610.22126) | 提出因果积分梯度（CIG）和因果期望梯度（CEG），通过沿因果图路由积分路径来利用特征间已知的因果结构，在保持积分梯度计算效率和公理化保证的同时实现可扩展的特征归因，避免将功劳错误归于下游效应而非真正的上游原因。 |
| [^32] | [Time-partitioning Gaussian Process Mixture Model](https://arxiv.org/abs/2610.20860) | 提出了一种贝叶斯非参数框架TPGPMM，通过自适应Pitman-Yor先验与特定状态高斯过程的结合，自动识别时间序列中的持续状态与变点，无需马尔可夫假设即可建模非平稳依赖，并能有效区分持续变化与瞬态冲击。 |
| [^33] | [Spectral Indistinguishability and Learning-Curve Identifiability in Quadratic Networks](https://arxiv.org/abs/2610.20678) | 该论文构造了谱测度完全一致（谱不可区分）但学习曲线仍可区分的二次网络对，证明谱摘要无法唯一辨识网络，而标量学习曲线可以实现辨识，且其查询复杂度随循环长度呈指数增长。 |
| [^34] | [Follow the Perturbed Leader for Linear Bandits: $\sqrt{T}$ Regret over General Convex Sets](https://arxiv.org/abs/2610.17818) | 本文提出了首个基于FTPL的对抗性线性赌博机算法，通过一个适应估计奖励的扰动分布，在一般凸动作集上实现了 $\sqrt{T}$ 遗憾，改进了此前 $T^{2/3}$ 的遗憾界，同时比连续指数权重方法具有更好的计算效率和更简单的实现。 |
| [^35] | [Cointegrated Regressors: Where Folded Concave Penalties Go Beyond LASSO](https://arxiv.org/abs/2610.14617) | 针对LASSO凸惩罚过度压缩协整回归变量系数而导致参数估计不一致的问题，本文采用折叠凹惩罚（SCAD和MCP）建立了局部与全局oracle性质，证明其能一致估计真实系数并显著改善变量选择和样本外预测效果。 |
| [^36] | [How Well Do Prior-Data Fitted Networks Amortize Bayesian Inference?](https://arxiv.org/abs/2610.13764) | 本文将先验数据拟合网络（PFNs）的部署误差分解为先验、上下文和摊销误差，发现PFNs虽能很好地恢复目标后验预测分布，但在罕见设置下仍存在残余摊销误差，而贝叶斯模型平均方法可有效减少这种不确定性并提升预测性能。 |
| [^37] | [The optimal information complexity of VC learning](https://arxiv.org/abs/2610.10600) | 本文通过构造一个eCMI为O(d)阶的随机化“5个基学习器多数投票”学习算法，首次经由CMI信息论分析框架恢复了VC类学习的最优PAC泛化保证。 |
| [^38] | [Leaner Transformers Can Easily Learn to Cluster](https://arxiv.org/abs/2610.09760) | 本文提出了一种嵌入维度仅需 d+⌈log₂k⌉ 但表达能力不变的更精简Transformer来执行k均值聚类的Lloyd算法，并系统刻画了训练Transformer学习聚类算法时影响收敛性与泛化能力的关键因素。 |
| [^39] | [On the Trade-off Between Information Loss and Generalization in Sparse Attention](https://arxiv.org/abs/2610.04424) | 本文通过Jensen-Shannon散度对稀疏注意力机制的信息损失进行理论刻画，并系统分析了信息损失与模型泛化能力之间的权衡关系。 |
| [^40] | [FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training](https://arxiv.org/abs/2610.00385) | FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。 |
| [^41] | [On the Sample Complexity of Active Learning with Membership Queries](https://arxiv.org/abs/2609.27241) | 本研究揭示了允许合成成员查询会显著改变统计学习的难度——某些在基于池的主动学习下只能实现多项式误差衰减的假设类，在允许合成查询后变得可指数级快速学习，表明成员查询合成是一种需要新分析工具来刻画的根本不同的学习模式。 |
| [^42] | [Online Inference in Distributional Temporal-Difference Learning](https://arxiv.org/abs/2608.14408) | 本文为分布时序差分学习提供了在线推断的理论框架，证明了平均估计器和自助法的渐近正态性，并扩展到了非光滑泛函的局部推断。 |
| [^43] | [Online Inference for Quantile Temporal Difference Learning in Distributional Reinforcement Learning](https://arxiv.org/abs/2608.12973) | 本文提出了一种基于随机缩放的在线推断方法，通过泛函中心极限定理支持，在不存储完整轨迹的情况下对分布强化学习中的QTD进行统计推断，显著减少了内存需求。 |
| [^44] | [Dynamics of Stochastic Momentum with Sparse Updates in High Dimensions](https://arxiv.org/abs/2605.28961) | 该论文首次在高维稀疏更新场景下精确刻画了动量优化的二阶矩动力学，揭示了其相变结构由动量保持时间尺度与学习时间尺度之比决定：学习慢于保持时极限等价于SGD，学习快于保持时系统失稳。 |
| [^45] | [Human-Centered Learning Mechanics: Observable Entropy Control under Protected-Risk Constraints](https://arxiv.org/abs/2605.22940) | 本文建立了在任务、风险与干预约束分离设定下的熵调控学习理论，通过噪声校准的对数行列式泛函导出精确谱力公式、可观测/预测零空间速度分解、标量反馈系数完整区间，以及多受保护风险下的极小极大可行性证书与锥投影公式，实现受保护风险约束下的可观测熵控制。 |
| [^46] | [Diffusion-Based Stochastic Operator Networks for Uncertainty Quantification in Stochastic Partial Differential Equations](https://arxiv.org/abs/2605.17107) | 该论文提出了随机算子网络（SON），通过将DeepONet与随机神经网络及扩散模型相结合，直接从含噪数据中学习，为随机偏微分方程的解同时提供均值预测和不确定性量化。 |
| [^47] | [Regime-Conditioned Evaluation in Multi-Context Bayesian Optimization](https://arxiv.org/abs/2605.04895) | 该论文揭示了迁移贝叶斯优化中采集函数的排名会随预算比例和先验质量等状态变量而逆转（98% 的已发表比较从未控制这一变量），并提出可用先验信息估计的可移植状态得分 PRS=(B/|A|)(1-ρ) 来预测这种排名转变，从而主张以状态条件化评估取代平均效应比较。 |
| [^48] | [Empowering Bayes with sources of uncertainty: from confidence modelling to sparse estimation](https://arxiv.org/abs/2605.03134) | 该论文提出不确定性来源（SoU）推断框架，通过在模型组件间显式编码异质置信度来扩展贝叶斯推断，使置信度与先验信息量可分别控制，并据此开发了诱导稀疏性的通用方法，可应用于线性/逻辑回归及贝叶斯神经网络的自剪枝。 |
| [^49] | [Measuring Differences between Conditional Distributions using Kernel Embeddings](https://arxiv.org/abs/2605.02260) | 本文提出了条件最大均值差异（CMMD）这一统一理论框架，通过不同层级的核嵌入度量方法，系统性地解决了条件分布之间差异度量缺乏统一理论处理的问题。 |
| [^50] | [2D Stability Selection: Design Jittering to Exclude Correlated Irrelevant Features](https://arxiv.org/abs/2605.02205) | 提出二维稳定性选择（2D-SS），通过在半样本和注入设计噪声的网格上聚合选择频率，能够排除违反不可表示条件的相关混淆特征，从而修复标准稳定性选择无法控制的假阳性问题。 |
| [^51] | [Adaptive Conditional Residual Sampling for Spectral Risk Optimisation under Decision-Dependent Uncertainty](https://arxiv.org/abs/2603.12507) | 提出自适应条件残差抽样（ACRS）方法，通过复用标准化残差向量构建确定性代理模型，高效求解决策依赖不确定性下的谱风险优化问题，并提供基于Wasserstein误差的最优性保证。 |
| [^52] | [Structural Causal Bottleneck Models](https://arxiv.org/abs/2603.08682) | 该论文提出结构因果瓶颈模型（SCBMs），用低维“瓶颈”统计量传递高维变量间的因果效应，证明了瓶颈变量在双射意义下可从观测数据识别，并表明瓶颈表示能够恢复工具变量方法的可识别性、并支持中介分析与后门调整下的有效因果效应估计。 |
| [^53] | [Accelerated Execution of Bayesian Neural Networks using a Single Probabilistic Forward Pass and Code Generation](https://arxiv.org/abs/2511.23440) | 该论文通过假设权重和激活服从高斯分布，用单次解析式的概率前向传播取代多次采样前向传播来加速贝叶斯神经网络，并构建了在嵌入式ARM CPU上训练、编译、优化和部署的端到端流水线。 |
| [^54] | [Robust Experimental Design via Generalised Bayesian Inference](https://arxiv.org/abs/2511.07671) | 本文提出广义贝叶斯最优实验设计（GBOED），通过用损失函数替代似然函数，在模型设定存在偏差时同时保证实验设计与统计推断的鲁棒性。 |
| [^55] | [SLIM: Stochastic Learning and Inference in Nonlinear GMM Models](https://arxiv.org/abs/2510.20996) | 提出了SLIM这一可扩展的随机近似算法，无需一致初始估计量或全局凸准则，即可对具有大量矩条件和参数的非线性GMM模型进行一致且渐近正态的估计与推断，并可扩展至百万级观测数据和聚类数据场景。 |
| [^56] | [Reliable data clustering with Bayesian community detection](https://arxiv.org/abs/2510.15013) | 该论文提出一种一步式聚类方法，利用贝叶斯社区检测和单一最小描述长度准则同时完成数据稀疏化与聚类，免除了任意阈值选择，在更高噪声和更少样本条件下优于传统两步法，且在无真实结构时能正确返回单一聚类。 |
| [^57] | [The Statistical Fairness-Accuracy Frontier](https://arxiv.org/abs/2508.17622) | 该论文在有限样本条件下研究了单一预测模型服务多个人口群体时的公平性-准确性权衡，推导了最坏情况最优估计量，刻画了有限样本对各群体福利的不对称影响，给出了最优样本分配策略，并提供了沿整个FA帕累托前沿的一致有限样本界。 |
| [^58] | [Reinforcement Learning in MDPs with Information-Ordered Policies](https://arxiv.org/abs/2508.03904) | 本文提出策略类上的“信息序”概念，通过反事实推断为连续状态动作空间的无限时域平均成本MDP设计了基于epoch的强化学习算法，取得了由信息序宽度 $w$ 决定的 $\widetilde O(H\sqrt{wdT/\alpha})$ 遗憾界，从而利用策略间数据的可迁移性降低了探索复杂度。 |
| [^59] | [Missing value imputation with adversarial random forests - MissARF](https://arxiv.org/abs/2507.15681) | MissARF是一种基于对抗性随机森林的快速缺失值插补方法，可在提供高质量单重与多重插补的同时，使多重插补无需额外计算成本。 |
| [^60] | [Closed-form Covariance Functions in Space and Time from B-spline Spectral Densities](https://arxiv.org/abs/2506.21953) | 本文通过推导任意阶数和节点配置的B样条谱基的逆傅里叶变换，构建了一族闭式的非参数平稳协方差函数，该族模型在具有谱密度的弱平稳过程的自协方差函数空间中一致稠密，并能适用于单变量、多变量及多维时空数据。 |
| [^61] | [Near-optimal estimates for the $\ell^p$-Lipschitz constants of deep random ReLU neural networks](https://arxiv.org/abs/2506.19695) | 本文证明了遵循He初始化变体的随机ReLU神经网络的 $\ell^p$-Lipschitz 常数的近最优估计，揭示了该常数在 $p \in [2,\infty]$ 时类似于高斯向量的 $\ell^{p'}$ 范数，而在 $p \in [1,2)$ 时则更接近高斯向量的 $\ell^2$ 范数。 |
| [^62] | [RAW-Explainer: Post-hoc Explanations of Path Reasoning Graph Neural Networks for Knowledge Graph Completion](https://arxiv.org/abs/2506.12558) | 本文提出RAW-Explainer，一种专为路径推理图神经网络设计的新型可解释性方法，通过随机游走目标函数和异构信息，为知识图谱链接预测生成连通、简洁且可解释的子图解释，并解决了评估小子图时的分布偏移问题。 |
| [^63] | [What is causal about causal models and representations?](https://arxiv.org/abs/2501.19335) | 该论文提出了一个将因果模型与数据生成过程分离的形式化框架，形式化了连接行动与干预的解释，并证明了不存在能同时满足一组期望性质且保持非循环性的行动-干预解释这一不可能性结果。 |
| [^64] | [Is the Policy Gradient a Gradient?](https://arxiv.org/abs/1906.07073) | 本文证明了省略折扣因子的策略梯度更新不必是任何函数的梯度，并构造反例表明该更新会使策略收敛到在折扣与未折扣目标下均全局最差的确定性极限。 |

# 详细

[^1]: 深度条件生成学习：模型与误差分析

    Deep Conditional Generative Learning: Model and Error Analysis

    [https://rss.arxiv.org/abs/2402.01460](https://rss.arxiv.org/abs/2402.01460)

    提出了一种基于ODE的深度生成方法，通过条件Follmer流来学习条件分布，通过离散化和深度神经网络实现高效转化。同时，通过Wasserstein距离的非渐近收敛速率，提供了第一个端到端误差分析，数值实验证明其在不同场景下的优越性。

    

    我们介绍了一种基于普通微分方程（ODE）的深度生成方法，用于学习条件分布，称为条件Follmer流。从标准高斯分布开始，所提出的流能够以高效的方式将其转化为目标条件分布，在时间1处达到稳定。为了有效实现，我们使用欧拉方法对流进行离散化，使用深度神经网络非参数化估计速度场。此外，我们导出了学习样本的分布与目标分布之间的Wasserstein距离的非渐近收敛速率，在条件分布学习中提供了第一个全面的端到端误差分析。我们的数值实验展示了它在一系列情况下的有效性，从标准的非参数化条件密度估计问题到涉及图像数据的更复杂的挑战，说明它优于各种现有方法。

    We introduce an Ordinary Differential Equation (ODE) based deep generative method for learning a conditional distribution, named the Conditional Follmer Flow. Starting from a standard Gaussian distribution, the proposed flow could efficiently transform it into the target conditional distribution at time 1. For effective implementation, we discretize the flow with Euler's method where we estimate the velocity field nonparametrically using a deep neural network. Furthermore, we derive a non-asymptotic convergence rate in the Wasserstein distance between the distribution of the learned samples and the target distribution, providing the first comprehensive end-to-end error analysis for conditional distribution learning via ODE flow. Our numerical experiments showcase its effectiveness across a range of scenarios, from standard nonparametric conditional density estimation problems to more intricate challenges involving image data, illustrating its superiority over various existing condition
    
[^2]: ENFR：基于流匹配参考的证据网络用于摊销式基于仿真的模型推断

    ENFR: Evidence Networks with Flow-Matched References for Amortized Simulation-Based Model Inference

    [https://arxiv.org/abs/2610.23825](https://arxiv.org/abs/2610.23825)

    本文提出ENFR方法，通过结合证据网络与流匹配参考技术，实现对贝叶斯证据的准确、快速且摊销式的估计，从而突破仅衡量证据比值的贝叶斯因子方法的局限，为高维拟合优度检验和模型误设检验提供直接解决方案。

    

    证据 $\mathcal{Z} = p(x | \mathcal{M})$，即数据生成模型 $\mathcal{M}$ 的证据，是贝叶斯框架中的一个强大工具：它能够实现模型推断、模型误设检验和拟合优度评估，可作为标准差异统计量（如 $\chi^2$）的替代方案。尽管用途广泛，证据的计算却是出了名的困难，即使似然函数可用时也是如此。无论是在基于似然的推断还是基于仿真的推断中，估计 $\mathcal{Z}$ 都能为高维拟合优度检验或模型误设检验提供直接的解决方案：使用 $\mathcal{Z}(x)$ 作为检验统计量可诱导出最小体积的 $\alpha$ 水平检验。此类检验对于评估模型保真度至关重要，而模型保真度是信任基于仿真的参数推断的必要条件。现有的学习方法主要针对贝叶斯因子，而贝叶斯因子只能衡量证据的比值。与之相反，我们提供了对……的准确、快速且摊销式的估计（原文摘要在此处截断）

    arXiv:2610.23825v1 Announce Type: cross  Abstract: The evidence, $\mathcal{Z} = p(x | \mathcal{M})$, of a data-generating model $\mathcal{M}$ is a powerful tool in Bayesian frameworks: it enables model inference, model misspecification tests, and goodness-of-fit assessment as an alternative to standard discrepancy statistics, e.g. $\chi^2$. While versatile, the evidence is notoriously difficult to compute, even when the likelihood is available. In both likelihood-based and simulation-based inference, estimating $\mathcal{Z}$ provides an immediate solution to high-dimensional goodness-of-fit or model misspecification tests: using $\mathcal{Z}(x)$ as the test statistic induces a minimum-volume level-$\alpha$ test. Such tests are essential to assess model fidelity, which is necessary to trust simulation-based parameter inference. Existing learned approaches largely target Bayes factors, which only measure evidence ratios. Instead, we provide accurate, fast, and amortized estimates of the 
    
[^3]: 无需重新学习物理对称性：球面上的SO(3)等变卷积神经网络（CNN）

    Learning without relearning physical symmetry: an SO(3)-equivariant convolutional network (CNN) on the sphere

    [https://arxiv.org/abs/2610.23821](https://arxiv.org/abs/2610.23821)

    本文提出了一种专为HEALPix球面像素化设计的SO(3)等变卷积神经网络，利用可操纵滤波器、平行移动和自旋加权特征实现可证明的精确等变性，使网络无需消耗训练数据来学习物理对称性。

    

    许多物理科学中最重要的数据集天然存在于球面上，且这些数据通常不带有优先的位置或取向。使用不反映这种对称性的方法会导致高得多的模拟需求，这对科学分析而言可能是灾难性的（迫使人们依赖廉价且不完善的模型，从而使结果产生偏差）。我们提出了一种新颖的球面规范等变网格架构实现，该架构专为球面的层次等面积等纬度像素化（HEALPix）而设计。SO(3)等变卷积由可操纵滤波器、平行移动和自旋加权特征构建而成。该架构直接提供了精确的对称性，因此网络无需花费训练数据去学习它。在给定频带限制的条件下，卷积运算被构建为具有可证明的精确等变性。我们的完整CNN可以将SO(3)不变性误差控制在……（原文在此截断）

    arXiv:2610.23821v1 Announce Type: cross  Abstract: Many of the most important datasets in the physical sciences live natively on the sphere, and these data often carry no preferred position or orientation. Using methods that do not reflect this symmetry leads to far higher simulation requirements, which can be catastrophic for scientific analyses (forcing reliance on cheap, imperfect models that bias results). We present a novel spherical implementation of a gauge equivariant mesh architecture that is specifically designed for the Hierarchical Equal Area isoLatitude Pixelization (\healpix{}) of the sphere. The $\mathrm{SO}(3)$-equivariant convolution is built from steerable filters, parallel transport and spin-weighted features. The architecture supplies the exact symmetry, so the network does not spend training data learning it. The convolution operations are constructed to give provably exact equivariance, given a band-limit. Our full CNNs can achieve SO(3)-invariance error at per-ce
    
[^4]: 面向任务的核Stein证书用于Rényi差分隐私

    Task-aware Kernel Stein certificates for R\'{e}nyi differential privacy

    [https://arxiv.org/abs/2610.23798](https://arxiv.org/abs/2610.23798)

    本文提出基于护航分布的核Stein差异方法，通过设计聚焦相关隐私方向的核函数，为Rényi差分隐私提供可计算的隐私证书及面向任务的隐私保证，并给出了高斯与拉普拉斯机制的精确恒等式和指数机制的界。

    

    基于Rényi散度的Rényi差分隐私已成为隐私保证领域的重要概念。然而，在高斯设定之外，Rényi散度可能难以计算。特别是，利用任务特定信息的Rényi差分隐私保证（如Rényi Pufferfish隐私中的情形）可能具有挑战性。本文使用基于护航分布的核Stein差异来为Rényi差分隐私提供证书。通过设计核函数聚焦于相关的隐私方向，我们还推导出了面向任务的隐私证书。我们为高斯机制和拉普拉斯机制提供了精确恒等式，并为指数机制提供了界，同时给出了独立的面向任务保证。一个真实数据集的例子说明了受保护的比较集和发布的子空间如何共同决定所得的证书。

    arXiv:2610.23798v1 Announce Type: cross  Abstract: R\`enyi differential privacy, based on R\`enyi divergence, has become a notion of choice for privacy guarantees. However, outside the Gaussian setting, R\`enyi divergence can be difficult to compute. In particular, R\`enyi differential privacy guarantees that exploit task-specific information, as in R\`enyi Pufferfish privacy, can be challenging. Here we use kernel Stein discrepancies based on escort distributions to provide certificates for R\`enyi differential privacy. By designing the kernel to focus on the relevant privacy directions, we also derive task-aware privacy certificates. We provide exact identities for Gaussian and Laplace mechanisms and bounds for exponential mechanisms, with separate task-aware guarantees. A real data set example illustrates how the protected comparison set and the released subspace jointly determine the resulting certificates.
    
[^5]: 缓慢非平稳约束马尔可夫决策过程原始-对偶学习中的参数共振

    Parametric Resonance in Primal-Dual Learning for Slowly Non-Stationary Constrained MDPs

    [https://arxiv.org/abs/2610.23788](https://arxiv.org/abs/2610.23788)

    本文揭示了缓慢非平稳约束马尔可夫决策过程中原始-对偶学习的不稳定性并非源于非平稳性本身，而是源于约束估计与测量失配所引发的参数共振，并定量给出了可容忍失配随强迫带宽的平方根增长、随对偶-原始步长比的平方根缩减的规律。

    

    本文在双时间尺度假设下分析了约束马尔可夫决策过程的原始-对偶方法，其中非平稳性通过两个渠道进入系统。在加性渠道中，约束水平发生振荡，对偶变量的振荡会饱和于由移动最优解所设定的下限，并被时间尺度比线性放大。在参数渠道中，约束灵敏度发生振荡，相干调制在任何深度下均不携带共振强迫：对偶更新的刚度调制恰好抵消了原始更新的阻尼调制。不稳定性反而来源于原始更新所使用的约束估计与对偶更新所使用的测量之间的失配，从而在系统对的固有频率的两倍处产生参数共振。可容忍的失配随强迫带宽的平方根而增大，并随对偶与原始步长比的平方根而缩减。这些结果……

    arXiv:2610.23788v1 Announce Type: cross  Abstract: Primal-dual methods for constrained Markov decision processes are analyzed under a two-timescale assumption, with non-stationarity entering through two channels. In the additive channel, where the constraint level oscillates, the dual oscillation saturates at a floor set by the moving optimum and is amplified linearly by the timescale ratio. In the parametric channel, where the constraint sensitivity oscillates, a coherent modulation carries no resonant forcing at any depth: the dual update's stiffness modulation cancels the primal update's damping modulation. Instability instead comes from a mismatch between the constraint estimate used by the primal update and the measurement used by the dual update, producing a parametric resonance at twice the pair's natural frequency. The tolerated mismatch grows with the square root of the forcing bandwidth and collapses with the square root of the dual-to-primal step-size ratio. These results ho
    
[^6]: 广宽度神经网络的介观学习动力学

    Mesoscopic learning dynamics of extensive-width neural networks

    [https://arxiv.org/abs/2610.23779](https://arxiv.org/abs/2610.23779)

    该论文通过刻画重叠量的分布，提出了一种介观描述方法，将广宽度两层神经网络学习任意非线性目标函数的动力学约化为少数集体变量的演化，从而克服了传统方法需要追踪发散数量序参量的难题。

    

    理解随机梯度下降如何在宽神经网络中学习内部表示（这些网络在丰富的目标函数上进行训练）是机器学习中的一个核心问题。我们在两层神经网络中研究这一问题，其中目标函数由具有相同架构的网络表示，非线性形式任意，且网络宽度与输入维度成比例增长。在这种情形下，描述学习动力学的标准方法需要追踪数量发散的重叠量（overlap），这些重叠量衡量了学习到的特征与目标特征之间的相关性，因此无法用少数几个序参量给出明确的描述。在本文中，我们通过刻画重叠量的分布，发展了一种学习的介观描述方法。在学习早期阶段，我们发现重叠统计量的高斯描述能够将学习动力学约化为少数几个集体变量，这些变量追踪着目标函数的学习过程。

    arXiv:2610.23779v1 Announce Type: cross  Abstract: Understanding how stochastic gradient descent learns internal representations in wide neural networks trained on rich target functions is a central problem in machine learning. We study this question in two-layer neural networks learning target functions represented by networks of the same architecture, with arbitrary nonlinearity, and with network width and input dimension growing proportionally. In this regime, standard approaches for describing the learning dynamics require tracking a diverging number of overlaps, which measure correlations between learned and target features, leaving no obvious description in terms of a few order parameters. Here, we develop a mesoscopic description of learning by characterizing the distribution of overlaps. In the early stages of learning, we find that a Gaussian description of overlap statistics reduces the learning dynamics to a few collective variables that track learning of the target function
    
[^7]: 具有无限方差梯度的非线性随机梯度下降（SGD）的精确尾部律

    Exact Tail Laws for Nonlinear SGD with Infinite-Variance Gradients

    [https://arxiv.org/abs/2610.23750](https://arxiv.org/abs/2610.23750)

    该论文为无限方差梯度噪声下的非线性恒定步长 SGD 最后一次迭代建立了精确的一阶尾部概率律，证明其尾部由单个大梯度冲击经局部 Hessian 流传输所主导，并由此给出显式的投影与超额目标尾部及极端分位数。

    

    当梯度噪声具有无限方差时，即使大多数迭代点已非常接近最优解，SGD 的期望超额目标仍可能为无穷。我们通过非线性、恒定步长 SGD 最后一次迭代的精确一阶尾部概率来刻画其精度。对于指数为 $t\in(1,2)$ 的正则变化新息，主导测度将单个大的梯度冲击沿局部 Hessian 流传输，这正是支配中心稳定 Ornstein–Uhlenbeck 极限的同一测度；我们证明了在容差收缩至中心尺度之上时该测度的相对精度，且允许状态依赖噪声和不断增长的观测时域。对于每一个可用的梯度矩 $p>1$，容许窗口均非空。主要技术步骤是中心波动尺度处的矩界以及对非线性误差的低阶矩控制。该结果给出了显式的投影尾部与超额目标尾部、极端分位数，以及一个因子 $b^{1-t}$。

    arXiv:2610.23750v1 Announce Type: cross  Abstract: With infinite-variance gradient noise, SGD can have infinite expected excess objective even when most iterates are close to the optimum. We characterize its accuracy through exact first-order tail probabilities for the last iterate of nonlinear, constant-stepsize SGD. For regularly varying innovations of index $t\in(1,2)$, the leading measure transports one large gradient shock through the local Hessian flow. This is the same measure that governs the central stable Ornstein--Uhlenbeck limit; we prove its relative accuracy at shrinking tolerances above the central scale, with state-dependent noise and a growing observation horizon. The admissible window is nonempty for every available gradient moment $p>1$. The main technical step is a moment bound at the central fluctuation scale and a lower-moment control of the nonlinear error. The result yields explicit projection and excess-objective tails, extreme quantiles, and a factor $b^{1-t}$
    
[^8]: 随机机器学习实验的等效性检验与运行次数设计

    Equivalence testing and run-count design for stochastic machine-learning experiments

    [https://arxiv.org/abs/2610.23700](https://arxiv.org/abs/2610.23700)

    本文提出了一套基于两级（种子×样本）方差模型和TOST等效性检验的严格认证方案，并给出将预设容差与目标功效换算为所需运行次数的闭式公式，从而取代以往仅凭两三次运行无显著差异就宣称方法等效的不可靠做法。

    

    关于两种训练方法表现相当的论断，往往仅建立在每种方法两到三次运行之间无显著差异的基础上。我们开发了一套用于认证单次运行指标（如AUROC，受试者工作特征曲线下面积）实际等效性的方案及其功效理论。该方案建立在一个两级（种子×样本）方差模型之上，将运行间离散度与样本级噪声分离开来。预设容差范围内的等效性通过两次单侧检验（TOST）进行验证，等价地也可以通过位于容差范围内的置信区间来验证；两级自助法（bootstrap）可覆盖复合统计量。一个闭式公式可将容差和目标功效转换为每种方法所需的运行次数；其移位t分布的改进版本则根据实际的两级过程进行校准。在270个单元网格上以及非高斯种子分布规律下的仿真实验证实，该公式在其设计相关范围内（从每种方法五次运行开始）是有效的。对于四次已发表的联合……（摘要在此处被截断）

    arXiv:2610.23700v1 Announce Type: new  Abstract: Claims that two training methods perform equally well often rest on a non-significant difference between two or three runs per method. We develop a protocol, with its power theory, for certifying practical equivalence in per-run metrics such as the AUROC (area under the receiver operating characteristic curve). It rests on a two-level (seed x item) variance model separating between-run dispersion from item-level noise. Equivalence within a pre-specified margin is tested by two one-sided tests (TOST), equivalently by a confidence interval inside the margin; a two-level bootstrap covers composite statistics. A closed-form formula converts a margin and a target power into runs per method; its shifted-t refinement is calibrated against the realised two-level process. Simulations on a 270-cell grid and under non-Gaussian seed laws confirm the formula within its design-relevant range, from five runs per method. For four published runs of a joi
    
[^9]: 无遗憾语言生成

    No-Regret Language Generation

    [https://arxiv.org/abs/2610.23641](https://arxiv.org/abs/2610.23641)

    本文提出无遗憾生成框架，用基于树的错误强制维度精确刻画生成任务的极小极大复杂度，揭示所有可容许生成率均可达成，并将其与在线预测联系起来，得到由Littlestone维度决定的错误界和遗憾界。

    

    我们提出“无遗憾生成”这一概念，它要求生成器在所有示范序列和兼容的目标语言上均匀地产生次线性数量的错误。我们还提出了一个不可知变体，允许使用可能与任何语言都不兼容的任意示范。对于这两种设定，我们引入了基于树的错误强制维度，它精确刻画了有限视野下的极小极大复杂度，并建立了无遗憾生成的充分必要条件。我们的刻画揭示了一个丰富的生成率谱系：在常数尺度调整下，每一种可容许的非递减曲线在可实现和不可知两种设定中都是可以达成的。最后，我们将生成复杂度与在线预测联系起来：对于Littlestone维度为有限值 d 的语言类，我们获得了 O(d log(1+T/d)) 的可实现错误界和 O(√(dT log(1+T/d))) 的不可知遗憾界。

    arXiv:2610.23641v1 Announce Type: new  Abstract: We introduce no-regret generation, which requires a generator to make a sublinear number of mistakes uniformly over all demonstration sequences and compatible target languages. We also formulate an agnostic variant that allows arbitrary demonstrations that may not be compatible with any languages. For both settings, we introduce tree-based mistake-forcing dimensions that exactly characterize finite-horizon minimax complexity and establish necessary and sufficient conditions for no-regret generation. Our characterizations reveal a rich spectrum of generation rates: up to constant rescaling, every admissible nondecreasing profile is attainable in both the realizable and agnostic settings. Finally, we connect generation complexity to online prediction: for language classes of finite Littlestone dimension \(d\), we obtain realizable mistake bounds of \(O(d\log(1+T/d))\) and agnostic regret bounds of \(O(\sqrt{dT\log(1+T/d)})\). The realizabl
    
[^10]: 高维点云的边界检测

    Boundary Detection for High-Dimensional Point Clouds

    [https://arxiv.org/abs/2610.23590](https://arxiv.org/abs/2610.23590)

    本文提出两种适用于高维、非均匀密度且含噪点云数据的新型边界检测方法，可同时输出边界分数和法向量估计，并在肿瘤边界检测与LLM激活几何分析中得到验证。

    

    我们提出了两种用于高维点云数据边界检测的新方法，旨在适用于密度不均匀、环境维度高且噪声较大的情况。第一种方法在进行局部降维后，于局部邻域上拟合一个分离超平面——当数据的内在维度可以被可靠估计时，该方法效果最佳。第二种方法通过测量局部邻域的方向均匀性来进行检测，在合成数据上取得了总体最优的表现。除了给出逐点的数值边界分数外，我们的算法还能在每个边界点处输出估计的法向量。我们在合成数据以及两个真实案例上验证了该方法：我们分析了空间转录组学数据，证明了能够检测肿瘤的边界；我们还研究了大型语言模型（LLM）的激活几何。

    arXiv:2610.23590v1 Announce Type: new  Abstract: We introduce two new methods for boundary detection in high-dimensional point cloud data, intended to work in situations with non-uniform density, high ambient dimension, and substantial noise. Our first method fits a separating hyperplane on local neighborhoods after doing local dimension reduction---this method works best when one can reliably estimate the intrinsic dimension of the data. Our second method measures the directional uniformity of local neighborhoods, and has the best overall performance on synthetic data. In addition to pointwise numerical boundary scores, our algorithms yield an estimated normal vector at each boundary point. We validate our methodology on synthetic data as well as two real examples: we analyze spatial transcriptomics data, where we show that we can detect the boundary of a tumor, and we also study LLM activation geometry.
    
[^11]: 迭代算法的数据自适应任意时刻有效包络的通用构建方法

    A general recipe for data-adaptive anytime-valid envelopes for iterative algorithms

    [https://arxiv.org/abs/2610.23570](https://arxiv.org/abs/2610.23570)

    该论文提出了一个通用方法，可为一大类迭代算法构建任意时刻有效且数据自适应的停止边界，将误差分析统一归结为对可预测加权噪声和满足时间一致集中性这一单一要求，克服了以往确定性包络过于保守、数据自适应包络只能逐例推导的局限。

    

    大多数针对迭代随机算法的收敛性保证都是在固定迭代次数处陈述的，无法为在数据相关的时刻停止算法提供合理依据。近期的工作构建了能够提供这种依据的任意时刻有效包络，但确定性的包络对于实际停止而言通常过于保守，而现有的数据自适应包络则是针对具体案例逐一推导的，每一种都将特定算法与特定噪声模型及边界构造耦合在一起。本文为一大类迭代算法提供了一个构建此类任意时刻有效、数据自适应边界的通用方法。这些算法中许多算法的性能度量满足一个单步递归关系，这将整个误差分析简化为一个单一的通用要求：对可预测加权的噪声和具有时间一致的集中性。我们通过一个强适应的 sub-ψ 条件将这一要求形式化，并借助连续与离散的混合……（原文摘要至此处截断）

    arXiv:2610.23570v1 Announce Type: cross  Abstract: Most convergence guarantees for iterative stochastic algorithms are stated at a fixed iteration count and do not justify stopping at a data-dependent time. Recent work has constructed anytime-valid envelopes that do, but deterministic envelopes are typically too conservative for practical stopping, and existing data-adaptive ones are derived case-by-case, each coupling a specific algorithm to a specific noise model and boundary construction. This paper provides a general recipe for constructing such anytime-valid data-adaptive boundaries for a large class of iterative algorithms. Performance measures of many of these algorithms satisfy a one-step recursion that reduces the entire error analysis to a single general requirement: time-uniform concentration for predictably weighted sums of noise. We formalize this through a strongly adapted sub-$\psi$ condition, and derive off-the-shelf boundary constructions via continuous and discrete mi
    
[^12]: 超越期望值：基于分布式蒙特卡洛树搜索的风险敏感规划

    Beyond Expected Values: Risk-Sensitive Planning with Distributional Monte-Carlo Tree Search

    [https://arxiv.org/abs/2610.23565](https://arxiv.org/abs/2610.23565)

    该论文提出了CATSO和PATSO两种分布式蒙特卡洛树搜索算法，通过在Q边维护完整的经验分布（而非仅存储均值估计）并结合CVaR与乐观奖励进行动作选择，实现了针对尾部风险的风险敏感规划，并证明了n次rollout后根节点值误差的O(n^{-1/2})理论界。

    

    蒙特卡洛树搜索（MCTS）通常只存储标量均值估计，这使其难以胜任诸如条件风险价值（CVaR）等尾部风险目标。我们提出了带乐观性的分类汤普森采样（CATSO）和带乐观性的粒子汤普森采样（PATSO），这是两种分布式MCTS算法，它们在Q边上维护有限支撑的经验备份分布。CATSO使用分类原子和狄利克雷汤普森采样来表示Q边分布，而PATSO则使用自适应粒子来表示。两种算法均利用汤普森采样的Q边分布的CVaR加上多项式乐观奖励来选择动作，并通过子节点Q边CVaR分数的访问加权平均来传播标量延续值。针对有限深度的嵌套CVaR规划目标，我们证明了在n次rollout后根节点值误差的O(n^{-1/2})界，该界同时上界了风险敏感的简单遗憾。CATSO增加了一个固定网格离散（此处原文被截断）……

    arXiv:2610.23565v1 Announce Type: new  Abstract: Monte-Carlo Tree Search (MCTS) typically stores only scalar mean estimates, which makes it poorly suited for tail-risk objectives such as conditional value-at-risk (CVaR). We introduce Categorical Thompson Sampling with Optimism (CATSO) and Particle Thompson Sampling with Optimism (PATSO), two distributional MCTS algorithms that maintain finite-support empirical backup laws at Q-edges. CATSO represents Q-edge laws with categorical atoms and Dirichlet Thompson sampling, while PATSO represents them with adaptive particles. Both algorithms select actions using the CVaR of a Thompson-sampled Q-edge law plus a polynomial optimism bonus, and propagate scalar continuation values through visit-weighted averages of child Q-edge CVaR scores. For the finite-depth nested CVaR planning target, we prove an $O(n^{-1/2})$ bound on the root value error after $n$ rollouts, which upper-bounds the risk-sensitive simple regret. CATSO adds a fixed-grid discre
    
[^13]: 超越条件独立的因果发现：有向无环图与循环图的可识别性、算法与统计推断

    Causal Discovery beyond Conditional Independence: Identifiability, Algorithms, and Statistical Inference for Directed Acyclic and Cyclic Graphs

    [https://arxiv.org/abs/2610.23543](https://arxiv.org/abs/2610.23543)

    本综述系统梳理了超越条件独立性的现代因果发现方法，提出将识别来源与统计估计相分离的统一框架，涵盖分布不对称、条件矩约束和跨环境变异等手段，并讨论了潜在混淆与循环图下的可识别性、算法与统计推断。

    

    从数据中学习因果结构需要对机制、噪声、观测方案和干预做出假设。历史上，因果发现主要依赖条件独立性。本综述聚焦于利用条件独立性之外的分布性质的现代因果发现方法。我们的组织原则是将识别来源与统计估计量及其计算实现分离开来。这一视角将条件独立方法与分布不对称性、条件矩约束以及跨环境或跨个体的变异联系起来。我们考察了存在潜在混淆与反馈情况下的因果发现，区分了关于观测变量的因果信息与潜在结构的重构，并强调了循环模型所需的额外语义。我们还区分了总体层面的可识别性与频率学派及贝叶斯一致性（摘要在此处被截断）。

    arXiv:2610.23543v1 Announce Type: cross  Abstract: Learning causal structure from data requires assumptions about mechanisms, noises, observation schemes, and interventions. Historically, causal discovery has relied primarily on conditional independence. This review focuses on modern causal discovery methods that exploit properties of distributions beyond conditional independence. Our organizing principle separates the source of identification from the statistical estimator and its computational implementation. This perspective connects conditional-independence methods with distributional asymmetries, conditional-moment restrictions, and variation across environments or individuals. We examine discovery under latent confounding and feedback, distinguishing causal information about observed variables from reconstruction of latent structure and emphasizing the additional semantics required by cyclic models. We also distinguish population identification, frequentist and Bayesian consisten
    
[^14]: 格拉斯曼流形上的自适应保形预测

    Adaptive conformal prediction on the Grassmannian

    [https://arxiv.org/abs/2610.23508](https://arxiv.org/abs/2610.23508)

    该论文提出了一个在格拉斯曼流形上构建局部自适应预测区域的保形预测框架，通过基不变的几何距离和学习到的难度估计器实现几何感知的非一致性度量，在保持有限样本覆盖保证的同时提升了不同难度下的覆盖均匀性。

    

    我们为格拉斯曼值响应的回归问题引入了一个保形预测框架。所提出的方法直接在格拉斯曼流形上以度量球的形式构建预测区域，使用基不变的几何距离作为非一致性度量，并通过学习到的预测难度估计器对相应的非一致性分数进行缩放。这产生了局部自适应、依赖输入的预测区域，这些区域尊重底层流形几何结构，同时在可交换性假设下保留了分裂保形预测的有限样本边际覆盖保证。我们评估了几种格拉斯曼距离度量，并在合成基准和翼型逆向形状设计问题上比较了它们的预测区域效率。结果凸显了几何感知非一致性分数的优势，同时局部自适应提高了不同预测难度级别下的经验覆盖均匀性。

    arXiv:2610.23508v1 Announce Type: cross  Abstract: We introduce a conformal prediction framework for regression with Grassmann-valued responses. The proposed method constructs prediction regions in the form of metric balls directly on the Grassmann manifold, using basis-invariant geometric distances as non-conformity measures and scaling the corresponding non-conformity scores through a learned prediction difficulty estimator. This yields locally adaptive, input-dependent prediction regions that respect the underlying manifold geometry, while retaining the finite-sample marginal coverage guarantee of split conformal prediction under exchangeability. We evaluate several Grassmannian distance metrics and compare their prediction-region efficiency on synthetic benchmarks and an airfoil inverse shape-design problem. The results highlight the advantages of geometry-aware non-conformity scores, while local adaptation improves empirical coverage uniformity across prediction difficulty levels 
    
[^15]: 迈向在线主动学习的最优样本选择调度策略

    Towards optimal selection schedules for online active learning

    [https://arxiv.org/abs/2610.23477](https://arxiv.org/abs/2610.23477)

    该论文在高维在线主动学习设定下证明了最优样本查询策略的本质：贪婪最小化目标增量的短视策略等价于小间隔或中间间隔选择策略，即只标注间隔落在随时间不断收缩的区间内的样本，以在标注预算或“查询代价 Δ + 错误代价”的目标下最优。

    

    我们研究在线线性分类问题，其中源源不断的未标注样本以数据流形式到达，而人工标注代价高昂。我们研究最优查询策略，即规定样本是否以及何时应被标注并用于一步训练，或应被直接丢弃，从而最小化给定的目标。我们聚焦于可实现情形，在高斯设计下、在数据维度与未标注样本数量成比例增长的高维区域中展开分析。我们考虑两个目标：1）在给定标注预算约束下的泛化误差；2）累积代价函数，其中每次分类错误的代价为单位成本，而每次查询的代价为 $\Delta$。我们首先证明，那些在任意时刻贪婪地最小化目标增量的短视策略对应于小间隔或中间间隔策略，即选择那些间隔落在随时间收缩的特定区间内的样本。在此类策略中，w……（摘要原文至此被截断）

    arXiv:2610.23477v1 Announce Type: cross  Abstract: We consider the problem of online linear classification, where a stream of unlabeled samples is available, and annotation is costly. We investigate the optimal query policy, that prescribes whether and when a sample should be labeled and used for a training step, or instead discarded, so as to minimize a given objective. We focus on the realizable case, under Gaussian design, in the high-dimensional regime where data dimension and number of unlabeled samples grow proportionally. We consider two objectives: 1) the generalization error under a labeling budget, and 2) a cumulative cost function where mistakes have unit cost and queries cost $\Delta$. We first show that the myopic policies that greedily minimize at all times the increment in the objective correspond to either small or intermediate-margin policies, that prescribe the selection of samples whose margins fall within certain intervals that shrink with time. Within this class, w
    
[^16]: 基于迁移学习的岭回归中的极小极大最优标度律

    Minimax-optimal Scaling Laws in Ridge Regression via Transfer Learning

    [https://arxiv.org/abs/2610.23337](https://arxiv.org/abs/2610.23337)

    本文在高维岭回归迁移学习框架下建立了风险确定性等价，并证明在共享回归函数和可交换幂律协方差谱的条件下可实现极小极大最优标度律，且最优源正则化仅依赖源数据、具有跨任务普适性。

    

    预训练模型常被跨下游任务复用，这引出一个自然的问题：这种迁移学习过程能否带来显著的性能提升，即体现在标度律方面。我们在高维回归的典型设定中研究这一问题：其中预训练的源模型通过岭回归训练，而微调后的目标模型通过与源估计器之间的 L2 距离进行正则化。首先，我们在同时存在模型偏移和协方差偏移的情形下，为该两步过程的风险建立了确定性等价表示。其次，通过将确定性等价特化到共享回归函数以及可交换的幂律协方差谱的情形，我们识别出一个区域，在该区域中所得到的标度律达到了极小极大最优指数。值得注意的是，在该区域中，最优的源正则化仅依赖于源数据，这表明其具有跨任务的普适性。

    arXiv:2610.23337v1 Announce Type: new  Abstract: Pretrained models are often reused across downstream tasks, motivating the question of whether this transfer learning procedure can provide a substantial performance improvement, namely, in terms of the scaling law. We address this question in the prototypical setting of high-dimensional regression, where (i) a pretrained \emph{source} model is trained via ridge regression, and (ii) a fine-tuned \emph{target} model is regularized via the $L_2$ distance from the source estimator. First, we establish a deterministic equivalent for the risk of this two-step procedure under both model and covariance shift. Next, by specializing the deterministic equivalent to a shared regression function and commuting power-law covariance spectra, we identify a regime where the resulting scaling law achieves the minimax-optimal exponent. Notably, in this regime, the optimal source regularization depends on source data only, indicating its universality across
    
[^17]: 损失函数还是网络架构？竞争风险生存分析中什么更重要

    The Loss or the Architecture? What Matters in Survival Analysis with Competing Risks

    [https://arxiv.org/abs/2610.23322](https://arxiv.org/abs/2610.23322)

    该论文将损失函数与网络架构解耦并独立评估，在20个表格和3个病理图像数据集上发现：在删失加权下直接建模累积发生率的损失函数比选择更好的骨干网络带来更大的收益。

    

    通常情况下，例如在医疗保健和预测性维护中，人们关心的是多个相互竞争的原因中哪一个会最先发生，以及其发生的概率：这就是竞争风险生存分析。相应的深度学习方法往往是作为“捆绑组合”开发的：一个新架构搭配一个新的训练损失函数，仅在一到五个数据集上与少数基线方法进行测试。因此，最先进的结果无法被归因于架构、损失函数还是评估协议。这种耦合并非偶然：由于每个竞争原因对应一条发生率曲线，网络的输出头与可在其上表达的损失函数是绑定在一起的。我们将这两个维度独立地进行变化：三种损失函数（其中两种是严格适当的）搭配四种骨干网络，跨越两种能够将这两个维度区分开的模态：20个表格数据集和3个病理图像数据集。有一种损失函数在两种模态中都胜出：即在删失加权下直接对累积发生率进行建模。它带来的收益是好骨干网络选择的两倍。

    arXiv:2610.23322v1 Announce Type: cross  Abstract: Often, \emph{eg} in healthcare and predictive maintenance, what matters is which of several competing causes will strike first, and with what probability: survival analysis with competing risks. Corresponding deep learning methods are developed as bundles: a new architecture together with a new training loss, tested on one to five datasets against a handful of baselines. Thus, a state-of-the-art result cannot be attributed to the architecture, the loss, or the evaluation protocol. The coupling is not accidental: with one incidence curve per cause, a network's head is tied to the losses expressible on it. We vary the two axes independently, three losses, two of them strictly proper, on four backbones, across two modalities that tease the axes apart: 20 tabular and 3 pathology-image datasets. One loss wins in both: modelling cumulative incidence directly under censoring weighting. It brings twice as much as benefits a good choice of back
    
[^18]: 一类基于Rényi熵的多层感知机Kolmogorov-Arnold几何集中度度量

    A Class of R\'enyi-based Concentration Metrics for Kolmogorov-Arnold Geometry in Multilayer Perceptrons

    [https://arxiv.org/abs/2610.23267](https://arxiv.org/abs/2610.23267)

    本文提出了一族基于Rényi熵的α-RC集中度度量，通过将多层感知机Kolmogorov-Arnold几何中雅可比子式的归一化幅值与均匀分布进行比较，克服了传统参与率指标因最大值依赖网络规模而无法跨架构解读集中度的局限性。

    

    人工智能模型的可解释性已成为核心关注点，尤其是对于多层感知机（MLP）这类黑盒模型。Kolmogorov-Arnold几何最近兴起为一种分析此类网络内部表示的框架，它通过雅可比矩阵的子式来考察隐藏层对输入变化的局部响应。少数子式支配其余子式的程度被称为集中度，此前一直用参与率来衡量。尽管该指标具有一些有用的性质，但其最大值取决于网络规模，因此在不参照架构的情况下，无法将给定数值解读为高或低集中度。为克服这一局限，本文提出了α-RC度量，这是一族基于Rényi的集中度度量，通过将子式的归一化幅值与均匀分布进行比较……（原文摘要在此处截断）

    arXiv:2610.23267v1 Announce Type: cross  Abstract: The interpretability of artificial intelligence models has become a central concern, especially for black-box models such as multilayer perceptrons (MLPs). Kolmogorov-Arnold Geometry has recently emerged as a framework for analysing the internal representations of these networks, by examining how the hidden layer locally responds to changes in the input through the minors of its Jacobian. The degree to which a few of these minors dominate the rest, known as their concentration, has so far been measured with the Participation Ratio. Although this metric has useful properties, its maximum value depends on the size of the network, so a given value cannot be read as high or low concentration without reference to the architecture. To overcome this limitation, in this paper we propose the $\alpha$-RC metrics, a family of R\'enyi-based concentration measures that compare the normalized magnitudes of the minors with the uniform distribution th
    
[^19]: 共形检验鞅的信息几何盲区

    Information-Geometric Blind Spots of Conformal Test Martingales

    [https://arxiv.org/abs/2610.23131](https://arxiv.org/abs/2610.23131)

    该论文揭示了共形检验鞅的信息几何盲区：凡保持符合性函数前推分布不变的分布偏移均无法被检测，并借助 A-等价类与 A-隐秘 I-投影给出该盲区的显式刻画与勾股分解，同时证明当符合性函数取 Neyman-Pearson 似然比时盲区消失。

    

    共形检验鞅（CTM）是通过由符合性函数 A 生成的 p 值来序贯检验可交换性的 e-过程。然而，任何保持前推分布 P∘A⁻¹ 不变的分布偏移都会使 p 值的分布保持不变，从而对所有基于 A 的共形检验鞅而言都是不可察觉的。我们通过 A-等价类 [P]_A 对这一盲区进行形式化，并证明对于任意目标分布 T，在 KL 散度意义下 [P]_A 中存在唯一的最近元素 Q*——即 A-隐秘 I-投影，其具有显式密度 q*(z) = t(z) p_S(A(z))/t_S(A(z))，并可分解为可检测分量与隐秘分量的勾股分解。当 A 取为 Neyman-Pearson 似然比时，隐秘分量消失，由此将我们的结果与最优 e 值设计联系起来。我们通过在高维图像数据上经验性地构造该 I-投影来展示这一理论，通过条件重采样 EMNIST……（原文摘要在此处截断）

    arXiv:2610.23131v1 Announce Type: cross  Abstract: Conformal test martingales (CTMs) are e-processes for sequentially testing exchangeability via p-values produced by a conformity function $A$. However, any distribution shift that preserves the push-forward $P\circ A^{-1}$ leaves the p-value distribution invariant, rendering it invisible to every CTM based on $A$. We formalise this blind spot through the \emph{$A$-equivalence class} $[P]_A$ and show that, for any target distribution $T$, there is a unique closest element $Q^\ast\in[P]_A$ in KL divergence---the \emph{$A$-cryptic I-projection}---with explicit density $q^\ast(z) = t(z)\,p_S(A(z))/t_S(A(z))$ and a Pythagorean decomposition into detectable and cryptic components. When $A$ is the Neyman-Pearson likelihood ratio, the cryptic component vanishes, connecting our result to optimal e-value design. We demonstrate the theory on high-dimensional image data by empirically constructing this I-projection. By conditionally resampling EMN
    
[^20]: 双稀疏尖峰Wigner模型的可区分性与恢复

    Distinguishability and Recovery for a Doubly Sparse Spiked Wigner Model

    [https://arxiv.org/abs/2610.23033](https://arxiv.org/abs/2610.23033)

    本文研究信号与噪声均稀疏的双稀疏尖峰Wigner模型，提出留一二次阈值化、符号同步及谱精炼等算法，在信噪比保持常数阶的条件下实现了稀疏秩一信号的支撑识别、精确恢复与强恢复，并给出了信号稀疏性、噪声稀疏性与计算成本之间的显式权衡刻画。

    

    我们通过多种算法和参数区域，研究具有独立稀疏噪声的尖峰Wigner模型中稀疏秩一信号的可区分性与恢复问题。信号和噪声具有各自独立的稀疏性参数，而信噪比（SNR）在矩阵规模增大时保持常数阶。对于亚高斯噪声以及非零坐标有界远离零的尖峰信号，一种留一二次阈值化程序（Ding等人2024年工作的推广）能够识别信号的支撑集。通过符号同步可对Rademacher尖峰实现精确恢复，通过谱精炼可实现强恢复。二次阈值化的搜索参数给出了信号稀疏性、噪声稀疏性与计算成本之间的显式权衡。我们还通过对角阈值化和二次阈值化研究了向亚高斯尖峰先验的推广。对于具有连续非零噪声的Rademacher尖峰，我们开发了一个可……

    arXiv:2610.23033v1 Announce Type: cross  Abstract: We study distinguishability and recovery of a sparse rank-one signal in a spiked Wigner model with independently sparse noise through a variety of algorithms and parameter regimes. The signal and noise have separate sparsity parameters, while the signal-to-noise ratio (SNR) remains of constant order as the matrix size grows. For sub-Gaussian noise and spikes whose nonzero coordinates are bounded away from zero, a leave-one-out quadratic thresholding procedure (a generalization of Ding et al., 2024) identifies the support. Sign synchronization yields exact recovery for a Rademacher spike, and spectral refinement yields strong recovery. The quadratic thresholding search parameter gives an explicit tradeoff between signal sparsity, noise sparsity, and computational cost. We also study extensions to sub-Gaussian spike priors through diagonal and quadratic thresholding. For Rademacher spikes with continuous nonzero noise, we develop an iden
    
[^21]: 迈向差分隐私优化的超参数迁移

    Towards Hyperparameter Transfer for Differentially Private Optimization

    [https://arxiv.org/abs/2610.22976](https://arxiv.org/abs/2610.22976)

    本文提出DP-μP框架，将最大更新参数化（μP）扩展到差分隐私设置，通过信噪比条件与谱缩放公式为DP-SGD、DP-Adam和DP-Muon优化器提供了逐层学习率缩放规则，从而实现差分隐私训练中的超参数迁移。

    

    缩放定律与超参数调优仍然是训练大规模模型的核心挑战，尤其在差分隐私（DP）条件下，由于噪声注入和逐样本梯度裁剪，优化问题变得更加复杂。虽然最大更新参数化（μP）能够在标准训练中实现零样本超参数迁移，但我们证明了在DP约束下它会失效，原因在于高维噪声与低秩梯度具有截然不同的谱特性。在本工作中，我们提出了DP-μP，这是一个具有理论依据的框架，将μP扩展到了隐私保护场景。通过分析梯度裁剪、噪声注入与特征学习之间的相互作用，我们推导出了一个控制逐层学习率缩放的信噪比（SNR）条件。结合我们提出的谱缩放公式，我们为DP-SGD、DP-Adam和DP-Muon优化器给出了具体的缩放规则。

    arXiv:2610.22976v1 Announce Type: new  Abstract: Scaling laws and hyperparameter tuning remain central challenges in training large-scale models, particularly under differential privacy (DP), where optimization is further complicated by noise injection and per-sample gradient clipping. While Maximal Update Parametrization ($\mu$P) allows for zero-shot hyperparameter transfer in standard training, we show that it fails under DP constraints due to the distinct spectral properties of high-dimensional noise versus low-rank gradients. In this work, we introduce DP-$\mu$P, a theoretically grounded framework that extends $\mu$P to the privacy-preserving setting. By analyzing the interaction between gradient clipping, noise injection, and feature learning, we derive a Signal-to-Noise Ratio (SNR) condition that governs layer-wise learning rate scaling. Combining this with our proposed spectral scaling formulation, we provide specific scaling rules for DP-SGD, DP-Adam, and DP-Muon optimizers. Em
    
[^22]: 共形策略选择：何时部署AI决策以进行针对性干预

    Conformal Policy Selection: When to Deploy AI Decisions for Targeted Intervention

    [https://arxiv.org/abs/2610.22930](https://arxiv.org/abs/2610.22930)

    该论文提出“共形策略选择”方法，将AI策略的针对性部署建模为多重检验问题，通过加权共形p值与两步选择-剪枝程序在控制错误发现率的前提下识别出能从AI策略中受益的个体，实现了有限样本FDR控制和双重稳健的渐近保证。

    

    人工智能（AI）策略正日益被用于在医学、金融、教育和公共服务等领域做出影响重大的决策，然而要负责任地部署这些策略，不仅需要知道一项策略在平均意义上是否有帮助，还需要知道它对谁有帮助。现有的离线策略评估和安全策略学习工具提供的是总体层面的保证，但某些个体仍可能遭受不利的处理效应。我们将针对性部署重新构建为一个多重检验问题：对于每个候选单元，检验其在AI策略下的结果是否会超过用户指定的阈值，并在被选中的集合中控制错误发现率（FDR）。我们的方法“共形策略选择”将能够考虑从行为策略到目标策略分布偏移的加权共形p值，与两步选择-剪枝程序相结合。我们证明了有限样本下的FDR控制，并建立了双重稳健的渐近保证。

    arXiv:2610.22930v1 Announce Type: cross  Abstract: Artificial-intelligence (AI) policies are increasingly used to make consequential decisions in medicine, finance, education, and public services, yet deploying them responsibly requires knowing not only whether a policy helps on average, but for whom. Existing tools for off-policy evaluation and safe policy learning provide population-level guarantees, but some individuals may experience adverse treatment effects. We recast targeted deployment as a multiple-testing problem: for each candidate unit, test whether the outcome under the AI policy would exceed a user-specified threshold, and control the false discovery rate (FDR) among the selected set. Our method, Conformal Policy Selection, combines weighted conformal $p$-values that account for the distributional shift from behavior to target policy with a two-step selection-and-pruning procedure. We prove finite-sample FDR control and establish a doubly robust asymptotic guarantee, wher
    
[^23]: 单元级聚合破坏因果基础模型的覆盖率

    Unit-level aggregation breaks coverage in causal foundation models

    [https://arxiv.org/abs/2610.22770](https://arxiv.org/abs/2610.22770)

    因果基础模型虽然在单元级估计上表现出色，但其可信区间在聚合到单元组（如全样本平均效应）时会严重退化，在ACIC 2016上全样本平均效应的区间仅有20%覆盖真实值。

    

    因果基础模型（CFMs）在处理效应估计方面展现出可喜的成果，其单元级估计在标准基准测试中具有竞争力。然而，许多因果分析是通过在单元组（如全样本、受处理单元或某个子群）上求平均的效应来为决策提供依据的，对于这类决策，只有当CFM的可信区间经过良好校准时才有用。我们检验了这些区间，发现无论是在简单的合成环境中还是在标准基准上，其覆盖率都会退化。例如，在ACIC 2016数据集上，CausalPFN经温度校准的95%区间，一旦对4802个单元中超过30个单元的组进行聚合，其覆盖率便低于名义水平，而全样本平均值的区间仅在20%的数据集中包含真实值。我们将这一问题归因于这些模型的训练方式及其区间的构造方式：它们一次只输出单个单元的预测分布，并在推理时将这些……（原文摘要到此截断）

    arXiv:2610.22770v1 Announce Type: cross  Abstract: Causal foundation models (CFMs) show promising results for treatment-effect estimation, with unit-level estimates that are competitive on standard benchmarks. Many causal analyses, however, inform decisions through effects averaged over groups of units, such as the full sample, the treated units, or a subgroup, and for such decisions a CFM is useful only if its credible intervals are calibrated. We examine these intervals and find that their coverage degrades, both in simple synthetic settings and on standard benchmarks. On ACIC 2016, for example, temperature-calibrated 95% intervals from CausalPFN fall below nominal coverage beyond groups of 30 out of 4802 units, and intervals for the full-sample average contain the true value in only 20% of datasets.   We trace this problem to how these models are trained and how their intervals are formed. They output a predictive distribution for one unit at a time and, at inference time, combine t
    
[^24]: ReLU神经网络对光滑函数的最优逼近

    Optimal approximation of smooth functions by ReLU neural networks

    [https://arxiv.org/abs/2610.22693](https://arxiv.org/abs/2610.22693)

    本文证明了宽度为O(N)、深度为O(L)的ReLU网络对任意光滑度s和维度D下的Hölder函数类均能达到最优逼近误差$(N^2L^2\log N)^{-s/D}$，并由此导出无额外对数因子的极小极大回归速率。

    

    我们研究了全连接ReLU网络对Hölder球 $\mathcal H_1^s([0,1]^D)$ 的一致逼近问题。除了一维Lipschitz情形外，现有构造与信息论效率极限之间仍存在差距。本文对任意 $s>0$ 和 $D\ge1$ 完全弥合了这一差距：宽度为 $O(N)$、深度为 $O(L)$、参数幅值为 $O(1)$ 的网络可达到 $(N^2L^2\log N)^{-s/D}$ 阶的一致逼近误差，且无需对宽度和深度之间的关系作任何假设。该构造采用了新的编码与解码机制，利用目标函数局部泰勒系数的高阶有限差分界。作为统计上的推论，在相应截断网络类上的经验风险最小化器可以达到极小极大回归速率 $n^{-2s/(2s+D)}$，且不含额外的对数因子。论文还对关键的逼近技术进行了详细阐述，涵盖……

    arXiv:2610.22693v1 Announce Type: cross  Abstract: We study uniform approximation of the H\"older ball $\mathcal H_1^s([0,1]^D)$ by fully connected ReLU networks. Except in the one-dimensional Lipschitz case, existing constructions remain separated from the information-theoretic efficiency limit. This article closes the gap for every $s>0$ and $D\ge1$: networks with width $O(N)$, depth $O(L)$, and parameter magnitude $O(1)$ attain uniform approximation error of order $(N^2L^2\log N)^{-s/D}$. No relation between width and depth is required. The construction uses new encoding and decoding mechanisms that exploit higher-order finite-difference bounds for local Taylor coefficients of the target function. As a statistical consequence, empirical risk minimizers over the corresponding clipped network classes attain the minimax regression rate $n^{-2s/(2s+D)}$ without additional logarithmic factors. The paper also gives a detailed exposition of the key approximation techniques, covering both e
    
[^25]: 线性探针是线性表示的证据吗？

    Is a Linear Probe Evidence of a Linear Representation?

    [https://arxiv.org/abs/2610.22655](https://arxiv.org/abs/2610.22655)

    该论文证明线性探针的高准确率不足以证明线性表示的存在——即使是随机特征也能实现线性读出——并提出反向预测性、谱集中度等三种非对称诊断方法，以区分真正的线性世界模型与核机器式的弯曲流形表示。

    

    线性表示假设认为，预训练视觉模型将生成性因素编码为特征空间中可线性解码的方向，并伴随一套标准方法：训练一个线性探针，若准确率高则宣称该因素被表示。我们证明这一测试过于宽松。从渐近意义上说，任何足够丰富的特征映射（包括随机特征映射）都能实现线性读出，因此高探针准确率与两种截然不同的几何结构相容：一种是真正的线性世界模型，其中与潜在变量对齐的子空间主导表示；另一种是核机器，其中潜在变量线性地嵌入在一个大得多的弯曲流形内部。这一区别至关重要，因为与探针正交的弯曲结构对于分布外泛化、鲁棒性和组合迁移都具有重要意义。我们提出三种非对称诊断方法来区分这两种情况：反向预测性、谱集中度以及分布外……（摘要在此处截断）

    arXiv:2610.22655v1 Announce Type: cross  Abstract: The linear representation hypothesis holds that pretrained vision models encode generative factors as linearly decodable directions in feature space, and comes with a standard methodology: train a linear probe, declare a factor represented if accuracy is high. We show this test is too permissive. Asymptotically, any sufficiently rich feature map, including a random one, admits a linear readout, so high probe accuracy is consistent with two very different geometries: a true linear world model, where the latent-aligned subspace dominates the representation, or a kernel machine, where the latents lie linearly inside a much larger curved manifold. This distinction matters because the curved structure orthogonal to the probe is relevant for out-of-distribution generalization, robustness, and compositional transfer. We propose three asymmetric diagnostics that distinguish the two cases: reverse predictivity, spectral concentration, and out-o
    
[^26]: 平坦损失与演化特征：模加法中的顿悟现象

    Flat Loss, Evolving Features: Grokking in Modular Addition

    [https://arxiv.org/abs/2610.22593](https://arxiv.org/abs/2610.22593)

    该论文证明网络可以在训练交叉熵几乎平坦地保持正值的条件下继续学习：在模加法任务上，ReLU网络在保持100%训练准确率的同时隐藏特征持续演化、留出集准确率从零升至90%以上，并通过两个有限窗口定理从理论上证明了这种平坦损失与顿悟式泛化的共存性。

    

    当一个网络的训练交叉熵几乎平坦地停留在正的水平上时，它还能继续学习吗？我们证明它可以。在模加法这一grokking的标准测试平台上，使用supported-Polar隐藏层更新、归一化输出层更新和权重衰减训练的ReLU网络，可以在训练交叉熵相对平坦的情况下保持100%的训练准确率，同时其隐藏特征持续变化，留出集准确率从零上升到90%以上。正确分类并不需要最小化交叉熵，因为交叉熵仍然依赖于预测置信度。我们通过两个基于构造初始化的有限窗口定理，对单隐层网络证明了这种共存性：其中一个定理给出隐藏特征的傅里叶占比的恒定增益，另一个定理则在任意低的正损失下成立，此时在训练数据上重新拟合的读出层会先于网络本身实现泛化。从随机初始化出发的实验在窗口上同样显示了这种共存现象。

    arXiv:2610.22593v1 Announce Type: new  Abstract: Can a network keep learning while its training cross-entropy stays nearly flat at a positive level? We show that it can. On modular addition, a standard testbed for grokking, ReLU networks trained with supported-Polar hidden updates, normalized output-layer updates and weight decay can keep 100% training accuracy while their hidden features change and held-out accuracy rises from zero to above 90%, at relatively flat training cross-entropy. Correct classification need not minimize cross-entropy, which still depends on prediction confidence. We prove this coexistence for one-hidden-layer networks in two finite-window theorems from constructed initializations: one gives a constant gain in the Fourier fraction of the hidden features, and the other holds at arbitrarily low positive loss, where a readout refitted on the training data generalizes before the network. Experiments from random initialization show the same coexistence on a window f
    
[^27]: 在黎曼流形的未知子流形上计算Fréchet均值

    Computing Fr\'echet on Unknown Submanifolds of Riemannian Manifolds

    [https://arxiv.org/abs/2610.22592](https://arxiv.org/abs/2610.22592)

    该论文提出一种基于图的迭代算法，在仅能访问未知黎曼流形的局部采样器和观测度量的条件下，自适应采样流形并近似测地线插值，从而实现了计算内在Fréchet均值的 $O(1/T)$ 期望目标间隙保证。

    

    经验平均值是统计学中的基本原语。在非线性空间上，其推广形式——Fréchet均值——通常不存在闭式解，其计算需要求解一个优化问题。我们研究嵌入在环境度量空间 $(X, d_{\mathrm{obs}})$ 中的未知紧致 $d$ 维黎曼流形 $M_1$ 上所支撑数据的内在Fréchet均值的计算。当观测数据在非欧几里得空间中能被更好地表示时（例如Wasserstein空间中的概率分布），就会出现这类情形。我们假设只能访问 $M_1$ 上的一个局部采样器以及观测度量 $d_{\mathrm{obs}}$，而不了解其内在几何结构。我们提出了一种基于图的迭代算法，该算法自适应地对流形进行采样并近似测地线插值。在温和的假设下，我们建立了使用至多 $\widetilde{...}$（摘要在此截断）的 $O(1/T)$ 期望目标间隙保证。

    arXiv:2610.22592v1 Announce Type: cross  Abstract: Empirical averages are fundamental statistical primitives. On nonlinear spaces, their generalization, the Fr\'echet mean, generally lacks a closed-form solution, and its computation requires solving an optimization problem. We study the computation of intrinsic Fr\'echet means for data supported on an unknown compact $d$-dimensional Riemannian manifold $M_{1}$ embedded in an ambient metric space $(X,d_{\operatorname{obs}})$. Such settings arise when observations are better represented in non-Euclidean spaces, like probability distributions in Wasserstein space. We assume access only to a local sampler on $M_{1}$ and the observation metric $d_{\operatorname{obs}}$, without knowledge of its intrinsic geometry. We propose a graph-based iterative algorithm that adaptively samples the manifold and approximates geodesic interpolation. Under mild assumptions, we establish an $O(1/T)$ expected objective gap guarantee using at most $\widetilde{
    
[^28]: 基于运动学插值子学习光滑动力学

    Learning smooth dynamics via Kinematic Interpolants

    [https://arxiv.org/abs/2610.22519](https://arxiv.org/abs/2610.22519)

    该论文提出运动学插值子（KI）框架，将随机插值方法扩展至包含位置与速度测量的相空间多快照数据，从而无需模拟即可可扩展地高保真重建潜在的非线性动力学。

    

    从真实世界的实验数据中推断动力学本质上是模糊的，因为许多合理的动力系统都可以解释在测量时刻所获得的相同位置观测数据。为了对从单细胞RNA到涡旋动力学乃至鸟群集群等各类应用进行更高保真度的建模，科学观测量日益包含有助于解决这一歧义问题的速度估计信息。然而，如何以可扩展的方式、且无需模拟或不必要的近似来融入这些丰富的信息，一直是一个困难的挑战。在本文中，我们提出了运动学插值子，这是一个新颖的框架，它将开创性的随机插值方法加以扩展，以在相空间的多个快照（即同时包含位置和速度测量的观测量）之间推断非线性动力学。在实验上，我们证明在多个科学数据集上，KI 实现了对潜在动力学的卓越重建。

    arXiv:2610.22519v1 Announce Type: new  Abstract: Inferring dynamics from real-world experimental data is inherently ambiguous, as many plausible dynamical systems can explain the same positional observations at measurement times. For higher fidelity modeling of applications ranging from single-cell RNA to vortex dynamics and even flocking of birds, scientific observables increasingly also include velocity estimates that serve to resolve this issue. However, incorporating this rich information in a scalable manner, free of simulation or unnecessary approximations, has remained a difficult challenge. In this paper, we introduce Kinematic Interpolants (KI), a novel framework that extends the seminal methodology of stochastic interpolants to infer nonlinear dynamics across multiple snapshots in phase space, i.e., observables with position and velocity measurements. Empirically, we demonstrate that across several scientific datasets, KI achieves superior reconstruction of the underlying dyn
    
[^29]: 面向策略性买家的来源感知动态定价

    Source-Aware Dynamic Pricing with Strategic Buyers

    [https://arxiv.org/abs/2610.22516](https://arxiv.org/abs/2610.22516)

    本文提出了SALE定价策略，利用已验证特征进行局部化价格探索并结合自报特征校正策略性操纵，从而解决高维半参数环境下存在策略性买家操纵的来源感知动态定价问题。

    

    在现代商业中，企业和平台通常使用来自不同来源的客户特征来进行个性化定价：一些特征通过卖家记录或第三方数据进行验证，而另一些特征则由买家自行报告，可能被寻求更低价格的买家进行策略性操纵。因此，我们提出了一个新颖的来源感知框架，用于高维半参数设置下具有策略性买家的上下文动态定价，其中买家估值遵循具有未知噪声分布的稀疏线性模型。为了考虑不同特征来源在可操纵性上的差异，同时实现有效的估值学习，我们开发了来源感知的局部化探索与校准利用定价策略，该策略利用已验证的特征来局部化价格探索，并结合已验证特征与自行报告特征在利用阶段校正策略性操纵。我们的局部化探索设计允许……

    arXiv:2610.22516v1 Announce Type: cross  Abstract: In modern commerce, firms and platforms typically use customer features from different sources to personalize prices: some features are verified using seller records or third-party data, while others are self-reported and may be strategically manipulated by buyers seeking lower prices. We therefore introduce a novel source-aware framework for contextual dynamic pricing with strategic buyers in a high-dimensional semiparametric setting, where buyer valuations follow a sparse linear model with an unknown noise distribution. To account for differences in manipulability across feature sources while enabling effective valuation learning, we develop the Source-Aware Localized exploration and calibrated Exploitation (SALE) pricing policy, which draws on verified features to localize price exploration and combines verified and reported features to correct for strategic manipulation during exploitation. Our localized exploration design allows c
    
[^30]: 脉冲神经网络中因果片段数量的上界

    Upper bounds on the number of causal pieces in spiking neural networks

    [https://arxiv.org/abs/2610.22484](https://arxiv.org/abs/2610.22484)

    本文首次为前馈单脉冲神经网络中因果片段的数量推导出两个互补的上界——一个基于多重伯努利数、适用于广泛脉冲神经元模型的组合界，以及一个基于超平面排列的界，从而填补了脉冲神经网络因果片段数量极限研究的空白。

    

    单脉冲神经网络的输入空间可以分解为因果片段，并且对于某些脉冲神经网络，因果片段的数量已被证明可以提供逼近误差的下界。虽然这种分解适用于广泛的脉冲神经元模型，但脉冲神经网络所能实现的因果片段数量的极限仍然未知。在这项工作中，我们推导了前馈单脉冲神经网络中因果片段数量的两个互补上界。首先，我们建立了一个由多重伯努利数给出的组合界，该界适用于一大类脉冲神经元模型，其阶为 $e^{2(L-1)N\log N+\mathcal{O}(N)}$，其中 $L$ 为网络深度，$N$ 为最大层宽。其次，对于一组特定的神经元模型，我们推导了一个基于超平面排列的界。应用后一种框架，我们得到了阶为 $\mathcal{O}\!\left(N^{(4L...}\right)$ 的界（摘要在此处被截断）。

    arXiv:2610.22484v1 Announce Type: cross  Abstract: The input space of single-spike neural networks decomposes into causal pieces, and the number of causal pieces has been shown to provide a lower bound on the approximation error for some spiking neural networks. While this decomposition is applicable to a wide range of spiking neuron models, the limits of how many causal pieces can be realised by spiking neural networks are still unknown. In this work, we derive two complementary upper bounds on the number of causal pieces in feedforward single-spike neural networks. First, we establish a combinatorial bound given by the poly-Bernoulli numbers, which applies to a broad class of spiking neuron models. The bound is of order $e^{2(L-1)N\log N+\mathcal{O}(N)}$, with network depth $L$ and maximum layer width $N$. Second, for a specific set of neuron models, we derive a bound based on hyperplane arrangements. Applying the latter framework, we obtain bounds of order $\mathcal{O}\!\left(N^{(4L
    
[^31]: 因果积分梯度：沿因果路径的可扩展特征归因

    Causal Integrated Gradients: Scalable Feature Attribution Along Causal Paths

    [https://arxiv.org/abs/2610.22126](https://arxiv.org/abs/2610.22126)

    提出因果积分梯度（CIG）和因果期望梯度（CEG），通过沿因果图路由积分路径来利用特征间已知的因果结构，在保持积分梯度计算效率和公理化保证的同时实现可扩展的特征归因，避免将功劳错误归于下游效应而非真正的上游原因。

    

    现有的考虑特征依赖关系的特征归因方法无法扩展到高维场景。我们提出了因果积分梯度和因果期望梯度（CEG），它们对积分梯度方法进行了扩展，以纳入输入特征之间的先验因果结构，同时保持其计算效率，并在因果图满足一定条件时保留其公理化保证。在许多场景中，特征之间的某些因果结构是已知的，例如转录因子因果性地调控靶基因的表达。然而，现有的基于梯度的方法无法利用这种结构，反而将归因功劳归于特征的下游效应而非其真正的上游原因，例如将靶基因排在调控它的转录因子之上。我们的核心思想是通过让积分路径沿因果图行进来嵌入这种因果结构。由此，该方法将每个特征的归因分解为……（摘要在此处截断）

    arXiv:2610.22126v1 Announce Type: cross  Abstract: Existing feature attribution methods that account for feature dependencies do not scale to high dimensions. We propose causal integrated gradients (CIG) and causal expected gradients (CEG), which extend integrated gradients to incorporate prior causal structure among input features while retaining its computational efficiency and, under conditions on the causal graph, its axiomatic guarantees. In many settings, some causal structure between features is already known, for example, that a transcription factor causally regulates a target gene's expression. Yet existing gradient-based methods cannot exploit this structure, instead crediting a feature's downstream effects over its true upstream cause, such as ranking the target gene above the transcription factor that regulates it. Our key idea is to embed this causal structure by routing the integration path along the causal graph. The method thereby decomposes each feature's attribution i
    
[^32]: 时间划分高斯过程混合模型

    Time-partitioning Gaussian Process Mixture Model

    [https://arxiv.org/abs/2610.20860](https://arxiv.org/abs/2610.20860)

    提出了一种贝叶斯非参数框架TPGPMM，通过自适应Pitman-Yor先验与特定状态高斯过程的结合，自动识别时间序列中的持续状态与变点，无需马尔可夫假设即可建模非平稳依赖，并能有效区分持续变化与瞬态冲击。

    

    我们提出了时间划分高斯过程混合模型（TPGPMM），这是一个用于识别时间序列中持续状态并建模非平稳依赖关系的贝叶斯非参数框架。该模型将连续分段上的自适应Pitman-Yor先验与特定状态的高斯过程相结合，而不施加状态内的马尔可夫依赖。通过学习Pitman-Yor折扣参数，模型能够适应同质和异质的状态长度。一种带有精确Gibbs边界更新的分裂-合并Metropolis-within-Gibbs采样器可联合学习变点和模型超参数，量化分段及状态内动态的不确定性。使用RBF核和固定的保守观测噪声水平，TPGPMM倾向于稀疏分段，能够区分持续变化与瞬态冲击。仿真实验证明了该模型对状态结构的恢复能力。该方法已应用于CBOE波动率…

    arXiv:2610.20860v1 Announce Type: cross  Abstract: We propose the Time-partitioning Gaussian Process Mixture Model (TPGPMM), a Bayesian nonparametric framework for identifying persistent regimes and modelling nonstationary dependence in time series. The model combines an adaptive Pitman--Yor prior on contiguous segmentations with regime-specific Gaussian processes, without imposing Markovian within-regime dependence. Learning the Pitman--Yor discount parameter allows adaptation to homogeneous and heterogeneous regime lengths.   A split--merge Metropolis-within-Gibbs sampler with exact Gibbs boundary updates jointly learns change points and model hyperparameters, quantifying uncertainty over segmentations and within-regime dynamics. With an RBF kernel and a fixed, conservative observation-noise level, TPGPMM favours sparse segmentations that distinguish persistent shifts from transient shocks. Simulations demonstrate recovery of the regime structure. Applications to the CBOE Volatility 
    
[^33]: 二次网络中的谱不可区分性与学习曲线可辨识性

    Spectral Indistinguishability and Learning-Curve Identifiability in Quadratic Networks

    [https://arxiv.org/abs/2610.20678](https://arxiv.org/abs/2610.20678)

    该论文构造了谱测度完全一致（谱不可区分）但学习曲线仍可区分的二次网络对，证明谱摘要无法唯一辨识网络，而标量学习曲线可以实现辨识，且其查询复杂度随循环长度呈指数增长。

    

    我们研究谱摘要和标量学习曲线能揭示中心化二次网络的哪些信息。我们构造了具有精确高斯单坐标和双坐标边缘分布的解析正输入密度。它们的完整四阶矩算子被一个固定整个对角教师代数的正交映射所交织，因此相应的锚定谱测度完全一致。对于具有非零循环不变量的选定常见教师，由于该映射不保持矩阵乘法，其因子化损失曲线则有所不同。循环长度 $r$ 可以任意大，对于每个固定构造，在训练时间内一致地给出 $O(\varepsilon^r)$ 的对比度。对于给定的网络对，带噪声的标量损失查询需要 $\varepsilon^{-2r}$ 量级的观测，而带标量梯度读出的有界初始化干预则需要 $\varepsilon^{-2}$ 量级的观测。一个等于配对梯度差的共享输入分数，能够达到……

    arXiv:2610.20678v1 Announce Type: cross  Abstract: We study what spectral summaries and scalar learning curves reveal about centered quadratic networks. We construct analytic positive input densities with exactly Gaussian one- and two-coordinate marginals. Their complete fourth-moment operators are intertwined by an orthogonal map fixing the entire diagonal teacher algebra, so the corresponding anchored spectral measures agree. For selected common teachers with a nonzero cycle invariant, their factorized loss curves differ because the map does not preserve matrix multiplication. Cycle length $r$, which can be arbitrarily large, gives $O(\varepsilon^r)$ contrast uniformly in training time for each fixed construction. For a supplied pair, noisy scalar loss queries require order $\varepsilon^{-2r}$ observations, while bounded initialization interventions with scalar gradient readouts require order $\varepsilon^{-2}$. A shared-input score, equal to a paired-gradient difference, attains the
    
[^34]: 线性赌博机的扰动跟随领导者算法：一般凸集上的 $\sqrt{T}$ 遗憾

    Follow the Perturbed Leader for Linear Bandits: $\sqrt{T}$ Regret over General Convex Sets

    [https://arxiv.org/abs/2610.17818](https://arxiv.org/abs/2610.17818)

    本文提出了首个基于FTPL的对抗性线性赌博机算法，通过一个适应估计奖励的扰动分布，在一般凸动作集上实现了 $\sqrt{T}$ 遗憾，改进了此前 $T^{2/3}$ 的遗憾界，同时比连续指数权重方法具有更好的计算效率和更简单的实现。

    

    我们提出了首个基于扰动跟随领导者（Follow-the-Perturbed-Leader, FTPL）的对抗性线性赌博机算法，该算法在一般凸动作集上实现了 $\sqrt{T}$ 遗憾。我们的算法仅使用线性优化预言机即可达到 $\widetilde{\mathcal{O}}(d^{5/4}\sqrt{T})$ 的遗憾；当进一步可获得最优设计探索分布时，遗憾可达到接近最优的 $\widetilde{\mathcal{O}}(d\sqrt{T})$。这改进了 Dani 和 Hayes（2006）的 FTPL 算法在一般凸集上 $T^{2/3}$ 的遗憾，并将 Lévy 等人（2026）在两个特殊动作集（超立方体和欧几里得球）上的 $\sqrt{T}$ 遗憾结果进行了推广。与之前基于连续指数权重的预言机高效线性赌博机算法相比，FTPL 方法通常具有更好的计算效率，且实现起来要简单得多。关键的算法要素是一个能适应估计奖励的扰动分布。

    arXiv:2610.17818v1 Announce Type: cross  Abstract: We give the first adversarial linear bandit algorithm based on Follow-the-Perturbed-Leader (FTPL) that achieves $\sqrt{T}$ regret over general convex action sets. Our algorithm attains $\widetilde{\mathcal{O}}(d^{5/4}\sqrt{T})$ regret using only linear optimization oracles, and the near-optimal $\widetilde{\mathcal{O}}(d\sqrt{T})$ regret when an optimal-design exploration distribution is further available. This improves on the $T^{2/3}$ regret of the FTPL algorithm of Dani and Hayes (2006) for general convex sets, and generalizes the $\sqrt{T}$ regret of L\'evy et al. (2026) over two special action sets (hypercube and Euclidean ball). Compared to previous oracle-efficient linear bandit algorithms based on continuous exponential weights, the FTPL approach generally has better computational efficiency, and is much simpler to implement.   The key algorithmic element is a perturbation distribution that adapts to the estimated rewards. Its 
    
[^35]: 协整回归变量：折叠凹惩罚超越LASSO之处

    Cointegrated Regressors: Where Folded Concave Penalties Go Beyond LASSO

    [https://arxiv.org/abs/2610.14617](https://arxiv.org/abs/2610.14617)

    针对LASSO凸惩罚过度压缩协整回归变量系数而导致参数估计不一致的问题，本文采用折叠凹惩罚（SCAD和MCP）建立了局部与全局oracle性质，证明其能一致估计真实系数并显著改善变量选择和样本外预测效果。

    

    LASSO是高维回归模型中最流行的正则化方法之一。然而，在包含大量回归变量的预测回归的实际应用中，协整关系十分常见，而LASSO的凸惩罚会过度压缩协整回归变量的系数，因此无法实现一致的参数估计。本文采用折叠凹惩罚（FCP），包括著名的SCAD和MCP，来解决这一问题。基于FCP，我们建立了两个渐近结果：(i) 惩罚回归的局部oracle性质，以及 (ii) 带有惩罚水平约束的回归的全局oracle性质。特别是，后者保证了一个全局解，该解能够一致地估计真实系数并具备oracle预测性能。模拟研究以及预测欧元区失业率的实际应用表明，FCP在变量选择和样本外预测方面均有改进。

    arXiv:2610.14617v1 Announce Type: cross  Abstract: LASSO is one of the most popular regularization methods for high-dimensional regression models. While cointegration is common in practical applications of predictive regressions with many regressors, the convex penalty of LASSO overly suppresses the coefficients of cointegrated regressors and therefore fails to deliver consistent parameter estimation. We employ folded concave penalties (FCP), including the well-known SCAD and MCP, to address this issue. Using FCP, we establish two asymptotic results: (i) a local oracle property for the penalized regression, and (ii) a global oracle property for regressions with a penalty-level constraint. In particular, the latter guarantees a global solution that consistently estimates the true coefficients and enjoys oracle prediction performance. Simulation studies and an application to predicting the euro area unemployment rate show that FCP improves variable selection and out-of-sample prediction 
    
[^36]: 先验数据拟合网络能多好地摊销贝叶斯推断？

    How Well Do Prior-Data Fitted Networks Amortize Bayesian Inference?

    [https://arxiv.org/abs/2610.13764](https://arxiv.org/abs/2610.13764)

    本文将先验数据拟合网络（PFNs）的部署误差分解为先验、上下文和摊销误差，发现PFNs虽能很好地恢复目标后验预测分布，但在罕见设置下仍存在残余摊销误差，而贝叶斯模型平均方法可有效减少这种不确定性并提升预测性能。

    

    先验数据拟合网络（PFNs）被训练用于近似合成先验的贝叶斯后验预测分布（PPD），然而关于其近似质量的研究结果却出人意料地稀少。从经验风险最小化的视角出发，我们将PFNs的部署误差分解为先验误差、上下文误差和摊销误差，并在现实先验下相对于参考后验预测分布来度量摊销误差。结果表明，PFNs通常能很好地恢复目标后验预测分布。然而，我们发现部分残余摊销误差仍然存在，尤其是在先验预训练分布中的罕见设置下，即使扩大模型规模也无法消除，这表现为独立预训练的PFNs之间的分歧，并在推理时给预测器留下不确定性。作为补救措施，我们研究了多种类型的贝叶斯模型平均方法，发现这些方法能够持续减少这种不确定性并改善预测性能。

    arXiv:2610.13764v1 Announce Type: cross  Abstract: Prior-data fitted networks (PFNs) are trained to approximate the Bayesian posterior predictive distribution (PPD) of a synthetic prior, yet results about the approximation quality are surprisingly thin. Taking an empirical risk minimization perspective, we decompose the deployment error of PFNs into prior, context, and amortization error, and measure the latter against a reference PPD under a realistic prior. Results show that PFNs generally recover the targeted PPD very well. Nevertheless, we find that parts of the residual amortization error persist, especially in \textit{rare} settings under the prior pretraining distribution, even when scaling the model size, which manifests in disagreement between independently pretrained PFNs and leaves uncertainty about the predictor at inference time. As a remedy, we study multiple types of Bayesian model averages and find that these can consistently reduce this uncertainty and improve predicti
    
[^37]: VC学习的最优信息复杂度

    The optimal information complexity of VC learning

    [https://arxiv.org/abs/2610.10600](https://arxiv.org/abs/2610.10600)

    本文通过构造一个eCMI为O(d)阶的随机化“5个基学习器多数投票”学习算法，首次经由CMI信息论分析框架恢复了VC类学习的最优PAC泛化保证。

    

    Steinke和Zakynthinou（2020）提出了条件互信息（CMI）框架，利用依赖于算法的信息论量来分析学习算法的信息复杂度。我们研究了其中一种量——评估条件互信息。一个有趣的问题是：能否通过基于CMI的依赖于算法的分析，恢复VC类的最优PAC保证？我们证明了这是可能的：在可实现情形下，我们构造了一个eCMI为O(d)阶的学习算法，从而恢复了该最优保证，其中d是概念类的VC维。特别地，我们的算法是一种随机化的“5个基学习器多数投票”算法，并具有最优的期望泛化保证。

    arXiv:2610.10600v1 Announce Type: cross  Abstract: Steinke and Zakynthinou(2020) introduces the Conditional Mutual Information (CMI) framework of analyzing the information complexity of learning algorithms based on algorithm-dependent information-theoretic quantities. We study one of these quantities, the evaluated Conditional Mutual Information (eCMI). It has been an interesting question whether the optimal PAC guarantee for VC classes can be recovered from the algorithm-dependent analyses via CMI. And we show that it is possible to recover this guarantee by constructing a learning algorithm whose eCMI is of order O(d) in the realizable case, where d is the VC-dimension of the concept class. Specially, our algorithm is a randomized Majority-of-5 base learners with optimal in-expectation generalization guarantee.
    
[^38]: 更精简的Transformer可以轻松学会聚类

    Leaner Transformers Can Easily Learn to Cluster

    [https://arxiv.org/abs/2610.09760](https://arxiv.org/abs/2610.09760)

    本文提出了一种嵌入维度仅需 d+⌈log₂k⌉ 但表达能力不变的更精简Transformer来执行k均值聚类的Lloyd算法，并系统刻画了训练Transformer学习聚类算法时影响收敛性与泛化能力的关键因素。

    

    Transformer具备上下文学习能力，某些已知的学习算法可以在模型的前向传播过程中执行。近期的研究表明，Transformer可以精确执行Lloyd算法，对d维空间中的n个点进行k均值聚类，其嵌入维度为 d_emb = d+k（因此需要大小为(d+k)²的注意力投影矩阵）。在本工作中，我们在这一结果的基础上进行了以下拓展：首先，我们提出了一种表达能力相同但规模更小的Transformer，它能以嵌入维度 d_emb = (d + ⌈log₂ k⌉) 执行Lloyd算法。其次，我们在给定聚类任务分布的条件下训练这些Transformer学习聚类算法，并从理论上刻画和通过实验验证了影响基于随机梯度的学习算法收敛性和分布内泛化能力的因素。最后，我们探究了一般性的聚类能力（摘要此处被截断）。

    arXiv:2610.09760v1 Announce Type: new  Abstract: Transformers have in-context learning capabilities, where some known learning algorithms can be executed in the forward pass through the model. Recent work shows that transformers can exactly perform Lloyd's algorithm for $k$-means clustering with $n$ points in $d$ dimensions with an embedding size $d_{\textsf{emb}} = d+k$ (thus, requiring attention projection matrices of size $(d+k)^2$). In this work, we build upon this result in the following ways: First, we present an equally expressive but smaller transformer that executes Lloyd's algorithm with embedding size $d_{\textsf{emb}} = (d + \lceil \log_2 k \rceil)$. Next, we train these transformers to learn the clustering algorithms given a distribution of clustering tasks, and theoretically characterize and empirically validate the factors affecting the convergence and in-distribution generalization of learning algorithms based on stochastic gradients. Finally, we probe the general clust
    
[^39]: 关于稀疏注意力中信息损失与泛化之间的权衡

    On the Trade-off Between Information Loss and Generalization in Sparse Attention

    [https://arxiv.org/abs/2610.04424](https://arxiv.org/abs/2610.04424)

    本文通过Jensen-Shannon散度对稀疏注意力机制的信息损失进行理论刻画，并系统分析了信息损失与模型泛化能力之间的权衡关系。

    

    为了缓解Transformer的二次复杂度瓶颈，稀疏注意力已成为一项关键技术。尽管稀疏Transformer在实践中取得了广泛的实证成功，但对其理论理解仍然零散。特别是，两个基本问题仍不清楚：(1) 稀疏化如何影响注意力机制的信息保真度？(2) 这种信息损失如何与模型的泛化行为相互作用？为弥合这一差距，本文对稀疏注意力机制的Jensen-Shannon（JS）散度和泛化差距进行了系统性分析。具体而言，我们首先通过JS散度刻画近似误差，并对截断质量alpha进行了基于顺序统计量的集中性分析——其中注意力分数被假定为独立同分布的sub-Gaussian随机变量。

    arXiv:2610.04424v2 Announce Type: replace  Abstract: To mitigate the quadratic complexity bottleneck of the Transformer, sparse attention has emerged as a pivotal technology. Despite the extensive empirical success of sparse Transformers, the theoretical understanding of sparse attention remains fragmented. In particular, two fundamental questions remain unclear: (1) How does sparsification affect the information fidelity of attention mechanisms? (2) How does this information loss interact with the generalization behavior of the model? To bridge this gap, this paper proposes a systematic analysis of the Jensen-Shannon (JS) divergence and of the generalization gap of sparse attention mechanisms. Specifically, we first characterize the approximation error via the JS divergence. Through an order-statistics-based concentration analysis of the truncation mass alpha --- where the attention scores are assumed to be independent and identically distributed sub-Gaussian random variables with par
    
[^40]: FAER：面向语言模型后训练的可审计、效用对齐的轨迹重放

    FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training

    [https://arxiv.org/abs/2610.00385](https://arxiv.org/abs/2610.00385)

    FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。

    

    重放选择器通常依据格式反馈、置信度、新鲜度或响应长度对缓存轨迹进行排序，然而缓存层面的正确性与下游学习器的效用是两个不同的目标。我们形式化了这种“选择到学习”的差距，并提出FAER作为一个可审计的全轨迹重放框架。其中免训练的固定选择器作为协议基线；FAER-UTILITY则是在互不相交的校准块上拟合的学习器感知选择器。归一化梯度对齐被报告为基线方法，而一次性的优化器感知虚拟更新提供了幅度感知的效用面。审计契约会在评估标签加入之前冻结观测字段和重放轨迹。在GSM8K数据集上使用Qwen2.5-1.5B-Instruct模型，匹配学习器的研究显示固定选择器在128次更新下质量达到0.6329，相比之下均匀采样为0.5482，格式反馈为0.6037。仅使用元数据的交叉拟合校准达到0.6...

    arXiv:2610.00385v1 Announce Type: cross  Abstract: Replay selectors often rank cached trajectories by format feedback, confidence, freshness, or response length, although cache-level correctness and downstream learner utility are distinct objectives. We formalize this selection-to-learning gap and introduce FAER as an auditable full-trajectory replay framework. Its training-free fixed selector is a protocol baseline; FAER-UTILITY is the learner-aware selector fitted on disjoint calibration blocks. The normalized gradient alignment is reported as a baseline, while a disposable optimizer-aware virtual update supplies a magnitude-aware utility surface. The audit contract freezes observed fields and replay traces before evaluation labels are joined. On GSM8K with Qwen2.5-1.5B-Instruct, the matched learner study reports quality 0.6329 for the fixed selector, compared with 0.5482 for uniform and 0.6037 for format-feedback under 128 updates. Metadata-only cross-fitted calibration reaches $0.6
    
[^41]: 关于带成员查询的主动学习的样本复杂度

    On the Sample Complexity of Active Learning with Membership Queries

    [https://arxiv.org/abs/2609.27241](https://arxiv.org/abs/2609.27241)

    本研究揭示了允许合成成员查询会显著改变统计学习的难度——某些在基于池的主动学习下只能实现多项式误差衰减的假设类，在允许合成查询后变得可指数级快速学习，表明成员查询合成是一种需要新分析工具来刻画的根本不同的学习模式。

    

    本工作重新审视了主动学习中的一个根本问题：合成任意查询的能力究竟有多强大？与基于池的主动学习相比——即学习者只能从给定的未标注数据池中选择查询——我们发现这种看似微小的查询能力变化可能会极大地改变统计学习的难度。特别地，某些在基于池的设置下本质上学习缓慢、其误差随样本数量仅呈多项式衰减的假设类，一旦允许合成查询，便变得可以指数级快速学习。这一显著的差距表明，成员查询的合成引发了一种根本不同的学习模式，这种模式未能被现有主动学习理论充分刻画，需要新的分析工具来表征其复杂度。受这一现象启发，我们提出了若干充分条件，展示了有趣的例子，并提出……

    arXiv:2609.27241v1 Announce Type: cross  Abstract: This work revisits a fundamental question in active learning: how powerful is the ability to synthesize arbitrary queries? Compared to pool-based active learning, where the learner only selects queries from a given unlabeled pool, we find that this seemingly mild change in query ability may dramatically alter the difficulty of statistical learning. In particular, some hypothesis classes that are inherently slow to learn in the pool-based setting, achieving only polynomial error decay in the number of samples, become exponentially learnable once synthesized queries are allowed. This striking gap suggests that membership query synthesis induces a fundamentally different mode of learning, one that is not adequately captured by existing active learning theory and calls for new analytical tools to characterize its complexity. Motivated by this phenomenon, we develop several sufficient conditions, present intriguing examples, and propose a c
    
[^42]: 分布时序差分学习中的在线推断

    Online Inference in Distributional Temporal-Difference Learning

    [https://arxiv.org/abs/2608.14408](https://arxiv.org/abs/2608.14408)

    本文为分布时序差分学习提供了在线推断的理论框架，证明了平均估计器和自助法的渐近正态性，并扩展到了非光滑泛函的局部推断。

    

    我们研究了在固定策略下，针对回报分布泛函的在线统计推断问题。回报分布通过非参数分布时序差分学习从单一马尔可夫轨迹中估计。对于Polyak-Ruppert平均估计器，我们证明了其根号T误差在Cramér空间中弱收敛于中心高斯随机元素。我们还证明了，在观测轨迹的条件下，自助法与原始平均值之间的根号T差异弱收敛于相同的高斯极限。这些结果证明了对于光滑统计泛函（包括方差、CVaR、期望短缺和期望分位数）的自助推断是合理的。对于非光滑统计泛函，我们开发了在有限多个阈值的$T^{-1/2}$邻域内估计回报CDF的局部渐近理论，以及其自助对应版本。该理论使我们能够进行推断。

    arXiv:2608.14408v1 Announce Type: cross  Abstract: We study online statistical inference for functionals of the return distribution under a fixed policy. The return distribution is estimated by nonparametric distributional temporal-difference learning from a single Markov trajectory. For the Polyak--Ruppert averaged estimator, we prove that its root-$T$ error converges weakly to a centered Gaussian random element in Cram\'er space. We also prove that, conditionally on the observed trajectory, the root-$T$ difference between the bootstrap and original averages converges weakly to the same Gaussian limit. These results justify bootstrap inference for smooth statistical functionals, including variance, CVaR, expected shortfall, and expectiles. For nonsmooth statistical functionals, we develop a local asymptotic theory for the estimated return CDF over $T^{-1/2}$-neighborhoods of finitely many thresholds, together with its bootstrap analogue. This theory allows us to conduct inference for 
    
[^43]: 分布强化学习中分位数时序差分学习的在线推断

    Online Inference for Quantile Temporal Difference Learning in Distributional Reinforcement Learning

    [https://arxiv.org/abs/2608.12973](https://arxiv.org/abs/2608.12973)

    本文提出了一种基于随机缩放的在线推断方法，通过泛函中心极限定理支持，在不存储完整轨迹的情况下对分布强化学习中的QTD进行统计推断，显著减少了内存需求。

    

    本文研究了分布强化学习中分位数时序差分学习（QTD）的统计推断方法。假设可以访问生成模型，我们首先为同步和异步QTD建立了泛函中心极限定理，表明QTD的平均迭代弱收敛于一个重新缩放的布朗运动。接着，我们提供了在线推断方法。基于随机缩放，该推断过程利用整个QTD路径上的信息构建了一个渐近枢轴统计量进行推断。同时，所提出的统计量可以在线计算，无需存储QTD迭代的完整轨迹。这大大降低了内存需求，并实现了分布强化学习中的高效统计推断。

    arXiv:2608.12973v1 Announce Type: cross  Abstract: In this paper, we study how to perform statistical inference for quantile temporal difference learning (QTD) in distributional reinforcement learning. Assuming access to a generative model, we first establish functional central limit theorems for both synchronous and asynchronous QTD, which show that the averaged iterates of QTD converge weakly to a rescaled Brownian motion. We next provide online inference methods. Based on random scaling, the inference procedure constructs an asymptotically pivotal statistic for inference by using the information along the whole QTD path. Meanwhile, the proposed statistic can be computed online without storing the entire trajectory of QTD iterates. This substantially reduces the memory requirement and enables efficient statistical inference in distributional reinforcement learning.
    
[^44]: 高维稀疏更新下随机动量的动力学

    Dynamics of Stochastic Momentum with Sparse Updates in High Dimensions

    [https://arxiv.org/abs/2605.28961](https://arxiv.org/abs/2605.28961)

    该论文首次在高维稀疏更新场景下精确刻画了动量优化的二阶矩动力学，揭示了其相变结构由动量保持时间尺度与学习时间尺度之比决定：学习慢于保持时极限等价于SGD，学习快于保持时系统失稳。

    

    现有的动量理论假设梯度以大致恒定的速率到达每个参数，但这一假设在实践中会因重尾数据分布和现代架构而被打破。我们从理论上分析了稀疏更新下两个可解析求解的动量模型的动力学：一个是具有稀疏输入的最小二乘模型，另一个是具有稀有类别的逻辑回归模型。两者都具有精确的闭式二阶矩动力学，我们在稀疏度、批次大小和动量衰减的三个标度指数下刻画了其高维极限。两个问题的相结构均由两个内在时间尺度之比决定：动量保持时间尺度（缓冲区能存活多少次有效更新）和学习时间尺度（减少平方误差需要多少次有效更新）。当学习远慢于保持时，极限与SGD一致；当学习更快时，系统不稳定；……

    arXiv:2605.28961v2 Announce Type: replace-cross  Abstract: Existing theory of momentum assumes that gradients arrive at every parameter at a roughly constant rate, an assumption violated in practice by heavy-tailed data distributions and modern architectures. We theoretically analyze the dynamics of two tractable models of momentum under sparse updates: a least squares model with sparse inputs and a logistic regression model with a rare class. Both admit exact closed-form second-moment dynamics whose high-dimensional limits we characterize across three scaling exponents for sparsity, batch size, and momentum decay. The phase structure on both problems is governed by the ratio of two intrinsic timescales: a momentum retention timescale (how many active updates the buffer survives) and a learning timescale (how many active updates it takes to reduce the squared error). When learning is much slower than retention, the limit matches SGD; when learning is faster, the system is unstable; whe
    
[^45]: 以人为中心的学习力学：受保护风险约束下的可观测熵控制

    Human-Centered Learning Mechanics: Observable Entropy Control under Protected-Risk Constraints

    [https://arxiv.org/abs/2605.22940](https://arxiv.org/abs/2605.22940)

    本文建立了在任务、风险与干预约束分离设定下的熵调控学习理论，通过噪声校准的对数行列式泛函导出精确谱力公式、可观测/预测零空间速度分解、标量反馈系数完整区间，以及多受保护风险下的极小极大可行性证书与锥投影公式，实现受保护风险约束下的可观测熵控制。

    

    熵降低并非一个安全准则：它可能使预测保持不变，或伴随着受保护群体风险的上升。我们研究在分别指定的任务、风险与干预约束下的熵调控学习。一个经噪声校准的对数行列式泛函给出了精确的谱力公式，以及将速度度量分解为可观测分量与预测零空间分量的方法。对于标量反馈，我们刻画了在保持任务下降比例并满足熵速率要求时系数的完整区间；而将干预限制在可观测方向上可能使该区间为空。对于多个受保护风险，我们在指定子空间与范数预算内推导出精确的极小极大可行性证书、针对不确定约束行的精确鲁棒扩展，以及保护后剩余熵削减的锥投影公式。这些公式采用标准的凸几何方法，但……

    arXiv:2605.22940v2 Announce Type: replace-cross  Abstract: Entropy reduction is not a safety criterion: it can leave predictions unchanged or accompany increasing risk for a protected group. We study entropy-regulated learning under separately specified task, risk, and intervention constraints. A noise-calibrated log-determinant functional yields an exact spectral force formula and a metric decomposition into observable and prediction-null velocities. For scalar feedback, we characterize the complete interval of coefficients preserving a task-descent fraction and meeting an entropy-rate request; restricting intervention to observable directions can make that interval empty. For several protected risks, we derive an exact minimax certificate of feasibility within a specified subspace and norm budget, an exact robust extension for uncertain constraint rows, and a cone-projection formula for the entropy reduction remaining after protection. These formulas use standard convex geometry but 
    
[^46]: 基于扩散模型的随机算子网络用于随机偏微分方程的不确定性量化

    Diffusion-Based Stochastic Operator Networks for Uncertainty Quantification in Stochastic Partial Differential Equations

    [https://arxiv.org/abs/2605.17107](https://arxiv.org/abs/2605.17107)

    该论文提出了随机算子网络（SON），通过将DeepONet与随机神经网络及扩散模型相结合，直接从含噪数据中学习，为随机偏微分方程的解同时提供均值预测和不确定性量化。

    

    我们引入了一个新颖的框架，用于对随机偏微分方程（SPDEs）相关的解算子进行不确定性量化。尽管随机偏微分方程在建模不确定性下的复杂物理系统中扮演着核心角色，但其实际应用通常需要指定模型不确定性的幅度和结构，而这些信息往往是未知的，且难以从含噪测量数据中推断出来。为了应对这一挑战，我们开发了一种随机算子学习框架，该框架能够直接从含噪数据中学习，并同时输出均值解场和不确定性量化结果。所提出的方法被称为随机算子网络，其结构是将深度算子网络与随机神经网络相结合，以建模随机性并实现概率预测。训练过程通过最小化哈密顿型损失函数并优化……

    arXiv:2605.17107v2 Announce Type: replace-cross  Abstract: We introduce a novel framework for uncertainty quantification of solution operators associated with stochastic partial differential equations (SPDEs). Although SPDEs play a central role in modeling complex physical systems under uncertainty, their practical use typically requires specifying the magnitude and structure of model uncertainties that are often unknown and difficult to infer from noisy measurements. To address this challenge, we develop a stochastic operator-learning framework that learns directly from noisy data and outputs both a mean solution field and a quantification of uncertainty. The proposed method, namely the Stochastic Operator Network (SON), is constructed by combining the structure of the Deep Operator Network (DeepONet) with Stochastic Neural Networks (SNNs) to model stochasticity and enable probabilistic prediction. The training procedure is carried out by minimizing a Hamiltonian-type loss and optimiz
    
[^47]: 多上下文贝叶斯优化中的状态条件化评估

    Regime-Conditioned Evaluation in Multi-Context Bayesian Optimization

    [https://arxiv.org/abs/2605.04895](https://arxiv.org/abs/2605.04895)

    该论文揭示了迁移贝叶斯优化中采集函数的排名会随预算比例和先验质量等状态变量而逆转（98% 的已发表比较从未控制这一变量），并提出可用先验信息估计的可移植状态得分 PRS=(B/|A|)(1-ρ) 来预测这种排名转变，从而主张以状态条件化评估取代平均效应比较。

    

    已发表的迁移贝叶斯优化（transfer-BO）比较研究通常在隐藏的状态变量上估计采集函数选择的平均处理效应，而实际应用者需要的是针对其特定先验质量、预算比例和度量指标的条件效应。对来自 NeurIPS、ICML、ICLR、AISTATS、UAI、TMLR、JMLR 和 AutoML-Conf（2022-2025）的 40 篇迁移贝叶斯优化论文的审查发现，98% 的论文从未将 B/|A| 作为受控变量轴进行变化。在相同的 GDSC2 基准上，仅改变预算就会逆转排名：当 B=50 时，Greedy 在 Hit@1 上比 UCB 高 0.050；而当 B=100 时，UCB 比 Greedy 高 0.035。我们用可移植状态得分（Portable Regime Score）PRS=(B/|A|)(1-ρ) 来刻画这一转变，其中 ρ 是先验秩相关系数，可以在主比较之前从试点上下文中估计得到。在涵盖化学、药物反应生物学和超参数优化（HPO）的 79 个条件中，层次模型给出 β=0.50（p=1.1e-9），且 19% 的条件落入等效区，其中 |adv……（原文摘要至此截断）

    arXiv:2605.04895v2 Announce Type: replace  Abstract: Published transfer-BO comparisons often estimate an average treatment effect of acquisition choice over hidden regime variables, while practitioners need the conditional effect for their specific prior quality, budget ratio, and metric. An audit of 40 transfer-BO papers from NeurIPS, ICML, ICLR, AISTATS, UAI, TMLR, JMLR, and AutoML-Conf (2022-2025) finds that 98% never vary B/|A| as a controlled axis. On the same GDSC2 benchmark, changing only the budget reverses the ranking: at B=50, Greedy outperforms UCB by 0.050 Hit@1, while at B=100, UCB outperforms Greedy by 0.035. We capture this transition with the Portable Regime Score PRS=(B/|A|)(1-rho), where rho is the prior rank correlation and can be estimated from pilot contexts before the main comparison. Across 79 conditions spanning chemistry, drug-response biology, and HPO, a hierarchical model gives beta=0.50 (p=1.1e-9), and 19% of conditions fall in an equivalence zone where |adv
    
[^48]: 用不确定性来源赋能贝叶斯推断：从置信度建模到稀疏估计

    Empowering Bayes with sources of uncertainty: from confidence modelling to sparse estimation

    [https://arxiv.org/abs/2605.03134](https://arxiv.org/abs/2605.03134)

    该论文提出不确定性来源（SoU）推断框架，通过在模型组件间显式编码异质置信度来扩展贝叶斯推断，使置信度与先验信息量可分别控制，并据此开发了诱导稀疏性的通用方法，可应用于线性/逻辑回归及贝叶斯神经网络的自剪枝。

    

    我们提出了不确定性来源推断，这是一个通用框架，通过允许研究者显式编码模型各组件间异质的置信度来扩展贝叶斯推断。这种异质性使概率处理能够在模型的不同部分选择性地变化，同时控制每个组件内部保留的不确定性及其向模型其余部分的传播。经验贝叶斯和轮廓型推断可作为这种选择性处理的极限情形而自然出现。由于置信度通过与先验信息量不同的机制发挥作用，两者可以分别控制，这为贝叶斯模型设计提供了额外的自由度。基于这种选择性处理，我们开发了一种在统计模型中诱导稀疏性的通用方法，并在线性回归、逻辑回归以及贝叶斯神经网络中展示了其应用，在贝叶斯神经网络中它能够实现自剪枝。

    arXiv:2605.03134v2 Announce Type: replace-cross  Abstract: We introduce sources of uncertainty (SoU) inference, a general framework that extends Bayesian inference by allowing researchers to explicitly encode heterogeneous confidence across model components. This heterogeneity allows the probabilistic treatment to vary selectively across the model, controlling both the uncertainty retained within each component and its propagation to the rest of the model. Empirical Bayes and profiling-type inference emerge as limiting configurations of this selective treatment. Because confidence acts through a mechanism distinct from prior informativeness, the two can be controlled separately, providing an additional degree of freedom for Bayesian model design. Building on this selective treatment, we develop a general approach for inducing sparsity in statistical models and illustrate its use in linear and logistic regression, as well as in Bayesian neural networks, where it enables self-pruning.
    
[^49]: 使用核嵌入度量条件分布之间的差异

    Measuring Differences between Conditional Distributions using Kernel Embeddings

    [https://arxiv.org/abs/2605.02260](https://arxiv.org/abs/2605.02260)

    本文提出了条件最大均值差异（CMMD）这一统一理论框架，通过不同层级的核嵌入度量方法，系统性地解决了条件分布之间差异度量缺乏统一理论处理的问题。

    

    比较条件分布是统计学和机器学习中的一项基本挑战，在众多领域都有广泛应用。尽管基于再生核希尔伯特空间（RKHS）中分布核嵌入来度量差异的方法提供了强大的非参数技术，但现有文献仍然较为分散，缺乏统一的理论处理。本文通过建立一个连贯的框架来弥补这一空白，用于研究基于核的方法，通过我们称之为条件最大均值差异（CMMD）来度量条件分布之间的散度。CMMD 由一系列我们称之为“层级”的度量组成，其中包含三个特例，每个特例使用不同类型的 RKHS 嵌入：CMMD$_0$（条件均值算子）、CMMD$_1$（条件均值嵌入）和 CMMD$_2$（联合均值嵌入）。我们还进一步引入了一个一般层级……

    arXiv:2605.02260v2 Announce Type: replace-cross  Abstract: Comparing conditional distributions is a fundamental challenge in statistics and machine learning, with applications across a wide range of domains. While proposed methods for measuring discrepancies using kernel embeddings of distributions in a reproducing kernel Hilbert space (RKHS) provide powerful non-parametric techniques, the existing literature remains fragmented and lacks a unified theoretical treatment. This paper addresses this gap by establishing a coherent framework for studying kernel-based methods to measure divergence between conditional distributions through what we refer to as conditional maximum mean discrepancy (CMMD). The CMMD consists of a family of metrics which we call levels, with three special cases each using a different type of RKHS embedding: CMMD$_0$ (conditional mean operators), CMMD$_1$ (conditional mean embeddings), and CMMD$_2$ (joint mean embeddings). We additionally introduce a general level $
    
[^50]: 二维稳定性选择：通过设计抖动排除相关的无关特征

    2D Stability Selection: Design Jittering to Exclude Correlated Irrelevant Features

    [https://arxiv.org/abs/2605.02205](https://arxiv.org/abs/2605.02205)

    提出二维稳定性选择（2D-SS），通过在半样本和注入设计噪声的网格上聚合选择频率，能够排除违反不可表示条件的相关混淆特征，从而修复标准稳定性选择无法控制的假阳性问题。

    

    稳定性选择（Stability Selection）对观测样本进行重抽样，但保持设计矩阵不变，因此无法拒绝那些与真实相关特征相关联、且违反不可表示条件的无关特征。这样的特征（“混淆变量”）会在相当大比例的重抽样中被选中，从而导致稳定性选择假阳性界所依赖的可交换性失效。我们提出二维稳定性选择（2D-SS），该方法在半样本以及注入设计噪声的网格上聚合选择频率，并在零噪声时退化为标准的稳定性选择。由于输入噪声在期望意义上起到Tikhonov正则化的作用，经过抖动的Lasso拟合相当于一个带噪声的弹性网，其支撑恢复由一个以噪声水平为索引的不可表示条件所支配。因此，抖动可以修复混淆变量违反Lasso条件的设计，尽管它也可能带来负面影响。我们给出了违反条件的特征……（原文摘要至此截断）

    arXiv:2605.02205v2 Announce Type: replace-cross  Abstract: Stability Selection resamples the observations but leaves the design untouched, so it cannot reject an irrelevant feature that is correlated with the relevant ones in a way that violates the irrepresentable condition. Such a feature (a ``confounder'') is selected in a substantial fraction of resamples, and the exchangeability behind the false-positive bound of Stability Selection fails. We propose 2D Stability Selection (2D-SS), which aggregates selection frequencies over half-samples and over a grid of injected design noise, and reduces to Stability Selection at zero noise. Because input noise acts, in expectation, as Tikhonov regularisation, a jittered Lasso fit is a noisy elastic net whose support recovery is governed by a noise-indexed irrepresentable condition. Jittering can therefore repair designs in which confounders violate the Lasso condition, although it can also hurt. We give the noise level at which a violated feat
    
[^51]: 决策依赖不确定性下谱风险优化的自适应条件残差抽样

    Adaptive Conditional Residual Sampling for Spectral Risk Optimisation under Decision-Dependent Uncertainty

    [https://arxiv.org/abs/2603.12507](https://arxiv.org/abs/2603.12507)

    提出自适应条件残差抽样（ACRS）方法，通过复用标准化残差向量构建确定性代理模型，高效求解决策依赖不确定性下的谱风险优化问题，并提供基于Wasserstein误差的最优性保证。

    

    我们考虑最小化谱风险目标函数，即期望成本加上加权条件风险价值（CVaR），其中不确定参数的分布依赖于决策，且只能通过随机模拟器在固定的抽样预算下进行抽样。我们提出自适应条件残差抽样（ACRS）方法。该方法在空间填充设计的每个点上抽取一次模拟样本，拟合决策依赖分布的异方差回归模型，并将标准化残差向量作为情景复用，从而构建一个确定性代理模型。ACRS在该代理模型上搜索若干局部最优解，在最有希望的候选点周围增加抽样，利用共同随机数在模拟器上对候选方案进行排序，最后采用尾部重要性抽样进行样本路径的局部精细化。我们证明代理模型最优解的最优性间隙受情景分布的Wasserstein误差约束……

    arXiv:2603.12507v3 Announce Type: replace  Abstract: We consider minimising a spectral risk objective, the expected cost plus a weighted conditional value-at-risk, when the distribution of the uncertain parameters depends on the decision and can only be sampled through a stochastic simulator under a fixed budget of draws. We propose adaptive conditional residual sampling (ACRS). It takes one simulator draw at each point of a space-filling design, fits a heteroscedastic regression model of the decision-dependent distribution and reuses the standardised residual vectors as scenarios to construct a deterministic surrogate. ACRS searches this surrogate for several local optima, adds draws around the most promising ones, ranks the candidates on the simulator with common random numbers and finishes with sample-path local refinement using tail importance sampling. We show that the optimality gap of the surrogate minimiser is bounded by the Wasserstein error of the scenario distribution and th
    
[^52]: 结构因果瓶颈模型

    Structural Causal Bottleneck Models

    [https://arxiv.org/abs/2603.08682](https://arxiv.org/abs/2603.08682)

    该论文提出结构因果瓶颈模型（SCBMs），用低维“瓶颈”统计量传递高维变量间的因果效应，证明了瓶颈变量在双射意义下可从观测数据识别，并表明瓶颈表示能够恢复工具变量方法的可识别性、并支持中介分析与后门调整下的有效因果效应估计。

    

    我们提出了结构因果瓶颈模型（SCBMs），这是一类新颖的结构因果模型，其中高维变量之间的因果效应由低维的汇总统计量（即“瓶颈”）来传递。SCBMs为机制特定的、目标依赖的降维提供了一个灵活的框架，同时仍可通过标准学习算法进行估计。我们证明了瓶颈变量可以从观测数据中被识别（至双射），并通过实验验证了这一结论。随后，我们研究了线性SCBMs在三种标准调整策略下的因果效应估计问题。对于工具变量方法，我们发现了当所有变量均为高维时，朴素的二阶段最小二乘法在总体层面存在不可识别性，并证明瓶颈表示可以恢复可识别性。对于中介分析和后门调整，我们证明了对瓶颈变量进行条件化可以得到有效的效应估计。

    arXiv:2603.08682v3 Announce Type: replace-cross  Abstract: We introduce structural causal bottleneck models (SCBMs), a novel class of structural causal models in which causal effects between high-dimensional variables are mediated by low-dimensional summary statistics, or *bottlenecks*. SCBMs provide a flexible framework for mechanism-specific, target-dependent dimension reduction while remaining estimable via standard learning algorithms. We prove that bottleneck variables are identifiable up to bijection from observational data, and validate this experimentally. We then study causal effect estimation in linear SCBMs under three standard adjustment strategies. For instrumental variables, we identify a population-level non-identifiability of na\"ive two-stage least squares when all variables are high-dimensional, and show that bottleneck representations restore identification. For mediation and backdoor adjustment, we prove that conditioning on bottleneck variables yields valid effect 
    
[^53]: 使用单次概率前向传播与代码生成加速贝叶斯神经网络的执行

    Accelerated Execution of Bayesian Neural Networks using a Single Probabilistic Forward Pass and Code Generation

    [https://arxiv.org/abs/2511.23440](https://arxiv.org/abs/2511.23440)

    该论文通过假设权重和激活服从高斯分布，用单次解析式的概率前向传播取代多次采样前向传播来加速贝叶斯神经网络，并构建了在嵌入式ARM CPU上训练、编译、优化和部署的端到端流水线。

    

    机器学习模型在诊断、天气预报、自然语言处理和自动驾驶等领域表现良好，但其有限的不确定性处理能力限制了在安全关键场景中的应用。传统神经网络往往无法检测分布外（OOD）数据，并可能输出看似自信却错误的预测。贝叶斯神经网络（BNN）通过提供概率估计解决了这一问题，但由于预测需要对权重分布进行采样并进行多次前向传播，计算成本很高。概率前向传播（PFP）通过假设权重和激活服从高斯分布，为随机变分推断（SVI）提供了一种高效的近似方法，实现了完全解析的不确定性传播，并用单次确定性前向传播取代了采样。我们提出了一个端到端流水线，用于在嵌入式ARM CPU上训练、编译、优化和部署基于PFP的贝叶斯神经网络。

    arXiv:2511.23440v2 Announce Type: replace  Abstract: Machine learning models perform well across domains such as diagnostics, weather forecasting, NLP, and autonomous driving, but their limited uncertainty handling restricts use in safety-critical settings. Traditional neural networks often fail to detect out-of-domain (OOD) data and may output confident yet incorrect predictions. Bayesian neural networks (BNNs) address this by providing probabilistic estimates, but incur high computational cost because predictions require sampling weight distributions and multiple forward passes. The Probabilistic Forward Pass (PFP) offers a highly efficient approximation to Stochastic Variational Inference (SVI) by assuming Gaussian-distributed weights and activations, enabling fully analytic uncertainty propagation and replacing sampling with a single deterministic forward pass. We present an end-to-end pipeline for training, compiling, optimizing, and deploying PFP-based BNNs on embedded ARM CPUs. 
    
[^54]: 基于广义贝叶斯推断的鲁棒实验设计

    Robust Experimental Design via Generalised Bayesian Inference

    [https://arxiv.org/abs/2511.07671](https://arxiv.org/abs/2511.07671)

    本文提出广义贝叶斯最优实验设计（GBOED），通过用损失函数替代似然函数，在模型设定存在偏差时同时保证实验设计与统计推断的鲁棒性。

    

    贝叶斯最优实验设计是一个开展实验的原则性框架，它利用贝叶斯推断来量化选择某个设计所能期望获得的信息量。然而，准确的贝叶斯推断依赖于一个假设，即对数据生成过程所建立的统计模型是正确设定的。如果该假设不成立，贝叶斯方法可能导致糟糕的推断以及不准确的信息增益估计。广义贝叶斯（或称Gibbs）推断是一种更鲁棒的概率推断框架，它用合适的损失函数替代贝叶斯更新中的似然函数。在本工作中，我们提出了广义贝叶斯最优实验设计（GBOED），将广义贝叶斯推断扩展到实验设计场景中，从而在设计与推断两方面均实现鲁棒性。利用贝叶斯决策理论的一个推广形式，我们推导了一种新的采集（函数）……

    arXiv:2511.07671v2 Announce Type: replace-cross  Abstract: Bayesian optimal experimental design is a principled framework for conducting experiments that leverages Bayesian inference to quantify how much information one can expect to gain from selecting a certain design. However, accurate Bayesian inference relies on the assumption that one's statistical model of the data-generating process is correctly specified. If this assumption is violated, Bayesian methods can lead to poor inference and estimates of information gain. Generalised Bayesian (or Gibbs) inference is a more robust probabilistic inference framework that replaces the likelihood in the Bayesian update by a suitable loss function. In this work, we present Generalised Bayesian Optimal Experimental Design (GBOED), an extension of generalised Bayesian inference to the experimental design setting which achieves robustness in both design and inference. Using a generalisation of Bayesian decision theory, we derive a new acquisit
    
[^55]: SLIM：非线性GMM模型中的随机学习与推断

    SLIM: Stochastic Learning and Inference in Nonlinear GMM Models

    [https://arxiv.org/abs/2510.20996](https://arxiv.org/abs/2510.20996)

    提出了SLIM这一可扩展的随机近似算法，无需一致初始估计量或全局凸准则，即可对具有大量矩条件和参数的非线性GMM模型进行一致且渐近正态的估计与推断，并可扩展至百万级观测数据和聚类数据场景。

    

    我们提出了SLIM（过度识别模型中的随机学习与推断），这是一种新的可扩展随机近似算法，适用于具有大量不可分离矩条件和参数的非线性广义矩估计（GMM）模型。研究表明，SLIM无需一致的初始估计量或全局凸的GMM准则，即可实现一致性与渐近正态性。当计算不确定性相对于抽样不确定性不可忽略时，基于随机缩放和插件法的置信集构造依然有效。可选的二阶精化能够达到全样本GMM效率。我们为SLIM开发了J检验。理论结果还被扩展至具有不等聚类规模的聚类数据模型。针对包含576个矩条件和380个参数的非线性需求系统的蒙特卡洛实验展示了计算效率的提升以及对一百万个观测值的可扩展性。一个涉及480万大学毕业生的LinkedIn应用案例展示了聚类估计及……

    arXiv:2510.20996v4 Announce Type: replace  Abstract: We propose SLIM (Stochastic Learning and Inference in overidentified Models), a new scalable stochastic approximation algorithm for nonlinear GMMs with many nonseparable moments and parameters. SLIM is shown to be consistent and asymptotically normal without a consistent initial estimator or a globally convex GMM criterion. Random-scaling and plug-in based confidence set constructions remain valid when computational uncertainty can be non-negligible relative to sampling uncertainty. An optional second-order refinement achieves full-sample GMM efficiency. We develop $J$-tests for SLIM. The theoretical results are extended to models with clustered data of unequal cluster sizes. Monte Carlo experiments for a nonlinear demand system with 576 moments and 380 parameters demonstrate computational gains and scalability to one million observations. A LinkedIn application with 4.8 million college graduates illustrates clustered estimation and 
    
[^56]: 基于贝叶斯社区发现的可靠数据聚类

    Reliable data clustering with Bayesian community detection

    [https://arxiv.org/abs/2510.15013](https://arxiv.org/abs/2510.15013)

    该论文提出一种一步式聚类方法，利用贝叶斯社区检测和单一最小描述长度准则同时完成数据稀疏化与聚类，免除了任意阈值选择，在更高噪声和更少样本条件下优于传统两步法，且在无真实结构时能正确返回单一聚类。

    

    从神经科学、基因组学到系统生物学和生态学，研究人员依靠对相似性数据进行聚类来揭示模块化结构。大多数方法分两步工作：先降低噪声，再进行聚类。第一步的执行并不考虑第二步的需求：这迫使研究者做出任意的选择，丢弃有用的信息，甚至可能在不存在真实结构的地方强行施加结构。本研究将这两个步骤结合在一起。利用成熟的贝叶斯社区检测方法，我们让单一的最小描述长度准则同时决定如何稀疏化数据以及如何聚类。这种一步式方法保留了全部样本用于推断，消除了任意阈值的选择，并在不存在真实结构时返回单一聚类。在合成数据上，我们的方法能够在更多噪声和更少样本的条件下恢复聚类，优于标准方法。在人脑基因共表达数据上，它找到了比现有方法更连贯、更稳定的聚类。

    arXiv:2510.15013v2 Announce Type: replace-cross  Abstract: From neuroscience and genomics to systems biology and ecology, researchers rely on clustering similarity data to uncover modular structure. Most methods work in two steps: first reduce noise, then cluster. The first step runs without regard to the second: it forces arbitrary choices, discards useful information, and can impose structure where none exists. Here we bring the two steps together. Using established Bayesian community detection methods, we let one minimum description length criterion choose both how much to sparsify the data and how to cluster it. This one-step approach keeps all the samples for inference, removes the arbitrary threshold choice, and returns a single cluster when there is no real structure. On synthetic data, our approach recovers clusters under more noise and with fewer samples than standard methods. On gene co-expression data from the human brain, it finds more coherent and more stable clusters than
    
[^57]: 统计公平性-准确性前沿

    The Statistical Fairness-Accuracy Frontier

    [https://arxiv.org/abs/2508.17622](https://arxiv.org/abs/2508.17622)

    该论文在有限样本条件下研究了单一预测模型服务多个人口群体时的公平性-准确性权衡，推导了最坏情况最优估计量，刻画了有限样本对各群体福利的不对称影响，给出了最优样本分配策略，并提供了沿整个FA帕累托前沿的一致有限样本界。

    

    我们研究了当单一预测模型需要服务于多个人口统计群体时的公平性与准确性权衡问题。理解这一权衡的一个有用工具是公平性-准确性（FA）帕累托前沿，它刻画了这样一组模型：这些模型无法在不损害公平性或准确性的前提下改进其中任何一方。虽然刻画FA前沿需要对数据分布的完全知识，但我们聚焦于有限样本情形，量化设计者在有限数据下能够多好地逼近前沿上的任意一点，并对最坏情况下的差距给出上界。特别地，我们推导出了依赖于设计者对协变量分布了解程度的最坏情况最优估计量。对于每个估计量，我们刻画了有限样本效应如何不对等地影响各群体的福利，并确定了最优的样本分配策略。最后，我们提供了沿整个FA前沿的一致有限样本界，量化……（摘要在此处被截断）

    arXiv:2508.17622v4 Announce Type: replace-cross  Abstract: We study fairness-accuracy tradeoffs when a single predictive model must serve multiple demographic groups. A useful tool for understanding this tradeoff is the fairness-accuracy (FA) Pareto frontier, which characterizes the set of models that cannot be improved in either fairness or accuracy without worsening the other. While characterizing the FA frontier requires full knowledge of the data distribution, we focus on the finite-sample regime, quantifying how well a designer can approximate any point on the frontier from limited data and bounding the worst-case gap. In particular, we derive worst-case-optimal estimators that depend on the designer's knowledge of the covariate distribution. For each estimator, we characterize how finite-sample effects asymmetrically impact each group's welfare and identify optimal sample allocation strategies. Finally, we provide uniform finite-sample bounds along the entire FA frontier, quantif
    
[^58]: 具有信息有序策略的马尔可夫决策过程强化学习

    Reinforcement Learning in MDPs with Information-Ordered Policies

    [https://arxiv.org/abs/2508.03904](https://arxiv.org/abs/2508.03904)

    本文提出策略类上的“信息序”概念，通过反事实推断为连续状态动作空间的无限时域平均成本MDP设计了基于epoch的强化学习算法，取得了由信息序宽度 $w$ 决定的 $\widetilde O(H\sqrt{wdT/\alpha})$ 遗憾界，从而利用策略间数据的可迁移性降低了探索复杂度。

    

    许多运营性马尔可夫决策过程（MDP）提供了结构化信息，使得在某一策略下生成的轨迹还可能揭示其他策略本会有怎样的表现。我们通过参数化策略类上的信息序来形式化这一现象：当在 $\pi_\theta$ 下收集的数据可用于估计 $\pi_{\theta‘}$ 的长期平均成本时，记作 $\pi_{\theta’} \preceq \pi_\theta$。我们为具有连续状态和动作空间的无限时域平均成本MDP开发了一种基于epoch的强化学习算法，该算法通过反事实推断来利用这种序关系。对于一个 $d$ 维策略类，若其信息序的宽度为 $w$，则该算法相对于该类中的最优策略实现了 $\widetilde O(H\sqrt{wdT/\alpha})$ 的遗憾，其中 $H$ 为偏差跨度上界，$\alpha$ 表示“信息比率”。因此，信息序的宽度刻画了该问题的有效探索复杂度。

    arXiv:2508.03904v2 Announce Type: replace-cross  Abstract: Many operational Markov decision processes provide structured information where a trajectory generated under one policy may also reveal how other policies would have performed. We formalize this through an information order over a parameterized policy class, writing $\pi_{\theta'} \preceq \pi_\theta$ when data collected under $\pi_\theta$ can be used to estimate the long-run average cost of $\pi_{\theta'}$. We develop an epoch-based reinforcement learning algorithm for infinite-horizon average-cost MDPs with continuous state and action spaces that exploits this order through counterfactual inference. For a $d$-dimensional policy class whose information order has width $w$, the algorithm achieves $\widetilde O(H\sqrt{wdT/\alpha})$ regret relative to the optimal policy in the class, where $H$ is a bias span upper bound and $\alpha$ denotes an "information ratio". Thus, the width captures the effective exploration complexity of th
    
[^59]: 基于对抗性随机森林的缺失值插补方法——MissARF

    Missing value imputation with adversarial random forests - MissARF

    [https://arxiv.org/abs/2507.15681](https://arxiv.org/abs/2507.15681)

    MissARF是一种基于对抗性随机森林的快速缺失值插补方法，可在提供高质量单重与多重插补的同时，使多重插补无需额外计算成本。

    

    处理缺失值是生物统计分析中的一个常见挑战，通常通过插补方法来解决。我们提出了一种新颖、快速且易于使用的插补方法，称为基于对抗性随机森林的缺失值插补（MissARF），该方法基于生成式机器学习，可同时提供单重插补和多重插补。MissARF利用对抗性随机森林（ARF）进行密度估计和数据合成。为了插补某个观测值的缺失值，我们以该观测的非缺失值为条件，从ARF生成的估计条件分布中进行采样。我们的实验表明，MissARF在插补质量和运行速度方面与最先进的单重和多重插补方法表现相当，且多重插补无需额外的计算成本。

    arXiv:2507.15681v2 Announce Type: replace-cross  Abstract: Handling missing values is a common challenge in biostatistical analyses, typically addressed by imputation methods. We propose a novel, fast, and easy-to-use imputation method called missing value imputation with adversarial random forests (MissARF), based on generative machine learning, that provides both single and multiple imputation. MissARF employs adversarial random forest (ARF) for density estimation and data synthesis. To impute a missing value of an observation, we condition on the non-missing values and sample from the estimated conditional distribution generated by ARF. Our experiments demonstrate that MissARF performs comparably to state-of-the-art single and multiple imputation methods in terms of imputation quality and fast runtime with no additional costs for multiple imputation.
    
[^60]: 基于B样条谱密度的时空闭式协方差函数

    Closed-form Covariance Functions in Space and Time from B-spline Spectral Densities

    [https://arxiv.org/abs/2506.21953](https://arxiv.org/abs/2506.21953)

    本文通过推导任意阶数和节点配置的B样条谱基的逆傅里叶变换，构建了一族闭式的非参数平稳协方差函数，该族模型在具有谱密度的弱平稳过程的自协方差函数空间中一致稠密，并能适用于单变量、多变量及多维时空数据。

    

    自协方差函数（ACF）是时间序列、空间及时空数据分析的核心。经典参数模型对于多变量和时空数据往往过于局限，这促使了对ACF的非参数描述的需求。Bochner定理保证任何正谱测度都可以通过逆傅里叶变换得到有效的ACF；然而，现有的谱域非参数方法很少能给出ACF本身的闭式表达式。我们通过推导任意阶数和节点位置的B样条谱基的逆傅里叶变换，发展出一类闭式的非参数ACF。这产生了一族具有以下三个性质的闭式平稳协方差模型：(i) 在所有具有谱密度的弱平稳过程的ACF中，该族模型对ACF是一致稠密的；(ii) 它适用于单变量、多变量和多维情形（摘要内容在此处截断）。

    arXiv:2506.21953v2 Announce Type: replace-cross  Abstract: The autocovariance function (ACF) is central to the analysis of time series, spatial, and spatio-temporal data. Classical parametric models are often too restrictive for multivariate and space-time data, which motivates non-parametric descriptions of the ACF. Bochner's Theorem guarantees that any positive spectral measure yields a valid ACF via the inverse Fourier transform; however, existing non-parametric approaches in the spectral domain rarely return closed-form expressions for the ACF itself. We develop a closed-form class of non-parametric ACFs by deriving the inverse Fourier transform of B-spline spectral bases with arbitrary degree and knot placement. This yields a family of closed-form stationary covariance models with three properties: (i) it is uniformly dense for the ACF, among the ACFs of weakly stationary processes that possess a spectral density; (ii) it accommodates univariate, multivariate, and multidimensional
    
[^61]: 深度随机ReLU神经网络的 $\ell^p$-Lipschitz 常数的近最优估计

    Near-optimal estimates for the $\ell^p$-Lipschitz constants of deep random ReLU neural networks

    [https://arxiv.org/abs/2506.19695](https://arxiv.org/abs/2506.19695)

    本文证明了遵循He初始化变体的随机ReLU神经网络的 $\ell^p$-Lipschitz 常数的近最优估计，揭示了该常数在 $p \in [2,\infty]$ 时类似于高斯向量的 $\ell^{p'}$ 范数，而在 $p \in [1,2)$ 时则更接近高斯向量的 $\ell^2$ 范数。

    

    本文研究了具有随机参数的ReLU神经网络 $\Phi: \mathbb{R}^d \to \mathbb{R}$ 在 $p \in [1,\infty]$ 范围内的 $\ell^p$-Lipschitz 常数。权重的分布遵循He初始化的一个变体。在零偏置网络的情况下，我们为宽网络推导出了高概率意义下的上下界，这两个界之间最多相差一个与网络深度呈对数关系的因子。值得注意的是，$\ell^p$-Lipschitz 常数的行为在 $p \in [1,2)$ 和 $p \in [2,\infty]$ 两个区间之间表现出显著差异。对于 $p \in [2,\infty]$，$\ell^p$-Lipschitz 常数的表现类似于 $\Vert g\Vert_{p'}$，其中 $g \in \mathbb{R}^d$ 是一个 $d$ 维标准高斯向量，且 $1/p + 1/p' = 1$。相反，对于 $p \in [1,2)$，$\ell^p$-Lipschitz 常数更接近于 $\Vert g \Vert_{2}$。我们将分析进一步扩展到偏置可能非零且从任意分布中抽取的网络。

    arXiv:2506.19695v2 Announce Type: replace-cross  Abstract: This paper studies the $\ell^p$-Lipschitz constants of ReLU neural networks $\Phi: \mathbb{R}^d \to \mathbb{R}$ with random parameters for $p \in [1,\infty]$. The distribution of the weights follows a variant of the He initialization. In the case of zero-bias networks, we derive high probability upper and lower bounds for wide networks that differ at most by a factor that is logarithmic in the network's depth. Remarkably, the behavior of the $\ell^p$-Lipschitz constant varies significantly between the regimes $ p \in [1,2) $ and $ p \in [2,\infty] $. For $p \in [2,\infty]$, the $\ell^p$-Lipschitz constant behaves similarly to $\Vert g\Vert_{p'}$, where $g \in \mathbb{R}^d$ is a $d$-dimensional standard Gaussian vector and $1/p + 1/p' = 1$. In contrast, for $p \in [1,2)$, the $\ell^p$-Lipschitz constant aligns more closely to $\Vert g \Vert_{2}$. We extend our analysis to networks with possibly non-zero biases drawn from arbitra
    
[^62]: RAW-Explainer：面向知识图谱补全的路径推理图神经网络事后解释方法

    RAW-Explainer: Post-hoc Explanations of Path Reasoning Graph Neural Networks for Knowledge Graph Completion

    [https://arxiv.org/abs/2506.12558](https://arxiv.org/abs/2506.12558)

    本文提出RAW-Explainer，一种专为路径推理图神经网络设计的新型可解释性方法，通过随机游走目标函数和异构信息，为知识图谱链接预测生成连通、简洁且可解释的子图解释，并解决了评估小子图时的分布偏移问题。

    

    路径推理图神经网络在链接预测等知识图谱补全任务上已展现出最先进的性能。然而，解释其预测结果仍然是一个具有挑战性的开放性问题，因为大多数GNN可解释性方法针对的是节点级或图级任务。在本文中，我们提出了RAW-Explainer（随机游走解释器），这是一种专为路径推理GNN量身定制的新型可解释性方法，能够为知识图谱中的链接预测生成连通、简洁且可解释的子图解释。我们的方法利用异构信息来识别事实性解释，并通过随机游走目标函数来增强连通性，同时使用参数化模型实现高效的集体解释。此外，RAW-Explainer通过提出一种稳健的评估策略，克服了在评估比完整图小数个数量级的解释子图时出现的分布偏移问题。

    arXiv:2506.12558v2 Announce Type: replace  Abstract: Path reasoning graph neural networks have demonstrated state-of-the-art performance on knowledge graph completion tasks such as link prediction. However, interpreting their predictions remains a challenging open problem, as most GNN explainability methods target node-level or graph-level tasks. In this paper, we propose RAW-Explainer ($\underline{Ra}$ndom $\underline{W}$alk Explainer), a novel explainability method tailored to path reasoning GNNs that generates connected, concise, and interpretable subgraph explanations for link prediction in knowledge graphs. Our method uses heterogeneous information to identify factual explanations and encourages connectivity through a random walk objective, using a parameterized model for efficient collective explanation. Furthermore, RAW-Explainer overcomes the distribution shift issue when evaluating explanatory subgraphs that are orders of magnitude smaller than the full graph by proposing a ro
    
[^63]: 因果模型与因果表示中的“因果”究竟指什么？

    What is causal about causal models and representations?

    [https://arxiv.org/abs/2501.19335](https://arxiv.org/abs/2501.19335)

    该论文提出了一个将因果模型与数据生成过程分离的形式化框架，形式化了连接行动与干预的解释，并证明了不存在能同时满足一组期望性质且保持非循环性的行动-干预解释这一不可能性结果。

    

    因果贝叶斯网络通常被描述为“因果的”，因为它们支持关于干预的推理。然而，干预是在数学模型中定义的，而模型的预测涉及的是现实世界中行动的效果。因此，因果模型是否以及如何对行动的效果做出可证伪的预测，取决于行动如何被解释为干预。这需要明确哪些行动对应哪些干预，而这一步骤往往被隐式处理。我们开发了一个形式化框架，将因果模型与其旨在描述的数据生成过程分离开来，并将连接行动与干预的解释形式化。在该框架内，我们证明了一个不可能性结果：没有任何解释能够在保持非循环性的同时满足一组期望的性质。在循环解释下，每一个拟合观测数据的模型……

    arXiv:2501.19335v3 Announce Type: replace-cross  Abstract: Causal Bayesian networks are often described as causal because they support reasoning about interventions. Yet interventions are defined in the mathematical model, whereas the resulting model predictions concern the effects of actions in the real world. Consequently, whether and how a causal model makes falsifiable predictions about the effects of actions depends on how actions are interpreted as interventions. This requires specifying which actions correspond to which interventions, a step that is frequently left implicit. We develop a formal framework that separates a causal model from the data-generating process it aims to describe and formalizes interpretations connecting actions to interventions. Within this framework, we prove an impossibility result: no interpretation can simultaneously satisfy a collection of desirable properties while remaining non-circular. Under a circular interpretation, every model that fits the ob
    
[^64]: 策略梯度真的是梯度吗？

    Is the Policy Gradient a Gradient?

    [https://arxiv.org/abs/1906.07073](https://arxiv.org/abs/1906.07073)

    本文证明了省略折扣因子的策略梯度更新不必是任何函数的梯度，并构造反例表明该更新会使策略收敛到在折扣与未折扣目标下均全局最差的确定性极限。

    

    策略梯度定理描述了期望折扣回报关于智能体策略参数的梯度。然而，许多策略梯度方法省略了外层折扣因子，由此产生的更新不必等于折扣目标的梯度。那么这个更新是否是另一个目标的梯度呢？这多年来一直是一个悬而未决的问题，这种理论上的不清晰也导致文献中出现了大量错误的表述。本文针对由此产生的按回合求和的在策略更新回答了这一问题，通过证明证明该更新不必等于任何函数的梯度。此外，作者还构建了一个反例：在该反例中，精确的期望更新会将策略推向一个确定性的极限，该极限无论对于折扣目标还是未折扣目标而言都是全局最差的，同时参数会发散。作者通过调研文献来引出这项工作，并表明文献中仍然存在（大量相关误用）。

    arXiv:1906.07073v3 Announce Type: replace  Abstract: The policy gradient theorem describes the gradient of the expected discounted return with respect to an agent's policy parameters. However, many policy gradient methods omit the outer discount factor, yielding an update that need not equal the gradient of the discounted objective. Is the resulting update the gradient of another objective? This has been an open question for several years, and this lack of theoretical clarity has led to an abundance of misstatements in the literature. We answer this question for the resulting episode-summed, on-policy update by proving that it need not equal the gradient of any function. Further, we construct a counterexample in which exact expected updates drive the policy toward a deterministic limit that is globally pessimal with respect to both the discounted and undiscounted objectives, while the parameter diverges. We motivate this work by surveying the literature and showing that there remains a
    

