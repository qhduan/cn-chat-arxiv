# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Agentic Distributional Regression: Conditional Density Estimation through Agent-Written Programs](https://arxiv.org/abs/2610.37611) | 提出智能体分布式回归（ADR）方法，让语言模型智能体编写基于概率分布库的简短程序来估计条件密度，得到的可检验公式不仅透明可评估，还具备有限样本误差界保证。 |
| [^2] | [QD-PINN:A Physics-Informed Neural Network Approach to Bayesian Quickest Detection in High-Dimensional Diffusions](https://arxiv.org/abs/2610.37473) | 本文提出基于物理信息神经网络的QD-PINN框架，通过建立自由边界偏微分方程与无穷维随机优化问题之间的新型等价关系，实现了对高维扩散过程中贝叶斯最快检测最优停时边界的高效估计，并克服了维数灾难。 |
| [^3] | [An Input Skip Can Reparameterize a VAE Without Improving the Generator](https://arxiv.org/abs/2610.37456) | 输入跳跃连接虽能降低VAE的重建误差，但本质上只是对模型的重参数化，并不改进生成器，甚至可能使生成质量指标恶化。 |
| [^4] | [Graph Representation Learning via Adaptive Expectile Aggregation](https://arxiv.org/abs/2610.37450) | 提出ExpectileGCN框架，通过可学习的期望分位数水平对邻域信息进行自适应聚合，突破了传统均值聚合的局限，尤其能利用类别相关的尺度差异提升异配图上的分类性能，并具备稳定性与直推式泛化的理论保证。 |
| [^5] | [An end-to-end framework for uncertainty-aware validation of generative models](https://arxiv.org/abs/2610.37398) | 提出SIGMA端到端框架，通过高斯混合模型将密度估计与原则性不确定性量化相结合，并构建了一种将模型不确定性剖面化的新拟合优度检验，解决了生成模型验证中忽略估计误差导致模型终将被拒绝的问题。 |
| [^6] | [LinMix: Data Mixing as Linear Domain Attribution](https://arxiv.org/abs/2610.37390) | LinMix将数据混合建模为线性域归因问题，通过对带顶点约束的单纯形进行无超参数的顶点采样使各领域效应去相关，从而用可解释的线性模型系数直接读出每个领域对模型性能的贡献。 |
| [^7] | [Cost-based variable selection for prediction models](https://arxiv.org/abs/2610.37353) | 本文提出了一种基于净收益的决策分析变量选择算法，在临床预测模型构建中综合考虑变量预测能力提升带来的收益与测量该变量的危害（费用、时间、资源和患者负担），相比传统的向后消除法和LASSO等统计方法更符合临床决策需求。 |
| [^8] | [Low-degree polynomials lower bounds for clustering in small dimension](https://arxiv.org/abs/2610.37272) | 该论文利用对偶BCH码构造了一类矩张量与均匀分布在低阶完全一致的乘积伯努利混合分布，证明了当聚类数K远大于维度d且中心分离度较小时，低次多项式算法无法完成聚类，从而给出了比已有结果更强的计算复杂性下界。 |
| [^9] | [Doubly Robust Deep Kernel Tests for Structured Outcomes](https://arxiv.org/abs/2610.37227) | 该论文提出 DR-KTE-Deep 检验，通过在双重稳健的均值嵌入差异表示上最大化估计功效准则来学习深度特征，从而对图像、文本、图等结构化高维结果下两种干预是否诱导相同分布实现高功效且渐近有效的统计检验。 |
| [^10] | [R\`enyi-Regularized Flow-Based Variational Inference for Bayesian Calibration](https://arxiv.org/abs/2610.37169) | 本文提出一种基于Rényi散度正则化项的流式变分推断方法，解决了贝叶斯校准中标准变分推断无法捕捉被低概率区域分隔的多模态后验分布的问题，使近似后验能够覆盖先验支撑并表示多个后验模态。 |
| [^11] | [Physics-Guided Flow Matching: A Generative AI Framework for Global Total Electron Content Forecasting](https://arxiv.org/abs/2610.37159) | 提出了一种物理引导的流匹配生成式AI框架，将偏微分方程描述的电离层动力学与数据驱动的统计模式自适应融合，实现次日全球总电子含量预测。 |
| [^12] | [Is Your LLM Relay Serving What It Claims? A low-cost and high-power test for LLM Relay Auditing](https://arxiv.org/abs/2610.37086) | 提出分布式指纹审计方法DFA，通过反复查询简单的有限选项探针构建端点指纹并利用柯西组合检验聚合证据生成端点级p值，实现了低成本、鲁棒且统计可解释的LLM中转服务审计。 |
| [^13] | [Online gas classification at the edge](https://arxiv.org/abs/2610.36999) | 受昆虫嗅觉系统启发，本文提出一种基于稀疏高维二进制表示的轻量神经网络架构，能够在边缘设备上从原始传感器瞬态响应中在线识别气体，性能媲美优化的非线性SVM，且适用于功耗和样本受限的真实场景。 |
| [^14] | [Intrinsic Information-Theoretic Analysis of ReLU Nets](https://arxiv.org/abs/2610.36864) | 该论文利用ReLU网络将输入空间固有划分为凸线性区域这一特性，把互信息分解为可解释的几何项，并提出无需超参数、可估计的“路由信息”插件估计器，从而实现对深度网络的内在信息论分析。 |
| [^15] | [Boundary Geometry Determines Optimal Shallow ReLU Approximation under Weighted Coefficient Constraints](https://arxiv.org/abs/2610.36854) | 本文通过加权变差空间将区域几何纳入输出系数约束，证明边界几何决定了浅层ReLU网络的最优逼近率：最优指数由与小边界帽相关的单一几何增长指数刻画，并为任意凸体、平面凸体、$C^2$ 凸体等给出了精确的最优多项式逼近指数。 |
| [^16] | [Joint Distributional Copula Regression for Insurance Claim Frequency and Severity via Boosting](https://arxiv.org/abs/2610.36835) | 本文提出一种基于GAMLSS的模型化boosting方法来拟合结构化加性分布copula回归模型，实现保险理赔频率与严重度的联合建模，兼具加性预测器层面的可解释性、变量筛选能力与有竞争力的预测性能。 |
| [^17] | [LLM knowledge for improved Out-of-Distribution Tabular Prediction - A Cross-Validated Regularisation Approach](https://arxiv.org/abs/2610.36806) | 该研究提出将从大语言模型引导出的系数先验作为正则化回归的收缩目标，仅通过源域交叉验证选择提供者与先验结构，即可在五个表格数据集上将分布外预测的均方误差最高改善7.9%。 |
| [^18] | [Generative Refinement Learning for Temporal Interpolation in Scientific Machine Learning](https://arxiv.org/abs/2610.36751) | 提出iFLEX模型，用条件扩散模型学习修正基于时间加权平均的简单插值所产生的残差，从而在低时间分辨率数据下更准确地重建中间科学场。 |
| [^19] | [Conservative Dueling Bandits: Preference Learning under Baseline Constraints](https://arxiv.org/abs/2610.36708) | 本文提出了保守上下文对决老虎机问题及MaxInP-BC算法，通过效用差距而非绝对奖励来认证安全性，克服了标准对决老虎机探索过于激进的问题，并首次证明直接迁移CLUCB的安全测试会导致线性遗憾。 |
| [^20] | [PDE-Based Bounds on Optimal Exponents in Active Simple Hypothesis Testing](https://arxiv.org/abs/2610.36657) | 本文通过结合Vijayan（2025）的偏微分方程刻画与Hamilton-Jacobi方程粘性解的比较原理，为主动简单假设检验的最优误差指数建立了由凸函数、集合和下哈密顿松弛参数化的一般上下界，从而推广了文献中的许多已知结果。 |
| [^21] | [Deep Bayesian Approaches for Interpretable Nonlinear Structural System Identification](https://arxiv.org/abs/2610.36633) | 本教程系统介绍了将可学习表示嵌入概率状态空间框架的深度贝叶斯滤波方法，通过结合物理归纳偏置与后验推断，实现了非线性结构系统辨识中兼具物理可解释性与不确定性感知的数据驱动动力学建模。 |
| [^22] | [Wasserstein approximation of measures via orthogonal frames: statistical and computational complexities](https://arxiv.org/abs/2610.36581) | 该论文将无监督学习中的稀疏表示基选择任务转化为用正交框架轴上测度进行Wasserstein逼近的非光滑优化问题，并证明经验风险最小化和局部初始化的随机梯度方法均能以近参数速率 n^{-1/2} 恢复总体最优的正交框架。 |
| [^23] | [Algorithmics for Graphical Separation Criteria](https://arxiv.org/abs/2610.36502) | 该论文证明了多种图分离准则均可通过可达性的模块化变体在线性时间内完成检验（达到理论最优），并基于 CIfly 框架提供了相应的 R 和 Python 实现。 |
| [^24] | [Second-order Sinkhorn Geometry and $\sqrt{n}$-Inference](https://arxiv.org/abs/2610.36488) | 本文将Sinkhorn散度的黎曼几何框架扩展至二阶，证明了Sinkhorn度量张量的Hadamard可微性并推导出Levi-Civita协变结构，从而支持√n速率的统计推断。 |
| [^25] | [Confidence Is Not a Promise: A Reference Audit of Typed Decision Models](https://arxiv.org/abs/2610.36311) | 该审计研究表明，类型化决策模型输出的概率只许可群体层面的统计保证而非单次决策的承诺——点估计阈值在190次评估中有147次失守，Beta下界规则虽无保证却比Learn then Test裁决更多项目，而跨语言与输入噪声会轻易打破在干净英语文本上校准的准确率承诺。 |
| [^26] | [Directional Statistics on Circles and Spheres: A New Angle on Data](https://arxiv.org/abs/2610.36227) | 本文指出角度数据天然位于圆和球而非直线上，直接算术平均可能导致错误结论（如朝北方位平均后指向南方），并通过风向与扩散MRI等实例介绍圆形/轴向均值、方向变异性和概率模型等方向统计工具，主张在选择统计方法前应先识别数据的几何结构与对称性。 |
| [^27] | [Approximation Families and the Statistical Efficiency of Denoising Score Matching](https://arxiv.org/abs/2610.36205) | 该论文证明了在 Ornstein-Uhlenbeck 流下对非高斯分布进行多项式得分匹配时，得分估计方差会沿多项式次数的某个子序列指数增长，且通过扩散半群平均损失（OISM）虽能降低每一固定次数的方差，却无法在足够小的时间消除这种指数不稳定性。 |
| [^28] | [Anytime Valid Confidence Bands for Distribution Functions](https://arxiv.org/abs/2610.36149) | 该论文为数据流的分布函数构建了在所有时间和所有点上同时有效、且允许观测值任意依赖过去的置信带，并进一步将其推广为适用于任意索引化时间平均分布函数族的统一构造框架。 |
| [^29] | [Symmetry Breaking Is the Price of Finite Capacity](https://arxiv.org/abs/2610.36138) | 这篇论文证明，当模型容量有限时，即使问题本身具有对称性（如旋转不变性），最优模型也会以精确可预测的量破坏对称性，其代价恰好等于最优单个方向与最优完整方向块之间的信号差，且这一结论同样适用于注意力头。 |
| [^30] | [Dirichlet Process Mixtures within Semiparametric Bayesian Networks: A Novel Conditional Density Estimation Approach](https://arxiv.org/abs/2610.36110) | 本文提出在半参数贝叶斯网络中用狄利克雷过程混合模型（DPM）替代核密度估计进行非参数条件概率分布建模，通过数据自适应聚类获得更紧凑的表示，在降低推理成本的同时提升了密度估计性能。 |
| [^31] | [Minimax-Optimal Small-Loss Bounds for Online Linear Optimisation on $\ell_p$ Balls](https://arxiv.org/abs/2610.36097) | 本文针对 $\ell_p$ 球上的全信息在线线性优化问题，提出了一种基于有界径向正则化项的 FTRL 算法，在无需预知时间范围与小损失量的情况下，取得了与下界匹配的极小极大最优小损失遗憾保证，并通过切换扩展进一步保证遗憾不超过 FTL 遗憾的两倍。 |
| [^32] | [Learning on-the-fly quasi-stationarity diagnostics from low-dimensional surrogates](https://arxiv.org/abs/2610.36078) | 该论文提出了一种基于非线性概率流粗粒化与基于仿真的推断的准平稳性诊断新方法，通过由低维集体变量定义的代理过程即时学习去相关时间估计，以替代传统的MCMC诊断方法。 |
| [^33] | [Anchored Stein Variational Gradient Descent Algorithm for Non-smooth Sampling](https://arxiv.org/abs/2610.36018) | 该论文提出锚定Stein变分梯度下降算法，通过引入更光滑的参考势能与锚定权重，无需计算目标势能梯度即可对非光滑目标分布进行采样，并在低正则性条件下给出了算法的非渐近收敛界。 |
| [^34] | [A Graph-informed Kriging model for sample-efficient Bayesian optimization of wind farm layouts](https://arxiv.org/abs/2610.35975) | 该论文提出一种图信息克里金框架，通过有向气动图将可复用的风机间气动交互信息嵌入代理模型的均值函数，从而显著提升高维风电场布局贝叶斯优化的样本效率与年发电量预测精度。 |
| [^35] | [Data-driven Discovery of Lie Symmetries and First Integrals with Latent ODEs](https://arxiv.org/abs/2610.35967) | 提出LatentLieODE方法，结合自编码器与潜变量神经ODE，能直接从视频等时间序列观测数据中自动发现动力系统的线性李对称性与首次积分。 |
| [^36] | [Calibrated for Which Policy? Conformal Uncertainty for Fixed World Models Under Intervention](https://arxiv.org/abs/2610.35877) | 该论文提出了一种加权保形校准框架：当部署策略发生干预变化时，通过分解为状态占据与动作倾向项的密度比，将固定世界模型的边际覆盖率保证迁移到目标策略，并利用源-目标域分类器的对数损失遗憾给出密度比估计误差的显式上界。 |
| [^37] | [When Does Data Value Reduce to Class Balance? A Coverage View of Per-Point Data Valuation](https://arxiv.org/abs/2610.34801) | 论文的核心贡献是从“覆盖”视角证明：当预算紧张且数据模式冗余时，基于固定逐点分数的 top-k 数据选择会因重复覆盖已选模式而产生闭式超额风险，数据价值本质上取决于已选点集，应采用自适应的类别平衡式选择规则。 |
| [^38] | [BRIDGE: An Acceptance Gate to Block False Improvements in Model Updates, with a Retrospective Clinical Evaluation](https://arxiv.org/abs/2610.29515) | BRIDGE 提出了一种模型无关的接受门控机制，通过交叉拟合与 bootstrap 置信下界检验来拦截模型更新中的虚假改进，将虚假改进率从约 20% 降至接近零，并经 12 项回顾性临床任务验证有效。 |
| [^39] | [Hilbert-Scale Stability of Posterior Diffusion Priors for Ill-Posed Signal Recovery](https://arxiv.org/abs/2610.27030) | 该论文在希尔伯特尺度框架下为线性逆问题的后验扩散采样建立了稳定性理论，在正向算子平滑条件、真值源条件与分数误差弱范数控制下，证明了学习到的扩散先验在不适定信号恢复中能够给出将噪声、分数近似与反向时间效应分离的非渐近恢复保证。 |
| [^40] | [Advantage of Sample Complexity in Quantum PAC Learning Requires Inverse Access to State-Preparation Unitaries](https://arxiv.org/abs/2609.38403) | 本文证明了在量子PAC学习中，查询复杂度对精度参数依赖的改进优势本质上必须依赖于对态制备酉算子逆的访问，而仅有前向访问的酉算子无法带来任何样本复杂度上的优势。 |
| [^41] | [Nuclear Norm-Regularized Bayesian Matrix Completion](https://arxiv.org/abs/2609.30078) | 本文提出了首个针对未知噪声方差下核范数正则化贝叶斯矩阵补全模型的采样器，并给出了复杂度为矩阵维度和目标精度倒数之多项式的显式非渐近保证。 |
| [^42] | [Momentum in large-batch training: Polyak enlarges the critical batch size, Nesterov improves data efficiency](https://arxiv.org/abs/2609.02728) | 该论文在幂律核回归框架下证明，在单遍大批量训练中 Polyak 动量可将临界学习率随批大小线性放大（从而扩大临界批大小约 1/(1-ρ) 倍），而 Nesterov 动量的临界学习率以 $B^\beta$（β>1）的更快速度增长，从而显著提升数据效率，并给出了刻画完整风险动力学的标度律与三区制批大小相图。 |
| [^43] | [Revenge of Monosemanticity: Specialized Neurons Improve Data Efficiency in MLPs](https://arxiv.org/abs/2608.24007) | 本文证明多层感知器通过发展特化神经元学习局部低维表示，而非全局低维表示，从而在聚类数据的回归问题中显著提升数据效率。 |
| [^44] | [K\"ahler landscapes for complex neural network descents and guarantees including a search and destroy of the Calabi-Yau manifold](https://arxiv.org/abs/2608.19584) | 本文提出了一种在复参数神经网络中使用Kähler信息度量和自然梯度下降的新方法，并针对Calabi-Yau流形上的不良曲率条件提供了理论保证，通过几何定义的全局势实现了搜索与消灭策略。 |
| [^45] | [ProxyGuard: Direct Reliability Inference for Randomized Data Release Mechanisms with Shared Targets](https://arxiv.org/abs/2608.18643) | ProxyGuard提出了一种基于密封目标集和预设风险的新方法，能够在随机数据发布机制中同时控制无效发布被误认为有效和有效发布未被认证的错误，无需独立目标批次或发布级依赖假设。 |
| [^46] | [Computational and Statistical Guarantees of the c-Rectified flow](https://arxiv.org/abs/2608.02487) | 本文提出代价感知的c-校正流，克服了普通校正流可能无法收敛到最优传输耦合（如高斯情形下协方差矩阵不可交换时失效）的缺陷，并为迭代校正流首次提供了计算复杂度与统计误差方面的理论保证。 |
| [^47] | [Distributionally Faithful Imputation via Positive Semi-Definite Kernel Density Estimation](https://arxiv.org/abs/2607.07767) | 该论文提出 PSD-Impute 模型，将 MCAR 缺失值插补重新表述为基于半正定核密度的密度估计问题，构建了具有闭式边缘解的凸优化问题并具备统计一致性，可同时实现单次与多重插补。 |
| [^48] | [An Asymptotic Theory of Chain-of-Thought in In-Context Learning](https://arxiv.org/abs/2606.03217) | 该论文建立了上下文学习中思维链推理的首个渐近理论，精确推导出泛化误差随推理深度、预训练数据量和上下文长度变化的公式，并揭示了指数级与多项式级改进、饱和和过度思考之间的相变及最优推理深度的缩放规律。 |
| [^49] | [CalArena: A Large-Scale Post-Hoc Calibration Benchmark](https://arxiv.org/abs/2605.30188) | 该论文提出了CalArena，一个覆盖近2000个实验、融合经典模型与基础模型预测、并提供数十种校准方法统一可复现实现的大规模标准化事后校准基准，用以系统评估哪些校准方法在实践中真正有效。 |
| [^50] | [Parametric Rates for Nonparametric Conditional-Moment Bilevel Optimization](https://arxiv.org/abs/2605.21341) | 该论文提出正交化超梯度估计器 OBiGrad，将超梯度偏差降低为滋扰误差的乘积，使条件矩双层优化达到参数化统计速率 O(T^{-1/2}+n^{-1/2})，并在固定迭代预算下显著降低了所需的样本量界限。 |
| [^51] | [Multi-Variable Conformal Prediction: Optimizing Prediction Sets using Multiple Calibration Variables](https://arxiv.org/abs/2605.12341) | 该论文提出多变量保形预测（MCP）框架，通过在单个优化问题中联合优化和校准多个预测集参数，消除了校准数据分割的需求且不牺牲有限样本覆盖保证，并基于情景理论提出了RemMCP和RelMCP两个可求解的变体。 |
| [^52] | [Phases of Muon: When Muon Eclipses SignSGD](https://arxiv.org/abs/2605.09552) | 该论文推导了随机Muon在高维矩阵最小二乘问题上的显式确定性动力学，发现Muon对有效批次大小存在非线性敏感性——在精确白化且有效批次较小时，Muon难以分辨较小的特征模态而收敛变慢（动量与有限Newton–Schulz迭代可缓解此效应），而SignSGD并无此相变，从而阐明了Muon超越SignSGD的条件。 |
| [^53] | [Optimal Experiments for Partial Causal Effect Identification](https://arxiv.org/abs/2605.06993) | 该论文提出“最大效力问题”这一新框架，用于在成本约束下选择最优实验组合以最大程度收紧部分可识别因果查询的界限，证明了该问题的NP难性，给出了评估认识效力的通用方法，并引入基于因果图结构的剪枝准则（如线性时间的路径拦截规则）来控制超指数级搜索空间。 |
| [^54] | [On the Optimal Sample Complexity of Offline Multi-Armed Bandits with KL Regularization](https://arxiv.org/abs/2605.02141) | 本文在多臂老虎机设置下对KL正则化离线学习算法KL-PCB给出了精确分析，证明其在大、小正则化条件下分别达到 $\tilde{O}(\eta SAC^{\pi^*}/\epsilon)$ 和 $\tilde{\Omega}(SAC^{\pi^*}/\epsilon^2)$ 的样本复杂度，从而基本刻画了KL正则化离线学习的最优样本复杂度。 |
| [^55] | [Expert-Guided Class-Conditional Goodness-of-Fit Scores for Interpretable Classification with Informative Missingness](https://arxiv.org/abs/2604.14809) | 该论文提出了一个专家引导的类条件拟合优度分数框架，将专家先验知识编码为可解释的拟合优度特征，以解决信息性缺失下的可解释分类问题，并成功应用于《全面禁止核试验条约》的地震监测合规评估。 |
| [^56] | [New explanations and inference for least angle regression](https://arxiv.org/abs/2602.02491) | 本文为最小角回归（LAR）提出了一个新的推断框架，将LAR解释为估计一个按总体相关参数递减排序的变量“路径”，从而揭示其新的数学性质并解决了诸如算法合适终止点等长期悬而未决的问题。 |
| [^57] | [Learning Sequential Decisions from Multiple Sources via Group-Robust Markov Decision Processes](https://arxiv.org/abs/2602.01825) | 该论文提出基于共享线性MDP表示的d-矩形分布鲁棒决策框架及GR-PEVI离线算法，通过按特征最坏情况聚合与数据依赖悲观惩罚，在鲁棒部分覆盖条件下实现了跨源异质多环境序贯决策学习的有限样本次优性保证。 |
| [^58] | [OneFlowSBI: One Model, Many Queries for Simulation-Based Inference](https://arxiv.org/abs/2601.22951) | 提出OneFlowSBI，一种掩码流匹配框架，通过在参数与观测的联合分布上训练单一网络，并利用三值掩码作为输入而非token化方式，实现任意条件或边缘分布的高效采样，在取得更低后验误差的同时大幅提升采样速度并具备大规模变量扩展能力。 |
| [^59] | [Latent Nonlinear Denoising Score Matching for Enhanced Learning of Structured Distributions](https://arxiv.org/abs/2512.06615) | 本文提出潜空间非线性去噪得分匹配（LNDSM），通过将结构感知的非线性潜空间扩散与VAE训练框架结合，并指定能反映数据定性特征的参考分布（如高斯混合模型），从而将结构信息直接融入加噪动力学，增强对结构化分布的学习。 |
| [^60] | [Orlicz-Sobolev with Musielak: An Efficient Regularization Approach for Graph-based IPM](https://arxiv.org/abs/2510.25591) | 该论文提出了一种基于Orlicz几何的广义Sobolev IPM方法，通过引入一类特定凸函数突破了传统Sobolev IPM受限于L^p几何结构的局限，为基于图的积分概率度量提供了高效的正则化途径。 |
| [^61] | [On The Variability of Concept Activation Vectors](https://arxiv.org/abs/2509.24058) | 本文对概念激活向量（CAV）的构建过程进行了细粒度的理论分析，量化了其因训练集随机采样而产生的变异性，并在四种不同模态的真实数据集上通过实验验证了结果的普适性。 |
| [^62] | [When is Fine-Tuning Scalable? Certified Recovery from Noisy Data and Linear-Cost Fine-Tuning on Structured Task Shifts](https://arxiv.org/abs/2509.00924) | 本文提出了一个模型特定的两阶段认证程序，证明单头Transformer在结构化任务偏移下仅需线性成本的微调，即可从噪声数据中以极小极大最优速率可认证地恢复源任务并实现分布外适应。 |
| [^63] | [Two-sample comparison through additive tree models for density ratios](https://arxiv.org/abs/2508.03059) | 本文提出基于新损失函数“平衡损失”的密度比加性树模型，使树模型既能利用监督学习算法高效训练，又能通过广义贝叶斯推断实现不确定性量化，从而完成双样本比较任务。 |
| [^64] | [Improved Regret Bounds for Gaussian Process Upper Confidence Bound in Bayesian Optimization](https://arxiv.org/abs/2506.01393) | 本文通过刻画GP-UCB输入序列的集中行为以更精细地分析高斯过程信息增益，证明了GP-UCB在Matérn核下可达到$\tilde{O}(\sqrt{T})$、在平方指数核下可达到$O(\sqrt{T \ln^2 T})$的遗憾界，填补了与已知最优遗憾界之间的差距。 |
| [^65] | [Doubly Robust Alignment for Large Language Models](https://arxiv.org/abs/2506.01183) | 提出了一种双重稳健的偏好优化算法（DRPO），只要偏好模型或参考策略中有一个被正确设定，即可实现一致的大语言模型对齐，有效解决了RLHF中的模型设定错误问题。 |
| [^66] | [Foreclassing: From Forecasts to Decisions](https://arxiv.org/abs/2503.04956) | 本文提出了“Foreclassing”这一全新机器学习问题，即用时间序列的未来观测值来定义其类别标签，给出了数学表述与相关理论结果，并提出了联合执行预测与分类的神经网络模型家族ForeClassNet作为研究基准。 |
| [^67] | [A Mechanistic Interpretation of Prescribed-Parameter Convolutional Kernels](https://arxiv.org/abs/2502.17061) | 本文从机制上解释了ROCKET等预设参数卷积方法在时间序列分类中有效的原因：证明其PPV池化特征在周期边界下对平移具有精确不变性，并揭示了有限随机核库会聚集在由参数分布决定的成对几何结构周围。 |
| [^68] | [Non-asymptotic Estimation of Sparse Partial Correlation via Joint Partial Regression](https://arxiv.org/abs/2502.08414) | 该论文提出了一种两阶段联合偏回归估计方法，能够以非渐近速率同时估计稀疏高维偏相关矩阵与逆协方差矩阵，并同时保证半正定性。 |
| [^69] | [Targeted Learning for Inference on Data Fairness](https://arxiv.org/abs/2502.04309) | 该论文提出利用目标学习这一非参数推断框架对训练数据生成过程的公平性进行统计推断，推导出人口均等、机会均等和条件互信息的估计量，仿真表明其能提高置信区间覆盖率并减少误检。 |
| [^70] | [The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex Regularization](https://arxiv.org/abs/2411.19193) | 提出截断势垒策略梯度框架，将截断水平 m 解释为失败风险的“价格”，证明训练价值可精确分解为失败前效用、余量成本与价格×风险三部分，并在凸正则化下实现了总体训练流的指数收敛。 |
| [^71] | [Temporal difference learning adapts to incomplete coverage: an oracle inequality over subgraphs](https://arxiv.org/abs/2411.09731) | 本文证明在表格设定下，普通的时序差分（TD）学习能自动兼具蒙特卡洛方法“仅需目标状态充分访问”与TD自身“渐近高效”的双重优势，并以子图Bellman方差为核心量给出了一个非渐近oracle不等式：无需预先指定任何子图，TD即对每个覆盖充分的子图自适应地达到最优误差界。 |
| [^72] | [An Alternative Graphical Lasso Algorithm for Precision Matrices](https://arxiv.org/abs/2403.12357) | 通过重新参数化精准矩阵最后一列，将正则化正态对数似然分解为两个易于最小化的凸函数，其中一个是Lasso回归问题，从而开发了一种简单的块坐标下降算法来计算GLasso更新，性能可与DP-GLas相媲美。 |

# 详细

[^1]: 智能体分布式回归：通过智能体编写的程序进行条件密度估计

    Agentic Distributional Regression: Conditional Density Estimation through Agent-Written Programs

    [https://arxiv.org/abs/2610.37611](https://arxiv.org/abs/2610.37611)

    提出智能体分布式回归（ADR）方法，让语言模型智能体编写基于概率分布库的简短程序来估计条件密度，得到的可检验公式不仅透明可评估，还具备有限样本误差界保证。

    

    本文提出了一种利用语言模型智能体从独立同分布样本中估计条件密度的方法。该智能体将估计器编写为一个简短的程序，对于每个协变量取值，该程序返回响应变量的估计条件分布。程序由概率分布库构建而成，因此返回的分布可以是参数分布、混合分布或多变量分解，其参数由协变量的显式函数给出。由于程序中的运算和常数的数量有限，拟合的条件密度是一个可以独立于智能体进行检验和求值的公式，而模型结构和拟合程序的选择则留给智能体完成。我们将这种方法称为智能体分布式回归。将返回的程序视为一个随机的条件密度估计器，我们在显式假设条件下推导出了有限样本误差界……

    arXiv:2610.37611v1 Announce Type: cross  Abstract: In this paper we propose a method for estimating conditional densities from an iid sample using a language-model agent. The agent writes the estimator as a short program that, for each covariate value, returns the estimated conditional distribution of the response. The program is built from a library of probability distributions, so the returned law can be a parametric distribution, a mixture or a multivariate factorization, with parameters given by explicit functions of the covariates. Since the number of operations and constants in the program is limited, the fitted conditional density is a formula that can be inspected and evaluated independently of the agent, while the choice of the model structure and of the fitting procedure is left to the agent. We call this approach agentic distributional regression (ADR). Treating the returned program as a random conditional density estimator, we derive a finite-sample error bound under explic
    
[^2]: QD-PINN：一种基于物理信息神经网络的贝叶斯高维扩散过程最快检测方法

    QD-PINN:A Physics-Informed Neural Network Approach to Bayesian Quickest Detection in High-Dimensional Diffusions

    [https://arxiv.org/abs/2610.37473](https://arxiv.org/abs/2610.37473)

    本文提出基于物理信息神经网络的QD-PINN框架，通过建立自由边界偏微分方程与无穷维随机优化问题之间的新型等价关系，实现了对高维扩散过程中贝叶斯最快检测最优停时边界的高效估计，并克服了维数灾难。

    

    最快检测问题（QDPs）广泛应用于众多领域。然而，相关变分不等式的互补性结构给传统数值方法带来了巨大挑战，尤其是在高维场景下。我们提出了一种基于物理信息神经网络（PINN）的框架，用于估计具有高维扩散过程和空间相关性的最快检测问题中的最优停时边界，同时克服维数灾难。在理论方面，我们建立了自由边界偏微分方程与无穷维随机优化问题之间的新型等价关系，从而严格论证了使用有限维神经网络逼近真实解的合理性。此外，我们通过使用深度集成和蒙特卡洛Dropout对学习到的停时边界进行不确定性量化，评估了所提框架的鲁棒性。结果表明，在多次独立训练运行和随机过程中变异性较低。

    arXiv:2610.37473v1 Announce Type: new  Abstract: Quickest detection problems (QDPs) arise in a wide range of applications. However, the complementarity structure of the associated variational inequalities poses substantial challenges for traditional numerical methods, particularly in high-dimensional settings. We propose a PINN-based framework for estimating the optimal stopping boundary in QDPs with high-dimensional diffusion processes and spatial dependence, while overcoming the curse of dimensionality. Theoretically, we establish a novel equivalence between the free-boundary PDE and an infinite-dimensional stochastic optimization problem, rigorously justifying the use of finite-dimensional neural networks to approximate the true solution. Furthermore, we assess the robustness of the proposed framework by quantifying uncertainty in the learned stopping boundaries using deep ensembles and Monte Carlo dropout. The results show low variability across independent training runs and stocha
    
[^3]: 输入跳跃连接可在不改进生成器的情况下对VAE进行重参数化

    An Input Skip Can Reparameterize a VAE Without Improving the Generator

    [https://arxiv.org/abs/2610.37456](https://arxiv.org/abs/2610.37456)

    输入跳跃连接虽能降低VAE的重建误差，但本质上只是对模型的重参数化，并不改进生成器，甚至可能使生成质量指标恶化。

    

    输入跳跃连接可以在不改变模型新样本分布的情况下降低重建误差。对于具有常数输入跳跃 0≤α<1 的高斯变分自编码器（VAE），通过将其解码器输出乘以 b=1-α、噪声方差乘以 b²，可以把无跳跃连接的模型写成跳跃连接的形式。此时观测变量与潜变量的分布以及精确的生成分数保持不变，而重建均方误差（MSE）则被乘以 b²。由于优化过程依赖于参数坐标，分开训练可能产生不同的生成器。在一个密度已知的高斯实验中，同时改变跳跃与观测方差会降低MSE，但会使能量分数恶化，并使通过重要性采样得到的负对数密度估计变差。在七条年度深海海啸评估与报告浮标（DART）记录上，在相同观测方差下的能量分数差异……（摘要在此处被截断）

    arXiv:2610.37456v1 Announce Type: cross  Abstract: An input skip can reduce reconstruction error while leaving a model's distribution of new samples unchanged. For a Gaussian variational autoencoder (VAE) with constant input skip $0\leq\alpha<1$, a model without a skip can be written in skip form by multiplying its decoder output by $b=1-\alpha$ and its noise variance by $b^2$. The distribution of observations and latent variables and its exact generative scores stay the same, while reconstruction mean squared error (MSE) is multiplied by $b^2$. Separate training can produce different generators because optimization depends on the parameter coordinates. In a Gaussian experiment with known density, changing skip and observation variance together reduces MSE but worsens energy score and the estimate of negative log density from importance sampling. On seven annual Deep-ocean Assessment and Reporting of Tsunamis (DART) records, energy score differences at the same observation variance hav
    
[^4]: 基于自适应期望分位数聚合的图表示学习

    Graph Representation Learning via Adaptive Expectile Aggregation

    [https://arxiv.org/abs/2610.37450](https://arxiv.org/abs/2610.37450)

    提出ExpectileGCN框架，通过可学习的期望分位数水平对邻域信息进行自适应聚合，突破了传统均值聚合的局限，尤其能利用类别相关的尺度差异提升异配图上的分类性能，并具备稳定性与直推式泛化的理论保证。

    

    图神经网络（GNN）在很大程度上依赖于邻域聚合，然而传统的均值聚合可能无法捕获与类别相关的分布信息，尤其是在异配图中。我们提出了ExpectileGCN，这是一种自适应的图聚合框架，它引入了可学习的期望分位数水平，以捕获均值之外的邻域信息。所提出的算子能够在最小值型与最大值型行为之间连续地调整聚合方式，而均值聚合只是其一种特例。基于sigmoid平滑和不定点迭代的可微近似使端到端优化成为可能。在理论上，我们为ExpectileGCN建立了稳定性性质和直推式泛化保证。我们进一步研究了其在异配邻域模型下的分类风险。对于信息丰富的邻域，我们证明期望分位数聚合可以利用类别相关的尺度差异来实现（摘要在此处截断）。

    arXiv:2610.37450v1 Announce Type: new  Abstract: Graph neural networks (GNNs) rely critically on neighborhood aggregation, yet conventional mean aggregation may fail to capture class-relevant distributional information, particularly in heterophilic graphs. We propose ExpectileGCN, an adaptive graph aggregation framework that introduces a learnable expectile level to capture neighborhood information beyond the mean. The proposed operator continuously adjusts aggregation between minimum- and maximum-type behaviors, with mean aggregation as a special case. A differentiable approximation based on sigmoid smoothing and fixed-point iterations enables end-to-end optimization. Theoretically, we establish stability properties and transductive generalization guarantees for ExpectileGCN. We further investigate its classification risk under heterophilic neighborhood models. For informative neighborhoods, we show that expectile aggregation can exploit class-dependent scale differences to achieve a 
    
[^5]: 一个用于生成模型不确定性感知验证的端到端框架

    An end-to-end framework for uncertainty-aware validation of generative models

    [https://arxiv.org/abs/2610.37398](https://arxiv.org/abs/2610.37398)

    提出SIGMA端到端框架，通过高斯混合模型将密度估计与原则性不确定性量化相结合，并构建了一种将模型不确定性剖面化的新拟合优度检验，解决了生成模型验证中忽略估计误差导致模型终将被拒绝的问题。

    

    使用生成模型进行密度估计是物理科学中的一项常见任务，例如粒子物理学中的背景估计。然而，现有的机器学习技术缺乏能够将不确定性传播到统计检验中的量化方法。这使得生成模型的验证只能局限于标准的拟合优度检验，而在这类检验中，生成模型被视为完全精确的（点零假设）。但拟合于有限训练集的模型永远不可能是完全精确的，因此如果忽略其不确定性，一旦测试集足够大以至于能够分辨出这种估计误差，任何足够强大的拟合优度检验最终都会拒绝该模型。我们提出了SIGMA，一个将密度估计与原则性不确定性量化相结合的端到端框架。我们使用高斯混合模型，其不确定性是可解析处理的，并进一步构建了一种新的拟合优度检验，其中这些不确定性被纳入剖面化处理。

    arXiv:2610.37398v1 Announce Type: new  Abstract: Density estimation with generative models is a common task in the physical sciences, for example in background estimation in particle physics. Existing machine learning techniques, however, lack methods to quantify uncertainties that can be propagated into a statistical test. That limits the validation of generative models to standard goodness-of-fit tests, in which the generative model is treated as exact (point-null hypothesis). But a model fitted to a finite training set is never exact, so if its uncertainties are neglected, any sufficiently powerful goodness-of-fit test will eventually reject it once the test set is large enough to resolve this estimation error. We propose SIGMA, an end-to-end framework combining density estimation with principled uncertainty quantification. We use Gaussian mixture models, for which uncertainties are tractable, and further construct a new goodness-of-fit test in which these uncertainties are profiled
    
[^6]: LinMix：将数据混合视为线性域归因

    LinMix: Data Mixing as Linear Domain Attribution

    [https://arxiv.org/abs/2610.37390](https://arxiv.org/abs/2610.37390)

    LinMix将数据混合建模为线性域归因问题，通过对带顶点约束的单纯形进行无超参数的顶点采样使各领域效应去相关，从而用可解释的线性模型系数直接读出每个领域对模型性能的贡献。

    

    语言模型的性能在很大程度上取决于预训练数据语料库中各领域数据的混合比例。流行的数据混合方法通过在随机混合的数据上训练许多小型代理模型，再拟合一个回归模型来从混合比例预测性能，从而选择这些领域权重。我们将这一做法视为一个领域归因问题，其中回归衡量每个领域对目标指标的贡献。从这一视角看，对代理数据混合进行标准的狄利克雷采样是一种糟糕的回归设计：它需要调参，且无法支持对数百个领域进行线性拟合。基于经典的最优实验设计，我们提出LinMix，它在带顶点约束的单纯形顶点处采样混合，使每个领域要么达到其上限，要么完全缺失。这种无超参数的顶点采样使各领域的效应去相关，从而得到一个可解释的线性模型，其系数即为各领域的归因。最优混合……（摘要在此处截断）

    arXiv:2610.37390v1 Announce Type: cross  Abstract: Language model performance depends strongly on the mixture of domains in the pretraining data corpus. Popular data mixing methods select these domain weights by training many small proxy models on random mixes and then fitting a regression model to predict performance from the mix. We recognize this practice as a domain attribution problem, where the regression measures each domain's contribution to the target metric. Through this lens, the standard Dirichlet sampling of proxy data mixes is a poor regression design that requires tuning and cannot support a linear fit for hundreds of domains. Building on classical optimal experimental design, we introduce LinMix, which samples mixes at the vertices of the capped simplex so that each domain is at its cap or absent. This hyperparameter-free vertex sampling decorrelates domain effects, enabling an interpretable linear model whose coefficients are the domain attributions. The optimal mix gi
    
[^7]: 基于成本的预测模型变量选择

    Cost-based variable selection for prediction models

    [https://arxiv.org/abs/2610.37353](https://arxiv.org/abs/2610.37353)

    本文提出了一种基于净收益的决策分析变量选择算法，在临床预测模型构建中综合考虑变量预测能力提升带来的收益与测量该变量的危害（费用、时间、资源和患者负担），相比传统的向后消除法和LASSO等统计方法更符合临床决策需求。

    

    目标：临床预测模型的变量选择通常基于与模型拟合相关的统计学准则。统计方法忽略了误分类的成本以及测量变量本身的危害。我们旨在开发一种基于净收益的决策分析变量选择算法。材料与方法：所提出的算法通过使用交叉验证穷举比较所有变量组合的净收益来选择预测变量集合，并针对预测变量较多的场景提供了一种基于向后消除法的扩展方法。当更好的预测性所带来的净收益增加超过测量该变量危害所导致的净收益减少时，该变量将被纳入。危害包括经济费用、时间、资源分配以及患者负担。我们通过一个模拟示例和三个真实案例研究对该算法进行了说明，并将其性能与传统的向后消除法和LASSO进行了比较。

    arXiv:2610.37353v1 Announce Type: cross  Abstract: Objective: Variable selection for clinical prediction models is commonly done on statistical grounds related to model fit. Statistical approaches ignore the cost of misclassifications and the harm of measuring a variable. We aim to develop a decision-analytic variable selection algorithm based on net benefit. Materials and Methods: The proposed algorithm selects the set of predictors by exhaustively comparing the net benefit of all combinations using cross validation, with an extension based on backward elimination for settings with many predictors. Variables are selected if the increased net benefit from better predictiveness exceeds the net benefit reduction from the harm of variable measurement. Harm includes financial expense, time, resource allocation, and patient burden. We illustrate the algorithm through a simulated example and three real case studies. We compare its performance to traditional backward elimination and LASSO. Re
    
[^8]: 小维数聚类问题的低次多项式下界

    Low-degree polynomials lower bounds for clustering in small dimension

    [https://arxiv.org/abs/2610.37272](https://arxiv.org/abs/2610.37272)

    该论文利用对偶BCH码构造了一类矩张量与均匀分布在低阶完全一致的乘积伯努利混合分布，证明了当聚类数K远大于维度d且中心分离度较小时，低次多项式算法无法完成聚类，从而给出了比已有结果更强的计算复杂性下界。

    

    我们研究从 $\mathbb{R}^d$ 中 $K$ 个各向同性分布的混合里抽取的 $n$ 个观测值的聚类问题的计算极限，重点关注组数 $K$ 远大于维度 $d$ 的情形。在此情形下，基于张量的方法可以在中心之间具有多对数级别分离度时恢复划分，但它们所需的样本量是 $dK$ 的一个很大且非显式的多项式；而现有的低次下界只能表明 $n$ 必须大于 $dK^2$。我们引入了 $\{0,1\}^d$ 上 $2K$ 个乘积伯努利分布的混合，其中 $K=(d+1)^m$，其中心由一个对偶 BCH 码构造。这一构造确保观测值的矩张量在不高于 $2m+1$ 阶时与 $\{-1,1\}^d$ 上均匀分布的矩张量完全一致。对于任意固定的 $m$，我们建立了一个低次多项式下界，表明只要分离度 $\Delta$ 满足……（原文摘要在此处截断）

    arXiv:2610.37272v1 Announce Type: new  Abstract: We study the computational limits of clustering $n$ observations drawn from a mixture of $K$ isotropic distributions in $\mathbb R^d$, in the regime where the number $K$ of groups is much larger than the dimension $d$. In this regime, tensor-based procedures recover the partition with a polylogarithmic separation between the centers, but they require a sample size that is a large and non-explicit polynomial of $dK$, whereas the existing low-degree lower bounds only show that $n$ must be larger than $dK^2$. We introduce a mixture of $2K$ product Bernoulli distributions on $\{0,1\}^d$, with $K=(d+1)^m$, whose centers are built from a dual BCH code. This construction ensures that the moment tensors of the observations coincide with those of the uniform distribution on $\{-1,1\}^d$ up to the order $2m+1$. For any fixed $m$, we establish a low-degree polynomial lower bound showing that clustering is hard as long as the separation $\Delta$ sat
    
[^9]: 面向结构化结果的双重稳健深度核检验

    Doubly Robust Deep Kernel Tests for Structured Outcomes

    [https://arxiv.org/abs/2610.37227](https://arxiv.org/abs/2610.37227)

    该论文提出 DR-KTE-Deep 检验，通过在双重稳健的均值嵌入差异表示上最大化估计功效准则来学习深度特征，从而对图像、文本、图等结构化高维结果下两种干预是否诱导相同分布实现高功效且渐近有效的统计检验。

    

    我们提出了一种检验方法，用于判断两种干预是否诱导出相同的结果分布，适用于结果为结构化且高维的情形，例如图像、文本和图数据。我们将干预分布表示为在所学习的深度特征上的均值嵌入，使得处理效应成为均值嵌入差异的一种双重稳健表示。我们的检验方法 DR-KTE-Deep 通过最大化该处理效应表示上的估计功效准则来优化所学习的特征。为该学习目标提供理论依据，我们证明了该功效准则刻画了检验在固定备择假设下的渐近信噪比，并为其正则化估计推导出一致的有限样本界。我们进一步证明，在所学核固定的情况下，只要干扰参数的估计速率合适，基于保留样本的检验在原假设下是渐近有效的。从实证结果来看，DR-KTE-Deep 相比固定……方法取得了显著更高的检验功效（摘要原文在此处截断）。

    arXiv:2610.37227v1 Announce Type: new  Abstract: We propose a test of whether two interventions induce the same outcome distribution when outcomes are structured and high-dimensional, such as images, text and graphs. We represent the interventional distributions as mean embeddings on learned deep features, such that the treatment effect is a doubly robust representation of the difference in mean embeddings. Our test, DR-KTE-Deep, optimises the learned features so as to maximise an estimated power criterion on this treatment effect representation. To motivate our learning objective, we show that the power criterion characterises the asymptotic signal-to-noise ratio of the test under fixed alternatives, and we derive a uniform finite-sample bound for its regularised estimate. We show that, with the learned kernel fixed, a held-out test is asymptotically valid under the null given suitable nuisance-estimation rates. Empirically, DR-KTE-Deep achieves substantially higher power than fixed a
    
[^10]: Rényi正则化的基于流的贝叶斯校准变分推断

    R\`enyi-Regularized Flow-Based Variational Inference for Bayesian Calibration

    [https://arxiv.org/abs/2610.37169](https://arxiv.org/abs/2610.37169)

    本文提出一种基于Rényi散度正则化项的流式变分推断方法，解决了贝叶斯校准中标准变分推断无法捕捉被低概率区域分隔的多模态后验分布的问题，使近似后验能够覆盖先验支撑并表示多个后验模态。

    

    基于归一化流的变分推断为贝叶斯模型校准提供了灵活的后验近似，但当后验分布存在多个被低概率质量区域分隔的模态时，该方法表现不佳。我们在一个双参数孪生实验中研究了这种行为，并对变分目标进行正则化。在高斯观测误差和高斯先验下，相应的最小化目标为期望Tikhonov校准目标减去近似分布的微分熵。然而，期望Tikhonov项倾向于使分布在其中一个模态附近集中，而熵最大化并不能保证多模态性。我们引入了一个基于Rényi散度的正则化项，以鼓励近似后验覆盖先验的支撑集。在连通双峰的情形中，流近似能够表示两个模态；在模态良好分离的情形中，流与高斯混合频繁（此处摘要文本截断）

    arXiv:2610.37169v1 Announce Type: new  Abstract: Variational inference with normalizing flows provides a flexible posterior approximation for Bayesian model calibration, but struggles when the posterior has several modes separated by regions of low probability mass. We study this behavior in a two-parameter twin experiment and regularize the variational objective. Under Gaussian observation errors and a Gaussian prior, the corresponding minimization objective is the expected Tikhonov calibration objective minus the differential entropy of the approximating distribution. However, the expected Tikhonov term favors concentration near one of the modes, while entropy maximization does not guarantee multimodality. We introduce a regularization term based on R\'enyi divergence that encourages the approximate posterior to cover the prior support. In the connected bimodal case, a flow approximation represents both modes; in the case with well-separated modes, the flow and Gaussian mixture frequ
    
[^11]: 物理引导的流匹配：一种用于全球总电子含量预测的生成式人工智能框架

    Physics-Guided Flow Matching: A Generative AI Framework for Global Total Electron Content Forecasting

    [https://arxiv.org/abs/2610.37159](https://arxiv.org/abs/2610.37159)

    提出了一种物理引导的流匹配生成式AI框架，将偏微分方程描述的电离层动力学与数据驱动的统计模式自适应融合，实现次日全球总电子含量预测。

    

    准确预测全球总电子含量（TEC）对于空间天气监测以及卫星通信和导航系统的可靠运行至关重要。我们提出了一种物理引导的生成式人工智能（AI）框架，用于次日全球TEC预测。该框架建立在流匹配的基础上，流匹配学习将样本从初始分布逐步变换为具有目标数据统计特性的新样本。所提出的模型将偏微分方程（PDE）所描述的电离层动力学纳入这一变换过程，利用物理知识引导初始样本与目标样本之间的变化。该模型进一步将物理动力学与从数据中捕获的统计模式相结合来学习这一变换，并在每个位置自适应地调整二者的相对贡献。该模型使用一年的数据进行训练和评估。

    arXiv:2610.37159v1 Announce Type: cross  Abstract: Accurate forecasting of global total electron content (TEC) is important for space weather monitoring and the reliable operation of satellite communication and navigation systems. We propose a physics-guided generative artificial intelligence (AI) framework for day-ahead global TEC forecasting. The framework builds on flow matching, which learns to gradually transform samples from an initial distribution into new samples that share the statistical characteristics of the target data. The proposed model incorporates ionospheric dynamics described by a partial differential equation (PDE) into this transformation process, guiding the changes between initial and target samples using physical knowledge. The model further combines the physical dynamics with statistical patterns captured from data to learn this transformation, adaptively adjusting their relative contributions at each location. The model is trained and evaluated using one year 
    
[^12]: 你的大模型中转服务名副其实吗？一种用于LLM中转服务审计的低成本、高功效检验方法

    Is Your LLM Relay Serving What It Claims? A low-cost and high-power test for LLM Relay Auditing

    [https://arxiv.org/abs/2610.37086](https://arxiv.org/abs/2610.37086)

    提出分布式指纹审计方法DFA，通过反复查询简单的有限选项探针构建端点指纹并利用柯西组合检验聚合证据生成端点级p值，实现了低成本、鲁棒且统计可解释的LLM中转服务审计。

    

    中转和转售API已成为访问大型语言模型（LLM）的常见方式，然而用户无法直接验证自己是否确实收到了所请求并付费的模型。这种不透明性催生了LLM中转服务审计的需求，但在黑盒访问条件下，现有方法难以同时实现低成本、对良性部署变化的鲁棒性以及统计上可解释的决策。为解决这些局限，我们提出了分布式指纹审计，这是一种仅需要可疑端点和可信参考端点的文本响应即可工作的统计方法。DFA反复查询语义简单的有限选项探针，利用由此产生的类别型响应分布构建端点指纹，并逐探针比较可疑端点与参考端点的指纹，再使用柯西组合检验聚合证据，从而在预先指定的显著性水平下产生用于审计决策的近似端点级p值。

    arXiv:2610.37086v1 Announce Type: cross  Abstract: Relay and reseller APIs have become a common way to access large language models (LLMs), yet users cannot directly verify whether they receive the models they request and pay for. This opacity motivates LLM relay auditing, but existing methods struggle to combine low cost, robustness to benign deployment changes, and statistically interpretable decisions under black-box access. To address these limitations, we propose distributional fingerprint auditing (DFA), a statistical method that requires only textual responses from suspect and trusted reference endpoints. DFA repeatedly queries semantically simple finite-choice probes and uses the resulting categorical response distributions to construct endpoint fingerprints. It compares the suspect and reference fingerprints probe by probe and aggregates the resulting evidence using the Cauchy combination test, yielding an approximate endpoint-level $p$-value for audit decisions at a prespecif
    
[^13]: 边缘端的在线气体分类

    Online gas classification at the edge

    [https://arxiv.org/abs/2610.36999](https://arxiv.org/abs/2610.36999)

    受昆虫嗅觉系统启发，本文提出一种基于稀疏高维二进制表示的轻量神经网络架构，能够在边缘设备上从原始传感器瞬态响应中在线识别气体，性能媲美优化的非线性SVM，且适用于功耗和样本受限的真实场景。

    

    化学传感领域的实际应用，从便携式安全监测器到移动机器人，都需要轻量化、响应快速、并且能够基于有限标注数据进行重新校准的模型。当前的气体分类流程假设采样周期可控、特征提取离线进行，并且依赖于计算开销高昂的模型。昆虫仅用几千个神经元和亚毫瓦级的功耗，就能在开放环境中可靠地对气味进行分类。受昆虫嗅觉系统的启发，我们提出了一种神经网络架构，该架构使用稀疏的高维二进制表示对原始传感器阵列读数进行分类。我们在单种气体和二元混合物的随机序列上对该架构进行评估，动态引入新样本，并从传感器的瞬态响应中识别气体种类。在这一设定下，该架构在气体混合物分类上优于线性支持向量机（SVM），并与经过优化的非线性支持向量机性能相当，同……（原文截断）

    arXiv:2610.36999v1 Announce Type: cross  Abstract: Field applications in chemical sensing, from portable safety monitors to mobile robots, require models that are lightweight, fast to respond, and can be recalibrated from limited labelled data. Current gas classification pipelines assume controlled sampling cycles and offline feature extraction, and rely on computationally expensive models. With a few thousand neurons and sub-milliwatt power consumption, insects reliably classify odours in open environments. Inspired by the insect olfactory system, we present a neural network architecture that classifies raw sensor array readings using a sparse, high-dimensional binary representation. We evaluate it on random sequences of single gases and binary mixtures, introducing new samples dynamically and classifying gas identity from the transient sensor response. In this setting, the architecture out-performs a linear \gls{svm} on gas mixtures and matches an optimised non-linear \gls{svm}, whil
    
[^14]: ReLU网络的内在信息论分析

    Intrinsic Information-Theoretic Analysis of ReLU Nets

    [https://arxiv.org/abs/2610.36864](https://arxiv.org/abs/2610.36864)

    该论文利用ReLU网络将输入空间固有划分为凸线性区域这一特性，把互信息分解为可解释的几何项，并提出无需超参数、可估计的“路由信息”插件估计器，从而实现对深度网络的内在信息论分析。

    

    互信息（MI）是深度网络表示学习分析的基础，但在高维空间中，互信息估计器依赖于超参数，其偏差往往会盖过信号本身。我们证明，连续分段线性（CPWL）网络（包括ReLU网络）可以进行一种内在的信息论分析，从而避免这一缺陷：这类网络会将输入空间内在地划分为凸线性区域，从而为每个输入分配一个离散的区域标签 $\Pi$，与连续表示 $T$ 和标签 $Y$ 并存。我们引入了一种将互信息 $I(Y;T)$ 分解为可解释几何项的方法。我们证明了其中的主导项——*路由信息* $I(Y; \Pi)$——存在一个可估计的插件（plug-in）估计器，在充分采样的情况下，该估计器近似于联合信息 $I(Y; \Pi, T)$ 的总体下界。当划分达到饱和时，函数等价的商运算会将实现……（相同函数）的区域折叠合并（原文在此处截断）。

    arXiv:2610.36864v1 Announce Type: new  Abstract: Mutual information (MI) is fundamental to representation-learning analysis in deep networks, but in high dimensions MI estimators depend on hyperparameters whose bias often dominates the signal. We show that continuous piecewise-linear (CPWL) networks, including ReLU networks, admit an intrinsic information-theoretic analysis that avoids this pitfall: their inherent partition of the input space into convex linear regions assigns to every input a discrete region label $\Pi$ alongside the continuous representation $T$ and label $Y$. We introduce a decomposition of the mutual information $I(Y;T)$ into interpretable geometric terms. We show that the dominant term, the *routing information* $I(Y; \Pi)$, admits an estimable plug-in estimator, that in the well-sampled regime, approximates a population lower bound on the joint information $I(Y; \Pi, T)$. When the partition saturates, a functional-equivalence quotient collapses regions implementi
    
[^15]: 边界几何决定加权系数约束下浅层ReLU网络的最优逼近

    Boundary Geometry Determines Optimal Shallow ReLU Approximation under Weighted Coefficient Constraints

    [https://arxiv.org/abs/2610.36854](https://arxiv.org/abs/2610.36854)

    本文通过加权变差空间将区域几何纳入输出系数约束，证明边界几何决定了浅层ReLU网络的最优逼近率：最优指数由与小边界帽相关的单一几何增长指数刻画，并为任意凸体、平面凸体、$C^2$ 凸体等给出了精确的最优多项式逼近指数。

    

    我们研究边界几何如何影响浅层ReLU神经网络在凸体 $K\subset\mathbb R^d$ 上的 $L^q$ 逼近。目标类是加权变差空间的单位球，这类空间通过将区域几何纳入输出系数的约束中，能够对经典空间进行扩展。我们为任意凸体建立了逼近误差的上界和下界，并确定了所有平面凸体及其与多面体的乘积（包括立方体和圆柱体）的最优多项式逼近指数。对于 $C^2$ 凸体，我们在一个依赖于维数且可延伸到 $q>2$ 的范围内确定了最优指数；当所有主曲率均为正时，该结果可推广到任意有限的 $q\ge2$。对于这些目标类，最优逼近指数对区域的依赖性由一个与小边界帽相关的单一几何增长指数所刻画。此外，我们还确定了保持（性能所需的）锐利权重阈值。

    arXiv:2610.36854v1 Announce Type: new  Abstract: We study how boundary geometry affects $L^q$ approximation by shallow ReLU neural networks on convex bodies $K\subset\mathbb R^d$. The target classes are unit balls of weighted variation spaces that can enlarge classical spaces by incorporating domain geometry into constraints on output coefficients. We establish upper and lower bounds for arbitrary convex bodies and determine the optimal polynomial approximation exponents for all planar convex bodies and their products with polytopes, including cubes and circular cylinders. For $C^2$ convex bodies, we determine the optimal exponents in a dimension-dependent range of $q$ extending beyond $2$. When all principal curvatures are positive, the result extends to every finite $q\ge2$. For these classes, the dependence of the optimal exponent on the domain is captured by a single geometric growth exponent associated with small boundary caps. We also identify sharp weight thresholds for retainin
    
[^16]: 基于Boosting的保险理赔频率与严重度联合分布Copula回归

    Joint Distributional Copula Regression for Insurance Claim Frequency and Severity via Boosting

    [https://arxiv.org/abs/2610.36835](https://arxiv.org/abs/2610.36835)

    本文提出一种基于GAMLSS的模型化boosting方法来拟合结构化加性分布copula回归模型，实现保险理赔频率与严重度的联合建模，兼具加性预测器层面的可解释性、变量筛选能力与有竞争力的预测性能。

    

    在本文中，我们提出了一种基于模型的boosting方法，用于拟合具有二元响应的结构化加性分布copula回归模型，特别聚焦于理赔频率与理赔严重度的联合建模。该方法基于位置、尺度和形状广义可加模型（GAMLSS），它扩展了GAM的一般思想，允许响应分布的多个参数（而不仅仅是条件均值）依赖于协变量。因此，该模型在加性预测器的层面上具有内在的可解释性，相比黑盒替代方案提供了更高的透明度和可审计性。我们使用合成数据和真实数据对所提出的框架进行了评估，展示了其恢复潜在协变量效应和依赖结构、识别信息性变量，以及相对于基准模型取得有竞争力预测性能的能力。

    arXiv:2610.36835v1 Announce Type: new  Abstract: In this paper, we propose a model-based boosting approach for fitting structured additive distributional copula regression models with bivariate responses, focusing particularly on the joint modeling of claim frequency and claim severity. The proposed approach is based on generalized additive models for location, scale, and shape (GAMLSS), which extend the general idea of GAMs by allowing multiple parameters of a response distribution, rather than only its conditional mean, to depend on covariates. The model is therefore intrinsically interpretable at the level of its additive predictors, providing greater transparency and auditability than black-box alternatives. We evaluate the proposed framework using synthetic and real-world data. We demonstrate its ability to recover the underlying covariate effects and dependence structure, identify informative variables, and achieve competitive predictive performance relative to benchmark models. 
    
[^17]: 利用大语言模型知识改进分布外表格预测——一种交叉验证正则化方法

    LLM knowledge for improved Out-of-Distribution Tabular Prediction - A Cross-Validated Regularisation Approach

    [https://arxiv.org/abs/2610.36806](https://arxiv.org/abs/2610.36806)

    该研究提出将从大语言模型引导出的系数先验作为正则化回归的收缩目标，仅通过源域交叉验证选择提供者与先验结构，即可在五个表格数据集上将分布外预测的均方误差最高改善7.9%。

    

    当部署人群与训练所用人群不同时，预测模型的性能会下降，而训练数据无法说明哪些特征-响应关系能够迁移。专家知识引导过程缓慢且难以规模化。我们研究由大语言模型（LLM）引导出的系数先验，作为正则化回归的收缩目标（而非收缩至零），能否改进分布外（OOD）预测。我们在五个表格数据集上按真实协变量进行划分，并仅通过源域交叉验证来选择模型提供者与先验结构。结果显示，先验在五个数据集中的四个上有所帮助，OOD均方误差最高改善7.9%；在二十四个使用先验的结果中，有十个在常规置信水平下呈现出显著的均值改进，而单一部署情形下的置信区间则宽得多。在分类任务上，增益体现在概率校准精度而非排序能力上；使用损坏先验的对照实验将这一收益归因于特征特定的内容，而非……

    arXiv:2610.36806v1 Announce Type: cross  Abstract: Prediction models degrade when the deployment population differs from the one used to train them, and training data cannot say which feature-response relationships will transfer. Expert elicitation is slow and unscalable. We ask whether coefficient priors elicited from large language models (LLMs) improve out-of-distribution (OOD) prediction as the shrinkage target of a regularised regression instead of zero, across five tabular datasets split on real covariates, selecting providers and prior structures by source-domain cross-validation alone. The priors help on four of five datasets, by up to 7.9% OOD mean squared error, and ten of twenty-four prior-informed results resolve the mean improvement at conventional confidence; single-deployment intervals are much wider. On classification the gain is in probabilistic accuracy, not ranking, and control tests with corrupted priors attribute the benefit to feature-specific content rather than 
    
[^18]: 科学机器学习中时间插值的生成式精炼学习

    Generative Refinement Learning for Temporal Interpolation in Scientific Machine Learning

    [https://arxiv.org/abs/2610.36751](https://arxiv.org/abs/2610.36751)

    提出iFLEX模型，用条件扩散模型学习修正基于时间加权平均的简单插值所产生的残差，从而在低时间分辨率数据下更准确地重建中间科学场。

    

    计算代价高昂的仿真与真实世界观测所提供的数据，其时间分辨率往往低于下游任务的需求。在固定网格点上对观测值取平均可能引入误差，而仅凭运动估计也无法确定物理场数值如何随时间变化。本工作研究了通过学习来修正简单的数值插值，能否改进中间科学场的重建。作者提出的模型——插值流专家模型iFLEX（interpolative FLow EXpert）——首先在每个网格点上对两次观测取加权平均来估计缺失场，其中权重更偏向于距离所请求时刻更近的那次观测。在训练过程中，将这一估计值从观测到的中间场中减去得到残差，并由一个条件扩散模型学习生成该残差。两次观测连同目标时间、观测间隔以及可用的物理参数共同作为模型的条件输入。

    arXiv:2610.36751v1 Announce Type: cross  Abstract: Computationally expensive simulations and real-world observations often provide data at a lower temporal resolution than downstream tasks require. Averaging observations at fixed grid points can introduce errors, and motion estimates alone do not determine how the physical field values change. In this work, we study whether learning to correct a simple numerical interpolation improves the reconstruction of intermediate scientific fields. Our model, interpolative FLow EXpert (iFLEX), first estimates the missing field by taking a weighted average of the two observations at each grid point, giving more weight to the observation closer to the requested time. During training, subtracting this estimate from an observed intermediate field gives the residual that a conditional diffusion model learns to generate. Both observations condition the model, together with the target time, the interval between observations, and available physical param
    
[^19]: 保守对决老虎机：基线约束下的偏好学习

    Conservative Dueling Bandits: Preference Learning under Baseline Constraints

    [https://arxiv.org/abs/2610.36708](https://arxiv.org/abs/2610.36708)

    本文提出了保守上下文对决老虎机问题及MaxInP-BC算法，通过效用差距而非绝对奖励来认证安全性，克服了标准对决老虎机探索过于激进的问题，并首次证明直接迁移CLUCB的安全测试会导致线性遗憾。

    

    上下文对决老虎机通过成对比较进行学习，这与根据人类偏好微调语言模型所使用的反馈方式相同。标准算法的探索过于激进，可能会向用户展示两个都很糟糕的选项。我们研究了保守的上下文对决老虎机问题，该问题要求在每一轮中，所展示的较优项目的累计得分始终保持在可信基线得分的 (1-α) 比例之上。项目的得分定义为其击败当前最佳可用项目的概率，因此该约束仅取决于效用差异，无需绝对奖励值。这也改变了安全性的认证方式：我们证明，若原封不动地迁移保守老虎机算法 CLUCB 的安全测试，即使基线的效用完全已知，也可能产生线性遗憾。对于线性效用场景，我们开发了 MaxInP-BC 算法，该算法通过效用差距来认证得分，并在必须回退时将基线与一个信息量丰富的配对项目进行组合。

    arXiv:2610.36708v1 Announce Type: cross  Abstract: Contextual dueling bandits learn from pairwise comparisons, the same feedback used to fine-tune language models from human preferences. Standard algorithms explore aggressively and can show users two poor options. We study conservative contextual dueling bandits, which require that, at every round, the cumulative score of the better displayed item stay above a $(1-\alpha)$ fraction of a trusted baseline's. An item's score is its probability of beating the best available item, so the constraint depends only on utility differences and needs no absolute rewards. This also changes how safety can be certified: we show that transferring the safety test of the conservative bandit algorithm CLUCB unchanged can incur linear regret, even when the baseline's utilities are known exactly. For linear utilities, we develop MaxInP-BC, which certifies scores through utility gaps and, when it must fall back, pairs the baseline with an informative partne
    
[^20]: 基于偏微分方程的主动简单假设检验最优指数的界

    PDE-Based Bounds on Optimal Exponents in Active Simple Hypothesis Testing

    [https://arxiv.org/abs/2610.36657](https://arxiv.org/abs/2610.36657)

    本文通过结合Vijayan（2025）的偏微分方程刻画与Hamilton-Jacobi方程粘性解的比较原理，为主动简单假设检验的最优误差指数建立了由凸函数、集合和下哈密顿松弛参数化的一般上下界，从而推广了文献中的许多已知结果。

    

    我们提出了主动简单假设检验（Active Simple Hypothesis Testing, ASHT）问题中最优误差指数的一般上下界。ASHT在信息论、老虎机问题以及运筹学等多个领域都得到了广泛研究，但对这一最优指数的精确刻画始终悬而未决。本工作将Vijayan（2025）近期提出的关于最优指数的偏微分方程（PDE）刻画与Hamilton-Jacobi偏微分方程粘性解的比较原理相结合。由此得到的界以凸函数、集合和下哈密顿松弛作为参数，这种灵活性使得该结果能够推广文献中的许多已知结论。

    arXiv:2610.36657v1 Announce Type: cross  Abstract: We present general upper and lower bounds on the optimal error exponent for the Active Simple Hypothesis Testing (ASHT) problem. ASHT has been extensively studied across diverse fields including Information Theory, Bandits, and Operations Research, a precise characterization of this optimal exponent has remained elusive. This work combines a recent Partial Differential Equation (PDE) characterization of the optimal exponent by Vijayan (2025) with the comparison principle of viscosity solutions of Hamilton-Jacobi PDEs. This leads to bounds that are parametrized by convex functions, sets and lower Hamiltonian relaxations, a flexibility that generalizes many known results in the literature.
    
[^21]: 面向可解释非线性结构系统辨识的深度贝叶斯方法

    Deep Bayesian Approaches for Interpretable Nonlinear Structural System Identification

    [https://arxiv.org/abs/2610.36633](https://arxiv.org/abs/2610.36633)

    本教程系统介绍了将可学习表示嵌入概率状态空间框架的深度贝叶斯滤波方法，通过结合物理归纳偏置与后验推断，实现了非线性结构系统辨识中兼具物理可解释性与不确定性感知的数据驱动动力学建模。

    

    系统辨识，即从测量数据中推断动态系统数学表示的过程，是工程诊断、预测和决策制定的基础。贝叶斯滤波为概率状态估计提供了一个有原则的框架，但传统上依赖于预先设定的转移和观测模型。深度贝叶斯滤波的最新进展通过在概率状态空间公式中嵌入可学习的表示来解决这一局限，在保留物理结构和不确定性感知推断的同时，实现了对未知动力学的数据驱动辨识。本教程系统性地介绍了用于非线性结构系统辨识的深度贝叶斯滤波方法，重点探讨学习到的动力学、物理归纳偏置与后验推断之间的相互作用。深度马尔可夫模型（Deep Markov Model）作为数据驱动的参考模型，而物理引导的深度马尔可夫模型……

    arXiv:2610.36633v1 Announce Type: cross  Abstract: System identification, namely the inference of mathematical representations of dynamical systems from measured data, is fundamental to engineering diagnostics, prognostics, and decision-making. Bayesian filtering offers a principled framework for probabilistic state estimation but traditionally relies on prescribed transition and observation models. Recent advances in deep Bayesian filtering address this limitation by embedding learnable representations within probabilistic state-space formulations, enabling data-driven identification of unknown dynamics while retaining physical structure and uncertainty-aware inference.   This tutorial provides a systematic introduction to deep Bayesian filtering for nonlinear structural system identification, focusing on the interplay between learned dynamics, physical inductive biases, and posterior inference. The Deep Markov Model serves as a data-driven reference, while the Physics-guided Deep Mar
    
[^22]: 通过正交框架实现测度的Wasserstein逼近：统计与计算复杂度

    Wasserstein approximation of measures via orthogonal frames: statistical and computational complexities

    [https://arxiv.org/abs/2610.36581](https://arxiv.org/abs/2610.36581)

    该论文将无监督学习中的稀疏表示基选择任务转化为用正交框架轴上测度进行Wasserstein逼近的非光滑优化问题，并证明经验风险最小化和局部初始化的随机梯度方法均能以近参数速率 n^{-1/2} 恢复总体最优的正交框架。

    

    诸如主成分分析（PCA）之类的降维方法能够识别出捕捉数据大部分变异的子空间，但无法确定该子空间内的一个特殊基。选择一个使样本能够被稀疏表示的基，是无监督学习中反复出现的任务，因子分析、字典学习和子空间聚类等方法都在处理这一问题。我们将此任务表述为：用支撑在某个正交框架各坐标轴上的测度对数据分布进行Wasserstein逼近，这进而归结为正交群 $\mathrm{O}(d)$ 上的一个非光滑优化问题。我们证明，在温和的假设条件下，无论是经验风险最小化方法，还是局部初始化的随机梯度方法，都能以近参数速率 $\widetilde{O}_d(n^{-1/2})$（其中 $n$ 为样本量）恢复出总体最优的正交框架。主要困难在于：尽管总体目标是光滑的，经验目标却非光滑——这种不匹配阻碍了……（原文在此处截断）

    arXiv:2610.36581v1 Announce Type: cross  Abstract: Dimension-reduction methods such as PCA identify a subspace capturing most of the variation in the data, but not a distinguished basis within it. Selecting a basis in which the samples are sparsely represented is a recurring task in unsupervised learning, addressed by methods in factor analysis, dictionary learning, and subspace clustering. We pose this task as Wasserstein approximation of the data distribution by measures supported on the axes of an orthogonal frame, which, in turn, reduces to a nonsmooth optimization problem over the orthogonal group $\mathrm{O}(d)$. We show that, under mild assumptions, both empirical risk minimizers and a locally initialized stochastic gradient method recover the population-optimal frame at the near-parametric rate $\widetilde{O}_d(n^{-1/2})$, where $n$ is the sample size. The main difficulty is that, although the population objective is smooth, the empirical objective is not-this mismatch preclude
    
[^23]: 图分离准则的算法研究

    Algorithmics for Graphical Separation Criteria

    [https://arxiv.org/abs/2610.36502](https://arxiv.org/abs/2610.36502)

    该论文证明了多种图分离准则均可通过可达性的模块化变体在线性时间内完成检验（达到理论最优），并基于 CIfly 框架提供了相应的 R 和 Python 实现。

    

    统计图模型是随机向量各分量之间依赖关系的表示方式，它们通过图分离准则来编码这些依赖关系。我们考虑针对学术界已开发的各类图和分离准则，检验分离陈述这一算法问题。我们证明了所研究的所有准则都可以通过可达性的模块化变体来检验。对于除 σ-分离之外的所有分离准则，这导出了线性时间算法。由于加载图本身所需时间就是图中边数的线性函数，这实际上已是可达到的最优时间。利用用于部署可达性模块化变体的 CIfly 框架，我们还提供了实现所提算法的 R 和 Python 代码，并在仿真研究中评估了实际运行时间与扩展性能。

    arXiv:2610.36502v1 Announce Type: new  Abstract: Statistical graphical models are representations of the dependences between the components of a random vector. They encode these dependences via graphical separation criteria. We consider the algorithmic problem of checking separation statements for the various graph classes and separation criteria that have been developed by the community. We establish that all criteria we study can be checked by modular variations of reachability. For all separation criteria but $\sigma$-separation, this yields a linear-time algorithm. This is effectively the optimal attainable time since loading the graph itself is linear-time in the number of edges in the graph. Using the CIfly framework for deploying modular variations of reachability, we also provide R and Python code implementing our algorithms. We evaluate empirical runtime and scaling behavior in a simulation study.
    
[^24]: 二阶Sinkhorn几何与√n推断

    Second-order Sinkhorn Geometry and $\sqrt{n}$-Inference

    [https://arxiv.org/abs/2610.36488](https://arxiv.org/abs/2610.36488)

    本文将Sinkhorn散度的黎曼几何框架扩展至二阶，证明了Sinkhorn度量张量的Hadamard可微性并推导出Levi-Civita协变结构，从而支持√n速率的统计推断。

    

    Sinkhorn散度是熵正则最优传输的一种去偏形式，因其计算与统计上的易处理性而被广泛应用于众多科学领域。然而，与Wasserstein距离不同，Sinkhorn散度既不是度量，也不自然地具有形式化的黎曼结构，而这些性质对于利用插值、重心平均、平行移动或其他几何操作的最优传输方法的许多应用至关重要。近期的工作通过在概率测度空间上构造一个形式化的黎曼切空间和度量张量（该度量张量对Sinkhorn散度进行二阶逼近）来弥补这些不足。在本工作中，我们从两条主线出发对这一框架进行了扩展。对于一类具有有限光滑性的代价函数，我们证明了Sinkhorn度量张量的Hadamard可微性，并推导出Levi-Civita协变

    arXiv:2610.36488v1 Announce Type: new  Abstract: The Sinkhorn divergence, a debiased form of entropic optimal transport, has been widely used in various scientific disciplines due to its computational and statistical tractability. Unlike the Wasserstein distance, however, the Sinkhorn divergence is not a metric and it does not naturally admit a formal Riemannian structure. These properties are central to many applications of optimal transport methods that utilize interpolation, barycentric averaging, parallel transport or other geometric operations. Recent work addresses these shortcomings by constructing a formal Riemannian tangent space and a metric tensor on the space of probability measures that quadratically approximates the Sinkhorn divergence. In this work, we extend this framework with two major threads of contributions. For a suitable class of finite-smoothness cost functions, we prove Hadamard differentiability of the Sinkhorn metric tensor and we derive the Levi-Civita covar
    
[^25]: 置信度并非承诺：对类型化决策模型的参考审计

    Confidence Is Not a Promise: A Reference Audit of Typed Decision Models

    [https://arxiv.org/abs/2610.36311](https://arxiv.org/abs/2610.36311)

    该审计研究表明，类型化决策模型输出的概率只许可群体层面的统计保证而非单次决策的承诺——点估计阈值在190次评估中有147次失守，Beta下界规则虽无保证却比Learn then Test裁决更多项目，而跨语言与输入噪声会轻易打破在干净英语文本上校准的准确率承诺。

    

    类型化决策模型返回的是调用方所提供选项之上的概率分布，而软件往往将这个数字当作一种保证来对待。我们测量了这一概率究竟许可了什么，将开放权重的 Laya（五个检查点）与仅能通过 API 访问的 Jev 1.13 进行了比较。一个概率所许可的是关于决策群体层面的陈述，而非关于单个决策的陈述；一个在其独立裁决的案例上承诺某准确率的门槛，只有在其流量相似的数据上检验过下界之后，才能守住这一承诺。在 190 次可交换评估中，由点估计选出的阈值在 147 次中打破了承诺，Learn then Test 在全部评估中都守住了承诺，而 Beta 下界规则在 128 次评估中平均裁决的项目多于 Learn then Test，且没有任何一次更少，但它并不具备相应的保证。在英语上校准的 Jev，在 51 个语言区域中仅有 8 个守住了 95% 的承诺。普通的输入噪声即可打破基于干净文本校准的承诺，而在探索性运行中……（原文此处截断）

    arXiv:2610.36311v1 Announce Type: cross  Abstract: Typed decision models return a probability distribution over options supplied by the caller, and software tends to act on that number as if it were a guarantee. We measure what it licenses, comparing the open-weight Laya, in five checkpoints, with Jev 1.13, available only through an API. A probability licenses a statement about groups of decisions, not about one decision, and a gate that promises an accuracy among the cases it decides alone keeps that promise only with a lower bound checked on data like its traffic. Across 190 exchangeable evaluations, a threshold chosen by point estimate breaks the promise in 147, Learn then Test keeps it in all of them, and a Beta lower-bound rule decides more items on average than Learn then Test in 128 and fewer in none, without its guarantee. Calibrated in English, Jev keeps a promise of 95% in 8 of 51 locales. Ordinary typing noise breaks a promise calibrated on clean text, and, in exploratory ru
    
[^26]: 圆与球上的方向统计：看待数据的新“角度”

    Directional Statistics on Circles and Spheres: A New Angle on Data

    [https://arxiv.org/abs/2610.36227](https://arxiv.org/abs/2610.36227)

    本文指出角度数据天然位于圆和球而非直线上，直接算术平均可能导致错误结论（如朝北方位平均后指向南方），并通过风向与扩散MRI等实例介绍圆形/轴向均值、方向变异性和概率模型等方向统计工具，主张在选择统计方法前应先识别数据的几何结构与对称性。

    

    两个靠近北方的罗盘方位，其普通算术平均值却可能指向南方：问题在于角度存在于圆周上，而非直线上。本文利用真实的风向观测数据、扩散磁共振成像（diffusion-MRI）方向以及来自机器学习的几何实例，介绍了圆和球上的圆形与轴向平均、方向变异性以及概率模型。其核心指导原则是：在选择统计方法之前，应先识别观测数据所具有的几何结构与对称性。

    arXiv:2610.36227v1 Announce Type: cross  Abstract: Two compass bearings near north can have an ordinary arithmetic mean pointing south: the difficulty is that angles live on a circle, not a straight line. Using real wind observations, diffusion-MRI orientations, and geometric examples from machine learning, this article introduces circular and axial averages, directional variability, and probability models on circles and spheres. The guiding principle is to identify the geometry and symmetries of the observations before choosing a statistical method.
    
[^27]: 逼近族与去噪得分匹配的统计效率

    Approximation Families and the Statistical Efficiency of Denoising Score Matching

    [https://arxiv.org/abs/2610.36205](https://arxiv.org/abs/2610.36205)

    该论文证明了在 Ornstein-Uhlenbeck 流下对非高斯分布进行多项式得分匹配时，得分估计方差会沿多项式次数的某个子序列指数增长，且通过扩散半群平均损失（OISM）虽能降低每一固定次数的方差，却无法在足够小的时间消除这种指数不稳定性。

    

    我们研究了在由前向扩散过程的无穷小生成元的特征函数梯度所生成的有限维逼近空间上进行的得分匹配。我们比较了一维 Ornstein-Uhlenbeck 流下的多项式逼近与环面上热流下的三角多项式逼近。在 Ornstein-Uhlenbeck 设定中，我们证明：对于每一个与标准高斯分布一致可比的非高斯初始分布，渐近得分估计方差会沿多项式次数的某个子序列呈指数增长。将得分匹配损失在扩散半群下取平均——我们称之为算子信息得分匹配——可以在每个固定次数下降低方差，但在足够小的时间处并不能消除这种指数不稳定性。一个局部极小极大下界表明，当估计得分的多项式投影而非完整得分时，同样会出现这种对高次数的依赖。

    arXiv:2610.36205v1 Announce Type: new  Abstract: We study score matching over finite-dimensional approximation spaces generated by gradients of eigenfunctions of the infinitesimal generator of a forward diffusion. We compare polynomial approximation under one-dimensional Ornstein-Uhlenbeck flow with trigonometric-polynomial approximation under heat flow on the torus. In the Ornstein-Uhlenbeck setting, we show that for every non-Gaussian initial distribution uniformly comparable to the standard Gaussian, the asymptotic score-estimation variance grows exponentially along a subsequence of polynomial degrees. Averaging the score-matching loss under the diffusion semigroup, which we call operator-informed score matching (OISM), reduces the variance at each fixed degree but does not eliminate this exponential instability at sufficiently small times. A local minimax lower bound shows that the same high-degree dependence arises when estimating the polynomial projection of the score, rather tha
    
[^28]: 分布函数的随时有效置信带

    Anytime Valid Confidence Bands for Distribution Functions

    [https://arxiv.org/abs/2610.36149](https://arxiv.org/abs/2610.36149)

    该论文为数据流的分布函数构建了在所有时间和所有点上同时有效、且允许观测值任意依赖过去的置信带，并进一步将其推广为适用于任意索引化时间平均分布函数族的统一构造框架。

    

    非参数推断通常需要刻画未知函数（例如累积分布函数 CDF）在整个定义域上的表现，而不仅限于某个特定点。我们为数据流的分布函数构建了在所有时间和所有点上同时有效的置信带，即使每个观测值可能以任意方式依赖于过去。我们首先为时间平均条件 CDF 推导出这样的置信带，并展示其宽度如何由二叉搜索树的深度决定。随后我们证明这一结果是一个更一般框架的实例，借助该框架可以为任意带索引的时间平均分布函数族，构建在时间和任意索引集上一致有效的置信带。我们在重要性加权 CDF、右删失 CDF 以及多元 CDF 上演示了该框架的应用。

    arXiv:2610.36149v1 Announce Type: new  Abstract: Nonparametric inference often requires characterizing an unknown function, such as a cumulative distribution function (CDF), across its entire domain rather than a specific point. We develop confidence bands for distribution functions of a data stream that are valid simultaneously over all times and all points, even when each observation may depend arbitrarily on the past. We begin by deriving such bands for the time-averaged conditional CDF and show how their widths are determined by the depth of a binary search tree. We then show that this result is an instance of a framework through which one can construct valid confidence bands uniformly over time and over an arbitrary index set for any indexed family of time-averaged distribution functions. We illustrate the framework on importance-weighted and right-censored CDFs, as well as multivariate CDFs.
    
[^29]: 对称性破缺是有限容量的代价

    Symmetry Breaking Is the Price of Finite Capacity

    [https://arxiv.org/abs/2610.36138](https://arxiv.org/abs/2610.36138)

    这篇论文证明，当模型容量有限时，即使问题本身具有对称性（如旋转不变性），最优模型也会以精确可预测的量破坏对称性，其代价恰好等于最优单个方向与最优完整方向块之间的信号差，且这一结论同样适用于注意力头。

    

    如果旋转一个问题从不改变其答案，人们可能会期望其最优模型具有旋转不变性；我们证明，当容量受限时，这一期望会以可预测的、精确的量失效。对称性将模型的方向分组为若干个强度相等方向组成的块，对称模型必须完整地使用每个块，而当预算切割某个块时，每个最优模型都会破坏对称性。对于一个具有 k 单元双线性预测器的高斯问题，对称性的对数损失代价恰好为 $(U_k-I_k)/(2\sigma^2)$，即最优 k 个方向的信号减去最优完整块 的信号。由于注意力头通过秩至多等于其头维度的相似度来比较词元（无论其前置计算如何），同样的代价支配着以平方误差拟合对称相似度的注意力头，并且在三维方向上以此方式训练的注意力头恰好在与预测一致的头维度处破坏对称性，且呈现预测的旋转。

    arXiv:2610.36138v1 Announce Type: new  Abstract: If rotating a problem never changes its answer, one might expect its best model to be rotation-invariant; we show that this fails, predictably and by an exact amount, when capacity is limited. A symmetry groups a model's directions into blocks of equally strong directions, a symmetric model must use each block whole, and when the budget cuts a block every optimal model breaks the symmetry. For a Gaussian problem with a $k$-unit bilinear predictor, the log-loss price of symmetry is exactly $(U_k-I_k)/(2\sigma^2)$, the signal of the best $k$ directions minus that of the best whole blocks. Because an attention head compares tokens through a similarity of rank at most its head dimension, whatever computation precedes it, the same price governs heads fitted to a symmetric similarity by squared error, and heads trained this way on three-dimensional directions break symmetry at exactly the predicted head dimensions, with the predicted rotation 
    
[^30]: 半参数贝叶斯网络中的狄利克雷过程混合模型：一种新颖的条件密度估计方法

    Dirichlet Process Mixtures within Semiparametric Bayesian Networks: A Novel Conditional Density Estimation Approach

    [https://arxiv.org/abs/2610.36110](https://arxiv.org/abs/2610.36110)

    本文提出在半参数贝叶斯网络中用狄利克雷过程混合模型（DPM）替代核密度估计进行非参数条件概率分布建模，通过数据自适应聚类获得更紧凑的表示，在降低推理成本的同时提升了密度估计性能。

    

    半参数贝叶斯网络（SPBN）是一类灵活的概率图模型，通过低维条件概率分布（CPD）来表示连续变量的联合分布。现有的SPBN模型采用核密度估计器（KDE）进行非参数CPD建模，但其“每个观测对应一个混合分量”的表示方式会导致推理成本增加和密度估计误差，这可能成为应用的限制。在这项工作中，我们在SPBN框架内定义了一种基于贝叶斯非参数狄利克雷过程混合模型（DPM）的全新非参数组件CPD建模方法。与KDE不同，DPM能够对观测数据进行自适应聚类，在保留非参数密度估计灵活性的同时，获得更紧凑的表示。这使得推理成本更低，同时提升了密度估计性能。我们通过扩展实现了所提出的基于DPM的半参数贝叶斯网络（DPM-SPBN）。

    arXiv:2610.36110v1 Announce Type: cross  Abstract: Semiparametric Bayesian networks (SPBNs) are flexible probabilistic graphical models for representing joint distributions of continuous variables through lower-dimensional conditional probability distributions (CPDs). Existing SPBN models use kernel density estimators (KDEs) for non-parametric CPD modeling, but their one-component-per-observation mixture representation can lead to increased inference costs and density estimation errors, which can be limiting. In this work, we define a new CPD modeling approach for non-parametric components based on Bayesian non-parametric Dirichlet Process Mixtures (DPMs) within SPBNs. DPMs provide a data-adaptive clustering of observations, yielding a more compact representation while retaining the flexibility of non-parametric density estimation, unlike KDE. This results in lower inference costs while improving density estimation performance. We implement the proposed DPM-based SPBN (DPM-SPBN) by ext
    
[^31]: $\ell_p$ 球上在线线性优化的极小极大最优小损失界

    Minimax-Optimal Small-Loss Bounds for Online Linear Optimisation on $\ell_p$ Balls

    [https://arxiv.org/abs/2610.36097](https://arxiv.org/abs/2610.36097)

    本文针对 $\ell_p$ 球上的全信息在线线性优化问题，提出了一种基于有界径向正则化项的 FTRL 算法，在无需预知时间范围与小损失量的情况下，取得了与下界匹配的极小极大最优小损失遗憾保证，并通过切换扩展进一步保证遗憾不超过 FTL 遗憾的两倍。

    

    我们研究 $\mathbb{R}^d$ 中单位 $\ell_p$ 球上的全信息在线线性优化问题，其中 $2\le p<\infty$ 固定，且损失的对偶范数至多为一。设 $\Delta_T$ 表示最优固定动作相对每轮最小损失的累计超出量。一个正则化领头跟随（FTRL）算法在任意损失序列上、且无需知道 $T$ 或 $\Delta_T$ 的情况下，即可取得遗憾值 $O_p(1+\min\{\Delta_T^{1-1/p},\, d^{1/2-1/p}\sqrt{\Delta_T}\})$。相匹配的下界证明表明，对于每个固定的 $p$，该速率是极小极大最优的，且当 $p\to\infty$ 时，其常数 $(2+o(1))p/\log p$ 在对 $d$ 和 $T$ 一致的保证中是最优的。该算法使用一个有界径向正则化项，并至多切换一次到其二次增强形式。在某些 FTL（跟随领头者）遗憾为一的路径上，其遗憾会随 $T$ 增长。一个切换式的扩展版本保持了该尖锐常数，同时还保证 $\mathrm{Reg}_T\le 2\,\mathrm{Reg}_T^{\mathrm{FTL}}$。对于独立同分布（摘要在此处截断）

    arXiv:2610.36097v1 Announce Type: cross  Abstract: We study full-information online linear optimisation on the unit $\ell_p$ ball in $\mathbb{R}^d$, for fixed $2\le p<\infty$ and losses of dual norm at most one. Let $\Delta_T$ denote the cumulative loss of the best fixed action above each round's minimum. A follow-the-regularised-leader (FTRL) algorithm attains, on every loss sequence and without knowing $T$ or $\Delta_T$, regret $O_p(1+\min\{\Delta_T^{1-1/p},\, d^{1/2-1/p}\sqrt{\Delta_T}\})$. Matching lower bounds show that this rate is minimax for each fixed $p$, and its constant $(2+o(1))p/\log p$ is optimal as $p\to\infty$ among guarantees uniform in $d$ and $T$. The algorithm uses a bounded radial regulariser and switches at most once to its quadratic augmentation. On some paths where follow-the-leader (FTL) has regret one, its regret grows with $T$. A switching extension keeps the sharp constant and also guarantees $\mathrm{Reg}_T\le 2\,\mathrm{Reg}_T^{\mathrm{FTL}}$. For i.i.d. 
    
[^32]: 从低维代理模型中即时学习准平稳性诊断方法

    Learning on-the-fly quasi-stationarity diagnostics from low-dimensional surrogates

    [https://arxiv.org/abs/2610.36078](https://arxiv.org/abs/2610.36078)

    该论文提出了一种基于非线性概率流粗粒化与基于仿真的推断的准平稳性诊断新方法，通过由低维集体变量定义的代理过程即时学习去相关时间估计，以替代传统的MCMC诊断方法。

    

    加速动力学方法（如并行副本动力学）需要对去相关时间的估计，在此时间之后，被困于某一亚稳态的过程将按照准平稳分布（QSD）进行分布。在Binder、Lelièvre和Simpson提出的方法中，去相关时间是通过将过程非线性条件概率流的粒子近似与受MCMC启发的诊断量耦合来定义的。我们探索了一种替代MCMC诊断的方法，该方法基于非线性概率流的粗粒化与基于仿真的推断。我们首先为以存活为条件的被吸收马尔可夫跳过程建立了一个粗粒化框架，证明了随时间一致的非渐近估计，该估计利用由集体变量定义的代理过程的非线性流来逼近真实过程的非线性概率流。该框架不假设可逆性……（原文在此处截断）

    arXiv:2610.36078v1 Announce Type: new  Abstract: Accelerated dynamics methods, such as parallel replica dynamics, require estimates of the decorrelation time, after which a process trapped in one of its metastable states is distributed according to the quasi-stationary distribution (QSD).   In a method proposed by Binder, Leli\`evre and Simpson, decorrelation times are defined by coupling a particle approximation of the non-linear conditioned probability flow of the process with MCMC-inspired diagnostics.   We explore an alternative to MCMC diagnostics, based on coarse-graining of non-linear probability flows and simulation-based inference.   We first develop a coarse-graining framework for killed Markov jump processes conditioned on survival, proving uniform-in-time, non-asymptotic estimates approximating the non-linear probability flow of   the true process by the non-linear flow of a surrogate process, defined with a collective variable. The framework does not assume reversibility o
    
[^33]: 用于非光滑采样的锚定Stein变分梯度下降算法

    Anchored Stein Variational Gradient Descent Algorithm for Non-smooth Sampling

    [https://arxiv.org/abs/2610.36018](https://arxiv.org/abs/2610.36018)

    该论文提出锚定Stein变分梯度下降算法，通过引入更光滑的参考势能与锚定权重，无需计算目标势能梯度即可对非光滑目标分布进行采样，并在低正则性条件下给出了算法的非渐近收敛界。

    

    我们提出了一种Stein变分框架，用于在目标势能 $U$ 可能非光滑的情况下从目标密度 $p\propto e^{-U}$ 中进行采样。其核心思想是引入一个更光滑的参考势能 $U_0$ 以及锚定权重 $a=e^{U-U_0}$。由此得到的锚定Stein变分梯度下降（SVGD）算法无需计算 $U$ 的梯度即可逼近目标分布 $p\propto e^{-U}$，并将标准SVGD作为特例包含在内。特别地，我们允许目标势能 $U$ 是局部Lipschitz的，而参考势能 $U_0$ 是连续可微的，同时还满足其他适当假设。在这种低正则性条件下，我们建立了锚定扰动的Kullback-Leibler一阶变分公式，确定了再生核希尔伯特空间中相应的最速下降方向，并基于锚定核化形式证明了锚定SVGD算法的非渐近收敛界。

    arXiv:2610.36018v1 Announce Type: new  Abstract: We introduce a Stein variational framework for sampling from a target density $p\propto e^{-U}$ when the target potential $U$ is possibly non-smooth. The key idea is to introduce a smoother reference potential $U_0$ and the anchor weight $a=e^{U-U_0}$. The resulting anchored Stein variational gradient descent (SVGD) algorithm is able to target $p\propto e^{-U}$ without computing the gradient of $U$ and include standard SVGD as a special case.   In particular, we allow the target potential $U$ to be locally Lipschitz while the reference potential $U_0$ is continuously differentiable, along with other suitable assumptions. Under this low regularity regime, we establish a Kullback-Leibler first variation formula for anchored perturbations, identify the corresponding steepest descent direction in the reproducing kernel Hilbert space and prove a nonasymptotic convergence bound for the anchored SVGD algorithm in terms of an anchored kernelized
    
[^34]: 一种面向风电场布局样本高效贝叶斯优化的图信息克里金模型

    A Graph-informed Kriging model for sample-efficient Bayesian optimization of wind farm layouts

    [https://arxiv.org/abs/2610.35975](https://arxiv.org/abs/2610.35975)

    该论文提出一种图信息克里金框架，通过有向气动图将可复用的风机间气动交互信息嵌入代理模型的均值函数，从而显著提升高维风电场布局贝叶斯优化的样本效率与年发电量预测精度。

    

    随着风机数量的增加，风电场布局优化变得日益困难，尤其是在使用高保真气动模型的情况下。传统的基于克里金的贝叶斯优化在高维布局空间中往往存在样本效率低的问题，这是因为稀疏的训练数据所能提供的域覆盖十分有限。我们开发了一个图信息克里金框架，将可复用的气动交互信息嵌入到代理模型的均值函数中。风电场被表示为一个有向气动图，其中在低维相对坐标空间中定义的成对风机交互被组装成风电场尺度的均值。这种结构化均值能够捕捉年发电量（AEP）的主要变化，使高斯过程只需对残差结构和不确定性进行建模。数值实验表明，图信息克里金在AEP预测精度上一致优于普通（原文在此处截断）。

    arXiv:2610.35975v1 Announce Type: cross  Abstract: Wind farm layout optimization becomes increasingly difficult as turbine count grows, especially when high-fidelity aerodynamic models are used. Conventional Kriging-based Bayesian optimization often suffers from low sample efficiency in high-dimensional layout spaces because sparse training data provide limited domain coverage. We develop a Graph-informed Kriging framework that embeds reusable aerodynamic interaction information into the surrogate mean function. The wind farm is represented as a directed aerodynamic graph, where pairwise turbine interactions defined in a low-dimensional relative-coordinate space are assembled into a farm-scale mean. This structured mean captures the dominant variation in annual energy production (AEP), allowing the Gaussian process to model only the residual structure and uncertainty. Numerical experiments show that Graph-informed Kriging consistently improves AEP prediction accuracy over ordinary and 
    
[^35]: 基于潜变量ODE的李对称性与首次积分的数据驱动发现

    Data-driven Discovery of Lie Symmetries and First Integrals with Latent ODEs

    [https://arxiv.org/abs/2610.35967](https://arxiv.org/abs/2610.35967)

    提出LatentLieODE方法，结合自编码器与潜变量神经ODE，能直接从视频等时间序列观测数据中自动发现动力系统的线性李对称性与首次积分。

    

    我们提出了LatentLieODE，这是一种新颖的神经ODE方法，能够从时间序列观测数据（如视频输入）中发现所学动力系统的线性李对称性和首次积分。该方法首先利用自编码器学习观测系统的低维表示，然后将轨迹映射到去相关的潜坐标 $z \in \mathbb{R}^d$。方法通过可微的李括号残差 $\mathbf{J}_f(z)\mathbf{A}z - \mathbf{A} f(z)$ 联合拟合一组线性无穷小生成元 $\mathbf{A}_k \in \mathbb{R}^{d \times d}$，同时用多层感知机（MLP）优化潜变量ODE $f(z) = \mathrm{d}z/\mathrm{d}t$ 以重构观测空间中的时间序列。在找到合适的生成元和 $f$ 之后，我们再拟合第三个网络 $H_\phi$ 来生成系统的首次积分。在六个基准系统（谐振子、单摆与双摆、Lotka-Volterra捕食模型、像素摆、球面测地线）上，我们发现每个系统……

    arXiv:2610.35967v1 Announce Type: cross  Abstract: We present LatentLieODE, a novel neural ODE method to discover linear Lie symmetries and first integrals of the learned dynamics from timeseries observations such as video inputs. It starts with an autoencoder, which learns a low dimensional representation of the observed system and then maps trajectories to decorrelated latent coordinates $z \in \mathbb{R}^d$. A bank of linear infinitesimal generators $\mathbf{A}_k \in \mathbb{R}^{d \times d}$ is fitted jointly against a differentiable Lie-bracket residual $\mathbf{J}_f(z)\mathbf{A}z - \mathbf{A} f(z)$, while an MLP optimises the latent ODE $f(z) = \mathrm{d}z/\mathrm{d}t$ to reconstruct the observation-space timeseries. After suitable generators and $f$ are found, we fit a third network $H_\phi$ to generate the first integral of the system. Across six benchmark systems (harmonic oscillator, simple and twin pendulums, Lotka-Volterra, pixel pendulum, sphere geodesics) we find each syst
    
[^36]: 为哪个策略校准？干预情形下固定世界模型的保形不确定性

    Calibrated for Which Policy? Conformal Uncertainty for Fixed World Models Under Intervention

    [https://arxiv.org/abs/2610.35877](https://arxiv.org/abs/2610.35877)

    该论文提出了一种加权保形校准框架：当部署策略发生干预变化时，通过分解为状态占据与动作倾向项的密度比，将固定世界模型的边际覆盖率保证迁移到目标策略，并利用源-目标域分类器的对数损失遗憾给出密度比估计误差的显式上界。

    

    改变部署策略可能会使世界模型的不确定性估计失效，而物理转移机制和预测器本身保持不变。对于一步预测，如果源分布与目标分布共享条件机制 P(Y|X)=Q(Y|X)，那么干预可以通过结果前的测度变化 w(X)=dQ_X/dP_X 来识别。对于由策略控制的动力学，该密度比可分解为状态占据项与动作倾向项。加权保形校准可以将边际覆盖率迁移到目标分布；使用估计的比率 ŵ 时，有 P_Q{Y ∈ C_ŵ(X)} ≥ 1-α-TV(Q,Q̂) ≥ 1-α-E_P|w-ŵ|。源域与目标域二分类器的总体对数损失遗憾可以对同一比率误差进行约束。若 h 是其截断后的目标后验，ρ 是目标域先验，Δ(h) 是总体对数损失遗憾，则有 E_P|w-ŵ_h| ≤ (1/(ργ))√(Δ(h)/2)。一种精确的抵消改进了……

    arXiv:2610.35877v1 Announce Type: cross  Abstract: Changing a deployment policy can invalidate a world model's uncertainty while leaving the physical transition mechanism and predictor unchanged. For one-step prediction, if source and target laws share the conditional mechanism $P(Y|X)=Q(Y|X)$, then the intervention is identified by the pre-outcome change of measure $w(X)=dQ_X/dP_X$. For policy-controlled dynamics this ratio factors into state-occupancy and action-propensity terms. Weighted conformal calibration transports marginal coverage to the target law; with an estimated ratio $\hat w$, $P_Q\{Y \in C_{\hat w}(X)\} \ge 1-\alpha-TV(Q,\hat Q) \ge 1-\alpha-E_P|w-\hat w|$. Population log-loss regret for a source-versus-target domain classifier bounds the same ratio error. If $h$ is its clipped target posterior, $\rho$ is the target-domain prior, and $\Delta(h)$ is population log-loss regret, then $E_P|w-\hat w_h| \le (1/(\rho\gamma))\sqrt{\Delta(h)/2}$. An exact cancellation improves 
    
[^37]: 何时数据价值会退化为类别平衡？逐点数据估值的覆盖视角

    When Does Data Value Reduce to Class Balance? A Coverage View of Per-Point Data Valuation

    [https://arxiv.org/abs/2610.34801](https://arxiv.org/abs/2610.34801)

    论文的核心贡献是从“覆盖”视角证明：当预算紧张且数据模式冗余时，基于固定逐点分数的 top-k 数据选择会因重复覆盖已选模式而产生闭式超额风险，数据价值本质上取决于已选点集，应采用自适应的类别平衡式选择规则。

    

    逐点数据估值分数——如 Data Shapley、影响函数、杠杆值——通常被用于选择训练数据：保留得分最高的前 k 个点。我们认为，在数据选择真正起作用的情形下，基于固定的、对数据模式盲视的逐点分数进行 top-k 选择恰恰是错误的规则。原因在于，一个点的价值并非固定数值，而是取决于已经选择了哪些其他点——一个已选点的近似重复样本无论分数多高，几乎不增加任何价值——而逐点标量分数无法表达这种交互作用。该理论精确量化了这种失败：在一个等潜在模式模型中，固定 top-k 选择相对于自适应覆盖选择会产生具有闭式表达的超额风险，因为它反复将预算花费在已被覆盖的模式上。当预算紧张、学习器表示冗余以及漏掉某个模式代价高昂时，这一差距最大。实用的规则很简单：当标签可用时，使用平衡（方法）……

    arXiv:2610.34801v1 Announce Type: new  Abstract: Per-point data-valuation scores -- Data Shapley, influence, leverage -- are routinely used to select training data: keep the points with the top $k$ scores. We argue that top-$k$ by a fixed, mode-blind per-point score is the wrong selection rule precisely in the regime where data selection matters. The reason is that a point's value is not a fixed number but depends on what else has been selected -- a near-duplicate of an already-chosen point adds almost nothing, however high its score -- and a per-point scalar cannot express this interaction.   The theory quantifies this failure exactly: in an equal latent-mode model, fixed top-$k$ selection incurs a closed-form excess risk over adaptive coverage because it repeatedly spends budget on already-covered modes. The gap is largest when the budget is tight, the learner representation is redundant, and missing a mode is costly.   The practical rule is simple: use balance when the labels availa
    
[^38]: BRIDGE：一种阻止模型更新中虚假改进的接受门控，附回顾性临床评估

    BRIDGE: An Acceptance Gate to Block False Improvements in Model Updates, with a Retrospective Clinical Evaluation

    [https://arxiv.org/abs/2610.29515](https://arxiv.org/abs/2610.29515)

    BRIDGE 提出了一种模型无关的接受门控机制，通过交叉拟合与 bootstrap 置信下界检验来拦截模型更新中的虚假改进，将虚假改进率从约 20% 降至接近零，并经 12 项回顾性临床任务验证有效。

    

    模型会被反复更新，而在验证数据上看似更好但在新数据上并非如此的更新是一种“虚假改进”，可能误导临床决策。朴素的规则会接受每一个正向的验证增益，而当同一批数据既用于选择又用于评判更新时，结果会过于乐观。BRIDGE 是一个模型无关的接受门控，它基于预测结果和可用标签、以固定的设置来决定是否接受每次更新。数据的每一折评判在其余折上选出的更新（交叉拟合），以减少选择偏差；接受条件要求增益的 bootstrap 置信下界超过一个裕度。实验涵盖四种设置：(A) 11 个新数据集，(B) 合成任务，(C) 12 项回顾性临床任务，以及 (D) 39 个支持性数据集。虚假改进率从朴素方法的 19.1% 降至 BRIDGE 的 1.1% (A)，从 20.7% 降至 1.7% (B)，从 20.0% 降至 0% (C)，从 12.5% 降至 0.5% (D)；所有关于数据集或任务上降低幅度的 95% 置信区间均不包含零。

    arXiv:2610.29515v1 Announce Type: new  Abstract: Models are updated repeatedly, and an update that looks better on validation data but not on new data is a false improvement that can misguide clinical decisions. The naive rule accepts every positive validation gain and is optimistic when the same data choose and judge the update. BRIDGE is a model-agnostic acceptance gate that decides each update from predictions and the available labels, with fixed settings. Each fold of the data judges the update chosen on the other folds (cross-fitting) to reduce selection bias; acceptance requires the gain's lower bootstrap bound to exceed a margin. The four settings are (A) 11 new datasets, (B) synthetic tasks, (C) 12 retrospective clinical tasks, and (D) 39 supporting datasets. The false-improvement rate falls from 19.1% (naive) to 1.1% (BRIDGE) in (A), 20.7% to 1.7% (B), 20.0% to 0% (C), and 12.5% to 0.5% (D); every 95% confidence interval of the reduction over datasets or tasks excludes zero (n
    
[^39]: 希尔伯特尺度上后验扩散先验用于不适定信号恢复的稳定性

    Hilbert-Scale Stability of Posterior Diffusion Priors for Ill-Posed Signal Recovery

    [https://arxiv.org/abs/2610.27030](https://arxiv.org/abs/2610.27030)

    该论文在希尔伯特尺度框架下为线性逆问题的后验扩散采样建立了稳定性理论，在正向算子平滑条件、真值源条件与分数误差弱范数控制下，证明了学习到的扩散先验在不适定信号恢复中能够给出将噪声、分数近似与反向时间效应分离的非渐近恢复保证。

    

    当正向算子会把未知信号平滑掉时，间接且带噪声的观测难以被反演。在这种情况下，不稳定的模态恰恰是数据揭示信息最少的模态，因此任何重构方法的成功都取决于先验信息与算子几何结构在不同正则性尺度上的相互作用方式。扩散先验在这类问题中已被广泛使用；然而，现有理论仍无法确定：在不适定正向算子作用下，学习到的分数函数何时仍能保持有效性。本文针对希尔伯特尺度上的线性逆问题研究后验扩散，并在一个显式的分数放大条件下推导出稳定性理论。在正向映射满足平滑条件、真值满足源条件、以及分数误差具有弱范数控制的假设下，我们推导出了希尔伯特空间 \(H^s\) 中的非渐近恢复界，该界将观测噪声、分数近似误差与反向时间（原文摘要在此处截断）的影响分离开来。

    arXiv:2610.27030v1 Announce Type: new  Abstract: Indirect and noisy observations are difficult to invert when the forward operator smooths out the unknown signal. In this case, the unstable modes are precisely the modes that the data reveals least, thus the success of any reconstruction method depends on how prior information interacts with the geometry of the operator across regularity scales. Diffusion priors have become ubiquitous for such problems; however, the current theory still does not determine when a learned score remains informative under ill-posed forward operators. In this paper, we study posterior diffusion for linear inverse problems on Hilbert scales and derive a stability theory under an explicit score-amplification condition. Under a smoothing condition on the forward map, a source condition on the truth, and weak-norm control of score error, we derive non-asymptotic recovery bounds in \(\Hs^s\) that separate observation noise, score approximation, and reverse-time d
    
[^40]: 量子PAC学习中的样本复杂度优势需要对态制备酉算子的逆的访问

    Advantage of Sample Complexity in Quantum PAC Learning Requires Inverse Access to State-Preparation Unitaries

    [https://arxiv.org/abs/2609.38403](https://arxiv.org/abs/2609.38403)

    本文证明了在量子PAC学习中，查询复杂度对精度参数依赖的改进优势本质上必须依赖于对态制备酉算子逆的访问，而仅有前向访问的酉算子无法带来任何样本复杂度上的优势。

    

    量子计算能否减少学习一个预测规则所需从未知概率分布中采样的数据量，是量子机器学习中的一个基本问题。量子PAC学习通过量子数据来研究这一问题，其中量子数据是一种量子态，其振幅的平方编码了用于采样经典学习数据的未知分布。仅拥有此类量子数据的副本时，最优的最坏情况样本复杂度渐近地与经典PAC学习相匹配。相比之下，同时访问该量子态的态制备酉算子及其逆算子，可以在可实现学习中改善查询复杂度对精度参数的依赖关系。然而，仅前向访问（即只访问酉算子本身而不访问其逆）是否能够带来这样的改进，一直尚不明确。在本工作中，通过对所有兼容的态制备酉算子及其有限环境维度取最坏情况，我们证明了仅前向访问的最优查询复杂度无法获得该优势，即样本复杂度的提升本质上依赖于对态制备酉算子逆的访问。

    arXiv:2609.38403v1 Announce Type: cross  Abstract: Whether quantum computation can reduce the amount of data sampled from an unknown probability distribution required to learn a prediction rule is a fundamental question in quantum machine learning. Quantum PAC learning studies this question using quantum data as a quantum state whose squared amplitudes encode the unknown distribution from which classical learning data are sampled. With only copies of such quantum data, the optimal worst-case sample complexity asymptotically matches that of classical PAC learning. In contrast, access to both a state-preparation unitary for this state and its inverse can improve the query-complexity dependence on the accuracy parameter in realizable learning. However, it has remained unclear whether forward-only access allows such an improvement.   In this work, taking the worst case over compatible state-preparation unitaries and their finite ambient dimensions, we show that the optimal forward-only que
    
[^41]: 核范数正则化的贝叶斯矩阵补全

    Nuclear Norm-Regularized Bayesian Matrix Completion

    [https://arxiv.org/abs/2609.30078](https://arxiv.org/abs/2609.30078)

    本文提出了首个针对未知噪声方差下核范数正则化贝叶斯矩阵补全模型的采样器，并给出了复杂度为矩阵维度和目标精度倒数之多项式的显式非渐近保证。

    

    矩阵补全是指从带噪声观测的元素中估计矩阵中缺失元素的问题，它是推荐系统以及面板数据中反事实结果估计等众多问题的基础。许多算法采用正则化最小二乘法来解决该问题，通常以核范数作为正则化项，但这种方法只能产生点估计，缺乏内置的不确定性量化。贝叶斯公式是一种自然的替代方案：如果噪声方差已知，基于核范数的先验可以产生对数凹的后验分布。然而遗憾的是，在实践中噪声方差并非先验已知，因此要实现完全贝叶斯方法，必须对噪声方差施加先验。我们给出了该模型的第一个具有显式非渐近保证的采样器：其复杂度关于矩阵维度以及目标精度的倒数是多项式级的。我们的技术是将噪声精度的分布离散化到……

    arXiv:2609.30078v1 Announce Type: cross  Abstract: Matrix completion, the problem of estimating missing entries in a matrix from noisily observed ones, underlies a diverse array of problems such as recommender systems and counterfactual outcome estimation in panel data. Many algorithms address the problem using regularized least squares, often with the nuclear norm as a regularizer, but this method yields a point estimate with no built-in uncertainty quantification. A Bayesian formulation is a natural alternative, and if the noise variance is known, the nuclear norm-based prior yields a log-concave posterior. Unfortunately, in practice, the noise variance will not be known a priori, so for a fully Bayesian approach, a prior must be imposed on it. We give the first sampler for this model with an explicit non-asymptotic guarantee: polynomial in the matrix dimensions and in the reciprocal of the target accuracy. Our technique is to discretize the distribution of the noise precision onto a
    
[^42]: 大批量训练中的动量：Polyak 动量扩大临界批大小，Nesterov 动量提升数据效率

    Momentum in large-batch training: Polyak enlarges the critical batch size, Nesterov improves data efficiency

    [https://arxiv.org/abs/2609.02728](https://arxiv.org/abs/2609.02728)

    该论文在幂律核回归框架下证明，在单遍大批量训练中 Polyak 动量可将临界学习率随批大小线性放大（从而扩大临界批大小约 1/(1-ρ) 倍），而 Nesterov 动量的临界学习率以 $B^\beta$（β>1）的更快速度增长，从而显著提升数据效率，并给出了刻画完整风险动力学的标度律与三区制批大小相图。

    

    我们在单遍（one-pass）训练机制下研究动量何时以及如何改善大批量训练，并以幂律核回归作为一个易于解析的设定。我们首先通过临界学习率刻画风险的稳定性，临界学习率定义为保证训练稳定的最大学习率，并得到 $\eta_{\mathrm{SGD}}^{\mathrm{crit}}\eqsim 1$、$\eta_{\mathrm{Polyak}}^{\mathrm{crit}}\eqsim \min\{1,B(1-\rho)\}$ 以及 $\eta_{\mathrm{Nesterov}}^{\mathrm{crit}}\eqsim \min\{1,B^\beta(1-\rho)\}$，其中 $B$ 为批大小，$\rho$ 为动量因子，$\beta>1$ 为容量指数。在该允许区域内，我们推导出完整风险动力学的标度律，刻画了训练从早期瞬态阶段、经幂律衰减、直至噪声底限的完整演化过程。随后，在固定数据预算下，我们在允许的学习率与动量因子范围内最小化最后一步风险，得到了一个包含三个区制的批大小相图，揭示了（原文摘要在此处截断）

    arXiv:2609.02728v1 Announce Type: cross  Abstract: We study when and how momentum improves large-batch training in the one-pass regime, using power-law kernel regression as a tractable setting. We first characterize risk stability through the critical learning rate, defined as the largest learning rate for stable training, and obtain $\eta_{\mathrm{SGD}}^{\mathrm{crit}}\eqsim 1$, $\eta_{\mathrm{Polyak}}^{\mathrm{crit}}\eqsim \min\{1,B(1-\rho)\}$, and $\eta_{\mathrm{Nesterov}}^{\mathrm{crit}}\eqsim \min\{1,B^\beta(1-\rho)\}$, where $B$ is the batch size, $\rho$ is the momentum factor, and $\beta>1$ is the capacity exponent. Within this admissible region, we derive scaling laws for the full risk dynamics, capturing the progression from an early transient, through power-law decay, to a noise floor. We then minimize the final-step risk over the admissible learning rates and momentum factors under a fixed data budget, yielding a three-regime batch-size phase diagram that reveals how the rol
    
[^43]: 单义性的复仇：特化神经元提升多层感知器的数据效率

    Revenge of Monosemanticity: Specialized Neurons Improve Data Efficiency in MLPs

    [https://arxiv.org/abs/2608.24007](https://arxiv.org/abs/2608.24007)

    本文证明多层感知器通过发展特化神经元学习局部低维表示，而非全局低维表示，从而在聚类数据的回归问题中显著提升数据效率。

    

    arXiv:2608.24007v1 公告类型：新 摘要：理解神经网络如何学习和组织特征对于理解其行为至关重要。现有许多特征学习理论侧重于全局低维预测几何结构的出现。我们表明这种观点是不完整的。在具有聚类数据的回归问题中，我们证明多层感知器（MLPs）自然发展出单义性特化神经元：单个神经元与输入空间特定区域相关的特定预测特征强对齐。MLPs并非学习单一的全局低维表示，而是学习一组局部低维表示，这些表示共同跨越一个高维空间。这种特化在理论上赋予MLPs相对于基于全局低维表示的特征学习方法在数据效率上的优势。

    arXiv:2608.24007v1 Announce Type: new  Abstract: Understanding how neural networks learn and organize features is central to understanding their behavior. Much existing theory of feature learning has focused on the emergence of a global low-dimensional predictive geometry. We show that this picture is incomplete. In regression problems with clustered data, we demonstrate that multilayer perceptrons (MLPs) naturally develop monosemantic specialized neurons: individual neurons become strongly aligned with a specific predictive feature relevant to a particular region of the input space. Rather than learning a single global low-dimensional representation, MLPs learn a collection of local low-dimensional representations that can collectively span a high-dimensional space. This specialization provably gives MLPs a data-efficiency advantage over feature-learning methods based on a global low-dimensional representation.
    
[^44]: 复杂神经网络下降的Kähler景观与包括Calabi-Yau流形搜索与消灭的保证

    K\"ahler landscapes for complex neural network descents and guarantees including a search and destroy of the Calabi-Yau manifold

    [https://arxiv.org/abs/2608.19584](https://arxiv.org/abs/2608.19584)

    本文提出了一种在复参数神经网络中使用Kähler信息度量和自然梯度下降的新方法，并针对Calabi-Yau流形上的不良曲率条件提供了理论保证，通过几何定义的全局势实现了搜索与消灭策略。

    

    我们研究复参数化网络的景观。我们的方法受参数的信息论流形视角以及经典优化保证的启发，尽管涉及复杂几何变体，如通过Dolbeault渐近。下降路径在交叉熵下承认Kähler信息度量，通过Wirtinger Hessian作用于对数似然势。我们关注一种下降更新规则，采用自然梯度下降，通过逆度量缩放的微分损失，使下降路径保持在全纯切丛中。我们强调Calabi-Yau信息流形，这些流形通过不良曲率条件的景观提供了理论保证。在Calabi-Yau度量下，特别是在非紧致设置中，具有全局势而非调用Calabi猜想的拓扑要求，一个楔入无处消失的全纯...

    arXiv:2608.19584v1 Announce Type: new  Abstract: We study landscapes for complex-parameterized networks. Our approach is motivated with an information-theoretic manifold perspective of the parameter and via classical optimization guarantees although of complex geometric variety such as through Dolbeault asymptotics. The descent path admits a K\"ahler information metric under a cross-entropy via the Wirtinger Hessian on the log-likelihood potential. We restrict attention to a descent update rule with natural gradient descent via a differentiated loss scaled by the inverse metric, so the descent path remains in the holomorphic tangent bundle. We emphasize Calabi-Yau information manifolds which profane theoretical guarantees via an ill-curvature-conditioned landscape. Under a Calabi-Yau metric, specifically in a non-compact setting with a global potential so defined geometrically rather than invoking the topological requirements of the Calabi conjecture, a wedged nowhere-vanishing holomor
    
[^45]: ProxyGuard：针对共享目标的随机数据发布机制的直接可靠性推断

    ProxyGuard: Direct Reliability Inference for Randomized Data Release Mechanisms with Shared Targets

    [https://arxiv.org/abs/2608.18643](https://arxiv.org/abs/2608.18643)

    ProxyGuard提出了一种基于密封目标集和预设风险的新方法，能够在随机数据发布机制中同时控制无效发布被误认为有效和有效发布未被认证的错误，无需独立目标批次或发布级依赖假设。

    

    arXiv:2608.18643v1 公告类型：新 摘要：研究人员通常从多个发布、变换或种子中选择一个代理数据集。搜索可能使无效发布看起来足够好，而一个足够的发布并不能证明其生成器是可靠的。ProxyGuard通过预设的有界风险和密封目标集来控制这两种错误。命名发布模式校正多重性并认证特定发布。直接共享目标模式在共同目标上评估独立机制抽取，下界其有利评分率，并减去无效发布贡献的有利评分界限。以目标为条件，发布评分是独立的，从而在没有独立目标批次或发布级$p$值依赖性假设的情况下，提供有限样本机制可靠性保证。我们证明均值惩罚是尖锐的，并推导出具有加性目标浓度的平滑评分证书。在一项注册的三要求研究中，d

    arXiv:2608.18643v1 Announce Type: new  Abstract: Researchers often choose a proxy dataset from many releases, transformations, or seeds. Search can make an invalid release appear adequate, while one adequate release does not establish that its generator is reliable. ProxyGuard controls both errors using prespecified bounded risks and a sealed target set. Named-release mode corrects for multiplicity and certifies specific releases. Direct shared-target mode evaluates independent mechanism draws on a common target, lower-bounds their favorable-score rate, and subtracts a bound on favorable scores contributed by invalid releases. Conditional on the target, release scores are independent, yielding a finite-sample mechanism-reliability guarantee without independent target batches or assumptions on release-level $p$-value dependence. We show that the mean-only penalty is sharp and derive a smooth-score certificate with additive target concentration. In a registered three-requirement study, d
    
[^46]: c-校正流的计算与统计保证

    Computational and Statistical Guarantees of the c-Rectified flow

    [https://arxiv.org/abs/2608.02487](https://arxiv.org/abs/2608.02487)

    本文提出代价感知的c-校正流，克服了普通校正流可能无法收敛到最优传输耦合（如高斯情形下协方差矩阵不可交换时失效）的缺陷，并为迭代校正流首次提供了计算复杂度与统计误差方面的理论保证。

    

    近年来，校正流（rectified flow）已成为大规模图像生成的基础框架，支撑着FLUX.1和Stable Diffusion 3等最先进的系统。尽管其在实践中取得了显著成功，但迭代校正流的计算与统计保证在很大程度上仍未被探索。我们通过研究c-校正流来解决这一问题，这是一类代价感知的校正流，它在保持端点边缘分布的同时，将速度场投影到一个梯度类上。普通校正流可能无法恢复最优传输耦合：在一个高斯案例研究中，当且仅当源分布与目标分布的协方差矩阵可交换时，迭代才会收敛到最优耦合。更一般地，对于从高斯分布到对数凹分布的耦合，普通校正流从独立初始化出发一步之后即可达到不动点。相比之下，在适当的正则性与一致可积性假设下（摘要在此处截断）……

    arXiv:2608.02487v2 Announce Type: replace  Abstract: Recently, rectified flow has emerged as a fundamental framework for large-scale image generation, powering state-of-the-art systems such as FLUX.1 and Stable Diffusion 3. Despite its remarkable empirical success, the computational and statistical guarantees of iterative rectified flow have remained largely unexplored. We address this problem by studying c-rectified flow, a cost-aware class of rectified flow that projects velocity fields onto a gradient class while preserving endpoint marginals. The ordinary rectified flow can fail to recover the optimal transport coupling: in a Gaussian case study, the iteration converges to the optimal coupling if and only if the source and target covariance matrices commute. More generally, for a Gaussian to log-concave coupling, the ordinary rectified flow can reach a fixed point after one step from independent initialization. In contrast, under suitable regularity and uniform-integrability assump
    
[^47]: 通过半正定核密度估计实现分布忠实的缺失值插补

    Distributionally Faithful Imputation via Positive Semi-Definite Kernel Density Estimation

    [https://arxiv.org/abs/2607.07767](https://arxiv.org/abs/2607.07767)

    该论文提出 PSD-Impute 模型，将 MCAR 缺失值插补重新表述为基于半正定核密度的密度估计问题，构建了具有闭式边缘解的凸优化问题并具备统计一致性，可同时实现单次与多重插补。

    

    缺失值会损害统计推断和机器学习流程，然而大多数插补方法依赖于启发式方法或限制性的参数假设，忽略了数据的联合分布。我们将完全随机缺失（MCAR）下的插补问题重新表述为基于掩码观测的密度估计问题：估计一个分布，使其观测到的边缘分布与数据中的边缘分布完全吻合。利用半正定（PSD）核密度，我们得到了一个具有闭式边缘解的凸经验风险问题，可通过牛顿内点法求解。我们进一步引入了一个非凸扩展，通过交替最小化实现自适应超参数选择。由此产生的 PSD-Impute 模型能够从同一拟合密度中生成单次插补和多重插补，并具有统计一致性，对于非常规则的概率分布能够克服维度灾难。实验表明该模型在分布拟合方面具有竞争力。

    arXiv:2607.07767v2 Announce Type: replace  Abstract: Missing values undermine statistical inference and machine learning pipelines, yet most imputation methods rely on heuristics or restrictive parametric assumptions that ignore the joint data distribution. We recast imputation under missing completely at random (MCAR) as density estimation from masked observations: estimate a distribution whose observed marginals exactly match those in the data. Leveraging positive semidefinite (PSD) kernel densities we obtain a convex empirical risk problem with closed form marginals, solvable by a Newton interior point method. We further introduce a non convex extension that supports adaptive hyperparameter selection through alternating minimization. The resulting PSD-Impute model yields both single and multiple imputations from the same fitted density, enjoys statistical consistency, beating the curse of dimensionality for very regular probabilities. Experiments indicate competitive distributional 
    
[^48]: 上下文学习中思维链的渐近理论

    An Asymptotic Theory of Chain-of-Thought in In-Context Learning

    [https://arxiv.org/abs/2606.03217](https://arxiv.org/abs/2606.03217)

    该论文建立了上下文学习中思维链推理的首个渐近理论，精确推导出泛化误差随推理深度、预训练数据量和上下文长度变化的公式，并揭示了指数级与多项式级改进、饱和和过度思考之间的相变及最优推理深度的缩放规律。

    

    思维链推理已成为一种广泛使用的机制，通过在推理时生成中间推理步骤来激发大型语言模型的多步推理能力。然而，泛化性能随CoT深度的缩放行为仍然知之甚少。为解答这一问题，我们研究了线性回归中用于上下文权重预测的CoT理论可解模型，其中测试时推理被表示为权重参数估计的迭代精化过程。利用高维渐近条件下的随机矩阵理论工具，我们推导出了泛化误差关于推理深度、预训练数据量和上下文长度的精确公式。我们的分析揭示了一个清晰的相变，将指数级改进、多项式级改进、饱和以及过度思考区分开来，并刻画了最优推理深度的缩放方式。我们进一步表明，更深的推理在……（原文在此截断）

    arXiv:2606.03217v2 Announce Type: replace  Abstract: Chain-of-thought (CoT) reasoning has become a widely used mechanism for eliciting multi-step reasoning in large language models by generating intermediate reasoning steps at inference time. Yet the scaling behavior of generalization with CoT depth remains poorly understood. To address this question, we study a theoretically solvable model of CoT for in-context weight prediction in linear regression, where test-time reasoning is represented as an iterative refinement of the weight-parameter estimate. Using tools from random matrix theory under high-dimensional asymptotics, we derive an exact formula for the generalization error as a function of reasoning depth, pretraining data amount, and context length. Our analysis reveals a sharp phase transition separating exponential and polynomial improvement, saturation, and overthinking, and characterizes how the optimal reasoning depth scales. We further show that deeper reasoning is most ef
    
[^49]: CalArena：一个大规模的事后校准基准测试

    CalArena: A Large-Scale Post-Hoc Calibration Benchmark

    [https://arxiv.org/abs/2605.30188](https://arxiv.org/abs/2605.30188)

    该论文提出了CalArena，一个覆盖近2000个实验、融合经典模型与基础模型预测、并提供数十种校准方法统一可复现实现的大规模标准化事后校准基准，用以系统评估哪些校准方法在实践中真正有效。

    

    可靠的概率估计在许多机器学习应用中至关重要，然而现代分类器往往校准不佳。事后校准提供了一种简单且广泛使用的解决方案，但由于所提出的方法数量众多，加之评估规模小且不一致，使得人们难以确定哪些方法在实践中真正有效。我们引入了一个大规模、标准化的事后校准基准，涵盖表格数据和计算机视觉任务中近2000个实验，包括二分类、多分类和大规模分类设置。我们的基准汇集了来自多种经典模型、现代深度学习架构和基础模型的预测，并在统一的评估框架内提供了数十种校准方法的统一、可复现实现。我们认为，适当评分规则中的事后改进提供了一个原则……（原文在此处截断）

    arXiv:2605.30188v3 Announce Type: replace-cross  Abstract: Reliable probability estimates are critical in many machine learning applications, yet modern classifiers are often poorly calibrated. Post-hoc calibration provides a simple and widely used solution, but the large number of proposed methods, combined with small-scale and inconsistent evaluations, makes it difficult to determine which approaches are truly effective in practice. We introduce a large-scale, standardized benchmark for post-hoc calibration, covering nearly 2000 experiments across tabular and computer vision tasks, including binary, multiclass, and large-scale classification settings. Our benchmark aggregates predictions from a diverse set of classical models, modern deep learning architectures, and foundation models, and provides unified, reproducible implementations of dozens of calibration methods within a common evaluation framework. We argue that Post-Hoc Improvement (PHI) in proper scoring rules offers a princi
    
[^50]: 非参数条件矩双层优化的参数化速率

    Parametric Rates for Nonparametric Conditional-Moment Bilevel Optimization

    [https://arxiv.org/abs/2605.21341](https://arxiv.org/abs/2605.21341)

    该论文提出正交化超梯度估计器 OBiGrad，将超梯度偏差降低为滋扰误差的乘积，使条件矩双层优化达到参数化统计速率 O(T^{-1/2}+n^{-1/2})，并在固定迭代预算下显著降低了所需的样本量界限。

    

    我们研究带有条件矩的函数式双层优化问题，其中插件式超梯度会从非参数学习的条件期望中继承一阶误差。我们提出了 OBiGrad，一种正交化的超梯度估计器，其偏差是滋扰误差的乘积。在我们的回归与复杂度条件下，使用直接插件式超梯度的投影梯度下降法在最佳迭代点上产生期望平稳性误差 $O(T^{-1/2}+n^{-1/2}+n^{-\nu/b})$，而 OBiGrad 则达到 $O(T^{-1/2}+n^{-1/2}+n^{-2\nu/b})$，其中 $b=\nu(2+p)+\gamma$ 综合了逼近指数和复杂度指数。因此，当 $\nu(2-p)\ge\gamma$ 时，OBiGrad 达到参数化统计速率 $O(T^{-1/2}+n^{-1/2})$，在固定迭代预算下，相比直接插件式方法改进了充分样本量界限。我们在工具变量回归和拟合Q回归问题上展示了这些改进效果。

    arXiv:2605.21341v2 Announce Type: replace  Abstract: We study functional bilevel optimization with conditional moments, where plug-in hypergradients can inherit first-order errors from nonparametrically learned conditional expectations. We introduce OBiGrad, an orthogonalized hypergradient estimator whose bias is a product of nuisance errors. Under our regression and complexity conditions, projected gradient descent with the direct plug-in hypergradient yields expected stationarity error $O(T^{-1/2}+n^{-1/2}+n^{-\nu/b})$ for the best iterate, whereas OBiGrad achieves $O(T^{-1/2}+n^{-1/2}+n^{-2\nu/b})$, where $b=\nu(2+p)+\gamma$ combines the approximation and complexity exponents. Consequently, when $\nu(2-p)\ge\gamma$, OBiGrad attains the parametric statistical rate $O(T^{-1/2}+n^{-1/2})$, improving the sufficient sample-size bound relative to the direct plug-in bound for a fixed iteration budget. We illustrate these gains on instrumental-variable and fitted-Q-regression problems, exam
    
[^51]: 多变量保形预测：使用多个校准变量优化预测集

    Multi-Variable Conformal Prediction: Optimizing Prediction Sets using Multiple Calibration Variables

    [https://arxiv.org/abs/2605.12341](https://arxiv.org/abs/2605.12341)

    该论文提出多变量保形预测（MCP）框架，通过在单个优化问题中联合优化和校准多个预测集参数，消除了校准数据分割的需求且不牺牲有限样本覆盖保证，并基于情景理论提出了RemMCP和RelMCP两个可求解的变体。

    

    分裂保形预测能够构建具有有限样本覆盖保证的预测集，但其校准阶段仅限于调整单一阈值参数。因此，预测集的形状必须事先固定，通常需要在校准数据的预留部分上进行拟合。我们提出了多变量保形预测（MCP），这是一个在单个优化问题中联合优化和校准多个预测集参数的框架，在不牺牲有限样本覆盖保证的前提下，消除了校准数据分割的必要。基于情景理论——一个用于认证数据驱动决策的原则性框架——我们提出了两个易于求解的变体：RemMCP，基于带约束移除的约束优化，它推广了分裂保形预测；以及RelMCP，基于带约束松弛的惩罚优化，它还支持非凸问题，尽管计算成本更高。

    arXiv:2605.12341v2 Announce Type: replace  Abstract: Split conformal prediction constructs prediction sets with finite-sample coverage guarantees, but its calibration stage is limited to adjusting a single threshold parameter. Hence, the shape of the prediction sets must be fixed beforehand, typically by fitting it on a reserved part of the calibration data. We introduce multi-variable conformal prediction (MCP), a framework that jointly optimizes and calibrates multiple prediction-set parameters in a single optimization problem, eliminating calibration data split without sacrificing finite-sample coverage guarantees. Building on scenario theory, a principled framework for certifying data-driven decisions, we propose two tractable variants: RemMCP, grounded in constrained optimization with constraint removal, which generalizes split conformal prediction; and RelMCP, based on penalized optimization with constraint relaxation, which also supports non-convex problems, albeit at a higher c
    
[^52]: Muon的相变：Muon何时超越SignSGD

    Phases of Muon: When Muon Eclipses SignSGD

    [https://arxiv.org/abs/2605.09552](https://arxiv.org/abs/2605.09552)

    该论文推导了随机Muon在高维矩阵最小二乘问题上的显式确定性动力学，发现Muon对有效批次大小存在非线性敏感性——在精确白化且有效批次较小时，Muon难以分辨较小的特征模态而收敛变慢（动量与有限Newton–Schulz迭代可缓解此效应），而SignSGD并无此相变，从而阐明了Muon超越SignSGD的条件。

    

    Muon及相关的谱优化器（spectral optimizers）近年来作为可扩展的随机优化方法展现出强大的实证性能，常常优于Adam，然而其行为仍缺乏充分理解。我们在高维矩阵值最小二乘问题上分析了随机Muon——它对小批量梯度的指数移动平均进行正交化——并将其与随机SignSGD（一个可与Adam相媲美的简单基线）进行比较。我们推导出显式的确定性动力学，为研究学习行为提供了一个易于处理的框架。我们证明，Muon对有效批次大小 $B_{\rm eff}=B(1+\beta)/(1-\beta)$ 具有非线性敏感性：在精确白化且 $B_{\rm eff}$ 较小时，Muon无法分辨较小的特征模态，从而使收敛变慢；动量和有限的Newton–Schulz迭代可起到类似岭（ridge）的作用，减轻这一效应。这种行为与SignSGD形成对比，后者不存在这种转变，因而导致不同的……（原文摘要在此处截断）

    arXiv:2605.09552v2 Announce Type: replace-cross  Abstract: Muon and related spectral optimizers have recently demonstrated strong empirical performance as scalable stochastic methods, often outperforming Adam. Yet their behavior remains poorly understood. We analyze stochastic Muon, which orthogonalizes an exponential moving average of minibatch gradients, on a high-dimensional matrix-valued least squares problem, and we compare it with stochastic SignSGD, a simple baseline comparable to Adam. We derive explicit deterministic dynamics that provide a tractable framework for studying learning behavior. We show that Muon has non-linear sensitivity to the effective batch $B_{\rm eff}=B(1+\beta)/(1-\beta)$. With exact whitening and small $B_{\rm eff}$ Muon fails to resolve smaller eigenmodes, slowing down convergence. Momentum and finite Newton--Shulz iterations can act as a ridge which lessen this effect. This behavior is in contrast with SignSGD which has no transition, leading to differe
    
[^53]: 部分因果效应识别的最优实验设计

    Optimal Experiments for Partial Causal Effect Identification

    [https://arxiv.org/abs/2605.06993](https://arxiv.org/abs/2605.06993)

    该论文提出“最大效力问题”这一新框架，用于在成本约束下选择最优实验组合以最大程度收紧部分可识别因果查询的界限，证明了该问题的NP难性，给出了评估认识效力的通用方法，并引入基于因果图结构的剪枝准则（如线性时间的路径拦截规则）来控制超指数级搜索空间。

    

    因果查询往往无法仅从观测数据中完全识别，而能够收紧识别界限的实验通常代价高昂。我们研究在观察实验结果之前，选择一个成本受限的实验子集以最大程度收紧目标查询界限的问题。我们将其形式化为“最大效力问题”，其中认识效力衡量一个实验在最坏情况下所能保证的界限宽度缩减量，并通过从0-1背包问题的归约证明该问题是NP难的。基于Duarte等人（2023）的多项式规划框架，我们给出了在离散设置中评估认识效力的通用程序。为控制超指数级的搜索空间，我们引入了两个仅依赖于因果图和查询的图剪枝准则：一种新颖的路径拦截规则，它利用区域结构在线性时间内证明实验的零效力。

    arXiv:2605.06993v2 Announce Type: replace  Abstract: Causal queries are often only partially identifiable from observational data, and experiments that could tighten the resulting bounds are typically costly. We study the problem of selecting, prior to observing experimental outcomes, a cost-constrained subset of experiments that maximally tightens bounds on a target query. We formalize this as the max-potency problem, where epistemic potency measures the worst-case reduction in bound width guaranteed by an experiment, and show that this problem is NP-hard via a reduction from 0-1 knapsack. Building on the polynomial-programming framework of Duarte et al. (2023), we give a general procedure for evaluating epistemic potency in discrete settings. To control the super-exponential search space, we introduce two graphical pruning criteria that depend only on the causal graph and the query: a novel path-interception rule that exploits district structure to certify zero potency in linear time
    
[^54]: 关于KL正则化离线多臂老虎机的最优样本复杂度

    On the Optimal Sample Complexity of Offline Multi-Armed Bandits with KL Regularization

    [https://arxiv.org/abs/2605.02141](https://arxiv.org/abs/2605.02141)

    本文在多臂老虎机设置下对KL正则化离线学习算法KL-PCB给出了精确分析，证明其在大、小正则化条件下分别达到 $\tilde{O}(\eta SAC^{\pi^*}/\epsilon)$ 和 $\tilde{\Omega}(SAC^{\pi^*}/\epsilon^2)$ 的样本复杂度，从而基本刻画了KL正则化离线学习的最优样本复杂度。

    

    Kullback-Leibler（KL）正则化在离线决策中被广泛使用，并具有诸多优势，这促使了近期关于基于KL正则化性能指标的离线学习样本复杂度的研究。然而，KL正则化离线学习的精确样本复杂度在很大程度上仍未被完全刻画。本文在多臂老虎机（MABs）的设置下研究这一问题。我们对KL-PCB算法（Zhao et al., 2026）进行了精确的分析，证明其在大正则化 $\eta = \tilde{O}(\epsilon^{-1})$ 下达到 $\tilde{O}(\eta SAC^{\pi^*}/\epsilon)$ 的样本复杂度，而在小正则化 $\eta = \tilde{\Omega}(\epsilon^{-1})$ 下达到 $\tilde{\Omega}(SAC^{\pi^*}/\epsilon^2)$ 的样本复杂度，其中 $\eta$ 为正则化参数，$S$ 为上下文数量，$A$ 为臂的数量，$C^{\pi^*}$ 为最优策略 $\pi^*$ 处的策略覆盖系数。

    arXiv:2605.02141v2 Announce Type: replace-cross  Abstract: Kullback-Leibler (KL) regularization is widely used in offline decision-making and offers several benefits, motivating recent work on the sample complexity of offline learning with respect to KL-regularized performance metrics. Nevertheless, the exact sample complexity of KL-regularized offline learning remains largely from fully characterized. In this paper, we study this question in the setting of multi-armed bandits (MABs). We provide a sharp analysis of KL-PCB (Zhao et al., 2026), showing that it achieves a sample complexity of $\tilde{O}(\eta SAC^{\pi^*}/\epsilon)$ under large regularization $\eta = \tilde{O}(\epsilon^{-1})$, and a sample complexity of $\tilde{\Omega}(SAC^{\pi^*}/\epsilon^2)$ under small regularization $\eta = \tilde{\Omega}(\epsilon^{-1})$, where $\eta$ is the regularization parameter, $S$ is the number of contexts, $A$ is the number of arms, $C^{\pi^*}$ policy coverage coefficient at the optimal policy $
    
[^55]: 面向信息性缺失下可解释分类的专家引导类条件拟合优度分数

    Expert-Guided Class-Conditional Goodness-of-Fit Scores for Interpretable Classification with Informative Missingness

    [https://arxiv.org/abs/2604.14809](https://arxiv.org/abs/2604.14809)

    该论文提出了一个专家引导的类条件拟合优度分数框架，将专家先验知识编码为可解释的拟合优度特征，以解决信息性缺失下的可解释分类问题，并成功应用于《全面禁止核试验条约》的地震监测合规评估。

    

    我们研究了一个具有三个关键挑战的分类问题：普遍存在的信息性缺失、将部分先验专家知识融入学习过程，以及对可解释决策规则的需求。我们提出了一个框架，通过针对一个或多个类别的专家引导类条件模型来编码先验知识，并利用该模型构建一小组可解释的拟合优度特征。这些特征量化了观测数据与专家模型的符合程度，分离出数据不同方面的贡献，包括观测部分和缺失部分。这些特征与少量透明的辅助摘要一起被输入一个简单的判别分类器，从而得到一个易于检查和论证的决策规则。我们在用于评估《全面禁止核试验条约》遵守情况的地震监测背景下开发并应用了该框架。

    arXiv:2604.14809v2 Announce Type: replace  Abstract: We study a classification problem with three key challenges: pervasive informative missingness, the integration of partial prior expert knowledge into the learning process, and the need for interpretable decision rules. We propose a framework that encodes prior knowledge through an expert-guided class-conditional model for one or more classes, and use this model to construct a small set of interpretable goodness-of-fit features. The features quantify how well the observed data agree with the expert model, isolating the contributions of different aspects of the data, including both observed and missing components. These features are combined with a few transparent auxiliary summaries in a simple discriminative classifier, resulting in a decision rule that is easy to inspect and justify. We develop and apply the framework in the context of seismic monitoring used to assess compliance with the Comprehensive Nuclear-Test-Ban Treaty. We s
    
[^56]: 最小角回归的新解释与推断

    New explanations and inference for least angle regression

    [https://arxiv.org/abs/2602.02491](https://arxiv.org/abs/2602.02491)

    本文为最小角回归（LAR）提出了一个新的推断框架，将LAR解释为估计一个按总体相关参数递减排序的变量“路径”，从而揭示其新的数学性质并解决了诸如算法合适终止点等长期悬而未决的问题。

    

    Efron等人（2004）提出了最小角回归（LAR）作为线性预测的一种算法，旨在作为前向选择的替代方法，并与惩罚回归存在联系。然而，LAR在某种程度上一直是一个“黑箱”，其输出的一些基本行为特性尚未被很好地理解，包括算法的合适终止点。我们为LAR的推断提供了一个新颖的框架，这也使得LAR能够从新的视角被理解，并由此发展出若干新的数学性质。在数据层面，LAR算法可以被视为估计一个总体层面的对应“路径”，该路径按照总体“相关”参数递减的顺序，沿着回归变量来组织响应均值；这些参数被证明对于解释变量贡献具有有意义的解释，其中零相关表示不重要的变量。

    arXiv:2602.02491v2 Announce Type: replace-cross  Abstract: Efron et al. (2004) introduced least angle regression (LAR) as an algorithm for linear predictions, intended as an alternative to forward selection with connections to penalized regression. However, LAR has remained somewhat of a "black box," where some basic behavioral properties of LAR output are not well understood, including an appropriate termination point for the algorithm. We provide a novel framework for inference with LAR, which also allows LAR to be understood from new perspectives with several newly developed mathematical properties. The LAR algorithm at a data level can viewed as estimating a population counterpart "path" that organizes a response mean along regressor variables which are ordered according to a decreasing series of population "correlation" parameters; such parameters are shown to have meaningful interpretations for explaining variable contributions whereby zero correlations denote unimportant variabl
    
[^57]: 通过群体鲁棒马尔可夫决策过程从多源环境中学习序贯决策

    Learning Sequential Decisions from Multiple Sources via Group-Robust Markov Decision Processes

    [https://arxiv.org/abs/2602.01825](https://arxiv.org/abs/2602.01825)

    该论文提出基于共享线性MDP表示的d-矩形分布鲁棒决策框架及GR-PEVI离线算法，通过按特征最坏情况聚合与数据依赖悲观惩罚，在鲁棒部分覆盖条件下实现了跨源异质多环境序贯决策学习的有限样本次优性保证。

    

    从多个源环境中学习一个共同策略需要考虑跨源异质性，而离线设置还带来了数据覆盖有限的额外挑战。我们将该问题表述为在共享线性MDP表示下的分布鲁棒马尔可夫决策过程，该表示包含源特定的转移和奖励组件。不确定性集通过这些组件的按特征混合来构建，产生了一个d-矩形结构，该结构既能刻画跨源异质性，又支持鲁棒Bellman递归。基于此表述，我们开发了群体鲁棒悲观值迭代（GR-PEVI），这是一种离线策略学习算法，它将按特征的最坏情况聚合与数据依赖的悲观惩罚相结合。在鲁棒部分覆盖条件下，我们为GR-PEVI建立了有限样本次优性保证，并将该分析扩展至聚类的源环境情形（摘要原文在此处被截断）。

    arXiv:2602.01825v2 Announce Type: replace-cross  Abstract: Learning a common policy from multiple source environments requires accounting for cross-source heterogeneity, and the offline setting introduces an additional challenge of limited data coverage. We formulate this problem as a distributionally robust Markov decision process under a shared linear MDP representation with source-specific transition and reward components. The uncertainty set is constructed through feature-wise mixtures of these components, yielding a $d$-rectangular structure that captures cross-source heterogeneity and admits a robust Bellman recursion. Building on this formulation, we develop Group-Robust Pessimistic Value Iteration (GR-PEVI), an offline policy-learning algorithm that combines feature-wise worst-case aggregation with a data-dependent pessimism penalty. Under a robust partial coverage condition, we establish finite-sample suboptimality guarantees for GR-PEVI and extend the analysis to clustered so
    
[^58]: OneFlowSBI：一个模型应对多种查询的基于仿真的推理

    OneFlowSBI: One Model, Many Queries for Simulation-Based Inference

    [https://arxiv.org/abs/2601.22951](https://arxiv.org/abs/2601.22951)

    提出OneFlowSBI，一种掩码流匹配框架，通过在参数与观测的联合分布上训练单一网络，并利用三值掩码作为输入而非token化方式，实现任意条件或边缘分布的高效采样，在取得更低后验误差的同时大幅提升采样速度并具备大规模变量扩展能力。

    

    基于仿真的推理（SBI）方法通常只摊销单一查询（通常是后验分布），因此任何进一步的条件分布或边缘分布都需要单独训练一个模型。现有的通用模型通过在同一个Transformer中对多种查询进行摊销来实现通用性，其中每个变量对应一个token，导致其注意力计算成本随变量数量呈二次方增长。我们提出了OneFlowSBI，这是一个掩码流匹配框架，它在参数与观测的联合分布上训练单个网络。通过一个三值掩码将每个坐标标记为已观测、待生成或缺失，只需改变掩码即可采样任意条件分布或边缘分布。掩码作为输入而非token化的方式，使主干网络能够遵循数据自身的结构并扩展到大量变量。在高维图像去噪和浅水推断任务上，OneFlowSBI相比基线方法取得了更低的后验误差，采样速度远快于基于Transformer的通用模型，并能处理缺失区域和部分观测的情况。

    arXiv:2601.22951v2 Announce Type: replace  Abstract: Simulation-based inference (SBI) methods typically amortize a single query, usually the posterior, so each further conditional or marginal needs its own model. Existing generalists amortize many queries in one transformer with one token per variable, so their attention cost grows quadratically with the number of variables. We introduce OneFlowSBI, a masked flow matching framework that trains one network on the joint distribution of parameters and observations. A ternary mask marks each coordinate as observed, generated or absent, so any conditional or marginal is sampled by changing the mask. The mask is an input, rather than a tokenization, so the backbone can follow the data's structure and scale to many variables. On high-dimensional image denoising and shallow water inference, OneFlowSBI achieves lower posterior error than the baselines, samples far faster than a transformer-based generalist, and handles missing regions and parti
    
[^59]: 面向结构化分布增强学习的潜空间非线性去噪得分匹配

    Latent Nonlinear Denoising Score Matching for Enhanced Learning of Structured Distributions

    [https://arxiv.org/abs/2512.06615](https://arxiv.org/abs/2512.06615)

    本文提出潜空间非线性去噪得分匹配（LNDSM），通过将结构感知的非线性潜空间扩散与VAE训练框架结合，并指定能反映数据定性特征的参考分布（如高斯混合模型），从而将结构信息直接融入加噪动力学，增强对结构化分布的学习。

    

    基于得分的生成模型通常采用带高斯参考分布的线性前向扩散，这限制了它们将结构信息直接融入加噪动力学的能力。我们提出了潜空间非线性去噪得分匹配（LNDSM），它将非线性的、结构感知的潜空间扩散与变分自编码器（VAE）训练框架相结合。在我们的构造中，前向扩散被设计为具有一个指定的参考分布（例如高斯混合模型），该参考分布能够反映潜空间数据分布的定性特征。我们利用 Euler-Maruyama 离散化以及由此得到的高斯条件转移，推导出 VAE 目标中交叉熵上界的一个可计算近似。为了稳定训练，我们去除了两个零均值控制变量，其方差在小时间步长下会变得很大。所得到的目标函数能够实现编码器（及其相关模块）的联合训练。

    arXiv:2512.06615v2 Announce Type: replace  Abstract: Score-based generative models commonly employ linear forward diffusions with Gaussian reference distributions, limiting their ability to directly incorporate structural information into the noising dynamics. We introduce latent nonlinear denoising score matching (LNDSM), which combines nonlinear, structure-informed latent diffusions with the variational autoencoder (VAE) training framework. In our construction, the forward diffusion is designed to admit a prescribed reference distribution, such as a Gaussian mixture model, that reflects qualitative features of the latent data distribution. We derive a computable approximation to a cross-entropy upper bound in the VAE objective using Euler-Maruyama discretization and the resulting Gaussian conditional transitions. To stabilize training, we remove two zero-mean control variates whose variances become large for small time steps. The resulting objective enables joint training of the enco
    
[^60]: 基于Musielak的Orlicz-Sobolev：一种面向图上IPM的高效正则化方法

    Orlicz-Sobolev with Musielak: An Efficient Regularization Approach for Graph-based IPM

    [https://arxiv.org/abs/2510.25591](https://arxiv.org/abs/2510.25591)

    该论文提出了一种基于Orlicz几何的广义Sobolev IPM方法，通过引入一类特定凸函数突破了传统Sobolev IPM受限于L^p几何结构的局限，为基于图的积分概率度量提供了高效的正则化途径。

    

    我们研究了支撑在图度量空间上的概率测度的Sobolev IPM（积分概率度量）问题，其中评判函数（critic function）被约束在由Sobolev范数定义的单位球内。Sobolev IPM在其定义中本质上与 $L^p$ 几何结构耦合，这限制了它融入 $L^p$ 范式之外其他先验几何结构的能力。相反，经典的最优传输是灵活的，只需简单地改变其基础代价（ground cost）便能适应各种几何结构。一个重要的例子是Orlicz-Wasserstein（OW），它利用Orlicz几何结构对标准 $p$ 阶Wasserstein中的 $L^p$ 进行了推广，并在推动机器学习方法发展方面发挥了重要作用。受OW最新进展的启发，在本工作中，我们利用一类特定的凸函数来构造Orlicz几何，以缓解Sobolev IPM的上述局限性，并提出了广义Sobolev IPM（

    arXiv:2510.25591v2 Announce Type: replace-cross  Abstract: We study the Sobolev IPM problem for probability measures supported on a graph metric space, where critic function is constrained to lie within the unit ball defined by Sobolev norm. Sobolev IPM is intrinsically coupled with $L^p$ geometric structure within its definition, limiting its ability to incorporate other prior geometry beyond the $L^p$ paradigm. Conversely, the classic optimal transport is flexible, easy to adapt to various geometric structures by simply changing its ground cost. An important example is Orlicz-Wasserstein (OW) which utilizes \emph{Orlicz} geometric structure to generalize the $L^p$ within the standard $p$-order Wasserstein, and remarkably play a vital role to advance machine learning methodologies. Inspired by recent advantages of OW, in this work, we leverage a specific class of convex functions for Orlicz geometry to mitigate such limitation for Sobolev IPM, and propose the generalized Sobolev IPM (
    
[^61]: 论概念激活向量的变异性

    On The Variability of Concept Activation Vectors

    [https://arxiv.org/abs/2509.24058](https://arxiv.org/abs/2509.24058)

    本文对概念激活向量（CAV）的构建过程进行了细粒度的理论分析，量化了其因训练集随机采样而产生的变异性，并在四种不同模态的真实数据集上通过实验验证了结果的普适性。

    

    人工智能中最紧迫的挑战之一是让模型对用户更加透明。近年来，可解释人工智能领域提出了众多方法来应对这一挑战。一个有前景的方向是使用基于概念的解释，即采用高层概念而非简单的特征重要性分数。在这类方法中，概念激活向量（CAVs，Kim et al., 2018）脱颖而出，成为其中的代表性方法之一。CAV的一个有趣之处在于，其计算需要从训练集中随机采样样本。因此，实际得到的向量可能因这种随机采样的不确定性而有所不同。在本文中，我们对CAV的构建提出了细粒度的理论分析，以量化其变异性。我们的结果在四种不同模态的多个真实数据集上的实验中得到验证，指向一个普适性结论：其方差……（摘要原文在此处截断）

    arXiv:2509.24058v2 Announce Type: replace-cross  Abstract: One of the most pressing challenges in artificial intelligence is to make models more transparent to their users. Recently, explainable artificial intelligence has come up with numerous methods to tackle this challenge. A promising avenue is to use concept-based explanations, that is, high-level concepts instead of plain feature importance scores. Among this class of methods, Concept Activation Vectors (CAVs, Kim et al., 2018) stand out as one of the main protagonists. One interesting aspect of CAVs is that their computation requires sampling random examples from the train set. Therefore, the actual vectors obtained may vary depending on the randomness of this sampling. In this paper, we propose a fine-grained theoretical analysis of CAV construction in order to quantify their variability. Our results, confirmed by experiments on several real-life datasets of four different modalities, point to an universal result: the variance
    
[^62]: 微调何时具有可扩展性？基于噪声数据的可认证恢复与结构化任务偏移下的线性成本微调

    When is Fine-Tuning Scalable? Certified Recovery from Noisy Data and Linear-Cost Fine-Tuning on Structured Task Shifts

    [https://arxiv.org/abs/2509.00924](https://arxiv.org/abs/2509.00924)

    本文提出了一个模型特定的两阶段认证程序，证明单头Transformer在结构化任务偏移下仅需线性成本的微调，即可从噪声数据中以极小极大最优速率可认证地恢复源任务并实现分布外适应。

    

    Transformer模型通常只需少量微调即可在相关任务之间进行迁移；我们在噪声非参数回归设定中研究这一问题，考虑表示源任务与偏移任务的连续映射 $f,g:[0,1]^d\to\mathbb{R}^D$，并使用具有基于 $\ell^{\infty}$ 对齐分数 $\alpha$ 的单查询、单头Transformer $\mathcal{T}(\cdot)=V\operatorname{Softmax}(-\alpha(\cdot,K))$。我们并不假设基于梯度的训练能够达到经验风险最小化器，而是给出了一个模型特定的两阶段过程，用于**认证**源任务恢复与分布外（OOD）适应能力。在高概率下，所得到的注意力头能够从带噪观测中以任意精度 $\varepsilon>0$、在每个固定的 $L^p([0,1]^d)$ 范数（$1\le p<\infty$）下恢复 $f$。对于Lipschitz目标，其收敛速率在对数因子意义下达到极小极大最优，且无需假设可以访问渐近经验风险最小化器的预言机。若 $D=1$ 且 $f$ 为Pfaffian函数，则……（摘要在此处截断）

    arXiv:2509.00924v2 Announce Type: replace  Abstract: Transformers often transfer across related tasks with little fine-tuning; we study this question in noisy nonparametric regression for continuous maps $f,g:[0,1]^d\to\mathbb{R}^D$, representing the source and shifted tasks, using a single-query, single-head transformer $\mathcal{T}(\cdot)=V\operatorname{Softmax}(-\alpha(\cdot,K))$ with $\ell^{\infty}$-based alignment scores $\alpha$. Rather than assume that gradient-based training reaches an empirical risk minimizer, we give a model-specific two-stage procedure \textbf{certifying} source-task recovery and out-of-distribution (OOD) adaptation. With high probability, the resulting attention head recovers $f$ to arbitrary accuracy $\varepsilon>0$ in every fixed $L^p([0,1]^d)$ norm, $1\le p<\infty$, from noisy observations. For Lipschitz targets, its rate is minimax-optimal up to logarithmic factors without assuming oracle access to an asymptotic ERM. If $D=1$ and $f$ is Pfaffian than th
    
[^63]: 基于密度比加性树模型的双样本比较

    Two-sample comparison through additive tree models for density ratios

    [https://arxiv.org/abs/2508.03059](https://arxiv.org/abs/2508.03059)

    本文提出基于新损失函数“平衡损失”的密度比加性树模型，使树模型既能利用监督学习算法高效训练，又能通过广义贝叶斯推断实现不确定性量化，从而完成双样本比较任务。

    

    两个密度之比直接刻画了它们之间的差异。我们考虑双样本比较问题，即在给定来自两个分布的独立同分布观测数据的条件下估计这一比值。为此，我们提出了用于密度比估计的加性树模型，并借助一种新的损失函数——平衡损失——设计了高效的算法。该损失函数使得基于树的模型能够使用最初为监督学习设计的多种算法进行训练，例如前向分步优化和梯度提升。此外，平衡损失类似于指数族核函数，并可作为具有共轭先验的伪似然函数。这一性质使得能够使用为贝叶斯加性回归树设计的回拟合采样器对密度比进行广义贝叶斯推断。我们的贝叶斯策略为推断出的密度比提供了不确定性量化，这对于应用而言至关重要。

    arXiv:2508.03059v5 Announce Type: replace-cross  Abstract: The ratio of two densities provides a direct characterization of their differences. We consider the two-sample comparison problem by estimating this ratio given i.i.d. observations from two distributions. To this end, we propose additive tree models for density ratio estimation along with efficient algorithms using a new loss function, the balancing loss. The loss allows tree-based models to be trained using several algorithms originally designed for supervised learning, such as forward-stagewise optimization and gradient boosting. Moreover, the balancing loss resembles an exponential family kernel, and it can serve as a pseudo-likelihood with conjugate priors. This property enables generalized Bayesian inference on the density ratio using backfitting samplers designed for Bayesian additive regression trees (BART). Our Bayesian strategy provides uncertainty quantification for the inferred density ratio, which is critical for ap
    
[^64]: 贝叶斯优化中高斯过程上置信界算法的更优遗憾界

    Improved Regret Bounds for Gaussian Process Upper Confidence Bound in Bayesian Optimization

    [https://arxiv.org/abs/2506.01393](https://arxiv.org/abs/2506.01393)

    本文通过刻画GP-UCB输入序列的集中行为以更精细地分析高斯过程信息增益，证明了GP-UCB在Matérn核下可达到$\tilde{O}(\sqrt{T})$、在平方指数核下可达到$O(\sqrt{T \ln^2 T})$的遗憾界，填补了与已知最优遗憾界之间的差距。

    

    本文研究贝叶斯优化问题（也称为高斯过程老虎机的贝叶斯设定），其中学习者的目标是在从已知高斯过程（GP）中抽取的函数下最小化遗憾。在具有一定平滑度的Matérn核条件下，我们证明高斯过程上置信界（GP-UCB）算法以高概率实现$\tilde{O}(\sqrt{T})$的累积遗憾。此外，我们的分析在平方指数核下得到了$O(\sqrt{T \ln^2 T})$的遗憾。这些结果填补了GP-UCB现有遗憾上界与Scarlett（2018）给出的最佳已知界之间的差距。我们证明的关键思想是刻画GP-UCB所产生的输入序列的集中行为，从而实现对高斯过程信息增益更为精细的分析。

    arXiv:2506.01393v4 Announce Type: replace-cross  Abstract: This paper addresses the Bayesian optimization problem (also referred to as the Bayesian setting of the Gaussian process bandit), where the learner seeks to minimize the regret under a function drawn from a known Gaussian process (GP). Under a Mat\'ern kernel with a certain degree of smoothness, we show that the Gaussian process upper confidence bound (GP-UCB) algorithm achieves $\tilde{O}(\sqrt{T})$ cumulative regret with high probability. Furthermore, our analysis yields $O(\sqrt{T \ln^2 T})$ regret under a squared exponential kernel. These results fill the gap between the existing regret upper bound for GP-UCB and the best-known bound provided by Scarlett (2018). The key idea in our proof is to capture the concentration behavior of the input sequence realized by GP-UCB, enabling a more refined analysis of the GP's information gain.
    
[^65]: 大语言模型的双重重稳健对齐

    Doubly Robust Alignment for Large Language Models

    [https://arxiv.org/abs/2506.01183](https://arxiv.org/abs/2506.01183)

    提出了一种双重稳健的偏好优化算法（DRPO），只要偏好模型或参考策略中有一个被正确设定，即可实现一致的大语言模型对齐，有效解决了RLHF中的模型设定错误问题。

    

    本文研究了基于人类反馈的强化学习（RLHF），用于将大语言模型与人类偏好进行对齐。尽管RLHF已经展现出令人瞩目的成果，但许多算法对底层偏好模型（如Bradley-Terry模型）、参考策略或奖励函数的设定错误高度敏感，从而导致不理想的微调效果。为了解决模型设定错误的问题，我们提出了一种双重稳健的偏好优化算法，该算法在偏好模型或参考策略中任意一个被正确设定的情况下（无需两者同时正确）仍能保持一致性。我们的方法在理论和实践中都展现出比最先进算法更优越、更稳健的性能。代码已发布于 https://github.com/DRPO4LLM/DRPO4LLM

    arXiv:2506.01183v3 Announce Type: replace-cross  Abstract: This paper studies reinforcement learning from human feedback (RLHF) for aligning large language models with human preferences. While RLHF has demonstrated promising results, many algorithms are highly sensitive to misspecifications in the underlying preference model (e.g., the Bradley-Terry model), the reference policy, or the reward function, resulting in undesirable fine-tuning. To address model misspecification, we propose a doubly robust preference optimization algorithm that remains consistent when either the preference model or the reference policy is correctly specified (without requiring both). Our proposal demonstrates superior and more robust performance than state-of-the-art algorithms, both in theory and in practice. The code is available at https://github.com/DRPO4LLM/DRPO4LLM
    
[^66]: Foreclassing：从预测到决策

    Foreclassing: From Forecasts to Decisions

    [https://arxiv.org/abs/2503.04956](https://arxiv.org/abs/2503.04956)

    本文提出了“Foreclassing”这一全新机器学习问题，即用时间序列的未来观测值来定义其类别标签，给出了数学表述与相关理论结果，并提出了联合执行预测与分类的神经网络模型家族ForeClassNet作为研究基准。

    

    时间序列预测被广泛用于在不确定性条件下为决策提供信息。在许多应用中，这些决策涉及依据尚未被观测到的结果对时间序列进行分类。在本文中，我们提出了Foreclassing，这是一个新的机器学习问题，其中时间序列的类别标签被定义为其未来观测值的函数。我们给出了Foreclassing的数学表述，并建立了刻画该问题的若干理论结果。这些结果描述了额外的未来信息如何影响贝叶斯分类风险，为学习到的Foreclassing模型的分类误差提供了上界，并证明了在训练中使用未来观测值可以带来优于传统分类器的有限样本改进。为了给未来研究提供基准，我们提出了ForeClassNet，这是一类能够同时执行预测和分类的神经网络模型家族。

    arXiv:2503.04956v3 Announce Type: replace  Abstract: Time series forecasts are widely used to inform decisions under uncertainty. In many applications, these decisions involve classifying a time series based on outcomes that have yet to be observed. In this paper, we introduce Foreclassing, a new machine learning problem in which the class label of a time series is defined as a function of its future observations. We provide a mathematical formulation of Foreclassing and establish several theoretical results characterizing the problem. These results describe how additional future information affects Bayes classification risk, provide bounds on the classification error of learned Foreclassing models, and demonstrate how future observations used during training can yield finite-sample improvements over conventional classifiers. To provide a benchmark for future research, we propose ForeClassNet, a family of neural network models that jointly perform forecasting and classification while a
    
[^67]: 预设参数卷积核的机制性解释

    A Mechanistic Interpretation of Prescribed-Parameter Convolutional Kernels

    [https://arxiv.org/abs/2502.17061](https://arxiv.org/abs/2502.17061)

    本文从机制上解释了ROCKET等预设参数卷积方法在时间序列分类中有效的原因：证明其PPV池化特征在周期边界下对平移具有精确不变性，并揭示了有限随机核库会聚集在由参数分布决定的成对几何结构周围。

    

    预设参数卷积变换仅使用固定核、简单池化操作和线性分类器即可实现强大的时间序列分类性能。其成功引发了一个机制性问题：这种表示的哪些性质使该构造有效？我们研究了正值比例（PPV），这是ROCKET、MiniROCKET和MultiROCKET共享的一种池化算子。我们在周期边界条件下建立了PPV对循环平移的精确不变性，并量化了在有限边界下偏离不变性的程度。作为补充分析，我们对平滑PPV替代物的敏感性进行了界定。随后，我们定义了由卷积参数分布所诱导的总体成对几何结构，并展示了有限随机核库如何在该几何结构周围聚集，从而将参数分布的作用与采样核的数量分离开来。在UCR分类数据集上的实验……

    arXiv:2502.17061v2 Announce Type: replace  Abstract: Prescribed-parameter convolutional transforms achieve strong time series classification performance using fixed kernels, simple pooling operations, and a linear classifier. Their success raises a mechanistic question: which properties of the representation make this construction effective? We study the proportion of positive values (PPV), a pooling operator shared by ROCKET, MiniROCKET, and MultiROCKET. We establish exact invariance to circular translations under periodic boundaries and quantify departures from invariance under finite boundaries. A complementary analysis bounds the sensitivity of a smooth PPV surrogate. We then define the population pairwise geometry induced by a distribution of convolutional parameters and show how a finite random kernel bank concentrates around that geometry. This separates the role of the parameter distribution from the number of sampled kernels. Experiments on the UCR classification collection ex
    
[^68]: 基于联合偏回归的稀疏偏相关非渐近估计

    Non-asymptotic Estimation of Sparse Partial Correlation via Joint Partial Regression

    [https://arxiv.org/abs/2502.08414](https://arxiv.org/abs/2502.08414)

    该论文提出了一种两阶段联合偏回归估计方法，能够以非渐近速率同时估计稀疏高维偏相关矩阵与逆协方差矩阵，并同时保证半正定性。

    

    偏相关分析在现代多变量分析中被广泛应用。然而，以往的方法缺乏非渐近保证，且渐近估计速率较慢。我们提出了一种方法，能够以非渐近速率同时估计稀疏的高维偏相关矩阵和逆协方差矩阵。该方法是一种两阶段估计方法，它利用了逆协方差矩阵与线性回归估计方法之间的联系，并同时强制保证矩阵的半正定性。我们还推导了一种高效的近端分裂算法用于数值计算估计值，并提供了该算法的快速实现。所提出方法的有效性在合成数据和真实世界数据上均得到了验证。

    arXiv:2502.08414v3 Announce Type: replace  Abstract: Partial correlation analysis is widely used in modern multivariate analysis. However, prior methods lack non-asymptotic guarantees and have slow asymptotic estimation rates. We present a method for simultaneously estimating sparse high-dimensional partial correlation matrices and inverse covariance with non-asymptotic rates. The method is a two-stage estimation which exploits the connection between the inverse covariance matrix and linear regression estimation method while enforcing positive semi-definiteness simultaneously. An efficient proximal splitting algorithm for numerically computing the estimate is also derived, and we provide a fast implementation of the algorithm. The effectiveness of the proposed method is demonstrated on both synthetic and real-world data.
    
[^69]: 基于目标学习的数据公平性推断

    Targeted Learning for Inference on Data Fairness

    [https://arxiv.org/abs/2502.04309](https://arxiv.org/abs/2502.04309)

    该论文提出利用目标学习这一非参数推断框架对训练数据生成过程的公平性进行统计推断，推导出人口均等、机会均等和条件互信息的估计量，仿真表明其能提高置信区间覆盖率并减少误检。

    

    随着算法决策即使在高风险场景中也变得日益普遍，人们在定义、检测和消除模型驱动的不公平结果方面投入了大量精力。先前的工作主要集中于估计给定预测模型的公平性属性，并基于固定模型在评估数据集上性能的变异性来量化不确定性。与此不同，我们还可以探讨模型从其训练数据中继承的不确定性；在此，我们评估的是数据公平性——即用于训练的数据生成过程中的公平性。我们使用目标学习（targeted learning）这一灵活的非参数推断框架对数据公平性进行推断，并推导出针对人口均等、机会均等和条件互信息的估计量。仿真结果表明，我们的数据公平性估计量提高了置信区间的覆盖率，并减少了对总体公平性的错误检测。

    arXiv:2502.04309v2 Announce Type: replace-cross  Abstract: As algorithmic decision making has become increasingly ubiquitous even in high-stakes settings, significant effort has been invested in defining, detecting, and eliminating unfair model-driven outcomes. Prior work has largely focused on estimating the fairness properties of a given predictive model, and quantifying uncertainty based on variability in a fixed model's performance across evaluation datasets. Alternatively, we can also ask about uncertainty that the model inherits from its training data; here we evaluate data fairness - fairness in the data generating process used for training. We perform inference on data fairness using targeted learning, a flexible framework for nonparametric inference, and derive estimators for demographic parity, equal opportunity, and conditional mutual information. Simulations show that our data fairness estimators improve confidence-interval coverage and reduce false detections of population
    
[^70]: 失败风险的代价：带凸正则化的截断势垒策略梯度

    The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex Regularization

    [https://arxiv.org/abs/2411.19193](https://arxiv.org/abs/2411.19193)

    提出截断势垒策略梯度框架，将截断水平 m 解释为失败风险的“价格”，证明训练价值可精确分解为失败前效用、余量成本与价格×风险三部分，并在凸正则化下实现了总体训练流的指数收敛。

    

    许多操作运行在必须极少耗尽的预算之上，例如磨损余量。一个包装器将剩余预算保留在状态中，减去一个随预算缩减而增长的势垒，并将组合奖励从下方截断于 -m，随后标准的策略梯度学习器无需改动即可正常训练。截断水平 m 是失败风险的“价格”，因为训练得到的价值恰好分解为失败前的效用、余量的成本以及价格乘以风险之和，因此在其优化误差范围内，策略训练返回的策略是在所有风险不更高的策略中，扣除该成本后获得最多此类效用的策略。当被计入的成本保持有界时，其风险超过对比策略的幅度至多为 1/m 的量级，且可由价值函数或验证数据加以认证。主要理论结果是：在任何沿网络单元运动为凸且对价格足够强的参数正则化下，总体训练流具有指数收敛性。

    arXiv:2411.19193v3 Announce Type: replace-cross  Abstract: Many operations run on a budget that must rarely run out, such as a wear allowance. A wrapper keeps the remaining budget in the state, subtracts a barrier that grows as the budget shrinks and clips the combined reward from below at -m, and a standard policy-gradient learner trains unchanged. The clip level m is a price of failure risk, since the trained value splits exactly into utility before failure, the cost of a margin and the price times the risk, so whatever policy training returns earns the most such utility net of that cost among policies with no greater risk, up to its optimization error. Its risk exceeds a comparison policy's by at most order 1/m when charged costs stay bounded, and value or validation data certify it. The main theoretical result is exponential convergence of the population training flow under any parameter regularizer convex along the motions of the network's units and strong enough against the price
    
[^71]: 时序差分学习对不完全覆盖的自适应性：子图上的Oracle不等式

    Temporal difference learning adapts to incomplete coverage: an oracle inequality over subgraphs

    [https://arxiv.org/abs/2411.09731](https://arxiv.org/abs/2411.09731)

    本文证明在表格设定下，普通的时序差分（TD）学习能自动兼具蒙特卡洛方法“仅需目标状态充分访问”与TD自身“渐近高效”的双重优势，并以子图Bellman方差为核心量给出了一个非渐近oracle不等式：无需预先指定任何子图，TD即对每个覆盖充分的子图自适应地达到最优误差界。

    

    时序差分（TD）与蒙特卡洛（MC）是强化学习中价值函数估计的两个基本原则。TD在价值估计上是渐近高效的，而MC只需要充分的目标状态访问。我们证明，在表格情形下，普通的TD会自动地结合这两种方法的优势。我们识别出“子图Bellman方差”——它源于子图内部的自举（bootstrapping）与子图外部的rollout——作为连接MC基准与高效TD基准的一个基本量。在独立轨迹的条件下，我们证明了一个非渐近的oracle不等式：对于每一个包含目标状态且覆盖充分的子图，完整TD的均方误差都被该方差除以样本量所界定，最多相差一个高阶余项，且该余项对占用测度的依赖仅限于该子图之内。算法本身并不需要给定任何子图。因此，TD无需选择子图或要求全局覆盖即可实现自适应。

    arXiv:2411.09731v3 Announce Type: replace-cross  Abstract: Temporal difference (TD) and Monte Carlo (MC) are two fundamental principles for value function estimation in reinforcement learning. TD is asymptotically efficient for value estimation, whereas MC requires only adequate target visitation. We show that these advantages are automatically combined by ordinary TD in the tabular setting. We identify the subgraph Bellman variance, arising from bootstrapping within a subgraph and rollouts outside it, as a fundamental quantity connecting the MC and efficient TD benchmarks. With independent trajectories, we prove a non-asymptotic oracle inequality: the mean squared error of full TD is bounded by this variance divided by the sample size for every sufficiently covered subgraph containing the target, up to a higher-order remainder whose occupancy dependence is confined to that subgraph. No subgraph is supplied to the algorithm. TD thus adapts without selecting a subgraph or requiring glob
    
[^72]: 一种用于精准矩阵的替代图形Lasso算法

    An Alternative Graphical Lasso Algorithm for Precision Matrices

    [https://arxiv.org/abs/2403.12357](https://arxiv.org/abs/2403.12357)

    通过重新参数化精准矩阵最后一列，将正则化正态对数似然分解为两个易于最小化的凸函数，其中一个是Lasso回归问题，从而开发了一种简单的块坐标下降算法来计算GLasso更新，性能可与DP-GLas相媲美。

    

    Graphical Lasso (GLasso)算法是一种快速且广泛用于估计稀疏精准矩阵的方法。本文通过使用新的并略微不同的精准矩阵最后一列的重新参数化，展示了正则化正态对数似然自然地分解成两个易于最小化的凸函数之和，其中之一是Lasso回归问题。这种分解是开发透明、简单的迭代块坐标下降算法的关键，用于计算GLasso更新，其性能与DP-GLas相当。

    arXiv:2403.12357v1 Announce Type: cross  Abstract: The Graphical Lasso (GLasso) algorithm is fast and widely used for estimating sparse precision matrices (Friedman et al., 2008). Its central role in the literature of high-dimensional covariance estimation rivals that of Lasso regression for sparse estimation of the mean vector. Some mysteries regarding its optimization target, convergence, positive-definiteness and performance have been unearthed, resolved and presented in Mazumder and Hastie (2011), leading to a new/improved (dual-primal) DP-GLasso. Using a new and slightly different reparametriztion of the last column of a precision matrix we show that the regularized normal log-likelihood naturally decouples into a sum of two easy to minimize convex functions one of which is a Lasso regression problem. This decomposition is the key in developing a transparent, simple iterative block coordinate descent algorithm for computing the GLasso updates with performance comparable to DP-GLas
    

