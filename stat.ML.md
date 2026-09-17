# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A General Kernel Framework for Non-CND Distance Measures Using |D|-Dimensional Sparse Landmark Embeddings](https://arxiv.org/abs/2609.19083) | 提出稀疏地标嵌入（SLE）核框架，通过紧支撑凸块函数将输入嵌入为稀疏特征向量，使得任意距离度量（包括非条件负定度量）都能构造出可证明半正定的核矩阵，从而完全摆脱核方法（如高斯过程）对希尔伯特距离条件的依赖。 |
| [^2] | [Fast Learning Rates for Physics-Informed Kernel Methods](https://arxiv.org/abs/2609.18901) | 本文为结合数值观测与微分观测的物理信息核估计器证明了有限样本误差界，揭示了预测误差的双区间结构：当微分观测有限时，误差速率同时依赖于数值与微分观测的数量，而当微分观测数量超过阈值后，误差速率达到饱和并与完美物理约束下的最优速率相匹配。 |
| [^3] | [Stable Filters for Generative Modeling of Graph Signals](https://arxiv.org/abs/2609.18759) | 本文针对漂移项结合图滤波器与图神经网络的图感知连续时间生成模型，推导了量化图扰动对生成分布影响的显式Wasserstein稳定性界，并据此提出了在保持图热扩散平滑性的同时增强结构稳定性的图滤波器设计原则框架。 |
| [^4] | [When Edit Flows are Edit Jumps: replicating Edit Flows and EvoFlows](https://arxiv.org/abs/2609.18745) | 本文证明Edit Flows与EvoFlows本质上是同一底层过程（连续时间中编辑逐个触发的纯跳跃式生成器匹配），并发布首个开源实现EditJumps——一个在166万同源抗体对上训练的通用抗体编辑器，可零样本编辑未见先导序列而无需按家族重新训练。 |
| [^5] | [Rank and computation of the pathlifting Jacobian of a DAG ReLU network](https://arxiv.org/abs/2609.18682) | 本文通过对骨架矩阵进行初等归纳证明了DAG ReLU网络路径提升雅可比矩阵的秩，并提出了一种无需反向传播、计算成本更低的雅可比矩阵计算方法。 |
| [^6] | [Revisiting Distributed Sign-Based Variance Reduction](https://arxiv.org/abs/2609.18656) | 本文通过提出在服务器端利用递归梯度增量的无偏压缩来跟踪全局梯度，解决了数据异构情况下符号聚合引入偏差的问题，首次在非凸随机优化和有限和优化中实现了基于符号的分布式方差缩减方法的最优收敛速率。 |
| [^7] | [How Many Labels Does Model Choice Need? Certificates and Budgets for Selective Prediction](https://arxiv.org/abs/2609.18622) | 该论文量化了比较模型选择性预测性能（AUGRC）所需的标签预算，通过预标签下界与覆盖线性规划证书证明：确定性选择模型在某些条件下几乎需要标注全部标签，而准确率选择可由少量分歧标签裁决。 |
| [^8] | [Provable Guarantees and Efficient Learning of Structural Equation Models with Latent Confounders](https://arxiv.org/abs/2609.18535) | 本文针对含潜在混杂因子的线性结构方程模型，提出了一种通过将精度矩阵分解为稀疏加低秩两部分来迭代重建观测变量因果有向无环图的高效算法，并给出了可证明的正确性保证。 |
| [^9] | [Spatially Adaptive Noise Injection](https://arxiv.org/abs/2609.18466) | 本文提出空间自适应噪声注入（SANI）采样框架，通过概率门控机制和空间自适应方差在逐像素层面动态调整噪声注入，在去噪器不确定的边缘纹理区域施加随机校正，而在分数估计精确的平滑区域保持确定性更新。 |
| [^10] | [Gradient Descent with Stochastic Subspaces via Persistence of Memory](https://arxiv.org/abs/2609.18416) | 本文提出“记忆持久性”技术，利用一个与梯度弱相关、可长期固定无需频繁更新的指导向量来引导随机子空间的生成，从而显著扩展并改进了大规模优化中的随机子空间梯度下降方法，且该向量可借助稀疏性或小批量等结构化特性以低成本高效获得。 |
| [^11] | [Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts](https://arxiv.org/abs/2609.18366) | 提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。 |
| [^12] | [Beyond Quadratic Loss: The Stability Phase Diagram of Adam](https://arxiv.org/abs/2609.18314) | 该研究通过绘制Adam优化器在$(\beta_1,\beta_2)$参数平面上的稳定性相图，发现一条近似线性边界$1-\beta_2=C(1-\beta_1)$可用于区分训练中是否出现损失尖峰，并揭示超二次损失景观（如高置信交叉熵损失形成的“核心-墙壁”结构）是决定该边界形状的关键因素。 |
| [^13] | [Preservation of Log-Concavity and Convergence of Wasserstein-Fisher-Rao Gradient Flows](https://arxiv.org/abs/2609.18118) | 本文证明Wasserstein-Fisher-Rao梯度流在满足曲率条件的强对数凹目标分布下能够保持强对数凹性，据此推导出对称化KL散度的显式非渐近收敛速率，无需暖启动，且收敛速率可加性分解为Wasserstein与Fisher-Rao两部分贡献。 |
| [^14] | [Matching Multi-Loop Complexities with a Single Loop: Optimal Optimization Stationarity and Best-Known Game Stationarity in Nonconvex--Concave Minimax Optimization](https://arxiv.org/abs/2609.17973) | 该论文提出了一种结合投影外梯度更新、对偶动量和移动近端中心的单循环投影阻尼外梯度算法，在非凸-凹极小极大优化中以单循环方法匹配了多循环方法的复杂度，同时实现了最优的优化平稳性保证和已知最佳的博弈平稳性保证。 |
| [^15] | [On the Identifiability of Mixed Ordinal and Exponential Family Causal DAGs under Linear Parametric Models](https://arxiv.org/abs/2609.17942) | 本文证明了在线性参数模型中，只要有序节点至少有三个类别且指数族节点至少有三个支撑点，连接这两类节点的每条边的方向都可仅凭联合分布在任意参数取值下被辨识，并通过反向论证证明了这两个条件的必要性。 |
| [^16] | [Symmetry without a manifold: intrinsic dimension on orbits](https://arxiv.org/abs/2609.17926) | 该论文证明在对称性轨道（如模加法任务）上标准内在维度估计器普遍失效，神经缩放行为不再遵循幂律，而是遵循关于隐藏层宽度的指数定律 $L(h)=L_\infty+A\exp(-c\,h^{\alpha})$。 |
| [^17] | [Generalized DCCQ: From Binary Quotients to Multinomial Simplex Geometry and Critical-Strip Coordinates](https://arxiv.org/abs/2609.17899) | 该论文将离散复补商（DCCQ）框架从二元伯努利计数推广到多项计数组合，证明了m≥2时多项DCCQ坐标映射是概率单纯形上的实解析微分同胚，其中二元情形给出临界线坐标、三元情形覆盖完整临界带，但明确不声称证明黎曼假设。 |
| [^18] | [TabPFN-3.5: Technical Report](https://arxiv.org/abs/2609.17895) | TabPFN-3.5 是一款新的旗舰表格基础模型，在标准及非独立同分布、多模态、高基数、宽表等实际表格任务上全面超越 TabPFN-3 和现有基线，并提供了速度提升最高 3 倍的 TabPFN-3.5-Fast 和增强多模态能力的 TabPFN-3.5-Plus 变体。 |
| [^19] | [Bracketing Uncertainty in Clustering Under the Manifold Hypothesis](https://arxiv.org/abs/2609.17892) | 该论文通过结合内在流形几何（体积增长与触及半径）和样本级度量（填充距离与密度），为互k近邻图聚类建立了阈值现象，从而界定了聚类结果存在不确定性的几何区间。 |
| [^20] | [Sharp margin-based generalization bounds for realizable SVM](https://arxiv.org/abs/2609.17845) | 该论文通过确定性删除问题的分析，证明了可实现情形下硬间隔支持向量机的泛化风险以概率至少\(1-\delta\)不超过\(\frac{C}{m}(K_m+\log\frac{1}{\delta})\)，其中\(K_m=r_m^2/\gamma_m^2\)为半径-间隔复杂度，得到了阶为\(1/m\)且依赖半径-间隔复杂度的尖锐泛化界。 |
| [^21] | [METALICA: METAdynamics and repLICA exchange for enhanced diffusion sampling](https://arxiv.org/abs/2609.17823) | METALICA通过副本交换机制在预训练扩散模型上实现元动力学，利用偏置势采样和重加权高效探索蛋白质构象的稀有状态，从而实现对稀有事件的有效发现。 |
| [^22] | [Approximating Measures on Function Spaces: Transport and Truncation](https://arxiv.org/abs/2609.17802) | 该论文提出了一类与易处理参考测度仅相差有限维映射的函数空间测度，通过分块三角传输映射对低维推前分布进行采样，再利用参考条件分布将其补全到函数空间，从而实现结构保持的高效函数空间测度逼近与采样。 |
| [^23] | [Random tilts to find stationary points in stochastic convex optimization](https://arxiv.org/abs/2609.17798) | 该论文证明正则化经验风险最小化结合随机倾斜扰动能以 $\sqrt{d/n}$ 的残差阶找到随机凸函数及相关变分不等式的驻点，并通过 $\sqrt{\log d/n}$ 量级的极小极大下界表明一定的维度依赖性是不可避免的。 |
| [^24] | [Efficient Robust Learning at the Information-Theoretic Limit](https://arxiv.org/abs/2609.17655) | 本文解决了 Blanc 遗留的开放问题，通过巧妙运用无悔学习器技术，首次给出了在 ERM 预言机辅助下达到信息论最优错误率 η+ε 的多项式时间鲁棒学习算法，并为具有三明治多项式性质的函数类提供了无需预言机的高效算法。 |
| [^25] | [Fenchel-Young Duality Gaps: Certified Early Stopping for Regularized Inverse Problems](https://arxiv.org/abs/2609.17629) | 本文提出了一个精确的对偶间隙恒等式，将正则化逆问题的总间隙分解为数据保真与正则项两个Fenchel-Young损失，从而给出可计算、无需oracle的误差界，实现了可认证的早停。 |
| [^26] | [Stability-Constrained Approximation in Spline KANs: Exact Layer Balancing and Budget-Compatible Saturation](https://arxiv.org/abs/2609.17619) | 该论文在严格逐层Lipschitz预算约束下研究深度样条KAN的逼近理论，精确求解了有限深度对角层平衡问题（给出最优层预算的闭式解与单遍最小化算法），并提出了保持预算约束的构造性样条离散化定理。 |
| [^27] | [R\'enyi Tracking Bounds for Langevin Dynamics with Moving Targets](https://arxiv.org/abs/2609.17577) | 该论文首次建立了具有离散目标更新的朗之万动力学的非渐近Rényi散度追踪界，并将其应用于基于连续Moreau包络的非光滑采样，给出了显式的参数选择和复杂度保证。 |
| [^28] | [Pay Only for Disagreement: Certified No-Regression Verdicts for Model Updates with Matching Label-Complexity Bounds](https://arxiv.org/abs/2609.17560) | 论文提出DISCERN协议，利用“模型间风险差仅存在于分歧输入上且无需标签即可观测”这一关键性质，通过零标签层与仅标注分歧样本的审计层两层序贯协议，为模型更新提供无回归认证，并证明标签复杂度为rho²/ε²，相比不考虑配对关系的审计器可节省1/ρ的标注成本。 |
| [^29] | [Learning Interaction Kernels from Collective Steady States](https://arxiv.org/abs/2609.12004) | 该论文提出了一种仅需从集体稳态的单快照观测中学习相互作用粒子系统相互作用核的方法，通过基于观测构型经验分布的正则化策略解决了本质上不适定的逆问题，实现了对相互作用规律的稳定准确恢复以及对集体行为乃至其动力学过程的忠实重现。 |
| [^30] | [Semiparametric Inference for Conditional Shapley Feature Importance](https://arxiv.org/abs/2609.10313) | 本文针对条件Shapley特征重要性提出了一种带K折交叉拟合和U统计量修正的半参数一步估计器，消除了蒙特卡洛偏差，在双重稳健速率条件下实现√n一致性与渐近正态性，并提供覆盖率有保证的Wald置信区间。 |
| [^31] | [Improved Regret Analysis for Parallel Gaussian Process Bandit Optimization](https://arxiv.org/abs/2608.16492) | 本文通过GP-BTS示例，证明无需初始不确定性采样阶段即可消除批量大小对遗憾上界的乘性影响，并在无噪声条件下实现更优的遗憾界限。 |
| [^32] | [Simple-regret rates and minimax optimality of fixed-prior expected improvement in Mat\'ern and squared-exponential RKHSs](https://arxiv.org/abs/2607.29245) | 本文证明了在Matérn核和平方指数核的再生核希尔伯特空间中，弱期望改进策略的简单遗憾率达到极小极大最优，分别以 $O(N^{-\nu/d})$ 和指数级速率收敛。 |
| [^33] | [Optimizing the Preconditioner: A Black-box Online-to-Nonconvex Conversion with Static Regret Minimization Oracles](https://arxiv.org/abs/2607.17607) | 本文提出了一种从随机非凸优化到在线凸优化中静态遗憾最小化的黑盒归约方法，解决了Chen和Hazan（2024）提出的开放问题，并证明任何具有O(√T)遗憾的OCO预言机都能恢复经典的O(T^{-1/2})收敛速率。 |
| [^34] | [Subjective Risk Decomposition: A New View for Uncertainty Quantification](https://arxiv.org/abs/2607.15196) | 该论文提出将不确定性度量视为主观风险分解的产物而非基本原语，证明了基于严格恰当损失对主观风险进行分解即可推导出认知不确定性与偶然不确定性，从而为不确定性量化提供了统一的理论框架和新范式。 |
| [^35] | [Conformal Prediction for Dyadic Regression Under Complex Missingness](https://arxiv.org/abs/2606.11136) | 本文提出了一个在复杂缺失机制下用于二元回归的共形预测框架，通过新颖的双射论证和多种程序（如行列方法和选择性共形）实现了有限样本有效性和掩码条件有效性。 |
| [^36] | [CP-factorization for high dimensional tensor time series and double projection iterations](https://arxiv.org/abs/2606.08560) | 本文提出基于CP分解的高维张量时间序列因子载荷估计方法，通过单次特征值分析和新型双重投影迭代算法，在因子相关、载荷非正交等一般条件下建立理论性质并提升收敛速度。 |
| [^37] | [On Finite-sample Concentration of Median of Incomplete U-Statistics](https://arxiv.org/abs/2606.00661) | 本文证明了不完整U统计量中位数（MoIU）的有限样本浓度界，克服了此前仅能获得松散$O(n^{-1/4})$界的理论挑战，实现了更紧的收敛速率。 |
| [^38] | [A Continuous-Time Ensemble Kalman-Bucy Smoother for Causal Inference and Model Discovery](https://arxiv.org/abs/2604.25157) | 本文提出了一种连续时间集合Kalman-Bucy平滑器（EnKBS），通过集合矩重构条件分布，为非线性动力系统的数据同化提供了无需导数、切线性或伴随模型的平滑框架，实现了超越滤波的不确定性降低，并可应用于因果推断与模型发现。 |
| [^39] | [Deep Learning for Sequential Decision Making under Uncertainty: Foundations, Frameworks, and Frontiers](https://arxiv.org/abs/2604.11507) | 本教程以运筹学/管理科学（OR/MS）为核心视角，系统性地连接了深度学习神经架构与不确定性下序贯决策的OR/MS方法，其核心观点是深度学习是对优化的补充而非替代。 |
| [^40] | [Symmetrizing Bregman Divergence on the Cone of Positive Definite Matrices: Which Mean to Use and Why](https://arxiv.org/abs/2603.28917) | 该论文揭示了正定矩阵锥上对称化Bregman散度的变分原理，证明前向对称化的规范均值是原始空间上的算术平均，而反向对称化的规范均值是对偶空间算术平均的拉回，在常用情形下分别对应算术、对数欧几里得和调和平均。 |
| [^41] | [Bayesian Quadrature](https://arxiv.org/abs/2602.16218) | 本综述首次系统全面地梳理了贝叶斯求积方法，涵盖其数学基础、建模-推断-采样三维分类体系、理论保证、数值实验对比以及实际应用中的挑战与局限性。 |
| [^42] | [Correcting Boundary Bias and Observation Independence in Bayesian Experimental Design](https://arxiv.org/abs/2602.01898) | 论文针对基于方差采集准则的高斯过程主动学习的两大缺陷——后验方差与观测内容无关以及边界处方差膨胀导致的过度采样，提出了修正方案，通过重构驱动的设计密度与基于后验均值的免训练变形，使采样更集中于目标函数变化剧烈的区域。 |
| [^43] | [Finite-Sample Unbiased Variance of MMD under Unbalanced Sampling: Exact Estimation and Quasi-Linear Computation](https://arxiv.org/abs/2601.13874) | 该论文推导了非平衡采样下MMD方差的有限样本无偏估计量，并通过拉普拉斯核的递归前缀-后缀累加方案将计算复杂度从 $\mathcal{O}(N^2)$ 降至 $\mathcal{O}(N \log N)$、内存仅需 $\mathcal{O}(N)$。 |
| [^44] | [An operator splitting analysis of Wasserstein--Fisher--Rao gradient flows](https://arxiv.org/abs/2511.18060) | 本文定量分析了求解 WFR 梯度流时 W-FR 算子分裂的顺序与步长的影响，并出人意料地证明：合理选择步长和算子顺序时，分裂方案可以比精确 WFR 流更快地收敛到目标分布。 |
| [^45] | [Optimal Post-processing of Synthetic Data for Pearson Correlation Matching](https://arxiv.org/abs/2510.02405) | 本文提出一种与生成器无关的后处理方法，通过对合成数据进行最小改动以恢复原始数据的皮尔逊相关矩阵，给出了该最小化问题的唯一显式解及修正幅度的理论界限，且在保持边际分布、数据几何结构和分类性能方面表现良好。 |
| [^46] | [Spectral gap of Metropolis-within-Gibbs under log-concavity](https://arxiv.org/abs/2509.26175) | 该论文通过精确估计一维随机游走Metropolis核的传导率，将对数凹分布下随机扫描Metropolis-within-Gibbs算法的谱隙下界从 $\Omega((\kappa^2 d)^{-1})$ 改进为 $\Omega((\kappa d)^{-1})$，证明其混合性能仅比精确Gibbs采样器差一个常数因子。 |
| [^47] | [A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](https://arxiv.org/abs/2509.19276) | 提出了一种免训练的扩散正则化Wasserstein梯度流方法（DWGF），利用预训练潜在扩散模型作为先验来求解不适定逆问题。 |
| [^48] | [Physics-Informed Sylvester Normalizing Flows for Bayesian Inference in Magnetic Resonance Spectroscopy](https://arxiv.org/abs/2505.03590) | 该论文提出了一种基于Sylvester归一化流的贝叶斯推断框架，结合融入物理先验知识的解码器，用于磁共振波谱中代谢物浓度的可靠定量化。 |
| [^49] | [Functional BART with Shape Priors: A Bayesian Tree Approach to Constrained Functional Regression](https://arxiv.org/abs/2502.16888) | 提出了一种结合样条表示与树形分割结构的非参数贝叶斯方法FBART用于函数对标量回归，并通过引入单调性、凸性等形状先验约束来增强估计与预测性能。 |
| [^50] | [Breaking the $T^{2/3}$ Barrier for Sequential Calibration](https://arxiv.org/abs/2406.13668) | 本文首次突破了序贯校准问题中 Foster & Vohra 提出的 $O(T^{2/3})$ 校准误差上界，改进了这一停滞二十余年的经典界限。 |
| [^51] | [Limits of Transfer Learning](https://arxiv.org/abs/2006.12694) | 该论文在算法搜索框架下证明了迁移学习的若干理论极限，表明迁移信息必须经过谨慎选择并与目标问题存在依赖关系，同时算法的概率变化程度决定了其性能改进的上限。 |

# 详细

[^1]: 基于|D|维稀疏地标嵌入的非CND距离度量通用核框架

    A General Kernel Framework for Non-CND Distance Measures Using |D|-Dimensional Sparse Landmark Embeddings

    [https://arxiv.org/abs/2609.19083](https://arxiv.org/abs/2609.19083)

    提出稀疏地标嵌入（SLE）核框架，通过紧支撑凸块函数将输入嵌入为稀疏特征向量，使得任意距离度量（包括非条件负定度量）都能构造出可证明半正定的核矩阵，从而完全摆脱核方法（如高斯过程）对希尔伯特距离条件的依赖。

    

    核方法，尤其是高斯过程（GP），需要希尔伯特距离度量（即其平方为条件负定（CND）的度量）来保证核矩阵的半正定性（PSD）；而这一条件在许多自然输入空间上并不成立，包括光滑流形和概率分布空间。我们提出了稀疏地标嵌入（SLE）核，彻底消除了这一要求。每个输入通过以全部|D|个训练点为中心的紧支撑凸块函数被嵌入为稀疏特征向量；在该嵌入空间中应用任何标准PSD核，即可得到对任意距离度量均可证明为半正定的核。紧支撑特性自动控制了嵌入的稀疏性，使得尽管环境维度很高，核矩阵仍保持良态且计算上可行。我们对PSD性质、稀疏性、稳定性以及普适性提供了理论保证。

    arXiv:2609.19083v1 Announce Type: cross  Abstract: Kernel methods, and Gaussian Processes (GPs) in particular, require a Hilbertian distance measure---one whose square is conditionally negative definite (CND)---to guarantee positive semi-definiteness (PSD) of the kernel matrix; a condition that fails for many natural input spaces, including smooth manifolds and spaces of probability distributions. We propose the Sparse Landmark Embedding (SLE) kernel, which eliminates this requirement entirely. Each input is embedded into a sparse feature vector via compactly supported bump functions centered at all |D| training points; applying any standard PSD kernel in this embedding space yields a kernel that is provably PSD for arbitrary distance measures. The compact support automatically controls embedding sparsity, keeping kernel matrices well-conditioned and computationally tractable despite the high ambient dimension. We provide theoretical guarantees on PSD, sparsity, stability, and universa
    
[^2]: 物理信息核方法的快速学习速率

    Fast Learning Rates for Physics-Informed Kernel Methods

    [https://arxiv.org/abs/2609.18901](https://arxiv.org/abs/2609.18901)

    本文为结合数值观测与微分观测的物理信息核估计器证明了有限样本误差界，揭示了预测误差的双区间结构：当微分观测有限时，误差速率同时依赖于数值与微分观测的数量，而当微分观测数量超过阈值后，误差速率达到饱和并与完美物理约束下的最优速率相匹配。

    

    在物理信息机器学习中，目标函数 $u^*$ 从带噪声的数值观测 $y_i=u^*(x_i)+ \varepsilon_i$ 中学习，同时结合微分信息，微分信息既可以由带噪声的观测 $d_j=(Du^*)(z_j)+\xi_j$ 给出，也可以由已知的物理约束 $Du^*=v$ 给出。我们考虑 $D$ 为线性微分算子的情形，并分析一种结合 $n$ 个数值观测与 $m$ 个微分观测的物理信息核估计器 $\hat u$。在此背景下，我们探究微分信息能在多大程度上改善预测，以及这种改善在数量上如何依赖于 $n$、$m$ 和 $D$。我们在数值模拟的支持下证明了有限样本界，揭示了预测误差的双区间结构：当 $m$ 有限时，误差速率同时依赖于 $n$ 和 $m$；当 $m$ 超过依赖于问题的阈值时，速率达到饱和，并与拥有完美约束 $D$ 时的最优速率（oracle rate）相匹配。

    arXiv:2609.18901v1 Announce Type: cross  Abstract: In physics-informed machine learning, a target function $u^*$ is learned from noisy value observations $y_i=u^*(x_i)+ \varepsilon_i$, together with differential information, given either by noisy observations $d_j=(Du^*)(z_j)+\xi_j$ or by a known physical constraint $Du^*=v$. We consider the setting where $D$ is a linear differential operator and analyze a physics-informed kernel estimator $\hat u$ combining $n$ value observations and $m$ differential observations. In this context, we ask how much can differential information improve predictions, and how does this improvement depend quantitatively on $n$, $m$, and $D$. We prove finite-sample bounds, supported by numerical simulations, revealing a two-regime structure for the prediction error. When $m$ is limited, the rate depends jointly on $n$ and $m$; when $m$ exceeds a problem-dependent threshold, the rate saturates and matches the oracle rate obtained when the perfect constraint $D
    
[^3]: 图信号生成建模中的稳定滤波器

    Stable Filters for Generative Modeling of Graph Signals

    [https://arxiv.org/abs/2609.18759](https://arxiv.org/abs/2609.18759)

    本文针对漂移项结合图滤波器与图神经网络的图感知连续时间生成模型，推导了量化图扰动对生成分布影响的显式Wasserstein稳定性界，并据此提出了在保持图热扩散平滑性的同时增强结构稳定性的图滤波器设计原则框架。

    

    在图上生成信号需要具有置换等变性且对相对结构扰动保持稳定的模型。尽管最近的图感知薛定谔桥模型将拓扑信息直接融入其参考动力学中，但图的扰动如何通过这些动力学传播并影响最终生成的分布仍不清楚。在本文中，我们分析了图感知连续时间生成模型的结构稳定性，该类模型的漂移项结合了图滤波器与可学习的图神经网络。我们推导出了显式的Wasserstein稳定性界，用以量化相对图扰动对生成分布的影响。受这些界的启发，我们提出了一个设计稳定图滤波器的原则性框架，该框架在保持图热扩散平滑行为的同时增强了结构稳定性。在合成信号和fMRI信号上的实验表明……

    arXiv:2609.18759v1 Announce Type: cross  Abstract: Generating signals on graphs requires permutation-equivariant models that exhibit stability with respect to relative structural perturbations. While recent graph-aware Schr\"odinger bridge models incorporate topology information directly into their reference dynamics, it is unclear how perturbations of the graph propagate through these dynamics and affect the resulting generated distributions. In this paper, we analyze the structural stability of graph-aware continuous-time generative models whose drift combines a graph filter with a learned graph neural network. We derive explicit Wasserstein stability bounds that quantify the effect of relative graph perturbations on the generated distributions. Motivated by these bounds, we introduce a principled framework for designing stable graph filters that preserve the smoothing behavior of graph heat diffusion, while boosting structural stability. Experiments on synthetic and fMRI signals sho
    
[^4]: 当编辑流即为编辑跳变：复现Edit Flows与EvoFlows

    When Edit Flows are Edit Jumps: replicating Edit Flows and EvoFlows

    [https://arxiv.org/abs/2609.18745](https://arxiv.org/abs/2609.18745)

    本文证明Edit Flows与EvoFlows本质上是同一底层过程（连续时间中编辑逐个触发的纯跳跃式生成器匹配），并发布首个开源实现EditJumps——一个在166万同源抗体对上训练的通用抗体编辑器，可零样本编辑未见先导序列而无需按家族重新训练。

    

    抗体先导物优化需要对现有候选分子进行少量且有界的编辑：不仅包括替换，还包括插入和删除。基于编辑的生成模型是唯一能够在不预先固定编辑位置、编辑次数或输出长度的情况下分配这种编辑预算的模型。然而，现有方法Edit Flows和EvoFlows并未发布代码或完整的训练规范。在本研究中，我们证明这两种方法遵循相同的底层过程——编辑以学习到的速率在连续时间中逐个触发——即有限序列上生成器匹配的纯跳跃情形。通过EditJumps，我们推出了该框架的首个开源实现，利用在166万个观察抗体空间（Observed Antibody Space）同源序列对上训练的单一通用抗体编辑器，为种子序列提出类同源变体，以零样本方式编辑未见过的先导序列，而无需原始方法所要求的按家族重新训练。

    arXiv:2609.18745v1 Announce Type: new  Abstract: Antibody lead optimization calls for a small, bounded set of edits to an existing candidate: substitutions, but also insertions and deletions. Edit-based generative models are the only ones that allocate such an edit budget without fixing the edit positions, the edit count, or the output length in advance. However, the existing approaches Edit Flows and EvoFlows did not release code or complete training specifications. Here, we show that both methods follow the same underlying process -- edits firing one at a time, at learned rates, in continuous time -- the pure-jump case of generator matching over finite sequences. With EditJumps we introduce the first open implementation of this framework, with a single generalist antibody editor trained on 1.66M Observed Antibody Space homolog pairs to propose homolog-like variants of a seed sequence, editing unseen leads zero-shot, without the per-family retraining original approaches require. Repli
    
[^5]: DAG ReLU网络路径提升雅可比矩阵的秩与计算

    Rank and computation of the pathlifting Jacobian of a DAG ReLU network

    [https://arxiv.org/abs/2609.18682](https://arxiv.org/abs/2609.18682)

    本文通过对骨架矩阵进行初等归纳证明了DAG ReLU网络路径提升雅可比矩阵的秩，并提出了一种无需反向传播、计算成本更低的雅可比矩阵计算方法。

    

    本文通过对网络的隐藏节点数量进行归纳，为DAG ReLU网络的路径提升雅可比矩阵的秩提供了一个自包含的证明。实际上，这种归纳是初等的，关键方法在于考虑网络的骨架矩阵（一个编码网络路径的稀疏矩阵），并将其中的一个隐藏神经元的表示转换为输出节点。该证明依赖于一些中间命题，这些命题将路径提升、其雅可比矩阵、网络参数及其骨架矩阵联系起来，除了能够得出路径提升雅可比矩阵秩的结论外，还提供了一种无需反向传播即可计算该矩阵的方法，其计算成本在实践中比常规的反向传播高效得多。本文附带一个Python模块，该模块实现了论文中针对前馈网络的各个命题，并用于实验性地量化计算……

    arXiv:2609.18682v1 Announce Type: cross  Abstract: This paper provides a self-contained proof of the rank of the pathlifting Jacobian of a DAG ReLU network by performing an induction on the network's number of hidden nodes. In fact, the induction is elementary, and the key recipe is to consider the skeleton matrix of the network, a sparse matrix encoding the network paths, and transform the representation of one of its hidden neurons into an output node. The proof relies on intermediate propositions which link the pathlifting, its Jacobian, the network parameters, and its skeleton matrix, which, on top of permitting to conclude on the rank of the pathlifting Jacobian, also provide a way to compute it without backpropagation and whose computation cost is super efficient in practice compare to usual backpropagation. The paper is provided with a Python module that implements the different propositions of the paper for feed forward networks and is used to experimentally quantifies the comp
    
[^6]: 重新审视基于符号的分布式方差缩减方法

    Revisiting Distributed Sign-Based Variance Reduction

    [https://arxiv.org/abs/2609.18656](https://arxiv.org/abs/2609.18656)

    本文通过提出在服务器端利用递归梯度增量的无偏压缩来跟踪全局梯度，解决了数据异构情况下符号聚合引入偏差的问题，首次在非凸随机优化和有限和优化中实现了基于符号的分布式方差缩减方法的最优收敛速率。

    

    基于符号的方法可以降低分布式环境中的通信成本，但当数据异构时，聚合本地符号可能会引入偏差。因此，现有的基于符号的方差缩减方法无法获得最优收敛速率。在本文中，我们解决了这个问题，并在非凸随机优化和有限和优化中都获得了最优收敛速率。我们首先给出了一个反例，表明即使使用精确的本地梯度，多数投票也可能无法逼近稳定点。受此局限性的启发，我们提出通过递归梯度增量的无偏压缩在服务器端跟踪全局梯度。由此，我们获得了 $\ell_1$ 范数的收敛速率 $O(\sqrt{d/K}+\sqrt d (a/(nK))^{1/3})$ 以及 $\ell_2$ 范数的收敛速率 $O(\sqrt{a/K}+\sqrt a/(nK)^{1/3})$。其中，$K$ 为迭代次数，$n$ 为工作节点数量，$d$ 为维度，$a=1+\omega$，其中 $\omega$……

    arXiv:2609.18656v1 Announce Type: new  Abstract: Sign-based methods reduce communication costs in distributed environments, but aggregating local signs can introduce bias when data are heterogeneous. As a result, existing sign-based variance reduction methods fail to obtain the optimal convergence rates. In this paper, we solve this problem and obtain optimal rates for both nonconvex stochastic and finite-sum optimization. We first give a counterexample showing that majority voting can fail to approach stationary points even with exact local gradients. Motivated by this limitation, we propose tracking the global gradient at the server through unbiased compression of recursive gradient increments. As a result, we can obtain the convergence rates of $O(\sqrt{d/K}+\sqrt d (a/(nK))^{1/3})$ for the $\ell_1$-norm and $O(\sqrt{a/K}+\sqrt a/(nK)^{1/3})$ for the $\ell_2$-norm. Here, $K$ is the iteration number, $n$ is the number of workers, $d$ is the dimension, and $a=1+\omega$, with $\omega$ 
    
[^7]: 模型选择需要多少标签？选择性预测的证书与预算

    How Many Labels Does Model Choice Need? Certificates and Budgets for Selective Prediction

    [https://arxiv.org/abs/2609.18622](https://arxiv.org/abs/2609.18622)

    该论文量化了比较模型选择性预测性能（AUGRC）所需的标签预算，通过预标签下界与覆盖线性规划证书证明：确定性选择模型在某些条件下几乎需要标注全部标签，而准确率选择可由少量分歧标签裁决。

    

    分类器可以做出相同的预测，却仍需要标签才能比较它们的选择性性能：置信度排序会以不同的方式加权相同的错误。我们针对广义风险-覆盖曲线下面积（AUGRC）量化了这一标签需求。预标签下界可以排除不充足的预算。当所有标签已知时，一个覆盖线性规划界定了足以确定胜者的最少标签数（即证书大小），对于K个候选者，该上界为K-1个标签。对于固定的K、独立均匀排序且预测完全相同的情况，预标签下界接近样本池的四分之一。当错误为独立于排序的独立同分布伯努利错误时，任何精确的获取策略在渐近意义上几乎需要读取全部标签，尽管双候选证书只需一半。在九个数据集上的108次特征面板比较中，分歧标签可以裁决所有准确率选择，却无法裁决任何AUGRC选择；在96种条件下，20%的预算被证明是不够的。

    arXiv:2609.18622v1 Announce Type: new  Abstract: Classifiers can make identical predictions yet require labels to compare their selective performance: confidence ranks weight the same errors differently. We quantify this requirement for the area under the generalized risk-coverage curve (AUGRC). A prelabel lower bound rules out insufficient budgets. With all labels known, a covering linear program bounds the minimum number of labels sufficient to fix the winner (the certificate size) within $K-1$ labels for $K$ candidates. For fixed $K$, independent uniform orders and identical predictions, the prelabel bound approaches one quarter of the pool. With iid Bernoulli errors independent of the orders, every exact acquisition policy reads almost all labels asymptotically, although a two-candidate certificate needs only half. Across 108 feature-panel comparisons on nine datasets, disagreement labels settle every accuracy choice but no AUGRC choice. A 20% budget is ruled out in 96 conditions; 
    
[^8]: 具有潜在混杂因子的结构方程模型的可证明保证与高效学习

    Provable Guarantees and Efficient Learning of Structural Equation Models with Latent Confounders

    [https://arxiv.org/abs/2609.18535](https://arxiv.org/abs/2609.18535)

    本文针对含潜在混杂因子的线性结构方程模型，提出了一种通过将精度矩阵分解为稀疏加低秩两部分来迭代重建观测变量因果有向无环图的高效算法，并给出了可证明的正确性保证。

    

    因果发现旨在从观测数据中恢复变量之间的因果关系。在众多领域中，探索变量间的因果关系仍然是一个重要课题，但潜在混杂因子的存在使这一任务变得极具挑战性。忽略这些混杂因子可能导致虚假关联和错误的边方向。本文研究了带潜在混杂因子的线性结构方程模型。我们提出了一种算法，该算法迭代地识别终端（观测）节点，并重建观测变量的有向无环图。为此，我们将观测变量的精度矩阵恢复为稀疏加低秩矩阵的形式：稀疏矩阵刻画观测变量之间的条件依赖关系，而低秩矩阵刻画少量潜在混杂因子的综合影响。我们证明，对于p个观测变量、r个潜在混杂因子和s条边，我们的方法能够正确地识别出因果图结构。

    arXiv:2609.18535v1 Announce Type: new  Abstract: Causal discovery aims to recover causal relationships from observed data. In various fields, exploring causal relationships among variables remains an important topic, but this task becomes challenging due to the existence of latent confounders. Ignoring such confounders can lead to false associations and incorrect edge directions. In this paper, we study the linear structural equation model with latent confounders. We propose an algorithm that iteratively identifies terminal (observed) nodes and reconstructs the directed acyclic graph of the observed variables. To do this, we recover the precision matrix of the observed variables as a sparse plus low-rank matrix: a sparse matrix captures the conditional dependencies among observed variables, while a low-rank matrix captures the combined influence of a few latent confounders. We establish that for $p$ observed variables, $r$ latent confounders and $s$ edges, our procedure correctly ident
    
[^9]: 空间自适应噪声注入

    Spatially Adaptive Noise Injection

    [https://arxiv.org/abs/2609.18466](https://arxiv.org/abs/2609.18466)

    本文提出空间自适应噪声注入（SANI）采样框架，通过概率门控机制和空间自适应方差在逐像素层面动态调整噪声注入，在去噪器不确定的边缘纹理区域施加随机校正，而在分数估计精确的平滑区域保持确定性更新。

    

    扩散采样器通过随机（DDPM）或确定性（DDIM）更新来逆转学习到的加噪过程，这两者代表了由一个标量噪声注入方差所控制的单一族的两个端点，且该方差在每个空间位置上以相同方式应用。这种统一的方法忽略了自然图像的几何特性：边缘和纹理等高曲率区域——去噪器在这些区域不确定性较高——能够受益于随机校正，而分数估计精确的平滑区域则会因注入噪声而退化。本工作研究了在给定时间步长上是否每个像素都需要随机校正，并提出了一种新颖的采样框架——空间自适应噪声注入（SANI），它可以在逐像素的基础上动态调整噪声的应用。SANI 将概率门控机制与推导得出的空间自适应方差相结合，确保仅在需要的位置注入噪声，从而细化复杂的图像特征。

    arXiv:2609.18466v1 Announce Type: new  Abstract: Diffusion samplers reverse a learned noising process using either stochastic (DDPM) or deterministic (DDIM) updates, which represent endpoints of a single family controlled by a scalar noise-injection variance that is applied identically at every spatial location. This uniform approach neglects the geometry of natural images: high-curvature regions such as edges and textures, where the denoiser is uncertain, benefit from stochastic correction, whereas smooth regions, where the score is precise, are degraded by injected noise. This work investigates whether each pixel requires stochastic correction at a given timestep and introduces Spatially Adaptive Noise Injection (SANI), a novel sampling framework that dynamically adjusts noise application on a per-pixel basis. SANI integrates a probabilistic gating mechanism with a derived spatially adaptive variance, ensuring that noise is injected precisely where needed to refine complex features w
    
[^10]: 基于记忆持久性的随机子空间梯度下降

    Gradient Descent with Stochastic Subspaces via Persistence of Memory

    [https://arxiv.org/abs/2609.18416](https://arxiv.org/abs/2609.18416)

    本文提出“记忆持久性”技术，利用一个与梯度弱相关、可长期固定无需频繁更新的指导向量来引导随机子空间的生成，从而显著扩展并改进了大规模优化中的随机子空间梯度下降方法，且该向量可借助稀疏性或小批量等结构化特性以低成本高效获得。

    

    随机子空间方法作为基于梯度下降的技术，在大规模优化问题中日益流行，尤其是在分布式环境中。本文引入“记忆持久性”技术，以极大地扩展和改进随机子空间方法。为此，我们利用一个与梯度仅弱相关的向量，为随机子空间的生成过程提供指导结构，而下降过程将沿着该随机子空间进行。该指导向量可以在大量迭代中保持固定，仅在较宽的间隔处刷新（我们可以根据问题参数对该间隔的大小提供理论保证）。在重要的机器学习场景中，例如涉及稀疏性或小批量结构的优化问题，我们表明可以利用结构化属性，以有效且计算成本低廉的方式获得该指导向量。

    arXiv:2609.18416v1 Announce Type: cross  Abstract: Stochastic subspace methods have gained popularity as gradient descent based techniques for large scale optimisation problems, especially in distributed settings. In this paper, we introduce the technique of "persistence of memory" to greatly extend and improve the random subspace methods. To this end, we leverage a vector that is only weakly correlated with the gradient in order to provide a guiding structure to the generative process of the random subspace along which the descent is going to take place. This guidance vector may be fixed for a large number of iterations, only to be refreshed at wide intervals (on whose size we can provide guarantees in terms of problem parameters). In important machine learning settings, such as optimisation problems embodying sparsity or a minibatch structure, we show that the guidance vector can be obtained in an effective and computationally inexpensive manner by leveraging the structured propertie
    
[^11]: 坏天才：超越任务特定捷径的反事实引导测试框架演化

    Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts

    [https://arxiv.org/abs/2609.18366](https://arxiv.org/abs/2609.18366)

    提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。

    

    可靠的智能体评估因自动测试框架优化而变得复杂，这类优化方法反复使用已发布的基准 $B_{\mathrm{rel}}$ 来引导一个提议者，该提议者围绕固定的目标智能体编辑提示词、记忆、检索、工具和控制代码。任务保留集虽然改变了语义任务，但基准协议保持不变，因此一个“坏天才”提议者可以生成一个作弊的测试框架，其在发布基准上的性能提升依赖于整个基准范围的捷径。我们提出了反事实测试框架搜索与演化，将测试框架演化建模为在保持有效性的基准反事实上的约束生成问题。在每次提议者更新后，一个挑战者会搜索能大幅摧毁性能提升的可执行协议变换。有效性防火墙检查任务语义是否得到保留，而确认集则决定反事实是否进入有限存档。我们形式化定义了一个精确的捷径中和基准 $B

    arXiv:2609.18366v1 Announce Type: new  Abstract: Reliable agent evaluation is complicated by automatic harness optimization, which repeatedly uses a released benchmark $B_{\mathrm{rel}}$ to guide a Proposer that edits prompts, memory, retrieval, tools, and control code around a fixed target agent. Task holdout varies semantic tasks but leaves the benchmark protocol fixed, so a "bad genius" Proposer can produce a cheating harness whose released-benchmark gain depends on a benchmark-wide shortcut. We introduce Counterfactual Harness Search and Evolution (CHASE), which casts harness evolution as constraint generation over validity-preserving benchmark counterfactuals. After each Proposer update, a Challenger searches for an executable protocol transformation with large gain destruction. A validity firewall checks that task semantics are preserved, while a confirmation set determines whether the counterfactual enters a finite archive. We formalize an exact shortcut-neutralized benchmark $B
    
[^12]: 超越二次损失：Adam优化器的稳定性相图

    Beyond Quadratic Loss: The Stability Phase Diagram of Adam

    [https://arxiv.org/abs/2609.18314](https://arxiv.org/abs/2609.18314)

    该研究通过绘制Adam优化器在$(\beta_1,\beta_2)$参数平面上的稳定性相图，发现一条近似线性边界$1-\beta_2=C(1-\beta_1)$可用于区分训练中是否出现损失尖峰，并揭示超二次损失景观（如高置信交叉熵损失形成的“核心-墙壁”结构）是决定该边界形状的关键因素。

    

    损失尖峰是神经网络训练中反复出现的不稳定性现象，可能由多种机制引起。特别是对于Adam优化器，宏观损失尖峰已被认为与优化器动力学相关，但其两个动量时间尺度如何支配这些尖峰仍不清楚。我们通过在$(\beta_1,\beta_2)$平面上绘制训练动力学图谱来研究这种依赖关系。在多种模型-任务设置中，一条近似线性的边界$1-\beta_2=C(1-\beta_1)$将出现尖峰与不出现尖峰的动力学区域分隔开来，而一维二次损失则产生近似三次方斜率的边界。一维超二次损失$L(x)\propto|x|^n$则恢复了近线性标度关系，并将边界系数与有效损失指数$n$联系起来。我们进一步表明，高置信度的交叉熵损失会发展出一种“核心-墙壁”景观，由狭窄的二次核心和随后陡峭的墙壁组成，这在优化器步长的尺度上产生有效的超二次行为。

    arXiv:2609.18314v1 Announce Type: new  Abstract: Loss spikes are recurrent instabilities in neural-network training and can arise from multiple mechanisms. For Adam in particular, macroscopic loss spikes have been linked to optimizer dynamics, yet how its two momentum timescales govern them remains unclear. We investigate this dependence by mapping training dynamics across the $(\beta_1,\beta_2)$ plane. Across a range of model--task settings, an approximately linear boundary, $1-\beta_2=C(1-\beta_1)$, separates spiky from non-spiky dynamics, whereas a one-dimensional quadratic loss produces approximately cubic slope. A one-dimensional superquadratic loss $L(x)\propto|x|^n$ recovers the near-linear scaling and links the boundary coefficient to the effective loss exponent $n$. We further show that confident cross-entropy losses develop a core--wall landscape comprising a narrow quadratic core followed by a steep wall, which produces effective superquadratic behavior at the scale of an op
    
[^13]: Wasserstein-Fisher-Rao梯度流的保对数凹性与收敛性

    Preservation of Log-Concavity and Convergence of Wasserstein-Fisher-Rao Gradient Flows

    [https://arxiv.org/abs/2609.18118](https://arxiv.org/abs/2609.18118)

    本文证明Wasserstein-Fisher-Rao梯度流在满足曲率条件的强对数凹目标分布下能够保持强对数凹性，据此推导出对称化KL散度的显式非渐近收敛速率，无需暖启动，且收敛速率可加性分解为Wasserstein与Fisher-Rao两部分贡献。

    

    我们研究了Wasserstein-Fisher-Rao（WFR）梯度流在从仅已知归一化常数的概率分布中进行采样时的收敛性。通过将Wasserstein输运与Fisher-Rao生灭动力学相结合，WFR流平衡了探索与选择，被认为是超越朗之万动力学加速收敛的一种有前景的机制。我们证明，对于一类满足额外曲率条件的强对数凹目标分布，WFR流能够保持强对数凹性；与之相比，Wasserstein流仅在高斯情形下才具备这一性质。利用这一结果，我们推导了对称化Kullback-Leibler散度的显式非渐近收敛速率，且无需当前估计中所要求的暖启动。特别地，我们证明收敛速率可加性地分解为Wasserstein与Fisher-Rao两部分贡献，从而确认了……

    arXiv:2609.18118v1 Announce Type: cross  Abstract: We study the convergence of Wasserstein-Fisher-Rao (WFR) gradient flows for sampling from probability distributions known up to a normalisation constant. By combining Wasserstein transport with Fisher-Rao birth-death dynamics, WFR flows balance exploration and selection. These flows have been recognised as a promising mechanism to accelerate convergence beyond Langevin dynamics. We show that for a class of strongly log-concave target distributions satisfying additional curvature conditions, WFR flows preserve strong log-concavity, in contrast to Wasserstein flows which enjoy this property only in the Gaussian setting. Exploiting this result, we derive explicit non-asymptotic convergence rates for the symmetrised Kullback-Leibler divergence, without requiring a warm-start as required in current estimates. In particular, we show that the convergence rate decomposes additively into Wasserstein and Fisher-Rao contributions, thereby confirm
    
[^14]: 单循环匹配多循环复杂度：非凸-凹极小极大优化中的最优优化平稳性与已知最佳博弈平稳性

    Matching Multi-Loop Complexities with a Single Loop: Optimal Optimization Stationarity and Best-Known Game Stationarity in Nonconvex--Concave Minimax Optimization

    [https://arxiv.org/abs/2609.17973](https://arxiv.org/abs/2609.17973)

    该论文提出了一种结合投影外梯度更新、对偶动量和移动近端中心的单循环投影阻尼外梯度算法，在非凸-凹极小极大优化中以单循环方法匹配了多循环方法的复杂度，同时实现了最优的优化平稳性保证和已知最佳的博弈平稳性保证。

    

    我们为光滑非凸-凹极小极大优化问题引入了一个新的单循环算法框架。由此得到的投影阻尼外梯度方法结合了投影外梯度更新、对偶动量和移动近端中心。在优化平稳性和博弈平稳性两种准则下，我们的方法在单循环一阶方法中达到了已知的最佳复杂度。对于优化平稳性，我们的方法达到了 $O(L^2D_Y\bar\Delta_0\varepsilon^{-3})$ 的梯度复杂度，其中 $L$ 是梯度Lipschitz常数，$D_Y$ 是对偶可行集直径的界，$\bar\Delta_0$ 是一个涉及值函数间隙和初始梯度的初始化量。此外，通过引入固定中心预热阶段，复杂度可以改进为 $O(L^2D_Y\Delta_\phi\varepsilon^{-3})$，外加一个可忽略的低阶可加代价，其中 $\Delta_\phi:=\phi(x_0)-\inf_x\phi(x)$。我们进一步

    arXiv:2609.17973v1 Announce Type: cross  Abstract: We introduce a new single-loop algorithmic framework for smooth nonconvex--concave minimax optimization. The resulting projected damped extragradient method combines projected extragradient updates, dual momentum, and a moving proximal center. Under both the optimization-stationarity and game-stationarity criteria, our method achieves the best-known complexity among single-loop first-order methods. For optimization stationarity, our method achieves a gradient complexity of $O(L^2D_Y\bar\Delta_0\varepsilon^{-3})$, where $L$ is the gradient Lipschitz constant, $D_Y$ bounds the diameter of the dual feasible set, and $\bar\Delta_0$ is an initialization quantity involving the value-function gap and the initial gradients. Moreover, by incorporating a fixed-center warm-up phase, the complexity can be improved to $O(L^2D_Y\Delta_\phi\varepsilon^{-3})$, up to an additive lower-order cost, where $\Delta_\phi:=\phi(x_0)-\inf_x\phi(x)$. We further
    
[^15]: 论线性参数模型下混合有序变量与指数族因果有向无环图（DAG）的可辨识性

    On the Identifiability of Mixed Ordinal and Exponential Family Causal DAGs under Linear Parametric Models

    [https://arxiv.org/abs/2609.17942](https://arxiv.org/abs/2609.17942)

    本文证明了在线性参数模型中，只要有序节点至少有三个类别且指数族节点至少有三个支撑点，连接这两类节点的每条边的方向都可仅凭联合分布在任意参数取值下被辨识，并通过反向论证证明了这两个条件的必要性。

    

    本文研究了线性参数模型（LPM）中节点服从有序logit模型或正则单参数指数族时的可辨识性问题。研究结果超越了经典的联立结构方程模型，也超越了节点观测来自同质分布族情形下的已有结论。主要结果证明了：只要有序节点具有至少三个类别，且指数族节点具有至少三个支撑点，则连接有序节点与指数族节点的每条边的方向都可以在任意参数取值下仅凭联合分布被辨识出来，且对充分统计量不作任何限制。反向结果（converse）表明这两个条件都是必要的：三类别条件仅在充分统计量为仿射函数时才起约束作用，而三支撑点条件在规范链接函数下起约束作用。该可辨识性保证还可进一步扩展到对每条此类混合有序边的定向。（注：原文摘要在此处截断，内容不完整）

    arXiv:2609.17942v1 Announce Type: new  Abstract: The problem of identifiability in linear parametric models (LPMs) whose nodes follow either an ordered logit model or a regular one-parameter exponential family is evaluated. The results go beyond classical structural equation models as well as results for nodes with observations from a homogeneous family of distributions. The main result establishes that the orientation of every edge joining an ordinal node to an exponential-family node is identifiable from the joint distribution alone at every parameter value, provided the ordinal node has at least three categories and the exponential-family node at least three points of support, with no restriction on the sufficient statistic. Converses show that both requirements are necessary: the three-category requirement is binding only for affine sufficient statistics, and the three-point requirement is binding under the canonical link. The guarantee extends to orienting every such mixed ordinal
    
[^16]: 无流形的对称性：轨道上的内在维度

    Symmetry without a manifold: intrinsic dimension on orbits

    [https://arxiv.org/abs/2609.17926](https://arxiv.org/abs/2609.17926)

    该论文证明在对称性轨道（如模加法任务）上标准内在维度估计器普遍失效，神经缩放行为不再遵循幂律，而是遵循关于隐藏层宽度的指数定律 $L(h)=L_\infty+A\exp(-c\,h^{\alpha})$。

    

    神经缩放指数的标准几何推导以数据流形的内在维度作为其输入。对于 $\mathbb{Z}_p$ 上的模加法任务，该推导没有输入可用。其精确的代数解是 $\mathbb{Z}_p$ 通过等距作用产生的轨道。仅凭传递性就使得标准维度估计器所依赖的比率统计量退化为一个点质量，因此该估计器是未定义的，且在此情形下两个最近邻距离恰好完全重合。在尺度 $\epsilon$ 上破坏对称性虽然能返回一个数值，但该数值随 $1/\epsilon$ 变化，不存在无标度平台。我们证明这种失效是普遍性的：在任何由群通过等距作用产生的有限轨道上，估计器报告的只是探测该集合的分辨率，而非维度。取代幂律的是关于隐藏层宽度的指数关系，$L(h)=L_\infty+A\exp(-c\,h^{\alpha})$，其 $R^2$ 达到 0.982 至 0.995，而幂律拟合的 $R^2$ 仅为 0.857 至 0.906。

    arXiv:2609.17926v1 Announce Type: new  Abstract: The standard geometric derivation of neural scaling exponents takes the intrinsic dimension of a data manifold as its input. On modular addition in $\mathbb{Z}_p$ that derivation has no input. The exact algebraic solution is an orbit of $\mathbb{Z}_p$ acting by isometries. Transitivity alone makes the ratio statistic underlying the standard dimension estimator a point mass, so the estimator is undefined, and here the two nearest neighbour distances coincide exactly. Breaking the symmetry at scale $\epsilon$ returns a number, but one that tracks $1/\epsilon$ with no scale free plateau. We show that the failure is general, since on any finite orbit of a group acting by isometries the estimator reports the resolution at which the set is probed rather than a dimension. What replaces the power law is exponential in hidden width, $L(h)=L_\infty+A\exp(-c\,h^{\alpha})$, with $R^2$ between 0.982 and 0.995 against 0.857 to 0.906 for a power law ad
    
[^17]: 广义DCCQ：从二元商到多项单纯形几何与临界带坐标

    Generalized DCCQ: From Binary Quotients to Multinomial Simplex Geometry and Critical-Strip Coordinates

    [https://arxiv.org/abs/2609.17899](https://arxiv.org/abs/2609.17899)

    该论文将离散复补商（DCCQ）框架从二元伯努利计数推广到多项计数组合，证明了m≥2时多项DCCQ坐标映射是概率单纯形上的实解析微分同胚，其中二元情形给出临界线坐标、三元情形覆盖完整临界带，但明确不声称证明黎曼假设。

    

    我们将离散复补商（DCCQ）框架从二元伯努利计数扩展到多项计数组合。对于m+1个类别，m是独立概率自由度的数目。整数计数向量在模去公共缩放因子后确定了m维概率单纯形的有理点。基于标准单纯形与对数比坐标几何，对于m≥2，我们定义了完整的多项DCCQ坐标映射，并证明它是一个实解析微分同胚。此前建立的二元基线（m=1）给出临界线坐标，而三元情形（m=2）给出完整的开放临界带；更高维的多项模型则保留m-2个额外的实对比。我们还给出了一对其余（one-versus-rest）特化形式、三元坐标的精确整数格实现，以及其对数比的双曲表示。本文不声称任何零点定位定理，也不声称证明了黎曼假设。

    arXiv:2609.17899v1 Announce Type: new  Abstract: We extend the discrete complex complement quotient (DCCQ) framework from binary Bernoulli counts to multinomial count compositions. For m+1 categories, m is the number of independent probability degrees of freedom. Integer count vectors modulo common scaling determine rational points of the m-dimensional probability simplex. Building on standard simplex and log-ratio coordinate geometry, for m >= 2 we define the full multinomial DCCQ coordinate map and show that it is a real-analytic diffeomorphism The previously established binary baseline m=1 gives a critical-line coordinate, while the ternary case m=2 gives the full open critical strip; higher multinomial models retain m-2 additional real contrasts. We also give a one-versus-rest specialization, an exact integer-lattice realization of the ternary coordinate, and a hyperbolic representation of its log-ratio. No zero-location theorem or proof of the Riemann Hypothesis is claimed.
    
[^18]: TabPFN-3.5：技术报告

    TabPFN-3.5: Technical Report

    [https://arxiv.org/abs/2609.17895](https://arxiv.org/abs/2609.17895)

    TabPFN-3.5 是一款新的旗舰表格基础模型，在标准及非独立同分布、多模态、高基数、宽表等实际表格任务上全面超越 TabPFN-3 和现有基线，并提供了速度提升最高 3 倍的 TabPFN-3.5-Fast 和增强多模态能力的 TabPFN-3.5-Plus 变体。

    

    我们推出 TabPFN-3.5，这是我们的全新旗舰表格基础模型。它在广泛的表格任务上显著超越了其前代模型 TabPFN-3 以及所有现有基线。TabPFN-3.5 在 TabArena 的标准表格预测任务上创造了新的最先进水平，并将其扩展到实际从业者会遇到的数据场景：具有时间或分组划分的非独立同分布数据、包含字符串、文本和图像的表格、高基数类别特征，以及具有众多特征的宽表。这些优势延续到我们的任务专用框架中：在关系型数据上达到最先进水平，并具备更强的时间序列预测能力。为了实现更快的推理，我们的变体 TabPFN-3.5-Fast 运行速度最高可达 TabPFN-3 的 3 倍，同时保留了大部分精度提升。此外，我们升级了 TabPFN-3.5-Plus，通过先进的文本和日期处理以及专有推理优化扩展了多模态能力。最后，我们发布了一个新的版本。

    arXiv:2609.17895v1 Announce Type: new  Abstract: We introduce TabPFN-3.5, our new flagship Tabular Foundation Model. It significantly outperforms its predecessor, TabPFN-3, and all existing baselines across a broad range of tabular problems. TabPFN-3.5 sets a new state of the art on standard tabular prediction in TabArena, and extends it to the data practitioners encounter in practice: non-i.i.d. data with temporal or grouped splits, tables with strings, text and images, high-cardinality categorical features, and wide tables with many features. These gains carry over to our task-specific harnesses: state of the art on relational data and stronger time-series forecasting. For faster inference, our variant TabPFN-3.5-Fast runs up to 3x faster than TabPFN-3 while keeping most of the accuracy gains. In addition, we upgrade TabPFN-3.5-Plus, expanding our multimodal capabilities with advanced text and date handling alongside proprietary inference optimizations. Finally, we release a new vers
    
[^19]: 流形假设下聚类的区间不确定性界定

    Bracketing Uncertainty in Clustering Under the Manifold Hypothesis

    [https://arxiv.org/abs/2609.17892](https://arxiv.org/abs/2609.17892)

    该论文通过结合内在流形几何（体积增长与触及半径）和样本级度量（填充距离与密度），为互k近邻图聚类建立了阈值现象，从而界定了聚类结果存在不确定性的几何区间。

    

    流形假设为聚类提供了一个自然的准则：根据每个点所来自的流形分量对数据进行划分。两个分量是否可分取决于一种几何上的权衡：分量之间的环境空间间隔与采样中的最大间隙之间的对比。在实践中，这种权衡很少被明确评估，导致标准方法即使在数据不支持唯一答案的情况下，也会过度承诺单一的聚类分配。我们通过将内在流形几何（体积增长与触及半径）与样本级度量（填充距离与密度）相结合，形式化了这一权衡，从而为互k近邻图建立了一个阈值现象：当偏移-填充比超过一个保守的上阈值时，分量分离得以保持；而低于一个下阈值时，分量会发生融合。这两个阈值之间的间隙定义了一个几何不确定性区域，在该区域中聚类的数量…

    arXiv:2609.17892v1 Announce Type: cross  Abstract: The manifold hypothesis suggests a natural criterion for clustering: partition data according to the manifold component from which each point is drawn. Whether two components are separable depends on a geometric tradeoff: the ambient separation between components versus the largest gap in sampling. In practice, this tradeoff is rarely assessed explicitly, leading standard methods to over-commit to a single clustering assignment even when the data do not support a unique answer. We formalize this tradeoff by combining intrinsic manifold geometry (volume growth and reach) with sample-level quantities (fill distance and density), yielding a threshold phenomenon for mutual-$k$-nearest-neighbor graphs: when the offset-to-fill ratio exceeds a conservative upper threshold, component separation is preserved; below a lower threshold, components fuse. The gap between these thresholds defines a geometric uncertainty zone in which the number of cl
    
[^20]: 可实现情形下支持向量机的尖锐间隔泛化界

    Sharp margin-based generalization bounds for realizable SVM

    [https://arxiv.org/abs/2609.17845](https://arxiv.org/abs/2609.17845)

    该论文通过确定性删除问题的分析，证明了可实现情形下硬间隔支持向量机的泛化风险以概率至少\(1-\delta\)不超过\(\frac{C}{m}(K_m+\log\frac{1}{\delta})\)，其中\(K_m=r_m^2/\gamma_m^2\)为半径-间隔复杂度，得到了阶为\(1/m\)且依赖半径-间隔复杂度的尖锐泛化界。

    

    设精确的齐次硬间隔支持向量机在实希尔伯特空间上，由某个Borel概率分布产生的m个独立观测样本进行训练。我们证明，当得分零被计为错误时，存在一个通用数值常数C，使得 \[ \Pp\left( \gamma_m>0,\quad \Risk(u_m)> \frac{C}{m} \left( K_m+\log\frac1\delta \right) \right) \le \delta. \] 其中\(\gamma_m\)是经验齐次间隔，\(u_m\)是精确的最小范数单位间隔分离器，\(r_m\)是最大训练半径，且在\(\{\gamma_m>0\}\)上\(K_m:=r_m^2\norm{u_m}^2=r_m^2/\gamma_m^2\)。证明由一个确定性的删除问题驱动：给定单位球中的向量\(x_1,\ldots,x_n\)，删除一个约束集合\(B\)，令\(u_B\)为满足所有保留单位间隔约束的、离原点最近的点。假设\(\norm{u_B}^2\le k\)，且每个被删除的向量都具有非正……（摘要在此截断）

    arXiv:2609.17845v1 Announce Type: cross  Abstract: Let the exact homogeneous hard-margin support vector machine be trained on \(m\) independent observations from a Borel probability law on a real Hilbert space. We prove that, with score zero counted as an error, there is a universal numerical constant \(C\) such that \[   \Pp\left(   \gamma_m>0,\quad   \Risk(u_m)>   \frac{C}{m}   \left(   K_m+\log\frac1\delta   \right)   \right)   \le \delta . \] Here \(\gamma_m\) is the empirical homogeneous margin, \(u_m\) is the exact minimum-norm unit-margin separator, \(r_m\) is the largest training radius, and \(K_m:=r_m^2\norm{u_m}^2=r_m^2/\gamma_m^2\) on \(\{\gamma_m>0\}\).   The proof is driven by a deterministic deletion problem. Given vectors \(x_1,\ldots,x_n\) in the unit ball, delete a set \(B\) of constraints and let \(u_B\) be the closest point to the origin that satisfies every retained unit-margin constraint. Suppose that \(\norm{u_B}^2\le k\) and that every deleted vector has nonposit
    
[^21]: METALICA：元动力学与副本交换实现增强扩散采样

    METALICA: METAdynamics and repLICA exchange for enhanced diffusion sampling

    [https://arxiv.org/abs/2609.17823](https://arxiv.org/abs/2609.17823)

    METALICA通过副本交换机制在预训练扩散模型上实现元动力学，利用偏置势采样和重加权高效探索蛋白质构象的稀有状态，从而实现对稀有事件的有效发现。

    

    许多蛋白质通过构象状态之间的转换来发挥功能，然而基于平衡系综训练的扩散模型很少能采样到稀有状态，因此需要更好的采样方法。我们提出了METALICA，该方法通过副本交换在预训练扩散模型上实现元动力学。它沿着集体变量累积偏置势，通过偏置采样使新样本远离已有样本，并将样本重新加权到无偏分布上。METALICA在每个扩散层级保持一个副本，形成一条通过副本间通信演化的马尔可夫链，并随着偏势的增长而就地优化。METALICA是序贯控制的对偶形式——后者由序贯蒙特卡洛在一批粒子上并行化采样器；而METALICA则是在扩散时间调度的各层级上进行并行，这使其能够从长链中生成样本，这对稀有事件的发现至关重要，并具有加速效果。

    arXiv:2609.17823v1 Announce Type: cross  Abstract: Many proteins function through transitions between conformational states, yet rare states are rarely sampled by diffusion models trained on an equilibrium ensemble, demanding better sampling methods. We introduce METALICA, which implements Metadynamics on a pretrained diffusion model via Replica Exchange. It accumulates a bias potential along a Collective Variable, repels new samples from previous ones through biased sampling, and reweights samples onto the unbiased distribution. METALICA holds one replica per diffusion level, forming a Markov Chain that evolves through inter-replica communication and is refined in place as the bias grows. METALICA is the dual of sequential control, in which Sequential Monte Carlo parallelizes the sampler over a batch of particles. Parallelism over the levels of the diffusion-time schedule instead allows METALICA to generate samples from long chains, essential for the discovery of rare events, with acc
    
[^22]: 函数空间上测度的逼近：传输与截断

    Approximating Measures on Function Spaces: Transport and Truncation

    [https://arxiv.org/abs/2609.17802](https://arxiv.org/abs/2609.17802)

    该论文提出了一类与易处理参考测度仅相差有限维映射的函数空间测度，通过分块三角传输映射对低维推前分布进行采样，再利用参考条件分布将其补全到函数空间，从而实现结构保持的高效函数空间测度逼近与采样。

    

    函数空间上的测度广泛出现在贝叶斯逆问题和生成建模中，通常相对于一个易处理的参考测度具有低维结构。我们引入了一类测度 $\mathcal{P}_\psi(\mu)$，这类测度与参考测度 $\mu$ 仅通过一个有限维映射 $\psi$ 而有所不同，同时在 $\psi$ 的纤维上保持参考测度的条件分布。该类中的成员由其在 $\psi$ 下的 $d$ 维推前分布完全确定，并具有便利的分块三角传输映射表示。样本首先从这个 $d$ 维分布中抽取，然后通过参考条件分布的采样补全到函数空间。对于高斯参考测度，这些传输映射是恒等映射的有限秩扰动。相比之下，最优传输映射则无法保持这种低维结构。对于在参考测度的 Cameron-Martin 几何下为迹类的协方差扰动，我们……

    arXiv:2609.17802v1 Announce Type: cross  Abstract: Measures on function spaces arise throughout Bayesian inverse problems and generative modeling, often with low-dimensional structure relative to a tractable reference measure. We introduce the class $\mathcal{P}_\psi(\mu)$ of measures that differ from a reference measure $\mu$ only through a finite-dimensional map $\psi$ while preserving the reference conditionals on its fibers. Class members are determined by their $d$-dimensional pushforwards under $\psi$ and admit convenient block-triangular transport map representations. Draws are taken from this $d$-dimensional distribution and then completed to function space through sampling of the reference conditionals. For Gaussian references, these transport maps are finite rank perturbations of the identity. In contrast, optimal transport maps do not preserve this low-dimensional structure. For covariance perturbations that are trace class in the Cameron-Martin geometry of the reference, we
    
[^23]: 随机倾斜法求解随机凸优化中的驻点

    Random tilts to find stationary points in stochastic convex optimization

    [https://arxiv.org/abs/2609.17798](https://arxiv.org/abs/2609.17798)

    该论文证明正则化经验风险最小化结合随机倾斜扰动能以 $\sqrt{d/n}$ 的残差阶找到随机凸函数及相关变分不等式的驻点，并通过 $\sqrt{\log d/n}$ 量级的极小极大下界表明一定的维度依赖性是不可避免的。

    

    我们研究寻找随机凸函数及相关变分不等式驻点的问题。对于这些问题，我们证明了正则化经验风险最小化结合随机倾斜扰动，在给定 n 个观测值的 d 维问题中，可以获得量级为 $\sqrt{d/n}$ 的驻点残差。我们还给出了一些补充结果，通过提供随 $\sqrt{\log d / n}$ 缩放的极小极大下界，表明与标准随机优化和经验风险最小化不同，一定的维度依赖性是必要的。

    arXiv:2609.17798v1 Announce Type: cross  Abstract: We consider the problem of finding stationary points of stochastic convex functions and related variational inequalities. For each, we show that regularized empirical risk minimization, coupled with a random tilting perturbation, obtains stationarity residual order $\sqrt{d/n}$ for $d$-dimensional problems given $n$ observations. We present a few complementary results that show that some dimension dependence is necessary, in distinction from standard stochastic optimization and empirical risk minimization, by providing minimax lower bounds scaling as $\sqrt{\log d / n}$.
    
[^24]: 信息论极限下的高效鲁棒学习

    Efficient Robust Learning at the Information-Theoretic Limit

    [https://arxiv.org/abs/2609.17655](https://arxiv.org/abs/2609.17655)

    本文解决了 Blanc 遗留的开放问题，通过巧妙运用无悔学习器技术，首次给出了在 ERM 预言机辅助下达到信息论最优错误率 η+ε 的多项式时间鲁棒学习算法，并为具有三明治多项式性质的函数类提供了无需预言机的高效算法。

    

    在近期一项重要工作中，Blanc（2026）给出了一种针对固定分布鲁棒学习布尔概念类的算法，该算法输出一个（随机化的）分类器，能够达到 η + ε 的最优错误率，其中 η 为噪声率。相比之下，众所周知，确定性假设无法达到低于 2η + ε 的错误率。Blanc 的算法在计算上效率低下，其工作中遗留的主要开放问题是：在可访问经验风险最小化（ERM）预言机的条件下，找到一种多项式时间算法。在本文中，我们解决了这一问题，并给出了这样一种算法。令人惊讶的是，我们的技术关键地利用了各种类型的无悔学习器。此外，我们给出了一种高效算法（无需 ERM 预言机），用于鲁棒学习任何在超收缩分布下允许三明治多项式的函数类。作为其中一个结果……

    arXiv:2609.17655v1 Announce Type: cross  Abstract: In an important recent work, Blanc (2026) gave an algorithm for robustly learning Boolean concept classes with respect to a fixed distribution that outputs a (randomized) classifier achieving the optimal error of $\eta + \varepsilon$ where $\eta$ is the noise rate. In contrast, it is well known that deterministic hypotheses cannot achieve error less than $2\eta + \varepsilon.$   Blanc's algorithm is computationally inefficient, and the main problem left open in his work is to find a polynomial-time algorithm given access to an oracle for empirical risk minimization (ERM). In this paper, we resolve this problem and give such an algorithm. Perhaps surprisingly, our techniques make crucial use of various types of no-regret learners.   Additionally, we give an efficient algorithm (no ERM oracle required) for robustly learning any function class that admits sandwiching polynomials with respect to hypercontractive distributions. As one conse
    
[^25]: Fenchel-Young对偶间隙：正则化逆问题的可认证早停方法

    Fenchel-Young Duality Gaps: Certified Early Stopping for Regularized Inverse Problems

    [https://arxiv.org/abs/2609.17629](https://arxiv.org/abs/2609.17629)

    本文提出了一个精确的对偶间隙恒等式，将正则化逆问题的总间隙分解为数据保真与正则项两个Fenchel-Young损失，从而给出可计算、无需oracle的误差界，实现了可认证的早停。

    

    我们研究正则化逆问题的可计算误差界与可认证早停方法，此类问题涉及数据保真项与正则项之间的权衡。分析依赖于一个精确的对偶间隙恒等式，它将 $F(\Phi\mu)+\lambda R(\mu)$ 的总间隙分解为数据保真Fenchel-Young损失与正则项Fenchel-Young损失两部分：$\Delta(\mu,h)=L_F(\Phi\mu\parallel h)+\lambda L_R(\mu\parallel\eta),\qquad \eta=-\Phi^\star h/\lambda$，该恒等式对任意原始点 $\mu$ 和任意对偶点 $h$ 均成立。数据保真项 $F$ 是严格凸的，因此在 $F^\star$ 可微之处，损失 $L_F(\Phi\mu\parallel h)$ 即为 $F$ 在预测值 $\Phi\mu$ 与 $\nabla F^\star(h)$ 之间的Bregman散度，且恰好在“镜像对齐” $h=\nabla F(\Phi\mu)$ 处消失。在某个对偶可行点 $\tilde h$ 处求值时，间隙 $\Delta(\mu,\tilde h)$ 是可计算的且“无需oracle”，也就是说它不需要任何关于……

    arXiv:2609.17629v1 Announce Type: cross  Abstract: We study computable error bounds and certified early stopping for regularized inverse problems, where a data-fidelity term is traded against a regularizer. The analysis relies on an exact duality-gap identity that splits the total gap of $F(\Phi\mu)+\lambda R(\mu)$ into a data-fidelity Fenchel--Young loss and a regularizer Fenchel--Young loss, $ \Delta(\mu,h)=L_F(\Phi\mu\parallel h)+\lambda L_R(\mu\parallel\eta),\qquad \eta=-\Phi^\star h/\lambda, $ valid for any primal point $\mu$ and any dual point $h$. The data-fidelity term $F$ is strictly convex, so wherever $F^\star$ is differentiable the loss $L_F(\Phi\mu\parallel h)$ is the Bregman divergence of~$F$ between the prediction $\Phi\mu$ and $\nabla F^\star(h)$, and it vanishes exactly at \emph{Mirror Alignment} $h=\nabla F(\Phi\mu)$. Evaluated at a dual-feasible point $\tilde h$, the gap~$\Delta(\mu,\tilde h)$ is computable and \emph{oracle-free}, meaning that it uses no knowledge of
    
[^26]: 样条KAN中的稳定性约束逼近：精确层平衡与预算兼容饱和

    Stability-Constrained Approximation in Spline KANs: Exact Layer Balancing and Budget-Compatible Saturation

    [https://arxiv.org/abs/2609.17619](https://arxiv.org/abs/2609.17619)

    该论文在严格逐层Lipschitz预算约束下研究深度样条KAN的逼近理论，精确求解了有限深度对角层平衡问题（给出最优层预算的闭式解与单遍最小化算法），并提出了保持预算约束的构造性样条离散化定理。

    

    深度样条叠加网络在逼近阶与跨深度稳定性之间存在固有的张力。我们研究在严格逐层Lipschitz预算约束下的逼近问题，并围绕两个量来组织这一研究：给定深度分解的分解稳定性复杂度，以及离散化算子的预算兼容逼近复杂度。首先，我们精确求解了固定非负包络矩阵链的有限深度对角平衡问题：最优的均匀层预算等于 $\|M_{L-1}\cdots M_0\|_{\infty\to\infty}^{1/L}$，对于矩形层可由一个显式的单遍最小化器达到，并对退化与非可达情形给出了完整的处理。该最优值可以任意大于网络本身的Lipschitz常数，因为转换为包络会破坏符号相消效应。其次，我们给出了一个构造性的样条离散化定理，在可控的精度下保持该预算约束。

    arXiv:2609.17619v1 Announce Type: cross  Abstract: Deep spline superposition networks face a tension between approximation order and stability across depth. We study approximation under a hard layerwise Lipschitz budget, and organise it around two quantities: the factorisation stability complexity of a given deep factorisation, and the budget-compatible approximation complexity of a discretisation operator.   First, we solve exactly the finite-depth diagonal balancing problem for a fixed chain of nonnegative envelope matrices: the optimal uniform layer budget equals $\|M_{L-1}\cdots M_0\|_{\infty\to\infty}^{1/L}$, attained by an explicit one-pass minimiser, for rectangular layers, with a complete treatment of degeneracies and non-attainment. The optimum can be arbitrarily larger than the Lipschitz constant of the network itself, because passing to envelopes destroys sign cancellation.   Second, we give a constructive spline discretisation theorem preserving the budget up to a controlle
    
[^27]: 具有移动目标的朗之万动力学的Rényi追踪界

    R\'enyi Tracking Bounds for Langevin Dynamics with Moving Targets

    [https://arxiv.org/abs/2609.17577](https://arxiv.org/abs/2609.17577)

    该论文首次建立了具有离散目标更新的朗之万动力学的非渐近Rényi散度追踪界，并将其应用于基于连续Moreau包络的非光滑采样，给出了显式的参数选择和复杂度保证。

    

    我们研究了当目标分布随时间变化时的朗之万扩散和朗之万蒙特卡洛（LMC）。在对数Sobolev不等式（LSI）的条件下，我们推导出了用于追踪当前目标的非渐近Rényi散度保证。该框架同时涵盖连续时间朗之万扩散及其离散化形式。随后，我们将这些结果应用于基于连续Moreau包络的非光滑采样方法。针对该方案，我们给出了平滑参数和步长的显式选择，以及相应的复杂度界。据我们所知，这些是首个针对具有离散目标更新的朗之万动力学的非渐近Rényi散度追踪界。

    arXiv:2609.17577v1 Announce Type: cross  Abstract: We study Langevin diffusion and Langevin Monte Carlo (LMC) when the target distribution changes over time. Under a log-Sobolev inequality (LSI), we derive non-asymptotic R\'enyi-divergence guarantees for tracking the current target. The framework covers continuous-time Langevin diffusion and its discretizations. We then apply the results to nonsmooth sampling based on successive Moreau envelopes. For this scheme, we give explicit choices of the smoothing parameters and step sizes, together with corresponding complexity bounds. To our knowledge, these are the first non-asymptotic R\'enyi-divergence tracking bounds for Langevin dynamics with discrete target updates.
    
[^28]: 仅为分歧付费：具有匹配标签复杂度界的模型更新无回归认证判定

    Pay Only for Disagreement: Certified No-Regression Verdicts for Model Updates with Matching Label-Complexity Bounds

    [https://arxiv.org/abs/2609.17560](https://arxiv.org/abs/2609.17560)

    论文提出DISCERN协议，利用“模型间风险差仅存在于分歧输入上且无需标签即可观测”这一关键性质，通过零标签层与仅标注分歧样本的审计层两层序贯协议，为模型更新提供无回归认证，并证明标签复杂度为rho²/ε²，相比不考虑配对关系的审计器可节省1/ρ的标注成本。

    

    每个生产模型都会被更新——通过重新训练、微调、量化或静默的供应商替换——而每次更新都存在比其替代模型表现更差的风险。我们将模型更新晋升问题形式化为认证的成对风险差审计。我们的出发点是一个支撑恒等式：两个模型之间的风险差存在于它们产生分歧的输入上，而这些输入无需标签即可观测。我们构建了DISCERN，一个序贯两层协议。零标签层仅通过无标签流量即可认证分歧率低于容差的良性更新。审计层通过anytime-valid置信序列仅对采样出的分歧样本进行标注，该序列在每个停止时刻以及任何标签路由规则下均有效，即使是对抗性的评判者也不例外。我们证明了有限样本有效性以及匹配的标签复杂度界，在速率层面为rho^2/eps^2量级，因此利用免费的分歧信息可证明地比任何不考虑配对关系的审计器节省1/rho的因子。

    arXiv:2609.17560v1 Announce Type: cross  Abstract: Every production model is updated, by retraining, fine-tuning, quantization, or a silent vendor swap, and each update risks being worse than what it replaced. We formalize update promotion as certified paired risk-difference auditing. Our starting point is a support identity: the risk difference between two models lives on the inputs where they disagree, observable without labels. We build DISCERN, a sequential two-tier protocol. A zero-label tier certifies benign updates whose disagreement rate is below tolerance from unlabeled traffic alone. An audited tier labels only sampled disagreements through an anytime-valid confidence sequence, valid at every stopping time and under any label-routing rule, even an adversarial judge. We prove finite-sample validity and matching label-complexity bounds of order rho^2/eps^2 at the rate level, so exploiting free disagreement provably saves a factor 1/rho over any pairing-blind auditor, and the gu
    
[^29]: 从集体稳态中学习相互作用核

    Learning Interaction Kernels from Collective Steady States

    [https://arxiv.org/abs/2609.12004](https://arxiv.org/abs/2609.12004)

    该论文提出了一种仅需从集体稳态的单快照观测中学习相互作用粒子系统相互作用核的方法，通过基于观测构型经验分布的正则化策略解决了本质上不适定的逆问题，实现了对相互作用规律的稳定准确恢复以及对集体行为乃至其动力学过程的忠实重现。

    

    我们提出了一种从集体行为的单快照观测中对相互作用粒子系统进行系统辨识的学习方法，这与依赖轨迹观测的现有方法不同。这一设定导致了一个本质上不适定的逆问题，我们通过一种基于观测构型经验分布的正则化策略来解决该问题，这些构型来自不同的、未被观测到的初始条件。我们在多种具有稳态和准稳态模式的代表性模型上测试了该学习程序，在这些模型中，集体行为编码了关于相互作用机制的隐含信息。结果表明，我们的方法能够稳定且准确地恢复潜在的相互作用规律，从而忠实地重现集体行为，在许多情况下甚至能够重现导致该集体行为的动力学过程。

    arXiv:2609.12004v1 Announce Type: cross  Abstract: We propose a learning procedure for system identification in interacting particle systems from single-snapshot observations of collective behaviors, unlike existing approaches that rely on observations of trajectories. This setting leads to a fundamentally ill-posed inverse problem, which we solve by using a regularization strategy based on the empirical distribution of observed configurations, drawn from different, unobserved initial conditions. We test our learning procedure on a variety of representative models with steady-state and quasi-stationary patterns, where collective behaviors encode implicit information about the interaction mechanisms, demonstrating that our approach enables stable and accurate recovery of the underlying interaction laws, leading to faithful reproduction of the collective behavior, and in many cases even of the dynamics leading up to it.
    
[^30]: 条件Shapley特征重要性的半参数推断

    Semiparametric Inference for Conditional Shapley Feature Importance

    [https://arxiv.org/abs/2609.10313](https://arxiv.org/abs/2609.10313)

    本文针对条件Shapley特征重要性提出了一种带K折交叉拟合和U统计量修正的半参数一步估计器，消除了蒙特卡洛偏差，在双重稳健速率条件下实现√n一致性与渐近正态性，并提供覆盖率有保证的Wald置信区间。

    

    Shapley值被广泛用于事后特征归因，但大多数估计器仅返回点估计量而无法量化不确定性，且流行的实现方法从边际分布中采样联盟外特征，这在特征相关时会导致重要性归因错误。本文研究了条件形式化方法，即联盟外特征在其真实条件分布下被积分出去。目标是一个全局的、基于损失的重要性度量，它将条件价值函数与SAGE风格的损失聚合相结合。我们提出了一种结合K折交叉拟合的一步估计器，并对平方损失进行U统计量修正，从而消除了朴素插入估计器的蒙特卡洛偏差；该估计器在双重稳健速率条件下具有√n一致性和渐近正态性，由此构造的Wald置信区间达到了名义覆盖率。此外，还给出了Pinsker型界以量化工作条件分布被误设时产生的偏差。

    arXiv:2609.10313v1 Announce Type: cross  Abstract: Shapley values are widely used for post-hoc feature attribution, but most estimators return point quantities and do not quantify uncertainty, and popular implementations sample out-of-coalition features from their marginal distribution, which misattributes importance when features are dependent. This paper studies the conditional formulation, in which out-of-coalition features are integrated out under their true conditional distribution. The target is a global, loss-based importance that pairs a conditional value function with a SAGE-style loss aggregation. We propose a one-step estimator with K-fold cross-fitting and a U-statistic correction of the squared loss that removes the Monte Carlo bias of the naive plug-in; it is $\sqrt{n}$-consistent and asymptotically normal under double-robust rate conditions, and the resulting Wald interval attains nominal coverage. A Pinsker-type bound quantifies the bias from misspecifying the working c
    
[^31]: 并行高斯过程强盗优化的改进遗憾分析

    Improved Regret Analysis for Parallel Gaussian Process Bandit Optimization

    [https://arxiv.org/abs/2608.16492](https://arxiv.org/abs/2608.16492)

    本文通过GP-BTS示例，证明无需初始不确定性采样阶段即可消除批量大小对遗憾上界的乘性影响，并在无噪声条件下实现更优的遗憾界限。

    

    本文研究了并行高斯过程（GP）强盗优化的遗憾分析。广泛使用的GP批量上置信界和GP批量汤普森采样（GP-BTS）的已知遗憾上界，在批量大小$Q$上存在一个乘性因子。为避免这种性能退化，现有分析需要在优化开始时对$Q$进行多项式数量的不确定性采样（US）。然而，这种初始US阶段在实践中往往效果不佳。本文以GP-BTS为例，表明无需初始US阶段即可实现无$Q$乘性因子的遗憾上界。此外，我们展示了在无噪声设置下，遗憾上界远优于有噪声设置，这与顺序GP强盗设置中的情况一致。

    arXiv:2608.16492v1 Announce Type: cross  Abstract: This paper studies the regret analysis for parallel Gaussian process (GP) bandit optimization. The known regret upper bounds for the widely used GP batched upper confidence bound and GP batched Thompson sampling (GP-BTS) suffer from a multiplicative factor with respect to the batch size $Q$. To avoid this degradation, existing analyses require a polynomial number of uncertainty sampling (US) for $Q$ at the beginning of optimization. However, this initial US phase is often ineffective in practice. This paper shows that the regret upper bound without the multiplicative factor on $Q$ can be achieved without the initial US phase, using GP-BTS as an example. Furthermore, we show much better regret upper bounds in the noiseless setting than in the noisy setting, as in the sequential GP bandit setting.
    
[^32]: Matérn核与平方指数核再生核希尔伯特空间中固定先验期望改进的简单遗憾率与极小极大最优性

    Simple-regret rates and minimax optimality of fixed-prior expected improvement in Mat\'ern and squared-exponential RKHSs

    [https://arxiv.org/abs/2607.29245](https://arxiv.org/abs/2607.29245)

    本文证明了在Matérn核和平方指数核的再生核希尔伯特空间中，弱期望改进策略的简单遗憾率达到极小极大最优，分别以 $O(N^{-\nu/d})$ 和指数级速率收敛。

    

    我们研究期望改进（EI）方法在最小化确定性函数 $f$ 时的表现，其中 $f$ 属于定义在非空紧集 $\mathcal X\subset\mathbb R^d$ 上的连续半正定核 $k$ 所对应的再生核希尔伯特空间 $\mathcal H_k$。函数值被精确观测（无噪声），EI 基于一个具有协方差 $\sigma^2k$（$\sigma>0$）的固定零均值高斯过程模型计算。弱EI策略是指查询期望改进至少为其最大值固定正比例的点的策略。我们借鉴贪婪逼近的思想，引入了顺序分离半径的概念，将排序后的所选点创新范数与Kolmogorov宽度联系起来。利用散乱数据逼近领域的标准幂函数估计和有限预算遗憾论证，我们得到了收敛率。在 $N$ 次初始后查询之后，每个弱EI策略对于光滑度为 $\nu>0$ 的各向同性Matérn核具有简单遗憾 $O(N^{-\nu/d})$，对于各向同性平方指数核具有简单遗憾 $O(\exp[-c_1\min\{N,N^{1/d}\log(eN)\}])$。

    arXiv:2607.29245v2 Announce Type: replace-cross  Abstract: We study expected improvement (EI) for minimizing a deterministic function $f$ in the RKHS $\mathcal H_k$ of a continuous positive-semidefinite kernel $k$ on a nonempty compact set $\mathcal X\subset\mathbb R^d$. Function values are observed exactly, and EI is computed from a fixed zero-mean Gaussian-process model with covariance $\sigma^2k$, $\sigma>0$. A weak-EI policy queries a point whose EI is at least a fixed positive fraction of its maximum.   We introduce a notion of sequential separation radius relating ranked selected-point innovation norms to Kolmogorov widths, drawing on greedy approximation. Standard power-function estimates from scattered-data approximation and a finite-budget regret argument yield the rates. After $N$ post-initial queries, every weak-EI policy has simple regret $O(N^{-\nu/d})$ for isotropic Mat\'ern kernels of smoothness $\nu>0$ and $O(\exp[-c_1\min\{N,N^{1/d}\log(eN)\}])$ for the isotropic squar
    
[^33]: 优化预条件子：一种基于静态遗憾最小化预言机的黑盒在线到非凸转换

    Optimizing the Preconditioner: A Black-box Online-to-Nonconvex Conversion with Static Regret Minimization Oracles

    [https://arxiv.org/abs/2607.17607](https://arxiv.org/abs/2607.17607)

    本文提出了一种从随机非凸优化到在线凸优化中静态遗憾最小化的黑盒归约方法，解决了Chen和Hazan（2024）提出的开放问题，并证明任何具有O(√T)遗憾的OCO预言机都能恢复经典的O(T^{-1/2})收敛速率。

    

    随机非凸优化是现代机器学习中训练深度网络和大语言模型的核心。我们给出了一个从随机非凸优化到在线凸优化（OCO）中普通静态遗憾最小化的黑盒归约，从而解决了Chen和Hazan（2024）提出的开放问题。我们的归约维护一个可预测的梯度跟踪器，同时一个黑盒在线学习器 $\mathcal{A}$ 选择一个预条件子，将该跟踪器转换为更新方向。给定一个值域以 $M$ 为界的 $\beta$-光滑函数和一个方差以 $\sigma^2$ 为界的无偏梯度预言机，我们将期望平均平方梯度范数界定为 $O(\sigma\sqrt{M\beta/T}+\sqrt{M\beta}\mathrm{Reg}_T(\mathcal{A})/T+\frac{M\beta}{T})$，其中 $\mathrm{Reg}_T(\mathcal{A})$ 是 $\mathcal{A}$ 的静态遗憾。因此，任何具有 $O(\sqrt{T})$ 遗憾的OCO预言机都能恢复经典的 $O(T^{-1/2})$ 收敛速率。

    arXiv:2607.17607v3 Announce Type: replace  Abstract: Stochastic nonconvex optimization is central to training deep networks and LLMs in modern machine learning. We give a black-box reduction from stochastic nonconvex optimization to ordinary static regret minimization in online convex optimization (OCO), thereby resolving the open problem posed by Chen and Hazan (2024). Our reduction maintains a predictable gradient tracker, while a black-box online learner $\mathcal{A}$ selects a preconditioner that transforms this tracker into the update direction. Given a \(\beta\)-smooth function with a range bounded by $M$ and an unbiased gradient oracle with variance bounded by $\sigma^2$, we bound the expected average squared gradient norm by $O(\sigma\sqrt{M\beta/T}+\sqrt{M\beta}\mathrm{Reg}_T(\mathcal{A})/T+\frac{M\beta}{T})$, where $\mathrm{Reg}_T(\mathcal{A})$ is the static regret of $\mathcal{A}$. Thus, any OCO oracle with $O(\sqrt{T})$ regret recovers the classical $O(T^{-1/2})$ convergenc
    
[^34]: 主观风险分解：不确定性量化的新视角

    Subjective Risk Decomposition: A New View for Uncertainty Quantification

    [https://arxiv.org/abs/2607.15196](https://arxiv.org/abs/2607.15196)

    该论文提出将不确定性度量视为主观风险分解的产物而非基本原语，证明了基于严格恰当损失对主观风险进行分解即可推导出认知不确定性与偶然不确定性，从而为不确定性量化提供了统一的理论框架和新范式。

    

    我们提出了一种关于不确定性量化的新颖观点。不确定性度量并非需要公理和论证的基本原语，而是更高层次建模决策所产生的结果。我们展示了如何通过基于严格恰当损失的主观风险分解来推导认知不确定性和偶然不确定性度量。反向交叉熵提供了一个突出的例子，其分解能够恢复经典的信息论不确定性项。同样的方法还恢复了不确定性量化文献中此前提出的众多度量，为它们提供了一个共同的理论基础。这启示了一种新的不确定性量化方法：给定建模场景和严格恰当损失，相应的认知与偶然不确定性项便可由主观风险分解诱导产生。随后我们将这一观点扩展至学习理论：我们引入并分析了超额风险、近似误差和（认知）估计误差等概念的主观风险类似物

    arXiv:2607.15196v3 Announce Type: replace-cross  Abstract: We present a novel viewpoint for uncertainty quantification. Uncertainty measures are not primitives, in need of axioms and argumentation, but instead consequences, of higher-level modelling decisions. We show how epistemic and aleatoric uncertainty measures can be derived via decomposition of a subjective risk, based on a strictly proper loss. Reverse cross entropy provides a prominent example, where decomposition recovers the classic information-theoretic uncertainty terms. The same approach recovers numerous measures previously proposed across the UQ literature, providing them a common theoretical foundation. This suggests a new approach to UQ: given a modelling scenario and strictly proper loss, the corresponding epistemic and aleatoric terms are induced by the subjective-risk decomposition. We then extend our view to learning theory: we introduce and analyse subjective risk analogues of excess risk, approximation error and
    
[^35]: 复杂缺失机制下二元回归的共形预测

    Conformal Prediction for Dyadic Regression Under Complex Missingness

    [https://arxiv.org/abs/2606.11136](https://arxiv.org/abs/2606.11136)

    本文提出了一个在复杂缺失机制下用于二元回归的共形预测框架，通过新颖的双射论证和多种程序（如行列方法和选择性共形）实现了有限样本有效性和掩码条件有效性。

    

    arXiv:2606.11136v3 公告类型：替换交叉 摘要：我们开发了一个在复杂缺失机制下用于二元回归问题的共形预测框架。在理论层面，我们建立了通用技术工具，用于在比可交换性更弱的分布不变性条件下证明共形预测的有限样本有效性。一个关键结果处理了样本本身是指标集随机子集的情况，这一场景未被现有理论覆盖，通过一种新颖的双射论证，构造了事件之间显式的保测对应关系。此外，我们提出了针对联合可交换数组的共形预测程序，包括全共形、分裂共形、利用行内和列内相似性的行列方法，以及实现掩码条件有效性的选择性共形程序。对于缺失元素，我们在非参数条件下建立了加权共形程序的渐近有效性。

    arXiv:2606.11136v3 Announce Type: replace-cross  Abstract: We develop a framework for conformal prediction in dyadic regression problems under complex missingness mechanisms. At the theoretical level, we develop general technical tools for establishing finite-sample validity of conformal prediction under distributional invariance conditions weaker than exchangeability. A key result handles the case where the sample itself is a random subset of the index set, a setting not covered by existing theory, via a novel bijection argument that constructs an explicit measure-preserving correspondence between events. In addition, we propose conformal prediction procedures for jointly exchangeable arrays, including full conformal, split conformal, a row-column approach exploiting similarities within rows and columns, and a selective conformal procedure achieving mask-conditional validity. For missing elements, we establish asymptotic validity of a weighted conformal procedure under a nonparametric
    
[^36]: 高维张量时间序列的CP分解与双重投影迭代

    CP-factorization for high dimensional tensor time series and double projection iterations

    [https://arxiv.org/abs/2606.08560](https://arxiv.org/abs/2606.08560)

    本文提出基于CP分解的高维张量时间序列因子载荷估计方法，通过单次特征值分析和新型双重投影迭代算法，在因子相关、载荷非正交等一般条件下建立理论性质并提升收敛速度。

    

    我们采用典型多重分解来建模高维张量时间序列。我们的主要目标是识别和估计CP分解中的因子载荷。我们提出了一种单次估计程序，该方法对基于数据序列依赖结构所构造的矩阵进行标准特征值分析。只要因子载荷向量线性独立，所提估计量的渐近性质即可在一般设定下建立，允许因子之间存在相关性，且因子载荷向量无需接近正交。该程序能够适应因子载荷向量的稀疏性，可以处理弱因子，并在广泛的应用场景中展现出强大的性能。为了进一步降低估计误差，我们还引入了一种基于新型双重投影方法的迭代算法。我们从理论上证明了改进的收敛速度

    arXiv:2606.08560v2 Announce Type: replace-cross  Abstract: We adopt the canonical polyadic (CP) decomposition to model high-dimensional tensor time series. Our primary goal is to identify and estimate the factor loadings in the CP decomposition. We propose a one-pass estimation procedure through standard eigen-analysis for a matrix constructed based on the serial dependence structure of the data. The asymptotic properties of the proposed estimator are established under a general setting as long as the factor loading vectors are linearly independent, allowing the factors to be correlated and the factor loading vectors to be not nearly orthogonal. The procedure adapts to the sparsity of the factor loading vectors, accommodates weak factors, and demonstrates strong performance across a wide range of scenarios. To further reduce estimation errors, we also introduce an iterative algorithm based on a novel double projection approach. We theoretically justify the improved convergence rate of 
    
[^37]: 关于不完整U统计量中位数的有限样本集中性

    On Finite-sample Concentration of Median of Incomplete U-Statistics

    [https://arxiv.org/abs/2606.00661](https://arxiv.org/abs/2606.00661)

    本文证明了不完整U统计量中位数（MoIU）的有限样本浓度界，克服了此前仅能获得松散$O(n^{-1/4})$界的理论挑战，实现了更紧的收敛速率。

    

    中位数均值（MoM）是一种强大的技术，在理论上能在底层数据分布具有重尾特性（例如，仅假设具有前两阶有限矩）时，实现参数估计的接近亚高斯的有限样本速率。最近的研究将此技术推广到中位数随机化U统计量（MoRU）和中位数不完整U统计量（MoIU），用于估计重尾成对核的期望。在\citet{pmlr-v97-clemencon19a}中，已证明MoRU的浓度速率随样本量按$O(n^{-1/2})$缩放。然而，尽管后者具有计算优势，MoIU的有限样本界分析仍是一个重大的理论挑战。正如作者所指出的，直接应用McDiarmid不等式会产生$O(n^{-1/4})$阶的松散界。在本工作中，我们证明了MoIU估计的有限样本浓度界。

    arXiv:2606.00661v2 Announce Type: replace-cross  Abstract: Median-of-means (MoM) is a powerful technique that theoretically enables near sub-Gaussian finite-sample rate for parameter estimation when the underlying data distribution is heavy-tailed (e.g., assumed to have only two first finite moments). A recent work has extrapolated this technique to median-of-\textit{randomized}-U-Statistics (MoRU) and median-of-\textit{incomplete}-U-Statistics (MoIU) for estimating expectations of heavy-tailed pairwise kernels. In \citet{pmlr-v97-clemencon19a}, a concentration rate that scales like $O(n^{-1/2})$ with sample size has been proven for MoRU. However, despite the computational advantage of the latter, the analysis of finite-sample bound for MoIU remains a significant theoretical challenge. As noted by the authors, a straightforward application of McDiarmid's inequality yields a loose bound of order $O(n^{-1/4})$. In this work, we prove a finite-sample concentration bound for the MoIU estim
    
[^38]: 用于因果推断与模型发现的连续时间集合Kalman-Bucy平滑器

    A Continuous-Time Ensemble Kalman-Bucy Smoother for Causal Inference and Model Discovery

    [https://arxiv.org/abs/2604.25157](https://arxiv.org/abs/2604.25157)

    本文提出了一种连续时间集合Kalman-Bucy平滑器（EnKBS），通过集合矩重构条件分布，为非线性动力系统的数据同化提供了无需导数、切线性或伴随模型的平滑框架，实现了超越滤波的不确定性降低，并可应用于因果推断与模型发现。

    

    数据同化（DA）将观测信息与模型预测相结合，以改进复杂系统中的状态估计。滤波仅利用过去和当前的观测，为在线预报提供了基础，但当底层动力学快速演化或发生状态转变时，滤波可能出现延迟和偏差。平滑则进一步纳入未来的观测，为回溯预报和再分析提供了自然的流程，能够实现超越滤波的不确定性降低。本文提出了一种用于非线性动力系统连续时间数据同化的集合Kalman-Bucy平滑器（EnKBS），其中平滑器的条件分布通过集合矩进行重构。由此得到一个免导数的框架，无需显式计算切线性或伴随模型，并能在无穷集合极限下恢复精确的平滑均值和协方差方程。

    arXiv:2604.25157v3 Announce Type: replace-cross  Abstract: Data assimilation (DA) integrates observational information with model predictions to improve state estimation in complex systems. While filtering provides the basis for online forecasts by using only past and present observations, it can exhibit delays and biases when the underlying dynamics evolve rapidly or undergo regime transitions. Smoothing, which additionally incorporates future observations, provides a natural pipeline for hindcasting and reanalysis that yields an uncertainty reduction beyond the filter. This paper introduces an ensemble Kalman--Bucy smoother (EnKBS) for continuous-time DA of nonlinear dynamical systems, where the smoother's conditional distributions are reconstructed using ensemble moments. The result is a derivative-free framework that does not require explicit computation of tangent-linear or adjoint models, which recovers the exact smoothing mean and covariance equations in the infinite-ensemble li
    
[^39]: 面向不确定性下序贯决策的深度学习：基础、框架与前沿

    Deep Learning for Sequential Decision Making under Uncertainty: Foundations, Frameworks, and Frontiers

    [https://arxiv.org/abs/2604.11507](https://arxiv.org/abs/2604.11507)

    本教程以运筹学/管理科学（OR/MS）为核心视角，系统性地连接了深度学习神经架构与不确定性下序贯决策的OR/MS方法，其核心观点是深度学习是对优化的补充而非替代。

    

    人工智能（AI）正日益超越预测的范畴，转而在复杂、不确定且动态的环境中支持决策。这一转变使其与运筹学和管理科学（OR/MS）形成了天然的交汇点，后者长期以来一直为不确定性下的序贯决策提供方法论基础。与此同时，深度学习的进展——包括前馈神经网络、循环架构、Transformer、大语言模型（LLM）以及深度强化学习——扩展了面向大规模决策的数据驱动建模方法。本教程以运筹学/管理科学为核心视角，探讨用于不确定性下序贯决策的深度学习，旨在架起神经架构与OR/MS决策方法之间的桥梁。其核心前提是：深度学习是对优化的补充，而非替代。深度学习带来了适应性和可扩展的近似能力，而OR/MS则提供了……（摘要原文在此处截断）

    arXiv:2604.11507v2 Announce Type: replace-cross  Abstract: Artificial intelligence (AI) is moving increasingly beyond prediction to support decisions in complex, uncertain, and dynamic environments. This shift creates a natural intersection with operations research and management science (OR/MS), which has long provided methodological foundations for sequential decision making under uncertainty. At the same time, deep learning advances, including feedforward neural networks, recurrent architectures, transformers, large language models (LLMs), and deep reinforcement learning, have expanded data-driven modeling for large-scale decisions. This tutorial presents an OR/MS-centered perspective on deep learning for sequential decision making under uncertainty, bridging neural architectures and OR/MS approaches to decision making. Its premise: deep learning complements optimization rather than replacing it. Deep learning brings adaptability and scalable approximation, whereas OR/MS provides th
    
[^40]: 正定矩阵锥上Bregman散度的对称化：使用哪种均值以及为什么

    Symmetrizing Bregman Divergence on the Cone of Positive Definite Matrices: Which Mean to Use and Why

    [https://arxiv.org/abs/2603.28917](https://arxiv.org/abs/2603.28917)

    该论文揭示了正定矩阵锥上对称化Bregman散度的变分原理，证明前向对称化的规范均值是原始空间上的算术平均，而反向对称化的规范均值是对偶空间算术平均的拉回，在常用情形下分别对应算术、对数欧几里得和调和平均。

    

    本工作揭示了在正定矩阵锥上，对由一般镜像映射所诱导的Bregman散度进行对称化背后的变分原理。我们证明，计算这种对称化的规范均值可以表述为：在公理化定义、满足特定性质的一组均值泛函上，最小化目标对称化散度。对于前向对称化，我们证明对于正定锥上的任何镜像映射，原始空间上的算术平均都是规范均值。对于反向对称化，我们证明规范均值是对偶空间上的算术平均再拉回到原始空间所得的结果。将这一结果应用于实践中常用的三种镜像映射，我们证明了在这些情形下，反向对称化的规范均值分别是算术平均、对数欧几里得平均和调和平均。我们的结果增进了对现有对称化方法的理解。

    arXiv:2603.28917v3 Announce Type: replace-cross  Abstract: This work uncovers variational principles behind symmetrizing the Bregman divergences induced by generic mirror maps over the cone of positive definite matrices. We show that computing the canonical means for this symmetrization can be posed as minimizing the desired symmetrized divergences over a set of mean functionals defined axiomatically to satisfy certain properties. For the forward symmetrization, we prove that the arithmetic mean over the primal space is canonical for any mirror map over the positive definite cone. For the reverse symmetrization, we show that the canonical mean is the arithmetic mean over the dual space, pulled back to the primal space. Applying this result to three common mirror maps used in practice, we show that the canonical means for reverse symmetrization, in those cases, turn out to be the arithmetic, log-Euclidean and harmonic means. Our results improve understanding of existing symmetrization p
    
[^41]: 贝叶斯求积

    Bayesian Quadrature

    [https://arxiv.org/abs/2602.16218](https://arxiv.org/abs/2602.16218)

    本综述首次系统全面地梳理了贝叶斯求积方法，涵盖其数学基础、建模-推断-采样三维分类体系、理论保证、数值实验对比以及实际应用中的挑战与局限性。

    

    arXiv:2602.16218v2 公告类型：替换 摘要：贝叶斯求积是一种基于模型的概率化数值积分方法，用于估计难以直接计算的积分或期望。尽管贝叶斯求积早在20世纪80年代就已得到推广，但至今尚未有系统而全面的论述发表。本综述旨在填补这一空白。我们从不同的视角回顾了贝叶斯求积的数学基础；提出了一个系统的分类体系，沿着建模、推断和采样三个维度对不同的贝叶斯求积方法进行分类；汇集了一般性的理论保证；并提供了一项受控的数值研究，探索并阐明了分类体系各维度上不同选择所产生的影响。我们还在现实层面对贝叶斯求积方法在实际应用中面临的挑战与局限性进行了评估，并提供了一份最新且近乎详尽无遗的参考文献目录，不仅涵盖了机器学……（原文摘要在此处截断）

    arXiv:2602.16218v2 Announce Type: replace  Abstract: Bayesian quadrature is a probabilistic, model-based approach to numerical integration, the estimation of intractable integrals, or expectations. Although Bayesian quadrature was popularised already in the 1980s, no systematic and comprehensive treatment has been published. The purpose of this survey is to fill this gap. We review the mathematical foundations of Bayesian quadrature from different points of view; present a systematic taxonomy for classifying different Bayesian quadrature methods along the three axes of modelling, inference, and sampling; collect general theoretical guarantees; and provide a controlled numerical study that explores and illustrates the effect of different choices along the axes of the taxonomy. We also provide a realistic assessment of practical challenges and limitations to application of Bayesian quadrature methods and include an up-to-date and nearly exhaustive bibliography that covers not only machin
    
[^42]: 贝叶斯实验设计中的边界偏差与观测无关性修正

    Correcting Boundary Bias and Observation Independence in Bayesian Experimental Design

    [https://arxiv.org/abs/2602.01898](https://arxiv.org/abs/2602.01898)

    论文针对基于方差采集准则的高斯过程主动学习的两大缺陷——后验方差与观测内容无关以及边界处方差膨胀导致的过度采样，提出了修正方案，通过重构驱动的设计密度与基于后验均值的免训练变形，使采样更集中于目标函数变化剧烈的区域。

    

    在许多实验场景中，主动学习可以通过依次选择测量位置来提高样本效率，这在实验成本高昂时尤为有价值。基于方差采集准则的高斯过程被广泛用于这一目的，但其存在两个局限性。首先，它们与观测内容无关：其后验方差仅取决于样本采集的位置，而不取决于测量到的内容，这削弱了其对所采集数据结构的敏感性。其次，它们会在边界附近放大方差，导致相比空间内部在空间边缘进行过度采样。这些局限性削弱了顺序采集本应带来的采样效率提升。我们针对这两个局限性提出了修正方案：我们推导了一种由重构驱动的设计密度，并利用后验均值构建了一种无需训练的变形，将更多的测量点放置在目标函数变化迅速的区域。

    arXiv:2602.01898v2 Announce Type: replace  Abstract: In many experimental settings, active learning can improve sample efficiency by sequentially selecting where to measure, which is particularly valuable when experiments are expensive. Gaussian processes with variance-based acquisition criteria are widely used for this purpose, but have two limitations. First, they are observation-independent: their posterior variance depends only on where samples are acquired, not on what is measured, impairing their sensitivity to the structure of the acquired data. Second, they inflate the variance near boundaries, leading to excessive sampling at the edges of the space compared to the interior. These limitations undermine the gains in sampling efficiency expected from sequential acquisition. We address both limitations. We derive a reconstruction-driven design density and use the posterior mean to build a training-free warp that places more measurements where the target function varies rapidly. A 
    
[^43]: 非平衡采样下MMD的有限样本无偏方差：精确估计与拟线性计算

    Finite-Sample Unbiased Variance of MMD under Unbalanced Sampling: Exact Estimation and Quasi-Linear Computation

    [https://arxiv.org/abs/2601.13874](https://arxiv.org/abs/2601.13874)

    该论文推导了非平衡采样下MMD方差的有限样本无偏估计量，并通过拉普拉斯核的递归前缀-后缀累加方案将计算复杂度从 $\mathcal{O}(N^2)$ 降至 $\mathcal{O}(N \log N)$、内存仅需 $\mathcal{O}(N)$。

    

    准确且高效地估计最大均值差异（MMD）的方差仍然具有挑战性，尤其是在样本量不平衡的情况下。在本文中，我们推导了MMD方差的有限样本无偏估计量。为了克服传统的 $\mathcal{O}(N^2)$ 计算瓶颈，我们为拉普拉斯核开发了一种递归前缀-后缀累加方案，将计算复杂度降低至 $\mathcal{O}(N \log N)$，同时仅需 $\mathcal{O}(N)$ 的内存。实验结果验证了所提估计量的理论精确性和数值稳定性，并展示了其在大规模数据集上的可扩展性。此外，该方法在时间序列生成对抗网络训练过程中监测分布收敛方面也表现出有效性。

    arXiv:2601.13874v3 Announce Type: replace-cross  Abstract: Accurately and efficiently estimating the variance of the Maximum Mean Discrepancy (MMD) remains challenging, particularly for unbalanced sample sizes. In this paper, we derive a finite-sample unbiased estimator of the MMD variance. To overcome the traditional $\mathcal{O}(N^2)$ computational bottleneck, we develop a recursive prefix-suffix accumulation scheme for the Laplace kernel, reducing the computational complexity to $\mathcal{O}(N \log N)$ while requiring $\mathcal{O}(N)$ memory. Experimental results verify the theoretical exactness and numerical stability of the proposed estimator and demonstrate its scalability on large datasets. Furthermore, the method proves effective for monitoring distributional convergence during the training of Time-series Generative Adversarial Networks (TimeGAN).
    
[^44]: Wasserstein–Fisher–Rao 梯度流的算子分裂分析

    An operator splitting analysis of Wasserstein--Fisher--Rao gradient flows

    [https://arxiv.org/abs/2511.18060](https://arxiv.org/abs/2511.18060)

    本文定量分析了求解 WFR 梯度流时 W-FR 算子分裂的顺序与步长的影响，并出人意料地证明：合理选择步长和算子顺序时，分裂方案可以比精确 WFR 流更快地收敛到目标分布。

    

    Wasserstein-Fisher-Rao（WFR）梯度流最近被提出作为一种强大的采样工具，它结合了纯 Wasserstein（W）梯度流和纯 Fisher-Rao（FR）梯度流两者的优点。现有的算法开发中隐式地使用了算子分裂技术来数值逼近 WFR 偏微分方程，即在给定步长内先求解 W 流，再求解 FR 流（或反之）。本工作研究了 W 算子与 FR 算子求解顺序的影响，并旨在提供定量分析。令人有些惊讶的是，我们证明，通过明智地选择步长和算子顺序，分裂方案（就模型时间而言）可以比精确的 WFR 流更快地收敛到目标分布。我们获得了描述两种分裂方案在一个时间步内演化的变分公式，并研究了在哪些情形下 W-FR 分裂方案更适用。

    arXiv:2511.18060v3 Announce Type: replace-cross  Abstract: Wasserstein-Fisher-Rao (WFR) gradient flows have been recently proposed as a powerful sampling tool that combines the advantages of pure Wasserstein (W) and pure Fisher-Rao (FR) gradient flows. Existing algorithmic developments implicitly make use of operator splitting techniques to numerically approximate the WFR partial differential equation, whereby the W flow is evaluated over a given step size and then the FR flow (or vice versa). This works investigates the impact of the order in which the W and FR operator are evaluated and aims to provide a quantitative analysis. Somewhat surprisingly, we show that with a judicious choice of step size and operator ordering, the split scheme can converge to the target distribution faster than the exact WFR flow (in terms of model time). We obtain variational formulae describing the evolution over one time step of both splitting schemes and investigate in which settings the W-FR split sho
    
[^45]: 面向皮尔逊相关匹配的合成数据最优后处理

    Optimal Post-processing of Synthetic Data for Pearson Correlation Matching

    [https://arxiv.org/abs/2510.02405](https://arxiv.org/abs/2510.02405)

    本文提出一种与生成器无关的后处理方法，通过对合成数据进行最小改动以恢复原始数据的皮尔逊相关矩阵，给出了该最小化问题的唯一显式解及修正幅度的理论界限，且在保持边际分布、数据几何结构和分类性能方面表现良好。

    

    在合成数据中保持相关性在多个应用中具有重要意义。现有方法主要在生成过程中处理相关性保持问题，而本文则将其视为一个后处理问题。给定原始数据与合成表格数据，我们寻求对合成数据集进行最小的改动，以恢复原始数据的皮尔逊相关矩阵。在适当的假设条件下，我们推导出了该最小化问题的唯一显式解，并给出了修正幅度与初始相关误差之间关系的理论界限。在数值实验方面，跨多个数据集和多种生成方法，所提出的方法在很大程度上保持了边际分布、t-SNE几何结构和分类性能。该方法与生成器无关，可以在数据合成之后应用，而无需修改生成过程。

    arXiv:2510.02405v3 Announce Type: replace-cross  Abstract: Preserving correlation in synthetic data is of interest in several applications. Existing approaches mainly address correlation preservation during the generation procedure. Here, we instead consider it as a postprocessing problem. Given original and synthetic tabular data, we seek the smallest change to the synthetic dataset that restores the Pearson correlation matrix of the original data. Under suitable assumptions, we derive a unique explicit solution to the minimization problem. We also provide a bound on the size of the correction in terms of the initial correlation error. On the numerical side, across several datasets and generation methods, the proposed approach largely preserves marginal distributions, t-SNE geometry, and classification performance. The method is generator-independent and can be applied after synthesis without modifying the generation procedure.
    
[^46]: 对数凹性下Metropolis-within-Gibbs算法的谱隙研究

    Spectral gap of Metropolis-within-Gibbs under log-concavity

    [https://arxiv.org/abs/2509.26175](https://arxiv.org/abs/2509.26175)

    该论文通过精确估计一维随机游走Metropolis核的传导率，将对数凹分布下随机扫描Metropolis-within-Gibbs算法的谱隙下界从 $\Omega((\kappa^2 d)^{-1})$ 改进为 $\Omega((\kappa d)^{-1})$，证明其混合性能仅比精确Gibbs采样器差一个常数因子。

    

    Metropolis-within-Gibbs（MwG）算法是一种广泛使用的马尔可夫链蒙特卡洛方法，适用于精确条件采样不可行时的高维分布采样问题。我们研究了采用随机游走Metropolis（RWM）更新的MwG算法，其中提议方差与相应的条件方差一致地可比。假设目标分布 $\pi$ 是一个条件数为 $\kappa$ 的 $d$ 维对数凹分布，我们为随机扫描版本的MwG建立了阶为 $\Omega((\kappa d)^{-1})$ 的谱隙下界，改进了此前已有的 $\Omega((\kappa^2 d)^{-1})$ 界。该结果是通过发展一维RWM核传导率的精确估计而获得的，这一估计技术本身可能也具有独立的研究价值。结果表明，在所述的一致调优条件下，MwG的混合速度可以显著更快，且其混合性能仅比精确的Gibbs采样器差一个常数因子。

    arXiv:2509.26175v2 Announce Type: replace  Abstract: The Metropolis-within-Gibbs (MwG) algorithm is a widely used Markov chain Monte Carlo method for sampling from high-dimensional distributions when exact conditional sampling is intractable. We study MwG with Random Walk Metropolis (RWM) updates, whose proposal variances are uniformly comparable to the corresponding conditional variances. Assuming the target $\pi$ is a $d$-dimensional log-concave distribution with condition number $\kappa$, we establish a spectral gap lower bound of order $\Omega((\kappa d)^{-1})$ for the random-scan version of MwG, improving on the previously available $\Omega((\kappa^2 d)^{-1})$ bound. This is obtained by developing sharp estimates of the conductance of one-dimensional RWM kernels, which may be of independent interest. The result shows that MwG can mix substantially faster under the stated uniform tuning condition and that its mixing performance is just a constant factor worse than that of the exact
    
[^47]: 一种利用潜在扩散模型求解逆问题的梯度流方法

    A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models

    [https://arxiv.org/abs/2509.19276](https://arxiv.org/abs/2509.19276)

    提出了一种免训练的扩散正则化Wasserstein梯度流方法（DWGF），利用预训练潜在扩散模型作为先验来求解不适定逆问题。

    

    求解不适定逆问题需要强大且灵活的先验。我们提出利用预训练的潜在扩散模型来完成这一任务，采用一种新的免训练方法，称为扩散正则化Wasserstein梯度流。具体而言，我们将后验采样问题表述为潜在空间中期望负对数后验目标的Wasserstein梯度流，并通过与扩散先验之间的Kullback-Leibler散度进行正则化。我们以StableDiffusion (Rombach et al., 2022) 作为先验，在标准基准上展示了我们方法的性能。

    arXiv:2509.19276v2 Announce Type: replace-cross  Abstract: Solving ill-posed inverse problems requires powerful and flexible priors. We propose leveraging pretrained latent diffusion models for this task through a new training-free approach, termed Diffusion-regularized Wasserstein Gradient Flow (DWGF). Specifically, we formulate the posterior sampling problem as a Wasserstein gradient flow in the latent space of an expected negative log posterior objective, regularized by a Kullback-Leibler divergence to the diffusion prior. We demonstrate the performance of our method on standard benchmarks using StableDiffusion (Rombach et al., 2022) as the prior.
    
[^48]: 面向磁共振波谱贝叶斯推断的物理信息Sylvester归一化流

    Physics-Informed Sylvester Normalizing Flows for Bayesian Inference in Magnetic Resonance Spectroscopy

    [https://arxiv.org/abs/2505.03590](https://arxiv.org/abs/2505.03590)

    该论文提出了一种基于Sylvester归一化流的贝叶斯推断框架，结合融入物理先验知识的解码器，用于磁共振波谱中代谢物浓度的可靠定量化。

    

    磁共振波谱（MRS）是一种测量组织代谢成分的无创技术，可为神经系统疾病、肿瘤检测及其他代谢功能障碍提供宝贵见解。然而，准确的代谢物定量化受到谱重叠、低信噪比及各种伪影等挑战的阻碍。传统方法如线性组合建模容易产生歧义，且通常仅能以Cramér-Rao界的形式提供估计精度的理论下界。本工作引入了一个使用Sylvester归一化流（SNFs）的贝叶斯推断框架，以近似代谢物浓度的后验分布，从而提高定量的可靠性。基于物理的解码器融入了MRS信号形成的先验知识，确保了符合实际的分布表示。我们在模拟的7T质子数据上对该方法进行了验证。

    arXiv:2505.03590v2 Announce Type: replace-cross  Abstract: Magnetic resonance spectroscopy (MRS) is a non-invasive technique to measure the metabolic composition of tissues, offering valuable insights into neurological disorders, tumor detection, and other metabolic dysfunctions. However, accurate metabolite quantification is hindered by challenges such as spectral overlap, low signal-to-noise ratio, and various artifacts. Traditional methods like linear-combination modeling are susceptible to ambiguities and commonly only provide a theoretical lower bound on estimation accuracy in the form of the Cram\'er-Rao bound. This work introduces a Bayesian inference framework using Sylvester normalizing flows (SNFs) to approximate posterior distributions over metabolite concentrations, enhancing quantification reliability. A physics-based decoder incorporates prior knowledge of MRS signal formation, ensuring realistic distribution representations. We validate the method on simulated 7T proton 
    
[^49]: 具有形状先验的函数型BART：一种用于约束函数回归的贝叶斯树方法

    Functional BART with Shape Priors: A Bayesian Tree Approach to Constrained Functional Regression

    [https://arxiv.org/abs/2502.16888](https://arxiv.org/abs/2502.16888)

    提出了一种结合样条表示与树形分割结构的非参数贝叶斯方法FBART用于函数对标量回归，并通过引入单调性、凸性等形状先验约束来增强估计与预测性能。

    

    受贝叶斯加性回归树（BART）在回归建模中显著成功的启发，我们提出了一种新颖的非参数贝叶斯方法，称为函数型BART（FBART），专门针对函数对标量回归设计。FBART利用基于样条的函数型响应表示，结合灵活的基于树的分割结构，有效捕捉响应曲线与标量预测变量之间复杂且异质的关系。为实现高效的后验推断，我们开发了一种定制的贝叶斯回拟合算法。此外，我们通过在响应曲线上引入形状约束（如单调性或凸性）对FBART进行了扩展，使得在已有形状先验信息的情况下能够实现更优的估计和预测。形状先验的使用确保后验样本满足所指定的函数约束。在温和的正则性条件下，我们建立了后验一致性等理论保证。

    arXiv:2502.16888v3 Announce Type: replace-cross  Abstract: Motivated by the remarkable success of Bayesian additive regression trees (BART) in regression modelling, we propose a novel nonparametric Bayesian method, termed Functional BART (FBART), tailored specifically for function-on-scalar regression. FBART leverages spline-based representations for functional responses coupled with a flexible tree-based partitioning structure, effectively capturing complex and heterogeneous relationships between response curves and scalar predictors. To facilitate efficient posterior inference, we develop a customized Bayesian backfitting algorithm. Additionally, we extend FBART by introducing shape constraints (e.g., monotonicity or convexity) on the response curves, enabling enhanced estimation and prediction when prior shape information is available. The use of shape priors ensures that posterior samples respect the specified functional constraints. Under mild regularity conditions, we establish p
    
[^50]: 突破序贯校准问题的 $T^{2/3}$ 瓶颈

    Breaking the $T^{2/3}$ Barrier for Sequential Calibration

    [https://arxiv.org/abs/2406.13668](https://arxiv.org/abs/2406.13668)

    本文首次突破了序贯校准问题中 Foster & Vohra 提出的 $O(T^{2/3})$ 校准误差上界，改进了这一停滞二十余年的经典界限。

    

    如果预测者做出的每个预测都能在其进行该预测的时间步子集上紧密逼近结果的经验分布，则称这组概率预测是校准的。我们研究了在标准 $\ell_1$ 校准误差度量下二值序列在线校准预测这一基本问题，该问题最早由 Foster & Vohra（1998）研究。他们提出了一个在 $T$ 个时间步后校准误差为 $O(T^{2/3})$ 的算法，并证明了 $\Omega(T^{1/2})$ 的下界。这些界限在二十年间一直停滞不前，直到 Qiao & Valiant（2021）通过引入一种名为“符号保持”的组合博弈，并证明该博弈的下界可以推出校准问题的下界，从而将下界改进为 $\Omega(T^{0.528})$。在本文中，我们首次对 Foster & Vohra 提出的 $O(T^{2/3})$ 校准误差上界做出了改进，我们通过引入一种变体

    arXiv:2406.13668v4 Announce Type: replace  Abstract: A set of probabilistic forecasts is calibrated if each prediction of the forecaster closely approximates the empirical distribution of outcomes on the subset of timesteps where that prediction was made. We study the fundamental problem of online calibrated forecasting of binary sequences under the standard $\ell_1$ calibration error metric, which was initially studied by Foster & Vohra (1998). They derived an algorithm with $O(T^{2/3})$ calibration error after $T$ time steps, and showed a lower bound of $\Omega(T^{1/2})$. These bounds remained stagnant for two decades, until Qiao & Valiant (2021) improved the lower bound to $\Omega(T^{0.528})$ by introducing a combinatorial game called sign preservation and showing that lower bounds for this game imply lower bounds for calibration.   In this paper, we give the first improvement to the $O(T^{2/3})$ upper bound on calibration error of Foster & Vohra. We do this by introducing a variant
    
[^51]: 迁移学习的极限

    Limits of Transfer Learning

    [https://arxiv.org/abs/2006.12694](https://arxiv.org/abs/2006.12694)

    该论文在算法搜索框架下证明了迁移学习的若干理论极限，表明迁移信息必须经过谨慎选择并与目标问题存在依赖关系，同时算法的概率变化程度决定了其性能改进的上限。

    

    迁移学习是指从一个问题领域中获取信息和洞察，并将其应用于新的问题领域。尽管迁移学习在实践中被广泛使用，但其理论发展仍不够完善。为了解决这一问题，我们证明了若干与迁移学习相关的新结果，表明需要仔细选择要迁移的信息集合，并且迁移的信息与目标问题之间必须存在依赖关系。此外，我们证明了使用迁移学习的算法的概率变化程度如何对其可能实现的改进量设定了上限。这些结果建立在机器学习的算法搜索框架之上，使得这些结论能够适用于广泛的迁移学习问题。

    arXiv:2006.12694v2 Announce Type: replace-cross  Abstract: Transfer learning involves taking information and insight from one problem domain and applying it to a new problem domain. Although widely used in practice, theory for transfer learning remains less well-developed. To address this, we prove several novel results related to transfer learning, showing the need to carefully select which sets of information to transfer and the need for dependence between transferred information and target problems. Furthermore, we prove how the degree of probabilistic change in an algorithm using transfer learning places an upper bound on the amount of improvement possible. These results build on the algorithmic search framework for machine learning, allowing the results to apply to a wide range of learning problems using transfer.
    

