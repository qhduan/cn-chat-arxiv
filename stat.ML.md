# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Elliptic Sampling: Optimal Interpolant Schedules for Gaussians and Beyond](https://arxiv.org/abs/2611.18542) | 该论文提出了一种基于复化ODE轨迹解析性质的插值调度设计新框架，通过保角映射精确求解得到由雅可比椭圆函数参数化的保方差椭圆调度族，并结合Zolotarev最优有理函数逼近构建了能根据目标分布条件数自适应优化的新采样算法Zoloflow。 |
| [^2] | [Why Finite Intelligence Works in an Infinite World: A Mathematical Theory](https://arxiv.org/abs/2611.18538) | 该论文提出“未来任务信息”理论，证明当任务请求共享集体结构时，有限的信息表示足以在无限环境中保持决策价值，并给出了有限保存的两个精确数学边界条件。 |
| [^3] | [On numerical issues of divergences between nearly equal distributions](https://arxiv.org/abs/2611.18511) | 该论文揭示了当类别分布几乎相等时，KL散度、JSD散度及熵等标准散度运算存在与softmax类似的数值精度问题，并提出了数值稳定的重新表述方案及经过属性测试验证的实现。 |
| [^4] | [Bregman geometry in regular exponential families: dual representations and relative smoothness](https://arxiv.org/abs/2611.18480) | 本文基于射影刻画，为正则指数族在期望参数化下建立了一般性的等价对偶Bregman表示构造（含归一化与唯一性结果），使对数密度可表达为对偶坐标系中的普通Bregman散度，并将β-与α-缩放对偶表示作为该框架的特例。 |
| [^5] | [Contrastive latent representation learning with denoising score matching](https://arxiv.org/abs/2611.18469) | 该论文提出ScoreCLR方法，通过推导香农熵梯度得到涉及潜在表示分数的等价目标，并利用去噪分数匹配技术由轻量级网络与编码器联合训练来估计排斥项，从而以新颖方式克服了对比学习中排斥力估计依赖大批量、高方差和维度灾难的问题。 |
| [^6] | [Node Localization by Multidimensional Scaling with Iterative Majorization](https://arxiv.org/abs/2611.18466) | 本文提出了一种基于迭代主化的综合多维缩放算法，可统一处理锚点、非欧氏距离、非对称及三向数据等多种数据特征，实现节点位置的精确估计。 |
| [^7] | [Mesh-Free Weak-Form Regression for Learning Interaction Kernels from Particle Trajectories](https://arxiv.org/abs/2611.18455) | 该论文提出一种无网格弱形式回归方法，可从含噪粒子轨迹中稳健高效地学习相互作用核，并给出了分离有限粒子、时间离散、观测噪声与采样误差的高概率系数误差理论界。 |
| [^8] | [Dynamic bigraphical Mat\'ern--Whittle processes: learning dependence graphs among spatial-temporal fields](https://arxiv.org/abs/2611.18454) | 该论文提出动态双图Matérn–Whittle（DBMW）过程，能够学习多变量时空场之间随时间变化（连边出现、消失甚至改变符号）的依赖图结构，同时支持高斯、计数和二项等多种响应类型，且联合分布可借助图拉普拉斯特征模态高效分解计算。 |
| [^9] | [Lipschitz-Constrained Discriminators in Generative Modeling: Asymptotic Behavior and Support Mismatch](https://arxiv.org/abs/2611.18394) | 本文刻画了生成建模中带 Lipschitz 约束的最优判别器在正则化参数 $L$ 趋于无穷时的渐近行为，涵盖了绝对连续且 Radon-Nikodym 导数有界、以及离散分布等多种情形。 |
| [^10] | [DECODE: Decomposable Environments for COntrolled Sequential-Decision Evaluation](https://arxiv.org/abs/2611.18367) | 提出了DECODE基准，通过共享已知动力学矩阵、可在记忆与噪声参数上分解控制的消费信贷催收环境，以受控实验方法评估规划范围和信息可用性等修改对序贯决策系统性能的具体影响。 |
| [^11] | [Small Is Not Dead: Distance-Aware Cluster Revival for e-means](https://arxiv.org/abs/2611.18238) | 针对 e-means 仅依据聚类规模判定死聚类、会误删覆盖稀疏孤立点群的小聚类的问题，本文提出增加基于中心间距离的第二条件——只有当小聚类的中心靠近其他中心时才将其移除复活，从而在含稀疏孤立点群的数据上显著降低聚类误差。 |
| [^12] | [LaMe: a Joint Latent Mediation Framework for High-Dimensional Systems](https://arxiv.org/abs/2611.18180) | 该论文提出LaMe，一个联合线性-高斯潜在中介框架，通过少量潜变量概括高维的暴露-中介-结果系统，在明确假设下识别自然直接与间接效应，克服了现有方法过度参数化、推断有偏和可扩展性受限的问题。 |
| [^13] | [Detection and Classification in Latent Spaces: High-Dimensional Analysis and Validation](https://arxiv.org/abs/2611.18164) | 该论文在潜在维度与样本规模成比例增长的高维状态下，对基础模型潜空间中的线性和二次检测器进行了Neyman-Pearson大偏差分析，推导出虚警与漏检错误概率的显式刻画，揭示了双下降乃至多重下降的非单调现象，并通过视觉、文本和音频数据集实验验证了理论预测与实际性能的高度一致性。 |
| [^14] | [Training-point consistency of kernel ridge regression with varying correlation scales](https://arxiv.org/abs/2611.18095) | 本文建立了变化尺度参数下核岭回归的统一理论框架，证明训练点风险的一致性当且仅当有效局部样本量发散到无穷大，从而无缝统一了长程、固定尺度和短程等各种情形。 |
| [^15] | [Efficient and Practical Clustering of Variables in Causal Graphs](https://arxiv.org/abs/2611.17971) | 本文提出了一类称为“导管”的新型聚类，并给出检测因果图中所有导管的可靠、完备且高效算法，该类聚类在真实世界因果图中自然出现，可大幅缩减图规模并支持已知的可识别性结果。 |
| [^16] | [Hierarchical Bayesian physics-augmented neural networks for populations of hyperelastic laws: specimen variability and uncertainty from scarce, noisy data](https://arxiv.org/abs/2611.17892) | 提出一种分层贝叶斯物理增强神经网络（非线性混合效应模型），可从稀疏、含噪数据中联合推断软组织等材料超弹性本构模型的群体分布、各样本个体差异及其不确定性。 |
| [^17] | [Complex Schr\"odinger Bridges](https://arxiv.org/abs/2611.17746) | 该论文提出了共形最优传输（COT）框架与复值薛定谔桥（CSBs）这一新型动态扩散桥方法，通过共形几何先验有效处理复向量空间中受缩放和旋转定律支配的物理系统的扩散任务。 |
| [^18] | [The Polyak-{\L}ojasiewicz Paper: A 10-Year Retrospective](https://arxiv.org/abs/2611.17745) | 本文是对引入Polyak-Łojasiewicz (PL)条件术语的论文的十年回顾，作者回顾了论文的起源、给出了AI辅助精炼的主定理版本，并讨论了对该工作的批评及相关扩展。 |
| [^19] | [Homogeneous pursuit in $\beta$ model](https://arxiv.org/abs/2611.17609) | 本文提出同质追踪方法，一种通过原始-对偶主化-最小化算法实现的 $\ell_0$ 约束最大似然估计方法，用于估计稀疏约束下的β模型，并证明了该算法具有良好的收敛速度。 |
| [^20] | [Uncertainty-Aware Continual Learning for 2D XRD Interpretation Using Simulated and Experimental Data](https://arxiv.org/abs/2611.17547) | 该论文提出了一个不确定性感知的持续学习概念验证框架，通过融合基于MMD的领域自适应、CNN-SNGP不确定性估计以及结合知识蒸馏的重放机制，实现二维XRD数据在模拟与实验域之间稳健且可持续更新的解释。 |
| [^21] | [Multifidelity Operator Inference for Dynamical Systems](https://arxiv.org/abs/2611.17503) | 本文提出MF-OpInf方法，通过受控制变量法启发的多保真度最小二乘公式，将有限的高保真度快照与丰富的低保真度快照相结合，为非线性动力系统学习更准确稳定的非侵入式降阶模型。 |
| [^22] | [SURFACE: Preference Distribution Matching on Pareto Fronts via Scalarization Steering](https://arxiv.org/abs/2611.17484) | 提出SURFACE方法，通过估计权重网格与帕累托前沿区域的对应关系并自适应更新标量化权重，使有限数量的帕累托解能够匹配期望的偏好分布。 |
| [^23] | [Distributionally Robust Risk Evaluation with Low-Rank Perturbations](https://arxiv.org/abs/2611.17447) | 该论文量化了将分布鲁棒扰动限制为行/列空间共享的低秩矩阵时最坏情况风险的损失，提出了基于曲率校正的投影梯度上升算法来计算最坏情况扰动，并给出了收敛性保证与可计算的风险证书。 |
| [^24] | [Enumerating Posterior-Supported Priors: First-Order Error of Finite-Integral Cross-Categorical Measurement Morphisms](https://arxiv.org/abs/2611.17390) | 提出枚举后验支撑先验框架，基于 Quot->MorTop->Set 三层范畴链显式记录底层谓词支撑、可逆性与误差所在层级，解决标准测度论 a.e. 约定下零测度支撑隐藏域外误差无法被显式计入的问题。 |
| [^25] | [Trajectory Density Ratios under Weak Support: A Survey across the Sciences](https://arxiv.org/abs/2611.11985) | 本文是一篇跨学科综述，系统梳理了轨迹密度比这一统一量在强化学习、分子模拟、气候归因和语言模型后训练等领域的核心作用，并分析了其在支撑薄弱（弱支持）情形下的存在性条件、方差爆炸与无定义等根本性失效问题。 |
| [^26] | [Watch Your Step: Adam's Margin Maximization Bias Depends on the Learning Rate Schedule](https://arxiv.org/abs/2611.01929) | 本文揭示了Adam的隐式偏差依赖于学习率调度——在常见的恒定或弱衰减调度下，Adam对齐次模型不再具有与轨迹无关的间隔最大化偏差。 |
| [^27] | [High-dimensional online calibration from harmonic weights](https://arxiv.org/abs/2610.07740) | 本文提出了一个基于调和权重、对过去结果进行调和平滑的简单在线校准算法，首次在高维多结果预测中以 $d^{O(1/\varepsilon)}$ 轮实现 $\varepsilon$-校准，将此前结果的维度依赖性指数级降低。 |
| [^28] | [Nearly Optimal Fixed-Confidence Best-Arm Identification with 1-Bit Feedback](https://arxiv.org/abs/2610.02771) | 本文在严格1比特反馈约束下提出了近最优的固定置信度最优臂识别算法，通过随机化阈值查询与自适应截断技术实现了间隙自适应的样本复杂度，并给出了相匹配的信息论下界。 |
| [^29] | [Generative sequence modeling for infinite memory processes via predictive states](https://arxiv.org/abs/2609.38524) | 本文提出一种基于预测状态的生成式序列建模新方法，能够处理具有无穷记忆的随机过程，当过去历史可被压缩为低维充分统计量时实现快速收敛，且估计问题的统计复杂度仅由预测状态空间的内在维数决定。 |
| [^30] | [Risk-Aware Quantile Learning for Personalized Dynamic Treatment Regimes](https://arxiv.org/abs/2608.05434) | 提出风险感知分位数动态治疗方案（RQDTR）这一统一框架，在优化累积潜在结果分位数的同时显式纳入治疗相关风险，并通过基于角度的公式化方法实现多类别治疗决策规则的联合学习。 |
| [^31] | [When Does On-Policy Interaction Help? Representational Tradeoffs in Value-Based Imitation Learning](https://arxiv.org/abs/2607.29617) | 该论文发现，在基于价值的模仿学习中，与专家进行同策略交互能放宽对学习者的表征要求——只需模型能实现专家的价值函数，而无需实现要求更严格的专家策略本身。 |
| [^32] | [Disentangling Continuous-Time Latent Dynamics: Identifiability of Latent SDEs via Diffusion Shifts](https://arxiv.org/abs/2606.28228) | 该论文证明了在未知非线性观测下，仅利用多个环境间扩散协方差的变化即可识别加性噪声潜在SDE的潜在坐标（至置换、缩放和常数平移），且无需对漂移项作任何稀疏性假设。 |
| [^33] | [Exact Constrained Decoding for Sequence Models: Controller-Augmented Hidden Markov Models and Beyond](https://arxiv.org/abs/2606.13850) | 提出控制器增强隐马尔可夫模型（CHMMs）框架，将路径级约束编译为有限状态控制器，借助增广链上的标准递推算法，在离散与连续时间中实现精确的约束后验计算与最大后验路径解码。 |
| [^34] | [Express Language Modeling](https://arxiv.org/abs/2606.10944) | Express 是一种将非因果注意力近似转换为具有匹配保证的因果近似的工具，与 Thinformer 结合后实现了已知最佳的因果注意力近似保证，并通过高效的 Triton 实现显著超越 FlashAttention 2，解决了语言建模中的长上下文预填充、KV 缓存压缩和长文本解码等四大资源瓶颈。 |
| [^35] | [Kernel-based potential mean-field games with unbiased random Fourier $U$-statistics](https://arxiv.org/abs/2605.29371) | 本文针对运行成本与终端成本均由核MMD惩罚刻画的势均场博弈，提出了一种基于无偏随机傅里叶U统计量的计算框架，以神经网络参数化漂移并通过随机梯度下降训练，同时证明了在惩罚发散及显式耦合条件下向终端约束问题的收敛性。 |
| [^36] | [Adaptive Calibration in Non-Stationary Environments](https://arxiv.org/abs/2605.11490) | 该论文提出了能够自动适应环境非平稳程度的在线预测校准算法，其校准误差可在独立同分布与对抗性两种情形之间平滑过渡，并在多种校准指标下提供了自适应保证。 |
| [^37] | [mlr3torch: A Deep Learning Framework in R based on mlr3 and torch](https://arxiv.org/abs/2604.18152) | 本文介绍了R包mlr3torch，这是一个构建于torch之上、深度集成mlr3生态系统的可扩展深度学习框架，它简化了神经网络的定义、训练和评估，并允许用户通过图语言在单一图中定义包含预处理、数据增强和网络架构的完整建模工作流程。 |
| [^38] | [Minimizing classical resources in variational measurement-based quantum computation for generative modeling](https://arxiv.org/abs/2604.11578) | 本文针对变分测量量子计算（VMBQC）中信道模型参数量是幺正模型两倍的局限，提出了最小化经典资源消耗的方案，使生成建模中的经典处理开销得到显著降低。 |
| [^39] | [Auto-differentiable data assimilation: Co-learning of states, dynamics, and filtering algorithms](https://arxiv.org/abs/2603.20891) | 本文提出了一种自动微分滤波框架，利用理论驱动的损失函数和基于梯度的优化，实现对状态、动力学模型以及滤波算法参数的联合协同学习，从而摆脱传统数据同化中昂贵的参数调优和对精确动力学模型的依赖。 |
| [^40] | [Fast Uncertainty Quantification for Kernel-Based Estimators in Large-Scale Causal Inference](https://arxiv.org/abs/2603.13662) | 本文通过扩展因果小自助法（cBLB），实现了大规模因果推断中基于核方法的估计量的快速不确定性量化，解决了传统自助法在数百万级样本数据集上计算不可行的问题。 |
| [^41] | [On sparsity, extremal structure, and monotonicity properties of Wasserstein and Gromov-Wasserstein optimal transport plans](https://arxiv.org/abs/2602.16265) | 本文系统梳理了Gromov-Wasserstein最优传输的关键性质，证明了在条件负定/正定（CND/CPD）条件下存在稀疏且支撑于置换上的最优传输计划，并给出不满足这些假设时置换并非最优的反例。 |
| [^42] | [Flow Matching from Viewpoint of Proximal Operators](https://arxiv.org/abs/2602.12683) | 该论文通过扩展Brenier势将最优传输条件流匹配（OT-CFM）的向量场精确表示为与势函数相关的近端算子（去噪器），并借助Lyapunov指数证明其流形稳定性：法向扰动线性收缩而切向扰动被保留。 |
| [^43] | [Deep Multivariate Models with Parametric Conditionals](https://arxiv.org/abs/2602.01953) | 该论文提出用每组变量以其余变量为条件的参数化条件概率分布来表示联合概率分布，从而构建几乎可通用于任意下游任务的深度多元模型，其训练等价于通过最大化极限分布的数据似然来学习一个参数化马尔可夫链核。 |
| [^44] | [Manifold Approximation leads to Robust Kernel Alignment](https://arxiv.org/abs/2510.22953) | 提出流形近似核对齐（MKA），通过将流形局部邻域几何融入核对齐任务，去除启发式度加权、给出闭式对齐分数及马尔可夫概率解释，并证明其对邻域大小的稳定性，从而实现更鲁棒的表示比较。 |
| [^45] | [Adversarial Attacks on Downstream Weather Forecasting Models: Application to Tropical Cyclone Trajectory Prediction](https://arxiv.org/abs/2510.10140) | 本文提出了Cyc-Attack对抗攻击方法，通过对上游天气预报模型施加细微扰动来操纵下游热带气旋轨迹预测，克服了检测系统黑盒不可微分和热带气旋样本极度稀缺导致的类别不平衡两大难题。 |
| [^46] | [Toric geometry of ReLU neural networks](https://arxiv.org/abs/2509.05894) | 该论文建立了ReLU神经网络与环面几何之间的联系，定义了ReLU扇、ReLU环面簇和ReLU卡地亚除子等新概念，借助代数几何工具系统研究了网络的精确函数实现问题与表达能力，并揭示了热带几何与环面几何之间的对应关系。 |
| [^47] | [Recovering the Euclidean Geometry of High-Dimensional Data Under Heterogeneous Noise](https://arxiv.org/abs/2507.18520) | 本文证明了对于具有底层低维几何结构的高维数据，即使在异构噪声严重且不均匀地破坏成对欧几里得距离的情况下，仍然可以恢复出其底层干净的欧几里得几何结构。 |
| [^48] | [Entropic smoothing and gradient approximation for Wasserstein distributionally robust optimization](https://arxiv.org/abs/2506.04948) | 本文证明了Wasserstein分布鲁棒优化的熵平滑正则化变体的近似梯度在正则化参数趋于零时收敛到原问题的次梯度，从而为随机梯度方法提供了收敛保证与收敛速率。 |
| [^49] | [Bias-Optimal Bounds for Stochastic Gradient Descent](https://arxiv.org/abs/2505.17965) | 本文在仅假设凸性和光滑性的条件下，为随机梯度下降推导出覆盖全部常数步长范围的偏差最优收敛界，并证明在临界步长处最优偏差虽无法精确达到但可被任意逼近。 |
| [^50] | [Benign Overfitting for Regression with Trained Two-Layer ReLU Networks](https://arxiv.org/abs/2410.06191) | 本文首次证明了通过梯度流训练的有限宽度两层ReLU网络在最小二乘回归中既能实现泛化又会过拟合数据，从而在任意有界回归函数下首次建立了良性过拟合结果。 |

# 详细

[^1]: 椭圆采样：面向高斯分布及更广泛情形的最优插值调度

    Elliptic Sampling: Optimal Interpolant Schedules for Gaussians and Beyond

    [https://arxiv.org/abs/2611.18542](https://arxiv.org/abs/2611.18542)

    该论文提出了一种基于复化ODE轨迹解析性质的插值调度设计新框架，通过保角映射精确求解得到由雅可比椭圆函数参数化的保方差椭圆调度族，并结合Zolotarev最优有理函数逼近构建了能根据目标分布条件数自适应优化的新采样算法Zoloflow。

    

    基于随机插值的生成建模需要确定两个关键选择：连接初始分布与目标分布的插值调度，以及由此得到的概率流ODE的高效离散化方法。在这项工作中，我们提出了一个基于（复化后）ODE轨迹解析性质的调度/离散化联合设计新框架，该解析性质进而保证了良好的数值行为。我们的公式可以通过保角映射精确求解，从而得到一个由雅可比椭圆函数参数化的、自然的保方差椭圆调度族。将椭圆调度与量身定制的离散化方法相结合，我们得到了Zoloflow——一种与Zolotarev最优有理函数逼近密切相关的新采样算法。关键之处在于，椭圆调度与Zoloflow都显式地纳入了目标分布的条件数 $\kappa$，并可退化为经典的……（摘要截断）

    arXiv:2611.18542v1 Announce Type: cross  Abstract: Generative modeling based on stochastic interpolants requires specifying two key choices: an interpolant schedule bridging the initial and target distributions, and an efficient discretization of the resulting probability-flow ODE. In this work, we introduce a new framework for joint schedule/discretization design based on the analyticity properties of the (complexified) ODE trajectories, which in turn ensure good numerical behavior. Our formulation can be solved exactly via conformal mapping, yielding a natural family of variance-preserving elliptic schedules, parametrized by Jacobi elliptic functions. Pairing an elliptic schedule with a tailored discretization, we obtain Zoloflow, a new sampling algorithm closely connected to Zolotarev's optimal rational function approximants. Crucially, both the elliptic schedule and Zoloflow explicitly incorporate the condition number $\kappa$ of the target distribution and reduce to the classical 
    
[^2]: 为什么有限智能能在无限世界中有效：一种数学理论

    Why Finite Intelligence Works in an Infinite World: A Mathematical Theory

    [https://arxiv.org/abs/2611.18538](https://arxiv.org/abs/2611.18538)

    该论文提出“未来任务信息”理论，证明当任务请求共享集体结构时，有限的信息表示足以在无限环境中保持决策价值，并给出了有限保存的两个精确数学边界条件。

    

    一个信息系统往往需要在知道自己将被问及什么之前就进行学习。当这些请求共享集体结构，而不是要求掌握散布在环境中任意位置的不相关事实时，有限的表示可以在不完整记录无限环境的情况下取得成功。我们通过“未来任务信息”来形式化这一现象，即相对于可用证据而言，保持决策价值所需的最小保留信息。一个联合决策-后悔状态记录了每个允许任务中各行动之间的后验比较。在紧性假设下，每个保留的表示都可以基于该状态进行校准替换，在不增加信息的情况下保留所有任务后悔；由此产生的信息-后悔曲线等于一个操作性编码率。两个精确的边界支配着有限保存：对于固定的行动数量是弱星闭包，以及当所有有限……（原文摘要在此处被截断）

    arXiv:2611.18538v1 Announce Type: cross  Abstract: An information system often has to learn before it knows what it will be asked. A finite representation can succeed without recording an infinite environment in full when those asks share collective structure, rather than demand unrelated facts scattered arbitrarily across it. We formalize this phenomenon through future-task information, the minimum retained information needed to preserve decision value relative to the available evidence. A joint decision-regret state records posterior comparisons among actions for every permitted task. Under compactness assumptions, every retained representation admits a calibrated replacement based on this state, preserving all task regrets without increasing information; the resulting information-regret curve equals an operational coding rate. Two exact boundaries govern finite preservation: weak-star closure for a fixed action count, and the singular mass of the source-evidence law when all finite 
    
[^3]: 关于几乎相等分布之间散度的数值问题

    On numerical issues of divergences between nearly equal distributions

    [https://arxiv.org/abs/2611.18511](https://arxiv.org/abs/2611.18511)

    该论文揭示了当类别分布几乎相等时，KL散度、JSD散度及熵等标准散度运算存在与softmax类似的数值精度问题，并提出了数值稳定的重新表述方案及经过属性测试验证的实现。

    

    Barron等人（2026年）最近发现了softmax和软交叉熵损失中的数值问题。我们证明在类别分布之间的标准散度运算中也存在类似的问题，这包括正向和反向Kullback-Leibler（KL）散度、广义Jensen-Shannon散度（JSD），以及蒙特卡洛KL估计和熵。当分布几乎相等时，舍入和抵消问题会影响前向计算和梯度计算。这些问题可能影响在优化目标中使用散度的场景，例如知识蒸馏、强化学习和近端方法。我们提出了数值稳定的重新表述方案，并为每种情况提供了示例实现，并通过基于属性的测试进行了验证。

    arXiv:2611.18511v1 Announce Type: new  Abstract: Barron et al. (2026) recently identified numerical issues in the softmax and soft cross-entropy loss. We show that similar issues are present in standard divergence operations between categorical distributions. This includes the forward and reverse Kullback-Leibler (KL) divergence and generalized Jensen-Shannon divergence (JSD), as well as Monte Carlo KL estimates and entropy. When the distributions are nearly equal, rounding and cancellation issues affect the forward and gradient computations. These issues can affect settings using divergences in optimization objectives, such as knowledge distillation, reinforcement learning, and proximal methods. We propose numerically stable reformulations and provide example implementations for each setting, validated by property-based tests.
    
[^4]: 正则指数族中的Bregman几何：对偶表示与相对光滑性

    Bregman geometry in regular exponential families: dual representations and relative smoothness

    [https://arxiv.org/abs/2611.18480](https://arxiv.org/abs/2611.18480)

    本文基于射影刻画，为正则指数族在期望参数化下建立了一般性的等价对偶Bregman表示构造（含归一化与唯一性结果），使对数密度可表达为对偶坐标系中的普通Bregman散度，并将β-与α-缩放对偶表示作为该框架的特例。

    

    正则指数族分布在统计学中无处不在，因为其具有优美且便利的性质。值得注意的是，这类分布可以用Bregman散度来表示，从而揭示了统计学、凸分析和信息几何之间的基本联系。然而，这一几何结构尚未被系统地用于刻画这些分布相对于其期望参数化的性质。基于射影刻画，我们建立了等价对偶Bregman表示的一般性构造，并给出了归一化与唯一性结果。这些表示将对数密度表达为对偶坐标系中的普通Bregman散度，从而为参数空间提供了几何刻画。所谓的β-与α-缩放对偶表示作为该框架的特殊实例自然涌现。我们进一步推导了共轭……（原摘要在此处截断）

    arXiv:2611.18480v1 Announce Type: cross  Abstract: Regular exponential-family distributions are ubiquitous in statistics due to their appealing and convenient properties. Notably, they can be expressed in terms of Bregman divergences, revealing fundamental connections between statistics, convex analysis, and information geometry. However, this geometric structure has not been systematically exploited to characterize these distributions with respect to their expectation parametrization. Building on a projective characterization, we establish a general construction of equivalent dual Bregman representations, together with normalization and uniqueness results. These representations express log-densities as ordinary Bregman divergences in dual coordinate systems, providing geometric characterizations of the parameter space. So-called $\beta$- and $\alpha$-scaled dual representations emerge as particular instances of this framework. We further derive dual Bregman representations of conjugat
    
[^5]: 基于去噪分数匹配的对比潜在表示学习

    Contrastive latent representation learning with denoising score matching

    [https://arxiv.org/abs/2611.18469](https://arxiv.org/abs/2611.18469)

    该论文提出ScoreCLR方法，通过推导香农熵梯度得到涉及潜在表示分数的等价目标，并利用去噪分数匹配技术由轻量级网络与编码器联合训练来估计排斥项，从而以新颖方式克服了对比学习中排斥力估计依赖大批量、高方差和维度灾难的问题。

    

    对比表示学习方法通过将同一样本的不同视图拉近，同时将其与不同样本的视图推开来进行学习。一个根本性的挑战在于对排斥力的估计，这通常需要较大的批处理规模，可能表现出高方差，并且容易受到维度灾难的影响。我们提出了ScoreCLR，这是一种对比表示学习方法，为估计这一排斥项提供了一种新颖的方式。我们考虑了通过最大化样本两个潜在视图之间互信息来进行表示学习的框架。通过推导香农熵关于编码器参数的梯度，我们得到了一个涉及潜在表示分数的等价目标函数。随后，我们使用去噪分数匹配技术，通过一个与编码器联合训练的轻量级网络来估计该分数。我们还进一步提出了正则化方法（摘要在此处被截断）。

    arXiv:2611.18469v1 Announce Type: cross  Abstract: Contrastive representation learning methods operate by pulling different views of the same data sample together while pushing them away from views of different samples. A fundamental challenge is the estimation of the repulsive force, which typically requires large batch sizes, can exhibit high variance, and is susceptible to the curse of dimensionality. We introduce ScoreCLR, a contrastive representation learning method that offers a novel way to estimate this repulsion term. We consider the framework of representation learning by maximizing the mutual information between two latent views of a sample. By deriving the gradient of the Shannon entropy with respect to the encoder parameters, we obtain an equivalent objective that involves the score of the latent representation. We then use denoising score matching techniques to estimate this score with a lightweight network that is trained jointly with the encoder. We further propose regu
    
[^6]: 基于迭代主化多维缩放的节点定位

    Node Localization by Multidimensional Scaling with Iterative Majorization

    [https://arxiv.org/abs/2611.18466](https://arxiv.org/abs/2611.18466)

    本文提出了一种基于迭代主化的综合多维缩放算法，可统一处理锚点、非欧氏距离、非对称及三向数据等多种数据特征，实现节点位置的精确估计。

    

    节点定位旨在基于从其他节点收集的数据来估计节点的位置。根据数据特征的不同，需要采用不同的方法来估计位置。本文提出了一种基于迭代主化的综合性多维缩放算法，该算法涵盖了多种不同的数据特征来估计节点位置。隐式归一化的损失函数能够提供绝对距离，并在给定锚点的情况下给出节点的绝对位置。该算法采用了一种新的收敛坐标下降过程来最小化主化函数，从而能够自然地处理锚点。通过对数据进行最优变换，非欧几里得距离等其他类型的数据也能被无缝处理。来自多个来源的数据，无论是非对称数据还是三向数据，均适用于同一广泛框架。

    arXiv:2611.18466v1 Announce Type: new  Abstract: Node localization aims to estimate the positions of nodes based on collected data from other nodes. Depending on the data characteristics, different procedures are utilized to estimate the locations. In this paper, we introduce a comprehensive multidimensional scaling algorithm based on iterative majorization that comprises many of the different data characteristics to estimate node locations. The implicitly normalized loss function provides absolute distances and, given anchors, absolute node locations. The algorithm uses a new convergent coordinate descent procedure for the minimization of the majorization function, allowing for a natural treatment of anchors. Optimal transformations of the data seamlessly allows other data than Euclidean distances. Data from multiple sources, resulting in either asymmetric or three-way data, fit the same extensive framework.
    
[^7]: 无网格弱形式回归：从粒子轨迹中学习相互作用核

    Mesh-Free Weak-Form Regression for Learning Interaction Kernels from Particle Trajectories

    [https://arxiv.org/abs/2611.18455](https://arxiv.org/abs/2611.18455)

    该论文提出一种无网格弱形式回归方法，可从含噪粒子轨迹中稳健高效地学习相互作用核，并给出了分离有限粒子、时间离散、观测噪声与采样误差的高概率系数误差理论界。

    

    我们提出了一种无网格弱形式回归方法，用于从含噪粒子轨迹中学习相互作用核。我们的方法使用光滑测试函数以避免对含噪数据进行求导，并利用经验粒子平均值以避免密度重建和空间网格划分。随后，我们对候选函数的稀疏组合进行拟合，并在采样的粒子对上评估相互作用项，从而避免了全粒子对的二次计算代价。通过将可辨识性从平均场极限转移过来，我们证明了高概率的系数误差界，该误差界将有限粒子、时间离散化、观测噪声和粒子对采样等误差分离开来，且不包含网格分辨率或密度估计相关的误差项。我们在一维（1D）牛顿排斥、二维捕食者-猎物以及10维距离局域化的实验中进行了评估。数值结果呈现出与理论系数误差界一致的趋势。我们还将弱形式与前向差分形式同LASSO、Ridge等方法进行了比较（摘要在此处截断）。

    arXiv:2611.18455v1 Announce Type: new  Abstract: We propose mesh-free weak-form regression for learning interaction kernels from noisy particle trajectories. Our method uses smooth test functions to avoid differentiating noisy data and empirical particle averages to avoid density reconstruction and spatial grids. We then fit sparse combinations of candidate functions, evaluating interaction terms on sampled particle pairs to avoid all-pair quadratic cost. By transferring identifiability from the mean-field limit, we prove high-probability coefficient error bounds that separate finite-particle, time-discretization, observation-noise, and pair-sampling errors, without grid-resolution or density-estimation terms. We evaluate one-dimensional (1D) Newtonian-repulsion, 2D predator-prey, and 10D distance-localized experiments. The numerical results show trends consistent with the theoretical coefficient-error bounds. Comparisons of weak and forward-difference formulations with LASSO, Ridge, a
    
[^8]: 动态双图Matérn–Whittle过程：学习时空场之间的依赖图

    Dynamic bigraphical Mat\'ern--Whittle processes: learning dependence graphs among spatial-temporal fields

    [https://arxiv.org/abs/2611.18454](https://arxiv.org/abs/2611.18454)

    该论文提出动态双图Matérn–Whittle（DBMW）过程，能够学习多变量时空场之间随时间变化（连边出现、消失甚至改变符号）的依赖图结构，同时支持高斯、计数和二项等多种响应类型，且联合分布可借助图拉普拉斯特征模态高效分解计算。

    

    许多研究在相同的地点上跨多个时期记录多个变量，例如监测站的污染物浓度或按区域统计的传染病病例数。一个核心问题是变量之间的依赖关系及其随时间的变化。我们为此类数据开发了动态双图Matérn–Whittle（DBMW）过程。每个变量都有一个通过自回归演化的空间场。变量上的一个图将各场的时间新息耦合起来，其连边随时间出现和消失，且一条连边可能改变符号。地点上的图拉普拉斯算子给出空间结构，规范指数族将场与高斯、计数或二项响应联系起来。潜在场在所有地点、变量和时期上的联合分布在图拉普拉斯算子的特征模态上可分解，其归一化常数与图和动态无关。给定潜在场后……

    arXiv:2611.18454v1 Announce Type: cross  Abstract: Many studies record several variables at the same sites over a sequence of periods, such as pollutant concentrations at monitoring stations or case counts of infectious diseases by district. A central question concerns the dependence between the variables and its change over time. We develop dynamic bigraphical Mat\'ern--Whittle (DBMW) processes for such data. Each variable has a spatial field that evolves by autoregression. A graph over the variables couples the temporal innovations of the fields. Its links appear and vanish over time, and a link may change sign. A graph Laplacian on the sites gives the spatial structure, and a canonical exponential family links the fields to Gaussian, count or binomial responses. The joint distribution of the latent fields over all sites, variables and periods factorizes over the eigenmodes of the graph Laplacian, and its normalizing constant is free of the graph and of the dynamics. Given the fields
    
[^9]: 生成建模中的Lipschitz约束判别器：渐近行为与支撑不匹配

    Lipschitz-Constrained Discriminators in Generative Modeling: Asymptotic Behavior and Support Mismatch

    [https://arxiv.org/abs/2611.18394](https://arxiv.org/abs/2611.18394)

    本文刻画了生成建模中带 Lipschitz 约束的最优判别器在正则化参数 $L$ 趋于无穷时的渐近行为，涵盖了绝对连续且 Radon-Nikodym 导数有界、以及离散分布等多种情形。

    

    我们研究了与两个概率分布 $Q$ 和 $P$ 的 $L$-Lipschitz 正则化 KL 散度相关联的最优判别器的渐近行为，该散度通过参数 $L>0$ 在缩放的 Wasserstein-1 度量与 KL 散度之间进行插值。参数 $L$ 起着 Lipschitz 约束的作用，并被视为一个正则化参数：较小的 $L$ 值强制判别器更具几何感知能力，而随着 $L$ 的增大，几何正则化被逐渐放松。已知结果 [3] 表明，当 $L\to\infty$ 时，这类散度收敛于 KL 散度。在本文中，我们刻画了最优判别器在 $L\to\infty$ 时在不同情形下的渐近行为：首先，在 $Q\ll P$ 且 Radon-Nikodym 导数 $\frac{dQ}{dP}$ 有上界且远离零的情况下；其次，当 $Q$ 和 $P$ 是满足 $Q\ll P$ 的离散分布时；最后，当 $Q$……

    arXiv:2611.18394v1 Announce Type: cross  Abstract: We study the asymptotic behavior of optimal discriminators associated to $L-$Lipschitz regularized KL-divergences of two probability distributions $Q$ and $P$ that interpolate between the rescaled Wasserstein$-1$ metric by some parameter $L>0$ and the KL divergence. The parameter $L$ plays the role of a Lipschitz constraint and is viewed as a regularization parameter: small values of $L$ enforce a more geometry-aware discriminator, while as $L$ increases the geometric regularization is progressively relaxed. It is a known result [3] that such divergences converge to the KL divergence as $L\to\infty$. In this paper, we characterize the asymptotic behavior of the optimal discriminators as $L\to\infty$ in different regimes: Firstly, under $Q\ll P$ and the Radon-Nikodym derivative $\frac{dQ}{dP}$ is bounded from above and bounded away from zero. Secondly, when $Q$ and $P$ are discrete distributions satisfying $Q\ll P$, and finally when $Q$
    
[^10]: DECODE：用于受控序贯决策评估的可分解环境

    DECODE: Decomposable Environments for COntrolled Sequential-Decision Evaluation

    [https://arxiv.org/abs/2611.18367](https://arxiv.org/abs/2611.18367)

    提出了DECODE基准，通过共享已知动力学矩阵、可在记忆与噪声参数上分解控制的消费信贷催收环境，以受控实验方法评估规划范围和信息可用性等修改对序贯决策系统性能的具体影响。

    

    当一个序贯决策系统未能达到最优时，其得分并不能揭示哪些修改能够改进其决策。DECODE 是一个基准测试，用于通过对规划方式和可用信息进行受控实验来检验此类修改。其消费信贷催收生成器提供了五个部分可观测的环境，这些环境共享已知的动力学矩阵，并在记忆参数和过程噪声上有所变化。实验在保持求解器其余部分不变的情况下，改变选定的输入或规划规则，每次比较都明确了求解器的状态输入、响应输入和规划范围。在确定性基础环境中（该环境被选中是因为使用隐藏状态时六个月规划与一个月规划之间存在较大差距），将规划范围从一个月延长到六个月，使两个梯度提升求解器的得分相对精确最优值（基于隐藏状态和借款人信息计算）提高了15到17个百分点。

    arXiv:2611.18367v1 Announce Type: new  Abstract: When a sequential decision system falls short of an optimum, its score does not reveal which modifications would improve its decisions. DECODE is a benchmark for testing such modifications through controlled experiments on planning and available information. Its consumer-credit collections generator provides five partially observed environments that share known dynamics matrices and vary a memory parameter and process noise. The experiments change selected inputs or planning rules while keeping the rest of the solver fixed. Each comparison specifies the solver's state inputs, response inputs and planning horizon. On the deterministic base environment, which was selected for a large gap between six-month and one-month planning with the hidden state, extending planning from one to six months increases the scores of two gradient-boosted solvers by 15 to 17 percentage points of the exact optimum, computed with the hidden state and borrower t
    
[^11]: 小并非消亡：面向 e-means 的距离感知聚类复活机制

    Small Is Not Dead: Distance-Aware Cluster Revival for e-means

    [https://arxiv.org/abs/2611.18238](https://arxiv.org/abs/2611.18238)

    针对 e-means 仅依据聚类规模判定死聚类、会误删覆盖稀疏孤立点群的小聚类的问题，本文提出增加基于中心间距离的第二条件——只有当小聚类的中心靠近其他中心时才将其移除复活，从而在含稀疏孤立点群的数据上显著降低聚类误差。

    

    e-means（Bessa 等人，2026）是一种新近提出的小批量 k-means 算法，现已并入 FAISS 库，在大规模嵌入数据集上能比 Lloyd 算法快约一个数量级地达到相同的误差水平。每处理五个小批量，该算法会移除所有运行中点数低于平均值十分之一的聚类，并将其中心重新用于分裂一个大聚类。在作者的消融实验中，移除这一被称为“死聚类复活”的机制所导致的误差上升，比移除滑动平均更新中加入的其他任何组件都要大。然而，该规则仅考虑聚类规模：它无法区分因冗余而变小，还是因覆盖稀疏、孤立的点群而变小，因此在包含此类点群的数据上，它会反复移除那些移除代价很高的中心。我们增加了一个基于中心间距离的第二条件：仅当小聚类的中心相对于彼此靠近另一个中心时，小聚类才会被移除。

    arXiv:2611.18238v1 Announce Type: new  Abstract: e-means (Bessa et al., 2026) is a recent mini-batch k-means algorithm, now part of the FAISS library, that reaches the error of Lloyd's algorithm about an order of magnitude faster on large embedding datasets. Every five mini-batches, it removes every cluster whose running point count falls below a tenth of the average and reuses its centre to split a large cluster. In the authors' ablation, removing this mechanism, called dead-cluster revival, raises the error more than removing any other component added to the moving-average update. The rule, however, considers only cluster size: it cannot distinguish a cluster that is small because it is redundant from one that covers a rare, isolated group of points, and on data with such groups it repeatedly removes centres whose removal is costly. We add a second condition, based on the distance between centres: a small cluster is removed only if its centre is close to another centre, relative to a
    
[^12]: LaMe：面向高维系统的联合潜在中介分析框架

    LaMe: a Joint Latent Mediation Framework for High-Dimensional Systems

    [https://arxiv.org/abs/2611.18180](https://arxiv.org/abs/2611.18180)

    该论文提出LaMe，一个联合线性-高斯潜在中介框架，通过少量潜变量概括高维的暴露-中介-结果系统，在明确假设下识别自然直接与间接效应，克服了现有方法过度参数化、推断有偏和可扩展性受限的问题。

    

    中介分析旨在理解暴露因素如何通过中间变量影响结果。然而，随着日益丰富的数据集的出现，成百上千个潜在中介变量使得这些系统难以建模和解释。潜变量方法为处理这类高维场景提供了一种自然的途径。然而，现有的潜在中介方法存在诸多局限，包括过度参数化、推断有偏或可扩展性受限。我们开发了LaMe，这是一个联合的线性-高斯框架，通过少量潜变量来概括高维的暴露-中介-结果系统。我们在一个明确的假设下（即显性测量完全捕捉潜中介变量）识别出自然直接效应和间接效应，并刻画了潜在表示能够有效约束观测交叉协方差结构的条件。模拟实验...

    arXiv:2611.18180v1 Announce Type: cross  Abstract: Mediation analysis seeks to understand how an exposure influences an outcome through intermediate variables. However, as increasingly rich datasets become available, hundreds to thousands of potential mediators make these systems difficult to model and hard to interpret. Latent-variable approaches offer a natural way to accommodate these high-dimensional settings. However, existing latent mediation methods face limitations, including overparameterization, biased inference, or limited scalability. We develop LaMe, a joint linear-Gaussian framework that summarizes a high-dimensional exposure-mediator-outcome system through a small number of latent variables. We identify natural direct and indirect effects under an explicit assumption that the manifest measurements fully capture the latent mediators, and characterize conditions under which the latent representation effectively restricts the observed cross-covariance structure. Simulations
    
[^13]: 潜空间中的检测与分类：高维分析与验证

    Detection and Classification in Latent Spaces: High-Dimensional Analysis and Validation

    [https://arxiv.org/abs/2611.18164](https://arxiv.org/abs/2611.18164)

    该论文在潜在维度与样本规模成比例增长的高维状态下，对基础模型潜空间中的线性和二次检测器进行了Neyman-Pearson大偏差分析，推导出虚警与漏检错误概率的显式刻画，揭示了双下降乃至多重下降的非单调现象，并通过视觉、文本和音频数据集实验验证了理论预测与实际性能的高度一致性。

    

    基础模型会产生高维潜在表示，其中经过适当设计的统计决策规则既能有效工作，又便于解析处理。我们在高维状态下研究这类检测器，即潜在维度与训练样本规模按比例增长的情形。对于线性和二次统计量，我们在Neyman-Pearson范式下推导出了虚警与漏检性能的显式大偏差刻画，并展示了二元分析如何扩展到多类别分类。所得到的错误概率及相应的衰减速率对形状参数呈现出非单调依赖性，包括双下降现象，在某些状态下甚至出现多重下降现象。在现代基础模型上对视觉、文本和音频数据集进行的实验表明，理论预测与实际性能高度吻合。特别地，所提出的框架使我们能够设定预定的虚警……

    arXiv:2611.18164v1 Announce Type: cross  Abstract: Foundation models produce high-dimensional latent representations in which suitably designed statistical decision rules can be effective yet analytically tractable. We study such detectors in the high-dimensional regime, where latent dimension and training sample size grow proportionally. For linear and quadratic statistics, we derive explicit large-deviation characterizations of false-alarm and miss-detection performance under the Neyman-Pearson paradigm, and show how the binary analysis extends to multi-class classification. The resulting error probabilities and corresponding decay rates exhibit non-monotonic dependence on the shape parameter, including double-descent and, in some regimes, multiple-descent. Experiments on vision, text, and audio datasets using modern foundation models show close agreement between theoretical predictions and empirical performance. In particular, the proposed framework allows us to set a prescribed fal
    
[^14]: 相关尺度变化下核岭回归的训练点一致性

    Training-point consistency of kernel ridge regression with varying correlation scales

    [https://arxiv.org/abs/2611.18095](https://arxiv.org/abs/2611.18095)

    本文建立了变化尺度参数下核岭回归的统一理论框架，证明训练点风险的一致性当且仅当有效局部样本量发散到无穷大，从而无缝统一了长程、固定尺度和短程等各种情形。

    

    核岭回归中的尺度参数在建模数据的相关结构方面起着关键作用。在现代应用中，采样密度与潜在的相关范围往往同时发生变化，这使得固定尺度的假设具有很强的局限性。然而，在变化的尺度参数下建立渐近理论从根本上是具有挑战性的。为填补这一空白，本文为变化尺度参数下的核岭回归建立了一个统一的理论框架。我们证明，在带有乘积指数核的均匀张量网格上，训练点风险的一致性成立当且仅当有效局部样本量发散到无穷大。这一判据覆盖了任意确定性尺度序列下的所有情形，无缝统一了长程、固定尺度和短程等各种情形。此外，我们还提供了一个显式的、与信号无关的确定性构造……

    arXiv:2611.18095v1 Announce Type: cross  Abstract: The scale parameter in kernel ridge regression plays a pivotal role in modeling the correlation structure of data. In modern applications, sampling density and the underlying correlation range often change simultaneously, making a fixed-scale assumption highly restrictive. However, establishing asymptotic theories under varying scale parameters is fundamentally challenging. To address this gap, this paper develops a unified theoretical framework for kernel ridge regression under varying scale parameters. We establish that, on uniform tensor grids with product exponential kernels, the consistency of the training-point risk holds if and only if the effective local sample size diverges to infinity. This sharp criterion universally covers all regimes along arbitrary deterministic scale sequences, seamlessly unifying the long-range, fixed-scale, and short-range. Furthermore, we provide an explicit, signal-independent construction of a deter
    
[^15]: 因果图中变量的高效实用聚类

    Efficient and Practical Clustering of Variables in Causal Graphs

    [https://arxiv.org/abs/2611.17971](https://arxiv.org/abs/2611.17971)

    本文提出了一类称为“导管”的新型聚类，并给出检测因果图中所有导管的可靠、完备且高效算法，该类聚类在真实世界因果图中自然出现，可大幅缩减图规模并支持已知的可识别性结果。

    

    对图中的变量进行聚类可以提高视觉清晰度并简化因果推断。要充分发挥聚类的潜力，需要一种自动化方法来检测图中有效的聚类。我们引入了一类称为“导管”的新型聚类，并提出了一种可靠、完备且高效的算法来检测图中的所有导管。除了算法上的可检测性之外，导管在源自现实世界应用的因果图中也自然出现，从而能够大幅缩减图的规模。此外，导管的结构使得关于聚类的已知可识别性结果得以应用。我们通过基于真实世界图的模拟研究验证了这些特性。

    arXiv:2611.17971v1 Announce Type: cross  Abstract: Clustering variables in a graph may improve visual clarity and simplify causal inference. Harnessing the full potential of clustering requires an automated method to detect useful clusters in a graph. We introduce a novel class of clusters called conduits, and present a sound, complete and efficient algorithm for detecting all conduits in the graph. In addition to algorithmic detectability, conduits appear naturally in causal graphs stemming from real-world applications, enabling substantial size reduction. Furthermore, the structure of conduits enables the application of known identifiability results regarding clustering. We demonstrate these properties in a simulation study with real-world graphs.
    
[^16]: 面向超弹性本构模型群体的分层贝叶斯物理增强神经网络：从稀疏、含噪数据中获取样本间变异性与不确定性

    Hierarchical Bayesian physics-augmented neural networks for populations of hyperelastic laws: specimen variability and uncertainty from scarce, noisy data

    [https://arxiv.org/abs/2611.17892](https://arxiv.org/abs/2611.17892)

    提出一种分层贝叶斯物理增强神经网络（非线性混合效应模型），可从稀疏、含噪数据中联合推断软组织等材料超弹性本构模型的群体分布、各样本个体差异及其不确定性。

    

    软组织等材料的超弹性本构关系在不同样本之间存在差异，而通常只能对少数样本进行测试。物理增强神经网络（PANNs）能够在不预设函数形式的情况下从数据中学习此类本构关系，同时保持其物理合理性；相关的贝叶斯模型版本则可量化单个样本或一组汇集样本的本构模型的不确定性。然而，汇集拟合会将样本之间的差异归因于测量噪声或对平均本构模型的不确定性，这两者都无法描述一个新的样本。我们首先针对单个样本开发了一种贝叶斯PANN，其先验设计保证了在数据稀少且含噪的情况下所学到的本构关系仍然物理合理。随后，我们将其扩展为分层贝叶斯PANN，即一种非线性混合效应模型：所有样本共享学习得到的多凸特征，而每个样本则是从本构模型总体中抽取的一个样本。特征、样本与总体通过变分推断进行联合推断……

    arXiv:2611.17892v1 Announce Type: cross  Abstract: The hyperelastic law of a material such as soft tissue differs between specimens, and often only a few specimens can be tested. Physics-augmented neural networks (PANNs) learn such laws from data without a prescribed functional form while keeping them physically admissible, and Bayesian versions of related models quantify the uncertainty of the law of one specimen or of a pooled group of specimens. A pooled fit, however, attributes the differences between specimens to measurement noise or to uncertainty about a mean law, and neither describes a new specimen. We first develop a Bayesian PANN for one specimen, whose priors keep the learned law plausible for few, noisy data. We then extend it to a hierarchical Bayesian PANN, a nonlinear mixed-effects model in which all specimens share learned polyconvex features and each specimen is a draw from a population of laws. Features, specimens, and population are inferred jointly by variational i
    
[^17]: 复值薛定谔桥

    Complex Schr\"odinger Bridges

    [https://arxiv.org/abs/2611.17746](https://arxiv.org/abs/2611.17746)

    该论文提出了共形最优传输（COT）框架与复值薛定谔桥（CSBs）这一新型动态扩散桥方法，通过共形几何先验有效处理复向量空间中受缩放和旋转定律支配的物理系统的扩散任务。

    

    扩散桥已成为促进任意数据分布之间相互转换的通用框架。然而，这些方法受到庞大优化空间及其底层高阶结构的阻碍。本文探索了扩散桥的一种变体，用于解决复向量空间中由缩放和旋转定律（如哈密顿结构）所支配的现实问题。为此，我们引入了共形最优传输（COT），这是一种复值传输框架，专为在几何提升下具有几何性与耗散动力学的物理系统而设计。在此基础上，我们提出了一类新的动态扩散桥，称为复值薛定谔桥（CSBs），其衍生算法能够利用共形几何先验来处理基于扩散的任务。为了提升可扩展性，我们提出了图像到图像的CSB（I²CSB），这是一种易于处理的扩散桥技术，具有解析形式的……

    arXiv:2611.17746v1 Announce Type: new  Abstract: Diffusion bridges have emerged as versatile frameworks for facilitating transformations between arbitrary data distributions. However, these approaches are hindered by an immense optimization space and its underlying high-order structures. This paper explores a variant of diffusion bridges for solving real-world problems in complex vector spaces governed by scaling and rotation laws, such as Hamiltonian structures. To address this, we introduce conformal optimal transport (COT), a complex-valued transportation framework designed for physical systems with geometric and dissipative dynamics under a geometric lifting. Building on this, we propose a new class of dynamical diffusion bridges, called complex Schr\"odinger bridges (CSBs), whose derived algorithms can handle diffusion-based tasks using a conformal geometric prior. For scalability, we propose image-to-image CSB (I$^2$CSB), a tractable diffusion-bridge technique with an analytic sp
    
[^18]: Polyak-Łojasiewicz 论文：十年回顾

    The Polyak-{\L}ojasiewicz Paper: A 10-Year Retrospective

    [https://arxiv.org/abs/2611.17745](https://arxiv.org/abs/2611.17745)

    本文是对引入Polyak-Łojasiewicz (PL)条件术语的论文的十年回顾，作者回顾了论文的起源、给出了AI辅助精炼的主定理版本，并讨论了对该工作的批评及相关扩展。

    

    2016年，我们撰写了一篇论文，为一个古老的想法引入了“Polyak-Łojasiewicz (PL) 条件”这一术语，当时并不确定读者会有何反应。这篇论文最终产生了超出我们预期的影响，我们感到惊讶并十分荣幸能获得2026年PKDD ECML时间检验奖。凭借10年后的后见之明，本文对PL条件进行了一次非正式且坦率的回顾。我们回顾了这篇论文的起源，给出了借助AI辅助精炼的主定理版本，审视了针对该工作的批评，并讨论了各种相关的想法与扩展。

    arXiv:2611.17745v1 Announce Type: cross  Abstract: In 2016, we wrote a paper introducing the phrase "Polyak-Lojasiewicz (PL) condition" for an old idea, unsure of how readers would react. The paper ultimately had a larger impact than we expected, and we are surprised and honoured to receive the 2026 PKDD ECML Test of Time Award. With 10 years of hindsight, this note offers an informal and candid retrospective on the PL condition. We take a look back at the origins of the paper, give an AI-assisted refined version of the main theorem, review criticisms of the work, and discuss a variety of related ideas and extensions.
    
[^19]: β模型中的同质追踪

    Homogeneous pursuit in $\beta$ model

    [https://arxiv.org/abs/2611.17609](https://arxiv.org/abs/2611.17609)

    本文提出同质追踪方法，一种通过原始-对偶主化-最小化算法实现的 $\ell_0$ 约束最大似然估计方法，用于估计稀疏约束下的β模型，并证明了该算法具有良好的收敛速度。

    

    随着大数据时代的到来，以网络形式呈现的数据已出现在众多重要领域之中。为了揭示并恢复拥有大量连接的核心单元的真实模型，带有稀疏性约束的β模型被提出，以满足统计学家和工程师的需求。该网络模型不仅继承了Erdős-Rényi模型的优异性质，还通过调节稀疏性参数 $s$ 展现出度异质性。在本文中，我们提出了同质追踪方法，这是一种通过原始-对偶算法实现的 $\ell_0$ 约束最大似然估计方法。为了减轻繁重的计算负担，我们找到了相应目标函数的替代函数，从而在原始-对偶过程中推导出主化-最小化算法。此外，经过细致的评估，我们所提出的算法在收缩映射的意义下具有良好的收敛速度特性。

    arXiv:2611.17609v1 Announce Type: cross  Abstract: Data in forms of networks has been encountered in various prevalent fields at the arrival of big data era. To unveil and recover the true model with core units with numerous links, the $\beta$-model under sparsity constraint is proposed to meet the expectation of statisticians and engineers. Not only does this network model inherit spectacular property from the Erd\H{o}s-R\'enyi model, but also exhibits degree heterogeneity through tuning sparsity parameter $s$. In this paper, we propose Homogeneous Pursuit (HP), an $\ell_0$-constrained maximum likelihood estimation method implemented through a primal-dual algorithm. To ease the harshness of computation, we find the surrogate function of the corresponding objective function to derive Majorization Minimization algorithm through the primal-dual process. Moreover, after meticulous evaluation, the algorithm brought about captures the property of fine convergence rate in a contraction mappi
    
[^20]: 基于模拟与实验数据的面向二维X射线衍射（XRD）解释的不确定性感知持续学习

    Uncertainty-Aware Continual Learning for 2D XRD Interpretation Using Simulated and Experimental Data

    [https://arxiv.org/abs/2611.17547](https://arxiv.org/abs/2611.17547)

    该论文提出了一个不确定性感知的持续学习概念验证框架，通过融合基于MMD的领域自适应、CNN-SNGP不确定性估计以及结合知识蒸馏的重放机制，实现二维XRD数据在模拟与实验域之间稳健且可持续更新的解释。

    

    自主实验系统需要能够解释异构测量数据，并随着数据积累保持预测性能的表征模型。开发此类模型需要解决数据源之间的领域不匹配、对陌生输入的过度自信预测，以及顺序更新过程中的灾难性遗忘等问题。在本工作中，我们提出了一个用于二维X射线衍射（XRD）数据不确定性感知持续解释的概念验证框架。该框架集成了多个成熟的组件：基于最大均值差异（MMD）的领域自适应方法，用于减小模拟数据与实验数据之间的差异；采用谱归一化神经高斯过程的卷积神经网络（CNN-SNGP），用于预测不确定性估计；以及结合知识蒸馏的重放机制，用于模型的顺序更新。我们使用模拟数据和先前收集的晶体……（摘要原文在此处被截断）实验数据对该框架进行了评估。

    arXiv:2611.17547v1 Announce Type: cross  Abstract: Autonomous experimentation systems require characterization models that can interpret heterogeneous measurements and maintain predictive performance as data accumulate. Developing such models involves addressing domain mismatch between data sources, overconfident predictions on unfamiliar inputs, and catastrophic forgetting during sequential updates. In this work, we present a proof-of-concept framework for uncertainty-aware continual interpretation of two-dimensional X-ray diffraction (XRD) data. The framework integrates established components: maximum mean discrepancy-based domain adaptation to reduce simulation-to-experiment discrepancies, a convolutional neural network with a spectral-normalized neural Gaussian process (CNN--SNGP) for predictive uncertainty estimation, and replay with knowledge distillation for sequential model updating. We evaluate the framework using simulated and previously-collected experimental data for crysta
    
[^21]: 面向动力系统的多保真度算子推断方法

    Multifidelity Operator Inference for Dynamical Systems

    [https://arxiv.org/abs/2611.17503](https://arxiv.org/abs/2611.17503)

    本文提出MF-OpInf方法，通过受控制变量法启发的多保真度最小二乘公式，将有限的高保真度快照与丰富的低保真度快照相结合，为非线性动力系统学习更准确稳定的非侵入式降阶模型。

    

    本文提出了一种多保真度的算子推断方法，用于学习非线性动力系统的非侵入式、基于物理的降阶模型。OpInf通过求解最小二乘回归问题，从快照数据的投影中学习降阶算子。然而，生成高保真度快照的计算成本可能非常昂贵，这限制了许多航空航天应用中的参数覆盖范围以及训练轨迹的时间跨度，从而可能导致降阶模型预测不准确或不稳定。本工作提出了MF-OpInf方法，通过一种受控制变量法启发的多保真度最小二乘公式，将有限的高保真度快照与更丰富的低保真度快照相结合，以在多条轨迹上学习单一的一组降阶算子。低保真度轨迹提供了额外的参数覆盖范围，并延伸超出高保真度训练的时间范围。

    arXiv:2611.17503v1 Announce Type: cross  Abstract: This paper presents a multifidelity (MF) formulation of Operator Inference (OpInf) for learning non-intrusive, physics-based reduced-order models (ROMs) of nonlinear dynamical systems. OpInf learns reduced operators from projections of snapshot data by solving a least squares regression problem. However, generating high-fidelity snapshots can be computationally expensive, limiting parameter coverage and the temporal extent of training trajectories for many aerospace applications, which can lead to inaccurate or unstable ROM predictions. This work proposes MF-OpInf, which combines limited high-fidelity snapshots with more plentiful low-fidelity snapshots through a control-variates-inspired MF least squares formulation to learn a single set of reduced operators across multiple trajectories. Low-fidelity trajectories provide additional parameter coverage and extend beyond the high-fidelity training time horizon. The low-fidelity snapshots
    
[^22]: SURFACE：通过标量化引导实现帕累托前沿上的偏好分布匹配

    SURFACE: Preference Distribution Matching on Pareto Fronts via Scalarization Steering

    [https://arxiv.org/abs/2611.17484](https://arxiv.org/abs/2611.17484)

    提出SURFACE方法，通过估计权重网格与帕累托前沿区域的对应关系并自适应更新标量化权重，使有限数量的帕累托解能够匹配期望的偏好分布。

    

    构建有限帕累托表示需要在帕累托前沿（PF）上分配有限数量的解。期望的偏好分布通过在某些PF区域分配更多解、在其他区域分配较少解来规定这种分配方式。标量化方法通过改变目标权重并求解单目标问题来生成帕累托解。然而，标量化权重的分布通常无法转化为PF上相同的帕累托解分布，因为权重到解的映射是非线性的且事先未知。为了应对这种失真，我们提出了SURFACE（基于图表估计的标量化更新以重新分配有限逼近方法），它能够自适应地更新标量化权重以匹配期望的PF分布。SURFACE从当前的标量化解中估计权重网格的每个区域如何对应于PF的相应区域，然后重新分配权重……

    arXiv:2611.17484v1 Announce Type: new  Abstract: Constructing a finite Pareto representation requires allocating a limited number of solutions across the Pareto front (PF). A desired preference distribution specifies this allocation by assigning more solutions to some PF regions and fewer to others. Scalarization generates Pareto solutions by varying objective weights and solving single-objective problems. However, a distribution of scalarization weights generally does not translate into the same distribution of Pareto solutions along the PF, because the weight-to-solution map is nonlinear and unknown in advance.To account for this distortion, we introduce SURFACE (Scalarization Updates for Redistributing Finite Approximations with Chart Estimation}), which adaptively updates scalarization weights to match a desired PF distribution. SURFACE estimates how each region of the weight grid corresponds to an area of the PF from the current scalarized solutions, then redistributes the weights
    
[^23]: 基于低秩扰动的分布鲁棒风险评估

    Distributionally Robust Risk Evaluation with Low-Rank Perturbations

    [https://arxiv.org/abs/2611.17447](https://arxiv.org/abs/2611.17447)

    该论文量化了将分布鲁棒扰动限制为行/列空间共享的低秩矩阵时最坏情况风险的损失，提出了基于曲率校正的投影梯度上升算法来计算最坏情况扰动，并给出了收敛性保证与可计算的风险证书。

    

    我们研究分布在受限于低秩矩阵扰动下的鲁棒风险，其中这些低秩矩阵的行空间和列空间在各观测之间共享，重点关注这种限制会牺牲多少最坏情况风险，以及如何计算受限后的最坏情况。在矩约束或Wasserstein预算下，我们推导了无限制最坏情况风险与低秩最坏情况风险之间差距的定量界限，刻画了差距消失的条件，并确定了最小的无损行和列维度。为了生成最坏情况扰动，我们对曲率校正后的凹问题应用投影梯度上升方法，并推导出可计算的风险证书。我们为校正后的目标函数建立了全局次线性收敛性，并在预算约束充分激活时证明了向最坏情况分布的线性收敛性。

    arXiv:2611.17447v1 Announce Type: cross  Abstract: We study distributionally robust risk under perturbations restricted to low-rank matrices whose row and column spaces are shared across observations, focusing on how much worst-case risk this restriction sacrifices and how to compute the restricted worst case. Under moment or Wasserstein budgets, we derive quantitative bounds on the gap between the unrestricted and low-rank worst-case risks, characterize when the gap vanishes, and identify the smallest lossless row and column dimensions. To generate worst-case perturbations, we apply projected gradient ascent to a curvature-corrected concave problem and derive computable risk certificates. We establish global sublinear convergence for the corrected objective and linear convergence to a worst-case distribution when the budget constraint is sufficiently active.
    
[^24]: 枚举后验支撑先验：有限积分跨范畴测量态射的一阶误差

    Enumerating Posterior-Supported Priors: First-Order Error of Finite-Integral Cross-Categorical Measurement Morphisms

    [https://arxiv.org/abs/2611.17390](https://arxiv.org/abs/2611.17390)

    提出枚举后验支撑先验框架，基于 Quot->MorTop->Set 三层范畴链显式记录底层谓词支撑、可逆性与误差所在层级，解决标准测度论 a.e. 约定下零测度支撑隐藏域外误差无法被显式计入的问题。

    

    在生成建模中，标准测度论采用几乎处处（a.e.）约定，这合理地忽略了由零测度支撑集所携带的任何误差分量：单个零测度元素可以将域外误差隐藏于零支撑内部，而永远不会被显式计入。本文的贡献不是提出一种新的分布距离，而是一种可验证的结构，该结构明确写出了哪个底层谓词支撑着高层对象、它在何时可逆、以及误差驻留在哪一层，由此我们提出了枚举后验支撑先验框架。该框架运行于三层范畴链 Quot->MorTop->Set 之上：前两层不假设任何点元素，并以二元谓词商划分作为对象；副本索引计数测度将一对多集合依次区分为单射，把对象收集从满射提升为商层双射；互斥……

    arXiv:2611.17390v1 Announce Type: cross  Abstract: In generative modeling, standard measure theory adopts the a.e. convention, which legitimately ignores any error component carried by a null-support set: a single null element can hide out-of-domain error inside null support and never be explicitly accounted for. The contribution of this paper is not a new distributional distance, but rather a verifiable structure that writes down which underlying predicate supports a high-level object, when it is invertible, and at which layer the error resides, from which we propose the enumerating posterior-supported prior framework. The framework operates on a three-layer categorical chain Quot->MorTop->Set: the first two layers posit no point elements and take binary-predicate quotient partitions as objects; the replica-index counting measure successively discriminates one-to-many collections into injections, lifting object collection from surjection to a quotient-layer bijection; the mutual exclu
    
[^25]: 弱支持下的轨迹密度比：跨科学的综述

    Trajectory Density Ratios under Weak Support: A Survey across the Sciences

    [https://arxiv.org/abs/2611.11985](https://arxiv.org/abs/2611.11985)

    本文是一篇跨学科综述，系统梳理了轨迹密度比这一统一量在强化学习、分子模拟、气候归因和语言模型后训练等领域的核心作用，并分析了其在支撑薄弱（弱支持）情形下的存在性条件、方差爆炸与无定义等根本性失效问题。

    

    arXiv:2611.11985v2 公告类型：replace-cross。摘要：科学领域的许多问题需要通过两个随机过程生成的序列来对它们进行比较。强化学习问的是：从旧策略的轨迹出发，一个新策略会带来怎样的回报；分子模拟问的是：从未结合态的样本出发，配体的结合能力提高了多少；气候归因问的是：在变暖背景下，一场热浪发生了怎样的偏移；语言模型后训练问的是：从记录下来的交互数据出发，部署后的智能体会做出什么行为。在所有这些情形中，有一个量可以同时回答上述所有问题：两个概率规律在其生成序列上的比值，即轨迹密度比。这个比值只在很强的条件下才存在。当一个过程到达了另一个过程无法到达的状态时，该比值不存在；当两者的重叠区域很薄时，其方差会随序列长度不受控制地增长；而当两者集中于不同的低维集合上时，该比值则完全无定义。弱支持（原文摘要在此处截断）

    arXiv:2611.11985v2 Announce Type: replace-cross  Abstract: Many problems across the sciences compare two stochastic processes through the sequences they generate. Reinforcement learning asks what a new policy would return from the trajectories of an old one; molecular simulation asks how much more a ligand binds from samples of the unbound state; climate attribution asks how a heat wave shifted under warming; and language-model post-training asks what a deployed agent will do from logged interactions. In every case, one quantity would answer all these questions at once: the ratio of the two probability laws over their generated sequences, the trajectory density ratio. This ratio exists only under strong conditions. When one process reaches states the other cannot, the ratio does not exist; when their overlap is thin, its variance grows without control with the sequence length; and when they concentrate on different lower-dimensional sets, the ratio is undefined altogether. Weak support
    
[^26]: 小心你的步伐：Adam的间隔最大化偏差取决于学习率调度

    Watch Your Step: Adam's Margin Maximization Bias Depends on the Learning Rate Schedule

    [https://arxiv.org/abs/2611.01929](https://arxiv.org/abs/2611.01929)

    本文揭示了Adam的隐式偏差依赖于学习率调度——在常见的恒定或弱衰减调度下，Adam对齐次模型不再具有与轨迹无关的间隔最大化偏差。

    

    我们研究了Adam的隐式偏差，以及其在二分类任务中于齐次模型上与符号梯度下降的相似性。或许令人惊讶的是，我们发现这些性质依赖于学习率调度。以往的工作研究了强衰减的学习率调度 $\eta(t)=\mathrm o\left(t^{1/L-1}\right)$（其中 $L$ 为齐次度），并证明了 $\ell_\infty$ 间隔最大化偏差；而我们聚焦于更常见的互补情形，包括恒定调度和弱衰减调度。对于多项式调度 $t^{\alpha}$ 和齐次度 $L$，我们证明当 $\alpha > \frac{1}{L} - \frac{1}{2}$ 时，Adam对深度线性网络不具有与轨迹无关的间隔最大化偏差（在KKT意义下）：首先，在此设置下不存在任何非平凡的、局部Lipschitz的凸隐式正则化器，包括任意平方范数；其次，在宽度为 $1$ 的网络中，该结论无需凸性假设即可成立……

    arXiv:2611.01929v2 Announce Type: replace  Abstract: We study the implicit bias of Adam and its similarity to sign gradient descent on homogeneous models in binary classification tasks. Perhaps surprisingly, we uncover that these depend on the learning rate schedule. While previous works studied strongly decaying learning rate schedules $\eta(t)=\mathrm o \left(t^{1/L-1}\right)$, where $L$ is the homogeneity degree, and showed $\ell_\infty$-margin-maximization biases, we focus on the more common complementary regime, including constant and weakly decaying schedules. For a polynomial schedule $t^{\alpha}$ and homogeneity degree $L$, we show that when $\alpha > \frac 1L - \frac12$, Adam has no trajectory-independent margin-maximization bias for deep linear networks (in the KKT sense): first, there is no non-trivial, locally Lipschitz, convex implicit regularizer in this setting, including any squared norm. Second, on width-$1$ networks, the result holds without requiring convexity, imply
    
[^27]: 基于调和权重的高维在线校准

    High-dimensional online calibration from harmonic weights

    [https://arxiv.org/abs/2610.07740](https://arxiv.org/abs/2610.07740)

    本文提出了一个基于调和权重、对过去结果进行调和平滑的简单在线校准算法，首次在高维多结果预测中以 $d^{O(1/\varepsilon)}$ 轮实现 $\varepsilon$-校准，将此前结果的维度依赖性指数级降低。

    

    我们研究了在任意凸集 $Y\subseteq\mathbb{R}^d$ 上、相对于任意误差范数 $\|\cdot\|_{L}$ 的多维预测在线校准问题。对于同时预测 $d$ 个二元结果（$Y=[0,1]^d$）的情形，我们给出了首个在每个固定精度下都能以关于 $d$ 为多项式的轮数实现 $\varepsilon$-校准的算法。该算法需要 $d^{O(1/\varepsilon)}$ 轮，相比此前界中的维度依赖性实现了指数级改进。对于多类别预测（$Y=\Delta_d$），我们获得了相同的 $d^{O(1/\varepsilon)}$ 速率，改进了 Peng 以及 Fishelson 等人给出的 $d^{\widetilde{O}(1/\varepsilon^2)}$ 界。我们的算法非常简单：在每一轮，它输出一个对过去结果进行调和平滑后的调和加权分布。同一个算法适用于所有预测集和范数。更一般地，它在 $\exp(O(\gamma(Y,L)/\varepsilon$……（原文摘要在此处截断）轮后即可实现 $\varepsilon$-校准。

    arXiv:2610.07740v1 Announce Type: cross  Abstract: We study the online calibration of multidimensional forecasts over an arbitrary convex set $Y\subseteq\mathbb{R}^d$ relative to an arbitrary error norm $\|\cdot\|_{L}$. For forecasting $d$ binary outcomes simultaneously ($Y=[0,1]^d$), we give the first algorithm that achieves $\varepsilon$-calibration in a number of rounds that is polynomial in $d$ for every fixed accuracy. It requires $d^{O(1/\varepsilon)}$ rounds, exponentially improving the dimension dependence of previous bounds. For multi-class forecasting ($Y=\Delta_d$), we obtain the same $d^{O(1/\varepsilon)}$ rate, improving the $d^{\widetilde{O}(1/\varepsilon^2)}$ bounds of Peng and Fishelson et al.   Our algorithm is simple: on each round, it outputs a harmonically weighted distribution over harmonically smoothed past outcomes. The same algorithm works for every forecast set and norm. More generally, it achieves $\varepsilon$-calibration after $\exp(O(\gamma(Y,L)/\varepsilon
    
[^28]: 具有1比特反馈的近最优固定置信度最优臂识别

    Nearly Optimal Fixed-Confidence Best-Arm Identification with 1-Bit Feedback

    [https://arxiv.org/abs/2610.02771](https://arxiv.org/abs/2610.02771)

    本文在严格1比特反馈约束下提出了近最优的固定置信度最优臂识别算法，通过随机化阈值查询与自适应截断技术实现了间隙自适应的样本复杂度，并给出了相匹配的信息论下界。

    

    我们研究在严格1比特反馈约束下的固定置信度最优臂识别问题。在每一轮中，学习者选择一个臂和一个查询集合，并且仅接收一个比特，该比特指示采样的奖励是否属于该集合。我们考虑一种具有逐臂定位的无分布有限方差设置，在这种设置下，直接的经验均值估计不再可用，截断处理变得不可避免。我们首先基于随机化阈值查询和截断尾积分恒等式，构建了一个时间一致的1比特均值估计基元。随后，我们将该基元嵌入到候选-挑战者式的最优臂识别算法中。固定截断算法提供了简单的任意时刻（ε,δ)-PAC保证，而分阶段自适应截断算法则将截断水平与当前分辨率相匹配，从而产生了间隙自适应的样本复杂度。我们还证明了一个K臂最坏情况的信息论下界（摘要原文在此处截断）。

    arXiv:2610.02771v1 Announce Type: cross  Abstract: We study fixed-confidence best-arm identification under strict 1-bit feedback constraints. At each round, the learner selects an arm and a query set, and receives only a single bit indicating whether the sampled reward belongs to that set. We consider a distribution-free finite-variance setting with arm-wise localization, where direct empirical mean estimation is no longer available and clipping becomes unavoidable. We first formulate a time-uniform 1-bit mean-estimation primitive based on randomized threshold queries and a clipped tail-integral identity. We then embed this primitive into candidate-challenger best-arm identification algorithms. A fixed-clipping algorithm gives a simple anytime $(\epsilon,\delta)$-PAC guarantee, while a phased adaptive-clipping algorithm matches the clipping level to the current resolution and yields a gap-adaptive sample complexity. We also prove a $K$-arm worst-case information-theoretic lower bound s
    
[^29]: 基于预测状态的无穷记忆过程生成式序列建模

    Generative sequence modeling for infinite memory processes via predictive states

    [https://arxiv.org/abs/2609.38524](https://arxiv.org/abs/2609.38524)

    本文提出一种基于预测状态的生成式序列建模新方法，能够处理具有无穷记忆的随机过程，当过去历史可被压缩为低维充分统计量时实现快速收敛，且估计问题的统计复杂度仅由预测状态空间的内在维数决定。

    

    我们考虑估计多元随机过程的一步预测条件分布。许多现有方法依赖于有限范围记忆、稀疏性或可加性等假设，这些假设对于具有长程非线性交互作用的过程可能并不适用。然而，在没有这些结构性假设的情况下，由于维数灾难，非参数估计面临巨大挑战。为了应对这一挑战，我们提出了一种基于过程预测状态的新估计方法，该方法可以处理可能具有无穷范围记忆的过程。我们证明了当过去的历史信息能够被压缩成一个足以预测未来的低维统计量时，我们的估计器能够达到快速收敛速度。具体而言，我们证明了该估计问题的统计复杂度由预测状态空间的内在维数决定。我们为基于该方法的一种具体实现建立了理论保证（摘要在此处被截断）。

    arXiv:2609.38524v1 Announce Type: new  Abstract: We consider estimating the one-step-ahead conditional distribution of a multivariate stochastic process. Many existing approaches rely on assumptions such as finite-range memory, sparsity, or additivity, which can be poorly suited to processes with long-range nonlinear interactions. However, without such structural assumptions, nonparametric estimation is challenging due to the curse of dimensionality. To address this challenge, we introduce a new estimation approach based on the predictive states of a process, possibly with infinite-range memory. We show that our estimator achieves fast convergence rates when the past history can be compressed into a low-dimensional statistic that is sufficient for predicting the future. Specifically, we show that the statistical complexity of the estimation problem is determined by the intrinsic dimension of the predictive state space. We establish guarantees for an instantiation of our method based on
    
[^30]: 面向个性化动态治疗方案的风险感知分位数学习

    Risk-Aware Quantile Learning for Personalized Dynamic Treatment Regimes

    [https://arxiv.org/abs/2608.05434](https://arxiv.org/abs/2608.05434)

    提出风险感知分位数动态治疗方案（RQDTR）这一统一框架，在优化累积潜在结果分位数的同时显式纳入治疗相关风险，并通过基于角度的公式化方法实现多类别治疗决策规则的联合学习。

    

    序贯临床决策往往不仅仅是最大化平均疗效。临床医生可能需要同时优化结果分布中与临床相关的尾部特征、控制治疗相关风险，并在多种治疗选项中进行选择。现有的分位数动态治疗方案（DTR）方法能够捕捉治疗结果的分布特征，但主要局限于仅以疗效为目标和二元治疗的场景。为了解决这些局限，我们提出了风险感知分位数动态治疗方案（RQDTR），这是一个统一框架，在优化累积潜在结果的预先指定分位数的同时，显式地纳入治疗相关风险。我们还开发了一种基于角度的公式化方法，用于跨多个治疗类别联合学习决策规则。我们的框架包含三个可解释的子类：仅疗效的分位数学习、基于约束的学习（原文此处截断）……

    arXiv:2608.05434v2 Announce Type: replace-cross  Abstract: Sequential clinical decision-making often involves more than maximizing average efficacy. Clinicians may need to simultaneously optimize clinically relevant tails of the outcome distribution, control treatment-related risk, and choose among multiple treatment options. Existing quantile dynamic treatment regime (DTR) methods capture distributional features of treatment outcomes but remain largely restricted to efficacy-only objectives and binary treatments. To address these limitations, we propose Risk-Aware Quantile Dynamic Treatment Regimes (RQDTR), a unified framework that optimizes a prespecified quantile of the cumulative potential outcome while explicitly incorporating treatment-related risk. We also develop an angle-based formulation for jointly learning decision rules across multiple treatment categories. Our framework includes three interpretable subclasses: efficacy-only quantile learning, constraint-based learning wit
    
[^31]: 同策略交互何时有益？基于价值的模仿学习中的表征权衡

    When Does On-Policy Interaction Help? Representational Tradeoffs in Value-Based Imitation Learning

    [https://arxiv.org/abs/2607.29617](https://arxiv.org/abs/2607.29617)

    该论文发现，在基于价值的模仿学习中，与专家进行同策略交互能放宽对学习者的表征要求——只需模型能实现专家的价值函数，而无需实现要求更严格的专家策略本身。

    

    模仿学习（IL）——即训练智能体从演示中复制专家行为——是机器人技术到语言模型训练等诸多应用的基础。众所周知，行为克隆（BC）等标准方法存在误差累积和性能停滞的问题，尤其当学习者无法完美表征专家策略时（例如在蒸馏场景中这是典型情况）。两种干预措施在经验上被广泛认为可以提升性能：沿学习者自身轨迹交互式地查询专家，以及在生成策略的过程中使用价值函数估计。我们研究了这些改进的本质及其可能令人惊讶的相互作用。我们的主要发现是，与专家的交互放宽了对学习者的表征要求：只需要一个能够实现专家价值函数的模型，从而绕过了（通常更严格的）实现专家策略的要求。（注：摘要原文在此处被截断）

    arXiv:2607.29617v2 Announce Type: replace-cross  Abstract: Imitation learning (IL)---training an agent to replicate expert behavior from demonstrations---underpins applications from robotics to language model training. Standard approaches such as Behavior Cloning (BC) are known to suffer from compounding errors and performance plateaus, particularly when the learner cannot perfectly represent the expert's policy (as is typical, e.g., in distillation). Two interventions are widely understood empirically to improve performance: querying the expert interactively along the learner's own trajectories, and using value function estimation en route to generating a policy.   We investigate the nature of these improvements and their potentially surprising interplay. Our main finding is that expert interaction relaxes the representational demands on the learner: one only needs a model capable of realizing the expert's value function, bypassing the (often stricter) requirement of realizing the exp
    
[^32]: 解耦连续时间潜在动力学：通过扩散系数变化实现潜在随机微分方程的可辨识性

    Disentangling Continuous-Time Latent Dynamics: Identifiability of Latent SDEs via Diffusion Shifts

    [https://arxiv.org/abs/2606.28228](https://arxiv.org/abs/2606.28228)

    该论文证明了在未知非线性观测下，仅利用多个环境间扩散协方差的变化即可识别加性噪声潜在SDE的潜在坐标（至置换、缩放和常数平移），且无需对漂移项作任何稀疏性假设。

    

    面向时间序列的因果表征学习在离散时间潜在因果模型中已建立了较强的可辨识性结果，但连续时间潜在随机微分方程（SDE）模型中的可辨识性问题在很大程度上仍悬而未决。我们利用环境引起的扩散协方差变化来填补这一空白。我们研究了通过未知非线性微分同胚观测到的加性噪声潜在SDE，其漂移项在不同环境间共享，而扩散协方差则因环境而异。我们证明：两个具有逐坐标成对不同方差比的对角扩散机制，可以在无需对漂移项施加任何稀疏性假设的条件下，将潜在坐标识别至置换、逐坐标缩放以及可能的常数平移的等价类内。我们首先针对线性Ornstein-Uhlenbeck系统证明了这一结果，随后将其推广至一般的加性噪声潜在SDE。在温和的光滑性条件下，瞬时漂移-雅可比因果图是可辨识的……

    arXiv:2606.28228v2 Announce Type: replace-cross  Abstract: Causal representation learning for time series has developed strong identifiability results in discrete-time latent causal models, but identifiability in continuous-time latent stochastic differential equation (SDE) models remains largely open. We address this gap using environment-induced shifts in diffusion covariance. We study additive-noise latent SDEs observed through an unknown nonlinear diffeomorphism, with shared drift but environment-specific diffusion covariance. We show that two diagonal diffusion regimes with pairwise distinct coordinate-wise variance ratios identify the latent coordinates up to permutation, coordinate-wise scaling, and a possible constant shift, without any sparsity assumption on the drift. We first prove this result for linear Ornstein-Uhlenbeck systems and then extend it to general additive-noise latent SDEs. Under mild smoothness, the instantaneous drift-Jacobian causal graph is identifiable up 
    
[^33]: 序列模型的精确约束解码：控制器增强隐马尔可夫模型及更多

    Exact Constrained Decoding for Sequence Models: Controller-Augmented Hidden Markov Models and Beyond

    [https://arxiv.org/abs/2606.13850](https://arxiv.org/abs/2606.13850)

    提出控制器增强隐马尔可夫模型（CHMMs）框架，将路径级约束编译为有限状态控制器，借助增广链上的标准递推算法，在离散与连续时间中实现精确的约束后验计算与最大后验路径解码。

    

    隐马尔可夫模型是序列推断的基础，然而在路径级约束（如先后顺序要求、访问基数或单调状态推进）上进行条件化时，可能引入需要额外动态规划内存的长程依赖。为解决这一问题，我们提出了控制器增强隐马尔可夫模型（CHMMs），这是一个将每个约束编译为追踪足够有限历史的有限状态控制器的框架。此后，在增广链上运行标准的向前-向后算法和Viterbi递推，即可在离散时间中计算精确的约束后验和最大后验（MAP）路径，并且通过均匀化方法，在连续时间的观测时刻计算增广状态序列的最大后验估计。我们建立了多项理论保证，同时为11个约束族提供了一套控制器编码目录，使该框架得以实际应用。在实验方面，我们评估了该类方法……（摘要原文在此处截断）

    arXiv:2606.13850v2 Announce Type: replace-cross  Abstract: Hidden Markov models are foundational for sequential inference, yet conditioning on pathwise constraints such as precedence requirements, visitation cardinalities, or monotonic state progression can introduce long-range dependencies that require additional memory for dynamic programming. To deal with this, we present Controller-Augmented Hidden Markov Models (CHMMs), a framework that compiles each constraint into a finite-state controller tracking a sufficient finite history, after which standard forward--backward and Viterbi recursions on the augmented chain compute exact constrained posteriors and maximum a posteriori paths in discrete time and, through uniformization, maximum a posteriori augmented-state sequences at observation times in continuous time. While we establish several theoretical guarantees, a catalog of controller encodings for 11 constraint families operationalizes our framework. Empirically, we evaluate class
    
[^34]: Express 语言建模

    Express Language Modeling

    [https://arxiv.org/abs/2606.10944](https://arxiv.org/abs/2606.10944)

    Express 是一种将非因果注意力近似转换为具有匹配保证的因果近似的工具，与 Thinformer 结合后实现了已知最佳的因果注意力近似保证，并通过高效的 Triton 实现显著超越 FlashAttention 2，解决了语言建模中的长上下文预填充、KV 缓存压缩和长文本解码等四大资源瓶颈。

    

    我们介绍了一种新工具 Express，用于将非因果注意力近似转换为具有匹配近似保证的因果近似。当与最先进的 Thinformer 近似相结合时，Express 改进了已知最佳的因果注意力保证：对于长度为 n 的序列，在仅使用 O(s) 内存和 O(s² log²(n)) 压缩开销的情况下，实现了 log^{3/2}(n)/s 的近似误差。我们将这些进展与高效的 I/O 感知 Triton 实现相结合，展示了相对于 FlashAttention 2 的显著加速，并利用 Express 克服了语言建模流程中的四个资源瓶颈：长上下文预填充、KV 缓存压缩、长文本内存受限解码以及长文本计算受限解码。

    arXiv:2606.10944v2 Announce Type: replace  Abstract: We introduce a new tool, Express, for converting a non-causal attention approximation into a causal approximation with matching approximation guarantees. When combined with the state-of-the-art Thinformer approximation, Express improves upon the best known causal attention guarantees, delivering $\log^{3/2}(n)/s$ approximation error with only $O(s)$ memory and $O(s^2 \log^2(n))$ compression overhead for a sequence of length $n$. We pair these developments with an efficient I/O-aware Triton implementation, demonstrate substantial speedups over FlashAttention 2, and use Express to overcome four resource bottlenecks in the language modeling pipeline: long-context prefill, KV cache compression, long-form memory-constrained decoding, and long-form compute-constrained decoding.
    
[^35]: 基于核的势均场博弈与无偏随机傅里叶U统计量

    Kernel-based potential mean-field games with unbiased random Fourier $U$-statistics

    [https://arxiv.org/abs/2605.29371](https://arxiv.org/abs/2605.29371)

    本文针对运行成本与终端成本均由核MMD惩罚刻画的势均场博弈，提出了一种基于无偏随机傅里叶U统计量的计算框架，以神经网络参数化漂移并通过随机梯度下降训练，同时证明了在惩罚发散及显式耦合条件下向终端约束问题的收敛性。

    

    我们研究势均场博弈的一个子类，其中运行交互成本和终端目标成本均通过再生核最大均值差异（MMD）惩罚来表示，并开发了一个利用这种核结构的计算框架。两种成本均通过一种无偏且计算成本随批大小线性增长的随机傅里叶U统计量表示，从有限样本经验分布中进行估计。受控扩散的漂移由神经网络参数化，并通过随机梯度下降进行训练。对于种群近极小化器，我们证明了当惩罚发散时收敛到终端约束问题，并表明在惩罚、随机特征数量和样本大小的显式耦合条件下，当学习到的控制策略在独立的有限样本上评估时，几乎必然恢复到相同的极限。该框架包括核MMD-（原文在此处截断）

    arXiv:2605.29371v2 Announce Type: replace-cross  Abstract: We study the subclass of potential mean-field games in which the running interaction cost and the terminal target cost are both expressed through reproducing-kernel maximum mean discrepancy (MMD) penalties, and develop a computational framework that exploits this kernel structure. Both costs are estimated from finite-sample empirical distributions using a random Fourier U-statistic representation that is unbiased and has linear cost in the batch size. The drift of the controlled diffusion is parametrized by a neural network and trained via stochastic gradient descent. For population near-minimizers we prove convergence to the terminal-constrained problem as the penalty diverges, and show that the same limit is recovered almost surely when the learned controls are evaluated on independent finite samples under explicit coupling conditions on the penalty, random-feature count and sample size. The framework includes the kernel-MMD-
    
[^36]: 非平稳环境中的自适应校准

    Adaptive Calibration in Non-Stationary Environments

    [https://arxiv.org/abs/2605.11490](https://arxiv.org/abs/2605.11490)

    该论文提出了能够自动适应环境非平稳程度的在线预测校准算法，其校准误差可在独立同分布与对抗性两种情形之间平滑过渡，并在多种校准指标下提供了自适应保证。

    

    在现代人工智能系统中，做出经过校准的在线预测是一个核心挑战。现有文献大多关注完全对抗性的环境，其中结果可能是任意的，这导致算法过于保守，在更温和的环境中（例如结果近乎平稳的情形下）可能表现次优。这一空白引出了一个自然的问题：我们能否设计出这样的在线预测算法，使其校准误差能够自动适应环境的非平稳程度，并在独立同分布（i.i.d.）情形与对抗性情形之间平滑过渡？我们对这一问题给出了肯定的回答，并开发了一套在多种校准指标下均能实现自适应校准保证的算法。具体而言，设 T 为轮数，W 为环境中未知的独立同分布段数，C∈[0,T] 为另一个未知的非平稳性度量，其定义为最小的 ℓ1 偏差……（摘要原文在此处截断）

    arXiv:2605.11490v3 Announce Type: replace  Abstract: Making calibrated online predictions is a central challenge in modern AI systems. Much of the existing literature focuses on fully adversarial environments where outcomes may be arbitrary, leading to conservative algorithms that can perform suboptimally in more benign settings, such as when outcomes are nearly stationary. This gap raises a natural question: can we design online prediction algorithms whose calibration error automatically adapts to the degree of non-stationarity in the environment, smoothly interpolating between i.i.d. and adversarial regimes? We answer this question in the affirmative and develop a suite of algorithms that achieve adaptive calibration guarantees under multiple calibration measures. Specifically, with $T$ being the number of rounds, $W$ being the unknown number of i.i.d. segments of the environment, and $C\in[0,T]$ being another unknown non-stationary measure defined as the minimal $\ell_1$ deviation o
    
[^37]: mlr3torch：基于mlr3和torch的R语言深度学习框架

    mlr3torch: A Deep Learning Framework in R based on mlr3 and torch

    [https://arxiv.org/abs/2604.18152](https://arxiv.org/abs/2604.18152)

    本文介绍了R包mlr3torch，这是一个构建于torch之上、深度集成mlr3生态系统的可扩展深度学习框架，它简化了神经网络的定义、训练和评估，并允许用户通过图语言在单一图中定义包含预处理、数据增强和网络架构的完整建模工作流程。

    

    深度学习（DL）已成为现代机器学习（ML）实践的基石。我们介绍R包mlr3torch，它是mlr3生态系统的一个可扩展的深度学习框架。该框架构建于torch包之上，简化了针对表格数据和通用张量（例如图像）进行分类和回归任务的神经网络的定义、训练与评估。该包实现了预定义的网络架构，并且可以轻松地将torch模型转换为mlr3学习器。它还允许用户将神经网络定义为图。这种表示方式基于mlr3pipelines中定义的图语言，使用户能够在单个图中定义完整的建模工作流程，包括预处理、数据增强和网络架构。通过与mlr3生态系统的集成，该包支持便捷的重采样、基准测试、预处理等功能。我们对该包的设计进行了说明……（原文摘要在此处截断）

    arXiv:2604.18152v2 Announce Type: replace-cross  Abstract: Deep learning (DL) has become a cornerstone of modern machine learning (ML) praxis. We introduce the R package mlr3torch, which is an extensible DL framework for the mlr3 ecosystem. It is built upon the torch package, and simplifies the definition, training, and evaluation of neural networks for both tabular data and generic tensors (e.g., images) for classification and regression. The package implements predefined architectures, and torch models can easily be converted to mlr3 learners. It also allows users to define neural networks as graphs. This representation is based on the graph language defined in mlr3pipelines and allows users to define the entire modeling workflow, including preprocessing, data augmentation, and network architecture, in a single graph. Through its integration into the mlr3 ecosystem, the package allows for convenient resampling, benchmarking, preprocessing, and more. We explain the package's design an
    
[^38]: 用于生成建模的变分测量量子计算中的经典资源最小化

    Minimizing classical resources in variational measurement-based quantum computation for generative modeling

    [https://arxiv.org/abs/2604.11578](https://arxiv.org/abs/2604.11578)

    本文针对变分测量量子计算（VMBQC）中信道模型参数量是幺正模型两倍的局限，提出了最小化经典资源消耗的方案，使生成建模中的经典处理开销得到显著降低。

    

    测量量子计算是一种量子信息处理框架，其中计算任务通过对高度纠缠的资源态执行单量子比特测量来实现。由于量子测量结果的不确定性，这些操作的随机结果若不加校正，将形成一个变分量子信道族。传统上，这种随机性需要通过经典处理加以校正，以确保确定性的幺正计算。最近，变分测量量子计算（VMBQC）被提出，利用这种测量引起的随机性在生成建模中获得优势。该方法的一个局限是，相应的信道模型相比幺正模型具有两倍的参数量，其规模为 $N \times D$，其中 $N$ 是逻辑量子比特的数量（宽度），$D$ 是 VMBQC 模型的深度。这可能……

    arXiv:2604.11578v2 Announce Type: replace-cross  Abstract: Measurement-based quantum computation (MBQC) is a framework for quantum information processing in which a computational task is carried out through one-qubit measurements on a highly entangled resource state. Due to the indeterminacy of the outcomes of a quantum measurement, the random outcomes of these operations, if not corrected, yield a variational quantum channel family. Traditionally, this randomness is corrected through classical processing in order to ensure deterministic unitary computations. Recently, variational measurement-based quantum computation (VMBQC) has been introduced to exploit this measurement-induced randomness to gain an advantage in generative modeling. A limitation of this approach is that the corresponding channel model has twice as many parameters compared to the unitary model, scaling as $N \times D$, where $N$ is the number of logical qubits (width) and $D$ is the depth of the VMBQC model. This can
    
[^39]: 自动微分数据同化：状态、动力学与滤波算法的协同学习

    Auto-differentiable data assimilation: Co-learning of states, dynamics, and filtering algorithms

    [https://arxiv.org/abs/2603.20891](https://arxiv.org/abs/2603.20891)

    本文提出了一种自动微分滤波框架，利用理论驱动的损失函数和基于梯度的优化，实现对状态、动力学模型以及滤波算法参数的联合协同学习，从而摆脱传统数据同化中昂贵的参数调优和对精确动力学模型的依赖。

    

    数据同化算法通过部分观测来估计动力学系统的状态，而这些算法的成功应用依赖于代价高昂的参数调优以及采用精确的动力学模型。本文提出了一个框架，通过我们称之为“自动微分滤波”的过程，在数据同化中联合学习状态、动力学以及滤波算法的参数。该框架利用一个具有理论依据的损失函数，使得能够借助基于梯度的优化（使用自动微分）从部分且有噪声的观测中进行学习。我们进一步展示了如何在den框架内学习或调优几种著名的数据同化方法。为了凸显自动微分滤波的多功能性，我们在跨越多个科学领域的动力系统上进行了实验，例如航空航天工程中的 Clohessy-Wiltshire 方程等。

    arXiv:2603.20891v2 Announce Type: replace-cross  Abstract: Data assimilation algorithms estimate the state of a dynamical system from partial observations, where the successful performance of these algorithms hinges on costly parameter tuning and on employing an accurate model for the dynamics. This paper introduces a framework for jointly learning the state, dynamics, and parameters of filtering algorithms in data assimilation through a process we refer to as auto-differentiable filtering. The framework leverages a theoretically motivated loss function that enables learning from partial, noisy observations via gradient-based optimization using auto-differentiation. We further demonstrate how several well-known data assimilation methods can be learned or tuned within this framework. To underscore the versatility of auto-differentiable filtering, we perform experiments on dynamical systems spanning multiple scientific domains, such as the Clohessy-Wiltshire equations from aerospace engi
    
[^40]: 大规模因果推断中基于核方法的估计量的快速不确定性量化

    Fast Uncertainty Quantification for Kernel-Based Estimators in Large-Scale Causal Inference

    [https://arxiv.org/abs/2603.13662](https://arxiv.org/abs/2603.13662)

    本文通过扩展因果小自助法（cBLB），实现了大规模因果推断中基于核方法的估计量的快速不确定性量化，解决了传统自助法在数百万级样本数据集上计算不可行的问题。

    

    核方法在因果推断中被广泛应用于处理效应估计、策略评估和策略学习等任务。自助法凭借其广泛的适用性，成为不确定性量化的标准工具。随着越来越大规模的数据集变得可用，例如来自美国国家生命统计系统（NVSS）的2023年美国出生数据，其中包含3,605,081例登记出生（经过队列筛选后，其中2,980,739例构成我们的分析样本），这些方法的计算需求大幅增加。众所周知，核方法随样本量增大时的扩展性较差，而自助法所需的重复重新拟合进一步加剧了这一局限。因此，在大规模场景下，基于自助法的核估计量推断在计算上可能变得不可行。在本文中，我们通过扩展因果小自助法（cBLB）来应对这些挑战……

    arXiv:2603.13662v2 Announce Type: replace-cross  Abstract: Kernel methods are widely used in causal inference for tasks such as treatment effect estimation, policy evaluation, and policy learning. The bootstrap is a standard tool for uncertainty quantification because of its broad applicability. As increasingly large datasets become available, such as the 2023 U.S. Natality data from the National Vital Statistics System (NVSS), which includes 3,605,081 registered births (of which 2,980,739 form our analytic sample after cohort restrictions), the computational demands of these methods increase substantially. Kernel methods are known to scale poorly with sample size, and this limitation is further exacerbated by the repeated re-fitting required by the bootstrap. As a result, bootstrap-based inference for kernel-based estimators can become computationally infeasible in large-scale settings. In this paper, we address these challenges by extending the causal Bag of Little Bootstraps (cBLB) 
    
[^41]: 关于Wasserstein与Gromov-Wasserstein最优传输计划的稀疏性、极值结构与单调性性质

    On sparsity, extremal structure, and monotonicity properties of Wasserstein and Gromov-Wasserstein optimal transport plans

    [https://arxiv.org/abs/2602.16265](https://arxiv.org/abs/2602.16265)

    本文系统梳理了Gromov-Wasserstein最优传输的关键性质，证明了在条件负定/正定（CND/CPD）条件下存在稀疏且支撑于置换上的最优传输计划，并给出不满足这些假设时置换并非最优的反例。

    

    本简报对Gromov-Wasserstein（GW）距离的一些关键性质进行了自成一体的综述，并与标准的线性最优传输（OT）框架进行比较。更具体地说，我探讨了以下问题：GW最优传输计划是否稀疏？在什么条件下存在支撑于置换上的最优计划？最优计划是否满足某种形式的循环单调性？特别是，我给出了条件负定/正定（CND/CPD）性质，并证明当这些性质成立时，存在稀疏且支撑于置换上的GW最优计划。在最后一节中，我还给出了在去掉CND或CPD假设时置换并非最优的例子。

    arXiv:2602.16265v3 Announce Type: replace-cross  Abstract: This note gives a self-contained overview of some key properties of the Gromov-Wasserstein (GW) distance, compared with the standard linear optimal transport (OT) framework. More specifically, I explore the following questions: are GW optimal transport plans sparse? under what conditions does there exist an optimal plan supported on a permutation? do optimal plans satisfy a form of cyclical monotonicity? In particular, I present the conditionally negative/positive definite (CND/CPD) properties and show that, when they hold, there exist GW optimal plans that are sparse and supported on a permutation. In the last section, I also give examples where no permutation is optimal when the CND or CPD assumption is dropped.
    
[^42]: 从近端算子视角看流匹配

    Flow Matching from Viewpoint of Proximal Operators

    [https://arxiv.org/abs/2602.12683](https://arxiv.org/abs/2602.12683)

    该论文通过扩展Brenier势将最优传输条件流匹配（OT-CFM）的向量场精确表示为与势函数相关的近端算子（去噪器），并借助Lyapunov指数证明其流形稳定性：法向扰动线性收缩而切向扰动被保留。

    

    我们讨论了流形假设下最优传输条件流匹配（OT-CFM）的动力学，其中目标分布的支撑集是一个低维流形。我们首先利用扩展的Brenier势导出了向量场的精确近端表达式；该向量场可由一个去噪器表示，而该去噪器可呈现为与势函数相关联的近端算子。基于向量场的这一显式近端形式，我们利用动力学的Lyapunov指数分析了初始点扰动下的流形稳定性。借助凸分析工具对局部行为进行分析，我们证明动力学在流形法向方向上的扰动呈线性收缩，而切向方向上的扰动则得以保留；这表明流形结构对扰动具有稳定性。为了支持该理论……

    arXiv:2602.12683v3 Announce Type: replace  Abstract: We discuss the dynamics of Optimal Transport Conditional Flow Matching (OT-CFM) under the manifold hypothesis, where the support of the target distribution is a low dimensional manifold. We first derive an exact proximal expression of the vector field, using an extended Brenier potential for the OT; the vector field can be expressed by a denoiser, which is presented as the proximal operator associated with the potential function. Based on this explicit proximal form of the vector field, we analyze the manifold stability for perturbations to initial points, using the Lyapunov exponents of the dynamics. By analyzing the local behavior with the tools in convex analysis, we show that the dynamics contract linearly in the perturbation of the normal directions to the manifold, while perturbations in the tangential directions are preserved; this implies the stability of the manifold structure to the perturbations. To justify the theoretical
    
[^43]: 具有参数化条件分布的深度多元模型

    Deep Multivariate Models with Parametric Conditionals

    [https://arxiv.org/abs/2602.01953](https://arxiv.org/abs/2602.01953)

    该论文提出用每组变量以其余变量为条件的参数化条件概率分布来表示联合概率分布，从而构建几乎可通用于任意下游任务的深度多元模型，其训练等价于通过最大化极限分布的数据似然来学习一个参数化马尔可夫链核。

    

    我们研究面向异构随机变量集合的深度多元模型。在计算机视觉的背景下，此类集合可以由图像、分割结果、图像属性以及潜在变量等组成。在开发此类模型时，大多数现有工作从某个应用任务出发，设计模型组件及其依赖关系以满足所选任务的需求。这种做法的缺点是限制了所得模型在其他下游任务上的适用性。与此不同，我们提出用每组变量以其余变量为条件的条件概率分布来表示联合概率分布。这样构建的模型几乎可以用于任何可能的下游任务。此类模型的学习可以被看作训练一个参数化的马尔可夫链核，即通过最大化其极限分布的数据似然来进行训练。这还具有一个额外的好处，即允许……（原文摘要在此处截断）

    arXiv:2602.01953v2 Announce Type: replace  Abstract: We consider deep multivariate models for heterogeneous collections of random variables. In the context of computer vision, such collections may e.g. consist of images, segmentations, image attributes, and latent variables. When developing such models, most existing works start from an application task and design the model components and their dependencies to meet the needs of the chosen task. This has the disadvantage of limiting the applicability of the resulting model for other downstream tasks. Here, instead, we propose to represent the joint probability distribution by means of conditional probability distributions for each group of variables conditioned on the rest. Such models can then be used for practically any possible downstream task. Their learning can be approached as training a parametrised Markov chain kernel by maximising the data likelihood of its limiting distribution. This has the additional advantage of allowing a 
    
[^44]: 流形近似带来鲁棒的核对齐

    Manifold Approximation leads to Robust Kernel Alignment

    [https://arxiv.org/abs/2510.22953](https://arxiv.org/abs/2510.22953)

    提出流形近似核对齐（MKA），通过将流形局部邻域几何融入核对齐任务，去除启发式度加权、给出闭式对齐分数及马尔可夫概率解释，并证明其对邻域大小的稳定性，从而实现更鲁棒的表示比较。

    

    中心化核对齐（CKA）是比较表示、判断网络等价性以及开展神经科学研究的一种常用度量。然而，CKA 没有考虑底层的流形结构，并且依赖众多启发式方法，导致其在不同数据规模下表现不一致。在本工作中，我们提出了流形近似核对齐（MKA），将流形几何（即局部邻域）融入到对齐任务中。我们为 MKA 推导了一个理论框架。我们的构造去除了依赖于表示的度加权，得到了闭式的对齐分数，并将每一行赋予了作为局部马尔可夫转移分布的概率解释。我们进一步建立了关于邻域大小的核级别与分数级别的稳定性。我们在合成数据集和真实世界示例上进行了实证评估，以刻画 MKA 并将其与同类方法进行比较。

    arXiv:2510.22953v3 Announce Type: replace-cross  Abstract: Centered kernel alignment (CKA) is a popular metric for comparing representations, determining equivalence of networks, and conducting neuroscience research. However, CKA does not account for the underlying manifold and relies on numerous heuristics that cause it to behave differently at different scales of data. In this work, we propose Manifold-approximated Kernel Alignment (MKA), which incorporates manifold geometry (i.e., local neighborhood) into the alignment task. We derive a theoretical framework for MKA. Our construction removes representation-dependent degree weighting, yields a closed-form alignment score, and gives each row a probabilistic interpretation as a local Markov transition distribution. We further establish kernel- and score-level stability with respect to the neighborhood size. We perform empirical evaluations on synthetic datasets and real-world examples to characterize and compare MKA to its contemporari
    
[^45]: 针对下游天气预报模型的对抗攻击：在热带气旋轨迹预测中的应用

    Adversarial Attacks on Downstream Weather Forecasting Models: Application to Tropical Cyclone Trajectory Prediction

    [https://arxiv.org/abs/2510.10140](https://arxiv.org/abs/2510.10140)

    本文提出了Cyc-Attack对抗攻击方法，通过对上游天气预报模型施加细微扰动来操纵下游热带气旋轨迹预测，克服了检测系统黑盒不可微分和热带气旋样本极度稀缺导致的类别不平衡两大难题。

    

    基于深度学习的天气预报（DLWF）模型利用过去的天气观测数据生成未来的预报，支持广泛的下游应用，其中包括热带气旋（TC）预测。本文研究了这些模型对对抗攻击的脆弱性，即对上游预报施加细微扰动即可改变下游的热带气旋轨迹预测。构建扰动输入以将下游输出引导至攻击者指定的轨迹是一个极具挑战性的问题，原因如下：首先，传统的热带气旋检测系统是不透明、不可微分的黑盒，使得标准的基于梯度的攻击方法无法实施；其次，热带气旋事件的极端稀有性导致了严重的类别不平衡问题，使得难以开发出能够生成与目标轨迹相符的逼真气旋路径的攻击方法。为克服这些限制，我们提出了Cyc-Attack这一新颖的方法。

    arXiv:2510.10140v3 Announce Type: replace  Abstract: Deep learning-based weather forecasting (DLWF) models leverage past weather observations to generate future forecasts, supporting a wide range of downstream applications, including tropical cyclone (TC) prediction. In this paper, we investigate the vulnerability of these models to adversarial attacks, where subtle perturbations to upstream forecasts can alter downstream TC trajectory predictions. Crafting perturbed inputs to steer downstream outputs toward attacker-specified trajectories is a challenging problem for several reasons. First, conventional TC detection systems are opaque, non-differentiable black boxes, making standard gradient-based attacks infeasible. Second, the extreme rarity of TC events leads to a severe class imbalance problem, making it difficult to develop attack methods that produce realistic-looking cyclone paths aligned with target trajectories. To overcome these limitations, we propose Cyc-Attack, a novel me
    
[^46]: ReLU神经网络的环面几何

    Toric geometry of ReLU neural networks

    [https://arxiv.org/abs/2509.05894](https://arxiv.org/abs/2509.05894)

    该论文建立了ReLU神经网络与环面几何之间的联系，定义了ReLU扇、ReLU环面簇和ReLU卡地亚除子等新概念，借助代数几何工具系统研究了网络的精确函数实现问题与表达能力，并揭示了热带几何与环面几何之间的对应关系。

    

    给定一个连续分段线性函数 $f:\mathbb{R}^{n_0} \to \mathbb{R}$ 和一个固定的前馈ReLU神经网络架构，精确函数实现问题问的是该架构的网络何时能够实现 $f$。与之相对，表达能力问题关注的是刻画由给定架构能够精确实现的所有函数。为了系统地解决这些问题，我们在环面几何与ReLU神经网络之间建立了联系，使得代数几何工具能够用于分析这些结构。从一个具有有理权重的无偏ReLU网络出发，我们定义了与该网络相关联的ReLU扇、ReLU环面簇以及ReLU卡地亚除子。这项工作还将热带几何与环面几何联系起来：前者捕捉ReLU网络的组合数据，而后者则恢复其底层的几何结构。作为应用，我们获得了……的完整刻画。

    arXiv:2509.05894v2 Announce Type: replace-cross  Abstract: Given a continuous piecewise linear function $f:\mathbb{R}^{n_0} \to \mathbb{R}$ and a fixed architecture of feedforward ReLU neural networks, the exact function realization problem asks when a network of that architecture realizes $f$. In contrast, expressivity concerns characterizing all functions exactly realizable by a given architecture. To address these questions systematically, we establish a connection between toric geometry and ReLU neural networks, allowing tools from algebraic geometry to analyze the structures. Starting with an unbiased ReLU network with rational weights, we define the ReLU fan, the ReLU toric variety, and the ReLU Cartier divisor associated with the network. This work also relates the tropical and the toric geometry: the former captures combinatorial data of ReLU networks, while the latter recovers the underlying geometric structure. As an application, we obtain a complete characterization of the f
    
[^47]: 在异构噪声下恢复高维数据的欧几里得几何结构

    Recovering the Euclidean Geometry of High-Dimensional Data Under Heterogeneous Noise

    [https://arxiv.org/abs/2507.18520](https://arxiv.org/abs/2507.18520)

    本文证明了对于具有底层低维几何结构的高维数据，即使在异构噪声严重且不均匀地破坏成对欧几里得距离的情况下，仍然可以恢复出其底层干净的欧几里得几何结构。

    

    成对欧几里得距离为高维数据提供了一种简单的几何表示，是包括聚类、可视化和流形学习在内的许多数据分析方法的基础。然而，在现实应用中，这些距离可能受到异构噪声的严重破坏，即不同的数据点会受到不同且可能很大的噪声水平影响。这种噪声会不均匀地扭曲距离，导致对底层数据几何结构的错误表示。在这项工作中，我们探讨是否仍然能够从被破坏的成对距离中恢复出底层的干净欧几里得几何结构。我们对这个问题给出了肯定的回答，表明——也许令人惊讶的是——即使噪声幅度在不同观测之间变化很大，这种恢复在广泛的具有底层低维几何结构的高维场景中也是可能的。我们开发了一种有原则的方法……

    arXiv:2507.18520v3 Announce Type: replace-cross  Abstract: Pairwise Euclidean distances provide a simple geometric representation of high-dimensional data and underlie many data analysis methods, including clustering, visualization, and manifold learning. However, in real-world applications, these distances can be severely corrupted by heterogeneous noise, where different data points are subject to different, and potentially large, noise levels. Such noise distorts the distances inhomogeneously, leading to a misrepresentation of the underlying data geometry. In this work, we ask whether the underlying clean Euclidean geometry can nevertheless be recovered from the corrupted pairwise distances. We answer this question affirmatively, showing that, perhaps surprisingly, such recovery is possible across a broad class of high-dimensional settings with underlying low-dimensional geometry, even when the noise magnitudes vary substantially across observations. We develop a principled approach 
    
[^48]: 面向Wasserstein分布鲁棒优化的熵平滑与梯度近似

    Entropic smoothing and gradient approximation for Wasserstein distributionally robust optimization

    [https://arxiv.org/abs/2506.04948](https://arxiv.org/abs/2506.04948)

    本文证明了Wasserstein分布鲁棒优化的熵平滑正则化变体的近似梯度在正则化参数趋于零时收敛到原问题的次梯度，从而为随机梯度方法提供了收敛保证与收敛速率。

    

    Wasserstein分布鲁棒优化为机器学习中数据分布潜在变化下的模型拟合提供了一个框架。我们研究了该问题的一个正则化变体，其中熵平滑产生了原始目标的采样近似。我们建立了当正则化参数趋于消失时，近似梯度收敛到未正则化目标次梯度的理论结果，从而为随机梯度方法提供了收敛保证。我们在一般假设下获得了定性收敛结果，并在额外的正则性条件下给出了收敛速率。特别地，我们证明了当正则化水平在迭代过程中逐渐降低时，未正则化目标值的收敛速率（直至采样误差）。我们的分析还产生了具有独立意义的结果，即用最大值函数的平滑版本的梯度来近似其次微分。

    arXiv:2506.04948v4 Announce Type: replace-cross  Abstract: Wasserstein distributionally robust optimization offers a framework for model fitting in machine learning under potential shifts in the data distribution. We study a regularized variant of this problem in which entropic smoothing produces a sampled approximation of the original objective. We establish convergence of the approximate gradients to subgradients of the unregularized objective as the regularization parameter vanishes, enabling convergence guarantees for stochastic gradient methods. We obtain qualitative convergence results under general assumptions and provide convergence rates under additional regularity. In particular, we prove rates for the convergence of the unregularized objective values, up to sampling errors, when the regularization level is decreased across iterations. Our analysis also yields results of independent interest on approximating the subdifferentials of maximum functions by gradients of their smoo
    
[^49]: 随机梯度下降的偏差最优界

    Bias-Optimal Bounds for Stochastic Gradient Descent

    [https://arxiv.org/abs/2505.17965](https://arxiv.org/abs/2505.17965)

    本文在仅假设凸性和光滑性的条件下，为随机梯度下降推导出覆盖全部常数步长范围的偏差最优收敛界，并证明在临界步长处最优偏差虽无法精确达到但可被任意逼近。

    

    随机梯度下降（SGD）的非渐近分析通常得到的界可分解为偏差项和方差项。在本工作中，我们聚焦于偏差部分，研究SGD在多大程度上能够匹配确定性梯度下降的最优收敛行为。仅在目标函数满足（强）凸性和光滑性的假设下，我们为常数步长的全部范围 $\gamma L \in (0,2)$ 推导出了新的界。这包括了此前在无额外方差假设条件下未曾被探索过的临界步长和较大步长区间，且同时涵盖凸和强凸两种情形。这些界是偏差最优的，即对于所有非临界步长，偏差项与梯度下降的最坏情形速率相一致。对于临界步长，我们证明无法获得具有最优偏差项的界，但该最优偏差可以被任意接近地逼近。

    arXiv:2505.17965v3 Announce Type: replace-cross  Abstract: The non-asymptotic analysis of Stochastic Gradient Descent (SGD) typically yields bounds that decompose into a bias term and a variance term. In this work, we focus on the bias component and study the extent to which SGD can match the optimal convergence behavior of deterministic gradient descent. Assuming only (strong) convexity and smoothness of the objective, we derive new bounds for the full range of constant step-sizes $\gamma L \in (0,2)$. This includes critical and large step-size regimes that were previously unexplored without additional variance assumptions, in both the convex and strongly convex cases. Those bounds are bias-optimal, in the sense that the bias term coincides with the worst-case rate of gradient descent, for all non-critical step-sizes. For critical step-sizes, we show that no bounds can be obtained with an optimal bias term, but that this optimal bias can be approximated arbitrarily closely. As an imme
    
[^50]: 基于训练后两层ReLU网络的回归中的良性过拟合

    Benign Overfitting for Regression with Trained Two-Layer ReLU Networks

    [https://arxiv.org/abs/2410.06191](https://arxiv.org/abs/2410.06191)

    本文首次证明了通过梯度流训练的有限宽度两层ReLU网络在最小二乘回归中既能实现泛化又会过拟合数据，从而在任意有界回归函数下首次建立了良性过拟合结果。

    

    我们研究了使用带ReLU激活函数的两层全连接神经网络、通过梯度流训练的最小二乘回归问题。我们的第一个结果是一个泛化结果，除了要求底层回归函数和噪声有界之外，不需要对它们做任何假设。我们在神经切向核机制下进行分析，通过将超额风险分解为估计误差和近似误差来建立泛化结果，并将梯度流视为一种隐式正则化器。这种在神经网络背景下的分解为梯度下降提供了一个新颖的视角，帮助我们避开一致收敛的陷阱。在本工作中，我们还证明了在相同设置下，训练后的网络会对数据产生过拟合。综合起来，这些结果首次建立了有限宽度ReLU网络在任意回归函数下的良性过拟合结论。

    arXiv:2410.06191v2 Announce Type: replace-cross  Abstract: We study the least-square regression problem with a two-layer fully-connected neural network, with ReLU activation function, trained by gradient flow. Our first result is a generalization result, that requires no assumptions on the underlying regression function or the noise other than that they are bounded. We operate in the neural tangent kernel regime, and our generalization result is developed via a decomposition of the excess risk into estimation and approximation errors, viewing gradient flow as an implicit regularizer. This decomposition in the context of neural networks is a novel perspective of gradient descent, and helps us avoid uniform convergence traps. In this work, we also establish that under the same setting, the trained network overfits to the data. Together, these results, establishes the first result on benign overfitting for finite-width ReLU networks for arbitrary regression functions.
    

