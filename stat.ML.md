# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Partially Personalized Federated Learning: Breaking the Curse of Data Heterogeneity.](http://arxiv.org/abs/2305.18285) | 本文提出了部分个性化联邦学习模型，将变量分为全局参数和个体本地参数，解决了数据异构问题，为每个客户端提供完美数据拟合的全局参数。此方法的共享的全局参数可用于学习优秀的数据表示，而个性化层则可用于特定客户端的微调。 |
| [^2] | [Learning Two-Layer Neural Networks, One (Giant) Step at a Time.](http://arxiv.org/abs/2305.18270) | 本文研究了浅层神经网络的训练动态及其条件，证明了动态下梯度下降可以通过有限数量的大批量梯度下降步骤来促进特征学习，并找到了多个和单一方向的最佳批量大小，有助于促进特征学习和方向的专业化。 |
| [^3] | [One Objective to Rule Them All: A Maximization Objective Fusing Estimation and Planning for Exploration.](http://arxiv.org/abs/2305.18258) | 提出一种在线强化学习方法Maximize to Explore (MEX)，只需优化一个无约束的目标函数，自动平衡探索和利用，实现次线性遗憾。 |
| [^4] | [High-Fidelity Image Compression with Score-based Generative Models.](http://arxiv.org/abs/2305.18231) | 本文提出了一种基于分数的生成模型的两阶段方法，该方法在图像压缩领域取得了显著的表现，实验证明该方法在一定比特率下能够提高图像的感知质量。 |
| [^5] | [Quantum Kernel Mixtures for Probabilistic Deep Learning.](http://arxiv.org/abs/2305.18204) | 本文提出了一种量子核混合方法，可以用于表示连续和离散随机变量的联合概率分布。该框架允许构建可微分的模型，适用于密度估计、推理和采样，以及各种机器学习任务，包括生成建模和判别学习。 |
| [^6] | [Rethinking Counterfactual Data Augmentation Under Confounding.](http://arxiv.org/abs/2305.18183) | 反事实数据增强是一种缓解数据中混淆偏差的方法，本文从因果的角度分析了混淆偏差对分类器的影响，提出了去除混淆偏差的手段，有助于在观察到的数据分布之外进行泛化。作者还提出了一个简单而有效的算法用于生成反事实图像，并证明了该方法在实际应用中的有效性。 |
| [^7] | [The minimax risk in testing the histogram of discrete distributions for uniformity under missing ball alternatives.](http://arxiv.org/abs/2305.18111) | 研究了离散分布样本对于类别间的均匀分布拟合问题下的极小极大风险，在缺少球形替代方案的情况下进行了讨论，通过离散直方图进行检验，获得了一种具有精确刻画的检验方法，并在实证研究中表现出了显著性。 |
| [^8] | [Leveraging Evolutionary Changes for Software Process Quality.](http://arxiv.org/abs/2305.18061) | 本文提出了一种利用演进变化来改善软件开发过程质量的方法，其包括使用统计过程控制和机器学习技术来分析应用程序生命周期管理所捕获的变更数据，实验表明该方法是有效的。 |
| [^9] | [Implicit Transfer Operator Learning: Multiple Time-Resolution Surrogates for Molecular Dynamics.](http://arxiv.org/abs/2305.18046) | ITO Learning是一个学习分子动力学多时间分辨率代理的框架，可以生成自洽的随机动力学，节省数百倍的时间。 |
| [^10] | [Combining Particle and Tensor-network Methods for Partial Differential Equations via Sketching.](http://arxiv.org/abs/2305.17884) | 本文提出了通过草图技术将粒子方法和张量网络方法结合的方法用于解决高维偏微分方程。这种方法包括粒子模拟和张量网络重新估计，并可用作粒子数控制的可替代方法。在模拟Fokker-Planck方程和量子虚时间演化方面，该方法表现出通用性和灵活性。 |
| [^11] | [Multinomial Logistic Regression: Asymptotic Normality on Null Covariates in High-Dimensions.](http://arxiv.org/abs/2305.17825) | 本文研究了高维多项式 logistic 模型中零协变量上最大似然估计的渐近正态性，为测试给定特征显着性提供了一种新方法。 |
| [^12] | [Acceleration of stochastic gradient descent with momentum by averaging: finite-sample rates and asymptotic normality.](http://arxiv.org/abs/2305.17665) | 研究了动量随机梯度下降（SGDM）和其Polyak-averaging版本的特性，表明在较大的批量大小下，小批量SGDM比小批量SGD更快地收敛到最优值的邻域。 |
| [^13] | [Reward Collapse in Aligning Large Language Models.](http://arxiv.org/abs/2305.17608) | 本文记录了大型语言模型训练中的奖励塌陷现象，导致在训练结束时，不同的提示生成的奖励分布相同。这主要是因为排名的目标函数无法在优化过程中考虑与提示相关的信息。 |
| [^14] | [Approximation-Generalization Trade-offs under (Approximate) Group Equivariance.](http://arxiv.org/abs/2305.17592) | 本论文详细研究了通过对称性明确地引入任务特定的归纳偏差所导致的逼近-泛化权衡，并且证明了这种模型在捕获任务特定对称性的同时会改进泛化。这一结果对于提高机器学习领域的性能具有非常大的帮助。 |
| [^15] | [On Neural Networks as Infinite Tree-Structured Probabilistic Graphical Models.](http://arxiv.org/abs/2305.17583) | 本文提出了一种创新方法，通过构建与神经网络完全对应的无限树状PGMs来解决深度神经网络(DNNs)缺乏PGMs的精确语义和明确定义的概率解释的问题。研究发现DNNs在前向传播时确实执行PGM推断的近似，这与现有研究不同，它阐明了DNNs对PGMs中的精确推理的更直接近似，潜在的好处包括改进DNNs的教学和解释，以及能够合并PGMs和DNNs的算法。 |
| [^16] | [Counterfactual Formulation of Patient-Specific Root Causes of Disease.](http://arxiv.org/abs/2305.17574) | 本文提出了一种针对疾病患者个体的根本原因的新公式，可以用于自动从数据中检测根本原因，并考虑了噪声标签和疾病流行率等因素，同时具有快速计算的优势。 |
| [^17] | [Auditing Fairness by Betting.](http://arxiv.org/abs/2305.17570) | 本文提供了一种通过赌博的方式进行公平性审计的方法，相比之前的方法，这种方法具有更高的实用性和效率，能够对不断产生的数据进行连续的监控，并处理因分布漂移导致的公平性问题。 |
| [^18] | [Provably Fast Finite Particle Variants of SVGD via Virtual Particle Stochastic Approximation.](http://arxiv.org/abs/2305.17558) | 本论文提出了两种基于虚拟粒子随机逼近的可证速限制变种的SVGD算法，具有可证速的有限粒子收敛率。 |
| [^19] | [Fair Clustering via Hierarchical Fair-Dirichlet Process.](http://arxiv.org/abs/2305.17557) | 本文提出了一种新的基于模型的公平聚类公式，该公式通过层次公平狄利克雷过程实现公平聚类的目标。 |
| [^20] | [PFNs Are Flexible Models for Real-World Bayesian Optimization.](http://arxiv.org/abs/2305.17535) | 本文使用灵活的PFN作为BO代理建模，该模型能够允许进一步信息纳入以进行非远视BO。在三种不同的问题上得到了很好的结果。 |
| [^21] | [The Implicit Regularization of Dynamical Stability in Stochastic Gradient Descent.](http://arxiv.org/abs/2305.17490) | 本文研究了随机梯度下降的动态稳定性隐式正则化，证明了其具有良好的泛化性。 |
| [^22] | [Deep Variational Lesion-Deficit Mapping.](http://arxiv.org/abs/2305.17478) | 本论文使用深度变分神经网络架构实现病变-缺陷推断任务，建立表达性层次模型，可估计联合损伤和缺陷分布，条件为潜在神经底物。 |
| [^23] | [Toward Understanding Generative Data Augmentation.](http://arxiv.org/abs/2305.17476) | 生成式数据增广通过从训练的生成模型中获得虚假的标记示例，提高分类性能；本文建立了一个普遍的稳定性界限，并发现其效果与生成模型的选择和训练集大小密切相关。 |
| [^24] | [Structured model selection via $\ell_1-\ell_2$ optimization.](http://arxiv.org/abs/2305.17467) | 通过稀疏最小二乘拟合一大组候选函数，使用 $\ell_1-\ell_2$ 稀疏优化方法进行结构模型选择，实现从不充分且嘈杂的时空数据中识别结构化动态系统；该方法在合成数据集上得到了验证，并证明具有理论保证和高效性。 |
| [^25] | [On the Noise Sensitivity of the Randomized SVD.](http://arxiv.org/abs/2305.17435) | 通过对R-SVD在低秩信号加噪声测量模型下的分析，证明了当信噪比(SNR)超过某个依赖于降维因子的可检测门限时，R-SVD产生的最大奇异值是一个离群值；在门限以下，没有离群值从奇异值块中产生 |
| [^26] | [No-Regret Online Reinforcement Learning with Adversarial Losses and Transitions.](http://arxiv.org/abs/2305.17380) | 本文提出了一种算法，可以处理对抗性损失和对抗性转换，且后悔逐渐增加与对手的恶意程度成比例。 |
| [^27] | [Stability-penalty-adaptive Follow-the-regularized-leader: Sparsity, Game-dependency, and Best-of-both-worlds.](http://arxiv.org/abs/2305.17301) | 本文开发了一种稳定性惩罚自适应（SPA）学习率，该学习率使FTRL具有稀疏性、游戏依赖性和最佳世界（BOBW）三种适应性类型，其中SPA-sparse算法可适应于未知的稀疏级别，SPA-game-dependency算法可根据所玩的游戏自适应地改变其行为，BOBW算法则是既具有稀疏性又具有游戏依赖性的适应性算法。 |
| [^28] | [Improving Stability in Decision Tree Models.](http://arxiv.org/abs/2305.17299) | 本文通过医疗应用的视角，提出了一种新的决策树距离度量，并用它来确定树的稳定水平。我们提出了一种新的培训稳定决策树的方法，并探究稳定性、预测能力和可解释性之间不可避免的权衡。 |
| [^29] | [Generalization Error without Independence: Denoising, Linear Regression, and Transfer Learning.](http://arxiv.org/abs/2305.17297) | 本论文研究了具有低秩结构但非独立同分布数据的情况，在分离训练和测试分布的假设下，解决了分布偏移问题，实验结果表明，在分布偏移的情况下，本方法显著提高了泛化误差的性能。 |
| [^30] | [GC-Flow: A Graph-Based Flow Network for Effective Clustering.](http://arxiv.org/abs/2305.17284) | GC-Flow是一种生成模型，可以同时建模类别条件概率和类别先验，通过配备高斯混合表示空间，保持预测能力的同时实现了良好分离的聚类。 |
| [^31] | [Optimizing NOTEARS Objectives via Topological Swaps.](http://arxiv.org/abs/2305.17277) | 本文提出了一种双层算法来解决学习DAGs中的非凸优化问题，其中外层利用拓扑交换优化拓扑顺序，通过开发一种候选交换对的方法，算法在学习高质量DAGs方面具有高效和稳定的优势。 |
| [^32] | [FineMorphs: Affine-diffeomorphic sequences for regression.](http://arxiv.org/abs/2305.17255) | FineMorphs是一种多元回归模型，通过形状分析中的微分同胚概念对模型状态进行优化，能够自然地减少（或增加）维度并适应大数据集。 |
| [^33] | [Causal Component Analysis.](http://arxiv.org/abs/2305.17225) | 本文介绍了一个中间问题：因果成分分析(CauCA)，它是独立成分分析(ICA)和因果表示学习(CRL)的泛化和特例，其目标是学习解混函数和因果机制，预设了因果图的知识。 |
| [^34] | [Fast and Minimax Optimal Estimation of Low-Rank Matrices via Non-Convex Gradient Descent.](http://arxiv.org/abs/2305.17224) | 本文提出一种针对低秩矩阵估计的方法，在保证极小极值优化性能的同时，解决了非凸梯度下降收敛缓慢的问题。 |
| [^35] | [Functional Flow Matching.](http://arxiv.org/abs/2305.17209) | 本文介绍了一种名为功能流匹配（FFM）的函数空间生成模型，该模型利用概率测度插值和学习底层函数空间上生成测度的向量场来生成数据分布。这种无需似然或模拟的方法在合成和真实世界基准数据集上表现优异，优于最近提出的几种函数空间生成模型。 |
| [^36] | [Error Bounds for Learning with Vector-Valued Random Features.](http://arxiv.org/abs/2305.17170) | 本文提供了对向量值随机特征学习的完整误差分析，包括在模型错误说明下向量值RF估计器的强一致性和在良好规定的情况下极小化最优收敛速率。 |
| [^37] | [Universal approximation with complex-valued deep narrow neural networks.](http://arxiv.org/abs/2305.16910) | 本文研究了具有有界宽度和任意深度的复值神经网络的普适性，发现当且仅当激活函数既不是全纯的，也不是反全纯的，也不是 $\mathbb{R}$-仿射的时，深窄的复值网络具有普适逼近能力。我们还发现足够的宽度依赖于考虑的激活函数，对于一类可允许的激活函数，宽度为 $n+m+4$ 是足够的。 |
| [^38] | [On the Generalization Capacities of Neural Controlled Differential Equations.](http://arxiv.org/abs/2305.16791) | 本文研究了使用神经控制微分方程进行监督学习的泛化能力问题，通过量化离散化偏差和利普希茨函数逼近误差，得到了经验风险最小化器与贝叶斯最优风险的泛化差距上界。 |
| [^39] | [Which Features are Learnt by Contrastive Learning? On the Role of Simplicity Bias in Class Collapse and Feature Suppression.](http://arxiv.org/abs/2305.16536) | 对比学习是一种表示学习技术，对于有监督的情况易于产生类坍塌，无监督情况下易于抑制类别相关的复杂特征；随机梯度下降方法偏向于寻找更简单的解决方案是导致这种现象的关键因素。 |
| [^40] | [Statistical post-processing of visibility ensemble forecasts.](http://arxiv.org/abs/2305.15325) | 本论文研究了后处理能见度集合预测的不同方法，发现非参数密度估计和高斯混合模型方法表现良好，并且可以显着提高集合预测的技能和可靠性。 |
| [^41] | [Bayesian approach to Gaussian process regression with uncertain inputs.](http://arxiv.org/abs/2305.11586) | 本文提出了一种新的高斯过程回归技术，通过贝叶斯方法将输入数据的不确定性纳入回归模型预测中。在数值实验中展示了该方法具有普适性和不错的表现。 |
| [^42] | [Double-Weighting for Covariate Shift Adaptation.](http://arxiv.org/abs/2305.08637) | 本文提出了一种双重加权的最小极大风险分类方法，可以有效避免协变量漂移对监督学习的影响。 |
| [^43] | [Convergence Analysis of Mean Shift.](http://arxiv.org/abs/2305.08463) | 本研究提出了均值漂移算法的模估计序列的收敛保证，并扩展了现有的涵盖解析核和Epanechnikov核的发现，意义在于涵盖了在基于KDE的模估计的渐近统计效率方面最优的非负核——双重核。 |
| [^44] | [Local Convergence of Gradient Descent-Ascent for Training Generative Adversarial Networks.](http://arxiv.org/abs/2305.08277) | 本论文研究了使用基于核函数的鉴别器训练GAN的梯度下降-上升算法的局部收敛性，揭示了学习率、正则化和带宽对其影响，同时展示了收敛、振荡或发散的相变现象。 |
| [^45] | [Robust Detection of Lead-Lag Relationships in Lagged Multi-Factor Models.](http://arxiv.org/abs/2305.06704) | 该论文提出了一种基于聚类的鲁棒检测滞后多因子模型中的领先滞后关系方法，并使用各种聚类技术和相似度度量方法实现了对领先滞后估计的聚合，从而强化了对原始宇宙中的一致关系的识别。 |
| [^46] | [On the existence of solutions to adversarial training in multiclass classification.](http://arxiv.org/abs/2305.00075) | 本文研究了多类分类中敌对训练的鲁棒解存在性问题，证明了每个模型中存在 Borel 可测的鲁棒分类器，并与最优传输和总变差正则化建立了联系。在二元分类问题中，对不可知分类器的敌对训练问题存在 Borel 可测的解。 |
| [^47] | [In-Context Operator Learning for Differential Equation Problems.](http://arxiv.org/abs/2304.07993) | 本文提出了一种新的神经网络方法INDEED，它可以同时学习不同微分方程问题的操作符，而无需重新训练，且只需要极少的演示。 |
| [^48] | [Diffusion Schr\"odinger Bridge Matching.](http://arxiv.org/abs/2303.16852) | 本文介绍了一种新的方法 Iterative Markovian Fitting，用于解决高维度 Schr\"odinger桥（SBs）问题，该方法的数值实验表现出在准确性和性能方面的显著优势。 |
| [^49] | [A Closer Look at Scoring Functions and Generalization Prediction.](http://arxiv.org/abs/2303.13589) | 本文研究了广义误差预测器的有效性，探讨了置信度、局部流形平滑度和模型一致性评分函数的优缺点，发现在复杂机制缺失的情况下，最先进的评分无法在分布转移和损坏下超越简单的模型一致性。同时，在受损训练数据的情况下，模型一致性打分仍然表现良好，并且集成多样性有助于提高泛化性能。 |
| [^50] | [Iterative Approximate Cross-Validation.](http://arxiv.org/abs/2303.02732) | 本文提出了一种新的方法，利用迭代一阶算法高效近似交叉验证，从而解决了大规模问题中因限制计算资源或早停而难以得到ERM问题确切解的问题。 |
| [^51] | [Depth Degeneracy in Neural Networks: Vanishing Angles in Fully Connected ReLU Networks on Initialization.](http://arxiv.org/abs/2302.09712) | 本文研究了深度神经网络中的深度退化现象，在全连接ReLU网络初始化时，两个输入之间的角度会趋近于0。通过使用组合展开，得到了其趋向于0的速度的精确公式，并验证了这些结果。 |
| [^52] | [Revisiting Discriminative vs. Generative Classifiers: Theory and Implications.](http://arxiv.org/abs/2302.02334) | 本文重新审视关于判别式与生成式分类器的经典主题，利用多类$\mathcal{H}$-一致性下界，证明了在温和的假设下，多类朴素贝叶斯分类器的样本要求比逻辑回归分类器多了$O(\log n)$。 |
| [^53] | [Beyond the Universal Law of Robustness: Sharper Laws for Random Features and Neural Tangent Kernels.](http://arxiv.org/abs/2302.01629) | 本文通过研究随机特征和神经切向核（NTK）的经验风险最小化，证明了在随机特征中，即使满足稳健性的通用定律所需的必要条件，模型也不具有任何过度参数化程度的稳健性。相对地，对于偶激活情况，NTK模型满足普遍下限，只要满足过参数条件就能稳健。这为机器学习中的稳健性提供了更尖锐的法则，超越了先前建立的普适定律。 |
| [^54] | [The contextual lasso: Sparse linear models via deep neural networks.](http://arxiv.org/abs/2302.00878) | 本论文提出了一种新的统计估计器——上下文套索，可以通过深度神经网络的方法解决解释性和拟合能力的矛盾问题，实现对可解释特征的稀疏拟合，并且稀疏模式和系数会随着上下文特征的变化而发生变化。 |
| [^55] | [Unconstrained Dynamic Regret via Sparse Coding.](http://arxiv.org/abs/2301.13349) | 本文探讨了在线线性优化（OLO）涉及无约束问题和动态遗憾问题的复杂性，提出了一种通过重新构造问题为稀疏编码的复杂度度量方式，在适应性和应用上有较好的应用价值。 |
| [^56] | [Variational sparse inverse Cholesky approximation for latent Gaussian processes via double Kullback-Leibler minimization.](http://arxiv.org/abs/2301.13303) | 本文提出了一种基于稀疏逆Cholesky因子的高斯分布的变分逼近方法，结合同样高效的SIC约束的Kullback-Leibler最优先验逼近，并在特定SIC排序和稀疏模式下，实现对潜在高斯过程的高度准确先验和后验逼近。与其他方法相比，该方法可以在类似计算复杂度下更准确地预测平稳核函数。 |
| [^57] | [Are Random Decompositions all we need in High Dimensional Bayesian Optimisation?.](http://arxiv.org/abs/2301.12844) | 本文研究了数据独立分解采样规则，证明了随机树分解采样器有利的理论保证，促进了随机分解上置信度算法（RDUCB）的发展。 |
| [^58] | [Estimating Causal Effects using a Multi-task Deep Ensemble.](http://arxiv.org/abs/2301.11351) | 通过学习研究群体中的共享和特定于组的信息，使用Causal Multi-task Deep Ensemble（CMDE）的方法可以有效地处理高维和多模态协变量，并提供因果效应的点估计不确定性。 |
| [^59] | [Maximum Optimality Margin: A Unified Approach for Contextual Linear Programming and Inverse Linear Programming.](http://arxiv.org/abs/2301.11260) | 本论文提出了一种名为“最大最优性边际”的新方法来解决预测-优化问题，通过下游优化的最优性条件设计机器学习损失函数，兼具计算效率和较好的理论性质，而且只需要训练数据中最优解的观测值。 |
| [^60] | [Learning Deformation Trajectories of Boltzmann Densities.](http://arxiv.org/abs/2301.07388) | 本文介绍了一种学习Boltzmann密度变形轨迹的方法，其中通过插值能量函数等实现Boltzmann密度的变形，然后找到一个时间依赖向量场，将样本从一个分布转移到另一个分布，其表现在高斯混合和量子力学粒子的Boltzmann密度上比KL-反散度更具优势。 |
| [^61] | [Relative Probability on Finite Outcome Spaces: A Systematic Examination of its Axiomatization, Properties, and Applications.](http://arxiv.org/abs/2212.14555) | 本文提出了将概率看作相对度量的观点，建立了有限结果空间上相对概率函数的公理化，提供了其实例和组合系统，并讨论了相对贝叶斯推断及其数字实现，证明了相对概率空间的拓扑闭包，突显了其在极限下保留信息的能力。 |
| [^62] | [Doubly Smoothed GDA: Global Convergent Algorithm for Constrained Nonconvex-Nonconcave Minimax Optimization.](http://arxiv.org/abs/2212.12978) | 本文提出了一种双重平滑梯度下降上升法 (DSGDA)，该算法可以应用于非凸-非凹极小极大优化，并且能够全局收敛并消除极限环。在一定条件下，DSGDA 的迭代复杂度达到了文献中单循环算法的最佳结果。 |
| [^63] | [Understanding the Impact of Adversarial Robustness on Accuracy Disparity.](http://arxiv.org/abs/2211.15762) | 本文通过研究高斯混合模型下的线性分类器，分析了对抗鲁棒性对准确性不平衡的影响，并证明了在稳定分布的一般家族中也存在类似影响。 |
| [^64] | [Zeroth-Order Alternating Gradient Descent Ascent Algorithms for a Class of Nonconvex-Nonconcave Minimax Problems.](http://arxiv.org/abs/2211.13668) | 本文提出了零阶交替梯度下降算法和零阶方差减少交替梯度下降算法，用于解决一类非凸非凹的极小极大问题，分别在确定性和随机环境下。它们是解决这类问题的第一和第二个迭代复杂度保证的零阶算法。 |
| [^65] | [Introduction to Online Nonstochastic Control.](http://arxiv.org/abs/2211.09619) | 介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。 |
| [^66] | [PAC-Bayesian Offline Contextual Bandits With Guarantees.](http://arxiv.org/abs/2210.13132) | 本文提出了一种通过PAC-Bayesian方法分析离线情境强化学习问题的新算法，该算法通过优化新的泛化界限提供了保证，并在实际情境中得到了验证。 |
| [^67] | [Correlating sparse sensing for large-scale traffic speed estimation: A Laplacian-enhanced low-rank tensor kriging approach.](http://arxiv.org/abs/2210.11780) | 本文提出了一种基于拉普拉斯增强的低秩张量样条克里金方法，用于在有限观测下进行大规模交通速度估计，以从不完整的数据中恢复可信的估计值。 |
| [^68] | [Selection by Prediction with Conformal p-values.](http://arxiv.org/abs/2210.01408) | 本论文提出一种基于符合p值的预测选择方法，使用统计证据的p值来控制虚阳性选择单位，可用于初步筛选职业招聘和药物发现的候选人集合。 |
| [^69] | [Many-body Approximation for Non-negative Tensors.](http://arxiv.org/abs/2209.15338) | 提出了一种名为多体逼近的方法来分解非负张量，通过能量建模来避免全局优化和目标秩选择的困难，可通过考虑模式之间的交互进行全局优化; 在许多任务中都展示了其有效性。 |
| [^70] | [Likelihood Adjusted Semidefinite Programs for Clustering Heterogeneous Data.](http://arxiv.org/abs/2209.15097) | 本论文提出了一种基于似然函数修正的半定规划方法用于异质数据聚类。经过实验表明，本方法在处理聚类形状不同的数据异质性时表现优异。 |
| [^71] | [Robust Methods for High-Dimensional Linear Learning.](http://arxiv.org/abs/2208.05447) | 该论文提出了一种在高维批量训练中具有高度鲁棒性和计算效率的线性学习方法，在多个应用程序中均能达到接近最优的估计速率，并提供了一个开源的Python库进行实现。 |
| [^72] | [Convergence of denoising diffusion models under the manifold hypothesis.](http://arxiv.org/abs/2208.05314) | 本文提供了基于流模型的去噪在流形假设下的收敛性分析，首次拓展到了目标分布受流形约束或通过经验分布给出的情况。 |
| [^73] | [Differentially Private Federated Combinatorial Bandits with Constraints.](http://arxiv.org/abs/2206.13192) | 本文研究了差分隐私联合组合赌博机问题，探讨了代理在共同学习时如何保持数据的隐私，并提出了在后悔和隐私之间实现平衡的重要性。 |
| [^74] | [Anchor Sampling for Federated Learning with Partial Client Participation.](http://arxiv.org/abs/2206.05891) | 提出一种针对部分客户端参与的联邦学习框架 FedAMD，其中核心思想是锚定抽样，将参与者分为锚定组和矿工组，以解决数据异构性。 |
| [^75] | [Mining Multi-Label Samples from Single Positive Labels.](http://arxiv.org/abs/2206.05764) | 本文提出了一种基于单一正标签的采样方法S2M，以实现对多标签样本的生成，从而避免了多标签数据集制作时高昂的注释成本。 |
| [^76] | [Generalization Error Bounds for Deep Neural Networks Trained by SGD.](http://arxiv.org/abs/2206.03299) | 通过对于适当参数规范的动态控制结合基于参数规范的 Rademacher 复杂度估计导出了深度神经网络的泛化误差界，适用于包括 MLP 和 CNN 在内的广泛网络架构，结果表明这个方法能够适应优化器和网络超参数的变化。 |
| [^77] | [Detecting hidden confounding in observational data using multiple environments.](http://arxiv.org/abs/2205.13935) | 使用独立数据生成过程下的多环境方法，可以检测观测数据中的未观察到的混淆因素，并提出了测试独立性的程序。 |
| [^78] | [Compressed Empirical Measures (in finite dimensions).](http://arxiv.org/abs/2204.08847) | 本论文探讨了在有限维再生核希尔伯特空间中压缩经验测度的方法，导出了关于这样一个近似的核心集必须有的大小的高概率下限，并开发了一些技术以将压缩方法应用于具体的推断问题。 |
| [^79] | [Sion's Minimax Theorem in Geodesic Metric Spaces and a Riemannian Extragradient Algorithm.](http://arxiv.org/abs/2202.06950) | 该论文提出了在测地度量空间中的Sion极小极大定理和黎曼外推算法，在保持问题可处理的同时，为非凸-非凹极小极大问题提供了一个广泛的推广。 |
| [^80] | [Composite Goodness-of-fit Tests with Kernels.](http://arxiv.org/abs/2111.10275) | 本文提出了一种基于核的假设检验方法，可以解决具有挑战性的复合检验问题，其核心思想是在正确的模型规范的零假设下，非参数地估计参数（或模拟器）分布。 |
| [^81] | [MMD Aggregated Two-Sample Test.](http://arxiv.org/abs/2110.15073) | 本文提出了两种新颖的基于最大均值差异（MMD）的非参数双样本核检验，并构造了一种自适应平均测试，称为MMDAgg，以解决平滑参数未知的问题。 |
| [^82] | [Sinkhorn Distributionally Robust Optimization.](http://arxiv.org/abs/2109.11926) | 本文通过使用Sinkhorn距离进行分布鲁棒优化，推导出更容易处理且在实际中更合理的最坏情况分布，提出了解决方案，并展示了其优越性能。 |
| [^83] | [SGD with a Constant Large Learning Rate Can Converge to Local Maxima.](http://arxiv.org/abs/2107.11774) | 本研究构建了最坏情况下的优化问题，证明了带有定常大学习率的SGD可能表现出许多奇怪且潜在的不良行为，包括：收敛于局部最大值、缓慢越过鞍点和更喜欢尖锐的最小值。这强调了深入分析SGD在深度学习中作用的重要性。 |
| [^84] | [Variance-Dependent Best Arm Identification.](http://arxiv.org/abs/2106.10417) | 本文研究了在多臂老虎机游戏中识别最优臂的问题。提出了一种自适应算法，该算法探索臂的奖励差距和方差，并使用一种称为分组中位数淘汰的新方法根据收集的信息做出未来决策。所提出的算法保证以概率(1-δ)输出最优臂，并使用最多的O（Σ(i=1)^n (σi²/Δi²+1/Δi)(lnδ-1+ln lnΔi-1)）个样本，这比方差独立算法获得了明显的优势。 |
| [^85] | [How good is Good-Turing for Markov samples?.](http://arxiv.org/abs/2102.01938) | 研究了在具有状态分布$[\pi_x:x \in \mathcal{X}]$和转移概率矩阵（t.p.m.）$P$的字母表$\mathcal{X}$上，Markov样本的缺失稳态质量（即缺失符号的总稳态概率）的GT估计器的收敛性。 |
| [^86] | [A Modern Introduction to Online Learning.](http://arxiv.org/abs/1912.13213) | 这本专著介绍了在线学习的基本概念以及凸优化背景下的一阶和二阶算法, 包括欧几里得和非欧几里得设置中的在线镜像下降或遵循正则化领导者等算法。 |
| [^87] | [Fast MLE Computation for the Dirichlet Multinomial.](http://arxiv.org/abs/1405.0099) | 本文提出了一种修改的方法来快速计算Dirichlet分布的MLE参数，相较于现有实现方法，只需要一遍遍历数据集就可以大大减少运行时间。 |

# 详细

[^1]: 部分个性化联邦学习：打破数据异构之咒

    Partially Personalized Federated Learning: Breaking the Curse of Data Heterogeneity. (arXiv:2305.18285v1 [cs.LG])

    [http://arxiv.org/abs/2305.18285](http://arxiv.org/abs/2305.18285)

    本文提出了部分个性化联邦学习模型，将变量分为全局参数和个体本地参数，解决了数据异构问题，为每个客户端提供完美数据拟合的全局参数。此方法的共享的全局参数可用于学习优秀的数据表示，而个性化层则可用于特定客户端的微调。

    

    我们提出了部分个性化联邦学习（FL）的模型，旨在平衡个性化与全局训练的合作性之间的关系。在我们的框架中，我们将变量分为全局参数和个体本地参数。证明了在正确的参数拆分下，可以找到允许每个客户端完美拟合其数据的全局参数，并将所得到的问题称为过度个性化问题。共享的全局参数可以用于学习优秀的数据表示，而个性化层则为特定客户端进行微调。此外，我们提出了一种简单的算法来解决部分个性化学习问题，为所有客户端带来了显著的益处。特别地，在许多情况下，如使用本地步骤，异步训练和拜占庭-鲁棒训练中，这种算法打破了数据异构的咒语。

    We present a partially personalized formulation of Federated Learning (FL) that strikes a balance between the flexibility of personalization and cooperativeness of global training. In our framework, we split the variables into global parameters, which are shared across all clients, and individual local parameters, which are kept private. We prove that under the right split of parameters, it is possible to find global parameters that allow each client to fit their data perfectly, and refer to the obtained problem as overpersonalized. For instance, the shared global parameters can be used to learn good data representations, whereas the personalized layers are fine-tuned for a specific client. Moreover, we present a simple algorithm for the partially personalized formulation that offers significant benefits to all clients. In particular, it breaks the curse of data heterogeneity in several settings, such as training with local steps, asynchronous training, and Byzantine-robust training.
    
[^2]: 学习两层神经网络：一次(巨大)的步骤。

    Learning Two-Layer Neural Networks, One (Giant) Step at a Time. (arXiv:2305.18270v1 [stat.ML])

    [http://arxiv.org/abs/2305.18270](http://arxiv.org/abs/2305.18270)

    本文研究了浅层神经网络的训练动态及其条件，证明了动态下梯度下降可以通过有限数量的大批量梯度下降步骤来促进特征学习，并找到了多个和单一方向的最佳批量大小，有助于促进特征学习和方向的专业化。

    

    我们研究了浅层神经网络的训练动态，研究了有限数量的大批量梯度下降步骤有助于在核心范围之外促进特征学习的条件。我们比较了批量大小和多个(但有限的)步骤的影响。我们分析了单步骤过程，发现批量大小为$n=O(d)$可以促进特征学习，但只适合学习单一方向或单索引模型。相比之下，$n=O(d^2)$对于学习多个方向和专业化至关重要。此外，我们证明“硬”方向缺乏前$\ell$个Hermite系数，仍未被发现，并且需要批量大小为$n=O(d^\ell)$才能被梯度下降捕获。经过几次迭代，情况发生变化：批量大小为$n=O(d)$足以学习新的目标方向，这些方向在Hermite基础上线性连接到之前学习的方向所涵盖的子空间。

    We study the training dynamics of shallow neural networks, investigating the conditions under which a limited number of large batch gradient descent steps can facilitate feature learning beyond the kernel regime. We compare the influence of batch size and that of multiple (but finitely many) steps. Our analysis of a single-step process reveals that while a batch size of $n = O(d)$ enables feature learning, it is only adequate for learning a single direction, or a single-index model. In contrast, $n = O(d^2)$ is essential for learning multiple directions and specialization. Moreover, we demonstrate that ``hard'' directions, which lack the first $\ell$ Hermite coefficients, remain unobserved and require a batch size of $n = O(d^\ell)$ for being captured by gradient descent. Upon iterating a few steps, the scenario changes: a batch-size of $n = O(d)$ is enough to learn new target directions spanning the subspace linearly connected in the Hermite basis to the previously learned directions,
    
[^3]: 一种融合估计和规划实现探索的最大化目标函数的在线强化学习方法

    One Objective to Rule Them All: A Maximization Objective Fusing Estimation and Planning for Exploration. (arXiv:2305.18258v1 [cs.LG])

    [http://arxiv.org/abs/2305.18258](http://arxiv.org/abs/2305.18258)

    提出一种在线强化学习方法Maximize to Explore (MEX)，只需优化一个无约束的目标函数，自动平衡探索和利用，实现次线性遗憾。

    

    在在线强化学习中，平衡探索和利用对于以有效的方式找到最优策略至关重要。为了实现这一目标，现有的在线强化学习算法通常包括三个组成部分：估计、规划和探索。然而，为了应对通用函数逼近器，在大多数情况下都需要使用不切实际的算法组件来激励探索，例如数据相关的级别集内优化或繁琐的采样过程。为了解决这一挑战，我们提出了一种易于实现的强化学习框架，称为Maximize to Explore (MEX) ，它只需要无约束地优化一个集成了估计和规划组件的单一目标函数，同时自动平衡探索和利用。理论上，我们证明了对于马尔可夫决策过程的通用函数逼近，MEX实现了一个次线性的遗憾，进一步：

    In online reinforcement learning (online RL), balancing exploration and exploitation is crucial for finding an optimal policy in a sample-efficient way. To achieve this, existing sample-efficient online RL algorithms typically consist of three components: estimation, planning, and exploration. However, in order to cope with general function approximators, most of them involve impractical algorithmic components to incentivize exploration, such as optimization within data-dependent level-sets or complicated sampling procedures. To address this challenge, we propose an easy-to-implement RL framework called \textit{Maximize to Explore} (\texttt{MEX}), which only needs to optimize \emph{unconstrainedly} a single objective that integrates the estimation and planning components while balancing exploration and exploitation automatically. Theoretically, we prove that \texttt{MEX} achieves a sublinear regret with general function approximations for Markov decision processes (MDP) and is further 
    
[^4]: 基于分数的生成模型的高保真图像压缩

    High-Fidelity Image Compression with Score-based Generative Models. (arXiv:2305.18231v1 [eess.IV])

    [http://arxiv.org/abs/2305.18231](http://arxiv.org/abs/2305.18231)

    本文提出了一种基于分数的生成模型的两阶段方法，该方法在图像压缩领域取得了显著的表现，实验证明该方法在一定比特率下能够提高图像的感知质量。

    

    尽管扩散生成模型在文本到图像生成中取得了巨大的成功，但在图像压缩领域复制这个成功却很困难。在本文中，我们展示了扩散模型可以显著提高在给定比特率下的感知质量，通过 FID 分数评估，表现超越了 PO-ELIC 和 HiFiC 的现有方法。我们通过一个简单但在理论上有动机的两阶段方法实现了这一点，该方法结合了以 MSE 为目标的自动编码器和一个进一步基于分数的解码器。然而，正如我们将展示的那样，实现细节很重要，最佳设计决策可能与典型的文本到图像模型有很大不同。

    Despite the tremendous success of diffusion generative models in text-to-image generation, replicating this success in the domain of image compression has proven difficult. In this paper, we demonstrate that diffusion can significantly improve perceptual quality at a given bit-rate, outperforming state-of-the-art approaches PO-ELIC and HiFiC as measured by FID score. This is achieved using a simple but theoretically motivated two-stage approach combining an autoencoder targeting MSE followed by a further score-based decoder. However, as we will show, implementation details matter and the optimal design decisions can differ greatly from typical text-to-image models.
    
[^5]: 概率深度学习的量子核混合方法

    Quantum Kernel Mixtures for Probabilistic Deep Learning. (arXiv:2305.18204v1 [cs.LG])

    [http://arxiv.org/abs/2305.18204](http://arxiv.org/abs/2305.18204)

    本文提出了一种量子核混合方法，可以用于表示连续和离散随机变量的联合概率分布。该框架允许构建可微分的模型，适用于密度估计、推理和采样，以及各种机器学习任务，包括生成建模和判别学习。

    

    本文提出了一种新的概率深度学习方法——量子核混合，它是从量子密度矩阵的数学形式中推导出来的。该方法提供了一种简单而有效的机制，用于表示连续和离散随机变量的联合概率分布。该框架允许构建可微分的模型，用于密度估计、推理和采样，从而能够整合到端到端的深度神经模型中。通过这样做，我们提供了一种多功能的边际和联合概率分布表示，可以开发一种可微分的、组合的和可逆的推理过程，涵盖了广泛的机器学习任务，包括密度估计、判别学习和生成建模。我们通过两个示例来说明该框架的广泛适用性：一个图像分类模型，它可以自然地转化为条件生成模型，得益于量子核混合的表示能力。

    This paper presents a novel approach to probabilistic deep learning (PDL), quantum kernel mixtures, derived from the mathematical formalism of quantum density matrices, which provides a simpler yet effective mechanism for representing joint probability distributions of both continuous and discrete random variables. The framework allows for the construction of differentiable models for density estimation, inference, and sampling, enabling integration into end-to-end deep neural models. In doing so, we provide a versatile representation of marginal and joint probability distributions that allows us to develop a differentiable, compositional, and reversible inference procedure that covers a wide range of machine learning tasks, including density estimation, discriminative learning, and generative modeling. We illustrate the broad applicability of the framework with two examples: an image classification model, which can be naturally transformed into a conditional generative model thanks to
    
[^6]: 重新思考混淆下的反事实数据增强

    Rethinking Counterfactual Data Augmentation Under Confounding. (arXiv:2305.18183v1 [cs.LG])

    [http://arxiv.org/abs/2305.18183](http://arxiv.org/abs/2305.18183)

    反事实数据增强是一种缓解数据中混淆偏差的方法，本文从因果的角度分析了混淆偏差对分类器的影响，提出了去除混淆偏差的手段，有助于在观察到的数据分布之外进行泛化。作者还提出了一个简单而有效的算法用于生成反事实图像，并证明了该方法在实际应用中的有效性。

    

    反事实数据增强最近被提出来作为缓解训练数据中混淆偏差的一种方法。这些偏差，比如虚假的关联，是由于数据生成过程中各种观察到的和未观察到的混淆变量引起的。本文正式分析了混淆偏差如何影响下游分类器，并从因果的角度探讨基于反事实数据增强的解决方案。我们探讨如何去除混淆偏差作为学习不变特征的手段，最终有助于在观察到的数据分布之外进行泛化。此外，我们提出了一个简单但强大的算法，用于生成反事实图像，有效地缓解混淆效应对下游分类器的影响。通过在MNIST变体和CelebA数据集上的实验，我们展示了我们的方法的有效性和实用性。

    Counterfactual data augmentation has recently emerged as a method to mitigate confounding biases in the training data for a machine learning model. These biases, such as spurious correlations, arise due to various observed and unobserved confounding variables in the data generation process. In this paper, we formally analyze how confounding biases impact downstream classifiers and present a causal viewpoint to the solutions based on counterfactual data augmentation. We explore how removing confounding biases serves as a means to learn invariant features, ultimately aiding in generalization beyond the observed data distribution. Additionally, we present a straightforward yet powerful algorithm for generating counterfactual images, which effectively mitigates the influence of confounding effects on downstream classifiers. Through experiments on MNIST variants and the CelebA datasets, we demonstrate the effectiveness and practicality of our approach.
    
[^7]: 在缺少球形替代方案下测试离散分布直方图均匀性的极小极大风险

    The minimax risk in testing the histogram of discrete distributions for uniformity under missing ball alternatives. (arXiv:2305.18111v1 [math.ST])

    [http://arxiv.org/abs/2305.18111](http://arxiv.org/abs/2305.18111)

    研究了离散分布样本对于类别间的均匀分布拟合问题下的极小极大风险，在缺少球形替代方案的情况下进行了讨论，通过离散直方图进行检验，获得了一种具有精确刻画的检验方法，并在实证研究中表现出了显著性。

    

    我们考虑测试一个来自许多类别的离散样本对于类别间的均匀分布拟合的问题。作为另一类替代假设，我们考虑去除半径为$\epsilon$的$\ell_p$球形替代方案，其中$p\leq 2$。我们给出了基于直方图（缺失类别、单例、碰撞的数量）的检验在样本数和维数趋向无穷大，$\epsilon\to0$时，渐进极小极大风险的一个精确刻画。例如，当$p=1$且期望样本数$n$与类别数$N$的比值很小（也称为“次线性”区域）时，渐进极小极大风险$R^*_\epsilon$趋近于$2\bar{\Phi}\left(n\epsilon^2/\sqrt{8N}\right)$，其中$\bar{\Phi}(x)$是正态残存函数。在一系列问题参数范围内的实证研究表明，这个估计在有限样本中很精确，并且我们的检验显著。

    We consider the problem of testing the fit of a discrete sample of items from many categories to the uniform distribution over the categories. As a class of alternative hypotheses, we consider the removal of an $\ell_p$ ball of radius $\epsilon$ around the uniform rate sequence for $p \leq 2$. We deliver a sharp characterization of the asymptotic minimax risk when $\epsilon \to 0$ as the number of samples and number of dimensions go to infinity, for testing based on the occurrences' histogram (number of absent categories, singletons, collisions, ...). For example, for $p=1$ and in the limit of a small expected number of samples $n$ compared to the number of categories $N$ (aka "sub-linear" regime), the minimax risk $R^*_\epsilon$ asymptotes to $2 \bar{\Phi}\left(n \epsilon^2/\sqrt{8N}\right) $, with $\bar{\Phi}(x)$ the normal survival function. Empirical studies over a range of problem parameters show that this estimate is accurate in finite samples, and that our test is significantly 
    
[^8]: 利用演进变化提高软件过程质量。

    Leveraging Evolutionary Changes for Software Process Quality. (arXiv:2305.18061v1 [cs.SE])

    [http://arxiv.org/abs/2305.18061](http://arxiv.org/abs/2305.18061)

    本文提出了一种利用演进变化来改善软件开发过程质量的方法，其包括使用统计过程控制和机器学习技术来分析应用程序生命周期管理所捕获的变更数据，实验表明该方法是有效的。

    

    现实世界中的软件应用必须不断演进才能保持相关性。传统的软件质量控制方法涉及软件质量模型和持续的代码检查工具。然而，软件开发过程的质量与最终软件产品的质量之间存在强关联和因果关系。因此，间接提高软件产品的质量需要改善软件开发过程的质量。本文提出了一种利用开发过程的演进变化来提高软件质量的新方法。该方法包括使用统计过程控制和机器学习技术来分析应用程序生命周期管理所捕获的变更数据。实验结果显示了该方法的有效性。

    Real-world software applications must constantly evolve to remain relevant. This evolution occurs when developing new applications or adapting existing ones to meet new requirements, make corrections, or incorporate future functionality. Traditional methods of software quality control involve software quality models and continuous code inspection tools. These measures focus on directly assessing the quality of the software. However, there is a strong correlation and causation between the quality of the development process and the resulting software product. Therefore, improving the development process indirectly improves the software product, too. To achieve this, effective learning from past processes is necessary, often embraced through post mortem organizational learning. While qualitative evaluation of large artifacts is common, smaller quantitative changes captured by application lifecycle management are often overlooked. In addition to software metrics, these smaller changes can 
    
[^9]: 隐式转移算子学习：分子动力学多时间分辨率代理

    Implicit Transfer Operator Learning: Multiple Time-Resolution Surrogates for Molecular Dynamics. (arXiv:2305.18046v1 [physics.chem-ph])

    [http://arxiv.org/abs/2305.18046](http://arxiv.org/abs/2305.18046)

    ITO Learning是一个学习分子动力学多时间分辨率代理的框架，可以生成自洽的随机动力学，节省数百倍的时间。

    

    计算分子系统的性质需要估计（未归一化的）玻尔兹曼分布的期望值。分子动力学（MD）是一种广泛采用的技术，用于近似这种量。然而，稳定的模拟需要非常小的积分时间步长（$10^{-15}$秒），而一些矩的收敛性，例如结合自由能或速率，可能依赖于长达$10^{-1}$秒的时间尺度上的采样过程，并且必须对每个分子系统进行独立模拟。在这里，我们提出了隐式转移算子（ITO）学习，这是一个学习具有多个时间分辨率的模拟过程代理的框架。我们使用具有新SE（3）等变体系结构的去噪扩散概率模型实现ITO，并展示结果模型可以在多个时间尺度上生成自洽的随机动力学，即使只有部分观测到系统。最后，我们提出了粗粒化的CG-SE3-ITO模型，并展示它可以在模拟过程中节省数百倍的时间。

    Computing properties of molecular systems rely on estimating expectations of the (unnormalized) Boltzmann distribution. Molecular dynamics (MD) is a broadly adopted technique to approximate such quantities. However, stable simulations rely on very small integration time-steps ($10^{-15}\,\mathrm{s}$), whereas convergence of some moments, e.g. binding free energy or rates, might rely on sampling processes on time-scales as long as $10^{-1}\, \mathrm{s}$, and these simulations must be repeated for every molecular system independently. Here, we present Implict Transfer Operator (ITO) Learning, a framework to learn surrogates of the simulation process with multiple time-resolutions. We implement ITO with denoising diffusion probabilistic models with a new SE(3) equivariant architecture and show the resulting models can generate self-consistent stochastic dynamics across multiple time-scales, even when the system is only partially observed. Finally, we present a coarse-grained CG-SE3-ITO mo
    
[^10]: 通过草图技术，将粒子方法和张量网络方法结合用于偏微分方程求解

    Combining Particle and Tensor-network Methods for Partial Differential Equations via Sketching. (arXiv:2305.17884v1 [math.NA])

    [http://arxiv.org/abs/2305.17884](http://arxiv.org/abs/2305.17884)

    本文提出了通过草图技术将粒子方法和张量网络方法结合的方法用于解决高维偏微分方程。这种方法包括粒子模拟和张量网络重新估计，并可用作粒子数控制的可替代方法。在模拟Fokker-Planck方程和量子虚时间演化方面，该方法表现出通用性和灵活性。

    

    本文提出了一种解决高维偏微分方程的张量网络框架，其中我们采用粒子模拟更新解决方案，并使用最近提出的张量列车草图技术将新解决方案重新估计为张量网络。我们的方法还可以被解释为通过假设粒子来自底层张量网络来执行粒子数控制的可替代方法。我们通过将其应用于两种特定的情景来展示我们方法的通用性和灵活性：通过Langevin动力学模拟Fokker-Planck方程和通过辅助场量子蒙特卡罗模拟量子虚时间演化。

    In this paper, we propose a general framework for solving high-dimensional partial differential equations with tensor networks. Our approach offers a comprehensive solution methodology, wherein we employ a combination of particle simulations to update the solution and re-estimations of the new solution as a tensor-network using a recently proposed tensor train sketching technique. Our method can also be interpreted as an alternative approach for performing particle number control by assuming the particles originate from an underlying tensor network. We demonstrate the versatility and flexibility of our approach by applying it to two specific scenarios: simulating the Fokker-Planck equation through Langevin dynamics and quantum imaginary time evolution via auxiliary-field quantum Monte Carlo.
    
[^11]: 多项式 Logistic 回归：高维空间中零协变量的渐近正态性研究

    Multinomial Logistic Regression: Asymptotic Normality on Null Covariates in High-Dimensions. (arXiv:2305.17825v1 [math.ST])

    [http://arxiv.org/abs/2305.17825](http://arxiv.org/abs/2305.17825)

    本文研究了高维多项式 logistic 模型中零协变量上最大似然估计的渐近正态性，为测试给定特征显着性提供了一种新方法。

    

    本文研究了在维数和样本量相同的高维情况下多项式 logistic 模型中最大似然估计的渐近分布。尽管经典的大样本理论在一定条件下提供了 MLE 的渐近正态性，但是这些经典结果在高维情况下有望失败，就像 Sur 和 Cand\`es [2019] 在二分类 logistic 情况下的开创性工作所记录的那样。本文通过开发零协变量上的多项式 logistic MLE (也称为交叉熵最小化器)的渐近正态性和渐近卡方结果，解决了三类及以上分类问题中的这个问题。我们的理论为测试给定特征的显着性提供了一种新的方法论。对合成数据的广泛模拟研究证实了这些渐近结果，并确认了用于测试给定特征的提议的 p 值的有效性。

    This paper investigates the asymptotic distribution of the maximum-likelihood estimate (MLE) in multinomial logistic models in the high-dimensional regime where dimension and sample size are of the same order. While classical large-sample theory provides asymptotic normality of the MLE under certain conditions, such classical results are expected to fail in high-dimensions as documented for the binary logistic case in the seminal work of Sur and Cand\`es [2019]. We address this issue in classification problems with 3 or more classes, by developing asymptotic normality and asymptotic chi-square results for the multinomial logistic MLE (also known as cross-entropy minimizer) on null covariates. Our theory leads to a new methodology to test the significance of a given feature. Extensive simulation studies on synthetic data corroborate these asymptotic results and confirm the validity of proposed p-values for testing the significance of a given feature.
    
[^12]: 通过平均加速动量随机梯度下降：有限样本速率和渐近正态性

    Acceleration of stochastic gradient descent with momentum by averaging: finite-sample rates and asymptotic normality. (arXiv:2305.17665v1 [cs.LG])

    [http://arxiv.org/abs/2305.17665](http://arxiv.org/abs/2305.17665)

    研究了动量随机梯度下降（SGDM）和其Polyak-averaging版本的特性，表明在较大的批量大小下，小批量SGDM比小批量SGD更快地收敛到最优值的邻域。

    

    动量随机梯度下降（SGDM）被广泛应用于许多机器学习和统计应用中。尽管SGDM相对于传统的随机梯度下降具有观察到的经验优势，但在优化过程中动量对不同学习率的作用的理论理解仍然是开放的。我们在强凸设置下分析了SGDM的有限样本收敛速率，并表明在较大的批量大小下，小批量SGDM比小批量SGD更快地收敛到最优值的邻域。此外，我们分析了SGDM估计量的Polyak平均版本，建立了它的渐近正态性，并证明了它与平均SGD的渐近等价性。

    Stochastic gradient descent with momentum (SGDM) has been widely used in many machine learning and statistical applications. Despite the observed empirical benefits of SGDM over traditional SGD, the theoretical understanding of the role of momentum for different learning rates in the optimization process remains widely open. We analyze the finite-sample convergence rate of SGDM under the strongly convex settings and show that, with a large batch size, the mini-batch SGDM converges faster than mini-batch SGD to a neighborhood of the optimal value. Furthermore, we analyze the Polyak-averaging version of the SGDM estimator, establish its asymptotic normality, and justify its asymptotic equivalence to the averaged SGD.
    
[^13]: 对齐大型语言模型中的奖励塌缩现象

    Reward Collapse in Aligning Large Language Models. (arXiv:2305.17608v1 [cs.LG])

    [http://arxiv.org/abs/2305.17608](http://arxiv.org/abs/2305.17608)

    本文记录了大型语言模型训练中的奖励塌陷现象，导致在训练结束时，不同的提示生成的奖励分布相同。这主要是因为排名的目标函数无法在优化过程中考虑与提示相关的信息。

    

    大型语言模型（LLMs），如ChatGPT和GPT-4，具有非凡的能力，部分原因在于将它们与训练在人类偏好上的奖励模型对齐，这些偏好通常表示为对响应提示的排名。本文记录了奖励塌陷现象，这是一种经验观察，其中基于排名的方法导致在训练的终止阶段生成的完整奖励分布\textit{无论}\textbf{prompt是什么}都是\textit{相同的}。这种结果是不可取的，因为像“写一篇关于你最好的朋友的简短故事”这样的开放式提示应生成完成它们的连续奖励范围，而像“新西兰的首都是什么”这样的特定提示应生成高或低奖励。我们的理论调查表明，奖励塌陷主要是由于基于排名的目标函数在优化过程中未能纳入与提示相关的信息所致。

    The extraordinary capabilities of large language models (LLMs) such as ChatGPT and GPT-4 are in part unleashed by aligning them with reward models that are trained on human preferences, which are often represented as rankings of responses to prompts. In this paper, we document the phenomenon of \textit{reward collapse}, an empirical observation where the prevailing ranking-based approach results in an \textit{identical} reward distribution \textit{regardless} of the prompts during the terminal phase of training. This outcome is undesirable as open-ended prompts like ``write a short story about your best friend'' should yield a continuous range of rewards for their completions, while specific prompts like ``what is the capital of New Zealand'' should generate either high or low rewards. Our theoretical investigation reveals that reward collapse is primarily due to the insufficiency of the ranking-based objective function to incorporate prompt-related information during optimization. Thi
    
[^14]: (近似)群等变性下的逼近-泛化权衡

    Approximation-Generalization Trade-offs under (Approximate) Group Equivariance. (arXiv:2305.17592v1 [cs.LG])

    [http://arxiv.org/abs/2305.17592](http://arxiv.org/abs/2305.17592)

    本论文详细研究了通过对称性明确地引入任务特定的归纳偏差所导致的逼近-泛化权衡，并且证明了这种模型在捕获任务特定对称性的同时会改进泛化。这一结果对于提高机器学习领域的性能具有非常大的帮助。

    

    通过对称性明确地引入任务特定的归纳偏差已成为高性能机器学习模型开发中的常规设计准则。例如，群等变神经网络在蛋白质和药物设计等各个领域和应用中展现了卓越的性能。这种模型的普遍感觉是，将相关对称性整合到模型中会增强泛化能力。此外，有人认为，当数据和/或模型只能表现出$\textit{近似}$或$\textit{部分}$对称性时，最优或最好性能的模型是一个模型对齐于数据对称性的模型。在本文中，我们对这些直觉进行了正式的统一研究。首先，我们提出一般的数量界限，证明捕获任务特定对称性的模型将导致改进的泛化。事实上，我们的结果不要求变换是有限的，甚至不需要形成完整的....

    The explicit incorporation of task-specific inductive biases through symmetry has emerged as a general design precept in the development of high-performance machine learning models. For example, group equivariant neural networks have demonstrated impressive performance across various domains and applications such as protein and drug design. A prevalent intuition about such models is that the integration of relevant symmetry results in enhanced generalization. Moreover, it is posited that when the data and/or the model may only exhibit $\textit{approximate}$ or $\textit{partial}$ symmetry, the optimal or best-performing model is one where the model symmetry aligns with the data symmetry. In this paper, we conduct a formal unified investigation of these intuitions. To begin, we present general quantitative bounds that demonstrate how models capturing task-specific symmetries lead to improved generalization. In fact, our results do not require the transformations to be finite or even form
    
[^15]: 关于神经网络作为无限树状概率图模型的论文研究

    On Neural Networks as Infinite Tree-Structured Probabilistic Graphical Models. (arXiv:2305.17583v1 [stat.ML])

    [http://arxiv.org/abs/2305.17583](http://arxiv.org/abs/2305.17583)

    本文提出了一种创新方法，通过构建与神经网络完全对应的无限树状PGMs来解决深度神经网络(DNNs)缺乏PGMs的精确语义和明确定义的概率解释的问题。研究发现DNNs在前向传播时确实执行PGM推断的近似，这与现有研究不同，它阐明了DNNs对PGMs中的精确推理的更直接近似，潜在的好处包括改进DNNs的教学和解释，以及能够合并PGMs和DNNs的算法。

    

    深度神经网络(DNNs)缺乏概率图模型(PGMs)的精确语义和明确定义的概率解释。本文提出了一种创新方法，通过构建与神经网络完全对应的无限树状PGMs来解决这个问题。我们的研究揭示了DNNs在前向传播期间确实执行PGM推断的近似，这与曾经的神经网络描述为核机器或无限大小的高斯过程的现有研究不同，它阐明了DNNs对PGMs中的精确推理的更直接近似。潜在的好处包括改进DNNs的教学和解释，以及能够合并PGMs和DNNs的算法。

    Deep neural networks (DNNs) lack the precise semantics and definitive probabilistic interpretation of probabilistic graphical models (PGMs). In this paper, we propose an innovative solution by constructing infinite tree-structured PGMs that correspond exactly to neural networks. Our research reveals that DNNs, during forward propagation, indeed perform approximations of PGM inference that are precise in this alternative PGM structure. Not only does our research complement existing studies that describe neural networks as kernel machines or infinite-sized Gaussian processes, it also elucidates a more direct approximation that DNNs make to exact inference in PGMs. Potential benefits include improved pedagogy and interpretation of DNNs, and algorithms that can merge the strengths of PGMs and DNNs.
    
[^16]: 疾病患者个体根本原因的反事实公式化

    Counterfactual Formulation of Patient-Specific Root Causes of Disease. (arXiv:2305.17574v1 [cs.AI])

    [http://arxiv.org/abs/2305.17574](http://arxiv.org/abs/2305.17574)

    本文提出了一种针对疾病患者个体的根本原因的新公式，可以用于自动从数据中检测根本原因，并考虑了噪声标签和疾病流行率等因素，同时具有快速计算的优势。

    

    疾病的根本原因直观地对应于增加诊断可能性的根本顶点。然而，这种根本原因的描述缺乏计算机算法发展所需的严格数学公式。在以前的工作中，使用干预主义者帐户定义了疾病的病人特定根本原因，该帐户仅攀升到珍珠的因果Ladder的第二层。在这个理论性的文章中，我们通过提出反事实的定义来攀升到第三层，以匹配基于固定事实数据的临床直觉。然后，我们展示了如何使用可解释的人工智能的Shapley值为每个变量分配根因贡献得分。提出的疾病患者个体根本原因的反事实公式化考虑了噪声标签，适应了疾病的流行率，并允许快速计算，无需反事实模拟。

    Root causes of disease intuitively correspond to root vertices that increase the likelihood of a diagnosis. This description of a root cause nevertheless lacks the rigorous mathematical formulation needed for the development of computer algorithms designed to automatically detect root causes from data. Prior work defined patient-specific root causes of disease using an interventionalist account that only climbs to the second rung of Pearl's Ladder of Causation. In this theoretical piece, we climb to the third rung by proposing a counterfactual definition matching clinical intuition based on fixed factual data alone. We then show how to assign a root causal contribution score to each variable using Shapley values from explainable artificial intelligence. The proposed counterfactual formulation of patient-specific root causes of disease accounts for noisy labels, adapts to disease prevalence and admits fast computation without the need for counterfactual simulation.
    
[^17]: 通过赌博进行公平性审计

    Auditing Fairness by Betting. (arXiv:2305.17570v1 [stat.ML])

    [http://arxiv.org/abs/2305.17570](http://arxiv.org/abs/2305.17570)

    本文提供了一种通过赌博的方式进行公平性审计的方法，相比之前的方法，这种方法具有更高的实用性和效率，能够对不断产生的数据进行连续的监控，并处理因分布漂移导致的公平性问题。

    

    我们提供了实用、高效、非参数方法，用于审计已部署的分类和回归模型的公平性。相比之前依赖于固定样本量的方法，我们的方法是序贯的，并允许对不断产生的数据进行连续的监控，因此非常适用于跟踪现实世界系统的公平性。我们也允许数据通过概率策略进行收集，而不是从人口中均匀采样。这使得审计可以在为其他目的收集的数据上进行。此外，该策略可以随时间改变，并且不同的子人群可以使用不同的策略。最后，我们的方法可以处理因模型变更或基础人群变更导致的分布漂移。我们的方法基于最近关于 anytime-valid 推断和博弈统计学的进展，尤其是"通过赌博进行测试"框架。这些联系确保了我们的方法具有可解释性、快速和提供统计保证。

    We provide practical, efficient, and nonparametric methods for auditing the fairness of deployed classification and regression models. Whereas previous work relies on a fixed-sample size, our methods are sequential and allow for the continuous monitoring of incoming data, making them highly amenable to tracking the fairness of real-world systems. We also allow the data to be collected by a probabilistic policy as opposed to sampled uniformly from the population. This enables auditing to be conducted on data gathered for another purpose. Moreover, this policy may change over time and different policies may be used on different subpopulations. Finally, our methods can handle distribution shift resulting from either changes to the model or changes in the underlying population. Our approach is based on recent progress in anytime-valid inference and game-theoretic statistics-the "testing by betting" framework in particular. These connections ensure that our methods are interpretable, fast, 
    
[^18]: 基于虚拟粒子随机逼近的可证速限制变种的SVGD算法。

    Provably Fast Finite Particle Variants of SVGD via Virtual Particle Stochastic Approximation. (arXiv:2305.17558v1 [stat.ML])

    [http://arxiv.org/abs/2305.17558](http://arxiv.org/abs/2305.17558)

    本论文提出了两种基于虚拟粒子随机逼近的可证速限制变种的SVGD算法，具有可证速的有限粒子收敛率。

    

    Stein变分梯度下降（SVGD）是一种流行的变分推断算法，它模拟相互作用的粒子系统以近似从目标分布中采样，具有各种领域的令人印象深刻的经验性能。在理论上，它的群体（即，无限粒子）极限动力学已经得到了很好的研究，但是SVGD在有限粒子体制下的行为则不太清楚。在这项工作中，我们设计了两种计算效率高的SVGD变体，即VP-SVGD（从概念上讲很优雅）和GB-SVGD（从经验上看很有效），具有可证速的有限粒子收敛率。我们引入了“虚拟粒子”的概念，并在概率测度空间中开发了人口极限SVGD动力学的新型随机逼近方法，它们可以使用有限数量的粒子精确实现。我们的算法可以看作是SVGD的特定随机批处理逼近，比普通方法更具计算效率。

    Stein Variational Gradient Descent (SVGD) is a popular variational inference algorithm which simulates an interacting particle system to approximately sample from a target distribution, with impressive empirical performance across various domains. Theoretically, its population (i.e, infinite-particle) limit dynamics is well studied but the behavior of SVGD in the finite-particle regime is much less understood. In this work, we design two computationally efficient variants of SVGD, namely VP-SVGD (which is conceptually elegant) and GB-SVGD (which is empirically effective), with provably fast finite-particle convergence rates. We introduce the notion of \emph{virtual particles} and develop novel stochastic approximations of population-limit SVGD dynamics in the space of probability measures, which are exactly implementable using a finite number of particles. Our algorithms can be viewed as specific random-batch approximations of SVGD, which are computationally more efficient than ordinar
    
[^19]: 通过层次公平狄利克雷过程实现公平聚类。

    Fair Clustering via Hierarchical Fair-Dirichlet Process. (arXiv:2305.17557v1 [stat.ML])

    [http://arxiv.org/abs/2305.17557](http://arxiv.org/abs/2305.17557)

    本文提出了一种新的基于模型的公平聚类公式，该公式通过层次公平狄利克雷过程实现公平聚类的目标。

    

    机器学习驱动的决策制定和政策制定的出现，导致越来越多地关注算法的公平性。由于聚类是最常用的无监督机器学习方法之一，因此自然而然地出现了大量有关“公平聚类”的文献。聚类中公平的一种普遍概念要求群集是“平衡的”，即受保护属性的每个级别在每个群集中都要大约相等地代表。在保持原始框架的基础上，这方面的文献在各个方面迅速扩展。在本文中，我们提出了一种新的基于模型的公平聚类公式，补充了现有文献几乎完全基于优化适当的目标函数的不足之处。

    The advent of ML-driven decision-making and policy formation has led to an increasing focus on algorithmic fairness. As clustering is one of the most commonly used unsupervised machine learning approaches, there has naturally been a proliferation of literature on {\em fair clustering}. A popular notion of fairness in clustering mandates the clusters to be {\em balanced}, i.e., each level of a protected attribute must be approximately equally represented in each cluster. Building upon the original framework, this literature has rapidly expanded in various aspects. In this article, we offer a novel model-based formulation of fair clustering, complementing the existing literature which is almost exclusively based on optimizing appropriate objective functions.
    
[^20]: PFN是适用于实际贝叶斯优化的灵活模型。

    PFNs Are Flexible Models for Real-World Bayesian Optimization. (arXiv:2305.17535v1 [cs.LG])

    [http://arxiv.org/abs/2305.17535](http://arxiv.org/abs/2305.17535)

    本文使用灵活的PFN作为BO代理建模，该模型能够允许进一步信息纳入以进行非远视BO。在三种不同的问题上得到了很好的结果。

    

    本文使用先验数据拟合网络(PFNs)作为贝叶斯优化(BO)的灵活代理。PFN是一种神经过程，被训练用于近似后验预测分布(PPD)，适用于任何可有效采样的先验分布。我们描述了如何利用这种灵活性来进行BO的代理建模。我们使用PFN来模拟一个朴素高斯过程(GP)，一个先进的GP和一个贝叶斯神经网络(BNN)。此外，我们展示了如何将进一步的信息纳入先验，例如允许有关最优位置的提示(用户先验)，忽略不相关的维度，并通过学习获取函数来执行非远视BO。这些扩展的灵活性为使用PFN进行BO开辟了广阔的可能性。我们在人工高斯过程样本和三个不同的超参数优化测试平台上展示了PFN对BO的有用性：HPO-B、Bayesmark和PD1。

    In this paper, we use Prior-data Fitted Networks (PFNs) as a flexible surrogate for Bayesian Optimization (BO). PFNs are neural processes that are trained to approximate the posterior predictive distribution (PPD) for any prior distribution that can be efficiently sampled from. We describe how this flexibility can be exploited for surrogate modeling in BO. We use PFNs to mimic a naive Gaussian process (GP), an advanced GP, and a Bayesian Neural Network (BNN). In addition, we show how to incorporate further information into the prior, such as allowing hints about the position of optima (user priors), ignoring irrelevant dimensions, and performing non-myopic BO by learning the acquisition function. The flexibility underlying these extensions opens up vast possibilities for using PFNs for BO. We demonstrate the usefulness of PFNs for BO in a large-scale evaluation on artificial GP samples and three different hyperparameter optimization testbeds: HPO-B, Bayesmark, and PD1. We publish code 
    
[^21]: 随机梯度下降中动态稳定性的隐式正则化

    The Implicit Regularization of Dynamical Stability in Stochastic Gradient Descent. (arXiv:2305.17490v1 [stat.ML])

    [http://arxiv.org/abs/2305.17490](http://arxiv.org/abs/2305.17490)

    本文研究了随机梯度下降的动态稳定性隐式正则化，证明了其具有良好的泛化性。

    

    本文从“动态稳定性”的角度研究了随机梯度下降（SGD）的隐式正则化。我们首先修正了现有SGD稳定性分析的问题，展示了Hessian矩阵的Frobenius范数和迹与不同稳定性概念的关系。特别地，如果全局最小值在SGD中是线性稳定的，则Hessian矩阵的迹必须小于或等于$2/\eta$，其中$\eta$表示学习率。然而，对于梯度下降（GD），稳定性只对Hessian矩阵的最大特征值施加类似的约束。我们接着分析了这些稳定极值的泛化性质，着重关注了两层ReLU网络和对角线性网络。特别地，我们建立了这两个模型的尖锐度度量和某些参数规范之间的“等价性”，从而证明了SGD的稳定极值具有良好的泛化性。然而，GD的稳定性正则化只在特定情况下产生泛化效益。最后，我们将我们的理论应用于深度线性网络问题，结果表明它对某些模型的表现优于Lasso或岭正则化。

    In this paper, we study the implicit regularization of stochastic gradient descent (SGD) through the lens of {\em dynamical stability} (Wu et al., 2018). We start by revising existing stability analyses of SGD, showing how the Frobenius norm and trace of Hessian relate to different notions of stability. Notably, if a global minimum is linearly stable for SGD, then the trace of Hessian must be less than or equal to $2/\eta$, where $\eta$ denotes the learning rate. By contrast, for gradient descent (GD), the stability imposes a similar constraint but only on the largest eigenvalue of Hessian. We then turn to analyze the generalization properties of these stable minima, focusing specifically on two-layer ReLU networks and diagonal linear networks. Notably, we establish the {\em equivalence} between these metrics of sharpness and certain parameter norms for the two models, which allows us to show that the stable minima of SGD provably generalize well. By contrast, the stability-induced reg
    
[^22]: 深度变分病变-缺陷映射

    Deep Variational Lesion-Deficit Mapping. (arXiv:2305.17478v1 [cs.LG])

    [http://arxiv.org/abs/2305.17478](http://arxiv.org/abs/2305.17478)

    本论文使用深度变分神经网络架构实现病变-缺陷推断任务，建立表达性层次模型，可估计联合损伤和缺陷分布，条件为潜在神经底物。

    

    人脑的功能组织的因果映射需要自然起源病理损伤的必要性证据，其规模仅在足够灵活的推断模型中具备。这需要具有足够灵活性的推断模型，既可捕获病理损伤的可观测分布，也可捕获神经底物的未观测分布。目前的模型框架 - 无论是大多数单变量的还是多变量的 - 要么忽略了分布式的损伤缺陷关系，要么没有明确地对它们建模，而是依靠与预测任务无关的特征化。在这里，我们开始将深度生成神经网络架构应用于损伤-缺陷推断任务，将其定义为基于潜在神经底物的联合损伤和缺陷分布的表达性层次模型的估计。我们使用变分卷积体积自编码器实现了这样的深度损伤缺陷推断。我们引入了一个全面的框架

    Causal mapping of the functional organisation of the human brain requires evidence of \textit{necessity} available at adequate scale only from pathological lesions of natural origin. This demands inferential models with sufficient flexibility to capture both the observable distribution of pathological damage and the unobserved distribution of the neural substrate. Current model frameworks -- both mass-univariate and multivariate -- either ignore distributed lesion-deficit relations or do not model them explicitly, relying on featurization incidental to a predictive task. Here we initiate the application of deep generative neural network architectures to the task of lesion-deficit inference, formulating it as the estimation of an expressive hierarchical model of the joint lesion and deficit distributions conditioned on a latent neural substrate. We implement such deep lesion deficit inference with variational convolutional volumetric auto-encoders. We introduce a comprehensive framework
    
[^23]: 探究生成式数据增广的意义

    Toward Understanding Generative Data Augmentation. (arXiv:2305.17476v1 [cs.LG])

    [http://arxiv.org/abs/2305.17476](http://arxiv.org/abs/2305.17476)

    生成式数据增广通过从训练的生成模型中获得虚假的标记示例，提高分类性能；本文建立了一个普遍的稳定性界限，并发现其效果与生成模型的选择和训练集大小密切相关。

    

    通过从训练的有条件生成模型中获得虚假的标记示例，生成式数据增广可以扩展数据集，并提高各种学习任务（包括（半）监督学习、少样本学习和对抗性鲁棒学习）中的分类性能。然而，目前很少有理论工作探究生成式数据增广的效果。为了填补这一空白，我们在这个非独立和同分布（non-i.i.d.）的设置中建立了一个普遍的稳定性界限，其中学习的分布依赖于原始训练集，通常与真实分布不同。我们的理论结果包括学习分布和真实分布之间的差异。结果表明，当发散项的阶数为$ o(\max\left( \log(m)\beta_m, 1 / \sqrt{m})\right)$时，生成式数据增广可以享受更快的学习速率，其中$m$为训练集大小，$\beta_m$为相应的稳定性常数。我们发现，界限的大小与发散阶数和训练集大小成正比，这表明生成式数据增广的效果与生成模型的选择和训练集的大小密切相关。

    Generative data augmentation, which scales datasets by obtaining fake labeled examples from a trained conditional generative model, boosts classification performance in various learning tasks including (semi-)supervised learning, few-shot learning, and adversarially robust learning. However, little work has theoretically investigated the effect of generative data augmentation. To fill this gap, we establish a general stability bound in this not independently and identically distributed (non-i.i.d.) setting, where the learned distribution is dependent on the original train set and generally not the same as the true distribution. Our theoretical result includes the divergence between the learned distribution and the true distribution. It shows that generative data augmentation can enjoy a faster learning rate when the order of divergence term is $o(\max\left( \log(m)\beta_m, 1 / \sqrt{m})\right)$, where $m$ is the train set size and $\beta_m$ is the corresponding stability constant. We f
    
[^24]: 通过 $\ell_1-\ell_2$ 优化进行结构模型选择

    Structured model selection via $\ell_1-\ell_2$ optimization. (arXiv:2305.17467v1 [stat.ML])

    [http://arxiv.org/abs/2305.17467](http://arxiv.org/abs/2305.17467)

    通过稀疏最小二乘拟合一大组候选函数，使用 $\ell_1-\ell_2$ 稀疏优化方法进行结构模型选择，实现从不充分且嘈杂的时空数据中识别结构化动态系统；该方法在合成数据集上得到了验证，并证明具有理论保证和高效性。

    

    自动化模型选择在科学和工程中具有重要应用。本文提出了一种学习方法，通过稀疏最小二乘拟合一大组候选函数，用一种非凸 $\ell_1-\ell_2$ 稀疏优化方法求解，通过交替方向乘法的方法进行。我们证明，如果候选函数集合形成边界正交系统的结构随机采样矩阵，就可以通过伯恩斯坦样式的不等式和一致性条件稳定恢复，并且误差有界。该学习方法在由粘性Burgers'方程和两个反应扩散方程产生的合成数据上进行了验证。计算结果证明了成功的理论保证和相对于环境维数和候选函数数量的效率。

    Automated model selection is an important application in science and engineering. In this work, we develop a learning approach for identifying structured dynamical systems from undersampled and noisy spatiotemporal data. The learning is performed by a sparse least-squares fitting over a large set of candidate functions via a nonconvex $\ell_1-\ell_2$ sparse optimization solved by the alternating direction method of multipliers. Using a Bernstein-like inequality with a coherence condition, we show that if the set of candidate functions forms a structured random sampling matrix of a bounded orthogonal system, the recovery is stable and the error is bounded. The learning approach is validated on synthetic data generated by the viscous Burgers' equation and two reaction-diffusion equations. The computational results demonstrate the theoretical guarantees of success and the efficiency with respect to the ambient dimension and the number of candidate functions.
    
[^25]: 关于随机SVD的噪声敏感性

    On the Noise Sensitivity of the Randomized SVD. (arXiv:2305.17435v1 [cs.IT])

    [http://arxiv.org/abs/2305.17435](http://arxiv.org/abs/2305.17435)

    通过对R-SVD在低秩信号加噪声测量模型下的分析，证明了当信噪比(SNR)超过某个依赖于降维因子的可检测门限时，R-SVD产生的最大奇异值是一个离群值；在门限以下，没有离群值从奇异值块中产生

    

    随机奇异值分解(R-SVD)是一种流行的基于草图的算法，用于有效计算大矩阵的部分奇异值分解。当矩阵是低秩时，R-SVD可以精确地产生其部分奇异值分解；但当秩较大时，它只能产生近似值。受数据科学和主成分分析(PCA)应用的驱动，我们在低秩信号加噪声测量模型下分析了R-SVD；具体来说，当其输入为尖峰型随机矩阵时。证明了R-SVD产生的奇异值表现出类似BBP的相变：当信噪比(SNR)超过某个依赖于降维因子的可检测门限时，最大奇异值是一个离群值；在门限以下，没有离群值从奇异值块中产生。我们进一步计算了地面真值信号奇异向量与R-SVD产生的近似值之间的重叠的渐近公式。降维具有负面的影响。

    The randomized singular value decomposition (R-SVD) is a popular sketching-based algorithm for efficiently computing the partial SVD of a large matrix. When the matrix is low-rank, the R-SVD produces its partial SVD exactly; but when the rank is large, it only yields an approximation.  Motivated by applications in data science and principal component analysis (PCA), we analyze the R-SVD under a low-rank signal plus noise measurement model; specifically, when its input is a spiked random matrix. The singular values produced by the R-SVD are shown to exhibit a BBP-like phase transition: when the SNR exceeds a certain detectability threshold, that depends on the dimension reduction factor, the largest singular value is an outlier; below the threshold, no outlier emerges from the bulk of singular values. We further compute asymptotic formulas for the overlap between the ground truth signal singular vectors and the approximations produced by the R-SVD.  Dimensionality reduction has the adve
    
[^26]: 具有对抗性损失和转换的无遗憾在线强化学习

    No-Regret Online Reinforcement Learning with Adversarial Losses and Transitions. (arXiv:2305.17380v1 [cs.LG])

    [http://arxiv.org/abs/2305.17380](http://arxiv.org/abs/2305.17380)

    本文提出了一种算法，可以处理对抗性损失和对抗性转换，且后悔逐渐增加与对手的恶意程度成比例。

    

    现有的对抗性马尔可夫决策过程的在线学习算法可以在与对手的$ T $轮交互之后实现${ O}(\sqrt{T})$的后悔，即使损失函数是由对手任意选择的，但前提是转移函数必须固定。这是因为已经有研究表明，对抗性转移函数使无悔学习变得不可能。尽管存在这种不可能性结果，我们开发了可以处理对抗性损失和对抗性转换的算法，后悔逐渐增加与对手的恶意程度成比例。更具体地说，我们首先提出了一种算法，它的后悔为$\widetilde{{O}}(\sqrt{T} + C^{\textsf{P}})$，其中$C^{\textsf{P}}$表示转换函数的对抗性，最多可以为${O}(T)$。虽然此算法本身需要$C^{\textsf{P}}$的知识，但我们还开发了一种黑盒缩减方法来消除此要求。此外，我们还展示了一种进一步的方法，使得算法能够处理任意长度的锚定期。

    Existing online learning algorithms for adversarial Markov Decision Processes achieve ${O}(\sqrt{T})$ regret after $T$ rounds of interactions even if the loss functions are chosen arbitrarily by an adversary, with the caveat that the transition function has to be fixed. This is because it has been shown that adversarial transition functions make no-regret learning impossible. Despite such impossibility results, in this work, we develop algorithms that can handle both adversarial losses and adversarial transitions, with regret increasing smoothly in the degree of maliciousness of the adversary. More concretely, we first propose an algorithm that enjoys $\widetilde{{O}}(\sqrt{T} + C^{\textsf{P}})$ regret where $C^{\textsf{P}}$ measures how adversarial the transition functions are and can be at most ${O}(T)$. While this algorithm itself requires knowledge of $C^{\textsf{P}}$, we further develop a black-box reduction approach that removes this requirement. Moreover, we also show that furth
    
[^27]: 稳定性惩罚自适应跟随正则化领袖：稀疏性、游戏依赖性和最佳世界的并存

    Stability-penalty-adaptive Follow-the-regularized-leader: Sparsity, Game-dependency, and Best-of-both-worlds. (arXiv:2305.17301v1 [cs.LG])

    [http://arxiv.org/abs/2305.17301](http://arxiv.org/abs/2305.17301)

    本文开发了一种稳定性惩罚自适应（SPA）学习率，该学习率使FTRL具有稀疏性、游戏依赖性和最佳世界（BOBW）三种适应性类型，其中SPA-sparse算法可适应于未知的稀疏级别，SPA-game-dependency算法可根据所玩的游戏自适应地改变其行为，BOBW算法则是既具有稀疏性又具有游戏依赖性的适应性算法。

    

    在顺序决策问题中，适应问题的困难程度是扩展算法适用性的关键属性。跟随正则化领袖近年来成为获取淘汰法中各种类型适应性的最有前途的方法之一。为了进一步推广这种适应性，我们为FTRL开发了一个通用的自适应学习率，称为稳定性惩罚自适应（SPA）学习率。该学习率产生的遗憾界共同取决于算法的稳定性和惩罚，其中FTRL的遗憾通常被分解。凭借这个结果，我们建立了几个具有三种适应性类型的算法：稀疏性、游戏依赖性和最佳世界（BOBW）。稀疏性经常出现在真实世界的问题中，但是，现有的稀疏多臂赌博算法$k$-arms假定事先已知稀疏级别$s \leq k$，而这在真实世界的情况下通常不是情况。为了适应未知的稀疏级别，我们提出了一种新算法SPA-sparse，该算法显示比现有稀疏算法的性能提高了。游戏依赖性是另一种适应性类型，当用于生成数据的游戏发生变化时，即必需的。我们提出了一种新算法SPA-game-dependency，该算法根据所玩的游戏自适应地改变其行为，并表明它比非自适应算法的性能更好。最后，我们提出了一个既具有稀疏性又具有游戏依赖性适应性的BOBW算法，并显示它比仅集中于一种适应性类型的算法表现更好。

    Adaptivity to the difficulties of a problem is a key property in sequential decision-making problems to broaden the applicability of algorithms. Follow-the-Regularized-Leader (FTRL) has recently emerged as one of the most promising approaches for obtaining various types of adaptivity in bandit problems. Aiming to further generalize this adaptivity, we develop a generic adaptive learning rate, called Stability-Penalty-Adaptive (SPA) learning rate for FTRL. This learning rate yields a regret bound jointly depending on stability and penalty of the algorithm, into which the regret of FTRL is typically decomposed. With this result, we establish several algorithms with three types of adaptivity: sparsity, game-dependency, and Best-of-Both-Worlds (BOBW). Sparsity frequently appears in real-world problems. However, existing sparse multi-armed bandit algorithms with $k$-arms assume that the sparsity level $s \leq k$ is known in advance, which is often not the case in real-world scenarios. To ad
    
[^28]: 提高决策树模型的稳定性

    Improving Stability in Decision Tree Models. (arXiv:2305.17299v1 [stat.ML])

    [http://arxiv.org/abs/2305.17299](http://arxiv.org/abs/2305.17299)

    本文通过医疗应用的视角，提出了一种新的决策树距离度量，并用它来确定树的稳定水平。我们提出了一种新的培训稳定决策树的方法，并探究稳定性、预测能力和可解释性之间不可避免的权衡。

    

    由于其结构易于理解，决策树通常在需要可解释性的应用中被广泛使用。近期的工作集中于改进决策树的各个方面，包括预测能力和鲁棒性；然而，其不稳定性虽然有充分的记录，但却得到了较少的关注。本文通过实际的医疗应用的视角，提出了稳定化决策树模型的一小步。由于稳定性和可解释性在医疗领域具有重要性，我们介绍了一种新的决策树距离度量，并将其用于确定树的稳定水平。我们提出了一种新的培训稳定决策树的方法，并调查了决策树模型之间不可避免的权衡，包括在稳定性、预测能力和可解释性之间。我们通过对六个数据集的广泛定量和定性分析展示了所提议方法的价值。

    Owing to their inherently interpretable structure, decision trees are commonly used in applications where interpretability is essential. Recent work has focused on improving various aspects of decision trees, including their predictive power and robustness; however, their instability, albeit well-documented, has been addressed to a lesser extent. In this paper, we take a step towards the stabilization of decision tree models through the lens of real-world health care applications due to the relevance of stability and interpretability in this space. We introduce a new distance metric for decision trees and use it to determine a tree's level of stability. We propose a novel methodology to train stable decision trees and investigate the existence of trade-offs that are inherent to decision tree models - including between stability, predictive power, and interpretability. We demonstrate the value of the proposed methodology through an extensive quantitative and qualitative analysis of six 
    
[^29]: 无独立性的泛化误差：去噪、线性回归和迁移学习

    Generalization Error without Independence: Denoising, Linear Regression, and Transfer Learning. (arXiv:2305.17297v1 [cs.LG])

    [http://arxiv.org/abs/2305.17297](http://arxiv.org/abs/2305.17297)

    本论文研究了具有低秩结构但非独立同分布数据的情况，在分离训练和测试分布的假设下，解决了分布偏移问题，实验结果表明，在分布偏移的情况下，本方法显著提高了泛化误差的性能。

    

    研究线性模型在真实数据中的泛化能力是统计学习中的一个核心问题。先前的一些重要工作验证了理论工作与真实数据的相关性，但这些工作由于技术假设存在限制，这些假设包括具有良好条件的协方差矩阵以及具有独立同分布数据，这些假设在真实数据中并不一定成立。此外，以前的一些关于分布偏移的工作通常对训练和测试数据的联合分布进行技术假设，并且不在真实数据上进行测试。为了解决这些问题并更好地对真实数据进行建模，我们研究了具有低秩结构但非独立同分布数据的情况，同时通过分离训练和测试分布的假设来解决分布偏移问题。我们还在这些松弛的假设下，研究了去噪问题、线性回归和迁移学习。我们的实验结果表明，相比以前的方法，在分布偏移的情况下，我们的方法显著提高了泛化误差的性能。

    Studying the generalization abilities of linear models with real data is a central question in statistical learning. While there exist a limited number of prior important works (Loureiro et al. (2021A, 2021B), Wei et al. 2022) that do validate theoretical work with real data, these works have limitations due to technical assumptions. These assumptions include having a well-conditioned covariance matrix and having independent and identically distributed data. These assumptions are not necessarily valid for real data. Additionally, prior works that do address distributional shifts usually make technical assumptions on the joint distribution of the train and test data (Tripuraneni et al. 2021, Wu and Xu 2020), and do not test on real data.  In an attempt to address these issues and better model real data, we look at data that is not I.I.D. but has a low-rank structure. Further, we address distributional shift by decoupling assumptions on the training and test distribution. We provide anal
    
[^30]: GC-Flow: 一种基于图的流网络用于有效聚类

    GC-Flow: A Graph-Based Flow Network for Effective Clustering. (arXiv:2305.17284v1 [cs.LG])

    [http://arxiv.org/abs/2305.17284](http://arxiv.org/abs/2305.17284)

    GC-Flow是一种生成模型，可以同时建模类别条件概率和类别先验，通过配备高斯混合表示空间，保持预测能力的同时实现了良好分离的聚类。

    

    图卷积网络（GCN）是直接建模半监督分类图数据类后验概率$p（y|\mathbf{x}）$的判别模型。虽然作为一种表示学习方法非常有效，但是从GCN中提取的节点表征常缺少有效聚类所需的有用信息，因为它们的目标不同。本研究设计了归一化流，用于替换GCN层，构建一种生成模型，同时建模类别条件概率$p(\mathbf{x}|y)$和类别先验$p(y)$。由此产生的神经网络GC-Flow保留了图卷积操作，同时配备了高斯混合表示空间。这有两个好处：它不仅保持了GCN的预测能力，还由于表示空间的结构而产生了良好分离的聚类。我们在各种基准数据集上展示了这些优势。此外，我们还展示了额外的参数化正则化优势和通用性。

    Graph convolutional networks (GCNs) are \emph{discriminative models} that directly model the class posterior $p(y|\mathbf{x})$ for semi-supervised classification of graph data. While being effective, as a representation learning approach, the node representations extracted from a GCN often miss useful information for effective clustering, because the objectives are different. In this work, we design normalizing flows that replace GCN layers, leading to a \emph{generative model} that models both the class conditional likelihood $p(\mathbf{x}|y)$ and the class prior $p(y)$. The resulting neural network, GC-Flow, retains the graph convolution operations while being equipped with a Gaussian mixture representation space. It enjoys two benefits: it not only maintains the predictive power of GCN, but also produces well-separated clusters, due to the structuring of the representation space. We demonstrate these benefits on a variety of benchmark data sets. Moreover, we show that additional par
    
[^31]: 通过拓扑交换优化NOTEARS目标

    Optimizing NOTEARS Objectives via Topological Swaps. (arXiv:2305.17277v1 [stat.ML])

    [http://arxiv.org/abs/2305.17277](http://arxiv.org/abs/2305.17277)

    本文提出了一种双层算法来解决学习DAGs中的非凸优化问题，其中外层利用拓扑交换优化拓扑顺序，通过开发一种候选交换对的方法，算法在学习高质量DAGs方面具有高效和稳定的优势。

    

    最近，在学习有向无环图（DAGs）的背景下，出现了一类有趣的非凸优化问题。这些问题涉及到在给定损失或得分函数的情况下，最小化一个惩罚图中存在循环的非凸连续约束。在这项工作中，我们探讨了与这类非凸程序相关的优化挑战。为了解决这些问题，我们提出了一种双层算法，以新颖的方式利用非凸约束。算法的外层通过迭代地交换DAG的拓扑顺序中的节点对来优化拓扑顺序。我们方法的一个关键创新是，开发了一种有效的方法来为每次迭代生成一组候选交换对。在内层中，给定拓扑顺序，我们利用能够处理线性约束的现成求解器。我们所提出算法的主要优势是，它保证收敛到优化问题的一个稳定点，而现有方法可能会陷入亚最优解中。我们在合成和真实世界数据集上的实验证明了我们算法在学习高质量DAGs方面的有效性和效率。

    Recently, an intriguing class of non-convex optimization problems has emerged in the context of learning directed acyclic graphs (DAGs). These problems involve minimizing a given loss or score function, subject to a non-convex continuous constraint that penalizes the presence of cycles in a graph. In this work, we delve into the optimization challenges associated with this class of non-convex programs. To address these challenges, we propose a bi-level algorithm that leverages the non-convex constraint in a novel way. The outer level of the algorithm optimizes over topological orders by iteratively swapping pairs of nodes within the topological order of a DAG. A key innovation of our approach is the development of an effective method for generating a set of candidate swapping pairs for each iteration. At the inner level, given a topological order, we utilize off-the-shelf solvers that can handle linear constraints. The key advantage of our proposed algorithm is that it is guaranteed to
    
[^32]: FineMorphs:用于回归的仿射-微分同胚序列模型

    FineMorphs: Affine-diffeomorphic sequences for regression. (arXiv:2305.17255v1 [stat.ML])

    [http://arxiv.org/abs/2305.17255](http://arxiv.org/abs/2305.17255)

    FineMorphs是一种多元回归模型，通过形状分析中的微分同胚概念对模型状态进行优化，能够自然地减少（或增加）维度并适应大数据集。

    

    本文提出了一种仿射和微分同胚变换序列的多元回归模型FineMorphs。该模型利用形状分析的概念，在学习期间通过由光滑向量场生成的微分同胚优化地“重塑”模型状态。仿射变换和向量场在最优控制环境中进行优化，该模型可以通过次优向量场自然地减少（或增加）维度并适应大数据集。我们推导了该模型的解存在性证明和最优性的必要条件。在真实数据集上进行的实验结果表明，FineMorphs在与文献中最先进和基于TensorFlow的稠密连接神经网络的比较中，取得了有利的结果。

    A multivariate regression model of affine and diffeomorphic transformation sequences - FineMorphs - is presented. Leveraging concepts from shape analysis, model states are optimally "reshaped" by diffeomorphisms generated by smooth vector fields during learning. Affine transformations and vector fields are optimized within an optimal control setting, and the model can naturally reduce (or increase) dimensionality and adapt to large datasets via suboptimal vector fields. An existence proof of solution and necessary conditions for optimality for the model are derived. Experimental results on real datasets from the UCI repository are presented, with favorable results in comparison with state-of-the-art in the literature and densely-connected neural networks in TensorFlow.
    
[^33]: 因果成分分析

    Causal Component Analysis. (arXiv:2305.17225v1 [stat.ML])

    [http://arxiv.org/abs/2305.17225](http://arxiv.org/abs/2305.17225)

    本文介绍了一个中间问题：因果成分分析(CauCA)，它是独立成分分析(ICA)和因果表示学习(CRL)的泛化和特例，其目标是学习解混函数和因果机制，预设了因果图的知识。

    

    独立成分分析(ICA)的目标是从混合观测到的变量中恢复独立的潜在变量。而因果表示学习(CRL)的目标是推断因果关系强相关性的潜在变量，以及编码它们的因果关系的未知图。我们引入了一个中间问题，称为因果成分分析(CauCA)。CauCA可以被看作是ICA的一种推广，对潜在成分之间的因果依赖建模，也是CRL的一个特例。与CRL不同的是，它预设了因果图的知识，仅关注于学习解混函数和因果机制。所有关于CauCA回收基础真相的不可能结果也适用于CRL，而可能性结果可以作为扩展CRL的基础。我们将从对潜在因果变量实施不同类型干预的多个数据集中表征CauCA的可识别性。

    Independent Component Analysis (ICA) aims to recover independent latent variables from observed mixtures thereof. Causal Representation Learning (CRL) aims instead to infer causally related (thus often statistically dependent) latent variables, together with the unknown graph encoding their causal relationships. We introduce an intermediate problem termed Causal Component Analysis (CauCA). CauCA can be viewed as a generalization of ICA, modelling the causal dependence among the latent components, and as a special case of CRL. In contrast to CRL, it presupposes knowledge of the causal graph, focusing solely on learning the unmixing function and the causal mechanisms. Any impossibility results regarding the recovery of the ground truth in CauCA also apply for CRL, while possibility results may serve as a stepping stone for extensions to CRL. We characterize CauCA identifiability from multiple datasets generated through different types of interventions on the latent causal variables. As a
    
[^34]: 非凸梯度下降法快速极小化低秩矩阵估计

    Fast and Minimax Optimal Estimation of Low-Rank Matrices via Non-Convex Gradient Descent. (arXiv:2305.17224v1 [math.OC])

    [http://arxiv.org/abs/2305.17224](http://arxiv.org/abs/2305.17224)

    本文提出一种针对低秩矩阵估计的方法，在保证极小极值优化性能的同时，解决了非凸梯度下降收敛缓慢的问题。

    

    本文研究了从噪声测量中估计低秩矩阵的问题，特别是旨在实现极小极值误差。在实践中，由于非凸梯度下降的能力可以扩展到大规模真实世界数据集，这个问题通常使用非凸梯度下降来解决。理论上，非凸梯度下降能够实现极小极值误差。但在实践中，它经常收敛得非常缓慢，以至于甚至无法在合理的时间内提供适度准确的估计值。另一方面，通过重新缩放或预处理改进非凸梯度下降的收敛方法也会大大放大测量误差，导致得到的估计比理论上可实现的极小极值误差少几个数量级的准确性。在本文中，我们提出了一种对通常的非凸梯度下降方法进行轻微修改的方法，解决了收敛缓慢的问题，同时可证明保留其极小极值优化性能。

    We study the problem of estimating a low-rank matrix from noisy measurements, with the specific goal of achieving minimax optimal error. In practice, the problem is commonly solved using non-convex gradient descent, due to its ability to scale to large-scale real-world datasets. In theory, non-convex gradient descent is capable of achieving minimax error. But in practice, it often converges extremely slowly, such that it cannot even deliver estimations of modest accuracy within reasonable time. On the other hand, methods that improve the convergence of non-convex gradient descent, through rescaling or preconditioning, also greatly amplify the measurement noise, resulting in estimations that are orders of magnitude less accurate than what is theoretically achievable with minimax optimal error. In this paper, we propose a slight modification to the usual non-convex gradient descent method that remedies the issue of slow convergence, while provably preserving its minimax optimality. Our p
    
[^35]: 功能流匹配

    Functional Flow Matching. (arXiv:2305.17209v1 [cs.LG])

    [http://arxiv.org/abs/2305.17209](http://arxiv.org/abs/2305.17209)

    本文介绍了一种名为功能流匹配（FFM）的函数空间生成模型，该模型利用概率测度插值和学习底层函数空间上生成测度的向量场来生成数据分布。这种无需似然或模拟的方法在合成和真实世界基准数据集上表现优异，优于最近提出的几种函数空间生成模型。

    

    本文提出了一种名为功能流匹配（Functional Flow Matching, FFM）的函数空间生成模型，该模型将最近引入的流匹配（Flow Matching）直接推广到无限维空间中进行。我们的方法首先定义了一组概率测度路径，在固定的高斯测度和数据分布之间进行插值，然后学习函数的底层空间上生成此测度路径的向量场。我们的方法不依赖于似然或模拟，因此非常适合函数空间的设置。我们不仅提供构建这种模型的理论框架，还对我们的技术进行了经验评估。通过对合成和真实世界基准数据集的实验，我们证明了我们提出的FFM方法优于最近提出的几种函数空间生成模型。

    In this work, we propose Functional Flow Matching (FFM), a function-space generative model that generalizes the recently-introduced Flow Matching model to operate directly in infinite-dimensional spaces. Our approach works by first defining a path of probability measures that interpolates between a fixed Gaussian measure and the data distribution, followed by learning a vector field on the underlying space of functions that generates this path of measures. Our method does not rely on likelihoods or simulations, making it well-suited to the function space setting. We provide both a theoretical framework for building such models and an empirical evaluation of our techniques. We demonstrate through experiments on synthetic and real-world benchmarks that our proposed FFM method outperforms several recently proposed function-space generative models.
    
[^36]: 向量值随机特征学习的误差界分析

    Error Bounds for Learning with Vector-Valued Random Features. (arXiv:2305.17170v1 [stat.ML])

    [http://arxiv.org/abs/2305.17170](http://arxiv.org/abs/2305.17170)

    本文提供了对向量值随机特征学习的完整误差分析，包括在模型错误说明下向量值RF估计器的强一致性和在良好规定的情况下极小化最优收敛速率。

    

    本文提供了对于向量值随机特征学习的完整误差分析。该理论是针对完全通用的无限维度输入-输出设定中的RF Ridge回归而开发的，但仍适用于并改进了现有的有限维度分析。与文献中其他类似的工作相比，本文提出的方法依赖于底层风险函数的直接分析，完全避免了基于随机矩阵的显式RF Ridge回归解决方案公式的使用。这消除了随机矩阵理论中的浓度结果或其对随机算子的推广的需求。本文建立的主要结果包括在模型错误说明下向量值RF估计器的强一致性和在良好规定的情况下极小化最优收敛速率。实现这些收敛速率所需的参数复杂度(随机特征数量)和样本复杂度(标记数据数量)与

    This paper provides a comprehensive error analysis of learning with vector-valued random features (RF). The theory is developed for RF ridge regression in a fully general infinite-dimensional input-output setting, but nonetheless applies to and improves existing finite-dimensional analyses. In contrast to comparable work in the literature, the approach proposed here relies on a direct analysis of the underlying risk functional and completely avoids the explicit RF ridge regression solution formula in terms of random matrices. This removes the need for concentration results in random matrix theory or their generalizations to random operators. The main results established in this paper include strong consistency of vector-valued RF estimators under model misspecification and minimax optimal convergence rates in the well-specified setting. The parameter complexity (number of random features) and sample complexity (number of labeled data) required to achieve such rates are comparable with 
    
[^37]: 带有复值的深窄神经网络的普适逼近

    Universal approximation with complex-valued deep narrow neural networks. (arXiv:2305.16910v1 [math.FA])

    [http://arxiv.org/abs/2305.16910](http://arxiv.org/abs/2305.16910)

    本文研究了具有有界宽度和任意深度的复值神经网络的普适性，发现当且仅当激活函数既不是全纯的，也不是反全纯的，也不是 $\mathbb{R}$-仿射的时，深窄的复值网络具有普适逼近能力。我们还发现足够的宽度依赖于考虑的激活函数，对于一类可允许的激活函数，宽度为 $n+m+4$ 是足够的。

    

    我们研究了具有有界宽度和任意深度的复值神经网络的普适性。在温和的假设下，我们给出了那些激活函数 $\varrho:\mathbb{CC}\to \mathbb{C}$ 的完整描述，这些函数具有这样一个属性：它们关联的网络是普适的，即能够在紧致域上逼近连续函数至任意精度。准确地说，我们表明了当且仅当它们的激活函数既不是全纯的，也不是反全纯的，也不是 $\mathbb{R}$-仿射的，深窄的复值网络是普适的。这是一个比宽度任意、深度固定的对偶设置中更大的函数类。与实值情况不同的是，足够的宽度依赖于考虑的激活函数。我们表明，宽度为 $2n+2m+5$ 总是足够的，并且通常 $\max\{2n,2m\}$ 是必要的。然而，我们证明了对于一类可允许的激活函数，宽度为 $n+m+4$ 是足够的。

    We study the universality of complex-valued neural networks with bounded widths and arbitrary depths. Under mild assumptions, we give a full description of those activation functions $\varrho:\mathbb{CC}\to \mathbb{C}$ that have the property that their associated networks are universal, i.e., are capable of approximating continuous functions to arbitrary accuracy on compact domains. Precisely, we show that deep narrow complex-valued networks are universal if and only if their activation function is neither holomorphic, nor antiholomorphic, nor $\mathbb{R}$-affine. This is a much larger class of functions than in the dual setting of arbitrary width and fixed depth. Unlike in the real case, the sufficient width differs significantly depending on the considered activation function. We show that a width of $2n+2m+5$ is always sufficient and that in general a width of $\max\{2n,2m\}$ is necessary. We prove, however, that a width of $n+m+4$ suffices for a rich subclass of the admissible acti
    
[^38]: 神经控制微分方程的泛化能力研究

    On the Generalization Capacities of Neural Controlled Differential Equations. (arXiv:2305.16791v1 [stat.ML])

    [http://arxiv.org/abs/2305.16791](http://arxiv.org/abs/2305.16791)

    本文研究了使用神经控制微分方程进行监督学习的泛化能力问题，通过量化离散化偏差和利普希茨函数逼近误差，得到了经验风险最小化器与贝叶斯最优风险的泛化差距上界。

    

    本文研究了使用神经控制微分方程（Kidger，Morrill等，2020）从不规则采样的时间序列样本中预测结果的监督学习设置。在我们的框架中，时间序列是一个未观察到的连续路径的离散化，结果通过一个具有未知向量场的控制微分方程依赖于这个路径。使用离散数据进行学习会引入离散偏差，我们精确地量化了这种偏差。通过使用关于控制微分方程流的连续性的理论结果，我们展示了逼近偏差直接与由浅层神经网络定义生成模型的利普希茨函数的逼近误差相关。通过结合最近的工作将神经网络的利普希茨常数与其泛化能力联系起来，我们上界了经验风险最小化器达到的期望损失与贝叶斯最优风险之间的泛化差距。

    We consider a supervised learning setup in which the goal is to predicts an outcome from a sample of irregularly sampled time series using Neural Controlled Differential Equations (Kidger, Morrill, et al. 2020). In our framework, the time series is a discretization of an unobserved continuous path, and the outcome depends on this path through a controlled differential equation with unknown vector field. Learning with discrete data thus induces a discretization bias, which we precisely quantify. Using theoretical results on the continuity of the flow of controlled differential equations, we show that the approximation bias is directly related to the approximation error of a Lipschitz function defining the generative model by a shallow neural network. By combining these result with recent work linking the Lipschitz constant of neural networks to their generalization capacities, we upper bound the generalization gap between the expected loss attained by the empirical risk minimizer and th
    
[^39]: 对比学习学到了哪些特征？关于简易偏差在类坍塌和特征抑制中的作用

    Which Features are Learnt by Contrastive Learning? On the Role of Simplicity Bias in Class Collapse and Feature Suppression. (arXiv:2305.16536v1 [cs.LG])

    [http://arxiv.org/abs/2305.16536](http://arxiv.org/abs/2305.16536)

    对比学习是一种表示学习技术，对于有监督的情况易于产生类坍塌，无监督情况下易于抑制类别相关的复杂特征；随机梯度下降方法偏向于寻找更简单的解决方案是导致这种现象的关键因素。

    

    对比学习具备无监督和有监督学习的表示学习技术，在有监督场景下易于坍塌同一类别内的子类表示，丢失一部分特征信息；而无监督学习则可能通过学习易于处理的类别无关特征而无视一些类别相关的复杂特征信息，这两种方法都会显著地降低表征的质量。本文提出了第一个统一严谨的框架来理解测试时的类坍塌和特征抑制产生的原因，相关分析表明，（随机）梯度下降方法偏向于寻找更简单的解决方案是导致子类表示坍塌和类别相关的复杂特征被抑制的关键因素。此外，我们利用提高嵌入维度和改进数据增强的方法来提供有效的预防措施。

    Contrastive learning (CL) has emerged as a powerful technique for representation learning, with or without label supervision. However, supervised CL is prone to collapsing representations of subclasses within a class by not capturing all their features, and unsupervised CL may suppress harder class-relevant features by focusing on learning easy class-irrelevant features; both significantly compromise representation quality. Yet, there is no theoretical understanding of \textit{class collapse} or \textit{feature suppression} at \textit{test} time. We provide the first unified theoretically rigorous framework to determine \textit{which} features are learnt by CL. Our analysis indicate that, perhaps surprisingly, bias of (stochastic) gradient descent towards finding simpler solutions is a key factor in collapsing subclass representations and suppressing harder class-relevant features. Moreover, we present increasing embedding dimensionality and improving the quality of data augmentations 
    
[^40]: 能见度集合预测的统计后处理

    Statistical post-processing of visibility ensemble forecasts. (arXiv:2305.15325v1 [stat.AP])

    [http://arxiv.org/abs/2305.15325](http://arxiv.org/abs/2305.15325)

    本论文研究了后处理能见度集合预测的不同方法，发现非参数密度估计和高斯混合模型方法表现良好，并且可以显着提高集合预测的技能和可靠性。

    

    能够准确可靠地预测能见度对于飞行气象，水路和道路运输具有至关重要的意义。现今，一些气象服务提供能见度的集合预测; 然而，相比于其他变量（如温度或风速），能见度预测的技能和可靠性降低很多。因此，强烈建议采用某种形式的校准，通常意味着通过参数或非参数方法（包括基于机器学习的技术）估计所涉及的天气数量的预测分布。由于根据世界气象组织的建议，通常以离散值报告能见度观测值，因此该特定变量的预测分布是离散概率分布，因此校准可以简化为分类问题。基于欧洲中期天气预报中心的能见度集合预测（ECMWF），我们研究了不同的方法用于后处理能见度概率预测。我们的发现表明，非参数密度估计和高斯混合模型方法表现良好，并且可以显着提高集合预测的技能和可靠性。

    To be able to produce accurate and reliable predictions of visibility has crucial importance in aviation meteorology, as well as in water- and road transportation. Nowadays, several meteorological services provide ensemble forecasts of visibility; however, the skill, and reliability of visibility predictions are far reduced compared to other variables, such as temperature or wind speed. Hence, some form of calibration is strongly advised, which usually means estimation of the predictive distribution of the weather quantity at hand either by parametric or non-parametric approaches, including also machine learning-based techniques. As visibility observations - according to the suggestion of the World Meteorological Organization - are usually reported in discrete values, the predictive distribution for this particular variable is a discrete probability law, hence calibration can be reduced to a classification problem. Based on visibility ensemble forecasts of the European Centre for Mediu
    
[^41]: 高斯过程回归的贝叶斯方法中融入不确定输入

    Bayesian approach to Gaussian process regression with uncertain inputs. (arXiv:2305.11586v1 [cs.LG])

    [http://arxiv.org/abs/2305.11586](http://arxiv.org/abs/2305.11586)

    本文提出了一种新的高斯过程回归技术，通过贝叶斯方法将输入数据的不确定性纳入回归模型预测中。在数值实验中展示了该方法具有普适性和不错的表现。

    

    传统高斯过程回归仅假设模型观测数据的输出具有噪声。然而，在许多科学和工程应用中，由于建模假设、测量误差等因素，观测数据的输入位置可能也存在不确定性。在本文中，我们提出了一种贝叶斯方法，将输入数据的可变性融入到高斯过程回归中。考虑两种可观测量——具有固定输入的噪声污染输出和具有先验分布定义的不确定输入，通过贝叶斯框架估计后验分布以推断不确定的数据位置。然后，利用边际化方法将这些输入的量化不确定性纳入高斯过程预测中。通过几个数值实验，展示了这种新回归技术的有效性，在其中观察到不同水平输入数据不确定性下的普适良好表现。

    Conventional Gaussian process regression exclusively assumes the existence of noise in the output data of model observations. In many scientific and engineering applications, however, the input locations of observational data may also be compromised with uncertainties owing to modeling assumptions, measurement errors, etc. In this work, we propose a Bayesian method that integrates the variability of input data into Gaussian process regression. Considering two types of observables -- noise-corrupted outputs with fixed inputs and those with prior-distribution-defined uncertain inputs, a posterior distribution is estimated via a Bayesian framework to infer the uncertain data locations. Thereafter, such quantified uncertainties of inputs are incorporated into Gaussian process predictions by means of marginalization. The effectiveness of this new regression technique is demonstrated through several numerical examples, in which a consistently good performance of generalization is observed, w
    
[^42]: 为协变量漂移自适应引入双重加权方法

    Double-Weighting for Covariate Shift Adaptation. (arXiv:2305.08637v1 [stat.ML])

    [http://arxiv.org/abs/2305.08637](http://arxiv.org/abs/2305.08637)

    本文提出了一种双重加权的最小极大风险分类方法，可以有效避免协变量漂移对监督学习的影响。

    

    监督学习中常常受到协变量漂移影响，即训练样本和测试样本的实例边缘分布不同但标签条件相同。现有方法通过使用比率p_te（x）/p_tr（x）对训练样本进行加权（重新加权方法），或者使用比率p_tr（x）/p_te（x）对测试样本进行加权（鲁棒方法）来解决这种协变量漂移。然而，在支持不匹配或上述比率取大值时，这些方法的性能可能很差。我们提出了一种最小极大风险分类(MRC)方法，通过对训练样本和测试样本进行加权来避免这种限制。此外，我们开发了有效的技术来获得两组加权，并推广了传统的核均值匹配方法。我们提供了新的生成模型和实际数据集上的实验结果来证明我们方法的优越性。

    Supervised learning is often affected by a covariate shift in which the marginal distributions of instances (covariates $x$) of training and testing samples $\mathrm{p}_\text{tr}(x)$ and $\mathrm{p}_\text{te}(x)$ are different but the label conditionals coincide. Existing approaches address such covariate shift by either using the ratio $\mathrm{p}_\text{te}(x)/\mathrm{p}_\text{tr}(x)$ to weight training samples (reweighting methods) or using the ratio $\mathrm{p}_\text{tr}(x)/\mathrm{p}_\text{te}(x)$ to weight testing samples (robust methods). However, the performance of such approaches can be poor under support mismatch or when the above ratios take large values. We propose a minimax risk classification (MRC) approach for covariate shift adaptation that avoids such limitations by weighting both training and testing samples. In addition, we develop effective techniques that obtain both sets of weights and generalize the conventional kernel mean matching method. We provide novel genera
    
[^43]: 均值漂移的收敛性分析

    Convergence Analysis of Mean Shift. (arXiv:2305.08463v1 [stat.ML])

    [http://arxiv.org/abs/2305.08463](http://arxiv.org/abs/2305.08463)

    本研究提出了均值漂移算法的模估计序列的收敛保证，并扩展了现有的涵盖解析核和Epanechnikov核的发现，意义在于涵盖了在基于KDE的模估计的渐近统计效率方面最优的非负核——双重核。

    

    均值漂移（MS）算法寻找核密度估计（KDE）的模。本研究提出了一种由MS算法产生的模估计序列的收敛保证，并在相当温和的条件下，借助于关于{\L}ojasiewicz不等式的论证，评估了收敛速度。我们的发现扩展了现有的涵盖解析核和Epanechnikov核的发现，意义在于涵盖了在基于KDE的模估计的渐近统计效率方面最优的非负核——双重核。

    The mean shift (MS) algorithm seeks a mode of the kernel density estimate (KDE). This study presents a convergence guarantee of the mode estimate sequence generated by the MS algorithm and an evaluation of the convergence rate, under fairly mild conditions, with the help of the argument concerning the {\L}ojasiewicz inequality. Our findings, which extend existing ones covering analytic kernels and the Epanechnikov kernel, are significant in that they cover the biweight kernel that is optimal among non-negative kernels in terms of the asymptotic statistical efficiency for the KDE-based mode estimation.
    
[^44]: 训练生成对抗网络的梯度下降-上升算法的局部收敛性研究

    Local Convergence of Gradient Descent-Ascent for Training Generative Adversarial Networks. (arXiv:2305.08277v1 [cs.LG])

    [http://arxiv.org/abs/2305.08277](http://arxiv.org/abs/2305.08277)

    本论文研究了使用基于核函数的鉴别器训练GAN的梯度下降-上升算法的局部收敛性，揭示了学习率、正则化和带宽对其影响，同时展示了收敛、振荡或发散的相变现象。

    

    生成对抗网络（GAN）是一种流行的复杂高维数据生成模型的训练方法。训练GAN的标准方法涉及对极小-极大优化问题进行梯度下降-上升（GDA）过程。由于动态的非线性性质，该过程通常很难分析。本研究重点研究了使用基于核函数的鉴别器训练GAN时的GDA局部动态。该收敛性分析是在[Becker et al. 2022]的“孤立点模型”假设下，对描述GDA迭代的非线性动力学系统进行线性化得到的。我们的分析揭示了学习率、正则化和核判别器的带宽对GDA局部收敛速度的影响。重要的是，我们展示了相变现象，表明系统何时收敛、振荡或发散。我们还提供了验证我们结论的数值模拟。

    Generative Adversarial Networks (GANs) are a popular formulation to train generative models for complex high dimensional data. The standard method for training GANs involves a gradient descent-ascent (GDA) procedure on a minimax optimization problem. This procedure is hard to analyze in general due to the nonlinear nature of the dynamics. We study the local dynamics of GDA for training a GAN with a kernel-based discriminator. This convergence analysis is based on a linearization of a non-linear dynamical system that describes the GDA iterations, under an \textit{isolated points model} assumption from [Becker et al. 2022]. Our analysis brings out the effect of the learning rates, regularization, and the bandwidth of the kernel discriminator, on the local convergence rate of GDA. Importantly, we show phase transitions that indicate when the system converges, oscillates, or diverges. We also provide numerical simulations that verify our claims.
    
[^45]: 滞后多因子模型中领先滞后关系的鲁棒检测

    Robust Detection of Lead-Lag Relationships in Lagged Multi-Factor Models. (arXiv:2305.06704v1 [stat.ML])

    [http://arxiv.org/abs/2305.06704](http://arxiv.org/abs/2305.06704)

    该论文提出了一种基于聚类的鲁棒检测滞后多因子模型中的领先滞后关系方法，并使用各种聚类技术和相似度度量方法实现了对领先滞后估计的聚合，从而强化了对原始宇宙中的一致关系的识别。

    

    在多元时间序列系统中，通过发现数据中固有的领先滞后关系，可以获得关键信息，这指的是两个相对时间互移的时间序列之间的依赖关系，可以用于控制、预测或聚类。我们开发了一种基于聚类的方法，用于鲁棒检测滞后多因子模型中的领先滞后关系。在我们的框架中，所设想的管道接收一组时间序列作为输入，并使用滑动窗口方法从每个输入时间序列中提取一组子序列时间序列。然后，我们应用各种聚类技术（例如K-means++和谱聚类），采用各种成对相似性度量，包括非线性的相似性度量。一旦聚类被提取出来，跨聚类的领先滞后估计被聚合起来，以增强对原始宇宙中一致关系的识别。由于多

    In multivariate time series systems, key insights can be obtained by discovering lead-lag relationships inherent in the data, which refer to the dependence between two time series shifted in time relative to one another, and which can be leveraged for the purposes of control, forecasting or clustering. We develop a clustering-driven methodology for the robust detection of lead-lag relationships in lagged multi-factor models. Within our framework, the envisioned pipeline takes as input a set of time series, and creates an enlarged universe of extracted subsequence time series from each input time series, by using a sliding window approach. We then apply various clustering techniques (e.g, K-means++ and spectral clustering), employing a variety of pairwise similarity measures, including nonlinear ones. Once the clusters have been extracted, lead-lag estimates across clusters are aggregated to enhance the identification of the consistent relationships in the original universe. Since multi
    
[^46]: 多类分类中敌对训练解的存在性研究

    On the existence of solutions to adversarial training in multiclass classification. (arXiv:2305.00075v1 [cs.LG])

    [http://arxiv.org/abs/2305.00075](http://arxiv.org/abs/2305.00075)

    本文研究了多类分类中敌对训练的鲁棒解存在性问题，证明了每个模型中存在 Borel 可测的鲁棒分类器，并与最优传输和总变差正则化建立了联系。在二元分类问题中，对不可知分类器的敌对训练问题存在 Borel 可测的解。

    

    本文研究了敌对训练在多类分类问题中的三种模型，旨在构建对抗扰动下鲁棒的分类器。我们证明了每个模型中存在 Borel 可测的鲁棒分类器，并提供了敌对训练问题的统一视角，拓展了作者之前的最优传输联系，并在多类情况下敌对训练和总变差正则化之间建立了新的联系。作为我们结果的推论，我们证明了在二元分类设置中，对不可知分类器的敌对训练问题存在 Borel 可测的解，这一结果改进了关于敌对训练的文献，文献中仅已知只有在特征空间的扩大通用 $σ$-代数内存在鲁棒的分类器。

    We study three models of the problem of adversarial training in multiclass classification designed to construct robust classifiers against adversarial perturbations of data in the agnostic-classifier setting. We prove the existence of Borel measurable robust classifiers in each model and provide a unified perspective of the adversarial training problem, expanding the connections with optimal transport initiated by the authors in previous work and developing new connections between adversarial training in the multiclass setting and total variation regularization. As a corollary of our results, we prove the existence of Borel measurable solutions to the agnostic adversarial training problem in the binary classification setting, a result that improves results in the literature of adversarial training, where robust classifiers were only known to exist within the enlarged universal $\sigma$-algebra of the feature space.
    
[^47]: 内在上下文算子学习用于微分方程问题

    In-Context Operator Learning for Differential Equation Problems. (arXiv:2304.07993v1 [cs.LG])

    [http://arxiv.org/abs/2304.07993](http://arxiv.org/abs/2304.07993)

    本文提出了一种新的神经网络方法INDEED，它可以同时学习不同微分方程问题的操作符，而无需重新训练，且只需要极少的演示。

    

    本文介绍了一种新的基于神经网络的方法——IN-context Differential Equation Encoder-Decoder（INDEED），用于从数据中同时学习操作符并在推理阶段将其应用于新问题，而无需进行任何权重更新。现有方法局限于使用神经网络来逼近特定的方程解或特定的操作符，需要重新训练来处理具有不同方程的新问题。通过训练单个神经网络作为操作符学习器，我们不仅可以摆脱为新问题重新训练（甚至微调）神经网络的困扰，还可以利用操作符之间共享的共同点，这样在学习新的操作符时只需要极少的演示即可。我们的数值结果显示了神经网络作为少样本学习器的能力，用于各种不同类型的微分方程问题，包括ODE和PDE的正向和反向问题，同时显示它可以推广学习能力。

    This paper introduces a new neural-network-based approach, namely IN-context Differential Equation Encoder-Decoder (INDEED), to simultaneously learn operators from data and apply it to new questions during the inference stage, without any weight update. Existing methods are limited to using a neural network to approximate a specific equation solution or a specific operator, requiring retraining when switching to a new problem with different equations. By training a single neural network as an operator learner, we can not only get rid of retraining (even fine-tuning) the neural network for new problems, but also leverage the commonalities shared across operators so that only a few demos are needed when learning a new operator. Our numerical results show the neural network's capability as a few-shot operator learner for a diversified type of differential equation problems, including forward and inverse problems of ODEs and PDEs, and also show that it can generalize its learning capabilit
    
[^48]: 扩散Schr\"odinger桥匹配

    Diffusion Schr\"odinger Bridge Matching. (arXiv:2303.16852v1 [stat.ML])

    [http://arxiv.org/abs/2303.16852](http://arxiv.org/abs/2303.16852)

    本文介绍了一种新的方法 Iterative Markovian Fitting，用于解决高维度 Schr\"odinger桥（SBs）问题，该方法的数值实验表现出在准确性和性能方面的显著优势。

    

    解决运输问题，在机器学习中有着许多应用，例如新型的质量传输方法，如去噪扩散模型（DDMs）和流匹配模型（FMMs），通过随机微分方程（SDE）或常微分方程（ODE）实现这样的传输。然而，虽然在许多应用中，近似确定性动态最优传输（OT）映射是可取的，因为具有吸引人的性质，但 DDMs 和 FMMs 并不能保证提供接近 OT 映射的传输。相反，Schr\"odinger桥（SBs）计算随机动态映射，可以恢复正则熵版本的 OT。不幸的是，现有的数值方法近似 SBs 的维度缩放差或在迭代中积累误差。在这项工作中，我们介绍了迭代马尔科夫拟合，一种解决高维度 SB 问题的新方法。我们将这个方法设计为一个迭代过程，将置信传播扩展到 KL 散度，利用条件独立性降低计算复杂度，并确保一致性和收敛性质。我们的数值实验证明了相对于现有成果方法，在准确性和性能方面都有显著优势。

    Solving transport problems, i.e. finding a map transporting one given distribution to another, has numerous applications in machine learning. Novel mass transport methods motivated by generative modeling have recently been proposed, e.g. Denoising Diffusion Models (DDMs) and Flow Matching Models (FMMs) implement such a transport through a Stochastic Differential Equation (SDE) or an Ordinary Differential Equation (ODE). However, while it is desirable in many applications to approximate the deterministic dynamic Optimal Transport (OT) map which admits attractive properties, DDMs and FMMs are not guaranteed to provide transports close to the OT map. In contrast, Schr\"odinger bridges (SBs) compute stochastic dynamic mappings which recover entropy-regularized versions of OT. Unfortunately, existing numerical methods approximating SBs either scale poorly with dimension or accumulate errors across iterations. In this work, we introduce Iterative Markovian Fitting, a new methodology for solv
    
[^49]: Scoring Functions 和 Generalization Prediction 的详细研究

    A Closer Look at Scoring Functions and Generalization Prediction. (arXiv:2303.13589v1 [cs.LG])

    [http://arxiv.org/abs/2303.13589](http://arxiv.org/abs/2303.13589)

    本文研究了广义误差预测器的有效性，探讨了置信度、局部流形平滑度和模型一致性评分函数的优缺点，发现在复杂机制缺失的情况下，最先进的评分无法在分布转移和损坏下超越简单的模型一致性。同时，在受损训练数据的情况下，模型一致性打分仍然表现良好，并且集成多样性有助于提高泛化性能。

    

    本文研究了广义误差预测器（GEPs）的效果，这些 GEPs 旨在通过从样本级分数中推导出数据集级误差估计值，从而预测模型在未见分布上的表现。然而，GEPs 常常利用不同的机制（例如，回归器、阈值函数、校准数据集等），来推导这种误差估计值，这会混淆特定评分函数的优点。因此，本文在机制选择独立的情况下，深入研究了流行的评分函数的有效性（置信度、局部流形平滑度、模型一致性）。我们发现，在复杂机制缺失的情况下，当估计分布转移和损坏下的误差时，最先进的置信度和平滑度基础评分无法超越简单的模型一致性。此外，在实际情况下，当训练数据受到损害时（例如标签噪声、测量噪声、欠采样），我们发现模型一致性打分仍然表现良好，并且集成多样性有助于提高泛化性能。

    Generalization error predictors (GEPs) aim to predict model performance on unseen distributions by deriving dataset-level error estimates from sample-level scores. However, GEPs often utilize disparate mechanisms (e.g., regressors, thresholding functions, calibration datasets, etc), to derive such error estimates, which can obfuscate the benefits of a particular scoring function. Therefore, in this work, we rigorously study the effectiveness of popular scoring functions (confidence, local manifold smoothness, model agreement), independent of mechanism choice. We find, absent complex mechanisms, that state-of-the-art confidence- and smoothness- based scores fail to outperform simple model-agreement scores when estimating error under distribution shifts and corruptions. Furthermore, on realistic settings where the training data has been compromised (e.g., label noise, measurement noise, undersampling), we find that model-agreement scores continue to perform well and that ensemble diversi
    
[^50]: 迭代近似交叉验证

    Iterative Approximate Cross-Validation. (arXiv:2303.02732v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2303.02732](http://arxiv.org/abs/2303.02732)

    本文提出了一种新的方法，利用迭代一阶算法高效近似交叉验证，从而解决了大规模问题中因限制计算资源或早停而难以得到ERM问题确切解的问题。

    

    交叉验证(CV)是评估和选择预测模型的最流行工具之一。然而，标准CV在折数较多时计算成本很高。最近，在经验风险最小化(ERM)框架下，一系列工作提出了基于完整数据集训练的ERM问题解的有效方法来近似CV。然而，在大规模问题中，由于有限的计算资源或早停的方式防止过拟合，很难得到ERM问题的确切解。本文提出了一种新的范式，在通过迭代一阶算法求解ERM问题时高效地近似CV，而无需运行到收敛状态。我们的新方法扩展了现有的CV近似保证，使其在整个算法轨迹中（包括收敛时）都成立，从而推广了现有的CV近似方法。最后，我们展示了该方法的准确性。

    Cross-validation (CV) is one of the most popular tools for assessing and selecting predictive models. However, standard CV suffers from high computational cost when the number of folds is large. Recently, under the empirical risk minimization (ERM) framework, a line of works proposed efficient methods to approximate CV based on the solution of the ERM problem trained on the full dataset. However, in large-scale problems, it can be hard to obtain the exact solution of the ERM problem, either due to limited computational resources or due to early stopping as a way of preventing overfitting. In this paper, we propose a new paradigm to efficiently approximate CV when the ERM problem is solved via an iterative first-order algorithm, without running until convergence. Our new method extends existing guarantees for CV approximation to hold along the whole trajectory of the algorithm, including at convergence, thus generalizing existing CV approximation methods. Finally, we illustrate the accu
    
[^51]: 神经网络中的深度退化：全连接ReLU网络初始化时，消失角度的现象 (arXiv:2302.09712v2 [stat.ML] 更新版)

    Depth Degeneracy in Neural Networks: Vanishing Angles in Fully Connected ReLU Networks on Initialization. (arXiv:2302.09712v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.09712](http://arxiv.org/abs/2302.09712)

    本文研究了深度神经网络中的深度退化现象，在全连接ReLU网络初始化时，两个输入之间的角度会趋近于0。通过使用组合展开，得到了其趋向于0的速度的精确公式，并验证了这些结果。

    

    尽管深度神经网络在各种任务上表现出色，但许多其性质仍未被理论上理解，其中一个谜团是深度退化现象：网络层数越深，初始化时网络越接近于常数函数。在本文中，我们研究了ReLU神经网络两个输入之间随着层数变化的角度演变情况。通过使用组合展开，我们找到了它随深度增加趋向于0的速度的精确公式，这些公式捕捉了微观波动。我们用Monte Carlo实验验证了我们的理论结果，并证明了结果准确地近似了有限网络的行为。这些公式以通过ReLU函数的相关高斯变量的混合矩形式给出。我们还发现了一个令人惊讶的组合现象。

    Despite remarkable performance on a variety of tasks, many properties of deep neural networks are not yet theoretically understood. One such mystery is the depth degeneracy phenomenon: the deeper you make your network, the closer your network is to a constant function on initialization. In this paper, we examine the evolution of the angle between two inputs to a ReLU neural network as a function of the number of layers. By using combinatorial expansions, we find precise formulas for how fast this angle goes to zero as depth increases. These formulas capture microscopic fluctuations that are not visible in the popular framework of infinite width limits, and leads to qualitatively different predictions. We validate our theoretical results with Monte Carlo experiments and show that our results accurately approximate finite network behaviour. The formulas are given in terms of the mixed moments of correlated Gaussians passed through the ReLU function. We also find a surprising combinatoria
    
[^52]: 重新审视判别式分类器与生成式分类器：理论与应用

    Revisiting Discriminative vs. Generative Classifiers: Theory and Implications. (arXiv:2302.02334v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02334](http://arxiv.org/abs/2302.02334)

    本文重新审视关于判别式与生成式分类器的经典主题，利用多类$\mathcal{H}$-一致性下界，证明了在温和的假设下，多类朴素贝叶斯分类器的样本要求比逻辑回归分类器多了$O(\log n)$。

    

    大规模深度模型预先在大规模标记或未标记数据上进行训练，可以有效地转移到下游任务。线性评估将预先训练的模型中的参数冻结，并单独训练一个线性分类器，这是一种有效且有吸引力的转移方法。然而，目前很少有研究线性评估中的分类器，除了默认的逻辑回归分类器。本文受到朴素贝叶斯的统计效率启发，重新审视了关于判别式与生成式分类器的经典主题。理论上，本文考虑使用代理损失而不是0-1损失进行分析，并将经典结果从二元情况推广到多类情况。我们表明，在温和的假设下，多类朴素贝叶斯需要$O(\log n)$个样本来接近其渐近误差，而相应的多类逻辑回归需要$O(n)$个样本，其中$n$是特征维度。为了证明这一点，我们提出了一个多类$\mathcal{H}$-一致性下界。

    A large-scale deep model pre-trained on massive labeled or unlabeled data transfers well to downstream tasks. Linear evaluation freezes parameters in the pre-trained model and trains a linear classifier separately, which is efficient and attractive for transfer. However, little work has investigated the classifier in linear evaluation except for the default logistic regression. Inspired by the statistical efficiency of naive Bayes, the paper revisits the classical topic on discriminative vs. generative classifiers. Theoretically, the paper considers the surrogate loss instead of the zero-one loss in analyses and generalizes the classical results from binary cases to multiclass ones. We show that, under mild assumptions, multiclass naive Bayes requires $O(\log n)$ samples to approach its asymptotic error while the corresponding multiclass logistic regression requires $O(n)$ samples, where $n$ is the feature dimension. To establish it, we present a multiclass $\mathcal{H}$-consistency bo
    
[^53]: 超越稳健性的普适定律：随机特征和神经切向核的更尖锐法则

    Beyond the Universal Law of Robustness: Sharper Laws for Random Features and Neural Tangent Kernels. (arXiv:2302.01629v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.01629](http://arxiv.org/abs/2302.01629)

    本文通过研究随机特征和神经切向核（NTK）的经验风险最小化，证明了在随机特征中，即使满足稳健性的通用定律所需的必要条件，模型也不具有任何过度参数化程度的稳健性。相对地，对于偶激活情况，NTK模型满足普遍下限，只要满足过参数条件就能稳健。这为机器学习中的稳健性提供了更尖锐的法则，超越了先前建立的普适定律。

    

    机器学习模型容易受到对抗性干扰，Bubeck和Sellke的一个有思想启示的文章通过过度参数化的视角分析了这一现象：平滑地插值数据需要的参数显著多于简单地记忆数据。然而，“普适”的法则仅为稳健性提供了必要条件，无法区分模型。本文通过专注于随机特征和神经切向核（NTK）的两个典型设置中的经验风险最小化来解决这些差距。我们证明，在随机特征中，即使满足稳健性的通用定律所需的必要条件，模型也不具有任何过度参数化程度的稳健性。相反，对于偶激活情况，NTK模型满足普遍下限，只要满足过参数条件就能稳健。这也解决了先前在NTK架构的最优性上的猜想。我们的结果为机器学习中的稳健性提供了更尖锐的法则，超越了先前建立的普适定律。

    Machine learning models are vulnerable to adversarial perturbations, and a thought-provoking paper by Bubeck and Sellke has analyzed this phenomenon through the lens of over-parameterization: interpolating smoothly the data requires significantly more parameters than simply memorizing it. However, this "universal" law provides only a necessary condition for robustness, and it is unable to discriminate between models. In this paper, we address these gaps by focusing on empirical risk minimization in two prototypical settings, namely, random features and the neural tangent kernel (NTK). We prove that, for random features, the model is not robust for any degree of over-parameterization, even when the necessary condition coming from the universal law of robustness is satisfied. In contrast, for even activations, the NTK model meets the universal lower bound, and it is robust as soon as the necessary condition on over-parameterization is fulfilled. This also addresses a conjecture in prior 
    
[^54]: 上下文套索：通过深度神经网络的方法实现稀疏线性模型

    The contextual lasso: Sparse linear models via deep neural networks. (arXiv:2302.00878v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.00878](http://arxiv.org/abs/2302.00878)

    本论文提出了一种新的统计估计器——上下文套索，可以通过深度神经网络的方法解决解释性和拟合能力的矛盾问题，实现对可解释特征的稀疏拟合，并且稀疏模式和系数会随着上下文特征的变化而发生变化。

    

    稀疏线性模型是可解释机器学习的黄金标准工具，本论文通过使用深度神经网络对稀疏线性模型进行改进，实现了可解释性和强大的拟合能力。上下文套索是一种新的统计估计器，它将输入特征分成可解释特征和上下文特征两组，并对可解释特征进行稀疏拟合，同时其稀疏模式和系数会随着上下文特征的变化而发生变化，这个过程通过深度神经网络无需参数地进行学习。

    Sparse linear models are a gold standard tool for interpretable machine learning, a field of emerging importance as predictive models permeate decision-making in many domains. Unfortunately, sparse linear models are far less flexible as functions of their input features than black-box models like deep neural networks. With this capability gap in mind, we study a not-uncommon situation where the input features dichotomize into two groups: explanatory features, which are candidates for inclusion as variables in an interpretable model, and contextual features, which select from the candidate variables and determine their effects. This dichotomy leads us to the contextual lasso, a new statistical estimator that fits a sparse linear model to the explanatory features such that the sparsity pattern and coefficients vary as a function of the contextual features. The fitting process learns this function nonparametrically via a deep neural network. To attain sparse coefficients, we train the net
    
[^55]: 通过稀疏编码实现无约束动态遗憾

    Unconstrained Dynamic Regret via Sparse Coding. (arXiv:2301.13349v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.13349](http://arxiv.org/abs/2301.13349)

    本文探讨了在线线性优化（OLO）涉及无约束问题和动态遗憾问题的复杂性，提出了一种通过重新构造问题为稀疏编码的复杂度度量方式，在适应性和应用上有较好的应用价值。

    

    受时间序列预测的影响，本研究探讨了在线线性优化（OLO）在两个问题结构的耦合下的情况：域无界，而算法的性能是通过动态遗憾来衡量的。处理任一问题都要求遗憾界限依赖于比较序列的某些复杂度量度 - 特别是无约束OLO中的比较器范数，以及动态遗憾中的路径长度。与最近一篇文章(Jacobsen& Cutkosky，2022)适应这两个复杂度量度相比，我们提出了一种通过重新构造问题为稀疏编码的复杂度度量方式。可以通过一个简单的模块化框架实现适应性，这个框架自然地利用了环境更复杂的前置知识。同时，我们还提出了一种新的静态无约束OLO梯度自适应算法，使用了新颖的连续时间机制设计。这可能是具有独立兴趣的。

    Motivated by time series forecasting, we study Online Linear Optimization (OLO) under the coupling of two problem structures: the domain is unbounded, and the performance of an algorithm is measured by its dynamic regret. Handling either of them requires the regret bound to depend on certain complexity measure of the comparator sequence -- specifically, the comparator norm in unconstrained OLO, and the path length in dynamic regret. In contrast to a recent work (Jacobsen & Cutkosky, 2022) that adapts to the combination of these two complexity measures, we propose an alternative complexity measure by recasting the problem into sparse coding. Adaptivity can be achieved by a simple modular framework, which naturally exploits more intricate prior knowledge of the environment. Along the way, we also present a new gradient adaptive algorithm for static unconstrained OLO, designed using novel continuous time machinery. This could be of independent interest.
    
[^56]: 双Kullback-Leibler最小化的变分稀疏逆Cholesky近似用于潜在高斯过程

    Variational sparse inverse Cholesky approximation for latent Gaussian processes via double Kullback-Leibler minimization. (arXiv:2301.13303v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.13303](http://arxiv.org/abs/2301.13303)

    本文提出了一种基于稀疏逆Cholesky因子的高斯分布的变分逼近方法，结合同样高效的SIC约束的Kullback-Leibler最优先验逼近，并在特定SIC排序和稀疏模式下，实现对潜在高斯过程的高度准确先验和后验逼近。与其他方法相比，该方法可以在类似计算复杂度下更准确地预测平稳核函数。

    

    为了实现可扩展和准确的潜在高斯过程推断，我们提出了一种基于一族具有稀疏逆Cholesky（SIC）因子的高斯分布的变分逼近。我们将该变分逼近的后验与类似的高效SIC约束的Kullback-Leibler最优先验逼近相结合。然后，我们重点研究了特定的SIC排序和基于最近邻的稀疏模式，从而产生了高度准确的先验和后验逼近。对于这种设置，我们的变分逼近可以通过每次迭代的对数多项式时间的随机梯度下降来计算。我们提供了数字比较，表明所提出的双Kullback-Leibler最优高斯过程逼近（DKLGP）有时可以比诸如诱导点和均值场逼近等在类似计算复杂度下更准确地预测平稳核函数。

    To achieve scalable and accurate inference for latent Gaussian processes, we propose a variational approximation based on a family of Gaussian distributions whose covariance matrices have sparse inverse Cholesky (SIC) factors. We combine this variational approximation of the posterior with a similar and efficient SIC-restricted Kullback-Leibler-optimal approximation of the prior. We then focus on a particular SIC ordering and nearest-neighbor-based sparsity pattern resulting in highly accurate prior and posterior approximations. For this setting, our variational approximation can be computed via stochastic gradient descent in polylogarithmic time per iteration. We provide numerical comparisons showing that the proposed double-Kullback-Leibler-optimal Gaussian-process approximation (DKLGP) can sometimes be vastly more accurate for stationary kernels than alternative approaches such as inducing-point and mean-field approximations at similar computational complexity.
    
[^57]: 在高维贝叶斯优化中，随机分解是否足够？

    Are Random Decompositions all we need in High Dimensional Bayesian Optimisation?. (arXiv:2301.12844v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.12844](http://arxiv.org/abs/2301.12844)

    本文研究了数据独立分解采样规则，证明了随机树分解采样器有利的理论保证，促进了随机分解上置信度算法（RDUCB）的发展。

    

    学习昂贵的黑盒函数分解有望将贝叶斯优化扩展到高维问题。然而，这些技术的成功取决于找到准确表示黑盒函数的适当分解。我们研究本文中关于数据独立分解采样规则的方法。我们发现，基于数据学习分解可以很容易地被误导到局部分解上，而这些分解在整个搜索空间中并不准确。然后，我们正式证明了基于随机树的分解采样器展现了有利的理论保证，可以有效权衡最大信息增益和实际黑盒函数及其分解之间的函数失配。这些结果促进了随机分解上置信度算法（RDUCB）的发展，该算法易于实现，几乎是即插即用的。

    Learning decompositions of expensive-to-evaluate black-box functions promises to scale Bayesian optimisation (BO) to high-dimensional problems. However, the success of these techniques depends on finding proper decompositions that accurately represent the black-box. While previous works learn those decompositions based on data, we investigate data-independent decomposition sampling rules in this paper. We find that data-driven learners of decompositions can be easily misled towards local decompositions that do not hold globally across the search space. Then, we formally show that a random tree-based decomposition sampler exhibits favourable theoretical guarantees that effectively trade off maximal information gain and functional mismatch between the actual black-box and its surrogate as provided by the decomposition. Those results motivate the development of the random decomposition upper-confidence bound algorithm (RDUCB) that is straightforward to implement - (almost) plug-and-play -
    
[^58]: 使用多任务深度集合估计因果效应

    Estimating Causal Effects using a Multi-task Deep Ensemble. (arXiv:2301.11351v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.11351](http://arxiv.org/abs/2301.11351)

    通过学习研究群体中的共享和特定于组的信息，使用Causal Multi-task Deep Ensemble（CMDE）的方法可以有效地处理高维和多模态协变量，并提供因果效应的点估计不确定性。

    

    已有许多方法被提出来用于因果效应估计，然而很少有方法可以处理像图像等具有复杂结构的数据。为了填补这一空白，我们提出了Causal Multi-task Deep Ensemble (CMDE)，这是一个新颖的框架，能够学习研究群体中的共享和特定于组的信息。我们提供了证明，证明了CMDE与先验中的多任务高斯过程（GP）具有同等效果。与多任务GP相比，CMDE可以有效地处理高维和多模态协变量，并提供因果效应的点估计不确定性。我们评估了我们的方法，并在各种类型的数据集和任务中发现CMDE在大多数任务上优于现有的最先进方法。

    A number of methods have been proposed for causal effect estimation, yet few have demonstrated efficacy in handling data with complex structures, such as images. To fill this gap, we propose Causal Multi-task Deep Ensemble (CMDE), a novel framework that learns both shared and group-specific information from the study population. We provide proofs demonstrating equivalency of CDME to a multi-task Gaussian process (GP) with a coregionalization kernel a priori. Compared to multi-task GP, CMDE efficiently handles high-dimensional and multi-modal covariates and provides pointwise uncertainty estimates of causal effects. We evaluate our method across various types of datasets and tasks and find that CMDE outperforms state-of-the-art methods on a majority of these tasks.
    
[^59]: 最大最优性边际：上下文线性规划和逆线性规划的统一方法

    Maximum Optimality Margin: A Unified Approach for Contextual Linear Programming and Inverse Linear Programming. (arXiv:2301.11260v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.11260](http://arxiv.org/abs/2301.11260)

    本论文提出了一种名为“最大最优性边际”的新方法来解决预测-优化问题，通过下游优化的最优性条件设计机器学习损失函数，兼具计算效率和较好的理论性质，而且只需要训练数据中最优解的观测值。

    

    本文研究了预测-优化问题，其中机器学习预测任务的输出用作某个下游优化问题（例如线性规划的目标系数向量）的输入。该问题也被称为预测分析或上下文线性规划。现有方法在很大程度上要么受到（i）优化不可解性（非凸目标函数）/统计效率低下（子优一般化界限）的困扰，要么要求强条件（例如没有约束或损失校准）。我们开发了一种名为“最大最优性边际”的新方法，通过下游优化的最优性条件设计机器学习损失函数。最大边际公式既具有计算效率，又具有好的学习程序的理论性质。更重要的是，我们的新方法只需要训练数据中最优解的观测值。

    In this paper, we study the predict-then-optimize problem where the output of a machine learning prediction task is used as the input of some downstream optimization problem, say, the objective coefficient vector of a linear program. The problem is also known as predictive analytics or contextual linear programming. The existing approaches largely suffer from either (i) optimization intractability (a non-convex objective function)/statistical inefficiency (a suboptimal generalization bound) or (ii) requiring strong condition(s) such as no constraint or loss calibration. We develop a new approach to the problem called \textit{maximum optimality margin} which designs the machine learning loss function by the optimality condition of the downstream optimization. The max-margin formulation enjoys both computational efficiency and good theoretical properties for the learning procedure. More importantly, our new approach only needs the observations of the optimal solution in the training data
    
[^60]: 学习Boltzmann密度的变形轨迹

    Learning Deformation Trajectories of Boltzmann Densities. (arXiv:2301.07388v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.07388](http://arxiv.org/abs/2301.07388)

    本文介绍了一种学习Boltzmann密度变形轨迹的方法，其中通过插值能量函数等实现Boltzmann密度的变形，然后找到一个时间依赖向量场，将样本从一个分布转移到另一个分布，其表现在高斯混合和量子力学粒子的Boltzmann密度上比KL-反散度更具优势。

    

    我们提出了一种连续标准化流的训练方法，可以在没有样本但存在能量函数的情况下使用。我们的方法依赖于能量函数$f_1$和广义高斯函数$f_0$之间的预定或学习插值$f_t$。能量函数的插值引起Boltzmann密度$p_t\propto e^{-f_t}$的插值，我们旨在找到一个沿着族$p_t$的时间依赖向量场$V_t$，将样本从一个分布转移到另一个分布。将样本沿着族$p_t$从一个分布转移到另一个分布的条件可以转化为$V_t$和$f_t$之间的PDE，我们优化$V_t$和$f_t$以满足此PDE。我们在高斯混合和双井势的量子力学粒子的Boltzmann密度上实验比较了所提出的训练目标与KL-反散度的差异。

    We introduce a training objective for continuous normalizing flows that can be used in the absence of samples but in the presence of an energy function. Our method relies on either a prescribed or a learnt interpolation $f_t$ of energy functions between the target energy $f_1$ and the energy function of a generalized Gaussian $f_0(x) = ||x/\sigma||_p^p$. The interpolation of energy functions induces an interpolation of Boltzmann densities $p_t \propto e^{-f_t}$ and we aim to find a time-dependent vector field $V_t$ that transports samples along the family $p_t$ of densities. The condition of transporting samples along the family $p_t$ can be translated to a PDE between $V_t$ and $f_t$ and we optimize $V_t$ and $f_t$ to satisfy this PDE. We experimentally compare the proposed training objective to the reverse KL-divergence on Gaussian mixtures and on the Boltzmann density of a quantum mechanical particle in a double-well potential.
    
[^61]: 有限结果空间上的相对概率：其公理化、属性和应用的系统考察

    Relative Probability on Finite Outcome Spaces: A Systematic Examination of its Axiomatization, Properties, and Applications. (arXiv:2212.14555v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2212.14555](http://arxiv.org/abs/2212.14555)

    本文提出了将概率看作相对度量的观点，建立了有限结果空间上相对概率函数的公理化，提供了其实例和组合系统，并讨论了相对贝叶斯推断及其数字实现，证明了相对概率空间的拓扑闭包，突显了其在极限下保留信息的能力。

    

    本文提出了将概率看作相对度量而非绝对度量的观点。为了证明这一概念，我们将焦点放在有限结果空间上，并建立了三个基本公理，以确立相对概率函数的要求。我们提供了一组这些函数的实例库和一个组合系统。此外，我们讨论了相对贝叶斯推断及其数字实现。最后，我们证明了相对概率空间的拓扑闭包，突显了其在极限下保留信息的能力。

    This work proposes a view of probability as a relative measure rather than an absolute one. To demonstrate this concept, we focus on finite outcome spaces and develop three fundamental axioms that establish requirements for relative probability functions. We then provide a library of examples of these functions and a system for composing them. Additionally, we discuss a relative version of Bayesian inference and its digital implementation. Finally, we prove the topological closure of the relative probability space, highlighting its ability to preserve information under limits.
    
[^62]: 双重平滑GDA：用于非凸-非凹极小极大优化的全局收敛算法

    Doubly Smoothed GDA: Global Convergent Algorithm for Constrained Nonconvex-Nonconcave Minimax Optimization. (arXiv:2212.12978v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2212.12978](http://arxiv.org/abs/2212.12978)

    本文提出了一种双重平滑梯度下降上升法 (DSGDA)，该算法可以应用于非凸-非凹极小极大优化，并且能够全局收敛并消除极限环。在一定条件下，DSGDA 的迭代复杂度达到了文献中单循环算法的最佳结果。

    

    非凸-非凹极小极大优化近年来受到了广泛的关注，其在机器学习中具有广泛的应用。然而，大多数现有算法不能保证全局收敛，甚至会遭受极限环的困扰。为了解决这个问题，我们提出了一种新颖的单循环算法，称为双重平滑梯度下降上升法 (DSGDA)，它能够自然地平衡原始与对偶更新，并且将极其具有挑战性的非凸-非凹例子中的极限环消除，包括 Forsaken，Bilinearly-coupled minimax，Sixth-order polynomial 和 PolarGame。我们进一步证明，在一个单侧的 $\theta\in(0,1)$ Kurdyka-\L{}ojasiewicz条件（或凸原始/凹对偶函数）下，DSGDA 可以找到一个游戏平衡点，并且具有迭代复杂度 $\mathcal{O}(\epsilon^{-2\max\{2\theta,1\}})$（或 $\mathcal{O}(\epsilon^{-4})$），这些与文献中单循环算法的最佳结果相匹配。

    Nonconvex-nonconcave minimax optimization has received intense attention over the last decade due to its broad applications in machine learning. Unfortunately, most existing algorithms cannot be guaranteed to converge globally and even suffer from limit cycles. To address this issue, we propose a novel single-loop algorithm called doubly smoothed gradient descent ascent method (DSGDA), which naturally balances the primal and dual updates. The proposed DSGDA can get rid of limit cycles in various challenging nonconvex-nonconcave examples in the literature, including Forsaken, Bilinearly-coupled minimax, Sixth-order polynomial, and PolarGame. We further show that under an one-sided Kurdyka-\L{}ojasiewicz condition with exponent $\theta\in(0,1)$ (resp. convex primal/concave dual function), DSGDA can find a game-stationary point with an iteration complexity of $\mathcal{O}(\epsilon^{-2\max\{2\theta,1\}})$ (resp. $\mathcal{O}(\epsilon^{-4})$). These match the best results for single-loop al
    
[^63]: 理解对准确性不平衡影响的对抗鲁棒性问题

    Understanding the Impact of Adversarial Robustness on Accuracy Disparity. (arXiv:2211.15762v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.15762](http://arxiv.org/abs/2211.15762)

    本文通过研究高斯混合模型下的线性分类器，分析了对抗鲁棒性对准确性不平衡的影响，并证明了在稳定分布的一般家族中也存在类似影响。

    

    尽管长期以来已经从经验上观察到对抗鲁棒性可能与标准准确性存在一些矛盾，并且可能对不同类别产生不平等影响，但它仍然是一个未解决的问题，即这些观察有多大程度的保持，以及类别不平衡在其中扮演什么样的角色。在本文中，我们试图通过更深入地研究高斯混合模型下的线性分类器来理解这个准确性不平衡问题。我们将对抗鲁棒性的影响分解成两部分：一部分是因为鲁棒性约束而会降低所有类别的标准准确性而固有的影响，另一部分是由于类别不平衡比率引起的，这将增加与标准训练相比的准确性差异。此外，我们还表明这些影响超越了高斯混合模型，通过将数据模型推广到稳定分布的一般家族。具体而言，我们证明了，虽然对抗鲁棒性的约束一致会减少所有类别的标准准确性，但通常会增加对少数类别的准确性不平衡。

    While it has long been empirically observed that adversarial robustness may be at odds with standard accuracy and may have further disparate impacts on different classes, it remains an open question to what extent such observations hold and how the class imbalance plays a role within. In this paper, we attempt to understand this question of accuracy disparity by taking a closer look at linear classifiers under a Gaussian mixture model. We decompose the impact of adversarial robustness into two parts: an inherent effect that will degrade the standard accuracy on all classes due to the robustness constraint, and the other caused by the class imbalance ratio, which will increase the accuracy disparity compared to standard training. Furthermore, we also show that such effects extend beyond the Gaussian mixture model, by generalizing our data model to the general family of stable distributions. More specifically, we demonstrate that while the constraint of adversarial robustness consistentl
    
[^64]: 一类非凸非凹极小极大问题的零阶交替梯度下降算法研究

    Zeroth-Order Alternating Gradient Descent Ascent Algorithms for a Class of Nonconvex-Nonconcave Minimax Problems. (arXiv:2211.13668v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2211.13668](http://arxiv.org/abs/2211.13668)

    本文提出了零阶交替梯度下降算法和零阶方差减少交替梯度下降算法，用于解决一类非凸非凹的极小极大问题，分别在确定性和随机环境下。它们是解决这类问题的第一和第二个迭代复杂度保证的零阶算法。

    

    本文考虑一类非凸非凹的极小极大问题，即NC-PL极小极大问题，其目标函数针对内部变量满足Polyak-Lôjasiewicz（PL）条件。我们提出了零阶交替梯度下降上升（ZO-AGDA）算法和零阶方差减少交替梯度下降上升（ZO-VRAGDA）算法，分别用于确定性和随机环境下解决NC-PL极小极大问题。使用ZO-AGDA和ZO-VRAGDA算法得到NC-PL极小极大问题的ε-稳定点所需的总函数值查询次数上界分别为O(ε^(-2))和O(ε^(-3))。据我们所知，它们是解决NC-PL极小极大问题的第一和第二个迭代复杂度保证的零阶算法。

    In this paper, we consider a class of nonconvex-nonconcave minimax problems, i.e., NC-PL minimax problems, whose objective functions satisfy the Polyak-\L ojasiewicz (PL) condition with respect to the inner variable. We propose a zeroth-order alternating gradient descent ascent (ZO-AGDA) algorithm and a zeroth-order variance reduced alternating gradient descent ascent (ZO-VRAGDA) algorithm for solving NC-PL minimax problem under the deterministic and the stochastic setting, respectively. The total number of function value queries to obtain an $\epsilon$-stationary point of ZO-AGDA and ZO-VRAGDA algorithm for solving NC-PL minimax problem is upper bounded by $\mathcal{O}(\varepsilon^{-2})$ and $\mathcal{O}(\varepsilon^{-3})$, respectively. To the best of our knowledge, they are the first two zeroth-order algorithms with the iteration complexity gurantee for solving NC-PL minimax problems.
    
[^65]: 在线非随机控制简介

    Introduction to Online Nonstochastic Control. (arXiv:2211.09619v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.09619](http://arxiv.org/abs/2211.09619)

    介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    

    本文介绍了一种新兴的动态系统控制与可微强化学习范式——在线非随机控制，并应用在线凸优化和凸松弛技术得到了具有可证明保证的新方法，在最佳和鲁棒控制方面取得了显著成果。与其他框架不同，该方法的目标是对抗性攻击，在无法预测扰动模型的情况下，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    This text presents an introduction to an emerging paradigm in control of dynamical systems and differentiable reinforcement learning called online nonstochastic control. The new approach applies techniques from online convex optimization and convex relaxations to obtain new methods with provable guarantees for classical settings in optimal and robust control.  The primary distinction between online nonstochastic control and other frameworks is the objective. In optimal control, robust control, and other control methodologies that assume stochastic noise, the goal is to perform comparably to an offline optimal strategy. In online nonstochastic control, both the cost functions as well as the perturbations from the assumed dynamical model are chosen by an adversary. Thus the optimal policy is not defined a priori. Rather, the target is to attain low regret against the best policy in hindsight from a benchmark class of policies.  This objective suggests the use of the decision making frame
    
[^66]: 具有保证的PAC-Bayesian离线情境强化学习算法

    PAC-Bayesian Offline Contextual Bandits With Guarantees. (arXiv:2210.13132v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.13132](http://arxiv.org/abs/2210.13132)

    本文提出了一种通过PAC-Bayesian方法分析离线情境强化学习问题的新算法，该算法通过优化新的泛化界限提供了保证，并在实际情境中得到了验证。

    

    本文提出了一种新的基于PAC-Bayesian方法的离线情境强化学习算法。与之前的方法不同，该方法不是从难以处理或不准确的界限推导学习原则。我们通过PAC-Bayesian方法分析问题，将策略解释为决策规则的混合物。这使我们能够提出新的泛化界限，并提供可解算法来优化它们。我们证明所得界限比竞争对手更紧，可以直接优化以在离线情况下自信地改进记录策略。我们的方法学习带保证的策略，使用所有可用数据，并不需要在保留集上调整更多的超参数。通过广泛的实验，我们展示了该方法在实际情景中提供性能保证的有效性。

    This paper introduces a new principled approach for off-policy learning in contextual bandits. Unlike previous work, our approach does not derive learning principles from intractable or loose bounds. We analyse the problem through the PAC-Bayesian lens, interpreting policies as mixtures of decision rules. This allows us to propose novel generalization bounds and provide tractable algorithms to optimize them. We prove that the derived bounds are tighter than their competitors, and can be optimized directly to confidently improve upon the logging policy offline. Our approach learns policies with guarantees, uses all available data and does not require tuning additional hyperparameters on held-out sets. We demonstrate through extensive experiments the effectiveness of our approach in providing performance guarantees in practical scenarios.
    
[^67]: 基于拉普拉斯增强的低秩张量样条克里金方法的稀疏传感大规模交通速度估计

    Correlating sparse sensing for large-scale traffic speed estimation: A Laplacian-enhanced low-rank tensor kriging approach. (arXiv:2210.11780v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.11780](http://arxiv.org/abs/2210.11780)

    本文提出了一种基于拉普拉斯增强的低秩张量样条克里金方法，用于在有限观测下进行大规模交通速度估计，以从不完整的数据中恢复可信的估计值。

    

    交通速度是表征道路网络流动性的核心因素，许多交通应用程序都依赖于它，如实时导航、动态路线规划和拥堵管理。传感和通信技术的快速进展使交通速度检测比以往任何时候都更加容易。然而，由于静态传感器的稀疏部署或移动传感器的低渗透，检测到的速度是不完整的，并且远离全网使用。此外，由于各种原因传感器容易出现误差或缺失数据，这些传感器检测到的速度会变得非常嘈杂。因此，需要有效的技术从不完整的数据中恢复可信的估计值。在本研究中，我们首先将问题确定为一个时空克里金问题，并提出了一种拉普拉斯增强的低秩张量完成（LETC）框架，其具有低秩性和多维相关性，用于在有限观测下进行大规模交通速度克里金。

    Traffic speed is central to characterizing the fluidity of the road network. Many transportation applications rely on it, such as real-time navigation, dynamic route planning, and congestion management. Rapid advances in sensing and communication techniques make traffic speed detection easier than ever. However, due to sparse deployment of static sensors or low penetration of mobile sensors, speeds detected are incomplete and far from network-wide use. In addition, sensors are prone to error or missing data due to various kinds of reasons, speeds from these sensors can become highly noisy. These drawbacks call for effective techniques to recover credible estimates from the incomplete data. In this work, we first identify the issue as a spatiotemporal kriging problem and propose a Laplacian enhanced low-rank tensor completion (LETC) framework featuring both lowrankness and multi-dimensional correlations for large-scale traffic speed kriging under limited observations. To be specific, th
    
[^68]: 基于符合p值的预测选择方法

    Selection by Prediction with Conformal p-values. (arXiv:2210.01408v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2210.01408](http://arxiv.org/abs/2210.01408)

    本论文提出一种基于符合p值的预测选择方法，使用统计证据的p值来控制虚阳性选择单位，可用于初步筛选职业招聘和药物发现的候选人集合。

    

    涉及到职业招聘和药物发现等决策和科学发现流程通常涉及多个阶段：在任何资源密集型步骤之前，通常会进行初始筛选，使用机器学习模型的预测来从大量候选人中筛选出少数人。我们研究旨在选择未观察结果超过用户指定值的候选人的筛选程序。我们开发了一种围绕任何预测模型的方法，以产生一个候选人集合，同时控制虚阳性选择单位的比例。我们的方法建立在符合推断框架之上，首先构建量化大型结果的统计证据的p值；然后通过将p值与多重检验文献中引入的阈值进行比较，确定短名单。在许多情况下，该过程选择的候选人的预测高于基于数据的阈值。我们的理论保证在温和的交换条件下成立。

    Decision making or scientific discovery pipelines such as job hiring and drug discovery often involve multiple stages: before any resource-intensive step, there is often an initial screening that uses predictions from a machine learning model to shortlist a few candidates from a large pool. We study screening procedures that aim to select candidates whose unobserved outcomes exceed user-specified values. We develop a method that wraps around any prediction model to produce a subset of candidates while controlling the proportion of falsely selected units. Building upon the conformal inference framework, our method first constructs p-values that quantify the statistical evidence for large outcomes; it then determines the shortlist by comparing the p-values to a threshold introduced in the multiple testing literature. In many cases, the procedure selects candidates whose predictions are above a data-dependent threshold. Our theoretical guarantee holds under mild exchangeability conditions
    
[^69]: 非负张量的多体逼近

    Many-body Approximation for Non-negative Tensors. (arXiv:2209.15338v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2209.15338](http://arxiv.org/abs/2209.15338)

    提出了一种名为多体逼近的方法来分解非负张量，通过能量建模来避免全局优化和目标秩选择的困难，可通过考虑模式之间的交互进行全局优化; 在许多任务中都展示了其有效性。

    

    我们提出了一种替代方法来分解非负张量，称为多体逼近。传统的分解方法假设表示具有低秩性，导致全局优化和目标秩选择的困难。我们通过张量的能量建模避免了这些问题，其中张量和其模式分别对应于概率分布和随机变量。我们的模型可以通过考虑模式之间的交互来进行全局优化，可以比秩更直观地进行调整。此外，我们将模式之间的相互作用可视化为张量网络，揭示了多体逼近和低秩逼近之间的非平凡关系。我们在张量完成和逼近中展示了我们方法的有效性。

    We present an alternative approach to decompose non-negative tensors, called many-body approximation. Traditional decomposition methods assume low-rankness in the representation, resulting in difficulties in global optimization and target rank selection. We avoid these problems by energy-based modeling of tensors, where a tensor and its mode correspond to a probability distribution and a random variable, respectively. Our model can be globally optimized in terms of the KL divergence minimization by taking the interaction between variables, i.e. modes, into account that can be tuned more intuitively than ranks. Furthermore, we visualize interactions between modes as tensor networks and reveal a nontrivial relationship between many-body approximation and low-rank approximation. We demonstrate the effectiveness of our approach in tensor completion and approximation.
    
[^70]: 基于似然函数修正的半定规划方法用于异质数据聚类

    Likelihood Adjusted Semidefinite Programs for Clustering Heterogeneous Data. (arXiv:2209.15097v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2209.15097](http://arxiv.org/abs/2209.15097)

    本论文提出了一种基于似然函数修正的半定规划方法用于异质数据聚类。经过实验表明，本方法在处理聚类形状不同的数据异质性时表现优异。

    

    聚类是一种广泛使用的无监督学习工具。基于模型的聚类是一种灵活的框架，用来处理聚类具有不同形状的数据的异质性。对于混合分布的基于似然的推断通常涉及非凸和高维的目标函数，带来了复杂的计算和统计挑战。在本文中，我们将基于似然函数修正的半定规划（LA-SDP）方法应用于异质数据聚类。我们的方法通过一组新的矩阵不等式实现了似然函数调整的凸松弛。我们证明，在混合组分的一些温和的前提条件下，LA-SDP 可以一致而有效地计算出最大似然估计值。我们的实验表明，与现有的方法相比，尤其是当聚类显著异质时，我们的方法在合成数据和真实数据的实验中表现优异。

    Clustering is a widely deployed unsupervised learning tool. Model-based clustering is a flexible framework to tackle data heterogeneity when the clusters have different shapes. Likelihood-based inference for mixture distributions often involves non-convex and high-dimensional objective functions, imposing difficult computational and statistical challenges. The classic expectation-maximization (EM) algorithm is a computationally thrifty iterative method that maximizes a surrogate function minorizing the log-likelihood of observed data in each iteration, which however suffers from bad local maxima even in the special case of the standard Gaussian mixture model with common isotropic covariance matrices. On the other hand, recent studies reveal that the unique global solution of a semidefinite programming (SDP) relaxed $K$-means achieves the information-theoretically sharp threshold for perfectly recovering the cluster labels under the standard Gaussian mixture model. In this paper, we ext
    
[^71]: 模型训练中鲁棒性高的高维线性学习方法

    Robust Methods for High-Dimensional Linear Learning. (arXiv:2208.05447v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2208.05447](http://arxiv.org/abs/2208.05447)

    该论文提出了一种在高维批量训练中具有高度鲁棒性和计算效率的线性学习方法，在多个应用程序中均能达到接近最优的估计速率，并提供了一个开源的Python库进行实现。

    

    我们提出了一种高维批处理中具有统计鲁棒性和计算有效性的线性学习方法，其中特征数d可能超过样本数n。我们在通用学习设置中采用了两种算法，取决于所考虑的损失函数是否是梯度Lipschitz的。然后，我们将我们的框架实例化到几个应用程序上，包括香草稀疏，组稀疏和低秩矩阵恢复。这导致了每个应用程序的高效和鲁棒的学习算法，在重尾分布和异常值的情况下，达到接近最优的估计速率。对于香草$s$-稀疏，我们能够在重尾和$\eta$-污染下达到$s\log(d)/n$的速率，计算成本与非鲁棒模拟相当。我们提供了一个开源的$\mathtt{Python}$库$\mathtt{linlearn}$来实现我们的算法，通过这个库进行数值实验，证明了我们方法的有效性和可扩展性。

    We propose statistically robust and computationally efficient linear learning methods in the high-dimensional batch setting, where the number of features $d$ may exceed the sample size $n$. We employ, in a generic learning setting, two algorithms depending on whether the considered loss function is gradient-Lipschitz or not. Then, we instantiate our framework on several applications including vanilla sparse, group-sparse and low-rank matrix recovery. This leads, for each application, to efficient and robust learning algorithms, that reach near-optimal estimation rates under heavy-tailed distributions and the presence of outliers. For vanilla $s$-sparsity, we are able to reach the $s\log (d)/n$ rate under heavy-tails and $\eta$-corruption, at a computational cost comparable to that of non-robust analogs. We provide an efficient implementation of our algorithms in an open-source $\mathtt{Python}$ library called $\mathtt{linlearn}$, by means of which we carry out numerical experiments whi
    
[^72]: 基于流模型的去噪在流形假设下的收敛性分析

    Convergence of denoising diffusion models under the manifold hypothesis. (arXiv:2208.05314v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2208.05314](http://arxiv.org/abs/2208.05314)

    本文提供了基于流模型的去噪在流形假设下的收敛性分析，首次拓展到了目标分布受流形约束或通过经验分布给出的情况。

    

    去噪流模型是一类生成模型，在图像和音频合成方面表现出最先进的性能。这样的模型近似于从目标分布到参考密度（通常为高斯分布）的正向噪声过程的时间反演。尽管它们具有强大的实证结果，但对这些模型的理论分析仍然有限。特别地，所有当前的方法都关键地假设目标分布相对于勒贝格测度存在密度。这不涵盖目标分布受低维流形约束或通过某些经验分布给出的情况。我们提供了第一个针对流模型在这种更加普遍的情况下的收敛性结果并提供一阶Wasserstein距离量化界限。

    Denoising diffusion models are a recent class of generative models exhibiting state-of-the-art performance in image and audio synthesis. Such models approximate the time-reversal of a forward noising process from a target distribution to a reference density, which is usually Gaussian. Despite their strong empirical results, the theoretical analysis of such models remains limited. In particular, all current approaches crucially assume that the target density admits a density w.r.t. the Lebesgue measure. This does not cover settings where the target distribution is supported on a lower-dimensional manifold or is given by some empirical distribution. In this paper, we bridge this gap by providing the first convergence results for diffusion models in this more general setting. In particular, we provide quantitative bounds on the Wasserstein distance of order one between the target data distribution and the generative distribution of the diffusion model.
    
[^73]: 带约束条件的差分隐私联合组合赌博机研究

    Differentially Private Federated Combinatorial Bandits with Constraints. (arXiv:2206.13192v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.13192](http://arxiv.org/abs/2206.13192)

    本文研究了差分隐私联合组合赌博机问题，探讨了代理在共同学习时如何保持数据的隐私，并提出了在后悔和隐私之间实现平衡的重要性。

    

    在在线学习模式中，合作学习范式（即联邦学习）快速增长。与大多数联邦学习情景不同的是，有很多情况下代理是竞争的。每个代理都想从其他人那里学习，但它分享给其他人学习的信息有可能是敏感的，因此它需要隐私。本文研究了一组代理同时解决类似的组合赌博机问题，同时保持质量约束。这些代理是否可以通过采用差分隐私来保持机密性，集体学习？我们观察到通信可以降低后悔。但是，保护敏感信息的差分隐私技术使数据变得很嘈杂，可能会恶化而不是有帮助地提高后悔。因此，我们指出决定何时通信以及学习哪些共享数据来在后悔和隐私之间实现功能平衡至关重要。

    There is a rapid increase in the cooperative learning paradigm in online learning settings, i.e., federated learning (FL). Unlike most FL settings, there are many situations where the agents are competitive. Each agent would like to learn from others, but the part of the information it shares for others to learn from could be sensitive; thus, it desires its privacy. This work investigates a group of agents working concurrently to solve similar combinatorial bandit problems while maintaining quality constraints. Can these agents collectively learn while keeping their sensitive information confidential by employing differential privacy? We observe that communicating can reduce the regret. However, differential privacy techniques for protecting sensitive information makes the data noisy and may deteriorate than help to improve regret. Hence, we note that it is essential to decide when to communicate and what shared data to learn to strike a functional balance between regret and privacy. F
    
[^74]: 针对部分客户端参与的联邦学习的锚定抽样方法

    Anchor Sampling for Federated Learning with Partial Client Participation. (arXiv:2206.05891v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.05891](http://arxiv.org/abs/2206.05891)

    提出一种针对部分客户端参与的联邦学习框架 FedAMD，其中核心思想是锚定抽样，将参与者分为锚定组和矿工组，以解决数据异构性。

    

    相较于全客户端参与，部分客户端参与是联邦学习中更常见的场景，但是会加重一些挑战，例如数据异构性。在部分客户端参与的情况下缺少非活动客户端的更新，可能会导致模型聚合偏离基于全客户端参与的聚合。通常提出采用在个体客户端上使用大批量来进行训练以解决数据异构性，但其在部分客户端参与的情况下的有效性不明确。在考虑这些挑战的基础上，我们提出了一种新的针对部分客户端参与的联邦学习框架，称为FedAMD，其核心思想是锚定抽样，将部分参与者分为锚定组和矿工组。

    Compared with full client participation, partial client participation is a more practical scenario in federated learning, but it may amplify some challenges in federated learning, such as data heterogeneity. The lack of inactive clients' updates in partial client participation makes it more likely for the model aggregation to deviate from the aggregation based on full client participation. Training with large batches on individual clients is proposed to address data heterogeneity in general, but their effectiveness under partial client participation is not clear. Motivated by these challenges, we propose to develop a novel federated learning framework, referred to as FedAMD, for partial client participation. The core idea is anchor sampling, which separates partial participants into anchor and miner groups. Each client in the anchor group aims at the local bullseye with the gradient computation using a large batch. Guided by the bullseyes, clients in the miner group steer multiple near
    
[^75]: 从单一正标签中挖掘多标签样本

    Mining Multi-Label Samples from Single Positive Labels. (arXiv:2206.05764v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.05764](http://arxiv.org/abs/2206.05764)

    本文提出了一种基于单一正标签的采样方法S2M，以实现对多标签样本的生成，从而避免了多标签数据集制作时高昂的注释成本。

    

    条件生成对抗网络（cGAN）在类别有条件的生成任务中已经证明具有出色的结果。为了同时控制多个条件，cGAN需要多标签训练数据集，其中可以为每个数据实例分配多个标签。然而，巨大的注释成本限制了多标签数据集在现实场景中的可访问性。因此，在本研究中，我们探索了实际设置的称为单正标签设置，其中每个数据实例仅由一个正标签注释，没有明确的负标签。为了在单正标签设置中生成多标签数据，我们提出了一种基于马尔可夫链蒙特卡罗方法的新型采样方法，称为单到多标签（S2M）采样。作为广泛适用的“附加”方法，我们提出的S2M采样方法使现有的无条件和条件GAN能够以最小的注释成本绘制高质量的多标签数据。在实际图像和文本数据集上进行的大量实验证明了我们提出的方法在各种情况下的有效性。

    Conditional generative adversarial networks (cGANs) have shown superior results in class-conditional generation tasks. To simultaneously control multiple conditions, cGANs require multi-label training datasets, where multiple labels can be assigned to each data instance. Nevertheless, the tremendous annotation cost limits the accessibility of multi-label datasets in real-world scenarios. Therefore, in this study we explore the practical setting called the single positive setting, where each data instance is annotated by only one positive label with no explicit negative labels. To generate multi-label data in the single positive setting, we propose a novel sampling approach called single-to-multi-label (S2M) sampling, based on the Markov chain Monte Carlo method. As a widely applicable "add-on" method, our proposed S2M sampling method enables existing unconditional and conditional GANs to draw high-quality multi-label data with a minimal annotation cost. Extensive experiments on real im
    
[^76]: 由 SGD 训练的深度神经网络的泛化误差界

    Generalization Error Bounds for Deep Neural Networks Trained by SGD. (arXiv:2206.03299v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.03299](http://arxiv.org/abs/2206.03299)

    通过对于适当参数规范的动态控制结合基于参数规范的 Rademacher 复杂度估计导出了深度神经网络的泛化误差界，适用于包括 MLP 和 CNN 在内的广泛网络架构，结果表明这个方法能够适应优化器和网络超参数的变化。

    

    本文通过将适当参数规范的动态控制和基于参数规范的 Rademacher 复杂度估计相结合，导出了由随机梯度下降（SGD）训练的深度神经网络的泛化误差界。这些界明确取决于沿训练轨迹的损失，并适用于包括多层感知机（MLP）和卷积神经网络（CNN）在内的广泛网络架构。与其他算法依赖的泛化估计（如基于全局稳定性的界）相比，我们的界不需要非凸损失函数的 $L$-平滑性，并且直接适用于 SGD，而不是随机 Langevin 梯度下降（SGLD）。数值结果表明，我们的界是非虚假和强健的，能够适应优化器和网络超参数的变化。

    Generalization error bounds for deep neural networks trained by stochastic gradient descent (SGD) are derived by combining a dynamical control of an appropriate parameter norm and the Rademacher complexity estimate based on parameter norms. The bounds explicitly depend on the loss along the training trajectory, and work for a wide range of network architectures including multilayer perceptron (MLP) and convolutional neural networks (CNN). Compared with other algorithm-depending generalization estimates such as uniform stability-based bounds, our bounds do not require $L$-smoothness of the nonconvex loss function, and apply directly to SGD instead of Stochastic Langevin gradient descent (SGLD). Numerical results show that our bounds are non-vacuous and robust with the change of optimizer and network hyperparameters.
    
[^77]: 使用多环境方法检测观测数据中的隐式混淆

    Detecting hidden confounding in observational data using multiple environments. (arXiv:2205.13935v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2205.13935](http://arxiv.org/abs/2205.13935)

    使用独立数据生成过程下的多环境方法，可以检测观测数据中的未观察到的混淆因素，并提出了测试独立性的程序。

    

    在因果推断中，常见的假设是没有隐式混淆。然而，在单个数据集中不能确定这个假设通常是不可能的。在独立的数据生成过程下，我们展示了一种方法来在多个来自不同环境的观测数据集中检测未观察到的混淆因素。我们提出了一种测试可验证的条件独立性的理论，这种独立性仅当存在混淆因素时才不存在，并检查了违反其假设的情况：退化和依赖机制以及忠实度违反。此外，我们提出了一种程序来测试这些独立性，并使用基于真实世界数据的半合成数据和模拟研究研究其经验有限样本行为。在大多数情况下，提出的程序能够正确预测存在隐式混淆，特别是当混淆偏差很大时。

    A common assumption in causal inference from observational data is that there is no hidden confounding. Yet it is, in general, impossible to verify this assumption from a single dataset. Under the assumption of independent causal mechanisms underlying the data-generating process, we demonstrate a way to detect unobserved confounders when having multiple observational datasets coming from different environments. We present a theory for testable conditional independencies that are only absent when there is hidden confounding and examine cases where we violate its assumptions: degenerate & dependent mechanisms, and faithfulness violations. Additionally, we propose a procedure to test these independencies and study its empirical finite-sample behavior using simulation studies and semi-synthetic data based on a real-world dataset. In most cases, the proposed procedure correctly predicts the presence of hidden confounding, particularly when the confounding bias is large.
    
[^78]: 有限维下的压缩经验测度

    Compressed Empirical Measures (in finite dimensions). (arXiv:2204.08847v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2204.08847](http://arxiv.org/abs/2204.08847)

    本论文探讨了在有限维再生核希尔伯特空间中压缩经验测度的方法，导出了关于这样一个近似的核心集必须有的大小的高概率下限，并开发了一些技术以将压缩方法应用于具体的推断问题。

    

    我们研究了有限维再生核希尔伯特空间（RKHSs）中压缩经验测度的方法。在这种情况下，经验测度包含在一个自然的凸集中，并且可以使用凸优化方法来近似。在某些条件下，这种近似会导致数据点的coreset。控制这样一个coreset必须有多大的一个关键数量是包含在经验凸集中的经验测量周围的最大球的大小。我们的大部分工作是在各种条件下导出关于这样一个球的大小的高概率下限。我们通过开发技术，使得我们能够将压缩方法应用于具体的推断问题，如核岭回归，来补充这种下限的派生。我们最后介绍了一种无限维RKHS的构造，其中压缩很差，突出了我们面临的一些困难。

    We study approaches for compressing the empirical measure in the context of finite dimensional reproducing kernel Hilbert spaces (RKHSs).In this context, the empirical measure is contained within a natural convex set and can be approximated using convex optimization methods.Such an approximation gives under certain conditions rise to a coreset of data points. A key quantity that controls how large such a coreset has to be is the size of the largest ball around the empirical measure that is contained within the empirical convex set. The bulk of our work is concerned with deriving high probability lower bounds on the size of such a ball under various conditions. We complement this derivation of the lower bound by developing techniques that allow us to apply the compression approach to concrete inference problems such as kernel ridge regression. We conclude with a construction of an infinite dimensional RKHS for which the compression is poor, highlighting some of the difficulties one face
    
[^79]: 在测地度量空间中的Sion极小极大定理和黎曼外推算法

    Sion's Minimax Theorem in Geodesic Metric Spaces and a Riemannian Extragradient Algorithm. (arXiv:2202.06950v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2202.06950](http://arxiv.org/abs/2202.06950)

    该论文提出了在测地度量空间中的Sion极小极大定理和黎曼外推算法，在保持问题可处理的同时，为非凸-非凹极小极大问题提供了一个广泛的推广。

    

    判断非凸-非凹问题是否存在鞍点通常是难以处理的。该论文向理解一类保持可处理的非凸-非凹极小极大问题迈出了一步。具体而言，它研究了测地度量空间上的极小极大问题，这提供了通常的凸-凹鞍点问题的广泛推广。论文的第一个主要结果是Sion极小极大定理的测地度量空间版本; 我们认为我们的证明是新颖且广泛可用的，因为它仅基于有限交叉性质。第二个主要结果是针对完整测地黎曼流形的专业化：在这里，我们设计和分析了光滑极小极大问题的一阶方法的复杂性。

    Deciding whether saddle points exist or are approximable for nonconvex-nonconcave problems is usually intractable. This paper takes a step towards understanding a broad class of nonconvex-nonconcave minimax problems that do remain tractable. Specifically, it studies minimax problems over geodesic metric spaces, which provide a vast generalization of the usual convex-concave saddle point problems. The first main result of the paper is a geodesic metric space version of Sion's minimax theorem; we believe our proof is novel and broadly accessible as it relies on the finite intersection property alone. The second main result is a specialization to geodesically complete Riemannian manifolds: here, we devise and analyze the complexity of first-order methods for smooth minimax problems.
    
[^80]: 带有核的复合适合性检验方法

    Composite Goodness-of-fit Tests with Kernels. (arXiv:2111.10275v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2111.10275](http://arxiv.org/abs/2111.10275)

    本文提出了一种基于核的假设检验方法，可以解决具有挑战性的复合检验问题，其核心思想是在正确的模型规范的零假设下，非参数地估计参数（或模拟器）分布。

    

    模型错误说明可能会对概率模型的实现造成重大挑战，这促使开发出一些直接解决此问题的鲁棒方法。但是，这些更为复杂的方法是否需要取决于模型是否真的错误，目前缺乏通用的方法回答这个问题。在本文中，我们提出了一种方法。更具体地说，我们提出了基于核的假设检验方法，用于具有挑战性的复合检验问题，即我们是否感兴趣的数据来自某些参数模型族中的任何分布。我们的测试利用基于最大均值差异和核Stein差异的最小距离估计器。它们具有广泛的适用性，包括当参数模型的密度已知除标准化常数外，或者如果模型采用模拟器形式。作为我们的主要结果，我们展示了在正确的模型规范的零假设下，我们能够非参数地估计参数（或模拟器）分布。我们提供了建立我们方法有效性的理论，并通过模拟和异常检测应用案例演示了其性能。

    Model misspecification can create significant challenges for the implementation of probabilistic models, and this has led to development of a range of robust methods which directly account for this issue. However, whether these more involved methods are required will depend on whether the model is really misspecified, and there is a lack of generally applicable methods to answer this question. In this paper, we propose one such method. More precisely, we propose kernel-based hypothesis tests for the challenging composite testing problem, where we are interested in whether the data comes from any distribution in some parametric family. Our tests make use of minimum distance estimators based on the maximum mean discrepancy and the kernel Stein discrepancy. They are widely applicable, including whenever the density of the parametric model is known up to normalisation constant, or if the model takes the form of a simulator. As our main result, we show that we are able to estimate the param
    
[^81]: MMD聚合双样本检验

    MMD Aggregated Two-Sample Test. (arXiv:2110.15073v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2110.15073](http://arxiv.org/abs/2110.15073)

    本文提出了两种新颖的基于最大均值差异（MMD）的非参数双样本核检验，并构造了一种自适应平均测试，称为MMDAgg，以解决平滑参数未知的问题。

    

    我们提出了两种新颖的基于最大均值差异（MMD）的非参数双样本核检验。首先，对于固定的核，我们使用排列或野蛮自举（wild bootstrap）构造了一个MMD检验，这两种流行的数值程序可确定测试阈值。我们证明这个测试可以在非渐近情况下控制I型错误的概率。因此，即使在小样本情况下，它仍然保持良好的校准性，这与以前的MMD测试不同，前者只能在渐近意义下保证正确的测试水平。当密度差异在Sobolev球中时，我们证明了我们的MMD检验在特定的核函数下是最优的，该核函数依赖于Sobolev球的平滑参数。在实践中，这个参数是未知的，因此不能使用具有特定核的最优MMD检验。为了解决这个问题，我们构造了一个自适应平均测试，称为MMDAgg。测试功率在Sobolev球的平滑参数上最大化。

    We propose two novel nonparametric two-sample kernel tests based on the Maximum Mean Discrepancy (MMD). First, for a fixed kernel, we construct an MMD test using either permutations or a wild bootstrap, two popular numerical procedures to determine the test threshold. We prove that this test controls the probability of type I error non-asymptotically. Hence, it can be used reliably even in settings with small sample sizes as it remains well-calibrated, which differs from previous MMD tests which only guarantee correct test level asymptotically. When the difference in densities lies in a Sobolev ball, we prove minimax optimality of our MMD test with a specific kernel depending on the smoothness parameter of the Sobolev ball. In practice, this parameter is unknown and, hence, the optimal MMD test with this particular kernel cannot be used. To overcome this issue, we construct an aggregated test, called MMDAgg, which is adaptive to the smoothness parameter. The test power is maximised ove
    
[^82]: Sinkhorn分布鲁棒优化

    Sinkhorn Distributionally Robust Optimization. (arXiv:2109.11926v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2109.11926](http://arxiv.org/abs/2109.11926)

    本文通过使用Sinkhorn距离进行分布鲁棒优化，推导出更容易处理且在实际中更合理的最坏情况分布，提出了解决方案，并展示了其优越性能。

    

    我们研究了使用Sinkhorn距离 -一种基于熵正则化的Wasserstein距离变体- 的分布鲁棒优化（DRO）。我们为一般名义分布推导了凸规划对偶重构。相比于Wasserstein DRO，对于更大类的损失函数，它在计算上更容易处理，它的最坏情况分布对实际应用更合理。为了解决对偶重构，我们开发了一种使用有偏梯度神经元的随机镜像下降算法，并分析了其收敛速度。最后，我们提供了使用合成和真实数据的数值实例，以证明其优越性能。

    We study distributionally robust optimization (DRO) with Sinkhorn distance -a variant of Wasserstein distance based on entropic regularization. We derive convex programming dual reformulation for a general nominal distribution. Compared with Wasserstein DRO, it is computationally tractable for a larger class of loss functions, and its worst-case distribution is more reasonable for practical applications. To solve the dual reformulation, we develop a stochastic mirror descent algorithm using biased gradient oracles and analyze its convergence rate. Finally, we provide numerical examples using synthetic and real data to demonstrate its superior performance.
    
[^83]: 带有定常大学习率的SGD可能收敛于局部最大值

    SGD with a Constant Large Learning Rate Can Converge to Local Maxima. (arXiv:2107.11774v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2107.11774](http://arxiv.org/abs/2107.11774)

    本研究构建了最坏情况下的优化问题，证明了带有定常大学习率的SGD可能表现出许多奇怪且潜在的不良行为，包括：收敛于局部最大值、缓慢越过鞍点和更喜欢尖锐的最小值。这强调了深入分析SGD在深度学习中作用的重要性。

    

    先前关于随机梯度下降（SGD）的研究通常着眼于其成功，本研究构建了最坏情况下的优化问题，证明了在先前研究通常假设不成立的情况下，SGD可能表现出许多奇怪且潜在的不良行为。具体来说，我们构建了景观和数据分布，使得（1）SGD收敛于局部最大值，（2）SGD缓慢越过鞍点，(3) SGD更喜欢尖锐的最小值而非平坦的最小值，(4) AMSGrad收敛于局部最大值。我们还通过极简的神经网络示例进行了实现。我们的研究强调了同时分析小批量采样、离散时间更新规则和现实景观以了解SGD在深度学习中的作用的重要性。

    Previous works on stochastic gradient descent (SGD) often focus on its success. In this work, we construct worst-case optimization problems illustrating that, when not in the regimes that the previous works often assume, SGD can exhibit many strange and potentially undesirable behaviors. Specifically, we construct landscapes and data distributions such that (1) SGD converges to local maxima, (2) SGD escapes saddle points arbitrarily slowly, (3) SGD prefers sharp minima over flat ones, and (4) AMSGrad converges to local maxima. We also realize results in a minimal neural network-like example. Our results highlight the importance of simultaneously analyzing the minibatch sampling, discrete-time updates rules, and realistic landscapes to understand the role of SGD in deep learning.
    
[^84]: 方差相关的最优臂识别

    Variance-Dependent Best Arm Identification. (arXiv:2106.10417v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2106.10417](http://arxiv.org/abs/2106.10417)

    本文研究了在多臂老虎机游戏中识别最优臂的问题。提出了一种自适应算法，该算法探索臂的奖励差距和方差，并使用一种称为分组中位数淘汰的新方法根据收集的信息做出未来决策。所提出的算法保证以概率(1-δ)输出最优臂，并使用最多的O（Σ(i=1)^n (σi²/Δi²+1/Δi)(lnδ-1+ln lnΔi-1)）个样本，这比方差独立算法获得了明显的优势。

    

    本文研究了在随机多臂老虎机游戏中识别最优臂的问题。给定一个从1到n标号的臂的集合，每个臂i都与一个支持[0,1]上平均值为θi和方差为σi²的未知奖励分布相关联。假设θ1>θ2≥...≥θn。我们提出了一种自适应算法，该算法探索臂的奖励差距和方差，并使用一种称为分组中位数淘汰的新方法根据收集的信息做出未来决策。所提出的算法保证以概率(1-δ)输出最优臂，并使用最多的O（Σ(i=1)^n (σi²/Δi²+1/Δi)(lnδ-1+ln lnΔi-1)）个样本，其中 Δi (i≥2)表示臂i与最优臂之间的奖励差距，我们定义 Δ1 = Δ2。这比方差独立算法获得了明显的优势。

    We study the problem of identifying the best arm in a stochastic multi-armed bandit game. Given a set of $n$ arms indexed from $1$ to $n$, each arm $i$ is associated with an unknown reward distribution supported on $[0,1]$ with mean $\theta_i$ and variance $\sigma_i^2$. Assume $\theta_1 > \theta_2 \geq \cdots \geq\theta_n$. We propose an adaptive algorithm which explores the gaps and variances of the rewards of the arms and makes future decisions based on the gathered information using a novel approach called \textit{grouped median elimination}. The proposed algorithm guarantees to output the best arm with probability $(1-\delta)$ and uses at most $O \left(\sum_{i = 1}^n \left(\frac{\sigma_i^2}{\Delta_i^2} + \frac{1}{\Delta_i}\right)(\ln \delta^{-1} + \ln \ln \Delta_i^{-1})\right)$ samples, where $\Delta_i$ ($i \geq 2$) denotes the reward gap between arm $i$ and the best arm and we define $\Delta_1 = \Delta_2$. This achieves a significant advantage over the variance-independent algorit
    
[^85]: Good-Turing在马尔可夫采样中的应用研究

    How good is Good-Turing for Markov samples?. (arXiv:2102.01938v3 [cs.IT] UPDATED)

    [http://arxiv.org/abs/2102.01938](http://arxiv.org/abs/2102.01938)

    研究了在具有状态分布$[\pi_x:x \in \mathcal{X}]$和转移概率矩阵（t.p.m.）$P$的字母表$\mathcal{X}$上，Markov样本的缺失稳态质量（即缺失符号的总稳态概率）的GT估计器的收敛性。

    

    Good-Turing（GT）估计器用于估计$n$个样本中缺失的质量（即缺失符号的总概率）是出现一次的符号数量除以$n$。本文研究了在具有状态分布$[\pi_x:x \in \mathcal{X}]$和转移概率矩阵（t.p.m.）$P$的字母表$\mathcal{X}$上，Markov样本的缺失稳态质量（即缺失符号的总稳态概率）的GT估计器的收敛性。我们展示GT的收敛性取决于$(P^{\sim x})^n$的收敛性，其中$P^{\sim x}$是在$P$的第$x$列置零后的矩阵。这个问题对于具有时间依赖性的应用程序非常重要和有趣，比如将概率赋给单词序列的语言模型，这些模型被建模为Markov模型。

    The Good-Turing (GT) estimator for the missing mass (i.e., total probability of missing symbols) in $n$ samples is the number of symbols that appeared exactly once divided by $n$. For i.i.d. samples, the bias and squared-error risk of the GT estimator can be shown to fall as $1/n$ by bounding the expected error uniformly over all symbols. In this work, we study convergence of the GT estimator for missing stationary mass (i.e., total stationary probability of missing symbols) of Markov samples on an alphabet $\mathcal{X}$ with stationary distribution $[\pi_x:x \in \mathcal{X}]$ and transition probability matrix (t.p.m.) $P$. This is an important and interesting problem because GT is widely used in applications with temporal dependencies such as language models assigning probabilities to word sequences, which are modelled as Markov. We show that convergence of GT depends on convergence of $(P^{\sim x})^n$, where $P^{\sim x}$ is $P$ with the $x$-th column zeroed out. This, in turn, depend
    
[^86]: 在线学习的现代介绍

    A Modern Introduction to Online Learning. (arXiv:1912.13213v6 [cs.LG] UPDATED)

    [http://arxiv.org/abs/1912.13213](http://arxiv.org/abs/1912.13213)

    这本专著介绍了在线学习的基本概念以及凸优化背景下的一阶和二阶算法, 包括欧几里得和非欧几里得设置中的在线镜像下降或遵循正则化领导者等算法。

    

    在这本专著中，我通过现代的在线凸优化视角介绍了在线学习的基本概念。在这里，在线学习指的是在最坏情况下的遗憾最小化框架。我介绍了一阶和二阶具有凸损失的在线学习算法，以欧几里得和非欧几里得设置为基础，所有算法都清晰地呈现为在线镜像下降或遵循正则化领导者及其变形的实例。特别关注算法参数的调整问题和通过自适应和无参数在线学习算法在无界域中的学习。 凸损失通过凸代理损失和随机化处理，来处理非凸损失。同时还简要讨论了赌博设置，涉及敌对和随机多臂赌博问题。这些笔记不需要先前对凸分析的了解，并且所有所需的数学工具都已严谨解释。

    In this monograph, I introduce the basic concepts of Online Learning through a modern view of Online Convex Optimization. Here, online learning refers to the framework of regret minimization under worst-case assumptions. I present first-order and second-order algorithms for online learning with convex losses, in Euclidean and non-Euclidean settings. All the algorithms are clearly presented as instantiation of Online Mirror Descent or Follow-The-Regularized-Leader and their variants. Particular attention is given to the issue of tuning the parameters of the algorithms and learning in unbounded domains, through adaptive and parameter-free online learning algorithms. Non-convex losses are dealt through convex surrogate losses and through randomization. The bandit setting is also briefly discussed, touching on the problem of adversarial and stochastic multi-armed bandits. These notes do not require prior knowledge of convex analysis and all the required mathematical tools are rigorously ex
    
[^87]: Dirichlet多项式的快速MLE计算

    Fast MLE Computation for the Dirichlet Multinomial. (arXiv:1405.0099v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/1405.0099](http://arxiv.org/abs/1405.0099)

    本文提出了一种修改的方法来快速计算Dirichlet分布的MLE参数，相较于现有实现方法，只需要一遍遍历数据集就可以大大减少运行时间。

    

    给定一个分类数据集，我们希望找到一个Dirichlet分布的参数，使得在该分布下，这个数据集的似然函数最大化。通常利用牛顿迭代法来求解，但目前的实现需要每次迭代都读取整个数据集。在本文中，我们提出了一种修改方法，只需要一次通过数据集，并大大减少了运行时间。此外，我们还从理论和实证的角度分析了该算法的性能，并提供了开源实现。

    Given a collection of categorical data, we want to find the parameters of a Dirichlet distribution which maximizes the likelihood of that data. Newton's method is typically used for this purpose but current implementations require reading through the entire dataset on each iteration. In this paper, we propose a modification which requires only a single pass through the dataset and substantially decreases running time. Furthermore we analyze both theoretically and empirically the performance of the proposed algorithm, and provide an open source implementation.
    

