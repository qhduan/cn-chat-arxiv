# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistical Indistinguishability of Learning Algorithms.](http://arxiv.org/abs/2305.14311) | 本文研究了学习算法的统计不可区分性。他们通过学习规则的结果相似性来检验两个输出分布是否相似。本文发现 TV 不可区分性与现有算法稳定性概念等价，并提供了相关算法。 |
| [^2] | [Evaluation of the MACE Force Field Architecture: from Medicinal Chemistry to Materials Science.](http://arxiv.org/abs/2305.14247) | MACE的机器学习力场架构在内域、外推和低数据范围任务中表现优秀，在处理非晶碳、小分子有机化学、大分子和液态水等领域时常常优于其他替代方案。即使只有50个随机选定的参考配置，该模型也能非常高效地复现实验分子振动光谱。 |
| [^3] | [ZeroSCROLLS: A Zero-Shot Benchmark for Long Text Understanding.](http://arxiv.org/abs/2305.14196) | ZeroSCROLLS是一个用于长文本自然语言理解的零Shot基准测试，包括六个任务和四个数据集，能够评估大型语言模型的性能。当前，GPT-4的平均得分最高，但在聚合任务等多个挑战上，仍有改进的空间。 |
| [^4] | [Improved Convergence of Score-Based Diffusion Models via Prediction-Correction.](http://arxiv.org/abs/2305.14164) | 本文通过使用预测校正方案，提高了基于得分扩散模型的收敛性。 |
| [^5] | [Goodness of fit by Neyman-Pearson testing.](http://arxiv.org/abs/2305.14137) | 本研究介绍了Neyman-Pearson检验在拟合优度研究中的应用，实现了一种名为NPLM的实用实现。和基于分类器方法相比，在探测到数据与期望分布的小偏差时，NPLM更灵敏且不会偏向任何类型的异常，比较适用于对撞机实验中对于新物理的不可知搜索。 future work 需要研究它在其他环境中的使用。 |
| [^6] | [Transferring Learning Trajectories of Neural Networks.](http://arxiv.org/abs/2305.14122) | 本研究提出了转移学习轨迹的算法，可将之前训练过的神经网络的学习轨迹应用在新的训练中，并能在任何直接训练之前实现非平凡的准确性。 |
| [^7] | [Cost-aware learning of relevant contextual variables within Bayesian optimization.](http://arxiv.org/abs/2305.14120) | 本文提出一种基于代价感知的模型选择BO方法SADCBO，通过对后验代理模型的敏感性分析来学习关于环境的相关情境信息，并通过平均模型预测来最小化优化代价，在实验中表现出卓越的性能。 |
| [^8] | [Sustainable Edge Intelligence Through Energy-Aware Early Exiting.](http://arxiv.org/abs/2305.14094) | 本文提出了能量自适应动态早期退出机制，通过能量感知的策略，在EH边缘设备中实现了高效准确推理。 |
| [^9] | [Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension.](http://arxiv.org/abs/2305.14077) | 这篇论文研究了固定维度下内核和神经网络的良性过拟合，发现良性过拟合的关键在于估计器的平滑度而不是维数，并证明在固定维度下中度导数的良性过拟合是不可能的。相反，我们证明了用序列核进行回归是可能出现良性过拟合的。 |
| [^10] | [Towards Understanding the Dynamics of Gaussian--Stein Variational Gradient Descent.](http://arxiv.org/abs/2305.14076) | 本文探究了高斯-斯坦变分梯度下降动态性。对于从高斯目标中采样，只要初始值是高斯的，具有双线性核的SVGD动态将保持高斯状态。当目标函数呈现出强对数凹性时，证明了均场高斯-SVGD动态会线性收敛于KL散度下最接近目标高斯分布。在有限粒子设置中，存在对均场极限的时间微步一致收敛以及线性收敛至目标高斯分布。 |
| [^11] | [DIVA: A Dirichlet Process Based Incremental Deep Clustering Algorithm via Variational Auto-Encoder.](http://arxiv.org/abs/2305.14067) | 本文提出了DIVA算法，一个基于狄利克雷过程的增量深度聚类框架，利用无限混合高斯作为先验，并利用一种记忆化的在线变分推理方法实现簇的动态适应移动，而不需要先知道特征的数量。该算法表现优越，特别是在增量特征的情况下。 |
| [^12] | [Expressive Losses for Verified Robustness via Convex Combinations.](http://arxiv.org/abs/2305.13991) | 通过基于凸组合的表达性损失，可以提高网络的对抗鲁棒性，最新的算法可以获得最先进的结果；这种方法通过对抗性攻击和IBP边界之间的简单凸组合进行实现。 |
| [^13] | [Data-Dependent Bounds for Online Portfolio Selection Without Lipschitzness and Smoothness.](http://arxiv.org/abs/2305.13946) | 本文提出了在线投资组合选择的第一个数据相关上界，算法显示亚线性遗憾率，并在数据“容易”时实现对数遗憾。 |
| [^14] | [Subsampling Error in Stochastic Gradient Langevin Diffusions.](http://arxiv.org/abs/2305.13882) | 该研究分析了随机梯度 Langevin 动力学在大型数据环境下使用子采样产生的误差。研究者提出了一种新的连续时间马尔可夫过程，该过程切换数据子集并可用于扩散子采样 MCMC 方法，并证明了该方法的收敛性。 |
| [^15] | [Stochastic PDE representation of random fields for large-scale Gaussian process regression and statistical finite element analysis.](http://arxiv.org/abs/2305.13879) | 本文针对工程学和机器学习中的贝叶斯建模，使用随机PDE表示来开发一种可扩展的框架，从而可以在几何复杂的域上进行大规模的统计有限元分析和高斯过程回归。 |
| [^16] | [On the Optimal Batch Size for Byzantine-Robust Distributed Learning.](http://arxiv.org/abs/2305.13856) | 本文研究的问题是在拜占庭容错分布式学习中，当梯度计算总数固定时，最佳的批处理大小随拜占庭工人的比例增加而增加。 |
| [^17] | [Covariate balancing using the integral probability metric for causal inference.](http://arxiv.org/abs/2305.13715) | 本文介绍了一种利用积分概率测量进行协变量平衡的因果推断方法，无需正确规定倾向得分或结果回归模型即可保证估计器的一致性，并且在实验中表现出优异性能。 |
| [^18] | [Deep Learning with Kernels through RKHM and the Perron-Frobenius Operator.](http://arxiv.org/abs/2305.13588) | 该论文提出了一种基于核方法的深度学习框架：深度RKHM，通过使用$C^*$代数获得更温和的界限，并提供了良性过拟合的理论解释。 |
| [^19] | [Squared Neural Families: A New Class of Tractable Density Models.](http://arxiv.org/abs/2305.13552) | 提出一种新的可计算密度模型类——平方神经分布族，其通过对神经网络的2范数进行平方和基于某个基础度量进行归一化，严格推广了经典指数族，具有闭性条件推断和可计算的边际分布。 |
| [^20] | [Statistical Guarantees of Group-Invariant GANs.](http://arxiv.org/abs/2305.13517) | 本研究提出了群不变GAN的统计保证，发现当学习群不变分布时，群不变GAN所需样本数会按群体大小的幂比例减少。 |
| [^21] | [Parameter estimation from an Ornstein-Uhlenbeck process with measurement noise.](http://arxiv.org/abs/2305.13498) | 本文研究了带有测量噪声的Ornstein-Uhlenbeck过程参数估计，提出了算法和方法能够分离热噪声和乘性噪声，并改善数据分析的参数估计精度。 |
| [^22] | [Nonparanormal Graph Quilting with Applications to Calcium Imaging.](http://arxiv.org/abs/2305.13491) | 本文研究了钙成像中的图拼接问题，提出了两种非高斯图形模型的解决方案，并在模拟和真实数据上验证了其有效性。 |
| [^23] | [A comprehensive theoretical framework for the optimization of neural networks classification performance with respect to weighted metrics.](http://arxiv.org/abs/2305.13472) | 本论文提出了一个理论框架，可以驱使模型优化加权分类度量标准，包括成本敏感学习、加权交叉熵损失函数和值加权技能得分等已确立的方法。 |
| [^24] | [Error-Tolerant Exact Query Learning of Finite Set Partitions with Same-Cluster Oracle.](http://arxiv.org/abs/2305.13402) | 本文提出了一个新问题：如何通过同簇预言机在存在有限对抗错误时积极学习完全恢复划分。我们建立了解析框架并证明了最坏情况下查询复杂度的上下界，并研究了适应性和查询复杂度之间的关系。 |
| [^25] | [From Random Search to Bandit Learning in Metric Measure Spaces.](http://arxiv.org/abs/2305.11509) | 本文介绍了随机搜索及其性能，引入了“散射维度”的概念，描述了底层函数的状态，量化了随机搜索的性能，并证明了在无噪声和有界噪声情况下的输出分别以一定概率收敛到最优值。 |
| [^26] | [Transfer Causal Learning: Causal Effect Estimation with Knowledge Transfer.](http://arxiv.org/abs/2305.09126) | 本文提出了一个名为$\ell_1$-TCL的通用框架，它使用知识迁移和Lasso回归来提高因果效应估计精度。 |
| [^27] | [First- and Second-Order Bounds for Adversarial Linear Contextual Bandits.](http://arxiv.org/abs/2305.00832) | 本文研究了允许$k$个臂的损失函数随时间而自由变化的对抗性线性上下文赌博情境。在假设环境较为温和的情况下，我们获得了一个关于Learner's Losses $V_T$的二阶损失值量级为$\tilde O(K\sqrt{d V_T})$和关于最佳策略$L_T^*$的一阶损失值量级为$\tilde O(K\sqrt{d L_T^*})$的界。 |
| [^28] | [A mean-field games laboratory for generative modeling.](http://arxiv.org/abs/2304.13534) | 本文提出了使用均场博弈作为实验室对生成模型进行设计和分析的方法，并建立了这种方法与主要流动和扩散型生成模型之间的关联。通过研究每个生成模型与它们相关的 MFG 的最优条件，本文提出了一个基于双人 MFG 的新的生成模型，该模型在提高样本多样性和逼真度的同时改善了解缠结和公平性。 |
| [^29] | [Did we personalize? Assessing personalization by an online reinforcement learning algorithm using resampling.](http://arxiv.org/abs/2304.05365) | 本论文提出了一种使用重复采样的政策评估方法，以评估在线 RL 算法实现的个性化程度。该方法可用于优化数字健康的个性化干预。 |
| [^30] | [Private Statistical Estimation of Many Quantiles.](http://arxiv.org/abs/2302.06943) | 本文主要研究如何在差分隐私条件下估计一个分布的多个分位数。它提出了两种方法：一种是通过私有地估计样本的经验分位数来估计分布的分位数，另一种是使用密度估计技术进行分位数函数估计，并且展示了两种方法之间的权衡。 |
| [^31] | [OPORP: One Permutation + One Random Projection.](http://arxiv.org/abs/2302.03505) | OPORP使用一种"计数草图"类型的数据降维/压缩方法，可以用于嵌入式检索，在保证较少的信息损失的前提下，显著降低了计算和存储的成本 |
| [^32] | [SE(3) diffusion model with application to protein backbone generation.](http://arxiv.org/abs/2302.02277) | 本文提出了SE（3）扩散模型及其理论基础，并使用FrameDiff框架在多个框架上学习SE（3）等变分数，成功生成可设计的长达500个氨基酸的单体背景。 |
| [^33] | [Sample Complexity of Probability Divergences under Group Symmetry.](http://arxiv.org/abs/2302.01915) | 本文研究了具有群不变性的分布变量在变分差异估计中的样本复杂度，发现在群大小维度相关的情况下，样本复杂度会有所降低，并在实验中得到了验证。 |
| [^34] | [Kernel Stein Discrepancy thinning: a theoretical perspective of pathologies and a practical fix with regularization.](http://arxiv.org/abs/2301.13528) | 本文针对后处理MCMC输出过程中的病态问题进行了理论分析，提出了正则化的Stein稀释算法，有效缓解了这些病态产生的影响，为提高逼近质量提供了新策略。 |
| [^35] | [Asymptotic Inference for Multi-Stage Stationary Treatment Policy with High Dimensional Features.](http://arxiv.org/abs/2301.12553) | 本研究填补了在高维特征变量存在的情况下，对于多阶段静态治疗策略本身进行推断的工作空白，提出了一种增强的估计器以提高价值函数的准确性。 |
| [^36] | [Sampling-based Nystr\"om Approximation and Kernel Quadrature.](http://arxiv.org/abs/2301.09517) | 本文提出了一种基于抽样的Nyström逼近方法用于核积分。同时，引入了一种非i.i.d.地标点的理论保证方法，使得提高了逼近的精度。 |
| [^37] | [Birth-death dynamics for sampling: Global convergence, approximations and their asymptotics.](http://arxiv.org/abs/2211.00450) | 本文研究了一种连续的出生死亡动态，并提出了弱假设。通过这种动态控制的概率密度指数级地快速收敛到吉布斯平衡测度，同时提出了一种实用的基于纯出生死亡动态的数值采样器，并对其逼近品质进行了定量评估。 |
| [^38] | [Online Convex Optimization with Unbounded Memory.](http://arxiv.org/abs/2210.09903) | 本论文提出了一种新的在线凸优化框架，可以处理决策历史的长期依赖关系，并介绍了用于量化依赖程度的$p$-有效内存容量的概念。 |
| [^39] | [Granger Causal Chain Discovery for Sepsis-Associated Derangements via Continuous-Time Hawkes Processes.](http://arxiv.org/abs/2209.04480) | 本文提出了一个基于连续时间霍克过程的Granger因果链发现方法，可用于推断EMR数据中多个患者特征之间的时间交互作用，并确定败血症相关异常的实验室值链。 |
| [^40] | [Faster federated optimization under second-order similarity.](http://arxiv.org/abs/2209.02257) | 提出两种新的联邦学习算法，SVRP 和 Catalyzed SVRP，它们都有较高的通信效率和性能表现，并广泛适用于分布式统计学习和差分隐私经验风险最小化等领域。 |
| [^41] | [Masked Bayesian Neural Networks : Computation and Optimality.](http://arxiv.org/abs/2206.00853) | 本文提出了一种新颖的稀疏贝叶斯神经网络（BNN），它可以使用掩码变量在节点级别上关闭一些节点，以产生稀疏的DNN结构。我们还设计了一个先验分布，使得后验分布具有理论上的最优性，并开发了一种高效的MCMC算法。该方法在几个基准数据集上表现良好，能够发现精简的DNN结构，具有与大型DNN相似的预测准确性和不确定性量化能力。 |
| [^42] | [A Case of Exponential Convergence Rates for SVM.](http://arxiv.org/abs/2205.10055) | 本文研究了SVM的指数级收敛速度，提出了一种简单的方法来获得快速收敛速度，并在没有假设硬Tsybakov边际条件的情况下展示了SVM的指数级收敛速度现象。 |
| [^43] | [Externally Valid Policy Choice.](http://arxiv.org/abs/2205.05561) | 本文研究了外部有效的个性化治疗策略的学习问题。研究表明，最大化福利的治疗策略对于实验和目标人群之间的结果分布变化具有鲁棒性。通过开发新的方法，作者提出了对结果和特征变化具有鲁棒性的策略学习方法，并强调实验人群内的治疗效果异质性对策略普适性的影响。 |

# 详细

[^1]: 学习算法的统计不可区分性

    Statistical Indistinguishability of Learning Algorithms. (arXiv:2305.14311v1 [cs.LG])

    [http://arxiv.org/abs/2305.14311](http://arxiv.org/abs/2305.14311)

    本文研究了学习算法的统计不可区分性。他们通过学习规则的结果相似性来检验两个输出分布是否相似。本文发现 TV 不可区分性与现有算法稳定性概念等价，并提供了相关算法。

    

    当两个不同的用户在他们自己的数据上使用相同的学习规则时，我们如何测试两个结果的分布是否相似？本文通过分布的总变差距离来研究学习规则结果的相似性。我们称学习规则在期望总变差距离上是小的，若其输出的后验分布在两个独立于相同分布的训练数据集上执行时。我们首先研究使用 TV 不可区分学习器的假设类可学习性。我们的主要结果是 TV 不可区分性和现有算法稳定性概念之间的信息论等价性，如可复制性和近似差分隐私。然后，我们提供了 TV 不可区分学习器的统计扩大和提升算法。

    When two different parties use the same learning rule on their own data, how can we test whether the distributions of the two outcomes are similar? In this paper, we study the similarity of outcomes of learning rules through the lens of the Total Variation (TV) distance of distributions. We say that a learning rule is TV indistinguishable if the expected TV distance between the posterior distributions of its outputs, executed on two training data sets drawn independently from the same distribution, is small. We first investigate the learnability of hypothesis classes using TV indistinguishable learners. Our main results are information-theoretic equivalences between TV indistinguishability and existing algorithmic stability notions such as replicability and approximate differential privacy. Then, we provide statistical amplification and boosting algorithms for TV indistinguishable learners.
    
[^2]: MACE力场架构的评估：从药物化学到材料科学

    Evaluation of the MACE Force Field Architecture: from Medicinal Chemistry to Materials Science. (arXiv:2305.14247v1 [physics.chem-ph])

    [http://arxiv.org/abs/2305.14247](http://arxiv.org/abs/2305.14247)

    MACE的机器学习力场架构在内域、外推和低数据范围任务中表现优秀，在处理非晶碳、小分子有机化学、大分子和液态水等领域时常常优于其他替代方案。即使只有50个随机选定的参考配置，该模型也能非常高效地复现实验分子振动光谱。

    

    MACE架构代表了机器学习力场在各种领域中的最新技术，能够处理内域、外推和低数据范围任务。本文对MACE进行了进一步评估，通过拟合已发表的基准数据集的模型来表明MACE在各种体系中的性能优于其他替代方案，包括非晶碳、一般的小分子有机化学、大分子和液态水。我们展示了模型在各个领域的能力，从约束几何优化到分子动力学模拟，发现其在所有测试领域都具有出色的性能。我们还表明，当在仅50个随机选定的参考配置上进行训练时，MACE即可非常高效地复现实验分子振动光谱。我们进一步证明，即使在大分子和弱相互作用的分子组装的情况下，这种基于严格局部的原子中心模型也是足够适用的。

    The MACE architecture represents the state of the art in the field of machine learning force fields for a variety of in-domain, extrapolation and low-data regime tasks. In this paper, we further evaluate MACE by fitting models for published benchmark datasets. We show that MACE generally outperforms alternatives for a wide range of systems from amorphous carbon and general small molecule organic chemistry to large molecules and liquid water. We demonstrate the capabilities of the model on tasks ranging from constrained geometry optimisation to molecular dynamics simulations and find excellent performance across all tested domains. We show that MACE is very data efficient, and can reproduce experimental molecular vibrational spectra when trained on as few as 50 randomly selected reference configurations. We further demonstrate that the strictly local atom-centered model is sufficient for such tasks even in the case of large molecules and weakly interacting molecular assemblies.
    
[^3]: ZeroSCROLLS：一个用于长文本理解的零Shot基准测试

    ZeroSCROLLS: A Zero-Shot Benchmark for Long Text Understanding. (arXiv:2305.14196v1 [cs.CL])

    [http://arxiv.org/abs/2305.14196](http://arxiv.org/abs/2305.14196)

    ZeroSCROLLS是一个用于长文本自然语言理解的零Shot基准测试，包括六个任务和四个数据集，能够评估大型语言模型的性能。当前，GPT-4的平均得分最高，但在聚合任务等多个挑战上，仍有改进的空间。

    

    我们介绍了 ZeroSCROLLS，这是一个用于长文本自然语言理解的零Shot基准测试，仅包含测试集而没有训练或开发数据。我们从SCROLLS基准测试中适应了六个任务，并添加了四个新数据集，包括两个新的信息融合任务，例如聚合正面评价的百分比。使用ZeroSCROLLS，我们对开源和闭源大型语言模型进行了全面评估，发现Claude优于ChatGPT，并且GPT-4获得了最高的平均分数。然而，在ZeroSCROLLS的多个开放挑战方面（例如，聚合任务），还有改进的空间，因为模型很难通过朴素的基准测试。由于最先进的技术还在不断更新，我们邀请研究人员在实时的ZeroSCROLLS排行榜上评估他们的想法。

    We introduce ZeroSCROLLS, a zero-shot benchmark for natural language understanding over long texts, which contains only test sets, without training or development data. We adapt six tasks from the SCROLLS benchmark, and add four new datasets, including two novel information fusing tasks, such as aggregating the percentage of positive reviews. Using ZeroSCROLLS, we conduct a comprehensive evaluation of both open-source and closed large language models, finding that Claude outperforms ChatGPT, and that GPT-4 achieves the highest average score. However, there is still room for improvement on multiple open challenges in ZeroSCROLLS, such as aggregation tasks, where models struggle to pass the naive baseline. As the state of the art is a moving target, we invite researchers to evaluate their ideas on the live ZeroSCROLLS leaderboard
    
[^4]: 通过预测修正提高基于得分扩散模型的收敛性

    Improved Convergence of Score-Based Diffusion Models via Prediction-Correction. (arXiv:2305.14164v1 [cs.LG])

    [http://arxiv.org/abs/2305.14164](http://arxiv.org/abs/2305.14164)

    本文通过使用预测校正方案，提高了基于得分扩散模型的收敛性。

    

    基于得分的生成模型（SGM）是从复杂数据分布中进行采样的强大工具。其基本思想是（i）通过向数据添加噪声运行时间为$T_1$的正向过程，（ii）估计其得分函数，并（iii）使用此估计值运行反向过程。由于反向过程以正向过程的平稳分布作为初始值，因此现有的分析范式要求$T_1\to\infty$。然而，从理论角度来看，对于给定的分数逼近精度，当$T_1$发散时，收敛保证将失败；从实际角度来看，$T_1$越大，计算成本就越高，并且会导致误差传播。本文通过考虑流行的预测器校正方案的一个版本来解决这个问题：在运行正向过程之后，我们首先通过不精确的 Langevin 动力学估计最终分布，然后恢复该过程。我们的关键技术贡献是提供了收敛保证。

    Score-based generative models (SGMs) are powerful tools to sample from complex data distributions. Their underlying idea is to (i) run a forward process for time $T_1$ by adding noise to the data, (ii) estimate its score function, and (iii) use such estimate to run a reverse process. As the reverse process is initialized with the stationary distribution of the forward one, the existing analysis paradigm requires $T_1\to\infty$. This is however problematic: from a theoretical viewpoint, for a given precision of the score approximation, the convergence guarantee fails as $T_1$ diverges; from a practical viewpoint, a large $T_1$ increases computational costs and leads to error propagation. This paper addresses the issue by considering a version of the popular predictor-corrector scheme: after running the forward process, we first estimate the final distribution via an inexact Langevin dynamics and then revert the process. Our key technical contribution is to provide convergence guarantees
    
[^5]: Neyman-Pearson检验的拟合优度研究

    Goodness of fit by Neyman-Pearson testing. (arXiv:2305.14137v1 [hep-ph])

    [http://arxiv.org/abs/2305.14137](http://arxiv.org/abs/2305.14137)

    本研究介绍了Neyman-Pearson检验在拟合优度研究中的应用，实现了一种名为NPLM的实用实现。和基于分类器方法相比，在探测到数据与期望分布的小偏差时，NPLM更灵敏且不会偏向任何类型的异常，比较适用于对撞机实验中对于新物理的不可知搜索。 future work 需要研究它在其他环境中的使用。

    

    当备选假设$H_1$足够通用，既不引入重大偏差又避免过度拟合时，Neyman-Pearson策略可以用于拟合优度检验。在高能物理的背景下，一种名为NPLM的实用实现已被开发，旨在探测标准模型未预料到的新物理效应，我们在本文中将该方法与其他拟合优度方法进行了比较。

    The Neyman-Pearson strategy for hypothesis testing can be employed for goodness of fit if the alternative hypothesis $\rm H_1$ is generic enough not to introduce a significant bias while at the same time avoiding overfitting. A practical implementation of this idea (dubbed NPLM) has been developed in the context of high energy physics, targeting the detection in collider data of new physical effects not foreseen by the Standard Model. In this paper we initiate a comparison of this methodology with other approaches to goodness of fit, and in particular with classifier-based strategies that share strong similarities with NPLM. NPLM emerges from our comparison as more sensitive to small departures of the data from the expected distribution and not biased towards detecting specific types of anomalies while being blind to others. These features make it more suited for agnostic searches for new physics at collider experiments. Its deployment in other contexts should be investigated.
    
[^6]: 神经网络学习轨迹的转移

    Transferring Learning Trajectories of Neural Networks. (arXiv:2305.14122v1 [cs.LG])

    [http://arxiv.org/abs/2305.14122](http://arxiv.org/abs/2305.14122)

    本研究提出了转移学习轨迹的算法，可将之前训练过的神经网络的学习轨迹应用在新的训练中，并能在任何直接训练之前实现非平凡的准确性。

    

    训练深度神经网络（DNN）是计算密集型的，这在执行重复训练运行（例如模型集成或知识蒸馏）时尤其成问题。一旦我们在某个数据集上训练了一个DNN，我们就拥有了其学习轨迹（即训练期间的中间参数序列），其中可能包含学习数据集的有用信息。然而，尚未尝试利用给定学习轨迹的这种信息进行另一种训练。本文将问题形式化为“转移”给定学习轨迹从一个初始参数到另一个初始参数，称为学习转移问题，并通过匹配沿轨迹逐渐平移对称性的梯度导出了第一个算法，以近似解决它。我们经验证明，转移参数在任何直接训练之前就能达到非平凡的准确性。此外，我们分析了转移参数的损失景观属性。

    Training deep neural networks (DNNs) is computationally expensive, which is problematic especially when performing duplicated training runs, such as model ensemble or knowledge distillation. Once we have trained one DNN on some dataset, we have its learning trajectory (i.e., a sequence of intermediate parameters during training) which may potentially contain useful information for learning the dataset. However, there has been no attempt to utilize such information of a given learning trajectory for another training. In this paper, we formulate the problem of "transferring" a given learning trajectory from one initial parameter to another one, called learning transfer problem, and derive the first algorithm to approximately solve it by matching gradients successively along the trajectory via permutation symmetry. We empirically show that the transferred parameters achieve non-trivial accuracy before any direct training. Also, we analyze the loss landscape property of the transferred par
    
[^7]: 基于代价感知的情境变量在贝叶斯优化中的学习

    Cost-aware learning of relevant contextual variables within Bayesian optimization. (arXiv:2305.14120v1 [cs.LG])

    [http://arxiv.org/abs/2305.14120](http://arxiv.org/abs/2305.14120)

    本文提出一种基于代价感知的模型选择BO方法SADCBO，通过对后验代理模型的敏感性分析来学习关于环境的相关情境信息，并通过平均模型预测来最小化优化代价，在实验中表现出卓越的性能。

    

    情境贝叶斯优化(CBO)是一种强大的框架，可针对设计变量优化黑盒昂贵的评估函数，并同时有效地整合关于环境的相关情境信息，如实验条件。然而，在许多实际场景中，情境变量的相关性不一定是预先已知的。此外，有时还可以最优化情境变量本身，这是当前CBO算法未考虑的设置。优化情境变量可能是昂贵的，这引出了确定一个最小相关子集的问题。在本文中，我们将这个问题作为一个代价感知的模型选择BO任务来构架，采用一种新方法，即基于敏感性分析的情境BO (SADCBO) 来解决这个问题。我们通过对特定输入点后验代理模型的敏感性分析来学习情境变量的相关性，同时通过平均模型预测来最小化优化的代价。SADCBO在多个合成和真实基准问题上进行了实证评估，显示出优于现有算法的性能。

    Contextual Bayesian Optimization (CBO) is a powerful framework for optimizing black-box, expensive-to-evaluate functions with respect to design variables, while simultaneously efficiently integrating relevant contextual information regarding the environment, such as experimental conditions. However, in many practical scenarios, the relevance of contextual variables is not necessarily known beforehand. Moreover, the contextual variables can sometimes be optimized themselves, a setting that current CBO algorithms do not take into account. Optimizing contextual variables may be costly, which raises the question of determining a minimal relevant subset. In this paper, we frame this problem as a cost-aware model selection BO task and address it using a novel method, Sensitivity-Analysis-Driven Contextual BO (SADCBO). We learn the relevance of context variables by sensitivity analysis of the posterior surrogate model at specific input points, whilst minimizing the cost of optimization by lev
    
[^8]: 通过能量感知的早期退出实现可持续的边缘智能

    Sustainable Edge Intelligence Through Energy-Aware Early Exiting. (arXiv:2305.14094v1 [eess.SY])

    [http://arxiv.org/abs/2305.14094](http://arxiv.org/abs/2305.14094)

    本文提出了能量自适应动态早期退出机制，通过能量感知的策略，在EH边缘设备中实现了高效准确推理。

    

    深度学习模型已成为物联网应用的一种有前途的解决方案。然而，由于其计算复杂性，深度学习模型消耗大量能量，这可能会快速耗尽电池并影响物联网设备的性能。为了实现可持续运行，本文考虑一个带有可充电电池和能量收获能力的边缘设备。除了环境能源的随机性外，收获速率通常不足以满足推理能源需求，在能源不可知的设备中会导致严重的性能降低。为了解决这个问题，我们提出了能量自适应动态早期退出机制，以实现在充满环境能源情况下的高效准确推理。

    Deep learning (DL) models have emerged as a promising solution for Internet of Things (IoT) applications. However, due to their computational complexity, DL models consume significant amounts of energy, which can rapidly drain the battery and compromise the performance of IoT devices. For sustainable operation, we consider an edge device with a rechargeable battery and energy harvesting (EH) capabilities. In addition to the stochastic nature of the ambient energy source, the harvesting rate is often insufficient to meet the inference energy requirements, leading to drastic performance degradation in energy-agnostic devices. To mitigate this problem, we propose energy-adaptive dynamic early exiting (EE) to enable efficient and accurate inference in an EH edge intelligence system. Our approach derives an energy-aware EE policy that determines the optimal amount of computational processing on a per-sample basis. The proposed policy balances the energy consumption to match the limited inco
    
[^9]: 警惕尖峰：固定维度下内核和神经网络的良性过拟合

    Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension. (arXiv:2305.14077v1 [stat.ML])

    [http://arxiv.org/abs/2305.14077](http://arxiv.org/abs/2305.14077)

    这篇论文研究了固定维度下内核和神经网络的良性过拟合，发现良性过拟合的关键在于估计器的平滑度而不是维数，并证明在固定维度下中度导数的良性过拟合是不可能的。相反，我们证明了用序列核进行回归是可能出现良性过拟合的。

    

    过度参数化的神经网络训练达到接近零的训练误差的成功引起了人们对良性过拟合现象的极大兴趣，即使估计器插值嘈杂的训练数据，它们还是具有统计一致性。尽管某些学习方法的固定维度下已经确定了良性过拟合，但目前的文献表明，对于典型内核方法和宽神经网络的回归，良性过拟合需要高维度设置，其中维数随着样本大小的增加而增加。本文表明，估计器的平滑度是关键，而不是维数：只有当估计器的导数足够大时，良性过拟合才可能发生。我们将现有的不一致性结果推广到非插值模型和更多内核，以表明在固定维度下中度导数的良性过拟合是不可能的。相反，我们证明了用序列核进行回归是可能出现良性过拟合的。

    The success of over-parameterized neural networks trained to near-zero training error has caused great interest in the phenomenon of benign overfitting, where estimators are statistically consistent even though they interpolate noisy training data. While benign overfitting in fixed dimension has been established for some learning methods, current literature suggests that for regression with typical kernel methods and wide neural networks, benign overfitting requires a high-dimensional setting where the dimension grows with the sample size. In this paper, we show that the smoothness of the estimators, and not the dimension, is the key: benign overfitting is possible if and only if the estimator's derivatives are large enough. We generalize existing inconsistency results to non-interpolating models and more kernels to show that benign overfitting with moderate derivatives is impossible in fixed dimension. Conversely, we show that benign overfitting is possible for regression with a seque
    
[^10]: 关于高斯-斯坦变分梯度下降动态性的探究

    Towards Understanding the Dynamics of Gaussian--Stein Variational Gradient Descent. (arXiv:2305.14076v1 [math.ST])

    [http://arxiv.org/abs/2305.14076](http://arxiv.org/abs/2305.14076)

    本文探究了高斯-斯坦变分梯度下降动态性。对于从高斯目标中采样，只要初始值是高斯的，具有双线性核的SVGD动态将保持高斯状态。当目标函数呈现出强对数凹性时，证明了均场高斯-SVGD动态会线性收敛于KL散度下最接近目标高斯分布。在有限粒子设置中，存在对均场极限的时间微步一致收敛以及线性收敛至目标高斯分布。

    

    Stein Variational Gradient Descent (SVGD)是一种非参数基于粒子的确定性采样算法。尽管其被广泛使用，但理解SVGD的理论属性一直是一个具有挑战性的问题。对于从高斯目标中采样，只要初始值是高斯的，具有双线性核的SVGD动态将保持高斯状态。受此事实的启发，我们通过双线性核将SVGD投影到高斯分布族中，即高斯变分推断 (GVI) 与 SVGD。我们通过考虑均场 PDE 和离散粒子系统，提供了一个完整的图像。当目标函数呈现出强对数凹性时，证明了均场高斯-SVGD动态会线性收敛于KL散度下最接近目标高斯分布。在有限粒子设置中，存在对均场极限的时间微步一致收敛以及线性收敛至目标高斯分布。我们的分析基于一个新的代数恒等式，该等式将目标高斯分布的费希尔信息矩阵与粒子均匀分布的费希尔信息矩阵相关联。这个等式为我们提供了透视 GVI with SVGD 在均场和粒子设置中的动态性的统一视角。

    Stein Variational Gradient Descent (SVGD) is a nonparametric particle-based deterministic sampling algorithm. Despite its wide usage, understanding the theoretical properties of SVGD has remained a challenging problem. For sampling from a Gaussian target, the SVGD dynamics with a bilinear kernel will remain Gaussian as long as the initializer is Gaussian. Inspired by this fact, we undertake a detailed theoretical study of the Gaussian-SVGD, i.e., SVGD projected to the family of Gaussian distributions via the bilinear kernel, or equivalently Gaussian variational inference (GVI) with SVGD. We present a complete picture by considering both the mean-field PDE and discrete particle systems. When the target is strongly log-concave, the mean-field Gaussian-SVGD dynamics is proven to converge linearly to the Gaussian distribution closest to the target in KL divergence. In the finite-particle setting, there is both uniform in time convergence to the mean-field limit and linear convergence in ti
    
[^11]: DIVA：基于狄利克雷过程的变分自编码器的增量深度聚类算法

    DIVA: A Dirichlet Process Based Incremental Deep Clustering Algorithm via Variational Auto-Encoder. (arXiv:2305.14067v1 [cs.LG])

    [http://arxiv.org/abs/2305.14067](http://arxiv.org/abs/2305.14067)

    本文提出了DIVA算法，一个基于狄利克雷过程的增量深度聚类框架，利用无限混合高斯作为先验，并利用一种记忆化的在线变分推理方法实现簇的动态适应移动，而不需要先知道特征的数量。该算法表现优越，特别是在增量特征的情况下。

    

    基于生成模型的深度聚类框架在分类复杂数据方面表现出色，但在处理动态和复杂特征方面受到限制，因为它们需要先知道簇的数量。本文提出了一个非参数深度聚类框架，采用无限混合高斯作为先验。我们的框架利用一种记忆化的在线变分推理方法，实现了簇的“出生”和“合并”移动，使我们的框架能够以“动态适应”的方式聚类数据，而不需要先知道特征的数量。我们把该框架命名为DIVA，即基于狄利克雷过程的增量深度聚类框架的变分自编码器。我们的框架在分类具有动态变化特征的复杂数据方面表现优越，特别是在增量特征的情况下，超过了最先进的基准。

    Generative model-based deep clustering frameworks excel in classifying complex data, but are limited in handling dynamic and complex features because they require prior knowledge of the number of clusters. In this paper, we propose a nonparametric deep clustering framework that employs an infinite mixture of Gaussians as a prior. Our framework utilizes a memoized online variational inference method that enables the "birth" and "merge" moves of clusters, allowing our framework to cluster data in a "dynamic-adaptive" manner, without requiring prior knowledge of the number of features. We name the framework as DIVA, a Dirichlet Process-based Incremental deep clustering framework via Variational Auto-Encoder. Our framework, which outperforms state-of-the-art baselines, exhibits superior performance in classifying complex data with dynamically changing features, particularly in the case of incremental features.
    
[^12]: 基于凸组合的表达性损失可以提高网络的对抗鲁棒性

    Expressive Losses for Verified Robustness via Convex Combinations. (arXiv:2305.13991v1 [cs.LG])

    [http://arxiv.org/abs/2305.13991](http://arxiv.org/abs/2305.13991)

    通过基于凸组合的表达性损失，可以提高网络的对抗鲁棒性，最新的算法可以获得最先进的结果；这种方法通过对抗性攻击和IBP边界之间的简单凸组合进行实现。

    

    先前的工作通常通过（扰动区域的子集）的最坏情况下限，或在对抗训练之上引入可验证性来训练具有已验证鲁棒性的网络。最先进性能的关键在于所使用的损失函数的表达能力，它应该能够匹配训练后要使用的验证器的紧密度。我们形式化定义了表达力，并表明它可以通过对抗性攻击和IBP边界之间的简单凸组合来满足。然后，我们展示了所得到的算法，命名为CC-IBP和MTL-IBP，在各种设置中均可以产生最先进的结果，尽管其概念上是简单的。特别地，在TinyImageNet和缩小的ImageNet上，对于半径为$ \frac{1} {255} $的$ \ell_ \infty $扰动，MTL-IBP可以将文献中最佳标准和验证准确性从$1.98\%$提高到$3.92\%$，同时仅依赖于单步自适应优化。

    In order to train networks for verified adversarial robustness, previous work typically over-approximates the worst-case loss over (subsets of) perturbation regions or induces verifiability on top of adversarial training. The key to state-of-the-art performance lies in the expressivity of the employed loss function, which should be able to match the tightness of the verifiers to be employed post-training. We formalize a definition of expressivity, and show that it can be satisfied via simple convex combinations between adversarial attacks and IBP bounds. We then show that the resulting algorithms, named CC-IBP and MTL-IBP, yield state-of-the-art results across a variety of settings in spite of their conceptual simplicity. In particular, for $\ell_\infty$ perturbations of radius $\frac{1}{255}$ on TinyImageNet and downscaled ImageNet, MTL-IBP improves on the best standard and verified accuracies from the literature by from $1.98\%$ to $3.92\%$ points while only relying on single-step ad
    
[^13]: 无需Lipschitzness和Smoothness的在线投资组合选择的数据相关上界

    Data-Dependent Bounds for Online Portfolio Selection Without Lipschitzness and Smoothness. (arXiv:2305.13946v1 [cs.LG])

    [http://arxiv.org/abs/2305.13946](http://arxiv.org/abs/2305.13946)

    本文提出了在线投资组合选择的第一个数据相关上界，算法显示亚线性遗憾率，并在数据“容易”时实现对数遗憾。

    

    本文介绍了在线投资组合选择中的第一种小损失和平稳变化的遗憾上界，并标志着在线凸优化具有非Lipschitz、非光滑损失的数据相关上界的首次实例。我们提出的算法在最坏情况下显示出亚线性遗憾率，并在数据“容易”时实现对数遗憾，每次迭代的时间几乎是投资选择数量的线性。遗憾上界是使用对数损失的新型光滑性表征、遵循具有自共轭正则化器的正则化领袖（FTRL）的局部范数分析、它们不一定是障碍的和具有log障碍的乐观FTRL的隐式变体来推导的。

    This work introduces the first small-loss and gradual-variation regret bounds for online portfolio selection, marking the first instances of data-dependent bounds for online convex optimization with non-Lipschitz, non-smooth losses. The algorithms we propose exhibit sublinear regret rates in the worst cases and achieve logarithmic regrets when the data is "easy," with per-iteration time almost linear in the number of investment alternatives. The regret bounds are derived using novel smoothness characterizations of the logarithmic loss, a local norm-based analysis of following the regularized leader (FTRL) with self-concordant regularizers, which are not necessarily barriers, and an implicit variant of optimistic FTRL with the log-barrier.
    
[^14]: 随机梯度 Langevin 扩散中的子采样误差

    Subsampling Error in Stochastic Gradient Langevin Diffusions. (arXiv:2305.13882v1 [stat.ML])

    [http://arxiv.org/abs/2305.13882](http://arxiv.org/abs/2305.13882)

    该研究分析了随机梯度 Langevin 动力学在大型数据环境下使用子采样产生的误差。研究者提出了一种新的连续时间马尔可夫过程，该过程切换数据子集并可用于扩散子采样 MCMC 方法，并证明了该方法的收敛性。

    

    随机梯度 Langevin 动力学 (SGLD) 通常用于大规模数据的统计学习中近似贝叶斯后验分布。与许多常规马尔可夫链蒙特卡罗 (MCMC) 算法不同，SGLD 对于后验分布不是稳定的。它有两个错误来源：第一个错误是由 Euler-Maruyama 离散化 Langevin 扩散过程引入的，第二个错误来自于数据子采样，这使得它适用于大规模数据环境。在本文中，我们考虑了 SGLD 的理想化版本，以分析该方法的纯子采样误差，我们可以将其视为基于扩散的子采样 MCMC 方法的最佳情况误差。事实上，我们引入并研究了随机梯度 Langevin 扩散 (SGLDiff)，这是一个连续时间马尔可夫过程，它遵循与数据子集相应的 Langevin 扩散，并在指数等待时间后切换该数据子集。在此，我们证明了瓦瑟斯坦距离 (Was)

    The Stochastic Gradient Langevin Dynamics (SGLD) are popularly used to approximate Bayesian posterior distributions in statistical learning procedures with large-scale data. As opposed to many usual Markov chain Monte Carlo (MCMC) algorithms, SGLD is not stationary with respect to the posterior distribution; two sources of error appear: The first error is introduced by an Euler--Maruyama discretisation of a Langevin diffusion process, the second error comes from the data subsampling that enables its use in large-scale data settings. In this work, we consider an idealised version of SGLD to analyse the method's pure subsampling error that we then see as a best-case error for diffusion-based subsampling MCMC methods. Indeed, we introduce and study the Stochastic Gradient Langevin Diffusion (SGLDiff), a continuous-time Markov process that follows the Langevin diffusion corresponding to a data subset and switches this data subset after exponential waiting times. There, we show that the Was
    
[^15]: 针对大规模高斯过程回归和统计有限元分析的随机PDE表示随机场

    Stochastic PDE representation of random fields for large-scale Gaussian process regression and statistical finite element analysis. (arXiv:2305.13879v1 [math.NA])

    [http://arxiv.org/abs/2305.13879](http://arxiv.org/abs/2305.13879)

    本文针对工程学和机器学习中的贝叶斯建模，使用随机PDE表示来开发一种可扩展的框架，从而可以在几何复杂的域上进行大规模的统计有限元分析和高斯过程回归。

    

    在工程学和机器学习的贝叶斯建模中，有效表示几何复杂域上的随机场至关重要。当前普遍使用的随机场表示仅限于无界域或在可能的场属性方面过于受限。因此，利用随机PDE与随机场之间的历史联系的新技术对于具有复杂几何形状和存在有限元离散化用于求解物理守恒方程的工程应用尤为吸引人。与随机场的密集协方差矩阵不同，其逆矩阵--精度矩阵通常是稀疏的，并等于类似Helmholtz的随机PDE的刚度矩阵。在本文中，我们使用SPDE表示来开发可扩展的框架，用于在几何复杂域上进行大规模的统计有限元分析（StatFEM）和高斯过程（GP）回归。我们使用SPDE公式

    The efficient representation of random fields on geometrically complex domains is crucial for Bayesian modelling in engineering and machine learning. Today's prevalent random field representations are restricted to unbounded domains or are too restrictive in terms of possible field properties. As a result, new techniques leveraging the historically established link between stochastic PDEs (SPDEs) and random fields are especially appealing for engineering applications with complex geometries which already have a finite element discretisation for solving the physical conservation equations. Unlike the dense covariance matrix of a random field, its inverse, the precision matrix, is usually sparse and equal to the stiffness matrix of a Helmholtz-like SPDE. In this paper, we use the SPDE representation to develop a scalable framework for large-scale statistical finite element analysis (statFEM) and Gaussian process (GP) regression on geometrically complex domains. We use the SPDE formulatio
    
[^16]: 论拜占庭容错分布式学习的最佳批处理大小

    On the Optimal Batch Size for Byzantine-Robust Distributed Learning. (arXiv:2305.13856v1 [cs.LG])

    [http://arxiv.org/abs/2305.13856](http://arxiv.org/abs/2305.13856)

    本文研究的问题是在拜占庭容错分布式学习中，当梯度计算总数固定时，最佳的批处理大小随拜占庭工人的比例增加而增加。

    

    近来，由于意外失误或恶意攻击导致计算设备异常行为的拜占庭容错分布式学习（BRDL）已成为热门研究课题。然而，在独立同分布（i.i.d.）的情况下，由于随机梯度的大方差，现有的BRDL方法仍会导致模型准确率显著下降。增加批处理大小是减少方差的简单而有效的方法。然而，当梯度计算总数固定时，过大的批处理大小会导致迭代次数过少（更新次数），可能也会降低模型准确率。针对这一挑战，本文主要研究在固定梯度计算总数的情况下最佳的批处理大小。具体而言，我们在理论和经验上表明，当梯度计算总数固定时，BRDL中最佳的批处理大小随拜占庭工人的比例增加而增加。

    Byzantine-robust distributed learning (BRDL), in which computing devices are likely to behave abnormally due to accidental failures or malicious attacks, has recently become a hot research topic. However, even in the independent and identically distributed (i.i.d.) case, existing BRDL methods will suffer from a significant drop on model accuracy due to the large variance of stochastic gradients. Increasing batch sizes is a simple yet effective way to reduce the variance. However, when the total number of gradient computation is fixed, a too-large batch size will lead to a too-small iteration number (update number), which may also degrade the model accuracy. In view of this challenge, we mainly study the optimal batch size when the total number of gradient computation is fixed in this work. In particular, we theoretically and empirically show that when the total number of gradient computation is fixed, the optimal batch size in BRDL increases with the fraction of Byzantine workers. Ther
    
[^17]: 利用积分概率测量进行协变量平衡的因果推断方法

    Covariate balancing using the integral probability metric for causal inference. (arXiv:2305.13715v1 [stat.ML])

    [http://arxiv.org/abs/2305.13715](http://arxiv.org/abs/2305.13715)

    本文介绍了一种利用积分概率测量进行协变量平衡的因果推断方法，无需正确规定倾向得分或结果回归模型即可保证估计器的一致性，并且在实验中表现出优异性能。

    

    在因果推断中，加权方法被广泛用于实现令人满意的协变量平衡。然而，现有的加权方法只有在某种模型（如倾向得分或结果回归模型）被正确规定时才具有理想的理论属性，并且即使模型被正确规定，相应的估计器在有限样本情况下也不表现良好。本文考虑利用积分概率度量（IPM）进行协变量平衡。确定最佳权重，使得针对给定的判别器，治疗组和对照组的加权经验分布具有最小的IPM值。我们证明了对应的估计器可以是一致的，而不需要正确地规定任何模型（既不是倾向得分模型也不是结果回归模型）。此外，我们在实验中表明，我们提出的方法优于已有的加权方法。

    Weighting methods in causal inference have been widely used to achieve a desirable level of covariate balancing. However, the existing weighting methods have desirable theoretical properties only when a certain model, either the propensity score or outcome regression model, is correctly specified. In addition, the corresponding estimators do not behave well for finite samples due to large variance even when the model is correctly specified. In this paper, we consider to use the integral probability metric (IPM), which is a metric between two probability measures, for covariate balancing. Optimal weights are determined so that weighted empirical distributions for the treated and control groups have the smallest IPM value for a given set of discriminators. We prove that the corresponding estimator can be consistent without correctly specifying any model (neither the propensity score nor the outcome regression model). In addition, we empirically show that our proposed method outperforms e
    
[^18]: 通过RKHM和Perron-Frobenius算子的深度学习

    Deep Learning with Kernels through RKHM and the Perron-Frobenius Operator. (arXiv:2305.13588v1 [stat.ML])

    [http://arxiv.org/abs/2305.13588](http://arxiv.org/abs/2305.13588)

    该论文提出了一种基于核方法的深度学习框架：深度RKHM，通过使用$C^*$代数获得更温和的界限，并提供了良性过拟合的理论解释。

    

    重现核希尔伯特$C^*$-模(RKHM)通过$C^*$代数对重现核希尔伯特空间(RKHS)进行了泛化，而Perron-Frobenius算子是与函数组合相关的线性算子。将这两个概念结合起来，我们提出了深度RKHM，一种基于核方法的深度学习框架。我们在这个设置中推导了一个新的Rademacher广义界限，并通过Perron-Frobenius算子提供了良性过拟合的理论解释。由于$C^*$代数的优势，该界限对输出维度的依赖性较现有界限更加温和。我们展示了$C^*$代数是深度学习的核心工具，使我们能够利用算子的乘积结构，并提供与卷积神经网络的明确联系。我们的理论分析为设计和分析深度核方法提供了一个新的视角。

    Reproducing kernel Hilbert $C^*$-module (RKHM) is a generalization of reproducing kernel Hilbert space (RKHS) by means of $C^*$-algebra, and the Perron-Frobenius operator is a linear operator related to the composition of functions. Combining these two concepts, we present deep RKHM, a deep learning framework for kernel methods. We derive a new Rademacher generalization bound in this setting and provide a theoretical interpretation of benign overfitting by means of Perron-Frobenius operators. By virtue of $C^*$-algebra, the dependency of the bound on output dimension is milder than existing bounds. We show that $C^*$-algebra is a suitable tool for deep learning with kernels, enabling us to take advantage of the product structure of operators and to provide a clear connection with convolutional neural networks. Our theoretical analysis provides a new lens through which one can design and analyze deep kernel methods.
    
[^19]: 平方神经分布族：一种新的可计算密度模型类

    Squared Neural Families: A New Class of Tractable Density Models. (arXiv:2305.13552v1 [cs.LG])

    [http://arxiv.org/abs/2305.13552](http://arxiv.org/abs/2305.13552)

    提出一种新的可计算密度模型类——平方神经分布族，其通过对神经网络的2范数进行平方和基于某个基础度量进行归一化，严格推广了经典指数族，具有闭性条件推断和可计算的边际分布。

    

    概率分布的灵活模型是许多机器学习任务的重要组成部分。我们开发并研究了一种新的概率分布类别，称为平方神经分布族（SNEFY），通过对神经网络的2范数进行平方并基于某个基础度量进行归一化。类似于无穷宽的神经网络和高斯过程之间的广泛联系的推理，我们展示了在许多感兴趣的情况下，SNEFY具有封闭形式的标准化常数，因此是灵活且完全可计算密度模型。SNEFY严格推广了经典的指数族，对于条件推断具有闭性，并且具有可计算的边际分布。我们在各种密度估计和条件密度估计任务中展示其实用性。

    Flexible models for probability distributions are an essential ingredient in many machine learning tasks. We develop and investigate a new class of probability distributions, which we call a Squared Neural Family (SNEFY), formed by squaring the 2-norm of a neural network and normalising it with respect to a base measure. Following the reasoning similar to the well established connections between infinitely wide neural networks and Gaussian processes, we show that SNEFYs admit a closed form normalising constants in many cases of interest, thereby resulting in flexible yet fully tractable density models. SNEFYs strictly generalise classical exponential families, are closed under conditioning, and have tractable marginal distributions. Their utility is illustrated on a variety of density estimation and conditional density estimation tasks. Software available at https://github.com/RussellTsuchida/snefy.
    
[^20]: Group-Invariant GAN的统计保证

    Statistical Guarantees of Group-Invariant GANs. (arXiv:2305.13517v1 [stat.ML])

    [http://arxiv.org/abs/2305.13517](http://arxiv.org/abs/2305.13517)

    本研究提出了群不变GAN的统计保证，发现当学习群不变分布时，群不变GAN所需样本数会按群体大小的幂比例减少。

    

    Group-Invariant生成对抗网络(GAN)是一种GAN，其中生成器和判别器具有硬性集团对称性。实证研究表明，这些网络能够学习具有显着改进数据效率的集团不变分布。在本研究中，我们旨在通过分析群不变GAN的样本复杂度减少来严格量化这种改进。我们的研究发现，在学习群不变分布时，群不变GAN所需样本数按照群体大小的幂比例减少，这个幂取决于分布支持的本质维度。据我们所知，这项工作是首个为群不变生成模型，特别是GAN提供统计估计的工作，并可以为其他群不变生成模型的研究提供借鉴。

    Group-invariant generative adversarial networks (GANs) are a type of GANs in which the generators and discriminators are hardwired with group symmetries. Empirical studies have shown that these networks are capable of learning group-invariant distributions with significantly improved data efficiency. In this study, we aim to rigorously quantify this improvement by analyzing the reduction in sample complexity for group-invariant GANs. Our findings indicate that when learning group-invariant distributions, the number of samples required for group-invariant GANs decreases proportionally with a power of the group size, and this power depends on the intrinsic dimension of the distribution's support. To our knowledge, this work presents the first statistical estimation for group-invariant generative models, specifically for GANs, and it may shed light on the study of other group-invariant generative models.
    
[^21]: 用于带测量噪声的Ornstein-Uhlenbeck过程参数估计

    Parameter estimation from an Ornstein-Uhlenbeck process with measurement noise. (arXiv:2305.13498v1 [stat.ML])

    [http://arxiv.org/abs/2305.13498](http://arxiv.org/abs/2305.13498)

    本文研究了带有测量噪声的Ornstein-Uhlenbeck过程参数估计，提出了算法和方法能够分离热噪声和乘性噪声，并改善数据分析的参数估计精度。

    

    本文旨在研究噪声对Ornstein-Uhlenbeck过程参数拟合的影响，重点考察了乘性噪声和热噪声对信号分离精度的影响。为了解决这些问题，我们提出了有效区分热噪声和乘性噪声、改善参数估计精度的算法和方法，探讨了乘性和热噪声对实际信号混淆的影响，并提出了解决方法。首先，我们提出了一种可以有效分离热噪声的算法，其性能可与Hamilton Monte Carlo (HMC)相媲美，但速度显著提高。随后，我们分析了乘性噪声，并证明了HMC无法隔离热噪声和乘性噪声。然而，我们展示了，在额外了解热噪声和乘性噪声之间比率的情况下，我们可以精确地估计参数和分离信号。

    This article aims to investigate the impact of noise on parameter fitting for an Ornstein-Uhlenbeck process, focusing on the effects of multiplicative and thermal noise on the accuracy of signal separation. To address these issues, we propose algorithms and methods that can effectively distinguish between thermal and multiplicative noise and improve the precision of parameter estimation for optimal data analysis. Specifically, we explore the impact of both multiplicative and thermal noise on the obfuscation of the actual signal and propose methods to resolve them. Firstly, we present an algorithm that can effectively separate thermal noise with comparable performance to Hamilton Monte Carlo (HMC) but with significantly improved speed. Subsequently, we analyze multiplicative noise and demonstrate that HMC is insufficient for isolating thermal and multiplicative noise. However, we show that, with additional knowledge of the ratio between thermal and multiplicative noise, we can accuratel
    
[^22]: 非参数高斯图拼接及其在钙成像中的应用

    Nonparanormal Graph Quilting with Applications to Calcium Imaging. (arXiv:2305.13491v1 [stat.ME])

    [http://arxiv.org/abs/2305.13491](http://arxiv.org/abs/2305.13491)

    本文研究了钙成像中的图拼接问题，提出了两种非高斯图形模型的解决方案，并在模拟和真实数据上验证了其有效性。

    

    概率图模型已成为一种重要的无监督学习工具，用于检测各种问题的网络结构，包括从双光子钙成像数据中估计功能神经元连接性。然而，在钙成像的情况下，技术限制只允许记录感兴趣区域中部分重叠的神经元层共同记录。在这种情况下，对于完整数据的图形估计需要针对边缘选择进行推断，当许多神经元对没有同时观察到时，这就导致了图拼接问题，在经验协方差矩阵存在块状缺失的情况下估计图形。以前已经研究了高斯图形模型的图拼接问题的解决方案；然而，来自钙成像的神经活动数据通常是非高斯的，因此需要一种更灵活的建模方法。因此，在我们的工作中，我们研究了两种非参数非高斯图钉合方法，允许在缺少块的情况下估计底层的图形结构。我们展示了我们的方法在模拟和真实的钙成像数据上的功效。

    Probabilistic graphical models have become an important unsupervised learning tool for detecting network structures for a variety of problems, including the estimation of functional neuronal connectivity from two-photon calcium imaging data. However, in the context of calcium imaging, technological limitations only allow for partially overlapping layers of neurons in a brain region of interest to be jointly recorded. In this case, graph estimation for the full data requires inference for edge selection when many pairs of neurons have no simultaneous observations. This leads to the Graph Quilting problem, which seeks to estimate a graph in the presence of block-missingness in the empirical covariance matrix. Solutions for the Graph Quilting problem have previously been studied for Gaussian graphical models; however, neural activity data from calcium imaging are often non-Gaussian, thereby requiring a more flexible modeling approach. Thus, in our work, we study two approaches for nonpara
    
[^23]: 关于神经网络分类性能优化基于加权度量的综合理论框架

    A comprehensive theoretical framework for the optimization of neural networks classification performance with respect to weighted metrics. (arXiv:2305.13472v1 [cs.LG])

    [http://arxiv.org/abs/2305.13472](http://arxiv.org/abs/2305.13472)

    本论文提出了一个理论框架，可以驱使模型优化加权分类度量标准，包括成本敏感学习、加权交叉熵损失函数和值加权技能得分等已确立的方法。

    

    在许多情况下，为了评估神经网络所做出的预测的准确程度，需要设计定制化和加权分类评分方法。然而，在训练阶段中最大化这些评分与最小化损失函数之间存在差异。本文提出了一个完整的理论框架，形式化了加权分类度量，并允许构建损失函数以驱使模型优化这些有趣的指标。经过详细的理论分析，我们发现我们的框架包括经典的成本敏感学习、加权交叉熵损失函数和值加权技能得分等已确立的方法。

    In many contexts, customized and weighted classification scores are designed in order to evaluate the goodness of the predictions carried out by neural networks. However, there exists a discrepancy between the maximization of such scores and the minimization of the loss function in the training phase. In this paper, we provide a complete theoretical setting that formalizes weighted classification metrics and then allows the construction of losses that drive the model to optimize these metrics of interest. After a detailed theoretical analysis, we show that our framework includes as particular instances well-established approaches such as classical cost-sensitive learning, weighted cross entropy loss functions and value-weighted skill scores.
    
[^24]: 通过同簇预言机的容错精确查询学习有限集合划分

    Error-Tolerant Exact Query Learning of Finite Set Partitions with Same-Cluster Oracle. (arXiv:2305.13402v1 [cs.DS])

    [http://arxiv.org/abs/2305.13402](http://arxiv.org/abs/2305.13402)

    本文提出了一个新问题：如何通过同簇预言机在存在有限对抗错误时积极学习完全恢复划分。我们建立了解析框架并证明了最坏情况下查询复杂度的上下界，并研究了适应性和查询复杂度之间的关系。

    

    本文研究了当存在有限的对抗错误时，仅通过同簇预言机来积极学习完全恢复划分的问题。首先突出了学习划分和相关聚类之间的新颖联系。然后利用这种联系为这个问题建立了一个Rényi-Ulam样式的解析框架，并证明了最坏情况下查询复杂度的上下界。此外，我们还限制了相关随机算法的期望性能。最后，我们研究了适应性和查询复杂度在该问题和相关变体中之间的关系。

    This paper initiates the study of active learning for exact recovery of partitions exclusively through access to a same-cluster oracle in the presence of bounded adversarial error. We first highlight a novel connection between learning partitions and correlation clustering. Then we use this connection to build a R\'enyi-Ulam style analytical framework for this problem, and prove upper and lower bounds on its worst-case query complexity. Further, we bound the expected performance of a relevant randomized algorithm. Finally, we study the relationship between adaptivity and query complexity for this problem and related variants.
    
[^25]: 从随机搜索到度量测度空间中的赌博学习

    From Random Search to Bandit Learning in Metric Measure Spaces. (arXiv:2305.11509v1 [cs.LG])

    [http://arxiv.org/abs/2305.11509](http://arxiv.org/abs/2305.11509)

    本文介绍了随机搜索及其性能，引入了“散射维度”的概念，描述了底层函数的状态，量化了随机搜索的性能，并证明了在无噪声和有界噪声情况下的输出分别以一定概率收敛到最优值。

    

    随机搜索是超参数优化中最常用的方法之一，对于深度学习模型的成功至关重要。尽管其性能令人惊叹，但很少有非启发式的理论用于描述其工作机制。本文给出了关于随机搜索的理论解释。我们引入了“散射维度”的概念，描述了底层函数的状态，并量化了随机搜索的性能。我们表明，当环境没有噪声时，随机搜索的输出以概率收敛到最优值，其速率为$ \widetilde{\mathcal{O}} \left( \left( \frac{1}{T} \right)^{ \frac{1}{d_s} } \right) $，其中$ d_s \ge 0 $是底层函数的散射维度。当观察到的函数值受到有界的独立同分布噪声影响时，随机搜索的输出以概率收敛到最优值，速率为$ \widetilde{\mathcal{O}} \left( \left( \frac{1}{T} \right)^{ \frac{2}{2+d_s} } \right) $。

    Random Search is one of the most widely-used method for Hyperparameter Optimization, and is critical to the success of deep learning models. Despite its astonishing performance, little non-heuristic theory has been developed to describe the underlying working mechanism. This paper gives a theoretical accounting of Random Search. We introduce the concept of \emph{scattering dimension} that describes the landscape of the underlying function, and quantifies the performance of random search. We show that, when the environment is noise-free, the output of random search converges to the optimal value in probability at rate $ \widetilde{\mathcal{O}} \left( \left( \frac{1}{T} \right)^{ \frac{1}{d_s} } \right) $, where $ d_s \ge 0 $ is the scattering dimension of the underlying function. When the observed function values are corrupted by bounded $iid$ noise, the output of random search converges to the optimal value in probability at rate $ \widetilde{\mathcal{O}} \left( \left( \frac{1}{T} \rig
    
[^26]: 知识迁移下的因果效应估计: 转移因果学习

    Transfer Causal Learning: Causal Effect Estimation with Knowledge Transfer. (arXiv:2305.09126v1 [cs.LG])

    [http://arxiv.org/abs/2305.09126](http://arxiv.org/abs/2305.09126)

    本文提出了一个名为$\ell_1$-TCL的通用框架，它使用知识迁移和Lasso回归来提高因果效应估计精度。

    

    本文研究了一种新颖的问题，即在相同的协变量（或特征）空间设置下通过知识迁移来提高因果效应估计精度，即同类别迁移学习（TL），将其称为转移因果学习（TCL）问题。我们提出了一个通用的框架$\ell_1$-TCL，其中包含$\ell_1$正则化TL来进行苦事参数估计和下游插件ACE估计器，包括结果回归、逆概率加权和双重稳健估计器。最重要的是，借助于Lasso用于高维回归，我们建立了非渐近恢复保证。

    A novel problem of improving causal effect estimation accuracy with the help of knowledge transfer under the same covariate (or feature) space setting, i.e., homogeneous transfer learning (TL), is studied, referred to as the Transfer Causal Learning (TCL) problem. While most recent efforts in adapting TL techniques to estimate average causal effect (ACE) have been focused on the heterogeneous covariate space setting, those methods are inadequate for tackling the TCL problem since their algorithm designs are based on the decomposition into shared and domain-specific covariate spaces. To address this issue, we propose a generic framework called \texttt{$\ell_1$-TCL}, which incorporates $\ell_1$ regularized TL for nuisance parameter estimation and downstream plug-in ACE estimators, including outcome regression, inverse probability weighted, and doubly robust estimators. Most importantly, with the help of Lasso for high-dimensional regression, we establish non-asymptotic recovery guarantee
    
[^27]: 对抗性线性上下文赌博的一阶和二阶界限

    First- and Second-Order Bounds for Adversarial Linear Contextual Bandits. (arXiv:2305.00832v1 [cs.LG])

    [http://arxiv.org/abs/2305.00832](http://arxiv.org/abs/2305.00832)

    本文研究了允许$k$个臂的损失函数随时间而自由变化的对抗性线性上下文赌博情境。在假设环境较为温和的情况下，我们获得了一个关于Learner's Losses $V_T$的二阶损失值量级为$\tilde O(K\sqrt{d V_T})$和关于最佳策略$L_T^*$的一阶损失值量级为$\tilde O(K\sqrt{d L_T^*})$的界。

    

    本文研究了对抗性线性上下文赌博的情境，该情境允许与K个臂相关联的损失函数随时间而自由变化。 假设d维上下文从已知分布中绘制，那么在T轮游戏期间最坏情况下的预期遗憾将以$\tilde O(\sqrt{Kd T})$的速度增长。在假设上下文的密度是对数凹的情况下，我们获得了一个二阶界，其在累积损失的二次矩$V_T$方面的量级为$\tilde O(K\sqrt{d V_T})$，以及一个与之密切相关的一阶界，其在最佳策略的累积损失$L_T^*$方面的量级为$\tilde O(K\sqrt{d L_T^*})$。由于$V_T$或$L_T^*$可能明显小于$T$，因此每当环境相对温和时，便会改善最坏情况的遗憾。本文使用概率单纯形上的连续指数权重算法的截断版本来获得结果

    We consider the adversarial linear contextual bandit setting, which allows for the loss functions associated with each of $K$ arms to change over time without restriction. Assuming the $d$-dimensional contexts are drawn from a fixed known distribution, the worst-case expected regret over the course of $T$ rounds is known to scale as $\tilde O(\sqrt{Kd T})$. Under the additional assumption that the density of the contexts is log-concave, we obtain a second-order bound of order $\tilde O(K\sqrt{d V_T})$ in terms of the cumulative second moment of the learner's losses $V_T$, and a closely related first-order bound of order $\tilde O(K\sqrt{d L_T^*})$ in terms of the cumulative loss of the best policy $L_T^*$. Since $V_T$ or $L_T^*$ may be significantly smaller than $T$, these improve over the worst-case regret whenever the environment is relatively benign. Our results are obtained using a truncated version of the continuous exponential weights algorithm over the probability simplex, which
    
[^28]: 用均场博弈为生成模型搭建实验室

    A mean-field games laboratory for generative modeling. (arXiv:2304.13534v1 [stat.ML])

    [http://arxiv.org/abs/2304.13534](http://arxiv.org/abs/2304.13534)

    本文提出了使用均场博弈作为实验室对生成模型进行设计和分析的方法，并建立了这种方法与主要流动和扩散型生成模型之间的关联。通过研究每个生成模型与它们相关的 MFG 的最优条件，本文提出了一个基于双人 MFG 的新的生成模型，该模型在提高样本多样性和逼真度的同时改善了解缠结和公平性。

    

    本文展示了均场博弈 (MFGs) 作为一种数学框架用于解释、增强和设计生成模型的多功能性。我们建立了 MFGs 与主要流动和扩散型生成模型之间关联，并通过不同的粒子动力学和代价函数推导了这三个类别的生成模型。此外，我们通过研究它们相关的 MFG 的最优条件——一组耦合的非线性偏微分方程，来研究每个生成模型的数学结构和特性。本文还提出了一个新的基于双人 MFG 的生成模型，其中一个代理合成样本，另一个代理对样本进行识别，理论和实验结果表明，该模型生成的样本多样且逼真，同时与基准模型相比，改善了解缠结和公平性。总之，本文突显了 MFGs 作为设计和分析生成模型的实验室的潜力。

    In this paper, we demonstrate the versatility of mean-field games (MFGs) as a mathematical framework for explaining, enhancing, and designing generative models. There is a pervasive sense in the generative modeling community that the various flow and diffusion-based generative models have some foundational common structure and interrelationships. We establish connections between MFGs and major classes of flow and diffusion-based generative models including continuous-time normalizing flows, score-based models, and Wasserstein gradient flows. We derive these three classes of generative models through different choices of particle dynamics and cost functions. Furthermore, we study the mathematical structure and properties of each generative model by studying their associated MFG's optimality condition, which is a set of coupled nonlinear partial differential equations (PDEs). The theory of MFGs, therefore, enables the study of generative models through the theory of nonlinear PDEs. Throu
    
[^29]: 我们实现了个性化治疗吗？使用重复采样的在线强化学习算法进行个性化评估

    Did we personalize? Assessing personalization by an online reinforcement learning algorithm using resampling. (arXiv:2304.05365v1 [cs.LG])

    [http://arxiv.org/abs/2304.05365](http://arxiv.org/abs/2304.05365)

    本论文提出了一种使用重复采样的政策评估方法，以评估在线 RL 算法实现的个性化程度。该方法可用于优化数字健康的个性化干预。

    

    在数字健康中，使用强化学习（RL）个性化治疗序列以支持用户采取更健康的行为越来越受到关注。这种连续决策问题涉及到基于用户的上下文（例如，先前的活动水平、位置等）在何时治疗以及如何治疗的决定。在线RL算法是这个问题的一个有前途的数据驱动方法，因为它基于每个用户的历史反馈进行学习，并利用这些知识个性化这些决策。然而，要决定是否应在实际部署的“优化”干预中包含RL算法，我们必须评估数据证据，表明RL算法实际上正在将治疗个性化适应其用户。由于RL算法中的随机性，人们可能会对其在某些状态下的学习并使用此学习来提供特定治疗的能力产生误解。我们使用工作定义的个性化，并介绍了一种重复采样政策评估方法来评估在线RL算法实现的个性化水平。我们使用模拟评估了我们提出的方法，并展示了我们的方法可以准确地识别个性化的策略。我们提出的方法在优化数字健康的个性化干预方面具有潜在应用。

    There is a growing interest in using reinforcement learning (RL) to personalize sequences of treatments in digital health to support users in adopting healthier behaviors. Such sequential decision-making problems involve decisions about when to treat and how to treat based on the user's context (e.g., prior activity level, location, etc.). Online RL is a promising data-driven approach for this problem as it learns based on each user's historical responses and uses that knowledge to personalize these decisions. However, to decide whether the RL algorithm should be included in an ``optimized'' intervention for real-world deployment, we must assess the data evidence indicating that the RL algorithm is actually personalizing the treatments to its users. Due to the stochasticity in the RL algorithm, one may get a false impression that it is learning in certain states and using this learning to provide specific treatments. We use a working definition of personalization and introduce a resamp
    
[^30]: 多个分位数的私有统计估计

    Private Statistical Estimation of Many Quantiles. (arXiv:2302.06943v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.06943](http://arxiv.org/abs/2302.06943)

    本文主要研究如何在差分隐私条件下估计一个分布的多个分位数。它提出了两种方法：一种是通过私有地估计样本的经验分位数来估计分布的分位数，另一种是使用密度估计技术进行分位数函数估计，并且展示了两种方法之间的权衡。

    

    本文研究在差分隐私条件下估计许多统计分位数的问题。更具体地，给定一个分布并且能够访问来自其独立同分布样本，我们考虑在特定点上估计其累积分布函数的逆函数（分位数函数）。例如，这项任务在私有数据生成中非常重要。我们提出了两种不同的方法。第一种方法是私下估计样本的经验分位数，并将此结果用作分布的分位数估计器。特别地，我们研究了 Kaplan等人最近发表的递归估计分位数的隐私算法的统计性质。第二种方法是使用密度估计技术进行均匀间隔内的分位数函数估计。特别地，我们展示了两种方法之间的权衡。当我们想要估计许多分位数时，最好使用第一种方法单独估计它们。另一方面，当我们想要在大区间上估计分位数函数时，第二种方法更有效。

    This work studies the estimation of many statistical quantiles under differential privacy. More precisely, given a distribution and access to i.i.d. samples from it, we study the estimation of the inverse of its cumulative distribution function (the quantile function) at specific points. For instance, this task is of key importance in private data generation. We present two different approaches. The first one consists in privately estimating the empirical quantiles of the samples and using this result as an estimator of the quantiles of the distribution. In particular, we study the statistical properties of the recently published algorithm introduced by Kaplan et al. 2022 that privately estimates the quantiles recursively. The second approach is to use techniques of density estimation in order to uniformly estimate the quantile function on an interval. In particular, we show that there is a tradeoff between the two methods. When we want to estimate many quantiles, it is better to estim
    
[^31]: OPORP：一次置换+一次随机投影

    OPORP: One Permutation + One Random Projection. (arXiv:2302.03505v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.03505](http://arxiv.org/abs/2302.03505)

    OPORP使用一种"计数草图"类型的数据降维/压缩方法，可以用于嵌入式检索，在保证较少的信息损失的前提下，显著降低了计算和存储的成本

    

    考虑两个$D$维数据向量（例如嵌入）：$u, v$。在许多基于嵌入的检索（EBR）应用程序中，$D=256\sim 1024$很常见。在本文中，OPORP（一次置换+一次随机投影）使用一种“计数草图”类型的数据结构变体进行数据降维/压缩。使用OPORP，我们首先对数据向量进行置换。生成随机向量$r$，i.i.d. ，满足：$E（r_i）=0，E（r_i^2）=1，E（r_i^3）=0，E（r_i^4）=s$。我们将$r$与所有置换数据向量相乘（作为点积）。然后，我们将$D$列分成$k$个相等长度的箱（bin），并汇总（即求和）每个箱中的值以从每个数据向量中获取$k$个样本。一个关键的步骤是将$k$个样本标准化为单位$l_2$范数。我们表明，估计方差本质上是：$(s-1)A + \frac{D-k}{D-1}\frac{1}{k}\left[ (1-\rho^2)^2 -2A\right]$，其中$A\geq 0$是数据（$u,v$）的函数

    Consider two $D$-dimensional data vectors (e.g., embeddings): $u, v$. In many embedding-based retrieval (EBR) applications where the vectors are generated from trained models, $D=256\sim 1024$ are common. In this paper, OPORP (one permutation + one random projection) uses a variant of the ``count-sketch'' type of data structures for achieving data reduction/compression. With OPORP, we first apply a permutation on the data vectors. A random vector $r$ is generated i.i.d. with moments: $E(r_i) = 0, E(r_i^2)=1, E(r_i^3) =0, E(r_i^4)=s$. We multiply (as dot product) $r$ with all permuted data vectors. Then we break the $D$ columns into $k$ equal-length bins and aggregate (i.e., sum) the values in each bin to obtain $k$ samples from each data vector. One crucial step is to normalize the $k$ samples to the unit $l_2$ norm. We show that the estimation variance is essentially: $(s-1)A + \frac{D-k}{D-1}\frac{1}{k}\left[ (1-\rho^2)^2 -2A\right]$, where $A\geq 0$ is a function of the data ($u,v$)
    
[^32]: 应用于蛋白质主链生成的SE（3）扩散模型

    SE(3) diffusion model with application to protein backbone generation. (arXiv:2302.02277v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02277](http://arxiv.org/abs/2302.02277)

    本文提出了SE（3）扩散模型及其理论基础，并使用FrameDiff框架在多个框架上学习SE（3）等变分数，成功生成可设计的长达500个氨基酸的单体背景。

    

    设计新型蛋白质结构仍然是生物医学和化学领域中的一项挑战。在这方面的工作中，一个三维刚性体（称为框架）上的扩散模型已经成功地生成了在自然界中没有观察到的新型功能蛋白主链。然而，在3D空间中的方向保持刚性运动的SE（3）扩散上缺乏明确的方法论框架，该框架在框架操作中保持群不变性。我们通过开发多个框架上SE（3）不变扩散模型的理论基础来解决这些缺点，然后提出了一种新的框架，FrameDiff，来学习多个框架上SE（3）等变分数。我们在单体背景生成上应用FrameDiff，并发现它可以生成可设计的单体背景，长达500个氨基酸，而不依赖于之前方法中必要的预训练蛋白质结构预测网络。我们发现我们的sa

    The design of novel protein structures remains a challenge in protein engineering for applications across biomedicine and chemistry. In this line of work, a diffusion model over rigid bodies in 3D (referred to as frames) has shown success in generating novel, functional protein backbones that have not been observed in nature. However, there exists no principled methodological framework for diffusion on SE(3), the space of orientation preserving rigid motions in R3, that operates on frames and confers the group invariance. We address these shortcomings by developing theoretical foundations of SE(3) invariant diffusion models on multiple frames followed by a novel framework, FrameDiff, for learning the SE(3) equivariant score over multiple frames. We apply FrameDiff on monomer backbone generation and find it can generate designable monomers up to 500 amino acids without relying on a pretrained protein structure prediction network that has been integral to previous methods. We find our sa
    
[^33]: 基于群对称性的概率差异的样本复杂度分析

    Sample Complexity of Probability Divergences under Group Symmetry. (arXiv:2302.01915v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2302.01915](http://arxiv.org/abs/2302.01915)

    本文研究了具有群不变性的分布变量在变分差异估计中的样本复杂度，发现在群大小维度相关的情况下，样本复杂度会有所降低，并在实验中得到了验证。

    

    我们对于具有群不变性的分布变量在变分差异估计中的样本复杂度进行了严谨的量化分析。在Wasserstein-1距离和Lipschitz正则化α差异的情况下，样本复杂度的降低与群大小的维度相关。对于最大均值差异（MMD），样本复杂度的改进更加复杂，因为它不仅取决于群大小，还取决于内核的选择。 数值模拟验证了我们的理论。

    We rigorously quantify the improvement in the sample complexity of variational divergence estimations for group-invariant distributions. In the cases of the Wasserstein-1 metric and the Lipschitz-regularized $\alpha$-divergences, the reduction of sample complexity is proportional to an ambient-dimension-dependent power of the group size. For the maximum mean discrepancy (MMD), the improvement of sample complexity is more nuanced, as it depends on not only the group size but also the choice of kernel. Numerical simulations verify our theories.
    
[^34]: 核斯坦距离稀释：关于病态的理论视角和正则化的实际修复（arXiv:2301.13528v2 [math.ST] 已更新）

    Kernel Stein Discrepancy thinning: a theoretical perspective of pathologies and a practical fix with regularization. (arXiv:2301.13528v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2301.13528](http://arxiv.org/abs/2301.13528)

    本文针对后处理MCMC输出过程中的病态问题进行了理论分析，提出了正则化的Stein稀释算法，有效缓解了这些病态产生的影响，为提高逼近质量提供了新策略。

    

    Stein稀释是一种有前途的算法，由（Riabiz et al.，2022）提出用于后处理马尔可夫链蒙特卡罗（MCMC）的输出。其主要原则是贪婪地最小化核化Stein差异（KSD），它仅需要对数目标分布的梯度，因此非常适合贝叶斯推断。 Stein稀释的主要优势是自动消除启动期，纠正最近的MCMC算法引入的偏差，并具有收敛至目标分布的渐近特性。然而， Stein稀释存在几个经验病态，可能导致劣质逼近，这在文献中已被观察到。 在本文中，我们对这些病态进行了理论分析，以明确识别涉及的机制，并提出了改进的策略。然后，我们引入了正则化的 Stein稀释算法来缓解已识别的病态。最后，我们提供了理论保证和扩展。

    Stein thinning is a promising algorithm proposed by (Riabiz et al., 2022) for post-processing outputs of Markov chain Monte Carlo (MCMC). The main principle is to greedily minimize the kernelized Stein discrepancy (KSD), which only requires the gradient of the log-target distribution, and is thus well-suited for Bayesian inference. The main advantages of Stein thinning are the automatic remove of the burn-in period, the correction of the bias introduced by recent MCMC algorithms, and the asymptotic properties of convergence towards the target distribution. Nevertheless, Stein thinning suffers from several empirical pathologies, which may result in poor approximations, as observed in the literature. In this article, we conduct a theoretical analysis of these pathologies, to clearly identify the mechanisms at stake, and suggest improved strategies. Then, we introduce the regularized Stein thinning algorithm to alleviate the identified pathologies. Finally, theoretical guarantees and exte
    
[^35]: 多阶段静态治疗策略的高维特征渐近推断

    Asymptotic Inference for Multi-Stage Stationary Treatment Policy with High Dimensional Features. (arXiv:2301.12553v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.12553](http://arxiv.org/abs/2301.12553)

    本研究填补了在高维特征变量存在的情况下，对于多阶段静态治疗策略本身进行推断的工作空白，提出了一种增强的估计器以提高价值函数的准确性。

    

    动态治疗规则是一系列针对个体特征量身定制的多阶段决策函数。在实践中，一类重要的治疗策略是多阶段静态治疗策略，其使用相同的决策函数来指定治疗分配概率，在决策时基于同时包括基线变量（例如人口统计学）和时变变量（例如常规检测到的疾病生物标志物）的一组特征。虽然已经有大量文献对与动态治疗策略相关的价值函数进行有效推断，但在高维特征变量存在的情况下，对于治疗策略本身的工作还很少。我们旨在填补这项工作的空白。具体而言，我们首先基于增强的倒数权重估计器估计多阶段静态治疗策略，以提高价值函数的准确性。

    Dynamic treatment rules or policies are a sequence of decision functions over multiple stages that are tailored to individual features. One important class of treatment policies for practice, namely multi-stage stationary treatment policies, prescribe treatment assignment probabilities using the same decision function over stages, where the decision is based on the same set of features consisting of both baseline variables (e.g., demographics) and time-evolving variables (e.g., routinely collected disease biomarkers). Although there has been extensive literature to construct valid inference for the value function associated with the dynamic treatment policies, little work has been done for the policies themselves, especially in the presence of high dimensional feature variables. We aim to fill in the gap in this work. Specifically, we first estimate the multistage stationary treatment policy based on an augmented inverse probability weighted estimator for the value function to increase
    
[^36]: 基于抽样的Nyström逼近和核积分。

    Sampling-based Nystr\"om Approximation and Kernel Quadrature. (arXiv:2301.09517v2 [math.NA] UPDATED)

    [http://arxiv.org/abs/2301.09517](http://arxiv.org/abs/2301.09517)

    本文提出了一种基于抽样的Nyström逼近方法用于核积分。同时，引入了一种非i.i.d.地标点的理论保证方法，使得提高了逼近的精度。

    

    我们分析与概率测量相关的正定核的Nyström逼近。我们首先证明了传统Nyström逼近在连续区间中使用i.i.d.抽样和奇异值分解的改进误差界，证明技巧借鉴了统计学习理论。我们进一步引入了Nyström逼近中的子空间精细选择，这是适用于非i.i.d.地标点的理论保证。最后，我们讨论了它们在凸核积分中的应用，并给出了新的理论保证以及数值观察。

    We analyze the Nystr\"om approximation of a positive definite kernel associated with a probability measure. We first prove an improved error bound for the conventional Nystr\"om approximation with i.i.d. sampling and singular-value decomposition in the continuous regime; the proof techniques are borrowed from statistical learning theory. We further introduce a refined selection of subspaces in Nystr\"om approximation with theoretical guarantees that is applicable to non-i.i.d. landmark points. Finally, we discuss their application to convex kernel quadrature and give novel theoretical guarantees as well as numerical observations.
    
[^37]: 采样的出生死亡动态：全局收敛，逼近及其渐近性质研究

    Birth-death dynamics for sampling: Global convergence, approximations and their asymptotics. (arXiv:2211.00450v2 [math.AP] UPDATED)

    [http://arxiv.org/abs/2211.00450](http://arxiv.org/abs/2211.00450)

    本文研究了一种连续的出生死亡动态，并提出了弱假设。通过这种动态控制的概率密度指数级地快速收敛到吉布斯平衡测度，同时提出了一种实用的基于纯出生死亡动态的数值采样器，并对其逼近品质进行了定量评估。

    

    本文以采样非凸位势吉布斯测度为挑战，研究了一种连续出生死亡动态。我们提出了一种弱假设，改进了先前[51,57]的结果，证明了由Kullback-Leibler散度或$\chi^2$散度控制的出生死亡概率密度会指数级快速地收敛到吉布斯平衡测度，其普适速率独立于势垒。为了构建基于纯出生死亡动态的实用数值采样器，我们考虑了一个交互粒子系统，它灵感来自于梯度流结构和经典的Fokker-Planck方程，并依赖于测量的基于核的逼近。通过梯度流的$\Gamma$-收敛技术，证明在环上，核化动态的光滑有界正解在有限时间间隔内，当核带宽收缩到零时，收敛于纯出生死亡动态。此外，我们使用了伽马收敛的技术对纯出生死亡过程的逼近品质进行了定量评估。

    Motivated by the challenge of sampling Gibbs measures with nonconvex potentials, we study a continuum birth-death dynamics. We improve results in previous works [51,57] and provide weaker hypotheses under which the probability density of the birth-death governed by Kullback-Leibler divergence or by $\chi^2$ divergence converge exponentially fast to the Gibbs equilibrium measure, with a universal rate that is independent of the potential barrier. To build a practical numerical sampler based on the pure birth-death dynamics, we consider an interacting particle system, which is inspired by the gradient flow structure and the classical Fokker-Planck equation and relies on kernel-based approximations of the measure. Using the technique of $\Gamma$-convergence of gradient flows, we show that on the torus, smooth and bounded positive solutions of the kernelized dynamics converge on finite time intervals, to the pure birth-death dynamics as the kernel bandwidth shrinks to zero. Moreover we pro
    
[^38]: 具有无限制内存的在线凸优化

    Online Convex Optimization with Unbounded Memory. (arXiv:2210.09903v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.09903](http://arxiv.org/abs/2210.09903)

    本论文提出了一种新的在线凸优化框架，可以处理决策历史的长期依赖关系，并介绍了用于量化依赖程度的$p$-有效内存容量的概念。

    

    在线凸优化（OCO）是在线学习中广泛使用的框架。然而，在很多应用中，学习者的损失不仅取决于当前的决策，还取决于直到那个时间点的所有决策历史。本文引入了一种OCO的扩展框架，“具有无限制内存的在线凸优化”，来捕捉对过去决策的长期依赖关系，并介绍了$p$-有效内存容量的概念，$H_p$，它量化了$p$阶影响的最大值。

    Online convex optimization (OCO) is a widely used framework in online learning. In each round, the learner chooses a decision in a convex set and an adversary chooses a convex loss function, and then the learner suffers the loss associated with their current decision. However, in many applications the learner's loss depends not only on the current decision but on the entire history of decisions until that point. The OCO framework and its existing generalizations do not capture this, and they can only be applied to many settings of interest after a long series of approximation arguments. They also leave open the question of whether the dependence on memory is tight because there are no non-trivial lower bounds. In this work we introduce a generalization of the OCO framework, ``Online Convex Optimization with Unbounded Memory'', that captures long-term dependence on past decisions. We introduce the notion of $p$-effective memory capacity, $H_p$, that quantifies the maximum influence of p
    
[^39]: 基于连续时间霍克过程的败血症相关异常的Granger因果链发现

    Granger Causal Chain Discovery for Sepsis-Associated Derangements via Continuous-Time Hawkes Processes. (arXiv:2209.04480v5 [stat.AP] UPDATED)

    [http://arxiv.org/abs/2209.04480](http://arxiv.org/abs/2209.04480)

    本文提出了一个基于连续时间霍克过程的Granger因果链发现方法，可用于推断EMR数据中多个患者特征之间的时间交互作用，并确定败血症相关异常的实验室值链。

    

    现代医疗保健系统正在增加对电子病历（EMR）进行连续自动监测，以更频繁地识别不良事件; 但是，许多事件（如败血症）没有已阐明的前驱症状（即事件链），可以用于在其病程早期识别和截获不良事件。临床相关且可解释的结果需要一个框架，该框架可以（i）推断EMR数据中多个患者特征之间的时间交互作用（例如实验室检查、生命体征等），并且可以（ii）确定在即将发生的不良事件（例如败血症）之前先导且特定的模式。在这项工作中，我们提出了一个线性多元霍克过程模型，结合ReLU链接函数，以恢复具有兴奋和抑制效应的Granger因果（GC）图。我们开发了一个可扩展的两阶段梯度下降方法，以获得最大似然估计值，通过广泛的数值模拟证明了其有效性。我们的方法在大型EMR数据集上进行了验证，并显示败血症相关实验室异常的可解释的时间模式。这些模式建议了一条实验室价值观的链，可以帮助早期检测和管理败血症相关异常。

    Modern health care systems are conducting continuous, automated surveillance of the electronic medical record (EMR) to identify adverse events with increasing frequency; however, many events such as sepsis do not have elucidated prodromes (i.e., event chains) that can be used to identify and intercept the adverse event early in its course. Clinically relevant and interpretable results require a framework that can (i) infer temporal interactions across multiple patient features found in EMR data (e.g., Labs, vital signs, etc.) and (ii) identify patterns that precede and are specific to an impending adverse event (e.g., sepsis). In this work, we propose a linear multivariate Hawkes process model, coupled with ReLU link function, to recover a Granger Causal (GC) graph with both exciting and inhibiting effects. We develop a scalable two-phase gradient-based method to obtain a maximum surrogate-likelihood estimator, which is shown to be effective via extensive numerical simulation. Our meth
    
[^40]: 基于二阶相似性的更快联邦优化

    Faster federated optimization under second-order similarity. (arXiv:2209.02257v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.02257](http://arxiv.org/abs/2209.02257)

    提出两种新的联邦学习算法，SVRP 和 Catalyzed SVRP，它们都有较高的通信效率和性能表现，并广泛适用于分布式统计学习和差分隐私经验风险最小化等领域。

    

    联邦学习是机器学习的一个分支，在通信约束下，多个客户端尝试在网络上协作学习模型。我们考虑在二阶函数相似性条件和强凸性下的有限和联邦优化，并提出了两种新算法：SVRP 和催化 SVRP。近年来，二阶相似性条件已经变得流行起来，并在许多应用中得到满足，包括分布式统计学习和差分隐私经验风险最小化。第一个算法 SVRP 组合了近似随机近端点评估、客户端抽样和方差缩减。我们证明了 SVRP 具有通信效率，并且在函数相似性足够高的情况下，可以获得优越的性能，优于许多现有算法。我们的第二个算法，Catalyzed SVRP 是 SVRP 的催化剂加速变体，可以实现更好的性能，并统一改进现有联邦学习算法。

    Federated learning (FL) is a subfield of machine learning where multiple clients try to collaboratively learn a model over a network under communication constraints. We consider finite-sum federated optimization under a second-order function similarity condition and strong convexity, and propose two new algorithms: SVRP and Catalyzed SVRP. This second-order similarity condition has grown popular recently, and is satisfied in many applications including distributed statistical learning and differentially private empirical risk minimization. The first algorithm, SVRP, combines approximate stochastic proximal point evaluations, client sampling, and variance reduction. We show that SVRP is communication efficient and achieves superior performance to many existing algorithms when function similarity is high enough. Our second algorithm, Catalyzed SVRP, is a Catalyst-accelerated variant of SVRP that achieves even better performance and uniformly improves upon existing algorithms for federate
    
[^41]: 掩码贝叶斯神经网络: 计算与优越性

    Masked Bayesian Neural Networks : Computation and Optimality. (arXiv:2206.00853v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2206.00853](http://arxiv.org/abs/2206.00853)

    本文提出了一种新颖的稀疏贝叶斯神经网络（BNN），它可以使用掩码变量在节点级别上关闭一些节点，以产生稀疏的DNN结构。我们还设计了一个先验分布，使得后验分布具有理论上的最优性，并开发了一种高效的MCMC算法。该方法在几个基准数据集上表现良好，能够发现精简的DNN结构，具有与大型DNN相似的预测准确性和不确定性量化能力。

    

    随着数据量和计算能力的增长，深度神经网络（DNN）的架构变得越来越复杂和庞大，因此需要简化这种复杂和庞大的DNN。在本文中，我们提出了一种新颖的稀疏贝叶斯神经网络（BNN），该网络可以找到一个适当复杂度的 DNN。我们在每个节点上使用掩码变量，根据后验分布关闭一些节点，以产生稀疏 DNN。我们设计了一个先验分布，使得后验分布具有理论上的最优性（即极小极大优越性和自适应性），并开发了一种高效的MCMC算法。通过分析几个基准数据集，我们证明所提出的BNN表现良好，与大型DNN相比，它发现了精简的DNN结构，具有相似的预测准确性和不确定性量化。

    As data size and computing power increase, the architectures of deep neural networks (DNNs) have been getting more complex and huge, and thus there is a growing need to simplify such complex and huge DNNs. In this paper, we propose a novel sparse Bayesian neural network (BNN) which searches a good DNN with an appropriate complexity. We employ the masking variables at each node which can turn off some nodes according to the posterior distribution to yield a nodewise sparse DNN. We devise a prior distribution such that the posterior distribution has theoretical optimalities (i.e. minimax optimality and adaptiveness), and develop an efficient MCMC algorithm. By analyzing several benchmark datasets, we illustrate that the proposed BNN performs well compared to other existing methods in the sense that it discovers well condensed DNN architectures with similar prediction accuracy and uncertainty quantification compared to large DNNs.
    
[^42]: SVM指数级收敛速度的案例

    A Case of Exponential Convergence Rates for SVM. (arXiv:2205.10055v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2205.10055](http://arxiv.org/abs/2205.10055)

    本文研究了SVM的指数级收敛速度，提出了一种简单的方法来获得快速收敛速度，并在没有假设硬Tsybakov边际条件的情况下展示了SVM的指数级收敛速度现象。

    

    分类问题通常是介绍机器学习课程中描述的第一个问题。历史上，瓦普尼克-切尔沃年科理论提供了分类的泛化保证。然而，这些保证基于难以处理的算法，这导致了分类中代理方法的理论。代理方法提供的保证基于校准不等式，已被证明在某些边际条件下非常次优，不能捕捉到指数级收敛现象。这些"超"快速率现在已经对于光滑的代理得到了很好的理解，但对于与著名的支持向量机相关的非光滑损失（如铰链损失），画面仍然模糊不清。本文介绍了一种简单的机制来获得快速收敛速度，并研究其用于SVM的情况。特别地，我们展示了SVM可以展现出指数级的收敛速度，即使没有假设硬Tsybakov边际条件。

    Classification is often the first problem described in introductory machine learning classes. Generalization guarantees of classification have historically been offered by Vapnik-Chervonenkis theory. Yet those guarantees are based on intractable algorithms, which has led to the theory of surrogate methods in classification. Guarantees offered by surrogate methods are based on calibration inequalities, which have been shown to be highly sub-optimal under some margin conditions, failing short to capture exponential convergence phenomena. Those "super" fast rates are becoming to be well understood for smooth surrogates, but the picture remains blurry for non-smooth losses such as the hinge loss, associated with the renowned support vector machines. In this paper, we present a simple mechanism to obtain fast convergence rates and we investigate its usage for SVM. In particular, we show that SVM can exhibit exponential convergence rates even without assuming the hard Tsybakov margin conditi
    
[^43]: 外部有效的策略选择

    Externally Valid Policy Choice. (arXiv:2205.05561v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2205.05561](http://arxiv.org/abs/2205.05561)

    本文研究了外部有效的个性化治疗策略的学习问题。研究表明，最大化福利的治疗策略对于实验和目标人群之间的结果分布变化具有鲁棒性。通过开发新的方法，作者提出了对结果和特征变化具有鲁棒性的策略学习方法，并强调实验人群内的治疗效果异质性对策略普适性的影响。

    

    我们考虑学习个性化治疗策略的问题，这些策略是外部有效或广义化的：它们在除了实验（或训练）人群外的其他目标人群中表现良好。我们首先证明，对于实验人群而言，最大化福利的策略对于实验和目标人群之间的结果（但不是特征）分布变化具有鲁棒性。然后，我们开发了新的方法来学习对结果和特征变化具有鲁棒性的策略。在这样做时，我们强调了实验人群内的治疗效果异质性如何影响策略的普适性。我们的方法可以使用实验或观察数据（其中治疗是内生的）。我们的许多方法可以使用线性规划实现。

    We consider the problem of learning personalized treatment policies that are externally valid or generalizable: they perform well in other target populations besides the experimental (or training) population from which data are sampled. We first show that welfare-maximizing policies for the experimental population are robust to shifts in the distribution of outcomes (but not characteristics) between the experimental and target populations. We then develop new methods for learning policies that are robust to shifts in outcomes and characteristics. In doing so, we highlight how treatment effect heterogeneity within the experimental population affects the generalizability of policies. Our methods may be used with experimental or observational data (where treatment is endogenous). Many of our methods can be implemented with linear programming.
    

