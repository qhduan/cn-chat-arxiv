# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improved sampling via learned diffusions.](http://arxiv.org/abs/2307.01198) | 通过学习扩散的方法改进了采样过程，引入了基于变分形式的路径空间度量，提出了对数方差损失，优化了采样性能。 |
| [^2] | [Fitting an ellipsoid to a quadratic number of random points.](http://arxiv.org/abs/2307.01181) | 将$n$个高斯随机向量拟合到以原点为中心的椭球体边界的问题$(\mathrm{P})$，我们提出了一个基于随机向量Gram矩阵集中性的改进方法，证明了当$n \leq d^2 / C$时，问题$(\mathrm{P})$具有很高的可行性概率。 |
| [^3] | [Learning Mixtures of Gaussians Using the DDPM Objective.](http://arxiv.org/abs/2307.01178) | 本文针对高斯混合模型这一基础分布族提供了首个可证明高效的结果，通过梯度下降对去噪扩散概率模型（DDPM）目标进行训练可以有效地恢复混合模型的参数。 |
| [^4] | [Neural Hilbert Ladders: Multi-Layer Neural Networks in Function Space.](http://arxiv.org/abs/2307.01177) | 本文提出了神经希尔伯特阶梯(NHL)的概念，它将多层神经网络描述为一系列的再生核希尔伯特空间，进一步推广了浅层神经网络的理论研究，并探讨了其在函数空间内的性质和应用。通过证明不同层次的NHL与多层NNs之间的对应关系，证明了学习NHL的泛化保证，并提出了NHL的特征动力学模型。最后，在ReLU和二次激活函数下展示了NHLs中的深度分离现象。 |
| [^5] | [Analyzing and Improving Greedy 2-Coordinate Updates for Equality-Constrained Optimization via Steepest Descent in the 1-Norm.](http://arxiv.org/abs/2307.01169) | 通过最速下降法，我们在等式约束优化问题中探索并改进了贪婪的二维坐标更新方法，在满足特定条件下取得了更快的收敛速度。此外，我们还将该方法推广到同时具有求和约束和边界约束的问题，并证明了在L1-范数下的最速下降法可以在更短的计算时间内取得更多的进展。 |
| [^6] | [Empirically Validating Conformal Prediction on Modern Vision Architectures Under Distribution Shift and Long-tailed Data.](http://arxiv.org/abs/2307.01088) | 本文在大规模数据集和模型上首次对分布转移和长尾类别分布下的合规预测方法进行了实证评估。研究发现，这些方法在分布转移和长尾设置下的性能大大下降，对于在现实世界和安全关键应用中的部署具有重要的局限性。 |
| [^7] | [Some challenges of calibrating differentiable agent-based models.](http://arxiv.org/abs/2307.01085) | 本文讨论了校准可微分的基于Agent的模型面临的挑战，同时提出了潜在的解决方案。 |
| [^8] | [Supervised Manifold Learning via Random Forest Geometry-Preserving Proximities.](http://arxiv.org/abs/2307.01077) | 本文通过使用随机森林近似的几何保持特性作为流形学习方法的初始化，展示了类条件流形学习的局限性，并提出了一种替代选择。这种方法能够在几乎所有流形学习方法中保持局部结构，并正确地维护全局结构。 |
| [^9] | [Transport, Variational Inference and Diffusions: with Applications to Annealed Flows and Schr\"odinger Bridges.](http://arxiv.org/abs/2307.01050) | 本文研究了最优运输和变分推断之间的联系，并提出了一种基于路径空间散度的采样和生成建模框架。通过开发新颖的基于得分的回火流技术和正则化的迭代比例拟合目标，本文展示了这些方法的潜力。 |
| [^10] | [Doubly Robust Estimation of Direct and Indirect Quantile Treatment Effects with Machine Learning.](http://arxiv.org/abs/2307.01049) | 该论文提出了一种双重稳健估计方法，用于估计机器学习下的直接和间接分位治疗效应，通过机器学习和交叉拟合来处理可观测选择偏差，并提出了乘法自助法进行统计推断。 |
| [^11] | [Quantum Machine Learning on Near-Term Quantum Devices: Current State of Supervised and Unsupervised Techniques for Real-World Applications.](http://arxiv.org/abs/2307.00908) | 近期量子设备上的量子机器学习应用中，我们着重研究了监督和无监督学习在现实世界场景的应用。我们探究了当前量子硬件上的QML实现的限制，并提出了克服这些限制的技术。与经典对应物相比较，这些QML实现的性能得到了评估。 |
| [^12] | [MADS: Modulated Auto-Decoding SIREN for time series imputation.](http://arxiv.org/abs/2307.00868) | 本论文提出了一种新的自解码框架MADS，用于时间序列插补。该方法基于隐式神经表示，利用SIREN的能力进行高保真重建，并采用超网络架构进行泛化。实验证明该模型在两个真实数据集上的表现优于现有最先进的方法。 |
| [^13] | [CardiGraphormer: Unveiling the Power of Self-Supervised Learning in Revolutionizing Drug Discovery.](http://arxiv.org/abs/2307.00859) | CardiGraphormer是一种革命性的方法，结合了自监督学习、图神经网络和保持基数注意力，颠覆了药物发现的方式。它利用自监督学习学习分子表示并利用图神经网络提取分子指纹，提高了预测性能和可解释性，同时减少了计算时间，并在处理复杂数据和执行各种与图结构相关的任务方面表现出色。 |
| [^14] | [Trading-Off Payments and Accuracy in Online Classification with Paid Stochastic Experts.](http://arxiv.org/abs/2307.00836) | 该研究探索了在线分类中使用付费随机专家的方法，通过权衡支付金额和准确性来进行预测。研究提出了一种在线学习算法，其总成本不超过预先知道所有专家生产力的预测算法成本的函数，通过结合Lipschitz Bandits和基于替代损失的在线分类，我们改进了现有的界限。 |
| [^15] | [Engression: Extrapolation for Nonlinear Regression?.](http://arxiv.org/abs/2307.00835) | Engression是一种非线性回归方法，通过使用分布回归技术和预加性噪声模型，在训练样本范围边界外也能可靠地进行外推。 |
| [^16] | [Worth of knowledge in deep learning.](http://arxiv.org/abs/2307.00712) | 该论文提出了一个模型-不可知框架，通过定量实验评估了数据量和估计范围对知识的价值的影响，阐明了数据和知识之间的复杂关系，并可应用于不同网络架构，提供对深度学习模型中先前知识作用的全面理解。 |
| [^17] | [Morse Neural Networks for Uncertainty Quantification.](http://arxiv.org/abs/2307.00667) | Morse神经网络是一种用于不确定性量化的深度生成模型，通过拟合KL散度损失，可以得到生成密度、OOD检测器、校准温度、生成采样器和距离感知分类器。它统一了多种技术应用，如OOD检测、异常检测和连续学习等。 |
| [^18] | [Variational Autoencoding Molecular Graphs with Denoising Diffusion Probabilistic Model.](http://arxiv.org/abs/2307.00623) | 这篇论文提出了一种新颖的分子深度生成模型，将分层结构融入概率潜在向量中，并通过去噪扩散概率模型来设计有效的分子潜在向量，用于分子性质预测。 |
| [^19] | [Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models.](http://arxiv.org/abs/2307.00619) | 该论文提出了一种新框架，通过利用预训练的潜在扩散模型来解决线性逆问题。理论分析证明了算法的可靠性，并且在多种问题上实验证明了其优越性能。 |
| [^20] | [Optimizing protein fitness using Gibbs sampling with Graph-based Smoothing.](http://arxiv.org/abs/2307.00494) | 使用基于图形平滑的Gibbs采样方法（GGS）优化蛋白质适应性，消除了突变距离的限制，同时提高了搜索效率。该方法在发现高适应性蛋白质方面达到了最先进水平。 |
| [^21] | [MissDiff: Training Diffusion Models on Tabular Data with Missing Values.](http://arxiv.org/abs/2307.00467) | 本研究提出了一个基于扩散的框架，用于处理带缺失值的表格数据，并证明了所提出的训练目标的一致性和上界性能。 |
| [^22] | [Adaptive Algorithms for Relaxed Pareto Set Identification.](http://arxiv.org/abs/2307.00424) | 本研究提出了一种自适应算法，用于宽松Pareto集的识别，通过放松策略来减少样本复杂度，并展示了在实际场景中的良好表现。 |
| [^23] | [Provably Efficient UCB-type Algorithms For Learning Predictive State Representations.](http://arxiv.org/abs/2307.00405) | 这篇论文提出了第一种已知的UCB类型方法用于学习预测状态表示（PSRs），并设计了一个新的奖励项来上界t |
| [^24] | [Sparse-Input Neural Network using Group Concave Regularization.](http://arxiv.org/abs/2307.00344) | 本文提出了一种使用组凹正则化进行特征选择的稀疏输入神经网络框架，该框架能够在高维环境中选择重要的特征并保持稳定的解。 |
| [^25] | [Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD.](http://arxiv.org/abs/2307.00310) | 本文开发了一种新的DP-SGD分析方法，可以在训练过程中对许多数据点的隐私泄漏进行更准确的评估。 |
| [^26] | [Applied Bayesian Structural Health Monitoring: inclinometer data anomaly detection and forecasting.](http://arxiv.org/abs/2307.00305) | 本文介绍了将贝叶斯技术应用于测斜仪数据的异常检测和预测，并且展示了如何通过量化和评估不确定性来最小化成本和风险。 |
| [^27] | [Bootstrapping the Cross-Validation Estimate.](http://arxiv.org/abs/2307.00260) | 本文提出了一种快速自助法，可以快速估计交叉验证估计的标准误差，并为衡量平均模型性能的总体参数产生有效的置信区间。 |
| [^28] | [Unified Transfer Learning Models for High-Dimensional Linear Regression.](http://arxiv.org/abs/2307.00238) | UTrans是一种统一转移学习模型，它能检测可转移变量和源数据，并具有较低的估计和预测误差，同时保持可解释性。 |
| [^29] | [Causal Structure Learning by Using Intersection of Markov Blankets.](http://arxiv.org/abs/2307.00227) | 本文提出了一种新颖的因果结构学习算法，该算法利用马尔可夫毯交集，并结合了贝叶斯网络和结构因果模型的特性。此外，还提出了EEMBI-PC，它是EEMBI的扩展版本，将PC算法的最后一步集成到EEMBI中。 |
| [^30] | [The Effect of Balancing Methods on Model Behavior in Imbalanced Classification Problems.](http://arxiv.org/abs/2307.00157) | 平衡方法对不平衡分类问题中模型行为产生显著影响。这些发现强调了平衡分析在模型训练中的重要性。 |
| [^31] | [High-Dimensional Bayesian Structure Learning in Gaussian Graphical Models using Marginal Pseudo-Likelihood.](http://arxiv.org/abs/2307.00127) | 该论文提出了两种创新的搜索算法，在高维图结构学习中使用边际伪似然函数解决计算复杂性问题，并且能够在短时间内生成可靠的估计。该方法提供了R软件包BDgraph的代码实现。 |
| [^32] | [Accelerating Inexact HyperGradient Descent for Bilevel Optimization.](http://arxiv.org/abs/2307.00126) | 提出一种加速非精确超梯度下降的方法用于双层优化，可以在较低的复杂度下找到一阶和二阶稳定点，成为双层优化和凸-凹极小极大优化问题中的最新最佳状态。 |
| [^33] | [Proximal nested sampling with data-driven priors for physical scientists.](http://arxiv.org/abs/2307.00056) | 近端嵌套抽样方法允许物理科学家应用贝叶斯模型选择于高维问题中，并展示了如何通过数据驱动先验的支持来扩展该方法。 |
| [^34] | [Learned harmonic mean estimation of the marginal likelihood with normalizing flows.](http://arxiv.org/abs/2307.00048) | 本文研究使用归一化流学习边缘似然的调和平均估计，在贝叶斯模型选择中解决了原始方法中的方差爆炸问题。 |
| [^35] | [TemperatureGAN: Generative Modeling of Regional Atmospheric Temperatures.](http://arxiv.org/abs/2306.17248) | TemperatureGAN是一个生成对抗网络，使用地面以上2m的大气温度数据，能够生成具有良好空间表示和与昼夜周期一致的时间动态的高保真样本。 |
| [^36] | [Near Optimal Heteroscedastic Regression with Symbiotic Learning.](http://arxiv.org/abs/2306.14288) | 本研究提出了一种基于共生学习的异方差回归的近似最优算法，可以在统计学、计量经济学、时间序列分析等领域，以及在不同来源数据质量不一的机器学习中应用。 |
| [^37] | [Lower Complexity Adaptation for Empirical Entropic Optimal Transport.](http://arxiv.org/abs/2306.13580) | 本文研究了经验熵正则化最优输运的统计表现，并证明了它遵循低复杂度适应原则，推导出了其统计界限及参数化速率。 |
| [^38] | [Online Heavy-tailed Change-point detection.](http://arxiv.org/abs/2306.09548) | 本文提出了一种在线变点检测算法，可以应对重尾分布且保证有限的假阳性率。 |
| [^39] | [Differentiating Metropolis-Hastings to Optimize Intractable Densities.](http://arxiv.org/abs/2306.07961) | 本文通过基于互联马尔科夫链的不偏微分，开发出一种无偏、低方差和自动的方法对复杂密度进行生成，从而实现对 MH 采样器的优化。 |
| [^40] | [Label Shift Quantification with Robustness Guarantees via Distribution Feature Matching.](http://arxiv.org/abs/2306.04376) | 本文提出一种名为DFM框架的方法，用于量化标签偏移，并证明了其性能上限和鲁棒性。使用基于核的DFM版本可以提高效率、可扩展性和鲁棒性。 |
| [^41] | [Entropic covariance models.](http://arxiv.org/abs/2306.03590) | 本文提出了一个通用的线性约束协方差矩阵变换的框架，并提出了一种估计方法，解决了一个凸问题，允许相对简单的渐近性和有限样本分析。研究的重点是关于建模相关矩阵和稀疏性方面的内容。 |
| [^42] | [Evaluation of the MACE Force Field Architecture: from Medicinal Chemistry to Materials Science.](http://arxiv.org/abs/2305.14247) | MACE的机器学习力场架构在内域、外推和低数据范围任务中表现优秀，在处理非晶碳、小分子有机化学、大分子和液态水等领域时常常优于其他替代方案。即使只有50个随机选定的参考配置，该模型也能非常高效地复现实验分子振动光谱。 |
| [^43] | [On Manifold Learning in Plato's Cave: Remarks on Manifold Learning and Physical Phenomena.](http://arxiv.org/abs/2304.14248) | 本文通过一个警示故事阐释了分析数据时，测量几何和底层现象几何差异带来的问题，以及这种差异在某些情况下如何导致对一个修正过的问题给出错误答案。这些问题适用于降维和无监督学习领域。 |
| [^44] | [Post-selection Inference for Conformal Prediction: Trading off Coverage for Precision.](http://arxiv.org/abs/2304.06158) | 本论文提出一种使用无分布信赖带的 uniform conformal inference 算法，实现任意数据相关误覆盖水平的有限样本预测保证的统一一致性推理。 |
| [^45] | [Implicit Balancing and Regularization: Generalization and Convergence Guarantees for Overparameterized Asymmetric Matrix Sensing.](http://arxiv.org/abs/2303.14244) | 本论文研究了过参数化低秩矩阵感知问题，证明了通过因子化方法训练的过参数化模型可以收敛，并且隐式平衡和正则化可以促进泛化。 |
| [^46] | [A High-dimensional Convergence Theorem for U-statistics with Applications to Kernel-based Testing.](http://arxiv.org/abs/2302.05686) | 本论文证明了一个适用于核心测试的高维U统计的收敛定理，并发现U统计的极限分布会经历从非退化高斯极限到退化极限的相变。这一现象对于高维情况下的非退化U统计具有较大方差和不对称分布的非高斯极限具有重要意义。此外，我们提出的界限适用于任何有限数量和维度的样本，与底层函数的特征值无关，并且在某些假设下与维度无关。我们还将我们的理论应用到两个常用的基于核函数的分布测试方法，MMD和KSD，来研究它们的高维性能。我们的结果能够准确预测测试功率如何与维度和带宽的关系。 |
| [^47] | [Dense Hebbian neural networks: a replica symmetric picture of supervised learning.](http://arxiv.org/abs/2212.00606) | 这篇论文研究了通过教师训练的密集Hebbian神经网络的计算能力，通过统计力学和蒙特卡罗模拟得到了一个相图，指出这些网络在大规模和结构简单的数据集下可以在超大存储或超高检测区域工作。 |
| [^48] | [Dense Hebbian neural networks: a replica symmetric picture of unsupervised learning.](http://arxiv.org/abs/2211.14067) | 本文研究了无监督训练的密集贺维神经网络，并通过统计力学方法和蒙特卡洛模拟分析了其计算能力。我们得到了一个相图，总结了网络性能与训练数据集质量、数量和网络存储之间的关系，并建立了统计力学中的宏观可观测量与机器学习中的损失函数的联系。 |
| [^49] | [Shapley Curves: A Smoothing Perspective.](http://arxiv.org/abs/2211.13289) | 本文以平滑的角度引入了Shapley曲线作为局部变量重要性的度量，提出了两种估计策略，并在特征的独立和依赖情况下得到了一致性和渐近正态性，为估计的Shapley曲线构建了置信区间并进行了推断，通过实验证实了渐近结果。应用中分析了哪些属性驱动车辆价格。 |
| [^50] | [Active Acquisition for Multimodal Temporal Data: A Challenging Decision-Making Task.](http://arxiv.org/abs/2211.05039) | 该论文提出了一个具有挑战性的决策任务，主动获取多模态时间数据。通过权衡获取成本和预测性能，学习代理程序来主动选择获取的输入模态。该方法能够解决具有实际相关推理技能的合成情景，并在真实数据集上成功学习到成本反应式的获取行为，但无法学习到自适应的获取策略，突显了任务的困难性。 |
| [^51] | [Neural Extended Kalman Filters for Learning and Predicting Dynamics of Structural Systems.](http://arxiv.org/abs/2210.04165) | 本论文提出了一种称为神经扩展卡尔曼滤波器（Neural EKF）的可学习卡尔曼滤波方法，用于学习复杂物理系统的潜在演化动力学。这种方法可以通过端到端训练来学习过程动力学和传感观测的建模，提高结构响应预测的准确性。 |
| [^52] | [The Vendi Score: A Diversity Evaluation Metric for Machine Learning.](http://arxiv.org/abs/2210.02410) | 本文提出了一种用于机器学习的多样性评估指标Vendi分数，它能够灵活地衡量不同形式的多样性，而且不需要参考数据集，适用于任何生成模型和数据集。 |
| [^53] | [Inference on Strongly Identified Functionals of Weakly Identified Functions.](http://arxiv.org/abs/2208.08291) | 本文研究了一种新的条件，使得即使干扰函数是弱标识的，也可以对功能进行强标识并进行推理。 |
| [^54] | [Feature-Based Time-Series Analysis in R using the theft Package.](http://arxiv.org/abs/2208.06146) | 本研究介绍了在R中使用theft包进行基于特征的时间序列分析的方法，并指出了当前存在的问题包括缺乏统一的访问点以及用户需要掌握多种编程语言来获得所有特征集。 |
| [^55] | [Deep Direct Discriminative Decoders for High-dimensional Time-series Data Analysis.](http://arxiv.org/abs/2205.10947) | 这篇论文提出了一种用于高维时间序列数据分析的新方法，即深度直接判别解码器（D4）。D4通过引入深度神经网络的表达能力和可扩展性，有效地估计了高维观测信号下的潜在状态过程，并在多个数据集上展示了比传统方法更好的性能。 |
| [^56] | [Externally Valid Policy Choice.](http://arxiv.org/abs/2205.05561) | 本文研究了外部有效的个性化治疗策略的学习问题。研究表明，最大化福利的治疗策略对于实验和目标人群之间的结果分布变化具有鲁棒性。通过开发新的方法，作者提出了对结果和特征变化具有鲁棒性的策略学习方法，并强调实验人群内的治疗效果异质性对策略普适性的影响。 |
| [^57] | [Bivariate vine copula based regression, bivariate level and quantile curves.](http://arxiv.org/abs/2205.02557) | 该论文提出了一种基于藤蔓顺序相关的回归模型，用于构建双变量分位数，并使用藤蔓相关的水平曲线。这种方法可以避免传统回归模型的一些问题，如变量转换、共线性和分位数交叉。 |
| [^58] | [Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations.](http://arxiv.org/abs/2204.02937) | 本文研究表明，简单的最后一层重新训练足以提高神经网络分类器对虚假相关性的鲁棒性，可以在虚假相关性基准测试中与最先进的方法相媲美或胜过，但其复杂度和计算开销较低。此外，对于在ImageNet训练的大型模型进行最后一层重新训练，仅几分钟的训练时间就可以显著降低对背景和纹理信息的依赖，提高对协变量转变的鲁棒性。 |
| [^59] | [Learning Hidden Markov Models When the Locations of Missing Observations are Unknown.](http://arxiv.org/abs/2203.06527) | 本文研究了当缺失观测的位置未知时学习隐马尔可夫模型的问题，并提供了不需要先验信息的重建算法。 |
| [^60] | [A Proximal Algorithm for Sampling.](http://arxiv.org/abs/2202.13975) | 本论文提出了一种用于处理缺乏平滑性的势能的采样问题的近端算法，在凸和非凸情况下均可适用。该算法的关键创新点在于基于拒绝采样的交替采样框架的实际实现，比现有方法更高效。 |
| [^61] | [Deviance Matrix Factorization.](http://arxiv.org/abs/2110.05674) | 该论文研究了一种适用于偏差数据损失的通用矩阵分解方法，并且通过应用广义线性模型理论提供了支持，该方法具有灵活的算法和处理结构性零元素的能力。作者通过模拟研究和案例研究证明了该方法的鲁棒性和应用广泛性。 |
| [^62] | [Contextual Inverse Optimization: Offline and Online Learning.](http://arxiv.org/abs/2106.14015) | 这项研究研究了具有反馈信息的离线和在线情境优化问题，通过观察最佳动作并最小化后悔来优化决策制定。 |
| [^63] | [Deep Proxy Causal Learning and its Application to Confounded Bandit Policy Evaluation.](http://arxiv.org/abs/2106.03907) | 本论文提出了一种深度代理因果学习（PCL）方法，用于在存在混淆因素的情况下估计治疗对结果的因果效应。通过构建治疗和代理之间的模型，并利用该模型在给定代理的情况下学习治疗对结果的影响，PCL可以保证恢复真实的因果效应。作者还提出了一种名为深度特征代理变量方法（DFPV）的新方法，用于处理高维和非线性复杂关系的情况，并表明DFPV在合成基准测试中的性能优于最先进的PCL方法。 |
| [^64] | [Bridging Offline Reinforcement Learning and Imitation Learning: A Tale of Pessimism.](http://arxiv.org/abs/2103.12021) | 本论文提出了一种新的联机强化学习框架，通过平滑插值的方式将模仿学习和纯联机强化学习统一起来。框架围绕着一种衡量行为策略与专家策略偏离程度的弱版本集中系数展开。通过该框架，研究者进一步研究了算法设计的问题：能否开发出实现最小极大最优性的算法？ |
| [^65] | [Transfer Learning in Deep Reinforcement Learning: A Survey.](http://arxiv.org/abs/2009.07888) | 这篇综述调查了深度强化学习领域中的迁移学习方法的最新进展，并提供了一个对这些方法进行分类的框架。分析了它们的目标、方法学、兼容的强化学习背景以及实际应用，并探讨了迁移学习与其他相关主题之间的联系。 |
| [^66] | [Learning to Switch Among Agents in a Team via 2-Layer Markov Decision Processes.](http://arxiv.org/abs/2002.04258) | 本文研究了在团队中学习切换代理控制的问题，并开发了一种在线学习算法，通过学习代理的策略和环境的转移概率，在不同自动化水平下使现有的强化学习代理能够工作。该算法的总遗憾与最佳切换策略相比是次线性的，当多个代理团队在相似环境中运行时，该算法从维护环境的共享置信界中获益匪浅。 |
| [^67] | [Meta Adaptation using Importance Weighted Demonstrations.](http://arxiv.org/abs/1911.10322) | 本文提出了一种使用重要权重示范的元适应性学习算法，通过对特定任务的先前知识进行分配重要权重，实现了在任何相关任务上的泛化。实验证明，该方法能够使机器人在多样化环境任务中进行训练，并通过少量示范适应未知环境。 |

# 详细

[^1]: 通过学习扩散改进采样

    Improved sampling via learned diffusions. (arXiv:2307.01198v1 [cs.LG])

    [http://arxiv.org/abs/2307.01198](http://arxiv.org/abs/2307.01198)

    通过学习扩散的方法改进了采样过程，引入了基于变分形式的路径空间度量，提出了对数方差损失，优化了采样性能。

    

    最近，一系列论文提出了基于深度学习的方法，使用控制扩散过程从非标准化目标密度中采样。在本研究中，我们将这些方法视为Schrödinger桥问题的特例，寻求给定先验分布和指定目标之间最可能的随机演化。我们进一步通过引入基于时间反演扩散过程的路径空间度量之间的差异的变分形式来推广这个框架。这个抽象的视角导致了可以通过梯度优化的实际损失，并将先前的目标作为特例。与此同时，它允许我们考虑除了已知存在模式坍缩问题的反向Kullback-Leibler差别之外的其他差别。特别地，我们提出了所谓的对数方差损失，它具有良好的数值特性，并显著提高了在所有考虑的情况下的性能。

    Recently, a series of papers proposed deep learning-based approaches to sample from unnormalized target densities using controlled diffusion processes. In this work, we identify these approaches as special cases of the Schr\"odinger bridge problem, seeking the most likely stochastic evolution between a given prior distribution and the specified target. We further generalize this framework by introducing a variational formulation based on divergences between path space measures of time-reversed diffusion processes. This abstract perspective leads to practical losses that can be optimized by gradient-based algorithms and includes previous objectives as special cases. At the same time, it allows us to consider divergences other than the reverse Kullback-Leibler divergence that is known to suffer from mode collapse. In particular, we propose the so-called log-variance loss, which exhibits favorable numerical properties and leads to significantly improved performance across all considered a
    
[^2]: 将大量随机点拟合成椭球体的问题

    Fitting an ellipsoid to a quadratic number of random points. (arXiv:2307.01181v1 [math.PR])

    [http://arxiv.org/abs/2307.01181](http://arxiv.org/abs/2307.01181)

    将$n$个高斯随机向量拟合到以原点为中心的椭球体边界的问题$(\mathrm{P})$，我们提出了一个基于随机向量Gram矩阵集中性的改进方法，证明了当$n \leq d^2 / C$时，问题$(\mathrm{P})$具有很高的可行性概率。

    

    我们考虑当$n, d \to \infty $时，将$n$个标准高斯随机向量拟合到以原点为中心的椭球体的边界的问题$(\mathrm{P})$。这个问题被猜测具有尖锐的可行性转变：对于任意$\varepsilon > 0$，如果$n \leq (1 - \varepsilon) d^2 / 4$，那么$(\mathrm{P})$有很高的概率有解；而如果$n \geq (1 + \varepsilon) d^2 /4$，那么$(\mathrm{P})$有很高的概率无解。目前，对于负面情况，只知道$n \geq d^2 / 2$是平凡的一个上界，而对于正面情况，已知的最好结果是假设$n \leq d^2 / \mathrm{polylog}(d)$。在这项工作中，我们利用Bartl和Mendelson关于随机向量的Gram矩阵集中性的一个关键结果改进了以前的方法。这使得我们可以给出一个简单的证明，当$n \leq d^2 / C$时，问题$(\mathrm{P})$有很高的概率是可行的，其中$C> 0$是一个（可能很大的）常数。

    We consider the problem $(\mathrm{P})$ of fitting $n$ standard Gaussian random vectors in $\mathbb{R}^d$ to the boundary of a centered ellipsoid, as $n, d \to \infty$. This problem is conjectured to have a sharp feasibility transition: for any $\varepsilon > 0$, if $n \leq (1 - \varepsilon) d^2 / 4$ then $(\mathrm{P})$ has a solution with high probability, while $(\mathrm{P})$ has no solutions with high probability if $n \geq (1 + \varepsilon) d^2 /4$. So far, only a trivial bound $n \geq d^2 / 2$ is known on the negative side, while the best results on the positive side assume $n \leq d^2 / \mathrm{polylog}(d)$. In this work, we improve over previous approaches using a key result of Bartl & Mendelson on the concentration of Gram matrices of random vectors under mild assumptions on their tail behavior. This allows us to give a simple proof that $(\mathrm{P})$ is feasible with high probability when $n \leq d^2 / C$, for a (possibly large) constant $C > 0$.
    
[^3]: 使用DDPM目标函数学习高斯混合模型

    Learning Mixtures of Gaussians Using the DDPM Objective. (arXiv:2307.01178v1 [cs.DS])

    [http://arxiv.org/abs/2307.01178](http://arxiv.org/abs/2307.01178)

    本文针对高斯混合模型这一基础分布族提供了首个可证明高效的结果，通过梯度下降对去噪扩散概率模型（DDPM）目标进行训练可以有效地恢复混合模型的参数。

    

    最近的研究表明，扩散模型可以学习几乎任何分布，前提是我们能够进行评分估计。然而，在什么设置下可以进行评分估计，以及何时可以实际上证明基于梯度的算法能够成功，这仍然不十分清楚。在这项工作中，我们首次在这些方面提供了可证明有效的结果，研究的是最基本的分布族之一，即高斯混合模型。我们证明了在以下两种设置下，通过梯度下降对去噪扩散概率模型（DDPM）目标进行训练可以高效地恢复混合模型的真实参数：1）我们证明了在随机初始化的情况下，梯度下降可以学习具有$d$维度和$1/\text{poly}(d)$-分隔中心的两个球面高斯混合模型。2）我们证明了在带有热启动的情况下，梯度下降可以学习具有$\Omega(\sqrt{\log(\min(K,d))})$-分隔中心的$K$个球面高斯混合模型。我们证明的一个关键因素是一个新的...

    Recent works have shown that diffusion models can learn essentially any distribution provided one can perform score estimation. Yet it remains poorly understood under what settings score estimation is possible, let alone when practical gradient-based algorithms for this task can provably succeed.  In this work, we give the first provably efficient results along these lines for one of the most fundamental distribution families, Gaussian mixture models. We prove that gradient descent on the denoising diffusion probabilistic model (DDPM) objective can efficiently recover the ground truth parameters of the mixture model in the following two settings: 1) We show gradient descent with random initialization learns mixtures of two spherical Gaussians in $d$ dimensions with $1/\text{poly}(d)$-separated centers. 2) We show gradient descent with a warm start learns mixtures of $K$ spherical Gaussians with $\Omega(\sqrt{\log(\min(K,d))})$-separated centers. A key ingredient in our proofs is a new 
    
[^4]: 神经希尔伯特阶梯：函数空间中的多层神经网络

    Neural Hilbert Ladders: Multi-Layer Neural Networks in Function Space. (arXiv:2307.01177v1 [cs.LG])

    [http://arxiv.org/abs/2307.01177](http://arxiv.org/abs/2307.01177)

    本文提出了神经希尔伯特阶梯(NHL)的概念，它将多层神经网络描述为一系列的再生核希尔伯特空间，进一步推广了浅层神经网络的理论研究，并探讨了其在函数空间内的性质和应用。通过证明不同层次的NHL与多层NNs之间的对应关系，证明了学习NHL的泛化保证，并提出了NHL的特征动力学模型。最后，在ReLU和二次激活函数下展示了NHLs中的深度分离现象。

    

    神经网络(NNs)所探索的函数空间的特征化是深度学习理论的重要方面。本文将具有任意宽度的多层NN视为定义特定层次的再生核希尔伯特空间(RKHS)的神经希尔伯特阶梯(NHL)。这使得我们能够定义一个函数空间和一个复杂度度量，该度量推广了浅层NNs的先前结果，并研究了它们在几个方面的理论特性和影响。首先，我们证明了L层NNs表示的函数与属于L层NHLs的函数之间的对应关系。其次，我们证明了学习具有受控复杂度度量的NHL的泛化保证。第三，对应于在无穷宽均场极限下训练多层NNs，我们导出了NHL的特征动力学，该动力学被描述为多个随机场的演化。第四，在ReLU和二次激活函数下展示了NHLs中的深度分离示例。

    The characterization of the functions spaces explored by neural networks (NNs) is an important aspect of deep learning theory. In this work, we view a multi-layer NN with arbitrary width as defining a particular hierarchy of reproducing kernel Hilbert spaces (RKHSs), named a Neural Hilbert Ladder (NHL). This allows us to define a function space and a complexity measure that generalize prior results for shallow NNs, and we then examine their theoretical properties and implications in several aspects. First, we prove a correspondence between functions expressed by L-layer NNs and those belonging to L-level NHLs. Second, we prove generalization guarantees for learning an NHL with the complexity measure controlled. Third, corresponding to the training of multi-layer NNs in the infinite-width mean-field limit, we derive an evolution of the NHL characterized as the dynamics of multiple random fields. Fourth, we show examples of depth separation in NHLs under ReLU and quadratic activation fun
    
[^5]: 通过最速下降法分析和改进基于贪婪的二维坐标更新在等式约束优化中的应用

    Analyzing and Improving Greedy 2-Coordinate Updates for Equality-Constrained Optimization via Steepest Descent in the 1-Norm. (arXiv:2307.01169v1 [math.OC])

    [http://arxiv.org/abs/2307.01169](http://arxiv.org/abs/2307.01169)

    通过最速下降法，我们在等式约束优化问题中探索并改进了贪婪的二维坐标更新方法，在满足特定条件下取得了更快的收敛速度。此外，我们还将该方法推广到同时具有求和约束和边界约束的问题，并证明了在L1-范数下的最速下降法可以在更短的计算时间内取得更多的进展。

    

    本文考虑在变量的求和约束下最小化一个平滑函数。通过利用贪婪的2维坐标更新与等式约束的最速下降法之间的联系，我们给出了一个收敛速度，该速度在满足近端Polyak-Lojasiewicz条件下比随机选择更快，并且与问题维度n无关。然后，我们考虑同时具有求和约束和边界约束的最小化问题，这在支持向量机对偶问题中出现。现有的贪婪规则要么只能保证微小的进展，要么需要O(n^2)的计算时间。我们证明了边界和求和约束的L1-范数最速下降法在每次迭代中可以比以前的规则取得更多的进展，并且可以在O(n log n)的时间内计算。

    We consider minimizing a smooth function subject to a summation constraint over its variables. By exploiting a connection between the greedy 2-coordinate update for this problem and equality-constrained steepest descent in the 1-norm, we give a convergence rate for greedy selection under a proximal Polyak-Lojasiewicz assumption that is faster than random selection and independent of the problem dimension $n$. We then consider minimizing with both a summation constraint and bound constraints, as arises in the support vector machine dual problem. Existing greedy rules for this setting either guarantee trivial progress only or require $O(n^2)$ time to compute. We show that boundand summation-constrained steepest descent in the L1-norm guarantees more progress per iteration than previous rules and can be computed in only $O(n \log n)$ time.
    
[^6]: 在分布转移和长尾数据下，对现代视觉架构进行合规预测的经验证实

    Empirically Validating Conformal Prediction on Modern Vision Architectures Under Distribution Shift and Long-tailed Data. (arXiv:2307.01088v1 [cs.LG])

    [http://arxiv.org/abs/2307.01088](http://arxiv.org/abs/2307.01088)

    本文在大规模数据集和模型上首次对分布转移和长尾类别分布下的合规预测方法进行了实证评估。研究发现，这些方法在分布转移和长尾设置下的性能大大下降，对于在现实世界和安全关键应用中的部署具有重要的局限性。

    

    合规预测已经成为一种可靠地为深度学习模型提供不确定性估计和安全保证的方法。然而，它的性能已知在分布转移和长尾类别分布下会下降，而这在现实世界的应用中经常存在。在本文中，我们对这些情况下的几种事后和基于训练的合规预测方法进行了性能表征，并首次在大规模数据集和模型上进行了实证评估。我们发现在许多合规方法和神经网络家族中，性能在分布转移下违反安全保证时大大下降。同样，在长尾设置中，我们发现许多类别的保证经常被违反。了解这些方法的局限性对于在现实世界和安全关键应用中部署是必要的。

    Conformal prediction has emerged as a rigorous means of providing deep learning models with reliable uncertainty estimates and safety guarantees. Yet, its performance is known to degrade under distribution shift and long-tailed class distributions, which are often present in real world applications. Here, we characterize the performance of several post-hoc and training-based conformal prediction methods under these settings, providing the first empirical evaluation on large-scale datasets and models. We show that across numerous conformal methods and neural network families, performance greatly degrades under distribution shifts violating safety guarantees. Similarly, we show that in long-tailed settings the guarantees are frequently violated on many classes. Understanding the limitations of these methods is necessary for deployment in real world and safety-critical applications.
    
[^7]: 校准可微分的基于Agent的模型的一些挑战

    Some challenges of calibrating differentiable agent-based models. (arXiv:2307.01085v1 [cs.MA])

    [http://arxiv.org/abs/2307.01085](http://arxiv.org/abs/2307.01085)

    本文讨论了校准可微分的基于Agent的模型面临的挑战，同时提出了潜在的解决方案。

    

    基于Agent的模型是一种有前途的模拟和推理复杂系统的方法，但是它们的应用受到了复杂性、离散性和参数推导和优化任务的困难的限制。这引起了构建可微分的基于Agent的模型作为克服这些困难的策略的兴趣，然而还存在一些挑战。在本文中，我们讨论并展示了一些实验，突出了这些挑战以及潜在的解决方案。

    Agent-based models (ABMs) are a promising approach to modelling and reasoning about complex systems, yet their application in practice is impeded by their complexity, discrete nature, and the difficulty of performing parameter inference and optimisation tasks. This in turn has sparked interest in the construction of differentiable ABMs as a strategy for combatting these difficulties, yet a number of challenges remain. In this paper, we discuss and present experiments that highlight some of these challenges, along with potential solutions.
    
[^8]: 通过随机森林保持几何特性的近似来进行监督流形学习

    Supervised Manifold Learning via Random Forest Geometry-Preserving Proximities. (arXiv:2307.01077v1 [stat.ML])

    [http://arxiv.org/abs/2307.01077](http://arxiv.org/abs/2307.01077)

    本文通过使用随机森林近似的几何保持特性作为流形学习方法的初始化，展示了类条件流形学习的局限性，并提出了一种替代选择。这种方法能够在几乎所有流形学习方法中保持局部结构，并正确地维护全局结构。

    

    流形学习方法旨在在高维空间中寻找内在的低维数据结构。主流的流形学习算法，例如Isomap，UMAP，t-SNE，Diffusion Map和Laplacian Eigenmaps，不使用数据标签，因此被认为是无监督的。现有的这些方法的有监督扩展仅适用于分类问题，并且由于使用了不保持顺序的类条件距离而未能揭示有意义的嵌入。在本文中，我们定量和可视化地展示了类条件流形学习的弱点，并提出了一种替代选择，在流形学习方法中使用数据几何保持的随机森林近似作为初始化。我们展示了使用这些近似方法进行局部结构保持在几乎所有流形学习方法中都是普遍的，并且使用基于扩散的方法能够正确地维护全局结构。

    Manifold learning approaches seek the intrinsic, low-dimensional data structure within a high-dimensional space. Mainstream manifold learning algorithms, such as Isomap, UMAP, $t$-SNE, Diffusion Map, and Laplacian Eigenmaps do not use data labels and are thus considered unsupervised. Existing supervised extensions of these methods are limited to classification problems and fall short of uncovering meaningful embeddings due to their construction using order non-preserving, class-conditional distances. In this paper, we show the weaknesses of class-conditional manifold learning quantitatively and visually and propose an alternate choice of kernel for supervised dimensionality reduction using a data-geometry-preserving variant of random forest proximities as an initialization for manifold learning methods. We show that local structure preservation using these proximities is near universal across manifold learning approaches and global structure is properly maintained using diffusion-based
    
[^9]: 运输、变分推断和扩散：应用于回火流和薛定谔桥的论文研究

    Transport, Variational Inference and Diffusions: with Applications to Annealed Flows and Schr\"odinger Bridges. (arXiv:2307.01050v1 [stat.ML])

    [http://arxiv.org/abs/2307.01050](http://arxiv.org/abs/2307.01050)

    本文研究了最优运输和变分推断之间的联系，并提出了一种基于路径空间散度的采样和生成建模框架。通过开发新颖的基于得分的回火流技术和正则化的迭代比例拟合目标，本文展示了这些方法的潜力。

    

    本文探讨了最优运输与变分推断之间的联系，重点研究了正向和反向随机微分方程以及Girsanov变换。我们提出了一个基于路径空间散度的采样和生成建模的原则性和系统性框架。我们的工作最终发展出一个新颖的基于得分的回火流技术（与统计物理中的Jarzynski和Crooks恒等式有关）和一个正则化的迭代比例拟合（IPF）型目标，不同于标准IPF的顺序性。通过一系列的生成建模示例和基于双井的稀有事件任务，我们展示了所提方法的潜力。

    This paper explores the connections between optimal transport and variational inference, with a focus on forward and reverse time stochastic differential equations and Girsanov transformations.We present a principled and systematic framework for sampling and generative modelling centred around divergences on path space. Our work culminates in the development of a novel score-based annealed flow technique (with connections to Jarzynski and Crooks identities from statistical physics) and a regularised iterative proportional fitting (IPF)-type objective, departing from the sequential nature of standard IPF. Through a series of generative modelling examples and a double-well-based rare event task, we showcase the potential of the proposed methods.
    
[^10]: 双重稳健估计机器学习下的直接和间接分位治疗效应

    Doubly Robust Estimation of Direct and Indirect Quantile Treatment Effects with Machine Learning. (arXiv:2307.01049v1 [econ.EM])

    [http://arxiv.org/abs/2307.01049](http://arxiv.org/abs/2307.01049)

    该论文提出了一种双重稳健估计方法，用于估计机器学习下的直接和间接分位治疗效应，通过机器学习和交叉拟合来处理可观测选择偏差，并提出了乘法自助法进行统计推断。

    

    我们提出了一种双重/无偏的机器学习估计方法，用于处理可观测选择偏差的情况下的直接和间接分位治疗效应。这使得能够将二进制治疗的因果效应在特定结果排名上分解为通过中介变量（称为中介因子）间接影响和（未经中介的）直接影响的组成部分。所提方法基于潜在结果的累积分布函数的有效得分函数，对于某些固定参数的偏差，如结果、处理和中介模型，是稳健的。我们通过机器学习来估计这些固定参数，并使用交叉拟合来减小对直接和间接分位治疗效应估计的过拟合偏差。我们证明了我们效果估计量的一致性和渐近正态性。我们还提出了一个乘法自助法进行统计推断，并展示了乘法自助法的有效性。

    We suggest double/debiased machine learning estimators of direct and indirect quantile treatment effects under a selection-on-observables assumption. This permits disentangling the causal effect of a binary treatment at a specific outcome rank into an indirect component that operates through an intermediate variable called mediator and an (unmediated) direct impact. The proposed method is based on the efficient score functions of the cumulative distribution functions of potential outcomes, which are robust to certain misspecifications of the nuisance parameters, i.e., the outcome, treatment, and mediator models. We estimate these nuisance parameters by machine learning and use cross-fitting to reduce overfitting bias in the estimation of direct and indirect quantile treatment effects. We establish uniform consistency and asymptotic normality of our effect estimators. We also propose a multiplier bootstrap for statistical inference and show the validity of the multiplier bootstrap. Fina
    
[^11]: 近期量子装置上的量子机器学习: 监督和无监督技术在现实世界应用的现状

    Quantum Machine Learning on Near-Term Quantum Devices: Current State of Supervised and Unsupervised Techniques for Real-World Applications. (arXiv:2307.00908v1 [quant-ph])

    [http://arxiv.org/abs/2307.00908](http://arxiv.org/abs/2307.00908)

    近期量子设备上的量子机器学习应用中，我们着重研究了监督和无监督学习在现实世界场景的应用。我们探究了当前量子硬件上的QML实现的限制，并提出了克服这些限制的技术。与经典对应物相比较，这些QML实现的性能得到了评估。

    

    在过去十年中，量子硬件在速度、量子比特数量和量子体积方面取得了相当大的进展，量子体积被定义为在近期量子设备上可以有效实现的量子电路的最大规模。因此，在实际硬件上应用量子机器学习(QML)以实现量子优势已经有了很大的增长。在这篇综述中，我们主要关注在量子硬件上实现的选定监督和无监督学习应用，特别针对现实世界场景。我们探讨并强调了QML在量子硬件上的当前限制。我们深入讨论了各种克服这些限制的技术，如编码技术、基态结构、误差补偿和梯度方法。此外，我们评估了这些QML实现与它们的经典对应物之间的性能对比。

    The past decade has seen considerable progress in quantum hardware in terms of the speed, number of qubits and quantum volume which is defined as the maximum size of a quantum circuit that can be effectively implemented on a near-term quantum device. Consequently, there has also been a rise in the number of works based on the applications of Quantum Machine Learning (QML) on real hardware to attain quantum advantage over their classical counterparts. In this survey, our primary focus is on selected supervised and unsupervised learning applications implemented on quantum hardware, specifically targeting real-world scenarios. Our survey explores and highlights the current limitations of QML implementations on quantum hardware. We delve into various techniques to overcome these limitations, such as encoding techniques, ansatz structure, error mitigation, and gradient methods. Additionally, we assess the performance of these QML implementations in comparison to their classical counterparts
    
[^12]: MADS：调控式自解码SIREN用于时间序列插补

    MADS: Modulated Auto-Decoding SIREN for time series imputation. (arXiv:2307.00868v1 [stat.ML])

    [http://arxiv.org/abs/2307.00868](http://arxiv.org/abs/2307.00868)

    本论文提出了一种新的自解码框架MADS，用于时间序列插补。该方法基于隐式神经表示，利用SIREN的能力进行高保真重建，并采用超网络架构进行泛化。实验证明该模型在两个真实数据集上的表现优于现有最先进的方法。

    

    由于所建模数据中具有潜在的显著变异性，时间序列插补在许多领域仍然是一个重要挑战。传统的插补方法通常对底层数据生成过程施加强假设，限制了它们的适用性，而研究人员最近开始探索深度学习在此任务中的潜力，受到这些模型在分类和回归问题上的强大性能的启发，应用范围广泛。在这项工作中，我们提出了一种新颖的基于隐式神经表示的时间序列插补自解码框架MADS。我们的方法利用了SIREN对信号和不规则数据进行高保真重建的能力，并将其与超网络架构相结合，通过学习时间序列空间的先验知识来实现泛化。我们在两个真实数据集上评估了我们的模型，并展示它超越了现有最先进的方法。

    Time series imputation remains a significant challenge across many fields due to the potentially significant variability in the type of data being modelled. Whilst traditional imputation methods often impose strong assumptions on the underlying data generation process, limiting their applicability, researchers have recently begun to investigate the potential of deep learning for this task, inspired by the strong performance shown by these models in both classification and regression problems across a range of applications. In this work we propose MADS, a novel auto-decoding framework for time series imputation, built upon implicit neural representations. Our method leverages the capabilities of SIRENs for high fidelity reconstruction of signals and irregular data, and combines it with a hypernetwork architecture which allows us to generalise by learning a prior over the space of time series. We evaluate our model on two real-world datasets, and show that it outperforms state-of-the-art
    
[^13]: CardiGraphormer: 揭示自监督学习在颠覆药物发现中的力量

    CardiGraphormer: Unveiling the Power of Self-Supervised Learning in Revolutionizing Drug Discovery. (arXiv:2307.00859v1 [cs.LG])

    [http://arxiv.org/abs/2307.00859](http://arxiv.org/abs/2307.00859)

    CardiGraphormer是一种革命性的方法，结合了自监督学习、图神经网络和保持基数注意力，颠覆了药物发现的方式。它利用自监督学习学习分子表示并利用图神经网络提取分子指纹，提高了预测性能和可解释性，同时减少了计算时间，并在处理复杂数据和执行各种与图结构相关的任务方面表现出色。

    

    在广阔的药物发现领域中，已知药物约有15,000种，但只有大约4,200种得到了批准，化学空间的组合性质提供了一项艰巨的挑战。尽管人工智能成为了有力的伙伴，传统的人工智能框架仍面临重大障碍。本文介绍了CardiGraphormer，这是一种划时代的方法，通过结合自监督学习（SSL）、图神经网络（GNN）和保持基数注意力，从而颠覆药物发现。CardiGraphormer是Graphormer和保持基数注意力的新颖组合，利用SSL学习有效的分子表示，并利用GNN提取分子指纹，提高了预测性能和可解释性，并减少了计算时间。它在处理分子结构等复杂数据方面表现出色，并能执行与节点、节点对、子图或整个图结构相关的任务。

    In the expansive realm of drug discovery, with approximately 15,000 known drugs and only around 4,200 approved, the combinatorial nature of the chemical space presents a formidable challenge. While Artificial Intelligence (AI) has emerged as a powerful ally, traditional AI frameworks face significant hurdles. This manuscript introduces CardiGraphormer, a groundbreaking approach that synergizes self-supervised learning (SSL), Graph Neural Networks (GNNs), and Cardinality Preserving Attention to revolutionize drug discovery. CardiGraphormer, a novel combination of Graphormer and Cardinality Preserving Attention, leverages SSL to learn potent molecular representations and employs GNNs to extract molecular fingerprints, enhancing predictive performance and interpretability while reducing computation time. It excels in handling complex data like molecular structures and performs tasks associated with nodes, pairs of nodes, subgraphs, or entire graph structures. CardiGraphormer's potential a
    
[^14]: 在在线分类中权衡支付和准确性：基于付费随机专家的研究

    Trading-Off Payments and Accuracy in Online Classification with Paid Stochastic Experts. (arXiv:2307.00836v1 [stat.ML])

    [http://arxiv.org/abs/2307.00836](http://arxiv.org/abs/2307.00836)

    该研究探索了在线分类中使用付费随机专家的方法，通过权衡支付金额和准确性来进行预测。研究提出了一种在线学习算法，其总成本不超过预先知道所有专家生产力的预测算法成本的函数，通过结合Lipschitz Bandits和基于替代损失的在线分类，我们改进了现有的界限。

    

    我们研究了在线分类中的付费随机专家。在这种情况下，每个专家在进行预测之前必须付费。我们支付给每个专家的金额通过某个未知的Lipschitz“生产力”函数直接影响他们的预测准确性。在每一轮中，学习者必须决定为每个专家支付多少金额，然后进行预测。他们所承担的成本是预测误差和所有专家的预付款的加权总和。我们引入了一种在线学习算法，其在T轮后的总成本最多超过事先知道所有专家生产力的预测算法的成本$\mathcal{O}(K^2(\log T)\sqrt{T})$，其中K是专家的数量。为了实现这个结果，我们结合了Lipschitz Bandits和基于替代损失的在线分类。这些工具使我们能够改进标准Lipschitz Bandit设置中的$T^{2/3}$阶段的界限。我们还对合成数据集进行了算法的实证评估。

    We investigate online classification with paid stochastic experts. Here, before making their prediction, each expert must be paid. The amount that we pay each expert directly influences the accuracy of their prediction through some unknown Lipschitz "productivity" function. In each round, the learner must decide how much to pay each expert and then make a prediction. They incur a cost equal to a weighted sum of the prediction error and upfront payments for all experts. We introduce an online learning algorithm whose total cost after $T$ rounds exceeds that of a predictor which knows the productivity of all experts in advance by at most $\mathcal{O}(K^2(\log T)\sqrt{T})$ where $K$ is the number of experts. In order to achieve this result, we combine Lipschitz bandits and online classification with surrogate losses. These tools allow us to improve upon the bound of order $T^{2/3}$ one would obtain in the standard Lipschitz bandit setting. Our algorithm is empirically evaluated on synthet
    
[^15]: Engression: 非线性回归的外推方法

    Engression: Extrapolation for Nonlinear Regression?. (arXiv:2307.00835v1 [stat.ME])

    [http://arxiv.org/abs/2307.00835](http://arxiv.org/abs/2307.00835)

    Engression是一种非线性回归方法，通过使用分布回归技术和预加性噪声模型，在训练样本范围边界外也能可靠地进行外推。

    

    外推对于许多统计学和机器学习应用至关重要，因为常常会遇到超出训练样本范围的测试数据。然而，对于非线性模型来说，外推是一个巨大的挑战。传统模型在这方面通常遇到困难：树集成模型在支持范围外提供连续的预测，而神经网络的预测往往变得不可控。这项工作旨在提供一种非线性回归方法，其可靠性在训练样本范围边界不会立即崩溃。我们的主要贡献是一种名为“engression”的新方法，它是一种预加性噪声模型的分布回归技术，其中噪声添加到协变量上并应用非线性转换。我们的实验结果表明，该模型通常适用于许多真实数据集。我们展示engression可以在一些假设下成功进行外推，例如严格限制噪声大小。

    Extrapolation is crucial in many statistical and machine learning applications, as it is common to encounter test data outside the training support. However, extrapolation is a considerable challenge for nonlinear models. Conventional models typically struggle in this regard: while tree ensembles provide a constant prediction beyond the support, neural network predictions tend to become uncontrollable. This work aims at providing a nonlinear regression methodology whose reliability does not break down immediately at the boundary of the training support. Our primary contribution is a new method called `engression' which, at its core, is a distributional regression technique for pre-additive noise models, where the noise is added to the covariates before applying a nonlinear transformation. Our experimental results indicate that this model is typically suitable for many real data sets. We show that engression can successfully perform extrapolation under some assumptions such as a strictl
    
[^16]: 深度学习中知识的价值

    Worth of knowledge in deep learning. (arXiv:2307.00712v1 [cs.LG])

    [http://arxiv.org/abs/2307.00712](http://arxiv.org/abs/2307.00712)

    该论文提出了一个模型-不可知框架，通过定量实验评估了数据量和估计范围对知识的价值的影响，阐明了数据和知识之间的复杂关系，并可应用于不同网络架构，提供对深度学习模型中先前知识作用的全面理解。

    

    知识是人类用来洞察世界的累积理解和经验。在深度学习中，先前的知识对于弥补数据驱动模型的缺点非常重要，例如数据依赖性、泛化能力和遵守约束。为了能够有效评估知识的价值，我们提出了一个受可解释机器学习启发的框架。通过定量实验，我们评估了数据量和估计范围对知识的价值的影响。我们的发现阐明了数据和知识之间的复杂关系，包括依赖、协同和替代效应。我们的模型无关框架可以应用于各种常见的网络架构，提供对深度学习模型中先前知识作用的全面理解。它还可以用于改进知情机器学习的性能，以及区分不适当的先前知识。

    Knowledge constitutes the accumulated understanding and experience that humans use to gain insight into the world. In deep learning, prior knowledge is essential for mitigating shortcomings of data-driven models, such as data dependence, generalization ability, and compliance with constraints. To enable efficient evaluation of the worth of knowledge, we present a framework inspired by interpretable machine learning. Through quantitative experiments, we assess the influence of data volume and estimation range on the worth of knowledge. Our findings elucidate the complex relationship between data and knowledge, including dependence, synergistic, and substitution effects. Our model-agnostic framework can be applied to a variety of common network architectures, providing a comprehensive understanding of the role of prior knowledge in deep learning models. It can also be used to improve the performance of informed machine learning, as well as distinguish improper prior knowledge.
    
[^17]: Morse神经网络用于不确定性量化

    Morse Neural Networks for Uncertainty Quantification. (arXiv:2307.00667v1 [stat.ML])

    [http://arxiv.org/abs/2307.00667](http://arxiv.org/abs/2307.00667)

    Morse神经网络是一种用于不确定性量化的深度生成模型，通过拟合KL散度损失，可以得到生成密度、OOD检测器、校准温度、生成采样器和距离感知分类器。它统一了多种技术应用，如OOD检测、异常检测和连续学习等。

    

    我们引入了一种新的用于不确定性量化的深度生成模型：Morse神经网络，它将非标准化的高斯密度推广为具有高维子流形的模式，而不仅仅是离散点。通过KL散度损失拟合Morse神经网络可以得到1）（非标准化的）生成密度，2）一种OOD检测器，3）一种校准温度，4）一种生成采样器，以及在有监督情况下5）一种距离感知分类器。Morse网络可以在预训练网络之上使用，以实现对训练数据的距离感知校准。由于其多功能性，Morse神经网络统一了许多技术：例如，（Macêdo等，2021年）的熵值检测器在OOD检测中的应用，（Ruff等，2018年）的单类深度支持向量描述方法在异常检测中的应用，或者（Sun等，2021年）的对比单类分类器在连续学习中的应用。

    We introduce a new deep generative model useful for uncertainty quantification: the Morse neural network, which generalizes the unnormalized Gaussian densities to have modes of high-dimensional submanifolds instead of just discrete points. Fitting the Morse neural network via a KL-divergence loss yields 1) a (unnormalized) generative density, 2) an OOD detector, 3) a calibration temperature, 4) a generative sampler, along with in the supervised case 5) a distance aware-classifier. The Morse network can be used on top of a pre-trained network to bring distance-aware calibration w.r.t the training data. Because of its versatility, the Morse neural networks unifies many techniques: e.g., the Entropic Out-of-Distribution Detector of (Mac\^edo et al., 2021) in OOD detection, the one class Deep Support Vector Description method of (Ruff et al., 2018) in anomaly detection, or the Contrastive One Class classifier in continuous learning (Sun et al., 2021). The Morse neural network has connectio
    
[^18]: 使用去噪扩散概率模型对分子图进行变分自动编码

    Variational Autoencoding Molecular Graphs with Denoising Diffusion Probabilistic Model. (arXiv:2307.00623v1 [cs.LG])

    [http://arxiv.org/abs/2307.00623](http://arxiv.org/abs/2307.00623)

    这篇论文提出了一种新颖的分子深度生成模型，将分层结构融入概率潜在向量中，并通过去噪扩散概率模型来设计有效的分子潜在向量，用于分子性质预测。

    

    在数据驱动的药物发现中，设计分子描述符是一个非常重要的任务。变分自动编码器(VAEs)等深度生成模型通过设计由分子结构导出的概率潜在向量作为描述符，提供了潜在的解决方案。这些模型可以在只有分子结构的大型数据集上进行训练，并应用于迁移学习。然而，通常VAE的潜在向量的近似后验分布假设为简单的多元高斯分布，而且协方差为零，这可能限制了表示潜在特征的性能。为了克服这个限制，我们提出了一种新颖的分子深度生成模型，将分层结构融入概率潜在向量中。我们通过去噪扩散概率模型(DDPM)实现了这一目标。通过一些实验证明了我们模型可以为分子性质预测设计出有效的分子潜在向量。

    In data-driven drug discovery, designing molecular descriptors is a very important task. Deep generative models such as variational autoencoders (VAEs) offer a potential solution by designing descriptors as probabilistic latent vectors derived from molecular structures. These models can be trained on large datasets, which have only molecular structures, and applied to transfer learning. Nevertheless, the approximate posterior distribution of the latent vectors of the usual VAE assumes a simple multivariate Gaussian distribution with zero covariance, which may limit the performance of representing the latent features. To overcome this limitation, we propose a novel molecular deep generative model that incorporates a hierarchical structure into the probabilistic latent vectors. We achieve this by a denoising diffusion probabilistic model (DDPM). We demonstrate that our model can design effective molecular latent vectors for molecular property prediction from some experiments by small dat
    
[^19]: 通过后验采样和潜在扩散模型可证解决线性逆问题

    Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models. (arXiv:2307.00619v1 [cs.LG])

    [http://arxiv.org/abs/2307.00619](http://arxiv.org/abs/2307.00619)

    该论文提出了一种新框架，通过利用预训练的潜在扩散模型来解决线性逆问题。理论分析证明了算法的可靠性，并且在多种问题上实验证明了其优越性能。

    

    我们提出了第一个利用预训练的潜在扩散模型解决线性逆问题的框架。先前提出的算法（如DPS和DDRM）仅适用于像素空间的扩散模型。我们在线性模型设置中从理论上分析了我们的算法，证明了样本恢复的可靠性。从我们的分析中获得的算法洞察力延伸到了实践中常考虑的更一般的设置。实验上，在包括随机修复、块修复、降噪、去模糊、去条纹和超分辨率等各种问题上，我们的算法在先前提出的后验采样算法中表现卓越。

    We present the first framework to solve linear inverse problems leveraging pre-trained latent diffusion models. Previously proposed algorithms (such as DPS and DDRM) only apply to pixel-space diffusion models. We theoretically analyze our algorithm showing provable sample recovery in a linear model setting. The algorithmic insight obtained from our analysis extends to more general settings often considered in practice. Experimentally, we outperform previously proposed posterior sampling algorithms in a wide variety of problems including random inpainting, block inpainting, denoising, deblurring, destriping, and super-resolution.
    
[^20]: 使用基于图形平滑的Gibbs采样优化蛋白质适应性。

    Optimizing protein fitness using Gibbs sampling with Graph-based Smoothing. (arXiv:2307.00494v1 [q-bio.BM])

    [http://arxiv.org/abs/2307.00494](http://arxiv.org/abs/2307.00494)

    使用基于图形平滑的Gibbs采样方法（GGS）优化蛋白质适应性，消除了突变距离的限制，同时提高了搜索效率。该方法在发现高适应性蛋白质方面达到了最先进水平。

    

    能够设计出在给定任务上具有更高适应性的新型蛋白质对许多医学领域来说都是革命性的。然而，通过穷举搜索海量序列空间是不可行的。以前的方法将搜索限制在从参考序列的小突变半径范围内，但这样的启发式方法极大地限制了设计空间。我们的工作旨在消除突变距离的限制，同时实现高效的探索。我们提出了基于图形平滑的Gibbs采样（GGS），它通过迭代应用带有梯度的Gibbs来提出有利的突变，并使用基于图形平滑的方法去除导致假阳性的噪声梯度。我们的方法在训练集中发现了高适应性蛋白质，最多具有8个突变。我们通过研究GFP和AAV设计问题、消融试验和基准模型来阐明结果。

    The ability to design novel proteins with higher fitness on a given task would be revolutionary for many fields of medicine. However, brute-force search through the combinatorially large space of sequences is infeasible. Prior methods constrain search to a small mutational radius from a reference sequence, but such heuristics drastically limit the design space. Our work seeks to remove the restriction on mutational distance while enabling efficient exploration. We propose Gibbs sampling with Graph-based Smoothing (GGS) which iteratively applies Gibbs with gradients to propose advantageous mutations using graph-based smoothing to remove noisy gradients that lead to false positives. Our method is state-of-the-art in discovering high-fitness proteins with up to 8 mutations from the training set. We study the GFP and AAV design problems, ablations, and baselines to elucidate the results. Code: https://github.com/kirjner/GGS
    
[^21]: 使用缺失值的表格数据上的训练扩散模型

    MissDiff: Training Diffusion Models on Tabular Data with Missing Values. (arXiv:2307.00467v1 [cs.LG])

    [http://arxiv.org/abs/2307.00467](http://arxiv.org/abs/2307.00467)

    本研究提出了一个基于扩散的框架，用于处理带缺失值的表格数据，并证明了所提出的训练目标的一致性和上界性能。

    

    扩散模型在建模数据分布和合成数据方面表现出了出色的性能。然而，传统的扩散模型需要完整或完全观察到的数据进行训练。在各种现实世界的应用中，包括医疗和金融，处理表格数据集时常常遇到数据不完整的问题。本工作提出了一个统一和有原则的基于扩散的框架，用于在各种缺失机制下从带缺失值的数据中学习。我们首先观察到广泛采用的“补全再生成”流程可能会导致有偏的学习目标。然后，我们建议在训练阶段屏蔽去噪评分匹配的回归损失。我们证明了所提出的方法在学习数据分布得分方面是一致的，并且所提出的训练目标在某些情况下作为负对数似然的上界。利用逼真和高效的多个表格数据集对所提出的框架进行了评估。

    The diffusion model has shown remarkable performance in modeling data distributions and synthesizing data. However, the vanilla diffusion model requires complete or fully observed data for training. Incomplete data is a common issue in various real-world applications, including healthcare and finance, particularly when dealing with tabular datasets. This work presents a unified and principled diffusion-based framework for learning from data with missing values under various missing mechanisms. We first observe that the widely adopted "impute-then-generate" pipeline may lead to a biased learning objective. Then we propose to mask the regression loss of Denoising Score Matching in the training phase. We prove the proposed method is consistent in learning the score of data distributions, and the proposed training objective serves as an upper bound for the negative likelihood in certain cases. The proposed framework is evaluated on multiple tabular datasets using realistic and efficacious 
    
[^22]: 宽松Pareto集识别的自适应算法

    Adaptive Algorithms for Relaxed Pareto Set Identification. (arXiv:2307.00424v1 [stat.ML])

    [http://arxiv.org/abs/2307.00424](http://arxiv.org/abs/2307.00424)

    本研究提出了一种自适应算法，用于宽松Pareto集的识别，通过放松策略来减少样本复杂度，并展示了在实际场景中的良好表现。

    

    本文重新审视了在多目标多臂赌博机模型中固定置信度下的Pareto最优集合的识别问题。由于准确识别Pareto集合的样本复杂度可能非常大，因此研究了允许输出一些额外近似最优臂的放松策略。在这项工作中，我们还解决了其他允许识别Pareto集合的相关子集的放松策略。值得注意的是，我们提出了一种称为自适应Pareto探索的单一抽样策略，可以与不同的停止规则结合使用，以考虑Pareto集合识别问题的不同放松策略。我们分析了这些不同组合的样本复杂度，并特别量化了在寻找识别最多$k$个Pareto最优臂时样本复杂度的减少。我们展示了自适应Pareto探索在一个真实场景中的良好实际性能，其中我们自适应地探索了几种疫苗接种策略。

    In this paper we revisit the fixed-confidence identification of the Pareto optimal set in a multi-objective multi-armed bandit model. As the sample complexity to identify the exact Pareto set can be very large, a relaxation allowing to output some additional near-optimal arms has been studied. In this work we also tackle alternative relaxations that allow instead to identify a relevant subset of the Pareto set. Notably, we propose a single sampling strategy, called Adaptive Pareto Exploration, that can be used in conjunction with different stopping rules to take into account different relaxations of the Pareto Set Identification problem. We analyze the sample complexity of these different combinations, quantifying in particular the reduction in sample complexity that occurs when one seeks to identify at most $k$ Pareto optimal arms. We showcase the good practical performance of Adaptive Pareto Exploration on a real-world scenario, in which we adaptively explore several vaccination stra
    
[^23]: 可证明高效的UCB类型算法用于学习预测状态表示

    Provably Efficient UCB-type Algorithms For Learning Predictive State Representations. (arXiv:2307.00405v1 [cs.LG])

    [http://arxiv.org/abs/2307.00405](http://arxiv.org/abs/2307.00405)

    这篇论文提出了第一种已知的UCB类型方法用于学习预测状态表示（PSRs），并设计了一个新的奖励项来上界t

    

    一般的顺序决策问题旨在通过基于过去观察和行动的历史来最大化累积奖励。最近的研究表明，如果顺序决策问题可以用预测状态表示（PSRs）建模低秩结构，那么它是可统计学习的。尽管有这些进展，但现有方法通常需要使用预先设计好的步骤或者是计算效率低下的或者是不可计算的。另一方面，上限置信区间（UCB）方法在赌博机和MDPs中被成功地作为计算效率高的方法，但对PSR这种更具挑战性的问题还没有进行研究，这是由于在这种更具挑战性的情况下，乐观型奖励的设计十分困难。本文提出了PSRs的第一种已知的UCB类型方法，其中包含了一个新的奖励项来上界t

    The general sequential decision-making problem, which includes Markov decision processes (MDPs) and partially observable MDPs (POMDPs) as special cases, aims at maximizing a cumulative reward by making a sequence of decisions based on a history of observations and actions over time. Recent studies have shown that the sequential decision-making problem is statistically learnable if it admits a low-rank structure modeled by predictive state representations (PSRs). Despite these advancements, existing approaches typically involve oracles or steps that are not computationally efficient. On the other hand, the upper confidence bound (UCB) based approaches, which have served successfully as computationally efficient methods in bandits and MDPs, have not been investigated for more general PSRs, due to the difficulty of optimistic bonus design in these more challenging settings. This paper proposes the first known UCB-type approach for PSRs, featuring a novel bonus term that upper bounds the t
    
[^24]: 使用组凹正则化的稀疏输入神经网络

    Sparse-Input Neural Network using Group Concave Regularization. (arXiv:2307.00344v1 [stat.ML])

    [http://arxiv.org/abs/2307.00344](http://arxiv.org/abs/2307.00344)

    本文提出了一种使用组凹正则化进行特征选择的稀疏输入神经网络框架，该框架能够在高维环境中选择重要的特征并保持稳定的解。

    

    同时进行特征选择和非线性函数估计在高维环境中是具有挑战性的，其中变量的数量超过了建模中可用的样本大小。在本文中，我们研究了神经网络中的特征选择问题。虽然组LASSO已经被用于神经网络的学习中选择变量，但它倾向于选择无关紧要的变量来弥补过度缩减的问题。为了克服这个限制，我们提出了一个稀疏输入神经网络框架，使用组凹正则化进行特征选择，适用于低维和高维设置。主要思想是对每个输入节点的所有出站连接的权重的l2范数应用适当的凹惩罚，从而得到一个只使用原始变量的一个小子集的神经网络。此外，我们基于向后路径优化开发了一个有效的算法来获得稳定的解。

    Simultaneous feature selection and non-linear function estimation are challenging, especially in high-dimensional settings where the number of variables exceeds the available sample size in modeling. In this article, we investigate the problem of feature selection in neural networks. Although the group LASSO has been utilized to select variables for learning with neural networks, it tends to select unimportant variables into the model to compensate for its over-shrinkage. To overcome this limitation, we propose a framework of sparse-input neural networks using group concave regularization for feature selection in both low-dimensional and high-dimensional settings. The main idea is to apply a proper concave penalty to the $l_2$ norm of weights from all outgoing connections of each input node, and thus obtain a neural net that only uses a small subset of the original variables. In addition, we develop an effective algorithm based on backward path-wise optimization to yield stable solutio
    
[^25]: 梯度相似：敏感度经常被过高估计在DP-SGD中

    Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD. (arXiv:2307.00310v1 [cs.LG])

    [http://arxiv.org/abs/2307.00310](http://arxiv.org/abs/2307.00310)

    本文开发了一种新的DP-SGD分析方法，可以在训练过程中对许多数据点的隐私泄漏进行更准确的评估。

    

    差分隐私随机梯度下降（DP-SGD）是私有深度学习的标准算法。虽然已知其隐私分析在最坏情况下是紧密的，但是一些实证结果表明，在常见的基准数据集上训练时，所得到的模型对许多数据点的隐私泄漏显著减少。在本文中，我们为DP-SGD开发了一种新的分析方法，捕捉到在数据集中具有相似邻居的点享受更好隐私性的直觉。形式上来说，这是通过修改从训练数据集计算得到的模型更新的每步隐私性分析来实现的。我们进一步开发了一个新的组合定理，以有效地利用这个新的每步分析来推理整个训练过程。总而言之，我们的评估结果表明，这种新颖的DP-SGD分析使我们能够正式地显示DP-SGD对许多数据点的隐私泄漏显著减少。

    Differentially private stochastic gradient descent (DP-SGD) is the canonical algorithm for private deep learning. While it is known that its privacy analysis is tight in the worst-case, several empirical results suggest that when training on common benchmark datasets, the models obtained leak significantly less privacy for many datapoints. In this paper, we develop a new analysis for DP-SGD that captures the intuition that points with similar neighbors in the dataset enjoy better privacy than outliers. Formally, this is done by modifying the per-step privacy analysis of DP-SGD to introduce a dependence on the distribution of model updates computed from a training dataset. We further develop a new composition theorem to effectively use this new per-step analysis to reason about an entire training run. Put all together, our evaluation shows that this novel DP-SGD analysis allows us to now formally show that DP-SGD leaks significantly less privacy for many datapoints. In particular, we ob
    
[^26]: 应用贝叶斯结构健康监测：测斜仪数据异常检测和预测

    Applied Bayesian Structural Health Monitoring: inclinometer data anomaly detection and forecasting. (arXiv:2307.00305v1 [cs.LG])

    [http://arxiv.org/abs/2307.00305](http://arxiv.org/abs/2307.00305)

    本文介绍了将贝叶斯技术应用于测斜仪数据的异常检测和预测，并且展示了如何通过量化和评估不确定性来最小化成本和风险。

    

    测斜仪探头是一种可以用来测量土方坡体变形的设备。本文展示了将贝叶斯技术应用于实际测斜仪数据的新颖方法，可以提供异常检测和预测功能。具体来说，本文详细介绍了对整个英国铁路网络中收集的测斜仪数据进行分析的情况。监测数据处理过程中，从业人员通常有两个目标，一是识别任何异常或危险的运动，二是通过预测来预测潜在未来的不良情况。在本文中，我们应用了不确定性量化（UQ）技术，通过实施贝叶斯方法，对测斜仪数据进行异常检测和预测。通过量化和评估适当的不确定性，可以最小化成本和风险。这个框架可以促进增强的决策和风险分析。我们展示了测斜仪数据可以被描述为

    Inclinometer probes are devices that can be used to measure deformations within earthwork slopes. This paper demonstrates a novel application of Bayesian techniques to real-world inclinometer data, providing both anomaly detection and forecasting. Specifically, this paper details an analysis of data collected from inclinometer data across the entire UK rail network.  Practitioners have effectively two goals when processing monitoring data. The first is to identify any anomalous or dangerous movements, and the second is to predict potential future adverse scenarios by forecasting. In this paper we apply Uncertainty Quantification (UQ) techniques by implementing a Bayesian approach to anomaly detection and forecasting for inclinometer data. Subsequently, both costs and risks may be minimised by quantifying and evaluating the appropriate uncertainties. This framework may then act as an enabler for enhanced decision making and risk analysis.  We show that inclinometer data can be described
    
[^27]: 基于自助法的交叉验证估计

    Bootstrapping the Cross-Validation Estimate. (arXiv:2307.00260v1 [stat.ME])

    [http://arxiv.org/abs/2307.00260](http://arxiv.org/abs/2307.00260)

    本文提出了一种快速自助法，可以快速估计交叉验证估计的标准误差，并为衡量平均模型性能的总体参数产生有效的置信区间。

    

    交叉验证是一种广泛应用于评估预测模型性能的技术。它可以避免对错误估计中的乐观偏差，尤其对于使用复杂统计学习算法构建的模型。然而，由于交叉验证估计是依赖于观测数据的随机值，因此准确量化估计的不确定性非常重要。特别是当使用交叉验证比较两个模型的性能时，必须确定错误估计的差异是否是由于偶然波动。尽管已经发展了各种方法来对交叉验证估计进行推断，但它们往往有许多限制，如严格的模型假设。本文提出了一种快速自助法，可以快速估计交叉验证估计的标准误差，并为衡量平均模型性能的总体参数产生有效的置信区间。

    Cross-validation is a widely used technique for evaluating the performance of prediction models. It helps avoid the optimism bias in error estimates, which can be significant for models built using complex statistical learning algorithms. However, since the cross-validation estimate is a random value dependent on observed data, it is essential to accurately quantify the uncertainty associated with the estimate. This is especially important when comparing the performance of two models using cross-validation, as one must determine whether differences in error estimates are a result of chance fluctuations. Although various methods have been developed for making inferences on cross-validation estimates, they often have many limitations, such as stringent model assumptions This paper proposes a fast bootstrap method that quickly estimates the standard error of the cross-validation estimate and produces valid confidence intervals for a population parameter measuring average model performance
    
[^28]: 高维线性回归的统一转移学习模型

    Unified Transfer Learning Models for High-Dimensional Linear Regression. (arXiv:2307.00238v1 [stat.ML])

    [http://arxiv.org/abs/2307.00238](http://arxiv.org/abs/2307.00238)

    UTrans是一种统一转移学习模型，它能检测可转移变量和源数据，并具有较低的估计和预测误差，同时保持可解释性。

    

    在现代数据分析中，当目标数据稀缺而源数据充足，或者源数据和目标数据的分布不同的情况下，转移学习在发挥重要作用。本文提出了一种可解释的统一转移学习模型，称为UTrans，该模型能够检测可转移变量和源数据。具体来说，我们建立了估计误差界限，并证明我们的界限低于仅有目标数据的界限。此外，我们基于假设检验提出了一种源数据检测算法，用于排除不可转移的数据。我们在多个实验中评估和比较了UTrans与现有算法。结果显示，UTrans在保持可解释性的同时，比现有方法具有更低的估计和预测误差。最后，我们将其应用于美国代际流动数据，并将我们提出的算法与经典的机器学习算法进行比较。

    Transfer learning plays a key role in modern data analysis when: (1) the target data are scarce but the source data are sufficient; (2) the distributions of the source and target data are heterogeneous. This paper develops an interpretable unified transfer learning model, termed as UTrans, which can detect both transferable variables and source data. More specifically, we establish the estimation error bounds and prove that our bounds are lower than those with target data only. Besides, we propose a source detection algorithm based on hypothesis testing to exclude the nontransferable data. We evaluate and compare UTrans to the existing algorithms in multiple experiments. It is shown that UTrans attains much lower estimation and prediction errors than the existing methods, while preserving interpretability. We finally apply it to the US intergenerational mobility data and compare our proposed algorithms to the classical machine learning algorithms.
    
[^29]: 利用马尔可夫毯交集进行因果结构学习

    Causal Structure Learning by Using Intersection of Markov Blankets. (arXiv:2307.00227v1 [stat.ML])

    [http://arxiv.org/abs/2307.00227](http://arxiv.org/abs/2307.00227)

    本文提出了一种新颖的因果结构学习算法，该算法利用马尔可夫毯交集，并结合了贝叶斯网络和结构因果模型的特性。此外，还提出了EEMBI-PC，它是EEMBI的扩展版本，将PC算法的最后一步集成到EEMBI中。

    

    在本文中，我们介绍了一种新颖的因果结构学习算法，称为内源和外源马尔可夫毯交集（EEMBI），它结合了贝叶斯网络和结构因果模型（SCM）的特性。此外，我们提出了EEMBI的扩展版本，即EEMBI-PC，它将PC算法的最后一步集成到EEMBI中。

    In this paper, we introduce a novel causal structure learning algorithm called Endogenous and Exogenous Markov Blankets Intersection (EEMBI), which combines the properties of Bayesian networks and Structural Causal Models (SCM). Furthermore, we propose an extended version of EEMBI, namely EEMBI-PC, which integrates the last step of the PC algorithm into EEMBI.
    
[^30]: 平衡方法对不平衡分类问题中模型行为的影响

    The Effect of Balancing Methods on Model Behavior in Imbalanced Classification Problems. (arXiv:2307.00157v1 [cs.LG])

    [http://arxiv.org/abs/2307.00157](http://arxiv.org/abs/2307.00157)

    平衡方法对不平衡分类问题中模型行为产生显著影响。这些发现强调了平衡分析在模型训练中的重要性。

    

    不平衡数据对分类问题构成了重要的挑战，因为模型性能受到对少数类别学习不足的影响。平衡方法通常被用来解决这个问题。然而，这些技术可能会导致过拟合或者信息丢失等问题。本研究探讨了平衡方法更具挑战性的方面——它们对模型行为的影响。为了捕捉这些变化，本研究使用了可解释人工智能工具来比较在平衡前后训练的模型。除了变量重要性方法外，本研究还使用了部分依赖轮廓和累积局部影响技术。进行了真实和模拟数据集的测试，并开发了一个开源Python包edgaro来方便进行这种分析。所得到的结果显示，由于平衡方法的影响，模型行为发生了显著变化，可能会使模型对平衡分布产生偏见。这些发现证实了平衡分析对模型行为有重要影响。

    Imbalanced data poses a significant challenge in classification as model performance is affected by insufficient learning from minority classes. Balancing methods are often used to address this problem. However, such techniques can lead to problems such as overfitting or loss of information. This study addresses a more challenging aspect of balancing methods - their impact on model behavior. To capture these changes, Explainable Artificial Intelligence tools are used to compare models trained on datasets before and after balancing. In addition to the variable importance method, this study uses the partial dependence profile and accumulated local effects techniques. Real and simulated datasets are tested, and an open-source Python package edgaro is developed to facilitate this analysis. The results obtained show significant changes in model behavior due to balancing methods, which can lead to biased models toward a balanced distribution. These findings confirm that balancing analysis sh
    
[^31]: 高维贝叶斯高斯图模型中的结构学习方法——利用边际伪似然函数

    High-Dimensional Bayesian Structure Learning in Gaussian Graphical Models using Marginal Pseudo-Likelihood. (arXiv:2307.00127v1 [stat.ME])

    [http://arxiv.org/abs/2307.00127](http://arxiv.org/abs/2307.00127)

    该论文提出了两种创新的搜索算法，在高维图结构学习中使用边际伪似然函数解决计算复杂性问题，并且能够在短时间内生成可靠的估计。该方法提供了R软件包BDgraph的代码实现。

    

    高斯图模型以图形形式描绘了多元正态分布中变量之间的条件依赖关系。这篇论文介绍了两种创新的搜索算法，利用边际伪似然函数来应对高维图结构学习中的计算复杂性问题。这些方法可以在标准计算机上在几分钟内快速生成对包含1000个变量的问题的可靠估计。对于对实际应用感兴趣的人，支持这种新方法的代码通过R软件包BDgraph提供。

    Gaussian graphical models depict the conditional dependencies between variables within a multivariate normal distribution in a graphical format. The identification of these graph structures is an area known as structure learning. However, when utilizing Bayesian methodologies in structure learning, computational complexities can arise, especially with high-dimensional graphs surpassing 250 nodes. This paper introduces two innovative search algorithms that employ marginal pseudo-likelihood to address this computational challenge. These methods can swiftly generate reliable estimations for problems encompassing 1000 variables in just a few minutes on standard computers. For those interested in practical applications, the code supporting this new approach is made available through the R package BDgraph.
    
[^32]: 加速非精确超梯度下降用于双层优化

    Accelerating Inexact HyperGradient Descent for Bilevel Optimization. (arXiv:2307.00126v1 [math.OC])

    [http://arxiv.org/abs/2307.00126](http://arxiv.org/abs/2307.00126)

    提出一种加速非精确超梯度下降的方法用于双层优化，可以在较低的复杂度下找到一阶和二阶稳定点，成为双层优化和凸-凹极小极大优化问题中的最新最佳状态。

    

    我们提出了一种解决一般非凸-凸双层优化问题的方法。我们的方法——"重新启动的加速超梯度下降" (RAHGD) 方法——可以找到一个 $\epsilon$-一阶稳定点，其 oracle 复杂度为 $\tilde{\mathcal{O}}(\kappa^{3.25}\epsilon^{-1.75})$，其中 $\kappa$ 是下层目标的条件数，$\epsilon$ 是期望精度。我们还提出了 RAHGD 的扰动变体，用于在相同的 oracle 复杂度下找到一个 $\big(\epsilon,\mathcal{O}(\kappa^{2.5}\sqrt{\epsilon}\,)\big)$-二阶稳定点。我们的结果在双层优化中实现了已知最好的理论保证，并且改进了现有凸-凹极小极大优化问题中找到二阶稳定点的上界复杂度，为最新的基准设置了一个新的最佳状态。我们进行了实证研究。

    We present a method for solving general nonconvex-strongly-convex bilevel optimization problems. Our method -- the \emph{Restarted Accelerated HyperGradient Descent} (\texttt{RAHGD}) method -- finds an $\epsilon$-first-order stationary point of the objective with $\tilde{\mathcal{O}}(\kappa^{3.25}\epsilon^{-1.75})$ oracle complexity, where $\kappa$ is the condition number of the lower-level objective and $\epsilon$ is the desired accuracy. We also propose a perturbed variant of \texttt{RAHGD} for finding an $\big(\epsilon,\mathcal{O}(\kappa^{2.5}\sqrt{\epsilon}\,)\big)$-second-order stationary point within the same order of oracle complexity. Our results achieve the best-known theoretical guarantees for finding stationary points in bilevel optimization and also improve upon the existing upper complexity bound for finding second-order stationary points in nonconvex-strongly-concave minimax optimization problems, setting a new state-of-the-art benchmark. Empirical studies are conducted t
    
[^33]: 用于物理科学家的基于数据驱动先验的近端嵌套抽样方法

    Proximal nested sampling with data-driven priors for physical scientists. (arXiv:2307.00056v1 [stat.ME])

    [http://arxiv.org/abs/2307.00056](http://arxiv.org/abs/2307.00056)

    近端嵌套抽样方法允许物理科学家应用贝叶斯模型选择于高维问题中，并展示了如何通过数据驱动先验的支持来扩展该方法。

    

    最近引入了近端嵌套抽样方法，以开辟贝叶斯模型选择在高维问题中的应用，例如计算成像。该框架适用于具有对数凸似然函数的模型，这在成像科学中非常普遍。本文有两个目的。首先，我们以教学的方式对近端嵌套抽样方法进行综述，以努力为物理科学家解释该框架。其次，我们展示了近端嵌套抽样方法如何在经验贝叶斯设置中扩展，以支持数据驱动的先验，如从训练数据中学习的深度神经网络。

    Proximal nested sampling was introduced recently to open up Bayesian model selection for high-dimensional problems such as computational imaging. The framework is suitable for models with a log-convex likelihood, which are ubiquitous in the imaging sciences. The purpose of this article is two-fold. First, we review proximal nested sampling in a pedagogical manner in an attempt to elucidate the framework for physical scientists. Second, we show how proximal nested sampling can be extended in an empirical Bayes setting to support data-driven priors, such as deep neural networks learned from training data.
    
[^34]: 使用归一化流学习边缘似然的调和平均估计

    Learned harmonic mean estimation of the marginal likelihood with normalizing flows. (arXiv:2307.00048v1 [stat.ME])

    [http://arxiv.org/abs/2307.00048](http://arxiv.org/abs/2307.00048)

    本文研究使用归一化流学习边缘似然的调和平均估计，在贝叶斯模型选择中解决了原始方法中的方差爆炸问题。

    

    计算边缘似然（也称为贝叶斯模型证据）是贝叶斯模型选择中的一项重要任务，它提供了一种有原则的定量比较模型的方法。学习的调和平均估计器解决了原始调和平均估计边缘似然的方差爆炸问题。学习的调和平均估计器学习了一个重要性采样目标分布，该分布近似于最优分布。虽然近似不必非常准确，但确保学习分布的概率质量包含在后验分布中是至关重要的，以避免方差爆炸问题。在先前的工作中，为了确保满足这个性质，在训练模型时引入了一种专门的优化问题。在本文中，我们引入了使用归一化流来表示重要性采样目标分布。基于流的模型通过最大似然从后验样本中进行训练。

    Computing the marginal likelihood (also called the Bayesian model evidence) is an important task in Bayesian model selection, providing a principled quantitative way to compare models. The learned harmonic mean estimator solves the exploding variance problem of the original harmonic mean estimation of the marginal likelihood. The learned harmonic mean estimator learns an importance sampling target distribution that approximates the optimal distribution. While the approximation need not be highly accurate, it is critical that the probability mass of the learned distribution is contained within the posterior in order to avoid the exploding variance problem. In previous work a bespoke optimization problem is introduced when training models in order to ensure this property is satisfied. In the current article we introduce the use of normalizing flows to represent the importance sampling target distribution. A flow-based model is trained on samples from the posterior by maximum likelihood e
    
[^35]: TemperatureGAN: 区域大气温度的生成建模

    TemperatureGAN: Generative Modeling of Regional Atmospheric Temperatures. (arXiv:2306.17248v1 [cs.LG])

    [http://arxiv.org/abs/2306.17248](http://arxiv.org/abs/2306.17248)

    TemperatureGAN是一个生成对抗网络，使用地面以上2m的大气温度数据，能够生成具有良好空间表示和与昼夜周期一致的时间动态的高保真样本。

    

    随机生成器对于估计气候对各个领域的影响非常有用。在各个领域中进行气候风险的预测，例如能源系统，需要准确（与基准真实数据有统计相似性）、可靠（不产生错误样本）和高效的生成器。我们利用来自北美陆地数据同化系统的数据，引入了TemperatureGAN，这是一个以月份、位置和时间段为条件的生成对抗网络，以每小时分辨率生成地面以上2m的大气温度。我们提出了评估方法和指标来衡量生成样本的质量。我们证明TemperatureGAN能够生成具有良好空间表示和与已知昼夜周期一致的时间动态的高保真样本。

    Stochastic generators are useful for estimating climate impacts on various sectors. Projecting climate risk in various sectors, e.g. energy systems, requires generators that are accurate (statistical resemblance to ground-truth), reliable (do not produce erroneous examples), and efficient. Leveraging data from the North American Land Data Assimilation System, we introduce TemperatureGAN, a Generative Adversarial Network conditioned on months, locations, and time periods, to generate 2m above ground atmospheric temperatures at an hourly resolution. We propose evaluation methods and metrics to measure the quality of generated samples. We show that TemperatureGAN produces high-fidelity examples with good spatial representation and temporal dynamics consistent with known diurnal cycles.
    
[^36]: 基于共生学习的异方差回归的近似最优算法研究

    Near Optimal Heteroscedastic Regression with Symbiotic Learning. (arXiv:2306.14288v1 [stat.ML])

    [http://arxiv.org/abs/2306.14288](http://arxiv.org/abs/2306.14288)

    本研究提出了一种基于共生学习的异方差回归的近似最优算法，可以在统计学、计量经济学、时间序列分析等领域，以及在不同来源数据质量不一的机器学习中应用。

    

    本研究针对经典的异方差线性回归问题展开讨论。假设我们有n个样本 $(\mathbf{x}_i, y_i) \in \mathbb{R}^d \times \mathbb{R}$，其中 $y_i = \langle \mathbf{w}^{*}, \mathbf{x}_i \rangle + \epsilon_i \cdot \langle \mathbf{f}^{*}, \mathbf{x}_i \rangle$， $\mathbf{x}_i \sim N(0,\mathbf{I})$，$\epsilon_i \sim N(0,1)$，我们的目标是估计 $\mathbf{w}^{*}$。在统计学、计量经济学、时间序列分析等领域，异方差模型具有广泛的应用，同时，在机器学习中如果数据来源不同，而不同来源的数据质量也不一，则异方差模型也显得特别相关。本研究表明，我们可以估计出$\mathbf{w}^{*}$的平方范数，误差为$\tilde{O}\left(\|\mathbf{f}^{*}\|^2 \cdot \left(\frac{1}{n} + \left(\frac{d}{n}\right)^2\right)\right)$，并证明了一个匹配的下限（上界存在对数因子）。本研究的结果显著改进了异方差回归问题的近似最优算法。

    We consider the classical problem of heteroscedastic linear regression, where we are given $n$ samples $(\mathbf{x}_i, y_i) \in \mathbb{R}^d \times \mathbb{R}$ obtained from $y_i = \langle \mathbf{w}^{*}, \mathbf{x}_i \rangle + \epsilon_i \cdot \langle \mathbf{f}^{*}, \mathbf{x}_i \rangle$, where $\mathbf{x}_i \sim N(0,\mathbf{I})$, $\epsilon_i \sim N(0,1)$, and our task is to estimate $\mathbf{w}^{*}$. In addition to the classical applications of heteroscedastic models in fields such as statistics, econometrics, time series analysis etc., it is also particularly relevant in machine learning when data is collected from multiple sources of varying but apriori unknown quality, e.g., large model training. Our work shows that we can estimate $\mathbf{w}^{*}$ in squared norm up to an error of $\tilde{O}\left(\|\mathbf{f}^{*}\|^2 \cdot \left(\frac{1}{n} + \left(\frac{d}{n}\right)^2\right)\right)$ and prove a matching lower bound (up to logarithmic factors). Our result substantially improves 
    
[^37]: 经验熵正则化最优输运的低复杂度适应性

    Lower Complexity Adaptation for Empirical Entropic Optimal Transport. (arXiv:2306.13580v1 [math.ST])

    [http://arxiv.org/abs/2306.13580](http://arxiv.org/abs/2306.13580)

    本文研究了经验熵正则化最优输运的统计表现，并证明了它遵循低复杂度适应原则，推导出了其统计界限及参数化速率。

    

    经验熵正则化最优输运 (EOT) 是优化输运 (OT) 的一种有效且计算可行的替代方案，对大规模数据分析有着广泛的应用。本文推导出了 EOT 成本的新的统计界限，并显示它们在熵正则化参数 $\epsilon$ 和样本大小 $n$ 的统计性能仅取决于两个概率测度之中较简单的那个。例如，在充分平滑的成本下，这会产生具有$\epsilon^{-d/2}$因子的参数化速率$n^{-1/2}$，其中$d$是两个总体测度的最小维度。这确认了经验EOT也遵循了最近才为未规则化OT确认的低复杂度适应原则的标志性特征。根据我们的理论，我们展示了欧几里得空间上的测度的经验熵Gromov-Wasserstein距离及其未规则化版本也遵循此原则。

    Entropic optimal transport (EOT) presents an effective and computationally viable alternative to unregularized optimal transport (OT), offering diverse applications for large-scale data analysis. In this work, we derive novel statistical bounds for empirical plug-in estimators of the EOT cost and show that their statistical performance in the entropy regularization parameter $\epsilon$ and the sample size $n$ only depends on the simpler of the two probability measures. For instance, under sufficiently smooth costs this yields the parametric rate $n^{-1/2}$ with factor $\epsilon^{-d/2}$, where $d$ is the minimum dimension of the two population measures. This confirms that empirical EOT also adheres to the lower complexity adaptation principle, a hallmark feature only recently identified for unregularized OT. As a consequence of our theory, we show that the empirical entropic Gromov-Wasserstein distance and its unregularized version for measures on Euclidean spaces also obey this princip
    
[^38]: 在线重尾变点检测

    Online Heavy-tailed Change-point detection. (arXiv:2306.09548v1 [stat.ML])

    [http://arxiv.org/abs/2306.09548](http://arxiv.org/abs/2306.09548)

    本文提出了一种在线变点检测算法，可以应对重尾分布且保证有限的假阳性率。

    

    我们研究了在线变点检测 (OCPD) 的算法，其中样本可能是重尾分布，一个接一个地呈现，并且必须尽早检测到底层均值的变化。我们提出了一种基于裁剪随机梯度下降 (SGD) 的算法，即使我们仅假定数据生成过程的第二阶矩有界，该算法也能正常工作。我们派生了在所有具有有界第二矩的分布族中最坏情况下的有限样本假阳性率 (FPR) 的保证。因此，我们的方法是第一个保证有限样本 FPR 的 OCPD 算法，即使数据是高维的，底层分布是重尾的。我们论文的技术贡献是展示了裁剪 SGD 可以估计随机向量的均值并同时在所有置信度值上提供置信度界限。我们将这个稳健的估计与并集边界论证相结合，构建一个有限的顺序变点算法。

    We study algorithms for online change-point detection (OCPD), where samples that are potentially heavy-tailed, are presented one at a time and a change in the underlying mean must be detected as early as possible. We present an algorithm based on clipped Stochastic Gradient Descent (SGD), that works even if we only assume that the second moment of the data generating process is bounded. We derive guarantees on worst-case, finite-sample false-positive rate (FPR) over the family of all distributions with bounded second moment. Thus, our method is the first OCPD algorithm that guarantees finite-sample FPR, even if the data is high dimensional and the underlying distributions are heavy-tailed. The technical contribution of our paper is to show that clipped-SGD can estimate the mean of a random vector and simultaneously provide confidence bounds at all confidence values. We combine this robust estimate with a union bound argument and construct a sequential change-point algorithm with finite
    
[^39]: 通过不偏微分对抗复杂密度生成，基于互联马尔科夫链不偏微分优化 MH 采样方法

    Differentiating Metropolis-Hastings to Optimize Intractable Densities. (arXiv:2306.07961v1 [stat.ML])

    [http://arxiv.org/abs/2306.07961](http://arxiv.org/abs/2306.07961)

    本文通过基于互联马尔科夫链的不偏微分，开发出一种无偏、低方差和自动的方法对复杂密度进行生成，从而实现对 MH 采样器的优化。

    

    在概率模型推理中，目标密度函数通常变得难以计算，需要使用 Monte Carlo 计算。本文开发了一种不偏微分 Metropolis-Hastings 采样器的方法，使我们可以通过概率推理来进行微分。通过将随机微分的最新进展与 Markov 链耦合方法相结合，可以实现无偏，低方差和自动的程序。这使我们能够将基于梯度的优化应用于由于繁琐的目标密度导致期望的情况下。我们通过在高斯混合模型中找到一个模棱两可的观察和在 Ising 模型中最大化比热来演示了我们的方法。

    When performing inference on probabilistic models, target densities often become intractable, necessitating the use of Monte Carlo samplers. We develop a methodology for unbiased differentiation of the Metropolis-Hastings sampler, allowing us to differentiate through probabilistic inference. By fusing recent advances in stochastic differentiation with Markov chain coupling schemes, the procedure can be made unbiased, low-variance, and automatic. This allows us to apply gradient-based optimization to objectives expressed as expectations over intractable target densities. We demonstrate our approach by finding an ambiguous observation in a Gaussian mixture model and by maximizing the specific heat in an Ising model.
    
[^40]: 基于分布特征匹配的标签偏移量量化及其鲁棒性保证

    Label Shift Quantification with Robustness Guarantees via Distribution Feature Matching. (arXiv:2306.04376v1 [stat.ML])

    [http://arxiv.org/abs/2306.04376](http://arxiv.org/abs/2306.04376)

    本文提出一种名为DFM框架的方法，用于量化标签偏移，并证明了其性能上限和鲁棒性。使用基于核的DFM版本可以提高效率、可扩展性和鲁棒性。

    

    量化学习处理在标签偏移下估计目标标签分布的任务。本文首先提出了一个统一的框架，分布特征匹配（DFM），将先前文献中引入的各种估计器恢复为特定实例。我们推导了DFM程序的一般性能界，改进了先前在特定情况下推导的界限的若干关键方面。然后，我们将这一分析扩展到研究DFM程序在未精确假设标签偏移量的情况下的鲁棒性，特别是在目标受到未知分布污染的情况下。这些理论发现在模拟和实际数据集上得到了详细的数字研究确认。我们还使用随机傅里叶特征原理介绍了一种高效，可扩展且具有鲁棒性的基于核的DFM版本。

    Quantification learning deals with the task of estimating the target label distribution under label shift. In this paper, we first present a unifying framework, distribution feature matching (DFM), that recovers as particular instances various estimators introduced in previous literature. We derive a general performance bound for DFM procedures, improving in several key aspects upon previous bounds derived in particular cases. We then extend this analysis to study robustness of DFM procedures in the misspecified setting under departure from the exact label shift hypothesis, in particular in the case of contamination of the target by an unknown distribution. These theoretical findings are confirmed by a detailed numerical study on simulated and real-world datasets. We also introduce an efficient, scalable and robust version of kernel-based DFM using the Random Fourier Feature principle.
    
[^41]: 熵协方差模型

    Entropic covariance models. (arXiv:2306.03590v1 [math.ST])

    [http://arxiv.org/abs/2306.03590](http://arxiv.org/abs/2306.03590)

    本文提出了一个通用的线性约束协方差矩阵变换的框架，并提出了一种估计方法，解决了一个凸问题，允许相对简单的渐近性和有限样本分析。研究的重点是关于建模相关矩阵和稀疏性方面的内容。

    

    在协方差矩阵估计中，找到合适的模型和有效的估计方法是一项挑战。文献中通常采用两种方法，一种是对协方差矩阵或其逆施加线性约束，另一种是考虑施加在协方差矩阵的矩阵对数上的线性约束。本文提出了一个通用的线性约束协方差矩阵变换的框架，包括上述例子。我们提出的估计方法解决了一个凸问题，并产生了一个M估计量，允许相对简单的渐近性和有限样本分析。在开发了一般理论之后，我们集中在建模相关矩阵和稀疏性方面。我们的几何洞察力允许我们扩展协方差矩阵建模中的一些最新结果。这包括提供相关矩阵空间的无限制参数化，这是一种替代利用变换的最新结果。我们还展示了如何对协方差矩阵的Cholesky因子施加稀疏性限制，这与现有方法不同。

    In covariance matrix estimation, one of the challenges lies in finding a suitable model and an efficient estimation method. Two commonly used approaches in the literature involve imposing linear restrictions on the covariance matrix or its inverse. Another approach considers linear restrictions on the matrix logarithm of the covariance matrix. In this paper, we present a general framework for linear restrictions on different transformations of the covariance matrix, including the mentioned examples. Our proposed estimation method solves a convex problem and yields an M-estimator, allowing for relatively straightforward asymptotic and finite sample analysis. After developing the general theory, we focus on modelling correlation matrices and on sparsity. Our geometric insights allow to extend various recent results in covariance matrix modelling. This includes providing unrestricted parametrizations of the space of correlation matrices, which is alternative to a recent result utilizing t
    
[^42]: MACE力场架构的评估：从药物化学到材料科学

    Evaluation of the MACE Force Field Architecture: from Medicinal Chemistry to Materials Science. (arXiv:2305.14247v1 [physics.chem-ph])

    [http://arxiv.org/abs/2305.14247](http://arxiv.org/abs/2305.14247)

    MACE的机器学习力场架构在内域、外推和低数据范围任务中表现优秀，在处理非晶碳、小分子有机化学、大分子和液态水等领域时常常优于其他替代方案。即使只有50个随机选定的参考配置，该模型也能非常高效地复现实验分子振动光谱。

    

    MACE架构代表了机器学习力场在各种领域中的最新技术，能够处理内域、外推和低数据范围任务。本文对MACE进行了进一步评估，通过拟合已发表的基准数据集的模型来表明MACE在各种体系中的性能优于其他替代方案，包括非晶碳、一般的小分子有机化学、大分子和液态水。我们展示了模型在各个领域的能力，从约束几何优化到分子动力学模拟，发现其在所有测试领域都具有出色的性能。我们还表明，当在仅50个随机选定的参考配置上进行训练时，MACE即可非常高效地复现实验分子振动光谱。我们进一步证明，即使在大分子和弱相互作用的分子组装的情况下，这种基于严格局部的原子中心模型也是足够适用的。

    The MACE architecture represents the state of the art in the field of machine learning force fields for a variety of in-domain, extrapolation and low-data regime tasks. In this paper, we further evaluate MACE by fitting models for published benchmark datasets. We show that MACE generally outperforms alternatives for a wide range of systems from amorphous carbon and general small molecule organic chemistry to large molecules and liquid water. We demonstrate the capabilities of the model on tasks ranging from constrained geometry optimisation to molecular dynamics simulations and find excellent performance across all tested domains. We show that MACE is very data efficient, and can reproduce experimental molecular vibrational spectra when trained on as few as 50 randomly selected reference configurations. We further demonstrate that the strictly local atom-centered model is sufficient for such tasks even in the case of large molecules and weakly interacting molecular assemblies.
    
[^43]: 关于洛克斯洞穴的流形学习：关于流形学习和物理现象的评论（arXiv:2304.14248v1 [stat.ML]）

    On Manifold Learning in Plato's Cave: Remarks on Manifold Learning and Physical Phenomena. (arXiv:2304.14248v1 [stat.ML])

    [http://arxiv.org/abs/2304.14248](http://arxiv.org/abs/2304.14248)

    本文通过一个警示故事阐释了分析数据时，测量几何和底层现象几何差异带来的问题，以及这种差异在某些情况下如何导致对一个修正过的问题给出错误答案。这些问题适用于降维和无监督学习领域。

    

    许多机器学习技术尝试通过测量不需要对物理现象或测量设备进行显式建模的低维流形结构来推断潜在物理现象的低维流形结构，这篇论文提出了关于测量几何和底层现象几何之间差异的警示故事。在普通情况下，这篇论文所展示的度量形变在数学上是直接而不可避免的，并且它只是数个类似效应中的一个。虽然这并不总是出现问题，但我们提供了一个标准且无害数据处理过程的例子，其中这种影响导致对一个看似简单的问题给出了错误的答案。尽管我们关注流形学习，但这些问题广泛适用于降维和无监督学习领域。

    Many techniques in machine learning attempt explicitly or implicitly to infer a low-dimensional manifold structure of an underlying physical phenomenon from measurements without an explicit model of the phenomenon or the measurement apparatus. This paper presents a cautionary tale regarding the discrepancy between the geometry of measurements and the geometry of the underlying phenomenon in a benign setting. The deformation in the metric illustrated in this paper is mathematically straightforward and unavoidable in the general case, and it is only one of several similar effects. While this is not always problematic, we provide an example of an arguably standard and harmless data processing procedure where this effect leads to an incorrect answer to a seemingly simple question. Although we focus on manifold learning, these issues apply broadly to dimensionality reduction and unsupervised learning.
    
[^44]: 为一致性预测的后选推理：权衡精度和覆盖范围

    Post-selection Inference for Conformal Prediction: Trading off Coverage for Precision. (arXiv:2304.06158v1 [stat.ME])

    [http://arxiv.org/abs/2304.06158](http://arxiv.org/abs/2304.06158)

    本论文提出一种使用无分布信赖带的 uniform conformal inference 算法，实现任意数据相关误覆盖水平的有限样本预测保证的统一一致性推理。

    

    一致性推理在为具有有限样本保证的黑盒机器学习预测算法提供不确定性量化上发挥了重要作用。传统上，一致性预测推理需要独立于数据的错误覆盖水平规范。在实际应用中，人们可能会在计算出预测集之后更新错误覆盖水平。例如，在二元分类的情况下，分析人员可能会从一个95％的预测集开始，并发现大多数预测集包含所有输出类别。如果两个类别都不可取，分析人员可能会考虑80％的预测集。具有数据相关的误覆盖水平和保证覆盖范围的预测集的构建可以被认为是一个后选推理问题。在这项工作中，我们使用无分布信赖带，开发了具有任意数据相关误覆盖水平的有限样本预测保证的统一一致性推理。

    Conformal inference has played a pivotal role in providing uncertainty quantification for black-box ML prediction algorithms with finite sample guarantees. Traditionally, conformal prediction inference requires a data-independent specification of miscoverage level. In practical applications, one might want to update the miscoverage level after computing the prediction set. For example, in the context of binary classification, the analyst might start with a $95\%$ prediction sets and see that most prediction sets contain all outcome classes. Prediction sets with both classes being undesirable, the analyst might desire to consider, say $80\%$ prediction set. Construction of prediction sets that guarantee coverage with data-dependent miscoverage level can be considered as a post-selection inference problem. In this work, we develop uniform conformal inference with finite sample prediction guarantee with arbitrary data-dependent miscoverage levels using distribution-free confidence bands f
    
[^45]: 隐式平衡和正则化：过参数化非对称矩阵感知中的泛化和收敛保证

    Implicit Balancing and Regularization: Generalization and Convergence Guarantees for Overparameterized Asymmetric Matrix Sensing. (arXiv:2303.14244v1 [cs.LG])

    [http://arxiv.org/abs/2303.14244](http://arxiv.org/abs/2303.14244)

    本论文研究了过参数化低秩矩阵感知问题，证明了通过因子化方法训练的过参数化模型可以收敛，并且隐式平衡和正则化可以促进泛化。

    

    最近，对于训练过参数化学习模型的基于梯度的方法的收敛和泛化属性有了重要进展。然而，其中许多方面，包括小随机初始化的角色以及模型的各种参数在梯度更新中如何耦合以促进良好的泛化，仍然是很神秘的。最近一系列的论文已经开始研究非凸对称半正定（PSD）矩阵感知问题的形式，在这个问题中需要从几个线性测量中重建一个低秩PSD矩阵。这种底层的对称性/PSD性对于现有的这个问题的收敛和泛化保证是至关重要的。在本文中，我们研究了一个一般的过参数化的低秩矩阵感知问题，其中希望从少量的线性测量中重建一个非对称矩形低秩矩阵。我们证明了通过因子化来训练的过参数化模型在这个问题上可以收敛，而隐式平衡和正则化可以促进泛化。

    Recently, there has been significant progress in understanding the convergence and generalization properties of gradient-based methods for training overparameterized learning models. However, many aspects including the role of small random initialization and how the various parameters of the model are coupled during gradient-based updates to facilitate good generalization remain largely mysterious. A series of recent papers have begun to study this role for non-convex formulations of symmetric Positive Semi-Definite (PSD) matrix sensing problems which involve reconstructing a low-rank PSD matrix from a few linear measurements. The underlying symmetry/PSDness is crucial to existing convergence and generalization guarantees for this problem. In this paper, we study a general overparameterized low-rank matrix sensing problem where one wishes to reconstruct an asymmetric rectangular low-rank matrix from a few linear measurements. We prove that an overparameterized model trained via factori
    
[^46]: 一种适用于核心测试的U统计的高维收敛定理

    A High-dimensional Convergence Theorem for U-statistics with Applications to Kernel-based Testing. (arXiv:2302.05686v3 [math.ST] UPDATED)

    [http://arxiv.org/abs/2302.05686](http://arxiv.org/abs/2302.05686)

    本论文证明了一个适用于核心测试的高维U统计的收敛定理，并发现U统计的极限分布会经历从非退化高斯极限到退化极限的相变。这一现象对于高维情况下的非退化U统计具有较大方差和不对称分布的非高斯极限具有重要意义。此外，我们提出的界限适用于任何有限数量和维度的样本，与底层函数的特征值无关，并且在某些假设下与维度无关。我们还将我们的理论应用到两个常用的基于核函数的分布测试方法，MMD和KSD，来研究它们的高维性能。我们的结果能够准确预测测试功率如何与维度和带宽的关系。

    

    我们证明了一个U统计的二次收敛定理，其中数据维度$d$可以随样本大小$n$的变化而变化。我们发现，一个U统计的极限分布会经历从非退化高斯极限到退化极限的相变，不论其退化性如何，只取决于一个矩比率。一个令人惊讶的结果是，在高维情况下，一个非退化的U统计可能具有一个具有较大方差和不对称分布的非高斯极限。我们的界限对任何有限的$n$和$d$都是有效的，与底层函数的个别特征值无关，并且在一个适度的假设下与维度无关。作为应用，我们将我们的理论应用到两个流行的基于核心的分布测试，MMD和KSD上，这些测试在高维性能的研究一直是有挑战性的。在一个简单的经验设置中，我们的结果正确地预测了在固定阈值下测试功率如何随着$d$和带宽的缩放。

    We prove a convergence theorem for U-statistics of degree two, where the data dimension $d$ is allowed to scale with sample size $n$. We find that the limiting distribution of a U-statistic undergoes a phase transition from the non-degenerate Gaussian limit to the degenerate limit, regardless of its degeneracy and depending only on a moment ratio. A surprising consequence is that a non-degenerate U-statistic in high dimensions can have a non-Gaussian limit with a larger variance and asymmetric distribution. Our bounds are valid for any finite $n$ and $d$, independent of individual eigenvalues of the underlying function, and dimension-independent under a mild assumption. As an application, we apply our theory to two popular kernel-based distribution tests, MMD and KSD, whose high-dimensional performance has been challenging to study. In a simple empirical setting, our results correctly predict how the test power at a fixed threshold scales with $d$ and the bandwidth.
    
[^47]: 密集式Hebbian神经网络：监督学习的对称图片

    Dense Hebbian neural networks: a replica symmetric picture of supervised learning. (arXiv:2212.00606v2 [cond-mat.dis-nn] UPDATED)

    [http://arxiv.org/abs/2212.00606](http://arxiv.org/abs/2212.00606)

    这篇论文研究了通过教师训练的密集Hebbian神经网络的计算能力，通过统计力学和蒙特卡罗模拟得到了一个相图，指出这些网络在大规模和结构简单的数据集下可以在超大存储或超高检测区域工作。

    

    我们考虑由教师（即监督学习）训练的密集的关联神经网络，并通过自旋玻璃的统计力学分析和蒙特卡罗模拟来研究它们的计算能力。特别地，我们得到了一个相图，总结了它们的性能如训练数据集的质量和数量、网络存储和噪声等控制参数的函数，这在网络尺寸大、数据集结构简单的极限下是有效的：这些网络可以在超大存储区域工作（与浅层神经网络相比，它们可以处理大量的模式），或者在超高检测区域工作（与浅层神经网络相比，它们可以在极低的信噪比下进行模式识别）。在以随机理论作为参考框架的指导下，我们还对这些网络在结构化数据集（如MNist）上展示的学习、存储和检索能力进行了数值测试。

    We consider dense, associative neural-networks trained by a teacher (i.e., with supervision) and we investigate their computational capabilities analytically, via statistical-mechanics of spin glasses, and numerically, via Monte Carlo simulations. In particular, we obtain a phase diagram summarizing their performance as a function of the control parameters such as quality and quantity of the training dataset, network storage and noise, that is valid in the limit of large network size and structureless datasets: these networks may work in a ultra-storage regime (where they can handle a huge amount of patterns, if compared with shallow neural networks) or in a ultra-detection regime (where they can perform pattern recognition at prohibitive signal-to-noise ratios, if compared with shallow neural networks). Guided by the random theory as a reference framework, we also test numerically learning, storing and retrieval capabilities shown by these networks on structured datasets as MNist and 
    
[^48]: 密集的贺维模型神经网络：无监督学习的对称副本描述

    Dense Hebbian neural networks: a replica symmetric picture of unsupervised learning. (arXiv:2211.14067v2 [cond-mat.dis-nn] UPDATED)

    [http://arxiv.org/abs/2211.14067](http://arxiv.org/abs/2211.14067)

    本文研究了无监督训练的密集贺维神经网络，并通过统计力学方法和蒙特卡洛模拟分析了其计算能力。我们得到了一个相图，总结了网络性能与训练数据集质量、数量和网络存储之间的关系，并建立了统计力学中的宏观可观测量与机器学习中的损失函数的联系。

    

    我们研究了无监督训练的密集关联神经网络，并通过统计力学方法和蒙特卡洛模拟进行了计算能力的分析。特别地，我们在大网络规模和无结构数据集的极限情况下获得了一个相图，总结了网络性能与训练数据集的质量、数量以及网络存储等控制参数之间的关系。此外，我们建立了统计力学中常用的宏观可观测量与机器学习中常用的损失函数之间的联系。在技术上，从解析的角度，我们运用Guerra的插值实现了大偏差和稳定性分析，用于处理与突触后电位相关的非高斯分布；从计算的角度，我们将Plefka近似插入到蒙特卡洛方案中，以加速突触强度的评估。

    We consider dense, associative neural-networks trained with no supervision and we investigate their computational capabilities analytically, via a statistical-mechanics approach, and numerically, via Monte Carlo simulations. In particular, we obtain a phase diagram summarizing their performance as a function of the control parameters such as the quality and quantity of the training dataset and the network storage, valid in the limit of large network size and structureless datasets. Moreover, we establish a bridge between macroscopic observables standardly used in statistical mechanics and loss functions typically used in the machine learning. As technical remarks, from the analytic side, we implement large deviations and stability analysis within Guerra's interpolation to tackle the not-Gaussian distributions involved in the post-synaptic potentials while, from the computational counterpart, we insert Plefka approximation in the Monte Carlo scheme, to speed up the evaluation of the syn
    
[^49]: Shapley曲线：一种平滑视角

    Shapley Curves: A Smoothing Perspective. (arXiv:2211.13289v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.13289](http://arxiv.org/abs/2211.13289)

    本文以平滑的角度引入了Shapley曲线作为局部变量重要性的度量，提出了两种估计策略，并在特征的独立和依赖情况下得到了一致性和渐近正态性，为估计的Shapley曲线构建了置信区间并进行了推断，通过实验证实了渐近结果。应用中分析了哪些属性驱动车辆价格。

    

    源自合作博弈理论，Shapley值已成为应用机器学习中最广泛使用的变量重要性度量之一。然而，对Shapley值的统计理解仍然有限。本文以非参数(或平滑)的角度，引入Shapley曲线作为局部变量重要性的度量。我们提出了两种估计策略，并在特征独立和依赖的情况下都得出了一致性和渐近正态性。这样，我们可以构建置信区间并对估计的Shapley曲线进行推断。我们提出了一种新颖的野蛮引导程序版本，专门调整以获得Shapley曲线的良好有限样本覆盖。渐近结果在大量实验证实了。在实证应用中，我们分析了哪些属性驱动了车辆的价格。

    Originating from cooperative game theory, Shapley values have become one of the most widely used measures for variable importance in applied Machine Learning. However, the statistical understanding of Shapley values is still limited. In this paper, we take a nonparametric (or smoothing) perspective by introducing Shapley curves as a local measure of variable importance. We propose two estimation strategies and derive the consistency and asymptotic normality both under independence and dependence among the features. This allows us to construct confidence intervals and conduct inference on the estimated Shapley curves. We propose a novel version of the wild bootstrap procedure, specifically adjusted to give good finite sample coverage of the Shapley curves. The asymptotic results are validated in extensive experiments. In an empirical application, we analyze which attributes drive the prices of vehicles.
    
[^50]: 多模态时间数据的主动获取：一个具有挑战性的决策任务

    Active Acquisition for Multimodal Temporal Data: A Challenging Decision-Making Task. (arXiv:2211.05039v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.05039](http://arxiv.org/abs/2211.05039)

    该论文提出了一个具有挑战性的决策任务，主动获取多模态时间数据。通过权衡获取成本和预测性能，学习代理程序来主动选择获取的输入模态。该方法能够解决具有实际相关推理技能的合成情景，并在真实数据集上成功学习到成本反应式的获取行为，但无法学习到自适应的获取策略，突显了任务的困难性。

    

    我们介绍了一个具有挑战性的决策任务，我们称之为多模态时间数据的主动获取（A2MT）。在许多实际场景中，输入特征在测试时不容易获得，必须以较大代价获取。通过A2MT，我们的目标是学习代理程序，使其能够主动选择要获取的输入模态，权衡获取成本与预测性能。A2MT扩展了之前的任务，称为主动特征获取，以便进行关于高维输入的时间决策。我们提出了一种基于Perceiver IO架构的方法来实现A2MT。我们的代理程序能够解决一个需要实际相关的跨模态推理技能的新颖合成情景。在两个大规模的真实数据集Kinetics-700和AudioSet上，我们的代理程序成功地学习了成本反应式的获取行为。然而，消融实验表明它们无法学习到自适应的获取策略，突显了该任务的困难性。

    We introduce a challenging decision-making task that we call active acquisition for multimodal temporal data (A2MT). In many real-world scenarios, input features are not readily available at test time and must instead be acquired at significant cost. With A2MT, we aim to learn agents that actively select which modalities of an input to acquire, trading off acquisition cost and predictive performance. A2MT extends a previous task called active feature acquisition to temporal decision making about high-dimensional inputs. We propose a method based on the Perceiver IO architecture to address A2MT in practice. Our agents are able to solve a novel synthetic scenario requiring practically relevant cross-modal reasoning skills. On two large-scale, real-world datasets, Kinetics-700 and AudioSet, our agents successfully learn cost-reactive acquisition behavior. However, an ablation reveals they are unable to learn adaptive acquisition strategies, emphasizing the difficulty of the task even for 
    
[^51]: 神经扩展卡尔曼滤波器用于学习和预测结构系统的动力学

    Neural Extended Kalman Filters for Learning and Predicting Dynamics of Structural Systems. (arXiv:2210.04165v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.04165](http://arxiv.org/abs/2210.04165)

    本论文提出了一种称为神经扩展卡尔曼滤波器（Neural EKF）的可学习卡尔曼滤波方法，用于学习复杂物理系统的潜在演化动力学。这种方法可以通过端到端训练来学习过程动力学和传感观测的建模，提高结构响应预测的准确性。

    

    准确的结构响应预测是结构健康监测和控制应用的主要驱动力。这往往需要所提出的模型充分捕捉复杂结构系统的基本动力学。在这项工作中，我们利用可学习的扩展卡尔曼滤波器（称为神经扩展卡尔曼滤波器）来学习复杂物理系统的潜在演化动力学。神经扩展卡尔曼滤波器是传统卡尔曼滤波器的广义版本，其中过程动力学和传感观测的建模可以通过神经网络来参数化，因此可以通过端到端训练来学习。该方法在变分推理框架下实现，卡尔曼滤波器通过感知测量进行推理。通常，传统的变分推理模型的参数是独立于潜在动力学模型的神经网络参数化的。这种特点使得推理和重构的准确性相对较弱。

    Accurate structural response prediction forms a main driver for structural health monitoring and control applications. This often requires the proposed model to adequately capture the underlying dynamics of complex structural systems. In this work, we utilize a learnable Extended Kalman Filter (EKF), named the Neural Extended Kalman Filter (Neural EKF) throughout this paper, for learning the latent evolution dynamics of complex physical systems. The Neural EKF is a generalized version of the conventional EKF, where the modeling of process dynamics and sensory observations can be parameterized by neural networks, therefore learned by end-to-end training. The method is implemented under the variational inference framework with the EKF conducting inference from sensing measurements. Typically, conventional variational inference models are parameterized by neural networks independent of the latent dynamics models. This characteristic makes the inference and reconstruction accuracy weakly b
    
[^52]: The Vendi分数: 一种用于机器学习的多样性评估指标

    The Vendi Score: A Diversity Evaluation Metric for Machine Learning. (arXiv:2210.02410v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.02410](http://arxiv.org/abs/2210.02410)

    本文提出了一种用于机器学习的多样性评估指标Vendi分数，它能够灵活地衡量不同形式的多样性，而且不需要参考数据集，适用于任何生成模型和数据集。

    

    多样性是机器学习（ML）中许多领域的重要标准，包括生成建模和数据集策划。然而，现有的衡量多样性的指标往往是针对特定领域的，并且灵活性有限。本文通过提出Vendi分数来解决多样性评估问题，该指标将生态学和量子统计力学的思想与ML相结合并进行扩展。Vendi分数定义为相似性矩阵的特征值的香农熵的指数函数。这个矩阵是由用户定义的相似性函数应用于要评估多样性的样本而诱导出的。通过使用相似性函数作为输入，Vendi分数使用户能够指定任何所需的多样性形式。与ML中的许多现有指标不同，Vendi分数不需要参考数据集或样本或标签的分布，因此它通用且适用于任何生成模型、解码算法和来自任何领域的数据集。

    Diversity is an important criterion for many areas of machine learning (ML), including generative modeling and dataset curation. However, existing metrics for measuring diversity are often domain-specific and limited in flexibility. In this paper, we address the diversity evaluation problem by proposing the Vendi Score, which connects and extends ideas from ecology and quantum statistical mechanics to ML. The Vendi Score is defined as the exponential of the Shannon entropy of the eigenvalues of a similarity matrix. This matrix is induced by a user-defined similarity function applied to the sample to be evaluated for diversity. In taking a similarity function as input, the Vendi Score enables its user to specify any desired form of diversity. Importantly, unlike many existing metrics in ML, the Vendi Score does not require a reference dataset or distribution over samples or labels, it is therefore general and applicable to any generative model, decoding algorithm, and dataset from any d
    
[^53]: 关于弱标识函数的强标识功能的推理

    Inference on Strongly Identified Functionals of Weakly Identified Functions. (arXiv:2208.08291v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.08291](http://arxiv.org/abs/2208.08291)

    本文研究了一种新的条件，使得即使干扰函数是弱标识的，也可以对功能进行强标识并进行推理。

    

    在各种应用中，包括非参数工具变量(NPIV)分析、未测到混淆下的近因果推理和缺失非随机数据与隐藏变量，我们对条件矩限制定义的干扰函数(例如平均因果效应)进行推理。这些干扰函数通常是弱标识的，即条件矩限制可以严重不良，同时也可以有多个解。有时，通过施加能够使函数以使关于功能的推理成为可能的速率来估计解决这个问题。在本文中，我们研究了一种新的条件，用于功能的强标识，即使干扰函数不是；也就是说，功能可以以$\sqrt{n}$的速率进行渐近正态估计。这个条件意味着修正干扰函数的存在，

    In a variety of applications, including nonparametric instrumental variable (NPIV) analysis, proximal causal inference under unmeasured confounding, and missing-not-at-random data with shadow variables, we are interested in inference on a continuous linear functional (e.g., average causal effects) of nuisance function (e.g., NPIV regression) defined by conditional moment restrictions. These nuisance functions are generally weakly identified, in that the conditional moment restrictions can be severely ill-posed as well as admit multiple solutions. This is sometimes resolved by imposing strong conditions that imply the function can be estimated at rates that make inference on the functional possible. In this paper, we study a novel condition for the functional to be strongly identified even when the nuisance function is not; that is, the functional is amenable to asymptotically-normal estimation at $\sqrt{n}$-rates. The condition implies the existence of debiasing nuisance functions, and
    
[^54]: 在R中使用theft包进行基于特征的时间序列分析

    Feature-Based Time-Series Analysis in R using the theft Package. (arXiv:2208.06146v4 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2208.06146](http://arxiv.org/abs/2208.06146)

    本研究介绍了在R中使用theft包进行基于特征的时间序列分析的方法，并指出了当前存在的问题包括缺乏统一的访问点以及用户需要掌握多种编程语言来获得所有特征集。

    

    时间序列在各个科学领域中被测量和分析。一种量化时间序列结构的方法是通过计算一组摘要统计量或"特征"，然后用特征向量的属性来表示时间序列。结果得到的特征空间是可解释和信息丰富的，使得传统的统计学习方法，包括聚类、回归和分类，可以应用于时间序列数据集。存在许多开源软件包在多种编程语言中计算时间序列特征集，包括catch22（22个特征：Matlab、R、Python、Julia）、feasts（42个特征：R）、tsfeatures（63个特征：R）、Kats（40个特征：Python）、tsfresh（779个特征：Python）和TSFEL（390个特征：Python）。然而，存在几个问题：（i）目前尚无这些软件包的单一访问点；（ii）要访问所有特征集，用户必须精通多种语言；（iii）th

    Time series are measured and analyzed across the sciences. One method of quantifying the structure of time series is by calculating a set of summary statistics or `features', and then representing a time series in terms of its properties as a feature vector. The resulting feature space is interpretable and informative, and enables conventional statistical learning approaches, including clustering, regression, and classification, to be applied to time-series datasets. Many open-source software packages for computing sets of time-series features exist across multiple programming languages, including catch22 (22 features: Matlab, R, Python, Julia), feasts (42 features: R), tsfeatures (63 features: R), Kats (40 features: Python), tsfresh (779 features: Python), and TSFEL (390 features: Python). However, there are several issues: (i) a singular access point to these packages is not currently available; (ii) to access all feature sets, users must be fluent in multiple languages; and (iii) th
    
[^55]: 高维时间序列数据分析的深度直接判别解码器

    Deep Direct Discriminative Decoders for High-dimensional Time-series Data Analysis. (arXiv:2205.10947v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2205.10947](http://arxiv.org/abs/2205.10947)

    这篇论文提出了一种用于高维时间序列数据分析的新方法，即深度直接判别解码器（D4）。D4通过引入深度神经网络的表达能力和可扩展性，有效地估计了高维观测信号下的潜在状态过程，并在多个数据集上展示了比传统方法更好的性能。

    

    状态空间模型（SSMs）被广泛应用于时间序列数据分析中。SSMs依赖于对状态和观测过程的明确定义。当观测数据的维度增加或观测数据分布偏离正态分布时，描述这些过程并不总是容易的，这成为建模的挑战。在这里，我们提出了一种用于高维观测过程的新的SSM表达形式。我们将这个解决方案称为深度直接判别解码器（D4）。D4将深度神经网络的表达能力和可扩展性引入到SSM表达形式中，使我们能够构建一个新的解决方案，通过高维观测信号高效地估计潜在的状态过程。我们在模拟和真实数据（如Lorenz吸引子、Langevin动力学、随机行走动力学和大鼠海马鞭状神经数据）上演示了D4解决方案，并展示了它比传统的SSMs和RNNs更好的性能。D4可以应用于

    The state-space models (SSMs) are widely utilized in the analysis of time-series data. SSMs rely on an explicit definition of the state and observation processes. Characterizing these processes is not always easy and becomes a modeling challenge when the dimension of observed data grows or the observed data distribution deviates from the normal distribution. Here, we propose a new formulation of SSM for high-dimensional observation processes. We call this solution the deep direct discriminative decoder (D4). The D4 brings deep neural networks' expressiveness and scalability to the SSM formulation letting us build a novel solution that efficiently estimates the underlying state processes through high-dimensional observation signal. We demonstrate the D4 solutions in simulated and real data such as Lorenz attractors, Langevin dynamics, random walk dynamics, and rat hippocampus spiking neural data and show that the D4 performs better than traditional SSMs and RNNs. The D4 can be applied t
    
[^56]: 外部有效的策略选择

    Externally Valid Policy Choice. (arXiv:2205.05561v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2205.05561](http://arxiv.org/abs/2205.05561)

    本文研究了外部有效的个性化治疗策略的学习问题。研究表明，最大化福利的治疗策略对于实验和目标人群之间的结果分布变化具有鲁棒性。通过开发新的方法，作者提出了对结果和特征变化具有鲁棒性的策略学习方法，并强调实验人群内的治疗效果异质性对策略普适性的影响。

    

    我们考虑学习个性化治疗策略的问题，这些策略是外部有效或广义化的：它们在除了实验（或训练）人群外的其他目标人群中表现良好。我们首先证明，对于实验人群而言，最大化福利的策略对于实验和目标人群之间的结果（但不是特征）分布变化具有鲁棒性。然后，我们开发了新的方法来学习对结果和特征变化具有鲁棒性的策略。在这样做时，我们强调了实验人群内的治疗效果异质性如何影响策略的普适性。我们的方法可以使用实验或观察数据（其中治疗是内生的）。我们的许多方法可以使用线性规划实现。

    We consider the problem of learning personalized treatment policies that are externally valid or generalizable: they perform well in other target populations besides the experimental (or training) population from which data are sampled. We first show that welfare-maximizing policies for the experimental population are robust to shifts in the distribution of outcomes (but not characteristics) between the experimental and target populations. We then develop new methods for learning policies that are robust to shifts in outcomes and characteristics. In doing so, we highlight how treatment effect heterogeneity within the experimental population affects the generalizability of policies. Our methods may be used with experimental or observational data (where treatment is endogenous). Many of our methods can be implemented with linear programming.
    
[^57]: 基于双变量藤蔓顺序相关的回归、双变量水平曲线和分位曲线

    Bivariate vine copula based regression, bivariate level and quantile curves. (arXiv:2205.02557v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2205.02557](http://arxiv.org/abs/2205.02557)

    该论文提出了一种基于藤蔓顺序相关的回归模型，用于构建双变量分位数，并使用藤蔓相关的水平曲线。这种方法可以避免传统回归模型的一些问题，如变量转换、共线性和分位数交叉。

    

    单变量分位数的统计分析已经很成熟。然而，对于多变量分位数的研究仍然有待深入。我们构建了基于藤蔓顺序相关的双变量（条件）分位数，使用藤蔓相关回归模型的水平曲线。藤蔓相关是一种由连续树形模型确定的图形模型，允许对边际分布和相关结构进行分开建模。我们引入了一种新的图形结构模型（由树序列给出），专为预测回归设置中两个响应的对称处理而设计。我们确立了模型的计算可行性和获得不同条件分布的简单方法。使用藤蔓相关，回归的典型不足，如需对预测变量进行转换或交互、共线性或分位数交叉，都可以避免。我们通过不同藤蔓相关分布来说明基于藤蔓相关的双变量水平曲线。

    The statistical analysis of univariate quantiles is a well developed research topic. However, there is a need for research in multivariate quantiles. We construct bivariate (conditional) quantiles using the level curves of vine copula based bivariate regression model. Vine copulas are graph theoretical models identified by a sequence of linked trees, which allow for separate modelling of marginal distributions and the dependence structure. We introduce a novel graph structure model (given by a tree sequence) specifically designed for a symmetric treatment of two responses in a predictive regression setting. We establish computational tractability of the model and a straight forward way of obtaining different conditional distributions. Using vine copulas the typical shortfalls of regression, as the need for transformations or interactions of predictors, collinearity or quantile crossings are avoided. We illustrate the copula based bivariate level curves for different copula distribution
    
[^58]: 最后一层重新训练足以提高对虚假相关性的鲁棒性

    Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations. (arXiv:2204.02937v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2204.02937](http://arxiv.org/abs/2204.02937)

    本文研究表明，简单的最后一层重新训练足以提高神经网络分类器对虚假相关性的鲁棒性，可以在虚假相关性基准测试中与最先进的方法相媲美或胜过，但其复杂度和计算开销较低。此外，对于在ImageNet训练的大型模型进行最后一层重新训练，仅几分钟的训练时间就可以显著降低对背景和纹理信息的依赖，提高对协变量转变的鲁棒性。

    

    神经网络分类器可以主要依靠简单的虚假特征（如背景）进行预测。然而，即使在这些情况下，我们展示了它们仍然经常学习与数据所需属性相关的核心特征，与最近的研究结果相反。受到这一启示的启发，我们证明了简单的最后一层重新训练可以在虚假相关性基准测试中与甚至胜过最先进的方法，但其复杂度和计算开销显著较低。此外，我们还展示了对于在ImageNet训练的大型模型上进行最后一层重新训练，仅经过几分钟的单GPU训练，也可以显著降低对背景和纹理信息的依赖，提高对协变量转变的鲁棒性。

    Neural network classifiers can largely rely on simple spurious features, such as backgrounds, to make predictions. However, even in these cases, we show that they still often learn core features associated with the desired attributes of the data, contrary to recent findings. Inspired by this insight, we demonstrate that simple last layer retraining can match or outperform state-of-the-art approaches on spurious correlation benchmarks, but with profoundly lower complexity and computational expenses. Moreover, we show that last layer retraining on large ImageNet-trained models can also significantly reduce reliance on background and texture information, improving robustness to covariate shift, after only minutes of training on a single GPU.
    
[^59]: 当缺失观测的位置未知时学习隐马尔可夫模型

    Learning Hidden Markov Models When the Locations of Missing Observations are Unknown. (arXiv:2203.06527v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2203.06527](http://arxiv.org/abs/2203.06527)

    本文研究了当缺失观测的位置未知时学习隐马尔可夫模型的问题，并提供了不需要先验信息的重建算法。

    

    隐马尔可夫模型（HMM）是用于序列数据分析的最常用的统计模型之一。HMM具有处理缺失数据的能力，这也是它具有通用性的关键之一。然而，标准的HMM学习算法基于缺失观测在观测序列中的位置已知的假设。在自然科学中，这种假设常常不成立，因此通常使用特殊变体的HMM，称为Silent-state HMMs（SHMMs）。尽管这些算法被广泛使用，但它们严重依赖于潜在链的特定结构假设，比如非循环性，这限制了这些方法的适用性。而且，即使在非循环情况下，已经证明这些方法可能导致重建效果差。本文研究了从具有未知缺失观测位置数据中学习HMM的一般问题。我们提供了不需要任何先验信息的重建算法。

    The Hidden Markov Model (HMM) is one of the most widely used statistical models for sequential data analysis. One of the key reasons for this versatility is the ability of HMM to deal with missing data. However, standard HMM learning algorithms rely crucially on the assumption that the positions of the missing observations \emph{within the observation sequence} are known. In the natural sciences, where this assumption is often violated, special variants of HMM, commonly known as Silent-state HMMs (SHMMs), are used. Despite their widespread use, these algorithms strongly rely on specific structural assumptions of the underlying chain, such as acyclicity, thus limiting the applicability of these methods. Moreover, even in the acyclic case, it has been shown that these methods can lead to poor reconstruction. In this paper we consider the general problem of learning an HMM from data with unknown missing observation locations. We provide reconstruction algorithms that do not require any as
    
[^60]: 一种用于采样的近端算法

    A Proximal Algorithm for Sampling. (arXiv:2202.13975v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.13975](http://arxiv.org/abs/2202.13975)

    本论文提出了一种用于处理缺乏平滑性的势能的采样问题的近端算法，在凸和非凸情况下均可适用。该算法的关键创新点在于基于拒绝采样的交替采样框架的实际实现，比现有方法更高效。

    

    我们研究了与缺乏平滑性的势能相关的采样问题。这些势能可以是凸的或非凸的。不同于标准的平滑设置，这些势能只被认为是弱平滑或非平滑的，或者是多个这样的函数的求和。我们开发了一种采样算法，该算法类似于用于这种具有挑战性的采样任务的优化问题的近端算法。我们的算法基于一种称为交替采样框架（ASF）的Gibbs采样的特殊情况。这项工作的关键贡献是基于拒绝采样的ASF的实际实现，适用于非凸和不一定平滑的凸势能。在本工作考虑的几乎所有采样案例中，我们的近端采样算法的复杂性都优于所有现有方法。

    We study sampling problems associated with potentials that lack smoothness. The potentials can be either convex or non-convex. Departing from the standard smooth setting, the potentials are only assumed to be weakly smooth or non-smooth, or the summation of multiple such functions. We develop a sampling algorithm that resembles proximal algorithms in optimization for this challenging sampling task. Our algorithm is based on a special case of Gibbs sampling known as the alternating sampling framework (ASF). The key contribution of this work is a practical realization of the ASF based on rejection sampling for both non-convex and convex potentials that are not necessarily smooth. In almost all the cases of sampling considered in this work, our proximal sampling algorithm achieves better complexity than all existing methods.
    
[^61]: 偏差矩阵分解

    Deviance Matrix Factorization. (arXiv:2110.05674v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2110.05674](http://arxiv.org/abs/2110.05674)

    该论文研究了一种适用于偏差数据损失的通用矩阵分解方法，并且通过应用广义线性模型理论提供了支持，该方法具有灵活的算法和处理结构性零元素的能力。作者通过模拟研究和案例研究证明了该方法的鲁棒性和应用广泛性。

    

    我们研究了一种用于偏差数据损失的通用矩阵分解方法，将普遍存在的奇异值分解扩展到平方误差损失之外。虽然之前已经有类似的方法，但我们的方法利用了广义线性模型（GLM）的经典统计方法，并提供了一个灵活的算法，可以通过条目权重来处理结构性零元素。此外，通过调整GLM理论的结果，我们通过以下方式支持这些分解：（i）在GLM设置下显示强一致性，（ii）通过广义Hosmer-Lemeshow检验检验所选择指数族分布的适应性，以及（iii）通过最大特征值间隔法确定分解的秩。为了进一步支持我们的发现，我们进行了模拟研究，评估对分解假设的鲁棒性，并使用图像人脸识别、自然语言处理、网络分析和生物医学等基准数据集进行了广泛的案例研究。

    We investigate a general matrix factorization for deviance-based data losses, extending the ubiquitous singular value decomposition beyond squared error loss. While similar approaches have been explored before, our method leverages classical statistical methodology from generalized linear models (GLMs) and provides an efficient algorithm that is flexible enough to allow for structural zeros via entry weights. Moreover, by adapting results from GLM theory, we provide support for these decompositions by (i) showing strong consistency under the GLM setup, (ii) checking the adequacy of a chosen exponential family via a generalized Hosmer-Lemeshow test, and (iii) determining the rank of the decomposition via a maximum eigenvalue gap method. To further support our findings, we conduct simulation studies to assess robustness to decomposition assumptions and extensive case studies using benchmark datasets from image face recognition, natural language processing, network analysis, and biomedica
    
[^62]: 上下文逆优化：离线和在线学习

    Contextual Inverse Optimization: Offline and Online Learning. (arXiv:2106.14015v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2106.14015](http://arxiv.org/abs/2106.14015)

    这项研究研究了具有反馈信息的离线和在线情境优化问题，通过观察最佳动作并最小化后悔来优化决策制定。

    

    我们研究了具有反馈信息的离线和在线情境优化问题，其中我们观察到的不是损失，而是一个具有完全了解目标函数的预测神经网络将会采取的最佳动作。我们的目标是最小化后悔，后悔定义为我们的损失与全知预测神经网络产生的损失之间的差异。在离线情境中，决策者可以获得过去时期的信息并需要做出一个决策，而在在线情境中，决策者根据每个时期的新一组可行动作和情境函数来动态优化决策。对于离线情境，我们将最优极小极大策略特征化，确定了可以作为数据产生的信息的基础几何形状的函数表现。在在线情境中，我们利用这种几何特征来优化累积后悔。我们开发了算法来找到累积后悔的最小化策略。

    We study the problems of offline and online contextual optimization with feedback information, where instead of observing the loss, we observe, after-the-fact, the optimal action an oracle with full knowledge of the objective function would have taken. We aim to minimize regret, which is defined as the difference between our losses and the ones incurred by an all-knowing oracle. In the offline setting, the decision-maker has information available from past periods and needs to make one decision, while in the online setting, the decision-maker optimizes decisions dynamically over time based a new set of feasible actions and contextual functions in each period. For the offline setting, we characterize the optimal minimax policy, establishing the performance that can be achieved as a function of the underlying geometry of the information induced by the data. In the online setting, we leverage this geometric characterization to optimize the cumulative regret. We develop an algorithm that y
    
[^63]: 深度代理因果学习及其在混淆赌博策略评估中的应用

    Deep Proxy Causal Learning and its Application to Confounded Bandit Policy Evaluation. (arXiv:2106.03907v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2106.03907](http://arxiv.org/abs/2106.03907)

    本论文提出了一种深度代理因果学习（PCL）方法，用于在存在混淆因素的情况下估计治疗对结果的因果效应。通过构建治疗和代理之间的模型，并利用该模型在给定代理的情况下学习治疗对结果的影响，PCL可以保证恢复真实的因果效应。作者还提出了一种名为深度特征代理变量方法（DFPV）的新方法，用于处理高维和非线性复杂关系的情况，并表明DFPV在合成基准测试中的性能优于最先进的PCL方法。

    

    代理因果学习（PCL）是一种在存在未观察到的混淆因素时，利用代理（结构化侧面信息）估计治疗对结果的因果效应的方法。这是通过两阶段回归实现的：在第一阶段，我们建模治疗和代理之间的关系；在第二阶段，我们利用这个模型来学习在给定代理提供的上下文下，治疗对结果的影响。PCL在可识别条件下保证恢复真实的因果效应。我们提出了一种新的PCL方法，深度特征代理变量方法（DFPV），以解决代理、治疗和结果为高维且具有非线性复杂关系的情况，如深度神经网络特征表示。我们表明DFPV在具有挑战性的合成基准测试中优于最近的最先进的PCL方法，包括涉及高维图像数据的设置。此外，我们还展示了PCL的应用...

    Proxy causal learning (PCL) is a method for estimating the causal effect of treatments on outcomes in the presence of unobserved confounding, using proxies (structured side information) for the confounder. This is achieved via two-stage regression: in the first stage, we model relations among the treatment and proxies; in the second stage, we use this model to learn the effect of treatment on the outcome, given the context provided by the proxies. PCL guarantees recovery of the true causal effect, subject to identifiability conditions. We propose a novel method for PCL, the deep feature proxy variable method (DFPV), to address the case where the proxies, treatments, and outcomes are high-dimensional and have nonlinear complex relationships, as represented by deep neural network features. We show that DFPV outperforms recent state-of-the-art PCL methods on challenging synthetic benchmarks, including settings involving high dimensional image data. Furthermore, we show that PCL can be app
    
[^64]: 联机强化学习与模仿学习的桥梁：一个悲观的故事

    Bridging Offline Reinforcement Learning and Imitation Learning: A Tale of Pessimism. (arXiv:2103.12021v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2103.12021](http://arxiv.org/abs/2103.12021)

    本论文提出了一种新的联机强化学习框架，通过平滑插值的方式将模仿学习和纯联机强化学习统一起来。框架围绕着一种衡量行为策略与专家策略偏离程度的弱版本集中系数展开。通过该框架，研究者进一步研究了算法设计的问题：能否开发出实现最小极大最优性的算法？

    

    联机（或批次）强化学习算法旨在从固定的数据集中学习最优策略，而无需主动收集数据。根据离线数据集的组成，主要使用两种方法：适用于专家数据集的模仿学习和通常需要均匀覆盖数据集的纯联机强化学习。从实践的角度来看，数据集通常偏离这两个极端，并且通常事先不知道确切的数据组成。为了填补这一差距，我们提出了一个新的联机强化学习框架，它在数据组成的两个极端之间平滑插值，从而统一了模仿学习和纯联机强化学习。新的框架围绕一个弱版本的集中系数展开，该系数衡量了行为策略与专家策略之间的偏离程度。在这个新的框架下，我们进一步研究了算法设计的问题：能否开发出一种实现最小极大最优性的算法？

    Offline (or batch) reinforcement learning (RL) algorithms seek to learn an optimal policy from a fixed dataset without active data collection. Based on the composition of the offline dataset, two main categories of methods are used: imitation learning which is suitable for expert datasets and vanilla offline RL which often requires uniform coverage datasets. From a practical standpoint, datasets often deviate from these two extremes and the exact data composition is usually unknown a priori. To bridge this gap, we present a new offline RL framework that smoothly interpolates between the two extremes of data composition, hence unifying imitation learning and vanilla offline RL. The new framework is centered around a weak version of the concentrability coefficient that measures the deviation from the behavior policy to the expert policy alone.  Under this new framework, we further investigate the question on algorithm design: can one develop an algorithm that achieves a minimax optimal r
    
[^65]: 深度强化学习中的迁移学习综述

    Transfer Learning in Deep Reinforcement Learning: A Survey. (arXiv:2009.07888v6 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2009.07888](http://arxiv.org/abs/2009.07888)

    这篇综述调查了深度强化学习领域中的迁移学习方法的最新进展，并提供了一个对这些方法进行分类的框架。分析了它们的目标、方法学、兼容的强化学习背景以及实际应用，并探讨了迁移学习与其他相关主题之间的联系。

    

    强化学习是解决序列决策问题的学习范式。近年来，随着深度神经网络的快速发展，强化学习取得了显著的进展。除了在机器人和游戏等诸多领域中具有良好前景的强化学习，迁移学习作为一种解决强化学习面临的各种挑战的方法已经出现，通过从外部专业知识中转移知识，以提高学习过程的效率和效果。在这项综述中，我们系统地调查了深度强化学习领域中的迁移学习方法的最新进展。具体而言，我们提供了一个对最先进的迁移学习方法进行分类的框架，在此框架下分析了它们的目标、方法学、兼容的强化学习背景以及实际应用。我们还探讨了迁移学习与其他相关主题之间的联系。

    Reinforcement learning is a learning paradigm for solving sequential decision-making problems. Recent years have witnessed remarkable progress in reinforcement learning upon the fast development of deep neural networks. Along with the promising prospects of reinforcement learning in numerous domains such as robotics and game-playing, transfer learning has arisen to tackle various challenges faced by reinforcement learning, by transferring knowledge from external expertise to facilitate the efficiency and effectiveness of the learning process. In this survey, we systematically investigate the recent progress of transfer learning approaches in the context of deep reinforcement learning. Specifically, we provide a framework for categorizing the state-of-the-art transfer learning approaches, under which we analyze their goals, methodologies, compatible reinforcement learning backbones, and practical applications. We also draw connections between transfer learning and other relevant topics 
    
[^66]: 通过2层马尔可夫决策过程学习在团队中切换代理的方法

    Learning to Switch Among Agents in a Team via 2-Layer Markov Decision Processes. (arXiv:2002.04258v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2002.04258](http://arxiv.org/abs/2002.04258)

    本文研究了在团队中学习切换代理控制的问题，并开发了一种在线学习算法，通过学习代理的策略和环境的转移概率，在不同自动化水平下使现有的强化学习代理能够工作。该算法的总遗憾与最佳切换策略相比是次线性的，当多个代理团队在相似环境中运行时，该算法从维护环境的共享置信界中获益匪浅。

    

    强化学习代理在通常以完全自主的方式工作的假设下开发和评估 - 它们将采取所有行动。本文的目标是开发算法，通过学习在代理之间切换控制，使现有的强化学习代理能够在不同的自动化水平下工作。为此，我们首先正式定义了通过2层马尔可夫决策过程在团队中学习切换控制的问题。然后，我们使用代理的策略和环境的转移概率的上置信界开发了一种在线学习算法，以找到一系列切换策略。我们的算法相对于最佳切换策略的总遗憾在学习步骤的数量上是次线性的，并且每当多个代理团队在相似的环境中运行时，我们的算法从维护环境的共享置信界中获得很大的好处。

    Reinforcement learning agents have been mostly developed and evaluated under the assumption that they will operate in a fully autonomous manner -- they will take all actions. In this work, our goal is to develop algorithms that, by learning to switch control between agents, allow existing reinforcement learning agents to operate under different automation levels. To this end, we first formally define the problem of learning to switch control among agents in a team via a 2-layer Markov decision process. Then, we develop an online learning algorithm that uses upper confidence bounds on the agents' policies and the environment's transition probabilities to find a sequence of switching policies. The total regret of our algorithm with respect to the optimal switching policy is sublinear in the number of learning steps and, whenever multiple teams of agents operate in a similar environment, our algorithm greatly benefits from maintaining shared confidence bounds for the environments' transit
    
[^67]: 使用重要权重示范的元适应性学习

    Meta Adaptation using Importance Weighted Demonstrations. (arXiv:1911.10322v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/1911.10322](http://arxiv.org/abs/1911.10322)

    本文提出了一种使用重要权重示范的元适应性学习算法，通过对特定任务的先前知识进行分配重要权重，实现了在任何相关任务上的泛化。实验证明，该方法能够使机器人在多样化环境任务中进行训练，并通过少量示范适应未知环境。

    

    由于其高样本效率，模仿学习变得极为流行。然而，在实际应用场景中，由于大多数任务的轨迹分布不断变化，仅仅基于连续聚合的数据来进行模型拟合是徒劳的。在某些情况下，分布发生如此大的变化，以至于智能体很难推断出新任务。我们提出了一种新颖的算法，通过对一组特定任务的先前知识进行分配重要权重，从而在任何相关任务上进行泛化。我们展示了一些实验，在这些实验中，机器人从多样化的环境任务中训练，并能够通过少量示范进行学习，从而适应未知环境。我们还开发了一个原型机器人系统，在视觉导航任务上测试我们的方法，并获得了能够验证这些假设的实验证据。

    Imitation learning has gained immense popularity because of its high sample-efficiency. However, in real-world scenarios, where the trajectory distribution of most of the tasks dynamically shifts, model fitting on continuously aggregated data alone would be futile. In some cases, the distribution shifts, so much, that it is difficult for an agent to infer the new task. We propose a novel algorithm to generalize on any related task by leveraging prior knowledge on a set of specific tasks, which involves assigning importance weights to each past demonstration. We show experiments where the robot is trained from a diversity of environmental tasks and is also able to adapt to an unseen environment, using few-shot learning. We also developed a prototype robot system to test our approach on the task of visual navigation, and experimental results obtained were able to confirm these suppositions.
    

