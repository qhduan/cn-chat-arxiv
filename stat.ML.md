# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Mitigating Covariate Shift in Misspecified Regression with Applications to Reinforcement Learning.](http://arxiv.org/abs/2401.12216) | 本文研究了在模型规范错误的条件下，由于分布转变导致的错误放大问题，并提出了一种新的算法来解决这个问题。 |
| [^2] | [The Dimension Strikes Back with Gradients: Generalization of Gradient Methods in Stochastic Convex Optimization.](http://arxiv.org/abs/2401.12058) | 本研究探讨了随机凸优化中梯度方法的泛化性能以及维度依赖性。我们发现标准的全批量梯度下降方法在维度$d = O(n^2)$时需要至少$\Omega(\sqrt{d})$个训练样本才能达到非平凡的测试误差。而对于标准的一遍随机梯度下降方法，同样的维度下界也适用。 |
| [^3] | [Integrating Statistical Significance and Discriminative Power in Pattern Discovery.](http://arxiv.org/abs/2401.12000) | 本论文将统计显著性和判别能力融入模式发现中，提出了一种方法来同时满足模式质量和统计标准，在三元聚类任务上的实验结果显示其有效性。 |
| [^4] | [Cross-Validation Conformal Risk Control.](http://arxiv.org/abs/2401.11974) | 本文提出了一种基于交叉验证的新型合规风险控制方法(CV-CRC)，它扩展了一致性预测的概念，能够控制更广泛的风险函数，并在预测器集合的平均风险上提供了理论保证。 |
| [^5] | [RUMBoost: Gradient Boosted Random Utility Models.](http://arxiv.org/abs/2401.11954) | RUMBoost模型将随机效用模型（RUMs）的解释性和行为鲁棒性与深度学习方法的泛化能力相结合，通过使用梯度提升回归树的集合来获得非线性效用函数的完整函数形式，实现了对输入变量的任何可能组合进行常数插补。 |
| [^6] | [Low-Tubal-Rank Tensor Recovery via Factorized Gradient Descent.](http://arxiv.org/abs/2401.11940) | 本文提出了一种通过分解梯度下降方法解决低胞状秩张量恢复问题的高效方法，该方法通过将大张量分解为两个较小的因子张量，在减少计算成本和存储需求的同时，确保了收敛性。 |
| [^7] | [Subgroup analysis methods for time-to-event outcomes in heterogeneous randomized controlled trials.](http://arxiv.org/abs/2401.11842) | 本论文评估了多种时间到事件结果的亚组分析算法，填补了这一领域的研究空白，并提出了一种新的数据生成过程，可以探索不同的异质性情景。 |
| [^8] | [Accelerating Approximate Thompson Sampling with Underdamped Langevin Monte Carlo.](http://arxiv.org/abs/2401.11665) | 本文提出了一种使用欠阻尼 Langevin Monte Carlo 加速的近似 Thompson 采样策略，通过特定势函数的设计改善了高维问题中的样本复杂度，并在高维赌博机问题中进行了验证。 |
| [^9] | [Nonparametric Estimation via Variance-Reduced Sketching.](http://arxiv.org/abs/2401.11646) | 本文提出了一种名为Variance-Reduced Sketching的框架，用于在高维度中估计密度函数和非参数回归函数。该方法通过将函数概念化为矩阵，并采用草图技术来降低维度灾难引起的方差，展示了鲁棒性能和显著改进。 |
| [^10] | [Efficient local linearity regularization to overcome catastrophic overfitting.](http://arxiv.org/abs/2401.11618) | 本研究引入了一种名为ELLE的正则化项，用于高效地减轻单步对抗性训练中的灾难性过拟合。它能够保持损失函数在输入上的局部线性性，与传统的正则化方法相比，ELLE更加高效，能够有效应对大对抗性扰动和长训练计划等困难情况。 |
| [^11] | [Understanding the Generalization Benefits of Late Learning Rate Decay.](http://arxiv.org/abs/2401.11600) | 本文研究了为什么使用大学习率长时间训练神经网络可以实现更好的泛化。通过观察训练和测试损失之间的关系，我们发现大学习率下的训练轨迹能够接近测试损失的最小值附近。基于这个发现，我们提出了一个与真实神经网络类似的非线性模型，并证明了使用大学习率进行延长阶段的训练可以实现更接近最优泛化的效果。 |
| [^12] | [Thompson Sampling for Stochastic Bandits with Noisy Contexts: An Information-Theoretic Regret Analysis.](http://arxiv.org/abs/2401.11565) | 本文研究了具有噪音上下文的随机赌臂问题，并提出了一种Thompson采样算法，通过贝叶斯框架进行分析，证明了算法的贝叶斯后悔，并扩展了问题到延迟观察真实上下文的情况，并实证了算法的性能。 |
| [^13] | [Enhancing selectivity using Wasserstein distance based reweighing.](http://arxiv.org/abs/2401.11562) | 我们设计了一种使用Wasserstein距离进行加权的算法，在标记的数据集上训练神经网络可以逼近在其他数据集上训练得到的结果。我们证明了算法可以输出接近最优的加权，且算法简单可扩展。我们的算法可以有意地引入分布偏移进行多目标优化。作为应用实例，我们训练了一个神经网络来识别对细胞信号传导的MAP激酶具有非结合性的小分子结合物。 |
| [^14] | [MoMA: Model-based Mirror Ascent for Offline Reinforcement Learning.](http://arxiv.org/abs/2401.11380) | MoMA提出了一种模型为基础的镜像上升算法，通过使用无限制的策略类别和一般函数逼近来实现离线强化学习，充分利用了模型为基础方法的优势。 |
| [^15] | [Quantum Machine Learning: from NISQ to Fault Tolerance.](http://arxiv.org/abs/2401.11351) | 本文提供了对量子机器学习领域的全面回顾，涵盖了在NISQ技术和容错量子计算硬件上使用的技术和算法，并深入讨论了与量子机器学习相关的基本概念和统计学习理论。 |
| [^16] | [Estimating heterogeneous treatment effect from survival outcomes via (orthogonal) censoring unbiased learning.](http://arxiv.org/abs/2401.11263) | 该论文开发了一种适用于具有和没有竞争风险的生存结果的截尾无偏变换方法，可以估计异质治疗效应。这种方法可以应用于更多最先进的适用于被截尾结果的HTE学习方法，并提供了限制有限样本过度风险的方法。 |
| [^17] | [AFS-BM: Enhancing Model Performance through Adaptive Feature Selection with Binary Masking.](http://arxiv.org/abs/2401.11250) | AFS-BM通过联合优化实现了自适应特征选择和模型训练，提高了模型准确性并减少了计算需求。 |
| [^18] | [Identification and Estimation of Conditional Average Partial Causal Effects via Instrumental Variable.](http://arxiv.org/abs/2401.11130) | 本文提出了一种通过工具变量法识别和估计连续处理的因果效应异质性的方法，并开发了三类相应的估计器，并对其进行了统计性质分析。 |
| [^19] | [Efficient Data Shapley for Weighted Nearest Neighbor Algorithms.](http://arxiv.org/abs/2401.11103) | 本研究提出了一种解决加权K最近邻算法高效计算Data Shapley值的方法，并通过实验证明其在数据质量判别方面优于未加权版本。 |
| [^20] | [Learning from Aggregate responses: Instance Level versus Bag Level Loss Functions.](http://arxiv.org/abs/2401.11081) | 本文研究了从聚合响应中学习的两种损失函数：包级别损失和实例级别损失，并发现实例级别损失可以被视为包级别损失的正则化形式。 |
| [^21] | [Provably Scalable Black-Box Variational Inference with Structured Variational Families.](http://arxiv.org/abs/2401.10989) | 本文研究了均值场变分族和满秩变分族之间的理论中间地带：结构化变分族，并通过理论证明结构化变分族可以在迭代复杂性上表现更好，缩放效果更好。 |
| [^22] | [Debiasing and a local analysis for population clustering using semidefinite programming.](http://arxiv.org/abs/2401.10927) | 本文研究了使用半正定规划进行人群聚类的问题，并提出了计算高效的算法。这些算法可以根据小样本数据的原始种群将数据分为两组，适用于种群之间差异较小的情况。 |
| [^23] | [Online estimation of the inverse of the Hessian for stochastic optimization with application to universal stochastic Newton algorithms.](http://arxiv.org/abs/2401.10923) | 该论文提出了一种在线估计Hessian矩阵逆的方法，利用Robbins-Monro过程，能够 drastical reducescomputational complexity,并发展了通用的随机牛顿算法，研究了所提方法的渐进效率。 |
| [^24] | [Early alignment in two-layer networks training is a two-edged sword.](http://arxiv.org/abs/2401.10791) | 本文研究了两层网络训练中的早期对齐现象，发现在小初始化和一个隐藏的ReLU层网络中，神经元会在训练的早期阶段向关键方向进行对齐，导致网络稀疏表示以及梯度流在收敛时的隐含偏好。然而，这种稀疏诱导的对齐也使得训练目标的最小化变得困难。 |
| [^25] | [The Effect of Intrinsic Dataset Properties on Generalization: Unraveling Learning Differences Between Natural and Medical Images.](http://arxiv.org/abs/2401.08865) | 本文研究了神经网络在自然图像和医学图像领域学习时的差异，提出了一个与训练集维度有关的泛化缩放定律，并认为医学图像数据集更高的固有“标签锐度”可能是两个领域之间显著差异的部分原因。 |
| [^26] | [Neural Stochastic Differential Equations with Change Points: A Generative Adversarial Approach.](http://arxiv.org/abs/2312.13152) | 本文提出了一种使用神经随机微分方程来建模时间序列的变点检测算法，通过生成对抗网络学习每个变点对应的神经随机微分方程的参数，并通过GAN判别器的输出检测变点。验证结果表明，该算法在合成和真实数据集上的性能优于传统方法。 |
| [^27] | [Towards Optimal Statistical Watermarking.](http://arxiv.org/abs/2312.07930) | 追求最优统计水印技术。通过将统计水印技术视为假设检验问题并引入伪随机生成器，我们实现了输出令牌和拒绝区域的耦合，实现了第一类错误和第二类错误之间的非平凡权衡，同时提出了最统一最有力的水印和最小化第二类错误的解决方案。我们还提供了独立同分布令牌数量的上下界，突显了改进的潜力。此外，我们还探讨了鲁棒性水印问题。 |
| [^28] | [Optimal Multi-Distribution Learning.](http://arxiv.org/abs/2312.05134) | 本论文提出了一种最优化多分布学习的方法，通过自适应采样来实现数据高效的学习。针对Vapnik-Chervonenkis (VC)维数为d的假设类，算法可以生成一个ε-最优随机假设，并且样本复杂度与最佳下界保持一致。同时，该算法的思想和理论还被进一步扩展以适应Rademacher类。最终提出的算法是奥拉克尔高效的，仅访问假设类。 |
| [^29] | [On the Nystrom Approximation for Preconditioning in Kernel Machines.](http://arxiv.org/abs/2312.03311) | 本文分析了核机器预处理中使用Nystrom逼近的权衡。研究表明，使用对数大小的样本能够让Nystrom逼近的预处理器几乎与梯度下降同样有效地加速。 |
| [^30] | [Annotation Sensitivity: Training Data Collection Methods Affect Model Performance.](http://arxiv.org/abs/2311.14212) | 该研究发现训练数据收集方法对注释本身和下游模型性能产生影响。在对仇恨言论和冒犯性语言进行注释收集的实验中，发现注释工具的设计选择会对模型的性能产生明显差异。 |
| [^31] | [On the Foundation of Distributionally Robust Reinforcement Learning.](http://arxiv.org/abs/2311.09018) | 该论文为分布鲁棒强化学习的理论基础做出了贡献，通过一个综合的建模框架，决策者在最坏情况下的分布转变下选择最优策略，并考虑了各种建模属性和对手引起的转变的灵活性。 |
| [^32] | [Approximating Langevin Monte Carlo with ResNet-like Neural Network architectures.](http://arxiv.org/abs/2311.03242) | 本论文提出了一种使用类似ResNet的神经网络架构来近似Langevin Monte Carlo算法，通过将来自简单参考分布的样本映射到目标分布的样本中来进行采样，具有较好的逼近速度和表达性。 |
| [^33] | [Generator Identification for Linear SDEs with Additive and Multiplicative Noise.](http://arxiv.org/abs/2310.19491) | 本文介绍了从具有给定初始状态的解过程的分布中识别线性随机微分方程（SDE）的发生器的条件，并且提供了对于具有加性和乘性噪声的SDE的识别条件。 |
| [^34] | [Learning an Inventory Control Policy with General Inventory Arrival Dynamics.](http://arxiv.org/abs/2310.17168) | 本文解决了学习具有一般库存到货动态下的库存控制策略的问题，同时允许修改订购数量以满足供应商的限制，并将周期性审核库存控制问题定义为外部决策过程。 |
| [^35] | [Learning bounded-degree polytrees with known skeleton.](http://arxiv.org/abs/2310.06333) | 本论文提出了一种高效学习已知骨架的有界度多树的算法，并给出了在多项式时间和样本复杂度内的有限样本保证。这对于复杂概率分布的学习具有重要意义。 |
| [^36] | [A Latent Variable Approach for Non-Hierarchical Multi-Fidelity Adaptive Sampling.](http://arxiv.org/abs/2310.03298) | 提出了一种基于潜变量的方法，用于非层次化多保真度自适应采样。该方法能够利用不同保真度模型之间的相关性以更高效地探索和利用设计空间。 |
| [^37] | [On the different regimes of Stochastic Gradient Descent.](http://arxiv.org/abs/2309.10688) | 这项研究解决了对于随机梯度下降（SGD）中不同模式的追踪和理解的问题，提供了一个相位图来区分噪声主导的SGD和大步骤主导的SGD。 |
| [^38] | [Decolonial AI Alignment: Vi\'{s}esadharma, Argument, and Artistic Expression.](http://arxiv.org/abs/2309.05030) | 本文提出了去殖民化人工智能对齐的三个建议：改变基本道德哲学为达尔玛哲学，允许多元主义的论证传统存在于对齐技术中，以及将价值认识论扩展到超越自然语言中的指令。 |
| [^39] | [Robust Uncertainty Quantification using Conformalised Monte Carlo Prediction.](http://arxiv.org/abs/2308.09647) | 这篇论文介绍了一种名为MC-CP的新型混合不确定性量化方法，通过将自适应蒙特卡洛dropout方法与合规预测相结合，实现了节省资源和产生鲁棒预测集/区间的目标。实验证明MC-CP在分类任务中相比其他先进方法具有显著提升 |
| [^40] | [Multiclass Online Learnability under Bandit Feedback.](http://arxiv.org/abs/2308.04620) | Bandit反馈下的在线多类学习的关键在于Bandit Littlestone维度的有限性，无论标签空间是否无界。 |
| [^41] | [Logarithmic Bayes Regret Bounds.](http://arxiv.org/abs/2306.09136) | 该论文提出了对于贝叶斯赌博机的首个有限时间对数遗憾边界，并用于高斯和线性赌博机，从而阐明了贝叶斯设置中先验价值以及对$\tilde{O}(\sqrt{n})$界限的改善。 |
| [^42] | [Posterior Collapse in Linear Conditional and Hierarchical Variational Autoencoders.](http://arxiv.org/abs/2306.05023) | 本文研究了高度相似的变分后验分布和先验分布之间的后验崩溃现象，特别地，通过对线性条件VAE和分层VAE进行分析，证明了这种现象是由于潜在变量层次关系不清晰而引起的。 |
| [^43] | [Data-Driven Regret Balancing for Online Model Selection in Bandits.](http://arxiv.org/abs/2306.02869) | 论文讨论在具有赌博反馈的随机环境中进行选择，提出了两种基于数据的模型选择算法，并证明了其保证。通过利用实际遗憾，这些算法在实际中取得了好效果。 |
| [^44] | [Better Batch for Deep Probabilistic Time Series Forecasting.](http://arxiv.org/abs/2305.17028) | 该研究提出了一种新的训练方法，通过在 mini-batch 中显式地学习误差的序列相关性，来提高深度概率时间序列预测的准确性和不确定性量化。 |
| [^45] | [Theoretical Analysis of Inductive Biases in Deep Convolutional Networks.](http://arxiv.org/abs/2305.08404) | 本文研究深卷积神经网络（CNN）中的归纳偏置，证明了$\mathcal{O}(\log d)$的深度就足以实现普适性，用CNN学习稀疏函数只需要$\tilde{\mathcal{O}}(\log^2d)$个样本。同时，通过局部连接网络（LCN）分析了权重共享和局部性的归纳偏置的区别，得出了它们在表示需要有限平移等变和高方向选择性的函数方面的优越性。 |
| [^46] | [Tight Non-asymptotic Inference via Sub-Gaussian Intrinsic Moment Norm.](http://arxiv.org/abs/2303.07287) | 本文提出了一种通过最大化一系列归一化矩来使用子高斯内在矩范实现紧凑的非渐进推断的方法，该方法可以导致更紧的Hoeffding子高斯浓度不等式，并且可以通过子高斯图检查具有有限样本大小的子高斯数据。 |
| [^47] | [Orthogonal Polynomials Approximation Algorithm (OPAA):a functional analytic approach to estimating probability densities.](http://arxiv.org/abs/2211.08594) | OPAA是一种功能分析方法的算法，通过找到平滑的概率分布函数估计值、计算归一化权重的估计值，并使用特殊的函数空间转换来估计证据，实现了并行计算的一次通过。它适用于估计概率密度函数，尤其在贝叶斯问题中估计归一化权重。 |
| [^48] | [Transfer learning with affine model transformation.](http://arxiv.org/abs/2210.09745) | 本文提出了一种叫做仿射模型转移的迁移学习方法，通过最小化期望平方损失，可以适应各种不同的方法，包括基于神经特征提取器的方法。对于这个方法也给出了理论上的解释。 |
| [^49] | [The Manifold Scattering Transform for High-Dimensional Point Cloud Data.](http://arxiv.org/abs/2206.10078) | 多样性散射变换是一种用于高维点云数据的深度特征提取器，在实现上采用了扩散映射理论，有效用于信号分类和流形分类任务。 |
| [^50] | [Towards Size-Independent Generalization Bounds for Deep Operator Nets.](http://arxiv.org/abs/2205.11359) | 本论文研究了深度操作器网络的泛化界限问题，在一类DeepONets中证明了它们的Rademacher复杂度的界限不会随网络宽度扩展而明确变化，并利用这个结果展示了如何选择Huber损失来获得不明确依赖于网络大小的泛化误差界限。 |
| [^51] | [Statistical-Computational Trade-offs in Tensor PCA and Related Problems via Communication Complexity.](http://arxiv.org/abs/2204.07526) | 本文通过通信复杂度推导出了对于内存受限算法在张量主成分分析中的计算下界，并且指定了解决该问题的算法必须在数据样本经过次数、样本大小和所需内存之间进行权衡。这些下界暗示了许多常用算法在样本大小不够大时需要更多的迭代次数。 |
| [^52] | [The Concordance Index decomposition: A measure for a deeper understanding of survival prediction models.](http://arxiv.org/abs/2203.00144) | 本文提出了一种将Concordance指数分解成两个部分的方法，用于评估生存预测模型的性能。该分解方法可以进行更细粒度的分析，揭示不同预测方法之间的优劣。实验证明，深度学习模型更好地利用了观测事件。 |
| [^53] | [High-dimensional Inference and FDR Control for Simulated Markov Random Fields.](http://arxiv.org/abs/2202.05612) | 本文提出了一种在高维背景下进行模拟马尔可夫随机场统计推断的方法，实现了一致性，并构建了两种误发现率控制程序。 |
| [^54] | [Wavelet Networks: Scale-Translation Equivariant Learning From Raw Time-Series.](http://arxiv.org/abs/2006.05259) | 本文介绍了一种利用时间序列固有对称性构建的小波网络，其表现出嵌套的非线性小波样的时频变换，实验证明其在原始波形上优于传统的CNN。 |
| [^55] | [Fast approximations in the homogeneous Ising model for use in scene analysis.](http://arxiv.org/abs/1712.02195) | 本文提供了一种快速近似计算同质 Ising 模型中重要量的方法，该方法的表现在模拟研究中表现良好，可用于场景分析。 |

# 详细

[^1]: 用于减轻分布转变的错误回归的方法及在强化学习中的应用

    Mitigating Covariate Shift in Misspecified Regression with Applications to Reinforcement Learning. (arXiv:2401.12216v1 [stat.ML])

    [http://arxiv.org/abs/2401.12216](http://arxiv.org/abs/2401.12216)

    本文研究了在模型规范错误的条件下，由于分布转变导致的错误放大问题，并提出了一种新的算法来解决这个问题。

    

    机器学习应用中普遍存在的一个现象是分布转变，指的是训练和部署条件之间的差异。由于分布转变通常导致性能下降，因此人们一直致力于算法干预以减轻这些不利影响。在本文中，我们研究了在模型规范错误的情况下分布转变的影响，具体关注L∞-错误回归和对抗性协变量转变，其中回归目标保持不变，而协变量分布任意变化。我们发现经验风险最小化或标准最小二乘回归可能导致不可取的错误放大，其中由于规范错误而产生的误差被训练和测试分布之间的密度比放大。作为我们的主要结果，我们提出了一种新的算法——受鲁棒优化技术启发——以避免这种不可取的错误放大现象。

    A pervasive phenomenon in machine learning applications is distribution shift, where training and deployment conditions for a machine learning model differ. As distribution shift typically results in a degradation in performance, much attention has been devoted to algorithmic interventions that mitigate these detrimental effects. In this paper, we study the effect of distribution shift in the presence of model misspecification, specifically focusing on $L_{\infty}$-misspecified regression and adversarial covariate shift, where the regression target remains fixed while the covariate distribution changes arbitrarily. We show that empirical risk minimization, or standard least squares regression, can result in undesirable misspecification amplification where the error due to misspecification is amplified by the density ratio between the training and testing distributions. As our main result, we develop a new algorithm -- inspired by robust optimization techniques -- that avoids this undes
    
[^2]: 用梯度来反击维度：随机凸优化中梯度方法的泛化研究

    The Dimension Strikes Back with Gradients: Generalization of Gradient Methods in Stochastic Convex Optimization. (arXiv:2401.12058v1 [cs.LG])

    [http://arxiv.org/abs/2401.12058](http://arxiv.org/abs/2401.12058)

    本研究探讨了随机凸优化中梯度方法的泛化性能以及维度依赖性。我们发现标准的全批量梯度下降方法在维度$d = O(n^2)$时需要至少$\Omega(\sqrt{d})$个训练样本才能达到非平凡的测试误差。而对于标准的一遍随机梯度下降方法，同样的维度下界也适用。

    

    我们研究了梯度方法在基础随机凸优化设置中的泛化性能，并着重关注其维度依赖性。首先，针对全批量梯度下降（GD），我们构造了一个在维度$d=O(n^2)$下的学习问题，其中经过调整以达到经验风险的最佳性能的标准GD（使用$n$个训练样本进行训练）以常数概率收敛到一个近似的经验风险最小化器，其人口过剩风险为$\Omega(1)$。我们的界限可以转化为标准GD需要$\Omega(\sqrt{d})$个训练样本才能达到非平凡的测试误差，从而回答了Feldman（2016）和Amir、Koren、Livni（2021b）提出的一个开放问题，并表明非平凡的维度依赖性是不可避免的。此外，对于标准的一遍随机梯度下降（SGD），我们发现同样的构造技术可以提供类似的$\Omega(\sqrt{d})$的下界。

    We study the generalization performance of gradient methods in the fundamental stochastic convex optimization setting, focusing on its dimension dependence. First, for full-batch gradient descent (GD) we give a construction of a learning problem in dimension $d=O(n^2)$, where the canonical version of GD (tuned for optimal performance of the empirical risk) trained with $n$ training examples converges, with constant probability, to an approximate empirical risk minimizer with $\Omega(1)$ population excess risk. Our bound translates to a lower bound of $\Omega (\sqrt{d})$ on the number of training examples required for standard GD to reach a non-trivial test error, answering an open question raised by Feldman (2016) and Amir, Koren, and Livni (2021b) and showing that a non-trivial dimension dependence is unavoidable. Furthermore, for standard one-pass stochastic gradient descent (SGD), we show that an application of the same construction technique provides a similar $\Omega(\sqrt{d})$ lo
    
[^3]: 将统计显著性和判别能力融入模式发现中

    Integrating Statistical Significance and Discriminative Power in Pattern Discovery. (arXiv:2401.12000v1 [cs.LG])

    [http://arxiv.org/abs/2401.12000](http://arxiv.org/abs/2401.12000)

    本论文将统计显著性和判别能力融入模式发现中，提出了一种方法来同时满足模式质量和统计标准，在三元聚类任务上的实验结果显示其有效性。

    

    模式发现在多个领域的描述性和预测性任务中起着核心作用。可操作的模式必须满足严格的统计显著性标准，并且在目标变量存在时进一步具有判别能力。我们的工作解决了在现有算法中将统计显著性和判别能力标准融入模式发现的尚未深入研究的领域，同时保持模式质量。我们还解决了一些算法引入的模式质量阈值如何调整以适应这些额外标准的问题。为了测试这种方法，我们选择三元聚类任务作为模式发现案例，并扩展了两个著名的贪婪和多目标优化三元聚类算法，即δ-Trimax和TriGen，它们使用了各种模式质量标准，例如均方残差（MSR）、最小二乘线（LSL）和多斜率测量（MSL）。三个案例研究的结果显示

    Pattern discovery plays a central role in both descriptive and predictive tasks across multiple domains. Actionable patterns must meet rigorous statistical significance criteria and, in the presence of target variables, further uphold discriminative power. Our work addresses the underexplored area of guiding pattern discovery by integrating statistical significance and discriminative power criteria into state-of-the-art algorithms while preserving pattern quality. We also address how pattern quality thresholds, imposed by some algorithms, can be rectified to accommodate these additional criteria. To test the proposed methodology, we select the triclustering task as the guiding pattern discovery case and extend well-known greedy and multi-objective optimization triclustering algorithms, $\delta$-Trimax and TriGen, that use various pattern quality criteria, such as Mean Squared Residual (MSR), Least Squared Lines (LSL), and Multi Slope Measure (MSL). Results from three case studies show 
    
[^4]: 交叉验证合规风险控制

    Cross-Validation Conformal Risk Control. (arXiv:2401.11974v1 [cs.LG])

    [http://arxiv.org/abs/2401.11974](http://arxiv.org/abs/2401.11974)

    本文提出了一种基于交叉验证的新型合规风险控制方法(CV-CRC)，它扩展了一致性预测的概念，能够控制更广泛的风险函数，并在预测器集合的平均风险上提供了理论保证。

    

    合规风险控制（CRC）是一种最近提出的技术，它应用于传统的点预测器上，以提供校准保证。在CRC中推广一致性预测（CP），通过从点预测器中提取一个预测器集合来控制风险函数（如误覆盖概率或错误负例率），从而确保校准性。原始的CRC需要将可用数据集分为训练和验证数据集。当数据可用性有限时，这可能导致预测器集合效率低下。本文介绍了一种基于交叉验证而不是原始CRC的新型CRC方法。所提出的交叉验证CRC（CV-CRC）将CP的一种版本扩展到CRC，可以控制更广泛的风险函数。CV-CRC被证明在预测器集合的平均风险上具有理论保证。此外，通过数值实验证明CV-CRC在实践中的有效性。

    Conformal risk control (CRC) is a recently proposed technique that applies post-hoc to a conventional point predictor to provide calibration guarantees. Generalizing conformal prediction (CP), with CRC, calibration is ensured for a set predictor that is extracted from the point predictor to control a risk function such as the probability of miscoverage or the false negative rate. The original CRC requires the available data set to be split between training and validation data sets. This can be problematic when data availability is limited, resulting in inefficient set predictors. In this paper, a novel CRC method is introduced that is based on cross-validation, rather than on validation as the original CRC. The proposed cross-validation CRC (CV-CRC) extends a version of the jackknife-minmax from CP to CRC, allowing for the control of a broader range of risk functions. CV-CRC is proved to offer theoretical guarantees on the average risk of the set predictor. Furthermore, numerical exper
    
[^5]: RUMBoost：梯度提升的随机效用模型

    RUMBoost: Gradient Boosted Random Utility Models. (arXiv:2401.11954v1 [cs.LG])

    [http://arxiv.org/abs/2401.11954](http://arxiv.org/abs/2401.11954)

    RUMBoost模型将随机效用模型（RUMs）的解释性和行为鲁棒性与深度学习方法的泛化能力相结合，通过使用梯度提升回归树的集合来获得非线性效用函数的完整函数形式，实现了对输入变量的任何可能组合进行常数插补。

    

    本文介绍了一种新颖的离散选择建模方法，RUMBoost模型，该模型将随机效用模型（RUMs）的可解释性和行为鲁棒性与深度学习方法的泛化能力和预测能力相结合。我们通过用梯度提升回归树的集合替换RUM的效用函数中的线性参数来获得非线性效用函数的完整函数形式。这使得可以直接从数据中为所有备选方案的效用值进行分段常数插补，以适应任何可能的输入变量组合。我们对集合引入了额外的约束条件，以确保效用函数具有三个关键特征：（i）每个备选方案的效用仅依赖于该备选方案的属性，（ii）边际效用单调性，以及（iii）内在可解释的函数形式，使得模型在整个输入空间中的响应都是已知的。

    This paper introduces the RUMBoost model, a novel discrete choice modelling approach that combines the interpretability and behavioural robustness of Random Utility Models (RUMs) with the generalisation and predictive ability of deep learning methods. We obtain the full functional form of non-linear utility specifications by replacing each linear parameter in the utility functions of a RUM with an ensemble of gradient boosted regression trees. This enables piece-wise constant utility values to be imputed for all alternatives directly from the data for any possible combination of input variables. We introduce additional constraints on the ensembles to ensure three crucial features of the utility specifications: (i) dependency of the utilities of each alternative on only the attributes of that alternative, (ii) monotonicity of marginal utilities, and (iii) an intrinsically interpretable functional form, where the exact response of the model is known throughout the entire input space. Fur
    
[^6]: 通过分解梯度下降实现低胞状秩张量恢复

    Low-Tubal-Rank Tensor Recovery via Factorized Gradient Descent. (arXiv:2401.11940v1 [cs.LG])

    [http://arxiv.org/abs/2401.11940](http://arxiv.org/abs/2401.11940)

    本文提出了一种通过分解梯度下降方法解决低胞状秩张量恢复问题的高效方法，该方法通过将大张量分解为两个较小的因子张量，在减少计算成本和存储需求的同时，确保了收敛性。

    

    本文研究了从少量被破坏的线性测量中恢复具有低胞状秩结构的张量的问题。传统方法需要计算张量奇异值分解（t-SVD），这是一种计算密集的过程，使它们难以处理大规模张量。为了解决这个挑战，我们提出了一种基于类似于Burer-Monteiro（BM）方法的分解过程的高效低胞状秩张量恢复方法。具体而言，我们的基本方法涉及将一个大张量分解为两个较小的因子张量，然后通过分解梯度下降（FGD）来解决问题。该策略消除了t-SVD计算的需要，从而减少了计算成本和存储需求。我们提供了严格的理论分析，以保证FGD在无噪声和有噪声情况下的收敛性。

    This paper considers the problem of recovering a tensor with an underlying low-tubal-rank structure from a small number of corrupted linear measurements. Traditional approaches tackling such a problem require the computation of tensor Singular Value Decomposition (t-SVD), that is a computationally intensive process, rendering them impractical for dealing with large-scale tensors. Aim to address this challenge, we propose an efficient and effective low-tubal-rank tensor recovery method based on a factorization procedure akin to the Burer-Monteiro (BM) method. Precisely, our fundamental approach involves decomposing a large tensor into two smaller factor tensors, followed by solving the problem through factorized gradient descent (FGD). This strategy eliminates the need for t-SVD computation, thereby reducing computational costs and storage requirements. We provide rigorous theoretical analysis to ensure the convergence of FGD under both noise-free and noisy situations. Additionally, it 
    
[^7]: 异质性随机对照试验中时间到事件结果的亚组分析方法

    Subgroup analysis methods for time-to-event outcomes in heterogeneous randomized controlled trials. (arXiv:2401.11842v1 [stat.ME])

    [http://arxiv.org/abs/2401.11842](http://arxiv.org/abs/2401.11842)

    本论文评估了多种时间到事件结果的亚组分析算法，填补了这一领域的研究空白，并提出了一种新的数据生成过程，可以探索不同的异质性情景。

    

    非显著的随机对照试验可能隐藏了对实验性药物有良好反应的亚组，从而阻碍了后续的发展。鉴定这种异质性治疗效应对于精准医学至关重要，为此已经开发出许多事后分析方法。虽然已经进行了几个基准测试来鉴定这些方法的优点和缺点，尤其是对于二进制和连续终点，但是对于时间到事件终点的亚组分析缺乏类似的系统实证评估。本工作旨在通过评估几种时间到事件结果的亚组分析算法来填补这个空白，通过三个不同的研究问题：是否存在异质性？什么生物标志物是导致这种异质性的原因？谁是对治疗有良好反应的人？在这种情况下，我们提出了一种新的合成和半合成数据生成过程，使人们能够探索广泛的异质性情景。

    Non-significant randomized control trials can hide subgroups of good responders to experimental drugs, thus hindering subsequent development. Identifying such heterogeneous treatment effects is key for precision medicine and many post-hoc analysis methods have been developed for that purpose. While several benchmarks have been carried out to identify the strengths and weaknesses of these methods, notably for binary and continuous endpoints, similar systematic empirical evaluation of subgroup analysis for time-to-event endpoints are lacking. This work aims to fill this gap by evaluating several subgroup analysis algorithms in the context of time-to-event outcomes, by means of three different research questions: Is there heterogeneity? What are the biomarkers responsible for such heterogeneity? Who are the good responders to treatment? In this context, we propose a new synthetic and semi-synthetic data generation process that allows one to explore a wide range of heterogeneity scenarios 
    
[^8]: 使用欠阻尼 Langevin Monte Carlo 加速近似 Thompson 采样

    Accelerating Approximate Thompson Sampling with Underdamped Langevin Monte Carlo. (arXiv:2401.11665v1 [stat.ML])

    [http://arxiv.org/abs/2401.11665](http://arxiv.org/abs/2401.11665)

    本文提出了一种使用欠阻尼 Langevin Monte Carlo 加速的近似 Thompson 采样策略，通过特定势函数的设计改善了高维问题中的样本复杂度，并在高维赌博机问题中进行了验证。

    

    使用欠阻尼 Langevin Monte Carlo 的近似 Thompson 采样方法扩展了其适用范围，从高斯后验采样扩展到更一般的平滑后验。然而，在高维问题中要求高准确性时，仍然面临可扩展性问题。为了解决这个问题，我们提出了一种近似 Thompson 采样策略，利用欠阻尼 Langevin Monte Carlo，后者是模拟高维后验的通用工具。基于标准的平滑性和对数凹性条件，我们研究了使用特定势函数的加速后验集中和采样。该设计改进了实现对数遗憾的样本复杂度，从$\mathcal{\tilde O}(d)$改进到$\mathcal{\tilde O}(\sqrt{d})$。我们还通过合成实验在高维赌博机问题中经验验证了我们算法的可扩展性和鲁棒性。

    Approximate Thompson sampling with Langevin Monte Carlo broadens its reach from Gaussian posterior sampling to encompass more general smooth posteriors. However, it still encounters scalability issues in high-dimensional problems when demanding high accuracy. To address this, we propose an approximate Thompson sampling strategy, utilizing underdamped Langevin Monte Carlo, where the latter is the go-to workhorse for simulations of high-dimensional posteriors. Based on the standard smoothness and log-concavity conditions, we study the accelerated posterior concentration and sampling using a specific potential function. This design improves the sample complexity for realizing logarithmic regrets from $\mathcal{\tilde O}(d)$ to $\mathcal{\tilde O}(\sqrt{d})$. The scalability and robustness of our algorithm are also empirically validated through synthetic experiments in high-dimensional bandit problems.
    
[^9]: 通过方差降低的草图进行非参数估计

    Nonparametric Estimation via Variance-Reduced Sketching. (arXiv:2401.11646v1 [stat.ML])

    [http://arxiv.org/abs/2401.11646](http://arxiv.org/abs/2401.11646)

    本文提出了一种名为Variance-Reduced Sketching的框架，用于在高维度中估计密度函数和非参数回归函数。该方法通过将函数概念化为矩阵，并采用草图技术来降低维度灾难引起的方差，展示了鲁棒性能和显著改进。

    

    非参数模型在各个科学和工程领域中备受关注。经典的核方法在低维情况下具有数值稳定性和统计可靠性，但在高维情况下由于维度灾难变得不够适用。在本文中，我们引入了一个名为Variance-Reduced Sketching（VRS）的新框架，专门用于在降低维度灾难的同时在高维度中估计密度函数和非参数回归函数。我们的框架将多变量函数概念化为无限大小的矩阵，并借鉴了数值线性代数文献中的一种新的草图技术来降低估计问题中的方差。我们通过一系列的模拟实验和真实数据应用展示了VRS的鲁棒性能。值得注意的是，在许多密度估计问题中，VRS相较于现有的神经网络估计器和经典的核方法表现出显著的改进。

    Nonparametric models are of great interest in various scientific and engineering disciplines. Classical kernel methods, while numerically robust and statistically sound in low-dimensional settings, become inadequate in higher-dimensional settings due to the curse of dimensionality. In this paper, we introduce a new framework called Variance-Reduced Sketching (VRS), specifically designed to estimate density functions and nonparametric regression functions in higher dimensions with a reduced curse of dimensionality. Our framework conceptualizes multivariable functions as infinite-size matrices, and facilitates a new sketching technique motivated by numerical linear algebra literature to reduce the variance in estimation problems. We demonstrate the robust numerical performance of VRS through a series of simulated experiments and real-world data applications. Notably, VRS shows remarkable improvement over existing neural network estimators and classical kernel methods in numerous density 
    
[^10]: 用于克服灾难性过拟合的高效本地线性正则化

    Efficient local linearity regularization to overcome catastrophic overfitting. (arXiv:2401.11618v1 [cs.LG])

    [http://arxiv.org/abs/2401.11618](http://arxiv.org/abs/2401.11618)

    本研究引入了一种名为ELLE的正则化项，用于高效地减轻单步对抗性训练中的灾难性过拟合。它能够保持损失函数在输入上的局部线性性，与传统的正则化方法相比，ELLE更加高效，能够有效应对大对抗性扰动和长训练计划等困难情况。

    

    单步对抗性训练中的灾难性过拟合 (CO) 导致对抗性测试准确率突然下降（甚至降至0%）。对于使用多步对抗性训练训练的模型，已观察到损失函数在输入上具有局部线性性，但这种特性在单步对抗性训练中丢失。为了解决单步对抗性训练中的CO问题，提出了几种通过正则化来强制损失函数局部线性性的方法。然而，由于双重反向传播，这些正则化项会显著减慢训练速度。与之相反，在本研究中，我们引入一种称为ELLE的正则化项，以在经典对抗性训练评估中有效且高效地减轻CO问题，在一些更困难的情况下也能起作用，例如大对抗性扰动和长训练计划。我们的正则化项在理论上与损失函数的曲率有联系，并且通过避免双重反向传播而具有比先前方法更低的计算成本。通过彻底的实验研究...

    Catastrophic overfitting (CO) in single-step adversarial training (AT) results in abrupt drops in the adversarial test accuracy (even down to 0%). For models trained with multi-step AT, it has been observed that the loss function behaves locally linearly with respect to the input, this is however lost in single-step AT. To address CO in single-step AT, several methods have been proposed to enforce local linearity of the loss via regularization. However, these regularization terms considerably slow down training due to Double Backpropagation. Instead, in this work, we introduce a regularization term, called ELLE, to mitigate CO effectively and efficiently in classical AT evaluations, as well as some more difficult regimes, e.g., large adversarial perturbations and long training schedules. Our regularization term can be theoretically linked to curvature of the loss function and is computationally cheaper than previous methods by avoiding Double Backpropagation. Our thorough experimental 
    
[^11]: 理解后期学习率衰减的泛化优势

    Understanding the Generalization Benefits of Late Learning Rate Decay. (arXiv:2401.11600v1 [cs.LG])

    [http://arxiv.org/abs/2401.11600](http://arxiv.org/abs/2401.11600)

    本文研究了为什么使用大学习率长时间训练神经网络可以实现更好的泛化。通过观察训练和测试损失之间的关系，我们发现大学习率下的训练轨迹能够接近测试损失的最小值附近。基于这个发现，我们提出了一个与真实神经网络类似的非线性模型，并证明了使用大学习率进行延长阶段的训练可以实现更接近最优泛化的效果。

    

    为什么用大学习率长时间训练的神经网络通常能够实现更好的泛化呢？本文通过研究神经网络中训练损失和测试损失之间的关系来探讨这个问题。通过可视化这些损失情况，我们观察到在大学习率下的训练轨迹会通过训练损失的极小值流形，最终接近于测试损失的极小值附近。在这个发现的基础上，我们引入了一个非线性模型，其损失景观与真实神经网络观察到的相似。通过在我们的模型上使用SGD研究训练过程，我们证明了在大学习率下的延长阶段可以将我们的模型引导向训练损失的最小范数解，这可能实现接近最优的泛化，从而证实了后期学习率衰减的经验优势。

    Why do neural networks trained with large learning rates for a longer time often lead to better generalization? In this paper, we delve into this question by examining the relation between training and testing loss in neural networks. Through visualization of these losses, we note that the training trajectory with a large learning rate navigates through the minima manifold of the training loss, finally nearing the neighborhood of the testing loss minimum. Motivated by these findings, we introduce a nonlinear model whose loss landscapes mirror those observed for real neural networks. Upon investigating the training process using SGD on our model, we demonstrate that an extended phase with a large learning rate steers our model towards the minimum norm solution of the training loss, which may achieve near-optimal generalization, thereby affirming the empirically observed benefits of late learning rate decay.
    
[^12]: Thompson采样用于具有噪音上下文的随机赌臂问题的信息论性后悔分析

    Thompson Sampling for Stochastic Bandits with Noisy Contexts: An Information-Theoretic Regret Analysis. (arXiv:2401.11565v1 [cs.LG])

    [http://arxiv.org/abs/2401.11565](http://arxiv.org/abs/2401.11565)

    本文研究了具有噪音上下文的随机赌臂问题，并提出了一种Thompson采样算法，通过贝叶斯框架进行分析，证明了算法的贝叶斯后悔，并扩展了问题到延迟观察真实上下文的情况，并实证了算法的性能。

    

    我们研究了一种随机上下文线性赌臂问题，其中代理通过一个未知噪声参数的噪声信道观察到真实上下文的噪声，我们的目标是设计一个动作策略，可以近似于具有奖励模型、噪声参数和从观察到的噪声上下文中真实上下文的预测分布的oracle的动作策略。在贝叶斯框架下，我们引入了一种针对具有高斯上下文噪声的高斯赌臂的Thompson采样算法。采用信息论分析，我们证明了我们的算法相对于oracle的动作策略的贝叶斯后悔。我们还将这个问题扩展到了代理在接收到奖励后延迟观察到真实上下文的情况，并展示了延迟真实上下文导致更低的贝叶斯后悔。最后，我们通过与基线算法的比较实证地展示了所提出算法的性能。

    We explore a stochastic contextual linear bandit problem where the agent observes a noisy, corrupted version of the true context through a noise channel with an unknown noise parameter. Our objective is to design an action policy that can approximate" that of an oracle, which has access to the reward model, the channel parameter, and the predictive distribution of the true context from the observed noisy context. In a Bayesian framework, we introduce a Thompson sampling algorithm for Gaussian bandits with Gaussian context noise. Adopting an information-theoretic analysis, we demonstrate the Bayesian regret of our algorithm concerning the oracle's action policy. We also extend this problem to a scenario where the agent observes the true context with some delay after receiving the reward and show that delayed true contexts lead to lower Bayesian regret. Finally, we empirically demonstrate the performance of the proposed algorithms against baselines.
    
[^13]: 使用Wasserstein距离进行加权以增强选择性

    Enhancing selectivity using Wasserstein distance based reweighing. (arXiv:2401.11562v1 [stat.ML])

    [http://arxiv.org/abs/2401.11562](http://arxiv.org/abs/2401.11562)

    我们设计了一种使用Wasserstein距离进行加权的算法，在标记的数据集上训练神经网络可以逼近在其他数据集上训练得到的结果。我们证明了算法可以输出接近最优的加权，且算法简单可扩展。我们的算法可以有意地引入分布偏移进行多目标优化。作为应用实例，我们训练了一个神经网络来识别对细胞信号传导的MAP激酶具有非结合性的小分子结合物。

    

    给定两个标记数据集𝒮和𝒯，我们设计了一种简单高效的贪婪算法来对损失函数进行加权，使得在𝒮上训练得到的神经网络权重的极限分布逼近在𝒯上训练得到的极限分布。在理论方面，我们证明了当输入数据集的度量熵有界时，我们的贪婪算法输出接近最优的加权，即网络权重的两个不变分布在总变差距离上可以证明接近。此外，该算法简单可扩展，并且我们还证明了算法的效率上界。我们的算法可以有意地引入分布偏移以进行（软）多目标优化。作为一个动机应用，我们训练了一个神经网络来识别对MNK2（一种细胞信号传导的MAP激酶）具有非结合性的小分子结合物。

    Given two labeled data-sets $\mathcal{S}$ and $\mathcal{T}$, we design a simple and efficient greedy algorithm to reweigh the loss function such that the limiting distribution of the neural network weights that result from training on $\mathcal{S}$ approaches the limiting distribution that would have resulted by training on $\mathcal{T}$.  On the theoretical side, we prove that when the metric entropy of the input data-sets is bounded, our greedy algorithm outputs a close to optimal reweighing, i.e., the two invariant distributions of network weights will be provably close in total variation distance. Moreover, the algorithm is simple and scalable, and we prove bounds on the efficiency of the algorithm as well.  Our algorithm can deliberately introduce distribution shift to perform (soft) multi-criteria optimization. As a motivating application, we train a neural net to recognize small molecule binders to MNK2 (a MAP Kinase, responsible for cell signaling) which are non-binders to MNK1
    
[^14]: MoMA: 模型为基础的镜像上升算法的离线强化学习

    MoMA: Model-based Mirror Ascent for Offline Reinforcement Learning. (arXiv:2401.11380v1 [cs.LG])

    [http://arxiv.org/abs/2401.11380](http://arxiv.org/abs/2401.11380)

    MoMA提出了一种模型为基础的镜像上升算法，通过使用无限制的策略类别和一般函数逼近来实现离线强化学习，充分利用了模型为基础方法的优势。

    

    模型为基础的离线强化学习方法在许多决策问题中取得了最先进的性能，得益于它们的样本效率和通用性。尽管取得了这些进展，但现有的模型为基础的离线强化学习方法要么侧重于理论研究而没有开发实际算法，要么依赖于受限的参数化策略空间，因此没有充分利用模型为基础方法固有的无限制策略空间的优势。为了解决这个限制，我们开发了MoMA，一个在离线数据部分覆盖下使用一般函数逼近的模型为基础的镜像上升算法。MoMA通过采用无限制的策略类别，区别于现有文献。在每个迭代中，MoMA在策略评估步骤中通过在过渡模型的置信区间内进行最小化过程来保守地估计值函数，然后使用一般函数逼近来更新策略，而不是常用的方法。

    Model-based offline reinforcement learning methods (RL) have achieved state-of-the-art performance in many decision-making problems thanks to their sample efficiency and generalizability. Despite these advancements, existing model-based offline RL approaches either focus on theoretical studies without developing practical algorithms or rely on a restricted parametric policy space, thus not fully leveraging the advantages of an unrestricted policy space inherent to model-based methods. To address this limitation, we develop MoMA, a model-based mirror ascent algorithm with general function approximations under partial coverage of offline data. MoMA distinguishes itself from existing literature by employing an unrestricted policy class. In each iteration, MoMA conservatively estimates the value function by a minimization procedure within a confidence set of transition models in the policy evaluation step, then updates the policy with general function approximations instead of commonly-use
    
[^15]: 量子机器学习：从NISQ到容错。

    Quantum Machine Learning: from NISQ to Fault Tolerance. (arXiv:2401.11351v1 [quant-ph])

    [http://arxiv.org/abs/2401.11351](http://arxiv.org/abs/2401.11351)

    本文提供了对量子机器学习领域的全面回顾，涵盖了在NISQ技术和容错量子计算硬件上使用的技术和算法，并深入讨论了与量子机器学习相关的基本概念和统计学习理论。

    

    量子机器学习是在量子设备上运行机器学习算法的过程，在学术界和商业界引起了广泛关注。本文对量子机器学习领域涌现的各种概念进行了全面而公正的回顾。这包括在噪声中间尺度量子（NISQ）技术中使用的技术，以及与容错量子计算硬件兼容的算法方法。我们的回顾涵盖了与量子机器学习相关的基本概念、算法和统计学习理论。

    Quantum machine learning, which involves running machine learning algorithms on quantum devices, has garnered significant attention in both academic and business circles. In this paper, we offer a comprehensive and unbiased review of the various concepts that have emerged in the field of quantum machine learning. This includes techniques used in Noisy Intermediate-Scale Quantum (NISQ) technologies and approaches for algorithms compatible with fault-tolerant quantum computing hardware. Our review covers fundamental concepts, algorithms, and the statistical learning theory pertinent to quantum machine learning.
    
[^16]: 通过（正交）完全无偏的截尾学习来估计生存结果的异质治疗效应

    Estimating heterogeneous treatment effect from survival outcomes via (orthogonal) censoring unbiased learning. (arXiv:2401.11263v1 [stat.ME])

    [http://arxiv.org/abs/2401.11263](http://arxiv.org/abs/2401.11263)

    该论文开发了一种适用于具有和没有竞争风险的生存结果的截尾无偏变换方法，可以估计异质治疗效应。这种方法可以应用于更多最先进的适用于被截尾结果的HTE学习方法，并提供了限制有限样本过度风险的方法。

    

    从观察数据中估计异质治疗效应（HTE）的方法主要集中在连续或二元结果上，较少关注生存结果，几乎没有关注竞争风险情景。在这项工作中，我们开发了适用于具有和没有竞争风险的生存结果的截尾无偏变换（CUTs）。使用这些CUTs将时间到事件结果转换后，对连续结果的HTE学习方法的直接应用可以产生一致估计的异质累积发生率效应、总效应和可分离直接效应。我们的CUTs可以使用比以前更多的最先进的适用于被截尾结果的HTE学习方法，特别是在竞争风险情景下。我们提供了通用的无模型学习特定oracle不等式来限制有限样本的过度风险。oracle效率结果取决于一个oracle选择器和从所有步骤中估计的干扰函数。

    Methods for estimating heterogeneous treatment effects (HTE) from observational data have largely focused on continuous or binary outcomes, with less attention paid to survival outcomes and almost none to settings with competing risks. In this work, we develop censoring unbiased transformations (CUTs) for survival outcomes both with and without competing risks.After converting time-to-event outcomes using these CUTs, direct application of HTE learners for continuous outcomes yields consistent estimates of heterogeneous cumulative incidence effects, total effects, and separable direct effects. Our CUTs enable application of a much larger set of state of the art HTE learners for censored outcomes than had previously been available, especially in competing risks settings. We provide generic model-free learner-specific oracle inequalities bounding the finite-sample excess risk. The oracle efficiency results depend on the oracle selector and estimated nuisance functions from all steps invol
    
[^17]: AFS-BM:通过自适应特征选择和二值屏蔽增强模型性能

    AFS-BM: Enhancing Model Performance through Adaptive Feature Selection with Binary Masking. (arXiv:2401.11250v1 [cs.LG])

    [http://arxiv.org/abs/2401.11250](http://arxiv.org/abs/2401.11250)

    AFS-BM通过联合优化实现了自适应特征选择和模型训练，提高了模型准确性并减少了计算需求。

    

    我们研究了机器学习领域中特征选择的问题，这是该领域中最关键的主题之一。尽管存在许多特征选择方法，但是这些方法面临可扩展性、处理高维数据、处理相关特征、适应可变特征重要性和整合领域知识等挑战。为了解决这些问题，我们引入了“自适应特征选择和二值屏蔽”(AFS-BM)。AFS-BM通过联合优化来同时进行特征选择和模型训练。具体而言，我们通过联合优化和二值屏蔽，在训练过程中持续调整特征集和模型参数。这种方法显著提高了模型的准确性，并减少了计算需求。我们进行了一系列的实验证明，将AFS-BM与已有的特征选择方法进行了比较。

    We study the problem of feature selection in general machine learning (ML) context, which is one of the most critical subjects in the field. Although, there exist many feature selection methods, however, these methods face challenges such as scalability, managing high-dimensional data, dealing with correlated features, adapting to variable feature importance, and integrating domain knowledge. To this end, we introduce the ``Adaptive Feature Selection with Binary Masking" (AFS-BM) which remedies these problems. AFS-BM achieves this by joint optimization for simultaneous feature selection and model training. In particular, we do the joint optimization and binary masking to continuously adapt the set of features and model parameters during the training process. This approach leads to significant improvements in model accuracy and a reduction in computational requirements. We provide an extensive set of experiments where we compare AFS-BM with the established feature selection methods usin
    
[^18]: 通过工具变量法识别和估计条件平均偏因果效应

    Identification and Estimation of Conditional Average Partial Causal Effects via Instrumental Variable. (arXiv:2401.11130v1 [cs.LG])

    [http://arxiv.org/abs/2401.11130](http://arxiv.org/abs/2401.11130)

    本文提出了一种通过工具变量法识别和估计连续处理的因果效应异质性的方法，并开发了三类相应的估计器，并对其进行了统计性质分析。

    

    近年来，对于估计异质因果效应的兴趣日益增加。本文引入了条件平均偏因果效应（CAPCE），以揭示连续处理的因果效应的异质性。我们给出了在工具变量设置下识别CAPCE的条件。我们开发了三类CAPCE估计器：筛选、参数化和再生核希尔伯特空间（RKHS）-基础，分析了它们的统计性质。我们通过合成和真实数据对提出的CAPCE估计器进行了说明。

    There has been considerable recent interest in estimating heterogeneous causal effects. In this paper, we introduce conditional average partial causal effects (CAPCE) to reveal the heterogeneity of causal effects with continuous treatment. We provide conditions for identifying CAPCE in an instrumental variable setting. We develop three families of CAPCE estimators: sieve, parametric, and reproducing kernel Hilbert space (RKHS)-based, and analyze their statistical properties. We illustrate the proposed CAPCE estimators on synthetic and real-world data.
    
[^19]: 加权最近邻算法的高效数据Shapley值计算方法

    Efficient Data Shapley for Weighted Nearest Neighbor Algorithms. (arXiv:2401.11103v1 [cs.DS])

    [http://arxiv.org/abs/2401.11103](http://arxiv.org/abs/2401.11103)

    本研究提出了一种解决加权K最近邻算法高效计算Data Shapley值的方法，并通过实验证明其在数据质量判别方面优于未加权版本。

    

    本文旨在解决数据评估文献中关于加权K最近邻算法的数据Shapley值（WKNN-Shapley）高效计算的问题。通过将硬标签KNN的准确度与离散权重视为效用函数，我们将WKNN-Shapley值的计算转化为一个计数问题，并引入了一个二次时间复杂度的算法，从而实现了对现有文献中O(N^K)的最佳结果的显著改进。我们开发了一个确定性近似算法，进一步提高了计算效率，同时保持了Shapley值的关键公平性质。通过大量实验，我们展示了WKNN-Shapley值的计算效率以及与未加权版本相比在判断数据质量方面的优势。

    This work aims to address an open problem in data valuation literature concerning the efficient computation of Data Shapley for weighted $K$ nearest neighbor algorithm (WKNN-Shapley). By considering the accuracy of hard-label KNN with discretized weights as the utility function, we reframe the computation of WKNN-Shapley into a counting problem and introduce a quadratic-time algorithm, presenting a notable improvement from $O(N^K)$, the best result from existing literature. We develop a deterministic approximation algorithm that further improves computational efficiency while maintaining the key fairness properties of the Shapley value. Through extensive experiments, we demonstrate WKNN-Shapley's computational efficiency and its superior performance in discerning data quality compared to its unweighted counterpart.
    
[^20]: 从聚合响应中学习：实例级别与包级别的损失函数比较

    Learning from Aggregate responses: Instance Level versus Bag Level Loss Functions. (arXiv:2401.11081v1 [cs.LG])

    [http://arxiv.org/abs/2401.11081](http://arxiv.org/abs/2401.11081)

    本文研究了从聚合响应中学习的两种损失函数：包级别损失和实例级别损失，并发现实例级别损失可以被视为包级别损失的正则化形式。

    

    由于隐私问题的增加，在许多实际应用中，训练数据在与学习者共享之前会被聚合起来，以保护用户敏感响应的隐私。在聚合学习框架中，数据集被分组成样本的包，每个包只提供一个聚合响应，提供了该包中个体响应的摘要。本文研究了从聚合响应中学习的两种自然损失函数：包级别损失和实例级别损失。在前者中，模型通过最小化聚合响应与聚合模型预测之间的损失来学习，而在后者中，模型旨在将个体预测与聚合响应拟合。在这项工作中，我们表明实例级别损失可以被视为包级别损失的正则化形式。这个观察让我们能够比较这两种方法关于所得估计值的偏差和方差，并引入了一种新的插值。

    Due to the rise of privacy concerns, in many practical applications the training data is aggregated before being shared with the learner, in order to protect privacy of users' sensitive responses. In an aggregate learning framework, the dataset is grouped into bags of samples, where each bag is available only with an aggregate response, providing a summary of individuals' responses in that bag. In this paper, we study two natural loss functions for learning from aggregate responses: bag-level loss and the instance-level loss. In the former, the model is learnt by minimizing a loss between aggregate responses and aggregate model predictions, while in the latter the model aims to fit individual predictions to the aggregate responses. In this work, we show that the instance-level loss can be perceived as a regularized form of the bag-level loss. This observation lets us compare the two approaches with respect to bias and variance of the resulting estimators, and introduce a novel interpol
    
[^21]: 具有结构化变分族的可证伸缩性黑盒变分推断

    Provably Scalable Black-Box Variational Inference with Structured Variational Families. (arXiv:2401.10989v1 [stat.ML])

    [http://arxiv.org/abs/2401.10989](http://arxiv.org/abs/2401.10989)

    本文研究了均值场变分族和满秩变分族之间的理论中间地带：结构化变分族，并通过理论证明结构化变分族可以在迭代复杂性上表现更好，缩放效果更好。

    

    已知具有满秩协方差逼近的变分族在黑盒变分推断中表现不佳，无论是从实证上还是理论上。事实上，最近对黑盒变分推断的计算复杂性结果表明，与均值场变分族相比，满秩变分族在问题的维度上扩展得很差。这对具有本地变量的分层贝叶斯模型尤为关键，它们的维度随着数据集的大小而增加。因此，迭代复杂性对数据集大小N存在明确的O(N^2)依赖。在本文中，我们探索了均值场变分族和满秩变分族之间的理论中间地带：结构化变分族。我们严格证明了某些尺度矩阵结构可以实现更好的迭代复杂性O(N)，从而与N的缩放更好地匹配。我们在现实中验证了我们的理论结果

    Variational families with full-rank covariance approximations are known not to work well in black-box variational inference (BBVI), both empirically and theoretically. In fact, recent computational complexity results for BBVI have established that full-rank variational families scale poorly with the dimensionality of the problem compared to e.g. mean field families. This is particularly critical to hierarchical Bayesian models with local variables; their dimensionality increases with the size of the datasets. Consequently, one gets an iteration complexity with an explicit $\mathcal{O}(N^2)$ dependence on the dataset size $N$. In this paper, we explore a theoretical middle ground between mean-field variational families and full-rank families: structured variational families. We rigorously prove that certain scale matrix structures can achieve a better iteration complexity of $\mathcal{O}(N)$, implying better scaling with respect to $N$. We empirically verify our theoretical results on l
    
[^22]: 使用半正定规划的去偏和局部分析进行人群聚类

    Debiasing and a local analysis for population clustering using semidefinite programming. (arXiv:2401.10927v1 [stat.ML])

    [http://arxiv.org/abs/2401.10927](http://arxiv.org/abs/2401.10927)

    本文研究了使用半正定规划进行人群聚类的问题，并提出了计算高效的算法。这些算法可以根据小样本数据的原始种群将数据分为两组，适用于种群之间差异较小的情况。

    

    本文考虑了从混合的2个次高斯分布中抽取的小数据样本的分区问题。我们分析了同一作者提出的计算高效的算法，将数据根据其原始种群大致分为两组，给定一个小样本。本文的研究动机是将个体根据其原始种群使用p个标记进行聚类，当任意两个种群之间的差异很小时。我们基于整数二次规划的半正定松弛形式构建，该规划问题本质上是在一个图上找到最大割，其中割中的边权重表示基于它们的p个特征的两个节点之间的不相似度得分。我们用Δ^2:=pγ来表示两个中心（均值向量）之间的ℓ_2^2距离，即μ^(1), μ^(2)∈ℝ^p。目标是在交换精度和计算效率之间提供全面的权衡。

    In this paper, we consider the problem of partitioning a small data sample of size $n$ drawn from a mixture of $2$ sub-gaussian distributions. In particular, we analyze computational efficient algorithms proposed by the same author, to partition data into two groups approximately according to their population of origin given a small sample. This work is motivated by the application of clustering individuals according to their population of origin using $p$ markers, when the divergence between any two of the populations is small. We build upon the semidefinite relaxation of an integer quadratic program that is formulated essentially as finding the maximum cut on a graph, where edge weights in the cut represent dissimilarity scores between two nodes based on their $p$ features. Here we use $\Delta^2 :=p \gamma$ to denote the $\ell_2^2$ distance between two centers (mean vectors), namely, $\mu^{(1)}$, $\mu^{(2)}$ $\in$ $\mathbb{R}^p$. The goal is to allow a full range of tradeoffs between
    
[^23]: 在具有通用随机牛顿算法应用于随机优化的情况下，关于Hessian矩阵的逆的在线估计

    Online estimation of the inverse of the Hessian for stochastic optimization with application to universal stochastic Newton algorithms. (arXiv:2401.10923v1 [math.OC])

    [http://arxiv.org/abs/2401.10923](http://arxiv.org/abs/2401.10923)

    该论文提出了一种在线估计Hessian矩阵逆的方法，利用Robbins-Monro过程，能够 drastical reducescomputational complexity,并发展了通用的随机牛顿算法，研究了所提方法的渐进效率。

    

    本文针对以期望形式表示的凸函数的次优随机优化问题，提出了一种直接递归估计逆Hessian矩阵的技术，采用了Robbins-Monro过程。该方法能够大幅降低计算复杂性，并且允许开发通用的随机牛顿方法，并研究所提方法的渐近效率。这项工作扩展了在随机优化中二阶算法的应用范围。

    This paper addresses second-order stochastic optimization for estimating the minimizer of a convex function written as an expectation. A direct recursive estimation technique for the inverse Hessian matrix using a Robbins-Monro procedure is introduced. This approach enables to drastically reduces computational complexity. Above all, it allows to develop universal stochastic Newton methods and investigate the asymptotic efficiency of the proposed approach. This work so expands the application scope of secondorder algorithms in stochastic optimization.
    
[^24]: 两层网络训练中的早期对齐是一把双刃剑

    Early alignment in two-layer networks training is a two-edged sword. (arXiv:2401.10791v1 [cs.LG])

    [http://arxiv.org/abs/2401.10791](http://arxiv.org/abs/2401.10791)

    本文研究了两层网络训练中的早期对齐现象，发现在小初始化和一个隐藏的ReLU层网络中，神经元会在训练的早期阶段向关键方向进行对齐，导致网络稀疏表示以及梯度流在收敛时的隐含偏好。然而，这种稀疏诱导的对齐也使得训练目标的最小化变得困难。

    

    使用一阶优化方法训练神经网络是深度学习成功的核心。初始化的规模是一个关键因素，因为小的初始化通常与特征学习模式相关，在这种模式下，梯度下降对简单解隐含偏好。本文提供了早期对齐阶段的普遍和量化描述，最初由Maennel等人提出。对于小初始化和一个隐藏的ReLU层网络，训练动态的早期阶段导致神经元向关键方向进行对齐。这种对齐引发了网络的稀疏表示，这与梯度流在收敛时的隐含偏好直接相关。然而，这种稀疏诱导的对齐是以在最小化训练目标方面遇到困难为代价的：我们还提供了一个简单的数据示例，其中超参数网络无法收敛到全局最小值。

    Training neural networks with first order optimisation methods is at the core of the empirical success of deep learning. The scale of initialisation is a crucial factor, as small initialisations are generally associated to a feature learning regime, for which gradient descent is implicitly biased towards simple solutions. This work provides a general and quantitative description of the early alignment phase, originally introduced by Maennel et al. (2018) . For small initialisation and one hidden ReLU layer networks, the early stage of the training dynamics leads to an alignment of the neurons towards key directions. This alignment induces a sparse representation of the network, which is directly related to the implicit bias of gradient flow at convergence. This sparsity inducing alignment however comes at the expense of difficulties in minimising the training objective: we also provide a simple data example for which overparameterised networks fail to converge towards global minima and
    
[^25]: Intrinsic Dataset Properties对泛化能力的影响：揭示自然图像和医学图像之间的学习差异

    The Effect of Intrinsic Dataset Properties on Generalization: Unraveling Learning Differences Between Natural and Medical Images. (arXiv:2401.08865v1 [cs.CV])

    [http://arxiv.org/abs/2401.08865](http://arxiv.org/abs/2401.08865)

    本文研究了神经网络在自然图像和医学图像领域学习时的差异，提出了一个与训练集维度有关的泛化缩放定律，并认为医学图像数据集更高的固有“标签锐度”可能是两个领域之间显著差异的部分原因。

    

    本文研究了神经网络在不同图像领域学习时的差异，这在从自然图像到其他专门领域（如医学图像）采用计算机视觉技术时通常被忽视。最近的研究发现，训练集的固有维度($d_{data}$)与网络的泛化错误一般会增加。然而，医学（放射学）和自然图像领域之间的这种关系的陡峭程度存在显著差异，且无现有的理论解释。我们通过建立并经验证一个与$d_{data}$相关的泛化缩放定律来解决这个知识空白，并提出考虑到医学图像数据集更高的固有“标签锐度”($K_F$)这一度量指标可以部分解释这两个领域之间的显著缩放差异。接下来，我们展示了利用测量这一指标可以提供的额外好处。

    This paper investigates discrepancies in how neural networks learn from different imaging domains, which are commonly overlooked when adopting computer vision techniques from the domain of natural images to other specialized domains such as medical images. Recent works have found that the generalization error of a trained network typically increases with the intrinsic dimension ($d_{data}$) of its training set. Yet, the steepness of this relationship varies significantly between medical (radiological) and natural imaging domains, with no existing theoretical explanation. We address this gap in knowledge by establishing and empirically validating a generalization scaling law with respect to $d_{data}$, and propose that the substantial scaling discrepancy between the two considered domains may be at least partially attributed to the higher intrinsic "label sharpness" ($K_F$) of medical imaging datasets, a metric which we propose. Next, we demonstrate an additional benefit of measuring th
    
[^26]: 带有变点的神经随机微分方程：一种生成对抗方法

    Neural Stochastic Differential Equations with Change Points: A Generative Adversarial Approach. (arXiv:2312.13152v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.13152](http://arxiv.org/abs/2312.13152)

    本文提出了一种使用神经随机微分方程来建模时间序列的变点检测算法，通过生成对抗网络学习每个变点对应的神经随机微分方程的参数，并通过GAN判别器的输出检测变点。验证结果表明，该算法在合成和真实数据集上的性能优于传统方法。

    

    随机微分方程广泛用于建模真实世界的随机现象。现有的工作主要集中在时间序列由单个随机微分方程建模的情况，这对于具有分布偏移的时间序列建模可能是局限的。在这项工作中，我们提出了一种将时间序列建模为神经随机微分方程的变点检测算法。给定一个时间序列数据集，所提出的方法联合学习未知的变点和与每个变点对应的不同神经随机微分方程模型的参数。具体而言，神经随机微分方程在生成对抗网络（GAN）的框架下进行学习，并且变点是根据GAN判别器的输出在前向传递中被检测到。在所提出的算法的每一步中，变点和SDE模型参数以交替的方式进行更新。通过对合成和真实数据集的数值结果进行验证，以验证我们的算法与传统方法的性能比较。

    Stochastic differential equations (SDEs) have been widely used to model real world random phenomena. Existing works mainly focus on the case where the time series is modeled by a single SDE, which might be restrictive for modeling time series with distributional shift. In this work, we propose a change point detection algorithm for time series modeled as neural SDEs. Given a time series dataset, the proposed method jointly learns the unknown change points and the parameters of distinct neural SDE models corresponding to each change point. Specifically, the SDEs are learned under the framework of generative adversarial networks (GANs) and the change points are detected based on the output of the GAN discriminator in a forward pass. At each step of the proposed algorithm, the change points and the SDE model parameters are updated in an alternating fashion. Numerical results on both synthetic and real datasets are provided to validate the performance of our algorithm in comparison to clas
    
[^27]: 追求最优统计水印技术

    Towards Optimal Statistical Watermarking. (arXiv:2312.07930v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.07930](http://arxiv.org/abs/2312.07930)

    追求最优统计水印技术。通过将统计水印技术视为假设检验问题并引入伪随机生成器，我们实现了输出令牌和拒绝区域的耦合，实现了第一类错误和第二类错误之间的非平凡权衡，同时提出了最统一最有力的水印和最小化第二类错误的解决方案。我们还提供了独立同分布令牌数量的上下界，突显了改进的潜力。此外，我们还探讨了鲁棒性水印问题。

    

    我们将统计水印技术作为一个假设检验问题进行研究，这是一个泛化了所有之前统计水印方法的通用框架。我们的关键是通过实践中的伪随机生成器实现输出令牌和拒绝区域的耦合，从而允许在第一类错误和第二类错误之间进行非平凡的权衡。我们在一般的假设检验环境下表征了最统一最有力的水印以及在模型无关的环境中最小化第二类错误。在输出是$n$个令牌的常见情况下，我们对需要保证小的第一类和第二类错误的独立同分布令牌数量建立了近乎匹配的上下界。与之前的工作中的$ h ^ {-2} $速率相比，我们相对于每个令牌的平均熵$h$的速率为$ \Theta(h ^ {-1} \log (1/h)) $，突显了改进的潜力。此外，我们提出了鲁棒性水印问题，其中用户都是...

    We study statistical watermarking by formulating it as a hypothesis testing problem, a general framework which subsumes all previous statistical watermarking methods. Key to our formulation is a coupling of the output tokens and the rejection region, realized by pseudo-random generators in practice, that allows non-trivial trade-off between the Type I error and Type II error. We characterize the Uniformly Most Powerful (UMP) watermark in the general hypothesis testing setting and the minimax Type II error in the model-agnostic setting. In the common scenario where the output is a sequence of $n$ tokens, we establish nearly matching upper and lower bounds on the number of i.i.d. tokens required to guarantee small Type I and Type II errors. Our rate of $\Theta(h^{-1} \log (1/h))$ with respect to the average entropy per token $h$ highlights potentials for improvement from the rate of $h^{-2}$ in the previous works. Moreover, we formulate the robust watermarking problem where users are all
    
[^28]: 最优化多分布学习

    Optimal Multi-Distribution Learning. (arXiv:2312.05134v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.05134](http://arxiv.org/abs/2312.05134)

    本论文提出了一种最优化多分布学习的方法，通过自适应采样来实现数据高效的学习。针对Vapnik-Chervonenkis (VC)维数为d的假设类，算法可以生成一个ε-最优随机假设，并且样本复杂度与最佳下界保持一致。同时，该算法的思想和理论还被进一步扩展以适应Rademacher类。最终提出的算法是奥拉克尔高效的，仅访问假设类。

    

    多分布学习（MDL）旨在学习一个共享模型，使得在k个不同的数据分布下，最小化最坏情况风险，已成为适应健壮性、公平性、多组合作等需求的统一框架。实现数据高效的MDL需要在学习过程中进行自适应采样，也称为按需采样。然而，最优样本复杂度的上下界之间存在较大差距。针对Vapnik-Chervonenkis（VC）维数为d的假设类，我们提出了一种新颖的算法，可生成一个ε-最优随机假设，其样本复杂度接近于（d+k）/ε^2（在某些对数因子中），与已知的最佳下界匹配。我们的算法思想和理论被进一步扩展，以适应Rademacher类。提出的算法是奥拉克尔高效的，仅仅访问假设类

    Multi-distribution learning (MDL), which seeks to learn a shared model that minimizes the worst-case risk across $k$ distinct data distributions, has emerged as a unified framework in response to the evolving demand for robustness, fairness, multi-group collaboration, etc. Achieving data-efficient MDL necessitates adaptive sampling, also called on-demand sampling, throughout the learning process. However, there exist substantial gaps between the state-of-the-art upper and lower bounds on the optimal sample complexity. Focusing on a hypothesis class of Vapnik-Chervonenkis (VC) dimension $d$, we propose a novel algorithm that yields an $varepsilon$-optimal randomized hypothesis with a sample complexity on the order of $(d+k)/\varepsilon^2$ (modulo some logarithmic factor), matching the best-known lower bound. Our algorithmic ideas and theory have been further extended to accommodate Rademacher classes. The proposed algorithms are oracle-efficient, which access the hypothesis class solely
    
[^29]: 对于核机器在预处理中的Nystrom逼近

    On the Nystrom Approximation for Preconditioning in Kernel Machines. (arXiv:2312.03311v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2312.03311](http://arxiv.org/abs/2312.03311)

    本文分析了核机器预处理中使用Nystrom逼近的权衡。研究表明，使用对数大小的样本能够让Nystrom逼近的预处理器几乎与梯度下降同样有效地加速。

    

    核方法是机器学习中一类流行的非线性预测模型。学习核模型的可扩展算法需要具有迭代性质，但由于糟糕的条件，收敛可能很慢。谱预处理是加快训练核模型迭代算法收敛速度的重要工具。然而，计算和存储谱预处理器可能代价高昂，会导致大量的计算和存储开销，限制了核方法在大型数据集问题上的应用。Nystrom逼近的谱预处理器通常更便宜和更容易计算和存储，并在实际应用中取得了成功。本文分析了使用这种逼近预处理器的权衡。具体来说，我们表明与数据集大小相关的对数样本数量能够让基于Nystrom逼近的预处理器几乎与梯度下降同样有效地加速。

    Kernel methods are a popular class of nonlinear predictive models in machine learning. Scalable algorithms for learning kernel models need to be iterative in nature, but convergence can be slow due to poor conditioning. Spectral preconditioning is an important tool to speed-up the convergence of such iterative algorithms for training kernel models. However computing and storing a spectral preconditioner can be expensive which can lead to large computational and storage overheads, precluding the application of kernel methods to problems with large datasets. A Nystrom approximation of the spectral preconditioner is often cheaper to compute and store, and has demonstrated success in practical applications. In this paper we analyze the trade-offs of using such an approximated preconditioner. Specifically, we show that a sample of logarithmic size (as a function of the size of the dataset) enables the Nystrom-based approximated preconditioner to accelerate gradient descent nearly as well as
    
[^30]: 注释敏感性：训练数据收集方法影响模型性能

    Annotation Sensitivity: Training Data Collection Methods Affect Model Performance. (arXiv:2311.14212v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2311.14212](http://arxiv.org/abs/2311.14212)

    该研究发现训练数据收集方法对注释本身和下游模型性能产生影响。在对仇恨言论和冒犯性语言进行注释收集的实验中，发现注释工具的设计选择会对模型的性能产生明显差异。

    

    当训练数据由人工注释者收集时，注释工具的设计、给予注释者的指示、注释者的特征以及他们之间的互动都可能对训练数据产生影响。这项研究证明了创建注释工具时的设计选择也会影响基于得到的注释训练的模型。我们引入了"注释敏感性"这个术语，用来指代注释数据收集方法对注释本身以及下游模型性能和预测的影响。我们在五种实验条件下对仇恨言论和冒犯性语言进行注释收集，随机将注释者分配到不同条件下。然后，在每个得到的五个数据集上对BERT模型进行微调，并在每个条件的保留部分上评估模型性能。我们发现在以下方面条件之间存在明显差异：1）仇恨言论/冒犯性语言注释的比例，2）模型性能。

    When training data are collected from human annotators, the design of the annotation instrument, the instructions given to annotators, the characteristics of the annotators, and their interactions can impact training data. This study demonstrates that design choices made when creating an annotation instrument also impact the models trained on the resulting annotations. We introduce the term annotation sensitivity to refer to the impact of annotation data collection methods on the annotations themselves and on downstream model performance and predictions. We collect annotations of hate speech and offensive language in five experimental conditions of an annotation instrument, randomly assigning annotators to conditions. We then fine-tune BERT models on each of the five resulting datasets and evaluate model performance on a holdout portion of each condition. We find considerable differences between the conditions for 1) the share of hate speech/offensive language annotations, 2) model per
    
[^31]: 关于分布鲁棒强化学习的基础

    On the Foundation of Distributionally Robust Reinforcement Learning. (arXiv:2311.09018v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.09018](http://arxiv.org/abs/2311.09018)

    该论文为分布鲁棒强化学习的理论基础做出了贡献，通过一个综合的建模框架，决策者在最坏情况下的分布转变下选择最优策略，并考虑了各种建模属性和对手引起的转变的灵活性。

    

    出于对在训练和部署之间环境变化时鲁棒策略的需求，我们为分布鲁棒强化学习的理论基础做出了贡献。通过一个以分布鲁棒马尔科夫决策过程（DRMDPs）为中心的综合建模框架，我们使决策者在一个由对手操纵的最坏情况分布转变下选择最优策略。通过统一和扩展现有的表述，我们严格构建了适用于决策者和对手的各种建模属性的DRMDPs，包括适应性粒度、探索历史依赖性、马尔科夫和马尔科夫时间齐次的决策者和对手动态。此外，我们深入研究了对手引起的转变的灵活性，研究了SA和S-矩形性。在这个DRMDP框架下，我们研究了实现鲁棒性所需的条件。

    Motivated by the need for a robust policy in the face of environment shifts between training and the deployment, we contribute to the theoretical foundation of distributionally robust reinforcement learning (DRRL). This is accomplished through a comprehensive modeling framework centered around distributionally robust Markov decision processes (DRMDPs). This framework obliges the decision maker to choose an optimal policy under the worst-case distributional shift orchestrated by an adversary. By unifying and extending existing formulations, we rigorously construct DRMDPs that embraces various modeling attributes for both the decision maker and the adversary. These attributes include adaptability granularity, exploring history-dependent, Markov, and Markov time-homogeneous decision maker and adversary dynamics. Additionally, we delve into the flexibility of shifts induced by the adversary, examining SA and S-rectangularity. Within this DRMDP framework, we investigate conditions for the e
    
[^32]: 使用类似ResNet的神经网络架构近似Langevin Monte Carlo

    Approximating Langevin Monte Carlo with ResNet-like Neural Network architectures. (arXiv:2311.03242v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.03242](http://arxiv.org/abs/2311.03242)

    本论文提出了一种使用类似ResNet的神经网络架构来近似Langevin Monte Carlo算法，通过将来自简单参考分布的样本映射到目标分布的样本中来进行采样，具有较好的逼近速度和表达性。

    

    我们通过构建一个神经网络，将来自简单参考分布（如标准正态分布）的样本映射到目标分布的样本中，从而从给定的目标分布中进行采样。为此，我们提出使用受Langevin Monte Carlo (LMC)算法启发的神经网络架构。基于LMC扰动结果，在Wasserstein-2距离上，我们展示了该架构对于平滑的对数凹目标分布的逼近速度。分析严重依赖于扰动LMC过程的中间度量的亚高斯性概念。特别地，我们根据不同扰动假设推导出了中间方差代理的增长界限。此外，我们提出了一种类似于深度残差神经网络的架构，并推导出了近似样本与目标分布映射的表达性结果。

    We sample from a given target distribution by constructing a neural network which maps samples from a simple reference, e.g. the standard normal distribution, to samples from the target. To that end, we propose using a neural network architecture inspired by the Langevin Monte Carlo (LMC) algorithm. Based on LMC perturbation results, we show approximation rates of the proposed architecture for smooth, log-concave target distributions measured in the Wasserstein-$2$ distance. The analysis heavily relies on the notion of sub-Gaussianity of the intermediate measures of the perturbed LMC process. In particular, we derive bounds on the growth of the intermediate variance proxies under different assumptions on the perturbations. Moreover, we propose an architecture similar to deep residual neural networks and derive expressivity results for approximating the sample to target distribution map.
    
[^33]: 具有加性和乘性噪声的线性随机微分方程的发生器识别

    Generator Identification for Linear SDEs with Additive and Multiplicative Noise. (arXiv:2310.19491v1 [math.ST])

    [http://arxiv.org/abs/2310.19491](http://arxiv.org/abs/2310.19491)

    本文介绍了从具有给定初始状态的解过程的分布中识别线性随机微分方程（SDE）的发生器的条件，并且提供了对于具有加性和乘性噪声的SDE的识别条件。

    

    本文提出了一种从具有给定固定初始状态的解过程的分布中识别线性随机微分方程（SDE）的发生器的条件。这些可识别性条件在使用线性SDE进行因果推断时至关重要，因为它们使得可以从其观测分布中识别出干预后的分布。我们具体推导出了识别具有加性噪声的线性SDE的发生器的充分必要条件，以及识别具有乘性噪声的线性SDE的发生器的充分条件。我们证明了对于两种类型的SDE，得到的条件是一般性的。此外，我们提供了对得到的可识别性条件的几何解释，以增强对其的理解。为了验证我们的理论结果，我们进行了一系列的模拟实验，这些实验支持并证实了我们所得到的结果。

    In this paper, we present conditions for identifying the generator of a linear stochastic differential equation (SDE) from the distribution of its solution process with a given fixed initial state. These identifiability conditions are crucial in causal inference using linear SDEs as they enable the identification of the post-intervention distributions from its observational distribution. Specifically, we derive a sufficient and necessary condition for identifying the generator of linear SDEs with additive noise, as well as a sufficient condition for identifying the generator of linear SDEs with multiplicative noise. We show that the conditions derived for both types of SDEs are generic. Moreover, we offer geometric interpretations of the derived identifiability conditions to enhance their understanding. To validate our theoretical results, we perform a series of simulations, which support and substantiate the established findings.
    
[^34]: 学习处理具有一般库存到货动态的库存控制策略

    Learning an Inventory Control Policy with General Inventory Arrival Dynamics. (arXiv:2310.17168v1 [cs.LG])

    [http://arxiv.org/abs/2310.17168](http://arxiv.org/abs/2310.17168)

    本文解决了学习具有一般库存到货动态下的库存控制策略的问题，同时允许修改订购数量以满足供应商的限制，并将周期性审核库存控制问题定义为外部决策过程。

    

    本文在面对一般到货动态的情况下解决了学习和回测库存控制策略的问题，我们将其称为数量随时间到货模型（QOT）。在实际供应链中，我们还允许修改订购数量以满足供应商的限制，例如订购最低数量和批次大小约束。据我们所知，这是第一篇处理任意到货动态或任意后续处理的订购数量的研究。在最近的工作（Madeka等，2022）的基础上，我们同样将周期性审核库存控制问题定义为外部决策过程，其中大部分状态不受代理的控制。Madeka等人（2022）展示了如何构建一个模拟器来回放历史数据以解决这类问题。在我们的例子中，我们将一个深度生成模型纳入到货过程的历史回放中。

    In this paper we address the problem of learning and backtesting inventory control policies in the presence of general arrival dynamics -- which we term as a quantity-over-time arrivals model (QOT). We also allow for order quantities to be modified as a post-processing step to meet vendor constraints such as order minimum and batch size constraints -- a common practice in real supply chains. To the best of our knowledge this is the first work to handle either arbitrary arrival dynamics or an arbitrary downstream post-processing of order quantities. Building upon recent work (Madeka et al., 2022) we similarly formulate the periodic review inventory control problem as an exogenous decision process, where most of the state is outside the control of the agent. Madeka et al. (2022) show how to construct a simulator that replays historic data to solve this class of problem. In our case, we incorporate a deep generative model for the arrivals process as part of the history replay. By formulat
    
[^35]: 学习具有已知骨架的有界度多树

    Learning bounded-degree polytrees with known skeleton. (arXiv:2310.06333v1 [cs.LG])

    [http://arxiv.org/abs/2310.06333](http://arxiv.org/abs/2310.06333)

    本论文提出了一种高效学习已知骨架的有界度多树的算法，并给出了在多项式时间和样本复杂度内的有限样本保证。这对于复杂概率分布的学习具有重要意义。

    

    我们为高维概率分布的一类丰富的多树（polytrees）——有界度多树，建立了高效适当学习的有限样本保证。有界度多树是贝叶斯网络的子类，贝叶斯网络是一种广泛研究的图模型类型。最近，Bhattacharyya等人（2021）通过提供一种高效算法，在已知无向图（骨架）的情况下，为1-多树恢复了有限样本保证。我们通过扩展他们的结果，提供了一种高效算法，可以在多项式时间和样本复杂度内学习任何有界度的$d$-多树。我们将算法与信息论样本复杂度的下界结合起来，表明对维度和目标精度参数的依赖几乎是紧致的。

    We establish finite-sample guarantees for efficient proper learning of bounded-degree polytrees, a rich class of high-dimensional probability distributions and a subclass of Bayesian networks, a widely-studied type of graphical model. Recently, Bhattacharyya et al. (2021) obtained finite-sample guarantees for recovering tree-structured Bayesian networks, i.e., 1-polytrees. We extend their results by providing an efficient algorithm which learns $d$-polytrees in polynomial time and sample complexity for any bounded $d$ when the underlying undirected graph (skeleton) is known. We complement our algorithm with an information-theoretic sample complexity lower bound, showing that the dependence on the dimension and target accuracy parameters are nearly tight.
    
[^36]: 一种用于非层次化多保真度自适应采样的潜变量方法

    A Latent Variable Approach for Non-Hierarchical Multi-Fidelity Adaptive Sampling. (arXiv:2310.03298v1 [stat.ML])

    [http://arxiv.org/abs/2310.03298](http://arxiv.org/abs/2310.03298)

    提出了一种基于潜变量的方法，用于非层次化多保真度自适应采样。该方法能够利用不同保真度模型之间的相关性以更高效地探索和利用设计空间。

    

    多保真度（MF）方法在提高替代模型和设计优化方面越来越受欢迎，通过整合来自不同低保真度（LF）模型的数据。尽管大多数现有的MF方法假定了一个固定的数据集，但是动态分配资源在不同保真度模型之间可以实现更高的探索和利用设计空间的效率。然而，大多数现有的MF方法依赖于保真度级别的层次假设，或者无法捕捉多个保真度级别之间的相互关系并利用其来量化未来样本的价值和导航自适应采样。为了解决这个障碍，我们提出了一个基于不同保真度模型的潜变量嵌入和相关的先验-后验分析的框架，以显式地利用它们的相关性进行自适应采样。在这个框架中，每个填充采样迭代包括两个步骤：首先我们确定具有最大潜力影响的位置。

    Multi-fidelity (MF) methods are gaining popularity for enhancing surrogate modeling and design optimization by incorporating data from various low-fidelity (LF) models. While most existing MF methods assume a fixed dataset, adaptive sampling methods that dynamically allocate resources among fidelity models can achieve higher efficiency in the exploring and exploiting the design space. However, most existing MF methods rely on the hierarchical assumption of fidelity levels or fail to capture the intercorrelation between multiple fidelity levels and utilize it to quantify the value of the future samples and navigate the adaptive sampling. To address this hurdle, we propose a framework hinged on a latent embedding for different fidelity models and the associated pre-posterior analysis to explicitly utilize their correlation for adaptive sampling. In this framework, each infill sampling iteration includes two steps: We first identify the location of interest with the greatest potential imp
    
[^37]: 关于随机梯度下降的不同模式

    On the different regimes of Stochastic Gradient Descent. (arXiv:2309.10688v1 [cs.LG])

    [http://arxiv.org/abs/2309.10688](http://arxiv.org/abs/2309.10688)

    这项研究解决了对于随机梯度下降（SGD）中不同模式的追踪和理解的问题，提供了一个相位图来区分噪声主导的SGD和大步骤主导的SGD。

    

    现代深度网络通过随机梯度下降（SGD）进行训练，其关键参数是每个步骤考虑的数据量或批量大小B以及步长或学习率η。对于小的B和大的η，SGD对应于参数的随机演化，其噪声幅度由“温度”T=η/B控制。然而当批量大小B≥B^*足够大时，这种描述被观察到失效，或者在温度足够小时简化为梯度下降（GD）。理解这些交叉发生的位置仍然是一个中心挑战。在这里，我们解决了这些问题，在一个教师-学生感知器分类模型中，我们展示了我们的关键预测仍适用于深度网络。具体来说，我们在B-η平面上获得了一个相位图，将三个动态阶段分开：（i）受温度控制的噪声主导的SGD，（ii）大步骤主导的SGD和

    Modern deep networks are trained with stochastic gradient descent (SGD) whose key parameters are the number of data considered at each step or batch size $B$, and the step size or learning rate $\eta$. For small $B$ and large $\eta$, SGD corresponds to a stochastic evolution of the parameters, whose noise amplitude is governed by the `temperature' $T\equiv \eta/B$. Yet this description is observed to break down for sufficiently large batches $B\geq B^*$, or simplifies to gradient descent (GD) when the temperature is sufficiently small. Understanding where these cross-overs take place remains a central challenge. Here we resolve these questions for a teacher-student perceptron classification model, and show empirically that our key predictions still apply to deep networks. Specifically, we obtain a phase diagram in the $B$-$\eta$ plane that separates three dynamical phases: $\textit{(i)}$ a noise-dominated SGD governed by temperature, $\textit{(ii)}$ a large-first-step-dominated SGD and
    
[^38]: 去殖民化的人工智能对齐：威色达尔玛、论证和艺术表达

    Decolonial AI Alignment: Vi\'{s}esadharma, Argument, and Artistic Expression. (arXiv:2309.05030v1 [cs.CY])

    [http://arxiv.org/abs/2309.05030](http://arxiv.org/abs/2309.05030)

    本文提出了去殖民化人工智能对齐的三个建议：改变基本道德哲学为达尔玛哲学，允许多元主义的论证传统存在于对齐技术中，以及将价值认识论扩展到超越自然语言中的指令。

    

    先前的研究已经阐明了人工智能（AI）开发和部署的殖民性。然而，这些研究很少涉及到对齐：即基于细致的人类反馈，调整大型语言模型（LLM）的行为与期望值一致。除了其他实践，殖民主义还有一部分是改变被殖民民族的信仰和价值观的历史；而当前的LLM对齐实践正是这一历史的复制。我们建议通过三个提议对AI对齐进行去殖民化：（a）将基本道德哲学从西方哲学转变为达尔玛哲学，（b）在对齐技术中允许论证和多元主义的传统，以及（c）将价值的认识论扩展到超越自然语言中的指令或命令。

    Prior work has explicated the coloniality of artificial intelligence (AI) development and deployment. One process that that work has not engaged with much is alignment: the tuning of large language model (LLM) behavior to be in line with desired values based on fine-grained human feedback. In addition to other practices, colonialism has a history of altering the beliefs and values of colonized peoples; this history is recapitulated in current LLM alignment practices. We suggest that AI alignment be decolonialized using three proposals: (a) changing the base moral philosophy from Western philosophy to dharma, (b) permitting traditions of argument and pluralism in alignment technologies, and (c) expanding the epistemology of values beyond instructions or commandments given in natural language.
    
[^39]: 使用合规的蒙特卡洛预测实现鲁棒的不确定性量化

    Robust Uncertainty Quantification using Conformalised Monte Carlo Prediction. (arXiv:2308.09647v1 [cs.LG])

    [http://arxiv.org/abs/2308.09647](http://arxiv.org/abs/2308.09647)

    这篇论文介绍了一种名为MC-CP的新型混合不确定性量化方法，通过将自适应蒙特卡洛dropout方法与合规预测相结合，实现了节省资源和产生鲁棒预测集/区间的目标。实验证明MC-CP在分类任务中相比其他先进方法具有显著提升

    

    在安全关键应用中部署深度学习模型仍然是一项非常具有挑战性的任务，需要对这些模型的可靠运行提供保证。不确定性量化（UQ）方法估计每个预测的模型置信度，通过考虑随机性和模型错误规范化的影响来指导决策。尽管最先进的UQ方法取得了一些进展，但它们在计算上要么非常昂贵，要么产生保守的预测集/区间。我们引入了一种新的混合UQ方法MC-CP，它将一种新的自适应蒙特卡洛（MC）dropout方法与合规预测（CP）相结合。MC-CP在运行时自适应调节传统的MC dropout以节省内存和计算资源，使得预测可以被CP使用，得到鲁棒的预测集/区间。通过全面的实验，我们展示了MC-CP相比MC dropout、RAPS和CQR等先进的UQ方法能够显著改善分类性能

    Deploying deep learning models in safety-critical applications remains a very challenging task, mandating the provision of assurances for the dependable operation of these models. Uncertainty quantification (UQ) methods estimate the model's confidence per prediction, informing decision-making by considering the effect of randomness and model misspecification. Despite the advances of state-of-the-art UQ methods, they are computationally expensive or produce conservative prediction sets/intervals. We introduce MC-CP, a novel hybrid UQ method that combines a new adaptive Monte Carlo (MC) dropout method with conformal prediction (CP). MC-CP adaptively modulates the traditional MC dropout at runtime to save memory and computation resources, enabling predictions to be consumed by CP, yielding robust prediction sets/intervals. Throughout comprehensive experiments, we show that MC-CP delivers significant improvements over advanced UQ methods, like MC dropout, RAPS and CQR, both in classificati
    
[^40]: 多类在线学习在Bandit反馈下的研究

    Multiclass Online Learnability under Bandit Feedback. (arXiv:2308.04620v1 [cs.LG])

    [http://arxiv.org/abs/2308.04620](http://arxiv.org/abs/2308.04620)

    Bandit反馈下的在线多类学习的关键在于Bandit Littlestone维度的有限性，无论标签空间是否无界。

    

    我们研究了在Bandit反馈下的多类在线分类问题。我们扩展了(daniely2013price)的结果，通过展示Bandit Littlestone维度的有限性是多类在线学习的必要且充分条件，即使标签空间是无界的。我们的结果补充了(hanneke2023multiclass)的最近工作，他们在标签空间无界的全信息设置中，展示了Littlestone维度刻画了在线多类学习的能力。

    We study online multiclass classification under bandit feedback. We extend the results of (daniely2013price) by showing that the finiteness of the Bandit Littlestone dimension is necessary and sufficient for bandit online multiclass learnability even when the label space is unbounded. Our result complements the recent work by (hanneke2023multiclass) who show that the Littlestone dimension characterizes online multiclass learnability in the full-information setting when the label space is unbounded.
    
[^41]: 对数贝叶斯遗憾边界

    Logarithmic Bayes Regret Bounds. (arXiv:2306.09136v1 [cs.LG])

    [http://arxiv.org/abs/2306.09136](http://arxiv.org/abs/2306.09136)

    该论文提出了对于贝叶斯赌博机的首个有限时间对数遗憾边界，并用于高斯和线性赌博机，从而阐明了贝叶斯设置中先验价值以及对$\tilde{O}(\sqrt{n})$界限的改善。

    

    我们为贝叶斯赌博机导出了首个有限时间对数遗憾边界。对于高斯赌博机，我们获得了一个$O(c_h \log^2 n)$的边界，其中$c_h$是与先验相关的常量。这与Lai（1987）的渐近下限相匹配。我们的证明与先前的工作有所不同，且简单且普遍。为了显示一般性，我们将我们的技术应用于线性赌博机。我们的界限阐明了贝叶斯设置中先验的价值，既可以作为目标，也可以作为传递给学习者的附加信息。它们显着改善了现有的$\tilde{O}(\sqrt{n})$界限，尽管存在下限，但已成为文献中的标准。

    We derive the first finite-time logarithmic regret bounds for Bayesian bandits. For Gaussian bandits, we obtain a $O(c_h \log^2 n)$ bound, where $c_h$ is a prior-dependent constant. This matches the asymptotic lower bound of Lai (1987). Our proofs mark a technical departure from prior works, and are simple and general. To show generality, we apply our technique to linear bandits. Our bounds shed light on the value of the prior in the Bayesian setting, both in the objective and as a side information given to the learner. They significantly improve the $\tilde{O}(\sqrt{n})$ bounds, that despite the existing lower bounds, have become standard in the literature.
    
[^42]: 线性条件VAE和分层VAE中的后验崩溃现象

    Posterior Collapse in Linear Conditional and Hierarchical Variational Autoencoders. (arXiv:2306.05023v1 [stat.ML])

    [http://arxiv.org/abs/2306.05023](http://arxiv.org/abs/2306.05023)

    本文研究了高度相似的变分后验分布和先验分布之间的后验崩溃现象，特别地，通过对线性条件VAE和分层VAE进行分析，证明了这种现象是由于潜在变量层次关系不清晰而引起的。

    

    在变分自编码器（VAE）中，后验崩溃现象指的是变分后验分布与先验分布的相似度过高，导致编码器提取的潜在变量保存的输入数据信息较少，无法为解码器的数据重建过程产生有意义的表示。尽管该现象一直是VAEs性能的研究热点，但是对于后验崩溃的理论却相对薄弱，特别是在非标准的VAEs中。本文通过对两类重要而常见又较少研究的VAEs进行非平凡的理论分析，即具有两个潜在变量层次的线性条件VAE和分层VAE，提升了对后验崩溃的理论认识，证明了其成因。

    The posterior collapse phenomenon in variational autoencoders (VAEs), where the variational posterior distribution closely matches the prior distribution, can hinder the quality of the learned latent variables. As a consequence of posterior collapse, the latent variables extracted by the encoder in VAEs preserve less information from the input data and thus fail to produce meaningful representations as input to the reconstruction process in the decoder. While this phenomenon has been an actively addressed topic related to VAEs performance, the theory for posterior collapse remains underdeveloped, especially beyond the standard VAEs. In this work, we advance the theoretical understanding of posterior collapse to two important and prevalent yet less studied classes of VAEs: conditional VAEs and hierarchical VAEs. Specifically, via a non-trivial theoretical analysis of linear conditional VAEs and hierarchical VAEs with two levels of latent, we prove that the cause of posterior collapses i
    
[^43]: 基于数据驱动的遗憾平衡在线模型选择的研究

    Data-Driven Regret Balancing for Online Model Selection in Bandits. (arXiv:2306.02869v1 [cs.LG])

    [http://arxiv.org/abs/2306.02869](http://arxiv.org/abs/2306.02869)

    论文讨论在具有赌博反馈的随机环境中进行选择，提出了两种基于数据的模型选择算法，并证明了其保证。通过利用实际遗憾，这些算法在实际中取得了好效果。

    

    我们考虑在具有赌博反馈的随机环境中进行顺序决策模型选择，其中元学习器可以使用一组基本学习器，并根据每个基本学习器推荐的策略动态决策。我们通过遗憾平衡来执行模型选择，但与此相关的最近文献不同的是，我们没有假设任何关于基本学习器的先验知识，如候选遗憾保证；相反，我们以数据驱动的方式揭示这些数量。因此，元学习器能够利用每个基本学习器在给定的学习环境中产生的实际遗憾（而不是期望遗憾），并挑选出最佳的遗憾。我们设计了两个模型选择算法，操作更为雄心勃勃的遗憾概念，并且除了通过遗憾平衡证明模型选择保证外，我们还在实验中展示了处理实际遗憾的令人信服的实际优势。

    We consider model selection for sequential decision making in stochastic environments with bandit feedback, where a meta-learner has at its disposal a pool of base learners, and decides on the fly which action to take based on the policies recommended by each base learner. Model selection is performed by regret balancing but, unlike the recent literature on this subject, we do not assume any prior knowledge about the base learners like candidate regret guarantees; instead, we uncover these quantities in a data-driven manner. The meta-learner is therefore able to leverage the realized regret incurred by each base learner for the learning environment at hand (as opposed to the expected regret), and single out the best such regret. We design two model selection algorithms operating with this more ambitious notion of regret and, besides proving model selection guarantees via regret balancing, we experimentally demonstrate the compelling practical benefits of dealing with actual regrets ins
    
[^44]: 深度概率时间序列预测的更好Batch方法

    Better Batch for Deep Probabilistic Time Series Forecasting. (arXiv:2305.17028v1 [stat.ML])

    [http://arxiv.org/abs/2305.17028](http://arxiv.org/abs/2305.17028)

    该研究提出了一种新的训练方法，通过在 mini-batch 中显式地学习误差的序列相关性，来提高深度概率时间序列预测的准确性和不确定性量化。

    

    深度概率时间序列预测因其能够提供有价值的不确定性量化而受到广泛关注。然而，许多现有模型过于简单化问题，假设误差过程是与时间无关的，从而忽略了误差过程中的序列相关性。这可能会降低预测的准确性，使这些模型对决策性任务的有效性减弱。为了克服这一限制，我们提出了一种创新的训练方法，将误差自相关性纳入考虑，以增强概率预测的准确性。我们的方法涉及构造一个mini-batch，作为$D$个连续时间序列段进行模型训练，并显式地学习一个协方差矩阵，覆盖了相邻时间步之间的误差相关性。由此产生的协方差矩阵可用于提高预测准确性和增强不确定性的量化。

    Deep probabilistic time series forecasting has gained significant attention due to its ability to provide valuable uncertainty quantification for decision-making tasks. However, many existing models oversimplify the problem by assuming the error process is time-independent, thereby overlooking the serial correlation in the error process. This oversight can potentially diminish the accuracy of the forecasts, rendering these models less effective for decision-making purposes. To overcome this limitation, we propose an innovative training method that incorporates error autocorrelation to enhance the accuracy of probabilistic forecasting. Our method involves constructing a mini-batch as a collection of $D$ consecutive time series segments for model training and explicitly learning a covariance matrix over each mini-batch that encodes the error correlation among adjacent time steps. The resulting covariance matrix can be used to improve prediction accuracy and enhance uncertainty quantifica
    
[^45]: 深卷积神经网络中归纳偏置的理论分析

    Theoretical Analysis of Inductive Biases in Deep Convolutional Networks. (arXiv:2305.08404v1 [cs.LG])

    [http://arxiv.org/abs/2305.08404](http://arxiv.org/abs/2305.08404)

    本文研究深卷积神经网络（CNN）中的归纳偏置，证明了$\mathcal{O}(\log d)$的深度就足以实现普适性，用CNN学习稀疏函数只需要$\tilde{\mathcal{O}}(\log^2d)$个样本。同时，通过局部连接网络（LCN）分析了权重共享和局部性的归纳偏置的区别，得出了它们在表示需要有限平移等变和高方向选择性的函数方面的优越性。

    

    本文研究卷积神经网络（CNN）中的归纳偏置，这被认为是CNN在视觉任务上表现异常出色的重要驱动因素。我们首先分析了CNN的普适性，即逼近连续函数的能力。我们证明了$\mathcal{O}(\log d)$的深度就足以实现普适性，其中$d$是输入维度。这相比于现有结果需要$\Omega(d)$的深度是一项重大改进。我们还证明了用CNN学习稀疏函数只需要$\tilde{\mathcal{O}}(\log^2d)$个样本，表明深度CNN可以有效地捕捉长程稀疏相关性。我们的研究还分析了共享权重和局部性的归纳偏置，通过对称性得出结论。为了区分这两种偏见，我们引入了局部连接网络（LCN）并证明了它们在表示需要有限平移等变和高方向选择性的函数方面的优越性。我们的结果为深CNN的成功提供了理论洞察力，同时更好地理解了它们的局限性。

    In this paper, we study the inductive biases in convolutional neural networks (CNNs), which are believed to be vital drivers behind CNNs' exceptional performance on vision-like tasks. We first analyze the universality of CNNs, i.e., the ability to approximate continuous functions. We prove that a depth of $\mathcal{O}(\log d)$ is sufficient for achieving universality, where $d$ is the input dimension. This is a significant improvement over existing results that required a depth of $\Omega(d)$. We also prove that learning sparse functions with CNNs needs only $\tilde{\mathcal{O}}(\log^2d)$ samples, indicating that deep CNNs can efficiently capture long-range sparse correlations. Note that all these are achieved through a novel combination of increased network depth and the utilization of multichanneling and downsampling.  Lastly, we study the inductive biases of weight sharing and locality through the lens of symmetry. To separate two biases, we introduce locally-connected networks (LCN
    
[^46]: 通过子高斯内在矩范实现紧凑的非渐进推断

    Tight Non-asymptotic Inference via Sub-Gaussian Intrinsic Moment Norm. (arXiv:2303.07287v1 [stat.ML])

    [http://arxiv.org/abs/2303.07287](http://arxiv.org/abs/2303.07287)

    本文提出了一种通过最大化一系列归一化矩来使用子高斯内在矩范实现紧凑的非渐进推断的方法，该方法可以导致更紧的Hoeffding子高斯浓度不等式，并且可以通过子高斯图检查具有有限样本大小的子高斯数据。

    This paper proposes a method of achieving tight non-asymptotic inference by using sub-Gaussian intrinsic moment norm through maximizing a series of normalized moments, which can lead to tighter Hoeffding's sub-Gaussian concentration inequalities and can be checked with sub-Gaussian plot for sub-Gaussian data with a finite sample size.

    在非渐进统计推断中，子高斯分布的方差类型参数起着至关重要的作用。然而，基于经验矩生成函数（MGF）的直接估计这些参数是不可行的。为此，我们建议通过最大化一系列归一化矩来使用子高斯内在矩范[Buldygin和Kozachenko（2000），定理1.3]。重要的是，推荐的范数不仅可以恢复相应MGF的指数矩界限，而且还可以导致更紧的Hoeffding子高斯浓度不等式。在实践中，我们提出了一种直观的方法，通过子高斯图检查具有有限样本大小的子高斯数据。可以通过简单的插入方法鲁棒地估计内在矩范数。我们的理论结果应用于非渐进分析，包括多臂赌博机。

    In non-asymptotic statistical inferences, variance-type parameters of sub-Gaussian distributions play a crucial role. However, direct estimation of these parameters based on the empirical moment generating function (MGF) is infeasible. To this end, we recommend using a sub-Gaussian intrinsic moment norm [Buldygin and Kozachenko (2000), Theorem 1.3] through maximizing a series of normalized moments. Importantly, the recommended norm can not only recover the exponential moment bounds for the corresponding MGFs, but also lead to tighter Hoeffding's sub-Gaussian concentration inequalities. In practice, {\color{black} we propose an intuitive way of checking sub-Gaussian data with a finite sample size by the sub-Gaussian plot}. Intrinsic moment norm can be robustly estimated via a simple plug-in approach. Our theoretical results are applied to non-asymptotic analysis, including the multi-armed bandit.
    
[^47]: 正交多项式逼近算法（OPAA）：一种用于估计概率密度的功能分析方法

    Orthogonal Polynomials Approximation Algorithm (OPAA):a functional analytic approach to estimating probability densities. (arXiv:2211.08594v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.08594](http://arxiv.org/abs/2211.08594)

    OPAA是一种功能分析方法的算法，通过找到平滑的概率分布函数估计值、计算归一化权重的估计值，并使用特殊的函数空间转换来估计证据，实现了并行计算的一次通过。它适用于估计概率密度函数，尤其在贝叶斯问题中估计归一化权重。

    

    我们提出了一种新的正交多项式逼近算法（OPAA），这是一种可并行化的算法，使用功能分析方法估计概率分布：首先，它找到了概率分布的平滑函数估计，无论它是否归一化；其次，算法提供了归一化权重的估计；第三，算法提出了一种新的计算方案来计算这些估计值。OPAA的核心组成部分是将联合分布的平方根转化为我们构造的特殊函数空间的特殊变换。通过这个变换，证据等于转换函数的$L^2$范数的平方。因此，可以通过转换系数的平方和来估计证据。计算可以并行化并在一次通过中完成。OPAA可以广泛应用于概率密度函数的估计。在贝叶斯问题中，它可以用于估计归一化权重。

    We present the new Orthogonal Polynomials Approximation Algorithm (OPAA), a parallelizable algorithm that estimates probability distributions using functional analytic approach: first, it finds a smooth functional estimate of the probability distribution, whether it is normalized or not; second, the algorithm provides an estimate of the normalizing weight; and third, the algorithm proposes a new computation scheme to compute such estimates.  A core component of OPAA is a special transform of the square root of the joint distribution into a special functional space of our construct. Through this transform, the evidence is equated with the $L^2$ norm of the transformed function, squared. Hence, the evidence can be estimated by the sum of squares of the transform coefficients. Computations can be parallelized and completed in one pass.  OPAA can be applied broadly to the estimation of probability density functions. In Bayesian problems, it can be applied to estimating the normalizing weig
    
[^48]: 利用仿射模型变换的迁移学习

    Transfer learning with affine model transformation. (arXiv:2210.09745v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.09745](http://arxiv.org/abs/2210.09745)

    本文提出了一种叫做仿射模型转移的迁移学习方法，通过最小化期望平方损失，可以适应各种不同的方法，包括基于神经特征提取器的方法。对于这个方法也给出了理论上的解释。

    

    由于在数据稀缺的情况下能够提高机器学习的预测能力，受到了广泛关注。传统上，使用给定的源模型集和来自目标领域的数据集，通过统计学习来适应预训练模型到目标领域，学习领域转移和领域特定因素。虽然这种方法在广泛的实际应用中取得了巨大的成功，但缺乏理论基础阻碍了进一步的方法发展。本文提出了一种称为仿射模型转移的通用类别的迁移学习回归方法，遵循期望平方损失最小化的原则。结果表明，仿射模型转移广泛包括各种现有方法，包括基于神经特征提取器的最常见过程。此外，本文阐明了仿射模型转移的理论特性。

    Supervised transfer learning has received considerable attention due to its potential to boost the predictive power of machine learning in scenarios where data are scarce. Generally, a given set of source models and a dataset from a target domain are used to adapt the pre-trained models to a target domain by statistically learning domain shift and domain-specific factors. While such procedurally and intuitively plausible methods have achieved great success in a wide range of real-world applications, the lack of a theoretical basis hinders further methodological development. This paper presents a general class of transfer learning regression called affine model transfer, following the principle of expected-square loss minimization. It is shown that the affine model transfer broadly encompasses various existing methods, including the most common procedure based on neural feature extractors. Furthermore, the current paper clarifies theoretical properties of the affine model transfer such 
    
[^49]: 高维点云数据的多样性散射变换

    The Manifold Scattering Transform for High-Dimensional Point Cloud Data. (arXiv:2206.10078v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.10078](http://arxiv.org/abs/2206.10078)

    多样性散射变换是一种用于高维点云数据的深度特征提取器，在实现上采用了扩散映射理论，有效用于信号分类和流形分类任务。

    

    多样性散射变换是一种用于定义在黎曼流形上的数据的深度特征提取器。它是将类似于卷积神经网络的操作扩展到一般流形的最早的例子之一。这个模型的最初工作主要关注其理论稳定性和不变性特性，但除了在具有预定义网格的二维曲面的情况下提供其数值实现的方法外，还没有提供其数值实现的方法。在这项工作中，我们提出了一种基于扩散映射理论的实用方案，用于将多样性散射变换应用于自然系统中产生的数据集，例如单细胞遗传学，其中数据是作为低维流形上的高维点云建模的。我们展示了我们的方法对于信号分类和流形分类任务的有效性。

    The manifold scattering transform is a deep feature extractor for data defined on a Riemannian manifold. It is one of the first examples of extending convolutional neural network-like operators to general manifolds. The initial work on this model focused primarily on its theoretical stability and invariance properties but did not provide methods for its numerical implementation except in the case of two-dimensional surfaces with predefined meshes. In this work, we present practical schemes, based on the theory of diffusion maps, for implementing the manifold scattering transform to datasets arising in naturalistic systems, such as single cell genetics, where the data is a high-dimensional point cloud modeled as lying on a low-dimensional manifold. We show that our methods are effective for signal classification and manifold classification tasks.
    
[^50]: 面向尺度无关的深度操作器网络的泛化界限

    Towards Size-Independent Generalization Bounds for Deep Operator Nets. (arXiv:2205.11359v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2205.11359](http://arxiv.org/abs/2205.11359)

    本论文研究了深度操作器网络的泛化界限问题，在一类DeepONets中证明了它们的Rademacher复杂度的界限不会随网络宽度扩展而明确变化，并利用这个结果展示了如何选择Huber损失来获得不明确依赖于网络大小的泛化误差界限。

    

    在最近的时期，机器学习方法在分析物理系统方面取得了重要进展。在这个主题中特别活跃的领域是"物理信息机器学习"，它专注于使用神经网络来数值求解微分方程。在这项工作中，我们旨在推进在训练DeepONets时测量样本外误差的理论 - 这是解决PDE系统最通用的方法之一。首先，针对一类DeepONets，我们证明了它们的Rademacher复杂度有一个界限，该界限不会明确地随着涉及的网络宽度扩展。其次，我们利用这一结果来展示如何选择Huber损失，使得对于这些DeepONet类，能够获得不明确依赖于网络大小的泛化误差界限。我们指出，我们的理论结果适用于任何目标是由DeepONets求解的PDE。

    In recent times machine learning methods have made significant advances in becoming a useful tool for analyzing physical systems. A particularly active area in this theme has been "physics-informed machine learning" which focuses on using neural nets for numerically solving differential equations. In this work, we aim to advance the theory of measuring out-of-sample error while training DeepONets -- which is among the most versatile ways to solve PDE systems in one-shot.  Firstly, for a class of DeepONets, we prove a bound on their Rademacher complexity which does not explicitly scale with the width of the nets involved. Secondly, we use this to show how the Huber loss can be chosen so that for these DeepONet classes generalization error bounds can be obtained that have no explicit dependence on the size of the nets. We note that our theoretical results apply to any PDE being targeted to be solved by DeepONets.
    
[^51]: 在张量主成分分析和相关问题中的统计计算权衡：通过通信复杂度

    Statistical-Computational Trade-offs in Tensor PCA and Related Problems via Communication Complexity. (arXiv:2204.07526v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2204.07526](http://arxiv.org/abs/2204.07526)

    本文通过通信复杂度推导出了对于内存受限算法在张量主成分分析中的计算下界，并且指定了解决该问题的算法必须在数据样本经过次数、样本大小和所需内存之间进行权衡。这些下界暗示了许多常用算法在样本大小不够大时需要更多的迭代次数。

    

    张量主成分分析是Montanari和Richard引入的一种风格化统计推断问题，用于研究从高阶矩张量中估计未知参数的计算难度。与其矩阵对应问题不同，张量主成分分析展现了一种统计计算差距，即在样本大小范围内，问题在信息理论上可解，但被认为在计算上较困难。本文利用通信复杂度推导出了内存受限算法在张量主成分分析中的计算下界。这些下界指定了成功解决张量主成分分析的任何算法在数据样本经过次数、样本大小和所需内存之间的权衡。尽管下界不能排除多项式时间算法，但它们意味着许多常用的算法，如梯度下降和幂迭代方法，在样本大小不够大时必须有更高的迭代次数。类似的下界还可以使用通信复杂度获得。

    Tensor PCA is a stylized statistical inference problem introduced by Montanari and Richard to study the computational difficulty of estimating an unknown parameter from higher-order moment tensors. Unlike its matrix counterpart, Tensor PCA exhibits a statistical-computational gap, i.e., a sample size regime where the problem is information-theoretically solvable but conjectured to be computationally hard. This paper derives computational lower bounds on the run-time of memory bounded algorithms for Tensor PCA using communication complexity. These lower bounds specify a trade-off among the number of passes through the data sample, the sample size, and the memory required by any algorithm that successfully solves Tensor PCA. While the lower bounds do not rule out polynomial-time algorithms, they do imply that many commonly-used algorithms, such as gradient descent and power method, must have a higher iteration count when the sample size is not large enough. Similar lower bounds are obtai
    
[^52]: Concordance指数的分解: 对生存预测模型深入理解的度量方法

    The Concordance Index decomposition: A measure for a deeper understanding of survival prediction models. (arXiv:2203.00144v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2203.00144](http://arxiv.org/abs/2203.00144)

    本文提出了一种将Concordance指数分解成两个部分的方法，用于评估生存预测模型的性能。该分解方法可以进行更细粒度的分析，揭示不同预测方法之间的优劣。实验证明，深度学习模型更好地利用了观测事件。

    

    Concordance指数（C-index）是生存分析中常用的评估预测模型性能的指标。本文提出了一种将C-index分解为两个数量的加权调和平均的方法：一个用于比较观测事件与其他观测事件的排序，另一个用于比较观测事件与被剪辑的情况的排序。这种分解方法可以对不同生存预测方法之间的优劣进行更细粒度的分析。通过与经典模型和最先进方法的基准比较，以及本文提出的新的变分生成神经网络方法（SurVED），展示了该分解方法的实用性。使用四个公开可用的具有不同剪辑水平的数据集评估模型的性能。通过C-index分解和合成剪辑，分析结果显示深度学习模型更好地利用了观测事件。

    The Concordance Index (C-index) is a commonly used metric in Survival Analysis for evaluating the performance of a prediction model. In this paper, we propose a decomposition of the C-index into a weighted harmonic mean of two quantities: one for ranking observed events versus other observed events, and the other for ranking observed events versus censored cases. This decomposition enables a finer-grained analysis of the relative strengths and weaknesses between different survival prediction methods. The usefulness of this decomposition is demonstrated through benchmark comparisons against classical models and state-of-the-art methods, together with the new variational generative neural-network-based method (SurVED) proposed in this paper. The performance of the models is assessed using four publicly available datasets with varying levels of censoring. Using the C-index decomposition and synthetic censoring, the analysis shows that deep learning models utilize the observed events more 
    
[^53]: 高维推断与模拟马尔可夫随机场的FDR控制

    High-dimensional Inference and FDR Control for Simulated Markov Random Fields. (arXiv:2202.05612v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2202.05612](http://arxiv.org/abs/2202.05612)

    本文提出了一种在高维背景下进行模拟马尔可夫随机场统计推断的方法，实现了一致性，并构建了两种误发现率控制程序。

    

    在各种科学领域中，确定与响应变量相关的重要特征是一项基本任务。本文探讨高维背景下模拟马尔可夫随机场的统计推断。我们提出了一种基于马尔可夫链蒙特卡罗极大似然估计（MCMC-MLE）与弹性网正则化的方法。在MCMC方法的温和条件下，我们的罚款MCMC-MLE方法实现了$\ell_{1}$一致性。我们提出了一个去相关的得分检验，确定其渐近正态性和一步估计量的渐近正态性，以及相应的置信区间。此外，我们基于p值和e值的渐近行为构建了两种误发现率控制程序。全面的数值模拟验证了所提方法的理论有效性。

    Identifying important features linked to a response variable is a fundamental task in various scientific domains. This article explores statistical inference for simulated Markov random fields in high-dimensional settings. We introduce a methodology based on Markov Chain Monte Carlo Maximum Likelihood Estimation (MCMC-MLE) with Elastic-net regularization. Under mild conditions on the MCMC method, our penalized MCMC-MLE method achieves $\ell_{1}$-consistency. We propose a decorrelated score test, establishing both its asymptotic normality and that of a one-step estimator, along with the associated confidence interval. Furthermore, we construct two false discovery rate control procedures via the asymptotic behaviors for both p-values and e-values. Comprehensive numerical simulations confirm the theoretical validity of the proposed methods.
    
[^54]: Wavelet Networks: 从原始时间序列学习尺度平移等变性的学习网络

    Wavelet Networks: Scale-Translation Equivariant Learning From Raw Time-Series. (arXiv:2006.05259v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2006.05259](http://arxiv.org/abs/2006.05259)

    本文介绍了一种利用时间序列固有对称性构建的小波网络，其表现出嵌套的非线性小波样的时频变换，实验证明其在原始波形上优于传统的CNN。

    

    利用特定数据领域中固有的对称性构建等变性神经网络，可显著提高数据效率和泛化能力。然而，大多数现有研究集中在平面和体积数据中产生的对称性上，而把一个关键的数据源基本未开发：时间序列。本文通过利用时间序列的固有对称性来构建等变性神经网络，填补了这一空白。我们确认了两个核心对称性：尺度和平移，并构建了适用于时间序列学习的尺度平移等变性神经网络。有趣的是，我们发现尺度平移等变性映射与小波变换具有很强的相似性。受到这种相似性的启发，我们将我们的网络称为小波网络，并展示它们执行嵌套的非线性小波样的时频变换。实证结果表明，小波网络在原始波形上优于传统的CNN。

    Leveraging the symmetries inherent to specific data domains for the construction of equivariant neural networks has lead to remarkable improvements in terms of data efficiency and generalization. However, most existing research focuses on symmetries arising from planar and volumetric data, leaving a crucial data source largely underexplored: time-series. In this work, we fill this gap by leveraging the symmetries inherent to time-series for the construction of equivariant neural network. We identify two core symmetries: *scale and translation*, and construct scale-translation equivariant neural networks for time-series learning. Intriguingly, we find that scale-translation equivariant mappings share strong resemblance with the wavelet transform. Inspired by this resemblance, we term our networks Wavelet Networks, and show that they perform nested non-linear wavelet-like time-frequency transforms. Empirical results show that Wavelet Networks outperform conventional CNNs on raw waveforms
    
[^55]: 用于场景分析的同质 Ising 模型的快速近似计算方法

    Fast approximations in the homogeneous Ising model for use in scene analysis. (arXiv:1712.02195v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/1712.02195](http://arxiv.org/abs/1712.02195)

    本文提供了一种快速近似计算同质 Ising 模型中重要量的方法，该方法的表现在模拟研究中表现良好，可用于场景分析。

    

    Ising 模型在许多应用中都很重要，但其归一化常数、活动顶点数的平均值和自旋相互作用的均值难以计算。我们提供了准确的近似值，使得在同质情况下可以数值计算这些量。模拟研究表明，与马尔可夫链蒙特卡洛方法相比，我们的方法性能良好，且所需时间只是那些随机方法的一小部分。我们的近似值在执行功能磁共振激活检测实验的贝叶斯推断以及植物生产中年增量空间图案的各向异性的似然比检验中得到了体现。

    The Ising model is important in statistical modeling and inference in many applications, however its normalizing constant, mean number of active vertices and mean spin interaction are intractable to compute. We provide accurate approximations that make it possible to numerically calculate these quantities in the homogeneous case. Simulation studies indicate good performance when compared to Markov Chain Monte Carlo methods and at a tiny fraction of the time taken by those stochastic approaches. The value of our approximations is illustrated in performing Bayesian inference in a functional Magnetic Resonance Imaging activation detection experiment, and also in likelihood ratio testing for anisotropy in the spatial patterns of yearly increases in pistachio tree yields.
    

