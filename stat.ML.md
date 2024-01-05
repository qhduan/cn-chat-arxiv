# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simulation-Based Inference with Quantile Regression.](http://arxiv.org/abs/2401.02413) | 提出了一种基于模拟推断和分位数回归的新方法，通过学习个体化的分位数来估计后验样本，并使用局部累积密度函数定义贝叶斯可信区间，具有更快的评估速度。同时，还可以集成后处理扩展步骤以保证后验估计的无偏性，而计算成本几乎可以忽略不计。 |
| [^2] | [A Survey Analyzing Generalization in Deep Reinforcement Learning.](http://arxiv.org/abs/2401.02349) | 本文调查了深度强化学习中的泛化性能。深度强化学习策略存在过拟合问题，限制了它们的鲁棒性和泛化能力。研究形式化和统一了提高泛化性和克服过拟合的不同解决方案。 |
| [^3] | [A Robust Quantile Huber Loss With Interpretable Parameter Adjustment In Distributional Reinforcement Learning.](http://arxiv.org/abs/2401.02325) | 这篇论文提出了一种鲁棒的分位数Huber损失函数，在分布强化学习中通过捕捉噪声并调整参数来增强对异常值的鲁棒性。实证测试验证了该方法的有效性。 |
| [^4] | [Robust bilinear factor analysis based on the matrix-variate $t$ distribution.](http://arxiv.org/abs/2401.02203) | 本文提出了一种基于矩阵变量t分布的鲁棒双线性因子分析（tbfa）模型，可以从重尾或受污染的矩阵数据中同时提取行和列变量的公共因子。 |
| [^5] | [Energy based diffusion generator for efficient sampling of Boltzmann distributions.](http://arxiv.org/abs/2401.02080) | 介绍了一种称为基于能量的扩散生成器的新型采样器，用于从任意目标分布中生成样本，并通过扩散模型和广义哈密顿动力学提高采样性能。在各种复杂分布函数上的实证评估中表现出优越性。 |
| [^6] | [U-Trustworthy Models.Reliability, Competence, and Confidence in Decision-Making.](http://arxiv.org/abs/2401.02062) | 提出了一种基于哲学文献的新的信任框架$\mathcal{U}$-信任度，用于评估AI系统的信任性，该框架挑战了传统的概率框架，并提出了解决方案以克服偏见和误导性信任评估的风险。 |
| [^7] | [Neural Collapse for Cross-entropy Class-Imbalanced Learning with Unconstrained ReLU Feature Model.](http://arxiv.org/abs/2401.02058) | 本文研究了在交叉熵类别不平衡学习中的神经折叠现象，证明了在存在类别不平衡情况下，神经折叠仍然存在，但类均值的几何特性会发生偏移。 |
| [^8] | [Hierarchical Clustering in ${\Lambda}$CDM Cosmologies via Persistence Energy.](http://arxiv.org/abs/2401.01988) | 通过持续能量方法研究了${\Lambda}$CDM宇宙学中的层次聚类，发现持续能量与红移值之间存在相关性，揭示了宇宙结构的动力学特征。 |
| [^9] | [Beyond Regrets: Geometric Metrics for Bayesian Optimization.](http://arxiv.org/abs/2401.01981) | 本论文提出了四个新的几何度量，可以比较贝叶斯优化算法在考虑查询点和全局最优解的几何特性时的性能。 |
| [^10] | [Probabilistic Modeling for Sequences of Sets in Continuous-Time.](http://arxiv.org/abs/2312.15045) | 本文提出了一个通用的连续时间序列集合的概率建模框架，适用于处理每个事件与一组项目相关联的情况。引入了适用于任何强度为基础的递归神经点过程模型的推理方法，可用于回答关于序列历史条件下的概率查询问题。 |
| [^11] | [Entropy and the Kullback-Leibler Divergence for Bayesian Networks: Computational Complexity and Efficient Implementation.](http://arxiv.org/abs/2312.01520) | 本文提出了一种计算贝叶斯网络中Shannon熵和Kullback-Leibler散度的高效算法，并通过一系列数值示例进行了演示。此外，还展示了如何将高斯贝叶斯网络中KL的计算复杂度从立方降低到二次。 |
| [^12] | [On Model Compression for Neural Networks: Framework, Algorithm, and Convergence Guarantee.](http://arxiv.org/abs/2303.06815) | 本文提出了一个框架和算法，从非凸优化的角度来进行神经网络模型压缩。算法解决了梯度消失/爆炸问题，并保证了收敛性。 |
| [^13] | [Controlling Moments with Kernel Stein Discrepancies.](http://arxiv.org/abs/2211.05408) | 本研究分析了核斯坦离差（KSD）控制性质，发现标准KSD无法控制矩的收敛，提出了可控制矩和弱收敛的下游扩散KSD，并且发展了可以准确描述$q$-Wasserstein收敛的KSD。 |
| [^14] | [Sliced gradient-enhanced Kriging for high-dimensional function approximation.](http://arxiv.org/abs/2204.03562) | 本文提出了一种名为切片GE-Kriging（SGE-Kriging）的新方法，用于解决高维函数逼近中的相关矩阵和超参数调整问题。 |
| [^15] | [Federated Optimization of Smooth Loss Functions.](http://arxiv.org/abs/2201.01954) | 本文研究了在联邦学习框架下的平均分布优化平滑损失函数的问题，提出了联邦低秩梯度下降（FedLRGD）算法来利用数据的平滑性，从而实现更好的优化结果。 |
| [^16] | [Fast approximations in the homogeneous Ising model for use in scene analysis.](http://arxiv.org/abs/1712.02195) | 本文提供了一种快速近似计算同质 Ising 模型中重要量的方法，该方法的表现在模拟研究中表现良好，可用于场景分析。 |

# 详细

[^1]: 基于分位数回归的模拟推断方法

    Simulation-Based Inference with Quantile Regression. (arXiv:2401.02413v1 [stat.ML])

    [http://arxiv.org/abs/2401.02413](http://arxiv.org/abs/2401.02413)

    提出了一种基于模拟推断和分位数回归的新方法，通过学习个体化的分位数来估计后验样本，并使用局部累积密度函数定义贝叶斯可信区间，具有更快的评估速度。同时，还可以集成后处理扩展步骤以保证后验估计的无偏性，而计算成本几乎可以忽略不计。

    

    我们提出了一种基于条件分位数回归的新型模拟推断（Simulation-Based Inference，SBI）方法——神经分位数估计（Neural Quantile Estimation，NQE）。NQE通过自回归方式学习每个后验维度的单一维度分位数，以数据和之前的后验维度为条件。后验样本通过使用单调三次埃尔米特样条插值预测分位数进行获取，并对尾部行为和多模态分布进行了特殊处理。我们引入了一种使用局部累积密度函数（CDF）的贝叶斯可信区间的替代定义，其评估速度比传统的最高后验密度区域（HPDR）快得多。在模拟预算有限和/或已知模型错误规范的情况下，可以将后处理扩展步骤集成到NQE中，以确保后验估计的无偏性，且附加的计算成本可以忽略不计。我们证明了所提出的NQE方法达到了最新的研究水平。

    We present Neural Quantile Estimation (NQE), a novel Simulation-Based Inference (SBI) method based on conditional quantile regression. NQE autoregressively learns individual one dimensional quantiles for each posterior dimension, conditioned on the data and previous posterior dimensions. Posterior samples are obtained by interpolating the predicted quantiles using monotonic cubic Hermite spline, with specific treatment for the tail behavior and multi-modal distributions. We introduce an alternative definition for the Bayesian credible region using the local Cumulative Density Function (CDF), offering substantially faster evaluation than the traditional Highest Posterior Density Region (HPDR). In case of limited simulation budget and/or known model misspecification, a post-processing broadening step can be integrated into NQE to ensure the unbiasedness of the posterior estimation with negligible additional computational cost. We demonstrate that the proposed NQE method achieves state-of
    
[^2]: 分析深度强化学习中泛化性能的调查

    A Survey Analyzing Generalization in Deep Reinforcement Learning. (arXiv:2401.02349v1 [cs.LG])

    [http://arxiv.org/abs/2401.02349](http://arxiv.org/abs/2401.02349)

    本文调查了深度强化学习中的泛化性能。深度强化学习策略存在过拟合问题，限制了它们的鲁棒性和泛化能力。研究形式化和统一了提高泛化性和克服过拟合的不同解决方案。

    

    利用深度神经网络解决高维状态或动作空间中的问题，强化学习研究在实践中取得了重要的成功和关注。尽管深度强化学习策略目前在许多领域中正在被应用，从医疗应用到自动驾驶车辆，但关于深度强化学习策略的泛化能力仍有许多待解答的问题。在本文中，我们将概述深度强化学习策略遇到过拟合问题的根本原因，限制了它们的鲁棒性和泛化能力。此外，我们将对提高泛化性和克服状态-动作值函数中的过拟合的不同解决方案进行形式化和统一。我们相信我们的研究可以为当前深度强化学习的进展提供一个简洁系统的统一分析，并有助于构建健壮的深度神经网络策略。

    Reinforcement learning research obtained significant success and attention with the utilization of deep neural networks to solve problems in high dimensional state or action spaces. While deep reinforcement learning policies are currently being deployed in many different fields from medical applications to self driving vehicles, there are still ongoing questions the field is trying to answer on the generalization capabilities of deep reinforcement learning policies. In this paper, we will outline the fundamental reasons why deep reinforcement learning policies encounter overfitting problems that limit their robustness and generalization capabilities. Furthermore, we will formalize and unify the diverse solution approaches to increase generalization, and overcome overfitting in state-action value functions. We believe our study can provide a compact systematic unified analysis for the current advancements in deep reinforcement learning, and help to construct robust deep neural policies 
    
[^3]: 分布强化学习中具有可解释参数调整的鲁棒分位数Huber损失

    A Robust Quantile Huber Loss With Interpretable Parameter Adjustment In Distributional Reinforcement Learning. (arXiv:2401.02325v1 [cs.LG])

    [http://arxiv.org/abs/2401.02325](http://arxiv.org/abs/2401.02325)

    这篇论文提出了一种鲁棒的分位数Huber损失函数，在分布强化学习中通过捕捉噪声并调整参数来增强对异常值的鲁棒性。实证测试验证了该方法的有效性。

    

    分布强化学习通过最小化分位数Huber损失函数来估计回报分布，该函数从高斯分布之间的Wasserstein距离计算中产生，捕捉到当前和目标分位数值中的噪声。与经典的分位数Huber损失相比，这种创新的损失函数增强了对异常值的鲁棒性，并且通过近似数据中噪声的数量来调整参数。实证测试在分布强化学习的常见应用Atari游戏和最近的对冲策略中验证了该方法的有效性。

    Distributional Reinforcement Learning (RL) estimates return distribution mainly by learning quantile values via minimizing the quantile Huber loss function, entailing a threshold parameter often selected heuristically or via hyperparameter search, which may not generalize well and can be suboptimal. This paper introduces a generalized quantile Huber loss function derived from Wasserstein distance (WD) calculation between Gaussian distributions, capturing noise in predicted (current) and target (Bellman-updated) quantile values. Compared to the classical quantile Huber loss, this innovative loss function enhances robustness against outliers. Notably, the classical Huber loss function can be seen as an approximation of our proposed loss, enabling parameter adjustment by approximating the amount of noise in the data during the learning process. Empirical tests on Atari games, a common application in distributional RL, and a recent hedging strategy using distributional RL, validate the eff
    
[^4]: 基于矩阵变量t分布的鲁棒双线性因子分析

    Robust bilinear factor analysis based on the matrix-variate $t$ distribution. (arXiv:2401.02203v1 [stat.ML])

    [http://arxiv.org/abs/2401.02203](http://arxiv.org/abs/2401.02203)

    本文提出了一种基于矩阵变量t分布的鲁棒双线性因子分析（tbfa）模型，可以从重尾或受污染的矩阵数据中同时提取行和列变量的公共因子。

    

    基于多元t分布的因子分析（tfa）是一种在重尾或受污染数据上提取公共因子的有用的鲁棒工具。然而，tfa只适用于向量数据。当tfa应用于矩阵数据时，通常会先将矩阵观测向量化。这带来了两个挑战：（i）数据的固有矩阵结构被破坏，（ii）鲁棒性可能丢失，因为向量化的矩阵数据通常会导致较高的数据维度，这容易导致tfa的崩溃。为了解决这些问题，本文从矩阵数据的内在结构出发，提出了一种新的鲁棒因子分析模型，即基于矩阵变量t分布的双线性因子分析（tbfa）。其创新之处在于它能够同时对感兴趣的行和列变量从重尾或受污染的矩阵数据中提取公共因子。本文提出了两种最大似然估计的高效算法来求解模型参数。

    Factor Analysis based on multivariate $t$ distribution ($t$fa) is a useful robust tool for extracting common factors on heavy-tailed or contaminated data. However, $t$fa is only applicable to vector data. When $t$fa is applied to matrix data, it is common to first vectorize the matrix observations. This introduces two challenges for $t$fa: (i) the inherent matrix structure of the data is broken, and (ii) robustness may be lost, as vectorized matrix data typically results in a high data dimension, which could easily lead to the breakdown of $t$fa. To address these issues, starting from the intrinsic matrix structure of matrix data, a novel robust factor analysis model, namely bilinear factor analysis built on the matrix-variate $t$ distribution ($t$bfa), is proposed in this paper. The novelty is that it is capable to simultaneously extract common factors for both row and column variables of interest on heavy-tailed or contaminated matrix data. Two efficient algorithms for maximum likeli
    
[^5]: 基于能量的扩散生成器用于高效采样Boltzmann分布

    Energy based diffusion generator for efficient sampling of Boltzmann distributions. (arXiv:2401.02080v1 [cs.LG])

    [http://arxiv.org/abs/2401.02080](http://arxiv.org/abs/2401.02080)

    介绍了一种称为基于能量的扩散生成器的新型采样器，用于从任意目标分布中生成样本，并通过扩散模型和广义哈密顿动力学提高采样性能。在各种复杂分布函数上的实证评估中表现出优越性。

    

    我们介绍了一种称为基于能量的扩散生成器的新型采样器，用于从任意目标分布中生成样本。采样模型采用类似变分自编码器的结构，利用解码器将来自简单分布的潜在变量转换为逼近目标分布的随机变量，并设计了基于扩散模型的编码器。利用扩散模型对复杂分布的强大建模能力，我们可以获得生成样本和目标分布之间的Kullback-Leibler散度的准确变分估计。此外，我们提出了基于广义哈密顿动力学的解码器，进一步提高采样性能。通过实证评估，我们展示了我们的方法在各种复杂分布函数上的有效性，展示了其相对于现有方法的优越性。

    We introduce a novel sampler called the energy based diffusion generator for generating samples from arbitrary target distributions. The sampling model employs a structure similar to a variational autoencoder, utilizing a decoder to transform latent variables from a simple distribution into random variables approximating the target distribution, and we design an encoder based on the diffusion model. Leveraging the powerful modeling capacity of the diffusion model for complex distributions, we can obtain an accurate variational estimate of the Kullback-Leibler divergence between the distributions of the generated samples and the target. Moreover, we propose a decoder based on generalized Hamiltonian dynamics to further enhance sampling performance. Through empirical evaluation, we demonstrate the effectiveness of our method across various complex distribution functions, showcasing its superiority compared to existing methods.
    
[^6]: U-信任模型 - 决策中的可靠性、能力和信心

    U-Trustworthy Models.Reliability, Competence, and Confidence in Decision-Making. (arXiv:2401.02062v1 [stat.ML])

    [http://arxiv.org/abs/2401.02062](http://arxiv.org/abs/2401.02062)

    提出了一种基于哲学文献的新的信任框架$\mathcal{U}$-信任度，用于评估AI系统的信任性，该框架挑战了传统的概率框架，并提出了解决方案以克服偏见和误导性信任评估的风险。

    

    随着对预测模型中偏见和歧视的担忧日益增长，人工智能社区越来越关注评估AI系统的信任性。传统上，信任的AI文献依赖于概率框架和校准作为信任的先决条件。在这项工作中，我们离开了这个观点，提出了一个受哲学文献中关于信任的启示的新的信任框架。我们提出了一个精确的数学定义，称为$\mathcal{U}$-信任度，专门为最大化效用函数的任务子集量身定制。我们认为一个模型的$\mathcal{U}$-信任度取决于它在这个任务子集中最大化贝叶斯效用的能力。我们的第一组结果挑战了概率框架，通过展示它可能偏向不太可信的模型，并引入误导性信任评估的风险。在$\mathcal{U}$-信任度的背景下，我们提供了一些解决方案，以克服这些挑战，并提高模型的信任度估计。

    With growing concerns regarding bias and discrimination in predictive models, the AI community has increasingly focused on assessing AI system trustworthiness. Conventionally, trustworthy AI literature relies on the probabilistic framework and calibration as prerequisites for trustworthiness. In this work, we depart from this viewpoint by proposing a novel trust framework inspired by the philosophy literature on trust. We present a precise mathematical definition of trustworthiness, termed $\mathcal{U}$-trustworthiness, specifically tailored for a subset of tasks aimed at maximizing a utility function. We argue that a model's $\mathcal{U}$-trustworthiness is contingent upon its ability to maximize Bayes utility within this task subset. Our first set of results challenges the probabilistic framework by demonstrating its potential to favor less trustworthy models and introduce the risk of misleading trustworthiness assessments. Within the context of $\mathcal{U}$-trustworthiness, we prov
    
[^7]: 使用无约束ReLU特征模型进行交叉熵类别不平衡学习的神经折叠

    Neural Collapse for Cross-entropy Class-Imbalanced Learning with Unconstrained ReLU Feature Model. (arXiv:2401.02058v1 [cs.LG])

    [http://arxiv.org/abs/2401.02058](http://arxiv.org/abs/2401.02058)

    本文研究了在交叉熵类别不平衡学习中的神经折叠现象，证明了在存在类别不平衡情况下，神经折叠仍然存在，但类均值的几何特性会发生偏移。

    

    目前训练深度神经网络进行分类任务的范式包括最小化经验风险，将训练损失值推向零，即使训练误差已经消失。在训练的最后阶段，观察到最后一层特征会折叠到它们的类均值，并且这些类均值会收敛到一个简单典型等角紧框（ETF）的顶点。这一现象被称为神经折叠（NC）。为了从理论上理解这一现象，最近的研究使用了一个简化的无约束特征模型来证明NC出现在训练问题的全局解中。然而，当训练数据集存在类别不平衡时，一些NC特性将不再成立。例如，当损失收敛时，类均值几何会偏离简单典型等角紧框。在本文中，我们将NC推广到不平衡情况下的交叉熵损失和无约束ReLU特征模型。我们证明，在训练问题的全局解中，当存在类别不平衡时，NC仍然存在，但对于类均值的几何特性会发生偏移。

    The current paradigm of training deep neural networks for classification tasks includes minimizing the empirical risk that pushes the training loss value towards zero, even after the training error has been vanished. In this terminal phase of training, it has been observed that the last-layer features collapse to their class-means and these class-means converge to the vertices of a simplex Equiangular Tight Frame (ETF). This phenomenon is termed as Neural Collapse (NC). To theoretically understand this phenomenon, recent works employ a simplified unconstrained feature model to prove that NC emerges at the global solutions of the training problem. However, when the training dataset is class-imbalanced, some NC properties will no longer be true. For example, the class-means geometry will skew away from the simplex ETF when the loss converges. In this paper, we generalize NC to imbalanced regime for cross-entropy loss under the unconstrained ReLU feature model. We prove that, while the wi
    
[^8]: ${\Lambda}$CDM宇宙学中的层次聚类通过持续能量方法

    Hierarchical Clustering in ${\Lambda}$CDM Cosmologies via Persistence Energy. (arXiv:2401.01988v1 [astro-ph.CO])

    [http://arxiv.org/abs/2401.01988](http://arxiv.org/abs/2401.01988)

    通过持续能量方法研究了${\Lambda}$CDM宇宙学中的层次聚类，发现持续能量与红移值之间存在相关性，揭示了宇宙结构的动力学特征。

    

    在这项研究中，我们利用拓扑数据分析的先进方法，研究宇宙网络的结构演化。我们的方法涉及到利用持续信号这一创新方法，将持续图重新概念化为$\mathbb R^2_+$空间中的信号，从而实现持续图的嵌入。利用这种方法，我们分析了三种典型的宇宙结构：团簇、纤维和虚空。其中的一个核心发现是持续能量与红移值之间的相关性，将持续同调与宇宙结构的演化联系起来，并提供了有关宇宙结构动力学的见解。

    In this research, we investigate the structural evolution of the cosmic web, employing advanced methodologies from Topological Data Analysis. Our approach involves leveraging $Persistence$ $Signals$, an innovative method from recent literature that facilitates the embedding of persistence diagrams into vector spaces by re-conceptualizing them as signals in $\mathbb R^2_+$. Utilizing this methodology, we analyze three quintessential cosmic structures: clusters, filaments, and voids. A central discovery is the correlation between $Persistence$ $Energy$ and redshift values, linking persistent homology with cosmic evolution and providing insights into the dynamics of cosmic structures.
    
[^9]: 超越遗憾：贝叶斯优化的几何度量

    Beyond Regrets: Geometric Metrics for Bayesian Optimization. (arXiv:2401.01981v1 [cs.LG])

    [http://arxiv.org/abs/2401.01981](http://arxiv.org/abs/2401.01981)

    本论文提出了四个新的几何度量，可以比较贝叶斯优化算法在考虑查询点和全局最优解的几何特性时的性能。

    

    贝叶斯优化是一种针对黑盒子目标函数的原则性优化策略。它在科学发现和实验设计等各种实际应用中的效果得到了证明。通常，贝叶斯优化的性能是通过基于遗憾的度量来评估的，如瞬时遗憾、简单遗憾和累积遗憾。这些度量仅依赖于函数评估，因此它们不考虑查询点和全局解之间的几何关系，也不考虑查询点本身。值得注意的是，它们不能区分是否成功找到了多个全局解。此外，它们也不能评估贝叶斯优化在给定搜索空间中利用和探索的能力。为了解决这些问题，我们提出了四个新的几何度量，即精确度、召回率、平均度和平均距离。这些度量使我们能够比较考虑查询点和全局最优解的几何特性的贝叶斯优化算法。

    Bayesian optimization is a principled optimization strategy for a black-box objective function. It shows its effectiveness in a wide variety of real-world applications such as scientific discovery and experimental design. In general, the performance of Bayesian optimization is assessed by regret-based metrics such as instantaneous, simple, and cumulative regrets. These metrics only rely on function evaluations, so that they do not consider geometric relationships between query points and global solutions, or query points themselves. Notably, they cannot discriminate if multiple global solutions are successfully found. Moreover, they do not evaluate Bayesian optimization's abilities to exploit and explore a search space given. To tackle these issues, we propose four new geometric metrics, i.e., precision, recall, average degree, and average distance. These metrics allow us to compare Bayesian optimization algorithms considering the geometry of both query points and global optima, or que
    
[^10]: 连续时间序列集合的概率建模

    Probabilistic Modeling for Sequences of Sets in Continuous-Time. (arXiv:2312.15045v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.15045](http://arxiv.org/abs/2312.15045)

    本文提出了一个通用的连续时间序列集合的概率建模框架，适用于处理每个事件与一组项目相关联的情况。引入了适用于任何强度为基础的递归神经点过程模型的推理方法，可用于回答关于序列历史条件下的概率查询问题。

    

    在连续时间事件数据的统计参数模型工具箱中，神经标记时间点过程是一个有价值的补充。这些模型适用于每个事件与单个项目（单个事件类型或“标记”）相关联的序列，但不适用于每个事件与一组项目相关联的实际情况。本文中，我们开发了一个通用的连续时间集合数值数据建模框架，与任何基于强度的递归神经点过程模型兼容。此外，我们还开发了推理方法，可使用这些模型回答诸如“在考虑序列历史的条件下，项目A在项目B之前观察到的概率”等概率查询问题。由于问题设置的连续时间性质和每个事件的潜在结果空间的组合极大，对于神经模型来说，计算这些查询的精确答案通常是不可行的。

    Neural marked temporal point processes have been a valuable addition to the existing toolbox of statistical parametric models for continuous-time event data. These models are useful for sequences where each event is associated with a single item (a single type of event or a "mark") -- but such models are not suited for the practical situation where each event is associated with a set of items. In this work, we develop a general framework for modeling set-valued data in continuous-time, compatible with any intensity-based recurrent neural point process model. In addition, we develop inference methods that can use such models to answer probabilistic queries such as "the probability of item $A$ being observed before item $B$," conditioned on sequence history. Computing exact answers for such queries is generally intractable for neural models due to both the continuous-time nature of the problem setting and the combinatorially-large space of potential outcomes for each event. To address th
    
[^11]: Bayesian网络的熵和Kullback-Leibler散度：计算复杂度和高效实现

    Entropy and the Kullback-Leibler Divergence for Bayesian Networks: Computational Complexity and Efficient Implementation. (arXiv:2312.01520v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2312.01520](http://arxiv.org/abs/2312.01520)

    本文提出了一种计算贝叶斯网络中Shannon熵和Kullback-Leibler散度的高效算法，并通过一系列数值示例进行了演示。此外，还展示了如何将高斯贝叶斯网络中KL的计算复杂度从立方降低到二次。

    

    贝叶斯网络（BNs）是机器学习和因果推断中的基础模型。它们的图结构可以处理高维问题，并将其分为稀疏的一系列较小问题，这是Judea Pearl的因果性的基础，也决定了它们的可解释性和可理解性。尽管它们很受欢迎，但在文献中几乎没有关于如何在最常见的分布假设下计算BNs的Shannon熵和Kullback-Leibler（KL）散度的资源。在本文中，我们利用BNs的图结构提供了计算效率高的算法，并用一整套数值示例说明了它们。在此过程中，我们展示了可以将高斯BNs的KL计算复杂度从立方降低到二次的可能性。

    Bayesian networks (BNs) are a foundational model in machine learning and causal inference. Their graphical structure can handle high-dimensional problems, divide them into a sparse collection of smaller ones, underlies Judea Pearl's causality, and determines their explainability and interpretability. Despite their popularity, there are almost no resources in the literature on how to compute Shannon's entropy and the Kullback-Leibler (KL) divergence for BNs under their most common distributional assumptions. In this paper, we provide computationally efficient algorithms for both by leveraging BNs' graphical structure, and we illustrate them with a complete set of numerical examples. In the process, we show it is possible to reduce the computational complexity of KL from cubic to quadratic for Gaussian BNs.
    
[^12]: 关于神经网络模型压缩的框架、算法和收敛保证

    On Model Compression for Neural Networks: Framework, Algorithm, and Convergence Guarantee. (arXiv:2303.06815v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.06815](http://arxiv.org/abs/2303.06815)

    本文提出了一个框架和算法，从非凸优化的角度来进行神经网络模型压缩。算法解决了梯度消失/爆炸问题，并保证了收敛性。

    

    模型压缩对于部署神经网络（NN）至关重要，特别是在许多应用程序中计算设备的内存和存储有限的情况下。本文关注两种神经网络模型压缩技术：低秩逼近和权重裁剪，这些技术目前非常流行。然而，使用低秩逼近和权重裁剪训练神经网络总是会遭受显著的准确性损失和收敛问题。本文提出了一个全面的框架，从非凸优化的新视角设计了适当的目标函数来进行模型压缩。然后，我们引入了一种块坐标下降（BCD）算法NN-BCD来解决非凸优化问题。我们算法的一个优点是可以获得具有闭式形式的高效迭代方案，从而避免了梯度消失/爆炸的问题。此外，我们的算法利用了Kurdyka-{\L}ojasiewicz (K{\L})性质，保证了算法的收敛性。

    Model compression is a crucial part of deploying neural networks (NNs), especially when the memory and storage of computing devices are limited in many applications. This paper focuses on two model compression techniques: low-rank approximation and weight pruning in neural networks, which are very popular nowadays. However, training NN with low-rank approximation and weight pruning always suffers significant accuracy loss and convergence issues. In this paper, a holistic framework is proposed for model compression from a novel perspective of nonconvex optimization by designing an appropriate objective function. Then, we introduce NN-BCD, a block coordinate descent (BCD) algorithm to solve the nonconvex optimization. One advantage of our algorithm is that an efficient iteration scheme can be derived with closed-form, which is gradient-free. Therefore, our algorithm will not suffer from vanishing/exploding gradient problems. Furthermore, with the Kurdyka-{\L}ojasiewicz (K{\L}) property o
    
[^13]: 用核斯坦离差控制矩

    Controlling Moments with Kernel Stein Discrepancies. (arXiv:2211.05408v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.05408](http://arxiv.org/abs/2211.05408)

    本研究分析了核斯坦离差（KSD）控制性质，发现标准KSD无法控制矩的收敛，提出了可控制矩和弱收敛的下游扩散KSD，并且发展了可以准确描述$q$-Wasserstein收敛的KSD。

    

    核斯坦离差（KSD）用于衡量分布逼近的质量，并且可以在目标密度具有不可计算的归一化常数时计算。显著的应用包括诊断近似MCMC采样器和非归一化统计模型的适配度检验。本文分析了KSD的收敛控制性质。我们首先证明了用于弱收敛控制的标准KSD无法控制矩的收敛。为了解决这个限制，我们提供了一组充分条件，下游扩散KSD可以同时控制矩和弱收敛。作为一个直接的结果，我们发展了对于每个$q>0$，第一组已知可以准确描述$q$-Wasserstein收敛的KSD。

    Kernel Stein discrepancies (KSDs) measure the quality of a distributional approximation and can be computed even when the target density has an intractable normalizing constant. Notable applications include the diagnosis of approximate MCMC samplers and goodness-of-fit tests for unnormalized statistical models. The present work analyzes the convergence control properties of KSDs. We first show that standard KSDs used for weak convergence control fail to control moment convergence. To address this limitation, we next provide sufficient conditions under which alternative diffusion KSDs control both moment and weak convergence. As an immediate consequence we develop, for each $q > 0$, the first KSDs known to exactly characterize $q$-Wasserstein convergence.
    
[^14]: 高维函数逼近的切片梯度增强克里金方法

    Sliced gradient-enhanced Kriging for high-dimensional function approximation. (arXiv:2204.03562v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2204.03562](http://arxiv.org/abs/2204.03562)

    本文提出了一种名为切片GE-Kriging（SGE-Kriging）的新方法，用于解决高维函数逼近中的相关矩阵和超参数调整问题。

    

    梯度增强克里金（GE-Kriging）是一种用于逼近昂贵计算模型的已建立的代理建模技术。然而，由于固有相关矩阵的大小以及相关的高维超参数调整问题，它在高维问题上变得不实用。为了解决这些问题，在本文中提出了一种名为切片GE-Kriging（SGE-Kriging）的新方法，用于减小相关矩阵的大小和超参数的数量。我们首先将训练样本集拆分为多个切片，并通过贝叶斯定理使用切片似然函数来逼近完整的似然函数，在切片似然函数中，我们使用多个小的相关矩阵来描述样本集的相关性，而不是一个大的矩阵。然后，我们通过学习超参数与导数之间的关系，将原始的高维超参数调整问题替换为一个低维问题。

    Gradient-enhanced Kriging (GE-Kriging) is a well-established surrogate modelling technique for approximating expensive computational models. However, it tends to get impractical for high-dimensional problems due to the size of the inherent correlation matrix and the associated high-dimensional hyper-parameter tuning problem. To address these issues, a new method, called sliced GE-Kriging (SGE-Kriging), is developed in this paper for reducing both the size of the correlation matrix and the number of hyper-parameters. We first split the training sample set into multiple slices, and invoke Bayes' theorem to approximate the full likelihood function via a sliced likelihood function, in which multiple small correlation matrices are utilized to describe the correlation of the sample set rather than one large one. Then, we replace the original high-dimensional hyper-parameter tuning problem with a low-dimensional counterpart by learning the relationship between the hyper-parameters and the der
    
[^15]: 平均分布优化平滑损失函数

    Federated Optimization of Smooth Loss Functions. (arXiv:2201.01954v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.01954](http://arxiv.org/abs/2201.01954)

    本文研究了在联邦学习框架下的平均分布优化平滑损失函数的问题，提出了联邦低秩梯度下降（FedLRGD）算法来利用数据的平滑性，从而实现更好的优化结果。

    

    在这项工作中，我们研究了在联邦学习框架下的经验风险最小化（ERM），其中一个中央服务器使用存储在$m$个客户端上的训练数据来最小化ERM目标函数。在这种情况下，联邦平均（FedAve）算法是确定ERM问题的$\epsilon$近似解的主要方法。类似于标准优化算法，FedAve的收敛分析仅依赖于优化参数中损失函数的平滑性。然而，损失函数在训练数据中通常也非常平滑。为了利用这种额外的平滑性，我们提出了联邦低秩梯度下降（FedLRGD）算法。由于数据的平滑性在损失函数上引入了近似的低秩结构，我们的方法首先在服务器和客户端之间进行了几轮通信，以学习服务器可以用来近似客户端梯度的权重。然后，我们的方法在服务器上解决ERM问题。

    In this work, we study empirical risk minimization (ERM) within a federated learning framework, where a central server minimizes an ERM objective function using training data that is stored across $m$ clients. In this setting, the Federated Averaging (FedAve) algorithm is the staple for determining $\epsilon$-approximate solutions to the ERM problem. Similar to standard optimization algorithms, the convergence analysis of FedAve only relies on smoothness of the loss function in the optimization parameter. However, loss functions are often very smooth in the training data too. To exploit this additional smoothness, we propose the Federated Low Rank Gradient Descent (FedLRGD) algorithm. Since smoothness in data induces an approximate low rank structure on the loss function, our method first performs a few rounds of communication between the server and clients to learn weights that the server can use to approximate clients' gradients. Then, our method solves the ERM problem at the server 
    
[^16]: 用于场景分析的同质 Ising 模型的快速近似计算方法

    Fast approximations in the homogeneous Ising model for use in scene analysis. (arXiv:1712.02195v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/1712.02195](http://arxiv.org/abs/1712.02195)

    本文提供了一种快速近似计算同质 Ising 模型中重要量的方法，该方法的表现在模拟研究中表现良好，可用于场景分析。

    

    Ising 模型在许多应用中都很重要，但其归一化常数、活动顶点数的平均值和自旋相互作用的均值难以计算。我们提供了准确的近似值，使得在同质情况下可以数值计算这些量。模拟研究表明，与马尔可夫链蒙特卡洛方法相比，我们的方法性能良好，且所需时间只是那些随机方法的一小部分。我们的近似值在执行功能磁共振激活检测实验的贝叶斯推断以及植物生产中年增量空间图案的各向异性的似然比检验中得到了体现。

    The Ising model is important in statistical modeling and inference in many applications, however its normalizing constant, mean number of active vertices and mean spin interaction are intractable to compute. We provide accurate approximations that make it possible to numerically calculate these quantities in the homogeneous case. Simulation studies indicate good performance when compared to Markov Chain Monte Carlo methods and at a tiny fraction of the time taken by those stochastic approaches. The value of our approximations is illustrated in performing Bayesian inference in a functional Magnetic Resonance Imaging activation detection experiment, and also in likelihood ratio testing for anisotropy in the spatial patterns of yearly increases in pistachio tree yields.
    

