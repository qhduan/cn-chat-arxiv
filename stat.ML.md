# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Average gradient outer product as a mechanism for deep neural collapse](https://arxiv.org/abs/2402.13728) | 本文通过提供证据表明，深度神经网络中的神经坍塌主要是通过平均梯度外积进行深度特征学习的，权重的奇异结构与AGOP高度相关，导致类内变异坍塌。 |
| [^2] | [Timer: Transformers for Time Series Analysis at Scale](https://arxiv.org/abs/2402.02368) | 本文旨在早期开发大规模时间序列模型（LTSM），通过预训练和GPT风格架构，克服深度模型在小样本场景中的性能瓶颈，并实现在时间序列分析中的大样本泛化能力、可扩展性和任务普适性。 |
| [^3] | [Individualized Multi-Treatment Response Curves Estimation using RBF-net with Shared Neurons.](http://arxiv.org/abs/2401.16571) | 我们提出了一种使用共享神经元的RBF网络的非参数化治疗效应估计方法，适用于多治疗设置。该方法能够建模治疗结果的共同性，并在贝叶斯框架下实现估计和推断，通过模拟实验证明了其数值性能，应用于真实临床数据后也得到了有趣的发现。 |
| [^4] | [Plug-and-Play Posterior Sampling under Mismatched Measurement and Prior Models.](http://arxiv.org/abs/2310.03546) | 本研究提出了一种插拔式后验采样算法（PnP-ULA），通过将物理测量模型与深度学习先验相结合，解决了成像逆问题。我们通过理论分析和数值验证，量化了PnP-ULA在不匹配后验分布下的误差界限，结果表明PnP-ULA对于测量模型和去噪器的不匹配非常敏感。 |
| [^5] | [On the convergence of dynamic implementations of Hamiltonian Monte Carlo and No U-Turn Samplers.](http://arxiv.org/abs/2307.03460) | 本文研究了动态实现的Hamiltonian Monte Carlo (HMC)算法和No U-Turn Sampler (NUTS) 的收敛性，证明了NUTS作为动态HMC的特例，并且在一定条件下具有遍历性和几何遍历性。同时改进了HMC的收敛性结果，证明了在目标分布为高斯分布的微扰情况下，无需任何有界条件，HMC也是遍历的。 |
| [^6] | [Lower Complexity Adaptation for Empirical Entropic Optimal Transport.](http://arxiv.org/abs/2306.13580) | 本文研究了经验熵正则化最优输运的统计表现，并证明了它遵循低复杂度适应原则，推导出了其统计界限及参数化速率。 |

# 详细

[^1]: 平均梯度外积作为深度神经坍塌机制的研究

    Average gradient outer product as a mechanism for deep neural collapse

    [https://arxiv.org/abs/2402.13728](https://arxiv.org/abs/2402.13728)

    本文通过提供证据表明，深度神经网络中的神经坍塌主要是通过平均梯度外积进行深度特征学习的，权重的奇异结构与AGOP高度相关，导致类内变异坍塌。

    

    Deep Neural Collapse (DNC)指的是深度神经网络(DNNs)最后几层数据表示的惊人刚性结构。尽管这种现象在各种情境中都得到了测量，但其出现只有部分被理解。本文提供了充分证据，表明DNC主要是通过平均梯度外积(AGOP)进行深度特征学习而发生的。相比于解释神经坍塌的特征不可知方法，如无约束特征模型，这一进展更进一步。我们继续提供证据表明，权重的右奇异向量和奇异值是DNN中类内变异坍塌的主要因素。正如最近的研究所示，这种奇异结构与AGOP的高度相关。然后我们在实验和理论上证明了AGOP在随机初始化的神经网络中引发神经坍塌。

    arXiv:2402.13728v1 Announce Type: new  Abstract: Deep Neural Collapse (DNC) refers to the surprisingly rigid structure of the data representations in the final layers of Deep Neural Networks (DNNs). Though the phenomenon has been measured in a wide variety of settings, its emergence is only partially understood. In this work, we provide substantial evidence that DNC formation occurs primarily through deep feature learning with the average gradient outer product (AGOP). This takes a step further compared to efforts that explain neural collapse via feature-agnostic approaches, such as the unconstrained features model. We proceed by providing evidence that the right singular vectors and values of the weights are responsible for the majority of within-class variability collapse in DNNs. As shown in recent work, this singular structure is highly correlated with that of the AGOP. We then establish experimentally and theoretically that AGOP induces neural collapse in a randomly initialized ne
    
[^2]: 计时器: 用于大规模时间序列分析的Transformer模型

    Timer: Transformers for Time Series Analysis at Scale

    [https://arxiv.org/abs/2402.02368](https://arxiv.org/abs/2402.02368)

    本文旨在早期开发大规模时间序列模型（LTSM），通过预训练和GPT风格架构，克服深度模型在小样本场景中的性能瓶颈，并实现在时间序列分析中的大样本泛化能力、可扩展性和任务普适性。

    

    深度学习在时间序列分析方面做出了显著贡献。然而，在现实世界的小样本场景中，深度模型可能遇到性能瓶颈，这可能由于当前基准测试中小模型的性能饱和而隐蔽。同时，通过大规模预训练，大模型在这些场景中展示了巨大的能力。随着大型语言模型的出现，取得了持续的进展，在少样本泛化能力、可扩展性和任务普适性方面展现了前所未有的能力，但这些能力在时间序列模型中不存在。为了改变目前在特定数据集上从头开始训练小模型的做法，本文旨在早期开发大规模时间序列模型（LTSM）。在预训练期间，我们策划了包含10亿个时间点的大规模数据集，将异构时间序列统一为单序列序列（S3）格式，并开发了面向LTSM的GPT风格架构。

    Deep learning has contributed remarkably to the advancement of time series analysis. Still, deep models can encounter performance bottlenecks in real-world small-sample scenarios, which can be concealed due to the performance saturation with small models on current benchmarks. Meanwhile, large models have demonstrated great powers in these scenarios through large-scale pre-training. Continuous progresses have been achieved as the emergence of large language models, exhibiting unprecedented ability in few-shot generalization, scalability, and task generality, which is however absent in time series models. To change the current practices of training small models on specific datasets from scratch, this paper aims at an early development of large time series models (LTSM). During pre-training, we curate large-scale datasets with up to 1 billion time points, unify heterogeneous time series into single-series sequence (S3) format, and develop the GPT-style architecture toward LTSMs. To meet 
    
[^3]: 使用共享神经元的RBF网络估计个体化多治疗反应曲线

    Individualized Multi-Treatment Response Curves Estimation using RBF-net with Shared Neurons. (arXiv:2401.16571v1 [stat.ME])

    [http://arxiv.org/abs/2401.16571](http://arxiv.org/abs/2401.16571)

    我们提出了一种使用共享神经元的RBF网络的非参数化治疗效应估计方法，适用于多治疗设置。该方法能够建模治疗结果的共同性，并在贝叶斯框架下实现估计和推断，通过模拟实验证明了其数值性能，应用于真实临床数据后也得到了有趣的发现。

    

    异质治疗效应估计是精确医学中的一个重要问题。我们的研究兴趣在于基于一些外部协变量，确定不同治疗方式的差异效应。我们提出了一种新颖的非参数化治疗效应估计方法，适用于多治疗设置。我们对响应曲线的非参数建模依赖于带有共享隐藏神经元的径向基函数（RBF）网络。因此，我们的模型有助于建模治疗结果的共同性。我们在贝叶斯框架下开发了估计和推断方案，并通过高效的马尔科夫链蒙特卡罗算法进行实现，适当地处理了分析各个方面的不确定性。通过模拟实验，展示了该方法的数值性能。将我们提出的方法应用于MIMIC数据后，我们得到了关于不同治疗策略对ICU住院时间和12小时SOFA评分的影响的一些有趣发现。

    Heterogeneous treatment effect estimation is an important problem in precision medicine. Specific interests lie in identifying the differential effect of different treatments based on some external covariates. We propose a novel non-parametric treatment effect estimation method in a multi-treatment setting. Our non-parametric modeling of the response curves relies on radial basis function (RBF)-nets with shared hidden neurons. Our model thus facilitates modeling commonality among the treatment outcomes. The estimation and inference schemes are developed under a Bayesian framework and implemented via an efficient Markov chain Monte Carlo algorithm, appropriately accommodating uncertainty in all aspects of the analysis. The numerical performance of the method is demonstrated through simulation experiments. Applying our proposed method to MIMIC data, we obtain several interesting findings related to the impact of different treatment strategies on the length of ICU stay and 12-hour SOFA sc
    
[^4]: 插拔式后验采样在不匹配测量和先验模型下的应用

    Plug-and-Play Posterior Sampling under Mismatched Measurement and Prior Models. (arXiv:2310.03546v1 [stat.ML])

    [http://arxiv.org/abs/2310.03546](http://arxiv.org/abs/2310.03546)

    本研究提出了一种插拔式后验采样算法（PnP-ULA），通过将物理测量模型与深度学习先验相结合，解决了成像逆问题。我们通过理论分析和数值验证，量化了PnP-ULA在不匹配后验分布下的误差界限，结果表明PnP-ULA对于测量模型和去噪器的不匹配非常敏感。

    

    后验采样已被证明是解决成像逆问题的强大贝叶斯方法。最近发展起来的插拔式未调整朗之万算法（PnP-ULA）通过将物理测量模型与使用图像去噪器指定的深度学习先验相结合，成为一种有前景的蒙特卡洛采样和最小均方误差（MMSE）估计方法。然而，PnP-ULA的采样分布与不匹配的数据保真度和去噪器之间的复杂关系尚未经过理论分析。我们通过提出一种后验-L2拟度量并利用它来量化PnP-ULA在不匹配的后验分布下的显式误差界限来填补这一空白。我们在多个逆问题上对我们的理论进行了数值验证，如从高斯混合模型和图像去模糊中采样。我们的结果表明，PnP-ULA的采样分布对于测量模型和去噪器的不匹配非常敏感，并可以精确地描述其特征。

    Posterior sampling has been shown to be a powerful Bayesian approach for solving imaging inverse problems. The recent plug-and-play unadjusted Langevin algorithm (PnP-ULA) has emerged as a promising method for Monte Carlo sampling and minimum mean squared error (MMSE) estimation by combining physical measurement models with deep-learning priors specified using image denoisers. However, the intricate relationship between the sampling distribution of PnP-ULA and the mismatched data-fidelity and denoiser has not been theoretically analyzed. We address this gap by proposing a posterior-L2 pseudometric and using it to quantify an explicit error bound for PnP-ULA under mismatched posterior distribution. We numerically validate our theory on several inverse problems such as sampling from Gaussian mixture models and image deblurring. Our results suggest that the sensitivity of the sampling distribution of PnP-ULA to a mismatch in the measurement model and the denoiser can be precisely characte
    
[^5]: 动态实现的Hamiltonian Monte Carlo和No U-Turn Samplers的收敛性

    On the convergence of dynamic implementations of Hamiltonian Monte Carlo and No U-Turn Samplers. (arXiv:2307.03460v1 [stat.CO])

    [http://arxiv.org/abs/2307.03460](http://arxiv.org/abs/2307.03460)

    本文研究了动态实现的Hamiltonian Monte Carlo (HMC)算法和No U-Turn Sampler (NUTS) 的收敛性，证明了NUTS作为动态HMC的特例，并且在一定条件下具有遍历性和几何遍历性。同时改进了HMC的收敛性结果，证明了在目标分布为高斯分布的微扰情况下，无需任何有界条件，HMC也是遍历的。

    

    针对动态实现的Hamiltonian Monte Carlo (HMC)算法，例如No U-Turn Sampler (NUTS)，在许多具有挑战性的推理问题中具有成功的经验证据，但关于它们行为的理论结果还不足。本文旨在填补这一空白。具体而言，我们考虑了一个称为动态HMC的通用MCMC算法类。我们证明了这个通用框架涵盖了NUTS作为一个特例，并且作为一个附带结果，证明了目标分布的不变性。其次，我们建立了使NUTS不可约和非周期的条件，并作为推论而证明了遍历性。在类似于HMC的条件下，我们还证明了NUTS具有几何遍历性。最后，我们改进了现有的HMC收敛性结果，证明了这个方法在目标分布是高斯分布的微扰的情况下，无需对步长和leapfrog步数进行任何有界条件，也是遍历的。

    There is substantial empirical evidence about the success of dynamic implementations of Hamiltonian Monte Carlo (HMC), such as the No U-Turn Sampler (NUTS), in many challenging inference problems but theoretical results about their behavior are scarce. The aim of this paper is to fill this gap. More precisely, we consider a general class of MCMC algorithms we call dynamic HMC. We show that this general framework encompasses NUTS as a particular case, implying the invariance of the target distribution as a by-product. Second, we establish conditions under which NUTS is irreducible and aperiodic and as a corrolary ergodic. Under conditions similar to the ones existing for HMC, we also show that NUTS is geometrically ergodic. Finally, we improve existing convergence results for HMC showing that this method is ergodic without any boundedness condition on the stepsize and the number of leapfrog steps, in the case where the target is a perturbation of a Gaussian distribution.
    
[^6]: 经验熵正则化最优输运的低复杂度适应性

    Lower Complexity Adaptation for Empirical Entropic Optimal Transport. (arXiv:2306.13580v1 [math.ST])

    [http://arxiv.org/abs/2306.13580](http://arxiv.org/abs/2306.13580)

    本文研究了经验熵正则化最优输运的统计表现，并证明了它遵循低复杂度适应原则，推导出了其统计界限及参数化速率。

    

    经验熵正则化最优输运 (EOT) 是优化输运 (OT) 的一种有效且计算可行的替代方案，对大规模数据分析有着广泛的应用。本文推导出了 EOT 成本的新的统计界限，并显示它们在熵正则化参数 $\epsilon$ 和样本大小 $n$ 的统计性能仅取决于两个概率测度之中较简单的那个。例如，在充分平滑的成本下，这会产生具有$\epsilon^{-d/2}$因子的参数化速率$n^{-1/2}$，其中$d$是两个总体测度的最小维度。这确认了经验EOT也遵循了最近才为未规则化OT确认的低复杂度适应原则的标志性特征。根据我们的理论，我们展示了欧几里得空间上的测度的经验熵Gromov-Wasserstein距离及其未规则化版本也遵循此原则。

    Entropic optimal transport (EOT) presents an effective and computationally viable alternative to unregularized optimal transport (OT), offering diverse applications for large-scale data analysis. In this work, we derive novel statistical bounds for empirical plug-in estimators of the EOT cost and show that their statistical performance in the entropy regularization parameter $\epsilon$ and the sample size $n$ only depends on the simpler of the two probability measures. For instance, under sufficiently smooth costs this yields the parametric rate $n^{-1/2}$ with factor $\epsilon^{-d/2}$, where $d$ is the minimum dimension of the two population measures. This confirms that empirical EOT also adheres to the lower complexity adaptation principle, a hallmark feature only recently identified for unregularized OT. As a consequence of our theory, we show that the empirical entropic Gromov-Wasserstein distance and its unregularized version for measures on Euclidean spaces also obey this princip
    

