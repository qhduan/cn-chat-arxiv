# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Backward and Galerkin Methods for the Finite State Master Equation](https://arxiv.org/abs/2403.04975) | 该论文提出了两种神经网络方法来解决有限状态均场博弈的主方程，通过数值实验验证了这两种方法的有效性。 |
| [^2] | [Nonparametric Instrumental Variable Regression through Stochastic Approximate Gradients](https://arxiv.org/abs/2402.05639) | 本文提出了一种通过随机近似梯度最小化投影群体风险的新型非参数仪器变量回归框架，并通过理论和实证实验证明了其竞争性能。此外，本文还处理了二元结果的情况，取得了有希望的结果。 |
| [^3] | [On Provable Length and Compositional Generalization](https://arxiv.org/abs/2402.04875) | 本研究针对包括深度集合、变压器、状态空间模型和简单递归神经网络等多种架构，探索了可证明的长度和组合泛化，认为对于长度和组合泛化，不同架构需要不同程度的表示识别。 |
| [^4] | [Dynamic Incremental Optimization for Best Subset Selection](https://arxiv.org/abs/2402.02322) | 本文研究了一类$\ell_0$正则化问题的对偶形式，并提出了一种高效的原对偶算法，通过充分利用对偶范围估计和增量策略，提高了最佳子集选择问题的解决方案的效率和统计性质。 |
| [^5] | [Guaranteed Nonconvex Factorization Approach for Tensor Train Recovery.](http://arxiv.org/abs/2401.02592) | 本研究提出了一种保证非凸分解方法用于张量列车恢复的新方法。该方法通过优化左正交TT格式来实现正交结构，并使用黎曼梯度下降算法来优化因子。我们证明了该方法具有局部线性收敛性，并且在满足受限等谱性质的条件下能够以线性速率收敛到真实张量。 |
| [^6] | [Upgrading VAE Training With Unlimited Data Plans Provided by Diffusion Models.](http://arxiv.org/abs/2310.19653) | 这项研究通过在预训练的扩散模型生成的样本上进行训练，有效减轻了VAE中编码器的过拟合问题。 |
| [^7] | [Boosted Control Functions.](http://arxiv.org/abs/2310.05805) | 本研究通过建立同时方程模型和控制函数与分布概括的新连接，解决了在存在未观察到的混淆情况下，针对不同训练和测试分布的预测问题。 |
| [^8] | [Distributionally Robust Machine Learning with Multi-source Data.](http://arxiv.org/abs/2309.02211) | 本文提出了一种基于多源数据的分布鲁棒机器学习方法，通过引入组分布鲁棒预测模型来提高具有分布偏移的目标人群的预测准确性。 |
| [^9] | [Non-parametric Hypothesis Tests for Distributional Group Symmetry.](http://arxiv.org/abs/2307.15834) | 该论文提出了用于分布对称性的非参数假设检验方法，适用于具有对称性的数据集。具体而言，该方法在紧致群作用下测试边际或联合分布的不变性，并提出了一种易于实施的条件蒙特卡罗检验。 |
| [^10] | [Variational Sequential Optimal Experimental Design using Reinforcement Learning.](http://arxiv.org/abs/2306.10430) | 该研究提出了一种基于贝叶斯框架和信息增益效用的变分序列最优实验设计方法，通过强化学习求解最优设计策略，适用于多种OED问题，结果具有更高的样本效率和更少的前向模型模拟次数。 |
| [^11] | [Learning High-Dimensional Nonparametric Differential Equations via Multivariate Occupation Kernel Functions.](http://arxiv.org/abs/2306.10189) | 本论文提出了一种线性方法，通过多元占位核函数在高维状态空间中学习非参数ODE系统，可以解决显式公式按二次方缩放的问题。这种方法在高度非线性的数据和图像数据中都具有通用性。 |
| [^12] | [Basis Function Encoding of Numerical Features in Factorization Machines for Improved Accuracy.](http://arxiv.org/abs/2305.14528) | 本文提供了一种能够将数字特征编码为基函数向量的方法，通过在因子机中将该方法应用于因子机中，可以改善推荐系统的准确性。 |
| [^13] | [Global Convergence of SGD On Two Layer Neural Nets.](http://arxiv.org/abs/2210.11452) | 该论文证明了当深度为2的神经网络采用足够平滑凸的激活函数时，SGD可以收敛到全局最小值，证明过程中引入Frobenius范数正则化与恰当分布的参数初始化，同时拓展了连续时间的收敛结果。 |
| [^14] | [Conditionally Calibrated Predictive Distributions by Probability-Probability Map: Application to Galaxy Redshift Estimation and Probabilistic Forecasting.](http://arxiv.org/abs/2205.14568) | 本研究提出了一种名为Cal-PIT的方法，通过学习一个概率-概率映射，解决了预测分布的诊断和校准问题，来实现有条件校准。 |
| [^15] | [Learning Bayesian Networks with Annealing Machine.](http://arxiv.org/abs/2006.06926) | 本文提出了一种用于贝叶斯网络结构学习的模拟退火机器方法，通过先进的候选父节点集合的确定和分解，以及整数规划问题的解决，能够在比特容量有限的情况下高效地解决基于评分的学习问题。 |

# 详细

[^1]: 深度反向和Galerkin方法用于有限状态主方程

    Deep Backward and Galerkin Methods for the Finite State Master Equation

    [https://arxiv.org/abs/2403.04975](https://arxiv.org/abs/2403.04975)

    该论文提出了两种神经网络方法来解决有限状态均场博弈的主方程，通过数值实验验证了这两种方法的有效性。

    

    本论文提出并分析了两种神经网络方法，用于解决有限状态均场博弈的主方程。解决MFGs为具有有限但大量代理人群的随机、微分博弈提供近似纳什均衡。主方程是一个偏微分方程（PDE），其解表征任何可能的初始分布下的MFG均衡。我们提出的第一种方法依赖于在时间分量上的反向归纳，而第二种方法直接解决了PDE，而无需离散化时间。对于两种方法，我们证明了两种类型的结果：存在神经网络，可以使算法的损失函数任意小，并且反之，如果损失很小，则神经网络是主方程解的良好近似。我们通过对文献中的基准问题进行了维度为15的数值实验，并做出对比。

    arXiv:2403.04975v1 Announce Type: cross  Abstract: This paper proposes and analyzes two neural network methods to solve the master equation for finite-state mean field games (MFGs). Solving MFGs provides approximate Nash equilibria for stochastic, differential games with finite but large populations of agents. The master equation is a partial differential equation (PDE) whose solution characterizes MFG equilibria for any possible initial distribution. The first method we propose relies on backward induction in a time component while the second method directly tackles the PDE without discretizing time. For both approaches, we prove two types of results: there exist neural networks that make the algorithms' loss functions arbitrarily small, and conversely, if the losses are small, then the neural networks are good approximations of the master equation's solution. We conclude the paper with numerical experiments on benchmark problems from the literature up to dimension 15, and a compariso
    
[^2]: 非参数仪器变量回归通过随机近似梯度

    Nonparametric Instrumental Variable Regression through Stochastic Approximate Gradients

    [https://arxiv.org/abs/2402.05639](https://arxiv.org/abs/2402.05639)

    本文提出了一种通过随机近似梯度最小化投影群体风险的新型非参数仪器变量回归框架，并通过理论和实证实验证明了其竞争性能。此外，本文还处理了二元结果的情况，取得了有希望的结果。

    

    本文提出了SAGD-IV，这是一种通过使用随机近似梯度来最小化投影群体风险的新型非参数仪器变量（NPIV）回归框架。仪器变量（IV）被广泛应用于计量经济学中，以解决在存在不可观测混淆因素的情况下的估计问题，并且机器学习社区致力于改进现有方法并在NPIV设置下设计新方法，该设置被认为是一个不适定的线性逆问题。我们提供了对我们算法的理论支持，并通过实证实验进一步证明了其竞争性能。此外，我们还处理了二元结果的情况，并取得了有希望的结果，而该情况在社区中没有得到与其连续对应物的同样关注。

    This paper proposes SAGD-IV, a novel framework for conducting nonparametric instrumental variable (NPIV) regression by employing stochastic approximate gradients to minimize the projected populational risk. Instrumental Variables (IVs) are widely used in econometrics to address estimation problems in the presence of unobservable confounders, and the Machine Learning community has devoted significant effort to improving existing methods and devising new ones in the NPIV setting, which is known to be an ill-posed linear inverse problem. We provide theoretical support for our algorithm and further exemplify its competitive performance through empirical experiments. Furthermore, we address, with promising results, the case of binary outcomes, which has not received as much attention from the community as its continuous counterpart.
    
[^3]: 关于可证明的长度和组合泛化

    On Provable Length and Compositional Generalization

    [https://arxiv.org/abs/2402.04875](https://arxiv.org/abs/2402.04875)

    本研究针对包括深度集合、变压器、状态空间模型和简单递归神经网络等多种架构，探索了可证明的长度和组合泛化，认为对于长度和组合泛化，不同架构需要不同程度的表示识别。

    

    长度泛化——对训练时未见到的更长序列的泛化能力，以及组合泛化——对训练时未见到的令牌组合的泛化能力，在序列到序列模型中是重要的非分布化泛化形式。在这项工作中，我们在包括深度集合、变压器、状态空间模型和简单递归神经网络在内的一系列架构中，朝着可证明的长度和组合泛化迈出了第一步。根据架构的不同，我们证明了不同程度的表示识别的必要性，例如与真实表示具有线性或排列关系。

    Length generalization -- the ability to generalize to longer sequences than ones seen during training, and compositional generalization -- the ability to generalize to token combinations not seen during training, are crucial forms of out-of-distribution generalization in sequence-to-sequence models. In this work, we take the first steps towards provable length and compositional generalization for a range of architectures, including deep sets, transformers, state space models, and simple recurrent neural nets. Depending on the architecture, we prove different degrees of representation identification, e.g., a linear or a permutation relation with ground truth representation, is necessary for length and compositional generalization.
    
[^4]: 动态增量优化用于最佳子集选择

    Dynamic Incremental Optimization for Best Subset Selection

    [https://arxiv.org/abs/2402.02322](https://arxiv.org/abs/2402.02322)

    本文研究了一类$\ell_0$正则化问题的对偶形式，并提出了一种高效的原对偶算法，通过充分利用对偶范围估计和增量策略，提高了最佳子集选择问题的解决方案的效率和统计性质。

    

    最佳子集选择被认为是稀疏学习问题的“黄金标准”。已经提出了各种优化技术来攻击这个非光滑非凸问题。本文研究了一类$\ell_0$正则化问题的对偶形式。基于原始问题和对偶问题的结构，我们提出了一种高效的原对偶算法。通过充分利用对偶范围估计和增量策略，我们的算法潜在地减少了冗余计算并改进了最佳子集选择的解决方案。理论分析和对合成和真实数据集的实验验证了所提出解决方案的效率和统计性质。

    Best subset selection is considered the `gold standard' for many sparse learning problems. A variety of optimization techniques have been proposed to attack this non-smooth non-convex problem. In this paper, we investigate the dual forms of a family of $\ell_0$-regularized problems. An efficient primal-dual algorithm is developed based on the primal and dual problem structures. By leveraging the dual range estimation along with the incremental strategy, our algorithm potentially reduces redundant computation and improves the solutions of best subset selection. Theoretical analysis and experiments on synthetic and real-world datasets validate the efficiency and statistical properties of the proposed solutions.
    
[^5]: 保证非凸分解方法用于张量列车恢复的研究

    Guaranteed Nonconvex Factorization Approach for Tensor Train Recovery. (arXiv:2401.02592v1 [stat.ML])

    [http://arxiv.org/abs/2401.02592](http://arxiv.org/abs/2401.02592)

    本研究提出了一种保证非凸分解方法用于张量列车恢复的新方法。该方法通过优化左正交TT格式来实现正交结构，并使用黎曼梯度下降算法来优化因子。我们证明了该方法具有局部线性收敛性，并且在满足受限等谱性质的条件下能够以线性速率收敛到真实张量。

    

    在本文中，我们首次提供了对于分解方法的收敛性保证。具体而言，为了避免尺度歧义并便于理论分析，我们优化所谓的左正交TT格式，强制使大部分因子彼此正交。为了确保正交结构，我们利用黎曼梯度下降（RGD）来优化Stiefel流形上的这些因子。我们首先深入研究TT分解问题，并建立了RGD的局部线性收敛性。值得注意的是，随着张量阶数的增加，收敛速率仅经历线性下降。然后，我们研究了感知问题，即从线性测量中恢复TT格式张量。假设感知算子满足受限等谱性质（RIP），我们证明在适当的初始化下，通过谱初始化获得，RGD也会以线性速率收敛到真实张量。此外，我们扩展了我们的研究。

    In this paper, we provide the first convergence guarantee for the factorization approach. Specifically, to avoid the scaling ambiguity and to facilitate theoretical analysis, we optimize over the so-called left-orthogonal TT format which enforces orthonormality among most of the factors. To ensure the orthonormal structure, we utilize the Riemannian gradient descent (RGD) for optimizing those factors over the Stiefel manifold. We first delve into the TT factorization problem and establish the local linear convergence of RGD. Notably, the rate of convergence only experiences a linear decline as the tensor order increases. We then study the sensing problem that aims to recover a TT format tensor from linear measurements. Assuming the sensing operator satisfies the restricted isometry property (RIP), we show that with a proper initialization, which could be obtained through spectral initialization, RGD also converges to the ground-truth tensor at a linear rate. Furthermore, we expand our 
    
[^6]: 使用扩散模型提供的无限数据计划升级VAE训练

    Upgrading VAE Training With Unlimited Data Plans Provided by Diffusion Models. (arXiv:2310.19653v1 [stat.ML])

    [http://arxiv.org/abs/2310.19653](http://arxiv.org/abs/2310.19653)

    这项研究通过在预训练的扩散模型生成的样本上进行训练，有效减轻了VAE中编码器的过拟合问题。

    

    变分自编码器（VAE）是一种常用的表示学习模型，但其编码器容易过拟合，因为它们是在有限的训练集上进行训练，而不是真实（连续）数据分布$p_{\mathrm{data}}(\mathbf{x})$。与之相反，扩散模型通过固定编码器避免了这个问题。这使得它们的表示不太可解释，但简化了训练，可以精确和连续地逼近$p_{\mathrm{data}}(\mathbf{x})$。在本文中，我们展示了通过在预训练的扩散模型生成的样本上训练，可以有效减轻VAE中编码器的过拟合问题。这些结果有些出人意料，因为最近的研究发现，在使用另一个生成模型生成的数据上训练时，生成性能会下降。我们分析了使用我们的方法训练的VAE的泛化性能、分摊差距和鲁棒性。

    Variational autoencoders (VAEs) are popular models for representation learning but their encoders are susceptible to overfitting (Cremer et al., 2018) because they are trained on a finite training set instead of the true (continuous) data distribution $p_{\mathrm{data}}(\mathbf{x})$. Diffusion models, on the other hand, avoid this issue by keeping the encoder fixed. This makes their representations less interpretable, but it simplifies training, enabling accurate and continuous approximations of $p_{\mathrm{data}}(\mathbf{x})$. In this paper, we show that overfitting encoders in VAEs can be effectively mitigated by training on samples from a pre-trained diffusion model. These results are somewhat unexpected as recent findings (Alemohammad et al., 2023; Shumailov et al., 2023) observe a decay in generative performance when models are trained on data generated by another generative model. We analyze generalization performance, amortization gap, and robustness of VAEs trained with our pro
    
[^7]: 提升控制函数

    Boosted Control Functions. (arXiv:2310.05805v1 [stat.ML])

    [http://arxiv.org/abs/2310.05805](http://arxiv.org/abs/2310.05805)

    本研究通过建立同时方程模型和控制函数与分布概括的新连接，解决了在存在未观察到的混淆情况下，针对不同训练和测试分布的预测问题。

    

    现代机器学习方法和大规模数据的可用性为从大量的协变量中准确预测目标数量打开了大门。然而，现有的预测方法在训练和测试数据不同的情况下表现不佳，尤其是在存在隐藏混淆的情况下。虽然对因果效应估计（例如仪器变量）已经对隐藏混淆进行了深入研究，但对于预测任务来说并非如此。本研究旨在填补这一空白，解决在存在未观察到的混淆的情况下，针对不同训练和测试分布的预测问题。具体而言，我们在机器学习的分布概括领域，以及计量经济学中的同时方程模型和控制函数之间建立了一种新的联系。我们的贡献的核心是描述在一组分布转变下的数据生成过程的分布概括同时方程模型（SIMDGs）。

    Modern machine learning methods and the availability of large-scale data opened the door to accurately predict target quantities from large sets of covariates. However, existing prediction methods can perform poorly when the training and testing data are different, especially in the presence of hidden confounding. While hidden confounding is well studied for causal effect estimation (e.g., instrumental variables), this is not the case for prediction tasks. This work aims to bridge this gap by addressing predictions under different training and testing distributions in the presence of unobserved confounding. In particular, we establish a novel connection between the field of distribution generalization from machine learning, and simultaneous equation models and control function from econometrics. Central to our contribution are simultaneous equation models for distribution generalization (SIMDGs) which describe the data-generating process under a set of distributional shifts. Within thi
    
[^8]: 基于多源数据的分布鲁棒机器学习

    Distributionally Robust Machine Learning with Multi-source Data. (arXiv:2309.02211v1 [stat.ML])

    [http://arxiv.org/abs/2309.02211](http://arxiv.org/abs/2309.02211)

    本文提出了一种基于多源数据的分布鲁棒机器学习方法，通过引入组分布鲁棒预测模型来提高具有分布偏移的目标人群的预测准确性。

    

    当目标分布与源数据集不同时，传统的机器学习方法可能导致较差的预测性能。本文利用多个数据源，并引入了一种基于组分布鲁棒预测模型来优化关于目标分布类的可解释方差的对抗性奖励。与传统的经验风险最小化相比，所提出的鲁棒预测模型改善了具有分布偏移的目标人群的预测准确性。我们证明了组分布鲁棒预测模型是源数据集条件结果模型的加权平均。我们利用这一关键鉴别结果来提高任意机器学习算法的鲁棒性，包括随机森林和神经网络等。我们设计了一种新的偏差校正估计器来估计通用机器学习算法的最优聚合权重，并展示了其在c方面的改进。

    Classical machine learning methods may lead to poor prediction performance when the target distribution differs from the source populations. This paper utilizes data from multiple sources and introduces a group distributionally robust prediction model defined to optimize an adversarial reward about explained variance with respect to a class of target distributions. Compared to classical empirical risk minimization, the proposed robust prediction model improves the prediction accuracy for target populations with distribution shifts. We show that our group distributionally robust prediction model is a weighted average of the source populations' conditional outcome models. We leverage this key identification result to robustify arbitrary machine learning algorithms, including, for example, random forests and neural networks. We devise a novel bias-corrected estimator to estimate the optimal aggregation weight for general machine-learning algorithms and demonstrate its improvement in the c
    
[^9]: 非参数假设检验对分配群对称性的研究

    Non-parametric Hypothesis Tests for Distributional Group Symmetry. (arXiv:2307.15834v1 [stat.ME])

    [http://arxiv.org/abs/2307.15834](http://arxiv.org/abs/2307.15834)

    该论文提出了用于分布对称性的非参数假设检验方法，适用于具有对称性的数据集。具体而言，该方法在紧致群作用下测试边际或联合分布的不变性，并提出了一种易于实施的条件蒙特卡罗检验。

    

    对称性在科学、机器学习和统计学中起着重要的作用。对于已知遵循对称性的数据，已经开发出了许多利用对称性的方法。然而，对于普遍群对称性的存在或不存在的统计检验几乎不存在。本研究提出了一种非参数假设检验方法，基于单个独立同分布样本，用于针对特定群的分布对称性。我们提供了适用于两种广泛情况的对称性检验的一般公式。第一种情况是测试在紧致群作用下的边际或联合分布的不变性。在这里，一个渐近无偏的检验只需要一个可计算的概率分布空间上的度量和能够均匀随机采样群元素的能力。在此基础上，我们提出了一种易于实施的条件蒙特卡罗检验，并证明它可以实现精确的p值。

    Symmetry plays a central role in the sciences, machine learning, and statistics. For situations in which data are known to obey a symmetry, a multitude of methods that exploit symmetry have been developed. Statistical tests for the presence or absence of general group symmetry, however, are largely non-existent. This work formulates non-parametric hypothesis tests, based on a single independent and identically distributed sample, for distributional symmetry under a specified group. We provide a general formulation of tests for symmetry that apply to two broad settings. The first setting tests for the invariance of a marginal or joint distribution under the action of a compact group. Here, an asymptotically unbiased test only requires a computable metric on the space of probability distributions and the ability to sample uniformly random group elements. Building on this, we propose an easy-to-implement conditional Monte Carlo test and prove that it achieves exact $p$-values with finitel
    
[^10]: 基于强化学习的变分序列最优实验设计方法

    Variational Sequential Optimal Experimental Design using Reinforcement Learning. (arXiv:2306.10430v1 [stat.ML])

    [http://arxiv.org/abs/2306.10430](http://arxiv.org/abs/2306.10430)

    该研究提出了一种基于贝叶斯框架和信息增益效用的变分序列最优实验设计方法，通过强化学习求解最优设计策略，适用于多种OED问题，结果具有更高的样本效率和更少的前向模型模拟次数。

    

    我们引入了变分序列最优实验设计 (vsOED) 的新方法，通过贝叶斯框架和信息增益效用来最优地设计有限序列的实验。具体而言，我们通过变分近似贝叶斯后验的下界估计期望效用。通过同时最大化变分下界和执行策略梯度更新来数值解决最优设计策略。我们将这种方法应用于一系列面向参数推断、模型区分和目标导向预测的OED问题。这些案例涵盖了显式和隐式似然函数、麻烦参数和基于物理的偏微分方程模型。我们的vsOED结果表明，与以前的顺序设计算法相比，样本效率大大提高，所需前向模型模拟次数减少了。

    We introduce variational sequential Optimal Experimental Design (vsOED), a new method for optimally designing a finite sequence of experiments under a Bayesian framework and with information-gain utilities. Specifically, we adopt a lower bound estimator for the expected utility through variational approximation to the Bayesian posteriors. The optimal design policy is solved numerically by simultaneously maximizing the variational lower bound and performing policy gradient updates. We demonstrate this general methodology for a range of OED problems targeting parameter inference, model discrimination, and goal-oriented prediction. These cases encompass explicit and implicit likelihoods, nuisance parameters, and physics-based partial differential equation models. Our vsOED results indicate substantially improved sample efficiency and reduced number of forward model simulations compared to previous sequential design algorithms.
    
[^11]: 通过多元占位核函数学习高维非参数微分方程

    Learning High-Dimensional Nonparametric Differential Equations via Multivariate Occupation Kernel Functions. (arXiv:2306.10189v1 [stat.ML])

    [http://arxiv.org/abs/2306.10189](http://arxiv.org/abs/2306.10189)

    本论文提出了一种线性方法，通过多元占位核函数在高维状态空间中学习非参数ODE系统，可以解决显式公式按二次方缩放的问题。这种方法在高度非线性的数据和图像数据中都具有通用性。

    

    从$d$维状态空间中$n$个轨迹快照中学习非参数的常微分方程（ODE）系统需要学习$d$个函数。除非具有额外的系统属性知识，例如稀疏性和对称性，否则显式的公式按二次方缩放。在这项工作中，我们提出了一种使用向量值再生核希尔伯特空间提供的隐式公式学习的线性方法。通过将ODE重写为更弱的积分形式，我们随后进行最小化并推导出我们的学习算法。最小化问题的解向量场依赖于与解轨迹相关的多元占位核函数。我们通过对高度非线性的模拟和真实数据进行实验证实了我们的方法，其中$d$可能超过100。我们进一步通过从图像数据学习非参数一阶拟线性偏微分方程展示了所提出的方法的多样性。

    Learning a nonparametric system of ordinary differential equations (ODEs) from $n$ trajectory snapshots in a $d$-dimensional state space requires learning $d$ functions of $d$ variables. Explicit formulations scale quadratically in $d$ unless additional knowledge about system properties, such as sparsity and symmetries, is available. In this work, we propose a linear approach to learning using the implicit formulation provided by vector-valued Reproducing Kernel Hilbert Spaces. By rewriting the ODEs in a weaker integral form, which we subsequently minimize, we derive our learning algorithm. The minimization problem's solution for the vector field relies on multivariate occupation kernel functions associated with the solution trajectories. We validate our approach through experiments on highly nonlinear simulated and real data, where $d$ may exceed 100. We further demonstrate the versatility of the proposed method by learning a nonparametric first order quasilinear partial differential 
    
[^12]: 基函数编码改善因子机中数字特征的准确性

    Basis Function Encoding of Numerical Features in Factorization Machines for Improved Accuracy. (arXiv:2305.14528v1 [cs.LG])

    [http://arxiv.org/abs/2305.14528](http://arxiv.org/abs/2305.14528)

    本文提供了一种能够将数字特征编码为基函数向量的方法，通过在因子机中将该方法应用于因子机中，可以改善推荐系统的准确性。

    

    因子机(FM)变体被广泛用于大规模实时内容推荐系统，因为它们在模型准确性和训练推理的低计算成本之间提供了出色的平衡。本文提供了一种系统、理论上合理的方法，通过将数值特征编码为所选函数集的函数值向量将数值特征纳入FM变体。

    Factorization machine (FM) variants are widely used for large scale real-time content recommendation systems, since they offer an excellent balance between model accuracy and low computational costs for training and inference. These systems are trained on tabular data with both numerical and categorical columns. Incorporating numerical columns poses a challenge, and they are typically incorporated using a scalar transformation or binning, which can be either learned or chosen a-priori. In this work, we provide a systematic and theoretically-justified way to incorporate numerical features into FM variants by encoding them into a vector of function values for a set of functions of one's choice.  We view factorization machines as approximators of segmentized functions, namely, functions from a field's value to the real numbers, assuming the remaining fields are assigned some given constants, which we refer to as the segment. From this perspective, we show that our technique yields a model
    
[^13]: 两层神经网络上SGD的全局收敛性证明

    Global Convergence of SGD On Two Layer Neural Nets. (arXiv:2210.11452v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.11452](http://arxiv.org/abs/2210.11452)

    该论文证明了当深度为2的神经网络采用足够平滑凸的激活函数时，SGD可以收敛到全局最小值，证明过程中引入Frobenius范数正则化与恰当分布的参数初始化，同时拓展了连续时间的收敛结果。

    

    在这篇论文中，我们证明了当深度为2的网络采用足够平滑且有边界的激活函数（比如sigmoid和tanh）时，SGD可以证明性地收敛到适当正则化的$\ell_2-$经验风险的全局最小值--对于任意数据和任意数量的门。我们在[1]的研究成果上进行了扩展，并在权重上添加了恒定量的Frobenius范数正则化，同时选取了恰当的分布对初始权重进行采样。我们还给出了一个连续时间的SGD收敛结果，同样适用于如SoftPlus这样的平滑无边界的激活函数。我们的关键想法是展示了存在于固定大小的神经网络上的损失函数，它们是“Villani函数”[1] Bin Shi, Weijie J. Su, and Michael I. Jordan. On learning rates and schr\"odinger operators, 2020. arXiv:2004.06977

    In this note we demonstrate provable convergence of SGD to the global minima of appropriately regularized $\ell_2-$empirical risk of depth $2$ nets -- for arbitrary data and with any number of gates, if they are using adequately smooth and bounded activations like sigmoid and tanh. We build on the results in [1] and leverage a constant amount of Frobenius norm regularization on the weights, along with sampling of the initial weights from an appropriate distribution. We also give a continuous time SGD convergence result that also applies to smooth unbounded activations like SoftPlus. Our key idea is to show the existence loss functions on constant sized neural nets which are "Villani Functions". [1] Bin Shi, Weijie J. Su, and Michael I. Jordan. On learning rates and schr\"odinger operators, 2020. arXiv:2004.06977
    
[^14]: 通过概率-概率映射实现有条件校准的预测分布：在银河红移估计和概率预测中的应用

    Conditionally Calibrated Predictive Distributions by Probability-Probability Map: Application to Galaxy Redshift Estimation and Probabilistic Forecasting. (arXiv:2205.14568v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2205.14568](http://arxiv.org/abs/2205.14568)

    本研究提出了一种名为Cal-PIT的方法，通过学习一个概率-概率映射，解决了预测分布的诊断和校准问题，来实现有条件校准。

    

    不确定性量化对于评估AI算法的预测能力至关重要。过去的研究致力于描述目标变量$y \in \mathbb{R}$在给定复杂输入特征$\mathbf{x} \in \mathcal{X}$的条件下的预测分布$F(y|\mathbf{x})$。然而，现有的预测分布（例如，归一化流和贝叶斯神经网络）往往缺乏条件校准，即给定输入$\mathbf{x}$的事件发生的概率与预测概率显著不同。当前的校准方法不能完全评估和实施有条件校准的预测分布。在这里，我们提出了一种名为Cal-PIT的方法，它通过从校准数据中学习一个概率-概率映射来同时解决预测分布的诊断和校准问题。关键思想是对概率积分变换分数进行$\mathbf{x}$的回归。估计的回归提供了对特征空间中条件覆盖的可解释诊断。

    Uncertainty quantification is crucial for assessing the predictive ability of AI algorithms. Much research has been devoted to describing the predictive distribution (PD) $F(y|\mathbf{x})$ of a target variable $y \in \mathbb{R}$ given complex input features $\mathbf{x} \in \mathcal{X}$. However, off-the-shelf PDs (from, e.g., normalizing flows and Bayesian neural networks) often lack conditional calibration with the probability of occurrence of an event given input $\mathbf{x}$ being significantly different from the predicted probability. Current calibration methods do not fully assess and enforce conditionally calibrated PDs. Here we propose \texttt{Cal-PIT}, a method that addresses both PD diagnostics and recalibration by learning a single probability-probability map from calibration data. The key idea is to regress probability integral transform scores against $\mathbf{x}$. The estimated regression provides interpretable diagnostics of conditional coverage across the feature space. 
    
[^15]: 用模拟退火机器学习贝叶斯网络

    Learning Bayesian Networks with Annealing Machine. (arXiv:2006.06926v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2006.06926](http://arxiv.org/abs/2006.06926)

    本文提出了一种用于贝叶斯网络结构学习的模拟退火机器方法，通过先进的候选父节点集合的确定和分解，以及整数规划问题的解决，能够在比特容量有限的情况下高效地解决基于评分的学习问题。

    

    最近的研究表明，模拟退火机器能够高精度地解决组合优化问题。模拟退火机器有潜力用于基于评分的贝叶斯网络结构学习。然而，模拟退火机器的比特容量目前有限。为了利用模拟退火技术，需要将基于评分的学习问题转化为在比特容量内的二次无约束二元优化问题。在本文中，我们提出了一种高效的转化方法，通过先进的候选父节点集合的确定和其分解。我们还提供了一个整数规划问题，以找到最小化所需比特数的分解。在包含变量从75到223的7个基准数据集上的实验结果表明，我们的方法所需的比特数比四代富士通数字退火器（一种采用半导体技术开发的全耦合模拟退火机器）的100K比特容量少。

    Recent studies have reported that annealing machines are capable of solving combinatorial optimization problems with high accuracy. Annealing machines can potentially be applied to score-based Bayesian network structure learning. However, the bit capacity of an annealing machine is currently limited. To utilize the annealing technology, converting score-based learning problems into quadratic unconstrained binary optimizations within the bit capacity is necessary. In this paper, we propose an efficient conversion method with the advanced identification of candidate parent sets and their decomposition. We also provide an integer programming problem to find the decomposition that minimizes the number of required bits. Experimental results on $7$ benchmark datasets with variables from $75$ to $223$ show that our approach requires less bits than the $100$K bit capacity of the fourth-generation Fujitsu Digital Annealer, a fully coupled annealing machine developed with semiconductor technolog
    

