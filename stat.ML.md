# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks](https://arxiv.org/abs/2402.19460) | 该论文评估了在ImageNet上的多种不确定性估计器，发现虽然有理论努力，但实践中尚未实现不确定性的解开，同时揭示了哪些估计器在特定任务上表现出色，为从业者提供见解并指导未来研究。 |
| [^2] | [Approximate information maximization for bandit games.](http://arxiv.org/abs/2310.12563) | 本论文提出了一种基于近似信息最大化的强盗游戏算法，通过最大化关键变量的信息近似值来进行优化，在传统强盗设置中表现出很强的性能，并证明了其对于两臂强盗问题的渐近最优性。 |
| [^3] | [On Consistency of Signatures Using Lasso.](http://arxiv.org/abs/2305.10413) | 本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。 |

# 详细

[^1]: 为标准化的任务专门指定不确定性量化：专门的不确定性用于专门的任务

    Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks

    [https://arxiv.org/abs/2402.19460](https://arxiv.org/abs/2402.19460)

    该论文评估了在ImageNet上的多种不确定性估计器，发现虽然有理论努力，但实践中尚未实现不确定性的解开，同时揭示了哪些估计器在特定任务上表现出色，为从业者提供见解并指导未来研究。

    

    不确定性量化，曾经是一个独立的任务，已经发展成为一个包含预测抑制、越界检测以及随机不确定性量化在内的任务谱系。最新的目标是解开不确定性：构建多个估计器，每个都专门定制于一个特定任务。因此，最近有大量不同意图的最新进展——这些往往完全偏离实际行为。本文在ImageNet上对多种不确定性估计器进行全面评估。我们发现，尽管有着颇有希望的理论努力，实践中仍未实现解开。此外，我们揭示了哪些不确定性估计器在哪些特定任务上表现出色，为从业者提供见解并引导未来研究朝着基于任务和解开的不确定性估计方法发展。我们的代码可在 https://github.com/bmucsanyi/bud 找到。

    arXiv:2402.19460v1 Announce Type: new  Abstract: Uncertainty quantification, once a singular task, has evolved into a spectrum of tasks, including abstained prediction, out-of-distribution detection, and aleatoric uncertainty quantification. The latest goal is disentanglement: the construction of multiple estimators that are each tailored to one and only one task. Hence, there is a plethora of recent advances with different intentions - that often entirely deviate from practical behavior. This paper conducts a comprehensive evaluation of numerous uncertainty estimators across diverse tasks on ImageNet. We find that, despite promising theoretical endeavors, disentanglement is not yet achieved in practice. Additionally, we reveal which uncertainty estimators excel at which specific tasks, providing insights for practitioners and guiding future research toward task-centric and disentangled uncertainty estimation methods. Our code is available at https://github.com/bmucsanyi/bud.
    
[^2]: 适用于强盗游戏的近似信息最大化方法

    Approximate information maximization for bandit games. (arXiv:2310.12563v1 [stat.ML])

    [http://arxiv.org/abs/2310.12563](http://arxiv.org/abs/2310.12563)

    本论文提出了一种基于近似信息最大化的强盗游戏算法，通过最大化关键变量的信息近似值来进行优化，在传统强盗设置中表现出很强的性能，并证明了其对于两臂强盗问题的渐近最优性。

    

    熵最大化和自由能最小化是用于模拟各种物理系统动态的一般物理原理。其中包括使用自由能原理对大脑内的决策进行建模，使用信息瓶颈原理对访问隐藏变量时优化准确性和复杂性的权衡，以及使用信息最大化进行随机环境导航。基于这一原理，我们提出了一种新的强盗算法类别，通过最大化系统中一个关键变量的信息近似来进行优化。为此，我们开发了一个基于物理的近似分析熵的表示方法，以预测每个动作的信息增益，并贪婪地选择信息增益最大的动作。这种方法在传统强盗设置中表现出很强的性能。受到其经验性成功的启发，我们证明了其对于两臂强盗问题的渐近最优性。

    Entropy maximization and free energy minimization are general physical principles for modeling the dynamics of various physical systems. Notable examples include modeling decision-making within the brain using the free-energy principle, optimizing the accuracy-complexity trade-off when accessing hidden variables with the information bottleneck principle (Tishby et al., 2000), and navigation in random environments using information maximization (Vergassola et al., 2007). Built on this principle, we propose a new class of bandit algorithms that maximize an approximation to the information of a key variable within the system. To this end, we develop an approximated analytical physics-based representation of an entropy to forecast the information gain of each action and greedily choose the one with the largest information gain. This method yields strong performances in classical bandit settings. Motivated by its empirical success, we prove its asymptotic optimality for the two-armed bandit
    
[^3]: 使用Lasso的签名一致性研究

    On Consistency of Signatures Using Lasso. (arXiv:2305.10413v1 [stat.ML])

    [http://arxiv.org/abs/2305.10413](http://arxiv.org/abs/2305.10413)

    本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。

    

    签名变换是连续和离散时间序列数据的迭代路径积分，它们的普遍非线性通过线性化特征选择问题。本文在理论和数值上重新审视了Lasso回归对于签名变换的一致性问题。我们的研究表明，对于更接近布朗运动或具有较弱跨维度相关性的过程和时间序列，签名定义为It\^o积分的Lasso回归更具一致性；对于均值回归过程和时间序列，其签名定义为Stratonovich积分在Lasso回归中具有更高的一致性。我们的发现强调了在统计推断和机器学习中选择适当的签名和随机模型的重要性。

    Signature transforms are iterated path integrals of continuous and discrete-time time series data, and their universal nonlinearity linearizes the problem of feature selection. This paper revisits the consistency issue of Lasso regression for the signature transform, both theoretically and numerically. Our study shows that, for processes and time series that are closer to Brownian motion or random walk with weaker inter-dimensional correlations, the Lasso regression is more consistent for their signatures defined by It\^o integrals; for mean reverting processes and time series, their signatures defined by Stratonovich integrals have more consistency in the Lasso regression. Our findings highlight the importance of choosing appropriate definitions of signatures and stochastic models in statistical inference and machine learning.
    

