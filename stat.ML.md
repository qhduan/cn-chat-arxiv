# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Fallacy of Minimizing Local Regret in the Sequential Task Setting](https://arxiv.org/abs/2403.10946) | 研究了强化学习中在序列任务设置下最小化局部遗憾的谬误，揭示了近视地最小化遗憾在实际应用中的复杂性。 |
| [^2] | [Gradient-based Discrete Sampling with Automatic Cyclical Scheduling](https://arxiv.org/abs/2402.17699) | 提出了一种使用自动循环调度的基于梯度的离散采样方法，有效应对高度多模态的离散分布，包括循环步长调度、循环平衡调度和自动调整超参数方案。 |
| [^3] | [Imputation using training labels and classification via label imputation.](http://arxiv.org/abs/2311.16877) | 本论文提出一种在填充缺失数据时将标签与输入堆叠的方法，能够显著提高填充效果，并同时填充标签和输入。该方法适用于各种类型的数据，且在实验证明具有有希望的准确性结果。 |
| [^4] | [Under-Parameterized Double Descent for Ridge Regularized Least Squares Denoising of Data on a Line.](http://arxiv.org/abs/2305.14689) | 本文研究了线性数据最小二乘岭正则化的去噪问题，证明了在欠参数化情况下会出现双峰谷现象。 |

# 详细

[^1]: 在序列任务设置中最小化局部遗憾的谬误

    The Fallacy of Minimizing Local Regret in the Sequential Task Setting

    [https://arxiv.org/abs/2403.10946](https://arxiv.org/abs/2403.10946)

    研究了强化学习中在序列任务设置下最小化局部遗憾的谬误，揭示了近视地最小化遗憾在实际应用中的复杂性。

    

    在强化学习领域，在线强化学习经常被概念化为一个优化问题，其中算法与未知环境交互以最小化累积遗憾。在静态设置中，可以获得强大的理论保证，如次线性（$\sqrt{T}$）遗憾界限，通常意味着收敛到最优策略并停止探索。然而，这些理论设置通常过分简化了真实世界强化学习实现中遇到的复杂性，其中任务按顺序到达，任务之间有重大变化，并且算法可能不允许在某些任务中进行自适应学习。我们研究超出结果分布的变化，涵盖奖励设计（从结果到奖励的映射）和允许的策略空间的变化。我们的结果揭示了在每个任务中近视地最小化遗憾的谬误：获得最优遗憾r

    arXiv:2403.10946v1 Announce Type: cross  Abstract: In the realm of Reinforcement Learning (RL), online RL is often conceptualized as an optimization problem, where an algorithm interacts with an unknown environment to minimize cumulative regret. In a stationary setting, strong theoretical guarantees, like a sublinear ($\sqrt{T}$) regret bound, can be obtained, which typically implies the convergence to an optimal policy and the cessation of exploration. However, these theoretical setups often oversimplify the complexities encountered in real-world RL implementations, where tasks arrive sequentially with substantial changes between tasks and the algorithm may not be allowed to adaptively learn within certain tasks. We study the changes beyond the outcome distributions, encompassing changes in the reward designs (mappings from outcomes to rewards) and the permissible policy spaces. Our results reveal the fallacy of myopically minimizing regret within each task: obtaining optimal regret r
    
[^2]: 使用自动循环调度的基于梯度的离散采样

    Gradient-based Discrete Sampling with Automatic Cyclical Scheduling

    [https://arxiv.org/abs/2402.17699](https://arxiv.org/abs/2402.17699)

    提出了一种使用自动循环调度的基于梯度的离散采样方法，有效应对高度多模态的离散分布，包括循环步长调度、循环平衡调度和自动调整超参数方案。

    

    离散分布，特别是在高维深度模型中，通常由于固有的不连续性而呈现高度多模态。虽然基于梯度的离散采样已被证明是有效的，但由于梯度信息，它容易陷入局部模式。为了解决这一挑战，我们提出了一种自动循环调度，旨在实现对多模态离散分布进行高效准确的采样。我们的方法包括三个关键部分：（1）循环步长调度，其中大步长发现新模式，小步长利用每个模式；（2）循环平衡调度，确保给定步长的“平衡”提案和马尔可夫链的高效率；以及（3）自动调整方案，用于调整循环调度中的超参数，实现在各种数据集上的自适应性且需最小调整。我们证明了我们的方法的非渐近收敛和推断保证。

    arXiv:2402.17699v1 Announce Type: new  Abstract: Discrete distributions, particularly in high-dimensional deep models, are often highly multimodal due to inherent discontinuities. While gradient-based discrete sampling has proven effective, it is susceptible to becoming trapped in local modes due to the gradient information. To tackle this challenge, we propose an automatic cyclical scheduling, designed for efficient and accurate sampling in multimodal discrete distributions. Our method contains three key components: (1) a cyclical step size schedule where large steps discover new modes and small steps exploit each mode; (2) a cyclical balancing schedule, ensuring ``balanced" proposals for given step sizes and high efficiency of the Markov chain; and (3) an automatic tuning scheme for adjusting the hyperparameters in the cyclical schedules, allowing adaptability across diverse datasets with minimal tuning. We prove the non-asymptotic convergence and inference guarantee for our method i
    
[^3]: 使用训练标签进行填充和通过标签填充进行分类

    Imputation using training labels and classification via label imputation. (arXiv:2311.16877v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.16877](http://arxiv.org/abs/2311.16877)

    本论文提出一种在填充缺失数据时将标签与输入堆叠的方法，能够显著提高填充效果，并同时填充标签和输入。该方法适用于各种类型的数据，且在实验证明具有有希望的准确性结果。

    

    在实际应用中，缺失数据是一个常见的问题。已经开发了各种填充方法来处理缺失数据。然而，尽管训练数据通常都有标签，但常见的填充方法通常只依赖于输入而忽略标签。在这项工作中，我们阐述了将标签堆叠到输入中可以显着提高输入的填充效果。此外，我们提出了一种分类策略，该策略将预测的测试标签初始化为缺失值，并将标签与输入堆叠在一起进行填充。这样可以同时填充标签和输入。而且，该技术能够处理具有缺失标签的训练数据，无需任何先前的填充，并且适用于连续型、分类型或混合型数据。实验证明在准确性方面取得了有希望的结果。

    Missing data is a common problem in practical settings. Various imputation methods have been developed to deal with missing data. However, even though the label is usually available in the training data, the common practice of imputation usually only relies on the input and ignores the label. In this work, we illustrate how stacking the label into the input can significantly improve the imputation of the input. In addition, we propose a classification strategy that initializes the predicted test label with missing values and stacks the label with the input for imputation. This allows imputing the label and the input at the same time. Also, the technique is capable of handling data training with missing labels without any prior imputation and is applicable to continuous, categorical, or mixed-type data. Experiments show promising results in terms of accuracy.
    
[^4]: 基于岭正则化的线性数据最小二乘去噪问题的欠参数化双谷效应

    Under-Parameterized Double Descent for Ridge Regularized Least Squares Denoising of Data on a Line. (arXiv:2305.14689v1 [stat.ML])

    [http://arxiv.org/abs/2305.14689](http://arxiv.org/abs/2305.14689)

    本文研究了线性数据最小二乘岭正则化的去噪问题，证明了在欠参数化情况下会出现双峰谷现象。

    

    研究了训练数据点数、统计模型参数数和模型的泛化能力之间的关系。已有的工作表明，过度参数化情况下可能出现双峰谷现象，而在欠参数化情况下则普遍存在标准偏差-方差权衡。本文提出了一个简单的例子，可以证明欠参数化情况下可以发生双峰谷现象。考虑嵌入高维空间中的线性数据最小二乘去噪问题中的岭正则化，通过推导出一种渐近准确的广义误差公式，我们发现了样本和参数的双谷效应，双峰谷位于插值点和过度参数化区域之间。此外，样本双谷曲线的高峰对应于估计量的范数曲线的高峰。

    The relationship between the number of training data points, the number of parameters in a statistical model, and the generalization capabilities of the model has been widely studied. Previous work has shown that double descent can occur in the over-parameterized regime, and believe that the standard bias-variance trade-off holds in the under-parameterized regime. In this paper, we present a simple example that provably exhibits double descent in the under-parameterized regime. For simplicity, we look at the ridge regularized least squares denoising problem with data on a line embedded in high-dimension space. By deriving an asymptotically accurate formula for the generalization error, we observe sample-wise and parameter-wise double descent with the peak in the under-parameterized regime rather than at the interpolation point or in the over-parameterized regime.  Further, the peak of the sample-wise double descent curve corresponds to a peak in the curve for the norm of the estimator,
    

