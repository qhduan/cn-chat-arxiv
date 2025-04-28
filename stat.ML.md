# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prior-Dependent Allocations for Bayesian Fixed-Budget Best-Arm Identification in Structured Bandits](https://arxiv.org/abs/2402.05878) | 本论文研究了在结构化赌博机中的贝叶斯固定预算最佳臂识别问题，提出了一种基于先验信息的固定分配算法，并引入了新的证明方法，以得到更紧密的多臂BAI界限。该方法在各种情况下展现出一致且稳健的性能，加深了我们对于该问题的理解。 |
| [^2] | [A Bias-Variance Decomposition for Ensembles over Multiple Synthetic Datasets](https://arxiv.org/abs/2402.03985) | 本研究通过对多个合成数据集进行偏差-方差分解，增加了对其理论理解。实验证明多个合成数据集对于高方差的下游预测器特别有益，并提供了一个简单的经验法则用于选择适当的合成数据集数量。 |
| [^3] | [Convergence of Momentum-Based Heavy Ball Method with Batch Updating and/or Approximate Gradients.](http://arxiv.org/abs/2303.16241) | 本文研究了含有批量更新和/或近似梯度的动量重球法的收敛性，由于采用了简化的梯度计算方法，大大减少了计算消耗，同时仍能保证收敛性。 |

# 详细

[^1]: 基于先验依赖分配的结构化赌博机中贝叶斯固定预算最佳臂识别

    Prior-Dependent Allocations for Bayesian Fixed-Budget Best-Arm Identification in Structured Bandits

    [https://arxiv.org/abs/2402.05878](https://arxiv.org/abs/2402.05878)

    本论文研究了在结构化赌博机中的贝叶斯固定预算最佳臂识别问题，提出了一种基于先验信息的固定分配算法，并引入了新的证明方法，以得到更紧密的多臂BAI界限。该方法在各种情况下展现出一致且稳健的性能，加深了我们对于该问题的理解。

    

    我们研究了在结构化赌博机中的贝叶斯固定预算最佳臂识别（BAI）问题。我们提出了一种算法，该算法基于先验信息和环境结构使用固定分配。我们在多个模型中提供了它在性能上的理论界限，包括线性和分层BAI的首个先验依赖上界。我们的主要贡献是引入了新的证明方法，相比现有方法，它能得到更紧密的多臂BAI界限。我们广泛比较了我们的方法与其他固定预算BAI方法，在各种设置中展示了其一致且稳健的性能。我们的工作改进了对于结构化赌博机中贝叶斯固定预算BAI的理解，并突出了我们的方法在实际场景中的有效性。

    We study the problem of Bayesian fixed-budget best-arm identification (BAI) in structured bandits. We propose an algorithm that uses fixed allocations based on the prior information and the structure of the environment. We provide theoretical bounds on its performance across diverse models, including the first prior-dependent upper bounds for linear and hierarchical BAI. Our key contribution is introducing new proof methods that result in tighter bounds for multi-armed BAI compared to existing methods. We extensively compare our approach to other fixed-budget BAI methods, demonstrating its consistent and robust performance in various settings. Our work improves our understanding of Bayesian fixed-budget BAI in structured bandits and highlights the effectiveness of our approach in practical scenarios.
    
[^2]: 对多个合成数据集的集成进行偏差-方差分解

    A Bias-Variance Decomposition for Ensembles over Multiple Synthetic Datasets

    [https://arxiv.org/abs/2402.03985](https://arxiv.org/abs/2402.03985)

    本研究通过对多个合成数据集进行偏差-方差分解，增加了对其理论理解。实验证明多个合成数据集对于高方差的下游预测器特别有益，并提供了一个简单的经验法则用于选择适当的合成数据集数量。

    

    最近的研究强调了为监督学习生成多个合成数据集的好处，包括增加准确性、更有效的模型选择和不确定性估计。这些好处在经验上有明确的支持，但对它们的理论理解目前非常有限。我们通过推导使用多个合成数据集的几种设置的偏差-方差分解，来增加理论理解。我们的理论预测，对于高方差的下游预测器，多个合成数据集将特别有益，并为均方误差和Brier分数的情况提供了一个简单的经验法则来选择合适的合成数据集数量。我们通过评估一个集成在多个合成数据集和几个真实数据集以及下游预测器上的性能来研究我们的理论在实践中的效果。结果验证了我们的理论，表明我们的洞察也在实践中具有相关性。

    Recent studies have highlighted the benefits of generating multiple synthetic datasets for supervised learning, from increased accuracy to more effective model selection and uncertainty estimation. These benefits have clear empirical support, but the theoretical understanding of them is currently very light. We seek to increase the theoretical understanding by deriving bias-variance decompositions for several settings of using multiple synthetic datasets. Our theory predicts multiple synthetic datasets to be especially beneficial for high-variance downstream predictors, and yields a simple rule of thumb to select the appropriate number of synthetic datasets in the case of mean-squared error and Brier score. We investigate how our theory works in practice by evaluating the performance of an ensemble over many synthetic datasets for several real datasets and downstream predictors. The results follow our theory, showing that our insights are also practically relevant.
    
[^3]: 采用批量更新和/或近似梯度的动量重球法的收敛性

    Convergence of Momentum-Based Heavy Ball Method with Batch Updating and/or Approximate Gradients. (arXiv:2303.16241v1 [math.OC])

    [http://arxiv.org/abs/2303.16241](http://arxiv.org/abs/2303.16241)

    本文研究了含有批量更新和/或近似梯度的动量重球法的收敛性，由于采用了简化的梯度计算方法，大大减少了计算消耗，同时仍能保证收敛性。

    

    本文研究了1964年Polyak引入的凸优化和非凸优化中广为人知的“动量重球”法，并在多种情况下确立了其收敛性。当要求解参数的维度非常高时，更新一部分而不是所有参数可以提高优化效率，称之为“批量更新”，若与梯度法配合使用，则理论上只需计算需要更新的参数的梯度，而在实际中，通过反向传播等方法仅计算部分梯度并不能减少计算量。因此，为了在每一步中减少CPU使用量，可以使用一阶微分或近似梯度代替真实梯度。我们的分析表明，在各种假设下，采用近似梯度信息和/或批量更新的动量重球法仍然可以收敛。

    In this paper, we study the well-known "Heavy Ball" method for convex and nonconvex optimization introduced by Polyak in 1964, and establish its convergence under a variety of situations. Traditionally, most algorthms use "full-coordinate update," that is, at each step, very component of the argument is updated. However, when the dimension of the argument is very high, it is more efficient to update some but not all components of the argument at each iteration. We refer to this as "batch updating" in this paper.  When gradient-based algorithms are used together with batch updating, in principle it is sufficient to compute only those components of the gradient for which the argument is to be updated. However, if a method such as back propagation is used to compute these components, computing only some components of gradient does not offer much savings over computing the entire gradient. Therefore, to achieve a noticeable reduction in CPU usage at each step, one can use first-order diffe
    

