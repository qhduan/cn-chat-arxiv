# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-dimensional Contextual Bandit Problem without Sparsity.](http://arxiv.org/abs/2306.11017) | 本论文研究了高维情境赌博问题，无需施加稀疏性要求，并提出了一种探索-开发算法以解决此问题。研究表明，可以通过平衡探索和开发实现最优速率。同时，还介绍了一种自适应探索-开发算法来找到最优平衡点。 |

# 详细

[^1]: 无稀疏性的高维情境赌博问题研究

    High-dimensional Contextual Bandit Problem without Sparsity. (arXiv:2306.11017v1 [stat.ML])

    [http://arxiv.org/abs/2306.11017](http://arxiv.org/abs/2306.11017)

    本论文研究了高维情境赌博问题，无需施加稀疏性要求，并提出了一种探索-开发算法以解决此问题。研究表明，可以通过平衡探索和开发实现最优速率。同时，还介绍了一种自适应探索-开发算法来找到最优平衡点。

    

    本研究探讨了高维线性情境赌博问题，其中特征数 $p$ 大于预算 $T$ 或甚至无限制。与此领域的大部分研究不同的是，我们不对回归系数施加稀疏性要求。相反，我们依靠最近关于过参数化模型的研究成果，从而能够在数据分布具有较小有效秩时分析最小范数插值估计器的性能。我们提出了一个探索-开发 (EtC) 算法来解决这个问题，并检验了它的性能。通过我们的分析，我们以 $T$ 为变量，导出了ETC算法的最优速率，并表明这个速率可以通过平衡探索和开发来实现。此外，我们介绍了一种自适应探索-开发 (AEtC)算法，它可以自适应地找到最优平衡点。我们通过一系列模拟评估了所提出算法的性能。

    In this research, we investigate the high-dimensional linear contextual bandit problem where the number of features $p$ is greater than the budget $T$, or it may even be infinite. Differing from the majority of previous works in this field, we do not impose sparsity on the regression coefficients. Instead, we rely on recent findings on overparameterized models, which enables us to analyze the performance the minimum-norm interpolating estimator when data distributions have small effective ranks. We propose an explore-then-commit (EtC) algorithm to address this problem and examine its performance. Through our analysis, we derive the optimal rate of the ETC algorithm in terms of $T$ and show that this rate can be achieved by balancing exploration and exploitation. Moreover, we introduce an adaptive explore-then-commit (AEtC) algorithm that adaptively finds the optimal balance. We assess the performance of the proposed algorithms through a series of simulations.
    

