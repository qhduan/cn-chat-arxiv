# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistical Inference of Optimal Allocations I: Regularities and their Implications](https://arxiv.org/abs/2403.18248) | 这项研究提出了一种函数可微方法来解决统计最优分配问题，通过对排序运算符的一般属性进行详细分析，推导出值函数的Hadamard可微性，并展示了如何利用函数偏微分法直接推导出值函数过程的渐近性质。 |
| [^2] | [Variational DAG Estimation via State Augmentation With Stochastic Permutations](https://arxiv.org/abs/2402.02644) | 使用状态扩展和随机排列进行变分DAG估计的方法可以超越竞争的贝叶斯和非贝叶斯基准方法，从而在估计贝叶斯网络结构方面取得更好的性能。 |
| [^3] | [Kernel Learning in Ridge Regression "Automatically" Yields Exact Low Rank Solution.](http://arxiv.org/abs/2310.11736) | 该论文研究了核岭回归问题中核学习的低秩解的性质。在只有低维子空间对响应变量有解释能力的情况下，通过该方法可以自动得到精确的低秩解，无需额外的正则化。 |
| [^4] | [Learning with Subset Stacking.](http://arxiv.org/abs/2112.06251) | 提出了一种新的回归算法LESS，通过生成以随机点为中心的子集并训练局部预测器，然后以新颖的方式组合预测器得到整体预测器。在多个数据集上测试表明，LESS是一种有竞争力且高效的监督学习方法。 |

# 详细

[^1]: 统计推断中的最优分配I：规律性及其影响

    Statistical Inference of Optimal Allocations I: Regularities and their Implications

    [https://arxiv.org/abs/2403.18248](https://arxiv.org/abs/2403.18248)

    这项研究提出了一种函数可微方法来解决统计最优分配问题，通过对排序运算符的一般属性进行详细分析，推导出值函数的Hadamard可微性，并展示了如何利用函数偏微分法直接推导出值函数过程的渐近性质。

    

    在这篇论文中，我们提出了一种用于解决统计最优分配问题的函数可微方法。通过对排序运算符的一般属性进行详细分析，我们首先推导出了值函数的Hadamard可微性。在我们的框架中，Hausdorff测度的概念以及几何测度论中的面积和共面积积分公式是核心。基于我们的Hadamard可微性结果，我们展示了如何利用函数偏微分法直接推导出二元约束最优分配问题的值函数过程以及两步ROC曲线估计量的渐近性质。此外，利用对凸和局部Lipschitz泛函的深刻见解，我们得到了最优分配问题的值函数的额外一般Frechet可微性结果。这些引人入胜的发现激励了我们

    arXiv:2403.18248v1 Announce Type: new  Abstract: In this paper, we develp a functional differentiability approach for solving statistical optimal allocation problems. We first derive Hadamard differentiability of the value function through a detailed analysis of the general properties of the sorting operator. Central to our framework are the concept of Hausdorff measure and the area and coarea integration formulas from geometric measure theory. Building on our Hadamard differentiability results, we demonstrate how the functional delta method can be used to directly derive the asymptotic properties of the value function process for binary constrained optimal allocation problems, as well as the two-step ROC curve estimator. Moreover, leveraging profound insights from geometric functional analysis on convex and local Lipschitz functionals, we obtain additional generic Fr\'echet differentiability results for the value functions of optimal allocation problems. These compelling findings moti
    
[^2]: 通过状态扩展和随机排列的方法进行变分DAG估计

    Variational DAG Estimation via State Augmentation With Stochastic Permutations

    [https://arxiv.org/abs/2402.02644](https://arxiv.org/abs/2402.02644)

    使用状态扩展和随机排列进行变分DAG估计的方法可以超越竞争的贝叶斯和非贝叶斯基准方法，从而在估计贝叶斯网络结构方面取得更好的性能。

    

    从观测数据中估计贝叶斯网络的结构，即有向无环图（DAG），是一个在统计和计算上都很困难的问题，在因果发现等领域有着重要应用。贝叶斯方法在解决这个任务方面是一个有希望的方向，因为它们允许进行不确定性量化，并处理众所周知的可识别性问题。从概率推断的角度来看，主要的挑战是（i）表示满足DAG约束的图的分布和（ii）估计底层组合空间的后验概率。我们提出了一种方法，通过在DAG和排列的扩展空间上构建联合分布来解决这些挑战。我们通过变分推断进行后验估计，在其中利用了离散分布的连续松弛。我们展示了我们的方法在一系列合成和实际数据上能够超越竞争的贝叶斯和非贝叶斯基准方法。

    Estimating the structure of a Bayesian network, in the form of a directed acyclic graph (DAG), from observational data is a statistically and computationally hard problem with essential applications in areas such as causal discovery. Bayesian approaches are a promising direction for solving this task, as they allow for uncertainty quantification and deal with well-known identifiability issues. From a probabilistic inference perspective, the main challenges are (i) representing distributions over graphs that satisfy the DAG constraint and (ii) estimating a posterior over the underlying combinatorial space. We propose an approach that addresses these challenges by formulating a joint distribution on an augmented space of DAGs and permutations. We carry out posterior estimation via variational inference, where we exploit continuous relaxations of discrete distributions. We show that our approach can outperform competitive Bayesian and non-Bayesian benchmarks on a range of synthetic and re
    
[^3]: 在Ridge回归中，核学习“自动”给出精确的低秩解

    Kernel Learning in Ridge Regression "Automatically" Yields Exact Low Rank Solution. (arXiv:2310.11736v1 [math.ST])

    [http://arxiv.org/abs/2310.11736](http://arxiv.org/abs/2310.11736)

    该论文研究了核岭回归问题中核学习的低秩解的性质。在只有低维子空间对响应变量有解释能力的情况下，通过该方法可以自动得到精确的低秩解，无需额外的正则化。

    

    我们考虑形式为$(x,x') \mapsto \phi(\|x-x'\|^2_\Sigma)$且由参数$\Sigma$参数化的核函数。对于这样的核函数，我们研究了核岭回归问题的变体，它同时优化了预测函数和再现核希尔伯特空间的参数$\Sigma$。从这个核岭回归问题中学到的$\Sigma$的特征空间可以告诉我们协变量空间中哪些方向对预测是重要的。假设协变量只通过低维子空间（中心均值子空间）对响应变量有非零的解释能力，我们发现有很高的概率下有限样本核学习目标的全局最小化者也是低秩的。更具体地说，最小化$\Sigma$的秩有很高的概率被中心均值子空间的维度所限制。这个现象很有趣，因为低秩特性是在没有使用任何对$\Sigma$的显式正则化的情况下实现的，例如核范数正则化等。

    We consider kernels of the form $(x,x') \mapsto \phi(\|x-x'\|^2_\Sigma)$ parametrized by $\Sigma$. For such kernels, we study a variant of the kernel ridge regression problem which simultaneously optimizes the prediction function and the parameter $\Sigma$ of the reproducing kernel Hilbert space. The eigenspace of the $\Sigma$ learned from this kernel ridge regression problem can inform us which directions in covariate space are important for prediction.  Assuming that the covariates have nonzero explanatory power for the response only through a low dimensional subspace (central mean subspace), we find that the global minimizer of the finite sample kernel learning objective is also low rank with high probability. More precisely, the rank of the minimizing $\Sigma$ is with high probability bounded by the dimension of the central mean subspace. This phenomenon is interesting because the low rankness property is achieved without using any explicit regularization of $\Sigma$, e.g., nuclear
    
[^4]: 学习与子集叠加

    Learning with Subset Stacking. (arXiv:2112.06251v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2112.06251](http://arxiv.org/abs/2112.06251)

    提出了一种新的回归算法LESS，通过生成以随机点为中心的子集并训练局部预测器，然后以新颖的方式组合预测器得到整体预测器。在多个数据集上测试表明，LESS是一种有竞争力且高效的监督学习方法。

    

    我们提出了一种新的回归算法，该算法从一组输入-输出对中进行学习。我们的算法适用于输入变量与输出变量之间的关系在预测空间中表现出异质行为的群体。该算法首先生成以输入空间中的随机点为中心的子集，然后为每个子集训练一个局部预测器。然后这些预测器以一种新颖的方式组合在一起，形成一个整体预测器。我们将此算法称为“学习与子集叠加”或LESS，因为它类似于叠加回归器的方法。我们将LESS与多个数据集上的最先进方法进行测试性能比较。我们的比较结果表明，LESS是一种有竞争力的监督学习方法。此外，我们观察到LESS在计算时间上也非常高效，并且可以直接进行并行实现。

    We propose a new regression algorithm that learns from a set of input-output pairs. Our algorithm is designed for populations where the relation between the input variables and the output variable exhibits a heterogeneous behavior across the predictor space. The algorithm starts with generating subsets that are concentrated around random points in the input space. This is followed by training a local predictor for each subset. Those predictors are then combined in a novel way to yield an overall predictor. We call this algorithm ``LEarning with Subset Stacking'' or LESS, due to its resemblance to the method of stacking regressors. We compare the testing performance of LESS with state-of-the-art methods on several datasets. Our comparison shows that LESS is a competitive supervised learning method. Moreover, we observe that LESS is also efficient in terms of computation time and it allows a straightforward parallel implementation.
    

