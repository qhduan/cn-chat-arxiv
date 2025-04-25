# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PARMESAN: Parameter-Free Memory Search and Transduction for Dense Prediction Tasks](https://arxiv.org/abs/2403.11743) | 通过引入转导的概念，提出了PARMESAN，一种用于解决密集预测任务的无参数内存搜索和转导方法，实现了灵活性和无需连续训练的学习。 |
| [^2] | [Hessian-Free Laplace in Bayesian Deep Learning](https://arxiv.org/abs/2403.10671) | 提出了一种无Hessian计算和求逆的Hessian-Free Laplace近似框架，通过对数后验和网络预测的曲率来估计后验的方差。 |
| [^3] | [Rao-Blackwellising Bayesian Causal Inference](https://arxiv.org/abs/2402.14781) | 本文结合顺序化的MCMC结构学习技术和梯度图学习的最新进展，构建了一个有效的贝叶斯因果推断框架，将因果结构推断问题分解为变量拓扑顺序推断和变量父节点集合推断，同时使用高斯过程进行因果机制建模实现精确边缘化，引入了一个Rao-Blackwell化方案。 |
| [^4] | [Precise Error Rates for Computationally Efficient Testing.](http://arxiv.org/abs/2311.00289) | 在高维设置中，我们提出了一种基于线性谱统计的测试方法，该方法在计算上非常高效，并且在所有计算上高效的测试中，实现了类型 I 和类型 II 错误率之间的最佳权衡曲线。 |
| [^5] | [Efficient Neural Network Approaches for Conditional Optimal Transport with Applications in Bayesian Inference.](http://arxiv.org/abs/2310.16975) | 提出了两种神经网络方法来逼近静态和动态条件最优传输问题的解，实现了对条件概率分布的采样和密度估计，适用于贝叶斯推断。算法利用神经网络参数化传输映射以提高可扩展性。 |

# 详细

[^1]: PARMESAN: 用于密集预测任务的无参数内存搜索与转导

    PARMESAN: Parameter-Free Memory Search and Transduction for Dense Prediction Tasks

    [https://arxiv.org/abs/2403.11743](https://arxiv.org/abs/2403.11743)

    通过引入转导的概念，提出了PARMESAN，一种用于解决密集预测任务的无参数内存搜索和转导方法，实现了灵活性和无需连续训练的学习。

    

    在这项工作中，我们通过转导推理来解决深度学习中的灵活性问题。我们提出了PARMESAN（无参数内存搜索与转导），这是一种可扩展的转导方法，利用内存模块来解决密集预测任务。在推断过程中，内存中的隐藏表示被搜索以找到相应的示例。与其他方法不同，PARMESAN通过修改内存内容学习，而无需进行任何连续训练或微调可学习参数。我们的方法与常用的神经结构兼容。

    arXiv:2403.11743v1 Announce Type: new  Abstract: In this work we address flexibility in deep learning by means of transductive reasoning. For adaptation to new tasks or new data, existing methods typically involve tuning of learnable parameters or even complete re-training from scratch, rendering such approaches unflexible in practice. We argue that the notion of separating computation from memory by the means of transduction can act as a stepping stone for solving these issues. We therefore propose PARMESAN (parameter-free memory search and transduction), a scalable transduction method which leverages a memory module for solving dense prediction tasks. At inference, hidden representations in memory are being searched to find corresponding examples. In contrast to other methods, PARMESAN learns without the requirement for any continuous training or fine-tuning of learnable parameters simply by modifying the memory content. Our method is compatible with commonly used neural architecture
    
[^2]: Bayesian深度学习中的无Hessian-Laplace

    Hessian-Free Laplace in Bayesian Deep Learning

    [https://arxiv.org/abs/2403.10671](https://arxiv.org/abs/2403.10671)

    提出了一种无Hessian计算和求逆的Hessian-Free Laplace近似框架，通过对数后验和网络预测的曲率来估计后验的方差。

    

    贝叶斯后验的Laplace近似（LA）是以最大后验估计为中心的高斯分布。它在贝叶斯深度学习中的吸引力源于能够在标准网络参数优化之后量化不确定性（即事后），从近似后验中抽样的便利性以及模型证据的解析形式。然而，LA的一个重要计算瓶颈是必须计算和求逆对数后验的Hessian矩阵。Hessian可以以多种方式近似，质量与网络、数据集和推断任务等多个因素有关。在本文中，我们提出了一个绕过Hessian计算和求逆的替代框架。无Hessian-Laplace（HFL）近似使用对数后验和网络预测的曲率来估计其方差。只需要两个点估计：最大后验估计和等价的曲率方向。

    arXiv:2403.10671v1 Announce Type: cross  Abstract: The Laplace approximation (LA) of the Bayesian posterior is a Gaussian distribution centered at the maximum a posteriori estimate. Its appeal in Bayesian deep learning stems from the ability to quantify uncertainty post-hoc (i.e., after standard network parameter optimization), the ease of sampling from the approximate posterior, and the analytic form of model evidence. However, an important computational bottleneck of LA is the necessary step of calculating and inverting the Hessian matrix of the log posterior. The Hessian may be approximated in a variety of ways, with quality varying with a number of factors including the network, dataset, and inference task. In this paper, we propose an alternative framework that sidesteps Hessian calculation and inversion. The Hessian-free Laplace (HFL) approximation uses curvature of both the log posterior and network prediction to estimate its variance. Only two point estimates are needed: the st
    
[^3]: Rao-Blackwellising Bayesian Causal Inference

    Rao-Blackwellising Bayesian Causal Inference

    [https://arxiv.org/abs/2402.14781](https://arxiv.org/abs/2402.14781)

    本文结合顺序化的MCMC结构学习技术和梯度图学习的最新进展，构建了一个有效的贝叶斯因果推断框架，将因果结构推断问题分解为变量拓扑顺序推断和变量父节点集合推断，同时使用高斯过程进行因果机制建模实现精确边缘化，引入了一个Rao-Blackwell化方案。

    

    贝叶斯因果推断，即推断用于下游因果推理任务中的因果模型的后验概率，构成了一个在文献中鲜有探讨的难解的计算推断问题。本文将基于顺序的MCMC结构学习技术与最近梯度图学习的进展相结合，构建了一个有效的贝叶斯因果推断框架。具体而言，我们将推断因果结构的问题分解为(i)推断变量之间的拓扑顺序以及(ii)推断每个变量的父节点集合。当限制每个变量的父节点数量时，我们可以在多项式时间内完全边缘化父节点集合。我们进一步使用高斯过程来建模未知的因果机制，从而允许其精确边缘化。这引入了一个Rao-Blackwell化方案，其中除了因果顺序之外，模型中的所有组件都被消除。

    arXiv:2402.14781v1 Announce Type: cross  Abstract: Bayesian causal inference, i.e., inferring a posterior over causal models for the use in downstream causal reasoning tasks, poses a hard computational inference problem that is little explored in literature. In this work, we combine techniques from order-based MCMC structure learning with recent advances in gradient-based graph learning into an effective Bayesian causal inference framework. Specifically, we decompose the problem of inferring the causal structure into (i) inferring a topological order over variables and (ii) inferring the parent sets for each variable. When limiting the number of parents per variable, we can exactly marginalise over the parent sets in polynomial time. We further use Gaussian processes to model the unknown causal mechanisms, which also allows their exact marginalisation. This introduces a Rao-Blackwellization scheme, where all components are eliminated from the model, except for the causal order, for whi
    
[^4]: 高效测试的精确错误率

    Precise Error Rates for Computationally Efficient Testing. (arXiv:2311.00289v1 [math.ST])

    [http://arxiv.org/abs/2311.00289](http://arxiv.org/abs/2311.00289)

    在高维设置中，我们提出了一种基于线性谱统计的测试方法，该方法在计算上非常高效，并且在所有计算上高效的测试中，实现了类型 I 和类型 II 错误率之间的最佳权衡曲线。

    

    我们重新审视了简单与简单假设检验的基本问题，特别关注计算复杂度，因为在高维设置中，统计上最优的似然比检验通常是计算上难以处理的。在经典的尖峰维格纳模型（具有一般性 i.i.d. 尖峰先验）中，我们展示了一个基于线性谱统计的现有测试实现了在计算上高效测试之间的最佳权衡曲线，即使存在更好的指数时间测试。这个结果是在一个适当复杂性理论的猜想条件下得到的，即一个自然加强已经建立的低次数猜想。我们的结果表明，谱是计算受限的测试的充分统计量（但不是所有测试的充分统计量）。据我们所知，我们的方法提供了首个用于推理关于有效计算所能实现的精确渐近测试误差的工具。

    We revisit the fundamental question of simple-versus-simple hypothesis testing with an eye towards computational complexity, as the statistically optimal likelihood ratio test is often computationally intractable in high-dimensional settings. In the classical spiked Wigner model (with a general i.i.d. spike prior) we show that an existing test based on linear spectral statistics achieves the best possible tradeoff curve between type I and type II error rates among all computationally efficient tests, even though there are exponential-time tests that do better. This result is conditional on an appropriate complexity-theoretic conjecture, namely a natural strengthening of the well-established low-degree conjecture. Our result shows that the spectrum is a sufficient statistic for computationally bounded tests (but not for all tests).  To our knowledge, our approach gives the first tool for reasoning about the precise asymptotic testing error achievable with efficient computation. The main
    
[^5]: 条件最优传输的高效神经网络方法及贝叶斯推断中的应用

    Efficient Neural Network Approaches for Conditional Optimal Transport with Applications in Bayesian Inference. (arXiv:2310.16975v1 [stat.ML])

    [http://arxiv.org/abs/2310.16975](http://arxiv.org/abs/2310.16975)

    提出了两种神经网络方法来逼近静态和动态条件最优传输问题的解，实现了对条件概率分布的采样和密度估计，适用于贝叶斯推断。算法利用神经网络参数化传输映射以提高可扩展性。

    

    我们提出了两种神经网络方法，分别逼近静态和动态条件最优传输问题的解。这两种方法可以对条件概率分布进行采样和密度估计，这是贝叶斯推断中的核心任务。我们的方法将目标条件分布表示为可处理的参考分布的转换，因此属于测度传输的框架。在该框架中，COT映射是一个典型的选择，具有唯一性和单调性等可取的属性。然而，相关的COT问题在中等维度下计算具有挑战性。为了提高可扩展性，我们的数值算法利用神经网络对COT映射进行参数化。我们的方法充分利用了COT问题的静态和动态表达形式的结构。PCP-Map将条件传输映射建模为部分输入凸神经网络（PICNN）的梯度。

    We present two neural network approaches that approximate the solutions of static and dynamic conditional optimal transport (COT) problems, respectively. Both approaches enable sampling and density estimation of conditional probability distributions, which are core tasks in Bayesian inference. Our methods represent the target conditional distributions as transformations of a tractable reference distribution and, therefore, fall into the framework of measure transport. COT maps are a canonical choice within this framework, with desirable properties such as uniqueness and monotonicity. However, the associated COT problems are computationally challenging, even in moderate dimensions. To improve the scalability, our numerical algorithms leverage neural networks to parameterize COT maps. Our methods exploit the structure of the static and dynamic formulations of the COT problem. PCP-Map models conditional transport maps as the gradient of a partially input convex neural network (PICNN) and 
    

