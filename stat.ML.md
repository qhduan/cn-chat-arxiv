# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Sample Complexity of Simple Binary Hypothesis Testing](https://arxiv.org/abs/2403.16981) | 该论文导出了一个公式，用于刻画简单二元假设检验的样本复杂度（乘法常数独立于$p$、$q$和所有错误参数），适用于不同的设置条件。 |
| [^2] | [Stochastic Hessian Fitting on Lie Group](https://arxiv.org/abs/2402.11858) | 本文研究了在随机Hessian-向量乘积上拟合Hessian或其逆，揭示了不同Hessian拟合方法的收敛速率，并证明了在特定李群上的Hessian拟合问题在轻微条件下是强凸的。 |

# 详细

[^1]: 简单二元假设检验的样本复杂度

    The Sample Complexity of Simple Binary Hypothesis Testing

    [https://arxiv.org/abs/2403.16981](https://arxiv.org/abs/2403.16981)

    该论文导出了一个公式，用于刻画简单二元假设检验的样本复杂度（乘法常数独立于$p$、$q$和所有错误参数），适用于不同的设置条件。

    

    简单的二元假设检验的样本复杂度是区分两个分布$p$和$q$所需的最小独立同分布样本数量，可以通过以下方式之一进行：(i) 无先验设置，类型-I错误最大为$\alpha$，类型-II错误最大为$\beta$; 或者 (ii) 贝叶斯设置，贝叶斯错误最大为$\delta$，先验分布为$(\alpha, 1-\alpha)$。 迄今为止，只在$\alpha = \beta$（无先验）或$\alpha = 1/2$（贝叶斯）时研究了此问题，并且已知样本复杂度可以用$p$和$q$之间的Hellinger散度来刻画，直到乘法常数。 在本文中，我们导出了一个公式，用来刻画样本复杂度（乘法常数独立于$p$、$q$和所有错误参数），适用于：(i) 先验设置中所有$0 \le \alpha, \beta \le 1/8$；以及 (ii) 贝叶斯设置中所有$\delta \le \alpha/4$。 特别地，该公式适用于

    arXiv:2403.16981v1 Announce Type: cross  Abstract: The sample complexity of simple binary hypothesis testing is the smallest number of i.i.d. samples required to distinguish between two distributions $p$ and $q$ in either: (i) the prior-free setting, with type-I error at most $\alpha$ and type-II error at most $\beta$; or (ii) the Bayesian setting, with Bayes error at most $\delta$ and prior distribution $(\alpha, 1-\alpha)$. This problem has only been studied when $\alpha = \beta$ (prior-free) or $\alpha = 1/2$ (Bayesian), and the sample complexity is known to be characterized by the Hellinger divergence between $p$ and $q$, up to multiplicative constants. In this paper, we derive a formula that characterizes the sample complexity (up to multiplicative constants that are independent of $p$, $q$, and all error parameters) for: (i) all $0 \le \alpha, \beta \le 1/8$ in the prior-free setting; and (ii) all $\delta \le \alpha/4$ in the Bayesian setting. In particular, the formula admits eq
    
[^2]: 在李群上的随机Hessian拟合

    Stochastic Hessian Fitting on Lie Group

    [https://arxiv.org/abs/2402.11858](https://arxiv.org/abs/2402.11858)

    本文研究了在随机Hessian-向量乘积上拟合Hessian或其逆，揭示了不同Hessian拟合方法的收敛速率，并证明了在特定李群上的Hessian拟合问题在轻微条件下是强凸的。

    

    本文研究了在随机Hessian-向量乘积上拟合Hessian或其逆。使用了一个Hessian拟合准则，可用于推导大部分常用方法，如BFGS、高斯牛顿、AdaGrad等。我们的研究揭示了不同Hessian拟合方法的不同收敛速率，例如，在欧几里德空间中的梯度下降的次线性速率和对称正定（SPL）矩阵和某些李群上的梯度下降的线性速率。在特定且足够一般的李群上的Hessian拟合问题在轻微条件下被证明是强凸的。为了确认我们的分析，这些方法在不同设置下进行了测试，如有噪声的Hessian-向量乘积、时变的Hessians和低精度算术。这些发现对依赖于随机二阶优化的方法是有用的。

    arXiv:2402.11858v1 Announce Type: cross  Abstract: This paper studies the fitting of Hessian or its inverse with stochastic Hessian-vector products. A Hessian fitting criterion, which can be used to derive most of the commonly used methods, e.g., BFGS, Gaussian-Newton, AdaGrad, etc., is used for the analysis. Our studies reveal different convergence rates for different Hessian fitting methods, e.g., sublinear rates for gradient descent in the Euclidean space and a commonly used closed-form solution, linear rates for gradient descent on the manifold of symmetric positive definite (SPL) matrices and certain Lie groups. The Hessian fitting problem is further shown to be strongly convex under mild conditions on a specific yet general enough Lie group. To confirm our analysis, these methods are tested under different settings like noisy Hessian-vector products, time varying Hessians, and low precision arithmetic. These findings are useful for stochastic second order optimizations that rely 
    

