# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Asymmetric matrix sensing by gradient descent with small random initialization.](http://arxiv.org/abs/2309.01796) | 本论文研究了矩阵感知问题，通过小的随机初始化应用因式梯度下降算法来重建低秩矩阵。特别地，引入了一个连续微分方程，称为“扰动梯度流”，并证明了在扰动被限制在一定范围内时，扰动梯度流能够快速收敛到真实的目标矩阵。 |

# 详细

[^1]: 用小的随机初始化梯度下降进行非对称矩阵感知

    Asymmetric matrix sensing by gradient descent with small random initialization. (arXiv:2309.01796v1 [cs.LG])

    [http://arxiv.org/abs/2309.01796](http://arxiv.org/abs/2309.01796)

    本论文研究了矩阵感知问题，通过小的随机初始化应用因式梯度下降算法来重建低秩矩阵。特别地，引入了一个连续微分方程，称为“扰动梯度流”，并证明了在扰动被限制在一定范围内时，扰动梯度流能够快速收敛到真实的目标矩阵。

    

    我们研究了矩阵感知，即从少量线性测量中重建低秩矩阵的问题。它可以被形式化为一个过参数化回归问题，可以通过因式分解的梯度下降解决，当从一个小的随机初始化开始。线性神经网络，特别是通过因式梯度下降进行矩阵感知，作为现代机器学习中非凸问题的典型模型，可以将复杂现象解开并详细研究。许多研究致力于研究非对称矩阵感知的特殊情况，例如非对称矩阵因式分解和对称半正定矩阵感知。我们的关键贡献是引入了一个连续微分方程，我们称之为“扰动梯度流”。我们证明了当扰动被限制在足够范围内时，扰动梯度流能够快速收敛到真实的目标矩阵。梯度下降对矩阵的动态

    We study matrix sensing, which is the problem of reconstructing a low-rank matrix from a few linear measurements. It can be formulated as an overparameterized regression problem, which can be solved by factorized gradient descent when starting from a small random initialization.  Linear neural networks, and in particular matrix sensing by factorized gradient descent, serve as prototypical models of non-convex problems in modern machine learning, where complex phenomena can be disentangled and studied in detail. Much research has been devoted to studying special cases of asymmetric matrix sensing, such as asymmetric matrix factorization and symmetric positive semi-definite matrix sensing.  Our key contribution is introducing a continuous differential equation that we call the $\textit{perturbed gradient flow}$. We prove that the perturbed gradient flow converges quickly to the true target matrix whenever the perturbation is sufficiently bounded. The dynamics of gradient descent for matr
    

