# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Double Cross-fit Doubly Robust Estimators: Beyond Series Regression](https://arxiv.org/abs/2403.15175) | 双交叉固定双稳健估计器针对因果推断中的预期条件协方差进行了研究，通过拆分训练数据并在独立样本上下调nuisance函数估计器，结构无关的错误分析以及更强假设的结果，提出了更精确的DCDR估计器。 |
| [^2] | [Functional Partial Least-Squares: Optimal Rates and Adaptation](https://arxiv.org/abs/2402.11134) | 该论文提出了一种新的函数偏最小二乘估计器，其在一类椭球上实现了（近乎）最优的收敛速率，并引入了适应未知逆问题度的提前停止规则。 |
| [^3] | [Mori-Zwanzig latent space Koopman closure for nonlinear autoencoder.](http://arxiv.org/abs/2310.10745) | 本研究提出了一种名为Mori-Zwanzig自编码器（MZ-AE）的新方法，用于在低维空间中稳健地逼近Koopman算子，通过非线性自编码器和Mori-Zwanzig形式主义的集成实现对有限不变Koopman子空间的逼近，从而增强了精确性和准确预测复杂系统行为的能力。 |
| [^4] | [Optimal Low-Rank Matrix Completion: Semidefinite Relaxations and Eigenvector Disjunctions.](http://arxiv.org/abs/2305.12292) | 该论文通过重新表述低秩矩阵填补问题为投影矩阵的非凸问题，实现了能够确定最优解的分离分支定界方案，并且通过新颖和紧密的凸松弛方法，使得最优性差距相对于现有方法减少了两个数量级。 |

# 详细

[^1]: 双交叉固定双稳健估计器：超越串行回归

    Double Cross-fit Doubly Robust Estimators: Beyond Series Regression

    [https://arxiv.org/abs/2403.15175](https://arxiv.org/abs/2403.15175)

    双交叉固定双稳健估计器针对因果推断中的预期条件协方差进行了研究，通过拆分训练数据并在独立样本上下调nuisance函数估计器，结构无关的错误分析以及更强假设的结果，提出了更精确的DCDR估计器。

    

    具有跨拟合交叉的双稳健估计器因其良好的结构无关错误保证而在因果推断中备受青睐。然而，当存在额外结构，例如H\"{o}lder平滑时，可以通过在独立样本上对训练数据进行拆分和下调nuisance函数估计器来构建更精确的“双交叉固定双稳健”（DCDR）估计器。我们研究了预期条件协方差的DCDR估计器，在因果推断和条件独立性检验中是一个感兴趣的函数，并得出了一系列逐渐更强假设的结果。首先，我们对DCDR估计器提供无需对nuisance函数或它们的估计器做出假设的结构无关错误分析。然后，假设nuisance函数是H\"{o}lder平滑，但不假设知晓真实平滑级别或协变量密度。

    arXiv:2403.15175v1 Announce Type: cross  Abstract: Doubly robust estimators with cross-fitting have gained popularity in causal inference due to their favorable structure-agnostic error guarantees. However, when additional structure, such as H\"{o}lder smoothness, is available then more accurate "double cross-fit doubly robust" (DCDR) estimators can be constructed by splitting the training data and undersmoothing nuisance function estimators on independent samples. We study a DCDR estimator of the Expected Conditional Covariance, a functional of interest in causal inference and conditional independence testing, and derive a series of increasingly powerful results with progressively stronger assumptions. We first provide a structure-agnostic error analysis for the DCDR estimator with no assumptions on the nuisance functions or their estimators. Then, assuming the nuisance functions are H\"{o}lder smooth, but without assuming knowledge of the true smoothness level or the covariate densit
    
[^2]: 函数偏最小二乘法：最优收敛率和自适应性

    Functional Partial Least-Squares: Optimal Rates and Adaptation

    [https://arxiv.org/abs/2402.11134](https://arxiv.org/abs/2402.11134)

    该论文提出了一种新的函数偏最小二乘估计器，其在一类椭球上实现了（近乎）最优的收敛速率，并引入了适应未知逆问题度的提前停止规则。

    

    我们考虑具有标量响应和 Hilbert 空间值预测变量的函数线性回归模型，这是一个众所周知的反问题。我们提出了一个与共轭梯度方法相关的函数偏最小二乘（PLS）估计的新公式。我们将展示该估计器在一类椭球上实现了（近乎）最优的收敛速率，并引入了一个能够适应未知逆问题度的提前停止规则。我们提供了估计器与主成分回归估计器之间的一些理论和仿真比较。

    arXiv:2402.11134v1 Announce Type: cross  Abstract: We consider the functional linear regression model with a scalar response and a Hilbert space-valued predictor, a well-known ill-posed inverse problem. We propose a new formulation of the functional partial least-squares (PLS) estimator related to the conjugate gradient method. We shall show that the estimator achieves the (nearly) optimal convergence rate on a class of ellipsoids and we introduce an early stopping rule which adapts to the unknown degree of ill-posedness. Some theoretical and simulation comparison between the estimator and the principal component regression estimator is provided.
    
[^3]: Mori-Zwanzig潜变空间Koopman闭包用于非线性自编码器

    Mori-Zwanzig latent space Koopman closure for nonlinear autoencoder. (arXiv:2310.10745v1 [cs.LG])

    [http://arxiv.org/abs/2310.10745](http://arxiv.org/abs/2310.10745)

    本研究提出了一种名为Mori-Zwanzig自编码器（MZ-AE）的新方法，用于在低维空间中稳健地逼近Koopman算子，通过非线性自编码器和Mori-Zwanzig形式主义的集成实现对有限不变Koopman子空间的逼近，从而增强了精确性和准确预测复杂系统行为的能力。

    

    Koopman算子提供了一种吸引人的方法来实现非线性系统的全局线性化，使其成为简化复杂动力学理解的宝贵方法。虽然数据驱动的方法在逼近有限Koopman算子方面表现出了潜力，但它们面临着各种挑战，例如选择合适的可观察量、降维和准确预测复杂系统行为的能力。本研究提出了一种名为Mori-Zwanzig自编码器（MZ-AE）的新方法，用于在低维空间中稳健地逼近Koopman算子。所提出的方法利用非线性自编码器提取关键可观察量来逼近有限不变Koopman子空间，并利用Mori-Zwanzig形式主义集成非马尔可夫校正机制。因此，该方法在非线性自编码器的潜变流形中产生了动力学的封闭表示，从而提高了精确性和...

    The Koopman operator presents an attractive approach to achieve global linearization of nonlinear systems, making it a valuable method for simplifying the understanding of complex dynamics. While data-driven methodologies have exhibited promise in approximating finite Koopman operators, they grapple with various challenges, such as the judicious selection of observables, dimensionality reduction, and the ability to predict complex system behaviours accurately. This study presents a novel approach termed Mori-Zwanzig autoencoder (MZ-AE) to robustly approximate the Koopman operator in low-dimensional spaces. The proposed method leverages a nonlinear autoencoder to extract key observables for approximating a finite invariant Koopman subspace and integrates a non-Markovian correction mechanism using the Mori-Zwanzig formalism. Consequently, this approach yields a closed representation of dynamics within the latent manifold of the nonlinear autoencoder, thereby enhancing the precision and s
    
[^4]: 最优低秩矩阵填补：半定松弛和特征向量分离

    Optimal Low-Rank Matrix Completion: Semidefinite Relaxations and Eigenvector Disjunctions. (arXiv:2305.12292v1 [cs.LG])

    [http://arxiv.org/abs/2305.12292](http://arxiv.org/abs/2305.12292)

    该论文通过重新表述低秩矩阵填补问题为投影矩阵的非凸问题，实现了能够确定最优解的分离分支定界方案，并且通过新颖和紧密的凸松弛方法，使得最优性差距相对于现有方法减少了两个数量级。

    

    低秩矩阵填补的目的是计算一个复杂度最小的矩阵，以尽可能准确地恢复给定的一组观测数据，并且具有众多应用，如产品推荐。不幸的是，现有的解决低秩矩阵填补的方法是启发式的，虽然高度可扩展并且通常能够确定高质量的解决方案，但不具备任何最优性保证。我们通过将低秩问题重新表述为投影矩阵的非凸问题，并实现一种分离分支定界方案来重新审视矩阵填补问题，以实现最优性导向。此外，我们通过将低秩矩阵分解为一组秩一矩阵的和，并通过 Shor 松弛来激励每个秩一矩阵中的每个 2*2 小矩阵的行列式为零，从而推导出一种新颖且通常很紧的凸松弛类。在数值实验中，相对于最先进的启发式方法，我们的新凸松弛方法将最优性差距减少了两个数量级。

    Low-rank matrix completion consists of computing a matrix of minimal complexity that recovers a given set of observations as accurately as possible, and has numerous applications such as product recommendation. Unfortunately, existing methods for solving low-rank matrix completion are heuristics that, while highly scalable and often identifying high-quality solutions, do not possess any optimality guarantees. We reexamine matrix completion with an optimality-oriented eye, by reformulating low-rank problems as convex problems over the non-convex set of projection matrices and implementing a disjunctive branch-and-bound scheme that solves them to certifiable optimality. Further, we derive a novel and often tight class of convex relaxations by decomposing a low-rank matrix as a sum of rank-one matrices and incentivizing, via a Shor relaxation, that each two-by-two minor in each rank-one matrix has determinant zero. In numerical experiments, our new convex relaxations decrease the optimali
    

