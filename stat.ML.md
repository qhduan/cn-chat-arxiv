# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Variance, Admissibility, and Stability of Empirical Risk Minimization.](http://arxiv.org/abs/2305.18508) | 本文指出，对于使用平方损失函数的经验风险最小化(ERM)，其次优性必须归因于大的偏差而非方差，并且在ERM的平方误差的偏差-方差分解中，方差项必然具有极小的失误率。作者还提供了Chatterjee的不可允许性定理的简单证明，并表示他们的估计表明ERM的稳定性。 |

# 详细

[^1]: 论经验风险最小化的方差、可允许性和稳定性

    On the Variance, Admissibility, and Stability of Empirical Risk Minimization. (arXiv:2305.18508v1 [math.ST])

    [http://arxiv.org/abs/2305.18508](http://arxiv.org/abs/2305.18508)

    本文指出，对于使用平方损失函数的经验风险最小化(ERM)，其次优性必须归因于大的偏差而非方差，并且在ERM的平方误差的偏差-方差分解中，方差项必然具有极小的失误率。作者还提供了Chatterjee的不可允许性定理的简单证明，并表示他们的估计表明ERM的稳定性。

    

    众所周知，使用平方损失的经验风险最小化可能会达到极小的最大失误率。本文的关键信息是，在温和的假设下，ERM的次优性必须归因于大的偏差而非方差。在ERM的平方误差的偏差-方差分解中，方差项必然具有极小的失误率。我们为固定设计提供了一个简单的、使用概率方法证明这一事实的证明。然后，我们在随机设计设置下为各种模型证明了这一结果。此外，我们提供了 Chatterjee 不可允许性定理 (Chatterjee, 2014, Theorem 1.4) 的简单证明，该定理指出，在固定设计设置中，ERM不能被排除为一种最优方法，并将该结果扩展到随机设计设置。我们还表明，我们的估计表明ERM的稳定性，为Caponnetto和Rakhlin(2006)的非Donsker类的主要结果提供了补充。

    It is well known that Empirical Risk Minimization (ERM) with squared loss may attain minimax suboptimal error rates (Birg\'e and Massart, 1993). The key message of this paper is that, under mild assumptions, the suboptimality of ERM must be due to large bias rather than variance. More precisely, in the bias-variance decomposition of the squared error of the ERM, the variance term necessarily enjoys the minimax rate. In the case of fixed design, we provide an elementary proof of this fact using the probabilistic method. Then, we prove this result for various models in the random design setting. In addition, we provide a simple proof of Chatterjee's admissibility theorem (Chatterjee, 2014, Theorem 1.4), which states that ERM cannot be ruled out as an optimal method, in the fixed design setting, and extend this result to the random design setting. We also show that our estimates imply stability of ERM, complementing the main result of Caponnetto and Rakhlin (2006) for non-Donsker classes.
    

