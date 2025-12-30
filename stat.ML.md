# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Revisiting the Last-Iterate Convergence of Stochastic Gradient Methods](https://arxiv.org/abs/2312.08531) | 研究了随机梯度方法的最终迭代收敛性，并提出了不需要限制性假设的最优收敛速率问题。 |

# 详细

[^1]: 重新审视随机梯度方法的最终迭代收敛性

    Revisiting the Last-Iterate Convergence of Stochastic Gradient Methods

    [https://arxiv.org/abs/2312.08531](https://arxiv.org/abs/2312.08531)

    研究了随机梯度方法的最终迭代收敛性，并提出了不需要限制性假设的最优收敛速率问题。

    

    在过去几年里，随机梯度下降（SGD）算法的最终迭代收敛引起了人们的兴趣，因为它在实践中表现良好但缺乏理论理解。对于Lipschitz凸函数，不同的研究建立了最佳的$O(\log(1/\delta)\log T/\sqrt{T})$或$O(\sqrt{\log(1/\delta)/T})$最终迭代的高概率收敛速率，其中$T$是时间跨度，$\delta$是失败概率。然而，为了证明这些界限，所有现有的工作要么局限于紧致域，要么需要几乎肯定有界的噪声。很自然地会问，不需要这两个限制性假设的情况下，SGD的最终迭代是否仍然可以保证最佳的收敛速率。除了这个重要问题外，还有很多理论问题仍然没有答案。

    arXiv:2312.08531v2 Announce Type: replace  Abstract: In the past several years, the last-iterate convergence of the Stochastic Gradient Descent (SGD) algorithm has triggered people's interest due to its good performance in practice but lack of theoretical understanding. For Lipschitz convex functions, different works have established the optimal $O(\log(1/\delta)\log T/\sqrt{T})$ or $O(\sqrt{\log(1/\delta)/T})$ high-probability convergence rates for the final iterate, where $T$ is the time horizon and $\delta$ is the failure probability. However, to prove these bounds, all the existing works are either limited to compact domains or require almost surely bounded noises. It is natural to ask whether the last iterate of SGD can still guarantee the optimal convergence rate but without these two restrictive assumptions. Besides this important question, there are still lots of theoretical problems lacking an answer. For example, compared with the last-iterate convergence of SGD for non-smoot
    

