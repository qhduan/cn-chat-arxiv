# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Convergence of Sign-based Random Reshuffling Algorithms for Nonconvex Optimization.](http://arxiv.org/abs/2310.15976) | 该论文通过证明signSGD算法在非凸优化问题中的随机重排（SignRR）的收敛性，弥补了现有分析中的缺陷，提出了SignRVR和SignRVM算法，并且都以较快的收敛速度收敛于全局最优解。 |

# 详细

[^1]: 非凸优化的基于符号随机重排算法的收敛性研究

    Convergence of Sign-based Random Reshuffling Algorithms for Nonconvex Optimization. (arXiv:2310.15976v1 [cs.LG])

    [http://arxiv.org/abs/2310.15976](http://arxiv.org/abs/2310.15976)

    该论文通过证明signSGD算法在非凸优化问题中的随机重排（SignRR）的收敛性，弥补了现有分析中的缺陷，提出了SignRVR和SignRVM算法，并且都以较快的收敛速度收敛于全局最优解。

    

    由于其通信效率较高，signSGD在非凸优化中很受欢迎。然而，现有对signSGD的分析基于假设每次迭代中的数据都是有放回采样的，这与实际实现中数据的随机重排和顺序馈送进算法的情况相矛盾。为了弥补这一差距，我们证明了signSGD在非凸优化中的随机重排（SignRR）的首个收敛结果。给定数据集大小$n$，数据迭代次数$T$，和随机梯度的方差限制$\sigma^2$，我们证明了SignRR的收敛速度与signSGD相同，为$O(\log(nT)/\sqrt{nT} + \|\sigma\|_1)$ \citep{bernstein2018signsgd}。接着，我们还提出了 SignRVR 和 SignRVM，分别利用了方差约减梯度和动量更新，都以$O(\log(nT)/\sqrt{nT})$的速度收敛。与signSGD的分析不同，我们的结果不需要每次迭代中极大的批次大小与同等数量的梯度进行比较。

    signSGD is popular in nonconvex optimization due to its communication efficiency. Yet, existing analyses of signSGD rely on assuming that data are sampled with replacement in each iteration, contradicting the practical implementation where data are randomly reshuffled and sequentially fed into the algorithm. We bridge this gap by proving the first convergence result of signSGD with random reshuffling (SignRR) for nonconvex optimization. Given the dataset size $n$, the number of epochs of data passes $T$, and the variance bound of a stochastic gradient $\sigma^2$, we show that SignRR has the same convergence rate $O(\log(nT)/\sqrt{nT} + \|\sigma\|_1)$ as signSGD \citep{bernstein2018signsgd}. We then present SignRVR and SignRVM, which leverage variance-reduced gradients and momentum updates respectively, both converging at $O(\log(nT)/\sqrt{nT})$. In contrast with the analysis of signSGD, our results do not require an extremely large batch size in each iteration to be of the same order a
    

