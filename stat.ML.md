# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust, randomized preconditioning for kernel ridge regression.](http://arxiv.org/abs/2304.12465) | 针对核岭回归问题，本文引入了两种强健的随机预处理技术，分别解决了全数据KRR问题和限制版KRR问题，克服了以往预处理器的故障模式。 |

# 详细

[^1]: 强健的随机预处理方法解决核岭回归问题

    Robust, randomized preconditioning for kernel ridge regression. (arXiv:2304.12465v1 [math.NA])

    [http://arxiv.org/abs/2304.12465](http://arxiv.org/abs/2304.12465)

    针对核岭回归问题，本文引入了两种强健的随机预处理技术，分别解决了全数据KRR问题和限制版KRR问题，克服了以往预处理器的故障模式。

    

    本论文介绍了两种随机预处理技术，用于强健地解决具有中大规模数据点（$10^4 \leq N \leq 10^7$）的核岭回归（KRR）问题。第一种方法，RPCholesky预处理，能够在假设核矩阵特征值有足够快速的多项式衰减的情况下，以$O（N ^ 2）$算法操作准确地解决全数据KRR问题。第二种方法，KRILL预处理，以$O（（N + k ^ 2）k \ logk）$的代价，为KRR问题的限制版本提供准确的解决方案，该版本涉及$k \ll N$选择的数据中心。所提出的方法解决了广泛的KRR问题，克服了以前的KRR预处理器的故障模式，使它们成为实际应用的理想选择。

    This paper introduces two randomized preconditioning techniques for robustly solving kernel ridge regression (KRR) problems with a medium to large number of data points ($10^4 \leq N \leq 10^7$). The first method, RPCholesky preconditioning, is capable of accurately solving the full-data KRR problem in $O(N^2)$ arithmetic operations, assuming sufficiently rapid polynomial decay of the kernel matrix eigenvalues. The second method, KRILL preconditioning, offers an accurate solution to a restricted version of the KRR problem involving $k \ll N$ selected data centers at a cost of $O((N + k^2) k \log k)$ operations. The proposed methods solve a broad range of KRR problems and overcome the failure modes of previous KRR preconditioners, making them ideal for practical applications.
    

