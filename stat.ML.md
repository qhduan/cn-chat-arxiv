# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the average-case complexity of learning output distributions of quantum circuits.](http://arxiv.org/abs/2305.05765) | 本文证明了砖墙随机量子电路输出分布学习是一个平均复杂度困难的问题，需要进行超多项式次数的查询才能有效解决。 |

# 详细

[^1]: 关于量子电路输出分布学习的平均复杂度

    On the average-case complexity of learning output distributions of quantum circuits. (arXiv:2305.05765v1 [quant-ph])

    [http://arxiv.org/abs/2305.05765](http://arxiv.org/abs/2305.05765)

    本文证明了砖墙随机量子电路输出分布学习是一个平均复杂度困难的问题，需要进行超多项式次数的查询才能有效解决。

    

    本文研究了砖墙随机量子电路的输出分布学习问题，并证明在统计查询模型下，该问题的平均复杂度是困难的。具体地，对于深度为$d$、由$n$个量子比特构成的砖墙随机量子电路，我们得出了三个主要结论：在超对数电路深度$d=\omega(\log(n))$时，任何学习算法都需要进行超多项式次数的查询才能在随机实例上实现恒定的成功概率。存在一个$d=O(n)$，这意味着任何学习算法需要进行$\Omega(2^n)$次查询才能在随机实例上实现$O(2^{-n})$的成功概率。在无限电路深度$d\to\infty$时，任何学习算法都需要进行$2^{2^{\Omega(n)}}$次查询才能在随机实例上实现$2^{-2^{\Omega(n)}}$的成功概率。作为一个独立的辅助结果，我们还证明了......（文章内容截断）

    In this work, we show that learning the output distributions of brickwork random quantum circuits is average-case hard in the statistical query model. This learning model is widely used as an abstract computational model for most generic learning algorithms. In particular, for brickwork random quantum circuits on $n$ qubits of depth $d$, we show three main results:  - At super logarithmic circuit depth $d=\omega(\log(n))$, any learning algorithm requires super polynomially many queries to achieve a constant probability of success over the randomly drawn instance.  - There exists a $d=O(n)$, such that any learning algorithm requires $\Omega(2^n)$ queries to achieve a $O(2^{-n})$ probability of success over the randomly drawn instance.  - At infinite circuit depth $d\to\infty$, any learning algorithm requires $2^{2^{\Omega(n)}}$ many queries to achieve a $2^{-2^{\Omega(n)}}$ probability of success over the randomly drawn instance.  As an auxiliary result of independent interest, we show 
    

