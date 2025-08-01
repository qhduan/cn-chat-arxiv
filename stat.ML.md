# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improved Convergence Rates of Windowed Anderson Acceleration for Symmetric Fixed-Point Iterations](https://arxiv.org/abs/2311.02490) | 窗口式安德森加速在对称不动点迭代中具有改进的根线性收敛率，模拟和实验结果证实其超越标准不动点方法。 |
| [^2] | [GANs Settle Scores!.](http://arxiv.org/abs/2306.01654) | 这篇论文提出了一种新的方法，通过变分方法来统一分析生成器的优化，并展示了在f-散度最小化和IPM GAN中生成器的最优解决方案。这种方法能够平滑分数匹配。 |

# 详细

[^1]: 对称不动点迭代的窗口式安德森加速收敛率的改进

    Improved Convergence Rates of Windowed Anderson Acceleration for Symmetric Fixed-Point Iterations

    [https://arxiv.org/abs/2311.02490](https://arxiv.org/abs/2311.02490)

    窗口式安德森加速在对称不动点迭代中具有改进的根线性收敛率，模拟和实验结果证实其超越标准不动点方法。

    

    本文研究了常用的窗口式安德森加速（AA）算法用于不动点方法，$x^{(k+1)}=q(x^{(k)})$。它首次证明了当算子$q$是线性且对称时，使用先前迭代的滑动窗口的窗口式AA算法能够改进根线性收敛因子，超过不动点迭代。当$q$是非线性的，但在固定点处具有对称雅可比矩阵时，经过略微修改的AA算法被证明对比不动点迭代具有类似的根线性收敛因子改进。模拟验证了我们的观察。此外，使用不同数据模型进行的实验表明，在Tyler的M估计中，AA明显优于标准的不动点方法。

    arXiv:2311.02490v2 Announce Type: replace-cross  Abstract: This paper studies the commonly utilized windowed Anderson acceleration (AA) algorithm for fixed-point methods, $x^{(k+1)}=q(x^{(k)})$. It provides the first proof that when the operator $q$ is linear and symmetric the windowed AA, which uses a sliding window of prior iterates, improves the root-linear convergence factor over the fixed-point iterations. When $q$ is nonlinear, yet has a symmetric Jacobian at a fixed point, a slightly modified AA algorithm is proved to have an analogous root-linear convergence factor improvement over fixed-point iterations. Simulations verify our observations. Furthermore, experiments with different data models demonstrate AA is significantly superior to the standard fixed-point methods for Tyler's M-estimation.
    
[^2]: GANs解决分数争议问题！

    GANs Settle Scores!. (arXiv:2306.01654v1 [cs.LG])

    [http://arxiv.org/abs/2306.01654](http://arxiv.org/abs/2306.01654)

    这篇论文提出了一种新的方法，通过变分方法来统一分析生成器的优化，并展示了在f-散度最小化和IPM GAN中生成器的最优解决方案。这种方法能够平滑分数匹配。

    

    生成对抗网络（GAN）由一个生成器和一个判别器组成，生成器被训练以学习期望数据的基础分布，而判别器则被训练以区分真实样本和生成器输出的样本。本文提出了一种统一的方法，通过变分方法来分析生成器优化。在f-散度最小化 GAN 中，我们表明最优生成器是通过将其输出分布的得分与数据分布的得分进行匹配得到的。在IPM GAN中，我们表明这个最优生成器匹配得分型函数，包括与所选IPM约束空间相关的核流场。此外，IPM-GAN优化可以看作是平滑分数匹配中的一种，其中数据和生成器分布的得分与在核函数上进行卷积处理。

    Generative adversarial networks (GANs) comprise a generator, trained to learn the underlying distribution of the desired data, and a discriminator, trained to distinguish real samples from those output by the generator. A majority of GAN literature focuses on understanding the optimality of the discriminator through integral probability metric (IPM) or divergence based analysis. In this paper, we propose a unified approach to analyzing the generator optimization through variational approach. In $f$-divergence-minimizing GANs, we show that the optimal generator is the one that matches the score of its output distribution with that of the data distribution, while in IPM GANs, we show that this optimal generator matches score-like functions, involving the flow-field of the kernel associated with a chosen IPM constraint space. Further, the IPM-GAN optimization can be seen as one of smoothed score-matching, where the scores of the data and the generator distributions are convolved with the 
    

