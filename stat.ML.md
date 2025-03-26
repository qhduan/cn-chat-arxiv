# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Not All Learnable Distribution Classes are Privately Learnable](https://arxiv.org/abs/2402.00267) | 这篇论文证明了一类分布虽然可以在有限样本下以总变差距离进行学习，但却无法在（ε，δ）-差分隐私下学习。 |
| [^2] | [Targeted Separation and Convergence with Kernel Discrepancies.](http://arxiv.org/abs/2209.12835) | 通过核差异度量，我们推导出了新的充分必要条件，实现了将目标分离出来，以及控制对目标的弱收敛性。此外，我们在$\mathbb{R}^d$上使用了这些结果来扩展了核Stein差异分离和收敛控制的已知条件，并开发了能够精确度量目标的弱收敛性的核差异度量。 |

# 详细

[^1]: 并非所有可学习的分布类都能在差分隐私下进行学习

    Not All Learnable Distribution Classes are Privately Learnable

    [https://arxiv.org/abs/2402.00267](https://arxiv.org/abs/2402.00267)

    这篇论文证明了一类分布虽然可以在有限样本下以总变差距离进行学习，但却无法在（ε，δ）-差分隐私下学习。

    

    我们给出了一个示例，展示了一类分布在有限样本下可以以总变差距离进行学习，但在（ε，δ）-差分隐私下无法学习。这推翻了Ashtiani的一个猜想。

    We give an example of a class of distributions that is learnable in total variation distance with a finite number of samples, but not learnable under $(\varepsilon, \delta)$-differential privacy. This refutes a conjecture of Ashtiani.
    
[^2]: 通过核差异实现有针对性的分离与收敛

    Targeted Separation and Convergence with Kernel Discrepancies. (arXiv:2209.12835v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2209.12835](http://arxiv.org/abs/2209.12835)

    通过核差异度量，我们推导出了新的充分必要条件，实现了将目标分离出来，以及控制对目标的弱收敛性。此外，我们在$\mathbb{R}^d$上使用了这些结果来扩展了核Stein差异分离和收敛控制的已知条件，并开发了能够精确度量目标的弱收敛性的核差异度量。

    

    最大均值差异（MMDs）如核Stein差异（KSD）已经成为广泛应用的中心，包括假设检验、采样器选择、分布近似和变分推断。在每个设置中，这些基于核的差异度量需要实现（i）将目标P与其他概率测度分离，甚至（ii）控制对P的弱收敛。在本文中，我们推导了确保（i）和（ii）的新的充分必要条件。对于可分的度量空间上的MMDs，我们描述了分离Bochner可嵌入测度的核，并引入简单的条件来分离所有具有无界核的测度和用有界核来控制收敛。我们利用这些结果在$\mathbb{R}^d$上大大扩展了KSD分离和收敛控制的已知条件，并开发了首个能够精确度量对P的弱收敛的KSDs。在这个过程中，我们强调了我们的结果的影响。

    Maximum mean discrepancies (MMDs) like the kernel Stein discrepancy (KSD) have grown central to a wide range of applications, including hypothesis testing, sampler selection, distribution approximation, and variational inference. In each setting, these kernel-based discrepancy measures are required to (i) separate a target P from other probability measures or even (ii) control weak convergence to P. In this article we derive new sufficient and necessary conditions to ensure (i) and (ii). For MMDs on separable metric spaces, we characterize those kernels that separate Bochner embeddable measures and introduce simple conditions for separating all measures with unbounded kernels and for controlling convergence with bounded kernels. We use these results on $\mathbb{R}^d$ to substantially broaden the known conditions for KSD separation and convergence control and to develop the first KSDs known to exactly metrize weak convergence to P. Along the way, we highlight the implications of our res
    

