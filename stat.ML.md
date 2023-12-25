# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Provable convergence guarantees for black-box variational inference.](http://arxiv.org/abs/2306.03638) | 本文提出了一种基于密集高斯变分族的梯度估计器，在此基础上使用近端和投影随机梯度下降，提供了黑盒变分推断收敛于逼真推断问题的第一个严格保证。 |
| [^2] | [Investigating the Corruption Robustness of Image Classifiers with Random Lp-norm Corruptions.](http://arxiv.org/abs/2305.05400) | 本研究探讨了使用随机Lp范数失真对图像分类器的训练和测试数据进行增强，并评估模型对不可感知随机失真的稳健性，发现稳健性可能会提高模型在随机失真方面的性能，但也可能会损害L∞范数的稳健性。 |
| [^3] | [Diffusion Bridge Mixture Transports, Schr\"odinger Bridge Problems and Generative Modeling.](http://arxiv.org/abs/2304.00917) | 本文提出了一种新的迭代算法IDBM，用于解决动态Schr\"odinger桥问题，该算法能够在每一步有效地耦合目标度量，并在各种应用中表现出竞争力。此外，还讨论了使用扩散过程的时间反演来定义一个近似传输简单分布到目标分布的生成过程的最新进展。 |

# 详细

[^1]: 黑盒变分推断的收敛性保证

    Provable convergence guarantees for black-box variational inference. (arXiv:2306.03638v1 [cs.LG])

    [http://arxiv.org/abs/2306.03638](http://arxiv.org/abs/2306.03638)

    本文提出了一种基于密集高斯变分族的梯度估计器，在此基础上使用近端和投影随机梯度下降，提供了黑盒变分推断收敛于逼真推断问题的第一个严格保证。

    

    尽管黑盒变分推断被广泛应用，但没有证明其随机优化成功的证明。我们提出这是现有随机优化证明中的理论差距，即具有异常噪声边界和复合非平滑目标的梯度估计器的挑战。对于密集的高斯变分族，我们观察到现有的基于再参数化的梯度估计器满足二次噪声界，并为使用该界限的近端和投影随机梯度下降提供新的收敛保证。这提供了第一个黑盒变分推断收敛于逼真推断问题的严格保证。

    While black-box variational inference is widely used, there is no proof that its stochastic optimization succeeds. We suggest this is due to a theoretical gap in existing stochastic optimization proofs-namely the challenge of gradient estimators with unusual noise bounds, and a composite non-smooth objective. For dense Gaussian variational families, we observe that existing gradient estimators based on reparameterization satisfy a quadratic noise bound and give novel convergence guarantees for proximal and projected stochastic gradient descent using this bound. This provides the first rigorous guarantee that black-box variational inference converges for realistic inference problems.
    
[^2]: 使用随机Lp范数失真探究图像分类器的腐败稳健性

    Investigating the Corruption Robustness of Image Classifiers with Random Lp-norm Corruptions. (arXiv:2305.05400v1 [cs.LG])

    [http://arxiv.org/abs/2305.05400](http://arxiv.org/abs/2305.05400)

    本研究探讨了使用随机Lp范数失真对图像分类器的训练和测试数据进行增强，并评估模型对不可感知随机失真的稳健性，发现稳健性可能会提高模型在随机失真方面的性能，但也可能会损害L∞范数的稳健性。

    

    稳健性是机器学习分类器实现安全和可靠的基本属性。在对图像分类模型的对抗稳健性和形式稳健性验证领域中，稳健性通常被定义为在Lp范数距离内对所有输入变化的稳定性。然而，对随机失真的稳健性通常通过在现实世界中观察到的变化来改进和评估，而很少考虑数学定义的Lp范数失真。本研究探讨了使用随机Lp范数失真来增强图像分类器的训练和测试数据。我们借鉴了对抗稳健性领域的方法来评估模型对不可感知随机失真的稳健性。我们实证和理论上研究了在不同Lp范数之间稳健性是否可转移，并得出结论，哪些Lp范数的失真应该用来训练和评估模型。我们发现训练数据增强可能会提高模型在随机失真方面的性能，但也可能会损害L∞范数的稳健性。

    Robustness is a fundamental property of machine learning classifiers to achieve safety and reliability. In the fields of adversarial robustness and formal robustness verification of image classification models, robustness is commonly defined as the stability to all input variations within an Lp-norm distance. However, robustness to random corruptions is usually improved and evaluated using variations observed in the real-world, while mathematically defined Lp-norm corruptions are rarely considered. This study investigates the use of random Lp-norm corruptions to augment the training and test data of image classifiers. We adapt an approach from the field of adversarial robustness to assess the model robustness to imperceptible random corruptions. We empirically and theoretically investigate whether robustness is transferable across different Lp-norms and derive conclusions on which Lp-norm corruptions a model should be trained and evaluated on. We find that training data augmentation wi
    
[^3]: 扩散桥混合传输、薛定谔桥问题和生成建模

    Diffusion Bridge Mixture Transports, Schr\"odinger Bridge Problems and Generative Modeling. (arXiv:2304.00917v1 [stat.ML])

    [http://arxiv.org/abs/2304.00917](http://arxiv.org/abs/2304.00917)

    本文提出了一种新的迭代算法IDBM，用于解决动态Schr\"odinger桥问题，该算法能够在每一步有效地耦合目标度量，并在各种应用中表现出竞争力。此外，还讨论了使用扩散过程的时间反演来定义一个近似传输简单分布到目标分布的生成过程的最新进展。

    

    动态薛定谔桥问题寻求定义在两个目标概率分布之间的传输的随机过程，同时最优地满足最接近参考过程的Kullback-Leibler散度的准则。我们提出了一种新的基于采样的迭代算法，即迭代扩散桥混合传输（IDBM），旨在解决动态薛定谔桥问题。IDBM过程表现出在每一步实现目标度量之间的有效耦合的有吸引力的属性。我们进行了IDBM过程的初始理论研究，建立了其收敛性质。理论发现通过许多数值实验证明了IDBM过程在各种应用中出色的性能。生成建模方面的最新进展使用扩散过程的时间反演来定义一个近似传输简单分布到目标分布的生成过程。本文提出了一种新的算法，称为迭代扩散桥混合传输（IDBM），用于解决动态薛定谔桥问题。IDBM在每一步实现目标度量之间的有效耦合，并且在各种应用中表现出良好的性能。我们的理论研究证明了IDBM算法的收敛性质。通过许多数值实验进一步说明了所提出的算法的有效性。此外，还讨论了生成建模方面的最新进展，它使用扩散过程的时间反演来近似目标分布。

    The dynamic Schr\"odinger bridge problem seeks a stochastic process that defines a transport between two target probability measures, while optimally satisfying the criteria of being closest, in terms of Kullback-Leibler divergence, to a reference process.  We propose a novel sampling-based iterative algorithm, the iterated diffusion bridge mixture transport (IDBM), aimed at solving the dynamic Schr\"odinger bridge problem. The IDBM procedure exhibits the attractive property of realizing a valid coupling between the target measures at each step. We perform an initial theoretical investigation of the IDBM procedure, establishing its convergence properties. The theoretical findings are complemented by numerous numerical experiments illustrating the competitive performance of the IDBM procedure across various applications.  Recent advancements in generative modeling employ the time-reversal of a diffusion process to define a generative process that approximately transports a simple distri
    

