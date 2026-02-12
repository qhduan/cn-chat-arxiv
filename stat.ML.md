# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neural Rank Collapse: Weight Decay and Small Within-Class Variability Yield Low-Rank Bias](https://arxiv.org/abs/2402.03991) | 神经网络中的权重衰减和小的类内变化与低秩偏差现象有关 |
| [^2] | [Optimal Estimates for Pairwise Learning with Deep ReLU Networks.](http://arxiv.org/abs/2305.19640) | 本文研究了深度ReLU网络中的成对学习，提出了一个针对一般损失函数的误差估计的尖锐界限，并基于成对最小二乘损失得出几乎最优的过度泛化误差界限。 |
| [^3] | [When does Metropolized Hamiltonian Monte Carlo provably outperform Metropolis-adjusted Langevin algorithm?.](http://arxiv.org/abs/2304.04724) | 本文表明，从高维平滑目标分布采样时，Metropolized Hamiltonian Monte Carlo (HMC)比Metropolis-adjusted Langevin算法（MALA）更有效。 |

# 详细

[^1]: 神经网络的权重衰减和类内变化小会导致低秩偏差

    Neural Rank Collapse: Weight Decay and Small Within-Class Variability Yield Low-Rank Bias

    [https://arxiv.org/abs/2402.03991](https://arxiv.org/abs/2402.03991)

    神经网络中的权重衰减和小的类内变化与低秩偏差现象有关

    

    近期在深度学习领域的研究显示了一个隐含的低秩偏差现象：深度网络中的权重矩阵往往近似为低秩，在训练过程中或从已经训练好的模型中去除相对较小的奇异值可以显著减小模型大小，同时保持甚至提升模型性能。然而，大多数关于神经网络低秩偏差的理论研究都涉及到简化的线性深度网络。在本文中，我们考虑了带有非线性激活函数和权重衰减参数的通用网络，并展示了一个有趣的神经秩崩溃现象，它将训练好的网络的低秩偏差与网络的神经崩溃特性联系起来：随着权重衰减参数的增加，网络中每一层的秩呈比例递减，与前面层的隐藏空间嵌入的类内变化成反比。我们的理论发现得到了支持。

    Recent work in deep learning has shown strong empirical and theoretical evidence of an implicit low-rank bias: weight matrices in deep networks tend to be approximately low-rank and removing relatively small singular values during training or from available trained models may significantly reduce model size while maintaining or even improving model performance. However, the majority of the theoretical investigations around low-rank bias in neural networks deal with oversimplified deep linear networks. In this work, we consider general networks with nonlinear activations and the weight decay parameter, and we show the presence of an intriguing neural rank collapse phenomenon, connecting the low-rank bias of trained networks with networks' neural collapse properties: as the weight decay parameter grows, the rank of each layer in the network decreases proportionally to the within-class variability of the hidden-space embeddings of the previous layers. Our theoretical findings are supporte
    
[^2]: 深度ReLU网络中的成对学习最优估计

    Optimal Estimates for Pairwise Learning with Deep ReLU Networks. (arXiv:2305.19640v1 [stat.ML])

    [http://arxiv.org/abs/2305.19640](http://arxiv.org/abs/2305.19640)

    本文研究了深度ReLU网络中的成对学习，提出了一个针对一般损失函数的误差估计的尖锐界限，并基于成对最小二乘损失得出几乎最优的过度泛化误差界限。

    

    成对学习指的是在损失函数中考虑一对样本的学习任务。本文研究了深度ReLU网络中的成对学习，并估计了过度泛化误差。对于满足某些温和条件的一般损失函数，建立了误差估计的尖锐界限，其误差估计的阶数为O（（Vlog（n）/ n）1 /（2-β））。特别地，对于成对最小二乘损失，我们得到了过度泛化误差的几乎最优界限，在真实的预测器满足某些光滑性正则性时，最优界限达到了最小化界限，差距仅为对数项。

    Pairwise learning refers to learning tasks where a loss takes a pair of samples into consideration. In this paper, we study pairwise learning with deep ReLU networks and estimate the excess generalization error. For a general loss satisfying some mild conditions, a sharp bound for the estimation error of order $O((V\log(n) /n)^{1/(2-\beta)})$ is established. In particular, with the pairwise least squares loss, we derive a nearly optimal bound of the excess generalization error which achieves the minimax lower bound up to a logrithmic term when the true predictor satisfies some smoothness regularities.
    
[^3]: Metropolized Hamiltonian Monte Carlo何时能证明优于Metropolis-adjusted Langevin算法？

    When does Metropolized Hamiltonian Monte Carlo provably outperform Metropolis-adjusted Langevin algorithm?. (arXiv:2304.04724v1 [stat.CO])

    [http://arxiv.org/abs/2304.04724](http://arxiv.org/abs/2304.04724)

    本文表明，从高维平滑目标分布采样时，Metropolized Hamiltonian Monte Carlo (HMC)比Metropolis-adjusted Langevin算法（MALA）更有效。

    

    本文分析了Metropolized Hamiltonian Monte Carlo (HMC)的混合时间，使用leapfrog积分器从$\mathbb{R}^d$分布中采样，该分布的对数密度平滑，具有Frobenius范数上的李普希茨黑塞，并满足等周性。我们将梯度复杂度限制为从一个暖启动达到$\epsilon$误差的总变异距离所需的$\tilde O(d^{1/4}\text{polylog}(1/\epsilon))$，并展示了选择比1更大的leapfrog步数的好处。为了超越Wu等人（2022）对Metropolis-adjusted Langevin algorithm (MALA)的分析，其在维度依赖性上是$\tilde{O}(d^{1/2}\text{polylog}(1/\epsilon))$，我们揭示了证明中的一个关键特征：连续HMC动态的位置和速度变量的离散化的联合分布近似不变。当通过leapfrog步数的归纳来展示这个关键特征时，我们能够获得各种量的矩的估计，这些量在限制Metropolized HMC的混合时间时是至关重要的，而在MALA中已知的类似结果是错误的。我们的结果表明，在采样高维平滑目标分布时，使用具有大量leapfrog步骤的Metropolized HMC可能比使用MALA更有效。

    We analyze the mixing time of Metropolized Hamiltonian Monte Carlo (HMC) with the leapfrog integrator to sample from a distribution on $\mathbb{R}^d$ whose log-density is smooth, has Lipschitz Hessian in Frobenius norm and satisfies isoperimetry. We bound the gradient complexity to reach $\epsilon$ error in total variation distance from a warm start by $\tilde O(d^{1/4}\text{polylog}(1/\epsilon))$ and demonstrate the benefit of choosing the number of leapfrog steps to be larger than 1. To surpass previous analysis on Metropolis-adjusted Langevin algorithm (MALA) that has $\tilde{O}(d^{1/2}\text{polylog}(1/\epsilon))$ dimension dependency in Wu et al. (2022), we reveal a key feature in our proof that the joint distribution of the location and velocity variables of the discretization of the continuous HMC dynamics stays approximately invariant. This key feature, when shown via induction over the number of leapfrog steps, enables us to obtain estimates on moments of various quantities tha
    

