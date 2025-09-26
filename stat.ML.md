# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Energy based diffusion generator for efficient sampling of Boltzmann distributions.](http://arxiv.org/abs/2401.02080) | 介绍了一种称为基于能量的扩散生成器的新型采样器，用于从任意目标分布中生成样本，并通过扩散模型和广义哈密顿动力学提高采样性能。在各种复杂分布函数上的实证评估中表现出优越性。 |
| [^2] | [Rosenthal-type inequalities for linear statistics of Markov chains.](http://arxiv.org/abs/2303.05838) | 本文建立了一种新的偏差界限，用于马尔可夫链的线性统计，我们关注界限与混合时间的依赖关系，并采用泊松分解的证明技术。 |

# 详细

[^1]: 基于能量的扩散生成器用于高效采样Boltzmann分布

    Energy based diffusion generator for efficient sampling of Boltzmann distributions. (arXiv:2401.02080v1 [cs.LG])

    [http://arxiv.org/abs/2401.02080](http://arxiv.org/abs/2401.02080)

    介绍了一种称为基于能量的扩散生成器的新型采样器，用于从任意目标分布中生成样本，并通过扩散模型和广义哈密顿动力学提高采样性能。在各种复杂分布函数上的实证评估中表现出优越性。

    

    我们介绍了一种称为基于能量的扩散生成器的新型采样器，用于从任意目标分布中生成样本。采样模型采用类似变分自编码器的结构，利用解码器将来自简单分布的潜在变量转换为逼近目标分布的随机变量，并设计了基于扩散模型的编码器。利用扩散模型对复杂分布的强大建模能力，我们可以获得生成样本和目标分布之间的Kullback-Leibler散度的准确变分估计。此外，我们提出了基于广义哈密顿动力学的解码器，进一步提高采样性能。通过实证评估，我们展示了我们的方法在各种复杂分布函数上的有效性，展示了其相对于现有方法的优越性。

    We introduce a novel sampler called the energy based diffusion generator for generating samples from arbitrary target distributions. The sampling model employs a structure similar to a variational autoencoder, utilizing a decoder to transform latent variables from a simple distribution into random variables approximating the target distribution, and we design an encoder based on the diffusion model. Leveraging the powerful modeling capacity of the diffusion model for complex distributions, we can obtain an accurate variational estimate of the Kullback-Leibler divergence between the distributions of the generated samples and the target. Moreover, we propose a decoder based on generalized Hamiltonian dynamics to further enhance sampling performance. Through empirical evaluation, we demonstrate the effectiveness of our method across various complex distribution functions, showcasing its superiority compared to existing methods.
    
[^2]: 线性统计的马尔可夫链的罗森塔尔不等式

    Rosenthal-type inequalities for linear statistics of Markov chains. (arXiv:2303.05838v2 [math.PR] UPDATED)

    [http://arxiv.org/abs/2303.05838](http://arxiv.org/abs/2303.05838)

    本文建立了一种新的偏差界限，用于马尔可夫链的线性统计，我们关注界限与混合时间的依赖关系，并采用泊松分解的证明技术。

    

    本文建立了一种新的偏差界限，用于几何遍历的马尔可夫链的可加函数，类似于独立随机变量的罗森塔尔和伯恩斯坦不等式。我们特别关注界限与相应链的混合时间的依赖关系。更准确地说，我们建立了与罗森塔尔不等式的鞍点版本中的常量以及表征底层马尔可夫核混合特性的常量相关的明确界限。最后，我们的证明技术是新颖的，并且基于泊松分解的反复应用。

    In this paper, we establish novel deviation bounds for additive functionals of geometrically ergodic Markov chains similar to Rosenthal and Bernstein inequalities for sums of independent random variables. We pay special attention to the dependence of our bounds on the mixing time of the corresponding chain. More precisely, we establish explicit bounds that are linked to the constants from the martingale version of the Rosenthal inequality, as well as the constants that characterize the mixing properties of the underlying Markov kernel. Finally, our proof technique is, up to our knowledge, new and based on a recurrent application of the Poisson decomposition.
    

