# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Functional Summaries via the Independent Component Laplace Process.](http://arxiv.org/abs/2309.00125) | 本论文提出了一种新的差分隐私函数性摘要机制，通过使用独立分量拉普拉斯过程对无限维的函数性摘要进行扰动，放宽了对数据轨迹的假设，并相对于传统的有限维子空间嵌入方法保留了更高的效用。实验证明了该机制的可行性和有效性。 |
| [^2] | [MCMC-Correction of Score-Based Diffusion Models for Model Composition.](http://arxiv.org/abs/2307.14012) | 本文提出了一种修正基于得分的扩散模型的方法，使其能够与各种MCMC方法结合，从而实现模型组合和进行更好的采样。 |

# 详细

[^1]: 通过独立分量拉普拉斯过程实现差分隐私的函数性摘要

    Differentially Private Functional Summaries via the Independent Component Laplace Process. (arXiv:2309.00125v1 [stat.ML])

    [http://arxiv.org/abs/2309.00125](http://arxiv.org/abs/2309.00125)

    本论文提出了一种新的差分隐私函数性摘要机制，通过使用独立分量拉普拉斯过程对无限维的函数性摘要进行扰动，放宽了对数据轨迹的假设，并相对于传统的有限维子空间嵌入方法保留了更高的效用。实验证明了该机制的可行性和有效性。

    

    在这项工作中，我们提出了一种称为独立分量拉普拉斯过程（ICLP）机制的差分隐私函数性摘要的新机制。通过将感兴趣的函数性摘要视为真正无限维对象，并使用ICLP噪声来扰动它们，该新机制放宽了关于数据轨迹的假设，并相对于文献中的经典有限维子空间嵌入方法保留了更高的效用。我们在多个函数空间中验证了所提出机制的可行性。我们考虑了几个统计估计问题，并通过轻微过平滑摘要来证明隐私成本不会主导统计误差，并且在渐近情况下可以忽略。对合成和真实数据集的数值实验证明了所提出机制的有效性。

    In this work, we propose a new mechanism for releasing differentially private functional summaries called the Independent Component Laplace Process, or ICLP, mechanism. By treating the functional summaries of interest as truly infinite-dimensional objects and perturbing them with the ICLP noise, this new mechanism relaxes assumptions on data trajectories and preserves higher utility compared to classical finite-dimensional subspace embedding approaches in the literature. We establish the feasibility of the proposed mechanism in multiple function spaces. Several statistical estimation problems are considered, and we demonstrate by slightly over-smoothing the summary, the privacy cost will not dominate the statistical error and is asymptotically negligible. Numerical experiments on synthetic and real datasets demonstrate the efficacy of the proposed mechanism.
    
[^2]: MCMC-修正基于得分的扩散模型用于模型组合

    MCMC-Correction of Score-Based Diffusion Models for Model Composition. (arXiv:2307.14012v1 [stat.ML])

    [http://arxiv.org/abs/2307.14012](http://arxiv.org/abs/2307.14012)

    本文提出了一种修正基于得分的扩散模型的方法，使其能够与各种MCMC方法结合，从而实现模型组合和进行更好的采样。

    

    扩散模型可以用得分或能量函数来参数化。能量参数化具有更好的理论特性，主要是它可以通过在提议样本中总能量的变化基于Metropolis-Hastings修正步骤来进行扩展采样过程。然而，它似乎产生了稍微较差的性能，更重要的是，由于基于得分的扩散模型的普遍流行，现有的预训练能量参数化模型的可用性受到限制。这种限制削弱了模型组合的目的，即将预训练模型组合起来从新分布中进行采样。然而，我们的提议建议保留得分参数化，而是通过对得分函数进行线积分来计算基于能量的接受概率。这使我们能够重用现有的扩散模型，并将反向过程与各种马尔可夫链蒙特卡罗（MCMC）方法组合起来。

    Diffusion models can be parameterised in terms of either a score or an energy function. The energy parameterisation has better theoretical properties, mainly that it enables an extended sampling procedure with a Metropolis--Hastings correction step, based on the change in total energy in the proposed samples. However, it seems to yield slightly worse performance, and more importantly, due to the widespread popularity of score-based diffusion, there are limited availability of off-the-shelf pre-trained energy-based ones. This limitation undermines the purpose of model composition, which aims to combine pre-trained models to sample from new distributions. Our proposal, however, suggests retaining the score parameterization and instead computing the energy-based acceptance probability through line integration of the score function. This allows us to re-use existing diffusion models and still combine the reverse process with various Markov-Chain Monte Carlo (MCMC) methods. We evaluate our 
    

