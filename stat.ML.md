# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Two Sides of The Same Coin: Bridging Deep Equilibrium Models and Neural ODEs via Homotopy Continuation.](http://arxiv.org/abs/2310.09583) | 通过同伦延续，我们建立了深度平衡模型（DEQs）和神经常微分方程（Neural ODEs）之间的连接，并提出了一种新的隐式模型HomoODE，它继承了DEQs的高精度性能和Neural ODEs的稳定性。 |
| [^2] | [Log-Gaussian Gamma Processes for Training Bayesian Neural Networks in Raman and CARS Spectroscopies.](http://arxiv.org/abs/2310.08055) | 本论文提出了一种利用gamma分布和log-Gaussian建模的方法，用于生成合成数据集以训练神经网络，解决了实际观测数据有限的挑战。通过应用于Raman和CARS光谱，同时训练两个贝叶斯神经网络来估计gamma过程的参数，可以估计基础的光谱并提供不确定性。 |
| [^3] | [Diffusion Generative Flow Samplers: Improving learning signals through partial trajectory optimization.](http://arxiv.org/abs/2310.02679) | 这项工作介绍了一种名为扩散生成流采样器（DGFS）的采样框架，通过将学习过程分解为短的部分轨迹段，实现从难以处理的高维密度函数中进行采样。它通过利用中间的学习信号和非策略探索能力来改善学习信号的分配问题。 |
| [^4] | [SimFBO: Towards Simple, Flexible and Communication-efficient Federated Bilevel Learning.](http://arxiv.org/abs/2305.19442) | SimFBO和其ShroFBO变体提出了一个简单、灵活且通信高效的FBO框架，可以应用于元学习和超参数优化任务。 |
| [^5] | [Unifying GANs and Score-Based Diffusion as Generative Particle Models.](http://arxiv.org/abs/2305.16150) | 本文提出了一个新框架，将生成器训练作为粒子模型的一个推广，从而统一了粒子和对抗生成模型。这个框架可以将生成器集成到基于分数扩散模型中，并在没有生成器的情况下训练GAN。 |
| [^6] | [Moment Matching Denoising Gibbs Sampling.](http://arxiv.org/abs/2305.11650) | 本文提出了动量匹配去噪Gibbs采样方法，可以在给定‘嘈杂’的模型的情况下，从干净的模型中有效地进行采样。 |
| [^7] | [Learning with Explanation Constraints.](http://arxiv.org/abs/2303.14496) | 本文研究了解释约束下的学习问题，提出了EPAC模型，探讨了使用这些解释时模型的益处，并提供了一种基于变分近似的算法解决方案。 |

# 详细

[^1]: 两枚硬币的两面：通过同伦延续连接深度平衡模型和神经常微分方程

    Two Sides of The Same Coin: Bridging Deep Equilibrium Models and Neural ODEs via Homotopy Continuation. (arXiv:2310.09583v1 [cs.LG])

    [http://arxiv.org/abs/2310.09583](http://arxiv.org/abs/2310.09583)

    通过同伦延续，我们建立了深度平衡模型（DEQs）和神经常微分方程（Neural ODEs）之间的连接，并提出了一种新的隐式模型HomoODE，它继承了DEQs的高精度性能和Neural ODEs的稳定性。

    

    深度平衡模型（DEQs）和神经常微分方程（Neural ODEs）是两种隐式模型的分支，以其卓越的性能和低内存消耗成就了显著的成功。虽然两者都是隐式模型，但DEQs和Neural ODEs是从不同的数学形式导出的。受同伦延续的启发，我们建立了这两种模型之间的联系，并表明它们实际上是同一个硬币的两面。同伦延续是一种基于对应ODE的解非线性方程组的经典方法。给定这种联系，我们提出了一种新的隐式模型称为HomoODE，它继承了DEQs的高精度性质和Neural ODEs的稳定性。与DEQs不同，HomoODE通过同伦延续使用修改后的神经常微分方程隐式地解决平衡点找寻问题。

    Deep Equilibrium Models (DEQs) and Neural Ordinary Differential Equations (Neural ODEs) are two branches of implicit models that have achieved remarkable success owing to their superior performance and low memory consumption. While both are implicit models, DEQs and Neural ODEs are derived from different mathematical formulations. Inspired by homotopy continuation, we establish a connection between these two models and illustrate that they are actually two sides of the same coin. Homotopy continuation is a classical method of solving nonlinear equations based on a corresponding ODE. Given this connection, we proposed a new implicit model called HomoODE that inherits the property of high accuracy from DEQs and the property of stability from Neural ODEs. Unlike DEQs, which explicitly solve an equilibrium-point-finding problem via Newton's methods in the forward pass, HomoODE solves the equilibrium-point-finding problem implicitly using a modified Neural ODE via homotopy continuation. Fur
    
[^2]: 用于Raman和CARS光谱学中贝叶斯神经网络的对数-高斯γ过程的训练方法

    Log-Gaussian Gamma Processes for Training Bayesian Neural Networks in Raman and CARS Spectroscopies. (arXiv:2310.08055v1 [stat.AP])

    [http://arxiv.org/abs/2310.08055](http://arxiv.org/abs/2310.08055)

    本论文提出了一种利用gamma分布和log-Gaussian建模的方法，用于生成合成数据集以训练神经网络，解决了实际观测数据有限的挑战。通过应用于Raman和CARS光谱，同时训练两个贝叶斯神经网络来估计gamma过程的参数，可以估计基础的光谱并提供不确定性。

    

    我们提出了一种利用gamma分布的随机变量和log-Gaussian建模的方法，用于生成适合训练神经网络的合成数据集。这种方法解决了各种应用中实际观测数据有限的挑战。我们将此方法应用于Raman和相干防-斯托克斯拉曼散射(CARS)光谱中，使用实验光谱估计gamma过程的参数。参数估计使用马尔可夫链蒙特卡洛方法进行，从而为模型提供完整贝叶斯后验分布，可用于合成数据生成。此外，我们使用高斯过程对Raman和CARS的加性和乘性背景函数进行建模。我们训练了两个贝叶斯神经网络来估计gamma过程的参数，然后可以用这些参数来估计基础的Raman光谱，并通过概率分布参数的估计同时提供不确定性。

    We propose an approach utilizing gamma-distributed random variables, coupled with log-Gaussian modeling, to generate synthetic datasets suitable for training neural networks. This addresses the challenge of limited real observations in various applications. We apply this methodology to both Raman and coherent anti-Stokes Raman scattering (CARS) spectra, using experimental spectra to estimate gamma process parameters. Parameter estimation is performed using Markov chain Monte Carlo methods, yielding a full Bayesian posterior distribution for the model which can be sampled for synthetic data generation. Additionally, we model the additive and multiplicative background functions for Raman and CARS with Gaussian processes. We train two Bayesian neural networks to estimate parameters of the gamma process which can then be used to estimate the underlying Raman spectrum and simultaneously provide uncertainty through the estimation of parameters of a probability distribution. We apply the trai
    
[^3]: 扩散生成流采样器：通过部分轨迹优化改善学习信号

    Diffusion Generative Flow Samplers: Improving learning signals through partial trajectory optimization. (arXiv:2310.02679v1 [cs.LG])

    [http://arxiv.org/abs/2310.02679](http://arxiv.org/abs/2310.02679)

    这项工作介绍了一种名为扩散生成流采样器（DGFS）的采样框架，通过将学习过程分解为短的部分轨迹段，实现从难以处理的高维密度函数中进行采样。它通过利用中间的学习信号和非策略探索能力来改善学习信号的分配问题。

    

    我们解决了从难以处理的高维密度函数中进行采样的问题，这是在机器学习和统计中经常出现的基本任务。我们扩展了最近的基于采样的方法，利用控制的随机过程来模拟这些目标密度的近似样本。这些方法的主要缺点是训练目标需要计算完整的轨迹，导致由于使用完整轨迹和只在终端时间存在的学习信号的使用而产生缓慢的信用分配问题。在这项工作中，我们提出了扩散生成流采样器（DGFS），这是一个基于采样的框架，可以将学习过程可行地分解为短的部分轨迹段，通过参数化一个额外的“流函数”。我们的方法借鉴了生成流网络（GFlowNets）的理论，使我们能够利用中间的学习信号，并从非策略探索能力中受益。

    We tackle the problem of sampling from intractable high-dimensional density functions, a fundamental task that often appears in machine learning and statistics. We extend recent sampling-based approaches that leverage controlled stochastic processes to model approximate samples from these target densities. The main drawback of these approaches is that the training objective requires full trajectories to compute, resulting in sluggish credit assignment issues due to use of entire trajectories and a learning signal present only at the terminal time. In this work, we present Diffusion Generative Flow Samplers (DGFS), a sampling-based framework where the learning process can be tractably broken down into short partial trajectory segments, via parameterizing an additional "flow function". Our method takes inspiration from the theory developed for generative flow networks (GFlowNets), allowing us to make use of intermediate learning signals and benefit from off-policy exploration capabilitie
    
[^4]: SimFBO：简单、灵活且通信高效的联邦双层学习

    SimFBO: Towards Simple, Flexible and Communication-efficient Federated Bilevel Learning. (arXiv:2305.19442v1 [cs.LG])

    [http://arxiv.org/abs/2305.19442](http://arxiv.org/abs/2305.19442)

    SimFBO和其ShroFBO变体提出了一个简单、灵活且通信高效的FBO框架，可以应用于元学习和超参数优化任务。

    

    近来，由于元学习、微调、超参数调整等领域中嵌套优化结构的出现，联邦双层优化（FBO）在机器学习和边缘计算中显示了巨大的潜力。然而，现有的FBO算法往往涉及复杂的计算，并需要每次迭代多个子循环，每个子循环包含多个通信轮。在本文中，我们提出了一个名为SimFBO的简单灵活的FBO框架，它易于实现，不需要子循环，并包括一种广义的服务器端聚合和更新以提高通信效率。我们进一步提出了系统级异构鲁棒FBO（ShroFBO）作为SimFBO的变体，其对本地计算的异构有更强的鲁棒性。我们证明了在部分客户端参与和无替换的客户端采样下，SimFBO和ShroFBO可以实现线性收敛加速，同时改进了样本和通信复杂度。实验证明了它们在图像分类数据集的元学习和真实世界数据集上的超参数优化任务中的有效性。

    Federated bilevel optimization (FBO) has shown great potential recently in machine learning and edge computing due to the emerging nested optimization structure in meta-learning, fine-tuning, hyperparameter tuning, etc. However, existing FBO algorithms often involve complicated computations and require multiple sub-loops per iteration, each of which contains a number of communication rounds. In this paper, we propose a simple and flexible FBO framework named SimFBO, which is easy to implement without sub-loops, and includes a generalized server-side aggregation and update for improving communication efficiency. We further propose System-level heterogeneity robust FBO (ShroFBO) as a variant of SimFBO with stronger resilience to heterogeneous local computation. We show that SimFBO and ShroFBO provably achieve a linear convergence speedup with partial client participation and client sampling without replacement, as well as improved sample and communication complexities. Experiments demons
    
[^5]: 统一GAN和基于分数扩散的粒子生成模型

    Unifying GANs and Score-Based Diffusion as Generative Particle Models. (arXiv:2305.16150v1 [cs.LG])

    [http://arxiv.org/abs/2305.16150](http://arxiv.org/abs/2305.16150)

    本文提出了一个新框架，将生成器训练作为粒子模型的一个推广，从而统一了粒子和对抗生成模型。这个框架可以将生成器集成到基于分数扩散模型中，并在没有生成器的情况下训练GAN。

    

    基于粒子的深度生成模型，例如梯度流和基于分数的扩散模型，由于其惊人的性能而最近受到关注。传统上，通过微分方程来移动粒子分布的方法被普遍认为是与以前广泛使用的生成对抗网络（GAN）相对立的，后者涉及到训练一个向前的生成器网络。在本文中，我们质疑这种解释，并提出了一个统一粒子和对抗生成模型的新框架，通过将生成器训练作为粒子模型的推广。这表明，生成器是任何这样的生成模型的可选附件。因此，将生成器集成到基于分数扩散模型中，并在没有生成器的情况下训练GAN自然地出现在我们的框架中。我们通过实证测试这些原始模型的可行性，这些模型是我们框架可能应用的概念证明。

    Particle-based deep generative models, such as gradient flows and score-based diffusion models, have recently gained traction thanks to their striking performance. Their principle of displacing particle distributions by differential equations is conventionally seen as opposed to the previously widespread generative adversarial networks (GANs), which involve training a pushforward generator network. In this paper, we challenge this interpretation and propose a novel framework that unifies particle and adversarial generative models by framing generator training as a generalization of particle models. This suggests that a generator is an optional addition to any such generative model. Consequently, integrating a generator into a score-based diffusion model and training a GAN without a generator naturally emerge from our framework. We empirically test the viability of these original models as proofs of concepts of potential applications of our framework.
    
[^6]: 动量匹配去噪Gibbs采样

    Moment Matching Denoising Gibbs Sampling. (arXiv:2305.11650v1 [stat.ML])

    [http://arxiv.org/abs/2305.11650](http://arxiv.org/abs/2305.11650)

    本文提出了动量匹配去噪Gibbs采样方法，可以在给定‘嘈杂’的模型的情况下，从干净的模型中有效地进行采样。

    

    能量基模型（EBMs）为建模复杂数据分布提供了一个通用的框架。然而，EBMs 的训练和采样仍然面临重大挑战。用于可扩展 EBM 训练的广泛使用的去噪分数匹配（DSM）方法存在不一致性问题，导致能量模型学习到“嘈杂”的数据分布。在本文中，我们提出了一种有效的采样框架：（伪）Gibbs采样与动量匹配，可以在给定经过DSM训练良好的“嘈杂”模型的情况下，从基础“干净”模型中有效地进行采样。我们探讨了我们的方法相对于相关方法的优势，并展示了如何将该方法扩展到高维数据集。

    Energy-Based Models (EBMs) offer a versatile framework for modeling complex data distributions. However, training and sampling from EBMs continue to pose significant challenges. The widely-used Denoising Score Matching (DSM) method for scalable EBM training suffers from inconsistency issues, causing the energy model to learn a `noisy' data distribution. In this work, we propose an efficient sampling framework: (pseudo)-Gibbs sampling with moment matching, which enables effective sampling from the underlying clean model when given a `noisy' model that has been well-trained via DSM. We explore the benefits of our approach compared to related methods and demonstrate how to scale the method to high-dimensional datasets.
    
[^7]: 解释约束下的学习

    Learning with Explanation Constraints. (arXiv:2303.14496v1 [cs.LG])

    [http://arxiv.org/abs/2303.14496](http://arxiv.org/abs/2303.14496)

    本文研究了解释约束下的学习问题，提出了EPAC模型，探讨了使用这些解释时模型的益处，并提供了一种基于变分近似的算法解决方案。

    

    尽管监督学习假设存在标注数据，但我们可能有关于模型应如何运行的先验信息。本文将其形式化为从解释约束中学习，并提供了一个学习理论框架，分析了这些解释如何提高模型的学习能力。本文的第一项关键贡献是通过定义我们称之为EPAC模型（在新数据期望中满足这些约束的模型）来回答哪些模型会受益于解释这一问题。我们使用标准的学习理论工具分析了这类模型。第二个关键贡献是对于由线性模型和两层神经网络的梯度信息给出的规范解释的限制（以其Rademacher复杂度为衡量标准）进行了表征。最后，我们通过一种变分近似提供了我们的框架的算法解决方案，它能够实现更好的性能并满足这些约束。

    While supervised learning assumes the presence of labeled data, we may have prior information about how models should behave. In this paper, we formalize this notion as learning from explanation constraints and provide a learning theoretic framework to analyze how such explanations can improve the learning of our models. For what models would explanations be helpful? Our first key contribution addresses this question via the definition of what we call EPAC models (models that satisfy these constraints in expectation over new data), and we analyze this class of models using standard learning theoretic tools. Our second key contribution is to characterize these restrictions (in terms of their Rademacher complexities) for a canonical class of explanations given by gradient information for linear models and two layer neural networks. Finally, we provide an algorithmic solution for our framework, via a variational approximation that achieves better performance and satisfies these constraint
    

