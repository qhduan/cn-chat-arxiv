# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-agent deep reinforcement learning with centralized training and decentralized execution for transportation infrastructure management.](http://arxiv.org/abs/2401.12455) | 这项研究提出了一种基于集中训练和分散执行的多Agent深度强化学习框架，用于管理交通基础设施系统的整个生命周期，在处理高维度空间中的不确定性和约束条件时能够降低长期风险和成本。 |
| [^2] | [Improving Denoising Diffusion Models via Simultaneous Estimation of Image and Noise.](http://arxiv.org/abs/2310.17167) | 通过重新参数化扩散过程并直接估计图像和噪声，本文改进了去噪扩散模型，提高了图像生成的速度和质量。 |

# 详细

[^1]: 基于集中训练和分散执行的多Agent深度强化学习在交通基础设施管理中的应用

    Multi-agent deep reinforcement learning with centralized training and decentralized execution for transportation infrastructure management. (arXiv:2401.12455v1 [cs.MA])

    [http://arxiv.org/abs/2401.12455](http://arxiv.org/abs/2401.12455)

    这项研究提出了一种基于集中训练和分散执行的多Agent深度强化学习框架，用于管理交通基础设施系统的整个生命周期，在处理高维度空间中的不确定性和约束条件时能够降低长期风险和成本。

    

    我们提出了一种多Agent深度强化学习框架，用于在交通基础设施的整个生命周期内进行管理。这种工程系统的生命周期管理是一个需要大量计算的任务，需要适当的顺序检查和维护决策，能够在处理不同的不确定性和约束条件时降低长期风险和成本，这些不确定性和约束条件存在于高维空间中。到目前为止，静态的基于年龄或条件的维护方法和基于风险或定期检查计划主要解决了这类优化问题。然而，在这些方法下，优化性、可扩展性和不确定性限制经常显现出来。本工作中的优化问题以约束的部分可观察马尔可夫决策过程(POMDPs)框架为基础，为具有观察不确定性、风险考虑和随机顺序决策的问题提供了综合的数学基础。

    We present a multi-agent Deep Reinforcement Learning (DRL) framework for managing large transportation infrastructure systems over their life-cycle. Life-cycle management of such engineering systems is a computationally intensive task, requiring appropriate sequential inspection and maintenance decisions able to reduce long-term risks and costs, while dealing with different uncertainties and constraints that lie in high-dimensional spaces. To date, static age- or condition-based maintenance methods and risk-based or periodic inspection plans have mostly addressed this class of optimization problems. However, optimality, scalability, and uncertainty limitations are often manifested under such approaches. The optimization problem in this work is cast in the framework of constrained Partially Observable Markov Decision Processes (POMDPs), which provides a comprehensive mathematical basis for stochastic sequential decision settings with observation uncertainties, risk considerations, and l
    
[^2]: 通过同时估计图像和噪声改进去噪扩散模型

    Improving Denoising Diffusion Models via Simultaneous Estimation of Image and Noise. (arXiv:2310.17167v1 [cs.LG])

    [http://arxiv.org/abs/2310.17167](http://arxiv.org/abs/2310.17167)

    通过重新参数化扩散过程并直接估计图像和噪声，本文改进了去噪扩散模型，提高了图像生成的速度和质量。

    

    本文介绍了两个关键的贡献，旨在通过反向扩散过程生成的图像的速度和质量。第一个贡献是通过以图像和噪声之间的四分之一圆弧上的角度重新参数化扩散过程，特别是设置传统的 $\displaystyle \sqrt{\bar{\alpha}}=\cos(\eta)$。这种重新参数化消除了两个奇异点，并允许将扩散演化表达为一个良好行为的常微分方程（ODE）。从而，可以有效地使用更高阶的ODE求解器，如Runge-Kutta方法。第二个贡献是直接使用我们的网络估计图像（$\mathbf{x}_0$）和噪声（$\mathbf{\epsilon}$），这使得逆向扩散过程中的更新步骤计算更加稳定，因为在过程的不同阶段准确估计图像和噪声都是至关重要的。在这些变化的基础上，我们的模型实现了...

    This paper introduces two key contributions aimed at improving the speed and quality of images generated through inverse diffusion processes. The first contribution involves reparameterizing the diffusion process in terms of the angle on a quarter-circular arc between the image and noise, specifically setting the conventional $\displaystyle \sqrt{\bar{\alpha}}=\cos(\eta)$. This reparameterization eliminates two singularities and allows for the expression of diffusion evolution as a well-behaved ordinary differential equation (ODE). In turn, this allows higher order ODE solvers such as Runge-Kutta methods to be used effectively. The second contribution is to directly estimate both the image ($\mathbf{x}_0$) and noise ($\mathbf{\epsilon}$) using our network, which enables more stable calculations of the update step in the inverse diffusion steps, as accurate estimation of both the image and noise are crucial at different stages of the process. Together with these changes, our model achie
    

