# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Incentive-Aware Synthetic Control: Accurate Counterfactual Estimation via Incentivized Exploration](https://arxiv.org/abs/2312.16307) | 本论文提出了一种为了解决合成对照方法中"重叠"假设的问题的激励感知合成对照方法。该方法通过激励单位采取通常不会考虑的干预措施，提供与激励相容的干预建议，从而实现在面板数据环境中准确估计反事实效果。 |
| [^2] | [Multi-agent deep reinforcement learning with centralized training and decentralized execution for transportation infrastructure management.](http://arxiv.org/abs/2401.12455) | 这项研究提出了一种基于集中训练和分散执行的多Agent深度强化学习框架，用于管理交通基础设施系统的整个生命周期，在处理高维度空间中的不确定性和约束条件时能够降低长期风险和成本。 |
| [^3] | [Improving Denoising Diffusion Models via Simultaneous Estimation of Image and Noise.](http://arxiv.org/abs/2310.17167) | 通过重新参数化扩散过程并直接估计图像和噪声，本文改进了去噪扩散模型，提高了图像生成的速度和质量。 |
| [^4] | [The Dark Side of ChatGPT: Legal and Ethical Challenges from Stochastic Parrots and Hallucination.](http://arxiv.org/abs/2304.14347) | ChatGPT带来的大语言模型(LLMs)虽然有很多优势，但是随机鹦鹉和幻觉等新的法律和伦理风险也随之而来。欧洲AI监管范式需要进一步发展以减轻这些风险。 |

# 详细

[^1]: 激励感知合成对照方法：通过激励探索进行准确的反事实估计

    Incentive-Aware Synthetic Control: Accurate Counterfactual Estimation via Incentivized Exploration

    [https://arxiv.org/abs/2312.16307](https://arxiv.org/abs/2312.16307)

    本论文提出了一种为了解决合成对照方法中"重叠"假设的问题的激励感知合成对照方法。该方法通过激励单位采取通常不会考虑的干预措施，提供与激励相容的干预建议，从而实现在面板数据环境中准确估计反事实效果。

    

    我们考虑合成对照方法（SCMs）的设定，这是一种在面板数据环境中估计被治疗对象的治疗效应的经典方法。我们揭示了SCMs中经常被忽视但普遍存在的“重叠”假设：一个被治疗的单位可以被写成保持控制的单位的某种组合（通常是凸或线性组合）。我们展示了如果单位选择自己的干预措施，并且单位之间的异质性足够大，以至于他们偏好不同的干预措施，重叠将不成立。为了解决这个问题，我们提出了一个框架，通过激励具有不同偏好的单位来采取他们通常不会考虑的干预措施。具体来说，我们利用信息设计和在线学习的工具，提出了一种SCM，通过为单位提供与激励相容的干预建议，在面板数据环境中激励探索。

    arXiv:2312.16307v2 Announce Type: replace-cross Abstract: We consider the setting of synthetic control methods (SCMs), a canonical approach used to estimate the treatment effect on the treated in a panel data setting. We shed light on a frequently overlooked but ubiquitous assumption made in SCMs of "overlap": a treated unit can be written as some combination -- typically, convex or linear combination -- of the units that remain under control. We show that if units select their own interventions, and there is sufficiently large heterogeneity between units that prefer different interventions, overlap will not hold. We address this issue by proposing a framework which incentivizes units with different preferences to take interventions they would not normally consider. Specifically, leveraging tools from information design and online learning, we propose a SCM that incentivizes exploration in panel data settings by providing incentive-compatible intervention recommendations to units. We e
    
[^2]: 基于集中训练和分散执行的多Agent深度强化学习在交通基础设施管理中的应用

    Multi-agent deep reinforcement learning with centralized training and decentralized execution for transportation infrastructure management. (arXiv:2401.12455v1 [cs.MA])

    [http://arxiv.org/abs/2401.12455](http://arxiv.org/abs/2401.12455)

    这项研究提出了一种基于集中训练和分散执行的多Agent深度强化学习框架，用于管理交通基础设施系统的整个生命周期，在处理高维度空间中的不确定性和约束条件时能够降低长期风险和成本。

    

    我们提出了一种多Agent深度强化学习框架，用于在交通基础设施的整个生命周期内进行管理。这种工程系统的生命周期管理是一个需要大量计算的任务，需要适当的顺序检查和维护决策，能够在处理不同的不确定性和约束条件时降低长期风险和成本，这些不确定性和约束条件存在于高维空间中。到目前为止，静态的基于年龄或条件的维护方法和基于风险或定期检查计划主要解决了这类优化问题。然而，在这些方法下，优化性、可扩展性和不确定性限制经常显现出来。本工作中的优化问题以约束的部分可观察马尔可夫决策过程(POMDPs)框架为基础，为具有观察不确定性、风险考虑和随机顺序决策的问题提供了综合的数学基础。

    We present a multi-agent Deep Reinforcement Learning (DRL) framework for managing large transportation infrastructure systems over their life-cycle. Life-cycle management of such engineering systems is a computationally intensive task, requiring appropriate sequential inspection and maintenance decisions able to reduce long-term risks and costs, while dealing with different uncertainties and constraints that lie in high-dimensional spaces. To date, static age- or condition-based maintenance methods and risk-based or periodic inspection plans have mostly addressed this class of optimization problems. However, optimality, scalability, and uncertainty limitations are often manifested under such approaches. The optimization problem in this work is cast in the framework of constrained Partially Observable Markov Decision Processes (POMDPs), which provides a comprehensive mathematical basis for stochastic sequential decision settings with observation uncertainties, risk considerations, and l
    
[^3]: 通过同时估计图像和噪声改进去噪扩散模型

    Improving Denoising Diffusion Models via Simultaneous Estimation of Image and Noise. (arXiv:2310.17167v1 [cs.LG])

    [http://arxiv.org/abs/2310.17167](http://arxiv.org/abs/2310.17167)

    通过重新参数化扩散过程并直接估计图像和噪声，本文改进了去噪扩散模型，提高了图像生成的速度和质量。

    

    本文介绍了两个关键的贡献，旨在通过反向扩散过程生成的图像的速度和质量。第一个贡献是通过以图像和噪声之间的四分之一圆弧上的角度重新参数化扩散过程，特别是设置传统的 $\displaystyle \sqrt{\bar{\alpha}}=\cos(\eta)$。这种重新参数化消除了两个奇异点，并允许将扩散演化表达为一个良好行为的常微分方程（ODE）。从而，可以有效地使用更高阶的ODE求解器，如Runge-Kutta方法。第二个贡献是直接使用我们的网络估计图像（$\mathbf{x}_0$）和噪声（$\mathbf{\epsilon}$），这使得逆向扩散过程中的更新步骤计算更加稳定，因为在过程的不同阶段准确估计图像和噪声都是至关重要的。在这些变化的基础上，我们的模型实现了...

    This paper introduces two key contributions aimed at improving the speed and quality of images generated through inverse diffusion processes. The first contribution involves reparameterizing the diffusion process in terms of the angle on a quarter-circular arc between the image and noise, specifically setting the conventional $\displaystyle \sqrt{\bar{\alpha}}=\cos(\eta)$. This reparameterization eliminates two singularities and allows for the expression of diffusion evolution as a well-behaved ordinary differential equation (ODE). In turn, this allows higher order ODE solvers such as Runge-Kutta methods to be used effectively. The second contribution is to directly estimate both the image ($\mathbf{x}_0$) and noise ($\mathbf{\epsilon}$) using our network, which enables more stable calculations of the update step in the inverse diffusion steps, as accurate estimation of both the image and noise are crucial at different stages of the process. Together with these changes, our model achie
    
[^4]: ChatGPT的黑暗面：来自随机鹦鹉和幻觉的法律和伦理挑战

    The Dark Side of ChatGPT: Legal and Ethical Challenges from Stochastic Parrots and Hallucination. (arXiv:2304.14347v1 [cs.CY])

    [http://arxiv.org/abs/2304.14347](http://arxiv.org/abs/2304.14347)

    ChatGPT带来的大语言模型(LLMs)虽然有很多优势，但是随机鹦鹉和幻觉等新的法律和伦理风险也随之而来。欧洲AI监管范式需要进一步发展以减轻这些风险。

    

    随着ChatGPT的推出，大语言模型（LLMs）正在动摇我们整个社会，快速改变我们的思维、创造和生活方式。然而，随着随机鹦鹉和幻觉等新的法律和伦理风险出现，新兴LLMs也带来了许多挑战。欧盟是第一个将重点放在AI模型监管上的司法管辖区。然而，新LLMs带来的风险可能会被新兴的欧盟监管范式所低估。因此，本函告警示欧洲AI监管范式必须进一步发展以减轻这些风险。

    With the launch of ChatGPT, Large Language Models (LLMs) are shaking up our whole society, rapidly altering the way we think, create and live. For instance, the GPT integration in Bing has altered our approach to online searching. While nascent LLMs have many advantages, new legal and ethical risks are also emerging, stemming in particular from stochastic parrots and hallucination. The EU is the first and foremost jurisdiction that has focused on the regulation of AI models. However, the risks posed by the new LLMs are likely to be underestimated by the emerging EU regulatory paradigm. Therefore, this correspondence warns that the European AI regulatory paradigm must evolve further to mitigate such risks.
    

