# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Deep Reinforcement Learning Through Adversarial Attacks and Training : A Survey](https://arxiv.org/abs/2403.00420) | 通过对抗性训练来改进DRL对条件变化的鲁棒性，研究者系统分析了当代对抗攻击方法，提供了详细见解。 |
| [^2] | [Deeper or Wider: A Perspective from Optimal Generalization Error with Sobolev Loss](https://arxiv.org/abs/2402.00152) | 本文比较了更深的神经网络和更宽的神经网络在Sobolev损失的最优泛化误差方面的表现，研究发现神经网络的架构受多种因素影响，参数数量更多倾向于选择更宽的网络，而样本点数量和损失函数规则性更高倾向于选择更深的网络。 |
| [^3] | [The Emergence of Reproducibility and Consistency in Diffusion Models](https://arxiv.org/abs/2310.05264) | 该论文研究了扩散模型中的一致模型可重复性现象，实验证实了无论模型框架、模型架构或训练过程如何，不同的扩散模型都能够一致地达到相同的数据分布和评分函数。此外，研究发现扩散模型在学习过程中受训练数据规模的影响，表现出两种不同的训练模式：记忆化模式和泛化模式。 |

# 详细

[^1]: 经由对抗攻击和训练的稳健深度强化学习：一项调查

    Robust Deep Reinforcement Learning Through Adversarial Attacks and Training : A Survey

    [https://arxiv.org/abs/2403.00420](https://arxiv.org/abs/2403.00420)

    通过对抗性训练来改进DRL对条件变化的鲁棒性，研究者系统分析了当代对抗攻击方法，提供了详细见解。

    

    深度强化学习（DRL）是一种训练自主代理在各种复杂环境中的方法。尽管在众所周知的环境中表现出色，但它仍然容易受到轻微条件变化的影响，引发了人们对其在现实应用中可靠性的担忧。为了提高可用性，DRL必须展示出可信度和鲁棒性。通过对抗性训练提高DRL对条件变化的鲁棒性是一种改进方式，通过训练代理针对环境动态的适当对抗性攻击。我们的工作致力于解决这一关键问题，对当代对抗攻击方法进行了深入分析，系统地对其进行分类，并比较它们的目标和操作机制。这种分类为我们提供了对对抗性攻击如何有效评估DRL代理的恢复力的详细见解，从而为开辟DRL在实际应用中的道路奠定了基础。

    arXiv:2403.00420v1 Announce Type: cross  Abstract: Deep Reinforcement Learning (DRL) is an approach for training autonomous agents across various complex environments. Despite its significant performance in well known environments, it remains susceptible to minor conditions variations, raising concerns about its reliability in real-world applications. To improve usability, DRL must demonstrate trustworthiness and robustness. A way to improve robustness of DRL to unknown changes in the conditions is through Adversarial Training, by training the agent against well suited adversarial attacks on the dynamics of the environment. Addressing this critical issue, our work presents an in-depth analysis of contemporary adversarial attack methodologies, systematically categorizing them and comparing their objectives and operational mechanisms. This classification offers a detailed insight into how adversarial attacks effectively act for evaluating the resilience of DRL agents, thereby paving the 
    
[^2]: 更深还是更宽: 从Sobolev损失的最优泛化误差角度看

    Deeper or Wider: A Perspective from Optimal Generalization Error with Sobolev Loss

    [https://arxiv.org/abs/2402.00152](https://arxiv.org/abs/2402.00152)

    本文比较了更深的神经网络和更宽的神经网络在Sobolev损失的最优泛化误差方面的表现，研究发现神经网络的架构受多种因素影响，参数数量更多倾向于选择更宽的网络，而样本点数量和损失函数规则性更高倾向于选择更深的网络。

    

    构建神经网络的架构是机器学习界一个具有挑战性的追求，到底是更深还是更宽一直是一个持续的问题。本文探索了更深的神经网络（DeNNs）和具有有限隐藏层的更宽的神经网络（WeNNs）在Sobolev损失的最优泛化误差方面的比较。通过分析研究发现，神经网络的架构可以受到多种因素的显著影响，包括样本点的数量，神经网络内的参数以及损失函数的规则性。具体而言，更多的参数倾向于选择WeNNs，而更多的样本点和更高的损失函数规则性倾向于选择DeNNs。最后，我们将这个理论应用于使用深度Ritz和物理感知神经网络（PINN）方法解决偏微分方程的问题。

    Constructing the architecture of a neural network is a challenging pursuit for the machine learning community, and the dilemma of whether to go deeper or wider remains a persistent question. This paper explores a comparison between deeper neural networks (DeNNs) with a flexible number of layers and wider neural networks (WeNNs) with limited hidden layers, focusing on their optimal generalization error in Sobolev losses. Analytical investigations reveal that the architecture of a neural network can be significantly influenced by various factors, including the number of sample points, parameters within the neural networks, and the regularity of the loss function. Specifically, a higher number of parameters tends to favor WeNNs, while an increased number of sample points and greater regularity in the loss function lean towards the adoption of DeNNs. We ultimately apply this theory to address partial differential equations using deep Ritz and physics-informed neural network (PINN) methods,
    
[^3]: 扩散模型中的可重复性和一致性的出现

    The Emergence of Reproducibility and Consistency in Diffusion Models

    [https://arxiv.org/abs/2310.05264](https://arxiv.org/abs/2310.05264)

    该论文研究了扩散模型中的一致模型可重复性现象，实验证实了无论模型框架、模型架构或训练过程如何，不同的扩散模型都能够一致地达到相同的数据分布和评分函数。此外，研究发现扩散模型在学习过程中受训练数据规模的影响，表现出两种不同的训练模式：记忆化模式和泛化模式。

    

    在这项工作中，我们研究了扩散模型中的一个有趣且普遍存在的现象，我们称之为“一致的模型可重复性”：在给定相同的起始噪声输入和确定性采样器的情况下，不同的扩散模型通常产生非常相似的输出。我们通过全面的实验证实了这一现象，表明不同的扩散模型无论扩散模型框架、模型架构或训练过程如何，在数据分布和评分函数上都能够一致地达到相同的结果。更令人惊讶的是，我们进一步的调查表明，扩散模型在学习受训数据规模影响下的不同分布。这一点得到了两种不同训练模式下模型可重复性的体现：（i）“记忆化模式”，其中扩散模型过度拟合于训练数据分布，和（ii）“泛化模式”，其中模型学习到了基础数据分布。

    arXiv:2310.05264v2 Announce Type: replace  Abstract: In this work, we investigate an intriguing and prevalent phenomenon of diffusion models which we term as "consistent model reproducibility": given the same starting noise input and a deterministic sampler, different diffusion models often yield remarkably similar outputs. We confirm this phenomenon through comprehensive experiments, implying that different diffusion models consistently reach the same data distribution and scoring function regardless of diffusion model frameworks, model architectures, or training procedures. More strikingly, our further investigation implies that diffusion models are learning distinct distributions affected by the training data size. This is supported by the fact that the model reproducibility manifests in two distinct training regimes: (i) "memorization regime", where the diffusion model overfits to the training data distribution, and (ii) "generalization regime", where the model learns the underlyin
    

