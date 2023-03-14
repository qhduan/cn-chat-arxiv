# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey on Automated Design of Metaheuristic Algorithms.](http://arxiv.org/abs/2303.06532) | 本文综述了自动设计元启发式算法的形式化、方法论、挑战和研究趋势，讨论了自动设计的潜在未来方向和开放问题。 |
| [^2] | [One Neuron Saved Is One Neuron Earned: On Parametric Efficiency of Quadratic Networks.](http://arxiv.org/abs/2303.06316) | 本文研究了二次神经元的参数效率，证明了其卓越性能是由于内在表达能力而非参数增加。 |
| [^3] | [Understanding the Synergies between Quality-Diversity and Deep Reinforcement Learning.](http://arxiv.org/abs/2303.06164) | 本文提出了广义演员-评论家QD-RL框架，用于QD-RL设置中的演员-评论家深度RL方法。该框架引入了两种新算法，PGA-ME（SAC）和PGA-ME（DroQ），将深度RL的最新进展应用于QD-RL设置，并解决了现有QD-RL算法无法解决的人形环境问题。 |
| [^4] | [Towards NeuroAI: Introducing Neuronal Diversity into Artificial Neural Networks.](http://arxiv.org/abs/2301.09245) | 引入神经元多样性可以解决人工神经网络的基本问题，走向神经人工智能。 |
| [^5] | [Classification and Generation of real-world data with an Associative Memory Model.](http://arxiv.org/abs/2207.04827) | 本文提出了一种基于联想记忆模型的多模态框架，可以以容错的方式存储和检索大量真实世界数据，并且可以用于推断缺失的模态。 |
| [^6] | [A comprehensive review of Binary Neural Network.](http://arxiv.org/abs/2110.06804) | 本文全面综述了二进制神经网络的最新发展，重点关注1位激活和1位卷积网络的权重，这些网络可以在微小的受限设备上实现和嵌入，并节省大量存储、计算成本和能量消耗。 |

# 详细

[^1]: 自动设计元启发式算法的综述

    A Survey on Automated Design of Metaheuristic Algorithms. (arXiv:2303.06532v1 [cs.NE])

    [http://arxiv.org/abs/2303.06532](http://arxiv.org/abs/2303.06532)

    本文综述了自动设计元启发式算法的形式化、方法论、挑战和研究趋势，讨论了自动设计的潜在未来方向和开放问题。

    This paper presents a broad picture of the formalization, methodologies, challenges, and research trends of automated design of metaheuristic algorithms, and discusses the potential future directions and open issues in this field.

    元启发式算法由于其能够独立于问题结构和问题领域进行搜索的能力，已经引起了学术界和工业界的广泛关注。通常，需要人类专家手动调整算法以适应解决目标问题。手动调整过程可能是费力的、容易出错的，并且需要大量的专业知识。这引起了对自动设计元启发式算法的越来越多的兴趣和需求，以减少人类干预。自动设计可以使高性能算法对更广泛的研究人员和实践者可用；通过利用计算能力来充分探索潜在的设计选择，自动设计可以达到甚至超过人类水平的设计。本文通过对现有工作的共同点和差异进行调查，提出了自动设计元启发式算法的形式化、方法论、挑战和研究趋势的广泛概述。我们还讨论了这一领域的潜在未来方向和开放问题。

    Metaheuristic algorithms have attracted wide attention from academia and industry due to their capability of conducting search independent of problem structures and problem domains. Often, human experts are requested to manually tailor algorithms to fit for solving a targeted problem. The manual tailoring process may be laborious, error-prone, and require intensive specialized knowledge. This gives rise to increasing interests and demands for automated design of metaheuristic algorithms with less human intervention. The automated design could make high-performance algorithms accessible to a much broader range of researchers and practitioners; and by leveraging computing power to fully explore the potential design choices, automated design could reach or even surpass human-level design. This paper presents a broad picture of the formalization, methodologies, challenges, and research trends of automated design of metaheuristic algorithms, by conducting a survey on the common grounds and 
    
[^2]: 一个神经元的节省就是一个神经元的收益：关于二次网络参数效率的研究

    One Neuron Saved Is One Neuron Earned: On Parametric Efficiency of Quadratic Networks. (arXiv:2303.06316v1 [cs.LG])

    [http://arxiv.org/abs/2303.06316](http://arxiv.org/abs/2303.06316)

    本文研究了二次神经元的参数效率，证明了其卓越性能是由于内在表达能力而非参数增加。

    This paper studies the parametric efficiency of quadratic neurons and confirms that their superior performance is due to intrinsic expressive capability rather than increased parameters.

    受生物神经系统中神经元多样性的启发，大量研究提出了设计新型人工神经元并将神经元多样性引入人工神经网络的方法。最近提出的二次神经元，将传统神经元中的内积操作替换为二次操作，在许多重要任务中取得了巨大成功。尽管二次神经元的结果很有前途，但仍存在一个未解决的问题：二次网络的卓越性能仅仅是由于参数增加还是由于内在表达能力？在未澄清这个问题的情况下，二次网络的性能总是令人怀疑。此外，解决这个问题就是找到二次网络的杀手应用。在本文中，通过理论和实证研究，我们展示了二次网络具有参数效率，从而确认了二次网络的卓越性能是由于其内在表达能力而非参数增加。

    Inspired by neuronal diversity in the biological neural system, a plethora of studies proposed to design novel types of artificial neurons and introduce neuronal diversity into artificial neural networks. Recently proposed quadratic neuron, which replaces the inner-product operation in conventional neurons with a quadratic one, have achieved great success in many essential tasks. Despite the promising results of quadratic neurons, there is still an unresolved issue: \textit{Is the superior performance of quadratic networks simply due to the increased parameters or due to the intrinsic expressive capability?} Without clarifying this issue, the performance of quadratic networks is always suspicious. Additionally, resolving this issue is reduced to finding killer applications of quadratic networks. In this paper, with theoretical and empirical studies, we show that quadratic networks enjoy parametric efficiency, thereby confirming that the superior performance of quadratic networks is due
    
[^3]: 理解质量多样性和深度强化学习之间的协同作用

    Understanding the Synergies between Quality-Diversity and Deep Reinforcement Learning. (arXiv:2303.06164v1 [cs.LG])

    [http://arxiv.org/abs/2303.06164](http://arxiv.org/abs/2303.06164)

    本文提出了广义演员-评论家QD-RL框架，用于QD-RL设置中的演员-评论家深度RL方法。该框架引入了两种新算法，PGA-ME（SAC）和PGA-ME（DroQ），将深度RL的最新进展应用于QD-RL设置，并解决了现有QD-RL算法无法解决的人形环境问题。

    This paper proposes a Generalized Actor-Critic QD-RL framework for actor-critic deep RL methods in the QD-RL setting. The framework introduces two new algorithms, PGA-ME (SAC) and PGA-ME (DroQ), which apply recent advancements in Deep RL to the QD-RL setting and solve the humanoid environment problem that existing QD-RL algorithms cannot solve.

    质量多样性（QD）和深度强化学习（RL）之间的协同作用已经导致了强大的混合QD-RL算法，展示了巨大的潜力，并带来了两个领域的最佳实践。然而，尽管其他RL算法取得了显著进展，但在先前的混合方法中仅使用了单个深度RL算法（TD3）。此外，QD和RL之间的优化过程存在根本差异，需要更加原则性的方法。我们提出了广义演员-评论家QD-RL，这是一个统一的模块化框架，用于QD-RL设置中的演员-评论家深度RL方法。该框架提供了一条研究深度RL在QD-RL设置中的见解的路径，这是在QD-RL中取得进展的重要且有效的方法。我们引入了两种新算法，PGA-ME（SAC）和PGA-ME（DroQ），将深度RL的最新进展应用于QD-RL设置，并解决了现有QD-RL算法无法解决的人形环境问题。

    The synergies between Quality-Diversity (QD) and Deep Reinforcement Learning (RL) have led to powerful hybrid QD-RL algorithms that have shown tremendous potential, and brings the best of both fields. However, only a single deep RL algorithm (TD3) has been used in prior hybrid methods despite notable progress made by other RL algorithms. Additionally, there are fundamental differences in the optimization procedures between QD and RL which would benefit from a more principled approach. We propose Generalized Actor-Critic QD-RL, a unified modular framework for actor-critic deep RL methods in the QD-RL setting. This framework provides a path to study insights from Deep RL in the QD-RL setting, which is an important and efficient way to make progress in QD-RL. We introduce two new algorithms, PGA-ME (SAC) and PGA-ME (DroQ) which apply recent advancements in Deep RL to the QD-RL setting, and solves the humanoid environment which was not possible using existing QD-RL algorithms. However, we 
    
[^4]: 走向神经人工智能：将神经元多样性引入人工神经网络

    Towards NeuroAI: Introducing Neuronal Diversity into Artificial Neural Networks. (arXiv:2301.09245v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2301.09245](http://arxiv.org/abs/2301.09245)

    引入神经元多样性可以解决人工神经网络的基本问题，走向神经人工智能。

    Introducing neuronal diversity can solve the fundamental problems of artificial neural networks and lead to NeuroAI.

    在整个历史上，人工智能的发展，特别是人工神经网络，一直对越来越深入的大脑理解持开放态度并不断受到启发，例如卷积神经网络的开创性工作neocognitron的启发。根据新兴领域神经人工智能的动机，大量的神经科学知识可以通过赋予网络更强大的能力来催化下一代人工智能的发展。我们知道，人类大脑有许多形态和功能不同的神经元，而人工神经网络几乎完全建立在单一神经元类型上。在人类大脑中，神经元多样性是各种生物智能行为的一个启动因素。由于人工网络是人类大脑的缩影，引入神经元多样性应该有助于解决人工网络的诸如效率、解释性等基本问题。

    Throughout history, the development of artificial intelligence, particularly artificial neural networks, has been open to and constantly inspired by the increasingly deepened understanding of the brain, such as the inspiration of neocognitron, which is the pioneering work of convolutional neural networks. Per the motives of the emerging field: NeuroAI, a great amount of neuroscience knowledge can help catalyze the next generation of AI by endowing a network with more powerful capabilities. As we know, the human brain has numerous morphologically and functionally different neurons, while artificial neural networks are almost exclusively built on a single neuron type. In the human brain, neuronal diversity is an enabling factor for all kinds of biological intelligent behaviors. Since an artificial network is a miniature of the human brain, introducing neuronal diversity should be valuable in terms of addressing those essential problems of artificial networks such as efficiency, interpret
    
[^5]: 基于联想记忆模型的真实世界数据分类和生成

    Classification and Generation of real-world data with an Associative Memory Model. (arXiv:2207.04827v3 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2207.04827](http://arxiv.org/abs/2207.04827)

    本文提出了一种基于联想记忆模型的多模态框架，可以以容错的方式存储和检索大量真实世界数据，并且可以用于推断缺失的模态。

    This paper proposes a multi-modality framework based on the associative memory model, which can store and retrieve a large amount of real-world data in a fault-tolerant manner, and can be used to infer missing modalities.

    回忆起多年未见的朋友的面孔是一项困难的任务。然而，如果你们偶然相遇，你们会轻易地认出彼此。生物记忆配备了一个令人印象深刻的压缩算法，可以存储必要的信息，然后推断细节以匹配感知。Willshaw Memory是一种用于皮层计算的简单抽象模型，实现了生物记忆的机制。使用我们最近提出的用于视觉模式的稀疏编码规则[34]，该模型可以以容错的方式存储和检索大量真实世界数据。在本文中，我们通过使用多模态框架扩展了基本联想记忆模型的能力。在这种设置中，记忆同时存储每个模式的几种模态（例如，视觉或文本）。训练后，当只感知到子集时，记忆可以用于推断缺失的模态。使用简单的编码器-记忆解码器，我们可以生成具有多个模态的数据。

    Drawing from memory the face of a friend you have not seen in years is a difficult task. However, if you happen to cross paths, you would easily recognize each other. The biological memory is equipped with an impressive compression algorithm that can store the essential, and then infer the details to match perception. The Willshaw Memory is a simple abstract model for cortical computations which implements mechanisms of biological memories. Using our recently proposed sparse coding prescription for visual patterns [34], this model can store and retrieve an impressive amount of real-world data in a fault-tolerant manner. In this paper, we extend the capabilities of the basic Associative Memory Model by using a Multiple-Modality framework. In this setting, the memory stores several modalities (e.g., visual, or textual) of each pattern simultaneously. After training, the memory can be used to infer missing modalities when just a subset is perceived. Using a simple encoder-memory decoder a
    
[^6]: 二进制神经网络的全面综述

    A comprehensive review of Binary Neural Network. (arXiv:2110.06804v4 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2110.06804](http://arxiv.org/abs/2110.06804)

    本文全面综述了二进制神经网络的最新发展，重点关注1位激活和1位卷积网络的权重，这些网络可以在微小的受限设备上实现和嵌入，并节省大量存储、计算成本和能量消耗。

    This article provides a comprehensive overview of recent developments in Binary Neural Networks (BNN), with a focus on 1-bit activations and 1-bit convolution networks. These networks can be implemented and embedded on tiny restricted devices, saving significant storage, computation cost, and energy consumption.

    深度学习（DL）最近改变了智能系统的发展，并被广泛应用于许多实际应用中。尽管DL具有各种好处和潜力，但在不同的计算受限和能量受限设备中需要进行DL处理。研究二进制神经网络（BNN）等具有改变游戏规则的技术以增加深度学习能力是很自然的。最近在BNN方面取得了显着进展，因为它们可以在微小的受限设备上实现和嵌入，并节省大量存储、计算成本和能量消耗。然而，几乎所有的BNN行为都会带来额外的内存、计算成本和更高的性能。本文提供了BNN最近发展的完整概述。本文专门关注1位激活和1位卷积网络的权重，与以前的调查混合使用低位作品相反。它对BNN的开发进行了全面调查。

    Deep learning (DL) has recently changed the development of intelligent systems and is widely adopted in many real-life applications. Despite their various benefits and potentials, there is a high demand for DL processing in different computationally limited and energy-constrained devices. It is natural to study game-changing technologies such as Binary Neural Networks (BNN) to increase deep learning capabilities. Recently remarkable progress has been made in BNN since they can be implemented and embedded on tiny restricted devices and save a significant amount of storage, computation cost, and energy consumption. However, nearly all BNN acts trade with extra memory, computation cost, and higher performance. This article provides a complete overview of recent developments in BNN. This article focuses exclusively on 1-bit activations and weights 1-bit convolution networks, contrary to previous surveys in which low-bit works are mixed in. It conducted a complete investigation of BNN's dev
    

