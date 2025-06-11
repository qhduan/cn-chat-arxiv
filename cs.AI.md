# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structuring Concept Space with the Musical Circle of Fifths by Utilizing Music Grammar Based Activations](https://arxiv.org/abs/2403.00790) | 提出了一种利用音乐语法调节尖峰神经网络激活的新颖方法，通过应用音乐理论中的和弦进行规则，展示了如何自然地跟随其他激活，最终将概念的映射结构化为音乐五度圆。 |
| [^2] | [Innate-Values-driven Reinforcement Learning for Cooperative Multi-Agent Systems.](http://arxiv.org/abs/2401.05572) | 本文提出了一个先天价值驱动增强学习（IVRL）模型，用于描述多智能体在合作中的复杂行为。该模型通过建立智能体对群体效用和系统成本的认知，满足其合作伙伴的需求，支持其社区并融入人类社会。 |
| [^3] | [Framework for developing quantitative agent based models based on qualitative expert knowledge: an organised crime use-case.](http://arxiv.org/abs/2308.00505) | 提出了一个基于定性专家知识的量化代理模型开发框架，该框架通过将定性数据翻译成定量规则，为模型构建者和领域专家提供了一个系统和透明的建模过程。以一个有组织犯罪的应用案例为例，演示了该框架的方法。 |
| [^4] | [Stabilizing Contrastive RL: Techniques for Offline Goal Reaching.](http://arxiv.org/abs/2306.03346) | 本文提出了一种稳定的对比强化学习方法，通过浅而宽的结构，结合谨慎的权重初始化和数据增强等实验方法，在具有挑战性的仿真基准测试中显著提高了性能，并演示了对比方法可以解决现实世界的机器人任务。 |
| [^5] | [StereoVAE: A lightweight stereo matching system through embedded GPUs.](http://arxiv.org/abs/2305.11566) | 本论文提出了通过嵌入式GPU实现的轻量级立体匹配系统-StereoVAE，该系统采用基于VAE的小型神经网络对传统匹配方法生成的小尺寸粗糙视差图进行上采样与细化，达到了提高匹配精度和保证实时处理的目的。 |

# 详细

[^1]: 利用音乐五度圆构建概念空间：基于音乐语法激活的方法

    Structuring Concept Space with the Musical Circle of Fifths by Utilizing Music Grammar Based Activations

    [https://arxiv.org/abs/2403.00790](https://arxiv.org/abs/2403.00790)

    提出了一种利用音乐语法调节尖峰神经网络激活的新颖方法，通过应用音乐理论中的和弦进行规则，展示了如何自然地跟随其他激活，最终将概念的映射结构化为音乐五度圆。

    

    在本文中，我们探讨了离散神经网络（如尖峰网络）的结构与钢琴曲的构成之间的有趣相似之处。虽然两者都涉及按顺序或并行激活的节点或音符，但后者受益于丰富的音乐理论，以指导有意义的组合。我们提出了一种新颖的方法，利用音乐语法来调节尖峰神经网络中的激活，允许将符号表示为吸引子。通过应用音乐理论中的和弦进行规则，我们展示了某些激活如何自然地跟随其他激活，类似于吸引的概念。此外，我们引入了调制音调的概念，以在网络内导航不同的吸引盆地。最终，我们展示了我们模型中概念的映射是由音乐五度圆构成的，突出了利用音乐理论的潜力。

    arXiv:2403.00790v1 Announce Type: cross  Abstract: In this paper, we explore the intriguing similarities between the structure of a discrete neural network, such as a spiking network, and the composition of a piano piece. While both involve nodes or notes that are activated sequentially or in parallel, the latter benefits from the rich body of music theory to guide meaningful combinations. We propose a novel approach that leverages musical grammar to regulate activations in a spiking neural network, allowing for the representation of symbols as attractors. By applying rules for chord progressions from music theory, we demonstrate how certain activations naturally follow others, akin to the concept of attraction. Furthermore, we introduce the concept of modulating keys to navigate different basins of attraction within the network. Ultimately, we show that the map of concepts in our model is structured by the musical circle of fifths, highlighting the potential for leveraging music theor
    
[^2]: 用于合作多智能体系统的先天价值驱动增强学习

    Innate-Values-driven Reinforcement Learning for Cooperative Multi-Agent Systems. (arXiv:2401.05572v1 [cs.LG])

    [http://arxiv.org/abs/2401.05572](http://arxiv.org/abs/2401.05572)

    本文提出了一个先天价值驱动增强学习（IVRL）模型，用于描述多智能体在合作中的复杂行为。该模型通过建立智能体对群体效用和系统成本的认知，满足其合作伙伴的需求，支持其社区并融入人类社会。

    

    先天价值描述了智能体的内在动机，反映了他们追求目标和发展多样技能以满足各种需求的固有兴趣和偏好。强化学习的本质是基于奖励驱动（如效用）的行为互动学习，类似于自然智能体。特别是在多智能体系统中，建立智能体对平衡群体效用和系统成本的认知，满足群体成员在合作中的需求，是个体为支持其社区和融入人类社会而学习的一个关键问题。本文提出了一种分层复合内在价值增强学习模型 - 先天价值驱动增强学习，用于描述多智能体合作中复杂的互动行为。

    Innate values describe agents' intrinsic motivations, which reflect their inherent interests and preferences to pursue goals and drive them to develop diverse skills satisfying their various needs. The essence of reinforcement learning (RL) is learning from interaction based on reward-driven (such as utilities) behaviors, much like natural agents. It is an excellent model to describe the innate-values-driven (IV) behaviors of AI agents. Especially in multi-agent systems (MAS), building the awareness of AI agents to balance the group utilities and system costs and satisfy group members' needs in their cooperation is a crucial problem for individuals learning to support their community and integrate human society in the long term. This paper proposes a hierarchical compound intrinsic value reinforcement learning model -innate-values-driven reinforcement learning termed IVRL to describe the complex behaviors of multi-agent interaction in their cooperation. We implement the IVRL architec
    
[^3]: 基于定性专家知识的量化代理模型开发框架：一个有组织犯罪的应用案例

    Framework for developing quantitative agent based models based on qualitative expert knowledge: an organised crime use-case. (arXiv:2308.00505v1 [cs.AI])

    [http://arxiv.org/abs/2308.00505](http://arxiv.org/abs/2308.00505)

    提出了一个基于定性专家知识的量化代理模型开发框架，该框架通过将定性数据翻译成定量规则，为模型构建者和领域专家提供了一个系统和透明的建模过程。以一个有组织犯罪的应用案例为例，演示了该框架的方法。

    

    为了对执法目的建模犯罪网络，需要将有限的数据转化为经过验证的基于代理的模型。当前刑事学建模中缺少一个为模型构建者和领域专家提供系统和透明框架的方法，该方法建立了计算犯罪建模的建模过程，包括将定性数据转化为定量规则。因此，我们提出了FREIDA（基于专家知识驱动的数据驱动代理模型框架）。在本文中，犯罪可卡因替代模型（CCRM）将作为示例案例，以演示FREIDA方法。对于CCRM，正在建模荷兰的一个有组织可卡因网络，试图通过移除首脑节点，使剩余代理重新组织，并将网络恢复到稳定状态。定性数据源，例如案件文件，文献和采访，被转化为经验法则。

    In order to model criminal networks for law enforcement purposes, a limited supply of data needs to be translated into validated agent-based models. What is missing in current criminological modelling is a systematic and transparent framework for modelers and domain experts that establishes a modelling procedure for computational criminal modelling that includes translating qualitative data into quantitative rules. For this, we propose FREIDA (Framework for Expert-Informed Data-driven Agent-based models). Throughout the paper, the criminal cocaine replacement model (CCRM) will be used as an example case to demonstrate the FREIDA methodology. For the CCRM, a criminal cocaine network in the Netherlands is being modelled where the kingpin node is being removed, the goal being for the remaining agents to reorganize after the disruption and return the network into a stable state. Qualitative data sources such as case files, literature and interviews are translated into empirical laws, and c
    
[^4]: 稳定对比强化学习: 离线目标达成的技术

    Stabilizing Contrastive RL: Techniques for Offline Goal Reaching. (arXiv:2306.03346v1 [cs.LG])

    [http://arxiv.org/abs/2306.03346](http://arxiv.org/abs/2306.03346)

    本文提出了一种稳定的对比强化学习方法，通过浅而宽的结构，结合谨慎的权重初始化和数据增强等实验方法，在具有挑战性的仿真基准测试中显著提高了性能，并演示了对比方法可以解决现实世界的机器人任务。

    

    计算机视觉和自然语言处理领域已经开发了自监督方法，强化学习也可以被视为自监督问题：学习达到任何目标，而不需要人类指定的奖励或标签。然而，为强化学习建立自监督基础实际上面临着一些重要的挑战。基于此前对比学习方法，我们进行了细致的剖析实验，并发现一个浅而宽的结构，结合谨慎的权重初始化和数据增强，可以显着提高与对比强化学习方法的性能，特别是在具有挑战性的仿真基准测试中。此外，我们还演示了通过这些设计决策，对比方法可以解决现实世界的机器人操作任务，其中任务由训练后提供的单个目标图像指定。

    In the same way that the computer vision (CV) and natural language processing (NLP) communities have developed self-supervised methods, reinforcement learning (RL) can be cast as a self-supervised problem: learning to reach any goal, without requiring human-specified rewards or labels. However, actually building a self-supervised foundation for RL faces some important challenges. Building on prior contrastive approaches to this RL problem, we conduct careful ablation experiments and discover that a shallow and wide architecture, combined with careful weight initialization and data augmentation, can significantly boost the performance of these contrastive RL approaches on challenging simulated benchmarks. Additionally, we demonstrate that, with these design decisions, contrastive approaches can solve real-world robotic manipulation tasks, with tasks being specified by a single goal image provided after training.
    
[^5]: 通过嵌入式GPU实现的轻量级立体匹配系统-StereoVAE

    StereoVAE: A lightweight stereo matching system through embedded GPUs. (arXiv:2305.11566v1 [cs.CV])

    [http://arxiv.org/abs/2305.11566](http://arxiv.org/abs/2305.11566)

    本论文提出了通过嵌入式GPU实现的轻量级立体匹配系统-StereoVAE，该系统采用基于VAE的小型神经网络对传统匹配方法生成的小尺寸粗糙视差图进行上采样与细化，达到了提高匹配精度和保证实时处理的目的。

    

    本论文提出了通过嵌入式GPU实现的轻量级立体匹配系统-StereoVAE，它打破了立体匹配中精度和处理速度之间的平衡，使得我们的嵌入式系统能够在保证实时处理的同时进一步提高匹配精度。我们的方法的主要思想是构建一个基于变分自编码器（VAE）的小型神经网络，对传统匹配方法生成的小尺寸粗糙视差图进行上采样与细化。这种混合结构不仅可以带来传统方法的计算复杂度优势，还可以保证神经网络的影响下的匹配精度。对KITTI 2015基准测试的广泛实验表明，我们的轻量级立体匹配系统在提高由不同算法生成的粗糙视差图的准确性方面表现出高鲁棒性，同时在嵌入式GPU上实时运行。

    We present a lightweight system for stereo matching through embedded GPUs. It breaks the trade-off between accuracy and processing speed in stereo matching, enabling our embedded system to further improve the matching accuracy while ensuring real-time processing. The main idea of our method is to construct a tiny neural network based on variational auto-encoder (VAE) to upsample and refinement a small size of coarse disparity map, which is first generated by a traditional matching method. The proposed hybrid structure cannot only bring the advantage of traditional methods in terms of computational complexity, but also ensure the matching accuracy under the impact of neural network. Extensive experiments on the KITTI 2015 benchmark demonstrate that our tiny system exhibits high robustness in improving the accuracy of the coarse disparity maps generated by different algorithms, while also running in real-time on embedded GPUs.
    

