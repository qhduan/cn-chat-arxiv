# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Maxwell's Demon at Work: Efficient Pruning by Leveraging Saturation of Neurons](https://arxiv.org/abs/2403.07688) | 重新评估深度神经网络中的死亡神经元现象，提出了Demon Pruning（DemP）方法，通过控制死亡神经元的产生，动态实现网络稀疏化。 |
| [^2] | [Structuring Concept Space with the Musical Circle of Fifths by Utilizing Music Grammar Based Activations](https://arxiv.org/abs/2403.00790) | 提出了一种利用音乐语法调节尖峰神经网络激活的新颖方法，通过应用音乐理论中的和弦进行规则，展示了如何自然地跟随其他激活，最终将概念的映射结构化为音乐五度圆。 |

# 详细

[^1]: Maxwell的恶魔之工作：通过利用神经元饱和实现有效修剪

    Maxwell's Demon at Work: Efficient Pruning by Leveraging Saturation of Neurons

    [https://arxiv.org/abs/2403.07688](https://arxiv.org/abs/2403.07688)

    重新评估深度神经网络中的死亡神经元现象，提出了Demon Pruning（DemP）方法，通过控制死亡神经元的产生，动态实现网络稀疏化。

    

    在训练深度神经网络时，$\textit{死亡神经元}$现象——在训练期间变得不活跃或饱和，输出为零的单元—传统上被视为不可取的，与优化挑战有关，并导致在不断学习的情况下丧失可塑性。本文重新评估了这一现象，专注于稀疏性和修剪。通过系统地探索各种超参数配置对死亡神经元的影响，我们揭示了它们有助于促进简单而有效的结构化修剪算法的潜力。我们提出了$\textit{Demon Pruning}$（DemP），一种控制死亡神经元扩张，动态导致网络稀疏性的方法。通过在活跃单元上注入噪声和采用单周期调度正则化策略的组合，DemP因其简单性和广泛适用性而脱颖而出。在CIFAR10上的实验中...

    arXiv:2403.07688v1 Announce Type: cross  Abstract: When training deep neural networks, the phenomenon of $\textit{dying neurons}$ $\unicode{x2013}$units that become inactive or saturated, output zero during training$\unicode{x2013}$ has traditionally been viewed as undesirable, linked with optimization challenges, and contributing to plasticity loss in continual learning scenarios. In this paper, we reassess this phenomenon, focusing on sparsity and pruning. By systematically exploring the impact of various hyperparameter configurations on dying neurons, we unveil their potential to facilitate simple yet effective structured pruning algorithms. We introduce $\textit{Demon Pruning}$ (DemP), a method that controls the proliferation of dead neurons, dynamically leading to network sparsity. Achieved through a combination of noise injection on active units and a one-cycled schedule regularization strategy, DemP stands out for its simplicity and broad applicability. Experiments on CIFAR10 an
    
[^2]: 利用音乐五度圆构建概念空间：基于音乐语法激活的方法

    Structuring Concept Space with the Musical Circle of Fifths by Utilizing Music Grammar Based Activations

    [https://arxiv.org/abs/2403.00790](https://arxiv.org/abs/2403.00790)

    提出了一种利用音乐语法调节尖峰神经网络激活的新颖方法，通过应用音乐理论中的和弦进行规则，展示了如何自然地跟随其他激活，最终将概念的映射结构化为音乐五度圆。

    

    在本文中，我们探讨了离散神经网络（如尖峰网络）的结构与钢琴曲的构成之间的有趣相似之处。虽然两者都涉及按顺序或并行激活的节点或音符，但后者受益于丰富的音乐理论，以指导有意义的组合。我们提出了一种新颖的方法，利用音乐语法来调节尖峰神经网络中的激活，允许将符号表示为吸引子。通过应用音乐理论中的和弦进行规则，我们展示了某些激活如何自然地跟随其他激活，类似于吸引的概念。此外，我们引入了调制音调的概念，以在网络内导航不同的吸引盆地。最终，我们展示了我们模型中概念的映射是由音乐五度圆构成的，突出了利用音乐理论的潜力。

    arXiv:2403.00790v1 Announce Type: cross  Abstract: In this paper, we explore the intriguing similarities between the structure of a discrete neural network, such as a spiking network, and the composition of a piano piece. While both involve nodes or notes that are activated sequentially or in parallel, the latter benefits from the rich body of music theory to guide meaningful combinations. We propose a novel approach that leverages musical grammar to regulate activations in a spiking neural network, allowing for the representation of symbols as attractors. By applying rules for chord progressions from music theory, we demonstrate how certain activations naturally follow others, akin to the concept of attraction. Furthermore, we introduce the concept of modulating keys to navigate different basins of attraction within the network. Ultimately, we show that the map of concepts in our model is structured by the musical circle of fifths, highlighting the potential for leveraging music theor
    

