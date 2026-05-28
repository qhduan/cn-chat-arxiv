# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Weak Convergence Analysis of Online Neural Actor-Critic Algorithms](https://arxiv.org/abs/2403.16825) | 在线神经演员-评论算法中，我们证明当隐藏单元和训练步数的数量$\rightarrow \infty$时，单层神经网络将收敛于随机ODE，通过建立数据样本的几何遍历性和使用泊松方程证明模型更新波动消失，演员神经网络和评论神经网络收敛到具有随机初始条件的ODE系统的解。 |
| [^2] | [Adversarial Fine-tuning of Compressed Neural Networks for Joint Improvement of Robustness and Efficiency](https://arxiv.org/abs/2403.09441) | 本研究探讨了对压缩神经网络进行对抗微调对提高鲁棒性和效率的影响。 |
| [^3] | [Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier](https://arxiv.org/abs/2212.04382) | 本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。 |
| [^4] | [DSSE: a drone swarm search environment.](http://arxiv.org/abs/2307.06240) | DSSE是一个无人机群集搜索环境，用于研究需要动态概率作为输入的强化学习算法。 |

# 详细

[^1]: 在线神经演员-评论算法的弱收敛分析

    Weak Convergence Analysis of Online Neural Actor-Critic Algorithms

    [https://arxiv.org/abs/2403.16825](https://arxiv.org/abs/2403.16825)

    在线神经演员-评论算法中，我们证明当隐藏单元和训练步数的数量$\rightarrow \infty$时，单层神经网络将收敛于随机ODE，通过建立数据样本的几何遍历性和使用泊松方程证明模型更新波动消失，演员神经网络和评论神经网络收敛到具有随机初始条件的ODE系统的解。

    

    我们证明，使用在线演员评论算法训练的单层神经网络在隐藏单元和训练步数的数量$\rightarrow \infty$时，收敛于一个随机常微分方程（ODE）。在线演员评论算法中，随着模型的更新，数据样本的分布会动态变化，这对于任何收敛分析来说都是一个关键挑战。我们在固定演员策略下建立了数据样本的几何遍历性。然后，使用泊松方程，我们证明由于随机到达的数据样本带来的模型更新波动会随着参数更新次数的增加$\rightarrow \infty$而消失。利用泊松方程和弱收敛技术，我们证明演员神经网络和评论神经网络收敛到具有随机初始条件的ODE系统的解。

    arXiv:2403.16825v1 Announce Type: new  Abstract: We prove that a single-layer neural network trained with the online actor critic algorithm converges in distribution to a random ordinary differential equation (ODE) as the number of hidden units and the number of training steps $\rightarrow \infty$. In the online actor-critic algorithm, the distribution of the data samples dynamically changes as the model is updated, which is a key challenge for any convergence analysis. We establish the geometric ergodicity of the data samples under a fixed actor policy. Then, using a Poisson equation, we prove that the fluctuations of the model updates around the limit distribution due to the randomly-arriving data samples vanish as the number of parameter updates $\rightarrow \infty$. Using the Poisson equation and weak convergence techniques, we prove that the actor neural network and critic neural network converge to the solutions of a system of ODEs with random initial conditions. Analysis of the 
    
[^2]: 对压缩神经网络进行对抗微调，共同提高鲁棒性和效率

    Adversarial Fine-tuning of Compressed Neural Networks for Joint Improvement of Robustness and Efficiency

    [https://arxiv.org/abs/2403.09441](https://arxiv.org/abs/2403.09441)

    本研究探讨了对压缩神经网络进行对抗微调对提高鲁棒性和效率的影响。

    

    随着深度学习（DL）模型越来越多地融入我们的日常生活中，确保它们的安全性，使其对抗对抗性攻击具有鲁棒性变得越来越关键。我们在这项研究中探讨了两种不同的模型压缩方法 -- 结构化权重剪枝和量化对抗鲁棒性的影响。我们特别研究了对压缩模型进行微调的效果，并提出了一种同时提高鲁棒性和效率的方法。

    arXiv:2403.09441v1 Announce Type: new  Abstract: As deep learning (DL) models are increasingly being integrated into our everyday lives, ensuring their safety by making them robust against adversarial attacks has become increasingly critical. DL models have been found to be susceptible to adversarial attacks which can be achieved by introducing small, targeted perturbations to disrupt the input data. Adversarial training has been presented as a mitigation strategy which can result in more robust models. This adversarial robustness comes with additional computational costs required to design adversarial attacks during training. The two objectives -- adversarial robustness and computational efficiency -- then appear to be in conflict of each other. In this work, we explore the effects of two different model compression methods -- structured weight pruning and quantization -- on adversarial robustness. We specifically explore the effects of fine-tuning on compressed models, and present th
    
[^3]: 分类器边界的结构：朴素贝叶斯分类器的案例研究

    Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier

    [https://arxiv.org/abs/2212.04382](https://arxiv.org/abs/2212.04382)

    本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。

    

    无论基于模型、训练数据还是二者组合，分类器将（可能复杂的）输入数据归入相对较少的输出类别之一。本文研究在输入空间为图的情况下，边界的结构——那些被分类为不同类别的邻近点——的特性。我们的科学背景是基于模型的朴素贝叶斯分类器，用于处理由下一代测序仪生成的DNA读数。我们展示了边界既是巨大的，又具有复杂的结构。我们创建了一种新的不确定性度量，称为邻居相似度，它将一个点的结果与其邻居的结果分布进行比较。这个度量不仅追踪了贝叶斯分类器的两个固有不确定性度量，还可以在没有固有不确定性度量的分类器上实现，但需要计算成本。

    Whether based on models, training data or a combination, classifiers place (possibly complex) input data into one of a relatively small number of output categories. In this paper, we study the structure of the boundary--those points for which a neighbor is classified differently--in the context of an input space that is a graph, so that there is a concept of neighboring inputs, The scientific setting is a model-based naive Bayes classifier for DNA reads produced by Next Generation Sequencers. We show that the boundary is both large and complicated in structure. We create a new measure of uncertainty, called Neighbor Similarity, that compares the result for a point to the distribution of results for its neighbors. This measure not only tracks two inherent uncertainty measures for the Bayes classifier, but also can be implemented, at a computational cost, for classifiers without inherent measures of uncertainty.
    
[^4]: DSSE: 无人机群集搜索环境

    DSSE: a drone swarm search environment. (arXiv:2307.06240v1 [cs.LG])

    [http://arxiv.org/abs/2307.06240](http://arxiv.org/abs/2307.06240)

    DSSE是一个无人机群集搜索环境，用于研究需要动态概率作为输入的强化学习算法。

    

    无人机群集搜索项目是一个基于PettingZoo的环境，与多智能体（或单智能体）强化学习算法配合使用。该环境中的智能体（无人机）必须找到目标（遇险人员），但不知道目标的位置，并且不会根据自身与目标的距离得到奖励。但是，智能体会接收到目标出现在地图某个单元格的概率。该项目的目标是帮助研究需要动态概率作为输入的强化学习算法。

    The Drone Swarm Search project is an environment, based on PettingZoo, that is to be used in conjunction with multi-agent (or single-agent) reinforcement learning algorithms. It is an environment in which the agents (drones), have to find the targets (shipwrecked people). The agents do not know the position of the target and do not receive rewards related to their own distance to the target(s). However, the agents receive the probabilities of the target(s) being in a certain cell of the map. The aim of this project is to aid in the study of reinforcement learning algorithms that require dynamic probabilities as inputs.
    

