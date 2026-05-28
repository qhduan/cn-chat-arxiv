# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Weak Convergence Analysis of Online Neural Actor-Critic Algorithms](https://arxiv.org/abs/2403.16825) | 在线神经演员-评论算法中，我们证明当隐藏单元和训练步数的数量$\rightarrow \infty$时，单层神经网络将收敛于随机ODE，通过建立数据样本的几何遍历性和使用泊松方程证明模型更新波动消失，演员神经网络和评论神经网络收敛到具有随机初始条件的ODE系统的解。 |
| [^2] | [Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier](https://arxiv.org/abs/2212.04382) | 本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。 |

# 详细

[^1]: 在线神经演员-评论算法的弱收敛分析

    Weak Convergence Analysis of Online Neural Actor-Critic Algorithms

    [https://arxiv.org/abs/2403.16825](https://arxiv.org/abs/2403.16825)

    在线神经演员-评论算法中，我们证明当隐藏单元和训练步数的数量$\rightarrow \infty$时，单层神经网络将收敛于随机ODE，通过建立数据样本的几何遍历性和使用泊松方程证明模型更新波动消失，演员神经网络和评论神经网络收敛到具有随机初始条件的ODE系统的解。

    

    我们证明，使用在线演员评论算法训练的单层神经网络在隐藏单元和训练步数的数量$\rightarrow \infty$时，收敛于一个随机常微分方程（ODE）。在线演员评论算法中，随着模型的更新，数据样本的分布会动态变化，这对于任何收敛分析来说都是一个关键挑战。我们在固定演员策略下建立了数据样本的几何遍历性。然后，使用泊松方程，我们证明由于随机到达的数据样本带来的模型更新波动会随着参数更新次数的增加$\rightarrow \infty$而消失。利用泊松方程和弱收敛技术，我们证明演员神经网络和评论神经网络收敛到具有随机初始条件的ODE系统的解。

    arXiv:2403.16825v1 Announce Type: new  Abstract: We prove that a single-layer neural network trained with the online actor critic algorithm converges in distribution to a random ordinary differential equation (ODE) as the number of hidden units and the number of training steps $\rightarrow \infty$. In the online actor-critic algorithm, the distribution of the data samples dynamically changes as the model is updated, which is a key challenge for any convergence analysis. We establish the geometric ergodicity of the data samples under a fixed actor policy. Then, using a Poisson equation, we prove that the fluctuations of the model updates around the limit distribution due to the randomly-arriving data samples vanish as the number of parameter updates $\rightarrow \infty$. Using the Poisson equation and weak convergence techniques, we prove that the actor neural network and critic neural network converge to the solutions of a system of ODEs with random initial conditions. Analysis of the 
    
[^2]: 分类器边界的结构：朴素贝叶斯分类器的案例研究

    Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier

    [https://arxiv.org/abs/2212.04382](https://arxiv.org/abs/2212.04382)

    本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。

    

    无论基于模型、训练数据还是二者组合，分类器将（可能复杂的）输入数据归入相对较少的输出类别之一。本文研究在输入空间为图的情况下，边界的结构——那些被分类为不同类别的邻近点——的特性。我们的科学背景是基于模型的朴素贝叶斯分类器，用于处理由下一代测序仪生成的DNA读数。我们展示了边界既是巨大的，又具有复杂的结构。我们创建了一种新的不确定性度量，称为邻居相似度，它将一个点的结果与其邻居的结果分布进行比较。这个度量不仅追踪了贝叶斯分类器的两个固有不确定性度量，还可以在没有固有不确定性度量的分类器上实现，但需要计算成本。

    Whether based on models, training data or a combination, classifiers place (possibly complex) input data into one of a relatively small number of output categories. In this paper, we study the structure of the boundary--those points for which a neighbor is classified differently--in the context of an input space that is a graph, so that there is a concept of neighboring inputs, The scientific setting is a model-based naive Bayes classifier for DNA reads produced by Next Generation Sequencers. We show that the boundary is both large and complicated in structure. We create a new measure of uncertainty, called Neighbor Similarity, that compares the result for a point to the distribution of results for its neighbors. This measure not only tracks two inherent uncertainty measures for the Bayes classifier, but also can be implemented, at a computational cost, for classifiers without inherent measures of uncertainty.
    

