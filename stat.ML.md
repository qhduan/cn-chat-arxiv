# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Structure-Preserving Kernel Method for Learning Hamiltonian Systems](https://arxiv.org/abs/2403.10070) | 提出了一种保结构的核岭回归方法，可以从噪声观测数据中恢复哈密顿函数，拓展了核回归方法，并具有出色的数值性能和收敛速度。 |
| [^2] | [Metrics on Markov Equivalence Classes for Evaluating Causal Discovery Algorithms](https://arxiv.org/abs/2402.04952) | 本文提出了三个新的距离度量指标（s/c距离、马尔科夫距离和忠实度距离），用于评估因果推断算法的输出图与真实情况的分离/连接程度。 |
| [^3] | [Structured Matrix Learning under Arbitrary Entrywise Dependence and Estimation of Markov Transition Kernel.](http://arxiv.org/abs/2401.02520) | 本论文提出了在任意元素间依赖下进行结构化矩阵估计的通用框架，并证明了提出的最小二乘估计器在各种噪声分布下的紧致性。此外，论文还提出了一个新颖的结果，论述了无关低秩矩阵的结构特点。最后，论文还展示了该框架在结构化马尔可夫转移核估计问题中的应用。 |

# 详细

[^1]: 用于学习哈密顿系统的保结构核方法

    A Structure-Preserving Kernel Method for Learning Hamiltonian Systems

    [https://arxiv.org/abs/2403.10070](https://arxiv.org/abs/2403.10070)

    提出了一种保结构的核岭回归方法，可以从噪声观测数据中恢复哈密顿函数，拓展了核回归方法，并具有出色的数值性能和收敛速度。

    

    提出了一种保结构的核岭回归方法，允许从包含哈密顿向量场的噪声观测数据集中恢复潜在的高维非线性哈密顿函数。该方法提出了一个闭式解，在这一设置中表现出优秀的数值性能，超越了文献中提出的其他技术。从方法论的角度看，该论文扩展了核回归方法，解决需要包含梯度线性函数的损失函数的问题，特别地，在这一背景下证明了微分再现属性和表示定理。分析了保结构核估计器和高斯后验均值估计器之间的关系。进行了完整的误差分析，提供使用固定和自适应正则化参数的收敛速度。所提出方法的优良性能得到了确认。

    arXiv:2403.10070v1 Announce Type: cross  Abstract: A structure-preserving kernel ridge regression method is presented that allows the recovery of potentially high-dimensional and nonlinear Hamiltonian functions out of datasets made of noisy observations of Hamiltonian vector fields. The method proposes a closed-form solution that yields excellent numerical performances that surpass other techniques proposed in the literature in this setup. From the methodological point of view, the paper extends kernel regression methods to problems in which loss functions involving linear functions of gradients are required and, in particular, a differential reproducing property and a Representer Theorem are proved in this context. The relation between the structure-preserving kernel estimator and the Gaussian posterior mean estimator is analyzed. A full error analysis is conducted that provides convergence rates using fixed and adaptive regularization parameters. The good performance of the proposed 
    
[^2]: 评估因果推断算法的马尔科夫等价类指标

    Metrics on Markov Equivalence Classes for Evaluating Causal Discovery Algorithms

    [https://arxiv.org/abs/2402.04952](https://arxiv.org/abs/2402.04952)

    本文提出了三个新的距离度量指标（s/c距离、马尔科夫距离和忠实度距离），用于评估因果推断算法的输出图与真实情况的分离/连接程度。

    

    许多最先进的因果推断方法旨在生成一个输出图，该图编码了生成数据过程的因果图的图形分离和连接陈述。在本文中，我们认为，对合成数据的因果推断方法进行评估应该包括分析该方法的输出与真实情况的分离/连接程度，以衡量这一明确目标的实现情况。我们证明现有的评估指标不能准确捕捉到两个因果图的分离/连接差异，并引入了三个新的距离度量指标，即s/c距离、马尔科夫距离和忠实度距离，以解决这个问题。我们通过玩具示例、实证实验和伪代码来补充我们的理论分析。

    Many state-of-the-art causal discovery methods aim to generate an output graph that encodes the graphical separation and connection statements of the causal graph that underlies the data-generating process. In this work, we argue that an evaluation of a causal discovery method against synthetic data should include an analysis of how well this explicit goal is achieved by measuring how closely the separations/connections of the method's output align with those of the ground truth. We show that established evaluation measures do not accurately capture the difference in separations/connections of two causal graphs, and we introduce three new measures of distance called s/c-distance, Markov distance and Faithfulness distance that address this shortcoming. We complement our theoretical analysis with toy examples, empirical experiments and pseudocode.
    
[^3]: 在任意元素间依赖下的结构化矩阵学习与马尔可夫转移核估计

    Structured Matrix Learning under Arbitrary Entrywise Dependence and Estimation of Markov Transition Kernel. (arXiv:2401.02520v1 [stat.ML])

    [http://arxiv.org/abs/2401.02520](http://arxiv.org/abs/2401.02520)

    本论文提出了在任意元素间依赖下进行结构化矩阵估计的通用框架，并证明了提出的最小二乘估计器在各种噪声分布下的紧致性。此外，论文还提出了一个新颖的结果，论述了无关低秩矩阵的结构特点。最后，论文还展示了该框架在结构化马尔可夫转移核估计问题中的应用。

    

    结构化矩阵估计问题通常在强噪声依赖假设下进行研究。本文考虑噪声低秩加稀疏矩阵恢复的一般框架，其中噪声矩阵可以来自任意具有元素间任意依赖的联合分布。我们提出了一个无关相位约束的最小二乘估计器，并且证明了它在各种噪声分布下都是紧致的，既满足确定性下界又匹配最小化风险。为了实现这一点，我们建立了一个新颖的结果，断言两个任意的低秩无关矩阵之间的差异必须在其元素上扩散能量，换句话说不能太稀疏，这揭示了无关低秩矩阵的结构，可能引起独立兴趣。然后，我们展示了我们框架在几个重要的统计机器学习问题中的应用。在估计结构化马尔可夫转移核的问题中，采用了这种方法。

    The problem of structured matrix estimation has been studied mostly under strong noise dependence assumptions. This paper considers a general framework of noisy low-rank-plus-sparse matrix recovery, where the noise matrix may come from any joint distribution with arbitrary dependence across entries. We propose an incoherent-constrained least-square estimator and prove its tightness both in the sense of deterministic lower bound and matching minimax risks under various noise distributions. To attain this, we establish a novel result asserting that the difference between two arbitrary low-rank incoherent matrices must spread energy out across its entries, in other words cannot be too sparse, which sheds light on the structure of incoherent low-rank matrices and may be of independent interest. We then showcase the applications of our framework to several important statistical machine learning problems. In the problem of estimating a structured Markov transition kernel, the proposed method
    

