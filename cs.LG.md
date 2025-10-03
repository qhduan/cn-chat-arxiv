# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LPAC: Learnable Perception-Action-Communication Loops with Applications to Coverage Control.](http://arxiv.org/abs/2401.04855) | 提出了一种可学习的感知-行动-通信(LPAC)架构，使用卷积神经网络处理环境感知，图神经网络实现机器人之间的信息交流，浅层多层感知机计算机器人的动作。使用集中式显微算法训练模型，实现机器人群体的协作。 |
| [^2] | [Machine learning for accuracy in density functional approximations.](http://arxiv.org/abs/2311.00196) | 本文回顾了近期在改进密度泛函及相关近似方法中应用机器学习的进展，讨论了在不同化学和材料类别之间设计可迁移的机器学习模型时可能面临的挑战和希望。 |
| [^3] | [Counterfactual Fairness for Predictions using Generative Adversarial Networks.](http://arxiv.org/abs/2310.17687) | 这篇论文提出了一种使用生成对抗网络实现对照因果公平性的方法，通过学习敏感属性的后代的对照分布来确保公平预测。 |
| [^4] | [Maximum Likelihood Estimation of Latent Variable Structural Equation Models: A Neural Network Approach.](http://arxiv.org/abs/2309.14073) | 本研究提出了一种新的图形结构，用于在线性和高斯性假设下稳定的潜变量结构方程模型。我们证明了计算该模型的最大似然估计等价于训练一个神经网络，并实现了一个基于GPU的算法来进行计算。 |
| [^5] | [Differentially Private Clustering in Data Streams.](http://arxiv.org/abs/2307.07449) | 本研究提出了首个针对$k$-means和$k$-median聚类的差分隐私流算法，在流模型中实现对数据隐私的保护，并使用尽可能少的空间。 |
| [^6] | [BOF-UCB: A Bayesian-Optimistic Frequentist Algorithm for Non-Stationary Contextual Bandits.](http://arxiv.org/abs/2307.03587) | BOF-UCB是一种用于非平稳环境下的背景线性赌博机的贝叶斯优化频率算法，其结合了贝叶斯和频率学派原则，提高了在动态环境中的性能。它利用贝叶斯更新推断后验分布，并使用频率学派方法计算上界信心界以平衡探索和开发。实验证明，BOF-UCB优于现有方法，是非平稳环境中顺序决策的有前途的解决方案。 |
| [^7] | [A* Search Without Expansions: Learning Heuristic Functions with Deep Q-Networks.](http://arxiv.org/abs/2102.04518) | 本文提出了一种使用深度Q网络学习启发式函数，通过只进行一次前向传递计算相邻节点的转移成本和启发式值之和，并在不显式生成这些子节点的情况下指导搜索的Q*搜索算法，以大幅减少计算时间。在魔方问题上的实验表明，该方法能够高效地解决具有大动作空间的问题。 |

# 详细

[^1]: LPAC: 可学习的感知-行动-通信循环及其在覆盖控制中的应用

    LPAC: Learnable Perception-Action-Communication Loops with Applications to Coverage Control. (arXiv:2401.04855v1 [cs.RO])

    [http://arxiv.org/abs/2401.04855](http://arxiv.org/abs/2401.04855)

    提出了一种可学习的感知-行动-通信(LPAC)架构，使用卷积神经网络处理环境感知，图神经网络实现机器人之间的信息交流，浅层多层感知机计算机器人的动作。使用集中式显微算法训练模型，实现机器人群体的协作。

    

    覆盖控制是指导机器人群体协同监测未知的感兴趣特征或现象的问题。在有限的通信和感知能力的分散设置中，这个问题具有挑战性。本文提出了一种可学习的感知-行动-通信(LPAC)架构来解决覆盖控制问题。在该解决方案中，卷积神经网络(CNN)处理了环境的局部感知；图神经网络(GNN)实现了邻近机器人之间的相关信息通信；最后，浅层多层感知机(MLP)计算机器人的动作。通信模块中的GNN通过计算应该与邻居通信哪些信息以及如何利用接收到的信息采取适当的行动来实现机器人群体的协作。我们使用一个知晓整个环境的集中式显微算法来进行模型的训练。

    Coverage control is the problem of navigating a robot swarm to collaboratively monitor features or a phenomenon of interest not known a priori. The problem is challenging in decentralized settings with robots that have limited communication and sensing capabilities. This paper proposes a learnable Perception-Action-Communication (LPAC) architecture for the coverage control problem. In the proposed solution, a convolution neural network (CNN) processes localized perception of the environment; a graph neural network (GNN) enables communication of relevant information between neighboring robots; finally, a shallow multi-layer perceptron (MLP) computes robot actions. The GNN in the communication module enables collaboration in the robot swarm by computing what information to communicate with neighbors and how to use received information to take appropriate actions. We train models using imitation learning with a centralized clairvoyant algorithm that is aware of the entire environment. Eva
    
[^2]: 用于提高密度泛函近似精确性的机器学习技术

    Machine learning for accuracy in density functional approximations. (arXiv:2311.00196v1 [physics.chem-ph])

    [http://arxiv.org/abs/2311.00196](http://arxiv.org/abs/2311.00196)

    本文回顾了近期在改进密度泛函及相关近似方法中应用机器学习的进展，讨论了在不同化学和材料类别之间设计可迁移的机器学习模型时可能面临的挑战和希望。

    

    机器学习技术已经成为计算化学中不可或缺的工具，用于加速原子模拟和材料设计。此外，机器学习方法有可能提高计算效率高的电子结构理论（如密度泛函理论）的预测能力，纠正密度泛函方法中的基本错误。本文综述了最近在应用机器学习改进密度泛函和相关近似方法的进展。通过示例应用有希望的模型于训练集之外的系统，讨论了在不同化学和材料类别之间设计可迁移的机器学习模型时可能面临的挑战和希望。

    Machine learning techniques have found their way into computational chemistry as indispensable tools to accelerate atomistic simulations and materials design. In addition, machine learning approaches hold the potential to boost the predictive power of computationally efficient electronic structure methods, such as density functional theory, to chemical accuracy and to correct for fundamental errors in density functional approaches. Here, recent progress in applying machine learning to improve the accuracy of density functional and related approximations is reviewed. Promises and challenges in devising machine learning models transferable between different chemistries and materials classes are discussed with the help of examples applying promising models to systems far outside their training sets.
    
[^3]: 使用生成对抗网络进行预测的对照因果公平性

    Counterfactual Fairness for Predictions using Generative Adversarial Networks. (arXiv:2310.17687v1 [cs.LG])

    [http://arxiv.org/abs/2310.17687](http://arxiv.org/abs/2310.17687)

    这篇论文提出了一种使用生成对抗网络实现对照因果公平性的方法，通过学习敏感属性的后代的对照分布来确保公平预测。

    

    由于法律、伦理和社会原因，预测中的公平性在实践中非常重要。通常通过对照因果公平性来实现，该公平性确保个体的预测与在不同敏感属性下的对照世界中的预测相同。然而，要实现对照因果公平性是具有挑战性的，因为对照是不可观察的。在本文中，我们开发了一种新颖的深度神经网络，称为对照因果公平性生成对抗网络（GCFN），用于在对照因果公平性下进行预测。具体而言，我们利用一个量身定制的生成对抗网络直接学习敏感属性的后代的对照分布，然后通过一种新颖的对照媒介正则化来实施公平预测。如果对照分布学习得足够好，我们的方法在数学上确保对照因果公平性的概念。因此，我们的GCFN解决了对照因果公平性问题。

    Fairness in predictions is of direct importance in practice due to legal, ethical, and societal reasons. It is often achieved through counterfactual fairness, which ensures that the prediction for an individual is the same as that in a counterfactual world under a different sensitive attribute. However, achieving counterfactual fairness is challenging as counterfactuals are unobservable. In this paper, we develop a novel deep neural network called Generative Counterfactual Fairness Network (GCFN) for making predictions under counterfactual fairness. Specifically, we leverage a tailored generative adversarial network to directly learn the counterfactual distribution of the descendants of the sensitive attribute, which we then use to enforce fair predictions through a novel counterfactual mediator regularization. If the counterfactual distribution is learned sufficiently well, our method is mathematically guaranteed to ensure the notion of counterfactual fairness. Thereby, our GCFN addre
    
[^4]: 潜变量结构方程模型的最大似然估计：一种神经网络方法

    Maximum Likelihood Estimation of Latent Variable Structural Equation Models: A Neural Network Approach. (arXiv:2309.14073v1 [stat.ML])

    [http://arxiv.org/abs/2309.14073](http://arxiv.org/abs/2309.14073)

    本研究提出了一种新的图形结构，用于在线性和高斯性假设下稳定的潜变量结构方程模型。我们证明了计算该模型的最大似然估计等价于训练一个神经网络，并实现了一个基于GPU的算法来进行计算。

    

    我们提出了一种在线性和高斯性假设下稳定的结构方程模型的图形结构。我们展示了计算这个模型的最大似然估计等价于训练一个神经网络。我们实现了一个基于GPU的算法来计算这些模型的最大似然估计。

    We propose a graphical structure for structural equation models that is stable under marginalization under linearity and Gaussianity assumptions. We show that computing the maximum likelihood estimation of this model is equivalent to training a neural network. We implement a GPU-based algorithm that computes the maximum likelihood estimation of these models.
    
[^5]: 数据流中的差分隐私聚类

    Differentially Private Clustering in Data Streams. (arXiv:2307.07449v1 [cs.DS])

    [http://arxiv.org/abs/2307.07449](http://arxiv.org/abs/2307.07449)

    本研究提出了首个针对$k$-means和$k$-median聚类的差分隐私流算法，在流模型中实现对数据隐私的保护，并使用尽可能少的空间。

    

    流模型是处理大规模现代数据分析的一种常见方法。在流模型中，数据点依次流入，算法只能对数据流进行一次遍历，目标是在使用尽可能少的空间的同时，在流中进行一些分析。聚类问题是基本的无监督机器学习原语，过去已经对流聚类算法进行了广泛的研究。然而，在许多实际应用中，数据隐私已成为一个核心关注点，非私有聚类算法在许多场景下不适用。在这项工作中，我们提供了第一个针对$k$-means和$k$-median聚类的差分私有流算法，该算法在长度最多为$T$的流上使用$poly(k,d,\log(T))$的空间来实现一个“常数”。

    The streaming model is an abstraction of computing over massive data streams, which is a popular way of dealing with large-scale modern data analysis. In this model, there is a stream of data points, one after the other. A streaming algorithm is only allowed one pass over the data stream, and the goal is to perform some analysis during the stream while using as small space as possible.  Clustering problems (such as $k$-means and $k$-median) are fundamental unsupervised machine learning primitives, and streaming clustering algorithms have been extensively studied in the past. However, since data privacy becomes a central concern in many real-world applications, non-private clustering algorithms are not applicable in many scenarios.  In this work, we provide the first differentially private streaming algorithms for $k$-means and $k$-median clustering of $d$-dimensional Euclidean data points over a stream with length at most $T$ using $poly(k,d,\log(T))$ space to achieve a {\it constant} 
    
[^6]: BOF-UCB: 一种用于非平稳环境下的上下界信心算法的贝叶斯优化频率算法

    BOF-UCB: A Bayesian-Optimistic Frequentist Algorithm for Non-Stationary Contextual Bandits. (arXiv:2307.03587v1 [cs.LG])

    [http://arxiv.org/abs/2307.03587](http://arxiv.org/abs/2307.03587)

    BOF-UCB是一种用于非平稳环境下的背景线性赌博机的贝叶斯优化频率算法，其结合了贝叶斯和频率学派原则，提高了在动态环境中的性能。它利用贝叶斯更新推断后验分布，并使用频率学派方法计算上界信心界以平衡探索和开发。实验证明，BOF-UCB优于现有方法，是非平稳环境中顺序决策的有前途的解决方案。

    

    我们提出了一种新颖的贝叶斯优化频率上下界信心算法（BOF-UCB），用于非平稳环境下的随机背景线性赌博机。贝叶斯和频率学派原则的独特结合增强了算法在动态环境中的适应性和性能。BOF-UCB算法利用顺序贝叶斯更新推断未知回归参数的后验分布，并随后采用频率学派方法通过最大化后验分布上的期望收益来计算上界信心界（UCB）。我们提供了BOF-UCB性能的理论保证，并在合成数据集和强化学习环境中的经典控制任务中展示了其有效性。我们的结果表明，BOF-UCB优于现有的方法，在非平稳环境中进行顺序决策是一个有前途的解决方案。

    We propose a novel Bayesian-Optimistic Frequentist Upper Confidence Bound (BOF-UCB) algorithm for stochastic contextual linear bandits in non-stationary environments. This unique combination of Bayesian and frequentist principles enhances adaptability and performance in dynamic settings. The BOF-UCB algorithm utilizes sequential Bayesian updates to infer the posterior distribution of the unknown regression parameter, and subsequently employs a frequentist approach to compute the Upper Confidence Bound (UCB) by maximizing the expected reward over the posterior distribution. We provide theoretical guarantees of BOF-UCB's performance and demonstrate its effectiveness in balancing exploration and exploitation on synthetic datasets and classical control tasks in a reinforcement learning setting. Our results show that BOF-UCB outperforms existing methods, making it a promising solution for sequential decision-making in non-stationary environments.
    
[^7]: 不扩展的A*搜索：用深度Q网络学习启发式函数

    A* Search Without Expansions: Learning Heuristic Functions with Deep Q-Networks. (arXiv:2102.04518v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2102.04518](http://arxiv.org/abs/2102.04518)

    本文提出了一种使用深度Q网络学习启发式函数，通过只进行一次前向传递计算相邻节点的转移成本和启发式值之和，并在不显式生成这些子节点的情况下指导搜索的Q*搜索算法，以大幅减少计算时间。在魔方问题上的实验表明，该方法能够高效地解决具有大动作空间的问题。

    

    高效地使用 A* 搜索解决具有大动作空间的问题对于人工智能社区几十年来一直非常重要。这是因为 A* 搜索的计算和存储需求随着动作空间的大小呈线性增长。当 A* 搜索使用计算代价高昂的函数逼近器（如深度神经网络）学习启发式函数时，这种负担变得更加明显。为了解决这个问题，我们引入了 Q* 搜索，一种使用深度 Q 网络引导搜索的搜索算法，以利用一个事实，即在不显式生成这些子节点的情况下，一个节点的子节点的转移成本和启发式值之和可以通过单次前向传递计算。这显着降低了计算时间，并且每次迭代只需要生成一个节点。我们使用 Q* 搜索来解决魔方问题，并将其们表示为一个包含 1872 个元动作的大动作空间。

    Efficiently solving problems with large action spaces using A* search has been of importance to the artificial intelligence community for decades. This is because the computation and memory requirements of A* search grow linearly with the size of the action space. This burden becomes even more apparent when A* search uses a heuristic function learned by computationally expensive function approximators, such as deep neural networks. To address this problem, we introduce Q* search, a search algorithm that uses deep Q-networks to guide search in order to take advantage of the fact that the sum of the transition costs and heuristic values of the children of a node can be computed with a single forward pass through a deep Q-network without explicitly generating those children. This significantly reduces computation time and requires only one node to be generated per iteration. We use Q* search to solve the Rubik's cube when formulated with a large action space that includes 1872 meta-action
    

