# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Neural Network based Double Machine Learning Estimator of Network Causal Effects](https://arxiv.org/abs/2403.11332) | 提出了一种结合图神经网络和双机器学习的新方法，能够准确和高效地估计直接和同行效应，处理网络混杂因素，并一致地估计所需的因果效应 |
| [^2] | [Data Augmentation Scheme for Raman Spectra with Highly Correlated Annotations](https://arxiv.org/abs/2402.00851) | 该论文提出了一种利用光谱可加性的方法，从给定数据集中生成具有统计独立标签的额外数据点，用于训练能够处理非高斯噪声和非线性依赖关系的卷积神经网络。 |
| [^3] | [A Parameter-free Adaptive Resonance Theory-based Topological Clustering Algorithm Capable of Continual Learning.](http://arxiv.org/abs/2305.01507) | 本文提出一种无需预设参数的ART拓扑聚类算法，通过引入参数估计方法实现持续学习，并在实验中证明其比现有聚类算法更优。 |

# 详细

[^1]: 基于图神经网络的网络因果效应双机器学习估计器

    Graph Neural Network based Double Machine Learning Estimator of Network Causal Effects

    [https://arxiv.org/abs/2403.11332](https://arxiv.org/abs/2403.11332)

    提出了一种结合图神经网络和双机器学习的新方法，能够准确和高效地估计直接和同行效应，处理网络混杂因素，并一致地估计所需的因果效应

    

    我们的论文解决了在社交网络数据中推断因果效应的挑战，这些数据具有个体之间复杂的相互依赖关系，导致单位之间不独立、干扰（单位的结果受邻居的处理影响）以及引入来自邻近单位的额外混杂因素等问题。我们提出了一种将图神经网络和双机器学习相结合的创新方法，能够使用单个观测社交网络准确高效地估计直接和同伴效应。我们的方法利用图同构网络与双机器学习相结合，有效调整网络混杂因素并一致地估计所需的因果效应。我们展示了我们的估计器既具有渐近正态性又半参数高效。我们对三个半合成状态下的四种最先进基线方法进行了全面评估

    arXiv:2403.11332v1 Announce Type: new  Abstract: Our paper addresses the challenge of inferring causal effects in social network data, characterized by complex interdependencies among individuals resulting in challenges such as non-independence of units, interference (where a unit's outcome is affected by neighbors' treatments), and introduction of additional confounding factors from neighboring units. We propose a novel methodology combining graph neural networks and double machine learning, enabling accurate and efficient estimation of direct and peer effects using a single observational social network. Our approach utilizes graph isomorphism networks in conjunction with double machine learning to effectively adjust for network confounders and consistently estimate the desired causal effects. We demonstrate that our estimator is both asymptotically normal and semiparametrically efficient. A comprehensive evaluation against four state-of-the-art baseline methods using three semi-synth
    
[^2]: 具有高度相关注释的拉曼光谱的数据增强方案

    Data Augmentation Scheme for Raman Spectra with Highly Correlated Annotations

    [https://arxiv.org/abs/2402.00851](https://arxiv.org/abs/2402.00851)

    该论文提出了一种利用光谱可加性的方法，从给定数据集中生成具有统计独立标签的额外数据点，用于训练能够处理非高斯噪声和非线性依赖关系的卷积神经网络。

    

    在生物技术中，拉曼光谱法作为一种过程分析技术（PAT）快速得到了广泛应用，它可以测量细胞密度、底物和产物浓度。由于拉曼光谱记录了分子的振动模式，因此可以非侵入性地在一个光谱中提供相关信息。通常，偏最小二乘（PLS）是从光谱中推断感兴趣变量信息的模型选择。然而，生物过程以其复杂性而闻名，其中卷积神经网络（CNN）是一个强大的替代方法。它们可以处理非高斯噪声，并考虑光束错位、像素故障或其他物质的存在。然而，它们在模型训练过程中需要大量数据，并且能够捕捉到过程变量之间的非线性依赖关系。在这项工作中，我们利用光谱的可加性来生成从给定数据集中得到的具有统计独立标签的额外数据点，以便训练网络。

    In biotechnology Raman Spectroscopy is rapidly gaining popularity as a process analytical technology (PAT) that measures cell densities, substrate- and product concentrations. As it records vibrational modes of molecules it provides that information non-invasively in a single spectrum. Typically, partial least squares (PLS) is the model of choice to infer information about variables of interest from the spectra. However, biological processes are known for their complexity where convolutional neural networks (CNN) present a powerful alternative. They can handle non-Gaussian noise and account for beam misalignment, pixel malfunctions or the presence of additional substances. However, they require a lot of data during model training, and they pick up non-linear dependencies in the process variables. In this work, we exploit the additive nature of spectra in order to generate additional data points from a given dataset that have statistically independent labels so that a network trained on
    
[^3]: 一种无需预设参数的自适应共振理论拓扑聚类算法，实现持续学习

    A Parameter-free Adaptive Resonance Theory-based Topological Clustering Algorithm Capable of Continual Learning. (arXiv:2305.01507v1 [cs.NE])

    [http://arxiv.org/abs/2305.01507](http://arxiv.org/abs/2305.01507)

    本文提出一种无需预设参数的ART拓扑聚类算法，通过引入参数估计方法实现持续学习，并在实验中证明其比现有聚类算法更优。

    

    一般来说，在自适应共振理论（ART）算法中，节点学习过程中的相似度阈值（即警觉参数）对聚类性能有重大影响。此外，拓扑聚类算法中的边缘删除阈值在自组织过程中生成互相分离的聚类中起重要作用。在本文中，我们提出了一种新的无需预设参数的ART拓扑聚类算法，通过引入参数估计方法实现持续学习。针对合成数据集和真实世界数据集的实验结果表明，所提算法在无预设参数的情况下具有比现有聚类算法更优的聚类性能。

    In general, a similarity threshold (i.e., a vigilance parameter) for a node learning process in Adaptive Resonance Theory (ART)-based algorithms has a significant impact on clustering performance. In addition, an edge deletion threshold in a topological clustering algorithm plays an important role in adaptively generating well-separated clusters during a self-organizing process. In this paper, we propose a new parameter-free ART-based topological clustering algorithm capable of continual learning by introducing parameter estimation methods. Experimental results with synthetic and real-world datasets show that the proposed algorithm has superior clustering performance to the state-of-the-art clustering algorithms without any parameter pre-specifications.
    

