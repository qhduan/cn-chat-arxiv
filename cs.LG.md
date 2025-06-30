# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Link Prediction under Heterophily: A Physics-Inspired Graph Neural Network Approach](https://arxiv.org/abs/2402.14802) | 图神经网络在异质图上的链路预测面临学习能力和表达能力方面的挑战，本论文提出了受物理启发的方法以增强节点分类性能。 |
| [^2] | [Distributional Reduction: Unifying Dimensionality Reduction and Clustering with Gromov-Wasserstein Projection](https://arxiv.org/abs/2402.02239) | 本文提出了一种新的分布约简方法，利用格罗莫夫-瓦瑟斯坦投影统一了降维和聚类，通过优化问题同时解决降维和聚类，实验证明了该方法在多个领域表现出卓越性能。 |
| [^3] | [Robust Multi-Modal Density Estimation.](http://arxiv.org/abs/2401.10566) | 本文提出了一种名为ROME的鲁棒多模态密度估计方法，该方法利用聚类将多模态样本集分割成多个单模态样本集，并通过简单的KDE估计来估计整体分布。这种方法解决了多模态、非正态和高相关分布估计的挑战。 |
| [^4] | [Semi-Supervised Learning Approach for Efficient Resource Allocation with Network Slicing in O-RAN.](http://arxiv.org/abs/2401.08861) | 本文提出了一种半监督学习方法，解决了O-RAN中网络切片和资源分配的问题。通过设计两个xAPPs，分别处理功率控制和物理资源块分配，我们的方法能够在用户设备之间实现最大化的加权吞吐量，并优先考虑增强型移动宽带和超可靠低延迟通信这两种服务类型。 |

# 详细

[^1]: 在异质性下的链路预测: 受物理启发的图神经网络方法

    Link Prediction under Heterophily: A Physics-Inspired Graph Neural Network Approach

    [https://arxiv.org/abs/2402.14802](https://arxiv.org/abs/2402.14802)

    图神经网络在异质图上的链路预测面临学习能力和表达能力方面的挑战，本论文提出了受物理启发的方法以增强节点分类性能。

    

    最近几年，由于其在对图表示的真实世界现象建模方面的灵活性，图神经网络（GNNs）已成为各种深度学习领域的事实标准。然而，GNNs的消息传递机制在学习能力和表达能力方面面临挑战，这限制了在异质图上实现高性能的能力，其中相邻节点经常具有不同的标签。大多数现有解决方案主要局限于针对节点分类任务的特定基准。这种狭窄的焦点限制了链路预测在多个应用中的潜在影响，包括推荐系统。例如，在社交网络中，两个用户可能由于某种潜在原因而连接，这使得提前预测这种连接具有挑战性。受物理启发的GNNs（如GRAFF）对提高节点分类性能提供了显著的贡献。

    arXiv:2402.14802v1 Announce Type: new  Abstract: In the past years, Graph Neural Networks (GNNs) have become the `de facto' standard in various deep learning domains, thanks to their flexibility in modeling real-world phenomena represented as graphs. However, the message-passing mechanism of GNNs faces challenges in learnability and expressivity, hindering high performance on heterophilic graphs, where adjacent nodes frequently have different labels. Most existing solutions addressing these challenges are primarily confined to specific benchmarks focused on node classification tasks. This narrow focus restricts the potential impact that link prediction under heterophily could offer in several applications, including recommender systems. For example, in social networks, two users may be connected for some latent reason, making it challenging to predict such connections in advance. Physics-Inspired GNNs such as GRAFF provided a significant contribution to enhance node classification perf
    
[^2]: 分布约简：用格罗莫夫-瓦瑟斯坦投影统一降维和聚类

    Distributional Reduction: Unifying Dimensionality Reduction and Clustering with Gromov-Wasserstein Projection

    [https://arxiv.org/abs/2402.02239](https://arxiv.org/abs/2402.02239)

    本文提出了一种新的分布约简方法，利用格罗莫夫-瓦瑟斯坦投影统一了降维和聚类，通过优化问题同时解决降维和聚类，实验证明了该方法在多个领域表现出卓越性能。

    

    无监督学习旨在捕捉潜在的大规模和高维数据集的结构。传统上，这涉及使用降维方法将数据投影到可解释的空间上，或将数据点组织成有意义的聚类。在实践中，这些方法通常是按顺序使用的，而不能保证聚类与降维相一致。在这项工作中，我们提出了一个新的观点：使用分布。通过利用最优输运的工具，特别是格罗莫夫-瓦瑟斯坦距离，我们将聚类和降维统一为一个称为分布约简的单一框架。这使我们能够通过单个优化问题同时解决聚类和降维。通过全面的实验证明了我们方法的多功能性和解释性，并表明它在各种图像和基因组数据集上优于现有方法。

    Unsupervised learning aims to capture the underlying structure of potentially large and high-dimensional datasets. Traditionally, this involves using dimensionality reduction methods to project data onto interpretable spaces or organizing points into meaningful clusters. In practice, these methods are used sequentially, without guaranteeing that the clustering aligns well with the conducted dimensionality reduction. In this work, we offer a fresh perspective: that of distributions. Leveraging tools from optimal transport, particularly the Gromov-Wasserstein distance, we unify clustering and dimensionality reduction into a single framework called distributional reduction. This allows us to jointly address clustering and dimensionality reduction with a single optimization problem. Through comprehensive experiments, we highlight the versatility and interpretability of our method and show that it outperforms existing approaches across a variety of image and genomics datasets.
    
[^3]: 鲁棒的多模态密度估计

    Robust Multi-Modal Density Estimation. (arXiv:2401.10566v1 [cs.LG])

    [http://arxiv.org/abs/2401.10566](http://arxiv.org/abs/2401.10566)

    本文提出了一种名为ROME的鲁棒多模态密度估计方法，该方法利用聚类将多模态样本集分割成多个单模态样本集，并通过简单的KDE估计来估计整体分布。这种方法解决了多模态、非正态和高相关分布估计的挑战。

    

    多模态概率预测模型的发展引发了对综合评估指标的需求。虽然有几个指标可以表征机器学习模型的准确性（例如，负对数似然、Jensen-Shannon散度），但这些指标通常作用于概率密度上。因此，将它们应用于纯粹基于样本的预测模型需要估计底层密度函数。然而，常见的方法如核密度估计（KDE）已被证明在鲁棒性方面存在不足，而更复杂的方法在多模态估计问题中尚未得到评估。在本文中，我们提出了一种非参数的密度估计方法ROME（RObust Multi-modal density Estimator），它解决了估计多模态、非正态和高相关分布的挑战。ROME利用聚类将多模态样本集分割成多个单模态样本集，然后结合简单的KDE估计来得到总体的估计结果。

    Development of multi-modal, probabilistic prediction models has lead to a need for comprehensive evaluation metrics. While several metrics can characterize the accuracy of machine-learned models (e.g., negative log-likelihood, Jensen-Shannon divergence), these metrics typically operate on probability densities. Applying them to purely sample-based prediction models thus requires that the underlying density function is estimated. However, common methods such as kernel density estimation (KDE) have been demonstrated to lack robustness, while more complex methods have not been evaluated in multi-modal estimation problems. In this paper, we present ROME (RObust Multi-modal density Estimator), a non-parametric approach for density estimation which addresses the challenge of estimating multi-modal, non-normal, and highly correlated distributions. ROME utilizes clustering to segment a multi-modal set of samples into multiple uni-modal ones and then combines simple KDE estimates obtained for i
    
[^4]: O-RAN中利用半监督学习方法进行网络切片的资源分配的研究

    Semi-Supervised Learning Approach for Efficient Resource Allocation with Network Slicing in O-RAN. (arXiv:2401.08861v1 [cs.NI])

    [http://arxiv.org/abs/2401.08861](http://arxiv.org/abs/2401.08861)

    本文提出了一种半监督学习方法，解决了O-RAN中网络切片和资源分配的问题。通过设计两个xAPPs，分别处理功率控制和物理资源块分配，我们的方法能够在用户设备之间实现最大化的加权吞吐量，并优先考虑增强型移动宽带和超可靠低延迟通信这两种服务类型。

    

    开放式无线接入网络（O-RAN）技术作为一种有前景的解决方案，为网络运营商提供了一个开放和有利的环境。在O-RAN内确保有效地协调x应用程序（xAPPs）对于网络切片和资源分配至关重要。本文介绍了一种创新的资源分配方法，旨在协调O-RAN中多个独立xAPPs的协调。我们的方法侧重于在用户设备（UE）之间最大化加权吞吐量，并分配物理资源块（PRBs）。我们优先考虑增强型移动宽带和超可靠低延迟通信这两种服务类型。为此，我们设计了两个xAPPs：每个UE的功率控制xAPP和PRB分配xAPP。所提出的方法包括两个部分的训练阶段，其中第一部分使用带有变分自动编码器的监督学习进行训练。

    The Open Radio Access Network (O-RAN) technology has emerged as a promising solution for network operators, providing them with an open and favorable environment. Ensuring effective coordination of x-applications (xAPPs) is crucial to enhance flexibility and optimize network performance within the O-RAN. In this paper, we introduce an innovative approach to the resource allocation problem, aiming to coordinate multiple independent xAPPs for network slicing and resource allocation in O-RAN. Our proposed method focuses on maximizing the weighted throughput among user equipments (UE), as well as allocating physical resource blocks (PRBs). We prioritize two service types, namely enhanced Mobile Broadband and Ultra Reliable Low Latency Communication. To achieve this, we have designed two xAPPs: a power control xAPP for each UE and a PRB allocation xAPP. The proposed method consists of a two-part training phase, where the first part uses supervised learning with a Variational Autoencoder tra
    

