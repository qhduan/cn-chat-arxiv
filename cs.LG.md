# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [BG-HGNN: Toward Scalable and Efficient Heterogeneous Graph Neural Network](https://arxiv.org/abs/2403.08207) | BG-HGNN提出了一种新颖的框架，有效地处理了现有HGNNs在复杂异构图上面临的参数爆炸和关系坍塌等挑战 |
| [^2] | [WiGenAI: The Symphony of Wireless and Generative AI via Diffusion Models.](http://arxiv.org/abs/2310.07312) | WiGenAI通过引入扩散模型，将生成式人工智能应用于无线通信系统中，为研究奠定基础。这篇文章介绍了扩散模型作为生成模型的最新范式，并讨论了它在无线通信系统中的应用。通过两个案例研究展示了扩散模型在开发韧性的AI本地通信系统中的潜力。 |
| [^3] | [Federated Learning on Heterogeneous Data via Adaptive Self-Distillation.](http://arxiv.org/abs/2305.19600) | 本文提出一种基于自适应自蒸馏的新型正则化技术来训练客户端模型，该正则化方案基于客户端本地模型预测和全局模型的相似性以及客户端的标签分布来自适应地调整客户端的训练数据。实验结果表明，该方法在各种基准数据集上优于目前流行的联邦学习方法。 |
| [^4] | [Freeze then Train: Towards Provable Representation Learning under Spurious Correlations and Feature Noise.](http://arxiv.org/abs/2210.11075) | 本文提出了一种称为冻结再变换(FTT)的算法，用于在存在虚假相关和特征噪声下实现可证明的表示学习。该算法首先冻结特征学习器，然后在其上训练分类器，利用学习到的核心特征，经过实验证明其有效性。 |

# 详细

[^1]: BG-HGNN: 朝向可扩展和高效的异构图神经网络

    BG-HGNN: Toward Scalable and Efficient Heterogeneous Graph Neural Network

    [https://arxiv.org/abs/2403.08207](https://arxiv.org/abs/2403.08207)

    BG-HGNN提出了一种新颖的框架，有效地处理了现有HGNNs在复杂异构图上面临的参数爆炸和关系坍塌等挑战

    

    许多计算机视觉和机器学习问题被建模为在异构图上的学习任务，具有来自不同类型节点和边的各种关系。异构图神经网络(HGNNs)是一种为异构图设计的有前途的神经模型类。现有HGNNs建立在传统GNNs基础上，利用不同的参数空间来建模不同的关系。然而，现有HGNNs的实际有效性通常局限于简单的异构图，具有少量关系类型。本文首先突出和证明现有HGNNs使用的标准方法不可避免地导致参数爆炸和关系坍塌，使得HGNNs对具有大量关系类型的复杂异构图 less有效或不实用。为了克服这一问题，我们引入了一种新颖的框架，Blend&Grind-HGNN (BG-HGNN)，通过仔细处理挑战来有效应对这些问题。

    arXiv:2403.08207v1 Announce Type: new  Abstract: Many computer vision and machine learning problems are modelled as learning tasks on heterogeneous graphs, featuring a wide array of relations from diverse types of nodes and edges. Heterogeneous graph neural networks (HGNNs) stand out as a promising neural model class designed for heterogeneous graphs. Built on traditional GNNs, existing HGNNs employ different parameter spaces to model the varied relationships. However, the practical effectiveness of existing HGNNs is often limited to simple heterogeneous graphs with few relation types. This paper first highlights and demonstrates that the standard approach employed by existing HGNNs inevitably leads to parameter explosion and relation collapse, making HGNNs less effective or impractical for complex heterogeneous graphs with numerous relation types. To overcome this issue, we introduce a novel framework, Blend&Grind-HGNN (BG-HGNN), which effectively tackles the challenges by carefully i
    
[^2]: WiGenAI: 通过扩散模型实现无线和生成式人工智能的交织

    WiGenAI: The Symphony of Wireless and Generative AI via Diffusion Models. (arXiv:2310.07312v1 [cs.IT])

    [http://arxiv.org/abs/2310.07312](http://arxiv.org/abs/2310.07312)

    WiGenAI通过引入扩散模型，将生成式人工智能应用于无线通信系统中，为研究奠定基础。这篇文章介绍了扩散模型作为生成模型的最新范式，并讨论了它在无线通信系统中的应用。通过两个案例研究展示了扩散模型在开发韧性的AI本地通信系统中的潜力。

    

    创新的基础模型，如GPT-3和稳定的扩散模型，已经在人工智能领域实现了范式转变，向生成式人工智能系统发展。从数据通信和网络的角度来看，人工智能和机器学习算法预计将广泛应用于未来无线通信系统的新一代中，强调了在新兴通信场景中需要新颖的AI本地解决方案。本文介绍生成式人工智能在无线通信系统中的应用，为该领域的研究奠定基础。介绍了扩散型生成模型作为生成模型的最新范式，并讨论了它们在无线通信系统中的应用。还提供了两个案例研究，展示了如何利用扩散模型开发具有韧性的AI本地通信系统。具体而言，我们提出了一种基于扩散模型的生成模型，以展示其在生成模型的应用中的优势。

    Innovative foundation models, such as GPT-3 and stable diffusion models, have made a paradigm shift in the realm of artificial intelligence (AI) towards generative AI-based systems. In unison, from data communication and networking perspective, AI and machine learning (AI/ML) algorithms are envisioned to be pervasively incorporated into the future generations of wireless communications systems, highlighting the need for novel AI-native solutions for the emergent communication scenarios. In this article, we outline the applications of generative AI in wireless communication systems to lay the foundations for research in this field. Diffusion-based generative models, as the new state-of-the-art paradigm of generative models, are introduced, and their applications in wireless communication systems are discussed. Two case studies are also presented to showcase how diffusion models can be exploited for the development of resilient AI-native communication systems. Specifically, we propose de
    
[^3]: 自适应自蒸馏下的异构数据联邦学习

    Federated Learning on Heterogeneous Data via Adaptive Self-Distillation. (arXiv:2305.19600v1 [cs.LG])

    [http://arxiv.org/abs/2305.19600](http://arxiv.org/abs/2305.19600)

    本文提出一种基于自适应自蒸馏的新型正则化技术来训练客户端模型，该正则化方案基于客户端本地模型预测和全局模型的相似性以及客户端的标签分布来自适应地调整客户端的训练数据。实验结果表明，该方法在各种基准数据集上优于目前流行的联邦学习方法。

    

    联邦学习是一种机器学习范式，它使得客户机可以聚合本地训练模型而无需共享任何本地训练数据从而训练全局模型。然而，实践中发现，每个客户端观察到的本地数据分布之间可能存在显著的不均匀性（例如类别不平衡）。在这种不均匀的数据分布下，联邦学习会出现“客户机漂移”问题，导致每个客户端收敛到其自己的局部最优解，这会降低模型的收敛速度并降低模型性能。为了解决这个问题，我们提出了一种基于自适应自蒸馏的新型正则化技术来训练客户端模型。我们的正则化方案基于客户端本地模型预测和全局模型的相似性以及客户端的标签分布来自适应地调整客户端的训练数据。该正则化技术可以轻松地集成在现有的联邦学习算法之上，而不需要对客户端或服务器代码进行任何更改，因此具有高度的可部署性。我们在各种基准数据集上验证了我们的方法，并展示了在非独立同分布数据下的优越性。

    Federated Learning (FL) is a machine learning paradigm that enables clients to jointly train a global model by aggregating the locally trained models without sharing any local training data. In practice, there can often be substantial heterogeneity (e.g., class imbalance) across the local data distributions observed by each of these clients. Under such non-iid data distributions across clients, FL suffers from the 'client-drift' problem where every client converges to its own local optimum. This results in slower convergence and poor performance of the aggregated model. To address this limitation, we propose a novel regularization technique based on adaptive self-distillation (ASD) for training models on the client side. Our regularization scheme adaptively adjusts to the client's training data based on: (1) the closeness of the local model's predictions with that of the global model and (2) the client's label distribution. The proposed regularization can be easily integrated atop exis
    
[^4]: 冻结再训练：在虚假相关和特征噪声下实现可证明的表示学习

    Freeze then Train: Towards Provable Representation Learning under Spurious Correlations and Feature Noise. (arXiv:2210.11075v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.11075](http://arxiv.org/abs/2210.11075)

    本文提出了一种称为冻结再变换(FTT)的算法，用于在存在虚假相关和特征噪声下实现可证明的表示学习。该算法首先冻结特征学习器，然后在其上训练分类器，利用学习到的核心特征，经过实验证明其有效性。

    

    在训练环境中存在虚假相关，如图像背景，可能使经验风险最小化方法(ERM)在测试环境中表现不佳。为了解决这个问题，Kirichenko等人(2022) 实证发现，即使存在虚假相关，与结果相关的核心特征仍然可以很好地学习。这开启了一种有前途的策略，即首先训练特征学习器而不是分类器，然后在测试环境中进行线性探测(重训练最后一层)。然而，缺乏一个理论上的理解何时以及为什么这种方法有效。在本文中，我们发现只有当与结果相关的核心特征关联的不可实现噪声小于虚假特征的噪声时，才能很好地学习这些特征，这在实践中并不一定成立。我们提供理论和实验证据支持这个发现，并阐述不可实现噪声的重要性。此外，我们提出了一种称为冻结再变换(FTT)的算法，首先冻结特征学习器，然后在其上训练分类器，利用学习到的核心特征。我们证明了FTT在特征学习器上的一个温和条件下保证有界的泛化误差。实验证明了FTT在各种数据集和虚假相关以及特征噪声设置下的有效性。

    The existence of spurious correlations such as image backgrounds in the training environment can make empirical risk minimization (ERM) perform badly in the test environment. To address this problem, Kirichenko et al. (2022) empirically found that the core features that are related to the outcome can still be learned well even with the presence of spurious correlations. This opens a promising strategy to first train a feature learner rather than a classifier, and then perform linear probing (last layer retraining) in the test environment. However, a theoretical understanding of when and why this approach works is lacking. In this paper, we find that core features are only learned well when their associated non-realizable noise is smaller than that of spurious features, which is not necessarily true in practice. We provide both theories and experiments to support this finding and to illustrate the importance of non-realizable noise. Moreover, we propose an algorithm called Freeze then T
    

