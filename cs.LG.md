# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Effective Acquisition Functions for Active Correlation Clustering](https://arxiv.org/abs/2402.03587) | 本文提出了三种有效的获取函数用于主动相关聚类，分别基于不一致性概念和信息论量。 |
| [^2] | [FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering.](http://arxiv.org/abs/2310.16152) | 本文提出了一种FLTrojan攻击方法，通过选择性权重篡改，从联邦语言模型中泄露隐私敏感用户数据。通过观察到FL中中间轮次的模型快照可以引起更大的隐私泄露，并发现隐私泄露可以通过篡改模型的选择性权重来加剧。 |
| [^3] | [Federated Learning on Heterogeneous Data via Adaptive Self-Distillation.](http://arxiv.org/abs/2305.19600) | 本文提出一种基于自适应自蒸馏的新型正则化技术来训练客户端模型，该正则化方案基于客户端本地模型预测和全局模型的相似性以及客户端的标签分布来自适应地调整客户端的训练数据。实验结果表明，该方法在各种基准数据集上优于目前流行的联邦学习方法。 |

# 详细

[^1]: 主动相关聚类的有效获取函数

    Effective Acquisition Functions for Active Correlation Clustering

    [https://arxiv.org/abs/2402.03587](https://arxiv.org/abs/2402.03587)

    本文提出了三种有效的获取函数用于主动相关聚类，分别基于不一致性概念和信息论量。

    

    相关聚类是一种强大的无监督学习范例，支持正和负的相似性。本文假设相似性事先未知，而是采用主动学习以一种成本有效的方式迭代地查询相似性。具体而言，我们开发了三种有效的获取函数用于在此设置下使用。其中一种基于不一致性概念（即当相似性违反传递性时）。其余两个基于信息论量，即熵和信息增益。

    Correlation clustering is a powerful unsupervised learning paradigm that supports positive and negative similarities. In this paper, we assume the similarities are not known in advance. Instead, we employ active learning to iteratively query similarities in a cost-efficient way. In particular, we develop three effective acquisition functions to be used in this setting. One is based on the notion of inconsistency (i.e., when similarities violate the transitive property). The remaining two are based on information-theoretic quantities, i.e., entropy and information gain.
    
[^2]: FLTrojan: 通过选择性权重篡改对联邦语言模型进行隐私泄露攻击

    FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering. (arXiv:2310.16152v1 [cs.CR])

    [http://arxiv.org/abs/2310.16152](http://arxiv.org/abs/2310.16152)

    本文提出了一种FLTrojan攻击方法，通过选择性权重篡改，从联邦语言模型中泄露隐私敏感用户数据。通过观察到FL中中间轮次的模型快照可以引起更大的隐私泄露，并发现隐私泄露可以通过篡改模型的选择性权重来加剧。

    

    联邦学习(Federated learning, FL)正成为许多技术应用中的关键组件，包括语言建模领域，其中个体FL参与者在其本地数据集中往往具有敏感的文本数据。然而，确定联邦语言模型中的隐私泄露程度并不简单，现有的攻击只是试图提取数据，而不考虑数据的敏感性或天真性。为了填补这一空白，在本文中，我们介绍了关于从联邦语言模型中泄露隐私敏感用户数据的两个新发现。首先，我们观察到FL中中间轮次的模型快照比最终训练模型能够造成更大的隐私泄露。其次，我们确定隐私泄露可以通过篡改模型的选择性权重来加剧，这些权重特别负责记忆敏感训练数据。我们展示了恶意客户端如何在FL中泄露其他用户的隐私敏感数据。

    Federated learning (FL) is becoming a key component in many technology-based applications including language modeling -- where individual FL participants often have privacy-sensitive text data in their local datasets. However, realizing the extent of privacy leakage in federated language models is not straightforward and the existing attacks only intend to extract data regardless of how sensitive or naive it is. To fill this gap, in this paper, we introduce two novel findings with regard to leaking privacy-sensitive user data from federated language models. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that privacy leakage can be aggravated by tampering with a model's selective weights that are specifically responsible for memorizing the sensitive training data. We show how a malicious client can leak the privacy-sensitive data of some other user in FL even
    
[^3]: 自适应自蒸馏下的异构数据联邦学习

    Federated Learning on Heterogeneous Data via Adaptive Self-Distillation. (arXiv:2305.19600v1 [cs.LG])

    [http://arxiv.org/abs/2305.19600](http://arxiv.org/abs/2305.19600)

    本文提出一种基于自适应自蒸馏的新型正则化技术来训练客户端模型，该正则化方案基于客户端本地模型预测和全局模型的相似性以及客户端的标签分布来自适应地调整客户端的训练数据。实验结果表明，该方法在各种基准数据集上优于目前流行的联邦学习方法。

    

    联邦学习是一种机器学习范式，它使得客户机可以聚合本地训练模型而无需共享任何本地训练数据从而训练全局模型。然而，实践中发现，每个客户端观察到的本地数据分布之间可能存在显著的不均匀性（例如类别不平衡）。在这种不均匀的数据分布下，联邦学习会出现“客户机漂移”问题，导致每个客户端收敛到其自己的局部最优解，这会降低模型的收敛速度并降低模型性能。为了解决这个问题，我们提出了一种基于自适应自蒸馏的新型正则化技术来训练客户端模型。我们的正则化方案基于客户端本地模型预测和全局模型的相似性以及客户端的标签分布来自适应地调整客户端的训练数据。该正则化技术可以轻松地集成在现有的联邦学习算法之上，而不需要对客户端或服务器代码进行任何更改，因此具有高度的可部署性。我们在各种基准数据集上验证了我们的方法，并展示了在非独立同分布数据下的优越性。

    Federated Learning (FL) is a machine learning paradigm that enables clients to jointly train a global model by aggregating the locally trained models without sharing any local training data. In practice, there can often be substantial heterogeneity (e.g., class imbalance) across the local data distributions observed by each of these clients. Under such non-iid data distributions across clients, FL suffers from the 'client-drift' problem where every client converges to its own local optimum. This results in slower convergence and poor performance of the aggregated model. To address this limitation, we propose a novel regularization technique based on adaptive self-distillation (ASD) for training models on the client side. Our regularization scheme adaptively adjusts to the client's training data based on: (1) the closeness of the local model's predictions with that of the global model and (2) the client's label distribution. The proposed regularization can be easily integrated atop exis
    

