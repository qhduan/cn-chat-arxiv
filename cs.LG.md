# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CLoRA: A Contrastive Approach to Compose Multiple LoRA Models](https://arxiv.org/abs/2403.19776) | CLoRA提出了一种对比方法，用于组合多个LoRA模型，解决了将不同概念LoRA模型无缝混合到一个图像中的挑战。 |
| [^2] | [Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy](https://arxiv.org/abs/2403.16591) | 论文探讨了本地差分隐私、贝叶斯隐私及其之间的相互关系，揭示了关于效用-隐私权衡的新见解，并提出了一个框架来突出攻击和防御策略的相互作用和效果。 |
| [^3] | [FAST: An Optimization Framework for Fast Additive Segmentation in Transparent ML](https://arxiv.org/abs/2402.12630) | FAST框架通过快速分段形状函数的优化和新的特征选择算法，使得透明的附加模型的拟合速度比现有方法快2个数量级。 |
| [^4] | [KIX: A Metacognitive Generalization Framework](https://arxiv.org/abs/2402.05346) | 人工智能代理缺乏通用行为，需要利用结构化知识表示。该论文提出了一种元认知泛化框架KIX，通过与对象的交互学习可迁移的交互概念和泛化能力，促进了知识与强化学习的融合，为实现人工智能系统的自主和通用行为提供了潜力。 |
| [^5] | [Federated Distributionally Robust Optimization with Non-Convex Objectives: Algorithm and Analysis.](http://arxiv.org/abs/2307.14364) | 本文提出了一个名为ASPIRE算法的异步分布式算法，用于解决联邦分布鲁棒优化问题，并引入了约束的D-范数不确定性集合，以灵活控制鲁棒性的程度。 |
| [^6] | [Hyperbolic Graph Neural Networks: A Review of Methods and Applications.](http://arxiv.org/abs/2202.13852) | 本文综述了当前超边界图神经网络的技术细节，提出了一个通用框架，并总结了每个组件的变体。此外，还介绍了各种与HGNN相关的应用和当前面临的挑战。 |

# 详细

[^1]: CLoRA: 一种对比方法来组合多个 LoRA 模型

    CLoRA: A Contrastive Approach to Compose Multiple LoRA Models

    [https://arxiv.org/abs/2403.19776](https://arxiv.org/abs/2403.19776)

    CLoRA提出了一种对比方法，用于组合多个LoRA模型，解决了将不同概念LoRA模型无缝混合到一个图像中的挑战。

    

    低秩调整（LoRA）已经成为图像生成领域中一种强大且受欢迎的技术，提供了一种高效的方式来调整和改进预训练的深度学习模型，而无需全面地重新训练。通过使用预训练的 LoRA 模型，例如代表特定猫和特定狗的模型，我们的目标是生成一个图像，该图像真实地体现了 LoRA 所定义的两种动物。然而，无缝地混合多个概念 LoRA 模型以捕获一个图像中的各种概念的任务被证明是一个重大挑战。常见方法往往表现不佳，主要是因为不同 LoRA 模型内的注意机制重叠，导致一个概念可能被完全忽略（例如漏掉了狗），或者概念被错误地组合在一起（例如生成两只猫的图像而不是一只猫和一只狗）。为了克服这一挑战，

    arXiv:2403.19776v1 Announce Type: cross  Abstract: Low-Rank Adaptations (LoRAs) have emerged as a powerful and popular technique in the field of image generation, offering a highly effective way to adapt and refine pre-trained deep learning models for specific tasks without the need for comprehensive retraining. By employing pre-trained LoRA models, such as those representing a specific cat and a particular dog, the objective is to generate an image that faithfully embodies both animals as defined by the LoRAs. However, the task of seamlessly blending multiple concept LoRAs to capture a variety of concepts in one image proves to be a significant challenge. Common approaches often fall short, primarily because the attention mechanisms within different LoRA models overlap, leading to scenarios where one concept may be completely ignored (e.g., omitting the dog) or where concepts are incorrectly combined (e.g., producing an image of two cats instead of one cat and one dog). To overcome th
    
[^2]: 揭示本地差分隐私、平均贝叶斯隐私和最大贝叶斯隐私之间的相互作用

    Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy

    [https://arxiv.org/abs/2403.16591](https://arxiv.org/abs/2403.16591)

    论文探讨了本地差分隐私、贝叶斯隐私及其之间的相互关系，揭示了关于效用-隐私权衡的新见解，并提出了一个框架来突出攻击和防御策略的相互作用和效果。

    

    机器学习的迅速发展导致了隐私定义的多样化，由于对隐私构成的威胁，包括本地差分隐私（LDP）的概念。虽然被广泛接受并在许多领域中被利用，但这种传统的隐私测量方法仍然存在一定限制，从无法防止推断披露到缺乏对对手背景知识的考虑。在这项全面研究中，我们引入贝叶斯隐私并深入探讨本地差分隐私和其贝叶斯对应物之间错综复杂的关系，揭示了关于效用-隐私权衡的新见解。我们引入了一个框架，概括了攻击和防御策略，突出它们之间的相互作用和效果。我们的理论贡献基于平均贝叶斯隐私（ABP）和最大贝叶斯隐私之间的严格定义和关系。

    arXiv:2403.16591v1 Announce Type: cross  Abstract: The swift evolution of machine learning has led to emergence of various definitions of privacy due to the threats it poses to privacy, including the concept of local differential privacy (LDP). Although widely embraced and utilized across numerous domains, this conventional approach to measure privacy still exhibits certain limitations, spanning from failure to prevent inferential disclosure to lack of consideration for the adversary's background knowledge. In this comprehensive study, we introduce Bayesian privacy and delve into the intricate relationship between local differential privacy and its Bayesian counterparts, unveiling novel insights into utility-privacy trade-offs. We introduce a framework that encapsulates both attack and defense strategies, highlighting their interplay and effectiveness. Our theoretical contributions are anchored in the rigorous definitions and relationships between Average Bayesian Privacy (ABP) and Max
    
[^3]: FAST: 一种用于快速透明机器学习中快速附加分割的优化框架

    FAST: An Optimization Framework for Fast Additive Segmentation in Transparent ML

    [https://arxiv.org/abs/2402.12630](https://arxiv.org/abs/2402.12630)

    FAST框架通过快速分段形状函数的优化和新的特征选择算法，使得透明的附加模型的拟合速度比现有方法快2个数量级。

    

    我们提出了FAST，一种用于快速附加分割的优化框架。FAST为数据集中的每个特征分段常数形状函数，以产生透明的附加模型。该框架利用一种新颖的优化过程适配这些模型，速度比现有最先进的方法，如可解释性增强机器 \citep{nori2019interpretml}，快约2个数量级。我们还在FAST框架中开发了新的特征选择算法，以适配性能良好的简约模型。通过实验证明，FAST提高了附加模型的计算效率和可解释性。

    arXiv:2402.12630v1 Announce Type: cross  Abstract: We present FAST, an optimization framework for fast additive segmentation. FAST segments piecewise constant shape functions for each feature in a dataset to produce transparent additive models. The framework leverages a novel optimization procedure to fit these models $\sim$2 orders of magnitude faster than existing state-of-the-art methods, such as explainable boosting machines \citep{nori2019interpretml}. We also develop new feature selection algorithms in the FAST framework to fit parsimonious models that perform well. Through experiments and case studies, we show that FAST improves the computational efficiency and interpretability of additive models.
    
[^4]: KIX: 一种元认知泛化框架

    KIX: A Metacognitive Generalization Framework

    [https://arxiv.org/abs/2402.05346](https://arxiv.org/abs/2402.05346)

    人工智能代理缺乏通用行为，需要利用结构化知识表示。该论文提出了一种元认知泛化框架KIX，通过与对象的交互学习可迁移的交互概念和泛化能力，促进了知识与强化学习的融合，为实现人工智能系统的自主和通用行为提供了潜力。

    

    人类和其他动物能够灵活解决各种任务，并且能够通过重复使用和应用长期积累的高级知识来适应新颖情境，这表现了一种泛化智能行为。但是人工智能代理更多地是专家，缺乏这种通用行为。人工智能代理需要理解和利用关键的结构化知识表示。我们提出了一种元认知泛化框架，称为Knowledge-Interaction-eXecution (KIX)，并且认为通过与对象的交互来利用类型空间可以促进学习可迁移的交互概念和泛化能力。这是将知识融入到强化学习中的一种自然方式，并有望成为人工智能系统中实现自主和通用行为的推广者。

    Humans and other animals aptly exhibit general intelligence behaviors in solving a variety of tasks with flexibility and ability to adapt to novel situations by reusing and applying high level knowledge acquired over time. But artificial agents are more of a specialist, lacking such generalist behaviors. Artificial agents will require understanding and exploiting critical structured knowledge representations. We present a metacognitive generalization framework, Knowledge-Interaction-eXecution (KIX), and argue that interactions with objects leveraging type space facilitate the learning of transferable interaction concepts and generalization. It is a natural way of integrating knowledge into reinforcement learning and promising to act as an enabler for autonomous and generalist behaviors in artificial intelligence systems.
    
[^5]: 具有非凸目标函数的联邦分布鲁棒优化：算法与分析

    Federated Distributionally Robust Optimization with Non-Convex Objectives: Algorithm and Analysis. (arXiv:2307.14364v1 [math.OC])

    [http://arxiv.org/abs/2307.14364](http://arxiv.org/abs/2307.14364)

    本文提出了一个名为ASPIRE算法的异步分布式算法，用于解决联邦分布鲁棒优化问题，并引入了约束的D-范数不确定性集合，以灵活控制鲁棒性的程度。

    

    分布鲁棒优化 (DRO) 旨在找到一个最优决策，以在概率分布的模糊集合中最小化最坏情况成本，已在各种应用中广泛应用，例如网络行为分析、风险管理等。然而，现有的DRO技术面临三个关键挑战：1）如何处理分布环境中的异步更新；2）如何有效利用先验分布；3）如何根据不同场景适当调整鲁棒性的程度。为此，我们提出了一种名为Asynchronous Single-looP alternatIve gRadient projEction (ASPIRE)算法的异步分布式算法，以处理联邦分布鲁棒优化 (FDRO) 问题。此外，我们还开发了一种新的不确定性集合，即约束的D-范数不确定性集合，以有效利用先验分布并灵活控制鲁棒性的程度。

    Distributionally Robust Optimization (DRO), which aims to find an optimal decision that minimizes the worst case cost over the ambiguity set of probability distribution, has been widely applied in diverse applications, e.g., network behavior analysis, risk management, etc. However, existing DRO techniques face three key challenges: 1) how to deal with the asynchronous updating in a distributed environment; 2) how to leverage the prior distribution effectively; 3) how to properly adjust the degree of robustness according to different scenarios. To this end, we propose an asynchronous distributed algorithm, named Asynchronous Single-looP alternatIve gRadient projEction (ASPIRE) algorithm with the itErative Active SEt method (EASE) to tackle the federated distributionally robust optimization (FDRO) problem. Furthermore, a new uncertainty set, i.e., constrained D-norm uncertainty set, is developed to effectively leverage the prior distribution and flexibly control the degree of robustness.
    
[^6]: 超边界图神经网络：方法和应用综述

    Hyperbolic Graph Neural Networks: A Review of Methods and Applications. (arXiv:2202.13852v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.13852](http://arxiv.org/abs/2202.13852)

    本文综述了当前超边界图神经网络的技术细节，提出了一个通用框架，并总结了每个组件的变体。此外，还介绍了各种与HGNN相关的应用和当前面临的挑战。

    

    图神经网络将传统的神经网络推广到了图结构数据，并因其出色的表征能力而受到广泛关注。尽管取得了显著的成就，但欧几里得模型在与图相关的学习中的性能仍然受到欧几里得几何的表征能力的限制，特别是对于具有高度非欧几里得潜在解剖的数据集。最近，超边界空间在处理具有树状结构和幂律分布的图数据方面越来越受欢迎，这归功于其指数级的增长特性。在本综述中，我们全面回顾了当前超边界图神经网络的技术细节，将它们统一为一个通用框架，并总结了每个组件的变体。更重要的是，我们介绍了各种与HGNN相关的应用。最后，我们还确定了一些挑战，这些挑战可能成为进一步发展图神经网络成就的指导方针。

    Graph neural networks generalize conventional neural networks to graph-structured data and have received widespread attention due to their impressive representation ability. In spite of the remarkable achievements, the performance of Euclidean models in graph-related learning is still bounded and limited by the representation ability of Euclidean geometry, especially for datasets with highly non-Euclidean latent anatomy. Recently, hyperbolic space has gained increasing popularity in processing graph data with tree-like structure and power-law distribution, owing to its exponential growth property. In this survey, we comprehensively revisit the technical details of the current hyperbolic graph neural networks, unifying them into a general framework and summarizing the variants of each component. More importantly, we present various HGNN-related applications. Last, we also identify several challenges, which potentially serve as guidelines for further flourishing the achievements of graph
    

