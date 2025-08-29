# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rethinking Invariance Regularization in Adversarial Training to Improve Robustness-Accuracy Trade-off](https://arxiv.org/abs/2402.14648) | 重新审视了基于表示的不变性正则化方法，提出了Asymmetrically Representation-regularized Adversarial Training (AR-AT)来解决“梯度冲突”和混合分布问题，改善鲁棒性-准确性权衡。 |
| [^2] | [Network Formation and Dynamics Among Multi-LLMs](https://arxiv.org/abs/2402.10659) | 分析了多个LLM在社交网络中的行为，发现它们在给定网络结构并被询问形成网络偏好时表现出与人类社交动态一致的原则。 |
| [^3] | [NetGPT: Generative Pretrained Transformer for Network Traffic.](http://arxiv.org/abs/2304.09513) | 本文提出了首个网络流量生成预训练变压器模型NetGPT，该模型可以优化网络任务的训练效率和有效性。 |

# 详细

[^1]: 在对抗训练中重新思考不变性正则化以改善鲁棒性-准确性权衡

    Rethinking Invariance Regularization in Adversarial Training to Improve Robustness-Accuracy Trade-off

    [https://arxiv.org/abs/2402.14648](https://arxiv.org/abs/2402.14648)

    重新审视了基于表示的不变性正则化方法，提出了Asymmetrically Representation-regularized Adversarial Training (AR-AT)来解决“梯度冲突”和混合分布问题，改善鲁棒性-准确性权衡。

    

    尽管对抗训练一直是抵抗对抗性样本（AEs）的最先进方法，但它们存在鲁棒性-准确性权衡问题。在这项研究中，我们重新审视基于表示的不变性正则化，学习具有辨别性却对抗性不变的表示，旨在缓解这种权衡。我们在经验上确定了妨碍不变性正则化的两个关键问题：（1）不变性损失和分类目标之间的“梯度冲突”，表明存在“崩溃解”，以及（2）由于干净和对抗性输入的分布发散而出现的混合分布问题。为了解决这些问题，我们提出了一种不对称表示正则化的对抗训练（AR-AT），该方法结合了一个停止梯度操作和一个预测器来避免“崩溃解”，灵感来自最近的非对比自监督学习。

    arXiv:2402.14648v1 Announce Type: cross  Abstract: Although adversarial training has been the state-of-the-art approach to defend against adversarial examples (AEs), they suffer from a robustness-accuracy trade-off. In this work, we revisit representation-based invariance regularization to learn discriminative yet adversarially invariant representations, aiming to mitigate this trade-off. We empirically identify two key issues hindering invariance regularization: (1) a "gradient conflict" between invariance loss and classification objectives, indicating the existence of "collapsing solutions," and (2) the mixture distribution problem arising from diverged distributions of clean and adversarial inputs. To address these issues, we propose Asymmetrically Representation-regularized Adversarial Training (AR-AT), which incorporates a stop-gradient operation and a pre-dictor in the invariance loss to avoid "collapsing solutions," inspired by a recent non-contrastive self-supervised learning a
    
[^2]: 多个LLM之间的网络形成与动态

    Network Formation and Dynamics Among Multi-LLMs

    [https://arxiv.org/abs/2402.10659](https://arxiv.org/abs/2402.10659)

    分析了多个LLM在社交网络中的行为，发现它们在给定网络结构并被询问形成网络偏好时表现出与人类社交动态一致的原则。

    

    社交网络影响行为、偏好和关系，在人类社会中对信息和规范的传播起着至关重要的作用。随着大型语言模型（LLMs）越来越多地融入社交和专业环境中，理解它们在社交网络和互动背景下的行为变得至关重要。我们的研究分析了标准网络结构和现实世界网络的行为，以确定多个LLMs的动态是否与人类社交动态一致。我们探讨了各种社交网络原则，包括微观层面的概念，如偏爱附着、三角闭合和同似性，以及宏观层面的概念，如社区结构和小世界现象。我们的研究发现表明，当向LLMs提供网络结构并询问它们对网络形成的偏好时，它们表现出所有这些原则。

    arXiv:2402.10659v1 Announce Type: cross  Abstract: Social networks influence behaviors, preferences, and relationships and play a crucial role in the dissemination of information and norms within human societies. As large language models (LLMs) increasingly integrate into social and professional environments, understanding their behavior within the context of social networks and interactions becomes essential. Our study analyzes the behaviors of standard network structures and real-world networks to determine whether the dynamics of multiple LLMs align with human social dynamics. We explore various social network principles, including micro-level concepts such as preferential attachment, triadic closure, and homophily, as well as macro-level concepts like community structure and the small-world phenomenon. Our findings suggest that LLMs demonstrate all these principles when they are provided with network structures and asked about their preferences regarding network formation. Furtherm
    
[^3]: NetGPT：网络流量生成预训练变压器模型

    NetGPT: Generative Pretrained Transformer for Network Traffic. (arXiv:2304.09513v1 [cs.NI])

    [http://arxiv.org/abs/2304.09513](http://arxiv.org/abs/2304.09513)

    本文提出了首个网络流量生成预训练变压器模型NetGPT，该模型可以优化网络任务的训练效率和有效性。

    

    预训练模型可以利用大规模的原始数据学习网络流量的基本特征，并为输入流量生成可区分的结果，而不考虑特定的下游任务。有效的预训练模型可以显著优化下游任务的训练效率和有效性，例如流量分类、攻击检测、资源调度、协议分析和流量生成。本文提出了NetGPT，旨在为网络流量构建预训练模型并解决多样的挑战。

    Pretrained models for network traffic can utilize large-scale raw data to learn the essential characteristics of network traffic, and generate distinguishable results for input traffic without considering specific downstream tasks. Effective pretrained models can significantly optimize the training efficiency and effectiveness of downstream tasks, such as traffic classification, attack detection, resource scheduling, protocol analysis, and traffic generation. Despite the great success of pretraining in natural language processing, there is no work in the network field. Considering the diverse demands and characteristics of network traffic and network tasks, it is non-trivial to build a pretrained model for network traffic and we face various challenges, especially the heterogeneous headers and payloads in the multi-pattern network traffic and the different dependencies for contexts of diverse downstream network tasks.  To tackle these challenges, in this paper, we make the first attemp
    

