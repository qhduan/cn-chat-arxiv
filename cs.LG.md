# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rethinking Invariance Regularization in Adversarial Training to Improve Robustness-Accuracy Trade-off](https://arxiv.org/abs/2402.14648) | 重新审视了基于表示的不变性正则化方法，提出了Asymmetrically Representation-regularized Adversarial Training (AR-AT)来解决“梯度冲突”和混合分布问题，改善鲁棒性-准确性权衡。 |
| [^2] | [FLASH: Federated Learning Across Simultaneous Heterogeneities](https://arxiv.org/abs/2402.08769) | FLASH是一个跨同时异质性的联邦学习方法，通过综合考虑数据质量、数据分布和延迟等因素，优于其他现有的方法。 |
| [^3] | [NetGPT: Generative Pretrained Transformer for Network Traffic.](http://arxiv.org/abs/2304.09513) | 本文提出了首个网络流量生成预训练变压器模型NetGPT，该模型可以优化网络任务的训练效率和有效性。 |

# 详细

[^1]: 在对抗训练中重新思考不变性正则化以改善鲁棒性-准确性权衡

    Rethinking Invariance Regularization in Adversarial Training to Improve Robustness-Accuracy Trade-off

    [https://arxiv.org/abs/2402.14648](https://arxiv.org/abs/2402.14648)

    重新审视了基于表示的不变性正则化方法，提出了Asymmetrically Representation-regularized Adversarial Training (AR-AT)来解决“梯度冲突”和混合分布问题，改善鲁棒性-准确性权衡。

    

    尽管对抗训练一直是抵抗对抗性样本（AEs）的最先进方法，但它们存在鲁棒性-准确性权衡问题。在这项研究中，我们重新审视基于表示的不变性正则化，学习具有辨别性却对抗性不变的表示，旨在缓解这种权衡。我们在经验上确定了妨碍不变性正则化的两个关键问题：（1）不变性损失和分类目标之间的“梯度冲突”，表明存在“崩溃解”，以及（2）由于干净和对抗性输入的分布发散而出现的混合分布问题。为了解决这些问题，我们提出了一种不对称表示正则化的对抗训练（AR-AT），该方法结合了一个停止梯度操作和一个预测器来避免“崩溃解”，灵感来自最近的非对比自监督学习。

    arXiv:2402.14648v1 Announce Type: cross  Abstract: Although adversarial training has been the state-of-the-art approach to defend against adversarial examples (AEs), they suffer from a robustness-accuracy trade-off. In this work, we revisit representation-based invariance regularization to learn discriminative yet adversarially invariant representations, aiming to mitigate this trade-off. We empirically identify two key issues hindering invariance regularization: (1) a "gradient conflict" between invariance loss and classification objectives, indicating the existence of "collapsing solutions," and (2) the mixture distribution problem arising from diverged distributions of clean and adversarial inputs. To address these issues, we propose Asymmetrically Representation-regularized Adversarial Training (AR-AT), which incorporates a stop-gradient operation and a pre-dictor in the invariance loss to avoid "collapsing solutions," inspired by a recent non-contrastive self-supervised learning a
    
[^2]: FLASH: 跨同时异质性的联邦学习

    FLASH: Federated Learning Across Simultaneous Heterogeneities

    [https://arxiv.org/abs/2402.08769](https://arxiv.org/abs/2402.08769)

    FLASH是一个跨同时异质性的联邦学习方法，通过综合考虑数据质量、数据分布和延迟等因素，优于其他现有的方法。

    

    联邦学习（FL）的关键前提是在不交换本地数据的情况下，训练机器学习模型在不同的数据所有者（客户端）之间。迄今为止，这个方法面临的一个重要挑战是客户端的异质性，这可能不仅来自数据分布的变化，还来自数据质量以及计算/通信延迟方面的差异。对这些不同且同时存在的异质性的综合视图至关重要；例如，延迟较低的客户端可能具有较差的数据质量，反之亦然。在这项工作中，我们提出了FLASH（跨同时异质性的联邦学习），一个轻量且灵活的客户端选择算法，通过权衡与客户端数据质量、数据分布和延迟相关的统计信息，优于最先进的FL框架。据我们所知，FLASH是第一个能够在统一的方法中处理所有这些异质性的方法。

    arXiv:2402.08769v1 Announce Type: new Abstract: The key premise of federated learning (FL) is to train ML models across a diverse set of data-owners (clients), without exchanging local data. An overarching challenge to this date is client heterogeneity, which may arise not only from variations in data distribution, but also in data quality, as well as compute/communication latency. An integrated view of these diverse and concurrent sources of heterogeneity is critical; for instance, low-latency clients may have poor data quality, and vice versa. In this work, we propose FLASH(Federated Learning Across Simultaneous Heterogeneities), a lightweight and flexible client selection algorithm that outperforms state-of-the-art FL frameworks under extensive sources of heterogeneity, by trading-off the statistical information associated with the client's data quality, data distribution, and latency. FLASH is the first method, to our knowledge, for handling all these heterogeneities in a unified m
    
[^3]: NetGPT：网络流量生成预训练变压器模型

    NetGPT: Generative Pretrained Transformer for Network Traffic. (arXiv:2304.09513v1 [cs.NI])

    [http://arxiv.org/abs/2304.09513](http://arxiv.org/abs/2304.09513)

    本文提出了首个网络流量生成预训练变压器模型NetGPT，该模型可以优化网络任务的训练效率和有效性。

    

    预训练模型可以利用大规模的原始数据学习网络流量的基本特征，并为输入流量生成可区分的结果，而不考虑特定的下游任务。有效的预训练模型可以显著优化下游任务的训练效率和有效性，例如流量分类、攻击检测、资源调度、协议分析和流量生成。本文提出了NetGPT，旨在为网络流量构建预训练模型并解决多样的挑战。

    Pretrained models for network traffic can utilize large-scale raw data to learn the essential characteristics of network traffic, and generate distinguishable results for input traffic without considering specific downstream tasks. Effective pretrained models can significantly optimize the training efficiency and effectiveness of downstream tasks, such as traffic classification, attack detection, resource scheduling, protocol analysis, and traffic generation. Despite the great success of pretraining in natural language processing, there is no work in the network field. Considering the diverse demands and characteristics of network traffic and network tasks, it is non-trivial to build a pretrained model for network traffic and we face various challenges, especially the heterogeneous headers and payloads in the multi-pattern network traffic and the different dependencies for contexts of diverse downstream network tasks.  To tackle these challenges, in this paper, we make the first attemp
    

