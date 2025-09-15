# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Space Group Informed Transformer for Crystalline Materials Generation](https://arxiv.org/abs/2403.15734) | CrystalFormer是一种基于变压器的自回归模型，专门设计用于受空间群控制的晶体材料生成，它通过预测单位胞中对称不等价原子的种类和位置来生成晶体，并在有效性、新颖性和稳定性等方面达到了与最先进性能相匹配的水平。 |
| [^2] | [Unveiling Group-Specific Distributed Concept Drift: A Fairness Imperative in Federated Learning](https://arxiv.org/abs/2402.07586) | 该研究在联邦学习中的分布式环境下，首次探索了在存在特定群体概念漂移的情况下实现公平性的挑战和解决方案。 |
| [^3] | [Is Adversarial Training with Compressed Datasets Effective?](https://arxiv.org/abs/2402.05675) | 本论文研究了在压缩数据集上训练的模型对对抗鲁棒性的影响，并提出了一种同时提高数据集压缩效率和对抗鲁棒性的方法。 |
| [^4] | [Sufficient Invariant Learning for Distribution Shift.](http://arxiv.org/abs/2210.13533) | 本文研究了分布转移情况下的充分不变学习，观察到之前的工作只学习了部分不变特征，我们提出了学习充分不变特征的重要性，并指出在分布转移时，从训练集中学习的部分不变特征可能不适用于测试集，限制了性能提升。 |

# 详细

[^1]: 基于空间群信息的晶体材料生成变压器

    Space Group Informed Transformer for Crystalline Materials Generation

    [https://arxiv.org/abs/2403.15734](https://arxiv.org/abs/2403.15734)

    CrystalFormer是一种基于变压器的自回归模型，专门设计用于受空间群控制的晶体材料生成，它通过预测单位胞中对称不等价原子的种类和位置来生成晶体，并在有效性、新颖性和稳定性等方面达到了与最先进性能相匹配的水平。

    

    我们引入了CrystalFormer，这是一种基于变压器的自回归模型，专门设计用于受空间群控制的晶体材料生成。空间群对称性显著简化了晶体空间，这对于数据和计算有效的晶体材料生成建模至关重要。通过利用Wyckoff位置的显著离散和顺序特性，CrystalFormer学会了通过直接预测单位胞中对称不等价原子的种类和位置来生成晶体。我们的结果表明，CrystalFormer在生成的晶体材料的有效性、新颖性和稳定性方面与标准基准上的最新性能相匹配。我们的分析还表明，CrystalFormer从数据中吸收了合理的固体化学信息用于生成建模。CrystalFormer统一了基于对称性的结构搜索和生成性预训练。

    arXiv:2403.15734v1 Announce Type: cross  Abstract: We introduce CrystalFormer, a transformer-based autoregressive model specifically designed for space group-controlled generation of crystalline materials. The space group symmetry significantly simplifies the crystal space, which is crucial for data and compute efficient generative modeling of crystalline materials. Leveraging the prominent discrete and sequential nature of the Wyckoff positions, CrystalFormer learns to generate crystals by directly predicting the species and locations of symmetry-inequivalent atoms in the unit cell. Our results demonstrate that CrystalFormer matches state-of-the-art performance on standard benchmarks for both validity, novelty, and stability of the generated crystalline materials. Our analysis also shows that CrystalFormer ingests sensible solid-state chemistry information from data for generative modeling. The CrystalFormer unifies symmetry-based structure search and generative pre-training in the re
    
[^2]: 揭示特定群体的分布式概念漂移: 联邦学习中的公平要求

    Unveiling Group-Specific Distributed Concept Drift: A Fairness Imperative in Federated Learning

    [https://arxiv.org/abs/2402.07586](https://arxiv.org/abs/2402.07586)

    该研究在联邦学习中的分布式环境下，首次探索了在存在特定群体概念漂移的情况下实现公平性的挑战和解决方案。

    

    在机器学习领域的不断发展中，确保公平性已成为一个重要关注点，推动了开发旨在减少决策过程中歧视结果的算法。然而，在存在特定群体的概念漂移的情况下实现公平性仍然是一个未被探索的领域，我们的研究代表了在这方面的开拓性努力。特定群体的概念漂移是指一个群体随时间经历概念漂移，而另一个群体却没有，导致公平性下降，即使准确性保持相对稳定。在联邦学习的框架下，客户端共同训练模型，其分布式性质进一步放大了这些挑战，因为每个客户端可以独立经历特定群体的概念漂移，同时仍共享相同的基本概念，从而创造了一个复杂而动态的环境来维持公平性。我们研究的一个重要贡献之一是对群体特定的概念漂移进行形式化和内部化的过程。

    In the evolving field of machine learning, ensuring fairness has become a critical concern, prompting the development of algorithms designed to mitigate discriminatory outcomes in decision-making processes. However, achieving fairness in the presence of group-specific concept drift remains an unexplored frontier, and our research represents pioneering efforts in this regard. Group-specific concept drift refers to situations where one group experiences concept drift over time while another does not, leading to a decrease in fairness even if accuracy remains fairly stable. Within the framework of federated learning, where clients collaboratively train models, its distributed nature further amplifies these challenges since each client can experience group-specific concept drift independently while still sharing the same underlying concept, creating a complex and dynamic environment for maintaining fairness. One of the significant contributions of our research is the formalization and intr
    
[^3]: 压缩数据集的对抗训练是否有效？

    Is Adversarial Training with Compressed Datasets Effective?

    [https://arxiv.org/abs/2402.05675](https://arxiv.org/abs/2402.05675)

    本论文研究了在压缩数据集上训练的模型对对抗鲁棒性的影响，并提出了一种同时提高数据集压缩效率和对抗鲁棒性的方法。

    

    数据集压缩（DC）是指从较大数据集中生成较小的合成数据集的一类最近的数据集压缩方法。这个合成数据集保留了原始数据集的基本信息，使得在其上训练的模型能够达到与在完整数据集上训练的模型相当的性能水平。目前大多数的DC方法主要关注如何在有限的数据预算下实现高测试性能，并没有直接解决对抗鲁棒性的问题。在本工作中，我们研究了在压缩数据集上训练的模型对对抗鲁棒性的影响。我们发现从DC方法获得的压缩数据集对模型的对抗鲁棒性没有有效的传递性。为了同时提高数据集压缩效率和对抗鲁棒性，我们提出了一种基于寻找数据集的最小有限覆盖（MFC）的新型鲁棒性感知数据集压缩方法。

    Dataset Condensation (DC) refers to the recent class of dataset compression methods that generate a smaller, synthetic, dataset from a larger dataset. This synthetic dataset retains the essential information of the original dataset, enabling models trained on it to achieve performance levels comparable to those trained on the full dataset. Most current DC methods have mainly concerned with achieving high test performance with limited data budget, and have not directly addressed the question of adversarial robustness. In this work, we investigate the impact of adversarial robustness on models trained with compressed datasets. We show that the compressed datasets obtained from DC methods are not effective in transferring adversarial robustness to models. As a solution to improve dataset compression efficiency and adversarial robustness simultaneously, we propose a novel robustness-aware dataset compression method based on finding the Minimal Finite Covering (MFC) of the dataset. The prop
    
[^4]: 分布转移的充分不变学习

    Sufficient Invariant Learning for Distribution Shift. (arXiv:2210.13533v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.13533](http://arxiv.org/abs/2210.13533)

    本文研究了分布转移情况下的充分不变学习，观察到之前的工作只学习了部分不变特征，我们提出了学习充分不变特征的重要性，并指出在分布转移时，从训练集中学习的部分不变特征可能不适用于测试集，限制了性能提升。

    

    机器学习算法在各种应用中展现出了卓越的性能。然而，在训练集和测试集的分布不同的情况下，保证性能仍然具有挑战性。为了改善分布转移情况下的性能，已经提出了一些方法，通过学习跨组或领域的不变特征来提高性能。然而，我们观察到之前的工作只部分地学习了不变特征。虽然先前的工作侧重于有限的不变特征，但我们首次提出了充分不变特征的重要性。由于只有训练集是经验性的，从训练集中学习得到的部分不变特征可能不存在于分布转移时的测试集中。因此，分布转移情况下的性能提高可能受到限制。本文认为从训练集中学习充分的不变特征对于分布转移情况至关重要。

    Machine learning algorithms have shown remarkable performance in diverse applications. However, it is still challenging to guarantee performance in distribution shifts when distributions of training and test datasets are different. There have been several approaches to improve the performance in distribution shift cases by learning invariant features across groups or domains. However, we observe that the previous works only learn invariant features partially. While the prior works focus on the limited invariant features, we first raise the importance of the sufficient invariant features. Since only training sets are given empirically, the learned partial invariant features from training sets might not be present in the test sets under distribution shift. Therefore, the performance improvement on distribution shifts might be limited. In this paper, we argue that learning sufficient invariant features from the training set is crucial for the distribution shift case. Concretely, we newly 
    

