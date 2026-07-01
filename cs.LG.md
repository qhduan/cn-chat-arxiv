# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding Domain-Aware Distribution Alignment in Budgeted Entity Matching](https://arxiv.org/abs/2606.27342) | 本文通过系统实验揭示了BEACON框架在预算约束下进行领域感知分布对齐的性能表现，为低资源实体匹配提供了关键见解。 |
| [^2] | [The Double-Edged Sword of Input Perturbations to Robust Accurate Fairness](https://arxiv.org/abs/2404.01356) | 该论文研究了深度神经网络对敌对输入扰动的敏感性，提出了新的鲁棒准确公平性定义，并介绍了一种敌对攻击方法和相应的解决方案。 |
| [^3] | [Corruption Robust Offline Reinforcement Learning with Human Feedback](https://arxiv.org/abs/2402.06734) | 我们研究了具有人类反馈的强化学习中的数据腐败鲁棒性问题，并设计了新颖的离线方法来处理损坏的数据，并且在不同的数据生成分布假设下具有性能保证。 |

# 详细

[^1]: 预算限制下的实体匹配中领域感知分布对齐的理解

    Understanding Domain-Aware Distribution Alignment in Budgeted Entity Matching

    [https://arxiv.org/abs/2606.27342](https://arxiv.org/abs/2606.27342)

    本文通过系统实验揭示了BEACON框架在预算约束下进行领域感知分布对齐的性能表现，为低资源实体匹配提供了关键见解。

    

    实体匹配（EM）是数据集成流程中的核心操作，它通过比较来自不同数据源的记录，判断它们是否指向同一真实世界实体。近期研究融入了领域信息和低资源学习技术，以更好地使EM系统适应现实场景。尽管这些方法展现出强大性能，但在实践中，它们在不同数据约束和监督程度下的表现尚不明确。本文研究了一种先进的低资源、领域感知EM方法——BEACON，并探讨了不同算法选择和数据可用性条件对其性能的影响。我们通过一系列针对性实验来评估这些变化，从而更深入地理解分布对齐的作用以及BEACON框架的行为特性。

    arXiv:2606.27342v1 Announce Type: cross  Abstract: Entity Matching (EM) is a core operation in the data integration pipeline, where records from different sources are compared to determine whether they refer to the same real-world entity. Recent work has incorporated domain information and low-resource learning techniques to better adapt EM systems to realistic settings. While these approaches have demonstrated strong performance, it remains unclear how they behave under varying data constraints and levels of supervision in practice. In this paper, we investigate a state-of-the-art method for low-resource, domain-aware EM--BEACON--and study how its performance is affected by different algorithmic choices and data availability conditions. We conduct a series of targeted experiments to evaluate these variations, providing deeper insight into the role of distribution alignment and the behavior of the BEACON framework.
    
[^2]: 输入扰动对鲁棒准确公平性的双刃剑

    The Double-Edged Sword of Input Perturbations to Robust Accurate Fairness

    [https://arxiv.org/abs/2404.01356](https://arxiv.org/abs/2404.01356)

    该论文研究了深度神经网络对敌对输入扰动的敏感性，提出了新的鲁棒准确公平性定义，并介绍了一种敌对攻击方法和相应的解决方案。

    

    深度神经网络(DNNs)被认为对敌对输入扰动敏感，导致预测的准确性或个体公平性降低。为了共同表征预测准确性和个体公平性对敌对扰动的敏感性，我们引入了一个名为鲁棒准确公平性的新定义。鲁棒准确公平性要求当实例及其相似对应物受到输入扰动时，预测与地面事实一致。我们提出一种敌对攻击方法RAFair，以暴露DNN中的虚假或偏见敌对缺陷，这些缺陷会欺骗准确性或损害个体公平性。然后，我们展示这样的敌对实例可以通过精心设计的良性扰动有效地解决，从而使它们的预测准确而公平。我们的工作探讨了输入对准确公平性的双刃剑。

    arXiv:2404.01356v1 Announce Type: cross  Abstract: Deep neural networks (DNNs) are known to be sensitive to adversarial input perturbations, leading to a reduction in either prediction accuracy or individual fairness. To jointly characterize the susceptibility of prediction accuracy and individual fairness to adversarial perturbations, we introduce a novel robustness definition termed robust accurate fairness. Informally, robust accurate fairness requires that predictions for an instance and its similar counterparts consistently align with the ground truth when subjected to input perturbations. We propose an adversarial attack approach dubbed RAFair to expose false or biased adversarial defects in DNN, which either deceive accuracy or compromise individual fairness. Then, we show that such adversarial instances can be effectively addressed by carefully designed benign perturbations, correcting their predictions to be accurate and fair. Our work explores the double-edged sword of input 
    
[^3]: 具有人类反馈的抗腐败离线强化学习

    Corruption Robust Offline Reinforcement Learning with Human Feedback

    [https://arxiv.org/abs/2402.06734](https://arxiv.org/abs/2402.06734)

    我们研究了具有人类反馈的强化学习中的数据腐败鲁棒性问题，并设计了新颖的离线方法来处理损坏的数据，并且在不同的数据生成分布假设下具有性能保证。

    

    我们研究了在离线环境中具有人类反馈的强化学习中的数据腐败鲁棒性问题。给定一组离线数据，其中包括轨迹对以及有关人类偏好的反馈，其中$\varepsilon$比例的轨迹对被损坏（例如，反馈翻转或轨迹特征被操纵），从而捕捉到对抗攻击或噪声人类偏好的影响。我们旨在设计算法，从损坏的数据中识别出接近最优的策略，并且具备可证明的保证。现有的理论研究分别研究了腐败鲁棒强化学习（在腐败下直接学习标量奖励）和离线强化学习（在没有腐败的情况下从人类反馈中学习）的设置；然而，它们并不适用于我们处理在离线环境中的损坏数据的问题。为此，我们设计了新颖的在数据生成分布覆盖各种假设下具有腐败鲁棒性的离线强化学习方法。在高层次上，我们的方法具有鲁棒亮点，并确保在不同的数据生成分布假设下的性能保证。

    We study data corruption robustness for reinforcement learning with human feedback (RLHF) in an offline setting. Given an offline dataset of pairs of trajectories along with feedback about human preferences, an $\varepsilon$-fraction of the pairs is corrupted (e.g., feedback flipped or trajectory features manipulated), capturing an adversarial attack or noisy human preferences. We aim to design algorithms that identify a near-optimal policy from the corrupted data, with provable guarantees. Existing theoretical works have separately studied the settings of corruption robust RL (learning from scalar rewards directly under corruption) and offline RLHF (learning from human feedback without corruption); however, they are inapplicable to our problem of dealing with corrupted data in offline RLHF setting. To this end, we design novel corruption robust offline RLHF methods under various assumptions on the coverage of the data-generating distributions. At a high level, our methodology robustif
    

