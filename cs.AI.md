# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding Domain-Aware Distribution Alignment in Budgeted Entity Matching](https://arxiv.org/abs/2606.27342) | 本文通过系统实验揭示了BEACON框架在预算约束下进行领域感知分布对齐的性能表现，为低资源实体匹配提供了关键见解。 |
| [^2] | [OpenRCA 2.0: From Outcome Labels to Causal Process Supervision](https://arxiv.org/abs/2606.27154) | 本文提出了PAVE协议和OpenRCA 2.0基准，通过前向验证的因果传播路径标注，揭示了LLM在根因分析中平均仅20.7%的成功率，突出了现有方法在复杂因果推理上的局限性。 |
| [^3] | [Clinical Harness for Governable Medical AI Skill Ecosystems](https://arxiv.org/abs/2606.26494) | 本文提出一种名为“临床管控框架”的运行时治理架构，通过注册、编排、防护和监控AI技能，解决医疗AI模型孤立化问题，并以骨质疏松症为例验证其在全生命周期护理中的有效性。 |
| [^4] | [A Concept of Possibility for Real-World Events](https://arxiv.org/abs/2510.02655) | 本文提出了一种针对现实世界事件的新可能性概念，通过考虑事件的先决条件和约束条件的概率来计算可能性，为规划问题提供了新的决策依据。 |
| [^5] | [The Double-Edged Sword of Input Perturbations to Robust Accurate Fairness](https://arxiv.org/abs/2404.01356) | 该论文研究了深度神经网络对敌对输入扰动的敏感性，提出了新的鲁棒准确公平性定义，并介绍了一种敌对攻击方法和相应的解决方案。 |
| [^6] | [Corruption Robust Offline Reinforcement Learning with Human Feedback](https://arxiv.org/abs/2402.06734) | 我们研究了具有人类反馈的强化学习中的数据腐败鲁棒性问题，并设计了新颖的离线方法来处理损坏的数据，并且在不同的数据生成分布假设下具有性能保证。 |

# 详细

[^1]: 预算限制下的实体匹配中领域感知分布对齐的理解

    Understanding Domain-Aware Distribution Alignment in Budgeted Entity Matching

    [https://arxiv.org/abs/2606.27342](https://arxiv.org/abs/2606.27342)

    本文通过系统实验揭示了BEACON框架在预算约束下进行领域感知分布对齐的性能表现，为低资源实体匹配提供了关键见解。

    

    实体匹配（EM）是数据集成流程中的核心操作，它通过比较来自不同数据源的记录，判断它们是否指向同一真实世界实体。近期研究融入了领域信息和低资源学习技术，以更好地使EM系统适应现实场景。尽管这些方法展现出强大性能，但在实践中，它们在不同数据约束和监督程度下的表现尚不明确。本文研究了一种先进的低资源、领域感知EM方法——BEACON，并探讨了不同算法选择和数据可用性条件对其性能的影响。我们通过一系列针对性实验来评估这些变化，从而更深入地理解分布对齐的作用以及BEACON框架的行为特性。

    arXiv:2606.27342v1 Announce Type: cross  Abstract: Entity Matching (EM) is a core operation in the data integration pipeline, where records from different sources are compared to determine whether they refer to the same real-world entity. Recent work has incorporated domain information and low-resource learning techniques to better adapt EM systems to realistic settings. While these approaches have demonstrated strong performance, it remains unclear how they behave under varying data constraints and levels of supervision in practice. In this paper, we investigate a state-of-the-art method for low-resource, domain-aware EM--BEACON--and study how its performance is affected by different algorithmic choices and data availability conditions. We conduct a series of targeted experiments to evaluate these variations, providing deeper insight into the role of distribution alignment and the behavior of the BEACON framework.
    
[^2]: OpenRCA 2.0：从结果标签到因果过程监督

    OpenRCA 2.0: From Outcome Labels to Causal Process Supervision

    [https://arxiv.org/abs/2606.27154](https://arxiv.org/abs/2606.27154)

    本文提出了PAVE协议和OpenRCA 2.0基准，通过前向验证的因果传播路径标注，揭示了LLM在根因分析中平均仅20.7%的成功率，突出了现有方法在复杂因果推理上的局限性。

    

    arXiv:2606.27154v1 公告类型：新文 摘要：根因分析（RCA）是对LLM代理能力（如长上下文理解、多步推理和工具使用）的整体考验。然而，现有数据集存在一个根本性差距：它们仅标记根因，而非连接根因与观察到的症状的传播路径，这大大简化了任务，使其退化为简单的模式匹配。为支持严格评估，我们引入了PAVE，一种逐步标记协议，利用故障注入中的已知干预来重建因果传播路径。其机制是前向验证：从原因推理到结果，而非从症状向后推断。应用PAVE生成了OpenRCA 2.0（500个实例），这是首个具有逐步因果注释的跨系统RCA基准，专为LLM代理设计。在11个前沿LLM中，平均仅20.7%的案例能成功恢复确切的根因集。为定位这一难点所在，我们放宽了...

    arXiv:2606.27154v1 Announce Type: new  Abstract: Root cause analysis (RCA) poses a holistic test of LLM agentic capabilities, such as long-context understanding, multi-step reasoning, and tool use. However, existing datasets suffer from a fundamental gap: they label only the root cause, not the propagation path connecting it to the observed symptom, which largely simplifies the task to naive pattern matching. To support rigorous evaluation, we introduce PAVE, a step-wise labeling protocol that leverages known interventions from fault injection to reconstruct causal propagation paths. The mechanism is forward verification: reasoning from cause to effect rather than inferring backward from symptoms. Applying PAVE yields OpenRCA 2.0 (500 instances), the first cross-system RCA benchmark with step-wise causal annotations for LLM agents. Across 11 frontier LLMs, recovering the exact root-cause set succeeds in only 20.7% of cases on average. To locate where this difficulty lies, we relax the 
    
[^3]: 可治理医疗AI技能生态系统的临床管控框架

    Clinical Harness for Governable Medical AI Skill Ecosystems

    [https://arxiv.org/abs/2606.26494](https://arxiv.org/abs/2606.26494)

    本文提出一种名为“临床管控框架”的运行时治理架构，通过注册、编排、防护和监控AI技能，解决医疗AI模型孤立化问题，并以骨质疏松症为例验证其在全生命周期护理中的有效性。

    

    arXiv:2606.26494v1 公告类型：新论文 摘要：医疗人工智能目前仍围绕孤立模型组织，而临床护理需要跨时间持续存在的可问责能力。我们提出临床AI技能与临床管控框架：一种用于注册、编排、防护和监控AI驱动临床能力的运行时治理架构。以骨质疏松症为例，我们展示了知识驱动、数据驱动和物理增强的技能如何在运行时治理下支持全生命周期护理。

    arXiv:2606.26494v1 Announce Type: new  Abstract: Medical AI remains organized around isolated models, whereas clinical care requires accountable capabilities that persist across time. We propose clinical AI skills and the Clinical Harness: a runtime governance architecture for registering, orchestrating, guarding and monitoring AI-enabled clinical capabilities. Using osteoporosis as an exemplar, we show how knowledge-driven, data-driven and physics-enhanced skills can support lifecycle care under runtime governance.
    
[^4]: 现实世界事件的可能性概念

    A Concept of Possibility for Real-World Events

    [https://arxiv.org/abs/2510.02655](https://arxiv.org/abs/2510.02655)

    本文提出了一种针对现实世界事件的新可能性概念，通过考虑事件的先决条件和约束条件的概率来计算可能性，为规划问题提供了新的决策依据。

    

    arXiv:2510.02655v2 公告类型：替换 摘要：本文提出了一种新的“可能性”概念，作为对L.A. Zadeh于1978年引入的现代标准概念的替代。这一新版本受原始概念启发，但在形式上与原始概念毫无共同之处，除了两者都采用Łukasiewicz多值逻辑联结词的解释。此外，本文并非试图提供一种普遍的可能性概念，而是专门聚焦于现实世界事件的可能性。一个事件被视为具有使其发生的先决条件和可能阻碍其发生的约束条件，而事件的可能性被计算为先决条件成立且约束条件不成立的概率函数。这一版本的可能性可恰当地应用于规划问题。当存在多个实现目标的计划时，该理论可用于确定哪个计划最可行。

    arXiv:2510.02655v2 Announce Type: replace  Abstract: This paper offers a new concept of {\it possibility} as an alternative to the now-a-days standard concept originally introduced by L.A. Zadeh in 1978. This new version was inspired by the original but, formally, has nothing in common with it other than that they both adopt the {\L}ukasiewicz multivalent interpretation of the logical connectives. Moreover, rather than seeking to provide a general notion of possibility, this focuses specifically on the possibility of a real-world event. An event is viewed as having prerequisites that enable its occurrence and constraints that may impede its occurrence, and the possibility of the event is computed as a function of the probabilities that the prerequisites hold and the constraints do not. This version of possibility might appropriately be applied to problems of planning. When there are multiple plans available for achieving a goal, this theory can be used to determine which plan is most p
    
[^5]: 输入扰动对鲁棒准确公平性的双刃剑

    The Double-Edged Sword of Input Perturbations to Robust Accurate Fairness

    [https://arxiv.org/abs/2404.01356](https://arxiv.org/abs/2404.01356)

    该论文研究了深度神经网络对敌对输入扰动的敏感性，提出了新的鲁棒准确公平性定义，并介绍了一种敌对攻击方法和相应的解决方案。

    

    深度神经网络(DNNs)被认为对敌对输入扰动敏感，导致预测的准确性或个体公平性降低。为了共同表征预测准确性和个体公平性对敌对扰动的敏感性，我们引入了一个名为鲁棒准确公平性的新定义。鲁棒准确公平性要求当实例及其相似对应物受到输入扰动时，预测与地面事实一致。我们提出一种敌对攻击方法RAFair，以暴露DNN中的虚假或偏见敌对缺陷，这些缺陷会欺骗准确性或损害个体公平性。然后，我们展示这样的敌对实例可以通过精心设计的良性扰动有效地解决，从而使它们的预测准确而公平。我们的工作探讨了输入对准确公平性的双刃剑。

    arXiv:2404.01356v1 Announce Type: cross  Abstract: Deep neural networks (DNNs) are known to be sensitive to adversarial input perturbations, leading to a reduction in either prediction accuracy or individual fairness. To jointly characterize the susceptibility of prediction accuracy and individual fairness to adversarial perturbations, we introduce a novel robustness definition termed robust accurate fairness. Informally, robust accurate fairness requires that predictions for an instance and its similar counterparts consistently align with the ground truth when subjected to input perturbations. We propose an adversarial attack approach dubbed RAFair to expose false or biased adversarial defects in DNN, which either deceive accuracy or compromise individual fairness. Then, we show that such adversarial instances can be effectively addressed by carefully designed benign perturbations, correcting their predictions to be accurate and fair. Our work explores the double-edged sword of input 
    
[^6]: 具有人类反馈的抗腐败离线强化学习

    Corruption Robust Offline Reinforcement Learning with Human Feedback

    [https://arxiv.org/abs/2402.06734](https://arxiv.org/abs/2402.06734)

    我们研究了具有人类反馈的强化学习中的数据腐败鲁棒性问题，并设计了新颖的离线方法来处理损坏的数据，并且在不同的数据生成分布假设下具有性能保证。

    

    我们研究了在离线环境中具有人类反馈的强化学习中的数据腐败鲁棒性问题。给定一组离线数据，其中包括轨迹对以及有关人类偏好的反馈，其中$\varepsilon$比例的轨迹对被损坏（例如，反馈翻转或轨迹特征被操纵），从而捕捉到对抗攻击或噪声人类偏好的影响。我们旨在设计算法，从损坏的数据中识别出接近最优的策略，并且具备可证明的保证。现有的理论研究分别研究了腐败鲁棒强化学习（在腐败下直接学习标量奖励）和离线强化学习（在没有腐败的情况下从人类反馈中学习）的设置；然而，它们并不适用于我们处理在离线环境中的损坏数据的问题。为此，我们设计了新颖的在数据生成分布覆盖各种假设下具有腐败鲁棒性的离线强化学习方法。在高层次上，我们的方法具有鲁棒亮点，并确保在不同的数据生成分布假设下的性能保证。

    We study data corruption robustness for reinforcement learning with human feedback (RLHF) in an offline setting. Given an offline dataset of pairs of trajectories along with feedback about human preferences, an $\varepsilon$-fraction of the pairs is corrupted (e.g., feedback flipped or trajectory features manipulated), capturing an adversarial attack or noisy human preferences. We aim to design algorithms that identify a near-optimal policy from the corrupted data, with provable guarantees. Existing theoretical works have separately studied the settings of corruption robust RL (learning from scalar rewards directly under corruption) and offline RLHF (learning from human feedback without corruption); however, they are inapplicable to our problem of dealing with corrupted data in offline RLHF setting. To this end, we design novel corruption robust offline RLHF methods under various assumptions on the coverage of the data-generating distributions. At a high level, our methodology robustif
    

