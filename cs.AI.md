# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Finite-Time Error Analysis of Online Model-Based Q-Learning with a Relaxed Sampling Model](https://arxiv.org/abs/2402.11877) | 本文通过有限时间分析以及实证评估，探讨了集成模型方法的Q学习在样本复杂度方面的优势。 |
| [^2] | [Deep Neural Networks: A Formulation Via Non-Archimedean Analysis](https://arxiv.org/abs/2402.00094) | 该论文引入了一种新的深度神经网络（DNNs）类别，其采用多层树状结构的架构并使用非阿基米德局部域的整数环进行编码。这些DNNs是稳健的对实值函数和实值平方可积函数的普遍逼近器。 |
| [^3] | [Semiring Provenance for Lightweight Description Logics.](http://arxiv.org/abs/2310.16472) | 这篇论文研究了在描述逻辑中使用半环溯源的框架，并定义了一种适用于轻量级描述逻辑的溯源语义。论文证明了在半环施加限制的情况下，语义满足一些重要的特性，并对why溯源方法进行了研究。 |
| [^4] | [Learning Expressive Priors for Generalization and Uncertainty Estimation in Neural Networks.](http://arxiv.org/abs/2307.07753) | 本文提出了一种用于神经网络的先验学习方法，通过利用可扩展和结构化的神经网络后验作为推广的信息先验，提高了神经网络的推广和不确定性估计能力。我们的方法在大规模上提供了表达性的概率表示，并产生了非空推广界限。我们的技术贡献是推导出可处理的目标函数，并提出了改进的推广界限计算方法。在经验上，我们证明了该方法在不确定性估计和推广方面的有效性。 |
| [^5] | [HEDI: First-Time Clinical Application and Results of a Biomechanical Evaluation and Visualisation Tool for Incisional Hernia Repair.](http://arxiv.org/abs/2307.01502) | HEDI是一种用于切口疝修复的生物力学评估和可视化工具，通过考虑腹壁的不稳定性，能够自动检测和评估疝的大小、体积和腹壁不稳定性。在31名患者的预手术评估中，HEDI显示出明显改善的成功率，所有患者在随访三年后仍然没有疼痛和疝再发。 |

# 详细

[^1]: 在具有放松采样模型的在线模型的有限时间误差分析下的Q学习

    Finite-Time Error Analysis of Online Model-Based Q-Learning with a Relaxed Sampling Model

    [https://arxiv.org/abs/2402.11877](https://arxiv.org/abs/2402.11877)

    本文通过有限时间分析以及实证评估，探讨了集成模型方法的Q学习在样本复杂度方面的优势。

    

    强化学习在模型为基础的方法的出现下取得了显著进展。在这些方法中，Q学习在无模型设置中被证明是一种强大的算法。然而，将Q学习扩展到基于模型的框架仍然相对未被探索。在本文中，我们深入研究了Q学习与基于模型方法相结合时的样本复杂度。通过理论分析和实证评估，我们试图阐明在哪些条件下，基于模型的Q学习在样本效率方面优于其无模型对应物。

    arXiv:2402.11877v1 Announce Type: cross  Abstract: Reinforcement learning has witnessed significant advancements, particularly with the emergence of model-based approaches. Among these, $Q$-learning has proven to be a powerful algorithm in model-free settings. However, the extension of $Q$-learning to a model-based framework remains relatively unexplored. In this paper, we delve into the sample complexity of $Q$-learning when integrated with a model-based approach. Through theoretical analyses and empirical evaluations, we seek to elucidate the conditions under which model-based $Q$-learning excels in terms of sample efficiency compared to its model-free counterpart.
    
[^2]: 深度神经网络: 非阿基米德分析的表述方式

    Deep Neural Networks: A Formulation Via Non-Archimedean Analysis

    [https://arxiv.org/abs/2402.00094](https://arxiv.org/abs/2402.00094)

    该论文引入了一种新的深度神经网络（DNNs）类别，其采用多层树状结构的架构并使用非阿基米德局部域的整数环进行编码。这些DNNs是稳健的对实值函数和实值平方可积函数的普遍逼近器。

    

    我们引入了一种新的深度神经网络（DNNs），采用多层树状结构的架构。这些架构使用非阿基米德局部域的整数环中的数字进行编码。这些环具有自然的层次结构，类似无限根树。这些环上的自然态射使我们能够构建有限的多层架构。新的DNNs是对在所提到的环上定义的实值函数的稳健的普遍逼近器。我们还证明了DNNs也是对在单位区间上定义的实值平方可积函数的稳健的普遍逼近器。

    We introduce a new class of deep neural networks (DNNs) with multilayered tree-like architectures. The architectures are codified using numbers from the ring of integers of non-Archimdean local fields. These rings have a natural hierarchical organization as infinite rooted trees. Natural morphisms on these rings allow us to construct finite multilayered architectures. The new DNNs are robust universal approximators of real-valued functions defined on the mentioned rings. We also show that the DNNs are robust universal approximators of real-valued square-integrable functions defined in the unit interval.
    
[^3]: 适用于轻量级描述逻辑的半环溯源

    Semiring Provenance for Lightweight Description Logics. (arXiv:2310.16472v1 [cs.LO])

    [http://arxiv.org/abs/2310.16472](http://arxiv.org/abs/2310.16472)

    这篇论文研究了在描述逻辑中使用半环溯源的框架，并定义了一种适用于轻量级描述逻辑的溯源语义。论文证明了在半环施加限制的情况下，语义满足一些重要的特性，并对why溯源方法进行了研究。

    

    我们研究了半环溯源——一种最初在关系数据库环境中定义的成功框架，用于描述逻辑。在此上下文中，本体公理被用交换半环的元素进行注释，并且这些注释根据它们的推导方式传播到本体的结果中。我们定义了一种溯源语义，适用于包括几种轻量级描述逻辑的语言，并展示了它与为带有特定类型注释（如模糊度）的本体定义的其他语义之间的关系。我们证明了在一些对半环施加限制的情况下，语义满足一些期望的特性（如扩展了数据库中定义的半环溯源）。然后我们专注于著名的why溯源方法，它允许计算每个加法幂等和乘法幂等的交换半环的半环溯源，并研究了与这种溯源方法相关的问题的复杂性。

    We investigate semiring provenance--a successful framework originally defined in the relational database setting--for description logics. In this context, the ontology axioms are annotated with elements of a commutative semiring and these annotations are propagated to the ontology consequences in a way that reflects how they are derived. We define a provenance semantics for a language that encompasses several lightweight description logics and show its relationships with semantics that have been defined for ontologies annotated with a specific kind of annotation (such as fuzzy degrees). We show that under some restrictions on the semiring, the semantics satisfies desirable properties (such as extending the semiring provenance defined for databases). We then focus on the well-known why-provenance, which allows to compute the semiring provenance for every additively and multiplicatively idempotent commutative semiring, and for which we study the complexity of problems related to the prov
    
[^4]: 学习神经网络中的表达性先验，提高推广和不确定性估计

    Learning Expressive Priors for Generalization and Uncertainty Estimation in Neural Networks. (arXiv:2307.07753v1 [cs.LG])

    [http://arxiv.org/abs/2307.07753](http://arxiv.org/abs/2307.07753)

    本文提出了一种用于神经网络的先验学习方法，通过利用可扩展和结构化的神经网络后验作为推广的信息先验，提高了神经网络的推广和不确定性估计能力。我们的方法在大规模上提供了表达性的概率表示，并产生了非空推广界限。我们的技术贡献是推导出可处理的目标函数，并提出了改进的推广界限计算方法。在经验上，我们证明了该方法在不确定性估计和推广方面的有效性。

    

    在这项工作中，我们提出了一种新的先验学习方法，用于提高深度神经网络中的推广和不确定性估计。关键思想是利用可扩展和结构化的神经网络后验作为具有推广保证的信息先验。我们学习到的先验在大规模上提供了表达性的概率表示，类似于在ImageNet上预训练模型的贝叶斯对应物，并进一步产生了非空推广界限。我们还将这个想法扩展到连续学习框架中，我们的先验的有利特性是可取的。主要的推动因素是我们的技术贡献：(1) Kronecker积求和的计算，(2) 推导和优化可处理的目标函数，从而导致改进的推广界限。在经验上，我们详尽地展示了该方法在不确定性估计和推广方面的有效性。

    In this work, we propose a novel prior learning method for advancing generalization and uncertainty estimation in deep neural networks. The key idea is to exploit scalable and structured posteriors of neural networks as informative priors with generalization guarantees. Our learned priors provide expressive probabilistic representations at large scale, like Bayesian counterparts of pre-trained models on ImageNet, and further produce non-vacuous generalization bounds. We also extend this idea to a continual learning framework, where the favorable properties of our priors are desirable. Major enablers are our technical contributions: (1) the sums-of-Kronecker-product computations, and (2) the derivations and optimizations of tractable objectives that lead to improved generalization bounds. Empirically, we exhaustively show the effectiveness of this method for uncertainty estimation and generalization.
    
[^5]: HEDI: 第一次临床应用的切口疝修复生物力学评估和可视化工具的结果

    HEDI: First-Time Clinical Application and Results of a Biomechanical Evaluation and Visualisation Tool for Incisional Hernia Repair. (arXiv:2307.01502v1 [cs.CV])

    [http://arxiv.org/abs/2307.01502](http://arxiv.org/abs/2307.01502)

    HEDI是一种用于切口疝修复的生物力学评估和可视化工具，通过考虑腹壁的不稳定性，能够自动检测和评估疝的大小、体积和腹壁不稳定性。在31名患者的预手术评估中，HEDI显示出明显改善的成功率，所有患者在随访三年后仍然没有疼痛和疝再发。

    

    腹壁缺陷通常导致疼痛、不适以及切口疝再发，全球范围内造成重大发病率和多次手术修复。对于大型疝，网格修复通常基于缺陷区域与固定重叠，而不考虑生物力学方面的因素，如肌肉激活、腹腔内压力、组织弹性和腹壁扩张。为了解决这个问题，我们提出了一种考虑不稳定腹壁的切口疝修复的生物力学方法。此外，我们介绍了HEDI，这是一种利用Valsalva动作的动态计算机断层扫描技术来自动检测和评估疝大小、体积和腹壁不稳定性的工具。我们在31名患者预手术评估中首次临床应用了HEDI，与报道的成功率相比，显示出明显改善，所有患者在随访三年后仍然没有疼痛和疝再发。

    Abdominal wall defects often lead to pain, discomfort, and recurrence of incisional hernias, resulting in significant morbidity and repeated surgical repairs worldwide. Mesh repair for large hernias is usually based on the defect area with a fixed overlap, without considering biomechanical aspects such as muscle activation, intra-abdominal pressure, tissue elasticity, and abdominal wall distention. To address this issue, we present a biomechanical approach to incisional hernia repair that takes into account the unstable abdominal wall. Additionally, we introduce HEDI, a tool that uses dynamic computed tomography with Valsalva maneuver to automatically detect and assess hernia size, volume, and abdominal wall instability. Our first clinical application of HEDI in the preoperative evaluation of 31 patients shows significantly improved success rates compared to reported rates, with all patients remaining pain-free and showing no hernia recurrence after three years of follow-up.
    

