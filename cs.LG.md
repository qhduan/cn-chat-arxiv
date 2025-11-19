# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Achieving $\tilde{O}(1/\epsilon)$ Sample Complexity for Constrained Markov Decision Process](https://arxiv.org/abs/2402.16324) | 该论文提出了一种算法，在约束马尔可夫决策过程中实现了约$O(1/\epsilon)$的样本复杂度，相比先前文献中已有的$O(1/\epsilon^2)$样本复杂度有所提升。 |
| [^2] | [High Dimensional Distributed Gradient Descent with Arbitrary Number of Byzantine Attackers.](http://arxiv.org/abs/2307.13352) | 本文提出了一种适用于高维问题、在任意数量拜占庭攻击者下的新方法，核心是一种直接的高维半验证均值估计方法，具有极小极值统计率。 |
| [^3] | [Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics.](http://arxiv.org/abs/2306.10656) | 本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。 |

# 详细

[^1]: 实现约$O(1/\epsilon)$的样本复杂度用于约束马尔可夫决策过程

    Achieving $\tilde{O}(1/\epsilon)$ Sample Complexity for Constrained Markov Decision Process

    [https://arxiv.org/abs/2402.16324](https://arxiv.org/abs/2402.16324)

    该论文提出了一种算法，在约束马尔可夫决策过程中实现了约$O(1/\epsilon)$的样本复杂度，相比先前文献中已有的$O(1/\epsilon^2)$样本复杂度有所提升。

    

    我们考虑约束马尔可夫决策过程（CMDP）的强化学习问题，在顺序学习和决策中满足安全性或资源约束方面起着关键作用。在这个问题中，我们拥有有限资源和未知转移概率的MDP。在每个阶段，我们采取一个行动，收集奖励并消耗一些资源，所有假设都是未知的，并且需要随着时间学习。在这项工作中，我们迈出了为CMDP问题推导出最优的问题相关保证的第一步。我们得出了一个对数遗憾界限，这转化为$O(\frac{\kappa}{\epsilon}\cdot\log^2(1/\epsilon))$的样本复杂度界限，其中$\kappa$是一个与问题相关的参数，但与$\epsilon$无关。我们的样本复杂度界限改进了先前文献中针对CMDP问题建立的$O(1/\epsilon^2)$样本复杂度。

    arXiv:2402.16324v1 Announce Type: new  Abstract: We consider the reinforcement learning problem for the constrained Markov decision process (CMDP), which plays a central role in satisfying safety or resource constraints in sequential learning and decision-making. In this problem, we are given finite resources and a MDP with unknown transition probabilities. At each stage, we take an action, collecting a reward and consuming some resources, all assumed to be unknown and need to be learned over time. In this work, we take the first step towards deriving optimal problem-dependent guarantees for the CMDP problems. We derive a logarithmic regret bound, which translates into a $O(\frac{\kappa}{\epsilon}\cdot\log^2(1/\epsilon))$ sample complexity bound, with $\kappa$ being a problem-dependent parameter, yet independent of $\epsilon$. Our sample complexity bound improves upon the state-of-art $O(1/\epsilon^2)$ sample complexity for CMDP problems established in the previous literature, in terms
    
[^2]: 高维分布式梯度下降算法在任意数量拜占庭攻击者下的研究

    High Dimensional Distributed Gradient Descent with Arbitrary Number of Byzantine Attackers. (arXiv:2307.13352v1 [cs.LG])

    [http://arxiv.org/abs/2307.13352](http://arxiv.org/abs/2307.13352)

    本文提出了一种适用于高维问题、在任意数量拜占庭攻击者下的新方法，核心是一种直接的高维半验证均值估计方法，具有极小极值统计率。

    

    近年来，具有拜占庭故障的强鲁棒分布式学习引起了广泛关注。然而，现有方法大多受到维度诅咒的限制，随着现代机器学习模型复杂性的增加，这个问题变得越来越严重。在本文中，我们设计了一种适用于高维问题、在任意数量拜占庭攻击者下的新方法。我们的设计核心是一种直接的高维半验证均值估计方法。我们的想法是首先识别一个子空间，通过工作机上传的梯度向量估计与该子空间垂直的均值分量，而通过辅助数据集估计该子空间内的均值分量。然后，我们将我们的新方法用作分布式学习问题的聚合器。我们的理论分析表明，新方法具有极小极值统计率。特别地，对维度的依赖性得到了显著改善。

    Robust distributed learning with Byzantine failures has attracted extensive research interests in recent years. However, most of existing methods suffer from curse of dimensionality, which is increasingly serious with the growing complexity of modern machine learning models. In this paper, we design a new method that is suitable for high dimensional problems, under arbitrary number of Byzantine attackers. The core of our design is a direct high dimensional semi-verified mean estimation method. Our idea is to identify a subspace first. The components of mean value perpendicular to this subspace can be estimated via gradient vectors uploaded from worker machines, while the components within this subspace are estimated using auxiliary dataset. We then use our new method as the aggregator of distributed learning problems. Our theoretical analysis shows that the new method has minimax optimal statistical rates. In particular, the dependence on dimensionality is significantly improved compar
    
[^3]: 虚拟人类生成模型：基于掩码建模的方法来学习人类特征

    Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics. (arXiv:2306.10656v1 [cs.LG])

    [http://arxiv.org/abs/2306.10656](http://arxiv.org/abs/2306.10656)

    本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。

    

    识别医疗属性、生活方式和人格之间的关系对于理解和改善身体和精神状况至关重要。本文提出了一种名为虚拟人类生成模型（VHGM）的机器学习模型，用于估计有关医疗保健、生活方式和个性的属性。VHGM是一个深度生成模型，使用掩码建模训练，在已知属性的条件下学习属性的联合分布。利用异构表格数据集，VHGM高效地学习了超过1,800个属性。我们数值评估了VHGM及其训练技术的性能。作为VHGM的概念验证，我们提出了几个应用程序，演示了用户情境，例如医疗属性的虚拟测量和生活方式的假设验证。

    Identifying the relationship between healthcare attributes, lifestyles, and personality is vital for understanding and improving physical and mental conditions. Machine learning approaches are promising for modeling their relationships and offering actionable suggestions. In this paper, we propose Virtual Human Generative Model (VHGM), a machine learning model for estimating attributes about healthcare, lifestyles, and personalities. VHGM is a deep generative model trained with masked modeling to learn the joint distribution of attributes conditioned on known ones. Using heterogeneous tabular datasets, VHGM learns more than 1,800 attributes efficiently. We numerically evaluate the performance of VHGM and its training techniques. As a proof-of-concept of VHGM, we present several applications demonstrating user scenarios, such as virtual measurements of healthcare attributes and hypothesis verifications of lifestyles.
    

