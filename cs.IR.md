# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Pattern-wise Transparent Sequential Recommendation](https://arxiv.org/abs/2402.11480) | 提出了一种模式透明的顺序推荐框架，通过将项目序列分解为多级模式并在概率空间中量化每个模式对结果的贡献，实现了透明的决策过程。 |
| [^2] | [A Comparative Evaluation of Quantification Methods.](http://arxiv.org/abs/2103.03223) | 本研究通过对24种不同量化方法在超过40个数据集上进行全面实证比较，填补了量化方法比较研究的空白。我们发现在二分类设置中，基于阈值选择的Median Sweep和TSMax方法、DyS框架和弗里德曼的方法表现最佳；而在多分类设置中，Generaliz方法表现良好。 |

# 详细

[^1]: 模式透明的顺序推荐

    Pattern-wise Transparent Sequential Recommendation

    [https://arxiv.org/abs/2402.11480](https://arxiv.org/abs/2402.11480)

    提出了一种模式透明的顺序推荐框架，通过将项目序列分解为多级模式并在概率空间中量化每个模式对结果的贡献，实现了透明的决策过程。

    

    透明的决策过程对于开发可靠和值得信赖的推荐系统至关重要。对于顺序推荐来说，意味着模型能够识别关键项目作为其推荐结果的理由。然而，同时实现模型透明度和推荐性能是具有挑战性的，特别是对于将整个项目序列作为输入而不加筛选的模型而言。在本文中，我们提出了一种名为PTSR的可解释框架，它实现了一种模式透明的决策过程。它将项目序列分解为多级模式，这些模式作为整个推荐过程的原子单元。每个模式对结果的贡献在概率空间中得到量化。通过精心设计的模式加权校正，即使在没有真实关键模式的情况下，也能学习模式的贡献。最终推荐

    arXiv:2402.11480v1 Announce Type: new  Abstract: A transparent decision-making process is essential for developing reliable and trustworthy recommender systems. For sequential recommendation, it means that the model can identify critical items asthe justifications for its recommendation results. However, achieving both model transparency and recommendation performance simultaneously is challenging, especially for models that take the entire sequence of items as input without screening. In this paper,we propose an interpretable framework (named PTSR) that enables a pattern-wise transparent decision-making process. It breaks the sequence of items into multi-level patterns that serve as atomic units for the entire recommendation process. The contribution of each pattern to the outcome is quantified in the probability space. With a carefully designed pattern weighting correction, the pattern contribution can be learned in the absence of ground-truth critical patterns. The final recommended
    
[^2]: 量化方法的比较评估

    A Comparative Evaluation of Quantification Methods. (arXiv:2103.03223v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2103.03223](http://arxiv.org/abs/2103.03223)

    本研究通过对24种不同量化方法在超过40个数据集上进行全面实证比较，填补了量化方法比较研究的空白。我们发现在二分类设置中，基于阈值选择的Median Sweep和TSMax方法、DyS框架和弗里德曼的方法表现最佳；而在多分类设置中，Generaliz方法表现良好。

    

    量化是指在数据集中预测类别分布的问题。它也代表着一个在监督式机器学习中不断发展的研究领域，近年来提出了大量不同的算法。然而，目前还没有一份全面的实证比较量化方法的研究，以支持算法选择。在本研究中，我们通过对超过40个数据集进行了24种不同量化方法的彻底实证性性能比较，包括二分类和多分类量化设置，填补了这一研究空白。我们观察到没有单一算法能够在所有竞争对手中始终表现最佳，但我们确定了一组在二分类设置中表现最佳的方法，包括基于阈值选择的Median Sweep和TSMax方法、DyS框架和弗里德曼的方法。对于多分类设置，我们观察到另一组算法表现良好，包括Generaliz方法。

    Quantification represents the problem of predicting class distributions in a dataset. It also represents a growing research field in supervised machine learning, for which a large variety of different algorithms has been proposed in recent years. However, a comprehensive empirical comparison of quantification methods that supports algorithm selection is not available yet. In this work, we close this research gap by conducting a thorough empirical performance comparison of 24 different quantification methods on overall more than 40 data sets, considering binary as well as multiclass quantification settings. We observe that no single algorithm generally outperforms all competitors, but identify a group of methods including the threshold selection-based Median Sweep and TSMax methods, the DyS framework, and Friedman's method that performs best in the binary setting. For the multiclass setting, we observe that a different group of algorithms yields good performance, including the Generaliz
    

