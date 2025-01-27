# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Regularized Encoder Training for Extreme Classification](https://arxiv.org/abs/2402.18434) | 本文提出了一种图正则化编码器训练方法用于极端分类，在实践中发现使用图数据来规范编码器训练比实施 GCN 效果更好。 |
| [^2] | [A Comparative Evaluation of Quantification Methods.](http://arxiv.org/abs/2103.03223) | 本研究通过对24种不同量化方法在超过40个数据集上进行全面实证比较，填补了量化方法比较研究的空白。我们发现在二分类设置中，基于阈值选择的Median Sweep和TSMax方法、DyS框架和弗里德曼的方法表现最佳；而在多分类设置中，Generaliz方法表现良好。 |

# 详细

[^1]: 图正则化编码器训练用于极端分类

    Graph Regularized Encoder Training for Extreme Classification

    [https://arxiv.org/abs/2402.18434](https://arxiv.org/abs/2402.18434)

    本文提出了一种图正则化编码器训练方法用于极端分类，在实践中发现使用图数据来规范编码器训练比实施 GCN 效果更好。

    

    arXiv:2402.18434v1 通告类型: 新的 摘要: 深度极端分类（XC）旨在训练编码器架构和配套的分类器架构，以从一个非常庞大的标签集合中为数据点打上最相关的子标签集合。在排名、推荐和标记中常见的XC应用中，通常会遇到训练数据极少的尾标签。图卷积网络（GCN）提供了一个方便但计算代价高昂的方法，可利用任务元数据并增强模型在这些设置中的准确性。本文正式确定了在若干用例中，通过用非GCN架构替换GCNs，完全可以避免GCNs的巨大计算成本。本文指出，在这些设置中，使用图数据来规范编码器训练比实施GCN更加有效。基于这些见解，提出了一种替代范式RAMEN，用于利用XC设置中的图元数据。

    arXiv:2402.18434v1 Announce Type: new  Abstract: Deep extreme classification (XC) aims to train an encoder architecture and an accompanying classifier architecture to tag a data point with the most relevant subset of labels from a very large universe of labels. XC applications in ranking, recommendation and tagging routinely encounter tail labels for which the amount of training data is exceedingly small. Graph convolutional networks (GCN) present a convenient but computationally expensive way to leverage task metadata and enhance model accuracies in these settings. This paper formally establishes that in several use cases, the steep computational cost of GCNs is entirely avoidable by replacing GCNs with non-GCN architectures. The paper notices that in these settings, it is much more effective to use graph data to regularize encoder training than to implement a GCN. Based on these insights, an alternative paradigm RAMEN is presented to utilize graph metadata in XC settings that offers 
    
[^2]: 量化方法的比较评估

    A Comparative Evaluation of Quantification Methods. (arXiv:2103.03223v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2103.03223](http://arxiv.org/abs/2103.03223)

    本研究通过对24种不同量化方法在超过40个数据集上进行全面实证比较，填补了量化方法比较研究的空白。我们发现在二分类设置中，基于阈值选择的Median Sweep和TSMax方法、DyS框架和弗里德曼的方法表现最佳；而在多分类设置中，Generaliz方法表现良好。

    

    量化是指在数据集中预测类别分布的问题。它也代表着一个在监督式机器学习中不断发展的研究领域，近年来提出了大量不同的算法。然而，目前还没有一份全面的实证比较量化方法的研究，以支持算法选择。在本研究中，我们通过对超过40个数据集进行了24种不同量化方法的彻底实证性性能比较，包括二分类和多分类量化设置，填补了这一研究空白。我们观察到没有单一算法能够在所有竞争对手中始终表现最佳，但我们确定了一组在二分类设置中表现最佳的方法，包括基于阈值选择的Median Sweep和TSMax方法、DyS框架和弗里德曼的方法。对于多分类设置，我们观察到另一组算法表现良好，包括Generaliz方法。

    Quantification represents the problem of predicting class distributions in a dataset. It also represents a growing research field in supervised machine learning, for which a large variety of different algorithms has been proposed in recent years. However, a comprehensive empirical comparison of quantification methods that supports algorithm selection is not available yet. In this work, we close this research gap by conducting a thorough empirical performance comparison of 24 different quantification methods on overall more than 40 data sets, considering binary as well as multiclass quantification settings. We observe that no single algorithm generally outperforms all competitors, but identify a group of methods including the threshold selection-based Median Sweep and TSMax methods, the DyS framework, and Friedman's method that performs best in the binary setting. For the multiclass setting, we observe that a different group of algorithms yields good performance, including the Generaliz
    

