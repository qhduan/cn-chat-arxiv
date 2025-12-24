# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier](https://arxiv.org/abs/2212.04382) | 本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。 |
| [^2] | [Boosted Control Functions.](http://arxiv.org/abs/2310.05805) | 本研究通过建立同时方程模型和控制函数与分布概括的新连接，解决了在存在未观察到的混淆情况下，针对不同训练和测试分布的预测问题。 |

# 详细

[^1]: 分类器边界的结构：朴素贝叶斯分类器的案例研究

    Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier

    [https://arxiv.org/abs/2212.04382](https://arxiv.org/abs/2212.04382)

    本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。

    

    无论基于模型、训练数据还是二者组合，分类器将（可能复杂的）输入数据归入相对较少的输出类别之一。本文研究在输入空间为图的情况下，边界的结构——那些被分类为不同类别的邻近点——的特性。我们的科学背景是基于模型的朴素贝叶斯分类器，用于处理由下一代测序仪生成的DNA读数。我们展示了边界既是巨大的，又具有复杂的结构。我们创建了一种新的不确定性度量，称为邻居相似度，它将一个点的结果与其邻居的结果分布进行比较。这个度量不仅追踪了贝叶斯分类器的两个固有不确定性度量，还可以在没有固有不确定性度量的分类器上实现，但需要计算成本。

    Whether based on models, training data or a combination, classifiers place (possibly complex) input data into one of a relatively small number of output categories. In this paper, we study the structure of the boundary--those points for which a neighbor is classified differently--in the context of an input space that is a graph, so that there is a concept of neighboring inputs, The scientific setting is a model-based naive Bayes classifier for DNA reads produced by Next Generation Sequencers. We show that the boundary is both large and complicated in structure. We create a new measure of uncertainty, called Neighbor Similarity, that compares the result for a point to the distribution of results for its neighbors. This measure not only tracks two inherent uncertainty measures for the Bayes classifier, but also can be implemented, at a computational cost, for classifiers without inherent measures of uncertainty.
    
[^2]: 提升控制函数

    Boosted Control Functions. (arXiv:2310.05805v1 [stat.ML])

    [http://arxiv.org/abs/2310.05805](http://arxiv.org/abs/2310.05805)

    本研究通过建立同时方程模型和控制函数与分布概括的新连接，解决了在存在未观察到的混淆情况下，针对不同训练和测试分布的预测问题。

    

    现代机器学习方法和大规模数据的可用性为从大量的协变量中准确预测目标数量打开了大门。然而，现有的预测方法在训练和测试数据不同的情况下表现不佳，尤其是在存在隐藏混淆的情况下。虽然对因果效应估计（例如仪器变量）已经对隐藏混淆进行了深入研究，但对于预测任务来说并非如此。本研究旨在填补这一空白，解决在存在未观察到的混淆的情况下，针对不同训练和测试分布的预测问题。具体而言，我们在机器学习的分布概括领域，以及计量经济学中的同时方程模型和控制函数之间建立了一种新的联系。我们的贡献的核心是描述在一组分布转变下的数据生成过程的分布概括同时方程模型（SIMDGs）。

    Modern machine learning methods and the availability of large-scale data opened the door to accurately predict target quantities from large sets of covariates. However, existing prediction methods can perform poorly when the training and testing data are different, especially in the presence of hidden confounding. While hidden confounding is well studied for causal effect estimation (e.g., instrumental variables), this is not the case for prediction tasks. This work aims to bridge this gap by addressing predictions under different training and testing distributions in the presence of unobserved confounding. In particular, we establish a novel connection between the field of distribution generalization from machine learning, and simultaneous equation models and control function from econometrics. Central to our contribution are simultaneous equation models for distribution generalization (SIMDGs) which describe the data-generating process under a set of distributional shifts. Within thi
    

