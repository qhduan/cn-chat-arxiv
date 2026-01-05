# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sparse-Input Neural Network using Group Concave Regularization.](http://arxiv.org/abs/2307.00344) | 本文提出了一种使用组凹正则化进行特征选择的稀疏输入神经网络框架，该框架能够在高维环境中选择重要的特征并保持稳定的解。 |

# 详细

[^1]: 使用组凹正则化的稀疏输入神经网络

    Sparse-Input Neural Network using Group Concave Regularization. (arXiv:2307.00344v1 [stat.ML])

    [http://arxiv.org/abs/2307.00344](http://arxiv.org/abs/2307.00344)

    本文提出了一种使用组凹正则化进行特征选择的稀疏输入神经网络框架，该框架能够在高维环境中选择重要的特征并保持稳定的解。

    

    同时进行特征选择和非线性函数估计在高维环境中是具有挑战性的，其中变量的数量超过了建模中可用的样本大小。在本文中，我们研究了神经网络中的特征选择问题。虽然组LASSO已经被用于神经网络的学习中选择变量，但它倾向于选择无关紧要的变量来弥补过度缩减的问题。为了克服这个限制，我们提出了一个稀疏输入神经网络框架，使用组凹正则化进行特征选择，适用于低维和高维设置。主要思想是对每个输入节点的所有出站连接的权重的l2范数应用适当的凹惩罚，从而得到一个只使用原始变量的一个小子集的神经网络。此外，我们基于向后路径优化开发了一个有效的算法来获得稳定的解。

    Simultaneous feature selection and non-linear function estimation are challenging, especially in high-dimensional settings where the number of variables exceeds the available sample size in modeling. In this article, we investigate the problem of feature selection in neural networks. Although the group LASSO has been utilized to select variables for learning with neural networks, it tends to select unimportant variables into the model to compensate for its over-shrinkage. To overcome this limitation, we propose a framework of sparse-input neural networks using group concave regularization for feature selection in both low-dimensional and high-dimensional settings. The main idea is to apply a proper concave penalty to the $l_2$ norm of weights from all outgoing connections of each input node, and thus obtain a neural net that only uses a small subset of the original variables. In addition, we develop an effective algorithm based on backward path-wise optimization to yield stable solutio
    

