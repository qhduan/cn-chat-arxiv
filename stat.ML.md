# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tree-based Learning for High-Fidelity Prediction of Chaos](https://arxiv.org/abs/2403.13836) | TreeDOX是一种基于树的方法，不需要超参数调整，使用时间延迟过度嵌入和额外树回归器进行特征降维和预测，并在深度预测混沌系统中表现出state-of-the-art的性能。 |
| [^2] | [Benign Overfitting without Linearity: Neural Network Classifiers Trained by Gradient Descent for Noisy Linear Data.](http://arxiv.org/abs/2202.05928) | 本文研究了使用梯度下降训练的神经网络在泛化时能够很好应对噪声数据的良性过拟合现象。研究表明，在特定条件下，神经网络能够将训练误差降至零并完美地适应带有噪声标签的数据，并同时达到最优的测试误差。 |

# 详细

[^1]: 基于树的学习用于深度预测混沌现象

    Tree-based Learning for High-Fidelity Prediction of Chaos

    [https://arxiv.org/abs/2403.13836](https://arxiv.org/abs/2403.13836)

    TreeDOX是一种基于树的方法，不需要超参数调整，使用时间延迟过度嵌入和额外树回归器进行特征降维和预测，并在深度预测混沌系统中表现出state-of-the-art的性能。

    

    深度预测混沌系统的时间演变是至关重要但具有挑战性的。现有解决方案需要进行超参数调整，这严重阻碍了它们的广泛应用。在这项工作中，我们引入了一种无需超参数调整的基于树的方法：TreeDOX。它使用时间延迟过度嵌入作为显式短期记忆，以及额外树回归器来执行特征降维和预测。我们使用Henon映射，Lorenz和Kuramoto-Sivashinsky系统以及现实世界的Southern Oscillation Index展示了TreeDOX的最先进性能。

    arXiv:2403.13836v1 Announce Type: new  Abstract: Model-free forecasting of the temporal evolution of chaotic systems is crucial but challenging. Existing solutions require hyperparameter tuning, significantly hindering their wider adoption. In this work, we introduce a tree-based approach not requiring hyperparameter tuning: TreeDOX. It uses time delay overembedding as explicit short-term memory and Extra-Trees Regressors to perform feature reduction and forecasting. We demonstrate the state-of-the-art performance of TreeDOX using the Henon map, Lorenz and Kuramoto-Sivashinsky systems, and the real-world Southern Oscillation Index.
    
[^2]: 不需要线性关系的良性过拟合：通过梯度下降训练的神经网络分类器用于噪声线性数据

    Benign Overfitting without Linearity: Neural Network Classifiers Trained by Gradient Descent for Noisy Linear Data. (arXiv:2202.05928v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.05928](http://arxiv.org/abs/2202.05928)

    本文研究了使用梯度下降训练的神经网络在泛化时能够很好应对噪声数据的良性过拟合现象。研究表明，在特定条件下，神经网络能够将训练误差降至零并完美地适应带有噪声标签的数据，并同时达到最优的测试误差。

    

    良性过拟合是指插值模型在存在噪声数据的情况下能够很好地泛化的现象，最早出现在使用梯度下降训练的神经网络模型中。为了更好地理解这一实证观察，我们考虑了两层神经网络在随机初始化后通过梯度下降在逻辑损失函数上进行插值训练的泛化误差。我们假设数据来自于明显分离的类条件对数凹分布，并允许训练标签中的一定比例被对手篡改。我们证明在这种情况下，神经网络表现出良性过拟合的特点：它们可以被驱动到零训练误差，完美地拟合任何有噪声的训练标签，并同时达到极小化最大化最优测试误差。与之前关于良性过拟合需要线性或基于核的预测器的工作相比，我们的分析在模型和学习动态都是基本非线性的情况下成立。

    Benign overfitting, the phenomenon where interpolating models generalize well in the presence of noisy data, was first observed in neural network models trained with gradient descent. To better understand this empirical observation, we consider the generalization error of two-layer neural networks trained to interpolation by gradient descent on the logistic loss following random initialization. We assume the data comes from well-separated class-conditional log-concave distributions and allow for a constant fraction of the training labels to be corrupted by an adversary. We show that in this setting, neural networks exhibit benign overfitting: they can be driven to zero training error, perfectly fitting any noisy training labels, and simultaneously achieve minimax optimal test error. In contrast to previous work on benign overfitting that require linear or kernel-based predictors, our analysis holds in a setting where both the model and learning dynamics are fundamentally nonlinear.
    

