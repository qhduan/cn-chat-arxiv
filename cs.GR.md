# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Probabilistic Point Cloud Modeling via Self-Organizing Gaussian Mixture Models.](http://arxiv.org/abs/2302.00047) | 本文提出了一种基于自组织高斯混合模型的概率点云建模方法，可以根据场景复杂度自动调整模型复杂度，相比现有技术具有更好的泛化性能。 |

# 详细

[^1]: 基于自组织高斯混合模型的概率点云建模

    Probabilistic Point Cloud Modeling via Self-Organizing Gaussian Mixture Models. (arXiv:2302.00047v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.00047](http://arxiv.org/abs/2302.00047)

    本文提出了一种基于自组织高斯混合模型的概率点云建模方法，可以根据场景复杂度自动调整模型复杂度，相比现有技术具有更好的泛化性能。

    This paper proposes a probabilistic point cloud modeling method based on self-organizing Gaussian mixture models, which can automatically adjust the model complexity according to the scene complexity, and has better generalization performance compared to existing techniques.

    本文提出了一种连续的概率建模方法，用于使用有限高斯混合模型（GMM）对空间点云数据进行建模，其中组件的数量基于场景复杂性进行调整。我们利用信息论学习中的自组织原理，根据传感器数据中的相关信息自动调整GMM模型的复杂度。该方法在具有不同场景复杂度的实际数据上与现有的点云建模技术进行了评估。

    This letter presents a continuous probabilistic modeling methodology for spatial point cloud data using finite Gaussian Mixture Models (GMMs) where the number of components are adapted based on the scene complexity. Few hierarchical and adaptive methods have been proposed to address the challenge of balancing model fidelity with size. Instead, state-of-the-art mapping approaches require tuning parameters for specific use cases, but do not generalize across diverse environments. To address this gap, we utilize a self-organizing principle from information-theoretic learning to automatically adapt the complexity of the GMM model based on the relevant information in the sensor data. The approach is evaluated against existing point cloud modeling techniques on real-world data with varying degrees of scene complexity.
    

