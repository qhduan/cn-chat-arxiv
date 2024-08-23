# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Spanning Multi-Asset Payoffs With ReLUs](https://arxiv.org/abs/2403.14231) | 提出了一种用ReLU解决跨度多资产回报问题的方法，通过前馈神经网络提供了更好的离散跨度避险结果。 |
| [^2] | [Neural networks for insurance pricing with frequency and severity data: a benchmark study from data preprocessing to technical tariff.](http://arxiv.org/abs/2310.12671) | 本研究通过深度学习结构的神经网络对频率-严重性保险定价进行了基准研究，比较了不同模型的性能，并提出了一种联合精算神经网络(CANN)的方法。 |

# 详细

[^1]: 用ReLU跨度多资产回报

    Spanning Multi-Asset Payoffs With ReLUs

    [https://arxiv.org/abs/2403.14231](https://arxiv.org/abs/2403.14231)

    提出了一种用ReLU解决跨度多资产回报问题的方法，通过前馈神经网络提供了更好的离散跨度避险结果。

    

    我们提出了利用香草篮子期权的分布式形式来解决多资产回报的跨度问题。我们发现，只有当回报函数为偶函数且绝对齐次函数时，此问题才有唯一解，并且我们建立了一个基于傅立叶的公式来计算解决方案。金融回报通常是分段线性的，导致可能可以明确推导出解决方案，但在数值上可能难以利用。相比于基于单资产香草避险的行业偏爱方法，单隐藏层前馈神经网络为离散跨度提供了一种自然而高效的数值替代方法。我们测试了这种方法用于一些典型回报，并发现与单资产香草避险的产业偏好方法相比，利用香草篮子期权获得了更好的避险结果。

    arXiv:2403.14231v1 Announce Type: new  Abstract: We propose a distributional formulation of the spanning problem of a multi-asset payoff by vanilla basket options. This problem is shown to have a unique solution if and only if the payoff function is even and absolutely homogeneous, and we establish a Fourier-based formula to calculate the solution. Financial payoffs are typically piecewise linear, resulting in a solution that may be derived explicitly, yet may also be hard to numerically exploit. One-hidden-layer feedforward neural networks instead provide a natural and efficient numerical alternative for discrete spanning. We test this approach for a selection of archetypal payoffs and obtain better hedging results with vanilla basket options compared to industry-favored approaches based on single-asset vanilla hedges.
    
[^2]: 利用频率和严重性数据进行保险定价的神经网络：从数据预处理到技术定价的基准研究

    Neural networks for insurance pricing with frequency and severity data: a benchmark study from data preprocessing to technical tariff. (arXiv:2310.12671v1 [cs.LG])

    [http://arxiv.org/abs/2310.12671](http://arxiv.org/abs/2310.12671)

    本研究通过深度学习结构的神经网络对频率-严重性保险定价进行了基准研究，比较了不同模型的性能，并提出了一种联合精算神经网络(CANN)的方法。

    

    保险公司通常使用广义线性模型来建模索赔的频率和严重性数据。由于其在其他领域的成功，机器学习技术在精算工具箱中越来越受欢迎。本文通过深度学习结构为频率-严重性保险定价与机器学习相关的文献做出了贡献。我们在四个保险数据集上进行了基准研究，这些数据集包含有多种类型的输入特征和频率-严重性目标。我们详细比较了广义线性模型在分箱输入数据、梯度提升树模型、前馈神经网络（FFNN）和联合精算神经网络（CANN）上的性能。我们的CANN将通过GLM和GBM分别建立的基线预测与神经网络校正相结合。我们解释了数据预处理步骤，特别关注通常存在于表格保险数据集中的多种类型的输入特征，比如邮编和数字编码。

    Insurers usually turn to generalized linear models for modelling claim frequency and severity data. Due to their success in other fields, machine learning techniques are gaining popularity within the actuarial toolbox. Our paper contributes to the literature on frequency-severity insurance pricing with machine learning via deep learning structures. We present a benchmark study on four insurance data sets with frequency and severity targets in the presence of multiple types of input features. We compare in detail the performance of: a generalized linear model on binned input data, a gradient-boosted tree model, a feed-forward neural network (FFNN), and the combined actuarial neural network (CANN). Our CANNs combine a baseline prediction established with a GLM and GBM, respectively, with a neural network correction. We explain the data preprocessing steps with specific focus on the multiple types of input features typically present in tabular insurance data sets, such as postal codes, nu
    

