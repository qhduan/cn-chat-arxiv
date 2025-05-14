# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Optimal Classification Trees Robust to Distribution Shifts.](http://arxiv.org/abs/2310.17772) | 本研究提出了一种学习对分布变化具有鲁棒性的最优分类树的方法，通过混合整数鲁棒优化技术将该问题转化为单阶段混合整数鲁棒优化问题，并设计了基于约束生成的解决过程。 |
| [^2] | [A stochastic optimization approach to train non-linear neural networks with regularization of higher-order total variation.](http://arxiv.org/abs/2308.02293) | 通过引入高阶总变差正则化的随机优化算法，可以高效地训练非线性神经网络，避免过拟合问题。 |

# 详细

[^1]: 学习对分布变化具有鲁棒性的最优分类树

    Learning Optimal Classification Trees Robust to Distribution Shifts. (arXiv:2310.17772v1 [cs.LG])

    [http://arxiv.org/abs/2310.17772](http://arxiv.org/abs/2310.17772)

    本研究提出了一种学习对分布变化具有鲁棒性的最优分类树的方法，通过混合整数鲁棒优化技术将该问题转化为单阶段混合整数鲁棒优化问题，并设计了基于约束生成的解决过程。

    

    我们考虑学习对训练和测试/部署数据之间的分布变化具有鲁棒性的分类树的问题。这个问题经常在高风险环境中出现，例如公共卫生和社会工作，其中数据通常是通过自我报告的调查收集的，这些调查对问题的表述方式、调查进行的时间和地点、以及受访者与调查员分享信息的舒适程度非常敏感。我们提出了一种基于混合整数鲁棒优化技术的学习最优鲁棒分类树的方法。特别地，我们证明学习最优鲁棒树的问题可以等价地表达为一个具有高度非线性和不连续目标的单阶段混合整数鲁棒优化问题。我们将这个问题等价地重新表述为一个两阶段线性鲁棒优化问题，为此我们设计了一个基于约束生成的定制解决过程。

    We consider the problem of learning classification trees that are robust to distribution shifts between training and testing/deployment data. This problem arises frequently in high stakes settings such as public health and social work where data is often collected using self-reported surveys which are highly sensitive to e.g., the framing of the questions, the time when and place where the survey is conducted, and the level of comfort the interviewee has in sharing information with the interviewer. We propose a method for learning optimal robust classification trees based on mixed-integer robust optimization technology. In particular, we demonstrate that the problem of learning an optimal robust tree can be cast as a single-stage mixed-integer robust optimization problem with a highly nonlinear and discontinuous objective. We reformulate this problem equivalently as a two-stage linear robust optimization problem for which we devise a tailored solution procedure based on constraint gene
    
[^2]: 用正则化高阶总变差的随机优化方法训练非线性神经网络

    A stochastic optimization approach to train non-linear neural networks with regularization of higher-order total variation. (arXiv:2308.02293v1 [stat.ME])

    [http://arxiv.org/abs/2308.02293](http://arxiv.org/abs/2308.02293)

    通过引入高阶总变差正则化的随机优化算法，可以高效地训练非线性神经网络，避免过拟合问题。

    

    尽管包括深度神经网络在内的高度表达的参数模型可以更好地建模复杂概念，但训练这种高度非线性模型已知会导致严重的过拟合风险。针对这个问题，本研究考虑了一种k阶总变差（k-TV）正则化，它被定义为要训练的参数模型的k阶导数的平方积分，通过惩罚k-TV来产生一个更平滑的函数，从而避免过拟合。尽管将k-TV项应用于一般的参数模型由于积分而导致计算复杂，本研究提供了一种随机优化算法，可以高效地训练带有k-TV正则化的一般模型，而无需进行显式的数值积分。这种方法可以应用于结构任意的深度神经网络的训练，因为它只需要进行简单的随机梯度优化即可实现。

    While highly expressive parametric models including deep neural networks have an advantage to model complicated concepts, training such highly non-linear models is known to yield a high risk of notorious overfitting. To address this issue, this study considers a $k$th order total variation ($k$-TV) regularization, which is defined as the squared integral of the $k$th order derivative of the parametric models to be trained; penalizing the $k$-TV is expected to yield a smoother function, which is expected to avoid overfitting. While the $k$-TV terms applied to general parametric models are computationally intractable due to the integration, this study provides a stochastic optimization algorithm, that can efficiently train general models with the $k$-TV regularization without conducting explicit numerical integration. The proposed approach can be applied to the training of even deep neural networks whose structure is arbitrary, as it can be implemented by only a simple stochastic gradien
    

