# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal signal propagation in ResNets through residual scaling.](http://arxiv.org/abs/2305.07715) | 本文为ResNets导出系统的有限尺寸理论，指出对于深层网络架构，缩放参数是优化信号传播和确保有效利用网络深度方面的关键。 |

# 详细

[^1]: 通过残差缩放实现ResNets的信号最优传递

    Optimal signal propagation in ResNets through residual scaling. (arXiv:2305.07715v1 [cond-mat.dis-nn])

    [http://arxiv.org/abs/2305.07715](http://arxiv.org/abs/2305.07715)

    本文为ResNets导出系统的有限尺寸理论，指出对于深层网络架构，缩放参数是优化信号传播和确保有效利用网络深度方面的关键。

    

    Residual网络（ResNets）在大深度上比前馈神经网络具有更好的训练能力和性能。引入跳过连接可以促进信号向更深层的传递。此外，先前的研究发现为残差分支添加缩放参数可以进一步提高泛化性能。尽管他们经验性地确定了这种缩放参数特别有利的取值范围，但其相关的性能提升及其在网络超参数上的普适性仍需要进一步理解。对于前馈神经网络（FFNets），有限尺寸理论在信号传播和超参数调节方面获得了重要洞见。我们在这里为ResNets导出了一个系统的有限尺寸理论，以研究信号传播及其对残差分支缩放的依赖性。我们导出响应函数的分析表达式，这是衡量网络对输入敏感性的一种指标，并表明对于深层网络架构，缩放参数在优化信号传播和确保有效利用网络深度方面发挥着至关重要的作用。

    Residual networks (ResNets) have significantly better trainability and thus performance than feed-forward networks at large depth. Introducing skip connections facilitates signal propagation to deeper layers. In addition, previous works found that adding a scaling parameter for the residual branch further improves generalization performance. While they empirically identified a particularly beneficial range of values for this scaling parameter, the associated performance improvement and its universality across network hyperparameters yet need to be understood. For feed-forward networks (FFNets), finite-size theories have led to important insights with regard to signal propagation and hyperparameter tuning. We here derive a systematic finite-size theory for ResNets to study signal propagation and its dependence on the scaling for the residual branch. We derive analytical expressions for the response function, a measure for the network's sensitivity to inputs, and show that for deep netwo
    

