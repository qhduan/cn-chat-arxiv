# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Packed-Ensembles for Efficient Uncertainty Estimation.](http://arxiv.org/abs/2210.09184) | Packed-Ensembles是一种能够在标准神经网络内运行的轻量级结构化集合，它通过精心调节编码空间的维度来设计。该方法在不损失效果的情况下提高了训练和推理速度。 |

# 详细

[^1]: 紧凑集成用于高效的不确定性估计

    Packed-Ensembles for Efficient Uncertainty Estimation. (arXiv:2210.09184v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.09184](http://arxiv.org/abs/2210.09184)

    Packed-Ensembles是一种能够在标准神经网络内运行的轻量级结构化集合，它通过精心调节编码空间的维度来设计。该方法在不损失效果的情况下提高了训练和推理速度。

    

    深度集成是实现关键指标（如准确性、校准、不确定性估计和超出分布检测）卓越性能的突出方法。但是，现实系统的硬件限制限制了更小的集合和较低容量的网络，严重损害了它们的性能和属性。我们引入了一种称为Packed-Ensembles（PE）的策略，通过精心调节其编码空间的维度来设计和训练轻量级结构化集合。我们利用组卷积将集合并行化为单个共享骨干，并进行前向传递以提高训练和推理速度。PE旨在在标准神经网络的内存限制内运行。

    Deep Ensembles (DE) are a prominent approach for achieving excellent performance on key metrics such as accuracy, calibration, uncertainty estimation, and out-of-distribution detection. However, hardware limitations of real-world systems constrain to smaller ensembles and lower-capacity networks, significantly deteriorating their performance and properties. We introduce Packed-Ensembles (PE), a strategy to design and train lightweight structured ensembles by carefully modulating the dimension of their encoding space. We leverage grouped convolutions to parallelize the ensemble into a single shared backbone and forward pass to improve training and inference speeds. PE is designed to operate within the memory limits of a standard neural network. Our extensive research indicates that PE accurately preserves the properties of DE, such as diversity, and performs equally well in terms of accuracy, calibration, out-of-distribution detection, and robustness to distribution shift. We make our c
    

