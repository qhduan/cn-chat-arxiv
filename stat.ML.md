# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Global-QSGD: Practical Floatless Quantization for Distributed Learning with Theoretical Guarantees.](http://arxiv.org/abs/2305.18627) | Global-QSGD是一种新颖的全局缩放量化机制，可以提高分布式学习的效率，并且不需要昂贵的误差反馈，并提供了高达$O(\ sqrt{n})$的额外压缩比。 |

# 详细

[^1]: 全局缩放量化：具有理论保证的分布式学习实用的无浮点量化

    Global-QSGD: Practical Floatless Quantization for Distributed Learning with Theoretical Guarantees. (arXiv:2305.18627v1 [cs.LG])

    [http://arxiv.org/abs/2305.18627](http://arxiv.org/abs/2305.18627)

    Global-QSGD是一种新颖的全局缩放量化机制，可以提高分布式学习的效率，并且不需要昂贵的误差反馈，并提供了高达$O(\ sqrt{n})$的额外压缩比。

    

    高效的分布式训练是推动深度学习近期进展的主要驱动力。然而，通信常常是系统的主要瓶颈并具有高昂的代价。因此，需要设计高效的通信机制，既能在经验上提高吞吐量，又能提供理论保证。在这项工作中，我们介绍了全局-QSGD，一种新颖的量化运算符，通过全局缩放设计来加速基于分布式学习。我们证明Global-QSGD是第一个理论上严格的Allreduce兼容压缩机制，通过在压缩误差和通信节省之间取得平衡来实现可证明的加速。重要的是，由于其固有的无偏性，Global-QSGD不依赖昂贵的误差反馈，并且相对于流行的QSGD量化能提供高达$O(\sqrt{n})$ 的额外压缩比（其中$n$表示工作者的数量）。为了获得理论保证，我们采用了信息论和凸分析技术。

    Efficient distributed training is a principal driver of recent advances in deep learning. However, communication often proves costly and becomes the primary bottleneck in these systems. As a result, there is a demand for the design of efficient communication mechanisms that can empirically boost throughput while providing theoretical guarantees. In this work, we introduce Global-QSGD, a novel family of quantization operators, engineered to accelerate distributed training based on global scaling. We demonstrate that Global-QSGD is the first theoretically rigorous Allreduce-compatible compression mechanism that achieves a provable speed-up by striking a balance between compression error and communication savings. Importantly, Global-QSGD does not rely on costly error feedback due to its inherent unbiasedness and offers up to $O(\sqrt{n})$ additional compression ratio compared to the popular QSGD quantization ($n$ represents the number of workers). To obtain theoretical guarantees, we gen
    

