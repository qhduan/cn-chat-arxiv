# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning](https://arxiv.org/abs/2402.15751) | 提出了一种稀疏MeZO方法，通过仅对精心选择的参数子集应用零阶优化，实现了在零阶LLM微调中减少参数以获得更好性能的目标 |
| [^2] | [TKN: Transformer-based Keypoint Prediction Network For Real-time Video Prediction.](http://arxiv.org/abs/2303.09807) | TKN是一种基于Transformer的实时视频预测解决方案，通过受限信息提取和并行预测方案来提升预测过程的速度，具有更高的精度和更低的计算成本。 |

# 详细

[^1]: 稀疏MeZO：在零阶LLM微调中减少参数以获得更好性能

    Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning

    [https://arxiv.org/abs/2402.15751](https://arxiv.org/abs/2402.15751)

    提出了一种稀疏MeZO方法，通过仅对精心选择的参数子集应用零阶优化，实现了在零阶LLM微调中减少参数以获得更好性能的目标

    

    在针对特定任务进行大型语言模型（LLMs）微调通常会产生令人印象深刻的结果，但由于基于梯度的训练中的反向传播而导致内存效率低下。最近提出的高效利用存储器的零阶（MeZO）优化器旨在解决这个问题，在训练过程中只需要前向传递，使其更符合内存友好性。然而，零阶优化中梯度估计的质量往往取决于数据的维数，这可能解释了为什么与各种任务中的标准微调相比，MeZO仍然表现出显著的性能下降。受到参数高效微调（PEFT）成功的启发，本文介绍了稀疏MeZO，这是一种新颖的内存高效的零阶优化方法，仅将ZO应用于精心选择的参数子集。我们提出了一种简单而有效的参数选择方案，获得了显著的性能提升。

    arXiv:2402.15751v1 Announce Type: cross  Abstract: While fine-tuning large language models (LLMs) for specific tasks often yields impressive results, it comes at the cost of memory inefficiency due to back-propagation in gradient-based training. Memory-efficient Zeroth-order (MeZO) optimizers, recently proposed to address this issue, only require forward passes during training, making them more memory-friendly. However, the quality of gradient estimates in zeroth order optimization often depends on the data dimensionality, potentially explaining why MeZO still exhibits significant performance drops compared to standard fine-tuning across various tasks. Inspired by the success of Parameter-Efficient Fine-Tuning (PEFT), this paper introduces Sparse MeZO, a novel memory-efficient zeroth-order optimization approach that applies ZO only to a carefully chosen subset of parameters. We propose a simple yet effective parameter selection scheme that yields significant performance gains with Spar
    
[^2]: TKN：基于Transformer的实时视频关键点预测网络

    TKN: Transformer-based Keypoint Prediction Network For Real-time Video Prediction. (arXiv:2303.09807v1 [cs.CV])

    [http://arxiv.org/abs/2303.09807](http://arxiv.org/abs/2303.09807)

    TKN是一种基于Transformer的实时视频预测解决方案，通过受限信息提取和并行预测方案来提升预测过程的速度，具有更高的精度和更低的计算成本。

    

    视频预测是一项具有广泛用途的复杂时间序列预测任务。然而，传统方法过于强调准确性，忽视了由于过于复杂的模型结构而导致的较慢的预测速度以及过多的冗余信息学习和GPU内存消耗。因此，我们提出了TKN，一种基于Transformer的关键点预测神经网络，通过受限信息提取和并行预测方案来提升预测过程的速度。TKN是我们目前所知的第一个实时视频预测解决方案，同时显著降低计算成本并保持其他性能。在KTH和Human Action 3D数据集上的大量实验表明，TKN在预测准确性和速度方面均优于现有的基准线。

    Video prediction is a complex time-series forecasting task with great potential in many use cases. However, conventional methods overemphasize accuracy while ignoring the slow prediction speed caused by complicated model structures that learn too much redundant information with excessive GPU memory consumption. Furthermore, conventional methods mostly predict frames sequentially (frame-by-frame) and thus are hard to accelerate. Consequently, valuable use cases such as real-time danger prediction and warning cannot achieve fast enough inference speed to be applicable in reality. Therefore, we propose a transformer-based keypoint prediction neural network (TKN), an unsupervised learning method that boost the prediction process via constrained information extraction and parallel prediction scheme. TKN is the first real-time video prediction solution to our best knowledge, while significantly reducing computation costs and maintaining other performance. Extensive experiments on KTH and Hum
    

