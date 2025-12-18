# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Variational Continual Test-Time Adaptation](https://arxiv.org/abs/2402.08182) | 本文介绍了VCoTTA，一种变分贝叶斯方法用于测量连续测试时适应性中的不确定性。采用变分预热策略将预训练的模型转为贝叶斯神经网络，在测试时通过均值教师更新策略来更新学生模型，结合源模型和教师模型的先验。实验证明该方法在减轻先验偏移方面有效。 |
| [^2] | [SketchOGD: Memory-Efficient Continual Learning.](http://arxiv.org/abs/2305.16424) | SketchOGD提出了一种内存高效的解决灾难性遗忘的方法，通过采用在线草图算法，将模型梯度压缩为固定大小的矩阵，从而改进了现有的算法——正交梯度下降（OGD）。 |

# 详细

[^1]: 变分连续测试时适应性

    Variational Continual Test-Time Adaptation

    [https://arxiv.org/abs/2402.08182](https://arxiv.org/abs/2402.08182)

    本文介绍了VCoTTA，一种变分贝叶斯方法用于测量连续测试时适应性中的不确定性。采用变分预热策略将预训练的模型转为贝叶斯神经网络，在测试时通过均值教师更新策略来更新学生模型，结合源模型和教师模型的先验。实验证明该方法在减轻先验偏移方面有效。

    

    先验偏移在只使用无标签测试数据的连续测试时适应性（CTTA）方法中至关重要，因为它可能导致严重的误差传播。在本文中，我们介绍了VCoTTA，一种用于测量CTTA中不确定性的变分贝叶斯方法。在源阶段，我们通过变分预热策略将预训练的确定性模型转化为贝叶斯神经网络（BNN），将不确定性注入模型中。在测试时，我们采用变分推断的均值教师更新策略，将学生模型和指数移动平均法用于教师模型。我们的新方法通过结合源模型和教师模型的先验来更新学生模型。证据下界被制定为学生模型和教师模型之间的交叉熵，以及先验混合的Kullback-Leibler（KL）散度。在三个数据集上的实验结果表明该方法在减轻在CTTA中的先验偏移方面的有效性。

    The prior drift is crucial in Continual Test-Time Adaptation (CTTA) methods that only use unlabeled test data, as it can cause significant error propagation. In this paper, we introduce VCoTTA, a variational Bayesian approach to measure uncertainties in CTTA. At the source stage, we transform a pre-trained deterministic model into a Bayesian Neural Network (BNN) via a variational warm-up strategy, injecting uncertainties into the model. During the testing time, we employ a mean-teacher update strategy using variational inference for the student model and exponential moving average for the teacher model. Our novel approach updates the student model by combining priors from both the source and teacher models. The evidence lower bound is formulated as the cross-entropy between the student and teacher models, along with the Kullback-Leibler (KL) divergence of the prior mixture. Experimental results on three datasets demonstrate the method's effectiveness in mitigating prior drift within th
    
[^2]: SketchOGD：内存高效的持续学习

    SketchOGD: Memory-Efficient Continual Learning. (arXiv:2305.16424v1 [cs.LG])

    [http://arxiv.org/abs/2305.16424](http://arxiv.org/abs/2305.16424)

    SketchOGD提出了一种内存高效的解决灾难性遗忘的方法，通过采用在线草图算法，将模型梯度压缩为固定大小的矩阵，从而改进了现有的算法——正交梯度下降（OGD）。

    

    当机器学习模型在一系列任务上持续训练时，它们容易忘记先前任务上学习到的知识，这种现象称为灾难性遗忘。现有的解决灾难性遗忘的方法往往涉及存储过去任务的信息，这意味着内存使用是确定实用性的主要因素。本文提出了一种内存高效的解决灾难性遗忘的方法，改进了一种已有的算法——正交梯度下降（OGD）。OGD利用先前模型梯度来找到维持先前数据点性能的权重更新。然而，由于存储先前模型梯度的内存成本随算法运行时间增长而增加，因此OGD不适用于任意长时间跨度的连续学习。针对这个问题，本文提出了SketchOGD。SketchOGD采用在线草图算法，将模型梯度压缩为固定大小的矩阵。

    When machine learning models are trained continually on a sequence of tasks, they are liable to forget what they learned on previous tasks -- a phenomenon known as catastrophic forgetting. Proposed solutions to catastrophic forgetting tend to involve storing information about past tasks, meaning that memory usage is a chief consideration in determining their practicality. This paper proposes a memory-efficient solution to catastrophic forgetting, improving upon an established algorithm known as orthogonal gradient descent (OGD). OGD utilizes prior model gradients to find weight updates that preserve performance on prior datapoints. However, since the memory cost of storing prior model gradients grows with the runtime of the algorithm, OGD is ill-suited to continual learning over arbitrarily long time horizons. To address this problem, this paper proposes SketchOGD. SketchOGD employs an online sketching algorithm to compress model gradients as they are encountered into a matrix of a fix
    

