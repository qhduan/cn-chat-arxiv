# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [REAL: Representation Enhanced Analytic Learning for Exemplar-free Class-incremental Learning](https://arxiv.org/abs/2403.13522) | 本文提出了REAL方法，通过构建双流基础预训练和表示增强蒸馏过程来增强提取器的表示，从而解决了无范例类增量学习中的遗忘问题。 |
| [^2] | [Variational Continual Test-Time Adaptation](https://arxiv.org/abs/2402.08182) | 本文介绍了VCoTTA，一种变分贝叶斯方法用于测量连续测试时适应性中的不确定性。采用变分预热策略将预训练的模型转为贝叶斯神经网络，在测试时通过均值教师更新策略来更新学生模型，结合源模型和教师模型的先验。实验证明该方法在减轻先验偏移方面有效。 |
| [^3] | [SketchOGD: Memory-Efficient Continual Learning.](http://arxiv.org/abs/2305.16424) | SketchOGD提出了一种内存高效的解决灾难性遗忘的方法，通过采用在线草图算法，将模型梯度压缩为固定大小的矩阵，从而改进了现有的算法——正交梯度下降（OGD）。 |
| [^4] | [Optimal Prediction Using Expert Advice and Randomized Littlestone Dimension.](http://arxiv.org/abs/2302.13849) | 本研究证明了在线学习中，在学习一个类别时，最优期望错误边界等于其随机化的Littlestone维度。在不可知的情况下，最优错误边界与最佳函数的错误次数之间存在特定关系。此外，该研究还解决了一个开放问题，并将其应用于预测问题。 |

# 详细

[^1]: REAL：用于无范例类增量学习的表示增强分析学习

    REAL: Representation Enhanced Analytic Learning for Exemplar-free Class-incremental Learning

    [https://arxiv.org/abs/2403.13522](https://arxiv.org/abs/2403.13522)

    本文提出了REAL方法，通过构建双流基础预训练和表示增强蒸馏过程来增强提取器的表示，从而解决了无范例类增量学习中的遗忘问题。

    

    无范例的类增量学习(EFCIL)旨在减轻类增量学习中的灾难性遗忘，而没有可用的历史数据。与存储历史样本的回放式CIL相比，EFCIL在无范例约束下更容易遗忘。在本文中，受最近发展的基于分析学习(AL)的CIL的启发，我们提出了一种用于EFCIL的表示增强分析学习(REAL)。REAL构建了一个双流基础预训练(DS-BPT)和一个表示增强蒸馏(RED)过程，以增强提取器的表示。DS-BPT在监督学习和自监督对比学习(SSCL)两个流中预训练模型，用于基础知识提取。RED过程将监督知识提炼到SSCL预训练骨干部分，促进后续的基于AL的CIL，将CIL转换为递归最小化学习

    arXiv:2403.13522v1 Announce Type: new  Abstract: Exemplar-free class-incremental learning (EFCIL) aims to mitigate catastrophic forgetting in class-incremental learning without available historical data. Compared with its counterpart (replay-based CIL) that stores historical samples, the EFCIL suffers more from forgetting issues under the exemplar-free constraint. In this paper, inspired by the recently developed analytic learning (AL) based CIL, we propose a representation enhanced analytic learning (REAL) for EFCIL. The REAL constructs a dual-stream base pretraining (DS-BPT) and a representation enhancing distillation (RED) process to enhance the representation of the extractor. The DS-BPT pretrains model in streams of both supervised learning and self-supervised contrastive learning (SSCL) for base knowledge extraction. The RED process distills the supervised knowledge to the SSCL pretrained backbone and facilitates a subsequent AL-basd CIL that converts the CIL to a recursive least
    
[^2]: 变分连续测试时适应性

    Variational Continual Test-Time Adaptation

    [https://arxiv.org/abs/2402.08182](https://arxiv.org/abs/2402.08182)

    本文介绍了VCoTTA，一种变分贝叶斯方法用于测量连续测试时适应性中的不确定性。采用变分预热策略将预训练的模型转为贝叶斯神经网络，在测试时通过均值教师更新策略来更新学生模型，结合源模型和教师模型的先验。实验证明该方法在减轻先验偏移方面有效。

    

    先验偏移在只使用无标签测试数据的连续测试时适应性（CTTA）方法中至关重要，因为它可能导致严重的误差传播。在本文中，我们介绍了VCoTTA，一种用于测量CTTA中不确定性的变分贝叶斯方法。在源阶段，我们通过变分预热策略将预训练的确定性模型转化为贝叶斯神经网络（BNN），将不确定性注入模型中。在测试时，我们采用变分推断的均值教师更新策略，将学生模型和指数移动平均法用于教师模型。我们的新方法通过结合源模型和教师模型的先验来更新学生模型。证据下界被制定为学生模型和教师模型之间的交叉熵，以及先验混合的Kullback-Leibler（KL）散度。在三个数据集上的实验结果表明该方法在减轻在CTTA中的先验偏移方面的有效性。

    The prior drift is crucial in Continual Test-Time Adaptation (CTTA) methods that only use unlabeled test data, as it can cause significant error propagation. In this paper, we introduce VCoTTA, a variational Bayesian approach to measure uncertainties in CTTA. At the source stage, we transform a pre-trained deterministic model into a Bayesian Neural Network (BNN) via a variational warm-up strategy, injecting uncertainties into the model. During the testing time, we employ a mean-teacher update strategy using variational inference for the student model and exponential moving average for the teacher model. Our novel approach updates the student model by combining priors from both the source and teacher models. The evidence lower bound is formulated as the cross-entropy between the student and teacher models, along with the Kullback-Leibler (KL) divergence of the prior mixture. Experimental results on three datasets demonstrate the method's effectiveness in mitigating prior drift within th
    
[^3]: SketchOGD：内存高效的持续学习

    SketchOGD: Memory-Efficient Continual Learning. (arXiv:2305.16424v1 [cs.LG])

    [http://arxiv.org/abs/2305.16424](http://arxiv.org/abs/2305.16424)

    SketchOGD提出了一种内存高效的解决灾难性遗忘的方法，通过采用在线草图算法，将模型梯度压缩为固定大小的矩阵，从而改进了现有的算法——正交梯度下降（OGD）。

    

    当机器学习模型在一系列任务上持续训练时，它们容易忘记先前任务上学习到的知识，这种现象称为灾难性遗忘。现有的解决灾难性遗忘的方法往往涉及存储过去任务的信息，这意味着内存使用是确定实用性的主要因素。本文提出了一种内存高效的解决灾难性遗忘的方法，改进了一种已有的算法——正交梯度下降（OGD）。OGD利用先前模型梯度来找到维持先前数据点性能的权重更新。然而，由于存储先前模型梯度的内存成本随算法运行时间增长而增加，因此OGD不适用于任意长时间跨度的连续学习。针对这个问题，本文提出了SketchOGD。SketchOGD采用在线草图算法，将模型梯度压缩为固定大小的矩阵。

    When machine learning models are trained continually on a sequence of tasks, they are liable to forget what they learned on previous tasks -- a phenomenon known as catastrophic forgetting. Proposed solutions to catastrophic forgetting tend to involve storing information about past tasks, meaning that memory usage is a chief consideration in determining their practicality. This paper proposes a memory-efficient solution to catastrophic forgetting, improving upon an established algorithm known as orthogonal gradient descent (OGD). OGD utilizes prior model gradients to find weight updates that preserve performance on prior datapoints. However, since the memory cost of storing prior model gradients grows with the runtime of the algorithm, OGD is ill-suited to continual learning over arbitrarily long time horizons. To address this problem, this paper proposes SketchOGD. SketchOGD employs an online sketching algorithm to compress model gradients as they are encountered into a matrix of a fix
    
[^4]: 使用专家建议和随机化的Littlestone维度进行最优预测

    Optimal Prediction Using Expert Advice and Randomized Littlestone Dimension. (arXiv:2302.13849v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.13849](http://arxiv.org/abs/2302.13849)

    本研究证明了在线学习中，在学习一个类别时，最优期望错误边界等于其随机化的Littlestone维度。在不可知的情况下，最优错误边界与最佳函数的错误次数之间存在特定关系。此外，该研究还解决了一个开放问题，并将其应用于预测问题。

    

    在在线学习中，经典的结果表明使用确定性学习器的最优错误边界可以通过Littlestone维度来实现（Littlestone '88）。我们证明了随机学习器的类似结果：我们证明了在学习一个类别 $\mathcal{H}$时，最优期望错误边界等于其随机化的Littlestone维度，即存在一个由 $\mathcal{H}$ 打碎的树，其平均深度为 $2d$，而 $d$ 是最大的维度。此外，我们进一步研究了在不可知的情况下，最优错误边界与 $\mathcal{H}$ 中最佳函数的错误次数 $k$ 之间的关系。我们证明了具有Littlestone维度 $d$ 的类别学习的最优随机化错误边界是 $k + \Theta (\sqrt{k d} + d )$。这也意味着确定性学习的最优错误边界是 $2k + O (\sqrt{k d} + d )$，从而解决了Auer和Long ['99]研究的一个开放问题。作为我们理论的一个应用，我们重新审视了经典问题的预测

    A classical result in online learning characterizes the optimal mistake bound achievable by deterministic learners using the Littlestone dimension (Littlestone '88). We prove an analogous result for randomized learners: we show that the optimal expected mistake bound in learning a class $\mathcal{H}$ equals its randomized Littlestone dimension, which is the largest $d$ for which there exists a tree shattered by $\mathcal{H}$ whose average depth is $2d$. We further study optimal mistake bounds in the agnostic case, as a function of the number of mistakes made by the best function in $\mathcal{H}$, denoted by $k$. We show that the optimal randomized mistake bound for learning a class with Littlestone dimension $d$ is $k + \Theta (\sqrt{k d} + d )$. This also implies an optimal deterministic mistake bound of $2k + O (\sqrt{k d} + d )$, thus resolving an open question which was studied by Auer and Long ['99].  As an application of our theory, we revisit the classical problem of prediction 
    

