# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benchmarking Neural Network Training Algorithms.](http://arxiv.org/abs/2306.07179) | 本文解决了神经网络训练算法基准测试中存在的三个挑战，提出了新的基准测试套件，以促进训练算法效率的进一步提高。 |

# 详细

[^1]: 神经网络训练算法基准测试

    Benchmarking Neural Network Training Algorithms. (arXiv:2306.07179v1 [cs.LG])

    [http://arxiv.org/abs/2306.07179](http://arxiv.org/abs/2306.07179)

    本文解决了神经网络训练算法基准测试中存在的三个挑战，提出了新的基准测试套件，以促进训练算法效率的进一步提高。

    

    训练算法是每个深度学习流程的重要组成部分。提高训练算法的效率可以节省时间、计算资源，并带来更好、更准确的模型。然而，我们目前还无法可靠地确定最先进的训练算法。本文通过具体实验，证明了加速训练的真正进展需要解决三个基本挑战：如何确定训练何时结束并精确测量训练时间，如何处理测量对确切工作负载详情的敏感性，并公平比较需要超参数调整的算法。为了增加对训练算法效率的了解，我们提出并设计了一些新的基准测试套件。

    Training algorithms, broadly construed, are an essential part of every deep learning pipeline. Training algorithm improvements that speed up training across a wide variety of workloads (e.g., better update rules, tuning protocols, learning rate schedules, or data selection schemes) could save time, save computational resources, and lead to better, more accurate, models. Unfortunately, as a community, we are currently unable to reliably identify training algorithm improvements, or even determine the state-of-the-art training algorithm. In this work, using concrete experiments, we argue that real progress in speeding up training requires new benchmarks that resolve three basic challenges faced by empirical comparisons of training algorithms: (1) how to decide when training is complete and precisely measure training time, (2) how to handle the sensitivity of measurements to exact workload details, and (3) how to fairly compare algorithms that require hyperparameter tuning. In order to add
    

