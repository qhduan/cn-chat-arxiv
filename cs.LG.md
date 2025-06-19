# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dynamic ASR Pathways: An Adaptive Masking Approach Towards Efficient Pruning of A Multilingual ASR Model.](http://arxiv.org/abs/2309.13018) | 本研究提出了一种自适应掩蔽方法，用于高效地压缩多语种ASR模型。该方法通过动态适应子网络结构，能够在减少性能损失的情况下得到稀疏的单语种模型或稀疏的多语种模型。实验证明，与现有的修剪方法相比，该方法在针对稀疏的单语种模型时表现更好，并且减少了对特定语言进行修剪的需求。 |
| [^2] | [Benchmarking Neural Network Training Algorithms.](http://arxiv.org/abs/2306.07179) | 本文解决了神经网络训练算法基准测试中存在的三个挑战，提出了新的基准测试套件，以促进训练算法效率的进一步提高。 |
| [^3] | [A fast Multiplicative Updates algorithm for Non-negative Matrix Factorization.](http://arxiv.org/abs/2303.17992) | 提出了一种快速的非负矩阵分解乘法更新算法，通过改进交替主体化最小化算法，实现了较快的求解和类似或更好的逼近精度结果。 |

# 详细

[^1]: 动态ASR路径：一种自适应掩蔽方法用于压缩多语种ASR模型的高效修剪

    Dynamic ASR Pathways: An Adaptive Masking Approach Towards Efficient Pruning of A Multilingual ASR Model. (arXiv:2309.13018v1 [eess.AS])

    [http://arxiv.org/abs/2309.13018](http://arxiv.org/abs/2309.13018)

    本研究提出了一种自适应掩蔽方法，用于高效地压缩多语种ASR模型。该方法通过动态适应子网络结构，能够在减少性能损失的情况下得到稀疏的单语种模型或稀疏的多语种模型。实验证明，与现有的修剪方法相比，该方法在针对稀疏的单语种模型时表现更好，并且减少了对特定语言进行修剪的需求。

    

    神经网络修剪是一种有效的方法，可以在性能损失最小的情况下压缩多语种自动语音识别（ASR）模型。然而，这需要对每种语言运行多轮修剪和重新训练。在这项工作中，我们提出了一种自适应掩蔽方法，以两种场景高效地修剪多语种ASR模型，分别得到了稀疏的单语种模型或稀疏的多语种模型（称为动态ASR路径）。我们的方法动态地适应子网络，避免对固定的子网络结构进行过早决策。我们证明了我们的方法在针对稀疏的单语种模型时优于现有的修剪方法。此外，我们还说明了动态ASR路径通过自不同的子网络初始化进行调整，共同发现和训练更好的单一多语种模型的子网络（路径），从而减少了对特定语言进行修剪的需求。

    Neural network pruning offers an effective method for compressing a multilingual automatic speech recognition (ASR) model with minimal performance loss. However, it entails several rounds of pruning and re-training needed to be run for each language. In this work, we propose the use of an adaptive masking approach in two scenarios for pruning a multilingual ASR model efficiently, each resulting in sparse monolingual models or a sparse multilingual model (named as Dynamic ASR Pathways). Our approach dynamically adapts the sub-network, avoiding premature decisions about a fixed sub-network structure. We show that our approach outperforms existing pruning methods when targeting sparse monolingual models. Further, we illustrate that Dynamic ASR Pathways jointly discovers and trains better sub-networks (pathways) of a single multilingual model by adapting from different sub-network initializations, thereby reducing the need for language-specific pruning.
    
[^2]: 神经网络训练算法基准测试

    Benchmarking Neural Network Training Algorithms. (arXiv:2306.07179v1 [cs.LG])

    [http://arxiv.org/abs/2306.07179](http://arxiv.org/abs/2306.07179)

    本文解决了神经网络训练算法基准测试中存在的三个挑战，提出了新的基准测试套件，以促进训练算法效率的进一步提高。

    

    训练算法是每个深度学习流程的重要组成部分。提高训练算法的效率可以节省时间、计算资源，并带来更好、更准确的模型。然而，我们目前还无法可靠地确定最先进的训练算法。本文通过具体实验，证明了加速训练的真正进展需要解决三个基本挑战：如何确定训练何时结束并精确测量训练时间，如何处理测量对确切工作负载详情的敏感性，并公平比较需要超参数调整的算法。为了增加对训练算法效率的了解，我们提出并设计了一些新的基准测试套件。

    Training algorithms, broadly construed, are an essential part of every deep learning pipeline. Training algorithm improvements that speed up training across a wide variety of workloads (e.g., better update rules, tuning protocols, learning rate schedules, or data selection schemes) could save time, save computational resources, and lead to better, more accurate, models. Unfortunately, as a community, we are currently unable to reliably identify training algorithm improvements, or even determine the state-of-the-art training algorithm. In this work, using concrete experiments, we argue that real progress in speeding up training requires new benchmarks that resolve three basic challenges faced by empirical comparisons of training algorithms: (1) how to decide when training is complete and precisely measure training time, (2) how to handle the sensitivity of measurements to exact workload details, and (3) how to fairly compare algorithms that require hyperparameter tuning. In order to add
    
[^3]: 一种快速的非负矩阵分解乘法更新算法

    A fast Multiplicative Updates algorithm for Non-negative Matrix Factorization. (arXiv:2303.17992v1 [math.OC])

    [http://arxiv.org/abs/2303.17992](http://arxiv.org/abs/2303.17992)

    提出了一种快速的非负矩阵分解乘法更新算法，通过改进交替主体化最小化算法，实现了较快的求解和类似或更好的逼近精度结果。

    

    非负矩阵分解是一种重要的无监督机器学习工具，可以将数据矩阵分解为易于解释的部分。过去三十年中出现了许多算法，其中一种广为人知的方法是由李飞飞和才华横溢于2002年提出的乘法更新算法。该算法在许多领域表现良好，具有简单易实现和可适应流行变体的特点。本文建议通过为每个替代子问题制作更紧密的Hessian矩阵的上限来改进乘法更新算法，并将其视为交替主体化最小化算法。在合成数据和实际数据上实践中观察到，所提出的fastMU算法通常比原始的乘法更新算法快数倍，同时在逼近精度方面实现了类似或更好的结果，收敛仍然得到保证。

    Nonnegative Matrix Factorization is an important tool in unsupervised machine learning to decompose a data matrix into a product of parts that are often interpretable. Many algorithms have been proposed during the last three decades. A well-known method is the Multiplicative Updates algorithm proposed by Lee and Seung in 2002. Multiplicative updates have many interesting features: they are simple to implement and can be adapted to popular variants such as sparse Nonnegative Matrix Factorization, and, according to recent benchmarks, is state-of-the-art for many problems where the loss function is not the Frobenius norm. In this manuscript, we propose to improve the Multiplicative Updates algorithm seen as an alternating majorization minimization algorithm by crafting a tighter upper bound of the Hessian matrix for each alternate subproblem. Convergence is still ensured and we observe in practice on both synthetic and real world dataset that the proposed fastMU algorithm is often several
    

