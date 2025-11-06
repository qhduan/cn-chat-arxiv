# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey of Graph Neural Networks in Real world: Imbalance, Noise, Privacy and OOD Challenges](https://arxiv.org/abs/2403.04468) | 本文调查了图神经网络在现实世界中面临的不平衡、噪声、隐私和OOD挑战，并致力于提高模型性能、可靠性和鲁棒性。 |
| [^2] | [Using AI libraries for Incompressible Computational Fluid Dynamics](https://arxiv.org/abs/2402.17913) | 本文提出了一种将AI软件和硬件应用于数值建模领域的新方法，通过重新利用AI方法，如CNN，来解决偏微分方程的标准操作，带来高性能、架构不可知性和易用性。 |
| [^3] | [Variable Selection in Maximum Mean Discrepancy for Interpretable Distribution Comparison.](http://arxiv.org/abs/2311.01537) | 本文研究了数据集比较中的变量选择问题，提出了一种基于最大平均差异的两样本测试方法，通过优化自动相关性检测权重来增强测试的功效，并引入稀疏正则化方法来解决正则化参数选择的问题。 |
| [^4] | [Neural networks learn to magnify areas near decision boundaries.](http://arxiv.org/abs/2301.11375) | 神经网络训练能够放大决策边界附近的局部区域，改善整个系统的泛化能力。 |

# 详细

[^1]: 关于图神经网络在现实世界中的调查：不平衡、噪声、隐私和OOD挑战

    A Survey of Graph Neural Networks in Real world: Imbalance, Noise, Privacy and OOD Challenges

    [https://arxiv.org/abs/2403.04468](https://arxiv.org/abs/2403.04468)

    本文调查了图神经网络在现实世界中面临的不平衡、噪声、隐私和OOD挑战，并致力于提高模型性能、可靠性和鲁棒性。

    

    arXiv:2403.04468v1 发布类型: 跨域 摘要: 图结构化数据表现出普适性和广泛适用性，涵盖社交网络分析、生物化学、金融欺诈检测和网络安全等多个领域。在利用图神经网络（GNNs）取得显著成功方面已经取得了重要进展。然而，在实际应用场景中，模型的训练环境往往远非理想，由于各种不利因素，包括数据分布不平衡、错误数据中存在噪声、敏感信息的隐私保护以及对于OOD场景的泛化能力，导致GNN模型的性能大幅下降。为解决这些问题，人们致力于改善GNN模型在实际应用场景中的性能，提高其可靠性和鲁棒性。本文全面调查了...

    arXiv:2403.04468v1 Announce Type: cross  Abstract: Graph-structured data exhibits universality and widespread applicability across diverse domains, such as social network analysis, biochemistry, financial fraud detection, and network security. Significant strides have been made in leveraging Graph Neural Networks (GNNs) to achieve remarkable success in these areas. However, in real-world scenarios, the training environment for models is often far from ideal, leading to substantial performance degradation of GNN models due to various unfavorable factors, including imbalance in data distribution, the presence of noise in erroneous data, privacy protection of sensitive information, and generalization capability for out-of-distribution (OOD) scenarios. To tackle these issues, substantial efforts have been devoted to improving the performance of GNN models in practical real-world scenarios, as well as enhancing their reliability and robustness. In this paper, we present a comprehensive surv
    
[^2]: 使用AI库进行不可压缩计算流体动力学

    Using AI libraries for Incompressible Computational Fluid Dynamics

    [https://arxiv.org/abs/2402.17913](https://arxiv.org/abs/2402.17913)

    本文提出了一种将AI软件和硬件应用于数值建模领域的新方法，通过重新利用AI方法，如CNN，来解决偏微分方程的标准操作，带来高性能、架构不可知性和易用性。

    

    最近，人们致力于开发高效开源库，以在不同的计算机架构（例如CPU、GPU和新的AI处理器）上执行人工智能（AI）相关的计算。这不仅使基于这些库的算法高效而且在不同架构之间可移植，还大大简化了使用AI开发方法的门槛。本文提出了一种新颖的方法论，将AI软件和硬件的强大功能带入数值建模领域，将AI方法（如卷积神经网络CNN）重新用于数值偏微分方程的标准操作。本工作的目标是将高性能、架构不可知性和易用性引入数值偏微分方程的解决领域。

    arXiv:2402.17913v1 Announce Type: cross  Abstract: Recently, there has been a huge effort focused on developing highly efficient open source libraries to perform Artificial Intelligence (AI) related computations on different computer architectures (for example, CPUs, GPUs and new AI processors). This has not only made the algorithms based on these libraries highly efficient and portable between different architectures, but also has substantially simplified the entry barrier to develop methods using AI. Here, we present a novel methodology to bring the power of both AI software and hardware into the field of numerical modelling by repurposing AI methods, such as Convolutional Neural Networks (CNNs), for the standard operations required in the field of the numerical solution of Partial Differential Equations (PDEs). The aim of this work is to bring the high performance, architecture agnosticism and ease of use into the field of the numerical solution of PDEs. We use the proposed methodol
    
[^3]: 在可解释的分布比较中的最大平均差异中的变量选择

    Variable Selection in Maximum Mean Discrepancy for Interpretable Distribution Comparison. (arXiv:2311.01537v1 [stat.ML])

    [http://arxiv.org/abs/2311.01537](http://arxiv.org/abs/2311.01537)

    本文研究了数据集比较中的变量选择问题，提出了一种基于最大平均差异的两样本测试方法，通过优化自动相关性检测权重来增强测试的功效，并引入稀疏正则化方法来解决正则化参数选择的问题。

    

    两样本测试是为了判断两个数据集是否来自同一分布。本文研究了两样本测试中的变量选择问题，即识别造成两个分布差异的变量（或维度）的任务。这个任务与模式分析和机器学习的许多问题相关，如数据集漂移适应、因果推断和模型验证。我们的方法基于基于最大平均差异（MMD）的两样本检验。我们优化针对各个变量定义的自动相关性检测（ARD）权重，以最大化基于MMD的检验的功率。对于这种优化，我们引入了稀疏正则化，并提出了两种方法来解决选择适当正则化参数的问题。一种方法是以数据驱动的方式确定正则化参数，另一种方法是合并不同正则化参数的结果。我们确认了这个方法的有效性。

    Two-sample testing decides whether two datasets are generated from the same distribution. This paper studies variable selection for two-sample testing, the task being to identify the variables (or dimensions) responsible for the discrepancies between the two distributions. This task is relevant to many problems of pattern analysis and machine learning, such as dataset shift adaptation, causal inference and model validation. Our approach is based on a two-sample test based on the Maximum Mean Discrepancy (MMD). We optimise the Automatic Relevance Detection (ARD) weights defined for individual variables to maximise the power of the MMD-based test. For this optimisation, we introduce sparse regularisation and propose two methods for dealing with the issue of selecting an appropriate regularisation parameter. One method determines the regularisation parameter in a data-driven way, and the other aggregates the results of different regularisation parameters. We confirm the validity of the pr
    
[^4]: 神经网络学习放大决策边界附近的区域

    Neural networks learn to magnify areas near decision boundaries. (arXiv:2301.11375v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.11375](http://arxiv.org/abs/2301.11375)

    神经网络训练能够放大决策边界附近的局部区域，改善整个系统的泛化能力。

    

    我们研究了训练如何塑造神经网络特征图诱导的黎曼几何。在宽度为无限的情况下，具有随机参数的神经网络在输入空间上引导高度对称的度量。训练分类任务的网络中的特征学习放大了沿决策边界的局部区域。这些变化与先前提出的用于手动调整核方法以改善泛化的几何方法一致。

    We study how training molds the Riemannian geometry induced by neural network feature maps. At infinite width, neural networks with random parameters induce highly symmetric metrics on input space. Feature learning in networks trained to perform classification tasks magnifies local areas along decision boundaries. These changes are consistent with previously proposed geometric approaches for hand-tuning of kernel methods to improve generalization.
    

