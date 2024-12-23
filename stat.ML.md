# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Boosting, Voting Classifiers and Randomized Sample Compression Schemes](https://arxiv.org/abs/2402.02976) | 本研究提出了一种随机提升算法来解决传统提升算法的性能问题，并通过构建一个通用框架将样本压缩方法扩展到支持随机学习算法，实现了在样本大小上具有单对数依赖的泛化错误。 |
| [^2] | [Variational measurement-based quantum computation for generative modeling.](http://arxiv.org/abs/2310.13524) | 这项研究提出了一种基于测量的变分量子计算算法，将量子测量的随机性视为计算资源，并应用于生成建模任务。 |
| [^3] | [Learning ECG signal features without backpropagation.](http://arxiv.org/abs/2307.01930) | 该论文提出了一种用于生成时间序列数据表示的新方法，依靠理论物理的思想以数据驱动的方式构建紧凑的表示。该方法能够捕捉数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性，并可以在广义设置中应用。 |
| [^4] | [Variable Selection for Kernel Two-Sample Tests.](http://arxiv.org/abs/2302.07415) | 本文提出了一种解决双样本检验中变量选择问题的框架，利用核最大均值差异统计量，以最大化方差正则化的MMD统计量。实验结果证明其超群表现。 |

# 详细

[^1]: 提升，投票分类器和随机采样压缩方案

    Boosting, Voting Classifiers and Randomized Sample Compression Schemes

    [https://arxiv.org/abs/2402.02976](https://arxiv.org/abs/2402.02976)

    本研究提出了一种随机提升算法来解决传统提升算法的性能问题，并通过构建一个通用框架将样本压缩方法扩展到支持随机学习算法，实现了在样本大小上具有单对数依赖的泛化错误。

    

    在提升中，我们旨在利用多个弱学习器来产生一个强学习器。这个范式的核心是将强学习器建模为一个投票分类器，它输出弱学习器的加权多数投票。尽管许多成功的提升算法，如标志性的AdaBoost，产生投票分类器，但它们的理论性能长期以来一直不够优化：迄今为止，已知的使投票分类器达到给定准确性所需的训练样本数的最佳界限总是至少包含至多两个对数因子，而这已经超过了一般的弱到强学习器所能实现的范围。在这项工作中，我们通过提出一种随机提升算法打破这一障碍，该算法输出的投票分类器在样本大小上包含单对数依赖的泛化错误。我们通过构建一个通用框架将样本压缩方法扩展到支持随机学习算法来获得这个结果。

    In boosting, we aim to leverage multiple weak learners to produce a strong learner. At the center of this paradigm lies the concept of building the strong learner as a voting classifier, which outputs a weighted majority vote of the weak learners. While many successful boosting algorithms, such as the iconic AdaBoost, produce voting classifiers, their theoretical performance has long remained sub-optimal: the best known bounds on the number of training examples necessary for a voting classifier to obtain a given accuracy has so far always contained at least two logarithmic factors above what is known to be achievable by general weak-to-strong learners. In this work, we break this barrier by proposing a randomized boosting algorithm that outputs voting classifiers whose generalization error contains a single logarithmic dependency on the sample size. We obtain this result by building a general framework that extends sample compression methods to support randomized learning algorithms ba
    
[^2]: 基于测量的变分量子计算用于生成建模

    Variational measurement-based quantum computation for generative modeling. (arXiv:2310.13524v1 [quant-ph])

    [http://arxiv.org/abs/2310.13524](http://arxiv.org/abs/2310.13524)

    这项研究提出了一种基于测量的变分量子计算算法，将量子测量的随机性视为计算资源，并应用于生成建模任务。

    

    基于测量的量子计算（MBQC）提供了一种基本独特的范例来设计量子算法。在MBQC中，由于量子测量的固有随机性，自然的操作不是确定性和幺正的，而是通过概率附带的。然而，到目前为止，MBQC的主要算法应用是完全抵消这种概率性质，以模拟表达在电路模型中的幺正计算。在这项工作中，我们提出了设计MBQC算法的思路，该算法接受这种固有随机性，并将MBQC中的随机附带视为计算资源。我们考虑了随机性有益的自然应用，即生成建模，这是一个以生成复杂概率分布为中心的机器学习任务。为了解决这个任务，我们提出了一个具有控制参数的变分MBQC算法，可以直接调整允许在计算中引入的随机程度。

    Measurement-based quantum computation (MBQC) offers a fundamentally unique paradigm to design quantum algorithms. Indeed, due to the inherent randomness of quantum measurements, the natural operations in MBQC are not deterministic and unitary, but are rather augmented with probabilistic byproducts. Yet, the main algorithmic use of MBQC so far has been to completely counteract this probabilistic nature in order to simulate unitary computations expressed in the circuit model. In this work, we propose designing MBQC algorithms that embrace this inherent randomness and treat the random byproducts in MBQC as a resource for computation. As a natural application where randomness can be beneficial, we consider generative modeling, a task in machine learning centered around generating complex probability distributions. To address this task, we propose a variational MBQC algorithm equipped with control parameters that allow to directly adjust the degree of randomness to be admitted in the comput
    
[^3]: 学习ECG信号特征的非反向传播方法

    Learning ECG signal features without backpropagation. (arXiv:2307.01930v1 [cs.LG])

    [http://arxiv.org/abs/2307.01930](http://arxiv.org/abs/2307.01930)

    该论文提出了一种用于生成时间序列数据表示的新方法，依靠理论物理的思想以数据驱动的方式构建紧凑的表示。该方法能够捕捉数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性，并可以在广义设置中应用。

    

    表示学习已经成为机器学习领域的一个关键研究领域，它旨在发现用于提高分类和预测等下游任务的原始数据的有效特征的有效方法。在本文中，我们提出了一种用于生成时间序列类型数据表示的新方法。这种方法依靠理论物理的思想以数据驱动的方式构建紧凑的表示，并可以捕捉到数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性。这个新方法旨在识别能够有效捕捉属于特定类别的样本之间共享特征的线性规律。通过随后利用这些规律在前向方式下生成一个与分类器无关的表示，它们可以在广义设置中应用。我们展示了我们方法的有效性。

    Representation learning has become a crucial area of research in machine learning, as it aims to discover efficient ways of representing raw data with useful features to increase the effectiveness, scope and applicability of downstream tasks such as classification and prediction. In this paper, we propose a novel method to generate representations for time series-type data. This method relies on ideas from theoretical physics to construct a compact representation in a data-driven way, and it can capture both the underlying structure of the data and task-specific information while still remaining intuitive, interpretable and verifiable. This novel methodology aims to identify linear laws that can effectively capture a shared characteristic among samples belonging to a specific class. By subsequently utilizing these laws to generate a classifier-agnostic representation in a forward manner, they become applicable in a generalized setting. We demonstrate the effectiveness of our approach o
    
[^4]: 变量选择在核双样本检验中的应用

    Variable Selection for Kernel Two-Sample Tests. (arXiv:2302.07415v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.07415](http://arxiv.org/abs/2302.07415)

    本文提出了一种解决双样本检验中变量选择问题的框架，利用核最大均值差异统计量，以最大化方差正则化的MMD统计量。实验结果证明其超群表现。

    

    本文考虑了两样本检验中的变量选择问题，旨在选择区分两组样本的最有信息变量。为了解决该问题，我们提出了一种基于核最大均值差异（MMD）的框架。我们的方法寻求一组变量，其预先确定的大小最大化方差正则化的MMD统计量。这种计算形式也对应于在文献中研究的控制类型I错误的同时最小化异质类型II错误。我们介绍了混合整数编程公式，并提供了线性和二次类型内核函数的精确和近似算法，并具有性能保证。实验结果证明了我们的框架的卓越性能。

    We consider the variable selection problem for two-sample tests, aiming to select the most informative variables to distinguish samples from two groups. To solve this problem, we propose a framework based on the kernel maximum mean discrepancy (MMD). Our approach seeks a group of variables with a pre-specified size that maximizes the variance-regularized MMD statistics. This formulation also corresponds to the minimization of asymptotic type-II error while controlling type-I error, as studied in the literature. We present mixed-integer programming formulations and offer exact and approximation algorithms with performance guarantees for linear and quadratic types of kernel functions. Experimental results demonstrate the superior performance of our framework.
    

