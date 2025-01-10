# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Two-Scale Complexity Measure for Deep Learning Models.](http://arxiv.org/abs/2401.09184) | 这篇论文介绍了一种用于统计模型的新容量测量2sED，可以可靠地限制泛化误差，并且与训练误差具有很好的相关性。此外，对于深度学习模型，我们展示了如何通过逐层迭代的方法有效地近似2sED，从而处理大量参数的情况。 |
| [^2] | [Causal Machine Learning for Moderation Effects.](http://arxiv.org/abs/2401.08290) | 本文提出了一种新的参数，平衡群体平均处理效应（BGATE），用于解释处理在群体间的效应差异，该参数基于因果机器学习方法，对离散处理进行估计。通过比较两个BGATE的差异，能更好地分析处理的异质性。 |
| [^3] | [A New Transformation Approach for Uplift Modeling with Binary Outcome.](http://arxiv.org/abs/2310.05549) | 本论文提出了一种新的二元结果提升建模转换方法，利用了零结果样本的信息并且易于使用。 (arXiv:2310.05549v1 [stat.ML]) |
| [^4] | [Optimality of Message-Passing Architectures for Sparse Graphs.](http://arxiv.org/abs/2305.10391) | 本研究证明了将消息传递神经网络应用于稀疏图的节点分类任务是渐近本地贝叶斯最优的，提出了一种实现最优分类器的算法，并将最优分类器的性能理论上与现有学习方法进行了比较。 |
| [^5] | [Explaining the Behavior of Black-Box Prediction Algorithms with Causal Learning.](http://arxiv.org/abs/2006.02482) | 本文提出了一种用于解释黑箱预测算法行为的因果学习方法，通过学习因果图表示来提供因果解释，弥补了现有方法的缺点，即解释单元更加可解释且考虑了宏观级特征和未测量的混淆。 |

# 详细

[^1]: 深度学习模型的两尺度复杂度测量

    A Two-Scale Complexity Measure for Deep Learning Models. (arXiv:2401.09184v1 [stat.ML])

    [http://arxiv.org/abs/2401.09184](http://arxiv.org/abs/2401.09184)

    这篇论文介绍了一种用于统计模型的新容量测量2sED，可以可靠地限制泛化误差，并且与训练误差具有很好的相关性。此外，对于深度学习模型，我们展示了如何通过逐层迭代的方法有效地近似2sED，从而处理大量参数的情况。

    

    我们引入了一种基于有效维度的统计模型新容量测量2sED。这个新的数量在对模型进行温和假设的情况下，可以可靠地限制泛化误差。此外，对于标准数据集和流行的模型架构的模拟结果表明，2sED与训练误差具有很好的相关性。对于马尔可夫模型，我们展示了如何通过逐层迭代的方法有效地从下方近似2sED，从而解决具有大量参数的深度学习模型。模拟结果表明，这种近似对不同的突出模型和数据集都很好。

    We introduce a novel capacity measure 2sED for statistical models based on the effective dimension. The new quantity provably bounds the generalization error under mild assumptions on the model. Furthermore, simulations on standard data sets and popular model architectures show that 2sED correlates well with the training error. For Markovian models, we show how to efficiently approximate 2sED from below through a layerwise iterative approach, which allows us to tackle deep learning models with a large number of parameters. Simulation results suggest that the approximation is good for different prominent models and data sets.
    
[^2]: 因果机器学习用于中介效应。 (arXiv:2401.08290v1 [econ.EM])

    Causal Machine Learning for Moderation Effects. (arXiv:2401.08290v1 [econ.EM])

    [http://arxiv.org/abs/2401.08290](http://arxiv.org/abs/2401.08290)

    本文提出了一种新的参数，平衡群体平均处理效应（BGATE），用于解释处理在群体间的效应差异，该参数基于因果机器学习方法，对离散处理进行估计。通过比较两个BGATE的差异，能更好地分析处理的异质性。

    

    对于任何决策者来说，了解决策（处理）对整体和子群的影响是非常有价值的。因果机器学习最近提供了用于估计群体平均处理效应（GATE）的工具，以更好地理解处理的异质性。本文解决了在考虑其他协变量变化的情况下解释群体间处理效应差异的难题。我们提出了一个新的参数，即平衡群体平均处理效应（BGATE），它衡量了具有特定分布的先验确定协变量的GATE。通过比较两个BGATE的差异，我们可以更有意义地分析异质性，而不仅仅比较两个GATE。这个参数的估计策略是基于无混淆设置中离散处理的双重/去偏机器学习，该估计量在标准条件下表现为$\sqrt{N}$一致性和渐近正态性。添加额外的标识

    It is valuable for any decision maker to know the impact of decisions (treatments) on average and for subgroups. The causal machine learning literature has recently provided tools for estimating group average treatment effects (GATE) to understand treatment heterogeneity better. This paper addresses the challenge of interpreting such differences in treatment effects between groups while accounting for variations in other covariates. We propose a new parameter, the balanced group average treatment effect (BGATE), which measures a GATE with a specific distribution of a priori-determined covariates. By taking the difference of two BGATEs, we can analyse heterogeneity more meaningfully than by comparing two GATEs. The estimation strategy for this parameter is based on double/debiased machine learning for discrete treatments in an unconfoundedness setting, and the estimator is shown to be $\sqrt{N}$-consistent and asymptotically normal under standard conditions. Adding additional identifyin
    
[^3]: 一个新的二元结果提升建模转换方法

    A New Transformation Approach for Uplift Modeling with Binary Outcome. (arXiv:2310.05549v1 [stat.ML])

    [http://arxiv.org/abs/2310.05549](http://arxiv.org/abs/2310.05549)

    本论文提出了一种新的二元结果提升建模转换方法，利用了零结果样本的信息并且易于使用。 (arXiv:2310.05549v1 [stat.ML])

    

    提升建模在市场营销和客户保留等领域中得到了有效应用，用于针对那些由于活动或治疗更有可能产生反应的客户。本文设计了一种新颖的二元结果转换方法，解锁了零结果样本的全部价值。

    Uplift modeling has been used effectively in fields such as marketing and customer retention, to target those customers who are more likely to respond due to the campaign or treatment. Essentially, it is a machine learning technique that predicts the gain from performing some action with respect to not taking it. A popular class of uplift models is the transformation approach that redefines the target variable with the original treatment indicator. These transformation approaches only need to train and predict the difference in outcomes directly. The main drawback of these approaches is that in general it does not use the information in the treatment indicator beyond the construction of the transformed outcome and usually is not efficient. In this paper, we design a novel transformed outcome for the case of the binary target variable and unlock the full value of the samples with zero outcome. From a practical perspective, our new approach is flexible and easy to use. Experimental resul
    
[^4]: 稀疏图的消息传递架构的最优性

    Optimality of Message-Passing Architectures for Sparse Graphs. (arXiv:2305.10391v1 [cs.LG])

    [http://arxiv.org/abs/2305.10391](http://arxiv.org/abs/2305.10391)

    本研究证明了将消息传递神经网络应用于稀疏图的节点分类任务是渐近本地贝叶斯最优的，提出了一种实现最优分类器的算法，并将最优分类器的性能理论上与现有学习方法进行了比较。

    

    我们研究了特征装饰图上的节点分类问题，在稀疏设置下，即节点的预期度数为节点数的O(1)时。这样的图通常被称为本地树状图。我们引入了一种叫做渐近本地贝叶斯最优性的节点分类任务的贝叶斯最优性概念，并根据这个标准计算了具有任意节点特征和边连接分布的相当一般的统计数据模型的最优分类器。该最优分类器可以使用消息传递图神经网络架构实现。然后我们计算了该分类器的泛化误差，并在一个已经研究充分的统计模型上从理论上与现有的学习方法进行比较。我们发现，在低图信号的情况下，最佳消息传递架构插值于标准MLP和一种典型的c架构之间。

    We study the node classification problem on feature-decorated graphs in the sparse setting, i.e., when the expected degree of a node is $O(1)$ in the number of nodes. Such graphs are typically known to be locally tree-like. We introduce a notion of Bayes optimality for node classification tasks, called asymptotic local Bayes optimality, and compute the optimal classifier according to this criterion for a fairly general statistical data model with arbitrary distributions of the node features and edge connectivity. The optimal classifier is implementable using a message-passing graph neural network architecture. We then compute the generalization error of this classifier and compare its performance against existing learning methods theoretically on a well-studied statistical model with naturally identifiable signal-to-noise ratios (SNRs) in the data. We find that the optimal message-passing architecture interpolates between a standard MLP in the regime of low graph signal and a typical c
    
[^5]: 用因果学习解释黑箱预测算法的行为

    Explaining the Behavior of Black-Box Prediction Algorithms with Causal Learning. (arXiv:2006.02482v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2006.02482](http://arxiv.org/abs/2006.02482)

    本文提出了一种用于解释黑箱预测算法行为的因果学习方法，通过学习因果图表示来提供因果解释，弥补了现有方法的缺点，即解释单元更加可解释且考虑了宏观级特征和未测量的混淆。

    

    因果学方法在解释黑箱预测模型（例如基于图像像素数据训练的深度神经网络）方面越来越受欢迎。然而，现有方法存在两个重要缺点：（i）“解释单元”是相关预测模型的微观级输入，例如图像像素，而不是更有用于理解如何可能改变算法行为的可解释的宏观级特征；（ii）现有方法假设特征与目标模型预测之间不存在未测量的混淆，这在解释单元是宏观级变量时不成立。我们关注的是在分析人员无法访问目标预测算法内部工作原理的重要情况，而只能根据特定输入查询模型输出的能力。为了在这种情况下提供因果解释，我们提出学习因果图表示，允许更好地理解算法的行为。

    Causal approaches to post-hoc explainability for black-box prediction models (e.g., deep neural networks trained on image pixel data) have become increasingly popular. However, existing approaches have two important shortcomings: (i) the "explanatory units" are micro-level inputs into the relevant prediction model, e.g., image pixels, rather than interpretable macro-level features that are more useful for understanding how to possibly change the algorithm's behavior, and (ii) existing approaches assume there exists no unmeasured confounding between features and target model predictions, which fails to hold when the explanatory units are macro-level variables. Our focus is on the important setting where the analyst has no access to the inner workings of the target prediction algorithm, rather only the ability to query the output of the model in response to a particular input. To provide causal explanations in such a setting, we propose to learn causal graphical representations that allo
    

