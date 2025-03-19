# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Accuracy-Preserving Calibration via Statistical Modeling on Probability Simplex](https://arxiv.org/abs/2402.13765) | 提出一种使用Concrete分布作为概率单纯形上的概率模型的保持精度的校准方法，并证明其在交叉熵损失上训练的DNN模型具有最优性，同时提出了一种有效的样本生成方法。 |
| [^2] | [A powerful rank-based correction to multiple testing under positive dependency.](http://arxiv.org/abs/2311.10900) | 我们提出了一种基于秩的多重检验修正方法，能够有效利用正相关的统计假设检验之间的依赖关系，并在存在正相关依赖情况下优于Bonferroni修正。我们的方法尤其适用于并行置换检验，在保证FWER控制的同时保持高统计功效。 |
| [^3] | [Kernel Single Proxy Control for Deterministic Confounding.](http://arxiv.org/abs/2308.04585) | 本研究考虑了具有未观测混淆因素的因果效应估计问题，在结果是确定性生成的情况下，提出了一种使用单一代理变量的内核方法，通过两阶段回归和最大矩约束的方法可以一致估计因果效应，并在合成数据集上成功恢复了因果效应。 |
| [^4] | [Unfair Utilities and First Steps Towards Improving Them.](http://arxiv.org/abs/2306.00636) | 该论文提出了一个新的公平框架——考虑政策优化哪个效用，定义了信息价值公平，提出不应使用不满足这一标准的实用程序，并探讨了修改实用程序以满足此公平标准可能对最优政策产生的影响。 |
| [^5] | [Linear Neural Network Layers Promote Learning Single- and Multiple-Index Models.](http://arxiv.org/abs/2305.15598) | 本研究探究了过度参数化的深度神经网络的偏见，发现在ReLU网络中添加线性层有助于逼近具有低秩线性算子和低表示成本函数组成的函数，从而得到一个与低维子空间垂直方向近乎恒定的插值函数。 |

# 详细

[^1]: 通过概率单纯形上的统计建模实现保持精度的校准

    Accuracy-Preserving Calibration via Statistical Modeling on Probability Simplex

    [https://arxiv.org/abs/2402.13765](https://arxiv.org/abs/2402.13765)

    提出一种使用Concrete分布作为概率单纯形上的概率模型的保持精度的校准方法，并证明其在交叉熵损失上训练的DNN模型具有最优性，同时提出了一种有效的样本生成方法。

    

    基于深度神经网络（DNNs）的分类模型必须进行校准，以评估预测结果的可靠性。一些最近的校准方法采用了概率单纯形上的概率模型。然而，这些校准方法无法保持预训练模型的准确性，即使这些模型具有很高的分类准确性。我们提出了一种使用Concrete分布作为概率单纯形上的概率模型的保持精度的校准方法。我们在理论上证明，在交叉熵损失上训练的DNN模型具有Concrete分布参数的最优性。我们还提出了一种有效的方法，可以合成生成样本，用于在概率单纯形上训练概率模型。我们证明了所提出的方法在精度保持校准任务上可以优于以往的方法，使用基准测试。

    arXiv:2402.13765v1 Announce Type: new  Abstract: Classification models based on deep neural networks (DNNs) must be calibrated to measure the reliability of predictions. Some recent calibration methods have employed a probabilistic model on the probability simplex. However, these calibration methods cannot preserve the accuracy of pre-trained models, even those with a high classification accuracy. We propose an accuracy-preserving calibration method using the Concrete distribution as the probabilistic model on the probability simplex. We theoretically prove that a DNN model trained on cross-entropy loss has optimality as the parameter of the Concrete distribution. We also propose an efficient method that synthetically generates samples for training probabilistic models on the probability simplex. We demonstrate that the proposed method can outperform previous methods in accuracy-preserving calibration tasks using benchmarks.
    
[^2]: 一种基于秩的多重检验正相关依赖的强大修正方法

    A powerful rank-based correction to multiple testing under positive dependency. (arXiv:2311.10900v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2311.10900](http://arxiv.org/abs/2311.10900)

    我们提出了一种基于秩的多重检验修正方法，能够有效利用正相关的统计假设检验之间的依赖关系，并在存在正相关依赖情况下优于Bonferroni修正。我们的方法尤其适用于并行置换检验，在保证FWER控制的同时保持高统计功效。

    

    我们开发了一种能够高效利用可能相关的统计假设检验之间正相关性的家族误差率(FWER)控制的新型多重假设检验修正算法$\texttt{max-rank}$。我们的方法概念上很直观，依赖于在计算的统计检验的秩域使用$\max$算子。通过理论和经验的比较，我们证明了在存在正相关依赖的情况下，我们的方法优于经常使用的Bonferroni修正，而在不存在正相关依赖的情况下等效。我们的优势随着测试数量的增加而增加，同时在保证FWER控制的情况下保持高统计功效。我们特别将我们的算法应用于并行置换检验的背景中，这是在我们主要应用的一种复杂预测场景中产生的情况下。

    We develop a novel multiple hypothesis testing correction with family-wise error rate (FWER) control that efficiently exploits positive dependencies between potentially correlated statistical hypothesis tests. Our proposed algorithm $\texttt{max-rank}$ is conceptually straight-forward, relying on the use of a $\max$-operator in the rank domain of computed test statistics. We compare our approach to the frequently employed Bonferroni correction, theoretically and empirically demonstrating its superiority over Bonferroni in the case of existing positive dependency, and its equivalence otherwise. Our advantage over Bonferroni increases as the number of tests rises, and we maintain high statistical power whilst ensuring FWER control. We specifically frame our algorithm in the context of parallel permutation testing, a scenario that arises in our primary application of conformal prediction, a recently popularized approach for quantifying uncertainty in complex predictive settings.
    
[^3]: 决定性混淆下的内核单一代理控制

    Kernel Single Proxy Control for Deterministic Confounding. (arXiv:2308.04585v1 [stat.ML])

    [http://arxiv.org/abs/2308.04585](http://arxiv.org/abs/2308.04585)

    本研究考虑了具有未观测混淆因素的因果效应估计问题，在结果是确定性生成的情况下，提出了一种使用单一代理变量的内核方法，通过两阶段回归和最大矩约束的方法可以一致估计因果效应，并在合成数据集上成功恢复了因果效应。

    

    本文考虑具有未观测混淆因素的因果效应估计问题，其中我们观测到与混淆因素相关的代理变量。尽管代理因果学习（PCL）使用两个代理变量来恢复真实的因果效应，我们证明如果结果是确定性生成的，则使用单个代理变量就足以进行因果估计，并概括了控制结果校准法（COCA）。我们提出了两种基于内核的方法：一种基于两阶段回归方法，另一种基于最大矩约束方法。我们证明了这两种方法都可以一致地估计因果效应，并通过合成数据集的实证实验成功地恢复了因果效应。

    We consider the problem of causal effect estimation with an unobserved confounder, where we observe a proxy variable that is associated with the confounder. Although Proxy Causal Learning (PCL) uses two proxy variables to recover the true causal effect, we show that a single proxy variable is sufficient for causal estimation if the outcome is generated deterministically, generalizing Control Outcome Calibration Approach (COCA). We propose two kernel-based methods for this setting: the first based on the two-stage regression approach, and the second based on a maximum moment restriction approach. We prove that both approaches can consistently estimate the causal effect, and we empirically demonstrate that we can successfully recover the causal effect on a synthetic dataset.
    
[^4]: 不公平的实用程序及其改进的第一步

    Unfair Utilities and First Steps Towards Improving Them. (arXiv:2306.00636v1 [stat.ML])

    [http://arxiv.org/abs/2306.00636](http://arxiv.org/abs/2306.00636)

    该论文提出了一个新的公平框架——考虑政策优化哪个效用，定义了信息价值公平，提出不应使用不满足这一标准的实用程序，并探讨了修改实用程序以满足此公平标准可能对最优政策产生的影响。

    

    许多公平标准对政策或预测器的选择进行限制。在这项工作中，我们提出了一个不同的思考公平的框架：我们考虑政策正在优化哪个效用，而不是限制政策或预测器的选择。我们定义了信息价值公平，并建议不使用不满足此标准的实用程序。我们描述了如何修改实用程序以满足这种公平标准，并讨论了这可能对相应的最优政策产生的影响。

    Many fairness criteria constrain the policy or choice of predictors. In this work, we propose a different framework for thinking about fairness: Instead of constraining the policy or choice of predictors, we consider which utility a policy is optimizing for. We define value of information fairness and propose to not use utilities that do not satisfy this criterion. We describe how to modify a utility to satisfy this fairness criterion and discuss the consequences this might have on the corresponding optimal policies.
    
[^5]: 线性神经网络层促进学习单指数和多指数模型

    Linear Neural Network Layers Promote Learning Single- and Multiple-Index Models. (arXiv:2305.15598v1 [cs.LG])

    [http://arxiv.org/abs/2305.15598](http://arxiv.org/abs/2305.15598)

    本研究探究了过度参数化的深度神经网络的偏见，发现在ReLU网络中添加线性层有助于逼近具有低秩线性算子和低表示成本函数组成的函数，从而得到一个与低维子空间垂直方向近乎恒定的插值函数。

    

    本文探究了深度大于两层的过度参数化神经网络的隐含偏见。我们的框架考虑了一类深度不同但容量相同的网络，它们具有不同的显式定义的表示成本。神经网络架构诱导的函数的表示成本是网络表示该函数所需的平方权重之和的最小值；它反映了与该架构相关的函数空间偏差。结果表明，将线性层添加到ReLU网络会产生一个表示成本，这有利于使用两层网络来逼近由低秩线性算子和具有低表示成本的函数组成的函数。具体来说，使用神经网络以最小的表示成本拟合训练数据会得到一个与低维子空间垂直方向近乎恒定的插值函数。

    This paper explores the implicit bias of overparameterized neural networks of depth greater than two layers. Our framework considers a family of networks of varying depths that all have the same capacity but different implicitly defined representation costs. The representation cost of a function induced by a neural network architecture is the minimum sum of squared weights needed for the network to represent the function; it reflects the function space bias associated with the architecture. Our results show that adding linear layers to a ReLU network yields a representation cost that favors functions that can be approximated by a low-rank linear operator composed with a function with low representation cost using a two-layer network. Specifically, using a neural network to fit training data with minimum representation cost yields an interpolating function that is nearly constant in directions orthogonal to a low-dimensional subspace. This means that the learned network will approximate
    

