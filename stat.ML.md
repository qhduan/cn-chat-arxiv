# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Latent Attention for Linear Time Transformers](https://arxiv.org/abs/2402.17512) | 提出了一种基于潜在向量定义注意力的方法，将标准transformer中的注意力机制的时间复杂度从二次方降低到与时间线性相关，表现与标准注意力媲美，但允许上下文窗口扩展到远远超出标准的范围。 |
| [^2] | [Average gradient outer product as a mechanism for deep neural collapse](https://arxiv.org/abs/2402.13728) | 本文通过提供证据表明，深度神经网络中的神经坍塌主要是通过平均梯度外积进行深度特征学习的，权重的奇异结构与AGOP高度相关，导致类内变异坍塌。 |
| [^3] | [Statistical Test for Generated Hypotheses by Diffusion Models](https://arxiv.org/abs/2402.11789) | 本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。 |
| [^4] | [DISCOUNT: Distributional Counterfactual Explanation With Optimal Transport.](http://arxiv.org/abs/2401.13112) | 本文提出了使用最优传输进行分布式对抗解释的方法DISCOUNT，将对抗解释的概念扩展到整个输入输出分布，并通过统计置信度来支撑这一方法。 |
| [^5] | [Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression.](http://arxiv.org/abs/2309.07810) | 这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。 |
| [^6] | [Benchmarks and Custom Package for Electrical Load Forecasting.](http://arxiv.org/abs/2307.07191) | 本文提供了一个全面的电力负荷预测存档，包括负荷领域特定的特征工程，帮助模型更好地模拟负荷数据，并提供了一种新的损失函数来最小化后续任务的成本。 |

# 详细

[^1]: Latent Attention for Linear Time Transformers

    Latent Attention for Linear Time Transformers

    [https://arxiv.org/abs/2402.17512](https://arxiv.org/abs/2402.17512)

    提出了一种基于潜在向量定义注意力的方法，将标准transformer中的注意力机制的时间复杂度从二次方降低到与时间线性相关，表现与标准注意力媲美，但允许上下文窗口扩展到远远超出标准的范围。

    

    标准transformer中的注意力机制的时间复杂度随着序列长度的增加呈二次方增长。我们引入一种通过定义潜在向量的注意力来将其降低到与时间线性相关的方法。该方法可以轻松作为标准注意力机制的替代品。我们的“Latte Transformer”模型可用于双向和单向任务，因果版本允许一种在推理语言生成任务中内存和时间高效的递归实现。标准transformer的下一个标记预测随着序列长度线性增长，而Latte Transformer计算下一个标记所需的时间是恒定的。我们的方法的实证表现可与标准注意力媲美，但允许将上下文窗口扩展到远远超出标准注意力实际可行的范围。

    arXiv:2402.17512v1 Announce Type: new  Abstract: The time complexity of the standard attention mechanism in a transformer scales quadratically with the length of the sequence. We introduce a method to reduce this to linear scaling with time, based on defining attention via latent vectors. The method is readily usable as a drop-in replacement for the standard attention mechanism. Our "Latte Transformer" model can be implemented for both bidirectional and unidirectional tasks, with the causal version allowing a recurrent implementation which is memory and time-efficient during inference of language generation tasks. Whilst next token prediction scales linearly with the sequence length for a standard transformer, a Latte Transformer requires constant time to compute the next token. The empirical performance of our method is comparable to standard attention, yet allows scaling to context windows much larger than practical in standard attention.
    
[^2]: 平均梯度外积作为深度神经坍塌机制的研究

    Average gradient outer product as a mechanism for deep neural collapse

    [https://arxiv.org/abs/2402.13728](https://arxiv.org/abs/2402.13728)

    本文通过提供证据表明，深度神经网络中的神经坍塌主要是通过平均梯度外积进行深度特征学习的，权重的奇异结构与AGOP高度相关，导致类内变异坍塌。

    

    Deep Neural Collapse (DNC)指的是深度神经网络(DNNs)最后几层数据表示的惊人刚性结构。尽管这种现象在各种情境中都得到了测量，但其出现只有部分被理解。本文提供了充分证据，表明DNC主要是通过平均梯度外积(AGOP)进行深度特征学习而发生的。相比于解释神经坍塌的特征不可知方法，如无约束特征模型，这一进展更进一步。我们继续提供证据表明，权重的右奇异向量和奇异值是DNN中类内变异坍塌的主要因素。正如最近的研究所示，这种奇异结构与AGOP的高度相关。然后我们在实验和理论上证明了AGOP在随机初始化的神经网络中引发神经坍塌。

    arXiv:2402.13728v1 Announce Type: new  Abstract: Deep Neural Collapse (DNC) refers to the surprisingly rigid structure of the data representations in the final layers of Deep Neural Networks (DNNs). Though the phenomenon has been measured in a wide variety of settings, its emergence is only partially understood. In this work, we provide substantial evidence that DNC formation occurs primarily through deep feature learning with the average gradient outer product (AGOP). This takes a step further compared to efforts that explain neural collapse via feature-agnostic approaches, such as the unconstrained features model. We proceed by providing evidence that the right singular vectors and values of the weights are responsible for the majority of within-class variability collapse in DNNs. As shown in recent work, this singular structure is highly correlated with that of the AGOP. We then establish experimentally and theoretically that AGOP induces neural collapse in a randomly initialized ne
    
[^3]: 通过扩散模型生成的假设的统计检验

    Statistical Test for Generated Hypotheses by Diffusion Models

    [https://arxiv.org/abs/2402.11789](https://arxiv.org/abs/2402.11789)

    本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。

    

    AI的增强性能加速了其融入科学研究。特别是，利用生成式AI创建科学假设是很有前途的，并且正在越来越多地应用于各个领域。然而，当使用AI生成的假设进行关键决策（如医学诊断）时，验证它们的可靠性至关重要。在本研究中，我们考虑使用扩散模型生成的图像进行医学诊断任务，并提出了一种统计检验来量化其可靠性。所提出的统计检验的基本思想是使用选择性推断框架，我们考虑在生成的图像是由经过训练的扩散模型产生的这一事实条件下的统计检验。利用所提出的方法，医学图像诊断结果的统计可靠性可以以p值的形式量化，从而实现在控制错误率的情况下进行决策。

    arXiv:2402.11789v1 Announce Type: cross  Abstract: The enhanced performance of AI has accelerated its integration into scientific research. In particular, the use of generative AI to create scientific hypotheses is promising and is increasingly being applied across various fields. However, when employing AI-generated hypotheses for critical decisions, such as medical diagnoses, verifying their reliability is crucial. In this study, we consider a medical diagnostic task using generated images by diffusion models, and propose a statistical test to quantify its reliability. The basic idea behind the proposed statistical test is to employ a selective inference framework, where we consider a statistical test conditional on the fact that the generated images are produced by a trained diffusion model. Using the proposed method, the statistical reliability of medical image diagnostic results can be quantified in the form of a p-value, allowing for decision-making with a controlled error rate. 
    
[^4]: DISCOUNT: 使用最优传输进行分布式对抗解释

    DISCOUNT: Distributional Counterfactual Explanation With Optimal Transport. (arXiv:2401.13112v1 [cs.AI])

    [http://arxiv.org/abs/2401.13112](http://arxiv.org/abs/2401.13112)

    本文提出了使用最优传输进行分布式对抗解释的方法DISCOUNT，将对抗解释的概念扩展到整个输入输出分布，并通过统计置信度来支撑这一方法。

    

    对抗解释是在黑盒决策模型中提供洞察力和可解释性的事实方法，通过确定导致不同结果的替代输入实例来实现。本文将对抗解释的概念扩展到分布上下文，从个体数据点扩大到整个输入输出分布，命名为分布式对抗解释。在分布式对抗解释中，我们的重点转向分析事实和对抗的分布属性，类似于评估个体实例及其结果决策的经典方法。我们利用最优传输来构建一个机会约束优化问题，旨在导出与事实对应的对抗分布，以统计置信度做支撑。我们提出的优化方法DISCOUNT在输入和输出分布之间平衡这种置信度。

    Counterfactual Explanations (CE) is the de facto method for providing insight and interpretability in black-box decision-making models by identifying alternative input instances that lead to different outcomes. This paper extends the concept of CEs to a distributional context, broadening the scope from individual data points to entire input and output distributions, named Distributional Counterfactual Explanation (DCE). In DCE, our focus shifts to analyzing the distributional properties of the factual and counterfactual, drawing parallels to the classical approach of assessing individual instances and their resulting decisions. We leverage Optimal Transport (OT) to frame a chance-constrained optimization problem, aiming to derive a counterfactual distribution that closely aligns with its factual counterpart, substantiated by statistical confidence. Our proposed optimization method, DISCOUNT, strategically balances this confidence across both input and output distributions. This algorit
    
[^5]: Spectrum-Aware Adjustment: 一种新的去偏方法框架及其在主成分回归中的应用

    Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression. (arXiv:2309.07810v1 [math.ST])

    [http://arxiv.org/abs/2309.07810](http://arxiv.org/abs/2309.07810)

    这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。

    

    我们引入了一个新的去偏方法框架，用于解决高维线性回归中现代去偏技术对协变量分布的约束问题。我们研究了特征数和样本数都很大且相近的普遍情况。在这种情况下，现代去偏技术使用自由度校正来除去正则化估计量的收缩偏差并进行推断。然而，该方法要求观测样本是独立同分布的，协变量遵循均值为零的高斯分布，并且能够获得可靠的特征协方差矩阵估计。当（i）协变量具有非高斯分布、重尾或非对称分布，（ii）设计矩阵的行呈异质性或存在依赖性，以及（iii）缺乏可靠的特征协方差估计时，这种方法就会遇到困难。为了应对这些问题，我们提出了一种新的策略，其中去偏校正是一步缩放的梯度下降步骤（适当缩放）。

    We introduce a new debiasing framework for high-dimensional linear regression that bypasses the restrictions on covariate distributions imposed by modern debiasing technology. We study the prevalent setting where the number of features and samples are both large and comparable. In this context, state-of-the-art debiasing technology uses a degrees-of-freedom correction to remove shrinkage bias of regularized estimators and conduct inference. However, this method requires that the observed samples are i.i.d., the covariates follow a mean zero Gaussian distribution, and reliable covariance matrix estimates for observed features are available. This approach struggles when (i) covariates are non-Gaussian with heavy tails or asymmetric distributions, (ii) rows of the design exhibit heterogeneity or dependencies, and (iii) reliable feature covariance estimates are lacking.  To address these, we develop a new strategy where the debiasing correction is a rescaled gradient descent step (suitably
    
[^6]: 用于电力负荷预测的基准和自定义包

    Benchmarks and Custom Package for Electrical Load Forecasting. (arXiv:2307.07191v1 [cs.LG])

    [http://arxiv.org/abs/2307.07191](http://arxiv.org/abs/2307.07191)

    本文提供了一个全面的电力负荷预测存档，包括负荷领域特定的特征工程，帮助模型更好地模拟负荷数据，并提供了一种新的损失函数来最小化后续任务的成本。

    

    负荷预测在电力行业中具有重要意义，可以为后续任务如电网调度提供参考，从而带来巨大的经济效益。然而，负荷预测与传统的时间序列预测之间存在许多差异。一方面，负荷预测的目标是最小化后续任务（如电网调度）的成本，而不仅仅追求预测准确性。另一方面，负荷受到许多外部因素的影响，如温度或日历变量。此外，预测的规模（如建筑级负荷和聚合级负荷）也会对预测结果产生重大影响。在本文中，我们提供了一个全面的负荷预测存档，其中包括负荷领域特定的特征工程，以帮助预测模型更好地模拟负荷数据。此外，与传统的损失函数仅追求准确性不同，我们还提供了一种方法来...

    Load forecasting is of great significance in the power industry as it can provide a reference for subsequent tasks such as power grid dispatch, thus bringing huge economic benefits. However, there are many differences between load forecasting and traditional time series forecasting. On the one hand, load forecasting aims to minimize the cost of subsequent tasks such as power grid dispatch, rather than simply pursuing prediction accuracy. On the other hand, the load is largely influenced by many external factors, such as temperature or calendar variables. In addition, the scale of predictions (such as building-level loads and aggregated-level loads) can also significantly impact the predicted results. In this paper, we provide a comprehensive load forecasting archive, which includes load domain-specific feature engineering to help forecasting models better model load data. In addition, different from the traditional loss function which only aims for accuracy, we also provide a method to
    

