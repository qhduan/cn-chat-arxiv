# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Systematic Approach to Robustness Modelling for Deep Convolutional Neural Networks.](http://arxiv.org/abs/2401.13751) | 本论文提出一种系统化方法，用于针对深度卷积神经网络进行鲁棒性建模。研究发现隐藏层数量对模型的推广性能有影响，同时还测试了模型大小、浮点精度、训练数据和模型输出的噪声水平等参数。为了改进模型的预测能力和计算成本，提出了一种使用诱发故障来建模故障概率的方法。 |
| [^2] | [The Computational Complexity of Finding Stationary Points in Non-Convex Optimization.](http://arxiv.org/abs/2310.09157) | 本文研究了在非凸优化中找到光滑目标函数的近似稳定点的计算复杂性和查询复杂性，并给出了相应的结果。对于$d=2$的情况，提供了一种零阶算法，只需要少量的函数值查询即可找到$\varepsilon$-近似稳定点。 |
| [^3] | [fmeffects: An R Package for Forward Marginal Effects.](http://arxiv.org/abs/2310.02008) | fmeffects是第一个实现前向边际效应（FMEs）的R软件包。 |
| [^4] | [Score diffusion models without early stopping: finite Fisher information is all you need.](http://arxiv.org/abs/2308.12240) | 无早停的分数扩散模型不需要得分函数的Lipschitz均匀条件，只需要有限的费舍尔信息。 |
| [^5] | [Identifiable causal inference with noisy treatment and no side information.](http://arxiv.org/abs/2306.10614) | 本论文提出了一种在没有侧面信息和具有复杂非线性依赖性的情况下，纠正因治疗变量不准确测量引起的因果效应估计偏差的模型，并证明了该模型的因果效应估计是可识别的。该方法使用了深度潜在变量模型和分摊权重变分客观函数进行训练。 |
| [^6] | [Identification and multiply robust estimation in causal mediation analysis with treatment noncompliance.](http://arxiv.org/abs/2304.10025) | 本文针对治疗不服从性提出了一种半参数框架来评估因果中介效应，提出了一组假设来识别自然中介效应并推导出成倍稳健估计器。 |

# 详细

[^1]: 一种针对深度卷积神经网络的鲁棒性建模的系统化方法

    A Systematic Approach to Robustness Modelling for Deep Convolutional Neural Networks. (arXiv:2401.13751v1 [cs.LG])

    [http://arxiv.org/abs/2401.13751](http://arxiv.org/abs/2401.13751)

    本论文提出一种系统化方法，用于针对深度卷积神经网络进行鲁棒性建模。研究发现隐藏层数量对模型的推广性能有影响，同时还测试了模型大小、浮点精度、训练数据和模型输出的噪声水平等参数。为了改进模型的预测能力和计算成本，提出了一种使用诱发故障来建模故障概率的方法。

    

    当有大量标记数据可用时，卷积神经网络已经被证明在许多领域都可以广泛应用。最近的趋势是使用具有越来越多可调参数的模型，以提高模型准确性，降低模型损失或创建更具对抗鲁棒性的模型，而这些目标通常相互矛盾。特别是，最近的理论研究提出了对更大模型能否推广到受控的训练和测试集之外的数据的疑问。因此，我们研究了ResNet模型中隐藏层的数量在MNIST、CIFAR10和CIFAR100数据集上的作用。我们测试了各种参数，包括模型的大小、浮点精度，以及训练数据和模型输出的噪声水平。为了改进模型的预测能力和计算成本，我们提供了一种使用诱发故障来建模故障概率的方法。

    Convolutional neural networks have shown to be widely applicable to a large number of fields when large amounts of labelled data are available. The recent trend has been to use models with increasingly larger sets of tunable parameters to increase model accuracy, reduce model loss, or create more adversarially robust models -- goals that are often at odds with one another. In particular, recent theoretical work raises questions about the ability for even larger models to generalize to data outside of the controlled train and test sets. As such, we examine the role of the number of hidden layers in the ResNet model, demonstrated on the MNIST, CIFAR10, CIFAR100 datasets. We test a variety of parameters including the size of the model, the floating point precision, and the noise level of both the training data and the model output. To encapsulate the model's predictive power and computational cost, we provide a method that uses induced failures to model the probability of failure as a fun
    
[^2]: 寻找非凸优化中的稳定点的计算复杂性

    The Computational Complexity of Finding Stationary Points in Non-Convex Optimization. (arXiv:2310.09157v1 [math.OC])

    [http://arxiv.org/abs/2310.09157](http://arxiv.org/abs/2310.09157)

    本文研究了在非凸优化中找到光滑目标函数的近似稳定点的计算复杂性和查询复杂性，并给出了相应的结果。对于$d=2$的情况，提供了一种零阶算法，只需要少量的函数值查询即可找到$\varepsilon$-近似稳定点。

    

    寻找非凸但光滑目标函数$f$在无限制的$d$维域上的近似稳定点，即梯度近似为零的点，是经典非凸优化中最基本的问题之一。然而，当问题的维度$d$与近似误差独立时，这个问题的计算复杂性和查询复杂性仍不十分清楚。在本文中，我们展示了以下计算复杂性和查询复杂性结果：1.在无限制的域中寻找近似稳定点的问题是PLS完全问题。2.对于$d=2$，我们提供了一种零阶算法，用于寻找$\varepsilon$-近似稳定点，只需要对目标函数进行最多$O(1/\varepsilon)$次函数值查询。3.我们证明当$d=2$时，任何算法至少需要$\Omega(1/\varepsilon)$次对目标函数和/或梯度的查询来找到$\varepsilon$-近似稳定点。

    Finding approximate stationary points, i.e., points where the gradient is approximately zero, of non-convex but smooth objective functions $f$ over unrestricted $d$-dimensional domains is one of the most fundamental problems in classical non-convex optimization. Nevertheless, the computational and query complexity of this problem are still not well understood when the dimension $d$ of the problem is independent of the approximation error. In this paper, we show the following computational and query complexity results:  1. The problem of finding approximate stationary points over unrestricted domains is PLS-complete.  2. For $d = 2$, we provide a zero-order algorithm for finding $\varepsilon$-approximate stationary points that requires at most $O(1/\varepsilon)$ value queries to the objective function.  3. We show that any algorithm needs at least $\Omega(1/\varepsilon)$ queries to the objective function and/or its gradient to find $\varepsilon$-approximate stationary points when $d=2$.
    
[^3]: fmeffects: 一个用于前向边际效应的R软件包

    fmeffects: An R Package for Forward Marginal Effects. (arXiv:2310.02008v1 [cs.LG])

    [http://arxiv.org/abs/2310.02008](http://arxiv.org/abs/2310.02008)

    fmeffects是第一个实现前向边际效应（FMEs）的R软件包。

    

    前向边际效应（FMEs）作为一种通用有效的模型不可知解释方法最近被引入。它们以“如果我们将$x$改变$h$，那么预测结果$\widehat{y}$会发生什么变化？”的形式提供易于理解和可操作的模型解释。本文介绍了fmeffects软件包，这是FMEs的第一个软件实现。讨论了相关的理论背景、软件包功能和处理方式，以及软件设计和未来扩展的选项。

    Forward marginal effects (FMEs) have recently been introduced as a versatile and effective model-agnostic interpretation method. They provide comprehensible and actionable model explanations in the form of: If we change $x$ by an amount $h$, what is the change in predicted outcome $\widehat{y}$? We present the R package fmeffects, the first software implementation of FMEs. The relevant theoretical background, package functionality and handling, as well as the software design and options for future extensions are discussed in this paper.
    
[^4]: 无早停的分数扩散模型：有限费舍尔信息就足够了

    Score diffusion models without early stopping: finite Fisher information is all you need. (arXiv:2308.12240v1 [math.ST])

    [http://arxiv.org/abs/2308.12240](http://arxiv.org/abs/2308.12240)

    无早停的分数扩散模型不需要得分函数的Lipschitz均匀条件，只需要有限的费舍尔信息。

    

    分数扩散模型是一种围绕着与随机微分方程相关的得分函数估计的生成模型。在获得近似的得分函数之后，利用它来模拟相应的时间逆过程，最终实现近似数据样本的生成。尽管这些模型具有显著的实际意义，但在涉及非常规得分和估计器的情况下，仍存在一个显著的挑战，即缺乏全面的定量结果。在几乎所有的Kullback Leibler散度的相关结果中，都假设得分函数或其近似在时间上是Lipschitz均匀的。然而，在实践中，这个条件非常严格，或者很难建立。为了解决这个问题，先前的研究主要是关注分数扩散模型的早停版本在KL散度上的收敛界限，并且...

    Diffusion models are a new class of generative models that revolve around the estimation of the score function associated with a stochastic differential equation. Subsequent to its acquisition, the approximated score function is then harnessed to simulate the corresponding time-reversal process, ultimately enabling the generation of approximate data samples. Despite their evident practical significance these models carry, a notable challenge persists in the form of a lack of comprehensive quantitative results, especially in scenarios involving non-regular scores and estimators. In almost all reported bounds in Kullback Leibler (KL) divergence, it is assumed that either the score function or its approximation is Lipschitz uniformly in time. However, this condition is very restrictive in practice or appears to be difficult to establish.  To circumvent this issue, previous works mainly focused on establishing convergence bounds in KL for an early stopped version of the diffusion model and
    
[^5]: 带有嘈杂治疗和没有侧面信息的可识别因果推断

    Identifiable causal inference with noisy treatment and no side information. (arXiv:2306.10614v1 [cs.LG])

    [http://arxiv.org/abs/2306.10614](http://arxiv.org/abs/2306.10614)

    本论文提出了一种在没有侧面信息和具有复杂非线性依赖性的情况下，纠正因治疗变量不准确测量引起的因果效应估计偏差的模型，并证明了该模型的因果效应估计是可识别的。该方法使用了深度潜在变量模型和分摊权重变分客观函数进行训练。

    

    在某些因果推断场景中，治疗（即原因）变量的测量存在不准确性，例如在流行病学或计量经济学中。未能纠正测量误差的影响可能导致偏差的因果效应估计。以前的研究没有从因果视角研究解决这个问题的方法，同时允许复杂的非线性依赖关系并且不假设可以访问侧面信息。对于这样的场景，本论文提出了一个模型，它假设存在一个连续的治疗变量，该变量测量不准确。建立在现有测量误差模型的基础上，我们证明了我们的模型的因果效应估计是可识别的，即使没有测量误差方差或其他侧面信息的知识。我们的方法依赖于深度潜在变量模型，其中高斯条件由神经网络参数化，并且我们开发了一个分摊权重变分客观函数来训练该模型。

    In some causal inference scenarios, the treatment (i.e. cause) variable is measured inaccurately, for instance in epidemiology or econometrics. Failure to correct for the effect of this measurement error can lead to biased causal effect estimates. Previous research has not studied methods that address this issue from a causal viewpoint while allowing for complex nonlinear dependencies and without assuming access to side information. For such as scenario, this paper proposes a model that assumes a continuous treatment variable which is inaccurately measured. Building on existing results for measurement error models, we prove that our model's causal effect estimates are identifiable, even without knowledge of the measurement error variance or other side information. Our method relies on a deep latent variable model where Gaussian conditionals are parameterized by neural networks, and we develop an amortized importance-weighted variational objective for training the model. Empirical resul
    
[^6]: 用于因果中介分析中具有治疗不服从性的识别和倍增稳健估计

    Identification and multiply robust estimation in causal mediation analysis with treatment noncompliance. (arXiv:2304.10025v1 [stat.ME])

    [http://arxiv.org/abs/2304.10025](http://arxiv.org/abs/2304.10025)

    本文针对治疗不服从性提出了一种半参数框架来评估因果中介效应，提出了一组假设来识别自然中介效应并推导出成倍稳健估计器。

    

    在实验和观察研究中，人们通常对了解干预方案如何改善最终结果的潜在机制感兴趣。因果中介分析旨在达到此目的，但主要限于治疗完全服从的情况，只有少数情况需要排除限制。在本文中，我们建立了一个半参数框架，用于在无需排除限制的情况下评估具有治疗不服从性的因果中介效应。我们提出了一组假设来识别整个研究人群的自然中介效应，并进一步针对由潜在服从行为特征化的亚人群中的主要自然中介效应进行识别。我们推导出了主要自然中介效应估计量的有效影响函数，这激励了一组倍增稳健估计器进行推论。这些被识别估计量的半参数效率理论。

    In experimental and observational studies, there is often interest in understanding the potential mechanism by which an intervention program improves the final outcome. Causal mediation analyses have been developed for this purpose but are primarily restricted to the case of perfect treatment compliance, with a few exceptions that require exclusion restriction. In this article, we establish a semiparametric framework for assessing causal mediation in the presence of treatment noncompliance without exclusion restriction. We propose a set of assumptions to identify the natural mediation effects for the entire study population and further, for the principal natural mediation effects within subpopulations characterized by the potential compliance behaviour. We derive the efficient influence functions for the principal natural mediation effect estimands, which motivate a set of multiply robust estimators for inference. The semiparametric efficiency theory for the identified estimands is der
    

