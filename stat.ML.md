# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-dimensional analysis of ridge regression for non-identically distributed data with a variance profile](https://arxiv.org/abs/2403.20200) | 研究了对于独立但非独立同分布数据的高维回归模型，提出了在岭正则化参数趋近于零时高维回归中的双谷现象。 |
| [^2] | [Auditing Fairness under Unobserved Confounding](https://arxiv.org/abs/2403.14713) | 在未观测混杂因素的情况下，本文展示了即使在放宽或甚至在排除所有相关风险因素被观测到的假设的情况下，仍然可以给出对高风险个体分配率的信息丰富的界限。 |
| [^3] | [A Theoretical Analysis of Nash Learning from Human Feedback under General KL-Regularized Preference](https://arxiv.org/abs/2402.07314) | 本论文从理论层面分析了一种关于一般偏好下纳什学习从人类反馈中的方法，通过对两个竞争的LLM进行博弈来找到一种一致生成响应的策略。 |
| [^4] | [Dropout Regularization Versus $\ell_2$-Penalization in the Linear Model.](http://arxiv.org/abs/2306.10529) | 研究发现，线性回归模型中采用dropout技术的统计行为具有更加微妙的与$\ell_2$正则化的联系，dropout并不像预期中那样具有稳定的正则化效果。 |
| [^5] | [Interpolation property of shallow neural networks.](http://arxiv.org/abs/2304.10552) | 本文证明了浅层神经网络可以插值任何数据集，即损失函数具有全局最小值为零的性质，此外还给出了该全局最小值处的惯性矩阵的表征，并提供了一种实用的概率方法来寻找插值点。 |
| [^6] | [Bagging Provides Assumption-free Stability.](http://arxiv.org/abs/2301.12600) | 本文证明了Bagging技术可提供无偏差稳定性，适用于各种数据分布和算法，具有良好的实证效果。 |
| [^7] | [A Recipe for Well-behaved Graph Neural Approximations of Complex Dynamics.](http://arxiv.org/abs/2301.04900) | 本文介绍了一种行为良好的图神经网络近似复杂动力学的方法，包括必要的偏置和适当的神经网络结构，并提出了评估泛化能力和推断时预测置信度的方法。 |
| [^8] | [Exceedance Probability Forecasting via Regression for Significant Wave Height Prediction.](http://arxiv.org/abs/2206.09821) | 本论文提出了一种基于回归的超标概率预测方法，用于预测显著波高，通过利用预测来估计超标概率，取得了更好的效果。 |

# 详细

[^1]: 对具有方差轮廓的非独立同分布数据的岭回归进行高维分析

    High-dimensional analysis of ridge regression for non-identically distributed data with a variance profile

    [https://arxiv.org/abs/2403.20200](https://arxiv.org/abs/2403.20200)

    研究了对于独立但非独立同分布数据的高维回归模型，提出了在岭正则化参数趋近于零时高维回归中的双谷现象。

    

    针对独立但非独立同分布数据，我们提出研究高维回归模型。假设观测到的预测变量集合是带有方差轮廓的随机矩阵，并且其维度以相应速率增长。在假设随机效应模型的情况下，我们研究了具有这种方差轮廓的岭估计器的线性回归的预测风险。在这种设置下，我们提供了该风险的确定性等价物以及岭估计器的自由度。对于某些方差轮廓类别，我们的工作突出了在岭正则化参数趋于零时，高维回归中的最小模最小二乘估计器出现双谷现象。我们还展示了一些方差轮廓f...

    arXiv:2403.20200v1 Announce Type: cross  Abstract: High-dimensional linear regression has been thoroughly studied in the context of independent and identically distributed data. We propose to investigate high-dimensional regression models for independent but non-identically distributed data. To this end, we suppose that the set of observed predictors (or features) is a random matrix with a variance profile and with dimensions growing at a proportional rate. Assuming a random effect model, we study the predictive risk of the ridge estimator for linear regression with such a variance profile. In this setting, we provide deterministic equivalents of this risk and of the degree of freedom of the ridge estimator. For certain class of variance profile, our work highlights the emergence of the well-known double descent phenomenon in high-dimensional regression for the minimum norm least-squares estimator when the ridge regularization parameter goes to zero. We also exhibit variance profiles f
    
[^2]: 在未观测混杂因素下审计公平性

    Auditing Fairness under Unobserved Confounding

    [https://arxiv.org/abs/2403.14713](https://arxiv.org/abs/2403.14713)

    在未观测混杂因素的情况下，本文展示了即使在放宽或甚至在排除所有相关风险因素被观测到的假设的情况下，仍然可以给出对高风险个体分配率的信息丰富的界限。

    

    决策系统中的一个基本问题是跨越人口统计线存在不公平性。然而，不公平性可能难以量化，特别是如果我们对公平性的理解依赖于难以衡量的风险等观念（例如，对于那些没有其治疗就会死亡的人平等获得治疗）。审计这种不公平性需要准确测量个体风险，而在未观测混杂的现实环境中，难以估计。在这些未观测到的因素“解释”明显差异的情况下，我们可能低估或高估不公平性。在本文中，我们展示了即使在放宽或（令人惊讶地）甚至在排除所有相关风险因素被观测到的假设的情况下，仍然可以对高风险个体的分配率给出信息丰富的界限。我们利用了在许多实际环境中（例如引入新型治疗）我们拥有在任何分配之前的数据的事实。

    arXiv:2403.14713v1 Announce Type: cross  Abstract: A fundamental problem in decision-making systems is the presence of inequity across demographic lines. However, inequity can be difficult to quantify, particularly if our notion of equity relies on hard-to-measure notions like risk (e.g., equal access to treatment for those who would die without it). Auditing such inequity requires accurate measurements of individual risk, which is difficult to estimate in the realistic setting of unobserved confounding. In the case that these unobservables "explain" an apparent disparity, we may understate or overstate inequity. In this paper, we show that one can still give informative bounds on allocation rates among high-risk individuals, even while relaxing or (surprisingly) even when eliminating the assumption that all relevant risk factors are observed. We utilize the fact that in many real-world settings (e.g., the introduction of a novel treatment) we have data from a period prior to any alloc
    
[^3]: 一种关于一般KL正则化偏好下纳什学习从人类反馈中的理论分析

    A Theoretical Analysis of Nash Learning from Human Feedback under General KL-Regularized Preference

    [https://arxiv.org/abs/2402.07314](https://arxiv.org/abs/2402.07314)

    本论文从理论层面分析了一种关于一般偏好下纳什学习从人类反馈中的方法，通过对两个竞争的LLM进行博弈来找到一种一致生成响应的策略。

    

    来自人类反馈的强化学习（RLHF）从一个概率偏好模型提供的偏好信号中学习，该模型以一个提示和两个响应作为输入，并产生一个分数，表示对一个响应相对于另一个响应的偏好程度。迄今为止，最流行的RLHF范式是基于奖励的，它从奖励建模的初始步骤开始，然后使用构建的奖励为后续的奖励优化阶段提供奖励信号。然而，奖励函数的存在是一个强假设，基于奖励的RLHF在表达能力上有局限性，不能捕捉到真实世界中复杂的人类偏好。在这项工作中，我们为最近提出的学习范式Nash学习从人类反馈（NLHF）提供了理论洞察力，该学习范式考虑了一个一般的偏好模型，并将对齐过程定义为两个竞争的LLM之间的博弈。学习目标是找到一个一致生成响应的策略。

    Reinforcement Learning from Human Feedback (RLHF) learns from the preference signal provided by a probabilistic preference model, which takes a prompt and two responses as input, and produces a score indicating the preference of one response against another. So far, the most popular RLHF paradigm is reward-based, which starts with an initial step of reward modeling, and the constructed reward is then used to provide a reward signal for the subsequent reward optimization stage. However, the existence of a reward function is a strong assumption and the reward-based RLHF is limited in expressivity and cannot capture the real-world complicated human preference.   In this work, we provide theoretical insights for a recently proposed learning paradigm, Nash learning from human feedback (NLHF), which considered a general preference model and formulated the alignment process as a game between two competitive LLMs. The learning objective is to find a policy that consistently generates responses
    
[^4]: 线性模型中的Dropout正则化与$\ell_2$-Penalization比较

    Dropout Regularization Versus $\ell_2$-Penalization in the Linear Model. (arXiv:2306.10529v1 [math.ST])

    [http://arxiv.org/abs/2306.10529](http://arxiv.org/abs/2306.10529)

    研究发现，线性回归模型中采用dropout技术的统计行为具有更加微妙的与$\ell_2$正则化的联系，dropout并不像预期中那样具有稳定的正则化效果。

    

    本研究探讨了线性回归模型中采用dropout的梯度下降算法的统计行为。具体而言，推导了迭代的期望和协方差矩阵的非渐近性界限。与文献中广泛引用的dropout与$\ell_2$正则化的期望联系不同的是，结果表明了由于梯度下降动态与dropout引入的附加随机性之间的相互作用，两者之间存在着更加微妙的关系。我们还研究了一种简化版的dropout，它不具有正则化作用，并收敛于最小平方估计器。

    We investigate the statistical behavior of gradient descent iterates with dropout in the linear regression model. In particular, non-asymptotic bounds for expectations and covariance matrices of the iterates are derived. In contrast with the widely cited connection between dropout and $\ell_2$-regularization in expectation, the results indicate a much more subtle relationship, owing to interactions between the gradient descent dynamics and the additional randomness induced by dropout. We also study a simplified variant of dropout which does not have a regularizing effect and converges to the least squares estimator.
    
[^5]: 浅层神经网络的插值性质

    Interpolation property of shallow neural networks. (arXiv:2304.10552v1 [cs.LG])

    [http://arxiv.org/abs/2304.10552](http://arxiv.org/abs/2304.10552)

    本文证明了浅层神经网络可以插值任何数据集，即损失函数具有全局最小值为零的性质，此外还给出了该全局最小值处的惯性矩阵的表征，并提供了一种实用的概率方法来寻找插值点。

    

    我们研究了超参数化神经网络的损失函数全局最小值的几何性质。在大多数优化问题中，损失函数是凸函数，这种情况下我们只有一个全局最小值，或者是非凸函数，在这种情况下我们有一个有限的全局最小值。在本文中，我们证明了在超参数化范围内，对于非小次数多项式的激活函数，浅层神经网络可以插值任何数据集，即损失函数具有全局最小值为零的性质。此外，如果存在这样的全局最小值，则全局最小值的轮廓有无穷多个点。此外，我们给出了在全局最小值处求解损失函数的海塞矩阵的表征，并在最后一节中，我们提供了一种实用的概率方法来寻找插值点。

    We study the geometry of global minima of the loss landscape of overparametrized neural networks. In most optimization problems, the loss function is convex, in which case we only have a global minima, or nonconvex, with a discrete number of global minima. In this paper, we prove that in the overparametrized regime, a shallow neural network can interpolate any data set, i.e. the loss function has a global minimum value equal to zero as long as the activation function is not a polynomial of small degree. Additionally, if such a global minimum exists, then the locus of global minima has infinitely many points. Furthermore, we give a characterization of the Hessian of the loss function evaluated at the global minima, and in the last section, we provide a practical probabilistic method of finding the interpolation point.
    
[^6]: Bagging提供无偏差稳定性。

    Bagging Provides Assumption-free Stability. (arXiv:2301.12600v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.12600](http://arxiv.org/abs/2301.12600)

    本文证明了Bagging技术可提供无偏差稳定性，适用于各种数据分布和算法，具有良好的实证效果。

    

    Bagging是稳定机器学习模型的一个重要技术。在本文中，我们针对任何模型的稳定性推导了一个有限样本保证。我们的结果不对数据分布、基本算法的属性或协变量的维数进行任何假设。我们的保证适用于多种变体的Bagging，并且是最优的常数。实证结果验证了我们的发现，表明Bagging成功稳定了即使是高度不稳定的基本算法。

    Bagging is an important technique for stabilizing machine learning models. In this paper, we derive a finite-sample guarantee on the stability of bagging for any model. Our result places no assumptions on the distribution of the data, on the properties of the base algorithm, or on the dimensionality of the covariates. Our guarantee applies to many variants of bagging and is optimal up to a constant. Empirical results validate our findings, showing that bagging successfully stabilizes even highly unstable base algorithms.
    
[^7]: 一种行为良好的图神经近似复杂动力学的方法

    A Recipe for Well-behaved Graph Neural Approximations of Complex Dynamics. (arXiv:2301.04900v2 [cond-mat.stat-mech] UPDATED)

    [http://arxiv.org/abs/2301.04900](http://arxiv.org/abs/2301.04900)

    本文介绍了一种行为良好的图神经网络近似复杂动力学的方法，包括必要的偏置和适当的神经网络结构，并提出了评估泛化能力和推断时预测置信度的方法。

    

    数据驱动的常微分方程近似提供了一种有前景的方法来发现动力系统模型，特别是对于缺乏明确原理的复杂系统。本文着重研究了一类由网络邻接矩阵耦合的常微分方程系统描述的复杂系统。许多现实世界中的系统，包括金融、社交和神经系统，属于这类动力学模型。我们提出了使用神经网络近似这种动力系统的关键要素，包括必要的偏置和适当的神经网络结构。强调与静态监督学习的区别，我们提倡在统计学习理论的经典假设之外评估泛化能力。为了在推断时估计预测的置信度，我们引入了一个专用的空模型。通过研究各种复杂网络动力学，我们展示了神经网络的能力。

    Data-driven approximations of ordinary differential equations offer a promising alternative to classical methods in discovering a dynamical system model, particularly in complex systems lacking explicit first principles. This paper focuses on a complex system whose dynamics is described with a system of ordinary differential equations, coupled via a network adjacency matrix. Numerous real-world systems, including financial, social, and neural systems, belong to this class of dynamical models. We propose essential elements for approximating such dynamical systems using neural networks, including necessary biases and an appropriate neural architecture. Emphasizing the differences from static supervised learning, we advocate for evaluating generalization beyond classical assumptions of statistical learning theory. To estimate confidence in prediction during inference time, we introduce a dedicated null model. By studying various complex network dynamics, we demonstrate the neural network'
    
[^8]: 基于回归的超标概率预测方法用于显著波高预测

    Exceedance Probability Forecasting via Regression for Significant Wave Height Prediction. (arXiv:2206.09821v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2206.09821](http://arxiv.org/abs/2206.09821)

    本论文提出了一种基于回归的超标概率预测方法，用于预测显著波高，通过利用预测来估计超标概率，取得了更好的效果。

    

    显著波高预测是海洋数据分析中的一个关键问题。预测显著波高对于估计波能产生是至关重要的。此外，及时预测大浪的到来对于确保航海作业的安全很重要。我们将预测显著波高的极端值作为超标概率预测问题。因此，我们旨在估计显著波高将超过预定义阈值的概率。通常使用概率二分类模型来解决这个任务。相反，我们提出了一种基于预测模型的新方法。该方法利用未来观测的预测来根据累积分布函数估计超标概率。我们使用来自加拿大哈利法克斯海岸的浮标数据进行了实验。结果表明，所提出的方法更好。

    Significant wave height forecasting is a key problem in ocean data analytics. Predicting the significant wave height is crucial for estimating the energy production from waves. Moreover, the timely prediction of large waves is important to ensure the safety of maritime operations, e.g. passage of vessels. We frame the task of predicting extreme values of significant wave height as an exceedance probability forecasting problem. Accordingly, we aim at estimating the probability that the significant wave height will exceed a predefined threshold. This task is usually solved using a probabilistic binary classification model. Instead, we propose a novel approach based on a forecasting model. The method leverages the forecasts for the upcoming observations to estimate the exceedance probability according to the cumulative distribution function. We carried out experiments using data from a buoy placed in the coast of Halifax, Canada. The results suggest that the proposed methodology is better
    

