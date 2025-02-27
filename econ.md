# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Seemingly unrelated Bayesian additive regression trees for cost-effectiveness analyses in healthcare](https://arxiv.org/abs/2404.02228) | 提出了适用于医疗保健成本效益分析的多元贝叶斯可加回归树扩展，克服了现有模型的局限性，可以处理多个相关结果变量的回归和分类分析 |
| [^2] | [Multiplayer War of Attrition with Asymmetric Private Information.](http://arxiv.org/abs/2302.09427) | 本文分析了对称类型分布下的多人消耗战博弈，在唯一均衡中发现不对称性导致参与者分层行为模式。参与者的耐心度、提供成本和估值会影响物品提供的速度。通过控制最强类型可以调节不对称性的效果。 |
| [^3] | [Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends!.](http://arxiv.org/abs/2206.04902) | 本文介绍了一种半全球框架，用于改进贝叶斯VAR模型的预测性能。该框架替代了传统的全局缩减参数，使用组别特定的缩减参数。通过广泛的模拟研究和实证应用，展示了该框架的优点。在稀疏/密集先验下，预测性能因评估的经济变量和时间框架而异，但动态模型平均法可以缓解这个问题。 |

# 详细

[^1]: 基于贝叶斯可加回归树的医疗保健成本效益分析

    Seemingly unrelated Bayesian additive regression trees for cost-effectiveness analyses in healthcare

    [https://arxiv.org/abs/2404.02228](https://arxiv.org/abs/2404.02228)

    提出了适用于医疗保健成本效益分析的多元贝叶斯可加回归树扩展，克服了现有模型的局限性，可以处理多个相关结果变量的回归和分类分析

    

    近年来的理论结果和模拟证据表明，贝叶斯可加回归树是一种非常有效的非参数回归方法。受到在卫生经济学中的成本效益分析的启发，我们提出了适用于具有多个相关结果变量的回归和分类分析的BART的多元扩展。我们的框架通过允许每个个体响应与不同树组相关联，同时处理结果之间的依赖关系，克服了现有多元BART模型的一些主要局限性。在连续结果的情况下，我们的模型本质上是表面无关回归的非参数版本。同样，我们针对二元结果的建议是非参数概括

    arXiv:2404.02228v1 Announce Type: cross  Abstract: In recent years, theoretical results and simulation evidence have shown Bayesian additive regression trees to be a highly-effective method for nonparametric regression. Motivated by cost-effectiveness analyses in health economics, where interest lies in jointly modelling the costs of healthcare treatments and the associated health-related quality of life experienced by a patient, we propose a multivariate extension of BART applicable in regression and classification analyses with several correlated outcome variables. Our framework overcomes some key limitations of existing multivariate BART models by allowing each individual response to be associated with different ensembles of trees, while still handling dependencies between the outcomes. In the case of continuous outcomes, our model is essentially a nonparametric version of seemingly unrelated regression. Likewise, our proposal for binary outcomes is a nonparametric generalisation of
    
[^2]: 对称私人信息下的多人消耗战分析

    Multiplayer War of Attrition with Asymmetric Private Information. (arXiv:2302.09427v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2302.09427](http://arxiv.org/abs/2302.09427)

    本文分析了对称类型分布下的多人消耗战博弈，在唯一均衡中发现不对称性导致参与者分层行为模式。参与者的耐心度、提供成本和估值会影响物品提供的速度。通过控制最强类型可以调节不对称性的效果。

    

    本文分析了对称类型分布下的多人消耗战博弈，研究了在提供不可分割的公共物品的私人机制中的对称性。在唯一均衡中，不对称性导致了一种分层行为模式，其中一个参与者以一定概率立即提供物品，而其他参与者在某个特定时刻之前没有提供的概率，并且这个时刻对不同的参与者是独特的。比较静态分析表明，耐心度更低、提供成本更低、估值更高的参与者能够更快地提供物品。延迟成本主要由立即退出参与者的最强类型决定。本文还研究了两种不对称性的变化：提升最强类型倾向于提高效率，而控制最强类型与对称性成本的符号相关。

    This paper analyzes a multiplayer war of attrition game with asymmetric type distributions in the setting of private provision of an indivisible public good. In the unique equilibrium, asymmetry leads to a stratified behavior pattern where one agent provides the good instantly with a positive probability, while each of the others has no probability of provision before a certain moment and this moment is idiosyncratic for different agents. Comparative statics show that an agent with less patience, lower cost of provision, and higher reputation in valuation provides the good uniformly faster. The cost of delay is mainly determined by the strongest type of the instant-exit agent. This paper investigates two types of variations of asymmetry: raising the strongest type tends to improve efficiency, whereas controlling the strongest type associates the effect of asymmetry with the sign of an intuitive measure of ``the cost of symmetry".
    
[^3]: 用贝叶斯VAR模型预测宏观经济数据：稀疏还是密集？要看情况！

    Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends!. (arXiv:2206.04902v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2206.04902](http://arxiv.org/abs/2206.04902)

    本文介绍了一种半全球框架，用于改进贝叶斯VAR模型的预测性能。该框架替代了传统的全局缩减参数，使用组别特定的缩减参数。通过广泛的模拟研究和实证应用，展示了该框架的优点。在稀疏/密集先验下，预测性能因评估的经济变量和时间框架而异，但动态模型平均法可以缓解这个问题。

    

    在建模和预测宏观经济变量时，向量自回归模型（VARs）被广泛应用。然而，在高维情况下，它们容易出现过拟合问题。贝叶斯方法，具体而言是缩减先验方法，已经显示出在提高预测性能方面取得了成功。在本文中，我们引入了半全球框架，其中我们用特定组别的缩减参数替代了传统的全局缩减参数。我们展示了如何将此框架应用于各种缩减先验，如全局-局部先验和随机搜索变量选择先验。我们通过广泛的模拟研究和对美国经济数据进行的实证应用，展示了所提出的框架的优点。此外，我们对正在进行的"稀疏假象"辩论进行了更深入的探讨，发现在稀疏/密集先验下的预测性能在评估的经济变量和时间框架中变化很大。然而，动态模型平均法可以缓解这个问题。

    Vectorautogressions (VARs) are widely applied when it comes to modeling and forecasting macroeconomic variables. In high dimensions, however, they are prone to overfitting. Bayesian methods, more concretely shrinking priors, have shown to be successful in improving prediction performance. In the present paper, we introduce the semi-global framework, in which we replace the traditional global shrinkage parameter with group-specific shrinkage parameters. We show how this framework can be applied to various shrinking priors, such as global-local priors and stochastic search variable selection priors. We demonstrate the virtues of the proposed framework in an extensive simulation study and in an empirical application forecasting data of the US economy. Further, we shed more light on the ongoing ``Illusion of Sparsity'' debate, finding that forecasting performances under sparse/dense priors vary across evaluated economic variables and across time frames. Dynamic model averaging, however, ca
    

