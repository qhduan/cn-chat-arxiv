# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unveiling the Potential of Robustness in Evaluating Causal Inference Models](https://arxiv.org/abs/2402.18392) | 介绍了一种新颖的分布式健壮度量（DRM）方法，以解决选择理想因果推断模型中健壮估计器的挑战。 |
| [^2] | [Persuading a Learning Agent](https://arxiv.org/abs/2402.09721) | 在一个重复的贝叶斯说服问题中，即使没有承诺能力，委托人可以通过使用上下文无遗憾学习算法来实现与经典无学习模型中具有承诺的委托人的最优效用无限接近的效果；在代理人使用上下文无交换遗憾学习算法的情况下，委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。 |

# 详细

[^1]: 揭示健壮性在评估因果推断模型中的潜力

    Unveiling the Potential of Robustness in Evaluating Causal Inference Models

    [https://arxiv.org/abs/2402.18392](https://arxiv.org/abs/2402.18392)

    介绍了一种新颖的分布式健壮度量（DRM）方法，以解决选择理想因果推断模型中健壮估计器的挑战。

    

    越来越多对个性化决策制定的需求导致人们对估计条件平均处理效应（CATE）产生了兴趣。机器学习和因果推断的交叉领域已经产生了各种有效的CATE估计器。然而，在实践中使用这些估计器通常受制于缺乏反事实标签，因此使用传统的交叉验证等模型选择程序来选择理想的CATE估计器变得具有挑战性。现有的CATE估计器选择方法，如插值和伪结果度量，面临着两个固有挑战。首先，它们需要确定度量形式和拟合干扰参数或插件学习者的基础机器学习模型。其次，它们缺乏针对选择健壮估计器的特定重点。为解决这些挑战，本文引入了一种新颖的方法，分布式健壮度量（DRM）。

    arXiv:2402.18392v1 Announce Type: cross  Abstract: The growing demand for personalized decision-making has led to a surge of interest in estimating the Conditional Average Treatment Effect (CATE). The intersection of machine learning and causal inference has yielded various effective CATE estimators. However, deploying these estimators in practice is often hindered by the absence of counterfactual labels, making it challenging to select the desirable CATE estimator using conventional model selection procedures like cross-validation. Existing approaches for CATE estimator selection, such as plug-in and pseudo-outcome metrics, face two inherent challenges. Firstly, they are required to determine the metric form and the underlying machine learning models for fitting nuisance parameters or plug-in learners. Secondly, they lack a specific focus on selecting a robust estimator. To address these challenges, this paper introduces a novel approach, the Distributionally Robust Metric (DRM), for 
    
[^2]: 说服一位学习代理

    Persuading a Learning Agent

    [https://arxiv.org/abs/2402.09721](https://arxiv.org/abs/2402.09721)

    在一个重复的贝叶斯说服问题中，即使没有承诺能力，委托人可以通过使用上下文无遗憾学习算法来实现与经典无学习模型中具有承诺的委托人的最优效用无限接近的效果；在代理人使用上下文无交换遗憾学习算法的情况下，委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。

    

    我们研究了一个重复的贝叶斯说服问题（更一般地，任何具有完全信息的广义委托-代理问题），其中委托人没有承诺能力，代理人使用算法来学习如何对委托人的信号做出响应。我们将这个问题简化为一个一次性的广义委托-代理问题，代理人近似地最佳响应。通过这个简化，我们可以证明：如果代理人使用上下文无遗憾学习算法，则委托人可以保证其效用与经典无学习模型中具有承诺的委托人的最优效用之间可以无限接近；如果代理人使用上下文无交换遗憾学习算法，则委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。委托人在学习模型与非学习模型中可以获得的效用之间的差距是有界的。

    arXiv:2402.09721v1 Announce Type: cross  Abstract: We study a repeated Bayesian persuasion problem (and more generally, any generalized principal-agent problem with complete information) where the principal does not have commitment power and the agent uses algorithms to learn to respond to the principal's signals. We reduce this problem to a one-shot generalized principal-agent problem with an approximately-best-responding agent. This reduction allows us to show that: if the agent uses contextual no-regret learning algorithms, then the principal can guarantee a utility that is arbitrarily close to the principal's optimal utility in the classic non-learning model with commitment; if the agent uses contextual no-swap-regret learning algorithms, then the principal cannot obtain any utility significantly more than the optimal utility in the non-learning model with commitment. The difference between the principal's obtainable utility in the learning model and the non-learning model is bound
    

