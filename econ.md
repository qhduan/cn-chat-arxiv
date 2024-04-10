# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fair integer programming under dichotomous preferences.](http://arxiv.org/abs/2306.13383) | 本文提出了一个统一的框架，通过最大化纳什乘积、最小选择概率或随机串行独裁等算法，在二元偏好条件下，公平地选择最优解，解决了使用整数线性规划做出真正公平决策的问题。 |
| [^2] | [Dynamic Ordered Panel Logit Models.](http://arxiv.org/abs/2107.03253) | 本文研究了固定效应面板数据的动态有序Logit模型，构建了一组不受固定效应约束的有效矩条件，并给出了矩条件识别模型公共参数的充分条件。通过广义矩法估计公共参数，并用蒙特卡罗模拟和实证研究验证了估计器的性能。 |

# 详细

[^1]: 二元偏好条件下公平整数规划

    Fair integer programming under dichotomous preferences. (arXiv:2306.13383v1 [cs.GT])

    [http://arxiv.org/abs/2306.13383](http://arxiv.org/abs/2306.13383)

    本文提出了一个统一的框架，通过最大化纳什乘积、最小选择概率或随机串行独裁等算法，在二元偏好条件下，公平地选择最优解，解决了使用整数线性规划做出真正公平决策的问题。

    

    除非控制所选最优解的选择概率，否则使用整数线性规划无法做出真正公平的决策。为此，我们提出了一个统一的框架，当二元决策变量代表具有二元偏好的代理时，他们只关心是否在最终解中被选中。我们开发了几种通用的算法来公平地选择最优解，例如通过最大化纳什乘积或最小选择概率，或使用代理人的随机排序作为选择标准（随机串行独裁）。因此，我们将解决整数线性规划的黑盒子程序嵌入了一个从头到尾都可以解释的框架中。此外，我们将我们的框架嵌入到合作谈判和概率社会选择的丰富文献中来研究所提出的方法的公理特性。最后，我们对特定应用中的提出的方法进行了评估。

    One cannot make truly fair decisions using integer linear programs unless one controls the selection probabilities of the (possibly many) optimal solutions. For this purpose, we propose a unified framework when binary decision variables represent agents with dichotomous preferences, who only care about whether they are selected in the final solution. We develop several general-purpose algorithms to fairly select optimal solutions, for example, by maximizing the Nash product or the minimum selection probability, or by using a random ordering of the agents as a selection criterion (Random Serial Dictatorship). As such, we embed the black-box procedure of solving an integer linear program into a framework that is explainable from start to finish. Moreover, we study the axiomatic properties of the proposed methods by embedding our framework into the rich literature of cooperative bargaining and probabilistic social choice. Lastly, we evaluate the proposed methods on a specific application,
    
[^2]: 动态有序面板Logit模型

    Dynamic Ordered Panel Logit Models. (arXiv:2107.03253v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2107.03253](http://arxiv.org/abs/2107.03253)

    本文研究了固定效应面板数据的动态有序Logit模型，构建了一组不受固定效应约束的有效矩条件，并给出了矩条件识别模型公共参数的充分条件。通过广义矩法估计公共参数，并用蒙特卡罗模拟和实证研究验证了估计器的性能。

    

    本文研究了一个用于固定效应面板数据的动态有序Logit模型。本文的主要贡献是构建了一组不受固定效应约束的有效矩条件。这些矩函数可以利用四个或更多期的数据计算，并且本文给出了矩条件能够识别模型的公共参数（回归系数、自回归参数和阈值参数）的充分条件。矩条件的可利用性表明可以利用广义矩法估计这些公共参数，本文通过蒙特卡罗模拟和利用英国家庭面板调查的自报健康状况进行经验说明，评估了该估计器的性能。

    This paper studies a dynamic ordered logit model for panel data with fixed effects. The main contribution of the paper is to construct a set of valid moment conditions that are free of the fixed effects. The moment functions can be computed using four or more periods of data, and the paper presents sufficient conditions for the moment conditions to identify the common parameters of the model, namely the regression coefficients, the autoregressive parameters, and the threshold parameters. The availability of moment conditions suggests that these common parameters can be estimated using the generalized method of moments, and the paper documents the performance of this estimator using Monte Carlo simulations and an empirical illustration to self-reported health status using the British Household Panel Survey.
    

