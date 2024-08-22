# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [From Time-inconsistency to Time-consistency for Optimal Stopping Problems](https://arxiv.org/abs/2404.02498) | 该论文研究了具有时间不一致偏好的最优停止问题，通过对时间不一致性水平的衡量，发现了概率失真导致时间不一致性的程度与转变为复杂策略所需的时间之间的关系。 |
| [^2] | [Bridging TSLS and JIVE.](http://arxiv.org/abs/2305.17615) | 本文提出了一种桥接TSLS和JIVE的新方法TSJI来处理内生性，具有用户定义参数λ，可以近似无偏，且在许多工具变量渐进情况下是一致且渐进正常的。 |

# 详细

[^1]: 从时间不一致到时间一致的最优停止问题

    From Time-inconsistency to Time-consistency for Optimal Stopping Problems

    [https://arxiv.org/abs/2404.02498](https://arxiv.org/abs/2404.02498)

    该论文研究了具有时间不一致偏好的最优停止问题，通过对时间不一致性水平的衡量，发现了概率失真导致时间不一致性的程度与转变为复杂策略所需的时间之间的关系。

    

    对于具有时间不一致偏好的最优停止问题，我们通过将从天真策略转变为复杂策略所需的时间来衡量时间不一致的固有水平。特别是，在重复实验中，当天真代理可以观察到她的实际行动序列与最初计划不一致时，她会根据后续实际行为的观察选择立即行动。该过程重复进行，直到她的实际行动序列在任何时间点都与她的计划一致。我们表明，对于累积概率理论的偏好值，其中时间不一致由于概率失真导致，概率失真程度越高，时间不一致程度就越严重，需要的时间转变从天真策略到复杂策略也更多。

    arXiv:2404.02498v1 Announce Type: new  Abstract: For optimal stopping problems with time-inconsistent preference, we measure the inherent level of time-inconsistency by taking the time needed to turn the naive strategies into the sophisticated ones. In particular, when in a repeated experiment the naive agent can observe her actual sequence of actions which are inconsistent with what she has planned at the initial time, she then chooses her immediate action based on the observations on her later actual behavior. The procedure is repeated until her actual sequence of actions are consistent with her plan at any time. We show that for the preference value of cumulative prospect theory, in which the time-inconsistency is due to the probability distortion, the higher the degree of probability distortion, the more severe the level of time-inconsistency, and the more time required to turn the naive strategies into the sophisticated ones.
    
[^2]: TSLS与JIVE的桥接

    Bridging TSLS and JIVE. (arXiv:2305.17615v1 [econ.EM])

    [http://arxiv.org/abs/2305.17615](http://arxiv.org/abs/2305.17615)

    本文提出了一种桥接TSLS和JIVE的新方法TSJI来处理内生性，具有用户定义参数λ，可以近似无偏，且在许多工具变量渐进情况下是一致且渐进正常的。

    

    在处理内生性时，经济学家经常实施TSLS。当工具变量数量众多时，TSLS偏倚很严重。因此，JIVE被提出来减少超识别的TSLS偏差。但是，这两种方法都有重大缺陷。当超识别度较高时，超定TSLS偏差很大，而JIVE不稳定。在本文中，我将TSLS和JIVE的优化问题桥接起来，解决了连接问题，并提出了一种新的估计器TSJI。TSJI具有用户定义的参数λ。通过将TSJI偏差近似到op（1/N）的方式，我找到了一个产生近似无偏TSJI的λ值。选择了具有所选λ值的TSJI不仅具有在第一阶段和第二阶段回归器数量固定时与TSLS相同的一阶分布，而且在许多工具变量渐进情况下是一致且渐进正常的。在三种不同的模拟设置下，我使用不同力度的工具测试了TSJI与TSLS和JIVE。

    Economists often implement TSLS to handle endogeneity. The bias of TSLS is severe when the number of instruments is large. Hence, JIVE has been proposed to reduce bias of over-identified TSLS. However, both methods have critical drawbacks. While over-identified TSLS has a large bias with a large degree of overidentification, JIVE is unstable. In this paper, I bridge the optimization problems of TSLS and JIVE, solve the connected problem and propose a new estimator TSJI. TSJI has a user-defined parameter $\lambda$. By approximating the bias of the TSJI up to op(1/N), I find a $\lambda$ value that produces approximately unbiased TSJI. TSJI with the selected $\lambda$ value not only has the same first order distribution as TSLS when the number of first-stage and second-stage regressors are fixed, but also is consistent and asymptotically normal under many-instrument asymptotics. Under three different simulation settings, I test TSJI against TSLS and JIVE with instruments of different stre
    

