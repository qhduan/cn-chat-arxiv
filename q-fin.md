# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [From Time-inconsistency to Time-consistency for Optimal Stopping Problems](https://arxiv.org/abs/2404.02498) | 该论文研究了具有时间不一致偏好的最优停止问题，通过对时间不一致性水平的衡量，发现了概率失真导致时间不一致性的程度与转变为复杂策略所需的时间之间的关系。 |
| [^2] | [Local Volatility in Interest Rate Models.](http://arxiv.org/abs/2301.13595) | 本文提出了一种实现利率模型中本地波动率的新方法，并在蒙特卡罗计算中应用。该方法很有效，但在短期和低期限互换中存在定价错误。 |

# 详细

[^1]: 从时间不一致到时间一致的最优停止问题

    From Time-inconsistency to Time-consistency for Optimal Stopping Problems

    [https://arxiv.org/abs/2404.02498](https://arxiv.org/abs/2404.02498)

    该论文研究了具有时间不一致偏好的最优停止问题，通过对时间不一致性水平的衡量，发现了概率失真导致时间不一致性的程度与转变为复杂策略所需的时间之间的关系。

    

    对于具有时间不一致偏好的最优停止问题，我们通过将从天真策略转变为复杂策略所需的时间来衡量时间不一致的固有水平。特别是，在重复实验中，当天真代理可以观察到她的实际行动序列与最初计划不一致时，她会根据后续实际行为的观察选择立即行动。该过程重复进行，直到她的实际行动序列在任何时间点都与她的计划一致。我们表明，对于累积概率理论的偏好值，其中时间不一致由于概率失真导致，概率失真程度越高，时间不一致程度就越严重，需要的时间转变从天真策略到复杂策略也更多。

    arXiv:2404.02498v1 Announce Type: new  Abstract: For optimal stopping problems with time-inconsistent preference, we measure the inherent level of time-inconsistency by taking the time needed to turn the naive strategies into the sophisticated ones. In particular, when in a repeated experiment the naive agent can observe her actual sequence of actions which are inconsistent with what she has planned at the initial time, she then chooses her immediate action based on the observations on her later actual behavior. The procedure is repeated until her actual sequence of actions are consistent with her plan at any time. We show that for the preference value of cumulative prospect theory, in which the time-inconsistency is due to the probability distortion, the higher the degree of probability distortion, the more severe the level of time-inconsistency, and the more time required to turn the naive strategies into the sophisticated ones.
    
[^2]: 利率模型中的本地波动率

    Local Volatility in Interest Rate Models. (arXiv:2301.13595v2 [q-fin.PR] UPDATED)

    [http://arxiv.org/abs/2301.13595](http://arxiv.org/abs/2301.13595)

    本文提出了一种实现利率模型中本地波动率的新方法，并在蒙特卡罗计算中应用。该方法很有效，但在短期和低期限互换中存在定价错误。

    

    提出了一种在利率模型中实现本地波动率的新方法。该方法的主要工具是小波动率近似。此近似方法非常有效，可以用于校准所有平值互换。它快速又精确。为了重现所有可用的互换价格，我们需要考虑正向波动率对当前互换率的依赖性。在此，我们假设正向波动率是一个确定性函数，具有在网格上的每一点上的行权价格、到期时间和到期时间的决定。我们确定这些函数并将它们应用于蒙特卡罗计算中。实验证明，这种方法运行良好。然而，在短期和低期限互换中，我们观察到互换定价错误。为了解决这个问题，我们需要修改场景生成过程。

    A new approach to Local Volatility implementation in the interest rate model is presented. The major tool of this approach is a small volatility approximation. This approximation works very well and it can be used to calibrate all ATM swaptions. It works fast and accurate. In order to reproduce all available swaption prices we need to take into account the dependence of forward volatility on the current swap rate. Here we assume that forward volatility is a deterministic function on strike, tenor, and expiration at every point on the grid. We determine these functions and apply them in Monte-Carlo calculations. It was demonstrated that this approach works well. However, in the case of short term and low tenor swaptions we observed errors in swaption pricing. To fix this problem we need to modify the scenario generation process.
    

