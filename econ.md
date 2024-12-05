# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Facility Location Games with Scaling Effects](https://arxiv.org/abs/2402.18908) | 研究了具有规模效应的设施选址游戏，提供了对于连续比例函数和分段线性比例函数的结果，适用于许多实际情景，同时探讨了近似机制设计设置下代理可能不再单峰偏好的条件与成本近似比率。 |
| [^2] | [Combinatorial Pen Testing (or Consumer Surplus of Deferred-Acceptance Auctions).](http://arxiv.org/abs/2301.12462) | 该论文提出了一种组合式渗透测试算法，通过类比拍卖理论框架，可以将近似最优的延迟接受机制转换为近似最优的渗透测试算法，并且在逼近程度上保证了额外开销的上限。 |

# 详细

[^1]: 具有规模效应的设施选址游戏

    Facility Location Games with Scaling Effects

    [https://arxiv.org/abs/2402.18908](https://arxiv.org/abs/2402.18908)

    研究了具有规模效应的设施选址游戏，提供了对于连续比例函数和分段线性比例函数的结果，适用于许多实际情景，同时探讨了近似机制设计设置下代理可能不再单峰偏好的条件与成本近似比率。

    

    我们考虑了经典的设施选址问题的一个变种，其中每个代理的个人成本函数等于他们距离设施的距离乘以一个由设施位置确定的比例因子。除了一般类别的连续比例函数外，我们还提供了适用于许多实际情景的比例函数的分段线性比例函数的结果。我们关注总成本和最大成本的目标，并描述了最优解的计算。然后我们转向近似机制设计设置，观察到代理的偏好可能不再是单峰的。因此，我们表征了确保代理具有单峰偏好的比例函数条件。在这些条件下，我们找到了能够通过strategyproof和anonymous me达到的总成本和最大成本近似比率的结果。

    arXiv:2402.18908v1 Announce Type: cross  Abstract: We take the classic facility location problem and consider a variation, in which each agent's individual cost function is equal to their distance from the facility multiplied by a scaling factor which is determined by the facility placement. In addition to the general class of continuous scaling functions, we also provide results for piecewise linear scaling functions which can effectively approximate or model the scaling of many real world scenarios. We focus on the objectives of total and maximum cost, describing the computation of the optimal solution. We then move to the approximate mechanism design setting, observing that the agents' preferences may no longer be single-peaked. Consequently, we characterize the conditions on scaling functions which ensure that agents have single-peaked preferences. Under these conditions, we find results on the total and maximum cost approximation ratios achievable by strategyproof and anonymous me
    
[^2]: 组合式渗透测试（或者延迟接受拍卖的消费者剩余值）

    Combinatorial Pen Testing (or Consumer Surplus of Deferred-Acceptance Auctions). (arXiv:2301.12462v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2301.12462](http://arxiv.org/abs/2301.12462)

    该论文提出了一种组合式渗透测试算法，通过类比拍卖理论框架，可以将近似最优的延迟接受机制转换为近似最优的渗透测试算法，并且在逼近程度上保证了额外开销的上限。

    

    渗透测试是当只能通过消耗资源的能力来衡量资源容量时，选择高容量资源的问题。我们有一组$n$只笔，每只笔都有未知的墨水量，我们的目标是选择一个可行的笔集合，使其中墨水总量最大。我们可以通过使用它们来获得更多信息，但这将消耗先前存在于笔中的墨水。算法将与标准基准（即最优的渗透测试算法）和全知基准（即如果了解笔中墨水的数量，则最优的选择）进行评估。我们通过将延迟接受拍卖和虚拟价值的拍卖理论框架类比，确定了最优和近似最优的渗透测试算法。我们的框架允许将任何近似最优的延迟接受机制转换为近似最优的渗透测试算法。此外，这些算法保证在逼近程度上存在额外开销至多$(1+o(1)) \ln n$。

    Pen testing is the problem of selecting high-capacity resources when the only way to measure the capacity of a resource expends its capacity. We have a set of $n$ pens with unknown amounts of ink and our goal is to select a feasible subset of pens maximizing the total ink in them. We are allowed to gather more information by writing with them, but this uses up ink that was previously in the pens. Algorithms are evaluated against the standard benchmark, i.e, the optimal pen testing algorithm, and the omniscient benchmark, i.e, the optimal selection if the quantity of ink in the pens are known.  We identify optimal and near optimal pen testing algorithms by drawing analogues to auction theoretic frameworks of deferred-acceptance auctions and virtual values. Our framework allows the conversion of any near optimal deferred-acceptance mechanism into a near optimal pen testing algorithm. Moreover, these algorithms guarantee an additional overhead of at most $(1+o(1)) \ln n$ in the approximat
    

