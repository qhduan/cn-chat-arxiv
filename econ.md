# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Auction Design with Flexible Royalty Payments](https://arxiv.org/abs/2403.19945) | 通过提出具有灵活版税支付机制的设计，我们解决了许可证拍卖中的最优拍卖问题，赢家支付线性版税直至上限，并可能接受审计。 |
| [^2] | [How Periodic Forecast Updates Influence MRP Planning Parameters: A Simulation Study](https://arxiv.org/abs/2403.11010) | 本研究调查了预测更新如何影响MRP计划参数，并提出了扩展MRP的方法以减轻信息更新对生产订单的干扰。 |
| [^3] | [Learning to Manipulate under Limited Information.](http://arxiv.org/abs/2401.16412) | 本研究通过训练神经网络在有限信息条件下学习如何利用不同投票方法进行操纵，发现某些投票方法在有限信息下容易被操纵，而其他方法不容易被操纵。 |
| [^4] | [Multilateral matching with scale economies.](http://arxiv.org/abs/2310.19479) | 本文研究了具有规模经济的多边匹配问题，并证明了在具有规模经济的市场中存在弱集合稳定的结果。该研究适用于那些多个合作伙伴带来优势的环境，并允许代理人对他们签订的合同进行讨价还价。 |
| [^5] | [A Note on the Continuity of Expected Utility Functions.](http://arxiv.org/abs/2310.16806) | 本文研究了期望效用函数的连续性，并给出了在简单概率空间和具有紧支撑的概率空间上，使得弱次序具有连续期望效用函数的必要且充分条件。 |

# 详细

[^1]: 具有灵活版税支付的最优拍卖设计

    Optimal Auction Design with Flexible Royalty Payments

    [https://arxiv.org/abs/2403.19945](https://arxiv.org/abs/2403.19945)

    通过提出具有灵活版税支付机制的设计，我们解决了许可证拍卖中的最优拍卖问题，赢家支付线性版税直至上限，并可能接受审计。

    

    我们研究了一种许可证拍卖的设计。每个参与者对自己赢得许可证后未来利润的信号。如果许可证被分配，赢家可以根据他报告的利润支付灵活的版税。委托人可以对赢家进行审计，这会产生成本，并对其收取有限的处罚。我们为最大化收入净审计成本的拍卖问题进行了求解。在这个拍卖中，赢家支付线性版税直至上限，超过上限则不再审计。一个更乐观的投标者会提前支付更多费用以换取较低的版税上限。

    arXiv:2403.19945v1 Announce Type: new  Abstract: We study the design of an auction for a license. Each agent has a signal about his future profit from winning the license. If the license is allocated, the winner can be charged a flexible royalty based on the profits he reports. The principal can audit the winner, at a cost, and charge limited penalties. We solve for the auction that maximizes revenue, net auditing costs. In this auction, the winner pays linear royalties up to a cap, beyond which there is no auditing. A more optimistic bidder pays more upfront in exchange for a lower royalty cap.
    
[^2]: 定期预测更新如何影响MRP计划参数：一项模拟研究

    How Periodic Forecast Updates Influence MRP Planning Parameters: A Simulation Study

    [https://arxiv.org/abs/2403.11010](https://arxiv.org/abs/2403.11010)

    本研究调查了预测更新如何影响MRP计划参数，并提出了扩展MRP的方法以减轻信息更新对生产订单的干扰。

    

    在许多供应链中，当前的数字化努力已经导致制造商和客户之间的信息交流得到改善。具体而言，需求预测通常由客户提供，并随着相关客户信息的改善而定期更新。本文研究了预测更新对物料需求计划（MRP）生产规划方法的影响。进行了模拟研究，以评估信息更新如何影响滚动视野MRP计划生产系统中设置规划参数。直观的结果是信息更新导致MRP标准生产订单出现干扰，因此开发了一个扩展MRP以减轻这些影响的方法。通过大规模的数值模拟实验表明，开发的MRP安全库存利用启发式方法显著改善了结果。

    arXiv:2403.11010v1 Announce Type: new  Abstract: In many supply chains, the current efforts at digitalization have led to improved information exchanges between manufacturers and their customers. Specifically, demand forecasts are often provided by the customers and regularly updated as the related customer information improves. In this paper, we investigate the influence of forecast updates on the production planning method of Material Requirements Planning (MRP). A simulation study was carried out to assess how updates in information affect the setting of planning parameters in a rolling horizon MRP planned production system. An intuitive result is that information updates lead to disturbances in the production orders for the MRP standard, and, therefore, an extension for MRP to mitigate these effects is developed. A large numerical simulation experiment shows that the MRP safety stock exploitation heuristic, that has been developed, leads to significantly improved results as far as 
    
[^3]: 学习在有限信息下进行操纵

    Learning to Manipulate under Limited Information. (arXiv:2401.16412v1 [cs.AI])

    [http://arxiv.org/abs/2401.16412](http://arxiv.org/abs/2401.16412)

    本研究通过训练神经网络在有限信息条件下学习如何利用不同投票方法进行操纵，发现某些投票方法在有限信息下容易被操纵，而其他方法不容易被操纵。

    

    根据社会选择理论的经典结果，任何合理的偏好投票方法有时会给个体提供报告不真实偏好的激励。对于比较投票方法来说，不同投票方法在多大程度上更或者更少抵抗这种策略性操纵已成为一个关键考虑因素。在这里，我们通过神经网络在不同规模下对限制信息下学习如何利用给定投票方法进行操纵的成功程度来衡量操纵的抵抗力。我们训练了将近40,000个不同规模的神经网络来对抗8种不同的投票方法，在6种限制信息情况下，进行包含5-21名选民和3-6名候选人的委员会规模选举的操纵。我们发现，一些投票方法，如Borda方法，在有限信息下可以被神经网络高度操纵，而其他方法，如Instant Runoff方法，虽然被一个理想的操纵者利润化操纵，但在有限信息下不会受到操纵。

    By classic results in social choice theory, any reasonable preferential voting method sometimes gives individuals an incentive to report an insincere preference. The extent to which different voting methods are more or less resistant to such strategic manipulation has become a key consideration for comparing voting methods. Here we measure resistance to manipulation by whether neural networks of varying sizes can learn to profitably manipulate a given voting method in expectation, given different types of limited information about how other voters will vote. We trained nearly 40,000 neural networks of 26 sizes to manipulate against 8 different voting methods, under 6 types of limited information, in committee-sized elections with 5-21 voters and 3-6 candidates. We find that some voting methods, such as Borda, are highly manipulable by networks with limited information, while others, such as Instant Runoff, are not, despite being quite profitably manipulated by an ideal manipulator with
    
[^4]: 带有规模经济的多边匹配问题

    Multilateral matching with scale economies. (arXiv:2310.19479v1 [econ.TH])

    [http://arxiv.org/abs/2310.19479](http://arxiv.org/abs/2310.19479)

    本文研究了具有规模经济的多边匹配问题，并证明了在具有规模经济的市场中存在弱集合稳定的结果。该研究适用于那些多个合作伙伴带来优势的环境，并允许代理人对他们签订的合同进行讨价还价。

    

    本文研究了多边匹配问题，即任何一组代理人都可以协商合同。我们假设存在规模经济，即只有当新签订的合同涉及一个弱相对较大的合作伙伴集合时，一个代理人才会用一些新合同替代一些原有合同。我们证明了在具有规模经济的市场中存在弱集合稳定的结果，并且在更强的规模经济条件下存在集合稳定的结果。我们的条件适用于那些多个合作伙伴带来优势的环境，并允许代理人对他们签订的合同进行讨价还价。

    This paper studies multilateral matching in which any set of agents can negotiate contracts. We assume scale economies in the sense that an agent substitutes some contracts with some new contracts only if the newly signed contracts involve a weakly larger set of partners. We show that a weakly setwise stable outcome exists in a market with scale economies and a setwise stable outcome exists under a stronger scale economies condition. Our conditions apply to environments in which more partners bring advantages, and allow agents to bargain over contracts signed by them.
    
[^5]: 对期望效用函数连续性的注释

    A Note on the Continuity of Expected Utility Functions. (arXiv:2310.16806v1 [econ.TH])

    [http://arxiv.org/abs/2310.16806](http://arxiv.org/abs/2310.16806)

    本文研究了期望效用函数的连续性，并给出了在简单概率空间和具有紧支撑的概率空间上，使得弱次序具有连续期望效用函数的必要且充分条件。

    

    本文研究了期望效用函数的连续性，并导出了一个弱次序在简单概率空间上具有连续期望效用函数的必要与充分条件。我们还验证了几乎相同的条件对于具有紧支撑的概率空间上的弱次序具有连续期望效用函数而言也是必要且充分的。

    In this paper, we study the continuity of expected utility functions, and derive a necessary and sufficient condition for a weak order on the space of simple probabilities to have a continuous expected utility function. We also verify that almost the same condition is necessary and sufficient for a weak order on the space of probabilities with compact-support to have a continuous expected utility function.
    

