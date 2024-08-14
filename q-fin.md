# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Similarity and Comparison Complexity](https://arxiv.org/abs/2401.17578) | 本研究提出了一个关于比较复杂性的理论，并证明了比较复杂性导致选择错误和系统性错误。我们的模型表明，选项的相似性和接近支配度是影响比较复杂性的两个重要因素。该研究的实证结果还表明，比较复杂性度量可以预测选择错误、选择不一致以及认知不确定性，并可以解释选择和评估中的经典异常现象。 |
| [^2] | [The McCormick martingale optimal transport.](http://arxiv.org/abs/2401.15552) | 本研究通过引入资产因果约束，提出了麦科密克松最佳传输模型（McCormick MOT），通过麦科密克松松弛来缓解计算上的挑战，并且实证结果表明其能够缩小价格界限，取得了显著的降低。 |
| [^3] | [Neural networks can detect model-free static arbitrage strategies.](http://arxiv.org/abs/2306.16422) | 本文证明了神经网络可以检测金融市场中的无模型静态套利机会，并可应用于交易证券数量较多的金融市场。我们的方法具有易处理性、有效性和稳健性，并使用真实金融数据进行了示例验证。 |

# 详细

[^1]: 相似性和比较的复杂性

    Similarity and Comparison Complexity

    [https://arxiv.org/abs/2401.17578](https://arxiv.org/abs/2401.17578)

    本研究提出了一个关于比较复杂性的理论，并证明了比较复杂性导致选择错误和系统性错误。我们的模型表明，选项的相似性和接近支配度是影响比较复杂性的两个重要因素。该研究的实证结果还表明，比较复杂性度量可以预测选择错误、选择不一致以及认知不确定性，并可以解释选择和评估中的经典异常现象。

    

    一些选择选项比其他选项更难比较。本文建立了一个关于什么导致比较复杂以及比较复杂如何在选择中产生系统性错误的理论。在我们的模型中，当选项共享相似特征（在固定价值差异的情况下）且更接近支配时，比较更容易。我们展示了如何在多属性、彩票和时间间隔选择领域通过这两个假设得出可操作的比较复杂性度量。使用关于二元选择的实验数据，我们证明了我们的复杂性度量可以预测选择错误、选择不一致以及认知不确定性在这三个领域中的存在。然后我们展示了在选择和评估中的经典异常现象，诸如情境效应、偏好逆转以及在对冒险和时间间隔前景的评估中表现出的明显概率加权和时间偏差，可以被理解为对比较复杂性的反应。

    Some choice options are more difficult to compare than others. This paper develops a theory of what makes a comparison complex, and how comparison complexity generates systematic mistakes in choice. In our model, options are easier to compare when they 1) share similar features, holding fixed their value difference, and 2) are closer to dominance. We show how these two postulates yield tractable measures of comparison complexity in the domains of multiattribute, lottery, and intertemporal choice. Using experimental data on binary choices, we demonstrate that our complexity measures predict choice errors, choice inconsistency, and cognitive uncertainty across all three domains. We then show how canonical anomalies in choice and valuation, such as context effects, preference reversals, and apparent probability weighting and present bias in the valuation of risky and intertemporal prospects, can be understood as responses to comparison complexity.
    
[^2]: McCormick马丁格尔最佳传输

    The McCormick martingale optimal transport. (arXiv:2401.15552v1 [q-fin.MF])

    [http://arxiv.org/abs/2401.15552](http://arxiv.org/abs/2401.15552)

    本研究通过引入资产因果约束，提出了麦科密克松最佳传输模型（McCormick MOT），通过麦科密克松松弛来缓解计算上的挑战，并且实证结果表明其能够缩小价格界限，取得了显著的降低。

    

    马丁格尔最佳传输（MOT）通常为期权提供广泛的价格界限，限制了它们的实际适用性。在本研究中，我们通过引入资产之间的因果约束，扩展了MOT，灵感来自随机过程的非预期性条件。然而，这引入了一个计算上具有挑战性的双线性规划。为了解决这个问题，我们提出了麦科密克松松弛来缓解双因果形式，并将其称为麦科密克松MOT。在标准假设下，建立了麦科密克松MOT的原始实现和强对偶性。实证结果显示，麦科密克松MOT能够缩小价格界限，平均降低了1％或4％。改善程度取决于期权的回报和相关香草期权的流动性。

    Martingale optimal transport (MOT) often yields broad price bounds for options, constraining their practical applicability. In this study, we extend MOT by incorporating causality constraints among assets, inspired by the nonanticipativity condition of stochastic processes. However, this introduces a computationally challenging bilinear program. To tackle this issue, we propose McCormick relaxations to ease the bicausal formulation and refer to it as McCormick MOT. The primal attainment and strong duality of McCormick MOT are established under standard assumptions. Empirically, McCormick MOT demonstrates the capability to narrow price bounds, achieving an average reduction of 1% or 4%. The degree of improvement depends on the payoffs of the options and the liquidity of the relevant vanilla options.
    
[^3]: 神经网络可以检测无模型静态套利策略

    Neural networks can detect model-free static arbitrage strategies. (arXiv:2306.16422v1 [q-fin.CP])

    [http://arxiv.org/abs/2306.16422](http://arxiv.org/abs/2306.16422)

    本文证明了神经网络可以检测金融市场中的无模型静态套利机会，并可应用于交易证券数量较多的金融市场。我们的方法具有易处理性、有效性和稳健性，并使用真实金融数据进行了示例验证。

    

    本文利用理论和数值方法证明了神经网络可以在市场存在套利机会时检测出无模型静态套利机会。由于使用了神经网络，我们的方法可以应用于交易证券数量较多的金融市场，并确保相应交易策略的几乎即时执行。为了证明其易处理性、有效性和稳健性，我们提供了使用真实金融数据的示例。从技术角度来看，我们证明了单个神经网络可以近似解决一类凸半无限规划问题，这是推导出我们的理论结果的关键。

    In this paper we demonstrate both theoretically as well as numerically that neural networks can detect model-free static arbitrage opportunities whenever the market admits some. Due to the use of neural networks, our method can be applied to financial markets with a high number of traded securities and ensures almost immediate execution of the corresponding trading strategies. To demonstrate its tractability, effectiveness, and robustness we provide examples using real financial data. From a technical point of view, we prove that a single neural network can approximately solve a class of convex semi-infinite programs, which is the key result in order to derive our theoretical results that neural networks can detect model-free static arbitrage strategies whenever the financial market admits such opportunities.
    

