# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Text mining arXiv: a look through quantitative finance papers.](http://arxiv.org/abs/2401.01751) | 本文通过文本挖掘技术和自然语言处理方法，研究了arXiv上的量化金融论文，发现了关于该领域的时间趋势、最常被引用的研究人员和期刊，以及不同算法进行主题建模的比较。 |
| [^2] | [Maximally Machine-Learnable Portfolios.](http://arxiv.org/abs/2306.05568) | 本文通过 MACE 算法，以随机森林和受限岭回归优化组合权重，实现了最大程度的可预测性和盈利能力，适用于任何预测算法和预测器集，可以处理大型组合。 |
| [^3] | [Power sector effects of alternative options for electrifying heavy-duty vehicles: go electric, and charge smartly.](http://arxiv.org/abs/2303.16629) | 研究了电动公路系统和电池电动车的替代方案对于电力部门的影响，发现可灵活充电的车辆共享BEV的电力部门成本最低，而使用电力燃料的重型车辆的成本最高。 |

# 详细

[^1]: 文本挖掘arXiv：对量化金融论文的观察

    Text mining arXiv: a look through quantitative finance papers. (arXiv:2401.01751v1 [cs.DL])

    [http://arxiv.org/abs/2401.01751](http://arxiv.org/abs/2401.01751)

    本文通过文本挖掘技术和自然语言处理方法，研究了arXiv上的量化金融论文，发现了关于该领域的时间趋势、最常被引用的研究人员和期刊，以及不同算法进行主题建模的比较。

    

    本文利用文本挖掘技术和自然语言处理方法，探索了arXiv预印本服务器上的论文，旨在发现这个庞大的研究集合中隐藏的有价值的见解。我们研究了从1997年到2022年在arXiv上发布的量化金融论文的内容。我们从整个文档中提取和分析关键信息，包括引用，以了解随时间变化的主题趋势，并找出这个领域中最常被引用的研究人员和期刊。此外，我们还比较了多种算法来进行主题建模，包括最先进的方法。

    This paper explores articles hosted on the arXiv preprint server with the aim to uncover valuable insights hidden in this vast collection of research. Employing text mining techniques and through the application of natural language processing methods, we examine the contents of quantitative finance papers posted in arXiv from 1997 to 2022. We extract and analyze crucial information from the entire documents, including the references, to understand the topics trends over time and to find out the most cited researchers and journals on this domain. Additionally, we compare numerous algorithms to perform topic modeling, including state-of-the-art approaches.
    
[^2]: 最大机器学习组合的构建方法

    Maximally Machine-Learnable Portfolios. (arXiv:2306.05568v1 [econ.EM])

    [http://arxiv.org/abs/2306.05568](http://arxiv.org/abs/2306.05568)

    本文通过 MACE 算法，以随机森林和受限岭回归优化组合权重，实现了最大程度的可预测性和盈利能力，适用于任何预测算法和预测器集，可以处理大型组合。

    

    对于股票回报，任何形式的可预测性都可以增强调整风险后的盈利能力。本文开发了一种协作机器学习算法，优化组合权重，以使得合成证券最大程度的可预测。具体来说，我们引入了MACE，Alternating Conditional Expectations的多元扩展，通过在方程的一侧使用随机森林和受限岭回归在另一侧实现了上述目标。相较于Lo和MacKinlay的最大可预测组合方法，本文有两个关键改进。第一，它适用于任何（非线性）预测算法和预测器集。第二，它可以处理大型组合。我们进行了日频和月频的实验，并发现在使用很少的条件信息时，可预测性和盈利能力显著增加。有趣的是，可预测性在好时和坏时都存在，并且MACE成功地导航了两者。

    When it comes to stock returns, any form of predictability can bolster risk-adjusted profitability. We develop a collaborative machine learning algorithm that optimizes portfolio weights so that the resulting synthetic security is maximally predictable. Precisely, we introduce MACE, a multivariate extension of Alternating Conditional Expectations that achieves the aforementioned goal by wielding a Random Forest on one side of the equation, and a constrained Ridge Regression on the other. There are two key improvements with respect to Lo and MacKinlay's original maximally predictable portfolio approach. First, it accommodates for any (nonlinear) forecasting algorithm and predictor set. Second, it handles large portfolios. We conduct exercises at the daily and monthly frequency and report significant increases in predictability and profitability using very little conditioning information. Interestingly, predictability is found in bad as well as good times, and MACE successfully navigates
    
[^3]: 重型车辆电气化的替代方案的电力部门影响:电气化和智能充电

    Power sector effects of alternative options for electrifying heavy-duty vehicles: go electric, and charge smartly. (arXiv:2303.16629v1 [econ.GN])

    [http://arxiv.org/abs/2303.16629](http://arxiv.org/abs/2303.16629)

    研究了电动公路系统和电池电动车的替代方案对于电力部门的影响，发现可灵活充电的车辆共享BEV的电力部门成本最低，而使用电力燃料的重型车辆的成本最高。

    

    在乘用车领域，电池电动车(BEV)已成为去碳化交通的最有前途的选择。对于重型车辆(HDV)，技术领域似乎更为开放。除了BEV外，还讨论了用于动态供电的电动公路系统(ERS)，以及使用氢燃料电池或电力燃料的卡车间接电气化。在这里，我们研究了这些替代方案的电力部门影响。我们将基于未来德国高可再生能源份额的情景，应用一个开源的容量扩展模型，利用详细的以路线为基础的卡车交通数据。结果表明，可灵活充电的车辆共享BEV的电力部门成本最低，而使用电力燃料的重型车辆的成本最高。如果BEV和ERS-BEV没有以优化的方式充电，电力部门成本会增加，但仍远低于使用氢或电力燃料的情景。这是相对较小电池、高度灵活的BEV在短途和中途步骤转移和超出道路广泛使用的优势的结果。

    In the passenger car segment, battery-electric vehicles (BEV) have emerged as the most promising option to decarbonize transportation. For heavy-duty vehicles (HDV), the technology space still appears to be more open. Aside from BEV, electric road systems (ERS) for dynamic power transfer are discussed, as well as indirect electrification with trucks that use hydrogen fuel cells or e-fuels. Here we investigate the power sector implications of these alternative options. We apply an open-source capacity expansion model to future scenarios of Germany with high renewable energy shares, drawing on detailed route-based truck traffic data. Results show that power sector costs are lowest for flexibly charged BEV that also carry out vehicle-to-grid operations, and highest for HDV using e-fuels. If BEV and ERS-BEV are not charged in an optimized way, power sector costs increase, but are still substantially lower than in scenarios with hydrogen or e-fuels. This is a consequence of the relatively p
    

