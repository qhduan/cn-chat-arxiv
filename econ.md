# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Improved Semi-parametric Bounds for Tail Probability and Expected Loss](https://arxiv.org/abs/2404.02400) | 本研究提出了对累积随机实现的尾部概率和期望线性损失的新的更尖锐界限，这些界限在基础分布半参数的情况下不受限制，补充了已有的结果，开辟了丰富的实际应用。 |
| [^2] | [Causal Effects in Matching Mechanisms with Strategically Reported Preferences.](http://arxiv.org/abs/2307.14282) | 本文提供一种考虑了策略性误报的因果效应识别方法，可以对学校分配对未来结果产生的影响进行准确度量。我们的方法适用于各种机制，并能够得到对策略行为鲁棒的因果效应的尖锐边界。 |
| [^3] | [Online Learning of Order Flow and Market Impact with Bayesian Change-Point Detection Methods.](http://arxiv.org/abs/2307.02375) | 本文提出了使用贝叶斯变点检测方法识别实时订单流变化的方法，并通过开发一种新的BOCPD方法，可以更准确地预测订单流和市场影响。实证结果表明，我们的模型在样本外预测性能上优于现有模型。 |

# 详细

[^1]: 关于累积随机实现的尾部概率和期望损失的改进半参数界限

    On Improved Semi-parametric Bounds for Tail Probability and Expected Loss

    [https://arxiv.org/abs/2404.02400](https://arxiv.org/abs/2404.02400)

    本研究提出了对累积随机实现的尾部概率和期望线性损失的新的更尖锐界限，这些界限在基础分布半参数的情况下不受限制，补充了已有的结果，开辟了丰富的实际应用。

    

    我们重新审视了当个别实现是独立的时，累积随机实现的尾部行为的基本问题，并在半参数的基础分布未受限制的情况下，开发了对尾部概率和期望线性损失的新的更尖锐的界限。我们的尖锐界限很好地补充了文献中已经建立的结果，包括基于聚合的方法，后者经常未能充分考虑独立性并使用不够优雅的证明。新的见解包括在非相同情况下的证明，达到界限的分布具有相等的范围属性，并且每个随机变量对总和的期望值的影响可以通过对Korkine恒等式的推广来孤立出来。我们表明，新的界限不仅补充了现有结果，而且开拓了大量的实际应用，包括改进定价。

    arXiv:2404.02400v1 Announce Type: new  Abstract: We revisit the fundamental issue of tail behavior of accumulated random realizations when individual realizations are independent, and we develop new sharper bounds on the tail probability and expected linear loss. The underlying distribution is semi-parametric in the sense that it remains unrestricted other than the assumed mean and variance. Our sharp bounds complement well-established results in the literature, including those based on aggregation, which often fail to take full account of independence and use less elegant proofs. New insights include a proof that in the non-identical case, the distributions attaining the bounds have the equal range property, and that the impact of each random variable on the expected value of the sum can be isolated using an extension of the Korkine identity. We show that the new bounds not only complement the extant results but also open up abundant practical applications, including improved pricing 
    
[^2]: 匹配机制中的因果效应与策略性报告偏好

    Causal Effects in Matching Mechanisms with Strategically Reported Preferences. (arXiv:2307.14282v1 [econ.EM])

    [http://arxiv.org/abs/2307.14282](http://arxiv.org/abs/2307.14282)

    本文提供一种考虑了策略性误报的因果效应识别方法，可以对学校分配对未来结果产生的影响进行准确度量。我们的方法适用于各种机制，并能够得到对策略行为鲁棒的因果效应的尖锐边界。

    

    越来越多的中央机构使用分配机制将学生分配到学校，以反映学生的偏好和学校的优先权。然而，大多数现实世界的机制会给学生提供一种策略性并误报他们的偏好的激励。在本文中，我们提供了一种识别因果效应的方法，该方法考虑了策略性的误报。误报可能使现有的点识别方法无效，我们推导出对策略行为鲁棒的因果效应的尖锐边界。我们的方法适用于任何机制，只要存在描述该机制分配规则的配对分数和截点。我们使用智利一个延迟接受机制的数据，该机制将学生分配到1000多个大学专业组合。学生出于策略考虑而行动，因为智利的机制限制了学生在偏好中提交的专业数量为八个。

    A growing number of central authorities use assignment mechanisms to allocate students to schools in a way that reflects student preferences and school priorities. However, most real-world mechanisms give students an incentive to be strategic and misreport their preferences. In this paper, we provide an identification approach for causal effects of school assignment on future outcomes that accounts for strategic misreporting. Misreporting may invalidate existing point-identification approaches, and we derive sharp bounds for causal effects that are robust to strategic behavior. Our approach applies to any mechanism as long as there exist placement scores and cutoffs that characterize that mechanism's allocation rule. We use data from a deferred acceptance mechanism that assigns students to more than 1,000 university-major combinations in Chile. Students behave strategically because the mechanism in Chile constrains the number of majors that students submit in their preferences to eight
    
[^3]: 在线学习订单流与市场影响的贝叶斯变点检测方法

    Online Learning of Order Flow and Market Impact with Bayesian Change-Point Detection Methods. (arXiv:2307.02375v1 [q-fin.TR])

    [http://arxiv.org/abs/2307.02375](http://arxiv.org/abs/2307.02375)

    本文提出了使用贝叶斯变点检测方法识别实时订单流变化的方法，并通过开发一种新的BOCPD方法，可以更准确地预测订单流和市场影响。实证结果表明，我们的模型在样本外预测性能上优于现有模型。

    

    金融订单流表现出明显的持久性，买入（卖出）交易常常会在一定时间段后跟随着进一步的买入（卖出）交易。这种持久性可以归因于大订单的划分和逐步执行。因此，可能会出现不同的订单流模式，可以通过适用于市场数据的适当时间序列模型来识别。在本文中，我们提出了使用贝叶斯在线变点检测（BOCPD）方法来实时识别制度性转变，并实现订单流和市场影响的在线预测。为了提高我们方法的有效性，我们开发了一种使用评分驱动方法的新型BOCPD方法。该方法适应每个制度内的时间相关性和时间变化参数。通过对纳斯达克数据的实证应用，我们发现：（i）我们新提出的模型展示出优于现有假设独立同分布的模型的样本外预测性能。

    Financial order flow exhibits a remarkable level of persistence, wherein buy (sell) trades are often followed by subsequent buy (sell) trades over extended periods. This persistence can be attributed to the division and gradual execution of large orders. Consequently, distinct order flow regimes might emerge, which can be identified through suitable time series models applied to market data. In this paper, we propose the use of Bayesian online change-point detection (BOCPD) methods to identify regime shifts in real-time and enable online predictions of order flow and market impact. To enhance the effectiveness of our approach, we have developed a novel BOCPD method using a score-driven approach. This method accommodates temporal correlations and time-varying parameters within each regime. Through empirical application to NASDAQ data, we have found that: (i) Our newly proposed model demonstrates superior out-of-sample predictive performance compared to existing models that assume i.i.d.
    

