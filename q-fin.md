# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Finite Difference Solution Ansatz approach in Least-Squares Monte Carlo.](http://arxiv.org/abs/2305.09166) | 本文提出了一种通用的数值方案，使用低维有限差分法的精确解来构建条件期望继续支付的假设，并将其用于线性回归，以提高在美式期权定价中最小二乘蒙特卡罗方法的精确性。 |
| [^2] | [Efficient OCR for Building a Diverse Digital History.](http://arxiv.org/abs/2304.02737) | 本研究使用对比训练的视觉编码器，将OCR建模为字符级图像检索问题，相比于已有架构更具样本效率和可扩展性，从而使数字历史更具代表性的文献史料得以更好地参与社区。 |
| [^3] | [Synthetic Controls with Multiple Outcomes: Estimating the Effects of Non-Pharmaceutical Interventions in the COVID-19 Pandemic.](http://arxiv.org/abs/2304.02272) | 本研究提出了一种新的合成控制方法，可以评估一种治疗对于多个结果变量的影响。在使用该方法估计瑞典在COVID-19大流行中非药物干预措施的效果时，我们得出结论：如果瑞典在3月份像欧洲其他国家一样实施了更严格的非药物干预措施，那么到7月份，COVID-19感染病例和死亡人数将减少约70％，而在5月初所有原因的死亡人数将减少约20％。 |

# 详细

[^1]: 最小二乘蒙特卡罗中的有限差分解法

    Finite Difference Solution Ansatz approach in Least-Squares Monte Carlo. (arXiv:2305.09166v1 [q-fin.GN])

    [http://arxiv.org/abs/2305.09166](http://arxiv.org/abs/2305.09166)

    本文提出了一种通用的数值方案，使用低维有限差分法的精确解来构建条件期望继续支付的假设，并将其用于线性回归，以提高在美式期权定价中最小二乘蒙特卡罗方法的精确性。

    

    本文提出了一种简单而有效的方法，以提高在美式期权定价中最小二乘蒙特卡罗方法的精确性。关键思想是使用低维有限差分法的精确解来构建条件期望继续支付的假设，用于线性回归。该方法在解决向后偏微分方程和蒙特卡罗模拟方面建立了桥梁，旨在实现两者的最佳结合。我们通过实际示例说明该技术，包括百慕大期权和最差发行人可赎回票据。该方法可被视为跨越各种资产类别的通用数值方案，特别是在任意维度下，作为定价美式衍生产品的准确方法。

    This article presents a simple but effective approach to improve the accuracy of Least-Squares Monte Carlo for American-style options. The key idea is to construct the ansatz of conditional expected continuation payoff using the exact solution from low dimensional finite difference methods, to be used in linear regression. This approach builds a bridge between solving backward partial differential equations and a Monte Carlo simulation, aiming at achieving the best of both worlds. We illustrate the technique with realistic examples including Bermuda options and worst of issuer callable notes. The method can be considered as a generic numerical scheme across various asset classes, in particular, as an accurate method for pricing American-style derivatives under arbitrary dimensions.
    
[^2]: 建设多样化数字历史的高效OCR

    Efficient OCR for Building a Diverse Digital History. (arXiv:2304.02737v1 [cs.CV])

    [http://arxiv.org/abs/2304.02737](http://arxiv.org/abs/2304.02737)

    本研究使用对比训练的视觉编码器，将OCR建模为字符级图像检索问题，相比于已有架构更具样本效率和可扩展性，从而使数字历史更具代表性的文献史料得以更好地参与社区。

    

    每天有成千上万的用户查阅数字档案，但他们可以使用的信息并不能代表各种文献史料的多样性。典型用于光学字符识别（OCR）的序列到序列架构——联合学习视觉和语言模型——在低资源文献集合中很难扩展，因为学习语言-视觉模型需要大量标记的序列和计算。本研究将OCR建模为字符级图像检索问题，使用对比训练的视觉编码器。因为该模型只学习字符的视觉特征，它比现有架构更具样本效率和可扩展性，能够在现有解决方案失败的情况下实现准确的OCR。关键是，该模型为社区参与在使数字历史更具代表性的文献史料方面开辟了新的途径。

    Thousands of users consult digital archives daily, but the information they can access is unrepresentative of the diversity of documentary history. The sequence-to-sequence architecture typically used for optical character recognition (OCR) - which jointly learns a vision and language model - is poorly extensible to low-resource document collections, as learning a language-vision model requires extensive labeled sequences and compute. This study models OCR as a character level image retrieval problem, using a contrastively trained vision encoder. Because the model only learns characters' visual features, it is more sample efficient and extensible than existing architectures, enabling accurate OCR in settings where existing solutions fail. Crucially, the model opens new avenues for community engagement in making digital history more representative of documentary history.
    
[^3]: 带多个结果的合成控制：COVID-19大流行中估计非药物干预效果

    Synthetic Controls with Multiple Outcomes: Estimating the Effects of Non-Pharmaceutical Interventions in the COVID-19 Pandemic. (arXiv:2304.02272v1 [econ.GN])

    [http://arxiv.org/abs/2304.02272](http://arxiv.org/abs/2304.02272)

    本研究提出了一种新的合成控制方法，可以评估一种治疗对于多个结果变量的影响。在使用该方法估计瑞典在COVID-19大流行中非药物干预措施的效果时，我们得出结论：如果瑞典在3月份像欧洲其他国家一样实施了更严格的非药物干预措施，那么到7月份，COVID-19感染病例和死亡人数将减少约70％，而在5月初所有原因的死亡人数将减少约20％。

    

    我们提出了一个将合成控制方法推广到多结果框架的算法，可以提高治疗效果评估的可靠性。这是通过在计算合成控制权重时，将传统的预处理时间维度与相关结果的额外维度进行补充来实现的。我们的推广对于评估一种治疗对于多个结果变量的影响的研究尤其有用。我们用瑞典在2020年前三个季度的数据来说明我们的方法，估计非药物干预措施对多个结果的影响。我们的结果表明，如果瑞典在3月份像欧洲其他国家一样实施了更严格的非药物干预措施，那么到7月份，COVID-19感染病例和死亡人数将减少约70％，而在5月初所有原因的死亡人数将减少约20％。而非药物干预措施对劳动力市场和经济结果的影响相对较小。

    We propose a generalization of the synthetic control method to a multiple-outcome framework, which improves the reliability of treatment effect estimation. This is done by supplementing the conventional pre-treatment time dimension with the extra dimension of related outcomes in computing the synthetic control weights. Our generalization can be particularly useful for studies evaluating the effect of a treatment on multiple outcome variables. To illustrate our method, we estimate the effects of non-pharmaceutical interventions (NPIs) on various outcomes in Sweden in the first 3 quarters of 2020. Our results suggest that if Sweden had implemented stricter NPIs like the other European countries by March, then there would have been about 70% fewer cumulative COVID-19 infection cases and deaths by July, and 20% fewer deaths from all causes in early May, whereas the impacts of the NPIs were relatively mild on the labor market and economic outcomes.
    

