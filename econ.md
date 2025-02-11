# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Database for the meta-analysis of the social cost of carbon (v2024.0)](https://arxiv.org/abs/2402.09125) | 该论文介绍了社会碳成本估计元分析数据库的新版本，新增了关于气候变化影响和福利函数形状的字段，并扩展了合作者和引用网络。 |
| [^2] | [Assessing Heterogeneity of Treatment Effects.](http://arxiv.org/abs/2306.15048) | 该论文介绍了一种评估治疗效果异质性的方法，通过使用治疗组和对照组结果的分位数范围，即使平均效果不显著，也可以提供有用的信息。 |
| [^3] | [Pairwise Valid Instruments.](http://arxiv.org/abs/2203.08050) | 本论文提出了一种名为Validity Set Instrumental Variable (VSIV)估计的方法，用于在工具部分无效时估计局部平均处理效应(LATEs)。方法利用可检验的工具有效性推论移除无效的工具-值对，提供剩余对的LATEs估计，并通过研究者指定的权重聚合成一个参数。所提出的VSIV估计器在弱条件下渐近正态，并消除或减少相对于标准LATE估计器的渐近偏差。 |

# 详细

[^1]: 社会碳成本的元分析数据库 (v2024.0)

    Database for the meta-analysis of the social cost of carbon (v2024.0)

    [https://arxiv.org/abs/2402.09125](https://arxiv.org/abs/2402.09125)

    该论文介绍了社会碳成本估计元分析数据库的新版本，新增了关于气候变化影响和福利函数形状的字段，并扩展了合作者和引用网络。

    

    本文介绍了社会碳成本估计元分析数据库的新版本。新增了记录，并添加了关于气候变化影响和福利函数形状的新字段。该数据库还扩展了合作者和引用网络。

    arXiv:2402.09125v1 Announce Type: new Abstract: A new version of the database for the meta-analysis of estimates of the social cost of carbon is presented. New records were added, and new fields on the impact of climate change and the shape of the welfare function. The database was extended to co-author and citation networks.
    
[^2]: 评估治疗效果的异质性

    Assessing Heterogeneity of Treatment Effects. (arXiv:2306.15048v1 [econ.EM])

    [http://arxiv.org/abs/2306.15048](http://arxiv.org/abs/2306.15048)

    该论文介绍了一种评估治疗效果异质性的方法，通过使用治疗组和对照组结果的分位数范围，即使平均效果不显著，也可以提供有用的信息。

    

    异质性治疗效果在经济学中非常重要，但是其评估常常受到个体治疗效果无法确定的困扰。例如，我们可能希望评估保险对本来不健康的人的健康影响，但是只给不健康的人买保险是不可行的，因此这些人的因果效应无法确定。又或者，我们可能对最低工资上涨中赢家的份额感兴趣，但是在没有观察到反事实的情况下，赢家也无法确定。这种异质性常常通过分位数治疗效果来评估，但其解释并不清晰，结论有时也不一致。我们展示了通过治疗组和对照组结果的分位数，这些数值范围是可以确定的，即使平均治疗效果并不显著，它们仍然可以提供有用信息。两个应用实例展示了这些范围如何帮助我们了解治疗效果的异质性。

    Treatment effect heterogeneity is of major interest in economics, but its assessment is often hindered by the fundamental lack of identification of the individual treatment effects. For example, we may want to assess the effect of insurance on the health of otherwise unhealthy individuals, but it is infeasible to insure only the unhealthy, and thus the causal effects for those are not identified. Or, we may be interested in the shares of winners from a minimum wage increase, while without observing the counterfactual, the winners are not identified. Such heterogeneity is often assessed by quantile treatment effects, which do not come with clear interpretation and the takeaway can sometimes be equivocal. We show that, with the quantiles of the treated and control outcomes, the ranges of these quantities are identified and can be informative even when the average treatment effects are not significant. Two applications illustrate how these ranges can inform us about heterogeneity of the t
    
[^3]: 一对一有效工具

    Pairwise Valid Instruments. (arXiv:2203.08050v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2203.08050](http://arxiv.org/abs/2203.08050)

    本论文提出了一种名为Validity Set Instrumental Variable (VSIV)估计的方法，用于在工具部分无效时估计局部平均处理效应(LATEs)。方法利用可检验的工具有效性推论移除无效的工具-值对，提供剩余对的LATEs估计，并通过研究者指定的权重聚合成一个参数。所提出的VSIV估计器在弱条件下渐近正态，并消除或减少相对于标准LATE估计器的渐近偏差。

    

    寻找有效的工具是困难的。我们提出了一种称为Validity Set Instrumental Variable (VSIV)估计的方法，用于估计局部平均处理效应(LATEs)在当工具部分无效时异质因果效应模型。我们考虑具有一对一有效工具的环境，也就是说，在某些工具-值对中工具是有效的。VSIV估计利用工具有效性的可检验推论来移除无效对，并提供剩余对的LATEs估计，可以使用研究者指定的权重将其聚合成一个感兴趣的参数。我们证明了在弱条件下，所提出的VSIV估计器是渐近正态的，并相对于标准LATE估计器(即不使用可检验推论来移除无效变异的LATE估计器)消除或减少了渐近偏差。我们通过基于应用的模拟评估了VSIV估计的有限样本特性，并应用我们的方法。

    Finding valid instruments is difficult. We propose Validity Set Instrumental Variable (VSIV) estimation, a method for estimating local average treatment effects (LATEs) in heterogeneous causal effect models when the instruments are partially invalid. We consider settings with pairwise valid instruments, that is, instruments that are valid for a subset of instrument value pairs. VSIV estimation exploits testable implications of instrument validity to remove invalid pairs and provides estimates of the LATEs for all remaining pairs, which can be aggregated into a single parameter of interest using researcher-specified weights. We show that the proposed VSIV estimators are asymptotically normal under weak conditions and remove or reduce the asymptotic bias relative to standard LATE estimators (that is, LATE estimators that do not use testable implications to remove invalid variation). We evaluate the finite sample properties of VSIV estimation in application-based simulations and apply our
    

