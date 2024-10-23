# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Identification with possibly invalid IVs.](http://arxiv.org/abs/2401.03990) | 本文提出了一种新的识别策略，通过使用准工具变量（准IVs）来识别模型。与传统的工具变量方法相比，使用互补的准IVs可以更容易找到。具体而言，通过结合一个被排除但可能内生的准IV和一个外生但可能被包含的准IV，可以实现识别。这种方法在处理离散或连续内生处理的模型上具有广泛适用性。 |
| [^2] | [Data-Driven Fixed-Point Tuning for Truncated Realized Variations.](http://arxiv.org/abs/2311.00905) | 本文提出了一种基于数据驱动的截断实现变异的固定点调整方法，有效估计积分波动性。 |
| [^3] | [Global Factors in Non-core Bank Funding and Exchange Rate Flexibility.](http://arxiv.org/abs/2310.11552) | 全球因素对发达经济体银行体系中非核心与核心资金比率的波动起主导作用，汇率灵活性能够在2008-2009年以外的时期减小这种影响。 |

# 详细

[^1]: 可能无效工具变量下的识别

    Identification with possibly invalid IVs. (arXiv:2401.03990v1 [econ.EM])

    [http://arxiv.org/abs/2401.03990](http://arxiv.org/abs/2401.03990)

    本文提出了一种新的识别策略，通过使用准工具变量（准IVs）来识别模型。与传统的工具变量方法相比，使用互补的准IVs可以更容易找到。具体而言，通过结合一个被排除但可能内生的准IV和一个外生但可能被包含的准IV，可以实现识别。这种方法在处理离散或连续内生处理的模型上具有广泛适用性。

    

    本文提出了一种新颖的识别策略，依赖于准工具变量（准IVs）。准IV是一个相关但可能无效的IV，因为它并非完全外生和/或被排除在外。我们展示了在两个互补的准IV共同相关的条件下，可以识别一系列具有离散或连续内生处理的模型。这些模型通常通过IV实现识别，例如具有秩不变性的分位数模型，具有齐次处理效应的加法模型和局部平均处理效应模型。为了实现识别，我们将一个被排除但可能是内生的准IV（例如“相关代理”，如先前的处理选择）与一个外生的（在被排除的准IV条件下）但可能被包含的准IV（例如随机分配或外生市场冲击）结合起来。在实践中，我们的识别策略应该更具吸引力，因为互补准IV比标准IV更容易找到。我们的方法还具有……

    This paper proposes a novel identification strategy relying on quasi-instrumental variables (quasi-IVs). A quasi-IV is a relevant but possibly invalid IV because it is not completely exogenous and/or excluded. We show that a variety of models with discrete or continuous endogenous treatment, which are usually identified with an IV - quantile models with rank invariance additive models with homogenous treatment effects, and local average treatment effect models - can be identified under the joint relevance of two complementary quasi-IVs instead. To achieve identification we complement one excluded but possibly endogenous quasi-IV (e.g., ``relevant proxies'' such as previous treatment choice) with one exogenous (conditional on the excluded quasi-IV) but possibly included quasi-IV (e.g., random assignment or exogenous market shocks). In practice, our identification strategy should be attractive since complementary quasi-IVs should be easier to find than standard IVs. Our approach also hol
    
[^2]: 数据驱动的截断实现变异的固定点调整方法

    Data-Driven Fixed-Point Tuning for Truncated Realized Variations. (arXiv:2311.00905v1 [math.ST])

    [http://arxiv.org/abs/2311.00905](http://arxiv.org/abs/2311.00905)

    本文提出了一种基于数据驱动的截断实现变异的固定点调整方法，有效估计积分波动性。

    

    在估计存在跳跃的半鞅的积分波动性和相关泛函时，许多方法需要指定调整参数的使用。在现有的理论中，调整参数被假设为确定性的，并且其值仅在渐近约束条件下指定。然而，在实证研究和模拟研究中，它们通常被选择为随机和数据相关的，实际上仅依赖于启发式方法。在本文中，我们考虑了一种基于一种随机固定点迭代的半鞅带跳跃的截断实现变异的新颖数据驱动调整程序。我们的方法是高度自动化的，可以减轻关于调整参数的微妙决策的需求，并且可以仅使用关于采样频率的信息进行实施。我们展示了我们的方法可以导致渐进有效的积分波动性估计，并展示了其在

    Many methods for estimating integrated volatility and related functionals of semimartingales in the presence of jumps require specification of tuning parameters for their use. In much of the available theory, tuning parameters are assumed to be deterministic, and their values are specified only up to asymptotic constraints. However, in empirical work and in simulation studies, they are typically chosen to be random and data-dependent, with explicit choices in practice relying on heuristics alone. In this paper, we consider novel data-driven tuning procedures for the truncated realized variations of a semimartingale with jumps, which are based on a type of stochastic fixed-point iteration. Being effectively automated, our approach alleviates the need for delicate decision-making regarding tuning parameters, and can be implemented using information regarding sampling frequency alone. We show our methods can lead to asymptotically efficient estimation of integrated volatility and exhibit 
    
[^3]: 非核心银行资金和汇率灵活性中的全球因素

    Global Factors in Non-core Bank Funding and Exchange Rate Flexibility. (arXiv:2310.11552v1 [econ.GN])

    [http://arxiv.org/abs/2310.11552](http://arxiv.org/abs/2310.11552)

    全球因素对发达经济体银行体系中非核心与核心资金比率的波动起主导作用，汇率灵活性能够在2008-2009年以外的时期减小这种影响。

    

    我们展示了发达经济体银行体系中非核心与核心资金比率的波动由少数几个既有实物性又有金融性质的全球因素驱动，国家特定因素没有发挥重要作用。汇率灵活性有助于减小非核心与核心比率受到全球因素的影响，但仅在重大全球金融震荡期间（如2008-2009年）明显起作用。

    We show that fluctuations in the ratio of non-core to core funding in the banking systems of advanced economies are driven by a handful of global factors of both real and financial natures, with country-specific factors playing no significant roles. Exchange rate flexibility helps insulate the non-core to core ratio from such global factors but only significantly so outside periods of major global financial disruptions, as in 2008-2009.
    

