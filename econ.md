# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structural Periodic Vector Autoregressions.](http://arxiv.org/abs/2401.14545) | 本文提出了一种新的方法，通过建模非季节性调整的原始数据的周期性特征，实现了直接的结构推断，从而可以捕捉季节效应并获得更精细的结果。 |
| [^2] | [The Power of Tests for Detecting $p$-Hacking.](http://arxiv.org/abs/2205.07950) | 本文从理论和模拟的层面，研究了$p$-hacking的可能形式对$p-value$分布和现有检测方法的影响，证明了测试揭示“p-hacking”的能力可能非常低，并且关键取决于具体的$p$-hacking策略和研究测试的实际效果的分布。 |

# 详细

[^1]: 结构性周期性向量自回归

    Structural Periodic Vector Autoregressions. (arXiv:2401.14545v1 [econ.EM])

    [http://arxiv.org/abs/2401.14545](http://arxiv.org/abs/2401.14545)

    本文提出了一种新的方法，通过建模非季节性调整的原始数据的周期性特征，实现了直接的结构推断，从而可以捕捉季节效应并获得更精细的结果。

    

    在使用结构变量自回归（SVAR）处理经过季节调整的宏观经济数据之前，通常需要通过季节调整技术去除原始数据中的季节性。然而，这种方法可能会扭曲数据中所包含的有价值信息。作为一种替代方法，本文提出了一种直接建模非季节性调整的原始数据周期性特征的方法，即结构性周期性向量自回归（SPVAR），其基于周期性向量自回归（PVAR）作为约化形式模型。与VAR相比，PVAR不仅可以允许周期性变化的截距，还可以允许周期性自回归参数和创新方差。由于这种更大的灵活性也导致参数数量增加，因此我们提出了线性约束估计技术。总体而言，SPVAR可以捕捉季节效应，并实现直接和更精细的结构推断。

    While seasonality inherent to raw macroeconomic data is commonly removed by seasonal adjustment techniques before it is used for structural inference, this approach might distort valuable information contained in the data. As an alternative method to commonly used structural vector autoregressions (SVAR) for seasonally adjusted macroeconomic data, this paper offers an approach in which the periodicity of not seasonally adjusted raw data is modeled directly by structural periodic vector autoregressions (SPVAR) that are based on periodic vector autoregressions (PVAR) as the reduced form model. In comparison to a VAR, the PVAR does allow not only for periodically time-varying intercepts, but also for periodic autoregressive parameters and innovations variances, respectively. As this larger flexibility leads also to an increased number of parameters, we propose linearly constrained estimation techniques. Overall, SPVARs allow to capture seasonal effects and enable a direct and more refined
    
[^2]: 测试揭示“p-hacking”的能力

    The Power of Tests for Detecting $p$-Hacking. (arXiv:2205.07950v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2205.07950](http://arxiv.org/abs/2205.07950)

    本文从理论和模拟的层面，研究了$p$-hacking的可能形式对$p-value$分布和现有检测方法的影响，证明了测试揭示“p-hacking”的能力可能非常低，并且关键取决于具体的$p$-hacking策略和研究测试的实际效果的分布。

    

    $p$-Hacking可能会削弱经验研究的有效性。一篇繁荣的经验文献调查了基于报告的$p$-value在研究中的分布的$p$-hacking的普遍存在。解释这个文献中的结果需要仔细理解用于检测不同类型$p$-hacking的方法的能力。我们从理论上研究了$p$-hacking的可能形式对报告的$p$-value分布和现有检测方法的能力的影响。能力可能非常低，关键取决于特定的$p$-hacking策略和研究测试的实际效果的分布。出版偏差可以增强测试无$p$-hacking和无出版偏差的联合零假设的能力。我们将测试的能力与$p$-hacking的成本相关联，并显示当$p$-hacking非常昂贵时，能力倾向于更大。蒙特卡罗模拟支持我们的理论结果。

    $p$-Hacking can undermine the validity of empirical studies. A flourishing empirical literature investigates the prevalence of $p$-hacking based on the empirical distribution of reported $p$-values across studies. Interpreting results in this literature requires a careful understanding of the power of methods used to detect different types of $p$-hacking. We theoretically study the implications of likely forms of $p$-hacking on the distribution of reported $p$-values and the power of existing methods for detecting it. Power can be quite low, depending crucially on the particular $p$-hacking strategy and the distribution of actual effects tested by the studies. Publication bias can enhance the power for testing the joint null hypothesis of no $p$-hacking and no publication bias. We relate the power of the tests to the costs of $p$-hacking and show that power tends to be larger when $p$-hacking is very costly. Monte Carlo simulations support our theoretical results.
    

