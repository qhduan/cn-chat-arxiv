# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Forecasted Treatment Effects.](http://arxiv.org/abs/2309.05639) | 本文考虑了在没有对照组的情况下估计和推断政策效果的问题。我们利用短期治疗前数据，基于预测反事实情况得到了个体治疗效果的无偏估计和平均治疗效果的一致且渐近正态的估计。我们发现，关注预测的无偏性而不是准确性是很重要的，而预测模型的正确规范并不是必需的，简单的基础函数回归可以达到无偏估计。在一定条件下，我们的方法具有一致性和渐近正态性。 |
| [^2] | [Testing for Stationary or Persistent Coefficient Randomness in Predictive Regressions.](http://arxiv.org/abs/2309.04926) | 本研究考虑了预测回归中系数随机性的检验，并发现在随机系数的持久性方面会影响各种检验的功效。我们建议在实际应用中根据潜在随机系数的持久性选择最合适的检验方法。 |
| [^3] | [News-driven Expectations and Volatility Clustering.](http://arxiv.org/abs/2309.04876) | 本文通过简单解释交易者对新闻的态度，解释了金融市场中的两个规律：尾重分布和波动聚类。 |
| [^4] | [Non-linear dimension reduction in factor-augmented vector autoregressions.](http://arxiv.org/abs/2309.04821) | 本文引入了非线性维度降低方法，在因子扩展向量自回归中分析经济冲击，证明了在经济周期动荡和高度波动数据下，该方法具有良好的预测性能，并能处理COVID-19疫情引起的离群值。 |
| [^5] | [Interpreting IV Estimators in Information Provision Experiments.](http://arxiv.org/abs/2309.04793) | 在信息提供实验中，使用信息提供作为工具变量可能无法产生正加权的信念效应的平均值。我们提出了一种移动工具变量框架和估计器，通过利用先验判断信念更新的方向，可以获得正加权的信念效应的平均值。 |
| [^6] | [Maintaining human wellbeing as socio-environmental systems undergo regime shifts.](http://arxiv.org/abs/2309.04578) | 本研究将能够闪烁的生态模型与人类适应模型相结合，从而探索闪烁对联合社会环境系统中人类福祉的影响，并重点研究了闪烁导致福祉下降的条件以及对转型变化的最佳时机的影响。 |
| [^7] | [Non-transitivity of the Win Ratio and Area Under the Receiver Operating Characteristics Curve (AUC): a case for evaluating the strength of stochastic comparisons.](http://arxiv.org/abs/2309.01791) | 本文报告并研究了胜率和接收器操作特性曲线下面积的长期不可转移行为，验证了传统统计量和胜率的差异，并强调了研究随机比较的相关性的重要性。 |
| [^8] | [Estimating the Value of Evidence-Based Decision Making.](http://arxiv.org/abs/2306.13681) | 本文提出了一个实证框架，用于估算证据决策的价值和统计精度投资回报。 |
| [^9] | [Price Discovery for Derivatives.](http://arxiv.org/abs/2302.13426) | 本研究提供了一个基本理论，研究了带有高阶信息的期权市场中价格的发现机制。与此同时，该研究还以内幕交易的形式呈现了其中的特例，给出了通货膨胀需求、价格冲击和信息效率的闭式解。 |
| [^10] | [Inside the West Wing: Lobbying as a contest.](http://arxiv.org/abs/2207.00800) | 这篇论文研究了当一个政府做出多个政策决策时，游说可以被看作是政府和特殊利益集团之间的竞争，并发现政府会通过给予特定利益集团特殊待遇来建立自己的政治资本，以在与其他利益集团的斗争中取得优势。 |
| [^11] | [The Shared Cost of Pursuing Shareholder Value.](http://arxiv.org/abs/2103.12138) | 文章使用股东大会时间差异的方法研究了股东偏好和对公司利他决策的影响，发现追求（某些）股东的价值具有分配成本，但大股东的监控可以避免这种由偏好异质性驱动的成本。 |

# 详细

[^1]: 预测治疗效果

    Forecasted Treatment Effects. (arXiv:2309.05639v1 [econ.EM])

    [http://arxiv.org/abs/2309.05639](http://arxiv.org/abs/2309.05639)

    本文考虑了在没有对照组的情况下估计和推断政策效果的问题。我们利用短期治疗前数据，基于预测反事实情况得到了个体治疗效果的无偏估计和平均治疗效果的一致且渐近正态的估计。我们发现，关注预测的无偏性而不是准确性是很重要的，而预测模型的正确规范并不是必需的，简单的基础函数回归可以达到无偏估计。在一定条件下，我们的方法具有一致性和渐近正态性。

    

    我们考虑在没有对照组的情况下估计和推断政策效果。我们基于使用一段短期治疗前数据预测反事实情况，得到了个体（异质）治疗效果的无偏估计和一致且渐近正态的平均治疗效果估计。我们表明，应该关注预测无偏性而不是准确性。对预测模型的正确规范并不是获得个体治疗效果的无偏估计所必需的。相反，在广泛的数据生成过程下，简单的基础函数（如多项式时间趋势）回归可以提供无偏性来估计个体反事实情况。基于模型的预测可能引入规范错误偏差，并且即使在正确规范下也不一定能提高性能。我们的预测平均治疗效果（FAT）估计器的一致性和渐近正态性在一定条件下得到保证。

    We consider estimation and inference of the effects of a policy in the absence of a control group. We obtain unbiased estimators of individual (heterogeneous) treatment effects and a consistent and asymptotically normal estimator of the average treatment effects, based on forecasting counterfactuals using a short time series of pre-treatment data. We show that the focus should be on forecast unbiasedness rather than accuracy. Correct specification of the forecasting model is not necessary to obtain unbiased estimates of individual treatment effects. Instead, simple basis function (e.g., polynomial time trends) regressions deliver unbiasedness under a broad class of data-generating processes for the individual counterfactuals. Basing the forecasts on a model can introduce misspecification bias and does not necessarily improve performance even under correct specification. Consistency and asymptotic normality of our Forecasted Average Treatment effects (FAT) estimator are attained under a
    
[^2]: 预测回归中固定系数随机性的检验：稳态与持久性系数的影响

    Testing for Stationary or Persistent Coefficient Randomness in Predictive Regressions. (arXiv:2309.04926v1 [econ.EM])

    [http://arxiv.org/abs/2309.04926](http://arxiv.org/abs/2309.04926)

    本研究考虑了预测回归中系数随机性的检验，并发现在随机系数的持久性方面会影响各种检验的功效。我们建议在实际应用中根据潜在随机系数的持久性选择最合适的检验方法。

    

    本研究考虑了预测回归中系数随机性的检验。我们关注系数随机性检验在随机系数的持久性方面的影响。我们发现，当随机系数是稳态的或I(0)时，Nyblom的LM检验在功效上不是最优的，这一点已经针对集成或I(1)随机系数的备择假设得到了证实。我们通过构建一些在随机系数为稳态时具有更高功效的检验来证明这一点，尽管在随机系数为集成时，这些检验在功效上被LM检验所支配。这意味着在不同的背景下，系数随机性的最佳检验是不同的，从而实证研究者应该考虑潜在随机系数的持久性，并相应地选择多个检验。特别是，我们通过理论和数值研究表明，LM检验与一种Wald型检验的乘积是一个较好的检验方法。

    This study considers tests for coefficient randomness in predictive regressions. Our focus is on how tests for coefficient randomness are influenced by the persistence of random coefficient. We find that when the random coefficient is stationary, or I(0), Nyblom's (1989) LM test loses its optimality (in terms of power), which is established against the alternative of integrated, or I(1), random coefficient. We demonstrate this by constructing tests that are more powerful than the LM test when random coefficient is stationary, although these tests are dominated in terms of power by the LM test when random coefficient is integrated. This implies that the best test for coefficient randomness differs from context to context, and practitioners should take into account the persistence of potentially random coefficient and choose from several tests accordingly. In particular, we show through theoretical and numerical investigations that the product of the LM test and a Wald-type test proposed
    
[^3]: 新闻驱动的期望与波动聚类

    News-driven Expectations and Volatility Clustering. (arXiv:2309.04876v1 [q-fin.GN])

    [http://arxiv.org/abs/2309.04876](http://arxiv.org/abs/2309.04876)

    本文通过简单解释交易者对新闻的态度，解释了金融市场中的两个规律：尾重分布和波动聚类。

    

    金融波动遵循两个引人注目的经验规律，适用于各种资产、各个市场和各个时间尺度：它是尾重的（更准确地说是符合幂律分布），并且在时间上倾向于聚类。许多有趣的模型已被提出来解释这些规律，特别是基于代理人的模型，通过一种复杂的非线性机制来模拟这两个经验定律，例如交易者在高度非线性的方式下在交易策略之间的切换。本文简单地解释了这两个规律，只涉及交易者对新闻的态度，这种解释几乎是根据金融市场参与者的传统二分法来定义的，投资者与投机者，他们的行为被简化为最简单的形式。假设长期投资者对资产的估值遵循基于新闻的随机漫步，因此捕捉到了投资者对基本新闻的持久、长期的记忆。

    Financial volatility obeys two fascinating empirical regularities that apply to various assets, on various markets, and on various time scales: it is fat-tailed (more precisely power-law distributed) and it tends to be clustered in time. Many interesting models have been proposed to account for these regularities, notably agent-based models, which mimic the two empirical laws through a complex mix of nonlinear mechanisms such as traders' switching between trading strategies in highly nonlinear way. This paper explains the two regularities simply in terms of traders' attitudes towards news, an explanation that follows almost by definition of the traditional dichotomy of financial market participants, investors versus speculators, whose behaviors are reduced to their simplest forms. Long-run investors' valuations of an asset are assumed to follow a news-driven random walk, thus capturing the investors' persistent, long memory of fundamental news. Short-term speculators' anticipated retur
    
[^4]: 非线性维度降低在因子扩展向量自回归中的应用

    Non-linear dimension reduction in factor-augmented vector autoregressions. (arXiv:2309.04821v1 [econ.EM])

    [http://arxiv.org/abs/2309.04821](http://arxiv.org/abs/2309.04821)

    本文引入了非线性维度降低方法，在因子扩展向量自回归中分析经济冲击，证明了在经济周期动荡和高度波动数据下，该方法具有良好的预测性能，并能处理COVID-19疫情引起的离群值。

    

    本文将非线性维度降低方法引入因子扩展向量自回归模型，分析不同经济冲击的影响。作者认为在经济周期动荡时，控制大维度数据集与潜在因子之间的非线性关系尤为有用。通过模拟实验证明，非线性维度降低技术在数据高度波动时具有良好的预测性能。在实证应用中，本文排除和包含COVID-19疫情观测，确定了货币政策和不确定性冲击。这两个应用表明非线性FAVAR方法能够处理COVID-19疫情引起的极大离群值，并在两种情景下获得可靠结果。

    This paper introduces non-linear dimension reduction in factor-augmented vector autoregressions to analyze the effects of different economic shocks. I argue that controlling for non-linearities between a large-dimensional dataset and the latent factors is particularly useful during turbulent times of the business cycle. In simulations, I show that non-linear dimension reduction techniques yield good forecasting performance, especially when data is highly volatile. In an empirical application, I identify a monetary policy as well as an uncertainty shock excluding and including observations of the COVID-19 pandemic. Those two applications suggest that the non-linear FAVAR approaches are capable of dealing with the large outliers caused by the COVID-19 pandemic and yield reliable results in both scenarios.
    
[^5]: 在信息提供实验中解释IV估计器

    Interpreting IV Estimators in Information Provision Experiments. (arXiv:2309.04793v1 [econ.EM])

    [http://arxiv.org/abs/2309.04793](http://arxiv.org/abs/2309.04793)

    在信息提供实验中，使用信息提供作为工具变量可能无法产生正加权的信念效应的平均值。我们提出了一种移动工具变量框架和估计器，通过利用先验判断信念更新的方向，可以获得正加权的信念效应的平均值。

    

    越来越多的文献使用信息提供实验来衡量“信念效应”——即信念变化对行为的影响——其中信息提供被用作信念的工具变量。我们展示了在具有异质信念效应的被动控制设计实验中，使用信息提供作为工具变量可能无法产生信念效应的正加权平均。我们提出了一种“移动工具变量”（MIV）框架和估计器，通过利用先验判断信念更新的方向，可以获得信念效应的正加权平均。与文献中常用的规范相比，我们的首选MIV可以过分加权具有较大先验误差的个体；此外，一些规范可能需要额外的假设才能产生正加权。

    A growing literature measures "belief effects" -- that is, the effect of a change in beliefs on one's actions -- using information provision experiments, where the provision of information is used as an instrument for beliefs. We show that in passive control design experiments with heterogeneous belief effects, using information provision as an instrument may not produce a positive weighted average of belief effects. We propose a "mover instrumental variables" (MIV) framework and estimator that attains a positive weighted average of belief effects by inferring the direction of belief updating using the prior. Relative to our preferred MIV, commonly used specifications in the literature produce a form of MIV that overweights individuals with larger prior errors; additionally, some specifications may require additional assumptions to generate positive weights.
    
[^6]: 在社会环境系统发生破局转变时维系人类福祉

    Maintaining human wellbeing as socio-environmental systems undergo regime shifts. (arXiv:2309.04578v1 [econ.TH])

    [http://arxiv.org/abs/2309.04578](http://arxiv.org/abs/2309.04578)

    本研究将能够闪烁的生态模型与人类适应模型相结合，从而探索闪烁对联合社会环境系统中人类福祉的影响，并重点研究了闪烁导致福祉下降的条件以及对转型变化的最佳时机的影响。

    

    全球环境变化正推动许多社会环境系统朝着临界阈值迈进，在这些阈值上，生态系统状态处于临界转折点, 需要干预来引导或避免即将发生的转变。闪烁是指系统在不同稳定状态之间摇摆的现象，被认为是对向不可取的生态制度进行无法逆转的转变的有用预警信号。然而，尽管闪烁可能预示一个生态转折点，这些动态也给人类的适应带来了独特的挑战。在这项工作中，我们将能够闪烁的生态模型与一个模拟人类对变化环境适应的模型联系起来。这使得我们能够探索闪烁对联合社会环境系统中自适应主体的效用的影响。我们重点研究了闪烁导致福祉不成比例下降的条件，并探讨了这些动态对转型变化的最佳时机的影响。

    Global environmental change is pushing many socio-environmental systems towards critical thresholds, where ecological systems' states are on the precipice of tipping points and interventions are needed to navigate or avert impending transitions. Flickering, where a system vacillates between alternative stable states, is touted as a useful early warning signal of irreversible transitions to undesirable ecological regimes. However, while flickering may presage an ecological tipping point, these dynamics also pose unique challenges for human adaptation. In this work, we link an ecological model that can exhibit flickering to a model of human adaptation to a changing environment. This allows us to explore the impact of flickering on the utility of adaptive agents in a coupled socio-environmental system. We highlight the conditions under which flickering causes wellbeing to decline disproportionately, and explore how these dynamics impact the optimal timing of a transformational change that
    
[^7]: 不可转移的胜率和接收器操作特性曲线下面积 (AUC) ：评估随机比较的相关性 (arXiv:2309.01791v1 [stat.ME])

    Non-transitivity of the Win Ratio and Area Under the Receiver Operating Characteristics Curve (AUC): a case for evaluating the strength of stochastic comparisons. (arXiv:2309.01791v1 [stat.ME])

    [http://arxiv.org/abs/2309.01791](http://arxiv.org/abs/2309.01791)

    本文报告并研究了胜率和接收器操作特性曲线下面积的长期不可转移行为，验证了传统统计量和胜率的差异，并强调了研究随机比较的相关性的重要性。

    

    胜率 (WR) 是随机对照试验中使用的一种新颖统计量，可以考虑事件结果内的层次关系。在本文中，我们报告并研究了胜率和紧密相关的接收器操作特性曲线下面积 (AUC) 的长期不可转移行为，并认为它们的可转移性不能被视为理所当然。关键是，传统的组内统计量（即平均值比较）始终是可转移的，而胜率可以检测到不可转移性。不可转移性提供了关于两个治疗组之间随机关系的有价值的信息，应该进行测试和报告。我们确定了可转移性的必要条件、不可转移性的充分条件，并在真实的大规模随机对照试验中展示了胜率对死亡时间的不可转移性。我们的结果可用于排除或评估不可转移性的可能性，并展示研究随机比较的相关性的重要性。

    The win ratio (WR) is a novel statistic used in randomized controlled trials that can account for hierarchies within event outcomes. In this paper we report and study the long-run non-transitive behavior of the win ratio and the closely related Area Under the Receiver Operating Characteristics Curve (AUC) and argue that their transitivity cannot be taken for granted. Crucially, traditional within-group statistics (i.e., comparison of means) are always transitive, while the WR can detect non-transitivity. Non-transitivity provides valuable information on the stochastic relationship between two treatment groups, which should be tested and reported. We specify the necessary conditions for transitivity, the sufficient conditions for non-transitivity and demonstrate non-transitivity in a real-life large randomized controlled trial for the WR of time-to-death. Our results can be used to rule out or evaluate possibility of non-transitivity and show the importance of studying the strength of s
    
[^8]: 估算基于证据决策的价值

    Estimating the Value of Evidence-Based Decision Making. (arXiv:2306.13681v1 [stat.ME])

    [http://arxiv.org/abs/2306.13681](http://arxiv.org/abs/2306.13681)

    本文提出了一个实证框架，用于估算证据决策的价值和统计精度投资回报。

    

    商业/政策决策通常基于随机实验和观察性研究的证据。本文提出了一个实证框架来估算基于证据的决策（EBDM）的价值和统计精度投资回报。

    Business/policy decisions are often based on evidence from randomized experiments and observational studies. In this article we propose an empirical framework to estimate the value of evidence-based decision making (EBDM) and the return on the investment in statistical precision.
    
[^9]: 期权的价格发现

    Price Discovery for Derivatives. (arXiv:2302.13426v5 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2302.13426](http://arxiv.org/abs/2302.13426)

    本研究提供了一个基本理论，研究了带有高阶信息的期权市场中价格的发现机制。与此同时，该研究还以内幕交易的形式呈现了其中的特例，给出了通货膨胀需求、价格冲击和信息效率的闭式解。

    

    本文通过一个模型，考虑了私有信息和高阶信息对期权市场价格的影响。模型允许有私有信息的交易者在状态-索赔集市场上交易。等价的期权形式下，我们考虑了拥有关于基础资产收益的分布的私有信息，并允许交易任意期权组合的操纵者。我们得出了通货膨胀需求、价格冲击和信息效率的闭式解，这些解提供了关于内幕交易的高阶信息，如任何给定的时刻交易期权策略，并将这些策略泛化到了波动率交易等实践领域。

    We obtain a basic theory of price discovery across derivative markets with respect to higher-order information, using a model where an agent with general private information regarding state probabilities is allowed to trade arbitrary portfolios of state-contingent claims. In an equivalent options formulation, the informed agent has private information regarding arbitrary aspects of the payoff distribution of an underlying asset and is allowed to trade arbitrary option portfolios. We characterize, in closed form, the informed demand, price impact, and information efficiency of prices. Our results offer a theory of insider trading on higher moments of the underlying payoff as a special case. The informed demand formula prescribes option strategies for trading on any given moment and extends those used in practice for, e.g. volatility trading.
    
[^10]: 西翼内幕：游说作为一场竞赛

    Inside the West Wing: Lobbying as a contest. (arXiv:2207.00800v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2207.00800](http://arxiv.org/abs/2207.00800)

    这篇论文研究了当一个政府做出多个政策决策时，游说可以被看作是政府和特殊利益集团之间的竞争，并发现政府会通过给予特定利益集团特殊待遇来建立自己的政治资本，以在与其他利益集团的斗争中取得优势。

    

    当政府做出许多不同的政策决策时，游说可以被视为政府和许多特殊利益集团之间的竞赛。政府通过其自身的政治资本与利益集团进行斗争。在这个世界中，我们发现政府想要“出售保护”，即在交换捐款的同时给予特定利益集团有利待遇。它这样做是为了建立自己的“战争基金”，提高在与其他利益集团争斗中的地位。直到它能够确定地赢得所有剩余的竞争。这与现有模型的观点形成鲜明对比，后者通常将游说视为信息或代理问题驱动的。

    When a government makes many different policy decisions, lobbying can be viewed as a contest between the government and many different special interest groups. The government fights lobbying by interest groups with its own political capital. In this world, we find that a government wants to `sell protection' -- give favourable treatment in exchange for contributions -- to certain interest groups. It does this in order to build its own `war chest' of political capital, which improves its position in fights with other interest groups. And it does so until it wins all remaining contests with certainty. This stands in contrast to existing models that often view lobbying as driven by information or agency problems.
    
[^11]: 追求股东价值的共同成本

    The Shared Cost of Pursuing Shareholder Value. (arXiv:2103.12138v9 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2103.12138](http://arxiv.org/abs/2103.12138)

    文章使用股东大会时间差异的方法研究了股东偏好和对公司利他决策的影响，发现追求（某些）股东的价值具有分配成本，但大股东的监控可以避免这种由偏好异质性驱动的成本。

    

    本文采用准实验性的方法，根据公司股东大会（AGMs）的时间差异，提出了一个可移植的框架，推断股东的偏好和对公司利他决策的影响，并将其应用于covid相关捐赠、最近针对俄罗斯的私人制裁以及公司2012-19年的利他立场。AGMs的媒体曝光带来的形象收益，使得与公司同义的股东（如密切相关的个人）支持昂贵的利他变革，而其他股东（如金融公司）反对这些变革。支持这些变革的影响使收益下降了30％：追求（某些）股东的价值具有分配成本，大股东的监控可以避免由偏好异质性驱动的成本。

    Using quasi-experimental variations from the timing of firms' Annual General Meetings (AGMs), we propose a portable framework to infer shareholders' preferences and influences on firms' prosocial decisions and apply it to covid-related donations, recent private sanctions on Russia, and firms' prosocial stances over 2012-19. Image gains from AGMs' media exposure drive shareholders synonymous with a firm, like closely-connected individuals, to support costly prosocial changes, while others, like financial corporations, oppose them. Influence supporting these changes lowers earnings by 30\%: pursuing the values of (some) shareholders has distributional costs, which the monitoring of large shareholders motivated by heterogeneous preferences could prevent.
    

