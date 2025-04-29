# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Causal Q-Aggregation for CATE Model Selection.](http://arxiv.org/abs/2310.16945) | 该论文提出了一种基于Q集成的CATE模型选择方法，其通过使用双重鲁棒损失实现了统计上的最佳预测模型选择遗憾率 |
| [^2] | [Preference Evolution under Stable Matching.](http://arxiv.org/abs/2304.11504) | 该研究提出了一个模型，结合稳定匹配和均衡概念，研究了内生匹配与偏好演变。研究表明，在偏见主义的影响下，适度的偏见高效型在演化中能够强制实现正向择偶和高效互动，在信息不完全的情况下，排他性高效偏好型成为稳定的均衡结果。 |
| [^3] | [Semiparametric Conditional Factor Models: Estimation and Inference.](http://arxiv.org/abs/2112.07121) | 本文引入了一种简单且可行的估计方法，用于具有潜在因子的半参数条件因子模型。应用于美国股票收益数据，发现了大量非零定价误差并记录了随时间的下降特征。 |

# 详细

[^1]: Causal Q-Aggregation for CATE Model Selection（CATE模型选择中的因果Q集成）

    Causal Q-Aggregation for CATE Model Selection. (arXiv:2310.16945v1 [stat.ML])

    [http://arxiv.org/abs/2310.16945](http://arxiv.org/abs/2310.16945)

    该论文提出了一种基于Q集成的CATE模型选择方法，其通过使用双重鲁棒损失实现了统计上的最佳预测模型选择遗憾率

    

    准确估计条件平均处理效应（CATE）是个性化决策的核心。尽管有大量用于CATE估计的模型，但由于因果推断的基本问题，模型选择是一项非常棘手的任务。最近的实证工作提供了有利于具有双重鲁棒性质的代理损失度量和模型集成的证据。然而，对于这些模型的理论理解还不够。直接应用先前的理论工作会由于模型选择问题的非凸性而导致次优的预测模型选择率。我们提供了现有主要CATE集成方法的遗憾率，并提出了一种基于双重鲁棒损失的Q集成的新的CATE模型集成方法。我们的主要结果表明，因果Q集成在预测模型选择的遗憾率上达到了统计上的最优值为$\frac{\log(M)}{n}$（其中$M$为模型数，$n$为样本数），加上高阶估计误差项

    Accurate estimation of conditional average treatment effects (CATE) is at the core of personalized decision making. While there is a plethora of models for CATE estimation, model selection is a nontrivial task, due to the fundamental problem of causal inference. Recent empirical work provides evidence in favor of proxy loss metrics with double robust properties and in favor of model ensembling. However, theoretical understanding is lacking. Direct application of prior theoretical work leads to suboptimal oracle model selection rates due to the non-convexity of the model selection problem. We provide regret rates for the major existing CATE ensembling approaches and propose a new CATE model ensembling approach based on Q-aggregation using the doubly robust loss. Our main result shows that causal Q-aggregation achieves statistically optimal oracle model selection regret rates of $\frac{\log(M)}{n}$ (with $M$ models and $n$ samples), with the addition of higher-order estimation error term
    
[^2]: 稳定匹配下的偏好演变模型

    Preference Evolution under Stable Matching. (arXiv:2304.11504v1 [econ.TH])

    [http://arxiv.org/abs/2304.11504](http://arxiv.org/abs/2304.11504)

    该研究提出了一个模型，结合稳定匹配和均衡概念，研究了内生匹配与偏好演变。研究表明，在偏见主义的影响下，适度的偏见高效型在演化中能够强制实现正向择偶和高效互动，在信息不完全的情况下，排他性高效偏好型成为稳定的均衡结果。

    

    我们提出了一个模型，研究了内生匹配与偏好演变。在短期内，个体的主观偏好同时确定了他们匹配的对象和他们与配对伙伴之间的社会互动方式，从而为他们带来物质回报。物质回报反过来影响了偏好的长期演变。为了恰当地模拟“匹配-互动”过程，我们结合了稳定匹配和均衡概念。我们的研究强调了偏见主义，即与自己类似的人进行匹配的偏好，对塑造我们的结果的重要性。在完全信息下，拥有一种弱偏见主义和效率偏好的偏见高效型 - 在演化过程中脱颖而出，因为它能够强制个体之间进行正向择偶和高效互动。在信息不完全的情况下，排他性高效偏好型 - 具有强偏见主义和效率偏好，但不具备与自己相匹配的偏好 - 成为稳定的均衡结果。

    We present a model that investigates preference evolution with endogenous matching. In the short run, individuals' subjective preferences simultaneously determine who they are matched with and how they behave in the social interactions with their matched partners, which results in material payoffs for them. Material payoffs in turn affect how preferences evolve in the long run. To properly model the "match-to-interact" process, we combine stable matching and equilibrium concepts. Our findings emphasize the importance of parochialism, a preference for matching with one's own kind, in shaping our results. Under complete information, the parochial efficient preference type -characterized by a weak form of parochialism and a preference for efficiency -stands out in the evolutionary process, because it is able to force positive assortative matching and efficient play among individuals carrying this preference type. Under incomplete information, the exclusionary efficient preference type
    
[^3]: 半参数条件因子模型：估计和推断

    Semiparametric Conditional Factor Models: Estimation and Inference. (arXiv:2112.07121v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2112.07121](http://arxiv.org/abs/2112.07121)

    本文引入了一种简单且可行的估计方法，用于具有潜在因子的半参数条件因子模型。应用于美国股票收益数据，发现了大量非零定价误差并记录了随时间的下降特征。

    

    本文引入了一种简单且可行的筛选估计方法，用于具有潜在因子的半参数条件因子模型。我们在不要求大 $T$ 的情况下建立了估计量的大-$N$渐近性质。我们还开发了一种简单的自助法来进行关于条件定价误差以及因子载荷函数形状的推断。这些结果使我们能够利用任意非线性特征函数来估计大量个体资产的条件因子结构，而不需要预先指定因子，同时允许我们区分特征在捕捉因子 beta 和 alpha（即不可分散风险和错定价）中的作用。我们将这些方法应用于美国个别股票收益的截面，并发现了大量非零定价误差的强有力证据，这些误差结合起来产生的套利组合具有超过3的夏普比率。我们还记录了明显的随时间下降的特征。

    This paper introduces a simple and tractable sieve estimation of semiparametric conditional factor models with latent factors. We establish large-$N$-asymptotic properties of the estimators without requiring large $T$. We also develop a simple bootstrap procedure for conducting inference about the conditional pricing errors as well as the shapes of the factor loading functions. These results enable us to estimate conditional factor structure of a large set of individual assets by utilizing arbitrary nonlinear functions of a number of characteristics without the need to pre-specify the factors, while allowing us to disentangle the characteristics' role in capturing factor betas from alphas (i.e., undiversifiable risk from mispricing). We apply these methods to the cross-section of individual U.S. stock returns and find strong evidence of large nonzero pricing errors that combine to produce arbitrage portfolios with Sharpe ratios above 3. We also document a significant decline in apparen
    

