# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Information Elicitation in Agency Games](https://arxiv.org/abs/2402.14005) | 该论文探讨了在代理游戏中如何通过代理的信息揭示来指导度量标准的选择，以解决信息不足的问题。 |
| [^2] | [Shall We Talk: Exploring Spontaneous Collaborations of Competing LLM Agents](https://arxiv.org/abs/2402.12327) | 该研究揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力，验证了计算社会科学的愿景，表明LLM代理可以用于模拟人类社会互动，包括自发合作的互动，为社会现象提供洞察。 |
| [^3] | [Moen Meets Rotemberg: An Earthly Model of the Divine Coincidence.](http://arxiv.org/abs/2401.12475) | 本文提出了一个关于神圣巧合的模型，解释了其在美国数据中的最近出现。该模型通过Moen的有向搜索和Rotemberg的价格调整成本，建立了失业与通胀之间的菲利普斯曲线，并保证了充分就业和价格稳定任务的一致性。 |
| [^4] | [Tournaments, Contestant Heterogeneity and Performance.](http://arxiv.org/abs/2401.05210) | 本文使用实地数据研究发现，选手之间的技能差异对绩效产生不对称影响，对低能选手有负面影响，对高能选手有正面影响。同时讨论了行为方法的解释和竞赛设计的最优影响。 |
| [^5] | [Covariate Shift Adaptation Robust to Density-Ratio Estimation.](http://arxiv.org/abs/2310.16638) | 该论文研究了在协变量偏移下的密度比估计的罕见问题，提出了一种适应性方法来减轻密度比估计的偏差对模型的影响。 |
| [^6] | [Econotaxis in modeling urbanization by labor force migration.](http://arxiv.org/abs/2303.09720) | 本研究提出了一个劳动力迁移模型，通过模拟发现模型可以产生聚集行为，并展现了两个经验规律。更进一步，研究证明了经济性趋向性，这是一种新型人类行为中心的趋向性，可以解释在现实世界中的劳动力聚集现象，这一结论突显了城市化与所导出的PDE系统中的吹起现象的相关性。 |

# 详细

[^1]: 代理游戏中的信息引导

    Information Elicitation in Agency Games

    [https://arxiv.org/abs/2402.14005](https://arxiv.org/abs/2402.14005)

    该论文探讨了在代理游戏中如何通过代理的信息揭示来指导度量标准的选择，以解决信息不足的问题。

    

    在数据收集和数据处理方面的可扩展化、通用化工具取得了快速进展，这使得公司和政策制定者能够采用更加复杂的度量标准作为决策指南成为可能。然而，决定计算哪些度量标准仍然是一个挑战，尤其是对于那些尚未知晓的信息限制下，公司的广泛计算度量标准的能力并不能解决“未知的未知”问题。为了在面对这种信息问题时指导度量标准的选择，我们转向评估代理本身，他们可能比委托人拥有更多关于如何有效衡量结果的信息。我们将这种互动建模为一个简单的代理游戏，询问：代理何时有动机向委托人揭示与成本相关变量的可观察性？存在两种效应：更好的信息降低了代理商的不透明度

    arXiv:2402.14005v1 Announce Type: cross  Abstract: Rapid progress in scalable, commoditized tools for data collection and data processing has made it possible for firms and policymakers to employ ever more complex metrics as guides for decision-making. These developments have highlighted a prevailing challenge -- deciding *which* metrics to compute. In particular, a firm's ability to compute a wider range of existing metrics does not address the problem of *unknown unknowns*, which reflects informational limitations on the part of the firm. To guide the choice of metrics in the face of this informational problem, we turn to the evaluated agents themselves, who may have more information than a principal about how to measure outcomes effectively. We model this interaction as a simple agency game, where we ask: *When does an agent have an incentive to reveal the observability of a cost-correlated variable to the principal?* There are two effects: better information reduces the agent's inf
    
[^2]: 我们应该交流吗：探索竞争LLM代理之间的自发合作

    Shall We Talk: Exploring Spontaneous Collaborations of Competing LLM Agents

    [https://arxiv.org/abs/2402.12327](https://arxiv.org/abs/2402.12327)

    该研究揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力，验证了计算社会科学的愿景，表明LLM代理可以用于模拟人类社会互动，包括自发合作的互动，为社会现象提供洞察。

    

    最近的进展表明，由大型语言模型（LLMs）驱动的代理具有模拟人类行为和社会动态的能力。然而，尚未研究LLM代理在没有明确指令的情况下自发建立合作关系的潜力。为了弥补这一空白，我们进行了三项案例研究，揭示了LLM代理甚至在竞争环境中也能自发形成合作关系的能力。这一发现不仅展示了LLM代理模拟人类社会中竞争与合作的能力，也验证了计算社会科学的一个有前途的愿景。具体来说，这表明LLM代理可以用于建模人类社会互动，包括那些自发合作的互动，从而提供对社会现象的洞察。这项研究的源代码可在https://github.com/wuzengqing001225/SABM_ShallWe 找到。

    arXiv:2402.12327v1 Announce Type: new  Abstract: Recent advancements have shown that agents powered by large language models (LLMs) possess capabilities to simulate human behaviors and societal dynamics. However, the potential for LLM agents to spontaneously establish collaborative relationships in the absence of explicit instructions has not been studied. To address this gap, we conduct three case studies, revealing that LLM agents are capable of spontaneously forming collaborations even within competitive settings. This finding not only demonstrates the capacity of LLM agents to mimic competition and cooperation in human societies but also validates a promising vision of computational social science. Specifically, it suggests that LLM agents could be utilized to model human social interactions, including those with spontaneous collaborations, thus offering insights into social phenomena. The source codes for this study are available at https://github.com/wuzengqing001225/SABM_ShallWe
    
[^3]: Moen遇到Rotemberg: 一个关于神圣巧合的实际模型

    Moen Meets Rotemberg: An Earthly Model of the Divine Coincidence. (arXiv:2401.12475v1 [econ.TH])

    [http://arxiv.org/abs/2401.12475](http://arxiv.org/abs/2401.12475)

    本文提出了一个关于神圣巧合的模型，解释了其在美国数据中的最近出现。该模型通过Moen的有向搜索和Rotemberg的价格调整成本，建立了失业与通胀之间的菲利普斯曲线，并保证了充分就业和价格稳定任务的一致性。

    

    本文提出了一个关于神圣巧合的模型，解释了其在美国数据中的最近出现。神圣巧合的重要性在于，它有助于解释疫情后通胀的行为，并保证了美联储的充分就业和价格稳定任务的一致性。在该模型中，失业与通胀之间的菲利普斯曲线源自Moen（1997）的有向搜索。由于Rotemberg（1982）的价格调整成本，该菲利普斯曲线是非垂直的。该模型的菲利普斯曲线保证了在失业率有效时通胀率保持在目标水平上，从而产生了神圣巧合。如果我们假设工资的降低（降低了员工的士气）对生产者来说比价格的上涨（使顾客不满意）更加成本高昂，那么菲利普斯曲线在神圣巧合点也会显示出一个拐点。

    This paper proposes a model of the divine coincidence, explaining its recent appearance in US data. The divine coincidence matters because it helps explain the behavior of inflation after the pandemic, and it guarantees that the full-employment and price-stability mandates of the Federal Reserve coincide. In the model, a Phillips curve relating unemployment to inflation arises from Moen's (1997) directed search. The Phillips curve is nonvertical thanks to Rotemberg's (1982) price-adjustment costs. The model's Phillips curve guarantees that the rate of inflation is on target whenever the rate of unemployment is efficient, generating the divine coincidence. If we assume that wage decreases -- which reduce workers' morale -- are more costly to producers than price increases -- which upset customers -- the Phillips curve also displays a kink at the point of divine coincidence.
    
[^4]: 锦标赛、选手异质性和绩效

    Tournaments, Contestant Heterogeneity and Performance. (arXiv:2401.05210v1 [econ.GN])

    [http://arxiv.org/abs/2401.05210](http://arxiv.org/abs/2401.05210)

    本文使用实地数据研究发现，选手之间的技能差异对绩效产生不对称影响，对低能选手有负面影响，对高能选手有正面影响。同时讨论了行为方法的解释和竞赛设计的最优影响。

    

    锦标赛经常被用作激励机制来提高绩效。本文使用实地数据，并展示了选手之间的技能差异对选手绩效的不对称影响。技能差异对低能选手的绩效有负面影响，但对高能选手的绩效有正面影响。我们讨论了不同行为方法来解释我们的研究结果，并讨论了结果对竞赛的最优设计的影响。此外，我们的研究揭示了两个重要的实证结果：(a) 象争取平权政策可能有助于减轻低能选手的不利影响，(b) 后续比赛阶段潜在未来选手的技能水平可能对高能选手的绩效产生不利影响，但不会影响低能选手。

    Tournaments are frequently used incentive mechanisms to enhance performance. In this paper, we use field data and show that skill disparities among contestants asymmetrically affect the performance of contestants. Skill disparities have detrimental effects on the performance of the lower-ability contestant but positive effects on the performance of the higher-ability contestant. We discuss the potential of different behavioral approaches to explain our findings and discuss the implications of our results for the optimal design of contests. Beyond that, our study reveals two important empirical results: (a) affirmative action-type policies may help to mitigate the adverse effects on lower-ability contestants, and (b) the skill level of potential future contestants in subsequent tournament stages can detrimentally influence the performance of higher-ability contestants but does not affect the lower-ability contestant.
    
[^5]: 适应密度比估计的协变量偏移适应

    Covariate Shift Adaptation Robust to Density-Ratio Estimation. (arXiv:2310.16638v1 [stat.ME])

    [http://arxiv.org/abs/2310.16638](http://arxiv.org/abs/2310.16638)

    该论文研究了在协变量偏移下的密度比估计的罕见问题，提出了一种适应性方法来减轻密度比估计的偏差对模型的影响。

    

    在一种情况下，我们可以访问具有协变量和结果的训练数据，而测试数据只包含协变量。在这种情况下，我们的主要目标是预测测试数据中缺失的结果。为了实现这个目标，我们在协变量偏移下训练参数回归模型，其中训练数据和测试数据之间的协变量分布不同。对于这个问题，现有研究提出了通过使用密度比的重要性加权来进行协变量偏移适应的方法。该方法通过对训练数据损失进行加权平均，每个权重是训练数据和测试数据之间的协变量密度比的估计，以近似测试数据的风险。尽管它允许我们获得一个最小化测试数据风险的模型，但其性能严重依赖于密度比估计的准确性。此外，即使密度比可以一致地估计，密度比的估计误差也会导致回归模型的估计器产生偏差。

    Consider a scenario where we have access to train data with both covariates and outcomes while test data only contains covariates. In this scenario, our primary aim is to predict the missing outcomes of the test data. With this objective in mind, we train parametric regression models under a covariate shift, where covariate distributions are different between the train and test data. For this problem, existing studies have proposed covariate shift adaptation via importance weighting using the density ratio. This approach averages the train data losses, each weighted by an estimated ratio of the covariate densities between the train and test data, to approximate the test-data risk. Although it allows us to obtain a test-data risk minimizer, its performance heavily relies on the accuracy of the density ratio estimation. Moreover, even if the density ratio can be consistently estimated, the estimation errors of the density ratio also yield bias in the estimators of the regression model's 
    
[^6]: 劳动力迁移模拟中的经济性趋向性

    Econotaxis in modeling urbanization by labor force migration. (arXiv:2303.09720v1 [nlin.AO])

    [http://arxiv.org/abs/2303.09720](http://arxiv.org/abs/2303.09720)

    本研究提出了一个劳动力迁移模型，通过模拟发现模型可以产生聚集行为，并展现了两个经验规律。更进一步，研究证明了经济性趋向性，这是一种新型人类行为中心的趋向性，可以解释在现实世界中的劳动力聚集现象，这一结论突显了城市化与所导出的PDE系统中的吹起现象的相关性。

    

    本研究采用主动布朗粒子框架，提出了一个简单的劳动力迁移微观模型。通过基于代理的模拟，我们发现我们的模型产生了从随机初始分布中聚集到一起的一群代理。此外，在我们的模型中观察到了Zipf和Okun定律这两个经验规律。为了揭示产生的聚集现象背后的机制，我们从我们的微观模型中导出了一个扩展的Keller-Segel系统。得到的宏观系统表明人力资源在现实世界中的聚集可以通过一种新型人类行为中心的趋向性来解释，这突显了城市化与所导出的PDE系统中的吹起现象的相关性。我们将其称为“经济性趋向性”。

    Individual participants in human society collectively exhibit aggregation behavior. In this study, we present a simple microscopic model of labor force migration by employing the active Brownian particles framework. Through agent-based simulations, we find that our model produces clusters of agents from a random initial distribution. Furthermore, two empirical regularities called Zipf's and Okun's laws were observed in our model. To reveal the mechanism underlying the reproduced agglomeration phenomena, we derived an extended Keller-Segel system, a classic model that describes the aggregation behavior of biological organisms called "taxis," from our microscopic model. The obtained macroscopic system indicates that the agglomeration of the workforce in real world can be accounted for through a new type of taxis central to human behavior, which highlights the relevance of urbanization to blow-up phenomena in the derived PDE system. We term it "econotaxis."
    

