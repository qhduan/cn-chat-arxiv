# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Minimax-Regret Sample Selection in Randomized Experiments](https://arxiv.org/abs/2403.01386) | 通过最小后悔框架，提出了在随机试验中优化样本选择以实现异质人群中最佳福利的问题，并在不同条件下推导出最优的样本选择方案。 |
| [^2] | [Random partitions, potential of the Shapley value, and games with externalities](https://arxiv.org/abs/2402.00394) | 本论文研究了博弈的潜力函数，发现了一种与Macho-Stadler等人的MPW解相关的独特潜力函数，它通过玩家随机划分的期望累积价值计算，推广了无外部性博弈的潜力，并且在存在外部性的情况下仍满足空玩家特性。 |
| [^3] | [Causal Models for Longitudinal and Panel Data: A Survey](https://arxiv.org/abs/2311.15458) | 该调研总结了最近关于因果面板数据的文献，重点是在具有纵向数据的情况下，可靠地估计二元干预的因果效应，并强调了对实证研究人员的实用建议。 |
| [^4] | [Flexible heat pumps: must-have or nice to have in a power sector with renewables?.](http://arxiv.org/abs/2307.12918) | 本研究使用开源的电力部门模型，研究了2030年德国大规模扩张分散式热泵对电力部门的影响。结果表明，热泵的推广可以通过太阳能光伏等可再生能源实现，且额外的备用容量和电力储存需求有限。 |
| [^5] | [Heterogeneous Autoregressions in Short T Panel Data Models.](http://arxiv.org/abs/2306.05299) | 本文研究了带有个体特定效应和异质性自回归系数的一阶自回归面板数据模型，并提出了估计自回归系数横截面分布矩的方法，结果表明标准广义矩估计器是有偏的。本文还比较了在均匀和异质性斜率下的小样本性质。该研究可应用于收入决定的经济分析。 |
| [^6] | [Art-ificial Intelligence: The Effect of AI Disclosure on Evaluations of Creative Content.](http://arxiv.org/abs/2303.06217) | 本研究探讨了披露使用AI创作创意内容如何影响人类对此类内容的评价。结果表明，AI披露对于创意或描述性短篇小说的评价没有实质性影响，但是对于以第一人称写成的情感诱发诗歌的评价有负面影响。这表明，当内容被视为明显“人类”时，对AI生成内容的反应可能是负面的。 |
| [^7] | [Modelling Large Dimensional Datasets with Markov Switching Factor Models.](http://arxiv.org/abs/2210.09828) | 本论文提出了一种新的大维近似因子模型，在加载因子中引入了由潜在马尔可夫过程驱动的制度变化，通过在模型中利用等效线性表示，通过主成分分析恢复了潜在因子，然后使用基于改进滤波器和平滑器的EM算法，估计加载因子和转换概率。这种方法的吸引力在于提供了所有估计量的闭式表达式，并且不需要知道真实因子的数量。通过一系列蒙特卡洛实验展示了该方法的良好有限样本性能，并通过对大型股票组合的应用验证了方法的实证效用。 |
| [^8] | [Generalized Social Marginal Welfare Weights Imply Inconsistent Comparisons of Tax Policies.](http://arxiv.org/abs/2102.07702) | 本文研究了广义社会边际福利权重（GSMWW），发现局部税收政策比较隐含着全局比较，而且对于没有功利主义结构的福利权重而言，这些隐含的全局比较是不一致的。 |

# 详细

[^1]: 随机实验中的最小后悔样本选择

    Minimax-Regret Sample Selection in Randomized Experiments

    [https://arxiv.org/abs/2403.01386](https://arxiv.org/abs/2403.01386)

    通过最小后悔框架，提出了在随机试验中优化样本选择以实现异质人群中最佳福利的问题，并在不同条件下推导出最优的样本选择方案。

    

    随机对照试验（RCTs）经常在存在许多可能对所评估的治疗效果有差异的子人群中进行。我们考虑了样本选择问题，即在异质人群中如何选择入组RRT，以优化福利。我们在最小后悔框架下形式化了这个问题，并在多种条件下推导出最优的样本选择方案。我们还强调了不同的目标和决策如何导致明显不同的关于最佳样本分配的指导，通过利用历史COVID-19试验数据进行了一项合成实验。

    arXiv:2403.01386v1 Announce Type: cross  Abstract: Randomized controlled trials (RCTs) are often run in settings with many subpopulations that may have differential benefits from the treatment being evaluated. We consider the problem of sample selection, i.e., whom to enroll in an RCT, such as to optimize welfare in a heterogeneous population. We formalize this problem within the minimax-regret framework, and derive optimal sample-selection schemes under a variety of conditions. We also highlight how different objectives and decisions can lead to notably different guidance regarding optimal sample allocation through a synthetic experiment leveraging historical COVID-19 trial data.
    
[^2]: 随机划分、Shapley值的潜力和具有外部性的博弈论

    Random partitions, potential of the Shapley value, and games with externalities

    [https://arxiv.org/abs/2402.00394](https://arxiv.org/abs/2402.00394)

    本论文研究了博弈的潜力函数，发现了一种与Macho-Stadler等人的MPW解相关的独特潜力函数，它通过玩家随机划分的期望累积价值计算，推广了无外部性博弈的潜力，并且在存在外部性的情况下仍满足空玩家特性。

    

    Shapley值等于玩家对博弈的潜力的贡献。潜力是一个最自然的对博弈的一种数值总结，可以计算为玩家随机划分的期望累积价值。这种计算将所有玩家的联盟形成集成起来，并且可以轻松推广到具有外部性的博弈中。我们研究了那些可以通过这种方式计算的具有外部性的博弈的潜力函数。结果表明，与Macho-Stadler等人(2007, J. Econ. Theory 135, 339-356)引入的MPW解相对应的潜力在以下意义上是唯一的。它作为玩家随机划分的期望累积价值得到，它推广了没有外部性的博弈的潜力，并且在存在外部性的情况下满足空玩家特性。

    The Shapley value equals a player's contribution to the potential of a game. The potential is a most natural one-number summary of a game, which can be computed as the expected accumulated worth of a random partition of the players. This computation integrates the coalition formation of all players and readily extends to games with externalities. We investigate those potential functions for games with externalities that can be computed this way. It turns out that the potential that corresponds to the MPW solution introduced by Macho-Stadler et al. (2007, J. Econ. Theory 135, 339-356), is unique in the following sense. It is obtained as a the expected accumulated worth of a random partition, it generalizes the potential for games without externalities, and it induces a solution that satisfies the null player property even in the presence of externalities.
    
[^3]: 纵向和面板数据的因果模型：一项调研

    Causal Models for Longitudinal and Panel Data: A Survey

    [https://arxiv.org/abs/2311.15458](https://arxiv.org/abs/2311.15458)

    该调研总结了最近关于因果面板数据的文献，重点是在具有纵向数据的情况下，可靠地估计二元干预的因果效应，并强调了对实证研究人员的实用建议。

    

    这项调研讨论了最近关于因果面板数据的文献。这些最近的文献主要关注可靠地估计具有纵向数据的二元干预的因果效应，强调了对实证研究人员的实用建议。它特别关注了因果效应中的异质性，在少量单位被处理且为分配模式设定了特定结构的情况下。该文献扩展了早期关于差分法或双因素固定效应估计器的工作。它一般更多地纳入了因子模型或交互固定效应。它还开发了使用合成对照方法的新方法。

    arXiv:2311.15458v2 Announce Type: replace  Abstract: This survey discusses the recent causal panel data literature. This recent literature has focused on credibly estimating causal effects of binary interventions in settings with longitudinal data, with an emphasis on practical advice for empirical researchers. It pays particular attention to heterogeneity in the causal effects, often in situations where few units are treated and with particular structures on the assignment pattern. The literature has extended earlier work on difference-in-differences or two-way-fixed-effect estimators. It has more generally incorporated factor models or interactive fixed effects. It has also developed novel methods using synthetic control approaches.
    
[^4]: 灵活的热泵：在可再生能源领域中的电力部门中是必需还是可选？

    Flexible heat pumps: must-have or nice to have in a power sector with renewables?. (arXiv:2307.12918v1 [econ.GN])

    [http://arxiv.org/abs/2307.12918](http://arxiv.org/abs/2307.12918)

    本研究使用开源的电力部门模型，研究了2030年德国大规模扩张分散式热泵对电力部门的影响。结果表明，热泵的推广可以通过太阳能光伏等可再生能源实现，且额外的备用容量和电力储存需求有限。

    

    热泵是减少供暖领域化石燃料使用的关键技术。向热泵的转变意味着冬季寒冷月份电力需求的增加。使用开源的电力部门模型，我们研究了2030年德国的分散式热泵大规模扩张对电力部门的影响，结合不同规模的缓冲热存储。假设热泵额外使用的电力在年度平衡中必须完全由可再生能源覆盖，我们量化了可再生能源所需的额外投资。如果风力扩张潜力有限，热泵的推广也可以通过太阳能光伏在欧洲互联连接的情况下作为附加成本较小的选择。即使在时间上不灵活的热泵的情况下，对额外备用容量和电力储存的需求通常仍然有限。我们进一步发现，在2至6小时的较小热存储容量下，系统的供需平衡仍然能够得到满足。

    Heat pumps are a key technology for reducing fossil fuel use in the heating sector. A transition to heat pumps implies an increase in electricity demand, especially in cold winter months. Using an open-source power sector model, we examine the power sector impacts of a massive expansion of decentralized heat pumps in Germany in 2030, combined with buffer heat storage of different sizes. Assuming that the additional electricity used by heat pumps has to be fully covered by renewable energies in a yearly balance, we quantify the required additional investments in renewable energy sources. If wind power expansion potentials are limited, the roll-out of heat pumps can also be accompanied by solar PV with little additional costs, making use of the European interconnection. The need for additional firm capacity and electricity storage generally remains limited even in the case of temporally inflexible heat pumps. We further find that relatively small heat storage capacities of 2 to 6 hours c
    
[^5]: 短面板数据模型中的异质性自回归

    Heterogeneous Autoregressions in Short T Panel Data Models. (arXiv:2306.05299v1 [econ.EM])

    [http://arxiv.org/abs/2306.05299](http://arxiv.org/abs/2306.05299)

    本文研究了带有个体特定效应和异质性自回归系数的一阶自回归面板数据模型，并提出了估计自回归系数横截面分布矩的方法，结果表明标准广义矩估计器是有偏的。本文还比较了在均匀和异质性斜率下的小样本性质。该研究可应用于收入决定的经济分析。

    

    本文考虑了带有个体特定效应和异质性自回归系数的一阶自回归面板数据模型。它提出了估计自回归系数横截面分布矩的方法，特别是关注前两个矩，假设自回归系数的随机系数模型，不对固定效应施加任何限制。结果表明，由均匀斜率下得到的标准广义矩估计器是有偏的。本文还研究了在分类分布下，假设有限个类别的自回归系数的概率分布被确定的条件。通过蒙特卡罗实验比较了提出的估计器在均匀和异质性斜率下的小样本性质和其他估计器之间的差异。异质性方法的效用可通过在收入决定的经济应用中的应用来说明。

    This paper considers a first-order autoregressive panel data model with individual-specific effects and a heterogeneous autoregressive coefficient. It proposes estimators for the moments of the cross-sectional distribution of the autoregressive coefficients, with a focus on the first two moments, assuming a random coefficient model for the autoregressive coefficients without imposing any restrictions on the fixed effects. It is shown that the standard generalized method of moments estimators obtained under homogeneous slopes are biased. The paper also investigates conditions under which the probability distribution of the autoregressive coefficients is identified assuming a categorical distribution with a finite number of categories. Small sample properties of the proposed estimators are investigated by Monte Carlo experiments and compared with alternatives both under homogeneous and heterogeneous slopes. The utility of the heterogeneous approach is illustrated in the case of earning d
    
[^6]: 人工智能对创意内容评价的影响：AI披露对创意内容评价的影响

    Art-ificial Intelligence: The Effect of AI Disclosure on Evaluations of Creative Content. (arXiv:2303.06217v1 [cs.CY])

    [http://arxiv.org/abs/2303.06217](http://arxiv.org/abs/2303.06217)

    本研究探讨了披露使用AI创作创意内容如何影响人类对此类内容的评价。结果表明，AI披露对于创意或描述性短篇小说的评价没有实质性影响，但是对于以第一人称写成的情感诱发诗歌的评价有负面影响。这表明，当内容被视为明显“人类”时，对AI生成内容的反应可能是负面的。

    This study explores how disclosure regarding the use of AI in the creation of creative content affects human evaluation of such content. The results show that AI disclosure has no meaningful effect on evaluation either for creative or descriptive short stories, but has a negative effect on evaluations for emotionally evocative poems written in the first person. This suggests that reactions to AI-generated content may be negative when the content is viewed as distinctly "human."

    生成式AI技术的出现，如OpenAI的ChatGPT聊天机器人，扩大了AI工具可以完成的任务范围，并实现了AI生成的创意内容。在本研究中，我们探讨了关于披露使用AI创作创意内容如何影响人类对此类内容的评价。在一系列预先注册的实验研究中，我们发现AI披露对于创意或描述性短篇小说的评价没有实质性影响，但是AI披露对于以第一人称写成的情感诱发诗歌的评价有负面影响。我们解释这个结果表明，当内容被视为明显“人类”时，对AI生成内容的反应可能是负面的。我们讨论了这项工作的影响，并概述了计划研究的途径，以更好地了解AI披露是否会影响创意内容的评价以及何时会影响。

    The emergence of generative AI technologies, such as OpenAI's ChatGPT chatbot, has expanded the scope of tasks that AI tools can accomplish and enabled AI-generated creative content. In this study, we explore how disclosure regarding the use of AI in the creation of creative content affects human evaluation of such content. In a series of pre-registered experimental studies, we show that AI disclosure has no meaningful effect on evaluation either for creative or descriptive short stories, but that AI disclosure has a negative effect on evaluations for emotionally evocative poems written in the first person. We interpret this result to suggest that reactions to AI-generated content may be negative when the content is viewed as distinctly "human." We discuss the implications of this work and outline planned pathways of research to better understand whether and when AI disclosure may affect the evaluation of creative content.
    
[^7]: 使用马尔可夫转换因子模型对大维数据集进行建模

    Modelling Large Dimensional Datasets with Markov Switching Factor Models. (arXiv:2210.09828v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2210.09828](http://arxiv.org/abs/2210.09828)

    本论文提出了一种新的大维近似因子模型，在加载因子中引入了由潜在马尔可夫过程驱动的制度变化，通过在模型中利用等效线性表示，通过主成分分析恢复了潜在因子，然后使用基于改进滤波器和平滑器的EM算法，估计加载因子和转换概率。这种方法的吸引力在于提供了所有估计量的闭式表达式，并且不需要知道真实因子的数量。通过一系列蒙特卡洛实验展示了该方法的良好有限样本性能，并通过对大型股票组合的应用验证了方法的实证效用。

    

    我们研究了一个新型的大维近似因子模型，其中加载因子由潜在的一阶马尔可夫过程驱动。通过利用模型的等价线性表示，我们首先通过主成分分析恢复潜在因子。然后我们将模型转化为状态空间形式，并通过基于改进版本的Baum-Lindgren-Hamilton-Kim滤波器和平滑器的EM算法，利用先前估计得到的因子来估计加载因子和转换概率。我们的方法具有吸引力，因为它为所有估计量提供了闭式表达式。更重要的是，它不需要知道真实因子的数量。我们推导了所提估计过程的理论性质，并通过一套全面的蒙特卡洛实验展示了它们的良好有限样本性能。通过对一个大型股票投资组合的应用，我们说明了我们方法的实证效用。

    We study a novel large dimensional approximate factor model with regime changes in the loadings driven by a latent first order Markov process. By exploiting the equivalent linear representation of the model, we first recover the latent factors by means of Principal Component Analysis. We then cast the model in state-space form, and we estimate loadings and transition probabilities through an EM algorithm based on a modified version of the Baum-Lindgren-Hamilton-Kim filter and smoother that makes use of the factors previously estimated. Our approach is appealing as it provides closed form expressions for all estimators. More importantly, it does not require knowledge of the true number of factors. We derive the theoretical properties of the proposed estimation procedure, and we show their good finite sample performance through a comprehensive set of Monte Carlo experiments. The empirical usefulness of our approach is illustrated through an application to a large portfolio of stocks.
    
[^8]: 广义社会边际福利权重意味着税收政策的不一致比较

    Generalized Social Marginal Welfare Weights Imply Inconsistent Comparisons of Tax Policies. (arXiv:2102.07702v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2102.07702](http://arxiv.org/abs/2102.07702)

    本文研究了广义社会边际福利权重（GSMWW），发现局部税收政策比较隐含着全局比较，而且对于没有功利主义结构的福利权重而言，这些隐含的全局比较是不一致的。

    

    本文关注Saez和Stantcheva（2016）的广义社会边际福利权重（GSMWW），它们汇总了由于税收政策而产生的损失和收益，同时包括非功利道德考虑因素。该方法评估了局部税收变化，而无需考虑全局社会目标。作者表明，局部税收政策比较隐含着全局比较。此外，每当福利权重没有功利主义结构时，这些隐含的全局比较就是不一致的。作者认为，不能简单地通过修改给不同人带来的利益权重来代表更广泛的道德价值观，而是需要更全面地修改功利主义方法。

    This paper concerns Saez and Stantcheva's (2016) generalized social marginal welfare weights (GSMWW), which aggregate losses and gains due to tax policies, while incorporating non-utilitarian ethical considerations. The approach evaluates local tax changes without a global social objective. I show that local tax policy comparisons implicitly entail global comparisons. Moreover, whenever welfare weights do not have a utilitarian structure, these implied global comparisons are inconsistent. I argue that broader ethical values cannot in general be represented simply by modifying the weights placed on benefits to different people, and a more thoroughgoing modification of the utilitarian approach is required.
    

