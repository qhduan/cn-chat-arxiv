# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Local Identification in the Instrumental Variable Multivariate Quantile Regression Model.](http://arxiv.org/abs/2401.11422) | 提出了基于最优输运的多元分位数回归模型，考虑结果变量中条目之间的相关性，并提供了局部识别结果。结果表明，所需的仪器变量（IV）的支持大小与结果向量的维度无关，只需IV足够信息量。 |
| [^2] | [Human and Machine Intelligence in n-Person Games with Partial Knowledge.](http://arxiv.org/abs/2302.13937) | 提出了有限知识下n人博弈的框架，引入了“游戏智能”机制和“防作弊性”概念，GI机制可以实际评估玩家的智能，应用广泛。 |
| [^3] | [Bayesian analysis of mixtures of lognormal distribution with an unknown number of components from grouped data.](http://arxiv.org/abs/2210.05115) | 本研究提出了一种用于估计收入的对数正态分布混合模型参数的贝叶斯分析方法，并通过模拟和实证数据的验证表明了其准确性和适用性。 |

# 详细

[^1]: 仪器变量多元分位数回归模型中的局部识别

    Local Identification in the Instrumental Variable Multivariate Quantile Regression Model. (arXiv:2401.11422v1 [econ.EM])

    [http://arxiv.org/abs/2401.11422](http://arxiv.org/abs/2401.11422)

    提出了基于最优输运的多元分位数回归模型，考虑结果变量中条目之间的相关性，并提供了局部识别结果。结果表明，所需的仪器变量（IV）的支持大小与结果向量的维度无关，只需IV足够信息量。

    

    Chernozhukov和Hansen（2005）引入的仪器变量（IV）分位数回归模型是分析内生性情况下分位数处理效应的有用工具，但当结果变量是多维的时，它对每个变量不同维度的联合分布保持沉默。为了克服这个限制，我们提出了一个基于最优输运的考虑结果变量中条目之间相关性的多元分位数回归模型。然后，我们为模型提供了一个局部识别结果。令人惊讶的是，我们发现，所需的用于识别的IV的支持大小与结果向量的维度无关，只要IV足够信息量。我们的结果来自我们建立的一个具有独立理论意义的一般识别定理。

    The instrumental variable (IV) quantile regression model introduced by Chernozhukov and Hansen (2005) is a useful tool for analyzing quantile treatment effects in the presence of endogeneity, but when outcome variables are multidimensional, it is silent on the joint distribution of different dimensions of each variable. To overcome this limitation, we propose an IV model built on the optimal-transport-based multivariate quantile that takes into account the correlation between the entries of the outcome variable. We then provide a local identification result for the model. Surprisingly, we find that the support size of the IV required for the identification is independent of the dimension of the outcome vector, as long as the IV is sufficiently informative. Our result follows from a general identification theorem that we establish, which has independent theoretical significance.
    
[^2]: 有限知识下n人博弈的人类与机器智能

    Human and Machine Intelligence in n-Person Games with Partial Knowledge. (arXiv:2302.13937v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2302.13937](http://arxiv.org/abs/2302.13937)

    提出了有限知识下n人博弈的框架，引入了“游戏智能”机制和“防作弊性”概念，GI机制可以实际评估玩家的智能，应用广泛。

    

    本文提出了一个新的框架——有限知识下的n人博弈，其中玩家只对游戏的某些方面（包括行动、结果和其他玩家）有限的了解。为了分析这些游戏，我介绍了一组新的概念和机制，重点关注人类和机器决策之间的相互作用。具体而言，我引入了两个主要概念：第一个是“游戏智能”（GI）机制，它通过考虑参考机器智能下的“错误”，不仅仅是游戏的结果，量化了玩家在游戏中展示出的智能。第二个是“防作弊性”，这是一种实用的、可计算的策略无关性的概念。GI机制提供了一种实用的方法来评估玩家，可以潜在地应用于从在线游戏到现实生活决策的各种游戏。

    In this note, I introduce a new framework called n-person games with partial knowledge, in which players have only limited knowledge about the aspects of the game -- including actions, outcomes, and other players. For example, playing an actual game of chess is a game of partial knowledge. To analyze these games, I introduce a set of new concepts and mechanisms for measuring the intelligence of players, with a focus on the interplay between human- and machine-based decision-making. Specifically, I introduce two main concepts: firstly, the Game Intelligence (GI) mechanism, which quantifies a player's demonstrated intelligence in a game by considering not only the game's outcome but also the "mistakes" made during the game according to the reference machine's intelligence. Secondly, I define gaming-proofness, a practical and computational concept of strategy-proofness. The GI mechanism provides a practicable way to assess players and can potentially be applied to a wide range of games, f
    
[^3]: 未知组件数的对数正态分布混合模型的贝叶斯分析和组合数据的马尔可夫链蒙特卡洛方法

    Bayesian analysis of mixtures of lognormal distribution with an unknown number of components from grouped data. (arXiv:2210.05115v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2210.05115](http://arxiv.org/abs/2210.05115)

    本研究提出了一种用于估计收入的对数正态分布混合模型参数的贝叶斯分析方法，并通过模拟和实证数据的验证表明了其准确性和适用性。

    

    本研究提出了一种可逆跳跃马尔可夫链蒙特卡洛方法，用于估计收入的对数正态分布混合模型的参数。通过使用模拟数据示例，我们检验了所提算法的性能以及基尼系数的后验分布的准确性。结果表明参数估计准确，即使考虑了不同的数据生成过程，后验分布仍接近真实分布。此外，基于更具吸引力的基尼系数的结果，我们还将该方法应用于来自日本的实际数据。实证案例表明日本在2020年存在两个子群，并且基尼系数的完整性得到了验证。

    This study proposes a reversible jump Markov chain Monte Carlo method for estimating parameters of lognormal distribution mixtures for income. Using simulated data examples, we examined the proposed algorithm's performance and the accuracy of posterior distributions of the Gini coefficients. Results suggest that the parameters were estimated accurately. Therefore, the posterior distributions are close to the true distributions even when the different data generating process is accounted for. Moreover, promising results for Gini coefficients encouraged us to apply our method to real data from Japan. The empirical examples indicate two subgroups in Japan (2020) and the Gini coefficients' integrity.
    

