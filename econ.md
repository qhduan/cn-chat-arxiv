# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Analyzing Games in Maker Protocol Part One: A Multi-Agent Influence Diagram Approach Towards Coordination](https://arxiv.org/abs/2402.15037) | 通过多agent影响图方法分析 Maker 协议中的游戏动态，并提出了促进协调和增强经济安全性的策略 |
| [^2] | [Arellano-Bond LASSO Estimator for Dynamic Linear Panel Models](https://arxiv.org/abs/2402.00584) | 提出了一种针对长时间序列数据的Arellano-Bond估计器偏差问题的两步法。首先利用LASSO选择信息量最大的矩条件，然后使用这些条件构造工具变量进行线性仪器变量估计。该方法通过样本切分和交叉拟合来避免过拟合偏差，并且在对时间序列维度$T$的条件要求更弱的情况下是一致的和渐近正态的。 |
| [^3] | [Do LLM Agents Exhibit Social Behavior?](https://arxiv.org/abs/2312.15198) | 研究探讨了LLM代理在与人类和其他代理互动时展示的社会行为，包括社会学习、社会偏好和合作行为，并开发了一个框架来评估它们与人类实验对象的互动。 |
| [^4] | [Linking Mechanisms: Limits and Robustness.](http://arxiv.org/abs/2309.07363) | 本文研究了配额机制在代理人面临多个决策且货币转移不可行时的应用。研究发现，随着决策数量的增加，配额机制可以实现与使用转移的独立机制相同的社会选择函数集合，并且对于代理对彼此的信念是鲁棒的。研究还发现，为了设置正确的配额，设计者必须对环境有精确的了解，否则只能实施平凡的社会选择规则。同时，配额使得报告的分布成为共知。 |
| [^5] | [Obvious Manipulations in Matching with and without Contracts.](http://arxiv.org/abs/2306.17773) | 在医生多对一匹配模型中，任何稳定的匹配规则都可以被医生操纵。但在有合同的模型中，医生最优匹配规则不容易被明显操纵，而医院最优匹配规则则容易被明显操纵。对于没有合同的多对一模型，医院最优匹配规则不容易被明显操纵。 |

# 详细

[^1]: 在 Maker 协议中分析游戏的第一部分：一种面向协调的多agent影响图方法

    Analyzing Games in Maker Protocol Part One: A Multi-Agent Influence Diagram Approach Towards Coordination

    [https://arxiv.org/abs/2402.15037](https://arxiv.org/abs/2402.15037)

    通过多agent影响图方法分析 Maker 协议中的游戏动态，并提出了促进协调和增强经济安全性的策略

    

    去中心化金融（DeFi）生态系统，以 Maker 协议为例，依赖于复杂的游戏来维持稳定性和安全性。了解这些游戏的动态对于确保系统的稳健性至关重要。本研究提出一种新颖的方法，利用多agent影响图（MAID），这是由 Koller 和 Milch 最初提出的，以解剖和分析 Maker 稳定币协议中的游戏。通过将 Maker 协议的用户和治理表示为代理人，将它们的互动表示为图中的边，我们捕捉到指导代理人行为的复杂影响网络。此外，在接下来的论文中，我们将展示一个纳什均衡模型，以阐明促进协调和增强生态系统内经济安全性的策略。通过这种方法，我们旨在推动利用这种方法引入一种新的形式验证方法，验证博弈论安全性

    arXiv:2402.15037v1 Announce Type: cross  Abstract: Decentralized Finance (DeFi) ecosystems, exemplified by the Maker Protocol, rely on intricate games to maintain stability and security. Understanding the dynamics of these games is crucial for ensuring the robustness of the system. This motivating research proposes a novel methodology leveraging Multi-Agent Influence Diagrams (MAID), originally proposed by Koller and Milch, to dissect and analyze the games within the Maker stablecoin protocol. By representing users and governance of the Maker protocol as agents and their interactions as edges in a graph, we capture the complex network of influences governing agent behaviors. Furthermore in the upcoming papers, we will show a Nash Equilibrium model to elucidate strategies that promote coordination and enhance economic security within the ecosystem. Through this approach, we aim to motivate the use of this method to introduce a new method of formal verification of game theoretic security
    
[^2]: Arellano-Bond LASSO估计器用于动态线性面板模型

    Arellano-Bond LASSO Estimator for Dynamic Linear Panel Models

    [https://arxiv.org/abs/2402.00584](https://arxiv.org/abs/2402.00584)

    提出了一种针对长时间序列数据的Arellano-Bond估计器偏差问题的两步法。首先利用LASSO选择信息量最大的矩条件，然后使用这些条件构造工具变量进行线性仪器变量估计。该方法通过样本切分和交叉拟合来避免过拟合偏差，并且在对时间序列维度$T$的条件要求更弱的情况下是一致的和渐近正态的。

    

    当数据的时间序列维度$T$较长时，Arellano-Bond估计器可能出现严重偏差。偏差的来源是过度识别的程度过大。我们提出了一种简单的两步方法来解决这个问题。第一步在每个时间段对横截面数据应用LASSO选择最有信息量的矩条件。第二步使用从第一步选择的矩条件构造的工具变量估计线性仪器。通过样本切分和交叉拟合两个阶段来避免过拟合偏差。使用随着样本大小增长的渐近序列，我们证明了新的估计器在对$T$的条件要求比Arellano-Bond估计器要弱的情况下是一致的和渐近正态的。我们的理论涵盖了包括依赖变量的多个滞后等高维协变量在内的模型，这在现代应用中很常见。

    The Arellano-Bond estimator can be severely biased when the time series dimension of the data, $T$, is long. The source of the bias is the large degree of overidentification. We propose a simple two-step approach to deal with this problem. The first step applies LASSO to the cross-section data at each time period to select the most informative moment conditions. The second step applies a linear instrumental variable estimator using the instruments constructed from the moment conditions selected in the first step. The two stages are combined using sample-splitting and cross-fitting to avoid overfitting bias. Using asymptotic sequences where the two dimensions of the panel grow with the sample size, we show that the new estimator is consistent and asymptotically normal under much weaker conditions on $T$ than the Arellano-Bond estimator. Our theory covers models with high dimensional covariates including multiple lags of the dependent variable, which are common in modern applications. We
    
[^3]: LLM代理表现出社会行为吗？

    Do LLM Agents Exhibit Social Behavior?

    [https://arxiv.org/abs/2312.15198](https://arxiv.org/abs/2312.15198)

    研究探讨了LLM代理在与人类和其他代理互动时展示的社会行为，包括社会学习、社会偏好和合作行为，并开发了一个框架来评估它们与人类实验对象的互动。

    

    大型语言模型（LLMs）的进展正在扩大它们在学术研究和实际应用中的效用。最近的社会科学研究探讨了使用这些“黑匣子”LLM代理来模拟复杂社会系统并潜在地替代人类实验对象的可能性。我们的研究深入探讨了这一新兴领域，调查了LLMs在与人类和其他代理进行互动时展示社会学习、社会偏好和合作行为（间接互惠）等关键社会交互原则的程度。我们为我们的研究制定了一个框架，其中涉及将涉及人类实验对象的经典实验调整为使用LLM代理。这种方法涉及一步一步的推理，模拟人类认知过程和零样本学习，以评估LLMs的天生偏好。我们对LLM代理行为的分析包括主要效应和次要效应。

    arXiv:2312.15198v2 Announce Type: replace  Abstract: The advances of Large Language Models (LLMs) are expanding their utility in both academic research and practical applications. Recent social science research has explored the use of these ``black-box'' LLM agents for simulating complex social systems and potentially substituting human subjects in experiments. Our study delves into this emerging domain, investigating the extent to which LLMs exhibit key social interaction principles, such as social learning, social preference, and cooperative behavior (indirect reciprocity), in their interactions with humans and other agents. We develop a framework for our study, wherein classical laboratory experiments involving human subjects are adapted to use LLM agents. This approach involves step-by-step reasoning that mirrors human cognitive processes and zero-shot learning to assess the innate preferences of LLMs. Our analysis of LLM agents' behavior includes both the primary effects and an in
    
[^4]: 链接机制：限制和鲁棒性

    Linking Mechanisms: Limits and Robustness. (arXiv:2309.07363v1 [econ.TH])

    [http://arxiv.org/abs/2309.07363](http://arxiv.org/abs/2309.07363)

    本文研究了配额机制在代理人面临多个决策且货币转移不可行时的应用。研究发现，随着决策数量的增加，配额机制可以实现与使用转移的独立机制相同的社会选择函数集合，并且对于代理对彼此的信念是鲁棒的。研究还发现，为了设置正确的配额，设计者必须对环境有精确的了解，否则只能实施平凡的社会选择规则。同时，配额使得报告的分布成为共知。

    

    配额机制在代理面对多个决策且货币转移不可行时常被用于引出私人信息。当决策数量增加时，配额机制渐近地实施与使用转移的独立机制相同的社会选择函数集合。我们分析了配额机制的鲁棒性。为了设置正确的配额，设计者必须对环境有精确的了解。我们证明，没有转移的情况下，只有平凡的社会选择规则可以以先验独立的方式实施。当配额不匹配真实的类型分布时，我们得到了决策误差的紧密界限。最后，我们证明在多代理的环境中，配额对于代理对彼此的信念是鲁棒的。关键是，配额使得报告的分布成为共知。

    Quota mechanisms are commonly used to elicit private information when agents face multiple decisions and monetary transfers are infeasible. As the number of decisions grows large, quotas asymptotically implement the same set of social choice functions as do separate mechanisms with transfers. We analyze the robustness of quota mechanisms. To set the correct quota, the designer must have precise knowledge of the environment. We show that, without transfers, only trivial social choice rules can be implemented in a prior-independent way. We obtain a tight bound on the decision error that results when the quota does not match the true type distribution. Finally, we show that in a multi-agent setting, quotas are robust to agents' beliefs about each other. Crucially, quotas make the distribution of reports common knowledge.
    
[^5]: 匹配中的明显操纵问题：有合同和无合同的情况。

    Obvious Manipulations in Matching with and without Contracts. (arXiv:2306.17773v1 [econ.TH])

    [http://arxiv.org/abs/2306.17773](http://arxiv.org/abs/2306.17773)

    在医生多对一匹配模型中，任何稳定的匹配规则都可以被医生操纵。但在有合同的模型中，医生最优匹配规则不容易被明显操纵，而医院最优匹配规则则容易被明显操纵。对于没有合同的多对一模型，医院最优匹配规则不容易被明显操纵。

    

    在一个医生多对一匹配模型中，无论是否有合同，医生的偏好是私有信息，医院的偏好是可替代的和公开的信息，任何稳定的匹配规则都可以被医生操纵。由于操纵无法完全避免，我们考虑了“明显操纵”的概念，并寻找至少可以防止这些操纵的稳定匹配规则（对医生而言）。对于有合同的模型，我们证明了：（i）医生最优匹配规则不容易被明显操纵；（ii）医院最优匹配规则即使在一对一模型中也容易被明显操纵。与（ii）相反，对于没有合同的多对一模型，我们证明了医院最优匹配规则不容易被明显操纵（当医院的偏好是可替代的）。此外，如果我们关注分位数稳定规则，则证明了医生最优匹配规则是唯一的不容易被明显操纵的规则。

    In a many-to-one matching model, with or without contracts, where doctors' preferences are private information and hospitals' preferences are substitutable and public information, any stable matching rule could be manipulated for doctors. Since manipulations can not be completely avoided, we consider the concept of \textit{obvious manipulations} and look for stable matching rules that prevent at least such manipulations (for doctors). For the model with contracts, we prove that: \textit{(i)} the doctor-optimal matching rule is non-obviously manipulable and \textit{(ii)} the hospital-optimal matching rule is obviously manipulable, even in the one-to-one model. In contrast to \textit{(ii)}, for a many-to-one model without contracts, we prove that the hospital-optimal matching rule is not obviously manipulable.% when hospitals' preferences are substitutable. Furthermore, if we focus on quantile stable rules, then we prove that the doctor-optimal matching rule is the only non-obviously man
    

