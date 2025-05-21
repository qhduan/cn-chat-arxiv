# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Don't (fully) exclude me, it's not necessary! Identification with semi-IVs.](http://arxiv.org/abs/2303.12667) | 本文提出了一种利用半工具变量实现离散内生变量模型识别的方法，对劳动、健康、教育等领域具有潜在应用价值。 |
| [^2] | [Individualized Treatment Allocation in Sequential Network Games.](http://arxiv.org/abs/2302.05747) | 本文针对顺序决策博弈中的互动主体，提出了一种个体化治疗分配方法，通过评估结果的固定分布并采用变分近似和贪婪优化算法，最大化了社会福利准则。 |

# 详细

[^1]: 不要完全排除我，这是不必要的! 半工具变量的识别

    Don't (fully) exclude me, it's not necessary! Identification with semi-IVs. (arXiv:2303.12667v1 [econ.EM])

    [http://arxiv.org/abs/2303.12667](http://arxiv.org/abs/2303.12667)

    本文提出了一种利用半工具变量实现离散内生变量模型识别的方法，对劳动、健康、教育等领域具有潜在应用价值。

    

    本文提出了一种识别离散内生变量模型的新方法，将其应用于连续潜在结果的不可分离模型的一般情况下进行研究。我们采用半工具变量（semi-IVs) 来实现潜在结果的非参数识别以及选择方程式的识别，因此也能够识别个体治疗效应。与标准工具变量 （IVs）需要强制性完全排除不同，半工具变量仅在一些潜在结果方程式中部分排除，而不是全部排除。实践中，需要在强化排除约束和找到支持范围更广、相关性假设更强的半工具变量之间权衡。我们的方法为识别、估计和反事实预测开辟了新的途径，并在许多领域，如劳动，健康和教育等方面具有潜在应用。

    This paper proposes a novel approach to identify models with a discrete endogenous variable, that I study in the general context of nonseparable models with continuous potential outcomes. I show that nonparametric identification of the potential outcome and selection equations, and thus of the individual treatment effects, can be obtained with semi-instrumental variables (semi-IVs), which are relevant but only partially excluded from the potential outcomes, i.e., excluded from one or more potential outcome equations, but not necessarily all. This contrasts with the full exclusion restriction imposed on standard instrumental variables (IVs), which is stronger than necessary for identification: IVs are only a special case of valid semi-IVs. In practice, there is a trade-off between imposing stronger exclusion restrictions, and finding semi-IVs with a larger support and stronger relevance assumptions. Since, in empirical work, the main obstacle for finding a valid IV is often the full exc
    
[^2]: 顺序网络博弈中的个体化治疗分配

    Individualized Treatment Allocation in Sequential Network Games. (arXiv:2302.05747v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2302.05747](http://arxiv.org/abs/2302.05747)

    本文针对顺序决策博弈中的互动主体，提出了一种个体化治疗分配方法，通过评估结果的固定分布并采用变分近似和贪婪优化算法，最大化了社会福利准则。

    

    设计个体化的治疗分配，以最大化互动主体的均衡福利，在政策相关的应用中有很大的意义。本文针对互动主体的顺序决策博弈，开发了一种方法来获得最优的治疗分配规则，通过评估结果的固定分布来最大化社会福利准则。在顺序决策博弈中，固定分布由Gibbs分布给出，由于解析和计算复杂性，很难对治疗分配进行优化。我们采用变分近似来优化固定分布，并使用贪婪优化算法来优化近似平衡福利的治疗分配。我们通过福利遗憾界限推导了变分近似的性能，对贪婪优化算法的性能进行了表征。我们在模拟实验中实现了我们提出的方法。

    Designing individualized allocation of treatments so as to maximize the equilibrium welfare of interacting agents has many policy-relevant applications. Focusing on sequential decision games of interacting agents, this paper develops a method to obtain optimal treatment assignment rules that maximize a social welfare criterion by evaluating stationary distributions of outcomes. Stationary distributions in sequential decision games are given by Gibbs distributions, which are difficult to optimize with respect to a treatment allocation due to analytical and computational complexity. We apply a variational approximation to the stationary distribution and optimize the approximated equilibrium welfare with respect to treatment allocation using a greedy optimization algorithm. We characterize the performance of the variational approximation, deriving a performance guarantee for the greedy optimization algorithm via a welfare regret bound. We implement our proposed method in simulation exerci
    

