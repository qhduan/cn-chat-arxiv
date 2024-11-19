# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Shill-Proof Auctions](https://arxiv.org/abs/2404.00475) | 本文研究了免疫作弊的拍卖形式，发现荷兰式拍卖（设有适当保留价）是唯一的最优且强免疫作弊的拍卖，同时荷兰式拍卖（没有保留价）是唯一同时高效和弱免疫作弊的先验独立拍卖。 |
| [^2] | [Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data](https://arxiv.org/abs/2404.00221) | 学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题 |
| [^3] | [Costly Persuasion by a Partially Informed Sender.](http://arxiv.org/abs/2401.14087) | 本研究探讨了具有高昂成本的贝叶斯说服模型，研究对象是一位私人且部分信息知情的发送者在进行公共实验。研究发现实验中好消息和坏消息的成本差异对均衡结果具有重要影响，坏消息成本高时，存在唯一的分离均衡，接收者受益于发送者的私有信息；而好消息成本高时，均衡情况可能出现汇集和部分汇集均衡，接收者可能会因为发送者私有信息而受到损害。 |
| [^4] | [Model-Agnostic Covariate-Assisted Inference on Partially Identified Causal Effects.](http://arxiv.org/abs/2310.08115) | 提出了一种模型不可知的推断方法，在部分可辨识的因果估计中应用广泛。该方法基于最优输运问题的对偶理论，能够适应随机实验和观测研究，并且具有统一有效和双重鲁棒性。 |
| [^5] | [Persuasion as Transportation.](http://arxiv.org/abs/2307.07672) | 本研究通过将说服问题归约为最优运输的Monge-Kantorovich问题，揭示了贝叶斯说服模型中多接收方问题的显式解集和结构性结果，并推广了价值的对偶表示和凹化公式。 |
| [^6] | [Bayes = Blackwell, Almost.](http://arxiv.org/abs/2302.13956) | 存在其他的更新规则可以使信息的价值变为正值，作者找到了所有这些规则。 |

# 详细

[^1]: 免疫作弊拍卖

    Shill-Proof Auctions

    [https://arxiv.org/abs/2404.00475](https://arxiv.org/abs/2404.00475)

    本文研究了免疫作弊的拍卖形式，发现荷兰式拍卖（设有适当保留价）是唯一的最优且强免疫作弊的拍卖，同时荷兰式拍卖（没有保留价）是唯一同时高效和弱免疫作弊的先验独立拍卖。

    

    在单品拍卖中，一个欺诈性的卖家可能会伪装成一个或多个竞标者，以操纵成交价格。本文对那些免疫作弊的拍卖格式进行了表征：一个利润最大化的卖家没有任何动机提交任何虚假报价。我们区分了强免疫作弊，即一个了解竞标者估值的卖家永远无法从作弊中获利，和弱免疫作弊，它仅要求从作弊中得到的平衡预期利润为非正。荷兰式拍卖（设有适当保留价）是唯一的最优和强免疫作弊拍卖。此外，荷兰式拍卖（没有保留价）是唯一的具有先验独立性的拍卖，既高效又弱免疫作弊。虽然存在多种策略证明、弱免疫作弊和最优拍卖；任何最优拍卖只能满足集合 {静态、策略证明、弱免疫作弊} 中的两个性质。

    arXiv:2404.00475v1 Announce Type: new  Abstract: In a single-item auction, a duplicitous seller may masquerade as one or more bidders in order to manipulate the clearing price. This paper characterizes auction formats that are shill-proof: a profit-maximizing seller has no incentive to submit any shill bids. We distinguish between strong shill-proofness, in which a seller with full knowledge of bidders' valuations can never profit from shilling, and weak shill-proofness, which requires only that the expected equilibrium profit from shilling is nonpositive. The Dutch auction (with suitable reserve) is the unique optimal and strongly shill-proof auction. Moreover, the Dutch auction (with no reserve) is the unique prior-independent auction that is both efficient and weakly shill-proof. While there are a multiplicity of strategy-proof, weakly shill-proof, and optimal auctions; any optimal auction can satisfy only two properties in the set {static, strategy-proof, weakly shill-proof}.
    
[^2]: 利用观测数据进行强健学习以获得最佳动态治疗方案

    Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data

    [https://arxiv.org/abs/2404.00221](https://arxiv.org/abs/2404.00221)

    学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题

    

    许多公共政策和医疗干预涉及其治疗分配中的动态性，治疗通常依据先前治疗的历史和相关特征对每个阶段的效果具有异质性。本文研究了统计学习最佳动态治疗方案(DTR)，根据个体的历史指导每个阶段的最佳治疗分配。我们提出了一种基于观测数据的逐步双重强健方法，在顺序可忽略性假设下学习最佳DTR。该方法通过向后归纳解决了顺序治疗分配问题，在每一步中，我们结合倾向评分和行动值函数(Q函数)的估计量，构建了政策价值的增强反向概率加权估计量。

    arXiv:2404.00221v1 Announce Type: cross  Abstract: Many public policies and medical interventions involve dynamics in their treatment assignments, where treatments are sequentially assigned to the same individuals across multiple stages, and the effect of treatment at each stage is usually heterogeneous with respect to the history of prior treatments and associated characteristics. We study statistical learning of optimal dynamic treatment regimes (DTRs) that guide the optimal treatment assignment for each individual at each stage based on the individual's history. We propose a step-wise doubly-robust approach to learn the optimal DTR using observational data under the assumption of sequential ignorability. The approach solves the sequential treatment assignment problem through backward induction, where, at each step, we combine estimators of propensity scores and action-value functions (Q-functions) to construct augmented inverse probability weighting estimators of values of policies 
    
[^3]: 高昂的说服成本与部分信息的发送者

    Costly Persuasion by a Partially Informed Sender. (arXiv:2401.14087v1 [econ.TH])

    [http://arxiv.org/abs/2401.14087](http://arxiv.org/abs/2401.14087)

    本研究探讨了具有高昂成本的贝叶斯说服模型，研究对象是一位私人且部分信息知情的发送者在进行公共实验。研究发现实验中好消息和坏消息的成本差异对均衡结果具有重要影响，坏消息成本高时，存在唯一的分离均衡，接收者受益于发送者的私有信息；而好消息成本高时，均衡情况可能出现汇集和部分汇集均衡，接收者可能会因为发送者私有信息而受到损害。

    

    本文研究了由一个拥有私有且部分信息的发送者进行的昂贵的贝叶斯说服模型，该发送者进行了一个公共实验。实验的成本是发送者信念的加权对数似然比函数的期望减少。这个模型通过一个沃尔德的顺序抽样问题得到微基础，其中好消息和坏消息的成本不同。我们关注满足D1准则的均衡。均衡结果取决于实验中获得好消息和坏消息的相对成本。如果坏消息的成本更高，则存在唯一的分离均衡，并且接收者明确受益于发送者的私有信息。如果好消息的成本更高，则单点交叉特性不成立。可能存在汇集和部分汇集均衡，在某些均衡中，接收者会明确受到发送者私有信息的伤害。

    I study a model of costly Bayesian persuasion by a privately and partially informed sender who conducts a public experiment. The cost of running an experiment is the expected reduction of a weighted log-likelihood ratio function of the sender's belief. This is microfounded by a Wald's sequential sampling problem where good news and bad news cost differently. I focus on equilibria that satisfy the D1 criterion. The equilibrium outcome depends on the relative costs of drawing good and bad news in the experiment. If bad news is more costly, there exists a unique separating equilibrium, and the receiver unambiguously benefits from the sender's private information. If good news is more costly, the single-crossing property fails. There may exist pooling and partial pooling equilibria, and in some equilibria, the receiver strictly suffers from sender private information.
    
[^4]: 模型不可知的辅助推断方法在部分可辨识因果效应上的应用

    Model-Agnostic Covariate-Assisted Inference on Partially Identified Causal Effects. (arXiv:2310.08115v1 [econ.EM])

    [http://arxiv.org/abs/2310.08115](http://arxiv.org/abs/2310.08115)

    提出了一种模型不可知的推断方法，在部分可辨识的因果估计中应用广泛。该方法基于最优输运问题的对偶理论，能够适应随机实验和观测研究，并且具有统一有效和双重鲁棒性。

    

    很多因果估计是部分可辨识的，因为它们依赖于潜在结果之间的不可观察联合分布。基于前处理协变量的分层可以获得更明确的部分可辨识性范围；然而，除非协变量为离散且支撑度相对较小，否则这种方法通常需要对给定协变量的潜在结果的条件分布进行一致估计。因此，现有的方法在模型错误或一致性假设被违反时可能失败。在本研究中，我们提出了一种基于最优输运问题的对偶理论的统一且模型不可知的推断方法，适用于广泛类别的部分可辨识估计。在随机实验中，我们的方法可以结合任何对条件分布的估计，并提供统一有效的推断，即使初始估计是任意不准确的。此外，我们的方法在观测研究中也是双重鲁棒的。

    Many causal estimands are only partially identifiable since they depend on the unobservable joint distribution between potential outcomes. Stratification on pretreatment covariates can yield sharper partial identification bounds; however, unless the covariates are discrete with relatively small support, this approach typically requires consistent estimation of the conditional distributions of the potential outcomes given the covariates. Thus, existing approaches may fail under model misspecification or if consistency assumptions are violated. In this study, we propose a unified and model-agnostic inferential approach for a wide class of partially identified estimands, based on duality theory for optimal transport problems. In randomized experiments, our approach can wrap around any estimates of the conditional distributions and provide uniformly valid inference, even if the initial estimates are arbitrarily inaccurate. Also, our approach is doubly robust in observational studies. Notab
    
[^5]: 说服作为交通工具

    Persuasion as Transportation. (arXiv:2307.07672v1 [econ.TH])

    [http://arxiv.org/abs/2307.07672](http://arxiv.org/abs/2307.07672)

    本研究通过将说服问题归约为最优运输的Monge-Kantorovich问题，揭示了贝叶斯说服模型中多接收方问题的显式解集和结构性结果，并推广了价值的对偶表示和凹化公式。

    

    我们考虑了一个贝叶斯说服模型，其中有一个知情的发送方和几个不知情的接收方。发送方可以通过私人信号影响接收方的信念，而发送方的目标取决于诱导信念的组合。我们将说服问题归约为最优运输的Monge-Kantorovich问题。借助最优运输理论的洞见，我们确定了几类多接收方问题的显式解集，得到了一般的结构性结果，导出了价值的对偶表示，并将著名的凹化公式推广到多接收方问题上。

    We consider a model of Bayesian persuasion with one informed sender and several uninformed receivers. The sender can affect receivers' beliefs via private signals, and the sender's objective depends on the combination of induced beliefs.  We reduce the persuasion problem to the Monge-Kantorovich problem of optimal transportation. Using insights from optimal transportation theory, we identify several classes of multi-receiver problems that admit explicit solutions, get general structural results, derive a dual representation for the value, and generalize the celebrated concavification formula for the value to multi-receiver problems.
    
[^6]: Bayes = Blackwell, 差不多。

    Bayes = Blackwell, Almost. (arXiv:2302.13956v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2302.13956](http://arxiv.org/abs/2302.13956)

    存在其他的更新规则可以使信息的价值变为正值，作者找到了所有这些规则。

    

    存在着除了Bayes'定律之外的更新规则，可以使信息的价值变为正值。我找到了所有这些规则。

    There are updating rules other than Bayes' law that render the value of information positive. I find all of them.
    

