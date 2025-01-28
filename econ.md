# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Strong Maximum Circulation Algorithm: A New Method for Aggregating Preference Rankings.](http://arxiv.org/abs/2307.15702) | 强大的最大环算法提出了一种集成偏好排序的新方法，通过删除投票图中的最大环路，得出与投票结果一致的唯一排序结果。 |
| [^2] | [Synthetic Decomposition for Counterfactual Predictions.](http://arxiv.org/abs/2307.05122) | 本论文提出了一种使用“源”地区数据进行目标地区政策预测的方法。通过制定可转移条件并构建合成的结果-政策关系来满足条件。我们开发了通用过程来构建反事实预测的置信区间，并证明其有效性。本研究应用该方法预测了德克萨斯州青少年就业率。 |
| [^3] | [Identifying Dynamic LATEs with a Static Instrument.](http://arxiv.org/abs/2305.18114) | 本研究讨论了利用静态二元工具来识别动态效应问题，从而识别不同潜在群体和治疗暴露下的治疗效应加权和，但可能存在负权重。同时，我们在不同的假设设置下考虑了动态治疗效果的点估计和部分识别。 |
| [^4] | [Heterogeneous Noise and Stable Miscoordination.](http://arxiv.org/abs/2305.10301) | 学习动态下，样本大小的异质性可能导致博弈出现稳定的错协调，而非纯粹均衡，实证检验上有重要启示。 |

# 详细

[^1]: 强大的最大环算法：一种集成偏好排序的新方法

    The Strong Maximum Circulation Algorithm: A New Method for Aggregating Preference Rankings. (arXiv:2307.15702v1 [cs.SI])

    [http://arxiv.org/abs/2307.15702](http://arxiv.org/abs/2307.15702)

    强大的最大环算法提出了一种集成偏好排序的新方法，通过删除投票图中的最大环路，得出与投票结果一致的唯一排序结果。

    

    我们提出了一种基于优化的方法，用于在每个决策者或选民对一对选择进行偏好表达的情况下集成偏好。挑战在于在一些冲突的投票情况下，尽可能与投票结果一致地得出一个排序。只有不包含环路的投票集合才是非冲突的，并且可以在选择之间引发一个部分顺序。我们的方法是基于这样一个观察：构成一个环路的投票集合可以被视为平局。然后，方法是从投票图中删除环路的并集，并根据剩余部分确定集成偏好。我们引入了强大的最大环路，它由一组环路的并集形成，删除它可以保证在引发的部分顺序中获得唯一结果。此外，它还包含在消除任何最大环路后剩下的所有集成偏好。与之相反的是，wel

    We present a new optimization-based method for aggregating preferences in settings where each decision maker, or voter, expresses preferences over pairs of alternatives. The challenge is to come up with a ranking that agrees as much as possible with the votes cast in cases when some of the votes conflict. Only a collection of votes that contains no cycles is non-conflicting and can induce a partial order over alternatives. Our approach is motivated by the observation that a collection of votes that form a cycle can be treated as ties. The method is then to remove unions of cycles of votes, or circulations, from the vote graph and determine aggregate preferences from the remainder.  We introduce the strong maximum circulation which is formed by a union of cycles, the removal of which guarantees a unique outcome in terms of the induced partial order. Furthermore, it contains all the aggregate preferences remaining following the elimination of any maximum circulation. In contrast, the wel
    
[^2]: 模拟分解进行反事实预测

    Synthetic Decomposition for Counterfactual Predictions. (arXiv:2307.05122v1 [econ.EM])

    [http://arxiv.org/abs/2307.05122](http://arxiv.org/abs/2307.05122)

    本论文提出了一种使用“源”地区数据进行目标地区政策预测的方法。通过制定可转移条件并构建合成的结果-政策关系来满足条件。我们开发了通用过程来构建反事实预测的置信区间，并证明其有效性。本研究应用该方法预测了德克萨斯州青少年就业率。

    

    当政策变量超出先前政策支持范围时，反事实预测是具有挑战性的。然而，在许多情况下，关于感兴趣政策的信息可以从不同的“源”地区得到，这些地区已经实施了类似的政策。在本论文中，我们提出了一种新的方法，利用来自源地区的数据来预测目标地区的新政策。我们不依赖于使用参数化规范的结构关系的外推，而是制定一个可转移条件，并构建一个合成的结果-政策关系，使其尽可能接近满足条件。合成关系考虑了可观测数据和结构关系的相似性。我们开发了一个通用过程来构建反事实预测的渐进置信区间，并证明了其渐进有效性。然后，我们将我们的提议应用于预测德克萨斯州青少年就业率。

    Counterfactual predictions are challenging when the policy variable goes beyond its pre-policy support. However, in many cases, information about the policy of interest is available from different ("source") regions where a similar policy has already been implemented. In this paper, we propose a novel method of using such data from source regions to predict a new policy in a target region. Instead of relying on extrapolation of a structural relationship using a parametric specification, we formulate a transferability condition and construct a synthetic outcome-policy relationship such that it is as close as possible to meeting the condition. The synthetic relationship weighs both the similarity in distributions of observables and in structural relationships. We develop a general procedure to construct asymptotic confidence intervals for counterfactual predictions and prove its asymptotic validity. We then apply our proposal to predict average teenage employment in Texas following a cou
    
[^3]: 利用静态工具识别动态的最小平均处理效应

    Identifying Dynamic LATEs with a Static Instrument. (arXiv:2305.18114v1 [econ.EM])

    [http://arxiv.org/abs/2305.18114](http://arxiv.org/abs/2305.18114)

    本研究讨论了利用静态二元工具来识别动态效应问题，从而识别不同潜在群体和治疗暴露下的治疗效应加权和，但可能存在负权重。同时，我们在不同的假设设置下考虑了动态治疗效果的点估计和部分识别。

    

    在很多情况下，研究人员感兴趣的是用静态二元工具（IV）来识别不可逆治疗的动态效应。例如，在对培训计划的动态效应进行评估时，只需要单个抽奖来确定资格。在这些情况下，通常采用每个时期的IV估计方法。在标准IV假设的动态扩展下，我们展示了这种IV估计法可以识别不同潜在群体和治疗暴露下的治疗效应加权和。但是，有可能出现负权重。我们在不同的假设设置下考虑了这种情况下动态治疗效果的点估计和部分识别。

    In many situations, researchers are interested in identifying dynamic effects of an irreversible treatment with a static binary instrumental variable (IV). For example, in evaluations of dynamic effects of training programs, with a single lottery determining eligibility. A common approach in these situations is to report per-period IV estimates. Under a dynamic extension of standard IV assumptions, we show that such IV estimators identify a weighted sum of treatment effects for different latent groups and treatment exposures. However, there is possibility of negative weights. We consider point and partial identification of dynamic treatment effects in this setting under different sets of assumptions.
    
[^4]: 异质性噪声和稳定的错协调

    Heterogeneous Noise and Stable Miscoordination. (arXiv:2305.10301v1 [econ.TH])

    [http://arxiv.org/abs/2305.10301](http://arxiv.org/abs/2305.10301)

    学习动态下，样本大小的异质性可能导致博弈出现稳定的错协调，而非纯粹均衡，实证检验上有重要启示。

    

    协调博弈存在两种均衡：纯粹均衡，所有玩家成功协调其行动，和混合均衡，玩家经常经历错协调。现有文献表明，在许多进化动态下，人口从几乎任何初始行动分布中收敛于纯均衡。相反，我们显示，在合理的学习动态下（即代理人观察对手的随机样本并相应地调整策略），当样本大小存在异质性时，可产生稳定的错协调。这发生在一些代理人基于小样本（典型证据）做出决策，而其他代理人则依赖大样本。最后，我们在一个谈判应用中证明了我们的结果的实证相关性。

    Coordination games admit two types of equilibria: pure equilibria, where all players successfully coordinate their actions, and mixed equilibria, where players frequently experience miscoordination. The existing literature shows that under many evolutionary dynamics, populations converge to a pure equilibrium from almost any initial distribution of actions. By contrast, we show that under plausible learning dynamics, where agents observe the actions of a random sample of their opponents and adjust their strategies accordingly, stable miscoordination can arise when there is heterogeneity in the sample sizes. This occurs when some agents make decisions based on small samples (anecdotal evidence) while others rely on large samples. Finally, we demonstrate the empirical relevance of our results in a bargaining application.
    

