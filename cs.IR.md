# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems.](http://arxiv.org/abs/2311.00388) | 本论文提出了一个名为AutoSAM的自动采样框架，用于对连续推荐系统中的用户行为进行非均匀处理。该框架通过自适应地学习历史行为的偏斜分布，并采样出信息丰富的子集，以构建更具可泛化性的连续推荐系统。 |
| [^2] | [The Search for Stability: Learning Dynamics of Strategic Publishers with Initial Documents.](http://arxiv.org/abs/2305.16695) | 本研究在信息检索博弈论模型中提出了相对排名原则（RRP）作为替代排名原则，以达成更稳定的搜索生态系统，并提供了理论和实证证据证明其学习动力学收敛性，同时展示了可能的出版商-用户权衡。 |

# 详细

[^1]: 实现自动采样对于连续推荐系统中用户行为的研究

    Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems. (arXiv:2311.00388v1 [cs.IR])

    [http://arxiv.org/abs/2311.00388](http://arxiv.org/abs/2311.00388)

    本论文提出了一个名为AutoSAM的自动采样框架，用于对连续推荐系统中的用户行为进行非均匀处理。该框架通过自适应地学习历史行为的偏斜分布，并采样出信息丰富的子集，以构建更具可泛化性的连续推荐系统。

    

    由于连续推荐系统能够有效捕捉动态用户偏好，因此它们在推荐领域中广受欢迎。当前连续推荐系统的一个默认设置是将每个历史行为均匀地视为正向交互。然而，实际上，这种设置有可能导致性能不佳，因为每个商品对用户的兴趣有不同的贡献。例如，购买的商品应该比点击的商品更重要。因此，我们提出了一个通用的自动采样框架，名为AutoSAM，用于非均匀地处理历史行为。具体而言，AutoSAM通过在标准的连续推荐架构中增加一个采样器层，自适应地学习原始输入的偏斜分布，并采样出信息丰富的子集，以构建更具可泛化性的连续推荐系统。为了克服非可微分采样操作的挑战，同时引入多个决策因素进行采样，我们还提出了进一步的方法。

    Sequential recommender systems (SRS) have gained widespread popularity in recommendation due to their ability to effectively capture dynamic user preferences. One default setting in the current SRS is to uniformly consider each historical behavior as a positive interaction. Actually, this setting has the potential to yield sub-optimal performance, as each item makes a distinct contribution to the user's interest. For example, purchased items should be given more importance than clicked ones. Hence, we propose a general automatic sampling framework, named AutoSAM, to non-uniformly treat historical behaviors. Specifically, AutoSAM augments the standard sequential recommendation architecture with an additional sampler layer to adaptively learn the skew distribution of the raw input, and then sample informative sub-sets to build more generalizable SRS. To overcome the challenges of non-differentiable sampling actions and also introduce multiple decision factors for sampling, we further int
    
[^2]: 寻求稳定性：具有初始文件的战略出版商的学习动态的研究

    The Search for Stability: Learning Dynamics of Strategic Publishers with Initial Documents. (arXiv:2305.16695v1 [cs.GT])

    [http://arxiv.org/abs/2305.16695](http://arxiv.org/abs/2305.16695)

    本研究在信息检索博弈论模型中提出了相对排名原则（RRP）作为替代排名原则，以达成更稳定的搜索生态系统，并提供了理论和实证证据证明其学习动力学收敛性，同时展示了可能的出版商-用户权衡。

    

    我们研究了一种信息检索的博弈论模型，其中战略出版商旨在在保持原始文档完整性的同时最大化自己排名第一的机会。我们表明，常用的PRP排名方案导致环境不稳定，游戏经常无法达到纯纳什均衡。我们将相对排名原则（RRP）作为替代排名原则，并介绍两个排名函数，它们是RRP的实例。我们提供了理论和实证证据，表明这些方法导致稳定的搜索生态系统，通过提供关于学习动力学收敛的积极结果。我们还定义出版商和用户的福利，并展示了可能的出版商-用户权衡，突显了确定搜索引擎设计师应选择哪种排名函数的复杂性。

    We study a game-theoretic model of information retrieval, in which strategic publishers aim to maximize their chances of being ranked first by the search engine, while maintaining the integrity of their original documents. We show that the commonly used PRP ranking scheme results in an unstable environment where games often fail to reach pure Nash equilibrium. We propose the Relative Ranking Principle (RRP) as an alternative ranking principle, and introduce two ranking functions that are instances of the RRP. We provide both theoretical and empirical evidence that these methods lead to a stable search ecosystem, by providing positive results on the learning dynamics convergence. We also define the publishers' and users' welfare, and demonstrate a possible publisher-user trade-off, which highlights the complexity of determining which ranking function should be selected by the search engine designer.
    

