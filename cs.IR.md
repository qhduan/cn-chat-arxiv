# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems.](http://arxiv.org/abs/2311.00388) | 本论文提出了一个名为AutoSAM的自动采样框架，用于对连续推荐系统中的用户行为进行非均匀处理。该框架通过自适应地学习历史行为的偏斜分布，并采样出信息丰富的子集，以构建更具可泛化性的连续推荐系统。 |

# 详细

[^1]: 实现自动采样对于连续推荐系统中用户行为的研究

    Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems. (arXiv:2311.00388v1 [cs.IR])

    [http://arxiv.org/abs/2311.00388](http://arxiv.org/abs/2311.00388)

    本论文提出了一个名为AutoSAM的自动采样框架，用于对连续推荐系统中的用户行为进行非均匀处理。该框架通过自适应地学习历史行为的偏斜分布，并采样出信息丰富的子集，以构建更具可泛化性的连续推荐系统。

    

    由于连续推荐系统能够有效捕捉动态用户偏好，因此它们在推荐领域中广受欢迎。当前连续推荐系统的一个默认设置是将每个历史行为均匀地视为正向交互。然而，实际上，这种设置有可能导致性能不佳，因为每个商品对用户的兴趣有不同的贡献。例如，购买的商品应该比点击的商品更重要。因此，我们提出了一个通用的自动采样框架，名为AutoSAM，用于非均匀地处理历史行为。具体而言，AutoSAM通过在标准的连续推荐架构中增加一个采样器层，自适应地学习原始输入的偏斜分布，并采样出信息丰富的子集，以构建更具可泛化性的连续推荐系统。为了克服非可微分采样操作的挑战，同时引入多个决策因素进行采样，我们还提出了进一步的方法。

    Sequential recommender systems (SRS) have gained widespread popularity in recommendation due to their ability to effectively capture dynamic user preferences. One default setting in the current SRS is to uniformly consider each historical behavior as a positive interaction. Actually, this setting has the potential to yield sub-optimal performance, as each item makes a distinct contribution to the user's interest. For example, purchased items should be given more importance than clicked ones. Hence, we propose a general automatic sampling framework, named AutoSAM, to non-uniformly treat historical behaviors. Specifically, AutoSAM augments the standard sequential recommendation architecture with an additional sampler layer to adaptively learn the skew distribution of the raw input, and then sample informative sub-sets to build more generalizable SRS. To overcome the challenges of non-differentiable sampling actions and also introduce multiple decision factors for sampling, we further int
    

