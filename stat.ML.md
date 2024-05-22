# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [What makes an image realistic?](https://arxiv.org/abs/2403.04493) | 论文讨论了如何设计能够可靠区分真实数据和不真实数据的函数，提出了通用评论者的概念作为一个新的解决方案。 |
| [^2] | [Criterion collapse and loss distribution control](https://arxiv.org/abs/2402.09802) | 该论文研究了"准则崩溃"的概念，即优化一个度量指标意味着另一个度量指标的最优性。研究结果发现，对于损失的伯努利分布，CVaR和DRO的结果远超出现有研究，同时发现了一些特定条件下，单调准则如倾斜ERM无法避免崩溃，而非单调的替代方案可以。 |
| [^3] | [Probabilistic Forecasting of Irregular Time Series via Conditional Flows](https://arxiv.org/abs/2402.06293) | 该论文提出了一种使用条件流进行不规则时间序列的概率预测的新模型ProFITi。该模型通过学习条件下未来值的联合分布，对具有缺失值的不规则时间序列进行预测，而不假设底层分布的固定形状。通过引入可逆三角形注意力层和可逆非线性激活函数，该模型取得了良好的实验结果。 |
| [^4] | [Computational Lower Bounds for Graphon Estimation via Low-degree Polynomials.](http://arxiv.org/abs/2308.15728) | 通过低次多项式计算图论估计存在计算障碍，传统的优化估计方法具有指数级的计算复杂度，而最优多项式时间估计器只能达到较慢的估计错误率。 |
| [^5] | [Incorporating Recklessness to Collaborative Filtering based Recommender Systems.](http://arxiv.org/abs/2308.02058) | 本文提出了一种将鲁莽行为引入基于矩阵分解的推荐系统学习过程的方法，通过控制风险水平来提高预测的数量和质量。 |

# 详细

[^1]: 使图像真实的因素是什么？

    What makes an image realistic?

    [https://arxiv.org/abs/2403.04493](https://arxiv.org/abs/2403.04493)

    论文讨论了如何设计能够可靠区分真实数据和不真实数据的函数，提出了通用评论者的概念作为一个新的解决方案。

    

    在过去的十年里，我们在生成看起来真实的数据方面取得了巨大进展，无论是图像、文本、音频还是视频。在这里，我们讨论了与之密切相关的问题，即量化现实主义，即设计能够可靠地区分真实数据和不真实数据的函数。从算法信息理论的观点出发，我们讨论了为什么这个问题很具挑战性，为什么一个好的生成模型单独不能解决它，以及一个好的解决方案应该是什么样的。特别是，我们引入了通用评论者的概念，不像对抗性评论者那样需要对抗性训练。尽管通用评论者并不立即实用，但它们既可以作为引导实际实现的北极星，也可以作为一个工具。

    arXiv:2403.04493v1 Announce Type: new  Abstract: The last decade has seen tremendous progress in our ability to generate realistic-looking data, be it images, text, audio, or video. Here, we discuss the closely related problem of quantifying realism, that is, designing functions that can reliably tell realistic data from unrealistic data. This problem turns out to be significantly harder to solve and remains poorly understood, despite its prevalence in machine learning and recent breakthroughs in generative AI. Drawing on insights from algorithmic information theory, we discuss why this problem is challenging, why a good generative model alone is insufficient to solve it, and what a good solution would look like. In particular, we introduce the notion of a universal critic, which unlike adversarial critics does not require adversarial training. While universal critics are not immediately practical, they can serve both as a North Star for guiding practical implementations and as a tool 
    
[^2]: 准则崩溃和损失分布控制

    Criterion collapse and loss distribution control

    [https://arxiv.org/abs/2402.09802](https://arxiv.org/abs/2402.09802)

    该论文研究了"准则崩溃"的概念，即优化一个度量指标意味着另一个度量指标的最优性。研究结果发现，对于损失的伯努利分布，CVaR和DRO的结果远超出现有研究，同时发现了一些特定条件下，单调准则如倾斜ERM无法避免崩溃，而非单调的替代方案可以。

    

    在这项工作中，我们考虑了"准则崩溃"的概念，即优化一个度量指标意味着另一个度量指标的最优性，特别关注各种学习准则下崩溃成误差概率最小化器的条件，从DRO和OCE风险（CVaR、倾斜ERM）到文献中探索的最新上升-下降算法的非单调准则（洪水、SoftAD）。我们展示了在伯努利分布损失的背景下，CVaR和DRO的现有结果远远超越了崩溃的范围，然后扩大了我们的范围，包括代理损失，展示了像倾斜ERM这样的单调准则无法避免崩溃的条件，而非单调的替代方案可以。

    arXiv:2402.09802v1 Announce Type: cross  Abstract: In this work, we consider the notion of "criterion collapse," in which optimization of one metric implies optimality in another, with a particular focus on conditions for collapse into error probability minimizers under a wide variety of learning criteria, ranging from DRO and OCE risks (CVaR, tilted ERM) to non-monotonic criteria underlying recent ascent-descent algorithms explored in the literature (Flooding, SoftAD). We show how collapse in the context of losses with a Bernoulli distribution goes far beyond existing results for CVaR and DRO, then expand our scope to include surrogate losses, showing conditions where monotonic criteria such as tilted ERM cannot avoid collapse, whereas non-monotonic alternatives can.
    
[^3]: 通过条件流进行不规则时间序列的概率预测

    Probabilistic Forecasting of Irregular Time Series via Conditional Flows

    [https://arxiv.org/abs/2402.06293](https://arxiv.org/abs/2402.06293)

    该论文提出了一种使用条件流进行不规则时间序列的概率预测的新模型ProFITi。该模型通过学习条件下未来值的联合分布，对具有缺失值的不规则时间序列进行预测，而不假设底层分布的固定形状。通过引入可逆三角形注意力层和可逆非线性激活函数，该模型取得了良好的实验结果。

    

    不规则采样的多变量时间序列具有缺失值的概率预测是许多领域的重要问题，包括医疗保健、天文学和气候学。目前该任务的最先进方法仅估计单个通道和单个时间点上观测值的边际分布，假设了一个固定形状的参数分布。在这项工作中，我们提出了一种新的模型ProFITi，用于使用条件归一化流对具有缺失值的不规则采样时间序列进行概率预测。该模型学习了在过去观测和查询的通道和时间上条件下时间序列未来值的联合分布，而不假设底层分布的固定形状。作为模型组件，我们引入了一种新颖的可逆三角形注意力层和一个可逆的非线性激活函数，能够在整个实数线上进行转换。我们在四个数据集上进行了大量实验，并证明了该模型的提议。

    Probabilistic forecasting of irregularly sampled multivariate time series with missing values is an important problem in many fields, including health care, astronomy, and climate. State-of-the-art methods for the task estimate only marginal distributions of observations in single channels and at single timepoints, assuming a fixed-shape parametric distribution. In this work, we propose a novel model, ProFITi, for probabilistic forecasting of irregularly sampled time series with missing values using conditional normalizing flows. The model learns joint distributions over the future values of the time series conditioned on past observations and queried channels and times, without assuming any fixed shape of the underlying distribution. As model components, we introduce a novel invertible triangular attention layer and an invertible non-linear activation function on and onto the whole real line. We conduct extensive experiments on four datasets and demonstrate that the proposed model pro
    
[^4]: 通过低次多项式计算图论估计的下界

    Computational Lower Bounds for Graphon Estimation via Low-degree Polynomials. (arXiv:2308.15728v1 [math.ST])

    [http://arxiv.org/abs/2308.15728](http://arxiv.org/abs/2308.15728)

    通过低次多项式计算图论估计存在计算障碍，传统的优化估计方法具有指数级的计算复杂度，而最优多项式时间估计器只能达到较慢的估计错误率。

    

    图论估计是网络分析中最基本的问题之一，在过去十年中受到了相当大的关注。从统计学的角度来看，高等提出了对于随机块模型（SBM）和非参数图论估计的图论估计的极小极差误差率。统计优化估计是基于约束最小二乘法，并且在维度上具有指数级的计算复杂度。从计算的角度来看，已知的最优多项式时间估计器是基于通用奇异值阈值（USVT），但是它只能达到比极小极差错误率慢得多的估计错误率。人们自然会想知道这样的差距是否是必要的。USVT的计算优化性或图论估计中的计算障碍的存在一直是一个长期存在的问题。在这项工作中，我们对此迈出了第一步，并为图论估计的计算障碍提供了严格的证据。

    Graphon estimation has been one of the most fundamental problems in network analysis and has received considerable attention in the past decade. From the statistical perspective, the minimax error rate of graphon estimation has been established by Gao et al (2015) for both stochastic block model (SBM) and nonparametric graphon estimation. The statistical optimal estimators are based on constrained least squares and have computational complexity exponential in the dimension. From the computational perspective, the best-known polynomial-time estimator is based on universal singular value thresholding (USVT), but it can only achieve a much slower estimation error rate than the minimax one. It is natural to wonder if such a gap is essential. The computational optimality of the USVT or the existence of a computational barrier in graphon estimation has been a long-standing open problem. In this work, we take the first step towards it and provide rigorous evidence for the computational barrie
    
[^5]: 整合鲁莽行为到基于协同过滤的推荐系统中

    Incorporating Recklessness to Collaborative Filtering based Recommender Systems. (arXiv:2308.02058v1 [cs.IR])

    [http://arxiv.org/abs/2308.02058](http://arxiv.org/abs/2308.02058)

    本文提出了一种将鲁莽行为引入基于矩阵分解的推荐系统学习过程的方法，通过控制风险水平来提高预测的数量和质量。

    

    包含可靠性测量的推荐系统往往在预测中更加保守，因为它们需要保持可靠性。这导致了这些系统可以提供的覆盖范围和新颖性的显著下降。在本文中，我们提出了在矩阵分解型推荐系统的学习过程中加入一项新的项，称为鲁莽行为，它可以控制在做出关于预测可靠性的决策时所希望的风险水平。实验结果表明，鲁莽行为不仅允许进行风险调控，还提高了推荐系统提供的预测的数量和质量。

    Recommender systems that include some reliability measure of their predictions tend to be more conservative in forecasting, due to their constraint to preserve reliability. This leads to a significant drop in the coverage and novelty that these systems can provide. In this paper, we propose the inclusion of a new term in the learning process of matrix factorization-based recommender systems, called recklessness, which enables the control of the risk level desired when making decisions about the reliability of a prediction. Experimental results demonstrate that recklessness not only allows for risk regulation but also improves the quantity and quality of predictions provided by the recommender system.
    

