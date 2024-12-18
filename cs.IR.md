# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Non-autoregressive Generative Models for Reranking Recommendation](https://arxiv.org/abs/2402.06871) | 本研究提出了一个非自回归的生成模型用于排序推荐，在多阶段推荐系统中扮演关键角色。该模型旨在提高效率和效果，并解决稀疏训练样本和动态候选项对模型收敛性的挑战。 |
| [^2] | [Impression-Aware Recommender Systems.](http://arxiv.org/abs/2308.07857) | 基于印象的推荐系统利用印象数据源提升推荐质量，通过综述分类推荐系统、数据集和评估方法，揭示开放性问题和未来研究方向。 |

# 详细

[^1]: 非自回归的生成模型用于排序推荐

    Non-autoregressive Generative Models for Reranking Recommendation

    [https://arxiv.org/abs/2402.06871](https://arxiv.org/abs/2402.06871)

    本研究提出了一个非自回归的生成模型用于排序推荐，在多阶段推荐系统中扮演关键角色。该模型旨在提高效率和效果，并解决稀疏训练样本和动态候选项对模型收敛性的挑战。

    

    在多阶段推荐系统中，重新排序通过建模项目之间的内部相关性起到了至关重要的作用。重新排序的关键挑战在于在排列的组合空间中探索最佳序列。最近的研究提出了生成器-评估器学习范式，生成器生成多个可行序列，评估器基于估计的列表得分选择最佳序列。生成器至关重要，而生成模型非常适合生成器函数。当前的生成模型采用自回归策略进行序列生成。然而，在实时工业系统中部署自回归模型是具有挑战性的。因此，我们提出了一个非自回归生成模型用于排序推荐（NAR4Rec），以提高效率和效果。为了解决与稀疏训练样本和动态候选项对模型收敛性的挑战，我们引入了一个m

    In a multi-stage recommendation system, reranking plays a crucial role by modeling the intra-list correlations among items.The key challenge of reranking lies in the exploration of optimal sequences within the combinatorial space of permutations. Recent research proposes a generator-evaluator learning paradigm, where the generator generates multiple feasible sequences and the evaluator picks out the best sequence based on the estimated listwise score. Generator is of vital importance, and generative models are well-suited for the generator function. Current generative models employ an autoregressive strategy for sequence generation. However, deploying autoregressive models in real-time industrial systems is challenging. Hence, we propose a Non-AutoRegressive generative model for reranking Recommendation (NAR4Rec) designed to enhance efficiency and effectiveness. To address challenges related to sparse training samples and dynamic candidates impacting model convergence, we introduce a m
    
[^2]: 基于印象的推荐系统

    Impression-Aware Recommender Systems. (arXiv:2308.07857v1 [cs.IR])

    [http://arxiv.org/abs/2308.07857](http://arxiv.org/abs/2308.07857)

    基于印象的推荐系统利用印象数据源提升推荐质量，通过综述分类推荐系统、数据集和评估方法，揭示开放性问题和未来研究方向。

    

    新型数据源为改进推荐系统的质量带来了新的机遇。印象是一种包含过去推荐（展示的项目）和传统互动的新型数据源。研究人员可以利用印象来优化用户偏好并克服当前推荐系统研究中的限制。印象的相关性和兴趣度逐年增加，因此需要对这类推荐系统中相关工作进行综述。我们提出了一篇关于使用印象的推荐系统的系统文献综述，侧重于研究中的三个基本方面：推荐系统、数据集和评估方法。我们对使用印象的推荐系统的论文进行了三个分类，详细介绍了每篇综述论文，描述了具有印象的数据集，并分析了现有的评估方法。最后，我们提出了值得关注的开放性问题和未来的研究方向，强调了文献中缺失的方面。

    Novel data sources bring new opportunities to improve the quality of recommender systems. Impressions are a novel data source containing past recommendations (shown items) and traditional interactions. Researchers may use impressions to refine user preferences and overcome the current limitations in recommender systems research. The relevance and interest of impressions have increased over the years; hence, the need for a review of relevant work on this type of recommenders. We present a systematic literature review on recommender systems using impressions, focusing on three fundamental angles in research: recommenders, datasets, and evaluation methodologies. We provide three categorizations of papers describing recommenders using impressions, present each reviewed paper in detail, describe datasets with impressions, and analyze the existing evaluation methodologies. Lastly, we present open questions and future directions of interest, highlighting aspects missing in the literature that
    

