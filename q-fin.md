# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Probability Distributions of Intraday Electricity Prices.](http://arxiv.org/abs/2310.02867) | 该论文提出了一种利用机器学习方法对电力日内价格概率进行预测的新方法，该方法通过学习数据中的经验分布选择最佳分布，并利用分布神经网络学习复杂模式，优于现有的基准模型。 |
| [^2] | [Advertiser Learning in Direct Advertising Markets.](http://arxiv.org/abs/2307.07015) | 广告商在直接广告市场中学习如何选择网站上的广告，调整他们的先验信念，并通过集合广告商信息来提高广告效果。 |

# 详细

[^1]: 学习电力日内价格概率分布

    Learning Probability Distributions of Intraday Electricity Prices. (arXiv:2310.02867v1 [econ.GN])

    [http://arxiv.org/abs/2310.02867](http://arxiv.org/abs/2310.02867)

    该论文提出了一种利用机器学习方法对电力日内价格概率进行预测的新方法，该方法通过学习数据中的经验分布选择最佳分布，并利用分布神经网络学习复杂模式，优于现有的基准模型。

    

    我们提出了一种新颖的机器学习方法，用于对小时级电力日内价格进行概率预测。与最近在数据丰富的概率预测方面的进展不同，该方法是非参数的，并从数据中学习到所有可能的经验分布中选择最佳分布。我们提出的模型是一种具有单调调整惩罚的多输出神经网络。这样的分布神经网络可以从数据丰富的环境中学习到电力价格的复杂模式，并且优于最先进的基准模型。

    We propose a novel machine learning approach to probabilistic forecasting of hourly intraday electricity prices. In contrast to recent advances in data-rich probabilistic forecasting that approximate the distributions with some features such as moments, our method is non-parametric and selects the best distribution from all possible empirical distributions learned from the data. The model we propose is a multiple output neural network with a monotonicity adjusting penalty. Such a distributional neural network can learn complex patterns in electricity prices from data-rich environments and it outperforms state-of-the-art benchmarks.
    
[^2]: 广告商在直接广告市场中的学习

    Advertiser Learning in Direct Advertising Markets. (arXiv:2307.07015v1 [econ.GN])

    [http://arxiv.org/abs/2307.07015](http://arxiv.org/abs/2307.07015)

    广告商在直接广告市场中学习如何选择网站上的广告，调整他们的先验信念，并通过集合广告商信息来提高广告效果。

    

    直接购买广告的广告商以固定价格从发布者和广告网络手中购买广告库存。这些广告商面临着在众多新的发布者网站中选择广告的复杂任务。我们提供证据表明广告商在做出这些选择时并不出色。相反，他们会在选择一个偏爱的集合之前尝试许多网站，与广告商学习相一致。随后，我们对广告商对发布者库存的需求进行建模，广告商可以在发布者的网站上了解广告效果。结果表明，广告商在以后放弃的网站上花费了大量资源进行广告投放，部分原因是他们对这些网站的广告效果存在过于乐观的先验信念。新网站上的广告商的预期点击率中位数为0.23%，比真实的中位数点击率0.045%高五倍。我们考虑了如何通过集合广告商信息来解决这个问题。具体而言，我们展示了具有类似视觉元素的广告可以获得类似的点击率。

    Direct buy advertisers procure advertising inventory at fixed rates from publishers and ad networks. Such advertisers face the complex task of choosing ads amongst myriad new publisher sites. We offer evidence that advertisers do not excel at making these choices. Instead, they try many sites before settling on a favored set, consistent with advertiser learning. We subsequently model advertiser demand for publisher inventory wherein advertisers learn about advertising efficacy across publishers' sites. Results suggest that advertisers spend considerable resources advertising on sites they eventually abandon -- in part because their prior beliefs about advertising efficacy on those sites are too optimistic. The median advertiser's expected CTR at a new site is 0.23%, five times higher than the true median CTR of 0.045%.  We consider how pooling advertiser information remediates this problem. Specifically, we show that ads with similar visual elements garner similar CTRs, enabling advert
    

