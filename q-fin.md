# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Enhanced Local Explainability of Random Forests: a Proximity-Based Approach.](http://arxiv.org/abs/2310.12428) | 这项研究提出了一种利用随机森林模型的特征空间中的邻近性来解释模型预测的方法，为模型预测提供了局部的解释性，与现有方法相辅相成。通过实验证明了这种方法在债券定价模型中的有效性。 |
| [^2] | [Visibility graph analysis of the grains and oilseeds indices.](http://arxiv.org/abs/2304.05760) | 本研究对粮油指数及其五个子指数进行了可见性图分析，六个可见性图都表现出幂律分布的度分布和小世界特征。玉米和大豆指数的可见性图表现出较弱的同配混合模式。 |

# 详细

[^1]: 实现随机森林的局部可解释性增强：基于邻近性的方法

    Towards Enhanced Local Explainability of Random Forests: a Proximity-Based Approach. (arXiv:2310.12428v1 [stat.ML])

    [http://arxiv.org/abs/2310.12428](http://arxiv.org/abs/2310.12428)

    这项研究提出了一种利用随机森林模型的特征空间中的邻近性来解释模型预测的方法，为模型预测提供了局部的解释性，与现有方法相辅相成。通过实验证明了这种方法在债券定价模型中的有效性。

    

    我们提出一种新的方法来解释随机森林（RF）模型的样本外性能，利用了任何RF都可以被表述为自适应加权K最近邻（KNN）模型的事实。具体而言，我们利用RF在特征空间中学到的点之间的邻近性，将随机森林的预测重写为训练数据点目标标签的加权平均值。这种线性性质有助于在训练集观测中为任何模型预测生成属性，从而为RF预测提供了局部的解释性，补充了SHAP等已有方法，这些方法则为特征空间维度上的模型预测生成属性。我们在训练于美国公司债券交易数据的债券定价模型中演示了这种方法，并将其与各种现有的模型解释方法进行了比较。

    We initiate a novel approach to explain the out of sample performance of random forest (RF) models by exploiting the fact that any RF can be formulated as an adaptive weighted K nearest-neighbors model. Specifically, we use the proximity between points in the feature space learned by the RF to re-write random forest predictions exactly as a weighted average of the target labels of training data points. This linearity facilitates a local notion of explainability of RF predictions that generates attributions for any model prediction across observations in the training set, and thereby complements established methods like SHAP, which instead generates attributions for a model prediction across dimensions of the feature space. We demonstrate this approach in the context of a bond pricing model trained on US corporate bond trades, and compare our approach to various existing approaches to model explainability.
    
[^2]: 粮油指数的可见性图分析

    Visibility graph analysis of the grains and oilseeds indices. (arXiv:2304.05760v1 [econ.GN])

    [http://arxiv.org/abs/2304.05760](http://arxiv.org/abs/2304.05760)

    本研究对粮油指数及其五个子指数进行了可见性图分析，六个可见性图都表现出幂律分布的度分布和小世界特征。玉米和大豆指数的可见性图表现出较弱的同配混合模式。

    

    粮油指数（GOI）及其小麦、玉米、大豆、稻米和大麦等五个子指数是每日价格指数，反映了全球主要农产品现货市场价格的变化。本文对GOI及其五个子指数进行了可见性图（VG）分析。最大似然估计表明，VG的度分布都显示出幂律尾巴，除了稻米。六个VG的平均聚类系数都很大（>0.5），并与VG的平均度数展现了良好的幂律关系。对于每个VG，节点的聚类系数在大度数时与其度数成反比，在小度数时与其度数成幂律相关。所有六个VG都表现出小世界特征，但程度不同。度-度相关系数表明，玉米和大豆指数的VG表现出较弱的同配混合模式，而其他四个VG的同配混合模式较弱。

    The Grains and Oilseeds Index (GOI) and its sub-indices of wheat, maize, soyabeans, rice, and barley are daily price indexes reflect the price changes of the global spot markets of staple agro-food crops. In this paper, we carry out a visibility graph (VG) analysis of the GOI and its five sub-indices. Maximum likelihood estimation shows that the degree distributions of the VGs display power-law tails, except for rice. The average clustering coefficients of the six VGs are quite large (>0.5) and exhibit a nice power-law relation with respect to the average degrees of the VGs. For each VG, the clustering coefficients of nodes are inversely proportional to their degrees for large degrees and are correlated to their degrees as a power law for small degrees. All the six VGs exhibit small-world characteristics to some extent. The degree-degree correlation coefficients shows that the VGs for maize and soyabeans indices exhibit weak assortative mixing patterns, while the other four VGs are wea
    

