# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models.](http://arxiv.org/abs/2401.10690) | 本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。 |
| [^2] | [Sustainable Transparency in Recommender Systems: Bayesian Ranking of Images for Explainability.](http://arxiv.org/abs/2308.01196) | 这项研究旨在实现推荐系统的可持续透明性，并提出了一种使用贝叶斯排名图像进行个性化解释的方法，以最大化透明度和用户信任。 |

# 详细

[^1]: 超越RMSE和MAE：引入EAUC来揭示偏见和不公平的迪亚德回归模型中的隐藏因素

    Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models. (arXiv:2401.10690v1 [cs.LG])

    [http://arxiv.org/abs/2401.10690](http://arxiv.org/abs/2401.10690)

    本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。

    

    迪亚德回归模型用于预测一对实体的实值结果，在许多领域中都是基础的（例如，在推荐系统中预测用户对产品的评分），在许多其他领域中也有许多潜力但尚未深入探索（例如，在个性化药理学中近似确定患者的适当剂量）。本研究中，我们证明个体实体观察值分布的非均匀性导致了最先进模型中的严重偏见预测，偏向于实体的观察过去值的平均值，并在另类但同样重要的情况下提供比随机预测更差的预测能力。我们表明，全局错误度量标准如均方根误差（RMSE）和平均绝对误差（MAE）不足以捕捉到这种现象，我们将其命名为另类偏见，并引入另类-曲线下面积（EAUC）作为一个新的补充度量，可以在所有研究的模型中量化它。

    Dyadic regression models, which predict real-valued outcomes for pairs of entities, are fundamental in many domains (e.g. predicting the rating of a user to a product in Recommender Systems) and promising and under exploration in many others (e.g. approximating the adequate dosage of a drug for a patient in personalized pharmacology). In this work, we demonstrate that non-uniformity in the observed value distributions of individual entities leads to severely biased predictions in state-of-the-art models, skewing predictions towards the average of observed past values for the entity and providing worse-than-random predictive power in eccentric yet equally important cases. We show that the usage of global error metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) is insufficient to capture this phenomenon, which we name eccentricity bias, and we introduce Eccentricity-Area Under the Curve (EAUC) as a new complementary metric that can quantify it in all studied models
    
[^2]: 可持续透明的推荐系统: 用于解释性的贝叶斯图像排名

    Sustainable Transparency in Recommender Systems: Bayesian Ranking of Images for Explainability. (arXiv:2308.01196v1 [cs.IR])

    [http://arxiv.org/abs/2308.01196](http://arxiv.org/abs/2308.01196)

    这项研究旨在实现推荐系统的可持续透明性，并提出了一种使用贝叶斯排名图像进行个性化解释的方法，以最大化透明度和用户信任。

    

    推荐系统在现代世界中变得至关重要，通常指导用户找到相关的内容或产品，并对用户和公民的决策产生重大影响。然而，确保这些系统的透明度和用户信任仍然是一个挑战；个性化解释已经成为一个解决方案，为推荐提供理由。在生成个性化解释的现有方法中，使用用户创建的视觉内容是一个特别有潜力的选项，有潜力最大化透明度和用户信任。然而，现有模型在这个背景下解释推荐时存在一些限制：可持续性是一个关键问题，因为它们经常需要大量的计算资源，导致的碳排放量与它们被整合到推荐系统中相当。此外，大多数模型使用的替代学习目标与排名最有效的目标不一致。

    Recommender Systems have become crucial in the modern world, commonly guiding users towards relevant content or products, and having a large influence over the decisions of users and citizens. However, ensuring transparency and user trust in these systems remains a challenge; personalized explanations have emerged as a solution, offering justifications for recommendations. Among the existing approaches for generating personalized explanations, using visual content created by the users is one particularly promising option, showing a potential to maximize transparency and user trust. Existing models for explaining recommendations in this context face limitations: sustainability has been a critical concern, as they often require substantial computational resources, leading to significant carbon emissions comparable to the Recommender Systems where they would be integrated. Moreover, most models employ surrogate learning goals that do not align with the objective of ranking the most effect
    

