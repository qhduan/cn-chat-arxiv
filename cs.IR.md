# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [More Is Less: When Do Recommenders Underperform for Data-rich Users?.](http://arxiv.org/abs/2304.07487) | 研究了推荐算法在数据量丰富和数据量贫乏的用户中的性能表现。发现在所有数据集中，精度在数据丰富的用户中始终更高；平均精度相当，但其方差很大；当评估过程中采用负样本抽样时，召回率产生反直觉结果，表现更好的是数据贫乏的用户；随着用户与推荐系统的互动增加，他们收到的推荐质量会降低。 |

# 详细

[^1]: 更多不一定就是更好：何时推荐算法在数据丰富的用户中表现不佳？

    More Is Less: When Do Recommenders Underperform for Data-rich Users?. (arXiv:2304.07487v1 [cs.IR])

    [http://arxiv.org/abs/2304.07487](http://arxiv.org/abs/2304.07487)

    研究了推荐算法在数据量丰富和数据量贫乏的用户中的性能表现。发现在所有数据集中，精度在数据丰富的用户中始终更高；平均精度相当，但其方差很大；当评估过程中采用负样本抽样时，召回率产生反直觉结果，表现更好的是数据贫乏的用户；随着用户与推荐系统的互动增加，他们收到的推荐质量会降低。

    

    推荐系统的用户通常在与算法互动的水平上有所不同，这可能影响他们收到推荐的质量，并导致不可取的性能差异。本文研究了对于十个基准数据集应用的一组流行评估指标，数据丰富和数据贫乏的用户性能在什么条件下会发散。我们发现，针对所有数据集，精度在数据丰富的用户中始终更高；平均精度均等，但其方差很大；召回率产生了一个反直觉的结果，算法在数据贫乏的用户中表现更好，当在评估过程中采用负样本抽样时，这种偏差更加严重。最后一个观察结果表明，随着用户与推荐系统的互动增加，他们收到的推荐质量会降低（以召回率衡量）。我们的研究清楚地表明，在现实世界设置中，评估合理的推荐系统很重要，因为不同用户有不同的系统互作程度。

    Users of recommender systems tend to differ in their level of interaction with these algorithms, which may affect the quality of recommendations they receive and lead to undesirable performance disparity. In this paper we investigate under what conditions the performance for data-rich and data-poor users diverges for a collection of popular evaluation metrics applied to ten benchmark datasets. We find that Precision is consistently higher for data-rich users across all the datasets; Mean Average Precision is comparable across user groups but its variance is large; Recall yields a counter-intuitive result where the algorithm performs better for data-poor than for data-rich users, which bias is further exacerbated when negative item sampling is employed during evaluation. The final observation suggests that as users interact more with recommender systems, the quality of recommendations they receive degrades (when measured by Recall). Our insights clearly show the importance of an evaluat
    

