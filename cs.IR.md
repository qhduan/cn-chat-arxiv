# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FMMRec: Fairness-aware Multimodal Recommendation.](http://arxiv.org/abs/2310.17373) | 本论文提出了一种名为FMMRec的公平感知多模态推荐方法，通过从模态表示中分离敏感和非敏感信息，实现更公平的表示学习。 |

# 详细

[^1]: FMMRec: 公平感知的多模态推荐

    FMMRec: Fairness-aware Multimodal Recommendation. (arXiv:2310.17373v1 [cs.IR])

    [http://arxiv.org/abs/2310.17373](http://arxiv.org/abs/2310.17373)

    本论文提出了一种名为FMMRec的公平感知多模态推荐方法，通过从模态表示中分离敏感和非敏感信息，实现更公平的表示学习。

    

    最近，多模态推荐因为可以有效解决数据稀疏问题并结合各种模态的表示而受到越来越多的关注。尽管多模态推荐在准确性方面表现出色，但引入不同的模态（例如图像、文本和音频）可能会将更多用户的敏感信息（例如性别和年龄）暴露给推荐系统，从而导致更严重的不公平问题。尽管已经有很多关于公平性的努力，但现有的公平性方法要么与多模态情境不兼容，要么由于忽视多模态内容的敏感信息而导致公平性性能下降。为了在多模态推荐中实现反事实公平性，我们提出了一种新颖的公平感知多模态推荐方法（称为FMMRec），通过从模态表示中分离敏感和非敏感信息，并利用分离后的模态表示来指导更公平的表示学习过程。

    Recently, multimodal recommendations have gained increasing attention for effectively addressing the data sparsity problem by incorporating modality-based representations. Although multimodal recommendations excel in accuracy, the introduction of different modalities (e.g., images, text, and audio) may expose more users' sensitive information (e.g., gender and age) to recommender systems, resulting in potentially more serious unfairness issues. Despite many efforts on fairness, existing fairness-aware methods are either incompatible with multimodal scenarios, or lead to suboptimal fairness performance due to neglecting sensitive information of multimodal content. To achieve counterfactual fairness in multimodal recommendations, we propose a novel fairness-aware multimodal recommendation approach (dubbed as FMMRec) to disentangle the sensitive and non-sensitive information from modal representations and leverage the disentangled modal representations to guide fairer representation learn
    

