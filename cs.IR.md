# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions](https://arxiv.org/abs/2403.15246) | 该论文引入了FollowIR数据集，包含严格的说明书评估基准和训练集，帮助信息检索模型更好地遵循真实世界的说明书。议论基于TREC会议的历史，旨在使信息检索模型能够根据详细说明书理解和判断相关性。 |
| [^2] | [Recommendation Fairness in Social Networks Over Time](https://arxiv.org/abs/2402.03450) | 本研究研究了社交网络推荐系统中推荐公平性随时间推移而变化的情况，并与动态网络属性进行了关联分析。结果表明，推荐公平性随时间改善，而少数群体比例和同质性与公平性有关。 |
| [^3] | [Link Me Baby One More Time: Social Music Discovery on Spotify.](http://arxiv.org/abs/2401.08818) | 本研究探讨了在社交音乐推荐中影响音乐互动的社交和环境因素。研究发现，接收者与发送者音乐品味相似、分享的音轨适合接收者的品味、接收者与发送者具有更强和更亲密的联系以及分享的艺术家在接收者的关系中受欢迎，这些因素都会增加接收者与新艺术家的互动。 |

# 详细

[^1]: FollowIR: 评估和教授信息检索模型以遵循说明书

    FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions

    [https://arxiv.org/abs/2403.15246](https://arxiv.org/abs/2403.15246)

    该论文引入了FollowIR数据集，包含严格的说明书评估基准和训练集，帮助信息检索模型更好地遵循真实世界的说明书。议论基于TREC会议的历史，旨在使信息检索模型能够根据详细说明书理解和判断相关性。

    

    现代大型语言模型（LLMs）能够遵循长且复杂的说明书，从而实现多样化的用户任务。然而，尽管信息检索（IR）模型使用LLMs作为其架构的支柱，几乎所有这些模型仍然只接受查询作为输入，没有说明书。对于最近一些接受说明书的模型来说，它们如何使用这些说明书还不清楚。我们引入了FollowIR数据集，其中包含严格的说明书评估基准，以及一个训练集，帮助IR模型学习更好地遵循现实世界的说明书。FollowIR基于TREC会议的悠久历史：正如TREC为人类标注员提供说明书（也称为叙述）来判断文档的相关性一样，因此IR模型应该能够根据这些详细说明书理解和确定相关性。我们的评估基准从三个经过深度判断的TREC收藏开始

    arXiv:2403.15246v1 Announce Type: cross  Abstract: Modern Large Language Models (LLMs) are capable of following long and complex instructions that enable a diverse amount of user tasks. However, despite Information Retrieval (IR) models using LLMs as the backbone of their architectures, nearly all of them still only take queries as input, with no instructions. For the handful of recent models that do take instructions, it's unclear how they use them. We introduce our dataset FollowIR, which contains a rigorous instruction evaluation benchmark as well as a training set for helping IR models learn to better follow real-world instructions. FollowIR builds off the long history of the TREC conferences: as TREC provides human annotators with instructions (also known as narratives) to determine document relevance, so should IR models be able to understand and decide relevance based on these detailed instructions. Our evaluation benchmark starts with three deeply judged TREC collections and al
    
[^2]: 社交网络随时间推移中的推荐公平性

    Recommendation Fairness in Social Networks Over Time

    [https://arxiv.org/abs/2402.03450](https://arxiv.org/abs/2402.03450)

    本研究研究了社交网络推荐系统中推荐公平性随时间推移而变化的情况，并与动态网络属性进行了关联分析。结果表明，推荐公平性随时间改善，而少数群体比例和同质性与公平性有关。

    

    在社交推荐系统中，推荐模型提供不同人口群体（如性别或种族）公平的可见性是至关重要的。现有大多数研究只研究了网络的固定快照，而忽视了网络随时间推移的变化。为了填补这一研究空白，我们研究了推荐公平性随时间的演变及其与动态网络属性的关系。我们通过评估六种推荐算法的公平性，并分析公平性与网络属性随时间的关联，来研究三个真实世界动态网络。我们还通过研究替代演化结果和不同网络属性的对照情景，来研究对网络属性的干预如何影响公平性。我们在实证数据集上的结果表明，不论推荐方法如何，推荐公平性随时间的改善。我们还发现两个网络属性，少数群体比例和同质性，与公平性有关。

    In social recommender systems, it is crucial that the recommendation models provide equitable visibility for different demographic groups, such as gender or race. Most existing research has addressed this problem by only studying individual static snapshots of networks that typically change over time. To address this gap, we study the evolution of recommendation fairness over time and its relation to dynamic network properties. We examine three real-world dynamic networks by evaluating the fairness of six recommendation algorithms and analyzing the association between fairness and network properties over time. We further study how interventions on network properties influence fairness by examining counterfactual scenarios with alternative evolution outcomes and differing network properties. Our results on empirical datasets suggest that recommendation fairness improves over time, regardless of the recommendation method. We also find that two network properties, minority ratio, and homo
    
[^3]: Link Me Baby One More Time: 在 Spotify 上的社交音乐发现

    Link Me Baby One More Time: Social Music Discovery on Spotify. (arXiv:2401.08818v1 [cs.SI])

    [http://arxiv.org/abs/2401.08818](http://arxiv.org/abs/2401.08818)

    本研究探讨了在社交音乐推荐中影响音乐互动的社交和环境因素。研究发现，接收者与发送者音乐品味相似、分享的音轨适合接收者的品味、接收者与发送者具有更强和更亲密的联系以及分享的艺术家在接收者的关系中受欢迎，这些因素都会增加接收者与新艺术家的互动。

    

    我们探讨影响个人之间音乐推荐和发现结果的社交和环境因素。具体来说，我们使用 Spotify 的数据来研究用户之间发送链接导致接收者与分享的艺术家的音乐互动。我们考虑了几个可能影响这一过程的因素，如发送者与接收者的关系强度，用户在 Spotify 社交网络中的角色，他们的音乐社交凝聚力，以及新艺术家与接收者的品味相似程度。我们发现，当接收者与发送者的音乐品味相似且分享的音轨适合他们的品味时，他们更有可能与新艺术家互动；当他们与发送者有更强和更亲密的联系时，也更有可能互动；以及当分享的艺术家在接收者的关系中受欢迎时，也更有可能互动。最后，我们利用这些发现构建了一个随机森林分类器，用于预测分享的音乐轨道是否会导致接收者的互动。

    We explore the social and contextual factors that influence the outcome of person-to-person music recommendations and discovery. Specifically, we use data from Spotify to investigate how a link sent from one user to another results in the receiver engaging with the music of the shared artist. We consider several factors that may influence this process, such as the strength of the sender-receiver relationship, the user's role in the Spotify social network, their music social cohesion, and how similar the new artist is to the receiver's taste. We find that the receiver of a link is more likely to engage with a new artist when (1) they have similar music taste to the sender and the shared track is a good fit for their taste, (2) they have a stronger and more intimate tie with the sender, and (3) the shared artist is popular with the receiver's connections. Finally, we use these findings to build a Random Forest classifier to predict whether a shared music track will result in the receiver
    

