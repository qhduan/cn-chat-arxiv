# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ChatGPT and Persuasive Technologies for the Management and Delivery of Personalized Recommendations in Hotel Hospitality.](http://arxiv.org/abs/2307.14298) | 本文研究了将ChatGPT和说服技术应用于酒店推荐系统的潜力，通过ChatGPT可以提供更准确和上下文感知的推荐，而说服技术可影响用户行为并增强推荐的说服力。 |
| [^2] | [Large Language Models are Competitive Near Cold-start Recommenders for Language- and Item-based Preferences.](http://arxiv.org/abs/2307.14225) | 大规模语言模型（LLMs）在冷启动情况下提供了与基于项目协同过滤（CF）方法相当的推荐性能，特别是在纯基于语言偏好的情况下。 |
| [^3] | [A Probabilistic Position Bias Model for Short-Video Recommendation Feeds.](http://arxiv.org/abs/2307.14059) | 本论文提出了一种概率位置偏差模型，用于解决社交媒体平台上短视频推荐中的问题。这种模型考虑了用户在浏览视频时的行为特点，能够更准确地估计每个视频的曝光概率。 |
| [^4] | [Multi-view Hypergraph Contrastive Policy Learning for Conversational Recommendation.](http://arxiv.org/abs/2307.14024) | 这篇论文提出了一种多视图超图对比策略学习的方法，用于会话推荐系统。该方法综合考虑了用户的喜欢、社交和不喜欢三个视图，并通过对比学习用户偏好，从而提高会话推荐的准确性。 |
| [^5] | [Domain Disentanglement with Interpolative Data Augmentation for Dual-Target Cross-Domain Recommendation.](http://arxiv.org/abs/2307.13910) | 本文提出了一种基于插值数据增强的解缠方法（DIDA-CDR）用于双目标跨领域推荐，旨在解决生成相关且多样化的增强用户表示以及有效将领域无关信息与领域特定信息分离的挑战。 |
| [^6] | [ClusterSeq: Enhancing Sequential Recommender Systems with Clustering based Meta-Learning.](http://arxiv.org/abs/2307.13766) | ClusterSeq是一种基于聚类的元学习顺序推荐系统，通过利用用户序列的动态信息提高了物品预测的准确性，并保留了次要用户的偏好，并利用了同一聚类中用户的集体知识。 |
| [^7] | [Interface Design to Mitigate Inflation in Recommender Systems.](http://arxiv.org/abs/2307.12424) | 本研究通过分析音乐发现应用中的数据发现，评分膨胀问题来源于异质的用户评分行为和个性化推荐的动态，通过修改评分界面可以显著改善这一问题。 |
| [^8] | [Bert4XMR: Cross-Market Recommendation with Bidirectional Encoder Representations from Transformer.](http://arxiv.org/abs/2305.15145) | 提出了一种名为Bert4XMR的新型跨市场推荐模型，能够建模不同市场的物品共现性，并减轻负迁移问题。 |
| [^9] | [Towards Hierarchical Policy Learning for Conversational Recommendation with Hypergraph-based Reinforcement Learning.](http://arxiv.org/abs/2305.02575) | 本文提出了一种新颖的基于超图强化学习的分层对话推荐模型，其中导演通过超图算法进行选择，帮助演员减少行动空间和指导对话朝着最具信息性的属性方向进行，并根据用户在对话中的偏好选择物品。 |
| [^10] | [Fairness in Recommendation: Foundations, Methods and Applications.](http://arxiv.org/abs/2205.13619) | 这篇论文对推荐系统中的公平性问题进行了系统调查，针对推荐过程中可能出现的数据或算法偏见，提供了一些方法和应用来提升推荐中的公平性。 |
| [^11] | [Sequential Recommendation with Graph Neural Networks.](http://arxiv.org/abs/2106.14226) | 提出了SURGE（SeqUential Recommendation with Graph neural networks）模型，通过图神经网络将用户历史行为中的不同偏好聚类成紧密的兴趣图，以更好地预测顺序推荐中用户的下一次互动。 |

# 详细

[^1]: ChatGPT和说服技术在酒店服务领域个性化推荐管理和提供中的应用

    ChatGPT and Persuasive Technologies for the Management and Delivery of Personalized Recommendations in Hotel Hospitality. (arXiv:2307.14298v1 [cs.IR])

    [http://arxiv.org/abs/2307.14298](http://arxiv.org/abs/2307.14298)

    本文研究了将ChatGPT和说服技术应用于酒店推荐系统的潜力，通过ChatGPT可以提供更准确和上下文感知的推荐，而说服技术可影响用户行为并增强推荐的说服力。

    

    推荐系统在酒店服务业已成为不可或缺的工具，为客人提供个性化和定制化的体验。近年来，大型语言模型（LLM），如ChatGPT和说服技术的进步，为提升这些系统的效果打开了新的途径。本文探讨了将ChatGPT和说服技术整合到酒店服务推荐系统中自动化和改进的潜力。首先，我们深入研究了ChatGPT的能力，它可以理解和生成类似人类的文本，从而实现更准确和上下文感知的推荐。我们讨论了将ChatGPT整合到推荐系统中的能力，突出了其分析用户偏好、从在线评论中提取有价值的洞见，并根据客人配置生成个性化推荐的能力。其次，我们研究了说服技术在影响用户行为和提升酒店推荐的说服效果方面的作用。

    Recommender systems have become indispensable tools in the hotel hospitality industry, enabling personalized and tailored experiences for guests. Recent advancements in large language models (LLMs), such as ChatGPT, and persuasive technologies, have opened new avenues for enhancing the effectiveness of those systems. This paper explores the potential of integrating ChatGPT and persuasive technologies for automating and improving hotel hospitality recommender systems. First, we delve into the capabilities of ChatGPT, which can understand and generate human-like text, enabling more accurate and context-aware recommendations. We discuss the integration of ChatGPT into recommender systems, highlighting the ability to analyze user preferences, extract valuable insights from online reviews, and generate personalized recommendations based on guest profiles. Second, we investigate the role of persuasive technology in influencing user behavior and enhancing the persuasive impact of hotel recomm
    
[^2]: 大规模语言模型在冷启动推荐系统中与基于语言和基于项目偏好竞争力相当

    Large Language Models are Competitive Near Cold-start Recommenders for Language- and Item-based Preferences. (arXiv:2307.14225v1 [cs.IR])

    [http://arxiv.org/abs/2307.14225](http://arxiv.org/abs/2307.14225)

    大规模语言模型（LLMs）在冷启动情况下提供了与基于项目协同过滤（CF）方法相当的推荐性能，特别是在纯基于语言偏好的情况下。

    

    传统的推荐系统利用用户的项目偏好历史来推荐用户可能喜欢的新内容。然而，现代对话界面允许用户表达基于语言的偏好，提供了一种根本不同的偏好输入方式。受最近大规模语言模型（LLMs）提示范式的成功启发，我们研究了它们在基于项目和基于语言偏好方面与最先进的基于项目协同过滤（CF）方法相比的推荐应用。为了支持这项研究，我们收集了一个新的数据集，其中包含从用户那里引发出来的基于项目和基于语言偏好，以及他们对各种（有偏见的）推荐项目和（无偏见的）随机项目的评分。在众多实验结果中，我们发现在纯基于语言偏好（没有项目偏好）的情况下，LLMs在接近冷启动情况下与基于项目的CF方法相比具有竞争力的推荐性能。

    Traditional recommender systems leverage users' item preference history to recommend novel content that users may like. However, modern dialog interfaces that allow users to express language-based preferences offer a fundamentally different modality for preference input. Inspired by recent successes of prompting paradigms for large language models (LLMs), we study their use for making recommendations from both item-based and language-based preferences in comparison to state-of-the-art item-based collaborative filtering (CF) methods. To support this investigation, we collect a new dataset consisting of both item-based and language-based preferences elicited from users along with their ratings on a variety of (biased) recommended items and (unbiased) random items. Among numerous experimental results, we find that LLMs provide competitive recommendation performance for pure language-based preferences (no item preferences) in the near cold-start case in comparison to item-based CF methods,
    
[^3]: 一种概率位置偏差模型用于短视频推荐中

    A Probabilistic Position Bias Model for Short-Video Recommendation Feeds. (arXiv:2307.14059v1 [cs.IR])

    [http://arxiv.org/abs/2307.14059](http://arxiv.org/abs/2307.14059)

    本论文提出了一种概率位置偏差模型，用于解决社交媒体平台上短视频推荐中的问题。这种模型考虑了用户在浏览视频时的行为特点，能够更准确地估计每个视频的曝光概率。

    

    当代基于网络的平台向用户展示排名列表的推荐内容，以尝试最大化用户满意度或业务指标。这些系统的目标通常是最大化被认为是“最大化奖励”的项目的曝光概率。这个通用框架包括流媒体应用、电子商务或职位推荐，甚至网络搜索。在每种用例中，可以使用位置偏差或用户模型来估计曝光概率，这些模型可以根据用户与展示的排名的互动方式进行特定调整。这些不同的问题设置中的一个统一因素是，通常只有一个或几个项目会在用户离开排序列表之前被参与（点击、流媒体等）。社交媒体平台上的短视频推送在几个方面与这种一般框架不同，最重要的是用户在点赞一篇帖子后通常不会离开推送。实际上，看似无限的推送引导用户滚动浏览视频。

    Modern web-based platforms show ranked lists of recommendations to users, attempting to maximise user satisfaction or business metrics. Typically, the goal of such systems boils down to maximising the exposure probability for items that are deemed "reward-maximising" according to a metric of interest. This general framing comprises streaming applications, as well as e-commerce or job recommendations, and even web search. Position bias or user models can be used to estimate exposure probabilities for each use-case, specifically tailored to how users interact with the presented rankings. A unifying factor in these diverse problem settings is that typically only one or several items will be engaged with (clicked, streamed,...) before a user leaves the ranked list. Short-video feeds on social media platforms diverge from this general framing in several ways, most notably that users do not tend to leave the feed after e.g. liking a post. Indeed, seemingly infinite feeds invite users to scro
    
[^4]: 多视图超图对比策略学习在会话推荐中的应用

    Multi-view Hypergraph Contrastive Policy Learning for Conversational Recommendation. (arXiv:2307.14024v1 [cs.IR])

    [http://arxiv.org/abs/2307.14024](http://arxiv.org/abs/2307.14024)

    这篇论文提出了一种多视图超图对比策略学习的方法，用于会话推荐系统。该方法综合考虑了用户的喜欢、社交和不喜欢三个视图，并通过对比学习用户偏好，从而提高会话推荐的准确性。

    

    会话推荐系统旨在通过交互式获取用户偏好，相应地向用户推荐物品。准确学习动态用户偏好对于会话推荐系统至关重要。先前的研究通过交互对话和物品知识中的两两关系来学习用户偏好，但往往忽视了在会话推荐系统中关系的复合性。具体而言，用户喜欢/不喜欢满足某些属性的物品（喜欢/不喜欢视图）。此外，社交影响是影响用户对物品的偏好的另一个重要因素（社交视图），但是先前的会话推荐系统往往忽略了这一因素。这三个视图的用户偏好本质上是不同的，但整体上是相关的。相同视图的用户偏好应该比不同视图的用户偏好更相似。喜欢视图的用户偏好应该与社交视图相似，但与不喜欢视图不同。

    Conversational recommendation systems (CRS) aim to interactively acquire user preferences and accordingly recommend items to users. Accurately learning the dynamic user preferences is of crucial importance for CRS. Previous works learn the user preferences with pairwise relations from the interactive conversation and item knowledge, while largely ignoring the fact that factors for a relationship in CRS are multiplex. Specifically, the user likes/dislikes the items that satisfy some attributes (Like/Dislike view). Moreover social influence is another important factor that affects user preference towards the item (Social view), while is largely ignored by previous works in CRS. The user preferences from these three views are inherently different but also correlated as a whole. The user preferences from the same views should be more similar than that from different views. The user preferences from Like View should be similar to Social View while different from Dislike View. To this end, w
    
[^5]: 基于插值数据增强的领域解缠方法用于双目标跨领域推荐

    Domain Disentanglement with Interpolative Data Augmentation for Dual-Target Cross-Domain Recommendation. (arXiv:2307.13910v1 [cs.IR])

    [http://arxiv.org/abs/2307.13910](http://arxiv.org/abs/2307.13910)

    本文提出了一种基于插值数据增强的解缠方法（DIDA-CDR）用于双目标跨领域推荐，旨在解决生成相关且多样化的增强用户表示以及有效将领域无关信息与领域特定信息分离的挑战。

    

    传统的单目标跨领域推荐旨在通过从包含相对丰富信息的源领域转移知识来改进在稀疏目标领域中的推荐性能。相比之下，近年来提出了双目标跨领域推荐来同时提高两个领域的推荐性能。然而，在双目标跨领域推荐中存在两个挑战：1）如何生成相关且多样化的增强用户表示；2）如何有效地将领域无关信息与领域特定信息分离开来，以捕捉全面的用户偏好。为了解决上述两个挑战，在本文中我们提出了一种基于解缠和插值数据增强的双目标跨领域推荐框架，称为DIDA-CDR。

    The conventional single-target Cross-Domain Recommendation (CDR) aims to improve the recommendation performance on a sparser target domain by transferring the knowledge from a source domain that contains relatively richer information. By contrast, in recent years, dual-target CDR has been proposed to improve the recommendation performance on both domains simultaneously. However, to this end, there are two challenges in dual-target CDR: (1) how to generate both relevant and diverse augmented user representations, and (2) how to effectively decouple domain-independent information from domain-specific information, in addition to domain-shared information, to capture comprehensive user preferences. To address the above two challenges, we propose a Disentanglement-based framework with Interpolative Data Augmentation for dual-target Cross-Domain Recommendation, called DIDA-CDR. In DIDA-CDR, we first propose an interpolative data augmentation approach to generating both relevant and diverse a
    
[^6]: ClusterSeq: 用基于聚类的元学习增强顺序推荐系统

    ClusterSeq: Enhancing Sequential Recommender Systems with Clustering based Meta-Learning. (arXiv:2307.13766v1 [cs.IR])

    [http://arxiv.org/abs/2307.13766](http://arxiv.org/abs/2307.13766)

    ClusterSeq是一种基于聚类的元学习顺序推荐系统，通过利用用户序列的动态信息提高了物品预测的准确性，并保留了次要用户的偏好，并利用了同一聚类中用户的集体知识。

    

    在实际场景中，顺序推荐系统的有效性受到了用户冷启动问题的限制，这是由于有限的交互使得无法准确确定用户的偏好。以前的研究试图通过将元学习与用户和物品侧信息相结合来解决这个问题。然而，这些方法在建模用户偏好动态方面面临着固有的挑战，尤其是对于展现出与更常见或“主要用户”不同偏好的“次要用户”。为了克服这些局限性，我们提出了一种新颖的方法，称为ClusterSeq，一种基于聚类的元学习顺序推荐系统。ClusterSeq利用用户序列中的动态信息来提高物品预测的准确性，即使没有侧信息。该模型保留了次要用户的偏好，而不会被主要用户所掩盖，并利用了同一聚类中用户的集体知识。

    In practical scenarios, the effectiveness of sequential recommendation systems is hindered by the user cold-start problem, which arises due to limited interactions for accurately determining user preferences. Previous studies have attempted to address this issue by combining meta-learning with user and item-side information. However, these approaches face inherent challenges in modeling user preference dynamics, particularly for "minor users" who exhibit distinct preferences compared to more common or "major users." To overcome these limitations, we present a novel approach called ClusterSeq, a Meta-Learning Clustering-Based Sequential Recommender System. ClusterSeq leverages dynamic information in the user sequence to enhance item prediction accuracy, even in the absence of side information. This model preserves the preferences of minor users without being overshadowed by major users, and it capitalizes on the collective knowledge of users within the same cluster. Extensive experiment
    
[^7]: 接口设计以缓解推荐系统中的评分膨胀问题

    Interface Design to Mitigate Inflation in Recommender Systems. (arXiv:2307.12424v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2307.12424](http://arxiv.org/abs/2307.12424)

    本研究通过分析音乐发现应用中的数据发现，评分膨胀问题来源于异质的用户评分行为和个性化推荐的动态，通过修改评分界面可以显著改善这一问题。

    

    推荐系统依赖用户提供的数据来学习物品质量并提供个性化推荐。在将评分聚合为物品质量时，预设了评分是物品质量的强有力指标。本研究通过从音乐发现应用中收集的数据来测试这个假设。我们的研究集中在两个导致评分膨胀的因素上：异质的用户评分行为和个性化推荐的动态。我们显示出用户评分行为在用户之间存在较大差异，导致物品质量估计更多反映了评分用户而不是物品本身的质量。此外，通过个性化推荐更有可能展示的物品可能会经历大幅增加的曝光，并有潜在的偏好。为了缓解这些影响，我们分析了一个随机对照试验中评分界面修改的结果。测试结果显示了显著的改进。

    Recommendation systems rely on user-provided data to learn about item quality and provide personalized recommendations. An implicit assumption when aggregating ratings into item quality is that ratings are strong indicators of item quality. In this work, we test this assumption using data collected from a music discovery application. Our study focuses on two factors that cause rating inflation: heterogeneous user rating behavior and the dynamics of personalized recommendations. We show that user rating behavior substantially varies by user, leading to item quality estimates that reflect the users who rated an item more than the item quality itself. Additionally, items that are more likely to be shown via personalized recommendations can experience a substantial increase in their exposure and potential bias toward them. To mitigate these effects, we analyze the results of a randomized controlled trial in which the rating interface was modified. The test resulted in a substantial improve
    
[^8]: Bert4XMR: 使用Transformer中的双向编码器表示进行跨市场推荐

    Bert4XMR: Cross-Market Recommendation with Bidirectional Encoder Representations from Transformer. (arXiv:2305.15145v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.15145](http://arxiv.org/abs/2305.15145)

    提出了一种名为Bert4XMR的新型跨市场推荐模型，能够建模不同市场的物品共现性，并减轻负迁移问题。

    

    在实际的跨国电商公司，如亚马逊和eBay，服务于多个国家和地区。一些市场的数据稀缺，而其他市场的数据丰富。近年来，跨市场推荐（XMR）已经提出，通过利用来自数据丰富市场的辅助信息来增强数据稀缺市场。先前的XMR算法采用了共享底层或整合市场间相似性等技术来优化XMR的性能。然而，现有方法存在两个关键限制：（1）忽略了数据丰富市场提供的物品共现性；（2）没有充分解决不同市场之间的差异导致的负迁移问题。为了解决这些限制，我们提出了一种新颖的基于会话的模型，称为Bert4XMR，它能够对不同市场的物品共现性进行建模和缓解负迁移。具体而言，我们采用预训练和微调的范式来促进模型的学习和性能优化。

    Real-world multinational e-commerce companies, such as Amazon and eBay, serve in multiple countries and regions. Some markets are data-scarce, while others are data-rich. In recent years, cross-market recommendation (XMR) has been proposed to bolster data-scarce markets by leveraging auxiliary information from data-rich markets. Previous XMR algorithms have employed techniques such as sharing bottom or incorporating inter-market similarity to optimize the performance of XMR. However, the existing approaches suffer from two crucial limitations: (1) They ignore the co-occurrences of items provided by data-rich markets. (2) They do not adequately tackle the issue of negative transfer stemming from disparities across diverse markets. In order to address these limitations, we propose a novel session-based model called Bert4XMR, which is able to model item co-occurrences across markets and mitigate negative transfer. Specifically, we employ the pre-training and fine-tuning paradigm to facili
    
[^9]: 基于超图强化学习的分层对话推荐中的策略学习

    Towards Hierarchical Policy Learning for Conversational Recommendation with Hypergraph-based Reinforcement Learning. (arXiv:2305.02575v1 [cs.IR])

    [http://arxiv.org/abs/2305.02575](http://arxiv.org/abs/2305.02575)

    本文提出了一种新颖的基于超图强化学习的分层对话推荐模型，其中导演通过超图算法进行选择，帮助演员减少行动空间和指导对话朝着最具信息性的属性方向进行，并根据用户在对话中的偏好选择物品。

    

    对话推荐系统旨在通过对话及时主动地获取用户的偏好，并推荐相应的物品。然而，现有的方法往往使用统一的决策模块或启发式规则，而忽略了不同决策过程的角色差异和相互作用。为此，本文提出了一种新颖的基于超图强化学习的分层对话推荐模型，其中导演通过超图算法进行选择，帮助演员减少行动空间，指导对话朝着最具信息性的属性方向进行，并根据用户在对话中的偏好选择物品。实验结果表明，与现有方法相比，本文方法在真实数据集上表现出更高的效果和优越性。

    Conversational recommendation systems (CRS) aim to timely and proactively acquire user dynamic preferred attributes through conversations for item recommendation. In each turn of CRS, there naturally have two decision-making processes with different roles that influence each other: 1) director, which is to select the follow-up option (i.e., ask or recommend) that is more effective for reducing the action space and acquiring user preferences; and 2) actor, which is to accordingly choose primitive actions (i.e., asked attribute or recommended item) that satisfy user preferences and give feedback to estimate the effectiveness of the director's option. However, existing methods heavily rely on a unified decision-making module or heuristic rules, while neglecting to distinguish the roles of different decision procedures, as well as the mutual influences between them. To address this, we propose a novel Director-Actor Hierarchical Conversational Recommender (DAHCR), where the director select
    
[^10]: 推荐系统中的公平性：基础、方法和应用

    Fairness in Recommendation: Foundations, Methods and Applications. (arXiv:2205.13619v5 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2205.13619](http://arxiv.org/abs/2205.13619)

    这篇论文对推荐系统中的公平性问题进行了系统调查，针对推荐过程中可能出现的数据或算法偏见，提供了一些方法和应用来提升推荐中的公平性。

    

    作为机器学习最普遍的应用之一，推荐系统在辅助人类决策中起着重要作用。用户的满意度和平台的利益与生成的推荐结果的质量密切相关。然而，作为一个高度数据驱动的系统，推荐系统可能受到数据或算法偏见的影响，从而产生不公平的结果，这可能削弱系统的可信赖性。因此，在推荐设置中解决潜在的不公平问题至关重要。最近，对推荐系统的公平性考虑引起了越来越多的关注，涉及提升推荐中的公平性的方法越来越多。然而，这些研究相对零散且缺乏系统化整理，因此对于新研究人员来说难以深入领域。这促使我们对推荐中现有公平性作品进行系统调查。

    As one of the most pervasive applications of machine learning, recommender systems are playing an important role on assisting human decision making. The satisfaction of users and the interests of platforms are closely related to the quality of the generated recommendation results. However, as a highly data-driven system, recommender system could be affected by data or algorithmic bias and thus generate unfair results, which could weaken the reliance of the systems. As a result, it is crucial to address the potential unfairness problems in recommendation settings. Recently, there has been growing attention on fairness considerations in recommender systems with more and more literature on approaches to promote fairness in recommendation. However, the studies are rather fragmented and lack a systematic organization, thus making it difficult to penetrate for new researchers to the domain. This motivates us to provide a systematic survey of existing works on fairness in recommendation. This
    
[^11]: 基于图神经网络的顺序推荐

    Sequential Recommendation with Graph Neural Networks. (arXiv:2106.14226v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2106.14226](http://arxiv.org/abs/2106.14226)

    提出了SURGE（SeqUential Recommendation with Graph neural networks）模型，通过图神经网络将用户历史行为中的不同偏好聚类成紧密的兴趣图，以更好地预测顺序推荐中用户的下一次互动。

    

    顺序推荐旨在利用用户的历史行为预测他们的下一次互动。现有的工作在顺序推荐中面临两个主要挑战。首先，用户的行为在他们丰富的历史序列中常常是隐式和嘈杂的偏好信号，无法充分反映用户的实际偏好。另外，用户的动态偏好往往会随时间迅速变化，因此很难捕捉到他们历史序列中的用户模式。在这项工作中，我们提出了一种名为SURGE（SeqUential Recommendation with Graph neural networks）的图神经网络模型来应对这两个问题。具体而言，SURGE通过基于度量学习，将长期用户行为中不同类型的偏好重新构造为紧密的物品-物品兴趣图中的聚类，从而帮助明确区分用户的核心兴趣。聚类在兴趣图中形成了密集的集群。

    Sequential recommendation aims to leverage users' historical behaviors to predict their next interaction. Existing works have not yet addressed two main challenges in sequential recommendation. First, user behaviors in their rich historical sequences are often implicit and noisy preference signals, they cannot sufficiently reflect users' actual preferences. In addition, users' dynamic preferences often change rapidly over time, and hence it is difficult to capture user patterns in their historical sequences. In this work, we propose a graph neural network model called SURGE (short for SeqUential Recommendation with Graph neural nEtworks) to address these two issues. Specifically, SURGE integrates different types of preferences in long-term user behaviors into clusters in the graph by re-constructing loose item sequences into tight item-item interest graphs based on metric learning. This helps explicitly distinguish users' core interests, by forming dense clusters in the interest graph.
    

