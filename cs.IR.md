# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MAP: A Model-agnostic Pretraining Framework for Click-through Rate Prediction.](http://arxiv.org/abs/2308.01737) | 提出了一个模型无关的预训练框架，用于点击率预测，可以更好地利用多字段分类数据和大量用户点击日志，学习更广义和有效的特征和实例表示。 |
| [^2] | [Evaluating ChatGPT text-mining of clinical records for obesity monitoring.](http://arxiv.org/abs/2308.01666) | 该研究评估了ChatGPT对临床记录中肥胖监测的文本挖掘能力，与之前的正则表达式相比，ChatGPT的召回率更高，但精确度略低。大型语言模型为兽医临床叙述提供了一个有潜力的工具。 |
| [^3] | [Fast Slate Policy Optimization: Going Beyond Plackett-Luce.](http://arxiv.org/abs/2308.01566) | 本文介绍了一种快速Slate策略优化方法，通过提出一种新的策略类，可以在大规模决策系统中有效地优化任意奖励函数，结果表明该方法在百万级别动作空间问题上具有很好的效果。 |
| [^4] | [Density Weighting for Multi-Interest Personalized Recommendation.](http://arxiv.org/abs/2308.01563) | 本文研究了多兴趣个性化推荐中数据不平衡带来的问题。通过使用合成数据集，我们展示了多用户表示（MUR）对于数据不平衡的敏感性，并提出了一种密度加权的方法来改进对于长尾物品的推荐效果。 |
| [^5] | [Fairness in Recommendation: Foundations, Methods and Applications.](http://arxiv.org/abs/2205.13619) | 这篇论文对推荐系统中的公平性问题进行了系统调查，针对推荐过程中可能出现的数据或算法偏见，提供了一些方法和应用来提升推荐中的公平性。 |

# 详细

[^1]: MAP: 一个模型无关的预训练框架用于点击率预测

    MAP: A Model-agnostic Pretraining Framework for Click-through Rate Prediction. (arXiv:2308.01737v1 [cs.IR])

    [http://arxiv.org/abs/2308.01737](http://arxiv.org/abs/2308.01737)

    提出了一个模型无关的预训练框架，用于点击率预测，可以更好地利用多字段分类数据和大量用户点击日志，学习更广义和有效的特征和实例表示。

    

    随着个性化在线服务的广泛应用，点击率（CTR）预测越来越受到关注和研究。CTR预测的最突出特点是其多字段分类数据格式和庞大而日益增长的数据量。神经模型的大容量有助于在监督学习范式下消化如此大量的数据，但是它们未能充分利用大量数据的潜力，因为1比特的点击信号不足以指导模型学习功能强大的特征和实例表示。自我监督学习范式提供了更有前景的预训练-微调解决方案，以更好地利用大量用户点击日志并学习更广义和有效的表示。然而，对于CTR预测的自我监督学习仍然是一个开放的问题，因为当前在这方面的工作仅仅是初步和基础的。为此，我们提出了一个模型无关的预训练框架。

    With the widespread application of personalized online services, click-through rate (CTR) prediction has received more and more attention and research. The most prominent features of CTR prediction are its multi-field categorical data format, and vast and daily-growing data volume. The large capacity of neural models helps digest such massive amounts of data under the supervised learning paradigm, yet they fail to utilize the substantial data to its full potential, since the 1-bit click signal is not sufficient to guide the model to learn capable representations of features and instances. The self-supervised learning paradigm provides a more promising pretrain-finetune solution to better exploit the large amount of user click logs, and learn more generalized and effective representations. However, self-supervised learning for CTR prediction is still an open question, since current works on this line are only preliminary and rudimentary. To this end, we propose a Model-agnostic pretrain
    
[^2]: 评估ChatGPT对肥胖监测中临床记录的文本挖掘能力

    Evaluating ChatGPT text-mining of clinical records for obesity monitoring. (arXiv:2308.01666v1 [cs.IR])

    [http://arxiv.org/abs/2308.01666](http://arxiv.org/abs/2308.01666)

    该研究评估了ChatGPT对临床记录中肥胖监测的文本挖掘能力，与之前的正则表达式相比，ChatGPT的召回率更高，但精确度略低。大型语言模型为兽医临床叙述提供了一个有潜力的工具。

    

    背景：兽医临床叙述仍然是一个很少被利用的资源，用于应对复杂疾病。在这里，我们比较了一个大型语言模型(ChatGPT)和之前开发的正则表达式(RegexT)在识别兽医叙述中超重体况评分(BCS)方面的能力。方法：使用RegexT或将叙述附加到发送给ChatGPT的提示中来提取4,415个匿名临床叙述中的BCS值，迫使模型返回BCS信息。通过手动审查数据进行比较。结果：RegexT的精确度(100%，95% CI 94.81-100%)高于ChatGPT的精确度(89.3%，95% CI 82.75-93.64%)。然而，ChatGPT的召回率(100%，95% CI 96.18-100%)要远高于RegexT的召回率(72.6%，95% CI 63.92-79.94%)。局限性：需要对提示工程进行微调以改善ChatGPT输出。结论：大型语言模型为创建各种机会提供了可能性，并且虽然复杂，但具有直观的界面用于in

    Background: Veterinary clinical narratives remain a largely untapped resource for addressing complex diseases. Here we compare the ability of a large language model (ChatGPT) and a previously developed regular expression (RegexT) to identify overweight body condition scores (BCS) in veterinary narratives. Methods: BCS values were extracted from 4,415 anonymised clinical narratives using either RegexT or by appending the narrative to a prompt sent to ChatGPT coercing the model to return the BCS information. Data were manually reviewed for comparison. Results: The precision of RegexT was higher (100%, 95% CI 94.81-100%) than the ChatGPT (89.3%; 95% CI82.75-93.64%). However, the recall of ChatGPT (100%. 95% CI 96.18-100%) was considerably higher than that of RegexT (72.6%, 95% CI 63.92-79.94%). Limitations: Subtle prompt engineering is needed to improve ChatGPT output. Conclusions: Large language models create diverse opportunities and, whilst complex, present an intuitive interface to in
    
[^3]: 快速Slate策略优化：超越Plackett-Luce

    Fast Slate Policy Optimization: Going Beyond Plackett-Luce. (arXiv:2308.01566v1 [cs.LG])

    [http://arxiv.org/abs/2308.01566](http://arxiv.org/abs/2308.01566)

    本文介绍了一种快速Slate策略优化方法，通过提出一种新的策略类，可以在大规模决策系统中有效地优化任意奖励函数，结果表明该方法在百万级别动作空间问题上具有很好的效果。

    

    大规模机器学习系统中一个越来越重要的构建模块是返回Slate，即给定一个查询返回有序的项目列表。该技术的应用包括搜索、信息检索和推荐系统。当行动空间很大时，决策系统会限制在特定结构中以快速完成在线查询。本文解决了这些大规模决策系统在给定任意奖励函数下的优化问题。我们将这个学习问题转化为策略优化框架，并提出了一种新的策略类，它源于决策函数的一种新颖放松。这导致了一个简单而高效的学习算法，可以扩展到大规模的动作空间。我们将我们的方法与常用的Plackett-Luce策略类进行比较，并展示了我们的方法在动作空间大小达到百万级别的问题上的有效性。

    An increasingly important building block of large scale machine learning systems is based on returning slates; an ordered lists of items given a query. Applications of this technology include: search, information retrieval and recommender systems. When the action space is large, decision systems are restricted to a particular structure to complete online queries quickly. This paper addresses the optimization of these large scale decision systems given an arbitrary reward function. We cast this learning problem in a policy optimization framework and propose a new class of policies, born from a novel relaxation of decision functions. This results in a simple, yet efficient learning algorithm that scales to massive action spaces. We compare our method to the commonly adopted Plackett-Luce policy class and demonstrate the effectiveness of our approach on problems with action space sizes in the order of millions.
    
[^4]: 多兴趣个性化推荐中的密度加权方法

    Density Weighting for Multi-Interest Personalized Recommendation. (arXiv:2308.01563v1 [cs.IR])

    [http://arxiv.org/abs/2308.01563](http://arxiv.org/abs/2308.01563)

    本文研究了多兴趣个性化推荐中数据不平衡带来的问题。通过使用合成数据集，我们展示了多用户表示（MUR）对于数据不平衡的敏感性，并提出了一种密度加权的方法来改进对于长尾物品的推荐效果。

    

    在推荐系统中，使用多用户表示（MUR）来建模用户行为而不是单个用户表示（SUR）已经证明可以提高个性化效果。然而，MUR的性能提升可能对物品和/或用户兴趣分布的偏斜性敏感。当数据分布高度偏斜时，学习多个表示的收益减小，因为模型在热门物品/兴趣上占主导地位，导致对于长尾物品的推荐效果差。因此，MUR方法对于数据稀疏性的鲁棒性是实现良好推荐效果的关键。然而，关于MUR和数据不平衡的研究在过去很大程度上是独立进行的。在本文中，我们更深入地研究了从不平衡数据分布中推断出的MUR的缺点。我们提出了几点贡献：（1）使用合成数据集，我们展示了MUR相对于数据不平衡的敏感性，（2）为了改进对于长尾物品的MUR，我们提出了一种密度加权的方法。

    Using multiple user representations (MUR) to model user behavior instead of a single user representation (SUR) has been shown to improve personalization in recommendation systems. However, the performance gains observed with MUR can be sensitive to the skewness in the item and/or user interest distribution. When the data distribution is highly skewed, the gains observed by learning multiple representations diminish since the model dominates on head items/interests, leading to poor performance on tail items. Robustness to data sparsity is therefore essential for MUR-based approaches to achieve good performance for recommendations. Yet, research in MUR and data imbalance have largely been done independently. In this paper, we delve deeper into the shortcomings of MUR inferred from imbalanced data distributions. We make several contributions: (1) Using synthetic datasets, we demonstrate the sensitivity of MUR with respect to data imbalance, (2) To improve MUR for tail items, we propose an
    
[^5]: 推荐系统中的公平性：基础、方法和应用

    Fairness in Recommendation: Foundations, Methods and Applications. (arXiv:2205.13619v5 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2205.13619](http://arxiv.org/abs/2205.13619)

    这篇论文对推荐系统中的公平性问题进行了系统调查，针对推荐过程中可能出现的数据或算法偏见，提供了一些方法和应用来提升推荐中的公平性。

    

    作为机器学习最普遍的应用之一，推荐系统在辅助人类决策中起着重要作用。用户的满意度和平台的利益与生成的推荐结果的质量密切相关。然而，作为一个高度数据驱动的系统，推荐系统可能受到数据或算法偏见的影响，从而产生不公平的结果，这可能削弱系统的可信赖性。因此，在推荐设置中解决潜在的不公平问题至关重要。最近，对推荐系统的公平性考虑引起了越来越多的关注，涉及提升推荐中的公平性的方法越来越多。然而，这些研究相对零散且缺乏系统化整理，因此对于新研究人员来说难以深入领域。这促使我们对推荐中现有公平性作品进行系统调查。

    As one of the most pervasive applications of machine learning, recommender systems are playing an important role on assisting human decision making. The satisfaction of users and the interests of platforms are closely related to the quality of the generated recommendation results. However, as a highly data-driven system, recommender system could be affected by data or algorithmic bias and thus generate unfair results, which could weaken the reliance of the systems. As a result, it is crucial to address the potential unfairness problems in recommendation settings. Recently, there has been growing attention on fairness considerations in recommender systems with more and more literature on approaches to promote fairness in recommendation. However, the studies are rather fragmented and lack a systematic organization, thus making it difficult to penetrate for new researchers to the domain. This motivates us to provide a systematic survey of existing works on fairness in recommendation. This
    

