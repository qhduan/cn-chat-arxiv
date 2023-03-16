# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dually Enhanced Propensity Score Estimation in Sequential Recommendation.](http://arxiv.org/abs/2303.08722) | 本文提出了一个名为DEPS的新算法，在顺序推荐系统中通过从用户和项目两个视角同时优化IPS估计器来减少偏差，提高推荐精度。 |
| [^2] | [Automated Query Generation for Evidence Collection from Web Search Engines.](http://arxiv.org/abs/2303.08652) | 本研究考虑了自动查询生成，即根据事实陈述自动生成搜索查询。研究引入了一个包含390个事实陈述和相关搜索查询和结果的中等规模证据收集数据集。 |
| [^3] | [Graph-less Collaborative Filtering.](http://arxiv.org/abs/2303.08537) | 本文提出了一个无图的协同过滤模型SimRec，通过知识蒸馏和对比学习，实现了教师GNN模型与轻量级学生网络之间的自适应知识转移，有效地解决了现有基于GNN的CF模型可能出现的过度平滑和噪声效应的问题，在多个数据集上都超过了目前最先进的方法。 |
| [^4] | [A Cross-institutional Evaluation on Breast Cancer Phenotyping NLP Algorithms on Electronic Health Records.](http://arxiv.org/abs/2303.08448) | 本研究通过对乳腺癌表型提取任务的评估，展示了基于BERT的临床NLP模型在不同临床环境中具有良好的泛化能力，并强调了使用转移学习开发广义临床NLP模型的潜力。 |
| [^5] | [Robust Preference-Guided Denoising for Graph based Social Recommendation.](http://arxiv.org/abs/2303.08346) | 本文提出了一种基于偏好引导的图形社交推荐去噪方法，以提高推荐的有效性和效率。 |
| [^6] | [The Elements of Visual Art Recommendation: Learning Latent Semantic Representations of Paintings.](http://arxiv.org/abs/2303.08182) | 本文研究了如何高效地捕捉视觉艺术的元素，提出了结合文本和视觉特征学习技术的推荐系统，用于个性化艺术品推荐，结果显示两者的结合可以捕捉最合适的隐藏语义关系。 |
| [^7] | [Graph Neural Network Surrogates of Fair Graph Filtering.](http://arxiv.org/abs/2303.08157) | 通过引入过滤器感知的通用近似框架，该方法定义了合适的图神经网络在运行时训练以满足统计平等约束，同时最小程度扰动原始后验情况下实现此目标。 |
| [^8] | [Authorship Conflicts in Academia: an International Cross-Discipline Survey.](http://arxiv.org/abs/2303.00386) | 这项国际跨学科调查发现，作者署名的冲突问题非常普遍，并且往往在学者的职业生涯早期就开始出现。 |

# 详细

[^1]: 顺序推荐中双重增强倾向分数算法的研究

    Dually Enhanced Propensity Score Estimation in Sequential Recommendation. (arXiv:2303.08722v1 [cs.IR])

    [http://arxiv.org/abs/2303.08722](http://arxiv.org/abs/2303.08722)

    本文提出了一个名为DEPS的新算法，在顺序推荐系统中通过从用户和项目两个视角同时优化IPS估计器来减少偏差，提高推荐精度。

    

    顺序推荐系统是基于大量隐式用户反馈数据进行建模的，当用户系统地低估/高估某些项目时，会受到偏见的影响。提出了基于倒数倾向分数（IPS）的无偏学习方法，以解决这个问题。在这些方法中，倾向分数估计通常仅限于项目视图，即将反馈数据视为与用户互动的项目序列。然而，反馈数据也可以从用户的角度来看待，作为与项目互动的用户序列。此外，两个视角可以共同增强倾向分数的估计。受此观察的启发，我们提出了从用户和项目视角估计倾向分数的DEPS算法。具体而言，DEPS从用户视角和项目视角优化两个IPS估计器，并通过最小化其加权平均值的方差来减少偏差。在两个公共数据集上的实验结果表明DEPS在减少偏差和提高推荐精度方面具有有效性。

    Sequential recommender systems train their models based on a large amount of implicit user feedback data and may be subject to biases when users are systematically under/over-exposed to certain items. Unbiased learning based on inverse propensity scores (IPS), which estimate the probability of observing a user-item pair given the historical information, has been proposed to address the issue. In these methods, propensity score estimation is usually limited to the view of item, that is, treating the feedback data as sequences of items that interacted with the users. However, the feedback data can also be treated from the view of user, as the sequences of users that interact with the items. Moreover, the two views can jointly enhance the propensity score estimation. Inspired by the observation, we propose to estimate the propensity scores from the views of user and item, called Dually Enhanced Propensity Score Estimation (DEPS). Specifically, given a target user-item pair and the corresp
    
[^2]: 自动查询生成用于从网络搜索引擎中收集证据。

    Automated Query Generation for Evidence Collection from Web Search Engines. (arXiv:2303.08652v1 [cs.CL])

    [http://arxiv.org/abs/2303.08652](http://arxiv.org/abs/2303.08652)

    本研究考虑了自动查询生成，即根据事实陈述自动生成搜索查询。研究引入了一个包含390个事实陈述和相关搜索查询和结果的中等规模证据收集数据集。

    

    人们普遍认为，可以通过在互联网上搜索信息来验证所谓的事实。这个过程需要事实核查员根据事实制定搜索查询并向搜索引擎提交，然后需要在搜索结果中识别相关和可信的段落，然后才能做出决策。在许多新闻和媒体组织中，这个过程由副编辑每天完成。在这里，我们问一个问题，那就是是否可能自动化第一步，即查询生成。我们是否能够根据类似于人类专家制定的事实陈述自动生成搜索查询？我们考虑相似性，无论是从文本相似性的角度还是从搜索引擎返回相关文档的角度。首先，我们介绍一个中等规模的证据收集数据集，其中包括390个事实陈述以及相关的人工生成的搜索查询和搜索结果。

    It is widely accepted that so-called facts can be checked by searching for information on the Internet. This process requires a fact-checker to formulate a search query based on the fact and to present it to a search engine. Then, relevant and believable passages need to be identified in the search results before a decision is made. This process is carried out by sub-editors at many news and media organisations on a daily basis. Here, we ask the question as to whether it is possible to automate the first step, that of query generation. Can we automatically formulate search queries based on factual statements which are similar to those formulated by human experts? Here, we consider similarity both in terms of textual similarity and with respect to relevant documents being returned by a search engine. First, we introduce a moderate-sized evidence collection dataset which includes 390 factual statements together with associated human-generated search queries and search results. Then, we i
    
[^3]: 无图协同过滤

    Graph-less Collaborative Filtering. (arXiv:2303.08537v1 [cs.IR])

    [http://arxiv.org/abs/2303.08537](http://arxiv.org/abs/2303.08537)

    本文提出了一个无图的协同过滤模型SimRec，通过知识蒸馏和对比学习，实现了教师GNN模型与轻量级学生网络之间的自适应知识转移，有效地解决了现有基于GNN的CF模型可能出现的过度平滑和噪声效应的问题，在多个数据集上都超过了目前最先进的方法。

    

    图神经网络已经在协同过滤任务中展示出了其在图结构用户-物品交互数据上表示学习的能力。然而，由于低通Laplacian平滑算子的过度平滑和噪声效应，现有的基于GNN的CF模型可能会生成难以区分且不准确的用户（物品）表示。为解决这些限制，本文提出了一个简单而有效的协同过滤模型（SimRec），将知识蒸馏和对比学习的能力融合在一起，实现了教师GNN模型与轻量级学生网络之间的自适应知识转移，在不需要构建图的情况下更好地发现用户和物品之间的相互关系。

    Graph neural networks (GNNs) have shown the power in representation learning over graph-structured user-item interaction data for collaborative filtering (CF) task. However, with their inherently recursive message propagation among neighboring nodes, existing GNN-based CF models may generate indistinguishable and inaccurate user (item) representations due to the over-smoothing and noise effect with low-pass Laplacian smoothing operators. In addition, the recursive information propagation with the stacked aggregators in the entire graph structures may result in poor scalability in practical applications. Motivated by these limitations, we propose a simple and effective collaborative filtering model (SimRec) that marries the power of knowledge distillation and contrastive learning. In SimRec, adaptive transferring knowledge is enabled between the teacher GNN model and a lightweight student network, to not only preserve the global collaborative signals, but also address the over-smoothing
    
[^4]: 通过对电子病历的乳腺癌表型NLP算法进行跨机构评估

    A Cross-institutional Evaluation on Breast Cancer Phenotyping NLP Algorithms on Electronic Health Records. (arXiv:2303.08448v1 [cs.CL])

    [http://arxiv.org/abs/2303.08448](http://arxiv.org/abs/2303.08448)

    本研究通过对乳腺癌表型提取任务的评估，展示了基于BERT的临床NLP模型在不同临床环境中具有良好的泛化能力，并强调了使用转移学习开发广义临床NLP模型的潜力。

    

    目标：在模型开发过程中，通常忽略临床大型语言模型的泛化能力。本研究通过乳腺癌表型提取任务，评估了基于BERT的临床NLP模型在不同临床环境下的泛化能力。方法：从明尼苏达大学和梅奥诊所的电子病历中收集了两种乳腺癌患者的临床语料库，并按照同一指南进行注释。我们开发了三种类型的NLP模型（条件随机场、双向长短期记忆和CancerBERT），从临床文本中提取癌症表型。使用不同的学习策略（模型转移与本地训练）对模型在不同测试集上进行泛化能力评估。评估实体覆盖率与模型性能的相关性得分。结果：在UMN和MC手动注释了200和161份临床文档。CancerBERT模型达到了最高的F1分数（0.896）和实体覆盖率（98.8%），优于其他模型。模型转移方法在两个机构中产生了类似于本地训练模型的结果，表明跨机构存在潜在的泛化性。结论：本研究展示了在不同临床环境中评估NLP模型的重要性，并强调了使用转移学习开发广义临床NLP模型的潜力。

    Objective: The generalizability of clinical large language models is usually ignored during the model development process. This study evaluated the generalizability of BERT-based clinical NLP models across different clinical settings through a breast cancer phenotype extraction task.  Materials and Methods: Two clinical corpora of breast cancer patients were collected from the electronic health records from the University of Minnesota and the Mayo Clinic, and annotated following the same guideline. We developed three types of NLP models (i.e., conditional random field, bi-directional long short-term memory and CancerBERT) to extract cancer phenotypes from clinical texts. The models were evaluated for their generalizability on different test sets with different learning strategies (model transfer vs. locally trained). The entity coverage score was assessed with their association with the model performances.  Results: We manually annotated 200 and 161 clinical documents at UMN and MC, re
    
[^5]: 基于偏好引导去噪的图形社交推荐鲁棒性研究

    Robust Preference-Guided Denoising for Graph based Social Recommendation. (arXiv:2303.08346v1 [cs.IR])

    [http://arxiv.org/abs/2303.08346](http://arxiv.org/abs/2303.08346)

    本文提出了一种基于偏好引导的图形社交推荐去噪方法，以提高推荐的有效性和效率。

    

    基于图神经网络(GNN)的社交推荐模型通过利用GNN在社交关系中的偏好相似性来提高用户偏好的预测准确性。然而，关于推荐的有效性和效率，很大一部分社交关系可能是冗余的甚至是嘈杂的，例如，在某个领域中，朋友之间不共享偏好是很正常的。现有模型没有完全解决这个关系冗余和噪音的问题，因为它们直接表征整个社交网络上的社交影响。在本文中，我们提出通过仅保留信息丰富的社交关系来改进基于图的社交推荐，以确保一个高效和有效的影响扩散，即图形去噪。我们设计的去噪方法是基于偏好引导的，以建模社交关系的信心，并通过提供一个去噪但更具信息量的社交图为推荐用户偏好学习提供帮助。

    Graph Neural Network(GNN) based social recommendation models improve the prediction accuracy of user preference by leveraging GNN in exploiting preference similarity contained in social relations. However, in terms of both effectiveness and efficiency of recommendation, a large portion of social relations can be redundant or even noisy, e.g., it is quite normal that friends share no preference in a certain domain. Existing models do not fully solve this problem of relation redundancy and noise, as they directly characterize social influence over the full social network. In this paper, we instead propose to improve graph based social recommendation by only retaining the informative social relations to ensure an efficient and effective influence diffusion, i.e., graph denoising. Our designed denoising method is preference-guided to model social relation confidence and benefits user preference learning in return by providing a denoised but more informative social graph for recommendation 
    
[^6]: 视觉艺术推荐的要素：学习画作的潜在语义表征

    The Elements of Visual Art Recommendation: Learning Latent Semantic Representations of Paintings. (arXiv:2303.08182v1 [cs.IR])

    [http://arxiv.org/abs/2303.08182](http://arxiv.org/abs/2303.08182)

    本文研究了如何高效地捕捉视觉艺术的元素，提出了结合文本和视觉特征学习技术的推荐系统，用于个性化艺术品推荐，结果显示两者的结合可以捕捉最合适的隐藏语义关系。

    

    艺术品推荐具有挑战性，因为它需要理解用户如何与高度主观的内容互动，艺术品中嵌入的概念的复杂性，以及它们可能引起用户的情感和认知反应。本文重点研究如何高效地捕捉视觉艺术的元素（即潜在语义关系），以进行个性化推荐。我们提出并研究了基于文本和视觉特征学习技术以及它们的组合的推荐系统。我们对推荐质量进行了小规模和大规模的用户中心评估。我们的结果表明，文本特征比视觉特征表现更好，而两者的结合可以捕捉艺术品推荐最合适的隐藏语义关系。最终，本文有助于理解如何提供适合用户兴趣和感知的内容。

    Artwork recommendation is challenging because it requires understanding how users interact with highly subjective content, the complexity of the concepts embedded within the artwork, and the emotional and cognitive reflections they may trigger in users. In this paper, we focus on efficiently capturing the elements (i.e., latent semantic relationships) of visual art for personalized recommendation. We propose and study recommender systems based on textual and visual feature learning techniques, as well as their combinations. We then perform a small-scale and a large-scale user-centric evaluation of the quality of the recommendations. Our results indicate that textual features compare favourably with visual ones, whereas a fusion of both captures the most suitable hidden semantic relationships for artwork recommendation. Ultimately, this paper contributes to our understanding of how to deliver content that suitably matches the user's interests and how they are perceived.
    
[^7]: 基于图神经网络的公平图过滤替代方法

    Graph Neural Network Surrogates of Fair Graph Filtering. (arXiv:2303.08157v1 [cs.LG])

    [http://arxiv.org/abs/2303.08157](http://arxiv.org/abs/2303.08157)

    通过引入过滤器感知的通用近似框架，该方法定义了合适的图神经网络在运行时训练以满足统计平等约束，同时最小程度扰动原始后验情况下实现此目标。

    

    通过边传播将先前的节点值转换为后来的分数的图滤波器通常支持影响人类的图挖掘任务，例如推荐和排名。因此，重要的是在满足节点组之间的统计平等约束方面使它们公平（例如，按其代表性将分数质量在性别之间均衡分配）。为了在最小程度地扰动原始后验情况下实现此目标，我们引入了一个过滤器感知的通用近似框架，用于后验目标。这定义了适当的图神经网络，其在运行时训练，类似于过滤器，但也在本地优化包括公平感知在内的大类目标。在一组8个过滤器和5个图形的实验中，我们的方法在满足统计平等约束方面表现得不亚于替代品，同时保留基于分数的社区成员推荐的AUC并在传播先前节拍时创建最小实用损失。

    Graph filters that transform prior node values to posterior scores via edge propagation often support graph mining tasks affecting humans, such as recommendation and ranking. Thus, it is important to make them fair in terms of satisfying statistical parity constraints between groups of nodes (e.g., distribute score mass between genders proportionally to their representation). To achieve this while minimally perturbing the original posteriors, we introduce a filter-aware universal approximation framework for posterior objectives. This defines appropriate graph neural networks trained at runtime to be similar to filters but also locally optimize a large class of objectives, including fairness-aware ones. Experiments on a collection of 8 filters and 5 graphs show that our approach performs equally well or better than alternatives in meeting parity constraints while preserving the AUC of score-based community member recommendation and creating minimal utility loss in prior diffusion.
    
[^8]: 学术界的作者冲突：一项国际跨学科调查

    Authorship Conflicts in Academia: an International Cross-Discipline Survey. (arXiv:2303.00386v2 [cs.DL] UPDATED)

    [http://arxiv.org/abs/2303.00386](http://arxiv.org/abs/2303.00386)

    这项国际跨学科调查发现，作者署名的冲突问题非常普遍，并且往往在学者的职业生涯早期就开始出现。

    

    学者间的合作已成为当代科学的重要特征，因此在出版物中列出的作者数量不断上升。然而，确定应该包括哪些作者以及他们的顺序涉及多种困难，往往会导致冲突。尽管关于学术冲突的大量文献，但它在主要社会人口学特征以及学术界经历的不同类型的交互方面的分布仍不清楚。为了填补这一差距，我们进行了一项国际跨学科调查，受到了来自41个研究领域和93个国家的752名学者的统计学代表整体学术劳动力的回答。我们的发现令人担忧，表明作者署名冲突在一个人的学术生涯早期即产生，甚至在硕士和博士水平上就普遍出现。

    Collaboration among scholars has emerged as a significant characteristic of contemporary science. As a result, the number of authors listed in publications continues to rise steadily. Unfortunately, determining the authors to be included in the byline and their respective order entails multiple difficulties which often lead to conflicts. Despite the large volume of literature about conflicts in academia, it remains unclear how exactly it is distributed over the main socio-demographic properties, as well as the different types of interactions academics experience. To address this gap, we conducted an international and cross-disciplinary survey answered by 752 academics from 41 fields of research and 93 countries that statistically well-represent the overall academic workforce. Our findings are concerning and suggest that authorship credit conflicts arise very early in one's academic career, even at the level of Master and Ph.D., and become increasingly common over time.
    

