# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data augmentation for recommender system: A semi-supervised approach using maximum margin matrix factorization.](http://arxiv.org/abs/2306.13050) | 本研究提出了一种基于最大边际矩阵分解的半监督方法来增广和细化协同过滤算法的评级预测。该方法利用自我训练来评估评分的置信度，并通过系统的数据增广策略来提高算法性能。 |
| [^2] | [Efficient Partitioning Method of Large-Scale Public Safety Spatio-Temporal Data based on Information Loss Constraints.](http://arxiv.org/abs/2306.12857) | 本文提出了一种基于信息丢失约束的大规模公共安全时空数据高效划分方法(IFL-LSTP)，可以显著减小数据规模，同时保持模型的准确性，确保分布式存储的负载平衡，同时保持数据划分的时空接近性。 |
| [^3] | [HypeRS: Building a Hypergraph-driven ensemble Recommender System.](http://arxiv.org/abs/2306.12800) | 本文提出了一个新的集成推荐系统，将不同模型的预测结果结合成一个超图排名框架，是第一个使用超图排名建模集成推荐系统的。超图可以建模高阶关系。 |
| [^4] | [On the Robustness of Generative Retrieval Models: An Out-of-Distribution Perspective.](http://arxiv.org/abs/2306.12756) | 本论文研究了生成式检索模型在超出分布（OOD）泛化方面的鲁棒性，定义了从三个方面衡量OOD鲁棒性，并分析了其与密集检索模型的比较。实验结果表明，生成式检索模型的OOD鲁棒性较弱，特别是在面向任务的超出分布场景中更为明显。针对造成鲁棒性较弱的原因，提出了潜在的解决方案。 |
| [^5] | [Recent Developments in Recommender Systems: A Survey.](http://arxiv.org/abs/2306.12680) | 本篇综述全面总结了推荐系统领域的最新进展和趋势，包括推荐系统分类，知识推荐系统，鲁棒性，数据偏见和公平性问题，以及评估度量。该研究还提供了未来研究的新方向。 |
| [^6] | [Resources and Evaluations for Multi-Distribution Dense Information Retrieval.](http://arxiv.org/abs/2306.12601) | 本文提出了一个新问题——多分布信息检索，并通过三个基准数据集展示了简单方法的有效性，可防止已知领域消耗大部分检索预算，平均提高Recall @ 100点3.8+，最高可达8.0个点。 |
| [^7] | [ReLoop2: Building Self-Adaptive Recommendation Models via Responsive Error Compensation Loop.](http://arxiv.org/abs/2306.08808) | 本文提出了ReLoop2，一种利用响应式误差补偿循环构建自适应推荐模型的方法，通过错误记忆模块来补偿模型预测误差，实现快速模型适应。 |
| [^8] | [IGB: Addressing The Gaps In Labeling, Features, Heterogeneity, and Size of Public Graph Datasets for Deep Learning Research.](http://arxiv.org/abs/2302.13522) | IGB是一个研究数据集工具，包含同质和异质性学术图形，规模巨大，并提供工具用于生成不同特性的合成图形，为GNN研究人员提供解决公共图形数据集在标记、特征、异质性和大小方面差距的有价值资源。 |

# 详细

[^1]: 推荐系统的数据增广：一种基于最大边际矩阵分解的半监督方法

    Data augmentation for recommender system: A semi-supervised approach using maximum margin matrix factorization. (arXiv:2306.13050v1 [cs.IR])

    [http://arxiv.org/abs/2306.13050](http://arxiv.org/abs/2306.13050)

    本研究提出了一种基于最大边际矩阵分解的半监督方法来增广和细化协同过滤算法的评级预测。该方法利用自我训练来评估评分的置信度，并通过系统的数据增广策略来提高算法性能。

    

    协同过滤已成为推荐系统开发的常用方法，其中，根据用户的过去喜好和其他用户的可用偏好信息预测其对新物品的评分。尽管CF方法很受欢迎，但其性能通常受观察到的条目的稀疏性的极大限制。本研究探讨最大边际矩阵分解（MMMF）的数据增广和细化方面，该方法是广泛接受的用于评级预测的CF技术，之前尚未进行研究。我们利用CF算法的固有特性来评估单个评分的置信度，并提出了一种基于自我训练的半监督评级增强方法。我们假设任何CF算法的预测低置信度是由于训练数据的某些不足，因此，通过采用系统的数据增广策略，可以提高算法的性能。

    Collaborative filtering (CF) has become a popular method for developing recommender systems (RS) where ratings of a user for new items is predicted based on her past preferences and available preference information of other users. Despite the popularity of CF-based methods, their performance is often greatly limited by the sparsity of observed entries. In this study, we explore the data augmentation and refinement aspects of Maximum Margin Matrix Factorization (MMMF), a widely accepted CF technique for the rating predictions, which have not been investigated before. We exploit the inherent characteristics of CF algorithms to assess the confidence level of individual ratings and propose a semi-supervised approach for rating augmentation based on self-training. We hypothesize that any CF algorithm's predictions with low confidence are due to some deficiency in the training data and hence, the performance of the algorithm can be improved by adopting a systematic data augmentation strategy
    
[^2]: 基于信息丢失约束的大规模公共安全时空数据高效划分方法

    Efficient Partitioning Method of Large-Scale Public Safety Spatio-Temporal Data based on Information Loss Constraints. (arXiv:2306.12857v1 [cs.LG])

    [http://arxiv.org/abs/2306.12857](http://arxiv.org/abs/2306.12857)

    本文提出了一种基于信息丢失约束的大规模公共安全时空数据高效划分方法(IFL-LSTP)，可以显著减小数据规模，同时保持模型的准确性，确保分布式存储的负载平衡，同时保持数据划分的时空接近性。

    

    大规模时空数据的存储、管理和应用在各种实际场景中广泛应用，包括公共安全。然而，由于现实世界数据的独特时空分布特征，大多数现有方法在数据时空接近度和分布式存储负载平衡方面存在限制。因此，本文提出了一种基于信息丢失约束的大规模公共安全时空数据高效划分方法(IFL-LSTP)。该IFL-LSTP模型针对大规模时空点数据，将时空划分模块(STPM)和图划分模块(GPM)相结合。该方法可以显著减小数据规模，同时保持模型的准确性，以提高划分效率。它还可以确保分布式存储的负载平衡，同时保持数据划分的时空接近性。

    The storage, management, and application of massive spatio-temporal data are widely applied in various practical scenarios, including public safety. However, due to the unique spatio-temporal distribution characteristics of re-al-world data, most existing methods have limitations in terms of the spatio-temporal proximity of data and load balancing in distributed storage. There-fore, this paper proposes an efficient partitioning method of large-scale public safety spatio-temporal data based on information loss constraints (IFL-LSTP). The IFL-LSTP model specifically targets large-scale spatio-temporal point da-ta by combining the spatio-temporal partitioning module (STPM) with the graph partitioning module (GPM). This approach can significantly reduce the scale of data while maintaining the model's accuracy, in order to improve the partitioning efficiency. It can also ensure the load balancing of distributed storage while maintaining spatio-temporal proximity of the data partitioning res
    
[^3]: HypeRS：构建基于超图驱动的集成推荐系统

    HypeRS: Building a Hypergraph-driven ensemble Recommender System. (arXiv:2306.12800v1 [cs.IR])

    [http://arxiv.org/abs/2306.12800](http://arxiv.org/abs/2306.12800)

    本文提出了一个新的集成推荐系统，将不同模型的预测结果结合成一个超图排名框架，是第一个使用超图排名建模集成推荐系统的。超图可以建模高阶关系。

    

    推荐系统旨在预测用户对物品的偏好。这篇论文提出一种新的集成推荐系统，将不同模型的预测结果结合成一个统一的超图排名框架，这是第一次使用超图排名建模推荐系统的集成。超图是图的一种推广，可以有效地建模高阶关系。我们通过对不同的推荐系统分配不同的超边权重来区分用户和物品之间的实际和预测连接，并在电影、音乐和新闻领域的四个数据集上进行了实验。

    Recommender systems are designed to predict user preferences over collections of items. These systems process users' previous interactions to decide which items should be ranked higher to satisfy their desires. An ensemble recommender system can achieve great recommendation performance by effectively combining the decisions generated by individual models. In this paper, we propose a novel ensemble recommender system that combines predictions made by different models into a unified hypergraph ranking framework. This is the first time that hypergraph ranking has been employed to model an ensemble of recommender systems. Hypergraphs are generalizations of graphs where multiple vertices can be connected via hyperedges, efficiently modeling high-order relations. We differentiate real and predicted connections between users and items by assigning different hyperedge weights to individual recommender systems. We perform experiments using four datasets from the fields of movie, music and news 
    
[^4]: 关于生成式检索模型的鲁棒性:基于超出分布视角的研究

    On the Robustness of Generative Retrieval Models: An Out-of-Distribution Perspective. (arXiv:2306.12756v1 [cs.IR])

    [http://arxiv.org/abs/2306.12756](http://arxiv.org/abs/2306.12756)

    本论文研究了生成式检索模型在超出分布（OOD）泛化方面的鲁棒性，定义了从三个方面衡量OOD鲁棒性，并分析了其与密集检索模型的比较。实验结果表明，生成式检索模型的OOD鲁棒性较弱，特别是在面向任务的超出分布场景中更为明显。针对造成鲁棒性较弱的原因，提出了潜在的解决方案。

    

    最近，生成式检索在信息检索领域日益受到关注，它通过直接生成标识符来检索文档。迄今为止，人们已经付出了很多努力来开发有效的生成式检索模型。然而，在鲁棒性方面却得到的关注较少。当一个新的检索范式进入到真实世界应用中时，衡量超出分布（OOD）泛化也是至关重要的，即生成式检索模型如何泛化到新的分布中。为了回答这个问题，我们首先从检索问题的三个方面定义OOD鲁棒性：1）查询变化；2）未知的查询类型；3）未知任务。基于这个分类法，我们进行实证研究，分析了几个代表性生成式检索模型与密集检索模型在OOD鲁棒性方面的比较。实证结果表明，生成式检索模型的OOD鲁棒性比密集检索模型弱，特别是在面向任务的OOD场景中更明显。我们进一步研究了造成生成式检索模型鲁棒性较弱的原因，并提出了改善它们OOD泛化性能的潜在解决方法。

    Recently, we have witnessed generative retrieval increasingly gaining attention in the information retrieval (IR) field, which retrieves documents by directly generating their identifiers. So far, much effort has been devoted to developing effective generative retrieval models. There has been less attention paid to the robustness perspective. When a new retrieval paradigm enters into the real-world application, it is also critical to measure the out-of-distribution (OOD) generalization, i.e., how would generative retrieval models generalize to new distributions. To answer this question, firstly, we define OOD robustness from three perspectives in retrieval problems: 1) The query variations; 2) The unforeseen query types; and 3) The unforeseen tasks. Based on this taxonomy, we conduct empirical studies to analyze the OOD robustness of several representative generative retrieval models against dense retrieval models. The empirical results indicate that the OOD robustness of generative re
    
[^5]: 推荐系统的最新发展：综述

    Recent Developments in Recommender Systems: A Survey. (arXiv:2306.12680v1 [cs.IR])

    [http://arxiv.org/abs/2306.12680](http://arxiv.org/abs/2306.12680)

    本篇综述全面总结了推荐系统领域的最新进展和趋势，包括推荐系统分类，知识推荐系统，鲁棒性，数据偏见和公平性问题，以及评估度量。该研究还提供了未来研究的新方向。

    

    这篇技术综述全面总结了推荐系统领域的最新进展。本研究的目的是提供领域内现状的概述，并强调推荐系统发展的最新趋势。该研究首先全面总结了主要推荐系统分类方法，包括个性化和群组推荐系统，然后深入探讨了基于知识的推荐系统类别。此外，该综述分析了推荐系统中的鲁棒性、数据偏见和公平性问题，并总结了评估度量用于评估这些系统的性能。最后，研究提供了有关推荐系统发展的最新趋势的见解，并强调了未来研究的新方向。

    In this technical survey, we comprehensively summarize the latest advancements in the field of recommender systems. The objective of this study is to provide an overview of the current state-of-the-art in the field and highlight the latest trends in the development of recommender systems. The study starts with a comprehensive summary of the main taxonomy of recommender systems, including personalized and group recommender systems, and then delves into the category of knowledge-based recommender systems. In addition, the survey analyzes the robustness, data bias, and fairness issues in recommender systems, summarizing the evaluation metrics used to assess the performance of these systems. Finally, the study provides insights into the latest trends in the development of recommender systems and highlights the new directions for future research in the field.
    
[^6]: 多分布稠密信息检索的资源和评估

    Resources and Evaluations for Multi-Distribution Dense Information Retrieval. (arXiv:2306.12601v1 [cs.IR])

    [http://arxiv.org/abs/2306.12601](http://arxiv.org/abs/2306.12601)

    本文提出了一个新问题——多分布信息检索，并通过三个基准数据集展示了简单方法的有效性，可防止已知领域消耗大部分检索预算，平均提高Recall @ 100点3.8+，最高可达8.0个点。

    

    我们引入并定义了多分布信息检索（IR）的新问题，即在给定查询的情况下，系统需要从多个集合中检索出段落，每个集合都来自不同的分布。其中一些集合和分布可能在训练时不可用。为了评估多分布检索的方法，我们从现有的单分布数据集设计了三个基准，分别是基于问题回答的数据集和两个基于实体匹配的数据集。我们提出了针对此任务的简单方法，该方法在域之间战略性地分配固定的检索预算（前k个段落），以防已知领域消耗大部分预算。我们展示我们的方法在数据集上导致了平均3.8+和高达8.0个Recall @ 100点的提高，并且在微调不同的基础检索模型时改进是一致的。我们的基准公开可用。

    We introduce and define the novel problem of multi-distribution information retrieval (IR) where given a query, systems need to retrieve passages from within multiple collections, each drawn from a different distribution. Some of these collections and distributions might not be available at training time. To evaluate methods for multi-distribution retrieval, we design three benchmarks for this task from existing single-distribution datasets, namely, a dataset based on question answering and two based on entity matching. We propose simple methods for this task which allocate the fixed retrieval budget (top-k passages) strategically across domains to prevent the known domains from consuming most of the budget. We show that our methods lead to an average of 3.8+ and up to 8.0 points improvements in Recall@100 across the datasets and that improvements are consistent when fine-tuning different base retrieval models. Our benchmarks are made publicly available.
    
[^7]: ReLoop2: 通过响应式误差补偿循环构建自适应推荐模型

    ReLoop2: Building Self-Adaptive Recommendation Models via Responsive Error Compensation Loop. (arXiv:2306.08808v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.08808](http://arxiv.org/abs/2306.08808)

    本文提出了ReLoop2，一种利用响应式误差补偿循环构建自适应推荐模型的方法，通过错误记忆模块来补偿模型预测误差，实现快速模型适应。

    

    工业推荐系统面临非静态环境下运行的挑战，其中数据分布的转移源于用户行为的演变。为了解决这个挑战，一种常见的方法是使用新观察到的数据定期重新训练或增量更新已部署的深度模型，产生连续的训练过程。然而，神经网络的传统学习范式依赖于小学习率的迭代梯度更新，使得大推荐模型很难适应。本文引入了ReLoop2，一种自我纠正的学习环路，通过响应式误差补偿促进在线推荐系统中的快速模型适应。本文受人类大脑中观察到的慢-快互补学习系统的启发，提出一个错误记忆模块，直接存储来自数据流的错误样本。随后利用这些存储的样本来补偿模型预测误差。

    Industrial recommender systems face the challenge of operating in non-stationary environments, where data distribution shifts arise from evolving user behaviors over time. To tackle this challenge, a common approach is to periodically re-train or incrementally update deployed deep models with newly observed data, resulting in a continual training process. However, the conventional learning paradigm of neural networks relies on iterative gradient-based updates with a small learning rate, making it slow for large recommendation models to adapt. In this paper, we introduce ReLoop2, a self-correcting learning loop that facilitates fast model adaptation in online recommender systems through responsive error compensation. Inspired by the slow-fast complementary learning system observed in human brains, we propose an error memory module that directly stores error samples from incoming data streams. These stored samples are subsequently leveraged to compensate for model prediction errors durin
    
[^8]: IGB: 针对公共图形数据集在标记、特征、异质性和大小方面的差距为深度学习研究提供了解决方案

    IGB: Addressing The Gaps In Labeling, Features, Heterogeneity, and Size of Public Graph Datasets for Deep Learning Research. (arXiv:2302.13522v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.13522](http://arxiv.org/abs/2302.13522)

    IGB是一个研究数据集工具，包含同质和异质性学术图形，规模巨大，并提供工具用于生成不同特性的合成图形，为GNN研究人员提供解决公共图形数据集在标记、特征、异质性和大小方面差距的有价值资源。

    

    图形神经网络 (GNNs) 已经展示了对于各种具有挑战性的真实应用的巨大潜力，但 GNN 研究中的一个主要障碍是缺乏大规模灵活的数据集。现有的大部分公共GNN数据集都相对较小，这限制了 GNN 的推广到未知数据的能力。很少有大型的图形数据集提供丰富的标记数据，这使得难以确定 GNN 模型在未知数据上的低准确性是由于训练数据不足还是模型无法推广。此外，训练 GNN 的数据集需要提供灵活性，以便深入研究各种因素对 GNN 模型训练的影响。在这项工作中，我们介绍了伊利诺伊图形基准 (IGB)，这是一个研究数据集工具，开发人员可以使用它来高精度地训练、审查和系统地评估GNN模型。IGB 包括同质和异质性学术图形，规模巨大，并且可以标记和操作，以模拟真实场景。该数据集还包括用于生成具有不同特性的合成图形的工具，使研究人员能够探索各种图形特性对 GNN 的影响。我们相信，伊利诺伊图形基准将为 GNN 研究团体提供有价值的资源，以解决公共图形数据集在标记、特征、异质性和大小方面的差距，以用于深度学习研究。

    Graph neural networks (GNNs) have shown high potential for a variety of real-world, challenging applications, but one of the major obstacles in GNN research is the lack of large-scale flexible datasets. Most existing public datasets for GNNs are relatively small, which limits the ability of GNNs to generalize to unseen data. The few existing large-scale graph datasets provide very limited labeled data. This makes it difficult to determine if the GNN model's low accuracy for unseen data is inherently due to insufficient training data or if the model failed to generalize. Additionally, datasets used to train GNNs need to offer flexibility to enable a thorough study of the impact of various factors while training GNN models.  In this work, we introduce the Illinois Graph Benchmark (IGB), a research dataset tool that the developers can use to train, scrutinize and systematically evaluate GNN models with high fidelity. IGB includes both homogeneous and heterogeneous academic graphs of enorm
    

