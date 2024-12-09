# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [All-in-One: Heterogeneous Interaction Modeling for Cold-Start Rating Prediction](https://arxiv.org/abs/2403.17740) | 提出了异质交互评分网络（HIRE）框架，通过异质交互模块（HIM）来共同建模异质交互并直接推断重要特征 |
| [^2] | [TPRF: A Transformer-based Pseudo-Relevance Feedback Model for Efficient and Effective Retrieval.](http://arxiv.org/abs/2401.13509) | 本文提出一种基于Transformer的伪相关反馈模型（TPRF），适用于资源受限的环境。TPRF相比其他深度语言模型在内存占用和推理时间方面具备更小的开销，并能有效地结合来自稠密文具表示的相关反馈信号。 |

# 详细

[^1]: 一体化：异质交互建模用于冷启动评分预测

    All-in-One: Heterogeneous Interaction Modeling for Cold-Start Rating Prediction

    [https://arxiv.org/abs/2403.17740](https://arxiv.org/abs/2403.17740)

    提出了异质交互评分网络（HIRE）框架，通过异质交互模块（HIM）来共同建模异质交互并直接推断重要特征

    

    冷启动评分预测是推荐系统中一个基本问题，已得到广泛研究。许多方法已经被提出，利用现有数据之间的显式关系，例如协同过滤、社交推荐和异构信息网络，以缓解冷启动用户和物品的数据不足问题。然而，基于不同角色之间的数据构建的显式关系可能不可靠且无关，从而限制了特定推荐任务的性能上限。受此启发，本文提出了一个灵活的框架，名为异质交互评分网络（HIRE）。HIRE不仅仅依赖于预先定义的交互模式或手动构建的异构信息网络。相反，我们设计了一个异质交互模块（HIM），来共同建模异质交互并直接推断重要特征。

    arXiv:2403.17740v1 Announce Type: cross  Abstract: Cold-start rating prediction is a fundamental problem in recommender systems that has been extensively studied. Many methods have been proposed that exploit explicit relations among existing data, such as collaborative filtering, social recommendations and heterogeneous information network, to alleviate the data insufficiency issue for cold-start users and items. However, the explicit relations constructed based on data between different roles may be unreliable and irrelevant, which limits the performance ceiling of the specific recommendation task. Motivated by this, in this paper, we propose a flexible framework dubbed heterogeneous interaction rating network (HIRE). HIRE dose not solely rely on the pre-defined interaction pattern or the manually constructed heterogeneous information network. Instead, we devise a Heterogeneous Interaction Module (HIM) to jointly model the heterogeneous interactions and directly infer the important in
    
[^2]: TPRF:一种基于Transformer的伪相关反馈模型，用于高效且有效的检索。

    TPRF: A Transformer-based Pseudo-Relevance Feedback Model for Efficient and Effective Retrieval. (arXiv:2401.13509v1 [cs.IR])

    [http://arxiv.org/abs/2401.13509](http://arxiv.org/abs/2401.13509)

    本文提出一种基于Transformer的伪相关反馈模型（TPRF），适用于资源受限的环境。TPRF相比其他深度语言模型在内存占用和推理时间方面具备更小的开销，并能有效地结合来自稠密文具表示的相关反馈信号。

    

    本文考虑在资源受限的环境中，如廉价云实例或嵌入式系统（如智能手机和智能手表）中，针对稠密检索器的伪相关反馈（PRF）方法，其中内存和CPU受限，没有GPU。为此，我们提出了一种基于Transformer的PRF方法（TPRF），与采用PRF机制的其他深度语言模型相比，具有更小的内存占用和更快的推理时间，较小的效果损失。TPRF学习如何有效地结合来自稠密文具表示的相关反馈信号。具体而言，TPRF提供了一种建模查询和相关反馈信号之间关系和权重的机制。该方法对所使用的具体稠密表示不加偏见，因此可以广泛应用于任何稠密检索器。

    This paper considers Pseudo-Relevance Feedback (PRF) methods for dense retrievers in a resource constrained environment such as that of cheap cloud instances or embedded systems (e.g., smartphones and smartwatches), where memory and CPU are limited and GPUs are not present. For this, we propose a transformer-based PRF method (TPRF), which has a much smaller memory footprint and faster inference time compared to other deep language models that employ PRF mechanisms, with a marginal effectiveness loss. TPRF learns how to effectively combine the relevance feedback signals from dense passage representations. Specifically, TPRF provides a mechanism for modelling relationships and weights between the query and the relevance feedback signals. The method is agnostic to the specific dense representation used and thus can be generally applied to any dense retriever.
    

