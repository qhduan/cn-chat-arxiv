# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unbiased Learning to Rank Meets Reality: Lessons from Baidu's Large-Scale Search Dataset](https://arxiv.org/abs/2404.02543) | 本研究从百度搜索引擎发布的大规模搜索数据集出发，探讨了无偏向学习排序技术在实际搜索引擎中的表现，发现与排名损失和查询-文档特征选择相比，ULTR技术并未带来明显的性能改进。 |
| [^2] | [ChatQA: Building GPT-4 Level Conversational QA Models.](http://arxiv.org/abs/2401.10225) | ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。 |
| [^3] | [Dual Correction Strategy for Ranking Distillation in Top-N Recommender System.](http://arxiv.org/abs/2109.03459) | 本文提出了一种双重修正策略（DCD），用于在推荐系统中更有效地将教师模型的排名信息转移到学生模型。这种方法不仅充分利用了学生模型的预测误差，还提供了更全面的视角，解决了松弛排名蒸馏方法的限制。 |

# 详细

[^1]: 无偏向学习排序遇到现实：百度大规模搜索数据集的经验教训

    Unbiased Learning to Rank Meets Reality: Lessons from Baidu's Large-Scale Search Dataset

    [https://arxiv.org/abs/2404.02543](https://arxiv.org/abs/2404.02543)

    本研究从百度搜索引擎发布的大规模搜索数据集出发，探讨了无偏向学习排序技术在实际搜索引擎中的表现，发现与排名损失和查询-文档特征选择相比，ULTR技术并未带来明显的性能改进。

    

    无偏向学习排序（ULTR）是一个用于学习用户点击数据的成熟框架，而这些数据往往受收集数据的排名者的偏见影响。虽然在理论上得到证明并在模拟中进行了广泛测试，但ULTR技术缺乏经验验证，尤其是在现代搜索引擎中。百度搜索引擎发布的WSDM Cup 2023数据集为评估主要ULTR技术在真实世界中的表现提供了难得的机会。尽管在WSDM Cup 2023期间有多次提交，以及随后的NTCIR ULTRE-2任务，但目前还不清楚观察到的改进是否源自应用ULTR或其他学习技术。我们重新审视并扩展了现有实验。我们发现，无偏向学习排序技术并不能明显提升性能，尤其是与排名损失和查询-文档特征选择带来的明显差异相比。

    arXiv:2404.02543v1 Announce Type: cross  Abstract: Unbiased learning-to-rank (ULTR) is a well-established framework for learning from user clicks, which are often biased by the ranker collecting the data. While theoretically justified and extensively tested in simulation, ULTR techniques lack empirical validation, especially on modern search engines. The dataset released for the WSDM Cup 2023, collected from Baidu's search engine, offers a rare opportunity to assess the real-world performance of prominent ULTR techniques. Despite multiple submissions during the WSDM Cup 2023 and the subsequent NTCIR ULTRE-2 task, it remains unclear whether the observed improvements stem from applying ULTR or other learning techniques. We revisit and extend the available experiments. We find that unbiased learning-to-rank techniques do not bring clear performance improvements, especially compared to the stark differences brought by the choice of ranking loss and query-document features. Our experiments 
    
[^2]: ChatQA: 构建GPT-4级对话问答模型

    ChatQA: Building GPT-4 Level Conversational QA Models. (arXiv:2401.10225v1 [cs.CL])

    [http://arxiv.org/abs/2401.10225](http://arxiv.org/abs/2401.10225)

    ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。

    

    在这项工作中，我们介绍了ChatQA，一系列具有GPT-4级别准确性的对话问答模型。具体地，我们提出了一个两阶段的指令调整方法，可以显著提高大型语言模型（LLM）在零-shot对话问答中的结果。为了处理对话问答中的检索问题，我们在多轮问答数据集上进行了密集检索器的微调，这样可以提供与使用最先进的查询重写模型相当的结果，同时大大降低部署成本。值得注意的是，我们的ChatQA-70B可以在10个对话问答数据集的平均分上超过GPT-4（54.14 vs. 53.90），而不依赖于OpenAI GPT模型的任何合成数据。

    In this work, we introduce ChatQA, a family of conversational question answering (QA) models, that obtain GPT-4 level accuracies. Specifically, we propose a two-stage instruction tuning method that can significantly improve the zero-shot conversational QA results from large language models (LLMs). To handle retrieval in conversational QA, we fine-tune a dense retriever on a multi-turn QA dataset, which provides comparable results to using the state-of-the-art query rewriting model while largely reducing deployment cost. Notably, our ChatQA-70B can outperform GPT-4 in terms of average score on 10 conversational QA datasets (54.14 vs. 53.90), without relying on any synthetic data from OpenAI GPT models.
    
[^3]: 推荐系统中用于排名蒸馏的双重修正策略

    Dual Correction Strategy for Ranking Distillation in Top-N Recommender System. (arXiv:2109.03459v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2109.03459](http://arxiv.org/abs/2109.03459)

    本文提出了一种双重修正策略（DCD），用于在推荐系统中更有效地将教师模型的排名信息转移到学生模型。这种方法不仅充分利用了学生模型的预测误差，还提供了更全面的视角，解决了松弛排名蒸馏方法的限制。

    

    知识蒸馏是将训练充分的大模型（教师）的知识转移到小模型（学生）的重要研究领域，对于推荐系统的实际部署而言，它已成为一个重要的研究方向。最近，松弛排名蒸馏（RRD）表明，在推荐列表中蒸馏排名信息能够显著提高性能。然而，该方法仍然存在以下限制：1）它未充分利用学生模型的预测误差，使得训练效率不高；2）它只蒸馏用户侧的排名信息，在稀疏的隐式反馈下提供的视角不足。本文提出了一种更高效的蒸馏方法，即双重修正策略（DCD），通过教师模型和学生模型预测之间的差异来决定要蒸馏的知识。

    Knowledge Distillation (KD), which transfers the knowledge of a well-trained large model (teacher) to a small model (student), has become an important area of research for practical deployment of recommender systems. Recently, Relaxed Ranking Distillation (RRD) has shown that distilling the ranking information in the recommendation list significantly improves the performance. However, the method still has limitations in that 1) it does not fully utilize the prediction errors of the student model, which makes the training not fully efficient, and 2) it only distills the user-side ranking information, which provides an insufficient view under the sparse implicit feedback. This paper presents Dual Correction strategy for Distillation (DCD), which transfers the ranking information from the teacher model to the student model in a more efficient manner. Most importantly, DCD uses the discrepancy between the teacher model and the student model predictions to decide which knowledge to be disti
    

