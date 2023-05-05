# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adaptive Selection of Anchor Items for CUR-based k-NN search with Cross-Encoders.](http://arxiv.org/abs/2305.02996) | 本文提出了一种自适应锚点选择方法，可以在保持较小的计算成本的同时，实现与随机抽样锚点相当或者更好的k-NN召回性能。 |
| [^2] | [Recent Advances in the Foundations and Applications of Unbiased Learning to Rank.](http://arxiv.org/abs/2305.02914) | 本文介绍了无偏学习排序（ULTR）的基础概念和最新进展，以及几种实际应用的方法。教程分为四个部分：偏差的概述，ULTR的最新估计技术，ULTR在实际应用中的表现，以及ULTR与排名公平性的联系。 |
| [^3] | [Maximizing Submodular Functions for Recommendation in the Presence of Biases.](http://arxiv.org/abs/2305.02806) | 该论文研究了如何在存在偏见的情况下，通过最大化子模函数来优化推荐系统。先前研究指出，基于公平性约束的干预可以确保比例代表性，并在存在偏见时获得接近最优的效用。而本文则探讨了一组能够捕捉这种目的的子模函数。 |
| [^4] | [Towards Hierarchical Policy Learning for Conversational Recommendation with Hypergraph-based Reinforcement Learning.](http://arxiv.org/abs/2305.02575) | 本文提出了一种新颖的基于超图强化学习的分层对话推荐模型，其中导演通过超图算法进行选择，帮助演员减少行动空间和指导对话朝着最具信息性的属性方向进行，并根据用户在对话中的偏好选择物品。 |
| [^5] | [Analyzing Hong Kong's Legal Judgments from a Computational Linguistics point-of-view.](http://arxiv.org/abs/2305.02558) | 本文提供了多种基于统计、机器学习和深度学习等方法来有效地分析香港的法律判决，并从中提取关键信息，解决了价格高和资源缺乏的问题。 |
| [^6] | [Inference at Scale Significance Testing for Large Search and Recommendation Experiments.](http://arxiv.org/abs/2305.02461) | 本文研究了大规模搜索和推荐实验的显著性检验行为，结果发现在大样本下Wilcoxon和Sign测试的1型错误率显著更高，建议在这种情况下使用bootstrap、随机化和t测试。 |
| [^7] | [Towards Imperceptible Document Manipulations against Neural Ranking Models.](http://arxiv.org/abs/2305.01860) | 该论文提出了一种针对神经排序模型的不易被检测到的对抗性攻击框架，称为“几乎不可察觉文档操作”（IDEM）。IDEM使用生成语言模型生成连结句，无法引入易于检测的错误，并且使用单独的位置合并策略来平衡扰动文本的相关性和连贯性，实验结果表明，IDEM可以在保持高人类评估得分的同时优于强基线。 |
| [^8] | [Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation.](http://arxiv.org/abs/2304.14668) | 本研究提出了一种基于对比知识蒸馏的集成建模方法EMKD，它采用多个并行网络作为序列编码器，在序列推荐中根据所有网络的输出分布推荐物品。实验证明，EMKD在两个真实世界数据集上的表现显著优于最先进的方法。 |
| [^9] | [Unsupervised Story Discovery from Continuous News Streams via Scalable Thematic Embedding.](http://arxiv.org/abs/2304.04099) | 本研究提出了一种新颖的主题嵌入方法和一个可扩展的无监督在线故事发现框架USTORY，可以动态表示文章和故事，并考虑它们共享的时间主题和新颖性，以帮助人们消化大量的新闻流。 |
| [^10] | [Reasoning with Language Model Prompting: A Survey.](http://arxiv.org/abs/2212.09597) | 本文提供了使用语言模型提示进行推理的前沿研究综合调查。讨论了新兴推理能力出现的潜在原因，并提供系统资源帮助初学者。 |
| [^11] | [Simplified TinyBERT: Knowledge Distillation for Document Retrieval.](http://arxiv.org/abs/2009.07531) | 本文提出了一种基于知识蒸馏的文档检索模型Simplified TinyBERT，它在提供15倍速度提升的情况下比BERT-Base表现更好。 |
| [^12] | [Vertex Nomination in Richly Attributed Networks.](http://arxiv.org/abs/2005.02151) | 本文探讨了富有属性网络中顶点提名的双重作用，并提出了一种新颖的基于内容感知的网络嵌入方法，证明该方法优于现有的不利用内容和上下文的顶点提名方法。 |

# 详细

[^1]: 带有交叉编码器的CUR k-NN搜索的自适应锚定项选择

    Adaptive Selection of Anchor Items for CUR-based k-NN search with Cross-Encoders. (arXiv:2305.02996v1 [cs.IR])

    [http://arxiv.org/abs/2305.02996](http://arxiv.org/abs/2305.02996)

    本文提出了一种自适应锚点选择方法，可以在保持较小的计算成本的同时，实现与随机抽样锚点相当或者更好的k-NN召回性能。

    

    本文提出了一种自适应锚点选择方法，以改善ANNCUR模型中高前k项的逼近误差和召回率。该方法可以在保持较小的计算成本的同时，实现与随机抽样锚点相当或者更好的k-NN召回性能。

    Cross-encoder models, which jointly encode and score a query-item pair, are typically prohibitively expensive for k-nearest neighbor search. Consequently, k-NN search is performed not with a cross-encoder, but with a heuristic retrieve (e.g., using BM25 or dual-encoder) and re-rank approach. Recent work proposes ANNCUR (Yadav et al., 2022) which uses CUR matrix factorization to produce an embedding space for efficient vector-based search that directly approximates the cross-encoder without the need for dual-encoders. ANNCUR defines this shared query-item embedding space by scoring the test query against anchor items which are sampled uniformly at random. While this minimizes average approximation error over all items, unsuitably high approximation error on top-k items remains and leads to poor recall of top-k (and especially top-1) items. Increasing the number of anchor items is a straightforward way of improving the approximation error and hence k-NN recall of ANNCUR but at the cost o
    
[^2]: 无偏学习排序的基础和应用的最新进展

    Recent Advances in the Foundations and Applications of Unbiased Learning to Rank. (arXiv:2305.02914v1 [cs.IR])

    [http://arxiv.org/abs/2305.02914](http://arxiv.org/abs/2305.02914)

    本文介绍了无偏学习排序（ULTR）的基础概念和最新进展，以及几种实际应用的方法。教程分为四个部分：偏差的概述，ULTR的最新估计技术，ULTR在实际应用中的表现，以及ULTR与排名公平性的联系。

    

    无偏学习排序（ULTR）领域自诞生以来一直处于非常活跃的状态，并在近年来取得了几项有影响力的进展。本教程既介绍了该领域的核心概念，又概述了其基础的最新进展以及其方法的几种应用。本教程分为四部分：首先，我们概述了可以用ULTR方法解决的不同形式的偏差。其次，我们全面讨论了ULTR领域的最新估计技术。第三，我们调查了ULTR在实际应用中的发布结果。第四，我们讨论了ULTR与排名公平性之间的联系。最后，我们简要反思了ULTR研究及其应用的未来。本教程旨在使对开发新的ULTR解决方案或在实际应用中利用它们的研究人员和工业实践者受益。

    Since its inception, the field of unbiased learning to rank (ULTR) has remained very active and has seen several impactful advancements in recent years. This tutorial provides both an introduction to the core concepts of the field and an overview of recent advancements in its foundations along with several applications of its methods. The tutorial is divided into four parts: Firstly, we give an overview of the different forms of bias that can be addressed with ULTR methods. Secondly, we present a comprehensive discussion of the latest estimation techniques in the ULTR field. Thirdly, we survey published results of ULTR in real-world applications. Fourthly, we discuss the connection between ULTR and fairness in ranking. We end by briefly reflecting on the future of ULTR research and its applications. This tutorial is intended to benefit both researchers and industry practitioners who are interested in developing new ULTR solutions or utilizing them in real-world applications.
    
[^3]: 在存在偏见的情况下，最大化子模函数用于推荐系统

    Maximizing Submodular Functions for Recommendation in the Presence of Biases. (arXiv:2305.02806v1 [cs.LG])

    [http://arxiv.org/abs/2305.02806](http://arxiv.org/abs/2305.02806)

    该论文研究了如何在存在偏见的情况下，通过最大化子模函数来优化推荐系统。先前研究指出，基于公平性约束的干预可以确保比例代表性，并在存在偏见时获得接近最优的效用。而本文则探讨了一组能够捕捉这种目的的子模函数。

    

    子集选择任务在推荐系统和搜索引擎中经常出现，要求选择一些最大化用户价值的物品子集。子集的价值往往呈现出递减的回报，因此，使用子模函数来建模。然而，在许多应用中，发现输入具有社会偏见，会降低输出子集的效用，因此需要干预以提高其效用。本文研究了一组子模函数的最大化，这些函数涵盖了上述应用中出现的函数。

    Subset selection tasks, arise in recommendation systems and search engines and ask to select a subset of items that maximize the value for the user. The values of subsets often display diminishing returns, and hence, submodular functions have been used to model them. If the inputs defining the submodular function are known, then existing algorithms can be used. In many applications, however, inputs have been observed to have social biases that reduce the utility of the output subset. Hence, interventions to improve the utility are desired. Prior works focus on maximizing linear functions -- a special case of submodular functions -- and show that fairness constraint-based interventions can not only ensure proportional representation but also achieve near-optimal utility in the presence of biases. We study the maximization of a family of submodular functions that capture functions arising in the aforementioned applications. Our first result is that, unlike linear functions, constraint-ba
    
[^4]: 基于超图强化学习的分层对话推荐中的策略学习

    Towards Hierarchical Policy Learning for Conversational Recommendation with Hypergraph-based Reinforcement Learning. (arXiv:2305.02575v1 [cs.IR])

    [http://arxiv.org/abs/2305.02575](http://arxiv.org/abs/2305.02575)

    本文提出了一种新颖的基于超图强化学习的分层对话推荐模型，其中导演通过超图算法进行选择，帮助演员减少行动空间和指导对话朝着最具信息性的属性方向进行，并根据用户在对话中的偏好选择物品。

    

    对话推荐系统旨在通过对话及时主动地获取用户的偏好，并推荐相应的物品。然而，现有的方法往往使用统一的决策模块或启发式规则，而忽略了不同决策过程的角色差异和相互作用。为此，本文提出了一种新颖的基于超图强化学习的分层对话推荐模型，其中导演通过超图算法进行选择，帮助演员减少行动空间，指导对话朝着最具信息性的属性方向进行，并根据用户在对话中的偏好选择物品。实验结果表明，与现有方法相比，本文方法在真实数据集上表现出更高的效果和优越性。

    Conversational recommendation systems (CRS) aim to timely and proactively acquire user dynamic preferred attributes through conversations for item recommendation. In each turn of CRS, there naturally have two decision-making processes with different roles that influence each other: 1) director, which is to select the follow-up option (i.e., ask or recommend) that is more effective for reducing the action space and acquiring user preferences; and 2) actor, which is to accordingly choose primitive actions (i.e., asked attribute or recommended item) that satisfy user preferences and give feedback to estimate the effectiveness of the director's option. However, existing methods heavily rely on a unified decision-making module or heuristic rules, while neglecting to distinguish the roles of different decision procedures, as well as the mutual influences between them. To address this, we propose a novel Director-Actor Hierarchical Conversational Recommender (DAHCR), where the director select
    
[^5]: 以计算语言学视角分析香港的法律判决

    Analyzing Hong Kong's Legal Judgments from a Computational Linguistics point-of-view. (arXiv:2305.02558v1 [cs.CL])

    [http://arxiv.org/abs/2305.02558](http://arxiv.org/abs/2305.02558)

    本文提供了多种基于统计、机器学习和深度学习等方法来有效地分析香港的法律判决，并从中提取关键信息，解决了价格高和资源缺乏的问题。

    

    利用计算语言学从法律判决中提取有用信息是信息检索领域早期提出的问题之一。目前，存在多个商业供应商自动化执行这些任务。然而，在分析香港法院系统的判决时，存在价格过高和缺乏资源的关键瓶颈。本文提供了几种基于统计学、机器学习、深度学习和零样本学习的方法，以有效地分析香港法院系统的法律判决。所提出的方法包括：（1）引文网络图生成，（2）PageRank算法，（3）关键词分析和摘要，（4）情感极性，以及（5）段落分类，以便能够提取单个判决以及群体判决的关键见解。这将使对香港判决的整体分析变得不那么繁琐。

    Analysis and extraction of useful information from legal judgments using computational linguistics was one of the earliest problems posed in the domain of information retrieval. Presently, several commercial vendors exist who automate such tasks. However, a crucial bottleneck arises in the form of exorbitant pricing and lack of resources available in analysis of judgements mete out by Hong Kong's Legal System. This paper attempts to bridge this gap by providing several statistical, machine learning, deep learning and zero-shot learning based methods to effectively analyze legal judgments from Hong Kong's Court System. The methods proposed consists of: (1) Citation Network Graph Generation, (2) PageRank Algorithm, (3) Keyword Analysis and Summarization, (4) Sentiment Polarity, and (5) Paragrah Classification, in order to be able to extract key insights from individual as well a group of judgments together. This would make the overall analysis of judgments in Hong Kong less tedious and m
    
[^6]: 大规模搜索和推荐实验的显著性检验

    Inference at Scale Significance Testing for Large Search and Recommendation Experiments. (arXiv:2305.02461v1 [cs.IR])

    [http://arxiv.org/abs/2305.02461](http://arxiv.org/abs/2305.02461)

    本文研究了大规模搜索和推荐实验的显著性检验行为，结果发现在大样本下Wilcoxon和Sign测试的1型错误率显著更高，建议在这种情况下使用bootstrap、随机化和t测试。

    

    许多信息检索研究已经进行了评估，以确定哪种统计技术适用于比较系统。然而，这些研究集中于TREC样式的实验，通常少于100个主题。没有类似的研究适用于大规模搜索和推荐实验；这些研究通常涉及数千个主题或用户以及更稀疏的相关性判断，因此不清楚分析传统TREC实验的建议是否适用于这些情况。在本文中，我们实证研究了大规模搜索和推荐评估数据的显著性检验行为。我们的结果显示，Wilcoxon和Sign测试显示出显著更高的1型错误率，而不是更一致符合预期错误率的bootstrap、随机化和t测试。虽然统计测试在样本较小时显示出功率差异，但在功率相同时显示出没有区别。

    A number of information retrieval studies have been done to assess which statistical techniques are appropriate for comparing systems. However, these studies are focused on TREC-style experiments, which typically have fewer than 100 topics. There is no similar line of work for large search and recommendation experiments; such studies typically have thousands of topics or users and much sparser relevance judgements, so it is not clear if recommendations for analyzing traditional TREC experiments apply to these settings. In this paper, we empirically study the behavior of significance tests with large search and recommendation evaluation data. Our results show that the Wilcoxon and Sign tests show significantly higher Type-1 error rates for large sample sizes than the bootstrap, randomization and t-tests, which were more consistent with the expected error rate. While the statistical tests displayed differences in their power for smaller sample sizes, they showed no difference in their po
    
[^7]: 针对神经排序模型的几乎不可察觉的文档篡改

    Towards Imperceptible Document Manipulations against Neural Ranking Models. (arXiv:2305.01860v1 [cs.IR])

    [http://arxiv.org/abs/2305.01860](http://arxiv.org/abs/2305.01860)

    该论文提出了一种针对神经排序模型的不易被检测到的对抗性攻击框架，称为“几乎不可察觉文档操作”（IDEM）。IDEM使用生成语言模型生成连结句，无法引入易于检测的错误，并且使用单独的位置合并策略来平衡扰动文本的相关性和连贯性，实验结果表明，IDEM可以在保持高人类评估得分的同时优于强基线。

    

    对抗性攻击已经开始应用于发现神经排序模型（NRMs）中的潜在漏洞，但是当前攻击方法常常会引入语法错误，无意义的表达，或不连贯的文本片段，这些都很容易被检测到。此外，当前方法严重依赖于使用与真实的NRM相似的模拟NRM来保证攻击效果，这使得它们在实践中难以使用。为了解决这些问题，我们提出了一个称为“几乎不可察觉文档操作”（IDEM）的框架，用于生成对算法和人类来说都不太明显的对抗文档。IDEM指示一个经过良好建立的生成语言模型（例如BART）生成连接句，而不会引入易于检测的错误，并采用单独的逐位置合并策略来平衡扰动文本的相关性和连贯性。在流行的MS MARCO基准上的实验结果表明，IDEM可以在保持高人类评估得分的同时，优于强基线。

    Adversarial attacks have gained traction in order to identify potential vulnerabilities in neural ranking models (NRMs), but current attack methods often introduce grammatical errors, nonsensical expressions, or incoherent text fragments, which can be easily detected. Additionally, current methods rely heavily on the use of a well-imitated surrogate NRM to guarantee the attack effect, which makes them difficult to use in practice. To address these issues, we propose a framework called Imperceptible DocumEnt Manipulation (IDEM) to produce adversarial documents that are less noticeable to both algorithms and humans. IDEM instructs a well-established generative language model, such as BART, to generate connection sentences without introducing easy-to-detect errors, and employs a separate position-wise merging strategy to balance relevance and coherence of the perturbed text. Experimental results on the popular MS MARCO benchmark demonstrate that IDEM can outperform strong baselines while 
    
[^8]: 基于对比知识蒸馏的集成建模在序列推荐中的应用

    Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation. (arXiv:2304.14668v1 [cs.IR])

    [http://arxiv.org/abs/2304.14668](http://arxiv.org/abs/2304.14668)

    本研究提出了一种基于对比知识蒸馏的集成建模方法EMKD，它采用多个并行网络作为序列编码器，在序列推荐中根据所有网络的输出分布推荐物品。实验证明，EMKD在两个真实世界数据集上的表现显著优于最先进的方法。

    

    序列推荐旨在捕捉用户的动态兴趣，预测用户下一次的偏好物品。多数方法使用深度神经网络作为序列编码器生成用户和物品表示。现有工作主要侧重于设计更强的序列编码器。然而，很少有尝试使用训练一组网络作为序列编码器的方法，这比单个网络更强大，因为一组并行网络可以产生多样化的预测结果，从而获得更好的准确性。本文提出了一种基于对比知识蒸馏的集成建模方法，即EMKD，在序列推荐中使用多个并行网络作为序列编码器，并根据所有这些网络的输出分布推荐物品。为了促进并行网络之间的知识转移，我们提出了一种新颖的对比知识蒸馏方法，它将知识从教师网络转移到多个学生网络中。在两个真实世界数据集上的实验表明，我们提出的EMKD显著优于最先进的序列推荐方法和集成基线。

    Sequential recommendation aims to capture users' dynamic interest and predicts the next item of users' preference. Most sequential recommendation methods use a deep neural network as sequence encoder to generate user and item representations. Existing works mainly center upon designing a stronger sequence encoder. However, few attempts have been made with training an ensemble of networks as sequence encoders, which is more powerful than a single network because an ensemble of parallel networks can yield diverse prediction results and hence better accuracy. In this paper, we present Ensemble Modeling with contrastive Knowledge Distillation for sequential recommendation (EMKD). Our framework adopts multiple parallel networks as an ensemble of sequence encoders and recommends items based on the output distributions of all these networks. To facilitate knowledge transfer between parallel networks, we propose a novel contrastive knowledge distillation approach, which performs knowledge tran
    
[^9]: 通过可扩展的主题嵌入从连续新闻流中无监督地发现故事

    Unsupervised Story Discovery from Continuous News Streams via Scalable Thematic Embedding. (arXiv:2304.04099v1 [cs.IR])

    [http://arxiv.org/abs/2304.04099](http://arxiv.org/abs/2304.04099)

    本研究提出了一种新颖的主题嵌入方法和一个可扩展的无监督在线故事发现框架USTORY，可以动态表示文章和故事，并考虑它们共享的时间主题和新颖性，以帮助人们消化大量的新闻流。

    

    无监督地发现实时相关新闻文章故事，有助于人们在不需要昂贵人工注释的情况下消化大量的新闻流。现有的无监督在线故事发现研究的普遍方法是用符号或基于图的嵌入来表示新闻文章，并将它们逐步聚类成故事。最近的大型语言模型有望进一步改善嵌入，但是通过无差别地编码文章中的所有信息来直接采用这些模型无法有效处理富含文本且不断发展的新闻流。在这项工作中，我们提出了一种新颖的主题嵌入方法，使用现成的预训练句子编码器来动态表示文章和故事，并考虑它们共享的时间主题。为了实现无监督在线故事发现的想法，引入了一个可扩展框架USTORY，包括两个主要技术，即主题和时间感知的动态嵌入和新颖性感知的自适应聚类。

    Unsupervised discovery of stories with correlated news articles in real-time helps people digest massive news streams without expensive human annotations. A common approach of the existing studies for unsupervised online story discovery is to represent news articles with symbolic- or graph-based embedding and incrementally cluster them into stories. Recent large language models are expected to improve the embedding further, but a straightforward adoption of the models by indiscriminately encoding all information in articles is ineffective to deal with text-rich and evolving news streams. In this work, we propose a novel thematic embedding with an off-the-shelf pretrained sentence encoder to dynamically represent articles and stories by considering their shared temporal themes. To realize the idea for unsupervised online story discovery, a scalable framework USTORY is introduced with two main techniques, theme- and time-aware dynamic embedding and novelty-aware adaptive clustering, fuel
    
[^10]: 使用语言模型提示进行推理：一项调查

    Reasoning with Language Model Prompting: A Survey. (arXiv:2212.09597v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09597](http://arxiv.org/abs/2212.09597)

    本文提供了使用语言模型提示进行推理的前沿研究综合调查。讨论了新兴推理能力出现的潜在原因，并提供系统资源帮助初学者。

    

    推理作为复杂问题解决的重要能力，可以为医疗诊断、谈判等各种实际应用提供后端支持。本文对使用语言模型提示进行推理的前沿研究进行了综合调查。我们介绍了研究成果的比较和总结，并提供了系统资源以帮助初学者。我们还讨论了新兴推理能力出现的潜在原因，并突出了未来的研究方向。资源可在 https://github.com/zjunlp/Prompt4ReasoningPapers 上获取（定期更新）。

    Reasoning, as an essential ability for complex problem-solving, can provide back-end support for various real-world applications, such as medical diagnosis, negotiation, etc. This paper provides a comprehensive survey of cutting-edge research on reasoning with language model prompting. We introduce research works with comparisons and summaries and provide systematic resources to help beginners. We also discuss the potential reasons for emerging such reasoning abilities and highlight future research directions. Resources are available at https://github.com/zjunlp/Prompt4ReasoningPapers (updated periodically).
    
[^11]: 简化版TinyBERT: 用于文档检索的知识蒸馏

    Simplified TinyBERT: Knowledge Distillation for Document Retrieval. (arXiv:2009.07531v2 [cs.IR] CROSS LISTED)

    [http://arxiv.org/abs/2009.07531](http://arxiv.org/abs/2009.07531)

    本文提出了一种基于知识蒸馏的文档检索模型Simplified TinyBERT，它在提供15倍速度提升的情况下比BERT-Base表现更好。

    

    尽管利用BERT模型进行文档排序十分有效，但这种方法的高计算成本限制了其使用。因此，本文首先在文档排序任务上实证研究了两个知识蒸馏模型的有效性。此外，在最近提出的TinyBERT模型基础上，提出了两种简化方案。两个不同并且广泛使用的基准测试的评估表明，具有所提出简化方案的Simplified TinyBERT不仅提升了TinyBERT，而且在提供15倍速度提升的情况下也明显优于BERT-Base。

    Despite the effectiveness of utilizing the BERT model for document ranking, the high computational cost of such approaches limits their uses. To this end, this paper first empirically investigates the effectiveness of two knowledge distillation models on the document ranking task. In addition, on top of the recently proposed TinyBERT model, two simplifications are proposed. Evaluations on two different and widely-used benchmarks demonstrate that Simplified TinyBERT with the proposed simplifications not only boosts TinyBERT, but also significantly outperforms BERT-Base when providing 15$\times$ speedup.
    
[^12]: 富有属性网络中的顶点提名

    Vertex Nomination in Richly Attributed Networks. (arXiv:2005.02151v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2005.02151](http://arxiv.org/abs/2005.02151)

    本文探讨了富有属性网络中顶点提名的双重作用，并提出了一种新颖的基于内容感知的网络嵌入方法，证明该方法优于现有的不利用内容和上下文的顶点提名方法。

    

    顶点提名是一项轻度监督的网络信息检索任务，在这个任务中，感兴趣的一张图的顶点被用来查询第二张图以发现感兴趣的第二张图的顶点。与其他信息检索任务类似，顶点提名方案的输出是第二张图中顶点的排序列表，理想情况下，未知的感兴趣的顶点应该集中在列表的顶部。顶点提名方案为高效地挖掘复杂网络中的相关信息提供了有用的工具。在本文中，我们从理论和实践两方面探讨了内容（即边缘和顶点属性）和上下文（即网络拓扑结构）在顶点提名中的双重作用。我们提供了必要和充分的条件，证明了利用内容和上下文的顶点提名方案能够超越仅利用内容或上下文的方案。虽然内容和上下文的联合效用在其他网络分析任务中已经得到证实，但我们证明在顶点提名的背景下，这种联合效用也是成立的。此外，我们提出了一种新颖的基于内容感知的网络嵌入方法，用于顶点提名，可以有效地结合局部和全局网络属性信息。我们在真实的社交和引用网络上进行了实验，证明了我们提出的方法优于不利用内容和上下文的现有的顶点提名方法。

    Vertex nomination is a lightly-supervised network information retrieval task in which vertices of interest in one graph are used to query a second graph to discover vertices of interest in the second graph. Similar to other information retrieval tasks, the output of a vertex nomination scheme is a ranked list of the vertices in the second graph, with the heretofore unknown vertices of interest ideally concentrating at the top of the list. Vertex nomination schemes provide a useful suite of tools for efficiently mining complex networks for pertinent information. In this paper, we explore, both theoretically and practically, the dual roles of content (i.e., edge and vertex attributes) and context (i.e., network topology) in vertex nomination. We provide necessary and sufficient conditions under which vertex nomination schemes that leverage both content and context outperform schemes that leverage only content or context separately. While the joint utility of both content and context has 
    

