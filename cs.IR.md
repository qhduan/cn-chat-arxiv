# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Mining Reviews in Open Source Code for Developers Trail: A Process Mining Approach.](http://arxiv.org/abs/2308.00686) | 本文提出使用过程挖掘方法，在开源代码中构建日志来追踪和了解开源社区的活跃成员和活动类型。 |
| [^2] | [TimePool: Visually Answer "Which and When" Questions On Univariate Time Series.](http://arxiv.org/abs/2308.00682) | TimePool 是一个用于解决一元时间序列分析需求的可视化原型，允许用户构建交互式查询并通过可视化的方式探索结果。 |
| [^3] | [Explainable Graph Spectral Clustering of Text Documents.](http://arxiv.org/abs/2308.00504) | 本文提出了一种可解释的文本文档的图谱聚类方法，通过展示组合拉普拉斯嵌入、K嵌入和词向量空间嵌入之间的等价性，构建了文本内容和聚类结果之间的桥梁。 |
| [^4] | [On the Effects of Regional Spelling Conventions in Retrieval Models.](http://arxiv.org/abs/2308.00480) | 本论文研究了区域拼写习惯对检索模型的影响，发现神经排名模型在同义词情况下具有良好的泛化能力，尽管训练数据中存在美式拼写偏差。规范化文档的拼写会影响所有模型的性能。 |
| [^5] | [Generative Query Reformulation for Effective Adhoc Search.](http://arxiv.org/abs/2308.00415) | 本论文研究了使用生成式语言模型进行查询重构的能力与传统的使用伪相关反馈的方法相比较，使用两个代表性的查询重构框架GenQR和GenPRF进行研究。 |
| [^6] | [Challenging the Myth of Graph Collaborative Filtering: a Reasoned and Reproducibility-driven Analysis.](http://arxiv.org/abs/2308.00404) | 本文研究挑战图协同过滤的神话，通过关注结果的可复制性，成功复制了六个流行的图推荐模型在几个常见和新数据集上的结果，并与传统协同过滤模型进行比较。 |
| [^7] | [Detection and Classification of Novel Attacks and Anomaly in IoT Network using Rule based Deep Learning Model.](http://arxiv.org/abs/2308.00005) | 本论文提出了一种基于规则的深度学习模型，用于检测和分类物联网网络中的新型攻击和异常。该模型能够有效改善传统机器学习模型在该领域中的表现。 |
| [^8] | [Jina Embeddings: A Novel Set of High-Performance Sentence Embedding Models.](http://arxiv.org/abs/2307.11224) | Jina Embeddings是一组高性能的句子嵌入模型，能够捕捉文本的语义本质。该论文详细介绍了Jina Embeddings的开发过程，并通过性能评估验证了其优越性能。 |
| [^9] | [In-Context Retrieval-Augmented Language Models.](http://arxiv.org/abs/2302.00083) | 本研究提出了一种上下文检索增强的语言模型（In-Context RALM）方法，通过将相关文件作为输入的一部分，无需对语言模型进行进一步的训练即可显著提高语言建模性能和源归因能力，并且相对于现有的RALM方法，它具有更简单的部署过程。 |

# 详细

[^1]: 在开源代码中挖掘开发者的评论：一种过程挖掘方法

    Mining Reviews in Open Source Code for Developers Trail: A Process Mining Approach. (arXiv:2308.00686v1 [cs.SE])

    [http://arxiv.org/abs/2308.00686](http://arxiv.org/abs/2308.00686)

    本文提出使用过程挖掘方法，在开源代码中构建日志来追踪和了解开源社区的活跃成员和活动类型。

    

    审计轨迹是任何日志中记录活动执行者的证据性指标。现代反应式系统（如事务处理系统、管理信息系统、决策支持系统甚至高级管理系统）在执行日常任务时记录用户的活动，其中一个最重要的原因可能是安全性。为了有效监控和管理隐私和信息访问，这些日志中捕获和记录的轨迹在这方面起着至关重要的作用。然而，在开源领域，情况并非如此。尽管自由软件的目标是允许访问、免费分发和修改代码的权利，但拥有此类审计轨迹可以帮助追踪和了解这些社区的活跃成员以及他们执行的活动类型。在本文中，我们提出使用过程挖掘来构建日志，尽可能利用在开源存储库中找到的数据，以产生一个过程挖掘模型。

    Audit trails are evidential indications of activities performers in any logs. Modern reactive systems such as transaction processing systems, management information systems, decision support systems and even executive management systems log activities of users as they perform their daily tasks for a number of reasons and perhaps one of the most important is security. In order to efficiently monitor and manage privacy and access to information, the trails as captured and recorded in these logs play a pivotal role in this regard. In Open Source realm, however, this is not the case. Although the objective with free software is to allow for access, free distribution and the rights to modify coding, having such audit trails can help to trace and understand how active members of these communities are and the type of activities they perform. In this paper, we propose using process mining to construct logs using as much data as can be found in open source repositories in order to produce a pro
    
[^2]: TimePool：可视化回答“哪个时刻以及何时”问题的一元时间序列

    TimePool: Visually Answer "Which and When" Questions On Univariate Time Series. (arXiv:2308.00682v1 [cs.HC])

    [http://arxiv.org/abs/2308.00682](http://arxiv.org/abs/2308.00682)

    TimePool 是一个用于解决一元时间序列分析需求的可视化原型，允许用户构建交互式查询并通过可视化的方式探索结果。

    

    当探索时间序列数据集时，分析师经常提出“哪个时刻以及何时”问题。例如，对于超过一百年的世界人均寿命数据，他们可能询问前十个在人均寿命上的国家和他们达到这一地位的时间段，或者那些比爱尔兰的人均寿命更长的国家以及何时。本文提出了TimePool，一个用于解决一元时间序列分析需求的新的可视化原型。它允许用户构建交互式的“哪个时刻以及何时”查询，并通过可视化的方式探索结果以获取洞见。

    When exploring time series datasets, analysts often pose "which and when" questions. For example, with world life expectancy data over one hundred years, they may inquire about the top 10 countries in life expectancy and the time period when they achieved this status, or which countries have had longer life expectancy than Ireland and when. This paper proposes TimePool, a new visualization prototype, to address this need for univariate time series analysis. It allows users to construct interactive "which and when" queries and visually explore the results for insights.
    
[^3]: 可解释的文本文档的图谱聚类

    Explainable Graph Spectral Clustering of Text Documents. (arXiv:2308.00504v1 [cs.LG])

    [http://arxiv.org/abs/2308.00504](http://arxiv.org/abs/2308.00504)

    本文提出了一种可解释的文本文档的图谱聚类方法，通过展示组合拉普拉斯嵌入、K嵌入和词向量空间嵌入之间的等价性，构建了文本内容和聚类结果之间的桥梁。

    

    光谱聚类方法以其能够表示不同形状、密度等的聚类而闻名。然而，将这些算法应用于文本文档时，其结果很难向用户解释，特别是由于在光谱空间中的嵌入与文档内容没有明显的关系。因此，迫切需要研究解释聚类结果的方法。本文提出了对此目标的贡献。我们提出了解释基于组合拉普拉斯的图谱聚类结果的方法。该方法基于展示组合拉普拉斯嵌入、K嵌入（本文提出）和词向量空间嵌入的（近似）等价性。从而构建了文本内容和聚类结果之间的桥梁。我们为这种方法提供了理论背景。我们进行了实验研究，结果表明，在有利条件下，K嵌入很好地近似了拉普拉斯嵌入。

    Spectral clustering methods are known for their ability to represent clusters of diverse shapes, densities etc. However, results of such algorithms, when applied e.g. to text documents, are hard to explain to the user, especially due to embedding in the spectral space which has no obvious relation to document contents. Therefore there is an urgent need to elaborate methods for explaining the outcome of the clustering. This paper presents a contribution towards this goal. We present a proposal of explanation of results of combinatorial Laplacian based graph spectral clustering. It is based on showing (approximate) equivalence of combinatorial Laplacian embedding, $K$-embedding (proposed in this paper) and term vector space embedding. Hence a bridge is constructed between the textual contents and the clustering results. We provide theoretical background for this approach. We performed experimental study showing that $K$-embedding approximates well Laplacian embedding under favourable blo
    
[^4]: 关于区域拼写习惯对检索模型的影响的研究

    On the Effects of Regional Spelling Conventions in Retrieval Models. (arXiv:2308.00480v1 [cs.IR])

    [http://arxiv.org/abs/2308.00480](http://arxiv.org/abs/2308.00480)

    本论文研究了区域拼写习惯对检索模型的影响，发现神经排名模型在同义词情况下具有良好的泛化能力，尽管训练数据中存在美式拼写偏差。规范化文档的拼写会影响所有模型的性能。

    

    神经排名模型的一个优势是它们在同义词情况下具有很好的泛化能力，即当两个单词具有相似或相同的含义时。本研究调查并量化了各种排名模型在一个明确的同义词情况下的表现：当单词仅因区域拼写习惯的差异而以不同的形式表达时（例如color vs colour）。我们首先探索了用于神经检索方法的预训练、训练和评估的数据集中美式英语和英式英语拼写惯例的普遍性，并发现美式拼写惯例要远远多于英式拼写习惯。尽管训练数据存在这些偏差，我们发现检索模型在这种同义词情况下通常具有良好的泛化能力。我们研究了在检索中对文档拼写进行规范化的影响，并观察到所有模型都受到文档拼写规范化的影响。尽管在规范化的情况下它们都经历了性能下降。

    One advantage of neural ranking models is that they are meant to generalise well in situations of synonymity i.e. where two words have similar or identical meanings. In this paper, we investigate and quantify how well various ranking models perform in a clear-cut case of synonymity: when words are simply expressed in different surface forms due to regional differences in spelling conventions (e.g., color vs colour). We first explore the prevalence of American and British English spelling conventions in datasets used for the pre-training, training and evaluation of neural retrieval methods, and find that American spelling conventions are far more prevalent. Despite these biases in the training data, we find that retrieval models often generalise well in this case of synonymity. We explore the effect of document spelling normalisation in retrieval and observe that all models are affected by normalising the document's spelling. While they all experience a drop in performance when normalis
    
[^5]: 为有效的Adhoc搜索生成查询重构

    Generative Query Reformulation for Effective Adhoc Search. (arXiv:2308.00415v1 [cs.IR])

    [http://arxiv.org/abs/2308.00415](http://arxiv.org/abs/2308.00415)

    本论文研究了使用生成式语言模型进行查询重构的能力与传统的使用伪相关反馈的方法相比较，使用两个代表性的查询重构框架GenQR和GenPRF进行研究。

    

    在信息检索（IR）中，自动重构用户查询是一种常用的方法，用于提高效果，如伪相关反馈方法。最近生成式语言模型的进展展示了它们生成与给定提示相关的响应的能力。鉴于这一成功，我们致力于研究这些模型执行查询重构的能力，以及它们与使用伪相关反馈的长期查询重构方法相比较。具体而言，我们研究了两种代表性的查询重构框架，GenQR和GenPRF。GenQR直接重构用户输入的查询，而GenPRF通过使用伪相关反馈信息为查询提供附加上下文。对于每种重构方法，我们采用不同的技术，包括微调和直接提示。

    Performing automatic reformulations of a user's query is a popular paradigm used in information retrieval (IR) for improving effectiveness -- as exemplified by the pseudo-relevance feedback approaches, which expand the query in order to alleviate the vocabulary mismatch problem. Recent advancements in generative language models have demonstrated their ability in generating responses that are relevant to a given prompt. In light of this success, we seek to study the capacity of such models to perform query reformulation and how they compare with long-standing query reformulation methods that use pseudo-relevance feedback. In particular, we investigate two representative query reformulation frameworks, GenQR and GenPRF. GenQR directly reformulates the user's input query, while GenPRF provides additional context for the query by making use of pseudo-relevance feedback information. For each reformulation method, we leverage different techniques, including fine-tuning and direct prompting, 
    
[^6]: 挑战图协同过滤的神话：一项基于推理和可复制性的分析

    Challenging the Myth of Graph Collaborative Filtering: a Reasoned and Reproducibility-driven Analysis. (arXiv:2308.00404v1 [cs.IR])

    [http://arxiv.org/abs/2308.00404](http://arxiv.org/abs/2308.00404)

    本文研究挑战图协同过滤的神话，通过关注结果的可复制性，成功复制了六个流行的图推荐模型在几个常见和新数据集上的结果，并与传统协同过滤模型进行比较。

    

    图神经网络模型（GNNs）的成功显著推动了推荐系统的发展，通过将用户和物品有效地建模为一个二分图和无向图。然而，许多原始的基于图的作品通常在未验证其在具体配置下的有效性的情况下采用基线论文的结果。我们的工作解决了这个问题，着重关注结果的可复制性。我们提出了一种成功复制了六个流行且最新的图推荐模型（NGCF、DGCF、LightGCN、SGL、UltraGCN和GFCF）在三个常见基准数据集（Gowalla、Yelp 2018和亚马逊图书）上的结果的代码。此外，我们将这些图模型与在离线评估中表现良好的传统协同过滤模型进行了比较。此外，我们还扩展了对两个缺乏现有文献中已建立设置的新数据集（Allrecipes和BookCrossing）的研究。由于在这些数据集上的性能与以前的基准数据集不同，使得我们对图推荐模型性能的评估结果更加深入和全面。

    The success of graph neural network-based models (GNNs) has significantly advanced recommender systems by effectively modeling users and items as a bipartite, undirected graph. However, many original graph-based works often adopt results from baseline papers without verifying their validity for the specific configuration under analysis. Our work addresses this issue by focusing on the replicability of results. We present a code that successfully replicates results from six popular and recent graph recommendation models (NGCF, DGCF, LightGCN, SGL, UltraGCN, and GFCF) on three common benchmark datasets (Gowalla, Yelp 2018, and Amazon Book). Additionally, we compare these graph models with traditional collaborative filtering models that historically performed well in offline evaluations. Furthermore, we extend our study to two new datasets (Allrecipes and BookCrossing) that lack established setups in existing literature. As the performance on these datasets differs from the previous bench
    
[^7]: 通过基于规则的深度学习模型检测和分类物联网网络中的新型攻击和异常

    Detection and Classification of Novel Attacks and Anomaly in IoT Network using Rule based Deep Learning Model. (arXiv:2308.00005v1 [cs.IR])

    [http://arxiv.org/abs/2308.00005](http://arxiv.org/abs/2308.00005)

    本论文提出了一种基于规则的深度学习模型，用于检测和分类物联网网络中的新型攻击和异常。该模型能够有效改善传统机器学习模型在该领域中的表现。

    

    攻击者现在使用复杂技术，如多态性，以改变每次新攻击的攻击模式。因此，检测新型攻击对于网络安全专家和研究人员来说已经成为最大的挑战。最近，异常和混合方法被用于检测网络攻击。而检测新型攻击是广泛应用物联网技术的关键。新型攻击可以轻易规避现有基于签名的检测方法，甚至难以被发现多年。现有的机器学习模型也未能检测出这些攻击，并且误报率很高。在本文中，提出了一种基于规则的深度神经网络技术作为解决检测新型攻击问题的框架。实验结果表明，该模型较好地改进了相应的基准结果，包括CICIDS 2017数据集。

    Attackers are now using sophisticated techniques, like polymorphism, to change the attack pattern for each new attack. Thus, the detection of novel attacks has become the biggest challenge for cyber experts and researchers. Recently, anomaly and hybrid approaches are used for the detection of network attacks. Detecting novel attacks, on the other hand, is a key enabler for a wide range of IoT applications. Novel attacks can easily evade existing signature-based detection methods and are extremely difficult to detect, even going undetected for years. Existing machine learning models have also failed to detect the attack and have a high rate of false positives. In this paper, a rule-based deep neural network technique has been proposed as a framework for addressing the problem of detecting novel attacks. The designed framework significantly improves respective benchmark results, including the CICIDS 2017 dataset. The experimental results show that the proposed model keeps a good balance 
    
[^8]: Jina Embeddings:一种新颖的高性能句子嵌入模型

    Jina Embeddings: A Novel Set of High-Performance Sentence Embedding Models. (arXiv:2307.11224v1 [cs.CL])

    [http://arxiv.org/abs/2307.11224](http://arxiv.org/abs/2307.11224)

    Jina Embeddings是一组高性能的句子嵌入模型，能够捕捉文本的语义本质。该论文详细介绍了Jina Embeddings的开发过程，并通过性能评估验证了其优越性能。

    

    Jina Embeddings由一组高性能的句子嵌入模型组成，能够将各种文本输入转化为数值表示，从而捕捉文本的语义本质。虽然这些模型并非专门设计用于文本生成，但在密集检索和语义文本相似性等应用中表现出色。本文详细介绍了Jina Embeddings的开发过程，从创建高质量的成对和三元数据集开始。它强调了数据清理在数据集准备中的关键作用，并对模型训练过程进行了深入探讨，最后利用Massive Textual Embedding Benchmark（MTEB）进行了全面的性能评估。

    Jina Embeddings constitutes a set of high-performance sentence embedding models adept at translating various textual inputs into numerical representations, thereby capturing the semantic essence of the text. While these models are not exclusively designed for text generation, they excel in applications such as dense retrieval and semantic textual similarity. This paper details the development of Jina Embeddings, starting with the creation of a high-quality pairwise and triplet dataset. It underlines the crucial role of data cleaning in dataset preparation, gives in-depth insights into the model training process, and concludes with a comprehensive performance evaluation using the Massive Textual Embedding Benchmark (MTEB).
    
[^9]: 上下文检索增强的语言模型

    In-Context Retrieval-Augmented Language Models. (arXiv:2302.00083v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.00083](http://arxiv.org/abs/2302.00083)

    本研究提出了一种上下文检索增强的语言模型（In-Context RALM）方法，通过将相关文件作为输入的一部分，无需对语言模型进行进一步的训练即可显著提高语言建模性能和源归因能力，并且相对于现有的RALM方法，它具有更简单的部署过程。

    

    检索增强的语言模型(RALM)方法在生成过程中，通过将相关文件从语料库中检索出来与语言模型(LM)进行协同，已被证明可以显著提高语言建模性能。此外，它们还可以缓解事实不准确的文本生成问题，并提供自然的源归因机制。现有的RALM方法着重于修改LM架构以便于整合外部信息，从而大大增加了部署的复杂性。本文提出了一种简单的替代方法，称为上下文RALM：保持LM架构不变，并在输入中添加检索到的文件，无需对LM进行任何进一步的训练。我们展示了基于现成的通用检索器的上下文RALM在模型大小和不同语料库中能够提供出人意料的大幅度的LM增益。我们还证明，文件检索和排名机制可以针对RALM设置进行专门优化。

    Retrieval-Augmented Language Modeling (RALM) methods, which condition a language model (LM) on relevant documents from a grounding corpus during generation, were shown to significantly improve language modeling performance. In addition, they can mitigate the problem of factually inaccurate text generation and provide natural source attribution mechanism. Existing RALM approaches focus on modifying the LM architecture in order to facilitate the incorporation of external information, significantly complicating deployment. This paper considers a simple alternative, which we dub In-Context RALM: leaving the LM architecture unchanged and prepending grounding documents to the input, without any further training of the LM. We show that In-Context RALM that builds on off-the-shelf general purpose retrievers provides surprisingly large LM gains across model sizes and diverse corpora. We also demonstrate that the document retrieval and ranking mechanism can be specialized to the RALM setting to 
    

