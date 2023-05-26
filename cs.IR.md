# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey on Asking Clarification Questions Datasets in Conversational Systems.](http://arxiv.org/abs/2305.15933) | 本文对询问澄清问题（ACQ）的相关研究进行了全面分析，并提供了公开可用数据集的详细比较和多个ACQ相关任务的基准，旨在协助ACQ技术的发展。 |
| [^2] | [Enhancing the Ranking Context of Dense Retrieval Methods through Reciprocal Nearest Neighbors.](http://arxiv.org/abs/2305.15720) | 为了解决稀疏标注在稠密检索模型训练中的问题，我们提出了基于证据的标签平滑方法，并且引入了逆向最近邻相似度度量方法来提高相关性估计的准确性。 |
| [^3] | [BookGPT: A General Framework for Book Recommendation Empowered by Large Language Model.](http://arxiv.org/abs/2305.15673) | 本文介绍了一种基于大型语言模型的通用图书推荐框架BookGPT，通过将生成式预训练变换器技术应用于图书推荐场景中的三种任务，即图书评分推荐、用户评分推荐和图书摘要推荐，实现了对图书推荐的有力改进。 |
| [^4] | [ConvGQR: Generative Query Reformulation for Conversational Search.](http://arxiv.org/abs/2305.15645) | 本文提出了一种新的面向会话搜索的ConvGQR框架，通过结合预训练语言模型来重新构造查询，从而提供更好的搜索查询。 |
| [^5] | [Text-Augmented Open Knowledge Graph Completion via Pre-Trained Language Models.](http://arxiv.org/abs/2305.15597) | TAGREAL是一种可自动生成高质量查询提示信息，从大型文本语料库中检索支持信息以从PLM中探测知识的方法，用于开放知识图谱补全中，在两个基准数据集上取得了最先进的表现，并且即使在有限的训练数据情况下，仍然具有突出的性能。 |
| [^6] | [Representation Online Matters: Practical End-to-End Diversification in Search and Recommender Systems.](http://arxiv.org/abs/2305.15534) | 为了改善搜索和推荐系统中的代表性，我们提出了一种端到端的多样化方法，并在Pinterest平台上实验和部署了可扩展的多样化机制，以改善美容和时尚类别中不同肤色的代表性。 |
| [^7] | [Large Language Models for User Interest Journeys.](http://arxiv.org/abs/2305.15498) | 该论文提出了使用大型语言模型(LLMs)对用户兴趣进行建模的方法，并通过定义兴趣旅程，提出了一种模型旨在提高推荐的质量，并提供了可解释性和新颖性。 |
| [^8] | [Exploring and Exploiting Data Heterogeneity in Recommendation.](http://arxiv.org/abs/2305.15431) | 本文探讨了推荐系统中数据的异质性对模型性能的影响，提出了一种通过聚类和迁移学习的方法，很好地应对了异质性问题，实验结果表明其优于现有基准方法。 |
| [^9] | [Integrating Item Relevance in Training Loss for Sequential Recommender Systems.](http://arxiv.org/abs/2305.10824) | 本文提出了一种融合项目相关性的新型训练损失函数，用于提高序列推荐系统对噪声的鲁棒性和性能。 |
| [^10] | [Complex Logical Reasoning over Knowledge Graphs using Large Language Models.](http://arxiv.org/abs/2305.01157) | 本文提出了一种使用大型语言模型的解耦方法，将复杂的知识图谱推理形式化为上下文知识图搜索和抽象逻辑查询推理的组合，与现有方法相比，它在多个逻辑查询结构的标准基准数据集上都表现出更好的性能，并且在更高复杂性的查询中获得了显着的性能提升。 |
| [^11] | [Aggretriever: A Simple Approach to Aggregate Textual Representations for Robust Dense Passage Retrieval.](http://arxiv.org/abs/2208.00511) | 这项工作提出了一种简单的方法，将预训练语言模型中的知识充分应用于密集式段落检索，称为Aggretriever，通过将上下文化的token嵌入聚合到密集向量中，相对于以前需要采用计算量昂贵的技术进行训练的DPR模型，Aggretriever不需引入实质性的训练开销，能显著提高在域内和零-shot评估中有效性。 |
| [^12] | [A Computational Inflection for Scientific Discovery.](http://arxiv.org/abs/2205.02007) | 本文介绍了一种计算变革的框架，利用最新的人工智能技术来增强科学发现和交流。这个框架有很多应用场景，作者提供了一个原型系统的初始实现，并探讨了未来研究和发展方向。 |
| [^13] | [Atrapos: Real-time Evaluation of Metapath Query Workloads.](http://arxiv.org/abs/2201.04058) | ATRapos是一种实时评估元路径查询工作负载的方法，它利用了高效稀疏矩阵乘法和中间结果缓存的组合，在使用定制的数据结构来检测查询之间的频繁子元路径来选择要缓存和重用的中间结果。实验结果表明，ATRapos可以加速探索性数据分析。 |

# 详细

[^1]: 对话系统中询问澄清问题数据集的综述

    A Survey on Asking Clarification Questions Datasets in Conversational Systems. (arXiv:2305.15933v1 [cs.IR])

    [http://arxiv.org/abs/2305.15933](http://arxiv.org/abs/2305.15933)

    本文对询问澄清问题（ACQ）的相关研究进行了全面分析，并提供了公开可用数据集的详细比较和多个ACQ相关任务的基准，旨在协助ACQ技术的发展。

    

    理解用户真实需求对于对话系统尤为重要，特别是在对话中用户提供的信息有限的情况下。因此，在这样的领域中，询问澄清问题（ACQ）以揭示用户意图为关键任务。然而，现有ACQ研究的关键限制是它们的不可比性，来自数据的不一致使用，不同的实验设置和评估策略。因此，本文为协助ACQ技术的发展，全面分析了当前ACQ研究状态，提供了公开可用数据集的详细比较，并讨论了应用评估指标以及多个与ACQ相关任务的基准。特别是，在对ACQ任务进行彻底分析后，我们讨论了一些相应的研究方向，以调查ACQ以及对话系统的发展。

    The ability to understand a user's underlying needs is critical for conversational systems, especially with limited input from users in a conversation. Thus, in such a domain, Asking Clarification Questions (ACQs) to reveal users' true intent from their queries or utterances arise as an essential task. However, it is noticeable that a key limitation of the existing ACQs studies is their incomparability, from inconsistent use of data, distinct experimental setups and evaluation strategies. Therefore, in this paper, to assist the development of ACQs techniques, we comprehensively analyse the current ACQs research status, which offers a detailed comparison of publicly available datasets, and discusses the applied evaluation metrics, joined with benchmarks for multiple ACQs-related tasks. In particular, given a thorough analysis of the ACQs task, we discuss a number of corresponding research directions for the investigation of ACQs as well as the development of conversational systems.
    
[^2]: 通过逆向最近邻提升稠密检索方法的排名上下文质量

    Enhancing the Ranking Context of Dense Retrieval Methods through Reciprocal Nearest Neighbors. (arXiv:2305.15720v1 [cs.IR])

    [http://arxiv.org/abs/2305.15720](http://arxiv.org/abs/2305.15720)

    为了解决稀疏标注在稠密检索模型训练中的问题，我们提出了基于证据的标签平滑方法，并且引入了逆向最近邻相似度度量方法来提高相关性估计的准确性。

    

    稀疏标注给稠密检索模型训练带来了持久的挑战，例如虚假负样本问题，即未标记的相关文档被错误地用作负样本，扭曲了训练信号。为了缓解这个问题，我们介绍了一种称为基于证据的标签平滑的计算方法，这是一种计算效率高的方法，可以避免惩罚模型将高相关性赋予虚假负样本。为了在给定查询的排名上下文中计算候选文档的目标相关性分布，与基本事实最相似的候选者被赋予非零相关概率，该概率基于它们与基本事实文档的相似度程度。作为相关性估计，我们利用了一种基于逆向最近邻的改进相似度度量，该度量还可单独用于后处理中重新排名候选者。通过在两个大规模的自适应文本检索数据集上进行广泛的实验，我们展示了本方法的优越性。

    Sparse annotation poses persistent challenges to training dense retrieval models, such as the problem of false negatives, i.e. unlabeled relevant documents that are spuriously used as negatives in contrastive learning, distorting the training signal. To alleviate this problem, we introduce evidence-based label smoothing, a computationally efficient method that prevents penalizing the model for assigning high relevance to false negatives. To compute the target relevance distribution over candidate documents within the ranking context of a given query, candidates most similar to the ground truth are assigned a non-zero relevance probability based on the degree of their similarity to the ground-truth document(s). As a relevance estimate we leverage an improved similarity metric based on reciprocal nearest neighbors, which can also be used independently to rerank candidates in post-processing. Through extensive experiments on two large-scale ad hoc text retrieval datasets we demonstrate th
    
[^3]: BookGPT：一种基于大型语言模型的通用图书推荐框架

    BookGPT: A General Framework for Book Recommendation Empowered by Large Language Model. (arXiv:2305.15673v1 [cs.IR])

    [http://arxiv.org/abs/2305.15673](http://arxiv.org/abs/2305.15673)

    本文介绍了一种基于大型语言模型的通用图书推荐框架BookGPT，通过将生成式预训练变换器技术应用于图书推荐场景中的三种任务，即图书评分推荐、用户评分推荐和图书摘要推荐，实现了对图书推荐的有力改进。

    

    随着生成式预训练变换器（GPT）等大型语言模型技术的不断发展和变化，各个领域的许多经典场景重新展现出新的机遇。本文将ChatGPT作为建模对象，首次将LLM技术并入传统的图书资源理解和推荐场景中，并付诸实践。本文基于ChatGPT构建了类似于聊天机器人的图书推荐系统框架（BookGPT），试图将ChatGPT应用于三种典型任务的推荐建模：图书评分推荐，用户评分推荐和图书摘要推荐，探索LLM技术在图书推荐场景中的可行性。同时，本文根据不同的图书推荐任务评估方案和现有的经典推荐模型，讨论了BookGPT在图书推荐场景下的优缺点，并进行了一系列实证比较和分析，证明基于LLM技术的BookGPT框架可以为图书推荐领域带来显著的改进。

    With the continuous development and change exhibited by large language model (LLM) technology, represented by generative pretrained transformers (GPTs), many classic scenarios in various fields have re-emerged with new opportunities. This paper takes ChatGPT as the modeling object, incorporates LLM technology into the typical book resource understanding and recommendation scenario for the first time, and puts it into practice. By building a ChatGPT-like book recommendation system (BookGPT) framework based on ChatGPT, this paper attempts to apply ChatGPT to recommendation modeling for three typical tasks, book rating recommendation, user rating recommendation, and book summary recommendation, and explores the feasibility of LLM technology in book recommendation scenarios. At the same time, based on different evaluation schemes for book recommendation tasks and the existing classic recommendation models, this paper discusses the advantages and disadvantages of the BookGPT in book recomme
    
[^4]: ConvGQR：面向会话搜索的生成式查询重构

    ConvGQR: Generative Query Reformulation for Conversational Search. (arXiv:2305.15645v1 [cs.IR])

    [http://arxiv.org/abs/2305.15645](http://arxiv.org/abs/2305.15645)

    本文提出了一种新的面向会话搜索的ConvGQR框架，通过结合预训练语言模型来重新构造查询，从而提供更好的搜索查询。

    

    在会话搜索中，用户当前搜索意图依赖于先前的对话历史。从整个对话上下文中确定一个良好的搜索查询是具有挑战性的。为避免查询编码器的昂贵重新训练，大部分现有方法尝试学习一个重写模型，通过模仿手动查询重写来去除当前查询的上下文。然而，手动重写的查询并不总是最好的搜索查询。训练重写模型会限制模型产生良好搜索查询的能力。本文提出一种新的框架ConvGQR，基于预训练语言模型（PLM），一个用于查询重写，另一个用于生成潜在答案，以重新构造会话查询。通过结合两者，ConvGQR可以提供更好的搜索查询。此外，为了将查询重构与检索性能联系起来，我们提出了一种基于特征选择的相似度分数模型，用于验证ConvGQR的有效性。

    In conversational search, the user's real search intent for the current turn is dependent on the previous conversation history. It is challenging to determine a good search query from the whole conversation context. To avoid the expensive re-training of the query encoder, most existing methods try to learn a rewriting model to de-contextualize the current query by mimicking the manual query rewriting. However, manually rewritten queries are not always the best search queries. Training a rewriting model on them would limit the model's ability to produce good search queries. Another useful hint is the potential answer to the question. In this paper, we propose ConvGQR, a new framework to reformulate conversational queries based on generative pre-trained language models (PLMs), one for query rewriting and another for generating potential answers. By combining both, ConvGQR can produce better search queries. In addition, to relate query reformulation to retrieval performance, we propose a 
    
[^5]: 基于预训练语言模型的文本增强开放知识图谱补全

    Text-Augmented Open Knowledge Graph Completion via Pre-Trained Language Models. (arXiv:2305.15597v1 [cs.CL])

    [http://arxiv.org/abs/2305.15597](http://arxiv.org/abs/2305.15597)

    TAGREAL是一种可自动生成高质量查询提示信息，从大型文本语料库中检索支持信息以从PLM中探测知识的方法，用于开放知识图谱补全中，在两个基准数据集上取得了最先进的表现，并且即使在有限的训练数据情况下，仍然具有突出的性能。

    

    开放知识图谱补全的任务是从已知事实中提取新的发现。现有的增强知识图谱补全的方法要么需要事实三元组以扩大图推理空间，要么需要手动设计提示信息以从预训练语言模型中提取知识，这些方法性能有限，需要专家昂贵的工作。为此，我们提出了TAGREAL，它自动生成高质量的查询提示信息，并从大型文本语料库中检索支持信息以从PLM中探测知识以完成知识图谱补全。结果表明，TAGREAL在两个基准数据集上实现了最先进的性能。我们发现，即使是在有限的训练数据情况下，TAGREAL的性能仍然非常突出，超过了现有的基于嵌入、基于图和基于PLM的方法。

    The mission of open knowledge graph (KG) completion is to draw new findings from known facts. Existing works that augment KG completion require either (1) factual triples to enlarge the graph reasoning space or (2) manually designed prompts to extract knowledge from a pre-trained language model (PLM), exhibiting limited performance and requiring expensive efforts from experts. To this end, we propose TAGREAL that automatically generates quality query prompts and retrieves support information from large text corpora to probe knowledge from PLM for KG completion. The results show that TAGREAL achieves state-of-the-art performance on two benchmark datasets. We find that TAGREAL has superb performance even with limited training data, outperforming existing embedding-based, graph-based, and PLM-based methods.
    
[^6]: 在搜索和推荐系统中，在线表示很重要：实用的端到端多样化方法。

    Representation Online Matters: Practical End-to-End Diversification in Search and Recommender Systems. (arXiv:2305.15534v1 [cs.IR])

    [http://arxiv.org/abs/2305.15534](http://arxiv.org/abs/2305.15534)

    为了改善搜索和推荐系统中的代表性，我们提出了一种端到端的多样化方法，并在Pinterest平台上实验和部署了可扩展的多样化机制，以改善美容和时尚类别中不同肤色的代表性。

    

    随着在线平台在各个人口统计学中的使用不断增长，用户经常表达希望在内容中感受到自己的代表性。为了改善搜索结果和推荐中的代表性，我们引入了端到端的多样化方法，确保多样化内容在这些系统的各个阶段中流动，从检索到排序。我们在多个Pinterest平台的生产界面中开发、实验和部署可扩展的多样化机制，包括搜索、相关产品和新用户主页，以改善美容和时尚内容中不同肤色的代表性。生产系统中的多样化包括三个组成部分：确定会触发多样化的请求，在检索阶段确保从大型内容语料库中检索到多样化的内容，最后，在排名阶段以自我调整的方式平衡多样性和效用的权衡。我们的方法从使用Strong-O开始。

    As the use of online platforms continues to grow across all demographics, users often express a desire to feel represented in the content. To improve representation in search results and recommendations, we introduce end-to-end diversification, ensuring that diverse content flows throughout the various stages of these systems, from retrieval to ranking. We develop, experiment, and deploy scalable diversification mechanisms in multiple production surfaces on the Pinterest platform, including Search, Related Products, and New User Homefeed, to improve the representation of different skin tones in beauty and fashion content. Diversification in production systems includes three components: identifying requests that will trigger diversification, ensuring diverse content is retrieved from the large content corpus during the retrieval stage, and finally, balancing the diversity-utility trade-off in a self-adjusting manner in the ranking stage. Our approaches, which evolved from using Strong-O
    
[^7]: 用户兴趣旅程的大型语言模型

    Large Language Models for User Interest Journeys. (arXiv:2305.15498v1 [cs.CL])

    [http://arxiv.org/abs/2305.15498](http://arxiv.org/abs/2305.15498)

    该论文提出了使用大型语言模型(LLMs)对用户兴趣进行建模的方法，并通过定义兴趣旅程，提出了一种模型旨在提高推荐的质量，并提供了可解释性和新颖性。

    

    大型语言模型（LLMs）已经展示出在自然语言理解和生成方面的令人瞩目能力。然而，它们在更深入地理解用户和改善个性化推荐平台体验方面的潜力还远未被发挥。本文旨在填补这一空白。我们提出了使用LLMs对用户兴趣进行建模的方法，并定义了兴趣旅程作为用户基于他们的活动而遍历过的兴趣状态序列。我们的实验证明，相对于传统的用户表示方法，我们提出的方法可以提高推荐的质量，并且生成的兴趣旅程为推荐过程提供了可解释性和新颖性。

    Large language models (LLMs) have shown impressive capabilities in natural language understanding and generation. Their potential for deeper user understanding and improved personalized user experience on recommendation platforms is, however, largely untapped. This paper aims to address this gap. Recommender systems today capture users' interests through encoding their historical activities on the platforms. The generated user representations are hard to examine or interpret. On the other hand, if we were to ask people about interests they pursue in their life, they might talk about their hobbies, like I just started learning the ukulele, or their relaxation routines, e.g., I like to watch Saturday Night Live, or I want to plant a vertical garden. We argue, and demonstrate through extensive experiments, that LLMs as foundation models can reason through user activities, and describe their interests in nuanced and interesting ways, similar to how a human would.  We define interest journe
    
[^8]: 探索和利用推荐系统中的数据异质性

    Exploring and Exploiting Data Heterogeneity in Recommendation. (arXiv:2305.15431v1 [cs.IR])

    [http://arxiv.org/abs/2305.15431](http://arxiv.org/abs/2305.15431)

    本文探讨了推荐系统中数据的异质性对模型性能的影响，提出了一种通过聚类和迁移学习的方法，很好地应对了异质性问题，实验结果表明其优于现有基准方法。

    

    大量的数据是数据驱动推荐模型的基础。数据异质性是大数据的内在特性，在现实推荐系统中广泛存在。它反映了子人口群体之间属性的差异。忽略推荐数据的异质性可能会限制推荐模型的性能，损害子人口的鲁棒性，并使模型误导数据偏见。然而，数据异质性在推荐界并没有受到足够的关注。因此，它激发我们充分探索和利用异质性来解决上述问题并辅助数据分析。在本文中，我们着重探讨了推荐数据中两类典型的异质性，即预测机制和协变量分布的异质性，并提出了一种通过双层聚类方法探索异质性的算法。此外，通过迁移学习机制利用了挖掘出来的异质性。在基准数据集上进行的实验表明，我们的方法优于现有的基准方法。

    Massive amounts of data are the foundation of data-driven recommendation models. As an inherent nature of big data, data heterogeneity widely exists in real-world recommendation systems. It reflects the differences in the properties among sub-populations. Ignoring the heterogeneity in recommendation data could limit the performance of recommendation models, hurt the sub-populational robustness, and make the models misled by biases. However, data heterogeneity has not attracted substantial attention in the recommendation community. Therefore, it inspires us to adequately explore and exploit heterogeneity for solving the above problems and assisting data analysis. In this work, we focus on exploring two representative categories of heterogeneity in recommendation data that is the heterogeneity of prediction mechanism and covariate distribution and propose an algorithm that explores the heterogeneity through a bilevel clustering method. Furthermore, the uncovered heterogeneity is exploite
    
[^9]: 融合项目相关性的序列推荐系统训练损失函数

    Integrating Item Relevance in Training Loss for Sequential Recommender Systems. (arXiv:2305.10824v1 [cs.IR])

    [http://arxiv.org/abs/2305.10824](http://arxiv.org/abs/2305.10824)

    本文提出了一种融合项目相关性的新型训练损失函数，用于提高序列推荐系统对噪声的鲁棒性和性能。

    

    序列推荐系统是一种受欢迎的推荐系统，它通过学习用户的历史数据来预测用户下一个可能与之交互的项目。然而，用户的交互可能会受到来自帐户共享、不一致的偏好或意外点击等噪声的影响。为了解决这个问题，我们（i）提出了一个考虑多个未来项目的新的评估协议，（ii）引入了一种新的关注相关性的损失函数，用于训练具有多个未来项目的序列推荐系统，以使其对噪声更加鲁棒。我们的关注相关性模型在传统评估协议中提高了NDCG@10约1.2%和HR约0.88%，而在新评估协议中，改进的NDCG@10约1.63%和HR约1.5%。

    Sequential Recommender Systems (SRSs) are a popular type of recommender system that learns from a user's history to predict the next item they are likely to interact with. However, user interactions can be affected by noise stemming from account sharing, inconsistent preferences, or accidental clicks. To address this issue, we (i) propose a new evaluation protocol that takes multiple future items into account and (ii) introduce a novel relevance-aware loss function to train a SRS with multiple future items to make it more robust to noise. Our relevance-aware models obtain an improvement of ~1.2% of NDCG@10 and 0.88% in the traditional evaluation protocol, while in the new evaluation protocol, the improvement is ~1.63% of NDCG@10 and ~1.5% of HR w.r.t the best performing models.
    
[^10]: 使用大型语言模型在知识图谱上进行复杂逻辑推理

    Complex Logical Reasoning over Knowledge Graphs using Large Language Models. (arXiv:2305.01157v1 [cs.LO])

    [http://arxiv.org/abs/2305.01157](http://arxiv.org/abs/2305.01157)

    本文提出了一种使用大型语言模型的解耦方法，将复杂的知识图谱推理形式化为上下文知识图搜索和抽象逻辑查询推理的组合，与现有方法相比，它在多个逻辑查询结构的标准基准数据集上都表现出更好的性能，并且在更高复杂性的查询中获得了显着的性能提升。

    

    在知识图谱上进行推理是一项具有挑战性的任务，它需要对实体之间的复杂关系以及它们之间的基础逻辑进行深入理解。当前的方法依赖于学习几何来嵌入实体的向量空间进行逻辑查询操作，但是它们在复杂查询和特定数据集表示方面表现不佳。本文提出了一种新颖的解耦方法，称为基于语言引导的知识图谱抽象推理（LARK），将复杂的知识图谱推理形式化为上下文知识图搜索和抽象逻辑查询推理的组合，以分别利用图形提取算法和大型语言模型的优势。我们的实验表明，所提出的方法在多个逻辑查询结构的标准基准数据集上优于现有的知识图谱推理方法，在更高复杂性的查询中获得了显着的性能提升。

    Reasoning over knowledge graphs (KGs) is a challenging task that requires a deep understanding of the complex relationships between entities and the underlying logic of their relations. Current approaches rely on learning geometries to embed entities in vector space for logical query operations, but they suffer from subpar performance on complex queries and dataset-specific representations. In this paper, we propose a novel decoupled approach, Language-guided Abstract Reasoning over Knowledge graphs (LARK), that formulates complex KG reasoning as a combination of contextual KG search and abstract logical query reasoning, to leverage the strengths of graph extraction algorithms and large language models (LLM), respectively. Our experiments demonstrate that the proposed approach outperforms state-of-the-art KG reasoning methods on standard benchmark datasets across several logical query constructs, with significant performance gain for queries of higher complexity. Furthermore, we show t
    
[^11]: Aggretriever：一种简单的聚合文本表示方法，用于强大的密集式段落检索

    Aggretriever: A Simple Approach to Aggregate Textual Representations for Robust Dense Passage Retrieval. (arXiv:2208.00511v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2208.00511](http://arxiv.org/abs/2208.00511)

    这项工作提出了一种简单的方法，将预训练语言模型中的知识充分应用于密集式段落检索，称为Aggretriever，通过将上下文化的token嵌入聚合到密集向量中，相对于以前需要采用计算量昂贵的技术进行训练的DPR模型，Aggretriever不需引入实质性的训练开销，能显著提高在域内和零-shot评估中有效性。

    

    预训练语言模型在很多知识密集型NLP任务中取得了成功。然而，最近的研究表明，如BERT这样的模型在将文本信息聚合成[CLS]向量以进行密集式段落检索（DPR）时并不是“结构上准备好的”。这种“准备不足”是由语言模型预训练和DPR微调之间的差距造成的。以前的解决方案要求使用计算量昂贵的技术，如硬负采样、交叉编码器蒸馏和更进一步的预训练来学习强大的DPR模型。在这项工作中，我们建议通过聚合上下文化的token嵌入到一个密集向量中，充分利用预训练语言模型在DPR中的知识，我们将其称为agg*。通过将来自[CLS] token和agg*的向量进行串联，我们的Aggretriever模型在不引入实质性的训练开销的情况下，显著提高了密集式检索模型在域内和零-shot评估中的有效性。可在h上获取代码。

    Pre-trained language models have been successful in many knowledge-intensive NLP tasks. However, recent work has shown that models such as BERT are not ``structurally ready'' to aggregate textual information into a [CLS] vector for dense passage retrieval (DPR). This ``lack of readiness'' results from the gap between language model pre-training and DPR fine-tuning. Previous solutions call for computationally expensive techniques such as hard negative mining, cross-encoder distillation, and further pre-training to learn a robust DPR model. In this work, we instead propose to fully exploit knowledge in a pre-trained language model for DPR by aggregating the contextualized token embeddings into a dense vector, which we call agg*. By concatenating vectors from the [CLS] token and agg*, our Aggretriever model substantially improves the effectiveness of dense retrieval models on both in-domain and zero-shot evaluations without introducing substantial training overhead. Code is available at h
    
[^12]: 科学发现的计算变革

    A Computational Inflection for Scientific Discovery. (arXiv:2205.02007v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.02007](http://arxiv.org/abs/2205.02007)

    本文介绍了一种计算变革的框架，利用最新的人工智能技术来增强科学发现和交流。这个框架有很多应用场景，作者提供了一个原型系统的初始实现，并探讨了未来研究和发展方向。

    

    我们正站在科学发现轨迹上一个重要的拐点上。随着社会的快速数字化转型，人类的科学知识和交流也在数字化的形式下不断增长。我们现在阅读和撰写的论文、预印本、书籍、代码、数据集、会议演示稿以及社交网络和协作和沟通平台上的交互等，大多已经以数字化的方式记录。这种转变导致了大量信息的创造和增长——其中很多已经可供公众获取——为分析和利用其的计算模型和系统开启了令人激动的机遇。与此同时，数据处理能力的指数增长推动了人工智能的显著进步，包括能够从非结构化文本中学习强大表示的大型神经语言模型。然而，需要进行重大改变，以在科学知识和交流的更大生态系统中有效整合这些进展，创建一种新的统一的科学交流范式。本文介绍了一种科学发现的计算变革——利用人工智能的最新进展，增强科学发现和交流的统一框架。我们展示了这个框架的潜力，提供了一个原型系统的初始实现，并讨论了未来的研究和发展方向。

    We stand at the foot of a significant inflection in the trajectory of scientific discovery. As society continues on its fast-paced digital transformation, so does humankind's collective scientific knowledge and discourse. We now read and write papers in digitized form, and a great deal of the formal and informal processes of science are captured digitally -including papers, preprints and books, code and datasets, conference presentations, and interactions in social networks and collaboration and communication platforms. The transition has led to the creation and growth of a tremendous amount of information -- much of which is available for public access -- opening exciting opportunities for computational models and systems that analyze and harness it. In parallel, exponential growth in data processing power has fueled remarkable advances in artificial intelligence, including large neural language models capable of learning powerful representations from unstructured text. Dramatic cha
    
[^13]: Atrapos: 元路径查询工作负载的实时评估

    Atrapos: Real-time Evaluation of Metapath Query Workloads. (arXiv:2201.04058v2 [cs.DB] UPDATED)

    [http://arxiv.org/abs/2201.04058](http://arxiv.org/abs/2201.04058)

    ATRapos是一种实时评估元路径查询工作负载的方法，它利用了高效稀疏矩阵乘法和中间结果缓存的组合，在使用定制的数据结构来检测查询之间的频繁子元路径来选择要缓存和重用的中间结果。实验结果表明，ATRapos可以加速探索性数据分析。

    

    异构信息网络（HIN）表示不同类型的实体及其之间的关系。探索、分析和提取这样的网络中的知识依赖于元路径查询，这些查询识别由多样的语义关系连接的实体对。然而，对于大规模的网络，实时评估元路径查询工作负载的计算成本非常高，当前的方法也没有利用查询之间的相互关系。在本文中，我们提出了一种名为 ATRAPOS 的新方法，用于实时评估元路径查询工作负载，该方法利用了高效稀疏矩阵乘法和中间结果缓存的组合。ATRAPOS 通过使用定制的数据结构——重叠树和相关的缓存策略，在实时检测到工作负载查询之间的频繁子元路径来选择要缓存和重用的中间结果。我们在真实数据上的实验研究表明，ATRAPOS 加速了探索性数据分析。

    Heterogeneous information networks (HINs) represent different types of entities and relationships between them. Exploring, analysing, and extracting knowledge from such networks relies on metapath queries that identify pairs of entities connected by relationships of diverse semantics. While the real-time evaluation of metapath query workloads on large, web-scale HINs is highly demanding in computational cost, current approaches do not exploit interrelationships among the queries. In this paper, we present ATRAPOS, a new approach for the real-time evaluation of metapath query workloads that leverages a combination of efficient sparse matrix multiplication and intermediate result caching. ATRAPOS selects intermediate results to cache and reuse by detecting frequent sub-metapaths among workload queries in real time, using a tailor-made data structure, the Overlap Tree, and an associated caching policy. Our experimental study on real data shows that ATRAPOS accelerates exploratory data ana
    

