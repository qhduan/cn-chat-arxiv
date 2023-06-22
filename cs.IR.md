# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Knowledge-based Multimodal Music Similarity.](http://arxiv.org/abs/2306.12249) | 本文研究了基于知识的多模态音乐相似性，旨在开发一个完全可解释和可解释的系统，为最终用户提供更多对音乐相似性和分类系统的控制和理解。 |
| [^2] | [CompMix: A Benchmark for Heterogeneous Question Answering.](http://arxiv.org/abs/2306.12235) | CompMix是一个异构问答系统的基准测试，有多个信息源和复杂意图，旨在提供公平的评估QA系统的能力。 |
| [^3] | [STAN: Stage-Adaptive Network for Multi-Task Recommendation by Learning User Lifecycle-Based Representation.](http://arxiv.org/abs/2306.12232) | 本论文提出基于学习用户生命周期表示的多任务推荐阶段自适应网络（STAN）框架，利用用户生命周期阶段来增强多任务学习的性能，在三个真实世界的数据集上实验结果表明STAN可以有效提高推荐性能和用户保留率。 |
| [^4] | [Post-hoc Selection of Pareto-Optimal Solutions in Search and Recommendation.](http://arxiv.org/abs/2306.12165) | 本文提出了一种名为“从乌托邦的人口距离”（PDU）的后选择策略，用于确定和选择 Pareto-最优解中的最佳解。该方法分析点的分布，通过估计 PDU 分数的点的平均位置来确定最佳解的可能位置，实验结果表明 PDU 在准确性和稳定性方面表现优异。 |
| [^5] | [Visualizing Relation Between (De)Motivating Topics and Public Stance toward COVID-19 Vaccine.](http://arxiv.org/abs/2306.12118) | 研究了社交媒体上COVID-19话题对公众接种疫苗态度的影响，提出了交互式可视化工具，可以分析话题共鸣和动力转移，增加研究与公众的透明度。 |
| [^6] | [Comparative analysis of various web crawler algorithms.](http://arxiv.org/abs/2306.12027) | 本文介绍了网络爬虫和网页排名算法在处理Web数据方面的重要性；在评估五种不同的爬取算法后，旨在确定最有效的爬取算法。 |
| [^7] | [Addressing the Rank Degeneration in Sequential Recommendation via Singular Spectrum Smoothing.](http://arxiv.org/abs/2306.11986) | 本文提出了一种通过奇异谱平滑算法缓解顺序推荐中序列与项目排名退化问题的方法，并提出了SSA指标来评估该问题的严重性。 |
| [^8] | [Sampling Individually-Fair Rankings that are Always Group Fair.](http://arxiv.org/abs/2306.11964) | 该论文提出了一种有效算法，从个体公平分布中采样排名，同时确保每个输出的排名都满足群体公平性约束。输出排名的期望效用至少是最优公平解的效用的$\alpha$倍，其中$\alpha$是一个量化公平约束紧度的参数。 |
| [^9] | [Multimodality Fusion for Smart Healthcare: a Journey from Data, Information, Knowledge to Wisdom.](http://arxiv.org/abs/2306.11963) | 本文综述了多模态医学数据融合在智慧医疗中的应用，提出了符合DIKW机制的通用融合框架，探讨了面临的挑战和未来的发展方向。 |
| [^10] | [Retrieval-Based Transformer for Table Augmentation.](http://arxiv.org/abs/2306.11843) | 本文提出了一种自动数据处理的新方法，其中使用基于检索的Transformer模型来解决表格增强任务，并采用自学习策略来训练模型以重构原始值或标题，以便减轻数据分析师在数据处理中的工作量。 |

# 详细

[^1]: 基于知识的多模态音乐相似性研究

    Knowledge-based Multimodal Music Similarity. (arXiv:2306.12249v1 [cs.SD])

    [http://arxiv.org/abs/2306.12249](http://arxiv.org/abs/2306.12249)

    本文研究了基于知识的多模态音乐相似性，旨在开发一个完全可解释和可解释的系统，为最终用户提供更多对音乐相似性和分类系统的控制和理解。

    

    音乐相似性是音乐检索、推荐系统和音乐分析的重要方面。而且，对于音乐专家来说，相似性可以研究作曲家和历史时期之间的类比和影响。目前，针对音乐相似性的方法主要依赖符号内容，这可能成本高昂且不总是易于获得。相比之下，使用音频信号的方法通常无法提供有关观察到的相似性背后原因的任何见解。本研究通过使用符号和音频内容来研究音乐相似性，解决了当前方法的局限性。本研究的目的是开发一个完全可解释和可解释的系统，可以为最终用户提供更多对音乐相似性和分类系统的控制和理解。

    Music similarity is an essential aspect of music retrieval, recommendation systems, and music analysis. Moreover, similarity is of vital interest for music experts, as it allows studying analogies and influences among composers and historical periods. Current approaches to musical similarity rely mainly on symbolic content, which can be expensive to produce and is not always readily available. Conversely, approaches using audio signals typically fail to provide any insight about the reasons behind the observed similarity. This research addresses the limitations of current approaches by focusing on the study of musical similarity using both symbolic and audio content. The aim of this research is to develop a fully explainable and interpretable system that can provide end-users with more control and understanding of music similarity and classification systems.
    
[^2]: CompMix: 一种异构问答系统的基准测试

    CompMix: A Benchmark for Heterogeneous Question Answering. (arXiv:2306.12235v1 [cs.IR])

    [http://arxiv.org/abs/2306.12235](http://arxiv.org/abs/2306.12235)

    CompMix是一个异构问答系统的基准测试，有多个信息源和复杂意图，旨在提供公平的评估QA系统的能力。

    

    事实为中心的问答系统经常需要访问多种异构信息源。通过共同考虑多个信息源，如知识库、文本收集和来自网络的表格，问答系统可以增强其答案覆盖范围和可信度。然而，现有的 QA 基准测试大多是为了构建单一的知识资源而设计的。这限制了这些基准测试的能力，无法公平地评估可以利用多个信息库的 QA 系统。为了弥补这一差距，我们发布了 CompMix，这是一种由众包问答构建的基准测试，自然地要求集成多种输入源。CompMix 共有 9,410 个问题，并具有多个复杂意图，如连接和时间条件。在 CompMix 上评估一系列 QA 系统强调了进一步研究利用异构信息源的必要性。

    Fact-centric question answering (QA) often requires access to multiple, heterogeneous, information sources. By jointly considering several sources like a knowledge base (KB), a text collection, and tables from the web, QA systems can enhance their answer coverage and confidence. However, existing QA benchmarks are mostly constructed with a single source of knowledge in mind. This limits capabilities of these benchmarks to fairly evaluate QA systems that can tap into more than one information repository. To bridge this gap, we release CompMix, a crowdsourced QA benchmark which naturally demands the integration of a mixture of input sources. CompMix has a total of 9,410 questions, and features several complex intents like joins and temporal conditions. Evaluation of a range of QA systems on CompMix highlights the need for further research on leveraging information from heterogeneous sources.
    
[^3]: 基于学习用户生命周期表示的多任务推荐阶段自适应网络（STAN）

    STAN: Stage-Adaptive Network for Multi-Task Recommendation by Learning User Lifecycle-Based Representation. (arXiv:2306.12232v1 [cs.IR])

    [http://arxiv.org/abs/2306.12232](http://arxiv.org/abs/2306.12232)

    本论文提出基于学习用户生命周期表示的多任务推荐阶段自适应网络（STAN）框架，利用用户生命周期阶段来增强多任务学习的性能，在三个真实世界的数据集上实验结果表明STAN可以有效提高推荐性能和用户保留率。

    

    推荐系统在许多在线平台上起着至关重要的作用，其主要目标是满足和留住用户。由于直接优化用户保留非常具有挑战性，因此通常会采用多种评估指标。现有方法通常将这些评估指标的优化形式化为多任务学习问题，但常常忽略了用户对不同任务的偏好是个性化的并且随时间变化的事实。确定和跟踪用户偏好的演变可以改善用户保留。为了解决这个问题，我们引入了“用户生命周期”的概念，由多个阶段组成，其特征是用户对不同任务偏好的变化。我们提出了一种新颖的阶段自适应网络（STAN）框架，用于建模用户生命周期阶段。STAN首先基于学习到的用户偏好来识别潜在的用户生命周期阶段，然后利用阶段表示增强多任务学习的性能。我们在三个真实世界的数据集上的实验结果证明了STAN在提高推荐性能和用户保留方面的有效性。

    Recommendation systems play a vital role in many online platforms, with their primary objective being to satisfy and retain users. As directly optimizing user retention is challenging, multiple evaluation metrics are often employed. Existing methods generally formulate the optimization of these evaluation metrics as a multitask learning problem, but often overlook the fact that user preferences for different tasks are personalized and change over time. Identifying and tracking the evolution of user preferences can lead to better user retention. To address this issue, we introduce the concept of "user lifecycle", consisting of multiple stages characterized by users' varying preferences for different tasks. We propose a novel Stage-Adaptive Network (STAN) framework for modeling user lifecycle stages. STAN first identifies latent user lifecycle stages based on learned user preferences, and then employs the stage representation to enhance multi-task learning performance. Our experimental r
    
[^4]: 搜索和推荐中 Pareto-最优解后选择策略研究

    Post-hoc Selection of Pareto-Optimal Solutions in Search and Recommendation. (arXiv:2306.12165v1 [cs.IR])

    [http://arxiv.org/abs/2306.12165](http://arxiv.org/abs/2306.12165)

    本文提出了一种名为“从乌托邦的人口距离”（PDU）的后选择策略，用于确定和选择 Pareto-最优解中的最佳解。该方法分析点的分布，通过估计 PDU 分数的点的平均位置来确定最佳解的可能位置，实验结果表明 PDU 在准确性和稳定性方面表现优异。

    

    信息检索（IR）和推荐系统（RS）任务从基于单一度量计算最终结果的排名过渡为多目标问题。解决这些问题会得到一组 Pareto-最优解，称为 Pareto frontier，其中没有目标可以进一步改善而不损害其他目标。原则上，Pareto frontier 上的所有点都有可能代表着基于两个或多个度量相结合选择的最佳模型候选者。我们提出了一种名为“从乌托邦的人口距离”（PDU）的新颖后选择策略，采用理论上的正当化技术来确定和选择 Pareto-最优解中的最佳解。具体而言，PDU 通过研究每个点与其乌托邦点（目标的理想性能）之间的距离来分析点的分布。在一定阈值范围内，通过估计 PDU 分数的点的平均位置来确定最佳解的可能位置。我们在合成和真实数据集上评估 PDU 并与其他知名的选择策略进行比较，结果表明 PDU 在准确性和稳定性方面表现优异。

    Information Retrieval (IR) and Recommender Systems (RS) tasks are moving from computing a ranking of final results based on a single metric to multi-objective problems. Solving these problems leads to a set of Pareto-optimal solutions, known as Pareto frontier, in which no objective can be further improved without hurting the others. In principle, all the points on the Pareto frontier are potential candidates to represent the best model selected with respect to the combination of two, or more, metrics. To our knowledge, there are no well-recognized strategies to decide which point should be selected on the frontier. In this paper, we propose a novel, post-hoc, theoretically-justified technique, named "Population Distance from Utopia" (PDU), to identify and select the one-best Pareto-optimal solution from the frontier. In detail, PDU analyzes the distribution of the points by investigating how far each point is from its utopia point (the ideal performance for the objectives). The possib
    
[^5]: 可视化探究与COVID-19疫苗接种态度相关的讨论话题

    Visualizing Relation Between (De)Motivating Topics and Public Stance toward COVID-19 Vaccine. (arXiv:2306.12118v1 [cs.CY])

    [http://arxiv.org/abs/2306.12118](http://arxiv.org/abs/2306.12118)

    研究了社交媒体上COVID-19话题对公众接种疫苗态度的影响，提出了交互式可视化工具，可以分析话题共鸣和动力转移，增加研究与公众的透明度。

    

    社交媒体在当今通讯中起到了至关重要的作用，但误导和恶意评论很容易占据话题，引导公众舆论。在COVID-19疫情期间，我们看到了不实信息的影响，公共卫生官员在试图激励公众接种疫苗时遭到了重大抵制。为了应对当前和任何未来的紧急威胁，并激励公众朝着一个共同的目标前进，我们需要了解公众动力的转移以及哪些话题在普通民众中有共鸣。在本研究中，我们提出了一个交互式可视化工具，以检查和分析COVID-19疫情期间Twitter-sphere中的话题，并了解关键因素是什么导致公众对接种疫苗的态度转变。该工具可以轻松推广为任何情景的视觉分析工具，并增加社交媒体数据对研究人员和普通民众的透明度。

    While social media plays a vital role in communication nowadays, misinformation and trolls can easily take over the conversation and steer public opinion on these platforms. We saw the effect of misinformation during the {COVID-19} pandemic when public health officials faced significant push-back while trying to motivate the public to vaccinate. To tackle the current and any future threats in emergencies and motivate the public towards a common goal, it is essential to understand how public motivation shifts and which topics resonate among the general population. In this study, we proposed an interactive visualization tool to inspect and analyze the topics that resonated among Twitter-sphere during the {COVID-19} pandemic and understand the key factors that shifted public stance for vaccination. This tool can easily be generalized for any scenario for visual analysis and to increase the transparency of social media data for researchers and the general population alike.
    
[^6]: 多种网络爬虫算法的比较分析

    Comparative analysis of various web crawler algorithms. (arXiv:2306.12027v1 [cs.IR])

    [http://arxiv.org/abs/2306.12027](http://arxiv.org/abs/2306.12027)

    本文介绍了网络爬虫和网页排名算法在处理Web数据方面的重要性；在评估五种不同的爬取算法后，旨在确定最有效的爬取算法。

    

    本文论述了网络爬虫和网页排名算法在处理世界各地网络数据方面的重要性。随着Web的急剧增长，高效的搜索和检索方法变得至关重要。网络爬虫是将非结构化数据转换为结构化数据的过程，从而实现有效的信息检索。此外，网页排名算法在评估网页的质量和受欢迎程度方面发挥着重要作用。本文探讨了这些算法的背景，并评估了五种不同的爬取算法：鲨鱼搜索，基于优先级的队列，朴素贝叶斯，广度优先和深度优先。本文的目标是确定最有效的网络爬虫算法。通过了解这些算法，我们可以提高自己在Web上导航和提取有价值信息的能力。

    This presentation focuses on the importance of web crawling and page ranking algorithms in dealing with the massive amount of data present on the World Wide Web. As the web continues to grow exponentially, efficient search and retrieval methods become crucial. Web crawling is a process that converts unstructured data into structured data, enabling effective information retrieval. Additionally, page ranking algorithms play a significant role in assessing the quality and popularity of web pages. The presentation explores the background of these algorithms and evaluates five different crawling algorithms: Shark Search, Priority-Based Queue, Naive Bayes, Breadth-First, and Depth-First. The goal is to identify the most effective algorithm for crawling web pages. By understanding these algorithms, we can enhance our ability to navigate the web and extract valuable information efficiently.
    
[^7]: 通过奇异谱平滑解决顺序推荐中的排名退化问题

    Addressing the Rank Degeneration in Sequential Recommendation via Singular Spectrum Smoothing. (arXiv:2306.11986v1 [cs.IR])

    [http://arxiv.org/abs/2306.11986](http://arxiv.org/abs/2306.11986)

    本文提出了一种通过奇异谱平滑算法缓解顺序推荐中序列与项目排名退化问题的方法，并提出了SSA指标来评估该问题的严重性。

    

    顺序推荐研究动态用户偏好建模并生成下一个项目预测。下一个项目的偏好通常是通过序列和项目表示之间的亲和度生成的。然而，由于数据稀疏问题，序列和项目表示都会遭受排名降级问题。排名退化问题严重损害了顺序推荐的表示。因此我们提出了通过理论连接序列表示降级问题与项目排名退化问题的方法。我们还发现了快速奇异值衰减现象与转换器序列输出和项目嵌入中的排名折叠问题之间的联系。我们提出了奇异值曲线下面积（SSA）评估指标，同时缓解顺序推荐中的序列和项目表示排名退化问题。

    Sequential recommendation (SR) investigates the dynamic user preferences modeling and generates the next-item prediction. The next item preference is typically generated by the affinity between the sequence and item representations. However, both sequence and item representations suffer from the rank degeneration issue due to the data sparsity problem. The rank degeneration issue significantly impairs the representations for SR. This motivates us to measure how severe is the rank degeneration issue and alleviate the sequence and item representation rank degeneration issues simultaneously for SR.  In this work, we theoretically connect the sequence representation degeneration issue with the item rank degeneration, particularly for short sequences and cold items. We also identify the connection between the fast singular value decay phenomenon and the rank collapse issue in transformer sequence output and item embeddings. We propose the area under the singular value curve metric to evalua
    
[^8]: 采样个体公平且满足群体公平的排名

    Sampling Individually-Fair Rankings that are Always Group Fair. (arXiv:2306.11964v1 [cs.CY])

    [http://arxiv.org/abs/2306.11964](http://arxiv.org/abs/2306.11964)

    该论文提出了一种有效算法，从个体公平分布中采样排名，同时确保每个输出的排名都满足群体公平性约束。输出排名的期望效用至少是最优公平解的效用的$\alpha$倍，其中$\alpha$是一个量化公平约束紧度的参数。

    

    在线平台上的排名可以帮助用户快速找到相关信息，如人物、新闻、媒体和产品。公平排名是一种为了满足群体公平性约束而优化一组项目排名的任务，已经在算法公平性、信息检索和机器学习领域引起了广泛关注。然而，近期的研究表明项目效用的不确定性是不公平的主要原因，并建议在输出中引入随机性。这种随机性经过仔细选择，以确保对每个项目进行充分且合理的代表（同时考虑不确定性）。然而，由于这种随机性，输出的排名可能会违反群体公平性约束。我们提出了一个有效的算法，从一个个体公平分布中抽样排名，同时确保每个输出的排名都满足群体公平性约束。输出排名的期望效用至少是最优公平解的效用的 $\alpha$ 倍，其中 $\alpha$ 是一个量化公平约束紧度的参数。我们在真实世界数据集上进行实验，证明了我们算法的高效性和有效性。

    Rankings on online platforms help their end-users find the relevant information -- people, news, media, and products -- quickly. Fair ranking tasks, which ask to rank a set of items to maximize utility subject to satisfying group-fairness constraints, have gained significant interest in the Algorithmic Fairness, Information Retrieval, and Machine Learning literature. Recent works, however, identify uncertainty in the utilities of items as a primary cause of unfairness and propose introducing randomness in the output. This randomness is carefully chosen to guarantee an adequate representation of each item (while accounting for the uncertainty). However, due to this randomness, the output rankings may violate group fairness constraints. We give an efficient algorithm that samples rankings from an individually-fair distribution while ensuring that every output ranking is group fair. The expected utility of the output ranking is at least $\alpha$ times the utility of the optimal fair solut
    
[^9]: 智慧医疗中的多模态融合:从数据、信息、知识到智慧之旅

    Multimodality Fusion for Smart Healthcare: a Journey from Data, Information, Knowledge to Wisdom. (arXiv:2306.11963v1 [cs.IR])

    [http://arxiv.org/abs/2306.11963](http://arxiv.org/abs/2306.11963)

    本文综述了多模态医学数据融合在智慧医疗中的应用，提出了符合DIKW机制的通用融合框架，探讨了面临的挑战和未来的发展方向。

    

    多模态医学数据融合已成为智慧医疗中的一种革新性方法，能够全面了解患者健康状况和个性化治疗方案。本文探讨了多模态融合为智慧医疗带来的从数据、信息和知识到智慧（DIKW）之旅。全面回顾了多模态医学数据融合的研究现状，重点关注了不同数据模态的集成方式。文章探讨了特征选择、基于规则的系统、机器学习、深度学习和自然语言处理等不同方法，用于多模态数据的融合和分析。同时，文章也着重讨论了多模态融合在医疗保健中面临的挑战。通过综合评述的框架和见解，提出了一个符合DIKW机制的通用多模态医疗数据融合框架。此外，文章还探讨了未来与预测、预防、个性化和治疗有关的医疗方向。

    Multimodal medical data fusion has emerged as a transformative approach in smart healthcare, enabling a comprehensive understanding of patient health and personalized treatment plans. In this paper, a journey from data, information, and knowledge to wisdom (DIKW) is explored through multimodal fusion for smart healthcare. A comprehensive review of multimodal medical data fusion focuses on the integration of various data modalities are presented. It explores different approaches such as Feature selection, Rule-based systems, Machine learning, Deep learning, and Natural Language Processing for fusing and analyzing multimodal data. The paper also highlights the challenges associated with multimodal fusion in healthcare. By synthesizing the reviewed frameworks and insights, a generic framework for multimodal medical data fusion is proposed while aligning with the DIKW mechanism. Moreover, it discusses future directions aligned with the four pillars of healthcare: Predictive, Preventive, Pe
    
[^10]: 基于检索的Transformer模型用于表格增强

    Retrieval-Based Transformer for Table Augmentation. (arXiv:2306.11843v1 [cs.CL])

    [http://arxiv.org/abs/2306.11843](http://arxiv.org/abs/2306.11843)

    本文提出了一种自动数据处理的新方法，其中使用基于检索的Transformer模型来解决表格增强任务，并采用自学习策略来训练模型以重构原始值或标题，以便减轻数据分析师在数据处理中的工作量。

    

    数据准备（也称数据整理）通常被认为是进行分析或构建机器学习模型时最耗费时间和精力的步骤之一。本文引入了一种自动数据处理的新方法，以试图减轻最终用户（例如数据分析师）在从数据湖中构建动态表格数据的过程中的工作量。我们旨在解决表格增强任务，包括行/列填充和数据插补。给定一组表格，我们提出了一种检索增强的自学习Transformer模型。我们的自学习策略是从语料库中随机去除表格，并训练检索模型以在给定部分表格作为输入的情况下重构原始值或标题。我们采用这种策略来首先训练密集的神经检索模型

    Data preparation, also called data wrangling, is considered one of the most expensive and time-consuming steps when performing analytics or building machine learning models. Preparing data typically involves collecting and merging data from complex heterogeneous, and often large-scale data sources, such as data lakes. In this paper, we introduce a novel approach toward automatic data wrangling in an attempt to alleviate the effort of end-users, e.g. data analysts, in structuring dynamic views from data lakes in the form of tabular data. We aim to address table augmentation tasks, including row/column population and data imputation. Given a corpus of tables, we propose a retrieval augmented self-trained transformer model. Our self-learning strategy consists in randomly ablating tables from the corpus and training the retrieval-based model to reconstruct the original values or headers given the partial tables as input. We adopt this strategy to first train the dense neural retrieval mode
    

