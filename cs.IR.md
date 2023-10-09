# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Policy-Gradient Training of Language Models for Ranking.](http://arxiv.org/abs/2310.04407) | 该论文提出了一种用于排序的语言模型的策略梯度训练算法Neural PG-RANK，通过将大规模语言模型实例化为Plackett-Luce排名策略，实现了对检索模型的原则性、端到端训练。 |
| [^2] | [On the Embedding Collapse when Scaling up Recommendation Models.](http://arxiv.org/abs/2310.04400) | 研究了可缩放推荐模型中嵌入层的崩溃现象，发现特征交互模块在一定程度上限制了嵌入学习，但也是提高可扩展性的关键因素。 |
| [^3] | [Workload-aware and Learned Z-Indexes.](http://arxiv.org/abs/2310.04268) | 本文提出了一种基于工作负载和学习的Z-索引变体，通过优化存储布局和搜索结构，改善了范围查询性能，并通过引入页面跳跃机制进一步提升查询性能。实验证明，该索引在范围查询时间、点查询性能和构建时间与索引大小之间保持了良好的平衡。 |
| [^4] | [Lending Interaction Wings to Recommender Systems with Conversational Agents.](http://arxiv.org/abs/2310.04230) | 本文提出了一种将对话代理与推荐系统结合的新范例CORE，通过统一的不确定性最小化框架，以离线训练和在线检验的形式实现了对话和推荐部分的交互。核心思想是将推荐系统作为离线相关性评分估计器，将对话代理作为在线相关性评分检查器，通过最小化不确定性来提高推荐系统的准确性和用户满意度。 |
| [^5] | [Keyword Augmented Retrieval: Novel framework for Information Retrieval integrated with speech interface.](http://arxiv.org/abs/2310.04205) | 这项研究工作介绍了一种关键词增强检索框架，该框架通过使用关键词来优化语言模型的上下文发现和答案生成，从而实现了快速、低成本的信息检索和语音接口集成。 |
| [^6] | [Searching COVID-19 clinical research using graphical abstracts.](http://arxiv.org/abs/2310.04094) | 该论文介绍了一种使用图形摘要来搜索COVID-19临床研究的方法。通过将摘要和图形摘要表示为本体术语图并在网络中匹配，可以更有效地对现有文献进行搜索。 |
| [^7] | [AdaRec: Adaptive Sequential Recommendation for Reinforcing Long-term User Engagement.](http://arxiv.org/abs/2310.03984) | AdaRec是一种自适应顺序推荐算法，通过引入基于距离的表示损失来提取潜在信息，以适应大规模在线推荐系统中用户行为模式的变化。 |
| [^8] | [An Efficient Content-based Time Series Retrieval System.](http://arxiv.org/abs/2310.03919) | 本论文提出了一种高效的基于内容的时间序列检索系统，可以在用户与系统实时交互的情况下，有效地度量和计算不同时间序列之间的相似度，满足用户从多个领域获取时间序列信息的需求。 |
| [^9] | [Living Lab Evaluation for Life and Social Sciences Search Platforms -- LiLAS at CLEF 2021.](http://arxiv.org/abs/2310.03859) | 本研究介绍了通过LiLAS实验室对生命科学和社会科学领域的真实学术搜索系统进行用户中心评估的方法，为参与者提供了系统内容的元数据和候选列表，并允许他们轻松集成自己的方法到实际系统中。 |
| [^10] | [Accurate Cold-start Bundle Recommendation via Popularity-based Coalescence and Curriculum Heating.](http://arxiv.org/abs/2310.03813) | 本文提出了CoHeat算法，一种准确的冷启动捆绑推荐方法。该算法通过结合历史和关联信息，应对捆绑互动分布的倾斜，并有效地学习潜在表示。 |
| [^11] | [Literature Based Discovery (LBD): Towards Hypothesis Generation and Knowledge Discovery in Biomedical Text Mining.](http://arxiv.org/abs/2310.03766) | LBD是在生物医学文本挖掘中通过自动发现医学术语之间的新关联来缩短发现潜在关联的时间的方法。 |
| [^12] | [FASER: Binary Code Similarity Search through the use of Intermediate Representations.](http://arxiv.org/abs/2310.03605) | 本论文提出了一种名为FASER的方法，通过使用中间表示进行二进制代码相似性搜索。该方法可以跨架构地识别函数，并明确编码函数的语义，以支持各种应用场景。 |
| [^13] | [Deep Neural Aggregation for Recommending Items to Group of Users.](http://arxiv.org/abs/2307.09447) | 本文针对群体用户推荐商品的问题，提出了两种新的深度学习模型，并通过实验证明了这些模型相比现有模型的改进效果。 |
| [^14] | [Topic-Centric Explanations for News Recommendation.](http://arxiv.org/abs/2306.07506) | 提出了一种基于主题的解释性新闻推荐模型，可以准确地识别相关文章并解释为什么推荐这些文章，同时提高了解释的可解释性度量。 |
| [^15] | [Making Large Language Models Interactive: A Pioneer Study on Supporting Complex Information-Seeking Tasks with Implicit Constraints.](http://arxiv.org/abs/2205.00584) | 本研究设计并部署了一个平台，用于收集复杂交互系统的数据，以解决当前交互系统无法理解一次性提出的复杂信息检索请求的问题。同时，研究发现当前的生成语言模型在提供准确的事实知识方面存在问题。 |

# 详细

[^1]: 用于排序的语言模型的策略梯度训练

    Policy-Gradient Training of Language Models for Ranking. (arXiv:2310.04407v1 [cs.CL])

    [http://arxiv.org/abs/2310.04407](http://arxiv.org/abs/2310.04407)

    该论文提出了一种用于排序的语言模型的策略梯度训练算法Neural PG-RANK，通过将大规模语言模型实例化为Plackett-Luce排名策略，实现了对检索模型的原则性、端到端训练。

    

    文本检索在将事实知识纳入到语言处理流程中的决策过程中起着关键作用，从聊天式网页搜索到问答系统。当前最先进的文本检索模型利用预训练的大规模语言模型（LLM）以达到有竞争力的性能，但通过典型的对比损失训练基于LLM的检索器需要复杂的启发式算法，包括选择困难的负样本和使用额外的监督作为学习信号。这种依赖于启发式算法的原因是对比损失本身是启发式的，不能直接优化处理流程末端决策质量的下游指标。为了解决这个问题，我们引入了神经PG-RANK，一种新的训练算法，通过将LLM实例化为Plackett-Luce排名策略，学习排序。神经PG-RANK为检索模型的端到端训练提供了一种原则性方法，作为更大的决策系统的一部分进行训练。

    Text retrieval plays a crucial role in incorporating factual knowledge for decision making into language processing pipelines, ranging from chat-based web search to question answering systems. Current state-of-the-art text retrieval models leverage pre-trained large language models (LLMs) to achieve competitive performance, but training LLM-based retrievers via typical contrastive losses requires intricate heuristics, including selecting hard negatives and using additional supervision as learning signals. This reliance on heuristics stems from the fact that the contrastive loss itself is heuristic and does not directly optimize the downstream metrics of decision quality at the end of the processing pipeline. To address this issue, we introduce Neural PG-RANK, a novel training algorithm that learns to rank by instantiating a LLM as a Plackett-Luce ranking policy. Neural PG-RANK provides a principled method for end-to-end training of retrieval models as part of larger decision systems vi
    
[^2]: 论可扩展推荐模型中嵌入坍缩现象的研究

    On the Embedding Collapse when Scaling up Recommendation Models. (arXiv:2310.04400v1 [cs.LG])

    [http://arxiv.org/abs/2310.04400](http://arxiv.org/abs/2310.04400)

    研究了可缩放推荐模型中嵌入层的崩溃现象，发现特征交互模块在一定程度上限制了嵌入学习，但也是提高可扩展性的关键因素。

    

    深度基础模型的最新进展引发了开发大型推荐模型以利用大量可用数据的有前景趋势。然而，我们试验放大现有的推荐模型时发现，扩大的模型并没有令人满意的改进。在这种情况下，我们研究了扩大模型的嵌入层，并发现了一种嵌入坍缩现象，这最终阻碍了可扩展性，在这种现象中，嵌入矩阵倾向于存在于低维子空间中。通过实证和理论分析，我们证明了推荐模型特定的特征交互模块具有双重作用。一方面，当与坍缩的嵌入交互时，该交互限制了嵌入学习，加剧了崩溃问题。另一方面，特征交互对于缓解假特征的拟合至关重要，从而提高可扩展性。基于这一分析，我们提出了一个简单而有效的方法

    Recent advances in deep foundation models have led to a promising trend of developing large recommendation models to leverage vast amounts of available data. However, we experiment to scale up existing recommendation models and observe that the enlarged models do not improve satisfactorily. In this context, we investigate the embedding layers of enlarged models and identify a phenomenon of embedding collapse, which ultimately hinders scalability, wherein the embedding matrix tends to reside in a low-dimensional subspace. Through empirical and theoretical analysis, we demonstrate that the feature interaction module specific to recommendation models has a two-sided effect. On the one hand, the interaction restricts embedding learning when interacting with collapsed embeddings, exacerbating the collapse issue. On the other hand, feature interaction is crucial in mitigating the fitting of spurious features, thereby improving scalability. Based on this analysis, we propose a simple yet effe
    
[^3]: 基于工作负载和学习的Z-索引

    Workload-aware and Learned Z-Indexes. (arXiv:2310.04268v1 [cs.DB])

    [http://arxiv.org/abs/2310.04268](http://arxiv.org/abs/2310.04268)

    本文提出了一种基于工作负载和学习的Z-索引变体，通过优化存储布局和搜索结构，改善了范围查询性能，并通过引入页面跳跃机制进一步提升查询性能。实验证明，该索引在范围查询时间、点查询性能和构建时间与索引大小之间保持了良好的平衡。

    

    本文提出了一种基于工作负载和学习的Z-索引的变体，该索引同时优化存储布局和搜索结构，作为解决空间索引的挑战的可行解决方案。具体来说，我们首先制定了一个成本函数，用于衡量Z-索引在数据集上的范围查询工作负载下的性能。然后，通过自适应分区和排序优化Z-索引结构，最小化成本函数。此外，我们设计了一种新颖的页面跳跃机制，通过减少对无关数据页面的访问来改善查询性能。我们广泛的实验证明，相比基线，我们的索引平均改善了40%的范围查询时间，同时始终表现得更好或与最先进的空间索引相当。此外，我们的索引在提供有利的构建时间和索引大小权衡的同时，保持良好的点查询性能。

    In this paper, a learned and workload-aware variant of a Z-index, which jointly optimizes storage layout and search structures, as a viable solution for the above challenges of spatial indexing. Specifically, we first formulate a cost function to measure the performance of a Z-index on a dataset for a range-query workload. Then, we optimize the Z-index structure by minimizing the cost function through adaptive partitioning and ordering for index construction. Moreover, we design a novel page-skipping mechanism to improve its query performance by reducing access to irrelevant data pages. Our extensive experiments show that our index improves range query time by 40% on average over the baselines, while always performing better or comparably to state-of-the-art spatial indexes. Additionally, our index maintains good point query performance while providing favourable construction time and index size tradeoffs.
    
[^4]: 将对话代理引入推荐系统中的互动

    Lending Interaction Wings to Recommender Systems with Conversational Agents. (arXiv:2310.04230v1 [cs.IR])

    [http://arxiv.org/abs/2310.04230](http://arxiv.org/abs/2310.04230)

    本文提出了一种将对话代理与推荐系统结合的新范例CORE，通过统一的不确定性最小化框架，以离线训练和在线检验的形式实现了对话和推荐部分的交互。核心思想是将推荐系统作为离线相关性评分估计器，将对话代理作为在线相关性评分检查器，通过最小化不确定性来提高推荐系统的准确性和用户满意度。

    

    在离线历史用户行为训练的推荐系统中，我们采用对话技术进行在线查询用户偏好。与以往系统地通过强化学习框架将对话和推荐部分结合的对话推荐方法不同，我们提出了CORE，一种基于离线训练和在线检验的新范例，通过统一的不确定性最小化框架，将对话代理和推荐系统连接起来。它可以以即插即用的方式为任何推荐平台带来好处。在这里，CORE将推荐系统视为离线相关性评分估计器，为每个项目产生一个估计的相关性评分；而对话代理被视为在线相关性评分检查器，在每个会话中检查这些估计分数。我们将不确定性定义为未经检查的相关性评分的总和。在这方面，对话代理通过查询属性或项目来最小化不确定性。

    Recommender systems trained on offline historical user behaviors are embracing conversational techniques to online query user preference. Unlike prior conversational recommendation approaches that systemically combine conversational and recommender parts through a reinforcement learning framework, we propose CORE, a new offline-training and online-checking paradigm that bridges a COnversational agent and REcommender systems via a unified uncertainty minimization framework. It can benefit any recommendation platform in a plug-and-play style. Here, CORE treats a recommender system as an offline relevance score estimator to produce an estimated relevance score for each item; while a conversational agent is regarded as an online relevance score checker to check these estimated scores in each session. We define uncertainty as the summation of unchecked relevance scores. In this regard, the conversational agent acts to minimize uncertainty via querying either attributes or items. Based on th
    
[^5]: 关键词增强检索: 集成语音接口的信息检索新框架

    Keyword Augmented Retrieval: Novel framework for Information Retrieval integrated with speech interface. (arXiv:2310.04205v1 [cs.IR])

    [http://arxiv.org/abs/2310.04205](http://arxiv.org/abs/2310.04205)

    这项研究工作介绍了一种关键词增强检索框架，该框架通过使用关键词来优化语言模型的上下文发现和答案生成，从而实现了快速、低成本的信息检索和语音接口集成。

    

    使用语言模型从结构化和非结构化数据的组合中快速、低成本地检索答案，而不产生幻觉，是阻止语言模型在知识检索自动化中应用的一大障碍。当想要集成语音接口时，这一问题变得更加突出。此外，对于商业搜索和聊天机器人应用来说，完全依赖商业大型语言模型（如GPT 3.5等）可能非常昂贵。本文作者通过首先开发基于关键词的搜索框架来解决这个问题，该框架增强了对要提供给大型语言模型的上下文的发现。关键词反过来是由语言模型生成并缓存，以便与查询生成的关键词进行比较。这显著减少了在文档中查找上下文所需的时间和成本。一旦上下文设置好了，语言模型就可以根据为问答定制的提示提供答案。这项研究工作表明，

    Retrieving answers in a quick and low cost manner without hallucinations from a combination of structured and unstructured data using Language models is a major hurdle which prevents employment of Language models in knowledge retrieval automation. This becomes accentuated when one wants to integrate a speech interface. Besides, for commercial search and chatbot applications, complete reliance on commercial large language models (LLMs) like GPT 3.5 etc. can be very costly. In this work, authors have addressed this problem by first developing a keyword based search framework which augments discovery of the context to be provided to the large language model. The keywords in turn are generated by LLM and cached for comparison with keywords generated by LLM against the query raised. This significantly reduces time and cost to find the context within documents. Once the context is set, LLM uses that to provide answers based on a prompt tailored for Q&A. This research work demonstrates that u
    
[^6]: 使用图形摘要搜索COVID-19临床研究

    Searching COVID-19 clinical research using graphical abstracts. (arXiv:2310.04094v1 [cs.IR])

    [http://arxiv.org/abs/2310.04094](http://arxiv.org/abs/2310.04094)

    该论文介绍了一种使用图形摘要来搜索COVID-19临床研究的方法。通过将摘要和图形摘要表示为本体术语图并在网络中匹配，可以更有效地对现有文献进行搜索。

    

    目标：图形摘要是对科学文章主要发现进行视觉总结的概念图。虽然图形摘要通常用于科学出版物中预测和总结主要结果，但我们将其作为表达对现有文献进行图形搜索的手段。材料和方法：我们考虑COVID-19开放研究数据集（CORD-19），这是一个包含超过一百万个摘要的语料库；每个摘要被描述为共现本体术语图，这些术语从统一医学语言系统（UMLS）和冠状病毒传染病本体（CIDO）中选择。图形摘要也被表示为本体术语图，可能还包括描述其相互作用的实用术语（例如，“相关”，“增加”，“引发”）。我们构建了一个包含语料库中提及的概念的共现网络；然后我们在网络上识别出图形摘要的最佳匹配项。我们利用图形数据库...

    Objective. Graphical abstracts are small graphs of concepts that visually summarize the main findings of scientific articles. While graphical abstracts are customarily used in scientific publications to anticipate and summarize their main results, we propose them as a means for expressing graph searches over existing literature. Materials and methods. We consider the COVID-19 Open Research Dataset (CORD-19), a corpus of more than one million abstracts; each of them is described as a graph of co-occurring ontological terms, selected from the Unified Medical Language System (UMLS) and the Ontology of Coronavirus Infectious Disease (CIDO). Graphical abstracts are also expressed as graphs of ontological terms, possibly augmented by utility terms describing their interactions (e.g., "associated with", "increases", "induces"). We build a co-occurrence network of concepts mentioned in the corpus; we then identify the best matches of graphical abstracts on the network. We exploit graph databas
    
[^7]: AdaRec：用于增强用户长期参与度的自适应顺序推荐算法

    AdaRec: Adaptive Sequential Recommendation for Reinforcing Long-term User Engagement. (arXiv:2310.03984v1 [cs.IR])

    [http://arxiv.org/abs/2310.03984](http://arxiv.org/abs/2310.03984)

    AdaRec是一种自适应顺序推荐算法，通过引入基于距离的表示损失来提取潜在信息，以适应大规模在线推荐系统中用户行为模式的变化。

    

    在顺序推荐任务中，人们越来越关注使用强化学习算法来优化用户的长期参与度。大规模在线推荐系统面临的一个挑战是用户行为模式（如互动频率和保留倾向）的不断复杂变化。当将问题建模为马尔科夫决策过程时，推荐系统的动态和奖励函数会不断受到这些变化的影响。现有的推荐系统强化学习算法会受到分布偏移问题的困扰，并难以适应这种马尔科夫决策过程。本文介绍了一种新的范式，称为自适应顺序推荐（AdaRec），来解决这个问题。AdaRec提出了一种基于距离的表示损失，从用户的互动轨迹中提取潜在信息。这些信息反映了强化学习策略与当前用户行为模式的匹配程度，并帮助策略识别推荐系统中的细微变化。

    Growing attention has been paid to Reinforcement Learning (RL) algorithms when optimizing long-term user engagement in sequential recommendation tasks. One challenge in large-scale online recommendation systems is the constant and complicated changes in users' behavior patterns, such as interaction rates and retention tendencies. When formulated as a Markov Decision Process (MDP), the dynamics and reward functions of the recommendation system are continuously affected by these changes. Existing RL algorithms for recommendation systems will suffer from distribution shift and struggle to adapt in such an MDP. In this paper, we introduce a novel paradigm called Adaptive Sequential Recommendation (AdaRec) to address this issue. AdaRec proposes a new distance-based representation loss to extract latent information from users' interaction trajectories. Such information reflects how RL policy fits to current user behavior patterns, and helps the policy to identify subtle changes in the recomm
    
[^8]: 一种高效的基于内容的时间序列检索系统

    An Efficient Content-based Time Series Retrieval System. (arXiv:2310.03919v1 [cs.IR])

    [http://arxiv.org/abs/2310.03919](http://arxiv.org/abs/2310.03919)

    本论文提出了一种高效的基于内容的时间序列检索系统，可以在用户与系统实时交互的情况下，有效地度量和计算不同时间序列之间的相似度，满足用户从多个领域获取时间序列信息的需求。

    

    基于内容的时间序列检索(CTSR)系统是一个信息检索系统，用户可以与来自多个领域(如金融、医疗和制造业)的时间序列进行交互。例如，用户想要了解时间序列的来源，可以将时间序列作为查询提交给CTSR系统，并检索与之相关的时间序列列表及相关元数据。通过分析检索到的元数据，用户可以获得有关时间序列来源的更多信息。由于CTSR系统需要处理来自不同领域的时间序列数据，因此需要一个高容量模型来有效地度量不同时间序列之间的相似度。此外，CTSR系统内的模型还需要以高效的方式计算相似度得分，以满足用户在实时交互中的需求。本文提出了一种有效且高效的CTSR模型，其性能优于其他替代模型，同时仍然提供合理的准确性。

    A Content-based Time Series Retrieval (CTSR) system is an information retrieval system for users to interact with time series emerged from multiple domains, such as finance, healthcare, and manufacturing. For example, users seeking to learn more about the source of a time series can submit the time series as a query to the CTSR system and retrieve a list of relevant time series with associated metadata. By analyzing the retrieved metadata, users can gather more information about the source of the time series. Because the CTSR system is required to work with time series data from diverse domains, it needs a high-capacity model to effectively measure the similarity between different time series. On top of that, the model within the CTSR system has to compute the similarity scores in an efficient manner as the users interact with the system in real-time. In this paper, we propose an effective and efficient CTSR model that outperforms alternative models, while still providing reasonable in
    
[^9]: 生活与社会科学搜索平台的Living Lab评估- LiLAS在CLEF 2021中

    Living Lab Evaluation for Life and Social Sciences Search Platforms -- LiLAS at CLEF 2021. (arXiv:2310.03859v1 [cs.IR])

    [http://arxiv.org/abs/2310.03859](http://arxiv.org/abs/2310.03859)

    本研究介绍了通过LiLAS实验室对生命科学和社会科学领域的真实学术搜索系统进行用户中心评估的方法，为参与者提供了系统内容的元数据和候选列表，并允许他们轻松集成自己的方法到实际系统中。

    

    在控制的离线评估活动（如TREC和CLEF）的元评估研究中，系统性能评估方面存在创新的需求，学术搜索领域也不例外。这可能与学术搜索中的相关性是多层次的事实有关，因此用户中心评估的方面变得越来越重要。学术搜索的Living Labs（LiLAS）实验室旨在通过允许参与者在生命科学和社会科学领域的两个真实学术搜索系统中评估其检索方法，加强用户中心生活实验室的概念。为此，我们为参与者提供了系统内容的元数据以及候选列表，要求将最相关的候选排在前面。利用STELLA基础设施，我们允许参与者将自己的方法轻松集成到真实系统中，并提供将方案部署到在线实验系统中的可能性。

    Meta-evaluation studies of system performances in controlled offline evaluation campaigns, like TREC and CLEF, show a need for innovation in evaluating IR-systems. The field of academic search is no exception to this. This might be related to the fact that relevance in academic search is multilayered and therefore the aspect of user-centric evaluation is becoming more and more important. The Living Labs for Academic Search (LiLAS) lab aims to strengthen the concept of user-centric living labs for the domain of academic search by allowing participants to evaluate their retrieval approaches in two real-world academic search systems from the life sciences and the social sciences. To this end, we provide participants with metadata on the systems' content as well as candidate lists with the task to rank the most relevant candidate to the top. Using the STELLA-infrastructure, we allow participants to easily integrate their approaches into the real-world systems and provide the possibility to
    
[^10]: 准确的冷启动捆绑推荐：基于流行度的聚合和课程加热

    Accurate Cold-start Bundle Recommendation via Popularity-based Coalescence and Curriculum Heating. (arXiv:2310.03813v1 [cs.IR])

    [http://arxiv.org/abs/2310.03813](http://arxiv.org/abs/2310.03813)

    本文提出了CoHeat算法，一种准确的冷启动捆绑推荐方法。该算法通过结合历史和关联信息，应对捆绑互动分布的倾斜，并有效地学习潜在表示。

    

    如何准确地向用户推荐冷启动捆绑？捆绑推荐中的冷启动问题在实际场景中至关重要，因为新建捆绑不断出现以满足各种营销目的。尽管其重要性，之前没有研究涉及冷启动捆绑推荐。此外，现有的冷启动物品推荐方法过于依赖历史信息，即使对于不受欢迎的捆绑也是如此，无法应对捆绑互动分布高度倾斜的主要挑战。在这项工作中，我们提出了CoHeat（基于流行度的聚合和课程加热），这是一种准确的冷启动捆绑推荐方法。CoHeat通过结合历史信息和关联信息来估计用户与捆绑之间的关系，以应对捆绑互动分布的高度倾斜问题。此外，CoHeat还通过利用课程学习和聚合特征学习效果地学习潜在表示。

    How can we accurately recommend cold-start bundles to users? The cold-start problem in bundle recommendation is critical in practical scenarios since new bundles are continuously created for various marketing purposes. Despite its importance, no previous studies have addressed cold-start bundle recommendation. Moreover, existing methods for cold-start item recommendation overly rely on historical information, even for unpopular bundles, failing to tackle the primary challenge of the highly skewed distribution of bundle interactions. In this work, we propose CoHeat (Popularity-based Coalescence and Curriculum Heating), an accurate approach for the cold-start bundle recommendation. CoHeat tackles the highly skewed distribution of bundle interactions by incorporating both historical and affiliation information based on the bundle's popularity when estimating the user-bundle relationship. Furthermore, CoHeat effectively learns latent representations by exploiting curriculum learning and co
    
[^11]: 基于文献的发现（LBD）：在生物医学文本挖掘中实现假设生成和知识发现

    Literature Based Discovery (LBD): Towards Hypothesis Generation and Knowledge Discovery in Biomedical Text Mining. (arXiv:2310.03766v1 [cs.IR])

    [http://arxiv.org/abs/2310.03766](http://arxiv.org/abs/2310.03766)

    LBD是在生物医学文本挖掘中通过自动发现医学术语之间的新关联来缩短发现潜在关联的时间的方法。

    

    生物医学知识以科学出版物的形式以惊人的速度增长。文本挖掘工具和方法代表了从这些半结构化和非结构化数据中提取隐藏模式和趋势的自动化方法。在生物医学文本挖掘中，基于文献的发现（LBD）是自动发现不同文献集中提到的医学术语之间的新关联的过程。LBD方法已被证明可以成功缩短在大量科学文献中隐藏的潜在关联的发现时间。该过程侧重于为疾病或症状等医学术语创建概念档案，并根据共享档案的统计显著性将其与药物和治疗联系起来。这种知识发现方法在1989年引入后仍然是文本挖掘的核心任务。

    Biomedical knowledge is growing in an astounding pace with a majority of this knowledge is represented as scientific publications. Text mining tools and methods represents automatic approaches for extracting hidden patterns and trends from this semi structured and unstructured data. In Biomedical Text mining, Literature Based Discovery (LBD) is the process of automatically discovering novel associations between medical terms otherwise mentioned in disjoint literature sets. LBD approaches proven to be successfully reducing the discovery time of potential associations that are hidden in the vast amount of scientific literature. The process focuses on creating concept profiles for medical terms such as a disease or symptom and connecting it with a drug and treatment based on the statistical significance of the shared profiles. This knowledge discovery approach introduced in 1989 still remains as a core task in text mining. Currently the ABC principle based two approaches namely open disco
    
[^12]: FASER: 通过中间表示进行二进制代码相似性搜索

    FASER: Binary Code Similarity Search through the use of Intermediate Representations. (arXiv:2310.03605v1 [cs.CR])

    [http://arxiv.org/abs/2310.03605](http://arxiv.org/abs/2310.03605)

    本论文提出了一种名为FASER的方法，通过使用中间表示进行二进制代码相似性搜索。该方法可以跨架构地识别函数，并明确编码函数的语义，以支持各种应用场景。

    

    能够识别跨架构软件中感兴趣的函数对于分析恶意软件、保护软件供应链或进行漏洞研究都是有用的。跨架构二进制代码相似性搜索已在许多研究中探索，并使用了各种不同的数据来源来实现其目标。通常使用的数据来源包括从二进制文件中提取的常见结构，如函数控制流图或二进制级调用图，反汇编过程的输出或动态分析方法的输出。其中一种受到较少关注的数据来源是二进制中间表示。二进制中间表示具有两个有趣的属性：它们的跨架构性质以及明确编码函数的语义以支持下游使用。在本文中，我们提出了一种名为FASER的函数字符串编码表示方法，它结合了长文档转换技术。

    Being able to identify functions of interest in cross-architecture software is useful whether you are analysing for malware, securing the software supply chain or conducting vulnerability research. Cross-Architecture Binary Code Similarity Search has been explored in numerous studies and has used a wide range of different data sources to achieve its goals. The data sources typically used draw on common structures derived from binaries such as function control flow graphs or binary level call graphs, the output of the disassembly process or the outputs of a dynamic analysis approach. One data source which has received less attention is binary intermediate representations. Binary Intermediate representations possess two interesting properties: they are cross architecture by their very nature and encode the semantics of a function explicitly to support downstream usage. Within this paper we propose Function as a String Encoded Representation (FASER) which combines long document transforme
    
[^13]: 为群体用户推荐商品的深度神经聚合

    Deep Neural Aggregation for Recommending Items to Group of Users. (arXiv:2307.09447v1 [cs.IR])

    [http://arxiv.org/abs/2307.09447](http://arxiv.org/abs/2307.09447)

    本文针对群体用户推荐商品的问题，提出了两种新的深度学习模型，并通过实验证明了这些模型相比现有模型的改进效果。

    

    现代社会花费了大量时间在数字交互上，我们的日常行为很多都通过数字手段完成。这导致了许多人工智能工具的出现，帮助我们在生活的各个方面进行辅助。对于数字社会来说，一个关键的工具是推荐系统，它是智能的系统，通过学习我们的过去行为，提出与我们兴趣相符的新行为建议。其中一些系统专门从用户群体的行为中学习，向希望共同完成某个任务的个体群体提出建议。在本文中，我们分析了群体推荐系统的现状，并提出了两种使用新兴的深度学习架构的模型。实验结果表明，与使用四个不同数据集的最新模型相比，采用我们提出的模型可以取得改进。该模型及所有实验的源代码都可供获取。

    Modern society devotes a significant amount of time to digital interaction. Many of our daily actions are carried out through digital means. This has led to the emergence of numerous Artificial Intelligence tools that assist us in various aspects of our lives. One key tool for the digital society is Recommender Systems, intelligent systems that learn from our past actions to propose new ones that align with our interests. Some of these systems have specialized in learning from the behavior of user groups to make recommendations to a group of individuals who want to perform a joint task. In this article, we analyze the current state of Group Recommender Systems and propose two new models that use emerging Deep Learning architectures. Experimental results demonstrate the improvement achieved by employing the proposed models compared to the state-of-the-art models using four different datasets. The source code of the models, as well as that of all the experiments conducted, is available i
    
[^14]: 基于主题的新闻推荐的解释性方法

    Topic-Centric Explanations for News Recommendation. (arXiv:2306.07506v1 [cs.IR])

    [http://arxiv.org/abs/2306.07506](http://arxiv.org/abs/2306.07506)

    提出了一种基于主题的解释性新闻推荐模型，可以准确地识别相关文章并解释为什么推荐这些文章，同时提高了解释的可解释性度量。

    

    新闻推荐系统被广泛应用于在线新闻网站，以帮助用户根据他们的兴趣找到相关文章。然而，推荐缺乏解释会导致用户的不信任和推荐的缺乏接受度。为了解决这个问题，我们提出了一种新的可解释的新闻模型，构建了一个基于主题的解释性推荐方法，可以准确地识别相关文章并解释为什么推荐这些文章，利用相关主题的信息。此外，我们的模型结合了两种用于评估主题质量的一致性度量，提供了这些解释的可解释性的度量。我们在MIND数据集上的实验结果表明，所提出的可解释性NRS优于其他几个基线系统，同时还能够产生可解释的主题。

    News recommender systems (NRS) have been widely applied for online news websites to help users find relevant articles based on their interests. Recent methods have demonstrated considerable success in terms of recommendation performance. However, the lack of explanation for these recommendations can lead to mistrust among users and lack of acceptance of recommendations. To address this issue, we propose a new explainable news model to construct a topic-aware explainable recommendation approach that can both accurately identify relevant articles and explain why they have been recommended, using information from associated topics. Additionally, our model incorporates two coherence metrics applied to assess topic quality, providing measure of the interpretability of these explanations. The results of our experiments on the MIND dataset indicate that the proposed explainable NRS outperforms several other baseline systems, while it is also capable of producing interpretable topics compared 
    
[^15]: 使大型语言模型具有交互功能：关于支持带有隐含约束的复杂信息检索任务的先驱研究

    Making Large Language Models Interactive: A Pioneer Study on Supporting Complex Information-Seeking Tasks with Implicit Constraints. (arXiv:2205.00584v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2205.00584](http://arxiv.org/abs/2205.00584)

    本研究设计并部署了一个平台，用于收集复杂交互系统的数据，以解决当前交互系统无法理解一次性提出的复杂信息检索请求的问题。同时，研究发现当前的生成语言模型在提供准确的事实知识方面存在问题。

    

    目前具有自然语言接口的交互系统缺乏理解同时表达多个隐含约束的复杂信息检索请求的能力，并且没有关于用户偏好的先前信息。在这种情况下，用户请求可以一次性以复杂和长的查询形式提出，与对话和探索式搜索模型不同，这些模型通常需要将短表达或查询逐步呈现给系统。我们设计并部署了一个平台来收集这种复杂交互系统的数据。此外，尽管当前的生成语言模型取得了进展，但这些模型在提供准确的事实知识方面存在幻觉。所有语言模型大多是在

    Current interactive systems with natural language interfaces lack the ability to understand a complex information-seeking request which expresses several implicit constraints at once, and there is no prior information about user preferences e.g.,"find hiking trails around San Francisco which are accessible with toddlers and have beautiful scenery in summer", where output is a list of possible suggestions for users to start their exploration. In such scenarios, user requests can be issued in one shot in the form of a complex and long query, unlike conversational and exploratory search models, where require short utterances or queries are often presented to the system step by step. We have designed and deployed a platform to collect the data from approaching such complex interactive systems. Moreover, despite with the current advancement of generative language models these models suffer from hallucination in providing accurate factual knowledge. All language models are mostly trained in 
    

