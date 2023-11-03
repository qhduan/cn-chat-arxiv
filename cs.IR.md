# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Collaborative Large Language Model for Recommender Systems.](http://arxiv.org/abs/2311.01343) | 本研究提出了CLLM4Rec，首个将大型语言模型与推荐系统的 ID 模式紧密集成的协同推荐算法，旨在解决语义差距、虚假相关和低效推荐等问题。通过扩展预训练语言模型的词汇表，并引入软硬提示策略，该算法能够准确地模拟用户和项目的协同与内容语义。 |
| [^2] | [Recommendations by Concise User Profiles from Review Text.](http://arxiv.org/abs/2311.01314) | 本研究提出了一种通过用户提供的评论文本构建简洁用户档案的方法，用于支持互动非常稀疏但信息丰富的用户。实验结果表明，精心选择的文本片段可以实现最佳性能，并超过了使用ChatGPT生成的用户档案的性能。 |
| [^3] | [VM-Rec: A Variational Mapping Approach for Cold-start User Recommendation.](http://arxiv.org/abs/2311.01304) | VM-Rec是一种用于解决冷启动用户推荐问题的变分映射方法，该方法基于生成表达能力强的嵌入的现有用户的互动，从而模拟生成冷启动用户的嵌入过程。 |
| [^4] | [Efficient Neural Ranking using Forward Indexes and Lightweight Encoders.](http://arxiv.org/abs/2311.01263) | 使用双编码器模型的语义匹配能力和快速前向索引，提高了神经排名的效率和延迟。同时，通过预计算表示和优化索引尺寸，进一步改善了计算效率和资源消耗。 |
| [^5] | [Navigating Complex Search Tasks with AI Copilots.](http://arxiv.org/abs/2311.01235) | 该论文介绍了使用AI副驾驶员来导航复杂搜索任务，并探讨了生成AI和辅助代理的出现对于支持复杂搜索任务的潜力和重要性。 |
| [^6] | [Bi-Preference Learning Heterogeneous Hypergraph Networks for Session-based Recommendation.](http://arxiv.org/abs/2311.01125) | 这篇论文提出了一种双偏好学习异构超图网络（BiPNet）的方法来解决基于会话的推荐中的价格偏好问题。这个方法考虑了价格偏好与兴趣偏好的互相影响，从而提高了推荐的准确性和个性化程度。 |
| [^7] | [Collaboration and Transition: Distilling Item Transitions into Multi-Query Self-Attention for Sequential Recommendation.](http://arxiv.org/abs/2311.01056) | 这篇论文提出了一种新的推荐系统方法，利用多查询自注意力和过渡感知嵌入蒸馏来捕捉用户交互序列中的合作和过渡信号。 |
| [^8] | [Evaluation Measures of Individual Item Fairness for Recommender Systems: A Critical Study.](http://arxiv.org/abs/2311.01013) | 本研究对推荐系统中个体项的公平性评估指标进行了关键研究，并指出了现有指标存在的限制和问题。通过重新定义和纠正这些指标，或者解释为何某些限制无法解决，我们提出了改进的方法。此外，我们还进行了全面的实证分析，验证了这些改进的有效性。 |
| [^9] | [Research Team Identification Based on Representation Learning of Academic Heterogeneous Information Network.](http://arxiv.org/abs/2311.00922) | 本文提出了一种基于学术异构信息网络表示学习的科研团队识别方法，通过利用节点级和元路径级的注意机制学习低维稠密实值向量表示，以有效识别和发现学术网络中的科研团队。 |
| [^10] | [Is Human Culture Locked by Evolution?.](http://arxiv.org/abs/2311.00719) | 本文分析了一个名为ZeroMat的算法的社会影响，表明人类文化受进化限制，个体的文化品味可以在没有历史数据的情况下以高精度预测。同时提供了解决方案和对中国政府法规和政策的解释。 |
| [^11] | [Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation.](http://arxiv.org/abs/2310.09874) | 本文利用大型语言模型（LLMs）来增强基于内容的推荐中的免训练数据集压缩方法，旨在通过生成文本内容来合成一个小而信息丰富的数据集，使得模型能够达到与在大型数据集上训练的模型相当的性能。 |
| [^12] | [DiskANN++: Efficient Page-based Search over Isomorphic Mapped Graph Index using Query-sensitivity Entry Vertex.](http://arxiv.org/abs/2310.00402) | 提出了一个优化的DiskANN++方法，使用查询敏感的入口顶点在同构映射图索引上进行高效基于页面的搜索。 |
| [^13] | [Ranking with Popularity Bias: User Welfare under Self-Amplification Dynamics.](http://arxiv.org/abs/2305.18333) | 研究了物品流行度、质量和位置偏差对用户福利的影响，提出了通过探索减轻流行度偏见负面影响的算法。 |
| [^14] | [Retrieval for Extremely Long Queries and Documents with RPRS: a Highly Efficient and Effective Transformer-based Re-Ranker.](http://arxiv.org/abs/2303.01200) | 该论文提出了一种基于新型比例相关度分数（RPRS）的高效有效的基于Transformer的重新排序器，用于处理极长查询和文档的检索任务。与先前的工作相比，在五个不同数据集上进行的广泛评估显示RPRS获得了显著更好的结果。此外，RPRS具有高效性，并且解决了QBD检索任务中低资源训练的问题。 |

# 详细

[^1]: 协同大型语言模型用于推荐系统

    Collaborative Large Language Model for Recommender Systems. (arXiv:2311.01343v1 [cs.IR])

    [http://arxiv.org/abs/2311.01343](http://arxiv.org/abs/2311.01343)

    本研究提出了CLLM4Rec，首个将大型语言模型与推荐系统的 ID 模式紧密集成的协同推荐算法，旨在解决语义差距、虚假相关和低效推荐等问题。通过扩展预训练语言模型的词汇表，并引入软硬提示策略，该算法能够准确地模拟用户和项目的协同与内容语义。

    

    最近，越来越多的人对基于预训练的大型语言模型（LLM）开发下一代推荐系统（RS）产生了兴趣，充分利用其编码知识和推理能力。然而，自然语言与推荐任务之间的语义差距仍未得到很好的解决，导致一些问题，如虚假相关的用户/项目描述符、对用户/项目内容的低效语言建模以及通过自动回归进行低效的推荐等。在本文中，我们提出了CLLM4Rec，这是第一个紧密集成LLM范式和RS的ID范式的生成RS，旨在同时解决上述挑战。我们首先使用用户/项目ID标记扩展了预训练LLM的词汇表，以忠实地模拟用户/项目的协同和内容语义。因此，在预训练阶段，提出了一种新颖的软硬提示策略，通过语言建模有效地学习用户/项目的协同/内容标记嵌入。

    Recently, there is a growing interest in developing next-generation recommender systems (RSs) based on pretrained large language models (LLMs), fully utilizing their encoded knowledge and reasoning ability. However, the semantic gap between natural language and recommendation tasks is still not well addressed, leading to multiple issues such as spuriously-correlated user/item descriptors, ineffective language modeling on user/item contents, and inefficient recommendations via auto-regression, etc. In this paper, we propose CLLM4Rec, the first generative RS that tightly integrates the LLM paradigm and ID paradigm of RS, aiming to address the above challenges simultaneously. We first extend the vocabulary of pretrained LLMs with user/item ID tokens to faithfully model the user/item collaborative and content semantics. Accordingly, in the pretraining stage, a novel soft+hard prompting strategy is proposed to effectively learn user/item collaborative/content token embeddings via language m
    
[^2]: 通过简洁的用户档案中的评论文本进行推荐

    Recommendations by Concise User Profiles from Review Text. (arXiv:2311.01314v1 [cs.IR])

    [http://arxiv.org/abs/2311.01314](http://arxiv.org/abs/2311.01314)

    本研究提出了一种通过用户提供的评论文本构建简洁用户档案的方法，用于支持互动非常稀疏但信息丰富的用户。实验结果表明，精心选择的文本片段可以实现最佳性能，并超过了使用ChatGPT生成的用户档案的性能。

    

    推荐系统对于受欢迎的物品和与用户的丰富互动（喜欢、评分等）最为成功。本研究探讨了一个困难且未被充分探索的情况，即如何支持互动非常稀疏但发布信息丰富的评论文本的用户。我们的实验研究涉及两个具有这些特点的图书社区。我们设计了一个基于Transformer的表征学习框架，涵盖用户-物品互动、物品内容和用户提供的评论。为了克服互动稀疏问题，我们设计了一些技术来选择构建简洁用户档案的最具信息量的线索。通过来自Amazon和Goodreads的数据集进行的全面实验表明，精心选择的文本片段可以实现最佳性能，甚至优于ChatGPT生成的用户档案。

    Recommender systems are most successful for popular items and users with ample interactions (likes, ratings etc.). This work addresses the difficult and underexplored case of supporting users who have very sparse interactions but post informative review texts. Our experimental studies address two book communities with these characteristics. We design a framework with Transformer-based representation learning, covering user-item interactions, item content, and user-provided reviews. To overcome interaction sparseness, we devise techniques for selecting the most informative cues to construct concise user profiles. Comprehensive experiments, with datasets from Amazon and Goodreads, show that judicious selection of text snippets achieves the best performance, even in comparison to ChatGPT-generated user profiles.
    
[^3]: VM-Rec：一种用于冷启动用户推荐的变分映射方法

    VM-Rec: A Variational Mapping Approach for Cold-start User Recommendation. (arXiv:2311.01304v1 [cs.IR])

    [http://arxiv.org/abs/2311.01304](http://arxiv.org/abs/2311.01304)

    VM-Rec是一种用于解决冷启动用户推荐问题的变分映射方法，该方法基于生成表达能力强的嵌入的现有用户的互动，从而模拟生成冷启动用户的嵌入过程。

    

    冷启动问题是大多数推荐系统面临的共同挑战。传统的推荐模型在冷启动用户的互动非常有限时通常难以生成具有足够表达能力的嵌入。此外，缺乏用户的辅助内容信息加剧了挑战的存在，使得大多数冷启动方法难以应用。为了解决这个问题，我们观察到，如果模型能够为相对更多互动的现有用户生成具有表达能力的嵌入，这些用户最初也是冷启动用户，那么我们可以建立一个从少量初始互动到具有表达能力的嵌入的映射，模拟为冷启动用户生成嵌入的过程。基于这个观察，我们提出了一种变分映射方法用于冷启动用户推荐（VM-Rec）。首先，我们根据冷启动用户的初始互动生成个性化的映射函数，并进行参数优化。

    The cold-start problem is a common challenge for most recommender systems. With extremely limited interactions of cold-start users, conventional recommender models often struggle to generate embeddings with sufficient expressivity. Moreover, the absence of auxiliary content information of users exacerbates the presence of challenges, rendering most cold-start methods difficult to apply. To address this issue, our motivation is based on the observation that if a model can generate expressive embeddings for existing users with relatively more interactions, who were also initially cold-start users, then we can establish a mapping from few initial interactions to expressive embeddings, simulating the process of generating embeddings for cold-start users. Based on this motivation, we propose a Variational Mapping approach for cold-start user Recommendation (VM-Rec). Firstly, we generate a personalized mapping function for cold-start users based on their initial interactions, and parameters 
    
[^4]: 使用前向索引和轻量级编码器的高效神经排名

    Efficient Neural Ranking using Forward Indexes and Lightweight Encoders. (arXiv:2311.01263v1 [cs.IR])

    [http://arxiv.org/abs/2311.01263](http://arxiv.org/abs/2311.01263)

    使用双编码器模型的语义匹配能力和快速前向索引，提高了神经排名的效率和延迟。同时，通过预计算表示和优化索引尺寸，进一步改善了计算效率和资源消耗。

    

    基于双编码器的密集检索模型已经成为信息检索领域的标准。它们采用了大型的基于Transformer的语言模型，但这些模型在资源和延迟方面效率低下。我们提出了快速前向索引——利用双编码器模型的语义匹配能力进行高效和有效的重新排名。我们的框架可以在非常高的检索深度下进行重新排名，并通过分数插值结合了词汇匹配和语义匹配的优点。此外，为了减轻双编码器的局限性，我们解决了两个主要挑战：首先，通过预计算表示、避免不必要的计算或降低编码器的复杂度，提高了计算效率，降低了排名的资源消耗和延迟。其次，我们优化了索引的内存占用和维护成本；我们提出了两种互补的技术来减小索引的尺寸。

    Dual-encoder-based dense retrieval models have become the standard in IR. They employ large Transformer-based language models, which are notoriously inefficient in terms of resources and latency. We propose Fast-Forward indexes -- vector forward indexes which exploit the semantic matching capabilities of dual-encoder models for efficient and effective re-ranking. Our framework enables re-ranking at very high retrieval depths and combines the merits of both lexical and semantic matching via score interpolation. Furthermore, in order to mitigate the limitations of dual-encoders, we tackle two main challenges: Firstly, we improve computational efficiency by either pre-computing representations, avoiding unnecessary computations altogether, or reducing the complexity of encoders. This allows us to considerably improve ranking efficiency and latency. Secondly, we optimize the memory footprint and maintenance cost of indexes; we propose two complementary techniques to reduce the index size a
    
[^5]: 使用AI副驾驶员导航复杂搜索任务

    Navigating Complex Search Tasks with AI Copilots. (arXiv:2311.01235v1 [cs.IR])

    [http://arxiv.org/abs/2311.01235](http://arxiv.org/abs/2311.01235)

    该论文介绍了使用AI副驾驶员来导航复杂搜索任务，并探讨了生成AI和辅助代理的出现对于支持复杂搜索任务的潜力和重要性。

    

    正如信息检索(IR)研究界的许多人所知和欣赏的那样，搜索远未解决。每天都有数百万人在搜索引擎上面对任务的困难。他们的困难通常与任务的内在复杂性以及搜索系统无法完全理解任务和提供相关结果有关。任务激发了搜索，创建了搜索者尝试连接/解决的差距/问题情况，并在他们处理不同任务方面时驱动搜索行为。复杂搜索任务需要的不仅是基本事实查找或搜索的支持。支持复杂任务的方法研究包括生成查询和网站建议，个性化和上下文化搜索，以及开发新的搜索体验，包括跨时间和空间。最近兴起的生成人工智能(AI)和基于该技术的辅助代理，或者说副驾驶员，的出现。

    As many of us in the information retrieval (IR) research community know and appreciate, search is far from being a solved problem. Millions of people struggle with tasks on search engines every day. Often, their struggles relate to the intrinsic complexity of their task and the failure of search systems to fully understand the task and serve relevant results. The task motivates the search, creating the gap/problematic situation that searchers attempt to bridge/resolve and drives search behavior as they work through different task facets. Complex search tasks require more than support for rudimentary fact finding or re-finding. Research on methods to support complex tasks includes work on generating query and website suggestions, personalizing and contextualizing search, and developing new search experiences, including those that span time and space. The recent emergence of generative artificial intelligence (AI) and the arrival of assistive agents, or copilots, based on this technology
    
[^6]: 对于基于会话的推荐的双偏好学习异构超图网络

    Bi-Preference Learning Heterogeneous Hypergraph Networks for Session-based Recommendation. (arXiv:2311.01125v1 [cs.IR])

    [http://arxiv.org/abs/2311.01125](http://arxiv.org/abs/2311.01125)

    这篇论文提出了一种双偏好学习异构超图网络（BiPNet）的方法来解决基于会话的推荐中的价格偏好问题。这个方法考虑了价格偏好与兴趣偏好的互相影响，从而提高了推荐的准确性和个性化程度。

    

    基于会话的推荐旨在基于匿名行为序列预测下一个购买的物品。许多经济研究表明，物品价格是影响用户购买决策的关键因素。然而，现有的基于会话的推荐方法只关注捕捉用户的兴趣偏好，忽略了用户的价格偏好。实际上，有两个主要的挑战阻碍我们获取价格偏好。首先，价格偏好与各种物品特征（即类别和品牌）密切相关，这要求我们从异构信息中挖掘价格偏好。其次，价格偏好和兴趣偏好是相互依赖的，共同决定用户的选择，这要求我们同时考虑价格和兴趣偏好进行意图建模。为了应对上述挑战，我们提出了一种新颖的基于会话的推荐方法——双偏好学习异构超图网络（BiPNet）。

    Session-based recommendation intends to predict next purchased items based on anonymous behavior sequences. Numerous economic studies have revealed that item price is a key factor influencing user purchase decisions. Unfortunately, existing methods for session-based recommendation only aim at capturing user interest preference, while ignoring user price preference. Actually, there are primarily two challenges preventing us from accessing price preference. Firstly, the price preference is highly associated to various item features (i.e., category and brand), which asks us to mine price preference from heterogeneous information. Secondly, price preference and interest preference are interdependent and collectively determine user choice, necessitating that we jointly consider both price and interest preference for intent modeling. To handle above challenges, we propose a novel approach Bi-Preference Learning Heterogeneous Hypergraph Networks (BiPNet) for session-based recommendation. Spec
    
[^7]: 合作与转换：将物品转换转化为多查询自注意力进行序列推荐

    Collaboration and Transition: Distilling Item Transitions into Multi-Query Self-Attention for Sequential Recommendation. (arXiv:2311.01056v1 [cs.IR])

    [http://arxiv.org/abs/2311.01056](http://arxiv.org/abs/2311.01056)

    这篇论文提出了一种新的推荐系统方法，利用多查询自注意力和过渡感知嵌入蒸馏来捕捉用户交互序列中的合作和过渡信号。

    

    现代推荐系统使用各种顺序模块，如自注意力来学习动态用户兴趣。然而，这些方法在捕捉用户交互序列中的合作和过渡信号方面效果较差。为了克服这些限制，我们提出了一种新方法，称为多查询自注意力与过渡感知嵌入蒸馏（MQSA-TED）。首先，我们提出了一个$L$-查询自注意力模块，使用灵活的窗口大小作为注意力查询来捕捉合作信号。此外，我们还引入了一种多查询自注意力方法，通过结合长查询和短查询来平衡建模用户偏好的偏差-方差权衡。

    Modern recommender systems employ various sequential modules such as self-attention to learn dynamic user interests. However, these methods are less effective in capturing collaborative and transitional signals within user interaction sequences. First, the self-attention architecture uses the embedding of a single item as the attention query, which is inherently challenging to capture collaborative signals. Second, these methods typically follow an auto-regressive framework, which is unable to learn global item transition patterns. To overcome these limitations, we propose a new method called Multi-Query Self-Attention with Transition-Aware Embedding Distillation (MQSA-TED). First, we propose an $L$-query self-attention module that employs flexible window sizes for attention queries to capture collaborative signals. In addition, we introduce a multi-query self-attention method that balances the bias-variance trade-off in modeling user preferences by combining long and short-query self-
    
[^8]: 评估个体项公平性的推荐系统评估指标：一项关键研究

    Evaluation Measures of Individual Item Fairness for Recommender Systems: A Critical Study. (arXiv:2311.01013v1 [cs.IR])

    [http://arxiv.org/abs/2311.01013](http://arxiv.org/abs/2311.01013)

    本研究对推荐系统中个体项的公平性评估指标进行了关键研究，并指出了现有指标存在的限制和问题。通过重新定义和纠正这些指标，或者解释为何某些限制无法解决，我们提出了改进的方法。此外，我们还进行了全面的实证分析，验证了这些改进的有效性。

    

    公平性是推荐系统中的一个新兴且具有挑战性的话题。近年来，出现了各种评估和改善公平性的方式。本研究对推荐系统中现有的公平性评估指标进行了研究。具体来说，我们仅关注个体项的曝光度公平性评估指标，旨在量化个体项在向用户推荐时的差异，与用户对项的相关性无关。我们收集了所有这些指标，并对它们的理论属性进行了批判性分析。我们发现每个指标都存在一系列限制，这些限制可能使得受影响的指标难以解释、计算或用于比较推荐。我们通过重新定义或纠正受影响的指标来解决这些限制，或者我们解释了为什么某些限制无法解决。我们还对这些公平性指标的原始版本和我们纠正后的版本进行了全面的实证分析。

    Fairness is an emerging and challenging topic in recommender systems. In recent years, various ways of evaluating and therefore improving fairness have emerged. In this study, we examine existing evaluation measures of fairness in recommender systems. Specifically, we focus solely on exposure-based fairness measures of individual items that aim to quantify the disparity in how individual items are recommended to users, separate from item relevance to users. We gather all such measures and we critically analyse their theoretical properties. We identify a series of limitations in each of them, which collectively may render the affected measures hard or impossible to interpret, to compute, or to use for comparing recommendations. We resolve these limitations by redefining or correcting the affected measures, or we argue why certain limitations cannot be resolved. We further perform a comprehensive empirical analysis of both the original and our corrected versions of these fairness measure
    
[^9]: 基于学术异构信息网络表示学习的研究团队识别

    Research Team Identification Based on Representation Learning of Academic Heterogeneous Information Network. (arXiv:2311.00922v1 [cs.IR])

    [http://arxiv.org/abs/2311.00922](http://arxiv.org/abs/2311.00922)

    本文提出了一种基于学术异构信息网络表示学习的科研团队识别方法，通过利用节点级和元路径级的注意机制学习低维稠密实值向量表示，以有效识别和发现学术网络中的科研团队。

    

    现实世界中的学术网络通常可以由由多类型节点和关系组成的异构信息网络来描述。现有关于同构信息网络的表示学习方法缺乏对异构信息网络的探索能力，无法应用于异构信息网络。针对从由庞大复杂的科技大数据组成的学术异构信息网络中有效识别和发现科研团队的实际需求，本文提出了一种基于学术异构信息网络表示学习的科研团队识别方法。该方法利用节点级和元路径级的注意机制，在保留网络中节点的丰富拓扑信息和语义信息的基础上，学习低维稠密实值向量表示。

    Academic networks in the real world can usually be described by heterogeneous information networks composed of multi-type nodes and relationships. Some existing research on representation learning for homogeneous information networks lacks the ability to explore heterogeneous information networks in heterogeneous information networks. It cannot be applied to heterogeneous information networks. Aiming at the practical needs of effectively identifying and discovering scientific research teams from the academic heterogeneous information network composed of massive and complex scientific and technological big data, this paper proposes a scientific research team identification method based on representation learning of academic heterogeneous information networks. The attention mechanism at node level and meta-path level learns low-dimensional, dense and real-valued vector representations on the basis of retaining the rich topological information of nodes in the network and the semantic info
    
[^10]: 人类文化受进化限制吗？

    Is Human Culture Locked by Evolution?. (arXiv:2311.00719v1 [cs.SI])

    [http://arxiv.org/abs/2311.00719](http://arxiv.org/abs/2311.00719)

    本文分析了一个名为ZeroMat的算法的社会影响，表明人类文化受进化限制，个体的文化品味可以在没有历史数据的情况下以高精度预测。同时提供了解决方案和对中国政府法规和政策的解释。

    

    人类文化已经演化了数千年，在互联网时代蓬勃发展。由于大数据的可用性，我们可以通过分析诸如MovieLens和豆瓣等网站上的用户项目评分值等表示来研究人类文化。工业工人已经将推荐系统应用于大数据以预测用户行为并促进网络流量。在本文中，我们分析了名为ZeroMat的算法的社会影响，以显示人类文化处于一种状态中，其中个人的文化品味可以在没有历史数据的情况下以高精度预测。我们还提供了解决这个问题的解决方案，并解释了当前中国政府的法规和政策。

    Human culture has evolved for thousands of years and thrived in the era of Internet. Due to the availability of big data, we could do research on human culture by analyzing its representation such as user item rating values on websites like MovieLens and Douban. Industrial workers have applied recommender systems in big data to predict user behavior and promote web traffic. In this paper, we analyze the social impact of an algorithm named ZeroMat to show that human culture is locked into a state where individual's cultural taste is predictable at high precision without historic data. We also provide solutions to this problem and interpretation of current Chinese government's regulations and policies.
    
[^11]: 利用大型语言模型（LLMs）增强基于内容的推荐的免训练数据集压缩

    Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation. (arXiv:2310.09874v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.09874](http://arxiv.org/abs/2310.09874)

    本文利用大型语言模型（LLMs）来增强基于内容的推荐中的免训练数据集压缩方法，旨在通过生成文本内容来合成一个小而信息丰富的数据集，使得模型能够达到与在大型数据集上训练的模型相当的性能。

    

    现代内容推荐（CBR）技术利用物品的内容信息为用户提供个性化服务，但在大型数据集上的资源密集型训练存在问题。为解决这个问题，本文探讨了对文本CBR进行数据集压缩的方法。数据集压缩的目标是合成一个小且信息丰富的数据集，使模型性能可以与在大型数据集上训练的模型相媲美。现有的压缩方法针对连续数据（如图像或嵌入向量）的分类任务而设计，直接应用于CBR存在局限性。为了弥补这一差距，我们研究了基于内容的推荐中高效的数据集压缩方法。受到大型语言模型（LLMs）在文本理解和生成方面出色的能力的启发，我们利用LLMs在数据集压缩期间生成文本内容。为了处理涉及用户和物品的交互数据，我们设计了一个双...

    Modern techniques in Content-based Recommendation (CBR) leverage item content information to provide personalized services to users, but suffer from resource-intensive training on large datasets. To address this issue, we explore the dataset condensation for textual CBR in this paper. The goal of dataset condensation is to synthesize a small yet informative dataset, upon which models can achieve performance comparable to those trained on large datasets. While existing condensation approaches are tailored to classification tasks for continuous data like images or embeddings, direct application of them to CBR has limitations. To bridge this gap, we investigate efficient dataset condensation for content-based recommendation. Inspired by the remarkable abilities of large language models (LLMs) in text comprehension and generation, we leverage LLMs to empower the generation of textual content during condensation. To handle the interaction data involving both users and items, we devise a dua
    
[^12]: DiskANN++: 使用查询敏感的入口顶点在同构映射图索引上进行高效基于页面的搜索

    DiskANN++: Efficient Page-based Search over Isomorphic Mapped Graph Index using Query-sensitivity Entry Vertex. (arXiv:2310.00402v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.00402](http://arxiv.org/abs/2310.00402)

    提出了一个优化的DiskANN++方法，使用查询敏感的入口顶点在同构映射图索引上进行高效基于页面的搜索。

    

    给定一个向量数据集X和一个查询向量xq，基于图的近似最近邻搜索(ANNS)旨在构建一个图索引G，并通过在G上搜索来近似返回与xq的最小距离向量。基于图的ANNS的主要缺点是图索引太大，无法适应尤其是大规模X的内存。为了解决这个问题，提出了一种基于产品量化(PQ)的混合方法DiskANN，它在内存中存储低维度的PQ索引，并在SSD中保留图索引，从而减小内存开销同时确保高搜索准确性。然而，它存在两个重要的I/O问题，严重影响了整体效率：(1)从入口顶点到查询邻域的长路径导致大量的I/O请求，以及(2)在路由过程中的冗余I/O请求。为了解决上述问题，我们提出了优化的DiskANN++。

    Given a vector dataset $\mathcal{X}$ and a query vector $\vec{x}_q$, graph-based Approximate Nearest Neighbor Search (ANNS) aims to build a graph index $G$ and approximately return vectors with minimum distances to $\vec{x}_q$ by searching over $G$. The main drawback of graph-based ANNS is that a graph index would be too large to fit into the memory especially for a large-scale $\mathcal{X}$. To solve this, a Product Quantization (PQ)-based hybrid method called DiskANN is proposed to store a low-dimensional PQ index in memory and retain a graph index in SSD, thus reducing memory overhead while ensuring a high search accuracy. However, it suffers from two I/O issues that significantly affect the overall efficiency: (1) long routing path from an entry vertex to the query's neighborhood that results in large number of I/O requests and (2) redundant I/O requests during the routing process. We propose an optimized DiskANN++ to overcome above issues. Specifically, for the first issue, we pre
    
[^13]: 具有流行度偏见的排名：自增强动态下的用户福利

    Ranking with Popularity Bias: User Welfare under Self-Amplification Dynamics. (arXiv:2305.18333v1 [cs.IR])

    [http://arxiv.org/abs/2305.18333](http://arxiv.org/abs/2305.18333)

    研究了物品流行度、质量和位置偏差对用户福利的影响，提出了通过探索减轻流行度偏见负面影响的算法。

    

    虽然已经确认流行度偏见在推荐（和其他基于排名的）系统中发挥作用，但其对用户福利的影响的详细分析仍然缺乏。我们提出了一种通用机制，通过它，物品的流行度、质量和位置偏差可以影响用户选择，并且可以负面影响各种推荐策略的集体用户效用。我们将问题表述为非平稳上下文脱靶机，强调不是为了消除流行度偏见而是为了减轻其负面影响而进行探索的重要性。首先，普通的有流行度偏差的推荐系统会通过混淆物品质量和流行度而引发线性遗憾。更一般地，我们展示了即使在线性设置下，由于流行度偏见的混淆效应，物品质量的可识别性也可能无法实现。然而，在足够变异的假设下，我们开发了一种高效的类UCB算法，并证明了有效的遗憾保证。我们通过实验验证了我们提出的算法的有效性，并证实了流行度偏见的负面影响。

    While popularity bias is recognized to play a role in recommmender (and other ranking-based) systems, detailed analyses of its impact on user welfare have largely been lacking. We propose a general mechanism by which item popularity, item quality, and position bias can impact user choice, and how it can negatively impact the collective user utility of various recommender policies. Formulating the problem as a non-stationary contextual bandit, we highlight the importance of exploration, not to eliminate popularity bias, but to mitigate its negative effects. First, naive popularity-biased recommenders are shown to induce linear regret by conflating item quality and popularity. More generally, we show that, even in linear settings, identifiability of item quality may not be possible due to the confounding effects of popularity bias. However, under sufficient variability assumptions, we develop an efficient UCB-style algorithm and prove efficient regret guarantees. We complement our analys
    
[^14]: 使用RPRS的高效有效的基于Transformer的重新排序器处理极长查询和文档的检索

    Retrieval for Extremely Long Queries and Documents with RPRS: a Highly Efficient and Effective Transformer-based Re-Ranker. (arXiv:2303.01200v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2303.01200](http://arxiv.org/abs/2303.01200)

    该论文提出了一种基于新型比例相关度分数（RPRS）的高效有效的基于Transformer的重新排序器，用于处理极长查询和文档的检索任务。与先前的工作相比，在五个不同数据集上进行的广泛评估显示RPRS获得了显著更好的结果。此外，RPRS具有高效性，并且解决了QBD检索任务中低资源训练的问题。

    

    在信息检索中，使用极长查询和文档进行检索是一个众所周知且具有挑战性的任务，通常称为查询-文档（QBD）检索。先前的工作中，专门设计用于处理长输入序列的Transformer模型在QBD任务中并没有展现出很高的效果。我们提出了一种基于新型比例相关度分数（RPRS）的重新排序器，用于计算查询与前k个候选文档之间的相关度分数。我们进行了广泛的评估，结果显示RPRS在五个不同数据集上比现有模型取得了显著更好的结果。此外，RPRS非常高效，因为在查询时间之前可以对所有文档进行预处理、嵌入和索引，使得我们的重新排序器具有O(N)的复杂度，其中N是查询和候选文档中句子的总数。此外，我们的方法解决了QBD检索任务中低资源训练的问题，因为它不需要大量的训练数据。

    Retrieval with extremely long queries and documents is a well-known and challenging task in information retrieval and is commonly known as Query-by-Document (QBD) retrieval. Specifically designed Transformer models that can handle long input sequences have not shown high effectiveness in QBD tasks in previous work. We propose a Re-Ranker based on the novel Proportional Relevance Score (RPRS) to compute the relevance score between a query and the top-k candidate documents. Our extensive evaluation shows RPRS obtains significantly better results than the state-of-the-art models on five different datasets. Furthermore, RPRS is highly efficient since all documents can be pre-processed, embedded, and indexed before query time which gives our re-ranker the advantage of having a complexity of O(N) where N is the total number of sentences in the query and candidate documents. Furthermore, our method solves the problem of the low-resource training in QBD retrieval tasks as it does not need larg
    

