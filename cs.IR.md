# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Co-evolving Vector Quantization for ID-based Recommendation.](http://arxiv.org/abs/2308.16761) | 这项工作提出了一种用于基于ID的推荐的共同演化向量量化框架（COVE），该框架能够自动学习和生成不同粒度级别下的实体分类信息，并在各种推荐任务中展现了有效性。 |
| [^2] | [Towards Better Query Classification with Multi-Expert Knowledge Condensation in JD Ads Search.](http://arxiv.org/abs/2308.01098) | 本文提出了一种知识蒸馏框架（KC），通过在严格的低延迟约束下提升在线FastText模型的查询分类性能，在京东广告搜索中取得了显著的性能提升。 |
| [^3] | [Perspectives on Large Language Models for Relevance Judgment.](http://arxiv.org/abs/2304.09161) | 本文讨论了LLMs协助人类专家进行相关性判断的可能方法和问题，制定了人机协作谱系，提供了一个基于LLM的相关性判断与经过训练的人类评估者判断相关性的初步实验，以及支持和反对使用LLMs进行自动相关性判断的两个对立观点以及妥协的观点。 |
| [^4] | [Talk the Walk: Synthetic Data Generation for Conversational Music Recommendation.](http://arxiv.org/abs/2301.11489) | TalkTheWalk 是一种新技术，通过利用精心策划的项目收藏中的领域专业知识来合成逼真高质量的会话数据，解决了构建会话式推荐系统所需的训练数据收集的困难。 |

# 详细

[^1]: 基于ID的推荐的共同演化向量量化

    Co-evolving Vector Quantization for ID-based Recommendation. (arXiv:2308.16761v1 [cs.IR])

    [http://arxiv.org/abs/2308.16761](http://arxiv.org/abs/2308.16761)

    这项工作提出了一种用于基于ID的推荐的共同演化向量量化框架（COVE），该框架能够自动学习和生成不同粒度级别下的实体分类信息，并在各种推荐任务中展现了有效性。

    

    类别信息对于提高推荐的质量和个性化起着至关重要的作用。然而，在基于ID的推荐中，项目类别信息的可用性并不一致。在这项工作中，我们提出了一种替代方法，以自动学习和生成实体（即用户和项目）在不同粒度级别上的分类信息，特别适用于基于ID的推荐。具体而言，我们设计了一个共同演化向量量化框架，即COVE，它能够同时学习和改进代码表示和实体嵌入，并以从随机初始化状态开始的端到端方式进行。通过其高度适应性，COVE可以轻松集成到现有的推荐模型中。我们验证了COVE在各种推荐任务中的有效性，包括列表完成、协同过滤和点击率预测，涵盖不同的推荐场景。

    Category information plays a crucial role in enhancing the quality and personalization of recommendations. Nevertheless, the availability of item category information is not consistently present, particularly in the context of ID-based recommendations. In this work, we propose an alternative approach to automatically learn and generate entity (i.e., user and item) categorical information at different levels of granularity, specifically for ID-based recommendation. Specifically, we devise a co-evolving vector quantization framework, namely COVE, which enables the simultaneous learning and refinement of code representation and entity embedding in an end-to-end manner, starting from the randomly initialized states. With its high adaptability, COVE can be easily integrated into existing recommendation models. We validate the effectiveness of COVE on various recommendation tasks including list completion, collaborative filtering, and click-through rate prediction, across different recommend
    
[^2]: 在京东广告搜索中利用多专家知识蒸馏实现更好的查询分类

    Towards Better Query Classification with Multi-Expert Knowledge Condensation in JD Ads Search. (arXiv:2308.01098v1 [cs.IR])

    [http://arxiv.org/abs/2308.01098](http://arxiv.org/abs/2308.01098)

    本文提出了一种知识蒸馏框架（KC），通过在严格的低延迟约束下提升在线FastText模型的查询分类性能，在京东广告搜索中取得了显著的性能提升。

    

    查询分类作为理解用户意图的有效方法，在现实世界的在线广告系统中具有重要意义。为了确保更低的延迟，常使用浅层模型（如FastText）进行高效的在线推断。然而，FastText模型的表征能力不足，导致分类性能较差，特别是在一些低频查询和尾部类别上。使用更深入且更复杂的模型（如BERT）是一种有效的解决方案，但它将导致更高的在线推断延迟和更昂贵的计算成本。因此，如何在推断效率和分类性能之间折衷显然具有重大实际意义。为了克服这个挑战，在本文中，我们提出了知识蒸馏（KC），一个简单而有效的知识蒸馏框架，以在严格的低延迟约束下提升在线FastText模型的分类性能。具体来说，我们提出了训练一个离线模型，通过蒸馏知识来改善在线模型的分类性能。

    Search query classification, as an effective way to understand user intents, is of great importance in real-world online ads systems. To ensure a lower latency, a shallow model (e.g. FastText) is widely used for efficient online inference. However, the representation ability of the FastText model is insufficient, resulting in poor classification performance, especially on some low-frequency queries and tailed categories. Using a deeper and more complex model (e.g. BERT) is an effective solution, but it will cause a higher online inference latency and more expensive computing costs. Thus, how to juggle both inference efficiency and classification performance is obviously of great practical importance. To overcome this challenge, in this paper, we propose knowledge condensation (KC), a simple yet effective knowledge distillation framework to boost the classification performance of the online FastText model under strict low latency constraints. Specifically, we propose to train an offline
    
[^3]: 大型语言模型在相关性评价中的应用

    Perspectives on Large Language Models for Relevance Judgment. (arXiv:2304.09161v1 [cs.IR])

    [http://arxiv.org/abs/2304.09161](http://arxiv.org/abs/2304.09161)

    本文讨论了LLMs协助人类专家进行相关性判断的可能方法和问题，制定了人机协作谱系，提供了一个基于LLM的相关性判断与经过训练的人类评估者判断相关性的初步实验，以及支持和反对使用LLMs进行自动相关性判断的两个对立观点以及妥协的观点。

    

    当被问及时，像ChatGPT这样的当前大型语言模型（LLMs）声称它们可以协助我们进行相关性判断。许多研究人员认为这不会导致可信的信息检索研究。在本文中，我们讨论了LLMs协助人类专家进行相关性判断的可能方法以及可能出现的问题和关注点。我们制定了一个人机协作谱系，可以将不同的相关性判断策略进行分类，基于人类对机器的依赖程度。针对“完全自动化评估”的极端点，我们进一步进行了基于LLM的相关性判断与经过训练的人类评估者判断的相关性的初步实验。我们通过分析文献、我们的初步实验证据以及我们作为信息检索研究人员的经验，提出了支持和反对使用LLMs进行自动相关性判断的两个对立观点以及妥协的观点。我们希望开始进行建设性的讨论。

    When asked, current large language models (LLMs) like ChatGPT claim that they can assist us with relevance judgments. Many researchers think this would not lead to credible IR research. In this perspective paper, we discuss possible ways for LLMs to assist human experts along with concerns and issues that arise. We devise a human-machine collaboration spectrum that allows categorizing different relevance judgment strategies, based on how much the human relies on the machine. For the extreme point of "fully automated assessment", we further include a pilot experiment on whether LLM-based relevance judgments correlate with judgments from trained human assessors. We conclude the paper by providing two opposing perspectives - for and against the use of LLMs for automatic relevance judgments - and a compromise perspective, informed by our analyses of the literature, our preliminary experimental evidence, and our experience as IR researchers.  We hope to start a constructive discussion withi
    
[^4]: Talk the Walk: 针对会话式音乐推荐的合成数据生成

    Talk the Walk: Synthetic Data Generation for Conversational Music Recommendation. (arXiv:2301.11489v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2301.11489](http://arxiv.org/abs/2301.11489)

    TalkTheWalk 是一种新技术，通过利用精心策划的项目收藏中的领域专业知识来合成逼真高质量的会话数据，解决了构建会话式推荐系统所需的训练数据收集的困难。

    

    推荐系统广泛存在，但用户往往很难在推荐质量较差时进行控制和调整。这促使了会话式推荐系统(CRSs)的发展，通过自然语言反馈提供对推荐的控制。然而，构建会话式推荐系统需要包含用户话语和涵盖多样化偏好范围的项目的会话训练数据。使用传统方法如众包，这样的数据收集起来非常困难。我们在项目集推荐的背景下解决了这个问题，注意到这个任务受到越来越多关注，动机在于音乐、新闻和食谱推荐等使用案例。我们提出了一种新技术TalkTheWalk，通过利用广泛可获得的精心策划的项目收藏中的领域专业知识来合成逼真高质量的会话数据，并展示了如何将其转化为相应的项目集策划。

    Recommendation systems are ubiquitous yet often difficult for users to control and adjust when recommendation quality is poor. This has motivated the development of conversational recommendation systems (CRSs), with control over recommendations provided through natural language feedback. However, building conversational recommendation systems requires conversational training data involving user utterances paired with items that cover a diverse range of preferences. Such data has proved challenging to collect scalably using conventional methods like crowdsourcing. We address it in the context of item-set recommendation, noting the increasing attention to this task motivated by use cases like music, news and recipe recommendation. We present a new technique, TalkTheWalk, that synthesizes realistic high-quality conversational data by leveraging domain expertise encoded in widely available curated item collections, showing how these can be transformed into corresponding item set curation c
    

