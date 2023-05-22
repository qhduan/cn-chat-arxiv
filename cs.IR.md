# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Does Generative Retrieval Scale to Millions of Passages?.](http://arxiv.org/abs/2305.11841) | 本研究对不同语料规模下的生成式检索技术进行了实证研究，扩展到包含8.8M篇章的MS MARCO检索任务，并评估了高达11B个参数的模型大小，揭示出了在索引期间使用合成查询作为文档表示的重要性。 |
| [^2] | [Visualization for Recommendation Explainability: A Survey and New Perspectives.](http://arxiv.org/abs/2305.11755) | 本文回顾了推荐系统中有关可视化解释的研究，提出了一组可能有益于设计解释性可视化推荐系统的指南。 |
| [^3] | [Inference-time Re-ranker Relevance Feedback for Neural Information Retrieval.](http://arxiv.org/abs/2305.11744) | 本文提出了一种利用重排器提供推理时间反馈来改进检索的方法，可以显著提高低召回率@ K下的检索性能。 |
| [^4] | [Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights.](http://arxiv.org/abs/2305.11700) | 本论文探究了在基于文本的协同过滤中使用大型语言模型所能带来的性能提升，并揭示了TCF程序扩展的极限。研究人员比较了使用不同大小的语言模型在基于文本的协同过滤算法中的性能表现。 |
| [^5] | [The Barriers to Online Clothing Websites for Visually Impaired People: An Interview and Observation Approach to Understanding Needs.](http://arxiv.org/abs/2305.11559) | 本研究揭示了视障人士网购服装时存在的问题和需求，需要开发提高可访问性、用户友好的服装网站。 |
| [^6] | [InstructIE: A Chinese Instruction-based Information Extraction Dataset.](http://arxiv.org/abs/2305.11527) | 介绍了一份中文的基于指令的信息提取数据集InstructIE，其中包括了270,000个弱监督的数据和1,000个高质量注释实例。实验结果表明当前的模型表现有待改进，该任务仍存在挑战。 |
| [^7] | [Recouple Event Field via Probabilistic Bias for Event Extraction.](http://arxiv.org/abs/2305.11498) | 提出了一种基于概率偏置的重新耦合事件场模型（ProCE），用于增强事件提取框架，以澄清来自模糊纠缠的事件字段，并重新耦合相应的澄清分布以捕获更多潜在信息字段。实验表明该方法有效且具有泛化性。 |
| [^8] | [TELeR: A General Taxonomy of LLM Prompts for Benchmarking Complex Tasks.](http://arxiv.org/abs/2305.11430) | 本文提出了一个通用分类法，可以用来设计具有特定属性的提示来执行各种复杂任务，从而解决了LLM在执行复杂任务方面的性能变异巨大的问题。 |
| [^9] | [Online Learning in a Creator Economy.](http://arxiv.org/abs/2305.11381) | 本文探讨了如何应用在线学习的方法优化创作者经济学中的平台收益，分析和比较了基于回报的和基于特征的两种合同类型。 |
| [^10] | [One Model for All Domains: Collaborative Domain-Prefix Tuning for Cross-Domain NER.](http://arxiv.org/abs/2301.10410) | 本论文提出了基于协作域前缀调整的跨领域实体识别，使用文本到文本生成的支撑领域相关指导来将知识转移至新域NER任务，避免了先前的为每个领域结束一个全新的NER模型的问题。 |

# 详细

[^1]: 生成式检索如何扩展到数百万篇文章？

    How Does Generative Retrieval Scale to Millions of Passages?. (arXiv:2305.11841v1 [cs.IR])

    [http://arxiv.org/abs/2305.11841](http://arxiv.org/abs/2305.11841)

    本研究对不同语料规模下的生成式检索技术进行了实证研究，扩展到包含8.8M篇章的MS MARCO检索任务，并评估了高达11B个参数的模型大小，揭示出了在索引期间使用合成查询作为文档表示的重要性。

    

    由可微搜索索引(Differentiable Search Index)推广而来的生成式检索的新兴范式,将经典的信息检索问题重新构造为序列到序列的建模任务，放弃了外部索引，并将整个文档语料库编码在单个Transformer中。虽然已经提出了许多不同的方法来提高生成式检索的有效性，但它们仅在约100k的文档语料库上进行了评估。我们进行了第一次根据不同的语料规模进行生成式检索技术的实证研究，最终扩展到整个8.8M篇章的MS MARCO检索任务，并评估了高达11B个参数的模型大小。我们揭示了关于将生成式检索扩展到数百万篇文章的几个发现，特别是索引期间使用合成查询作为文档表示的核心重要性，以及考虑并非现有的建议架构修改时的无效性。

    Popularized by the Differentiable Search Index, the emerging paradigm of generative retrieval re-frames the classic information retrieval problem into a sequence-to-sequence modeling task, forgoing external indices and encoding an entire document corpus within a single Transformer. Although many different approaches have been proposed to improve the effectiveness of generative retrieval, they have only been evaluated on document corpora on the order of 100k in size. We conduct the first empirical study of generative retrieval techniques across various corpus scales, ultimately scaling up to the entire MS MARCO passage ranking task with a corpus of 8.8M passages and evaluating model sizes up to 11B parameters. We uncover several findings about scaling generative retrieval to millions of passages; notably, the central importance of using synthetic queries as document representations during indexing, the ineffectiveness of existing proposed architecture modifications when accounting for c
    
[^2]: 推荐系统解释的可视化：综述和新视角

    Visualization for Recommendation Explainability: A Survey and New Perspectives. (arXiv:2305.11755v1 [cs.IR])

    [http://arxiv.org/abs/2305.11755](http://arxiv.org/abs/2305.11755)

    本文回顾了推荐系统中有关可视化解释的研究，提出了一组可能有益于设计解释性可视化推荐系统的指南。

    

    为推荐提供系统生成的解释是实现透明且值得信赖的推荐系统的重要步骤。可解释的推荐系统为输出提供了人类可理解的基础。在过去的20年中，可解释的推荐引起了推荐系统研究社区的广泛关注。本文旨在全面回顾推荐系统中有关可视化解释的研究工作。更具体地，我们根据解释目标、解释范围、解释样式和解释格式这四个维度系统地审查推荐系统中有关解释的文献。认识到可视化的重要性，我们从解释性视觉方式的角度途径推荐系统文献，即使用可视化作为解释的显示样式。因此，我们得出了一组可能有益于设计解释性可视化推荐系统的指南。

    Providing system-generated explanations for recommendations represents an important step towards transparent and trustworthy recommender systems. Explainable recommender systems provide a human-understandable rationale for their outputs. Over the last two decades, explainable recommendation has attracted much attention in the recommender systems research community. This paper aims to provide a comprehensive review of research efforts on visual explanation in recommender systems. More concretely, we systematically review the literature on explanations in recommender systems based on four dimensions, namely explanation goal, explanation scope, explanation style, and explanation format. Recognizing the importance of visualization, we approach the recommender system literature from the angle of explanatory visualizations, that is using visualizations as a display style of explanation. As a result, we derive a set of guidelines that might be constructive for designing explanatory visualizat
    
[^3]: 面向神经信息检索的推理时间重排反馈

    Inference-time Re-ranker Relevance Feedback for Neural Information Retrieval. (arXiv:2305.11744v1 [cs.IR])

    [http://arxiv.org/abs/2305.11744](http://arxiv.org/abs/2305.11744)

    本文提出了一种利用重排器提供推理时间反馈来改进检索的方法，可以显著提高低召回率@ K下的检索性能。

    

    神经信息检索通常采用检索和重排框架：先使用双编码器网络检索K（例如100）个候选项，然后再使用更强大的交叉编码器模型对这些候选项进行重新排序，以使更好的候选项排名更高。重排器通常产生比检索器更好的候选分数，但仅限于查看前K个检索到的候选项，因此无法提高检索性能（以Recall @ K为度量）。在本文中，我们利用重排器通过提供推理时间相关反馈来改进检索。具体而言，我们利用重排器的预测对测试实例的重要信息进行了检索器查询表示的更新。我们的方法可以通过轻量级的推理时间蒸馏来实现，目的是使检索器的候选分数更接近于重排器的分数。然后使用更新后的查询向量执行第二个检索步骤。通过实验证明，我们的方法可以显著提高检索性能，特别是在低召回率@ K下。

    Neural information retrieval often adopts a retrieve-and-rerank framework: a bi-encoder network first retrieves K (e.g., 100) candidates that are then re-ranked using a more powerful cross-encoder model to rank the better candidates higher. The re-ranker generally produces better candidate scores than the retriever, but is limited to seeing only the top K retrieved candidates, thus providing no improvements in retrieval performance as measured by Recall@K. In this work, we leverage the re-ranker to also improve retrieval by providing inference-time relevance feedback to the retriever. Concretely, we update the retriever's query representation for a test instance using a lightweight inference-time distillation of the re-ranker's prediction for that instance. The distillation loss is designed to bring the retriever's candidate scores closer to those of the re-ranker. A second retrieval step is then performed with the updated query vector. We empirically show that our approach, which can 
    
[^4]: 探究利用大型语言模型探索基于文本的协同过滤的极限：发现和认识

    Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights. (arXiv:2305.11700v1 [cs.IR])

    [http://arxiv.org/abs/2305.11700](http://arxiv.org/abs/2305.11700)

    本论文探究了在基于文本的协同过滤中使用大型语言模型所能带来的性能提升，并揭示了TCF程序扩展的极限。研究人员比较了使用不同大小的语言模型在基于文本的协同过滤算法中的性能表现。

    

    基于文本的协同过滤成为现今文本和新闻推荐的主流方法，利用文本编码器或语言模型(LMs)表示物品。然而，现有的文本协同过滤模型主要集中在使用中小型的LMs上，如果将物品编码器替换为最大最强大的1750亿参数的GPT-3模型，将会对推荐性能产生什么影响尚不确定。作者开展了一系列实验，探索TCF程序的性能极限。具体来说，作者将物品编码器规模从一亿扩大到一百亿以揭示TCF程序的扩展极限，同时还探究了使用超大LMs是否能实现推荐任务的通用物品表示方法。此外，作者比较了使用最强大的LMs和中等LMs实现的基于文本协同过滤的性能差异。

    Text-based collaborative filtering (TCF) has become the mainstream approach for text and news recommendation, utilizing text encoders, also known as language models (LMs), to represent items. However, existing TCF models primarily focus on using small or medium-sized LMs. It remains uncertain what impact replacing the item encoder with one of the largest and most powerful LMs, such as the 175-billion parameter GPT-3 model, would have on recommendation performance. Can we expect unprecedented results? To this end, we conduct an extensive series of experiments aimed at exploring the performance limits of the TCF paradigm. Specifically, we increase the size of item encoders from one hundred million to one hundred billion to reveal the scaling limits of the TCF paradigm. We then examine whether these extremely large LMs could enable a universal item representation for the recommendation task. Furthermore, we compare the performance of the TCF paradigm utilizing the most powerful LMs to the
    
[^5]: 视障人士网购服装的障碍：理解需求的访谈和观察方法

    The Barriers to Online Clothing Websites for Visually Impaired People: An Interview and Observation Approach to Understanding Needs. (arXiv:2305.11559v1 [cs.HC])

    [http://arxiv.org/abs/2305.11559](http://arxiv.org/abs/2305.11559)

    本研究揭示了视障人士网购服装时存在的问题和需求，需要开发提高可访问性、用户友好的服装网站。

    

    视障人士在日常生活中面临许多挑战，其中购物是最具挑战性的。许多人选择在线购物，但网购服装也存在许多限制和障碍。本文通过观察和访谈两种方法进行了研究，以填补了解视障人士在网上选购服装时的行为的空白。研究结果显示，购物网站的服装描述存在不准确、误导和矛盾的情况；视障人士主要依靠（不可靠的）搜索工具，并通过查看客户评论来检查产品描述；此外，视障人士对网购时接受他人帮助持谨慎态度，担心侵犯隐私、丧失独立性和信任问题。因此，我们需要开发可访问、用户友好的服装网站，以解决视障人士所面临的特定挑战。

    Visually impaired (VI) people often face challenges when performing everyday tasks and identify shopping for clothes as one of the most challenging. Many engage in online shopping, which eliminates some challenges of physical shopping. However, clothes shopping online suffers from many other limitations and barriers. More research is needed to address these challenges, and extant works often base their findings on interviews alone, providing only subjective, recall-biased information. We conducted two complementary studies using both observational and interview approaches to fill a gap in understanding about VI people's behaviour when selecting and purchasing clothes online. Our findings show that shopping websites suffer from inaccurate, misleading, and contradictory clothing descriptions; that VI people mainly rely on (unreliable) search tools and check product descriptions by reviewing customer comments. Our findings also indicate that VI people are hesitant to accept assistance fro
    
[^6]: InstructIE: 一份基于指令的中文信息提取数据集

    InstructIE: A Chinese Instruction-based Information Extraction Dataset. (arXiv:2305.11527v1 [cs.CL])

    [http://arxiv.org/abs/2305.11527](http://arxiv.org/abs/2305.11527)

    介绍了一份中文的基于指令的信息提取数据集InstructIE，其中包括了270,000个弱监督的数据和1,000个高质量注释实例。实验结果表明当前的模型表现有待改进，该任务仍存在挑战。

    

    我们引入了一项新的信息提取任务，称为基于指令的信息提取 (Instruction-based IE)，它旨在要求系统遵循特定的指令或指南来提取信息。为了促进该领域的研究，我们构建了一个数据集，称为InstructIE，其中包括来自中文维基百科的 270,000 个弱监督数据和 1,000 个高质量众包注释实例。我们进一步评估了各种基线模型在InstructIE数据集上的表现。结果表明，尽管当前的模型表现很有希望，但仍有改进的空间。此外，我们进行了全面的案例研究分析，强调了基于指令的信息提取任务中固有的挑战。代码和数据集可在 https://github.com/zjunlp/DeepKE/tree/main/example/llm 找到。

    We introduce a new Information Extraction (IE) task dubbed Instruction-based IE, which aims to ask the system to follow specific instructions or guidelines to extract information. To facilitate research in this area, we construct a dataset called InstructIE, consisting of 270,000 weakly supervised data from Chinese Wikipedia and 1,000 high-quality crowdsourced annotated instances. We further evaluate the performance of various baseline models on the InstructIE dataset. The results reveal that although current models exhibit promising performance, there is still room for improvement. Furthermore, we conduct a comprehensive case study analysis, underlining the challenges inherent in the Instruction-based IE task. Code and dataset are available at https://github.com/zjunlp/DeepKE/tree/main/example/llm.
    
[^7]: 基于概率偏置的事件抽取中重新耦合事件场

    Recouple Event Field via Probabilistic Bias for Event Extraction. (arXiv:2305.11498v1 [cs.CL])

    [http://arxiv.org/abs/2305.11498](http://arxiv.org/abs/2305.11498)

    提出了一种基于概率偏置的重新耦合事件场模型（ProCE），用于增强事件提取框架，以澄清来自模糊纠缠的事件字段，并重新耦合相应的澄清分布以捕获更多潜在信息字段。实验表明该方法有效且具有泛化性。

    

    事件抽取（EE）旨在从事件提及中识别和分类事件触发器和参数，已经受益于预训练语言模型（PLM）。然而，现有基于PLM的方法忽略了触发/参数字段的信息，这对于理解事件模式是至关重要的。为此，我们提出了一种概率重新耦合模型增强事件提取框架（ProCE）。具体来说，我们首先将语法相关的事件字段建模为概率偏置，以澄清来自模糊纠缠的事件字段。此外，考虑到EE中同一触发器/参数的多次出现，我们探索了同一触发器/参数的多个字段之间的概率交互策略，以重新耦合相应的澄清分布并捕获更多潜在信息字段。在EE数据集上的实验证明了我们提出的方法的有效性和泛化性。

    Event Extraction (EE), aiming to identify and classify event triggers and arguments from event mentions, has benefited from pre-trained language models (PLMs). However, existing PLM-based methods ignore the information of trigger/argument fields, which is crucial for understanding event schemas. To this end, we propose a Probabilistic reCoupling model enhanced Event extraction framework (ProCE). Specifically, we first model the syntactic-related event fields as probabilistic biases, to clarify the event fields from ambiguous entanglement. Furthermore, considering multiple occurrences of the same triggers/arguments in EE, we explore probabilistic interaction strategies among multiple fields of the same triggers/arguments, to recouple the corresponding clarified distributions and capture more latent information fields. Experiments on EE datasets demonstrate the effectiveness and generalization of our proposed approach.
    
[^8]: TELeR：用于基准测试复杂任务的LLM提示的通用分类法

    TELeR: A General Taxonomy of LLM Prompts for Benchmarking Complex Tasks. (arXiv:2305.11430v1 [cs.AI])

    [http://arxiv.org/abs/2305.11430](http://arxiv.org/abs/2305.11430)

    本文提出了一个通用分类法，可以用来设计具有特定属性的提示来执行各种复杂任务，从而解决了LLM在执行复杂任务方面的性能变异巨大的问题。

    

    尽管LLM在传统对话环境中理解和生成文本时取得了巨大成功，但它们在执行不明确的复杂任务方面的潜力仍然受到很少的研究。本文提出了一种通用分类法，可以用来设计具有特定属性的提示，以执行各种复杂任务，从而解决了使用不同提示类型/风格和提示提供的不同详细程度时LLM性能变化巨大的问题。这个分类法将使未来的基准测试研究能够报告研究中使用的特定提示类别，从而实现跨不同研究的有意义的比较。

    While LLMs have shown great success in understanding and generating text in traditional conversational settings, their potential for performing ill-defined complex tasks is largely under-studied. Indeed, we are yet to conduct comprehensive benchmarking studies with multiple LLMs that are exclusively focused on a complex task. However, conducting such benchmarking studies is challenging because of the large variations in LLMs' performance when different prompt types/styles are used and different degrees of detail are provided in the prompts. To address this issue, the paper proposes a general taxonomy that can be used to design prompts with specific properties in order to perform a wide range of complex tasks. This taxonomy will allow future benchmarking studies to report the specific categories of prompts used as part of the study, enabling meaningful comparisons across different studies. Also, by establishing a common standard through this taxonomy, researchers will be able to draw mo
    
[^9]: 创作者经济学中的在线学习方法

    Online Learning in a Creator Economy. (arXiv:2305.11381v1 [cs.GT])

    [http://arxiv.org/abs/2305.11381](http://arxiv.org/abs/2305.11381)

    本文探讨了如何应用在线学习的方法优化创作者经济学中的平台收益，分析和比较了基于回报的和基于特征的两种合同类型。

    

    创作者经济学革新了通过在线平台获取利润的方式。本文通过将创作者经济学建模为用户、平台和内容创作者之间的三方博弈，探讨了如何应用在线学习的方法，通过合同和推荐系统优化平台收益。该研究主要分析和比较了两种类型的合同：基于回报的合同和基于特征的合同。

    The creator economy has revolutionized the way individuals can profit through online platforms. In this paper, we initiate the study of online learning in the creator economy by modeling the creator economy as a three-party game between the users, platform, and content creators, with the platform interacting with the content creator under a principal-agent model through contracts to encourage better content. Additionally, the platform interacts with the users to recommend new content, receive an evaluation, and ultimately profit from the content, which can be modeled as a recommender system.  Our study aims to explore how the platform can jointly optimize the contract and recommender system to maximize the utility in an online learning fashion. We primarily analyze and compare two families of contracts: return-based contracts and feature-based contracts. Return-based contracts pay the content creator a fraction of the reward the platform gains. In contrast, feature-based contracts pay 
    
[^10]: 适用于所有领域的一个模型：基于协作域前缀调整的跨领域实体识别

    One Model for All Domains: Collaborative Domain-Prefix Tuning for Cross-Domain NER. (arXiv:2301.10410v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.10410](http://arxiv.org/abs/2301.10410)

    本论文提出了基于协作域前缀调整的跨领域实体识别，使用文本到文本生成的支撑领域相关指导来将知识转移至新域NER任务，避免了先前的为每个领域结束一个全新的NER模型的问题。

    

    解决实际场景中低资源问题是跨领域实体识别的一个挑战性任务。先前典型的解决方案主要通过使用来自丰富资源领域的数据进行预训练语言模型(PLMs)获得NER模型并将其适应于目标领域。由于不同领域实体类型之间的不匹配问题，先前的方法通常调整所有PLMs的参数，从而为每个领域结束一个全新的NER模型。此外，当前的模型只关注于利用一个普通来源领域中的知识，而未能成功地将来自多个来源领域的知识转移到目标上。为了解决这些问题，我们基于文本到文本生成的PLM引入了协作域前缀调整跨领域NER(CP-NER)。具体来说，我们呈现了用于文本到文本生成的支撑领域相关指导来将知识转移至新域NER任务而无需结构修改。我们利用冻结的PLMs并进行协作域前缀调整。

    Cross-domain NER is a challenging task to address the low-resource problem in practical scenarios. Previous typical solutions mainly obtain a NER model by pre-trained language models (PLMs) with data from a rich-resource domain and adapt it to the target domain. Owing to the mismatch issue among entity types in different domains, previous approaches normally tune all parameters of PLMs, ending up with an entirely new NER model for each domain. Moreover, current models only focus on leveraging knowledge in one general source domain while failing to successfully transfer knowledge from multiple sources to the target. To address these issues, we introduce Collaborative Domain-Prefix Tuning for cross-domain NER (CP-NER) based on text-to-text generative PLMs. Specifically, we present text-to-text generation grounding domain-related instructors to transfer knowledge to new domain NER tasks without structural modifications. We utilize frozen PLMs and conduct collaborative domain-prefix tuning
    

