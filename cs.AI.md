# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Text clustering with LLM embeddings](https://arxiv.org/abs/2403.15112) | 研究表明，LLM嵌入能够捕捉结构化语言的细微差别，BERT在性能上领先于轻量级选项，增加嵌入维度和摘要技术并不一致地提高聚类效率 |
| [^2] | [Neuro-Symbolic Video Search](https://arxiv.org/abs/2403.11021) | 提出了一种神经网络符号视频搜索系统，该系统利用视觉-语言模型进行语义理解，并通过状态机和时间逻辑公式对事件的长期演变进行推理，从而实现高效的场景识别。 |
| [^3] | [Governance of Generative Artificial Intelligence for Companies](https://arxiv.org/abs/2403.08802) | 本综述填补了有关企业中生成式人工智能（GenAI）治理的研究空白，提出了一个框架，旨在利用业务机会并减轻与GenAI整合相关风险。 |
| [^4] | [Using text embedding models and vector databases as text classifiers with the example of medical data](https://arxiv.org/abs/2402.16886) | 向量数据库和嵌入模型的应用为文本分类器提供了强大的方式来表达数据模式，特别是在医疗领域中开始有着广泛的应用。 |
| [^5] | [ARKS: Active Retrieval in Knowledge Soup for Code Generation](https://arxiv.org/abs/2402.12317) | ARKS是一种用于代码生成的先进策略，通过活跃检索和整合各种信息源，能够提高大型语言模型的性能，为解决与频繁更新的库和长尾编程语言相关的现实编程问题提供了新的可能性。 |
| [^6] | [CultureLLM: Incorporating Cultural Differences into Large Language Models](https://arxiv.org/abs/2402.10946) | 提出了一种名为CultureLLM的成本效益高的解决方案，通过使用世界价值调查（WVS）作为种子数据，并通过提出的语义数据增强来将文化差异纳入大型语言模型中，成功微调得到了涵盖富裕和低资源语言的9种文化特定LLMs以及一个统一模型（CultureLLM-One）。 |
| [^7] | [CIDER: Category-Guided Intent Disentanglement for Accurate Personalized News Recommendation.](http://arxiv.org/abs/2310.09401) | CIDER是一种基于类别引导的个性化新闻推荐框架，通过意图分离和一致性的新闻表示来准确理解新闻文章的多个意图，并区分用户不同的后阅读偏好。 |
| [^8] | [Representation Learning for Sequential Volumetric Design Tasks.](http://arxiv.org/abs/2309.02583) | 本研究提出了一种顺序体积设计任务的表示学习方法，通过利用transformer模型从专家的设计序列中提取有用的表示来提高自动生成体积设计的质量，以及支持设计偏好评估和程序化设计生成。 |
| [^9] | [From Robustness to Explainability and Back Again.](http://arxiv.org/abs/2306.03048) | 本文介绍了一种解决形式解释可扩展性限制的新算法，通过回答鲁棒性查询来计算解释，并建立了形式解释复杂性和鲁棒性复杂性之间的直接关系。 |
| [^10] | [Synergies Between Federated Learning and O-RAN: Towards an Elastic Virtualized Architecture for Multiple Distributed Machine Learning Services.](http://arxiv.org/abs/2305.02109) | 本文研究了联邦学习在现代无线网络下的挑战，提出了一种方法称为动态多服务联邦学习（DMS-FL）来解决这个问题。同时，还提出了一种名为弹性虚拟化联邦学习（EV-FL）的分布式机器学习架构，来支持DMS-FL中的设计要求。 |

# 详细

[^1]: 使用LLM嵌入进行文本聚类

    Text clustering with LLM embeddings

    [https://arxiv.org/abs/2403.15112](https://arxiv.org/abs/2403.15112)

    研究表明，LLM嵌入能够捕捉结构化语言的细微差别，BERT在性能上领先于轻量级选项，增加嵌入维度和摘要技术并不一致地提高聚类效率

    

    文本聚类是组织不断增长的数字内容的重要方法，有助于结构化和发现未分类数据中的隐藏模式。在这项研究中，我们调查了不同文本嵌入（特别是大型语言模型LLMs中使用的）和聚类算法如何影响文本数据集的聚类方式。进行了一系列实验以评估嵌入是如何影响聚类结果的，以及通过摘要进行降维和嵌入大小调整的作用。结果显示，LLM嵌入在捕获结构化语言的细微差别方面表现出色，而BERT在性能上领先于轻量级选项。此外，我们发现增加嵌入维度和摘要技术并不一致地提高聚类效率，这表明这些策略需要仔细分析才能在实际模型中使用。这些结果突出了一种

    arXiv:2403.15112v1 Announce Type: cross  Abstract: Text clustering is an important approach for organising the growing amount of digital content, helping to structure and find hidden patterns in uncategorised data. In this research, we investigated how different textual embeddings - particularly those used in large language models (LLMs) - and clustering algorithms affect how text datasets are clustered. A series of experiments were conducted to assess how embeddings influence clustering results, the role played by dimensionality reduction through summarisation, and embedding size adjustment. Results reveal that LLM embeddings excel at capturing the nuances of structured language, while BERT leads the lightweight options in performance. In addition, we find that increasing embedding dimensionality and summarisation techniques do not uniformly improve clustering efficiency, suggesting that these strategies require careful analysis to use in real-life models. These results highlight a co
    
[^2]: 神经符号视频搜索

    Neuro-Symbolic Video Search

    [https://arxiv.org/abs/2403.11021](https://arxiv.org/abs/2403.11021)

    提出了一种神经网络符号视频搜索系统，该系统利用视觉-语言模型进行语义理解，并通过状态机和时间逻辑公式对事件的长期演变进行推理，从而实现高效的场景识别。

    

    近年来视频数据生产的空前激增需求高效的工具，以从视频中提取有意义的帧供下游任务使用。 长期时间推理是帧检索系统的一个关键要求。 虽然 VideoLLaMA 和 ViCLIP 等最先进的基础模型在短期语义理解方面表现优异，但它们在跨帧的长期推理方面却令人惊讶地失败。 这种失败的一个关键原因是它们将逐帧感知和时间推理交织成单个深度网络。 因此，解耦但共同设计语义理解和时间推理对于高效的场景识别是至关重要的。 我们提出了一种系统，利用视觉-语言模型对单个帧进行语义理解，但有效地通过使用状态机和时间逻辑（TL）公式对事件的长期演变进行推理，这些公式在本质上捕捉了记忆。

    arXiv:2403.11021v1 Announce Type: cross  Abstract: The unprecedented surge in video data production in recent years necessitates efficient tools to extract meaningful frames from videos for downstream tasks. Long-term temporal reasoning is a key desideratum for frame retrieval systems. While state-of-the-art foundation models, like VideoLLaMA and ViCLIP, are proficient in short-term semantic understanding, they surprisingly fail at long-term reasoning across frames. A key reason for this failure is that they intertwine per-frame perception and temporal reasoning into a single deep network. Hence, decoupling but co-designing semantic understanding and temporal reasoning is essential for efficient scene identification. We propose a system that leverages vision-language models for semantic understanding of individual frames but effectively reasons about the long-term evolution of events using state machines and temporal logic (TL) formulae that inherently capture memory. Our TL-based reas
    
[^3]: 企业中生成式人工智能的治理

    Governance of Generative Artificial Intelligence for Companies

    [https://arxiv.org/abs/2403.08802](https://arxiv.org/abs/2403.08802)

    本综述填补了有关企业中生成式人工智能（GenAI）治理的研究空白，提出了一个框架，旨在利用业务机会并减轻与GenAI整合相关风险。

    

    生成式人工智能（GenAI），特别是像ChatGPT这样的大型语言模型，已迅速进入企业，但缺乏充分的治理，带来机遇和挑战。尽管对GenAI具有变革性质和监管措施的广泛讨论，但有限的研究涉及组织治理，包括技术和业务视角。本综述填补了这一空白，调查了最近的研究。它不仅仅是总结，还通过制定适用于企业内的GenAI治理框架来进行。我们的框架详细描述了范围、目标和治理机制，旨在利用业务机会并减轻与GenAI整合相关风险。该研究提供了一种专注于GenAI治理的方法，为企业在负责任的AI采用挑战中提供了实用见解。对于技术人员来说，也有助于拓宽他们的视角。

    arXiv:2403.08802v1 Announce Type: new  Abstract: Generative Artificial Intelligence (GenAI), specifically large language models like ChatGPT, has swiftly entered organizations without adequate governance, posing both opportunities and risks. Despite extensive debates on GenAI's transformative nature and regulatory measures, limited research addresses organizational governance, encompassing technical and business perspectives. This review paper fills this gap by surveying recent works. It goes beyond mere summarization by developing a framework for GenAI governance within companies. Our framework outlines the scope, objectives, and governance mechanisms tailored to harness business opportunities and mitigate risks associated with GenAI integration. This research contributes a focused approach to GenAI governance, offering practical insights for companies navigating the challenges of responsible AI adoption. It is also valuable for a technical audience to broaden their perspective as inc
    
[^4]: 使用文本嵌入模型和向量数据库作为文本分类器的研究，以医疗数据为例

    Using text embedding models and vector databases as text classifiers with the example of medical data

    [https://arxiv.org/abs/2402.16886](https://arxiv.org/abs/2402.16886)

    向量数据库和嵌入模型的应用为文本分类器提供了强大的方式来表达数据模式，特别是在医疗领域中开始有着广泛的应用。

    

    大型语言模型（LLMs）的出现是令人兴奋的，并已在许多领域找到应用，但通常情况下，医学领域的标准要求非常高。与LLMs配合使用，向量嵌入模型和向量数据库提供了一种强大的方式来表达各种数据模式，这些数据模式容易被典型的机器学习模型所理解。除了方便地向这些向量数据库添加信息、知识和数据外，它们还提供了一个令人信服的理由，即将其应用于通常由人类完成的检索信息任务的各种领域。Google的研究人员开发了一个清晰的替代模型Med-PaLM，专门旨在与临床医师的医学知识水平匹配。在训练分类器和开发模型时，保持事实和减少偏见是至关重要的。在本文中，我们探讨了向量数据库和嵌入模型的应用

    arXiv:2402.16886v1 Announce Type: cross  Abstract: The advent of Large Language Models (LLMs) is promising and has found application in numerous fields, but as it often is with the medical field, the bar is typically quite high [5]. In tandem with LLMs, vector embedding models and vector databases provide a robust way of expressing numerous modes of data that are easily digestible by typical machine learning models. Along with the ease of adding information, knowledge, and data to these vector databases, they provide a compelling reason to apply them in numerous fields where the task of retrieving information is typically done by humans. Researchers at Google have developed a clear alternative model, Med-PaLM [6] specifically designed to match a clinician's level of accuracy when it comes to medical knowledge. When training classifiers, and developing models, it is imperative to maintain factuality and reduce bias [4]. Here, we explore the use of vector databases and embedding models a
    
[^5]: ARKS：用于代码生成中的知识汤中的活跃检索

    ARKS: Active Retrieval in Knowledge Soup for Code Generation

    [https://arxiv.org/abs/2402.12317](https://arxiv.org/abs/2402.12317)

    ARKS是一种用于代码生成的先进策略，通过活跃检索和整合各种信息源，能够提高大型语言模型的性能，为解决与频繁更新的库和长尾编程语言相关的现实编程问题提供了新的可能性。

    

    最近，检索增强生成（RAG）范式引起了人们的极大关注，因为它有潜力将外部知识整合到大型语言模型（LLMs）中，而无需进一步训练。尽管在自然语言应用中得到广泛探讨，但它在代码生成中的利用仍未得到充分探索。在本文中，我们介绍了一种名为知识汤中的活跃检索(ARKS)的先进策略，用于泛化大型语言模型以生成代码。与依靠单一来源不同，我们构建了一个将网页搜索、文档、执行反馈和进化代码片段整合在一起的知识汤。我们采用了一种积极的检索策略，该策略迭代地优化查询并更新知识汤。为了评估ARKS的性能，我们编制了一个新的基准，其中包括与频繁更新的库和长尾编程语言相关的现实编程问题。在ChatGPT和CodeLlama上的实验结果表明，ARKS比使用传统方法能够更好地产生代码。

    arXiv:2402.12317v1 Announce Type: cross  Abstract: Recently the retrieval-augmented generation (RAG) paradigm has raised much attention for its potential in incorporating external knowledge into large language models (LLMs) without further training. While widely explored in natural language applications, its utilization in code generation remains under-explored. In this paper, we introduce Active Retrieval in Knowledge Soup (ARKS), an advanced strategy for generalizing large language models for code. In contrast to relying on a single source, we construct a knowledge soup integrating web search, documentation, execution feedback, and evolved code snippets. We employ an active retrieval strategy that iteratively refines the query and updates the knowledge soup. To assess the performance of ARKS, we compile a new benchmark comprising realistic coding problems associated with frequently updated libraries and long-tail programming languages. Experimental results on ChatGPT and CodeLlama de
    
[^6]: 将文化差异纳入大型语言模型的研究

    CultureLLM: Incorporating Cultural Differences into Large Language Models

    [https://arxiv.org/abs/2402.10946](https://arxiv.org/abs/2402.10946)

    提出了一种名为CultureLLM的成本效益高的解决方案，通过使用世界价值调查（WVS）作为种子数据，并通过提出的语义数据增强来将文化差异纳入大型语言模型中，成功微调得到了涵盖富裕和低资源语言的9种文化特定LLMs以及一个统一模型（CultureLLM-One）。

    

    大型语言模型（LLMs）被报道偏向于某些文化，因为训练数据主要来自英语语料库。由于多语种文化数据通常较难收集，现有的工作通过提示工程或特定文化的预训练来处理这一问题。然而，它们可能忽视了低资源文化的知识缺乏，并需要大量的计算资源。本文提出了CultureLLM，这是一个成本效益高的解决方案，可将文化差异纳入LLMs中。CultureLLM采用世界价值调查（WVS）作为种子数据，并通过提出的语义数据增强生成语义等效的训练数据。仅使用来自WVS的50个种子样本和增强数据，我们对9种包括富裕和低资源语言的文化特定LLMs和一个统一模型（CultureLLM-One）进行了微调。对60个与文化相关的数据集进行的大量实验表明，CultureLLM在增强LLM的文化特性方面取得了显著的成果。

    arXiv:2402.10946v1 Announce Type: cross  Abstract: Large language models (LLMs) are reported to be partial to certain cultures owing to the training data dominance from the English corpora. Since multilingual cultural data are often expensive to collect, existing efforts handle this by prompt engineering or culture-specific pre-training. However, they might overlook the knowledge deficiency of low-resource culture and require extensive computing resources. In this paper, we propose CultureLLM, a cost-effective solution to incorporate cultural differences into LLMs. CultureLLM adopts World Value Survey (WVS) as seed data and generates semantically equivalent training data via the proposed semantic data augmentation. Using only 50 seed samples from WVS with augmented data, we fine-tune culture-specific LLMs and one unified model (CultureLLM-One) for 9 cultures covering rich and low-resource languages. Extensive experiments on 60 culture-related datasets demonstrate that CultureLLM signif
    
[^7]: CIDER: 基于类别引导的意图分离方法用于准确的个性化新闻推荐

    CIDER: Category-Guided Intent Disentanglement for Accurate Personalized News Recommendation. (arXiv:2310.09401v1 [cs.IR])

    [http://arxiv.org/abs/2310.09401](http://arxiv.org/abs/2310.09401)

    CIDER是一种基于类别引导的个性化新闻推荐框架，通过意图分离和一致性的新闻表示来准确理解新闻文章的多个意图，并区分用户不同的后阅读偏好。

    

    个性化新闻推荐旨在帮助用户找到与其兴趣相符的新闻文章，这在缓解用户信息过载问题方面起到至关重要的作用。尽管许多最近的研究致力于改进用户和新闻的表示方法，但以下挑战很少被研究：（C1）如何准确理解一篇新闻文章中包含的多个意图？以及（C2）如何区分用户点击历史中对新闻文章有不同后阅读偏好的情况？为了同时解决这两个挑战，在本文中，我们提出了一种新的个性化新闻推荐框架（CIDER），它利用（1）基于类别引导的意图分离来解决（C1）和（2）基于一致性的新闻表示来解决（C2）。此外，我们将类别预测纳入CIDER的训练过程作为辅助任务，这提供了额外的监督信号，以增强意图分离。在两个真实数据集上进行了广泛的实验。

    Personalized news recommendation aims to assist users in finding news articles that align with their interests, which plays a pivotal role in mitigating users' information overload problem. Although many recent works have been studied for better user and news representations, the following challenges have been rarely studied: (C1) How to precisely comprehend a range of intents coupled within a news article? and (C2) How to differentiate news articles with varying post-read preferences in users' click history? To tackle both challenges together, in this paper, we propose a novel personalized news recommendation framework (CIDER) that employs (1) category-guided intent disentanglement for (C1) and (2) consistency-based news representation for (C2). Furthermore, we incorporate a category prediction into the training process of CIDER as an auxiliary task, which provides supplementary supervisory signals to enhance intent disentanglement. Extensive experiments on two real-world datasets rev
    
[^8]: 顺序体积设计任务的表示学习

    Representation Learning for Sequential Volumetric Design Tasks. (arXiv:2309.02583v1 [cs.LG])

    [http://arxiv.org/abs/2309.02583](http://arxiv.org/abs/2309.02583)

    本研究提出了一种顺序体积设计任务的表示学习方法，通过利用transformer模型从专家的设计序列中提取有用的表示来提高自动生成体积设计的质量，以及支持设计偏好评估和程序化设计生成。

    

    体积设计，也称为质量设计，是专业建筑设计中的第一步关键性任务，具有顺序性。由于体积设计过程复杂，顺序化设计过程中包含了对设计师有价值的信息。许多努力已经被投入到自动生成合理的体积设计上，但生成的设计解决方案的质量存在差异，并且评估一个设计解决方案要么需要一套过于全面的度量标准，要么需要昂贵的人力专业知识。而之前的方法主要关注学习最终设计，而不是顺序设计任务，我们提出利用专家或高性能设计序列的设计知识，并使用基于transformer的模型提取有用的表示。然后，我们提出利用所学的表示在关键的下游应用中，如设计偏好评估和程序化设计生成。我们开发了prefer

    Volumetric design, also called massing design, is the first and critical step in professional building design which is sequential in nature. As the volumetric design process is complex, the underlying sequential design process encodes valuable information for designers. Many efforts have been made to automatically generate reasonable volumetric designs, but the quality of the generated design solutions varies, and evaluating a design solution requires either a prohibitively comprehensive set of metrics or expensive human expertise. While previous approaches focused on learning only the final design instead of sequential design tasks, we propose to encode the design knowledge from a collection of expert or high-performing design sequences and extract useful representations using transformer-based models. Later we propose to utilize the learned representations for crucial downstream applications such as design preference evaluation and procedural design generation. We develop the prefere
    
[^9]: 从鲁棒性到可解释性再到鲁棒性。(arXiv:2306.03048v2 [cs.AI] 已更新)

    From Robustness to Explainability and Back Again. (arXiv:2306.03048v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2306.03048](http://arxiv.org/abs/2306.03048)

    本文介绍了一种解决形式解释可扩展性限制的新算法，通过回答鲁棒性查询来计算解释，并建立了形式解释复杂性和鲁棒性复杂性之间的直接关系。

    

    与可解释的人工智能（XAI）的临时方法相比，形式解释提供了严密性的重要保证。然而，形式解释在某些分类器家族中的可扩展性差，其中最重要的是神经网络。因此，人们担心形式解释是否可以作为其他方法的补充，以提供可信赖的人工智能。本文解决了形式解释可扩展性的限制，并提出了用于计算形式解释的新算法。这种新算法通过回答一系列鲁棒性查询来计算解释，并且这些查询的数量最多与特征数量成线性关系。因此，所提出的算法在实际复杂性和鲁棒性的复杂性之间建立了直接关系。更重要的是，本文推广了形式解释的定义，从而允许使用其他方法来改进解释的可扩展性。

    In contrast with ad-hoc methods for eXplainable Artificial Intelligence (XAI), formal explainability offers important guarantees of rigor. However, formal explainability is hindered by poor scalability for some families of classifiers, the most significant being neural networks. As a result, there are concerns as to whether formal explainability might serve to complement other approaches in delivering trustworthy AI. This paper addresses the limitation of scalability of formal explainability, and proposes novel algorithms for computing formal explanations. The novel algorithm computes explanations by answering instead a number of robustness queries, and such that the number of such queries is at most linear on the number of features. Consequently, the proposed algorithm establishes a direct relationship between the practical complexity of formal explainability and that of robustness. More importantly, the paper generalizes the definition of formal explanation, thereby allowing the use 
    
[^10]: 联邦学习与O-RAN的协同：面向多个分布式机器学习服务的弹性虚拟化架构

    Synergies Between Federated Learning and O-RAN: Towards an Elastic Virtualized Architecture for Multiple Distributed Machine Learning Services. (arXiv:2305.02109v1 [cs.NI])

    [http://arxiv.org/abs/2305.02109](http://arxiv.org/abs/2305.02109)

    本文研究了联邦学习在现代无线网络下的挑战，提出了一种方法称为动态多服务联邦学习（DMS-FL）来解决这个问题。同时，还提出了一种名为弹性虚拟化联邦学习（EV-FL）的分布式机器学习架构，来支持DMS-FL中的设计要求。

    

    联邦学习是最流行的分布式机器学习技术，但是在现代无线网络中实现联邦学习面临着许多挑战，主要包括网络条件的动态性、系统中多个联邦学习服务/任务的并存以及联邦学习服务与其他网络服务的并行执行等。针对这些挑战，本文提出了一种名为动态多服务联邦学习（DMS-FL）的联邦学习泛型架构，并通过提出一种新的分布式机器学习架构——弹性虚拟化联邦学习（EV-FL）来解决DMS-FL中的三个未探索的设计问题。

    Federated learning (FL) is the most popular distributed machine learning technique. However, implementation of FL over modern wireless networks faces key challenges caused by (i) dynamics of the network conditions, (ii) coexistence of multiple FL services/tasks in the system, and (iii) concurrent execution of FL services with other network services, which are not jointly considered in prior works. Motivated by these challenges, we introduce a generic FL paradigm over next-generation (NextG) networks, called dynamic multi-service FL (DMS-FL). We identify three unexplored design considerations in DMS-FL: (i) FL service operator accumulation, (ii) wireless resource fragmentation, and (iii) signal strength fluctuations. We take the first steps towards addressing these design considerations through proposing a novel distributed ML architecture called elastic virtualized FL (EV-FL). EV-FL unleashes the full potential of Open RAN (O-RAN) systems and introduces an elastic resource provisioning
    

