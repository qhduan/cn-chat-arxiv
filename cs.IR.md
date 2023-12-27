# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Collaboration and Transition: Distilling Item Transitions into Multi-Query Self-Attention for Sequential Recommendation.](http://arxiv.org/abs/2311.01056) | 这篇论文提出了一种新的推荐系统方法，利用多查询自注意力和过渡感知嵌入蒸馏来捕捉用户交互序列中的合作和过渡信号。 |
| [^2] | [FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning.](http://arxiv.org/abs/2309.08420) | 提出了一种名为FedDCSR的联邦跨领域顺序推荐框架，通过解缠表示学习来处理不同领域之间的序列特征异质性，并保护数据隐私。 |
| [^3] | [Knowledge Graph Prompting for Multi-Document Question Answering.](http://arxiv.org/abs/2308.11730) | 这篇论文提出了一种知识图谱引导的方法，用于在多文档问答任务中为大型语言模型（LLMs）提示正确的上下文。通过构建多个文档上的知识图谱，并设计基于语言模型的图遍历器，该方法能够帮助LLMs在MD-QA中进行答案预测。 |
| [^4] | [Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations.](http://arxiv.org/abs/2307.05722) | 本论文探索了大规模语言模型在在线职位推荐中对图数据的理解能力，并提出了新的框架来分析行为图，发现其中的潜在模式和关系。 |
| [^5] | [The Search for Stability: Learning Dynamics of Strategic Publishers with Initial Documents.](http://arxiv.org/abs/2305.16695) | 本研究在信息检索博弈论模型中提出了相对排名原则（RRP）作为替代排名原则，以达成更稳定的搜索生态系统，并提供了理论和实证证据证明其学习动力学收敛性，同时展示了可能的出版商-用户权衡。 |

# 详细

[^1]: 合作与转换：将物品转换转化为多查询自注意力进行序列推荐

    Collaboration and Transition: Distilling Item Transitions into Multi-Query Self-Attention for Sequential Recommendation. (arXiv:2311.01056v1 [cs.IR])

    [http://arxiv.org/abs/2311.01056](http://arxiv.org/abs/2311.01056)

    这篇论文提出了一种新的推荐系统方法，利用多查询自注意力和过渡感知嵌入蒸馏来捕捉用户交互序列中的合作和过渡信号。

    

    现代推荐系统使用各种顺序模块，如自注意力来学习动态用户兴趣。然而，这些方法在捕捉用户交互序列中的合作和过渡信号方面效果较差。为了克服这些限制，我们提出了一种新方法，称为多查询自注意力与过渡感知嵌入蒸馏（MQSA-TED）。首先，我们提出了一个$L$-查询自注意力模块，使用灵活的窗口大小作为注意力查询来捕捉合作信号。此外，我们还引入了一种多查询自注意力方法，通过结合长查询和短查询来平衡建模用户偏好的偏差-方差权衡。

    Modern recommender systems employ various sequential modules such as self-attention to learn dynamic user interests. However, these methods are less effective in capturing collaborative and transitional signals within user interaction sequences. First, the self-attention architecture uses the embedding of a single item as the attention query, which is inherently challenging to capture collaborative signals. Second, these methods typically follow an auto-regressive framework, which is unable to learn global item transition patterns. To overcome these limitations, we propose a new method called Multi-Query Self-Attention with Transition-Aware Embedding Distillation (MQSA-TED). First, we propose an $L$-query self-attention module that employs flexible window sizes for attention queries to capture collaborative signals. In addition, we introduce a multi-query self-attention method that balances the bias-variance trade-off in modeling user preferences by combining long and short-query self-
    
[^2]: FedDCSR: 通过解缠表示学习实现联邦跨领域顺序推荐

    FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning. (arXiv:2309.08420v1 [cs.LG])

    [http://arxiv.org/abs/2309.08420](http://arxiv.org/abs/2309.08420)

    提出了一种名为FedDCSR的联邦跨领域顺序推荐框架，通过解缠表示学习来处理不同领域之间的序列特征异质性，并保护数据隐私。

    

    近年来，利用来自多个领域的用户序列数据的跨领域顺序推荐(CSR)受到了广泛关注。然而，现有的CSR方法需要在领域之间共享原始用户数据，这违反了《通用数据保护条例》(GDPR)。因此，有必要将联邦学习(FL)和CSR相结合，充分利用不同领域的知识，同时保护数据隐私。然而，不同领域之间的序列特征异质性对FL的整体性能有显著影响。在本文中，我们提出了FedDCSR，这是一种通过解缠表示学习的新型联邦跨领域顺序推荐框架。具体而言，为了解决不同领域之间的序列特征异质性，我们引入了一种称为领域内-领域间序列表示解缠(SRD)的方法，将用户序列特征解缠成领域共享和领域专属特征。

    Cross-domain Sequential Recommendation (CSR) which leverages user sequence data from multiple domains has received extensive attention in recent years. However, the existing CSR methods require sharing origin user data across domains, which violates the General Data Protection Regulation (GDPR). Thus, it is necessary to combine federated learning (FL) and CSR to fully utilize knowledge from different domains while preserving data privacy. Nonetheless, the sequence feature heterogeneity across different domains significantly impacts the overall performance of FL. In this paper, we propose FedDCSR, a novel federated cross-domain sequential recommendation framework via disentangled representation learning. Specifically, to address the sequence feature heterogeneity across domains, we introduce an approach called inter-intra domain sequence representation disentanglement (SRD) to disentangle the user sequence features into domain-shared and domain-exclusive features. In addition, we design
    
[^3]: 多文档问答中的知识图谱引导

    Knowledge Graph Prompting for Multi-Document Question Answering. (arXiv:2308.11730v1 [cs.CL])

    [http://arxiv.org/abs/2308.11730](http://arxiv.org/abs/2308.11730)

    这篇论文提出了一种知识图谱引导的方法，用于在多文档问答任务中为大型语言模型（LLMs）提示正确的上下文。通过构建多个文档上的知识图谱，并设计基于语言模型的图遍历器，该方法能够帮助LLMs在MD-QA中进行答案预测。

    

    大型语言模型（LLMs）的“预训练、提示、预测”范式在开放域问答（OD-QA）中取得了显著的成功。然而，很少有工作在多文档问答（MD-QA）场景下探索这个范式，这是一个要求对不同文档的内容和结构之间的逻辑关联有深入理解的任务。为了填补这一重要的空白，我们提出了一种知识图谱引导（KGP）方法，用于在MD-QA中为LLMs提示正确的上下文，该方法包括图构建模块和图遍历模块。对于图构建，我们使用节点来表示文段或文档结构（例如，页面/表格），而使用边来表示文段之间的语义/词汇相似性或者文档内的结构关系。对于图遍历，我们设计了一个基于LM的图遍历器，它在节点之间导航并收集支持性的文段，以帮助LLMs在MD-QA中进行答案预测。

    The 'pre-train, prompt, predict' paradigm of large language models (LLMs) has achieved remarkable success in open-domain question answering (OD-QA). However, few works explore this paradigm in the scenario of multi-document question answering (MD-QA), a task demanding a thorough understanding of the logical associations among the contents and structures of different documents. To fill this crucial gap, we propose a Knowledge Graph Prompting (KGP) method to formulate the right context in prompting LLMs for MD-QA, which consists of a graph construction module and a graph traversal module. For graph construction, we create a knowledge graph (KG) over multiple documents with nodes symbolizing passages or document structures (e.g., pages/tables), and edges denoting the semantic/lexical similarity between passages or intra-document structural relations. For graph traversal, we design an LM-guided graph traverser that navigates across nodes and gathers supporting passages assisting LLMs in MD
    
[^4]: 探索大规模语言模型在在线职位推荐中对图数据的理解

    Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations. (arXiv:2307.05722v1 [cs.AI])

    [http://arxiv.org/abs/2307.05722](http://arxiv.org/abs/2307.05722)

    本论文探索了大规模语言模型在在线职位推荐中对图数据的理解能力，并提出了新的框架来分析行为图，发现其中的潜在模式和关系。

    

    大规模语言模型（LLMs）在各个领域展示了其出色的能力，彻底改变了自然语言处理任务。然而，它们在职位推荐中对行为图的理解潜力仍然未被充分探索。本文旨在揭示大规模语言模型在理解行为图方面的能力，并利用这种理解来提升在线招聘中的推荐，包括促进非分布式的应用。我们提出了一个新的框架，利用大规模语言模型提供的丰富上下文信息和语义表示来分析行为图并揭示其中的潜在模式和关系。具体而言，我们提出了一个元路径提示构造器，利用LLM推荐器首次理解行为图，并设计了相应的路径增强模块来缓解基于路径的序列输入引入的提示偏差。通过利用将LM的特点引入到行为图的大规模数据分析中，我们取得了显著的实验结果，证明了我们提出的方法的有效性和性能。

    Large Language Models (LLMs) have revolutionized natural language processing tasks, demonstrating their exceptional capabilities in various domains. However, their potential for behavior graph understanding in job recommendations remains largely unexplored. This paper focuses on unveiling the capability of large language models in understanding behavior graphs and leveraging this understanding to enhance recommendations in online recruitment, including the promotion of out-of-distribution (OOD) application. We present a novel framework that harnesses the rich contextual information and semantic representations provided by large language models to analyze behavior graphs and uncover underlying patterns and relationships. Specifically, we propose a meta-path prompt constructor that leverages LLM recommender to understand behavior graphs for the first time and design a corresponding path augmentation module to alleviate the prompt bias introduced by path-based sequence input. By leveragin
    
[^5]: 寻求稳定性：具有初始文件的战略出版商的学习动态的研究

    The Search for Stability: Learning Dynamics of Strategic Publishers with Initial Documents. (arXiv:2305.16695v1 [cs.GT])

    [http://arxiv.org/abs/2305.16695](http://arxiv.org/abs/2305.16695)

    本研究在信息检索博弈论模型中提出了相对排名原则（RRP）作为替代排名原则，以达成更稳定的搜索生态系统，并提供了理论和实证证据证明其学习动力学收敛性，同时展示了可能的出版商-用户权衡。

    

    我们研究了一种信息检索的博弈论模型，其中战略出版商旨在在保持原始文档完整性的同时最大化自己排名第一的机会。我们表明，常用的PRP排名方案导致环境不稳定，游戏经常无法达到纯纳什均衡。我们将相对排名原则（RRP）作为替代排名原则，并介绍两个排名函数，它们是RRP的实例。我们提供了理论和实证证据，表明这些方法导致稳定的搜索生态系统，通过提供关于学习动力学收敛的积极结果。我们还定义出版商和用户的福利，并展示了可能的出版商-用户权衡，突显了确定搜索引擎设计师应选择哪种排名函数的复杂性。

    We study a game-theoretic model of information retrieval, in which strategic publishers aim to maximize their chances of being ranked first by the search engine, while maintaining the integrity of their original documents. We show that the commonly used PRP ranking scheme results in an unstable environment where games often fail to reach pure Nash equilibrium. We propose the Relative Ranking Principle (RRP) as an alternative ranking principle, and introduce two ranking functions that are instances of the RRP. We provide both theoretical and empirical evidence that these methods lead to a stable search ecosystem, by providing positive results on the learning dynamics convergence. We also define the publishers' and users' welfare, and demonstrate a possible publisher-user trade-off, which highlights the complexity of determining which ranking function should be selected by the search engine designer.
    

