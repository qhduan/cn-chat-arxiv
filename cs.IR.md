# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [T2Ranking: A large-scale Chinese Benchmark for Passage Ranking.](http://arxiv.org/abs/2304.03679) | T2Ranking是一个大规模的中文段落排序基准数据集，使用了4级分级相关性评分，以解决现有数据集在数据规模、细粒度相关性注释和错误负面问题方面的限制。 |
| [^2] | [From Retrieval to Generation: Efficient and Effective Entity Set Expansion.](http://arxiv.org/abs/2304.03531) | 本文提出了GenExpan，一种基于生成式预训练语言模型的实体集扩展框架，利用前缀树保证实体生成的有效性，采用自动生成的类名来引导模型生成同一类实体，从而提高了效率和可扩展性。 |
| [^3] | [Generative Recommendation: Towards Next-generation Recommender Paradigm.](http://arxiv.org/abs/2304.03516) | 生成式AI可以克服推荐系统中的限制，使其能够生成满足用户特定信息需求的内容，并且用户可以通过自然语言指令来指导内容生成。 |
| [^4] | [Continuous Input Embedding Size Search For Recommender Systems.](http://arxiv.org/abs/2304.03501) | 提出了一种新的方法CONTINUOUS，可以对潜在因子模型进行连续嵌入大小搜索，它通过将嵌入大小选择建模为连续变量解决了先前工作中的挑战，并在三个基准数据集上的实验中证实了它的有效性和高效性。 |
| [^5] | [CAPOT: Creating Robust Dense Query Encoders using Post Training Contrastive Alignment.](http://arxiv.org/abs/2304.03401) | CAPOT使用后训练对比对齐的方法，提高模型对于噪声查询的健壮性，表现类似于数据增强但没有其开销。 |
| [^6] | [Graph Collaborative Signals Denoising and Augmentation for Recommendation.](http://arxiv.org/abs/2304.03344) | 本文提出了一种新的图邻接矩阵，它包括了用户-用户和项目-项目的相关性，以及一个经过适当设计的用户-项目交互矩阵，并通过预训练和top-K采样增强了用户-项目交互矩阵，以更好地适应所有用户的需求。 |
| [^7] | [ChatGPT-Crawler: Find out if ChatGPT really knows what it's talking about.](http://arxiv.org/abs/2304.03325) | 本文分析了从不同对话QA语料库中生成的ChatGPT的响应，并比较了其与正确答案的相似度。研究发现ChatGPT在某些情况下提供了错误的答案，提供了潜在用户和开发者的宝贵见解。 |
| [^8] | [Multi-Modal Self-Supervised Learning for Recommendation.](http://arxiv.org/abs/2302.10632) | 本论文提出了一种名为“多模态自监督学习”的方法，通过有效地学习模态感知用户偏好和跨模态依赖关系的自我监控信号，以提高推荐系统的性能，实验结果表明其效果优于现有方法和多模态推荐。 |
| [^9] | [Complex QA and language models hybrid architectures, Survey.](http://arxiv.org/abs/2302.09051) | 本文综述了语言模型架构和策略的最新进展，并重点关注混合技术在复杂问题回答中的应用，讨论了该领域的挑战和未来研究方向。 |
| [^10] | [Clustering-based Imputation for Dropout Buyers in Large-scale Online Experimentation.](http://arxiv.org/abs/2209.06125) | 本文提出一种基于聚类方法的在线实验数据填补方法，将不完整指标值的用户分为访客和缺失购买者两组，使用$k$-最近邻填补方法，并考虑实验特定的特征和用户的购物路径活动，同时使用分层和聚类结合的方式提高填补效率。 |

# 详细

[^1]: T2Ranking：一个大规模的中文段落排序基准数据集

    T2Ranking: A large-scale Chinese Benchmark for Passage Ranking. (arXiv:2304.03679v1 [cs.IR])

    [http://arxiv.org/abs/2304.03679](http://arxiv.org/abs/2304.03679)

    T2Ranking是一个大规模的中文段落排序基准数据集，使用了4级分级相关性评分，以解决现有数据集在数据规模、细粒度相关性注释和错误负面问题方面的限制。

    

    段落排名包括两个阶段：段落检索和段落重新排序，这是信息检索领域中学术界和工业界都关注的重要而具有挑战性的主题。然而，用于段落排名的常用数据集通常关注英语语言。对于非英语语境，如中文，现有的数据集在数据规模、细粒度相关性注释和错误负面问题方面受到限制。为了解决这个问题，我们引入了T2Ranking，这是一个针对中文段落排序的大规模基准数据集。T2Ranking包括来自真实搜索引擎的超过300K个查询和超过2M个唯一的段落。专家评注员被招募，为查询-段落对提供4级分级相关性评分（细粒度），而不是二进制相关性判断（粗粒度）。为了减少错误负面问题，在执行相关性注释时考虑更多具有较高多样性的段落，特别是在测试集中，以确保最大化评注的质量。

    Passage ranking involves two stages: passage retrieval and passage re-ranking, which are important and challenging topics for both academics and industries in the area of Information Retrieval (IR). However, the commonly-used datasets for passage ranking usually focus on the English language. For non-English scenarios, such as Chinese, the existing datasets are limited in terms of data scale, fine-grained relevance annotation and false negative issues. To address this problem, we introduce T2Ranking, a large-scale Chinese benchmark for passage ranking. T2Ranking comprises more than 300K queries and over 2M unique passages from real-world search engines. Expert annotators are recruited to provide 4-level graded relevance scores (fine-grained) for query-passage pairs instead of binary relevance judgments (coarse-grained). To ease the false negative issues, more passages with higher diversities are considered when performing relevance annotations, especially in the test set, to ensure a m
    
[^2]: 从检索到生成：高效且有效的实体集扩展方法

    From Retrieval to Generation: Efficient and Effective Entity Set Expansion. (arXiv:2304.03531v1 [cs.CL])

    [http://arxiv.org/abs/2304.03531](http://arxiv.org/abs/2304.03531)

    本文提出了GenExpan，一种基于生成式预训练语言模型的实体集扩展框架，利用前缀树保证实体生成的有效性，采用自动生成的类名来引导模型生成同一类实体，从而提高了效率和可扩展性。

    

    实体集扩展（ESE）是一项至关重要的任务，旨在扩展由小的种子实体集描述的目标语义类的实体。大多数现有的ESE方法是基于检索的框架，需要提取实体的上下文特征，并计算种子实体和候选实体之间的相似性。为了实现这两个目的，它们必须迭代地遍历语料库和数据集中提供的实体词汇，导致效率和可扩展性较差。实验结果表明，基于检索的ESE方法消耗的时间与实体词汇和语料库的大小成线性增长。本文首先提出了一种生成式ESE框架，Generative Entity Set Expansion (GenExpan)，它利用生成式预训练语言模型来完成ESE任务。具体而言，采用前缀树来保证实体生成的有效性，并采用自动生成的类名来引导模型生成同一类实体。

    Entity Set Expansion (ESE) is a critical task aiming to expand entities of the target semantic class described by a small seed entity set. Most existing ESE methods are retrieval-based frameworks that need to extract the contextual features of entities and calculate the similarity between seed entities and candidate entities. To achieve the two purposes, they should iteratively traverse the corpus and the entity vocabulary provided in the datasets, resulting in poor efficiency and scalability. The experimental results indicate that the time consumed by the retrieval-based ESE methods increases linearly with entity vocabulary and corpus size. In this paper, we firstly propose a generative ESE framework, Generative Entity Set Expansion (GenExpan), which utilizes a generative pre-trained language model to accomplish ESE task. Specifically, a prefix tree is employed to guarantee the validity of entity generation, and automatically generated class names are adopted to guide the model to gen
    
[^3]: 生成式推荐：走向下一代推荐系统范式。

    Generative Recommendation: Towards Next-generation Recommender Paradigm. (arXiv:2304.03516v1 [cs.IR])

    [http://arxiv.org/abs/2304.03516](http://arxiv.org/abs/2304.03516)

    生成式AI可以克服推荐系统中的限制，使其能够生成满足用户特定信息需求的内容，并且用户可以通过自然语言指令来指导内容生成。

    

    推荐系统通常从项目集合中检索项目进行个性化推荐。然而，这种基于检索的推荐范式面临两个限制：1）语料库中的人工生成项目可能无法满足用户的多样化信息需求，2）用户通常通过点击等被动且低效的反馈方式调整推荐内容。近年来，人工智能生成内容在各个领域取得显著成功，具有克服这些限制的潜力：1）生成式人工智能可以生成个性化的内容以满足用户特定的信息需求，2）新兴的ChatGPT通过自然语言指令显著提高了用户准确表达信息需求的能力。在这种情况下，人工智能生成内容的大爆发指引我们走向下一代推荐范式，具有两个新的目标：1）通过生成式人工智能生成个性化内容，2）整合用户指令以指导由人工智能生成的内容。

    Recommender systems typically retrieve items from an item corpus for personalized recommendations. However, such a retrieval-based recommender paradigm faces two limitations: 1) the human-generated items in the corpus might fail to satisfy the users' diverse information needs, and 2) users usually adjust the recommendations via passive and inefficient feedback such as clicks. Nowadays, AI-Generated Content (AIGC) has revealed significant success across various domains, offering the potential to overcome these limitations: 1) generative AI can produce personalized items to meet users' specific information needs, and 2) the newly emerged ChatGPT significantly facilitates users to express information needs more precisely via natural language instructions. In this light, the boom of AIGC points the way towards the next-generation recommender paradigm with two new objectives: 1) generating personalized content through generative AI, and 2) integrating user instructions to guide content gene
    
[^4]: 推荐系统的连续输入嵌入大小搜索

    Continuous Input Embedding Size Search For Recommender Systems. (arXiv:2304.03501v1 [cs.IR])

    [http://arxiv.org/abs/2304.03501](http://arxiv.org/abs/2304.03501)

    提出了一种新的方法CONTINUOUS，可以对潜在因子模型进行连续嵌入大小搜索，它通过将嵌入大小选择建模为连续变量解决了先前工作中的挑战，并在三个基准数据集上的实验中证实了它的有效性和高效性。

    

    潜在因子模型是现今推荐系统最流行的基础，其性能卓越。潜在因子模型通过对用户和项目进行表示，用于对成对相似度的计算。所有嵌入向量传统上都被限制在一个相对较大的统一大小（例如256维）。随着当代电子商务中用户和项目目录指数级增长，这种设计显然变得效率低下。为了促进轻量级推荐，强化学习（RL）最近开辟了一些机会，用于识别不同用户/项目的不同嵌入大小。然而，受到搜索效率和学习最优RL策略的限制，现有的基于RL的方法被限制为高度离散的预定义嵌入大小选项。这导致了一个被广泛忽视的潜力，可以在给定计算预算下引入更细的粒度来获得更好的推荐效果。在本文中，我们提出了一种新方法，称为CONTINUOUS，可以对潜在因子模型进行连续嵌入大小搜索。CONTINUOUS通过将嵌入大小选择建模为连续变量和制定可微优化问题的形式来解决之前工作的挑战。在三个基准数据集上的实验证实了CONTINUOUS优于基线的优越性，验证了动态优化嵌入大小的有效性和高效性。

    Latent factor models are the most popular backbones for today's recommender systems owing to their prominent performance. Latent factor models represent users and items as real-valued embedding vectors for pairwise similarity computation, and all embeddings are traditionally restricted to a uniform size that is relatively large (e.g., 256-dimensional). With the exponentially expanding user base and item catalog in contemporary e-commerce, this design is admittedly becoming memory-inefficient. To facilitate lightweight recommendation, reinforcement learning (RL) has recently opened up opportunities for identifying varying embedding sizes for different users/items. However, challenged by search efficiency and learning an optimal RL policy, existing RL-based methods are restricted to highly discrete, predefined embedding size choices. This leads to a largely overlooked potential of introducing finer granularity into embedding sizes to obtain better recommendation effectiveness under a giv
    
[^5]: CAPOT: 使用后训练对比对齐创建强健的密集查询编码器

    CAPOT: Creating Robust Dense Query Encoders using Post Training Contrastive Alignment. (arXiv:2304.03401v1 [cs.IR])

    [http://arxiv.org/abs/2304.03401](http://arxiv.org/abs/2304.03401)

    CAPOT使用后训练对比对齐的方法，提高模型对于噪声查询的健壮性，表现类似于数据增强但没有其开销。

    

    上下文词表示的成功和神经信息检索的进步使得基于密集向量的检索成为段落和文档排名的标准方法。双编码器虽然有效和高效，但对查询分布和嘈杂查询变化很脆弱。数据增强可以使模型更加健壮，但会引入训练集生成的开销，并需要重新训练和索引重建。我们提出了 Contrastive Alignment POst Training (CAPOT)，一种高效的微调方法，通过冻结文档编码器，让查询编码器学习将嘈杂查询与其未更改的根对齐，以提高模型的健壮性。我们评估了 CAPOT 在 MSMARCO、自然问题和 Trivia QA 段落检索的嘈杂变体上，发现 CAPOT 具有与数据增强类似的影响，但没有它的开销。

    The success of contextual word representations and advances in neural information retrieval have made dense vector-based retrieval a standard approach for passage and document ranking. While effective and efficient, dual-encoders are brittle to variations in query distributions and noisy queries. Data augmentation can make models more robust but introduces overhead to training set generation and requires retraining and index regeneration. We present Contrastive Alignment POst Training (CAPOT), a highly efficient finetuning method that improves model robustness without requiring index regeneration, the training set optimization, or alteration. CAPOT enables robust retrieval by freezing the document encoder while the query encoder learns to align noisy queries with their unaltered root. We evaluate CAPOT noisy variants of MSMARCO, Natural Questions, and Trivia QA passage retrieval, finding CAPOT has a similar impact as data augmentation with none of its overhead.
    
[^6]: 推荐系统的图协作信号去噪与增强

    Graph Collaborative Signals Denoising and Augmentation for Recommendation. (arXiv:2304.03344v1 [cs.IR])

    [http://arxiv.org/abs/2304.03344](http://arxiv.org/abs/2304.03344)

    本文提出了一种新的图邻接矩阵，它包括了用户-用户和项目-项目的相关性，以及一个经过适当设计的用户-项目交互矩阵，并通过预训练和top-K采样增强了用户-项目交互矩阵，以更好地适应所有用户的需求。

    

    图协作过滤（GCF）是捕捉推荐系统中高阶协同信号的流行技术。然而，GCF的双向邻接矩阵，其定义了基于用户-项目交互进行聚合的邻居，对于有大量交互但不足的用户/项目来说可能是嘈杂的。此外，邻接矩阵忽略了用户-用户和项目-项目之间的相关性，这可能限制了聚合的有益邻居的范围。在这项工作中，我们提出了一种新的图邻接矩阵，它包括了用户-用户和项目-项目的相关性，以及一个经过适当设计的用户-项目交互矩阵，以平衡所有用户之间的交互数量。为了实现这一点，我们预先训练了一个基于图的推荐方法来获得用户/项目嵌入，然后通过top-K采样增强了用户-项目交互矩阵。我们还增强了对称的用户-用户和项目-项目相关组件，以更好地适应所有用户的需求。

    Graph collaborative filtering (GCF) is a popular technique for capturing high-order collaborative signals in recommendation systems. However, GCF's bipartite adjacency matrix, which defines the neighbors being aggregated based on user-item interactions, can be noisy for users/items with abundant interactions and insufficient for users/items with scarce interactions. Additionally, the adjacency matrix ignores user-user and item-item correlations, which can limit the scope of beneficial neighbors being aggregated.  In this work, we propose a new graph adjacency matrix that incorporates user-user and item-item correlations, as well as a properly designed user-item interaction matrix that balances the number of interactions across all users. To achieve this, we pre-train a graph-based recommendation method to obtain users/items embeddings, and then enhance the user-item interaction matrix via top-K sampling. We also augment the symmetric user-user and item-item correlation components to th
    
[^7]: ChatGPT-Crawler：发现ChatGPT是否真的知道自己在说什么。（arXiv:2304.03325v1 [cs.CL]）

    ChatGPT-Crawler: Find out if ChatGPT really knows what it's talking about. (arXiv:2304.03325v1 [cs.CL])

    [http://arxiv.org/abs/2304.03325](http://arxiv.org/abs/2304.03325)

    本文分析了从不同对话QA语料库中生成的ChatGPT的响应，并比较了其与正确答案的相似度。研究发现ChatGPT在某些情况下提供了错误的答案，提供了潜在用户和开发者的宝贵见解。

    

    大型语言模型因其在各种任务上的出色表现而引起了人们的极大兴趣。其中，OpenAI开发的ChatGPT已经成为早期采用者中非常流行的模型，他们甚至将其视为客户服务、教育、医疗和金融等许多领域的破坏性技术。理解这些初期用户的观点非常重要，因为它可以为不同领域技术的潜在优势、劣势、成功或失败提供有价值的洞见。本研究考察了ChatGPT从不同对话QA语料库中生成的响应。研究使用BERT相似度分数将这些响应与正确答案进行比较，并获得自然语言推理（NLI）标签。还计算并比较了评估分数，以确定GPT-3＆GPT-4的整体性能。此外，该研究还确定了ChatGPT提供错误答案的情况，为相关领域提供了洞见。

    Large language models have gained considerable interest for their impressive performance on various tasks. Among these models, ChatGPT developed by OpenAI has become extremely popular among early adopters who even regard it as a disruptive technology in many fields like customer service, education, healthcare, and finance. It is essential to comprehend the opinions of these initial users as it can provide valuable insights into the potential strengths, weaknesses, and success or failure of the technology in different areas. This research examines the responses generated by ChatGPT from different Conversational QA corpora. The study employed BERT similarity scores to compare these responses with correct answers and obtain Natural Language Inference(NLI) labels. Evaluation scores were also computed and compared to determine the overall performance of GPT-3 \& GPT-4. Additionally, the study identified instances where ChatGPT provided incorrect answers to questions, providing insights into
    
[^8]: 多模态自监督学习用于推荐系统

    Multi-Modal Self-Supervised Learning for Recommendation. (arXiv:2302.10632v4 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2302.10632](http://arxiv.org/abs/2302.10632)

    本论文提出了一种名为“多模态自监督学习”的方法，通过有效地学习模态感知用户偏好和跨模态依赖关系的自我监控信号，以提高推荐系统的性能，实验结果表明其效果优于现有方法和多模态推荐。

    

    多模态分享平台（例如 TikTok、YouTube）的崛起使得个性化推荐系统能够将各种模式（例如视觉、文本和声音）纳入潜在用户表示法。虽然现有的多模态推荐工作利用多媒体内容特征增强物品嵌入，但它们的模型表示能力受到重标签依赖性和稀疏用户行为数据的影响。受自监控学习在减轻标签稀缺性问题方面的最新进展的启发，我们探索有效地学习模态感知用户偏好和跨模态依赖关系的自我监控信号。为此，我们提出了一种新的多模态自监控学习（MMSSL）方法，解决了两个关键挑战。具体而言，为了表征用户-物品协同视图和物品多模态语义视图之间的相互依赖关系，我们设计了一种模态感知的交互结构学习组件；为了在模态相关的推荐任务中进一步利用自我监控信号，我们开发了一种学习用户内部模态分布的模态相关的预文本任务。对各种真实世界推荐数据集的实验结果表明，MMSSL优于最先进的推荐基线和多模态对应物，特别是在用户-物品交互稀缺的情况下。

    The online emergence of multi-modal sharing platforms (eg, TikTok, Youtube) is powering personalized recommender systems to incorporate various modalities (eg, visual, textual and acoustic) into the latent user representations. While existing works on multi-modal recommendation exploit multimedia content features in enhancing item embeddings, their model representation capability is limited by heavy label reliance and weak robustness on sparse user behavior data. Inspired by the recent progress of self-supervised learning in alleviating label scarcity issue, we explore deriving self-supervision signals with effectively learning of modality-aware user preference and cross-modal dependencies. To this end, we propose a new Multi-Modal Self-Supervised Learning (MMSSL) method which tackles two key challenges. Specifically, to characterize the inter-dependency between the user-item collaborative view and item multi-modal semantic view, we design a modality-aware interactive structure learnin
    
[^9]: 复杂问答和语言模型混合架构综述

    Complex QA and language models hybrid architectures, Survey. (arXiv:2302.09051v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.09051](http://arxiv.org/abs/2302.09051)

    本文综述了语言模型架构和策略的最新进展，并重点关注混合技术在复杂问题回答中的应用，讨论了该领域的挑战和未来研究方向。

    

    本文回顾了语言模型架构和策略的最新进展，重点关注混合技术在复杂问题回答中的应用。大型语言模型能够在标准问题上利用公共数据，但在解决更具体的复杂问题时（如在不同文化中个人自由概念的变化如何？什么是为减少气候变化而实现的最佳发电方法组合？），需要特定的架构、知识、技能、方法、敏感数据保护、可解释性、人类审批和多功能反馈。最近的项目如ChatGPT和GALACTICA允许非专业人员了解LLM在复杂QA中的巨大潜力以及同等强大的局限性。在本文中，我们首先审查所需的技能和评估技术。然后，我们综述了现有的混合架构，将LLM与基于规则的方法、信息检索、知识图谱和其他AI/ML技术相结合。最后，我们指出这些CQA系统的挑战，并提出未来研究的可能方向。

    This paper reviews the state-of-the-art of language models architectures and strategies for "complex" question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and an
    
[^10]: 基于聚类的缺失购买者在线实验数据填补方法

    Clustering-based Imputation for Dropout Buyers in Large-scale Online Experimentation. (arXiv:2209.06125v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.06125](http://arxiv.org/abs/2209.06125)

    本文提出一种基于聚类方法的在线实验数据填补方法，将不完整指标值的用户分为访客和缺失购买者两组，使用$k$-最近邻填补方法，并考虑实验特定的特征和用户的购物路径活动，同时使用分层和聚类结合的方式提高填补效率。

    

    在线实验中，合适的度量指标（比如购买）可以提供支持假设和增强决策过程的强有力证据。但是，在线实验中经常出现不完整的度量指标，使得可用数据比计划的在线实验（比如A/B测试）要少得多。在这项工作中，我们引入了缺失购买者的概念，并将指标值不完整的用户分为两组：访客和缺失购买者。为了分析不完整的指标，我们提出了一种基于聚类的$k$-最近邻填补方法。我们提出的填补方法考虑了实验特定的特征和用户沿购物路径的活动，允许不同的用户有不同的填补值。为了方便地填补在线实验中大规模数据集，所提出的方法使用分层和聚类结合的方式。所提出方法的性能与现有的比较方法相比较为优。

    In online experimentation, appropriate metrics (e.g., purchase) provide strong evidence to support hypotheses and enhance the decision-making process. However, incomplete metrics are frequently occurred in the online experimentation, making the available data to be much fewer than the planned online experiments (e.g., A/B testing). In this work, we introduce the concept of dropout buyers and categorize users with incomplete metric values into two groups: visitors and dropout buyers. For the analysis of incomplete metrics, we propose a clustering-based imputation method using $k$-nearest neighbors. Our proposed imputation method considers both the experiment-specific features and users' activities along their shopping paths, allowing different imputation values for different users. To facilitate efficient imputation of large-scale data sets in online experimentation, the proposed method uses a combination of stratification and clustering. The performance of the proposed method is compar
    

