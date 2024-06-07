# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models](https://arxiv.org/abs/2403.10081) | 提出了一种新框架DRAGIN，旨在解决大型语言模型在文本生成过程中动态检索和生成中存在的问题。 |
| [^2] | [Deep Learning Based Named Entity Recognition Models for Recipes](https://arxiv.org/abs/2402.17447) | 该研究基于深度学习，针对食谱文本开发了命名实体识别模型，通过系统的数据处理和分析，构建了用于自动生成新食谱的数据集。 |
| [^3] | [ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval](https://arxiv.org/abs/2402.15838) | ListT5通过Fusion-in-Decoder技术实现列表重排，在零-shot检索任务中表现出优越性能，效率高于之前的模型，并解决了以往列表重排器的中间段丢失问题。 |
| [^4] | [On the Embedding Collapse when Scaling up Recommendation Models.](http://arxiv.org/abs/2310.04400) | 研究了可缩放推荐模型中嵌入层的崩溃现象，发现特征交互模块在一定程度上限制了嵌入学习，但也是提高可扩展性的关键因素。 |
| [^5] | [Synthesizing Political Zero-Shot Relation Classification via Codebook Knowledge, NLI, and ChatGPT.](http://arxiv.org/abs/2308.07876) | 该论文通过利用已建立的注释编码本的知识，探索零样本方法用于政治事件本体关系分类，并介绍一种基于自然语言推理的方法，名为ZSP。ZSP采用了一种树查询框架，提高了解释性、效率和对模式更改的适应性。在细粒度根代码分类上，ZSP的性能明显优于ChatGPT，F1得分提高了40%。 |
| [^6] | [What's happening in your neighborhood? A Weakly Supervised Approach to Detect Local News.](http://arxiv.org/abs/2301.08146) | 该论文介绍了一种自动化的本地新闻检测和基于内容的本地新闻推荐方法，通过弱监督框架和自动化数据处理，与传统方法相比具有更高的准确性和覆盖率。 |

# 详细

[^1]: DRAGIN：基于大型语言模型实时信息需求的动态检索增强生成

    DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models

    [https://arxiv.org/abs/2403.10081](https://arxiv.org/abs/2403.10081)

    提出了一种新框架DRAGIN，旨在解决大型语言模型在文本生成过程中动态检索和生成中存在的问题。

    

    动态检索增强生成（RAG）范式在大型语言模型（LLMs）的文本生成过程中主动决定何时以及何时检索。该范式的两个关键元素是确定激活检索模块的最佳时机（决定何时检索）以及一旦触发检索，制定适当的查询（确定要检索什么）。然而，当前动态RAG方法在两个方面都存在不足。首先，决定何时进行检索的策略通常依赖于静态规则。此外，决定要检索什么的策略通常局限于LLM的最近一句或最后几个标记，而LLM的实时信息需求可能跨越整个上下文。为克服这些局限性，我们引入了一个新框架DRAGIN， 即基于LLMs实时信息需求的动态检索增强生成。

    arXiv:2403.10081v1 Announce Type: new  Abstract: Dynamic retrieval augmented generation (RAG) paradigm actively decides when and what to retrieve during the text generation process of Large Language Models (LLMs). There are two key elements of this paradigm: identifying the optimal moment to activate the retrieval module (deciding when to retrieve) and crafting the appropriate query once retrieval is triggered (determining what to retrieve). However, current dynamic RAG methods fall short in both aspects. Firstly, the strategies for deciding when to retrieve often rely on static rules. Moreover, the strategies for deciding what to retrieve typically limit themselves to the LLM's most recent sentence or the last few tokens, while the LLM's real-time information needs may span across the entire context. To overcome these limitations, we introduce a new framework, DRAGIN, i.e., Dynamic Retrieval Augmented Generation based on the real-time Information Needs of LLMs. Our framework is specif
    
[^2]: 食谱的深度学习命名实体识别模型

    Deep Learning Based Named Entity Recognition Models for Recipes

    [https://arxiv.org/abs/2402.17447](https://arxiv.org/abs/2402.17447)

    该研究基于深度学习，针对食谱文本开发了命名实体识别模型，通过系统的数据处理和分析，构建了用于自动生成新食谱的数据集。

    

    食物通过各种努力方式影响着我们的生活，包括口味、营养、健康和可持续性。食谱是通过非结构化文本代代相传的文化胶囊。自动识别命名实体的协议，即食谱文本的基本组成部分，对于各种应用来说都具有巨大价值，从信息提取到新颖食谱生成。命名实体识别是一种从已知标签的非结构化或半结构化数据中提取信息的技术。我们从手动注释的6,611个成分短语的数据开始，累积创建了26,445个短语的增强数据集。同时，我们系统地清理和分析了来自RecipeDB的成分短语，这是黄金标准的食谱数据存储库，并使用Stanford NER进行了标注。基于分析，我们使用基于聚类的方法对88,526个短语的子集进行了取样，同时保留了多样性。

    arXiv:2402.17447v1 Announce Type: cross  Abstract: Food touches our lives through various endeavors, including flavor, nourishment, health, and sustainability. Recipes are cultural capsules transmitted across generations via unstructured text. Automated protocols for recognizing named entities, the building blocks of recipe text, are of immense value for various applications ranging from information extraction to novel recipe generation. Named entity recognition is a technique for extracting information from unstructured or semi-structured data with known labels. Starting with manually-annotated data of 6,611 ingredient phrases, we created an augmented dataset of 26,445 phrases cumulatively. Simultaneously, we systematically cleaned and analyzed ingredient phrases from RecipeDB, the gold-standard recipe data repository, and annotated them using the Stanford NER. Based on the analysis, we sampled a subset of 88,526 phrases using a clustering-based approach while preserving the diversity
    
[^3]: ListT5: 基于解码器内融合的列表重排方法改善零-shot检索

    ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval

    [https://arxiv.org/abs/2402.15838](https://arxiv.org/abs/2402.15838)

    ListT5通过Fusion-in-Decoder技术实现列表重排，在零-shot检索任务中表现出优越性能，效率高于之前的模型，并解决了以往列表重排器的中间段丢失问题。

    

    我们提出了ListT5，一种基于在训练和推断时处理多个候选段落的Fusion-in-Decoder（FiD）的新型重排方法。我们还介绍了一个基于m元锦标赛排序和输出缓存的列表排序的高效推断框架。我们在BEIR基准上评估和比较我们的模型，证明了ListT5（1）在平均NDCG@10得分上比最先进的RankT5基线表现出明显的+1.3增益，（2）具有与逐点排名模型可比拟的效率，并超越以前的列表排序模型的效率，（3）克服了以前列表重排器的中间段丢失问题。我们的代码、模型检查点和评估框架完全开源在 \url{https://github.com/soyoung97/ListT5}。

    arXiv:2402.15838v1 Announce Type: new  Abstract: We propose ListT5, a novel reranking approach based on Fusion-in-Decoder (FiD) that handles multiple candidate passages at both train and inference time. We also introduce an efficient inference framework for listwise ranking based on m-ary tournament sort with output caching. We evaluate and compare our model on the BEIR benchmark for zero-shot retrieval task, demonstrating that ListT5 (1) outperforms the state-of-the-art RankT5 baseline with a notable +1.3 gain in the average NDCG@10 score, (2) has an efficiency comparable to pointwise ranking models and surpasses the efficiency of previous listwise ranking models, and (3) overcomes the lost-in-the-middle problem of previous listwise rerankers. Our code, model checkpoints, and the evaluation framework are fully open-sourced at \url{https://github.com/soyoung97/ListT5}.
    
[^4]: 论可扩展推荐模型中嵌入坍缩现象的研究

    On the Embedding Collapse when Scaling up Recommendation Models. (arXiv:2310.04400v1 [cs.LG])

    [http://arxiv.org/abs/2310.04400](http://arxiv.org/abs/2310.04400)

    研究了可缩放推荐模型中嵌入层的崩溃现象，发现特征交互模块在一定程度上限制了嵌入学习，但也是提高可扩展性的关键因素。

    

    深度基础模型的最新进展引发了开发大型推荐模型以利用大量可用数据的有前景趋势。然而，我们试验放大现有的推荐模型时发现，扩大的模型并没有令人满意的改进。在这种情况下，我们研究了扩大模型的嵌入层，并发现了一种嵌入坍缩现象，这最终阻碍了可扩展性，在这种现象中，嵌入矩阵倾向于存在于低维子空间中。通过实证和理论分析，我们证明了推荐模型特定的特征交互模块具有双重作用。一方面，当与坍缩的嵌入交互时，该交互限制了嵌入学习，加剧了崩溃问题。另一方面，特征交互对于缓解假特征的拟合至关重要，从而提高可扩展性。基于这一分析，我们提出了一个简单而有效的方法

    Recent advances in deep foundation models have led to a promising trend of developing large recommendation models to leverage vast amounts of available data. However, we experiment to scale up existing recommendation models and observe that the enlarged models do not improve satisfactorily. In this context, we investigate the embedding layers of enlarged models and identify a phenomenon of embedding collapse, which ultimately hinders scalability, wherein the embedding matrix tends to reside in a low-dimensional subspace. Through empirical and theoretical analysis, we demonstrate that the feature interaction module specific to recommendation models has a two-sided effect. On the one hand, the interaction restricts embedding learning when interacting with collapsed embeddings, exacerbating the collapse issue. On the other hand, feature interaction is crucial in mitigating the fitting of spurious features, thereby improving scalability. Based on this analysis, we propose a simple yet effe
    
[^5]: 通过编码本知识、自然语言推理和ChatGPT来合成政治零样本关系分类

    Synthesizing Political Zero-Shot Relation Classification via Codebook Knowledge, NLI, and ChatGPT. (arXiv:2308.07876v1 [cs.CL])

    [http://arxiv.org/abs/2308.07876](http://arxiv.org/abs/2308.07876)

    该论文通过利用已建立的注释编码本的知识，探索零样本方法用于政治事件本体关系分类，并介绍一种基于自然语言推理的方法，名为ZSP。ZSP采用了一种树查询框架，提高了解释性、效率和对模式更改的适应性。在细粒度根代码分类上，ZSP的性能明显优于ChatGPT，F1得分提高了40%。

    

    最近的事件编码的监督模型在性能方面远远超过模式匹配方法。然而，它们仅仅依赖于新的注释，忽视了专家数据库中的大量知识，限制了它们在细粒度分类中的适用性。为了解决这些限制，我们通过利用已建立的注释编码本的知识，探索零样本方法用于政治事件本体关系分类。我们的研究涵盖了ChatGPT和一种新颖的基于自然语言推理的方法，名为ZSP。ZSP采用了一种树查询框架，将任务分解为上下文、语态和类别消歧的不同层次。该框架提高了解释性、效率和对模式更改的适应性。通过在我们新策划的数据集上进行大量实验，我们指出了ChatGPT中的不稳定性问题，并突出了ZSP的卓越性能。ZSP在细粒度根代码分类的F1得分上取得了令人印象深刻的提高40%。

    Recent supervised models for event coding vastly outperform pattern-matching methods. However, their reliance solely on new annotations disregards the vast knowledge within expert databases, hindering their applicability to fine-grained classification. To address these limitations, we explore zero-shot approaches for political event ontology relation classification, by leveraging knowledge from established annotation codebooks. Our study encompasses both ChatGPT and a novel natural language inference (NLI) based approach named ZSP. ZSP adopts a tree-query framework that deconstructs the task into context, modality, and class disambiguation levels. This framework improves interpretability, efficiency, and adaptability to schema changes. By conducting extensive experiments on our newly curated datasets, we pinpoint the instability issues within ChatGPT and highlight the superior performance of ZSP. ZSP achieves an impressive 40% improvement in F1 score for fine-grained Rootcode classific
    
[^6]: 你所在社区发生了什么？一种弱监督方法用于发现本地新闻。

    What's happening in your neighborhood? A Weakly Supervised Approach to Detect Local News. (arXiv:2301.08146v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2301.08146](http://arxiv.org/abs/2301.08146)

    该论文介绍了一种自动化的本地新闻检测和基于内容的本地新闻推荐方法，通过弱监督框架和自动化数据处理，与传统方法相比具有更高的准确性和覆盖率。

    

    本地新闻是影响特定地理区域（如城市、县和州）用户的新闻子集。检测本地新闻是准确地推荐本地新闻的关键步骤。基于最新的自然语言处理技术，我们开发了一种集成化的流程，实现了自动化本地新闻检测和基于内容的本地新闻推荐。本文着重介绍了管道的第一步骤：（1）结合领域知识和自动数据处理的弱监督框架，（2）可扩展到多语言设置。与斯坦福CoreNLP NER模型相比，我们的流程在经过真实世界和人工标记数据的评估时具有更高的精度和召回率。

    Local news articles are a subset of news that impact users in a geographical area, such as a city, county, or state. Detecting local news (Step 1) and subsequently deciding its geographical location as well as radius of impact (Step 2) are two important steps towards accurate local news recommendation. Naive rule-based methods, such as detecting city names from the news title, tend to give erroneous results due to lack of understanding of the news content. Empowered by the latest development in natural language processing, we develop an integrated pipeline that enables automatic local news detection and content-based local news recommendations. In this paper, we focus on Step 1 of the pipeline, which highlights: (1) a weakly supervised framework incorporated with domain knowledge and auto data processing, and (2) scalability to multi-lingual settings. Compared with Stanford CoreNLP NER model, our pipeline has higher precision and recall evaluated on a real-world and human-labeled datas
    

