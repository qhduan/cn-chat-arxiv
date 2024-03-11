# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Is Cosine-Similarity of Embeddings Really About Similarity?](https://arxiv.org/abs/2403.05440) | 余弦相似度可以产生任意和无意义的“相似性”，受正则化控制，并讨论了深层模型学习中的影响。 |
| [^2] | [Harnessing Multi-Role Capabilities of Large Language Models for Open-Domain Question Answering](https://arxiv.org/abs/2403.05217) | 提出了LLMQA框架，利用大型语言模型在开放领域问答中扮演生成器、重新排序器和评估器等多重角色，结合了检索和生成证据的优势。 |
| [^3] | [Personalized Audiobook Recommendations at Spotify Through Graph Neural Networks](https://arxiv.org/abs/2403.05185) | 通过引入双塔模型和异质图神经网络，该研究提出了一个可扩展的推荐系统，以应对Spotify在引入有声读物后面临的个性化推荐挑战。 |
| [^4] | [Multi-Tower Multi-Interest Recommendation with User Representation Repel](https://arxiv.org/abs/2403.05122) | 提出了一种具有用户表示排斥的新型多塔多兴趣框架，解决了多兴趣学习方法面临的训练和部署目标差异、无法访问商品信息以及难以工业采用等问题。 |
| [^5] | [Aligning Large Language Models for Controllable Recommendations](https://arxiv.org/abs/2403.05063) | 通过引入监督学习任务和强化学习对齐程序，研究人员提出了一种方法来改善大型语言模型适应推荐指令和减少格式错误的能力。 |
| [^6] | [Can't Remember Details in Long Documents? You Need Some R&R](https://arxiv.org/abs/2403.05004) | 引入R&R方法，结合reprompting和in-context retrieval两种新型提示方式，提高了在长文档上的问答任务的准确性。 |
| [^7] | [Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning](https://arxiv.org/abs/2403.04875) | GPTRec模型使用Next-K策略来生成推荐，与传统的Top-K模型不同，可以更好地考虑超出准确性指标的复杂项目间相互依赖性。 |
| [^8] | [ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data](https://arxiv.org/abs/2403.04871) | ACORN提出了一种高性能和与谓词无关的混合搜索方法，通过引入谓词子图遍历来模拟理论上理想但实际上不切实际的混合搜索策略。 |
| [^9] | [Automating the Information Extraction from Semi-Structured Interview Transcripts](https://arxiv.org/abs/2403.04819) | 本研究提出了一种新的自动系统，结合了BERT嵌入和HDBSCAN聚类，可以从半结构化面谈文本中快速提取信息，为研究人员提供了一个便捷的工具来分析和可视化主题结构。 |
| [^10] | [LLM vs. Lawyers: Identifying a Subset of Summary Judgments in a Large UK Case Law Dataset](https://arxiv.org/abs/2403.04791) | 使用大型语言模型（LLM）对比传统的自然语言处理方法，可以更有效地从大型英国法院判决数据集中识别摘要裁定案例，取得了更高的F1得分。 |
| [^11] | [Filter Bubble or Homogenization? Disentangling the Long-Term Effects of Recommendations on User Consumption Patterns](https://arxiv.org/abs/2402.15013) | 本文解析了推荐算法对用户行为的长期影响，探讨了同质化和滤泡效应之间的关系，发现个性化推荐能够缓解滤泡效应。 |
| [^12] | [Fairness Rising from the Ranks: HITS and PageRank on Homophilic Networks](https://arxiv.org/abs/2402.13787) | 本文研究了链接分析算法在阻止少数群体在网络中达到高排名位置的条件，发现PageRank能够平衡排名中少数群体的代表性，而HITS则在同构网络中通过新颖的理论分析放大了现有的偏见。 |
| [^13] | [SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks.](http://arxiv.org/abs/2401.15299) | SupplyGraph是一个基准数据集，用于使用图神经网络进行供应链规划。该数据集包含了来自孟加拉国一家领先快速消费品公司的实际数据，用于优化、预测和解决供应链问题。数据集中的时间数据作为节点特征，可用于销售预测、生产计划和故障识别。 |
| [^14] | [Density-based User Representation through Gaussian Process Regression for Multi-interest Personalized Retrieval.](http://arxiv.org/abs/2310.20091) | 本研究引入了一种基于密度的用户表示(DURs)，利用高斯过程回归实现了有效的多兴趣推荐和检索。该方法不仅能够捕捉用户的兴趣变化，还具备不确定性感知能力，并且适用于大量用户的规模。 |
| [^15] | [Recall, Robustness, and Lexicographic Evaluation.](http://arxiv.org/abs/2302.11370) | 该论文从正式的角度反思了排名中召回率的测量问题，提出召回方向的概念和词典式方法，并分析了其鲁棒性。 |
| [^16] | [A Survey on Modern Recommendation System based on Big Data.](http://arxiv.org/abs/2206.02631) | 这份综述全面调研了基于大数据的现代推荐系统的发展和挑战，总结了四种主要类型的推荐技术，并指出了未来研究的潜在领域。 |

# 详细

[^1]: 嵌入的余弦相似性真的只是关于相似性吗？

    Is Cosine-Similarity of Embeddings Really About Similarity?

    [https://arxiv.org/abs/2403.05440](https://arxiv.org/abs/2403.05440)

    余弦相似度可以产生任意和无意义的“相似性”，受正则化控制，并讨论了深层模型学习中的影响。

    

    余弦相似度是两个向量之间夹角的余弦，或者等价地说是它们归一化后的点积。一个常见的应用是通过将余弦相似度应用于学习的低维特征嵌入来量化高维对象之间的语义相似性。在实践中，这种方法有时比嵌入向量之间的未归一化点积效果更好，但有时也更差。为了深入了解这一经验观察，我们研究了由正则化线性模型导出的嵌入，其中封闭形式的解决方案有助于分析洞察力。我们在分析上推导出余弦相似性如何产生任意且因此无意义的“相似性”。对于一些线性模型，相似性甚至不是唯一的，而对于其他一些模型，它们受到正则化的隐式控制。我们讨论了线性模型之外的影响：在学习深层模型时，会采用不同正则化的组合。

    arXiv:2403.05440v1 Announce Type: cross  Abstract: Cosine-similarity is the cosine of the angle between two vectors, or equivalently the dot product between their normalizations. A popular application is to quantify semantic similarity between high-dimensional objects by applying cosine-similarity to a learned low-dimensional feature embedding. This can work better but sometimes also worse than the unnormalized dot-product between embedded vectors in practice. To gain insight into this empirical observation, we study embeddings derived from regularized linear models, where closed-form solutions facilitate analytical insights. We derive analytically how cosine-similarity can yield arbitrary and therefore meaningless `similarities.' For some linear models the similarities are not even unique, while for others they are implicitly controlled by the regularization. We discuss implications beyond linear models: a combination of different regularizations are employed when learning deep models
    
[^2]: 利用大型语言模型的多角色能力进行开放领域问答

    Harnessing Multi-Role Capabilities of Large Language Models for Open-Domain Question Answering

    [https://arxiv.org/abs/2403.05217](https://arxiv.org/abs/2403.05217)

    提出了LLMQA框架，利用大型语言模型在开放领域问答中扮演生成器、重新排序器和评估器等多重角色，结合了检索和生成证据的优势。

    

    开放领域问答（ODQA）已经成为信息系统中的一个关键研究焦点。现有方法主要遵循两种范式来收集证据：（1）\textit{检索-然后阅读}范式从外部语料库中检索相关文档；和（2）\textit{生成-然后阅读}范式使用大型语言模型（LLMs）生成相关文档。然而，这两种方法都不能完全满足证据的多方面要求。因此，我们提出了LLMQA，一个通用框架，将ODQA过程分为三个基本步骤：查询扩展，文档选择和答案生成，结合了检索和生成证据的优势。由于LLMs展示了其出色的能力来完成各种任务，我们指导LLMs在我们的框架内扮演生成器、重新排序器和评估器等多种角色，使它们融合在ODQA过程中协作。

    arXiv:2403.05217v1 Announce Type: cross  Abstract: Open-domain question answering (ODQA) has emerged as a pivotal research spotlight in information systems. Existing methods follow two main paradigms to collect evidence: (1) The \textit{retrieve-then-read} paradigm retrieves pertinent documents from an external corpus; and (2) the \textit{generate-then-read} paradigm employs large language models (LLMs) to generate relevant documents. However, neither can fully address multifaceted requirements for evidence. To this end, we propose LLMQA, a generalized framework that formulates the ODQA process into three basic steps: query expansion, document selection, and answer generation, combining the superiority of both retrieval-based and generation-based evidence. Since LLMs exhibit their excellent capabilities to accomplish various tasks, we instruct LLMs to play multiple roles as generators, rerankers, and evaluators within our framework, integrating them to collaborate in the ODQA process. 
    
[^3]: 通过图神经网络在Spotify上进行个性化有声读物推荐

    Personalized Audiobook Recommendations at Spotify Through Graph Neural Networks

    [https://arxiv.org/abs/2403.05185](https://arxiv.org/abs/2403.05185)

    通过引入双塔模型和异质图神经网络，该研究提出了一个可扩展的推荐系统，以应对Spotify在引入有声读物后面临的个性化推荐挑战。

    

    在不断发展的数字音频领域中，以其音乐和谈话内容而闻名的Spotify最近向其庞大用户群引入了有声读物。尽管前景看好，但这一举措为个性化推荐带来了重大挑战。与音乐和播客不同，最初需要付费的有声读物在购买前无法轻松略读，这增加了推荐的相关性的挑战。此外，将新内容类型引入现有平台导致数据极度稀疏，因为大多数用户对这种新内容类型不熟悉。最后，向数百万用户推荐内容要求模型反应迅速且可扩展性强。为了解决这些挑战，我们利用播客和音乐用户喜好，引入了2T-HGNN，这是一个由异质图神经网络（HGNNs）和双塔（2T）模型组成的可扩展推荐系统。这一新颖方法揭示了项目之间微妙的关系。

    arXiv:2403.05185v1 Announce Type: cross  Abstract: In the ever-evolving digital audio landscape, Spotify, well-known for its music and talk content, has recently introduced audiobooks to its vast user base. While promising, this move presents significant challenges for personalized recommendations. Unlike music and podcasts, audiobooks, initially available for a fee, cannot be easily skimmed before purchase, posing higher stakes for the relevance of recommendations. Furthermore, introducing a new content type into an existing platform confronts extreme data sparsity, as most users are unfamiliar with this new content type. Lastly, recommending content to millions of users requires the model to react fast and be scalable. To address these challenges, we leverage podcast and music user preferences and introduce 2T-HGNN, a scalable recommendation system comprising Heterogeneous Graph Neural Networks (HGNNs) and a Two Tower (2T) model. This novel approach uncovers nuanced item relationship
    
[^4]: 具有用户表示排斥的多塔多兴趣推荐

    Multi-Tower Multi-Interest Recommendation with User Representation Repel

    [https://arxiv.org/abs/2403.05122](https://arxiv.org/abs/2403.05122)

    提出了一种具有用户表示排斥的新型多塔多兴趣框架，解决了多兴趣学习方法面临的训练和部署目标差异、无法访问商品信息以及难以工业采用等问题。

    

    在信息过载的时代，学术界和工业界都深刻认识到推荐系统的价值。特别是多兴趣序列推荐是近年来受到越来越多关注的一个子领域。通过生成多用户表示，多兴趣学习模型在理论上和经验上都比单用户表示模型具有更强的表达能力。尽管该领域取得了重大进展，但仍存在三个主要问题困扰着多兴趣学习方法的性能和可采用性，即训练和部署目标之间的差异、无法访问商品信息以及由于其单塔架构而难以工业采用。我们通过提出一种具有用户表示排斥的新型多塔多兴趣框架来解决这些挑战。通过跨多个大规模实验结果，我们证明了我们的方法的有效性。

    arXiv:2403.05122v1 Announce Type: cross  Abstract: In the era of information overload, the value of recommender systems has been profoundly recognized in academia and industry alike. Multi-interest sequential recommendation, in particular, is a subfield that has been receiving increasing attention in recent years. By generating multiple-user representations, multi-interest learning models demonstrate superior expressiveness than single-user representation models, both theoretically and empirically. Despite major advancements in the field, three major issues continue to plague the performance and adoptability of multi-interest learning methods, the difference between training and deployment objectives, the inability to access item information, and the difficulty of industrial adoption due to its single-tower architecture. We address these challenges by proposing a novel multi-tower multi-interest framework with user representation repel. Experimental results across multiple large-scale 
    
[^5]: 调整大型语言模型以实现可控的推荐

    Aligning Large Language Models for Controllable Recommendations

    [https://arxiv.org/abs/2403.05063](https://arxiv.org/abs/2403.05063)

    通过引入监督学习任务和强化学习对齐程序，研究人员提出了一种方法来改善大型语言模型适应推荐指令和减少格式错误的能力。

    

    受到大型语言模型（LLMs）异常的智能启发，研究人员已开始探索将它们应用于开创下一代推荐系统 - 这些系统具有对话、可解释和可控的特性。然而，现有文献主要集中在将领域特定知识整合到LLMs中以提高准确性，通常忽略了遵循指令的能力。为填补这一空白，我们首先引入一组监督学习任务，标记来源于传统推荐模型的标签，旨在明确改善LLMs遵循特定推荐指令的熟练程度。随后，我们开发了一种基于强化学习的对齐程序，进一步加强了LLMs在响应用户意图和减少格式错误方面的能力。通过在两个真实世界数据集上进行广泛实验，我们的方法标记着

    arXiv:2403.05063v1 Announce Type: cross  Abstract: Inspired by the exceptional general intelligence of Large Language Models (LLMs), researchers have begun to explore their application in pioneering the next generation of recommender systems - systems that are conversational, explainable, and controllable. However, existing literature primarily concentrates on integrating domain-specific knowledge into LLMs to enhance accuracy, often neglecting the ability to follow instructions. To address this gap, we initially introduce a collection of supervised learning tasks, augmented with labels derived from a conventional recommender model, aimed at explicitly improving LLMs' proficiency in adhering to recommendation-specific instructions. Subsequently, we develop a reinforcement learning-based alignment procedure to further strengthen LLMs' aptitude in responding to users' intentions and mitigating formatting errors. Through extensive experiments on two real-world datasets, our method markedl
    
[^6]: 无法记住长文档中的细节？您需要一些R&R

    Can't Remember Details in Long Documents? You Need Some R&R

    [https://arxiv.org/abs/2403.05004](https://arxiv.org/abs/2403.05004)

    引入R&R方法，结合reprompting和in-context retrieval两种新型提示方式，提高了在长文档上的问答任务的准确性。

    

    长上下文大型语言模型（LLMs）在诸如长篇文档上的问答（QA）等任务中表现出潜力，但它们往往会错过上下文文档中间的重要信息。在这里，我们介绍了一个名为$\textit{R&R}$的方法，它结合了两种新型基于提示的方法，称为$\textit{reprompting}$和$\textit{in-context retrieval}$（ICR），以减轻文档型QA中的这种影响。在$\textit{reprompting}$中，我们周期性地在整个上下文文档中重复提示说明，以提醒LLM其原始任务。在ICR中，我们并不指示LLM直接回答问题，而是指示它检索与给定问题最相关的前$k$个段落编号，然后将其用作第二个QA提示中的缩略上下文。我们使用GPT-4 Turbo和Claude-2.1在长度达到80k标记的文档上测试了R&R，并平均观察到QA准确率提升了16个百分点。

    arXiv:2403.05004v1 Announce Type: cross  Abstract: Long-context large language models (LLMs) hold promise for tasks such as question-answering (QA) over long documents, but they tend to miss important information in the middle of context documents (arXiv:2307.03172v3). Here, we introduce $\textit{R&R}$ -- a combination of two novel prompt-based methods called $\textit{reprompting}$ and $\textit{in-context retrieval}$ (ICR) -- to alleviate this effect in document-based QA. In reprompting, we repeat the prompt instructions periodically throughout the context document to remind the LLM of its original task. In ICR, rather than instructing the LLM to answer the question directly, we instruct it to retrieve the top $k$ passage numbers most relevant to the given question, which are then used as an abbreviated context in a second QA prompt. We test R&R with GPT-4 Turbo and Claude-2.1 on documents up to 80k tokens in length and observe a 16-point boost in QA accuracy on average. Our further an
    
[^7]: 用强化学习将GPTRec与超出准确性目标对齐

    Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning

    [https://arxiv.org/abs/2403.04875](https://arxiv.org/abs/2403.04875)

    GPTRec模型使用Next-K策略来生成推荐，与传统的Top-K模型不同，可以更好地考虑超出准确性指标的复杂项目间相互依赖性。

    

    Transformer模型的改编，如BERT4Rec和SASRec，在顺序推荐任务中取得了 accuracy-based 指标，如NDCG方面的最先进性能。这些模型将项目视为标记，然后利用评分-排名方法（Top-K策略），其中模型首先计算项目得分，然后根据此分数对其进行排名。虽然该方法对于准确性指标效果很好，但很难将其用于优化更复杂的超出准确性指标，如多样性。最近，提出了使用不同 Next-K 策略的GPTRec模型，作为Top-K模型的替代方案。与传统的Top-K推荐相比，Next-K会逐个项目生成推荐，因此，可以考虑超出准确性指标中重要的复杂项目间相互依赖性。

    arXiv:2403.04875v1 Announce Type: cross  Abstract: Adaptations of Transformer models, such as BERT4Rec and SASRec, achieve state-of-the-art performance in the sequential recommendation task according to accuracy-based metrics, such as NDCG. These models treat items as tokens and then utilise a score-and-rank approach (Top-K strategy), where the model first computes item scores and then ranks them according to this score. While this approach works well for accuracy-based metrics, it is hard to use it for optimising more complex beyond-accuracy metrics such as diversity. Recently, the GPTRec model, which uses a different Next-K strategy, has been proposed as an alternative to the Top-K models. In contrast with traditional Top-K recommendations, Next-K generates recommendations item-by-item and, therefore, can account for complex item-to-item interdependencies important for the beyond-accuracy measures. However, the original GPTRec paper focused only on accuracy in experiments and needed 
    
[^8]: ACORN：高性能和谓词无关的矢量嵌入和结构化数据搜索

    ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data

    [https://arxiv.org/abs/2403.04871](https://arxiv.org/abs/2403.04871)

    ACORN提出了一种高性能和与谓词无关的混合搜索方法，通过引入谓词子图遍历来模拟理论上理想但实际上不切实际的混合搜索策略。

    

    应用程序越来越多地利用混合模态数据，并且必须联合搜索矢量数据，如嵌入图像、文本和视频，以及结构化数据，如属性和关键词。针对这种混合搜索环境的提出的方法要么性能不佳，要么支持一组非常受限制的搜索谓词（例如，仅支持小的相等谓词集），使其对许多应用程序来说不切实际。为了解决这个问题，我们提出了ACORN，一种高性能且与谓词无关的混合搜索方法。ACORN基于Hierarchical Navigable Small Worlds（HNSW），这是一种最先进的基于图的近似最近邻索引，并且可以通过扩展现有的HNSW库有效实现。ACORN引入了谓词子图遍历的概念，以模拟一种在理论上理想但在实践中不切实际的混合搜索策略。ACORN的谓词无关构造算法旨在实现这种有效果。

    arXiv:2403.04871v1 Announce Type: new  Abstract: Applications increasingly leverage mixed-modality data, and must jointly search over vector data, such as embedded images, text and video, as well as structured data, such as attributes and keywords. Proposed methods for this hybrid search setting either suffer from poor performance or support a severely restricted set of search predicates (e.g., only small sets of equality predicates), making them impractical for many applications. To address this, we present ACORN, an approach for performant and predicate-agnostic hybrid search. ACORN builds on Hierarchical Navigable Small Worlds (HNSW), a state-of-the-art graph-based approximate nearest neighbor index, and can be implemented efficiently by extending existing HNSW libraries. ACORN introduces the idea of predicate subgraph traversal to emulate a theoretically ideal, but impractical, hybrid search strategy. ACORN's predicate-agnostic construction algorithm is designed to enable this effe
    
[^9]: 从半结构化面谈文本中自动提取信息

    Automating the Information Extraction from Semi-Structured Interview Transcripts

    [https://arxiv.org/abs/2403.04819](https://arxiv.org/abs/2403.04819)

    本研究提出了一种新的自动系统，结合了BERT嵌入和HDBSCAN聚类，可以从半结构化面谈文本中快速提取信息，为研究人员提供了一个便捷的工具来分析和可视化主题结构。

    

    本文探讨了开发和应用一种自动系统，旨在从半结构化面谈文本中提取信息。由于传统的定性分析方法，如编码，劳动密集型的本质，对可以促进分析过程的工具存在着重大需求。我们的研究探讨了各种主题建模技术，并得出结论，适用于分析面谈文本的最佳模型是BERT嵌入和HDBSCAN聚类的结合。我们提出了一个用户友好的软件原型，使研究人员，包括那些没有编程技能的人，能够高效处理和可视化面谈数据的主题结构。该工具不仅促进了定性分析的初期阶段，还为揭示的主题之间的相互联系提供了见解，从而增强了定性分析的深度。

    arXiv:2403.04819v1 Announce Type: new  Abstract: This paper explores the development and application of an automated system designed to extract information from semi-structured interview transcripts. Given the labor-intensive nature of traditional qualitative analysis methods, such as coding, there exists a significant demand for tools that can facilitate the analysis process. Our research investigates various topic modeling techniques and concludes that the best model for analyzing interview texts is a combination of BERT embeddings and HDBSCAN clustering. We present a user-friendly software prototype that enables researchers, including those without programming skills, to efficiently process and visualize the thematic structure of interview data. This tool not only facilitates the initial stages of qualitative analysis but also offers insights into the interconnectedness of topics revealed, thereby enhancing the depth of qualitative analysis.
    
[^10]: LLM对抗律师：在大型英国案例法律数据集中识别摘要裁定的子集

    LLM vs. Lawyers: Identifying a Subset of Summary Judgments in a Large UK Case Law Dataset

    [https://arxiv.org/abs/2403.04791](https://arxiv.org/abs/2403.04791)

    使用大型语言模型（LLM）对比传统的自然语言处理方法，可以更有效地从大型英国法院判决数据集中识别摘要裁定案例，取得了更高的F1得分。

    

    为了进行法律领域的计算研究，高效地识别与特定法律问题相关的法院裁决数据集是一项至关重要但具有挑战性的努力。本研究填补了文献中关于如何从大量英国法院决定的文集中隔离案例（在我们的案例中是摘要裁定）的空白。我们介绍了两种计算方法的比较分析：（1）传统的基于自然语言处理的方法，利用专家生成的关键字和逻辑运算符，以及（2）创新性地将Claude 2大语言模型应用于基于特定内容提示分类案例。我们使用了包含356,011份英国法院判决的剑桥法学文集，并确定大型语言模型的加权F1得分为0.94，而关键字的得分为0.78。尽管经过迭代改进，基于关键字的搜索逻辑未能捕捉法律语言中的细微差别。

    arXiv:2403.04791v1 Announce Type: new  Abstract: To undertake computational research of the law, efficiently identifying datasets of court decisions that relate to a specific legal issue is a crucial yet challenging endeavour. This study addresses the gap in the literature working with large legal corpora about how to isolate cases, in our case summary judgments, from a large corpus of UK court decisions. We introduce a comparative analysis of two computational methods: (1) a traditional natural language processing-based approach leveraging expert-generated keywords and logical operators and (2) an innovative application of the Claude 2 large language model to classify cases based on content-specific prompts. We use the Cambridge Law Corpus of 356,011 UK court decisions and determine that the large language model achieves a weighted F1 score of 0.94 versus 0.78 for keywords. Despite iterative refinement, the search logic based on keywords fails to capture nuances in legal language. We 
    
[^11]: 推荐算法对用户消费模式长期影响的解析: 滤泡还是同质化？

    Filter Bubble or Homogenization? Disentangling the Long-Term Effects of Recommendations on User Consumption Patterns

    [https://arxiv.org/abs/2402.15013](https://arxiv.org/abs/2402.15013)

    本文解析了推荐算法对用户行为的长期影响，探讨了同质化和滤泡效应之间的关系，发现个性化推荐能够缓解滤泡效应。

    

    推荐算法在塑造我们的媒体选择方面起着至关重要的作用，因此了解它们对用户行为的长期影响至关重要。这些算法通常与两个关键结果相关联：同质化，即使用户具有不同的基本偏好，也会消费相似的内容，以及滤泡效应，即具有不同偏好的个人仅消费与其偏好一致的内容（与其他用户几乎没有重叠）。先前的研究假设同质化和滤泡效应之间存在权衡，并展示个性化推荐通过促进同质化来缓解滤泡效应。然而，由于这一对这两种效应之间的权衡的假设，先前的工作无法发展出一种更为细致的看法，即推荐系统可能如何独立影响同质化和滤泡效应。我们对同质化和滤泡效应进行了更精细的定义

    arXiv:2402.15013v1 Announce Type: cross  Abstract: Recommendation algorithms play a pivotal role in shaping our media choices, which makes it crucial to comprehend their long-term impact on user behavior. These algorithms are often linked to two critical outcomes: homogenization, wherein users consume similar content despite disparate underlying preferences, and the filter bubble effect, wherein individuals with differing preferences only consume content aligned with their preferences (without much overlap with other users). Prior research assumes a trade-off between homogenization and filter bubble effects and then shows that personalized recommendations mitigate filter bubbles by fostering homogenization. However, because of this assumption of a tradeoff between these two effects, prior work cannot develop a more nuanced view of how recommendation systems may independently impact homogenization and filter bubble effects. We develop a more refined definition of homogenization and the 
    
[^12]: 从排名中崛起的公平性：HITS和PageRank在同质网络上的应用

    Fairness Rising from the Ranks: HITS and PageRank on Homophilic Networks

    [https://arxiv.org/abs/2402.13787](https://arxiv.org/abs/2402.13787)

    本文研究了链接分析算法在阻止少数群体在网络中达到高排名位置的条件，发现PageRank能够平衡排名中少数群体的代表性，而HITS则在同构网络中通过新颖的理论分析放大了现有的偏见。

    

    在本文中，我们研究了链接分析算法在阻止少数群体达到高排名位置的条件。我们发现，使用中心性度量标准的最常见链接算法，如PageRank和HITS，在网络中可能再现甚至放大对少数群体的偏见。然而，它们的行为有所不同：一方面，我们凭经验证明，PageRank在大部分排名位置上反映了度分布，并且可以平衡少数群体在排名靠前的节点中的代表性；另一方面，我们发现HITS通过新颖的理论分析在同构网络中放大了现有的偏见，支撑证据为实证结果。我们发现HITS中偏见放大的根本原因是网络中存在的同质性水平，通过一个具有两个社区的不断发展的网络模型进行建模。我们以合成和真实数据集阐明了我们的理论分析。

    arXiv:2402.13787v1 Announce Type: cross  Abstract: In this paper, we investigate the conditions under which link analysis algorithms prevent minority groups from reaching high ranking slots. We find that the most common link-based algorithms using centrality metrics, such as PageRank and HITS, can reproduce and even amplify bias against minority groups in networks. Yet, their behavior differs: one one hand, we empirically show that PageRank mirrors the degree distribution for most of the ranking positions and it can equalize representation of minorities among the top ranked nodes; on the other hand, we find that HITS amplifies pre-existing bias in homophilic networks through a novel theoretical analysis, supported by empirical results. We find the root cause of bias amplification in HITS to be the level of homophily present in the network, modeled through an evolving network model with two communities. We illustrate our theoretical analysis on both synthetic and real datasets and we pr
    
[^13]: SupplyGraph: 使用图神经网络进行供应链规划的基准数据集

    SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks. (arXiv:2401.15299v1 [cs.LG])

    [http://arxiv.org/abs/2401.15299](http://arxiv.org/abs/2401.15299)

    SupplyGraph是一个基准数据集，用于使用图神经网络进行供应链规划。该数据集包含了来自孟加拉国一家领先快速消费品公司的实际数据，用于优化、预测和解决供应链问题。数据集中的时间数据作为节点特征，可用于销售预测、生产计划和故障识别。

    

    图神经网络（GNNs）在不同领域如运输、生物信息学、语言处理和计算机视觉中取得了重要进展。然而，在将GNNs应用于供应链网络方面，目前尚缺乏研究。供应链网络在结构上类似于图形，使其成为应用GNN方法的理想选择。这为优化、预测和解决供应链问题开辟了无限可能。然而，此方法的一个主要障碍在于缺乏真实世界的基准数据集以促进使用GNN来研究和解决供应链问题。为了解决这个问题，我们提供了一个来自孟加拉国一家领先的快速消费品公司的实际基准数据集，该数据集侧重于用于生产目的的供应链规划的时间任务。该数据集包括时间数据作为节点特征，以实现销售预测、生产计划和故障识别。

    Graph Neural Networks (GNNs) have gained traction across different domains such as transportation, bio-informatics, language processing, and computer vision. However, there is a noticeable absence of research on applying GNNs to supply chain networks. Supply chain networks are inherently graph-like in structure, making them prime candidates for applying GNN methodologies. This opens up a world of possibilities for optimizing, predicting, and solving even the most complex supply chain problems. A major setback in this approach lies in the absence of real-world benchmark datasets to facilitate the research and resolution of supply chain problems using GNNs. To address the issue, we present a real-world benchmark dataset for temporal tasks, obtained from one of the leading FMCG companies in Bangladesh, focusing on supply chain planning for production purposes. The dataset includes temporal data as node features to enable sales predictions, production planning, and the identification of fa
    
[^14]: 基于高斯过程回归的密度用户表示方法用于多兴趣个性化检索

    Density-based User Representation through Gaussian Process Regression for Multi-interest Personalized Retrieval. (arXiv:2310.20091v1 [cs.IR])

    [http://arxiv.org/abs/2310.20091](http://arxiv.org/abs/2310.20091)

    本研究引入了一种基于密度的用户表示(DURs)，利用高斯过程回归实现了有效的多兴趣推荐和检索。该方法不仅能够捕捉用户的兴趣变化，还具备不确定性感知能力，并且适用于大量用户的规模。

    

    在设计个性化推荐系统中，准确建模用户的各种多样化和动态的兴趣仍然是一个重大挑战。现有的用户建模方法，如单点和多点表示，存在准确性、多样性、计算成本和适应性方面的局限性。为了克服这些不足，我们引入了一种新颖的模型——基于密度的用户表示(DURs)，它利用高斯过程回归实现有效的多兴趣推荐和检索。我们的方法GPR4DUR利用DURs来捕捉用户的兴趣变化，无需手动调整，同时具备不确定性感知能力，并且适用于大量用户的规模。使用真实世界的离线数据集进行的实验证实了GPR4DUR的适应性和效率，而使用模拟用户的在线实验则证明了它通过有效利用模型的不确定性，能够解决探索-开发的平衡问题。

    Accurate modeling of the diverse and dynamic interests of users remains a significant challenge in the design of personalized recommender systems. Existing user modeling methods, like single-point and multi-point representations, have limitations w.r.t. accuracy, diversity, computational cost, and adaptability. To overcome these deficiencies, we introduce density-based user representations (DURs), a novel model that leverages Gaussian process regression for effective multi-interest recommendation and retrieval. Our approach, GPR4DUR, exploits DURs to capture user interest variability without manual tuning, incorporates uncertainty-awareness, and scales well to large numbers of users. Experiments using real-world offline datasets confirm the adaptability and efficiency of GPR4DUR, while online experiments with simulated users demonstrate its ability to address the exploration-exploitation trade-off by effectively utilizing model uncertainty.
    
[^15]: 召回率、鲁棒性和词典式评估

    Recall, Robustness, and Lexicographic Evaluation. (arXiv:2302.11370v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2302.11370](http://arxiv.org/abs/2302.11370)

    该论文从正式的角度反思了排名中召回率的测量问题，提出召回方向的概念和词典式方法，并分析了其鲁棒性。

    

    研究人员使用召回率来评估各种检索、推荐和机器学习任务中的排名。尽管在集合评估中有关召回率的俗语解释，但研究社区远未理解排名召回率的原理。对召回率缺乏原理理解或动机导致信息检索社区批评召回率是否有用作为一个指标。在这个背景下，我们从正式的角度反思排名中召回率的测量问题。我们的分析由三个原则组成：召回率、鲁棒性和词典式评估。首先，我们正式定义“召回方向”为敏感于底部排名相关条目移动的度量。其次，我们从可能的搜索者和内容提供者的鲁棒性角度分析了我们的召回方向概念。最后，我们通过开发一个实用的词典式方法来扩展对召回的概念和理论处理。

    Researchers use recall to evaluate rankings across a variety of retrieval, recommendation, and machine learning tasks. While there is a colloquial interpretation of recall in set-based evaluation, the research community is far from a principled understanding of recall metrics for rankings. The lack of principled understanding of or motivation for recall has resulted in criticism amongst the retrieval community that recall is useful as a measure at all. In this light, we reflect on the measurement of recall in rankings from a formal perspective. Our analysis is composed of three tenets: recall, robustness, and lexicographic evaluation. First, we formally define `recall-orientation' as sensitivity to movement of the bottom-ranked relevant item. Second, we analyze our concept of recall orientation from the perspective of robustness with respect to possible searchers and content providers. Finally, we extend this conceptual and theoretical treatment of recall by developing a practical pref
    
[^16]: 基于大数据的现代推荐系统综述

    A Survey on Modern Recommendation System based on Big Data. (arXiv:2206.02631v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2206.02631](http://arxiv.org/abs/2206.02631)

    这份综述全面调研了基于大数据的现代推荐系统的发展和挑战，总结了四种主要类型的推荐技术，并指出了未来研究的潜在领域。

    

    本综述全面探索了推荐系统的发展和当前状态，这些系统已广泛整合到各种网络应用中。它重点关注个性化推荐策略在在线产品或服务中的进展。我们将推荐技术分为四种主要类型：基于内容的、协同过滤的、基于知识的和混合的，每种类型都解决了独特的情景。本综述详细审视了推荐系统的历史背景和最新的创新方法，特别是那些使用大数据的方法。此外，本综述还确定并讨论了现代推荐系统面临的关键挑战，如数据稀疏性、可扩展性问题以及对推荐的多样性需求。综述最后强调了这些挑战作为未来研究的潜在领域。

    This survey provides an exhaustive exploration of the evolution and current state of recommendation systems, which have seen widespread integration in various web applications. It focuses on the advancement of personalized recommendation strategies for online products or services. We categorize recommendation techniques into four primary types: content-based, collaborative filtering-based, knowledge-based, and hybrid-based, each addressing unique scenarios. The survey offers a detailed examination of the historical context and the latest innovative approaches in recommendation systems, particularly those employing big data. Additionally, it identifies and discusses key challenges faced by modern recommendation systems, such as data sparsity, scalability issues, and the need for diversity in recommendations. The survey concludes by highlighting these challenges as potential areas for fruitful future research in the field.
    

