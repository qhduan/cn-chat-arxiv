# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Comparing Apples to Apples: Generating Aspect-Aware Comparative Sentences from User Review.](http://arxiv.org/abs/2307.03691) | 该论文提出了一个模型，利用用户评论和相关项目特征生成对比评价句子，以帮助用户找到最适合的产品。该模型包括项目编码模块、比较生成模块和个性化解码方法，并通过人类评估验证了生成句子的相关性和真实性。 |
| [^2] | [Undecimated Wavelet Transform for Word Embedded Semantic Marginal Autoencoder in Security improvement and Denoising different Languages.](http://arxiv.org/abs/2307.03679) | 本研究提出了一种将不可降解小波变换与嵌入语义边缘自动编码器相结合的新策略，用于改善多语言安全措施和降噪。该系统通过提取特征并保留数据中的时间和地理链接，成功捕获重要信息，并提高了系统检测异常和区分合法内容与危险威胁的能力。 |
| [^3] | [Structure Guided Multi-modal Pre-trained Transformer for Knowledge Graph Reasoning.](http://arxiv.org/abs/2307.03591) | 本研究提出了一种图结构引导的多模态预训练Transformer用于知识图谱推理。当前的多模态预训练Transformer模型未能充分利用知识图谱的结构信息，限制了其推理性能。 |
| [^4] | [A Network Resource Allocation Recommendation Method with An Improved Similarity Measure.](http://arxiv.org/abs/2307.03399) | 提出了一种名为PIM+RA的网络资源分配推荐方法，该方法通过利用改进的相似度度量和二分网络，实现了个性化推荐和更好的推荐资源分配，显著提升了推荐的准确性、覆盖范围、多样性和新颖性，同时也实现了对长尾物品的有效曝光。 |
| [^5] | [InfoSync: Information Synchronization across Multilingual Semi-structured Tables.](http://arxiv.org/abs/2307.03313) | 该论文提出了一个名为InfoSync的新数据集和一种两步方法，用于跨语言半结构化表格的信息同步。通过信息对齐和信息更新，该方法在InfoSync数据集上获得了高效的性能，验证了其有效性。 |
| [^6] | [Optimal Bandwidth Selection for DENCLUE.](http://arxiv.org/abs/2307.03206) | 本文提出了一种计算DENCLUE算法最优参数的新方法，并在实验部分讨论了其性能。 |
| [^7] | [JEPOO: Highly Accurate Joint Estimation of Pitch, Onset and Offset for Music Information Retrieval.](http://arxiv.org/abs/2306.01304) | 本文提出了一种名为JEPOO的音乐信息检索方法，能够准确估计音高、起始和终止，支持单音高和多音高数据，比现有最先进的方法精度提升高达10.6%，8.3%和10.3%。 |
| [^8] | [Evaluating Embedding APIs for Information Retrieval.](http://arxiv.org/abs/2305.06300) | 本篇论文旨在通过对语义嵌入API在实际检索场景中的分析,为从业者和研究人员找到适当的服务。结果表明，在英语上使用API重新排名BM25的结果是一种预算友好的最优做法。 |
| [^9] | [Bridging the Gap Between Indexing and Retrieval for Differentiable Search Index with Query Generation.](http://arxiv.org/abs/2206.10128) | 本文识别和解决了当前可微搜索索引模型的一个重要问题：在索引和检索过程中存在的数据分布不匹配。为了解决这个问题，提出了一个简单而有效的索引框架。 |

# 详细

[^1]: 将苹果与苹果进行比较：从用户评论生成纵向感知的比较句子

    Comparing Apples to Apples: Generating Aspect-Aware Comparative Sentences from User Review. (arXiv:2307.03691v1 [cs.CL])

    [http://arxiv.org/abs/2307.03691](http://arxiv.org/abs/2307.03691)

    该论文提出了一个模型，利用用户评论和相关项目特征生成对比评价句子，以帮助用户找到最适合的产品。该模型包括项目编码模块、比较生成模块和个性化解码方法，并通过人类评估验证了生成句子的相关性和真实性。

    

    在众多相似的选择中找到最佳产品是非常耗时的。比较句子可以帮助我们以突出的方式对比一个项目与其他项目，在此过程中强调出重要特征。基于用户对一个或多个项目的评论及相关项目特征，我们生成比较评论句子来帮助用户找到最适合的产品。具体来说，我们的模型包括三个连续组件：（i）一个项目编码模块用于对项目进行编码比较，（ii）一个比较生成模块以自回归的方式生成比较句子，（iii）一种用于用户个性化的新型解码方法。我们展示了我们的流程能够生成流畅且多样的比较句子。我们进行了人类评估研究来验证我们生成的句子的相关性和真实性，结果表明我们的算法能够生成相关且真实的比较评论句子。

    It is time-consuming to find the best product among many similar alternatives. Comparative sentences can help to contrast one item from others in a way that highlights important features of an item that stand out. Given reviews of one or multiple items and relevant item features, we generate comparative review sentences to aid users to find the best fit. Specifically, our model consists of three successive components in a transformer: (i) an item encoding module to encode an item for comparison, (ii) a comparison generation module that generates comparative sentences in an autoregressive manner, (iii) a novel decoding method for user personalization. We show that our pipeline generates fluent and diverse comparative sentences. We run experiments on the relevance and fidelity of our generated sentences in a human evaluation study and find that our algorithm creates comparative review sentences that are relevant and truthful.
    
[^2]: 嵌入语义边缘自动编码器中的不可降解小波变换对多语言安全改进和降噪的研究

    Undecimated Wavelet Transform for Word Embedded Semantic Marginal Autoencoder in Security improvement and Denoising different Languages. (arXiv:2307.03679v1 [cs.CL])

    [http://arxiv.org/abs/2307.03679](http://arxiv.org/abs/2307.03679)

    本研究提出了一种将不可降解小波变换与嵌入语义边缘自动编码器相结合的新策略，用于改善多语言安全措施和降噪。该系统通过提取特征并保留数据中的时间和地理链接，成功捕获重要信息，并提高了系统检测异常和区分合法内容与危险威胁的能力。

    

    通过将不可降解小波变换与嵌入语义边缘自动编码器（WESMA）相结合，本研究提出了一种改善安全措施和降噪多种语言的新策略。这些策略的整合旨在解决数据处理应用中的鲁棒性、隐私性和多语言性问题。不可降解小波变换被用作特征提取工具，以识别输入数据中突出的语言模式和结构特性。通过采用这种变换，提议的系统可以在保留数据中的时间和地理链接的同时，成功捕获重要信息。这通过增加系统检测异常、发现隐藏模式以及区分合法内容和危险威胁的能力来改善安全措施。嵌入语义边缘自动编码器还可以作为一个智能框架来降维和降噪数据。

    By combining the undecimated wavelet transform within a Word Embedded Semantic Marginal Autoencoder (WESMA), this research study provides a novel strategy for improving security measures and denoising multiple languages. The incorporation of these strategies is intended to address the issues of robustness, privacy, and multilingualism in data processing applications. The undecimated wavelet transform is used as a feature extraction tool to identify prominent language patterns and structural qualities in the input data. The proposed system may successfully capture significant information while preserving the temporal and geographical links within the data by employing this transform. This improves security measures by increasing the system's ability to detect abnormalities, discover hidden patterns, and distinguish between legitimate content and dangerous threats. The Word Embedded Semantic Marginal Autoencoder also functions as an intelligent framework for dimensionality and noise redu
    
[^3]: 结构引导的多模态预训练Transformer用于知识图谱推理

    Structure Guided Multi-modal Pre-trained Transformer for Knowledge Graph Reasoning. (arXiv:2307.03591v1 [cs.AI])

    [http://arxiv.org/abs/2307.03591](http://arxiv.org/abs/2307.03591)

    本研究提出了一种图结构引导的多模态预训练Transformer用于知识图谱推理。当前的多模态预训练Transformer模型未能充分利用知识图谱的结构信息，限制了其推理性能。

    

    多模态知识图谱(MKGs)直观地组织了各种模式的信息，可以惠及多个实际的下游任务，如推荐系统和视觉问答。然而，大多数MKGs仍然远离完整，这促使了MKG推理模型的兴起。最近，随着通用人工架构的发展，预训练transformer模型引起了越来越多的关注，特别是对于多模态场景。然而，多模态预训练transformer (MPT)用于知识图推理 (KGR) 的研究仍处于早期阶段。作为MKG和其他多模态数据的最大区别，MKG中丰富的结构信息仍然无法在现有的MPT模型中充分利用。大多数模型只将图结构用作匹配与同一实体相连的图像和文本的检索映射。这种方式阻碍了它们的推理性能。为此，我们提出了图结构引导的多模态预训练Transformer用于知识图谱推理的方法。

    Multimodal knowledge graphs (MKGs), which intuitively organize information in various modalities, can benefit multiple practical downstream tasks, such as recommendation systems, and visual question answering. However, most MKGs are still far from complete, which motivates the flourishing of MKG reasoning models. Recently, with the development of general artificial architectures, the pretrained transformer models have drawn increasing attention, especially for multimodal scenarios. However, the research of multimodal pretrained transformer (MPT) for knowledge graph reasoning (KGR) is still at an early stage. As the biggest difference between MKG and other multimodal data, the rich structural information underlying the MKG still cannot be fully leveraged in existing MPT models. Most of them only utilize the graph structure as a retrieval map for matching images and texts connected with the same entity. This manner hinders their reasoning performances. To this end, we propose the graph S
    
[^4]: 一个改进的相似度度量的网络资源分配推荐方法

    A Network Resource Allocation Recommendation Method with An Improved Similarity Measure. (arXiv:2307.03399v1 [cs.IR])

    [http://arxiv.org/abs/2307.03399](http://arxiv.org/abs/2307.03399)

    提出了一种名为PIM+RA的网络资源分配推荐方法，该方法通过利用改进的相似度度量和二分网络，实现了个性化推荐和更好的推荐资源分配，显著提升了推荐的准确性、覆盖范围、多样性和新颖性，同时也实现了对长尾物品的有效曝光。

    

    推荐系统被认为是管理信息过载的有效工具。然而，传统算法在这类系统中的应用主要强调精确的推荐，因此忽视了覆盖率、多样性和物品的新颖性等其他重要方面。这种方法导致长尾物品的曝光率较低。本文提出了一种名为PIM+RA的方法，旨在个性化推荐并更有目的地分配推荐资源。该方法利用包含自连接边和权重的二分网络，并采用改进的Pearson相关系数进行更好的重新分配。PIM+RA的评估不仅在准确性方面显示了显著的改进，而且在推荐的覆盖范围、多样性和新颖性方面也有显著的增强。它通过为长尾物品提供有效曝光的同时，允许自定义参数，实现了更好的推荐频率平衡。

    Recommender systems have been acknowledged as efficacious tools for managing information overload. Nevertheless, conventional algorithms adopted in such systems primarily emphasize precise recommendations and, consequently, overlook other vital aspects like the coverage, diversity, and novelty of items. This approach results in less exposure for long-tail items. In this paper, to personalize the recommendations and allocate recommendation resources more purposively, a method named PIM+RA is proposed. This method utilizes a bipartite network that incorporates self-connecting edges and weights. Furthermore, an improved Pearson correlation coefficient is employed for better redistribution. The evaluation of PIM+RA demonstrates a significant enhancement not only in accuracy but also in coverage, diversity, and novelty of the recommendation. It leads to a better balance in recommendation frequency by providing effective exposure to long-tail items, while allowing customized parameters to ad
    
[^5]: InfoSync：跨多语言半结构化表格的信息同步

    InfoSync: Information Synchronization across Multilingual Semi-structured Tables. (arXiv:2307.03313v1 [cs.CL])

    [http://arxiv.org/abs/2307.03313](http://arxiv.org/abs/2307.03313)

    该论文提出了一个名为InfoSync的新数据集和一种两步方法，用于跨语言半结构化表格的信息同步。通过信息对齐和信息更新，该方法在InfoSync数据集上获得了高效的性能，验证了其有效性。

    

    跨语言半结构化数据的信息同步是具有挑战性的。例如，应该跨语言同步维基百科表格。为解决这个问题，我们引入了一个新的数据集InfoSyncC，并提出了一种两步方法实现表格同步。InfoSync包含了14种语言的10万个以实体为中心的表格（维基百科Infoboxes），其中一部分（3.5K对）是手动注释的。提出的方法包括1）信息对齐来映射行和2）信息更新来更新跨多语言表格中对齐表格中的缺失/过时信息。在InfoSync上进行评估时，信息对齐实现了87.91的F1得分（英文<->非英文）。为了评估信息更新，我们对603个表格对的Infoboxes进行了人工辅助的维基百科编辑。我们的方法在维基百科上取得了77.28%的接受率，显示出了方法的有效性。

    Information Synchronization of semi-structured data across languages is challenging. For instance, Wikipedia tables in one language should be synchronized across languages. To address this problem, we introduce a new dataset InfoSyncC and a two-step method for tabular synchronization. InfoSync contains 100K entity-centric tables (Wikipedia Infoboxes) across 14 languages, of which a subset (3.5K pairs) are manually annotated. The proposed method includes 1) Information Alignment to map rows and 2) Information Update for updating missing/outdated information for aligned tables across multilingual tables. When evaluated on InfoSync, information alignment achieves an F1 score of 87.91 (en <-> non-en). To evaluate information updation, we perform human-assisted Wikipedia edits on Infoboxes for 603 table pairs. Our approach obtains an acceptance rate of 77.28% on Wikipedia, showing the effectiveness of the proposed method.
    
[^6]: DENCLUE的最优带宽选择

    Optimal Bandwidth Selection for DENCLUE. (arXiv:2307.03206v1 [cs.LG])

    [http://arxiv.org/abs/2307.03206](http://arxiv.org/abs/2307.03206)

    本文提出了一种计算DENCLUE算法最优参数的新方法，并在实验部分讨论了其性能。

    

    在现代工业中，聚类算法是算法工程师的日常工作。尽管在2010年之前，聚类算法经历了快速增长，但在深度学习成为机器学习应用的实际工业标准之后，与该研究主题相关的创新停滞不前。2007年，提出了一种名为DENCLUE的基于密度的聚类算法，用于解决非线性数据结构的聚类问题。然而，直到2011年，该算法的参数选择问题仍然被大部分忽视。本文提出了一种计算DENCLUE算法最优参数的新方法，并在实验部分讨论了其性能。

    In modern day industry, clustering algorithms are daily routines of algorithm engineers. Although clustering algorithms experienced rapid growth before 2010. Innovation related to the research topic has stagnated after deep learning became the de facto industrial standard for machine learning applications. In 2007, a density-based clustering algorithm named DENCLUE was invented to solve clustering problem for nonlinear data structures. However, its parameter selection problem was largely neglected until 2011. In this paper, we propose a new approach to compute the optimal parameters for the DENCLUE algorithm, and discuss its performance in the experiment section.
    
[^7]: JEPOO：音乐信息检索中准确估计音高、起始和终止的联合方法

    JEPOO: Highly Accurate Joint Estimation of Pitch, Onset and Offset for Music Information Retrieval. (arXiv:2306.01304v1 [cs.SD])

    [http://arxiv.org/abs/2306.01304](http://arxiv.org/abs/2306.01304)

    本文提出了一种名为JEPOO的音乐信息检索方法，能够准确估计音高、起始和终止，支持单音高和多音高数据，比现有最先进的方法精度提升高达10.6%，8.3%和10.3%。

    

    旋律提取是音乐信息检索中的核心任务，而音高、起始和终止的估计是旋律提取的关键子任务。现有方法的准确性有限，并且只适用于单音高或多音高数据中的一种类型。本文提出了一种名为JEPOO的高度准确的音高、起始和终止联合估计方法。我们通过新颖的模型设计和一种名为帕累托模调损失的优化技术，解决了联合学习优化和处理单音高和多音高数据的挑战。这是第一种能够准确处理单音高和多音高音乐数据，甚至混合类型数据的方法。在广泛的真实数据集上进行的全面实验研究表明，JEPOO在预测音高、起始和终止方面比最先进的方法分别高出10.6％、8.3％和10.3％，同时对于各种类型的数据和乐器具有鲁棒性。

    Melody extraction is a core task in music information retrieval, and the estimation of pitch, onset and offset are key sub-tasks in melody extraction. Existing methods have limited accuracy, and work for only one type of data, either single-pitch or multipitch. In this paper, we propose a highly accurate method for joint estimation of pitch, onset and offset, named JEPOO. We address the challenges of joint learning optimization and handling both single-pitch and multi-pitch data through novel model design and a new optimization technique named Pareto modulated loss with loss weight regularization. This is the first method that can accurately handle both single-pitch and multi-pitch music data, and even a mix of them. A comprehensive experimental study on a wide range of real datasets shows that JEPOO outperforms state-ofthe-art methods by up to 10.6%, 8.3% and 10.3% for the prediction of Pitch, Onset and Offset, respectively, and JEPOO is robust for various types of data and instrument
    
[^8]: 评估信息检索的嵌入式API

    Evaluating Embedding APIs for Information Retrieval. (arXiv:2305.06300v1 [cs.IR])

    [http://arxiv.org/abs/2305.06300](http://arxiv.org/abs/2305.06300)

    本篇论文旨在通过对语义嵌入API在实际检索场景中的分析,为从业者和研究人员找到适当的服务。结果表明，在英语上使用API重新排名BM25的结果是一种预算友好的最优做法。

    

    语言模型不断增大使得其普及化成为了一项挑战，因此许多公司和初创企业通过API向社区提供大型语言模型的访问权限。其中一个适用于密集检索的特定API是语义嵌入式API，其可构建给定文本的向量表示。在拥有越来越多API的情况下，本文旨在分析在实际检索场景中语义嵌入式API以帮助从业者和研究人员根据他们的需求找到适当的服务。具体而言，我们希望调查现有API在领域泛化和多语言检索方面的能力。为此，我们在两个标准基准BEIR和MIRACL上评估了嵌入式API。我们发现，使用API重新排名BM25结果是一种预算友好的方法，并且在英语上最有效，与标准做法即作为第一阶段检索器不同。

    The ever-increasing size of language models curtails their widespread access to the community, thereby galvanizing many companies and startups into offering access to large language models through APIs. One particular API, suitable for dense retrieval, is the semantic embedding API that builds vector representations of a given text. With a growing number of APIs at our disposal, in this paper, our goal is to analyze semantic embedding APIs in realistic retrieval scenarios in order to assist practitioners and researchers in finding suitable services according to their needs. Specifically, we wish to investigate the capabilities of existing APIs on domain generalization and multilingual retrieval. For this purpose, we evaluate the embedding APIs on two standard benchmarks, BEIR, and MIRACL. We find that re-ranking BM25 results using the APIs is a budget-friendly approach and is most effective on English, in contrast to the standard practice, i.e., employing them as first-stage retrievers
    
[^9]: 将索引和检索桥接起来，为具有查询生成的可微搜索索引填补差距

    Bridging the Gap Between Indexing and Retrieval for Differentiable Search Index with Query Generation. (arXiv:2206.10128v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2206.10128](http://arxiv.org/abs/2206.10128)

    本文识别和解决了当前可微搜索索引模型的一个重要问题：在索引和检索过程中存在的数据分布不匹配。为了解决这个问题，提出了一个简单而有效的索引框架。

    

    可微搜索索引(DSI)是一种新兴的信息检索范式。与传统的检索架构不同，其中索引和检索是两个不同的组件，DSI使用单个transformer模型来执行索引和检索。在本文中，我们确定并解决了当前DSI模型的一个重要问题：DSI索引和检索过程之间出现的数据分布不匹配。具体而言，我们认为，在索引过程中，当前DSI方法学习构建长文档的文本与文档标识符之间的连接，但检索过程中使用的查询通常比索引的文档要短得多。当将DSI用于跨语言检索时，这个问题进一步加剧，因为文档文本和查询文本处于不同的语言中。为了解决当前DSI模型的这个根本问题，我们提出了一个简单而有效的DSI索引框架，c

    The Differentiable Search Index (DSI) is an emerging paradigm for information retrieval. Unlike traditional retrieval architectures where index and retrieval are two different and separate components, DSI uses a single transformer model to perform both indexing and retrieval.  In this paper, we identify and tackle an important issue of current DSI models: the data distribution mismatch that occurs between the DSI indexing and retrieval processes. Specifically, we argue that, at indexing, current DSI methods learn to build connections between the text of long documents and the identifier of the documents, but then retrieval of document identifiers is based on queries that are commonly much shorter than the indexed documents. This problem is further exacerbated when using DSI for cross-lingual retrieval, where document text and query text are in different languages.  To address this fundamental problem of current DSI models, we propose a simple yet effective indexing framework for DSI, c
    

