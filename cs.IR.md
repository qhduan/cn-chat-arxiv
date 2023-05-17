# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Large Language Models are Built-in Autoregressive Search Engines.](http://arxiv.org/abs/2305.09612) | 本篇论文指出大型语言模型可以作为内置搜索引擎，通过提供一些上下文演示直接生成Web URLs，在文档检索中表现出色。 |
| [^2] | [Robust and lightweight audio fingerprint for Automatic Content Recognition.](http://arxiv.org/abs/2305.09559) | 本文提出了一种鲁棒且轻量级的音频指纹识别系统，可用于自动内容识别，具有高度可扩展性和低功耗，并在专有ACR数据集精度、检索速度、内存使用和鲁棒性等所有评估指标上均显著优于基于最小哈希的音频指纹。 |
| [^3] | [Life of PII -- A PII Obfuscation Transformer.](http://arxiv.org/abs/2305.09550) | “Life of PII”是一种新颖的混淆变换器框架，用于将PII转化为人造PII同时尽可能地保留原始信息、意图和上下文，使我们能够有选择地混淆文档中的敏感信息，同时保留文档的统计和语义特性。 |
| [^4] | [Consumer-side Fairness in Recommender Systems: A Systematic Survey of Methods and Evaluation.](http://arxiv.org/abs/2305.09330) | 推荐系统的公平性问题越来越引起人们的关注，尤其是用户公平性，已经提出了很多解决方案来减轻用户在使用推荐系统过程中体验到的歧视问题。 |
| [^5] | [Fairness and Diversity in Information Access Systems.](http://arxiv.org/abs/2305.09319) | 本论文探讨了信息访问系统中多样性和公平性的关系。多样性和公平性是两个独立但密切相关的概念，推动多样性可以有效地实现公平。 |
| [^6] | [Hybrid and Collaborative Passage Reranking.](http://arxiv.org/abs/2305.09313) | 该论文提出了一种名为HybRank的混合与协作的段落再排序方法，通过利用上游检索器的相似性度量实现段落协作，再利用稀疏和密集检索器的词汇和语义属性进行重新排序，该方法可以增强包括先前被重新排序的段落列表在内的任意段落列表，并在实验证明了性能稳定的提升。 |
| [^7] | [HyHTM: Hyperbolic Geometry based Hierarchical Topic Models.](http://arxiv.org/abs/2305.09258) | HyHTM是一种基于双曲几何的层次主题模型，通过将双曲几何中的层次信息纳入主题模型中，显式地建模主题层次结构。相较于传统方法，HyHTM更好地关注主题之间的父子关系，并产生了连贯的主题层次结构。同时，HyHTM的计算速度更快，内存占用更小。 |
| [^8] | [Text2Cohort: Democratizing the NCI Imaging Data Commons with Natural Language Cohort Discovery.](http://arxiv.org/abs/2305.07637) | Text2Cohort是一个基于大语言模型的工具箱，可以将用户输入转化为IDC数据库查询，促进自然语言队列发现，减少研究人员查询IDC数据库的学习曲线，实现了癌症成像数据的民主化。 |
| [^9] | [Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation.](http://arxiv.org/abs/2304.14668) | 本研究提出了一种基于对比知识蒸馏的集成建模方法EMKD，它采用多个并行网络作为序列编码器，在序列推荐中根据所有网络的输出分布推荐物品。实验证明，EMKD在两个真实世界数据集上的表现显著优于最先进的方法。 |
| [^10] | [Meta-optimized Contrastive Learning for Sequential Recommendation.](http://arxiv.org/abs/2304.07763) | 本文提出了 MCLRec 模型，该模型在数据增强和可学习模型增强操作的基础上，解决了现有对比学习方法难以推广和训练数据不足的问题。 |
| [^11] | [BERT4Loc: BERT for Location -- POI Recommender System.](http://arxiv.org/abs/2208.01375) | BERT4Loc是一种基于BERT的位置推荐系统，将地理位置信息和用户偏好相结合，提供个性化的基于位置的建议，相较于预测序列中下一个POI的模型，提供更相关的推荐，在基准数据集上实现了较好的性能。 |
| [^12] | [UNIQORN: Unified Question Answering over RDF Knowledge Graphs and Natural Language Text.](http://arxiv.org/abs/2108.08614) | 本文提出了一个名为UNIQORN的问答系统，它能够无缝地处理RDF数据和文本，使用fine-tuned BERT模型为问题构建上下文图，并使用图算法确定与问题相关的子图来回答问题。 |

# 详细

[^1]: 大型语言模型是内置的自回归搜索引擎

    Large Language Models are Built-in Autoregressive Search Engines. (arXiv:2305.09612v1 [cs.CL])

    [http://arxiv.org/abs/2305.09612](http://arxiv.org/abs/2305.09612)

    本篇论文指出大型语言模型可以作为内置搜索引擎，通过提供一些上下文演示直接生成Web URLs，在文档检索中表现出色。

    

    文档检索是标准网络搜索引擎的关键阶段。现有的双编码器密集检索器独立地获取问题和文档的表示，只允许它们之间的浅层交互。为了克服这个限制，最近的自回归搜索引擎通过直接生成候选池中相关文档的标识符来替换双编码器架构。然而，这种自回归搜索引擎的训练成本随着候选文档数量的增加而急剧上升。在本文中，我们发现大型语言模型（LLM）可以遵循人类指示直接生成文档检索的URL。令人惊讶的是，当提供一些{Query-URL}对作为上下文演示时，LLMs可以生成Web URL，其中近90％的相应文档包含开放域问题的正确答案。这样，LLMs可以被认为是内置搜索引擎，因为它们没有明确训练以映射问题和文档之间的相关性。

    Document retrieval is a key stage of standard Web search engines. Existing dual-encoder dense retrievers obtain representations for questions and documents independently, allowing for only shallow interactions between them. To overcome this limitation, recent autoregressive search engines replace the dual-encoder architecture by directly generating identifiers for relevant documents in the candidate pool. However, the training cost of such autoregressive search engines rises sharply as the number of candidate documents increases. In this paper, we find that large language models (LLMs) can follow human instructions to directly generate URLs for document retrieval.  Surprisingly, when providing a few {Query-URL} pairs as in-context demonstrations, LLMs can generate Web URLs where nearly 90\% of the corresponding documents contain correct answers to open-domain questions. In this way, LLMs can be thought of as built-in search engines, since they have not been explicitly trained to map qu
    
[^2]: 自动内容识别中的鲁棒且轻量级音频指纹识别系统

    Robust and lightweight audio fingerprint for Automatic Content Recognition. (arXiv:2305.09559v1 [cs.SD])

    [http://arxiv.org/abs/2305.09559](http://arxiv.org/abs/2305.09559)

    本文提出了一种鲁棒且轻量级的音频指纹识别系统，可用于自动内容识别，具有高度可扩展性和低功耗，并在专有ACR数据集精度、检索速度、内存使用和鲁棒性等所有评估指标上均显著优于基于最小哈希的音频指纹。

    

    本研究提出了一种新的音频指纹识别系统，用于自动内容识别（ACR）。通过使用信号处理技术和统计变换，我们的方法生成音频片段的紧凑指纹，这些指纹对现实世界中存在的噪声降级具有鲁棒性。该系统具有高度可扩展性，能够使用来自数百万台电视的指纹识别数千小时的内容。指纹的高时间相关性和利用现有的GPU兼容的近似最近邻（ANN）搜索算法使这一点成为可能。此外，指纹生成可以在计算受限的低功耗设备上运行，使其可以被广泛应用。

    This research paper presents a novel audio fingerprinting system for Automatic Content Recognition (ACR). By using signal processing techniques and statistical transformations, our proposed method generates compact fingerprints of audio segments that are robust to noise degradations present in real-world audio. The system is designed to be highly scalable, with the ability to identify thousands of hours of content using fingerprints generated from millions of TVs. The fingerprint's high temporal correlation and utilization of existing GPU-compatible Approximate Nearest Neighbour (ANN) search algorithms make this possible. Furthermore, the fingerprint generation can run on low-power devices with limited compute, making it accessible to a wide range of applications. Experimental results show improvements in our proposed system compared to a min-hash based audio fingerprint on all evaluated metrics, including accuracy on proprietary ACR datasets, retrieval speed, memory usage, and robustn
    
[^3]: PII的生命--一种PII混淆变换器

    Life of PII -- A PII Obfuscation Transformer. (arXiv:2305.09550v1 [cs.CL])

    [http://arxiv.org/abs/2305.09550](http://arxiv.org/abs/2305.09550)

    “Life of PII”是一种新颖的混淆变换器框架，用于将PII转化为人造PII同时尽可能地保留原始信息、意图和上下文，使我们能够有选择地混淆文档中的敏感信息，同时保留文档的统计和语义特性。

    

    在当今大型语言模型和数据驱动服务的世界中，保护敏感信息至关重要。一种常见的方法是使用数据扰动技术来减少(敏感)个人身份识别信息(PII)数据的过度实用性，同时保持其统计和语义特性。数据扰动方法经常导致显着的信息损失，使它们难以使用。在本文中，我们提出了“PII的生命”--一种新颖的混淆变换器框架，用于将PII转化为人造PII同时尽可能地保留原始信息、意图和上下文。我们的方法包括一个API来与给定的文档进行接口，一个基于配置的混淆器和一个基于Transformer架构的模型，在自然语言处理任务和LLMs中表现出高的上下文保存性能。我们的基于Transformer的方法学习了原始PII和其转换后的人造PII对应的映射，使我们能够有选择地混淆文档中的敏感信息，同时保留文档的统计和语义特性。

    Protecting sensitive information is crucial in today's world of Large Language Models (LLMs) and data-driven services. One common method used to preserve privacy is by using data perturbation techniques to reduce overreaching utility of (sensitive) Personal Identifiable Information (PII) data while maintaining its statistical and semantic properties. Data perturbation methods often result in significant information loss, making them impractical for use. In this paper, we propose 'Life of PII', a novel Obfuscation Transformer framework for transforming PII into faux-PII while preserving the original information, intent, and context as much as possible. Our approach includes an API to interface with the given document, a configuration-based obfuscator, and a model based on the Transformer architecture, which has shown high context preservation and performance in natural language processing tasks and LLMs.  Our Transformer-based approach learns mapping between the original PII and its tra
    
[^4]: 推荐系统中的用户公平性: 方法和评估的系统调查

    Consumer-side Fairness in Recommender Systems: A Systematic Survey of Methods and Evaluation. (arXiv:2305.09330v1 [cs.IR])

    [http://arxiv.org/abs/2305.09330](http://arxiv.org/abs/2305.09330)

    推荐系统的公平性问题越来越引起人们的关注，尤其是用户公平性，已经提出了很多解决方案来减轻用户在使用推荐系统过程中体验到的歧视问题。

    

    在数字化水平不断提高的当前社会中，面临着可扩展性方面的巨大挑战。推荐系统已经成为帮助用户导航日益增长的数据量，以及帮助供应商向感兴趣的用户营销产品的必不可少的工具。机器学习方法中的歧视问题日益突出，这促使学术界和工业界研究如何确保推荐系统的公平性。在推荐系统中，这些问题在职业推荐中得到了很好的体现，历史数据中的偏见可能导致推荐系统将一个性别与较低的工资或刻板印象联系起来。特别地，用户公平性关注如何减轻用户在使用推荐系统过程中体验到的歧视问题，该领域已经出现了很多不同的方法来解决不同类型的歧视。所述歧视的性质取决于所处的情境。

    In the current landscape of ever-increasing levels of digitalization, we are facing major challenges pertaining to scalability. Recommender systems have become irreplaceable both for helping users navigate the increasing amounts of data and, conversely, aiding providers in marketing products to interested users. The growing awareness of discrimination in machine learning methods has recently motivated both academia and industry to research how fairness can be ensured in recommender systems. For recommender systems, such issues are well exemplified by occupation recommendation, where biases in historical data may lead to recommender systems relating one gender to lower wages or to the propagation of stereotypes. In particular, consumer-side fairness, which focuses on mitigating discrimination experienced by users of recommender systems, has seen a vast number of diverse approaches for addressing different types of discrimination. The nature of said discrimination depends on the setting 
    
[^5]: 信息访问系统中的公平性和多样性

    Fairness and Diversity in Information Access Systems. (arXiv:2305.09319v1 [cs.IR])

    [http://arxiv.org/abs/2305.09319](http://arxiv.org/abs/2305.09319)

    本论文探讨了信息访问系统中多样性和公平性的关系。多样性和公平性是两个独立但密切相关的概念，推动多样性可以有效地实现公平。

    

    欧洲委员会（EC）设立的人工智能高级专家组提出了实现值得信赖的人工智能的七个重要要求之一是多样性、非歧视和公平性。本文试图通过关注信息访问系统和排名文献，阐述多样性和公平性两个独立概念的密切相关性。这两个概念不应该被互换使用，因为它们代表了两个不同的价值，但我们认为它们也不能被认为是完全不相关的。推动多样性并不意味着公平，但促进多样性确实可以有效地实现公平的结果，这是几种方法提出的直觉，旨在减少不平等问题。

    Among the seven key requirements to achieve trustworthy AI proposed by the High-Level Expert Group on Artificial Intelligence (AI-HLEG) established by the European Commission (EC), the fifth requirement ("Diversity, non-discrimination and fairness") declares: "In order to achieve Trustworthy AI, we must enable inclusion and diversity throughout the entire AI system's life cycle. [...] This requirement is closely linked with the principle of fairness". In this paper, we try to shed light on how closely these two distinct concepts, diversity and fairness, may be treated by focusing on information access systems and ranking literature. These concepts should not be used interchangeably because they do represent two different values, but what we argue is that they also cannot be considered totally unrelated or divergent. Having diversity does not imply fairness, but fostering diversity can effectively lead to fair outcomes, an intuition behind several methods proposed to mitigate the dispar
    
[^6]: 混合与协作的段落再排序方法

    Hybrid and Collaborative Passage Reranking. (arXiv:2305.09313v1 [cs.IR])

    [http://arxiv.org/abs/2305.09313](http://arxiv.org/abs/2305.09313)

    该论文提出了一种名为HybRank的混合与协作的段落再排序方法，通过利用上游检索器的相似性度量实现段落协作，再利用稀疏和密集检索器的词汇和语义属性进行重新排序，该方法可以增强包括先前被重新排序的段落列表在内的任意段落列表，并在实验证明了性能稳定的提升。

    

    在段落检索系统中，初始检索结果可能不尽如人意，需要通过重新排序方案进行改善。现有的段落重新排序方案主要集中于丰富查询和每个段落之间的交互，忽略了在初始检索列表中排名靠前的多个段落之间的上下文关系。为解决这个问题，我们提出了一种混合与协作的段落再排序方法（HybRank），该方法利用上游检索器的相似性度量进行段落协作，并结合稀疏和密集检索器的词汇和语义属性进行重新排序。此外，基于现成的检索器特征，HybRank是一个插件再排序器，能够增强包括先前重新排序的段落列表在内的任意段落列表。大量实验证明了比普遍的检索和再排序方法性能稳定的提升，并验证了HybRank的核心组件的有效性。

    In passage retrieval system, the initial passage retrieval results may be unsatisfactory, which can be refined by a reranking scheme. Existing solutions to passage reranking focus on enriching the interaction between query and each passage separately, neglecting the context among the top-ranked passages in the initial retrieval list. To tackle this problem, we propose a Hybrid and Collaborative Passage Reranking (HybRank) method, which leverages the substantial similarity measurements of upstream retrievers for passage collaboration and incorporates the lexical and semantic properties of sparse and dense retrievers for reranking. Besides, built on off-the-shelf retriever features, HybRank is a plug-in reranker capable of enhancing arbitrary passage lists including previously reranked ones. Extensive experiments demonstrate the stable improvements of performance over prevalent retrieval and reranking methods, and verify the effectiveness of the core components of HybRank.
    
[^7]: HyHTM: 基于双曲几何的层次主题模型

    HyHTM: Hyperbolic Geometry based Hierarchical Topic Models. (arXiv:2305.09258v1 [cs.IR])

    [http://arxiv.org/abs/2305.09258](http://arxiv.org/abs/2305.09258)

    HyHTM是一种基于双曲几何的层次主题模型，通过将双曲几何中的层次信息纳入主题模型中，显式地建模主题层次结构。相较于传统方法，HyHTM更好地关注主题之间的父子关系，并产生了连贯的主题层次结构。同时，HyHTM的计算速度更快，内存占用更小。

    

    层次主题模型对于发现文档集合中的主题层次结构非常有用。然而，传统的层次主题模型常常产生低层次主题与其高层次主题无关且不够具体的层次结构。此外，这些方法计算成本较高。我们提出了一种名为 HyHTM 的双曲几何层次主题模型，通过将双曲几何中的层次信息纳入主题模型中，显式地建模主题层次结构，从而解决了这些限制。与四个基线做实验结果表明，HyHTM 可以更好地关注主题之间父子关系。HyHTM 产生连贯的主题层次结构，从通用的高层次主题到具体的低层次主题。此外，我们的模型计算速度更快，内存占用更小。我们已经公开了算法的源代码。

    Hierarchical Topic Models (HTMs) are useful for discovering topic hierarchies in a collection of documents. However, traditional HTMs often produce hierarchies where lowerlevel topics are unrelated and not specific enough to their higher-level topics. Additionally, these methods can be computationally expensive. We present HyHTM - a Hyperbolic geometry based Hierarchical Topic Models - that addresses these limitations by incorporating hierarchical information from hyperbolic geometry to explicitly model hierarchies in topic models. Experimental results with four baselines show that HyHTM can better attend to parent-child relationships among topics. HyHTM produces coherent topic hierarchies that specialise in granularity from generic higher-level topics to specific lowerlevel topics. Further, our model is significantly faster and leaves a much smaller memory footprint than our best-performing baseline.We have made the source code for our algorithm publicly accessible.
    
[^8]: Text2Cohort: 自然语言队列发现对癌症影像数据共享平台的民主化

    Text2Cohort: Democratizing the NCI Imaging Data Commons with Natural Language Cohort Discovery. (arXiv:2305.07637v1 [cs.LG])

    [http://arxiv.org/abs/2305.07637](http://arxiv.org/abs/2305.07637)

    Text2Cohort是一个基于大语言模型的工具箱，可以将用户输入转化为IDC数据库查询，促进自然语言队列发现，减少研究人员查询IDC数据库的学习曲线，实现了癌症成像数据的民主化。

    

    影像数据共享平台(IDC)是一个基于云的数据库，为研究人员提供开放获取的癌症成像数据和分析工具，旨在促进医学成像研究中的协作。然而，由于其复杂和技术性质，查询IDC数据库以进行队列发现和访问成像数据对研究人员来说具有显著的学习曲线。我们开发了基于大语言模型（LLM）的Text2Cohort工具箱，通过提示工程将用户输入转化为IDC数据库查询，并将查询的响应返回给用户，以促进自然语言队列发现。此外，实现了自动校正以解决查询中的语法和语义错误，通过将错误传回模型进行解释和校正。我们对50个自然语言用户输入进行了Text2Cohort评估，范围从信息提取到队列发现。结果查询和输出由两位计算机科学家进行了确认。

    The Imaging Data Commons (IDC) is a cloud-based database that provides researchers with open access to cancer imaging data and tools for analysis, with the goal of facilitating collaboration in medical imaging research. However, querying the IDC database for cohort discovery and access to imaging data has a significant learning curve for researchers due to its complex and technical nature. We developed Text2Cohort, a large language model (LLM) based toolkit to facilitate natural language cohort discovery by translating user input into IDC database queries through prompt engineering and returning the query's response to the user. Furthermore, autocorrection is implemented to resolve syntax and semantic errors in queries by passing the errors back to the model for interpretation and correction. We evaluate Text2Cohort on 50 natural language user inputs ranging from information extraction to cohort discovery. The resulting queries and outputs were verified by two computer scientists to me
    
[^9]: 基于对比知识蒸馏的集成建模在序列推荐中的应用

    Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation. (arXiv:2304.14668v1 [cs.IR])

    [http://arxiv.org/abs/2304.14668](http://arxiv.org/abs/2304.14668)

    本研究提出了一种基于对比知识蒸馏的集成建模方法EMKD，它采用多个并行网络作为序列编码器，在序列推荐中根据所有网络的输出分布推荐物品。实验证明，EMKD在两个真实世界数据集上的表现显著优于最先进的方法。

    

    序列推荐旨在捕捉用户的动态兴趣，预测用户下一次的偏好物品。多数方法使用深度神经网络作为序列编码器生成用户和物品表示。现有工作主要侧重于设计更强的序列编码器。然而，很少有尝试使用训练一组网络作为序列编码器的方法，这比单个网络更强大，因为一组并行网络可以产生多样化的预测结果，从而获得更好的准确性。本文提出了一种基于对比知识蒸馏的集成建模方法，即EMKD，在序列推荐中使用多个并行网络作为序列编码器，并根据所有这些网络的输出分布推荐物品。为了促进并行网络之间的知识转移，我们提出了一种新颖的对比知识蒸馏方法，它将知识从教师网络转移到多个学生网络中。在两个真实世界数据集上的实验表明，我们提出的EMKD显著优于最先进的序列推荐方法和集成基线。

    Sequential recommendation aims to capture users' dynamic interest and predicts the next item of users' preference. Most sequential recommendation methods use a deep neural network as sequence encoder to generate user and item representations. Existing works mainly center upon designing a stronger sequence encoder. However, few attempts have been made with training an ensemble of networks as sequence encoders, which is more powerful than a single network because an ensemble of parallel networks can yield diverse prediction results and hence better accuracy. In this paper, we present Ensemble Modeling with contrastive Knowledge Distillation for sequential recommendation (EMKD). Our framework adopts multiple parallel networks as an ensemble of sequence encoders and recommends items based on the output distributions of all these networks. To facilitate knowledge transfer between parallel networks, we propose a novel contrastive knowledge distillation approach, which performs knowledge tran
    
[^10]: 序列推荐中的元优化对比学习

    Meta-optimized Contrastive Learning for Sequential Recommendation. (arXiv:2304.07763v1 [cs.IR])

    [http://arxiv.org/abs/2304.07763](http://arxiv.org/abs/2304.07763)

    本文提出了 MCLRec 模型，该模型在数据增强和可学习模型增强操作的基础上，解决了现有对比学习方法难以推广和训练数据不足的问题。

    

    对比学习方法是解决稀疏且含噪声推荐数据的一个新兴方法。然而，现有的对比学习方法要么只针对手工制作的数据进行训练数据和模型增强，要么只使用模型增强方法，这使得模型很难推广。为了更好地训练模型，本文提出了一种称为元优化对比学习的模型。该模型结合了数据增强和可学习模型增强操作。

    Contrastive Learning (CL) performances as a rising approach to address the challenge of sparse and noisy recommendation data. Although having achieved promising results, most existing CL methods only perform either hand-crafted data or model augmentation for generating contrastive pairs to find a proper augmentation operation for different datasets, which makes the model hard to generalize. Additionally, since insufficient input data may lead the encoder to learn collapsed embeddings, these CL methods expect a relatively large number of training data (e.g., large batch size or memory bank) to contrast. However, not all contrastive pairs are always informative and discriminative enough for the training processing. Therefore, a more general CL-based recommendation model called Meta-optimized Contrastive Learning for sequential Recommendation (MCLRec) is proposed in this work. By applying both data augmentation and learnable model augmentation operations, this work innovates the standard 
    
[^11]: BERT4Loc：基于BERT的位置推荐系统--POI推荐

    BERT4Loc: BERT for Location -- POI Recommender System. (arXiv:2208.01375v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2208.01375](http://arxiv.org/abs/2208.01375)

    BERT4Loc是一种基于BERT的位置推荐系统，将地理位置信息和用户偏好相结合，提供个性化的基于位置的建议，相较于预测序列中下一个POI的模型，提供更相关的推荐，在基准数据集上实现了较好的性能。

    

    推荐兴趣点（POI）是一项具有挑战性的任务，需要从基于位置的社交媒体平台中提取全面的位置数据。为了提供有效的基于位置的推荐，分析用户的历史行为和偏好是非常重要的。本研究提出了一种复杂的位置感知推荐系统，利用Transformers中的双向编码器表示（BERT）提供个性化的基于位置的建议。我们的模型将地理位置信息和用户偏好相结合，相比于预测序列中下一个POI的模型，提供更相关的推荐。在两个基准数据集上的实验表明，我们基于BERT的模型优于各种最先进的序列模型。此外，我们通过附加实验展示了所提出模型的有效性。

    Recommending points of interest (POIs) is a challenging task that requires extracting comprehensive location data from location-based social media platforms. To provide effective location-based recommendations, it's important to analyze users' historical behavior and preferences. In this study, we present a sophisticated location-aware recommendation system that uses Bidirectional Encoder Representations from Transformers (BERT) to offer personalized location-based suggestions. Our model combines location information and user preferences to provide more relevant recommendations compared to models that predict the next POI in a sequence. Our experiments on two benchmark dataset show that our BERT-based model outperforms various state-of-the-art sequential models. Moreover, we see the effectiveness of the proposed model for quality through additional experiments.
    
[^12]: UNIQORN：统一的RDF知识图谱与自然语言文本问答系统

    UNIQORN: Unified Question Answering over RDF Knowledge Graphs and Natural Language Text. (arXiv:2108.08614v5 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2108.08614](http://arxiv.org/abs/2108.08614)

    本文提出了一个名为UNIQORN的问答系统，它能够无缝地处理RDF数据和文本，使用fine-tuned BERT模型为问题构建上下文图，并使用图算法确定与问题相关的子图来回答问题。

    

    问题回答在知识图谱和其他RDF数据上已经取得了巨大的进展，许多优秀的系统可以为自然语言问题或电报查询提供清晰的答案。其中一些系统将文本源作为附加证据纳入回答过程，但不能计算仅存在于文本中的答案。相反，IR和NLP社区的系统已经解决了有关文本的QA问题，但是这些系统几乎不利用语义数据和知识。本文提出了第一个可以无缝操作混合RDF数据集和文本语料库或单个来源的复杂问题的系统，在统一框架中进行操作。我们的方法称为UNIQORN，通过使用经过精细调整的BERT模型从RDF数据和/或文本语料库中检索与问题相关的证据来动态构建上下文图。结果图通常非常丰富但高度嘈杂。UNIQORN通过用于组Steiner树的图算法来处理这个输入，从而确定与问题相关的子图，进而回答问题。

    Question answering over knowledge graphs and other RDF data has been greatly advanced, with a number of good systems providing crisp answers for natural language questions or telegraphic queries. Some of these systems incorporate textual sources as additional evidence for the answering process, but cannot compute answers that are present in text alone. Conversely, systems from the IR and NLP communities have addressed QA over text, but such systems barely utilize semantic data and knowledge. This paper presents the first system for complex questions that can seamlessly operate over a mixture of RDF datasets and text corpora, or individual sources, in a unified framework. Our method, called UNIQORN, builds a context graph on-the-fly, by retrieving question-relevant evidences from the RDF data and/or a text corpus, using fine-tuned BERT models. The resulting graph is typically rich but highly noisy. UNIQORN copes with this input by a graph algorithm for Group Steiner Trees, that identifi
    

