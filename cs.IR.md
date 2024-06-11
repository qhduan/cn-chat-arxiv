# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Are Large Language Models Good at Utility Judgments?](https://arxiv.org/abs/2403.19216) | 大型语言模型（LLMs）在评估段落实用性方面的能力进行了全面研究，实验发现受过良好指导的LLMs可以... |
| [^2] | [Interpretable User Satisfaction Estimation for Conversational Systems with Large Language Models](https://arxiv.org/abs/2403.12388) | 本文提出了一种名为SPUR的方法，通过LLMs更有效地从自然语言话语中提取用户满意度的可解释信号，并能够利用迭代提示框架进行用户满意度评估。 |
| [^3] | [FedHCDR: Federated Cross-Domain Recommendation with Hypergraph Signal Decoupling](https://arxiv.org/abs/2403.02630) | 该研究提出了FedHCDR框架，通过超图信号解耦的方式解决了联邦跨领域推荐中不同领域数据异质性的问题。 |
| [^4] | [Model Editing at Scale leads to Gradual and Catastrophic Forgetting](https://arxiv.org/abs/2401.07453) | 评估了当前模型编辑方法在规模化情况下的表现，发现随着模型被顺序编辑多个事实，它会逐渐遗忘先前的事实及执行下游任务的能力。 |
| [^5] | [Analysis and Validation of Image Search Engines in Histopathology.](http://arxiv.org/abs/2401.03271) | 本文对组织病理学图像搜索引擎进行了分析和验证，对其性能进行了评估。研究发现，某些搜索引擎在效率和速度上表现出色，但精度较低，而其他搜索引擎的准确率较高，但运行效率较低。 |
| [^6] | [Video Recommendation Using Social Network Analysis and User Viewing Patterns.](http://arxiv.org/abs/2308.12743) | 本论文旨在填补现有基于隐性反馈的视频推荐系统研究空白，通过社交网络分析和用户观看模式来构建有效的视频推荐模型。 |
| [^7] | [DAPR: A Benchmark on Document-Aware Passage Retrieval.](http://arxiv.org/abs/2305.13915) | DAPR是一个文档感知段落检索的基准测试，挑战在于如何从长文档中找到正确的段落并返回准确结果。 |
| [^8] | [Mixer: Image to Multi-Modal Retrieval Learning for Industrial Application.](http://arxiv.org/abs/2305.03972) | 提出了一种新的可伸缩高效的图像到跨模态检索范式Mixer，解决了领域差距、跨模态数据对齐和融合、繁杂的数据训练标签以及海量查询和及时响应等问题。 |
| [^9] | [Algorithmic neutrality.](http://arxiv.org/abs/2303.05103) | 研究算法中立性以及与算法偏见的关系，以搜索引擎为案例研究，得出搜索中立性是不可能的结论。 |
| [^10] | [Visual Acuity Prediction on Real-Life Patient Data Using a Machine Learning Based Multistage System.](http://arxiv.org/abs/2204.11970) | 本研究提供了一种使用机器学习技术开发预测模型的多阶段系统，可高精度预测三种眼疾患者的视力变化，并辅助眼科医生进行临床决策和患者咨询。 |

# 详细

[^1]: 大型语言模型擅长实用性判断吗？

    Are Large Language Models Good at Utility Judgments?

    [https://arxiv.org/abs/2403.19216](https://arxiv.org/abs/2403.19216)

    大型语言模型（LLMs）在评估段落实用性方面的能力进行了全面研究，实验发现受过良好指导的LLMs可以...

    

    检索增强生成（RAG）被认为是缓解大型语言模型（LLMs）幻觉问题的一种有前途的方法，并且近期已受到研究人员的广泛关注。由于检索模型在语义理解上的局限性，RAG的成功在很大程度上取决于LLMs识别具有实用性的段落的能力。最近的研究探讨了LLMs评估检索中段落相关性的能力，但对评估支持问答的段落实用性的工作还很有限。在本工作中，我们进行了一项关于LLMs在开放域QA实用性评估方面能力的全面研究。具体而言，我们引入了一个基准测试程序和不同特征的候选段落集合，促进了与五个代表性LLMs的一系列实验。我们的实验证明：（i）受过良好指导的LLMs可以进行...

    arXiv:2403.19216v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) is considered to be a promising approach to alleviate the hallucination issue of large language models (LLMs), and it has received widespread attention from researchers recently. Due to the limitation in the semantic understanding of retrieval models, the success of RAG heavily lies on the ability of LLMs to identify passages with utility. Recent efforts have explored the ability of LLMs to assess the relevance of passages in retrieval, but there has been limited work on evaluating the utility of passages in supporting question answering. In this work, we conduct a comprehensive study about the capabilities of LLMs in utility evaluation for open-domain QA. Specifically, we introduce a benchmarking procedure and collection of candidate passages with different characteristics, facilitating a series of experiments with five representative LLMs. Our experiments reveal that: (i) well-instructed LLMs can di
    
[^2]: 基于大型语言模型的可解释对话系统用户满意度估计

    Interpretable User Satisfaction Estimation for Conversational Systems with Large Language Models

    [https://arxiv.org/abs/2403.12388](https://arxiv.org/abs/2403.12388)

    本文提出了一种名为SPUR的方法，通过LLMs更有效地从自然语言话语中提取用户满意度的可解释信号，并能够利用迭代提示框架进行用户满意度评估。

    

    准确而可解释的用户满意度估计对于了解、评估和持续改进对话系统至关重要。本文表明，与基于特征化的机器学习模型或文本嵌入的现有方法相比，LLMs能够更有效地从自然语言话语中提取用户满意度的可解释信号。此外，LLM可以通过一个迭代提示框架，并利用标记示例的监督进行用户满意度评估。

    arXiv:2403.12388v1 Announce Type: cross  Abstract: Accurate and interpretable user satisfaction estimation (USE) is critical for understanding, evaluating, and continuously improving conversational systems. Users express their satisfaction or dissatisfaction with diverse conversational patterns in both general-purpose (ChatGPT and Bing Copilot) and task-oriented (customer service chatbot) conversational systems. Existing approaches based on featurized ML models or text embeddings fall short in extracting generalizable patterns and are hard to interpret. In this work, we show that LLMs can extract interpretable signals of user satisfaction from their natural language utterances more effectively than embedding-based approaches. Moreover, an LLM can be tailored for USE via an iterative prompting framework using supervision from labeled examples. The resulting method, Supervised Prompting for User satisfaction Rubrics (SPUR), not only has higher accuracy but is more interpretable as it sco
    
[^3]: FedHCDR: 具有超图信号解耦的联邦跨领域推荐

    FedHCDR: Federated Cross-Domain Recommendation with Hypergraph Signal Decoupling

    [https://arxiv.org/abs/2403.02630](https://arxiv.org/abs/2403.02630)

    该研究提出了FedHCDR框架，通过超图信号解耦的方式解决了联邦跨领域推荐中不同领域数据异质性的问题。

    

    近年来，跨领域推荐（CDR）备受关注，利用来自多个领域的用户数据来增强推荐性能。然而，当前的CDR方法需要跨领域共享用户数据，违反了《通用数据保护条例》（GDPR）。因此，已提出了许多联邦跨领域推荐（FedCDR）方法。然而，不同领域间的数据异质性不可避免地影响了联邦学习的整体性能。在这项研究中，我们提出了FedHCDR，一种具有超图信号解耦的新型联邦跨领域推荐框架。具体地，为了解决不同领域之间的数据异质性，我们引入一种称为超图信号解耦（HSD）的方法，将用户特征解耦为领域独有和领域共享特征。该方法采用高通和低通超图滤波器来进行解耦。

    arXiv:2403.02630v1 Announce Type: new  Abstract: In recent years, Cross-Domain Recommendation (CDR) has drawn significant attention, which utilizes user data from multiple domains to enhance the recommendation performance. However, current CDR methods require sharing user data across domains, thereby violating the General Data Protection Regulation (GDPR). Consequently, numerous approaches have been proposed for Federated Cross-Domain Recommendation (FedCDR). Nevertheless, the data heterogeneity across different domains inevitably influences the overall performance of federated learning. In this study, we propose FedHCDR, a novel Federated Cross-Domain Recommendation framework with Hypergraph signal decoupling. Specifically, to address the data heterogeneity across domains, we introduce an approach called hypergraph signal decoupling (HSD) to decouple the user features into domain-exclusive and domain-shared features. The approach employs high-pass and low-pass hypergraph filters to de
    
[^4]: 规模化模型编辑会导致渐进性和突发性遗忘

    Model Editing at Scale leads to Gradual and Catastrophic Forgetting

    [https://arxiv.org/abs/2401.07453](https://arxiv.org/abs/2401.07453)

    评估了当前模型编辑方法在规模化情况下的表现，发现随着模型被顺序编辑多个事实，它会逐渐遗忘先前的事实及执行下游任务的能力。

    

    在大型语言模型中编辑知识是一种具有吸引力的能力，它使我们能够在预训练期间纠正错误学习的事实，同时使用不断增长的新事实列表更新模型。我们认为，为了使模型编辑具有实际效用，我们必须能够对同一模型进行多次编辑。因此，我们评估了当前规模下的模型编辑方法，重点关注两种最先进的方法：ROME 和 MEMIT。我们发现，随着模型被顺序编辑多个事实，它不断地遗忘先前编辑过的事实以及执行下游任务的能力。这种遗忘分为两个阶段--初始的渐进性遗忘阶段，随后是突然或灾难性的遗忘。

    arXiv:2401.07453v2 Announce Type: replace-cross  Abstract: Editing knowledge in large language models is an attractive capability to have which allows us to correct incorrectly learnt facts during pre-training, as well as update the model with an ever-growing list of new facts. While existing model editing techniques have shown promise, they are usually evaluated using metrics for reliability, specificity and generalization over one or few edits. We argue that for model editing to have practical utility, we must be able to make multiple edits to the same model. With this in mind, we evaluate the current model editing methods at scale, focusing on two state of the art methods: ROME and MEMIT. We find that as the model is edited sequentially with multiple facts, it continually forgets previously edited facts and the ability to perform downstream tasks. This forgetting happens in two phases -- an initial gradual but progressive forgetting phase followed by abrupt or catastrophic forgettin
    
[^5]: 组织病理学图像搜索引擎的分析与验证

    Analysis and Validation of Image Search Engines in Histopathology. (arXiv:2401.03271v1 [eess.IV])

    [http://arxiv.org/abs/2401.03271](http://arxiv.org/abs/2401.03271)

    本文对组织病理学图像搜索引擎进行了分析和验证，对其性能进行了评估。研究发现，某些搜索引擎在效率和速度上表现出色，但精度较低，而其他搜索引擎的准确率较高，但运行效率较低。

    

    在组织学和病理学图像档案中搜索相似图像是一项关键任务，可以在各种目的中帮助患者匹配，从分类和诊断到预后和预测。全玻片图像是组织标本的高度详细数字表示，匹配全玻片图像可以作为患者匹配的关键方法。本文对四种搜索方法，视觉词袋（BoVW）、Yottixel、SISH、RetCCL及其一些潜在变种进行了广泛的分析和验证。我们分析了它们的算法和结构，并评估了它们的性能。为了进行评估，我们使用了四个内部数据集（1269位患者）和三个公共数据集（1207位患者），总计超过200,000个属于五个主要部位的38个不同类别/亚型的图像块。某些搜索引擎，例如BoVW，表现出显着的效率和速度，但精度较低。相反，其他搜索引擎例如SISH表现出较高的准确率，但运行效率较低。

    Searching for similar images in archives of histology and histopathology images is a crucial task that may aid in patient matching for various purposes, ranging from triaging and diagnosis to prognosis and prediction. Whole slide images (WSIs) are highly detailed digital representations of tissue specimens mounted on glass slides. Matching WSI to WSI can serve as the critical method for patient matching. In this paper, we report extensive analysis and validation of four search methods bag of visual words (BoVW), Yottixel, SISH, RetCCL, and some of their potential variants. We analyze their algorithms and structures and assess their performance. For this evaluation, we utilized four internal datasets ($1269$ patients) and three public datasets ($1207$ patients), totaling more than $200,000$ patches from $38$ different classes/subtypes across five primary sites. Certain search engines, for example, BoVW, exhibit notable efficiency and speed but suffer from low accuracy. Conversely, searc
    
[^6]: 使用社交网络分析和用户观看模式的视频推荐方法

    Video Recommendation Using Social Network Analysis and User Viewing Patterns. (arXiv:2308.12743v1 [cs.SI])

    [http://arxiv.org/abs/2308.12743](http://arxiv.org/abs/2308.12743)

    本论文旨在填补现有基于隐性反馈的视频推荐系统研究空白，通过社交网络分析和用户观看模式来构建有效的视频推荐模型。

    

    随着视频点播平台的迅猛崛起，用户面临着从大量内容中筛选出与自己喜好相符的节目的挑战。为了解决这个信息过载的困境，视频点播服务越来越多地加入了利用算法分析用户行为并建议个性化内容的推荐系统。然而，大多数现有的推荐系统依赖于用户明确的反馈，如评分和评论，但这种反馈的收集往往困难且耗时。因此，在建立有效的视频推荐模型方面，利用用户的隐性反馈模式可能提供了一条替代途径，避免了对明确评分的需求。然而，现有文献对于基于隐性反馈的推荐系统，特别是在建模视频观看行为方面，尚缺乏足够的探索。因此，本文旨在填补这一研究空白。

    With the meteoric rise of video-on-demand (VOD) platforms, users face the challenge of sifting through an expansive sea of content to uncover shows that closely match their preferences. To address this information overload dilemma, VOD services have increasingly incorporated recommender systems powered by algorithms that analyze user behavior and suggest personalized content. However, a majority of existing recommender systems depend on explicit user feedback in the form of ratings and reviews, which can be difficult and time-consuming to collect at scale. This presents a key research gap, as leveraging users' implicit feedback patterns could provide an alternative avenue for building effective video recommendation models, circumventing the need for explicit ratings. However, prior literature lacks sufficient exploration into implicit feedback-based recommender systems, especially in the context of modeling video viewing behavior. Therefore, this paper aims to bridge this research gap 
    
[^7]: DAPR：文档感知段落检索的基准测试

    DAPR: A Benchmark on Document-Aware Passage Retrieval. (arXiv:2305.13915v1 [cs.IR])

    [http://arxiv.org/abs/2305.13915](http://arxiv.org/abs/2305.13915)

    DAPR是一个文档感知段落检索的基准测试，挑战在于如何从长文档中找到正确的段落并返回准确结果。

    

    最近的神经检索主要关注短文本的排名，并且在处理长文档方面存在挑战。现有的工作主要评估排名段落或整个文档。然而，许多情况下，用户希望从庞大的语料库中找到长文档中的相关段落，例如法律案例，研究论文等，此时段落往往提供很少的文档上下文，这就挑战了当前的方法找到正确的文档并返回准确的结果。为了填补这个空白，我们提出并命名了Document-Aware Passage Retrieval（DAPR）任务，并构建了一个包括来自不同领域的多个数据集的基准测试，涵盖了DAPR和整个文档检索。在实验中，我们通过不同的方法，包括在文档摘要中添加文档级别的内容，汇总段落表示和使用BM25进行混合检索，扩展了最先进的神经段落检索器。这个混合检索系统，总体基准测试显示，我们提出的DAPR任务是一个具有挑战性和重要性的问题，需要进一步研究。

    Recent neural retrieval mainly focuses on ranking short texts and is challenged with long documents. Existing work mainly evaluates either ranking passages or whole documents. However, there are many cases where the users want to find a relevant passage within a long document from a huge corpus, e.g. legal cases, research papers, etc. In this scenario, the passage often provides little document context and thus challenges the current approaches to finding the correct document and returning accurate results. To fill this gap, we propose and name this task Document-Aware Passage Retrieval (DAPR) and build a benchmark including multiple datasets from various domains, covering both DAPR and whole-document retrieval. In experiments, we extend the state-of-the-art neural passage retrievers with document-level context via different approaches including prepending document summary, pooling over passage representations, and hybrid retrieval with BM25. The hybrid-retrieval systems, the overall b
    
[^8]: Mixer: 应用于工业应用的图像到跨模态检索学习

    Mixer: Image to Multi-Modal Retrieval Learning for Industrial Application. (arXiv:2305.03972v1 [cs.IR])

    [http://arxiv.org/abs/2305.03972](http://arxiv.org/abs/2305.03972)

    提出了一种新的可伸缩高效的图像到跨模态检索范式Mixer，解决了领域差距、跨模态数据对齐和融合、繁杂的数据训练标签以及海量查询和及时响应等问题。

    

    跨模态检索一直是电子商务平台和内容分享社交媒体中普遍存在的需求，其中查询是一张图片，文档是具有图片和文本描述的项目。然而，目前这种检索任务仍面临诸多挑战，包括领域差距、跨模态数据对齐和融合、繁杂的数据训练标签以及海量查询和及时响应等问题。为此，我们提出了一种名为Mixer的新型可伸缩和高效的图像查询到跨模态检索学习范式。Mixer通过自适应地整合多模态数据、更高效地挖掘偏斜和嘈杂的数据，并可扩展到高负载量，解决了这些问题。

    Cross-modal retrieval, where the query is an image and the doc is an item with both image and text description, is ubiquitous in e-commerce platforms and content-sharing social media. However, little research attention has been paid to this important application. This type of retrieval task is challenging due to the facts: 1)~domain gap exists between query and doc. 2)~multi-modality alignment and fusion. 3)~skewed training data and noisy labels collected from user behaviors. 4)~huge number of queries and timely responses while the large-scale candidate docs exist. To this end, we propose a novel scalable and efficient image query to multi-modal retrieval learning paradigm called Mixer, which adaptively integrates multi-modality data, mines skewed and noisy data more efficiently and scalable to high traffic. The Mixer consists of three key ingredients: First, for query and doc image, a shared encoder network followed by separate transformation networks are utilized to account for their
    
[^9]: 算法中立性

    Algorithmic neutrality. (arXiv:2303.05103v2 [cs.CY] UPDATED)

    [http://arxiv.org/abs/2303.05103](http://arxiv.org/abs/2303.05103)

    研究算法中立性以及与算法偏见的关系，以搜索引擎为案例研究，得出搜索中立性是不可能的结论。

    

    偏见影响着越来越多掌控我们生活的算法。预测性警务系统错误地高估有色人种社区的犯罪率；招聘算法削弱了合格的女性候选人的机会；人脸识别软件难以识别黑皮肤的面部。算法偏见已经受到了重视，相比之下，算法中立性却基本被忽视了。算法中立性是我的研究主题。我提出了三个问题。算法中立性是什么？算法中立性是否可能？当我们考虑算法中立性时，我们可以从算法偏见中学到什么？为了具体回答这些问题，我选择了一个案例研究：搜索引擎。借鉴关于科学中立性的研究，我认为只有当搜索引擎的排名不受某些价值观的影响时，搜索引擎才是中立的，比如政治意识形态或搜索引擎运营商的经济利益。我认为搜索中立性是不可能的。

    Bias infects the algorithms that wield increasing control over our lives. Predictive policing systems overestimate crime in communities of color; hiring algorithms dock qualified female candidates; and facial recognition software struggles to recognize dark-skinned faces. Algorithmic bias has received significant attention. Algorithmic neutrality, in contrast, has been largely neglected. Algorithmic neutrality is my topic. I take up three questions. What is algorithmic neutrality? Is algorithmic neutrality possible? When we have algorithmic neutrality in mind, what can we learn about algorithmic bias? To answer these questions in concrete terms, I work with a case study: search engines. Drawing on work about neutrality in science, I say that a search engine is neutral only if certain values -- like political ideologies or the financial interests of the search engine operator -- play no role in how the search engine ranks pages. Search neutrality, I argue, is impossible. Its impossibili
    
[^10]: 基于机器学习的多阶段系统对真实患者数据进行视力预测

    Visual Acuity Prediction on Real-Life Patient Data Using a Machine Learning Based Multistage System. (arXiv:2204.11970v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2204.11970](http://arxiv.org/abs/2204.11970)

    本研究提供了一种使用机器学习技术开发预测模型的多阶段系统，可高精度预测三种眼疾患者的视力变化，并辅助眼科医生进行临床决策和患者咨询。

    

    现实生活中，眼科学中的玻璃体手术药物治疗是治疗年龄相关性黄斑变性（AMD）、糖尿病性黄斑水肿（DME）和视网膜静脉阻塞（RVO）相关疾病的一种普遍治疗方法。然而，在真实世界的情况下，由于数据的异质性和不完整性，患者往往会在多年时间内失去视力，尽管接受治疗。本文采用多种IT系统，提出了一种用于研究的数据集成流程，该流程融合了德国一家最佳医疗保健医院的眼科部门的不同IT系统。经过使用机器学习技术开发预测模型，我们实现了对患者视力的预测。我们的结果表明，我们的系统可以为三种疾病的预测提供高准确性。此外，我们还展示了我们的系统可以作为工具，辅助眼科医生进行临床决策和患者咨询。

    In ophthalmology, intravitreal operative medication therapy (IVOM) is a widespread treatment for diseases related to the age-related macular degeneration (AMD), the diabetic macular edema (DME), as well as the retinal vein occlusion (RVO). However, in real-world settings, patients often suffer from loss of vision on time scales of years despite therapy, whereas the prediction of the visual acuity (VA) and the earliest possible detection of deterioration under real-life conditions is challenging due to heterogeneous and incomplete data. In this contribution, we present a workflow for the development of a research-compatible data corpus fusing different IT systems of the department of ophthalmology of a German maximum care hospital. The extensive data corpus allows predictive statements of the expected progression of a patient and his or her VA in each of the three diseases. We found out for the disease AMD a significant deterioration of the visual acuity over time. Within our proposed m
    

