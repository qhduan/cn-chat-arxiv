# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fundus: A Simple-to-Use News Scraper Optimized for High Quality Extractions](https://arxiv.org/abs/2403.15279) | Fundus是一个简单易用的新闻爬虫工具，通过手工定制的内容提取器，针对每个支持的在线报纸格式指南进行优化，实现高质量的新闻文章提取，同时结合爬取和内容提取于一体，为非技术用户提供统一使用界面。 |
| [^2] | [FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions](https://arxiv.org/abs/2403.15246) | 该论文引入了FollowIR数据集，包含严格的说明书评估基准和训练集，帮助信息检索模型更好地遵循真实世界的说明书。议论基于TREC会议的历史，旨在使信息检索模型能够根据详细说明书理解和判断相关性。 |
| [^3] | [Bilateral Unsymmetrical Graph Contrastive Learning for Recommendation](https://arxiv.org/abs/2403.15075) | 提出了一种名为双侧不对称图对比学习（BusGCL）的框架，通过考虑用户-项目节点关系密度的双侧不对称性，实现了更好的推理用户和项目图。 |
| [^4] | [Large language model-powered chatbots for internationalizing student support in higher education](https://arxiv.org/abs/2403.14702) | 该研究探索了将GPT-3.5和GPT-4 Turbo驱动的聊天机器人技术整合到高等教育中，以提升国际化，并利用数字化转型，结果显示这些聊天机器人在提供全面回复、用户偏好和低错误率方面非常有效，并展示了提升可访问性、效率和满意度的潜力。 |
| [^5] | [A2CI: A Cloud-based, Service-oriented Geospatial Cyberinfrastructure to Support Atmospheric Research](https://arxiv.org/abs/2403.14693) | 本论文介绍了一个基于云的、面向服务的地理空间下层结构A2CI，旨在支持大气研究，能有效应对收集和整理的大量地球科学数据所带来的挑战。 |
| [^6] | [SyllabusQA: A Course Logistics Question Answering Dataset](https://arxiv.org/abs/2403.14666) | SyllabusQA数据集是一个包含63个真实课程大纲的开源数据集，对36个专业涵盖5,078对多样化的开放式课程逻辑相关问题-答案对进行了详细收集，旨在评估答案事实性，多个强基线模型在该任务上表现出色，但仍存在与人类之间的显著差距。 |
| [^7] | [Evaluating Large Language Models as Generative User Simulators for Conversational Recommendation](https://arxiv.org/abs/2403.09738) | 大型语言模型作为生成式用户模拟器在对话推荐中展现出潜力，新的协议通过五个任务评估了语言模型模拟人类行为的准确程度，揭示了模型与人类行为的偏差，并提出了如何通过模型选择和提示策略减少这些偏差。 |
| [^8] | [On Image Search in Histopathology.](http://arxiv.org/abs/2401.08699) | 这篇论文综述了组织病理学图像搜索技术的最新发展，为计算病理学研究人员提供了简明的概述，旨在寻求有效、快速和高效的图像搜索方法。 |
| [^9] | [Reformulating Sequential Recommendation: Learning Dynamic User Interest with Content-enriched Language Modeling.](http://arxiv.org/abs/2309.10435) | 本研究提出了一个新的顺序推荐范式 LANCER，利用预训练语言模型的语义理解能力生成更加人性化的个性化推荐。在多个基准数据集上的实验结果表明，该方法有效且有希望，并为了解顺序推荐的影响提供了有价值的见解。 |

# 详细

[^1]: Fundus：一个简单易用的新闻爬虫，优化高质量提取

    Fundus: A Simple-to-Use News Scraper Optimized for High Quality Extractions

    [https://arxiv.org/abs/2403.15279](https://arxiv.org/abs/2403.15279)

    Fundus是一个简单易用的新闻爬虫工具，通过手工定制的内容提取器，针对每个支持的在线报纸格式指南进行优化，实现高质量的新闻文章提取，同时结合爬取和内容提取于一体，为非技术用户提供统一使用界面。

    

    本文介绍了Fundus，一个用户友好的新闻爬虫，使用户可以仅凭几行代码获得数百万高质量的新闻文章。与现有的新闻爬虫不同，我们使用手工定制的、专门针对每个支持的在线报纸的格式指南的内容提取器。这样我们可以优化我们的爬取质量，以确保检索到的新闻文章完整且没有HTML痕迹。此外，我们的框架将爬取（从网络或大型网络归档中检索HTML）和内容提取结合到一个单一的流水线中。通过为预定义的一组报纸提供统一的界面，我们的目标是使Fundus即使对非技术用户也易于使用。本文概述了框架，讨论了我们的设计选择，并针对其他流行的新闻爬虫进行了比较评估。我们的评估表明，Fundus取得了...

    arXiv:2403.15279v1 Announce Type: new  Abstract: This paper introduces Fundus, a user-friendly news scraper that enables users to obtain millions of high-quality news articles with just a few lines of code. Unlike existing news scrapers, we use manually crafted, bespoke content extractors that are specifically tailored to the formatting guidelines of each supported online newspaper. This allows us to optimize our scraping for quality such that retrieved news articles are textually complete and without HTML artifacts. Further, our framework combines both crawling (retrieving HTML from the web or large web archives) and content extraction into a single pipeline. By providing a unified interface for a predefined collection of newspapers, we aim to make Fundus broadly usable even for non-technical users. This paper gives an overview of the framework, discusses our design choices, and presents a comparative evaluation against other popular news scrapers. Our evaluation shows that Fundus yie
    
[^2]: FollowIR: 评估和教授信息检索模型以遵循说明书

    FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions

    [https://arxiv.org/abs/2403.15246](https://arxiv.org/abs/2403.15246)

    该论文引入了FollowIR数据集，包含严格的说明书评估基准和训练集，帮助信息检索模型更好地遵循真实世界的说明书。议论基于TREC会议的历史，旨在使信息检索模型能够根据详细说明书理解和判断相关性。

    

    现代大型语言模型（LLMs）能够遵循长且复杂的说明书，从而实现多样化的用户任务。然而，尽管信息检索（IR）模型使用LLMs作为其架构的支柱，几乎所有这些模型仍然只接受查询作为输入，没有说明书。对于最近一些接受说明书的模型来说，它们如何使用这些说明书还不清楚。我们引入了FollowIR数据集，其中包含严格的说明书评估基准，以及一个训练集，帮助IR模型学习更好地遵循现实世界的说明书。FollowIR基于TREC会议的悠久历史：正如TREC为人类标注员提供说明书（也称为叙述）来判断文档的相关性一样，因此IR模型应该能够根据这些详细说明书理解和确定相关性。我们的评估基准从三个经过深度判断的TREC收藏开始

    arXiv:2403.15246v1 Announce Type: cross  Abstract: Modern Large Language Models (LLMs) are capable of following long and complex instructions that enable a diverse amount of user tasks. However, despite Information Retrieval (IR) models using LLMs as the backbone of their architectures, nearly all of them still only take queries as input, with no instructions. For the handful of recent models that do take instructions, it's unclear how they use them. We introduce our dataset FollowIR, which contains a rigorous instruction evaluation benchmark as well as a training set for helping IR models learn to better follow real-world instructions. FollowIR builds off the long history of the TREC conferences: as TREC provides human annotators with instructions (also known as narratives) to determine document relevance, so should IR models be able to understand and decide relevance based on these detailed instructions. Our evaluation benchmark starts with three deeply judged TREC collections and al
    
[^3]: 双侧不对称图对比学习用于推荐

    Bilateral Unsymmetrical Graph Contrastive Learning for Recommendation

    [https://arxiv.org/abs/2403.15075](https://arxiv.org/abs/2403.15075)

    提出了一种名为双侧不对称图对比学习（BusGCL）的框架，通过考虑用户-项目节点关系密度的双侧不对称性，实现了更好的推理用户和项目图。

    

    最近的方法利用图对比学习在图结构的用户-项目交互数据中进行协同过滤，展示了其在推荐任务中的有效性。然而，它们忽略了用户-项目节点之间的差异关系密度导致多跳图交互计算后双向节点的图适应性不同，这限制了现有模型实现理想结果的能力。为了解决这一问题，我们提出了一个新的推荐任务框架，称为双侧不对称图对比学习（BusGCL），考虑了用户-项目节点关系密度的双侧不对称性，通过双侧切片对比训练更好地推理用户和项目图。特别地，考虑基于超图的图卷积网络（GCN）在挖掘隐式相似性方面的聚合能力更适合用户节点

    arXiv:2403.15075v1 Announce Type: cross  Abstract: Recent methods utilize graph contrastive Learning within graph-structured user-item interaction data for collaborative filtering and have demonstrated their efficacy in recommendation tasks. However, they ignore that the difference relation density of nodes between the user- and item-side causes the adaptability of graphs on bilateral nodes to be different after multi-hop graph interaction calculation, which limits existing models to achieve ideal results. To solve this issue, we propose a novel framework for recommendation tasks called Bilateral Unsymmetrical Graph Contrastive Learning (BusGCL) that consider the bilateral unsymmetry on user-item node relation density for sliced user and item graph reasoning better with bilateral slicing contrastive training. Especially, taking into account the aggregation ability of hypergraph-based graph convolutional network (GCN) in digging implicit similarities is more suitable for user nodes, emb
    
[^4]: 大型语言模型驱动的聊天机器人在高等教育中的国际化学生支持

    Large language model-powered chatbots for internationalizing student support in higher education

    [https://arxiv.org/abs/2403.14702](https://arxiv.org/abs/2403.14702)

    该研究探索了将GPT-3.5和GPT-4 Turbo驱动的聊天机器人技术整合到高等教育中，以提升国际化，并利用数字化转型，结果显示这些聊天机器人在提供全面回复、用户偏好和低错误率方面非常有效，并展示了提升可访问性、效率和满意度的潜力。

    

    这项研究探讨了将由GPT-3.5和GPT-4 Turbo提供动力的聊天机器人技术整合到高等教育中，以增强国际化并利用数字化转型。研究深入探讨了大型语言模型（LLMs）的设计、实施和应用，以改善学生参与、信息获取和支持。利用Python 3、GPT API、LangChain和Chroma Vector Store等技术，研究强调为聊天机器人测试创造高质量、及时和相关的转录数据集。研究结果表明聊天机器人在提供全面回复、用户对传统方法的偏好以及低错误率方面的有效性。强调聊天机器人的实时参与、记忆能力和关键数据访问，研究展示了其提升可访问性、效率和满意度的潜力。最后，研究提出聊天机器人显著有助于高等教育。

    arXiv:2403.14702v1 Announce Type: cross  Abstract: This research explores the integration of chatbot technology powered by GPT-3.5 and GPT-4 Turbo into higher education to enhance internationalization and leverage digital transformation. It delves into the design, implementation, and application of Large Language Models (LLMs) for improving student engagement, information access, and support. Utilizing technologies like Python 3, GPT API, LangChain, and Chroma Vector Store, the research emphasizes creating a high-quality, timely, and relevant transcript dataset for chatbot testing. Findings indicate the chatbot's efficacy in providing comprehensive responses, its preference over traditional methods by users, and a low error rate. Highlighting the chatbot's real-time engagement, memory capabilities, and critical data access, the study demonstrates its potential to elevate accessibility, efficiency, and satisfaction. Concluding, the research suggests the chatbot significantly aids higher
    
[^5]: A2CI：基于云的面向服务的地理空间下层结构，支持大气研究

    A2CI: A Cloud-based, Service-oriented Geospatial Cyberinfrastructure to Support Atmospheric Research

    [https://arxiv.org/abs/2403.14693](https://arxiv.org/abs/2403.14693)

    本论文介绍了一个基于云的、面向服务的地理空间下层结构A2CI，旨在支持大气研究，能有效应对收集和整理的大量地球科学数据所带来的挑战。

    

    大地科学数据为科学界提供了巨大的机遇。利用遥感卫星、地面传感器网络甚至社交媒体输入收集到的丰富信息，现在可以进行更多大规模、长期和高分辨率的研究。然而，NASA和其他政府机构每小时收集和整理的数百TB信息对于希望改善对地球大气系统的理解的大气科学家来说构成了重大挑战。这些挑战包括大量数据的有效发现、组织、分析和可视化。本文报告了一个由NSF资助的项目的成果，该项目开发了一个地理空间下层结构——A2CI（大气分析下层结构），以支持大气研究。首先我们介绍了基于服务的系统框架，然后详细描述了...

    arXiv:2403.14693v1 Announce Type: cross  Abstract: Big earth science data offers the scientific community great opportunities. Many more studies at large-scales, over long-terms and at high resolution can now be conducted using the rich information collected by remote sensing satellites, ground-based sensor networks, and even social media input. However, the hundreds of terabytes of information collected and compiled on an hourly basis by NASA and other government agencies present a significant challenge for atmospheric scientists seeking to improve the understanding of the Earth atmospheric system. These challenges include effective discovery, organization, analysis and visualization of large amounts of data. This paper reports the outcomes of an NSF-funded project that developed a geospatial cyberinfrastructure -- the A2CI (Atmospheric Analysis Cyberinfrastructure) -- to support atmospheric research. We first introduce the service-oriented system framework then describe in detail the
    
[^6]: SyllabusQA：一个课程逻辑问题回答数据集

    SyllabusQA: A Course Logistics Question Answering Dataset

    [https://arxiv.org/abs/2403.14666](https://arxiv.org/abs/2403.14666)

    SyllabusQA数据集是一个包含63个真实课程大纲的开源数据集，对36个专业涵盖5,078对多样化的开放式课程逻辑相关问题-答案对进行了详细收集，旨在评估答案事实性，多个强基线模型在该任务上表现出色，但仍存在与人类之间的显著差距。

    

    自动化教学助理和聊天机器人有显著潜力减轻人类教师的工作量，尤其是对于与课程逻辑相关的问题回答，这对学生很重要，但对教师来说是重复的。然而，由于隐私问题，缺乏公开可用的数据集。我们介绍了SyllabusQA，这是一个开源数据集，包含63个真实课程大纲，涵盖36个专业，包含5,078对多样化的开放式课程逻辑相关问题-答案对，问题类型和答案格式都是多样的。由于许多逻辑相关问题包含关键信息，如考试日期，评估答案的事实性很重要。我们在该任务上对几个强基线进行了基准测试，从大型语言模型提示到检索增强生成。我们发现，尽管在传统的文本相似性指标上接近人类表现，但在准确性方面仍存在显著差距。

    arXiv:2403.14666v1 Announce Type: cross  Abstract: Automated teaching assistants and chatbots have significant potential to reduce the workload of human instructors, especially for logistics-related question answering, which is important to students yet repetitive for instructors. However, due to privacy concerns, there is a lack of publicly available datasets. We introduce SyllabusQA, an open-source dataset with 63 real course syllabi covering 36 majors, containing 5,078 open-ended course logistics-related question-answer pairs that are diverse in both question types and answer formats. Since many logistics-related questions contain critical information like the date of an exam, it is important to evaluate the factuality of answers. We benchmark several strong baselines on this task, from large language model prompting to retrieval-augmented generation. We find that despite performing close to humans on traditional metrics of textual similarity, there remains a significant gap between
    
[^7]: 评估大语言模型作为对话推荐中生成用户模拟器

    Evaluating Large Language Models as Generative User Simulators for Conversational Recommendation

    [https://arxiv.org/abs/2403.09738](https://arxiv.org/abs/2403.09738)

    大型语言模型作为生成式用户模拟器在对话推荐中展现出潜力，新的协议通过五个任务评估了语言模型模拟人类行为的准确程度，揭示了模型与人类行为的偏差，并提出了如何通过模型选择和提示策略减少这些偏差。

    

    合成用户是对话推荐系统评估中成本效益较高的真实用户代理。大型语言模型表现出在模拟类似人类行为方面的潜力，这引发了它们能否代表多样化用户群体的问题。我们引入了一个新的协议，用于衡量语言模型能够准确模拟对话推荐中人类行为的程度。该协议由五个任务组成，每个任务旨在评估合成用户应该表现出的关键特性：选择要谈论的物品，表达二进制偏好，表达开放式偏好，请求推荐以及提供反馈。通过对基准模拟器的评估，我们展示了这些任务有效地揭示了语言模型与人类行为的偏差，并提供了关于如何通过模型选择和提示策略减少这些偏差的见解。

    arXiv:2403.09738v1 Announce Type: cross  Abstract: Synthetic users are cost-effective proxies for real users in the evaluation of conversational recommender systems. Large language models show promise in simulating human-like behavior, raising the question of their ability to represent a diverse population of users. We introduce a new protocol to measure the degree to which language models can accurately emulate human behavior in conversational recommendation. This protocol is comprised of five tasks, each designed to evaluate a key property that a synthetic user should exhibit: choosing which items to talk about, expressing binary preferences, expressing open-ended preferences, requesting recommendations, and giving feedback. Through evaluation of baseline simulators, we demonstrate these tasks effectively reveal deviations of language models from human behavior, and offer insights on how to reduce the deviations with model selection and prompting strategies.
    
[^8]: 关于组织病理学图像搜索的研究

    On Image Search in Histopathology. (arXiv:2401.08699v1 [eess.IV])

    [http://arxiv.org/abs/2401.08699](http://arxiv.org/abs/2401.08699)

    这篇论文综述了组织病理学图像搜索技术的最新发展，为计算病理学研究人员提供了简明的概述，旨在寻求有效、快速和高效的图像搜索方法。

    

    组织病理学的病理图像可以通过装有摄像头的显微镜或全扫描仪获取。利用相似性计算基于这些图像匹配患者，在研究和临床环境中具有重要潜力。最近搜索技术的进展使得可以对各种组织类型的细胞结构进行微妙的量化，促进比较，并在与诊断和治疗过的病例数据库进行比较时实现关于诊断、预后和新患者预测的推断。本文全面回顾了组织病理学图像搜索技术的最新发展，为计算病理学研究人员提供了简明的概述，以寻求有效、快速和高效的图像搜索方法。

    Pathology images of histopathology can be acquired from camera-mounted microscopes or whole slide scanners. Utilizing similarity calculations to match patients based on these images holds significant potential in research and clinical contexts. Recent advancements in search technologies allow for nuanced quantification of cellular structures across diverse tissue types, facilitating comparisons and enabling inferences about diagnosis, prognosis, and predictions for new patients when compared against a curated database of diagnosed and treated cases. In this paper, we comprehensively review the latest developments in image search technologies for histopathology, offering a concise overview tailored for computational pathology researchers seeking effective, fast and efficient image search methods in their work.
    
[^9]: 重塑顺序推荐系统：利用内容增强语言建模学习动态用户兴趣

    Reformulating Sequential Recommendation: Learning Dynamic User Interest with Content-enriched Language Modeling. (arXiv:2309.10435v1 [cs.IR])

    [http://arxiv.org/abs/2309.10435](http://arxiv.org/abs/2309.10435)

    本研究提出了一个新的顺序推荐范式 LANCER，利用预训练语言模型的语义理解能力生成更加人性化的个性化推荐。在多个基准数据集上的实验结果表明，该方法有效且有希望，并为了解顺序推荐的影响提供了有价值的见解。

    

    推荐系统对在线应用至关重要，而顺序推荐由于其表达能力强大，能够捕捉到动态用户兴趣而广泛使用。然而，先前的顺序建模方法在捕捉上下文信息方面仍存在局限性。主要的原因是语言模型常常缺乏对领域特定知识和物品相关文本内容的理解。为了解决这个问题，我们采用了一种新的顺序推荐范式，并提出了LANCER，它利用预训练语言模型的语义理解能力生成个性化推荐。我们的方法弥合了语言模型与推荐系统之间的差距，产生了更加人性化的推荐。通过对多个基准数据集上的实验，我们验证了我们的方法的有效性，展示了有希望的结果，并提供了对我们模型对顺序推荐的影响的有价值的见解。

    Recommender systems are essential for online applications, and sequential recommendation has enjoyed significant prevalence due to its expressive ability to capture dynamic user interests. However, previous sequential modeling methods still have limitations in capturing contextual information. The primary reason for this issue is that language models often lack an understanding of domain-specific knowledge and item-related textual content. To address this issue, we adopt a new sequential recommendation paradigm and propose LANCER, which leverages the semantic understanding capabilities of pre-trained language models to generate personalized recommendations. Our approach bridges the gap between language models and recommender systems, resulting in more human-like recommendations. We demonstrate the effectiveness of our approach through experiments on several benchmark datasets, showing promising results and providing valuable insights into the influence of our model on sequential recomm
    

