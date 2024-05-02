# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unbiased Learning to Rank Meets Reality: Lessons from Baidu's Large-Scale Search Dataset](https://arxiv.org/abs/2404.02543) | 本研究从百度搜索引擎发布的大规模搜索数据集出发，探讨了无偏向学习排序技术在实际搜索引擎中的表现，发现与排名损失和查询-文档特征选择相比，ULTR技术并未带来明显的性能改进。 |
| [^2] | [The Power of Noise: Redefining Retrieval for RAG Systems.](http://arxiv.org/abs/2401.14887) | 本研究通过分析和评估检索增强生成（RAG）系统中的信息检索（IR）组件，填补了目前研究中忽视的领域，在有效的RAG的提示表述中，不相关文档的包含可能会对系统性能产生负面影响。 |
| [^3] | [Next Visit Diagnosis Prediction via Medical Code-Centric Multimodal Contrastive EHR Modelling with Hierarchical Regularisation.](http://arxiv.org/abs/2401.11648) | 通过医学代码中心的多模态对比EHR建模预测下次就诊诊断，并通过分层正则化提高性能。 |

# 详细

[^1]: 无偏向学习排序遇到现实：百度大规模搜索数据集的经验教训

    Unbiased Learning to Rank Meets Reality: Lessons from Baidu's Large-Scale Search Dataset

    [https://arxiv.org/abs/2404.02543](https://arxiv.org/abs/2404.02543)

    本研究从百度搜索引擎发布的大规模搜索数据集出发，探讨了无偏向学习排序技术在实际搜索引擎中的表现，发现与排名损失和查询-文档特征选择相比，ULTR技术并未带来明显的性能改进。

    

    无偏向学习排序（ULTR）是一个用于学习用户点击数据的成熟框架，而这些数据往往受收集数据的排名者的偏见影响。虽然在理论上得到证明并在模拟中进行了广泛测试，但ULTR技术缺乏经验验证，尤其是在现代搜索引擎中。百度搜索引擎发布的WSDM Cup 2023数据集为评估主要ULTR技术在真实世界中的表现提供了难得的机会。尽管在WSDM Cup 2023期间有多次提交，以及随后的NTCIR ULTRE-2任务，但目前还不清楚观察到的改进是否源自应用ULTR或其他学习技术。我们重新审视并扩展了现有实验。我们发现，无偏向学习排序技术并不能明显提升性能，尤其是与排名损失和查询-文档特征选择带来的明显差异相比。

    arXiv:2404.02543v1 Announce Type: cross  Abstract: Unbiased learning-to-rank (ULTR) is a well-established framework for learning from user clicks, which are often biased by the ranker collecting the data. While theoretically justified and extensively tested in simulation, ULTR techniques lack empirical validation, especially on modern search engines. The dataset released for the WSDM Cup 2023, collected from Baidu's search engine, offers a rare opportunity to assess the real-world performance of prominent ULTR techniques. Despite multiple submissions during the WSDM Cup 2023 and the subsequent NTCIR ULTRE-2 task, it remains unclear whether the observed improvements stem from applying ULTR or other learning techniques. We revisit and extend the available experiments. We find that unbiased learning-to-rank techniques do not bring clear performance improvements, especially compared to the stark differences brought by the choice of ranking loss and query-document features. Our experiments 
    
[^2]: 噪声的力量：重新定义RAG系统的检索

    The Power of Noise: Redefining Retrieval for RAG Systems. (arXiv:2401.14887v1 [cs.IR])

    [http://arxiv.org/abs/2401.14887](http://arxiv.org/abs/2401.14887)

    本研究通过分析和评估检索增强生成（RAG）系统中的信息检索（IR）组件，填补了目前研究中忽视的领域，在有效的RAG的提示表述中，不相关文档的包含可能会对系统性能产生负面影响。

    

    检索增强生成（RAG）系统相对于传统的大型语言模型（LLMs）代表了一个重大进步。RAG系统通过整合通过信息检索（IR）阶段检索的外部数据来增强其生成能力，克服了标准LLMs的限制，后者仅限于其预先训练的知识和有限的上下文窗口。这个领域的大部分研究主要集中在RAG系统内LLMs的生成方面。我们的研究填补了这一空白，通过全面而批判性地分析IR组件对RAG系统的影响。本文分析了一个检索器在有效的RAG的提示表述中应该具备的特征，重点关注应该检索哪种类型的文档。我们评估了各种因素，如文档与提示的相关性，它们的位置以及上下文中包含的数量。我们的发现揭示出，包含不相关的文档可能会…

    Retrieval-Augmented Generation (RAG) systems represent a significant advancement over traditional Large Language Models (LLMs). RAG systems enhance their generation ability by incorporating external data retrieved through an Information Retrieval (IR) phase, overcoming the limitations of standard LLMs, which are restricted to their pre-trained knowledge and limited context window. Most research in this area has predominantly concentrated on the generative aspect of LLMs within RAG systems. Our study fills this gap by thoroughly and critically analyzing the influence of IR components on RAG systems. This paper analyzes which characteristics a retriever should possess for an effective RAG's prompt formulation, focusing on the type of documents that should be retrieved. We evaluate various elements, such as the relevance of the documents to the prompt, their position, and the number included in the context. Our findings reveal, among other insights, that including irrelevant documents can
    
[^3]: 通过具有分层正则化的医学代码中心的多模态对比EHR建模预测下次就诊诊断

    Next Visit Diagnosis Prediction via Medical Code-Centric Multimodal Contrastive EHR Modelling with Hierarchical Regularisation. (arXiv:2401.11648v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2401.11648](http://arxiv.org/abs/2401.11648)

    通过医学代码中心的多模态对比EHR建模预测下次就诊诊断，并通过分层正则化提高性能。

    

    在医疗保健中，利用电子健康记录（EHR）预测下次就诊的诊断是一项必要的任务，对于制定医疗保健提供者和患者的主动未来计划至关重要。然而，之前的许多研究并没有充分解决EHR数据固有的异构和分层特征，必然导致次优的性能。为此，我们提出了NECHO，一种新颖的医学代码中心的多模态对比EHR学习框架，其中包括分层正则化。首先，我们使用定制的网络设计和一对双模态对比损失融合涵盖医学代码、人口统计数据和临床笔记的多方面信息，所有这些都围绕着医学代码表现。我们还使用医学本体中的父级信息来规范特定模态的编码器，以学习EHR数据的层次结构。对MIMIC-III数据进行的一系列实验证明了我们方法的有效性。

    Predicting next visit diagnosis using Electronic Health Records (EHR) is an essential task in healthcare, critical for devising proactive future plans for both healthcare providers and patients. Nonetheless, many preceding studies have not sufficiently addressed the heterogeneous and hierarchical characteristics inherent in EHR data, inevitably leading to sub-optimal performance. To this end, we propose NECHO, a novel medical code-centric multimodal contrastive EHR learning framework with hierarchical regularisation. First, we integrate multifaceted information encompassing medical codes, demographics, and clinical notes using a tailored network design and a pair of bimodal contrastive losses, all of which pivot around a medical code representation. We also regularise modality-specific encoders using a parental level information in medical ontology to learn hierarchical structure of EHR data. A series of experiments on MIMIC-III data demonstrates effectiveness of our approach.
    

