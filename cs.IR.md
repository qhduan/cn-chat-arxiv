# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning.](http://arxiv.org/abs/2309.08420) | 提出了一种名为FedDCSR的联邦跨领域顺序推荐框架，通过解缠表示学习来处理不同领域之间的序列特征异质性，并保护数据隐私。 |
| [^2] | [Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models.](http://arxiv.org/abs/2306.10933) | 本文介绍了KAR框架，它从大型语言模型中获取两种类型的外部知识，分别是用户偏好的推理知识和项目的事实知识。通过混合专家适配器将推理和事实知识转换为增强向量，以便与现有的协同过滤推荐算法兼容。 |
| [^3] | [Knowledge Extraction in Low-Resource Scenarios: Survey and Perspective.](http://arxiv.org/abs/2202.08063) | 低资源情境下，如何让知识抽取更好地从非结构化文本中提取信息？本文调研了三种解决范式：高资源数据、更强的模型和数据与模型的结合，提出了未来的研究方向。 |

# 详细

[^1]: FedDCSR: 通过解缠表示学习实现联邦跨领域顺序推荐

    FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning. (arXiv:2309.08420v1 [cs.LG])

    [http://arxiv.org/abs/2309.08420](http://arxiv.org/abs/2309.08420)

    提出了一种名为FedDCSR的联邦跨领域顺序推荐框架，通过解缠表示学习来处理不同领域之间的序列特征异质性，并保护数据隐私。

    

    近年来，利用来自多个领域的用户序列数据的跨领域顺序推荐(CSR)受到了广泛关注。然而，现有的CSR方法需要在领域之间共享原始用户数据，这违反了《通用数据保护条例》(GDPR)。因此，有必要将联邦学习(FL)和CSR相结合，充分利用不同领域的知识，同时保护数据隐私。然而，不同领域之间的序列特征异质性对FL的整体性能有显著影响。在本文中，我们提出了FedDCSR，这是一种通过解缠表示学习的新型联邦跨领域顺序推荐框架。具体而言，为了解决不同领域之间的序列特征异质性，我们引入了一种称为领域内-领域间序列表示解缠(SRD)的方法，将用户序列特征解缠成领域共享和领域专属特征。

    Cross-domain Sequential Recommendation (CSR) which leverages user sequence data from multiple domains has received extensive attention in recent years. However, the existing CSR methods require sharing origin user data across domains, which violates the General Data Protection Regulation (GDPR). Thus, it is necessary to combine federated learning (FL) and CSR to fully utilize knowledge from different domains while preserving data privacy. Nonetheless, the sequence feature heterogeneity across different domains significantly impacts the overall performance of FL. In this paper, we propose FedDCSR, a novel federated cross-domain sequential recommendation framework via disentangled representation learning. Specifically, to address the sequence feature heterogeneity across domains, we introduce an approach called inter-intra domain sequence representation disentanglement (SRD) to disentangle the user sequence features into domain-shared and domain-exclusive features. In addition, we design
    
[^2]: 基于大型语言模型的开放世界推荐系统

    Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models. (arXiv:2306.10933v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.10933](http://arxiv.org/abs/2306.10933)

    本文介绍了KAR框架，它从大型语言模型中获取两种类型的外部知识，分别是用户偏好的推理知识和项目的事实知识。通过混合专家适配器将推理和事实知识转换为增强向量，以便与现有的协同过滤推荐算法兼容。

    

    推荐系统在各种在线服务中都扮演着至关重要的角色。但是，它们在特定领域内进行训练和部署的封闭性限制了它们访问开放世界知识的能力。最近，大型语言模型(LLM)的出现在编码广泛的世界知识和展示推理能力方面显示出了希望。尽管如此，直接使用LLM作为推荐人之前的尝试并没有取得令人满意的结果。在本文中，我们提出了一种基于大型语言模型的开放世界知识增强推荐框架(KAR)，以从LLM获取两种类型的外部知识--用户偏好的推理知识和项目的事实知识。我们介绍了因子分解提示来引导对用户喜好的准确推理。生成的推理和事实知识通过混合专家适配器有效地转换并压缩为增强向量，以便与现有的协同过滤推荐算法兼容。

    Recommender systems play a vital role in various online services. However, the insulated nature of training and deploying separately within a specific domain limits their access to open-world knowledge. Recently, the emergence of large language models (LLMs) has shown promise in bridging this gap by encoding extensive world knowledge and demonstrating reasoning capability. Nevertheless, previous attempts to directly use LLMs as recommenders have not achieved satisfactory results. In this work, we propose an Open-World Knowledge Augmented Recommendation Framework with Large Language Models, dubbed KAR, to acquire two types of external knowledge from LLMs -- the reasoning knowledge on user preferences and the factual knowledge on items. We introduce factorization prompting to elicit accurate reasoning on user preferences. The generated reasoning and factual knowledge are effectively transformed and condensed into augmented vectors by a hybrid-expert adaptor in order to be compatible with
    
[^3]: 低资源情境下的知识抽取：调研与展望

    Knowledge Extraction in Low-Resource Scenarios: Survey and Perspective. (arXiv:2202.08063v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2202.08063](http://arxiv.org/abs/2202.08063)

    低资源情境下，如何让知识抽取更好地从非结构化文本中提取信息？本文调研了三种解决范式：高资源数据、更强的模型和数据与模型的结合，提出了未来的研究方向。

    

    知识抽取（KE）旨在从非结构化文本中提取结构信息，通常遭受数据匮乏和出现未见类型（低资源情境）的困扰。许多神经网络方法已广泛研究并取得了令人瞩目的表现。本文对低资源情境下KE进行文献综述，并将现有的工作系统性地分为三种范式：（1）利用高资源数据，（2）利用更强的模型，（3）同时利用数据和模型。此外，本文提出有前途的应用，并概述了未来研究的一些潜在方向。我们希望我们的调研可以帮助学术和工业界更好地理解这一领域，激发更多的创意，提升更广泛的应用。

    Knowledge Extraction (KE), aiming to extract structural information from unstructured texts, often suffers from data scarcity and emerging unseen types, i.e., low-resource scenarios. Many neural approaches to low-resource KE have been widely investigated and achieved impressive performance. In this paper, we present a literature review towards KE in low-resource scenarios, and systematically categorize existing works into three paradigms: (1) exploiting higher-resource data, (2) exploiting stronger models, and (3) exploiting data and models together. In addition, we highlight promising applications and outline some potential directions for future research. We hope that our survey can help both the academic and industrial communities to better understand this field, inspire more ideas, and boost broader applications.
    

