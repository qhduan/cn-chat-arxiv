# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LOCALINTEL: Generating Organizational Threat Intelligence from Global and Local Cyber Knowledge.](http://arxiv.org/abs/2401.10036) | LOCALINTEL是一个自动化的知识上下文化系统，利用大型语言模型的能力，从全球和本地知识数据库中自动生成组织的威胁情报。 |
| [^2] | [Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation.](http://arxiv.org/abs/2310.09874) | 本文利用大型语言模型（LLMs）来增强基于内容的推荐中的免训练数据集压缩方法，旨在通过生成文本内容来合成一个小而信息丰富的数据集，使得模型能够达到与在大型数据集上训练的模型相当的性能。 |

# 详细

[^1]: LOCALINTEL：从全球和本地网络知识生成组织威胁情报

    LOCALINTEL: Generating Organizational Threat Intelligence from Global and Local Cyber Knowledge. (arXiv:2401.10036v1 [cs.CR])

    [http://arxiv.org/abs/2401.10036](http://arxiv.org/abs/2401.10036)

    LOCALINTEL是一个自动化的知识上下文化系统，利用大型语言模型的能力，从全球和本地知识数据库中自动生成组织的威胁情报。

    

    安全操作中心（SoC）分析师从公开访问的全球威胁数据库中收集威胁报告，并手动自定义以适应特定组织的需求。这些分析师还依赖于内部存储库，作为组织的私有本地知识数据库。可信的网络情报、关键操作细节和相关组织信息都存储在这些本地知识数据库中。分析师利用这些全球和本地知识数据库从事一项繁重的任务，手动创建组织独特的威胁响应和缓解策略。最近，大型语言模型（LLMs）已经展示了高效处理大规模多样化知识源的能力。我们利用这种能力来处理全球和本地知识数据库，自动化生成组织特定的威胁情报。在这项工作中，我们提出了LOCALINTEL，这是一个新颖的自动化知识上下文化系统，可以从全球和本地知识数据库中生成组织的威胁情报。

    Security Operations Center (SoC) analysts gather threat reports from openly accessible global threat databases and customize them manually to suit a particular organization's needs. These analysts also depend on internal repositories, which act as private local knowledge database for an organization. Credible cyber intelligence, critical operational details, and relevant organizational information are all stored in these local knowledge databases. Analysts undertake a labor intensive task utilizing these global and local knowledge databases to manually create organization's unique threat response and mitigation strategies. Recently, Large Language Models (LLMs) have shown the capability to efficiently process large diverse knowledge sources. We leverage this ability to process global and local knowledge databases to automate the generation of organization-specific threat intelligence.  In this work, we present LOCALINTEL, a novel automated knowledge contextualization system that, upon 
    
[^2]: 利用大型语言模型（LLMs）增强基于内容的推荐的免训练数据集压缩

    Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation. (arXiv:2310.09874v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.09874](http://arxiv.org/abs/2310.09874)

    本文利用大型语言模型（LLMs）来增强基于内容的推荐中的免训练数据集压缩方法，旨在通过生成文本内容来合成一个小而信息丰富的数据集，使得模型能够达到与在大型数据集上训练的模型相当的性能。

    

    现代内容推荐（CBR）技术利用物品的内容信息为用户提供个性化服务，但在大型数据集上的资源密集型训练存在问题。为解决这个问题，本文探讨了对文本CBR进行数据集压缩的方法。数据集压缩的目标是合成一个小且信息丰富的数据集，使模型性能可以与在大型数据集上训练的模型相媲美。现有的压缩方法针对连续数据（如图像或嵌入向量）的分类任务而设计，直接应用于CBR存在局限性。为了弥补这一差距，我们研究了基于内容的推荐中高效的数据集压缩方法。受到大型语言模型（LLMs）在文本理解和生成方面出色的能力的启发，我们利用LLMs在数据集压缩期间生成文本内容。为了处理涉及用户和物品的交互数据，我们设计了一个双...

    Modern techniques in Content-based Recommendation (CBR) leverage item content information to provide personalized services to users, but suffer from resource-intensive training on large datasets. To address this issue, we explore the dataset condensation for textual CBR in this paper. The goal of dataset condensation is to synthesize a small yet informative dataset, upon which models can achieve performance comparable to those trained on large datasets. While existing condensation approaches are tailored to classification tasks for continuous data like images or embeddings, direct application of them to CBR has limitations. To bridge this gap, we investigate efficient dataset condensation for content-based recommendation. Inspired by the remarkable abilities of large language models (LLMs) in text comprehension and generation, we leverage LLMs to empower the generation of textual content during condensation. To handle the interaction data involving both users and items, we devise a dua
    

