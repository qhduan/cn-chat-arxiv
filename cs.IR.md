# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [EASRec: Elastic Architecture Search for Efficient Long-term Sequential Recommender Systems](https://arxiv.org/abs/2402.00390) | EASRec是一个针对顺序推荐系统的弹性架构搜索方法，通过自动剪枝技术和先进模型架构结合，以及资源受限神经架构搜索技术，实现了降低计算成本和资源消耗的同时保持或增强准确性。 |
| [^2] | [Agent-OM: Leveraging LLM Agents for Ontology Matching](https://arxiv.org/abs/2312.00326) | 本研究提出了Agent-OM，利用LLM代理为本体匹配系统引入了新的设计范式。 |

# 详细

[^1]: EASRec：用于高效长期顺序推荐系统的弹性架构搜索

    EASRec: Elastic Architecture Search for Efficient Long-term Sequential Recommender Systems

    [https://arxiv.org/abs/2402.00390](https://arxiv.org/abs/2402.00390)

    EASRec是一个针对顺序推荐系统的弹性架构搜索方法，通过自动剪枝技术和先进模型架构结合，以及资源受限神经架构搜索技术，实现了降低计算成本和资源消耗的同时保持或增强准确性。

    

    在数据丰富的时代，从海量信息中提取有意义的见解的能力至关重要。我们的研究解决了当前顺序推荐系统（SRSs）在计算和资源效率方面存在的问题，特别是那些采用了基于注意力模型（如SASRec）的系统。这些系统旨在为各种应用提供下一个项目的推荐，从电子商务到社交网络。然而，这些系统在推理阶段会产生相当大的计算成本和资源消耗。为了解决这些问题，我们的研究提出了一种结合自动剪枝技术和先进模型架构的新方法。我们还探索了在推荐系统领域中流行的资源受限神经架构搜索（NAS）技术的潜力，以调整模型以减少FLOPs、延迟和能量使用，同时保持或增强准确性。我们的工作的主要贡献是开发了一种

    In this age where data is abundant, the ability to distill meaningful insights from the sea of information is essential. Our research addresses the computational and resource inefficiencies that current Sequential Recommender Systems (SRSs) suffer from. especially those employing attention-based models like SASRec, These systems are designed for next-item recommendations in various applications, from e-commerce to social networks. However, such systems suffer from substantial computational costs and resource consumption during the inference stage. To tackle these issues, our research proposes a novel method that combines automatic pruning techniques with advanced model architectures. We also explore the potential of resource-constrained Neural Architecture Search (NAS), a technique prevalent in the realm of recommendation systems, to fine-tune models for reduced FLOPs, latency, and energy usage while retaining or even enhancing accuracy. The main contribution of our work is developing 
    
[^2]: Agent-OM：利用LLM代理进行本体匹配

    Agent-OM: Leveraging LLM Agents for Ontology Matching

    [https://arxiv.org/abs/2312.00326](https://arxiv.org/abs/2312.00326)

    本研究提出了Agent-OM，利用LLM代理为本体匹配系统引入了新的设计范式。

    

    本体匹配（OM）能够实现不同本体之间的语义互操作性，通过对齐相关实体来解决其概念异构性。本研究引入了一种新颖的基于代理的LLM设计范式，命名为Agent-OM，包括两个用于检索和匹配的同体代理以及一组基于提示的简单OM工具。

    arXiv:2312.00326v2 Announce Type: replace  Abstract: Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM, consisting of two Siamese agents for retrieval and matching, with a set of simple prompt-based OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAE
    

