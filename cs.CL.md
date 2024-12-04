# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Text clustering with LLM embeddings](https://arxiv.org/abs/2403.15112) | 研究表明，LLM嵌入能够捕捉结构化语言的细微差别，BERT在性能上领先于轻量级选项，增加嵌入维度和摘要技术并不一致地提高聚类效率 |
| [^2] | [AutoGuide: Automated Generation and Selection of State-Aware Guidelines for Large Language Model Agents](https://arxiv.org/abs/2403.08978) | AutoGuide通过提取嵌入在离线数据中的知识，生成一组状态感知指南，从而弥合大型语言模型中的知识差距，为代理的决策过程提供有用的知识。 |
| [^3] | [Large Language Models for Data Annotation: A Survey](https://arxiv.org/abs/2402.13446) | 大型语言模型的出现为自动化数据标注提供机遇，该调查独特关注LLM在数据标注中的效用，贡献主要集中在LLM-Based数据标注、评估LLM生成的标注以及使用LLM生成的标注学习等三个核心方面。 |
| [^4] | [ARKS: Active Retrieval in Knowledge Soup for Code Generation](https://arxiv.org/abs/2402.12317) | ARKS是一种用于代码生成的先进策略，通过活跃检索和整合各种信息源，能够提高大型语言模型的性能，为解决与频繁更新的库和长尾编程语言相关的现实编程问题提供了新的可能性。 |
| [^5] | [CultureLLM: Incorporating Cultural Differences into Large Language Models](https://arxiv.org/abs/2402.10946) | 提出了一种名为CultureLLM的成本效益高的解决方案，通过使用世界价值调查（WVS）作为种子数据，并通过提出的语义数据增强来将文化差异纳入大型语言模型中，成功微调得到了涵盖富裕和低资源语言的9种文化特定LLMs以及一个统一模型（CultureLLM-One）。 |

# 详细

[^1]: 使用LLM嵌入进行文本聚类

    Text clustering with LLM embeddings

    [https://arxiv.org/abs/2403.15112](https://arxiv.org/abs/2403.15112)

    研究表明，LLM嵌入能够捕捉结构化语言的细微差别，BERT在性能上领先于轻量级选项，增加嵌入维度和摘要技术并不一致地提高聚类效率

    

    文本聚类是组织不断增长的数字内容的重要方法，有助于结构化和发现未分类数据中的隐藏模式。在这项研究中，我们调查了不同文本嵌入（特别是大型语言模型LLMs中使用的）和聚类算法如何影响文本数据集的聚类方式。进行了一系列实验以评估嵌入是如何影响聚类结果的，以及通过摘要进行降维和嵌入大小调整的作用。结果显示，LLM嵌入在捕获结构化语言的细微差别方面表现出色，而BERT在性能上领先于轻量级选项。此外，我们发现增加嵌入维度和摘要技术并不一致地提高聚类效率，这表明这些策略需要仔细分析才能在实际模型中使用。这些结果突出了一种

    arXiv:2403.15112v1 Announce Type: cross  Abstract: Text clustering is an important approach for organising the growing amount of digital content, helping to structure and find hidden patterns in uncategorised data. In this research, we investigated how different textual embeddings - particularly those used in large language models (LLMs) - and clustering algorithms affect how text datasets are clustered. A series of experiments were conducted to assess how embeddings influence clustering results, the role played by dimensionality reduction through summarisation, and embedding size adjustment. Results reveal that LLM embeddings excel at capturing the nuances of structured language, while BERT leads the lightweight options in performance. In addition, we find that increasing embedding dimensionality and summarisation techniques do not uniformly improve clustering efficiency, suggesting that these strategies require careful analysis to use in real-life models. These results highlight a co
    
[^2]: AutoGuide: 大型语言模型代理的自动生成和选择状态感知指南

    AutoGuide: Automated Generation and Selection of State-Aware Guidelines for Large Language Model Agents

    [https://arxiv.org/abs/2403.08978](https://arxiv.org/abs/2403.08978)

    AutoGuide通过提取嵌入在离线数据中的知识，生成一组状态感知指南，从而弥合大型语言模型中的知识差距，为代理的决策过程提供有用的知识。

    

    大型语言模型（LLMs）的主要局限性是它们对世界的理解受限。这给基于LLMs的代理带来了重大困难，特别是在预训练的LLMs缺乏足够知识的领域。在本文中，我们介绍了一个名为AutoGuide的新框架，通过利用离线经验中的隐含知识来弥合预训练LLMs中的知识差距。具体而言，AutoGuide通过提取一组状态感知指南有效地提取嵌入在离线数据中的知识。每个状态感知指南以简洁的自然语言表达，并遵循条件结构，清晰描述适用的状态。因此，由此产生的指南为向代理当前的决策过程提供有用的知识提供了一种原则性的方法。我们展示了我们的方法在顺序任务中大幅领先于竞争的基于LLMs的基线。

    arXiv:2403.08978v1 Announce Type: new  Abstract: The primary limitation of large language models (LLMs) is their restricted understanding of the world. This poses significant difficulties for LLM-based agents, particularly in domains where pre-trained LLMs lack sufficient knowledge. In this paper, we introduce a novel framework, called AutoGuide, that bridges the knowledge gap in pre-trained LLMs by leveraging implicit knowledge in offline experiences. Specifically, AutoGuide effectively extracts knowledge embedded in offline data by extracting a set of state-aware guidelines. Importantly, each state-aware guideline is expressed in concise natural language and follows a conditional structure, clearly describing the state where it is applicable. As such, the resulting guidelines enable a principled way to provide helpful knowledge pertinent to an agent's current decision-making process. We show that our approach outperforms competitive LLM-based baselines by a large margin in sequential
    
[^3]: 大型语言模型用于数据标注：一项调查

    Large Language Models for Data Annotation: A Survey

    [https://arxiv.org/abs/2402.13446](https://arxiv.org/abs/2402.13446)

    大型语言模型的出现为自动化数据标注提供机遇，该调查独特关注LLM在数据标注中的效用，贡献主要集中在LLM-Based数据标注、评估LLM生成的标注以及使用LLM生成的标注学习等三个核心方面。

    

    数据标注是将原始数据标记或打标签与相关信息，对于提高机器学习模型的有效性至关重要。然而，这一过程劳动密集且昂贵。先进的大型语言模型（LLMs）的出现，例如GPT-4，为革新和自动化数据标注的复杂过程提供了前所未有的机遇。虽然现有的调查已经广泛涵盖了LLM的架构、训练和一般应用，但本文独特地关注它们在数据标注中的具体效用。该调查对LLM-Based数据标注、评估LLM生成的标注以及使用LLM生成的标注学习这三个核心方面做出了贡献。此外，论文包括了一种使用LLMs进行数据标注的方法学深度分类法，一个对整合LLM生成的标注的模型的学习策略进行全面审查，以及对其进行详细讨论。

    arXiv:2402.13446v1 Announce Type: new  Abstract: Data annotation is the labeling or tagging of raw data with relevant information, essential for improving the efficacy of machine learning models. The process, however, is labor-intensive and expensive. The emergence of advanced Large Language Models (LLMs), exemplified by GPT-4, presents an unprecedented opportunity to revolutionize and automate the intricate process of data annotation. While existing surveys have extensively covered LLM architecture, training, and general applications, this paper uniquely focuses on their specific utility for data annotation. This survey contributes to three core aspects: LLM-Based Data Annotation, Assessing LLM-generated Annotations, and Learning with LLM-generated annotations. Furthermore, the paper includes an in-depth taxonomy of methodologies employing LLMs for data annotation, a comprehensive review of learning strategies for models incorporating LLM-generated annotations, and a detailed discussi
    
[^4]: ARKS：用于代码生成中的知识汤中的活跃检索

    ARKS: Active Retrieval in Knowledge Soup for Code Generation

    [https://arxiv.org/abs/2402.12317](https://arxiv.org/abs/2402.12317)

    ARKS是一种用于代码生成的先进策略，通过活跃检索和整合各种信息源，能够提高大型语言模型的性能，为解决与频繁更新的库和长尾编程语言相关的现实编程问题提供了新的可能性。

    

    最近，检索增强生成（RAG）范式引起了人们的极大关注，因为它有潜力将外部知识整合到大型语言模型（LLMs）中，而无需进一步训练。尽管在自然语言应用中得到广泛探讨，但它在代码生成中的利用仍未得到充分探索。在本文中，我们介绍了一种名为知识汤中的活跃检索(ARKS)的先进策略，用于泛化大型语言模型以生成代码。与依靠单一来源不同，我们构建了一个将网页搜索、文档、执行反馈和进化代码片段整合在一起的知识汤。我们采用了一种积极的检索策略，该策略迭代地优化查询并更新知识汤。为了评估ARKS的性能，我们编制了一个新的基准，其中包括与频繁更新的库和长尾编程语言相关的现实编程问题。在ChatGPT和CodeLlama上的实验结果表明，ARKS比使用传统方法能够更好地产生代码。

    arXiv:2402.12317v1 Announce Type: cross  Abstract: Recently the retrieval-augmented generation (RAG) paradigm has raised much attention for its potential in incorporating external knowledge into large language models (LLMs) without further training. While widely explored in natural language applications, its utilization in code generation remains under-explored. In this paper, we introduce Active Retrieval in Knowledge Soup (ARKS), an advanced strategy for generalizing large language models for code. In contrast to relying on a single source, we construct a knowledge soup integrating web search, documentation, execution feedback, and evolved code snippets. We employ an active retrieval strategy that iteratively refines the query and updates the knowledge soup. To assess the performance of ARKS, we compile a new benchmark comprising realistic coding problems associated with frequently updated libraries and long-tail programming languages. Experimental results on ChatGPT and CodeLlama de
    
[^5]: 将文化差异纳入大型语言模型的研究

    CultureLLM: Incorporating Cultural Differences into Large Language Models

    [https://arxiv.org/abs/2402.10946](https://arxiv.org/abs/2402.10946)

    提出了一种名为CultureLLM的成本效益高的解决方案，通过使用世界价值调查（WVS）作为种子数据，并通过提出的语义数据增强来将文化差异纳入大型语言模型中，成功微调得到了涵盖富裕和低资源语言的9种文化特定LLMs以及一个统一模型（CultureLLM-One）。

    

    大型语言模型（LLMs）被报道偏向于某些文化，因为训练数据主要来自英语语料库。由于多语种文化数据通常较难收集，现有的工作通过提示工程或特定文化的预训练来处理这一问题。然而，它们可能忽视了低资源文化的知识缺乏，并需要大量的计算资源。本文提出了CultureLLM，这是一个成本效益高的解决方案，可将文化差异纳入LLMs中。CultureLLM采用世界价值调查（WVS）作为种子数据，并通过提出的语义数据增强生成语义等效的训练数据。仅使用来自WVS的50个种子样本和增强数据，我们对9种包括富裕和低资源语言的文化特定LLMs和一个统一模型（CultureLLM-One）进行了微调。对60个与文化相关的数据集进行的大量实验表明，CultureLLM在增强LLM的文化特性方面取得了显著的成果。

    arXiv:2402.10946v1 Announce Type: cross  Abstract: Large language models (LLMs) are reported to be partial to certain cultures owing to the training data dominance from the English corpora. Since multilingual cultural data are often expensive to collect, existing efforts handle this by prompt engineering or culture-specific pre-training. However, they might overlook the knowledge deficiency of low-resource culture and require extensive computing resources. In this paper, we propose CultureLLM, a cost-effective solution to incorporate cultural differences into LLMs. CultureLLM adopts World Value Survey (WVS) as seed data and generates semantically equivalent training data via the proposed semantic data augmentation. Using only 50 seed samples from WVS with augmented data, we fine-tune culture-specific LLMs and one unified model (CultureLLM-One) for 9 cultures covering rich and low-resource languages. Extensive experiments on 60 culture-related datasets demonstrate that CultureLLM signif
    

