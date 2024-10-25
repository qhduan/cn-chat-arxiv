# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoRD: An Automatic and End-to-End System for Rare Disease Knowledge Graph Construction Based on Ontologies-enhanced Large Language Models](https://arxiv.org/abs/2403.00953) | AutoRD是一个自动化端到端系统，使用大型语言模型和医学知识图构建罕见疾病知识图，实现了整体F1得分47.3%，相对于基础LLM有14.4%的提升。 |
| [^2] | [Head-wise Shareable Attention for Large Language Models](https://arxiv.org/abs/2402.11819) | 本文提出了面向大语言模型的头部共享注意力的观点，提出了两种在注意力头之间共享参数的内存高效方法，以解决大型语言模型参数数量巨大导致部署受限的问题。 |
| [^3] | [Large Language Model for Table Processing: A Survey](https://arxiv.org/abs/2402.05121) | 该调查综述了大型语言模型在表格处理中的应用，包括传统的表格问题回答和事实验证，以及新兴的表格操作和高级表格数据分析。还讨论了LLMs的最新范例，特别关注了指导调整、提示和基于代理的方法。 |

# 详细

[^1]: AutoRD：一种基于本体增强的大型语言模型的罕见疾病知识图构建的自动化端到端系统

    AutoRD: An Automatic and End-to-End System for Rare Disease Knowledge Graph Construction Based on Ontologies-enhanced Large Language Models

    [https://arxiv.org/abs/2403.00953](https://arxiv.org/abs/2403.00953)

    AutoRD是一个自动化端到端系统，使用大型语言模型和医学知识图构建罕见疾病知识图，实现了整体F1得分47.3%，相对于基础LLM有14.4%的提升。

    

    目标：我们的目标是创建一个名为AutoRD的端到端系统，该系统自动从临床文本中提取有关罕见疾病的信息。我们进行了各种测试来评估AutoRD的性能，并在本文中强调了其优势和局限性。方法：我们的系统AutoRD是一个软件流水线，涉及数据预处理、实体提取、关系提取、实体校准和知识图构建。我们使用大型语言模型和由开源医学本体发展而来的医学知识图来实现这一目标。我们通过实体提取、关系提取以及知识图构建性能对系统进行定量评估。结果：AutoRD取得了47.3%的整体F1分数，较基础LLM提高了14.4%。具体来说，AutoRD实现了56.1%的整体实体提取F1分数（罕见疾病：83.5%，疾病：35.8%，s

    arXiv:2403.00953v1 Announce Type: cross  Abstract: Objectives: Our objective is to create an end-to-end system called AutoRD, which automates extracting information from clinical text about rare diseases. We have conducted various tests to evaluate the performance of AutoRD and highlighted its strengths and limitations in this paper.   Materials and Methods: Our system, AutoRD, is a software pipeline involving data preprocessing, entity extraction, relation extraction, entity calibration, and knowledge graph construction. We implement this using large language models and medical knowledge graphs developed from open-source medical ontologies. We quantitatively evaluate our system on entity extraction, relation extraction, and the performance of knowledge graph construction.   Results: AutoRD achieves an overall F1 score of 47.3%, a 14.4% improvement compared to the base LLM. In detail, AutoRD achieves an overall entity extraction F1 score of 56.1% (rare_disease: 83.5%, disease: 35.8%, s
    
[^2]: 大语言模型的适用于头部共享注意力

    Head-wise Shareable Attention for Large Language Models

    [https://arxiv.org/abs/2402.11819](https://arxiv.org/abs/2402.11819)

    本文提出了面向大语言模型的头部共享注意力的观点，提出了两种在注意力头之间共享参数的内存高效方法，以解决大型语言模型参数数量巨大导致部署受限的问题。

    

    大型语言模型(LLMs)由于参数数量巨大受到限制，这限制了它们在边缘设备上的部署。参数共享是一种有利的解决方案，鼓励权重重用，有效地减少内存使用量并降低性能下降。然而，当前的参数共享技术主要专注于像BERT这样的小规模模型，并采用粗粒度的共享规则，例如逐层共享。鉴于LLMs的普及，这变得有限，并且共享整个层或块显然降低了参数共享的灵活性。在本文中，我们提出了$\textbf{面向大语言模型的头部共享注意力}$的观点。我们进一步提出了两种在注意力头之间共享参数的内存高效方法，特别关注LLMs。它们都使用相同的动态策略选择共享的参数矩阵。第一种方法直接重复使用预训练权重而无需重新训练。

    arXiv:2402.11819v1 Announce Type: new  Abstract: Large Language Models (LLMs) suffer from huge number of parameters, which restricts their deployment on edge devices. Weight sharing is one promising solution that encourages weight reuse, effectively reducing memory usage with less performance drop. However, current weight sharing techniques primarily focus on small-scale models like BERT and employ coarse-grained sharing rules, e.g., layer-wise. This becomes limiting given the prevalence of LLMs and sharing an entire layer or block obviously diminishes the flexibility of weight sharing. In this paper, we present a perspective on $\textit{$\textbf{head-wise shareable attention for large language models}$}$. We further propose two memory-efficient methods that share parameters across attention heads, with a specific focus on LLMs. Both of them use the same dynamic strategy to select the shared weight matrices. The first method directly reuses the pre-trained weights without retraining, d
    
[^3]: 大型语言模型在表格处理中的应用：一项调查

    Large Language Model for Table Processing: A Survey

    [https://arxiv.org/abs/2402.05121](https://arxiv.org/abs/2402.05121)

    该调查综述了大型语言模型在表格处理中的应用，包括传统的表格问题回答和事实验证，以及新兴的表格操作和高级表格数据分析。还讨论了LLMs的最新范例，特别关注了指导调整、提示和基于代理的方法。

    

    表格通常是二维结构化的，用于存储大量数据，在数据库查询、电子表格计算和从网络表格生成报告等日常活动中起着重要作用。利用大型语言模型（LLMs）自动化这些以表格为中心的任务可以带来重大的公众利益，引起了学术界和工业界的兴趣。该调查对表格任务进行了广泛的概述，不仅涵盖传统领域如表格问题回答（Table QA）和事实验证，还包括最近强调的方面，如表格操作和高级表格数据分析。此外，它还超越了早期的预训练和微调小型语言模型的策略，包括LLM使用中的最新范例。重点是LLMs领域内的指导调整、提示和基于代理的方法。最后，我们重点介绍了几个挑战，涵盖私有部署、高效推断和 LLMS 发展等方面。

    Tables, typically two-dimensional and structured to store large amounts of data, are essential in daily activities like database queries, spreadsheet calculations, and generating reports from web tables. Automating these table-centric tasks with Large Language Models (LLMs) offers significant public benefits, garnering interest from academia and industry. This survey provides an extensive overview of table tasks, encompassing not only the traditional areas like table question answering (Table QA) and fact verification, but also newly emphasized aspects such as table manipulation and advanced table data analysis. Additionally, it goes beyond the early strategies of pre-training and fine-tuning small language models, to include recent paradigms in LLM usage. The focus here is particularly on instruction-tuning, prompting, and agent-based approaches within the realm of LLMs. Finally, we highlight several challenges, ranging from private deployment and efficient inference to the developmen
    

