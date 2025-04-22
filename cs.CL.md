# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Is Translation All You Need? A Study on Solving Multilingual Tasks with Large Language Models](https://arxiv.org/abs/2403.10258) | 提出了通过本地语言提示来解决文化相关任务的方法，并呼吁发展强大的多语言LLMs。 |
| [^2] | [Embedding Ontologies via Incoprorating Extensional and Intensional Knowledge](https://arxiv.org/abs/2402.01677) | 本文提出了一种新型本体嵌入方法EIKE，通过整合外延知识和内涵知识，在外延空间和内涵空间中表示本体，并采用基于几何的方法和预训练的语言模型对实例、概念和关系进行嵌入建模。 |
| [^3] | [GLoRE: Evaluating Logical Reasoning of Large Language Models.](http://arxiv.org/abs/2310.09107) | 本论文介绍了GLoRE，一个评估大型语言模型逻辑推理能力的基准，实验结果表明开放式LLM模型的逻辑推理能力需要提高。研究提出了一种自一致性探测方法和微调方法来改进ChatGPT和开放式LLM的性能。 |

# 详细

[^1]: 翻译到底是你所需要的全部吗？使用大型语言模型解决多语言任务的研究

    Is Translation All You Need? A Study on Solving Multilingual Tasks with Large Language Models

    [https://arxiv.org/abs/2403.10258](https://arxiv.org/abs/2403.10258)

    提出了通过本地语言提示来解决文化相关任务的方法，并呼吁发展强大的多语言LLMs。

    

    大型语言模型（LLMs）展示了强大的多语言能力；然而，由于训练语料库不平衡，它们大多是以英语为中心的。现有研究利用这一现象来提高它们在自然语言处理任务上的多语言性能。在本研究中，我们将评估从自然语言处理任务扩展到真实用户查询。我们发现，尽管将文本翻译成英语可以帮助提高以英语为中心的LLMs在多语言自然语言处理任务中的性能，但并不一定适用于所有场景。对于需要深入理解语言的文化相关任务，以本地语言提示更为有前景，因为它可以捕捉与文化和语言相关的微妙之处。因此，我们主张着力发展强大的多语言LLMs，而不仅仅是以英语为中心的LLMs。

    arXiv:2403.10258v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated strong multilingual capabilities; yet, they are mostly English-centric due to the imbalanced training corpora. Existing works leverage this phenomenon to improve their multilingual performances on NLP tasks. In this work, we extend the evaluation from NLP tasks to real user queries. We find that even though translation into English can help improve the performance of multilingual NLP tasks for English-centric LLMs, it may not be optimal for all scenarios. For culture-related tasks that need deep language understanding, prompting in the native language proves to be more promising since it can capture the nuances related to culture and language. Therefore, we advocate for more efforts towards the development of strong multilingual LLMs instead of just English-centric LLMs.
    
[^2]: 通过整合外延知识和内涵知识嵌入本体

    Embedding Ontologies via Incoprorating Extensional and Intensional Knowledge

    [https://arxiv.org/abs/2402.01677](https://arxiv.org/abs/2402.01677)

    本文提出了一种新型本体嵌入方法EIKE，通过整合外延知识和内涵知识，在外延空间和内涵空间中表示本体，并采用基于几何的方法和预训练的语言模型对实例、概念和关系进行嵌入建模。

    

    本体包含领域内丰富的知识，可以分为两个类别，即外延知识和内涵知识。外延知识提供关于本体中特定概念所属的具体实例的信息，而内涵知识详细描述了概念之间的内在属性、特征和语义关联。然而，现有的本体嵌入方法未能同时充分考虑外延知识和内涵知识。在本文中，我们提出了一种名为EIKE（Extensional and Intensional Knowledge Embedding）的新型本体嵌入方法，通过在外延空间和内涵空间中表示本体。EIKE提出了一个统一的框架，用于将实例、概念及其关系嵌入到本体中，采用基于几何的方法对外延知识进行建模，并使用预训练的语言模型对内涵知识进行建模。

    Ontologies contain rich knowledge within domain, which can be divided into two categories, namely extensional knowledge and intensional knowledge. Extensional knowledge provides information about the concrete instances that belong to specific concepts in the ontology, while intensional knowledge details inherent properties, characteristics, and semantic associations among concepts. However, existing ontology embedding approaches fail to take both extensional knowledge and intensional knowledge into fine consideration simultaneously. In this paper, we propose a novel ontology embedding approach named EIKE (Extensional and Intensional Knowledge Embedding) by representing ontologies in two spaces, called extensional space and intensional space. EIKE presents a unified framework for embedding instances, concepts and their relations in an ontology, applying a geometry-based method to model extensional knowledge and a pretrained language model to model intensional knowledge, which can captur
    
[^3]: GLoRE: 评估大型语言模型的逻辑推理能力

    GLoRE: Evaluating Logical Reasoning of Large Language Models. (arXiv:2310.09107v1 [cs.CL])

    [http://arxiv.org/abs/2310.09107](http://arxiv.org/abs/2310.09107)

    本论文介绍了GLoRE，一个评估大型语言模型逻辑推理能力的基准，实验结果表明开放式LLM模型的逻辑推理能力需要提高。研究提出了一种自一致性探测方法和微调方法来改进ChatGPT和开放式LLM的性能。

    

    最近，包括GPT-4和新兴社区模型在内的大型语言模型(LLMs)展示了显著的通用语言理解能力。然而，对这些LLMs的逻辑推理能力进行评估的尝试还很少，而这是自然语言理解的一个重要方面。为了鼓励进一步研究，我们引入了GLoRE，一个精心组织的通用逻辑推理评估基准，包含了12个覆盖三种不同类型任务的数据集。我们的实验结果显示，与人类和监督微调的性能相比，开放式LLM模型的逻辑推理能力需要进一步提高；ChatGPT和GPT-4展示了较强的逻辑推理能力，GPT-4大幅超过了ChatGPT。我们提出了一种自一致性探测方法来提高ChatGPT的准确性，以及一种微调方法来提高开放式LLM的性能。

    Recently, large language models (LLMs), including notable models such as GPT-4 and burgeoning community models, have showcased significant general language understanding abilities. However, there has been a scarcity of attempts to assess the logical reasoning capacities of these LLMs, an essential facet of natural language understanding. To encourage further investigation in this area, we introduce GLoRE, a meticulously assembled General Logical Reasoning Evaluation benchmark comprised of 12 datasets that span three different types of tasks. Our experimental results show that compared to the performance of human and supervised fine-tuning, the logical reasoning capabilities of open LLM models necessitate additional improvement; ChatGPT and GPT-4 show a strong capability of logical reasoning, with GPT-4 surpassing ChatGPT by a large margin. We propose a self-consistency probing method to enhance the accuracy of ChatGPT and a fine-tuned method to boost the performance of an open LLM. We 
    

