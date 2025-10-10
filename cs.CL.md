# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Depression Detection on Social Media with Large Language Models](https://arxiv.org/abs/2403.10750) | 提出了名为DORIS的新型抑郁症检测系统，将医学知识和大语言模型的最新进展相结合，通过分析个人在社交媒体上的帖子历史记录来确定抑郁症患者，以提高早期检测和干预。 |
| [^2] | [ActiveRAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/abs/2402.13547) | ActiveRAG是一个创新的RAG框架，通过引入主动学习机制，利用知识构建和认知联结机制来提升大型语言模型（LLMs）的内在认知，实现了明显的性能提升。 |
| [^3] | [Assessing LLMs' Mathematical Reasoning in Financial Document Question Answering](https://arxiv.org/abs/2402.11194) | 通过实验评估了LLMs在金融表格问答中的数学推理能力，发现引入了一种新型提示技术，能够在性能上胜过其他基线模型 |

# 详细

[^1]: 社交媒体上利用大语言模型进行抑郁症检测

    Depression Detection on Social Media with Large Language Models

    [https://arxiv.org/abs/2403.10750](https://arxiv.org/abs/2403.10750)

    提出了名为DORIS的新型抑郁症检测系统，将医学知识和大语言模型的最新进展相结合，通过分析个人在社交媒体上的帖子历史记录来确定抑郁症患者，以提高早期检测和干预。

    

    抑郁症造成危害。然而，由于缺乏心理健康意识和对病症耻辱感的恐惧，许多患者并未积极寻求诊断和治疗，导致不利后果。抑郁症检测旨在通过分析社交媒体上个人帖子的历史记录来确定个体是否患有抑郁症，这可显著有助于早期检测和干预。它主要面临两个关键挑战：1）需要专业医学知识，2）需要高准确性和可解释性。为了解决这一问题，我们提出了一个名为DORIS的新型抑郁症检测系统，结合了医学知识和大语言模型的最新进展。具体来说，为了解决第一个挑战，我们提出了一种基于大语言模型的解决方案，首先对高危文本进行标注以确定是否符合医学诊断标准。

    arXiv:2403.10750v1 Announce Type: cross  Abstract: Depression harms. However, due to a lack of mental health awareness and fear of stigma, many patients do not actively seek diagnosis and treatment, leading to detrimental outcomes. Depression detection aims to determine whether an individual suffers from depression by analyzing their history of posts on social media, which can significantly aid in early detection and intervention. It mainly faces two key challenges: 1) it requires professional medical knowledge, and 2) it necessitates both high accuracy and explainability. To address it, we propose a novel depression detection system called DORIS, combining medical knowledge and the recent advances in large language models (LLMs). Specifically, to tackle the first challenge, we proposed an LLM-based solution to first annotate whether high-risk texts meet medical diagnostic criteria. Further, we retrieve texts with high emotional intensity and summarize critical information from the his
    
[^2]: ActiveRAG: 通过主动学习揭示知识的宝藏

    ActiveRAG: Revealing the Treasures of Knowledge via Active Learning

    [https://arxiv.org/abs/2402.13547](https://arxiv.org/abs/2402.13547)

    ActiveRAG是一个创新的RAG框架，通过引入主动学习机制，利用知识构建和认知联结机制来提升大型语言模型（LLMs）的内在认知，实现了明显的性能提升。

    

    arXiv:2402.13547v1 公告类型：新摘要：检索增强生成（RAG）引入了一种新的大型语言模型（LLM）范例，有助于解决知识密集型任务。然而，当前的RAG模型将LLMs定位为被动的知识接收器，从而限制了它们学习和理解外部知识的能力。本文提出了ActiveRAG，它是一种创新的RAG框架，从被动知识获取转变为主动学习机制。这种方法利用知识构建机制通过将外部知识与先前获取或记忆的知识相关联来更深入地理解外部知识。随后，它设计了认知联结机制以合并来自思维和知识构建链的成果，从而校准LLMs的内在认知。我们的实验结果表明，ActiveRAG超越了先前的RAG模型，在问题回答上实现了5%的改进。

    arXiv:2402.13547v1 Announce Type: new  Abstract: Retrieval Augmented Generation (RAG) has introduced a new paradigm for Large Language Models (LLMs), aiding in the resolution of knowledge-intensive tasks. However, current RAG models position LLMs as passive knowledge receptors, thereby restricting their capacity for learning and comprehending external knowledge. In this paper, we present ActiveRAG, an innovative RAG framework that shifts from passive knowledge acquisition to an active learning mechanism. This approach utilizes the Knowledge Construction mechanism to develop a deeper understanding of external knowledge by associating it with previously acquired or memorized knowledge. Subsequently, it designs the Cognitive Nexus mechanism to incorporate the outcomes from both chains of thought and knowledge construction, thereby calibrating the intrinsic cognition of LLMs. Our experimental results demonstrate that ActiveRAG surpasses previous RAG models, achieving a 5% improvement on qu
    
[^3]: 在金融文档问答中评估LLMs的数学推理能力

    Assessing LLMs' Mathematical Reasoning in Financial Document Question Answering

    [https://arxiv.org/abs/2402.11194](https://arxiv.org/abs/2402.11194)

    通过实验评估了LLMs在金融表格问答中的数学推理能力，发现引入了一种新型提示技术，能够在性能上胜过其他基线模型

    

    大型语言模型（LLMs）在自然语言理解方面表现出色，但它们在具有结构化表格和非结构化文本混合的复杂数学推理方面的能力尚不确定。本研究探讨了LLMs在四个金融表格问答数据集上的数学推理能力：TATQA、FinQA、ConvFinQA和Multihiertt。通过对各种模型和提示技术进行广泛实验，我们评估了LLMs如何适应复杂表格和数学任务。我们关注对表格复杂性的敏感性以及在增加算术推理步骤数量时性能变化。结果揭示了LLMs处理半结构化表格中复杂数学场景的能力和局限性。最终，我们引入了一种针对半结构化文档的新型提示技术，在性能方面与其他基线相匹配或胜过，并提供了对LLMs能力的微妙理解。

    arXiv:2402.11194v1 Announce Type: new  Abstract: Large Language Models (LLMs), excel in natural language understanding, but their capability for complex mathematical reasoning with an amalgamation of structured tables and unstructured text is uncertain. This study explores LLMs' mathematical reasoning on four financial tabular question-answering datasets: TATQA, FinQA, ConvFinQA, and Multihiertt. Through extensive experiments with various models and prompting techniques, we assess how LLMs adapt to complex tables and mathematical tasks. We focus on sensitivity to table complexity and performance variations with an increasing number of arithmetic reasoning steps. The results provide insights into LLMs' capabilities and limitations in handling complex mathematical scenarios for semi-structured tables. Ultimately, we introduce a novel prompting technique tailored to semi-structured documents, matching or outperforming other baselines in performance while providing a nuanced understanding 
    

