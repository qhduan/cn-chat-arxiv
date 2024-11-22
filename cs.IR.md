# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [REAR: A Relevance-Aware Retrieval-Augmented Framework for Open-Domain Question Answering](https://arxiv.org/abs/2402.17497) | 提出了一种名为REAR的新方法，旨在解决大型语言模型在检索增强生成中无法准确评估检索文档相关性的问题，通过增强对检索文档相关性的自我意识，能够自适应地利用外部知识。 |

# 详细

[^1]: REAR：一种面向开放域问答的关注度感知检索增强框架

    REAR: A Relevance-Aware Retrieval-Augmented Framework for Open-Domain Question Answering

    [https://arxiv.org/abs/2402.17497](https://arxiv.org/abs/2402.17497)

    提出了一种名为REAR的新方法，旨在解决大型语言模型在检索增强生成中无法准确评估检索文档相关性的问题，通过增强对检索文档相关性的自我意识，能够自适应地利用外部知识。

    

    考虑到有限的内部参数化知识，检索增强生成（RAG）被广泛用于扩展大型语言模型（LLMs）的知识范围。尽管在RAG研究上进行了大量努力，但在现有方法中，LLMs 无法准确评估检索文档的相关性，因此很可能导致对外部知识（即检索文档）的误导甚至错误利用。为解决这一问题，本文提出了 REAR，一种面向开放域问答（QA）的关注度感知检索增强方法。作为关键动机，我们旨在增强LLMs对来源相关性的自我意识，以便在RAG系统中自适应地利用外部知识。特别是，我们开发了一种新的基于LLM的RAG系统架构，通过整合一个精确评估检索文档相关性的特别设计的排名头。此外，我们提出了一种改进的训练方法。

    arXiv:2402.17497v1 Announce Type: new  Abstract: Considering the limited internal parametric knowledge, retrieval-augmented generation (RAG) has been widely used to extend the knowledge scope of large language models (LLMs). Despite the extensive efforts on RAG research, in existing methods, LLMs cannot precisely assess the relevance of retrieved documents, thus likely leading to misleading or even incorrect utilization of external knowledge (i.e., retrieved documents). To address this issue, in this paper, we propose REAR, a RElevance-Aware Retrieval-augmented approach for open-domain question answering (QA). As the key motivation, we aim to enhance the self-awareness of source relevance for LLMs, so as to adaptively utilize external knowledge in RAG systems. Specially, we develop a new architecture for LLM based RAG system, by incorporating a specially designed rank head that precisely assesses the relevance of retrieved documents. Furthermore, we propose an improved training method 
    

