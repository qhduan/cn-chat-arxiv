# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving Topic Relevance Model by Mix-structured Summarization and LLM-based Data Augmentation](https://arxiv.org/abs/2404.02616) | 通过混合结构化摘要和基于LLM的数据增强方法，改进了主题相关性模型，使其能够更好地学习查询与文档之间的相关度。 |
| [^2] | [The Vector Grounding Problem.](http://arxiv.org/abs/2304.01481) | 本文讨论了大型语言模型中的向量接地问题，通过区分内部表示接地的不同方式，总结出五个概念。 |

# 详细

[^1]: 通过混合结构化摘要和基于LLM的数据增强来改进主题相关性模型

    Improving Topic Relevance Model by Mix-structured Summarization and LLM-based Data Augmentation

    [https://arxiv.org/abs/2404.02616](https://arxiv.org/abs/2404.02616)

    通过混合结构化摘要和基于LLM的数据增强方法，改进了主题相关性模型，使其能够更好地学习查询与文档之间的相关度。

    

    查询和文档之间的主题相关性是社交搜索的一个非常重要的部分，可以评估文档与用户需求之间的匹配程度。在大多数社交搜索场景中，如大众点评，建模搜索相关性总是面临两个挑战。一个是许多社交搜索中的文档非常长且包含大量冗余信息。另一个问题是搜索相关性模型的训练数据很难获得，尤其是对于多分类相关性模型。为了解决以上两个问题，我们首先将查询与基于查询的摘要以及不带查询的文档摘要合并，作为主题相关性模型的输入，这有助于模型学习查询和文档核心主题之间的相关度。然后，我们利用大型语言模型（LLM）的语言理解和生成能力，从现有训练数据中重新编写和生成查询。

    arXiv:2404.02616v1 Announce Type: cross  Abstract: Topic relevance between query and document is a very important part of social search, which can evaluate the degree of matching between document and user's requirement. In most social search scenarios such as Dianping, modeling search relevance always faces two challenges. One is that many documents in social search are very long and have much redundant information. The other is that the training data for search relevance model is difficult to get, especially for multi-classification relevance model. To tackle above two problems, we first take query concatenated with the query-based summary and the document summary without query as the input of topic relevance model, which can help model learn the relevance degree between query and the core topic of document. Then, we utilize the language understanding and generation abilities of large language model (LLM) to rewrite and generate query from queries and documents in existing training da
    
[^2]: 向量接地问题

    The Vector Grounding Problem. (arXiv:2304.01481v1 [cs.CL])

    [http://arxiv.org/abs/2304.01481](http://arxiv.org/abs/2304.01481)

    本文讨论了大型语言模型中的向量接地问题，通过区分内部表示接地的不同方式，总结出五个概念。

    

    大型语言模型(LLMs)在处理复杂的语言任务上表现出色，引发了对它们能力本质的激烈辩论。不同于人类，这些模型只能从文本数据中学习语言，没有与真实世界的直接交互。尽管如此，它们能够生成关于各种话题似乎有意义的文本。这一印象深刻的成就重新引起了对经典“符号接地问题”的关注，这个问题质疑了经典符号AI系统的内部表示和输出能否具有内在意义。与这些系统不同，现代LLMs是计算向量而不是符号的人工神经网络。然而，这样的系统也有类似的问题，我们称之为向量接地问题。本文有两个主要目标。首先，我们区分了生物或人工系统中内部表示可以接地的各种方式，确定了五个不同的概念

    The remarkable performance of large language models (LLMs) on complex linguistic tasks has sparked a lively debate on the nature of their capabilities. Unlike humans, these models learn language exclusively from textual data, without direct interaction with the real world. Nevertheless, they can generate seemingly meaningful text about a wide range of topics. This impressive accomplishment has rekindled interest in the classical 'Symbol Grounding Problem,' which questioned whether the internal representations and outputs of classical symbolic AI systems could possess intrinsic meaning. Unlike these systems, modern LLMs are artificial neural networks that compute over vectors rather than symbols. However, an analogous problem arises for such systems, which we dub the Vector Grounding Problem. This paper has two primary objectives. First, we differentiate various ways in which internal representations can be grounded in biological or artificial systems, identifying five distinct notions 
    

