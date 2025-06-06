# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bias in Language Models: Beyond Trick Tests and Toward RUTEd Evaluation](https://arxiv.org/abs/2402.12649) | 这项研究探讨了语言模型中偏见的负面影响，研究了"技巧测试"与更现实世界中表现的RUTEd评估之间的关联性，特别关注性别-职业偏见，并进行了多项评估比较。 |
| [^2] | [The Vector Grounding Problem.](http://arxiv.org/abs/2304.01481) | 本文讨论了大型语言模型中的向量接地问题，通过区分内部表示接地的不同方式，总结出五个概念。 |

# 详细

[^1]: 语言模型中的偏见：超越技巧测试，走向RUTEd评估

    Bias in Language Models: Beyond Trick Tests and Toward RUTEd Evaluation

    [https://arxiv.org/abs/2402.12649](https://arxiv.org/abs/2402.12649)

    这项研究探讨了语言模型中偏见的负面影响，研究了"技巧测试"与更现实世界中表现的RUTEd评估之间的关联性，特别关注性别-职业偏见，并进行了多项评估比较。

    

    Bias benchmarks are a popular method for studying the negative impacts of bias in LLMs, yet there has been little empirical investigation of whether these benchmarks are actually indicative of how real world harm may manifest in the real world. In this work, we study the correspondence between such decontextualized "trick tests" and evaluations that are more grounded in Realistic Use and Tangible {Effects (i.e. RUTEd evaluations). We explore this correlation in the context of gender-occupation bias--a popular genre of bias evaluation. We compare three de-contextualized evaluations adapted from the current literature to three analogous RUTEd evaluations applied to long-form content generation. We conduct each evaluation for seven instruction-tuned LLMs. For the RUTEd evaluations, we conduct repeated trials of three text generation tasks: children's bedtime stories, user personas, and English language learning exercises. We found no corres

    arXiv:2402.12649v1 Announce Type: new  Abstract: Bias benchmarks are a popular method for studying the negative impacts of bias in LLMs, yet there has been little empirical investigation of whether these benchmarks are actually indicative of how real world harm may manifest in the real world. In this work, we study the correspondence between such decontextualized "trick tests" and evaluations that are more grounded in Realistic Use and Tangible {Effects (i.e. RUTEd evaluations). We explore this correlation in the context of gender-occupation bias--a popular genre of bias evaluation. We compare three de-contextualized evaluations adapted from the current literature to three analogous RUTEd evaluations applied to long-form content generation. We conduct each evaluation for seven instruction-tuned LLMs. For the RUTEd evaluations, we conduct repeated trials of three text generation tasks: children's bedtime stories, user personas, and English language learning exercises. We found no corres
    
[^2]: 向量接地问题

    The Vector Grounding Problem. (arXiv:2304.01481v1 [cs.CL])

    [http://arxiv.org/abs/2304.01481](http://arxiv.org/abs/2304.01481)

    本文讨论了大型语言模型中的向量接地问题，通过区分内部表示接地的不同方式，总结出五个概念。

    

    大型语言模型(LLMs)在处理复杂的语言任务上表现出色，引发了对它们能力本质的激烈辩论。不同于人类，这些模型只能从文本数据中学习语言，没有与真实世界的直接交互。尽管如此，它们能够生成关于各种话题似乎有意义的文本。这一印象深刻的成就重新引起了对经典“符号接地问题”的关注，这个问题质疑了经典符号AI系统的内部表示和输出能否具有内在意义。与这些系统不同，现代LLMs是计算向量而不是符号的人工神经网络。然而，这样的系统也有类似的问题，我们称之为向量接地问题。本文有两个主要目标。首先，我们区分了生物或人工系统中内部表示可以接地的各种方式，确定了五个不同的概念

    The remarkable performance of large language models (LLMs) on complex linguistic tasks has sparked a lively debate on the nature of their capabilities. Unlike humans, these models learn language exclusively from textual data, without direct interaction with the real world. Nevertheless, they can generate seemingly meaningful text about a wide range of topics. This impressive accomplishment has rekindled interest in the classical 'Symbol Grounding Problem,' which questioned whether the internal representations and outputs of classical symbolic AI systems could possess intrinsic meaning. Unlike these systems, modern LLMs are artificial neural networks that compute over vectors rather than symbols. However, an analogous problem arises for such systems, which we dub the Vector Grounding Problem. This paper has two primary objectives. First, we differentiate various ways in which internal representations can be grounded in biological or artificial systems, identifying five distinct notions 
    

