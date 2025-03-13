# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models](https://arxiv.org/abs/2402.15131) | 提出了一种互动式KBQA框架，通过直接与知识库互动生成逻辑形式，开发了用于KB交互的通用API，并设计了示例来指导大型语言模型进行推理。 |
| [^2] | [LLM-SQL-Solver: Can LLMs Determine SQL Equivalence?.](http://arxiv.org/abs/2312.10321) | 本研究探讨了LLM是否能够确定两个SQL查询的等价关系，并提出了两种提示技术来帮助LLM生成高质量的响应。 |
| [^3] | [Personality Traits in Large Language Models.](http://arxiv.org/abs/2307.00184) | 该研究介绍了一种综合方法，用于验证大型语言模型（LLMs）生成的文本中展示的人格特质。研究发现，部分LLMs在特定提示配置下模拟的人格可靠且有效，特别是对于更大和经过指导微调的模型。此外，LLMs的输出中的人格特质可以根据需要进行塑造。 |

# 详细

[^1]: 互动式知识库问答：基于大型语言模型的多轮交互式知识库问答

    Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models

    [https://arxiv.org/abs/2402.15131](https://arxiv.org/abs/2402.15131)

    提出了一种互动式KBQA框架，通过直接与知识库互动生成逻辑形式，开发了用于KB交互的通用API，并设计了示例来指导大型语言模型进行推理。

    

    本研究探讨了知识库问答（KBQA）的领域。KBQA被认为是一项具有挑战性的任务，特别是在将复杂问题解析为可执行逻辑形式方面。传统的基于语义解析（SP）的方法需要大量的数据注释，这导致了显著的成本。最近，由大型语言模型（LLM）推动的少样本上下文学习的出现展示了很好的能力。然而，在低资源情景下充分利用LLMs将问题解析为逻辑形式是一个重大挑战。为了应对这些障碍，我们引入了互动式知识库问答（Interactive-KBQA），这是一个旨在通过与知识库（KBs）直接互动来生成逻辑形式的框架。在这个框架内，我们开发了三个用于KB交互的通用API。对于每种复杂问题类别，我们设计了示例来指导LLMs完成推理过程。我们的方法取得了具有竞争力的结果。

    arXiv:2402.15131v1 Announce Type: cross  Abstract: This study explores the realm of knowledge-base question answering (KBQA). KBQA is considered a challenging task, particularly in parsing intricate questions into executable logical forms. Traditional semantic parsing (SP)-based methods require extensive data annotations, which result in significant costs. Recently, the advent of few-shot in-context learning, powered by large language models (LLMs), has showcased promising capabilities. Yet, fully leveraging LLMs to parse questions into logical forms in low-resource scenarios poses a substantial challenge. To tackle these hurdles, we introduce Interactive-KBQA, a framework designed to generate logical forms through direct interaction with knowledge bases (KBs). Within this framework, we have developed three generic APIs for KB interaction. For each category of complex question, we devised exemplars to guide LLMs through the reasoning processes. Our method achieves competitive results o
    
[^2]: LLM-SQL-Solver: LLM能够确定SQL等价关系吗？

    LLM-SQL-Solver: Can LLMs Determine SQL Equivalence?. (arXiv:2312.10321v2 [cs.DB] UPDATED)

    [http://arxiv.org/abs/2312.10321](http://arxiv.org/abs/2312.10321)

    本研究探讨了LLM是否能够确定两个SQL查询的等价关系，并提出了两种提示技术来帮助LLM生成高质量的响应。

    

    判断两个SQL查询之间的等价关系是数据管理和SQL生成中的一个基本问题，具有许多实际应用（即，在文本到SQL任务中评估生成的SQL查询的质量）。虽然研究界多年来一直在考虑SQL的等价性，但它存在相当大的困难，并且没有完整的解决方案。最近，大型语言模型（LLMs）在对话、问答和解决数学问题方面展现出强大的推理能力。在本文中，我们研究了LLMs是否可以用于确定两个SQL查询的等价性（语义等价和宽松等价）。为了帮助LLMs生成高质量的响应，我们提出了两种提示技术：Miniature & Mull和Explain & Compare。前一种技术被用于评估语义等价性，它要求LLMs在简单的数据库实例上执行查询，然后探索是否存在反例。

    Judging the equivalence between two SQL queries is a fundamental problem with many practical applications in data management and SQL generation (i.e., evaluating the quality of generated SQL queries in text-to-SQL task). While the research community has reasoned about SQL equivalence for decades, it poses considerable difficulties and no complete solutions exist. Recently, Large Language Models (LLMs) have shown strong reasoning capability in conversation, question answering and solving mathematics challenges. In this paper, we study if LLMs can be used to determine the equivalence between SQL queries under two notions of SQL equivalence (semantic equivalence and relaxed equivalence). To assist LLMs in generating high quality responses, we present two prompting techniques: Miniature & Mull and Explain & Compare. The former technique is used to evaluate the semantic equivalence in which it asks LLMs to execute a query on a simple database instance and then explore if a counterexample ex
    
[^3]: 大型语言模型中的人格特质

    Personality Traits in Large Language Models. (arXiv:2307.00184v1 [cs.CL])

    [http://arxiv.org/abs/2307.00184](http://arxiv.org/abs/2307.00184)

    该研究介绍了一种综合方法，用于验证大型语言模型（LLMs）生成的文本中展示的人格特质。研究发现，部分LLMs在特定提示配置下模拟的人格可靠且有效，特别是对于更大和经过指导微调的模型。此外，LLMs的输出中的人格特质可以根据需要进行塑造。

    

    大型语言模型（LLMs）的出现彻底改变了自然语言处理，使得能够生成连贯且上下文相关的文本。随着LLMs越来越多地用于驱动对话代理，这些模型通过训练大量人工生成的数据获得的人格特质引起了人们的关注。由于人格是决定交流效果的重要因素，我们提出了一种全面的方法来进行验证的心理测量测试，并对从广泛使用的LLMs生成的文本中展示的人格特质进行量化、分析和塑造。我们发现：1）某些LLMs的输出中模拟的人格（在特定的提示配置下）是可靠和有效的；2）LLM模拟的人格的可靠性和有效性的证据对于更大的和经过指导微调的模型更强；3）LLM输出中的人格可以根据需要的维度进行塑造，以模仿特定的人格特点。

    The advent of large language models (LLMs) has revolutionized natural language processing, enabling the generation of coherent and contextually relevant text. As LLMs increasingly power conversational agents, the synthesized personality embedded in these models by virtue of their training on large amounts of human-generated data draws attention. Since personality is an important factor determining the effectiveness of communication, we present a comprehensive method for administering validated psychometric tests and quantifying, analyzing, and shaping personality traits exhibited in text generated from widely-used LLMs. We find that: 1) personality simulated in the outputs of some LLMs (under specific prompting configurations) is reliable and valid; 2) evidence of reliability and validity of LLM-simulated personality is stronger for larger and instruction fine-tuned models; and 3) personality in LLM outputs can be shaped along desired dimensions to mimic specific personality profiles. 
    

