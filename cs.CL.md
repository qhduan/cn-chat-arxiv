# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A RAG-based Question Answering System Proposal for Understanding Islam: MufassirQAS LLM.](http://arxiv.org/abs/2401.15378) | 基于RAG的MufassirQAS问答系统利用NLP技术建立联系并准确回答复杂问题，提高了LLMs的准确性和透明度，帮助理解伊斯兰教的复杂性和教义深度。 |
| [^2] | [Unleashing Infinite-Length Input Capacity for Large-scale Language Models with Self-Controlled Memory System.](http://arxiv.org/abs/2304.13343) | 该论文提出了一种自控内存系统，可以使大规模语言模型能够处理任意长度的输入，从而显著提高模型的性能表现。 |

# 详细

[^1]: 基于RAG的理解伊斯兰教问题回答系统提案：MufassirQAS LLM

    A RAG-based Question Answering System Proposal for Understanding Islam: MufassirQAS LLM. (arXiv:2401.15378v1 [cs.CL])

    [http://arxiv.org/abs/2401.15378](http://arxiv.org/abs/2401.15378)

    基于RAG的MufassirQAS问答系统利用NLP技术建立联系并准确回答复杂问题，提高了LLMs的准确性和透明度，帮助理解伊斯兰教的复杂性和教义深度。

    

    学习和理解宗教存在复杂性和教义深度的挑战。问答机器人作为解决这些挑战的问题回答系统，可以帮助。LLM聊天机器人利用自然语言处理技术建立主题之间的联系，准确回答复杂问题。这些能力使其成为用于宗教启蒙的问题回答聊天机器人的理想选择。然而，LLM也有生成虚假信息的倾向，称为幻觉。聊天机器人的回答可能包含侮辱个人宗教信仰、跨宗派冲突和有争议或敏感的话题的内容。它需要避免这种情况，而不会宣扬仇恨言论或冒犯某些群体的人或他们的信仰。本研究使用基于向量数据库的检索增强生成（RAG）方法来提高LLMs的准确性和透明度。我们的问答系统称为"MufassirQAS"。我们创建了一个模型来评估该系统并证明其在解决宗教行业问题中的效果。

    There exist challenges in learning and understanding religions as the presence of complexity and depth of religious doctrines and teachings. Chatbots as question-answering systems can help in solving these challenges. LLM chatbots use NLP techniques to establish connections between topics and accurately respond to complex questions. These capabilities make it perfect to be used in enlightenment on religion as a question answering chatbot. However, LLMs also have a tendency to generate false information, known as hallucination. The responses of the chatbots can include content that insults personal religious beliefs, interfaith conflicts, and controversial or sensitive topics. It needs to avoid such cases without promoting hate speech or offending certain groups of people or their beliefs. This study uses a vector database-based Retrieval Augmented Generation (RAG) approach to enhance the accuracy and transparency of LLMs. Our question-answering system is called as "MufassirQAS". We cre
    
[^2]: 自控内存系统释放大规模语言模型的无限输入容量

    Unleashing Infinite-Length Input Capacity for Large-scale Language Models with Self-Controlled Memory System. (arXiv:2304.13343v1 [cs.CL])

    [http://arxiv.org/abs/2304.13343](http://arxiv.org/abs/2304.13343)

    该论文提出了一种自控内存系统，可以使大规模语言模型能够处理任意长度的输入，从而显著提高模型的性能表现。

    

    大规模语言模型（LLMs）受制于无法处理过长的输入。为了解决这个问题，我们提出了自控内存（SCM）系统，以释放大规模语言模型的无限输入容量。我们的SCM系统由三个关键模块组成：语言模型代理、内存流和内存控制器。语言模型代理迭代地处理超长输入，并将所有历史信息存储在内存流中。内存控制器为代理提供长期存储器（归档存储器）和短期存储器（闪存），以生成精确连贯的响应。控制器确定应激活哪些来自归档存储器的记忆，并如何将它们合并到模型输入中。我们的SCM系统可以与任何LLMs集成，以使它们能够处理超长文本而无需修改或微调。实验结果表明，我们的SCM系统使得LLMs能够处理长度高达8192个令牌的输入，实现了在多个基准数据集上的最佳表现，证明了它在提高大规模语言模型性能方面的有效性。

    Large-scale Language Models (LLMs) are constrained by their inability to process lengthy inputs. To address this limitation, we propose the Self-Controlled Memory (SCM) system to unleash infinite-length input capacity for large-scale language models. Our SCM system is composed of three key modules: the language model agent, the memory stream, and the memory controller. The language model agent iteratively processes ultra-long inputs and stores all historical information in the memory stream. The memory controller provides the agent with both long-term memory (archived memory) and short-term memory (flash memory) to generate precise and coherent responses. The controller determines which memories from archived memory should be activated and how to incorporate them into the model input. Our SCM system can be integrated with any LLMs to enable them to process ultra-long texts without any modification or fine-tuning. Experimental results show that our SCM system enables LLMs, which are not
    

