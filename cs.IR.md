# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PromptCrypt: Prompt Encryption for Secure Communication with Large Language Models](https://arxiv.org/abs/2402.05868) | PromptCrypt是一种使用表情符号对用户输入进行加密的机制，保护了大型语言模型（LLM）中用户的隐私，防止数据泄露和解密。 |
| [^2] | [LitLLM: A Toolkit for Scientific Literature Review](https://arxiv.org/abs/2402.01788) | "LitLLM: A Toolkit for Scientific Literature Review" 提出了一个基于 RAG 原则的工具包，通过使用专门的提示和指导技术，结合大型语言模型（LLM），实现了科学文献综述的自动化。这个工具包不仅可以通过转化摘要为关键词进行文献检索，还可以通过补充相关论文或关键词进行定制化的检索。 |
| [^3] | [Knowledge Graph Reasoning Based on Attention GCN.](http://arxiv.org/abs/2312.10049) | 本论文通过将图卷积神经网络（GCN）与注意力机制相结合，提出了一种新颖的技术来增强知识图谱推理，通过检查实体之间及其邻居节点之间的关系以及整合实体的属性和相互作用，生成丰富的隐式特征向量，以提高实体分类和链接预测等任务的性能。 |
| [^4] | [Summaries, Highlights, and Action items: Design, implementation and evaluation of an LLM-powered meeting recap system.](http://arxiv.org/abs/2307.15793) | 这项研究设计、实现和评估了一种基于LLM的会议总结系统，通过减少个人会议负担和增加会议输出的清晰度和一致性，提高了会议体验。 |

# 详细

[^1]: PromptCrypt: 使用表情符号对大型语言模型进行安全通信的提示加密

    PromptCrypt: Prompt Encryption for Secure Communication with Large Language Models

    [https://arxiv.org/abs/2402.05868](https://arxiv.org/abs/2402.05868)

    PromptCrypt是一种使用表情符号对用户输入进行加密的机制，保护了大型语言模型（LLM）中用户的隐私，防止数据泄露和解密。

    

    基于云的大型语言模型（LLM）如ChatGPT在日常操作中变得越来越重要，成为各种应用程序中的重要工具。虽然这些模型在可访问性和功能性方面带来了重大好处，但它们也引入了重要的隐私问题：在云基础架构中传输和存储用户数据会产生重大的数据泄露和未经授权访问敏感信息的风险；即使数据的传输和存储被加密，LLM服务提供商仍然知道数据的真实内容，从而阻止个人或实体放心使用此类LLM服务。为了解决这些问题，本文提出了一种简单但有效的机制PromptCrypt来保护用户隐私。它使用表情符号对用户输入进行加密，然后将其发送到LLM，有效地使其对人类或LLM的检查无法理解，同时保留原始提示的意图，从而确保用户隐私。

    Cloud-based large language models (LLMs) such as ChatGPT have increasingly become integral to daily operations, serving as vital tools across various applications. While these models offer substantial benefits in terms of accessibility and functionality, they also introduce significant privacy concerns: the transmission and storage of user data in cloud infrastructures pose substantial risks of data breaches and unauthorized access to sensitive information; even if the transmission and storage of data is encrypted, the LLM service provider itself still knows the real contents of the data, preventing individuals or entities from confidently using such LLM services. To address these concerns, this paper proposes a simple yet effective mechanism PromptCrypt to protect user privacy. It uses Emoji to encrypt the user inputs before sending them to LLM, effectively rendering them indecipherable to human or LLM's examination while retaining the original intent of the prompt, thus ensuring the 
    
[^2]: LitLLM：科学文献综述工具包

    LitLLM: A Toolkit for Scientific Literature Review

    [https://arxiv.org/abs/2402.01788](https://arxiv.org/abs/2402.01788)

    "LitLLM: A Toolkit for Scientific Literature Review" 提出了一个基于 RAG 原则的工具包，通过使用专门的提示和指导技术，结合大型语言模型（LLM），实现了科学文献综述的自动化。这个工具包不仅可以通过转化摘要为关键词进行文献检索，还可以通过补充相关论文或关键词进行定制化的检索。

    

    进行科学论文的文献综述对于理解研究、其限制以及构建在现有工作基础上是必不可少的。这是一项繁琐的任务，因此自动文献综述生成器变得有吸引力。然而，许多使用大型语言模型（LLM）生成此类综述的现有工作存在显著限制。它们倾向于产生虚构的非实际信息，并忽略它们未受过训练的最新研究。为了解决这些限制，我们提出了一个基于检索增强生成（RAG）原则的工具包，在LLM的帮助下，使用专门的提示和指导技术。我们的系统首先通过将用户提供的摘要转化为关键词来进行网络搜索，以检索相关论文，其中使用了现成的LLM。作者可以通过补充相关论文或关键词来改进搜索，从而实现定制化的检索过程。其次，系统根据-

    Conducting literature reviews for scientific papers is essential for understanding research, its limitations, and building on existing work. It is a tedious task which makes an automatic literature review generator appealing. Unfortunately, many existing works that generate such reviews using Large Language Models (LLMs) have significant limitations. They tend to hallucinate-generate non-actual information-and ignore the latest research they have not been trained on. To address these limitations, we propose a toolkit that operates on Retrieval Augmented Generation (RAG) principles, specialized prompting and instructing techniques with the help of LLMs. Our system first initiates a web search to retrieve relevant papers by summarizing user-provided abstracts into keywords using an off-the-shelf LLM. Authors can enhance the search by supplementing it with relevant papers or keywords, contributing to a tailored retrieval process. Second, the system re-ranks the retrieved papers based on t
    
[^3]: 基于注意力GCN的知识图谱推理

    Knowledge Graph Reasoning Based on Attention GCN. (arXiv:2312.10049v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2312.10049](http://arxiv.org/abs/2312.10049)

    本论文通过将图卷积神经网络（GCN）与注意力机制相结合，提出了一种新颖的技术来增强知识图谱推理，通过检查实体之间及其邻居节点之间的关系以及整合实体的属性和相互作用，生成丰富的隐式特征向量，以提高实体分类和链接预测等任务的性能。

    

    我们提出了一种新颖的技术，通过将图卷积神经网络（GCN）与注意力机制相结合来增强知识图谱推理。该方法利用注意力机制来检查实体之间及其邻居节点之间的关系，从而为每个实体开发详细的特征向量。GCN使用共享参数有效地表示相邻实体的特征。我们首先学习实体之间的相似度，以进行节点表示学习。通过整合实体的属性和它们的相互作用，该方法为每个实体生成了丰富的隐式特征向量，提高了实体分类和链接预测等任务的性能，优于传统的神经网络模型。总之，这项工作为搜索引擎、问答系统、推荐系统和数据整合任务等多个应用领域提供了重要的方法支持。

    We propose a novel technique to enhance Knowledge Graph Reasoning by combining Graph Convolution Neural Network (GCN) with the Attention Mechanism. This approach utilizes the Attention Mechanism to examine the relationships between entities and their neighboring nodes, which helps to develop detailed feature vectors for each entity. The GCN uses shared parameters to effectively represent the characteristics of adjacent entities. We first learn the similarity of entities for node representation learning. By integrating the attributes of the entities and their interactions, this method generates extensive implicit feature vectors for each entity, improving performance in tasks including entity classification and link prediction, outperforming traditional neural network models. To conclude, this work provides crucial methodological support for a range of applications, such as search engines, question-answering systems, recommendation systems, and data integration tasks.
    
[^4]: 概要、亮点和行动项目：设计、实现和评估基于LLM的会议总结系统

    Summaries, Highlights, and Action items: Design, implementation and evaluation of an LLM-powered meeting recap system. (arXiv:2307.15793v1 [cs.HC])

    [http://arxiv.org/abs/2307.15793](http://arxiv.org/abs/2307.15793)

    这项研究设计、实现和评估了一种基于LLM的会议总结系统，通过减少个人会议负担和增加会议输出的清晰度和一致性，提高了会议体验。

    

    会议在工作协调中发挥着关键的基础设施作用。近年来，由于向混合和远程工作的转变，越来越多的会议正在转移到在线计算机媒体空间。这导致了新的问题（例如在更不吸引人的会议上花费更多的时间）和新的机会（例如自动转录/字幕和总结支持）。最近的大型语言模型（LLMs）在对话总结方面取得了进展，通过减少个人的会议负担和增加会议输出的清晰度和一致性，有可能提高会议体验。尽管存在这种潜力，但由于长篇转录和无法根据用户的上下文捕捉到多样的总结需求，它们面临着技术限制。为了填补这些差距，我们设计、实现并在上下文中评估了一种会议总结系统。我们首先构思了两个明显的总结表示方式——重要亮点和结构化的分级会议纪要视图。我们开发了一个系统来实现这些表示方法。

    Meetings play a critical infrastructural role in the coordination of work. In recent years, due to shift to hybrid and remote work, more meetings are moving to online Computer Mediated Spaces. This has led to new problems (e.g. more time spent in less engaging meetings) and new opportunities (e.g. automated transcription/captioning and recap support). Recent advances in large language models (LLMs) for dialog summarization have the potential to improve the experience of meetings by reducing individuals' meeting load and increasing the clarity and alignment of meeting outputs. Despite this potential, they face technological limitation due to long transcripts and inability to capture diverse recap needs based on user's context. To address these gaps, we design, implement and evaluate in-context a meeting recap system. We first conceptualize two salient recap representations -- important highlights, and a structured, hierarchical minutes view. We develop a system to operationalize the rep
    

