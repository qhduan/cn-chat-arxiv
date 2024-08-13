# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ghost Sentence: A Tool for Everyday Users to Copyright Data from Large Language Models](https://arxiv.org/abs/2403.15740) | 通过在文档中插入个人密码并识别生成内容中的“幽灵句子”，普通用户可以确认大型语言模型是否滥用其数据，从而实现数据版权保护。 |
| [^2] | [PK-ICR: Persona-Knowledge Interactive Context Retrieval for Grounded Dialogue.](http://arxiv.org/abs/2302.06674) | PK-ICR是一种基于角色和知识的互动上下文检索方法，可以在复杂的多场景对话中同时识别角色和知识。通过利用神经问答检索模型，该方法可以在较少的计算资源下实现检索，并且通过引入空-正向排名测试方法来提高排名性能。 |

# 详细

[^1]: Ghost Sentence：一种供普通用户使用的工具，用于对大型语言模型中的数据进行版权保护

    Ghost Sentence: A Tool for Everyday Users to Copyright Data from Large Language Models

    [https://arxiv.org/abs/2403.15740](https://arxiv.org/abs/2403.15740)

    通过在文档中插入个人密码并识别生成内容中的“幽灵句子”，普通用户可以确认大型语言模型是否滥用其数据，从而实现数据版权保护。

    

    Web用户数据在预训练大型语言模型（LLMs）及其微调变种的生态系统中起着核心作用。本文提出了一种方法，建议用户在其文档中反复插入个人密码，使LLMs能够记忆这些密码。这些用户文档中隐藏的密码，被称为“幽灵句子”，一旦它们出现在LLMs生成的内容中，用户就可以确信他们的数据被用于训练。为了探索这种版权工具的有效性和用法，我们利用幽灵句子定义了“用户训练数据识别”任务。我们创建了来自不同来源、不同规模的多个数据集，并使用不同规模的LLMs进行测试。为了评估，我们引入了一个最后$k$个单词验证的方式。

    arXiv:2403.15740v1 Announce Type: new  Abstract: Web user data plays a central role in the ecosystem of pre-trained large language models (LLMs) and their fine-tuned variants. Billions of data are crawled from the web and fed to LLMs. How can \textit{\textbf{everyday web users}} confirm if LLMs misuse their data without permission? In this work, we suggest that users repeatedly insert personal passphrases into their documents, enabling LLMs to memorize them. These concealed passphrases in user documents, referred to as \textit{ghost sentences}, once they are identified in the generated content of LLMs, users can be sure that their data is used for training. To explore the effectiveness and usage of this copyrighting tool, we define the \textit{user training data identification} task with ghost sentences. Multiple datasets from various sources at different scales are created and tested with LLMs of different sizes. For evaluation, we introduce a last $k$ words verification manner along 
    
[^2]: PK-ICR: 基于角色和知识的互动上下文检索进行基于场景对话

    PK-ICR: Persona-Knowledge Interactive Context Retrieval for Grounded Dialogue. (arXiv:2302.06674v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.06674](http://arxiv.org/abs/2302.06674)

    PK-ICR是一种基于角色和知识的互动上下文检索方法，可以在复杂的多场景对话中同时识别角色和知识。通过利用神经问答检索模型，该方法可以在较少的计算资源下实现检索，并且通过引入空-正向排名测试方法来提高排名性能。

    

    鉴别与对话系统相关的角色和知识对于基于场景的对话应答生成至关重要。然而，目前每个对话基本上都是孤立研究的，而最近的工作中引入了更实际的多场景对话任务。我们将角色和知识双上下文识别定义为为给定的对话同时识别角色和知识的任务，在复杂的多场景对话设置中可能具有提升重要性。我们开发了一种新的基于检索的检索方法，可以同时利用对话的所有上下文信息。我们的方法通过使用神经问答检索模型，需要较少的计算资源。我们进一步介绍了一种新的空-正向排名测试方法，用于衡量与数据增强相关的语义差异样本（即困难负样本）的排名性能。

    Identifying relevant persona or knowledge for conversational systems is critical to grounded dialogue response generation. However, each grounding has been mostly researched in isolation with more practical multi-context dialogue tasks introduced in recent works. We define Persona and Knowledge Dual Context Identification as the task to identify persona and knowledge jointly for a given dialogue, which could be of elevated importance in complex multi-context dialogue settings. We develop a novel grounding retrieval method that utilizes all contexts of dialogue simultaneously. Our method requires less computational power via utilizing neural QA retrieval models. We further introduce our novel null-positive rank test which measures ranking performance on semantically dissimilar samples (i.e. hard negatives) in relation to data augmentation.
    

