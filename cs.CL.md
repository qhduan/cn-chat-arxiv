# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ghost Sentence: A Tool for Everyday Users to Copyright Data from Large Language Models](https://arxiv.org/abs/2403.15740) | 通过在文档中插入个人密码并识别生成内容中的“幽灵句子”，普通用户可以确认大型语言模型是否滥用其数据，从而实现数据版权保护。 |
| [^2] | [RAGGED: Towards Informed Design of Retrieval Augmented Generation Systems](https://arxiv.org/abs/2403.09040) | RAGGED框架分析和优化了检索增强生成系统，揭示了不同模型适合不同RAG设置的事实，编码器-解码器模型随文档数量增加而改善，而仅解码器模型只能有效利用少量文档。 |

# 详细

[^1]: Ghost Sentence：一种供普通用户使用的工具，用于对大型语言模型中的数据进行版权保护

    Ghost Sentence: A Tool for Everyday Users to Copyright Data from Large Language Models

    [https://arxiv.org/abs/2403.15740](https://arxiv.org/abs/2403.15740)

    通过在文档中插入个人密码并识别生成内容中的“幽灵句子”，普通用户可以确认大型语言模型是否滥用其数据，从而实现数据版权保护。

    

    Web用户数据在预训练大型语言模型（LLMs）及其微调变种的生态系统中起着核心作用。本文提出了一种方法，建议用户在其文档中反复插入个人密码，使LLMs能够记忆这些密码。这些用户文档中隐藏的密码，被称为“幽灵句子”，一旦它们出现在LLMs生成的内容中，用户就可以确信他们的数据被用于训练。为了探索这种版权工具的有效性和用法，我们利用幽灵句子定义了“用户训练数据识别”任务。我们创建了来自不同来源、不同规模的多个数据集，并使用不同规模的LLMs进行测试。为了评估，我们引入了一个最后$k$个单词验证的方式。

    arXiv:2403.15740v1 Announce Type: new  Abstract: Web user data plays a central role in the ecosystem of pre-trained large language models (LLMs) and their fine-tuned variants. Billions of data are crawled from the web and fed to LLMs. How can \textit{\textbf{everyday web users}} confirm if LLMs misuse their data without permission? In this work, we suggest that users repeatedly insert personal passphrases into their documents, enabling LLMs to memorize them. These concealed passphrases in user documents, referred to as \textit{ghost sentences}, once they are identified in the generated content of LLMs, users can be sure that their data is used for training. To explore the effectiveness and usage of this copyrighting tool, we define the \textit{user training data identification} task with ghost sentences. Multiple datasets from various sources at different scales are created and tested with LLMs of different sizes. For evaluation, we introduce a last $k$ words verification manner along 
    
[^2]: RAGGED:朝着基于检索增强生成系统的知情设计

    RAGGED: Towards Informed Design of Retrieval Augmented Generation Systems

    [https://arxiv.org/abs/2403.09040](https://arxiv.org/abs/2403.09040)

    RAGGED框架分析和优化了检索增强生成系统，揭示了不同模型适合不同RAG设置的事实，编码器-解码器模型随文档数量增加而改善，而仅解码器模型只能有效利用少量文档。

    

    arXiv:2403.09040v1 声明类型: 新的 摘要: 检索增强生成（RAG）通过为文档型问答等任务提供附加上下文，极大地提升了语言模型（LMs）的性能。尽管具有潜力，但RAG的效力高度依赖于其配置，从而引发一个问题：什么是最佳RAG配置？为了回答这个问题，我们引入了RAGGED框架来分析和优化RAG系统。在一组代表性的文档型问答任务上，我们研究了两种经典的稀疏和密集检索器，以及四种在编码器-解码器和仅解码器结构中表现优异的LMs。通过RAGGED，我们发现不同模型适合完全不同的RAG设置。虽然编码器-解码器模型随着更多文档的增加而单调提升，但我们发现仅解码器模型只能有效地使用<5个文档，尽管通常具有更长的上下文窗口。RAGGED进一步揭示了LMs的上下文利用习惯，我们发现编码器-解码器模型...

    arXiv:2403.09040v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) greatly benefits language models (LMs) by providing additional context for tasks such as document-based question answering (DBQA). Despite its potential, the power of RAG is highly dependent on its configuration, raising the question: What is the optimal RAG configuration? To answer this, we introduce the RAGGED framework to analyze and optimize RAG systems. On a set of representative DBQA tasks, we study two classic sparse and dense retrievers, and four top-performing LMs in encoder-decoder and decoder-only architectures. Through RAGGED, we uncover that different models suit substantially varied RAG setups. While encoder-decoder models monotonically improve with more documents, we find decoder-only models can only effectively use < 5 documents, despite often having a longer context window. RAGGED offers further insights into LMs' context utilization habits, where we find that encoder-decoder models r
    

