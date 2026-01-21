# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GNN2R: Weakly-Supervised Rationale-Providing Question Answering over Knowledge Graphs.](http://arxiv.org/abs/2312.02317) | GNN2R是一种基于图神经网络的两步推理模型，通过弱监督训练，能够在知识图谱问答中提供最终答案以及推理子图的理由。该方法解决了现有方法缺乏解释以及效率低下的问题。 |
| [^2] | [Contextualising Levels of Language Resourcedness affecting Digital Processing of Text.](http://arxiv.org/abs/2309.17035) | 本文讨论了数字处理文本时，语言资源水平对其影响的上下文化分类，并提出了一个将语言划分为五个等级的矩阵，从而解决了将语言只划分为LRL和HRL两种类型的问题。 |
| [^3] | [Generative Language Models on Nucleotide Sequences of Human Genes.](http://arxiv.org/abs/2307.10634) | 本研究开发了一种生成语言模型，用于处理人类基因的核苷酸序列，填补了DNA序列生成模型研究的空白。 |

# 详细

[^1]: GNN2R: 基于弱监督的知识图谱问答中提供理由的问题回答方法

    GNN2R: Weakly-Supervised Rationale-Providing Question Answering over Knowledge Graphs. (arXiv:2312.02317v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.02317](http://arxiv.org/abs/2312.02317)

    GNN2R是一种基于图神经网络的两步推理模型，通过弱监督训练，能够在知识图谱问答中提供最终答案以及推理子图的理由。该方法解决了现有方法缺乏解释以及效率低下的问题。

    

    目前大多数基于知识图谱的多跳问题回答方法只提供最终的确定答案，而没有解释，对于普通用户难以理解和查看的KG实体集。这严重限制了知识图谱问答在现实场景中的应用。本文提出了一种基于图神经网络的两步推理模型（GNN2R）来解决这个问题。GNN2R能够通过仅有的问题-最终答案对提供最终答案以及作为最终答案背后的推理子图的理由，且仅需要通过弱监督进行训练。我们对GNN2R进行了大量评估，并进行了详细的实验。

    Most current methods for multi-hop question answering (QA) over knowledge graphs (KGs) only provide final conclusive answers without explanations, such as a set of KG entities that is difficult for normal users to review and comprehend. This issue severely limits the application of KG-based QA in real-world scenarios. However, it is non-trivial to solve due to two challenges: First, annotations of reasoning chains of multi-hop questions, which could serve as supervision for explanation generation, are usually lacking. Second, it is difficult to maintain high efficiency when explicit KG triples need to be retrieved to generate explanations. In this paper, we propose a novel Graph Neural Network-based Two-Step Reasoning model (GNN2R) to solve this issue. GNN2R can provide both final answers and reasoning subgraphs as a rationale behind final answers efficiently with only weak supervision that is available through question-final answer pairs. We extensively evaluated GNN2R with detailed a
    
[^2]: 上下文化影响文本数字处理的语言资源水平的分类

    Contextualising Levels of Language Resourcedness affecting Digital Processing of Text. (arXiv:2309.17035v1 [cs.CL])

    [http://arxiv.org/abs/2309.17035](http://arxiv.org/abs/2309.17035)

    本文讨论了数字处理文本时，语言资源水平对其影响的上下文化分类，并提出了一个将语言划分为五个等级的矩阵，从而解决了将语言只划分为LRL和HRL两种类型的问题。

    

    应用领域如数字人文学和聊天机器人等工具都涉及到对自然语言的处理，从数字化纸质文件到语音生成。内容的语言通常被划分为资源匮乏语言（LRL）或资源丰富语言（HRL）。非洲语言被认为是资源匮乏语言，而英语则是资源最丰富的语言。为了为这些语言开发软件系统以完成各种任务，使用了各种语言资源。本文认为对于所有语言来说，将其划分为LRL和HRL两种对立的类型是有问题的。通过对社会中语言资源的清晰理解，我们开发了一个矩阵来将语言划分为"非常LRL"、"LRL"、"RL"、"HRL"和"非常HRL"。这种划分基于连接情景基础设施、情景对话流、情景知识等的分类。

    Application domains such as digital humanities and tool like chatbots involve some form of processing natural language, from digitising hardcopies to speech generation. The language of the content is typically characterised as either a low resource language (LRL) or high resource language (HRL), also known as resource-scarce and well-resourced languages, respectively. African languages have been characterized as resource-scarce languages (Bosch et al. 2007; Pretorius & Bosch 2003; Keet & Khumalo 2014) and English is by far the most well-resourced language. Varied language resources are used to develop software systems for these languages to accomplish a wide range of tasks. In this paper we argue that the dichotomous typology LRL and HRL for all languages is problematic. Through a clear understanding of language resources situated in a society, a matrix is developed that characterizes languages as Very LRL, LRL, RL, HRL and Very HRL. The characterization is based on the typology of con
    
[^3]: 人类基因核苷酸序列的生成语言模型

    Generative Language Models on Nucleotide Sequences of Human Genes. (arXiv:2307.10634v1 [q-bio.GN])

    [http://arxiv.org/abs/2307.10634](http://arxiv.org/abs/2307.10634)

    本研究开发了一种生成语言模型，用于处理人类基因的核苷酸序列，填补了DNA序列生成模型研究的空白。

    

    自然语言处理领域的语言模型，特别是基于Transformer的模型，取得了巨大的成功。然而，在DNA相关的生物信息学领域，生成模型的研究相对较少。因此，本研究旨在开发一种类似于GPT-3的自回归生成语言模型，用于处理人类基因的核苷酸序列。考虑到处理整个DNA序列需要大量计算资源，我们决定在更小的尺度上进行研究，重点关注人类基因的核苷酸序列，而不是整个DNA。这个决策并不改变问题的结构，因为DNA和基因都可以看作由四种不同的核苷酸组成的一维序列。

    Language models, primarily transformer-based ones, obtained colossal success in NLP. To be more precise, studies like BERT in NLU and works such as GPT-3 for NLG are very crucial. DNA sequences are very close to natural language in terms of structure, so if the DNA-related bioinformatics domain is concerned, discriminative models, like DNABert, exist. Yet, the generative side of the coin is mainly unexplored to the best of our knowledge. Consequently, we focused on developing an autoregressive generative language model like GPT-3 for DNA sequences. Because working with whole DNA sequences is challenging without substantial computational resources, we decided to carry out our study on a smaller scale, focusing on nucleotide sequences of human genes, unique parts in DNA with specific functionalities, instead of the whole DNA. This decision did not change the problem structure a lot due to the fact that both DNA and genes can be seen as 1D sequences consisting of four different nucleotide
    

