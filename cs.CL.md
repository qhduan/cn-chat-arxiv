# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Open Source Conversational LLMs do not know most Spanish words](https://arxiv.org/abs/2403.15491) | 本研究评估了开源对话式LLMs对西班牙语单词的了解程度，结果显示它们无法正确使用大部分单词写句子。 |
| [^2] | [FaaF: Facts as a Function for the evaluation of RAG systems](https://arxiv.org/abs/2403.03888) | FaaF是一种新的事实验证方法，利用语言模型的函数调用能力和面向RAG事实回忆评估的框架，显着提高了LM识别不支持事实的能力，并在效率和成本方面取得了明显的改进。 |
| [^3] | [HiQA: A Hierarchical Contextual Augmentation RAG for Massive Documents QA](https://arxiv.org/abs/2402.01767) | HiQA是一个先进的多文档问答框架，使用分层的上下文增强和多路径检索机制，解决了大规模文档问答中的检索准确性问题，并在多文档环境中展示了最先进的性能。 |
| [^4] | [SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding](https://arxiv.org/abs/2401.09340) | 本研究通过系统性地扩展室内环境中的3D视觉-语言学习，提出了首个百万规模的3D视觉-语言数据集SceneVerse，以解决3D视觉-语言对齐面临的几个重要挑战。 |
| [^5] | [C-Pack: Packaged Resources To Advance General Chinese Embedding.](http://arxiv.org/abs/2309.07597) | C-Pack是一套推进普通汉语嵌入领域的资源，包括全面汉语文本嵌入基准、大规模文本嵌入数据集和涵盖多个尺寸的嵌入模型系列。该资源集在C-MTEB基准上实现了最高+10%的表现，并通过整合和优化一套训练方法进一步提升了效果。此外，C-Pack还发布了英语文本嵌入数据和模型，实现了最先进的性能。该资源集可公开获取。 |

# 详细

[^1]: 开源对话式LLMs不了解大部分西班牙语单词

    Open Source Conversational LLMs do not know most Spanish words

    [https://arxiv.org/abs/2403.15491](https://arxiv.org/abs/2403.15491)

    本研究评估了开源对话式LLMs对西班牙语单词的了解程度，结果显示它们无法正确使用大部分单词写句子。

    

    在对大型语言模型（LLMs），特别是可以与用户进行交互的对话模型的兴趣不断增加的情况下，开发了大量开源聊天LLMs。这些模型在广泛的基准上进行评估，以评估它们在回答问题或解决几乎任何可能的主题上的能力，或者测试它们理解或解释文本的能力。然而，对这些模型对语言的了解的评估却受到了远远较少的关注。例如，它们能够识别和使用不同语言中的单词数。本文通过在参考词典中测试部分单词的样本来评估开源聊天LLMs对西班牙语单词的了解程度。结果显示，开源聊天LLMs对重要部分单词产生了错误的含义，并且无法正确使用大部分单词在上下文中写句子。

    arXiv:2403.15491v1 Announce Type: new  Abstract: The growing interest in Large Language Models (LLMs) and in particular in conversational models with which users can interact has led to the development of a large number of open-source chat LLMs. These models are evaluated on a wide range of benchmarks to assess their capabilities in answering questions or solving problems on almost any possible topic or to test their ability to reason or interpret texts. Instead, the evaluation of the knowledge that these models have of the languages has received much less attention. For example, the words that they can recognize and use in different languages. In this paper, we evaluate the knowledge that open-source chat LLMs have of Spanish words by testing a sample of words in a reference dictionary. The results show that open-source chat LLMs produce incorrect meanings for an important fraction of the words and are not able to use most of the words correctly to write sentences with context. These 
    
[^2]: FaaF：作为RAG系统评估的事实函数

    FaaF: Facts as a Function for the evaluation of RAG systems

    [https://arxiv.org/abs/2403.03888](https://arxiv.org/abs/2403.03888)

    FaaF是一种新的事实验证方法，利用语言模型的函数调用能力和面向RAG事实回忆评估的框架，显着提高了LM识别不支持事实的能力，并在效率和成本方面取得了明显的改进。

    

    从参考资料中准确提取事实对于评估检索增强生成（RAG）系统的性能至关重要，因为它直接探查了检索和生成的质量。然而，可靠高效地执行这种评估仍然是一个挑战。最近的工作侧重于通过提示语言模型（LM）评估器进行事实验证，然而我们证明，在信息不完整或不准确的情况下，这些方法是不可靠的。我们引入了FaaF（Facts as a Function），这是一种利用LM的函数调用能力和面向RAG事实回忆评估的框架的新方法。与基于提示的方法相比，FaaF显着提高了LM识别文本中不支持事实的能力，同时提高了效率，降低了成本几倍。

    arXiv:2403.03888v1 Announce Type: new  Abstract: Factual recall from a reference source is crucial for evaluating the performance of Retrieval Augmented Generation (RAG) systems, as it directly probes into the quality of both retrieval and generation. However, it still remains a challenge to perform this evaluation reliably and efficiently. Recent work has focused on fact verification via prompting language model (LM) evaluators, however we demonstrate that these methods are unreliable in the presence of incomplete or inaccurate information. We introduce Facts as a Function (FaaF), a new approach to fact verification that utilizes the function calling abilities of LMs and a framework for RAG factual recall evaluation. FaaF substantially improves the ability of LMs to identify unsupported facts in text with incomplete information whilst improving efficiency and lowering cost by several times, compared to prompt-based approaches.
    
[^3]: HiQA：一种用于大规模文档问答的分层上下文增强的RAG模型

    HiQA: A Hierarchical Contextual Augmentation RAG for Massive Documents QA

    [https://arxiv.org/abs/2402.01767](https://arxiv.org/abs/2402.01767)

    HiQA是一个先进的多文档问答框架，使用分层的上下文增强和多路径检索机制，解决了大规模文档问答中的检索准确性问题，并在多文档环境中展示了最先进的性能。

    

    随着利用外部工具的语言模型代理迅速发展，使用补充文档和检索增强生成（RAG）方法的问答（QA）方法学取得了重要进展。这种进步提高了语言模型的回答质量，并减轻了幻觉的出现。然而，当面临大量无法区分的文档时，这些方法在检索准确性方面表现有限，给实际应用带来了显著挑战。针对这些新兴的挑战，我们提出了HiQA，这是一个先进的多文档问答（MDQA）框架，将级联的元数据整合到内容中，同时具备多路径检索机制。我们还发布了一个名为MasQA的基准来评估和研究MDQA。最后，HiQA在多文档环境中展示了最先进的性能。

    As language model agents leveraging external tools rapidly evolve, significant progress has been made in question-answering(QA) methodologies utilizing supplementary documents and the Retrieval-Augmented Generation (RAG) approach. This advancement has improved the response quality of language models and alleviates the appearance of hallucination. However, these methods exhibit limited retrieval accuracy when faced with massive indistinguishable documents, presenting notable challenges in their practical application. In response to these emerging challenges, we present HiQA, an advanced framework for multi-document question-answering (MDQA) that integrates cascading metadata into content as well as a multi-route retrieval mechanism. We also release a benchmark called MasQA to evaluate and research in MDQA. Finally, HiQA demonstrates the state-of-the-art performance in multi-document environments.
    
[^4]: SceneVerse：为基于场景的场景理解扩展3D视觉-语言学习

    SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding

    [https://arxiv.org/abs/2401.09340](https://arxiv.org/abs/2401.09340)

    本研究通过系统性地扩展室内环境中的3D视觉-语言学习，提出了首个百万规模的3D视觉-语言数据集SceneVerse，以解决3D视觉-语言对齐面临的几个重要挑战。

    

    3D视觉-语言对齐，即将语言与3D物理环境对齐，是发展具身体能力的智能体的基石。与2D领域最近的进展相比，将语言与3D场景对齐面临着几个重要挑战：（i）3D场景固有复杂性，由于多样的物体配置、丰富的属性和错综复杂的关系；（ii）支持基于场景学习的配对3D视觉-语言数据的稀缺性；以及（iii）缺乏从基于场景的3D数据中提炼知识的统一学习框架。在这项工作中，我们旨在通过系统地扩展室内环境中的3D视觉-语言学习，从而解决3D视觉-语言领域中的这三大挑战。我们介绍首个百万规模的3D视觉-语言数据集SceneVerse，包含约68K个3D室内场景，包括250万个视觉语言

    arXiv:2401.09340v2 Announce Type: replace-cross  Abstract: 3D vision-language grounding, which focuses on aligning language with the 3D physical environment, stands as a cornerstone in the development of embodied agents. In comparison to recent advancements in the 2D domain, grounding language in 3D scenes faces several significant challenges: (i) the inherent complexity of 3D scenes due to the diverse object configurations, their rich attributes, and intricate relationships; (ii) the scarcity of paired 3D vision-language data to support grounded learning; and (iii) the absence of a unified learning framework to distill knowledge from grounded 3D data. In this work, we aim to address these three major challenges in 3D vision-language by examining the potential of systematically upscaling 3D vision-language learning in indoor environments. We introduce the first million-scale 3D vision-language dataset, SceneVerse, encompassing about 68K 3D indoor scenes and comprising 2.5M vision-langu
    
[^5]: C-Pack: 推进普通汉语嵌入的打包资源

    C-Pack: Packaged Resources To Advance General Chinese Embedding. (arXiv:2309.07597v1 [cs.CL])

    [http://arxiv.org/abs/2309.07597](http://arxiv.org/abs/2309.07597)

    C-Pack是一套推进普通汉语嵌入领域的资源，包括全面汉语文本嵌入基准、大规模文本嵌入数据集和涵盖多个尺寸的嵌入模型系列。该资源集在C-MTEB基准上实现了最高+10%的表现，并通过整合和优化一套训练方法进一步提升了效果。此外，C-Pack还发布了英语文本嵌入数据和模型，实现了最先进的性能。该资源集可公开获取。

    

    我们介绍了C-Pack，这是一套显著推进普通汉语嵌入领域的资源。C-Pack包括三个关键资源。1）C-MTEB是一个涵盖6个任务和35个数据集的全面汉语文本嵌入基准。2）C-MTP是一个从标记和未标记的汉语语料库中策划的大规模文本嵌入数据集，用于训练嵌入模型。3）C-TEM是一个涵盖多个尺寸的嵌入模型系列。我们的模型在C-MTEB上的表现优于之前的所有汉语文本嵌入达到了发布时的最高+10%。我们还整合和优化了C-TEM的整套训练方法。除了我们关于普通汉语嵌入的资源外，我们还发布了我们的英语文本嵌入数据和模型。这些英语模型在MTEB基准上实现了最先进的性能；与此同时，我们发布的英语数据比汉语数据大2倍。所有这些资源都可以在https://github.com/FlagOpen/FlagEmbedding上公开获取。

    We introduce C-Pack, a package of resources that significantly advance the field of general Chinese embeddings. C-Pack includes three critical resources. 1) C-MTEB is a comprehensive benchmark for Chinese text embeddings covering 6 tasks and 35 datasets. 2) C-MTP is a massive text embedding dataset curated from labeled and unlabeled Chinese corpora for training embedding models. 3) C-TEM is a family of embedding models covering multiple sizes. Our models outperform all prior Chinese text embeddings on C-MTEB by up to +10% upon the time of the release. We also integrate and optimize the entire suite of training methods for C-TEM. Along with our resources on general Chinese embedding, we release our data and models for English text embeddings. The English models achieve state-of-the-art performance on MTEB benchmark; meanwhile, our released English data is 2 times larger than the Chinese data. All these resources are made publicly available at https://github.com/FlagOpen/FlagEmbedding.
    

