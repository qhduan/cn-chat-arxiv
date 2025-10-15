# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Jailbreaking Proprietary Large Language Models using Word Substitution Cipher](https://arxiv.org/abs/2402.10601) | 本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。 |
| [^2] | [GRDD: A Dataset for Greek Dialectal NLP.](http://arxiv.org/abs/2308.00802) | 本文介绍了一个用于研究现代希腊方言的大规模数据集GRDD，并使用该数据集进行方言识别实验，结果显示即使是简单的机器学习模型也能在该任务上表现良好。 |

# 详细

[^1]: 使用单词替换密码来越狱专有的大型语言模型

    Jailbreaking Proprietary Large Language Models using Word Substitution Cipher

    [https://arxiv.org/abs/2402.10601](https://arxiv.org/abs/2402.10601)

    本文使用密码技术编码了越狱提示，成功地绕过了大型语言模型对有害问题的检测，实验结果显示攻击成功率高达59.42%。

    

    大型语言模型（LLMs）遵循道德和伦理准则，但仍然容易受到名为Jailbreak的创意提示的影响，这些提示可以绕过对齐过程。然而，大多数越狱提示包含自然语言（主要是英语）中的有害问题，可以被LLMs自身检测到。本文提出了使用密码技术编码的越狱提示。我们首先在最先进的LLM，GPT-4上进行了一个试点研究，解码了使用各种密码技术加密的几个安全句子，发现简单的单词替换密码可以被最有效地解码。受此结果启发，我们使用这种编码技术来编写越狱提示。我们提供了将不安全单词映射到安全单词，并使用这些映射的单词提出不安全问题的映射。实验结果显示，我们提出的越狱攻击成功率（高达59.42%）。

    arXiv:2402.10601v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are aligned to moral and ethical guidelines but remain susceptible to creative prompts called Jailbreak that can bypass the alignment process. However, most jailbreaking prompts contain harmful questions in the natural language (mainly English), which can be detected by the LLM themselves. In this paper, we present jailbreaking prompts encoded using cryptographic techniques. We first present a pilot study on the state-of-the-art LLM, GPT-4, in decoding several safe sentences that have been encrypted using various cryptographic techniques and find that a straightforward word substitution cipher can be decoded most effectively. Motivated by this result, we use this encoding technique for writing jailbreaking prompts. We present a mapping of unsafe words with safe words and ask the unsafe question using these mapped words. Experimental results show an attack success rate (up to 59.42%) of our proposed jailbrea
    
[^2]: GRDD: 希腊方言自然语言处理的数据集

    GRDD: A Dataset for Greek Dialectal NLP. (arXiv:2308.00802v1 [cs.CL])

    [http://arxiv.org/abs/2308.00802](http://arxiv.org/abs/2308.00802)

    本文介绍了一个用于研究现代希腊方言的大规模数据集GRDD，并使用该数据集进行方言识别实验，结果显示即使是简单的机器学习模型也能在该任务上表现良好。

    

    本文介绍了一个用于研究现代希腊方言的数据集。该数据集包含了克里特、庞提、北希腊和塞浦路斯希腊四种方言的原始文本数据。尽管存在不平衡，但该数据集是相当大的，并且是创建现代希腊方言类似资源的首次尝试。我们还使用该数据集进行方言识别，并尝试了传统的机器学习算法和简单的深度学习架构。结果显示，在这个任务上表现非常好，这可能表明所研究的方言具有足够的独特特征，即使是简单的机器学习模型也能在该任务上表现良好。针对表现最佳的算法进行了错误分析，结果显示在一些情况下错误是由于数据集清理不足造成的。

    In this paper, we present a dataset for the computational study of a number of Modern Greek dialects. It consists of raw text data from four dialects of Modern Greek, Cretan, Pontic, Northern Greek and Cypriot Greek. The dataset is of considerable size, albeit imbalanced, and presents the first attempt to create large scale dialectal resources of this type for Modern Greek dialects. We then use the dataset to perform dialect idefntification. We experiment with traditional ML algorithms, as well as simple DL architectures. The results show very good performance on the task, potentially revealing that the dialects in question have distinct enough characteristics allowing even simple ML models to perform well on the task. Error analysis is performed for the top performing algorithms showing that in a number of cases the errors are due to insufficient dataset cleaning.
    

