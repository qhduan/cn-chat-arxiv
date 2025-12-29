# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generative Language Models on Nucleotide Sequences of Human Genes.](http://arxiv.org/abs/2307.10634) | 本研究开发了一种生成语言模型，用于处理人类基因的核苷酸序列，填补了DNA序列生成模型研究的空白。 |

# 详细

[^1]: 人类基因核苷酸序列的生成语言模型

    Generative Language Models on Nucleotide Sequences of Human Genes. (arXiv:2307.10634v1 [q-bio.GN])

    [http://arxiv.org/abs/2307.10634](http://arxiv.org/abs/2307.10634)

    本研究开发了一种生成语言模型，用于处理人类基因的核苷酸序列，填补了DNA序列生成模型研究的空白。

    

    自然语言处理领域的语言模型，特别是基于Transformer的模型，取得了巨大的成功。然而，在DNA相关的生物信息学领域，生成模型的研究相对较少。因此，本研究旨在开发一种类似于GPT-3的自回归生成语言模型，用于处理人类基因的核苷酸序列。考虑到处理整个DNA序列需要大量计算资源，我们决定在更小的尺度上进行研究，重点关注人类基因的核苷酸序列，而不是整个DNA。这个决策并不改变问题的结构，因为DNA和基因都可以看作由四种不同的核苷酸组成的一维序列。

    Language models, primarily transformer-based ones, obtained colossal success in NLP. To be more precise, studies like BERT in NLU and works such as GPT-3 for NLG are very crucial. DNA sequences are very close to natural language in terms of structure, so if the DNA-related bioinformatics domain is concerned, discriminative models, like DNABert, exist. Yet, the generative side of the coin is mainly unexplored to the best of our knowledge. Consequently, we focused on developing an autoregressive generative language model like GPT-3 for DNA sequences. Because working with whole DNA sequences is challenging without substantial computational resources, we decided to carry out our study on a smaller scale, focusing on nucleotide sequences of human genes, unique parts in DNA with specific functionalities, instead of the whole DNA. This decision did not change the problem structure a lot due to the fact that both DNA and genes can be seen as 1D sequences consisting of four different nucleotide
    

