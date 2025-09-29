# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GRAFFORD: A Benchmark Dataset for Testing the Knowledge of Object Affordances of Language and Vision Models](https://arxiv.org/abs/2402.12881) | 该论文提出了一个名为GRAFFORD的基准数据集，用于测试语言和视觉模型对物体可供性知识的表现，实验结果显示当前预训练语言模型在理解不常见物体可供性方面存在推理能力的局限。 |
| [^2] | [Constituency Parsing using LLMs.](http://arxiv.org/abs/2310.19462) | 本文研究了使用大型语言模型（LLMs）进行成分句法分析的潜力，通过采用线性化策略将输出树结构转化为符号序列，进一步提高了任务的效果。实验结果对LLMs的性能、泛化能力和成分句法分析中的挑战进行了深入研究。 |

# 详细

[^1]: GRAFFORD: 用于测试语言和视觉模型对物体可供性知识的基准数据集

    GRAFFORD: A Benchmark Dataset for Testing the Knowledge of Object Affordances of Language and Vision Models

    [https://arxiv.org/abs/2402.12881](https://arxiv.org/abs/2402.12881)

    该论文提出了一个名为GRAFFORD的基准数据集，用于测试语言和视觉模型对物体可供性知识的表现，实验结果显示当前预训练语言模型在理解不常见物体可供性方面存在推理能力的局限。

    

    我们调查了预训练语言模型（LMs）和预训练视觉-语言模型（VLMs）中关于物体可供性的知识。基于Transformer的大型预训练语言模型（PTLM）从大量未标记文本中学习上下文表示，并在下游NLU任务中表现出色。与此同时，越来越多的文献表明，PTLM在推理和基础方面存在不一致且不直观的失败。为了首次定量衡量基础（或缺乏）的影响，我们精心策划了一个关于物体可供性的新颖而全面的数据集-- GrAFFORD，包含15个可供性类别。与视觉和语言领域收集的可供性数据集不同，我们用现场句子标注了对象和可供性。实验结果显示，当涉及不常见的物体可供性时，PTLM表现出有限的推理能力。我们还观察到PTLM在理解不常见物体可供性时存在困难。

    arXiv:2402.12881v1 Announce Type: new  Abstract: We investigate the knowledge of object affordances in pre-trained language models (LMs) and pre-trained Vision-Language models (VLMs). Transformers-based large pre-trained language models (PTLM) learn contextual representation from massive amounts of unlabeled text and are shown to perform impressively in downstream NLU tasks. In parallel, a growing body of literature shows that PTLMs fail inconsistently and non-intuitively, showing a lack of reasoning and grounding. To take a first step toward quantifying the effect of grounding (or lack thereof), we curate a novel and comprehensive dataset of object affordances -- GrAFFORD, characterized by 15 affordance classes. Unlike affordance datasets collected in vision and language domains, we annotate in-the-wild sentences with objects and affordances. Experimental results reveal that PTLMs exhibit limited reasoning abilities when it comes to uncommon object affordances. We also observe that pr
    
[^2]: 使用大型语言模型进行成分句法分析

    Constituency Parsing using LLMs. (arXiv:2310.19462v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.19462](http://arxiv.org/abs/2310.19462)

    本文研究了使用大型语言模型（LLMs）进行成分句法分析的潜力，通过采用线性化策略将输出树结构转化为符号序列，进一步提高了任务的效果。实验结果对LLMs的性能、泛化能力和成分句法分析中的挑战进行了深入研究。

    

    成分句法分析是一个基础但尚未解决的自然语言处理任务。本文探索了最近大型语言模型（LLMs）在各个领域和任务中展现出的卓越性能在解决这一任务上的潜力。我们采用三种线性化策略将输出的树结构转化为符号序列，使得LLMs可以通过生成线性化树来解决成分句法分析。我们使用多种不同的LLMs进行实验，包括ChatGPT、GPT-4、OPT、LLaMA和Alpaca，并将它们的性能与最先进的成分句法分析器进行比较。我们的实验涵盖了零样本学习、少样本学习和全样本学习的不同设置，并在一个领域内和五个领域外的测试数据集上评估模型。我们的发现揭示了LLMs的性能、泛化能力和成分句法分析中的挑战。

    Constituency parsing is a fundamental yet unsolved natural language processing task. In this paper, we explore the potential of recent large language models (LLMs) that have exhibited remarkable performance across various domains and tasks to tackle this task. We employ three linearization strategies to transform output trees into symbol sequences, such that LLMs can solve constituency parsing by generating linearized trees. We conduct experiments using a diverse range of LLMs, including ChatGPT, GPT-4, OPT, LLaMA, and Alpaca, comparing their performance against the state-of-the-art constituency parsers. Our experiments encompass zero-shot, few-shot, and full-training learning settings, and we evaluate the models on one in-domain and five out-of-domain test datasets. Our findings reveal insights into LLMs' performance, generalization abilities, and challenges in constituency parsing.
    

