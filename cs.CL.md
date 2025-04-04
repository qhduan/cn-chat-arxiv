# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Much are LLMs Contaminated? A Comprehensive Survey and the LLMSanitize Library](https://arxiv.org/abs/2404.00699) | LLM受到污染可能导致其性能不可靠，挑战了自然语言处理领域的整体进展。 |
| [^2] | [Rethinking Optimization and Architecture for Tiny Language Models](https://arxiv.org/abs/2402.02791) | 本研究重新思考了微型语言模型的优化和架构，通过经验研究发现了在微型语言模型中特别有效的设计公式，并在多语种数据集上训练了高性能的微型语言模型。 |
| [^3] | [LEACE: Perfect linear concept erasure in closed form.](http://arxiv.org/abs/2306.03819) | 本文介绍了一种闭合形式的方法LEACE，可在删除指定特征的同时尽可能少地改变表示，并可证明防止所有线性分类器检测到概念。作者用“概念擦除”这一新方法将其应用于大型语言模型，在测量语言模型对词性的依赖性和减少BERT嵌入中的性别偏差任务中得出良好表现。 |

# 详细

[^1]: LLM受到多少污染？一项全面调查和LLMSanitize库

    How Much are LLMs Contaminated? A Comprehensive Survey and the LLMSanitize Library

    [https://arxiv.org/abs/2404.00699](https://arxiv.org/abs/2404.00699)

    LLM受到污染可能导致其性能不可靠，挑战了自然语言处理领域的整体进展。

    

    随着近年来大型语言模型（LLMs）的崛起，新的机会正在出现，但也带来了新的挑战，污染问题迅速变得至关重要。企业应用和人工智能筹款已经达到一定规模，流行的问答基准提高几个百分点可能意味着数百万美元，对模型的完整性施加了巨大压力。同时，追踪LLMs见过的数据变得越来越困难；对于像GPT-4和Claude-3这样的闭源模型，他们不透露任何有关训练集的信息。因此，污染成为一个关键问题：LLMs的性能可能不再可靠，因为其高性能至少部分归因于其先前接触到的数据。这种局限性危及了自然语言处理领域的整体进展，然而，如何有效解决这一问题仍然缺乏方法。

    arXiv:2404.00699v1 Announce Type: new  Abstract: With the rise of Large Language Models (LLMs) in recent years, new opportunities are emerging, but also new challenges, and contamination is quickly becoming critical. Business applications and fundraising in AI have reached a scale at which a few percentage points gained on popular question-answering benchmarks could translate into dozens of millions of dollars, placing high pressure on model integrity. At the same time, it is becoming harder and harder to keep track of the data that LLMs have seen; if not impossible with closed-source models like GPT-4 and Claude-3 not divulging any information on the training set. As a result, contamination becomes a critical issue: LLMs' performance may not be reliable anymore, as the high performance may be at least partly due to their previous exposure to the data. This limitation jeopardizes the entire progress in the field of NLP, yet, there remains a lack of methods on how to efficiently address
    
[^2]: 重新思考微型语言模型的优化和架构

    Rethinking Optimization and Architecture for Tiny Language Models

    [https://arxiv.org/abs/2402.02791](https://arxiv.org/abs/2402.02791)

    本研究重新思考了微型语言模型的优化和架构，通过经验研究发现了在微型语言模型中特别有效的设计公式，并在多语种数据集上训练了高性能的微型语言模型。

    

    大型语言模型（LLMs）的威力通过大量的数据和计算资源得到了证明。然而，在移动设备上应用语言模型面临着计算和内存成本的巨大挑战，迫切需要高性能的微型语言模型。受复杂训练过程的限制，优化语言模型的许多细节很少得到仔细研究。在本研究中，基于一个具有10亿参数的微型语言模型，我们仔细设计了一系列经验研究来分析每个组件的影响。主要讨论了三个方面，即神经架构、参数初始化和优化策略。多个设计公式在微型语言模型中经验性地被证明特别有效，包括分词器压缩、架构调整、参数继承和多轮训练。然后，我们在1.6T多语种数据集上训练了PanGu-$\pi$-1B Pro和PanGu-$\pi$-1.5B Pro。

    The power of large language models (LLMs) has been demonstrated through numerous data and computing resources. However, the application of language models on mobile devices is facing huge challenge on the computation and memory costs, that is, tiny language models with high performance are urgently required. Limited by the highly complex training process, there are many details for optimizing language models that are seldom studied carefully. In this study, based on a tiny language model with 1B parameters, we carefully design a series of empirical study to analyze the effect of each component. Three perspectives are mainly discussed, i.e., neural architecture, parameter initialization, and optimization strategy. Several design formulas are empirically proved especially effective for tiny language models, including tokenizer compression, architecture tweaking, parameter inheritance and multiple-round training. Then we train PanGu-$\pi$-1B Pro and PanGu-$\pi$-1.5B Pro on 1.6T multilingu
    
[^3]: LEACE：闭合形式中的完美线性概念擦除

    LEACE: Perfect linear concept erasure in closed form. (arXiv:2306.03819v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.03819](http://arxiv.org/abs/2306.03819)

    本文介绍了一种闭合形式的方法LEACE，可在删除指定特征的同时尽可能少地改变表示，并可证明防止所有线性分类器检测到概念。作者用“概念擦除”这一新方法将其应用于大型语言模型，在测量语言模型对词性的依赖性和减少BERT嵌入中的性别偏差任务中得出良好表现。

    

    概念擦除旨在从表征中删除指定的特征。它可以提高公平性（例如，防止分类器使用性别或种族）和可解释性（例如，删除概念以观察模型行为的变化）。我们引入了LEAst-squares概念擦除（LEACE），这是一种闭合形式的方法，可证明防止所有线性分类器检测到概念，同时尽可能地改变表示，如广泛类别的范数所测量的那样。我们使用名为“概念擦除”的新方法将LEACE应用于大型语言模型，擦除每个层中的目标概念信息。我们在两个任务上展示了我们的方法：测量语言模型对词性信息的依赖性，以及减少BERT嵌入中的性别偏差。代码可在https://github.com/EleutherAI/concept-erasure上找到。

    Concept erasure aims to remove specified features from a representation. It can improve fairness (e.g. preventing a classifier from using gender or race) and interpretability (e.g. removing a concept to observe changes in model behavior). We introduce LEAst-squares Concept Erasure (LEACE), a closed-form method which provably prevents all linear classifiers from detecting a concept while changing the representation as little as possible, as measured by a broad class of norms. We apply LEACE to large language models with a novel procedure called "concept scrubbing," which erases target concept information from every layer in the network. We demonstrate our method on two tasks: measuring the reliance of language models on part-of-speech information, and reducing gender bias in BERT embeddings. Code is available at https://github.com/EleutherAI/concept-erasure.
    

