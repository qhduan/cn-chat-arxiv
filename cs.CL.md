# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RankMamba, Benchmarking Mamba's Document Ranking Performance in the Era of Transformers](https://arxiv.org/abs/2403.18276) | Mamba模型基于状态空间模型，在多个序列建模任务中取得了与Transformer相当的性能，并在经典信息检索任务--文档排名中展现了其有效性。 |
| [^2] | [GRACE: Discriminator-Guided Chain-of-Thought Reasoning.](http://arxiv.org/abs/2305.14934) | GRACE是一种判别器引导的思维链推理的逐步解码方法，通过使用一个正确性判别器来评分下一步候选，解决了语言模型在多步推理中容易得到错误答案的问题。在多个数学和符号推理任务中，GRACE相较于其他方法在性能上有明显的提升。 |

# 详细

[^1]: RankMamba，在Transformer时代对Mamba文档排名性能的基准测试

    RankMamba, Benchmarking Mamba's Document Ranking Performance in the Era of Transformers

    [https://arxiv.org/abs/2403.18276](https://arxiv.org/abs/2403.18276)

    Mamba模型基于状态空间模型，在多个序列建模任务中取得了与Transformer相当的性能，并在经典信息检索任务--文档排名中展现了其有效性。

    

    Transformer结构在自然语言处理（NLP）、计算机视觉（CV）和信息检索(IR)等多个应用的机器学习领域取得了巨大成功。Transformer架构的核心机制--注意力，在训练中需要$O(n^2)$的时间复杂度，在推断中需要$O(n)$的时间复杂度。许多工作已经提出改进注意力机制的可扩展性，比如Flash Attention和Multi-query Attention。另一方面的工作旨在设计新的机制来取代注意力。最近，基于状态空间模型的一个显著模型结构--Mamba，在多个序列建模任务中取得了与Transformer相当的性能。

    arXiv:2403.18276v1 Announce Type: cross  Abstract: Transformer structure has achieved great success in multiple applied machine learning communities, such as natural language processing (NLP), computer vision (CV) and information retrieval (IR). Transformer architecture's core mechanism -- attention requires $O(n^2)$ time complexity in training and $O(n)$ time complexity in inference. Many works have been proposed to improve the attention mechanism's scalability, such as Flash Attention and Multi-query Attention. A different line of work aims to design new mechanisms to replace attention. Recently, a notable model structure -- Mamba, which is based on state space models, has achieved transformer-equivalent performance in multiple sequence modeling tasks.   In this work, we examine \mamba's efficacy through the lens of a classical IR task -- document ranking. A reranker model takes a query and a document as input, and predicts a scalar relevance score. This task demands the language mod
    
[^2]: GRACE: 判别器引导的思维链推理

    GRACE: Discriminator-Guided Chain-of-Thought Reasoning. (arXiv:2305.14934v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.14934](http://arxiv.org/abs/2305.14934)

    GRACE是一种判别器引导的思维链推理的逐步解码方法，通过使用一个正确性判别器来评分下一步候选，解决了语言模型在多步推理中容易得到错误答案的问题。在多个数学和符号推理任务中，GRACE相较于其他方法在性能上有明显的提升。

    

    在多步推理的背景下，例如使用思维链，语言模型往往会对错误的步骤分配较高的可能性。因此，优化解决方案可能性的解码策略往往会产生错误的解决方案。为了解决这个问题，我们提出了一种称为GRACE的引导思维链推理的逐步解码方法，该方法通过一个正确性判别器训练来引导解码过程产生正确的推理步骤。GRACE使用一个在正确和错误步骤上进行对比损失训练的判别器，该判别器在解码过程中基于正确性对下一步候选进行评分。重要的是，GRACE只需要从语言模型中采样，而不需要进行语言模型的训练或微调。我们使用FLAN-T5和LLaMA系列的模型，对四个数学和两个符号推理任务进行了GRACE的评估，在大多数设置中，与贪婪解码、验证器和自一致性相比，GRACE展现出了显著的性能提升。

    In the context of multi-step reasoning, e.g., with chain-of-thought, language models (LMs) can easily assign a high likelihood to incorrect steps. As a result, decoding strategies that optimize for solution likelihood often yield incorrect solutions. To address this issue, we propose Guiding chain-of-thought ReAsoning with a CorrectnEss Discriminator (GRACE), a stepwise decoding approach that steers the decoding process towards producing correct reasoning steps. GRACE employs a discriminator trained with a contrastive loss over correct and incorrect steps, which is used during decoding to score next-step candidates based on their correctness. Importantly, GRACE only requires sampling from the LM, without the need for LM training or fine-tuning. Using models from FLAN-T5 and LLaMA families, we evaluate GRACE over four math and two symbolic reasoning tasks, where it exhibits substantial performance gains compared to greedy decoding, verifiers, and self-consistency in most settings. When 
    

