# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [EBBS: An Ensemble with Bi-Level Beam Search for Zero-Shot Machine Translation](https://arxiv.org/abs/2403.00144) | 提出了一种集成方法EBBS，配合新颖的双层束搜索算法，能够优于直接和通过第三语言进行的翻译，并实现知识蒸馏来提高推理效率。 |
| [^2] | [Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality](https://arxiv.org/abs/2402.13954) | 本文通过提出的代理函数在迭代屏蔽实验中评估了转换器模型所编码的社会偏见，并比较了其与其他评估方法的偏见估计，发现转换器模型中存在相对较高的宗教和残疾偏见，而性别偏见则相对较低。 |
| [^3] | [Quantitative knowledge retrieval from large language models](https://arxiv.org/abs/2402.07770) | 本文探讨了大型语言模型（LLMs）作为定量知识检索的可行性，以辅助数据分析任务。提出了一个提示工程框架，将LLMs作为科学文献潜在空间的接口。讨论了使用LLMs作为“专家”的影响和挑战。 |
| [^4] | [Preserving Knowledge Invariance: Rethinking Robustness Evaluation of Open Information Extraction.](http://arxiv.org/abs/2305.13981) | 本文提出了第一个模拟评估开放式信息提取模型在真实世界中的基准测试，并通过判断模型在整个团体上的表现是否始终准确来评估模型的鲁棒性。 |

# 详细

[^1]: EBBS: 一个具有双层束搜索的集成方法用于零翻译机器翻译

    EBBS: An Ensemble with Bi-Level Beam Search for Zero-Shot Machine Translation

    [https://arxiv.org/abs/2403.00144](https://arxiv.org/abs/2403.00144)

    提出了一种集成方法EBBS，配合新颖的双层束搜索算法，能够优于直接和通过第三语言进行的翻译，并实现知识蒸馏来提高推理效率。

    

    当我们用特定的翻译方向训练多语言模型时，零翻译的能力就会出现；模型可以直接在未见过的方向进行翻译。另外，零翻译也可以通过第三种语言（例如英语）来实现。在我们的工作中，我们发现直接和通过第三种语言进行的翻译都存在噪音，并且表现不尽如人意。我们提出了EBBS，一个具有新颖的双层束搜索算法的集成方法，其中每个集成组件在下层逐步探索自己的预测，但它们通过上层的“软投票”机制进行同步。在两个流行的多语言翻译数据集上的结果表明，EBBS始终优于直接和通过第三种语言进行的翻译，以及现有的集成技术。此外，我们可以将集成的知识传回到多语言模型中，以提高推理效率；值得注意的是，我们的E

    arXiv:2403.00144v1 Announce Type: cross  Abstract: The ability of zero-shot translation emerges when we train a multilingual model with certain translation directions; the model can then directly translate in unseen directions. Alternatively, zero-shot translation can be accomplished by pivoting through a third language (e.g., English). In our work, we observe that both direct and pivot translations are noisy and achieve less satisfactory performance. We propose EBBS, an ensemble method with a novel bi-level beam search algorithm, where each ensemble component explores its own prediction step by step at the lower level but they are synchronized by a "soft voting" mechanism at the upper level. Results on two popular multilingual translation datasets show that EBBS consistently outperforms direct and pivot translations as well as existing ensemble techniques. Further, we can distill the ensemble's knowledge back to the multilingual model to improve inference efficiency; profoundly, our E
    
[^2]: 通过预测质量间接测量掩盖语言模型中的社会偏见

    Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality

    [https://arxiv.org/abs/2402.13954](https://arxiv.org/abs/2402.13954)

    本文通过提出的代理函数在迭代屏蔽实验中评估了转换器模型所编码的社会偏见，并比较了其与其他评估方法的偏见估计，发现转换器模型中存在相对较高的宗教和残疾偏见，而性别偏见则相对较低。

    

    社会和政治科学家经常旨在从文本数据表示（嵌入）中发现和衡量不同的偏见。创新的基于转换器的语言模型生成具有上下文感知的令牌嵌入，并在各种自然语言任务中取得了最先进的性能，但已被证明在下游应用中编码了不需要的偏见。本文通过提出的代理函数在迭代屏蔽实验中评估由训练有遮蔽语言建模目标的转换器所编码的社会偏见，以测量转换器模型预测质量，并评估MLM对不利群体和有利群体的偏好。我们比较使用两个基准数据集的偏见估计与其他评估方法产生的偏见，发现考虑的MLMs中存在相对较高的宗教和残疾偏见，而相对于另一个数据集，一个数据集中存在较低的性别偏见。

    arXiv:2402.13954v1 Announce Type: new  Abstract: Social and political scientists often aim to discover and measure distinct biases from text data representations (embeddings). Innovative transformer-based language models produce contextually-aware token embeddings and have achieved state-of-the-art performance for a variety of natural language tasks, but have been shown to encode unwanted biases for downstream applications. In this paper, we evaluate the social biases encoded by transformers trained with the masked language modeling objective using proposed proxy functions within an iterative masking experiment to measure the quality of transformer models' predictions, and assess the preference of MLMs towards disadvantaged and advantaged groups. We compare bias estimations with those produced by other evaluation methods using two benchmark datasets, finding relatively high religious and disability biases across considered MLMs and low gender bias in one dataset relative to the other. 
    
[^3]: 大型语言模型中的定量知识检索

    Quantitative knowledge retrieval from large language models

    [https://arxiv.org/abs/2402.07770](https://arxiv.org/abs/2402.07770)

    本文探讨了大型语言模型（LLMs）作为定量知识检索的可行性，以辅助数据分析任务。提出了一个提示工程框架，将LLMs作为科学文献潜在空间的接口。讨论了使用LLMs作为“专家”的影响和挑战。

    

    大型语言模型（LLM）因其生成具有说服力的自然语言序列的能力而被广泛研究，但其作为定量信息检索的实用性尚不明确。本文探讨了将LLMs作为定量知识检索机制的可行性，以帮助数据分析任务，如贝叶斯模型的先验分布引导和缺失数据的填补。我们提出了一个提示工程框架，将LLMs视为科学文献潜在空间的接口，在不同上下文和领域中比较响应与更成熟的方法。讨论了使用LLMs作为“专家”的影响和挑战。

    Large language models (LLMs) have been extensively studied for their abilities to generate convincing natural language sequences, however their utility for quantitative information retrieval is less well understood. In this paper we explore the feasibility of LLMs as a mechanism for quantitative knowledge retrieval to aid data analysis tasks such as elicitation of prior distributions for Bayesian models and imputation of missing data. We present a prompt engineering framework, treating an LLM as an interface to a latent space of scientific literature, comparing responses in different contexts and domains against more established approaches. Implications and challenges of using LLMs as 'experts' are discussed.
    
[^4]: 保持知识不变性：重新思考开放信息抽取的鲁棒性评估

    Preserving Knowledge Invariance: Rethinking Robustness Evaluation of Open Information Extraction. (arXiv:2305.13981v1 [cs.CL])

    [http://arxiv.org/abs/2305.13981](http://arxiv.org/abs/2305.13981)

    本文提出了第一个模拟评估开放式信息提取模型在真实世界中的基准测试，并通过判断模型在整个团体上的表现是否始终准确来评估模型的鲁棒性。

    

    鲁棒性是确保自然语言处理模型能够成功应用于现实世界中的关键因素，特别是对于信息抽取任务而言。然而，大多数先前的评估基准都专注于验证配对匹配的正确性，忽略了关键的鲁棒性测量。在本文中，我们提出了第一个基准测试，模拟在真实世界中评估开放式信息提取模型的情况，其中同一知识含义的句法和表达分布会各不相同。我们设计和注释了一个大规模的测试平台，其中每个示例都是一个知识不变的团体，由具有相同含义但结构不同的句子组成。通过进一步阐述鲁棒性指标，当模型在整个团体上的表现始终准确时，被判定为鲁棒性强。我们对过去十年中发表的几种典型模型进行了实验。

    The robustness to distribution changes ensures that NLP models can be successfully applied in the realistic world, especially for information extraction tasks. However, most prior evaluation benchmarks have been devoted to validating pairwise matching correctness, ignoring the crucial measurement of robustness. In this paper, we present the first benchmark that simulates the evaluation of open information extraction models in the real world, where the syntactic and expressive distributions under the same knowledge meaning may drift variously. We design and annotate a large-scale testbed in which each example is a knowledge-invariant clique that consists of sentences with structured knowledge of the same meaning but with different syntactic and expressive forms. By further elaborating the robustness metric, a model is judged to be robust if its performance is consistently accurate on the overall cliques. We perform experiments on typical models published in the last decade as well as a 
    

