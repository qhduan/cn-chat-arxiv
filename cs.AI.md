# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Causal Graph Discovery Using Large Language Models](https://rss.arxiv.org/abs/2402.01207) | 提出了一个新的框架，利用大型语言模型进行高效的因果图发现，采用了广度优先搜索方法，只需要线性数量的查询，同时能轻松结合观察数据以提高性能，具有高效性和数据效率，并在真实因果图上取得了最先进的结果，展示了其在不同领域的广泛适用性潜力。 |
| [^2] | [Reanalyzing L2 Preposition Learning with Bayesian Mixed Effects and a Pretrained Language Model.](http://arxiv.org/abs/2302.08150) | 该论文使用贝叶斯和神经模型分析了中国学习者对英语介词的理解，重要的结果是发现了学生能力、任务类型和刺激句子之间关键的交互。 |

# 详细

[^1]: 使用大型语言模型的高效因果图发现

    Efficient Causal Graph Discovery Using Large Language Models

    [https://rss.arxiv.org/abs/2402.01207](https://rss.arxiv.org/abs/2402.01207)

    提出了一个新的框架，利用大型语言模型进行高效的因果图发现，采用了广度优先搜索方法，只需要线性数量的查询，同时能轻松结合观察数据以提高性能，具有高效性和数据效率，并在真实因果图上取得了最先进的结果，展示了其在不同领域的广泛适用性潜力。

    

    我们提出了一个新的框架，利用LLMs进行完整的因果图发现。之前基于LLM的方法采用了成对查询的方法，但这需要二次查询的数量，对于较大的因果图来说很快变得不可行。相反，提出的框架采用了广度优先搜索（BFS）的方法，只需要线性数量的查询。我们还展示了当有所观察数据可用时，提出的方法可以轻松地进行结合以提高性能。除了更具时间和数据效率外，提出的框架在不同大小的真实因果图上取得了最先进的结果。结果证明了提出方法在发现因果关系方面的有效性和效率，展示了其在不同领域的因果图发现任务中的广泛适用性潜力。

    We propose a novel framework that leverages LLMs for full causal graph discovery. While previous LLM-based methods have used a pairwise query approach, this requires a quadratic number of queries which quickly becomes impractical for larger causal graphs. In contrast, the proposed framework uses a breadth-first search (BFS) approach which allows it to use only a linear number of queries. We also show that the proposed method can easily incorporate observational data when available, to improve performance. In addition to being more time and data-efficient, the proposed framework achieves state-of-the-art results on real-world causal graphs of varying sizes. The results demonstrate the effectiveness and efficiency of the proposed method in discovering causal relationships, showcasing its potential for broad applicability in causal graph discovery tasks across different domains.
    
[^2]: 用贝叶斯混合效应和预训练语言模型重新分析L2介词学习

    Reanalyzing L2 Preposition Learning with Bayesian Mixed Effects and a Pretrained Language Model. (arXiv:2302.08150v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.08150](http://arxiv.org/abs/2302.08150)

    该论文使用贝叶斯和神经模型分析了中国学习者对英语介词的理解，重要的结果是发现了学生能力、任务类型和刺激句子之间关键的交互。

    

    我们使用贝叶斯和神经模型来分析中国学习者接受两个测试的前后反应数据，测试他们对英语介词的理解。结果大多数重复了之前基于频率分析的发现，并新发现了学生能力、任务类型和刺激句子之间关键的交互。鉴于数据的稀疏性和学习者的高度多样性，贝叶斯方法最为有用；但我们也看到了使用语言模型概率作为语法和可学性预测器的潜力。

    We use both Bayesian and neural models to dissect a data set of Chinese learners' pre- and post-interventional responses to two tests measuring their understanding of English prepositions. The results mostly replicate previous findings from frequentist analyses and newly reveal crucial interactions between student ability, task type, and stimulus sentence. Given the sparsity of the data as well as high diversity among learners, the Bayesian method proves most useful; but we also see potential in using language model probabilities as predictors of grammaticality and learnability.
    

