# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benchmarking Large Language Models in Complex Question Answering Attribution using Knowledge Graphs.](http://arxiv.org/abs/2401.14640) | 该研究介绍了一种用于评估问题回答归因的新方法，并通过对大型语言模型进行基准测试，发现现有的评估器在细粒度的归因设置下表现不佳，同时在复杂的引文-陈述推理中也存在弱点。 |

# 详细

[^1]: 在复杂问题回答归因中使用知识图谱对大型语言模型进行基准测试

    Benchmarking Large Language Models in Complex Question Answering Attribution using Knowledge Graphs. (arXiv:2401.14640v1 [cs.CL])

    [http://arxiv.org/abs/2401.14640](http://arxiv.org/abs/2401.14640)

    该研究介绍了一种用于评估问题回答归因的新方法，并通过对大型语言模型进行基准测试，发现现有的评估器在细粒度的归因设置下表现不佳，同时在复杂的引文-陈述推理中也存在弱点。

    

    问题回答的归因是为生成的陈述提供引用, 并且引起了广泛的研究关注。目前的自动评估归因的方法往往基于大型语言模型(LLM), 但仍然不足, 特别是在识别归因之间细微差别和引用与陈述之间的复杂关系方面。为了比较这些归因评估方法并开发新的方法, 我们引入了一组细粒度的类别(即支持, 不足, 矛盾和无关), 用于衡量归因, 并通过利用知识图谱(KG)为问题-回答对自动生成不同类别的归因, 开发了一个复杂的归因问题回答(CAQA)基准。我们的分析显示, 现有的评估器在细粒度的归因设置下表现不佳, 在复杂的引文-陈述推理中存在弱点。

    The attribution of question answering is to provide citations for supporting generated statements, and has attracted wide research attention. The current methods for automatically evaluating the attribution, which are often based on Large Language Models (LLMs), are still inadequate, particularly in recognizing subtle differences between attributions, and complex relationships between citations and statements. To compare these attribution evaluation methods and develop new ones, we introduce a set of fine-grained categories (i.e., supportive, insufficient, contradictory and irrelevant) for measuring the attribution, and develop a Complex Attributed Question Answering (CAQA) benchmark by leveraging knowledge graphs (KGs) for automatically generating attributions of different categories to question-answer pairs. Our analysis reveals that existing evaluators perform poorly under fine-grained attribution settings and exhibit weaknesses in complex citation-statement reasoning. Our CAQA benc
    

