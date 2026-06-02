# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoEval Done Right: Using Synthetic Data for Model Evaluation](https://arxiv.org/abs/2403.07008) | 提出了用合成数据进行模型评估的方法，通过高效和统计上合理的算法，在GPT-4实验中有效的人工标记样本大小增加了50%。 |
| [^2] | [Malaysian English News Decoded: A Linguistic Resource for Named Entity and Relation Extraction](https://arxiv.org/abs/2402.14521) | 通过构建一个含有实体和关系标注的马来西亚英语新闻数据集，并对spaCy NER工具进行微调，本研究显著改善了马来西亚英语中NER的性能。 |

# 详细

[^1]: 自动评价正确: 使用合成数据进行模型评估

    AutoEval Done Right: Using Synthetic Data for Model Evaluation

    [https://arxiv.org/abs/2403.07008](https://arxiv.org/abs/2403.07008)

    提出了用合成数据进行模型评估的方法，通过高效和统计上合理的算法，在GPT-4实验中有效的人工标记样本大小增加了50%。

    

    机器学习模型的评估使用人工标记的验证数据可能既昂贵又耗时。可以使用AI标记的合成数据来减少此类目的人工注释数量，这一过程称为自动评估。我们提出了用于此目的的高效和统计上合理的算法，可以提高样本效率，同时保持不偏。这些算法在与GPT-4进行的实验中将有效的人工标记样本大小增加了高达50%。

    arXiv:2403.07008v1 Announce Type: cross  Abstract: The evaluation of machine learning models using human-labeled validation data can be expensive and time-consuming. AI-labeled synthetic data can be used to decrease the number of human annotations required for this purpose in a process called autoevaluation. We suggest efficient and statistically principled algorithms for this purpose that improve sample efficiency while remaining unbiased. These algorithms increase the effective human-labeled sample size by up to 50% on experiments with GPT-4.
    
[^2]: 马来西亚英语新闻解析：一个用于命名实体和关系抽取的语言资源

    Malaysian English News Decoded: A Linguistic Resource for Named Entity and Relation Extraction

    [https://arxiv.org/abs/2402.14521](https://arxiv.org/abs/2402.14521)

    通过构建一个含有实体和关系标注的马来西亚英语新闻数据集，并对spaCy NER工具进行微调，本研究显著改善了马来西亚英语中NER的性能。

    

    标准英语和马来西亚英语存在明显差异，在马来西亚英语的自然语言处理（NLP）任务中存在挑战。本文介绍了一个包含200篇新闻文章的马来西亚英语新闻（MEN）数据集，手动对实体和关系进行了标注，并通过对spaCy NER工具进行微调验证了针对马来西亚英语定制的数据集可以显著提高NER在马来西亚英语中的性能。

    arXiv:2402.14521v1 Announce Type: new  Abstract: Standard English and Malaysian English exhibit notable differences, posing challenges for natural language processing (NLP) tasks on Malaysian English. Unfortunately, most of the existing datasets are mainly based on standard English and therefore inadequate for improving NLP tasks in Malaysian English. An experiment using state-of-the-art Named Entity Recognition (NER) solutions on Malaysian English news articles highlights that they cannot handle morphosyntactic variations in Malaysian English. To the best of our knowledge, there is no annotated dataset available to improvise the model. To address these issues, we constructed a Malaysian English News (MEN) dataset, which contains 200 news articles that are manually annotated with entities and relations. We then fine-tuned the spaCy NER tool and validated that having a dataset tailor-made for Malaysian English could improve the performance of NER in Malaysian English significantly. This
    

