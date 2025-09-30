# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Machines Do See Color: A Guideline to Classify Different Forms of Racist Discourse in Large Corpora.](http://arxiv.org/abs/2401.09333) | 本文提供了一个逐步可推广的准则，用于在大规模语料库中识别和分类不同形式的种族主义言论。通过对种族主义的概念化和上下文化，以及使用XLM-R和XLM-R-Racismo模型，我们展示了在大规模语料库中进行种族主义分类的优势。 |
| [^2] | [Continual Dialogue State Tracking via Example-Guided Question Answering.](http://arxiv.org/abs/2305.13721) | 本文建议将对话状态跟踪重构为由例子引导的粒度问题回答任务，以最小化服务之间的任务转移，获得持续的学习效益。通过结合简单的持续学习策略，可以在基准数据集上获得最先进的性能。 |

# 详细

[^1]: 机器能够看到颜色：大规模语料库中分类不同形式种族主义言论的准则

    Machines Do See Color: A Guideline to Classify Different Forms of Racist Discourse in Large Corpora. (arXiv:2401.09333v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.09333](http://arxiv.org/abs/2401.09333)

    本文提供了一个逐步可推广的准则，用于在大规模语料库中识别和分类不同形式的种族主义言论。通过对种族主义的概念化和上下文化，以及使用XLM-R和XLM-R-Racismo模型，我们展示了在大规模语料库中进行种族主义分类的优势。

    

    目前识别和分类文本中的种族主义语言的方法主要依赖小规模的质性方法或大规模的方法，专注于明显的种族主义言论。本文提供了一个逐步可推广的准则，用于在大规模语料库中识别和分类不同形式的种族主义言论。在我们的方法中，我们首先将种族主义及其不同表现形式进行概念化。然后，我们将这些种族主义表现形式置于感兴趣的时间和地点背景下，以便研究人员能够识别它们的话语形式。最后，我们应用了XLM-RoBERTa（XLM-R），这是一个具有先进上下文理解能力的跨语言监督文本分类模型。我们展示了XLM-R和XLM-R-Racismo（我们的预训练模型）在大规模语料库中对种族主义进行分类的性能优于其他最先进的方法。我们通过使用涉及2018年至2021年厄瓜多尔本土群体的推文语料库来说明我们的方法。

    Current methods to identify and classify racist language in text rely on small-n qualitative approaches or large-n approaches focusing exclusively on overt forms of racist discourse. This article provides a step-by-step generalizable guideline to identify and classify different forms of racist discourse in large corpora. In our approach, we start by conceptualizing racism and its different manifestations. We then contextualize these racist manifestations to the time and place of interest, which allows researchers to identify their discursive form. Finally, we apply XLM-RoBERTa (XLM-R), a cross-lingual model for supervised text classification with a cutting-edge contextual understanding of text. We show that XLM-R and XLM-R-Racismo, our pretrained model, outperform other state-of-the-art approaches in classifying racism in large corpora. We illustrate our approach using a corpus of tweets relating to the Ecuadorian ind\'igena community between 2018 and 2021.
    
[^2]: 基于示例引导问答的持续对话状态跟踪

    Continual Dialogue State Tracking via Example-Guided Question Answering. (arXiv:2305.13721v1 [cs.CL])

    [http://arxiv.org/abs/2305.13721](http://arxiv.org/abs/2305.13721)

    本文建议将对话状态跟踪重构为由例子引导的粒度问题回答任务，以最小化服务之间的任务转移，获得持续的学习效益。通过结合简单的持续学习策略，可以在基准数据集上获得最先进的性能。

    

    对话系统需要不断更新以适应新服务，但是简单地使用新服务的数据进行训练会降低先前学习的服务的性能。本文发现，对话状态跟踪(DST)是一个简单的自然语言理解任务，我们建议将其重构为一组由例子引导的粒度问题回答任务，以最小化服务之间的任务转移，从而获得持续的学习效益。我们的方法可以减轻特定服务的记忆负担，并教会模型将所给问题和示例用于从对话中提取必要信息。我们发现，一个只有6000万个参数的模型可以通过学习从检索器获取的上下文示例获得巨大的提升。将我们的方法与简单的持续学习策略相结合，可以在基准数据集上获得最先进的性能，证明了我们方法的有效性。

    Dialogue systems are frequently updated to accommodate new services, but naively updating them by continually training with data for new services in diminishing performance on previously learnt services. Motivated by the insight that dialogue state tracking (DST), a crucial component of dialogue systems that estimates the user's goal as a conversation proceeds, is a simple natural language understanding task, we propose reformulating it as a bundle of granular example-guided question answering tasks to minimize the task shift between services and thus benefit continual learning. Our approach alleviates service-specific memorization and teaches a model to contextualize the given question and example to extract the necessary information from the conversation. We find that a model with just 60M parameters can achieve a significant boost by learning to learn from in-context examples retrieved by a retriever trained to identify turns with similar dialogue state changes. Combining our method
    

