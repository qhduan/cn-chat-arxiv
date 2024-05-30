# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining](https://arxiv.org/abs/2310.07713) | InstructRetro是目前规模最大的使用检索预训练的LLM，扩展了基础模型Retro 48B，通过指令调优在各种零样例任务上取得显著改进。 |
| [^2] | [UP5: Unbiased Foundation Model for Fairness-aware Recommendation.](http://arxiv.org/abs/2305.12090) | 本研究提出了一种新颖的基础模型UP5，它采用反事实公平促进技术来消除大型语言模型中的偏见，从而实现面向公平性的推荐。 |

# 详细

[^1]: InstructRetro: 检索增强的预训练中指令调优

    InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining

    [https://arxiv.org/abs/2310.07713](https://arxiv.org/abs/2310.07713)

    InstructRetro是目前规模最大的使用检索预训练的LLM，扩展了基础模型Retro 48B，通过指令调优在各种零样例任务上取得显著改进。

    

    使用检索增强技术对自回归大型语言模型（LLM）进行预训练可以提高困惑度和事实准确性。然而，现有的预训练检索增强LLM的规模仍然有限（如Retro具有75亿个参数），这限制了指令调优和零样例泛化的效果。本文介绍了Retro 48B，这是目前规模最大的使用检索预训练的LLM。具体来说，我们使用检索技术从1.2万亿个标记中继续预训练一个43B的GPT模型，并借助Retro方法将其扩展到4800亿个参数。值得注意的是，所得到的基础模型Retro 48B在困惑度方面显著优于仅使用1.2万亿个标记进行训练的43B GPT模型，且只增加了2.58%的GPU使用时间，展示了该方法的显著扩展潜力。在对Retro进行指令调优后，InstructRetro在各种零样例任务上表现出显著的改进。

    Pretraining auto-regressive large language models (LLMs) with retrieval demonstrates better perplexity and factual accuracy by leveraging external databases. However, the size of existing pretrained retrieval-augmented LLM is still limited (e.g., Retro has 7.5B parameters), which limits the effectiveness of instruction tuning and zero-shot generalization. In this work, we introduce Retro 48B, the largest LLM pretrained with retrieval. Specifically, we continue to pretrain a 43B GPT model on additional 100 billion tokens using the Retro augmentation method by retrieving from 1.2 trillion tokens. Notably, the obtained foundation model, Retro 48B, largely outperforms the counterpart GPT 43B trained on 1.2T tokens in terms of perplexity with only 2.58% additional GPU hours, demonstrating the significant scaling potential of the method. After instruction tuning on Retro, InstructRetro demonstrates significant improvement over the instruction tuned GPT on a wide range of zero-shot tasks. Spe
    
[^2]: UP5: 面向公平性推荐的无偏基础模型

    UP5: Unbiased Foundation Model for Fairness-aware Recommendation. (arXiv:2305.12090v1 [cs.IR])

    [http://arxiv.org/abs/2305.12090](http://arxiv.org/abs/2305.12090)

    本研究提出了一种新颖的基础模型UP5，它采用反事实公平促进技术来消除大型语言模型中的偏见，从而实现面向公平性的推荐。

    

    基于大型语言模型（LLM）等基础模型的最新进展，已将它们推到了推荐系统（RS）的前沿。此外，RS中的公平性很关键，因为许多用户将其用于决策和需求履行。然而，目前尚缺乏对推荐基础模型展示公平性水平和公平处理不同用户群组的适当方法的理解。本文侧重于用户方面的不公平问题，并通过彻底检查表明，LLMs中存在不公平性，导致不公平的推荐结果。为了消除LLM中的偏差以实现面向公平性的推荐，我们引入了一种基于反事实公平促进技术的新型无偏P5（UP5）基础模型。CFP包括两个子模块：个性化前缀提示和Prompt混合，从而增强了个体敏感属性的公平性。

    Recent advancements in foundation models such as large language models (LLM) have propelled them to the forefront of recommender systems (RS). Moreover, fairness in RS is critical since many users apply it for decision-making and demand fulfillment. However, at present, there is a lack of understanding regarding the level of fairness exhibited by recommendation foundation models and the appropriate methods for equitably treating different groups of users in foundation models. In this paper, we focus on user-side unfairness problem and show through a thorough examination that there is unfairness involved in LLMs that lead to unfair recommendation results. To eliminate bias from LLM for fairness-aware recommendation, we introduce a novel Unbiased P5 (UP5) foundation model based on Counterfactually-Fair-Prompting (CFP) techniques. CFP includes two sub-modules: a personalized prefix prompt that enhances fairness with respect to individual sensitive attributes, and a Prompt Mixture that int
    

