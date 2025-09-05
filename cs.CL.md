# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploring Hybrid Linguistic Features for Turkish Text Readability.](http://arxiv.org/abs/2306.03774) | 本文结合神经网络模型和各语言层面上的特征，开发出一种先进的土耳其文本可读性工具，发现了影响土耳其文本可读性的关键语言特征。 |
| [^2] | [ChatGPT as a Text Simplification Tool to Remove Bias.](http://arxiv.org/abs/2305.06166) | ChatGPT作为文本简化工具可以去除语言模型在训练过程中对某些特定群体的偏见，减少模型的歧视性。（注：ChatGPT是一种基于Transformer的自然语言处理模型） |

# 详细

[^1]: 使用混合语言特征提高土耳其文本可读性的研究

    Exploring Hybrid Linguistic Features for Turkish Text Readability. (arXiv:2306.03774v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.03774](http://arxiv.org/abs/2306.03774)

    本文结合神经网络模型和各语言层面上的特征，开发出一种先进的土耳其文本可读性工具，发现了影响土耳其文本可读性的关键语言特征。

    

    本文首次对土耳其文本的自动可读性评估进行了全面研究。我们结合了最先进的神经网络模型和词汇、形态句法、语法和话语水平的语言特征，开发了一个先进的可读性工具。我们评估了传统可读性公式与现代自动方法的效果，并确定了影响土耳其文本可读性的关键语言特征。

    This paper presents the first comprehensive study on automatic readability assessment of Turkish texts. We combine state-of-the-art neural network models with linguistic features at lexical, morphosyntactic, syntactic and discourse levels to develop an advanced readability tool. We evaluate the effectiveness of traditional readability formulas compared to modern automated methods and identify key linguistic features that determine the readability of Turkish texts.
    
[^2]: ChatGPT作为去除偏见的文本简化工具

    ChatGPT as a Text Simplification Tool to Remove Bias. (arXiv:2305.06166v1 [cs.CL])

    [http://arxiv.org/abs/2305.06166](http://arxiv.org/abs/2305.06166)

    ChatGPT作为文本简化工具可以去除语言模型在训练过程中对某些特定群体的偏见，减少模型的歧视性。（注：ChatGPT是一种基于Transformer的自然语言处理模型）

    

    在训练期间，语言模型可以捕捉到特定子群体的特定语言信号，如果模型学习了捕捉某个群体的语言，可能会导致歧视。如果模型开始将特定语言与某个特定群体联系起来，基于此语言做出的任何决策都将与其受保护特征有着强烈的相关性。我们探索了一种可能的偏见缓解技术，即文本简化。这个想法的驱动力是简化文本应该标准化语言，使其以一种方式说话，同时保持相同的含义。实验显示，针对敏感属性预测的分类器精度会因使用简化数据而下降高达17%。

    The presence of specific linguistic signals particular to a certain sub-group of people can be picked up by language models during training. This may lead to discrimination if the model has learnt to pick up on a certain group's language. If the model begins to associate specific language with a distinct group, any decisions made based upon this language would hold a strong correlation to a decision based on their protected characteristic. We explore a possible technique for bias mitigation in the form of simplification of text. The driving force of this idea is that simplifying text should standardise language to one way of speaking while keeping the same meaning. The experiment shows promising results as the classifier accuracy for predicting the sensitive attribute drops by up to 17% for the simplified data.
    

