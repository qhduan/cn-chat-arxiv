# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Post-Training Attribute Unlearning in Recommender Systems](https://arxiv.org/abs/2403.06737) | 本文提出了一种后训练属性取消学习（PoT-AU）的方法，通过设计两部分损失函数，旨在在推荐系统中保护用户的敏感属性。 |

# 详细

[^1]: 在推荐系统中进行后训练属性遗忘

    Post-Training Attribute Unlearning in Recommender Systems

    [https://arxiv.org/abs/2403.06737](https://arxiv.org/abs/2403.06737)

    本文提出了一种后训练属性取消学习（PoT-AU）的方法，通过设计两部分损失函数，旨在在推荐系统中保护用户的敏感属性。

    

    随着推荐系统中日益增长的隐私问题，推荐取消学习越来越受到关注。现有研究主要使用训练数据，即模型输入，作为取消学习目标。然而，即使模型在训练过程中没有明确遇到，攻击者仍可以从模型中提取私人信息。我们将这些未见信息称为属性，并将其视为取消学习目标。为了保护用户的敏感属性，属性取消学习（AU）旨在使目标属性难以分辨。本文侧重于AU的一个严格但实际的设置，即后训练属性取消学习（PoT-AU），其中取消学习只能在推荐模型训练完成后执行。为了解决推荐系统中的PoT-AU问题，我们提出了一个两部分损失函数。第一部分是可区分性损失，我们设计了一个基于分布的度量

    arXiv:2403.06737v1 Announce Type: new  Abstract: With the growing privacy concerns in recommender systems, recommendation unlearning is getting increasing attention. Existing studies predominantly use training data, i.e., model inputs, as unlearning target. However, attackers can extract private information from the model even if it has not been explicitly encountered during training. We name this unseen information as \textit{attribute} and treat it as unlearning target. To protect the sensitive attribute of users, Attribute Unlearning (AU) aims to make target attributes indistinguishable. In this paper, we focus on a strict but practical setting of AU, namely Post-Training Attribute Unlearning (PoT-AU), where unlearning can only be performed after the training of the recommendation model is completed. To address the PoT-AU problem in recommender systems, we propose a two-component loss function. The first component is distinguishability loss, where we design a distribution-based meas
    

