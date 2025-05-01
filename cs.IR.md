# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RankingSHAP -- Listwise Feature Attribution Explanations for Ranking Models](https://arxiv.org/abs/2403.16085) | 本文针对排名模型的特征归因进行了严格定义，并提出了RankingSHAP作为一种逐项排名归因方法，突破了当前解释评估方案的局限，提出了两种新的评估范式。 |

# 详细

[^1]: RankingSHAP -- 针对排名模型的逐项特征归因解释

    RankingSHAP -- Listwise Feature Attribution Explanations for Ranking Models

    [https://arxiv.org/abs/2403.16085](https://arxiv.org/abs/2403.16085)

    本文针对排名模型的特征归因进行了严格定义，并提出了RankingSHAP作为一种逐项排名归因方法，突破了当前解释评估方案的局限，提出了两种新的评估范式。

    

    特征归因是一种常用的解释类型，用于在训练模型后事后解释预测。然而，在信息检索领域，这种方法并没有得到很好的研究。重要的是，特征归因很少被严格定义，除了将最重要的特征归因为最高值之外。什么是比其他特征更重要的特征往往被模糊地描述。因此，大多数方法只关注选择最重要的特征，不充分利用甚至忽视特征内的相对重要性。在这项工作中，我们严格定义了排名模型特征归因的概念，并列出了一个有效归因应具备的基本属性。然后，我们提出RankingSHAP作为逐项排名归因方法的具体实例。与目前关注选择的解释评估方案相反，我们提出了两种用于评估归因的新颖评估范式。

    arXiv:2403.16085v1 Announce Type: new  Abstract: Feature attributions are a commonly used explanation type, when we want to posthoc explain the prediction of a trained model. Yet, they are not very well explored in IR. Importantly, feature attribution has rarely been rigorously defined, beyond attributing the most important feature the highest value. What it means for a feature to be more important than others is often left vague. Consequently, most approaches focus on just selecting the most important features and under utilize or even ignore the relative importance within features. In this work, we rigorously define the notion of feature attribution for ranking models, and list essential properties that a valid attribution should have. We then propose RankingSHAP as a concrete instantiation of a list-wise ranking attribution method. Contrary to current explanation evaluation schemes that focus on selections, we propose two novel evaluation paradigms for evaluating attributions over l
    

