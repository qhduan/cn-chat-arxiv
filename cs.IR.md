# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Representation Learning with Large Language Models for Recommendation.](http://arxiv.org/abs/2310.15950) | 这篇论文介绍了一个模型-不可知的框架RLMRec，通过使用大语言模型（LLMs）来增强传统的基于ID的推荐系统，并解决了可扩展性问题、仅依赖文本的限制以及提示输入限制等挑战。 |

# 详细

[^1]: 用大语言模型进行推荐中的表示学习

    Representation Learning with Large Language Models for Recommendation. (arXiv:2310.15950v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.15950](http://arxiv.org/abs/2310.15950)

    这篇论文介绍了一个模型-不可知的框架RLMRec，通过使用大语言模型（LLMs）来增强传统的基于ID的推荐系统，并解决了可扩展性问题、仅依赖文本的限制以及提示输入限制等挑战。

    

    推荐系统在深度学习和图神经网络的影响下取得了显著进展，特别是在捕捉复杂的用户-物品关系方面。然而，这些基于图的推荐系统严重依赖于基于ID的数据，可能忽略了与用户和物品相关的有价值的文本信息，导致学到的表示不够富有信息。此外，隐式反馈数据的利用引入了潜在的噪声和偏差，给用户偏好学习的有效性带来了挑战。尽管将大语言模型（LLMs）与传统的基于ID的推荐系统相结合已经引起了人们的关注，但在实际推荐系统中有效实施还需要解决可扩展性问题、仅依赖文本的限制以及提示输入限制等挑战。为了解决这些挑战，我们提出了一个模型不可知的框架RLMRec，旨在通过LLM强化表示来增强现有的推荐系统。

    Recommender systems have seen significant advancements with the influence of deep learning and graph neural networks, particularly in capturing complex user-item relationships. However, these graph-based recommenders heavily depend on ID-based data, potentially disregarding valuable textual information associated with users and items, resulting in less informative learned representations. Moreover, the utilization of implicit feedback data introduces potential noise and bias, posing challenges for the effectiveness of user preference learning. While the integration of large language models (LLMs) into traditional ID-based recommenders has gained attention, challenges such as scalability issues, limitations in text-only reliance, and prompt input constraints need to be addressed for effective implementation in practical recommender systems. To address these challenges, we propose a model-agnostic framework RLMRec that aims to enhance existing recommenders with LLM-empowered representati
    

