# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Semantic Search Engine for Mathlib4](https://arxiv.org/abs/2403.13310) | 提出了一个用于Mathlib4的语义搜索引擎，能够接受非正式查询并找到相关定理，为解决在mathlib4中搜索困难问题提供了新的方法。 |
| [^2] | [CIDER: Category-Guided Intent Disentanglement for Accurate Personalized News Recommendation.](http://arxiv.org/abs/2310.09401) | CIDER是一种基于类别引导的个性化新闻推荐框架，通过意图分离和一致性的新闻表示来准确理解新闻文章的多个意图，并区分用户不同的后阅读偏好。 |

# 详细

[^1]: 一个用于Mathlib4的语义搜索引擎

    A Semantic Search Engine for Mathlib4

    [https://arxiv.org/abs/2403.13310](https://arxiv.org/abs/2403.13310)

    提出了一个用于Mathlib4的语义搜索引擎，能够接受非正式查询并找到相关定理，为解决在mathlib4中搜索困难问题提供了新的方法。

    

    交互式定理证明器Lean使得可以验证正式数学证明，并且得到一个不断扩大的社区的支持。该生态系统的核心是其数学库mathlib4，为扩展范围的数学理论的形式化奠定了基础。然而，在mathlib4中搜索定理可能具有挑战性。为了成功在mathlib4中搜索，用户通常需要熟悉其命名约定或文档字符串。因此，创建一个语义搜索引擎，可以方便地被具有不同熟悉程度的mathlib4的个人使用是非常重要的。在本文中，我们提出了一个用于mathlib4的语义搜索引擎，可以接受非正式查询并找到相关定理。我们还建立了一个用于评估各种mathlib4搜索引擎性能的基准。

    arXiv:2403.13310v1 Announce Type: cross  Abstract: The interactive theorem prover, Lean, enables the verification of formal mathematical proofs and is backed by an expanding community. Central to this ecosystem is its mathematical library, mathlib4, which lays the groundwork for the formalization of an expanding range of mathematical theories. However, searching for theorems in mathlib4 can be challenging. To successfully search in mathlib4, users often need to be familiar with its naming conventions or documentation strings. Therefore, creating a semantic search engine that can be used easily by individuals with varying familiarity with mathlib4 is very important. In this paper, we present a semantic search engine for mathlib4 that accepts informal queries and finds the relevant theorems. We also establish a benchmark for assessing the performance of various search engines for mathlib4.
    
[^2]: CIDER: 基于类别引导的意图分离方法用于准确的个性化新闻推荐

    CIDER: Category-Guided Intent Disentanglement for Accurate Personalized News Recommendation. (arXiv:2310.09401v1 [cs.IR])

    [http://arxiv.org/abs/2310.09401](http://arxiv.org/abs/2310.09401)

    CIDER是一种基于类别引导的个性化新闻推荐框架，通过意图分离和一致性的新闻表示来准确理解新闻文章的多个意图，并区分用户不同的后阅读偏好。

    

    个性化新闻推荐旨在帮助用户找到与其兴趣相符的新闻文章，这在缓解用户信息过载问题方面起到至关重要的作用。尽管许多最近的研究致力于改进用户和新闻的表示方法，但以下挑战很少被研究：（C1）如何准确理解一篇新闻文章中包含的多个意图？以及（C2）如何区分用户点击历史中对新闻文章有不同后阅读偏好的情况？为了同时解决这两个挑战，在本文中，我们提出了一种新的个性化新闻推荐框架（CIDER），它利用（1）基于类别引导的意图分离来解决（C1）和（2）基于一致性的新闻表示来解决（C2）。此外，我们将类别预测纳入CIDER的训练过程作为辅助任务，这提供了额外的监督信号，以增强意图分离。在两个真实数据集上进行了广泛的实验。

    Personalized news recommendation aims to assist users in finding news articles that align with their interests, which plays a pivotal role in mitigating users' information overload problem. Although many recent works have been studied for better user and news representations, the following challenges have been rarely studied: (C1) How to precisely comprehend a range of intents coupled within a news article? and (C2) How to differentiate news articles with varying post-read preferences in users' click history? To tackle both challenges together, in this paper, we propose a novel personalized news recommendation framework (CIDER) that employs (1) category-guided intent disentanglement for (C1) and (2) consistency-based news representation for (C2). Furthermore, we incorporate a category prediction into the training process of CIDER as an auxiliary task, which provides supplementary supervisory signals to enhance intent disentanglement. Extensive experiments on two real-world datasets rev
    

