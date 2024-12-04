# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Using text embedding models and vector databases as text classifiers with the example of medical data](https://arxiv.org/abs/2402.16886) | 向量数据库和嵌入模型的应用为文本分类器提供了强大的方式来表达数据模式，特别是在医疗领域中开始有着广泛的应用。 |
| [^2] | [CIDER: Category-Guided Intent Disentanglement for Accurate Personalized News Recommendation.](http://arxiv.org/abs/2310.09401) | CIDER是一种基于类别引导的个性化新闻推荐框架，通过意图分离和一致性的新闻表示来准确理解新闻文章的多个意图，并区分用户不同的后阅读偏好。 |

# 详细

[^1]: 使用文本嵌入模型和向量数据库作为文本分类器的研究，以医疗数据为例

    Using text embedding models and vector databases as text classifiers with the example of medical data

    [https://arxiv.org/abs/2402.16886](https://arxiv.org/abs/2402.16886)

    向量数据库和嵌入模型的应用为文本分类器提供了强大的方式来表达数据模式，特别是在医疗领域中开始有着广泛的应用。

    

    大型语言模型（LLMs）的出现是令人兴奋的，并已在许多领域找到应用，但通常情况下，医学领域的标准要求非常高。与LLMs配合使用，向量嵌入模型和向量数据库提供了一种强大的方式来表达各种数据模式，这些数据模式容易被典型的机器学习模型所理解。除了方便地向这些向量数据库添加信息、知识和数据外，它们还提供了一个令人信服的理由，即将其应用于通常由人类完成的检索信息任务的各种领域。Google的研究人员开发了一个清晰的替代模型Med-PaLM，专门旨在与临床医师的医学知识水平匹配。在训练分类器和开发模型时，保持事实和减少偏见是至关重要的。在本文中，我们探讨了向量数据库和嵌入模型的应用

    arXiv:2402.16886v1 Announce Type: cross  Abstract: The advent of Large Language Models (LLMs) is promising and has found application in numerous fields, but as it often is with the medical field, the bar is typically quite high [5]. In tandem with LLMs, vector embedding models and vector databases provide a robust way of expressing numerous modes of data that are easily digestible by typical machine learning models. Along with the ease of adding information, knowledge, and data to these vector databases, they provide a compelling reason to apply them in numerous fields where the task of retrieving information is typically done by humans. Researchers at Google have developed a clear alternative model, Med-PaLM [6] specifically designed to match a clinician's level of accuracy when it comes to medical knowledge. When training classifiers, and developing models, it is imperative to maintain factuality and reduce bias [4]. Here, we explore the use of vector databases and embedding models a
    
[^2]: CIDER: 基于类别引导的意图分离方法用于准确的个性化新闻推荐

    CIDER: Category-Guided Intent Disentanglement for Accurate Personalized News Recommendation. (arXiv:2310.09401v1 [cs.IR])

    [http://arxiv.org/abs/2310.09401](http://arxiv.org/abs/2310.09401)

    CIDER是一种基于类别引导的个性化新闻推荐框架，通过意图分离和一致性的新闻表示来准确理解新闻文章的多个意图，并区分用户不同的后阅读偏好。

    

    个性化新闻推荐旨在帮助用户找到与其兴趣相符的新闻文章，这在缓解用户信息过载问题方面起到至关重要的作用。尽管许多最近的研究致力于改进用户和新闻的表示方法，但以下挑战很少被研究：（C1）如何准确理解一篇新闻文章中包含的多个意图？以及（C2）如何区分用户点击历史中对新闻文章有不同后阅读偏好的情况？为了同时解决这两个挑战，在本文中，我们提出了一种新的个性化新闻推荐框架（CIDER），它利用（1）基于类别引导的意图分离来解决（C1）和（2）基于一致性的新闻表示来解决（C2）。此外，我们将类别预测纳入CIDER的训练过程作为辅助任务，这提供了额外的监督信号，以增强意图分离。在两个真实数据集上进行了广泛的实验。

    Personalized news recommendation aims to assist users in finding news articles that align with their interests, which plays a pivotal role in mitigating users' information overload problem. Although many recent works have been studied for better user and news representations, the following challenges have been rarely studied: (C1) How to precisely comprehend a range of intents coupled within a news article? and (C2) How to differentiate news articles with varying post-read preferences in users' click history? To tackle both challenges together, in this paper, we propose a novel personalized news recommendation framework (CIDER) that employs (1) category-guided intent disentanglement for (C1) and (2) consistency-based news representation for (C2). Furthermore, we incorporate a category prediction into the training process of CIDER as an auxiliary task, which provides supplementary supervisory signals to enhance intent disentanglement. Extensive experiments on two real-world datasets rev
    

