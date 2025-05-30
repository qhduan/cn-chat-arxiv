# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Leveraging Large Language Models for Relevance Judgments in Legal Case Retrieval](https://arxiv.org/abs/2403.18405) | 设计一种新颖的几轮工作流程，专门用于法律案例的相关判断，能够通过模仿人类注释者的过程并整合专家推理，提高相关性判断的准确性。 |
| [^2] | [Ensuring User-side Fairness in Dynamic Recommender Systems.](http://arxiv.org/abs/2308.15651) | 本文提出了一种名为FADE的端到端框架，通过微调策略动态减轻推荐系统中用户群体之间的性能差异。 |

# 详细

[^1]: 利用大型语言模型进行法律案例检索中的相关性判断

    Leveraging Large Language Models for Relevance Judgments in Legal Case Retrieval

    [https://arxiv.org/abs/2403.18405](https://arxiv.org/abs/2403.18405)

    设计一种新颖的几轮工作流程，专门用于法律案例的相关判断，能够通过模仿人类注释者的过程并整合专家推理，提高相关性判断的准确性。

    

    收集法律案例检索的相关判决是一项具有挑战性且耗时的任务。准确判断两个法律案例之间的相关性需要阅读冗长的文本并具备高水平的领域专业知识以提取法律事实并作出司法判断。随着先进的大型语言模型的出现，一些最近的研究表明使用LLM（Large Language Models）进行相关性判断是有前途的。然而，将一般性大型语言模型应用于法律案例检索中可靠的相关性判断的方法尚未得到充分探讨。为了填补这一研究空白，我们设计了一种新颖的几轮工作流程，专门用于法律案例的相关判断。所提出的工作流程将注释过程分解为一系列阶段，模仿人类注释者所使用的过程，并使专家推理能够灵活地整合以增强相关性判断的准确性。

    arXiv:2403.18405v1 Announce Type: new  Abstract: Collecting relevant judgments for legal case retrieval is a challenging and time-consuming task. Accurately judging the relevance between two legal cases requires a considerable effort to read the lengthy text and a high level of domain expertise to extract Legal Facts and make juridical judgments. With the advent of advanced large language models, some recent studies have suggested that it is promising to use LLMs for relevance judgment. Nonetheless, the method of employing a general large language model for reliable relevance judgments in legal case retrieval is yet to be thoroughly explored. To fill this research gap, we devise a novel few-shot workflow tailored to the relevant judgment of legal cases. The proposed workflow breaks down the annotation process into a series of stages, imitating the process employed by human annotators and enabling a flexible integration of expert reasoning to enhance the accuracy of relevance judgments.
    
[^2]: 在动态推荐系统中确保用户侧公平性

    Ensuring User-side Fairness in Dynamic Recommender Systems. (arXiv:2308.15651v1 [cs.IR])

    [http://arxiv.org/abs/2308.15651](http://arxiv.org/abs/2308.15651)

    本文提出了一种名为FADE的端到端框架，通过微调策略动态减轻推荐系统中用户群体之间的性能差异。

    

    用户侧群体公平性对现代推荐系统至关重要，它旨在减轻由敏感属性（如性别、种族或年龄）定义的用户群体之间的性能差异。我们发现这种差异往往会随着时间的推移而持续存在甚至增加。这需要在动态环境中有效解决用户侧公平性的方法，然而这在文献中很少被探讨。然而，用于确保用户侧公平性（即减少性能差异）的典型方法——公平约束重新排名，在动态设定中面临两个基本挑战：（1）基于排名的公平约束的非可微性，阻碍了端到端训练范式；（2）时间效率低下，阻碍了对用户偏好变化的快速适应。在本文中，我们提出了一种名为FADE的端到端框架，通过微调策略动态减轻性能差异。为了解决上述挑战，FADE提出了一种 fine-tuning 策略。

    User-side group fairness is crucial for modern recommender systems, as it aims to alleviate performance disparity between groups of users defined by sensitive attributes such as gender, race, or age. We find that the disparity tends to persist or even increase over time. This calls for effective ways to address user-side fairness in a dynamic environment, which has been infrequently explored in the literature. However, fairness-constrained re-ranking, a typical method to ensure user-side fairness (i.e., reducing performance disparity), faces two fundamental challenges in the dynamic setting: (1) non-differentiability of the ranking-based fairness constraint, which hinders the end-to-end training paradigm, and (2) time-inefficiency, which impedes quick adaptation to changes in user preferences. In this paper, we propose FAir Dynamic rEcommender (FADE), an end-to-end framework with fine-tuning strategy to dynamically alleviate performance disparity. To tackle the above challenges, FADE u
    

