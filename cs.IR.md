# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights.](http://arxiv.org/abs/2305.11700) | 本论文探究了在基于文本的协同过滤中使用大型语言模型所能带来的性能提升，并揭示了TCF程序扩展的极限。研究人员比较了使用不同大小的语言模型在基于文本的协同过滤算法中的性能表现。 |
| [^2] | [Vertical Semi-Federated Learning for Efficient Online Advertising.](http://arxiv.org/abs/2209.15635) | 垂直半联合学习为在线广告领域提供了高效的解决方案，通过学习一个联合感知的局部模型以应对传统垂直联合学习的限制。 |

# 详细

[^1]: 探究利用大型语言模型探索基于文本的协同过滤的极限：发现和认识

    Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights. (arXiv:2305.11700v1 [cs.IR])

    [http://arxiv.org/abs/2305.11700](http://arxiv.org/abs/2305.11700)

    本论文探究了在基于文本的协同过滤中使用大型语言模型所能带来的性能提升，并揭示了TCF程序扩展的极限。研究人员比较了使用不同大小的语言模型在基于文本的协同过滤算法中的性能表现。

    

    基于文本的协同过滤成为现今文本和新闻推荐的主流方法，利用文本编码器或语言模型(LMs)表示物品。然而，现有的文本协同过滤模型主要集中在使用中小型的LMs上，如果将物品编码器替换为最大最强大的1750亿参数的GPT-3模型，将会对推荐性能产生什么影响尚不确定。作者开展了一系列实验，探索TCF程序的性能极限。具体来说，作者将物品编码器规模从一亿扩大到一百亿以揭示TCF程序的扩展极限，同时还探究了使用超大LMs是否能实现推荐任务的通用物品表示方法。此外，作者比较了使用最强大的LMs和中等LMs实现的基于文本协同过滤的性能差异。

    Text-based collaborative filtering (TCF) has become the mainstream approach for text and news recommendation, utilizing text encoders, also known as language models (LMs), to represent items. However, existing TCF models primarily focus on using small or medium-sized LMs. It remains uncertain what impact replacing the item encoder with one of the largest and most powerful LMs, such as the 175-billion parameter GPT-3 model, would have on recommendation performance. Can we expect unprecedented results? To this end, we conduct an extensive series of experiments aimed at exploring the performance limits of the TCF paradigm. Specifically, we increase the size of item encoders from one hundred million to one hundred billion to reveal the scaling limits of the TCF paradigm. We then examine whether these extremely large LMs could enable a universal item representation for the recommendation task. Furthermore, we compare the performance of the TCF paradigm utilizing the most powerful LMs to the
    
[^2]: 垂直半联合学习用于高效在线广告

    Vertical Semi-Federated Learning for Efficient Online Advertising. (arXiv:2209.15635v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.15635](http://arxiv.org/abs/2209.15635)

    垂直半联合学习为在线广告领域提供了高效的解决方案，通过学习一个联合感知的局部模型以应对传统垂直联合学习的限制。

    

    传统的垂直联合学习架构存在两个主要问题：1）适用范围受限于重叠样本；2）实时联合服务的系统挑战较高，这限制了其在广告系统中的应用。为解决这些问题，我们提出了一种新的学习设置——半垂直联合学习(Semi-VFL)，以应对这些挑战。半垂直联合学习旨在实现垂直联合学习的实际工业应用方式，通过学习一个联合感知的局部模型，该模型表现优于单方模型，同时保持了局部服务的便利性。为此，我们提出了精心设计的联合特权学习框架(JPL)，来解决被动方特征缺失和适应整个样本空间这两个问题。具体而言，我们构建了一个推理高效的适用于整个样本空间的单方学生模型，同时保持了联合特征扩展的优势。新的表示蒸馏

    The traditional vertical federated learning schema suffers from two main issues: 1) restricted applicable scope to overlapped samples and 2) high system challenge of real-time federated serving, which limits its application to advertising systems. To this end, we advocate a new learning setting Semi-VFL (Vertical Semi-Federated Learning) to tackle these challenge. Semi-VFL is proposed to achieve a practical industry application fashion for VFL, by learning a federation-aware local model which performs better than single-party models and meanwhile maintain the convenience of local-serving. For this purpose, we propose the carefully designed Joint Privileged Learning framework (JPL) to i) alleviate the absence of the passive party's feature and ii) adapt to the whole sample space. Specifically, we build an inference-efficient single-party student model applicable to the whole sample space and meanwhile maintain the advantage of the federated feature extension. New representation distilla
    

