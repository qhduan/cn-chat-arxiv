# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unfolded Self-Reconstruction LSH: Towards Machine Unlearning in Approximate Nearest Neighbour Search.](http://arxiv.org/abs/2304.02350) | 本文提出了一种基于数据依赖的哈希方法USR-LSH，该方法通过展开实例级数据重建的优化更新，提高了数据的信息保留能力，同时还提出了一种动态的遗忘机制，使得数据可以快速删除和插入，无需重新训练，这是一种具有数据隐私和安全要求的在线ANN搜索的实际解决方案。 |
| [^2] | [MoocRadar: A Fine-grained and Multi-aspect Knowledge Repository for Improving Cognitive Student Modeling in MOOCs.](http://arxiv.org/abs/2304.02205) | 本文介绍了 MoocRadar，一种多方面的、细粒度的知识库，用于提高 MOOC 中认知学生建模的精度。 |
| [^3] | [A Simple and Effective Method of Cross-Lingual Plagiarism Detection.](http://arxiv.org/abs/2304.01352) | 该论文提出了一种简单有效的跨语言抄袭检测方法，不依赖机器翻译和词义消歧，使用开放的多语言同义词库进行候选检索任务和预训练的基于多语言BERT的语言模型进行详细分析，在多个基准测试中取得了最先进的结果。 |
| [^4] | [On Modeling Long-Term User Engagement from Stochastic Feedback.](http://arxiv.org/abs/2302.06101) | 本文提出了一种高效的基于数据的用户参与度与物品相关性建模方法，特别考虑了推荐系统中用户反馈和终止行为的随机性。 |

# 详细

[^1]: 未折叠自重建局部敏感哈希：走向近似最近邻搜索中的机器遗忘

    Unfolded Self-Reconstruction LSH: Towards Machine Unlearning in Approximate Nearest Neighbour Search. (arXiv:2304.02350v1 [cs.IR])

    [http://arxiv.org/abs/2304.02350](http://arxiv.org/abs/2304.02350)

    本文提出了一种基于数据依赖的哈希方法USR-LSH，该方法通过展开实例级数据重建的优化更新，提高了数据的信息保留能力，同时还提出了一种动态的遗忘机制，使得数据可以快速删除和插入，无需重新训练，这是一种具有数据隐私和安全要求的在线ANN搜索的实际解决方案。

    

    近似最近邻搜索是搜索引擎、推荐系统等的重要组成部分。许多最近的工作都是基于学习的数据分布依赖哈希，实现了良好的检索性能。但是，由于对用户隐私和安全的需求不断增加，我们经常需要从机器学习模型中删除用户数据信息以满足特定的隐私和安全要求。这种需求需要ANN搜索算法支持快速的在线数据删除和插入。当前的基于学习的哈希方法需要重新训练哈希函数，这是由于大规模数据的时间成本太高而难以承受的。为了解决这个问题，我们提出了一种新型的数据依赖哈希方法，名为unfolded self-reconstruction locality-sensitive hashing (USR-LSH)。我们的USR-LSH展开了实例级数据重建的优化更新，这比数据无关的LSH更能保留数据信息。此外，我们的USR-LSH提出了一种动态的遗忘机制，用于快速的数据删除和插入，无需重新训练。实验结果表明，USR-LSH在检索准确性和时间效率方面优于现有的哈希方法。USR-LSH是具有数据隐私和安全要求的在线ANN搜索的实际解决方案。

    Approximate nearest neighbour (ANN) search is an essential component of search engines, recommendation systems, etc. Many recent works focus on learning-based data-distribution-dependent hashing and achieve good retrieval performance. However, due to increasing demand for users' privacy and security, we often need to remove users' data information from Machine Learning (ML) models to satisfy specific privacy and security requirements. This need requires the ANN search algorithm to support fast online data deletion and insertion. Current learning-based hashing methods need retraining the hash function, which is prohibitable due to the vast time-cost of large-scale data. To address this problem, we propose a novel data-dependent hashing method named unfolded self-reconstruction locality-sensitive hashing (USR-LSH). Our USR-LSH unfolded the optimization update for instance-wise data reconstruction, which is better for preserving data information than data-independent LSH. Moreover, our US
    
[^2]: MoocRadar: 一种多方面的、细粒度的知识库，用于提高 MOOC 中认知学生建模的精度

    MoocRadar: A Fine-grained and Multi-aspect Knowledge Repository for Improving Cognitive Student Modeling in MOOCs. (arXiv:2304.02205v1 [cs.AI])

    [http://arxiv.org/abs/2304.02205](http://arxiv.org/abs/2304.02205)

    本文介绍了 MoocRadar，一种多方面的、细粒度的知识库，用于提高 MOOC 中认知学生建模的精度。

    

    学生建模是智能教育中推断学生学习特征的一项基本任务。尽管从知识跟踪和认知诊断的最近尝试提出了几个有希望改进当前模型可用性和有效性的方向，但现有公共数据集仍然不足以满足这些潜在解决方案的需求，因为它们忽略了完整的练习情境、细粒度的概念和认知标签。本文介绍了 MoocRadar，它是一个由 2,513 个练习问题、5,600 个知识概念和超过 1200 万行为记录组成的细粒度、多方面的知识库。具体而言，我们提出了一个框架，以保证细粒度概念和认知标签的高质量和全面性注释。统计和实验结果表明，我们的数据集为未来改进智能教育模型提供了一个基础。

    Student modeling, the task of inferring a student's learning characteristics through their interactions with coursework, is a fundamental issue in intelligent education. Although the recent attempts from knowledge tracing and cognitive diagnosis propose several promising directions for improving the usability and effectiveness of current models, the existing public datasets are still insufficient to meet the need for these potential solutions due to their ignorance of complete exercising contexts, fine-grained concepts, and cognitive labels. In this paper, we present MoocRadar, a fine-grained, multi-aspect knowledge repository consisting of 2,513 exercise questions, 5,600 knowledge concepts, and over 12 million behavioral records. Specifically, we propose a framework to guarantee a high-quality and comprehensive annotation of fine-grained concepts and cognitive labels. The statistical and experimental results indicate that our dataset provides the basis for the future improvements of e
    
[^3]: 一种简单有效的跨语言抄袭检测方法

    A Simple and Effective Method of Cross-Lingual Plagiarism Detection. (arXiv:2304.01352v1 [cs.CL])

    [http://arxiv.org/abs/2304.01352](http://arxiv.org/abs/2304.01352)

    该论文提出了一种简单有效的跨语言抄袭检测方法，不依赖机器翻译和词义消歧，使用开放的多语言同义词库进行候选检索任务和预训练的基于多语言BERT的语言模型进行详细分析，在多个基准测试中取得了最先进的结果。

    

    我们提出了一种简单的跨语言抄袭检测方法，适用于大量的语言。该方法利用开放的多语言同义词库进行候选检索任务，并利用预训练的基于多语言BERT的语言模型进行详细分析。该方法在使用时不依赖机器翻译和词义消歧，因此适用于许多语言，包括资源匮乏的语言。该方法在多个现有和新的基准测试中展示了其有效性，在法语、俄语和亚美尼亚语等语言中取得了最先进的结果。

    We present a simple cross-lingual plagiarism detection method applicable to a large number of languages. The presented approach leverages open multilingual thesauri for candidate retrieval task and pre-trained multilingual BERT-based language models for detailed analysis. The method does not rely on machine translation and word sense disambiguation when in use, and therefore is suitable for a large number of languages, including under-resourced languages. The effectiveness of the proposed approach is demonstrated for several existing and new benchmarks, achieving state-of-the-art results for French, Russian, and Armenian languages.
    
[^4]: 论建立基于随机反馈的长期用户参与度模型

    On Modeling Long-Term User Engagement from Stochastic Feedback. (arXiv:2302.06101v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2302.06101](http://arxiv.org/abs/2302.06101)

    本文提出了一种高效的基于数据的用户参与度与物品相关性建模方法，特别考虑了推荐系统中用户反馈和终止行为的随机性。

    

    推荐系统的终极目标是提高用户参与度。强化学习是实现此目标的一种有前途的范例，因为它直接优化了序贯推荐的整体表现。然而，现有的基于强化学习的方法需要保存推荐的物品以及其他候选物品，这会导致巨大的计算开销。本文提出了一种高效的替代方法，不需要候选项，而是直接从数据中建立用户参与度与物品之间的相关性。此外，所提出的方法考虑了用户反馈和终止行为的随机性，在推荐系统中具有普适性但在以前的基于强化学习的工作中很少被讨论。在真实推荐系统的在线 A/B 实验中，我们证实了所提出的方法的有效性和建立两种类型随机模型的重要性。

    An ultimate goal of recommender systems (RS) is to improve user engagement. Reinforcement learning (RL) is a promising paradigm for this goal, as it directly optimizes overall performance of sequential recommendation. However, many existing RL-based approaches induce huge computational overhead, because they require not only the recommended items but also all other candidate items to be stored. This paper proposes an efficient alternative that does not require the candidate items. The idea is to model the correlation between user engagement and items directly from data. Moreover, the proposed approach consider randomness in user feedback and termination behavior, which are ubiquitous for RS but rarely discussed in RL-based prior work. With online A/B experiments on real-world RS, we confirm the efficacy of the proposed approach and the importance of modeling the two types of randomness.
    

