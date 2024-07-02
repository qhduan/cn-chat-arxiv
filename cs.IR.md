# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models](https://arxiv.org/abs/2403.03900) | Mamba4Rec是首个探索选择性状态空间模型用于高效序列推荐的工作，能够在保持推断效率的同时提升模型性能。 |
| [^2] | [Towards Efficient and Effective Unlearning of Large Language Models for Recommendation](https://arxiv.org/abs/2403.03536) | 提出了E2URec，这是为了解决大型语言模型在推荐系统中遗忘特定用户数据所面临的效率和有效性方面的挑战。 |
| [^3] | [MuGI: Enhancing Information Retrieval through Multi-Text Generation Intergration with Large Language Models.](http://arxiv.org/abs/2401.06311) | MuGI是一个简单而有效的多文本生成集成框架，它通过与大型语言模型合作生成多个伪参考文献，并将其与查询集成以提升信息检索性能。在实验中，MuGI模型在TREC DL数据集上的BM25性能上取得了18%以上的增强，并在BEIR上提高了7.5%。 |
| [^4] | [Autumn: A Scalable Read Optimized LSM-tree based Key-Value Stores with Fast Point and Range Read Speed.](http://arxiv.org/abs/2305.05074) | Autumn是一个可扩展的、面向读操作优化的LSM-tree键值存储引擎，其创新之处在于通过动态调整相邻两层之间的容量比来不断提高读性能，使得点读和区间读成本从之前最优的$O(logN)$复杂度优化到了$O(\sqrt{logN})$。 |

# 详细

[^1]: Mamba4Rec：针对具有选择性状态空间模型的高效序列推荐

    Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models

    [https://arxiv.org/abs/2403.03900](https://arxiv.org/abs/2403.03900)

    Mamba4Rec是首个探索选择性状态空间模型用于高效序列推荐的工作，能够在保持推断效率的同时提升模型性能。

    

    序列推荐旨在估计动态用户偏好和历史用户行为之间的顺序依赖关系。本文提出了Mamba4Rec，这是首个探索选择性SSM潜力以实现高效序列推荐的工作。通过基本的Mamba块构建，结合一系列顺序建模技术，我们进一步提升了模型性能，同时保持了推断效率。实验证明，Mamba4Rec能够很好地处理序列推荐的有效性问题。

    arXiv:2403.03900v1 Announce Type: new  Abstract: Sequential recommendation aims to estimate the dynamic user preferences and sequential dependencies among historical user behaviors. Although Transformer-based models have proven to be effective for sequential recommendation, they suffer from the inference inefficiency problem stemming from the quadratic computational complexity of attention operators, especially for long-range behavior sequences. Inspired by the recent success of state space models (SSMs), we propose Mamba4Rec, which is the first work to explore the potential of selective SSMs for efficient sequential recommendation. Built upon the basic Mamba block which is a selective SSM with an efficient hardware-aware parallel algorithm, we incorporate a series of sequential modeling techniques to further promote the model performance and meanwhile maintain the inference efficiency. Experiments on two public datasets demonstrate that Mamba4Rec is able to well address the effectiven
    
[^2]: 为推荐而设计的大型语言模型的高效和有效的遗忘

    Towards Efficient and Effective Unlearning of Large Language Models for Recommendation

    [https://arxiv.org/abs/2403.03536](https://arxiv.org/abs/2403.03536)

    提出了E2URec，这是为了解决大型语言模型在推荐系统中遗忘特定用户数据所面临的效率和有效性方面的挑战。

    

    大型语言模型（LLMs）的显著进展产生了一项有前途的研究方向，即利用LLMs作为推荐系统（LLMRec）。 LLMRec的有效性源自LLMs固有的开放世界知识和推理能力。 LLMRec通过基于用户互动数据的指导调整获得推荐功能。 然而，为了保护用户隐私并优化效用，LLMRec还必须有意忘记特定用户数据，这通常称为推荐遗忘。 在LLMs时代，推荐遗忘在\textit{效率}和\textit{有效性}方面为LLMRec带来了新挑战。 现有的遗忘方法需要更新LLMRec中数十亿参数，这是昂贵且耗时的。 此外，它们在遗忘过程中总是影响模型效用。 为此，我们提出了\textbf{E2URec}，第一

    arXiv:2403.03536v1 Announce Type: cross  Abstract: The significant advancements in large language models (LLMs) give rise to a promising research direction, i.e., leveraging LLMs as recommenders (LLMRec). The efficacy of LLMRec arises from the open-world knowledge and reasoning capabilities inherent in LLMs. LLMRec acquires the recommendation capabilities through instruction tuning based on user interaction data. However, in order to protect user privacy and optimize utility, it is also crucial for LLMRec to intentionally forget specific user data, which is generally referred to as recommendation unlearning. In the era of LLMs, recommendation unlearning poses new challenges for LLMRec in terms of \textit{inefficiency} and \textit{ineffectiveness}. Existing unlearning methods require updating billions of parameters in LLMRec, which is costly and time-consuming. Besides, they always impact the model utility during the unlearning process. To this end, we propose \textbf{E2URec}, the first
    
[^3]: MuGI:通过与大型语言模型的多文本生成集成增强信息检索

    MuGI: Enhancing Information Retrieval through Multi-Text Generation Intergration with Large Language Models. (arXiv:2401.06311v1 [cs.IR])

    [http://arxiv.org/abs/2401.06311](http://arxiv.org/abs/2401.06311)

    MuGI是一个简单而有效的多文本生成集成框架，它通过与大型语言模型合作生成多个伪参考文献，并将其与查询集成以提升信息检索性能。在实验中，MuGI模型在TREC DL数据集上的BM25性能上取得了18%以上的增强，并在BEIR上提高了7.5%。

    

    大型语言模型（LLM）已经成为语言技术领域的一个重要力量。它们强大的推理能力和广泛的知识库使其在各个自然语言处理领域，包括信息检索（IR）方面具备了出色的零-shot泛化能力。在本文中，我们对LLM生成的文档在IR中的实用性进行了深入研究。我们引入了一个简单而有效的框架，即多文本生成集成（MuGI），来增强现有的IR方法。具体而言，我们引导LLM生成多个伪参考文献，并将其与查询进行集成以进行检索。无需训练的MuGI模型超越了现有的查询扩展策略，在TREC DL数据集上的BM25上取得了新的标准，并在BEIR上提高了7.5%。通过MuGI，我们构建了一个快速且高保真度的重排序方法。

    Large Language Models (LLMs) have emerged as a pivotal force in language technology. Their robust reasoning capabilities and expansive knowledge repositories have enabled exceptional zero-shot generalization abilities across various facets of the natural language processing field, including information retrieval (IR). In this paper, we conduct an in-depth investigation into the utility of documents generated by LLMs for IR. We introduce a simple yet effective framework, Multi-Text Generation Integration (MuGI), to augment existing IR methodologies. Specifically, we prompt LLMs to generate multiple pseudo references and integrate with query for retrieval. The training-free MuGI model eclipses existing query expansion strategies, setting a new standard in sparse retrieval. It outstrips supervised counterparts like ANCE and DPR, achieving a notable over 18% enhancement in BM25 on the TREC DL dataset and a 7.5% increase on BEIR. Through MuGI, we have forged a rapid and high-fidelity re-ran
    
[^4]: Autumn：基于LSM-tree的可扩展的面向读操作优化的键值存储引擎

    Autumn: A Scalable Read Optimized LSM-tree based Key-Value Stores with Fast Point and Range Read Speed. (arXiv:2305.05074v1 [cs.DB])

    [http://arxiv.org/abs/2305.05074](http://arxiv.org/abs/2305.05074)

    Autumn是一个可扩展的、面向读操作优化的LSM-tree键值存储引擎，其创新之处在于通过动态调整相邻两层之间的容量比来不断提高读性能，使得点读和区间读成本从之前最优的$O(logN)$复杂度优化到了$O(\sqrt{logN})$。

    

    基于Log Structured Merge Trees (LSM-tree)的键值存储引擎被广泛应用于许多存储系统中，以支持更新、点读和区间读等各种操作。本文中，我们提出了一个名为Autumn的可扩展的、面向读操作优化的基于LSM-tree的键值存储引擎，它具有最少的点读和区间读成本。通过动态调整相邻两层之间的容量比来不断提高读性能，点读和区间读成本从之前最优的$O(logN)$复杂度优化到了$O(\sqrt{logN})$，并应用了新的Garnering合并策略。Autumn是一个可扩展的、面向读操作优化的LSM-tree键值存储引擎。

    The Log Structured Merge Trees (LSM-tree) based key-value stores are widely used in many storage systems to support a variety of operations such as updates, point reads, and range reads. Traditionally, LSM-tree's merge policy organizes data into multiple levels of exponentially increasing capacity to support high-speed writes. However, we contend that the traditional merge policies are not optimized for reads. In this work, we present Autumn, a scalable and read optimized LSM-tree based key-value stores with minimal point and range read cost. The key idea in improving the read performance is to dynamically adjust the capacity ratio between two adjacent levels as more data are stored. As a result, smaller levels gradually increase their capacities and merge more often. In particular, the point and range read cost improves from the previous best known $O(logN)$ complexity to $O(\sqrt{logN})$ in Autumn by applying the new novel Garnering merge policy. While Garnering merge policy optimize
    

