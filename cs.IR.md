# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLM4SBR: A Lightweight and Effective Framework for Integrating Large Language Models in Session-based Recommendation](https://arxiv.org/abs/2402.13840) | 该研究提出了LLM4SBR框架，是第一个适合在基于会话的推荐中集成大型语言模型的轻量且有效框架。 |
| [^2] | [LD4MRec: Simplifying and Powering Diffusion Model for Multimedia Recommendation.](http://arxiv.org/abs/2309.15363) | LD4MRec是一种简化和加强多媒体推荐的扩散模型，解决了行为数据噪声对推荐性能的负面影响、经典扩散模型计算量过大以及现有反向过程不适用于离散行为数据的挑战。 |

# 详细

[^1]: LLM4SBR: 一个轻量且有效的框架，用于在基于会话的推荐中集成大型语言模型

    LLM4SBR: A Lightweight and Effective Framework for Integrating Large Language Models in Session-based Recommendation

    [https://arxiv.org/abs/2402.13840](https://arxiv.org/abs/2402.13840)

    该研究提出了LLM4SBR框架，是第一个适合在基于会话的推荐中集成大型语言模型的轻量且有效框架。

    

    传统的基于会话的推荐(SBR)利用来自匿名用户的会话行为序列进行推荐。虽然这种策略非常高效，但牺牲了商品的固有语义信息，使模型难以理解会话的真正意图，导致推荐结果缺乏可解释性。近年来，大型语言模型(LLMs)在各个领域蓬勃发展，为解决上述挑战带来了一线希望。受LLMs影响，探讨LLMs与推荐系统(RS)集成的研究如雨后春笋般涌现。然而，受限于高时间和空间成本，以及会话数据短暂且匿名的特性，第一个适合工业部署的LLM推荐框架在SBR领域尚未出现。为了解决上述挑战，我们...

    arXiv:2402.13840v1 Announce Type: cross  Abstract: Traditional session-based recommendation (SBR) utilizes session behavior sequences from anonymous users for recommendation. Although this strategy is highly efficient, it sacrifices the inherent semantic information of the items, making it difficult for the model to understand the true intent of the session and resulting in a lack of interpretability in the recommended results. Recently, large language models (LLMs) have flourished across various domains, offering a glimpse of hope in addressing the aforementioned challenges. Inspired by the impact of LLMs, research exploring the integration of LLMs with the Recommender system (RS) has surged like mushrooms after rain. However, constrained by high time and space costs, as well as the brief and anonymous nature of session data, the first LLM recommendation framework suitable for industrial deployment has yet to emerge in the field of SBR. To address the aforementioned challenges, we hav
    
[^2]: LD4MRec:简化和加强多媒体推荐的扩散模型

    LD4MRec: Simplifying and Powering Diffusion Model for Multimedia Recommendation. (arXiv:2309.15363v1 [cs.IR])

    [http://arxiv.org/abs/2309.15363](http://arxiv.org/abs/2309.15363)

    LD4MRec是一种简化和加强多媒体推荐的扩散模型，解决了行为数据噪声对推荐性能的负面影响、经典扩散模型计算量过大以及现有反向过程不适用于离散行为数据的挑战。

    

    多媒体推荐旨在根据历史行为数据和项目的多模态信息预测用户的未来行为。然而，行为数据中的噪声，产生于与不感兴趣的项目的非预期用户交互，对推荐性能产生不利影响。最近，扩散模型实现了高质量的信息生成，其中反向过程根据受损状态迭代地推断未来信息。它满足了在嘈杂条件下的预测任务需求，并激发了对其在预测用户行为方面的应用的探索。然而，还需要解决几个挑战：1）经典扩散模型需要过多的计算，这不符合推荐系统的效率要求。2）现有的反向过程主要设计用于连续型数据，而行为信息是离散型的。因此，需要有效的方法来生成离散行为。

    Multimedia recommendation aims to predict users' future behaviors based on historical behavioral data and item's multimodal information. However, noise inherent in behavioral data, arising from unintended user interactions with uninteresting items, detrimentally impacts recommendation performance. Recently, diffusion models have achieved high-quality information generation, in which the reverse process iteratively infers future information based on the corrupted state. It meets the need of predictive tasks under noisy conditions, and inspires exploring their application to predicting user behaviors. Nonetheless, several challenges must be addressed: 1) Classical diffusion models require excessive computation, which does not meet the efficiency requirements of recommendation systems. 2) Existing reverse processes are mainly designed for continuous data, whereas behavioral information is discrete in nature. Therefore, an effective method is needed for the generation of discrete behaviora
    

