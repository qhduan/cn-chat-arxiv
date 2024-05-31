# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Croissant: A Metadata Format for ML-Ready Datasets](https://arxiv.org/abs/2403.19546) | Croissant是一种面向机器学习数据集的元数据格式，使数据集更易发现、可移植和互操作，有助于解决ML数据管理和负责任AI中的重要挑战。 |
| [^2] | [Lookahead: An Inference Acceleration Framework for Large Language Model with Lossless Generation Accuracy.](http://arxiv.org/abs/2312.12728) | 本研究介绍了一种通用的推理加速框架，用于提高大型语言模型（LLMs）的推理速度，并在保持生成准确性的同时降低成本。该框架在支付宝的检索增强生成（RAG）系统中得到了应用。 |
| [^3] | [A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models.](http://arxiv.org/abs/2310.09497) | 本研究通过评估现有的逐点、逐对和列表提示方法，揭示了大规模语言模型在零样本排名任务中的效果和效率的权衡。我们发现逐点方法的效率高但效果差，逐对方法效果好但计算复杂。为了提高效率，我们提出了一种集合提示方法。 |
| [^4] | [Better Generalization with Semantic IDs: A case study in Ranking for Recommendations.](http://arxiv.org/abs/2306.08121) | 本文提出使用语义ID解决推荐系统中的物品冷启动问题，这些ID是从内容嵌入中学习的，可以捕捉概念的层次关系，相较于完全消除ID特征的方法，语义ID能更好地提高推荐质量。 |

# 详细

[^1]: Croissant：一种面向机器学习数据集的元数据格式

    Croissant: A Metadata Format for ML-Ready Datasets

    [https://arxiv.org/abs/2403.19546](https://arxiv.org/abs/2403.19546)

    Croissant是一种面向机器学习数据集的元数据格式，使数据集更易发现、可移植和互操作，有助于解决ML数据管理和负责任AI中的重要挑战。

    

    数据是机器学习（ML）的关键资源，但处理数据仍然是一个主要的摩擦点。本文介绍了Croissant，一种用于数据集的元数据格式，简化了数据被ML工具和框架使用的方式。Croissant使数据集更易发现、可移植和互操作，从而解决了ML数据管理和负责任AI中的重要挑战。Croissant已得到几个流行数据集库的支持，涵盖数十万个数据集，可以加载到最流行的ML框架中。

    arXiv:2403.19546v1 Announce Type: cross  Abstract: Data is a critical resource for Machine Learning (ML), yet working with data remains a key friction point. This paper introduces Croissant, a metadata format for datasets that simplifies how data is used by ML tools and frameworks. Croissant makes datasets more discoverable, portable and interoperable, thereby addressing significant challenges in ML data management and responsible AI. Croissant is already supported by several popular dataset repositories, spanning hundreds of thousands of datasets, ready to be loaded into the most popular ML frameworks.
    
[^2]: Lookahead:一种用于具有无损生成准确性的大型语言模型的推理加速框架

    Lookahead: An Inference Acceleration Framework for Large Language Model with Lossless Generation Accuracy. (arXiv:2312.12728v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2312.12728](http://arxiv.org/abs/2312.12728)

    本研究介绍了一种通用的推理加速框架，用于提高大型语言模型（LLMs）的推理速度，并在保持生成准确性的同时降低成本。该框架在支付宝的检索增强生成（RAG）系统中得到了应用。

    

    随着大型语言模型（LLMs）在各种任务中取得了重大进展，如问答、翻译、文本摘要和对话系统，尤其是对于像支付宝这样为数十亿用户提供重要金融产品的需要准确信息的情况，信息的准确性变得至关重要。为了解决这个问题，支付宝开发了一种称为检索增强生成（RAG）系统的方法，该系统将LLMs与最准确和最新的信息相结合。然而，对于为数百万用户提供服务的真实产品来说，LLMs的推理速度成为一个关键因素，而不仅仅是一个实验性的模型。因此，本文提出了一种通用的推理加速框架，通过加速推理过程，实现了我们的RAG系统的速度大幅提升和成本降低，同时保持着无损的生成准确性。在传统的推理过程中，每个令牌都由LLMs按顺序生成，导致的时间消耗与生成的令牌数成正比。

    As Large Language Models (LLMs) have made significant advancements across various tasks, such as question answering, translation, text summarization, and dialogue systems, the need for accuracy in information becomes crucial, especially for serious financial products serving billions of users like Alipay. To address this, Alipay has developed a Retrieval-Augmented Generation (RAG) system that grounds LLMs on the most accurate and up-to-date information. However, for a real-world product serving millions of users, the inference speed of LLMs becomes a critical factor compared to a mere experimental model.  Hence, this paper presents a generic framework for accelerating the inference process, resulting in a substantial increase in speed and cost reduction for our RAG system, with lossless generation accuracy. In the traditional inference process, each token is generated sequentially by the LLM, leading to a time consumption proportional to the number of generated tokens. To enhance this 
    
[^3]: 一种用于大规模语言模型的零样本排名的高效集合方法

    A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models. (arXiv:2310.09497v1 [cs.IR])

    [http://arxiv.org/abs/2310.09497](http://arxiv.org/abs/2310.09497)

    本研究通过评估现有的逐点、逐对和列表提示方法，揭示了大规模语言模型在零样本排名任务中的效果和效率的权衡。我们发现逐点方法的效率高但效果差，逐对方法效果好但计算复杂。为了提高效率，我们提出了一种集合提示方法。

    

    大规模语言模型（LLM）在零样本文档排名任务中展示了惊人的有效性。针对基于LLM的零样本排名，已经提出了逐点，逐对和列表提示方法。我们的研究首先在一个一致的实验框架内进行了对这些现有方法的彻底评估，考虑了模型大小，标记消耗，延迟等因素。这种首次的比较评估让我们能够确定每种方法在效果和效率之间固有的权衡。我们发现，逐点方法在效率上得分很高，但在有效性上存在问题。相反，逐对方法表现出优越的有效性，但计算复杂度较高。为了进一步提高基于LLM的零样本排名的效率，我们提出了一种新颖的集合提示方法。我们的方法减少了LLM推理的次数和排名过程中的提示标记消耗量。

    Large Language Models (LLMs) demonstrate impressive effectiveness in zero-shot document ranking tasks. Pointwise, Pairwise, and Listwise prompting approaches have been proposed for LLM-based zero-shot ranking. Our study begins by thoroughly evaluating these existing approaches within a consistent experimental framework, considering factors like model size, token consumption, latency, among others. This first-of-its-kind comparative evaluation of these approaches allows us to identify the trade-offs between effectiveness and efficiency inherent in each approach. We find that while Pointwise approaches score high on efficiency, they suffer from poor effectiveness. Conversely, Pairwise approaches demonstrate superior effectiveness but incur high computational overhead. To further enhance the efficiency of LLM-based zero-shot ranking, we propose a novel Setwise prompting approach. Our approach reduces the number of LLM inferences and the amount of prompt token consumption during the rankin
    
[^4]: 使用语义ID进行更好的泛化：推荐排名的案例研究

    Better Generalization with Semantic IDs: A case study in Ranking for Recommendations. (arXiv:2306.08121v1 [cs.IR])

    [http://arxiv.org/abs/2306.08121](http://arxiv.org/abs/2306.08121)

    本文提出使用语义ID解决推荐系统中的物品冷启动问题，这些ID是从内容嵌入中学习的，可以捕捉概念的层次关系，相较于完全消除ID特征的方法，语义ID能更好地提高推荐质量。

    

    在推荐模型中，训练好的物品表示是至关重要的。通常，一项商品会被分配一个唯一的随机生成的ID，并且通常会通过学习与随机ID值相对应的嵌入来表示。虽然这种方法被广泛使用，但在物品数量大且物品服从幂律分布的情况下——这是真实世界推荐系统的典型特征——会有一定局限性。这会导致物品冷启动问题，模型无法对尾部和以前未见过的物品进行可靠的推荐。完全消除这些ID特征及其学习的嵌入以解决冷启动问题会严重降低推荐质量。基于内容的物品嵌入更为可靠，但对于用户过去的物品交互序列来说，它们成本高且使用困难。本文中，我们使用语义ID来表示离散的物品，这些ID是通过使用RQ-VAE从内容嵌入中学习的，可以捕捉概念的层次关系。

    Training good representations for items is critical in recommender models. Typically, an item is assigned a unique randomly generated ID, and is commonly represented by learning an embedding corresponding to the value of the random ID. Although widely used, this approach have limitations when the number of items are large and items are power-law distributed -- typical characteristics of real-world recommendation systems. This leads to the item cold-start problem, where the model is unable to make reliable inferences for tail and previously unseen items. Removing these ID features and their learned embeddings altogether to combat cold-start issue severely degrades the recommendation quality. Content-based item embeddings are more reliable, but they are expensive to store and use, particularly for users' past item interaction sequence. In this paper, we use Semantic IDs, a compact discrete item representations learned from content embeddings using RQ-VAE that captures hierarchy of concep
    

