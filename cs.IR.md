# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [VM-Rec: A Variational Mapping Approach for Cold-start User Recommendation.](http://arxiv.org/abs/2311.01304) | VM-Rec是一种用于解决冷启动用户推荐问题的变分映射方法，该方法基于生成表达能力强的嵌入的现有用户的互动，从而模拟生成冷启动用户的嵌入过程。 |
| [^2] | [Uncertainty-Aware Multi-View Visual Semantic Embedding.](http://arxiv.org/abs/2309.08154) | 这篇论文提出了一种不确定性感知的多视图视觉语义嵌入框架，在图像-文本检索中有效地利用语义信息进行相似性度量，通过引入不确定性感知损失函数，充分利用二进制标签的不确定性，并将整体匹配分解为多个视图-文本匹配。 |
| [^3] | [Sustainable Transparency in Recommender Systems: Bayesian Ranking of Images for Explainability.](http://arxiv.org/abs/2308.01196) | 这项研究旨在实现推荐系统的可持续透明性，并提出了一种使用贝叶斯排名图像进行个性化解释的方法，以最大化透明度和用户信任。 |
| [^4] | [Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations.](http://arxiv.org/abs/2307.05722) | 本论文探索了大规模语言模型在在线职位推荐中对图数据的理解能力，并提出了新的框架来分析行为图，发现其中的潜在模式和关系。 |

# 详细

[^1]: VM-Rec：一种用于冷启动用户推荐的变分映射方法

    VM-Rec: A Variational Mapping Approach for Cold-start User Recommendation. (arXiv:2311.01304v1 [cs.IR])

    [http://arxiv.org/abs/2311.01304](http://arxiv.org/abs/2311.01304)

    VM-Rec是一种用于解决冷启动用户推荐问题的变分映射方法，该方法基于生成表达能力强的嵌入的现有用户的互动，从而模拟生成冷启动用户的嵌入过程。

    

    冷启动问题是大多数推荐系统面临的共同挑战。传统的推荐模型在冷启动用户的互动非常有限时通常难以生成具有足够表达能力的嵌入。此外，缺乏用户的辅助内容信息加剧了挑战的存在，使得大多数冷启动方法难以应用。为了解决这个问题，我们观察到，如果模型能够为相对更多互动的现有用户生成具有表达能力的嵌入，这些用户最初也是冷启动用户，那么我们可以建立一个从少量初始互动到具有表达能力的嵌入的映射，模拟为冷启动用户生成嵌入的过程。基于这个观察，我们提出了一种变分映射方法用于冷启动用户推荐（VM-Rec）。首先，我们根据冷启动用户的初始互动生成个性化的映射函数，并进行参数优化。

    The cold-start problem is a common challenge for most recommender systems. With extremely limited interactions of cold-start users, conventional recommender models often struggle to generate embeddings with sufficient expressivity. Moreover, the absence of auxiliary content information of users exacerbates the presence of challenges, rendering most cold-start methods difficult to apply. To address this issue, our motivation is based on the observation that if a model can generate expressive embeddings for existing users with relatively more interactions, who were also initially cold-start users, then we can establish a mapping from few initial interactions to expressive embeddings, simulating the process of generating embeddings for cold-start users. Based on this motivation, we propose a Variational Mapping approach for cold-start user Recommendation (VM-Rec). Firstly, we generate a personalized mapping function for cold-start users based on their initial interactions, and parameters 
    
[^2]: 不确定性感知的多视图视觉语义嵌入

    Uncertainty-Aware Multi-View Visual Semantic Embedding. (arXiv:2309.08154v1 [cs.CV])

    [http://arxiv.org/abs/2309.08154](http://arxiv.org/abs/2309.08154)

    这篇论文提出了一种不确定性感知的多视图视觉语义嵌入框架，在图像-文本检索中有效地利用语义信息进行相似性度量，通过引入不确定性感知损失函数，充分利用二进制标签的不确定性，并将整体匹配分解为多个视图-文本匹配。

    

    图像-文本检索的关键挑战是有效地利用语义信息来衡量视觉和语言数据之间的相似性。然而，使用实例级的二进制标签，其中每个图像与一个文本配对，无法捕捉不同语义单元之间的多个对应关系，从而导致多模态语义理解中的不确定性。尽管最近的研究通过更复杂的模型结构或预训练技术捕捉了细粒度信息，但很少有研究直接建模对应关系的不确定性以充分利用二进制标签。为了解决这个问题，我们提出了一种不确定性感知的多视图视觉语义嵌入（UAMVSE）框架，该框架将整体图像-文本匹配分解为多个视图-文本匹配。我们的框架引入了一种不确定性感知损失函数（UALoss），通过自适应地建模每个视图-文本对应关系的不确定性来计算每个视图-文本损失的权重。

    The key challenge in image-text retrieval is effectively leveraging semantic information to measure the similarity between vision and language data. However, using instance-level binary labels, where each image is paired with a single text, fails to capture multiple correspondences between different semantic units, leading to uncertainty in multi-modal semantic understanding. Although recent research has captured fine-grained information through more complex model structures or pre-training techniques, few studies have directly modeled uncertainty of correspondence to fully exploit binary labels. To address this issue, we propose an Uncertainty-Aware Multi-View Visual Semantic Embedding (UAMVSE)} framework that decomposes the overall image-text matching into multiple view-text matchings. Our framework introduce an uncertainty-aware loss function (UALoss) to compute the weighting of each view-text loss by adaptively modeling the uncertainty in each view-text correspondence. Different we
    
[^3]: 可持续透明的推荐系统: 用于解释性的贝叶斯图像排名

    Sustainable Transparency in Recommender Systems: Bayesian Ranking of Images for Explainability. (arXiv:2308.01196v1 [cs.IR])

    [http://arxiv.org/abs/2308.01196](http://arxiv.org/abs/2308.01196)

    这项研究旨在实现推荐系统的可持续透明性，并提出了一种使用贝叶斯排名图像进行个性化解释的方法，以最大化透明度和用户信任。

    

    推荐系统在现代世界中变得至关重要，通常指导用户找到相关的内容或产品，并对用户和公民的决策产生重大影响。然而，确保这些系统的透明度和用户信任仍然是一个挑战；个性化解释已经成为一个解决方案，为推荐提供理由。在生成个性化解释的现有方法中，使用用户创建的视觉内容是一个特别有潜力的选项，有潜力最大化透明度和用户信任。然而，现有模型在这个背景下解释推荐时存在一些限制：可持续性是一个关键问题，因为它们经常需要大量的计算资源，导致的碳排放量与它们被整合到推荐系统中相当。此外，大多数模型使用的替代学习目标与排名最有效的目标不一致。

    Recommender Systems have become crucial in the modern world, commonly guiding users towards relevant content or products, and having a large influence over the decisions of users and citizens. However, ensuring transparency and user trust in these systems remains a challenge; personalized explanations have emerged as a solution, offering justifications for recommendations. Among the existing approaches for generating personalized explanations, using visual content created by the users is one particularly promising option, showing a potential to maximize transparency and user trust. Existing models for explaining recommendations in this context face limitations: sustainability has been a critical concern, as they often require substantial computational resources, leading to significant carbon emissions comparable to the Recommender Systems where they would be integrated. Moreover, most models employ surrogate learning goals that do not align with the objective of ranking the most effect
    
[^4]: 探索大规模语言模型在在线职位推荐中对图数据的理解

    Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations. (arXiv:2307.05722v1 [cs.AI])

    [http://arxiv.org/abs/2307.05722](http://arxiv.org/abs/2307.05722)

    本论文探索了大规模语言模型在在线职位推荐中对图数据的理解能力，并提出了新的框架来分析行为图，发现其中的潜在模式和关系。

    

    大规模语言模型（LLMs）在各个领域展示了其出色的能力，彻底改变了自然语言处理任务。然而，它们在职位推荐中对行为图的理解潜力仍然未被充分探索。本文旨在揭示大规模语言模型在理解行为图方面的能力，并利用这种理解来提升在线招聘中的推荐，包括促进非分布式的应用。我们提出了一个新的框架，利用大规模语言模型提供的丰富上下文信息和语义表示来分析行为图并揭示其中的潜在模式和关系。具体而言，我们提出了一个元路径提示构造器，利用LLM推荐器首次理解行为图，并设计了相应的路径增强模块来缓解基于路径的序列输入引入的提示偏差。通过利用将LM的特点引入到行为图的大规模数据分析中，我们取得了显著的实验结果，证明了我们提出的方法的有效性和性能。

    Large Language Models (LLMs) have revolutionized natural language processing tasks, demonstrating their exceptional capabilities in various domains. However, their potential for behavior graph understanding in job recommendations remains largely unexplored. This paper focuses on unveiling the capability of large language models in understanding behavior graphs and leveraging this understanding to enhance recommendations in online recruitment, including the promotion of out-of-distribution (OOD) application. We present a novel framework that harnesses the rich contextual information and semantic representations provided by large language models to analyze behavior graphs and uncover underlying patterns and relationships. Specifically, we propose a meta-path prompt constructor that leverages LLM recommender to understand behavior graphs for the first time and design a corresponding path augmentation module to alleviate the prompt bias introduced by path-based sequence input. By leveragin
    

