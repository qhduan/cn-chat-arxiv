# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DiFashion: Towards Personalized Outfit Generation](https://arxiv.org/abs/2402.17279) | 引入生成式服装推荐任务（GOR），旨在合成一组时尚图片并组装成视觉和谐的、定制给个人用户的服装。 |
| [^2] | [Non-autoregressive Generative Models for Reranking Recommendation](https://arxiv.org/abs/2402.06871) | 本研究提出了一个非自回归的生成模型用于排序推荐，在多阶段推荐系统中扮演关键角色。该模型旨在提高效率和效果，并解决稀疏训练样本和动态候选项对模型收敛性的挑战。 |
| [^3] | [TransGNN: Harnessing the Collaborative Power of Transformers and Graph Neural Networks for Recommender Systems.](http://arxiv.org/abs/2308.14355) | TransGNN是一种将Transformer和GNN层交替结合以相互增强其能力的新型模型，用于解决当前基于GNN的推荐系统面临的感受域有限和存在噪音连接的挑战。 |
| [^4] | [Anonymity at Risk? Assessing Re-Identification Capabilities of Large Language Models.](http://arxiv.org/abs/2308.11103) | 本研究评估了大型语言模型在重新识别匿名个人方面的能力，并发现模型大小、输入长度和指令调整是最重要的决定因素。 |
| [^5] | [Fair Learning to Rank with Distribution-free Risk Control.](http://arxiv.org/abs/2306.07188) | 本论文提出了一种新的后置模型无关方法，公平LTR-RC，它不需要昂贵的训练，在保证公平性的同时，还能在效用和公平之间实现有效的权衡。 |
| [^6] | [The Search for Stability: Learning Dynamics of Strategic Publishers with Initial Documents.](http://arxiv.org/abs/2305.16695) | 本研究在信息检索博弈论模型中提出了相对排名原则（RRP）作为替代排名原则，以达成更稳定的搜索生态系统，并提供了理论和实证证据证明其学习动力学收敛性，同时展示了可能的出版商-用户权衡。 |

# 详细

[^1]: DiFashion: 迈向个性化服装生成

    DiFashion: Towards Personalized Outfit Generation

    [https://arxiv.org/abs/2402.17279](https://arxiv.org/abs/2402.17279)

    引入生成式服装推荐任务（GOR），旨在合成一组时尚图片并组装成视觉和谐的、定制给个人用户的服装。

    

    服装推荐（OR）在时尚领域的发展经历了两个不同阶段：预定义的服装推荐和个性化的服装组合。虽然取得了这些进展，但两个阶段都面临现有时尚产品带来的限制，阻碍了它们满足用户多样化时尚需求的有效性。AI生成内容的出现为OR克服这些约束铺平了道路，展示了个性化服装生成的潜力。为了追求这一目标，我们引入了一项名为生成式服装推荐（GOR）的创新任务，其目标是合成一组时尚图片，并将它们组装成视觉和谐的、定制给个人用户的服装。GOR的主要目标集中在实现生成服装的高保真度、兼容性和个性化。为实现这些目标，我们提出了DiFashion，一个生成式服装推荐

    arXiv:2402.17279v1 Announce Type: new  Abstract: The evolution of Outfit Recommendation (OR) in the realm of fashion has progressed through two distinct phases: Pre-defined Outfit Recommendation and Personalized Outfit Composition. Despite these advancements, both phases face limitations imposed by existing fashion products, hindering their effectiveness in meeting users' diverse fashion needs. The emergence of AI-generated content has paved the way for OR to overcome these constraints, demonstrating the potential for personalized outfit generation.   In pursuit of this, we introduce an innovative task named Generative Outfit Recommendation (GOR), with the goal of synthesizing a set of fashion images and assembling them to form visually harmonious outfits customized to individual users. The primary objectives of GOR revolve around achieving high fidelity, compatibility, and personalization of the generated outfits. To accomplish these, we propose DiFashion, a generative outfit recommen
    
[^2]: 非自回归的生成模型用于排序推荐

    Non-autoregressive Generative Models for Reranking Recommendation

    [https://arxiv.org/abs/2402.06871](https://arxiv.org/abs/2402.06871)

    本研究提出了一个非自回归的生成模型用于排序推荐，在多阶段推荐系统中扮演关键角色。该模型旨在提高效率和效果，并解决稀疏训练样本和动态候选项对模型收敛性的挑战。

    

    在多阶段推荐系统中，重新排序通过建模项目之间的内部相关性起到了至关重要的作用。重新排序的关键挑战在于在排列的组合空间中探索最佳序列。最近的研究提出了生成器-评估器学习范式，生成器生成多个可行序列，评估器基于估计的列表得分选择最佳序列。生成器至关重要，而生成模型非常适合生成器函数。当前的生成模型采用自回归策略进行序列生成。然而，在实时工业系统中部署自回归模型是具有挑战性的。因此，我们提出了一个非自回归生成模型用于排序推荐（NAR4Rec），以提高效率和效果。为了解决与稀疏训练样本和动态候选项对模型收敛性的挑战，我们引入了一个m

    In a multi-stage recommendation system, reranking plays a crucial role by modeling the intra-list correlations among items.The key challenge of reranking lies in the exploration of optimal sequences within the combinatorial space of permutations. Recent research proposes a generator-evaluator learning paradigm, where the generator generates multiple feasible sequences and the evaluator picks out the best sequence based on the estimated listwise score. Generator is of vital importance, and generative models are well-suited for the generator function. Current generative models employ an autoregressive strategy for sequence generation. However, deploying autoregressive models in real-time industrial systems is challenging. Hence, we propose a Non-AutoRegressive generative model for reranking Recommendation (NAR4Rec) designed to enhance efficiency and effectiveness. To address challenges related to sparse training samples and dynamic candidates impacting model convergence, we introduce a m
    
[^3]: TransGNN: 利用Transformer和图神经网络的协同能力来做推荐系统

    TransGNN: Harnessing the Collaborative Power of Transformers and Graph Neural Networks for Recommender Systems. (arXiv:2308.14355v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2308.14355](http://arxiv.org/abs/2308.14355)

    TransGNN是一种将Transformer和GNN层交替结合以相互增强其能力的新型模型，用于解决当前基于GNN的推荐系统面临的感受域有限和存在噪音连接的挑战。

    

    图神经网络(GNNs)已经被证明是推荐系统中有前途的解决方案，通过对用户-物品交互图进行建模来进行协同过滤(CF)。现有基于GNN的推荐系统的核心是通过在用户-物品交互边上进行递归消息传递来改进编码嵌入。尽管它们已经证明是有效的，但是当前基于GNN的方法面临着有限的感受域和存在噪音 "兴趣无关" 连接的挑战。相比之下，基于Transformer的方法在自适应和全局信息聚合方面表现出色。然而，它们在捕捉复杂、纠缠的结构信息方面在大规模交互图中的应用受到困扰。在本文中，我们提出了TransGNN，这是一种新颖的模型，通过交替地结合Transformer和GNN层来相互增强它们的能力。

    Graph Neural Networks (GNNs) have emerged as promising solutions for collaborative filtering (CF) through the modeling of user-item interaction graphs. The nucleus of existing GNN-based recommender systems involves recursive message passing along user-item interaction edges to refine encoded embeddings. Despite their demonstrated effectiveness, current GNN-based methods encounter challenges of limited receptive fields and the presence of noisy ``interest-irrelevant'' connections. In contrast, Transformer-based methods excel in aggregating information adaptively and globally. Nevertheless, their application to large-scale interaction graphs is hindered by inherent complexities and challenges in capturing intricate, entangled structural information. In this paper, we propose TransGNN, a novel model that integrates Transformer and GNN layers in an alternating fashion to mutually enhance their capabilities. Specifically, TransGNN leverages Transformer layers to broaden the receptive field 
    
[^4]: 大型语言模型的再识别能力：匿名面临风险吗？

    Anonymity at Risk? Assessing Re-Identification Capabilities of Large Language Models. (arXiv:2308.11103v1 [cs.CL])

    [http://arxiv.org/abs/2308.11103](http://arxiv.org/abs/2308.11103)

    本研究评估了大型语言模型在重新识别匿名个人方面的能力，并发现模型大小、输入长度和指令调整是最重要的决定因素。

    

    在欧盟和瑞士，法院裁决中自然人和法人的匿名性是隐私保护的关键方面。随着大型语言模型（LLMs）的出现，对于匿名人员的大规模再识别的担忧日益增长。根据瑞士联邦最高法院的要求，我们通过使用来自瑞士联邦最高法院的实际法律数据构建了一个概念验证，来探讨LLMs重新识别法院裁决中个人的潜力。在最初的实验之后，我们构建了一个经过匿名化处理的维基百科数据集，作为一个更严格的测试场地来进一步研究研究结果。通过引入并应用文本中再识别人员的新任务，我们还引入了新的性能衡量指标。我们系统地分析了影响成功再识别的因素，确定模型大小、输入长度和指令调整是最重要的决定因素之一。尽管在匿名化处理后，LLMs在重新识别上的成功率很高，但在某些情况下仍然存在风险。

    Anonymity of both natural and legal persons in court rulings is a critical aspect of privacy protection in the European Union and Switzerland. With the advent of LLMs, concerns about large-scale re-identification of anonymized persons are growing. In accordance with the Federal Supreme Court of Switzerland, we explore the potential of LLMs to re-identify individuals in court rulings by constructing a proof-of-concept using actual legal data from the Swiss federal supreme court. Following the initial experiment, we constructed an anonymized Wikipedia dataset as a more rigorous testing ground to further investigate the findings. With the introduction and application of the new task of re-identifying people in texts, we also introduce new metrics to measure performance. We systematically analyze the factors that influence successful re-identifications, identifying model size, input length, and instruction tuning among the most critical determinants. Despite high re-identification rates on
    
[^5]: 无分布风险控制的公平学习排序

    Fair Learning to Rank with Distribution-free Risk Control. (arXiv:2306.07188v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.07188](http://arxiv.org/abs/2306.07188)

    本论文提出了一种新的后置模型无关方法，公平LTR-RC，它不需要昂贵的训练，在保证公平性的同时，还能在效用和公平之间实现有效的权衡。

    

    在线经济中，学习排序方法对用户和物品提供者至关重要。LTR模型的公平性对于按比例分配曝光至关重要。当具有相同相关性的项接收略有不同的分数时，确定性排名模型可能导致不公平的曝光分配。随机LTR模型，包括Plackett-Luce（PL）模型，解决了公平性问题，但在计算成本和性能保证方面存在局限性。为了克服这些局限性，我们提出了公平LTR-RC，一种新的后置模型无关方法。公平LTR-RC利用预先训练的评分函数创建随机LTR模型，消除了昂贵的训练需求。此外，公平LTR-RC使用无分布式风险控制框架对用户指定的效用提供有限的样本保证。通过另外结合Thresholded PL（TPL）模型，我们能够在效用和公平之间实现有效的权衡。实验结果显示，FairLTR-RC在公平性和效用性指标上优于现有方法。

    Learning to Rank (LTR) methods are vital in online economies, affecting users and item providers. Fairness in LTR models is crucial to allocate exposure proportionally to item relevance. The deterministic ranking model can lead to unfair exposure distribution when items with the same relevance receive slightly different scores. Stochastic LTR models, incorporating the Plackett-Luce (PL) model, address fairness issues but have limitations in computational cost and performance guarantees. To overcome these limitations, we propose FairLTR-RC, a novel post-hoc model-agnostic method. FairLTR-RC leverages a pretrained scoring function to create a stochastic LTR model, eliminating the need for expensive training. Furthermore, FairLTR-RC provides finite-sample guarantees on a user-specified utility using distribution-free risk control framework. By additionally incorporating the Thresholded PL (TPL) model, we are able to achieve an effective trade-off between utility and fairness. Experimental
    
[^6]: 寻求稳定性：具有初始文件的战略出版商的学习动态的研究

    The Search for Stability: Learning Dynamics of Strategic Publishers with Initial Documents. (arXiv:2305.16695v1 [cs.GT])

    [http://arxiv.org/abs/2305.16695](http://arxiv.org/abs/2305.16695)

    本研究在信息检索博弈论模型中提出了相对排名原则（RRP）作为替代排名原则，以达成更稳定的搜索生态系统，并提供了理论和实证证据证明其学习动力学收敛性，同时展示了可能的出版商-用户权衡。

    

    我们研究了一种信息检索的博弈论模型，其中战略出版商旨在在保持原始文档完整性的同时最大化自己排名第一的机会。我们表明，常用的PRP排名方案导致环境不稳定，游戏经常无法达到纯纳什均衡。我们将相对排名原则（RRP）作为替代排名原则，并介绍两个排名函数，它们是RRP的实例。我们提供了理论和实证证据，表明这些方法导致稳定的搜索生态系统，通过提供关于学习动力学收敛的积极结果。我们还定义出版商和用户的福利，并展示了可能的出版商-用户权衡，突显了确定搜索引擎设计师应选择哪种排名函数的复杂性。

    We study a game-theoretic model of information retrieval, in which strategic publishers aim to maximize their chances of being ranked first by the search engine, while maintaining the integrity of their original documents. We show that the commonly used PRP ranking scheme results in an unstable environment where games often fail to reach pure Nash equilibrium. We propose the Relative Ranking Principle (RRP) as an alternative ranking principle, and introduce two ranking functions that are instances of the RRP. We provide both theoretical and empirical evidence that these methods lead to a stable search ecosystem, by providing positive results on the learning dynamics convergence. We also define the publishers' and users' welfare, and demonstrate a possible publisher-user trade-off, which highlights the complexity of determining which ranking function should be selected by the search engine designer.
    

