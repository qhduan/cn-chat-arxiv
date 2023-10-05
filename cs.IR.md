# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Potential Factors Leading to Popularity Unfairness in Recommender Systems: A User-Centered Analysis.](http://arxiv.org/abs/2310.02961) | 该论文研究了导致推荐系统中用户端流行度偏见不公平的因素。 |
| [^2] | [Auto-FP: An Experimental Study of Automated Feature Preprocessing for Tabular Data.](http://arxiv.org/abs/2310.02540) | 本文研究了如何自动化表格数据的特征预处理（Auto-FP），将其建模为超参数优化或神经网络架构搜索问题，并扩展了各种算法来解决Auto-FP问题。 |
| [^3] | [Shaping the Epochal Individuality and Generality: The Temporal Dynamics of Uncertainty and Prediction Error in Musical Improvisation.](http://arxiv.org/abs/2310.02518) | 这项研究探索了音乐即兴创作中不确定性和预测误差的时序动态，发现了音高和音高-节奏序列中的特定时间模式和时代特征，以及节奏序列的一致不确定度。这些发现突显了节奏在音乐创作中的重要性。 |
| [^4] | [Linear Recurrent Units for Sequential Recommendation.](http://arxiv.org/abs/2310.02367) | 本研究提出了一种用于顺序推荐的线性循环单元（LRURec）。与当前的自注意模型相比，LRURec具有快速的推断速度、能够进行增量推断、更小的模型大小和可并行训练。通过优化架构并引入非线性，LRURec在顺序推荐任务中取得了有效的结果。 |
| [^5] | [Beyond-Accuracy: A Review on Diversity, Serendipity and Fairness in Recommender Systems Based on Graph Neural Networks.](http://arxiv.org/abs/2310.02294) | 本综述论文关注于基于图神经网络的推荐系统中的多样性、意外性和公平性问题，超越传统的准确性评估，并讨论了模型开发的不同阶段。 |
| [^6] | [MedCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval.](http://arxiv.org/abs/2307.00589) | MedCPT是一种用于生物医学领域零样本语义信息检索的对比预训练转换器模型。通过使用大规模PubMed搜索日志进行训练，MedCPT在六个生物医学信息检索任务中创造了新的最佳性能，超过了其他基线模型，同时还能生成更好的生物医学文章和句子。 |
| [^7] | [CompoDiff: Versatile Composed Image Retrieval With Latent Diffusion.](http://arxiv.org/abs/2303.11916) | CompoDiff 是一种多功能的组合图像检索模型，通过接受各种条件，具有潜在扩散的能力，并在 FashionIQ 上实现了新的零样本最新技术水平。其特征位于完整的 CLIP 嵌入空间中，可以直接用于所有利用 CLIP 空间的模型。 |
| [^8] | [Reconsidering Learning Objectives in Unbiased Recommendation: A Distribution Shift Perspective.](http://arxiv.org/abs/2206.03851) | 本文从分布转移视角出发，研究了从偏向反馈中学习无偏算法进行推荐的问题。通过建立无偏推荐与分布转移的关系，对现有无偏学习方法进行了理论解释并提出了两个泛化界限。 |
| [^9] | [SR-HetGNN:Session-based Recommendation with Heterogeneous Graph Neural Network.](http://arxiv.org/abs/2108.05641) | 本文提出了一种基于异构图神经网络的会话推荐方法SR-HetGNN，通过学习会话嵌入并捕捉匿名用户的特定偏好，以改进会话推荐系统的效果和准确性。 |

# 详细

[^1]: 推荐系统中导致流行度不公平的潜在因素：基于用户的分析

    Potential Factors Leading to Popularity Unfairness in Recommender Systems: A User-Centered Analysis. (arXiv:2310.02961v1 [cs.IR])

    [http://arxiv.org/abs/2310.02961](http://arxiv.org/abs/2310.02961)

    该论文研究了导致推荐系统中用户端流行度偏见不公平的因素。

    

    流行度偏差是推荐系统中众所周知的问题，其中少数流行物品在输入数据中被过度代表，而其他大部分不那么流行的物品则被低估。这种不平等的代表往往导致推荐结果中物品的暴露存在偏见。已有大量研究从物品的角度研究了这种偏差，并试图通过增强对不那么流行物品的推荐来缓解。然而，最近的研究揭示了这种偏差对用户的影响。对于对流行物品有着不同容忍度的用户而言，推荐系统并不能公平地为他们提供服务：对于对不那么流行物品感兴趣的用户，在他们的推荐中会得到更多的流行物品，而对于对流行物品感兴趣的用户，则被推荐了他们想要的物品。主要原因是流行度偏差使得流行物品被过度推荐。本文旨在探究导致推荐系统中用户端流行度偏见不公平的因素。

    Popularity bias is a well-known issue in recommender systems where few popular items are over-represented in the input data, while majority of other less popular items are under-represented. This disparate representation often leads to bias in exposure given to the items in the recommendation results. Extensive research examined this bias from item perspective and attempted to mitigate it by enhancing the recommendation of less popular items. However, a recent research has revealed the impact of this bias on users. Users with different degree of tolerance toward popular items are not fairly served by the recommendation system: users interested in less popular items receive more popular items in their recommendations, while users interested in popular items are recommended what they want. This is mainly due to the popularity bias that popular items are over-recommended. In this paper, we aim at investigating the factors leading to this user-side unfairness of popularity bias in recommen
    
[^2]: Auto-FP:自动化特征预处理在表格数据上的实验研究

    Auto-FP: An Experimental Study of Automated Feature Preprocessing for Tabular Data. (arXiv:2310.02540v1 [cs.LG])

    [http://arxiv.org/abs/2310.02540](http://arxiv.org/abs/2310.02540)

    本文研究了如何自动化表格数据的特征预处理（Auto-FP），将其建模为超参数优化或神经网络架构搜索问题，并扩展了各种算法来解决Auto-FP问题。

    

    传统的机器学习模型，如线性模型和基于树的模型，在工业中被广泛使用。这些模型对数据分布敏感，因此特征预处理是确保模型质量良好的关键步骤。手动构建特征预处理流程很具挑战性，因为数据科学家需要在选择哪些预处理器以及以什么顺序组合它们方面作出困难的决策。在本文中，我们研究了如何自动化表格数据的特征预处理（Auto-FP）。由于搜索空间较大，暴力解决方案代价太高。为了解决这个挑战，我们有趣地观察到Auto-FP可以被建模为超参数优化（HPO）或神经网络架构搜索（NAS）问题。这个观察使我们能够扩展各种HPO和NAS算法来解决Auto-FP问题。我们进行了全面的评估和分析，共进行了15个...

    Classical machine learning models, such as linear models and tree-based models, are widely used in industry. These models are sensitive to data distribution, thus feature preprocessing, which transforms features from one distribution to another, is a crucial step to ensure good model quality. Manually constructing a feature preprocessing pipeline is challenging because data scientists need to make difficult decisions about which preprocessors to select and in which order to compose them. In this paper, we study how to automate feature preprocessing (Auto-FP) for tabular data. Due to the large search space, a brute-force solution is prohibitively expensive. To address this challenge, we interestingly observe that Auto-FP can be modelled as either a hyperparameter optimization (HPO) or a neural architecture search (NAS) problem. This observation enables us to extend a variety of HPO and NAS algorithms to solve the Auto-FP problem. We conduct a comprehensive evaluation and analysis of 15 
    
[^3]: 塑造时代的个性与共性：音乐即兴创作中的不确定性和预测误差的时序动态

    Shaping the Epochal Individuality and Generality: The Temporal Dynamics of Uncertainty and Prediction Error in Musical Improvisation. (arXiv:2310.02518v1 [cs.SD])

    [http://arxiv.org/abs/2310.02518](http://arxiv.org/abs/2310.02518)

    这项研究探索了音乐即兴创作中不确定性和预测误差的时序动态，发现了音高和音高-节奏序列中的特定时间模式和时代特征，以及节奏序列的一致不确定度。这些发现突显了节奏在音乐创作中的重要性。

    

    音乐即兴创作，就像即兴演讲一样，展现了即兴者的思维状态和情感特质的精妙方面。然而，揭示这种个性的具体音乐组成部分仍未被广泛探索。在大脑的统计学习和预测处理框架内，本研究考察了音乐即兴创作中不确定性和惊讶（预测误差）的时序动态。本研究采用HBSL模型分析了一个包含456段爵士即兴创作的语料库，跨越1905年至2009年，涵盖了78位不同的爵士音乐家。结果表明，尤其是在音高和音高-节奏序列中，惊讶和不确定性呈现出独特的时间模式，揭示了从20世纪初到21世纪的时代特征。相反，节奏序列在不同时代表现出一致的不确定度。此外，声学特性在不同时期保持不变。这些发现突显了节奏在音乐创作中的重要性。

    Musical improvisation, much like spontaneous speech, reveals intricate facets of the improviser's state of mind and emotional character. However, the specific musical components that reveal such individuality remain largely unexplored. Within the framework of brain's statistical learning and predictive processing, this study examined the temporal dynamics of uncertainty and surprise (prediction error) in a piece of musical improvisation. This study employed the HBSL model to analyze a corpus of 456 Jazz improvisations, spanning 1905 to 2009, from 78 distinct Jazz musicians. The results indicated distinctive temporal patterns of surprise and uncertainty, especially in pitch and pitch-rhythm sequences, revealing era-specific features from the early 20th to the 21st centuries. Conversely, rhythm sequences exhibited a consistent degree of uncertainty across eras. Further, the acoustic properties remain unchanged across different periods. These findings highlight the importance of how tempo
    
[^4]: 用于顺序推荐的线性循环单元

    Linear Recurrent Units for Sequential Recommendation. (arXiv:2310.02367v1 [cs.IR])

    [http://arxiv.org/abs/2310.02367](http://arxiv.org/abs/2310.02367)

    本研究提出了一种用于顺序推荐的线性循环单元（LRURec）。与当前的自注意模型相比，LRURec具有快速的推断速度、能够进行增量推断、更小的模型大小和可并行训练。通过优化架构并引入非线性，LRURec在顺序推荐任务中取得了有效的结果。

    

    当前的顺序推荐依赖于基于自注意的推荐模型。然而，这些模型计算代价高，往往对实时推荐来说过于缓慢。此外，自注意操作是在序列层级上进行的，因此对于低成本的增量推断来说具有挑战性。受到高效语言建模的最新进展的启发，我们提出了用于顺序推荐的线性循环单元（LRURec）。类似于循环神经网络，LRURec具有快速的推断速度，并且能够对顺序输入进行增量推断。通过分解线性循环操作并在我们的框架中设计递归并行化，LRURec提供了减小模型大小和可并行训练的额外优势。此外，我们通过实施一系列修改来优化LRURec的架构，以解决缺乏非线性和改善训练动态的问题。为了验证我们提出的LRURec的有效性

    State-of-the-art sequential recommendation relies heavily on self-attention-based recommender models. Yet such models are computationally expensive and often too slow for real-time recommendation. Furthermore, the self-attention operation is performed at a sequence-level, thereby making low-cost incremental inference challenging. Inspired by recent advances in efficient language modeling, we propose linear recurrent units for sequential recommendation (LRURec). Similar to recurrent neural networks, LRURec offers rapid inference and can achieve incremental inference on sequential inputs. By decomposing the linear recurrence operation and designing recursive parallelization in our framework, LRURec provides the additional benefits of reduced model size and parallelizable training. Moreover, we optimize the architecture of LRURec by implementing a series of modifications to address the lack of non-linearity and improve training dynamics. To validate the effectiveness of our proposed LRURe
    
[^5]: 超越准确性: 基于图神经网络的推荐系统中的多样性、意外性和公平性综述

    Beyond-Accuracy: A Review on Diversity, Serendipity and Fairness in Recommender Systems Based on Graph Neural Networks. (arXiv:2310.02294v1 [cs.IR])

    [http://arxiv.org/abs/2310.02294](http://arxiv.org/abs/2310.02294)

    本综述论文关注于基于图神经网络的推荐系统中的多样性、意外性和公平性问题，超越传统的准确性评估，并讨论了模型开发的不同阶段。

    

    通过向用户提供个性化建议，推荐系统已经成为许多在线平台的重要组成部分。协同过滤，特别是使用图神经网络（GNN）的基于图的方法，在推荐准确性方面取得了很好的结果。然而，准确性并不总是评估推荐系统性能最重要的标准，因为除了准确性之外，推荐多样性、意外性和公平性等方面也会对用户参与和满意度产生强烈影响。本综述论文关注于解决基于GNN的推荐系统中的这些维度，超越传统以准确性为中心的视角。我们首先回顾了最近在改善准确性-多样性权衡、促进意外性和公平性方面的方法方面的发展。我们讨论了模型开发的不同阶段，包括数据预处理、图构建、嵌入初始化。

    By providing personalized suggestions to users, recommender systems have become essential to numerous online platforms. Collaborative filtering, particularly graph-based approaches using Graph Neural Networks (GNNs), have demonstrated great results in terms of recommendation accuracy. However, accuracy may not always be the most important criterion for evaluating recommender systems' performance, since beyond-accuracy aspects such as recommendation diversity, serendipity, and fairness can strongly influence user engagement and satisfaction. This review paper focuses on addressing these dimensions in GNN-based recommender systems, going beyond the conventional accuracy-centric perspective. We begin by reviewing recent developments in approaches that improve not only the accuracy-diversity trade-off but also promote serendipity and fairness in GNN-based recommender systems. We discuss different stages of model development including data preprocessing, graph construction, embedding initia
    
[^6]: MedCPT: 使用大规模PubMed搜索日志的对比预训练转换器进行零样本生物医学信息检索

    MedCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval. (arXiv:2307.00589v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2307.00589](http://arxiv.org/abs/2307.00589)

    MedCPT是一种用于生物医学领域零样本语义信息检索的对比预训练转换器模型。通过使用大规模PubMed搜索日志进行训练，MedCPT在六个生物医学信息检索任务中创造了新的最佳性能，超过了其他基线模型，同时还能生成更好的生物医学文章和句子。

    

    信息检索在生物医学知识获取和临床决策支持中至关重要。尽管最近的进展表明语言模型编码器在语义检索方面表现更好，但训练这些模型需要大量的查询-文章注释，在生物医学领域很难获得。因此，大多数生物医学信息检索系统只进行词汇匹配。为此，我们引入了MedCPT，这是一种首创的用于生物医学领域零样本语义信息检索的对比预训练转换器模型。为了训练MedCPT，我们从PubMed收集了255 million个用户点击日志，这是前所未有的规模。利用这些数据，我们使用对比学习来训练一对密切集成的检索器和重排器。实验结果显示，MedCPT在六个生物医学信息检索任务中取得了新的最佳性能，优于包括更大模型（如GPT-3大小的cpt-text-XL）在内的各种基线模型。此外，MedCPT还能够生成更好的生物医学文章和句子。

    Information retrieval (IR) is essential in biomedical knowledge acquisition and clinical decision support. While recent progress has shown that language model encoders perform better semantic retrieval, training such models requires abundant query-article annotations that are difficult to obtain in biomedicine. As a result, most biomedical IR systems only conduct lexical matching. In response, we introduce MedCPT, a first-of-its-kind Contrastively Pre-trained Transformer model for zero-shot semantic IR in biomedicine. To train MedCPT, we collected an unprecedented scale of 255 million user click logs from PubMed. With such data, we use contrastive learning to train a pair of closely-integrated retriever and re-ranker. Experimental results show that MedCPT sets new state-of-the-art performance on six biomedical IR tasks, outperforming various baselines including much larger models such as GPT-3-sized cpt-text-XL. In addition, MedCPT also generates better biomedical article and sentence 
    
[^7]: CompoDiff: 基于潜在扩散的多功能组合图像检索

    CompoDiff: Versatile Composed Image Retrieval With Latent Diffusion. (arXiv:2303.11916v1 [cs.CV])

    [http://arxiv.org/abs/2303.11916](http://arxiv.org/abs/2303.11916)

    CompoDiff 是一种多功能的组合图像检索模型，通过接受各种条件，具有潜在扩散的能力，并在 FashionIQ 上实现了新的零样本最新技术水平。其特征位于完整的 CLIP 嵌入空间中，可以直接用于所有利用 CLIP 空间的模型。

    

    本文提出了一种新颖的基于扩散的模型 CompoDiff，用于解决具有潜在扩散的组合图像检索（CIR）问题，并提供了一个由 1800 万个参考图像、条件和相应的目标图像三元组组成的新数据集，用于训练模型。CompoDiff 不仅在像 FashionIQ 这样的 CIR 基准测试上实现了新的零样本最新技术水平，而且还通过接收各种条件（如负文本和图像遮罩条件），使得 CIR 更加多功能，这是现有 CIR 方法所不具备的。此外，CompoDiff 特征位于完整的 CLIP 嵌入空间中，因此它们可以直接用于利用 CLIP 空间的所有现有模型。训练所使用的代码和数据集，以及预训练权重可在 https://github.com/navervision/CompoDiff 上获得。

    This paper proposes a novel diffusion-based model, CompoDiff, for solving Composed Image Retrieval (CIR) with latent diffusion and presents a newly created dataset of 18 million reference images, conditions, and corresponding target image triplets to train the model. CompoDiff not only achieves a new zero-shot state-of-the-art on a CIR benchmark such as FashionIQ but also enables a more versatile CIR by accepting various conditions, such as negative text and image mask conditions, which are unavailable with existing CIR methods. In addition, the CompoDiff features are on the intact CLIP embedding space so that they can be directly used for all existing models exploiting the CLIP space. The code and dataset used for the training, and the pre-trained weights are available at https://github.com/navervision/CompoDiff
    
[^8]: 在无偏推荐中重新考虑学习目标：分布转移视角下的研究

    Reconsidering Learning Objectives in Unbiased Recommendation: A Distribution Shift Perspective. (arXiv:2206.03851v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2206.03851](http://arxiv.org/abs/2206.03851)

    本文从分布转移视角出发，研究了从偏向反馈中学习无偏算法进行推荐的问题。通过建立无偏推荐与分布转移的关系，对现有无偏学习方法进行了理论解释并提出了两个泛化界限。

    

    本文研究了从偏向反馈中学习无偏算法进行推荐的问题，我们从一个新颖的分布转移视角来解决这个问题。最近在无偏推荐领域的研究中，通过各种技术如重新加权、多任务学习和元学习，取得了最新的成果。尽管它们在实证上取得了成功，但大部分缺乏理论保证，导致了理论和最新算法之间的显著差距。本文提出了对现有无偏学习目标为何适用于无偏推荐的理论理解。我们建立了无偏推荐与分布转移之间的密切关系，显示了现有的无偏学习目标隐含地将有偏的训练分布与无偏的测试分布对齐。基于这个关系，我们针对现有的无偏学习方法发展了两个泛化界限并分析了它们的学习行为。

    This work studies the problem of learning unbiased algorithms from biased feedback for recommendation. We address this problem from a novel distribution shift perspective. Recent works in unbiased recommendation have advanced the state-of-the-art with various techniques such as re-weighting, multi-task learning, and meta-learning. Despite their empirical successes, most of them lack theoretical guarantees, forming non-negligible gaps between theories and recent algorithms. In this paper, we propose a theoretical understanding of why existing unbiased learning objectives work for unbiased recommendation. We establish a close connection between unbiased recommendation and distribution shift, which shows that existing unbiased learning objectives implicitly align biased training and unbiased test distributions. Built upon this connection, we develop two generalization bounds for existing unbiased learning methods and analyze their learning behavior. Besides, as a result of the distributio
    
[^9]: SR-HetGNN:基于异构图神经网络的会话推荐系统

    SR-HetGNN:Session-based Recommendation with Heterogeneous Graph Neural Network. (arXiv:2108.05641v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2108.05641](http://arxiv.org/abs/2108.05641)

    本文提出了一种基于异构图神经网络的会话推荐方法SR-HetGNN，通过学习会话嵌入并捕捉匿名用户的特定偏好，以改进会话推荐系统的效果和准确性。

    

    会话推荐系统的目的是根据先前的会话序列预测用户的下一次点击。目前的研究通常根据用户会话序列中的项目转换来学习用户偏好。然而，会话序列中的其他有效信息，如用户配置文件，往往被忽视，这可能导致模型无法学习用户的具体偏好。在本文中，我们提出了一种基于异构图神经网络的会话推荐方法，命名为SR-HetGNN，它可以通过异构图神经网络（HetGNN）学习会话嵌入，并捕捉匿名用户的特定偏好。具体而言，SR-HetGNN首先根据会话序列构建包含各种类型节点的异构图，可以捕捉项目、用户和会话之间的依赖关系。其次，HetGNN捕捉项目之间的复杂转换并学习包含项目嵌入的特征。

    The purpose of the Session-Based Recommendation System is to predict the user's next click according to the previous session sequence. The current studies generally learn user preferences according to the transitions of items in the user's session sequence. However, other effective information in the session sequence, such as user profiles, are largely ignored which may lead to the model unable to learn the user's specific preferences. In this paper, we propose a heterogeneous graph neural network-based session recommendation method, named SR-HetGNN, which can learn session embeddings by heterogeneous graph neural network (HetGNN), and capture the specific preferences of anonymous users. Specifically, SR-HetGNN first constructs heterogeneous graphs containing various types of nodes according to the session sequence, which can capture the dependencies among items, users, and sessions. Second, HetGNN captures the complex transitions between items and learns the item embeddings containing
    

