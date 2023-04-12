# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prompt Learning for News Recommendation.](http://arxiv.org/abs/2304.05263) | 本文介绍了在新闻推荐领域首次采用预训练，提示学习和预测范例来开发Prompt4NR框架的实验。该框架将预测点击候选新闻的任务转化为cloze-style填空式掩码预测任务，从而更好地利用预训练过程中的丰富语义信息和语言知识。 |
| [^2] | [Audio Bank: A High-Level Acoustic Signal Representation for Audio Event Recognition.](http://arxiv.org/abs/2304.05067) | 本文提出了一种名为Audio Bank的音频信号表征框架，其由在时间-频率空间中表示每个音频类别的独特音频检测器组成。使用非负矩阵分解降低特征向量的维数，同时保留其可区分性和丰富的语义信息。在两个公开可用数据集上使用多个分类器进行音频识别，证明了该框架的高效性。 |
| [^3] | [Unbiased Pairwise Learning from Implicit Feedback for Recommender Systems without Biased Variance Control.](http://arxiv.org/abs/2304.05066) | 本文提出了一种名为“无偏成对学习（NPLwVC）”的新框架，解决了推荐系统中隐式反馈数据的偏估问题，实验结果表明其在两个公共数据集上表现优于现有最先进方法。 |
| [^4] | [A Comprehensive Survey on Deep Graph Representation Learning.](http://arxiv.org/abs/2304.05055) | 本文综述了深度图表示学习的研究现状和存在的问题，并指出利用深度学习已经显示出巨大的优势和潜力。 |
| [^5] | [Diffusion Recommender Model.](http://arxiv.org/abs/2304.04971) | 本论文提出了一种新颖的扩散推荐模型（DiffRec）来逐步去噪地学习用户交互生成的过程，并针对推荐系统中的冷启动问题和稀疏数据等独特挑战进行了扩展，实验结果显示其在推荐准确性和稳健性方面优于现有方法。 |
| [^6] | [AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations.](http://arxiv.org/abs/2304.04959) | AdaTT是一种适用于推荐系统的自适应任务融合深度网络模型，通过利用残差机制和门控机制实现任务之间的融合，自适应地学习共享知识和任务特定知识，在多个任务上可以显著优于现有的最先进基线模型。 |
| [^7] | [Explicit and Implicit Semantic Ranking Framework.](http://arxiv.org/abs/2304.04918) | 本文提出了一个名为sRank的通用语义学习排名框架，它使用transformer模型，能够在智能回复和环境临床智能等真实应用中，实现11.7%的离线准确度提升。 |
| [^8] | [Similarity search in the blink of an eye with compressed indices.](http://arxiv.org/abs/2304.04759) | 本文提出一种新的向量压缩方法局部自适应量化(LVQ)，并在基于图的索引的关键优化下实现减少有效带宽同时启用随机访问友好的快速相似性计算，从而在性能和内存占用方面创造了新的最佳表现。 |
| [^9] | [Graph Collaborative Signals Denoising and Augmentation for Recommendation.](http://arxiv.org/abs/2304.03344) | 本文提出了一种新的图邻接矩阵，它包括了用户-用户和项目-项目的相关性，以及一个经过适当设计的用户-项目交互矩阵，并通过预训练和top-K采样增强了用户-项目交互矩阵，以更好地适应所有用户的需求。 |
| [^10] | [Manipulating Federated Recommender Systems: Poisoning with Synthetic Users and Its Countermeasures.](http://arxiv.org/abs/2304.03054) | 本文提出了一种新的攻击方法，利用合成的恶意用户上传有毒的梯度来在联邦推荐系统中有效地操纵目标物品的排名和曝光率。在两个真实世界的推荐数据集上进行了大量实验。 |
| [^11] | [XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation.](http://arxiv.org/abs/2209.02544) | 本文提出一种极简的推荐图形对比学习方法(XSimGCL)，发现有效减轻流行度偏见与促进长尾物品发现并不需要过多的图形增强。 |

# 详细

[^1]: 新闻推荐中的提示学习

    Prompt Learning for News Recommendation. (arXiv:2304.05263v1 [cs.IR])

    [http://arxiv.org/abs/2304.05263](http://arxiv.org/abs/2304.05263)

    本文介绍了在新闻推荐领域首次采用预训练，提示学习和预测范例来开发Prompt4NR框架的实验。该框架将预测点击候选新闻的任务转化为cloze-style填空式掩码预测任务，从而更好地利用预训练过程中的丰富语义信息和语言知识。

    

    最近的一些新闻推荐（NR）方法通过引入预训练语言模型（PLM）来编码新闻表示，采用精心设计的推荐特定神经网络和目标函数来遵循香草预训练和微调范例。由于任务目标与PLM不一致，我们认为他们的建模范式未能充分利用预训练过程中嵌入的丰富语义信息和语言知识。最近，预训练，提示和预测范例在自然语言处理领域取得了许多成功。在本文中，我们第一次尝试使用这种新范例来开发一个新闻推荐中的Prompt Learning (Prompt4NR) 框架，将预测用户是否会点击候选新闻的任务转化为填空式掩码预测任务。具体来说，我们设计了一系列prompt模板，包括离散、连续...

    Some recent \textit{news recommendation} (NR) methods introduce a Pre-trained Language Model (PLM) to encode news representation by following the vanilla pre-train and fine-tune paradigm with carefully-designed recommendation-specific neural networks and objective functions. Due to the inconsistent task objective with that of PLM, we argue that their modeling paradigm has not well exploited the abundant semantic information and linguistic knowledge embedded in the pre-training process. Recently, the pre-train, prompt, and predict paradigm, called \textit{prompt learning}, has achieved many successes in natural language processing domain. In this paper, we make the first trial of this new paradigm to develop a \textit{Prompt Learning for News Recommendation} (Prompt4NR) framework, which transforms the task of predicting whether a user would click a candidate news as a cloze-style mask-prediction task. Specifically, we design a series of prompt templates, including discrete, continuous, 
    
[^2]: Audio Bank：一种用于音频事件识别的高级声学信号表征。

    Audio Bank: A High-Level Acoustic Signal Representation for Audio Event Recognition. (arXiv:2304.05067v1 [eess.AS])

    [http://arxiv.org/abs/2304.05067](http://arxiv.org/abs/2304.05067)

    本文提出了一种名为Audio Bank的音频信号表征框架，其由在时间-频率空间中表示每个音频类别的独特音频检测器组成。使用非负矩阵分解降低特征向量的维数，同时保留其可区分性和丰富的语义信息。在两个公开可用数据集上使用多个分类器进行音频识别，证明了该框架的高效性。

    

    自动音频事件识别在使人机交互更加紧密方面发挥着关键作用，并在工业自动化、控制和监视系统中具有广泛的适用性。音频事件由纷繁复杂的声韵模式组成，它们在谐波上纠缠在一起。音频识别主要受低级和中级特征的控制，这些特征已证明其识别能力，但计算成本高且语义意义低。本文提出了一种新的计算效率高的音频识别框架。Audio Bank是一种新的音频高级表征，由在频率时间空间中表示每个音频类别的独特音频检测器组成。通过使用非负矩阵分解减少结果特征向量的维数，同时保留其可区分性和丰富的语义信息。对两个公开可用数据集使用多个分类器（SVM、神经网络、高斯过程分类和k最近邻）进行音频识别，证明了Audio Bank框架的高效性。

    Automatic audio event recognition plays a pivotal role in making human robot interaction more closer and has a wide applicability in industrial automation, control and surveillance systems. Audio event is composed of intricate phonic patterns which are harmonically entangled. Audio recognition is dominated by low and mid-level features, which have demonstrated their recognition capability but they have high computational cost and low semantic meaning. In this paper, we propose a new computationally efficient framework for audio recognition. Audio Bank, a new high-level representation of audio, is comprised of distinctive audio detectors representing each audio class in frequency-temporal space. Dimensionality of the resulting feature vector is reduced using non-negative matrix factorization preserving its discriminability and rich semantic information. The high audio recognition performance using several classifiers (SVM, neural network, Gaussian process classification and k-nearest ne
    
[^3]: 面向推荐系统的无偏成对学习算法

    Unbiased Pairwise Learning from Implicit Feedback for Recommender Systems without Biased Variance Control. (arXiv:2304.05066v1 [cs.IR])

    [http://arxiv.org/abs/2304.05066](http://arxiv.org/abs/2304.05066)

    本文提出了一种名为“无偏成对学习（NPLwVC）”的新框架，解决了推荐系统中隐式反馈数据的偏估问题，实验结果表明其在两个公共数据集上表现优于现有最先进方法。

    

    推荐系统的模型训练一般基于显式反馈和隐式反馈两种数据。隐式反馈数据中仅包含正反馈，因此很难判断未互动项到底是负反馈还是未曝光。同时，稀有物品的相关性往往被低估。为了解决这些问题，先前提出了无偏成对学习算法，但存在偏差方差控制问题。本文提出一种名为“无偏成对学习（NPLwVC）”的新框架，不需要偏差方差控制项，从而简化算法。两个公共数据集的实验结果表明，NPLwVC在保持无偏性的同时显著优于现有最先进方法。

    Generally speaking, the model training for recommender systems can be based on two types of data, namely explicit feedback and implicit feedback. Moreover, because of its general availability, we see wide adoption of implicit feedback data, such as click signal. There are mainly two challenges for the application of implicit feedback. First, implicit data just includes positive feedback. Therefore, we are not sure whether the non-interacted items are really negative or positive but not displayed to the corresponding user. Moreover, the relevance of rare items is usually underestimated since much fewer positive feedback of rare items is collected compared with popular ones. To tackle such difficulties, both pointwise and pairwise solutions are proposed before for unbiased relevance learning. As pairwise learning suits well for the ranking tasks, the previously proposed unbiased pairwise learning algorithm already achieves state-of-the-art performance. Nonetheless, the existing unbiased 
    
[^4]: 深度图表示学习综述

    A Comprehensive Survey on Deep Graph Representation Learning. (arXiv:2304.05055v1 [cs.LG])

    [http://arxiv.org/abs/2304.05055](http://arxiv.org/abs/2304.05055)

    本文综述了深度图表示学习的研究现状和存在的问题，并指出利用深度学习已经显示出巨大的优势和潜力。

    

    图表示学习旨在将高维稀疏的图结构数据有效地编码成低维密集向量，这是一个基本任务，在包括机器学习和数据挖掘在内的一系列领域都得到了广泛的研究。传统图嵌入方法遵循这样一种基本思想，即图中相互连接的节点的嵌入矢量仍然能够保持相对接近的距离，从而保留了图中节点之间的结构信息。然而，这种方法存在以下问题：（i）传统方法的模型容量受限，限制了学习性能; （ii）现有技术通常依赖于无监督学习策略，无法与最新的学习范式相结合；（iii）表示学习和下游任务相互依存，应共同加强。随着深度学习的显着成功，深度图表示学习已经显示出巨大的潜力和优势。

    Graph representation learning aims to effectively encode high-dimensional sparse graph-structured data into low-dimensional dense vectors, which is a fundamental task that has been widely studied in a range of fields, including machine learning and data mining. Classic graph embedding methods follow the basic idea that the embedding vectors of interconnected nodes in the graph can still maintain a relatively close distance, thereby preserving the structural information between the nodes in the graph. However, this is sub-optimal due to: (i) traditional methods have limited model capacity which limits the learning performance; (ii) existing techniques typically rely on unsupervised learning strategies and fail to couple with the latest learning paradigms; (iii) representation learning and downstream tasks are dependent on each other which should be jointly enhanced. With the remarkable success of deep learning, deep graph representation learning has shown great potential and advantages 
    
[^5]: 扩散推荐模型

    Diffusion Recommender Model. (arXiv:2304.04971v1 [cs.IR])

    [http://arxiv.org/abs/2304.04971](http://arxiv.org/abs/2304.04971)

    本论文提出了一种新颖的扩散推荐模型（DiffRec）来逐步去噪地学习用户交互生成的过程，并针对推荐系统中的冷启动问题和稀疏数据等独特挑战进行了扩展，实验结果显示其在推荐准确性和稳健性方面优于现有方法。

    

    生成模型（如生成对抗网络（GANs）和变分自动编码器（VAEs））被广泛应用于建模用户交互的生成过程。然而，这些生成模型存在固有的局限性，如GANs的不稳定性和VAEs的受限表征能力。这些限制妨碍了复杂用户交互生成过程的准确建模，例如由各种干扰因素导致的嘈杂交互。考虑到扩散模型（DMs）在图像合成方面相对于传统的生成模型具有显着优势，我们提出了一种新颖的扩散推荐模型（称为DiffRec），以逐步去噪的方式学习生成过程。为了保留用户交互中的个性化信息，DiffRec减少了添加的噪声，并避免将用户交互损坏为像图像合成中的纯噪声。此外，我们扩展了传统的DMs以应对实际推荐系统中的独特挑战，如冷启动问题和稀疏的用户-物品交互数据。在几个真实数据集上的实验结果表明，DiffRec在推荐准确性和稳健性方面优于现有方法。

    Generative models such as Generative Adversarial Networks (GANs) and Variational Auto-Encoders (VAEs) are widely utilized to model the generative process of user interactions. However, these generative models suffer from intrinsic limitations such as the instability of GANs and the restricted representation ability of VAEs. Such limitations hinder the accurate modeling of the complex user interaction generation procedure, such as noisy interactions caused by various interference factors. In light of the impressive advantages of Diffusion Models (DMs) over traditional generative models in image synthesis, we propose a novel Diffusion Recommender Model (named DiffRec) to learn the generative process in a denoising manner. To retain personalized information in user interactions, DiffRec reduces the added noises and avoids corrupting users' interactions into pure noises like in image synthesis. In addition, we extend traditional DMs to tackle the unique challenges in practical recommender 
    
[^6]: AdaTT: 自适应任务融合网络用于多任务学习推荐系统

    AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations. (arXiv:2304.04959v1 [cs.IR])

    [http://arxiv.org/abs/2304.04959](http://arxiv.org/abs/2304.04959)

    AdaTT是一种适用于推荐系统的自适应任务融合深度网络模型，通过利用残差机制和门控机制实现任务之间的融合，自适应地学习共享知识和任务特定知识，在多个任务上可以显著优于现有的最先进基线模型。

    

    多任务学习旨在通过同时在多个任务上训练机器学习模型来提高性能和效率。然而，多任务学习面临两个挑战：1）对任务之间的关系进行建模，以有效地共享知识；2）联合学习任务特定和共享知识。本文介绍了一种新型的自适应任务融合网络（AdaTT）来解决这两个挑战。AdaTT是一个深度融合网络，在多个级别上使用任务特定和可选共享融合单元构建。通过利用残差机制和门控机制实现任务之间的融合，这些单元自适应地学习共享知识和任务特定知识。为了评估AdaTT的性能，我们在公共基准测试集和工业级推荐数据集上使用不同的任务组进行实验。结果表明，AdaTT可以显著优于现有的最先进基线模型。

    Multi-task learning (MTL) aims at enhancing the performance and efficiency of machine learning models by training them on multiple tasks simultaneously. However, MTL research faces two challenges: 1) modeling the relationships between tasks to effectively share knowledge between them, and 2) jointly learning task-specific and shared knowledge. In this paper, we present a novel model Adaptive Task-to-Task Fusion Network (AdaTT) to address both challenges. AdaTT is a deep fusion network built with task specific and optional shared fusion units at multiple levels. By leveraging a residual mechanism and gating mechanism for task-to-task fusion, these units adaptively learn shared knowledge and task specific knowledge. To evaluate the performance of AdaTT, we conduct experiments on a public benchmark and an industrial recommendation dataset using various task groups. Results demonstrate AdaTT can significantly outperform existing state-of-the-art baselines.
    
[^7]: 显式和隐式语义排序框架

    Explicit and Implicit Semantic Ranking Framework. (arXiv:2304.04918v1 [cs.IR])

    [http://arxiv.org/abs/2304.04918](http://arxiv.org/abs/2304.04918)

    本文提出了一个名为sRank的通用语义学习排名框架，它使用transformer模型，能够在智能回复和环境临床智能等真实应用中，实现11.7%的离线准确度提升。

    

    在许多实际应用中，核心难题是将一个查询与一个可变且有限的文档集中的最佳文档进行匹配。现有的工业解决方案，特别是延迟受限的服务，通常依赖于相似性算法，这些算法为了速度而牺牲了质量。本文介绍了一个通用的语义学习排名框架，自我训练语义交叉关注排名（sRank）。这个基于transformer的框架使用线性成对损失，具有可变的训练批量大小、实现质量提升和高效率，并已成功应用于微软公司的两个工业任务：智能回复（SR）和环境临床智能（ACI）的真实大规模数据集上。在智能回复中，$sRank$通过基于消费者和支持代理信息的预定义解决方案选择最佳答案，帮助用户实时获得技术支持。在SR任务上，$sRank$实现了11.7%的离线top-one准确度提升，比之前的系统更加优秀。

    The core challenge in numerous real-world applications is to match an inquiry to the best document from a mutable and finite set of candidates. Existing industry solutions, especially latency-constrained services, often rely on similarity algorithms that sacrifice quality for speed. In this paper we introduce a generic semantic learning-to-rank framework, Self-training Semantic Cross-attention Ranking (sRank). This transformer-based framework uses linear pairwise loss with mutable training batch sizes and achieves quality gains and high efficiency, and has been applied effectively to show gains on two industry tasks at Microsoft over real-world large-scale data sets: Smart Reply (SR) and Ambient Clinical Intelligence (ACI). In Smart Reply, $sRank$ assists live customers with technical support by selecting the best reply from predefined solutions based on consumer and support agent messages. It achieves 11.7% gain in offline top-one accuracy on the SR task over the previous system, and 
    
[^8]: 压缩索引实现瞬间相似性搜索

    Similarity search in the blink of an eye with compressed indices. (arXiv:2304.04759v1 [cs.LG])

    [http://arxiv.org/abs/2304.04759](http://arxiv.org/abs/2304.04759)

    本文提出一种新的向量压缩方法局部自适应量化(LVQ)，并在基于图的索引的关键优化下实现减少有效带宽同时启用随机访问友好的快速相似性计算，从而在性能和内存占用方面创造了新的最佳表现。

    

    如今，数据以向量表示。在海量数据中寻找与给定查询相似的向量是一项广泛应用的问题。本文提出了创建更快、更小的索引以运行这些搜索的新技术。为此，我们介绍了一种新的向量压缩方法，局部自适应量化(LVQ)，它同时减少内存占用和改善搜索性能，对搜索准确性的影响最小。LVQ被设计为与基于图的索引一起工作以实现减少有效带宽同时启用随机访问友好的快速相似性计算。我们的实验结果表明，在现代数据中心系统中针对基于图的索引进行关键优化后，LVQ的性能和内存占用方面创造了新的最佳表现。在处理数十亿个向量时，LVQ超过第二佳方案：

    Nowadays, data is represented by vectors. Retrieving those vectors, among millions and billions, that are similar to a given query is a ubiquitous problem of relevance for a wide range of applications. In this work, we present new techniques for creating faster and smaller indices to run these searches. To this end, we introduce a novel vector compression method, Locally-adaptive Vector Quantization (LVQ), that simultaneously reduces memory footprint and improves search performance, with minimal impact on search accuracy. LVQ is designed to work optimally in conjunction with graph-based indices, reducing their effective bandwidth while enabling random-access-friendly fast similarity computations. Our experimental results show that LVQ, combined with key optimizations for graph-based indices in modern datacenter systems, establishes the new state of the art in terms of performance and memory footprint. For billions of vectors, LVQ outcompetes the second-best alternatives: (1) in the low
    
[^9]: 推荐系统的图协作信号去噪与增强

    Graph Collaborative Signals Denoising and Augmentation for Recommendation. (arXiv:2304.03344v1 [cs.IR])

    [http://arxiv.org/abs/2304.03344](http://arxiv.org/abs/2304.03344)

    本文提出了一种新的图邻接矩阵，它包括了用户-用户和项目-项目的相关性，以及一个经过适当设计的用户-项目交互矩阵，并通过预训练和top-K采样增强了用户-项目交互矩阵，以更好地适应所有用户的需求。

    

    图协作过滤（GCF）是捕捉推荐系统中高阶协同信号的流行技术。然而，GCF的双向邻接矩阵，其定义了基于用户-项目交互进行聚合的邻居，对于有大量交互但不足的用户/项目来说可能是嘈杂的。此外，邻接矩阵忽略了用户-用户和项目-项目之间的相关性，这可能限制了聚合的有益邻居的范围。在这项工作中，我们提出了一种新的图邻接矩阵，它包括了用户-用户和项目-项目的相关性，以及一个经过适当设计的用户-项目交互矩阵，以平衡所有用户之间的交互数量。为了实现这一点，我们预先训练了一个基于图的推荐方法来获得用户/项目嵌入，然后通过top-K采样增强了用户-项目交互矩阵。我们还增强了对称的用户-用户和项目-项目相关组件，以更好地适应所有用户的需求。

    Graph collaborative filtering (GCF) is a popular technique for capturing high-order collaborative signals in recommendation systems. However, GCF's bipartite adjacency matrix, which defines the neighbors being aggregated based on user-item interactions, can be noisy for users/items with abundant interactions and insufficient for users/items with scarce interactions. Additionally, the adjacency matrix ignores user-user and item-item correlations, which can limit the scope of beneficial neighbors being aggregated.  In this work, we propose a new graph adjacency matrix that incorporates user-user and item-item correlations, as well as a properly designed user-item interaction matrix that balances the number of interactions across all users. To achieve this, we pre-train a graph-based recommendation method to obtain users/items embeddings, and then enhance the user-item interaction matrix via top-K sampling. We also augment the symmetric user-user and item-item correlation components to th
    
[^10]: 操纵联邦推荐系统: 用合成用户进行攻击及其对策

    Manipulating Federated Recommender Systems: Poisoning with Synthetic Users and Its Countermeasures. (arXiv:2304.03054v1 [cs.IR])

    [http://arxiv.org/abs/2304.03054](http://arxiv.org/abs/2304.03054)

    本文提出了一种新的攻击方法，利用合成的恶意用户上传有毒的梯度来在联邦推荐系统中有效地操纵目标物品的排名和曝光率。在两个真实世界的推荐数据集上进行了大量实验。

    

    联邦推荐系统（FedRecs）被认为是一种保护隐私的技术，可以在不共享用户数据的情况下协同学习推荐模型。因为所有参与者都可以通过上传梯度直接影响系统，所以FedRecs容易受到恶意客户的攻击，尤其是利用合成用户进行的攻击更加有效。本文提出了一种新的攻击方法，可以在不依赖任何先前知识的情况下，通过一组合成的恶意用户上传有毒的梯度来有效地操纵目标物品的排名和曝光率。我们在两个真实世界的推荐数据集上对两种广泛使用的FedRecs （Fed-NCF和Fed-LightGCN）进行了大量实验。

    Federated Recommender Systems (FedRecs) are considered privacy-preserving techniques to collaboratively learn a recommendation model without sharing user data. Since all participants can directly influence the systems by uploading gradients, FedRecs are vulnerable to poisoning attacks of malicious clients. However, most existing poisoning attacks on FedRecs are either based on some prior knowledge or with less effectiveness. To reveal the real vulnerability of FedRecs, in this paper, we present a new poisoning attack method to manipulate target items' ranks and exposure rates effectively in the top-$K$ recommendation without relying on any prior knowledge. Specifically, our attack manipulates target items' exposure rate by a group of synthetic malicious users who upload poisoned gradients considering target items' alternative products. We conduct extensive experiments with two widely used FedRecs (Fed-NCF and Fed-LightGCN) on two real-world recommendation datasets. The experimental res
    
[^11]: XSimGCL：面向极简推荐图形对比学习的探索

    XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation. (arXiv:2209.02544v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2209.02544](http://arxiv.org/abs/2209.02544)

    本文提出一种极简的推荐图形对比学习方法(XSimGCL)，发现有效减轻流行度偏见与促进长尾物品发现并不需要过多的图形增强。

    

    最近，对比学习(Contrastive learning, CL)在提高推荐系统性能方面扮演着重要角色。基于CL的推荐模型的原理是: 确保从用户-物品二分图的不同图形增强中派生的表示一致性。这种自监督方法可以从原始数据中提取通用特征，减轻数据稀疏性的问题。尽管这种方法很有效，但是其性能提升的因素尚未被完全理解。本文提出了对CL对推荐的影响的新见解。我们的发现表明，CL使模型学习到更均匀分布的用户和物品表示，从而减轻了盛行的流行度偏见，促进了长尾物品的发现。我们的分析还表明，之前认为必不可少的图形增强在基于CL的推荐中相对不可靠且影响有限。

    Contrastive learning (CL) has recently been demonstrated critical in improving recommendation performance. The underlying principle of CL-based recommendation models is to ensure the consistency between representations derived from different graph augmentations of the user-item bipartite graph. This self-supervised approach allows for the extraction of general features from raw data, thereby mitigating the issue of data sparsity. Despite the effectiveness of this paradigm, the factors contributing to its performance gains have yet to be fully understood. This paper provides novel insights into the impact of CL on recommendation. Our findings indicate that CL enables the model to learn more evenly distributed user and item representations, which alleviates the prevalent popularity bias and promoting long-tail items. Our analysis also suggests that the graph augmentations, previously considered essential, are relatively unreliable and of limited significance in CL-based recommendation. B
    

