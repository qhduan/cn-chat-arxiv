# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PBNR: Prompt-based News Recommender System.](http://arxiv.org/abs/2304.07862) | 本文提出了一种基于提示学习和预训练的语言模型的新闻推荐系统(PBNR)，它能够更好地捕捉用户的兴趣并提高推荐准确性，实验表明PBNR超过了其他现有方法。 |
| [^2] | [Cold-Start based Multi-Scenario Ranking Model for Click-Through Rate Prediction.](http://arxiv.org/abs/2304.07858) | 本文提出了一种名为Cold-Start based Multi-scenario Network的方法，用于在在线旅游平台中的点击率预测问题中解决冷启动用户问题。 |
| [^3] | [Syntactic Complexity Identification, Measurement, and Reduction Through Controlled Syntactic Simplification.](http://arxiv.org/abs/2304.07774) | 本研究提出了一种控制简化方法，可在不丢失信息的情况下，基于句子中的实际信息三元组简化句子，以进行知识图谱的创建。 |
| [^4] | [Meta-optimized Contrastive Learning for Sequential Recommendation.](http://arxiv.org/abs/2304.07763) | 本文提出了 MCLRec 模型，该模型在数据增强和可学习模型增强操作的基础上，解决了现有对比学习方法难以推广和训练数据不足的问题。 |
| [^5] | [Hierarchical and Contrastive Representation Learning for Knowledge-aware Recommendation.](http://arxiv.org/abs/2304.07506) | 本文提出了HiCON框架，运用层次化对比表示学习提高了学习到的节点表示的区分能力，同时采用分层消息聚合机制避免了邻居的指数级扩展，从而有效地解决了知识驱动推荐中的过度平滑问题。 |
| [^6] | [Temporal Aggregation and Propagation Graph Neural Networks for Dynamic Representation.](http://arxiv.org/abs/2304.07503) | 本文提出了 TAP-GNN，通过整个邻域的时间聚合和传播来有效地建模动态图中的时间关系，从而在图流场景中支持高效的在线推理。 |
| [^7] | [More Is Less: When Do Recommenders Underperform for Data-rich Users?.](http://arxiv.org/abs/2304.07487) | 研究了推荐算法在数据量丰富和数据量贫乏的用户中的性能表现。发现在所有数据集中，精度在数据丰富的用户中始终更高；平均精度相当，但其方差很大；当评估过程中采用负样本抽样时，召回率产生反直觉结果，表现更好的是数据贫乏的用户；随着用户与推荐系统的互动增加，他们收到的推荐质量会降低。 |
| [^8] | [Self-supervised Auxiliary Loss for Metric Learning in Music Similarity-based Retrieval and Auto-tagging.](http://arxiv.org/abs/2304.07449) | 本论文提出了一种自监督学习方法，在自动标注方面已经证明其有效性。我们引入了自监督辅助损失的度量学习方法来解决音乐相似度检索问题，并发现同时使用自监督和监督信号训练模型的优势，而不冻结预训练模型。此外，避免在微调阶段使用数据增强可以提高性能。 |
| [^9] | [Zero-Shot Multi-Label Topic Inference with Sentence Encoders.](http://arxiv.org/abs/2304.07382) | 本文研究了如何利用句子编码器进行“零样本主题推断”任务，并通过实验证明了Sentence-BERT在通用性方面优于其他编码器，而在效率方面则优先选择通用句子编码器。 |
| [^10] | [Diffusion Recommender Model.](http://arxiv.org/abs/2304.04971) | 本论文提出了一种新颖的扩散推荐模型（DiffRec）来逐步去噪地学习用户交互生成的过程，并针对推荐系统中的冷启动问题和稀疏数据等独特挑战进行了扩展，实验结果显示其在推荐准确性和稳健性方面优于现有方法。 |
| [^11] | [Delving into E-Commerce Product Retrieval with Vision-Language Pre-training.](http://arxiv.org/abs/2304.04377) | 本文提出了一种基于对比学习的视觉-语言预训练方法，用于解决淘宝搜索的检索问题。该方法采用了针对大规模检索任务的负采样策略，并在真实场景中取得了卓越的性能，目前服务数亿用户。 |
| [^12] | [Manipulating Federated Recommender Systems: Poisoning with Synthetic Users and Its Countermeasures.](http://arxiv.org/abs/2304.03054) | 本文提出了一种新的攻击方法，利用合成的恶意用户上传有毒的梯度来在联邦推荐系统中有效地操纵目标物品的排名和曝光率。在两个真实世界的推荐数据集上进行了大量实验。 |
| [^13] | [Schema-aware Reference as Prompt Improves Data-Efficient Knowledge Graph Construction.](http://arxiv.org/abs/2210.10709) | 提出了一种以检索增强的架构感知参考作为提示的方法，可动态利用人类注释和弱监督数据所继承的架构和知识，指导生成具有更好语义连贯性和一致性的结构化知识，从而在数据效率和知识质量方面具有优越性。 |
| [^14] | [RL4RS: A Real-World Dataset for Reinforcement Learning based Recommender System.](http://arxiv.org/abs/2110.11073) | RL4RS是一个新的基于强化学习的推荐系统数据集，为此提供了一种替代使用人造数据集和半仿真推荐系统数据集的方法，并提出了新的系统评估框架。 |

# 详细

[^1]: PBNR：基于提示的新闻推荐系统

    PBNR: Prompt-based News Recommender System. (arXiv:2304.07862v1 [cs.IR])

    [http://arxiv.org/abs/2304.07862](http://arxiv.org/abs/2304.07862)

    本文提出了一种基于提示学习和预训练的语言模型的新闻推荐系统(PBNR)，它能够更好地捕捉用户的兴趣并提高推荐准确性，实验表明PBNR超过了其他现有方法。

    

    在线新闻平台通常使用个性化的新闻推荐方法帮助用户发现符合他们兴趣的文章。这些方法通常预测用户与候选文章之间的匹配得分，以反映用户对该文章的偏好。一些先前的工作使用语言模型技术，例如注意力机制，根据用户的过去行为来捕捉用户的兴趣，并理解文章的内容。然而，这些现有的模型架构如果考虑了额外的信息，就需要进行调整。预训练的大型语言模型近年来取得了显着发展，这些预训练模型具有更好的单词关系捕捉和理解上下文的能力，并且具有迁移学习和降低下游任务训练时间的优势。与此同时，提示学习是一种新开发的技术，通过为更好的文本生成构建任务特定的指导，利用预训练的语言模型。在本文中，我们提出了一种基于提示的新闻推荐系统（PBNR），它结合了提示学习和预训练的语言模型以增强新闻推荐的准确性。我们在真实数据集上评估了PBNR，并进行了实验来将其与最先进的推荐方法进行比较。结果表明，我们提出的PBNR模型在推荐用户感兴趣的新闻文章方面优于其他方法。

    Online news platforms often use personalized news recommendation methods to help users discover articles that align with their interests. These methods typically predict a matching score between a user and a candidate article to reflect the user's preference for the article. Some previous works have used language model techniques, such as the attention mechanism, to capture users' interests based on their past behaviors, and to understand the content of articles. However, these existing model architectures require adjustments if additional information is taken into account. Pre-trained large language models, which can better capture word relationships and comprehend contexts, have seen a significant development in recent years, and these pre-trained models have the advantages of transfer learning and reducing the training time for downstream tasks. Meanwhile, prompt learning is a newly developed technique that leverages pre-trained language models by building task-specific guidance for
    
[^2]: 面向点击率预测的冷启动多场景排名模型

    Cold-Start based Multi-Scenario Ranking Model for Click-Through Rate Prediction. (arXiv:2304.07858v1 [cs.IR])

    [http://arxiv.org/abs/2304.07858](http://arxiv.org/abs/2304.07858)

    本文提出了一种名为Cold-Start based Multi-scenario Network的方法，用于在在线旅游平台中的点击率预测问题中解决冷启动用户问题。

    

    在本文中，我们关注多场景点击率（CTR）预测，即为所有场景训练一个统一的模型。现有的基于多场景的CTR方法在在线旅游平台（OTPs）领域中存在困难，因为它们忽略了数据非常有限的冷启动用户。为了弥补这一空白，我们提出了一种名为Cold-Start based Multi-scenario Network (CSMN)的新方法。

    Online travel platforms (OTPs), e.g., Ctrip.com or Fliggy.com, can effectively provide travel-related products or services to users. In this paper, we focus on the multi-scenario click-through rate (CTR) prediction, i.e., training a unified model to serve all scenarios. Existing multi-scenario based CTR methods struggle in the context of OTP setting due to the ignorance of the cold-start users who have very limited data. To fill this gap, we propose a novel method named Cold-Start based Multi-scenario Network (CSMN). Specifically, it consists of two basic components including: 1) User Interest Projection Network (UIPN), which firstly purifies users' behaviors by eliminating the scenario-irrelevant information in behaviors with respect to the visiting scenario, followed by obtaining users' scenario-specific interests by summarizing the purified behaviors with respect to the target item via an attention mechanism; and 2) User Representation Memory Network (URMN), which benefits cold-star
    
[^3]: 控制语法简化的句法复杂性鉴别、度量和减少

    Syntactic Complexity Identification, Measurement, and Reduction Through Controlled Syntactic Simplification. (arXiv:2304.07774v1 [cs.CL])

    [http://arxiv.org/abs/2304.07774](http://arxiv.org/abs/2304.07774)

    本研究提出了一种控制简化方法，可在不丢失信息的情况下，基于句子中的实际信息三元组简化句子，以进行知识图谱的创建。

    

    文本简化是自然语言处理中的一个领域，可以通过简化方式探索更易懂的文本。但是，了解并从结构化的文本中提取知识通常很难，因为它通常采用复合句和复杂句式。现有的基于神经网络的方法能够简化句子以提高可读性，同时使用简单的英语替换词和摘要句子和段落。但是，在从结构化的文本中创建知识图谱的过程中，摘要长句子和替换词是不可取的，因为这可能导致信息丢失。因此，本研究提出一种基于句子中的实际信息三元组的控制简化方法。我们提出了一种基于经典句法依存的方法

    Text simplification is one of the domains in Natural Language Processing (NLP) that offers an opportunity to understand the text in a simplified manner for exploration. However, it is always hard to understand and retrieve knowledge from unstructured text, which is usually in the form of compound and complex sentences. There are state-of-the-art neural network-based methods to simplify the sentences for improved readability while replacing words with plain English substitutes and summarising the sentences and paragraphs. In the Knowledge Graph (KG) creation process from unstructured text, summarising long sentences and substituting words is undesirable since this may lead to information loss. However, KG creation from text requires the extraction of all possible facts (triples) with the same mentions as in the text. In this work, we propose a controlled simplification based on the factual information in a sentence, i.e., triple. We present a classical syntactic dependency-based approac
    
[^4]: 序列推荐中的元优化对比学习

    Meta-optimized Contrastive Learning for Sequential Recommendation. (arXiv:2304.07763v1 [cs.IR])

    [http://arxiv.org/abs/2304.07763](http://arxiv.org/abs/2304.07763)

    本文提出了 MCLRec 模型，该模型在数据增强和可学习模型增强操作的基础上，解决了现有对比学习方法难以推广和训练数据不足的问题。

    

    对比学习方法是解决稀疏且含噪声推荐数据的一个新兴方法。然而，现有的对比学习方法要么只针对手工制作的数据进行训练数据和模型增强，要么只使用模型增强方法，这使得模型很难推广。为了更好地训练模型，本文提出了一种称为元优化对比学习的模型。该模型结合了数据增强和可学习模型增强操作。

    Contrastive Learning (CL) performances as a rising approach to address the challenge of sparse and noisy recommendation data. Although having achieved promising results, most existing CL methods only perform either hand-crafted data or model augmentation for generating contrastive pairs to find a proper augmentation operation for different datasets, which makes the model hard to generalize. Additionally, since insufficient input data may lead the encoder to learn collapsed embeddings, these CL methods expect a relatively large number of training data (e.g., large batch size or memory bank) to contrast. However, not all contrastive pairs are always informative and discriminative enough for the training processing. Therefore, a more general CL-based recommendation model called Meta-optimized Contrastive Learning for sequential Recommendation (MCLRec) is proposed in this work. By applying both data augmentation and learnable model augmentation operations, this work innovates the standard 
    
[^5]: 层次化对比表示学习用于知识驱动推荐

    Hierarchical and Contrastive Representation Learning for Knowledge-aware Recommendation. (arXiv:2304.07506v1 [cs.IR])

    [http://arxiv.org/abs/2304.07506](http://arxiv.org/abs/2304.07506)

    本文提出了HiCON框架，运用层次化对比表示学习提高了学习到的节点表示的区分能力，同时采用分层消息聚合机制避免了邻居的指数级扩展，从而有效地解决了知识驱动推荐中的过度平滑问题。

    

    将知识图谱融入推荐是缓解数据稀疏性的有效方法。然而，现有的知识驱动方法通常通过枚举图的邻居进行递归嵌入传播。随着跳数的增加，节点的邻居数呈指数级增长，迫使节点在此递归传播中了解大量邻居以提炼高阶语义相关性。这可能会引入更多有害噪声，导致学习到的节点表示彼此难以区分，即众所周知的过度平滑问题。为了缓解这个问题，我们提出了一种名为HiCON的知识驱动推荐的Hierarchical and Contrastive表示学习框架。具体而言，为了避免邻居的指数级扩展，我们提出了一种分层消息聚合机制，以单独与低阶邻居和元路径受限的高阶邻居交互。此外，为了提高学习表示的区分能力，我们提出了一种对比学习策略，其中鼓励相似的节点相互靠近，而把不相似的节点推开。在三个基准数据集上的实验结果验证了我们所提出方法的有效性。

    Incorporating knowledge graph into recommendation is an effective way to alleviate data sparsity. Most existing knowledge-aware methods usually perform recursive embedding propagation by enumerating graph neighbors. However, the number of nodes' neighbors grows exponentially as the hop number increases, forcing the nodes to be aware of vast neighbors under this recursive propagation for distilling the high-order semantic relatedness. This may induce more harmful noise than useful information into recommendation, leading the learned node representations to be indistinguishable from each other, that is, the well-known over-smoothing issue. To relieve this issue, we propose a Hierarchical and CONtrastive representation learning framework for knowledge-aware recommendation named HiCON. Specifically, for avoiding the exponential expansion of neighbors, we propose a hierarchical message aggregation mechanism to interact separately with low-order neighbors and meta-path-constrained high-order
    
[^6]: 动态表示的时间聚合和传播图神经网络

    Temporal Aggregation and Propagation Graph Neural Networks for Dynamic Representation. (arXiv:2304.07503v1 [cs.LG])

    [http://arxiv.org/abs/2304.07503](http://arxiv.org/abs/2304.07503)

    本文提出了 TAP-GNN，通过整个邻域的时间聚合和传播来有效地建模动态图中的时间关系，从而在图流场景中支持高效的在线推理。

    

    时间图展示了节点之间在连续时间内的动态交互，其拓扑随时间流逝而演变。节点的整个时间邻域显示了节点的变化偏好。然而，为了简化起见，先前的工作通常使用有限的邻居生成动态表示，这导致性能不佳和在线推理延迟高。因此，在本文中，我们提出了一种基于整个邻域的时间图卷积的新方法，即时间聚合和传播图神经网络（TAP-GNN）。具体而言，我们首先通过展开时间图来分析动态表示问题的计算复杂度，使用聚合和传播（AP）块来显著减少历史邻居的重复计算。最终，TAP-GNN支持在图流场景中进行在线推理，既高效又有效地建模动态图中的时间关系。

    Temporal graphs exhibit dynamic interactions between nodes over continuous time, whose topologies evolve with time elapsing.  The whole temporal neighborhood of nodes reveals the varying preferences of nodes.  However, previous works usually generate dynamic representation with limited neighbors for simplicity, which results in both inferior performance and high latency of online inference.  Therefore, in this paper, we propose a novel method of temporal graph convolution with the whole neighborhood, namely Temporal Aggregation and Propagation Graph Neural Networks (TAP-GNN).  Specifically, we firstly analyze the computational complexity of the dynamic representation problem by unfolding the temporal graph in a message-passing paradigm.  The expensive complexity motivates us to design the AP (aggregation and propagation) block, which significantly reduces the repeated computation of historical neighbors.  The final TAP-GNN supports online inference in the graph stream scenario, which i
    
[^7]: 更多不一定就是更好：何时推荐算法在数据丰富的用户中表现不佳？

    More Is Less: When Do Recommenders Underperform for Data-rich Users?. (arXiv:2304.07487v1 [cs.IR])

    [http://arxiv.org/abs/2304.07487](http://arxiv.org/abs/2304.07487)

    研究了推荐算法在数据量丰富和数据量贫乏的用户中的性能表现。发现在所有数据集中，精度在数据丰富的用户中始终更高；平均精度相当，但其方差很大；当评估过程中采用负样本抽样时，召回率产生反直觉结果，表现更好的是数据贫乏的用户；随着用户与推荐系统的互动增加，他们收到的推荐质量会降低。

    

    推荐系统的用户通常在与算法互动的水平上有所不同，这可能影响他们收到推荐的质量，并导致不可取的性能差异。本文研究了对于十个基准数据集应用的一组流行评估指标，数据丰富和数据贫乏的用户性能在什么条件下会发散。我们发现，针对所有数据集，精度在数据丰富的用户中始终更高；平均精度均等，但其方差很大；召回率产生了一个反直觉的结果，算法在数据贫乏的用户中表现更好，当在评估过程中采用负样本抽样时，这种偏差更加严重。最后一个观察结果表明，随着用户与推荐系统的互动增加，他们收到的推荐质量会降低（以召回率衡量）。我们的研究清楚地表明，在现实世界设置中，评估合理的推荐系统很重要，因为不同用户有不同的系统互作程度。

    Users of recommender systems tend to differ in their level of interaction with these algorithms, which may affect the quality of recommendations they receive and lead to undesirable performance disparity. In this paper we investigate under what conditions the performance for data-rich and data-poor users diverges for a collection of popular evaluation metrics applied to ten benchmark datasets. We find that Precision is consistently higher for data-rich users across all the datasets; Mean Average Precision is comparable across user groups but its variance is large; Recall yields a counter-intuitive result where the algorithm performs better for data-poor than for data-rich users, which bias is further exacerbated when negative item sampling is employed during evaluation. The final observation suggests that as users interact more with recommender systems, the quality of recommendations they receive degrades (when measured by Recall). Our insights clearly show the importance of an evaluat
    
[^8]: 自监督辅助损失用于基于音乐相似度检索和自动标注的度量学习

    Self-supervised Auxiliary Loss for Metric Learning in Music Similarity-based Retrieval and Auto-tagging. (arXiv:2304.07449v1 [cs.SD])

    [http://arxiv.org/abs/2304.07449](http://arxiv.org/abs/2304.07449)

    本论文提出了一种自监督学习方法，在自动标注方面已经证明其有效性。我们引入了自监督辅助损失的度量学习方法来解决音乐相似度检索问题，并发现同时使用自监督和监督信号训练模型的优势，而不冻结预训练模型。此外，避免在微调阶段使用数据增强可以提高性能。

    

    在音乐信息检索领域，基于相似度的检索和自动标记是关键组成部分。考虑到人类监督信号的限制性和不可扩展性，让模型从其他来源学习以提高其性能变得至关重要。自监督学习，仅依赖于从音乐音频数据中派生的学习信号，在自动标记的背景下已经证明了其有效性。在这项研究中，我们提出了一种模型，采用自监督学习方法来解决基于相似度的检索问题，并引入了我们的度量学习方法，使用自监督辅助损失。此外，与传统的自监督学习方法不同的是，我们发现了同时使用自监督和监督信号训练模型的优点，而不冻结预训练模型。我们还发现，避免在微调阶段使用数据增强可以提高性能。

    In the realm of music information retrieval, similarity-based retrieval and auto-tagging serve as essential components. Given the limitations and non-scalability of human supervision signals, it becomes crucial for models to learn from alternative sources to enhance their performance. Self-supervised learning, which exclusively relies on learning signals derived from music audio data, has demonstrated its efficacy in the context of auto-tagging. In this study, we propose a model that builds on the self-supervised learning approach to address the similarity-based retrieval challenge by introducing our method of metric learning with a self-supervised auxiliary loss. Furthermore, diverging from conventional self-supervised learning methodologies, we discovered the advantages of concurrently training the model with both self-supervision and supervision signals, without freezing pre-trained models. We also found that refraining from employing augmentation during the fine-tuning phase yields
    
[^9]: 利用句子编码器进行零样本多标签主题推断

    Zero-Shot Multi-Label Topic Inference with Sentence Encoders. (arXiv:2304.07382v1 [cs.CL])

    [http://arxiv.org/abs/2304.07382](http://arxiv.org/abs/2304.07382)

    本文研究了如何利用句子编码器进行“零样本主题推断”任务，并通过实验证明了Sentence-BERT在通用性方面优于其他编码器，而在效率方面则优先选择通用句子编码器。

    

    句子编码器在许多下游文本挖掘任务中表现优秀，因此被认为是相当通用。受到这一启发，我们进行了一项详细研究，探讨如何利用这些句子编码器进行“零样本主题推断”任务，其中主题是由用户实时定义/提供的。在七个不同的数据集上进行的大量实验表明，相比其他编码器，Sentence-BERT表现出卓越的通用性，而当效率成为首要考虑因素时，可以优先选择通用句子编码器。

    Sentence encoders have indeed been shown to achieve superior performances for many downstream text-mining tasks and, thus, claimed to be fairly general. Inspired by this, we performed a detailed study on how to leverage these sentence encoders for the "zero-shot topic inference" task, where the topics are defined/provided by the users in real-time. Extensive experiments on seven different datasets demonstrate that Sentence-BERT demonstrates superior generality compared to other encoders, while Universal Sentence Encoder can be preferred when efficiency is a top priority.
    
[^10]: 扩散推荐模型

    Diffusion Recommender Model. (arXiv:2304.04971v1 [cs.IR])

    [http://arxiv.org/abs/2304.04971](http://arxiv.org/abs/2304.04971)

    本论文提出了一种新颖的扩散推荐模型（DiffRec）来逐步去噪地学习用户交互生成的过程，并针对推荐系统中的冷启动问题和稀疏数据等独特挑战进行了扩展，实验结果显示其在推荐准确性和稳健性方面优于现有方法。

    

    生成模型（如生成对抗网络（GANs）和变分自动编码器（VAEs））被广泛应用于建模用户交互的生成过程。然而，这些生成模型存在固有的局限性，如GANs的不稳定性和VAEs的受限表征能力。这些限制妨碍了复杂用户交互生成过程的准确建模，例如由各种干扰因素导致的嘈杂交互。考虑到扩散模型（DMs）在图像合成方面相对于传统的生成模型具有显着优势，我们提出了一种新颖的扩散推荐模型（称为DiffRec），以逐步去噪的方式学习生成过程。为了保留用户交互中的个性化信息，DiffRec减少了添加的噪声，并避免将用户交互损坏为像图像合成中的纯噪声。此外，我们扩展了传统的DMs以应对实际推荐系统中的独特挑战，如冷启动问题和稀疏的用户-物品交互数据。在几个真实数据集上的实验结果表明，DiffRec在推荐准确性和稳健性方面优于现有方法。

    Generative models such as Generative Adversarial Networks (GANs) and Variational Auto-Encoders (VAEs) are widely utilized to model the generative process of user interactions. However, these generative models suffer from intrinsic limitations such as the instability of GANs and the restricted representation ability of VAEs. Such limitations hinder the accurate modeling of the complex user interaction generation procedure, such as noisy interactions caused by various interference factors. In light of the impressive advantages of Diffusion Models (DMs) over traditional generative models in image synthesis, we propose a novel Diffusion Recommender Model (named DiffRec) to learn the generative process in a denoising manner. To retain personalized information in user interactions, DiffRec reduces the added noises and avoids corrupting users' interactions into pure noises like in image synthesis. In addition, we extend traditional DMs to tackle the unique challenges in practical recommender 
    
[^11]: 探究基于视觉-语言预训练的电商产品检索技术

    Delving into E-Commerce Product Retrieval with Vision-Language Pre-training. (arXiv:2304.04377v1 [cs.IR])

    [http://arxiv.org/abs/2304.04377](http://arxiv.org/abs/2304.04377)

    本文提出了一种基于对比学习的视觉-语言预训练方法，用于解决淘宝搜索的检索问题。该方法采用了针对大规模检索任务的负采样策略，并在真实场景中取得了卓越的性能，目前服务数亿用户。

    

    电商搜索引擎包括检索阶段和排名阶段，其中检索阶段根据用户查询返回候选产品集。最近，将文本信息和视觉线索结合的视觉-语言预训练在检索任务中广受欢迎。本文提出了一种新颖的V+L预训练方法，用于解决淘宝搜索的检索问题。我们设计了一种基于对比学习的视觉预训练任务，优于常规基于回归的视觉预训练任务。此外，我们采用了两种针对大规模检索任务量身定制的负采样策略。除此之外，我们还介绍了我们的方法在真实场景中的在线部署细节。深入的离线/在线实验证明了我们的方法在检索任务中具有卓越的性能。我们的方法被用作淘宝搜索的一个检索通道，并实时服务着数亿用户。

    E-commerce search engines comprise a retrieval phase and a ranking phase, where the first one returns a candidate product set given user queries. Recently, vision-language pre-training, combining textual information with visual clues, has been popular in the application of retrieval tasks. In this paper, we propose a novel V+L pre-training method to solve the retrieval problem in Taobao Search. We design a visual pre-training task based on contrastive learning, outperforming common regression-based visual pre-training tasks. In addition, we adopt two negative sampling schemes, tailored for the large-scale retrieval task. Besides, we introduce the details of the online deployment of our proposed method in real-world situations. Extensive offline/online experiments demonstrate the superior performance of our method on the retrieval task. Our proposed method is employed as one retrieval channel of Taobao Search and serves hundreds of millions of users in real time.
    
[^12]: 操纵联邦推荐系统: 用合成用户进行攻击及其对策

    Manipulating Federated Recommender Systems: Poisoning with Synthetic Users and Its Countermeasures. (arXiv:2304.03054v1 [cs.IR])

    [http://arxiv.org/abs/2304.03054](http://arxiv.org/abs/2304.03054)

    本文提出了一种新的攻击方法，利用合成的恶意用户上传有毒的梯度来在联邦推荐系统中有效地操纵目标物品的排名和曝光率。在两个真实世界的推荐数据集上进行了大量实验。

    

    联邦推荐系统（FedRecs）被认为是一种保护隐私的技术，可以在不共享用户数据的情况下协同学习推荐模型。因为所有参与者都可以通过上传梯度直接影响系统，所以FedRecs容易受到恶意客户的攻击，尤其是利用合成用户进行的攻击更加有效。本文提出了一种新的攻击方法，可以在不依赖任何先前知识的情况下，通过一组合成的恶意用户上传有毒的梯度来有效地操纵目标物品的排名和曝光率。我们在两个真实世界的推荐数据集上对两种广泛使用的FedRecs （Fed-NCF和Fed-LightGCN）进行了大量实验。

    Federated Recommender Systems (FedRecs) are considered privacy-preserving techniques to collaboratively learn a recommendation model without sharing user data. Since all participants can directly influence the systems by uploading gradients, FedRecs are vulnerable to poisoning attacks of malicious clients. However, most existing poisoning attacks on FedRecs are either based on some prior knowledge or with less effectiveness. To reveal the real vulnerability of FedRecs, in this paper, we present a new poisoning attack method to manipulate target items' ranks and exposure rates effectively in the top-$K$ recommendation without relying on any prior knowledge. Specifically, our attack manipulates target items' exposure rate by a group of synthetic malicious users who upload poisoned gradients considering target items' alternative products. We conduct extensive experiments with two widely used FedRecs (Fed-NCF and Fed-LightGCN) on two real-world recommendation datasets. The experimental res
    
[^13]: 以架构感知参考作为提示提高了数据有效的知识图谱构建

    Schema-aware Reference as Prompt Improves Data-Efficient Knowledge Graph Construction. (arXiv:2210.10709v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.10709](http://arxiv.org/abs/2210.10709)

    提出了一种以检索增强的架构感知参考作为提示的方法，可动态利用人类注释和弱监督数据所继承的架构和知识，指导生成具有更好语义连贯性和一致性的结构化知识，从而在数据效率和知识质量方面具有优越性。

    

    随着预训练语言模型的发展，许多基于提示的方法被提出并在数据有效的知识图谱构建中取得了令人瞩目的表现。然而，现有的基于提示的学习方法仍存在几个潜在的限制：（i）自然语言和预定义模式的输出结构化知识之间的语义差距，这意味着模型无法充分利用受限模板的语义知识；（ii）基于局部个体实例的表示学习限制了性能，给定了不充足的特征，这些特征不能释放预先训练语言模型的潜在类比能力。受这些观察的启发，我们提出了一种检索增强的方法，使用检索得到的架构感知参考作为提示，提高了数据有效的知识图谱构建的语义连贯性和一致性。在两个标准数据集上的实验结果表明，相比现有的基于提示和非提示的方法，我们提出的方法在数据效率和知识质量方面具有优越性。

    With the development of pre-trained language models, many prompt-based approaches to data-efficient knowledge graph construction have been proposed and achieved impressive performance. However, existing prompt-based learning methods for knowledge graph construction are still susceptible to several potential limitations: (i) semantic gap between natural language and output structured knowledge with pre-defined schema, which means model cannot fully exploit semantic knowledge with the constrained templates; (ii) representation learning with locally individual instances limits the performance given the insufficient features, which are unable to unleash the potential analogical capability of pre-trained language models. Motivated by these observations, we propose a retrieval-augmented approach, which retrieves schema-aware Reference As Prompt (RAP), for data-efficient knowledge graph construction. It can dynamically leverage schema and knowledge inherited from human-annotated and weak-supe
    
[^14]: RL4RS：一种基于强化学习的推荐系统的真实世界数据集

    RL4RS: A Real-World Dataset for Reinforcement Learning based Recommender System. (arXiv:2110.11073v5 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2110.11073](http://arxiv.org/abs/2110.11073)

    RL4RS是一个新的基于强化学习的推荐系统数据集，为此提供了一种替代使用人造数据集和半仿真推荐系统数据集的方法，并提出了新的系统评估框架。

    

    基于强化学习的推荐系统目标在于从一批收集的数据中学习到一个良好的策略，将推荐问题转化为多步决策任务。然而，现有的基于强化学习的推荐系统研究通常存在巨大的现实差距。本文首次介绍了一个开源的真实世界数据集——RL4RS，旨在取代之前RL-based RS领域由于资源限制而使用的人工和半仿真RS数据集。与学术研究不同的是，RL-based RS面临着部署前需要进行良好验证的困难。我们尝试提出一种新的系统评估框架，包括环境模拟评估、环境评估、反事实策略评估以及建立于测试集的环境评估。综上所述，本文介绍了一个新的资源RL4RS（用于推荐系统的强化学习），并对现实差距等特殊问题进行了思考，并提供了两个真实世界数据集。

    Reinforcement learning based recommender systems (RL-based RS) aim at learning a good policy from a batch of collected data, by casting recommendations to multi-step decision-making tasks. However, current RL-based RS research commonly has a large reality gap. In this paper, we introduce the first open-source real-world dataset, RL4RS, hoping to replace the artificial datasets and semi-simulated RS datasets previous studies used due to the resource limitation of the RL-based RS domain. Unlike academic RL research, RL-based RS suffers from the difficulties of being well-validated before deployment. We attempt to propose a new systematic evaluation framework, including evaluation of environment simulation, evaluation on environments, counterfactual policy evaluation, and evaluation on environments built from test set. In summary, the RL4RS (Reinforcement Learning for Recommender Systems), a new resource with special concerns on the reality gaps, contains two real-world datasets, data und
    

