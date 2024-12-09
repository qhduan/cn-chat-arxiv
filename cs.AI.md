# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond the Answers: Reviewing the Rationality of Multiple Choice Question Answering for the Evaluation of Large Language Models](https://rss.arxiv.org/abs/2402.01349) | 对于评估大型语言模型中多选题回答的合理性进行了回顾，发现当前基于多选题回答的基准可能无法充分捕捉大型语言模型的真实能力。 |
| [^2] | [Colour and Brush Stroke Pattern Recognition in Abstract Art using Modified Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/2403.18397) | 本文通过引入改进的深度卷积生成对抗网络(mDCGAN)，针对高质量艺术品生成进行了研究，解决了普遍训练问题，有效探索抽象绘画中的颜色和笔触模式。 |
| [^3] | [All-in-One: Heterogeneous Interaction Modeling for Cold-Start Rating Prediction](https://arxiv.org/abs/2403.17740) | 提出了异质交互评分网络（HIRE）框架，通过异质交互模块（HIM）来共同建模异质交互并直接推断重要特征 |
| [^4] | [Masked Attention is All You Need for Graphs](https://arxiv.org/abs/2402.10793) | 提出了一种在图上学习的简单替代方法，称为掩码注意力（MAG），其利用注意力矩阵来创建定制的注意力模式，在长距离任务上表现出色并胜过其他方法。 |
| [^5] | [Graph Inference Acceleration by Learning MLPs on Graphs without Supervision](https://arxiv.org/abs/2402.08918) | 该论文提出了一个简单而有效的框架SimMLP，通过在图上无监督学习MLPs，提高了在延迟敏感的应用中的泛化能力。 |
| [^6] | [PAC Privacy Preserving Diffusion Models](https://arxiv.org/abs/2312.01201) | 提出了一种PAC隐私保护扩散模型，通过将私有分类器指导集成到采样过程中增强隐私保护，并发展了一种新的度量标准来衡量隐私水平，在保护性能方面表现出卓越表现。 |
| [^7] | [A Simple Data Augmentation for Feature Distribution Skewed Federated Learning.](http://arxiv.org/abs/2306.09363) | 本文针对特征分布偏斜的联邦学习提出了FedRDN方法，在输入层级上实现了数据增强，将整个联邦数据集的统计信息注入到本地客户端数据中，以缓解特征漂移问题。 |

# 详细

[^1]: 超越答案：对于评估大型语言模型中多选题回答的合理性的回顾

    Beyond the Answers: Reviewing the Rationality of Multiple Choice Question Answering for the Evaluation of Large Language Models

    [https://rss.arxiv.org/abs/2402.01349](https://rss.arxiv.org/abs/2402.01349)

    对于评估大型语言模型中多选题回答的合理性进行了回顾，发现当前基于多选题回答的基准可能无法充分捕捉大型语言模型的真实能力。

    

    在自然语言处理领域，大型语言模型（LLMs）引发了一场范式转变，显著提升了自然语言生成任务的性能。尽管取得了这些进展，对LLMs的全面评估仍然是社区面临的必然挑战。最近，将多选题回答（MCQA）作为LLMs的基准已经引起了广泛关注。本研究调查了MCQA作为LLMs评估方法的合理性。如果LLMs真正理解问题的语义，它们的性能应该在从相同问题派生的各种配置上表现一致。然而，我们的实证结果表明LLMs的响应一致性存在显著差异，我们将之定义为LLMs的响应可变性综合征（REVAS），这表明目前基于MCQA的基准可能无法充分捕捉LLMs的真实能力，强调了对更合适的评估方法的需要。

    In the field of natural language processing (NLP), Large Language Models (LLMs) have precipitated a paradigm shift, markedly enhancing performance in natural language generation tasks. Despite these advancements, the comprehensive evaluation of LLMs remains an inevitable challenge for the community. Recently, the utilization of Multiple Choice Question Answering (MCQA) as a benchmark for LLMs has gained considerable traction. This study investigates the rationality of MCQA as an evaluation method for LLMs. If LLMs genuinely understand the semantics of questions, their performance should exhibit consistency across the varied configurations derived from the same questions. Contrary to this expectation, our empirical findings suggest a notable disparity in the consistency of LLM responses, which we define as REsponse VAriability Syndrome (REVAS) of the LLMs, indicating that current MCQA-based benchmarks may not adequately capture the true capabilities of LLMs, which underscores the need f
    
[^2]: 使用改进的深度卷积生成对抗网络在抽象艺术中进行颜色和笔触模式识别

    Colour and Brush Stroke Pattern Recognition in Abstract Art using Modified Deep Convolutional Generative Adversarial Networks

    [https://arxiv.org/abs/2403.18397](https://arxiv.org/abs/2403.18397)

    本文通过引入改进的深度卷积生成对抗网络(mDCGAN)，针对高质量艺术品生成进行了研究，解决了普遍训练问题，有效探索抽象绘画中的颜色和笔触模式。

    

    抽象艺术是一种广受欢迎、被广泛讨论的艺术形式，通常能够描绘出艺术家的情感。许多研究人员尝试使用机器学习和深度学习的边缘检测、笔触和情感识别算法来研究抽象艺术。本文描述了使用生成对抗神经网络(GAN)对广泛分布的抽象绘画进行研究。 GAN具有学习和再现分布的能力，使研究人员能够有效地探索和研究生成的图像空间。然而，挑战在于开发一种能够克服常见训练问题的高效GAN架构。本文通过引入专门设计用于高质量艺术品生成的改进DCGAN(mDCGAN)来解决这一挑战。该方法涉及对所做修改的深入探讨，深入研究DCGAN的复杂工作。

    arXiv:2403.18397v1 Announce Type: cross  Abstract: Abstract Art is an immensely popular, discussed form of art that often has the ability to depict the emotions of an artist. Many researchers have made attempts to study abstract art in the form of edge detection, brush stroke and emotion recognition algorithms using machine and deep learning. This papers describes the study of a wide distribution of abstract paintings using Generative Adversarial Neural Networks(GAN). GANs have the ability to learn and reproduce a distribution enabling researchers and scientists to effectively explore and study the generated image space. However, the challenge lies in developing an efficient GAN architecture that overcomes common training pitfalls. This paper addresses this challenge by introducing a modified-DCGAN (mDCGAN) specifically designed for high-quality artwork generation. The approach involves a thorough exploration of the modifications made, delving into the intricate workings of DCGANs, opt
    
[^3]: 一体化：异质交互建模用于冷启动评分预测

    All-in-One: Heterogeneous Interaction Modeling for Cold-Start Rating Prediction

    [https://arxiv.org/abs/2403.17740](https://arxiv.org/abs/2403.17740)

    提出了异质交互评分网络（HIRE）框架，通过异质交互模块（HIM）来共同建模异质交互并直接推断重要特征

    

    冷启动评分预测是推荐系统中一个基本问题，已得到广泛研究。许多方法已经被提出，利用现有数据之间的显式关系，例如协同过滤、社交推荐和异构信息网络，以缓解冷启动用户和物品的数据不足问题。然而，基于不同角色之间的数据构建的显式关系可能不可靠且无关，从而限制了特定推荐任务的性能上限。受此启发，本文提出了一个灵活的框架，名为异质交互评分网络（HIRE）。HIRE不仅仅依赖于预先定义的交互模式或手动构建的异构信息网络。相反，我们设计了一个异质交互模块（HIM），来共同建模异质交互并直接推断重要特征。

    arXiv:2403.17740v1 Announce Type: cross  Abstract: Cold-start rating prediction is a fundamental problem in recommender systems that has been extensively studied. Many methods have been proposed that exploit explicit relations among existing data, such as collaborative filtering, social recommendations and heterogeneous information network, to alleviate the data insufficiency issue for cold-start users and items. However, the explicit relations constructed based on data between different roles may be unreliable and irrelevant, which limits the performance ceiling of the specific recommendation task. Motivated by this, in this paper, we propose a flexible framework dubbed heterogeneous interaction rating network (HIRE). HIRE dose not solely rely on the pre-defined interaction pattern or the manually constructed heterogeneous information network. Instead, we devise a Heterogeneous Interaction Module (HIM) to jointly model the heterogeneous interactions and directly infer the important in
    
[^4]: 掩码注意力是图的关键

    Masked Attention is All You Need for Graphs

    [https://arxiv.org/abs/2402.10793](https://arxiv.org/abs/2402.10793)

    提出了一种在图上学习的简单替代方法，称为掩码注意力（MAG），其利用注意力矩阵来创建定制的注意力模式，在长距离任务上表现出色并胜过其他方法。

    

    图神经网络（GNNs）和消息传递算法的变种主要用于在图上学习，这在很大程度上归功于它们的灵活性、速度和令人满意的性能。然而，设计强大而通用的GNNs需要大量的研究工作，通常依赖于精心选择的手工制作的消息传递操作符。受此启发，我们提出了一种在图上学习的非常简单的替代方法，它完全依赖于注意力。图被表示为节点或边集，并通过掩码注意权重矩阵来强制它们的连接，有效地为每个图创建定制的注意力模式。尽管其简单性，用于图的掩码注意力（MAG）在长距离任务上表现出色，并在55多个节点和图级任务上优于强消息传递基线和更复杂的基于注意力的方法。

    arXiv:2402.10793v1 Announce Type: cross  Abstract: Graph neural networks (GNNs) and variations of the message passing algorithm are the predominant means for learning on graphs, largely due to their flexibility, speed, and satisfactory performance. The design of powerful and general purpose GNNs, however, requires significant research efforts and often relies on handcrafted, carefully-chosen message passing operators. Motivated by this, we propose a remarkably simple alternative for learning on graphs that relies exclusively on attention. Graphs are represented as node or edge sets and their connectivity is enforced by masking the attention weight matrix, effectively creating custom attention patterns for each graph. Despite its simplicity, masked attention for graphs (MAG) has state-of-the-art performance on long-range tasks and outperforms strong message passing baselines and much more involved attention-based methods on over 55 node and graph-level tasks. We also show significantly 
    
[^5]: 通过无监督在图上学习多层感知机（MLP）加速图推理

    Graph Inference Acceleration by Learning MLPs on Graphs without Supervision

    [https://arxiv.org/abs/2402.08918](https://arxiv.org/abs/2402.08918)

    该论文提出了一个简单而有效的框架SimMLP，通过在图上无监督学习MLPs，提高了在延迟敏感的应用中的泛化能力。

    

    图神经网络（GNNs）已经在各种图学习任务中展示出了有效性，但是它们对消息传递的依赖限制了它们在延迟敏感的应用中的部署，比如金融欺诈检测。最近的研究探索了从GNNs中提取知识到多层感知机（MLPs）来加速推理。然而，这种任务特定的有监督蒸馏限制了对未见节点的泛化，而在延迟敏感的应用中这种情况很常见。为此，我们提出了一种简单而有效的框架SimMLP，用于在图上无监督学习MLPs，以增强泛化能力。SimMLP利用自监督对齐GNNs和MLPs之间的节点特征和图结构之间的精细和泛化的相关性，并提出了两种策略来减轻平凡解的风险。从理论上讲，

    arXiv:2402.08918v1 Announce Type: cross Abstract: Graph Neural Networks (GNNs) have demonstrated effectiveness in various graph learning tasks, yet their reliance on message-passing constraints their deployment in latency-sensitive applications such as financial fraud detection. Recent works have explored distilling knowledge from GNNs to Multi-Layer Perceptrons (MLPs) to accelerate inference. However, this task-specific supervised distillation limits generalization to unseen nodes, which are prevalent in latency-sensitive applications. To this end, we present \textbf{\textsc{SimMLP}}, a \textbf{\textsc{Sim}}ple yet effective framework for learning \textbf{\textsc{MLP}}s on graphs without supervision, to enhance generalization. \textsc{SimMLP} employs self-supervised alignment between GNNs and MLPs to capture the fine-grained and generalizable correlation between node features and graph structures, and proposes two strategies to alleviate the risk of trivial solutions. Theoretically, w
    
[^6]: PAC隐私保护扩散模型

    PAC Privacy Preserving Diffusion Models

    [https://arxiv.org/abs/2312.01201](https://arxiv.org/abs/2312.01201)

    提出了一种PAC隐私保护扩散模型，通过将私有分类器指导集成到采样过程中增强隐私保护，并发展了一种新的度量标准来衡量隐私水平，在保护性能方面表现出卓越表现。

    

    数据隐私保护正在引起研究人员的越来越多的关注。扩散模型（DMs），尤其是具有严格的差分隐私，有可能生成既具有高隐私性又具有良好视觉质量的图像。然而，挑战在于确保在私有化特定数据属性时的强大保护，当前模型在这些方面经常存在不足。为了解决这些挑战，我们引入了PAC隐私保护扩散模型，这是一种利用扩散原理并确保“可能大致正确（PAC）”隐私性的模型。我们通过将私有分类器指导集成到Langevin采样过程中来增强隐私保护。此外，认识到在衡量模型隐私性方面存在差距，我们开发了一种新的度量标准来衡量隐私水平。我们的模型通过这个新度量标准评估，并通过高斯矩阵计算支持PAC界限，表现出更优异的隐私性能。

    arXiv:2312.01201v2 Announce Type: replace-cross  Abstract: Data privacy protection is garnering increased attention among researchers. Diffusion models (DMs), particularly with strict differential privacy, can potentially produce images with both high privacy and visual quality. However, challenges arise such as in ensuring robust protection in privatizing specific data attributes, areas where current models often fall short. To address these challenges, we introduce the PAC Privacy Preserving Diffusion Model, a model leverages diffusion principles and ensure Probably Approximately Correct (PAC) privacy. We enhance privacy protection by integrating a private classifier guidance into the Langevin Sampling Process. Additionally, recognizing the gap in measuring the privacy of models, we have developed a novel metric to gauge privacy levels. Our model, assessed with this new metric and supported by Gaussian matrix computations for the PAC bound, has shown superior performance in privacy p
    
[^7]: 一种简单的面向特征分布偏斜联邦学习的数据增强方法

    A Simple Data Augmentation for Feature Distribution Skewed Federated Learning. (arXiv:2306.09363v1 [cs.LG])

    [http://arxiv.org/abs/2306.09363](http://arxiv.org/abs/2306.09363)

    本文针对特征分布偏斜的联邦学习提出了FedRDN方法，在输入层级上实现了数据增强，将整个联邦数据集的统计信息注入到本地客户端数据中，以缓解特征漂移问题。

    

    联邦学习（FL）是一种分布式协作学习方法，可以确保隐私保护。然而，由于数据异构性（即非独立同分布数据），它的性能必然受到影响。本文针对特征分布偏斜的FL场景展开研究，提出了一种通用的数据增强方法，以减轻由本地数据集之间潜在分布不同导致的特征漂移问题。

    Federated learning (FL) facilitates collaborative learning among multiple clients in a distributed manner, while ensuring privacy protection. However, its performance is inevitably degraded as suffering data heterogeneity, i.e., non-IID data. In this paper, we focus on the feature distribution skewed FL scenario, which is widespread in real-world applications. The main challenge lies in the feature shift caused by the different underlying distributions of local datasets. While the previous attempts achieved progress, few studies pay attention to the data itself, the root of this issue. Therefore, the primary goal of this paper is to develop a general data augmentation technique at the input level, to mitigate the feature shift. To achieve this goal, we propose FedRDN, a simple yet remarkably effective data augmentation method for feature distribution skewed FL, which randomly injects the statistics of the dataset from the entire federation into the client's data. By this, our method ca
    

