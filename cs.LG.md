# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Online Federated Learning with Correlated Noise](https://arxiv.org/abs/2403.16542) | 提出一种利用相关噪声提高效用并确保隐私的差分隐私在线联邦学习算法，解决了DP噪声和本地更新带来的挑战，并在动态环境中建立了动态遗憾界。 |
| [^2] | [CDC: A Simple Framework for Complex Data Clustering](https://arxiv.org/abs/2403.03670) | 提出了一个简单但有效的复杂数据聚类框架（CDC），能够以线性复杂度高效处理不同类型的数据，并通过图滤波器和高质量锚点来融合几何结构和属性信息，具有很高的集群能力。 |
| [^3] | [Convergence Analysis of Split Federated Learning on Heterogeneous Data](https://arxiv.org/abs/2402.15166) | 本文填补了分裂联邦学习在各异数据上收敛分析的空白，提供了针对强凸和一般凸目标的SFL收敛分析，收敛速率分别为$O(1/T)$和$O(1/\sqrt[3]{T})。 |
| [^4] | [Mean-Field Analysis for Learning Subspace-Sparse Polynomials with Gaussian Input](https://arxiv.org/abs/2402.08948) | 本文研究了使用随机梯度下降和双层神经网络学习子空间稀疏多项式的均场流动。我们提出了合并阶梯属性的无基础推广，并建立了SGD可学习性的必要条件。此外，我们证明了稍强的条件可以保证损失函数的指数衰减至零。 |
| [^5] | [Synergizing Spatial Optimization with Large Language Models for Open-Domain Urban Itinerary Planning](https://arxiv.org/abs/2402.07204) | 本文提出了Open-domain Urban Itinerary Planning (OUIP)任务，用于根据用户以自然语言描述的请求直接生成行程，通过结合空间优化和大型语言模型(LLM)，提供个性化的城市行程定制服务。 |
| [^6] | [A Two-Scale Complexity Measure for Deep Learning Models.](http://arxiv.org/abs/2401.09184) | 这篇论文介绍了一种用于统计模型的新容量测量2sED，可以可靠地限制泛化误差，并且与训练误差具有很好的相关性。此外，对于深度学习模型，我们展示了如何通过逐层迭代的方法有效地近似2sED，从而处理大量参数的情况。 |
| [^7] | [On the Effect of Contextual Information on Human Delegation Behavior in Human-AI collaboration.](http://arxiv.org/abs/2401.04729) | 本研究探讨了在人工智能协作中提供上下文信息对人类委托行为的影响，发现提供上下文信息显著提高了人工智能与人类团队的表现，并且委托行为在不同上下文信息下发生显著变化。这项研究推进了对人工智能委托中人工智能与人类互动的理解，并为设计更有效的协作系统提供了见解。 |
| [^8] | [HAAQI-Net: A non-intrusive neural music quality assessment model for hearing aids.](http://arxiv.org/abs/2401.01145) | HAAQI-Net是一种适用于助听器用户的非侵入性神经音质评估模型，通过使用BLSTM和注意力机制，以及预训练的BEATs进行声学特征提取，能够快速且准确地预测音乐的HAAQI得分，相比传统方法具有更高的性能和更低的推理时间。 |
| [^9] | [A New Transformation Approach for Uplift Modeling with Binary Outcome.](http://arxiv.org/abs/2310.05549) | 本论文提出了一种新的二元结果提升建模转换方法，利用了零结果样本的信息并且易于使用。 (arXiv:2310.05549v1 [stat.ML]) |
| [^10] | [Optimality of Message-Passing Architectures for Sparse Graphs.](http://arxiv.org/abs/2305.10391) | 本研究证明了将消息传递神经网络应用于稀疏图的节点分类任务是渐近本地贝叶斯最优的，提出了一种实现最优分类器的算法，并将最优分类器的性能理论上与现有学习方法进行了比较。 |
| [^11] | [Explaining the Behavior of Black-Box Prediction Algorithms with Causal Learning.](http://arxiv.org/abs/2006.02482) | 本文提出了一种用于解释黑箱预测算法行为的因果学习方法，通过学习因果图表示来提供因果解释，弥补了现有方法的缺点，即解释单元更加可解释且考虑了宏观级特征和未测量的混淆。 |

# 详细

[^1]: 具有相关噪声的差分隐私在线联邦学习

    Differentially Private Online Federated Learning with Correlated Noise

    [https://arxiv.org/abs/2403.16542](https://arxiv.org/abs/2403.16542)

    提出一种利用相关噪声提高效用并确保隐私的差分隐私在线联邦学习算法，解决了DP噪声和本地更新带来的挑战，并在动态环境中建立了动态遗憾界。

    

    我们提出了一种新颖的差分隐私算法，用于在线联邦学习，利用时间相关的噪声来提高效用同时确保连续发布的模型的隐私性。为了解决源自DP噪声和本地更新带来的流式非独立同分布数据的挑战，我们开发了扰动迭代分析来控制DP噪声对效用的影响。此外，我们展示了在准强凸条件下如何有效管理来自本地更新的漂移误差。在$(\epsilon, \delta)$-DP预算范围内，我们建立了整个时间段上的动态遗憾界，量化了关键参数的影响以及动态环境变化的强度。数值实验证实了所提算法的有效性。

    arXiv:2403.16542v1 Announce Type: new  Abstract: We propose a novel differentially private algorithm for online federated learning that employs temporally correlated noise to improve the utility while ensuring the privacy of the continuously released models. To address challenges stemming from DP noise and local updates with streaming noniid data, we develop a perturbed iterate analysis to control the impact of the DP noise on the utility. Moreover, we demonstrate how the drift errors from local updates can be effectively managed under a quasi-strong convexity condition. Subject to an $(\epsilon, \delta)$-DP budget, we establish a dynamic regret bound over the entire time horizon that quantifies the impact of key parameters and the intensity of changes in dynamic environments. Numerical experiments validate the efficacy of the proposed algorithm.
    
[^2]: CDC：复杂数据聚类的简单框架

    CDC: A Simple Framework for Complex Data Clustering

    [https://arxiv.org/abs/2403.03670](https://arxiv.org/abs/2403.03670)

    提出了一个简单但有效的复杂数据聚类框架（CDC），能够以线性复杂度高效处理不同类型的数据，并通过图滤波器和高质量锚点来融合几何结构和属性信息，具有很高的集群能力。

    

    在当今数据驱动的数字时代，收集到的数据量以及复杂度（如多视图、非欧几里得和多关联性）正在呈指数甚至更快地增长。聚类无监督地从数据中提取有效知识，在实践中非常有用。然而，现有方法独立开发，处理一个特定挑战，牺牲其他挑战。在这项工作中，我们提出了一个简单但有效的复杂数据聚类（CDC）框架，可以以线性复杂度高效处理不同类型的数据。我们首先利用图滤波器融合几何结构和属性信息。然后通过一种新颖的保存相似性的正则化器自适应学习高质量锚点来降低复杂度。我们从理论和实验上说明了我们提出的方法的集群能力。特别是，我们将CDC部署到规模为111M的图数据中。

    arXiv:2403.03670v1 Announce Type: new  Abstract: In today's data-driven digital era, the amount as well as complexity, such as multi-view, non-Euclidean, and multi-relational, of the collected data are growing exponentially or even faster. Clustering, which unsupervisely extracts valid knowledge from data, is extremely useful in practice. However, existing methods are independently developed to handle one particular challenge at the expense of the others. In this work, we propose a simple but effective framework for complex data clustering (CDC) that can efficiently process different types of data with linear complexity. We first utilize graph filtering to fuse geometry structure and attribute information. We then reduce the complexity with high-quality anchors that are adaptively learned via a novel similarity-preserving regularizer. We illustrate the cluster-ability of our proposed method theoretically and experimentally. In particular, we deploy CDC to graph data of size 111M.
    
[^3]: 分布式异构数据上的分裂联邦学习的收敛分析

    Convergence Analysis of Split Federated Learning on Heterogeneous Data

    [https://arxiv.org/abs/2402.15166](https://arxiv.org/abs/2402.15166)

    本文填补了分裂联邦学习在各异数据上收敛分析的空白，提供了针对强凸和一般凸目标的SFL收敛分析，收敛速率分别为$O(1/T)$和$O(1/\sqrt[3]{T})。

    

    分裂联邦学习（SFL）是一种最近的分布式方法，用于在多个客户端之间进行协作模型训练。在SFL中，全局模型通常被分为两部分，其中客户端以并行联邦方式训练一部分，主服务器训练另一部分。尽管最近关于SFL算法发展的研究很多，但SFL的收敛分析在文献中还未有提及，本文旨在弥补这一空白。对SFL进行分析可能比对联邦学习（FL）的分析更具挑战性，这是由于客户端和主服务器之间可能存在双速更新。我们提供了针对异构数据上强凸和一般凸目标的SFL收敛分析。收敛速率分别为$O(1/T)$和$O(1/\sqrt[3]{T})$，其中$T$表示SFL训练的总轮数。我们进一步将分析扩展到非凸目标和一些客户端可能在训练过程中不可用的情况。

    arXiv:2402.15166v1 Announce Type: cross  Abstract: Split federated learning (SFL) is a recent distributed approach for collaborative model training among multiple clients. In SFL, a global model is typically split into two parts, where clients train one part in a parallel federated manner, and a main server trains the other. Despite the recent research on SFL algorithm development, the convergence analysis of SFL is missing in the literature, and this paper aims to fill this gap. The analysis of SFL can be more challenging than that of federated learning (FL), due to the potential dual-paced updates at the clients and the main server. We provide convergence analysis of SFL for strongly convex and general convex objectives on heterogeneous data. The convergence rates are $O(1/T)$ and $O(1/\sqrt[3]{T})$, respectively, where $T$ denotes the total number of rounds for SFL training. We further extend the analysis to non-convex objectives and where some clients may be unavailable during trai
    
[^4]: 使用高斯输入学习子空间稀疏多项式的均场分析

    Mean-Field Analysis for Learning Subspace-Sparse Polynomials with Gaussian Input

    [https://arxiv.org/abs/2402.08948](https://arxiv.org/abs/2402.08948)

    本文研究了使用随机梯度下降和双层神经网络学习子空间稀疏多项式的均场流动。我们提出了合并阶梯属性的无基础推广，并建立了SGD可学习性的必要条件。此外，我们证明了稍强的条件可以保证损失函数的指数衰减至零。

    

    在这项工作中，我们研究了使用随机梯度下降和双层神经网络学习子空间稀疏多项式的均场流动，其中输入分布是标准高斯分布，输出仅依赖于输入在低维子空间上的投影。我们提出了Abbe等人(2022年)中合并阶梯属性的无基础推广，并建立了SGD可学习性的必要条件。此外，我们证明了此条件几乎是充分的，即比必要条件稍强的条件可以保证损失函数的指数衰减至零。

    arXiv:2402.08948v1 Announce Type: new Abstract: In this work, we study the mean-field flow for learning subspace-sparse polynomials using stochastic gradient descent and two-layer neural networks, where the input distribution is standard Gaussian and the output only depends on the projection of the input onto a low-dimensional subspace. We propose a basis-free generalization of the merged-staircase property in Abbe et al. (2022) and establish a necessary condition for the SGD-learnability. In addition, we prove that the condition is almost sufficient, in the sense that a condition slightly stronger than the necessary condition can guarantee the exponential decay of the loss functional to zero.
    
[^5]: 结合空间优化和大型语言模型的开放领域城市行程规划

    Synergizing Spatial Optimization with Large Language Models for Open-Domain Urban Itinerary Planning

    [https://arxiv.org/abs/2402.07204](https://arxiv.org/abs/2402.07204)

    本文提出了Open-domain Urban Itinerary Planning (OUIP)任务，用于根据用户以自然语言描述的请求直接生成行程，通过结合空间优化和大型语言模型(LLM)，提供个性化的城市行程定制服务。

    

    本文首次提出了Open-domain Urban Itinerary Planning (OUIP)任务，用于根据用户以自然语言描述的请求直接生成行程。OUIP与传统行程规划不同，传统规划限制了用户表达更详细的需求，阻碍了真正的个性化。最近，大型语言模型(LLM)在处理多样化任务方面表现出潜力。然而，由于非实时信息、不完整的知识和不足的空间意识，它们无法独立地提供满意的用户体验。鉴于此，我们提出了一个名为ItiNera的OUIP系统，将空间优化与大型语言模型(LLM)相结合，根据用户需求提供个性化的城市行程定制服务。具体来说，我们开发了一个基于LLM的流水线，用于提取和更新兴趣点特征，以创建用户自己的个性化兴趣点数据库。对于每个用户请求，我们利用LLM进行协同实现优化。

    In this paper, we for the first time propose the task of Open-domain Urban Itinerary Planning (OUIP) for citywalk, which directly generates itineraries based on users' requests described in natural language. OUIP is different from conventional itinerary planning, which limits users from expressing more detailed needs and hinders true personalization. Recently, large language models (LLMs) have shown potential in handling diverse tasks. However, due to non-real-time information, incomplete knowledge, and insufficient spatial awareness, they are unable to independently deliver a satisfactory user experience in OUIP. Given this, we present ItiNera, an OUIP system that synergizes spatial optimization with Large Language Models (LLMs) to provide services that customize urban itineraries based on users' needs. Specifically, we develop an LLM-based pipeline for extracting and updating POI features to create a user-owned personalized POI database. For each user request, we leverage LLM in coop
    
[^6]: 深度学习模型的两尺度复杂度测量

    A Two-Scale Complexity Measure for Deep Learning Models. (arXiv:2401.09184v1 [stat.ML])

    [http://arxiv.org/abs/2401.09184](http://arxiv.org/abs/2401.09184)

    这篇论文介绍了一种用于统计模型的新容量测量2sED，可以可靠地限制泛化误差，并且与训练误差具有很好的相关性。此外，对于深度学习模型，我们展示了如何通过逐层迭代的方法有效地近似2sED，从而处理大量参数的情况。

    

    我们引入了一种基于有效维度的统计模型新容量测量2sED。这个新的数量在对模型进行温和假设的情况下，可以可靠地限制泛化误差。此外，对于标准数据集和流行的模型架构的模拟结果表明，2sED与训练误差具有很好的相关性。对于马尔可夫模型，我们展示了如何通过逐层迭代的方法有效地从下方近似2sED，从而解决具有大量参数的深度学习模型。模拟结果表明，这种近似对不同的突出模型和数据集都很好。

    We introduce a novel capacity measure 2sED for statistical models based on the effective dimension. The new quantity provably bounds the generalization error under mild assumptions on the model. Furthermore, simulations on standard data sets and popular model architectures show that 2sED correlates well with the training error. For Markovian models, we show how to efficiently approximate 2sED from below through a layerwise iterative approach, which allows us to tackle deep learning models with a large number of parameters. Simulation results suggest that the approximation is good for different prominent models and data sets.
    
[^7]: 关于上下文信息对人类在人工智能协作中的委托行为的影响

    On the Effect of Contextual Information on Human Delegation Behavior in Human-AI collaboration. (arXiv:2401.04729v1 [cs.HC])

    [http://arxiv.org/abs/2401.04729](http://arxiv.org/abs/2401.04729)

    本研究探讨了在人工智能协作中提供上下文信息对人类委托行为的影响，发现提供上下文信息显著提高了人工智能与人类团队的表现，并且委托行为在不同上下文信息下发生显著变化。这项研究推进了对人工智能委托中人工智能与人类互动的理解，并为设计更有效的协作系统提供了见解。

    

    人工智能的不断增强能力为人工智能与人类的协作带来了新的可能性。利用现有的互补能力，让人们将个别实例委托给人工智能是一种有前景的方法。然而，使人们有效地委托实例需要他们评估自己和人工智能在给定任务的背景下的能力。在这项工作中，我们探讨了在人类决定将实例委托给人工智能时提供上下文信息的效果。我们发现，提供上下文信息显著提高了人工智能与人类团队的表现。此外，我们还表明，当参与者接收到不同类型的上下文信息时，委托行为会发生显著变化。总体而言，这项研究推进了人工智能委托中人工智能与人类互动的理解，并为设计更有效的协作系统提供了可行的见解。

    The constantly increasing capabilities of artificial intelligence (AI) open new possibilities for human-AI collaboration. One promising approach to leverage existing complementary capabilities is allowing humans to delegate individual instances to the AI. However, enabling humans to delegate instances effectively requires them to assess both their own and the AI's capabilities in the context of the given task. In this work, we explore the effects of providing contextual information on human decisions to delegate instances to an AI. We find that providing participants with contextual information significantly improves the human-AI team performance. Additionally, we show that the delegation behavior changes significantly when participants receive varying types of contextual information. Overall, this research advances the understanding of human-AI interaction in human delegation and provides actionable insights for designing more effective collaborative systems.
    
[^8]: HAAQI-Net: 一种适用于助听器的非侵入性神经音质评估模型

    HAAQI-Net: A non-intrusive neural music quality assessment model for hearing aids. (arXiv:2401.01145v1 [eess.AS])

    [http://arxiv.org/abs/2401.01145](http://arxiv.org/abs/2401.01145)

    HAAQI-Net是一种适用于助听器用户的非侵入性神经音质评估模型，通过使用BLSTM和注意力机制，以及预训练的BEATs进行声学特征提取，能够快速且准确地预测音乐的HAAQI得分，相比传统方法具有更高的性能和更低的推理时间。

    

    本文介绍了HAAQI-Net，一种针对助听器用户定制的非侵入性深度学习音质评估模型。与传统方法如Hearing Aid Audio Quality Index (HAAQI) 不同，HAAQI-Net采用了带有注意力机制的双向长短期记忆网络(BLSTM)。该模型以评估的音乐样本和听力损失模式作为输入，生成预测的HAAQI得分。模型采用了预训练的来自音频变换器(BEATs)的双向编码器表示进行声学特征提取。通过将预测分数与真实分数进行比较，HAAQI-Net达到了0.9257的长期一致性相关(LCC)，0.9394的斯皮尔曼等级相关系数(SRCC)，和0.0080的均方误差(MSE)。值得注意的是，这种高性能伴随着推理时间的大幅减少：从62.52秒(HAAQI)减少到2.71秒(HAAQI-Net)，为助听器用户提供了高效的音质评估模型。

    This paper introduces HAAQI-Net, a non-intrusive deep learning model for music quality assessment tailored to hearing aid users. In contrast to traditional methods like the Hearing Aid Audio Quality Index (HAAQI), HAAQI-Net utilizes a Bidirectional Long Short-Term Memory (BLSTM) with attention. It takes an assessed music sample and a hearing loss pattern as input, generating a predicted HAAQI score. The model employs the pre-trained Bidirectional Encoder representation from Audio Transformers (BEATs) for acoustic feature extraction. Comparing predicted scores with ground truth, HAAQI-Net achieves a Longitudinal Concordance Correlation (LCC) of 0.9257, Spearman's Rank Correlation Coefficient (SRCC) of 0.9394, and Mean Squared Error (MSE) of 0.0080. Notably, this high performance comes with a substantial reduction in inference time: from 62.52 seconds (by HAAQI) to 2.71 seconds (by HAAQI-Net), serving as an efficient music quality assessment model for hearing aid users.
    
[^9]: 一个新的二元结果提升建模转换方法

    A New Transformation Approach for Uplift Modeling with Binary Outcome. (arXiv:2310.05549v1 [stat.ML])

    [http://arxiv.org/abs/2310.05549](http://arxiv.org/abs/2310.05549)

    本论文提出了一种新的二元结果提升建模转换方法，利用了零结果样本的信息并且易于使用。 (arXiv:2310.05549v1 [stat.ML])

    

    提升建模在市场营销和客户保留等领域中得到了有效应用，用于针对那些由于活动或治疗更有可能产生反应的客户。本文设计了一种新颖的二元结果转换方法，解锁了零结果样本的全部价值。

    Uplift modeling has been used effectively in fields such as marketing and customer retention, to target those customers who are more likely to respond due to the campaign or treatment. Essentially, it is a machine learning technique that predicts the gain from performing some action with respect to not taking it. A popular class of uplift models is the transformation approach that redefines the target variable with the original treatment indicator. These transformation approaches only need to train and predict the difference in outcomes directly. The main drawback of these approaches is that in general it does not use the information in the treatment indicator beyond the construction of the transformed outcome and usually is not efficient. In this paper, we design a novel transformed outcome for the case of the binary target variable and unlock the full value of the samples with zero outcome. From a practical perspective, our new approach is flexible and easy to use. Experimental resul
    
[^10]: 稀疏图的消息传递架构的最优性

    Optimality of Message-Passing Architectures for Sparse Graphs. (arXiv:2305.10391v1 [cs.LG])

    [http://arxiv.org/abs/2305.10391](http://arxiv.org/abs/2305.10391)

    本研究证明了将消息传递神经网络应用于稀疏图的节点分类任务是渐近本地贝叶斯最优的，提出了一种实现最优分类器的算法，并将最优分类器的性能理论上与现有学习方法进行了比较。

    

    我们研究了特征装饰图上的节点分类问题，在稀疏设置下，即节点的预期度数为节点数的O(1)时。这样的图通常被称为本地树状图。我们引入了一种叫做渐近本地贝叶斯最优性的节点分类任务的贝叶斯最优性概念，并根据这个标准计算了具有任意节点特征和边连接分布的相当一般的统计数据模型的最优分类器。该最优分类器可以使用消息传递图神经网络架构实现。然后我们计算了该分类器的泛化误差，并在一个已经研究充分的统计模型上从理论上与现有的学习方法进行比较。我们发现，在低图信号的情况下，最佳消息传递架构插值于标准MLP和一种典型的c架构之间。

    We study the node classification problem on feature-decorated graphs in the sparse setting, i.e., when the expected degree of a node is $O(1)$ in the number of nodes. Such graphs are typically known to be locally tree-like. We introduce a notion of Bayes optimality for node classification tasks, called asymptotic local Bayes optimality, and compute the optimal classifier according to this criterion for a fairly general statistical data model with arbitrary distributions of the node features and edge connectivity. The optimal classifier is implementable using a message-passing graph neural network architecture. We then compute the generalization error of this classifier and compare its performance against existing learning methods theoretically on a well-studied statistical model with naturally identifiable signal-to-noise ratios (SNRs) in the data. We find that the optimal message-passing architecture interpolates between a standard MLP in the regime of low graph signal and a typical c
    
[^11]: 用因果学习解释黑箱预测算法的行为

    Explaining the Behavior of Black-Box Prediction Algorithms with Causal Learning. (arXiv:2006.02482v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2006.02482](http://arxiv.org/abs/2006.02482)

    本文提出了一种用于解释黑箱预测算法行为的因果学习方法，通过学习因果图表示来提供因果解释，弥补了现有方法的缺点，即解释单元更加可解释且考虑了宏观级特征和未测量的混淆。

    

    因果学方法在解释黑箱预测模型（例如基于图像像素数据训练的深度神经网络）方面越来越受欢迎。然而，现有方法存在两个重要缺点：（i）“解释单元”是相关预测模型的微观级输入，例如图像像素，而不是更有用于理解如何可能改变算法行为的可解释的宏观级特征；（ii）现有方法假设特征与目标模型预测之间不存在未测量的混淆，这在解释单元是宏观级变量时不成立。我们关注的是在分析人员无法访问目标预测算法内部工作原理的重要情况，而只能根据特定输入查询模型输出的能力。为了在这种情况下提供因果解释，我们提出学习因果图表示，允许更好地理解算法的行为。

    Causal approaches to post-hoc explainability for black-box prediction models (e.g., deep neural networks trained on image pixel data) have become increasingly popular. However, existing approaches have two important shortcomings: (i) the "explanatory units" are micro-level inputs into the relevant prediction model, e.g., image pixels, rather than interpretable macro-level features that are more useful for understanding how to possibly change the algorithm's behavior, and (ii) existing approaches assume there exists no unmeasured confounding between features and target model predictions, which fails to hold when the explanatory units are macro-level variables. Our focus is on the important setting where the analyst has no access to the inner workings of the target prediction algorithm, rather only the ability to query the output of the model in response to a particular input. To provide causal explanations in such a setting, we propose to learn causal graphical representations that allo
    

