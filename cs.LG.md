# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Identifiable Latent Neural Causal Models](https://arxiv.org/abs/2403.15711) | 该研究确定了在潜在附加噪声模型背景下导致可识别性的分布变化类型的充分且必要条件，同时提出了当只有部分分布变化满足条件时的部分可识别性结果。 |
| [^2] | [Scaling Laws for Downstream Task Performance of Large Language Models](https://arxiv.org/abs/2402.04177) | 本研究探讨了在转移学习环境中大型语言模型的尺度行为，发现微调数据集的大小和预训练数据与下游数据的分布一致性对下游性能有显著影响。 |
| [^3] | [Better Fair than Sorry: Adversarial Missing Data Imputation for Fair GNNs.](http://arxiv.org/abs/2311.01591) | 该论文提出了一种针对公平GNN的对抗性缺失数据填充模型，以解决现有公平GNN的假设问题。实验证明此模型的有效性。 |
| [^4] | [E2Net: Resource-Efficient Continual Learning with Elastic Expansion Network.](http://arxiv.org/abs/2309.16117) | E2Net是一种资源高效的持续学习方法，通过核心子网蒸馏和精确的回放样本选择，实现了卓越的准确性和较小的遗忘，在相同的计算和存储限制下最大程度地减少了处理时间。 |
| [^5] | [Machine learning for option pricing: an empirical investigation of network architectures.](http://arxiv.org/abs/2307.07657) | 广义高速公路网络结构在期权定价问题中的应用表现出更高的准确性和更短的训练时间。 |

# 详细

[^1]: 可识别的潜在神经因果模型

    Identifiable Latent Neural Causal Models

    [https://arxiv.org/abs/2403.15711](https://arxiv.org/abs/2403.15711)

    该研究确定了在潜在附加噪声模型背景下导致可识别性的分布变化类型的充分且必要条件，同时提出了当只有部分分布变化满足条件时的部分可识别性结果。

    

    因果表征学习旨在从低级观测数据中揭示潜在的高级因果表征。它特别擅长预测在未见分布变化下，因为这些变化通常可以解释为干预的后果。因此，利用{已见}分布变化成为帮助识别因果表征的自然策略，进而有助于预测以前{未见}分布的情况。确定这些分布变化的类型（或条件）对于因果表征的可识别性至关重要。该工作建立了在潜在附加噪声模型背景下，表征导致可识别性的分布变化类型的充分且必要条件。此外，我们提出了当只有部分分布变化满足条件时的部分可识别性结果。

    arXiv:2403.15711v1 Announce Type: new  Abstract: Causal representation learning seeks to uncover latent, high-level causal representations from low-level observed data. It is particularly good at predictions under unseen distribution shifts, because these shifts can generally be interpreted as consequences of interventions. Hence leveraging {seen} distribution shifts becomes a natural strategy to help identifying causal representations, which in turn benefits predictions where distributions are previously {unseen}. Determining the types (or conditions) of such distribution shifts that do contribute to the identifiability of causal representations is critical. This work establishes a {sufficient} and {necessary} condition characterizing the types of distribution shifts for identifiability in the context of latent additive noise models. Furthermore, we present partial identifiability results when only a portion of distribution shifts meets the condition. In addition, we extend our findin
    
[^2]: 大型语言模型的下游任务性能的尺度律

    Scaling Laws for Downstream Task Performance of Large Language Models

    [https://arxiv.org/abs/2402.04177](https://arxiv.org/abs/2402.04177)

    本研究探讨了在转移学习环境中大型语言模型的尺度行为，发现微调数据集的大小和预训练数据与下游数据的分布一致性对下游性能有显著影响。

    

    尺度律提供了重要的见解，可以指导大型语言模型（LLM）的设计。现有研究主要集中在研究预训练（上游）损失的尺度律。然而，在转移学习环境中，LLM先在无监督数据集上进行预训练，然后在下游任务上进行微调，我们通常也关心下游性能。在这项工作中，我们研究了在转移学习环境中的尺度行为，其中LLM被微调用于机器翻译任务。具体而言，我们研究了预训练数据的选择和大小对下游性能（翻译质量）的影响，使用了两个评价指标：下游交叉熵和BLEU分数。我们的实验证明，微调数据集的大小和预训练数据与下游数据的分布一致性显著影响尺度行为。在充分一致性情况下，下游交叉熵和BLEU分数都会逐渐提升。

    Scaling laws provide important insights that can guide the design of large language models (LLMs). Existing work has primarily focused on studying scaling laws for pretraining (upstream) loss. However, in transfer learning settings, in which LLMs are pretrained on an unsupervised dataset and then finetuned on a downstream task, we often also care about the downstream performance. In this work, we study the scaling behavior in a transfer learning setting, where LLMs are finetuned for machine translation tasks. Specifically, we investigate how the choice of the pretraining data and its size affect downstream performance (translation quality) as judged by two metrics: downstream cross-entropy and BLEU score. Our experiments indicate that the size of the finetuning dataset and the distribution alignment between the pretraining and downstream data significantly influence the scaling behavior. With sufficient alignment, both downstream cross-entropy and BLEU score improve monotonically with 
    
[^3]: 更好的公平性胜于遗憾：针对公平GNN的对抗性缺失数据填充

    Better Fair than Sorry: Adversarial Missing Data Imputation for Fair GNNs. (arXiv:2311.01591v1 [cs.LG])

    [http://arxiv.org/abs/2311.01591](http://arxiv.org/abs/2311.01591)

    该论文提出了一种针对公平GNN的对抗性缺失数据填充模型，以解决现有公平GNN的假设问题。实验证明此模型的有效性。

    

    本文解决了在缺失保护属性的情况下学习公平图神经网络（GNNs）的问题。在许多相关任务中，决策可能会对特定社区产生不成比例的影响，而GNNs已经在这些任务中取得了最先进的结果。然而，现有的公平GNNs工作要么假设保护属性是完全被观察到的，要么假设缺失数据的填充是公平的。实际上，填充中的偏差会传播到模型的结果中，导致它们过高地估计了其预测的公平性。我们通过提出Better Fair than Sorry（BFtS），为公平GNNs使用的保护属性的公平缺失数据填充模型来解决这个挑战。BFtS背后的关键设计原则是填充应该近似于公平GNN的最困难情况，即在最优化公平性最困难的情况下。我们使用一个三方对抗方案来实现这个想法，在这个方案中，两个对手共同对抗公平GNN。通过使用合成和实际数据集的实验证明了BFtS的有效性。

    This paper addresses the problem of learning fair Graph Neural Networks (GNNs) under missing protected attributes. GNNs have achieved state-of-the-art results in many relevant tasks where decisions might disproportionately impact specific communities. However, existing work on fair GNNs assumes that either protected attributes are fully-observed or that the missing data imputation is fair. In practice, biases in the imputation will be propagated to the model outcomes, leading them to overestimate the fairness of their predictions. We address this challenge by proposing Better Fair than Sorry (BFtS), a fair missing data imputation model for protected attributes used by fair GNNs. The key design principle behind BFtS is that imputations should approximate the worst-case scenario for the fair GNN -- i.e. when optimizing fairness is the hardest. We implement this idea using a 3-player adversarial scheme where two adversaries collaborate against the fair GNN. Experiments using synthetic and
    
[^4]: E2Net: 弹性扩展网络实现资源高效的持续学习

    E2Net: Resource-Efficient Continual Learning with Elastic Expansion Network. (arXiv:2309.16117v1 [cs.LG])

    [http://arxiv.org/abs/2309.16117](http://arxiv.org/abs/2309.16117)

    E2Net是一种资源高效的持续学习方法，通过核心子网蒸馏和精确的回放样本选择，实现了卓越的准确性和较小的遗忘，在相同的计算和存储限制下最大程度地减少了处理时间。

    

    持续学习方法旨在学习新任务而不消除以前的知识。然而，持续学习通常需要大量的计算能力和存储容量才能达到令人满意的性能。在本文中，我们提出了一种资源高效的持续学习方法，称为弹性扩展网络（E2Net）。通过核心子网蒸馏和精确的回放样本选择，E2Net在相同的计算和存储限制下实现了卓越的平均准确性和较小的遗忘，并最大程度地减少了处理时间。在E2Net中，我们提出了代表性网络蒸馏，通过评估参数数量和与工作网络的输出相似性来识别代表性的核心子网，蒸馏工作网络内的类似子网以减轻对重演缓冲区的依赖，并促进跨先前任务的知识转移。为了提高存储资源利用率，我们还提出了子网约束经验回放方法。

    Continual Learning methods are designed to learn new tasks without erasing previous knowledge. However, Continual Learning often requires massive computational power and storage capacity for satisfactory performance. In this paper, we propose a resource-efficient continual learning method called the Elastic Expansion Network (E2Net). Leveraging core subnet distillation and precise replay sample selection, E2Net achieves superior average accuracy and diminished forgetting within the same computational and storage constraints, all while minimizing processing time. In E2Net, we propose Representative Network Distillation to identify the representative core subnet by assessing parameter quantity and output similarity with the working network, distilling analogous subnets within the working network to mitigate reliance on rehearsal buffers and facilitating knowledge transfer across previous tasks. To enhance storage resource utilization, we then propose Subnet Constraint Experience Replay t
    
[^5]: 机器学习用于期权定价：对网络结构的实证研究

    Machine learning for option pricing: an empirical investigation of network architectures. (arXiv:2307.07657v1 [q-fin.CP])

    [http://arxiv.org/abs/2307.07657](http://arxiv.org/abs/2307.07657)

    广义高速公路网络结构在期权定价问题中的应用表现出更高的准确性和更短的训练时间。

    

    本文考虑了使用适当的输入数据（模型参数）和相应输出数据（期权价格或隐含波动率）来学习期权价格或隐含波动率的监督学习问题。大部分相关文献都使用（普通的）前馈神经网络结构来连接用于学习将输入映射到输出的神经元。在本文中，受到图像分类方法和用于偏微分方程机器学习方法的最新进展的启发，我们通过实证研究来探究网络结构的选择如何影响机器学习算法的精确度和训练时间。我们发现，在期权定价问题中，我们主要关注Black-Scholes和Heston模型，广义高速公路网络结构相较于其他变体在均方误差和训练时间方面表现更好。此外，在计算隐含波动率方面，

    We consider the supervised learning problem of learning the price of an option or the implied volatility given appropriate input data (model parameters) and corresponding output data (option prices or implied volatilities). The majority of articles in this literature considers a (plain) feed forward neural network architecture in order to connect the neurons used for learning the function mapping inputs to outputs. In this article, motivated by methods in image classification and recent advances in machine learning methods for PDEs, we investigate empirically whether and how the choice of network architecture affects the accuracy and training time of a machine learning algorithm. We find that for option pricing problems, where we focus on the Black--Scholes and the Heston model, the generalized highway network architecture outperforms all other variants, when considering the mean squared error and the training time as criteria. Moreover, for the computation of the implied volatility, a
    

