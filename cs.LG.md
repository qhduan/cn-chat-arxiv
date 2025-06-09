# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SARI: Simplistic Average and Robust Identification based Noisy Partial Label Learning](https://arxiv.org/abs/2402.04835) | SARI是一个简约的框架，通过利用嘈杂部分标签，结合平均策略和识别策略，实现了部分标签学习中的深度神经网络分类器训练，并显著提升了准确性。 |
| [^2] | [Theoretical and experimental study of SMOTE: limitations and comparisons of rebalancing strategies](https://arxiv.org/abs/2402.03819) | SMOTE是一种处理不平衡数据集的常用重新平衡策略，它通过复制原始少数样本来重新生成原始分布。本研究证明了SMOTE的密度在少数样本分布的边界附近逐渐减小，从而验证了BorderLine SMOTE策略的合理性。此外，研究还提出了两种新的SMOTE相关策略，并与其他重新平衡方法进行了比较。最终发现，在数据集极度不平衡的情况下，SMOTE、提出的方法或欠采样程序是最佳的策略。 |
| [^3] | [A Fisher-Rao gradient flow for entropy-regularised Markov decision processes in Polish spaces.](http://arxiv.org/abs/2310.02951) | 该论文研究了在Polish空间中的熵正则化Markov决策过程上的Fisher-Rao策略梯度流的全局收敛性和指数收敛性，并证明了梯度流在梯度评估方面的稳定性，为自然策略梯度流的性能提供了洞见。 |
| [^4] | [MCMC-Correction of Score-Based Diffusion Models for Model Composition.](http://arxiv.org/abs/2307.14012) | 本文提出了一种修正基于得分的扩散模型的方法，使其能够与各种MCMC方法结合，从而实现模型组合和进行更好的采样。 |
| [^5] | [Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning.](http://arxiv.org/abs/2307.04726) | 该论文介绍了一种名为状态重构扩散策略 (SRDP) 的新方法，该方法在最新的扩散策略类中引入了状态重构特征学习，以解决脱机强化学习中的分布偏移和有效表示策略的问题。 |
| [^6] | [Graph Neural Networks Provably Benefit from Structural Information: A Feature Learning Perspective.](http://arxiv.org/abs/2306.13926) | 本论文研究了GNN在神经网络特征学习理论中的作用。 发现图卷积网络显著增强了良性过拟合区域，在这个区域内信号学习超越了噪声记忆。 |
| [^7] | [Infinite-Dimensional Diffusion Models.](http://arxiv.org/abs/2302.10130) | 该论文提出了一种在无限维度中直接制定扩散基于的生成模型的方法，相比于传统的先离散化再应用扩散模型的方法，这种方法能够避免参数细化导致算法性能下降，同时提供了维度无关的距离界限，为无限维扩散模型设计提供了准则。 |
| [^8] | [ByzSecAgg: A Byzantine-Resistant Secure Aggregation Scheme for Federated Learning Based on Coded Computing and Vector Commitment.](http://arxiv.org/abs/2302.09913) | 本文提出了一种基于编码计算和向量承诺的拜占庭抵抗安全聚合方案，用于联邦学习。该方案通过RAM秘密共享将本地更新分割成较小子向量，并使用双重RAMP共享技术实现成对距离的安全计算。 |

# 详细

[^1]: SARI: 简洁平均与鲁棒性基于嘈杂部分标签学习

    SARI: Simplistic Average and Robust Identification based Noisy Partial Label Learning

    [https://arxiv.org/abs/2402.04835](https://arxiv.org/abs/2402.04835)

    SARI是一个简约的框架，通过利用嘈杂部分标签，结合平均策略和识别策略，实现了部分标签学习中的深度神经网络分类器训练，并显著提升了准确性。

    

    部分标签学习 (PLL) 是一种弱监督学习范式，其中每个训练实例都与一组候选标签 (部分标签) 成对，其中一个是真正的标签。嘈杂部分标签学习 (NPLL) 放宽了这个约束，允许一些部分标签不包含真正的标签，增加了问题的实用性。我们的工作集中在 NPLL 上，并提出了一个简约的框架 SARI，通过利用加权最近邻算法将伪标签分配给图像。然后，这些伪标签与图像配对用于训练深度神经网络分类器，采用标签平滑和标准正则化技术。随后，利用分类器的特征和预测结果来改进和提高伪标签的准确性。SARI结合了文献中基于平均策略 (伪标签) 和基于识别策略 (分类器训练)的优点。我们进行了详尽的实验评估，验证了SARI的有效性和性能提升。

    Partial label learning (PLL) is a weakly-supervised learning paradigm where each training instance is paired with a set of candidate labels (partial label), one of which is the true label. Noisy PLL (NPLL) relaxes this constraint by allowing some partial labels to not contain the true label, enhancing the practicality of the problem. Our work centers on NPLL and presents a minimalistic framework called SARI that initially assigns pseudo-labels to images by exploiting the noisy partial labels through a weighted nearest neighbour algorithm. These pseudo-label and image pairs are then used to train a deep neural network classifier with label smoothing and standard regularization techniques. The classifier's features and predictions are subsequently employed to refine and enhance the accuracy of pseudo-labels. SARI combines the strengths of Average Based Strategies (in pseudo labelling) and Identification Based Strategies (in classifier training) from the literature. We perform thorough ex
    
[^2]: SMOTE的理论和实验研究：关于重新平衡策略的限制和比较

    Theoretical and experimental study of SMOTE: limitations and comparisons of rebalancing strategies

    [https://arxiv.org/abs/2402.03819](https://arxiv.org/abs/2402.03819)

    SMOTE是一种处理不平衡数据集的常用重新平衡策略，它通过复制原始少数样本来重新生成原始分布。本研究证明了SMOTE的密度在少数样本分布的边界附近逐渐减小，从而验证了BorderLine SMOTE策略的合理性。此外，研究还提出了两种新的SMOTE相关策略，并与其他重新平衡方法进行了比较。最终发现，在数据集极度不平衡的情况下，SMOTE、提出的方法或欠采样程序是最佳的策略。

    

    SMOTE（Synthetic Minority Oversampling Technique）是处理不平衡数据集常用的重新平衡策略。我们证明了在渐进情况下，SMOTE（默认参数）通过简单复制原始少数样本来重新生成原始分布。我们还证明了在少数样本分布的支持边界附近，SMOTE的密度会减小，从而验证了常见的BorderLine SMOTE策略。随后，我们提出了两种新的SMOTE相关策略，并将它们与现有的重新平衡方法进行了比较。我们发现，只有当数据集极度不平衡时才需要重新平衡策略。对于这种数据集，SMOTE、我们提出的方法或欠采样程序是最佳的策略。

    Synthetic Minority Oversampling Technique (SMOTE) is a common rebalancing strategy for handling imbalanced data sets. Asymptotically, we prove that SMOTE (with default parameter) regenerates the original distribution by simply copying the original minority samples. We also prove that SMOTE density vanishes near the boundary of the support of the minority distribution, therefore justifying the common BorderLine SMOTE strategy. Then we introduce two new SMOTE-related strategies, and compare them with state-of-the-art rebalancing procedures. We show that rebalancing strategies are only required when the data set is highly imbalanced. For such data sets, SMOTE, our proposals, or undersampling procedures are the best strategies.
    
[^3]: Fisher-Rao梯度流在Polish空间中对熵正则化Markov决策过程的研究

    A Fisher-Rao gradient flow for entropy-regularised Markov decision processes in Polish spaces. (arXiv:2310.02951v1 [math.OC])

    [http://arxiv.org/abs/2310.02951](http://arxiv.org/abs/2310.02951)

    该论文研究了在Polish空间中的熵正则化Markov决策过程上的Fisher-Rao策略梯度流的全局收敛性和指数收敛性，并证明了梯度流在梯度评估方面的稳定性，为自然策略梯度流的性能提供了洞见。

    

    我们研究了在Polish状态和动作空间中无限时域的熵正则化的Markov决策过程的Fisher-Rao策略梯度流的全局收敛性。这个流是策略镜像下降方法的连续时间类比。我们证明了梯度流的全局良定义性，并展示了它对最优策略的指数收敛性。此外，我们证明了梯度流在梯度评估方面的稳定性，为对数线性策略参数化的自然策略梯度流的性能提供了洞见。为了克服目标函数非凸性和熵正则化引起的不连续性所带来的挑战，我们利用性能差别引理和梯度与镜像下降流之间的对偶关系。

    We study the global convergence of a Fisher-Rao policy gradient flow for infinite-horizon entropy-regularised Markov decision processes with Polish state and action space. The flow is a continuous-time analogue of a policy mirror descent method. We establish the global well-posedness of the gradient flow and demonstrate its exponential convergence to the optimal policy. Moreover, we prove the flow is stable with respect to gradient evaluation, offering insights into the performance of a natural policy gradient flow with log-linear policy parameterisation. To overcome challenges stemming from the lack of the convexity of the objective function and the discontinuity arising from the entropy regulariser, we leverage the performance difference lemma and the duality relationship between the gradient and mirror descent flows.
    
[^4]: MCMC-修正基于得分的扩散模型用于模型组合

    MCMC-Correction of Score-Based Diffusion Models for Model Composition. (arXiv:2307.14012v1 [stat.ML])

    [http://arxiv.org/abs/2307.14012](http://arxiv.org/abs/2307.14012)

    本文提出了一种修正基于得分的扩散模型的方法，使其能够与各种MCMC方法结合，从而实现模型组合和进行更好的采样。

    

    扩散模型可以用得分或能量函数来参数化。能量参数化具有更好的理论特性，主要是它可以通过在提议样本中总能量的变化基于Metropolis-Hastings修正步骤来进行扩展采样过程。然而，它似乎产生了稍微较差的性能，更重要的是，由于基于得分的扩散模型的普遍流行，现有的预训练能量参数化模型的可用性受到限制。这种限制削弱了模型组合的目的，即将预训练模型组合起来从新分布中进行采样。然而，我们的提议建议保留得分参数化，而是通过对得分函数进行线积分来计算基于能量的接受概率。这使我们能够重用现有的扩散模型，并将反向过程与各种马尔可夫链蒙特卡罗（MCMC）方法组合起来。

    Diffusion models can be parameterised in terms of either a score or an energy function. The energy parameterisation has better theoretical properties, mainly that it enables an extended sampling procedure with a Metropolis--Hastings correction step, based on the change in total energy in the proposed samples. However, it seems to yield slightly worse performance, and more importantly, due to the widespread popularity of score-based diffusion, there are limited availability of off-the-shelf pre-trained energy-based ones. This limitation undermines the purpose of model composition, which aims to combine pre-trained models to sample from new distributions. Our proposal, however, suggests retaining the score parameterization and instead computing the energy-based acceptance probability through line integration of the score function. This allows us to re-use existing diffusion models and still combine the reverse process with various Markov-Chain Monte Carlo (MCMC) methods. We evaluate our 
    
[^5]: 脱机强化学习中的离散策略的扩散策略

    Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning. (arXiv:2307.04726v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.04726](http://arxiv.org/abs/2307.04726)

    该论文介绍了一种名为状态重构扩散策略 (SRDP) 的新方法，该方法在最新的扩散策略类中引入了状态重构特征学习，以解决脱机强化学习中的分布偏移和有效表示策略的问题。

    

    脱机强化学习 (RL) 方法利用以前的经验来学习比用于数据收集的行为策略更好的策略。与行为克隆相反，行为克隆假设数据是从专家演示中收集的，而脱机 RL 可以使用非专家数据和多模态行为策略。然而，脱机 RL 算法在处理分布偏移和有效表示策略方面面临挑战，因为训练过程中缺乏在线交互。先前关于脱机 RL 的工作使用条件扩散模型来表示数据集中的多模态行为。然而，这些方法并没有针对缓解脱机分布状态泛化而制定。我们介绍了一种新的方法，名为状态重构扩散策略 (SRDP)，将状态重构特征学习纳入到最新的扩散策略类中，以解决脱机分布通用化问题。状态重构损失促进了更详细的描述。

    Offline Reinforcement Learning (RL) methods leverage previous experiences to learn better policies than the behavior policy used for data collection. In contrast to behavior cloning, which assumes the data is collected from expert demonstrations, offline RL can work with non-expert data and multimodal behavior policies. However, offline RL algorithms face challenges in handling distribution shifts and effectively representing policies due to the lack of online interaction during training. Prior work on offline RL uses conditional diffusion models to represent multimodal behavior in the dataset. Nevertheless, these methods are not tailored toward alleviating the out-of-distribution state generalization. We introduce a novel method, named State Reconstruction for Diffusion Policies (SRDP), incorporating state reconstruction feature learning in the recent class of diffusion policies to address the out-of-distribution generalization problem. State reconstruction loss promotes more descript
    
[^6]: 图神经网络从结构信息中获益的证明：一个特征学习的视角

    Graph Neural Networks Provably Benefit from Structural Information: A Feature Learning Perspective. (arXiv:2306.13926v1 [cs.LG])

    [http://arxiv.org/abs/2306.13926](http://arxiv.org/abs/2306.13926)

    本论文研究了GNN在神经网络特征学习理论中的作用。 发现图卷积网络显著增强了良性过拟合区域，在这个区域内信号学习超越了噪声记忆。

    

    图神经网络(GNNs)在图表示学习方面取得了先驱性进展，在处理图输入时表现出比多层感知器(MLPs)更优越的特征学习和性能。然而，理解GNN的特征学习方面仍处于初始阶段。本研究旨在通过使用梯度下降训练研究图卷积在神经网络特征学习理论中的作用来弥补这一差距。我们提供了对两层图卷积网络(GCNs)中信号学习和噪声记忆的不同刻画，并将它们与两层卷积神经网络(CNNs)进行对比。我们的研究结果表明，与对应的CNNs相比，图卷积网络显著增强了良性过拟合区域，在这个区域内信号学习超越了噪声记忆，并且近似于因子$\sqrt{D}^{q-2}$，其中$D$表示节点的期望度数，$q$表示ReLU激活功能的幂次。

    Graph neural networks (GNNs) have pioneered advancements in graph representation learning, exhibiting superior feature learning and performance over multilayer perceptrons (MLPs) when handling graph inputs. However, understanding the feature learning aspect of GNNs is still in its initial stage. This study aims to bridge this gap by investigating the role of graph convolution within the context of feature learning theory in neural networks using gradient descent training. We provide a distinct characterization of signal learning and noise memorization in two-layer graph convolutional networks (GCNs), contrasting them with two-layer convolutional neural networks (CNNs). Our findings reveal that graph convolution significantly augments the benign overfitting regime over the counterpart CNNs, where signal learning surpasses noise memorization, by approximately factor $\sqrt{D}^{q-2}$, with $D$ denoting a node's expected degree and $q$ being the power of the ReLU activation function where 
    
[^7]: 无限维扩散模型

    Infinite-Dimensional Diffusion Models. (arXiv:2302.10130v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.10130](http://arxiv.org/abs/2302.10130)

    该论文提出了一种在无限维度中直接制定扩散基于的生成模型的方法，相比于传统的先离散化再应用扩散模型的方法，这种方法能够避免参数细化导致算法性能下降，同时提供了维度无关的距离界限，为无限维扩散模型设计提供了准则。

    

    扩散模型对于许多应用领域都产生了深远的影响，包括那些数据本质上是无限维的领域，如图像或时间序列。标准方法是首先离散化数据，然后将扩散模型应用于离散的数据。然而，这种方法在细化离散化参数时通常会导致算法性能下降。在本文中，我们直接在无限维度中制定基于扩散的生成模型，并将其应用于函数的生成建模。我们证明了我们的公式在无限维度环境中是良好定义的，并提供了从样本到目标测度的维度无关的距离界限。利用我们的理论，我们还制定了无限维扩散模型设计的准则。对于图像分布，这些准则与当前用于扩散模型的经典选择一致。对于其他分布...

    Diffusion models have had a profound impact on many application areas, including those where data are intrinsically infinite-dimensional, such as images or time series. The standard approach is first to discretize and then to apply diffusion models to the discretized data. While such approaches are practically appealing, the performance of the resulting algorithms typically deteriorates as discretization parameters are refined. In this paper, we instead directly formulate diffusion-based generative models in infinite dimensions and apply them to the generative modeling of functions. We prove that our formulations are well posed in the infinite-dimensional setting and provide dimension-independent distance bounds from the sample to the target measure. Using our theory, we also develop guidelines for the design of infinite-dimensional diffusion models. For image distributions, these guidelines are in line with the canonical choices currently made for diffusion models. For other distribut
    
[^8]: 基于编码计算和向量承诺的拜占庭抵抗安全聚合方案，用于联邦学习 (arXiv:2302.09913v2 [cs.CR] UPDATED)

    ByzSecAgg: A Byzantine-Resistant Secure Aggregation Scheme for Federated Learning Based on Coded Computing and Vector Commitment. (arXiv:2302.09913v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2302.09913](http://arxiv.org/abs/2302.09913)

    本文提出了一种基于编码计算和向量承诺的拜占庭抵抗安全聚合方案，用于联邦学习。该方案通过RAM秘密共享将本地更新分割成较小子向量，并使用双重RAMP共享技术实现成对距离的安全计算。

    

    本文提出了一种高效的联邦学习保护方案，可以抵御拜占庭攻击和隐私泄露。这种方案通过处理单个更新来管理对抗行为，并在抵御串通节点的同时保护数据隐私。然而，用于对更新向量进行安全秘密共享的通信负载可能非常高。为了解决这个问题，本文提出了一种将本地更新分割成较小子向量并使用RAM秘密共享的方案。但是，这种共享方法无法进行双线性计算，例如需要异常检测算法的成对距离计算。为了克服这个问题，每个用户都会运行另一轮RAMP共享，该共享具有不同的数据嵌入其中。这种受编码计算思想启发的技术实现了成对距离的安全计算。

    In this paper, we propose an efficient secure aggregation scheme for federated learning that is protected against Byzantine attacks and privacy leakages. Processing individual updates to manage adversarial behavior, while preserving privacy of data against colluding nodes, requires some sort of secure secret sharing. However, communication load for secret sharing of long vectors of updates can be very high. To resolve this issue, in the proposed scheme, local updates are partitioned into smaller sub-vectors and shared using ramp secret sharing. However, this sharing method does not admit bi-linear computations, such as pairwise distance calculations, needed by outlier-detection algorithms. To overcome this issue, each user runs another round of ramp sharing, with different embedding of data in the sharing polynomial. This technique, motivated by ideas from coded computing, enables secure computation of pairwise distance. In addition, to maintain the integrity and privacy of the local u
    

