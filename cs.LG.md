# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [IT Intrusion Detection Using Statistical Learning and Testbed Measurements](https://arxiv.org/abs/2402.13081) | 该研究通过统计学习方法和基础设施连续测量数据，以及在内部测试台上进行攻击模拟，实现了IT基础设施中的自动入侵检测。 |
| [^2] | [Variance Reduction and Low Sample Complexity in Stochastic Optimization via Proximal Point Method](https://arxiv.org/abs/2402.08992) | 本文提出了一种通过近端点方法进行随机优化的方法，能够在弱条件下获得低样本复杂度，并实现方差减少的目标。 |
| [^3] | [Compound Returns Reduce Variance in Reinforcement Learning](https://arxiv.org/abs/2402.03903) | 复合回报是一种新的强化学习方法，在降低方差和提高样本效率方面具有重要的贡献和创新。 |
| [^4] | [Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier](https://arxiv.org/abs/2212.04382) | 本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。 |
| [^5] | [HUTFormer: Hierarchical U-Net Transformer for Long-Term Traffic Forecasting.](http://arxiv.org/abs/2307.14596) | HUTFormer是一种用于长期交通预测的分层U-Net Transformer模型，通过利用多尺度表示来解决长期预测中的挑战和问题。 |
| [^6] | [Correlated Noise in Epoch-Based Stochastic Gradient Descent: Implications for Weight Variances.](http://arxiv.org/abs/2306.05300) | 研究挑战了在时间上是不相关的假设，并强调了epoch-based噪声相关性对离散时间带动量的SGD的权重方差的影响。 |
| [^7] | [Neural Exploitation and Exploration of Contextual Bandits.](http://arxiv.org/abs/2305.03784) | 本文提出了一种新型的神经网络策略"EE-Net"，它用于多臂赌博机的利用和探索，在学习奖励函数的同时也适应性地学习潜在收益。 |
| [^8] | [Trajectory-Aware Eligibility Traces for Off-Policy Reinforcement Learning.](http://arxiv.org/abs/2301.11321) | 提出一种轨迹感知的资格追踪多步运算符，可以同时表达每个决策和轨迹感知的方法，并解决了被完全裁剪的资格追踪无法逆转的问题。 |

# 详细

[^1]: 使用统计学习和测试台测量进行IT入侵检测

    IT Intrusion Detection Using Statistical Learning and Testbed Measurements

    [https://arxiv.org/abs/2402.13081](https://arxiv.org/abs/2402.13081)

    该研究通过统计学习方法和基础设施连续测量数据，以及在内部测试台上进行攻击模拟，实现了IT基础设施中的自动入侵检测。

    

    我们研究了IT基础设施中的自动入侵检测，特别是识别攻击开始、攻击类型以及攻击者采取的动作顺序的问题，基于基础设施的连续测量。我们应用统计学习方法，包括隐马尔可夫模型（HMM）、长短期记忆（LSTM）和随机森林分类器（RFC），将观测序列映射到预测攻击动作序列。与大多数相关研究不同，我们拥有丰富的数据来训练模型并评估其预测能力。数据来自我们在内部测试台上生成的跟踪数据，在这里我们对模拟的IT基础设施进行攻击。我们工作的核心是一个机器学习管道，将来自高维观测空间的测量映射到低维空间或少量观测符号的空间。我们研究离线和在线入侵检测

    arXiv:2402.13081v1 Announce Type: new  Abstract: We study automated intrusion detection in an IT infrastructure, specifically the problem of identifying the start of an attack, the type of attack, and the sequence of actions an attacker takes, based on continuous measurements from the infrastructure. We apply statistical learning methods, including Hidden Markov Model (HMM), Long Short-Term Memory (LSTM), and Random Forest Classifier (RFC) to map sequences of observations to sequences of predicted attack actions. In contrast to most related research, we have abundant data to train the models and evaluate their predictive power. The data comes from traces we generate on an in-house testbed where we run attacks against an emulated IT infrastructure. Central to our work is a machine-learning pipeline that maps measurements from a high-dimensional observation space to a space of low dimensionality or to a small set of observation symbols. Investigating intrusions in offline as well as onli
    
[^2]: 通过近端点方法进行随机优化中的方差减少和低样本复杂性

    Variance Reduction and Low Sample Complexity in Stochastic Optimization via Proximal Point Method

    [https://arxiv.org/abs/2402.08992](https://arxiv.org/abs/2402.08992)

    本文提出了一种通过近端点方法进行随机优化的方法，能够在弱条件下获得低样本复杂度，并实现方差减少的目标。

    

    本文提出了一种随机近端点法来解决随机凸复合优化问题。随机优化中的高概率结果通常依赖于对随机梯度噪声的限制性假设，例如子高斯分布。本文只假设了随机梯度的有界方差等弱条件，建立了一种低样本复杂度以获得关于所提方法收敛的高概率保证。此外，本工作的一个显著方面是发展了一个用于解决近端子问题的子程序，它同时也是一种用于减少方差的新技术。

    arXiv:2402.08992v1 Announce Type: cross Abstract: This paper proposes a stochastic proximal point method to solve a stochastic convex composite optimization problem. High probability results in stochastic optimization typically hinge on restrictive assumptions on the stochastic gradient noise, for example, sub-Gaussian distributions. Assuming only weak conditions such as bounded variance of the stochastic gradient, this paper establishes a low sample complexity to obtain a high probability guarantee on the convergence of the proposed method. Additionally, a notable aspect of this work is the development of a subroutine to solve the proximal subproblem, which also serves as a novel technique for variance reduction.
    
[^3]: 复合回报降低强化学习中的方差

    Compound Returns Reduce Variance in Reinforcement Learning

    [https://arxiv.org/abs/2402.03903](https://arxiv.org/abs/2402.03903)

    复合回报是一种新的强化学习方法，在降低方差和提高样本效率方面具有重要的贡献和创新。

    

    多步回报，例如$n$步回报和$\lambda$回报，通常用于提高强化学习方法的样本效率。多步回报的方差成为其长度的限制因素，过度远望未来会增加方差并逆转多步学习的好处。在我们的工作中，我们展示了复合回报（$n$步回报的加权平均）降低方差的能力。我们首次证明了任何与给定$n$步回报具有相同收缩模数的复合回报的方差严格较低。我们还证明了这种降低方差的特性改善了线性函数逼近下时序差分学习的有限样本复杂性。由于一般复合回报的实施可能代价高昂，我们引入了两个自助回报，它们在保持高效性的同时降低了方差，即使在使用小批量经验回放时也是如此。我们进行了实验，显示……

    Multistep returns, such as $n$-step returns and $\lambda$-returns, are commonly used to improve the sample efficiency of reinforcement learning (RL) methods. The variance of the multistep returns becomes the limiting factor in their length; looking too far into the future increases variance and reverses the benefits of multistep learning. In our work, we demonstrate the ability of compound returns -- weighted averages of $n$-step returns -- to reduce variance. We prove for the first time that any compound return with the same contraction modulus as a given $n$-step return has strictly lower variance. We additionally prove that this variance-reduction property improves the finite-sample complexity of temporal-difference learning under linear function approximation. Because general compound returns can be expensive to implement, we introduce two-bootstrap returns which reduce variance while remaining efficient, even when using minibatched experience replay. We conduct experiments showing
    
[^4]: 分类器边界的结构：朴素贝叶斯分类器的案例研究

    Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier

    [https://arxiv.org/abs/2212.04382](https://arxiv.org/abs/2212.04382)

    本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。

    

    无论基于模型、训练数据还是二者组合，分类器将（可能复杂的）输入数据归入相对较少的输出类别之一。本文研究在输入空间为图的情况下，边界的结构——那些被分类为不同类别的邻近点——的特性。我们的科学背景是基于模型的朴素贝叶斯分类器，用于处理由下一代测序仪生成的DNA读数。我们展示了边界既是巨大的，又具有复杂的结构。我们创建了一种新的不确定性度量，称为邻居相似度，它将一个点的结果与其邻居的结果分布进行比较。这个度量不仅追踪了贝叶斯分类器的两个固有不确定性度量，还可以在没有固有不确定性度量的分类器上实现，但需要计算成本。

    Whether based on models, training data or a combination, classifiers place (possibly complex) input data into one of a relatively small number of output categories. In this paper, we study the structure of the boundary--those points for which a neighbor is classified differently--in the context of an input space that is a graph, so that there is a concept of neighboring inputs, The scientific setting is a model-based naive Bayes classifier for DNA reads produced by Next Generation Sequencers. We show that the boundary is both large and complicated in structure. We create a new measure of uncertainty, called Neighbor Similarity, that compares the result for a point to the distribution of results for its neighbors. This measure not only tracks two inherent uncertainty measures for the Bayes classifier, but also can be implemented, at a computational cost, for classifiers without inherent measures of uncertainty.
    
[^5]: HUTFormer：用于长期交通预测的分层U-Net Transformer

    HUTFormer: Hierarchical U-Net Transformer for Long-Term Traffic Forecasting. (arXiv:2307.14596v1 [cs.LG])

    [http://arxiv.org/abs/2307.14596](http://arxiv.org/abs/2307.14596)

    HUTFormer是一种用于长期交通预测的分层U-Net Transformer模型，通过利用多尺度表示来解决长期预测中的挑战和问题。

    

    交通预测旨在基于历史观测数据预测交通状况，是智能交通领域中的重要研究课题。最近的空间-时间图神经网络（STGNNs）通过将顺序模型与图卷积网络相结合取得了显著的进展。然而，由于复杂性问题，STGNNs仅聚焦于短期交通预测，如1小时预测，而忽视了更实际的长期预测。本文首次尝试探索长期交通预测，例如1天的预测。为此，我们首先揭示了在利用多尺度表示方面的独特挑战。然后，我们提出了一种新颖的分层U-Net TransFormer（HUTFormer）来解决长期交通预测的问题。HUTFormer由分层编码器和解码器组成，共同生成和利用交通的多尺度表示。

    Traffic forecasting, which aims to predict traffic conditions based on historical observations, has been an enduring research topic and is widely recognized as an essential component of intelligent transportation. Recent proposals on Spatial-Temporal Graph Neural Networks (STGNNs) have made significant progress by combining sequential models with graph convolution networks. However, due to high complexity issues, STGNNs only focus on short-term traffic forecasting, e.g., 1-hour forecasting, while ignoring more practical long-term forecasting. In this paper, we make the first attempt to explore long-term traffic forecasting, e.g., 1-day forecasting. To this end, we first reveal its unique challenges in exploiting multi-scale representations. Then, we propose a novel Hierarchical U-net TransFormer (HUTFormer) to address the issues of long-term traffic forecasting. HUTFormer consists of a hierarchical encoder and decoder to jointly generate and utilize multi-scale representations of traff
    
[^6]: 基于Epoch的随机梯度下降中相关噪声：权重方差的影响

    Correlated Noise in Epoch-Based Stochastic Gradient Descent: Implications for Weight Variances. (arXiv:2306.05300v1 [cs.LG])

    [http://arxiv.org/abs/2306.05300](http://arxiv.org/abs/2306.05300)

    研究挑战了在时间上是不相关的假设，并强调了epoch-based噪声相关性对离散时间带动量的SGD的权重方差的影响。

    

    随机梯度下降（SGD）已成为神经网络优化的基石，但认为SGD引入的噪声在时间上是不相关的，尽管epoch-based训练是无处不在的。在这项工作中，我们对此进行了挑战，并调查了epoch-based噪声相关性对离散时间带动量的SGD的稳态分布的影响，限于二次损失。我们的主要贡献有两个：首先，我们计算训练epoch时噪声的精确自相关性，假设该噪声独立于权重向量中的小波动;其次，我们探索epoch-based学习方案引入的相关性对SGD动态的影响。我们发现，在曲率大于一个超参数相关值的方向上，还原了不相关噪声的结果。然而，在相对平坦的方向上，权重方差显着减小。我们使用简单的二维图例对这些结果进行了直观解释。总的来说，我们的工作提供了关于epoch-based SGD中相关噪声影响的见解，可以指导设计更有效的优化算法。

    Stochastic gradient descent (SGD) has become a cornerstone of neural network optimization, yet the noise introduced by SGD is often assumed to be uncorrelated over time, despite the ubiquity of epoch-based training. In this work, we challenge this assumption and investigate the effects of epoch-based noise correlations on the stationary distribution of discrete-time SGD with momentum, limited to a quadratic loss. Our main contributions are twofold: first, we calculate the exact autocorrelation of the noise for training in epochs under the assumption that the noise is independent of small fluctuations in the weight vector; second, we explore the influence of correlations introduced by the epoch-based learning scheme on SGD dynamics. We find that for directions with a curvature greater than a hyperparameter-dependent crossover value, the results for uncorrelated noise are recovered. However, for relatively flat directions, the weight variance is significantly reduced. We provide an intui
    
[^7]: 多臂赌博机的上下文利用与探索的神经网络研究

    Neural Exploitation and Exploration of Contextual Bandits. (arXiv:2305.03784v1 [cs.LG])

    [http://arxiv.org/abs/2305.03784](http://arxiv.org/abs/2305.03784)

    本文提出了一种新型的神经网络策略"EE-Net"，它用于多臂赌博机的利用和探索，在学习奖励函数的同时也适应性地学习潜在收益。

    

    本文研究利用神经网络进行上下文多臂赌博机的利用和探索。我们提出了一个名为"EE-Net"的新型神经网络利用和探索策略，它使用一个神经网络（利用网络）来学习奖励函数，另一个神经网络（探索网络）来适应性地学习相对于当前估计奖励的潜在收益。

    In this paper, we study utilizing neural networks for the exploitation and exploration of contextual multi-armed bandits. Contextual multi-armed bandits have been studied for decades with various applications. To solve the exploitation-exploration trade-off in bandits, there are three main techniques: epsilon-greedy, Thompson Sampling (TS), and Upper Confidence Bound (UCB). In recent literature, a series of neural bandit algorithms have been proposed to adapt to the non-linear reward function, combined with TS or UCB strategies for exploration. In this paper, instead of calculating a large-deviation based statistical bound for exploration like previous methods, we propose, ``EE-Net,'' a novel neural-based exploitation and exploration strategy. In addition to using a neural network (Exploitation network) to learn the reward function, EE-Net uses another neural network (Exploration network) to adaptively learn the potential gains compared to the currently estimated reward for exploration
    
[^8]: 轨迹感知的资格追踪在离线强化学习中的应用

    Trajectory-Aware Eligibility Traces for Off-Policy Reinforcement Learning. (arXiv:2301.11321v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.11321](http://arxiv.org/abs/2301.11321)

    提出一种轨迹感知的资格追踪多步运算符，可以同时表达每个决策和轨迹感知的方法，并解决了被完全裁剪的资格追踪无法逆转的问题。

    

    离线多步返回的非政策学习对于节约样本的强化学习至关重要，但抵消偏差的同时不加剧方差是具有挑战性的。一般来说，非政策偏差是通过资格追踪的方法来进行修正的，资格追踪通过通吃因子(Impotance Sampling)比例对过去的时间差分误差进行重新加权以纠正。许多离线算法都依赖这种机制，不同的是针对IS的统计估计方法所采用的“裁剪IS比例”协议的不同。不幸的是，一旦资格追踪被完全裁剪，其影响就无法逆转。这已经导致了将多个过去经历同时考虑在内的信用分配策略的出现。这些轨迹感知的方法尚未得到广泛的分析，它们的理论依据仍然不确定。本文提出了一种多步运算符，可以同时表达每个决策和轨迹感知的方法，并证明它们的收敛条件。

    Off-policy learning from multistep returns is crucial for sample-efficient reinforcement learning, but counteracting off-policy bias without exacerbating variance is challenging. Classically, off-policy bias is corrected in a per-decision manner: past temporal-difference errors are re-weighted by the instantaneous Importance Sampling (IS) ratio after each action via eligibility traces. Many off-policy algorithms rely on this mechanism, along with differing protocols for cutting the IS ratios to combat the variance of the IS estimator. Unfortunately, once a trace has been fully cut, the effect cannot be reversed. This has led to the development of credit-assignment strategies that account for multiple past experiences at a time. These trajectory-aware methods have not been extensively analyzed, and their theoretical justification remains uncertain. In this paper, we propose a multistep operator that can express both per-decision and trajectory-aware methods. We prove convergence conditi
    

