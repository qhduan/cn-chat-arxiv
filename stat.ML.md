# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [When Your AI Deceives You: Challenges with Partial Observability of Human Evaluators in Reward Learning](https://arxiv.org/abs/2402.17747) | RLHF在考虑部分观察性时可能导致策略欺骗性地夸大性能或过度辩护行为，我们提出了数学条件来解决这些问题，并警告不要盲目应用RLHF在部分可观测情况下。 |
| [^2] | [Shaving Weights with Occam's Razor: Bayesian Sparsification for Neural Networks Using the Marginal Likelihood](https://arxiv.org/abs/2402.15978) | 提出了一种基于边缘似然的贝叶斯稀疏化神经网络的方法，通过有效利用贝叶斯边缘似然和稀疏诱导先验，使神经网络更易稀疏化，并采用自动奥卡姆剃刀选择最适合的模型，以实现高效的权重削减。 |
| [^3] | [Robustly overfitting latents for flexible neural image compression](https://arxiv.org/abs/2401.17789) | 这项研究提出了一种鲁棒的过拟合潜变量方法来改进神经图像压缩模型，通过使用SGA+，可以显著提高性能并减少对超参数选择的敏感性。 |
| [^4] | [Fast Empirical Scenarios.](http://arxiv.org/abs/2307.03927) | 该论文提出了两种快速的经验场景提取算法，一种识别之前未观察到的场景并提供场景的协方差矩阵表示，另一种从已实现的世界状态中选择重要的数据点，并与高阶样本矩一致，这些算法计算效率高且适用于一致的基于场景的建模和高维数值积分。 |
| [^5] | [On the Global Convergence of Risk-Averse Policy Gradient Methods with Expected Conditional Risk Measures.](http://arxiv.org/abs/2301.10932) | 本论文研究了具有期望条件风险度量的风险厌恶策略梯度方法，提出了策略梯度更新，证明了其在约束和无约束情况下的全局收敛性和迭代复杂度，并测试了REINFORCE和actor-critic算法的风险厌恶变体来展示方法的实用价值和风险控制的重要性。 |
| [^6] | [Provably Learning Diverse Features in Multi-View Data with Midpoint Mixup.](http://arxiv.org/abs/2210.13512) | 本文提出了一种基于中点 Mixup 的多视角数据学习方法，相比于传统经验风险最小化方法能够更好地学习每个类别的所有特征，具有更好的泛化和鲁棒性。 |
| [^7] | [A Nonparametric Approach with Marginals for Modeling Consumer Choice.](http://arxiv.org/abs/2208.06115) | 本文提出了一种基于边缘分布的简单而有效的非参数消费者选择建模方法，在任何选择集合中会把选择概率的集合一致地描述出来。 |

# 详细

[^1]: 当你的AI欺骗你：在奖励学习中人类评估者部分可观测性的挑战

    When Your AI Deceives You: Challenges with Partial Observability of Human Evaluators in Reward Learning

    [https://arxiv.org/abs/2402.17747](https://arxiv.org/abs/2402.17747)

    RLHF在考虑部分观察性时可能导致策略欺骗性地夸大性能或过度辩护行为，我们提出了数学条件来解决这些问题，并警告不要盲目应用RLHF在部分可观测情况下。

    

    强化学习从人类反馈（RLHF）的过去分析假设人类完全观察到环境。当人类反馈仅基于部分观察时会发生什么？我们对两种失败情况进行了正式定义：欺骗和过度辩护。通过将人类建模为对轨迹信念的Boltzmann-理性，我们证明了RLHF保证会导致策略欺骗性地夸大其性能、为了留下印象而过度辩护或者两者兼而有之的条件。为了帮助解决这些问题，我们数学地刻画了环境部分可观测性如何转化为（缺乏）学到的回报函数中的模糊性。在某些情况下，考虑环境部分可观测性使得在理论上可能恢复回报函数和最优策略，而在其他情况下，存在不可减少的模糊性。我们警告不要盲目应用RLHF在部分可观测情况下。

    arXiv:2402.17747v1 Announce Type: cross  Abstract: Past analyses of reinforcement learning from human feedback (RLHF) assume that the human fully observes the environment. What happens when human feedback is based only on partial observations? We formally define two failure cases: deception and overjustification. Modeling the human as Boltzmann-rational w.r.t. a belief over trajectories, we prove conditions under which RLHF is guaranteed to result in policies that deceptively inflate their performance, overjustify their behavior to make an impression, or both. To help address these issues, we mathematically characterize how partial observability of the environment translates into (lack of) ambiguity in the learned return function. In some cases, accounting for partial observability makes it theoretically possible to recover the return function and thus the optimal policy, while in other cases, there is irreducible ambiguity. We caution against blindly applying RLHF in partially observa
    
[^2]: 使用奥卡姆剃刀削减权重：使用边缘似然的贝叶斯稀疏化神经网络

    Shaving Weights with Occam's Razor: Bayesian Sparsification for Neural Networks Using the Marginal Likelihood

    [https://arxiv.org/abs/2402.15978](https://arxiv.org/abs/2402.15978)

    提出了一种基于边缘似然的贝叶斯稀疏化神经网络的方法，通过有效利用贝叶斯边缘似然和稀疏诱导先验，使神经网络更易稀疏化，并采用自动奥卡姆剃刀选择最适合的模型，以实现高效的权重削减。

    

    神经网络稀疏化是一个有前途的途径，可以节省计算时间和内存成本，特别是在许多成功的人工智能模型变得过大以至无法直接部署在消费类硬件的时代。虽然很多工作都集中在不同的权重剪枝准则上，但网络的总体稀疏性，即可以在不损失质量的情况下剪枝的能力，经常被忽视。我们提出了通过边缘似然量（Marginal likelihood）的稀疏性（SpaM），一个稀疏化框架，重点强调使用贝叶斯边缘似然与稀疏诱导先验相结合，使神经网络更易稀疏化的有效性。我们的方法实现了一个自动的奥卡姆剃刀，选择最想要削减的模型，以依然能够很好地解释数据，无论是对于结构化还是非结构化的稀疏化。此外，我们展示了拉普拉斯近似中使用的预计算后验黑塞近似的效果。

    arXiv:2402.15978v1 Announce Type: new  Abstract: Neural network sparsification is a promising avenue to save computational time and memory costs, especially in an age where many successful AI models are becoming too large to na\"ively deploy on consumer hardware. While much work has focused on different weight pruning criteria, the overall sparsifiability of the network, i.e., its capacity to be pruned without quality loss, has often been overlooked. We present Sparsifiability via the Marginal likelihood (SpaM), a pruning framework that highlights the effectiveness of using the Bayesian marginal likelihood in conjunction with sparsity-inducing priors for making neural networks more sparsifiable. Our approach implements an automatic Occam's razor that selects the most sparsifiable model that still explains the data well, both for structured and unstructured sparsification. In addition, we demonstrate that the pre-computed posterior Hessian approximation used in the Laplace approximation
    
[^3]: 弹性神经图像压缩中的鲁棒过拟合潜变量

    Robustly overfitting latents for flexible neural image compression

    [https://arxiv.org/abs/2401.17789](https://arxiv.org/abs/2401.17789)

    这项研究提出了一种鲁棒的过拟合潜变量方法来改进神经图像压缩模型，通过使用SGA+，可以显著提高性能并减少对超参数选择的敏感性。

    

    神经图像压缩取得了很大的进展。最先进的模型基于变分自编码器，胜过了传统模型。神经压缩模型学会将图像编码为量化的潜变量表示，然后将其高效地发送给解码器，解码器再将量化的潜变量解码为重建图像。虽然这些模型在实践中取得了成功，但由于优化不完美以及编码器和解码器容量的限制，它们导致了次优结果。最近的研究表明，如何利用随机Gumbel退火（SGA）来改进预训练的神经图像压缩模型的潜变量。我们通过引入SGA+扩展了这个想法，SGA+包含了三种不同的方法，这些方法都建立在SGA的基础上。此外，我们对我们提出的方法进行了详细分析，展示了它们如何改进性能，并且证明它们对超参数选择不敏感。此外，我们还展示了如何将每个方法扩展到三个而不是两个。

    Neural image compression has made a great deal of progress. State-of-the-art models are based on variational autoencoders and are outperforming classical models. Neural compression models learn to encode an image into a quantized latent representation that can be efficiently sent to the decoder, which decodes the quantized latent into a reconstructed image. While these models have proven successful in practice, they lead to sub-optimal results due to imperfect optimization and limitations in the encoder and decoder capacity. Recent work shows how to use stochastic Gumbel annealing (SGA) to refine the latents of pre-trained neural image compression models. We extend this idea by introducing SGA+, which contains three different methods that build upon SGA. Further, we give a detailed analysis of our proposed methods, show how they improve performance, and show that they are less sensitive to hyperparameter choices. Besides, we show how each method can be extended to three- instead of two
    
[^4]: 快速经验场景

    Fast Empirical Scenarios. (arXiv:2307.03927v1 [stat.ML])

    [http://arxiv.org/abs/2307.03927](http://arxiv.org/abs/2307.03927)

    该论文提出了两种快速的经验场景提取算法，一种识别之前未观察到的场景并提供场景的协方差矩阵表示，另一种从已实现的世界状态中选择重要的数据点，并与高阶样本矩一致，这些算法计算效率高且适用于一致的基于场景的建模和高维数值积分。

    

    我们希望从大型和高维面板数据中提取一小部分与样本矩一致的代表性场景。在两种新算法中，第一种算法识别之前未观察到的场景，并提供了一种基于场景的协方差矩阵表示。第二种算法从已实现的世界状态中选择重要的数据点，并与高阶样本矩信息一致。这两种算法计算效率高，并可用于一致的基于场景的建模和高维数值积分。广泛的数值基准测试研究和在投资组合优化中的应用支持所提出的算法。

    We seek to extract a small number of representative scenarios from large and high-dimensional panel data that are consistent with sample moments. Among two novel algorithms, the first identifies scenarios that have not been observed before, and comes with a scenario-based representation of covariance matrices. The second proposal picks important data points from states of the world that have already realized, and are consistent with higher-order sample moment information. Both algorithms are efficient to compute, and lend themselves to consistent scenario-based modeling and high-dimensional numerical integration. Extensive numerical benchmarking studies and an application in portfolio optimization favor the proposed algorithms.
    
[^5]: 关于具有期望条件风险度量的风险厌恶策略梯度方法的全局收敛性

    On the Global Convergence of Risk-Averse Policy Gradient Methods with Expected Conditional Risk Measures. (arXiv:2301.10932v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.10932](http://arxiv.org/abs/2301.10932)

    本论文研究了具有期望条件风险度量的风险厌恶策略梯度方法，提出了策略梯度更新，证明了其在约束和无约束情况下的全局收敛性和迭代复杂度，并测试了REINFORCE和actor-critic算法的风险厌恶变体来展示方法的实用价值和风险控制的重要性。

    

    风险敏感的强化学习已经成为控制不确定结果和确保各种顺序决策问题的可靠性能的流行工具。虽然针对风险敏感的强化学习已经开发出了策略梯度方法，但这些方法是否具有与风险中性情况下相同的全局收敛保证还不清楚。本文考虑了一类动态时间一致风险度量，称为期望条件风险度量（ECRM），并为基于ECRM的目标函数推导出策略梯度更新。在约束直接参数化和无约束softmax参数化下，我们提供了相应的风险厌恶策略梯度算法的全局收敛性和迭代复杂度。我们进一步测试了REINFORCE和actor-critic算法的风险厌恶变体，以展示我们的方法的有效性和风险控制的重要性。

    Risk-sensitive reinforcement learning (RL) has become a popular tool to control the risk of uncertain outcomes and ensure reliable performance in various sequential decision-making problems. While policy gradient methods have been developed for risk-sensitive RL, it remains unclear if these methods enjoy the same global convergence guarantees as in the risk-neutral case. In this paper, we consider a class of dynamic time-consistent risk measures, called Expected Conditional Risk Measures (ECRMs), and derive policy gradient updates for ECRM-based objective functions. Under both constrained direct parameterization and unconstrained softmax parameterization, we provide global convergence and iteration complexities of the corresponding risk-averse policy gradient algorithms. We further test risk-averse variants of REINFORCE and actor-critic algorithms to demonstrate the efficacy of our method and the importance of risk control.
    
[^6]: 用中点 Mixup 在多视角数据中可证明学习多元特征

    Provably Learning Diverse Features in Multi-View Data with Midpoint Mixup. (arXiv:2210.13512v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.13512](http://arxiv.org/abs/2210.13512)

    本文提出了一种基于中点 Mixup 的多视角数据学习方法，相比于传统经验风险最小化方法能够更好地学习每个类别的所有特征，具有更好的泛化和鲁棒性。

    

    Mixup 是一种数据增强技术，依赖于使用数据点和标签的随机凸组合进行训练。近年来，Mixup 已成为训练最先进的图像分类模型的标准基元，因为它在泛化和鲁棒性方面比经验风险最小化有明显的优势。在这项工作中，我们试图从特征学习的角度解释一些这种成功。我们关注的分类问题是，每个类别可能具有多个相关特征（或视图），可用于正确预测类别。我们的主要理论结果表明，在具有每类两个特征的一类非平凡数据分布中，使用经验风险最小化训练 2 层卷积网络可能会导致几乎所有类别只学习一个特征，而使用 Mixup 的特定实例进行训练可以成功地学习每个类别的两个特征。

    Mixup is a data augmentation technique that relies on training using random convex combinations of data points and their labels. In recent years, Mixup has become a standard primitive used in the training of state-of-the-art image classification models due to its demonstrated benefits over empirical risk minimization with regards to generalization and robustness. In this work, we try to explain some of this success from a feature learning perspective. We focus our attention on classification problems in which each class may have multiple associated features (or views) that can be used to predict the class correctly. Our main theoretical results demonstrate that, for a non-trivial class of data distributions with two features per class, training a 2-layer convolutional network using empirical risk minimization can lead to learning only one feature for almost all classes while training with a specific instantiation of Mixup succeeds in learning both features for every class. We also show
    
[^7]: 基于边缘分布的非参数消费者选择建模方法

    A Nonparametric Approach with Marginals for Modeling Consumer Choice. (arXiv:2208.06115v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2208.06115](http://arxiv.org/abs/2208.06115)

    本文提出了一种基于边缘分布的简单而有效的非参数消费者选择建模方法，在任何选择集合中会把选择概率的集合一致地描述出来。

    

    鉴于消费者在不同选择集合中作出选择的数据，开发描述和预测消费者选择行为的简洁模型是一个主要挑战。其中一种选择模型是边缘分布模型，该模型仅需要规定随机效用的边缘分布即可解释选项数据。在本文中，我们开发了一种精确的选择概率集合的特征化方法，该集合可以在任何集合中一致地通过边缘分布模型来描述。允许根据其效用的边缘分布将选择集合进行分组，我们展示了(a)验证这个模型与选择概率数据的一致性在多项式时间内是可能的，(b)最接近拟合的方法可以简化为解决混合整数凸规划问题。我们的结果表明，与多项式Logit模型和m相比，边缘分布模型提供了更好的表现能力。

    Given data on choices made by consumers for different assortments, a key challenge is to develop parsimonious models that describe and predict consumer choice behavior. One such choice model is the marginal distribution model which requires only the specification of the marginal distributions of the random utilities of the alternatives to explain choice data. In this paper, we develop an exact characterisation of the set of choice probabilities which are representable by the marginal distribution model consistently across any collection of assortments. Allowing for the possibility of alternatives to be grouped based on the marginal distribution of their utilities, we show (a) verifying consistency of choice probability data with this model is possible in polynomial time and (b) finding the closest fit reduces to solving a mixed integer convex program. Our results show that the marginal distribution model provides much better representational power as compared to multinomial logit and m
    

