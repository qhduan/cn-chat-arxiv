# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Epsilon-Greedy Thompson Sampling to Bayesian Optimization](https://arxiv.org/abs/2403.00540) | 将$\varepsilon$-greedy策略引入Thompson采样以改进贝叶斯优化中的开发功能，并实证表明其有效性。 |
| [^2] | [Gaussian Ensemble Belief Propagation for Efficient Inference in High-Dimensional Systems](https://arxiv.org/abs/2402.08193) | 高斯模型集成置信传播算法（GEnBP）是一种用于高维系统中高效推断的方法，通过集成卡尔曼滤波器和高斯置信传播等技术相结合，能有效处理高维状态、参数和复杂的依赖结构。 |
| [^3] | [Detecting algorithmic bias in medical AI-models](https://arxiv.org/abs/2312.02959) | 本文提出了一种创新的框架，用于检测医疗AI决策支持系统中的算法偏倚，通过采用CART算法有效地识别医疗AI模型中的潜在偏倚，并在合成数据实验和真实临床环境中验证了其有效性。 |
| [^4] | [Robust Multi-Modal Density Estimation.](http://arxiv.org/abs/2401.10566) | 本文提出了一种名为ROME的鲁棒多模态密度估计方法，该方法利用聚类将多模态样本集分割成多个单模态样本集，并通过简单的KDE估计来估计整体分布。这种方法解决了多模态、非正态和高相关分布估计的挑战。 |
| [^5] | [On Linear Separation Capacity of Self-Supervised Representation Learning.](http://arxiv.org/abs/2310.19041) | 本研究通过探究在多流形模型下，学习的表示何时可以线性分离流形，揭示了自监督学习在数据增强方面的额外好处，从而改善了线性分离能力的信息论最优速率。 |
| [^6] | [Droplets of Good Representations: Grokking as a First Order Phase Transition in Two Layer Networks.](http://arxiv.org/abs/2310.03789) | 本研究将自适应核方法应用于两个师生模型，预测了特征学习和 Grokking 的性质，并展示了 Grokking 与相变理论之间的映射关系。 |
| [^7] | [Beyond Stationarity: Convergence Analysis of Stochastic Softmax Policy Gradient Methods.](http://arxiv.org/abs/2310.02671) | 本论文研究了随机Softmax策略梯度方法的收敛性分析，提出了一种动态策略梯度的组合方法，并通过对参数进行反向训练来更好地利用相关性结构，实现向全局最优值的收敛。 |
| [^8] | [Towards Causal Foundation Model: on Duality between Causal Inference and Attention.](http://arxiv.org/abs/2310.00809) | 该论文提出了一种名为Causal Inference with Attention (CInA)的新方法，利用因果推断和注意力的对偶关系，在复杂任务中实现了零样本的因果推断。 |
| [^9] | [Identifiable causal inference with noisy treatment and no side information.](http://arxiv.org/abs/2306.10614) | 本论文提出了一种在没有侧面信息和具有复杂非线性依赖性的情况下，纠正因治疗变量不准确测量引起的因果效应估计偏差的模型，并证明了该模型的因果效应估计是可识别的。该方法使用了深度潜在变量模型和分摊权重变分客观函数进行训练。 |
| [^10] | [Latent Optimal Paths by Gumbel Propagation for Variational Bayesian Dynamic Programming.](http://arxiv.org/abs/2306.02568) | 该论文使用动态规划和Gumbel传播在VAE的潜在空间中获得结构化稀疏最优路径，从而使得模型可以依赖于未观察到的结构特征信息，并成功实现了文本转语音和歌声合成。 |
| [^11] | [Adversarially Robust PAC Learnability of Real-Valued Functions.](http://arxiv.org/abs/2206.12977) | 该论文研究了在实值函数中对抗鲁棒PAC学习性，发现有限胖折射维的类既可以在实现和不可知设置中被学习，凸函数类可以正确学习，而一些非凸函数类需要不正当的学习算法。 |
| [^12] | [Exceedance Probability Forecasting via Regression for Significant Wave Height Prediction.](http://arxiv.org/abs/2206.09821) | 本论文提出了一种基于回归的超标概率预测方法，用于预测显著波高，通过利用预测来估计超标概率，取得了更好的效果。 |
| [^13] | [Seeded graph matching for the correlated Wigner model via the projected power method.](http://arxiv.org/abs/2204.04099) | 本文研究了在相关维格纳模型下的有种子图匹配问题，通过分析表明，使用投影功率方法（PPM）作为图匹配算法可以在给定接近真实匹配的种子的情况下高概率地改进种子并恢复真实匹配。 |
| [^14] | [Graph Neural Network Sensitivity Under Probabilistic Error Model.](http://arxiv.org/abs/2203.07831) | 本文研究了概率误差模型对图卷积网络（GCN）性能的影响，并证明了误差模型下邻接矩阵的受限性。通过实验验证了这种误差界限，并研究了GCN在这种概率误差模型下的准确性敏感性。 |
| [^15] | [MARS via LASSO.](http://arxiv.org/abs/2111.11694) | 本文提出了一种自然lasso变体的MARS方法，通过减少对维度的依赖来获得收敛率，并与使用平滑性约束的非参数估计技术联系在一起。 |

# 详细

[^1]: Epsilon-Greedy Thompson Sampling用于贝叶斯优化

    Epsilon-Greedy Thompson Sampling to Bayesian Optimization

    [https://arxiv.org/abs/2403.00540](https://arxiv.org/abs/2403.00540)

    将$\varepsilon$-greedy策略引入Thompson采样以改进贝叶斯优化中的开发功能，并实证表明其有效性。

    

    Thompson采样（TS）被认为是解决贝叶斯优化中开发-探索困境的解决方案。 虽然它通过随机生成和最大化高斯过程（GP）后验的样本路径来优先进行探索，但TS在每次执行探索后通过收集关于真实目标函数的信息来弱化其开发功能。 本研究将在TS中引入$\varepsilon$-greedy策略，这是一种在强化学习中被广泛应用的选择策略，以改进其开发功能。 我们首先描述了TS应用于BO的两个极端，即通用TS和样本平均TS。前者和后者分别提倡探索和开发。 然后我们使用$\varepsilon$-greedy策略在两个极端之间随机切换。 $\varepsilon \in (0,1)$的小值优先考虑开发，反之亦然。 我们实证表明$\varepsilon$-greedy T

    arXiv:2403.00540v1 Announce Type: new  Abstract: Thompson sampling (TS) serves as a solution for addressing the exploitation-exploration dilemma in Bayesian optimization (BO). While it prioritizes exploration by randomly generating and maximizing sample paths of Gaussian process (GP) posteriors, TS weakly manages its exploitation by gathering information about the true objective function after each exploration is performed. In this study, we incorporate the epsilon-greedy ($\varepsilon$-greedy) policy, a well-established selection strategy in reinforcement learning, into TS to improve its exploitation. We first delineate two extremes of TS applied for BO, namely the generic TS and a sample-average TS. The former and latter promote exploration and exploitation, respectively. We then use $\varepsilon$-greedy policy to randomly switch between the two extremes. A small value of $\varepsilon \in (0,1)$ prioritizes exploitation, and vice versa. We empirically show that $\varepsilon$-greedy T
    
[^2]: 高斯模型集成置信传播用于高维系统中的高效推断

    Gaussian Ensemble Belief Propagation for Efficient Inference in High-Dimensional Systems

    [https://arxiv.org/abs/2402.08193](https://arxiv.org/abs/2402.08193)

    高斯模型集成置信传播算法（GEnBP）是一种用于高维系统中高效推断的方法，通过集成卡尔曼滤波器和高斯置信传播等技术相结合，能有效处理高维状态、参数和复杂的依赖结构。

    

    高维模型中的高效推断仍然是机器学习中的一个核心挑战。本文介绍了一种名为高斯模型集成置信传播（GEnBP）算法的方法，该方法是集成卡尔曼滤波器和高斯置信传播（GaBP）方法的结合。GEnBP通过在图模型结构中传递低秩本地信息来更新集成模型。这种组合继承了每种方法的有利特性。集成技术使得GEnBP能够处理高维状态、参数和复杂的、嘈杂的黑箱生成过程。在图模型结构中使用本地信息确保了该方法适用于分布式计算，并能高效地处理复杂的依赖结构。当集成大小远小于推断维度时，GEnBP特别有优势。这种情况在空时建模、图像处理和物理模型反演等领域经常出现。GEnBP可以应用于一般性问题。

    Efficient inference in high-dimensional models remains a central challenge in machine learning. This paper introduces the Gaussian Ensemble Belief Propagation (GEnBP) algorithm, a fusion of the Ensemble Kalman filter and Gaussian belief propagation (GaBP) methods. GEnBP updates ensembles by passing low-rank local messages in a graphical model structure. This combination inherits favourable qualities from each method. Ensemble techniques allow GEnBP to handle high-dimensional states, parameters and intricate, noisy, black-box generation processes. The use of local messages in a graphical model structure ensures that the approach is suited to distributed computing and can efficiently handle complex dependence structures. GEnBP is particularly advantageous when the ensemble size is considerably smaller than the inference dimension. This scenario often arises in fields such as spatiotemporal modelling, image processing and physical model inversion. GEnBP can be applied to general problem s
    
[^3]: 在医疗AI模型中检测算法偏倚

    Detecting algorithmic bias in medical AI-models

    [https://arxiv.org/abs/2312.02959](https://arxiv.org/abs/2312.02959)

    本文提出了一种创新的框架，用于检测医疗AI决策支持系统中的算法偏倚，通过采用CART算法有效地识别医疗AI模型中的潜在偏倚，并在合成数据实验和真实临床环境中验证了其有效性。

    

    随着机器学习和人工智能医疗决策支持系统日益普及，确保这些系统以公平、公正的方式提供患者结果变得同样重要。本文提出了一种创新的框架，用于检测医疗AI决策支持系统中的算法偏倚区域。我们的方法通过采用分类与回归树（CART）算法，在脓毒症预测背景下有效地识别医疗AI模型中的潜在偏倚。我们通过进行一系列合成数据实验验证了我们的方法，展示了其在受控环境中准确估计偏倚区域的能力。这一概念的有效性通过使用亚特兰大乔治亚州格雷迪纪念医院的电子病历进行实验进一步得到验证。这些测试展示了我们策略在临床中的实际应用。

    arXiv:2312.02959v3 Announce Type: replace-cross  Abstract: With the growing prevalence of machine learning and artificial intelligence-based medical decision support systems, it is equally important to ensure that these systems provide patient outcomes in a fair and equitable fashion. This paper presents an innovative framework for detecting areas of algorithmic bias in medical-AI decision support systems. Our approach efficiently identifies potential biases in medical-AI models, specifically in the context of sepsis prediction, by employing the Classification and Regression Trees (CART) algorithm. We verify our methodology by conducting a series of synthetic data experiments, showcasing its ability to estimate areas of bias in controlled settings precisely. The effectiveness of the concept is further validated by experiments using electronic medical records from Grady Memorial Hospital in Atlanta, Georgia. These tests demonstrate the practical implementation of our strategy in a clini
    
[^4]: 鲁棒的多模态密度估计

    Robust Multi-Modal Density Estimation. (arXiv:2401.10566v1 [cs.LG])

    [http://arxiv.org/abs/2401.10566](http://arxiv.org/abs/2401.10566)

    本文提出了一种名为ROME的鲁棒多模态密度估计方法，该方法利用聚类将多模态样本集分割成多个单模态样本集，并通过简单的KDE估计来估计整体分布。这种方法解决了多模态、非正态和高相关分布估计的挑战。

    

    多模态概率预测模型的发展引发了对综合评估指标的需求。虽然有几个指标可以表征机器学习模型的准确性（例如，负对数似然、Jensen-Shannon散度），但这些指标通常作用于概率密度上。因此，将它们应用于纯粹基于样本的预测模型需要估计底层密度函数。然而，常见的方法如核密度估计（KDE）已被证明在鲁棒性方面存在不足，而更复杂的方法在多模态估计问题中尚未得到评估。在本文中，我们提出了一种非参数的密度估计方法ROME（RObust Multi-modal density Estimator），它解决了估计多模态、非正态和高相关分布的挑战。ROME利用聚类将多模态样本集分割成多个单模态样本集，然后结合简单的KDE估计来得到总体的估计结果。

    Development of multi-modal, probabilistic prediction models has lead to a need for comprehensive evaluation metrics. While several metrics can characterize the accuracy of machine-learned models (e.g., negative log-likelihood, Jensen-Shannon divergence), these metrics typically operate on probability densities. Applying them to purely sample-based prediction models thus requires that the underlying density function is estimated. However, common methods such as kernel density estimation (KDE) have been demonstrated to lack robustness, while more complex methods have not been evaluated in multi-modal estimation problems. In this paper, we present ROME (RObust Multi-modal density Estimator), a non-parametric approach for density estimation which addresses the challenge of estimating multi-modal, non-normal, and highly correlated distributions. ROME utilizes clustering to segment a multi-modal set of samples into multiple uni-modal ones and then combines simple KDE estimates obtained for i
    
[^5]: 自监督表示学习的线性分离能力研究

    On Linear Separation Capacity of Self-Supervised Representation Learning. (arXiv:2310.19041v1 [stat.ML])

    [http://arxiv.org/abs/2310.19041](http://arxiv.org/abs/2310.19041)

    本研究通过探究在多流形模型下，学习的表示何时可以线性分离流形，揭示了自监督学习在数据增强方面的额外好处，从而改善了线性分离能力的信息论最优速率。

    

    自监督学习的最新进展强调了数据增强在从无标签数据中学习数据表示中的有效性。在这些增强表示之上训练线性模型可以得到一个熟练的分类器。尽管在实践中表现出色，但是数据增强如何将非线性数据结构解开为线性可分离表示的机制仍然不清楚。本文旨在通过研究在从多流形模型中绘制数据时，学习到的表示在何种条件下可以线性分离流形来填补这一差距。我们的研究揭示了数据增强除了提供观察数据外，还提供了额外的信息，从而可以改善线性分离容量的信息论最优速率。特别是，我们证明自监督学习可以以比无监督学习更小的距离线性分离流形，突显了数据增强的额外好处。

    Recent advances in self-supervised learning have highlighted the efficacy of data augmentation in learning data representation from unlabeled data. Training a linear model atop these enhanced representations can yield an adept classifier. Despite the remarkable empirical performance, the underlying mechanisms that enable data augmentation to unravel nonlinear data structures into linearly separable representations remain elusive. This paper seeks to bridge this gap by investigating under what conditions learned representations can linearly separate manifolds when data is drawn from a multi-manifold model. Our investigation reveals that data augmentation offers additional information beyond observed data and can thus improve the information-theoretic optimal rate of linear separation capacity. In particular, we show that self-supervised learning can linearly separate manifolds with a smaller distance than unsupervised learning, underscoring the additional benefits of data augmentation. 
    
[^6]: 好表示的液滴：在两层网络中 grokking 作为一阶相变

    Droplets of Good Representations: Grokking as a First Order Phase Transition in Two Layer Networks. (arXiv:2310.03789v1 [stat.ML])

    [http://arxiv.org/abs/2310.03789](http://arxiv.org/abs/2310.03789)

    本研究将自适应核方法应用于两个师生模型，预测了特征学习和 Grokking 的性质，并展示了 Grokking 与相变理论之间的映射关系。

    

    深度神经网络 (DNN) 的一个关键特性是在训练过程中能够学习新的特征。这种深度学习的有趣方面在最近报道的 Grokking 现象中表现得最为明显。虽然主要体现为测试准确性的突变增加，但 Grokking 也被认为是一种超越懒惰学习/高斯过程 (GP) 的现象，涉及特征学习。在这里，我们将特征学习理论的最新发展，自适应核方法，应用于具有立方多项式和模加法教师的两个师生模型。我们在这些模型上提供了关于特征学习和 Grokking 性质的分析预测，并展示了 Grokking 与相变理论之间的映射关系。我们表明，在 Grokking 之后，DNN 的状态类似于一阶相变后的混合相。在这个混合相中，DNN 生成了与之前明显不同的教师的有用内部表示。

    A key property of deep neural networks (DNNs) is their ability to learn new features during training. This intriguing aspect of deep learning stands out most clearly in recently reported Grokking phenomena. While mainly reflected as a sudden increase in test accuracy, Grokking is also believed to be a beyond lazy-learning/Gaussian Process (GP) phenomenon involving feature learning. Here we apply a recent development in the theory of feature learning, the adaptive kernel approach, to two teacher-student models with cubic-polynomial and modular addition teachers. We provide analytical predictions on feature learning and Grokking properties of these models and demonstrate a mapping between Grokking and the theory of phase transitions. We show that after Grokking, the state of the DNN is analogous to the mixed phase following a first-order phase transition. In this mixed phase, the DNN generates useful internal representations of the teacher that are sharply distinct from those before the 
    
[^7]: 超越稳定性：随机Softmax策略梯度方法的收敛分析

    Beyond Stationarity: Convergence Analysis of Stochastic Softmax Policy Gradient Methods. (arXiv:2310.02671v1 [math.OC] CROSS LISTED)

    [http://arxiv.org/abs/2310.02671](http://arxiv.org/abs/2310.02671)

    本论文研究了随机Softmax策略梯度方法的收敛性分析，提出了一种动态策略梯度的组合方法，并通过对参数进行反向训练来更好地利用相关性结构，实现向全局最优值的收敛。

    

    马尔可夫决策过程（MDP）是一种形式化框架，用于建模和解决序贯决策问题。在有限时间范围内，这些问题与最优停止或特定供应链问题以及大型语言模型的训练相关。与无限时间范围内的MDP不同，最优策略并不是稳定的，策略必须在每个时期单独进行学习。实际上，往往同时训练所有参数，忽视了动态规划所暗示的内在结构。本文介绍了一种动态规划和策略梯度的组合方法，称为动态策略梯度，其中参数在时间上以反向方式进行训练。对于表格Softmax参数化，我们对同时和动态策略梯度在精确梯度和采样梯度设置下向全局最优值进行了收敛分析，且没有引入正则化。结果表明，使用动态策略梯度训练可以更好地利用相关性结构，并提供了收敛性证明。

    Markov Decision Processes (MDPs) are a formal framework for modeling and solving sequential decision-making problems. In finite-time horizons such problems are relevant for instance for optimal stopping or specific supply chain problems, but also in the training of large language models. In contrast to infinite horizon MDPs optimal policies are not stationary, policies must be learned for every single epoch. In practice all parameters are often trained simultaneously, ignoring the inherent structure suggested by dynamic programming. This paper introduces a combination of dynamic programming and policy gradient called dynamic policy gradient, where the parameters are trained backwards in time. For the tabular softmax parametrisation we carry out the convergence analysis for simultaneous and dynamic policy gradient towards global optima, both in the exact and sampled gradient settings without regularisation. It turns out that the use of dynamic policy gradient training much better exploi
    
[^8]: 指向因果基础模型: 因果推断与注意力的对偶关系

    Towards Causal Foundation Model: on Duality between Causal Inference and Attention. (arXiv:2310.00809v1 [cs.LG])

    [http://arxiv.org/abs/2310.00809](http://arxiv.org/abs/2310.00809)

    该论文提出了一种名为Causal Inference with Attention (CInA)的新方法，利用因果推断和注意力的对偶关系，在复杂任务中实现了零样本的因果推断。

    

    基于因果推断和注意力之间的对偶连接，我们提出了一种名为Causal Inference with Attention (CInA)的理论上完备的方法，利用多个无标签数据集进行自监督因果学习，并在新数据的未见任务上实现零样本因果推断。我们的实证结果表明了我们的方法在复杂任务中的有效性。

    Foundation models have brought changes to the landscape of machine learning, demonstrating sparks of human-level intelligence across a diverse array of tasks. However, a gap persists in complex tasks such as causal inference, primarily due to challenges associated with intricate reasoning steps and high numerical precision requirements. In this work, we take a first step towards building causally-aware foundation models for complex tasks. We propose a novel, theoretically sound method called Causal Inference with Attention (CInA), which utilizes multiple unlabeled datasets to perform self-supervised causal learning, and subsequently enables zero-shot causal inference on unseen tasks with new data. This is based on our theoretical results that demonstrate the primal-dual connection between optimal covariate balancing and self-attention, facilitating zero-shot causal inference through the final layer of a trained transformer-type architecture. We demonstrate empirically that our approach
    
[^9]: 带有嘈杂治疗和没有侧面信息的可识别因果推断

    Identifiable causal inference with noisy treatment and no side information. (arXiv:2306.10614v1 [cs.LG])

    [http://arxiv.org/abs/2306.10614](http://arxiv.org/abs/2306.10614)

    本论文提出了一种在没有侧面信息和具有复杂非线性依赖性的情况下，纠正因治疗变量不准确测量引起的因果效应估计偏差的模型，并证明了该模型的因果效应估计是可识别的。该方法使用了深度潜在变量模型和分摊权重变分客观函数进行训练。

    

    在某些因果推断场景中，治疗（即原因）变量的测量存在不准确性，例如在流行病学或计量经济学中。未能纠正测量误差的影响可能导致偏差的因果效应估计。以前的研究没有从因果视角研究解决这个问题的方法，同时允许复杂的非线性依赖关系并且不假设可以访问侧面信息。对于这样的场景，本论文提出了一个模型，它假设存在一个连续的治疗变量，该变量测量不准确。建立在现有测量误差模型的基础上，我们证明了我们的模型的因果效应估计是可识别的，即使没有测量误差方差或其他侧面信息的知识。我们的方法依赖于深度潜在变量模型，其中高斯条件由神经网络参数化，并且我们开发了一个分摊权重变分客观函数来训练该模型。

    In some causal inference scenarios, the treatment (i.e. cause) variable is measured inaccurately, for instance in epidemiology or econometrics. Failure to correct for the effect of this measurement error can lead to biased causal effect estimates. Previous research has not studied methods that address this issue from a causal viewpoint while allowing for complex nonlinear dependencies and without assuming access to side information. For such as scenario, this paper proposes a model that assumes a continuous treatment variable which is inaccurately measured. Building on existing results for measurement error models, we prove that our model's causal effect estimates are identifiable, even without knowledge of the measurement error variance or other side information. Our method relies on a deep latent variable model where Gaussian conditionals are parameterized by neural networks, and we develop an amortized importance-weighted variational objective for training the model. Empirical resul
    
[^10]: Gumbel传播下的潜在最优路径变分贝叶斯动态规划

    Latent Optimal Paths by Gumbel Propagation for Variational Bayesian Dynamic Programming. (arXiv:2306.02568v1 [stat.ML])

    [http://arxiv.org/abs/2306.02568](http://arxiv.org/abs/2306.02568)

    该论文使用动态规划和Gumbel传播在VAE的潜在空间中获得结构化稀疏最优路径，从而使得模型可以依赖于未观察到的结构特征信息，并成功实现了文本转语音和歌声合成。

    

    我们提出了一种统一方法，使用动态规划和Gumbel传播在变分自编码器（VAE）的潜在空间中获取结构化稀疏最优路径。我们通过概率软化解，即随机最优路径，来解决经典最优路径问题，并将广泛的DP问题转化为有向无环图，其中所有可能的路径遵循Gibbs分布。我们通过Gumbel分布的属性显示Gibbs分布与消息传递算法的等价性，并提供了变分贝叶斯推理所需的所有要素。我们的方法获取了潜在最优路径，使生成任务的端到端训练成为可能，其中模型依赖于未观察到的结构特征的信息。我们验证了我们方法的行为，并展示了其在两个真实世界应用中的适用性：文本转语音和歌声合成。

    We propose a unified approach to obtain structured sparse optimal paths in the latent space of a variational autoencoder (VAE) using dynamic programming and Gumbel propagation. We solve the classical optimal path problem by a probability softening solution, called the stochastic optimal path, and transform a wide range of DP problems into directed acyclic graphs in which all possible paths follow a Gibbs distribution. We show the equivalence of the Gibbs distribution to a message-passing algorithm by the properties of the Gumbel distribution and give all the ingredients required for variational Bayesian inference. Our approach obtaining latent optimal paths enables end-to-end training for generative tasks in which models rely on the information of unobserved structural features. We validate the behavior of our approach and showcase its applicability in two real-world applications: text-to-speech and singing voice synthesis.
    
[^11]: 在实值函数中对抗鲁棒PAC学习性的研究

    Adversarially Robust PAC Learnability of Real-Valued Functions. (arXiv:2206.12977v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.12977](http://arxiv.org/abs/2206.12977)

    该论文研究了在实值函数中对抗鲁棒PAC学习性，发现有限胖折射维的类既可以在实现和不可知设置中被学习，凸函数类可以正确学习，而一些非凸函数类需要不正当的学习算法。

    

    我们研究了在使用$\ell_p$损失和任意扰动集的回归设置中，对测试时对抗性攻击的稳健性。我们探讨了在这种情况下哪些函数类是PAC可学习的。我们表明有限胖折射维的类既可以在实现和不可知设置中被学习。此外，对于凸函数类，它们甚至可以正确学习。相比之下，一些非凸函数类显然需要不正当的学习算法。我们的主要技术基于构建一个由胖折射维决定大小的具有对抗鲁棒性的样本压缩方案。在此过程中，我们介绍了一个新颖的面向实值函数的不可知样本压缩方案，这可能是具有独立兴趣的。

    We study robustness to test-time adversarial attacks in the regression setting with $\ell_p$ losses and arbitrary perturbation sets. We address the question of which function classes are PAC learnable in this setting. We show that classes of finite fat-shattering dimension are learnable in both realizable and agnostic settings. Moreover, for convex function classes, they are even properly learnable. In contrast, some non-convex function classes provably require improper learning algorithms. Our main technique is based on a construction of an adversarially robust sample compression scheme of a size determined by the fat-shattering dimension. Along the way, we introduce a novel agnostic sample compression scheme for real-valued functions, which may be of independent interest.
    
[^12]: 基于回归的超标概率预测方法用于显著波高预测

    Exceedance Probability Forecasting via Regression for Significant Wave Height Prediction. (arXiv:2206.09821v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2206.09821](http://arxiv.org/abs/2206.09821)

    本论文提出了一种基于回归的超标概率预测方法，用于预测显著波高，通过利用预测来估计超标概率，取得了更好的效果。

    

    显著波高预测是海洋数据分析中的一个关键问题。预测显著波高对于估计波能产生是至关重要的。此外，及时预测大浪的到来对于确保航海作业的安全很重要。我们将预测显著波高的极端值作为超标概率预测问题。因此，我们旨在估计显著波高将超过预定义阈值的概率。通常使用概率二分类模型来解决这个任务。相反，我们提出了一种基于预测模型的新方法。该方法利用未来观测的预测来根据累积分布函数估计超标概率。我们使用来自加拿大哈利法克斯海岸的浮标数据进行了实验。结果表明，所提出的方法更好。

    Significant wave height forecasting is a key problem in ocean data analytics. Predicting the significant wave height is crucial for estimating the energy production from waves. Moreover, the timely prediction of large waves is important to ensure the safety of maritime operations, e.g. passage of vessels. We frame the task of predicting extreme values of significant wave height as an exceedance probability forecasting problem. Accordingly, we aim at estimating the probability that the significant wave height will exceed a predefined threshold. This task is usually solved using a probabilistic binary classification model. Instead, we propose a novel approach based on a forecasting model. The method leverages the forecasts for the upcoming observations to estimate the exceedance probability according to the cumulative distribution function. We carried out experiments using data from a buoy placed in the coast of Halifax, Canada. The results suggest that the proposed methodology is better
    
[^13]: 通过投影功率方法进行相关维格纳模型的有种子图匹配

    Seeded graph matching for the correlated Wigner model via the projected power method. (arXiv:2204.04099v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2204.04099](http://arxiv.org/abs/2204.04099)

    本文研究了在相关维格纳模型下的有种子图匹配问题，通过分析表明，使用投影功率方法（PPM）作为图匹配算法可以在给定接近真实匹配的种子的情况下高概率地改进种子并恢复真实匹配。

    

    在图匹配问题中，我们观察两个图G和H，并通过最大化边协议的一些度量来找到它们顶点之间的赋值（或匹配）。我们假设观察到的图对G和H是从相关维格纳模型中抽取的，这是一个用于相关加权图的流行模型，其中G和H的邻接矩阵的元素是独立的高斯分布，并且G的每条边与H的其中一条边（由未知的匹配确定）相关联，边相关性由参数σ∈[0,1)描述。在本文中，我们分析了作为“有种子”的图匹配算法的“投影功率方法”（PPM）的性能，其中我们提供一个部分正确的初始匹配（称为种子）作为附加信息。我们证明，如果种子足够接近真实匹配，则PPM在高概率下会迭代改进种子并恢复真实匹配（或恢复最大化边协议）。

    In the \emph{graph matching} problem we observe two graphs $G,H$ and the goal is to find an assignment (or matching) between their vertices such that some measure of edge agreement is maximized. We assume in this work that the observed pair $G,H$ has been drawn from the correlated Wigner model -- a popular model for correlated weighted graphs -- where the entries of the adjacency matrices of $G$ and $H$ are independent Gaussians and each edge of $G$ is correlated with one edge of $H$ (determined by the unknown matching) with the edge correlation described by a parameter $\sigma\in [0,1)$. In this paper, we analyse the performance of the \emph{projected power method} (PPM) as a \emph{seeded} graph matching algorithm where we are given an initial partially correct matching (called the seed) as side information. We prove that if the seed is close enough to the ground-truth matching, then with high probability, PPM iteratively improves the seed and recovers the ground-truth matching (eithe
    
[^14]: 图形神经网络在概率误差模型下的敏感性

    Graph Neural Network Sensitivity Under Probabilistic Error Model. (arXiv:2203.07831v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2203.07831](http://arxiv.org/abs/2203.07831)

    本文研究了概率误差模型对图卷积网络（GCN）性能的影响，并证明了误差模型下邻接矩阵的受限性。通过实验验证了这种误差界限，并研究了GCN在这种概率误差模型下的准确性敏感性。

    

    图卷积网络（GCN）可以通过图卷积成功学习图信号表示。图卷积依赖于图滤波器，其中包含数据的拓扑依赖关系并传播数据特征。然而，在传播矩阵（例如邻接矩阵）中的估计误差可能对图滤波器和GCNs产生重大影响。本文研究概率图误差模型对GCN性能的影响。我们证明了在误差模型下的邻接矩阵受到图大小和误差概率函数的限制。我们进一步分析了带有自循环的归一化邻接矩阵的上界。最后，我们通过在合成数据集上运行实验来说明误差界限，并研究简单GCN在这种概率误差模型下的准确性敏感性。

    Graph convolutional networks (GCNs) can successfully learn the graph signal representation by graph convolution. The graph convolution depends on the graph filter, which contains the topological dependency of data and propagates data features. However, the estimation errors in the propagation matrix (e.g., the adjacency matrix) can have a significant impact on graph filters and GCNs. In this paper, we study the effect of a probabilistic graph error model on the performance of the GCNs. We prove that the adjacency matrix under the error model is bounded by a function of graph size and error probability. We further analytically specify the upper bound of a normalized adjacency matrix with self-loop added. Finally, we illustrate the error bounds by running experiments on a synthetic dataset and study the sensitivity of a simple GCN under this probabilistic error model on accuracy.
    
[^15]: MARS via LASSO.（arXiv:2111.11694v2 [math.ST] 已更新）

    MARS via LASSO. (arXiv:2111.11694v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2111.11694](http://arxiv.org/abs/2111.11694)

    本文提出了一种自然lasso变体的MARS方法，通过减少对维度的依赖来获得收敛率，并与使用平滑性约束的非参数估计技术联系在一起。

    

    多元自适应回归样条（Multivariate Adaptive Regression Splines，MARS）是Friedman在1991年提出的一种非参数回归方法。MARS将简单的非线性和非加性函数拟合到回归数据上。本文提出并研究了MARS方法的一种自然lasso变体。我们的方法是基于最小二乘估计，通过考虑MARS基础函数的无限维线性组合并强加基于变分的复杂度约束条件来获得函数的凸类。虽然我们的估计是定义为无限维优化问题的解，但其可以通过有限维凸优化来计算。在一些标准设计假设下，我们证明了我们的估计器仅在维度上对数收敛，因此在一定程度上避免了通常的维度灾难。我们还表明，我们的方法自然地与基于平滑性约束的非参数估计技术相联系。

    Multivariate adaptive regression splines (MARS) is a popular method for nonparametric regression introduced by Friedman in 1991. MARS fits simple nonlinear and non-additive functions to regression data. We propose and study a natural lasso variant of the MARS method. Our method is based on least squares estimation over a convex class of functions obtained by considering infinite-dimensional linear combinations of functions in the MARS basis and imposing a variation based complexity constraint. Our estimator can be computed via finite-dimensional convex optimization, although it is defined as a solution to an infinite-dimensional optimization problem. Under a few standard design assumptions, we prove that our estimator achieves a rate of convergence that depends only logarithmically on dimension and thus avoids the usual curse of dimensionality to some extent. We also show that our method is naturally connected to nonparametric estimation techniques based on smoothness constraints. We i
    

