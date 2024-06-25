# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Dynamical Model of Neural Scaling Laws](https://rss.arxiv.org/abs/2402.01092) | 这篇论文提出了一个动力学模型来解释神经缩放定律。通过分析梯度下降训练的随机特征模型，研究发现训练时间和模型大小的缩放具有不同的幂律指数，而计算最优缩放规则要求增加训练步数快于增加模型参数，与实证观察相一致。 |
| [^2] | [Learning Action-based Representations Using Invariance](https://arxiv.org/abs/2403.16369) | 提出了一种新的方法，动作双模拟编码，通过递归不变性约束扩展了单步控制性，学习了一个可以平滑折扣远期元素的多步控制度量 |
| [^3] | [Fast, accurate and lightweight sequential simulation-based inference using Gaussian locally linear mappings](https://arxiv.org/abs/2403.07454) | 使用结构混合概率分布提供了准确的后验推断，同时具有更小的计算占用量，相较于现有的基于神经网络的SBI方法。 |
| [^4] | [Hallmarks of Optimization Trajectories in Neural Networks and LLMs: The Lengths, Bends, and Dead Ends](https://arxiv.org/abs/2403.07379) | 分析神经网络和LLMs中优化轨迹的复杂性，揭示了优化过程中的关键特征，包括方向探索和方向正则化。 |
| [^5] | [Validation of ML-UQ calibration statistics using simulated reference values: a sensitivity analysis](https://arxiv.org/abs/2403.00423) | 本研究探讨了ML-UQ校准统计量的使用问题，发现一些统计量对于生成分布的选择过于敏感，可能影响校准诊断。 |
| [^6] | [Fixed Confidence Best Arm Identification in the Bayesian Setting](https://arxiv.org/abs/2402.10429) | 该研究在贝叶斯设置中探讨了固定置信度最佳臂识别问题，证明了传统频率设定下的算法在此设置下表现次优，并引入了一种性能与理论下限相匹配的连续排除变种。 |
| [^7] | [Gradient descent induces alignment between weights and the empirical NTK for deep non-linear networks](https://arxiv.org/abs/2402.05271) | 了解神经网络从输入-标签对中提取统计信息的机制是监督学习中最重要的未解决问题之一。前人的研究表明，在训练过程中，权重的格拉姆矩阵与模型的平均梯度外积成正比，这被称为神经特征分析（NFA）。本研究解释了这种相关性的出现，并发现NFA等价于权重矩阵的左奇异结构与与这些权重相关的经验神经切线核的显著成分之间的对齐。在早期训练阶段，可以通过解析的方式预测NFA的发展速度。 |
| [^8] | [On Parameter Estimation in Deviated Gaussian Mixture of Experts](https://arxiv.org/abs/2402.05220) | 在偏离高斯混合专家模型中，本文通过构造Voronoi-based损失函数来解决参数估计问题。 |
| [^9] | [On Least Squares Estimation in Softmax Gating Mixture of Experts](https://arxiv.org/abs/2402.02952) | 本研究探讨了在确定性MoE模型下使用最小二乘估计器的性能，并建立了强可识别性条件来描述不同类型专家函数的收敛行为。 |
| [^10] | [Is Temperature Sample Efficient for Softmax Gaussian Mixture of Experts?.](http://arxiv.org/abs/2401.13875) | 本文研究了温度对Softmax高斯混合专家的采样效率的影响，证明了由于温度和其他模型参数之间的相互作用，参数估计的收敛速度较慢，并且可能很慢。 |
| [^11] | [Generative Fractional Diffusion Models.](http://arxiv.org/abs/2310.17638) | 该论文提出了一种基于分数阶扩散模型的生成模型，其中通过将分数布朗运动表示为奥恩斯坦-乌伦贝克过程的随机积分，实现了驱动噪声收敛到非马尔可夫过程的效果。这是第一个在具有无限二次变差的随机过程上构建生成模型的尝试。 |
| [^12] | [A General Theory for Softmax Gating Multinomial Logistic Mixture of Experts.](http://arxiv.org/abs/2310.14188) | 该论文提出了一种通用理论，用于研究Softmax Gating Multinomial Logistic Mixture of Experts模型。通过建立模型的收敛速度，揭示了softmax gating和专家函数之间存在的互作用，同时提出了一种修改后的softmax gating函数。 |
| [^13] | [Optimal Best Arm Identification with Fixed Confidence in Restless Bandits.](http://arxiv.org/abs/2310.13393) | 在多臂赌博机中，我们研究了以固定置信度进行最优臂识别。我们提出了一种新的策略来识别具有最大平均值的臂，同时限制了决策错误的概率，并建立了停止时间增长率的下界。 |
| [^14] | [Discovering Mixtures of Structural Causal Models from Time Series Data.](http://arxiv.org/abs/2310.06312) | 这项研究提出了一种从时间序列数据中发现混合结构因果模型的方法，通过推断潜在因果模型以及每个样本属于特定混合成分的概率，通过实验证明了该方法在因果发现任务中的优越性。 |
| [^15] | [Causal Fair Machine Learning via Rank-Preserving Interventional Distributions.](http://arxiv.org/abs/2307.12797) | 通过保持排序的干预分布，我们提出了一种因果公平的机器学习方法，通过在一个理想的世界中消除受保护属性对目标的因果影响来减少不公平。 |
| [^16] | [Of Mice and Mates: Automated Classification and Modelling of Mouse Behaviour in Groups using a Single Model across Cages.](http://arxiv.org/abs/2306.03066) | 本文提供了自动分类和模拟群体老鼠行为的工具，通过单一模型跨笼使用置换矩阵匹配老鼠身份，在家鼠环境下研究老鼠可以捕捉到个体行为的时间因素，而且无需人为干预。 |
| [^17] | [Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB.](http://arxiv.org/abs/2303.00890) | 本研究比较了BBOB上五种高维贝叶斯优化算法与传统方法以及CMA-ES算法的性能，结果表明... (根据论文的具体内容进行总结) |
| [^18] | [Communication-Efficient Distributed Estimation and Inference for Cox's Model.](http://arxiv.org/abs/2302.12111) | 我们提出了一种高效的分布式算法，用于在高维稀疏Cox比例风险模型中估计和推断，通过引入一种新的去偏差方法，我们可以产生渐近有效的分布式置信区间，并提供了有效的分布式假设检验。 |

# 详细

[^1]: 神经缩放定律的动力学模型

    A Dynamical Model of Neural Scaling Laws

    [https://rss.arxiv.org/abs/2402.01092](https://rss.arxiv.org/abs/2402.01092)

    这篇论文提出了一个动力学模型来解释神经缩放定律。通过分析梯度下降训练的随机特征模型，研究发现训练时间和模型大小的缩放具有不同的幂律指数，而计算最优缩放规则要求增加训练步数快于增加模型参数，与实证观察相一致。

    

    在各种任务中，神经网络的性能随着训练时间、数据集大小和模型大小的增加而预测性地提高，跨多个数量级。这种现象被称为神经缩放定律。最重要的是计算最优缩放定律，它报告了在选择最佳模型大小时性能与计算数量的关系。我们分析了一个通过梯度下降进行训练和泛化的随机特征模型作为网络训练和泛化的可解模型。这个模型复现了关于神经缩放定律的许多观察结果。首先，我们的模型对于为什么训练时间和模型大小的缩放具有不同的幂律指数提出了一个预测。因此，理论预测了一种不对称的计算最优缩放规则，其中训练步数的增加速度快于模型参数的增加速度，与最近的实证观察一致。其次，观察到在训练的早期，网络会收敛到无限宽度情况下的结果。

    On a variety of tasks, the performance of neural networks predictably improves with training time, dataset size and model size across many orders of magnitude. This phenomenon is known as a neural scaling law. Of fundamental importance is the compute-optimal scaling law, which reports the performance as a function of units of compute when choosing model sizes optimally. We analyze a random feature model trained with gradient descent as a solvable model of network training and generalization. This reproduces many observations about neural scaling laws. First, our model makes a prediction about why the scaling of performance with training time and with model size have different power law exponents. Consequently, the theory predicts an asymmetric compute-optimal scaling rule where the number of training steps are increased faster than model parameters, consistent with recent empirical observations. Second, it has been observed that early in training, networks converge to their infinite-wi
    
[^2]: 使用不变性学习基于动作的表示

    Learning Action-based Representations Using Invariance

    [https://arxiv.org/abs/2403.16369](https://arxiv.org/abs/2403.16369)

    提出了一种新的方法，动作双模拟编码，通过递归不变性约束扩展了单步控制性，学习了一个可以平滑折扣远期元素的多步控制度量

    

    强化学习代理使用高维度观测必须能够在许多外源性干扰中识别相关状态特征。一个能够捕捉可控性的表示通过确定影响代理控制的因素来识别这些状态元素。虽然诸如逆动力学和互信息等方法可以捕捉有限数量的时间步的可控性，但捕获长时间元素仍然是一个具有挑战性的问题。短视的可控性可以捕捉代理即将撞向墙壁的瞬间，但不能在代理还有一定距离之时捕捉墙壁的控制相关性。为解决这个问题，我们提出了动作双模拟编码，这是一种受到双模拟不变量假度量启发的方法，它通过递归不变性约束扩展了单步控制性。通过这种方式，动作双模拟学习了一个平滑折扣远期元素的多步控制度量。

    arXiv:2403.16369v1 Announce Type: cross  Abstract: Robust reinforcement learning agents using high-dimensional observations must be able to identify relevant state features amidst many exogeneous distractors. A representation that captures controllability identifies these state elements by determining what affects agent control. While methods such as inverse dynamics and mutual information capture controllability for a limited number of timesteps, capturing long-horizon elements remains a challenging problem. Myopic controllability can capture the moment right before an agent crashes into a wall, but not the control-relevance of the wall while the agent is still some distance away. To address this we introduce action-bisimulation encoding, a method inspired by the bisimulation invariance pseudometric, that extends single-step controllability with a recursive invariance constraint. By doing this, action-bisimulation learns a multi-step controllability metric that smoothly discounts dist
    
[^3]: 使用高斯局部线性映射进行快速、准确和轻量级的顺序仿真推断

    Fast, accurate and lightweight sequential simulation-based inference using Gaussian locally linear mappings

    [https://arxiv.org/abs/2403.07454](https://arxiv.org/abs/2403.07454)

    使用结构混合概率分布提供了准确的后验推断，同时具有更小的计算占用量，相较于现有的基于神经网络的SBI方法。

    

    arXiv:2403.07454v1 公告类型: 跨领域 摘要: 针对具有难以处理的似然函数的复杂模型的贝叶斯推断可以使用多次调用计算模拟器的算法来解决。 这些方法被统称为“基于仿真的推断”（SBI）。 最近的SBI方法利用神经网络（NN）提供近似但表达丰富的构造，用于不可用的似然函数和后验分布。 然而，它们通常无法实现准确性和计算需求之间的最佳折衷。 在这项工作中，我们提出了一种提供似然函数和后验分布近似的替代方法，使用结构化的概率分布混合物。 相对于最先进的基于NN的SBI方法，我们的方法在产生准确的后验推断的同时，具有更小的计算占用量。 我们在SBI文献中的几个基准模型上展示了我们的结果。

    arXiv:2403.07454v1 Announce Type: cross  Abstract: Bayesian inference for complex models with an intractable likelihood can be tackled using algorithms performing many calls to computer simulators. These approaches are collectively known as "simulation-based inference" (SBI). Recent SBI methods have made use of neural networks (NN) to provide approximate, yet expressive constructs for the unavailable likelihood function and the posterior distribution. However, they do not generally achieve an optimal trade-off between accuracy and computational demand. In this work, we propose an alternative that provides both approximations to the likelihood and the posterior distribution, using structured mixtures of probability distributions. Our approach produces accurate posterior inference when compared to state-of-the-art NN-based SBI methods, while exhibiting a much smaller computational footprint. We illustrate our results on several benchmark models from the SBI literature.
    
[^4]: 神经网络和LLMs中优化轨迹的特征：长度、拐点和死胡同

    Hallmarks of Optimization Trajectories in Neural Networks and LLMs: The Lengths, Bends, and Dead Ends

    [https://arxiv.org/abs/2403.07379](https://arxiv.org/abs/2403.07379)

    分析神经网络和LLMs中优化轨迹的复杂性，揭示了优化过程中的关键特征，包括方向探索和方向正则化。

    

    我们提出了一种全新的方法来理解神经网络的机制，通过分析其优化轨迹中包含的丰富参数结构。为此，我们引入了一些关于优化轨迹复杂性的自然概念，既定性又定量地揭示了各种优化选择（如动量、权重衰减和批大小）之间所涉及的内在微妙和相互作用。我们利用这些概念来提供关于深度神经网络优化本质的关键特征：何时顺利进行，何时陷入死胡同。此外，基于我们的轨迹视角，我们揭示了动量和权重衰减之间促进方向探索的交织行为，以及其他一些行为的方向正则化行为。我们在大规模视觉和语言设置中进行实验，包括具有最多120亿个参数的大型语言模型（LLMs）。

    arXiv:2403.07379v1 Announce Type: cross  Abstract: We propose a fresh take on understanding the mechanisms of neural networks by analyzing the rich structure of parameters contained within their optimization trajectories. Towards this end, we introduce some natural notions of the complexity of optimization trajectories, both qualitative and quantitative, which reveal the inherent nuance and interplay involved between various optimization choices, such as momentum, weight decay, and batch size. We use them to provide key hallmarks about the nature of optimization in deep neural networks: when it goes right, and when it finds itself in a dead end. Further, thanks to our trajectory perspective, we uncover an intertwined behaviour of momentum and weight decay that promotes directional exploration, as well as a directional regularization behaviour of some others. We perform experiments over large-scale vision and language settings, including large language models (LLMs) with up to 12 billio
    
[^5]: 使用模拟参考值验证ML-UQ校准统计量：一项敏感性分析

    Validation of ML-UQ calibration statistics using simulated reference values: a sensitivity analysis

    [https://arxiv.org/abs/2403.00423](https://arxiv.org/abs/2403.00423)

    本研究探讨了ML-UQ校准统计量的使用问题，发现一些统计量对于生成分布的选择过于敏感，可能影响校准诊断。

    

    一些流行的机器学习不确定性量化（ML-UQ）校准统计量没有预定义的参考值，主要用于比较研究。因此，校准几乎从不被验证，诊断留给读者的判断。提出了基于实际不确定性导出的合成校准数据集的模拟参考值，以弥补这一问题。由于用于模拟合成误差的生成概率分布通常没有约束，所以模拟参考值对生成分布选择的敏感性可能会成为问题，对校准诊断产生怀疑。本研究探讨了这一问题的各个方面，并显示一些统计量对于用于验证时生成分布的选择过于敏感，当生成分布未知时。例如，

    arXiv:2403.00423v1 Announce Type: cross  Abstract: Some popular Machine Learning Uncertainty Quantification (ML-UQ) calibration statistics do not have predefined reference values and are mostly used in comparative studies. In consequence, calibration is almost never validated and the diagnostic is left to the appreciation of the reader. Simulated reference values, based on synthetic calibrated datasets derived from actual uncertainties, have been proposed to palliate this problem. As the generative probability distribution for the simulation of synthetic errors is often not constrained, the sensitivity of simulated reference values to the choice of generative distribution might be problematic, shedding a doubt on the calibration diagnostic. This study explores various facets of this problem, and shows that some statistics are excessively sensitive to the choice of generative distribution to be used for validation when the generative distribution is unknown. This is the case, for instan
    
[^6]: 在贝叶斯设置中的固定置信度最佳臂识别问题

    Fixed Confidence Best Arm Identification in the Bayesian Setting

    [https://arxiv.org/abs/2402.10429](https://arxiv.org/abs/2402.10429)

    该研究在贝叶斯设置中探讨了固定置信度最佳臂识别问题，证明了传统频率设定下的算法在此设置下表现次优，并引入了一种性能与理论下限相匹配的连续排除变种。

    

    我们考虑了贝叶斯设置中的固定置信度最佳臂识别（FC-BAI）问题。该问题旨在在已知先验采样的情况下以固定置信水平找到均值最大的臂。大多数关于FC-BAI问题的研究都是在频率设定中进行的，在该设定下，游戏开始前即确定了赌博模型。我们证明了在贝叶斯设置中，传统的在频率设定中研究的FC-BAI算法（如track-and-stop和top-two算法）会导致任意次优的表现。我们同时证明了在贝叶斯设置下预期样本数的下限，并引入了一种连续排除的变种，其性能与下限相匹配，最多差一个对数因子。仿真验证了理论结果。

    arXiv:2402.10429v1 Announce Type: cross  Abstract: We consider the fixed-confidence best arm identification (FC-BAI) problem in the Bayesian Setting. This problem aims to find the arm of the largest mean with a fixed confidence level when the bandit model has been sampled from the known prior. Most studies on the FC-BAI problem have been conducted in the frequentist setting, where the bandit model is predetermined before the game starts. We show that the traditional FC-BAI algorithms studied in the frequentist setting, such as track-and-stop and top-two algorithms, result in arbitrary suboptimal performances in the Bayesian setting. We also prove a lower bound of the expected number of samples in the Bayesian setting and introduce a variant of successive elimination that has a matching performance with the lower bound up to a logarithmic factor. Simulations verify the theoretical results.
    
[^7]: 梯度下降引发了深度非线性网络权重与经验NTK之间的对齐

    Gradient descent induces alignment between weights and the empirical NTK for deep non-linear networks

    [https://arxiv.org/abs/2402.05271](https://arxiv.org/abs/2402.05271)

    了解神经网络从输入-标签对中提取统计信息的机制是监督学习中最重要的未解决问题之一。前人的研究表明，在训练过程中，权重的格拉姆矩阵与模型的平均梯度外积成正比，这被称为神经特征分析（NFA）。本研究解释了这种相关性的出现，并发现NFA等价于权重矩阵的左奇异结构与与这些权重相关的经验神经切线核的显著成分之间的对齐。在早期训练阶段，可以通过解析的方式预测NFA的发展速度。

    

    理解神经网络从输入-标签对中提取统计信息的机制是监督学习中最重要的未解决问题之一。先前的研究已经确定，在一般结构的训练神经网络中，权重的格拉姆矩阵与模型的平均梯度外积成正比，这个说法被称为神经特征分析（NFA）。然而，这些数量在训练过程中如何相关尚不清楚。在这项工作中，我们解释了这种相关性的出现。我们发现NFA等价于权重矩阵的左奇异结构与与这些权重相关的经验神经切线核的显著成分之间的对齐。我们证明了先前研究中引入的NFA是由隔离这种对齐的中心化NFA驱动的。我们还展示了在早期训练阶段，可以通过解析的方式预测NFA的发展速度。

    Understanding the mechanisms through which neural networks extract statistics from input-label pairs is one of the most important unsolved problems in supervised learning. Prior works have identified that the gram matrices of the weights in trained neural networks of general architectures are proportional to the average gradient outer product of the model, in a statement known as the Neural Feature Ansatz (NFA). However, the reason these quantities become correlated during training is poorly understood. In this work, we explain the emergence of this correlation. We identify that the NFA is equivalent to alignment between the left singular structure of the weight matrices and a significant component of the empirical neural tangent kernels associated with those weights. We establish that the NFA introduced in prior works is driven by a centered NFA that isolates this alignment. We show that the speed of NFA development can be predicted analytically at early training times in terms of sim
    
[^8]: 关于偏离高斯混合专家模型参数估计的研究

    On Parameter Estimation in Deviated Gaussian Mixture of Experts

    [https://arxiv.org/abs/2402.05220](https://arxiv.org/abs/2402.05220)

    在偏离高斯混合专家模型中，本文通过构造Voronoi-based损失函数来解决参数估计问题。

    

    本文考虑了在偏离高斯混合专家模型中的参数估计问题，其中数据由$(1 - \lambda^{\ast}) g_0(Y| X)+ \lambda^{\ast} \sum_{i = 1}^{k_{\ast}} p_{i}^{\ast} f(Y|(a_{i}^{\ast})^{\top}X+b_i^{\ast},\sigma_{i}^{\ast})$生成，其中$X, Y$分别是协变量向量和响应变量，$g_{0}(Y|X)$是已知函数，$\lambda^{\ast} \in [0, 1]$是真实但未知的混合比例，$(p_{i}^{\ast}, a_{i}^{\ast}, b_{i}^{\ast}, \sigma_{i}^{\ast})$对于$1 \leq i \leq k^{\ast}$是高斯混合专家模型的未知参数。该问题源自于拟合优度检验，当我们希望检验数据是否由$g_{0}(Y|X)$（零假设）生成，还是由整个混合（备择假设）生成。基于专家函数的代数结构和$g_0$与混合部分的可区分性，我们构造了新的基于Voronoi的损失函数来捕捉c

    We consider the parameter estimation problem in the deviated Gaussian mixture of experts in which the data are generated from $(1 - \lambda^{\ast}) g_0(Y| X)+ \lambda^{\ast} \sum_{i = 1}^{k_{\ast}} p_{i}^{\ast} f(Y|(a_{i}^{\ast})^{\top}X+b_i^{\ast},\sigma_{i}^{\ast})$, where $X, Y$ are respectively a covariate vector and a response variable, $g_{0}(Y|X)$ is a known function, $\lambda^{\ast} \in [0, 1]$ is true but unknown mixing proportion, and $(p_{i}^{\ast}, a_{i}^{\ast}, b_{i}^{\ast}, \sigma_{i}^{\ast})$ for $1 \leq i \leq k^{\ast}$ are unknown parameters of the Gaussian mixture of experts. This problem arises from the goodness-of-fit test when we would like to test whether the data are generated from $g_{0}(Y|X)$ (null hypothesis) or they are generated from the whole mixture (alternative hypothesis). Based on the algebraic structure of the expert functions and the distinguishability between $g_0$ and the mixture part, we construct novel Voronoi-based loss functions to capture the c
    
[^9]: 关于Softmax Gating混合专家模型中最小二乘估计的研究

    On Least Squares Estimation in Softmax Gating Mixture of Experts

    [https://arxiv.org/abs/2402.02952](https://arxiv.org/abs/2402.02952)

    本研究探讨了在确定性MoE模型下使用最小二乘估计器的性能，并建立了强可识别性条件来描述不同类型专家函数的收敛行为。

    

    专家模型是一种统计机器学习设计，使用Softmax Gating函数聚合多个专家网络，以形成一个更复杂和表达力更强的模型。尽管由于可扩展性而在多个应用领域中广泛使用，但MoE模型的数学和统计性质复杂且难以分析。因此，以前的理论工作主要集中在概率MoE模型上，这些模型假设数据是由高斯MoE模型生成的，这在实践中是不切实际的。在这项工作中，我们研究了在确定性MoE模型下最小二乘估计器（LSE）的性能，在该模型中，数据根据回归模型进行采样，这是一个尚未被充分探索的设置。我们建立了一个称为强可识别性的条件，以表征不同类型专家函数的收敛行为。我们证明了对于强可识别专家的估计速度，即

    Mixture of experts (MoE) model is a statistical machine learning design that aggregates multiple expert networks using a softmax gating function in order to form a more intricate and expressive model. Despite being commonly used in several applications owing to their scalability, the mathematical and statistical properties of MoE models are complex and difficult to analyze. As a result, previous theoretical works have primarily focused on probabilistic MoE models by imposing the impractical assumption that the data are generated from a Gaussian MoE model. In this work, we investigate the performance of the least squares estimators (LSE) under a deterministic MoE model where the data are sampled according to a regression model, a setting that has remained largely unexplored. We establish a condition called strong identifiability to characterize the convergence behavior of various types of expert functions. We demonstrate that the rates for estimating strongly identifiable experts, namel
    
[^10]: 温度对Softmax高斯混合专家是否具有采样效率？

    Is Temperature Sample Efficient for Softmax Gaussian Mixture of Experts?. (arXiv:2401.13875v1 [stat.ML])

    [http://arxiv.org/abs/2401.13875](http://arxiv.org/abs/2401.13875)

    本文研究了温度对Softmax高斯混合专家的采样效率的影响，证明了由于温度和其他模型参数之间的相互作用，参数估计的收敛速度较慢，并且可能很慢。

    

    最近，密集-稀疏门控专家混合模型（MoE）已成为广为使用的稀疏MoE的有效替代方案。与后者模型中固定激活的专家数量不同，前者模型利用温度来控制softmax权重分布和MoE的稀疏性，在训练过程中稳定专家的专业化。然而，尽管以前有尝试从理论上理解稀疏MoE的方法，对于密集到稀疏门控MoE的全面分析仍然困难重重。因此，本文旨在探讨密集到稀疏门控对Gaussian MoE下的最大似然估计的影响。我们证明由于温度和其他模型参数之间通过一些偏微分方程的相互作用，参数估计的收敛速度比任何多项式速率都要慢，并且可能慢到$\mathcal{

    Dense-to-sparse gating mixture of experts (MoE) has recently become an effective alternative to a well-known sparse MoE. Rather than fixing the number of activated experts as in the latter model, which could limit the investigation of potential experts, the former model utilizes the temperature to control the softmax weight distribution and the sparsity of the MoE during training in order to stabilize the expert specialization. Nevertheless, while there are previous attempts to theoretically comprehend the sparse MoE, a comprehensive analysis of the dense-to-sparse gating MoE has remained elusive. Therefore, we aim to explore the impacts of the dense-to-sparse gate on the maximum likelihood estimation under the Gaussian MoE in this paper. We demonstrate that due to interactions between the temperature and other model parameters via some partial differential equations, the convergence rates of parameter estimations are slower than any polynomial rates, and could be as slow as $\mathcal{
    
[^11]: 生成分数阶扩散模型

    Generative Fractional Diffusion Models. (arXiv:2310.17638v1 [cs.LG])

    [http://arxiv.org/abs/2310.17638](http://arxiv.org/abs/2310.17638)

    该论文提出了一种基于分数阶扩散模型的生成模型，其中通过将分数布朗运动表示为奥恩斯坦-乌伦贝克过程的随机积分，实现了驱动噪声收敛到非马尔可夫过程的效果。这是第一个在具有无限二次变差的随机过程上构建生成模型的尝试。

    

    我们将基于分数布朗运动（FBM）的连续时间框架推广到基于分数布朗运动的近似形式。我们通过将FBM表示为家族奥恩斯坦-乌伦贝克过程的随机积分，推导出连续再参数化技巧和逆时模型，定义了具有驱动噪声收敛到无限二次变差的非马尔可夫过程的生成分数阶扩散模型（GFDM）。FBM的赫斯特指数$H \in (0,1)$ 可以控制路径变换分布的粗糙度。据我们所知，这是第一次尝试在具有无限二次变差的随机过程上建立生成模型。

    We generalize the continuous time framework for score-based generative models from an underlying Brownian motion (BM) to an approximation of fractional Brownian motion (FBM). We derive a continuous reparameterization trick and the reverse time model by representing FBM as a stochastic integral over a family of Ornstein-Uhlenbeck processes to define generative fractional diffusion models (GFDM) with driving noise converging to a non-Markovian process of infinite quadratic variation. The Hurst index $H\in(0,1)$ of FBM enables control of the roughness of the distribution transforming path. To the best of our knowledge, this is the first attempt to build a generative model upon a stochastic process with infinite quadratic variation.
    
[^12]: 一种Softmax Gating Multinomial Logistic Mixture of Experts的通用理论

    A General Theory for Softmax Gating Multinomial Logistic Mixture of Experts. (arXiv:2310.14188v1 [stat.ML])

    [http://arxiv.org/abs/2310.14188](http://arxiv.org/abs/2310.14188)

    该论文提出了一种通用理论，用于研究Softmax Gating Multinomial Logistic Mixture of Experts模型。通过建立模型的收敛速度，揭示了softmax gating和专家函数之间存在的互作用，同时提出了一种修改后的softmax gating函数。

    

    Mixture-of-experts（MoE）模型通过门控函数将多个子模型的能力结合起来，在许多回归和分类应用中实现更好的性能。从理论上讲，虽然之前已经尝试通过高斯MoE模型中最大似然估计的收敛性分析来理解该模型在回归设置下的行为，但在分类问题的设置下缺乏相关分析。我们通过建立softmax gating multinomial logistic MoE模型的密度估计和参数估计的收敛速度来弥补这一空白。值得注意的是，当部分专家参数消失时，由于softmax gating和专家函数之间存在固有的偏微分方程互作用，这些收敛速度比多项式速度更慢。为了解决这个问题，我们提出使用一种新颖的修改softmax gating函数。

    Mixture-of-experts (MoE) model incorporates the power of multiple submodels via gating functions to achieve greater performance in numerous regression and classification applications. From a theoretical perspective, while there have been previous attempts to comprehend the behavior of that model under the regression settings through the convergence analysis of maximum likelihood estimation in the Gaussian MoE model, such analysis under the setting of a classification problem has remained missing in the literature. We close this gap by establishing the convergence rates of density estimation and parameter estimation in the softmax gating multinomial logistic MoE model. Notably, when part of the expert parameters vanish, these rates are shown to be slower than polynomial rates owing to an inherent interaction between the softmax gating and expert functions via partial differential equations. To address this issue, we propose using a novel class of modified softmax gating functions which 
    
[^13]: 在多臂赌博机中以固定置信度进行最优臂识别

    Optimal Best Arm Identification with Fixed Confidence in Restless Bandits. (arXiv:2310.13393v1 [stat.ML])

    [http://arxiv.org/abs/2310.13393](http://arxiv.org/abs/2310.13393)

    在多臂赌博机中，我们研究了以固定置信度进行最优臂识别。我们提出了一种新的策略来识别具有最大平均值的臂，同时限制了决策错误的概率，并建立了停止时间增长率的下界。

    

    我们研究了在具有有限数目臂的多臂赌博机中以不断变化的形式进行最优臂识别。每个臂产生的离散时间数据形成了一个取值在共同、有限状态空间中的同质马尔可夫链。每个臂的状态转移由一个遵循单参数指数族的转移概率矩阵（TPM）捕获。每个臂的TPM的实值参数是未知的，属于给定空间。给定在臂的共同状态空间上定义的函数f，目标是在样本数最少的情况下识别最优臂，即在该臂的稳态分布下评估f的平均值最大的臂，同时满足对决策错误概率（即固定置信度区间）的上界。在渐进性的误差概率趋于零的情况下，建立了期望停止时间增长率的下界。此外，我们提出了一种最优臂识别策略。

    We study best arm identification in a restless multi-armed bandit setting with finitely many arms. The discrete-time data generated by each arm forms a homogeneous Markov chain taking values in a common, finite state space. The state transitions in each arm are captured by an ergodic transition probability matrix (TPM) that is a member of a single-parameter exponential family of TPMs. The real-valued parameters of the arm TPMs are unknown and belong to a given space. Given a function $f$ defined on the common state space of the arms, the goal is to identify the best arm -- the arm with the largest average value of $f$ evaluated under the arm's stationary distribution -- with the fewest number of samples, subject to an upper bound on the decision's error probability (i.e., the fixed-confidence regime). A lower bound on the growth rate of the expected stopping time is established in the asymptote of a vanishing error probability. Furthermore, a policy for best arm identification is propo
    
[^14]: 从时间序列数据中发现混合结构因果模型

    Discovering Mixtures of Structural Causal Models from Time Series Data. (arXiv:2310.06312v1 [cs.LG])

    [http://arxiv.org/abs/2310.06312](http://arxiv.org/abs/2310.06312)

    这项研究提出了一种从时间序列数据中发现混合结构因果模型的方法，通过推断潜在因果模型以及每个样本属于特定混合成分的概率，通过实验证明了该方法在因果发现任务中的优越性。

    

    在金融、气候科学和神经科学等领域，从时间序列数据中推断因果关系是一个巨大的挑战。尽管现代技术可以处理变量之间的非线性关系和灵活的噪声分布，但它们依赖于简化假设，即数据来自相同的潜在因果模型。在这项工作中，我们放松了这个假设，从来源于不同因果模型混合的时间序列数据中进行因果发现。我们推断了潜在的结构性因果模型，以及每个样本属于特定混合成分的后验概率。我们的方法采用了一个端对端的训练过程，最大化了数据似然的证据下界。通过对合成和真实世界数据集的广泛实验，我们证明了我们的方法在因果发现任务中超越了最先进的基准方法，尤其是当数据来自不同的潜在因果模型时。

    In fields such as finance, climate science, and neuroscience, inferring causal relationships from time series data poses a formidable challenge. While contemporary techniques can handle nonlinear relationships between variables and flexible noise distributions, they rely on the simplifying assumption that data originates from the same underlying causal model. In this work, we relax this assumption and perform causal discovery from time series data originating from mixtures of different causal models. We infer both the underlying structural causal models and the posterior probability for each sample belonging to a specific mixture component. Our approach employs an end-to-end training process that maximizes an evidence-lower bound for data likelihood. Through extensive experimentation on both synthetic and real-world datasets, we demonstrate that our method surpasses state-of-the-art benchmarks in causal discovery tasks, particularly when the data emanates from diverse underlying causal
    
[^15]: 通过保持排序的干预分布实现因果公平的机器学习

    Causal Fair Machine Learning via Rank-Preserving Interventional Distributions. (arXiv:2307.12797v1 [cs.LG])

    [http://arxiv.org/abs/2307.12797](http://arxiv.org/abs/2307.12797)

    通过保持排序的干预分布，我们提出了一种因果公平的机器学习方法，通过在一个理想的世界中消除受保护属性对目标的因果影响来减少不公平。

    

    如果相同的个体得到相同的对待，而不同的个体得到不同的对待，那么一个决策被定义为公平的。根据这个定义，在设计机器学习模型以减少自动决策系统中的不公平时，必须引入因果思考来引入受保护属性。根据最近的提议，我们将个体定义为在一个假设的、理想的（FiND）世界中是规范上相等的，这个世界中受保护属性对目标没有（直接或间接）的因果影响。我们提出保持排序的干预分布来定义这个FiND世界的估计目标，并提出了一个估计方法。通过模拟和实证数据的验证，我们提供了对方法和生成模型的评价标准。通过这些，我们展示了我们的干预方法有效地识别出最受歧视的个体并减少不公平。

    A decision can be defined as fair if equal individuals are treated equally and unequals unequally. Adopting this definition, the task of designing machine learning models that mitigate unfairness in automated decision-making systems must include causal thinking when introducing protected attributes. Following a recent proposal, we define individuals as being normatively equal if they are equal in a fictitious, normatively desired (FiND) world, where the protected attribute has no (direct or indirect) causal effect on the target. We propose rank-preserving interventional distributions to define an estimand of this FiND world and a warping method for estimation. Evaluation criteria for both the method and resulting model are presented and validated through simulations and empirical data. With this, we show that our warping approach effectively identifies the most discriminated individuals and mitigates unfairness.
    
[^16]: 《鼠类与配偶：单一模型自动对群体中的老鼠行为进行分类和建模跨笼》

    Of Mice and Mates: Automated Classification and Modelling of Mouse Behaviour in Groups using a Single Model across Cages. (arXiv:2306.03066v1 [cs.CV])

    [http://arxiv.org/abs/2306.03066](http://arxiv.org/abs/2306.03066)

    本文提供了自动分类和模拟群体老鼠行为的工具，通过单一模型跨笼使用置换矩阵匹配老鼠身份，在家鼠环境下研究老鼠可以捕捉到个体行为的时间因素，而且无需人为干预。

    

    行为实验通常在专门的竞技场中进行，但这可能会混淆分析。为了解决这个问题，我们提供了工具来研究家鼠环境中的老鼠，为生物学家提供了捕捉个体行为的时间因素和模拟最小人为干预下笼友之间互动和相互依赖的可能性。我们开发了“活动标签模块”（ALM）来自动对老鼠行为进行分类，并开发了一种新的“群体行为模型”（GBM）来概括他们在笼子中的联合行为，使用置换矩阵将每个笼子中的老鼠身份与模型匹配。我们还发布了两个数据集，用于训练行为分类器（ABODe）和行为建模（IMADGE）。

    Behavioural experiments often happen in specialised arenas, but this may confound the analysis. To address this issue, we provide tools to study mice in the homecage environment, equipping biologists with the possibility to capture the temporal aspect of the individual's behaviour and model the interaction and interdependence between cage-mates with minimal human intervention. We develop the Activity Labelling Module (ALM) to automatically classify mouse behaviour from video, and a novel Group Behaviour Model (GBM) for summarising their joint behaviour across cages, using a permutation matrix to match the mouse identities in each cage to the model. We also release two datasets, ABODe for training behaviour classifiers and IMADGE for modelling behaviour.
    
[^17]: BBOB上高维贝叶斯优化算法的比较

    Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB. (arXiv:2303.00890v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.00890](http://arxiv.org/abs/2303.00890)

    本研究比较了BBOB上五种高维贝叶斯优化算法与传统方法以及CMA-ES算法的性能，结果表明... (根据论文的具体内容进行总结)

    

    贝叶斯优化是一类基于黑盒、基于代理的启发式算法，可以有效地优化评估成本高、只能拥有有限的评估预算的问题。贝叶斯优化在解决工业界的数值优化问题中尤为受欢迎，因为目标函数的评估通常依赖耗时的模拟或物理实验。然而，许多工业问题涉及大量参数，这给贝叶斯优化算法带来了挑战，其性能在维度超过15个变量时常常下降。虽然已经提出了许多新算法来解决这个问题，但目前还不清楚哪种算法在哪种优化场景中表现最好。本研究比较了5种最新的高维贝叶斯优化算法与传统贝叶斯优化和CMA-ES算法在COCA环境下24个BBOB函数上的性能，在维度从10到60个变量不断增加的情况下进行了对比。我们的结果证实了...

    Bayesian Optimization (BO) is a class of black-box, surrogate-based heuristics that can efficiently optimize problems that are expensive to evaluate, and hence admit only small evaluation budgets. BO is particularly popular for solving numerical optimization problems in industry, where the evaluation of objective functions often relies on time-consuming simulations or physical experiments. However, many industrial problems depend on a large number of parameters. This poses a challenge for BO algorithms, whose performance is often reported to suffer when the dimension grows beyond 15 variables. Although many new algorithms have been proposed to address this problem, it is not well understood which one is the best for which optimization scenario.  In this work, we compare five state-of-the-art high-dimensional BO algorithms, with vanilla BO and CMA-ES on the 24 BBOB functions of the COCO environment at increasing dimensionality, ranging from 10 to 60 variables. Our results confirm the su
    
[^18]: 面向Cox模型的高效通信式分布式估计和推断

    Communication-Efficient Distributed Estimation and Inference for Cox's Model. (arXiv:2302.12111v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2302.12111](http://arxiv.org/abs/2302.12111)

    我们提出了一种高效的分布式算法，用于在高维稀疏Cox比例风险模型中估计和推断，通过引入一种新的去偏差方法，我们可以产生渐近有效的分布式置信区间，并提供了有效的分布式假设检验。

    

    针对因隐私和所有权问题无法共享个体数据的多中心生物医学研究，我们开发了高维稀疏Cox比例风险模型的通信高效迭代分布式算法用于估计和推断。我们证明了即使进行了相对较少的迭代，我们的估计值在非常温和的条件下可以达到与理想全样本估计值相同的收敛速度。为了构建高维危险回归系数的线性组合的置信区间，我们引入了一种新的去偏差方法，建立了中心极限定理，并提供了一致的方差估计，可以产生渐近有效的分布式置信区间。此外，我们提供了基于装饰分数检验的任意坐标元素的有效和强大的分布式假设检验。我们还允许时间依赖协变量以及被审查的生存时间。在多种数据集上进行了广泛的数字实验，证明了算法的有效性和效率。

    Motivated by multi-center biomedical studies that cannot share individual data due to privacy and ownership concerns, we develop communication-efficient iterative distributed algorithms for estimation and inference in the high-dimensional sparse Cox proportional hazards model. We demonstrate that our estimator, even with a relatively small number of iterations, achieves the same convergence rate as the ideal full-sample estimator under very mild conditions. To construct confidence intervals for linear combinations of high-dimensional hazard regression coefficients, we introduce a novel debiased method, establish central limit theorems, and provide consistent variance estimators that yield asymptotically valid distributed confidence intervals. In addition, we provide valid and powerful distributed hypothesis tests for any coordinate element based on a decorrelated score test. We allow time-dependent covariates as well as censored survival times. Extensive numerical experiments on both s
    

