# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Training Dynamics of Multi-Head Softmax Attention for In-Context Learning: Emergence, Convergence, and Optimality](https://arxiv.org/abs/2402.19442) | 研究了多头softmax注意力模型在上下文学习中的训练动态，证明了全局收敛性，并发现了“任务分配”现象，梯度流动分为热身、涌现和收敛三个阶段，最终证明了梯度流的最优性。 |
| [^2] | [When Your AI Deceives You: Challenges with Partial Observability of Human Evaluators in Reward Learning](https://arxiv.org/abs/2402.17747) | RLHF在考虑部分观察性时可能导致策略欺骗性地夸大性能或过度辩护行为，我们提出了数学条件来解决这些问题，并警告不要盲目应用RLHF在部分可观测情况下。 |
| [^3] | [Large Stepsize Gradient Descent for Logistic Loss: Non-Monotonicity of the Loss Improves Optimization Efficiency](https://arxiv.org/abs/2402.15926) | 该研究表明对于具有线性可分数据的逻辑回归问题，设置一个恒定但较大的步长，在初始震荡后可以实现较快的收敛，并且在一定步骤后可以达到加速的收敛速率，这种方法无需动量或变步长调度器。 |
| [^4] | [Estimating Unknown Population Sizes Using the Hypergeometric Distribution](https://arxiv.org/abs/2402.14220) | 提出了一种使用超几何似然解决估计离散分布挑战的新方法，即使存在严重的欠采样，也能实现，且在人口规模估计的准确性和学习能力方面优于其他方法。 |
| [^5] | [Asymptotics of Learning with Deep Structured (Random) Features](https://arxiv.org/abs/2402.13999) | 在高维情况下，我们提供了学习输出层测试误差的严格渐近特性，并对使用高斯彩虹神经网络进行学习的问题做出了重要贡献 |
| [^6] | [Efficient Low-Rank Matrix Estimation, Experimental Design, and Arm-Set-Dependent Low-Rank Bandits](https://arxiv.org/abs/2402.11156) | 提出一种新型低秩矩阵估计方法LowPopArt，通过最小化量B(Q)提供更紧密的恢复保证，同时提出了一种新颖的实验设计标准，以及两种适用于一般Arm集的低秩线性赌博算法。 |
| [^7] | [On Computationally Efficient Multi-Class Calibration](https://arxiv.org/abs/2402.07821) | 提出了一种在多类别预测问题中多样化的投影平滑校准概念，并且给出了多项式时间复杂度的重新校准算法，从而实现了计算效率和强大的预测保证之间的权衡。 |
| [^8] | [Stochastic Gradient Flow Dynamics of Test Risk and its Exact Solution for Weak Features](https://arxiv.org/abs/2402.07626) | 本研究通过路径积分方法探索了连续时间随机梯度流动力学中的测试风险，并在小学习率情况下给出了计算纯梯度流动和随机梯度流动的测试风险曲线之间差异的一般公式。通过应用于一个弱特征模型，我们分析了随机项对动力学的修正效果，并与离散时间随机梯度下降的模拟结果进行了比较，结果显示出一致性。 |
| [^9] | [Noise-Adaptive Confidence Sets for Linear Bandits and Application to Bayesian Optimization](https://arxiv.org/abs/2402.07341) | 这项研究提出了一种对线性强化学习领域中未知噪声水平的自适应置信区间，与已有方法相比，在维度较大时具有显著的改进。此外，针对有界奖励，还提出了一种方差自适应置信区间，具有更好的数值性能。 |
| [^10] | [Self-Correcting Self-Consuming Loops for Generative Model Training](https://arxiv.org/abs/2402.07087) | 本论文研究了使用合成数据进行生成模型训练时可能出现的自我消耗循环问题，并提出了一种通过引入理想的修正函数来稳定训练的方法。同时，我们还提出了自我修正函数来近似理想的修正函数，并通过实验证实了其有效性。 |
| [^11] | [Bandit Convex Optimisation](https://arxiv.org/abs/2402.06535) | 这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。 |
| [^12] | [How Uniform Random Weights Induce Non-uniform Bias: Typical Interpolating Neural Networks Generalize with Narrow Teachers](https://arxiv.org/abs/2402.06323) | 在插值神经网络中，均匀随机权重可以产生非均匀偏差，因此通常插值神经网络会与窄教师NN一样很好地泛化。 |
| [^13] | [A High Dimensional Model for Adversarial Training: Geometry and Trade-Offs](https://arxiv.org/abs/2402.05674) | 本论文研究了高维模型中的对抗训练，引入了一个可处理的数学模型，并给出了对抗性经验风险最小化器的充分统计的精确渐近描述。研究结果表明存在可以防御而不惩罚准确性的方向，揭示了防御非鲁棒特征的优势。 |
| [^14] | [On Provable Length and Compositional Generalization](https://arxiv.org/abs/2402.04875) | 本研究针对包括深度集合、变压器、状态空间模型和简单递归神经网络等多种架构，探索了可证明的长度和组合泛化，认为对于长度和组合泛化，不同架构需要不同程度的表示识别。 |
| [^15] | [Stereographic Spherical Sliced Wasserstein Distances](https://arxiv.org/abs/2402.02345) | 本文提出了一种快速且高度并行的用于比较球形测度的距离，使用了立体投影和广义Radon变换，称之为立体投影球面切片瓦瑟斯坦（S3W）距离。通过仔细处理立体投影引起的距离畸变，并进行了理论分析，证明了该方法在速度和效果上的优势。 |
| [^16] | [Prepare Non-classical Collective Spin State by Reinforcement Learning.](http://arxiv.org/abs/2401.16320) | 通过强化学习设计控制场的方案成功生成了非经典态，以应用于自旋压缩态的产生。该方法在保持压缩和纠缠的同时提供了不同的控制序列，并观察到控制脉冲密集应用可以提高结果的性能。 |
| [^17] | [Improving Antibody Humanness Prediction using Patent Data.](http://arxiv.org/abs/2401.14442) | 本研究利用专利数据提高了抗体人性预测的能力，通过多阶段、多损失的训练过程以及弱监督对比学习的方法，成功地预测了抗体序列的人性评分。 |
| [^18] | [Conformal Prediction Sets Improve Human Decision Making.](http://arxiv.org/abs/2401.13744) | 该研究表明，通过规范预测量化模型的不确定性，可以提高人类决策的准确性和效果，对人机协同决策具有实用价值。 |
| [^19] | [Demonstration-Regularized RL.](http://arxiv.org/abs/2310.17303) | 通过演示-正则化提高强化学习的采样效率，并找到最优策略的样本复杂度，该复杂度与专家演示数量成反比。 |
| [^20] | [The Fundamental Dilemma of Bayesian Active Meta-learning.](http://arxiv.org/abs/2310.14968) | 在贝叶斯主动元学习中，贪婪追求可转移知识可能会损害对可转移参数的估计，学习者面临任务识别和可转移知识获取之间的困境。 |
| [^21] | [Understanding deep neural networks through the lens of their non-linearity.](http://arxiv.org/abs/2310.11439) | 本文提出了一个理论上有效的解决方案，通过亲和度评分追踪深度神经网络中的非线性传播，尤其关注计算机视觉应用。实验证实了所提出方法的实用性和对广泛应用的潜力。 |
| [^22] | [Risk Assessment and Statistical Significance in the Age of Foundation Models.](http://arxiv.org/abs/2310.07132) | 本论文提出了一个分布框架，用于评估具有统计显著性的基础模型的风险。通过一种新的统计相对测试方法，该框架结合了一阶和二阶随机优势，并借鉴了计量经济学和数学金融中常用的平均风险模型。在给定指定度量量化的防护栏的情况下，我们还开发了一种基于风险意识的基础模型选择方法。受数学金融中的投资组合优化和选择理论的启发，我们为每个模型定义了一个"度量组合"，并根据这些组合的随机优势进行模型选择。 |
| [^23] | [Fisher-Rao distance and pullback SPD cone distances between multivariate normal distributions.](http://arxiv.org/abs/2307.10644) | 本研究提出了一种快速和鲁棒的方法来近似计算多元正态分布之间的Fisher-Rao距离，并引入了一类基于正态流形嵌入到高维对称正定锥子流形的距离。 |
| [^24] | [Quantum Machine Learning on Near-Term Quantum Devices: Current State of Supervised and Unsupervised Techniques for Real-World Applications.](http://arxiv.org/abs/2307.00908) | 近期量子设备上的量子机器学习应用中，我们着重研究了监督和无监督学习在现实世界场景的应用。我们探究了当前量子硬件上的QML实现的限制，并提出了克服这些限制的技术。与经典对应物相比较，这些QML实现的性能得到了评估。 |
| [^25] | [Efficient and Multiply Robust Risk Estimation under General Forms of Dataset Shift.](http://arxiv.org/abs/2306.16406) | 本文研究了在通用的数据集转移条件下，利用半参数效率理论，高效估计目标总体风险的问题。 |
| [^26] | [Federated Learning You May Communicate Less Often!.](http://arxiv.org/abs/2306.05862) | 本研究针对联邦学习设定，探讨了通信次数对泛化误差的影响，并建立了PAC-Bayes和率失真理论限制，这些限制对广泛的损失函数和学习算法适用。 |
| [^27] | [Interpretable Deep Clustering.](http://arxiv.org/abs/2306.04785) | 本文提出了一种可解释的深度学习框架，通过自我监督的方式从数据点中标识信息量丰富的特征，设计了一个模型和门矩阵来预测可解释的实例和聚类级别的聚类分配，并在合成和实际数据中验证了其可靠性和可解释性。 |
| [^28] | [Nonlinear Distributionally Robust Optimization.](http://arxiv.org/abs/2306.03202) | 本文提出一种新的非线性分布鲁棒优化算法，用于处理一类分布鲁棒优化问题，通过 Gateaux Derivative 处理一般风险度量。经过实验验证，该方法成功处理分布的非线性目标函数。 |
| [^29] | [Random Function Descent.](http://arxiv.org/abs/2305.01377) | 本文提出了随机函数下降(RFD)算法，可以在随机环境中计算出步长并且与贝叶斯优化中的梯度下降算法相同。在合成基准测试中，RFD算法比未调整的Adam方法表现更好，提出的heuristic扩展可与调整后的Adam方法相媲美。 |
| [^30] | [Auditing and Generating Synthetic Data with Controllable Trust Trade-offs.](http://arxiv.org/abs/2304.10819) | 本论文提出了一个审计框架，能够以全面的方式评估合成数据和AI模型的具体效果，包括偏见和歧视预防、对真实数据的忠实程度、效用、鲁棒性和隐私保护。在多个用例中，审计框架平衡了信任和效用之间的权衡。 |
| [^31] | [The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning.](http://arxiv.org/abs/2304.05366) | 本论文阐述了无免费午餐定理的监督学习中的限制，证明了归纳偏差可以提高学习算法的效果，并且展示了神经网络模型的偏好与现实世界的数据分布相关。 |
| [^32] | [PAC-Bayesian Soft Actor-Critic Learning.](http://arxiv.org/abs/2301.12776) | 本文提出了一种使用PAC-Bayesian bound作为Soft Actor-Critic (SAC)算法评论家训练目标的方法，以解决训练不稳定的问题，并通过评论家引导的随机搜索探索多个未来来提高在线学习性能。在多个经典控制和运动任务中，该算法具有样本效率和遗憾最小化方面的明显优势。 |
| [^33] | [Identifying Peer Influence in Therapeutic Communities.](http://arxiv.org/abs/2203.14223) | 本研究调查了治疗社区中的同伴影响或角色模型效应对于成功毕业的影响。通过分析三个治疗社区的观察数据，我们发现肯定的同伴交流对于居民在自己离开之前成功毕业与否有显著影响。 |
| [^34] | [The Bayesian Learning Rule.](http://arxiv.org/abs/2107.04562) | 许多机器学习算法都可以归结为贝叶斯学习规则，该规则通过利用自然梯度来逼近后验分布，从而得到广泛的算法应用。这一工作不仅统一了现有算法，还帮助我们设计新的算法。 |

# 详细

[^1]: 多头softmax注意力机制在上下文学习中的训练动态：涌现、收敛和最优性

    Training Dynamics of Multi-Head Softmax Attention for In-Context Learning: Emergence, Convergence, and Optimality

    [https://arxiv.org/abs/2402.19442](https://arxiv.org/abs/2402.19442)

    研究了多头softmax注意力模型在上下文学习中的训练动态，证明了全局收敛性，并发现了“任务分配”现象，梯度流动分为热身、涌现和收敛三个阶段，最终证明了梯度流的最优性。

    

    我们研究了用于上下文学习的多任务线性回归的多头softmax注意力模型的梯度流动力学。我们证明了在适当的初始化选择下，梯度流动的全局收敛性。此外，我们证明了在梯度流动动力学中出现了有趣的“任务分配”现象，每个注意力头都专注于解决多任务模型中的单个任务。具体而言，我们证明了梯度流动动力学可以分为三个阶段——热身阶段，在这个阶段损失减少速度较慢，注意力头逐渐倾向于各自的任务；涌现阶段，在这个阶段，每个头选择一个单独的任务，损失迅速减少；和收敛阶段，在这个阶段，注意力参数收敛到一个极限。此外，我们证明了梯度流在学习极限模型方面的最优性。

    arXiv:2402.19442v1 Announce Type: cross  Abstract: We study the dynamics of gradient flow for training a multi-head softmax attention model for in-context learning of multi-task linear regression. We establish the global convergence of gradient flow under suitable choices of initialization. In addition, we prove that an interesting "task allocation" phenomenon emerges during the gradient flow dynamics, where each attention head focuses on solving a single task of the multi-task model. Specifically, we prove that the gradient flow dynamics can be split into three phases -- a warm-up phase where the loss decreases rather slowly and the attention heads gradually build up their inclination towards individual tasks, an emergence phase where each head selects a single task and the loss rapidly decreases, and a convergence phase where the attention parameters converge to a limit. Furthermore, we prove the optimality of gradient flow in the sense that the limiting model learned by gradient flo
    
[^2]: 当你的AI欺骗你：在奖励学习中人类评估者部分可观测性的挑战

    When Your AI Deceives You: Challenges with Partial Observability of Human Evaluators in Reward Learning

    [https://arxiv.org/abs/2402.17747](https://arxiv.org/abs/2402.17747)

    RLHF在考虑部分观察性时可能导致策略欺骗性地夸大性能或过度辩护行为，我们提出了数学条件来解决这些问题，并警告不要盲目应用RLHF在部分可观测情况下。

    

    强化学习从人类反馈（RLHF）的过去分析假设人类完全观察到环境。当人类反馈仅基于部分观察时会发生什么？我们对两种失败情况进行了正式定义：欺骗和过度辩护。通过将人类建模为对轨迹信念的Boltzmann-理性，我们证明了RLHF保证会导致策略欺骗性地夸大其性能、为了留下印象而过度辩护或者两者兼而有之的条件。为了帮助解决这些问题，我们数学地刻画了环境部分可观测性如何转化为（缺乏）学到的回报函数中的模糊性。在某些情况下，考虑环境部分可观测性使得在理论上可能恢复回报函数和最优策略，而在其他情况下，存在不可减少的模糊性。我们警告不要盲目应用RLHF在部分可观测情况下。

    arXiv:2402.17747v1 Announce Type: cross  Abstract: Past analyses of reinforcement learning from human feedback (RLHF) assume that the human fully observes the environment. What happens when human feedback is based only on partial observations? We formally define two failure cases: deception and overjustification. Modeling the human as Boltzmann-rational w.r.t. a belief over trajectories, we prove conditions under which RLHF is guaranteed to result in policies that deceptively inflate their performance, overjustify their behavior to make an impression, or both. To help address these issues, we mathematically characterize how partial observability of the environment translates into (lack of) ambiguity in the learned return function. In some cases, accounting for partial observability makes it theoretically possible to recover the return function and thus the optimal policy, while in other cases, there is irreducible ambiguity. We caution against blindly applying RLHF in partially observa
    
[^3]: 逻辑回归的大步梯度下降：损失的非单调性提高了优化效率

    Large Stepsize Gradient Descent for Logistic Loss: Non-Monotonicity of the Loss Improves Optimization Efficiency

    [https://arxiv.org/abs/2402.15926](https://arxiv.org/abs/2402.15926)

    该研究表明对于具有线性可分数据的逻辑回归问题，设置一个恒定但较大的步长，在初始震荡后可以实现较快的收敛，并且在一定步骤后可以达到加速的收敛速率，这种方法无需动量或变步长调度器。

    

    我们考虑了梯度下降（GD）与具有线性可分数据的逻辑回归结合使用的恒定步长情况，其中恒定步长$\eta$非常大，以至于损失在初始阶段会震荡。我们展示了GD在$\mathcal{O}(\eta)$步内迅速退出这种初始震荡阶段，并在额外的$t$步之后实现了一个$\tilde{\mathcal{O}}(1 / (\eta t) )$的收敛速率。我们的结果意味着，给定$T$步的预算，使用积极的步长$\eta:= \Theta( T)$，无需使用任何动量或变步长调度器，GD可以实现一个$\tilde{\mathcal{O}}(1/T^2)$的加速损失。我们的证明技术多才多艺，还可以处理一般分类损失函数（其中需要指数尾部来实现$\tilde{\mathcal{O}}(1/T^2)$的加速）、神经切线核区域的非线性预测器，以及具有大步长的在线随机梯度下降（SGD）。

    arXiv:2402.15926v1 Announce Type: new  Abstract: We consider gradient descent (GD) with a constant stepsize applied to logistic regression with linearly separable data, where the constant stepsize $\eta$ is so large that the loss initially oscillates. We show that GD exits this initial oscillatory phase rapidly -- in $\mathcal{O}(\eta)$ steps -- and subsequently achieves an $\tilde{\mathcal{O}}(1 / (\eta t) )$ convergence rate after $t$ additional steps. Our results imply that, given a budget of $T$ steps, GD can achieve an accelerated loss of $\tilde{\mathcal{O}}(1/T^2)$ with an aggressive stepsize $\eta:= \Theta( T)$, without any use of momentum or variable stepsize schedulers. Our proof technique is versatile and also handles general classification loss functions (where exponential tails are needed for the $\tilde{\mathcal{O}}(1/T^2)$ acceleration), nonlinear predictors in the neural tangent kernel regime, and online stochastic gradient descent (SGD) with a large stepsize, under sui
    
[^4]: 使用超几何分布估计未知人口规模

    Estimating Unknown Population Sizes Using the Hypergeometric Distribution

    [https://arxiv.org/abs/2402.14220](https://arxiv.org/abs/2402.14220)

    提出了一种使用超几何似然解决估计离散分布挑战的新方法，即使存在严重的欠采样，也能实现，且在人口规模估计的准确性和学习能力方面优于其他方法。

    

    多元超几何分布描述从划分为多个类别的离散元素总体中进行无放回抽样。在文献中存在的一个空白中，我们解决了估计离散分布的挑战，当总体规模和其构成类别的大小均未知时。在这里，我们提出了一种使用超几何似然解决这一估计挑战的新方法，即使存在严重的欠采样也能实现。我们开发了我们的方法，以解释一个数据生成过程，其中地面真实值是有条件的连续潜变量混合分布，比如协同过滤，使用变分自动编码器框架。实证数据模拟表明，我们的方法在人口规模估计的准确性和学习能力方面均优于其他用于建模计数数据的似然函数。

    arXiv:2402.14220v1 Announce Type: new  Abstract: The multivariate hypergeometric distribution describes sampling without replacement from a discrete population of elements divided into multiple categories. Addressing a gap in the literature, we tackle the challenge of estimating discrete distributions when both the total population size and the sizes of its constituent categories are unknown. Here, we propose a novel solution using the hypergeometric likelihood to solve this estimation challenge, even in the presence of severe under-sampling. We develop our approach to account for a data generating process where the ground-truth is a mixture of distributions conditional on a continuous latent variable, such as with collaborative filtering, using the variational autoencoder framework. Empirical data simulation demonstrates that our method outperforms other likelihood functions used to model count data, both in terms of accuracy of population size estimate and in its ability to learn an 
    
[^5]: 深度结构化（随机）特征学习的渐近分析

    Asymptotics of Learning with Deep Structured (Random) Features

    [https://arxiv.org/abs/2402.13999](https://arxiv.org/abs/2402.13999)

    在高维情况下，我们提供了学习输出层测试误差的严格渐近特性，并对使用高斯彩虹神经网络进行学习的问题做出了重要贡献

    

    针对一大类特征映射，我们在输入维度、隐藏层宽度和训练样本数量成比例增长的高维极限下，提供了与学习输出层相关的测试误差的严格渐近特性刻画。这一特征以特征的总体协方差为基础。我们的工作部分受到使用高斯彩虹神经网络进行学习的问题的启发，即具有随机但结构化权重的深层非线性全连接网络，其按行的协方差进一步允许依赖于之前层的权重。对于这样的网络，我们还推导出了一个以权重矩阵为基础的特征协方差的闭合形式公式。我们进一步发现，在某些情况下，我们的结果能够捕捉通过梯度下降训练的具有有限宽度的深度神经网络学习到的特征映射。

    arXiv:2402.13999v1 Announce Type: cross  Abstract: For a large class of feature maps we provide a tight asymptotic characterisation of the test error associated with learning the readout layer, in the high-dimensional limit where the input dimension, hidden layer widths, and number of training samples are proportionally large. This characterization is formulated in terms of the population covariance of the features. Our work is partially motivated by the problem of learning with Gaussian rainbow neural networks, namely deep non-linear fully-connected networks with random but structured weights, whose row-wise covariances are further allowed to depend on the weights of previous layers. For such networks we also derive a closed-form formula for the feature covariance in terms of the weight matrices. We further find that in some cases our results can capture feature maps learned by deep, finite-width neural networks trained under gradient descent.
    
[^6]: 高效的低秩矩阵估计、实验设计和基于Arm集的低秩赌博机

    Efficient Low-Rank Matrix Estimation, Experimental Design, and Arm-Set-Dependent Low-Rank Bandits

    [https://arxiv.org/abs/2402.11156](https://arxiv.org/abs/2402.11156)

    提出一种新型低秩矩阵估计方法LowPopArt，通过最小化量B(Q)提供更紧密的恢复保证，同时提出了一种新颖的实验设计标准，以及两种适用于一般Arm集的低秩线性赌博算法。

    

    我们研究了低秩矩阵迹回归和相关的低秩矩阵赌博问题。假设可以访问协变量的分布，我们提出了一种名为LowPopArt的新型低秩矩阵估计方法，并提供了其依赖于一个新颖数量B(Q)的恢复保证，该数量表征了问题的难度，其中Q是测量分布的协方差矩阵。我们展示了我们的方法在几个问题中可以提供比经典的核范数惩罚最小二乘法（Koltchinskii等人，2011）更紧密的恢复保证。为了在从任意给定的测量集合A中进行有限测量的情况下执行高效估计，我们还提出了一种新颖的实验设计标准，该标准以计算效率最小化B(Q)。我们利用我们的新颖估计器和实验设计推导了两种适用于一般Arm集的低秩线性赌博算法，其享有改进的

    arXiv:2402.11156v1 Announce Type: cross  Abstract: We study low-rank matrix trace regression and the related problem of low-rank matrix bandits. Assuming access to the distribution of the covariates, we propose a novel low-rank matrix estimation method called LowPopArt and provide its recovery guarantee that depends on a novel quantity denoted by B(Q) that characterizes the hardness of the problem, where Q is the covariance matrix of the measurement distribution. We show that our method can provide tighter recovery guarantees than classical nuclear norm penalized least squares (Koltchinskii et al., 2011) in several problems. To perform efficient estimation with a limited number of measurements from an arbitrarily given measurement set A, we also propose a novel experimental design criterion that minimizes B(Q) with computational efficiency. We leverage our novel estimator and design of experiments to derive two low-rank linear bandit algorithms for general arm sets that enjoy improved 
    
[^7]: 论计算有效的多类别校准问题

    On Computationally Efficient Multi-Class Calibration

    [https://arxiv.org/abs/2402.07821](https://arxiv.org/abs/2402.07821)

    提出了一种在多类别预测问题中多样化的投影平滑校准概念，并且给出了多项式时间复杂度的重新校准算法，从而实现了计算效率和强大的预测保证之间的权衡。

    

    考虑一个多类别标记问题，其中标记可以在[1,k]范围内取值，而预测器预测的是标记的分布。在这项工作中，我们研究了以下基础问题：是否存在多类别校准的概念，可以给出对有意义的预测的强大保证，并且可以在多项式时间和样本复杂度下实现？先前的校准概念在计算效率和表达能力之间存在着权衡：它们要么在k的样本复杂度上呈指数级增长，要么需要求解计算难题，要么给出的保证相当弱。我们的主要贡献是提出了一种能够实现所有这些期望的校准概念：我们在多类别预测中制定了一个稳健的投影平滑校准概念，并给出了新的重新校准算法，以在这个定义下以多项式时间复杂度校准预测器。投影平滑校准为多类别预测提供了强大的保证。

    Consider a multi-class labelling problem, where the labels can take values in $[k]$, and a predictor predicts a distribution over the labels. In this work, we study the following foundational question: Are there notions of multi-class calibration that give strong guarantees of meaningful predictions and can be achieved in time and sample complexities polynomial in $k$? Prior notions of calibration exhibit a tradeoff between computational efficiency and expressivity: they either suffer from having sample complexity exponential in $k$, or needing to solve computationally intractable problems, or give rather weak guarantees.   Our main contribution is a notion of calibration that achieves all these desiderata: we formulate a robust notion of projected smooth calibration for multi-class predictions, and give new recalibration algorithms for efficiently calibrating predictors under this definition with complexity polynomial in $k$. Projected smooth calibration gives strong guarantees for al
    
[^8]: 随机梯度流动力学中的测试风险及其弱特征的精确解

    Stochastic Gradient Flow Dynamics of Test Risk and its Exact Solution for Weak Features

    [https://arxiv.org/abs/2402.07626](https://arxiv.org/abs/2402.07626)

    本研究通过路径积分方法探索了连续时间随机梯度流动力学中的测试风险，并在小学习率情况下给出了计算纯梯度流动和随机梯度流动的测试风险曲线之间差异的一般公式。通过应用于一个弱特征模型，我们分析了随机项对动力学的修正效果，并与离散时间随机梯度下降的模拟结果进行了比较，结果显示出一致性。

    

    本研究探讨了学习理论中连续时间随机梯度流动力学的测试风险。利用路径积分公式，在小学习率的情况下，提供了计算纯梯度流动和随机梯度流动的测试风险曲线之间差异的一般公式。我们将这一通用理论应用到一个简单的弱特征模型中，该模型展示了双峰现象，并明确计算了动力学中增加的随机项随时间和模型参数的修正。分析结果与离散时间随机梯度下降的模拟进行了比较，显示出良好的一致性。

    We investigate the test risk of continuous-time stochastic gradient flow dynamics in learning theory. Using a path integral formulation we provide, in the regime of a small learning rate, a general formula for computing the difference between test risk curves of pure gradient and stochastic gradient flows. We apply the general theory to a simple model of weak features, which displays the double descent phenomenon, and explicitly compute the corrections brought about by the added stochastic term in the dynamics, as a function of time and model parameters. The analytical results are compared to simulations of discrete-time stochastic gradient descent and show good agreement.
    
[^9]: 对线性强化学习领域的噪声自适应置信区间及其在贝叶斯优化中的应用

    Noise-Adaptive Confidence Sets for Linear Bandits and Application to Bayesian Optimization

    [https://arxiv.org/abs/2402.07341](https://arxiv.org/abs/2402.07341)

    这项研究提出了一种对线性强化学习领域中未知噪声水平的自适应置信区间，与已有方法相比，在维度较大时具有显著的改进。此外，针对有界奖励，还提出了一种方差自适应置信区间，具有更好的数值性能。

    

    在序贯决策中，适应未知噪声水平是一个非常重要但具有挑战性的问题，因为有效的探索通常需要对噪声水平有一定的了解，而噪声水平通常只能粗略地指定。我们在线性强化学习领域取得了显著进展，主要有两方面。首先，我们提出了一种新颖的置信区间，该置信区间在未知的亚高斯参数σ_*^2上是“半自适应”的，意味着（归一化的）置信宽度与√（dσ_*^2 + σ_0^2）成正比，其中d为维度，σ_0^2为指定的（已知）亚高斯参数，其值可能比σ_*^2大得多。相比于Abbasi-Yadkori等人（2011）的标准置信区间的√（dσ_0^2），这是一个显著的改进，特别是当d较大时。我们证明了这导致了线性强化学习中改进的后悔边界。其次，对于有界奖励，我们提出了一种新颖的方差自适应置信区间，具有更好的数值性能。

    Adapting to a priori unknown noise level is a very important but challenging problem in sequential decision-making as efficient exploration typically requires knowledge of the noise level, which is often loosely specified. We report significant progress in addressing this issue in linear bandits in two respects. First, we propose a novel confidence set that is `semi-adaptive' to the unknown sub-Gaussian parameter $\sigma_*^2$ in the sense that the (normalized) confidence width scales with $\sqrt{d\sigma_*^2 + \sigma_0^2}$ where $d$ is the dimension and $\sigma_0^2$ is the specified sub-Gaussian parameter (known) that can be much larger than $\sigma_*^2$. This is a significant improvement over $\sqrt{d\sigma_0^2}$ of the standard confidence set of Abbasi-Yadkori et al. (2011), especially when $d$ is large. We show that this leads to an improved regret bound in linear bandits. Second, for bounded rewards, we propose a novel variance-adaptive confidence set that has a much improved numeri
    
[^10]: 自我纠正自我消耗循环用于生成模型训练

    Self-Correcting Self-Consuming Loops for Generative Model Training

    [https://arxiv.org/abs/2402.07087](https://arxiv.org/abs/2402.07087)

    本论文研究了使用合成数据进行生成模型训练时可能出现的自我消耗循环问题，并提出了一种通过引入理想的修正函数来稳定训练的方法。同时，我们还提出了自我修正函数来近似理想的修正函数，并通过实验证实了其有效性。

    

    随着合成数据在互联网上的质量越来越高以及数量不断增加，机器学习模型越来越多地在人工和机器生成的数据的混合上进行训练。尽管使用合成数据进行表征学习的成功案例有很多，但是在生成模型训练中使用合成数据会产生"自我消耗循环"，这可能导致训练不稳定甚至崩溃，除非满足某些条件。我们的论文旨在稳定自我消耗的生成模型训练。我们的理论结果表明，通过引入一个理想的修正函数，将数据点映射为更有可能来自真实数据分布的样本，可以使自我消耗循环的稳定性呈指数增加。然后，我们提出了自我修正函数，它依赖于专家知识（例如，编程在模拟器中的物理定律），并且旨在自动且大规模地近似理想的修正函数。我们通过实验证实了自我纠正自我消耗循环在生成模型训练中的有效性。

    As synthetic data becomes higher quality and proliferates on the internet, machine learning models are increasingly trained on a mix of human- and machine-generated data. Despite the successful stories of using synthetic data for representation learning, using synthetic data for generative model training creates "self-consuming loops" which may lead to training instability or even collapse, unless certain conditions are met. Our paper aims to stabilize self-consuming generative model training. Our theoretical results demonstrate that by introducing an idealized correction function, which maps a data point to be more likely under the true data distribution, self-consuming loops can be made exponentially more stable. We then propose self-correction functions, which rely on expert knowledge (e.g. the laws of physics programmed in a simulator), and aim to approximate the idealized corrector automatically and at scale. We empirically validate the effectiveness of self-correcting self-consum
    
[^11]: Bandit Convex Optimisation（强盗凸优化）

    Bandit Convex Optimisation

    [https://arxiv.org/abs/2402.06535](https://arxiv.org/abs/2402.06535)

    这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。

    

    强盗凸优化是研究零阶凸优化的基本框架。本文介绍了用于解决该问题的许多工具，包括切平面方法、内点方法、连续指数权重、梯度下降和在线牛顿步骤。解释了许多假设和设置之间的细微差别。尽管在这里没有太多真正新的东西，但一些现有工具以新颖的方式应用于获得新算法。一些界限稍微改进了一些。

    Bandit convex optimisation is a fundamental framework for studying zeroth-order convex optimisation. These notes cover the many tools used for this problem, including cutting plane methods, interior point methods, continuous exponential weights, gradient descent and online Newton step. The nuances between the many assumptions and setups are explained. Although there is not much truly new here, some existing tools are applied in novel ways to obtain new algorithms. A few bounds are improved in minor ways.
    
[^12]: 均匀随机权重如何引起不均匀偏差：典型插值神经网络与窄教师的普遍性

    How Uniform Random Weights Induce Non-uniform Bias: Typical Interpolating Neural Networks Generalize with Narrow Teachers

    [https://arxiv.org/abs/2402.06323](https://arxiv.org/abs/2402.06323)

    在插值神经网络中，均匀随机权重可以产生非均匀偏差，因此通常插值神经网络会与窄教师NN一样很好地泛化。

    

    背景。一个主要的理论难题是当神经网络被训练到零误差（即插值数据）时，为什么超参数化神经网络（NN）能够很好地泛化。通常，NN是使用随机梯度下降（SGD）或其变种之一训练的。然而，最近的实证研究检验了从看似均匀的参数先验中采样的随机NN对数据的泛化能力：该NN对训练集进行了完美分类。有趣的是，这样的NN样本通常像SGD训练的NN一样泛化良好。贡献。我们证明了如果存在与标签一致的窄“教师NN”，那么这样的随机NN插值器通常能很好地泛化。具体而言，我们证明了在NN参数化中的“平坦”先验通过NN结构中的冗余引入了丰富的NN函数先验。特别是，这会对较简单的函数产生偏向，这些函数需要较少的相关参数。

    Background. A main theoretical puzzle is why over-parameterized Neural Networks (NNs) generalize well when trained to zero loss (i.e., so they interpolate the data). Usually, the NN is trained with Stochastic Gradient Descent (SGD) or one of its variants. However, recent empirical work examined the generalization of a random NN that interpolates the data: the NN was sampled from a seemingly uniform prior over the parameters, conditioned on that the NN perfectly classifying the training set. Interestingly, such a NN sample typically generalized as well as SGD-trained NNs.   Contributions. We prove that such a random NN interpolator typically generalizes well if there exists an underlying narrow ``teacher NN" that agrees with the labels. Specifically, we show that such a `flat' prior over the NN parametrization induces a rich prior over the NN functions, due to the redundancy in the NN structure. In particular, this creates a bias towards simpler functions, which require less relevant pa
    
[^13]: 高维模型的对抗训练：几何和权衡

    A High Dimensional Model for Adversarial Training: Geometry and Trade-Offs

    [https://arxiv.org/abs/2402.05674](https://arxiv.org/abs/2402.05674)

    本论文研究了高维模型中的对抗训练，引入了一个可处理的数学模型，并给出了对抗性经验风险最小化器的充分统计的精确渐近描述。研究结果表明存在可以防御而不惩罚准确性的方向，揭示了防御非鲁棒特征的优势。

    

    本研究在高维情况下，即维度$d$和数据点数$n$与固定比例$\alpha = n / d$发散的上下文中，研究了基于边际的线性分类器中的对抗训练。我们引入了一个可处理的数学模型，可以研究数据和对抗攻击者几何之间的相互作用，同时捕捉到对抗鲁棒性文献中观察到的核心现象。我们的主要理论贡献是在通用的凸且非递增损失函数下，对于对抗性经验风险最小化器的充分统计的精确渐近描述。我们的结果使我们能够精确地刻画数据中与更高的泛化/鲁棒性权衡相关的方向，由一个鲁棒性度量和一个有用性度量定义。特别地，我们揭示了存在一些方向，可以进行防御而不惩罚准确性。最后，我们展示了防御非鲁棒特征的优势。

    This work investigates adversarial training in the context of margin-based linear classifiers in the high-dimensional regime where the dimension $d$ and the number of data points $n$ diverge with a fixed ratio $\alpha = n / d$. We introduce a tractable mathematical model where the interplay between the data and adversarial attacker geometries can be studied, while capturing the core phenomenology observed in the adversarial robustness literature. Our main theoretical contribution is an exact asymptotic description of the sufficient statistics for the adversarial empirical risk minimiser, under generic convex and non-increasing losses. Our result allow us to precisely characterise which directions in the data are associated with a higher generalisation/robustness trade-off, as defined by a robustness and a usefulness metric. In particular, we unveil the existence of directions which can be defended without penalising accuracy. Finally, we show the advantage of defending non-robust featu
    
[^14]: 关于可证明的长度和组合泛化

    On Provable Length and Compositional Generalization

    [https://arxiv.org/abs/2402.04875](https://arxiv.org/abs/2402.04875)

    本研究针对包括深度集合、变压器、状态空间模型和简单递归神经网络等多种架构，探索了可证明的长度和组合泛化，认为对于长度和组合泛化，不同架构需要不同程度的表示识别。

    

    长度泛化——对训练时未见到的更长序列的泛化能力，以及组合泛化——对训练时未见到的令牌组合的泛化能力，在序列到序列模型中是重要的非分布化泛化形式。在这项工作中，我们在包括深度集合、变压器、状态空间模型和简单递归神经网络在内的一系列架构中，朝着可证明的长度和组合泛化迈出了第一步。根据架构的不同，我们证明了不同程度的表示识别的必要性，例如与真实表示具有线性或排列关系。

    Length generalization -- the ability to generalize to longer sequences than ones seen during training, and compositional generalization -- the ability to generalize to token combinations not seen during training, are crucial forms of out-of-distribution generalization in sequence-to-sequence models. In this work, we take the first steps towards provable length and compositional generalization for a range of architectures, including deep sets, transformers, state space models, and simple recurrent neural nets. Depending on the architecture, we prove different degrees of representation identification, e.g., a linear or a permutation relation with ground truth representation, is necessary for length and compositional generalization.
    
[^15]: Stereographic Spherical Sliced Wasserstein Distances - 应用于球形概率分布比较的立体投影球面切片瓦瑟斯坦距离

    Stereographic Spherical Sliced Wasserstein Distances

    [https://arxiv.org/abs/2402.02345](https://arxiv.org/abs/2402.02345)

    本文提出了一种快速且高度并行的用于比较球形测度的距离，使用了立体投影和广义Radon变换，称之为立体投影球面切片瓦瑟斯坦（S3W）距离。通过仔细处理立体投影引起的距离畸变，并进行了理论分析，证明了该方法在速度和效果上的优势。

    

    在地质学、医学领域、计算机视觉和深度表示学习等各个领域，比较球形概率分布是非常重要的。基于最优传输的距离，比如瓦瑟斯坦距离，对于比较概率测度已经引发了活跃的研究，以开发计算效率高的球形概率测度的变体。本文介绍了一种高速且高度并行化的用于比较球形测度的距离，使用了立体投影和广义Radon变换，我们称之为立体投影球面切片瓦瑟斯坦（S3W）距离。我们仔细处理了立体投影引起的距离畸变，并对我们提出的度量及其具有旋转不变性的变体进行了广泛的理论分析。最后，我们评估了所提出的度量的性能，并将其与最近的基线进行了比较，从遥感和处理效率两个方面进行了评估。

    Comparing spherical probability distributions is of great interest in various fields, including geology, medical domains, computer vision, and deep representation learning. The utility of optimal transport-based distances, such as the Wasserstein distance, for comparing probability measures has spurred active research in developing computationally efficient variations of these distances for spherical probability measures. This paper introduces a high-speed and highly parallelizable distance for comparing spherical measures using the stereographic projection and the generalized Radon transform, which we refer to as the Stereographic Spherical Sliced Wasserstein (S3W) distance. We carefully address the distance distortion caused by the stereographic projection and provide an extensive theoretical analysis of our proposed metric and its rotationally invariant variation. Finally, we evaluate the performance of the proposed metrics and compare them with recent baselines in terms of both spe
    
[^16]: 利用强化学习生成非经典集合自旋态的方案

    Prepare Non-classical Collective Spin State by Reinforcement Learning. (arXiv:2401.16320v1 [quant-ph])

    [http://arxiv.org/abs/2401.16320](http://arxiv.org/abs/2401.16320)

    通过强化学习设计控制场的方案成功生成了非经典态，以应用于自旋压缩态的产生。该方法在保持压缩和纠缠的同时提供了不同的控制序列，并观察到控制脉冲密集应用可以提高结果的性能。

    

    我们提出了一种利用强化学习来设计控制场的方案，用于生成非经典态。该方案以应用于开放集体自旋模型中的自旋压缩态为例，其中设计了一个线性控制项来控制动力学。强化学习代理根据以耗散和去相干为特征的环境中的相干自旋态开始，确定了控制脉冲的时间序列。与恒定控制方案相比，这种方法提供了多种控制序列，保持了集体自旋压缩和纠缠。观察到控制脉冲的密集应用可以增强结果的性能。此外，通过添加控制操作，性能得到了轻微增强。所提出的策略在较大系统中展现了更高的效果。对储备热激发对控制结果有不利影响。应该确认这一点。

    We propose a scheme leveraging reinforcement learning to engineer control fields for generating non-classical states. It is exemplified by the application to prepare spin squeezed state for an open collective spin model where a linear control term is designed to govern the dynamics. The reinforcement learning agent determines the temporal sequence of control pulses, commencing from coherent spin state in an environment characterized by dissipation and dephasing. When compared to constant control scenarios, this approach provides various control sequences maintaining collective spin squeezing and entanglement. It is observed that denser application of the control pulses enhances the performance of the outcomes. Furthermore, there is a minor enhancement in the performance by adding control actions. The proposed strategy demonstrates increased effectiveness for larger systems. And thermal excitations of the reservoir are detrimental to the control outcomes. It should be confirmed that thi
    
[^17]: 利用专利数据提高抗体人性预测能力

    Improving Antibody Humanness Prediction using Patent Data. (arXiv:2401.14442v1 [q-bio.QM])

    [http://arxiv.org/abs/2401.14442](http://arxiv.org/abs/2401.14442)

    本研究利用专利数据提高了抗体人性预测的能力，通过多阶段、多损失的训练过程以及弱监督对比学习的方法，成功地预测了抗体序列的人性评分。

    

    我们研究了利用专利数据来提高抗体人性预测的潜力，采用了多阶段、多损失的训练过程。抗体人性作为对抗体治疗的免疫反应的代理，是药物发现中的主要原因之一，在临床环境中使用抗体治疗面临着具有挑战性的障碍。我们将初始学习阶段视为一个弱监督对比学习问题，每个抗体序列与可能有多个功能标识符相关联，目标是学习一个编码器，根据其专利属性将它们分组。然后，我们冻结对比编码器的一部分，并继续使用交叉熵损失在专利数据上训练，以预测给定抗体序列的人性评分。我们通过对三个不同的免疫原性数据集进行推理，展示了专利数据和我们的方法的效用。我们的实证结果表明，l

    We investigate the potential of patent data for improving the antibody humanness prediction using a multi-stage, multi-loss training process. Humanness serves as a proxy for the immunogenic response to antibody therapeutics, one of the major causes of attrition in drug discovery and a challenging obstacle for their use in clinical settings. We pose the initial learning stage as a weakly-supervised contrastive-learning problem, where each antibody sequence is associated with possibly multiple identifiers of function and the objective is to learn an encoder that groups them according to their patented properties. We then freeze a part of the contrastive encoder and continue training it on the patent data using the cross-entropy loss to predict the humanness score of a given antibody sequence. We illustrate the utility of the patent data and our approach by performing inference on three different immunogenicity datasets, unseen during training. Our empirical results demonstrate that the l
    
[^18]: 《规范预测集提升人类决策能力》

    Conformal Prediction Sets Improve Human Decision Making. (arXiv:2401.13744v1 [cs.LG])

    [http://arxiv.org/abs/2401.13744](http://arxiv.org/abs/2401.13744)

    该研究表明，通过规范预测量化模型的不确定性，可以提高人类决策的准确性和效果，对人机协同决策具有实用价值。

    

    作为对日常查询的回应，人类明确地表达不确定性，并在不确定的情况下提供替代答案。通过规范预测输出校准的预测集，模仿了人类的这种行为；更大的预测集表示更大的不确定性，同时提供了替代方案。在这项工作中，我们通过实施预注册的随机对照试验，并给人类受试者提供规范预测集，研究了规范预测集对人类决策的实用性。通过统计学显著性，我们发现当人类获得规范预测集时，他们在任务上的准确性比使用相同覆盖保证的固定尺寸预测集时有所提高。结果表明，用规范预测量化模型的不确定性有助于人机协同决策和人工智能团队的决策。

    In response to everyday queries, humans explicitly signal uncertainty and offer alternative answers when they are unsure. Machine learning models that output calibrated prediction sets through conformal prediction mimic this human behaviour; larger sets signal greater uncertainty while providing alternatives. In this work, we study the usefulness of conformal prediction sets as an aid for human decision making by conducting a pre-registered randomized controlled trial with conformal prediction sets provided to human subjects. With statistical significance, we find that when humans are given conformal prediction sets their accuracy on tasks improves compared to fixed-size prediction sets with the same coverage guarantee. The results show that quantifying model uncertainty with conformal prediction is helpful for human-in-the-loop decision making and human-AI teams.
    
[^19]: 通过演示-正则化强化学习增强采样效率

    Demonstration-Regularized RL. (arXiv:2310.17303v1 [stat.ML])

    [http://arxiv.org/abs/2310.17303](http://arxiv.org/abs/2310.17303)

    通过演示-正则化提高强化学习的采样效率，并找到最优策略的样本复杂度，该复杂度与专家演示数量成反比。

    

    通过将专家演示纳入其中，可以在提高强化学习(SRL)的采样效率方面产生经验效果。本文在理论上量化这些额外信息降低了SRL的采样复杂性的程度。具体而言，我们研究了通过KL正则化利用专家演示学习的策略的演示-正则化强化学习。我们的研究发现，在有限状态下，在$\widetilde{\mathcal{O}}(\mathrm{Poly}(S,A,H)/(\varepsilon^2 N^{\mathrm{E}}))$的样本复杂度内，使用$N^{\mathrm{E}}$个专家演示能够找到最优策略，并在线性马尔科夫决策过程中，在$\widetilde{\mathcal{O}}(\mathrm{Poly}(d,H)/(\varepsilon^2 N^{\mathrm{E}}))$的样本复杂度内找到最优策略，其中$\varepsilon$是目标精度，$H$是规定，$A$是动作的数量，$S$是有限状态的数量，在线性情况下，$d$是特征空间的维数。

    Incorporating expert demonstrations has empirically helped to improve the sample efficiency of reinforcement learning (RL). This paper quantifies theoretically to what extent this extra information reduces RL's sample complexity. In particular, we study the demonstration-regularized reinforcement learning that leverages the expert demonstrations by KL-regularization for a policy learned by behavior cloning. Our findings reveal that using $N^{\mathrm{E}}$ expert demonstrations enables the identification of an optimal policy at a sample complexity of order $\widetilde{\mathcal{O}}(\mathrm{Poly}(S,A,H)/(\varepsilon^2 N^{\mathrm{E}}))$ in finite and $\widetilde{\mathcal{O}}(\mathrm{Poly}(d,H)/(\varepsilon^2 N^{\mathrm{E}}))$ in linear Markov decision processes, where $\varepsilon$ is the target precision, $H$ the horizon, $A$ the number of action, $S$ the number of states in the finite case and $d$ the dimension of the feature space in the linear case. As a by-product, we provide tight con
    
[^20]: 贝叶斯主动元学习的基本困境

    The Fundamental Dilemma of Bayesian Active Meta-learning. (arXiv:2310.14968v1 [cs.LG])

    [http://arxiv.org/abs/2310.14968](http://arxiv.org/abs/2310.14968)

    在贝叶斯主动元学习中，贪婪追求可转移知识可能会损害对可转移参数的估计，学习者面临任务识别和可转移知识获取之间的困境。

    

    许多应用需要估计在多个不同但相关的数据稀缺任务环境中推广的参数。贝叶斯主动元学习是一种顺序最优实验设计的形式，为解决这类问题提供了一个框架。主动元学习者的目标是在当前任务的特殊特征（任务特定参数）的情况下获得可转移的知识（估计可转移的参数）。我们证明，在这种情况下，贪婪追求这个目标实际上可能会损害对可转移参数的估计（引起所谓的负迁移）。学习者面临着一个类似但不同于勘探-利用困境的困境：他们应该花费他们的获取预算来追求可转移的知识，还是用来确定当前任务特定的参数？我们理论上证明，一些任务存在不可避免且任意大的负迁移威胁，任务的识别对于重新寻找可迁移参数至关重要。

    Many applications involve estimation of parameters that generalize across multiple diverse, but related, data-scarce task environments. Bayesian active meta-learning, a form of sequential optimal experimental design, provides a framework for solving such problems. The active meta-learner's goal is to gain transferable knowledge (estimate the transferable parameters) in the presence of idiosyncratic characteristics of the current task (task-specific parameters). We show that in such a setting, greedy pursuit of this goal can actually hurt estimation of the transferable parameters (induce so-called negative transfer). The learner faces a dilemma akin to but distinct from the exploration--exploitation dilemma: should they spend their acquisition budget pursuing transferable knowledge, or identifying the current task-specific parameters? We show theoretically that some tasks pose an inevitable and arbitrarily large threat of negative transfer, and that task identification is critical to re
    
[^21]: 通过非线性研究深度神经网络的理解

    Understanding deep neural networks through the lens of their non-linearity. (arXiv:2310.11439v1 [cs.LG])

    [http://arxiv.org/abs/2310.11439](http://arxiv.org/abs/2310.11439)

    本文提出了一个理论上有效的解决方案，通过亲和度评分追踪深度神经网络中的非线性传播，尤其关注计算机视觉应用。实验证实了所提出方法的实用性和对广泛应用的潜力。

    

    深度神经网络(DNN)的显著成功常常归因于它们的高表达能力和近似任意复杂函数的能力。事实上，DNN是高度非线性的模型，其中引入的激活函数在其中起到了重要作用。然而，尽管许多研究通过近似能力的视角研究了DNN的表达能力，但量化DNN或个别激活函数的非线性仍然是一个开放性问题。在本文中，我们提出了第一个在具体关注计算机视觉应用中追踪非线性传播的理论有效解决方案。我们提出的亲和度评分允许我们深入了解各种不同体系结构和学习范式的内部工作原理。我们提供了大量的实验结果，突出了所提出的亲和度评分的实际效用和潜在应用的可能性。

    The remarkable success of deep neural networks (DNN) is often attributed to their high expressive power and their ability to approximate functions of arbitrary complexity. Indeed, DNNs are highly non-linear models, and activation functions introduced into them are largely responsible for this. While many works studied the expressive power of DNNs through the lens of their approximation capabilities, quantifying the non-linearity of DNNs or of individual activation functions remains an open problem. In this paper, we propose the first theoretically sound solution to track non-linearity propagation in deep neural networks with a specific focus on computer vision applications. Our proposed affinity score allows us to gain insights into the inner workings of a wide range of different architectures and learning paradigms. We provide extensive experimental results that highlight the practical utility of the proposed affinity score and its potential for long-reaching applications.
    
[^22]: 在基础模型时代的风险评估和统计显著性

    Risk Assessment and Statistical Significance in the Age of Foundation Models. (arXiv:2310.07132v1 [cs.LG])

    [http://arxiv.org/abs/2310.07132](http://arxiv.org/abs/2310.07132)

    本论文提出了一个分布框架，用于评估具有统计显著性的基础模型的风险。通过一种新的统计相对测试方法，该框架结合了一阶和二阶随机优势，并借鉴了计量经济学和数学金融中常用的平均风险模型。在给定指定度量量化的防护栏的情况下，我们还开发了一种基于风险意识的基础模型选择方法。受数学金融中的投资组合优化和选择理论的启发，我们为每个模型定义了一个"度量组合"，并根据这些组合的随机优势进行模型选择。

    

    我们提出了一个分布框架，用于评估具有统计显著性的基础模型的社会技术风险。我们的方法依赖于一种基于实际随机变量的一阶和二阶随机优势的新的统计相对测试。我们表明，这个测试中的二阶统计与在计量经济学和数学金融中常用的平均风险模型相联系，用于在选择方案时平衡风险和效用。利用这个框架，我们正式开发了一种基于风险意识的基础模型选择方法，给定由指定度量量化的防护栏。受数学金融中的投资组合优化和选择理论的启发，我们为每个模型定义了一个"度量组合"，作为聚合一系列度量的手段，并根据这些组合的随机优势进行模型选择。我们的测试的统计显著性在理论上由通过中心极限的渐近分析支持。

    We propose a distributional framework for assessing socio-technical risks of foundation models with quantified statistical significance. Our approach hinges on a new statistical relative testing based on first and second order stochastic dominance of real random variables. We show that the second order statistics in this test are linked to mean-risk models commonly used in econometrics and mathematical finance to balance risk and utility when choosing between alternatives. Using this framework, we formally develop a risk-aware approach for foundation model selection given guardrails quantified by specified metrics. Inspired by portfolio optimization and selection theory in mathematical finance, we define a \emph{metrics portfolio} for each model as a means to aggregate a collection of metrics, and perform model selection based on the stochastic dominance of these portfolios. The statistical significance of our tests is backed theoretically by an asymptotic analysis via central limit th
    
[^23]: Fisher-Rao距离和逆推到SPD锥距离在多元正态分布之间的应用

    Fisher-Rao distance and pullback SPD cone distances between multivariate normal distributions. (arXiv:2307.10644v1 [cs.LG])

    [http://arxiv.org/abs/2307.10644](http://arxiv.org/abs/2307.10644)

    本研究提出了一种快速和鲁棒的方法来近似计算多元正态分布之间的Fisher-Rao距离，并引入了一类基于正态流形嵌入到高维对称正定锥子流形的距离。

    

    许多科学领域，如扩散张量成像、结构张量计算机视觉、雷达信号处理和机器学习等，都存在着多元正态分布的数据集。为了处理这些正态数据集以进行过滤、分类或聚类等下游任务，需要定义合适的正态和它们之间的路径之间的差异度量。Fisher-Rao距离，作为Fisher信息度量引起的Riemann几何距离，是一种合理的度量距离，但除了一些特殊情况外，并没有闭式求解。本文首先报告了一种快速且鲁棒的方法，可以精确地近似计算多元正态分布之间的Fisher-Rao距离。其次，我们介绍了一类基于正态流形到高维对称正定锥的子流形的微分同胚嵌入的距离。

    Data sets of multivariate normal distributions abound in many scientific areas like diffusion tensor imaging, structure tensor computer vision, radar signal processing, machine learning, just to name a few. In order to process those normal data sets for downstream tasks like filtering, classification or clustering, one needs to define proper notions of dissimilarities between normals and paths joining them. The Fisher-Rao distance defined as the Riemannian geodesic distance induced by the Fisher information metric is such a principled metric distance which however is not known in closed-form excepts for a few particular cases. In this work, we first report a fast and robust method to approximate arbitrarily finely the Fisher-Rao distance between multivariate normal distributions. Second, we introduce a class of distances based on diffeomorphic embeddings of the normal manifold into a submanifold of the higher-dimensional symmetric positive-definite cone corresponding to the manifold of
    
[^24]: 近期量子装置上的量子机器学习: 监督和无监督技术在现实世界应用的现状

    Quantum Machine Learning on Near-Term Quantum Devices: Current State of Supervised and Unsupervised Techniques for Real-World Applications. (arXiv:2307.00908v1 [quant-ph])

    [http://arxiv.org/abs/2307.00908](http://arxiv.org/abs/2307.00908)

    近期量子设备上的量子机器学习应用中，我们着重研究了监督和无监督学习在现实世界场景的应用。我们探究了当前量子硬件上的QML实现的限制，并提出了克服这些限制的技术。与经典对应物相比较，这些QML实现的性能得到了评估。

    

    在过去十年中，量子硬件在速度、量子比特数量和量子体积方面取得了相当大的进展，量子体积被定义为在近期量子设备上可以有效实现的量子电路的最大规模。因此，在实际硬件上应用量子机器学习(QML)以实现量子优势已经有了很大的增长。在这篇综述中，我们主要关注在量子硬件上实现的选定监督和无监督学习应用，特别针对现实世界场景。我们探讨并强调了QML在量子硬件上的当前限制。我们深入讨论了各种克服这些限制的技术，如编码技术、基态结构、误差补偿和梯度方法。此外，我们评估了这些QML实现与它们的经典对应物之间的性能对比。

    The past decade has seen considerable progress in quantum hardware in terms of the speed, number of qubits and quantum volume which is defined as the maximum size of a quantum circuit that can be effectively implemented on a near-term quantum device. Consequently, there has also been a rise in the number of works based on the applications of Quantum Machine Learning (QML) on real hardware to attain quantum advantage over their classical counterparts. In this survey, our primary focus is on selected supervised and unsupervised learning applications implemented on quantum hardware, specifically targeting real-world scenarios. Our survey explores and highlights the current limitations of QML implementations on quantum hardware. We delve into various techniques to overcome these limitations, such as encoding techniques, ansatz structure, error mitigation, and gradient methods. Additionally, we assess the performance of these QML implementations in comparison to their classical counterparts
    
[^25]: 通用形式下的高效且多重稳健的风险估计方法在数据转移中

    Efficient and Multiply Robust Risk Estimation under General Forms of Dataset Shift. (arXiv:2306.16406v1 [stat.ME])

    [http://arxiv.org/abs/2306.16406](http://arxiv.org/abs/2306.16406)

    本文研究了在通用的数据集转移条件下，利用半参数效率理论，高效估计目标总体风险的问题。

    

    统计机器学习方法经常面临来自感兴趣总体的有限数据的挑战。一种解决方法是利用来自辅助源总体的数据，这些数据与目标领域的某些条件分布相同或以其他方式相连。利用这种"数据转移"条件的技术被称为"领域适应"或"迁移学习"。尽管有大量关于数据转移的文献，但很少有研究探讨如何有效利用辅助总体来提高目标总体上机器学习任务风险评估的准确性。在本文中，我们利用半参数效率理论研究了在不同的数据集转移条件下高效估计目标总体风险的一般问题。我们考虑了一类通用的数据集转移条件，其中包括三种流行条件——协变量、标签和概念转移——作为特例。我们允许部分非重叠。

    Statistical machine learning methods often face the challenge of limited data available from the population of interest. One remedy is to leverage data from auxiliary source populations, which share some conditional distributions or are linked in other ways with the target domain. Techniques leveraging such \emph{dataset shift} conditions are known as \emph{domain adaptation} or \emph{transfer learning}. Despite extensive literature on dataset shift, limited works address how to efficiently use the auxiliary populations to improve the accuracy of risk evaluation for a given machine learning task in the target population.  In this paper, we study the general problem of efficiently estimating target population risk under various dataset shift conditions, leveraging semiparametric efficiency theory. We consider a general class of dataset shift conditions, which includes three popular conditions -- covariate, label and concept shift -- as special cases. We allow for partially non-overlappi
    
[^26]: 联邦学习：减少通信次数！

    Federated Learning You May Communicate Less Often!. (arXiv:2306.05862v1 [stat.ML])

    [http://arxiv.org/abs/2306.05862](http://arxiv.org/abs/2306.05862)

    本研究针对联邦学习设定，探讨了通信次数对泛化误差的影响，并建立了PAC-Bayes和率失真理论限制，这些限制对广泛的损失函数和学习算法适用。

    

    本研究探讨了联邦学习(Federated Learning, FL)模型在一般性的设置下的泛化误差。具体来说，我们研究了客户端和参数服务器之间通信次数的泛化误差演变，即客户端计算的本地模型在参数服务器上合并的频率对泛化误差的影响。我们建立了PAC-Bayes和率失真理论对泛化误差的限制，明确考虑通信次数对误差的影响，另外还考虑了参与设备数量K和个人数据集大小n对误差的影响。这些限制适用于广泛的损失函数和学习算法，似乎是FL设置中首次出现的。此外，我们将我们的限制应用于FL类型的支持向量机(FSVM)；我们在这种情况下推导了更明确的泛化误差限制。

    We investigate the generalization error of statistical learning models in a Federated Learning (FL) setting. Specifically, we study the evolution of the generalization error with the number of communication rounds between the clients and the parameter server, i.e., the effect on the generalization error of how often the local models as computed by the clients are aggregated at the parameter server. We establish PAC-Bayes and rate-distortion theoretic bounds on the generalization error that account explicitly for the effect of the number of rounds, say $ R \in \mathbb{N}$, in addition to the number of participating devices $K$ and individual datasets size $n$. The bounds, which apply in their generality for a large class of loss functions and learning algorithms, appear to be the first of their kind for the FL setting. Furthermore, we apply our bounds to FL-type Support Vector Machines (FSVM); and we derive (more) explicit bounds on the generalization error in this case. In particular, 
    
[^27]: 可解释的深度聚类

    Interpretable Deep Clustering. (arXiv:2306.04785v1 [cs.LG])

    [http://arxiv.org/abs/2306.04785](http://arxiv.org/abs/2306.04785)

    本文提出了一种可解释的深度学习框架，通过自我监督的方式从数据点中标识信息量丰富的特征，设计了一个模型和门矩阵来预测可解释的实例和聚类级别的聚类分配，并在合成和实际数据中验证了其可靠性和可解释性。

    

    聚类是一项广泛应用于数据分析中的基础学习任务。例如，生物学家经常使用聚类分配来分析基因组序列、医疗记录或图像。由于下游分析通常在聚类级别上执行，因此从业者寻求可靠且可解释的聚类模型。本文提出了一种新的深度学习框架，它可以预测可解释的实例和聚类级别的聚类分配。首先，我们提出一个自我监督的过程来从每个数据点中标识出一组信息量丰富的特征子集。然后，我们设计了一个模型，用于预测聚类分配和一个门矩阵，用于引导聚类级别的特征选择。我们证明了所提出的方法可以使用合成和实际数据可靠地预测聚类分配。此外，我们验证了我们的模型可以在实例和聚类级别上产生可解释的结果。

    Clustering is a fundamental learning task widely used as a first step in data analysis. For example, biologists often use cluster assignments to analyze genome sequences, medical records, or images. Since downstream analysis is typically performed at the cluster level, practitioners seek reliable and interpretable clustering models. We propose a new deep-learning framework that predicts interpretable cluster assignments at the instance and cluster levels. First, we present a self-supervised procedure to identify a subset of informative features from each data point. Then, we design a model that predicts cluster assignments and a gate matrix that leads to cluster-level feature selection. We show that the proposed method can reliably predict cluster assignments using synthetic and real data. Furthermore, we verify that our model leads to interpretable results at a sample and cluster level.
    
[^28]: 非线性分布鲁棒优化

    Nonlinear Distributionally Robust Optimization. (arXiv:2306.03202v1 [stat.ML])

    [http://arxiv.org/abs/2306.03202](http://arxiv.org/abs/2306.03202)

    本文提出一种新的非线性分布鲁棒优化算法，用于处理一类分布鲁棒优化问题，通过 Gateaux Derivative 处理一般风险度量。经过实验验证，该方法成功处理分布的非线性目标函数。

    

    本文关注一类分布鲁棒优化（DRO）问题，其中目标函数在分布上可能是非线性的，这与现有的文献有所不同。为解决在概率空间中优化非线性函数面临的理论和计算挑战，我们提出了一种Derivative和相应的平滑度概念，基于Gateaux Derivative来处理一般风险度量。我们通过Var、entropic risk和有限支持集上的三个运行风险度量示例来解释这些概念。然后，我们为概率空间中一般非线性优化问题提出了一种基于G-derivative的Frank-Wolfe（FW）算法，并以完全独立于范数的方式推导出其收敛性在提出的平滑度概念下。我们利用FW算法的设置来设计一种计算非线性DRO问题鞍点的方法。我们通过数值实验展示了我们方法处理分布的非线性目标函数的成功。

    This article focuses on a class of distributionally robust optimization (DRO) problems where, unlike the growing body of the literature, the objective function is potentially non-linear in the distribution. Existing methods to optimize nonlinear functions in probability space use the Frechet derivatives, which present both theoretical and computational challenges. Motivated by this, we propose an alternative notion for the derivative and corresponding smoothness based on Gateaux (G)-derivative for generic risk measures. These concepts are explained via three running risk measure examples of variance, entropic risk, and risk on finite support sets. We then propose a G-derivative based Frank-Wolfe~(FW) algorithm for generic non-linear optimization problems in probability spaces and establish its convergence under the proposed notion of smoothness in a completely norm-independent manner. We use the set-up of the FW algorithm to devise a methodology to compute a saddle point of the non-lin
    
[^29]: 随机函数下降法

    Random Function Descent. (arXiv:2305.01377v1 [math.OC])

    [http://arxiv.org/abs/2305.01377](http://arxiv.org/abs/2305.01377)

    本文提出了随机函数下降(RFD)算法，可以在随机环境中计算出步长并且与贝叶斯优化中的梯度下降算法相同。在合成基准测试中，RFD算法比未调整的Adam方法表现更好，提出的heuristic扩展可与调整后的Adam方法相媲美。

    

    虽然梯度下降方法在机器学习中十分常见，但是选择正确的步长经常需要进行“超参数调整”。这是因为回溯程序如Armijo's准则依赖于每个步骤中的质量评估，而这些评估在随机情况下不可用。由于优化方案可以用Taylor逼近来解释，我们将Taylor逼近替换为条件期望（最佳的$L^2$估计），提出了“随机函数下降”（RFD）。 在Bayesian优化中常见的一些轻微假设的情况下，我们证明了RFD与梯度下降算法是相同的，但是在随机情况下具有可计算的步长。我们在合成基准测试中比未调整的Adam方法表现更好。为了缩小与调整后的Adam算法之间的性能差距，我们提出了一种启发式扩展，可与调整后的Adam方法相媲美。

    While gradient based methods are ubiquitous in machine learning, selecting the right step size often requires "hyperparameter tuning". This is because backtracking procedures like Armijo's rule depend on quality evaluations in every step, which are not available in a stochastic context. Since optimization schemes can be motivated using Taylor approximations, we replace the Taylor approximation with the conditional expectation (the best $L^2$ estimator) and propose "Random Function Descent" (RFD). Under light assumptions common in Bayesian optimization, we prove that RFD is identical to gradient descent, but with calculable step sizes, even in a stochastic context. We beat untuned Adam in synthetic benchmarks. To close the performance gap to tuned Adam, we propose a heuristic extension competitive with tuned Adam.
    
[^30]: 可控的信任权衡下的合成数据审计与生成

    Auditing and Generating Synthetic Data with Controllable Trust Trade-offs. (arXiv:2304.10819v1 [cs.LG])

    [http://arxiv.org/abs/2304.10819](http://arxiv.org/abs/2304.10819)

    本论文提出了一个审计框架，能够以全面的方式评估合成数据和AI模型的具体效果，包括偏见和歧视预防、对真实数据的忠实程度、效用、鲁棒性和隐私保护。在多个用例中，审计框架平衡了信任和效用之间的权衡。

    

    现实中收集的数据往往存在偏差、不平衡，并且有泄露敏感和隐私信息的风险。这一事实引发了创建合成数据集的想法，以减轻真实数据中固有的风险、偏见、伤害和隐私问题。这个概念依赖于生成AI模型，以产生不偏执、保护隐私的合成数据，同时忠实于真实数据。在这种新范式中，我们如何知道这种方法是否兑现了其承诺？我们提出了一个审计框架，提供了对合成数据集和基于它们训练的AI模型的全面评估，围绕偏见和歧视的预防、对真实数据的忠实程度、效用、鲁棒性和隐私保护。我们通过审计多个生成模型在不同用例中展示了我们的框架，包括教育、医疗保健、银行、人力资源，以及从表格，时间序列到自然语言的不同模态。我们的用例展示了在合成数据生成中平衡信任和效用的权衡的重要性。

    Data collected from the real world tends to be biased, unbalanced, and at risk of exposing sensitive and private information. This reality has given rise to the idea of creating synthetic datasets to alleviate risk, bias, harm, and privacy concerns inherent in the real data. This concept relies on Generative AI models to produce unbiased, privacy-preserving synthetic data while being true to the real data. In this new paradigm, how can we tell if this approach delivers on its promises? We present an auditing framework that offers a holistic assessment of synthetic datasets and AI models trained on them, centered around bias and discrimination prevention, fidelity to the real data, utility, robustness, and privacy preservation. We showcase our framework by auditing multiple generative models on diverse use cases, including education, healthcare, banking, human resources, and across different modalities, from tabular, to time-series, to natural language. Our use cases demonstrate the imp
    
[^31]: 《无免费午餐定理、科尔莫戈洛夫复杂性及归纳偏差在机器学习中的作用》

    The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning. (arXiv:2304.05366v1 [cs.LG])

    [http://arxiv.org/abs/2304.05366](http://arxiv.org/abs/2304.05366)

    本论文阐述了无免费午餐定理的监督学习中的限制，证明了归纳偏差可以提高学习算法的效果，并且展示了神经网络模型的偏好与现实世界的数据分布相关。

    

    监督学习的无免费午餐定理指出，没有一个学习算法可以解决所有问题，或者所有学习算法在均匀分布的学习问题上平均精度达到完全相同。因此，这些定理经常被引用来支持个别问题需要特别定制的归纳偏差的概念。我们认为，尽管几乎所有均匀采样的数据集具有高复杂性，但现实世界中的问题不成比例地产生低复杂度的数据，并且我们认为神经网络模型也具有同样的偏好，这种偏好使用科尔莫戈洛夫复杂度进行了形式化。值得注意的是，我们展示了为特定领域设计的体系结构，例如计算机视觉，可以压缩各种看似不相关的领域的数据集。我们的实验表明，预先训练和即使是随机初始化的语言模型都更喜欢生成低复杂度的序列。尽管无免费午餐定理似乎表明各个问题需要专门的学习算法，但我们解释说，学习算法通常可以通过编码关于真实世界数据分布的先前知识的归纳偏差来改进。

    No free lunch theorems for supervised learning state that no learner can solve all problems or that all learners achieve exactly the same accuracy on average over a uniform distribution on learning problems. Accordingly, these theorems are often referenced in support of the notion that individual problems require specially tailored inductive biases. While virtually all uniformly sampled datasets have high complexity, real-world problems disproportionately generate low-complexity data, and we argue that neural network models share this same preference, formalized using Kolmogorov complexity. Notably, we show that architectures designed for a particular domain, such as computer vision, can compress datasets on a variety of seemingly unrelated domains. Our experiments show that pre-trained and even randomly initialized language models prefer to generate low-complexity sequences. Whereas no free lunch theorems seemingly indicate that individual problems require specialized learners, we exp
    
[^32]: PAC-Bayesian软演员-评论家学习

    PAC-Bayesian Soft Actor-Critic Learning. (arXiv:2301.12776v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.12776](http://arxiv.org/abs/2301.12776)

    本文提出了一种使用PAC-Bayesian bound作为Soft Actor-Critic (SAC)算法评论家训练目标的方法，以解决训练不稳定的问题，并通过评论家引导的随机搜索探索多个未来来提高在线学习性能。在多个经典控制和运动任务中，该算法具有样本效率和遗憾最小化方面的明显优势。

    

    演员-评论家算法通过两个分别作策略评估和改进的功能逼近器来解决增强学习(RL)的双重目标。此方法的实用性是以训练不稳定为代价的，主要原因是评论家逼近误差对演员的破坏性影响。我们通过首次采用一个现有的可能近似正确(PAC)Bayesian界限作为Soft Actor-Critic (SAC)算法的评论家训练目标来解决这个瓶颈。此外，我们进一步证明了当随机演员通过评论家引导的随机搜索探索多个未来时，在线学习性能显著提高。我们观察到我们得到的算法在多个经典控制和运动任务中，在样本效率和遗憾最小化方面与现有技术相比具有明显优势。

    Actor-critic algorithms address the dual goals of reinforcement learning (RL), policy evaluation and improvement, via two separate function approximators. The practicality of this approach comes at the expense of training instability, caused mainly by the destructive effect of the approximation errors of the critic on the actor. We tackle this bottleneck by employing an existing Probably Approximately Correct (PAC) Bayesian bound for the first time as the critic training objective of the Soft Actor-Critic (SAC) algorithm. We further demonstrate that online learning performance improves significantly when a stochastic actor explores multiple futures by critic-guided random search. We observe our resulting algorithm to compare favorably to the state of the art on multiple classical control and locomotion tasks in terms of both sample efficiency and regret minimization.
    
[^33]: 在治疗社区中识别同伴影响

    Identifying Peer Influence in Therapeutic Communities. (arXiv:2203.14223v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2203.14223](http://arxiv.org/abs/2203.14223)

    本研究调查了治疗社区中的同伴影响或角色模型效应对于成功毕业的影响。通过分析三个治疗社区的观察数据，我们发现肯定的同伴交流对于居民在自己离开之前成功毕业与否有显著影响。

    

    我们研究在治疗社区中是否存在同伴影响或角色模型效应对于成功毕业的影响。我们分析了3个保留了居民之间确认和修正交流记录以及他们入住和离开日期的治疗社区的匿名个体观察数据。这些确认交流使我们能够形成同伴网络，而入住和离开日期使我们能够定义感兴趣的因果效应。我们将因果角色模型效应概念化为居民（自我）观察到他们的某个社交联系人（例如，给予肯定的同伴）在自我离开之前成功毕业与不成功毕业的预期结果之间的差异。由于同伴影响通常与观察数据中未观察到的同质性混淆，我们使用潜变量模型对网络进行建模以估计同质性并将其包含在结果方程中。我们提供了一个理论保证，它可以解决网络中的内生性问题并提供一致的估计。

    We investigate if there is a peer influence or role model effect on successful graduation from Therapeutic Communities (TCs). We analyze anonymized individual-level observational data from 3 TCs that kept records of written exchanges of affirmations and corrections among residents, and their precise entry and exit dates. The affirmations allow us to form peer networks, and the entry and exit dates allow us to define a causal effect of interest. We conceptualize the causal role model effect as measuring the difference in the expected outcome of a resident (ego) who can observe one of their social contacts (e.g., peers who gave affirmations), to be successful in graduating before the ego's exit vs not successfully graduating before the ego's exit. Since peer influence is usually confounded with unobserved homophily in observational data, we model the network with a latent variable model to estimate homophily and include it in the outcome equation. We provide a theoretical guarantee that 
    
[^34]: 贝叶斯学习规则

    The Bayesian Learning Rule. (arXiv:2107.04562v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2107.04562](http://arxiv.org/abs/2107.04562)

    许多机器学习算法都可以归结为贝叶斯学习规则，该规则通过利用自然梯度来逼近后验分布，从而得到广泛的算法应用。这一工作不仅统一了现有算法，还帮助我们设计新的算法。

    

    我们展示了许多机器学习算法是一个称为贝叶斯学习规则的单一算法的特例。这个规则是从贝叶斯原理推导出来的，可以从优化、深度学习和图形模型等领域得到广泛的算法。这包括经典算法如岭回归、牛顿法和卡尔曼滤波器，以及现代深度学习算法如随机梯度下降、RMSprop和Dropout。推导这些算法的关键思想是使用自然梯度估计的候选分布来逼近后验分布。不同的候选分布会导致不同的算法，对自然梯度的进一步逼近则会产生这些算法的变种。我们的工作不仅统一、泛化和改进了现有算法，还帮助我们设计新的算法。

    We show that many machine-learning algorithms are specific instances of a single algorithm called the Bayesian learning rule. The rule, derived from Bayesian principles, yields a wide-range of algorithms from fields such as optimization, deep learning, and graphical models. This includes classical algorithms such as ridge regression, Newton's method, and Kalman filter, as well as modern deep-learning algorithms such as stochastic-gradient descent, RMSprop, and Dropout. The key idea in deriving such algorithms is to approximate the posterior using candidate distributions estimated by using natural gradients. Different candidate distributions result in different algorithms and further approximations to natural gradients give rise to variants of those algorithms. Our work not only unifies, generalizes, and improves existing algorithms, but also helps us design new ones.
    

