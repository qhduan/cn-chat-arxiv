# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data Dependent Regret Guarantees Against General Comparators for Full or Bandit Feedback.](http://arxiv.org/abs/2303.06526) | 该论文提出了一个数据相关的在线学习算法框架，可以在全专家反馈和Bandit反馈设置中具有数据相关的遗憾保证，适用于各种问题场景。 |
| [^2] | [Deep Reinforcement Learning Based Power Allocation for Minimizing AoI and Energy Consumption in MIMO-NOMA IoT Systems.](http://arxiv.org/abs/2303.06411) | 本文提出了一种基于深度强化学习的MIMO-NOMA IoT系统功率分配方法，以最小化AoI和能耗。 |
| [^3] | [Learning to Precode for Integrated Sensing and Communications Systems.](http://arxiv.org/abs/2303.06381) | 本文提出了一种无监督学习神经模型，用于设计集成感知和通信（ISAC）系统的传输预编码器，以最大化最坏情况下的目标照明功率，同时确保所有用户的最小信干噪比（SINR）。通过数值模拟，证明了该方法在存在信道估计误差的情况下优于传统的基于优化的方法，同时产生较小的计算复杂度，并且在不同的信道条件下具有良好的泛化能力。 |
| [^4] | [Privacy-Preserving Cooperative Visible Light Positioning for Nonstationary Environment: A Federated Learning Perspective.](http://arxiv.org/abs/2303.06361) | 本文提出了一种基于联邦学习的合作可见光定位方案，通过共同训练适应环境变化的全局模型，提高了在非静态环境下的定位精度和泛化能力。 |
| [^5] | [Deflated HeteroPCA: Overcoming the curse of ill-conditioning in heteroskedastic PCA.](http://arxiv.org/abs/2303.06198) | 本文提出了一种新的算法，称为缩减异方差PCA，它在克服病态问题的同时实现了近乎最优和无条件数的理论保证。 |

# 详细

[^1]: 数据相关的在线学习算法框架

    Data Dependent Regret Guarantees Against General Comparators for Full or Bandit Feedback. (arXiv:2303.06526v1 [cs.LG])

    [http://arxiv.org/abs/2303.06526](http://arxiv.org/abs/2303.06526)

    该论文提出了一个数据相关的在线学习算法框架，可以在全专家反馈和Bandit反馈设置中具有数据相关的遗憾保证，适用于各种问题场景。

    This paper proposes a data-dependent online learning algorithm framework that has data-dependent regret guarantees in both full expert feedback and bandit feedback settings, applicable for a wide variety of problem scenarios.

    我们研究了对抗性在线学习问题，并创建了一个完全在线的算法框架，具有在全专家反馈和Bandit反馈设置中具有数据相关的遗憾保证。我们研究了我们的算法对一般比较器的预期性能，使其适用于各种问题场景。我们的算法从通用预测角度工作，使用的性能度量是对任意比较器序列的预期遗憾，即我们的损失与竞争损失序列之间的差异。竞争类可以设计为包括固定臂选择、切换Bandit、上下文Bandit、周期Bandit或任何其他感兴趣的竞争。竞争类中的序列通常由具体应用程序确定，并应相应地设计。我们的算法既不使用也不需要任何有关损失序列的初步信息，完全在线。其

    We study the adversarial online learning problem and create a completely online algorithmic framework that has data dependent regret guarantees in both full expert feedback and bandit feedback settings. We study the expected performance of our algorithm against general comparators, which makes it applicable for a wide variety of problem scenarios. Our algorithm works from a universal prediction perspective and the performance measure used is the expected regret against arbitrary comparator sequences, which is the difference between our losses and a competing loss sequence. The competition class can be designed to include fixed arm selections, switching bandits, contextual bandits, periodic bandits or any other competition of interest. The sequences in the competition class are generally determined by the specific application at hand and should be designed accordingly. Our algorithm neither uses nor needs any preliminary information about the loss sequences and is completely online. Its
    
[^2]: 基于深度强化学习的MIMO-NOMA IoT系统功率分配，以最小化AoI和能耗

    Deep Reinforcement Learning Based Power Allocation for Minimizing AoI and Energy Consumption in MIMO-NOMA IoT Systems. (arXiv:2303.06411v1 [cs.IT])

    [http://arxiv.org/abs/2303.06411](http://arxiv.org/abs/2303.06411)

    本文提出了一种基于深度强化学习的MIMO-NOMA IoT系统功率分配方法，以最小化AoI和能耗。

    This paper proposes a deep reinforcement learning based power allocation method for MIMO-NOMA IoT systems to minimize AoI and energy consumption.

    多输入多输出和非正交多址（MIMO-NOMA）物联网（IoT）系统可以显著提高信道容量和频谱效率，以支持实时应用。时延（AoI）是实时应用的重要指标，但没有文献最小化MIMO-NOMA IoT系统的AoI，这促使我们进行这项工作。在MIMO-NOMA IoT系统中，基站（BS）确定样本收集要求并为每个IoT设备分配传输功率。每个设备根据样本收集要求确定是否采样数据，并采用分配的功率将采样的数据通过MIMO-NOMA信道传输到BS。然后，BS采用连续干扰消除（SIC）技术解码每个设备传输的数据信号。样本收集要求和功率分配将影响系统的AoI和能耗。这是至关重要的。

    Multi-input multi-out and non-orthogonal multiple access (MIMO-NOMA) internet-of-things (IoT) systems can improve channel capacity and spectrum efficiency distinctly to support the real-time applications. Age of information (AoI) is an important metric for real-time application, but there is no literature have minimized AoI of the MIMO-NOMA IoT system, which motivates us to conduct this work. In MIMO-NOMA IoT system, the base station (BS) determines the sample collection requirements and allocates the transmission power for each IoT device. Each device determines whether to sample data according to the sample collection requirements and adopts the allocated power to transmit the sampled data to the BS over MIMO-NOMA channel. Afterwards, the BS employs successive interference cancelation (SIC) technique to decode the signal of the data transmitted by each device. The sample collection requirements and power allocation would affect AoI and energy consumption of the system. It is critical
    
[^3]: 学习预编码用于集成感知和通信系统

    Learning to Precode for Integrated Sensing and Communications Systems. (arXiv:2303.06381v1 [eess.SP])

    [http://arxiv.org/abs/2303.06381](http://arxiv.org/abs/2303.06381)

    本文提出了一种无监督学习神经模型，用于设计集成感知和通信（ISAC）系统的传输预编码器，以最大化最坏情况下的目标照明功率，同时确保所有用户的最小信干噪比（SINR）。通过数值模拟，证明了该方法在存在信道估计误差的情况下优于传统的基于优化的方法，同时产生较小的计算复杂度，并且在不同的信道条件下具有良好的泛化能力。

    This paper proposes an unsupervised learning neural model to design transmit precoders for integrated sensing and communication (ISAC) systems to maximize the worst-case target illumination power while ensuring a minimum signal-to-interference-plus-noise ratio (SINR) for all the users. The proposed method outperforms traditional optimization-based methods in presence of channel estimation errors while incurring lesser computational complexity and generalizing well across different channel conditions that were not shown during training.

    本文提出了一种无监督学习神经模型，用于设计集成感知和通信（ISAC）系统的传输预编码器，以最大化最坏情况下的目标照明功率，同时确保所有用户的最小信干噪比（SINR）。从上行导频和回波中学习传输预编码器的问题可以看作是一个参数化函数估计问题，我们提出使用神经网络模型来学习这个函数。为了学习神经网络参数，我们开发了一种基于一阶最优性条件的损失函数，以纳入SINR和功率约束。通过数值模拟，我们证明了所提出的方法在存在信道估计误差的情况下优于传统的基于优化的方法，同时产生较小的计算复杂度，并且在不同的信道条件下具有良好的泛化能力，这些条件在训练期间没有显示出来。

    In this paper, we present an unsupervised learning neural model to design transmit precoders for integrated sensing and communication (ISAC) systems to maximize the worst-case target illumination power while ensuring a minimum signal-to-interference-plus-noise ratio (SINR) for all the users. The problem of learning transmit precoders from uplink pilots and echoes can be viewed as a parameterized function estimation problem and we propose to learn this function using a neural network model. To learn the neural network parameters, we develop a novel loss function based on the first-order optimality conditions to incorporate the SINR and power constraints. Through numerical simulations, we demonstrate that the proposed method outperforms traditional optimization-based methods in presence of channel estimation errors while incurring lesser computational complexity and generalizing well across different channel conditions that were not shown during training.
    
[^4]: 面向非静态环境的隐私保护合作可见光定位：联邦学习视角

    Privacy-Preserving Cooperative Visible Light Positioning for Nonstationary Environment: A Federated Learning Perspective. (arXiv:2303.06361v1 [eess.SP])

    [http://arxiv.org/abs/2303.06361](http://arxiv.org/abs/2303.06361)

    本文提出了一种基于联邦学习的合作可见光定位方案，通过共同训练适应环境变化的全局模型，提高了在非静态环境下的定位精度和泛化能力。

    This paper proposes a cooperative visible light positioning scheme based on federated learning, which improves the positioning accuracy and generalization capability in nonstationary environments by jointly training a global model adaptive to environmental changes without sharing private data of users.

    可见光定位（VLP）作为一种有前途的室内定位技术，已经引起了足够的关注。然而，在非静态环境下，由于高度时变的信道，VLP的性能受到限制。为了提高非静态环境下的定位精度和泛化能力，本文提出了一种基于联邦学习（FL）的合作VLP方案。利用FL框架，用户可以共同训练适应环境变化的全局模型，而不共享用户的私有数据。此外，提出了一种合作可见光定位网络（CVPosNet），以加速收敛速度和提高定位精度。仿真结果表明，所提出的方案在非静态环境下优于基准方案。

    Visible light positioning (VLP) has drawn plenty of attention as a promising indoor positioning technique. However, in nonstationary environments, the performance of VLP is limited because of the highly time-varying channels. To improve the positioning accuracy and generalization capability in nonstationary environments, a cooperative VLP scheme based on federated learning (FL) is proposed in this paper. Exploiting the FL framework, a global model adaptive to environmental changes can be jointly trained by users without sharing private data of users. Moreover, a Cooperative Visible-light Positioning Network (CVPosNet) is proposed to accelerate the convergence rate and improve the positioning accuracy. Simulation results show that the proposed scheme outperforms the benchmark schemes, especially in nonstationary environments.
    
[^5]: 克服异方差PCA中病态问题的缩减算法

    Deflated HeteroPCA: Overcoming the curse of ill-conditioning in heteroskedastic PCA. (arXiv:2303.06198v1 [math.ST])

    [http://arxiv.org/abs/2303.06198](http://arxiv.org/abs/2303.06198)

    本文提出了一种新的算法，称为缩减异方差PCA，它在克服病态问题的同时实现了近乎最优和无条件数的理论保证。

    This paper proposes a novel algorithm, called Deflated-HeteroPCA, that overcomes the curse of ill-conditioning in heteroskedastic PCA while achieving near-optimal and condition-number-free theoretical guarantees.

    本文关注于从受污染的数据中估计低秩矩阵X*的列子空间。当存在异方差噪声和不平衡的维度（即n2 >> n1）时，如何在容纳最广泛的信噪比范围的同时获得最佳的统计精度变得特别具有挑战性。虽然最先进的算法HeteroPCA成为解决这个问题的强有力的解决方案，但它遭受了“病态问题的诅咒”，即随着X*的条件数增长，其性能会下降。为了克服这个关键问题而不影响允许的信噪比范围，我们提出了一种新的算法，称为缩减异方差PCA，它在$\ell_2$和$\ell_{2,\infty}$统计精度方面实现了近乎最优和无条件数的理论保证。所提出的算法将谱分成两部分

    This paper is concerned with estimating the column subspace of a low-rank matrix $\boldsymbol{X}^\star \in \mathbb{R}^{n_1\times n_2}$ from contaminated data. How to obtain optimal statistical accuracy while accommodating the widest range of signal-to-noise ratios (SNRs) becomes particularly challenging in the presence of heteroskedastic noise and unbalanced dimensionality (i.e., $n_2\gg n_1$). While the state-of-the-art algorithm $\textsf{HeteroPCA}$ emerges as a powerful solution for solving this problem, it suffers from "the curse of ill-conditioning," namely, its performance degrades as the condition number of $\boldsymbol{X}^\star$ grows. In order to overcome this critical issue without compromising the range of allowable SNRs, we propose a novel algorithm, called $\textsf{Deflated-HeteroPCA}$, that achieves near-optimal and condition-number-free theoretical guarantees in terms of both $\ell_2$ and $\ell_{2,\infty}$ statistical accuracy. The proposed algorithm divides the spectrum
    

