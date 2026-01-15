# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hamiltonian Property Testing](https://arxiv.org/abs/2403.02968) | 本文研究了Hamiltonian的本地性测试作为属性测试问题，重点在于确定未知的$n$比特Hamiltonian是否是$k$局部的，通过对$H$的时间演化进行访问来解决问题。 |
| [^2] | [Lens: A Foundation Model for Network Traffic](https://arxiv.org/abs/2402.03646) | "Lens"是一个基于T5架构的基础网络流量模型，通过学习大规模无标签数据的预训练表示，能够在流量理解和生成任务中取得精确的预测和生成。 |
| [^3] | [Soft Contrastive Learning for Time Series](https://arxiv.org/abs/2312.16424) | 提出了一种名为SoftCLT的方法，通过引入实例级和时间级软对比损失，解决了在时间序列中忽略固有相关性所导致的学习表示质量下降的问题。 |
| [^4] | [Kernel Limit of Recurrent Neural Networks Trained on Ergodic Data Sequences.](http://arxiv.org/abs/2308.14555) | 本文研究了循环神经网络在遍历数据序列上训练时的核极限，利用数学方法对其渐近特性进行了描述，并证明了RNN收敛到与随机代数方程的不动点耦合的无穷维ODE的解。这对于理解和改进循环神经网络具有重要意义。 |
| [^5] | [Tipping Point Forecasting in Non-Stationary Dynamics on Function Spaces.](http://arxiv.org/abs/2308.08794) | 本文提出了一种利用循环神经算子学习非平稳动力系统演化的方法，并且通过基于不确定性的方法检测未来的翻车点。同时，我们还提出了一种符合预测框架，通过监测与物理约束的偏离来预测翻车点，从而使得预测结果具有严格的不确定性度量。 |
| [^6] | [A Survey on Personalized Affective Computing in Human-Machine Interaction.](http://arxiv.org/abs/2304.00377) | 论文调查了情感计算中的个性化方法，将其分为七类，并给出了调查文献的统计元分析。 |
| [^7] | [Reinforcement Learning with Exogenous States and Rewards.](http://arxiv.org/abs/2303.12957) | 该研究提出了一种强化学习的方法，通过将MDP分解为外生和内生两个部分，优化内生奖励，在状态空间的内生和外生状态空间没有事先给出的情况下，提出了正确的算法进行自动发现。 |

# 详细

[^1]: Hamiltonian性质测试

    Hamiltonian Property Testing

    [https://arxiv.org/abs/2403.02968](https://arxiv.org/abs/2403.02968)

    本文研究了Hamiltonian的本地性测试作为属性测试问题，重点在于确定未知的$n$比特Hamiltonian是否是$k$局部的，通过对$H$的时间演化进行访问来解决问题。

    

    本文研究了Hamiltonian本地性测试作为一个属性测试问题，即确定一个未知的$n$比特Hamiltonian $H$是否是$k$局部的，或者与所有$k$局部Hamiltonian都相距$\varepsilon$，并通过对$H$的时间演化进行访问来解决问题。

    arXiv:2403.02968v1 Announce Type: cross  Abstract: Locality is a fundamental feature of many physical time evolutions. Assumptions on locality and related structural properties also underlie recently proposed procedures for learning an unknown Hamiltonian from access to the induced time evolution. However, no protocols to rigorously test whether an unknown Hamiltonian is local were known. We investigate Hamiltonian locality testing as a property testing problem, where the task is to determine whether an unknown $n$-qubit Hamiltonian $H$ is $k$-local or $\varepsilon$-far from all $k$-local Hamiltonians, given access to the time evolution along $H$. First, we emphasize the importance of the chosen distance measure: With respect to the operator norm, a worst-case distance measure, incoherent quantum locality testers require $\tilde{\Omega}(2^n)$ many time evolution queries and an expected total evolution time of $\tilde{\Omega}(2^n / \varepsilon)$, and even coherent testers need $\Omega(2
    
[^2]: Lens: 网络流量的基础模型

    Lens: A Foundation Model for Network Traffic

    [https://arxiv.org/abs/2402.03646](https://arxiv.org/abs/2402.03646)

    "Lens"是一个基于T5架构的基础网络流量模型，通过学习大规模无标签数据的预训练表示，能够在流量理解和生成任务中取得精确的预测和生成。

    

    网络流量是指通过互联网或连接计算机的任何系统发送和接收的信息量。分析和理解网络流量对于提高网络安全和管理至关重要。然而，由于数据包的特殊特性，如异构标头和缺乏语义的加密负载，网络流量的分析带来了巨大的挑战。为了捕捉流量的潜在语义，一些研究采用了基于Transformer编码器或解码器的预训练技术，从大规模的流量数据中学习表示。然而，这些方法通常只在流量理解（分类）或流量生成任务中表现出色。为了解决这个问题，我们开发了Lens，这是一个基础的网络流量模型，利用T5架构从大规模的无标签数据中学习预训练表示。借助编码器-解码器框架的优势，该模型能够捕捉全局和局部特征，实现精确的流量预测和生成。

    Network traffic refers to the amount of information being sent and received over the internet or any system that connects computers. Analyzing and understanding network traffic is vital for improving network security and management. However, the analysis of network traffic poses great challenges due to the unique characteristics of data packets, such as heterogeneous headers and encrypted payload lacking semantics. To capture the latent semantics of traffic, a few studies have adopted pre-training techniques based on the Transformer encoder or decoder to learn the representations from large-scale traffic data. However, these methods typically excel only in traffic understanding (classification) or traffic generation tasks. To address this issue, we develop Lens, a foundational network traffic model that leverages the T5 architecture to learn the pre-trained representations from large-scale unlabeled data. Harnessing the strength of the encoder-decoder framework, which captures the glob
    
[^3]: 时间序列的软对比学习

    Soft Contrastive Learning for Time Series

    [https://arxiv.org/abs/2312.16424](https://arxiv.org/abs/2312.16424)

    提出了一种名为SoftCLT的方法，通过引入实例级和时间级软对比损失，解决了在时间序列中忽略固有相关性所导致的学习表示质量下降的问题。

    

    对比学习已经被证明在自监督学习中对于从时间序列中学习表示是有效的。然而，将时间序列中相似的实例或相邻时间戳的值进行对比会忽略它们固有的相关性，从而导致学习表示的质量下降。为了解决这个问题，我们提出了SoftCLT，一种简单而有效的时间序列软对比学习策略。这是通过引入从零到一的软赋值的实例级和时间级对比损失来实现的。具体来说，我们为1)基于数据空间上的时间序列之间的距离定义了实例级对比损失的软赋值，并为2)基于时间戳之间的差异定义了时间级对比损失。SoftCLT是一种即插即用的时间序列对比学习方法，可以提高学习表示的质量，没有过多复杂的设计。

    arXiv:2312.16424v2 Announce Type: replace-cross  Abstract: Contrastive learning has shown to be effective to learn representations from time series in a self-supervised way. However, contrasting similar time series instances or values from adjacent timestamps within a time series leads to ignore their inherent correlations, which results in deteriorating the quality of learned representations. To address this issue, we propose SoftCLT, a simple yet effective soft contrastive learning strategy for time series. This is achieved by introducing instance-wise and temporal contrastive loss with soft assignments ranging from zero to one. Specifically, we define soft assignments for 1) instance-wise contrastive loss by the distance between time series on the data space, and 2) temporal contrastive loss by the difference of timestamps. SoftCLT is a plug-and-play method for time series contrastive learning that improves the quality of learned representations without bells and whistles. In experi
    
[^4]: 循环神经网络在遍历数据序列上训练的核极限

    Kernel Limit of Recurrent Neural Networks Trained on Ergodic Data Sequences. (arXiv:2308.14555v1 [cs.LG])

    [http://arxiv.org/abs/2308.14555](http://arxiv.org/abs/2308.14555)

    本文研究了循环神经网络在遍历数据序列上训练时的核极限，利用数学方法对其渐近特性进行了描述，并证明了RNN收敛到与随机代数方程的不动点耦合的无穷维ODE的解。这对于理解和改进循环神经网络具有重要意义。

    

    本文开发了数学方法来描述循环神经网络（RNN）的渐近特性，其中隐藏单元的数量、序列中的数据样本、隐藏状态的更新和训练步骤同时趋于无穷大。对于具有简化权重矩阵的RNN，我们证明了RNN收敛到与随机代数方程的不动点耦合的无穷维ODE的解。分析需要解决RNN所特有的几个挑战。在典型的均场应用中（例如前馈神经网络），离散的更新量为$\mathcal{O}(\frac{1}{N})$，更新的次数为$\mathcal{O}(N)$。因此，系统可以表示为适当ODE/PDE的Euler逼近，当$N \rightarrow \infty$时收敛到该ODE/PDE。然而，RNN的隐藏层更新为$\mathcal{O}(1)$。因此，RNN不能表示为ODE/PDE的离散化和标准均场技术。

    Mathematical methods are developed to characterize the asymptotics of recurrent neural networks (RNN) as the number of hidden units, data samples in the sequence, hidden state updates, and training steps simultaneously grow to infinity. In the case of an RNN with a simplified weight matrix, we prove the convergence of the RNN to the solution of an infinite-dimensional ODE coupled with the fixed point of a random algebraic equation. The analysis requires addressing several challenges which are unique to RNNs. In typical mean-field applications (e.g., feedforward neural networks), discrete updates are of magnitude $\mathcal{O}(\frac{1}{N})$ and the number of updates is $\mathcal{O}(N)$. Therefore, the system can be represented as an Euler approximation of an appropriate ODE/PDE, which it will converge to as $N \rightarrow \infty$. However, the RNN hidden layer updates are $\mathcal{O}(1)$. Therefore, RNNs cannot be represented as a discretization of an ODE/PDE and standard mean-field tec
    
[^5]: 功能空间中非平稳动力学中的翻车点预测

    Tipping Point Forecasting in Non-Stationary Dynamics on Function Spaces. (arXiv:2308.08794v1 [cs.LG])

    [http://arxiv.org/abs/2308.08794](http://arxiv.org/abs/2308.08794)

    本文提出了一种利用循环神经算子学习非平稳动力系统演化的方法，并且通过基于不确定性的方法检测未来的翻车点。同时，我们还提出了一种符合预测框架，通过监测与物理约束的偏离来预测翻车点，从而使得预测结果具有严格的不确定性度量。

    

    翻车点是非平稳和混沌动力系统演化中的突变、剧烈且常常不可逆的变化。例如，预计温室气体浓度的增加会导致低云覆盖的急剧减少，被称为气候学的翻车点。在本文中，我们利用一种新颖的循环神经算子（RNO）学习这种非平稳动力系统的演化，RNO可以学习函数空间之间的映射关系。在仅训练RNO在翻车点之前的动力学数据之后，我们采用基于不确定性的方法来检测未来的翻车点。具体而言，我们提出了一个符合预测框架，通过监测与物理约束（如守恒量和偏微分方程）偏离来预测翻车点，从而使得对这些突变的预测伴随着一种严格的不确定性度量。我们将我们提出的方法应用于非平稳常微分方程和偏微分方程的案例。

    Tipping points are abrupt, drastic, and often irreversible changes in the evolution of non-stationary and chaotic dynamical systems. For instance, increased greenhouse gas concentrations are predicted to lead to drastic decreases in low cloud cover, referred to as a climatological tipping point. In this paper, we learn the evolution of such non-stationary dynamical systems using a novel recurrent neural operator (RNO), which learns mappings between function spaces. After training RNO on only the pre-tipping dynamics, we employ it to detect future tipping points using an uncertainty-based approach. In particular, we propose a conformal prediction framework to forecast tipping points by monitoring deviations from physics constraints (such as conserved quantities and partial differential equations), enabling forecasting of these abrupt changes along with a rigorous measure of uncertainty. We illustrate our proposed methodology on non-stationary ordinary and partial differential equations,
    
[^6]: 个性化情感计算在人机交互中的调查

    A Survey on Personalized Affective Computing in Human-Machine Interaction. (arXiv:2304.00377v1 [cs.HC])

    [http://arxiv.org/abs/2304.00377](http://arxiv.org/abs/2304.00377)

    论文调查了情感计算中的个性化方法，将其分为七类，并给出了调查文献的统计元分析。

    

    在计算机领域中，个性化的目的是通过优化一个或多个性能指标并遵守特定约束条件来训练迎合特定个人或人群的模型。本文讨论了情感和人格计算（以下简称情感计算）中个性化的必要性，并对情感计算中个性化的最新方法进行了调查。我们的调查涵盖了训练技术和目标，以实现情感计算模型的个性化定制。我们将现有的方法分为七类：（1）面向特定目标的模型，（2）面向特定群体的模型，（3）基于加权的方法，（4）微调方法，（5）多任务学习，（6）生成式模型和（7）特征增强。此外，我们提供了对调查文献的统计元分析，分析了不同情感计算任务、交互模式、交互上下文以及所涉及领域的普遍性。

    In computing, the aim of personalization is to train a model that caters to a specific individual or group of people by optimizing one or more performance metrics and adhering to specific constraints. In this paper, we discuss the need for personalization in affective and personality computing (hereinafter referred to as affective computing). We present a survey of state-of-the-art approaches for personalization in affective computing. Our review spans training techniques and objectives towards the personalization of affective computing models. We group existing approaches into seven categories: (1) Target-specific Models, (2) Group-specific Models, (3) Weighting-based Approaches, (4) Fine-tuning Approaches, (5) Multitask Learning, (6) Generative-based Models, and (7) Feature Augmentation. Additionally, we provide a statistical meta-analysis of the surveyed literature, analyzing the prevalence of different affective computing tasks, interaction modes, interaction contexts, and the leve
    
[^7]: 具有外部状态和奖励的强化学习

    Reinforcement Learning with Exogenous States and Rewards. (arXiv:2303.12957v1 [cs.LG])

    [http://arxiv.org/abs/2303.12957](http://arxiv.org/abs/2303.12957)

    该研究提出了一种强化学习的方法，通过将MDP分解为外生和内生两个部分，优化内生奖励，在状态空间的内生和外生状态空间没有事先给出的情况下，提出了正确的算法进行自动发现。

    

    外部状态变量和奖励会通过向奖励信号注入不可控的变化而减慢强化学习的速度。本文对外部状态变量和奖励进行了正式化，并表明如果奖励函数加法分解成内生和外生两个部分，MDP可以分解为一个外生马尔可夫奖励过程（基于外部奖励）和一个内生马尔可夫决策过程（优化内生奖励）。内生MDP的任何最优策略也是原始MDP的最优策略，但由于内生奖励通常具有降低的方差，因此内生MDP更容易求解。我们研究了状态空间分解为内外生状态空间的情况，而这种状态空间分解并没有给出，而是必须发现。本文介绍并证明了在线性组合下发现内生和外生状态空间的算法的正确性。

    Exogenous state variables and rewards can slow reinforcement learning by injecting uncontrolled variation into the reward signal. This paper formalizes exogenous state variables and rewards and shows that if the reward function decomposes additively into endogenous and exogenous components, the MDP can be decomposed into an exogenous Markov Reward Process (based on the exogenous reward) and an endogenous Markov Decision Process (optimizing the endogenous reward). Any optimal policy for the endogenous MDP is also an optimal policy for the original MDP, but because the endogenous reward typically has reduced variance, the endogenous MDP is easier to solve. We study settings where the decomposition of the state space into exogenous and endogenous state spaces is not given but must be discovered. The paper introduces and proves correctness of algorithms for discovering the exogenous and endogenous subspaces of the state space when they are mixed through linear combination. These algorithms
    

