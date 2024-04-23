# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rethinking Adversarial Inverse Reinforcement Learning: From the Angles of Policy Imitation and Transferable Reward Recovery](https://arxiv.org/abs/2403.14593) | 重新思考对抗逆强化学习中的策略模仿和可转移奖励恢复，提出了一个混合框架PPO-AIRL + SAC以解决SAC算法在AIRL训练中无法全面解开奖励函数的问题。 |
| [^2] | [Machine learning-based system reliability analysis with Gaussian Process Regression](https://arxiv.org/abs/2403.11125) | 本文提出了基于高斯过程回归的机器学习系统可靠性分析方法，并通过几个定理探讨了最优学习策略，包括考虑和忽略样本之间的相关性以及顺序多个训练样本增益的理论最优策略。 |
| [^3] | [Soft-constrained Schrodinger Bridge: a Stochastic Control Approach](https://arxiv.org/abs/2403.01717) | 提出了软约束薛定谔桥(SSB)控制问题，在允许终端分布与预先指定分布不同的情况下，惩罚两者之间的Kullback-Leibler散度。理论上推导出了SSB解，显示最优控制过程的终端分布是μT和其他分布的几何混合，并将结果扩展到时间序列设置。 |
| [^4] | [Self-Consistent Conformal Prediction](https://arxiv.org/abs/2402.07307) | 自洽的符合预测方法能够提供既符合校准的预测又符合以模型预测的动作为条件的预测区间，为决策者提供了严格的、针对具体动作的决策保证。 |
| [^5] | [On diffusion-based generative models and their error bounds: The log-concave case with full convergence estimates](https://arxiv.org/abs/2311.13584) | 我们提出了对于基于扩散的生成模型在强对数凹数据分布假设下的完整收敛理论保证，获得了对于参数估计和采样算法的最优上限估计。 |
| [^6] | [Isolated pulsar population synthesis with simulation-based inference.](http://arxiv.org/abs/2312.14848) | 本论文使用模拟推断方法结合脉冲星种群合成，来限制孤立银河射电脉冲星的磁旋转特性。 |
| [^7] | [Uncovering ECG Changes during Healthy Aging using Explainable AI.](http://arxiv.org/abs/2310.07463) | 本文使用可解释的人工智能技术分析了健康个体的心电图数据，并识别出随年龄增长呼吸率的下降及SDANN值异常高作为老年人的指标。 |
| [^8] | [Transformer Fusion with Optimal Transport.](http://arxiv.org/abs/2310.05719) | 本文介绍了一种使用最优输运来融合基于Transformer的网络的方法，可以对齐各种架构组件并允许不同大小的模型的融合，提供了一种新的高效压缩Transformer的方式。 |
| [^9] | [Equation Discovery with Bayesian Spike-and-Slab Priors and Efficient Kernels.](http://arxiv.org/abs/2310.05387) | 该论文提出了一种基于核学习和贝叶斯Spike-and-Slab先验的方程式发现方法，通过核回归和贝叶斯稀疏分布，能够有效处理数据稀疏性和噪声问题，并进行不确定性量化和高效的后验推断和函数估计。 |
| [^10] | [Corrected generalized cross-validation for finite ensembles of penalized estimators.](http://arxiv.org/abs/2310.01374) | 本文研究了广义交叉验证（GCV）在有限惩罚估计器集合中估计预测风险的一致性问题，并提出了一种修正方法（CGCV）来解决这个问题。 |
| [^11] | [Simultaneous inference for generalized linear models with unmeasured confounders.](http://arxiv.org/abs/2309.07261) | 本文研究了存在混淆效应时的广义线性模型的大规模假设检验问题，并提出了一种利用正交结构和线性投影的统计估计和推断框架，解决了由于未测混淆因素引起的偏差问题。 |
| [^12] | [Classification with Deep Neural Networks and Logistic Loss.](http://arxiv.org/abs/2307.16792) | 本文提出了一种新颖的oracle型不等式，通过解决逻辑损失的目标函数无界性限制，推导出使用逻辑损失训练的全连接ReLU深度神经网络分类器的最优收敛速率，仅要求数据的条件类概率具有H\"older平滑性，并且考虑了组合假设，使得该方法具有更广泛的适用性。 |
| [^13] | [Communication-Efficient Federated Learning through Importance Sampling.](http://arxiv.org/abs/2306.12625) | 本文提出了一种通过重要性抽样实现有效通信的联邦学习方法，大大降低了发送模型更新的高通信成本，利用服务器端客户端分布和附加信息的接近关系，只需要较少的通信量即可实现。 |
| [^14] | [Differentiable Neural Networks with RePU Activation: with Applications to Score Estimation and Isotonic Regression.](http://arxiv.org/abs/2305.00608) | 该论文介绍了使用RePU激活函数的可微分神经网络，在近似$C^s$平滑函数及其导数的同时建立了下限误差界，并证明了其在降低维度灾难方面的能力，此外还提出了一种使用RePU网络的惩罚保序回归(PDIR)方法。 |
| [^15] | [Decentralized Online Regularized Learning Over Random Time-Varying Graphs.](http://arxiv.org/abs/2206.03861) | 本文研究了随机时变图上的分散在线正则化线性回归算法，提出了非负超-鞅不等式的估计误差，证明了算法在满足样本路径时空兴奋条件时，节点的估计可以收敛于未知的真实参数向量。 |

# 详细

[^1]: 重新思考对抗逆强化学习：从策略模仿和可转移奖励恢复的角度

    Rethinking Adversarial Inverse Reinforcement Learning: From the Angles of Policy Imitation and Transferable Reward Recovery

    [https://arxiv.org/abs/2403.14593](https://arxiv.org/abs/2403.14593)

    重新思考对抗逆强化学习中的策略模仿和可转移奖励恢复，提出了一个混合框架PPO-AIRL + SAC以解决SAC算法在AIRL训练中无法全面解开奖励函数的问题。

    

    对抗逆强化学习（AIRL）作为模仿学习中的基石方法。本文重新思考了AIRL的两个不同角度：策略模仿和可转移奖励恢复。我们从用Soft Actor-Critic（SAC）替换AIRL中的内置算法开始，以增强样本效率，这要归功于SAC的离策略形式和相对于AIRL而言可识别的马尔可夫决策过程（MDP）模型。这确实在策略模仿方面表现出显著的改进，但不慎给可转移奖励恢复带来了缺点。为了解决这个问题，我们阐述了SAC算法本身在AIRL训练过程中无法全面解开奖励函数，提出了一个混合框架，PPO-AIRL + SAC，以获得令人满意的转移效果。此外，我们分析了环境提取解开的奖励的能力。

    arXiv:2403.14593v1 Announce Type: new  Abstract: Adversarial inverse reinforcement learning (AIRL) stands as a cornerstone approach in imitation learning. This paper rethinks the two different angles of AIRL: policy imitation and transferable reward recovery. We begin with substituting the built-in algorithm in AIRL with soft actor-critic (SAC) during the policy optimization process to enhance sample efficiency, thanks to the off-policy formulation of SAC and identifiable Markov decision process (MDP) models with respect to AIRL. It indeed exhibits a significant improvement in policy imitation but accidentally brings drawbacks to transferable reward recovery. To learn this issue, we illustrate that the SAC algorithm itself is not feasible to disentangle the reward function comprehensively during the AIRL training process, and propose a hybrid framework, PPO-AIRL + SAC, for satisfactory transfer effect. Additionally, we analyze the capability of environments to extract disentangled rewa
    
[^2]: 基于高斯过程回归的机器学习系统可靠性分析

    Machine learning-based system reliability analysis with Gaussian Process Regression

    [https://arxiv.org/abs/2403.11125](https://arxiv.org/abs/2403.11125)

    本文提出了基于高斯过程回归的机器学习系统可靠性分析方法，并通过几个定理探讨了最优学习策略，包括考虑和忽略样本之间的相关性以及顺序多个训练样本增益的理论最优策略。

    

    arXiv:2403.11125v1 公告类型: 交叉 摘要: 基于机器学习的可靠性分析方法在计算效率和准确性方面取得了巨大进展。最近，已经提出许多有效的学习策略来增强计算性能。然而，其中很少有人探讨了理论上的最优学习策略。在这篇文章中，我们提出了几个定理来促进这种探索。具体来说，详细阐述了考虑和忽略候选设计样本之间相关性的情况。此外，我们证明了众所周知的 U 学习函数可以重新制定为在忽略 Kriging 相关性的情况下的最优学习函数。此外，还通过带有相应损失函数的贝叶斯估计数学上探讨了顺序多个训练样本增益的理论上最优学习策略。模拟结果表明最优学习策略……

    arXiv:2403.11125v1 Announce Type: cross  Abstract: Machine learning-based reliability analysis methods have shown great advancements for their computational efficiency and accuracy. Recently, many efficient learning strategies have been proposed to enhance the computational performance. However, few of them explores the theoretical optimal learning strategy. In this article, we propose several theorems that facilitates such exploration. Specifically, cases that considering and neglecting the correlations among the candidate design samples are well elaborated. Moreover, we prove that the well-known U learning function can be reformulated to the optimal learning function for the case neglecting the Kriging correlation. In addition, the theoretical optimal learning strategy for sequential multiple training samples enrichment is also mathematically explored through the Bayesian estimate with the corresponding lost functions. Simulation results show that the optimal learning strategy consid
    
[^3]: 软约束薛定谔桥：一种随机控制方法

    Soft-constrained Schrodinger Bridge: a Stochastic Control Approach

    [https://arxiv.org/abs/2403.01717](https://arxiv.org/abs/2403.01717)

    提出了软约束薛定谔桥(SSB)控制问题，在允许终端分布与预先指定分布不同的情况下，惩罚两者之间的Kullback-Leibler散度。理论上推导出了SSB解，显示最优控制过程的终端分布是μT和其他分布的几何混合，并将结果扩展到时间序列设置。

    

    薛定谔桥可以被视为一个连续时间的随机控制问题，目标是找到一个最优控制扩散过程，其具有预先指定的终端分布μT。我们提出通过允许终端分布与μT不同但惩罚两个分布之间的Kullback-Leibler散度来泛化这个随机控制问题。我们将这个新的控制问题称为软约束薛定谔桥(SSB)。这项工作的主要贡献是对SSB解的理论推导，表明最优控制过程的终端分布是μT和另一些分布的几何混合。这个结果进一步扩展到时间序列设置。SSB的一个应用是鲁棒生成扩散模型的开发。我们提出了一个基于分数匹配的算法来从几何混合中进行抽样，并展示了其用途

    arXiv:2403.01717v1 Announce Type: cross  Abstract: Schr\"{o}dinger bridge can be viewed as a continuous-time stochastic control problem where the goal is to find an optimally controlled diffusion process with a pre-specified terminal distribution $\mu_T$. We propose to generalize this stochastic control problem by allowing the terminal distribution to differ from $\mu_T$ but penalizing the Kullback-Leibler divergence between the two distributions. We call this new control problem soft-constrained Schr\"{o}dinger bridge (SSB). The main contribution of this work is a theoretical derivation of the solution to SSB, which shows that the terminal distribution of the optimally controlled process is a geometric mixture of $\mu_T$ and some other distribution. This result is further extended to a time series setting. One application of SSB is the development of robust generative diffusion models. We propose a score matching-based algorithm for sampling from geometric mixtures and showcase its us
    
[^4]: 自洽的符合预测

    Self-Consistent Conformal Prediction

    [https://arxiv.org/abs/2402.07307](https://arxiv.org/abs/2402.07307)

    自洽的符合预测方法能够提供既符合校准的预测又符合以模型预测的动作为条件的预测区间，为决策者提供了严格的、针对具体动作的决策保证。

    

    在机器学习指导下的决策中，决策者通常在具有相同预测结果的情境中采取相同的行动。符合预测帮助决策者量化动作的结果不确定性，从而实现更好的风险管理。受这种观点的启发，我们引入了自洽的符合预测，它产生了既符合Venn-Abers校准的预测，又符合以模型预测引发的动作为条件的符合预测区间。我们的方法可以后验地应用于任何黑盒预测器，提供严格的、针对具体动作的决策保证。数值实验表明，我们的方法在区间的效率和条件的有效性之间达到了平衡。

    In decision-making guided by machine learning, decision-makers often take identical actions in contexts with identical predicted outcomes. Conformal prediction helps decision-makers quantify outcome uncertainty for actions, allowing for better risk management. Inspired by this perspective, we introduce self-consistent conformal prediction, which yields both Venn-Abers calibrated predictions and conformal prediction intervals that are valid conditional on actions prompted by model predictions. Our procedure can be applied post-hoc to any black-box predictor to provide rigorous, action-specific decision-making guarantees. Numerical experiments show our approach strikes a balance between interval efficiency and conditional validity.
    
[^5]: 关于基于扩散的生成模型及其误差界限：完全收敛估计下的对数凹情况

    On diffusion-based generative models and their error bounds: The log-concave case with full convergence estimates

    [https://arxiv.org/abs/2311.13584](https://arxiv.org/abs/2311.13584)

    我们提出了对于基于扩散的生成模型在强对数凹数据分布假设下的完整收敛理论保证，获得了对于参数估计和采样算法的最优上限估计。

    

    我们在强对数凹数据分布的假设下为基于扩散的生成模型的收敛行为提供了完整的理论保证，而我们用于得分估计的逼近函数类由Lipschitz连续函数组成。我们通过一个激励性例子展示了我们方法的强大之处，即从具有未知均值的高斯分布中进行采样。在这种情况下，我们对相关的优化问题，即得分估计，提供了明确的估计，同时将其与相应的采样估计结合起来。因此，我们获得了最好的已知上限估计，涉及关键感兴趣的数量，如数据分布（具有未知均值的高斯分布）与我们的采样算法之间的Wasserstein-2距离的维度和收敛速率。

    arXiv:2311.13584v2 Announce Type: replace  Abstract: We provide full theoretical guarantees for the convergence behaviour of diffusion-based generative models under the assumption of strongly log-concave data distributions while our approximating class of functions used for score estimation is made of Lipschitz continuous functions. We demonstrate via a motivating example, sampling from a Gaussian distribution with unknown mean, the powerfulness of our approach. In this case, explicit estimates are provided for the associated optimization problem, i.e. score approximation, while these are combined with the corresponding sampling estimates. As a result, we obtain the best known upper bound estimates in terms of key quantities of interest, such as the dimension and rates of convergence, for the Wasserstein-2 distance between the data distribution (Gaussian with unknown mean) and our sampling algorithm.   Beyond the motivating example and in order to allow for the use of a diverse range o
    
[^6]: 用基于模拟推断的孤立脉冲星种群合成

    Isolated pulsar population synthesis with simulation-based inference. (arXiv:2312.14848v1 [astro-ph.HE] CROSS LISTED)

    [http://arxiv.org/abs/2312.14848](http://arxiv.org/abs/2312.14848)

    本论文使用模拟推断方法结合脉冲星种群合成，来限制孤立银河射电脉冲星的磁旋转特性。

    

    我们将脉冲星种群合成与基于模拟推断相结合，以限制孤立银河射电脉冲星的磁旋转特性。我们首先构建了一个灵活的框架来模拟中子星的诞生特性和演化，重点是它们的动力学、旋转和磁性特征。特别是，我们从对数正态分布中采样初始磁场强度B和自转周期P，并用幂律来捕捉后期磁场的衰减。每个对数正态分布由均值μlogB，μlogP和标准差σlogB，σlogP描述，而幂律由指数a_late描述，共计五个自由参数。然后我们模拟了星体的射电发射和观测偏差，以模拟三个射电调查中的探测，并通过改变输入参数产生了一个大型的合成P-Ṗ图数据库。接着我们采用基于模拟推断的方法进行推断

    We combine pulsar population synthesis with simulation-based inference to constrain the magneto-rotational properties of isolated Galactic radio pulsars. We first develop a flexible framework to model neutron-star birth properties and evolution, focusing on their dynamical, rotational and magnetic characteristics. In particular, we sample initial magnetic-field strengths, $B$, and spin periods, $P$, from log-normal distributions and capture the late-time magnetic-field decay with a power law. Each log-normal is described by a mean, $\mu_{\log B}, \mu_{\log P}$, and standard deviation, $\sigma_{\log B}, \sigma_{\log P}$, while the power law is characterized by the index, $a_{\rm late}$, resulting in five free parameters. We subsequently model the stars' radio emission and observational biases to mimic detections with three radio surveys, and produce a large database of synthetic $P$-$\dot{P}$ diagrams by varying our input parameters. We then follow a simulation-based inference approach 
    
[^7]: 使用可解释的人工智能揭示健康衰老过程中的心电图变化

    Uncovering ECG Changes during Healthy Aging using Explainable AI. (arXiv:2310.07463v1 [eess.SP])

    [http://arxiv.org/abs/2310.07463](http://arxiv.org/abs/2310.07463)

    本文使用可解释的人工智能技术分析了健康个体的心电图数据，并识别出随年龄增长呼吸率的下降及SDANN值异常高作为老年人的指标。

    

    心血管疾病仍然是全球领先的死因。这需要对心脏衰老过程有深入的了解，以诊断心血管健康状况的限制。传统上，对个体心电图（ECG）特征随年龄变化的分析提供了这些见解。然而，这些特征虽然有信息量，但可能掩盖了底层数据关系。在本文中，我们使用深度学习模型和基于树的模型分析来自健康个体的ECG数据，包括原始信号和ECG特征格式。然后，我们使用可解释的AI技术来识别对于区分年龄组别最有辨别力的ECG特征或原始信号特征。我们的分析与基于树的分类器揭示了随年龄增长呼吸率下降，并识别出SDANN值异常高作为老年人的指标，可将其与年轻人区分开来。

    Cardiovascular diseases remain the leading global cause of mortality. This necessitates a profound understanding of heart aging processes to diagnose constraints in cardiovascular fitness. Traditionally, most of such insights have been drawn from the analysis of electrocardiogram (ECG) feature changes of individuals as they age. However, these features, while informative, may potentially obscure underlying data relationships. In this paper, we employ a deep-learning model and a tree-based model to analyze ECG data from a robust dataset of healthy individuals across varying ages in both raw signals and ECG feature format. Explainable AI techniques are then used to identify ECG features or raw signal characteristics are most discriminative for distinguishing between age groups. Our analysis with tree-based classifiers reveal age-related declines in inferred breathing rates and identifies notably high SDANN values as indicative of elderly individuals, distinguishing them from younger adul
    
[^8]: 使用最优输运器合并Transformer

    Transformer Fusion with Optimal Transport. (arXiv:2310.05719v1 [cs.LG])

    [http://arxiv.org/abs/2310.05719](http://arxiv.org/abs/2310.05719)

    本文介绍了一种使用最优输运来融合基于Transformer的网络的方法，可以对齐各种架构组件并允许不同大小的模型的融合，提供了一种新的高效压缩Transformer的方式。

    

    融合是一种将多个独立训练的神经网络合并以结合它们的能力的技术。过去的尝试仅限于全连接、卷积和残差网络的情况。本文提出了一种系统的方法，利用最优输运来融合两个或多个基于Transformer的网络，以（软）对齐各种架构组件。我们详细描述了一种层对齐的抽象方法，可以推广到任意架构，例如多头自注意力、层归一化和残差连接。我们通过各种消融研究讨论了如何处理这些架构组件。此外，我们的方法允许不同大小的模型进行融合（异构融合），为Transformer的压缩提供了一种新的高效方法。我们通过Vision Transformer进行图像分类任务以及自然语言

    Fusion is a technique for merging multiple independently-trained neural networks in order to combine their capabilities. Past attempts have been restricted to the case of fully-connected, convolutional, and residual networks. In this paper, we present a systematic approach for fusing two or more transformer-based networks exploiting Optimal Transport to (soft-)align the various architectural components. We flesh out an abstraction for layer alignment, that can generalize to arbitrary architectures -- in principle -and we apply this to the key ingredients of Transformers such as multi-head self-attention, layer-normalization, and residual connections, and we discuss how to handle them via various ablation studies. Furthermore, our method allows the fusion of models of different sizes (heterogeneous fusion), providing a new and efficient way for compression of Transformers. The proposed approach is evaluated on both image classification tasks via Vision Transformer and natural language
    
[^9]: 基于贝叶斯Spike-and-Slab先验和高效核函数的方程式发现方法

    Equation Discovery with Bayesian Spike-and-Slab Priors and Efficient Kernels. (arXiv:2310.05387v1 [cs.LG])

    [http://arxiv.org/abs/2310.05387](http://arxiv.org/abs/2310.05387)

    该论文提出了一种基于核学习和贝叶斯Spike-and-Slab先验的方程式发现方法，通过核回归和贝叶斯稀疏分布，能够有效处理数据稀疏性和噪声问题，并进行不确定性量化和高效的后验推断和函数估计。

    

    从数据中发现控制方程对于许多科学和工程应用非常重要。然而，尽管有一些有希望的成功案例，现有方法仍然面临着数据稀疏性和噪声问题的挑战，这在实践中随处可见。此外，最先进的方法缺乏不确定性量化和/或训练成本高昂。为了克服这些局限性，我们提出了一种基于核学习和贝叶斯Spike-and-Slab先验（KBASS）的新型方程式发现方法。我们使用核回归来估计目标函数，这种方法具有灵活性、表达力，并且对于数据稀疏性和噪声更加稳健。我们将其与贝叶斯Spike-and-Slab先验结合使用，后者是一种理想的贝叶斯稀疏分布，用于有效的算子选择和不确定性量化。我们开发了一种基于期望传播期望最大化（EP-EM）算法的有效后验推断和函数估计方法。为了克服核回归的计算挑战，我们使用了一种快速方法。

    Discovering governing equations from data is important to many scientific and engineering applications. Despite promising successes, existing methods are still challenged by data sparsity as well as noise issues, both of which are ubiquitous in practice. Moreover, state-of-the-art methods lack uncertainty quantification and/or are costly in training. To overcome these limitations, we propose a novel equation discovery method based on Kernel learning and BAyesian Spike-and-Slab priors (KBASS). We use kernel regression to estimate the target function, which is flexible, expressive, and more robust to data sparsity and noises. We combine it with a Bayesian spike-and-slab prior -- an ideal Bayesian sparse distribution -- for effective operator selection and uncertainty quantification. We develop an expectation propagation expectation-maximization (EP-EM) algorithm for efficient posterior inference and function estimation. To overcome the computational challenge of kernel regression, we pla
    
[^10]: 有限惩罚估计器集合的修正广义交叉验证

    Corrected generalized cross-validation for finite ensembles of penalized estimators. (arXiv:2310.01374v1 [math.ST])

    [http://arxiv.org/abs/2310.01374](http://arxiv.org/abs/2310.01374)

    本文研究了广义交叉验证（GCV）在有限惩罚估计器集合中估计预测风险的一致性问题，并提出了一种修正方法（CGCV）来解决这个问题。

    

    广义交叉验证（GCV）是一种广泛使用的方法，用于估计在样本外进行预测的风险平方，并采用标量自由度调整（以乘法增加）来调整训练误差的平方。本文研究了GCV一致估计任意惩罚最小二乘估计器集合预测风险的能力。我们发现，对于任何大于一的有限大小的估计器集合，GCV是不一致的。为了弥补这个缺点，我们提出了一个纠正，它涉及到对每个集合成分的自由度调整训练误差的额外标量修正（以加法增加）。所提出的估计器（称为CGCV）保持了GCV的计算优势，既不需要样本分裂，模型重拟，也不需要包外风险估计。该估计器源自对集合风险分解的细致检查和该分解中各个成分的两种中间风险估计器。

    Generalized cross-validation (GCV) is a widely-used method for estimating the squared out-of-sample prediction risk that employs a scalar degrees of freedom adjustment (in a multiplicative sense) to the squared training error. In this paper, we examine the consistency of GCV for estimating the prediction risk of arbitrary ensembles of penalized least squares estimators. We show that GCV is inconsistent for any finite ensemble of size greater than one. Towards repairing this shortcoming, we identify a correction that involves an additional scalar correction (in an additive sense) based on degrees of freedom adjusted training errors from each ensemble component. The proposed estimator (termed CGCV) maintains the computational advantages of GCV and requires neither sample splitting, model refitting, or out-of-bag risk estimation. The estimator stems from a finer inspection of ensemble risk decomposition and two intermediate risk estimators for the components in this decomposition. We prov
    
[^11]: 具有未测混淆因素的广义线性模型的同时推断

    Simultaneous inference for generalized linear models with unmeasured confounders. (arXiv:2309.07261v1 [stat.ME])

    [http://arxiv.org/abs/2309.07261](http://arxiv.org/abs/2309.07261)

    本文研究了存在混淆效应时的广义线性模型的大规模假设检验问题，并提出了一种利用正交结构和线性投影的统计估计和推断框架，解决了由于未测混淆因素引起的偏差问题。

    

    在基因组研究中，常常进行成千上万个同时假设检验，以确定差异表达的基因。然而，由于存在未测混淆因素，许多标准统计方法可能存在严重的偏差。本文研究了存在混淆效应时的多元广义线性模型的大规模假设检验问题。在任意混淆机制下，我们提出了一个统一的统计估计和推断方法，利用正交结构并将线性投影整合到三个关键阶段中。首先，利用多元响应变量分离边际和不相关的混淆效应，恢复混淆系数的列空间。随后，利用$\ell_1$正则化进行稀疏性估计，并强加正交性限制于混淆系数，联合估计潜在因子和主要效应。最后，我们结合投影和加权偏差校正步骤。

    Tens of thousands of simultaneous hypothesis tests are routinely performed in genomic studies to identify differentially expressed genes. However, due to unmeasured confounders, many standard statistical approaches may be substantially biased. This paper investigates the large-scale hypothesis testing problem for multivariate generalized linear models in the presence of confounding effects. Under arbitrary confounding mechanisms, we propose a unified statistical estimation and inference framework that harnesses orthogonal structures and integrates linear projections into three key stages. It first leverages multivariate responses to separate marginal and uncorrelated confounding effects, recovering the confounding coefficients' column space. Subsequently, latent factors and primary effects are jointly estimated, utilizing $\ell_1$-regularization for sparsity while imposing orthogonality onto confounding coefficients. Finally, we incorporate projected and weighted bias-correction steps 
    
[^12]: 使用深度神经网络和逻辑损失进行分类

    Classification with Deep Neural Networks and Logistic Loss. (arXiv:2307.16792v1 [stat.ML])

    [http://arxiv.org/abs/2307.16792](http://arxiv.org/abs/2307.16792)

    本文提出了一种新颖的oracle型不等式，通过解决逻辑损失的目标函数无界性限制，推导出使用逻辑损失训练的全连接ReLU深度神经网络分类器的最优收敛速率，仅要求数据的条件类概率具有H\"older平滑性，并且考虑了组合假设，使得该方法具有更广泛的适用性。

    

    使用逻辑损失（即交叉熵损失）训练的深度神经网络在各种二分类任务中取得了显著的进展。然而，关于使用深度神经网络和逻辑损失进行二分类的泛化分析仍然很少。逻辑损失的目标函数的无界性是导致推导出令人满意的泛化界限的主要障碍。本文旨在通过建立一种新颖而优雅的oracle型不等式来填补这一空白，该不等式使我们能够处理目标函数的有界性限制，并利用它推导出使用逻辑损失训练的全连接ReLU深度神经网络分类器的收敛速率。特别地，我们仅需要数据的条件类概率$\eta$的H\"older平滑性，就可以获得最优的收敛速率（仅限于对数因子）。此外，我们考虑了一个组合假设，要求$\eta$是若干向量值函数的复合函数，其中每个向量值函数都是独立的。

    Deep neural networks (DNNs) trained with the logistic loss (i.e., the cross entropy loss) have made impressive advancements in various binary classification tasks. However, generalization analysis for binary classification with DNNs and logistic loss remains scarce. The unboundedness of the target function for the logistic loss is the main obstacle to deriving satisfying generalization bounds. In this paper, we aim to fill this gap by establishing a novel and elegant oracle-type inequality, which enables us to deal with the boundedness restriction of the target function, and using it to derive sharp convergence rates for fully connected ReLU DNN classifiers trained with logistic loss. In particular, we obtain optimal convergence rates (up to log factors) only requiring the H\"older smoothness of the conditional class probability $\eta$ of data. Moreover, we consider a compositional assumption that requires $\eta$ to be the composition of several vector-valued functions of which each co
    
[^13]: 通过重要性抽样实现有效通信的联邦学习

    Communication-Efficient Federated Learning through Importance Sampling. (arXiv:2306.12625v1 [cs.LG])

    [http://arxiv.org/abs/2306.12625](http://arxiv.org/abs/2306.12625)

    本文提出了一种通过重要性抽样实现有效通信的联邦学习方法，大大降低了发送模型更新的高通信成本，利用服务器端客户端分布和附加信息的接近关系，只需要较少的通信量即可实现。

    

    客户端向服务器发送模型更新的高通信成本是可扩展联邦学习（FL）的重要瓶颈。现有方法中，使用随机压缩方法实现了最先进的比特率-准确性折衷——其中客户端n发送来自仅为该客户端的概率分布qφ（n）的样本，服务器使用这些样本估计客户端分布的平均值。然而，这种方法没有充分利用FL的设置，其中服务器在整个训练过程中具有预数据分布pθ的附加信息，该分布与客户端分布qφ（n）在Kullback-Leibler（KL）发散方面接近。在本文中，我们利用服务器端客户端分布qφ（n)与附加信息pθ之间的这种接近关系，并提出了一种框架，该框架需要大约Dkl（qφ（n）|| pθ）位的通信量。

    The high communication cost of sending model updates from the clients to the server is a significant bottleneck for scalable federated learning (FL). Among existing approaches, state-of-the-art bitrate-accuracy tradeoffs have been achieved using stochastic compression methods -- in which the client $n$ sends a sample from a client-only probability distribution $q_{\phi^{(n)}}$, and the server estimates the mean of the clients' distributions using these samples. However, such methods do not take full advantage of the FL setup where the server, throughout the training process, has side information in the form of a pre-data distribution $p_{\theta}$ that is close to the client's distribution $q_{\phi^{(n)}}$ in Kullback-Leibler (KL) divergence. In this work, we exploit this closeness between the clients' distributions $q_{\phi^{(n)}}$'s and the side information $p_{\theta}$ at the server, and propose a framework that requires approximately $D_{KL}(q_{\phi^{(n)}}|| p_{\theta})$ bits of com
    
[^14]: 使用RePU激活函数的可微分神经网络：在得分估计和保序回归中的应用。

    Differentiable Neural Networks with RePU Activation: with Applications to Score Estimation and Isotonic Regression. (arXiv:2305.00608v1 [stat.ML])

    [http://arxiv.org/abs/2305.00608](http://arxiv.org/abs/2305.00608)

    该论文介绍了使用RePU激活函数的可微分神经网络，在近似$C^s$平滑函数及其导数的同时建立了下限误差界，并证明了其在降低维度灾难方面的能力，此外还提出了一种使用RePU网络的惩罚保序回归(PDIR)方法。

    

    我们研究了由修正后的幂单元（RePU）函数激活的可微分神经网络的属性。我们展示了RePU神经网络的偏导数可以由混合激活RePU网络来表示，并推导了导数RePU网络函数类的复杂度的上界。在使用RePU激活的深度神经网络中，我们建立了同时近似$C^s$平滑函数及其导数的误差界。此外，当数据具有近似低维支持时，我们推导出改进的逼近误差界，证明了RePU网络减缓维度灾难的能力。为了说明我们的结果的实用性，我们考虑了深度得分匹配估计器(DSME)，并提出了一种使用RePU网络的惩罚保序回归(PDIR)。我们在假定目标函数属于$C^s$平滑函数类的情况下为DSME和PDIR建立非渐近超额风险界。

    We study the properties of differentiable neural networks activated by rectified power unit (RePU) functions. We show that the partial derivatives of RePU neural networks can be represented by RePUs mixed-activated networks and derive upper bounds for the complexity of the function class of derivatives of RePUs networks. We establish error bounds for simultaneously approximating $C^s$ smooth functions and their derivatives using RePU-activated deep neural networks. Furthermore, we derive improved approximation error bounds when data has an approximate low-dimensional support, demonstrating the ability of RePU networks to mitigate the curse of dimensionality. To illustrate the usefulness of our results, we consider a deep score matching estimator (DSME) and propose a penalized deep isotonic regression (PDIR) using RePU networks. We establish non-asymptotic excess risk bounds for DSME and PDIR under the assumption that the target functions belong to a class of $C^s$ smooth functions. We 
    
[^15]: 随机时变图上的分散在线正则化学习

    Decentralized Online Regularized Learning Over Random Time-Varying Graphs. (arXiv:2206.03861v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.03861](http://arxiv.org/abs/2206.03861)

    本文研究了随机时变图上的分散在线正则化线性回归算法，提出了非负超-鞅不等式的估计误差，证明了算法在满足样本路径时空兴奋条件时，节点的估计可以收敛于未知的真实参数向量。

    

    本文研究了在随机时变图上的分散在线正则化线性回归算法。在每个时间步中，每个节点都运行一个在线估计算法，该算法包括创新项（处理自身新测量值）、共识项（加权平均自身及其邻居的估计，带有加性和乘性通信噪声）和正则化项（防止过度拟合）。不要求回归矩阵和图满足特殊的统计假设，如相互独立、时空独立或平稳性。我们发展了非负超-鞅不等式的估计误差，并证明了如果算法增益、图和回归矩阵共同满足样本路径时空兴奋条件，节点的估计几乎可以肯定地收敛于未知的真实参数向量。特别地，通过选择适当的算法增益，该条件成立。

    We study the decentralized online regularized linear regression algorithm over random time-varying graphs. At each time step, every node runs an online estimation algorithm consisting of an innovation term processing its own new measurement, a consensus term taking a weighted sum of estimations of its own and its neighbors with additive and multiplicative communication noises and a regularization term preventing over-fitting. It is not required that the regression matrices and graphs satisfy special statistical assumptions such as mutual independence, spatio-temporal independence or stationarity. We develop the nonnegative supermartingale inequality of the estimation error, and prove that the estimations of all nodes converge to the unknown true parameter vector almost surely if the algorithm gains, graphs and regression matrices jointly satisfy the sample path spatio-temporal persistence of excitation condition. Especially, this condition holds by choosing appropriate algorithm gains 
    

