# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion models for probabilistic programming.](http://arxiv.org/abs/2311.00474) | 我们提出了一种新的扩散模型变分推断（DMVI）方法，用于在概率编程语言中进行自动近似推断。DMVI可以更准确地进行后验推断，而且易于实现和使用，对神经网络模型没有任何约束。 |
| [^2] | [A General Theoretical Paradigm to Understand Learning from Human Preferences.](http://arxiv.org/abs/2310.12036) | 本文研究了学习从人类偏好中学习的实际算法的理论基础，推导出一个新的一般目标，绕过了两个重要的近似。这种方法允许直接从收集的数据中学习策略而无需奖励模型的训练。 |
| [^3] | [Droplets of Good Representations: Grokking as a First Order Phase Transition in Two Layer Networks.](http://arxiv.org/abs/2310.03789) | 本研究将自适应核方法应用于两个师生模型，预测了特征学习和 Grokking 的性质，并展示了 Grokking 与相变理论之间的映射关系。 |
| [^4] | [Efficient Algorithms for the CCA Family: Unconstrained Objectives with Unbiased Gradients.](http://arxiv.org/abs/2310.01012) | 本论文提出了一个新颖的无约束目标，通过应用随机梯度下降（SGD）到CCA目标，实现了一系列快速算法，包括随机PLS、随机CCA和深度CCA。这些方法在各种基准测试中表现出比先前最先进方法更快的收敛速度和更高的相关性恢复。 |
| [^5] | [Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting.](http://arxiv.org/abs/2307.11494) | 本研究提出了一种面向概率时间序列预测的自引导扩散模型，称为TSDiff。该模型不需要辅助网络或训练过程的改变，在预测、改进和合成数据生成等时间序列任务上展现出了竞争力。 |
| [^6] | [Stability and Generalization of Stochastic Compositional Gradient Descent Algorithms.](http://arxiv.org/abs/2307.03357) | 本文通过统计学习理论的算法稳定性，分析了随机组合梯度下降算法的稳定性和泛化性，引入了组合一致稳定性概念并与SCO问题的泛化性建立了定量关系。 |
| [^7] | [Why Shallow Networks Struggle with Approximating and Learning High Frequency: A Numerical Study.](http://arxiv.org/abs/2306.17301) | 本文通过数值研究探讨了浅层神经网络在逼近和学习高频率方面的困难，重点是通过分析激活函数的谱分析来理解问题的原因。 |
| [^8] | [Differentially Private Wireless Federated Learning Using Orthogonal Sequences.](http://arxiv.org/abs/2306.08280) | 本文提出了一种使用正交序列的FLORAS方法，可消除发送端的信道状态信息，同时提供了项目级和客户级的差分隐私保证。FLORAS可以灵活地实现不同的差分隐私等级，并且通过推导收敛界限，实现了收敛速度和隐私保证之间的平稳权衡。 |
| [^9] | [Hinge-Wasserstein: Mitigating Overconfidence in Regression by Classification.](http://arxiv.org/abs/2306.00560) | 该论文提出了一种基于Wasserstein距离的损失函数hinge-Wasserstein，用于缓解回归任务中由于过度自信导致的不确定性问题。这种损失函数有效提高了aleatoric和epistemic不确定性的质量。 |

# 详细

[^1]: 概率编程的扩散模型

    Diffusion models for probabilistic programming. (arXiv:2311.00474v1 [cs.LG])

    [http://arxiv.org/abs/2311.00474](http://arxiv.org/abs/2311.00474)

    我们提出了一种新的扩散模型变分推断（DMVI）方法，用于在概率编程语言中进行自动近似推断。DMVI可以更准确地进行后验推断，而且易于实现和使用，对神经网络模型没有任何约束。

    

    我们提出了扩散模型变分推断（DMVI），这是一种在概率编程语言（PPL）中进行自动近似推断的新方法。DMVI利用扩散模型作为对真实后验分布的变分近似，通过导出贝叶斯建模中使用的边际似然目标的新约束。DMVI易于实现，在PPL中进行无障碍推断，不像使用归一化流的变分推断那样具有缺点，并且对基础神经网络模型不做任何约束。我们在一组常见的贝叶斯模型上评估了DMVI，并表明它的后验推断一般比PPL中使用的现代方法更准确，同时具有类似的计算成本并且需要较少的手动调整。

    We propose Diffusion Model Variational Inference (DMVI), a novel method for automated approximate inference in probabilistic programming languages (PPLs). DMVI utilizes diffusion models as variational approximations to the true posterior distribution by deriving a novel bound to the marginal likelihood objective used in Bayesian modelling. DMVI is easy to implement, allows hassle-free inference in PPLs without the drawbacks of, e.g., variational inference using normalizing flows, and does not make any constraints on the underlying neural network model. We evaluate DMVI on a set of common Bayesian models and show that its posterior inferences are in general more accurate than those of contemporary methods used in PPLs while having a similar computational cost and requiring less manual tuning.
    
[^2]: 一个理论框架来理解从人类偏好中学习的一般方法

    A General Theoretical Paradigm to Understand Learning from Human Preferences. (arXiv:2310.12036v1 [cs.AI])

    [http://arxiv.org/abs/2310.12036](http://arxiv.org/abs/2310.12036)

    本文研究了学习从人类偏好中学习的实际算法的理论基础，推导出一个新的一般目标，绕过了两个重要的近似。这种方法允许直接从收集的数据中学习策略而无需奖励模型的训练。

    

    目前从人类偏好中学习的流行方法依赖于两个重要的近似：第一假设可以用逐点奖励替代成对偏好。第二个假设是在这些逐点奖励上训练的奖励模型可以从收集到的数据泛化到策略采样的超出分布的数据。最近，提出了一种称为直接偏好优化(DPO)的方法，该方法绕过了第二个近似，并直接从收集的数据中学习策略而无需奖励模型阶段。然而，这种方法仍然严重依赖于第一个近似。在本文中，我们试图对这些实际算法进行更深入的理论理解。特别地，我们推导出了一个新的一般目标，称为ΨPO，用于从人类偏好中学习，该目标以成对偏好的形式表达，因此绕过了这两个近似。这个新的一般目标使我们能够进行一种新的从训练数据直接学习策略的方法而无需进行奖励模型的训练。

    The prevalent deployment of learning from human preferences through reinforcement learning (RLHF) relies on two important approximations: the first assumes that pairwise preferences can be substituted with pointwise rewards. The second assumes that a reward model trained on these pointwise rewards can generalize from collected data to out-of-distribution data sampled by the policy. Recently, Direct Preference Optimisation (DPO) has been proposed as an approach that bypasses the second approximation and learn directly a policy from collected data without the reward modelling stage. However, this method still heavily relies on the first approximation.  In this paper we try to gain a deeper theoretical understanding of these practical algorithms. In particular we derive a new general objective called $\Psi$PO for learning from human preferences that is expressed in terms of pairwise preferences and therefore bypasses both approximations. This new general objective allows us to perform an 
    
[^3]: 好表示的液滴：在两层网络中 grokking 作为一阶相变

    Droplets of Good Representations: Grokking as a First Order Phase Transition in Two Layer Networks. (arXiv:2310.03789v1 [stat.ML])

    [http://arxiv.org/abs/2310.03789](http://arxiv.org/abs/2310.03789)

    本研究将自适应核方法应用于两个师生模型，预测了特征学习和 Grokking 的性质，并展示了 Grokking 与相变理论之间的映射关系。

    

    深度神经网络 (DNN) 的一个关键特性是在训练过程中能够学习新的特征。这种深度学习的有趣方面在最近报道的 Grokking 现象中表现得最为明显。虽然主要体现为测试准确性的突变增加，但 Grokking 也被认为是一种超越懒惰学习/高斯过程 (GP) 的现象，涉及特征学习。在这里，我们将特征学习理论的最新发展，自适应核方法，应用于具有立方多项式和模加法教师的两个师生模型。我们在这些模型上提供了关于特征学习和 Grokking 性质的分析预测，并展示了 Grokking 与相变理论之间的映射关系。我们表明，在 Grokking 之后，DNN 的状态类似于一阶相变后的混合相。在这个混合相中，DNN 生成了与之前明显不同的教师的有用内部表示。

    A key property of deep neural networks (DNNs) is their ability to learn new features during training. This intriguing aspect of deep learning stands out most clearly in recently reported Grokking phenomena. While mainly reflected as a sudden increase in test accuracy, Grokking is also believed to be a beyond lazy-learning/Gaussian Process (GP) phenomenon involving feature learning. Here we apply a recent development in the theory of feature learning, the adaptive kernel approach, to two teacher-student models with cubic-polynomial and modular addition teachers. We provide analytical predictions on feature learning and Grokking properties of these models and demonstrate a mapping between Grokking and the theory of phase transitions. We show that after Grokking, the state of the DNN is analogous to the mixed phase following a first-order phase transition. In this mixed phase, the DNN generates useful internal representations of the teacher that are sharply distinct from those before the 
    
[^4]: CCA家族的高效算法：无约束目标与无偏梯度

    Efficient Algorithms for the CCA Family: Unconstrained Objectives with Unbiased Gradients. (arXiv:2310.01012v1 [cs.LG])

    [http://arxiv.org/abs/2310.01012](http://arxiv.org/abs/2310.01012)

    本论文提出了一个新颖的无约束目标，通过应用随机梯度下降（SGD）到CCA目标，实现了一系列快速算法，包括随机PLS、随机CCA和深度CCA。这些方法在各种基准测试中表现出比先前最先进方法更快的收敛速度和更高的相关性恢复。

    

    典型相关分析（CCA）方法在多视角学习中具有基础性作用。正则化线性CCA方法可以看作是偏最小二乘（PLS）的推广，并与广义特征值问题（GEP）框架统一。然而，这些线性方法的传统算法在大规模数据上计算上是不可行的。深度CCA的扩展显示出很大的潜力，但目前的训练过程缓慢且复杂。我们首先提出了一个描述GEPs的顶级子空间的新颖无约束目标。我们的核心贡献是一系列快速算法，用随机梯度下降（SGD）应用于相应的CCA目标，从而获得随机PLS、随机CCA和深度CCA。这些方法在所有标准CCA和深度CCA基准测试中显示出比先前最先进方法更快的收敛速度和更高的相关性恢复。这样的速度使我们能够首次进行大规模生物数据的PLS分析。

    The Canonical Correlation Analysis (CCA) family of methods is foundational in multi-view learning. Regularised linear CCA methods can be seen to generalise Partial Least Squares (PLS) and unified with a Generalized Eigenvalue Problem (GEP) framework. However, classical algorithms for these linear methods are computationally infeasible for large-scale data. Extensions to Deep CCA show great promise, but current training procedures are slow and complicated. First we propose a novel unconstrained objective that characterizes the top subspace of GEPs. Our core contribution is a family of fast algorithms for stochastic PLS, stochastic CCA, and Deep CCA, simply obtained by applying stochastic gradient descent (SGD) to the corresponding CCA objectives. These methods show far faster convergence and recover higher correlations than the previous state-of-the-art on all standard CCA and Deep CCA benchmarks. This speed allows us to perform a first-of-its-kind PLS analysis of an extremely large bio
    
[^5]: 预测、改进、合成：面向概率时间序列预测的自引导扩散模型

    Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting. (arXiv:2307.11494v1 [cs.LG])

    [http://arxiv.org/abs/2307.11494](http://arxiv.org/abs/2307.11494)

    本研究提出了一种面向概率时间序列预测的自引导扩散模型，称为TSDiff。该模型不需要辅助网络或训练过程的改变，在预测、改进和合成数据生成等时间序列任务上展现出了竞争力。

    

    扩散模型在各个领域的生成建模任务中取得了最先进的性能。之前关于时间序列扩散模型的研究主要集中在开发针对特定预测或填补任务的条件模型。在这项工作中，我们探索了面向多种时间序列应用的任务不可知条件下的扩散模型的潜力。我们提出了TSDiff，一种面向时间序列的无条件训练的扩散模型。我们的自引导机制在推理过程中使得TSDiff能够为下游任务进行条件设置，而无需辅助网络或改变训练过程。我们在三个不同的时间序列任务上展示了我们方法的有效性：预测、改进和合成数据生成。首先，我们表明TSDiff与几种任务特定的条件预测方法相竞争（预测）。其次，我们利用TSDiff学到的隐性概率密度来迭代地改进p

    Diffusion models have achieved state-of-the-art performance in generative modeling tasks across various domains. Prior works on time series diffusion models have primarily focused on developing conditional models tailored to specific forecasting or imputation tasks. In this work, we explore the potential of task-agnostic, unconditional diffusion models for several time series applications. We propose TSDiff, an unconditionally trained diffusion model for time series. Our proposed self-guidance mechanism enables conditioning TSDiff for downstream tasks during inference, without requiring auxiliary networks or altering the training procedure. We demonstrate the effectiveness of our method on three different time series tasks: forecasting, refinement, and synthetic data generation. First, we show that TSDiff is competitive with several task-specific conditional forecasting methods (predict). Second, we leverage the learned implicit probability density of TSDiff to iteratively refine the p
    
[^6]: 随机组合梯度下降算法的稳定性和泛化

    Stability and Generalization of Stochastic Compositional Gradient Descent Algorithms. (arXiv:2307.03357v1 [cs.LG])

    [http://arxiv.org/abs/2307.03357](http://arxiv.org/abs/2307.03357)

    本文通过统计学习理论的算法稳定性，分析了随机组合梯度下降算法的稳定性和泛化性，引入了组合一致稳定性概念并与SCO问题的泛化性建立了定量关系。

    

    许多机器学习任务可以被形式化为随机组合优化（SCO）问题，例如强化学习、AUC最大化和元学习，其中目标函数涉及与期望相关的嵌套组合。虽然已经有大量研究致力于研究SCO算法的收敛行为，但对于它们的泛化性能如何，即从训练示例构建的学习算法在未来的测试示例上的行为如何，却很少有研究。在本文中，我们通过统计学习理论框架下的算法稳定性，提供了随机组合梯度下降算法的稳定性和泛化性分析。首先，我们引入了一种稳定性概念，称为组合一致稳定性，并建立了它与SCO问题的泛化性之间的定量关系。然后，我们为两种流行的随机组合优化问题建立了组合一致稳定性结果。

    Many machine learning tasks can be formulated as a stochastic compositional optimization (SCO) problem such as reinforcement learning, AUC maximization, and meta-learning, where the objective function involves a nested composition associated with an expectation. While a significant amount of studies has been devoted to studying the convergence behavior of SCO algorithms, there is little work on understanding their generalization, i.e., how these learning algorithms built from training examples would behave on future test examples. In this paper, we provide the stability and generalization analysis of stochastic compositional gradient descent algorithms through the lens of algorithmic stability in the framework of statistical learning theory. Firstly, we introduce a stability concept called compositional uniform stability and establish its quantitative relation with generalization for SCO problems. Then, we establish the compositional uniform stability results for two popular stochastic
    
[^7]: 浅层网络在逼近和学习高频率方面的困难：一个数值研究

    Why Shallow Networks Struggle with Approximating and Learning High Frequency: A Numerical Study. (arXiv:2306.17301v1 [cs.LG])

    [http://arxiv.org/abs/2306.17301](http://arxiv.org/abs/2306.17301)

    本文通过数值研究探讨了浅层神经网络在逼近和学习高频率方面的困难，重点是通过分析激活函数的谱分析来理解问题的原因。

    

    本研究通过对分析和实验的综合数值研究，解释了为什么两层神经网络在机器精度和计算成本等实际因素中，处理高频率的逼近和学习存在困难。具体而言，研究了以下基本计算问题：（1）在有限的机器精度下可以达到的最佳精度，（2）实现给定精度所需的计算成本，以及（3）对扰动的稳定性。研究的关键是相应激活函数的格拉姆矩阵的谱分析，该分析还显示了激活函数属性在这个问题中的作用。

    In this work, a comprehensive numerical study involving analysis and experiments shows why a two-layer neural network has difficulties handling high frequencies in approximation and learning when machine precision and computation cost are important factors in real practice. In particular, the following fundamental computational issues are investigated: (1) the best accuracy one can achieve given a finite machine precision, (2) the computation cost to achieve a given accuracy, and (3) stability with respect to perturbations. The key to the study is the spectral analysis of the corresponding Gram matrix of the activation functions which also shows how the properties of the activation function play a role in the picture.
    
[^8]: 使用正交序列的差分隐私无线联合学习方法

    Differentially Private Wireless Federated Learning Using Orthogonal Sequences. (arXiv:2306.08280v1 [cs.IT])

    [http://arxiv.org/abs/2306.08280](http://arxiv.org/abs/2306.08280)

    本文提出了一种使用正交序列的FLORAS方法，可消除发送端的信道状态信息，同时提供了项目级和客户级的差分隐私保证。FLORAS可以灵活地实现不同的差分隐私等级，并且通过推导收敛界限，实现了收敛速度和隐私保证之间的平稳权衡。

    

    本文提出了一种新的隐私保护上行空中计算方法FLORAS，用于单输入单输出（SISO）无线联合学习（FL）系统。FLORAS从通信设计的角度出发，利用正交序列的性质消除了发送端的信道状态信息（CSIT）要求。从隐私保护的角度来看，我们证明FLORAS可以提供项目级和客户级差分隐私（DP）保证。此外，通过调整系统参数，FLORAS可以在不增加成本的情况下灵活地实现不同的DP等级。我们推导出了一个新的FL收敛界限，结合隐私保证，可以在收敛速度和差分隐私级别之间实现平稳的权衡。数值结果证明了FLORAS相对于基准AirComp方法的优势，并验证了我们的分析结果可以指导不同权衡条件下的隐私保护FL的设计。

    We propose a novel privacy-preserving uplink over-the-air computation (AirComp) method, termed FLORAS, for single-input single-output (SISO) wireless federated learning (FL) systems. From the communication design perspective, FLORAS eliminates the requirement of channel state information at the transmitters (CSIT) by leveraging the properties of orthogonal sequences. From the privacy perspective, we prove that FLORAS can offer both item-level and client-level differential privacy (DP) guarantees. Moreover, by adjusting the system parameters, FLORAS can flexibly achieve different DP levels at no additional cost. A novel FL convergence bound is derived which, combined with the privacy guarantees, allows for a smooth tradeoff between convergence rate and differential privacy levels. Numerical results demonstrate the advantages of FLORAS compared with the baseline AirComp method, and validate that our analytical results can guide the design of privacy-preserving FL with different tradeoff 
    
[^9]: Hinge-Wasserstein: 通过分类避免回归中的过度自信

    Hinge-Wasserstein: Mitigating Overconfidence in Regression by Classification. (arXiv:2306.00560v1 [cs.LG])

    [http://arxiv.org/abs/2306.00560](http://arxiv.org/abs/2306.00560)

    该论文提出了一种基于Wasserstein距离的损失函数hinge-Wasserstein，用于缓解回归任务中由于过度自信导致的不确定性问题。这种损失函数有效提高了aleatoric和epistemic不确定性的质量。

    

    现代深度神经网络在性能方面得到了巨大的提高，但它们容易产生过度自信。在模糊甚至不可预测的现实世界场景中，这种过度自信可能对应用程序的安全性构成重大风险。针对回归任务，采用回归-分类方法有潜力缓解这些歧义，因为它可以预测所需输出的离散概率密度。然而，密度估计仍然倾向于过度自信，尤其是在使用常见的NLL损失函数训练时。为了缓解这种过度自信的问题，我们提出了一种基于Wasserstein距离的损失函数，即hinge-Wasserstein。与以前的工作相比，此损失显着提高了两种不确定性的质量： aleatoric不确定性和epistemic不确定性。我们在合成数据集上展示了新损失的能力，其中两种类型的不确定性可以分别控制。此外，作为现实世界场景的演示，我们在基准数据集上评估了我们的方法。

    Modern deep neural networks are prone to being overconfident despite their drastically improved performance. In ambiguous or even unpredictable real-world scenarios, this overconfidence can pose a major risk to the safety of applications. For regression tasks, the regression-by-classification approach has the potential to alleviate these ambiguities by instead predicting a discrete probability density over the desired output. However, a density estimator still tends to be overconfident when trained with the common NLL loss. To mitigate the overconfidence problem, we propose a loss function, hinge-Wasserstein, based on the Wasserstein Distance. This loss significantly improves the quality of both aleatoric and epistemic uncertainty, compared to previous work. We demonstrate the capabilities of the new loss on a synthetic dataset, where both types of uncertainty are controlled separately. Moreover, as a demonstration for real-world scenarios, we evaluate our approach on the benchmark dat
    

