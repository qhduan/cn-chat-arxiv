# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Discrete Latent Graph Generative Modeling with Diffusion Bridges](https://arxiv.org/abs/2403.16883) | GLAD是一个在离散潜在空间上操作的图生成模型，通过适应扩散桥结构学习其离散潜在空间的先验，避免了依赖于原始数据空间的分解，在图生成任务中表现出优越性。 |
| [^2] | [On the Asymptotic Mean Square Error Optimality of Diffusion Probabilistic Models](https://arxiv.org/abs/2403.02957) | 本论文通过严格证明一个特定的DPM去噪策略在大量扩散步数下收敛到均方误差最优条件均值估计器，突出了DPM由渐近最优的去噪器组成，同时具有强大生成器的独特视角。 |
| [^3] | [Fast Rates in Online Convex Optimization by Exploiting the Curvature of Feasible Sets](https://arxiv.org/abs/2402.12868) | 该论文提出了一种新的分析方法，通过利用可行集的曲率，在在线凸优化中实现了快速收敛速度。 |
| [^4] | [Predictive Uncertainty Quantification via Risk Decompositions for Strictly Proper Scoring Rules](https://arxiv.org/abs/2402.10727) | 通过引入风险分解和适当评分规则，我们提出了一个通用框架来量化预测不确定性的不同来源，并澄清了它们之间的关系。 |
| [^5] | [Combining Evidence Across Filtrations](https://arxiv.org/abs/2402.09698) | 这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。 |
| [^6] | [Exploration by Optimization with Hybrid Regularizers: Logarithmic Regret with Adversarial Robustness in Partial Monitoring](https://arxiv.org/abs/2402.08321) | 这篇论文介绍了一种在部分监测问题中探索优化的方法，通过使用混合正则化器可以提高在随机和对抗环境中的遗憾界限。 |
| [^7] | [How Uniform Random Weights Induce Non-uniform Bias: Typical Interpolating Neural Networks Generalize with Narrow Teachers](https://arxiv.org/abs/2402.06323) | 在插值神经网络中，均匀随机权重可以产生非均匀偏差，因此通常插值神经网络会与窄教师NN一样很好地泛化。 |
| [^8] | [On Calibration and Conformal Prediction of Deep Classifiers](https://arxiv.org/abs/2402.05806) | 本文研究了温度缩放对符合预测方法的影响，通过实证研究发现，校准对自适应C方法产生了有害的影响。 |
| [^9] | [Convergence Analysis for General Probability Flow ODEs of Diffusion Models in Wasserstein Distances](https://arxiv.org/abs/2401.17958) | 本文提供了在2-Wasserstein距离中的一般类概率流ODE抽样器的非渐近收敛性分析，假设得分估计准确。 |
| [^10] | [On diffusion-based generative models and their error bounds: The log-concave case with full convergence estimates](https://arxiv.org/abs/2311.13584) | 我们提出了对于基于扩散的生成模型在强对数凹数据分布假设下的完整收敛理论保证，获得了对于参数估计和采样算法的最优上限估计。 |
| [^11] | [qPOTS: Efficient batch multiobjective Bayesian optimization via Pareto optimal Thompson sampling.](http://arxiv.org/abs/2310.15788) | 提出了一种简单但有效的多目标贝叶斯优化方法，通过Thompson采样从GP Pareto前沿中选择新的候选者，避免了繁杂的获取函数优化步骤。 |
| [^12] | [Manifold Learning with Sparse Regularised Optimal Transport.](http://arxiv.org/abs/2307.09816) | 这篇论文介绍了一种利用稀疏正则最优传输进行流形学习的方法，该方法构建了一个稀疏自适应的亲和矩阵，并在连续极限下与拉普拉斯型算子一致。 |
| [^13] | [Random-Set Convolutional Neural Network (RS-CNN) for Epistemic Deep Learning.](http://arxiv.org/abs/2307.05772) | 这篇论文提出了一种新的随机集合卷积神经网络（RS-CNN）用于分类，通过预测信念函数而不是概率矢量集合，以表示模型的置信度和认识不确定性。基于认识论深度学习方法，该模型能够估计由有限训练集引起的认识不确定性。 |

# 详细

[^1]: 带扩散桥的离散潜在图生成建模

    Discrete Latent Graph Generative Modeling with Diffusion Bridges

    [https://arxiv.org/abs/2403.16883](https://arxiv.org/abs/2403.16883)

    GLAD是一个在离散潜在空间上操作的图生成模型，通过适应扩散桥结构学习其离散潜在空间的先验，避免了依赖于原始数据空间的分解，在图生成任务中表现出优越性。

    

    学习潜在空间中的图生成模型相比于在原始数据空间上操作的模型受到较少关注，迄今表现出的性能乏善可陈。我们提出了GLAD，一个潜在空间图生成模型。与大多数先前的潜在空间图生成模型不同，GLAD在保留图结构的离散性质方面运行，无需进行诸如潜在空间连续性等不自然的假设。我们通过将扩散桥调整到其结构，来学习我们离散潜在空间的先验。通过在适当构建的潜在空间上操作，我们避免依赖于常用于在原始数据空间操作的模型中的分解。我们在一系列图基准数据集上进行实验，明显展示了离散潜在空间的优越性，并取得了最先进的图生成性能，使GLA

    arXiv:2403.16883v1 Announce Type: new  Abstract: Learning graph generative models over latent spaces has received less attention compared to models that operate on the original data space and has so far demonstrated lacklustre performance. We present GLAD a latent space graph generative model. Unlike most previous latent space graph generative models, GLAD operates on a discrete latent space that preserves to a significant extent the discrete nature of the graph structures making no unnatural assumptions such as latent space continuity. We learn the prior of our discrete latent space by adapting diffusion bridges to its structure. By operating over an appropriately constructed latent space we avoid relying on decompositions that are often used in models that operate in the original data space. We present experiments on a series of graph benchmark datasets which clearly show the superiority of the discrete latent space and obtain state of the art graph generative performance, making GLA
    
[^2]: 关于扩散概率模型渐近均方误差最优性的研究

    On the Asymptotic Mean Square Error Optimality of Diffusion Probabilistic Models

    [https://arxiv.org/abs/2403.02957](https://arxiv.org/abs/2403.02957)

    本论文通过严格证明一个特定的DPM去噪策略在大量扩散步数下收敛到均方误差最优条件均值估计器，突出了DPM由渐近最优的去噪器组成，同时具有强大生成器的独特视角。

    

    最近，扩散概率模型（DPMs）在去噪任务中展现出巨大潜力。尽管它们在实际应用中很有用，但它们的理论理解存在明显的差距。本文通过严格证明特定DPM去噪策略在大量扩散步数下收敛到均方误差（MSE）最优条件均值估计器（CME），为该领域提供了新的理论见解。研究的基于DPM的去噪器在训练过程中与DPMs共享，但在训练后的逆推理过程中仅传递条件均值。我们强调了DPM由渐近最优的去噪器组成的独特视角，同时通过在逆过程中切换重新采样的方式继承了一个强大的生成器。通过数值结果验证了理论发现。

    arXiv:2403.02957v1 Announce Type: new  Abstract: Diffusion probabilistic models (DPMs) have recently shown great potential for denoising tasks. Despite their practical utility, there is a notable gap in their theoretical understanding. This paper contributes novel theoretical insights by rigorously proving the asymptotic convergence of a specific DPM denoising strategy to the mean square error (MSE)-optimal conditional mean estimator (CME) over a large number of diffusion steps. The studied DPM-based denoiser shares the training procedure of DPMs but distinguishes itself by forwarding only the conditional mean during the reverse inference process after training. We highlight the unique perspective that DPMs are composed of an asymptotically optimal denoiser while simultaneously inheriting a powerful generator by switching re-sampling in the reverse process on and off. The theoretical findings are validated by numerical results.
    
[^3]: 通过利用可行集的曲率，在在线凸优化中实现快速收敛速度

    Fast Rates in Online Convex Optimization by Exploiting the Curvature of Feasible Sets

    [https://arxiv.org/abs/2402.12868](https://arxiv.org/abs/2402.12868)

    该论文提出了一种新的分析方法，通过利用可行集的曲率，在在线凸优化中实现了快速收敛速度。

    

    在本文中，我们探讨了在线凸优化（OCO），介绍了一种通过利用可行集的曲率提供快速收敛速度的新分析。我们首先证明，如果最优决策位于可行集的边界上且基础损失函数的梯度非零，则算法在随机环境中可以达到$O(\rho \log T)$的遗憾上界。其中，$\rho > 0$是包含最优决策并围绕可行集的最小球体的半径。

    arXiv:2402.12868v1 Announce Type: new  Abstract: In this paper, we explore online convex optimization (OCO) and introduce a new analysis that provides fast rates by exploiting the curvature of feasible sets. In online linear optimization, it is known that if the average gradient of loss functions is larger than a certain value, the curvature of feasible sets can be exploited by the follow-the-leader (FTL) algorithm to achieve a logarithmic regret. This paper reveals that algorithms adaptive to the curvature of loss functions can also leverage the curvature of feasible sets. We first prove that if an optimal decision is on the boundary of a feasible set and the gradient of an underlying loss function is non-zero, then the algorithm achieves a regret upper bound of $O(\rho \log T)$ in stochastic environments. Here, $\rho > 0$ is the radius of the smallest sphere that includes the optimal decision and encloses the feasible set. Our approach, unlike existing ones, can work directly with co
    
[^4]: 通过风险分解实现严格适当评分规则的预测不确定性量化

    Predictive Uncertainty Quantification via Risk Decompositions for Strictly Proper Scoring Rules

    [https://arxiv.org/abs/2402.10727](https://arxiv.org/abs/2402.10727)

    通过引入风险分解和适当评分规则，我们提出了一个通用框架来量化预测不确定性的不同来源，并澄清了它们之间的关系。

    

    在各个领域的预测模型应用中，区分预测不确定性的来源至关重要。尽管提出了许多不确定性度量，但并没有严格的定义来解开它们。此外，不同不确定性量化措施之间的关系仍然有些不清晰。在这项工作中，我们引入了一个根植于统计推理的通用框架，不仅允许创建新的不确定性度量，还澄清了它们之间的相互关系。我们的方法利用统计风险来区分aleatoric和epistemic不确定性成分，并利用适当的评分规则对其进行量化。为了使其在实践中易于处理，我们提出了在这一框架中整合贝叶斯推理的想法，并讨论了所提近似的性质。

    arXiv:2402.10727v1 Announce Type: cross  Abstract: Distinguishing sources of predictive uncertainty is of crucial importance in the application of forecasting models across various domains. Despite the presence of a great variety of proposed uncertainty measures, there are no strict definitions to disentangle them. Furthermore, the relationship between different measures of uncertainty quantification remains somewhat unclear. In this work, we introduce a general framework, rooted in statistical reasoning, which not only allows the creation of new uncertainty measures but also clarifies their interrelations. Our approach leverages statistical risk to distinguish aleatoric and epistemic uncertainty components and utilizes proper scoring rules to quantify them. To make it practically tractable, we propose an idea to incorporate Bayesian reasoning into this framework and discuss the properties of the proposed approximation.
    
[^5]: 合并不同过滤器中的证据

    Combining Evidence Across Filtrations

    [https://arxiv.org/abs/2402.09698](https://arxiv.org/abs/2402.09698)

    这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。

    

    在任何时刻有效的顺序推理中，已知任何可接受的推理方法必须基于测试鞅和它们的组合广义化，称为e进程，它们是非负进程，其在任何任意停时的期望上界不超过一。e进程量化了针对复合零假设的一系列结果的累积证据。本文研究了使用不同信息集（即过滤器）计算的e进程的合并方法，针对一个零假设。尽管在相同过滤器上构建的e进程可以轻松地合并（例如，通过平均），但在不同过滤器上构建的e进程不能那么容易地合并，因为它们在较粗的过滤器中的有效性不能转换为在更细的过滤器中的有效性。我们讨论了文献中三个具体例子：可交换性测试，独立性测试等。

    arXiv:2402.09698v1 Announce Type: cross  Abstract: In anytime-valid sequential inference, it is known that any admissible inference procedure must be based on test martingales and their composite generalization, called e-processes, which are nonnegative processes whose expectation at any arbitrary stopping time is upper-bounded by one. An e-process quantifies the accumulated evidence against a composite null hypothesis over a sequence of outcomes. This paper studies methods for combining e-processes that are computed using different information sets, i.e., filtrations, for a null hypothesis. Even though e-processes constructed on the same filtration can be combined effortlessly (e.g., by averaging), e-processes constructed on different filtrations cannot be combined as easily because their validity in a coarser filtration does not translate to validity in a finer filtration. We discuss three concrete examples of such e-processes in the literature: exchangeability tests, independence te
    
[^6]: 使用混合正则化器的优化探索：在部分监测中具有对抗鲁棒性的对数遗憾

    Exploration by Optimization with Hybrid Regularizers: Logarithmic Regret with Adversarial Robustness in Partial Monitoring

    [https://arxiv.org/abs/2402.08321](https://arxiv.org/abs/2402.08321)

    这篇论文介绍了一种在部分监测问题中探索优化的方法，通过使用混合正则化器可以提高在随机和对抗环境中的遗憾界限。

    

    部分监测是一种具有有限观测的在线决策问题的通用框架。为了从这种有限观测中做出决策，需要找到一个适当的探索分布。最近，提出了一种用于此目的的强大方法，即通过优化进行探索（ExO），它利用追踪正则化最优方法，在广泛的在线决策问题中实现对抗环境下的最优界限。然而，在随机环境中纯粹应用ExO会显著降低遗憾界限。为了解决这个局部可观测游戏中的问题，我们首先建立了一个新颖的ExO与混合正则化器的框架和分析。这个发展使我们能够显著改进最佳双赢算法（BOBW）的现有遗憾界限，在随机和对抗环境中都实现了几乎最优的界限。特别地，我们得出了一个随机遗憾界限为$O(\sum_{a \neq a^*} k^2 m^2$

    Partial monitoring is a generic framework of online decision-making problems with limited observations. To make decisions from such limited observations, it is necessary to find an appropriate distribution for exploration. Recently, a powerful approach for this purpose, exploration by optimization (ExO), was proposed, which achieves the optimal bounds in adversarial environments with follow-the-regularized-leader for a wide range of online decision-making problems. However, a naive application of ExO in stochastic environments significantly degrades regret bounds. To resolve this problem in locally observable games, we first establish a novel framework and analysis for ExO with a hybrid regularizer. This development allows us to significantly improve the existing regret bounds of best-of-both-worlds (BOBW) algorithms, which achieves nearly optimal bounds both in stochastic and adversarial environments. In particular, we derive a stochastic regret bound of $O(\sum_{a \neq a^*} k^2 m^2 \
    
[^7]: 均匀随机权重如何引起不均匀偏差：典型插值神经网络与窄教师的普遍性

    How Uniform Random Weights Induce Non-uniform Bias: Typical Interpolating Neural Networks Generalize with Narrow Teachers

    [https://arxiv.org/abs/2402.06323](https://arxiv.org/abs/2402.06323)

    在插值神经网络中，均匀随机权重可以产生非均匀偏差，因此通常插值神经网络会与窄教师NN一样很好地泛化。

    

    背景。一个主要的理论难题是当神经网络被训练到零误差（即插值数据）时，为什么超参数化神经网络（NN）能够很好地泛化。通常，NN是使用随机梯度下降（SGD）或其变种之一训练的。然而，最近的实证研究检验了从看似均匀的参数先验中采样的随机NN对数据的泛化能力：该NN对训练集进行了完美分类。有趣的是，这样的NN样本通常像SGD训练的NN一样泛化良好。贡献。我们证明了如果存在与标签一致的窄“教师NN”，那么这样的随机NN插值器通常能很好地泛化。具体而言，我们证明了在NN参数化中的“平坦”先验通过NN结构中的冗余引入了丰富的NN函数先验。特别是，这会对较简单的函数产生偏向，这些函数需要较少的相关参数。

    Background. A main theoretical puzzle is why over-parameterized Neural Networks (NNs) generalize well when trained to zero loss (i.e., so they interpolate the data). Usually, the NN is trained with Stochastic Gradient Descent (SGD) or one of its variants. However, recent empirical work examined the generalization of a random NN that interpolates the data: the NN was sampled from a seemingly uniform prior over the parameters, conditioned on that the NN perfectly classifying the training set. Interestingly, such a NN sample typically generalized as well as SGD-trained NNs.   Contributions. We prove that such a random NN interpolator typically generalizes well if there exists an underlying narrow ``teacher NN" that agrees with the labels. Specifically, we show that such a `flat' prior over the NN parametrization induces a rich prior over the NN functions, due to the redundancy in the NN structure. In particular, this creates a bias towards simpler functions, which require less relevant pa
    
[^8]: 关于深度分类器的校准和符合预测研究

    On Calibration and Conformal Prediction of Deep Classifiers

    [https://arxiv.org/abs/2402.05806](https://arxiv.org/abs/2402.05806)

    本文研究了温度缩放对符合预测方法的影响，通过实证研究发现，校准对自适应C方法产生了有害的影响。

    

    在许多分类应用中，深度神经网络（DNN）基于分类器的预测需要伴随一些置信度指示。针对这个目标，有两种流行的后处理方法：1）校准：修改分类器的softmax值，使其最大值（与预测相关）更好地估计正确概率；和2）符合预测（CP）：设计一个基于softmax值的分数，从中产生一组预测，具有理论上保证正确类别边际覆盖的特性。尽管在实践中两种指示都可能是需要的，但到目前为止它们之间的相互作用尚未得到研究。为了填补这一空白，在本文中，我们研究了温度缩放，这是最常见的校准技术，对重要的CP方法的影响。我们首先进行了一项广泛的实证研究，其中显示了一些重要的洞察，其中包括令人惊讶的发现，即校准对流行的自适应C方法产生了有害的影响。

    In many classification applications, the prediction of a deep neural network (DNN) based classifier needs to be accompanied with some confidence indication. Two popular post-processing approaches for that aim are: 1) calibration: modifying the classifier's softmax values such that their maximum (associated with the prediction) better estimates the correctness probability; and 2) conformal prediction (CP): devising a score (based on the softmax values) from which a set of predictions with theoretically guaranteed marginal coverage of the correct class is produced. While in practice both types of indications can be desired, so far the interplay between them has not been investigated. Toward filling this gap, in this paper we study the effect of temperature scaling, arguably the most common calibration technique, on prominent CP methods. We start with an extensive empirical study that among other insights shows that, surprisingly, calibration has a detrimental effect on popular adaptive C
    
[^9]: 在Wasserstein距离中的扩散模型的一般概率流ODE的收敛性分析

    Convergence Analysis for General Probability Flow ODEs of Diffusion Models in Wasserstein Distances

    [https://arxiv.org/abs/2401.17958](https://arxiv.org/abs/2401.17958)

    本文提供了在2-Wasserstein距离中的一般类概率流ODE抽样器的非渐近收敛性分析，假设得分估计准确。

    

    基于概率流常微分方程（ODE）的基于得分的生成模型在各种应用中取得了显著的成功。虽然文献中提出了各种快速的基于ODE的抽样器并在实践中使用，但对概率流ODE的收敛性属性的理论理解仍然非常有限。在本文中，我们提供了适用于2-Wasserstein距离中的一般类概率流ODE抽样器的首个非渐近收敛性分析结果，假设准确的得分估计。接下来，我们考虑了各种示例，并确定了相应基于ODE的抽样器的迭代复杂度的结果。

    Score-based generative modeling with probability flow ordinary differential equations (ODEs) has achieved remarkable success in a variety of applications. While various fast ODE-based samplers have been proposed in the literature and employed in practice, the theoretical understandings about convergence properties of the probability flow ODE are still quite limited. In this paper, we provide the first non-asymptotic convergence analysis for a general class of probability flow ODE samplers in 2-Wasserstein distance, assuming accurate score estimates. We then consider various examples and establish results on the iteration complexity of the corresponding ODE-based samplers.
    
[^10]: 关于基于扩散的生成模型及其误差界限：完全收敛估计下的对数凹情况

    On diffusion-based generative models and their error bounds: The log-concave case with full convergence estimates

    [https://arxiv.org/abs/2311.13584](https://arxiv.org/abs/2311.13584)

    我们提出了对于基于扩散的生成模型在强对数凹数据分布假设下的完整收敛理论保证，获得了对于参数估计和采样算法的最优上限估计。

    

    我们在强对数凹数据分布的假设下为基于扩散的生成模型的收敛行为提供了完整的理论保证，而我们用于得分估计的逼近函数类由Lipschitz连续函数组成。我们通过一个激励性例子展示了我们方法的强大之处，即从具有未知均值的高斯分布中进行采样。在这种情况下，我们对相关的优化问题，即得分估计，提供了明确的估计，同时将其与相应的采样估计结合起来。因此，我们获得了最好的已知上限估计，涉及关键感兴趣的数量，如数据分布（具有未知均值的高斯分布）与我们的采样算法之间的Wasserstein-2距离的维度和收敛速率。

    arXiv:2311.13584v2 Announce Type: replace  Abstract: We provide full theoretical guarantees for the convergence behaviour of diffusion-based generative models under the assumption of strongly log-concave data distributions while our approximating class of functions used for score estimation is made of Lipschitz continuous functions. We demonstrate via a motivating example, sampling from a Gaussian distribution with unknown mean, the powerfulness of our approach. In this case, explicit estimates are provided for the associated optimization problem, i.e. score approximation, while these are combined with the corresponding sampling estimates. As a result, we obtain the best known upper bound estimates in terms of key quantities of interest, such as the dimension and rates of convergence, for the Wasserstein-2 distance between the data distribution (Gaussian with unknown mean) and our sampling algorithm.   Beyond the motivating example and in order to allow for the use of a diverse range o
    
[^11]: qPOTS: 高效的批量多目标贝叶斯优化算法

    qPOTS: Efficient batch multiobjective Bayesian optimization via Pareto optimal Thompson sampling. (arXiv:2310.15788v1 [math.OC])

    [http://arxiv.org/abs/2310.15788](http://arxiv.org/abs/2310.15788)

    提出了一种简单但有效的多目标贝叶斯优化方法，通过Thompson采样从GP Pareto前沿中选择新的候选者，避免了繁杂的获取函数优化步骤。

    

    传统进化方法在多目标优化中非常有效，但对目标进行大量查询可能不利于目标花费很多或者计算量很大的时候。用高斯过程（GP）替代物和贝叶斯优化（BO）来解决多目标优化是一种高效的方法。多目标贝叶斯优化(MOBO)涉及构建一个被优化用来获得新观察候选的获取函数。这个“内部”优化可能很困难，因为获取函数是非凸的，不可微的和/或者不出波，MOBO的成功在很大程度上依赖于这个内部优化。我们摒弃这个困难的获取函数优化步骤，提出一种简单但有效的基于Thompson采样的方法($q\texttt{POTS}$)，其中新的候选者是从通过求解一个更便宜的多个后验样本路径的GP Pareto前沿中选择的。

    Classical evolutionary approaches for multiobjective optimization are quite effective but incur a lot of queries to the objectives; this can be prohibitive when objectives are expensive oracles. A sample-efficient approach to solving multiobjective optimization is via Gaussian process (GP) surrogates and Bayesian optimization (BO). Multiobjective Bayesian optimization (MOBO) involves the construction of an acquisition function which is optimized to acquire new observation candidates. This ``inner'' optimization can be hard due to various reasons: acquisition functions being nonconvex, nondifferentiable and/or unavailable in analytical form; the success of MOBO heavily relies on this inner optimization. We do away with this hard acquisition function optimization step and propose a simple, but effective, Thompson sampling based approach ($q\texttt{POTS}$) where new candidate(s) are chosen from the Pareto frontier of random GP posterior sample paths obtained by solving a much cheaper mult
    
[^12]: 用稀疏正则最优传输进行流形学习

    Manifold Learning with Sparse Regularised Optimal Transport. (arXiv:2307.09816v1 [stat.ML])

    [http://arxiv.org/abs/2307.09816](http://arxiv.org/abs/2307.09816)

    这篇论文介绍了一种利用稀疏正则最优传输进行流形学习的方法，该方法构建了一个稀疏自适应的亲和矩阵，并在连续极限下与拉普拉斯型算子一致。

    

    流形学习是现代统计学和数据科学中的一个核心任务。许多数据集（细胞、文档、图像、分子）可以被表示为嵌入在高维环境空间中的点云，然而数据固有的自由度通常远远少于环境维度的数量。检测数据嵌入的潜在流形是许多下游分析的先决条件。现实世界的数据集经常受到噪声观测和抽样的影响，因此提取关于潜在流形的信息是一个重大挑战。我们提出了一种利用对称版本的最优传输和二次正则化的流形学习方法，它构建了一个稀疏自适应的亲和矩阵，可以解释为双随机核归一化的推广。我们证明了在连续极限下产生的核与拉普拉斯型算子一致，并建立了该方法的健壮性。

    Manifold learning is a central task in modern statistics and data science. Many datasets (cells, documents, images, molecules) can be represented as point clouds embedded in a high dimensional ambient space, however the degrees of freedom intrinsic to the data are usually far fewer than the number of ambient dimensions. The task of detecting a latent manifold along which the data are embedded is a prerequisite for a wide family of downstream analyses. Real-world datasets are subject to noisy observations and sampling, so that distilling information about the underlying manifold is a major challenge. We propose a method for manifold learning that utilises a symmetric version of optimal transport with a quadratic regularisation that constructs a sparse and adaptive affinity matrix, that can be interpreted as a generalisation of the bistochastic kernel normalisation. We prove that the resulting kernel is consistent with a Laplace-type operator in the continuous limit, establish robustness
    
[^13]: 随机集合卷积神经网络（RS-CNN）用于认识论深度学习

    Random-Set Convolutional Neural Network (RS-CNN) for Epistemic Deep Learning. (arXiv:2307.05772v1 [cs.LG])

    [http://arxiv.org/abs/2307.05772](http://arxiv.org/abs/2307.05772)

    这篇论文提出了一种新的随机集合卷积神经网络（RS-CNN）用于分类，通过预测信念函数而不是概率矢量集合，以表示模型的置信度和认识不确定性。基于认识论深度学习方法，该模型能够估计由有限训练集引起的认识不确定性。

    

    机器学习越来越多地应用于安全关键领域，对抗攻击的鲁棒性至关重要，错误的预测可能导致潜在的灾难性后果。这突出了学习系统需要能够确定模型对其预测的置信度以及与之相关联的认识不确定性的手段，“知道一个模型不知道”。在本文中，我们提出了一种新颖的用于分类的随机集合卷积神经网络（RS-CNN），其预测信念函数而不是概率矢量集合，使用随机集合的数学，即对样本空间的幂集的分布。基于认识论深度学习方法，随机集模型能够表示机器学习中由有限训练集引起的“认识性”不确定性。我们通过近似预测信念函数相关联的置信集的大小来估计认识不确定性。

    Machine learning is increasingly deployed in safety-critical domains where robustness against adversarial attacks is crucial and erroneous predictions could lead to potentially catastrophic consequences. This highlights the need for learning systems to be equipped with the means to determine a model's confidence in its prediction and the epistemic uncertainty associated with it, 'to know when a model does not know'. In this paper, we propose a novel Random-Set Convolutional Neural Network (RS-CNN) for classification which predicts belief functions rather than probability vectors over the set of classes, using the mathematics of random sets, i.e., distributions over the power set of the sample space. Based on the epistemic deep learning approach, random-set models are capable of representing the 'epistemic' uncertainty induced in machine learning by limited training sets. We estimate epistemic uncertainty by approximating the size of credal sets associated with the predicted belief func
    

