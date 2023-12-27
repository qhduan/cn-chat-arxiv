# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Asymptotically free sketched ridge ensembles: Risks, cross-validation, and tuning.](http://arxiv.org/abs/2310.04357) | 该论文利用随机矩阵理论建立了一致性估计方法，用于估计素描岭回归集合的预测风险，从而实现了正则化和素描参数的高效一致调整。 |
| [^2] | [Beta Diffusion.](http://arxiv.org/abs/2309.07867) | beta扩散是一种新型生成模型方法，通过引入去掩盖和去噪的技术，利用缩放和偏移的beta分布进行乘法转换，实现在有界范围内生成数据。相比于传统的基于扩散的生成模型，它通过KL散度上界进行优化，证明了效果更好。 |
| [^3] | [Sample Complexity for Quadratic Bandits: Hessian Dependent Bounds and Optimal Algorithms.](http://arxiv.org/abs/2306.12383) | 本文针对随机零阶优化的问题，提出了二次型目标函数及其局部几何结构的最优Hessian相关样本复杂度，从信息论角度提供Hessian相关复杂度的下界，并提供了一种Hessian无关的算法可普遍实现所有Hessian实例的渐近最优样本复杂度。 |
| [^4] | [Embedding Inequalities for Barron-type Spaces.](http://arxiv.org/abs/2305.19082) | 本文测量了Barron空间和谱Barron空间之间的关系，并提供了嵌入不等式。 |
| [^5] | [Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective.](http://arxiv.org/abs/2305.15408) | 本文从理论层面探究了带有“思维链”提示的大型语言模型在解决基本数学和决策问题中的能力，发现自回归Transformer大小恒定即可解决任务，揭示了“思维链”提示的背后机制。 |
| [^6] | [Learning Rate Free Bayesian Inference in Constrained Domains.](http://arxiv.org/abs/2305.14943) | 我们的算法是学习率无关的约束域采样算法，并提出了一个统一框架，能够处理多种约束采样问题，实现了与现有算法相竞争的性能。 |
| [^7] | [Implicitly normalized forecaster with clipping for linear and non-linear heavy-tailed multi-armed bandits.](http://arxiv.org/abs/2305.06743) | 本文提出了一种针对奖励分布重尾的MAB问题的隐式规范化预测器，证明该方法在线性和非线性重尾随机MAB问题上是最优的。 |
| [^8] | [Learning with Explanation Constraints.](http://arxiv.org/abs/2303.14496) | 本文研究了解释约束下的学习问题，提出了EPAC模型，探讨了使用这些解释时模型的益处，并提供了一种基于变分近似的算法解决方案。 |
| [^9] | [FuNVol: A Multi-Asset Implied Volatility Market Simulator using Functional Principal Components and Neural SDEs.](http://arxiv.org/abs/2303.00859) | FuNVol是一个多资产隐含波动率市场模拟器，使用函数主成分和神经SDE生成真实历史价格的IV表面序列，并在无静态套利的表面次流形内产生一致的市场情景。同时，使用模拟表面进行对冲可以生成与实现P＆L一致的损益分布。 |
| [^10] | [Quantum Learning Theory Beyond Batch Binary Classification.](http://arxiv.org/abs/2302.07409) | 这篇论文通过研究量子学习理论拓展了批处理多类学习、在线布尔学习和在线多类学习，并提出了第一个具有量子示例的在线学习模型。 |
| [^11] | [Private Statistical Estimation of Many Quantiles.](http://arxiv.org/abs/2302.06943) | 本文主要研究如何在差分隐私条件下估计一个分布的多个分位数。它提出了两种方法：一种是通过私有地估计样本的经验分位数来估计分布的分位数，另一种是使用密度估计技术进行分位数函数估计，并且展示了两种方法之间的权衡。 |
| [^12] | [Scalable PAC-Bayesian Meta-Learning via the PAC-Optimal Hyper-Posterior: From Theory to Practice.](http://arxiv.org/abs/2211.07206) | 本研究使用PAC-Bayesian理论提供了元学习的泛化界限，并推导出了最佳性能保证的闭式优化超后验(PACOH)。通过理论分析和案例研究，我们展示了这些保证在元学习中相对于PAC-Bayesian每个任务学习界限的改进。 |
| [^13] | [Robust Fine-Tuning of Deep Neural Networks with Hessian-based Generalization Guarantees.](http://arxiv.org/abs/2206.02659) | 本文通过Hessian-based分析，提出一种基于距离的泛化度量方法，用于理解深度神经网络微调的泛化特性。通过PAC-Bayesian分析，给出了基于Hessian距离的微调模型泛化界。此外，还对微调面对标签噪声的问题进行了研究，并提出了一种相关算法和泛化误差保证。 |

# 详细

[^1]: 渐进免费素描稀疏岭集合：风险，交叉验证和调整

    Asymptotically free sketched ridge ensembles: Risks, cross-validation, and tuning. (arXiv:2310.04357v1 [math.ST])

    [http://arxiv.org/abs/2310.04357](http://arxiv.org/abs/2310.04357)

    该论文利用随机矩阵理论建立了一致性估计方法，用于估计素描岭回归集合的预测风险，从而实现了正则化和素描参数的高效一致调整。

    

    我们利用随机矩阵理论，建立了推广交叉验证（GCV）用于估计素描岭回归集合的预测风险的一致性，从而实现了正则化和素描参数的高效一致调整。我们的结果适用于一类广泛的渐进免费素描，对数据假设非常温和。对于平方预测风险，我们提供了一个分解成等效非素描隐含岭偏差和基于素描的方差的方法，并证明风险可以通过仅调整无限集合中的素描大小来全局优化。对于一般的亚二次预测风险函数，我们扩展了GCV来构建一致的风险估计，从而在Wasserstein-2度量下获得了GCV修正的预测的分布收敛性。这特别允许在训练数据条件下构建具有渐进正确覆盖率的预测区间。我们还提出了一种“集合技巧”，通过这种技巧可以推断未经过描绘的风险。

    We employ random matrix theory to establish consistency of generalized cross validation (GCV) for estimating prediction risks of sketched ridge regression ensembles, enabling efficient and consistent tuning of regularization and sketching parameters. Our results hold for a broad class of asymptotically free sketches under very mild data assumptions. For squared prediction risk, we provide a decomposition into an unsketched equivalent implicit ridge bias and a sketching-based variance, and prove that the risk can be globally optimized by only tuning sketch size in infinite ensembles. For general subquadratic prediction risk functionals, we extend GCV to construct consistent risk estimators, and thereby obtain distributional convergence of the GCV-corrected predictions in Wasserstein-2 metric. This in particular allows construction of prediction intervals with asymptotically correct coverage conditional on the training data. We also propose an "ensemble trick" whereby the risk for unsket
    
[^2]: Beta Diffusion. (arXiv:2309.07867v1 [cs.LG])

    Beta Diffusion. (arXiv:2309.07867v1 [cs.LG])

    [http://arxiv.org/abs/2309.07867](http://arxiv.org/abs/2309.07867)

    beta扩散是一种新型生成模型方法，通过引入去掩盖和去噪的技术，利用缩放和偏移的beta分布进行乘法转换，实现在有界范围内生成数据。相比于传统的基于扩散的生成模型，它通过KL散度上界进行优化，证明了效果更好。

    

    我们引入了beta扩散，一种将去掩盖和去噪集成到一起的新型生成建模方法，用于在有界范围内生成数据。使用了缩放和偏移的beta分布，beta扩散利用了随时间的乘法转换来创建正向和反向的扩散过程，同时维持着正向边缘分布和反向条件分布，给定任意时间点的数据。与传统的基于扩散的生成模型不同，传统模型依赖于加性高斯噪声和重新加权的证据下界（ELBO），beta扩散是乘法的，并且通过从KL散度的凸性推导出来的KL散度上界（KLUB）进行优化。我们证明了所提出的KLUB相对于负ELBO来说对于优化beta扩散更加有效，负ELBO也可以作为相同KL散度的KLUB，只是其两个参数交换了位置。beta扩散的损失函数以Bregman散度为指标来表示。

    We introduce beta diffusion, a novel generative modeling method that integrates demasking and denoising to generate data within bounded ranges. Using scaled and shifted beta distributions, beta diffusion utilizes multiplicative transitions over time to create both forward and reverse diffusion processes, maintaining beta distributions in both the forward marginals and the reverse conditionals, given the data at any point in time. Unlike traditional diffusion-based generative models relying on additive Gaussian noise and reweighted evidence lower bounds (ELBOs), beta diffusion is multiplicative and optimized with KL-divergence upper bounds (KLUBs) derived from the convexity of the KL divergence. We demonstrate that the proposed KLUBs are more effective for optimizing beta diffusion compared to negative ELBOs, which can also be derived as the KLUBs of the same KL divergence with its two arguments swapped. The loss function of beta diffusion, expressed in terms of Bregman divergence, furt
    
[^3]: 二次型赌臂机的样本复杂度：Hessian相关性界限和最优算法

    Sample Complexity for Quadratic Bandits: Hessian Dependent Bounds and Optimal Algorithms. (arXiv:2306.12383v1 [cs.LG])

    [http://arxiv.org/abs/2306.12383](http://arxiv.org/abs/2306.12383)

    本文针对随机零阶优化的问题，提出了二次型目标函数及其局部几何结构的最优Hessian相关样本复杂度，从信息论角度提供Hessian相关复杂度的下界，并提供了一种Hessian无关的算法可普遍实现所有Hessian实例的渐近最优样本复杂度。

    

    在随机零阶优化中，了解如何充分利用底层目标函数的局部几何结构是一个实际相关的问题。我们考虑一种基本情况，即目标函数是二次型的，并且提供了最优Hessian相关样本复杂度的第一个紧密刻画。我们的贡献具有双重性质。首先，从信息论的角度出发，通过引入一种称为能量分配的概念来捕捉搜索算法和目标函数几何结构之间的交互，证明了Hessian相关复杂度的紧密下界。通过解决最优能量谱，得到了配套的上限。其次，算法方面，我们展示了存在一种Hessian无关的算法，能够普遍实现所有Hessian实例的渐近最优样本复杂度。我们算法能够实现的渐近最优样本复杂度对于重尾噪声分布仍然有效。

    In stochastic zeroth-order optimization, a problem of practical relevance is understanding how to fully exploit the local geometry of the underlying objective function. We consider a fundamental setting in which the objective function is quadratic, and provide the first tight characterization of the optimal Hessian-dependent sample complexity. Our contribution is twofold. First, from an information-theoretic point of view, we prove tight lower bounds on Hessian-dependent complexities by introducing a concept called energy allocation, which captures the interaction between the searching algorithm and the geometry of objective functions. A matching upper bound is obtained by solving the optimal energy spectrum. Then, algorithmically, we show the existence of a Hessian-independent algorithm that universally achieves the asymptotic optimal sample complexities for all Hessian instances. The optimal sample complexities achieved by our algorithm remain valid for heavy-tailed noise distributio
    
[^4]: Barron型空间的嵌入不等式

    Embedding Inequalities for Barron-type Spaces. (arXiv:2305.19082v1 [stat.ML])

    [http://arxiv.org/abs/2305.19082](http://arxiv.org/abs/2305.19082)

    本文测量了Barron空间和谱Barron空间之间的关系，并提供了嵌入不等式。

    

    深度学习理论中的一个基本问题是理解高维条件下两层神经网络的逼近和泛化性质。为了解决这个问题，研究人员引入了Barron空间$\mathcal{B}_s(\Omega)$和谱Barron空间$\mathcal{F}_s(\Omega)$，其中指数$s$表征了这些空间中函数的平滑性，$\Omega\subset\mathbb{R}^d$表示输入域。然而，两种类型的Barron空间之间的关系仍不清楚。本文通过以下不等式建立了这些空间之间的连续嵌入：对于任意$\delta\in(0,1),s\in\mathbb{N}^{+}$和$f:\Omega \mapsto \mathbb{R}$，都有\[ \delta\gamma^{\delta-s}_{\Omega}\|f\|_{\mathcal{F}_{s-\delta}(\Omega)}\lesssim_s \|f\|_{\mathcal{B}_s(\Omega)}\lesssim_s \|f\|_{\mathcal{F}_{s+1}(\Omega)}, \]其中$\gamma_{\Omega}=\sup_{\|v\|_2=1,x\in\Omega}|v^Tx|$，$\lesssim_s$表示仅与平滑参数$s$有关的常数。

    One of the fundamental problems in deep learning theory is understanding the approximation and generalization properties of two-layer neural networks in high dimensions. In order to tackle this issue, researchers have introduced the Barron space $\mathcal{B}_s(\Omega)$ and the spectral Barron space $\mathcal{F}_s(\Omega)$, where the index $s$ characterizes the smoothness of functions within these spaces and $\Omega\subset\mathbb{R}^d$ represents the input domain. However, it is still not clear what is the relationship between the two types of Barron spaces. In this paper, we establish continuous embeddings between these spaces as implied by the following inequality: for any $\delta\in (0,1), s\in \mathbb{N}^{+}$ and $f: \Omega \mapsto\mathbb{R}$, it holds that \[ \delta\gamma^{\delta-s}_{\Omega}\|f\|_{\mathcal{F}_{s-\delta}(\Omega)}\lesssim_s \|f\|_{\mathcal{B}_s(\Omega)}\lesssim_s \|f\|_{\mathcal{F}_{s+1}(\Omega)}, \] where $\gamma_{\Omega}=\sup_{\|v\|_2=1,x\in\Omega}|v^Tx|$ and notab
    
[^5]: 从理论角度揭示“思维链”背后的奥秘

    Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective. (arXiv:2305.15408v1 [cs.LG])

    [http://arxiv.org/abs/2305.15408](http://arxiv.org/abs/2305.15408)

    本文从理论层面探究了带有“思维链”提示的大型语言模型在解决基本数学和决策问题中的能力，发现自回归Transformer大小恒定即可解决任务，揭示了“思维链”提示的背后机制。

    

    最近的研究发现，"思维链"提示能够显著提高大型语言模型（LLMs）的性能，特别是在涉及数学或推理的复杂任务中。尽管获得了巨大的实证成功，但“思维链”背后的机制以及它如何释放LLMs的潜力仍然是神秘的。本文首次从理论上回答了这些问题。具体而言，我们研究了LLMs带有“思维链”在解决基本数学和决策问题中的能力。我们首先给出一个不可能的结果，表明任何有限深度的Transformer都不能直接输出正确的基本算术/方程任务的答案，除非模型大小随着输入长度的增加呈超多项式增长。相反，我们通过构造证明，大小恒定的自回归Transformer足以通过使用常用的数学语言形式生成“思维链”推导来解决这两个任务。

    Recent studies have discovered that Chain-of-Thought prompting (CoT) can dramatically improve the performance of Large Language Models (LLMs), particularly when dealing with complex tasks involving mathematics or reasoning. Despite the enormous empirical success, the underlying mechanisms behind CoT and how it unlocks the potential of LLMs remain elusive. In this paper, we take a first step towards theoretically answering these questions. Specifically, we examine the capacity of LLMs with CoT in solving fundamental mathematical and decision-making problems. We start by giving an impossibility result showing that any bounded-depth Transformer cannot directly output correct answers for basic arithmetic/equation tasks unless the model size grows super-polynomially with respect to the input length. In contrast, we then prove by construction that autoregressive Transformers of a constant size suffice to solve both tasks by generating CoT derivations using a commonly-used math language forma
    
[^6]: 学习率无关的约束域Bayesian推断

    Learning Rate Free Bayesian Inference in Constrained Domains. (arXiv:2305.14943v1 [stat.ML])

    [http://arxiv.org/abs/2305.14943](http://arxiv.org/abs/2305.14943)

    我们的算法是学习率无关的约束域采样算法，并提出了一个统一框架，能够处理多种约束采样问题，实现了与现有算法相竞争的性能。

    

    我们引入了一套新的基于粒子的算法，用于在约束域内进行采样，这是完全与学习率无关的。我们的方法利用了凸优化中的硬币投注思想，以及约束采样作为概率测度空间上镜像优化问题的观点。基于这个观点，我们还提出了一个统一框架，用于几种现有的约束采样算法，包括镜像Langevin动力学和镜像Stein变分梯度下降。我们在一系列的数值实验中展示了算法的性能，包括从单纯形目标进行采样、带公平性约束进行采样以及后选择推断中的约束采样问题。我们的结果表明，我们的算法在不需要调整任何超参数的情况下，实现了与现有约束采样方法相竞争的性能。

    We introduce a suite of new particle-based algorithms for sampling on constrained domains which are entirely learning rate free. Our approach leverages coin betting ideas from convex optimisation, and the viewpoint of constrained sampling as a mirrored optimisation problem on the space of probability measures. Based on this viewpoint, we also introduce a unifying framework for several existing constrained sampling algorithms, including mirrored Langevin dynamics and mirrored Stein variational gradient descent. We demonstrate the performance of our algorithms on a range of numerical examples, including sampling from targets on the simplex, sampling with fairness constraints, and constrained sampling problems in post-selection inference. Our results indicate that our algorithms achieve competitive performance with existing constrained sampling methods, without the need to tune any hyperparameters.
    
[^7]: 针对线性和非线性重尾多臂老虎机的隐式范数预测器的修剪

    Implicitly normalized forecaster with clipping for linear and non-linear heavy-tailed multi-armed bandits. (arXiv:2305.06743v1 [cs.LG])

    [http://arxiv.org/abs/2305.06743](http://arxiv.org/abs/2305.06743)

    本文提出了一种针对奖励分布重尾的MAB问题的隐式规范化预测器，证明该方法在线性和非线性重尾随机MAB问题上是最优的。

    

    已知隐式范数预测器（在线镜像下降，以Tsallis熵作为prox函数）是对抗性多臂老虎机问题（MAB）的最佳算法。但是，大多数复杂性结果都依赖于有界奖励或其他限制性假设。最近有关最佳二者结合算法的研究已经针对对手性和随机重尾MAB设置进行了探讨。这个算法在这两种情况下都是最优的，但不能充分利用数据。在本文中，我们针对奖励分布重尾的MAB问题提出了带剪辑的隐式规范化预测器。我们在奖励分布上提出渐进收敛性结果，并证明所提出的方法对于线性和非线性重尾随机MAB问题是最优的。我们还证明了与最好的二者结合算法相比，该算法通常表现更好。

    Implicitly Normalized Forecaster (online mirror descent with Tsallis entropy as prox-function) is known to be an optimal algorithm for adversarial multi-armed problems (MAB). However, most of the complexity results rely on bounded rewards or other restrictive assumptions. Recently closely related best-of-both-worlds algorithm were proposed for both adversarial and stochastic heavy-tailed MAB settings. This algorithm is known to be optimal in both settings, but fails to exploit data fully. In this paper, we propose Implicitly Normalized Forecaster with clipping for MAB problems with heavy-tailed distribution on rewards. We derive convergence results under mild assumptions on rewards distribution and show that the proposed method is optimal for both linear and non-linear heavy-tailed stochastic MAB problems. Also we show that algorithm usually performs better compared to best-of-two-worlds algorithm.
    
[^8]: 解释约束下的学习

    Learning with Explanation Constraints. (arXiv:2303.14496v1 [cs.LG])

    [http://arxiv.org/abs/2303.14496](http://arxiv.org/abs/2303.14496)

    本文研究了解释约束下的学习问题，提出了EPAC模型，探讨了使用这些解释时模型的益处，并提供了一种基于变分近似的算法解决方案。

    

    尽管监督学习假设存在标注数据，但我们可能有关于模型应如何运行的先验信息。本文将其形式化为从解释约束中学习，并提供了一个学习理论框架，分析了这些解释如何提高模型的学习能力。本文的第一项关键贡献是通过定义我们称之为EPAC模型（在新数据期望中满足这些约束的模型）来回答哪些模型会受益于解释这一问题。我们使用标准的学习理论工具分析了这类模型。第二个关键贡献是对于由线性模型和两层神经网络的梯度信息给出的规范解释的限制（以其Rademacher复杂度为衡量标准）进行了表征。最后，我们通过一种变分近似提供了我们的框架的算法解决方案，它能够实现更好的性能并满足这些约束。

    While supervised learning assumes the presence of labeled data, we may have prior information about how models should behave. In this paper, we formalize this notion as learning from explanation constraints and provide a learning theoretic framework to analyze how such explanations can improve the learning of our models. For what models would explanations be helpful? Our first key contribution addresses this question via the definition of what we call EPAC models (models that satisfy these constraints in expectation over new data), and we analyze this class of models using standard learning theoretic tools. Our second key contribution is to characterize these restrictions (in terms of their Rademacher complexities) for a canonical class of explanations given by gradient information for linear models and two layer neural networks. Finally, we provide an algorithmic solution for our framework, via a variational approximation that achieves better performance and satisfies these constraint
    
[^9]: FuNVol：使用函数主成分和神经SDE的多资产隐含波动率市场模拟器

    FuNVol: A Multi-Asset Implied Volatility Market Simulator using Functional Principal Components and Neural SDEs. (arXiv:2303.00859v2 [q-fin.CP] UPDATED)

    [http://arxiv.org/abs/2303.00859](http://arxiv.org/abs/2303.00859)

    FuNVol是一个多资产隐含波动率市场模拟器，使用函数主成分和神经SDE生成真实历史价格的IV表面序列，并在无静态套利的表面次流形内产生一致的市场情景。同时，使用模拟表面进行对冲可以生成与实现P＆L一致的损益分布。

    

    我们介绍了一种新的方法，使用函数数据分析和神经随机微分方程，结合概率积分变换惩罚来生成多个资产的隐含波动率表面序列，该方法忠实于历史价格。我们证明了学习IV表面和价格的联合动态产生的市场情景与历史特征一致，并且在没有静态套利的表面次流形内。最后，我们证明使用模拟表面进行对冲会生成与实现P＆L一致的损益分布。

    Here, we introduce a new approach for generating sequences of implied volatility (IV) surfaces across multiple assets that is faithful to historical prices. We do so using a combination of functional data analysis and neural stochastic differential equations (SDEs) combined with a probability integral transform penalty to reduce model misspecification. We demonstrate that learning the joint dynamics of IV surfaces and prices produces market scenarios that are consistent with historical features and lie within the sub-manifold of surfaces that are essentially free of static arbitrage. Finally, we demonstrate that delta hedging using the simulated surfaces generates profit and loss (P&L) distributions that are consistent with realised P&Ls.
    
[^10]: 《超越批处理二元分类的量子学习理论》

    Quantum Learning Theory Beyond Batch Binary Classification. (arXiv:2302.07409v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.07409](http://arxiv.org/abs/2302.07409)

    这篇论文通过研究量子学习理论拓展了批处理多类学习、在线布尔学习和在线多类学习，并提出了第一个具有量子示例的在线学习模型。

    

    Arunachalam和de Wolf（2018）证明了在可实现和糊涂设置下，量子批处理学习布尔函数的样本复杂性与相应的经典样本复杂性具有相同的形式和数量级。在本文中，我们将这个明显令人惊讶的结果推广到了批处理多类学习、在线布尔学习和在线多类学习。对于我们的在线学习结果，我们首先考虑了Dawid和Tewari（2022）经典模型的自适应对手变体。然后，我们引入了第一个（据我们所知）具有量子示例的在线学习模型。

    Arunachalam and de Wolf (2018) showed that the sample complexity of quantum batch learning of boolean functions, in the realizable and agnostic settings, has the same form and order as the corresponding classical sample complexities. In this paper, we extend this, ostensibly surprising, message to batch multiclass learning, online boolean learning, and online multiclass learning. For our online learning results, we first consider an adaptive adversary variant of the classical model of Dawid and Tewari (2022). Then, we introduce the first (to the best of our knowledge) model of online learning with quantum examples.
    
[^11]: 多个分位数的私有统计估计

    Private Statistical Estimation of Many Quantiles. (arXiv:2302.06943v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.06943](http://arxiv.org/abs/2302.06943)

    本文主要研究如何在差分隐私条件下估计一个分布的多个分位数。它提出了两种方法：一种是通过私有地估计样本的经验分位数来估计分布的分位数，另一种是使用密度估计技术进行分位数函数估计，并且展示了两种方法之间的权衡。

    

    本文研究在差分隐私条件下估计许多统计分位数的问题。更具体地，给定一个分布并且能够访问来自其独立同分布样本，我们考虑在特定点上估计其累积分布函数的逆函数（分位数函数）。例如，这项任务在私有数据生成中非常重要。我们提出了两种不同的方法。第一种方法是私下估计样本的经验分位数，并将此结果用作分布的分位数估计器。特别地，我们研究了 Kaplan等人最近发表的递归估计分位数的隐私算法的统计性质。第二种方法是使用密度估计技术进行均匀间隔内的分位数函数估计。特别地，我们展示了两种方法之间的权衡。当我们想要估计许多分位数时，最好使用第一种方法单独估计它们。另一方面，当我们想要在大区间上估计分位数函数时，第二种方法更有效。

    This work studies the estimation of many statistical quantiles under differential privacy. More precisely, given a distribution and access to i.i.d. samples from it, we study the estimation of the inverse of its cumulative distribution function (the quantile function) at specific points. For instance, this task is of key importance in private data generation. We present two different approaches. The first one consists in privately estimating the empirical quantiles of the samples and using this result as an estimator of the quantiles of the distribution. In particular, we study the statistical properties of the recently published algorithm introduced by Kaplan et al. 2022 that privately estimates the quantiles recursively. The second approach is to use techniques of density estimation in order to uniformly estimate the quantile function on an interval. In particular, we show that there is a tradeoff between the two methods. When we want to estimate many quantiles, it is better to estim
    
[^12]: 可扩展的PAC-Bayesian元学习：从理论到实践的PAC-Optimal超后验

    Scalable PAC-Bayesian Meta-Learning via the PAC-Optimal Hyper-Posterior: From Theory to Practice. (arXiv:2211.07206v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.07206](http://arxiv.org/abs/2211.07206)

    本研究使用PAC-Bayesian理论提供了元学习的泛化界限，并推导出了最佳性能保证的闭式优化超后验(PACOH)。通过理论分析和案例研究，我们展示了这些保证在元学习中相对于PAC-Bayesian每个任务学习界限的改进。

    

    元学习旨在通过从相关学习任务的数据集中获取有用的归纳偏好，加速对新任务的学习过程。然而，在实践中，可用的相关任务数量通常很小，大多数现有方法假设任务数量丰富，使它们不切实际且容易过拟合。元学习文献中的一个核心问题是如何进行正则化以确保对未见任务的泛化。在这项工作中，我们使用PAC-Bayesian理论提供了一种理论分析，并提出了元学习的泛化界限，这是由Rothfuss等人（2021）首次推导出来的。关键是，该界限使我们能够得到最佳性能保证的闭式优化超后验，称为PACOH。我们提供了一种理论分析和实证案例研究，在哪些条件下以及在多大程度上这些元学习的保证改进了PAC-Bayesian每个任务学习界限。

    Meta-Learning aims to speed up the learning process on new tasks by acquiring useful inductive biases from datasets of related learning tasks. While, in practice, the number of related tasks available is often small, most of the existing approaches assume an abundance of tasks; making them unrealistic and prone to overfitting. A central question in the meta-learning literature is how to regularize to ensure generalization to unseen tasks. In this work, we provide a theoretical analysis using the PAC-Bayesian theory and present a generalization bound for meta-learning, which was first derived by Rothfuss et al. (2021). Crucially, the bound allows us to derive the closed form of the optimal hyper-posterior, referred to as PACOH, which leads to the best performance guarantees. We provide a theoretical analysis and empirical case study under which conditions and to what extent these guarantees for meta-learning improve upon PAC-Bayesian per-task learning bounds. The closed-form PACOH inspi
    
[^13]: 具有基于Hessian的泛化保证的深度神经网络鲁棒微调

    Robust Fine-Tuning of Deep Neural Networks with Hessian-based Generalization Guarantees. (arXiv:2206.02659v5 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.02659](http://arxiv.org/abs/2206.02659)

    本文通过Hessian-based分析，提出一种基于距离的泛化度量方法，用于理解深度神经网络微调的泛化特性。通过PAC-Bayesian分析，给出了基于Hessian距离的微调模型泛化界。此外，还对微调面对标签噪声的问题进行了研究，并提出了一种相关算法和泛化误差保证。

    

    本文考虑在目标任务上对预训练的深度神经网络进行微调。我们研究微调的泛化特性，以理解过拟合问题，这在目标数据集较小或训练标签噪声时经常观察到。现有的深度网络泛化度量依赖于与微调模型的初始化（即预训练网络）距离和深度网络的噪声稳定性等概念。本文通过PAC-Bayesian分析确定了一种基于Hessian的距离度量，它与微调模型的观察到的泛化差距相关性很强。从理论上我们证明了基于Hessian距离的微调模型的泛化界。我们还对微调对抗标签噪声进行了扩展研究，过拟合是一个关键问题；我们提出了一种算法，并在类条件独立假设下给出了该算法的泛化误差保证。

    We consider fine-tuning a pretrained deep neural network on a target task. We study the generalization properties of fine-tuning to understand the problem of overfitting, which has often been observed (e.g., when the target dataset is small or when the training labels are noisy). Existing generalization measures for deep networks depend on notions such as distance from the initialization (i.e., the pretrained network) of the fine-tuned model and noise stability properties of deep networks. This paper identifies a Hessian-based distance measure through PAC-Bayesian analysis, which is shown to correlate well with observed generalization gaps of fine-tuned models. Theoretically, we prove Hessian distance-based generalization bounds for fine-tuned models. We also describe an extended study of fine-tuning against label noise, where overfitting is against a critical problem; We present an algorithm and a generalization error guarantee for this algorithm under a class conditional independent 
    

