# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Lipschitz constant of random neural networks.](http://arxiv.org/abs/2311.01356) | 本文研究了随机ReLU神经网络的Lipschitz常数，对于浅层神经网络，我们得到了Lipschitz常数的精确刻画，对于足够宽度的深层神经网络，我们给出了上下界，并匹配一个依赖于深度的对数因子。 |
| [^2] | [Contextualized Policy Recovery: Modeling and Interpreting Medical Decisions with Adaptive Imitation Learning.](http://arxiv.org/abs/2310.07918) | 本论文提出了一种上下文化政策恢复方法用于建模复杂的医疗决策过程，以解决现有模型在准确性和可解释性之间的权衡问题。该方法将决策策略拆分为上下文特定策略，通过多任务学习来实现建模，并提供复杂行为的简洁描述。 |
| [^3] | [Coarse-Graining Hamiltonian Systems Using WSINDy.](http://arxiv.org/abs/2310.05879) | 本论文研究了使用WSINDy进行粗粒化哈密顿系统的问题，扩展了WSINDy在相互作用粒子系统中的粗粒化能力。通过识别近似对称性和处理外部扰动，WSINDy成功地识别出降维的哈密顿系统，从而有效地捕捉了相关自由度的动力学。 |
| [^4] | [Beta Diffusion.](http://arxiv.org/abs/2309.07867) | beta扩散是一种新型生成模型方法，通过引入去掩盖和去噪的技术，利用缩放和偏移的beta分布进行乘法转换，实现在有界范围内生成数据。相比于传统的基于扩散的生成模型，它通过KL散度上界进行优化，证明了效果更好。 |
| [^5] | [QuantEase: Optimization-based Quantization for Language Models -- An Efficient and Intuitive Algorithm.](http://arxiv.org/abs/2309.01885) | QuantEase是一种基于优化的语言模型量化算法，通过逐层量化和基于坐标下降的算法，高质量地解决了复杂的非凸量化问题，并引入了对异常值敏感的变种方法。 |
| [^6] | [GeoPhy: Differentiable Phylogenetic Inference via Geometric Gradients of Tree Topologies.](http://arxiv.org/abs/2307.03675) | GeoPhy是一种创新的、完全可微的系统发育推断方法，通过在连续几何空间中表示拓扑分布，实现了可扩展的变分推断，克服了不限制拓扑结构的挑战。 |
| [^7] | [Learning Causally Disentangled Representations via the Principle of Independent Causal Mechanisms.](http://arxiv.org/abs/2306.01213) | 本文通过定义独立因果机制，提出了ICM-VAE框架，使得学习因果解缠绕表示更准确 |
| [^8] | [A Policy Gradient Method for Confounded POMDPs.](http://arxiv.org/abs/2305.17083) | 本文提出了一种针对混淆部分可观测马尔可夫决策过程的新型策略梯度方法，该方法在离线设置下可同时处理连续状态和观察空间，具有高效性和准确性。 |
| [^9] | [On the Identifiability of Markov Switching Models.](http://arxiv.org/abs/2305.15925) | 本文研究了马尔科夫转换模型的可辨识性，通过非线性高斯参数化迁移分布实现第一阶段马尔科夫依赖结构中的可辨识性条件。该方法适用于依赖于政权的因果发现和高维时间序列分割。 |
| [^10] | [On the Trade-off of Intra-/Inter-class Diversity for Supervised Pre-training.](http://arxiv.org/abs/2305.12224) | 本文研究了监督预训练数据集中类内多样性和类间多样性之间的权衡对下游任务表现的影响，并理论上证明了下游性能单调地取决于这两种多样性。最佳的类别样本比（#类别 / #每类样本数）与预训练数据集大小无关，可以应用于预测最佳的预训练类别数。 |
| [^11] | [RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment.](http://arxiv.org/abs/2304.06767) | RAFT框架引入了奖励排名微调方法，用于对齐生成型基础模型，以解决强化学习带来的低效和不稳定性问题。 |
| [^12] | [Bandwidth Selection for Gaussian Kernel Ridge Regression via Jacobian Control.](http://arxiv.org/abs/2205.11956) | 本文提出了一种基于雅可比控制的带宽选择启发式方法，该方法具有闭式、计算非常轻的特点，并且在关注带宽的同时可以获得更好的模型泛化性能。 |

# 详细

[^1]: 关于随机神经网络的Lipschitz常数

    On the Lipschitz constant of random neural networks. (arXiv:2311.01356v1 [stat.ML])

    [http://arxiv.org/abs/2311.01356](http://arxiv.org/abs/2311.01356)

    本文研究了随机ReLU神经网络的Lipschitz常数，对于浅层神经网络，我们得到了Lipschitz常数的精确刻画，对于足够宽度的深层神经网络，我们给出了上下界，并匹配一个依赖于深度的对数因子。

    

    实证研究广泛证明神经网络对输入的微小对抗性扰动非常敏感。这些所谓的对抗性示例的最坏情况鲁棒性可以通过神经网络的Lipschitz常数来量化。然而，关于这个量的理论结果在文献中仅有少数。在本文中，我们开始研究随机ReLU神经网络的Lipschitz常数，即选择随机权重并采用ReLU激活函数的神经网络。对于浅层神经网络，我们将Lipschitz常数刻画到一个绝对数值常数。此外，我们将我们的分析扩展到足够宽度的深层神经网络，我们证明了Lipschitz常数的上下界。这些界匹配到一个依赖于深度的对数因子上。

    Empirical studies have widely demonstrated that neural networks are highly sensitive to small, adversarial perturbations of the input. The worst-case robustness against these so-called adversarial examples can be quantified by the Lipschitz constant of the neural network. However, only few theoretical results regarding this quantity exist in the literature. In this paper, we initiate the study of the Lipschitz constant of random ReLU neural networks, i.e., neural networks whose weights are chosen at random and which employ the ReLU activation function. For shallow neural networks, we characterize the Lipschitz constant up to an absolute numerical constant. Moreover, we extend our analysis to deep neural networks of sufficiently large width where we prove upper and lower bounds for the Lipschitz constant. These bounds match up to a logarithmic factor that depends on the depth.
    
[^2]: 上下文化政策恢复：通过自适应模仿学习对医疗决策进行建模和解释

    Contextualized Policy Recovery: Modeling and Interpreting Medical Decisions with Adaptive Imitation Learning. (arXiv:2310.07918v1 [cs.LG])

    [http://arxiv.org/abs/2310.07918](http://arxiv.org/abs/2310.07918)

    本论文提出了一种上下文化政策恢复方法用于建模复杂的医疗决策过程，以解决现有模型在准确性和可解释性之间的权衡问题。该方法将决策策略拆分为上下文特定策略，通过多任务学习来实现建模，并提供复杂行为的简洁描述。

    

    可解释的策略学习旨在从观察到的行为中估计可理解的决策策略；然而，现有模型在准确性和可解释性之间存在权衡。这种权衡限制了基于数据驱动的对人类决策过程的解释，例如，审计医疗决策的偏见和次优实践，我们需要决策过程的模型，能够提供复杂行为的简洁描述。现有方法基本上由于将潜在决策过程表示为通用策略而负担了这种权衡，而实际上人类决策是动态的，可以随上下文信息而大幅改变。因此，我们提出了上下文化政策恢复（CPR），将建模复杂决策过程的问题重新定义为多任务学习问题，其中复杂决策策略由特定上下文的策略组成。CPR将每个上下文特定策略建模为线性的观察-动作映射

    Interpretable policy learning seeks to estimate intelligible decision policies from observed actions; however, existing models fall short by forcing a tradeoff between accuracy and interpretability. This tradeoff limits data-driven interpretations of human decision-making process. e.g. to audit medical decisions for biases and suboptimal practices, we require models of decision processes which provide concise descriptions of complex behaviors. Fundamentally, existing approaches are burdened by this tradeoff because they represent the underlying decision process as a universal policy, when in fact human decisions are dynamic and can change drastically with contextual information. Thus, we propose Contextualized Policy Recovery (CPR), which re-frames the problem of modeling complex decision processes as a multi-task learning problem in which complex decision policies are comprised of context-specific policies. CPR models each context-specific policy as a linear observation-to-action mapp
    
[^3]: 使用WSINDy进行粗粒化哈密顿系统

    Coarse-Graining Hamiltonian Systems Using WSINDy. (arXiv:2310.05879v1 [physics.comp-ph])

    [http://arxiv.org/abs/2310.05879](http://arxiv.org/abs/2310.05879)

    本论文研究了使用WSINDy进行粗粒化哈密顿系统的问题，扩展了WSINDy在相互作用粒子系统中的粗粒化能力。通过识别近似对称性和处理外部扰动，WSINDy成功地识别出降维的哈密顿系统，从而有效地捕捉了相关自由度的动力学。

    

    在相互作用粒子系统的背景下，已经证明了弱形态稀疏识别非线性动力学算法(WSINDy)具有粗粒化能力。在本工作中，我们将这种能力扩展到具有近似对称性的哈密顿动力学的粗粒化问题上。这种近似对称性通常导致存在一个降维的哈密顿系统，可以有效地捕捉相关自由度的动力学。导出这样的降维系统，或者通过数值方法对其进行近似，是一个持续的挑战。我们证明了WSINDy可以成功地在对称不精确性和外部噪声的影响下识别出这个降维的哈密顿系统。这在一部分是因为这样的系统如何被解析地导出是非平凡的。WSINDy自然地保留了哈密顿结构。

    The Weak-form Sparse Identification of Nonlinear Dynamics algorithm (WSINDy) has been demonstrated to offer coarse-graining capabilities in the context of interacting particle systems ( https://doi.org/10.1016/j.physd.2022.133406 ). In this work we extend this capability to the problem of coarse-graining Hamiltonian dynamics which possess approximate symmetries. Such approximate symmetries often lead to the existence of a Hamiltonian system of reduced dimension that may be used to efficiently capture the dynamics of the relevant degrees of freedom. Deriving such reduced systems, or approximating them numerically, is an ongoing challenge. We demonstrate that WSINDy can successfully identify this reduced Hamiltonian system in the presence of large perturbations imparted from both the inexact nature of the symmetry and extrinsic noise. This is significant in part due to the nontrivial means by which such systems are derived analytically. WSINDy naturally preserves the Hamiltonian structur
    
[^4]: Beta Diffusion. (arXiv:2309.07867v1 [cs.LG])

    Beta Diffusion. (arXiv:2309.07867v1 [cs.LG])

    [http://arxiv.org/abs/2309.07867](http://arxiv.org/abs/2309.07867)

    beta扩散是一种新型生成模型方法，通过引入去掩盖和去噪的技术，利用缩放和偏移的beta分布进行乘法转换，实现在有界范围内生成数据。相比于传统的基于扩散的生成模型，它通过KL散度上界进行优化，证明了效果更好。

    

    我们引入了beta扩散，一种将去掩盖和去噪集成到一起的新型生成建模方法，用于在有界范围内生成数据。使用了缩放和偏移的beta分布，beta扩散利用了随时间的乘法转换来创建正向和反向的扩散过程，同时维持着正向边缘分布和反向条件分布，给定任意时间点的数据。与传统的基于扩散的生成模型不同，传统模型依赖于加性高斯噪声和重新加权的证据下界（ELBO），beta扩散是乘法的，并且通过从KL散度的凸性推导出来的KL散度上界（KLUB）进行优化。我们证明了所提出的KLUB相对于负ELBO来说对于优化beta扩散更加有效，负ELBO也可以作为相同KL散度的KLUB，只是其两个参数交换了位置。beta扩散的损失函数以Bregman散度为指标来表示。

    We introduce beta diffusion, a novel generative modeling method that integrates demasking and denoising to generate data within bounded ranges. Using scaled and shifted beta distributions, beta diffusion utilizes multiplicative transitions over time to create both forward and reverse diffusion processes, maintaining beta distributions in both the forward marginals and the reverse conditionals, given the data at any point in time. Unlike traditional diffusion-based generative models relying on additive Gaussian noise and reweighted evidence lower bounds (ELBOs), beta diffusion is multiplicative and optimized with KL-divergence upper bounds (KLUBs) derived from the convexity of the KL divergence. We demonstrate that the proposed KLUBs are more effective for optimizing beta diffusion compared to negative ELBOs, which can also be derived as the KLUBs of the same KL divergence with its two arguments swapped. The loss function of beta diffusion, expressed in terms of Bregman divergence, furt
    
[^5]: QuantEase: 基于优化的语言模型量化--一种高效而直观的算法

    QuantEase: Optimization-based Quantization for Language Models -- An Efficient and Intuitive Algorithm. (arXiv:2309.01885v1 [stat.ML])

    [http://arxiv.org/abs/2309.01885](http://arxiv.org/abs/2309.01885)

    QuantEase是一种基于优化的语言模型量化算法，通过逐层量化和基于坐标下降的算法，高质量地解决了复杂的非凸量化问题，并引入了对异常值敏感的变种方法。

    

    随着大型语言模型（LLM）的普及，对于能够实现其高效部署的压缩技术的兴趣日益增加。本研究侧重于LLM的后训练量化（PTQ）。借鉴最近的进展，我们的工作引入了QuantEase，一个逐层量化框架，其中各个层面经过单独的量化。该问题被视为离散结构化的非凸优化问题，促使我们开发了基于坐标下降（CD）技术的算法。这些基于CD的方法为复杂的非凸逐层量化问题提供了高质量的解决方案。值得注意的是，我们的CD方法具有简单的更新步骤，仅依赖于矩阵和向量运算，避免了矩阵求逆或分解的需要。我们还探索了一种对异常值敏感的变种方法，允许保留具有完全精度的重要权重（异常值）。我们的提议达到了最先进的状态。

    With the rising popularity of Large Language Models (LLMs), there has been an increasing interest in compression techniques that enable their efficient deployment. This study focuses on the Post-Training Quantization (PTQ) of LLMs. Drawing from recent advances, our work introduces QuantEase, a layer-wise quantization framework where individual layers undergo separate quantization. The problem is framed as a discrete-structured non-convex optimization, prompting the development of algorithms rooted in Coordinate Descent (CD) techniques. These CD-based methods provide high-quality solutions to the complex non-convex layer-wise quantization problems. Notably, our CD-based approach features straightforward updates, relying solely on matrix and vector operations, circumventing the need for matrix inversion or decomposition. We also explore an outlier-aware variant of our approach, allowing for retaining significant weights (outliers) with complete precision. Our proposal attains state-of-th
    
[^6]: GeoPhy: 利用几何梯度实现可微分的系统发育推断

    GeoPhy: Differentiable Phylogenetic Inference via Geometric Gradients of Tree Topologies. (arXiv:2307.03675v1 [cs.LG])

    [http://arxiv.org/abs/2307.03675](http://arxiv.org/abs/2307.03675)

    GeoPhy是一种创新的、完全可微的系统发育推断方法，通过在连续几何空间中表示拓扑分布，实现了可扩展的变分推断，克服了不限制拓扑结构的挑战。

    

    系统发育推断是在分子进化模型基础上进行的，它对于理解生物数据中的进化关系至关重要。考虑到进化树变量的不确定性，包括树拓扑结构和分支上的进化距离，对于准确地从分子数据中推断物种关系以及需要进行变量边缘化的任务来说至关重要。变分贝叶斯方法是开发可扩展、实用模型的关键，然而，在不限制可能的树拓扑结构的组合数的情况下进行系统发育推断仍然具有挑战性。在本研究中，我们引入了一种新颖的、完全可微的系统发育推断公式，利用连续几何空间中的拓扑分布来表示。通过对设计空间和渐近矩的实际考虑，我们的方法GeoPhy可以实现变分推断而不限制拓扑结构的多样性。

    Phylogenetic inference, grounded in molecular evolution models, is essential for understanding the evolutionary relationships in biological data. Accounting for the uncertainty of phylogenetic tree variables, which include tree topologies and evolutionary distances on branches, is crucial for accurately inferring species relationships from molecular data and tasks requiring variable marginalization. Variational Bayesian methods are key to developing scalable, practical models; however, it remains challenging to conduct phylogenetic inference without restricting the combinatorially vast number of possible tree topologies. In this work, we introduce a novel, fully differentiable formulation of phylogenetic inference that leverages a unique representation of topological distributions in continuous geometric spaces. Through practical considerations on design spaces and control variates for gradient estimations, our approach, GeoPhy, enables variational inference without limiting the topolo
    
[^7]: 基于独立因果机制原则学习因果解缠绕表示

    Learning Causally Disentangled Representations via the Principle of Independent Causal Mechanisms. (arXiv:2306.01213v1 [cs.LG])

    [http://arxiv.org/abs/2306.01213](http://arxiv.org/abs/2306.01213)

    本文通过定义独立因果机制，提出了ICM-VAE框架，使得学习因果解缠绕表示更准确

    

    学习解缠绕的因果表示是一个具有挑战性的问题，近年来因其对提取下游任务的有意义信息而引起了广泛关注。本文从独立因果机制的角度定义了一种新的因果解缠绕概念。我们提出了ICM-VAE框架，通过因因果关系观察标签来监督学习因果解缠绕表示。我们使用可学习的基于流的微分同胚函数将噪声变量映射到潜在因果变量中来建模因果机制。此外，为了促进因果要素的解缠绕，我们提出了一种因果解缠绕先验，利用已知的因果结构来鼓励在潜在空间中学习因果分解分布。在相对温和的条件下，我们提供了理论结果，显示了因果要素和机制的可识别性，直到排列和逐元重参数化的限度。我们进行了实证研究...

    Learning disentangled causal representations is a challenging problem that has gained significant attention recently due to its implications for extracting meaningful information for downstream tasks. In this work, we define a new notion of causal disentanglement from the perspective of independent causal mechanisms. We propose ICM-VAE, a framework for learning causally disentangled representations supervised by causally related observed labels. We model causal mechanisms using learnable flow-based diffeomorphic functions to map noise variables to latent causal variables. Further, to promote the disentanglement of causal factors, we propose a causal disentanglement prior that utilizes the known causal structure to encourage learning a causally factorized distribution in the latent space. Under relatively mild conditions, we provide theoretical results showing the identifiability of causal factors and mechanisms up to permutation and elementwise reparameterization. We empirically demons
    
[^8]: 一种针对混淆部分可观测马尔可夫决策过程的策略梯度方法

    A Policy Gradient Method for Confounded POMDPs. (arXiv:2305.17083v1 [stat.ML])

    [http://arxiv.org/abs/2305.17083](http://arxiv.org/abs/2305.17083)

    本文提出了一种针对混淆部分可观测马尔可夫决策过程的新型策略梯度方法，该方法在离线设置下可同时处理连续状态和观察空间，具有高效性和准确性。

    

    本文提出了一种针对具有连续状态和观察空间的混淆部分可观测马尔可夫决策过程（POMDP）的策略梯度方法，在离线设置下使用。我们首先建立了一个新颖的识别结果，以在离线数据下非参数地估计POMDP中的任何历史依赖策略梯度。识别结果使我们能够解决一系列条件矩限制，并采用具有一般函数逼近的最小最大学习过程来估计策略梯度。然后，我们针对预先指定的策略类提供了一个有限样本的非渐近估计界限，以了解样本大小、时间长度、集中度系数和求解条件矩限制的伪正则度量对于均匀估计梯度的影响。最后，通过在梯度上升算法中使用所提出的梯度估计，我们展示了所提出的算法在找到历史依赖性策略梯度方面的全局收敛性。

    In this paper, we propose a policy gradient method for confounded partially observable Markov decision processes (POMDPs) with continuous state and observation spaces in the offline setting. We first establish a novel identification result to non-parametrically estimate any history-dependent policy gradient under POMDPs using the offline data. The identification enables us to solve a sequence of conditional moment restrictions and adopt the min-max learning procedure with general function approximation for estimating the policy gradient. We then provide a finite-sample non-asymptotic bound for estimating the gradient uniformly over a pre-specified policy class in terms of the sample size, length of horizon, concentratability coefficient and the measure of ill-posedness in solving the conditional moment restrictions. Lastly, by deploying the proposed gradient estimation in the gradient ascent algorithm, we show the global convergence of the proposed algorithm in finding the history-depe
    
[^9]: 关于马尔科夫转换模型的可辨识性研究

    On the Identifiability of Markov Switching Models. (arXiv:2305.15925v1 [stat.ML])

    [http://arxiv.org/abs/2305.15925](http://arxiv.org/abs/2305.15925)

    本文研究了马尔科夫转换模型的可辨识性，通过非线性高斯参数化迁移分布实现第一阶段马尔科夫依赖结构中的可辨识性条件。该方法适用于依赖于政权的因果发现和高维时间序列分割。

    

    最近，潜变量模型的可辨识性因其在可解释性或分布泛化方面的应用而备受关注。本文探讨了作为将最近的结果扩展到序列潜变量模型的第一步的马尔科夫转换模型的可辨识性。我们在第一阶段马尔科夫依赖结构中提出了可辨识性条件，并通过非线性高斯参数化迁移分布。我们的实验展示了我们方法在依赖于政权的因果发现和高维时间序列分割方面的适用性。

    Identifiability of latent variable models has recently gained interest in terms of its applications to interpretability or out of distribution generalisation. In this work, we study identifiability of Markov Switching Models as a first step towards extending recent results to sequential latent variable models. We present identifiability conditions within first-order Markov dependency structures, and parametrise the transition distribution via non-linear Gaussians. Our experiments showcase the applicability of our approach for regime-dependent causal discovery and high-dimensional time series segmentation.
    
[^10]: 监督预训练中类内/类间多样性的权衡

    On the Trade-off of Intra-/Inter-class Diversity for Supervised Pre-training. (arXiv:2305.12224v1 [cs.LG])

    [http://arxiv.org/abs/2305.12224](http://arxiv.org/abs/2305.12224)

    本文研究了监督预训练数据集中类内多样性和类间多样性之间的权衡对下游任务表现的影响，并理论上证明了下游性能单调地取决于这两种多样性。最佳的类别样本比（#类别 / #每类样本数）与预训练数据集大小无关，可以应用于预测最佳的预训练类别数。

    

    预训练数据集对于构建最先进的机器学习模型至关重要，因此需要对它们对下游任务的影响进行严格研究。在本文中，我们研究了监督预训练数据集中类内多样性（每个类别的样本数）和类间多样性（类别数）之间的权衡对下游表现的影响。实证表明，当预训练数据集大小固定时，最佳的下游表现取决于类内/类间多样性的平衡。为了了解其基本机制，我们理论上证明了下游表现单调地取决于两种多样性。值得注意的是，我们的理论揭示了最佳的类别样本比（#类别 / #每类样本数）不受预训练数据集大小的影响，这启发我们应用预测最佳的预训练类别数。我们通过实验证明了这种应用的有效性，性能提升约为2个百分点。

    Pre-training datasets are critical for building state-of-the-art machine learning models, motivating rigorous study on their impact on downstream tasks. In this work, we study the impact of the trade-off between the intra-class diversity (the number of samples per class) and the inter-class diversity (the number of classes) of a supervised pre-training dataset. Empirically, we found that with the size of the pre-training dataset fixed, the best downstream performance comes with a balance on the intra-/inter-class diversity. To understand the underlying mechanism, we show theoretically that the downstream performance depends monotonically on both types of diversity. Notably, our theory reveals that the optimal class-to-sample ratio (#classes / #samples per class) is invariant to the size of the pre-training dataset, which motivates an application of predicting the optimal number of pre-training classes. We demonstrate the effectiveness of this application by an improvement of around 2 p
    
[^11]: RAFT: 奖励排名微调用于生成型基础模型对齐

    RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment. (arXiv:2304.06767v1 [cs.LG])

    [http://arxiv.org/abs/2304.06767](http://arxiv.org/abs/2304.06767)

    RAFT框架引入了奖励排名微调方法，用于对齐生成型基础模型，以解决强化学习带来的低效和不稳定性问题。

    

    生成型基础模型容易受到广泛的无监督训练数据带来的隐式偏见的影响。这些偏见可能导致子优样本、扭曲的结果和不公平，可能产生重大影响。因此，将这些模型与人的伦理和偏好对齐是确保它们在真实应用中负责任和有效的部署的关键步骤。以往的研究主要采用人类反馈的强化学习（ RLHF）作为解决这个问题的手段。在 RL 算法的指导下，用人类反馈指导的奖励模型对生成模型进行微调。然而， RL 算法的低效性和不稳定性常常会对生成模型的成功对齐产生重大障碍，因此需要开发一种更为强大和简化的方法。为此，我们引入了一个新的框架，即奖励排名微调（ RAFT ），旨在对齐生成基础模型。

    Generative foundation models are susceptible to implicit biases that can arise from extensive unsupervised training data. Such biases can produce suboptimal samples, skewed outcomes, and unfairness, with potentially significant repercussions. Consequently, aligning these models with human ethics and preferences is an essential step toward ensuring their responsible and effective deployment in real-world applications. Prior research has primarily employed Reinforcement Learning from Human Feedback (RLHF) as a means of addressing this problem, wherein generative models are fine-tuned using RL algorithms guided by a human-feedback-informed reward model. However, the inefficiencies and instabilities associated with RL algorithms frequently present substantial obstacles to the successful alignment of generative models, necessitating the development of a more robust and streamlined approach. To this end, we introduce a new framework, Reward rAnked FineTuning (RAFT), designed to align generat
    
[^12]: 通过雅可比控制选择高斯核岭回归带宽

    Bandwidth Selection for Gaussian Kernel Ridge Regression via Jacobian Control. (arXiv:2205.11956v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2205.11956](http://arxiv.org/abs/2205.11956)

    本文提出了一种基于雅可比控制的带宽选择启发式方法，该方法具有闭式、计算非常轻的特点，并且在关注带宽的同时可以获得更好的模型泛化性能。

    

    大多数机器学习方法需要调整超参数。对于高斯核岭回归，超参数是带宽。带宽指定核函数的长度尺度，必须小心选择才能获得具有良好泛化性能模型。带宽选择的默认方法是交叉验证和边缘似然最大化，这通常会产生良好的结果，尽管计算成本高。此外，这些方法提供的估计往往具有非常高的方差，特别是在训练数据不足时。受雅可比正则化的启发，我们制定了一个近似表达式，用于描述高斯核岭回归推断函数的导数如何取决于核带宽。然后，我们使用这个表达式来提出一种基于雅可比控制的闭式、计算非常轻的带宽选择启发式方法。此外，这个雅可比表达式表明了在检查带宽选择的质量时应关注什么。

    Most machine learning methods require tuning of hyper-parameters. For kernel ridge regression with the Gaussian kernel, the hyper-parameter is the bandwidth. The bandwidth specifies the length-scale of the kernel and has to be carefully selected in order to obtain a model with good generalization. The default methods for bandwidth selection is cross-validation and marginal likelihood maximization, which often yields good results, albeit at high computational costs. Furthermore, the estimates provided by these methods tend to have very high variance, especially when training data are scarce. Inspired by Jacobian regularization, we formulate an approximate expression for how the derivatives of the functions inferred by kernel ridge regression with the Gaussian kernel depend on the kernel bandwidth. We then use this expression to propose a closed-form, computationally feather-light, bandwidth selection heuristic based on controlling the Jacobian. In addition, the Jacobian expression illum
    

