# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Conditional Generative Learning: Model and Error Analysis](https://rss.arxiv.org/abs/2402.01460) | 提出了一种基于ODE的深度生成方法，通过条件Follmer流来学习条件分布，通过离散化和深度神经网络实现高效转化。同时，通过Wasserstein距离的非渐近收敛速率，提供了第一个端到端误差分析，数值实验证明其在不同场景下的优越性。 |
| [^2] | [Functional Bilevel Optimization for Machine Learning](https://arxiv.org/abs/2403.20233) | 介绍了机器学习中的函数双层优化问题，提出了不依赖于强凸假设的方法，并展示了在仪表回归和强化学习任务中使用神经网络的优势。 |
| [^3] | [Towards Model-Agnostic Posterior Approximation for Fast and Accurate Variational Autoencoders](https://arxiv.org/abs/2403.08941) | 最近的研究表明，为了确保高质量的推断模型，可以通过迭代训练最大化与推断模型相关的目标函数，以解决变分自编码器中推断模型近似不准确导致的局部最优解问题。 |
| [^4] | [On a Neural Implementation of Brenier's Polar Factorization](https://arxiv.org/abs/2403.03071) | 提出了Brenier的极分解定理的神经实现，探讨了在机器学习中的应用，并通过神经网络参数化潜在函数$u$，从最新神经最优输运领域的进展中汲取灵感。 |
| [^5] | [Data-Efficient Operator Learning via Unsupervised Pretraining and In-Context Learning](https://arxiv.org/abs/2402.15734) | 该论文提出了一种通过无监督预训练和上下文学习方法实现PDE运算符学习的高效方式，以提高数据效率并改善模型的外域性能。 |
| [^6] | [Improved Evidential Deep Learning via a Mixture of Dirichlet Distributions](https://arxiv.org/abs/2402.06160) | 本文通过混合狄利克雷分布来改进证据深度学习（EDL）方法，解决了现有方法中认知不确定性在无限样本限制下可能不会消失的问题。 |
| [^7] | [Sharp Rates in Dependent Learning Theory: Avoiding Sample Size Deflation for the Square Loss](https://arxiv.org/abs/2402.05928) | 本文研究了依赖学习理论中的尖锐率，主要是为了避免样本大小缩减对方差产生影响。当假设类别的拓扑结构符合某些条件时，经验风险最小化者的性能与类别的复杂性和二阶统计量有关。 |
| [^8] | [Learning Metrics that Maximise Power for Accelerated A/B-Tests](https://arxiv.org/abs/2402.03915) | 本论文提出了一种新方法，通过从短期信号中学习指标，直接最大化指标与北极度量标准之间的统计能力，从而减少在线控制实验的成本。 |
| [^9] | [Flora: Low-Rank Adapters Are Secretly Gradient Compressors](https://arxiv.org/abs/2402.03293) | 本文研究了低秩适配器的动力学，并提出了一种基于随机投影的方法Flora，通过重新采样投影矩阵实现高秩更新，同时减少优化状态的空间复杂度。 |
| [^10] | [Graph topological property recovery with heat and wave dynamics-based features on graphsD.](http://arxiv.org/abs/2309.09924) | 本文提出了一种名为图微分方程网络（GDeNet）的方法，利用热和波动方程动力学特征来恢复图的拓扑属性，能够在各种下游任务中获得优秀的表现，同时在实际应用中也展现了较好的性能。 |
| [^11] | [Multimodal Learning Without Labeled Multimodal Data: Guarantees and Applications.](http://arxiv.org/abs/2306.04539) | 本文研究在只有带标签的单模态数据和自然出现的多模态数据的情况下，如何量化多模态交互的挑战，并提出了两个下界和一个上界来量化多模态交互量。 |
| [^12] | [Improved Stability and Generalization Analysis of the Decentralized SGD Algorithm.](http://arxiv.org/abs/2306.02939) | 本文提出了新的算法稳定性理论来改进分布式SGD算法的泛化性能分析，推翻了现有技术对通信图负面影响的观点，并展示了D-SGD在凸设置中与经典SGD算法泛化界相同。 |
| [^13] | [The Mean Squared Error of the Ridgeless Least Squares Estimator under General Assumptions on Regression Errors.](http://arxiv.org/abs/2305.12883) | 该论文研究了基于一般回归误差假设的无噪声回归最小二乘估计值的均方误差，并发现包含大量不重要的参数可以有效地降低估计器的均方误差。 |
| [^14] | [Robust Knowledge Transfer in Tiered Reinforcement Learning.](http://arxiv.org/abs/2302.05534) | 本文研究了层级增强学习中的知识传输，提出了一种新颖的在线学习算法，在没有先验知识的任务相似性的情况下实现强大的知识传输。 |

# 详细

[^1]: 深度条件生成学习：模型与误差分析

    Deep Conditional Generative Learning: Model and Error Analysis

    [https://rss.arxiv.org/abs/2402.01460](https://rss.arxiv.org/abs/2402.01460)

    提出了一种基于ODE的深度生成方法，通过条件Follmer流来学习条件分布，通过离散化和深度神经网络实现高效转化。同时，通过Wasserstein距离的非渐近收敛速率，提供了第一个端到端误差分析，数值实验证明其在不同场景下的优越性。

    

    我们介绍了一种基于普通微分方程（ODE）的深度生成方法，用于学习条件分布，称为条件Follmer流。从标准高斯分布开始，所提出的流能够以高效的方式将其转化为目标条件分布，在时间1处达到稳定。为了有效实现，我们使用欧拉方法对流进行离散化，使用深度神经网络非参数化估计速度场。此外，我们导出了学习样本的分布与目标分布之间的Wasserstein距离的非渐近收敛速率，在条件分布学习中提供了第一个全面的端到端误差分析。我们的数值实验展示了它在一系列情况下的有效性，从标准的非参数化条件密度估计问题到涉及图像数据的更复杂的挑战，说明它优于各种现有方法。

    We introduce an Ordinary Differential Equation (ODE) based deep generative method for learning a conditional distribution, named the Conditional Follmer Flow. Starting from a standard Gaussian distribution, the proposed flow could efficiently transform it into the target conditional distribution at time 1. For effective implementation, we discretize the flow with Euler's method where we estimate the velocity field nonparametrically using a deep neural network. Furthermore, we derive a non-asymptotic convergence rate in the Wasserstein distance between the distribution of the learned samples and the target distribution, providing the first comprehensive end-to-end error analysis for conditional distribution learning via ODE flow. Our numerical experiments showcase its effectiveness across a range of scenarios, from standard nonparametric conditional density estimation problems to more intricate challenges involving image data, illustrating its superiority over various existing condition
    
[^2]: 机器学习中的函数双层优化

    Functional Bilevel Optimization for Machine Learning

    [https://arxiv.org/abs/2403.20233](https://arxiv.org/abs/2403.20233)

    介绍了机器学习中的函数双层优化问题，提出了不依赖于强凸假设的方法，并展示了在仪表回归和强化学习任务中使用神经网络的优势。

    

    在本文中，我们介绍了针对机器学习中的双层优化问题的一种新的函数视角，其中内部目标在函数空间上被最小化。这些类型的问题通常通过在参数设置下开发的方法来解决，其中内部目标对于预测函数的参数强凸。函数视角不依赖于此假设，特别允许使用超参数化的神经网络作为内部预测函数。我们提出了可扩展和高效的算法来解决函数双层优化问题，并展示了我们方法在适合自然函数双层结构的仪表回归和强化学习任务上的优势。

    arXiv:2403.20233v1 Announce Type: cross  Abstract: In this paper, we introduce a new functional point of view on bilevel optimization problems for machine learning, where the inner objective is minimized over a function space. These types of problems are most often solved by using methods developed in the parametric setting, where the inner objective is strongly convex with respect to the parameters of the prediction function. The functional point of view does not rely on this assumption and notably allows using over-parameterized neural networks as the inner prediction function. We propose scalable and efficient algorithms for the functional bilevel optimization problem and illustrate the benefits of our approach on instrumental regression and reinforcement learning tasks, which admit natural functional bilevel structures.
    
[^3]: 面向模型无关后验逼近的快速准确变分自编码器

    Towards Model-Agnostic Posterior Approximation for Fast and Accurate Variational Autoencoders

    [https://arxiv.org/abs/2403.08941](https://arxiv.org/abs/2403.08941)

    最近的研究表明，为了确保高质量的推断模型，可以通过迭代训练最大化与推断模型相关的目标函数，以解决变分自编码器中推断模型近似不准确导致的局部最优解问题。

    

    变分自编码器（VAEs）的推断包括学习两个模型：（1）生成模型，将潜在空间上的简单分布转换为观测数据分布，以及（2）推断模型，近似给定数据的潜在编码后验。这两个组件通过对生成模型对数边际似然的下界进行联合学习。在联合训练的早期阶段，推断模型很差地近似了潜在编码后验。最近的研究表明，这导致优化陷入局部最优解，对学习到的生成模型造成负面影响。因此，最近的研究建议通过迭代训练确保高质量的推断模型：相对于生成模型的每次更新之前最大化与推断模型相关的目标函数。不幸的是，迭代训练效率低，需要启发式标准来从迭代中恢复。

    arXiv:2403.08941v1 Announce Type: cross  Abstract: Inference for Variational Autoencoders (VAEs) consists of learning two models: (1) a generative model, which transforms a simple distribution over a latent space into the distribution over observed data, and (2) an inference model, which approximates the posterior of the latent codes given data. The two components are learned jointly via a lower bound to the generative model's log marginal likelihood. In early phases of joint training, the inference model poorly approximates the latent code posteriors. Recent work showed that this leads optimization to get stuck in local optima, negatively impacting the learned generative model. As such, recent work suggests ensuring a high-quality inference model via iterative training: maximizing the objective function relative to the inference model before every update to the generative model. Unfortunately, iterative training is inefficient, requiring heuristic criteria for reverting from iterative
    
[^4]: 论Brenier的极分解的神经实现

    On a Neural Implementation of Brenier's Polar Factorization

    [https://arxiv.org/abs/2403.03071](https://arxiv.org/abs/2403.03071)

    提出了Brenier的极分解定理的神经实现，探讨了在机器学习中的应用，并通过神经网络参数化潜在函数$u$，从最新神经最优输运领域的进展中汲取灵感。

    

    在1991年，Brenier证明了一个定理，将$QR$分解（分为半正定矩阵$\times$酉矩阵）推广到任意矢量场$F:\mathbb{R}^d\rightarrow \mathbb{R}^d$。这个被称为极分解定理的定理表明，任意场$F$都可以表示为凸函数$u$的梯度与保测度映射$M$的复合，即$F=\nabla u \circ M$。我们提出了这一具有深远理论意义的结果的实际实现，并探讨了在机器学习中可能的应用。该定理与最优输运（OT）理论密切相关，我们借鉴了神经最优输运领域的最新进展，将潜在函数$u$参数化为输入凸神经网络。映射$M$可以通过使用$u^*$，即$u$的凸共轭，逐点计算得到，即$M=\nabla u^* \circ F$，或者作为辅助网络学习得到。因为$M$在基因

    arXiv:2403.03071v1 Announce Type: cross  Abstract: In 1991, Brenier proved a theorem that generalizes the $QR$ decomposition for square matrices -- factored as PSD $\times$ unitary -- to any vector field $F:\mathbb{R}^d\rightarrow \mathbb{R}^d$. The theorem, known as the polar factorization theorem, states that any field $F$ can be recovered as the composition of the gradient of a convex function $u$ with a measure-preserving map $M$, namely $F=\nabla u \circ M$. We propose a practical implementation of this far-reaching theoretical result, and explore possible uses within machine learning. The theorem is closely related to optimal transport (OT) theory, and we borrow from recent advances in the field of neural optimal transport to parameterize the potential $u$ as an input convex neural network. The map $M$ can be either evaluated pointwise using $u^*$, the convex conjugate of $u$, through the identity $M=\nabla u^* \circ F$, or learned as an auxiliary network. Because $M$ is, in gene
    
[^5]: 通过无监督预训练和上下文学习实现高效的运算符学习

    Data-Efficient Operator Learning via Unsupervised Pretraining and In-Context Learning

    [https://arxiv.org/abs/2402.15734](https://arxiv.org/abs/2402.15734)

    该论文提出了一种通过无监督预训练和上下文学习方法实现PDE运算符学习的高效方式，以提高数据效率并改善模型的外域性能。

    

    近年来，人们见证了将机器学习方法与物理领域特定洞察力相结合，以解决基于偏微分方程（PDEs）的科学问题的潜力。然而，由于数据密集，这些方法仍然需要大量PDE数据。 这重新引入了对昂贵的数值PDE解决方案的需求，部分削弱了避免这些昂贵模拟的原始目标。 在这项工作中，为了寻求数据效率，我们设计了用于PDE运算符学习的无监督预训练和上下文学习方法。 为了减少对带有模拟解的训练数据的需求，我们使用基于重构的代理任务在未标记的PDE数据上预训练神经运算符。 为了提高超出分布性能，我们进一步帮助神经运算符灵活地利用上下文学习方法，而无需额外的训练成本或设计。 在各种PD上进行了大量实证评估

    arXiv:2402.15734v1 Announce Type: new  Abstract: Recent years have witnessed the promise of coupling machine learning methods and physical domain-specific insight for solving scientific problems based on partial differential equations (PDEs). However, being data-intensive, these methods still require a large amount of PDE data. This reintroduces the need for expensive numerical PDE solutions, partially undermining the original goal of avoiding these expensive simulations. In this work, seeking data efficiency, we design unsupervised pretraining and in-context learning methods for PDE operator learning. To reduce the need for training data with simulated solutions, we pretrain neural operators on unlabeled PDE data using reconstruction-based proxy tasks. To improve out-of-distribution performance, we further assist neural operators in flexibly leveraging in-context learning methods, without incurring extra training costs or designs. Extensive empirical evaluations on a diverse set of PD
    
[^6]: 通过混合狄利克雷分布改进证据深度学习

    Improved Evidential Deep Learning via a Mixture of Dirichlet Distributions

    [https://arxiv.org/abs/2402.06160](https://arxiv.org/abs/2402.06160)

    本文通过混合狄利克雷分布来改进证据深度学习（EDL）方法，解决了现有方法中认知不确定性在无限样本限制下可能不会消失的问题。

    

    本文探讨了一种现代的预测不确定性估计方法，称为证据深度学习（EDL），其中通过最小化特定的目标函数，训练单个神经网络模型以学习预测分布上的元分布。尽管现有方法在经验性能方面表现强大，但Bengs等人的最近研究发现了现有方法的一个根本缺陷：即使在无限样本限制下，学习到的认知不确定性可能不会消失。通过提供文献中一类广泛使用的目标函数的统一视角，我们得到了这个观察的证实。我们的分析揭示了EDL方法本质上通过最小化分布与与样本大小无关的目标分布之间的特定差异度量来训练元分布，从而产生错误的认知不确定性。基于理论原则，我们提出通过将其建模为狄利克雷分布混合物来学习一致目标分布，从而改进了EDL方法。

    This paper explores a modern predictive uncertainty estimation approach, called evidential deep learning (EDL), in which a single neural network model is trained to learn a meta distribution over the predictive distribution by minimizing a specific objective function. Despite their strong empirical performance, recent studies by Bengs et al. identify a fundamental pitfall of the existing methods: the learned epistemic uncertainty may not vanish even in the infinite-sample limit. We corroborate the observation by providing a unifying view of a class of widely used objectives from the literature. Our analysis reveals that the EDL methods essentially train a meta distribution by minimizing a certain divergence measure between the distribution and a sample-size-independent target distribution, resulting in spurious epistemic uncertainty. Grounded in theoretical principles, we propose learning a consistent target distribution by modeling it with a mixture of Dirichlet distributions and lear
    
[^7]: 依赖学习理论中的尖锐率：避免样本大小缩减的平方损失

    Sharp Rates in Dependent Learning Theory: Avoiding Sample Size Deflation for the Square Loss

    [https://arxiv.org/abs/2402.05928](https://arxiv.org/abs/2402.05928)

    本文研究了依赖学习理论中的尖锐率，主要是为了避免样本大小缩减对方差产生影响。当假设类别的拓扑结构符合某些条件时，经验风险最小化者的性能与类别的复杂性和二阶统计量有关。

    

    本文研究了具有依赖性（β-混合）数据和平方损失的统计学习，在一个假设类别Φ_p的子集F中，其中Φ_p是范数∥f∥_Φ_p≡sup_m≥1 m^{-1/p}∥f∥_L^m，其中p∈[2，∞]。我们的研究动机是在具有依赖性数据的学习中寻找尖锐的噪声交互项或方差代理。在没有任何可实现性假设的情况下，典型的非渐近结果显示出方差代理通过底层协变量过程的混合时间进行了乘积缩减。我们证明，只要在我们的假设类别F上，L^2和Φ_p的拓扑是可比较的，即Φ_p是一个弱亚高斯类别：∥f∥_Φ_p≲∥f∥_L^2^η，其中η∈(0，1]，经验风险最小化者在其主导项中只实现了一种只依赖于类别复杂性和二阶统计量的速率。我们的结果适用于许多依赖性数据模型。

    In this work, we study statistical learning with dependent ($\beta$-mixing) data and square loss in a hypothesis class $\mathscr{F}\subset L_{\Psi_p}$ where $\Psi_p$ is the norm $\|f\|_{\Psi_p} \triangleq \sup_{m\geq 1} m^{-1/p} \|f\|_{L^m} $ for some $p\in [2,\infty]$. Our inquiry is motivated by the search for a sharp noise interaction term, or variance proxy, in learning with dependent data. Absent any realizability assumption, typical non-asymptotic results exhibit variance proxies that are deflated \emph{multiplicatively} by the mixing time of the underlying covariates process. We show that whenever the topologies of $L^2$ and $\Psi_p$ are comparable on our hypothesis class $\mathscr{F}$ -- that is, $\mathscr{F}$ is a weakly sub-Gaussian class: $\|f\|_{\Psi_p} \lesssim \|f\|_{L^2}^\eta$ for some $\eta\in (0,1]$ -- the empirical risk minimizer achieves a rate that only depends on the complexity of the class and second order statistics in its leading term. Our result holds whether t
    
[^8]: 学习最大化加速A/B测试的指标

    Learning Metrics that Maximise Power for Accelerated A/B-Tests

    [https://arxiv.org/abs/2402.03915](https://arxiv.org/abs/2402.03915)

    本论文提出了一种新方法，通过从短期信号中学习指标，直接最大化指标与北极度量标准之间的统计能力，从而减少在线控制实验的成本。

    

    在技术公司中，在线控制实验是一种重要的工具，可以实现自信的决策。定义了一个北极度量标准（如长期收入或用户保留），在A/B测试中，能够在这个指标上有统计显著提升的系统变体可以被认为是优越的。然而，北极度量标准通常具有时延和不敏感性。因此，实验的成本很高：实验需要长时间运行，即使如此，二类错误（即假阴性）仍然普遍存在。为了解决这个问题，我们提出了一种从短期信号中学习指标的方法，这些指标直接最大化它们相对于北极度量标准所具有的统计能力。我们展示了现有方法容易过拟合的问题，即更高的平均度量敏感性并不意味着改进了二类错误，我们建议通过最小化指标在过去实验的$log$上产生的$p$-value来解决。我们从两个社交媒体应用程序中收集了这样的数据集。

    Online controlled experiments are a crucial tool to allow for confident decision-making in technology companies. A North Star metric is defined (such as long-term revenue or user retention), and system variants that statistically significantly improve on this metric in an A/B-test can be considered superior. North Star metrics are typically delayed and insensitive. As a result, the cost of experimentation is high: experiments need to run for a long time, and even then, type-II errors (i.e. false negatives) are prevalent.   We propose to tackle this by learning metrics from short-term signals that directly maximise the statistical power they harness with respect to the North Star. We show that existing approaches are prone to overfitting, in that higher average metric sensitivity does not imply improved type-II errors, and propose to instead minimise the $p$-values a metric would have produced on a log of past experiments. We collect such datasets from two social media applications with
    
[^9]: Flora: 低秩适配器是悄悄的梯度压缩器

    Flora: Low-Rank Adapters Are Secretly Gradient Compressors

    [https://arxiv.org/abs/2402.03293](https://arxiv.org/abs/2402.03293)

    本文研究了低秩适配器的动力学，并提出了一种基于随机投影的方法Flora，通过重新采样投影矩阵实现高秩更新，同时减少优化状态的空间复杂度。

    

    尽管大型神经网络展示了完成不同任务的显着能力，但它们需要过多的内存使用来存储训练的优化状态。为了缓解这个问题，提出低秩适配（LoRA）来通过训练更少的参数来减少优化状态。然而，LoRA将整体权重更新矩阵限制为低秩，限制了模型的性能。在这项工作中，我们研究了LoRA的动力学，并确定它可以近似为随机投影。基于这一观察，我们提出了Flora，它能够通过重新采样投影矩阵实现高秩更新，同时享受优化状态的次线性空间复杂度。我们在不同任务和模型架构上进行实验证实了我们方法的有效性。

    Despite large neural networks demonstrating remarkable abilities to complete different tasks, they require excessive memory usage to store the optimization states for training. To alleviate this, the low-rank adaptation (LoRA) is proposed to reduce the optimization states by training fewer parameters. However, LoRA restricts overall weight update matrices to be low-rank, limiting the model performance. In this work, we investigate the dynamics of LoRA and identify that it can be approximated by a random projection. Based on this observation, we propose Flora, which is able to achieve high-rank updates by resampling the projection matrices while enjoying the sublinear space complexity of optimization states. We conduct experiments across different tasks and model architectures to verify the effectiveness of our approach.
    
[^10]: 基于热和波动动力学特征的图拓扑属性恢复

    Graph topological property recovery with heat and wave dynamics-based features on graphsD. (arXiv:2309.09924v1 [cs.LG])

    [http://arxiv.org/abs/2309.09924](http://arxiv.org/abs/2309.09924)

    本文提出了一种名为图微分方程网络（GDeNet）的方法，利用热和波动方程动力学特征来恢复图的拓扑属性，能够在各种下游任务中获得优秀的表现，同时在实际应用中也展现了较好的性能。

    

    本文提出了一种名为图微分方程网络（GDeNet）的方法，利用图上的PDE解的表达能力，为各种下游任务获得连续的节点和图级表示。我们推导出了热和波动方程动力学与图的谱特性以及连续时间随机游走在图上行为之间的理论结果。我们通过恢复随机图生成参数、Ricci曲率和持久同调等方式实验证明了这些动力学能够捕捉到图形几何和拓扑的显著方面。此外，我们还展示了GDeNet在包括引用图、药物分子和蛋白质在内的真实世界数据集上的优越性能。

    In this paper, we propose Graph Differential Equation Network (GDeNet), an approach that harnesses the expressive power of solutions to PDEs on a graph to obtain continuous node- and graph-level representations for various downstream tasks. We derive theoretical results connecting the dynamics of heat and wave equations to the spectral properties of the graph and to the behavior of continuous-time random walks on graphs. We demonstrate experimentally that these dynamics are able to capture salient aspects of graph geometry and topology by recovering generating parameters of random graphs, Ricci curvature, and persistent homology. Furthermore, we demonstrate the superior performance of GDeNet on real-world datasets including citation graphs, drug-like molecules, and proteins.
    
[^11]: 无标记多模态数据的多模态学习：保证和应用

    Multimodal Learning Without Labeled Multimodal Data: Guarantees and Applications. (arXiv:2306.04539v1 [cs.LG])

    [http://arxiv.org/abs/2306.04539](http://arxiv.org/abs/2306.04539)

    本文研究在只有带标签的单模态数据和自然出现的多模态数据的情况下，如何量化多模态交互的挑战，并提出了两个下界和一个上界来量化多模态交互量。

    

    在许多共同学习多个模态的机器学习系统中，一个核心的研究问题是理解多模态交互的本质：在从两个都没有的模态学习时出现了新的任务相关信息。我们在半监督的情况下研究这一交互量化的挑战，只使用带标签的单模态数据和自然出现的多模态数据（例如，无标签的图像和标题，视频和相应的音频）。利用精确的信息论交互定义，我们的主要贡献是推导下界和上界，量化这种半监督设置下的多模态交互量。我们提出了基于模态共享信息量和单独训练的单模态分类器之间的不一致性的两个下界，并通过连接到近似算法来推导上界。

    In many machine learning systems that jointly learn from multiple modalities, a core research question is to understand the nature of multimodal interactions: the emergence of new task-relevant information during learning from both modalities that was not present in either alone. We study this challenge of interaction quantification in a semi-supervised setting with only labeled unimodal data and naturally co-occurring multimodal data (e.g., unlabeled images and captions, video and corresponding audio) but when labeling them is time-consuming. Using a precise information-theoretic definition of interactions, our key contributions are the derivations of lower and upper bounds to quantify the amount of multimodal interactions in this semi-supervised setting. We propose two lower bounds based on the amount of shared information between modalities and the disagreement between separately trained unimodal classifiers, and derive an upper bound through connections to approximate algorithms fo
    
[^12]: 分布式SGD算法的稳定性与泛化分析改进

    Improved Stability and Generalization Analysis of the Decentralized SGD Algorithm. (arXiv:2306.02939v1 [cs.LG])

    [http://arxiv.org/abs/2306.02939](http://arxiv.org/abs/2306.02939)

    本文提出了新的算法稳定性理论来改进分布式SGD算法的泛化性能分析，推翻了现有技术对通信图负面影响的观点，并展示了D-SGD在凸设置中与经典SGD算法泛化界相同。

    

    本文基于算法稳定性，提出了分布式随机梯度下降(D-SGD)算法的新的泛化误差分析方法。得到的结果大大改进了现有技术，并推翻了它们关于通信图对泛化的负面影响的观点。例如，在凸设置中，无论图的选择如何，D-SGD具有与经典SGD算法相同的泛化界。我们发现这种反直觉的结果来自于考虑本地参数的平均值，这会隐藏一个与分布式场景不兼容的最终全局平均化步骤。考虑到这一观察结果，我们倡导分析本地参数的上确界，并展示了在这种情况下，图确实对泛化产生影响。与之前的结果不同，我们的分析即使对于非连接图也能产生非平凡边界。

    This paper presents a new generalization error analysis for the Decentralized Stochastic Gradient Descent (D-SGD) algorithm based on algorithmic stability. The obtained results largely improve upon state-of-the-art results, and even invalidate their claims that the communication graph has a detrimental effect on generalization. For instance, we show that in convex settings, D-SGD has the same generalization bounds as the classical SGD algorithm, no matter the choice of graph. We exhibit that this counter-intuitive result comes from considering the average of local parameters, which hides a final global averaging step incompatible with the decentralized scenario. In light of this observation, we advocate to analyze the supremum over local parameters and show that in this case, the graph does have an impact on the generalization. Unlike prior results, our analysis yields non-vacuous bounds even for non-connected graphs.
    
[^13]: 基于一般回归误差假设来研究无噪声回归最小二乘估计值的均方误差

    The Mean Squared Error of the Ridgeless Least Squares Estimator under General Assumptions on Regression Errors. (arXiv:2305.12883v1 [math.ST])

    [http://arxiv.org/abs/2305.12883](http://arxiv.org/abs/2305.12883)

    该论文研究了基于一般回归误差假设的无噪声回归最小二乘估计值的均方误差，并发现包含大量不重要的参数可以有效地降低估计器的均方误差。

    

    近年来，最小$\ell_2$范数（无岭）插值最小二乘估计器的研究方兴未艾。然而，大多数分析都局限于简单的回归误差结构，假设误差是独立同分布的，具有零均值和相同的方差，与特征向量无关。此外，这些理论分析的主要重点是样本外预测风险。本文通过检查无岭插值最小二乘估计器的均方误差，允许更一般的回归误差假设，打破了现有文献的局限性。具体而言，我们研究过度参数化的潜在好处，通过描绘有限样本中的均方误差来表征均方误差。我们的研究结果表明，相对于样本量，包含大量不重要的参数可以有效地降低估计器的均方误差。

    In recent years, there has been a significant growth in research focusing on minimum $\ell_2$ norm (ridgeless) interpolation least squares estimators. However, the majority of these analyses have been limited to a simple regression error structure, assuming independent and identically distributed errors with zero mean and common variance, independent of the feature vectors. Additionally, the main focus of these theoretical analyses has been on the out-of-sample prediction risk. This paper breaks away from the existing literature by examining the mean squared error of the ridgeless interpolation least squares estimator, allowing for more general assumptions about the regression errors. Specifically, we investigate the potential benefits of overparameterization by characterizing the mean squared error in a finite sample. Our findings reveal that including a large number of unimportant parameters relative to the sample size can effectively reduce the mean squared error of the estimator. N
    
[^14]: 强大的层级增强学习中的知识传输

    Robust Knowledge Transfer in Tiered Reinforcement Learning. (arXiv:2302.05534v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.05534](http://arxiv.org/abs/2302.05534)

    本文研究了层级增强学习中的知识传输，提出了一种新颖的在线学习算法，在没有先验知识的任务相似性的情况下实现强大的知识传输。

    

    本文研究了层级增强学习设置，这是一个并行传输学习框架，在这个框架中，目标是将知识从低层（源）任务传输到高层（目标）任务，以减少后者的探索风险，同时并行解决这两个任务。与先前的工作不同，我们不假设低层和高层任务共享相同的动态或奖励函数，并且专注于在没有先验知识的任务相似性的情况下实现强大的知识传输。我们确定了一个称为“最优值支配”的自然而必要的条件，适用于我们的目标。在这个条件下，我们提出了一种新颖的在线学习算法，使得对于高层任务，在部分状态上可以实现恒定的遗憾，这取决于任务相似性，并在两个任务不相似时保持接近最优遗憾；而对于低层任务，它可以在不做出牺牲的情况下保持接近最优。此外，我们进一步研究了具有多个低层任务的情况。

    In this paper, we study the Tiered Reinforcement Learning setting, a parallel transfer learning framework, where the goal is to transfer knowledge from the low-tier (source) task to the high-tier (target) task to reduce the exploration risk of the latter while solving the two tasks in parallel. Unlike previous work, we do not assume the low-tier and high-tier tasks share the same dynamics or reward functions, and focus on robust knowledge transfer without prior knowledge on the task similarity. We identify a natural and necessary condition called the ``Optimal Value Dominance'' for our objective. Under this condition, we propose novel online learning algorithms such that, for the high-tier task, it can achieve constant regret on partial states depending on the task similarity and retain near-optimal regret when the two tasks are dissimilar, while for the low-tier task, it can keep near-optimal without making sacrifice. Moreover, we further study the setting with multiple low-tier tasks
    

