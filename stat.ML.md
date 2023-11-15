# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Generalization Properties of Diffusion Models.](http://arxiv.org/abs/2311.01797) | 本文对扩散模型的泛化属性进行了理论研究，建立了基于评分法的扩散模型的训练动态中泛化差距的理论估计，并在停止训练时可以避免维度诅咒。进一步将定量分析扩展到了数据依赖的情景。 |
| [^2] | [High-probability Convergence Bounds for Nonlinear Stochastic Gradient Descent Under Heavy-tailed Noise.](http://arxiv.org/abs/2310.18784) | 本研究探讨了一类非线性随机梯度下降方法的高概率收敛边界。对于具有Lipschitz连续梯度的强凸损失函数，即使噪声是重尾的，结果证明了对失败概率的对数依赖。这些结果适用于剪切、归一化和量化等任何具有有界输出的非线性函数。 |
| [^3] | [On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers.](http://arxiv.org/abs/2310.14421) | 本文研究了针对AI分类器的对抗鲁棒性度量的存在性、唯一性和可扩展性，提出了可以验证的数学条件，并在合成基准测试和生物医学应用中进行了实际计算和解释。 |
| [^4] | [Towards the Fundamental Limits of Knowledge Transfer over Finite Domains.](http://arxiv.org/abs/2310.07838) | 本论文研究了在有限领域中从教师到学生分类器进行知识传递的统计效率，发现特权信息会加速传递，通过使用一种新颖的损失函数达到了知识传递的基本限制。 |
| [^5] | [A Corrected Expected Improvement Acquisition Function Under Noisy Observations.](http://arxiv.org/abs/2310.05166) | 这个论文提出了一个修正的期望改善采集函数，在贝叶斯优化中解决了对于有噪声观测的情况下忽略候选解不确定性的问题。 |
| [^6] | [Unbiased Learning of Deep Generative Models with Structured Discrete Representations.](http://arxiv.org/abs/2306.08230) | 该论文提出了一种名为结构化变分自编码器的深度生成模型，它通过图像模型的结构和可解释性以及深度学习的适用于高维数据的灵活似然，结合两种框架的优势。同时，该论文还提出了一种学习SVAE的新算法，与此同时，推导出了一种计算自然梯度的方法，这些优化创新使得SVAE首次能与最先进的时间序列模型进行比较。 |
| [^7] | [Axiomatization of Interventional Probability Distributions.](http://arxiv.org/abs/2305.04479) | 本论文提供了一种简单和清晰的因果理论，它不需要使用任何建模假设，包括大多数具有潜在变量和因果循环的情况，并且不假定存在底层真正的因果图-事实上，它是因果图的副产品。 |
| [^8] | [Generalized partitioned local depth.](http://arxiv.org/abs/2303.10167) | 本文提出了一个广义的凝聚概念，构建在分区局部深度的技术基础上，扩展了早期结果并应用于具有不确定性的数据的社区发现中。 |
| [^9] | [PAC-Bayesian Generalization Bounds for Adversarial Generative Models.](http://arxiv.org/abs/2302.08942) | 将PAC-Bayesian理论扩展到生成模型，为基于Wasserstein距离和总变差距离的模型提供了泛化界，为Wasserstein GAN和Energy-Based GAN提供了新的训练目标，并在合成数据集上展示出非虚空泛化界。 |
| [^10] | [A Theoretical Understanding of Shallow Vision Transformers: Learning, Generalization, and Sample Complexity.](http://arxiv.org/abs/2302.06015) | 本文提供了第一份对于浅层ViT进行训练的理论分析，证明了使用SGD训练会产生稀疏的注意力图，目前的样本复杂度与标记相关令牌的分数倒数、标记级别的令牌噪声水平和初始模型错误呈正相关关系。 |

# 详细

[^1]: 关于扩散模型的泛化属性

    On the Generalization Properties of Diffusion Models. (arXiv:2311.01797v1 [cs.LG])

    [http://arxiv.org/abs/2311.01797](http://arxiv.org/abs/2311.01797)

    本文对扩散模型的泛化属性进行了理论研究，建立了基于评分法的扩散模型的训练动态中泛化差距的理论估计，并在停止训练时可以避免维度诅咒。进一步将定量分析扩展到了数据依赖的情景。

    

    扩散模型是一类生成模型，用于建立一个随机传输映射，将经验观测到的但未知的目标分布与已知的先验分布联系起来。尽管在实际应用中取得了显著的成功，但对其泛化能力的理论理解仍未充分发展。本文对扩散模型的泛化属性进行了全面的理论研究。我们建立了基于评分法的扩散模型的训练动态中泛化差距的理论估计，表明在样本大小$n$和模型容量$m$上都存在多项式小的泛化误差($O(n^{-2/5}+m^{-4/5})$)，在停止训练时可以避免维度诅咒（即数据维度不呈指数级增长）。此外，我们将定量分析扩展到了一个数据依赖的情景，其中目标分布被描绘为一系列的概率密度函数。

    Diffusion models are a class of generative models that serve to establish a stochastic transport map between an empirically observed, yet unknown, target distribution and a known prior. Despite their remarkable success in real-world applications, a theoretical understanding of their generalization capabilities remains underdeveloped. This work embarks on a comprehensive theoretical exploration of the generalization attributes of diffusion models. We establish theoretical estimates of the generalization gap that evolves in tandem with the training dynamics of score-based diffusion models, suggesting a polynomially small generalization error ($O(n^{-2/5}+m^{-4/5})$) on both the sample size $n$ and the model capacity $m$, evading the curse of dimensionality (i.e., not exponentially large in the data dimension) when early-stopped. Furthermore, we extend our quantitative analysis to a data-dependent scenario, wherein target distributions are portrayed as a succession of densities with progr
    
[^2]: 高概率收敛边界下的非线性随机梯度下降在重尾噪声下的研究

    High-probability Convergence Bounds for Nonlinear Stochastic Gradient Descent Under Heavy-tailed Noise. (arXiv:2310.18784v1 [cs.LG])

    [http://arxiv.org/abs/2310.18784](http://arxiv.org/abs/2310.18784)

    本研究探讨了一类非线性随机梯度下降方法的高概率收敛边界。对于具有Lipschitz连续梯度的强凸损失函数，即使噪声是重尾的，结果证明了对失败概率的对数依赖。这些结果适用于剪切、归一化和量化等任何具有有界输出的非线性函数。

    

    最近几个研究工作研究了随机梯度下降（SGD）及其剪切变体的高概率收敛。与普通的SGD相比，剪切SGD在实际中更加稳定，并且在理论上有对数依赖于失败概率的额外好处。然而，其他实际非线性SGD变体（如符号SGD、量化SGD和归一化SGD）的收敛性理解要少得多，这些方法实现了改进的通信效率或加速收敛。在本工作中，我们研究了一类广义非线性SGD方法的高概率收敛边界。对于具有Lipschitz连续梯度的强凸损失函数，即使噪声是重尾的，我们证明了对失败概率的对数依赖。与剪切SGD的结果相比，我们的结果更为一般，适用于具有有界输出的任何非线性函数，如剪切、归一化和量化。

    Several recent works have studied the convergence \textit{in high probability} of stochastic gradient descent (SGD) and its clipped variant. Compared to vanilla SGD, clipped SGD is practically more stable and has the additional theoretical benefit of logarithmic dependence on the failure probability. However, the convergence of other practical nonlinear variants of SGD, e.g., sign SGD, quantized SGD and normalized SGD, that achieve improved communication efficiency or accelerated convergence is much less understood. In this work, we study the convergence bounds \textit{in high probability} of a broad class of nonlinear SGD methods. For strongly convex loss functions with Lipschitz continuous gradients, we prove a logarithmic dependence on the failure probability, even when the noise is heavy-tailed. Strictly more general than the results for clipped SGD, our results hold for any nonlinearity with bounded (component-wise or joint) outputs, such as clipping, normalization, and quantizati
    
[^3]: 对AI分类器的对抗鲁棒性度量的存在性，唯一性和可扩展性研究

    On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers. (arXiv:2310.14421v1 [stat.ML])

    [http://arxiv.org/abs/2310.14421](http://arxiv.org/abs/2310.14421)

    本文研究了针对AI分类器的对抗鲁棒性度量的存在性、唯一性和可扩展性，提出了可以验证的数学条件，并在合成基准测试和生物医学应用中进行了实际计算和解释。

    

    本文提出并证明了针对（局部）唯一可逆分类器、广义线性模型（GLM）和熵AI（EAI）具有最小对抗路径（MAP）和最小对抗距离（MAD）的存在性、唯一性和明确的分析计算的简单可验证的数学条件。在常见的合成基准测试数据集上，针对神经网络、提升随机森林、GLM和EAI等各类AI工具进行MAP和MAD的实际计算、比较和解释，包括双卷状螺旋线及其扩展以及两个生物医学数据问题（用于健康保险理赔预测和心脏病发作致死率分类）。在生物医学应用中，展示了MAP如何在预定义的可访问控制变量子集中提供唯一的最小患者特定风险缓解干预措施。

    Simply-verifiable mathematical conditions for existence, uniqueness and explicit analytical computation of minimal adversarial paths (MAP) and minimal adversarial distances (MAD) for (locally) uniquely-invertible classifiers, for generalized linear models (GLM), and for entropic AI (EAI) are formulated and proven. Practical computation of MAP and MAD, their comparison and interpretations for various classes of AI tools (for neuronal networks, boosted random forests, GLM and EAI) are demonstrated on the common synthetic benchmarks: on a double Swiss roll spiral and its extensions, as well as on the two biomedical data problems (for the health insurance claim predictions, and for the heart attack lethality classification). On biomedical applications it is demonstrated how MAP provides unique minimal patient-specific risk-mitigating interventions in the predefined subsets of accessible control variables.
    
[^4]: 探索有限领域知识传递的基本限制

    Towards the Fundamental Limits of Knowledge Transfer over Finite Domains. (arXiv:2310.07838v1 [cs.LG])

    [http://arxiv.org/abs/2310.07838](http://arxiv.org/abs/2310.07838)

    本论文研究了在有限领域中从教师到学生分类器进行知识传递的统计效率，发现特权信息会加速传递，通过使用一种新颖的损失函数达到了知识传递的基本限制。

    

    我们对通过从教师到概率化学生分类器的n个样本进行知识传递的统计效率进行了表征，其中输入空间S和标签A为有限域。我们发现，在三个渐进级别上的特权信息可以加快传递的速度。在第一级别上，只有具有困难标签的样本是已知的，最大似然估计器能够达到最小化速率sqrt(|S||A|/n)。第二级别上，除了已知的困难标签样本外，还有采样标签的教师概率可用，这将收敛速度的下界提高到|S||A|/n。然而，在第二个数据采集协议下，最小化交叉熵损失的朴素适应会导致渐近偏差的学生。我们克服了这个限制，并通过使用一种新颖的经验变体的平方误差逻辑损失来实现了基本限制。第三级别进一步赋予学生软标签。

    We characterize the statistical efficiency of knowledge transfer through $n$ samples from a teacher to a probabilistic student classifier with input space $\mathcal S$ over labels $\mathcal A$. We show that privileged information at three progressive levels accelerates the transfer. At the first level, only samples with hard labels are known, via which the maximum likelihood estimator attains the minimax rate $\sqrt{{|{\mathcal S}||{\mathcal A}|}/{n}}$. The second level has the teacher probabilities of sampled labels available in addition, which turns out to boost the convergence rate lower bound to ${{|{\mathcal S}||{\mathcal A}|}/{n}}$. However, under this second data acquisition protocol, minimizing a naive adaptation of the cross-entropy loss results in an asymptotically biased student. We overcome this limitation and achieve the fundamental limit by using a novel empirical variant of the squared error logit loss. The third level further equips the student with the soft labels (com
    
[^5]: 一个在有噪声观测下修正的期望改善采集函数

    A Corrected Expected Improvement Acquisition Function Under Noisy Observations. (arXiv:2310.05166v1 [cs.LG])

    [http://arxiv.org/abs/2310.05166](http://arxiv.org/abs/2310.05166)

    这个论文提出了一个修正的期望改善采集函数，在贝叶斯优化中解决了对于有噪声观测的情况下忽略候选解不确定性的问题。

    

    序列最大化期望改善(EI)是贝叶斯优化中最常用的策略之一，因其简单性和处理噪声观测的能力而广泛应用。特别是，在噪声环境中，改善函数通常使用最佳后验均值作为最佳候选解。然而，在许多解析的EI类型方法中，常常忽略与候选解相关的不确定性：在无噪声的情况下导出了一个闭合形式的采集函数，然后应用于有噪声观测的情况。为了解决这个限制，我们提出了一种修正EI的方法，将高斯过程(GP)模型提供的协方差信息纳入其闭合形式表达式中。这个采集函数与经典的无噪声结果相吻合，我们认为它应该取代贝叶斯优化软件包、教程和教材中的那个公式。这个改进的采集函数为有噪声和无噪声的解提供了良好的适用性。

    Sequential maximization of expected improvement (EI) is one of the most widely used policies in Bayesian optimization because of its simplicity and ability to handle noisy observations. In particular, the improvement function often uses the best posterior mean as the best incumbent in noisy settings. However, the uncertainty associated with the incumbent solution is often neglected in many analytic EI-type methods: a closed-form acquisition function is derived in the noise-free setting, but then applied to the setting with noisy observations. To address this limitation, we propose a modification of EI that corrects its closed-form expression by incorporating the covariance information provided by the Gaussian Process (GP) model. This acquisition function specializes to the classical noise-free result, and we argue should replace that formula in Bayesian optimization software packages, tutorials, and textbooks. This enhanced acquisition provides good generality for noisy and noiseless s
    
[^6]: 结构化离散表示的深度生成模型的无偏学习

    Unbiased Learning of Deep Generative Models with Structured Discrete Representations. (arXiv:2306.08230v1 [cs.LG])

    [http://arxiv.org/abs/2306.08230](http://arxiv.org/abs/2306.08230)

    该论文提出了一种名为结构化变分自编码器的深度生成模型，它通过图像模型的结构和可解释性以及深度学习的适用于高维数据的灵活似然，结合两种框架的优势。同时，该论文还提出了一种学习SVAE的新算法，与此同时，推导出了一种计算自然梯度的方法，这些优化创新使得SVAE首次能与最先进的时间序列模型进行比较。

    

    通过将图形模型与深度学习架构组合，我们学习具有两种框架优势的生成模型。 结构化变分自编码器（SVAE）从图形模型继承结构和可解释性，从深度学习中继承了适用于高维数据的灵活似然，但是会带来相当大的优化挑战。 我们提出了学习SVAE的新算法，并且首次证明了SVAE在含有缺失数据且包含离散潜变量时处理多模态不确定性的能力。我们的内存高效隐式微分方案使得SVAE可以通过梯度下降来学习，并且证明了鲁棒性。为了更快地学习准确的图形模型参数，我们推导了一种计算自然梯度的方法，而不需要手动进行导出，从而避免了先前工作中发现的偏差。这些优化创新使得首次能够将SVAE与最先进的时间序列模型进行比较。

    By composing graphical models with deep learning architectures, we learn generative models with the strengths of both frameworks. The structured variational autoencoder (SVAE) inherits structure and interpretability from graphical models, and flexible likelihoods for high-dimensional data from deep learning, but poses substantial optimization challenges. We propose novel algorithms for learning SVAEs, and are the first to demonstrate the SVAE's ability to handle multimodal uncertainty when data is missing by incorporating discrete latent variables. Our memory-efficient implicit differentiation scheme makes the SVAE tractable to learn via gradient descent, while demonstrating robustness to incomplete optimization. To more rapidly learn accurate graphical model parameters, we derive a method for computing natural gradients without manual derivations, which avoids biases found in prior work. These optimization innovations enable the first comparisons of the SVAE to state-of-the-art time s
    
[^7]: 干预概率分布的公理化

    Axiomatization of Interventional Probability Distributions. (arXiv:2305.04479v1 [math.ST])

    [http://arxiv.org/abs/2305.04479](http://arxiv.org/abs/2305.04479)

    本论文提供了一种简单和清晰的因果理论，它不需要使用任何建模假设，包括大多数具有潜在变量和因果循环的情况，并且不假定存在底层真正的因果图-事实上，它是因果图的副产品。

    

    因果干预是因果推断中的基本工具。在结构因果模型的情况下，它被公理化为do-演算规则。我们提供了一种简单的公理化方法，用于区分不同类型的干预分布的概率分布族。我们的公理化方法整洁地导致了一种简单和清晰的因果理论，具有几个优点：它不需要使用任何建模假设，例如结构性因果模型所强加的假设；它只依赖于单个变量的干预；它包括大多数具有潜在变量和因果循环的情况；更重要的是，它不假定存在底层真正的因果图--事实上，因果图是我们理论的副产品。我们展示了，在我们的公理化方法下，干预分布对于定义的干预因果图是马尔可夫的，并且观察到的联合概率分布对于获得的因果图是马尔可夫的；这些结果是一致的。

    Causal intervention is an essential tool in causal inference. It is axiomatized under the rules of do-calculus in the case of structure causal models. We provide simple axiomatizations for families of probability distributions to be different types of interventional distributions. Our axiomatizations neatly lead to a simple and clear theory of causality that has several advantages: it does not need to make use of any modeling assumptions such as those imposed by structural causal models; it only relies on interventions on single variables; it includes most cases with latent variables and causal cycles; and more importantly, it does not assume the existence of an underlying true causal graph--in fact, a causal graph is a by-product of our theory. We show that, under our axiomatizations, the intervened distributions are Markovian to the defined intervened causal graphs, and an observed joint probability distribution is Markovian to the obtained causal graph; these results are consistent 
    
[^8]: 广义划分局部深度

    Generalized partitioned local depth. (arXiv:2303.10167v1 [stat.ML])

    [http://arxiv.org/abs/2303.10167](http://arxiv.org/abs/2303.10167)

    本文提出了一个广义的凝聚概念，构建在分区局部深度的技术基础上，扩展了早期结果并应用于具有不确定性的数据的社区发现中。

    

    本文提供了一个最近由Berenhaut、Moore和Melvin [Proccedings of the National Academy of Sciences, 119 (4) (2022)]提出的凝聚概念的概括。所提出的表述基于分区局部深度的技术并提炼了两个关键概率概念：局部相关性和支持分割。早期结果在新的背景下得到扩展，并包括在具有不确定性的数据中揭示社区的应用示例。

    In this paper we provide a generalization of the concept of cohesion as introduced recently by Berenhaut, Moore and Melvin [Proceedings of the National Academy of Sciences, 119 (4) (2022)]. The formulation presented builds on the technique of partitioned local depth by distilling two key probabilistic concepts: local relevance and support division. Earlier results are extended within the new context, and examples of applications to revealing communities in data with uncertainty are included.
    
[^9]: 面向对抗生成模型的PAC-Bayesian泛化界

    PAC-Bayesian Generalization Bounds for Adversarial Generative Models. (arXiv:2302.08942v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.08942](http://arxiv.org/abs/2302.08942)

    将PAC-Bayesian理论扩展到生成模型，为基于Wasserstein距离和总变差距离的模型提供了泛化界，为Wasserstein GAN和Energy-Based GAN提供了新的训练目标，并在合成数据集上展示出非虚空泛化界。

    

    我们将PAC-Bayesian理论扩展到生成模型，并为基于Wasserstein距离和总变差距离的模型开发了泛化界。我们第一个关于Wasserstein距离的结果假设实例空间是有界的，而我们的第二个结果利用了降维的优势。我们的结果自然适用于Wasserstein GAN和Energy-Based GAN，而我们的界限为这两种GAN提供了新的训练目标。尽管我们的工作主要是理论性的，但我们进行了数值实验，展示了Wasserstein GAN在合成数据集上的非虚空泛化界。

    We extend PAC-Bayesian theory to generative models and develop generalization bounds for models based on the Wasserstein distance and the total variation distance. Our first result on the Wasserstein distance assumes the instance space is bounded, while our second result takes advantage of dimensionality reduction. Our results naturally apply to Wasserstein GANs and Energy-Based GANs, and our bounds provide new training objectives for these two. Although our work is mainly theoretical, we perform numerical experiments showing non-vacuous generalization bounds for Wasserstein GANs on synthetic datasets.
    
[^10]: 浅层视觉Transformer的理论理解：学习、泛化和样本复杂性的分析

    A Theoretical Understanding of Shallow Vision Transformers: Learning, Generalization, and Sample Complexity. (arXiv:2302.06015v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.06015](http://arxiv.org/abs/2302.06015)

    本文提供了第一份对于浅层ViT进行训练的理论分析，证明了使用SGD训练会产生稀疏的注意力图，目前的样本复杂度与标记相关令牌的分数倒数、标记级别的令牌噪声水平和初始模型错误呈正相关关系。

    

    近年来，具有自我注意机制的视觉Transformer（ViTs）在许多视觉任务中取得了巨大的实证成功。然而，由于层间的非凸交互，理论上的学习和泛化分析大多是难以理解的。本文提供了对于一项分类任务，使用一个自我注意层和两层感知机的浅层ViT进行训练的第一篇理论分析，建立了对于数据模型的描述，该模型可以同时表征标记相关和标记不相关的令牌。我们界定了达到零泛化误差的样本复杂性。我们的样本复杂性限制与标记相关令牌的部分倒数、标记级别的令牌噪声水平和初始模型误差呈正相关。我们还证明了使用随机梯度下降SGD（stochastic gradient descent）进行训练过程会导致稀疏的注意力图，这是对于注意力成功的一种形式证明。此外，本文指出，适当的令牌确定是确保实现最优性能的关键。

    Vision Transformers (ViTs) with self-attention modules have recently achieved great empirical success in many vision tasks. Due to non-convex interactions across layers, however, theoretical learning and generalization analysis is mostly elusive. Based on a data model characterizing both label-relevant and label-irrelevant tokens, this paper provides the first theoretical analysis of training a shallow ViT, i.e., one self-attention layer followed by a two-layer perceptron, for a classification task. We characterize the sample complexity to achieve a zero generalization error. Our sample complexity bound is positively correlated with the inverse of the fraction of label-relevant tokens, the token noise level, and the initial model error. We also prove that a training process using stochastic gradient descent (SGD) leads to a sparse attention map, which is a formal verification of the general intuition about the success of attention. Moreover, this paper indicates that a proper token spa
    

