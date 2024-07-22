# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Truly No-Regret Learning in Constrained MDPs](https://arxiv.org/abs/2402.15776) | 本文首次肯定回答了一个开放问题，即是否可以在不允许错误抵消的情况下，通过将一种常见的安全约束模型扩展到具有多个约束的CMDPs，提出了一种可以实现次线性后悔的新方法。 |
| [^2] | [Simple, unified analysis of Johnson-Lindenstrauss with applications](https://arxiv.org/abs/2402.10232) | 这项工作提出了Johnson-Lindenstrauss（JL）引理的简单统一分析，简化和统一了各种构造，包括球形、高斯、二进制硬币和次高斯模型，通过创新性地将Hanson-Wright不等式拓展到高维度，标志着对数据固有几何的保持取得重大进展。 |
| [^3] | [Global optimality under amenable symmetry constraints](https://arxiv.org/abs/2402.07613) | 该论文研究了在可接受的对称约束条件下的全局最优性问题，提出了一种满足对称性质的函数或度量，并通过引入轨道凸体和coycle等工具解决了这一问题。具体应用包括不变核均值嵌入和基于对称约束的运输方案最优性。这些结果与不变性检验的Hunt-Stein定理相关。 |
| [^4] | [Improved motif-scaffolding with SE(3) flow matching.](http://arxiv.org/abs/2401.04082) | 本文提出了一种使用SE(3)流匹配的图案支架方法，通过图案摊销和图案引导两种方法，可以生成结构上多样性更高的支架，与之前的最先进方法相比，成功率相当甚至更高。 |
| [^5] | [Efficient Neural Network Approaches for Conditional Optimal Transport with Applications in Bayesian Inference.](http://arxiv.org/abs/2310.16975) | 提出了两种神经网络方法来逼近静态和动态条件最优传输问题的解，实现了对条件概率分布的采样和密度估计，适用于贝叶斯推断。算法利用神经网络参数化传输映射以提高可扩展性。 |
| [^6] | [Discovering environments with XRM.](http://arxiv.org/abs/2309.16748) | 本文提出了一种用于发现环境的算法 XRM，它通过训练两个孪生网络，每个网络从训练数据的一半中学习，并模仿其兄弟网络的错误分类，解决了现有方法需要依赖人工注释环境信息的问题。 |
| [^7] | [Differentially Private Latent Diffusion Models.](http://arxiv.org/abs/2305.15759) | 本文提出使用差分隐私训练潜在扩散模型(LDMs)，通过预训练自编码器将高维像素空间转变为低维潜在空间实现更高效快速的DMs训练，并且通过只微调注意力模块减少了可训练参数的数量。 |
| [^8] | [DF2M: An Explainable Deep Bayesian Nonparametric Model for High-Dimensional Functional Time Series.](http://arxiv.org/abs/2305.14543) | 本文提出一种名为DF2M的模型，用于分析高维函数时间序列。该模型采用印度自助餐过程和深度核函数的多任务高斯过程捕捉时间动态，与传统的深度学习模型相比，DF2M提供了更好的可解释性和卓越的预测准确性。 |
| [^9] | [Introduction to Online Nonstochastic Control.](http://arxiv.org/abs/2211.09619) | 介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。 |

# 详细

[^1]: 受限制MDP中的真正无悔学习

    Truly No-Regret Learning in Constrained MDPs

    [https://arxiv.org/abs/2402.15776](https://arxiv.org/abs/2402.15776)

    本文首次肯定回答了一个开放问题，即是否可以在不允许错误抵消的情况下，通过将一种常见的安全约束模型扩展到具有多个约束的CMDPs，提出了一种可以实现次线性后悔的新方法。

    

    受约束的马尔可夫决策过程（CMDPs）是在强化学习中建模安全约束的常见方式。目前用于高效解决CMDPs的最先进方法基于原始-对偶算法。对于这些算法，所有当前已知的后悔界都允许错误抵消——可以通过在一个回合中的约束违反来用严格的约束满足在另一个回合中。这使得在线学习过程不安全，因为它仅保证最终（混合）策略的安全性，而在学习过程中不保证安全。正如Efroni等人（2020年）指出的，原始-对偶算法是否可以在不允许错误抵消的情况下可证明地实现次线性后悔是一个开放问题。在本文中，我们给出了第一个肯定的答案。我们首先将关于正则化原始-对偶方案的最后迭代收敛性通用化到具有多个约束的CMDPs上。基于这一见解，我们提出了一种基于模型的原始

    arXiv:2402.15776v1 Announce Type: new  Abstract: Constrained Markov decision processes (CMDPs) are a common way to model safety constraints in reinforcement learning. State-of-the-art methods for efficiently solving CMDPs are based on primal-dual algorithms. For these algorithms, all currently known regret bounds allow for error cancellations -- one can compensate for a constraint violation in one round with a strict constraint satisfaction in another. This makes the online learning process unsafe since it only guarantees safety for the final (mixture) policy but not during learning. As Efroni et al. (2020) pointed out, it is an open question whether primal-dual algorithms can provably achieve sublinear regret if we do not allow error cancellations. In this paper, we give the first affirmative answer. We first generalize a result on last-iterate convergence of regularized primal-dual schemes to CMDPs with multiple constraints. Building upon this insight, we propose a model-based primal
    
[^2]: Johnson-Lindenstrauss的简单统一分析及其应用

    Simple, unified analysis of Johnson-Lindenstrauss with applications

    [https://arxiv.org/abs/2402.10232](https://arxiv.org/abs/2402.10232)

    这项工作提出了Johnson-Lindenstrauss（JL）引理的简单统一分析，简化和统一了各种构造，包括球形、高斯、二进制硬币和次高斯模型，通过创新性地将Hanson-Wright不等式拓展到高维度，标志着对数据固有几何的保持取得重大进展。

    

    在这项工作中，我们提出了Johnson-Lindenstrauss（JL）引理的简单统一分析，这是处理高维数据至关重要的降维领域中的基石。我们的方法不仅简化了理解，还将各种构造统一到JL框架下，包括球形、高斯、二进制硬币和次高斯模型。这种简化和统一在保持数据固有几何的重要性方面取得了重大进展，对从流算法到强化学习等各种应用至关重要。值得注意的是，我们在这个简化框架内提出了球形构造有效性的第一个严格证明。我们贡献的核心是将Hanson-Wright不等式拓展到高维度，具有明确的常数，这标志着文献中质的飞跃。通过运用简单而强大的概率工具

    arXiv:2402.10232v1 Announce Type: new  Abstract: In this work, we present a simple and unified analysis of the Johnson-Lindenstrauss (JL) lemma, a cornerstone in the field of dimensionality reduction critical for managing high-dimensional data. Our approach not only simplifies the understanding but also unifies various constructions under the JL framework, including spherical, Gaussian, binary coin, and sub-Gaussian models. This simplification and unification make significant strides in preserving the intrinsic geometry of data, essential across diverse applications from streaming algorithms to reinforcement learning. Notably, we deliver the first rigorous proof of the spherical construction's effectiveness within this simplified framework. At the heart of our contribution is an innovative extension of the Hanson-Wright inequality to high dimensions, complete with explicit constants, marking a substantial leap in the literature. By employing simple yet powerful probabilistic tools and 
    
[^3]: 在可接受的对称约束条件下的全局最优性

    Global optimality under amenable symmetry constraints

    [https://arxiv.org/abs/2402.07613](https://arxiv.org/abs/2402.07613)

    该论文研究了在可接受的对称约束条件下的全局最优性问题，提出了一种满足对称性质的函数或度量，并通过引入轨道凸体和coycle等工具解决了这一问题。具体应用包括不变核均值嵌入和基于对称约束的运输方案最优性。这些结果与不变性检验的Hunt-Stein定理相关。

    

    我们研究是否存在一种满足可接受变换群指定的对称性质的函数或度量，即同时满足以下两个条件：（1）最小化给定的凸性泛函或风险，（2）满足可容忍对称约束。这种对称性质的例子包括不变性、可变性或准不变性。我们的结果依赖于Stein和Le Cam的老思想，以及在可接受群的遍历定理中出现的近似群平均值。在凸分析中，一类称为轨道凸体的凸集显得至关重要，我们在非参数设置中确定了这类轨道凸体的性质。我们还展示了一个称为coycle的简单装置如何将不同形式的对称性转化为一个问题。作为应用，我们得出了关于不变核均值嵌入和在对称约束下运输方案最优性的Monge-Kantorovich定理的结果。我们还解释了与不变性检验的Hunt-Stein定理的联系。

    We ask whether there exists a function or measure that (1) minimizes a given convex functional or risk and (2) satisfies a symmetry property specified by an amenable group of transformations. Examples of such symmetry properties are invariance, equivariance, or quasi-invariance. Our results draw on old ideas of Stein and Le Cam and on approximate group averages that appear in ergodic theorems for amenable groups. A class of convex sets known as orbitopes in convex analysis emerges as crucial, and we establish properties of such orbitopes in nonparametric settings. We also show how a simple device called a cocycle can be used to reduce different forms of symmetry to a single problem. As applications, we obtain results on invariant kernel mean embeddings and a Monge-Kantorovich theorem on optimality of transport plans under symmetry constraints. We also explain connections to the Hunt-Stein theorem on invariant tests.
    
[^4]: 使用SE(3)流匹配改进了图案支架技术

    Improved motif-scaffolding with SE(3) flow matching. (arXiv:2401.04082v1 [q-bio.QM])

    [http://arxiv.org/abs/2401.04082](http://arxiv.org/abs/2401.04082)

    本文提出了一种使用SE(3)流匹配的图案支架方法，通过图案摊销和图案引导两种方法，可以生成结构上多样性更高的支架，与之前的最先进方法相比，成功率相当甚至更高。

    

    蛋白质设计通常从一个图案的期望功能开始，图案支架旨在构建一个功能性蛋白质。最近，生成模型在设计各种图案的支架方面取得了突破性的成功。然而，生成的支架往往缺乏结构多样性，这可能会影响湿实验验证的成功。在这项工作中，我们将FrameFlow，一种用于蛋白质主链生成的SE(3)流匹配模型扩展到使用两种互补的方法进行图案支架。第一种方法是图案摊销，即使用数据增强策略，将FrameFlow训练为以图案为输入。第二种方法是图案引导，它使用FrameFlow的条件分数估计进行支架构建，并且不需要额外的训练。这两种方法的成功率与之前的最先进方法相当或更高，并且可以产生结构上多样性更高2.5倍的支架。

    Protein design often begins with knowledge of a desired function from a motif which motif-scaffolding aims to construct a functional protein around. Recently, generative models have achieved breakthrough success in designing scaffolds for a diverse range of motifs. However, the generated scaffolds tend to lack structural diversity, which can hinder success in wet-lab validation. In this work, we extend FrameFlow, an SE(3) flow matching model for protein backbone generation, to perform motif-scaffolding with two complementary approaches. The first is motif amortization, in which FrameFlow is trained with the motif as input using a data augmentation strategy. The second is motif guidance, which performs scaffolding using an estimate of the conditional score from FrameFlow, and requires no additional training. Both approaches achieve an equivalent or higher success rate than previous state-of-the-art methods, with 2.5 times more structurally diverse scaffolds. Code: https://github.com/ mi
    
[^5]: 条件最优传输的高效神经网络方法及贝叶斯推断中的应用

    Efficient Neural Network Approaches for Conditional Optimal Transport with Applications in Bayesian Inference. (arXiv:2310.16975v1 [stat.ML])

    [http://arxiv.org/abs/2310.16975](http://arxiv.org/abs/2310.16975)

    提出了两种神经网络方法来逼近静态和动态条件最优传输问题的解，实现了对条件概率分布的采样和密度估计，适用于贝叶斯推断。算法利用神经网络参数化传输映射以提高可扩展性。

    

    我们提出了两种神经网络方法，分别逼近静态和动态条件最优传输问题的解。这两种方法可以对条件概率分布进行采样和密度估计，这是贝叶斯推断中的核心任务。我们的方法将目标条件分布表示为可处理的参考分布的转换，因此属于测度传输的框架。在该框架中，COT映射是一个典型的选择，具有唯一性和单调性等可取的属性。然而，相关的COT问题在中等维度下计算具有挑战性。为了提高可扩展性，我们的数值算法利用神经网络对COT映射进行参数化。我们的方法充分利用了COT问题的静态和动态表达形式的结构。PCP-Map将条件传输映射建模为部分输入凸神经网络（PICNN）的梯度。

    We present two neural network approaches that approximate the solutions of static and dynamic conditional optimal transport (COT) problems, respectively. Both approaches enable sampling and density estimation of conditional probability distributions, which are core tasks in Bayesian inference. Our methods represent the target conditional distributions as transformations of a tractable reference distribution and, therefore, fall into the framework of measure transport. COT maps are a canonical choice within this framework, with desirable properties such as uniqueness and monotonicity. However, the associated COT problems are computationally challenging, even in moderate dimensions. To improve the scalability, our numerical algorithms leverage neural networks to parameterize COT maps. Our methods exploit the structure of the static and dynamic formulations of the COT problem. PCP-Map models conditional transport maps as the gradient of a partially input convex neural network (PICNN) and 
    
[^6]: 用XRM发现环境

    Discovering environments with XRM. (arXiv:2309.16748v1 [cs.LG])

    [http://arxiv.org/abs/2309.16748](http://arxiv.org/abs/2309.16748)

    本文提出了一种用于发现环境的算法 XRM，它通过训练两个孪生网络，每个网络从训练数据的一半中学习，并模仿其兄弟网络的错误分类，解决了现有方法需要依赖人工注释环境信息的问题。

    

    成功的跨领域泛化需要环境注释。然而，这些注释的获取是资源密集型的，并且它们对模型性能的影响受人类注释者的期望和感知偏差的限制。因此，为了实现应用领域全面泛化的鲁棒性AI系统，我们必须开发一种算法来自动发现引发广泛泛化的环境。目前的提案根据训练误差将示例划分为不同的类，但存在一个根本问题。这些方法添加了超参数和早停策略，而这些参数是无法在没有人类注释环境的验证集的情况下进行调整的，而这些信息正是要发现的信息。在本文中，我们提出了 Cross-Risk-Minimization (XRM) 来解决这个问题。XRM 训练两个孪生网络，每个网络从训练数据的一个随机一半中学习，同时模仿其兄弟网络所做的自信的错误分类。XRM 提供了超参数调整的方法，并且不需要依赖人工注释的环境信息。

    Successful out-of-distribution generalization requires environment annotations. Unfortunately, these are resource-intensive to obtain, and their relevance to model performance is limited by the expectations and perceptual biases of human annotators. Therefore, to enable robust AI systems across applications, we must develop algorithms to automatically discover environments inducing broad generalization. Current proposals, which divide examples based on their training error, suffer from one fundamental problem. These methods add hyper-parameters and early-stopping criteria that are impossible to tune without a validation set with human-annotated environments, the very information subject to discovery. In this paper, we propose Cross-Risk-Minimization (XRM) to address this issue. XRM trains two twin networks, each learning from one random half of the training data, while imitating confident held-out mistakes made by its sibling. XRM provides a recipe for hyper-parameter tuning, does not 
    
[^7]: 差分隐私潜在扩散模型

    Differentially Private Latent Diffusion Models. (arXiv:2305.15759v1 [stat.ML])

    [http://arxiv.org/abs/2305.15759](http://arxiv.org/abs/2305.15759)

    本文提出使用差分隐私训练潜在扩散模型(LDMs)，通过预训练自编码器将高维像素空间转变为低维潜在空间实现更高效快速的DMs训练，并且通过只微调注意力模块减少了可训练参数的数量。

    

    扩散模型(DMs)被广泛用于生成高质量图像数据集。然而，由于它们直接在高维像素空间中运行，DMs的优化计算成本高，需要长时间的训练。这导致由于差分隐私的可组合性属性，大量噪音注入到差分隐私学习过程中。为了解决这个挑战，我们提出使用差分隐私训练潜在扩散模型(LDMs)。LDMs使用强大的预训练自编码器将高维像素空间减少到更低维的潜在空间，使训练DMs更加高效和快速。与[Ghalebikesabi等人，2023]预先用公共数据预训练DMs，然后再用隐私数据进行微调不同，我们仅微调LDMs中不同层的注意力模块以获得隐私敏感数据，相对于整个DM微调，可减少大约96%的可训练参数数量。

    Diffusion models (DMs) are widely used for generating high-quality image datasets. However, since they operate directly in the high-dimensional pixel space, optimization of DMs is computationally expensive, requiring long training times. This contributes to large amounts of noise being injected into the differentially private learning process, due to the composability property of differential privacy. To address this challenge, we propose training Latent Diffusion Models (LDMs) with differential privacy. LDMs use powerful pre-trained autoencoders to reduce the high-dimensional pixel space to a much lower-dimensional latent space, making training DMs more efficient and fast. Unlike [Ghalebikesabi et al., 2023] that pre-trains DMs with public data then fine-tunes them with private data, we fine-tune only the attention modules of LDMs at varying layers with privacy-sensitive data, reducing the number of trainable parameters by approximately 96% compared to fine-tuning the entire DM. We te
    
[^8]: DF2M：一种可解释的用于高维函数时间序列分析的深度贝叶斯非参数模型

    DF2M: An Explainable Deep Bayesian Nonparametric Model for High-Dimensional Functional Time Series. (arXiv:2305.14543v1 [stat.ML])

    [http://arxiv.org/abs/2305.14543](http://arxiv.org/abs/2305.14543)

    本文提出一种名为DF2M的模型，用于分析高维函数时间序列。该模型采用印度自助餐过程和深度核函数的多任务高斯过程捕捉时间动态，与传统的深度学习模型相比，DF2M提供了更好的可解释性和卓越的预测准确性。

    

    本文提出Deep Functional Factor Model(DF2M)，一种用于分析高维函数时间序列的贝叶斯非参数模型。DF2M利用印度自助餐过程和深度核函数的多任务高斯过程来捕捉非马尔科夫和非线性时间动态。与许多黑匣子深度学习模型不同，DF2M通过构建因子模型并将深度神经网络融入核函数中，提供了一种可解释的使用神经网络的方法。此外，我们还开发了一种计算高效的变分推理算法来推断DF2M。四个真实数据集的实证结果表明，与传统的深度学习模型相比，DF2M提供了更好的可解释性和卓越的预测准确性。

    In this paper, we present Deep Functional Factor Model (DF2M), a Bayesian nonparametric model for analyzing high-dimensional functional time series. The DF2M makes use of the Indian Buffet Process and the multi-task Gaussian Process with a deep kernel function to capture non-Markovian and nonlinear temporal dynamics. Unlike many black-box deep learning models, the DF2M provides an explainable way to use neural networks by constructing a factor model and incorporating deep neural networks within the kernel function. Additionally, we develop a computationally efficient variational inference algorithm for inferring the DF2M. Empirical results from four real-world datasets demonstrate that the DF2M offers better explainability and superior predictive accuracy compared to conventional deep learning models for high-dimensional functional time series.
    
[^9]: 在线非随机控制简介

    Introduction to Online Nonstochastic Control. (arXiv:2211.09619v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.09619](http://arxiv.org/abs/2211.09619)

    介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    

    本文介绍了一种新兴的动态系统控制与可微强化学习范式——在线非随机控制，并应用在线凸优化和凸松弛技术得到了具有可证明保证的新方法，在最佳和鲁棒控制方面取得了显著成果。与其他框架不同，该方法的目标是对抗性攻击，在无法预测扰动模型的情况下，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    This text presents an introduction to an emerging paradigm in control of dynamical systems and differentiable reinforcement learning called online nonstochastic control. The new approach applies techniques from online convex optimization and convex relaxations to obtain new methods with provable guarantees for classical settings in optimal and robust control.  The primary distinction between online nonstochastic control and other frameworks is the objective. In optimal control, robust control, and other control methodologies that assume stochastic noise, the goal is to perform comparably to an offline optimal strategy. In online nonstochastic control, both the cost functions as well as the perturbations from the assumed dynamical model are chosen by an adversary. Thus the optimal policy is not defined a priori. Rather, the target is to attain low regret against the best policy in hindsight from a benchmark class of policies.  This objective suggests the use of the decision making frame
    

