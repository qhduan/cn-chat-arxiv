# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Discrete Latent Graph Generative Modeling with Diffusion Bridges](https://arxiv.org/abs/2403.16883) | GLAD是一个在离散潜在空间上操作的图生成模型，通过适应扩散桥结构学习其离散潜在空间的先验，避免了依赖于原始数据空间的分解，在图生成任务中表现出优越性。 |
| [^2] | [Computational-Statistical Gaps for Improper Learning in Sparse Linear Regression](https://arxiv.org/abs/2402.14103) | 该研究探讨了稀疏线性回归中的计算统计差距问题，为了高效地找到可以在样本上实现非平凡预测误差的潜在密集估计的回归向量，需要至少 $\Omega(k \log (d/k))$ 个样本。 |
| [^3] | [Multi-class Temporal Logic Neural Networks](https://arxiv.org/abs/2402.12397) | 提出了一种结合神经网络和信号时间逻辑的方法，用于多类别时间序列数据的分类，关键贡献包括引入边界概念和利用STL属性增强结果的可解释性。 |
| [^4] | [Accelerating Look-ahead in Bayesian Optimization: Multilevel Monte Carlo is All you Need](https://arxiv.org/abs/2402.02111) | 本文利用多层蒙特卡洛方法加速贝叶斯优化中的前瞻过程，并证明在涉及嵌套期望和最大化的问题中具有优势。 |
| [^5] | [On the numerical reliability of nonsmooth autodiff: a MaxPool case study.](http://arxiv.org/abs/2401.02736) | 本文研究了涉及非平滑MaxPool操作的神经网络自动微分的数值可靠性，并发现最近的研究表明AD几乎在每个地方都与导数相符，即使在存在非平滑操作的情况下也是如此。但在实践中，AD使用的是浮点数，需要探索可能导致AD数值不正确的情况。通过研究不同选择的非平滑MaxPool雅可比矩阵对训练过程的影响，我们找到了分歧区和补偿区两个可能导致AD数值不正确的子集。 |
| [^6] | [Analysis of learning a flow-based generative model from limited sample complexity.](http://arxiv.org/abs/2310.03575) | 我们分析了从有限样本复杂度中训练基于流的生成模型的问题，并提供了尖锐的端到端分析。我们找到了学习到的速度场的紧凑特性，并描述了生成流的近似，该近似将基本高斯密度推向目标密度。我们还提供了生成混合物均值与目标混合物均值之间距离的闭式公式，并证明其衰减速度为$\Theta_n(\frac{1}{n})$，这实际上是贝叶斯最优的。 |
| [^7] | [Regularization and Optimal Multiclass Learning.](http://arxiv.org/abs/2309.13692) | 本研究旨在研究正则化在多类别学习中的作用，以及其在一些特定情景下的最优学习算法。我们使用一对一包含图(OIGs)展示了结构风险最小化、最大熵原则和贝叶斯推理等算法原则的最优学习算法。 |
| [^8] | [Information Geometry of Wasserstein Statistics on Shapes and Affine Deformations.](http://arxiv.org/abs/2307.12508) | 在这篇论文中，我们研究了Wasserstein统计在仿射变形统计模型中的信息几何特征，比较了信息几何和Wasserstein几何的估计器的优缺点，并发现Wasserstein估计量在椭圆对称仿射变形模型中是矩估计量，在波形为高斯分布时与信息几何估计量重合。 |
| [^9] | [Latent Optimal Paths by Gumbel Propagation for Variational Bayesian Dynamic Programming.](http://arxiv.org/abs/2306.02568) | 该论文使用动态规划和Gumbel传播在VAE的潜在空间中获得结构化稀疏最优路径，从而使得模型可以依赖于未观察到的结构特征信息，并成功实现了文本转语音和歌声合成。 |
| [^10] | [Combining Distance to Class Centroids and Outlier Discounting for Improved Learning with Noisy Labels.](http://arxiv.org/abs/2303.09470) | 本文提出了结合类中心距离和异常值折扣的方法，用于解决在存在噪声标签的情况下训练机器学习模型的问题，并通过实验证明了其有效性 。 |
| [^11] | [Controlling Moments with Kernel Stein Discrepancies.](http://arxiv.org/abs/2211.05408) | 本研究分析了核斯坦离差（KSD）控制性质，发现标准KSD无法控制矩的收敛，提出了可控制矩和弱收敛的下游扩散KSD，并且发展了可以准确描述$q$-Wasserstein收敛的KSD。 |

# 详细

[^1]: 带扩散桥的离散潜在图生成建模

    Discrete Latent Graph Generative Modeling with Diffusion Bridges

    [https://arxiv.org/abs/2403.16883](https://arxiv.org/abs/2403.16883)

    GLAD是一个在离散潜在空间上操作的图生成模型，通过适应扩散桥结构学习其离散潜在空间的先验，避免了依赖于原始数据空间的分解，在图生成任务中表现出优越性。

    

    学习潜在空间中的图生成模型相比于在原始数据空间上操作的模型受到较少关注，迄今表现出的性能乏善可陈。我们提出了GLAD，一个潜在空间图生成模型。与大多数先前的潜在空间图生成模型不同，GLAD在保留图结构的离散性质方面运行，无需进行诸如潜在空间连续性等不自然的假设。我们通过将扩散桥调整到其结构，来学习我们离散潜在空间的先验。通过在适当构建的潜在空间上操作，我们避免依赖于常用于在原始数据空间操作的模型中的分解。我们在一系列图基准数据集上进行实验，明显展示了离散潜在空间的优越性，并取得了最先进的图生成性能，使GLA

    arXiv:2403.16883v1 Announce Type: new  Abstract: Learning graph generative models over latent spaces has received less attention compared to models that operate on the original data space and has so far demonstrated lacklustre performance. We present GLAD a latent space graph generative model. Unlike most previous latent space graph generative models, GLAD operates on a discrete latent space that preserves to a significant extent the discrete nature of the graph structures making no unnatural assumptions such as latent space continuity. We learn the prior of our discrete latent space by adapting diffusion bridges to its structure. By operating over an appropriately constructed latent space we avoid relying on decompositions that are often used in models that operate in the original data space. We present experiments on a series of graph benchmark datasets which clearly show the superiority of the discrete latent space and obtain state of the art graph generative performance, making GLA
    
[^2]: 稀疏线性回归中不当学习的计算统计差距

    Computational-Statistical Gaps for Improper Learning in Sparse Linear Regression

    [https://arxiv.org/abs/2402.14103](https://arxiv.org/abs/2402.14103)

    该研究探讨了稀疏线性回归中的计算统计差距问题，为了高效地找到可以在样本上实现非平凡预测误差的潜在密集估计的回归向量，需要至少 $\Omega(k \log (d/k))$ 个样本。

    

    我们研究了稀疏线性回归中不当学习的计算统计差距。具体来说，给定来自维度为 $d$ 的 $k$-稀疏线性模型的 $n$ 个样本，我们询问了在时间多项式中的最小样本复杂度，以便高效地找到一个对这 $n$ 个样本达到非平凡预测误差的潜在密集估计的回归向量。信息理论上，这可以用 $\Theta(k \log (d/k))$ 个样本实现。然而，尽管在文献中很显著，但没有已知的多项式时间算法可以在不附加对模型的其他限制的情况下使用少于 $\Theta(d)$ 个样本达到相同的保证。类似地，现有的困难结果要么仅限于适当设置，在该设置中估计值也必须是稀疏的，要么仅适用于特定算法。

    arXiv:2402.14103v1 Announce Type: new  Abstract: We study computational-statistical gaps for improper learning in sparse linear regression. More specifically, given $n$ samples from a $k$-sparse linear model in dimension $d$, we ask what is the minimum sample complexity to efficiently (in time polynomial in $d$, $k$, and $n$) find a potentially dense estimate for the regression vector that achieves non-trivial prediction error on the $n$ samples. Information-theoretically this can be achieved using $\Theta(k \log (d/k))$ samples. Yet, despite its prominence in the literature, there is no polynomial-time algorithm known to achieve the same guarantees using less than $\Theta(d)$ samples without additional restrictions on the model. Similarly, existing hardness results are either restricted to the proper setting, in which the estimate must be sparse as well, or only apply to specific algorithms.   We give evidence that efficient algorithms for this task require at least (roughly) $\Omega(
    
[^3]: 多类别时间逻辑神经网络

    Multi-class Temporal Logic Neural Networks

    [https://arxiv.org/abs/2402.12397](https://arxiv.org/abs/2402.12397)

    提出了一种结合神经网络和信号时间逻辑的方法，用于多类别时间序列数据的分类，关键贡献包括引入边界概念和利用STL属性增强结果的可解释性。

    

    时间序列数据可以代表无人系统（如无人机和自动驾驶汽车）的行为。在这一领域，二元和多类别分类问题受到了广泛关注。神经网络是一种流行的分类数据的方法；然而，它们缺乏可解释性，这在从中提取有意义的信息方面构成了重要挑战。信号时间逻辑（STL）是一种描述定时行为属性的形式化语言。我们提出了一种将所有这些元素结合在一起的方法：使用表示STL规范的神经网络进行时间序列数据的多类别分类。我们提供了两个关键贡献：1）我们引入了多类别分类的边界概念，2）我们引入了基于STL的属性来增强结果的可解释性。我们在两个数据集上评估了我们的方法，并与最先进的基准进行了比较。

    arXiv:2402.12397v1 Announce Type: cross  Abstract: Time-series data can represent the behaviors of autonomous systems, such as drones and self-driving cars. The problem of binary and multi-class classification has received a lot of attention in this field. Neural networks represent a popular approach to classifying data; However, they lack interpretability, which poses a significant challenge in extracting meaningful information from them. Signal Temporal Logic (STL) is a formalism to describe the properties of timed behaviors. We propose a method that combines all of the above: neural networks that represent STL specifications for multi-class classification of time-series data. We offer two key contributions: 1) We introduce a notion of margin for multi-class classification, and 2) we introduce the use of STL-based attributes for enhancing the interpretability of the results. We evaluate our method on two datasets and compare with state-of-the-art baselines.
    
[^4]: 加速贝叶斯优化中的前瞻：多层蒙特卡洛就够了

    Accelerating Look-ahead in Bayesian Optimization: Multilevel Monte Carlo is All you Need

    [https://arxiv.org/abs/2402.02111](https://arxiv.org/abs/2402.02111)

    本文利用多层蒙特卡洛方法加速贝叶斯优化中的前瞻过程，并证明在涉及嵌套期望和最大化的问题中具有优势。

    

    我们利用多层蒙特卡洛(MLMC)来提高涉及嵌套期望和最大化的多步前瞻贝叶斯优化(BO)方法的性能。普通蒙特卡洛的复杂度在嵌套操作中会降低，而MLMC能够以规范蒙特卡洛收敛速度解决这类问题，而且不依赖于维度和平滑性假设。我们的理论研究主要关注一步和两步前瞻采集函数的近似改进，但正如我们所讨论的，这种方法在多种方面是可推广的，包括超越BO的背景。我们通过数值验证了我们的发现，并在几个基准示例中展示了MLMC在BO中的优势。代码在这里获取：https://github.com/Shangda-Yang/MLMCBO。

    We leverage multilevel Monte Carlo (MLMC) to improve the performance of multi-step look-ahead Bayesian optimization (BO) methods that involve nested expectations and maximizations. The complexity rate of naive Monte Carlo degrades for nested operations, whereas MLMC is capable of achieving the canonical Monte Carlo convergence rate for this type of problem, independently of dimension and without any smoothness assumptions. Our theoretical study focuses on the approximation improvements for one- and two-step look-ahead acquisition functions, but, as we discuss, the approach is generalizable in various ways, including beyond the context of BO. Findings are verified numerically and the benefits of MLMC for BO are illustrated on several benchmark examples. Code is available here https://github.com/Shangda-Yang/MLMCBO.
    
[^5]: 关于非平滑自动微分的数值可靠性：MaxPool案例研究

    On the numerical reliability of nonsmooth autodiff: a MaxPool case study. (arXiv:2401.02736v1 [cs.LG])

    [http://arxiv.org/abs/2401.02736](http://arxiv.org/abs/2401.02736)

    本文研究了涉及非平滑MaxPool操作的神经网络自动微分的数值可靠性，并发现最近的研究表明AD几乎在每个地方都与导数相符，即使在存在非平滑操作的情况下也是如此。但在实践中，AD使用的是浮点数，需要探索可能导致AD数值不正确的情况。通过研究不同选择的非平滑MaxPool雅可比矩阵对训练过程的影响，我们找到了分歧区和补偿区两个可能导致AD数值不正确的子集。

    

    本文考虑了涉及非平滑MaxPool操作的神经网络自动微分（AD）的可靠性问题。我们研究了在不同精度级别（16位、32位、64位）和卷积架构（LeNet、VGG和ResNet）以及不同数据集（MNIST、CIFAR10、SVHN和ImageNet）上的AD行为。尽管AD可能是错误的，但最近的研究表明，它在几乎每个地方都与导数相符，即使在存在非平滑操作（如MaxPool和ReLU）的情况下也是如此。另一方面，在实践中，AD使用的是浮点数（而不是实数），因此需要探索AD可能在数值上不正确的子集。这些子集包括分歧区（AD在实数上不正确）和补偿区（AD在浮点数上不正确但在实数上正确）。我们使用SGD进行训练过程，并研究了MaxPool非平滑雅可比矩阵的不同选择对训练过程的影响。

    This paper considers the reliability of automatic differentiation (AD) for neural networks involving the nonsmooth MaxPool operation. We investigate the behavior of AD across different precision levels (16, 32, 64 bits) and convolutional architectures (LeNet, VGG, and ResNet) on various datasets (MNIST, CIFAR10, SVHN, and ImageNet). Although AD can be incorrect, recent research has shown that it coincides with the derivative almost everywhere, even in the presence of nonsmooth operations (such as MaxPool and ReLU). On the other hand, in practice, AD operates with floating-point numbers (not real numbers), and there is, therefore, a need to explore subsets on which AD can be numerically incorrect. These subsets include a bifurcation zone (where AD is incorrect over reals) and a compensation zone (where AD is incorrect over floating-point numbers but correct over reals). Using SGD for the training process, we study the impact of different choices of the nonsmooth Jacobian for the MaxPool
    
[^6]: 从有限的样本复杂度中学习基于流的生成模型的分析

    Analysis of learning a flow-based generative model from limited sample complexity. (arXiv:2310.03575v1 [stat.ML])

    [http://arxiv.org/abs/2310.03575](http://arxiv.org/abs/2310.03575)

    我们分析了从有限样本复杂度中训练基于流的生成模型的问题，并提供了尖锐的端到端分析。我们找到了学习到的速度场的紧凑特性，并描述了生成流的近似，该近似将基本高斯密度推向目标密度。我们还提供了生成混合物均值与目标混合物均值之间距离的闭式公式，并证明其衰减速度为$\Theta_n(\frac{1}{n})$，这实际上是贝叶斯最优的。

    

    我们研究训练一个由两层自编码器参数化的流式生成模型，以从高维高斯混合模型中抽样的问题。我们对这个问题进行了尖锐的端到端分析。首先，我们提供了一个紧密的闭式特征化学习到的速度场，当参数化为一个在目标分布上从有限数量的样本$ n $中进行训练的浅层去噪自编码器时。在此分析的基础上，我们提供了对应的生成流的尖锐描述，将基本高斯密度推向目标密度的近似。特别地，我们提供了生成混合物的均值与目标混合物均值之间的距离的闭式公式，我们证明这个距离会衰减为$\Theta_n(\frac{1}{n})$。最后，这个速率被证明实际上是贝叶斯最优的。

    We study the problem of training a flow-based generative model, parametrized by a two-layer autoencoder, to sample from a high-dimensional Gaussian mixture. We provide a sharp end-to-end analysis of the problem. First, we provide a tight closed-form characterization of the learnt velocity field, when parametrized by a shallow denoising auto-encoder trained on a finite number $n$ of samples from the target distribution. Building on this analysis, we provide a sharp description of the corresponding generative flow, which pushes the base Gaussian density forward to an approximation of the target density. In particular, we provide closed-form formulae for the distance between the mean of the generated mixture and the mean of the target mixture, which we show decays as $\Theta_n(\frac{1}{n})$. Finally, this rate is shown to be in fact Bayes-optimal.
    
[^7]: 正则化与最优多类别学习

    Regularization and Optimal Multiclass Learning. (arXiv:2309.13692v1 [cs.LG])

    [http://arxiv.org/abs/2309.13692](http://arxiv.org/abs/2309.13692)

    本研究旨在研究正则化在多类别学习中的作用，以及其在一些特定情景下的最优学习算法。我们使用一对一包含图(OIGs)展示了结构风险最小化、最大熵原则和贝叶斯推理等算法原则的最优学习算法。

    

    以经验风险最小化（ERM）为代表的典型学习算法已被发现在一些学习非均匀收敛的情景中无法成功应用。因此，在机器学习实践中存在许多更丰富的算法技术来控制模型容量。然而，在这些更一般的情境中，没有一种技术或原则能够脱颖而出来描述最优学习的特征。本文旨在表征正则化在多类别学习中的作用，这可能是ERM失败的最简单情景，而标签集是任意的。我们利用一对一包含图（OIGs）展示了与传统算法原则相结合的最优学习算法：奥卡姆剃刀原则所体现的结构风险最小化（SRM），最大熵原则和贝叶斯推理。值得注意的是，我们引入了一种在结构风险最小化上进行放松的最优学习器。

    The quintessential learning algorithm of empirical risk minimization (ERM) is known to fail in various settings for which uniform convergence does not characterize learning. It is therefore unsurprising that the practice of machine learning is rife with considerably richer algorithmic techniques for successfully controlling model capacity. Nevertheless, no such technique or principle has broken away from the pack to characterize optimal learning in these more general settings.  The purpose of this work is to characterize the role of regularization in perhaps the simplest setting for which ERM fails: multiclass learning with arbitrary label sets. Using one-inclusion graphs (OIGs), we exhibit optimal learning algorithms that dovetail with tried-and-true algorithmic principles: Occam's Razor as embodied by structural risk minimization (SRM), the principle of maximum entropy, and Bayesian reasoning. Most notably, we introduce an optimal learner which relaxes structural risk minimization on
    
[^8]: 形状和仿射变形的Wasserstein统计的信息几何

    Information Geometry of Wasserstein Statistics on Shapes and Affine Deformations. (arXiv:2307.12508v1 [math.ST])

    [http://arxiv.org/abs/2307.12508](http://arxiv.org/abs/2307.12508)

    在这篇论文中，我们研究了Wasserstein统计在仿射变形统计模型中的信息几何特征，比较了信息几何和Wasserstein几何的估计器的优缺点，并发现Wasserstein估计量在椭圆对称仿射变形模型中是矩估计量，在波形为高斯分布时与信息几何估计量重合。

    

    信息几何和Wasserstein几何是介绍概率分布流形中的两个主要结构，它们捕捉了不同的特征。我们在仿射变形统计模型的Li和Zhao（2023）框架中研究了Wasserstein几何的特征，它是位置-尺度模型的多维泛化。我们比较了基于信息几何和Wasserstein几何的估计器的优点和缺点。在Wasserstein几何中，概率分布的形状和仿射变形是分离的，表明在对波形扰动具有鲁棒性的同时，会损失Fisher效率。我们证明了在椭圆对称仿射变形模型的情况下Wasserstein估计量是矩估计量。它与信息几何估计量（最大似然估计量）仅在波形为高斯分布时重合。Wasserstein效率的作用是...

    Information geometry and Wasserstein geometry are two main structures introduced in a manifold of probability distributions, and they capture its different characteristics. We study characteristics of Wasserstein geometry in the framework of Li and Zhao (2023) for the affine deformation statistical model, which is a multi-dimensional generalization of the location-scale model. We compare merits and demerits of estimators based on information geometry and Wasserstein geometry. The shape of a probability distribution and its affine deformation are separated in the Wasserstein geometry, showing its robustness against the waveform perturbation in exchange for the loss in Fisher efficiency. We show that the Wasserstein estimator is the moment estimator in the case of the elliptically symmetric affine deformation model. It coincides with the information-geometrical estimator (maximum-likelihood estimator) when and only when the waveform is Gaussian. The role of the Wasserstein efficiency is 
    
[^9]: Gumbel传播下的潜在最优路径变分贝叶斯动态规划

    Latent Optimal Paths by Gumbel Propagation for Variational Bayesian Dynamic Programming. (arXiv:2306.02568v1 [stat.ML])

    [http://arxiv.org/abs/2306.02568](http://arxiv.org/abs/2306.02568)

    该论文使用动态规划和Gumbel传播在VAE的潜在空间中获得结构化稀疏最优路径，从而使得模型可以依赖于未观察到的结构特征信息，并成功实现了文本转语音和歌声合成。

    

    我们提出了一种统一方法，使用动态规划和Gumbel传播在变分自编码器（VAE）的潜在空间中获取结构化稀疏最优路径。我们通过概率软化解，即随机最优路径，来解决经典最优路径问题，并将广泛的DP问题转化为有向无环图，其中所有可能的路径遵循Gibbs分布。我们通过Gumbel分布的属性显示Gibbs分布与消息传递算法的等价性，并提供了变分贝叶斯推理所需的所有要素。我们的方法获取了潜在最优路径，使生成任务的端到端训练成为可能，其中模型依赖于未观察到的结构特征的信息。我们验证了我们方法的行为，并展示了其在两个真实世界应用中的适用性：文本转语音和歌声合成。

    We propose a unified approach to obtain structured sparse optimal paths in the latent space of a variational autoencoder (VAE) using dynamic programming and Gumbel propagation. We solve the classical optimal path problem by a probability softening solution, called the stochastic optimal path, and transform a wide range of DP problems into directed acyclic graphs in which all possible paths follow a Gibbs distribution. We show the equivalence of the Gibbs distribution to a message-passing algorithm by the properties of the Gumbel distribution and give all the ingredients required for variational Bayesian inference. Our approach obtaining latent optimal paths enables end-to-end training for generative tasks in which models rely on the information of unobserved structural features. We validate the behavior of our approach and showcase its applicability in two real-world applications: text-to-speech and singing voice synthesis.
    
[^10]: 结合类中心距离和异常值折扣的方法，提高在存在噪声标签的情况下训练机器学习模型的效果

    Combining Distance to Class Centroids and Outlier Discounting for Improved Learning with Noisy Labels. (arXiv:2303.09470v1 [cs.LG])

    [http://arxiv.org/abs/2303.09470](http://arxiv.org/abs/2303.09470)

    本文提出了结合类中心距离和异常值折扣的方法，用于解决在存在噪声标签的情况下训练机器学习模型的问题，并通过实验证明了其有效性 。

    

    本文提出了一种新的方法，用于解决在存在噪声标签的情况下训练机器学习模型的挑战。通过在物品的潜在空间中巧妙地使用距离类中心的方法，再结合折扣策略以减少距离所有类中心（即异常值）远的样本的重要性，我们的方法有效解决了噪声标签的问题。我们的方法是基于这样的想法：在训练的早期阶段，距离各自类中心更远的样本更可能是噪声。通过在几个流行的基准数据集上进行广泛实验，我们证明了我们的方法的有效性。结果表明，我们的方法在存在噪声标签的情况下，可以明显提高分类准确性，表现优于当前领域的最优方法。

    In this paper, we propose a new approach for addressing the challenge of training machine learning models in the presence of noisy labels. By combining a clever usage of distance to class centroids in the items' latent space with a discounting strategy to reduce the importance of samples far away from all the class centroids (i.e., outliers), our method effectively addresses the issue of noisy labels. Our approach is based on the idea that samples farther away from their respective class centroid in the early stages of training are more likely to be noisy. We demonstrate the effectiveness of our method through extensive experiments on several popular benchmark datasets. Our results show that our approach outperforms the state-of-the-art in this area, achieving significant improvements in classification accuracy when the dataset contains noisy labels.
    
[^11]: 用核斯坦离差控制矩

    Controlling Moments with Kernel Stein Discrepancies. (arXiv:2211.05408v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.05408](http://arxiv.org/abs/2211.05408)

    本研究分析了核斯坦离差（KSD）控制性质，发现标准KSD无法控制矩的收敛，提出了可控制矩和弱收敛的下游扩散KSD，并且发展了可以准确描述$q$-Wasserstein收敛的KSD。

    

    核斯坦离差（KSD）用于衡量分布逼近的质量，并且可以在目标密度具有不可计算的归一化常数时计算。显著的应用包括诊断近似MCMC采样器和非归一化统计模型的适配度检验。本文分析了KSD的收敛控制性质。我们首先证明了用于弱收敛控制的标准KSD无法控制矩的收敛。为了解决这个限制，我们提供了一组充分条件，下游扩散KSD可以同时控制矩和弱收敛。作为一个直接的结果，我们发展了对于每个$q>0$，第一组已知可以准确描述$q$-Wasserstein收敛的KSD。

    Kernel Stein discrepancies (KSDs) measure the quality of a distributional approximation and can be computed even when the target density has an intractable normalizing constant. Notable applications include the diagnosis of approximate MCMC samplers and goodness-of-fit tests for unnormalized statistical models. The present work analyzes the convergence control properties of KSDs. We first show that standard KSDs used for weak convergence control fail to control moment convergence. To address this limitation, we next provide sufficient conditions under which alternative diffusion KSDs control both moment and weak convergence. As an immediate consequence we develop, for each $q > 0$, the first KSDs known to exactly characterize $q$-Wasserstein convergence.
    

