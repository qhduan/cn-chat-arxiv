# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sequential transport maps using SoS density estimation and $\alpha$-divergences](https://arxiv.org/abs/2402.17943) | 本研究探讨了使用SoS密度和α-离散度来近似中间密度的顺序传输映射框架，通过将两者结合，可以有效地解决凸优化问题，进而实现从未标准化的密度生成样本。 |
| [^2] | [Non-asymptotic Convergence of Discrete-time Diffusion Models: New Approach and Improved Rate](https://arxiv.org/abs/2402.13901) | 本文提出了离散时间扩散模型的新方法，改进了对更大类的分布的收敛保证，并提高了具有有界支撑的分布的收敛速率。 |
| [^3] | [Hypergraph Node Classification With Graph Neural Networks](https://arxiv.org/abs/2402.05569) | 本研究提出了一种简单高效的框架，利用加权子图扩展的图神经网络(WCE-GNN)实现了超图节点分类。实验证明，WCE-GNN具有优秀的预测效果和较低的计算复杂度。 |
| [^4] | [$\alpha$-Divergence Loss Function for Neural Density Ratio Estimation](https://arxiv.org/abs/2402.02041) | 本文提出了一种应用于神经密度比估计的$\alpha$-散度损失函数($\alpha$-Div)，通过简洁实现和稳定优化解决了现有方法中存在的优化问题。实验证明了这种损失函数的稳定性，并提出了对DRE任务的估计准确性的研究，同时给出了样本要求的解决方案。 |
| [^5] | [Time-Uniform Confidence Spheres for Means of Random Vectors](https://arxiv.org/abs/2311.08168) | 该研究提出了时间均匀置信球序列，可以同时高概率地包含各种样本量下随机向量的均值，并针对不同分布假设进行了扩展和统一分析。 |
| [^6] | [Efficient error and variance estimation for randomized matrix computations](https://arxiv.org/abs/2207.06342) | 该论文提出了用于随机矩阵计算的高效误差和方差估计方法，可帮助评估输出质量并指导算法参数选择。 |
| [^7] | [Fitting an ellipsoid to a quadratic number of random points.](http://arxiv.org/abs/2307.01181) | 将$n$个高斯随机向量拟合到以原点为中心的椭球体边界的问题$(\mathrm{P})$，我们提出了一个基于随机向量Gram矩阵集中性的改进方法，证明了当$n \leq d^2 / C$时，问题$(\mathrm{P})$具有很高的可行性概率。 |
| [^8] | [Minimax Signal Detection in Sparse Additive Models.](http://arxiv.org/abs/2304.09398) | 本研究针对稀疏加性模型中的信号检测问题建立了极小极大分离速率，揭示了稀疏性和函数空间选择之间的非平凡交互作用，并研究了对稀疏性的自适应性和其在通用函数空间中的适用性。在Sobolev空间设置下，我们还讨论了对稀疏性和平滑性的自适应性。 |
| [^9] | [Importance is Important: A Guide to Informed Importance Tempering Methods.](http://arxiv.org/abs/2304.06251) | 本论文详细介绍了一种易于实施的MCMC算法IIT及其在许多情况下的应用。该算法始终接受有信息的提议，可与其他MCMC技术相结合，并带来新的优化抽样器的机会。 |
| [^10] | [Differentially Private Bootstrap: New Privacy Analysis and Inference Strategies.](http://arxiv.org/abs/2210.06140) | 本文研究了一种差分隐私引导采样方法，提供了隐私成本的新结果，可用于推断样本分布并构建置信区间，同时指出了现有文献中的误用。随着采样次数趋近无限大，此方法逐渐满足更严格的差分隐私要求。 |

# 详细

[^1]: 使用SoS密度估计和α-离散度的顺序传输映射

    Sequential transport maps using SoS density estimation and $\alpha$-divergences

    [https://arxiv.org/abs/2402.17943](https://arxiv.org/abs/2402.17943)

    本研究探讨了使用SoS密度和α-离散度来近似中间密度的顺序传输映射框架，通过将两者结合，可以有效地解决凸优化问题，进而实现从未标准化的密度生成样本。

    

    基于传输的密度估计方法因其能够有效地从近似密度生成样本而受到越来越多的关注。我们进一步调查了提出的顺序传输映射框架，该框架建立在一系列组成的Knothe-Rosenblatt（KR）映射之上。其中每个映射都是通过首先估计中等复杂度的中间密度，然后通过计算从参考密度到预计算近似密度的精确KR映射而构建的。在我们的工作中，我们探索了使用SoS密度和α-离散度来近似中间密度。有趣的是，将SoS密度与α-离散度相结合产生了凸优化问题，可以通过半定编程有效地解决。α-离散度的主要优势在于使得能够处理未标准化的密度，从而提供...

    arXiv:2402.17943v1 Announce Type: cross  Abstract: Transport-based density estimation methods are receiving growing interest because of their ability to efficiently generate samples from the approximated density. We further invertigate the sequential transport maps framework proposed from arXiv:2106.04170 arXiv:2303.02554, which builds on a sequence of composed Knothe-Rosenblatt (KR) maps. Each of those maps are built by first estimating an intermediate density of moderate complexity, and then by computing the exact KR map from a reference density to the precomputed approximate density. In our work, we explore the use of Sum-of-Squares (SoS) densities and $\alpha$-divergences for approximating the intermediate densities. Combining SoS densities with $\alpha$-divergence interestingly yields convex optimization problems which can be efficiently solved using semidefinite programming. The main advantage of $\alpha$-divergences is to enable working with unnormalized densities, which provide
    
[^2]: 离散时间扩散模型的非渐近收敛：新方法和改进速率

    Non-asymptotic Convergence of Discrete-time Diffusion Models: New Approach and Improved Rate

    [https://arxiv.org/abs/2402.13901](https://arxiv.org/abs/2402.13901)

    本文提出了离散时间扩散模型的新方法，改进了对更大类的分布的收敛保证，并提高了具有有界支撑的分布的收敛速率。

    

    最近，去噪扩散模型作为一种强大的生成技术出现，将噪声转化为数据。理论上主要研究了连续时间扩散模型的收敛性保证，并且仅在文献中对具有有界支撑的分布的离散时间扩散模型进行了获得。本文为更大类的分布建立了离散时间扩散模型的收敛性保证，并进一步改进了对具有有界支撑的分布的收敛速率。特别地，首先为具有有限二阶矩的平滑和一般（可能非光滑）分布建立了收敛速率。然后将结果专门应用于一些有明确参数依赖关系的有趣分布类别，包括具有Lipschitz分数、高斯混合分布和具有有界支撑的分布。

    arXiv:2402.13901v1 Announce Type: new  Abstract: The denoising diffusion model emerges recently as a powerful generative technique that converts noise into data. Theoretical convergence guarantee has been mainly studied for continuous-time diffusion models, and has been obtained for discrete-time diffusion models only for distributions with bounded support in the literature. In this paper, we establish the convergence guarantee for substantially larger classes of distributions under discrete-time diffusion models and further improve the convergence rate for distributions with bounded support. In particular, we first establish the convergence rates for both smooth and general (possibly non-smooth) distributions having finite second moment. We then specialize our results to a number of interesting classes of distributions with explicit parameter dependencies, including distributions with Lipschitz scores, Gaussian mixture distributions, and distributions with bounded support. We further 
    
[^3]: 使用图神经网络进行超图节点分类

    Hypergraph Node Classification With Graph Neural Networks

    [https://arxiv.org/abs/2402.05569](https://arxiv.org/abs/2402.05569)

    本研究提出了一种简单高效的框架，利用加权子图扩展的图神经网络(WCE-GNN)实现了超图节点分类。实验证明，WCE-GNN具有优秀的预测效果和较低的计算复杂度。

    

    超图是用来模拟现实世界数据中的高阶相互作用的关键。图神经网络（GNNs）的成功揭示了神经网络处理具有成对交互的数据的能力。这激发了使用神经网络处理具有高阶相互作用的数据的想法，从而导致了超图神经网络（HyperGNNs）的发展。GNNs和HyperGNNs通常被认为是不同的，因为它们被设计用于处理不同几何拓扑的数据。然而，在本文中，我们在理论上证明，在节点分类的上下文中，大多数HyperGNNs可以使用带有超图的加权子图扩展的GNN来近似。这导致了WCE-GNN，一种简单高效的框架，包括一个GNN和一个加权子图扩展（WCE），用于超图节点分类。对于九个真实世界的超图节点分类数据集的实验表明，WCE-GNN不仅具有优秀的预测效果，而且具有较低的计算复杂度。

    Hypergraphs, with hyperedges connecting more than two nodes, are key for modelling higher-order interactions in real-world data. The success of graph neural networks (GNNs) reveals the capability of neural networks to process data with pairwise interactions. This inspires the usage of neural networks for data with higher-order interactions, thereby leading to the development of hypergraph neural networks (HyperGNNs). GNNs and HyperGNNs are typically considered distinct since they are designed for data on different geometric topologies. However, in this paper, we theoretically demonstrate that, in the context of node classification, most HyperGNNs can be approximated using a GNN with a weighted clique expansion of the hypergraph. This leads to WCE-GNN, a simple and efficient framework comprising a GNN and a weighted clique expansion (WCE), for hypergraph node classification. Experiments on nine real-world hypergraph node classification benchmarks showcase that WCE-GNN demonstrates not o
    
[^4]: 用于神经密度比估计的$\alpha$-散度损失函数

    $\alpha$-Divergence Loss Function for Neural Density Ratio Estimation

    [https://arxiv.org/abs/2402.02041](https://arxiv.org/abs/2402.02041)

    本文提出了一种应用于神经密度比估计的$\alpha$-散度损失函数($\alpha$-Div)，通过简洁实现和稳定优化解决了现有方法中存在的优化问题。实验证明了这种损失函数的稳定性，并提出了对DRE任务的估计准确性的研究，同时给出了样本要求的解决方案。

    

    最近，神经网络在机器学习中的基础技术密度比估计(DRE)方面取得了最先进的结果。然而，现有方法因DRE的损失函数而出现了优化问题：KL散度需要大样本，训练损失梯度消失，损失函数梯度有偏。因此，本文提出了一种提供简洁实现和稳定优化的$\alpha$-散度损失函数($\alpha$-Div)。此外，还给出了对所提出的损失函数的技术验证。实验证明了所提出的损失函数的稳定性，并研究了DRE任务的估计准确性。此外，本研究还提出了使用所提出的损失函数进行DRE的样本要求，以$L_1$误差的上界联系起来，该上界将高维度DRE任务中的维度诅咒作为一个共同问题。

    Recently, neural networks have produced state-of-the-art results for density-ratio estimation (DRE), a fundamental technique in machine learning. However, existing methods bear optimization issues that arise from the loss functions of DRE: a large sample requirement of Kullback--Leibler (KL)-divergence, vanishing of train loss gradients, and biased gradients of the loss functions. Thus, an $\alpha$-divergence loss function ($\alpha$-Div) that offers concise implementation and stable optimization is proposed in this paper. Furthermore, technical justifications for the proposed loss function are presented. The stability of the proposed loss function is empirically demonstrated and the estimation accuracy of DRE tasks is investigated. Additionally, this study presents a sample requirement for DRE using the proposed loss function in terms of the upper bound of $L_1$ error, which connects a curse of dimensionality as a common problem in high-dimensional DRE tasks.
    
[^5]: 随机向量均值的时间均匀置信球

    Time-Uniform Confidence Spheres for Means of Random Vectors

    [https://arxiv.org/abs/2311.08168](https://arxiv.org/abs/2311.08168)

    该研究提出了时间均匀置信球序列，可以同时高概率地包含各种样本量下随机向量的均值，并针对不同分布假设进行了扩展和统一分析。

    

    我们推导并研究了时间均匀置信球——包含随机向量均值并且跨越所有样本量具有很高概率的置信球序列（CSSs）。受Catoni和Giulini原始工作启发，我们统一并扩展了他们的分析，涵盖顺序设置并处理各种分布假设。我们的结果包括有界随机向量的经验伯恩斯坦CSS（导致新颖的经验伯恩斯坦置信区间，渐近宽度按照真实未知方差成比例缩放）、用于子-$\psi$随机向量的CSS（包括子伽马、子泊松和子指数分布）、和用于重尾随机向量（仅有两阶矩）的CSS。最后，我们提供了两个抵抗Huber噪声污染的CSS。第一个是我们经验伯恩斯坦CSS的鲁棒版本，第二个扩展了单变量序列最近的工作。

    arXiv:2311.08168v2 Announce Type: replace-cross  Abstract: We derive and study time-uniform confidence spheres -- confidence sphere sequences (CSSs) -- which contain the mean of random vectors with high probability simultaneously across all sample sizes. Inspired by the original work of Catoni and Giulini, we unify and extend their analysis to cover both the sequential setting and to handle a variety of distributional assumptions. Our results include an empirical-Bernstein CSS for bounded random vectors (resulting in a novel empirical-Bernstein confidence interval with asymptotic width scaling proportionally to the true unknown variance), CSSs for sub-$\psi$ random vectors (which includes sub-gamma, sub-Poisson, and sub-exponential), and CSSs for heavy-tailed random vectors (two moments only). Finally, we provide two CSSs that are robust to contamination by Huber noise. The first is a robust version of our empirical-Bernstein CSS, and the second extends recent work in the univariate se
    
[^6]: 针对随机矩阵计算的高效误差和方差估计

    Efficient error and variance estimation for randomized matrix computations

    [https://arxiv.org/abs/2207.06342](https://arxiv.org/abs/2207.06342)

    该论文提出了用于随机矩阵计算的高效误差和方差估计方法，可帮助评估输出质量并指导算法参数选择。

    

    随机矩阵算法已成为科学计算和机器学习中必不可少的工具。为了安全地在应用中使用这些算法，需要结合后验误差估计来评估输出的质量。为满足这一需求，本文提出了两种诊断方法：用于随机低秩逼近的留一法误差估计器和一种杰基刀重采样方法，用于估计随机矩阵计算的输出方差。这两种诊断方法对于随机低秩逼近算法（如随机奇异值分解和随机Nystrom逼近）计算迅速，并提供可用于评估计算输出质量和指导算法参数选择的有用信息。

    arXiv:2207.06342v4 Announce Type: replace-cross  Abstract: Randomized matrix algorithms have become workhorse tools in scientific computing and machine learning. To use these algorithms safely in applications, they should be coupled with posterior error estimates to assess the quality of the output. To meet this need, this paper proposes two diagnostics: a leave-one-out error estimator for randomized low-rank approximations and a jackknife resampling method to estimate the variance of the output of a randomized matrix computation. Both of these diagnostics are rapid to compute for randomized low-rank approximation algorithms such as the randomized SVD and randomized Nystr\"om approximation, and they provide useful information that can be used to assess the quality of the computed output and guide algorithmic parameter choices.
    
[^7]: 将大量随机点拟合成椭球体的问题

    Fitting an ellipsoid to a quadratic number of random points. (arXiv:2307.01181v1 [math.PR])

    [http://arxiv.org/abs/2307.01181](http://arxiv.org/abs/2307.01181)

    将$n$个高斯随机向量拟合到以原点为中心的椭球体边界的问题$(\mathrm{P})$，我们提出了一个基于随机向量Gram矩阵集中性的改进方法，证明了当$n \leq d^2 / C$时，问题$(\mathrm{P})$具有很高的可行性概率。

    

    我们考虑当$n, d \to \infty $时，将$n$个标准高斯随机向量拟合到以原点为中心的椭球体的边界的问题$(\mathrm{P})$。这个问题被猜测具有尖锐的可行性转变：对于任意$\varepsilon > 0$，如果$n \leq (1 - \varepsilon) d^2 / 4$，那么$(\mathrm{P})$有很高的概率有解；而如果$n \geq (1 + \varepsilon) d^2 /4$，那么$(\mathrm{P})$有很高的概率无解。目前，对于负面情况，只知道$n \geq d^2 / 2$是平凡的一个上界，而对于正面情况，已知的最好结果是假设$n \leq d^2 / \mathrm{polylog}(d)$。在这项工作中，我们利用Bartl和Mendelson关于随机向量的Gram矩阵集中性的一个关键结果改进了以前的方法。这使得我们可以给出一个简单的证明，当$n \leq d^2 / C$时，问题$(\mathrm{P})$有很高的概率是可行的，其中$C> 0$是一个（可能很大的）常数。

    We consider the problem $(\mathrm{P})$ of fitting $n$ standard Gaussian random vectors in $\mathbb{R}^d$ to the boundary of a centered ellipsoid, as $n, d \to \infty$. This problem is conjectured to have a sharp feasibility transition: for any $\varepsilon > 0$, if $n \leq (1 - \varepsilon) d^2 / 4$ then $(\mathrm{P})$ has a solution with high probability, while $(\mathrm{P})$ has no solutions with high probability if $n \geq (1 + \varepsilon) d^2 /4$. So far, only a trivial bound $n \geq d^2 / 2$ is known on the negative side, while the best results on the positive side assume $n \leq d^2 / \mathrm{polylog}(d)$. In this work, we improve over previous approaches using a key result of Bartl & Mendelson on the concentration of Gram matrices of random vectors under mild assumptions on their tail behavior. This allows us to give a simple proof that $(\mathrm{P})$ is feasible with high probability when $n \leq d^2 / C$, for a (possibly large) constant $C > 0$.
    
[^8]: 稀疏加性模型中的极小极大信号检测

    Minimax Signal Detection in Sparse Additive Models. (arXiv:2304.09398v1 [math.ST])

    [http://arxiv.org/abs/2304.09398](http://arxiv.org/abs/2304.09398)

    本研究针对稀疏加性模型中的信号检测问题建立了极小极大分离速率，揭示了稀疏性和函数空间选择之间的非平凡交互作用，并研究了对稀疏性的自适应性和其在通用函数空间中的适用性。在Sobolev空间设置下，我们还讨论了对稀疏性和平滑性的自适应性。

    

    在高维度的建模需求中，稀疏加性模型是一种有吸引力的选择。我们研究了信号检测问题，并建立了一个稀疏加性信号检测的极小极大分离速率。我们的结果是非渐近的，并适用于单变量分量函数属于一般再生核希尔伯特空间的情况。与估计理论不同，极小极大分离速率揭示了稀疏性和函数空间选择之间的非平凡交互作用。我们还研究了对稀疏性的自适应性，并建立了一个通用函数空间的自适应测试速率；在某些空间中，自适应性是可能的，而在其他空间中则会产生不可避免的代价。最后，我们在Sobolev空间设置下研究了对稀疏性和平滑性的自适应性，并更正了文献中存在的一些说法。

    Sparse additive models are an attractive choice in circumstances calling for modelling flexibility in the face of high dimensionality. We study the signal detection problem and establish the minimax separation rate for the detection of a sparse additive signal. Our result is nonasymptotic and applicable to the general case where the univariate component functions belong to a generic reproducing kernel Hilbert space. Unlike the estimation theory, the minimax separation rate reveals a nontrivial interaction between sparsity and the choice of function space. We also investigate adaptation to sparsity and establish an adaptive testing rate for a generic function space; adaptation is possible in some spaces while others impose an unavoidable cost. Finally, adaptation to both sparsity and smoothness is studied in the setting of Sobolev space, and we correct some existing claims in the literature.
    
[^9]: 实用指南：关于知情重要性调节方法的详细介绍

    Importance is Important: A Guide to Informed Importance Tempering Methods. (arXiv:2304.06251v1 [stat.CO])

    [http://arxiv.org/abs/2304.06251](http://arxiv.org/abs/2304.06251)

    本论文详细介绍了一种易于实施的MCMC算法IIT及其在许多情况下的应用。该算法始终接受有信息的提议，可与其他MCMC技术相结合，并带来新的优化抽样器的机会。

    

    知情重要性调节 (IIT) 是一种易于实施的MCMC算法，可视为通常的Metropolis-Hastings算法的扩展，具有始终接受有信息的提议的特殊功能，在Zhou和Smith（2022年）的研究中表明在一些常见情况下收敛更快。本文开发了一个新的、全面的指南，介绍了IIT在许多情况下的应用。首先，我们提出了两种IIT方案，这些方案在离散空间上的运行速度比现有的知情MCMC方法更快，因为它们不需要计算所有相邻状态的后验概率。其次，我们将IIT与其他MCMC技术（包括模拟回火、伪边缘和多重尝试方法，在一般状态空间上实施为Metropolis-Hastings方案，可能遭受低接受率的问题）进行了整合。使用IIT使我们能够始终接受提议，并带来了优化抽样器的新机会，这是在Metropolis-Hastings算法下不可能的。最后，我们提供了一个实用的指南，以选择IIT方案和调整算法参数。对各种模型的实验结果证明了我们所提出的方法的有效性。

    Informed importance tempering (IIT) is an easy-to-implement MCMC algorithm that can be seen as an extension of the familiar Metropolis-Hastings algorithm with the special feature that informed proposals are always accepted, and which was shown in Zhou and Smith (2022) to converge much more quickly in some common circumstances. This work develops a new, comprehensive guide to the use of IIT in many situations. First, we propose two IIT schemes that run faster than existing informed MCMC methods on discrete spaces by not requiring the posterior evaluation of all neighboring states. Second, we integrate IIT with other MCMC techniques, including simulated tempering, pseudo-marginal and multiple-try methods (on general state spaces), which have been conventionally implemented as Metropolis-Hastings schemes and can suffer from low acceptance rates. The use of IIT allows us to always accept proposals and brings about new opportunities for optimizing the sampler which are not possible under th
    
[^10]: 差分隐私引导采样：新的隐私分析与推断策略

    Differentially Private Bootstrap: New Privacy Analysis and Inference Strategies. (arXiv:2210.06140v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.06140](http://arxiv.org/abs/2210.06140)

    本文研究了一种差分隐私引导采样方法，提供了隐私成本的新结果，可用于推断样本分布并构建置信区间，同时指出了现有文献中的误用。随着采样次数趋近无限大，此方法逐渐满足更严格的差分隐私要求。

    

    差分隐私机制通过引入随机性来保护个人信息，但在应用中，统计推断仍然缺乏通用技术。本文研究了一个差分隐私引导采样方法，通过发布多个私有引导采样估计来推断样本分布并构建置信区间。我们的隐私分析提供了单个差分隐私引导采样估计的隐私成本新结果，适用于任何差分隐私机制，并指出了现有文献中引导采样的一些误用。使用Gaussian-DP（GDP）框架，我们证明从满足 $(\mu/\sqrt{(2-2/\mathrm{e})B})$-GDP 的机制中释放 $B$ 个差分隐私引导采样估计，在 $B$ 趋近无限大时渐近地满足 $\mu$-GDP。此外，我们使用差分隐私引导采样估计的反卷积对样本分布进行准确推断。

    Differentially private (DP) mechanisms protect individual-level information by introducing randomness into the statistical analysis procedure. Despite the availability of numerous DP tools, there remains a lack of general techniques for conducting statistical inference under DP. We examine a DP bootstrap procedure that releases multiple private bootstrap estimates to infer the sampling distribution and construct confidence intervals (CIs). Our privacy analysis presents new results on the privacy cost of a single DP bootstrap estimate, applicable to any DP mechanisms, and identifies some misapplications of the bootstrap in the existing literature. Using the Gaussian-DP (GDP) framework (Dong et al.,2022), we show that the release of $B$ DP bootstrap estimates from mechanisms satisfying $(\mu/\sqrt{(2-2/\mathrm{e})B})$-GDP asymptotically satisfies $\mu$-GDP as $B$ goes to infinity. Moreover, we use deconvolution with the DP bootstrap estimates to accurately infer the sampling distribution
    

