# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Randomization Can Reduce Both Bias and Variance: A Case Study in Random Forests](https://arxiv.org/abs/2402.12668) | 随机森林相对于装袋法具有减少偏差的能力，在揭示数据模式和高信噪比情况下表现更好的特点，为随机森林在不同信噪比环境下的成功提供了解释和实用见解。 |
| [^2] | [Resilience of the quadratic Littlewood-Offord problem](https://arxiv.org/abs/2402.10504) | 论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。 |
| [^3] | [Fun with Flags: Robust Principal Directions via Flag Manifolds.](http://arxiv.org/abs/2401.04071) | 本研究提出了一种统一的PCA和其变种框架，该框架基于线性子空间旗帜，并引入了对异常值和数据流形的考虑。通过在旗帜流形上进行优化问题的求解，结合主测地线近似，提出了一系列新的降维算法。 |
| [^4] | [Solving PDEs on Spheres with Physics-Informed Convolutional Neural Networks.](http://arxiv.org/abs/2308.09605) | 本文严格分析了在球面上解决PDEs的物理信息卷积神经网络（PICNN），通过使用最新的逼近结果和球谐分析，证明了逼近误差与Sobolev范数的上界，并建立了快速收敛速率。实验结果也验证了理论分析的有效性。 |
| [^5] | [Adaptive Principal Component Regression with Applications to Panel Data.](http://arxiv.org/abs/2307.01357) | 本文提出了自适应主成分回归方法，并在面板数据中的应用中获得了均匀有限样本保证。该方法可以用于面板数据中的实验设计，特别是当干预方案是自适应分配的情况。 |
| [^6] | [A Primal-dual Approach for Solving Variational Inequalities with General-form Constraints.](http://arxiv.org/abs/2210.15659) | 本文介绍了一种解决具有一般形式约束的变分不等式的原始-对偶方法，并使用一种温启动技术，在每次迭代中近似地解决子问题，我们证明了它的收敛性，并展示了它的收敛速度比其精确对应物更快。 |
| [^7] | [On the Generalized Likelihood Ratio Test and One-Class Classifiers.](http://arxiv.org/abs/2210.12494) | 本文考虑了一类分类器和广义似然比检验的问题，证明了多层感知器神经网络和支持向量机模型在收敛时会表现为广义似然比检验。同时，作者还展示了一类最小二乘SVM在收敛时也能达到广义似然比检验的效果。 |

# 详细

[^1]: 随机化既可以减少偏差又可以减少方差：随机森林的案例研究

    Randomization Can Reduce Both Bias and Variance: A Case Study in Random Forests

    [https://arxiv.org/abs/2402.12668](https://arxiv.org/abs/2402.12668)

    随机森林相对于装袋法具有减少偏差的能力，在揭示数据模式和高信噪比情况下表现更好的特点，为随机森林在不同信噪比环境下的成功提供了解释和实用见解。

    

    我们研究了往往被忽视的现象，首次在\cite{breiman2001random}中指出，即随机森林似乎比装袋法减少了偏差。受\cite{mentch2020randomization}一篇有趣的论文的启发，其中作者认为随机森林减少了有效自由度，并且只有在低信噪比（SNR）环境下才能胜过装袋集成，我们探讨了随机森林如何能够揭示被装袋法忽视的数据模式。我们在实证中证明，在存在这种模式的情况下，随机森林不仅可以减小偏差还能减小方差，并且当信噪比高时随机森林的表现愈发好于装袋集成。我们的观察为解释随机森林在各种信噪比情况下的真实世界成功提供了见解，并增进了我们对随机森林与装袋集成在每次分割注入的随机化方面的差异的理解。我们的调查结果还提供了实用见解。

    arXiv:2402.12668v1 Announce Type: cross  Abstract: We study the often overlooked phenomenon, first noted in \cite{breiman2001random}, that random forests appear to reduce bias compared to bagging. Motivated by an interesting paper by \cite{mentch2020randomization}, where the authors argue that random forests reduce effective degrees of freedom and only outperform bagging ensembles in low signal-to-noise ratio (SNR) settings, we explore how random forests can uncover patterns in the data missed by bagging. We empirically demonstrate that in the presence of such patterns, random forests reduce bias along with variance and increasingly outperform bagging ensembles when SNR is high. Our observations offer insights into the real-world success of random forests across a range of SNRs and enhance our understanding of the difference between random forests and bagging ensembles with respect to the randomization injected into each split. Our investigations also yield practical insights into the 
    
[^2]: 二次Littlewood-Offord问题的弹性

    Resilience of the quadratic Littlewood-Offord problem

    [https://arxiv.org/abs/2402.10504](https://arxiv.org/abs/2402.10504)

    论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。

    

    我们研究了高维数据的统计鲁棒性。我们的结果提供了关于对抗性噪声对二次Radamecher混沌$\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$反集中特性的影响的估计，其中$M$是一个固定的（高维）矩阵，$\boldsymbol{\xi}$是一个共形Rademacher向量。具体来说，我们探讨了$\boldsymbol{\xi}$能够承受多少对抗性符号翻转而不“膨胀”$\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$，从而“去除”原始分布导致更“有粒度”和对抗性偏倚的分布。我们的结果为二次和双线性Rademacher混沌的统计鲁棒性提供了下限估计；这些结果在关键区域被证明是渐近紧的。

    arXiv:2402.10504v1 Announce Type: cross  Abstract: We study the statistical resilience of high-dimensional data. Our results provide estimates as to the effects of adversarial noise over the anti-concentration properties of the quadratic Radamecher chaos $\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$, where $M$ is a fixed (high-dimensional) matrix and $\boldsymbol{\xi}$ is a conformal Rademacher vector. Specifically, we pursue the question of how many adversarial sign-flips can $\boldsymbol{\xi}$ sustain without "inflating" $\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$ and thus "de-smooth" the original distribution resulting in a more "grainy" and adversarially biased distribution. Our results provide lower bound estimations for the statistical resilience of the quadratic and bilinear Rademacher chaos; these are shown to be asymptotically tight across key regimes.
    
[^3]: 旗帜游戏：通过旗帜流形来获得鲁棒的主方向

    Fun with Flags: Robust Principal Directions via Flag Manifolds. (arXiv:2401.04071v1 [cs.CV])

    [http://arxiv.org/abs/2401.04071](http://arxiv.org/abs/2401.04071)

    本研究提出了一种统一的PCA和其变种框架，该框架基于线性子空间旗帜，并引入了对异常值和数据流形的考虑。通过在旗帜流形上进行优化问题的求解，结合主测地线近似，提出了一系列新的降维算法。

    

    主成分分析（PCA）及其对流形和异常数据的扩展，在计算机视觉和机器学习中是不可或缺的。本研究提出了PCA及其变种的统一形式，引入了基于线性子空间旗帜的框架，即逐渐增加维度的嵌套线性子空间的层次结构，不仅允许共同实现，还产生了新的未曾探索的变种。我们从广义化传统的PCA方法开始，这些方法要么最大化方差，要么最小化重构误差。我们扩展这些解释，通过考虑异常值和数据流形，开发出了大量新的降维算法。为了设计一种通用的计算方法，我们将鲁棒和对偶形式的PCA重新构建为在旗帜流形上的优化问题。然后，我们将主测地线近似（切线PCA）整合到这个基于旗帜的框架中，创造出一种新的方法。

    Principal component analysis (PCA), along with its extensions to manifolds and outlier contaminated data, have been indispensable in computer vision and machine learning. In this work, we present a unifying formalism for PCA and its variants, and introduce a framework based on the flags of linear subspaces, \ie a hierarchy of nested linear subspaces of increasing dimension, which not only allows for a common implementation but also yields novel variants, not explored previously. We begin by generalizing traditional PCA methods that either maximize variance or minimize reconstruction error. We expand these interpretations to develop a wide array of new dimensionality reduction algorithms by accounting for outliers and the data manifold. To devise a common computational approach, we recast robust and dual forms of PCA as optimization problems on flag manifolds. We then integrate tangent space approximations of principal geodesic analysis (tangent-PCA) into this flag-based framework, crea
    
[^4]: 使用物理信息卷积神经网络在球面上解决偏微分方程

    Solving PDEs on Spheres with Physics-Informed Convolutional Neural Networks. (arXiv:2308.09605v1 [math.NA])

    [http://arxiv.org/abs/2308.09605](http://arxiv.org/abs/2308.09605)

    本文严格分析了在球面上解决PDEs的物理信息卷积神经网络（PICNN），通过使用最新的逼近结果和球谐分析，证明了逼近误差与Sobolev范数的上界，并建立了快速收敛速率。实验结果也验证了理论分析的有效性。

    

    物理信息神经网络（PINNs）已被证明在解决各种实验角度中的偏微分方程（PDEs）方面非常高效。一些最近的研究还提出了针对表面，包括球面上的PDEs的PINN算法。然而，对于PINNs的数值性能，尤其是在表面或流形上的PINNs，仍然缺乏理论理解。本文中，我们对用于在球面上解决PDEs的物理信息卷积神经网络（PICNN）进行了严格分析。通过使用和改进深度卷积神经网络和球谐分析的最新逼近结果，我们证明了该逼近误差与Sobolev范数的上界。随后，我们将这一结果与创新的局部复杂度分析相结合，建立了PICNN的快速收敛速率。我们的理论结果也得到了实验的验证和补充。鉴于这些发现，

    Physics-informed neural networks (PINNs) have been demonstrated to be efficient in solving partial differential equations (PDEs) from a variety of experimental perspectives. Some recent studies have also proposed PINN algorithms for PDEs on surfaces, including spheres. However, theoretical understanding of the numerical performance of PINNs, especially PINNs on surfaces or manifolds, is still lacking. In this paper, we establish rigorous analysis of the physics-informed convolutional neural network (PICNN) for solving PDEs on the sphere. By using and improving the latest approximation results of deep convolutional neural networks and spherical harmonic analysis, we prove an upper bound for the approximation error with respect to the Sobolev norm. Subsequently, we integrate this with innovative localization complexity analysis to establish fast convergence rates for PICNN. Our theoretical results are also confirmed and supplemented by our experiments. In light of these findings, we expl
    
[^5]: 自适应主成分回归在面板数据中的应用

    Adaptive Principal Component Regression with Applications to Panel Data. (arXiv:2307.01357v1 [cs.LG])

    [http://arxiv.org/abs/2307.01357](http://arxiv.org/abs/2307.01357)

    本文提出了自适应主成分回归方法，并在面板数据中的应用中获得了均匀有限样本保证。该方法可以用于面板数据中的实验设计，特别是当干预方案是自适应分配的情况。

    

    主成分回归(PCR)是一种流行的固定设计误差变量回归技术，它是线性回归的推广，观测的协变量受到随机噪声的污染。我们在数据收集时提供了在线（正则化）PCR的第一次均匀有限样本保证。由于分析固定设计中PCR的证明技术无法很容易地扩展到在线设置，我们的结果依赖于将现代鞅浓度的工具适应到误差变量设置中。作为我们界限的应用，我们在面板数据设置中提供了实验设计框架，当干预被自适应地分配时。我们的框架可以被认为是合成控制和合成干预框架的泛化，其中数据是通过自适应干预分配策略收集的。

    Principal component regression (PCR) is a popular technique for fixed-design error-in-variables regression, a generalization of the linear regression setting in which the observed covariates are corrupted with random noise. We provide the first time-uniform finite sample guarantees for online (regularized) PCR whenever data is collected adaptively. Since the proof techniques for analyzing PCR in the fixed design setting do not readily extend to the online setting, our results rely on adapting tools from modern martingale concentration to the error-in-variables setting. As an application of our bounds, we provide a framework for experiment design in panel data settings when interventions are assigned adaptively. Our framework may be thought of as a generalization of the synthetic control and synthetic interventions frameworks, where data is collected via an adaptive intervention assignment policy.
    
[^6]: 解决具有一般形式约束的变分不等式的原始-对偶方法

    A Primal-dual Approach for Solving Variational Inequalities with General-form Constraints. (arXiv:2210.15659v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.15659](http://arxiv.org/abs/2210.15659)

    本文介绍了一种解决具有一般形式约束的变分不等式的原始-对偶方法，并使用一种温启动技术，在每次迭代中近似地解决子问题，我们证明了它的收敛性，并展示了它的收敛速度比其精确对应物更快。

    

    Yang等人最近通过一种一阶梯度方法解决了具有等式和不等式约束的变分不等式（VI）的问题。但是，提出的原始-对偶方法称为ACVI仅适用于可以计算其子问题的解析解的情况，因此一般情况仍然是一个开放的问题。在本文中，我们采用一种温启动技术，在每次迭代中近似地解决子问题，并使用在先前迭代中找到的近似解初始化变量。我们证明了它的收敛性，并表明当算子为$L$-Lipschitz且单调时，这种不精确的ACVI方法的最后一次迭代的间隙函数下降的速度为$\mathcal{O}(\frac{1}{\sqrt{K}})$，前提是错误以适当的速度下降。有趣的是，我们展示了在数值实验中，通常这种技术比其精确对应物收敛更快。此外，对于不等式约束的情况

    Yang et al. (2023) recently addressed the open problem of solving Variational Inequalities (VIs) with equality and inequality constraints through a first-order gradient method. However, the proposed primal-dual method called ACVI is applicable when we can compute analytic solutions of its subproblems; thus, the general case remains an open problem. In this paper, we adopt a warm-starting technique where we solve the subproblems approximately at each iteration and initialize the variables with the approximate solution found at the previous iteration. We prove its convergence and show that the gap function of the last iterate of this inexact-ACVI method decreases at a rate of $\mathcal{O}(\frac{1}{\sqrt{K}})$ when the operator is $L$-Lipschitz and monotone, provided that the errors decrease at appropriate rates. Interestingly, we show that often in numerical experiments, this technique converges faster than its exact counterpart. Furthermore, for the cases when the inequality constraints
    
[^7]: 关于广义似然比检验和一类分类器

    On the Generalized Likelihood Ratio Test and One-Class Classifiers. (arXiv:2210.12494v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.12494](http://arxiv.org/abs/2210.12494)

    本文考虑了一类分类器和广义似然比检验的问题，证明了多层感知器神经网络和支持向量机模型在收敛时会表现为广义似然比检验。同时，作者还展示了一类最小二乘SVM在收敛时也能达到广义似然比检验的效果。

    

    一类分类（OCC）是决定观察样本是否属于目标类的问题。我们考虑在包含目标类样本的数据集上学习一个表现为广义似然比检验（GLRT）的OCC模型的问题。当目标类的统计信息可用时，GLRT解决了相同的问题。GLRT是一个众所周知且在特定条件下可证明最佳的分类器。为此，我们考虑了多层感知器神经网络（NN）和支持向量机（SVM）模型。它们使用人工数据集训练为两类分类器，其中替代类使用在目标类数据集的定义域上均匀生成的随机样本。我们证明，在适当的假设下，模型在大数据集上收敛到了GLRT。此外，我们还展示了具有适当核函数的一类最小二乘SVM（OCLSSVM）在收敛时表现为GLRT。

    One-class classification (OCC) is the problem of deciding whether an observed sample belongs to a target class. We consider the problem of learning an OCC model that performs as the generalized likelihood ratio test (GLRT), given a dataset containing samples of the target class. The GLRT solves the same problem when the statistics of the target class are available. The GLRT is a well-known and provably optimal (under specific assumptions) classifier. To this end, we consider both the multilayer perceptron neural network (NN) and the support vector machine (SVM) models. They are trained as two-class classifiers using an artificial dataset for the alternative class, obtained by generating random samples, uniformly over the domain of the target-class dataset. We prove that, under suitable assumptions, the models converge (with a large dataset) to the GLRT. Moreover, we show that the one-class least squares SVM (OCLSSVM) with suitable kernels at convergence performs as the GLRT. Lastly, we
    

