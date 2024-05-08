# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://arxiv.org/abs/2402.12875) | 思维链赋予变压器模型执行固有串行计算的能力，提高了变压器在算术和符号推理任务中的准确性。 |
| [^2] | [Misspecification uncertainties in near-deterministic regression](https://arxiv.org/abs/2402.01810) | 该论文研究了近确定性回归中错误规范化的不确定性问题，并提出了一种组合模型，以准确预测和控制参数不确定性。 |
| [^3] | [Scalable network reconstruction in subquadratic time.](http://arxiv.org/abs/2401.01404) | 这篇论文提出了一个可扩展的网络重建算法，能够在次二次时间内实现结果，通过随机的二阶邻居搜索产生最佳的边候选。 |
| [^4] | [Bayesian Quantile Regression with Subset Selection: A Posterior Summarization Perspective.](http://arxiv.org/abs/2311.02043) | 本研究提出了一种基于贝叶斯决策分析的方法，对于任何贝叶斯回归模型，可以得到每个条件分位数的最佳和可解释的线性估计值和不确定性量化。该方法是一种适用于特定分位数子集选择的有效工具。 |
| [^5] | [MLP-Mixer as a Wide and Sparse MLP.](http://arxiv.org/abs/2306.01470) | 深度学习中常用的MLP有潜力提高性能。本研究揭示MLP-Mixer 可以作为具有稀疏权重的宽MLP有效地工作。 |
| [^6] | [The Projected Covariance Measure for assumption-lean variable significance testing.](http://arxiv.org/abs/2211.02039) | 该论文介绍了一种假设简约的变量重要性检验方法，利用非参数或机器学习方法实现稳健的误差控制和高功效。 |

# 详细

[^1]: 思维链激发变压器解决固有串行问题的能力

    Chain of Thought Empowers Transformers to Solve Inherently Serial Problems

    [https://arxiv.org/abs/2402.12875](https://arxiv.org/abs/2402.12875)

    思维链赋予变压器模型执行固有串行计算的能力，提高了变压器在算术和符号推理任务中的准确性。

    

    指导模型生成一系列中间步骤，即思维链（CoT），是提高大型语言模型（LLMs）在算术和符号推理任务上准确性的高效方法。然而，CoT背后的机制仍不清楚。这项工作通过表达性的视角提供了对解码器专用变压器的CoT能力的理论理解。在概念上，CoT赋予模型执行固有串行计算的能力，而这种能力在变压器中缺乏，特别是当深度较低时。先前的作品已经表明，在没有CoT的情况下，具有有限精度$\mathsf{poly}(n)$嵌入尺寸的恒定深度变压器只能在$\mathsf{TC}^0$中解决问题。我们首先展示了具有常数位精度的恒定深度变压器的更紧密的表达性上界，它只能解决$\mathsf{AC}^0$中的问题。

    arXiv:2402.12875v1 Announce Type: new  Abstract: Instructing the model to generate a sequence of intermediate steps, a.k.a., a chain of thought (CoT), is a highly effective method to improve the accuracy of large language models (LLMs) on arithmetics and symbolic reasoning tasks. However, the mechanism behind CoT remains unclear. This work provides a theoretical understanding of the power of CoT for decoder-only transformers through the lens of expressiveness. Conceptually, CoT empowers the model with the ability to perform inherently serial computation, which is otherwise lacking in transformers, especially when depth is low. Given input length $n$, previous works have shown that constant-depth transformers with finite precision $\mathsf{poly}(n)$ embedding size can only solve problems in $\mathsf{TC}^0$ without CoT. We first show an even tighter expressiveness upper bound for constant-depth transformers with constant-bit precision, which can only solve problems in $\mathsf{AC}^0$, a 
    
[^2]: 近确定性回归中的错误规范化不确定性

    Misspecification uncertainties in near-deterministic regression

    [https://arxiv.org/abs/2402.01810](https://arxiv.org/abs/2402.01810)

    该论文研究了近确定性回归中错误规范化的不确定性问题，并提出了一种组合模型，以准确预测和控制参数不确定性。

    

    期望损失是模型泛化误差的上界，可用于学习的鲁棒PAC-Bayes边界。然而，损失最小化被认为忽略了错误规范化，即模型不能完全复制观测结果。这导致大数据或欠参数化极限下对参数不确定性的显著低估。我们分析近确定性、错误规范化和欠参数化替代模型的泛化误差，这是科学和工程中广泛相关的一个领域。我们证明后验分布必须覆盖每个训练点，以避免发散的泛化误差，并导出一个符合这个约束的组合模型。对于线性模型，这种高效的方法产生的额外开销最小。这种高效方法在模型问题上进行了演示，然后应用于原子尺度机器学习中的高维数据集。

    The expected loss is an upper bound to the model generalization error which admits robust PAC-Bayes bounds for learning. However, loss minimization is known to ignore misspecification, where models cannot exactly reproduce observations. This leads to significant underestimates of parameter uncertainties in the large data, or underparameterized, limit. We analyze the generalization error of near-deterministic, misspecified and underparametrized surrogate models, a regime of broad relevance in science and engineering. We show posterior distributions must cover every training point to avoid a divergent generalization error and derive an ensemble {ansatz} that respects this constraint, which for linear models incurs minimal overhead. The efficient approach is demonstrated on model problems before application to high dimensional datasets in atomistic machine learning. Parameter uncertainties from misspecification survive in the underparametrized limit, giving accurate prediction and boundin
    
[^3]: 可扩展的子二次时间网络重建

    Scalable network reconstruction in subquadratic time. (arXiv:2401.01404v1 [cs.DS])

    [http://arxiv.org/abs/2401.01404](http://arxiv.org/abs/2401.01404)

    这篇论文提出了一个可扩展的网络重建算法，能够在次二次时间内实现结果，通过随机的二阶邻居搜索产生最佳的边候选。

    

    网络重建是指在只有关于条件偶联的观测数据，例如时间序列或图模型的独立样本的情况下，确定N个节点之间未观测到的成对耦合。针对这个问题提出的算法的可扩展性的主要障碍是似乎无法避免的二次复杂度O(N^2)，即要考虑每种可能的成对耦合至少一次，尽管大多数感兴趣的网络都是稀疏的，非零耦合的数量只有O(N)。在这里，我们提出了一个适用于广泛重建问题的通用算法，其在子二次时间内实现结果，其数据相关复杂度宽松上界为O(N^(3/2)logN)，但具有更典型的对数线性复杂度O(Nlog^2 N)。我们的算法依赖于一个随机的二阶邻居搜索，产生了最佳的边候选。

    Network reconstruction consists in determining the unobserved pairwise couplings between $N$ nodes given only observational data on the resulting behavior that is conditioned on those couplings -- typically a time-series or independent samples from a graphical model. A major obstacle to the scalability of algorithms proposed for this problem is a seemingly unavoidable quadratic complexity of $O(N^2)$, corresponding to the requirement of each possible pairwise coupling being contemplated at least once, despite the fact that most networks of interest are sparse, with a number of non-zero couplings that is only $O(N)$. Here we present a general algorithm applicable to a broad range of reconstruction problems that achieves its result in subquadratic time, with a data-dependent complexity loosely upper bounded by $O(N^{3/2}\log N)$, but with a more typical log-linear complexity of $O(N\log^2N)$. Our algorithm relies on a stochastic second neighbor search that produces the best edge candidat
    
[^4]: 基于子集选择的贝叶斯分位回归：后验总结视角

    Bayesian Quantile Regression with Subset Selection: A Posterior Summarization Perspective. (arXiv:2311.02043v1 [stat.ME])

    [http://arxiv.org/abs/2311.02043](http://arxiv.org/abs/2311.02043)

    本研究提出了一种基于贝叶斯决策分析的方法，对于任何贝叶斯回归模型，可以得到每个条件分位数的最佳和可解释的线性估计值和不确定性量化。该方法是一种适用于特定分位数子集选择的有效工具。

    

    分位回归是一种强大的工具，用于推断协变量如何影响响应分布的特定分位数。现有方法要么分别估计每个感兴趣分位数的条件分位数，要么使用半参数或非参数模型估计整个条件分布。前者经常产生不适合实际数据的模型，并且不在分位数之间共享信息，而后者则以复杂且受限制的模型为特点，难以解释和计算效率低下。此外，这两种方法都不适合于特定分位数的子集选择。相反，我们从贝叶斯决策分析的角度出发，提出了线性分位估计、不确定性量化和子集选择的基本问题。对于任何贝叶斯回归模型，我们为每个基于模型的条件分位数推导出最佳和可解释的线性估计值和不确定性量化。我们的方法引入了一种分位数聚焦的方法。

    Quantile regression is a powerful tool for inferring how covariates affect specific percentiles of the response distribution. Existing methods either estimate conditional quantiles separately for each quantile of interest or estimate the entire conditional distribution using semi- or non-parametric models. The former often produce inadequate models for real data and do not share information across quantiles, while the latter are characterized by complex and constrained models that can be difficult to interpret and computationally inefficient. Further, neither approach is well-suited for quantile-specific subset selection. Instead, we pose the fundamental problems of linear quantile estimation, uncertainty quantification, and subset selection from a Bayesian decision analysis perspective. For any Bayesian regression model, we derive optimal and interpretable linear estimates and uncertainty quantification for each model-based conditional quantile. Our approach introduces a quantile-focu
    
[^5]: MLP-Mixer作为宽且稀疏的MLP

    MLP-Mixer as a Wide and Sparse MLP. (arXiv:2306.01470v1 [cs.LG])

    [http://arxiv.org/abs/2306.01470](http://arxiv.org/abs/2306.01470)

    深度学习中常用的MLP有潜力提高性能。本研究揭示MLP-Mixer 可以作为具有稀疏权重的宽MLP有效地工作。

    

    多层感知器(MLP)是深度学习中被广泛应用于多种问题的基础组件。然而，最近基于MLP的架构(特别是MLP-Mixer)的实证成功表明，提高MLP的性能仍具有潜在的潜力。在本研究中，我们发现MLP-Mixer有效地作为具有某些稀疏权重的宽MLP。最初，我们澄清Mixer的混合层可以作为具有稀疏权重且由Kronecker乘积表示的更宽MLP的有效表达。该表达式自然地定义了一组置换-Kronecker(PK)家族，可以被视为混合层的一般类，也可以被视为Monarch矩阵的一种近似。随后，由于PK家族有效构成具有稀疏权重的宽MLP，因此，可以应用Golubeva、Neyshabur和Gur-Ari(2021)提出的假设，即预测性能：

    Multi-layer perceptron (MLP) is a fundamental component of deep learning that has been extensively employed for various problems. However, recent empirical successes in MLP-based architectures, particularly the progress of the MLP-Mixer, have revealed that there is still hidden potential in improving MLPs to achieve better performance. In this study, we reveal that the MLP-Mixer works effectively as a wide MLP with certain sparse weights. Initially, we clarify that the mixing layer of the Mixer has an effective expression as a wider MLP whose weights are sparse and represented by the Kronecker product. This expression naturally defines a permuted-Kronecker (PK) family, which can be regarded as a general class of mixing layers and is also regarded as an approximation of Monarch matrices. Subsequently, because the PK family effectively constitutes a wide MLP with sparse weights, one can apply the hypothesis proposed by Golubeva, Neyshabur and Gur-Ari (2021) that the prediction performanc
    
[^6]: 假设简约的变量重要性检验的投影协方差测量

    The Projected Covariance Measure for assumption-lean variable significance testing. (arXiv:2211.02039v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2211.02039](http://arxiv.org/abs/2211.02039)

    该论文介绍了一种假设简约的变量重要性检验方法，利用非参数或机器学习方法实现稳健的误差控制和高功效。

    

    在统计学中，对于给定附加协变量Z，测试变量或变量组X对于预测响应Y的重要性是一项普遍任务。一种简单但常见的方法是指定一个线性模型，然后测试X的回归系数是否为非零。然而，当模型错误指定时，测试的功效可能很差，例如当X参与复杂的交互作用，或者导致许多错误拒绝。在这项工作中，我们研究了测试条件均值独立的无模型假设，即给定X和Z，Y的条件均值不依赖于X。我们提出了一个简单而通用的框架，可以利用灵活的非参数或机器学习方法，如加法模型或随机森林，实现稳健的误差控制和高功效。该过程包括使用这些方法进行回归，首先使用一半的数据估计以X和Z为基础的Y的一种投影形式。

    Testing the significance of a variable or group of variables $X$ for predicting a response $Y$, given additional covariates $Z$, is a ubiquitous task in statistics. A simple but common approach is to specify a linear model, and then test whether the regression coefficient for $X$ is non-zero. However, when the model is misspecified, the test may have poor power, for example when $X$ is involved in complex interactions, or lead to many false rejections. In this work we study the problem of testing the model-free null of conditional mean independence, i.e. that the conditional mean of $Y$ given $X$ and $Z$ does not depend on $X$. We propose a simple and general framework that can leverage flexible nonparametric or machine learning methods, such as additive models or random forests, to yield both robust error control and high power. The procedure involves using these methods to perform regressions, first to estimate a form of projection of $Y$ on $X$ and $Z$ using one half of the data, an
    

