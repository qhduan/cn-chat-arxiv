# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multiclass Learning from Noisy Labels for Non-decomposable Performance Measures](https://rss.arxiv.org/abs/2402.01055) | 本论文提出了用于从带有噪声标签的数据中学习非可分解性能度量的多类学习算法。这些算法分别适用于单调凸性和线性比率两类性能度量，并基于类条件噪声模型进行噪声校正。 |
| [^2] | [Robust Learning of Noisy Time Series Collections Using Stochastic Process Models with Motion Codes](https://arxiv.org/abs/2402.14081) | 使用具有学习谱核的混合高斯过程的潜变量模型方法，针对嘈杂时间序列数据进行鲁棒学习。 |
| [^3] | [Free-form Flows: Make Any Architecture a Normalizing Flow.](http://arxiv.org/abs/2310.16624) | 本文提出了一种训练过程，通过使用变量转换公式梯度的高效估计器，克服了归一化流设计在解析逆变换方面的限制。这使得任何保持维度的神经网络都可以作为生成模型进行最大似然训练，并在分子生成和反问题基准测试中取得优秀的结果。 |
| [^4] | [Anytime-valid t-tests and confidence sequences for Gaussian means with unknown variance.](http://arxiv.org/abs/2310.03722) | 本文提出了两种新的“e-process”和置信序列方法，分别通过替换Lai的混合方法，并分析了所得结果的宽度。 |
| [^5] | [Multi-study R-learner for Heterogeneous Treatment Effect Estimation.](http://arxiv.org/abs/2306.01086) | 本文提出了一种名为多研究R-learner的方法，能够很好地估计多研究中的异质性处理效应，具有鲁棒性，并在现实癌症数据实验中表现出更小的估计误差。 |

# 详细

[^1]: 从有噪声标签学习非可分解性能度量的多类学习

    Multiclass Learning from Noisy Labels for Non-decomposable Performance Measures

    [https://rss.arxiv.org/abs/2402.01055](https://rss.arxiv.org/abs/2402.01055)

    本论文提出了用于从带有噪声标签的数据中学习非可分解性能度量的多类学习算法。这些算法分别适用于单调凸性和线性比率两类性能度量，并基于类条件噪声模型进行噪声校正。

    

    近年来，学习从带有噪声标签的数据中得到良好分类器引起了广泛关注。大多数关于从有噪声标签学习的工作都集中在标准的基于损失的性能度量上。然而，许多机器学习问题需要使用非可分解性能度量，这些度量不能表示为单个示例上的损失的期望或总和；其中包括类不平衡设置中的H-mean，Q-mean和G-mean，以及信息检索中的Micro F1。在本文中，我们设计了算法，用于学习两类广泛的多类非可分解性能度量，即单调凸性和线性比率，它们包括上述所有示例。我们的工作基于Narasimhan等人的Frank-Wolfe和Bisection算法(2015)。在这两种情况下，我们在广泛研究的类条件噪声模型家族下开发了算法的噪声校正版本。我们提供了遗憾(超额风险)上界。

    There has been much interest in recent years in learning good classifiers from data with noisy labels. Most work on learning from noisy labels has focused on standard loss-based performance measures. However, many machine learning problems require using non-decomposable performance measures which cannot be expressed as the expectation or sum of a loss on individual examples; these include for example the H-mean, Q-mean and G-mean in class imbalance settings, and the Micro $F_1$ in information retrieval. In this paper, we design algorithms to learn from noisy labels for two broad classes of multiclass non-decomposable performance measures, namely, monotonic convex and ratio-of-linear, which encompass all the above examples. Our work builds on the Frank-Wolfe and Bisection based methods of Narasimhan et al. (2015). In both cases, we develop noise-corrected versions of the algorithms under the widely studied family of class-conditional noise models. We provide regret (excess risk) bounds 
    
[^2]: 使用具有运动代码的随机过程模型对嘈杂时间序列集合进行鲁棒学习

    Robust Learning of Noisy Time Series Collections Using Stochastic Process Models with Motion Codes

    [https://arxiv.org/abs/2402.14081](https://arxiv.org/abs/2402.14081)

    使用具有学习谱核的混合高斯过程的潜变量模型方法，针对嘈杂时间序列数据进行鲁棒学习。

    

    虽然时间序列分类和预测问题已经得到广泛研究，但具有任意时间序列长度的嘈杂时间序列数据的情况仍具挑战性。每个时间序列实例可以看作是嘈杂动态模型的一个样本实现，其特点是连续随机过程。对于许多应用，数据是混合的，由多个随机过程建模的几种类型的嘈杂时间序列序列组成，使得预测和分类任务变得更具挑战性。我们不是简单地将数据回归到每种时间序列类型，而是采用具有学习谱核的混合高斯过程的潜变量模型方法。更具体地说，我们为每种类型的嘈杂时间序列数据自动分配一个称为其运动代码的签名向量。然后，在每个分配的运动代码的条件下，我们推断出相关性的稀疏近似。

    arXiv:2402.14081v1 Announce Type: cross  Abstract: While time series classification and forecasting problems have been extensively studied, the cases of noisy time series data with arbitrary time sequence lengths have remained challenging. Each time series instance can be thought of as a sample realization of a noisy dynamical model, which is characterized by a continuous stochastic process. For many applications, the data are mixed and consist of several types of noisy time series sequences modeled by multiple stochastic processes, making the forecasting and classification tasks even more challenging. Instead of regressing data naively and individually to each time series type, we take a latent variable model approach using a mixtured Gaussian processes with learned spectral kernels. More specifically, we auto-assign each type of noisy time series data a signature vector called its motion code. Then, conditioned on each assigned motion code, we infer a sparse approximation of the corr
    
[^3]: 自由形式流动：使任何架构成为归一化流

    Free-form Flows: Make Any Architecture a Normalizing Flow. (arXiv:2310.16624v1 [cs.LG])

    [http://arxiv.org/abs/2310.16624](http://arxiv.org/abs/2310.16624)

    本文提出了一种训练过程，通过使用变量转换公式梯度的高效估计器，克服了归一化流设计在解析逆变换方面的限制。这使得任何保持维度的神经网络都可以作为生成模型进行最大似然训练，并在分子生成和反问题基准测试中取得优秀的结果。

    

    归一化流是直接最大化可能性的生成模型。以前，归一化流的设计在很大程度上受到对解析逆变换的需要限制。通过使用对变量转换公式的梯度的高效估计器进行训练，我们克服了这个限制。这使得任何保持维度的神经网络都可以通过最大似然训练作为生成模型。我们的方法允许将重点放在精确调整归纳偏见以适应手头的任务上。具体而言，我们在分子生成基准测试中利用$E(n)$-等变网络取得了出色的结果。此外，我们的方法在一个反问题基准测试中也具有竞争力，同时采用现成的ResNet架构。

    Normalizing Flows are generative models that directly maximize the likelihood. Previously, the design of normalizing flows was largely constrained by the need for analytical invertibility. We overcome this constraint by a training procedure that uses an efficient estimator for the gradient of the change of variables formula. This enables any dimension-preserving neural network to serve as a generative model through maximum likelihood training. Our approach allows placing the emphasis on tailoring inductive biases precisely to the task at hand. Specifically, we achieve excellent results in molecule generation benchmarks utilizing $E(n)$-equivariant networks. Moreover, our method is competitive in an inverse problem benchmark, while employing off-the-shelf ResNet architectures.
    
[^4]: 未知方差下的高斯均值的任意有效T检验和置信序列

    Anytime-valid t-tests and confidence sequences for Gaussian means with unknown variance. (arXiv:2310.03722v1 [math.ST])

    [http://arxiv.org/abs/2310.03722](http://arxiv.org/abs/2310.03722)

    本文提出了两种新的“e-process”和置信序列方法，分别通过替换Lai的混合方法，并分析了所得结果的宽度。

    

    在1976年，Lai构造了一个非平凡的均值$\mu$的高斯分布的置信序列，该分布的方差$\sigma$是未知的。他使用了关于$\sigma$的不适当（右Haar）混合和关于$\mu$的不适当（平坦）混合。在本文中，我们详细说明了他构建的细节，其中使用了广义的不可积分鞅和扩展的维尔不等式。尽管这确实产生了一个顺序T检验，但由于他的鞅不可积分，它并没有产生一个“e-process”。在本文中，我们为相同的设置开发了两个新的“e-process”和置信序列：一个是在缩减滤波器中的测试鞅，另一个是在规范数据滤波器中的“e-process”。这些分别是通过将Lai的平坦混合替换为高斯混合，并将对$\sigma$的右Haar混合替换为在零空间下的最大似然估计，就像在通用推断中一样。我们还分析了所得结果的宽度。

    In 1976, Lai constructed a nontrivial confidence sequence for the mean $\mu$ of a Gaussian distribution with unknown variance $\sigma$. Curiously, he employed both an improper (right Haar) mixture over $\sigma$ and an improper (flat) mixture over $\mu$. Here, we elaborate carefully on the details of his construction, which use generalized nonintegrable martingales and an extended Ville's inequality. While this does yield a sequential t-test, it does not yield an ``e-process'' (due to the nonintegrability of his martingale). In this paper, we develop two new e-processes and confidence sequences for the same setting: one is a test martingale in a reduced filtration, while the other is an e-process in the canonical data filtration. These are respectively obtained by swapping Lai's flat mixture for a Gaussian mixture, and swapping the right Haar mixture over $\sigma$ with the maximum likelihood estimate under the null, as done in universal inference. We also analyze the width of resulting 
    
[^5]: 多研究R-learner用于异质性处理效应估计

    Multi-study R-learner for Heterogeneous Treatment Effect Estimation. (arXiv:2306.01086v1 [stat.ME])

    [http://arxiv.org/abs/2306.01086](http://arxiv.org/abs/2306.01086)

    本文提出了一种名为多研究R-learner的方法，能够很好地估计多研究中的异质性处理效应，具有鲁棒性，并在现实癌症数据实验中表现出更小的估计误差。

    

    我们提出了一种通用的算法类来估计多个研究中的异质性处理效应。我们的方法称为多研究R-learner，可以概括R-learner以考虑研究间的异质性并实现调整混淆的跨研究鲁棒性。多研究R-learner能够灵活地融合许多机器学习技术以估计异质性处理效应、困扰函数和成员概率。我们表明，多研究R-learner处理效应估计器在序列估计框架内是渐近正常的。此外，我们通过现实癌症数据实验证明，随着研究间的异质性增加，与R-learner相比，我们的方法估计误差更小。

    We propose a general class of algorithms for estimating heterogeneous treatment effects on multiple studies. Our approach, called the multi-study R-learner, generalizes the R-learner to account for between-study heterogeneity and achieves cross-study robustness of confounding adjustment. The multi-study R-learner is flexible in its ability to incorporate many machine learning techniques for estimating heterogeneous treatment effects, nuisance functions, and membership probabilities. We show that the multi-study R-learner treatment effect estimator is asymptotically normal within the series estimation framework. Moreover, we illustrate via realistic cancer data experiments that our approach results in lower estimation error than the R-learner as between-study heterogeneity increases.
    

