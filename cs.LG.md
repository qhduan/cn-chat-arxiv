# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multiple-Input Auto-Encoder Guided Feature Selection for IoT Intrusion Detection Systems](https://arxiv.org/abs/2403.15511) | 该论文提出了一种名为多输入自动编码器(MIAE)的新型神经网络架构，通过训练MIAE模型，在无监督学习模式下将异构输入转换为较低维表示，有助于分类器区分正常行为和不同类型的攻击。 |
| [^2] | [MGAS: Multi-Granularity Architecture Search for Effective and Efficient Neural Networks.](http://arxiv.org/abs/2310.15074) | MGAS是一个多粒度架构搜索的统一框架，通过学习特定粒度级别的离散化函数，自适应地确定剩余比例，从而实现同时优化模型大小和模型性能。 |
| [^3] | [Fast, Sample-Efficient, Affine-Invariant Private Mean and Covariance Estimation for Subgaussian Distributions.](http://arxiv.org/abs/2301.12250) | 本论文提出了一种快速的差分私有算法，用于具有几乎最优样本复杂度的高维协方差感知均值估计。在Mahalanobis误差度量中，也就是相对于协方差的平均误差中，我们的算法使得$\hat \mu$更接近$\mu$。 |

# 详细

[^1]: IoT入侵检测系统中的多输入自动编码器引导特征选择

    Multiple-Input Auto-Encoder Guided Feature Selection for IoT Intrusion Detection Systems

    [https://arxiv.org/abs/2403.15511](https://arxiv.org/abs/2403.15511)

    该论文提出了一种名为多输入自动编码器(MIAE)的新型神经网络架构，通过训练MIAE模型，在无监督学习模式下将异构输入转换为较低维表示，有助于分类器区分正常行为和不同类型的攻击。

    

    入侵检测系统(IDSs)受益于IoT数据特征的多样性和泛化，数据的多样性使得在IoT IDSs中训练有效的机器学习模型变得困难。本文首先介绍了一种名为多输入自动编码器(MIAE)的新型神经网络架构。MIAE由多个子编码器组成，可以处理具有不同特征的不同来源的输入。 MIAE模型以无监督学习模式进行训练，将异构输入转换为较低维表示，有助于分类器区分正常行为和不同类型的攻击。

    arXiv:2403.15511v1 Announce Type: cross  Abstract: While intrusion detection systems (IDSs) benefit from the diversity and generalization of IoT data features, the data diversity (e.g., the heterogeneity and high dimensions of data) also makes it difficult to train effective machine learning models in IoT IDSs. This also leads to potentially redundant/noisy features that may decrease the accuracy of the detection engine in IDSs. This paper first introduces a novel neural network architecture called Multiple-Input Auto-Encoder (MIAE). MIAE consists of multiple sub-encoders that can process inputs from different sources with different characteristics. The MIAE model is trained in an unsupervised learning mode to transform the heterogeneous inputs into lower-dimensional representation, which helps classifiers distinguish between normal behaviour and different types of attacks. To distil and retain more relevant features but remove less important/redundant ones during the training process,
    
[^2]: MGAS: 多粒度架构搜索以实现高效且有效的神经网络

    MGAS: Multi-Granularity Architecture Search for Effective and Efficient Neural Networks. (arXiv:2310.15074v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.15074](http://arxiv.org/abs/2310.15074)

    MGAS是一个多粒度架构搜索的统一框架，通过学习特定粒度级别的离散化函数，自适应地确定剩余比例，从而实现同时优化模型大小和模型性能。

    

    可微分架构搜索(DAS)通过时间高效的自动化改变了神经网络架构搜索(NAS)的方式，从离散候选采样和评估转变为可微分超网络优化和离散化。然而，现有的DAS方法要么只进行粗粒度的操作级搜索，要么手动定义剩余的细粒度的核级和权重级单位的比例，从而无法同时优化模型大小和模型性能。此外，这些方法为了减少内存消耗而牺牲了搜索质量。为了解决这些问题，我们引入了多粒度架构搜索(MGAS)，这是一个统一的框架，旨在全面而内存高效地探索多粒度搜索空间，发现既有效又高效的神经网络。具体来说，我们学习了针对每个粒度级别的离散化函数，根据不断演化的架构自适应地确定剩余的比例。

    Differentiable architecture search (DAS) revolutionizes neural architecture search (NAS) with time-efficient automation, transitioning from discrete candidate sampling and evaluation to differentiable super-net optimization and discretization. However, existing DAS methods either only conduct coarse-grained operation-level search or manually define the remaining ratios for fine-grained kernel-level and weight-level units, which fail to simultaneously optimize model size and model performance. Furthermore, these methods compromise search quality to reduce memory consumption. To tackle these issues, we introduce multi-granularity architecture search (MGAS), a unified framework which aims to comprehensively and memory-efficiently explore the multi-granularity search space to discover both effective and efficient neural networks. Specifically, we learn discretization functions specific to each granularity level to adaptively determine the remaining ratios according to the evolving architec
    
[^3]: 高维次高斯分布的快速，样本有效，仿射不变私有均值和协方差估计

    Fast, Sample-Efficient, Affine-Invariant Private Mean and Covariance Estimation for Subgaussian Distributions. (arXiv:2301.12250v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.12250](http://arxiv.org/abs/2301.12250)

    本论文提出了一种快速的差分私有算法，用于具有几乎最优样本复杂度的高维协方差感知均值估计。在Mahalanobis误差度量中，也就是相对于协方差的平均误差中，我们的算法使得$\hat \mu$更接近$\mu$。

    

    我们提出了一种快速的差分私有算法，用于具有几乎最优样本复杂度的高维协方差感知均值估计。以前已知只有指数时间估计器才能实现此保证。给定从具有未知均值 $μ$ 和协方差 $Σ$ 的（亚）高斯分布中抽取的$n$个样本，我们的 $(\varepsilon,\delta)$-差分式私有估计器生成$\tilde{\mu}$，使得只要 $n \gtrsim \tfrac d {\alpha^2} + \tfrac{d \sqrt{\log 1/\delta}}{\alpha \varepsilon}+\frac{d\log 1/\delta}{\varepsilon}$，就满足 $\|\mu - \tilde{\mu}\|_{\Sigma} \leq \alpha$。Mahalanobis误差度量 $\|\mu - \hat{\mu}\|_{\Sigma}$ 衡量了$\hat \mu$与$\mu$在$\Sigma$相对距离; 它表征了样本平均值的误差。我们的算法运行时间为$\tilde{O}(nd^{\omega - 1} + nd/\varepsilon)$，其中$\omega < 2.38$是矩阵乘法指数。我们使用 Brown、Gaboardi、Smith、Ullman 和 Zakynthiadaki[BGSUZ18] 的指数时间方法来计算问题的最优估计的足够统计量，并将其用于通过随机线性代数构造线性时间估计器。

    We present a fast, differentially private algorithm for high-dimensional covariance-aware mean estimation with nearly optimal sample complexity. Only exponential-time estimators were previously known to achieve this guarantee. Given $n$ samples from a (sub-)Gaussian distribution with unknown mean $\mu$ and covariance $\Sigma$, our $(\varepsilon,\delta)$-differentially private estimator produces $\tilde{\mu}$ such that $\|\mu - \tilde{\mu}\|_{\Sigma} \leq \alpha$ as long as $n \gtrsim \tfrac d {\alpha^2} + \tfrac{d \sqrt{\log 1/\delta}}{\alpha \varepsilon}+\frac{d\log 1/\delta}{\varepsilon}$. The Mahalanobis error metric $\|\mu - \hat{\mu}\|_{\Sigma}$ measures the distance between $\hat \mu$ and $\mu$ relative to $\Sigma$; it characterizes the error of the sample mean. Our algorithm runs in time $\tilde{O}(nd^{\omega - 1} + nd/\varepsilon)$, where $\omega < 2.38$ is the matrix multiplication exponent.  We adapt an exponential-time approach of Brown, Gaboardi, Smith, Ullman, and Zakynthi
    

