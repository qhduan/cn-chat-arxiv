# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit](https://arxiv.org/abs/2402.06388) | 该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。 |
| [^2] | [Sample Path Regularity of Gaussian Processes from the Covariance Kernel](https://arxiv.org/abs/2312.14886) | 本文提供了关于高斯过程样本路径正则性的新颖和紧凑的特征描述，通过协方差核对应的GP样本路径达到一定正则性的充分必要条件，对常用于机器学习应用中的GPs的样本路径正则性进行了探讨。 |
| [^3] | [MFAI: A Scalable Bayesian Matrix Factorization Approach to Leveraging Auxiliary Information](https://arxiv.org/abs/2303.02566) | MFAI是一种可扩展的贝叶斯矩阵分解方法，通过利用辅助信息来克服由于数据质量差导致的挑战，具有灵活建模非线性关系和对辅助信息的鲁棒性。 |
| [^4] | [A new high-resolution indoor radon map for Germany using a machine learning based probabilistic exposure model.](http://arxiv.org/abs/2310.11143) | 本研究提出了一种基于机器学习的概率暴露模型，可以更准确地估计德国室内氡气分布，并具有更高的空间分辨率。 |

# 详细

[^1]: 关于随机梯度下降（SGD）的收敛速度及其在修改的多臂赌博机上的策略梯度应用

    On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit

    [https://arxiv.org/abs/2402.06388](https://arxiv.org/abs/2402.06388)

    该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。

    

    我们提出了一个自包含的证明，证明了当学习速率遵循逆时间衰减规则时，随机梯度下降（SGD）的收敛速度；接下来，我们将这些结果应用于带有L2正则化的修改的策略梯度多臂赌博机（MAB）的收敛性分析。

    We present a self-contained proof of the convergence rate of the Stochastic Gradient Descent (SGD) when the learning rate follows an inverse time decays schedule; we next apply the results to the convergence of a modified form of policy gradient Multi-Armed Bandit (MAB) with $L2$ regularization.
    
[^2]: 来自协方差核的高斯过程样本路径正则性

    Sample Path Regularity of Gaussian Processes from the Covariance Kernel

    [https://arxiv.org/abs/2312.14886](https://arxiv.org/abs/2312.14886)

    本文提供了关于高斯过程样本路径正则性的新颖和紧凑的特征描述，通过协方差核对应的GP样本路径达到一定正则性的充分必要条件，对常用于机器学习应用中的GPs的样本路径正则性进行了探讨。

    

    高斯过程（GPs）是定义函数空间上的概率分布的最常见形式主义。尽管GPs的应用广泛，但对于GP样本路径的全面理解，即它们定义概率测度的函数空间，尚缺乏。在实践中，GPs不是通过概率测度构建的，而是通过均值函数和协方差核构建的。本文针对协方差核提供了GP样本路径达到给定正则性所需的充分必要条件。我们使用H\"older正则性框架，因为它提供了特别简单的条件，在平稳和各向同性GPs的情况下进一步简化。然后，我们证明我们的结果允许对机器学习应用中常用的GPs的样本路径正则性进行新颖且异常紧凑的表征。

    arXiv:2312.14886v2 Announce Type: replace  Abstract: Gaussian processes (GPs) are the most common formalism for defining probability distributions over spaces of functions. While applications of GPs are myriad, a comprehensive understanding of GP sample paths, i.e. the function spaces over which they define a probability measure, is lacking. In practice, GPs are not constructed through a probability measure, but instead through a mean function and a covariance kernel. In this paper we provide necessary and sufficient conditions on the covariance kernel for the sample paths of the corresponding GP to attain a given regularity. We use the framework of H\"older regularity as it grants particularly straightforward conditions, which simplify further in the cases of stationary and isotropic GPs. We then demonstrate that our results allow for novel and unusually tight characterisations of the sample path regularities of the GPs commonly used in machine learning applications, such as the Mat\'
    
[^3]: MFAI:一种可扩展的贝叶斯矩阵分解方法来利用辅助信息

    MFAI: A Scalable Bayesian Matrix Factorization Approach to Leveraging Auxiliary Information

    [https://arxiv.org/abs/2303.02566](https://arxiv.org/abs/2303.02566)

    MFAI是一种可扩展的贝叶斯矩阵分解方法，通过利用辅助信息来克服由于数据质量差导致的挑战，具有灵活建模非线性关系和对辅助信息的鲁棒性。

    

    在各种实际情况下，矩阵分解方法在数据质量差的情况下往往表现不佳，例如数据稀疏性高和信噪比低。在这里，我们考虑利用辅助信息的矩阵分解问题，辅助信息在实际应用中是大量可用的，以克服由于数据质量差引起的挑战。与现有方法主要依赖于简单线性模型将辅助信息与主数据矩阵结合不同，我们提出将梯度增强树集成到概率矩阵分解框架中以有效地利用辅助信息(MFAI)。因此，MFAI自然地继承了梯度增强树的几个显著特点，如灵活建模非线性关系、对辅助信息中的不相关特征和缺失值具有鲁棒性。MFAI中的参数可以在经验贝叶斯框架下自动确定，使其适应于利用辅助信息。

    In various practical situations, matrix factorization methods suffer from poor data quality, such as high data sparsity and low signal-to-noise ratio (SNR). Here, we consider a matrix factorization problem by utilizing auxiliary information, which is massively available in real-world applications, to overcome the challenges caused by poor data quality. Unlike existing methods that mainly rely on simple linear models to combine auxiliary information with the main data matrix, we propose to integrate gradient boosted trees in the probabilistic matrix factorization framework to effectively leverage auxiliary information (MFAI). Thus, MFAI naturally inherits several salient features of gradient boosted trees, such as the capability of flexibly modeling nonlinear relationships and robustness to irrelevant features and missing values in auxiliary information. The parameters in MFAI can be automatically determined under the empirical Bayes framework, making it adaptive to the utilization of a
    
[^4]: 一种基于机器学习的概率暴露模型的德国高分辨率室内氡气地图

    A new high-resolution indoor radon map for Germany using a machine learning based probabilistic exposure model. (arXiv:2310.11143v1 [stat.ML])

    [http://arxiv.org/abs/2310.11143](http://arxiv.org/abs/2310.11143)

    本研究提出了一种基于机器学习的概率暴露模型，可以更准确地估计德国室内氡气分布，并具有更高的空间分辨率。

    

    室内氡气是一种致癌的放射性气体，可以在室内积累。通常情况下，全国范围内的室内氡暴露是基于广泛的测量活动估计得来的。然而，样本的特征往往与人口特征不同，这是由于许多相关因素，如地质源氡气的可用性或楼层水平。此外，样本大小通常不允许以高空间分辨率进行暴露估计。我们提出了一种基于模型的方法，可以比纯数据方法更加现实地估计室内氡分布，并具有更高的空间分辨率。我们采用了两阶段建模方法：1）应用分位数回归森林，使用环境和建筑数据作为预测因子，估计了德国每个住宅楼的每个楼层的室内氡概率分布函数；2）使用概率蒙特卡罗抽样技术使它们组合和。

    Radon is a carcinogenic, radioactive gas that can accumulate indoors. Indoor radon exposure at the national scale is usually estimated on the basis of extensive measurement campaigns. However, characteristics of the sample often differ from the characteristics of the population due to the large number of relevant factors such as the availability of geogenic radon or floor level. Furthermore, the sample size usually does not allow exposure estimation with high spatial resolution. We propose a model-based approach that allows a more realistic estimation of indoor radon distribution with a higher spatial resolution than a purely data-based approach. We applied a two-stage modelling approach: 1) a quantile regression forest using environmental and building data as predictors was applied to estimate the probability distribution function of indoor radon for each floor level of each residential building in Germany; (2) a probabilistic Monte Carlo sampling technique enabled the combination and
    

