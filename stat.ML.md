# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Voronoi Candidates for Bayesian Optimization](https://arxiv.org/abs/2402.04922) | 使用Voronoi候选点边界可以在贝叶斯优化中有效地优化黑盒函数，提高了多起始连续搜索的执行时间。 |
| [^2] | [Iterative Methods for Vecchia-Laplace Approximations for Latent Gaussian Process Models.](http://arxiv.org/abs/2310.12000) | 这篇文章介绍了用于潜在高斯过程模型中的Vecchia-Laplace近似法的迭代方法，相比于传统的Cholesky分解方法，可以显著加快计算速度。 |
| [^3] | [Memorization with neural nets: going beyond the worst case.](http://arxiv.org/abs/2310.00327) | 本文研究了神经网络的插值问题，提出了一种简单的随机算法，在给定的数据集和两个类的情况下，能够以很高的概率构建一个插值的神经网络。这些结果与训练数据规模无关。 |
| [^4] | [The Score-Difference Flow for Implicit Generative Modeling.](http://arxiv.org/abs/2304.12906) | 本文提出了一种新的评分差异流模型(SD flow)，它可以最优地减少两个分布之间的散度，同时解决Schr​​ödinger桥问题。与去噪扩散模型不同，它没有对先验分布施加任何限制，在一些基准数据集中优于其他方法。 |

# 详细

[^1]: Voronoi Candidates用于贝叶斯优化

    Voronoi Candidates for Bayesian Optimization

    [https://arxiv.org/abs/2402.04922](https://arxiv.org/abs/2402.04922)

    使用Voronoi候选点边界可以在贝叶斯优化中有效地优化黑盒函数，提高了多起始连续搜索的执行时间。

    

    贝叶斯优化（BO）为高效优化黑盒函数提供了一种优雅的方法。然而，采集准则需要进行具有挑战性的内部优化，这可能引起很大的开销。许多实际的BO方法，尤其是在高维情况下，不采用对采集函数进行形式化连续优化，而是在有限的空间填充候选集上进行离散搜索。在这里，我们提议使用候选点，其位于当前设计点的Voronoi镶嵌边界上，因此它们与两个或多个设计点等距离。我们讨论了通过直接采样Voronoi边界而不明确生成镶嵌的策略，从而适应高维度中的大设计。通过使用高斯过程和期望改进来对一组测试问题进行优化，我们的方法在不损失准确性的情况下显著提高了多起始连续搜索的执行时间。

    Bayesian optimization (BO) offers an elegant approach for efficiently optimizing black-box functions. However, acquisition criteria demand their own challenging inner-optimization, which can induce significant overhead. Many practical BO methods, particularly in high dimension, eschew a formal, continuous optimization of the acquisition function and instead search discretely over a finite set of space-filling candidates. Here, we propose to use candidates which lie on the boundary of the Voronoi tessellation of the current design points, so they are equidistant to two or more of them. We discuss strategies for efficient implementation by directly sampling the Voronoi boundary without explicitly generating the tessellation, thus accommodating large designs in high dimension. On a battery of test problems optimized via Gaussian processes with expected improvement, our proposed approach significantly improves the execution time of a multi-start continuous search without a loss in accuracy
    
[^2]: Vecchia-Laplace近似法在潜在高斯过程模型中的迭代方法

    Iterative Methods for Vecchia-Laplace Approximations for Latent Gaussian Process Models. (arXiv:2310.12000v1 [stat.ME])

    [http://arxiv.org/abs/2310.12000](http://arxiv.org/abs/2310.12000)

    这篇文章介绍了用于潜在高斯过程模型中的Vecchia-Laplace近似法的迭代方法，相比于传统的Cholesky分解方法，可以显著加快计算速度。

    

    潜在高斯过程（GP）模型是灵活的概率非参数函数模型。Vecchia近似是用于克服大数据计算瓶颈的准确近似方法，Laplace近似是一种快速方法，可以近似非高斯似然函数的边缘似然和后验预测分布，并具有渐近收敛保证。然而，当与直接求解方法（如Cholesky分解）结合使用时，Vecchia-Laplace近似的计算复杂度增长超线性地随样本大小增加。因此，与Vecchia-Laplace近似计算相关的运算在通常情况下是最准确的大型数据集时会变得非常缓慢。在本文中，我们提出了几种用于Vecchia-Laplace近似推断的迭代方法，相比于基于Cholesky的计算，可以大大加快计算速度。我们对我们的方法进行了分析。

    Latent Gaussian process (GP) models are flexible probabilistic non-parametric function models. Vecchia approximations are accurate approximations for GPs to overcome computational bottlenecks for large data, and the Laplace approximation is a fast method with asymptotic convergence guarantees to approximate marginal likelihoods and posterior predictive distributions for non-Gaussian likelihoods. Unfortunately, the computational complexity of combined Vecchia-Laplace approximations grows faster than linearly in the sample size when used in combination with direct solver methods such as the Cholesky decomposition. Computations with Vecchia-Laplace approximations thus become prohibitively slow precisely when the approximations are usually the most accurate, i.e., on large data sets. In this article, we present several iterative methods for inference with Vecchia-Laplace approximations which make computations considerably faster compared to Cholesky-based calculations. We analyze our propo
    
[^3]: 神经网络的记忆化：超越最坏情况

    Memorization with neural nets: going beyond the worst case. (arXiv:2310.00327v1 [stat.ML])

    [http://arxiv.org/abs/2310.00327](http://arxiv.org/abs/2310.00327)

    本文研究了神经网络的插值问题，提出了一种简单的随机算法，在给定的数据集和两个类的情况下，能够以很高的概率构建一个插值的神经网络。这些结果与训练数据规模无关。

    

    在实践中，深度神经网络通常能够轻松地插值其训练数据。为了理解这一现象，许多研究都旨在量化神经网络架构的记忆能力：即在任意放置这些点并任意分配标签的情况下，架构能够插值的最大点数。然而，对于实际数据，人们直觉地期望存在一种良性结构，使得插值在比记忆能力建议的较小网络尺寸上已经发生。在本文中，我们通过采用实例特定的观点来研究插值。我们引入了一个简单的随机算法，它可以在多项式时间内给定一个固定的有限数据集和两个类的情况下，以很高的概率构建出一个插值三层神经网络。所需的参数数量与这两个类的几何特性及其相互排列有关。因此，我们获得了与训练数据规模无关的保证。

    In practice, deep neural networks are often able to easily interpolate their training data. To understand this phenomenon, many works have aimed to quantify the memorization capacity of a neural network architecture: the largest number of points such that the architecture can interpolate any placement of these points with any assignment of labels. For real-world data, however, one intuitively expects the presence of a benign structure so that interpolation already occurs at a smaller network size than suggested by memorization capacity. In this paper, we investigate interpolation by adopting an instance-specific viewpoint. We introduce a simple randomized algorithm that, given a fixed finite dataset with two classes, with high probability constructs an interpolating three-layer neural network in polynomial time. The required number of parameters is linked to geometric properties of the two classes and their mutual arrangement. As a result, we obtain guarantees that are independent of t
    
[^4]: 评分差值流模型用于隐式生成建模

    The Score-Difference Flow for Implicit Generative Modeling. (arXiv:2304.12906v1 [cs.LG])

    [http://arxiv.org/abs/2304.12906](http://arxiv.org/abs/2304.12906)

    本文提出了一种新的评分差异流模型(SD flow)，它可以最优地减少两个分布之间的散度，同时解决Schr​​ödinger桥问题。与去噪扩散模型不同，它没有对先验分布施加任何限制，在一些基准数据集中优于其他方法。

    

    隐式生成建模(IGM)旨在生成符合目标数据分布特征的合成数据样本。最近的研究(例如评分匹配网络、扩散模型)从通过环境空间中的动态扰动或流将合成源数据推向目标分布的角度解决了IGM问题。我们引入了任意目标和源分布之间的评分差异(SD)作为流，它可以最优地减少它们之间的Kullback-Leibler散度，同时解决Schr​​ödinger桥问题。我们将SD流应用于方便的代理分布，当且仅当原始分布对齐时，它们是对齐的。我们在某些条件下展示了这种公式与去噪扩散模型的形式一致性。然而，与扩散模型不同，SD流没有对先验分布施加任何限制。我们还表明，在无限辨别器能力的极限下，生成对抗网络的训练包含SD流。我们的实验表明，SD流在几个基准数据集上优于先前的最新技术。

    Implicit generative modeling (IGM) aims to produce samples of synthetic data matching the characteristics of a target data distribution. Recent work (e.g. score-matching networks, diffusion models) has approached the IGM problem from the perspective of pushing synthetic source data toward the target distribution via dynamical perturbations or flows in the ambient space. We introduce the score difference (SD) between arbitrary target and source distributions as a flow that optimally reduces the Kullback-Leibler divergence between them while also solving the Schr\"odinger bridge problem. We apply the SD flow to convenient proxy distributions, which are aligned if and only if the original distributions are aligned. We demonstrate the formal equivalence of this formulation to denoising diffusion models under certain conditions. However, unlike diffusion models, SD flow places no restrictions on the prior distribution. We also show that the training of generative adversarial networks includ
    

