# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Consistency and adaptivity are complementary targets for the validation of variance-based uncertainty quantification metrics in machine learning regression tasks.](http://arxiv.org/abs/2309.06240) | 这篇论文研究了机器学习回归任务中基于方差的不确定性量化度量的验证，发现一致性和适应性是互补的验证目标，并提出了适应性验证方法。 |
| [^2] | [Cliqueful graphs as a means of calculating the maximal number of maximum cliques of simple graphs.](http://arxiv.org/abs/2307.14120) | 本论文研究了充满团图的概念，并且发现在简单图中，充满团图的最大数量取决于饱和复合充满团图。通过具体计算，我们得到了在n个顶点上具有最多最大团数量的图形式表达式。 |
| [^3] | [Universal Online Learning with Gradual Variations: A Multi-layer Online Ensemble Approach.](http://arxiv.org/abs/2307.08360) | 该论文提出了一种具有两个不同级别自适应性的在线凸优化方法，对不同类型的损失函数具有多种遗憾界，并在分析中直接应用于小损失界。同时，它与对抗性/随机凸优化和博弈论有着深刻的联系。 |
| [^4] | [Data-Adaptive Probabilistic Likelihood Approximation for Ordinary Differential Equations.](http://arxiv.org/abs/2306.05566) | 本文提出了一种数据自适应的贝叶斯滤波范式来解决现有概率ODE求解器无法解决的深层的局部最小值问题，结果表明该方法比现有概率ODE求解器更精确。 |
| [^5] | [Classical Verification of Quantum Learning.](http://arxiv.org/abs/2306.04843) | 本文提出了一个经典验证量子学习的框架，以便经典客户委托学习给不可信的量子服务器，并成功解决了对仗学习奇偶性和傅里叶稀疏函数不可信量子证明的问题。 |
| [^6] | [Kernel Quadrature with Randomly Pivoted Cholesky.](http://arxiv.org/abs/2306.03955) | 本文提出了一种新的使用随机选择纯量分解算法的核求积方法，可以在达到可比的求积误差达到率的同时显著降低计算复杂度，并可以应用于任意核的复杂几何结构。 |
| [^7] | [Loss-Optimal Classification Trees: A Generalized Framework and the Logistic Case.](http://arxiv.org/abs/2306.00857) | 本论文为损失最小化分类树的解决提供了一个通用框架，基于逻辑斯蒂回归的解决方式具有具有解释性特征和竞争泛化能力。 |
| [^8] | [Solving Linear Inverse Problems using Higher-Order Annealed Langevin Diffusion.](http://arxiv.org/abs/2305.05014) | 本论文提出了一种高阶 Langevin 动力学的算法，可更高效地采样未知变量后验分布，同时加入淬火过程，能应用于离散未知变量情况，数值实验表明在多个任务中相对竞争方法具有更高性能。 |
| [^9] | [Low-complexity subspace-descent over symmetric positive definite manifold.](http://arxiv.org/abs/2305.02041) | 本文提出了一种基于黎曼子空间下降算法的对称正定流形上的函数最小化方法，其具有低复杂度和避免昂贵矩阵操作和计算黎曼梯度的优点。 |
| [^10] | [LAVA: Data Valuation without Pre-Specified Learning Algorithms.](http://arxiv.org/abs/2305.00054) | LAVA是一个学习算法无关的数据价值评估方法，它结合了学习算法的统计特性和训练数据的属性，通过迭代估计数据值来实现。LAVA比现有方法计算速度更快，精度更高，并且可以为不同的应用提供有意义的数据排名。 |
| [^11] | [Temporal Robustness against Data Poisoning.](http://arxiv.org/abs/2302.03684) | 该论文提出了一种针对数据污染的时序威胁模型，通过利用数据的时间戳，引入了提前时间和持续时间这两个指标，从而定义了数据污染的时序鲁棒性，并提供了一种有效的保护方法。 |
| [^12] | [Neural network based generation of 1-dimensional stochastic fields with turbulent velocity statistics.](http://arxiv.org/abs/2211.11580) | 该论文介绍了一个名为NN-Turb的全卷积神经网络随机模型，用于以湍流速度统计特性生成1维场，而无需接触湍流数据。生成的过程满足Kolmogorov 2/3结构函数定律，同时还表现出负偏斜度和间歇性。 |

# 详细

[^1]: 一致性和适应性是验证机器学习回归任务中基于方差的不确定性量化度量的互补目标

    Consistency and adaptivity are complementary targets for the validation of variance-based uncertainty quantification metrics in machine learning regression tasks. (arXiv:2309.06240v1 [stat.ML])

    [http://arxiv.org/abs/2309.06240](http://arxiv.org/abs/2309.06240)

    这篇论文研究了机器学习回归任务中基于方差的不确定性量化度量的验证，发现一致性和适应性是互补的验证目标，并提出了适应性验证方法。

    

    可靠的不确定性量化是材料和化学科学中许多研究的焦点。目前已经认识到平均校准是不足够的，大多数研究都使用额外的方法来测试条件校准，即一致性。一致性主要通过可靠性图来评估。然而，除了平均校准之外还存在一种方法，即基于输入特征的条件校准，也就是适应性。实际上，适应性是ML-UQ方法的最终用户关注的主要问题，他们寻求对特征空间中的任何点的预测和不确定性的可靠性。本文旨在展示一致性和适应性是互补的验证目标，并且好的一致性并不意味着好的适应性。文章提出并在一个典型示例上进行了适应性验证方法的说明。

    Reliable uncertainty quantification (UQ) in machine learning (ML) regression tasks is becoming the focus of many studies in materials and chemical science. It is now well understood that average calibration is insufficient, and most studies implement additional methods testing the conditional calibration with respect to uncertainty, i.e. consistency. Consistency is assessed mostly by so-called reliability diagrams. There exists however another way beyond average calibration, which is conditional calibration with respect to input features, i.e. adaptivity. In practice, adaptivity is the main concern of the final users of a ML-UQ method, seeking for the reliability of predictions and uncertainties for any point in features space. This article aims to show that consistency and adaptivity are complementary validation targets, and that a good consistency does not imply a good adaptivity. Adapted validation methods are proposed and illustrated on a representative example.
    
[^2]: 作为计算简单图的最大团的最大数量的手段的充满团图

    Cliqueful graphs as a means of calculating the maximal number of maximum cliques of simple graphs. (arXiv:2307.14120v1 [math.CO])

    [http://arxiv.org/abs/2307.14120](http://arxiv.org/abs/2307.14120)

    本论文研究了充满团图的概念，并且发现在简单图中，充满团图的最大数量取决于饱和复合充满团图。通过具体计算，我们得到了在n个顶点上具有最多最大团数量的图形式表达式。

    

    一个简单图在n个顶点上可能包含许多最大团。但它可能包含多少个呢？我们将展示最大团的最大数量取决于所谓的充满团图，具体地说，如果n≥15，我们将展示它取决于饱和复合充满团图。利用这一点，我们将展示包含3^{⌊n/3⌋}c个最大团的图在n个顶点上具有最多的最大团数量，其中c∈{1,4/3,2}，取决于n模3的值。

    A simple graph on $n$ vertices may contain a lot of maximum cliques. But how many can it potentially contain? We will show that the maximum number of maximum cliques is taken over so-called cliqueful graphs, more specifically, later we will show that it is taken over saturated composite cliqueful graphs, if $n \ge 15$. Using this we will show that the graph that contains $3^{\lfloor n/3 \rfloor}c$ maxcliques has the most number of maxcliques on $n$ vertices, where $c\in\{1,\frac{4}{3},2\}$, depending on $n \text{ mod } 3$.
    
[^3]: 具有逐渐变化的通用在线学习：一种多层在线集成方法

    Universal Online Learning with Gradual Variations: A Multi-layer Online Ensemble Approach. (arXiv:2307.08360v1 [cs.LG])

    [http://arxiv.org/abs/2307.08360](http://arxiv.org/abs/2307.08360)

    该论文提出了一种具有两个不同级别自适应性的在线凸优化方法，对不同类型的损失函数具有多种遗憾界，并在分析中直接应用于小损失界。同时，它与对抗性/随机凸优化和博弈论有着深刻的联系。

    

    在本文中，我们提出了一种具有两个不同级别自适应性的在线凸优化方法。在更高级别上，我们的方法对损失函数的具体类型和曲率不知情，而在更低级别上，它可以利用环境的良好性质并获得问题相关保证。具体而言，对于强凸、指数凹和凸损失函数，我们分别获得了$O(\ln V_T)$、$O(d \ln V_T)$和$\hat{O}(\sqrt{V_T})$的遗憾界，其中$d$是维度，$V_T$表示问题相关的梯度变化，$\hat{O}(\cdot)$表示在$V_T$上省略对数因子。我们的结果具有广泛的影响和应用。它不仅保证了最坏情况下的性能，还直接导出了分析中的小损失界。此外，它与对抗性/随机凸优化和博弈论有着深刻的联系，进一步验证了其实际潜力。我们的方法基于...

    In this paper, we propose an online convex optimization method with two different levels of adaptivity. On a higher level, our method is agnostic to the specific type and curvature of the loss functions, while at a lower level, it can exploit the niceness of the environments and attain problem-dependent guarantees. To be specific, we obtain $\mathcal{O}(\ln V_T)$, $\mathcal{O}(d \ln V_T)$ and $\hat{\mathcal{O}}(\sqrt{V_T})$ regret bounds for strongly convex, exp-concave and convex loss functions, respectively, where $d$ is the dimension, $V_T$ denotes problem-dependent gradient variations and $\hat{\mathcal{O}}(\cdot)$-notation omits logarithmic factors on $V_T$. Our result finds broad implications and applications. It not only safeguards the worst-case guarantees, but also implies the small-loss bounds in analysis directly. Besides, it draws deep connections with adversarial/stochastic convex optimization and game theory, further validating its practical potential. Our method is based
    
[^4]: 数据自适应概率似然逼近常微分方程

    Data-Adaptive Probabilistic Likelihood Approximation for Ordinary Differential Equations. (arXiv:2306.05566v1 [stat.ML])

    [http://arxiv.org/abs/2306.05566](http://arxiv.org/abs/2306.05566)

    本文提出了一种数据自适应的贝叶斯滤波范式来解决现有概率ODE求解器无法解决的深层的局部最小值问题，结果表明该方法比现有概率ODE求解器更精确。

    

    常微分方程（ODEs）的参数推断在许多科学应用中具有基本重要性。虽然ODE解通常由确定性算法近似，但有关概率求解器的新研究表明，它们通过更好地考虑数字误差产生更可靠的参数估计。然而，许多ODE系统对其参数值非常敏感。这在似然函数中产生了深层的局部最小值——现有的概率求解器尚未解决这个问题。在这里，我们展示了一种用于概率ODE解的贝叶斯滤波范式，通过数据自适应地学习嘈杂的ODE观察结果，可以显着降低参数的敏感性。我们的方法适用于具有部分未观测分量和任意非高斯噪声的ODEs。几个例子表明，它比现有的概率ODE求解器更精确，甚至在某些情况下比精确ODE似然函数更精确。

    Parameter inference for ordinary differential equations (ODEs) is of fundamental importance in many scientific applications. While ODE solutions are typically approximated by deterministic algorithms, new research on probabilistic solvers indicates that they produce more reliable parameter estimates by better accounting for numerical errors. However, many ODE systems are highly sensitive to their parameter values. This produces deep local minima in the likelihood function -- a problem which existing probabilistic solvers have yet to resolve. Here, we show that a Bayesian filtering paradigm for probabilistic ODE solution can dramatically reduce sensitivity to parameters by learning from the noisy ODE observations in a data-adaptive manner. Our method is applicable to ODEs with partially unobserved components and with arbitrary non-Gaussian noise. Several examples demonstrate that it is more accurate than existing probabilistic ODE solvers, and even in some cases than the exact ODE likel
    
[^5]: 量子学习的经典验证

    Classical Verification of Quantum Learning. (arXiv:2306.04843v1 [quant-ph])

    [http://arxiv.org/abs/2306.04843](http://arxiv.org/abs/2306.04843)

    本文提出了一个经典验证量子学习的框架，以便经典客户委托学习给不可信的量子服务器，并成功解决了对仗学习奇偶性和傅里叶稀疏函数不可信量子证明的问题。

    

    量子数据访问和量子处理可以使某些经典难以处理的学习任务变得可行。然而，量子能力在不久的将来只能提供给少数人使用。因此，需要可靠的方案，允许经典客户委托学习给不可信的量子服务器，以促进广泛获得量子学习的优势。本文基于最近引入的古典机器学习交互证明系统框架，构建了一个经典验证量子学习的框架。我们展示了一些经典学习者无法自行高效求解的学习问题，但当与不可信的量子证明者交互时，他们可以高效且可靠地解决这些问题。具体而言，我们考虑了关于具有均匀输入边缘密度的对仗学习奇偶性和傅里叶稀疏函数的问题。我们提出了一个新的量子数据访问模型，称为“叠加混合量子样例”。

    Quantum data access and quantum processing can make certain classically intractable learning tasks feasible. However, quantum capabilities will only be available to a select few in the near future. Thus, reliable schemes that allow classical clients to delegate learning to untrusted quantum servers are required to facilitate widespread access to quantum learning advantages. Building on a recently introduced framework of interactive proof systems for classical machine learning, we develop a framework for classical verification of quantum learning. We exhibit learning problems that a classical learner cannot efficiently solve on their own, but that they can efficiently and reliably solve when interacting with an untrusted quantum prover. Concretely, we consider the problems of agnostic learning parities and Fourier-sparse functions with respect to distributions with uniform input marginal. We propose a new quantum data access model that we call "mixture-of-superpositions" quantum example
    
[^6]: 随机选择纯量分解的核求积方法

    Kernel Quadrature with Randomly Pivoted Cholesky. (arXiv:2306.03955v1 [math.NA])

    [http://arxiv.org/abs/2306.03955](http://arxiv.org/abs/2306.03955)

    本文提出了一种新的使用随机选择纯量分解算法的核求积方法，可以在达到可比的求积误差达到率的同时显著降低计算复杂度，并可以应用于任意核的复杂几何结构。

    

    本文使用随机选择纯量分解的采样算法提出了一种新的重现核希尔伯特空间函数求积规则。所得的计算过程与既有的核求积方法相比，在精度和求解复杂度方面具有更好的性能。理论和实验结果表明，随机选择纯量分解的方法快速且具有可比的求积误差达到率，与基于连续体积采样、稀疏化和重组的更为昂贵的求积方案相匹配。随机选择纯量分解易于适应任意核的复杂几何结构，为核求积开辟了新的潜力。

    This paper presents new quadrature rules for functions in a reproducing kernel Hilbert space using nodes drawn by a sampling algorithm known as randomly pivoted Cholesky. The resulting computational procedure compares favorably to previous kernel quadrature methods, which either achieve low accuracy or require solving a computationally challenging sampling problem. Theoretical and numerical results show that randomly pivoted Cholesky is fast and achieves comparable quadrature error rates to more computationally expensive quadrature schemes based on continuous volume sampling, thinning, and recombination. Randomly pivoted Cholesky is easily adapted to complicated geometries with arbitrary kernels, unlocking new potential for kernel quadrature.
    
[^7]: 损失最小化分类树：一个通用的框架和逻辑斯蒂回归情况研究

    Loss-Optimal Classification Trees: A Generalized Framework and the Logistic Case. (arXiv:2306.00857v1 [stat.ML])

    [http://arxiv.org/abs/2306.00857](http://arxiv.org/abs/2306.00857)

    本论文为损失最小化分类树的解决提供了一个通用框架，基于逻辑斯蒂回归的解决方式具有具有解释性特征和竞争泛化能力。

    

    分类树是可解释机器学习中最常见的模型之一，尽管这样的模型通常采用贪婪策略构建，但是近年来，由于混合整数规划（MIP）求解器的显著进展，已经开发了几种精确的学习问题的表述。本文认为其中一些最相关的训练模型可以封装在一个通用框架内，其实例由损失函数和正则化器的规定塑造。接下来，我们介绍了这个框架的新颖实现：具体来说，我们考虑逻辑损失，它在MIP设置中通过线性分段逼近处理，并与$\ell_1$-规则化项相结合。最终的最优逻辑树模型在数值上被证明能够诱导具有增强可解释性特征和竞争性泛化能力的树，与最先进的基于MIP的方法相比。

    The Classification Tree (CT) is one of the most common models in interpretable machine learning. Although such models are usually built with greedy strategies, in recent years, thanks to remarkable advances in Mixer-Integer Programming (MIP) solvers, several exact formulations of the learning problem have been developed. In this paper, we argue that some of the most relevant ones among these training models can be encapsulated within a general framework, whose instances are shaped by the specification of loss functions and regularizers. Next, we introduce a novel realization of this framework: specifically, we consider the logistic loss, handled in the MIP setting by a linear piece-wise approximation, and couple it with $\ell_1$-regularization terms. The resulting Optimal Logistic Tree model numerically proves to be able to induce trees with enhanced interpretability features and competitive generalization capabilities, compared to the state-of-the-art MIP-based approaches.
    
[^8]: 使用高阶淬火随机漂移解决线性反问题

    Solving Linear Inverse Problems using Higher-Order Annealed Langevin Diffusion. (arXiv:2305.05014v1 [stat.ML])

    [http://arxiv.org/abs/2305.05014](http://arxiv.org/abs/2305.05014)

    本论文提出了一种高阶 Langevin 动力学的算法，可更高效地采样未知变量后验分布，同时加入淬火过程，能应用于离散未知变量情况，数值实验表明在多个任务中相对竞争方法具有更高性能。

    

    我们提出了一种基于高阶 Langevin 漂移的线性反问题解决方案。更具体地，我们提出了预处理的二阶和三阶 Langevin 动力学，这些动力学明显地从我们感兴趣的未知变量的后验分布中采样，同时比其一阶对应物和两种动力学的非预处理版本更具计算效率。此外，我们证明了两种预处理动力学是良定义的，并且具有与非预处理情况相同的唯一不变分布。我们还加入了一个淬火过程，这具有双重优点，一方面进一步加速了算法的收敛速度，另一方面，允许我们适应未知变量为离散的情况。在两个不同的任务（MIMO 符号检测和通道估计）的数值实验中，展示了我们方法的通用性，并说明了相对于竞争方法（包括基于学习的方法）所实现的高性能。

    We propose a solution for linear inverse problems based on higher-order Langevin diffusion. More precisely, we propose pre-conditioned second-order and third-order Langevin dynamics that provably sample from the posterior distribution of our unknown variables of interest while being computationally more efficient than their first-order counterpart and the non-conditioned versions of both dynamics. Moreover, we prove that both pre-conditioned dynamics are well-defined and have the same unique invariant distributions as the non-conditioned cases. We also incorporate an annealing procedure that has the double benefit of further accelerating the convergence of the algorithm and allowing us to accommodate the case where the unknown variables are discrete. Numerical experiments in two different tasks (MIMO symbol detection and channel estimation) showcase the generality of our method and illustrate the high performance achieved relative to competing approaches (including learning-based ones)
    
[^9]: 对称正定流形上低复杂度的子空间下降算法

    Low-complexity subspace-descent over symmetric positive definite manifold. (arXiv:2305.02041v1 [stat.ML])

    [http://arxiv.org/abs/2305.02041](http://arxiv.org/abs/2305.02041)

    本文提出了一种基于黎曼子空间下降算法的对称正定流形上的函数最小化方法，其具有低复杂度和避免昂贵矩阵操作和计算黎曼梯度的优点。

    

    本文提出了一种低复杂度的黎曼子空间下降算法，用于在对称正定（SPD）流形上对函数进行最小化。与现有的黎曼梯度下降变体不同的是，所提出的方法利用 carefully chosen 的子空间，使得更新可以写成迭代的 Cholesky 因子和一个稀疏矩阵的乘积形式。由此产生的更新避免了昂贵的矩阵操作，如矩阵指数和密集矩阵乘法，这些操作通常在几乎所有其他 Riemannian 优化算法中都是必需的。

    This work puts forth low-complexity Riemannian subspace descent algorithms for the minimization of functions over the symmetric positive definite (SPD) manifold. Different from the existing Riemannian gradient descent variants, the proposed approach utilizes carefully chosen subspaces that allow the update to be written as a product of the Cholesky factor of the iterate and a sparse matrix. The resulting updates avoid the costly matrix operations like matrix exponentiation and dense matrix multiplication, which are generally required in almost all other Riemannian optimization algorithms on SPD manifold. We further identify a broad class of functions, arising in diverse applications, such as kernel matrix learning, covariance estimation of Gaussian distributions, maximum likelihood parameter estimation of elliptically contoured distributions, and parameter estimation in Gaussian mixture model problems, over which the Riemannian gradients can be calculated efficiently. The proposed uni-
    
[^10]: LAVA: 无需预定学习算法的数据价值评估

    LAVA: Data Valuation without Pre-Specified Learning Algorithms. (arXiv:2305.00054v1 [cs.LG])

    [http://arxiv.org/abs/2305.00054](http://arxiv.org/abs/2305.00054)

    LAVA是一个学习算法无关的数据价值评估方法，它结合了学习算法的统计特性和训练数据的属性，通过迭代估计数据值来实现。LAVA比现有方法计算速度更快，精度更高，并且可以为不同的应用提供有意义的数据排名。

    

    传统的数据价值评估问题是如何公平地分配学习算法的验证性能，致使计算得到的数据价值依赖于底层学习算法的许多设计选择。本文提出了一种新的框架LAVA，该框架结合了学习算法的统计特性和训练数据的属性，迭代估计数据值，使其无视下游的学习算法。我们展示了LAVA比现有方法计算速度更快，精度更高，并且它可以为不同的应用提供有意义的数据排名。

    Traditionally, data valuation is posed as a problem of equitably splitting the validation performance of a learning algorithm among the training data. As a result, the calculated data values depend on many design choices of the underlying learning algorithm. However, this dependence is undesirable for many use cases of data valuation, such as setting priorities over different data sources in a data acquisition process and informing pricing mechanisms in a data marketplace. In these scenarios, data needs to be valued before the actual analysis and the choice of the learning algorithm is still undetermined then. Another side-effect of the dependence is that to assess the value of individual points, one needs to re-run the learning algorithm with and without a point, which incurs a large computation burden.  This work leapfrogs over the current limits of data valuation methods by introducing a new framework that can value training data in a way that is oblivious to the downstream learning
    
[^11]: 数据污染中的时序鲁棒性

    Temporal Robustness against Data Poisoning. (arXiv:2302.03684v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.03684](http://arxiv.org/abs/2302.03684)

    该论文提出了一种针对数据污染的时序威胁模型，通过利用数据的时间戳，引入了提前时间和持续时间这两个指标，从而定义了数据污染的时序鲁棒性，并提供了一种有效的保护方法。

    

    数据污染考虑了通过恶意训练数据操纵机器学习算法行为的情况。现有的数据污染威胁模型都围绕着一个单一指标，即被污染样本的数量。因此，如果攻击者能够以可承受的代价污染比预期更多的样本，就像许多实际场景中一样，他们可能能够在很短的时间内使现有的防御措施失效。为了解决这个问题，我们利用数据的出生日期时间戳，这些时间戳通常是可用的但过去被忽略。利用这些时间戳，我们提出了一个带有两个新型指标（提前时间和持续时间）的数据污染的时序威胁模型，分别衡量攻击提前开始的时间和攻击持续的时间。利用这些指标，我们定义了数据污染的时序鲁棒性的概念，即使有大量被污染的样本，也能提供有意义的保护。我们提出一种方法

    Data poisoning considers cases when an adversary manipulates the behavior of machine learning algorithms through malicious training data. Existing threat models of data poisoning center around a single metric, the number of poisoned samples. In consequence, if attackers can poison more samples than expected with affordable overhead, as in many practical scenarios, they may be able to render existing defenses ineffective in a short time. To address this issue, we leverage timestamps denoting the birth dates of data, which are often available but neglected in the past. Benefiting from these timestamps, we propose a temporal threat model of data poisoning with two novel metrics, earliness and duration, which respectively measure how long an attack started in advance and how long an attack lasted. Using these metrics, we define the notions of temporal robustness against data poisoning, providing a meaningful sense of protection even with unbounded amounts of poisoned samples. We present a 
    
[^12]: 基于神经网络的1维随机场生成，具有湍流速度统计特性

    Neural network based generation of 1-dimensional stochastic fields with turbulent velocity statistics. (arXiv:2211.11580v2 [eess.SP] UPDATED)

    [http://arxiv.org/abs/2211.11580](http://arxiv.org/abs/2211.11580)

    该论文介绍了一个名为NN-Turb的全卷积神经网络随机模型，用于以湍流速度统计特性生成1维场，而无需接触湍流数据。生成的过程满足Kolmogorov 2/3结构函数定律，同时还表现出负偏斜度和间歇性。

    

    我们定义并研究了一个全卷积神经网络随机模型，名为NN-Turb，用于生成具有湍流速度统计特性的1维场。这样，生成的过程满足Kolmogorov 2/3结构函数定律。它还呈现出各个尺度上的负偏斜度（即Kolmogorov 4/5定律）和间歇性。此外，我们的模型从不接触湍流数据，只需要训练所需的结构函数在各个尺度上的统计行为。

    We define and study a fully-convolutional neural network stochastic model, NN-Turb, which generates 1-dimensional fields with turbulent velocity statistics. Thus, the generated process satisfies the Kolmogorov 2/3 law for second order structure function. It also presents negative skewness across scales (i.e. Kolmogorov 4/5 law) and exhibits intermittency. Furthermore, our model is never in contact with turbulent data and only needs the desired statistical behavior of the structure functions across scales for training.
    

