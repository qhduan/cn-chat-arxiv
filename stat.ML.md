# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reliable uncertainty with cheaper neural network ensembles: a case study in industrial parts classification](https://arxiv.org/abs/2403.10182) | 研究在工业零部件分类中探讨了利用更便宜的神经网络集成实现可靠的不确定性估计的方法 |
| [^2] | [Replicability is Asymptotically Free in Multi-armed Bandits](https://arxiv.org/abs/2402.07391) | 本论文研究在多臂赌博机问题中，通过引入探索-再确定算法和连续淘汰算法，以及谨慎选择置信区间的幅度，实现了可复制性，并证明了当时间界足够大时，可复制算法的额外代价是不必要的。 |
| [^3] | [Probabilistic Forecasting of Irregular Time Series via Conditional Flows](https://arxiv.org/abs/2402.06293) | 该论文提出了一种使用条件流进行不规则时间序列的概率预测的新模型ProFITi。该模型通过学习条件下未来值的联合分布，对具有缺失值的不规则时间序列进行预测，而不假设底层分布的固定形状。通过引入可逆三角形注意力层和可逆非线性激活函数，该模型取得了良好的实验结果。 |
| [^4] | [Neural Bayes Estimators for Irregular Spatial Data using Graph Neural Networks.](http://arxiv.org/abs/2310.02600) | 通过使用图神经网络，该论文提出了一种解决非规则空间数据的参数估计问题的方法，扩展了神经贝叶斯估计器的应用范围，并带来了显著的计算优势。 |
| [^5] | [Maximum Mean Discrepancy Meets Neural Networks: The Radon-Kolmogorov-Smirnov Test.](http://arxiv.org/abs/2309.02422) | 本文将最大均差相似度应用于神经网络，并提出了一种称为Radon-Kolmogorov-Smirnov（RKS）检验的方法，该方法将样本均值差异最大化的问题推广到多维空间和更高平滑度顺序，同时与神经网络密切相关。 |
| [^6] | [Measure transfer via stochastic slicing and matching.](http://arxiv.org/abs/2307.05705) | 本文研究了通过切片和配准过程定义的测量转移和逼近问题的迭代方案，并对随机切片和配准方案提供了几乎必然收敛的证明。 |
| [^7] | [Symmetry & Critical Points for Symmetric Tensor Decompositions Problems.](http://arxiv.org/abs/2306.07886) | 本文研究了将一个实对称张量分解成秩为1项之和的非凸优化问题，得到了精确的分析估计，并发现了各种阻碍局部优化方法的几何障碍和由于对称性导致的丰富的临界点集合。 |

# 详细

[^1]: 用更便宜的神经网络集成实现可靠的不确定性：工业零部件分类案例研究

    Reliable uncertainty with cheaper neural network ensembles: a case study in industrial parts classification

    [https://arxiv.org/abs/2403.10182](https://arxiv.org/abs/2403.10182)

    研究在工业零部件分类中探讨了利用更便宜的神经网络集成实现可靠的不确定性估计的方法

    

    在运筹学(OR)中，预测模型经常会遇到数据分布与训练数据分布不同的场景。近年来，神经网络(NNs)在图像分类等领域的出色性能使其在OR中备受关注。然而，当面对OOD数据时，NNs往往会做出自信但不正确的预测。不确定性估计为自信的模型提供了一个解决方案，当输出应(不应)被信任时进行通信。因此，在OR领域中，NNs中的可靠不确定性量化至关重要。由多个独立NNs组成的深度集合已经成为一种有前景的方法，不仅提供强大的预测准确性，还能可靠地估计不确定性。然而，它们的部署由于较大的计算需求而具有挑战性。最近的基础研究提出了更高效的NN集成，即sna

    arXiv:2403.10182v1 Announce Type: new  Abstract: In operations research (OR), predictive models often encounter out-of-distribution (OOD) scenarios where the data distribution differs from the training data distribution. In recent years, neural networks (NNs) are gaining traction in OR for their exceptional performance in fields such as image classification. However, NNs tend to make confident yet incorrect predictions when confronted with OOD data. Uncertainty estimation offers a solution to overconfident models, communicating when the output should (not) be trusted. Hence, reliable uncertainty quantification in NNs is crucial in the OR domain. Deep ensembles, composed of multiple independent NNs, have emerged as a promising approach, offering not only strong predictive accuracy but also reliable uncertainty estimation. However, their deployment is challenging due to substantial computational demands. Recent fundamental research has proposed more efficient NN ensembles, namely the sna
    
[^2]: 在多臂赌博机中，可复制性渐进自由

    Replicability is Asymptotically Free in Multi-armed Bandits

    [https://arxiv.org/abs/2402.07391](https://arxiv.org/abs/2402.07391)

    本论文研究在多臂赌博机问题中，通过引入探索-再确定算法和连续淘汰算法，以及谨慎选择置信区间的幅度，实现了可复制性，并证明了当时间界足够大时，可复制算法的额外代价是不必要的。

    

    本研究受可复制的机器学习需求的推动，研究了随机多臂赌博机问题。特别地，我们考虑了一个可复制算法，确保算法的操作序列不受数据集固有随机性的影响。我们观察到，现有算法所需的遗憾值比不可复制算法多$O(1/\rho^2)$倍，其中$\rho$是非复制程度。然而，我们证明了当给定的$\rho$下时间界$T$足够大时，此额外代价是不必要的，前提是谨慎选择置信区间的幅度。我们引入了一个先探索后决策的算法，在决策之前均匀选择动作。此外，我们还研究了一个连续淘汰算法，在每个阶段结束时淘汰次优动作。为了确保这些算法的可复制性，我们将随机性引入决策制定中。

    This work is motivated by the growing demand for reproducible machine learning. We study the stochastic multi-armed bandit problem. In particular, we consider a replicable algorithm that ensures, with high probability, that the algorithm's sequence of actions is not affected by the randomness inherent in the dataset. We observe that existing algorithms require $O(1/\rho^2)$ times more regret than nonreplicable algorithms, where $\rho$ is the level of nonreplication. However, we demonstrate that this additional cost is unnecessary when the time horizon $T$ is sufficiently large for a given $\rho$, provided that the magnitude of the confidence bounds is chosen carefully. We introduce an explore-then-commit algorithm that draws arms uniformly before committing to a single arm. Additionally, we examine a successive elimination algorithm that eliminates suboptimal arms at the end of each phase. To ensure the replicability of these algorithms, we incorporate randomness into their decision-ma
    
[^3]: 通过条件流进行不规则时间序列的概率预测

    Probabilistic Forecasting of Irregular Time Series via Conditional Flows

    [https://arxiv.org/abs/2402.06293](https://arxiv.org/abs/2402.06293)

    该论文提出了一种使用条件流进行不规则时间序列的概率预测的新模型ProFITi。该模型通过学习条件下未来值的联合分布，对具有缺失值的不规则时间序列进行预测，而不假设底层分布的固定形状。通过引入可逆三角形注意力层和可逆非线性激活函数，该模型取得了良好的实验结果。

    

    不规则采样的多变量时间序列具有缺失值的概率预测是许多领域的重要问题，包括医疗保健、天文学和气候学。目前该任务的最先进方法仅估计单个通道和单个时间点上观测值的边际分布，假设了一个固定形状的参数分布。在这项工作中，我们提出了一种新的模型ProFITi，用于使用条件归一化流对具有缺失值的不规则采样时间序列进行概率预测。该模型学习了在过去观测和查询的通道和时间上条件下时间序列未来值的联合分布，而不假设底层分布的固定形状。作为模型组件，我们引入了一种新颖的可逆三角形注意力层和一个可逆的非线性激活函数，能够在整个实数线上进行转换。我们在四个数据集上进行了大量实验，并证明了该模型的提议。

    Probabilistic forecasting of irregularly sampled multivariate time series with missing values is an important problem in many fields, including health care, astronomy, and climate. State-of-the-art methods for the task estimate only marginal distributions of observations in single channels and at single timepoints, assuming a fixed-shape parametric distribution. In this work, we propose a novel model, ProFITi, for probabilistic forecasting of irregularly sampled time series with missing values using conditional normalizing flows. The model learns joint distributions over the future values of the time series conditioned on past observations and queried channels and times, without assuming any fixed shape of the underlying distribution. As model components, we introduce a novel invertible triangular attention layer and an invertible non-linear activation function on and onto the whole real line. We conduct extensive experiments on four datasets and demonstrate that the proposed model pro
    
[^4]: 使用图神经网络的非规则空间数据的神经贝叶斯估计器

    Neural Bayes Estimators for Irregular Spatial Data using Graph Neural Networks. (arXiv:2310.02600v1 [stat.ME])

    [http://arxiv.org/abs/2310.02600](http://arxiv.org/abs/2310.02600)

    通过使用图神经网络，该论文提出了一种解决非规则空间数据的参数估计问题的方法，扩展了神经贝叶斯估计器的应用范围，并带来了显著的计算优势。

    

    神经贝叶斯估计器是一种以快速和免似然方式逼近贝叶斯估计器的神经网络。它们在空间模型和数据中的使用非常吸引人，因为估计经常是计算上的瓶颈。然而，到目前为止，空间应用中的神经贝叶斯估计器仅限于在规则的网格上收集的数据。这些估计器目前还依赖于预先规定的空间位置，这意味着神经网络需要重新训练以适应新的数据集；这使它们在许多应用中变得不实用，并阻碍了它们的广泛应用。在本研究中，我们采用图神经网络来解决从任意空间位置收集的数据进行参数估计的重要问题。除了将神经贝叶斯估计扩展到非规则空间数据之外，我们的架构还带来了显着的计算优势，因为该估计器可以用于任何排列或数量的位置和独立的重复实验中。

    Neural Bayes estimators are neural networks that approximate Bayes estimators in a fast and likelihood-free manner. They are appealing to use with spatial models and data, where estimation is often a computational bottleneck. However, neural Bayes estimators in spatial applications have, to date, been restricted to data collected over a regular grid. These estimators are also currently dependent on a prescribed set of spatial locations, which means that the neural network needs to be re-trained for new data sets; this renders them impractical in many applications and impedes their widespread adoption. In this work, we employ graph neural networks to tackle the important problem of parameter estimation from data collected over arbitrary spatial locations. In addition to extending neural Bayes estimation to irregular spatial data, our architecture leads to substantial computational benefits, since the estimator can be used with any arrangement or number of locations and independent repli
    
[^5]: 最大均差相似度遇上神经网络：Radon-Kolmogorov-Smirnov检验

    Maximum Mean Discrepancy Meets Neural Networks: The Radon-Kolmogorov-Smirnov Test. (arXiv:2309.02422v1 [stat.ML])

    [http://arxiv.org/abs/2309.02422](http://arxiv.org/abs/2309.02422)

    本文将最大均差相似度应用于神经网络，并提出了一种称为Radon-Kolmogorov-Smirnov（RKS）检验的方法，该方法将样本均值差异最大化的问题推广到多维空间和更高平滑度顺序，同时与神经网络密切相关。

    

    最大均差相似度（MMD）是一类基于最大化两个分布$P$和$Q$之间样本均值差异的非参数双样本检验，其中考虑了所有在某个函数空间$\mathcal{F}$中的数据变换$f$的选择。受到最近将所谓的Radon有界变差函数（RBV）和神经网络联系起来的工作的启发（Parhi和Nowak, 2021, 2023），我们研究了将$\mathcal{F}$取为给定平滑度顺序$k \geq 0$下的RBV空间中的单位球的MMD。这个检验被称为Radon-Kolmogorov-Smirnov（RKS）检验，可以看作是对多维空间和更高平滑度顺序的经典Kolmogorov-Smirnov（KS）检验的一般化。它还与神经网络密切相关：我们证明RKS检验中的证据函数$f$，即达到最大均差的函数，总是一个二次样条函数。

    Maximum mean discrepancy (MMD) refers to a general class of nonparametric two-sample tests that are based on maximizing the mean difference over samples from one distribution $P$ versus another $Q$, over all choices of data transformations $f$ living in some function space $\mathcal{F}$. Inspired by recent work that connects what are known as functions of $\textit{Radon bounded variation}$ (RBV) and neural networks (Parhi and Nowak, 2021, 2023), we study the MMD defined by taking $\mathcal{F}$ to be the unit ball in the RBV space of a given smoothness order $k \geq 0$. This test, which we refer to as the $\textit{Radon-Kolmogorov-Smirnov}$ (RKS) test, can be viewed as a generalization of the well-known and classical Kolmogorov-Smirnov (KS) test to multiple dimensions and higher orders of smoothness. It is also intimately connected to neural networks: we prove that the witness in the RKS test -- the function $f$ achieving the maximum mean difference -- is always a ridge spline of degree
    
[^6]: 通过随机切片和配准进行测量转移

    Measure transfer via stochastic slicing and matching. (arXiv:2307.05705v1 [math.NA])

    [http://arxiv.org/abs/2307.05705](http://arxiv.org/abs/2307.05705)

    本文研究了通过切片和配准过程定义的测量转移和逼近问题的迭代方案，并对随机切片和配准方案提供了几乎必然收敛的证明。

    

    本论文研究了通过切片和配准过程定义的测量转移和逼近问题的迭代方案。类似于切片Wasserstein距离，这些方案受益于一维最优输运问题的闭式解的可用性和相关计算优势。尽管这些方案已经在数据科学应用中取得了成功，但关于它们的收敛性的结果不太多。本文的主要贡献是对随机切片和配准方案提供了几乎必然收敛的证明。该证明建立在将其解释为Wasserstein空间上的随机梯度下降方案的基础之上。同时还展示了关于逐步图像变形的数值示例。

    This paper studies iterative schemes for measure transfer and approximation problems, which are defined through a slicing-and-matching procedure. Similar to the sliced Wasserstein distance, these schemes benefit from the availability of closed-form solutions for the one-dimensional optimal transport problem and the associated computational advantages. While such schemes have already been successfully utilized in data science applications, not too many results on their convergence are available. The main contribution of this paper is an almost sure convergence proof for stochastic slicing-and-matching schemes. The proof builds on an interpretation as a stochastic gradient descent scheme on the Wasserstein space. Numerical examples on step-wise image morphing are demonstrated as well.
    
[^7]: 对称张量分解问题的对称性与临界点

    Symmetry & Critical Points for Symmetric Tensor Decompositions Problems. (arXiv:2306.07886v1 [math.OC])

    [http://arxiv.org/abs/2306.07886](http://arxiv.org/abs/2306.07886)

    本文研究了将一个实对称张量分解成秩为1项之和的非凸优化问题，得到了精确的分析估计，并发现了各种阻碍局部优化方法的几何障碍和由于对称性导致的丰富的临界点集合。

    

    本文考虑了将一个实对称张量分解成秩为1项之和的非凸优化问题。利用其丰富的对称结构，导出Puiseux级数表示的一系列临界点，并获得了关于临界值和Hessian谱的精确分析估计。这些结果揭示了各种几何障碍，阻碍了局部优化方法的使用，最后，利用一个牛顿多面体论证了固定对称性的所有临界点的完全枚举，并证明了与全局最小值的集合相比，由于对称性的存在，临界点的集合可能会显示出组合的丰富性。

    We consider the non-convex optimization problem associated with the decomposition of a real symmetric tensor into a sum of rank one terms. Use is made of the rich symmetry structure to derive Puiseux series representations of families of critical points, and so obtain precise analytic estimates on the critical values and the Hessian spectrum. The sharp results make possible an analytic characterization of various geometric obstructions to local optimization methods, revealing in particular a complex array of saddles and local minima which differ by their symmetry, structure and analytic properties. A desirable phenomenon, occurring for all critical points considered, concerns the index of a point, i.e., the number of negative Hessian eigenvalues, increasing with the value of the objective function. Lastly, a Newton polytope argument is used to give a complete enumeration of all critical points of fixed symmetry, and it is shown that contrarily to the set of global minima which remains 
    

