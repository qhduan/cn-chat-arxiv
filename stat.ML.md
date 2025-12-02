# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CAS: A General Algorithm for Online Selective Conformal Prediction with FCR Control](https://arxiv.org/abs/2403.07728) | CAS框架允许在在线选择性预测中控制FCR，通过自适应选择和校准集构造输出符合预测区间 |
| [^2] | [Koopman operators with intrinsic observables in rigged reproducing kernel Hilbert spaces](https://arxiv.org/abs/2403.02524) | 本文提出了一种基于装配再生核希尔伯特空间内在结构和jets几何概念的估计Koopman算子的新方法JetDMD，通过明确的误差界和收敛率证明其优越性，为Koopman算子的数值估计提供了更精确的方法，同时在装配希尔伯特空间框架内提出了扩展Koopman算子的概念，有助于深入理解估计的Koopman特征函数。 |
| [^3] | [From Large to Small Datasets: Size Generalization for Clustering Algorithm Selection](https://arxiv.org/abs/2402.14332) | 通过引入尺寸泛化概念，研究了在半监督设置下的聚类算法选择问题，提出了能够在小实例上保证准确度最高的算法也将在原始大实例上拥有最高准确度的条件。 |
| [^4] | [Estimating the Mixing Coefficients of Geometrically Ergodic Markov Processes](https://arxiv.org/abs/2402.07296) | 该论文提出了一种方法来估计几何遗传马尔可夫过程的混合系数，我们通过在满足特定条件和无需密度假设的情况下，得到了估计器的预期误差收敛速度和高概率界限。 |
| [^5] | [Families of costs with zero and nonnegative MTW tensor in optimal transport.](http://arxiv.org/abs/2401.00953) | 这篇论文介绍了在最优传送中使用形式为$\mathsf{c}(x, y) = \mathsf{u}(x^{\mathfrak{t}}y)$的费用函数时的零和非负MTW张量的计算方法，并提供了MTW张量在零向量上为零的条件以及相应的线性ODE的简化方法。此外，还给出了逆函数的解析表达式以及一些具体的应用情况。 |
| [^6] | [RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization.](http://arxiv.org/abs/2310.07983) | RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。 |
| [^7] | [Sparse PCA With Multiple Components.](http://arxiv.org/abs/2209.14790) | 本研究提出了一种新的方法来解决稀疏主成分分析问题，通过将正交性条件重新表述为秩约束，并同时对稀疏性和秩约束进行优化。我们设计了紧凑的半正定松弛来提供高质量的上界，当每个主成分的个体稀疏性被指定时，我们通过额外的二阶锥不等式加强上界。 |

# 详细

[^1]: CAS: 一种具有FCR控制的在线选择性符合预测的通用算法

    CAS: A General Algorithm for Online Selective Conformal Prediction with FCR Control

    [https://arxiv.org/abs/2403.07728](https://arxiv.org/abs/2403.07728)

    CAS框架允许在在线选择性预测中控制FCR，通过自适应选择和校准集构造输出符合预测区间

    

    我们研究了在线方式下后选择预测推断的问题。为了避免将资源耗费在不重要的单位上，在报告其预测区间之前对当前个体进行初步选择在在线预测任务中是常见且有意义的。由于在线选择导致所选预测区间中存在时间多重性，因此控制实时误覆盖陈述率（FCR）来测量平均误覆盖误差是重要的。我们开发了一个名为CAS（适应性选择后校准）的通用框架，可以包裹任何预测模型和在线选择规则，以输出后选择的预测区间。如果选择了当前个体，我们首先对历史数据进行自适应选择来构建校准集，然后为未观察到的标签输出符合预测区间。我们为校准集提供了可行的构造方式

    arXiv:2403.07728v1 Announce Type: cross  Abstract: We study the problem of post-selection predictive inference in an online fashion. To avoid devoting resources to unimportant units, a preliminary selection of the current individual before reporting its prediction interval is common and meaningful in online predictive tasks. Since the online selection causes a temporal multiplicity in the selected prediction intervals, it is important to control the real-time false coverage-statement rate (FCR) to measure the averaged miscoverage error. We develop a general framework named CAS (Calibration after Adaptive Selection) that can wrap around any prediction model and online selection rule to output post-selection prediction intervals. If the current individual is selected, we first perform an adaptive selection on historical data to construct a calibration set, then output a conformal prediction interval for the unobserved label. We provide tractable constructions for the calibration set for 
    
[^2]: 在装配再生核希尔伯特空间中具有内在可观测性的Koopman算子

    Koopman operators with intrinsic observables in rigged reproducing kernel Hilbert spaces

    [https://arxiv.org/abs/2403.02524](https://arxiv.org/abs/2403.02524)

    本文提出了一种基于装配再生核希尔伯特空间内在结构和jets几何概念的估计Koopman算子的新方法JetDMD，通过明确的误差界和收敛率证明其优越性，为Koopman算子的数值估计提供了更精确的方法，同时在装配希尔伯特空间框架内提出了扩展Koopman算子的概念，有助于深入理解估计的Koopman特征函数。

    

    本文提出了一种新颖的方法，用于估计装配再生核希尔伯特空间（RKHS）上定义的Koopman算子及其谱。我们提出了一种估计方法，称为Jet Dynamic Mode Decomposition（JetDMD），利用RKHS的内在结构和称为jets的几何概念来增强Koopman算子的估计。该方法在精确度上优化了传统的扩展动态模态分解（EDMD），特别是在特征值的数值估计方面。本文通过明确的误差界和特殊正定内核的收敛率证明了JetDMD的优越性，为其性能提供了坚实的理论基础。我们还深入探讨了Koopman算子的谱分析，在装配希尔伯特空间框架内提出了扩展Koopman算子的概念。这个概念有助于更深入地理解估计的Koopman特征函数并捕捉

    arXiv:2403.02524v1 Announce Type: cross  Abstract: This paper presents a novel approach for estimating the Koopman operator defined on a reproducing kernel Hilbert space (RKHS) and its spectra. We propose an estimation method, what we call Jet Dynamic Mode Decomposition (JetDMD), leveraging the intrinsic structure of RKHS and the geometric notion known as jets to enhance the estimation of the Koopman operator. This method refines the traditional Extended Dynamic Mode Decomposition (EDMD) in accuracy, especially in the numerical estimation of eigenvalues. This paper proves JetDMD's superiority through explicit error bounds and convergence rate for special positive definite kernels, offering a solid theoretical foundation for its performance. We also delve into the spectral analysis of the Koopman operator, proposing the notion of extended Koopman operator within a framework of rigged Hilbert space. This notion leads to a deeper understanding of estimated Koopman eigenfunctions and captu
    
[^3]: 从大规模到小规模数据集：用于聚类算法选择的尺寸泛化

    From Large to Small Datasets: Size Generalization for Clustering Algorithm Selection

    [https://arxiv.org/abs/2402.14332](https://arxiv.org/abs/2402.14332)

    通过引入尺寸泛化概念，研究了在半监督设置下的聚类算法选择问题，提出了能够在小实例上保证准确度最高的算法也将在原始大实例上拥有最高准确度的条件。

    

    在聚类算法选择中，我们会得到一个大规模数据集，并要有效地选择要使用的聚类算法。我们在半监督设置下研究了这个问题，其中有一个未知的基准聚类，我们只能通过昂贵的oracle查询来访问。理想情况下，聚类算法的输出将与基本事实结构上接近。我们通过引入一种聚类算法准确性的尺寸泛化概念来解决这个问题。我们确定在哪些条件下我们可以（1）对大规模聚类实例进行子采样，（2）在较小实例上评估一组候选算法，（3）保证在小实例上准确度最高的算法将在原始大实例上拥有最高的准确度。我们为三种经典聚类算法提供了理论尺寸泛化保证：单链接、k-means++和Gonzalez的k中心启发式（一种平滑的变种）。

    arXiv:2402.14332v1 Announce Type: new  Abstract: In clustering algorithm selection, we are given a massive dataset and must efficiently select which clustering algorithm to use. We study this problem in a semi-supervised setting, with an unknown ground-truth clustering that we can only access through expensive oracle queries. Ideally, the clustering algorithm's output will be structurally close to the ground truth. We approach this problem by introducing a notion of size generalization for clustering algorithm accuracy. We identify conditions under which we can (1) subsample the massive clustering instance, (2) evaluate a set of candidate algorithms on the smaller instance, and (3) guarantee that the algorithm with the best accuracy on the small instance will have the best accuracy on the original big instance. We provide theoretical size generalization guarantees for three classic clustering algorithms: single-linkage, k-means++, and (a smoothed variant of) Gonzalez's k-centers heuris
    
[^4]: 估计几何遗传马尔可夫过程的混合系数

    Estimating the Mixing Coefficients of Geometrically Ergodic Markov Processes

    [https://arxiv.org/abs/2402.07296](https://arxiv.org/abs/2402.07296)

    该论文提出了一种方法来估计几何遗传马尔可夫过程的混合系数，我们通过在满足特定条件和无需密度假设的情况下，得到了估计器的预期误差收敛速度和高概率界限。

    

    我们提出了一种方法来估计实值几何遗传马尔可夫过程的单个β-混合系数从一个单一的样本路径X0，X1，...，Xn。在对密度的标准光滑条件下，即对于每个m，对$(X_0,X_m)$对的联合密度都属于某个已知$s>0$的 Besov 空间$B^s_{1,\infty}(\mathbb R^2)$，我们得到了我们在这种情况下的估计器的预期误差的收敛速度为$\mathcal{O}(\log(n) n^{-[s]/(2[s]+2)})$ 的收敛速度。我们通过对估计误差的高概率界限进行了补充，并在状态空间有限的情况下获得了这些界限的类比。在这种情况下不需要密度的假设；预期误差率显示为$\mathcal O(\log(

    We propose methods to estimate the individual $\beta$-mixing coefficients of a real-valued geometrically ergodic Markov process from a single sample-path $X_0,X_1, \dots,X_n$. Under standard smoothness conditions on the densities, namely, that the joint density of the pair $(X_0,X_m)$ for each $m$ lies in a Besov space $B^s_{1,\infty}(\mathbb R^2)$ for some known $s>0$, we obtain a rate of convergence of order $\mathcal{O}(\log(n) n^{-[s]/(2[s]+2)})$ for the expected error of our estimator in this case\footnote{We use $[s]$ to denote the integer part of the decomposition $s=[s]+\{s\}$ of $s \in (0,\infty)$ into an integer term and a {\em strictly positive} remainder term $\{s\} \in (0,1]$.}. We complement this result with a high-probability bound on the estimation error, and further obtain analogues of these bounds in the case where the state-space is finite. Naturally no density assumptions are required in this setting; the expected error rate is shown to be of order $\mathcal O(\log(
    
[^5]: 拥有零和非负MTW张量的费用族在最优传送中的应用

    Families of costs with zero and nonnegative MTW tensor in optimal transport. (arXiv:2401.00953v1 [math.AP])

    [http://arxiv.org/abs/2401.00953](http://arxiv.org/abs/2401.00953)

    这篇论文介绍了在最优传送中使用形式为$\mathsf{c}(x, y) = \mathsf{u}(x^{\mathfrak{t}}y)$的费用函数时的零和非负MTW张量的计算方法，并提供了MTW张量在零向量上为零的条件以及相应的线性ODE的简化方法。此外，还给出了逆函数的解析表达式以及一些具体的应用情况。

    

    我们计算了在$\mathbb{R}^n$上具有形式$\mathsf{c}(x, y) = \mathsf{u}(x^{\mathfrak{t}}y)$的费用函数的最优传送问题的MTW张量（或交叉曲率）。其中，$\mathsf{u}$是一个具有逆函数$\mathsf{s}$的标量函数，$x^{\ft}y$是属于$\mathbb{R}^n$开子集的向量$x，y$的非退化双线性配对。MTW张量在Kim-McCann度量下对于零向量的条件是一个四阶非线性ODE，可以被简化为具有常数系数$P$和$S$的形式为$\mathsf{s}^{(2)} - S\mathsf{s}^{(1)} + P\mathsf{s} = 0$的线性ODE。最终得到的逆函数包括Lambert和广义反双曲/三角函数。平方欧氏度量和$\log$型费用是这些解的实例。这个家族的最优映射也是显式的。

    We compute explicitly the MTW tensor (or cross curvature) for the optimal transport problem on $\mathbb{R}^n$ with a cost function of form $\mathsf{c}(x, y) = \mathsf{u}(x^{\mathfrak{t}}y)$, where $\mathsf{u}$ is a scalar function with inverse $\mathsf{s}$, $x^{\ft}y$ is a nondegenerate bilinear pairing of vectors $x, y$ belonging to an open subset of $\mathbb{R}^n$. The condition that the MTW-tensor vanishes on null vectors under the Kim-McCann metric is a fourth-order nonlinear ODE, which could be reduced to a linear ODE of the form $\mathsf{s}^{(2)} - S\mathsf{s}^{(1)} + P\mathsf{s} = 0$ with constant coefficients $P$ and $S$. The resulting inverse functions include {\it Lambert} and {\it generalized inverse hyperbolic\slash trigonometric} functions. The square Euclidean metric and $\log$-type costs are equivalent to instances of these solutions. The optimal map for the family is also explicit. For cost functions of a similar form on a hyperboloid model of the hyperbolic space and u
    
[^6]: RandCom：去中心化随机通信跳跃方法用于分布式随机优化

    RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization. (arXiv:2310.07983v1 [cs.LG])

    [http://arxiv.org/abs/2310.07983](http://arxiv.org/abs/2310.07983)

    RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。

    

    具有随机通信跳过的分布式优化方法因其在加速通信复杂性方面具有的优势而受到越来越多的关注。然而，现有的研究主要集中在强凸确定性设置的集中式通信协议上。在本研究中，我们提出了一种名为RandCom的分布式优化方法，它采用了概率性的本地更新。我们分析了RandCom在随机非凸、凸和强凸设置中的性能，并证明了它能够通过通信概率来渐近地减少通信开销。此外，我们证明当节点数量增加时，RandCom能够实现线性加速。在随机强凸设置中，我们进一步证明了RandCom可以通过独立于网络的步长实现线性加速。此外，我们将RandCom应用于联邦学习，并提供了关于实现线性加速的潜力的积极结果。

    Distributed optimization methods with random communication skips are gaining increasing attention due to their proven benefits in accelerating communication complexity. Nevertheless, existing research mainly focuses on centralized communication protocols for strongly convex deterministic settings. In this work, we provide a decentralized optimization method called RandCom, which incorporates probabilistic local updates. We analyze the performance of RandCom in stochastic non-convex, convex, and strongly convex settings and demonstrate its ability to asymptotically reduce communication overhead by the probability of communication. Additionally, we prove that RandCom achieves linear speedup as the number of nodes increases. In stochastic strongly convex settings, we further prove that RandCom can achieve linear speedup with network-independent stepsizes. Moreover, we apply RandCom to federated learning and provide positive results concerning the potential for achieving linear speedup and
    
[^7]: 多组分的稀疏主成分分析

    Sparse PCA With Multiple Components. (arXiv:2209.14790v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2209.14790](http://arxiv.org/abs/2209.14790)

    本研究提出了一种新的方法来解决稀疏主成分分析问题，通过将正交性条件重新表述为秩约束，并同时对稀疏性和秩约束进行优化。我们设计了紧凑的半正定松弛来提供高质量的上界，当每个主成分的个体稀疏性被指定时，我们通过额外的二阶锥不等式加强上界。

    

    稀疏主成分分析是一种用于以可解释的方式解释高维数据集方差的基本技术。这涉及解决一个稀疏性和正交性约束的凸最大化问题，其计算复杂度非常高。大多数现有的方法通过迭代计算一个稀疏主成分并缩减协方差矩阵来解决稀疏主成分分析，但在寻找多个相互正交的主成分时，这些方法不能保证所得解的正交性和最优性。我们挑战这种现状，通过将正交性条件重新表述为秩约束，并同时对稀疏性和秩约束进行优化。我们设计了紧凑的半正定松弛来提供高质量的上界，当每个主成分的个体稀疏性被指定时，我们通过额外的二阶锥不等式加强上界。此外，我们采用另一种方法来加强上界，我们使用额外的二阶锥不等式来加强上界。

    Sparse Principal Component Analysis (sPCA) is a cardinal technique for obtaining combinations of features, or principal components (PCs), that explain the variance of high-dimensional datasets in an interpretable manner. This involves solving a sparsity and orthogonality constrained convex maximization problem, which is extremely computationally challenging. Most existing works address sparse PCA via methods-such as iteratively computing one sparse PC and deflating the covariance matrix-that do not guarantee the orthogonality, let alone the optimality, of the resulting solution when we seek multiple mutually orthogonal PCs. We challenge this status by reformulating the orthogonality conditions as rank constraints and optimizing over the sparsity and rank constraints simultaneously. We design tight semidefinite relaxations to supply high-quality upper bounds, which we strengthen via additional second-order cone inequalities when each PC's individual sparsity is specified. Further, we de
    

