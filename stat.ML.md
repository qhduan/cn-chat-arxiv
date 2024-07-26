# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [No Free Prune: Information-Theoretic Barriers to Pruning at Initialization](https://rss.arxiv.org/abs/2402.01089) | 本文解释了为什么在初始化时修剪神经网络困难，并提出了一个关于有效参数数量的理论解释。我们指出，在嘈杂数据中鲁棒地插值的稀疏神经网络需要严重依赖于数据的掩码。为此，我们怀疑在训练过程中和训练后修剪是必要的。 |
| [^2] | [Efficient Combinatorial Optimization via Heat Diffusion](https://arxiv.org/abs/2403.08757) | 通过热扩散实现了高效的组合优化，克服了现有方法在搜索全局最优时效率有限的问题。 |
| [^3] | [Goodness-of-Fit and Clustering of Spherical Data: the QuadratiK package in R and Python](https://arxiv.org/abs/2402.02290) | QuadratiK软件包是一个在R和Python中实现的数据分析工具，它提供了一套全面的拟合度测试和基于核方法的聚类技术，特别适用于处理球形数据。 |
| [^4] | [Predicting the structure of dynamic graphs.](http://arxiv.org/abs/2401.04280) | 本文提出了一种预测动态图结构的方法，利用时间序列方法预测未来时间点的节点度，并结合通量平衡分析方法获得未来图的结构，评估了该方法在合成和真实数据集上的实用性和适用性。 |
| [^5] | [Unsupervised Outlier Detection using Random Subspace and Subsampling Ensembles of Dirichlet Process Mixtures.](http://arxiv.org/abs/2401.00773) | 提出一种基于随机子空间和子抽样集合的Dirichlet过程高斯混合模型的无监督异常检测方法，提高了计算效率和检测器的鲁棒性。 |
| [^6] | [Generative Learning of Continuous Data by Tensor Networks.](http://arxiv.org/abs/2310.20498) | 张量网络生成模型一般适用于二进制或类别数据，这篇论文介绍了一种新型张量网络生成模型，它可以用于学习连续数据分布，并展示了该模型在合成和真实数据集上的性能表现。 |
| [^7] | [Lattice Approximations in Wasserstein Space.](http://arxiv.org/abs/2310.09149) | 本论文研究了在Wasserstein空间中通过离散和分段常数测度进行的结构逼近方法。结果表明，对于满秩的格点按比例缩放后得到的Voronoi分割逼近的测度误差是$O(h)$，逼近的$N$项误差为$O(N^{-\frac1d})$，并且可以推广到非紧支撑测度。 |
| [^8] | [The most likely common cause.](http://arxiv.org/abs/2306.17557) | 对于因果不充分的情况下的共同原因问题，我们使用广义最大似然方法来识别共同原因C，与最大熵原则密切相关。对于两个二元对称变量的研究揭示了类似于二阶相变的条件概率非解析行为。 |
| [^9] | [Clustering with minimum spanning trees: How good can it be?.](http://arxiv.org/abs/2303.05679) | 本文研究了使用最小生成树（MST）进行分区数据聚类任务的意义程度，并发现MST方法在总体上具有很强的竞争力。此外，通过回顾、研究、扩展和推广现有的MST-based划分方案，我们提出了一些新的和值得注意的方法。总体上，Genie和信息论方法往往优于其他非MST算法，在某些情况下MST方法可能不如其他算法。 |
| [^10] | [Sequence Generation via Subsequence Similarity: Theory and Application to UAV Identification.](http://arxiv.org/abs/2301.08403) | 本文探究了一种单次生成模型的多样性，主要聚焦于子序列相似性如何影响整个序列相似性，并通过生成子序列相似的序列来增强数据集。 |
| [^11] | [Normalised clustering accuracy: An asymmetric external cluster validity measure.](http://arxiv.org/abs/2209.02935) | 本文提出了一种非对称的外部聚类有效度量方法，旨在区分不同任务类型上表现良好和系统性表现不佳的聚类算法。与传统的内部度量不同，该方法利用参考真实分组进行评估，并弥补了现有方法在最坏情况下的误差。 |
| [^12] | [Nonparametric extensions of randomized response for private confidence sets.](http://arxiv.org/abs/2202.08728) | 本文提出了一种随机响应机制的扩展，可在局部差分隐私约束下计算非参数非渐进统计推断，由此得到的结果可用于生成效率高的置信区间和时间均匀置信序列。利用这些方法可以进行实证研究并产生私有模拟。 |

# 详细

[^1]: 无免费修剪：初始化时剪枝的信息论障碍

    No Free Prune: Information-Theoretic Barriers to Pruning at Initialization

    [https://rss.arxiv.org/abs/2402.01089](https://rss.arxiv.org/abs/2402.01089)

    本文解释了为什么在初始化时修剪神经网络困难，并提出了一个关于有效参数数量的理论解释。我们指出，在嘈杂数据中鲁棒地插值的稀疏神经网络需要严重依赖于数据的掩码。为此，我们怀疑在训练过程中和训练后修剪是必要的。

    

    “抽奖中奖者”是否在初始化时存在，引发了一个令人着迷的问题：深度学习是否需要大型模型，或者可以在不训练包含它们的密集模型的情况下迅速识别和训练稀疏网络。然而，尝试在初始化时找到这些稀疏子网络（“初始化时修剪”）的努力在广泛上都没有成功。我们提出了一个理论解释，基于模型的有效参数数量$p_\text{eff}$，由最终网络中非零权重的数量和稀疏掩码与数据之间的相互信息的总和给出。我们展示了“鲁棒性定律”（arXiv:2105.12806）延伸到稀疏网络，其中常规参数数量被$p_\text{eff}$所取代，这意味着一个能够在嘈杂数据中鲁棒地插值的稀疏神经网络需要严重依赖于数据的掩码。我们假设在训练过程中和训练后修剪。

    The existence of "lottery tickets" arXiv:1803.03635 at or near initialization raises the tantalizing question of whether large models are necessary in deep learning, or whether sparse networks can be quickly identified and trained without ever training the dense models that contain them. However, efforts to find these sparse subnetworks without training the dense model ("pruning at initialization") have been broadly unsuccessful arXiv:2009.08576. We put forward a theoretical explanation for this, based on the model's effective parameter count, $p_\text{eff}$, given by the sum of the number of non-zero weights in the final network and the mutual information between the sparsity mask and the data. We show the Law of Robustness of arXiv:2105.12806 extends to sparse networks with the usual parameter count replaced by $p_\text{eff}$, meaning a sparse neural network which robustly interpolates noisy data requires a heavily data-dependent mask. We posit that pruning during and after training 
    
[^2]: 通过热扩散实现高效的组合优化

    Efficient Combinatorial Optimization via Heat Diffusion

    [https://arxiv.org/abs/2403.08757](https://arxiv.org/abs/2403.08757)

    通过热扩散实现了高效的组合优化，克服了现有方法在搜索全局最优时效率有限的问题。

    

    论文探讨了通过热扩散来实现高效的组合优化。针对现有方法只能在每次迭代中访问解空间的一小部分这一限制，提出了一种框架来解决一般的组合优化问题，并且在一系列最具挑战性和广泛遇到的组合优化中展现出卓越性能。

    arXiv:2403.08757v1 Announce Type: cross  Abstract: Combinatorial optimization problems are widespread but inherently challenging due to their discrete nature.The primary limitation of existing methods is that they can only access a small fraction of the solution space at each iteration, resulting in limited efficiency for searching the global optimal. To overcome this challenge, diverging from conventional efforts of expanding the solver's search scope, we focus on enabling information to actively propagate to the solver through heat diffusion. By transforming the target function while preserving its optima, heat diffusion facilitates information flow from distant regions to the solver, providing more efficient navigation. Utilizing heat diffusion, we propose a framework for solving general combinatorial optimization problems. The proposed methodology demonstrates superior performance across a range of the most challenging and widely encountered combinatorial optimizations. Echoing rec
    
[^3]: 球形数据的拟合度和聚类：R和Python中的QuadratiK软件包

    Goodness-of-Fit and Clustering of Spherical Data: the QuadratiK package in R and Python

    [https://arxiv.org/abs/2402.02290](https://arxiv.org/abs/2402.02290)

    QuadratiK软件包是一个在R和Python中实现的数据分析工具，它提供了一套全面的拟合度测试和基于核方法的聚类技术，特别适用于处理球形数据。

    

    我们介绍了QuadratiK软件包，该软件包包含了创新的数据分析方法。该软件包在R和Python中实现，提供了一套全面的适应度拟合测试和基于核方法的二次距离的聚类技术，从而弥合了统计学和机器学习文献之间的差距。我们的软件实现了单样本、双样本和k样本适应度拟合测试，提供了一种高效且数学上合理的方法来评估概率分布的拟合度。我们的软件扩展了功能，包括基于泊松核密度的$d$维球上均匀性测试，以及从泊松核密度中生成随机样本的算法。特别值得注意的是，我们的软件还包括一种针对球形数据而特别量身定制的独特聚类算法，该算法利用了球面上基于泊松核密度的混合模型。同时，我们的软件还包括其他图形功能。

    We introduce the QuadratiK package that incorporates innovative data analysis methodologies. The presented software, implemented in both R and Python, offers a comprehensive set of goodness-of-fit tests and clustering techniques using kernel-based quadratic distances, thereby bridging the gap between the statistical and machine learning literatures. Our software implements one, two and k-sample tests for goodness of fit, providing an efficient and mathematically sound way to assess the fit of probability distributions. Expanded capabilities of our software include supporting tests for uniformity on the $d$-dimensional Sphere based on Poisson kernel densities, and algorithms for generating random samples from Poisson kernel densities. Particularly noteworthy is the incorporation of a unique clustering algorithm specifically tailored for spherical data that leverages a mixture of Poisson-kernel-based densities on the sphere. Alongside this, our software includes additional graphical func
    
[^4]: 预测动态图的结构

    Predicting the structure of dynamic graphs. (arXiv:2401.04280v1 [cs.LG])

    [http://arxiv.org/abs/2401.04280](http://arxiv.org/abs/2401.04280)

    本文提出了一种预测动态图结构的方法，利用时间序列方法预测未来时间点的节点度，并结合通量平衡分析方法获得未来图的结构，评估了该方法在合成和真实数据集上的实用性和适用性。

    

    动态图嵌入、归纳和增量学习有助于预测任务，如节点分类和链接预测。然而，从图的时间序列中预测未来时间步的图结构，允许有新节点，并没有受到太多关注。在本文中，我们提出了一种这样的方法。我们使用时间序列方法预测未来时间点的节点度，并将其与通量平衡分析（一种在生物化学中使用的线性规划方法）结合起来，以获得未来图的结构。此外，我们探索了不同参数值的预测图分布。我们使用合成和真实数据集评估了该方法，并展示了其实用性和适用性。

    Dynamic graph embeddings, inductive and incremental learning facilitate predictive tasks such as node classification and link prediction. However, predicting the structure of a graph at a future time step from a time series of graphs, allowing for new nodes has not gained much attention. In this paper, we present such an approach. We use time series methods to predict the node degree at future time points and combine it with flux balance analysis -- a linear programming method used in biochemistry -- to obtain the structure of future graphs. Furthermore, we explore the predictive graph distribution for different parameter values. We evaluate this method using synthetic and real datasets and demonstrate its utility and applicability.
    
[^5]: 使用随机子空间和Dirichlet过程混合模型的子抽样集合进行无监督异常检测

    Unsupervised Outlier Detection using Random Subspace and Subsampling Ensembles of Dirichlet Process Mixtures. (arXiv:2401.00773v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2401.00773](http://arxiv.org/abs/2401.00773)

    提出一种基于随机子空间和子抽样集合的Dirichlet过程高斯混合模型的无监督异常检测方法，提高了计算效率和检测器的鲁棒性。

    

    概率混合模型被认为是一种有价值的工具，用于无监督异常检测，因为它们具有解释性，并且在统计原理上有直观基础。在这个框架内，Dirichlet过程混合模型作为传统有限混合模型在聚类和异常检测任务中的一个引人注目的替代选择。然而，尽管它们明显具有优势，但在无监督异常检测中广泛采用Dirichlet过程混合模型受到与构建检测器过程中的计算效率和对异常值的敏感性有关的挑战的阻碍。为了解决这些挑战，我们提出了一种基于Dirichlet过程高斯混合模型集合的新型异常检测方法。所提出的方法是一种完全无监督的算法，利用了随机子空间和子抽样集合，不仅确保了高效计算，还增强了结果异常检测器的鲁棒性。

    Probabilistic mixture models are acknowledged as a valuable tool for unsupervised outlier detection owing to their interpretability and intuitive grounding in statistical principles. Within this framework, Dirichlet process mixture models emerge as a compelling alternative to conventional finite mixture models for both clustering and outlier detection tasks. However, despite their evident advantages, the widespread adoption of Dirichlet process mixture models in unsupervised outlier detection has been hampered by challenges related to computational inefficiency and sensitivity to outliers during the construction of detectors. To tackle these challenges, we propose a novel outlier detection method based on ensembles of Dirichlet process Gaussian mixtures. The proposed method is a fully unsupervised algorithm that capitalizes on random subspace and subsampling ensembles, not only ensuring efficient computation but also enhancing the robustness of the resulting outlier detector. Moreover,
    
[^6]: 通过张量网络生成连续数据的生成学习

    Generative Learning of Continuous Data by Tensor Networks. (arXiv:2310.20498v1 [cs.LG])

    [http://arxiv.org/abs/2310.20498](http://arxiv.org/abs/2310.20498)

    张量网络生成模型一般适用于二进制或类别数据，这篇论文介绍了一种新型张量网络生成模型，它可以用于学习连续数据分布，并展示了该模型在合成和真实数据集上的性能表现。

    

    张量网络除了用于建模多体量子系统外，还成为解决机器学习问题的一类有前景的模型，尤其是在无监督生成学习中。然而，以量子启发式为特点的张量网络生成模型之前主要局限于二进制或类别数据，限制了它们在现实世界建模问题中的效用。我们通过引入一种能够学习包含连续随机变量的分布的新型张量网络生成模型，克服了这一局限。我们首先在矩阵积态的设置下开发了我们的方法，证明了这个模型族能够以任意精度逼近任何相对平滑的概率密度函数的一般表达性定理。然后，我们在几个合成和真实世界数据集上评估了这个模型的性能，发现该模型具有较好的表现。

    Beyond their origin in modeling many-body quantum systems, tensor networks have emerged as a promising class of models for solving machine learning problems, notably in unsupervised generative learning. While possessing many desirable features arising from their quantum-inspired nature, tensor network generative models have previously been largely restricted to binary or categorical data, limiting their utility in real-world modeling problems. We overcome this by introducing a new family of tensor network generative models for continuous data, which are capable of learning from distributions containing continuous random variables. We develop our method in the setting of matrix product states, first deriving a universal expressivity theorem proving the ability of this model family to approximate any reasonably smooth probability density function with arbitrary precision. We then benchmark the performance of this model on several synthetic and real-world datasets, finding that the model 
    
[^7]: 微分水平空间中的格点逼近

    Lattice Approximations in Wasserstein Space. (arXiv:2310.09149v1 [stat.ML])

    [http://arxiv.org/abs/2310.09149](http://arxiv.org/abs/2310.09149)

    本论文研究了在Wasserstein空间中通过离散和分段常数测度进行的结构逼近方法。结果表明，对于满秩的格点按比例缩放后得到的Voronoi分割逼近的测度误差是$O(h)$，逼近的$N$项误差为$O(N^{-\frac1d})$，并且可以推广到非紧支撑测度。

    

    我们考虑在Wasserstein空间$W_p(\mathbb{R}^d)$中通过离散和分段常数测度来对测度进行结构逼近。我们证明，如果一个满秩的格点$\Lambda$按照$h\in(0,1]$的比例进行缩放，那么基于$h\Lambda$的Voronoi分割得到的测度逼近是$O(h)$，不论$d$或$p$的取值。之后，我们使用覆盖论证证明，对于紧支撑的测度的$N$项逼近是$O(N^{-\frac1d})$，这与最优量化器和经验测度逼近在大多数情况下已知的速率相匹配。最后，我们将这些结果推广到非紧支撑测度，要求其具有足够的衰减性质。

    We consider structured approximation of measures in Wasserstein space $W_p(\mathbb{R}^d)$ for $p\in[1,\infty)$ by discrete and piecewise constant measures based on a scaled Voronoi partition of $\mathbb{R}^d$. We show that if a full rank lattice $\Lambda$ is scaled by a factor of $h\in(0,1]$, then approximation of a measure based on the Voronoi partition of $h\Lambda$ is $O(h)$ regardless of $d$ or $p$. We then use a covering argument to show that $N$-term approximations of compactly supported measures is $O(N^{-\frac1d})$ which matches known rates for optimal quantizers and empirical measure approximation in most instances. Finally, we extend these results to noncompactly supported measures with sufficient decay.
    
[^8]: 最可能的共同原因

    The most likely common cause. (arXiv:2306.17557v1 [physics.data-an])

    [http://arxiv.org/abs/2306.17557](http://arxiv.org/abs/2306.17557)

    对于因果不充分的情况下的共同原因问题，我们使用广义最大似然方法来识别共同原因C，与最大熵原则密切相关。对于两个二元对称变量的研究揭示了类似于二阶相变的条件概率非解析行为。

    

    对于两个随机变量A和B的共同原因原则在因果不充分的情况下进行了研究，当它们的共同原因C被认为已经存在，但只观测到了A和B的联合概率。因此，C不能被唯一确定（潜在混杂因子问题）。我们展示了广义最大似然方法可以应用于这种情况，并且允许识别与共同原因原则一致的C。它与最大熵原则密切相关。对两个二元对称变量的研究揭示了条件概率的非解析行为，类似于二阶相变。这发生在观察到的概率分布从相关到反相关的过渡期间。讨论了广义似然方法与其他方法（如预测似然和最小共同原因熵）之间的关系。

    The common cause principle for two random variables $A$ and $B$ is examined in the case of causal insufficiency, when their common cause $C$ is known to exist, but only the joint probability of $A$ and $B$ is observed. As a result, $C$ cannot be uniquely identified (the latent confounder problem). We show that the generalized maximum likelihood method can be applied to this situation and allows identification of $C$ that is consistent with the common cause principle. It closely relates to the maximum entropy principle. Investigation of the two binary symmetric variables reveals a non-analytic behavior of conditional probabilities reminiscent of a second-order phase transition. This occurs during the transition from correlation to anti-correlation in the observed probability distribution. The relation between the generalized likelihood approach and alternative methods, such as predictive likelihood and the minimum common cause entropy, is discussed. The consideration of the common cause
    
[^9]: 使用最小生成树进行聚类：能有多好？

    Clustering with minimum spanning trees: How good can it be?. (arXiv:2303.05679v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2303.05679](http://arxiv.org/abs/2303.05679)

    本文研究了使用最小生成树（MST）进行分区数据聚类任务的意义程度，并发现MST方法在总体上具有很强的竞争力。此外，通过回顾、研究、扩展和推广现有的MST-based划分方案，我们提出了一些新的和值得注意的方法。总体上，Genie和信息论方法往往优于其他非MST算法，在某些情况下MST方法可能不如其他算法。

    

    最小生成树（MST）在许多模式识别任务中可以提供方便的数据集表示，并且计算相对较快。本文中，我们量化了MST在低维空间的分区数据聚类任务中的意义程度。通过识别最佳（oracle）算法与大量基准数据的专家标签之间的一致性上限，我们发现MST方法在总体上具有很强的竞争力。接下来，我们不是提出另一个只在有限的示例上表现良好的算法，而是回顾、研究、扩展和推广现有的最新MST-based划分方案。这导致了一些新的和值得注意的方法。总体上，Genie和信息论方法往往优于非MST算法，如k-means，高斯混合，谱聚类，Birch，基于密度和经典层次聚类程序。尽管如此，我们还是发现MST方法在某些情况下可能不如其他算法。

    Minimum spanning trees (MSTs) provide a convenient representation of datasets in numerous pattern recognition activities. Moreover, they are relatively fast to compute. In this paper, we quantify the extent to which they can be meaningful in partitional data clustering tasks in low-dimensional spaces. By identifying the upper bounds for the agreement between the best (oracle) algorithm and the expert labels from a large battery of benchmark data, we discover that MST methods are overall very competitive. Next, instead of proposing yet another algorithm that performs well on a limited set of examples, we review, study, extend, and generalise existing, state-of-the-art MST-based partitioning schemes. This leads to a few new and noteworthy approaches. Overall, Genie and the information-theoretic methods often outperform the non-MST algorithms such as k-means, Gaussian mixtures, spectral clustering, Birch, density-based, and classical hierarchical agglomerative procedures. Nevertheless, we
    
[^10]: 通过子序列相似性生成序列：理论及其在无人机识别中的应用

    Sequence Generation via Subsequence Similarity: Theory and Application to UAV Identification. (arXiv:2301.08403v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.08403](http://arxiv.org/abs/2301.08403)

    本文探究了一种单次生成模型的多样性，主要聚焦于子序列相似性如何影响整个序列相似性，并通过生成子序列相似的序列来增强数据集。

    

    生成人工合成序列的能力在广泛的应用中至关重要，而深度学习架构和生成框架的最新进展已经极大地促进了这一过程。本文使用一种单次生成模型来采样，通过相似性生成子序列，并证明了子序列相似性对整个序列相似性的影响，给出了相应的界限。我们使用一种一次性生成模型来从单个序列的范围内取样，并生成子序列相似的序列，证明了数据集增强方面的实用性。

    The ability to generate synthetic sequences is crucial for a wide range of applications, and recent advances in deep learning architectures and generative frameworks have greatly facilitated this process. Particularly, unconditional one-shot generative models constitute an attractive line of research that focuses on capturing the internal information of a single image or video to generate samples with similar contents. Since many of those one-shot models are shifting toward efficient non-deep and non-adversarial approaches, we examine the versatility of a one-shot generative model for augmenting whole datasets. In this work, we focus on how similarity at the subsequence level affects similarity at the sequence level, and derive bounds on the optimal transport of real and generated sequences based on that of corresponding subsequences. We use a one-shot generative model to sample from the vicinity of individual sequences and generate subsequence-similar ones and demonstrate the improvem
    
[^11]: 规范化聚类准确度：一种非对称的外部聚类有效度量

    Normalised clustering accuracy: An asymmetric external cluster validity measure. (arXiv:2209.02935v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.02935](http://arxiv.org/abs/2209.02935)

    本文提出了一种非对称的外部聚类有效度量方法，旨在区分不同任务类型上表现良好和系统性表现不佳的聚类算法。与传统的内部度量不同，该方法利用参考真实分组进行评估，并弥补了现有方法在最坏情况下的误差。

    

    没有一个最好的聚类算法，我们仍然希望能够区分出在某些任务类型上表现良好和系统性表现不佳的方法。传统上，聚类算法使用内部或外部有效度量进行评估。内部度量量化所得分区的不同方面，例如，簇紧密度的平均程度或点的可分离性。然而，它们的有效性是有问题的，因为它们促使的聚类有时可能是无意义的。另一方面，外部度量将算法的输出与由专家提供的参考真实分组进行比较。在本文中，我们认为常用的经典分区相似性评分，例如规范化互信息、Fowlkes-Mallows或调整兰德指数，缺少一些可取的属性，例如，它们不能正确识别最坏情况，也不易解释。

    There is no, nor will there ever be, single best clustering algorithm, but we would still like to be able to distinguish between methods which work well on certain task types and those that systematically underperform. Clustering algorithms are traditionally evaluated using either internal or external validity measures. Internal measures quantify different aspects of the obtained partitions, e.g., the average degree of cluster compactness or point separability. Yet, their validity is questionable, because the clusterings they promote can sometimes be meaningless. External measures, on the other hand, compare the algorithms' outputs to the reference, ground truth groupings that are provided by experts. In this paper, we argue that the commonly-used classical partition similarity scores, such as the normalised mutual information, Fowlkes-Mallows, or adjusted Rand index, miss some desirable properties, e.g., they do not identify worst-case scenarios correctly or are not easily interpretab
    
[^12]: 随机响应私有置信集的非参数扩展

    Nonparametric extensions of randomized response for private confidence sets. (arXiv:2202.08728v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2202.08728](http://arxiv.org/abs/2202.08728)

    本文提出了一种随机响应机制的扩展，可在局部差分隐私约束下计算非参数非渐进统计推断，由此得到的结果可用于生成效率高的置信区间和时间均匀置信序列。利用这些方法可以进行实证研究并产生私有模拟。

    

    本文提出了一种在局部差分隐私（LDP）约束下执行非参数、非渐进统计推断的方法，用于计算具有均值$\mu^\star$的有界观测$(X_1,\dots,X_n)$的置信区间（CI）和时间均匀置信序列（CS），当只有访问私有数据$(Z_1,\dots,Z_n)$时。为了实现这一点，我们引入了一个非参数的、顺序交互的 Warner 的著名“随机响应”机制的推广，为任意有界随机变量满足 LDP，并提供 CIs 和 CSs，用于访问所得私有化的观测值的均值。例如，我们的结果在固定时间和时间均匀区域都产生了 Hoeffding 不等式的私有模拟。我们将这些 Hoeffding  类型的 CSs 扩展到捕获时间变化（非平稳）的均值，最后说明了如何利用这些方法进行实证。

    This work derives methods for performing nonparametric, nonasymptotic statistical inference for population means under the constraint of local differential privacy (LDP). Given bounded observations $(X_1, \dots, X_n)$ with mean $\mu^\star$ that are privatized into $(Z_1, \dots, Z_n)$, we present confidence intervals (CI) and time-uniform confidence sequences (CS) for $\mu^\star$ when only given access to the privatized data. To achieve this, we introduce a nonparametric and sequentially interactive generalization of Warner's famous ``randomized response'' mechanism, satisfying LDP for arbitrary bounded random variables, and then provide CIs and CSs for their means given access to the resulting privatized observations. For example, our results yield private analogues of Hoeffding's inequality in both fixed-time and time-uniform regimes. We extend these Hoeffding-type CSs to capture time-varying (non-stationary) means, and conclude by illustrating how these methods can be used to conduct
    

