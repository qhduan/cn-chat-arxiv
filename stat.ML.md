# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [VITS : Variational Inference Thomson Sampling for contextual bandits.](http://arxiv.org/abs/2307.10167) | VITS是一种基于高斯变分推理的新算法，用于情境背离问题的汤普森抽样。它提供了强大的后验近似，计算效率高，并且在线性情境背离问题中达到与传统TS相同阶数的次线性遗憾上界。 |
| [^2] | [Rethinking Backdoor Attacks.](http://arxiv.org/abs/2307.10163) | 本文重新思考了后门攻击问题，发现在没有关于训练数据分布的结构信息的情况下，后门攻击与数据中自然产生的特征是不可区分的，因此难以检测。作者还重新审视现有的抵御后门攻击的方法，并探索了一种关于后门攻击的替代视角。 |
| [^3] | [Pattern Recovery in Penalized and Thresholded Estimation and its Geometry.](http://arxiv.org/abs/2307.10158) | 我们提出了一种惩罚化和阈值化估计的模式恢复方法，并定义了模式和恢复条件。对于LASSO，无噪声恢复条件和互不表示条件起到了相同的作用。 |
| [^4] | [Curvature-based Clustering on Graphs.](http://arxiv.org/abs/2307.10155) | 本研究通过利用图的几何性质，实现了基于离散Ricci曲率的聚类算法，可以识别图结构中的密集连接子结构，包括单成员社区和混合成员社区，以及在线图上的社区检测，并提供了实验证据支持。 |
| [^5] | [Memory Efficient And Minimax Distribution Estimation Under Wasserstein Distance Using Bayesian Histograms.](http://arxiv.org/abs/2307.10099) | 本文研究了在Wasserstein距离下的贝叶斯直方图的内存高效性和最小化分布估计问题，并证明了当维度小于2v时，只需要使用n^{d/2v}个bins就可以达到最小化率最优性，相较于现有算法的内存占用可以减少多项式因子$n^{1 - d/2v}$，同时构建后验均值直方图和后验本身的复杂度可以超线性增加。 |
| [^6] | [Revisiting invariances and introducing priors in Gromov-Wasserstein distances.](http://arxiv.org/abs/2307.10093) | 本文提出了一种新的基于最优传输的距离，增强的Gromov-Wasserstein，它在Gromov-Wasserstein距离的基础上引入了对变换刚度的控制和特征对齐，并应用于单细胞多组学和迁移学习任务中，展示了其在机器学习中的实用性和改进性能。 |
| [^7] | [A Dual Formulation for Probabilistic Principal Component Analysis.](http://arxiv.org/abs/2307.10078) | 本文探讨了概率主成分分析在希尔伯特空间中的双重表述方法，并发展了适用于核方法的生成框架。作者通过实验证明了该方法能兼容核主成分分析，并在虚拟和真实数据集上进行了验证。 |
| [^8] | [Entropy regularization in probabilistic clustering.](http://arxiv.org/abs/2307.10065) | 该论文提出了一种新颖的贝叶斯聚类配置估计器，通过熵正则化后处理过程，减少了稀疏填充簇的数量并增强了可解释性。 |
| [^9] | [Convergence Guarantees for Stochastic Subgradient Methods in Nonsmooth Nonconvex Optimization.](http://arxiv.org/abs/2307.10053) | 本文研究了非平滑非凸优化中随机次梯度方法的收敛性质，并提出了一种新的框架，证明了其在单时间尺度和双时间尺度情况下的全局收敛性，包括了多种已知的SGD类型方法。对于有限和形式的目标函数，证明了这些方法能够在随机选择的步长和初始点上找到Clarke稳定点。 |
| [^10] | [FaIRGP: A Bayesian Energy Balance Model for Surface Temperatures Emulation.](http://arxiv.org/abs/2307.10052) | FaIRGP是一种新的数据驱动代理模型，它满足能量平衡模型的物理温度响应方程，同时具备了从观测中学习和进行推断的能力。 |
| [^11] | [Impatient Bandits: Optimizing for the Long-Term Without Delay.](http://arxiv.org/abs/2307.09943) | 这里是中文总结出的一句话要点：这篇论文研究了在推荐系统中提高用户长期满意度的问题，通过开发一个预测延迟奖励的模型和设计一个利用该模型的赌博算法来解决了通过测量短期代理奖励反映实际长期目标不完美的挑战。 |
| [^12] | [Spuriosity Didn't Kill the Classifier: Using Invariant Predictions to Harness Spurious Features.](http://arxiv.org/abs/2307.09933) | 本研究通过理论证明和算法提出，展示了在没有标签的情况下如何利用不稳定特征来提高分类器的性能。 |
| [^13] | [Manifold Learning with Sparse Regularised Optimal Transport.](http://arxiv.org/abs/2307.09816) | 这篇论文介绍了一种利用稀疏正则最优传输进行流形学习的方法，该方法构建了一个稀疏自适应的亲和矩阵，并在连续极限下与拉普拉斯型算子一致。 |
| [^14] | [The Connection Between R-Learning and Inverse-Variance Weighting for Estimation of Heterogeneous Treatment Effects.](http://arxiv.org/abs/2307.09700) | R-Learning在估计条件平均治疗效果时采用了逆变数加权的形式来稳定回归，并简化了偏差项。 |
| [^15] | [Multi-view self-supervised learning for multivariate variable-channel time series.](http://arxiv.org/abs/2307.09614) | 本论文提出了一种多视角自监督学习方法，用于处理多变量通道时间序列数据，在不同数据集之间进行迁移学习。通过预训练和微调，结合传递神经网络和TS2Vec损失，该方法在大多数设置下表现优于其他方法。 |
| [^16] | [Sequential Monte Carlo Learning for Time Series Structure Discovery.](http://arxiv.org/abs/2307.09607) | 本文提出了一种顺序蒙特卡洛学习的方法，用于自动发现复杂时间序列数据的准确模型。在实验中显示，该方法相对于之前的方法，具有较快的运行速度并能够发现合理的模型结构。 |
| [^17] | [Self-Compatibility: Evaluating Causal Discovery without Ground Truth.](http://arxiv.org/abs/2307.09552) | 本论文提出了一种在没有基准数据的情况下评估因果发现方法的新方法，通过在不同变量子集上学习的因果图之间的兼容性检测，来伪证因果关系的推断正确性。 |
| [^18] | [BOF-UCB: A Bayesian-Optimistic Frequentist Algorithm for Non-Stationary Contextual Bandits.](http://arxiv.org/abs/2307.03587) | BOF-UCB是一种用于非平稳环境下的背景线性赌博机的贝叶斯优化频率算法，其结合了贝叶斯和频率学派原则，提高了在动态环境中的性能。它利用贝叶斯更新推断后验分布，并使用频率学派方法计算上界信心界以平衡探索和开发。实验证明，BOF-UCB优于现有方法，是非平稳环境中顺序决策的有前途的解决方案。 |
| [^19] | [Multi-class Graph Clustering via Approximated Effective $p$-Resistance.](http://arxiv.org/abs/2306.08617) | 本文提出了一种近似计算有效$p$-阻抗并将其应用于多类图聚类，该方法可以通过参数$p$偏向于具有高内部连通性或者更小的簇内顶点之间的最短路径距离的聚类。 |
| [^20] | [Probabilistic Distance-Based Outlier Detection.](http://arxiv.org/abs/2305.09446) | 本文提出了一种将距离法异常检测分数转化为可解释的概率估计的通用方法，该方法使用与其他数据点的距离建模距离概率分布，将距离法异常检测分数转换为异常概率，提高了正常点和异常点之间的对比度，而不会影响检测性能。 |
| [^21] | [Critical Points and Convergence Analysis of Generative Deep Linear Networks Trained with Bures-Wasserstein Loss.](http://arxiv.org/abs/2303.03027) | 本文使用布雷-瓦瑟斯坦距离训练协方差矩阵的深度矩阵分解模型，并在有限秩矩阵空间内表征关键点和最小化问题，最终确定了梯度下降算法的收敛性。 |
| [^22] | [Generalization Error Bounds for Noisy, Iterative Algorithms via Maximal Leakage.](http://arxiv.org/abs/2302.14518) | 通过最大泄露分析噪声迭代算法的泛化误差界限，证明了如果更新函数在L2-范数下有界且加性噪声为各向同性高斯噪声，则可以得到一个半封闭形式下的最大泄露上界，同时展示了更新函数的假设如何影响噪声的最优选择。 |
| [^23] | [CO-BED: Information-Theoretic Contextual Optimization via Bayesian Experimental Design.](http://arxiv.org/abs/2302.14015) | CO-BED是一个通用的、与模型无关的框架，用于通过贝叶斯实验设计的信息理论来进行上下文优化。它采用黑箱变分方法同时估计和优化设计，可以适应离散动作，并在多个实验中展示出竞争性能。 |
| [^24] | [Sequential Kernelized Independence Testing.](http://arxiv.org/abs/2212.07383) | 该论文介绍了顺序核独立性测试的方法，以解决传统批量测试在流数据上的问题，实现了根据任务复杂性自适应调整样本大小，并在收集新数据后持续监测和控制误报率。 |
| [^25] | [Alpha-divergence Variational Inference Meets Importance Weighted Auto-Encoders: Methodology and Asymptotics.](http://arxiv.org/abs/2210.06226) | 本文提出了VR-IWAE下界，该下界是IWAE下界的推广，采用无偏梯度估计器能够实现与VR下界相同的随机梯度下降过程，对该下界进行了理论分析，揭示了其优势和不足，并通过示例验证了理论观点。 |
| [^26] | [Prediction intervals for neural network models using weighted asymmetric loss functions.](http://arxiv.org/abs/2210.04318) | 本论文提出了一种使用加权不对称损失函数的方法，生成可靠的预测区间，适用于复杂的机器学习情境，可扩展为参数化函数的PI预测。 |
| [^27] | [Cooperation in the Latent Space: The Benefits of Adding Mixture Components in Variational Autoencoders.](http://arxiv.org/abs/2209.15514) | 本研究展示了在变分自编码器中添加混合成分的好处，并证明了混合成分的增加能够提高其在图像和单细胞数据集上的潜在表示能力。这表明使用混合VAE是获取更灵活变分逼近的标准方法。 |
| [^28] | [The Value of Out-of-Distribution Data.](http://arxiv.org/abs/2208.10967) | 不同分布的数据可以对任务的泛化误差产生非单调的影响，使用少量不同分布的数据进行训练是有价值的。 |
| [^29] | [Learning from time-dependent streaming data with online stochastic algorithms.](http://arxiv.org/abs/2205.12549) | 本文研究了处理时间相关的流式数据的在线随机算法，并通过非渐进分析建立了新颖的启发式算法，加速收敛。实验证明时间变化的小批量SGD方法可以打破依赖结构，有偏倚的SGD方法具有与无偏倚方法相当的性能，并且使用Polyak-Ruppert平均化方法能够加快随机优化算法的收敛。 |
| [^30] | [Finite-Time Analysis of Natural Actor-Critic for POMDPs.](http://arxiv.org/abs/2202.09753) | 本文分析了部分观察的马尔科夫决策过程（POMDPs）下自然演员-评论家方法的有限时间特性，并对使用有限状态控制器产生的错误进行了明确的表征。 |
| [^31] | [Weisfeiler and Leman go Machine Learning: The Story so far.](http://arxiv.org/abs/2112.09992) | Weisfeiler-Leman算法被广泛应用于处理图和关系数据。本文全面介绍了该算法在监督学习中的应用，包括理论背景、扩展、与等变神经网格的联系、并列出了当前应用和未来研究方向。 |
| [^32] | [MixPath: A Unified Approach for One-shot Neural Architecture Search.](http://arxiv.org/abs/2001.05887) | 本论文提出了一种名为MixPath的统一的一次性神经架构搜索方法，通过训练一次性的多路径超网络来准确评估候选架构。采用一种新颖的机制称为Shadow Batch Normalization（SBN）来解决多路径结构的特征差异问题，稳定优化并提高排名性能。 |

# 详细

[^1]: VITS: 基于变分推理的汤普森抽样用于情境背离问题的算法

    VITS : Variational Inference Thomson Sampling for contextual bandits. (arXiv:2307.10167v1 [stat.ML])

    [http://arxiv.org/abs/2307.10167](http://arxiv.org/abs/2307.10167)

    VITS是一种基于高斯变分推理的新算法，用于情境背离问题的汤普森抽样。它提供了强大的后验近似，计算效率高，并且在线性情境背离问题中达到与传统TS相同阶数的次线性遗憾上界。

    

    本文介绍并分析了一种用于情境背离问题的汤普森抽样（TS）算法的变体。传统的TS算法在每轮需要从当前的后验分布中抽样，而这通常是难以计算的。为了解决这个问题，可以使用近似推理技术并提供接近后验分布的样本。然而，当前的近似技术要么估计不准确（拉普拉斯近似），要么计算开销较大（MCMC方法，集成抽样...）。在本文中，我们提出了一种新的算法，基于高斯变分推理的变分推理汤普森抽样（VITS）。这种方法提供了强大的后验近似，并且容易从中抽样，而且计算效率高，是TS的理想选择。此外，我们还证明了在线性情境背离问题中，VITS实现了与传统TS相同阶数的次线性遗憾上界，与维度和回合数成正比。

    In this paper, we introduce and analyze a variant of the Thompson sampling (TS) algorithm for contextual bandits. At each round, traditional TS requires samples from the current posterior distribution, which is usually intractable. To circumvent this issue, approximate inference techniques can be used and provide samples with distribution close to the posteriors. However, current approximate techniques yield to either poor estimation (Laplace approximation) or can be computationally expensive (MCMC methods, Ensemble sampling...). In this paper, we propose a new algorithm, Varational Inference Thompson sampling VITS, based on Gaussian Variational Inference. This scheme provides powerful posterior approximations which are easy to sample from, and is computationally efficient, making it an ideal choice for TS. In addition, we show that VITS achieves a sub-linear regret bound of the same order in the dimension and number of round as traditional TS for linear contextual bandit. Finally, we 
    
[^2]: 重新思考后门攻击

    Rethinking Backdoor Attacks. (arXiv:2307.10163v1 [cs.CR])

    [http://arxiv.org/abs/2307.10163](http://arxiv.org/abs/2307.10163)

    本文重新思考了后门攻击问题，发现在没有关于训练数据分布的结构信息的情况下，后门攻击与数据中自然产生的特征是不可区分的，因此难以检测。作者还重新审视现有的抵御后门攻击的方法，并探索了一种关于后门攻击的替代视角。

    

    在后门攻击中，对手会将恶意构造的后门示例插入训练集中，使得生成的模型容易受到操纵。防御这种攻击通常涉及将这些插入的示例视为训练集中的异常值，并使用鲁棒统计学的技术来检测和删除它们。在这项工作中，我们提出了一种不同的解决后门攻击问题的方法。具体而言，我们展示了在没有关于训练数据分布的结构信息的情况下，后门攻击与数据中自然产生的特征是不可区分的--因此无法在一般意义上“检测”它们。然后，根据这一观察，我们重新审视现有的抵御后门攻击的方法，并表征它们所做出的（常常是潜在的）假设以及它们依赖的假设。最后，我们探索了一种关于后门攻击的替代视角：假设这些攻击对应于训练数据中最强的特征。

    In a backdoor attack, an adversary inserts maliciously constructed backdoor examples into a training set to make the resulting model vulnerable to manipulation. Defending against such attacks typically involves viewing these inserted examples as outliers in the training set and using techniques from robust statistics to detect and remove them.  In this work, we present a different approach to the backdoor attack problem. Specifically, we show that without structural information about the training data distribution, backdoor attacks are indistinguishable from naturally-occurring features in the data--and thus impossible to "detect" in a general sense. Then, guided by this observation, we revisit existing defenses against backdoor attacks and characterize the (often latent) assumptions they make and on which they depend. Finally, we explore an alternative perspective on backdoor attacks: one that assumes these attacks correspond to the strongest feature in the training data. Under this a
    
[^3]: 惩罚化和阈值化估计中的模式恢复及其几何

    Pattern Recovery in Penalized and Thresholded Estimation and its Geometry. (arXiv:2307.10158v1 [math.ST])

    [http://arxiv.org/abs/2307.10158](http://arxiv.org/abs/2307.10158)

    我们提出了一种惩罚化和阈值化估计的模式恢复方法，并定义了模式和恢复条件。对于LASSO，无噪声恢复条件和互不表示条件起到了相同的作用。

    

    我们考虑惩罚估计的框架，其中惩罚项由实值的多面体规范给出，其中包括诸如LASSO（以及其许多变体如广义LASSO）、SLOPE、OSCAR、PACS等方法。每个估计器可以揭示未知参数向量的不同结构或“模式”。我们定义了基于次微分的模式的一般概念，并形式化了一种衡量其复杂性的方法。对于模式恢复，我们提供了一个特定模式以正概率被该过程检测到的最小条件，即所谓的可达性条件。利用我们的方法，我们还引入了更强的无噪声恢复条件。对于LASSO，众所周知，互不表示条件是使模式恢复的概率大于1/2所必需的，并且我们展示了无噪声恢复起到了完全相同的作用，从而扩展和统一了互不表示条件。

    We consider the framework of penalized estimation where the penalty term is given by a real-valued polyhedral gauge, which encompasses methods such as LASSO (and many variants thereof such as the generalized LASSO), SLOPE, OSCAR, PACS and others. Each of these estimators can uncover a different structure or ``pattern'' of the unknown parameter vector. We define a general notion of patterns based on subdifferentials and formalize an approach to measure their complexity. For pattern recovery, we provide a minimal condition for a particular pattern to be detected by the procedure with positive probability, the so-called accessibility condition. Using our approach, we also introduce the stronger noiseless recovery condition. For the LASSO, it is well known that the irrepresentability condition is necessary for pattern recovery with probability larger than $1/2$ and we show that the noiseless recovery plays exactly the same role, thereby extending and unifying the irrepresentability conditi
    
[^4]: 基于曲率的图聚类

    Curvature-based Clustering on Graphs. (arXiv:2307.10155v1 [cs.SI])

    [http://arxiv.org/abs/2307.10155](http://arxiv.org/abs/2307.10155)

    本研究通过利用图的几何性质，实现了基于离散Ricci曲率的聚类算法，可以识别图结构中的密集连接子结构，包括单成员社区和混合成员社区，以及在线图上的社区检测，并提供了实验证据支持。

    

    无监督节点聚类（或社区检测）是经典的图学习任务。本文研究利用图的几何性质来识别密集连接的子结构以形成聚类或社区的算法。我们的方法实现了离散Ricci曲率及其相关的几何流，通过这些流，图的边权重演化以揭示其社区结构。我们考虑了几种离散曲率概念，并分析了相应算法的实用性。与之前的文献相比，我们不仅研究了单成员社区检测，即每个节点只属于一个社区，还研究了混合成员社区检测，即社区可能重叠。对于后者，我们认为在线图上执行社区检测有益处，即图的对偶图。我们提供了我们基于曲率的聚类算法的理论和实证证据。此外，我们还提供了几个重新实现和评估的实验。

    Unsupervised node clustering (or community detection) is a classical graph learning task. In this paper, we study algorithms, which exploit the geometry of the graph to identify densely connected substructures, which form clusters or communities. Our method implements discrete Ricci curvatures and their associated geometric flows, under which the edge weights of the graph evolve to reveal its community structure. We consider several discrete curvature notions and analyze the utility of the resulting algorithms. In contrast to prior literature, we study not only single-membership community detection, where each node belongs to exactly one community, but also mixed-membership community detection, where communities may overlap. For the latter, we argue that it is beneficial to perform community detection on the line graph, i.e., the graph's dual. We provide both theoretical and empirical evidence for the utility of our curvature-based clustering algorithms. In addition, we give several re
    
[^5]: 内存高效和最小化分布估计下的Wasserstein距离使用贝叶斯直方图

    Memory Efficient And Minimax Distribution Estimation Under Wasserstein Distance Using Bayesian Histograms. (arXiv:2307.10099v1 [math.ST])

    [http://arxiv.org/abs/2307.10099](http://arxiv.org/abs/2307.10099)

    本文研究了在Wasserstein距离下的贝叶斯直方图的内存高效性和最小化分布估计问题，并证明了当维度小于2v时，只需要使用n^{d/2v}个bins就可以达到最小化率最优性，相较于现有算法的内存占用可以减少多项式因子$n^{1 - d/2v}$，同时构建后验均值直方图和后验本身的复杂度可以超线性增加。

    

    我们研究了$[0,1]^d$上的Wasserstein $W_v, 1 \leq v < \infty$距离下的贝叶斯直方图分布估计。我们新的发现是，当$d < 2v$时，直方图具有特殊的"内存高效"属性，即在参考样本大小$n$的情况下，需要顺序$n^{d/2v}$的bins来获得最小化率最优性。这个结果对于后验均值直方图和相对于后验收缩来说是成立的：对于Borel概率测度和一些光滑密度类别。所获得的内存占用优势比现有的最小化最优算法多出了$n$的多项式因子；例如，与Borel概率测度类别中的经验测度（最小化估计器）相比，内存占用可以减少$n^{1 - d/2v}$倍。此外，构建后验均值直方图和后验本身可以超线性地进行$n$。

    We study Bayesian histograms for distribution estimation on $[0,1]^d$ under the Wasserstein $W_v, 1 \leq v < \infty$ distance in the i.i.d sampling regime. We newly show that when $d < 2v$, histograms possess a special \textit{memory efficiency} property, whereby in reference to the sample size $n$, order $n^{d/2v}$ bins are needed to obtain minimax rate optimality. This result holds for the posterior mean histogram and with respect to posterior contraction: under the class of Borel probability measures and some classes of smooth densities. The attained memory footprint overcomes existing minimax optimal procedures by a polynomial factor in $n$; for example an $n^{1 - d/2v}$ factor reduction in the footprint when compared to the empirical measure, a minimax estimator in the Borel probability measure class. Additionally constructing both the posterior mean histogram and the posterior itself can be done super--linearly in $n$. Due to the popularity of the $W_1,W_2$ metrics and the covera
    
[^6]: 重新审视不变性并引入先验知识在Gromov-Wasserstein距离中

    Revisiting invariances and introducing priors in Gromov-Wasserstein distances. (arXiv:2307.10093v1 [cs.LG])

    [http://arxiv.org/abs/2307.10093](http://arxiv.org/abs/2307.10093)

    本文提出了一种新的基于最优传输的距离，增强的Gromov-Wasserstein，它在Gromov-Wasserstein距离的基础上引入了对变换刚度的控制和特征对齐，并应用于单细胞多组学和迁移学习任务中，展示了其在机器学习中的实用性和改进性能。

    

    由于其能够比较度量空间中的测度并且对等度变换具有不变性，Gromov-Wasserstein距离在机器学习中有很多应用。然而，在某些应用中，这种不变性可能过于灵活而不可取。此外，Gromov-Wasserstein距离仅考虑输入数据集中的成对样本相似性，而忽略原始特征表示。我们提出了一种新的基于最优传输的距离，称为增强的Gromov-Wasserstein，它允许对变换的刚度有一定控制。它还结合了特征对齐，使我们能够更好地利用输入数据上的先验知识以提高性能。我们提出了对所提出的度量的理论洞察力。然后，我们展示了它在单细胞多组学对齐任务和机器学习中的迁移学习场景中的实用性。

    Gromov-Wasserstein distance has found many applications in machine learning due to its ability to compare measures across metric spaces and its invariance to isometric transformations. However, in certain applications, this invariance property can be too flexible, thus undesirable. Moreover, the Gromov-Wasserstein distance solely considers pairwise sample similarities in input datasets, disregarding the raw feature representations. We propose a new optimal transport-based distance, called Augmented Gromov-Wasserstein, that allows for some control over the level of rigidity to transformations. It also incorporates feature alignments, enabling us to better leverage prior knowledge on the input data for improved performance. We present theoretical insights into the proposed metric. We then demonstrate its usefulness for single-cell multi-omic alignment tasks and a transfer learning scenario in machine learning.
    
[^7]: 概率主成分分析的双重表述

    A Dual Formulation for Probabilistic Principal Component Analysis. (arXiv:2307.10078v1 [cs.LG])

    [http://arxiv.org/abs/2307.10078](http://arxiv.org/abs/2307.10078)

    本文探讨了概率主成分分析在希尔伯特空间中的双重表述方法，并发展了适用于核方法的生成框架。作者通过实验证明了该方法能兼容核主成分分析，并在虚拟和真实数据集上进行了验证。

    

    本文中，我们在希尔伯特空间中对概率主成分分析进行了表述，并展示了最优解在对偶空间中的表示。这使得我们能够发展出一种适用于核方法的生成框架。此外，我们展示了它如何吸纳了核主成分分析，并在一个虚拟数据集和一个真实数据集上进行了演示。

    In this paper, we characterize Probabilistic Principal Component Analysis in Hilbert spaces and demonstrate how the optimal solution admits a representation in dual space. This allows us to develop a generative framework for kernel methods. Furthermore, we show how it englobes Kernel Principal Component Analysis and illustrate its working on a toy and a real dataset.
    
[^8]: 概率聚类中的熵正则化

    Entropy regularization in probabilistic clustering. (arXiv:2307.10065v1 [stat.ME])

    [http://arxiv.org/abs/2307.10065](http://arxiv.org/abs/2307.10065)

    该论文提出了一种新颖的贝叶斯聚类配置估计器，通过熵正则化后处理过程，减少了稀疏填充簇的数量并增强了可解释性。

    

    贝叶斯非参数混合模型被广泛用于聚类观测。然而，这种方法的一个主要缺点是估计的分区经常呈现不平衡的簇频率，其中只有少数几个簇占主导地位，而大量稀疏填充的簇存在。除非我们接受忽略一些观测和簇，否则这种特点会导致结果无法解释。将后验分布解释为惩罚似然度，我们展示了不平衡性可以解释为估计分区涉及的代价函数直接后果。根据我们的发现，我们提出了一种新颖的贝叶斯聚类配置估计器。该估计器等同于一种后处理过程，减少了稀疏填充簇的数量并增强了可解释性。该过程采取了贝叶斯估计的熵正则化形式。虽然在计算上很方便...

    Bayesian nonparametric mixture models are widely used to cluster observations. However, one major drawback of the approach is that the estimated partition often presents unbalanced clusters' frequencies with only a few dominating clusters and a large number of sparsely-populated ones. This feature translates into results that are often uninterpretable unless we accept to ignore a relevant number of observations and clusters. Interpreting the posterior distribution as penalized likelihood, we show how the unbalance can be explained as a direct consequence of the cost functions involved in estimating the partition. In light of our findings, we propose a novel Bayesian estimator of the clustering configuration. The proposed estimator is equivalent to a post-processing procedure that reduces the number of sparsely-populated clusters and enhances interpretability. The procedure takes the form of entropy-regularization of the Bayesian estimate. While being computationally convenient with res
    
[^9]: 非平滑非凸优化中随机次梯度方法的收敛性保证

    Convergence Guarantees for Stochastic Subgradient Methods in Nonsmooth Nonconvex Optimization. (arXiv:2307.10053v1 [math.OC])

    [http://arxiv.org/abs/2307.10053](http://arxiv.org/abs/2307.10053)

    本文研究了非平滑非凸优化中随机次梯度方法的收敛性质，并提出了一种新的框架，证明了其在单时间尺度和双时间尺度情况下的全局收敛性，包括了多种已知的SGD类型方法。对于有限和形式的目标函数，证明了这些方法能够在随机选择的步长和初始点上找到Clarke稳定点。

    

    本文研究了随机梯度下降（SGD）方法及其变种在训练由非平滑激活函数构建的神经网络中的收敛性质。我们提出了一种新颖的框架，为更新动量项和变量的步长分配了不同的时间尺度。在一些温和的条件下，我们证明了我们提出的框架在单时间尺度和双时间尺度情况下的全局收敛性。我们还证明了我们提出的框架包含了很多已知的SGD类型方法，包括heavy-ball SGD、SignSGD、Lion、normalized SGD和clipped SGD。此外，当目标函数采用有限和形式时，我们基于我们提出的框架证明了这些SGD类型方法的收敛性质。特别地，在温和的假设下，我们证明了这些SGD类型方法在随机选择的步长和初始点上能够找到目标函数的Clarke稳定点。

    In this paper, we investigate the convergence properties of the stochastic gradient descent (SGD) method and its variants, especially in training neural networks built from nonsmooth activation functions. We develop a novel framework that assigns different timescales to stepsizes for updating the momentum terms and variables, respectively. Under mild conditions, we prove the global convergence of our proposed framework in both single-timescale and two-timescale cases. We show that our proposed framework encompasses a wide range of well-known SGD-type methods, including heavy-ball SGD, SignSGD, Lion, normalized SGD and clipped SGD. Furthermore, when the objective function adopts a finite-sum formulation, we prove the convergence properties for these SGD-type methods based on our proposed framework. In particular, we prove that these SGD-type methods find the Clarke stationary points of the objective function with randomly chosen stepsizes and initial points under mild assumptions. Preli
    
[^10]: FaIRGP:面向表面温度模拟的贝叶斯能量平衡模型

    FaIRGP: A Bayesian Energy Balance Model for Surface Temperatures Emulation. (arXiv:2307.10052v1 [stat.AP])

    [http://arxiv.org/abs/2307.10052](http://arxiv.org/abs/2307.10052)

    FaIRGP是一种新的数据驱动代理模型，它满足能量平衡模型的物理温度响应方程，同时具备了从观测中学习和进行推断的能力。

    

    代理模型或简化复杂气候模型是通过最小的计算资源生成关键气候量预测的地球系统模型。通过时间序列建模或更先进的机器学习技术，基于数据驱动的代理模型已成为一个有希望的研究方向，能够生成与最先进的地球系统模型在视觉上难以区分的空间分辨率气候响应。然而，它们缺乏物理上的可解释性，限制了它们的广泛应用。本文介绍了一种新的数据驱动代理模型FaIRGP，它满足能量平衡模型的物理温度响应方程。结果是一个既能够从观测中学习又具有坚实物理基础的代理模型，可用于对气候系统进行推断。此外，我们的贝叶斯方法提供了一种有原则且数学可行的方法来进行推断。

    Emulators, or reduced complexity climate models, are surrogate Earth system models that produce projections of key climate quantities with minimal computational resources. Using time-series modeling or more advanced machine learning techniques, data-driven emulators have emerged as a promising avenue of research, producing spatially resolved climate responses that are visually indistinguishable from state-of-the-art Earth system models. Yet, their lack of physical interpretability limits their wider adoption. In this work, we introduce FaIRGP, a data-driven emulator that satisfies the physical temperature response equations of an energy balance model. The result is an emulator that (i) enjoys the flexibility of statistical machine learning models and can learn from observations, and (ii) has a robust physical grounding with interpretable parameters that can be used to make inference about the climate system. Further, our Bayesian approach allows a principled and mathematically tractabl
    
[^11]: 这里是翻译过的论文标题: 过去曾翻译《Impatient Bandits: Optimizing for the Long-Term Without Delay》

    Impatient Bandits: Optimizing for the Long-Term Without Delay. (arXiv:2307.09943v1 [cs.LG])

    [http://arxiv.org/abs/2307.09943](http://arxiv.org/abs/2307.09943)

    这里是中文总结出的一句话要点：这篇论文研究了在推荐系统中提高用户长期满意度的问题，通过开发一个预测延迟奖励的模型和设计一个利用该模型的赌博算法来解决了通过测量短期代理奖励反映实际长期目标不完美的挑战。

    

    这里是翻译过的论文摘要：推荐系统在在线平台上是一个普遍存在的功能。越来越多的情况下，它们明确地被任务为提高用户的长期满意度。在这个背景下，我们研究了一个内容探索任务，将其形式化为一个具有延迟奖励的多臂赌博问题。我们观察到，在选择学习信号时存在明显的权衡：等待完全的奖励可能需要几周时间，这会影响学习发生的速度，而测量短期代理奖励则不完美地反映了实际的长期目标。我们通过两个步骤来解决这个挑战。首先，我们开发了一个预测延迟奖励的模型，该模型可以整合迄今所获得的所有信息。通过贝叶斯滤波器组合完整的观察结果以及部分（短期或中期）的结果，从而得到概率信念。其次，我们设计了一个利用这个新的预测模型的赌博算法。该算法可以快速学习识别内容。

    Recommender systems are a ubiquitous feature of online platforms. Increasingly, they are explicitly tasked with increasing users' long-term satisfaction. In this context, we study a content exploration task, which we formalize as a multi-armed bandit problem with delayed rewards. We observe that there is an apparent trade-off in choosing the learning signal: Waiting for the full reward to become available might take several weeks, hurting the rate at which learning happens, whereas measuring short-term proxy rewards reflects the actual long-term goal only imperfectly. We address this challenge in two steps. First, we develop a predictive model of delayed rewards that incorporates all information obtained to date. Full observations as well as partial (short or medium-term) outcomes are combined through a Bayesian filter to obtain a probabilistic belief. Second, we devise a bandit algorithm that takes advantage of this new predictive model. The algorithm quickly learns to identify conten
    
[^12]: Spuriosity并没有导致分类器失败：利用不变的预测来利用虚假特征

    Spuriosity Didn't Kill the Classifier: Using Invariant Predictions to Harness Spurious Features. (arXiv:2307.09933v1 [cs.LG])

    [http://arxiv.org/abs/2307.09933](http://arxiv.org/abs/2307.09933)

    本研究通过理论证明和算法提出，展示了在没有标签的情况下如何利用不稳定特征来提高分类器的性能。

    

    为了避免在域外数据上的失败，最近的研究试图提取具有与标签在不同域之间稳定或不变关系的特征，舍弃与标签在不同域之间关系变化的"虚假"或不稳定特征。然而，不稳定特征常常携带关于标签的补充信息，如果在测试域中正确使用，可以提高性能。我们的主要贡献是显示在没有标签的情况下学习如何在测试域中使用这些不稳定特征是可能的。特别是，我们证明基于稳定特征的伪标签提供了足够的指导来做到这一点，前提是在给定标签的条件下，稳定特征和不稳定特征是条件独立的。基于这个理论洞见，我们提出了稳定特征增强（SFB）算法：(i)学习一个能够分离稳定特征和条件独立不稳定特征的预测器；(ii)使用稳定特征预测来适应测试域

    To avoid failures on out-of-distribution data, recent works have sought to extract features that have a stable or invariant relationship with the label across domains, discarding the "spurious" or unstable features whose relationship with the label changes across domains. However, unstable features often carry complementary information about the label that could boost performance if used correctly in the test domain. Our main contribution is to show that it is possible to learn how to use these unstable features in the test domain without labels. In particular, we prove that pseudo-labels based on stable features provide sufficient guidance for doing so, provided that stable and unstable features are conditionally independent given the label. Based on this theoretical insight, we propose Stable Feature Boosting (SFB), an algorithm for: (i) learning a predictor that separates stable and conditionally-independent unstable features; and (ii) using the stable-feature predictions to adapt t
    
[^13]: 用稀疏正则最优传输进行流形学习

    Manifold Learning with Sparse Regularised Optimal Transport. (arXiv:2307.09816v1 [stat.ML])

    [http://arxiv.org/abs/2307.09816](http://arxiv.org/abs/2307.09816)

    这篇论文介绍了一种利用稀疏正则最优传输进行流形学习的方法，该方法构建了一个稀疏自适应的亲和矩阵，并在连续极限下与拉普拉斯型算子一致。

    

    流形学习是现代统计学和数据科学中的一个核心任务。许多数据集（细胞、文档、图像、分子）可以被表示为嵌入在高维环境空间中的点云，然而数据固有的自由度通常远远少于环境维度的数量。检测数据嵌入的潜在流形是许多下游分析的先决条件。现实世界的数据集经常受到噪声观测和抽样的影响，因此提取关于潜在流形的信息是一个重大挑战。我们提出了一种利用对称版本的最优传输和二次正则化的流形学习方法，它构建了一个稀疏自适应的亲和矩阵，可以解释为双随机核归一化的推广。我们证明了在连续极限下产生的核与拉普拉斯型算子一致，并建立了该方法的健壮性。

    Manifold learning is a central task in modern statistics and data science. Many datasets (cells, documents, images, molecules) can be represented as point clouds embedded in a high dimensional ambient space, however the degrees of freedom intrinsic to the data are usually far fewer than the number of ambient dimensions. The task of detecting a latent manifold along which the data are embedded is a prerequisite for a wide family of downstream analyses. Real-world datasets are subject to noisy observations and sampling, so that distilling information about the underlying manifold is a major challenge. We propose a method for manifold learning that utilises a symmetric version of optimal transport with a quadratic regularisation that constructs a sparse and adaptive affinity matrix, that can be interpreted as a generalisation of the bistochastic kernel normalisation. We prove that the resulting kernel is consistent with a Laplace-type operator in the continuous limit, establish robustness
    
[^14]: R-Learning与异质性治疗效果估计中的逆变数加权的连接

    The Connection Between R-Learning and Inverse-Variance Weighting for Estimation of Heterogeneous Treatment Effects. (arXiv:2307.09700v1 [stat.ME])

    [http://arxiv.org/abs/2307.09700](http://arxiv.org/abs/2307.09700)

    R-Learning在估计条件平均治疗效果时采用了逆变数加权的形式来稳定回归，并简化了偏差项。

    

    我们的动机是为了探讨广泛流行的“R-Learner”的性能。像其他估计条件平均治疗效果（CATEs）的方法一样，R-Learning可以表示为加权伪结果回归（POR）。先前对POR技术的比较已经仔细注意了伪结果转换的选择。然而，我们认为性能的主要驱动因素实际上是权重的选择。具体地说，我们认为R-Learning隐式地执行了加权形式的POR，其中权重稳定了回归，并允许对偏差项进行方便的简化。

    Our motivation is to shed light the performance of the widely popular "R-Learner." Like many other methods for estimating conditional average treatment effects (CATEs), R-Learning can be expressed as a weighted pseudo-outcome regression (POR). Previous comparisons of POR techniques have paid careful attention to the choice of pseudo-outcome transformation. However, we argue that the dominant driver of performance is actually the choice of weights. Specifically, we argue that R-Learning implicitly performs an inverse-variance weighted form of POR. These weights stabilize the regression and allow for convenient simplifications of bias terms.
    
[^15]: 多视角自监督学习用于多变量通道时间序列

    Multi-view self-supervised learning for multivariate variable-channel time series. (arXiv:2307.09614v1 [stat.ML])

    [http://arxiv.org/abs/2307.09614](http://arxiv.org/abs/2307.09614)

    本论文提出了一种多视角自监督学习方法，用于处理多变量通道时间序列数据，在不同数据集之间进行迁移学习。通过预训练和微调，结合传递神经网络和TS2Vec损失，该方法在大多数设置下表现优于其他方法。

    

    对多变量生物医学时间序列数据进行标注是一项繁重和昂贵的任务。自监督对比学习通过对未标记数据进行预训练来减少对大型标记数据集的需求。然而，对于多变量时间序列数据，输入通道的集合在不同应用之间通常会有所变化，而大多数现有工作并不允许在具有不同输入通道集合的数据集之间进行迁移学习。我们提出了一种学习一种编码器来分别处理所有输入通道的方法。然后，我们使用传递神经网络在通道之间提取单一表示。我们通过在一个具有六个脑电图通道的数据集上进行预训练，并在一个具有两个不同脑电图通道的数据集上进行微调来展示这种方法的潜力。我们比较了具有传递神经网络和不具有传递神经网络的网络在不同对比损失函数下的性能。我们发现我们的方法结合了TS2Vec损失在大多数设置中的表现优于其他所有方法。

    Labeling of multivariate biomedical time series data is a laborious and expensive process. Self-supervised contrastive learning alleviates the need for large, labeled datasets through pretraining on unlabeled data. However, for multivariate time series data the set of input channels often varies between applications, and most existing work does not allow for transfer between datasets with different sets of input channels. We propose learning one encoder to operate on all input channels individually. We then use a message passing neural network to extract a single representation across channels. We demonstrate the potential of this method by pretraining our network on a dataset with six EEG channels and finetuning on a dataset with two different EEG channels. We compare networks with and without the message passing neural network across different contrastive loss functions. We show that our method combined with the TS2Vec loss outperforms all other methods in most settings.
    
[^16]: 顺序蒙特卡洛学习用于时间序列结构发现

    Sequential Monte Carlo Learning for Time Series Structure Discovery. (arXiv:2307.09607v1 [cs.LG])

    [http://arxiv.org/abs/2307.09607](http://arxiv.org/abs/2307.09607)

    本文提出了一种顺序蒙特卡洛学习的方法，用于自动发现复杂时间序列数据的准确模型。在实验中显示，该方法相对于之前的方法，具有较快的运行速度并能够发现合理的模型结构。

    

    本文提出了一种自动发现复杂时间序列数据准确模型的新方法。在高斯过程时间序列模型的符号空间上工作的贝叶斯非参数先验中，我们提出了一种集成顺序蒙特卡洛（SMC）和旋换MCMC的新型结构学习算法，以实现高效的后验推断。我们的方法可以在“在线”设置中使用，其中新数据顺序地合并在时间中，并且可以在“离线”设置中使用，通过使用历史数据的嵌套子集对后验进行退火。对真实世界的时间序列进行的实验测量结果显示，我们的方法相比之前针对相同模型族的MCMC和贪心搜索结构学习算法可以提供10倍至100倍的运行时间加速。我们使用我们的方法对1,428个计量经济数据集的知名基准进行了首次大规模的高斯过程时间序列结构学习评估。结果表明我们的方法可以发现合理的模型结构。

    This paper presents a new approach to automatically discovering accurate models of complex time series data. Working within a Bayesian nonparametric prior over a symbolic space of Gaussian process time series models, we present a novel structure learning algorithm that integrates sequential Monte Carlo (SMC) and involutive MCMC for highly effective posterior inference. Our method can be used both in "online" settings, where new data is incorporated sequentially in time, and in "offline" settings, by using nested subsets of historical data to anneal the posterior. Empirical measurements on real-world time series show that our method can deliver 10x--100x runtime speedups over previous MCMC and greedy-search structure learning algorithms targeting the same model family. We use our method to perform the first large-scale evaluation of Gaussian process time series structure learning on a prominent benchmark of 1,428 econometric datasets. The results show that our method discovers sensible 
    
[^17]: 自我兼容性：在没有基准数据的情况下评估因果发现的方法。

    Self-Compatibility: Evaluating Causal Discovery without Ground Truth. (arXiv:2307.09552v1 [cs.LG])

    [http://arxiv.org/abs/2307.09552](http://arxiv.org/abs/2307.09552)

    本论文提出了一种在没有基准数据的情况下评估因果发现方法的新方法，通过在不同变量子集上学习的因果图之间的兼容性检测，来伪证因果关系的推断正确性。

    

    鉴于因果基本事实非常罕见，因果发现算法通常只在模拟数据上进行评估。这令人担忧，因为模拟反映了关于噪声分布、模型类别等生成过程的常见假设。在这项工作中，我们提出了一种新的方法，用于在没有基准数据的情况下对因果发现算法的输出进行伪证。我们的关键见解是，尽管统计学习寻求数据点子集之间的稳定性，但因果学习应该寻求变量子集之间的稳定性。基于这个见解，我们的方法依赖于在不同变量子集上学习的因果图之间的兼容性概念。我们证明了检测不兼容性可以伪证因果关系被错误推断的原因，这是因为假设违反或有限样本效应带来的错误。虽然通过这种兼容性测试只是对良好性能的必要条件，但我们认为它提供了强有力的证据。

    As causal ground truth is incredibly rare, causal discovery algorithms are commonly only evaluated on simulated data. This is concerning, given that simulations reflect common preconceptions about generating processes regarding noise distributions, model classes, and more. In this work, we propose a novel method for falsifying the output of a causal discovery algorithm in the absence of ground truth. Our key insight is that while statistical learning seeks stability across subsets of data points, causal learning should seek stability across subsets of variables. Motivated by this insight, our method relies on a notion of compatibility between causal graphs learned on different subsets of variables. We prove that detecting incompatibilities can falsify wrongly inferred causal relations due to violation of assumptions or errors from finite sample effects. Although passing such compatibility tests is only a necessary criterion for good performance, we argue that it provides strong evidenc
    
[^18]: BOF-UCB: 一种用于非平稳环境下的上下界信心算法的贝叶斯优化频率算法

    BOF-UCB: A Bayesian-Optimistic Frequentist Algorithm for Non-Stationary Contextual Bandits. (arXiv:2307.03587v1 [cs.LG])

    [http://arxiv.org/abs/2307.03587](http://arxiv.org/abs/2307.03587)

    BOF-UCB是一种用于非平稳环境下的背景线性赌博机的贝叶斯优化频率算法，其结合了贝叶斯和频率学派原则，提高了在动态环境中的性能。它利用贝叶斯更新推断后验分布，并使用频率学派方法计算上界信心界以平衡探索和开发。实验证明，BOF-UCB优于现有方法，是非平稳环境中顺序决策的有前途的解决方案。

    

    我们提出了一种新颖的贝叶斯优化频率上下界信心算法（BOF-UCB），用于非平稳环境下的随机背景线性赌博机。贝叶斯和频率学派原则的独特结合增强了算法在动态环境中的适应性和性能。BOF-UCB算法利用顺序贝叶斯更新推断未知回归参数的后验分布，并随后采用频率学派方法通过最大化后验分布上的期望收益来计算上界信心界（UCB）。我们提供了BOF-UCB性能的理论保证，并在合成数据集和强化学习环境中的经典控制任务中展示了其有效性。我们的结果表明，BOF-UCB优于现有的方法，在非平稳环境中进行顺序决策是一个有前途的解决方案。

    We propose a novel Bayesian-Optimistic Frequentist Upper Confidence Bound (BOF-UCB) algorithm for stochastic contextual linear bandits in non-stationary environments. This unique combination of Bayesian and frequentist principles enhances adaptability and performance in dynamic settings. The BOF-UCB algorithm utilizes sequential Bayesian updates to infer the posterior distribution of the unknown regression parameter, and subsequently employs a frequentist approach to compute the Upper Confidence Bound (UCB) by maximizing the expected reward over the posterior distribution. We provide theoretical guarantees of BOF-UCB's performance and demonstrate its effectiveness in balancing exploration and exploitation on synthetic datasets and classical control tasks in a reinforcement learning setting. Our results show that BOF-UCB outperforms existing methods, making it a promising solution for sequential decision-making in non-stationary environments.
    
[^19]: 基于近似有效的$p$-阻抗的多类图聚类

    Multi-class Graph Clustering via Approximated Effective $p$-Resistance. (arXiv:2306.08617v1 [cs.LG])

    [http://arxiv.org/abs/2306.08617](http://arxiv.org/abs/2306.08617)

    本文提出了一种近似计算有效$p$-阻抗并将其应用于多类图聚类，该方法可以通过参数$p$偏向于具有高内部连通性或者更小的簇内顶点之间的最短路径距离的聚类。

    

    本文提出了一种近似计算有效$p$-阻抗并将其应用于多类聚类。基于图拉普拉斯和其$p$-拉普拉斯推广的谱方法一直是非欧几里得聚类技术的支柱。$p$-拉普拉斯的优点在于参数$p$对聚类结构具有可控偏倚。$p$-拉普拉斯特征向量法的缺点在于难以计算第三和更高阶特征向量。因此，我们动机在于使用由$p$-拉普拉斯引导的$p$-阻抗进行聚类。对于$p$-阻抗而言，小$p$会偏向于具有高内部连通性的聚类，而大$p$则会偏向于大小“范围”的聚类，即更小的簇内顶点之间的最短路径距离。然而，计算$p$-阻抗成本很高。我们通过开发$p$-阻抗的近似方法来克服这一问题。我们证明了上下界。

    This paper develops an approximation to the (effective) $p$-resistance and applies it to multi-class clustering. Spectral methods based on the graph Laplacian and its generalization to the graph $p$-Laplacian have been a backbone of non-euclidean clustering techniques. The advantage of the $p$-Laplacian is that the parameter $p$ induces a controllable bias on cluster structure. The drawback of $p$-Laplacian eigenvector based methods is that the third and higher eigenvectors are difficult to compute. Thus, instead, we are motivated to use the $p$-resistance induced by the $p$-Laplacian for clustering. For $p$-resistance, small $p$ biases towards clusters with high internal connectivity while large $p$ biases towards clusters of small ``extent,'' that is a preference for smaller shortest-path distances between vertices in the cluster. However, the $p$-resistance is expensive to compute. We overcome this by developing an approximation to the $p$-resistance. We prove upper and lower bounds
    
[^20]: 概率距离法异常检测

    Probabilistic Distance-Based Outlier Detection. (arXiv:2305.09446v1 [cs.LG])

    [http://arxiv.org/abs/2305.09446](http://arxiv.org/abs/2305.09446)

    本文提出了一种将距离法异常检测分数转化为可解释的概率估计的通用方法，该方法使用与其他数据点的距离建模距离概率分布，将距离法异常检测分数转换为异常概率，提高了正常点和异常点之间的对比度，而不会影响检测性能。

    

    距离法异常检测方法的分数难以解释，因此在没有额外的上下文信息的情况下，很难确定正常点和异常点之间的截断阈值。我们描述了将距离法异常检测分数转化为可解释的概率估计的通用方法。该转换是排名稳定的，并增加了正常点和异常点之间的对比度。确定数据点之间的距离关系是识别数据中最近邻关系所必需的，然而大多数计算出的距离通常被丢弃。我们展示了可以使用与其他数据点的距离来建模距离概率分布，并随后使用这些分布将距离法异常检测分数转换为异常概率。我们的实验表明，概率转换不会影响众多表格和图像基准数据集上的检测性能，但会产生可解释性。

    The scores of distance-based outlier detection methods are difficult to interpret, making it challenging to determine a cut-off threshold between normal and outlier data points without additional context. We describe a generic transformation of distance-based outlier scores into interpretable, probabilistic estimates. The transformation is ranking-stable and increases the contrast between normal and outlier data points. Determining distance relationships between data points is necessary to identify the nearest-neighbor relationships in the data, yet, most of the computed distances are typically discarded. We show that the distances to other data points can be used to model distance probability distributions and, subsequently, use the distributions to turn distance-based outlier scores into outlier probabilities. Our experiments show that the probabilistic transformation does not impact detection performance over numerous tabular and image benchmark datasets but results in interpretable
    
[^21]: 布雷-瓦瑟斯坦距离训练下的生成式深度线性网络的关键点和收敛性分析

    Critical Points and Convergence Analysis of Generative Deep Linear Networks Trained with Bures-Wasserstein Loss. (arXiv:2303.03027v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2303.03027](http://arxiv.org/abs/2303.03027)

    本文使用布雷-瓦瑟斯坦距离训练协方差矩阵的深度矩阵分解模型，并在有限秩矩阵空间内表征关键点和最小化问题，最终确定了梯度下降算法的收敛性。

    

    本文探讨了一种使用布雷-瓦瑟斯坦距离训练协方差矩阵的深度矩阵分解模型。相较于以往的研究，我们所提出的模型在损失函数和生成式设置上有所不同。我们在有限秩矩阵空间内表征了该方法的关键点和最小化问题。针对低秩矩阵而言，该方法的海森矩阵理论上可能会爆炸，这为优化方法的收敛性分析带来了挑战。我们确定了梯度下降算法中使用损失的平滑微扰版本时的收敛性，并在初始权重的一定假设条件下证明了有限步长梯度下降的收敛性。

    We consider a deep matrix factorization model of covariance matrices trained with the Bures-Wasserstein distance. While recent works have made important advances in the study of the optimization problem for overparametrized low-rank matrix approximation, much emphasis has been placed on discriminative settings and the square loss. In contrast, our model considers another interesting type of loss and connects with the generative setting. We characterize the critical points and minimizers of the Bures-Wasserstein distance over the space of rank-bounded matrices. For low-rank matrices the Hessian of this loss can theoretically blow up, which creates challenges to analyze convergence of optimizaton methods. We establish convergence results for gradient flow using a smooth perturbative version of the loss and convergence results for finite step size gradient descent under certain assumptions on the initial weights.
    
[^22]: 通过最大泄露分析噪声迭代算法的泛化误差界限

    Generalization Error Bounds for Noisy, Iterative Algorithms via Maximal Leakage. (arXiv:2302.14518v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.14518](http://arxiv.org/abs/2302.14518)

    通过最大泄露分析噪声迭代算法的泛化误差界限，证明了如果更新函数在L2-范数下有界且加性噪声为各向同性高斯噪声，则可以得到一个半封闭形式下的最大泄露上界，同时展示了更新函数的假设如何影响噪声的最优选择。

    

    我们采用信息论框架来分析一类迭代式、带有噪声的学习算法的泛化行为。由于这类算法具有随机性，并且包含常用的算法（如随机梯度 Langevin 动力学），所以在信息论度量下研究它们尤为合适。在本文中，我们使用最大泄露（等价于无穷阶 Sibson 互信息）度量，因其易于分析且可以同时获得泛化误差大概率上界和期望值上界。我们证明了如果更新函数（如梯度）在L2-范数下有界，且加性噪声为各向同性高斯噪声，则可以得到一个半封闭形式下的最大泄露上界。另外，我们还展示了更新函数的假设如何影响噪声的最优选择（即最小化产生的最大泄露）。最后，我们计算了...

    We adopt an information-theoretic framework to analyze the generalization behavior of the class of iterative, noisy learning algorithms. This class is particularly suitable for study under information-theoretic metrics as the algorithms are inherently randomized, and it includes commonly used algorithms such as Stochastic Gradient Langevin Dynamics (SGLD). Herein, we use the maximal leakage (equivalently, the Sibson mutual information of order infinity) metric, as it is simple to analyze, and it implies both bounds on the probability of having a large generalization error and on its expected value. We show that, if the update function (e.g., gradient) is bounded in $L_2$-norm and the additive noise is isotropic Gaussian noise, then one can obtain an upper-bound on maximal leakage in semi-closed form. Furthermore, we demonstrate how the assumptions on the update function affect the optimal (in the sense of minimizing the induced maximal leakage) choice of the noise. Finally, we compute 
    
[^23]: CO-BED：通过贝叶斯实验设计的信息理论上下文优化

    CO-BED: Information-Theoretic Contextual Optimization via Bayesian Experimental Design. (arXiv:2302.14015v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.14015](http://arxiv.org/abs/2302.14015)

    CO-BED是一个通用的、与模型无关的框架，用于通过贝叶斯实验设计的信息理论来进行上下文优化。它采用黑箱变分方法同时估计和优化设计，可以适应离散动作，并在多个实验中展示出竞争性能。

    

    我们通过贝叶斯实验设计的视角对上下文优化问题进行了形式化，并提出了CO-BED - 一个通用的、与模型无关的框架，用于使用信息理论原则设计上下文实验。在制定合适的基于信息的目标后，我们采用黑箱变分方法在单一随机梯度方案中同时估计和优化设计。此外，为了适应我们框架中的离散动作，我们提议利用连续松弛方案，这可以自然地集成到我们变分目标中。因此，CO-BED为各种上下文优化问题提供了通用的自动化解决方案。我们在许多实验中演示了其有效性，即使与定制的、特定于模型的替代方法相比，CO-BED也表现出了竞争性能。

    We formalize the problem of contextual optimization through the lens of Bayesian experimental design and propose CO-BED -- a general, model-agnostic framework for designing contextual experiments using information-theoretic principles. After formulating a suitable information-based objective, we employ black-box variational methods to simultaneously estimate it and optimize the designs in a single stochastic gradient scheme. In addition, to accommodate discrete actions within our framework, we propose leveraging continuous relaxation schemes, which can naturally be integrated into our variational objective. As a result, CO-BED provides a general and automated solution to a wide range of contextual optimization problems. We illustrate its effectiveness in a number of experiments, where CO-BED demonstrates competitive performance even when compared to bespoke, model-specific alternatives.
    
[^24]: 顺序核独立性测试

    Sequential Kernelized Independence Testing. (arXiv:2212.07383v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2212.07383](http://arxiv.org/abs/2212.07383)

    该论文介绍了顺序核独立性测试的方法，以解决传统批量测试在流数据上的问题，实现了根据任务复杂性自适应调整样本大小，并在收集新数据后持续监测和控制误报率。

    

    独立性测试是一个经典的统计问题，在固定采集数据之前的批量设置中得到了广泛研究。然而，实践者们往往更喜欢能够根据问题的复杂性进行自适应的程序，而不是事先设定样本大小。理想情况下，这样的程序应该（a）在简单任务上尽早停止（在困难任务上稍后停止），因此更好地利用可用资源，以及（b）在收集新数据之后，持续监测数据并高效地整合统计证据，同时控制误报率。经典的批量测试不适用于流数据：在数据观察后进行有效推断需要对多重测试进行校正，这导致了低功率。遵循通过投注进行测试的原则，我们设计了顺序核独立性测试，克服了这些缺点。我们通过采用由核相关性测度（如Hilbert-）启发的投注来说明我们的广泛框架。

    Independence testing is a classical statistical problem that has been extensively studied in the batch setting when one fixes the sample size before collecting data. However, practitioners often prefer procedures that adapt to the complexity of a problem at hand instead of setting sample size in advance. Ideally, such procedures should (a) stop earlier on easy tasks (and later on harder tasks), hence making better use of available resources, and (b) continuously monitor the data and efficiently incorporate statistical evidence after collecting new data, while controlling the false alarm rate. Classical batch tests are not tailored for streaming data: valid inference after data peeking requires correcting for multiple testing which results in low power. Following the principle of testing by betting, we design sequential kernelized independence tests that overcome such shortcomings. We exemplify our broad framework using bets inspired by kernelized dependence measures, e.g., the Hilbert-
    
[^25]: Alpha-divergence变分推断与重要性加权自编码器的结合：方法和渐近性

    Alpha-divergence Variational Inference Meets Importance Weighted Auto-Encoders: Methodology and Asymptotics. (arXiv:2210.06226v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.06226](http://arxiv.org/abs/2210.06226)

    本文提出了VR-IWAE下界，该下界是IWAE下界的推广，采用无偏梯度估计器能够实现与VR下界相同的随机梯度下降过程，对该下界进行了理论分析，揭示了其优势和不足，并通过示例验证了理论观点。

    

    针对目标后验分布和变分分布之间的alpha散度，已经提出了几个涉及变分Rényi (VR)下界的算法。尽管有令人满意的实证结果，但这些算法都采用了有偏的随机梯度下降过程，因此缺乏理论保证。本文对VR-IWAE下界进行了正式化和研究，该下界是重要性加权自编码器(IWAE)下界的推广。我们证明了VR-IWAE下界具有几个可取的特性，特别是在重新参数化的情况下与VR下界导致相同的随机梯度下降过程，但这次是依靠无偏梯度估计器。然后，我们提供了对VR-IWAE下界以及标准IWAE下界的两种互补的理论分析。这些分析揭示了这些下界的好处和缺点。最后，我们通过玩具和真实数据示例来说明我们的理论观点。

    Several algorithms involving the Variational R\'enyi (VR) bound have been proposed to minimize an alpha-divergence between a target posterior distribution and a variational distribution. Despite promising empirical results, those algorithms resort to biased stochastic gradient descent procedures and thus lack theoretical guarantees. In this paper, we formalize and study the VR-IWAE bound, a generalization of the Importance Weighted Auto-Encoder (IWAE) bound. We show that the VR-IWAE bound enjoys several desirable properties and notably leads to the same stochastic gradient descent procedure as the VR bound in the reparameterized case, but this time by relying on unbiased gradient estimators. We then provide two complementary theoretical analyses of the VR-IWAE bound and thus of the standard IWAE bound. Those analyses shed light on the benefits or lack thereof of these bounds. Lastly, we illustrate our theoretical claims over toy and real-data examples.
    
[^26]: 使用加权不对称损失函数的神经网络模型预测区间

    Prediction intervals for neural network models using weighted asymmetric loss functions. (arXiv:2210.04318v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.04318](http://arxiv.org/abs/2210.04318)

    本论文提出了一种使用加权不对称损失函数的方法，生成可靠的预测区间，适用于复杂的机器学习情境，可扩展为参数化函数的PI预测。

    

    我们提出了一种简单而有效的方法来生成近似和预测趋势的预测区间（PIs）。我们利用加权不对称损失函数来估计PI的下限和上限，权重由区间宽度确定。我们提供了该方法的简洁数学证明，展示了如何将其扩展到为参数化函数推导PI，并论证了该方法为预测相关变量的PI而有效的原因。我们在基于神经网络的模型的真实世界预测任务上对该方法进行了测试，结果表明它在复杂的机器学习情境下可以产生可靠的PI。

    We propose a simple and efficient approach to generate prediction intervals (PIs) for approximated and forecasted trends. Our method leverages a weighted asymmetric loss function to estimate the lower and upper bounds of the PIs, with the weights determined by the interval width. We provide a concise mathematical proof of the method, show how it can be extended to derive PIs for parametrised functions and argue why the method works for predicting PIs of dependent variables. The presented tests of the method on a real-world forecasting task using a neural network-based model show that it can produce reliable PIs in complex machine learning scenarios.
    
[^27]: 在潜在空间中的合作：在变分自编码器中添加混合成分的好处

    Cooperation in the Latent Space: The Benefits of Adding Mixture Components in Variational Autoencoders. (arXiv:2209.15514v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.15514](http://arxiv.org/abs/2209.15514)

    本研究展示了在变分自编码器中添加混合成分的好处，并证明了混合成分的增加能够提高其在图像和单细胞数据集上的潜在表示能力。这表明使用混合VAE是获取更灵活变分逼近的标准方法。

    

    本文展示了混合成分在共同适应最大化ELBO时的合作方式。我们借鉴了最近在多个和自适应重要性采样文献中的进展。我们使用单独的编码器网络对混合成分进行建模，并在实证上证明ELBO随混合成分数量的增加是单调非减的。这些结果适用于MNIST、FashionMNIST和CIFAR-10数据集上的不同VAE架构。本工作还表明增加混合成分的数量能够改善VAE在图像和单细胞数据集上的潜在表示能力。这种合作行为表明，使用混合VAE应被视为获取更灵活的变分近似的标准方法。最后，我们首次在大范围的消融实验中将混合VAE与归一化流、层次模型和/或VampPrior进行了比较和结合。

    In this paper, we show how the mixture components cooperate when they jointly adapt to maximize the ELBO. We build upon recent advances in the multiple and adaptive importance sampling literature. We then model the mixture components using separate encoder networks and show empirically that the ELBO is monotonically non-decreasing as a function of the number of mixture components. These results hold for a range of different VAE architectures on the MNIST, FashionMNIST, and CIFAR-10 datasets. In this work, we also demonstrate that increasing the number of mixture components improves the latent-representation capabilities of the VAE on both image and single-cell datasets. This cooperative behavior motivates that using Mixture VAEs should be considered a standard approach for obtaining more flexible variational approximations. Finally, Mixture VAEs are here, for the first time, compared and combined with normalizing flows, hierarchical models and/or the VampPrior in an extensive ablation 
    
[^28]: 不同分布的数据价值

    The Value of Out-of-Distribution Data. (arXiv:2208.10967v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.10967](http://arxiv.org/abs/2208.10967)

    不同分布的数据可以对任务的泛化误差产生非单调的影响，使用少量不同分布的数据进行训练是有价值的。

    

    我们期望随着类似任务样本的增加，泛化误差会减小；而随着来自不同分布（OOD）任务样本的增加，泛化误差会增大。在这项工作中，我们展示了一个反直觉的现象：任务的泛化误差可以是样本从OOD任务中的数量的非单调函数。随着OOD样本数量的增加，目标任务的泛化误差在超过一个阈值之前会先减小后增大。换句话说，使用少量OOD数据进行训练是有价值的。我们在合成数据集上使用Fisher线性判别和计算机视觉基准数据集（如MNIST、CIFAR-10、CINIC-10、PACS和DomainNet）上的深度网络来展示和分析这一现象。在我们知道哪些样本属于OOD的理想情况下，我们展示了可以利用目标和OOD经验风险的适当加权目标来利用这些非单调趋势。尽管实际应用有限，但这表明如果我们能够检测到OOD样本，这种方法可能是有价值的。

    We expect the generalization error to improve with more samples from a similar task, and to deteriorate with more samples from an out-of-distribution (OOD) task. In this work, we show a counter-intuitive phenomenon: the generalization error of a task can be a non-monotonic function of the number of OOD samples. As the number of OOD samples increases, the generalization error on the target task improves before deteriorating beyond a threshold. In other words, there is value in training on small amounts of OOD data. We use Fisher's Linear Discriminant on synthetic datasets and deep networks on computer vision benchmarks such as MNIST, CIFAR-10, CINIC-10, PACS and DomainNet to demonstrate and analyze this phenomenon. In the idealistic setting where we know which samples are OOD, we show that these non-monotonic trends can be exploited using an appropriately weighted objective of the target and OOD empirical risk. While its practical utility is limited, this does suggest that if we can det
    
[^29]: 学习处理时间相关的流式数据的在线随机算法

    Learning from time-dependent streaming data with online stochastic algorithms. (arXiv:2205.12549v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2205.12549](http://arxiv.org/abs/2205.12549)

    本文研究了处理时间相关的流式数据的在线随机算法，并通过非渐进分析建立了新颖的启发式算法，加速收敛。实验证明时间变化的小批量SGD方法可以打破依赖结构，有偏倚的SGD方法具有与无偏倚方法相当的性能，并且使用Polyak-Ruppert平均化方法能够加快随机优化算法的收敛。

    

    本文探讨了在时间相关且有偏倚梯度估计下的流式优化问题。我们分析了一些一阶方法，包括随机梯度下降（SGD）、小批量SGD和时间变化的小批量SGD，以及它们的Polyak-Ruppert平均值。我们的非渐进分析建立了新颖的启发式算法，将依赖性、偏倚和凸性水平联系起来，实现了加速收敛。具体来说，我们的研究结果表明：（i）时间变化的小批量SGD方法能够打破长期和短期的依赖结构；（ii）有偏倚的SGD方法可以达到与无偏倚方法相当的性能；（iii）使用Polyak-Ruppert平均化方法可以加速随机优化算法的收敛。为了验证我们的理论发现，我们在模拟和现实的时间相关数据上进行了一系列实验。

    This paper addresses stochastic optimization in a streaming setting with time-dependent and biased gradient estimates. We analyze several first-order methods, including Stochastic Gradient Descent (SGD), mini-batch SGD, and time-varying mini-batch SGD, along with their Polyak-Ruppert averages. Our non-asymptotic analysis establishes novel heuristics that link dependence, biases, and convexity levels, enabling accelerated convergence. Specifically, our findings demonstrate that (i) time-varying mini-batch SGD methods have the capability to break long- and short-range dependence structures, (ii) biased SGD methods can achieve comparable performance to their unbiased counterparts, and (iii) incorporating Polyak-Ruppert averaging can accelerate the convergence of the stochastic optimization algorithms. To validate our theoretical findings, we conduct a series of experiments using both simulated and real-life time-dependent data.
    
[^30]: 部分观察的马尔科夫决策过程（POMDPs）的自然演员-评论家方法的有限时间分析

    Finite-Time Analysis of Natural Actor-Critic for POMDPs. (arXiv:2202.09753v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.09753](http://arxiv.org/abs/2202.09753)

    本文分析了部分观察的马尔科夫决策过程（POMDPs）下自然演员-评论家方法的有限时间特性，并对使用有限状态控制器产生的错误进行了明确的表征。

    

    我们考虑了有限或可数无限状态空间的部分观察的马尔科夫决策过程（POMDPs）的强化学习问题，其中控制器只能访问基础控制马尔科夫链的噪声观测。我们考虑了一种自然的演员-评论家方法，该方法采用有限的内部存储器进行策略参数化，并使用多步时序差异学习算法进行策略评估。凭借我们的知识，我们首次确立了部分观察系统下基于函数逼近的演员-评论家方法的非渐近全局收敛性。特别地，除了在MDPs中出现的函数逼近和统计误差之外，我们还明确地表征了由于使用有限状态控制器而产生的错误。这种额外的错误是以传统的POMDPs中的信心状态和使用有限状态时的隐藏状态的后验分布之间的总变差距离来表示的。

    We consider the reinforcement learning problem for partially observed Markov decision processes (POMDPs) with large or even countably infinite state spaces, where the controller has access to only noisy observations of the underlying controlled Markov chain. We consider a natural actor-critic method that employs a finite internal memory for policy parameterization, and a multi-step temporal difference learning algorithm for policy evaluation. We establish, to the best of our knowledge, the first non-asymptotic global convergence of actor-critic methods for partially observed systems under function approximation. In particular, in addition to the function approximation and statistical errors that also arise in MDPs, we explicitly characterize the error due to the use of finite-state controllers. This additional error is stated in terms of the total variation distance between the traditional belief state in POMDPs and the posterior distribution of the hidden state when using a finite-sta
    
[^31]: Weisfeiler和Leman来做机器学习了：目前的研究进展。

    Weisfeiler and Leman go Machine Learning: The Story so far. (arXiv:2112.09992v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2112.09992](http://arxiv.org/abs/2112.09992)

    Weisfeiler-Leman算法被广泛应用于处理图和关系数据。本文全面介绍了该算法在监督学习中的应用，包括理论背景、扩展、与等变神经网格的联系、并列出了当前应用和未来研究方向。

    

    近年来，基于Weisfeiler-Leman算法的算法和神经架构已成为处理图和关系数据的机器学习的强大工具。本文全面介绍算法在机器学习环境中的使用情况，重点关注监督学习。我们讨论了理论背景，展示了如何将其用于监督图形和节点表示学习，讨论了最近的扩展，并概述了算法与（置换）等变神经网格的联系。此外，我们还概述了当前的应用和未来的研究方向以刺激进一步的研究。

    In recent years, algorithms and neural architectures based on the Weisfeiler-Leman algorithm, a well-known heuristic for the graph isomorphism problem, have emerged as a powerful tool for machine learning with graphs and relational data. Here, we give a comprehensive overview of the algorithm's use in a machine-learning setting, focusing on the supervised regime. We discuss the theoretical background, show how to use it for supervised graph and node representation learning, discuss recent extensions, and outline the algorithm's connection to (permutation-)equivariant neural architectures. Moreover, we give an overview of current applications and future directions to stimulate further research.
    
[^32]: MixPath: 一种统一的一次性神经架构搜索方法

    MixPath: A Unified Approach for One-shot Neural Architecture Search. (arXiv:2001.05887v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2001.05887](http://arxiv.org/abs/2001.05887)

    本论文提出了一种名为MixPath的统一的一次性神经架构搜索方法，通过训练一次性的多路径超网络来准确评估候选架构。采用一种新颖的机制称为Shadow Batch Normalization（SBN）来解决多路径结构的特征差异问题，稳定优化并提高排名性能。

    

    在神经架构设计中，混合多个卷积核被证明是有优势的。然而，当前的两阶段神经架构搜索方法主要局限于单路径搜索空间。如何高效地搜索多路径结构的模型仍然是一个难题。在本文中，我们的动机是训练一个一次性的多路径超网络来准确评估候选架构。具体来说，我们发现在所研究的搜索空间中，从多个路径中求和的特征向量几乎是单个路径的倍数。这种差异扰乱了超网络的训练和排名能力。因此，我们提出了一种新颖的机制，称为Shadow Batch Normalization（SBN），来规范差异的特征统计。大量实验证明，SBN能够稳定优化和提高排名性能。我们将我们的统一多路径一次性方法称为MixPath，可以生成一系列能达到最新技术水平的模型。

    Blending multiple convolutional kernels is proved advantageous in neural architecture design. However, current two-stage neural architecture search methods are mainly limited to single-path search spaces. How to efficiently search models of multi-path structures remains a difficult problem. In this paper, we are motivated to train a one-shot multi-path supernet to accurately evaluate the candidate architectures. Specifically, we discover that in the studied search spaces, feature vectors summed from multiple paths are nearly multiples of those from a single path. Such disparity perturbs the supernet training and its ranking ability. Therefore, we propose a novel mechanism called Shadow Batch Normalization (SBN) to regularize the disparate feature statistics. Extensive experiments prove that SBNs are capable of stabilizing the optimization and improving ranking performance. We call our unified multi-path one-shot approach as MixPath, which generates a series of models that achieve state
    

