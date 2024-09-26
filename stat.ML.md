# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT.](http://arxiv.org/abs/2401.03302) | 本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。 |
| [^2] | [Optimal vintage factor analysis with deflation varimax.](http://arxiv.org/abs/2310.10545) | 本文提出了一种采用通货紧缩变量旋转的拟合因子分析方法，在每一行上逐步求解正交矩阵，相比于传统方法具有更好的计算性能和灵活性，并且在更广泛的背景下提供了理论保证。 |
| [^3] | [Non-backtracking Graph Neural Networks.](http://arxiv.org/abs/2310.07430) | 非回溯图神经网络(NBA-GNN)通过不考虑先前访问节点的消息来解决图神经网络本地更新中的冗余问题，并且在随机块模型恢复方面表现出良好的性能。 |
| [^4] | [Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization.](http://arxiv.org/abs/2310.03234) | 本文研究了一种新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)，通过扩展已有的研究，我们研究了非光滑弱凸FCCO的问题，并提出了一种单循环算法来找到Moreau环的ε-稳定点。 |
| [^5] | [WASA: WAtermark-based Source Attribution for Large Language Model-Generated Data.](http://arxiv.org/abs/2310.00646) | 本文提出了一种基于水印的框架WASA，通过允许大型语言模型生成带有嵌入源信息的合成文本水印来解决源归属和数据来源的问题。 |
| [^6] | [On Collaboration in Distributed Parameter Estimation with Resource Constraints.](http://arxiv.org/abs/2307.06442) | 在资源约束下的分布参数估计中，我们研究了传感器/代理数据收集和协作策略，通过最大化费舍尔信息或最小化Cramer-Rao界来解决传感器/代理的数据收集和协作策略设计问题。 |
| [^7] | [Projective Proximal Gradient Descent for A Class of Nonconvex Nonsmooth Optimization Problems: Fast Convergence Without Kurdyka-Lojasiewicz (KL) Property.](http://arxiv.org/abs/2304.10499) | 本文提出了一个投影近端梯度下降算法(PPGD)，成功地解决了一类非凸非光滑优化问题。该算法可以实现局部快速收敛，当迭代次数 $k \geq k_0$ 时，PPGD 可以以 $\cO(1/k^2)$ 的快速收敛率收敛。 |

# 详细

[^1]: 行动中的现实主义：使用YOLOv8和DeiT从医学图像中诊断脑肿瘤的异常感知

    Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT. (arXiv:2401.03302v1 [eess.IV])

    [http://arxiv.org/abs/2401.03302](http://arxiv.org/abs/2401.03302)

    本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。

    

    在医学科学领域，由于脑肿瘤在患者中的罕见程度，可靠地检测和分类脑肿瘤仍然是一个艰巨的挑战。因此，在异常情况下检测肿瘤的能力对于确保及时干预和改善患者结果至关重要。本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤。来自国家脑映射实验室（NBML）的精选数据集包括81名患者，其中包括30例肿瘤病例和51例正常病例。检测和分类流程被分为两个连续的任务。检测阶段包括全面的数据分析和预处理，以修改图像样本和每个类别的患者数量，以符合真实世界场景中的异常分布（9个正常样本对应1个肿瘤样本）。此外，在测试中除了常见的评估指标外，我们还采用了... [摘要长度已达到上限]

    In the field of medical sciences, reliable detection and classification of brain tumors from images remains a formidable challenge due to the rarity of tumors within the population of patients. Therefore, the ability to detect tumors in anomaly scenarios is paramount for ensuring timely interventions and improved patient outcomes. This study addresses the issue by leveraging deep learning (DL) techniques to detect and classify brain tumors in challenging situations. The curated data set from the National Brain Mapping Lab (NBML) comprises 81 patients, including 30 Tumor cases and 51 Normal cases. The detection and classification pipelines are separated into two consecutive tasks. The detection phase involved comprehensive data analysis and pre-processing to modify the number of image samples and the number of patients of each class to anomaly distribution (9 Normal per 1 Tumor) to comply with real world scenarios. Next, in addition to common evaluation metrics for the testing, we emplo
    
[^2]: 优化拟合因子分析与通货紧缩变量旋转

    Optimal vintage factor analysis with deflation varimax. (arXiv:2310.10545v1 [stat.ML])

    [http://arxiv.org/abs/2310.10545](http://arxiv.org/abs/2310.10545)

    本文提出了一种采用通货紧缩变量旋转的拟合因子分析方法，在每一行上逐步求解正交矩阵，相比于传统方法具有更好的计算性能和灵活性，并且在更广泛的背景下提供了理论保证。

    

    通货紧缩变量旋转是一种重要的因子分析方法，旨在首先找到原始数据的低维表示，然后寻求旋转，使旋转后的低维表示具有科学意义。尽管Principal Component Analysis (PCA) followed by the varimax rotation被广泛应用于拟合因子分析，但由于varimax rotation需要在正交矩阵集合上解非凸优化问题，因此很难提供理论保证。本文提出了一种逐行求解正交矩阵的通货紧缩变量旋转过程。除了在计算上的优势和灵活性之外，我们还能在广泛的背景下对所提出的过程进行完全的理论保证。在PCA之后采用这种新的varimax方法作为第二步，我们进一步分析了这个两步过程在一个更一般的因子模型的情况下。

    Vintage factor analysis is one important type of factor analysis that aims to first find a low-dimensional representation of the original data, and then to seek a rotation such that the rotated low-dimensional representation is scientifically meaningful. Perhaps the most widely used vintage factor analysis is the Principal Component Analysis (PCA) followed by the varimax rotation. Despite its popularity, little theoretical guarantee can be provided mainly because varimax rotation requires to solve a non-convex optimization over the set of orthogonal matrices.  In this paper, we propose a deflation varimax procedure that solves each row of an orthogonal matrix sequentially. In addition to its net computational gain and flexibility, we are able to fully establish theoretical guarantees for the proposed procedure in a broad context.  Adopting this new varimax approach as the second step after PCA, we further analyze this two step procedure under a general class of factor models. Our resul
    
[^3]: 非回溯图神经网络

    Non-backtracking Graph Neural Networks. (arXiv:2310.07430v1 [cs.LG])

    [http://arxiv.org/abs/2310.07430](http://arxiv.org/abs/2310.07430)

    非回溯图神经网络(NBA-GNN)通过不考虑先前访问节点的消息来解决图神经网络本地更新中的冗余问题，并且在随机块模型恢复方面表现出良好的性能。

    

    著名的图神经网络的消息传递更新允许使用本地和计算上可跟踪的更新来表示大规模图。然而，本地更新受到回溯的影响，即消息通过同一条边两次流动并重访先前访问的节点。由于消息流的数量随着更新的次数呈指数级增加，本地更新中的冗余阻碍了图神经网络准确识别下游任务的特定消息流。在这项工作中，我们通过非回溯的图神经网络（NBA-GNN）解决了这种冗余，该网络在更新消息时不考虑先前访问节点的消息。我们进一步研究了NBA-GNN如何缓解GNN的过度压缩，并建立了NBA-GNN和非回溯更新在随机块模型恢复方面出色性能之间的联系。我们通过实验证实了我们的NBA-

    The celebrated message-passing updates for graph neural networks allow the representation of large-scale graphs with local and computationally tractable updates. However, the local updates suffer from backtracking, i.e., a message flows through the same edge twice and revisits the previously visited node. Since the number of message flows increases exponentially with the number of updates, the redundancy in local updates prevents the graph neural network from accurately recognizing a particular message flow for downstream tasks. In this work, we propose to resolve such a redundancy via the non-backtracking graph neural network (NBA-GNN) that updates a message without incorporating the message from the previously visited node. We further investigate how NBA-GNN alleviates the over-squashing of GNNs, and establish a connection between NBA-GNN and the impressive performance of non-backtracking updates for stochastic block model recovery. We empirically verify the effectiveness of our NBA-
    
[^4]: 非光滑弱凸有限和耦合组合优化

    Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization. (arXiv:2310.03234v1 [math.OC])

    [http://arxiv.org/abs/2310.03234](http://arxiv.org/abs/2310.03234)

    本文研究了一种新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)，通过扩展已有的研究，我们研究了非光滑弱凸FCCO的问题，并提出了一种单循环算法来找到Moreau环的ε-稳定点。

    

    本文研究了一类新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)。由于其在机器学习和人工智能领域的广泛应用以及其解决基于经验风险最小化的随机算法的局限性，FCCO引起了越来越多的关注。然而，目前对于FCCO的研究假设内外函数都是光滑的，限制了其能够解决更多种类的问题的潜力。我们的研究从非光滑弱凸FCCO的角度进行了扩展，其中外函数是弱凸且非递减的，内函数是弱凸的。我们分析了一种单循环算法，并确定其在找到Moreau环的ε-稳定点的复杂度。

    This paper investigates new families of compositional optimization problems, called $\underline{\bf n}$on-$\underline{\bf s}$mooth $\underline{\bf w}$eakly-$\underline{\bf c}$onvex $\underline{\bf f}$inite-sum $\underline{\bf c}$oupled $\underline{\bf c}$ompositional $\underline{\bf o}$ptimization (NSWC FCCO). There has been a growing interest in FCCO due to its wide-ranging applications in machine learning and AI, as well as its ability to address the shortcomings of stochastic algorithms based on empirical risk minimization. However, current research on FCCO presumes that both the inner and outer functions are smooth, limiting their potential to tackle a more diverse set of problems. Our research expands on this area by examining non-smooth weakly-convex FCCO, where the outer function is weakly convex and non-decreasing, and the inner function is weakly-convex. We analyze a single-loop algorithm and establish its complexity for finding an $\epsilon$-stationary point of the Moreau env
    
[^5]: WASA：基于水印的大型语言模型生成数据的源归属

    WASA: WAtermark-based Source Attribution for Large Language Model-Generated Data. (arXiv:2310.00646v1 [cs.LG])

    [http://arxiv.org/abs/2310.00646](http://arxiv.org/abs/2310.00646)

    本文提出了一种基于水印的框架WASA，通过允许大型语言模型生成带有嵌入源信息的合成文本水印来解决源归属和数据来源的问题。

    

    大型语言模型（LLM）的出色性能和其商业化的巨大潜力引发了对其训练数据知识产权（IP）的严重关注。特别是，LLM生成的合成文本可能侵犯被用于训练LLM的数据的知识产权。为此，我们能够（a）通过水印识别出对LLM生成的合成文本做出贡献的数据提供者（源归属）；以及（b）验证文本数据是否来自于某个数据提供者对LLM进行了训练（数据来源）。在本文中，我们展示了通过水印技术可以解决这两个问题，即通过让LLM生成具有嵌入源信息的合成文本水印来实现。我们确定了这种水印技术框架的关键特性（例如源归属准确性、抵抗对手攻击的鲁棒性），并提出了一个满足这些要求的WAtermarking for Source Attribution（WASA）框架.

    The impressive performances of large language models (LLMs) and their immense potential for commercialization have given rise to serious concerns over the intellectual property (IP) of their training data. In particular, the synthetic texts generated by LLMs may infringe the IP of the data being used to train the LLMs. To this end, it is imperative to be able to (a) identify the data provider who contributed to the generation of a synthetic text by an LLM (source attribution) and (b) verify whether the text data from a data provider has been used to train an LLM (data provenance). In this paper, we show that both problems can be solved by watermarking, i.e., by enabling an LLM to generate synthetic texts with embedded watermarks that contain information about their source(s). We identify the key properties of such watermarking frameworks (e.g., source attribution accuracy, robustness against adversaries), and propose a WAtermarking for Source Attribution (WASA) framework that satisfies
    
[^6]: 在资源约束下的分布参数估计中的协作研究

    On Collaboration in Distributed Parameter Estimation with Resource Constraints. (arXiv:2307.06442v1 [cs.LG])

    [http://arxiv.org/abs/2307.06442](http://arxiv.org/abs/2307.06442)

    在资源约束下的分布参数估计中，我们研究了传感器/代理数据收集和协作策略，通过最大化费舍尔信息或最小化Cramer-Rao界来解决传感器/代理的数据收集和协作策略设计问题。

    

    我们研究了考虑资源约束和不同传感器/代理收集的观测之间的相关性的参数估计的传感器/代理数据收集和协作策略。具体地，我们考虑了一组传感器/代理，每个传感器/代理样本来自多元高斯分布的不同变量，并且具有不同的估计目标，我们将传感器/代理的数据收集和协作策略设计问题阐述为费舍尔信息最大化（或Cramer-Rao界最小化）问题。当变量之间的相关性知识可用时，我们可以分析地识别出两个特定情况：（1）不能利用样本之间的相关性知识进行协作估计的情况，（2）最优数据收集策略涉及投资有限资源以协作采样和转移已知统计信息的情况。

    We study sensor/agent data collection and collaboration policies for parameter estimation, accounting for resource constraints and correlation between observations collected by distinct sensors/agents. Specifically, we consider a group of sensors/agents each samples from different variables of a multivariate Gaussian distribution and has different estimation objectives, and we formulate a sensor/agent's data collection and collaboration policy design problem as a Fisher information maximization (or Cramer-Rao bound minimization) problem. When the knowledge of correlation between variables is available, we analytically identify two particular scenarios: (1) where the knowledge of the correlation between samples cannot be leveraged for collaborative estimation purposes and (2) where the optimal data collection policy involves investing scarce resources to collaboratively sample and transfer information that is not of immediate interest and whose statistics are already known, with the sol
    
[^7]: 一类非凸非光滑优化问题的投影近端梯度下降算法：不需要 Kurdyka-Lojasiewicz（KL）性质也能实现快速收敛

    Projective Proximal Gradient Descent for A Class of Nonconvex Nonsmooth Optimization Problems: Fast Convergence Without Kurdyka-Lojasiewicz (KL) Property. (arXiv:2304.10499v1 [math.OC])

    [http://arxiv.org/abs/2304.10499](http://arxiv.org/abs/2304.10499)

    本文提出了一个投影近端梯度下降算法(PPGD)，成功地解决了一类非凸非光滑优化问题。该算法可以实现局部快速收敛，当迭代次数 $k \geq k_0$ 时，PPGD 可以以 $\cO(1/k^2)$ 的快速收敛率收敛。

    

    非凸非光滑优化问题在统计学和机器学习中具有重要意义且具有挑战性。在本文中，我们提出了解决一类非凸非光滑优化问题的投影近端梯度下降算法(PPGD)，其中非凸性和非光滑性源自一个非凸但分段凸的非光滑正则化项。与现有基于 Kurdyka-\L{}ojasiewicz (K\L{}) 性质对非凸非光滑问题进行加速 PGD 方法的收敛分析不同，我们提供了一种新的理论分析，证明了 PPGD 在温和假设下在一类非凸非光滑问题中实现了快速局部收敛。证明了当迭代次数 $k \geq k_0$ 时，PPGD 可以以 $\cO(1/k^2)$ 的快速收敛率收敛，其中 $k_0$ 是一个有限的常数。该算法在光滑且凸目标函数的一阶方法具有利普希茨连续梯度的情况下实现了局部 Nesterov 的最优收敛速度。实验结果表明......（此处省略）。

    Nonconvex and nonsmooth optimization problems are important and challenging for statistics and machine learning. In this paper, we propose Projected Proximal Gradient Descent (PPGD) which solves a class of nonconvex and nonsmooth optimization problems, where the nonconvexity and nonsmoothness come from a nonsmooth regularization term which is nonconvex but piecewise convex. In contrast with existing convergence analysis of accelerated PGD methods for nonconvex and nonsmooth problems based on the Kurdyka-\L{}ojasiewicz (K\L{}) property, we provide a new theoretical analysis showing local fast convergence of PPGD. It is proved that PPGD achieves a fast convergence rate of $\cO(1/k^2)$ when the iteration number $k \ge k_0$ for a finite $k_0$ on a class of nonconvex and nonsmooth problems under mild assumptions, which is locally Nesterov's optimal convergence rate of first-order methods on smooth and convex objective function with Lipschitz continuous gradient. Experimental results demonst
    

