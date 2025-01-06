# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Kernel PCA for Out-of-Distribution Detection](https://arxiv.org/abs/2402.02949) | 本论文提出了使用核PCA进行外分布检测的方法，通过在主成分子空间中引入非线性映射，实现了对内分布和外分布数据的有效区分。 |
| [^2] | [Gradual Domain Adaptation: Theory and Algorithms.](http://arxiv.org/abs/2310.13852) | 本文研究了渐进域自适应中的渐进自训练算法，提出了一个改进的泛化界限，并指出了中间域在源域和目标域之间均匀放置的重要性。 |
| [^3] | [Ano-SuPs: Multi-size anomaly detection for manufactured products by identifying suspected patches.](http://arxiv.org/abs/2309.11120) | Ano-SuPs是一种通过识别可疑区块来进行制造产品的多尺度异常检测的两阶段策略方法。它可以解决图像背景复杂性和异常模式的挑战，并具有较高的准确性和鲁棒性。 |
| [^4] | [Sparsified Simultaneous Confidence Intervals for High-Dimensional Linear Models.](http://arxiv.org/abs/2307.07574) | 提出了一种稀疏化同时置信区间的方法，用于高维线性模型的统计推断。通过将某些区间的上下界收缩为零，该方法能够确定不重要的协变量并将其排除在最终模型之外，同时通过其他区间判断出可信和显著的协变量。 |

# 详细

[^1]: 外分布检测的核PCA

    Kernel PCA for Out-of-Distribution Detection

    [https://arxiv.org/abs/2402.02949](https://arxiv.org/abs/2402.02949)

    本论文提出了使用核PCA进行外分布检测的方法，通过在主成分子空间中引入非线性映射，实现了对内分布和外分布数据的有效区分。

    

    外分布（OoD）检测对于深度神经网络（DNN）的可靠性至关重要。现有的研究表明，直接应用于DNN特征的主成分分析（PCA）在检测来自内分布（InD）数据的OoD数据方面不足够。PCA的失败表明，仅通过在线性子空间中进行简单处理无法很好地将OoD和InD中的网络特征分离开来，而可以通过适当的非线性映射来解决。在这项工作中，我们利用核PCA（KPCA）框架进行OoD检测，寻找OoD和InD特征以显著不同的模式分配的子空间。我们设计了两种特征映射，在KPCA中引入非线性内核，以促进在主成分张成的子空间中InD和OoD数据之间的可分性。然后，通过在这种子空间中的重构误差，可以有效地得到$\mathcal{O}(1)$时间复杂度的检测结果。

    Out-of-Distribution (OoD) detection is vital for the reliability of Deep Neural Networks (DNNs). Existing works have shown the insufficiency of Principal Component Analysis (PCA) straightforwardly applied on the features of DNNs in detecting OoD data from In-Distribution (InD) data. The failure of PCA suggests that the network features residing in OoD and InD are not well separated by simply proceeding in a linear subspace, which instead can be resolved through proper nonlinear mappings. In this work, we leverage the framework of Kernel PCA (KPCA) for OoD detection, seeking subspaces where OoD and InD features are allocated with significantly different patterns. We devise two feature mappings that induce non-linear kernels in KPCA to advocate the separability between InD and OoD data in the subspace spanned by the principal components. Given any test sample, the reconstruction error in such subspace is then used to efficiently obtain the detection result with $\mathcal{O}(1)$ time comp
    
[^2]: 渐进域自适应：理论与算法

    Gradual Domain Adaptation: Theory and Algorithms. (arXiv:2310.13852v1 [cs.LG])

    [http://arxiv.org/abs/2310.13852](http://arxiv.org/abs/2310.13852)

    本文研究了渐进域自适应中的渐进自训练算法，提出了一个改进的泛化界限，并指出了中间域在源域和目标域之间均匀放置的重要性。

    

    无监督域自适应（UDA）是将模型从有标记的源域适应到无标记的目标域的一种一次性方法。尽管被广泛应用，但当源域和目标域之间的分布偏移较大时，UDA面临巨大挑战。渐进域自适应（GDA）通过使用中间域逐渐从源域适应到目标域来缓解这个限制。在这项工作中，我们首先从理论上分析了一种常见的GDA算法——渐进自训练，并提供了与Kumar等人（2020）相比显著改进的泛化界限。我们的理论分析得出一个有趣的观点：为了最小化目标域上的泛化误差，中间域的顺序应该均匀地放置在源域和目标域之间的Wasserstein测地线上。这个观点在中间域缺失或稀缺的情况下尤其有用，而这在现实世界的应用中经常出现。

    Unsupervised domain adaptation (UDA) adapts a model from a labeled source domain to an unlabeled target domain in a one-off way. Though widely applied, UDA faces a great challenge whenever the distribution shift between the source and the target is large. Gradual domain adaptation (GDA) mitigates this limitation by using intermediate domains to gradually adapt from the source to the target domain. In this work, we first theoretically analyze gradual self-training, a popular GDA algorithm, and provide a significantly improved generalization bound compared with Kumar et al. (2020). Our theoretical analysis leads to an interesting insight: to minimize the generalization error on the target domain, the sequence of intermediate domains should be placed uniformly along the Wasserstein geodesic between the source and target domains. The insight is particularly useful under the situation where intermediate domains are missing or scarce, which is often the case in real-world applications. Based
    
[^3]: Ano-SuPs: 通过识别可疑的区块进行制造产品的多尺度异常检测

    Ano-SuPs: Multi-size anomaly detection for manufactured products by identifying suspected patches. (arXiv:2309.11120v1 [stat.ML])

    [http://arxiv.org/abs/2309.11120](http://arxiv.org/abs/2309.11120)

    Ano-SuPs是一种通过识别可疑区块来进行制造产品的多尺度异常检测的两阶段策略方法。它可以解决图像背景复杂性和异常模式的挑战，并具有较高的准确性和鲁棒性。

    

    基于图像的系统因其提供丰富的制造状态信息、低实施成本和高采集速度而受到欢迎。然而，图像背景的复杂性和各种异常模式给现有的矩阵分解方法带来了新的挑战，这些方法不足以满足建模需求。此外，异常的不确定性可能导致异常的污染问题，使得设计的模型和方法对外部干扰非常敏感。为了解决这些挑战，我们提出了一种通过识别可疑区块（Ano-SuPs）来检测异常的两阶段策略异常检测方法。具体来说，我们提出了通过两次重建输入图像来检测带有异常的区块的方法：第一步是通过去除那些可疑区块来获得一组正常区块，第二步是使用这些正常区块来优化对带有异常区块的识别。我们通过实验证明了这种方法的效果。

    Image-based systems have gained popularity owing to their capacity to provide rich manufacturing status information, low implementation costs and high acquisition rates. However, the complexity of the image background and various anomaly patterns pose new challenges to existing matrix decomposition methods, which are inadequate for modeling requirements. Moreover, the uncertainty of the anomaly can cause anomaly contamination problems, making the designed model and method highly susceptible to external disturbances. To address these challenges, we propose a two-stage strategy anomaly detection method that detects anomalies by identifying suspected patches (Ano-SuPs). Specifically, we propose to detect the patches with anomalies by reconstructing the input image twice: the first step is to obtain a set of normal patches by removing those suspected patches, and the second step is to use those normal patches to refine the identification of the patches with anomalies. To demonstrate its ef
    
[^4]: 高维线性模型的稀疏化同时置信区间

    Sparsified Simultaneous Confidence Intervals for High-Dimensional Linear Models. (arXiv:2307.07574v1 [stat.ME])

    [http://arxiv.org/abs/2307.07574](http://arxiv.org/abs/2307.07574)

    提出了一种稀疏化同时置信区间的方法，用于高维线性模型的统计推断。通过将某些区间的上下界收缩为零，该方法能够确定不重要的协变量并将其排除在最终模型之外，同时通过其他区间判断出可信和显著的协变量。

    

    鉴于模型选择过程引入的不确定性难以考虑，对高维回归系数的统计推断具有挑战性。一个关键问题仍未解决，即是否可能以及如何将模型的推断嵌入到系数的同时推断中？为此，我们提出了一种称为稀疏化同时置信区间的概念。我们的区间在某些上下界上进行了稀疏，即缩小为零（例如，$[0,0]$），表示相应协变量的不重要性。这些协变量应该从最终模型中排除。其余的区间，无论是包含零（例如，$[-1,1]$或$[0,1]$）还是不包含零（例如，$[2,3]$），分别表示可信和显著的协变量。所提出的方法可以与各种选择过程相结合，使其非常适合比较它们的使用。

    Statistical inference of the high-dimensional regression coefficients is challenging because the uncertainty introduced by the model selection procedure is hard to account for. A critical question remains unsettled; that is, is it possible and how to embed the inference of the model into the simultaneous inference of the coefficients? To this end, we propose a notion of simultaneous confidence intervals called the sparsified simultaneous confidence intervals. Our intervals are sparse in the sense that some of the intervals' upper and lower bounds are shrunken to zero (i.e., $[0,0]$), indicating the unimportance of the corresponding covariates. These covariates should be excluded from the final model. The rest of the intervals, either containing zero (e.g., $[-1,1]$ or $[0,1]$) or not containing zero (e.g., $[2,3]$), indicate the plausible and significant covariates, respectively. The proposed method can be coupled with various selection procedures, making it ideal for comparing their u
    

