# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Conformal online model aggregation](https://arxiv.org/abs/2403.15527) | 该论文提出了一种基于投票的在线依从模型聚合方法，可以根据过去表现调整模型权重。 |
| [^2] | [An Ordering of Divergences for Variational Inference with Factorized Gaussian Approximations](https://arxiv.org/abs/2403.13748) | 不同的散度排序可以通过它们的变分近似误估不确定性的各种度量，并且因子化近似无法同时匹配这些度量中的任意两个 |
| [^3] | [Pattern Recovery in Penalized and Thresholded Estimation and its Geometry.](http://arxiv.org/abs/2307.10158) | 我们提出了一种惩罚化和阈值化估计的模式恢复方法，并定义了模式和恢复条件。对于LASSO，无噪声恢复条件和互不表示条件起到了相同的作用。 |
| [^4] | [Spatiotemporal Besov Priors for Bayesian Inverse Problems.](http://arxiv.org/abs/2306.16378) | 本研究通过将贝索夫过程推广到时空领域，以更好地处理贝叶斯逆问题中的时空重建。通过替换随机系数，该方法能够保持边缘特征并模拟动态变化图像的时空相关性。 |

# 详细

[^1]: 依从在线模型聚合

    Conformal online model aggregation

    [https://arxiv.org/abs/2403.15527](https://arxiv.org/abs/2403.15527)

    该论文提出了一种基于投票的在线依从模型聚合方法，可以根据过去表现调整模型权重。

    

    依从预测为机器学习模型提供了一种合理的不确定性量化概念，而不需要做出强烈的分布假设。它适用于任何黑盒预测模型，并将点预测转换成具有预定义边际覆盖保证的集预测。然而，依从预测只在事先确定底层机器学习模型的情况下起作用。依从预测中相对较少涉及的问题是模型选择和/或聚合：对于给定的问题，应该如何依从化众多预测方法（随机森林、神经网络、正则化线性模型等）？本文提出了一种新的依从模型聚合方法，用于在线设置，该方法基于将来自多个算法的预测集进行投票，其中根据过去表现调整模型上的权重。

    arXiv:2403.15527v1 Announce Type: cross  Abstract: Conformal prediction equips machine learning models with a reasonable notion of uncertainty quantification without making strong distributional assumptions. It wraps around any black-box prediction model and converts point predictions into set predictions that have a predefined marginal coverage guarantee. However, conformal prediction only works if we fix the underlying machine learning model in advance. A relatively unaddressed issue in conformal prediction is that of model selection and/or aggregation: for a given problem, which of the plethora of prediction methods (random forests, neural nets, regularized linear models, etc.) should we conformalize? This paper proposes a new approach towards conformal model aggregation in online settings that is based on combining the prediction sets from several algorithms by voting, where weights on the models are adapted over time based on past performance.
    
[^2]: 变分推断中因子化高斯近似的差异排序

    An Ordering of Divergences for Variational Inference with Factorized Gaussian Approximations

    [https://arxiv.org/abs/2403.13748](https://arxiv.org/abs/2403.13748)

    不同的散度排序可以通过它们的变分近似误估不确定性的各种度量，并且因子化近似无法同时匹配这些度量中的任意两个

    

    在变分推断（VI）中，给定一个难以处理的分布$p$，问题是从一些更易处理的族$\mathcal{Q}$中计算最佳近似$q$。通常情况下，这种近似是通过最小化Kullback-Leibler (KL)散度来找到的。然而，存在其他有效的散度选择，当$\mathcal{Q}$不包含$p$时，每个散度都支持不同的解决方案。我们分析了在高斯的密集协方差矩阵被对角协方差矩阵的高斯近似所影响的VI结果中，散度选择如何影响VI结果。在这种设置中，我们展示了不同的散度可以通过它们的变分近似误估不确定性的各种度量，如方差、精度和熵，进行\textit{排序}。我们还得出一个不可能定理，表明无法通过因子化近似同时匹配这些度量中的任意两个；因此

    arXiv:2403.13748v1 Announce Type: cross  Abstract: Given an intractable distribution $p$, the problem of variational inference (VI) is to compute the best approximation $q$ from some more tractable family $\mathcal{Q}$. Most commonly the approximation is found by minimizing a Kullback-Leibler (KL) divergence. However, there exist other valid choices of divergences, and when $\mathcal{Q}$ does not contain~$p$, each divergence champions a different solution. We analyze how the choice of divergence affects the outcome of VI when a Gaussian with a dense covariance matrix is approximated by a Gaussian with a diagonal covariance matrix. In this setting we show that different divergences can be \textit{ordered} by the amount that their variational approximations misestimate various measures of uncertainty, such as the variance, precision, and entropy. We also derive an impossibility theorem showing that no two of these measures can be simultaneously matched by a factorized approximation; henc
    
[^3]: 惩罚化和阈值化估计中的模式恢复及其几何

    Pattern Recovery in Penalized and Thresholded Estimation and its Geometry. (arXiv:2307.10158v1 [math.ST])

    [http://arxiv.org/abs/2307.10158](http://arxiv.org/abs/2307.10158)

    我们提出了一种惩罚化和阈值化估计的模式恢复方法，并定义了模式和恢复条件。对于LASSO，无噪声恢复条件和互不表示条件起到了相同的作用。

    

    我们考虑惩罚估计的框架，其中惩罚项由实值的多面体规范给出，其中包括诸如LASSO（以及其许多变体如广义LASSO）、SLOPE、OSCAR、PACS等方法。每个估计器可以揭示未知参数向量的不同结构或“模式”。我们定义了基于次微分的模式的一般概念，并形式化了一种衡量其复杂性的方法。对于模式恢复，我们提供了一个特定模式以正概率被该过程检测到的最小条件，即所谓的可达性条件。利用我们的方法，我们还引入了更强的无噪声恢复条件。对于LASSO，众所周知，互不表示条件是使模式恢复的概率大于1/2所必需的，并且我们展示了无噪声恢复起到了完全相同的作用，从而扩展和统一了互不表示条件。

    We consider the framework of penalized estimation where the penalty term is given by a real-valued polyhedral gauge, which encompasses methods such as LASSO (and many variants thereof such as the generalized LASSO), SLOPE, OSCAR, PACS and others. Each of these estimators can uncover a different structure or ``pattern'' of the unknown parameter vector. We define a general notion of patterns based on subdifferentials and formalize an approach to measure their complexity. For pattern recovery, we provide a minimal condition for a particular pattern to be detected by the procedure with positive probability, the so-called accessibility condition. Using our approach, we also introduce the stronger noiseless recovery condition. For the LASSO, it is well known that the irrepresentability condition is necessary for pattern recovery with probability larger than $1/2$ and we show that the noiseless recovery plays exactly the same role, thereby extending and unifying the irrepresentability conditi
    
[^4]: 贝索夫先验在贝叶斯逆问题中的时空应用

    Spatiotemporal Besov Priors for Bayesian Inverse Problems. (arXiv:2306.16378v1 [stat.ME])

    [http://arxiv.org/abs/2306.16378](http://arxiv.org/abs/2306.16378)

    本研究通过将贝索夫过程推广到时空领域，以更好地处理贝叶斯逆问题中的时空重建。通过替换随机系数，该方法能够保持边缘特征并模拟动态变化图像的时空相关性。

    

    近年来，科学技术的快速发展促使对捕捉数据特征（如突变或明显对比度）的适当统计工具的需求。许多数据科学应用需要从具有不连续性或奇异性的时间相关对象序列中进行时空重建，如带有边缘的动态计算机断层影像（CT）图像。传统的基于高斯过程（GP）的方法可能无法提供令人满意的解决方案，因为它们往往提供过度平滑的先验候选。最近，通过随机系数的小波展开定义的贝索夫过程（BP）被提出作为这类贝叶斯逆问题的更合适的先验。BP在成像分析中表现出优于GP的性能，能够产生保留边缘特征的重建结果，但没有自动地纳入动态变化图像中的时间相关性。本文将BP推广到时空领域（STBP），通过在小波展开中替换随机系数，实现了时空相关性的建模。

    Fast development in science and technology has driven the need for proper statistical tools to capture special data features such as abrupt changes or sharp contrast. Many applications in the data science seek spatiotemporal reconstruction from a sequence of time-dependent objects with discontinuity or singularity, e.g. dynamic computerized tomography (CT) images with edges. Traditional methods based on Gaussian processes (GP) may not provide satisfactory solutions since they tend to offer over-smooth prior candidates. Recently, Besov process (BP) defined by wavelet expansions with random coefficients has been proposed as a more appropriate prior for this type of Bayesian inverse problems. While BP outperforms GP in imaging analysis to produce edge-preserving reconstructions, it does not automatically incorporate temporal correlation inherited in the dynamically changing images. In this paper, we generalize BP to the spatiotemporal domain (STBP) by replacing the random coefficients in 
    

