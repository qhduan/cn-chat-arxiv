# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Causal disentanglement of multimodal data.](http://arxiv.org/abs/2310.18471) | 这篇论文介绍了一种可以利用多模态数据和已知物理学知识发现因果关系的因果表示学习算法。 |
| [^2] | [Conditional Sampling of Variational Autoencoders via Iterated Approximate Ancestral Sampling.](http://arxiv.org/abs/2308.09078) | 本文提出了两种原始方法来解决变分自编码器（VAE）条件采样中的困难，并在采样任务中展示了改进性能。 |
| [^3] | [Multi-Source Domain Adaptation through Dataset Dictionary Learning in Wasserstein Space.](http://arxiv.org/abs/2307.14953) | 本文提出了一种基于字典学习和最优传输的MSDA框架，通过将每个域表示为字典原子的Wasserstein重心来缓解数据分布偏移。根据该字典，提出了两种新的MSDA方法，分别基于目标域标记样本的重构和在原子分布上学习的分类器的集成。在多个基准测试集上进行的实验证明，这些方法在分类任务上取得了显著的改进效果。 |
| [^4] | [Spuriosity Didn't Kill the Classifier: Using Invariant Predictions to Harness Spurious Features.](http://arxiv.org/abs/2307.09933) | 本研究通过理论证明和算法提出，展示了在没有标签的情况下如何利用不稳定特征来提高分类器的性能。 |
| [^5] | [Adaptive Linear Estimating Equations.](http://arxiv.org/abs/2307.07320) | 本文提出了一种解决自适应线性回归模型中非正态渐近行为的方法，使用自适应线性估计方程构建去偏估计量，并在多臂老虎机的背景下保持了最小二乘估计量的非渐近性能。 |
| [^6] | [Solving Kernel Ridge Regression with Gradient-Based Optimization Methods.](http://arxiv.org/abs/2306.16838) | 本研究提出了一种新的方法来解决核岭回归问题，通过等价的目标函数形式和基于梯度的优化方法，我们不仅可以使用其他惩罚方法，还能够从梯度下降的角度研究核岭回归。通过提前停止的正则化，我们推导出了一个闭合解，即核梯度流（KGF），并证明了KGF和KRR之间的差异。我们还将KRR泛化，使用$\ell_1$和$\ell_\infty$惩罚方法，并发现使用这些方法得到的解与前向分步回归和符号梯度下降结合提前停止得到的解非常相似。因此，我们减少了计算复杂度重的近端梯度下降算法的需求。 |
| [^7] | [More PAC-Bayes bounds: From bounded losses, to losses with general tail behaviors, to anytime-validity.](http://arxiv.org/abs/2306.12214) | 本文提出了一种新的高概率PAC-Bayes界限，在有界和一般尾部行为的损失中均适用。此外，这些界限还能够保持随时有效性。 |
| [^8] | [Hierarchical clustering with dot products recovers hidden tree structure.](http://arxiv.org/abs/2305.15022) | 本文发现一种基于点积的层次聚类算法，可以通过最大平均点积合并聚类，并且输出的树结构可用于准确估计数据的生成层次结构，树形恢复性能优于现有方法。 |
| [^9] | [Scale Matters: Attribution Meets the Wavelet Domain to Explain Model Sensitivity to Image Corruptions.](http://arxiv.org/abs/2305.14979) | 该论文介绍了一种基于小波域的属性方法WCAM，能够解释神经网络模型对图像损坏的敏感性，确定预测的足够信息，并阐明缩放如何增加准确性。 |
| [^10] | [Zero-Shot Batch-Level Anomaly Detection.](http://arxiv.org/abs/2302.07849) | 本文提出了一种名为“自适应中心表示”的方法，用于零样本批次级异常检测。该方法利用批量归一化来训练现成的深度异常检测器，可以自动零样本泛化为未见过的AD任务。在实验中，该方法显示出了在多种数据集上的优秀表现，对表格数据进行了零样本AD。 |
| [^11] | [Versatile Energy-Based Probabilistic Models for High Energy Physics.](http://arxiv.org/abs/2302.00695) | 本文提出了一个多功能的能量概率模型，用于描述高能物理事件，可用于参数化的事件生成，异常信号探测以及粒子识别。 |
| [^12] | [Implicit models, latent compression, intrinsic biases, and cheap lunches in community detection.](http://arxiv.org/abs/2210.09186) | 本文提出了一种将社区检测目标与其对应的隐式网络生成模型相联系的解决方案，可以计算网络在任意目标下的描述长度，比较不同算法的性能，同时还可以访问隐式模型。 |
| [^13] | [Dual control variate for faster black-box variational inference.](http://arxiv.org/abs/2210.07290) | 本论文提出了双控制变量方法，能够同时减少数据子抽样和蒙特卡罗抽样带来的梯度估计方差，提高黑盒变分推断的准确性和效率。 |

# 详细

[^1]: 多模态数据的因果分解

    Causal disentanglement of multimodal data. (arXiv:2310.18471v1 [cs.LG])

    [http://arxiv.org/abs/2310.18471](http://arxiv.org/abs/2310.18471)

    这篇论文介绍了一种可以利用多模态数据和已知物理学知识发现因果关系的因果表示学习算法。

    

    因果表示学习算法发现了数据的较低维度表示，可以对因果关系进行可解释的解释；由于实现这样的可解释表示很具挑战性，许多因果学习算法利用了指示先验信息的元素，例如（线性）结构因果模型、干预数据或弱监督。然而，在探索性因果表示学习中，这些元素和先验信息可能不可用或不合适。相反，科学数据集通常具有多个模态或基于物理学的约束，并且已经证明在完全无监督的设置中使用这种科学的多模态数据可以改善因果分解。因此，我们引入了一种因果表示学习算法（causalPIMA），它可以利用多模态数据和已知的物理学知识发现具有因果关系的重要特征。我们的创新算法利用新的可微参数化来学习这种因果关系。

    Causal representation learning algorithms discover lower-dimensional representations of data that admit a decipherable interpretation of cause and effect; as achieving such interpretable representations is challenging, many causal learning algorithms utilize elements indicating prior information, such as (linear) structural causal models, interventional data, or weak supervision. Unfortunately, in exploratory causal representation learning, such elements and prior information may not be available or warranted. Alternatively, scientific datasets often have multiple modalities or physics-based constraints, and the use of such scientific, multimodal data has been shown to improve disentanglement in fully unsupervised settings. Consequently, we introduce a causal representation learning algorithm (causalPIMA) that can use multimodal data and known physics to discover important features with causal relationships. Our innovative algorithm utilizes a new differentiable parametrization to lear
    
[^2]: 通过迭代近似祖先采样实现变分自编码器的条件采样

    Conditional Sampling of Variational Autoencoders via Iterated Approximate Ancestral Sampling. (arXiv:2308.09078v1 [cs.LG])

    [http://arxiv.org/abs/2308.09078](http://arxiv.org/abs/2308.09078)

    本文提出了两种原始方法来解决变分自编码器（VAE）条件采样中的困难，并在采样任务中展示了改进性能。

    

    在各种应用中，如缺失数据填充，需要对变分自编码器（VAE）进行条件采样，但这是计算上不可行的。渐近精确条件采样的原则选择是Metropolis-within-Gibbs（MWG）。然而，我们观察到VAE倾向于学习结构化潜变量空间，这是一个常见的期望特性，但却导致MWG采样器远离目标分布。本文克服了MWG的局限性：我们系统地概述了在VAE上上述局限性，并提出了两种原始方法来解决这些问题，并在一组采样任务上展示了所提方法的改进性能。

    Conditional sampling of variational autoencoders (VAEs) is needed in various applications, such as missing data imputation, but is computationally intractable. A principled choice for asymptotically exact conditional sampling is Metropolis-within-Gibbs (MWG). However, we observe that the tendency of VAEs to learn a structured latent space, a commonly desired property, can cause the MWG sampler to get "stuck" far from the target distribution. This paper mitigates the limitations of MWG: we systematically outline the pitfalls in the context of VAEs, propose two original methods that address these pitfalls, and demonstrate an improved performance of the proposed methods on a set of sampling tasks.
    
[^3]: 在Wasserstein空间中通过数据集字典学习进行多源域自适应

    Multi-Source Domain Adaptation through Dataset Dictionary Learning in Wasserstein Space. (arXiv:2307.14953v1 [cs.LG])

    [http://arxiv.org/abs/2307.14953](http://arxiv.org/abs/2307.14953)

    本文提出了一种基于字典学习和最优传输的MSDA框架，通过将每个域表示为字典原子的Wasserstein重心来缓解数据分布偏移。根据该字典，提出了两种新的MSDA方法，分别基于目标域标记样本的重构和在原子分布上学习的分类器的集成。在多个基准测试集上进行的实验证明，这些方法在分类任务上取得了显著的改进效果。

    

    本文旨在解决多源域自适应（MSDA）问题，该问题旨在在从多个标记的源域转移知识到未标记的目标域时缓解数据分布偏移。我们提出了一种基于字典学习和最优传输的新型MSDA框架。我们将MSDA中的每个域解释为经验分布。因此，我们将每个域表达为字典原子的Wasserstein重心，这些原子是经验分布。我们提出了一种新的通过小批量学习的算法DaDiL：（i）原子分布；（ii）重心坐标矩阵。根据我们的字典，我们提出了两种新的MSDA方法：DaDiL-R，基于目标域标记样本的重构；DaDiL-E，基于在原子分布上学习的分类器的集成。我们在3个基准测试集中评估了我们的方法：Caltech-Office、Office 31和CRWU，在分类上改进了以前的最先进技术3.15％、2.29％和7.71％。

    This paper seeks to solve Multi-Source Domain Adaptation (MSDA), which aims to mitigate data distribution shifts when transferring knowledge from multiple labeled source domains to an unlabeled target domain. We propose a novel MSDA framework based on dictionary learning and optimal transport. We interpret each domain in MSDA as an empirical distribution. As such, we express each domain as a Wasserstein barycenter of dictionary atoms, which are empirical distributions. We propose a novel algorithm, DaDiL, for learning via mini-batches: (i) atom distributions; (ii) a matrix of barycentric coordinates. Based on our dictionary, we propose two novel methods for MSDA: DaDil-R, based on the reconstruction of labeled samples in the target domain, and DaDiL-E, based on the ensembling of classifiers learned on atom distributions. We evaluate our methods in 3 benchmarks: Caltech-Office, Office 31, and CRWU, where we improved previous state-of-the-art by 3.15%, 2.29%, and 7.71% in classification 
    
[^4]: Spuriosity并没有导致分类器失败：利用不变的预测来利用虚假特征

    Spuriosity Didn't Kill the Classifier: Using Invariant Predictions to Harness Spurious Features. (arXiv:2307.09933v1 [cs.LG])

    [http://arxiv.org/abs/2307.09933](http://arxiv.org/abs/2307.09933)

    本研究通过理论证明和算法提出，展示了在没有标签的情况下如何利用不稳定特征来提高分类器的性能。

    

    为了避免在域外数据上的失败，最近的研究试图提取具有与标签在不同域之间稳定或不变关系的特征，舍弃与标签在不同域之间关系变化的"虚假"或不稳定特征。然而，不稳定特征常常携带关于标签的补充信息，如果在测试域中正确使用，可以提高性能。我们的主要贡献是显示在没有标签的情况下学习如何在测试域中使用这些不稳定特征是可能的。特别是，我们证明基于稳定特征的伪标签提供了足够的指导来做到这一点，前提是在给定标签的条件下，稳定特征和不稳定特征是条件独立的。基于这个理论洞见，我们提出了稳定特征增强（SFB）算法：(i)学习一个能够分离稳定特征和条件独立不稳定特征的预测器；(ii)使用稳定特征预测来适应测试域

    To avoid failures on out-of-distribution data, recent works have sought to extract features that have a stable or invariant relationship with the label across domains, discarding the "spurious" or unstable features whose relationship with the label changes across domains. However, unstable features often carry complementary information about the label that could boost performance if used correctly in the test domain. Our main contribution is to show that it is possible to learn how to use these unstable features in the test domain without labels. In particular, we prove that pseudo-labels based on stable features provide sufficient guidance for doing so, provided that stable and unstable features are conditionally independent given the label. Based on this theoretical insight, we propose Stable Feature Boosting (SFB), an algorithm for: (i) learning a predictor that separates stable and conditionally-independent unstable features; and (ii) using the stable-feature predictions to adapt t
    
[^5]: 自适应线性估计方程

    Adaptive Linear Estimating Equations. (arXiv:2307.07320v1 [math.ST])

    [http://arxiv.org/abs/2307.07320](http://arxiv.org/abs/2307.07320)

    本文提出了一种解决自适应线性回归模型中非正态渐近行为的方法，使用自适应线性估计方程构建去偏估计量，并在多臂老虎机的背景下保持了最小二乘估计量的非渐近性能。

    

    顺序数据收集已成为增强数据收集过程效率的广泛采用的技术。尽管具有优势，但这种数据收集机制常常给统计推断过程引入复杂性。例如，在自适应线性回归模型中，普通最小二乘（OLS）估计量可能表现出非正态的渐近行为，从而对准确的推断和解释提出挑战。本文提出了一种构建去偏估计量的通用方法，该方法采用自适应线性估计方程的思想，并在理论上保证了渐近正态性，并讨论了实现近似最优渐近方差的问题。我们的估计量的一个显著特点是，在多臂老虎机的背景下，我们的估计量保留了最小二乘估计量的非渐近性能，同时获得了渐近正态性。因此，本工作解决了自适应线性回归模型中非正态渐近行为的问题，并为统计推断提供了可靠的方法。

    Sequential data collection has emerged as a widely adopted technique for enhancing the efficiency of data gathering processes. Despite its advantages, such data collection mechanism often introduces complexities to the statistical inference procedure. For instance, the ordinary least squares (OLS) estimator in an adaptive linear regression model can exhibit non-normal asymptotic behavior, posing challenges for accurate inference and interpretation. In this paper, we propose a general method for constructing debiased estimator which remedies this issue. It makes use of the idea of adaptive linear estimating equations, and we establish theoretical guarantees of asymptotic normality, supplemented by discussions on achieving near-optimal asymptotic variance. A salient feature of our estimator is that in the context of multi-armed bandits, our estimator retains the non-asymptotic performance of the least square estimator while obtaining asymptotic normality property. Consequently, this work
    
[^6]: 用基于梯度的优化方法解决核岭回归问题

    Solving Kernel Ridge Regression with Gradient-Based Optimization Methods. (arXiv:2306.16838v1 [stat.ML])

    [http://arxiv.org/abs/2306.16838](http://arxiv.org/abs/2306.16838)

    本研究提出了一种新的方法来解决核岭回归问题，通过等价的目标函数形式和基于梯度的优化方法，我们不仅可以使用其他惩罚方法，还能够从梯度下降的角度研究核岭回归。通过提前停止的正则化，我们推导出了一个闭合解，即核梯度流（KGF），并证明了KGF和KRR之间的差异。我们还将KRR泛化，使用$\ell_1$和$\ell_\infty$惩罚方法，并发现使用这些方法得到的解与前向分步回归和符号梯度下降结合提前停止得到的解非常相似。因此，我们减少了计算复杂度重的近端梯度下降算法的需求。

    

    核岭回归（KRR）是线性岭回归的非线性推广。在这里，我们引入了KRR目标函数的等价形式，为使用其他惩罚方法和从梯度下降的角度研究核岭回归打开了可能。通过连续时间的视角，我们推导出了一个闭合解——核梯度流（KGF），通过提前停止的正则化，让我们能够在KGF和KRR之间理论上界定差异。我们用$\ell_1$和$\ell_\infty$惩罚方法将KRR泛化，并利用类似KGF和KRR之间的相似性，使用这些惩罚方法得到的解与使用前向分步回归（也称为坐标下降）和符号梯度下降结合提前停止得到的解非常相似。因此，减少了计算复杂度重的近端梯度下降算法的需求。

    Kernel ridge regression, KRR, is a non-linear generalization of linear ridge regression. Here, we introduce an equivalent formulation of the objective function of KRR, opening up both for using other penalties than the ridge penalty and for studying kernel ridge regression from the perspective of gradient descent. Using a continuous-time perspective, we derive a closed-form solution, kernel gradient flow, KGF, with regularization through early stopping, which allows us to theoretically bound the differences between KGF and KRR. We generalize KRR by replacing the ridge penalty with the $\ell_1$ and $\ell_\infty$ penalties and utilize the fact that analogously to the similarities between KGF and KRR, the solutions obtained when using these penalties are very similar to those obtained from forward stagewise regression (also known as coordinate descent) and sign gradient descent in combination with early stopping. Thus the need for computationally heavy proximal gradient descent algorithms
    
[^7]: 更多的PAC-Bayes Bounds：从有界损失到具有一般性尾部行为的损失，到任何时间均有效的损失。

    More PAC-Bayes bounds: From bounded losses, to losses with general tail behaviors, to anytime-validity. (arXiv:2306.12214v1 [stat.ML])

    [http://arxiv.org/abs/2306.12214](http://arxiv.org/abs/2306.12214)

    本文提出了一种新的高概率PAC-Bayes界限，在有界和一般尾部行为的损失中均适用。此外，这些界限还能够保持随时有效性。

    

    本文针对不同类型的损失提出了新的高概率PAC-Bayes界限。首先，针对有界范围的损失，我们提出了Catoni界的加强版本，适用于所有参数值的统一界。这导致了新的快速速率和混合速率上限，这些上限可解释性强且比文献中先前界限更紧。其次，针对更一般的尾部行为的损失，我们引入了两个新的无参数上限：当损失的累积生成函数有界时，我们引入了一个PAC-Bayes Chernoff类比，另一个上限是损失的二阶矩有界。这两个上限是利用一种基于可能事件空间的离散化的新技术获得的，“在概率”参数优化问题。最后，我们使用一种适用于任何现有界限的简单技术将所有先前结果扩展到任何时间有效的上限。

    In this paper, we present new high-probability PAC-Bayes bounds for different types of losses. Firstly, for losses with a bounded range, we present a strengthened version of Catoni's bound that holds uniformly for all parameter values. This leads to new fast rate and mixed rate bounds that are interpretable and tighter than previous bounds in the literature. Secondly, for losses with more general tail behaviors, we introduce two new parameter-free bounds: a PAC-Bayes Chernoff analogue when the loss' cumulative generating function is bounded, and a bound when the loss' second moment is bounded. These two bounds are obtained using a new technique based on a discretization of the space of possible events for the "in probability" parameter optimization problem. Finally, we extend all previous results to anytime-valid bounds using a simple technique applicable to any existing bound.
    
[^8]: 基于点积的层次聚类可以恢复隐藏的树形结构

    Hierarchical clustering with dot products recovers hidden tree structure. (arXiv:2305.15022v1 [stat.ML])

    [http://arxiv.org/abs/2305.15022](http://arxiv.org/abs/2305.15022)

    本文发现一种基于点积的层次聚类算法，可以通过最大平均点积合并聚类，并且输出的树结构可用于准确估计数据的生成层次结构，树形恢复性能优于现有方法。

    

    本文提供了一个对于已有凝聚聚类算法的新视角，专注于层次结构的恢复。我们建议一种简单的标准算法变体，其中聚类是通过最大平均点积而不是最小距离或簇内方差来合并的。我们证明了此算法输出的树可以作为数据生成层次结构的可靠估计。关键技术创新在于理解模型中的层次信息如何转化为可从数据中恢复的树形几何信息，并同时增长样本大小和数据维数的好处。我们在真实数据上展示了优于现有方法（如UPGMA、Ward's方法和HDBSCAN）的树形恢复性能。

    In this paper we offer a new perspective on the well established agglomerative clustering algorithm, focusing on recovery of hierarchical structure. We recommend a simple variant of the standard algorithm, in which clusters are merged by maximum average dot product and not, for example, by minimum distance or within-cluster variance. We demonstrate that the tree output by this algorithm provides a bona fide estimate of generative hierarchical structure in data, under a generic probabilistic graphical model. The key technical innovations are to understand how hierarchical information in this model translates into tree geometry which can be recovered from data, and to characterise the benefits of simultaneously growing sample size and data dimension. We demonstrate superior tree recovery performance with real data over existing approaches such as UPGMA, Ward's method, and HDBSCAN.
    
[^9]: 尺度很重要：基于小波域的属性方法解释模型对图像损坏的敏感性

    Scale Matters: Attribution Meets the Wavelet Domain to Explain Model Sensitivity to Image Corruptions. (arXiv:2305.14979v1 [cs.CV])

    [http://arxiv.org/abs/2305.14979](http://arxiv.org/abs/2305.14979)

    该论文介绍了一种基于小波域的属性方法WCAM，能够解释神经网络模型对图像损坏的敏感性，确定预测的足够信息，并阐明缩放如何增加准确性。

    

    神经网络在计算机视觉方面表现出了出色的性能，但它们在实际应用中的部署由于对图像损坏的敏感性而具有挑战性。现有的属性方法对于解释对图像损坏的敏感性是无效的，而强健性领域的文献仅提供基于模型的解释。然而，在图像损坏的情况下，审查模型的行为能力对于提高用户信任至关重要。为此，我们介绍了Wavelet sCale Attribution Method (WCAM)，它是从像素域到空间尺度域的属性方法的概括。在空间尺度域中进行属性揭示了模型的关注点和尺度。我们展示WCAM解释了模型在图像破坏下的失效，确定了预测的足够信息，并解释了如何通过缩放增加准确性。

    Neural networks have shown remarkable performance in computer vision, but their deployment in real-world scenarios is challenging due to their sensitivity to image corruptions. Existing attribution methods are uninformative for explaining the sensitivity to image corruptions, while the literature on robustness only provides model-based explanations. However, the ability to scrutinize models' behavior under image corruptions is crucial to increase the user's trust. Towards this end, we introduce the Wavelet sCale Attribution Method (WCAM), a generalization of attribution from the pixel domain to the space-scale domain. Attribution in the space-scale domain reveals where and on what scales the model focuses. We show that the WCAM explains models' failures under image corruptions, identifies sufficient information for prediction, and explains how zoom-in increases accuracy.
    
[^10]: 零样本批次级异常检测

    Zero-Shot Batch-Level Anomaly Detection. (arXiv:2302.07849v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.07849](http://arxiv.org/abs/2302.07849)

    本文提出了一种名为“自适应中心表示”的方法，用于零样本批次级异常检测。该方法利用批量归一化来训练现成的深度异常检测器，可以自动零样本泛化为未见过的AD任务。在实验中，该方法显示出了在多种数据集上的优秀表现，对表格数据进行了零样本AD。

    

    异常检测（AD）在许多安全关键的应用领域中发挥着关键作用。适应正常数据分布漂移的异常检测器调整，特别是当没有针对“新正常”进行训练的数据时，这一挑战导致产生了零样本AD技术。在本文中，我们提出了一种名为自适应中心表示（ACR）的简单而有效的方法，用于零样本批次级AD。我们的方法使用批量归一化来训练现成的深度异常检测器（例如深度SVDD）来适应一组相互关联的训练数据分布，使其能够自动零样本泛化为未见过的AD任务。这个简单的方法，批量归一化加元训练，是一种非常有效和多功能的工具。我们的结果展示了对表格数据的第一个零样本AD结果，并在来自专业领域的图像数据的零样本异常检测和分段方面优于现有方法。

    Anomaly detection (AD) plays a crucial role in many safety-critical application domains. The challenge of adapting an anomaly detector to drift in the normal data distribution, especially when no training data is available for the "new normal," has led to the development of zero-shot AD techniques. In this paper, we propose a simple yet effective method called Adaptive Centered Representations (ACR) for zero-shot batch-level AD. Our approach trains off-the-shelf deep anomaly detectors (such as deep SVDD) to adapt to a set of inter-related training data distributions in combination with batch normalization, enabling automatic zero-shot generalization for unseen AD tasks. This simple recipe, batch normalization plus meta-training, is a highly effective and versatile tool. Our results demonstrate the first zero-shot AD results for tabular data and outperform existing methods in zero-shot anomaly detection and segmentation on image data from specialized domains.
    
[^11]: 多功能能量概率模型在高能物理中的应用

    Versatile Energy-Based Probabilistic Models for High Energy Physics. (arXiv:2302.00695v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.00695](http://arxiv.org/abs/2302.00695)

    本文提出了一个多功能的能量概率模型，用于描述高能物理事件，可用于参数化的事件生成，异常信号探测以及粒子识别。

    

    作为一种经典的生成建模方法，基于能量的模型具有能量函数形式灵活性的天然优势。最近，基于能量的模型在计算机视觉和自然语言处理中建模高维数据方面取得了巨大成功。与这些进展一致，我们建立了一个多功能能量概率模型，用于描述来自大型强子对撞机的高能物理事件。该框架基于一个强大的生成模型，并描述了更高阶的粒子间相互作用，适用于不同的编码体系结构和隐式生成。在应用方面，它可以作为强大的参数化事件生成器用于物理仿真，一种泛用的无假设关联的异常信号探测器，以及用于粒子识别的增强事件分类器。

    As a classical generative modeling approach, energy-based models have the natural advantage of flexibility in the form of the energy function. Recently, energy-based models have achieved great success in modeling high-dimensional data in computer vision and natural language processing. In line with these advancements, we build a multi-purpose energy-based probabilistic model for High Energy Physics events at the Large Hadron Collider. This framework builds on a powerful generative model and describes higher-order inter-particle interactions.It suits different encoding architectures and builds on implicit generation. As for applicational aspects, it can serve as a powerful parameterized event generator for physics simulation, a generic anomalous signal detector free from spurious correlations, and an augmented event classifier for particle identification.
    
[^12]: 隐式模型、潜在压缩、内在偏差和廉价午餐在社区检测中的应用

    Implicit models, latent compression, intrinsic biases, and cheap lunches in community detection. (arXiv:2210.09186v6 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2210.09186](http://arxiv.org/abs/2210.09186)

    本文提出了一种将社区检测目标与其对应的隐式网络生成模型相联系的解决方案，可以计算网络在任意目标下的描述长度，比较不同算法的性能，同时还可以访问隐式模型。

    

    社区检测的任务旨在将网络划分为节点集群，以总结其大规模结构，已经引出了许多具有不同目标的竞争算法。 一些社区检测方法是推断性的，通过概率生成模型明确地导出聚类目标，而其他方法是描述性的，根据特定应用的目标将网络分成子集，这使得在同一规模下比较这些方法变得具有挑战性。本文提出了将任何社区检测目标（推断性或描述性）与其相应的隐式网络生成模型相联系的解决方案。这使我们能够计算网络及其在任意目标下的分区的描述长度，无需“地面实况”标签即可比较不同算法的性能，同时还可以访问隐式模型，这是其他方法所不具备的。

    The task of community detection, which aims to partition a network into clusters of nodes to summarize its large-scale structure, has spawned the development of many competing algorithms with varying objectives. Some community detection methods are inferential, explicitly deriving the clustering objective through a probabilistic generative model, while other methods are descriptive, dividing a network according to an objective motivated by a particular application, making it challenging to compare these methods on the same scale. Here we present a solution to this problem that associates any community detection objective, inferential or descriptive, with its corresponding implicit network generative model. This allows us to compute the description length of a network and its partition under arbitrary objectives, providing a principled measure to compare the performance of different algorithms without the need for "ground truth" labels. Our approach also gives access to instances of the
    
[^13]: 双控制变量加速黑盒变分推断

    Dual control variate for faster black-box variational inference. (arXiv:2210.07290v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.07290](http://arxiv.org/abs/2210.07290)

    本论文提出了双控制变量方法，能够同时减少数据子抽样和蒙特卡罗抽样带来的梯度估计方差，提高黑盒变分推断的准确性和效率。

    

    黑盒变分推断是一种广泛使用的贝叶斯后验推断框架，但在某些情况下，梯度估计中的高方差会损害准确性和效率。这种方差来自两个随机源：数据子抽样和蒙特卡罗抽样。现有的控制变量仅解决蒙特卡罗噪声，而增量梯度方法通常仅解决数据子抽样，我们提出了一种新的“双”控制变量，能够同时减少两种噪声源的方差。我们确认这导致了减少方差和在多个现实世界应用中提高优化效果。

    Black-box variational inference is a widely-used framework for Bayesian posterior inference, but in some cases suffers from high variance in gradient estimates, harming accuracy and efficiency. This variance comes from two sources of randomness: Data subsampling and Monte Carlo sampling. Whereas existing control variates only address Monte Carlo noise and incremental gradient methods typically only address data subsampling, we propose a new "dual" control variate capable of jointly reducing variance from both sources of noise. We confirm that this leads to reduced variance and improved optimization in several real-world applications.
    

