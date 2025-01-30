# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Universality of Coupling-based Normalizing Flows](https://arxiv.org/abs/2402.06578) | 我们提出了一个新颖的理论框架，用于理解基于耦合的标准化流的表达能力，并提出了一个新的分布普适性定理来克服以前工作的限制。这些结果支持耦合架构的表达能力，并弥补了实证结果和理论理解之间的差距。 |
| [^2] | [SPDE priors for uncertainty quantification of end-to-end neural data assimilation schemes](https://arxiv.org/abs/2402.01855) | SPDE先验在最优插值中的应用及其与神经网络的联合学习问题，为大规模地球物理数据集的时空插值提供了一种新的方法。 |
| [^3] | [Efficient Computation of Sparse and Robust Maximum Association Estimators](https://arxiv.org/abs/2311.17563) | 本文研究如何在高维稀疏设置中利用新的优化程序实现鲁棒稀疏关联估计，通过增广Lagrange算法和自适应梯度下降的组合，提供了更精确的算法，并展示了相对现有算法的优势。 |
| [^4] | [Provably Stable Feature Rankings with SHAP and LIME.](http://arxiv.org/abs/2401.15800) | 这项研究提出了一种通过利用多重假设检验的思想，来设计可靠地排名机器学习模型中最重要特征的特征归因方法，旨在解决SHAP和LIME等常用方法由于随机采样导致的高度不稳定性问题。实验证明了该方法的有效性和计算效率。 |
| [^5] | [Matrix Supermartingales and Randomized Matrix Concentration Inequalities.](http://arxiv.org/abs/2401.15567) | 本文提出了针对鞅相关或可交换随机对称矩阵的新集中不等式，这些不等式在多种尾条件下成立，在洛伊纳顺序表示，并且有时在任意数据相关停止时间都适用。 |
| [^6] | [Imputation using training labels and classification via label imputation.](http://arxiv.org/abs/2311.16877) | 本论文提出一种在填充缺失数据时将标签与输入堆叠的方法，能够显著提高填充效果，并同时填充标签和输入。该方法适用于各种类型的数据，且在实验证明具有有希望的准确性结果。 |

# 详细

[^1]: 关于基于耦合的标准化流的普适性

    On the Universality of Coupling-based Normalizing Flows

    [https://arxiv.org/abs/2402.06578](https://arxiv.org/abs/2402.06578)

    我们提出了一个新颖的理论框架，用于理解基于耦合的标准化流的表达能力，并提出了一个新的分布普适性定理来克服以前工作的限制。这些结果支持耦合架构的表达能力，并弥补了实证结果和理论理解之间的差距。

    

    我们提出了一个新颖的理论框架，用于理解基于耦合的标准化流（如RealNVP）的表达能力。尽管耦合流在科学应用中很普遍，但由于其受限的架构，对于耦合流的全面理解仍然困难。现有的定理在实际应用中存在限制，因为它们需要使用任意病态的神经网络。此外，我们还证明了这些结构本质上导致体积保持流，这是一个限制表达能力的基本约束。我们提出了一种新的基于分布的耦合标准化流普适性定理，克服了以前工作的几个限制。我们的结果支持耦合架构具有表达能力的普遍经验，并为选择耦合函数的表达能力提供了细致入微的观点，填补了实证结果和理论理解之间的差距。

    We present a novel theoretical framework for understanding the expressive power of coupling-based normalizing flows such as RealNVP. Despite their prevalence in scientific applications, a comprehensive understanding of coupling flows remains elusive due to their restricted architectures. Existing theorems fall short as they require the use of arbitrarily ill-conditioned neural networks, limiting practical applicability. Additionally, we demonstrate that these constructions inherently lead to volume-preserving flows, a property which we show to be a fundamental constraint for expressivity. We propose a new distributional universality theorem for coupling-based normalizing flows, which overcomes several limitations of prior work. Our results support the general wisdom that the coupling architecture is expressive and provide a nuanced view for choosing the expressivity of coupling functions, bridging a gap between empirical results and theoretical understanding.
    
[^2]: SPDE先验在端到端神经数据同化方案的不确定性量化中的应用

    SPDE priors for uncertainty quantification of end-to-end neural data assimilation schemes

    [https://arxiv.org/abs/2402.01855](https://arxiv.org/abs/2402.01855)

    SPDE先验在最优插值中的应用及其与神经网络的联合学习问题，为大规模地球物理数据集的时空插值提供了一种新的方法。

    

    大规模地球物理数据集的时空插值通常通过最优插值(Optimal Interpolation，OI)和更复杂的基于模型或数据驱动的数据同化技术来处理。在过去的十年中，随机偏微分方程(Spatio-temporal Partial Differential Equations，SPDE)和高斯马尔科夫随机场(Gaussian Markov Random Fields，GMRF)之间的联系开辟了一条新的途径，用于处理最优插值中的大数据集和物理诱导协方差矩阵。深度学习社区的最新进展也使得可以将这个问题视为嵌入数据同化变分框架的神经网络体系结构的联合学习问题。重建任务被视为一个包含在变分内部成本中的先验学习问题和后者的基于梯度的最小化：先验模型和求解器都被表示为具有自动微分的神经网络，可以通过最小化损失函数来训练，该损失函数通常被表示为一些真实值和重建值之间的均方误差。

    The spatio-temporal interpolation of large geophysical datasets has historically been adressed by Optimal Interpolation (OI) and more sophisticated model-based or data-driven DA techniques. In the last ten years, the link established between Stochastic Partial Differential Equations (SPDE) and Gaussian Markov Random Fields (GMRF) opened a new way of handling both large datasets and physically-induced covariance matrix in Optimal Interpolation. Recent advances in the deep learning community also enables to adress this problem as neural architecture embedding data assimilation variational framework. The reconstruction task is seen as a joint learning problem of the prior involved in the variational inner cost and the gradient-based minimization of the latter: both prior models and solvers are stated as neural networks with automatic differentiation which can be trained by minimizing a loss function, typically stated as the mean squared error between some ground truth and the reconstructi
    
[^3]: 高效计算稀疏和鲁棒最大关联估计量

    Efficient Computation of Sparse and Robust Maximum Association Estimators

    [https://arxiv.org/abs/2311.17563](https://arxiv.org/abs/2311.17563)

    本文研究如何在高维稀疏设置中利用新的优化程序实现鲁棒稀疏关联估计，通过增广Lagrange算法和自适应梯度下降的组合，提供了更精确的算法，并展示了相对现有算法的优势。

    

    虽然鲁棒统计估计量受到异常值的影响较小，但它们的计算通常更具挑战性，特别是在高维稀疏设置中。新的优化程序，主要在计算机科学领域开发，为鲁棒统计领域提供了新的可能性。本文研究了如何利用这些程序来实现鲁棒稀疏关联估计。该问题被拆分为一个鲁棒估计步骤，接着是一个余项解耦的（双边）凸问题的优化。采用增广Lagrange算法和自适应梯度下降的组合，还包括适当的约束条件以诱导稀疏性。我们提供了有关算法精度的结果，并展示了在这一背景下相对现有算法的优势。高维实证示例强调了该方法的实用性。

    arXiv:2311.17563v2 Announce Type: replace-cross  Abstract: Although robust statistical estimators are less affected by outlying observations, their computation is usually more challenging. This is particularly the case in high-dimensional sparse settings. The availability of new optimization procedures, mainly developed in the computer science domain, offers new possibilities for the field of robust statistics. This paper investigates how such procedures can be used for robust sparse association estimators. The problem can be split into a robust estimation step followed by an optimization for the remaining decoupled, (bi-)convex problem. A combination of the augmented Lagrangian algorithm and adaptive gradient descent is implemented to also include suitable constraints for inducing sparsity. We provide results concerning the precision of the algorithm and show the advantages over existing algorithms in this context. High-dimensional empirical examples underline the usefulness of this p
    
[^4]: 使用SHAP和LIME进行可证明稳定的特征排名

    Provably Stable Feature Rankings with SHAP and LIME. (arXiv:2401.15800v1 [stat.ML])

    [http://arxiv.org/abs/2401.15800](http://arxiv.org/abs/2401.15800)

    这项研究提出了一种通过利用多重假设检验的思想，来设计可靠地排名机器学习模型中最重要特征的特征归因方法，旨在解决SHAP和LIME等常用方法由于随机采样导致的高度不稳定性问题。实验证明了该方法的有效性和计算效率。

    

    特征归因是了解机器学习模型预测的普遍工具。然而，用于评分输入变量的常用方法，如SHAP和LIME，由于随机采样而具有高度不稳定性。借鉴多重假设检验的思想，我们设计了能够以高概率正确排名最重要特征的归因方法。我们的算法RankSHAP保证$K$个最高Shapley值具有超过$1-\alpha$的正确排序概率。实证结果证明了其有效性和令人印象深刻的计算效率。我们还在之前的工作基础上为LIME提供了类似的结果，确保以正确顺序选择最重要的特征。

    Feature attributions are ubiquitous tools for understanding the predictions of machine learning models. However, popular methods for scoring input variables such as SHAP and LIME suffer from high instability due to random sampling. Leveraging ideas from multiple hypothesis testing, we devise attribution methods that correctly rank the most important features with high probability. Our algorithm RankSHAP guarantees that the $K$ highest Shapley values have the proper ordering with probability exceeding $1-\alpha$. Empirical results demonstrate its validity and impressive computational efficiency. We also build on previous work to yield similar results for LIME, ensuring the most important features are selected in the right order.
    
[^5]: 矩阵超鞅和随机矩阵集中不等式

    Matrix Supermartingales and Randomized Matrix Concentration Inequalities. (arXiv:2401.15567v1 [math.PR])

    [http://arxiv.org/abs/2401.15567](http://arxiv.org/abs/2401.15567)

    本文提出了针对鞅相关或可交换随机对称矩阵的新集中不等式，这些不等式在多种尾条件下成立，在洛伊纳顺序表示，并且有时在任意数据相关停止时间都适用。

    

    我们在多种尾条件下，提出了针对鞅相关或可交换随机对称矩阵的新集中不等式，包括标准的切尔诺夫上界和自归一化重尾设置。这些不等式通常以洛伊纳顺序表示，并且有时在任意数据相关停止时间都成立。在此过程中，我们探索了矩阵超鞅和极值不等式的理论，可能具有独立的研究价值。

    We present new concentration inequalities for either martingale dependent or exchangeable random symmetric matrices under a variety of tail conditions, encompassing standard Chernoff bounds to self-normalized heavy-tailed settings. These inequalities are often randomized in a way that renders them strictly tighter than existing deterministic results in the literature, are typically expressed in the Loewner order, and are sometimes valid at arbitrary data-dependent stopping times.  Along the way, we explore the theory of matrix supermartingales and maximal inequalities, potentially of independent interest.
    
[^6]: 使用训练标签进行填充和通过标签填充进行分类

    Imputation using training labels and classification via label imputation. (arXiv:2311.16877v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.16877](http://arxiv.org/abs/2311.16877)

    本论文提出一种在填充缺失数据时将标签与输入堆叠的方法，能够显著提高填充效果，并同时填充标签和输入。该方法适用于各种类型的数据，且在实验证明具有有希望的准确性结果。

    

    在实际应用中，缺失数据是一个常见的问题。已经开发了各种填充方法来处理缺失数据。然而，尽管训练数据通常都有标签，但常见的填充方法通常只依赖于输入而忽略标签。在这项工作中，我们阐述了将标签堆叠到输入中可以显着提高输入的填充效果。此外，我们提出了一种分类策略，该策略将预测的测试标签初始化为缺失值，并将标签与输入堆叠在一起进行填充。这样可以同时填充标签和输入。而且，该技术能够处理具有缺失标签的训练数据，无需任何先前的填充，并且适用于连续型、分类型或混合型数据。实验证明在准确性方面取得了有希望的结果。

    Missing data is a common problem in practical settings. Various imputation methods have been developed to deal with missing data. However, even though the label is usually available in the training data, the common practice of imputation usually only relies on the input and ignores the label. In this work, we illustrate how stacking the label into the input can significantly improve the imputation of the input. In addition, we propose a classification strategy that initializes the predicted test label with missing values and stacks the label with the input for imputation. This allows imputing the label and the input at the same time. Also, the technique is capable of handling data training with missing labels without any prior imputation and is applicable to continuous, categorical, or mixed-type data. Experiments show promising results in terms of accuracy.
    

