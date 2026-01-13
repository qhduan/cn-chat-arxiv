# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reimagining Anomalies: What If Anomalies Were Normal?](https://arxiv.org/abs/2402.14469) | 方法提出了一种新颖的解释方法，生成多个反事实示例以捕获异常的多样概念，为用户提供对触发异常检测器机制的高级语义解释，允许探索“假设情景”。 |
| [^2] | [Learning Operators with Stochastic Gradient Descent in General Hilbert Spaces](https://arxiv.org/abs/2402.04691) | 本研究在一般希尔伯特空间中使用随机梯度下降（SGD）学习算子，提出了适用于目标算子的规则条件，并建立了SGD算法的收敛速度上界，同时展示了对于非线性算子学习的有效性及线性近似收敛特性。 |
| [^3] | [A Convex Framework for Confounding Robust Inference.](http://arxiv.org/abs/2309.12450) | 本文提出了一个支撑鲁棒推断的凸框架，通过利用凸规划提供策略价值的精确下界。此外，该方法还可以进行多种扩展，并且具有强理论保证。 |
| [^4] | [The Interpolating Information Criterion for Overparameterized Models.](http://arxiv.org/abs/2307.07785) | 本文提出了一个插值信息准则，用于过参数化模型的模型选择问题。通过建立贝叶斯对偶形式，该准则将先验选择纳入模型评估，并考虑了先验误设、模型的几何和谱特性。该准则在实证和理论行为方面与已知结果一致。 |
| [^5] | [Generative Modeling via Hierarchical Tensor Sketching.](http://arxiv.org/abs/2304.05305) | 本文提出了一种利用分层张量草图来近似高维概率密度的方法，通过随机奇异值分解技术解决线性方程达到此目的，其算法复杂度在高维密度维度上呈线性规模。 |

# 详细

[^1]: 重新构想异常：如果异常是正常的呢？

    Reimagining Anomalies: What If Anomalies Were Normal?

    [https://arxiv.org/abs/2402.14469](https://arxiv.org/abs/2402.14469)

    方法提出了一种新颖的解释方法，生成多个反事实示例以捕获异常的多样概念，为用户提供对触发异常检测器机制的高级语义解释，允许探索“假设情景”。

    

    基于深度学习的方法在图像异常检测方面取得了突破，但其复杂性给理解为何实例被预测为异常带来了相当大的挑战。我们引入了一种新颖的解释方法，为每个异常生成多个反事实示例，捕获异常的多样概念。反事实示例是对异常的修改，被异常检测器视为正常。该方法提供了触发异常检测器机制的高级语义解释，允许用户探索“假设情景”。对不同图像数据集进行的定性和定量分析显示，该方法应用于最先进的异常检测器可以实现对检测器的高质量语义解释。

    arXiv:2402.14469v1 Announce Type: cross  Abstract: Deep learning-based methods have achieved a breakthrough in image anomaly detection, but their complexity introduces a considerable challenge to understanding why an instance is predicted to be anomalous. We introduce a novel explanation method that generates multiple counterfactual examples for each anomaly, capturing diverse concepts of anomalousness. A counterfactual example is a modification of the anomaly that is perceived as normal by the anomaly detector. The method provides a high-level semantic explanation of the mechanism that triggered the anomaly detector, allowing users to explore "what-if scenarios." Qualitative and quantitative analyses across various image datasets show that the method applied to state-of-the-art anomaly detectors can achieve high-quality semantic explanations of detectors.
    
[^2]: 在一般希尔伯特空间中使用随机梯度下降学习算子

    Learning Operators with Stochastic Gradient Descent in General Hilbert Spaces

    [https://arxiv.org/abs/2402.04691](https://arxiv.org/abs/2402.04691)

    本研究在一般希尔伯特空间中使用随机梯度下降（SGD）学习算子，提出了适用于目标算子的规则条件，并建立了SGD算法的收敛速度上界，同时展示了对于非线性算子学习的有效性及线性近似收敛特性。

    

    本研究探讨了利用随机梯度下降（SGD）在一般希尔伯特空间中学习算子的方法。我们提出了针对目标算子的弱和强规则条件，以描述其内在结构和复杂性。在这些条件下，我们建立了SGD算法的收敛速度的上界，并进行了极小值下界分析，进一步说明我们的收敛分析和规则条件定量地刻画了使用SGD算法解决算子学习问题的可行性。值得强调的是，我们的收敛分析对于非线性算子学习仍然有效。我们证明了SGD估计器将收敛于非线性目标算子的最佳线性近似。此外，将我们的分析应用于基于矢量值和实值再生核希尔伯特空间的算子学习问题，产生了新的收敛结果，从而完善了现有文献的结论。

    This study investigates leveraging stochastic gradient descent (SGD) to learn operators between general Hilbert spaces. We propose weak and strong regularity conditions for the target operator to depict its intrinsic structure and complexity. Under these conditions, we establish upper bounds for convergence rates of the SGD algorithm and conduct a minimax lower bound analysis, further illustrating that our convergence analysis and regularity conditions quantitatively characterize the tractability of solving operator learning problems using the SGD algorithm. It is crucial to highlight that our convergence analysis is still valid for nonlinear operator learning. We show that the SGD estimator will converge to the best linear approximation of the nonlinear target operator. Moreover, applying our analysis to operator learning problems based on vector-valued and real-valued reproducing kernel Hilbert spaces yields new convergence results, thereby refining the conclusions of existing litera
    
[^3]: 支撑鲁棒推断的凸框架

    A Convex Framework for Confounding Robust Inference. (arXiv:2309.12450v1 [stat.ML])

    [http://arxiv.org/abs/2309.12450](http://arxiv.org/abs/2309.12450)

    本文提出了一个支撑鲁棒推断的凸框架，通过利用凸规划提供策略价值的精确下界。此外，该方法还可以进行多种扩展，并且具有强理论保证。

    

    我们研究了受未观察到的混淆因素影响的离线上下文强化学习中的策略评估问题。传统的敏感性分析方法常被用来在给定的不确定性集合上估计在最坏混淆情况下的策略价值。然而，现有的工作通常为了可行性而采用一些粗糙的松弛不确定性集合的方法，导致对策略价值的估计过于保守。在本文中，我们提出了一种通用估计器，利用凸规划提供了策略价值的一个较为精确的下界。我们的估计器的广泛适用性使得其能够进行多种扩展，例如基于f-分歧的敏感性分析、基于交叉验证和信息准则的模型选择以及利用上界进行鲁棒策略学习等。此外，我们的估计方法可以通过强对偶性重新表述为经验风险最小化问题，从而利用M技术提供了对所提出估计器的强理论保证。

    We study policy evaluation of offline contextual bandits subject to unobserved confounders. Sensitivity analysis methods are commonly used to estimate the policy value under the worst-case confounding over a given uncertainty set. However, existing work often resorts to some coarse relaxation of the uncertainty set for the sake of tractability, leading to overly conservative estimation of the policy value. In this paper, we propose a general estimator that provides a sharp lower bound of the policy value using convex programming. The generality of our estimator enables various extensions such as sensitivity analysis with f-divergence, model selection with cross validation and information criterion, and robust policy learning with the sharp lower bound. Furthermore, our estimation method can be reformulated as an empirical risk minimization problem thanks to the strong duality, which enables us to provide strong theoretical guarantees of the proposed estimator using techniques of the M-
    
[^4]: 过参数化模型的插值信息准则

    The Interpolating Information Criterion for Overparameterized Models. (arXiv:2307.07785v1 [stat.ML])

    [http://arxiv.org/abs/2307.07785](http://arxiv.org/abs/2307.07785)

    本文提出了一个插值信息准则，用于过参数化模型的模型选择问题。通过建立贝叶斯对偶形式，该准则将先验选择纳入模型评估，并考虑了先验误设、模型的几何和谱特性。该准则在实证和理论行为方面与已知结果一致。

    

    本文考虑了过参数化估计器的模型选择问题，其中模型参数的数量超过数据集的大小。传统的信息准则通常考虑大数据极限，对模型大小进行惩罚。然而，在现代设置中，这些准则不适用，因为过参数化模型往往表现良好。对于任何过参数化模型，我们证明存在一个对偶的欠参数化模型，具有相同的边缘似然性，从而建立了贝叶斯对偶形式。这使得过参数化设置中可以使用更多经典方法，揭示了插值信息准则，一种自然地将先验选择纳入模型选择的模型质量度量。我们的新信息准则考虑了先验误设、模型的几何和谱特性，并且在该区域与已知的经验和理论行为一致。

    The problem of model selection is considered for the setting of interpolating estimators, where the number of model parameters exceeds the size of the dataset. Classical information criteria typically consider the large-data limit, penalizing model size. However, these criteria are not appropriate in modern settings where overparameterized models tend to perform well. For any overparameterized model, we show that there exists a dual underparameterized model that possesses the same marginal likelihood, thus establishing a form of Bayesian duality. This enables more classical methods to be used in the overparameterized setting, revealing the Interpolating Information Criterion, a measure of model quality that naturally incorporates the choice of prior into the model selection. Our new information criterion accounts for prior misspecification, geometric and spectral properties of the model, and is numerically consistent with known empirical and theoretical behavior in this regime.
    
[^5]: 基于分层张量草图的生成建模

    Generative Modeling via Hierarchical Tensor Sketching. (arXiv:2304.05305v1 [math.NA])

    [http://arxiv.org/abs/2304.05305](http://arxiv.org/abs/2304.05305)

    本文提出了一种利用分层张量草图来近似高维概率密度的方法，通过随机奇异值分解技术解决线性方程达到此目的，其算法复杂度在高维密度维度上呈线性规模。

    

    我们提出了一种使用经验分布来近似高维概率密度的分层张量网络方法。该方法利用随机奇异值分解（SVD）技术，并涉及在该张量网络中解线性方程以获得张量核心。该算法的复杂性在高维密度的维度上呈线性规模。通过几个数值实验，对估计误差进行了分析，证明了此方法的有效性。

    We propose a hierarchical tensor-network approach for approximating high-dimensional probability density via empirical distribution. This leverages randomized singular value decomposition (SVD) techniques and involves solving linear equations for tensor cores in this tensor network. The complexity of the resulting algorithm scales linearly in the dimension of the high-dimensional density. An analysis of estimation error demonstrates the effectiveness of this method through several numerical experiments.
    

