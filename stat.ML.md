# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Convergence Guarantees for RMSProp and Adam in Generalized-smooth Non-convex Optimization with Affine Noise Variance](https://arxiv.org/abs/2404.01436) | 本文提出了对于RMSProp和Adam在非凸优化中的紧致收敛性分析，首次展示了在最宽松的假设下的收敛性结果，并展示了RMSProp和Adam的迭代复杂度分别为$\mathcal O(\epsilon^{-4})$。 |
| [^2] | [Uncertainty in Graph Neural Networks: A Survey](https://arxiv.org/abs/2403.07185) | 本调查旨在全面概述图神经网络中的不确定性，并提出了关于如何识别、量化和利用不确定性来增强模型性能和 GNN 预测可靠性的观点。 |
| [^3] | [Variational Entropy Search for Adjusting Expected Improvement](https://arxiv.org/abs/2402.11345) | 本论文通过变分推断的方法，将期望改进（EI）视为最大值熵搜索（MES）的特殊情况，提出了变分熵搜索（VES）方法和 VES-Gamma 算法，成功调整 EI 并展示其在贝叶斯优化方面的实用性。 |
| [^4] | [Fourier Circuits in Neural Networks: Unlocking the Potential of Large Language Models in Mathematical Reasoning and Modular Arithmetic](https://arxiv.org/abs/2402.09469) | 本研究探索了神经网络和Transformer在数学推理和模运算中的潜力。我们分析了单隐藏层神经网络和单层Transformer在解决复杂代数学习任务中的特征。阐明了边缘最大化原则对单隐藏层神经网络的影响。 |
| [^5] | [How Much is Unseen Depends Chiefly on Information About the Seen](https://arxiv.org/abs/2402.05835) | 该论文发现，在未知种群中属于未在训练数据中出现的类的数据点的比例几乎完全取决于训练数据中出现相同次数的类的数量。论文提出了一个遗传算法，能够根据样本找到一个具有最小均方误差的估计量。 |
| [^6] | [Online estimation of the inverse of the Hessian for stochastic optimization with application to universal stochastic Newton algorithms.](http://arxiv.org/abs/2401.10923) | 该论文提出了一种在线估计Hessian矩阵逆的方法，利用Robbins-Monro过程，能够 drastical reducescomputational complexity,并发展了通用的随机牛顿算法，研究了所提方法的渐进效率。 |
| [^7] | [Plug-and-Play Posterior Sampling under Mismatched Measurement and Prior Models.](http://arxiv.org/abs/2310.03546) | 本研究提出了一种插拔式后验采样算法（PnP-ULA），通过将物理测量模型与深度学习先验相结合，解决了成像逆问题。我们通过理论分析和数值验证，量化了PnP-ULA在不匹配后验分布下的误差界限，结果表明PnP-ULA对于测量模型和去噪器的不匹配非常敏感。 |
| [^8] | [Testing for Overfitting.](http://arxiv.org/abs/2305.05792) | 本文提出了一种能够使用训练数据进行评估模型性能的假设检验方法，可以准确地定义和检测过拟合。 |
| [^9] | [Learning time-scales in two-layers neural networks.](http://arxiv.org/abs/2303.00055) | 本文研究了两层神经网络的学习动态，发现经验风险的下降速率是非单调的。在分布符合单指数模型的高维宽两层神经网络中，我们通过学习率参数化清晰的阶段转换，并提供了对网络学习动态的全面分析。我们还为早期学习时所学模型的简单性提供了理论解释。 |
| [^10] | [When Doubly Robust Methods Meet Machine Learning for Estimating Treatment Effects from Real-World Data: A Comparative Study.](http://arxiv.org/abs/2204.10969) | 本研究比较了多种常用的双重稳健方法，探讨了它们使用治疗模型和结果模型的策略异同，并研究了如何结合机器学习技术以提高其性能。 |

# 详细

[^1]: RMSProp和Adam在具有仿射噪声方差的广义光滑非凸优化中的收敛性保证

    Convergence Guarantees for RMSProp and Adam in Generalized-smooth Non-convex Optimization with Affine Noise Variance

    [https://arxiv.org/abs/2404.01436](https://arxiv.org/abs/2404.01436)

    本文提出了对于RMSProp和Adam在非凸优化中的紧致收敛性分析，首次展示了在最宽松的假设下的收敛性结果，并展示了RMSProp和Adam的迭代复杂度分别为$\mathcal O(\epsilon^{-4})$。

    

    本文在坐标级别广义光滑性和仿射噪声方差的最宽松假设下，为非凸优化中的RMSProp和Adam提供了首个收敛性分析。首先分析了RMSProp，它是一种具有自适应学习率但没有一阶动量的Adam的特例。具体地，为了解决自适应更新、无界梯度估计和Lipschitz常数之间的依赖挑战，我们证明了下降引理中的一阶项收敛，并且其分母由梯度范数的函数上界限制。基于这一结果，我们展示了使用适当的超参数的RMSProp收敛到一个$\epsilon$-稳定点，其迭代复杂度为$\mathcal O(\epsilon^{-4})$。然后，将我们的分析推广到Adam，额外的挑战是由于梯度与一阶动量之间的不匹配。我们提出了一个新的上界限制

    arXiv:2404.01436v1 Announce Type: cross  Abstract: This paper provides the first tight convergence analyses for RMSProp and Adam in non-convex optimization under the most relaxed assumptions of coordinate-wise generalized smoothness and affine noise variance. We first analyze RMSProp, which is a special case of Adam with adaptive learning rates but without first-order momentum. Specifically, to solve the challenges due to dependence among adaptive update, unbounded gradient estimate and Lipschitz constant, we demonstrate that the first-order term in the descent lemma converges and its denominator is upper bounded by a function of gradient norm. Based on this result, we show that RMSProp with proper hyperparameters converges to an $\epsilon$-stationary point with an iteration complexity of $\mathcal O(\epsilon^{-4})$. We then generalize our analysis to Adam, where the additional challenge is due to a mismatch between the gradient and first-order momentum. We develop a new upper bound on
    
[^2]: 图神经网络中的不确定性：一项调查

    Uncertainty in Graph Neural Networks: A Survey

    [https://arxiv.org/abs/2403.07185](https://arxiv.org/abs/2403.07185)

    本调查旨在全面概述图神经网络中的不确定性，并提出了关于如何识别、量化和利用不确定性来增强模型性能和 GNN 预测可靠性的观点。

    

    图神经网络（GNNs）已被广泛应用于各种现实世界的应用中。然而，GNNs的预测不确定性源自数据中的固有随机性和模型训练误差等多种因素，可能导致不稳定和错误的预测。因此，识别、量化和利用不确定性对于增强模型性能和GNN预测的可靠性是至关重要的。本调查旨在从不确定性的角度全面概述GNNs，并强调其在图学习中的整合。我们比较和总结了现有的图不确定性理论和方法，以及相应的下游任务。通过这种方式，我们弥合了理论与实践之间的差距，同时连接不同的GNN社区。此外，我们的工作为这一领域的未来方向提供了宝贵的见解。

    arXiv:2403.07185v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) have been extensively used in various real-world applications. However, the predictive uncertainty of GNNs stemming from diverse sources such as inherent randomness in data and model training errors can lead to unstable and erroneous predictions. Therefore, identifying, quantifying, and utilizing uncertainty are essential to enhance the performance of the model for the downstream tasks as well as the reliability of the GNN predictions. This survey aims to provide a comprehensive overview of the GNNs from the perspective of uncertainty with an emphasis on its integration in graph learning. We compare and summarize existing graph uncertainty theory and methods, alongside the corresponding downstream tasks. Thereby, we bridge the gap between theory and practice, meanwhile connecting different GNN communities. Moreover, our work provides valuable insights into promising directions in this field.
    
[^3]: 变分熵搜索用于调整期望改进

    Variational Entropy Search for Adjusting Expected Improvement

    [https://arxiv.org/abs/2402.11345](https://arxiv.org/abs/2402.11345)

    本论文通过变分推断的方法，将期望改进（EI）视为最大值熵搜索（MES）的特殊情况，提出了变分熵搜索（VES）方法和 VES-Gamma 算法，成功调整 EI 并展示其在贝叶斯优化方面的实用性。

    

    Bayesian optimization 是一种广泛使用的优化黑盒函数的技术，期望改进（EI）是该领域中最常用的获取函数。虽然 EI 通常被视为与其他信息理论获取函数（如熵搜索（ES）和最大值熵搜索（MES））不同，但我们的工作揭示了，通过变分推断（VI）方法，EI 可以被视为 MES 的一种特殊情况。在这一背景下，我们开发了变分熵搜索（VES）方法和 VES-Gamma 算法，通过将信息理论概念的原则整合到 EI 中来调整 EI。VES-Gamma 的有效性在各种测试函数和真实数据集中得到了证明，突出了它在贝叶斯优化场景中的理论和实际用途。

    arXiv:2402.11345v1 Announce Type: cross  Abstract: Bayesian optimization is a widely used technique for optimizing black-box functions, with Expected Improvement (EI) being the most commonly utilized acquisition function in this domain. While EI is often viewed as distinct from other information-theoretic acquisition functions, such as entropy search (ES) and max-value entropy search (MES), our work reveals that EI can be considered a special case of MES when approached through variational inference (VI). In this context, we have developed the Variational Entropy Search (VES) methodology and the VES-Gamma algorithm, which adapts EI by incorporating principles from information-theoretic concepts. The efficacy of VES-Gamma is demonstrated across a variety of test functions and read datasets, highlighting its theoretical and practical utilities in Bayesian optimization scenarios.
    
[^4]: 神经网络中的傅立叶电路：解锁大规模语言模型在数学推理和模运算中的潜力

    Fourier Circuits in Neural Networks: Unlocking the Potential of Large Language Models in Mathematical Reasoning and Modular Arithmetic

    [https://arxiv.org/abs/2402.09469](https://arxiv.org/abs/2402.09469)

    本研究探索了神经网络和Transformer在数学推理和模运算中的潜力。我们分析了单隐藏层神经网络和单层Transformer在解决复杂代数学习任务中的特征。阐明了边缘最大化原则对单隐藏层神经网络的影响。

    

    在机器学习不断发展的背景下，理解神经网络和Transformer所利用的内部表示是一个关键挑战。本研究在近期的研究基础上，对网络采用特定计算策略背后的原因进行了探索。我们的研究聚焦于涉及k个输入的复杂代数学习任务，即模运算的加法。我们对单隐藏层神经网络和单层Transformer在解决这一任务中学到的特征进行了深入的分析。我们理论框架的一个关键是阐明边缘最大化原则对单隐藏层神经网络采用的特征的影响。其中，p表示模数，Dp表示k个输入的模运算数据集，m表示网络输出。

    arXiv:2402.09469v1 Announce Type: new  Abstract: In the evolving landscape of machine learning, a pivotal challenge lies in deciphering the internal representations harnessed by neural networks and Transformers. Building on recent progress toward comprehending how networks execute distinct target functions, our study embarks on an exploration of the underlying reasons behind networks adopting specific computational strategies. We direct our focus to the complex algebraic learning task of modular addition involving $k$ inputs. Our research presents a thorough analytical characterization of the features learned by stylized one-hidden layer neural networks and one-layer Transformers in addressing this task.   A cornerstone of our theoretical framework is the elucidation of how the principle of margin maximization shapes the features adopted by one-hidden layer neural networks. Let $p$ denote the modulus, $D_p$ denote the dataset of modular arithmetic with $k$ inputs and $m$ denote the net
    
[^5]: 不可见数据取决于已知信息的多少

    How Much is Unseen Depends Chiefly on Information About the Seen

    [https://arxiv.org/abs/2402.05835](https://arxiv.org/abs/2402.05835)

    该论文发现，在未知种群中属于未在训练数据中出现的类的数据点的比例几乎完全取决于训练数据中出现相同次数的类的数量。论文提出了一个遗传算法，能够根据样本找到一个具有最小均方误差的估计量。

    

    乍一看可能有些违反直觉：我们发现，在预期中，未知种群中属于在训练数据中没有出现的类的数据点的比例几乎完全由训练数据中出现相同次数的类的数量$f_k$确定。虽然在理论上我们证明了由该估计量引起的偏差在样本大小指数级衰减，但在实践中，高方差阻止我们直接使用它作为样本覆盖估计量。但是，我们对$f_k$之间的依赖关系进行了精确的描述，从而产生了多个不同期望值表示的搜索空间，可以确定地实例化为估计量。因此，我们转向优化，并开发了一种遗传算法，仅根据样本搜索平均均方误差（MSE）最小的估计量。在我们的实验证明，我们的遗传算法发现了具有明显较小方差的估计量。

    It might seem counter-intuitive at first: We find that, in expectation, the proportion of data points in an unknown population-that belong to classes that do not appear in the training data-is almost entirely determined by the number $f_k$ of classes that do appear in the training data the same number of times. While in theory we show that the difference of the induced estimator decays exponentially in the size of the sample, in practice the high variance prevents us from using it directly for an estimator of the sample coverage. However, our precise characterization of the dependency between $f_k$'s induces a large search space of different representations of the expected value, which can be deterministically instantiated as estimators. Hence, we turn to optimization and develop a genetic algorithm that, given only the sample, searches for an estimator with minimal mean-squared error (MSE). In our experiments, our genetic algorithm discovers estimators that have a substantially smalle
    
[^6]: 在具有通用随机牛顿算法应用于随机优化的情况下，关于Hessian矩阵的逆的在线估计

    Online estimation of the inverse of the Hessian for stochastic optimization with application to universal stochastic Newton algorithms. (arXiv:2401.10923v1 [math.OC])

    [http://arxiv.org/abs/2401.10923](http://arxiv.org/abs/2401.10923)

    该论文提出了一种在线估计Hessian矩阵逆的方法，利用Robbins-Monro过程，能够 drastical reducescomputational complexity,并发展了通用的随机牛顿算法，研究了所提方法的渐进效率。

    

    本文针对以期望形式表示的凸函数的次优随机优化问题，提出了一种直接递归估计逆Hessian矩阵的技术，采用了Robbins-Monro过程。该方法能够大幅降低计算复杂性，并且允许开发通用的随机牛顿方法，并研究所提方法的渐近效率。这项工作扩展了在随机优化中二阶算法的应用范围。

    This paper addresses second-order stochastic optimization for estimating the minimizer of a convex function written as an expectation. A direct recursive estimation technique for the inverse Hessian matrix using a Robbins-Monro procedure is introduced. This approach enables to drastically reduces computational complexity. Above all, it allows to develop universal stochastic Newton methods and investigate the asymptotic efficiency of the proposed approach. This work so expands the application scope of secondorder algorithms in stochastic optimization.
    
[^7]: 插拔式后验采样在不匹配测量和先验模型下的应用

    Plug-and-Play Posterior Sampling under Mismatched Measurement and Prior Models. (arXiv:2310.03546v1 [stat.ML])

    [http://arxiv.org/abs/2310.03546](http://arxiv.org/abs/2310.03546)

    本研究提出了一种插拔式后验采样算法（PnP-ULA），通过将物理测量模型与深度学习先验相结合，解决了成像逆问题。我们通过理论分析和数值验证，量化了PnP-ULA在不匹配后验分布下的误差界限，结果表明PnP-ULA对于测量模型和去噪器的不匹配非常敏感。

    

    后验采样已被证明是解决成像逆问题的强大贝叶斯方法。最近发展起来的插拔式未调整朗之万算法（PnP-ULA）通过将物理测量模型与使用图像去噪器指定的深度学习先验相结合，成为一种有前景的蒙特卡洛采样和最小均方误差（MMSE）估计方法。然而，PnP-ULA的采样分布与不匹配的数据保真度和去噪器之间的复杂关系尚未经过理论分析。我们通过提出一种后验-L2拟度量并利用它来量化PnP-ULA在不匹配的后验分布下的显式误差界限来填补这一空白。我们在多个逆问题上对我们的理论进行了数值验证，如从高斯混合模型和图像去模糊中采样。我们的结果表明，PnP-ULA的采样分布对于测量模型和去噪器的不匹配非常敏感，并可以精确地描述其特征。

    Posterior sampling has been shown to be a powerful Bayesian approach for solving imaging inverse problems. The recent plug-and-play unadjusted Langevin algorithm (PnP-ULA) has emerged as a promising method for Monte Carlo sampling and minimum mean squared error (MMSE) estimation by combining physical measurement models with deep-learning priors specified using image denoisers. However, the intricate relationship between the sampling distribution of PnP-ULA and the mismatched data-fidelity and denoiser has not been theoretically analyzed. We address this gap by proposing a posterior-L2 pseudometric and using it to quantify an explicit error bound for PnP-ULA under mismatched posterior distribution. We numerically validate our theory on several inverse problems such as sampling from Gaussian mixture models and image deblurring. Our results suggest that the sensitivity of the sampling distribution of PnP-ULA to a mismatch in the measurement model and the denoiser can be precisely characte
    
[^8]: 过拟合的测试

    Testing for Overfitting. (arXiv:2305.05792v1 [stat.ML])

    [http://arxiv.org/abs/2305.05792](http://arxiv.org/abs/2305.05792)

    本文提出了一种能够使用训练数据进行评估模型性能的假设检验方法，可以准确地定义和检测过拟合。

    

    在机器学习中，高复杂度的模型常见过拟合现象，即模型能够很好地代表数据，但无法推广到基础数据生成过程。解决过拟合的典型方法是在留置集上计算经验风险，一旦风险开始增加，就停止（或标记何时停止）。虽然这种方法输出了良好泛化的模型，但其实现原理主要是启发式的。本文讨论过拟合问题，解释了为什么使用训练数据进行评估时，标准渐近和浓度结果不成立。我们随后提出并阐述了一个假设检验，通过该检验可以对使用训练数据评估模型性能，并量化地定义和检测过拟合。我们依靠确保经验均值应该高概率地近似其真实均值的浓度界限，以得出他们应该相互接近的结论。

    High complexity models are notorious in machine learning for overfitting, a phenomenon in which models well represent data but fail to generalize an underlying data generating process. A typical procedure for circumventing overfitting computes empirical risk on a holdout set and halts once (or flags that/when) it begins to increase. Such practice often helps in outputting a well-generalizing model, but justification for why it works is primarily heuristic.  We discuss the overfitting problem and explain why standard asymptotic and concentration results do not hold for evaluation with training data. We then proceed to introduce and argue for a hypothesis test by means of which both model performance may be evaluated using training data, and overfitting quantitatively defined and detected. We rely on said concentration bounds which guarantee that empirical means should, with high probability, approximate their true mean to conclude that they should approximate each other. We stipulate co
    
[^9]: 两层神经网络中学习时间尺度的研究

    Learning time-scales in two-layers neural networks. (arXiv:2303.00055v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.00055](http://arxiv.org/abs/2303.00055)

    本文研究了两层神经网络的学习动态，发现经验风险的下降速率是非单调的。在分布符合单指数模型的高维宽两层神经网络中，我们通过学习率参数化清晰的阶段转换，并提供了对网络学习动态的全面分析。我们还为早期学习时所学模型的简单性提供了理论解释。

    

    多层神经网络的梯度下降学习具有多个引人注意的特点。尤其是，在大批量数据平均后，经验风险的下降速率是非单调的。几乎没有进展的长周期和快速下降的间隔交替出现。这些连续的学习阶段往往在非常不同的时间尺度上进行。最后，在早期阶段学习的模型通常是“简单的”或“易于学习的”，尽管以难以形式化的方式。本文研究了分布符合单指数模型的高维宽两层神经网络的梯度流动力学，在一系列新的严密结果、非严密数学推导和数值实验的基础上，提供了对网络学习动态的全面分析。我们特别指出，我们通过学习率参数化清晰的阶段转换，并展示了它们与长周期的出现和消失有关。我们还为早期学习时所学模型的简单性提供了理论解释，并证明它们可以用于规范训练过程。

    Gradient-based learning in multi-layer neural networks displays a number of striking features. In particular, the decrease rate of empirical risk is non-monotone even after averaging over large batches. Long plateaus in which one observes barely any progress alternate with intervals of rapid decrease. These successive phases of learning often take place on very different time scales. Finally, models learnt in an early phase are typically `simpler' or `easier to learn' although in a way that is difficult to formalize.  Although theoretical explanations of these phenomena have been put forward, each of them captures at best certain specific regimes. In this paper, we study the gradient flow dynamics of a wide two-layer neural network in high-dimension, when data are distributed according to a single-index model (i.e., the target function depends on a one-dimensional projection of the covariates). Based on a mixture of new rigorous results, non-rigorous mathematical derivations, and numer
    
[^10]: 当双重稳健方法遇到机器学习：用于从实际数据中估计治疗效果的比较研究

    When Doubly Robust Methods Meet Machine Learning for Estimating Treatment Effects from Real-World Data: A Comparative Study. (arXiv:2204.10969v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2204.10969](http://arxiv.org/abs/2204.10969)

    本研究比较了多种常用的双重稳健方法，探讨了它们使用治疗模型和结果模型的策略异同，并研究了如何结合机器学习技术以提高其性能。

    

    观察性队列研究越来越常用于比较效果研究，以评估治疗方法的安全性。最近，各种双重稳健方法已被提出，通过匹配、加权和回归等不同方式，通过组合治疗模型和结果模型来估计平均治疗效应。双重稳健估计器的关键优势在于，它们要求治疗模型或结果模型之一被正确规定，以获得平均治疗效果的一致估计值，从而导致更准确、通常更精确的推断。然而，很少有工作去理解双重稳健估计器由于使用治疗和结果模型的独特策略如何不同以及如何结合机器学习技术以提高其性能。在这里，我们检查了多个受欢迎的双重稳健方法，并使用不同的治疗和结果模型比较它们的性能。

    Observational cohort studies are increasingly being used for comparative effectiveness research to assess the safety of therapeutics. Recently, various doubly robust methods have been proposed for average treatment effect estimation by combining the treatment model and the outcome model via different vehicles, such as matching, weighting, and regression. The key advantage of doubly robust estimators is that they require either the treatment model or the outcome model to be correctly specified to obtain a consistent estimator of average treatment effects, and therefore lead to a more accurate and often more precise inference. However, little work has been done to understand how doubly robust estimators differ due to their unique strategies of using the treatment and outcome models and how machine learning techniques can be combined to boost their performance. Here we examine multiple popular doubly robust methods and compare their performance using different treatment and outcome modeli
    

