# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Private graphon estimation via sum-of-squares](https://arxiv.org/abs/2403.12213) | 基于和平方法的私有图估计算法首次实现了学习随机块模型和图估计的纯节点差分隐私算法，具有多项式运行时间，与之前最佳的信息论节点私有机制具有相匹配的统计效用保证。 |
| [^2] | [Visualization for Trust in Machine Learning Revisited: The State of the Field in 2023](https://arxiv.org/abs/2403.12005) | 2023年的研究显示，可解释和可信赖的机器学习可视化仍然是一个重要且不断发展的领域，为各种领域提供了趋势、见解和挑战。 |
| [^3] | [High-probability Convergence Bounds for Nonlinear Stochastic Gradient Descent Under Heavy-tailed Noise.](http://arxiv.org/abs/2310.18784) | 本研究探讨了一类非线性随机梯度下降方法的高概率收敛边界。对于具有Lipschitz连续梯度的强凸损失函数，即使噪声是重尾的，结果证明了对失败概率的对数依赖。这些结果适用于剪切、归一化和量化等任何具有有界输出的非线性函数。 |
| [^4] | [Generalized Schr\"odinger Bridge Matching.](http://arxiv.org/abs/2310.02233) | 广义薛定谔桥匹配是一种新的分布匹配算法，通过将任务特定的状态成本考虑在内，推广了现代分布匹配算法，并可用于解决条件随机最优控制问题。 |
| [^5] | [Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood.](http://arxiv.org/abs/2309.05153) | 本文通过协同扩散恢复似然（CDRL）提出了一种方法，用于学习和采样一系列基于能量的模型（EBMs），通过在不断嘈杂化的数据集版本上定义不同噪声水平的EBMs，并与初始化模型配对协同训练。这种方法旨在关闭EBMs和其他生成框架之间的样本质量差距。 |
| [^6] | [Confident Feature Ranking.](http://arxiv.org/abs/2307.15361) | 提出了一种确定性特征排序的方法，该方法通过特征重要性值的两两比较，可以产生排序和同时的置信区间，并且可以选择前k个集合。 |
| [^7] | [Learning time-scales in two-layers neural networks.](http://arxiv.org/abs/2303.00055) | 本文研究了两层神经网络的学习动态，发现经验风险的下降速率是非单调的。在分布符合单指数模型的高维宽两层神经网络中，我们通过学习率参数化清晰的阶段转换，并提供了对网络学习动态的全面分析。我们还为早期学习时所学模型的简单性提供了理论解释。 |

# 详细

[^1]: 通过二次和方法进行私有图估计

    Private graphon estimation via sum-of-squares

    [https://arxiv.org/abs/2403.12213](https://arxiv.org/abs/2403.12213)

    基于和平方法的私有图估计算法首次实现了学习随机块模型和图估计的纯节点差分隐私算法，具有多项式运行时间，与之前最佳的信息论节点私有机制具有相匹配的统计效用保证。

    

    我们开发了用于学习随机块模型和图估计的第一个纯节点差分隐私算法，对于任意常数个块，具有多项式运行时间。统计效用保证与先前最佳的信息论（指数时间）节点私有机制相匹配。该算法基于一个基于指数机制的得分函数，该函数定义为依赖于块数量的二次和松弛。我们结果的关键要素是：(1) 在形式上定义为二次优化在双重随机矩阵的多胞体上的距离的特征化块图定义，(2) 一般的多项式优化的和平方法在任意多胞体上的收敛结果，以及(3) 执行利普希茨扩展的得分函数作为二次和算法范例的一般方法。

    arXiv:2403.12213v1 Announce Type: cross  Abstract: We develop the first pure node-differentially-private algorithms for learning stochastic block models and for graphon estimation with polynomial running time for any constant number of blocks. The statistical utility guarantees match those of the previous best information-theoretic (exponential-time) node-private mechanisms for these problems. The algorithm is based on an exponential mechanism for a score function defined in terms of a sum-of-squares relaxation whose level depends on the number of blocks. The key ingredients of our results are (1) a characterization of the distance between the block graphons in terms of a quadratic optimization over the polytope of doubly stochastic matrices, (2) a general sum-of-squares convergence result for polynomial optimization over arbitrary polytopes, and (3) a general approach to perform Lipschitz extensions of score functions as part of the sum-of-squares algorithmic paradigm.
    
[^2]: 2023年机器学习中信任可视化的最新进展

    Visualization for Trust in Machine Learning Revisited: The State of the Field in 2023

    [https://arxiv.org/abs/2403.12005](https://arxiv.org/abs/2403.12005)

    2023年的研究显示，可解释和可信赖的机器学习可视化仍然是一个重要且不断发展的领域，为各种领域提供了趋势、见解和挑战。

    

    可解释和可信赖的机器学习可视化仍然是信息可视化和视觉分析领域中最重要和深入研究的领域之一，涉及医学、金融和生物信息学等各种应用领域。在我们2020年的最新报告中，包括了200种技术，我们坚持收集同行评审的文章，描述可视化技术，根据先前建立的包含119个类别的分类模式对其进行分类，并在在线调查浏览器中提供了542种技术的结果集。在本调查文章中，我们介绍了截至2023年秋季关于这一数据集的新分析结果，并讨论了在机器学习中使用可视化的趋势、见解和八个开放挑战。我们的结果证实了可视化技术在增加对机器学习模型的信任方面呈快速增长的趋势。

    arXiv:2403.12005v1 Announce Type: cross  Abstract: Visualization for explainable and trustworthy machine learning remains one of the most important and heavily researched fields within information visualization and visual analytics with various application domains, such as medicine, finance, and bioinformatics. After our 2020 state-of-the-art report comprising 200 techniques, we have persistently collected peer-reviewed articles describing visualization techniques, categorized them based on the previously established categorization schema consisting of 119 categories, and provided the resulting collection of 542 techniques in an online survey browser. In this survey article, we present the updated findings of new analyses of this dataset as of fall 2023 and discuss trends, insights, and eight open challenges for using visualizations in machine learning. Our results corroborate the rapidly growing trend of visualization techniques for increasing trust in machine learning models in the p
    
[^3]: 高概率收敛边界下的非线性随机梯度下降在重尾噪声下的研究

    High-probability Convergence Bounds for Nonlinear Stochastic Gradient Descent Under Heavy-tailed Noise. (arXiv:2310.18784v1 [cs.LG])

    [http://arxiv.org/abs/2310.18784](http://arxiv.org/abs/2310.18784)

    本研究探讨了一类非线性随机梯度下降方法的高概率收敛边界。对于具有Lipschitz连续梯度的强凸损失函数，即使噪声是重尾的，结果证明了对失败概率的对数依赖。这些结果适用于剪切、归一化和量化等任何具有有界输出的非线性函数。

    

    最近几个研究工作研究了随机梯度下降（SGD）及其剪切变体的高概率收敛。与普通的SGD相比，剪切SGD在实际中更加稳定，并且在理论上有对数依赖于失败概率的额外好处。然而，其他实际非线性SGD变体（如符号SGD、量化SGD和归一化SGD）的收敛性理解要少得多，这些方法实现了改进的通信效率或加速收敛。在本工作中，我们研究了一类广义非线性SGD方法的高概率收敛边界。对于具有Lipschitz连续梯度的强凸损失函数，即使噪声是重尾的，我们证明了对失败概率的对数依赖。与剪切SGD的结果相比，我们的结果更为一般，适用于具有有界输出的任何非线性函数，如剪切、归一化和量化。

    Several recent works have studied the convergence \textit{in high probability} of stochastic gradient descent (SGD) and its clipped variant. Compared to vanilla SGD, clipped SGD is practically more stable and has the additional theoretical benefit of logarithmic dependence on the failure probability. However, the convergence of other practical nonlinear variants of SGD, e.g., sign SGD, quantized SGD and normalized SGD, that achieve improved communication efficiency or accelerated convergence is much less understood. In this work, we study the convergence bounds \textit{in high probability} of a broad class of nonlinear SGD methods. For strongly convex loss functions with Lipschitz continuous gradients, we prove a logarithmic dependence on the failure probability, even when the noise is heavy-tailed. Strictly more general than the results for clipped SGD, our results hold for any nonlinearity with bounded (component-wise or joint) outputs, such as clipping, normalization, and quantizati
    
[^4]: 广义薛定谔桥匹配

    Generalized Schr\"odinger Bridge Matching. (arXiv:2310.02233v1 [stat.ML])

    [http://arxiv.org/abs/2310.02233](http://arxiv.org/abs/2310.02233)

    广义薛定谔桥匹配是一种新的分布匹配算法，通过将任务特定的状态成本考虑在内，推广了现代分布匹配算法，并可用于解决条件随机最优控制问题。

    

    现代分布匹配算法用于训练扩散或流模型，直接规定了两个边界分布之间的边缘分布的时间演变。在这项工作中，我们考虑了一个广义的分布匹配设置，其中这些边缘分布仅以某些任务特定目标函数的解形式隐含描述。这个问题设置被称为广义薛定谔桥(GSB)，在许多科学领域内和机器学习之外广泛出现。我们提出了广义薛定谔桥匹配(GSBM)，这是一种受最近进展启发的新的匹配算法，将它们推广到动能最小化之外，并考虑到任务特定的状态成本。我们证明这样的泛化可以被建模为求解条件随机最优控制问题，其中可以使用高效的变分近似，并借助路径积分理论进一步去偏差。与解决GSB问题的先前方法相比，

    Modern distribution matching algorithms for training diffusion or flow models directly prescribe the time evolution of the marginal distributions between two boundary distributions. In this work, we consider a generalized distribution matching setup, where these marginals are only implicitly described as a solution to some task-specific objective function. The problem setup, known as the Generalized Schr\"odinger Bridge (GSB), appears prevalently in many scientific areas both within and without machine learning. We propose Generalized Schr\"odinger Bridge Matching (GSBM), a new matching algorithm inspired by recent advances, generalizing them beyond kinetic energy minimization and to account for task-specific state costs. We show that such a generalization can be cast as solving conditional stochastic optimal control, for which efficient variational approximations can be used, and further debiased with the aid of path integral theory. Compared to prior methods for solving GSB problems,
    
[^5]: 通过协同扩散恢复似然学习基于能量的模型

    Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood. (arXiv:2309.05153v1 [stat.ML])

    [http://arxiv.org/abs/2309.05153](http://arxiv.org/abs/2309.05153)

    本文通过协同扩散恢复似然（CDRL）提出了一种方法，用于学习和采样一系列基于能量的模型（EBMs），通过在不断嘈杂化的数据集版本上定义不同噪声水平的EBMs，并与初始化模型配对协同训练。这种方法旨在关闭EBMs和其他生成框架之间的样本质量差距。

    

    在高维数据上使用最大似然估计训练能量基准模型（EBMs）可能具有挑战性且耗时较长。因此，EBMs和其他生成框架（如GANs和扩散模型）之间存在明显的样本质量差距。为了弥补这一差距，受最近通过最大化扩散恢复似然（DRL）来学习EBMs的努力的启发，我们提出了协同扩散恢复似然（CDRL），一种有效的方法来可行地学习和从一系列EBMs中进行采样，这些EBMs定义在越来越嘈杂的数据集版本上，并与每个EBM的初始化模型配对。在每个噪声水平上，初始化模型学习在EBM的采样过程中分摊，而两个模型在协同训练框架内共同估计。初始化模型生成的样本作为起始点，经过EBM的几个采样步骤进行改进。通过改进后的样本，通过最大化恢复似然来优化EBM。

    Training energy-based models (EBMs) with maximum likelihood estimation on high-dimensional data can be both challenging and time-consuming. As a result, there a noticeable gap in sample quality between EBMs and other generative frameworks like GANs and diffusion models. To close this gap, inspired by the recent efforts of learning EBMs by maximimizing diffusion recovery likelihood (DRL), we propose cooperative diffusion recovery likelihood (CDRL), an effective approach to tractably learn and sample from a series of EBMs defined on increasingly noisy versons of a dataset, paired with an initializer model for each EBM. At each noise level, the initializer model learns to amortize the sampling process of the EBM, and the two models are jointly estimated within a cooperative training framework. Samples from the initializer serve as starting points that are refined by a few sampling steps from the EBM. With the refined samples, the EBM is optimized by maximizing recovery likelihood, while t
    
[^6]: 确定性特征排序

    Confident Feature Ranking. (arXiv:2307.15361v1 [stat.ML])

    [http://arxiv.org/abs/2307.15361](http://arxiv.org/abs/2307.15361)

    提出了一种确定性特征排序的方法，该方法通过特征重要性值的两两比较，可以产生排序和同时的置信区间，并且可以选择前k个集合。

    

    特征重要性的解释通常依赖于特征的相对顺序而不是数值本身，也就是排序。然而，由于计算重要性值时使用的样本量较小，排序可能不稳定。我们提出了一种事后重要性方法，可以产生一种排序和同时的置信区间。基于特征重要性值的两两比较，我们的方法可以保证高概率包含“真实”（无限样本）排序，并允许选择前k个集合。

    Interpretation of feature importance values often relies on the relative order of the features rather than on the value itself, referred to as ranking. However, the order may be unstable due to the small sample sizes used in calculating the importance values. We propose that post-hoc importance methods produce a ranking and simultaneous confident intervals for the rankings. Based on pairwise comparisons of the feature importance values, our method is guaranteed to include the ``true'' (infinite sample) ranking with high probability and allows for selecting top-k sets.
    
[^7]: 两层神经网络中学习时间尺度的研究

    Learning time-scales in two-layers neural networks. (arXiv:2303.00055v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.00055](http://arxiv.org/abs/2303.00055)

    本文研究了两层神经网络的学习动态，发现经验风险的下降速率是非单调的。在分布符合单指数模型的高维宽两层神经网络中，我们通过学习率参数化清晰的阶段转换，并提供了对网络学习动态的全面分析。我们还为早期学习时所学模型的简单性提供了理论解释。

    

    多层神经网络的梯度下降学习具有多个引人注意的特点。尤其是，在大批量数据平均后，经验风险的下降速率是非单调的。几乎没有进展的长周期和快速下降的间隔交替出现。这些连续的学习阶段往往在非常不同的时间尺度上进行。最后，在早期阶段学习的模型通常是“简单的”或“易于学习的”，尽管以难以形式化的方式。本文研究了分布符合单指数模型的高维宽两层神经网络的梯度流动力学，在一系列新的严密结果、非严密数学推导和数值实验的基础上，提供了对网络学习动态的全面分析。我们特别指出，我们通过学习率参数化清晰的阶段转换，并展示了它们与长周期的出现和消失有关。我们还为早期学习时所学模型的简单性提供了理论解释，并证明它们可以用于规范训练过程。

    Gradient-based learning in multi-layer neural networks displays a number of striking features. In particular, the decrease rate of empirical risk is non-monotone even after averaging over large batches. Long plateaus in which one observes barely any progress alternate with intervals of rapid decrease. These successive phases of learning often take place on very different time scales. Finally, models learnt in an early phase are typically `simpler' or `easier to learn' although in a way that is difficult to formalize.  Although theoretical explanations of these phenomena have been put forward, each of them captures at best certain specific regimes. In this paper, we study the gradient flow dynamics of a wide two-layer neural network in high-dimension, when data are distributed according to a single-index model (i.e., the target function depends on a one-dimensional projection of the covariates). Based on a mixture of new rigorous results, non-rigorous mathematical derivations, and numer
    

