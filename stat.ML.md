# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Position Paper: Challenges and Opportunities in Topological Deep Learning](https://arxiv.org/abs/2402.08871) | 拓扑深度学习将拓扑特征引入深度学习模型，可作为图表示学习和几何深度学习的补充，给各种机器学习环境提供了自然选择。本文讨论了拓扑深度学习中的开放问题，并提出了未来的研究机会。 |
| [^2] | [Position Paper: Bayesian Deep Learning in the Age of Large-Scale AI](https://arxiv.org/abs/2402.00809) | 《在大规模人工智能时代的贝叶斯深度学习》这篇立场论文探讨了贝叶斯深度学习在各种不同设置下的优势，并指出了与之相关的挑战和有趣的研究方向。未来的研究重点将放在如何将大规模基础模型与贝叶斯深度学习相结合，以发挥它们的全部潜力。 |
| [^3] | [Towards Enhanced Local Explainability of Random Forests: a Proximity-Based Approach.](http://arxiv.org/abs/2310.12428) | 这项研究提出了一种利用随机森林模型的特征空间中的邻近性来解释模型预测的方法，为模型预测提供了局部的解释性，与现有方法相辅相成。通过实验证明了这种方法在债券定价模型中的有效性。 |
| [^4] | [Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States.](http://arxiv.org/abs/2303.17963) | 本文提出了一种面向未知具有潜在状态系统的学习优化控制方法，并给出了概率性能保证，同时提出了一种验证任意控制律性能的方法。 |

# 详细

[^1]: 位置论文：拓扑深度学习中的挑战与机遇

    Position Paper: Challenges and Opportunities in Topological Deep Learning

    [https://arxiv.org/abs/2402.08871](https://arxiv.org/abs/2402.08871)

    拓扑深度学习将拓扑特征引入深度学习模型，可作为图表示学习和几何深度学习的补充，给各种机器学习环境提供了自然选择。本文讨论了拓扑深度学习中的开放问题，并提出了未来的研究机会。

    

    拓扑深度学习是一个快速发展的领域，它利用拓扑特征来理解和设计深度学习模型。本文认为，通过融入拓扑概念，拓扑深度学习可以补充图表示学习和几何深度学习，并成为各种机器学习环境下的自然选择。为此，本文讨论了拓扑深度学习中的开放问题，涵盖了从实用益处到理论基础的各个方面。针对每个问题，它概述了潜在的解决方案和未来的研究机会。同时，本文也是对科学界的邀请，希望积极参与拓扑深度学习研究，开发这个新兴领域的潜力。

    arXiv:2402.08871v1 Announce Type: new Abstract: Topological deep learning (TDL) is a rapidly evolving field that uses topological features to understand and design deep learning models. This paper posits that TDL may complement graph representation learning and geometric deep learning by incorporating topological concepts, and can thus provide a natural choice for various machine learning settings. To this end, this paper discusses open problems in TDL, ranging from practical benefits to theoretical foundations. For each problem, it outlines potential solutions and future research opportunities. At the same time, this paper serves as an invitation to the scientific community to actively participate in TDL research to unlock the potential of this emerging field.
    
[^2]: 《在大规模人工智能时代的贝叶斯深度学习》的立场论文

    Position Paper: Bayesian Deep Learning in the Age of Large-Scale AI

    [https://arxiv.org/abs/2402.00809](https://arxiv.org/abs/2402.00809)

    《在大规模人工智能时代的贝叶斯深度学习》这篇立场论文探讨了贝叶斯深度学习在各种不同设置下的优势，并指出了与之相关的挑战和有趣的研究方向。未来的研究重点将放在如何将大规模基础模型与贝叶斯深度学习相结合，以发挥它们的全部潜力。

    

    在当前的深度学习研究领域中，人们主要关注在涉及大规模图像和语言数据集的监督任务中实现高预测准确性。然而，更广泛的视角揭示了许多被忽视的度量标准、任务和数据类型，如不确定性、主动和持续学习以及科学数据，这些方面需要关注。贝叶斯深度学习（BDL）是一条有前景的道路，可以在这些不同的设置中提供优势。本文认为BDL可以提升深度学习的能力。它重新审视了BDL的优势、承认了现有的挑战，并重点介绍了一些旨在解决这些障碍的有趣的研究方向。展望未来，讨论集中在可能的方式上，将大规模基础模型与BDL相结合，以充分发挥它们的潜力。

    In the current landscape of deep learning research, there is a predominant emphasis on achieving high predictive accuracy in supervised tasks involving large image and language datasets. However, a broader perspective reveals a multitude of overlooked metrics, tasks, and data types, such as uncertainty, active and continual learning, and scientific data, that demand attention. Bayesian deep learning (BDL) constitutes a promising avenue, offering advantages across these diverse settings. This paper posits that BDL can elevate the capabilities of deep learning. It revisits the strengths of BDL, acknowledges existing challenges, and highlights some exciting research avenues aimed at addressing these obstacles. Looking ahead, the discussion focuses on possible ways to combine large-scale foundation models with BDL to unlock their full potential.
    
[^3]: 实现随机森林的局部可解释性增强：基于邻近性的方法

    Towards Enhanced Local Explainability of Random Forests: a Proximity-Based Approach. (arXiv:2310.12428v1 [stat.ML])

    [http://arxiv.org/abs/2310.12428](http://arxiv.org/abs/2310.12428)

    这项研究提出了一种利用随机森林模型的特征空间中的邻近性来解释模型预测的方法，为模型预测提供了局部的解释性，与现有方法相辅相成。通过实验证明了这种方法在债券定价模型中的有效性。

    

    我们提出一种新的方法来解释随机森林（RF）模型的样本外性能，利用了任何RF都可以被表述为自适应加权K最近邻（KNN）模型的事实。具体而言，我们利用RF在特征空间中学到的点之间的邻近性，将随机森林的预测重写为训练数据点目标标签的加权平均值。这种线性性质有助于在训练集观测中为任何模型预测生成属性，从而为RF预测提供了局部的解释性，补充了SHAP等已有方法，这些方法则为特征空间维度上的模型预测生成属性。我们在训练于美国公司债券交易数据的债券定价模型中演示了这种方法，并将其与各种现有的模型解释方法进行了比较。

    We initiate a novel approach to explain the out of sample performance of random forest (RF) models by exploiting the fact that any RF can be formulated as an adaptive weighted K nearest-neighbors model. Specifically, we use the proximity between points in the feature space learned by the RF to re-write random forest predictions exactly as a weighted average of the target labels of training data points. This linearity facilitates a local notion of explainability of RF predictions that generates attributions for any model prediction across observations in the training set, and thereby complements established methods like SHAP, which instead generates attributions for a model prediction across dimensions of the feature space. We demonstrate this approach in the context of a bond pricing model trained on US corporate bond trades, and compare our approach to various existing approaches to model explainability.
    
[^4]: 面向未知具有潜在状态系统的学习优化控制方法

    Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States. (arXiv:2303.17963v1 [eess.SY])

    [http://arxiv.org/abs/2303.17963](http://arxiv.org/abs/2303.17963)

    本文提出了一种面向未知具有潜在状态系统的学习优化控制方法，并给出了概率性能保证，同时提出了一种验证任意控制律性能的方法。

    

    随着控制工程方法应用于越来越复杂的系统，数据驱动的系统辨识方法成为物理建模的有希望的替代方法。然而，许多这些方法依赖于状态测量的可用性，而复杂系统的状态通常不是直接可测量的。因此，可能需要同时估计动力学和潜在状态，从而更加具有挑战性地设计具有性能保证的控制器。本文提出了一种新方法，用于计算具有潜在状态的未知非线性系统的最优输入轨迹。对结果输入轨迹进行了概率性能保证，并提出了一种验证任意控制律性能的方法。本文在数值模拟中展示了所提出方法的有效性。

    As control engineering methods are applied to increasingly complex systems, data-driven approaches for system identification appear as a promising alternative to physics-based modeling. While many of these approaches rely on the availability of state measurements, the states of a complex system are often not directly measurable. It may then be necessary to jointly estimate the dynamics and a latent state, making it considerably more challenging to design controllers with performance guarantees. This paper proposes a novel method for the computation of an optimal input trajectory for unknown nonlinear systems with latent states. Probabilistic performance guarantees are derived for the resulting input trajectory, and an approach to validate the performance of arbitrary control laws is presented. The effectiveness of the proposed method is demonstrated in a numerical simulation.
    

