# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Modeling Latent Selection with Structural Causal Models.](http://arxiv.org/abs/2401.06925) | 本文介绍了一种在结构因果模型中对潜在选择进行建模的方法，并展示了它如何帮助进行因果推理任务，包括处理选择偏差。 |
| [^2] | [Temporally Disentangled Representation Learning under Unknown Nonstationarity.](http://arxiv.org/abs/2310.18615) | 本研究在非平稳情况下，探索了时间解缠表示学习的马尔可夫假设，并提出了一种无需辅助变量观测的方法来恢复独立的潜在分量。 |
| [^3] | [Towards Enhanced Local Explainability of Random Forests: a Proximity-Based Approach.](http://arxiv.org/abs/2310.12428) | 这项研究提出了一种利用随机森林模型的特征空间中的邻近性来解释模型预测的方法，为模型预测提供了局部的解释性，与现有方法相辅相成。通过实验证明了这种方法在债券定价模型中的有效性。 |
| [^4] | [Conformal prediction for frequency-severity modeling.](http://arxiv.org/abs/2307.13124) | 这个论文提出了一个非参数的模型无关框架，用于建立保险理赔的预测区间，并具有有限样本的统计保证，扩展了split conformal prediction技术到两阶段频率-严重性建模领域，并通过使用随机森林作为严重性模型，利用了袋外机制消除了校准集的需要，并实现了具有自适应宽度的预测区间的生成。 |
| [^5] | [Pattern Recovery in Penalized and Thresholded Estimation and its Geometry.](http://arxiv.org/abs/2307.10158) | 我们提出了一种惩罚化和阈值化估计的模式恢复方法，并定义了模式和恢复条件。对于LASSO，无噪声恢复条件和互不表示条件起到了相同的作用。 |

# 详细

[^1]: 用结构因果模型对潜在选择进行建模

    Modeling Latent Selection with Structural Causal Models. (arXiv:2401.06925v1 [cs.AI])

    [http://arxiv.org/abs/2401.06925](http://arxiv.org/abs/2401.06925)

    本文介绍了一种在结构因果模型中对潜在选择进行建模的方法，并展示了它如何帮助进行因果推理任务，包括处理选择偏差。

    

    选择偏倚在现实世界的数据中是普遍存在的，如果不正确处理可能导致误导性结果。我们引入了对结构因果模型（SCMs）进行条件操作的方法，以从因果的角度对潜在选择进行建模。我们展示了条件操作将具有明确潜在选择机制的SCM转换为没有此类选择机制的SCM，这在一定程度上编码了根据原始SCM选择的亚总体的因果语义。此外，我们还展示了该条件操作保持SCMs的简洁性，无环性和线性性，并与边际化操作相符合。由于这些特性与边际化和干预结合起来，条件操作为在潜在细节已经去除的因果模型中进行因果推理任务提供了一个有价值的工具。我们通过例子演示了如何将因果推断的经典结果推广以包括选择偏倚。

    Selection bias is ubiquitous in real-world data, and can lead to misleading results if not dealt with properly. We introduce a conditioning operation on Structural Causal Models (SCMs) to model latent selection from a causal perspective. We show that the conditioning operation transforms an SCM with the presence of an explicit latent selection mechanism into an SCM without such selection mechanism, which partially encodes the causal semantics of the selected subpopulation according to the original SCM. Furthermore, we show that this conditioning operation preserves the simplicity, acyclicity, and linearity of SCMs, and commutes with marginalization. Thanks to these properties, combined with marginalization and intervention, the conditioning operation offers a valuable tool for conducting causal reasoning tasks within causal models where latent details have been abstracted away. We demonstrate by example how classical results of causal inference can be generalized to include selection b
    
[^2]: 未知非平稳情况下的时间解缠表示学习

    Temporally Disentangled Representation Learning under Unknown Nonstationarity. (arXiv:2310.18615v1 [cs.LG])

    [http://arxiv.org/abs/2310.18615](http://arxiv.org/abs/2310.18615)

    本研究在非平稳情况下，探索了时间解缠表示学习的马尔可夫假设，并提出了一种无需辅助变量观测的方法来恢复独立的潜在分量。

    

    在具有时延潜在因果影响的时序数据的无监督因果表示学习中，通过利用时间结构在稳态情况下已经建立了有关因果相关潜在变量解缠的强可识别性结果。然而，在非平稳情况下，现有研究只部分解决了这个问题，要么利用观测到的辅助变量（如类别标签和/或域索引）作为辅助信息，要么假设简化的潜在因果动力学。这两者限制了方法的适用范围。在本研究中，我们进一步探索了非平稳环境中时间延迟相关过程的马尔可夫假设，并证明了在温和条件下，可以在不观察辅助变量的情况下从非线性混合中恢复独立的潜在分量，但可能存在排列和分量级转换。然后，我们引入了一个有原则的估计框架NCTRL来实现。

    In unsupervised causal representation learning for sequential data with time-delayed latent causal influences, strong identifiability results for the disentanglement of causally-related latent variables have been established in stationary settings by leveraging temporal structure. However, in nonstationary setting, existing work only partially addressed the problem by either utilizing observed auxiliary variables (e.g., class labels and/or domain indexes) as side information or assuming simplified latent causal dynamics. Both constrain the method to a limited range of scenarios. In this study, we further explored the Markov Assumption under time-delayed causally related process in nonstationary setting and showed that under mild conditions, the independent latent components can be recovered from their nonlinear mixture up to a permutation and a component-wise transformation, without the observation of auxiliary variables. We then introduce NCTRL, a principled estimation framework, to r
    
[^3]: 实现随机森林的局部可解释性增强：基于邻近性的方法

    Towards Enhanced Local Explainability of Random Forests: a Proximity-Based Approach. (arXiv:2310.12428v1 [stat.ML])

    [http://arxiv.org/abs/2310.12428](http://arxiv.org/abs/2310.12428)

    这项研究提出了一种利用随机森林模型的特征空间中的邻近性来解释模型预测的方法，为模型预测提供了局部的解释性，与现有方法相辅相成。通过实验证明了这种方法在债券定价模型中的有效性。

    

    我们提出一种新的方法来解释随机森林（RF）模型的样本外性能，利用了任何RF都可以被表述为自适应加权K最近邻（KNN）模型的事实。具体而言，我们利用RF在特征空间中学到的点之间的邻近性，将随机森林的预测重写为训练数据点目标标签的加权平均值。这种线性性质有助于在训练集观测中为任何模型预测生成属性，从而为RF预测提供了局部的解释性，补充了SHAP等已有方法，这些方法则为特征空间维度上的模型预测生成属性。我们在训练于美国公司债券交易数据的债券定价模型中演示了这种方法，并将其与各种现有的模型解释方法进行了比较。

    We initiate a novel approach to explain the out of sample performance of random forest (RF) models by exploiting the fact that any RF can be formulated as an adaptive weighted K nearest-neighbors model. Specifically, we use the proximity between points in the feature space learned by the RF to re-write random forest predictions exactly as a weighted average of the target labels of training data points. This linearity facilitates a local notion of explainability of RF predictions that generates attributions for any model prediction across observations in the training set, and thereby complements established methods like SHAP, which instead generates attributions for a model prediction across dimensions of the feature space. We demonstrate this approach in the context of a bond pricing model trained on US corporate bond trades, and compare our approach to various existing approaches to model explainability.
    
[^4]: 频率-严重性建模的符合性预测

    Conformal prediction for frequency-severity modeling. (arXiv:2307.13124v1 [stat.ME])

    [http://arxiv.org/abs/2307.13124](http://arxiv.org/abs/2307.13124)

    这个论文提出了一个非参数的模型无关框架，用于建立保险理赔的预测区间，并具有有限样本的统计保证，扩展了split conformal prediction技术到两阶段频率-严重性建模领域，并通过使用随机森林作为严重性模型，利用了袋外机制消除了校准集的需要，并实现了具有自适应宽度的预测区间的生成。

    

    我们提出了一个非参数的模型无关框架，用于建立保险理赔的预测区间，并具有有限样本的统计保证，将分割符合性预测技术扩展到两阶段频率-严重性建模领域。通过模拟和真实数据集展示了该框架的有效性。当基础严重性模型是随机森林时，我们扩展了两阶段分割符合性预测过程，展示了如何利用袋外机制消除校准集的需要，并实现具有自适应宽度的预测区间的生成。

    We present a nonparametric model-agnostic framework for building prediction intervals of insurance claims, with finite sample statistical guarantees, extending the technique of split conformal prediction to the domain of two-stage frequency-severity modeling. The effectiveness of the framework is showcased with simulated and real datasets. When the underlying severity model is a random forest, we extend the two-stage split conformal prediction procedure, showing how the out-of-bag mechanism can be leveraged to eliminate the need for a calibration set and to enable the production of prediction intervals with adaptive width.
    
[^5]: 惩罚化和阈值化估计中的模式恢复及其几何

    Pattern Recovery in Penalized and Thresholded Estimation and its Geometry. (arXiv:2307.10158v1 [math.ST])

    [http://arxiv.org/abs/2307.10158](http://arxiv.org/abs/2307.10158)

    我们提出了一种惩罚化和阈值化估计的模式恢复方法，并定义了模式和恢复条件。对于LASSO，无噪声恢复条件和互不表示条件起到了相同的作用。

    

    我们考虑惩罚估计的框架，其中惩罚项由实值的多面体规范给出，其中包括诸如LASSO（以及其许多变体如广义LASSO）、SLOPE、OSCAR、PACS等方法。每个估计器可以揭示未知参数向量的不同结构或“模式”。我们定义了基于次微分的模式的一般概念，并形式化了一种衡量其复杂性的方法。对于模式恢复，我们提供了一个特定模式以正概率被该过程检测到的最小条件，即所谓的可达性条件。利用我们的方法，我们还引入了更强的无噪声恢复条件。对于LASSO，众所周知，互不表示条件是使模式恢复的概率大于1/2所必需的，并且我们展示了无噪声恢复起到了完全相同的作用，从而扩展和统一了互不表示条件。

    We consider the framework of penalized estimation where the penalty term is given by a real-valued polyhedral gauge, which encompasses methods such as LASSO (and many variants thereof such as the generalized LASSO), SLOPE, OSCAR, PACS and others. Each of these estimators can uncover a different structure or ``pattern'' of the unknown parameter vector. We define a general notion of patterns based on subdifferentials and formalize an approach to measure their complexity. For pattern recovery, we provide a minimal condition for a particular pattern to be detected by the procedure with positive probability, the so-called accessibility condition. Using our approach, we also introduce the stronger noiseless recovery condition. For the LASSO, it is well known that the irrepresentability condition is necessary for pattern recovery with probability larger than $1/2$ and we show that the noiseless recovery plays exactly the same role, thereby extending and unifying the irrepresentability conditi
    

