# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Does Differentially Private Synthetic Data Lead to Synthetic Discoveries?](https://arxiv.org/abs/2403.13612) | 评估差分隐私合成生物医学数据上的Mann-Whitney U检验，以确定在隐私保护合成数据上执行的统计假设检验是否可能导致测试有效性的丧失或功率下降。 |
| [^2] | [Concept-based explainability for an EEG transformer model.](http://arxiv.org/abs/2307.12745) | 本研究尝试将基于概念激活向量（CAVs）的方法应用于脑电图（EEG）数据的解释性，通过定义解释性概念和选择相关数据集，以实现对大规模转换器模型中深度学习模型的理解。 |
| [^3] | [Leveraging Task Structures for Improved Identifiability in Neural Network Representations.](http://arxiv.org/abs/2306.14861) | 本文提出一种基于任务结构的可识别性理论，拓展了先前仅限于单任务分类的工作。任务分布的存在定义了一个潜在变量的条件先验，将可识别性的等价类降低到排列和缩放。在假设任务之间存在因果关系时，该方法可以实现简单的最大边际似然优化，并在因果表示学习方面具有下游应用的可行性。 |
| [^4] | [DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model.](http://arxiv.org/abs/2306.01001) | 本文提出了一种扩散模型中的负荷预测不确定性量化方法，采用Seq2Seq网络结构来分离两种类型的不确定性并处理异常情况，不仅着眼于预测条件期望值。 |
| [^5] | [A Geometric Perspective on Diffusion Models.](http://arxiv.org/abs/2305.19947) | 本文研究了扩散模型的几何结构，发现通过一个明确的准线性采样轨迹和另一个隐式的去噪轨迹平滑连接了数据分布和噪声分布，建立了基于ODE的最优采样和经典的均值漂移算法之间的理论关系。 |

# 详细

[^1]: 差分隐私合成数据能导致合成发现吗？

    Does Differentially Private Synthetic Data Lead to Synthetic Discoveries?

    [https://arxiv.org/abs/2403.13612](https://arxiv.org/abs/2403.13612)

    评估差分隐私合成生物医学数据上的Mann-Whitney U检验，以确定在隐私保护合成数据上执行的统计假设检验是否可能导致测试有效性的丧失或功率下降。

    

    合成数据已被提出作为共享敏感生物医学数据的匿名化解决方案。理想情况下，合成数据应保留原始数据的结构和统计特性，同时保护个体主体的隐私。差分隐私（DP）目前被认为是平衡这种权衡的最佳方法。本研究的目的是评估在差分隐私生物医学数据上进行的Mann-Whitney U检验在I型和II型错误方面，以确定在隐私保护合成数据上执行的统计假设检验是否可能导致测试有效性的丧失或功率下降。

    arXiv:2403.13612v1 Announce Type: new  Abstract: Background: Synthetic data has been proposed as a solution for sharing anonymized versions of sensitive biomedical datasets. Ideally, synthetic data should preserve the structure and statistical properties of the original data, while protecting the privacy of the individual subjects. Differential privacy (DP) is currently considered the gold standard approach for balancing this trade-off.   Objectives: The aim of this study is to evaluate the Mann-Whitney U test on DP-synthetic biomedical data in terms of Type I and Type II errors, in order to establish whether statistical hypothesis testing performed on privacy preserving synthetic data is likely to lead to loss of test's validity or decreased power.   Methods: We evaluate the Mann-Whitney U test on DP-synthetic data generated from real-world data, including a prostate cancer dataset (n=500) and a cardiovascular dataset (n=70 000), as well as on data drawn from two Gaussian distribution
    
[^2]: EEG转换器模型的基于概念的可解释性

    Concept-based explainability for an EEG transformer model. (arXiv:2307.12745v1 [cs.LG])

    [http://arxiv.org/abs/2307.12745](http://arxiv.org/abs/2307.12745)

    本研究尝试将基于概念激活向量（CAVs）的方法应用于脑电图（EEG）数据的解释性，通过定义解释性概念和选择相关数据集，以实现对大规模转换器模型中深度学习模型的理解。

    

    深度学习模型由于其规模、结构以及训练过程中的内在随机性而变得复杂。选择数据集和归纳偏见也增加了额外的复杂性。为了解释这些挑战，Kim等人（2018）引入了概念激活向量（CAVs），旨在从人类对齐的概念角度理解深度模型的内部状态。这些概念对应于潜在空间中的方向，使用线性判别法进行识别。尽管该方法首先应用于图像分类，但后来被适应到包括自然语言处理在内的其他领域。在本研究中，我们尝试将该方法应用于Kostas等人的BENDR（2021）的脑电图（EEG）数据，以实现可解释性。这项努力的关键部分包括定义解释性概念和选择相关数据集以将概念与潜在空间相对应。我们的重点是EEG概念形成的两个机制。

    Deep learning models are complex due to their size, structure, and inherent randomness in training procedures. Additional complexity arises from the selection of datasets and inductive biases. Addressing these challenges for explainability, Kim et al. (2018) introduced Concept Activation Vectors (CAVs), which aim to understand deep models' internal states in terms of human-aligned concepts. These concepts correspond to directions in latent space, identified using linear discriminants. Although this method was first applied to image classification, it was later adapted to other domains, including natural language processing. In this work, we attempt to apply the method to electroencephalogram (EEG) data for explainability in Kostas et al.'s BENDR (2021), a large-scale transformer model. A crucial part of this endeavor involves defining the explanatory concepts and selecting relevant datasets to ground concepts in the latent space. Our focus is on two mechanisms for EEG concept formation
    
[^3]: 利用任务结构提高神经网络表示的可识别性

    Leveraging Task Structures for Improved Identifiability in Neural Network Representations. (arXiv:2306.14861v1 [stat.ML])

    [http://arxiv.org/abs/2306.14861](http://arxiv.org/abs/2306.14861)

    本文提出一种基于任务结构的可识别性理论，拓展了先前仅限于单任务分类的工作。任务分布的存在定义了一个潜在变量的条件先验，将可识别性的等价类降低到排列和缩放。在假设任务之间存在因果关系时，该方法可以实现简单的最大边际似然优化，并在因果表示学习方面具有下游应用的可行性。

    

    本文扩展了监督学习中可辨别性的理论，考虑了在拥有任务分布的情况下的后果。在这种情况下，我们展示了即使在回归的情况下也可以实现可识别性，扩展了先前仅限于单任务分类情况的工作。此外，我们展示了任务分布的存在定义了一个潜在变量的条件先验，将可识别性的等价类降低到排列和缩放，这是一个更强大和更有用的结果。当我们进一步假设这些任务之间存在因果关系时，我们的方法可以实现简单的最大边际似然优化，并在因果表示学习方面具有下游应用的可行性。在经验上，我们验证了我们的模型在恢复合成和现实世界数据的规范表示方面优于更一般的无监督模型。

    This work extends the theory of identifiability in supervised learning by considering the consequences of having access to a distribution of tasks. In such cases, we show that identifiability is achievable even in the case of regression, extending prior work restricted to the single-task classification case. Furthermore, we show that the existence of a task distribution which defines a conditional prior over latent variables reduces the equivalence class for identifiability to permutations and scaling, a much stronger and more useful result. When we further assume a causal structure over these tasks, our approach enables simple maximum marginal likelihood optimization together with downstream applicability to causal representation learning. Empirically, we validate that our model outperforms more general unsupervised models in recovering canonical representations for synthetic and real-world data.
    
[^4]: DiffLoad:扩散模型中的负荷预测不确定性量化

    DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model. (arXiv:2306.01001v1 [cs.LG])

    [http://arxiv.org/abs/2306.01001](http://arxiv.org/abs/2306.01001)

    本文提出了一种扩散模型中的负荷预测不确定性量化方法，采用Seq2Seq网络结构来分离两种类型的不确定性并处理异常情况，不仅着眼于预测条件期望值。

    

    电力负荷预测对电力系统的决策制定，如机组投入和能源管理等具有重要意义。近年来，各种基于自监督神经网络的方法已经被应用于电力负荷预测，以提高预测准确性和捕捉不确定性。然而，大多数现有的方法是基于高斯似然方法的，它旨在在给定的协变量下准确估计分布期望值。这种方法很难适应存在分布偏移和异常值的时间数据。在本文中，我们提出了一种基于扩散的Seq2seq结构来估计本体不确定性，并使用鲁棒的加性柯西分布来估计物象不确定性。我们展示了我们的方法能够分离两种类型的不确定性并处理突变情况，而不是准确预测条件期望。

    Electrical load forecasting is of great significance for the decision makings in power systems, such as unit commitment and energy management. In recent years, various self-supervised neural network-based methods have been applied to electrical load forecasting to improve forecasting accuracy and capture uncertainties. However, most current methods are based on Gaussian likelihood methods, which aim to accurately estimate the distribution expectation under a given covariate. This kind of approach is difficult to adapt to situations where temporal data has a distribution shift and outliers. In this paper, we propose a diffusion-based Seq2seq structure to estimate epistemic uncertainty and use the robust additive Cauchy distribution to estimate aleatoric uncertainty. Rather than accurately forecasting conditional expectations, we demonstrate our method's ability in separating two types of uncertainties and dealing with the mutant scenarios.
    
[^5]: 扩散模型的几何视角

    A Geometric Perspective on Diffusion Models. (arXiv:2305.19947v1 [cs.CV])

    [http://arxiv.org/abs/2305.19947](http://arxiv.org/abs/2305.19947)

    本文研究了扩散模型的几何结构，发现通过一个明确的准线性采样轨迹和另一个隐式的去噪轨迹平滑连接了数据分布和噪声分布，建立了基于ODE的最优采样和经典的均值漂移算法之间的理论关系。

    

    近年来，针对扩散模型的高效训练和快速采样方法取得了显著进展。最近的一个重要进展是使用随机微分方程（SDE）来描述数据扰动和生成建模，以实现统一的数学框架。本文揭示了扩散模型的几个有趣的几何结构，并为其采样动力学提供了简单而强大的解释。通过仔细检查一种流行的方差爆炸SDE及其保持边际的普通微分方程（ODE）用于采样，我们发现数据分布和噪声分布通过一个明确的准线性采样轨迹和另一个隐式的去噪轨迹平滑连接，即使在视觉质量方面也收敛更快。我们还建立起基于ODE的最优采样和经典的均值漂移（寻找模式）算法之间的理论关系。

    Recent years have witnessed significant progress in developing efficient training and fast sampling approaches for diffusion models. A recent remarkable advancement is the use of stochastic differential equations (SDEs) to describe data perturbation and generative modeling in a unified mathematical framework. In this paper, we reveal several intriguing geometric structures of diffusion models and contribute a simple yet powerful interpretation to their sampling dynamics. Through carefully inspecting a popular variance-exploding SDE and its marginal-preserving ordinary differential equation (ODE) for sampling, we discover that the data distribution and the noise distribution are smoothly connected with an explicit, quasi-linear sampling trajectory, and another implicit denoising trajectory, which even converges faster in terms of visual quality. We also establish a theoretical relationship between the optimal ODE-based sampling and the classic mean-shift (mode-seeking) algorithm, with w
    

