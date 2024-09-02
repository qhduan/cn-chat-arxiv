# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Transformers are Expressive, But Are They Expressive Enough for Regression?](https://arxiv.org/abs/2402.15478) | Transformer在逼近连续函数方面存在困难，是否真正是通用函数逼近器仍有待考证 |
| [^2] | [DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model.](http://arxiv.org/abs/2306.01001) | 本文提出了一种扩散模型中的负荷预测不确定性量化方法，采用Seq2Seq网络结构来分离两种类型的不确定性并处理异常情况，不仅着眼于预测条件期望值。 |
| [^3] | [Incorporating Unlabelled Data into Bayesian Neural Networks.](http://arxiv.org/abs/2304.01762) | 该论文提出了一种利用未标记数据学习贝叶斯神经网络（BNNs）的对比框架，通过该框架提出了一种同时具备自监督学习的标签效率和贝叶斯方法中的不确定性估计的实用BNN算法。最后，该方法在半监督和低预算主动学习问题中展现出了数据高效学习的优势。 |
| [^4] | [Wasserstein multivariate auto-regressive models for modeling distributional time series and its application in graph learning.](http://arxiv.org/abs/2207.05442) | 本文提出了一种新的自回归模型，用于分析多元分布时间序列。并且在Wasserstein空间中建模了随机对象，提供了该模型的解的存在性和一致估计器。此方法可以应用于年龄分布和自行车共享网络的观察数据。 |

# 详细

[^1]: Transformer是表现力强大的，但是对于回归任务来说表现力足够吗？

    Transformers are Expressive, But Are They Expressive Enough for Regression?

    [https://arxiv.org/abs/2402.15478](https://arxiv.org/abs/2402.15478)

    Transformer在逼近连续函数方面存在困难，是否真正是通用函数逼近器仍有待考证

    

    Transformer已成为自然语言处理中至关重要的技术，在机器翻译和摘要等应用中表现出色。随着它们的广泛应用，一些研究尝试分析Transformer的表现力。神经网络的表现力指的是它能够逼近的函数类。一个神经网络是完全表现力的，如果它可以充当通用函数逼近器。我们尝试分析Transformer的表现力。与现有观点相反，我们的研究结果表明，Transformer在可靠逼近连续函数方面存在困难，依赖于具有可观区间的分段常数逼近。关键问题是：“Transformer是否真正是通用函数逼近器？”为了解决这个问题，我们进行了彻底的调查，通过实验提供理论见解和支持证据。我们的贡献包括了一个理论分析……（摘要未完整）

    arXiv:2402.15478v1 Announce Type: new  Abstract: Transformers have become pivotal in Natural Language Processing, demonstrating remarkable success in applications like Machine Translation and Summarization. Given their widespread adoption, several works have attempted to analyze the expressivity of Transformers. Expressivity of a neural network is the class of functions it can approximate. A neural network is fully expressive if it can act as a universal function approximator. We attempt to analyze the same for Transformers. Contrary to existing claims, our findings reveal that Transformers struggle to reliably approximate continuous functions, relying on piecewise constant approximations with sizable intervals. The central question emerges as: "\textit{Are Transformers truly Universal Function Approximators}?" To address this, we conduct a thorough investigation, providing theoretical insights and supporting evidence through experiments. Our contributions include a theoretical analysi
    
[^2]: DiffLoad:扩散模型中的负荷预测不确定性量化

    DiffLoad: Uncertainty Quantification in Load Forecasting with Diffusion Model. (arXiv:2306.01001v1 [cs.LG])

    [http://arxiv.org/abs/2306.01001](http://arxiv.org/abs/2306.01001)

    本文提出了一种扩散模型中的负荷预测不确定性量化方法，采用Seq2Seq网络结构来分离两种类型的不确定性并处理异常情况，不仅着眼于预测条件期望值。

    

    电力负荷预测对电力系统的决策制定，如机组投入和能源管理等具有重要意义。近年来，各种基于自监督神经网络的方法已经被应用于电力负荷预测，以提高预测准确性和捕捉不确定性。然而，大多数现有的方法是基于高斯似然方法的，它旨在在给定的协变量下准确估计分布期望值。这种方法很难适应存在分布偏移和异常值的时间数据。在本文中，我们提出了一种基于扩散的Seq2seq结构来估计本体不确定性，并使用鲁棒的加性柯西分布来估计物象不确定性。我们展示了我们的方法能够分离两种类型的不确定性并处理突变情况，而不是准确预测条件期望。

    Electrical load forecasting is of great significance for the decision makings in power systems, such as unit commitment and energy management. In recent years, various self-supervised neural network-based methods have been applied to electrical load forecasting to improve forecasting accuracy and capture uncertainties. However, most current methods are based on Gaussian likelihood methods, which aim to accurately estimate the distribution expectation under a given covariate. This kind of approach is difficult to adapt to situations where temporal data has a distribution shift and outliers. In this paper, we propose a diffusion-based Seq2seq structure to estimate epistemic uncertainty and use the robust additive Cauchy distribution to estimate aleatoric uncertainty. Rather than accurately forecasting conditional expectations, we demonstrate our method's ability in separating two types of uncertainties and dealing with the mutant scenarios.
    
[^3]: 将未标记数据纳入贝叶斯神经网络中

    Incorporating Unlabelled Data into Bayesian Neural Networks. (arXiv:2304.01762v1 [cs.LG])

    [http://arxiv.org/abs/2304.01762](http://arxiv.org/abs/2304.01762)

    该论文提出了一种利用未标记数据学习贝叶斯神经网络（BNNs）的对比框架，通过该框架提出了一种同时具备自监督学习的标签效率和贝叶斯方法中的不确定性估计的实用BNN算法。最后，该方法在半监督和低预算主动学习问题中展现出了数据高效学习的优势。

    

    我们提出了一个对贝叶斯神经网络（BNNs）中先验分布进行学习的对比框架，利用未标记数据来优化。基于该框架，我们提出了一种实用的BNN算法，同时具备自监督学习的标签效率和贝叶斯方法中的根据原则的不确定性估计。最后，我们展示了我们的方法在半监督和低预算主动学习问题中的数据高效学习优势。

    We develop a contrastive framework for learning better prior distributions for Bayesian Neural Networks (BNNs) using unlabelled data. With this framework, we propose a practical BNN algorithm that offers the label-efficiency of self-supervised learning and the principled uncertainty estimates of Bayesian methods. Finally, we demonstrate the advantages of our approach for data-efficient learning in semi-supervised and low-budget active learning problems.
    
[^4]: Wasserstein多元自回归模型用于建模分布时间序列及其在图形学习中的应用

    Wasserstein multivariate auto-regressive models for modeling distributional time series and its application in graph learning. (arXiv:2207.05442v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2207.05442](http://arxiv.org/abs/2207.05442)

    本文提出了一种新的自回归模型，用于分析多元分布时间序列。并且在Wasserstein空间中建模了随机对象，提供了该模型的解的存在性和一致估计器。此方法可以应用于年龄分布和自行车共享网络的观察数据。

    

    我们提出了一种新的自回归模型，用于统计分析多元分布时间序列。感兴趣的数据包括一组在实线有界间隔上支持的概率测度的多个系列，并且被不同时间瞬间所索引。概率测度被建模为Wasserstein空间中的随机对象。我们通过在Lebesgue测度的切空间中建立自回归模型，首先对所有原始测度进行居中处理，以便它们的Fréchet平均值成为Lebesgue测度。利用迭代随机函数系统的理论，提供了这样一个模型的解的存在性、唯一性和平稳性的结果。我们还提出了模型系数的一致估计器。除了对模拟数据的分析，我们还使用两个实际数据集进行了模型演示：一个是不同国家年龄分布的观察数据集，另一个是巴黎自行车共享网络的观察数据集。

    We propose a new auto-regressive model for the statistical analysis of multivariate distributional time series. The data of interest consist of a collection of multiple series of probability measures supported over a bounded interval of the real line, and that are indexed by distinct time instants. The probability measures are modelled as random objects in the Wasserstein space. We establish the auto-regressive model in the tangent space at the Lebesgue measure by first centering all the raw measures so that their Fr\'echet means turn to be the Lebesgue measure. Using the theory of iterated random function systems, results on the existence, uniqueness and stationarity of the solution of such a model are provided. We also propose a consistent estimator for the model coefficient. In addition to the analysis of simulated data, the proposed model is illustrated with two real data sets made of observations from age distribution in different countries and bike sharing network in Paris. Final
    

