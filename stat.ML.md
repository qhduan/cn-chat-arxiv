# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reliable uncertainty with cheaper neural network ensembles: a case study in industrial parts classification](https://arxiv.org/abs/2403.10182) | 研究在工业零部件分类中探讨了利用更便宜的神经网络集成实现可靠的不确定性估计的方法 |
| [^2] | [Prediction-Powered Ranking of Large Language Models](https://arxiv.org/abs/2402.17826) | 该研究提出了一种统计框架，可以衡量人类与模型偏好之间的不确定性，从而进行大型语言模型的预测排名。 |
| [^3] | [Scaling laws for learning with real and surrogate data](https://arxiv.org/abs/2402.04376) | 本研究探讨了将替代数据与真实数据整合以进行训练的方案，发现整合替代数据能够显著降低测试误差，并提出了一个扩展规律来描述混合模型的测试误差，可以用于预测最优加权和收益。 |
| [^4] | [A path-norm toolkit for modern networks: consequences, promises and challenges.](http://arxiv.org/abs/2310.01225) | 本文介绍了适用于现代神经网络的路径范数工具包，可以包括具有偏差、跳跃连接和最大池化的通用DAG ReLU网络。这个工具包恢复或超越了已知的路径范数界限，并挑战了基于路径范数的一些具体承诺。 |
| [^5] | [Towards Size-Independent Generalization Bounds for Deep Operator Nets.](http://arxiv.org/abs/2205.11359) | 本论文研究了深度操作器网络的泛化界限问题，在一类DeepONets中证明了它们的Rademacher复杂度的界限不会随网络宽度扩展而明确变化，并利用这个结果展示了如何选择Huber损失来获得不明确依赖于网络大小的泛化误差界限。 |

# 详细

[^1]: 用更便宜的神经网络集成实现可靠的不确定性：工业零部件分类案例研究

    Reliable uncertainty with cheaper neural network ensembles: a case study in industrial parts classification

    [https://arxiv.org/abs/2403.10182](https://arxiv.org/abs/2403.10182)

    研究在工业零部件分类中探讨了利用更便宜的神经网络集成实现可靠的不确定性估计的方法

    

    在运筹学(OR)中，预测模型经常会遇到数据分布与训练数据分布不同的场景。近年来，神经网络(NNs)在图像分类等领域的出色性能使其在OR中备受关注。然而，当面对OOD数据时，NNs往往会做出自信但不正确的预测。不确定性估计为自信的模型提供了一个解决方案，当输出应(不应)被信任时进行通信。因此，在OR领域中，NNs中的可靠不确定性量化至关重要。由多个独立NNs组成的深度集合已经成为一种有前景的方法，不仅提供强大的预测准确性，还能可靠地估计不确定性。然而，它们的部署由于较大的计算需求而具有挑战性。最近的基础研究提出了更高效的NN集成，即sna

    arXiv:2403.10182v1 Announce Type: new  Abstract: In operations research (OR), predictive models often encounter out-of-distribution (OOD) scenarios where the data distribution differs from the training data distribution. In recent years, neural networks (NNs) are gaining traction in OR for their exceptional performance in fields such as image classification. However, NNs tend to make confident yet incorrect predictions when confronted with OOD data. Uncertainty estimation offers a solution to overconfident models, communicating when the output should (not) be trusted. Hence, reliable uncertainty quantification in NNs is crucial in the OR domain. Deep ensembles, composed of multiple independent NNs, have emerged as a promising approach, offering not only strong predictive accuracy but also reliable uncertainty estimation. However, their deployment is challenging due to substantial computational demands. Recent fundamental research has proposed more efficient NN ensembles, namely the sna
    
[^2]: 大型语言模型的预测排名

    Prediction-Powered Ranking of Large Language Models

    [https://arxiv.org/abs/2402.17826](https://arxiv.org/abs/2402.17826)

    该研究提出了一种统计框架，可以衡量人类与模型偏好之间的不确定性，从而进行大型语言模型的预测排名。

    

    大型语言模型通常根据其与人类偏好的一致性水平进行排名--如果一个模型的输出更受人类偏好，那么它就比其他模型更好。本文提出了一种统计框架来弥合人类与模型偏好之间可能引入的不一致性。

    arXiv:2402.17826v1 Announce Type: cross  Abstract: Large language models are often ranked according to their level of alignment with human preferences -- a model is better than other models if its outputs are more frequently preferred by humans. One of the most popular ways to elicit human preferences utilizes pairwise comparisons between the outputs provided by different models to the same inputs. However, since gathering pairwise comparisons by humans is costly and time-consuming, it has become a very common practice to gather pairwise comparisons by a strong large language model -- a model strongly aligned with human preferences. Surprisingly, practitioners cannot currently measure the uncertainty that any mismatch between human and model preferences may introduce in the constructed rankings. In this work, we develop a statistical framework to bridge this gap. Given a small set of pairwise comparisons by humans and a large set of pairwise comparisons by a model, our framework provid
    
[^3]: 使用真实数据和替代数据进行学习的扩展规律

    Scaling laws for learning with real and surrogate data

    [https://arxiv.org/abs/2402.04376](https://arxiv.org/abs/2402.04376)

    本研究探讨了将替代数据与真实数据整合以进行训练的方案，发现整合替代数据能够显著降低测试误差，并提出了一个扩展规律来描述混合模型的测试误差，可以用于预测最优加权和收益。

    

    收集大量高质量的数据通常被限制在成本昂贵或不切实际的范围内, 这是机器学习中的一个关键瓶颈。相反地, 可以将来自目标分布的小规模数据集与来自公共数据集、不同情况下收集的数据或由生成模型合成的数据相结合, 作为替代数据。我们提出了一种简单的方案来将替代数据整合到训练中, 并使用理论模型和实证研究探索其行为。我们的主要发现是：(i) 整合替代数据可以显著降低原始分布的测试误差；(ii) 为了获得这种效益, 使用最优加权经验风险最小化非常关键；(iii) 在混合使用真实数据和替代数据训练的模型的测试误差可以很好地用一个扩展规律来描述。这可以用来预测最优加权和收益。

    Collecting large quantities of high-quality data is often prohibitively expensive or impractical, and a crucial bottleneck in machine learning. One may instead augment a small set of $n$ data points from the target distribution with data from more accessible sources like public datasets, data collected under different circumstances, or synthesized by generative models. Blurring distinctions, we refer to such data as `surrogate data'.   We define a simple scheme for integrating surrogate data into training and use both theoretical models and empirical studies to explore its behavior. Our main findings are: $(i)$ Integrating surrogate data can significantly reduce the test error on the original distribution; $(ii)$ In order to reap this benefit, it is crucial to use optimally weighted empirical risk minimization; $(iii)$ The test error of models trained on mixtures of real and surrogate data is well described by a scaling law. This can be used to predict the optimal weighting and the gai
    
[^4]: 一种适用于现代网络的路径范数工具包：影响、前景和挑战

    A path-norm toolkit for modern networks: consequences, promises and challenges. (arXiv:2310.01225v1 [stat.ML])

    [http://arxiv.org/abs/2310.01225](http://arxiv.org/abs/2310.01225)

    本文介绍了适用于现代神经网络的路径范数工具包，可以包括具有偏差、跳跃连接和最大池化的通用DAG ReLU网络。这个工具包恢复或超越了已知的路径范数界限，并挑战了基于路径范数的一些具体承诺。

    

    本文介绍了第一个完全能够包括具有偏差、跳跃连接和最大池化的通用DAG ReLU网络的路径范数工具包。这个工具包不仅适用于最广泛的基于路径范数的现代神经网络，还可以恢复或超越已知的此类范数的最尖锐界限。这些扩展的路径范数还享有路径范数的常规优点：计算简便、对网络的对称性具有不变性，在前馈网络上比操作符范数的乘积（另一种常用的复杂度度量）具有更好的锐度。工具包的多功能性和易于实施使我们能够通过数值评估在ImageNet上对ResNet的最尖锐界限来挑战基于路径范数的具体承诺。

    This work introduces the first toolkit around path-norms that is fully able to encompass general DAG ReLU networks with biases, skip connections and max pooling. This toolkit notably allows us to establish generalization bounds for real modern neural networks that are not only the most widely applicable path-norm based ones, but also recover or beat the sharpest known bounds of this type. These extended path-norms further enjoy the usual benefits of path-norms: ease of computation, invariance under the symmetries of the network, and improved sharpness on feedforward networks compared to the product of operators' norms, another complexity measure most commonly used.  The versatility of the toolkit and its ease of implementation allow us to challenge the concrete promises of path-norm-based generalization bounds, by numerically evaluating the sharpest known bounds for ResNets on ImageNet.
    
[^5]: 面向尺度无关的深度操作器网络的泛化界限

    Towards Size-Independent Generalization Bounds for Deep Operator Nets. (arXiv:2205.11359v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2205.11359](http://arxiv.org/abs/2205.11359)

    本论文研究了深度操作器网络的泛化界限问题，在一类DeepONets中证明了它们的Rademacher复杂度的界限不会随网络宽度扩展而明确变化，并利用这个结果展示了如何选择Huber损失来获得不明确依赖于网络大小的泛化误差界限。

    

    在最近的时期，机器学习方法在分析物理系统方面取得了重要进展。在这个主题中特别活跃的领域是"物理信息机器学习"，它专注于使用神经网络来数值求解微分方程。在这项工作中，我们旨在推进在训练DeepONets时测量样本外误差的理论 - 这是解决PDE系统最通用的方法之一。首先，针对一类DeepONets，我们证明了它们的Rademacher复杂度有一个界限，该界限不会明确地随着涉及的网络宽度扩展。其次，我们利用这一结果来展示如何选择Huber损失，使得对于这些DeepONet类，能够获得不明确依赖于网络大小的泛化误差界限。我们指出，我们的理论结果适用于任何目标是由DeepONets求解的PDE。

    In recent times machine learning methods have made significant advances in becoming a useful tool for analyzing physical systems. A particularly active area in this theme has been "physics-informed machine learning" which focuses on using neural nets for numerically solving differential equations. In this work, we aim to advance the theory of measuring out-of-sample error while training DeepONets -- which is among the most versatile ways to solve PDE systems in one-shot.  Firstly, for a class of DeepONets, we prove a bound on their Rademacher complexity which does not explicitly scale with the width of the nets involved. Secondly, we use this to show how the Huber loss can be chosen so that for these DeepONet classes generalization error bounds can be obtained that have no explicit dependence on the size of the nets. We note that our theoretical results apply to any PDE being targeted to be solved by DeepONets.
    

