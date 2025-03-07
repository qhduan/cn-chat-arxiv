# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Estimation of multiple mean vectors in high dimension](https://arxiv.org/abs/2403.15038) | 通过凸组合的方法估计高维空间中不同概率分布的多维均值，引入了两种权重确定策略：一种通过测试程序识别低方差的相邻均值，提出了封闭形式插补公式；另一种通过最小化二次风险的上置信界确定权重，通过理论分析得出方法对经验均值的二次风险改进，在维度渐近的角度上渐近地接近 Oracle（Minimax）改进。 |
| [^2] | [How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage](https://arxiv.org/abs/2402.10065) | 本论文研究了每个数据点的成员推断攻击，量化了每个数据点的成员泄露，并评估了两种隐私防御措施的效果。 |
| [^3] | [Which Frequencies do CNNs Need? Emergent Bottleneck Structure in Feature Learning](https://arxiv.org/abs/2402.08010) | 本文描述了CNN中卷积瓶颈（CBN）结构的出现，网络在前几层将输入表示转换为在少数频率和通道上受支持的表示，然后通过最后几层映射回输出。CBN秩定义了保留在瓶颈中的频率的数量和类型，并部分证明了参数范数与深度和CBN秩的比例成正比。此外，我们还展示了网络的参数范数依赖于函数的规则性。我们发现任何具有接近最优参数范数的网络都会展示出CBN结构，这解释了下采样的常见实践；我们还验证了CBN结构在下采样下仍然成立。最后，我们使用CBN结构来解释...（摘要完整内容请见正文） |
| [^4] | [PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation](https://arxiv.org/abs/2402.04355) | PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。 |
| [^5] | [Careful with that Scalpel: Improving Gradient Surgery with an EMA](https://arxiv.org/abs/2402.02998) | 通过将训练损失梯度和辅助梯度在训练梯度方向上的正交投影结合起来，使用EMA（指数移动平均）可以改进梯度手术，提高深度学习估计管道的性能。 |
| [^6] | [Canonical normalizing flows for manifold learning.](http://arxiv.org/abs/2310.12743) | 该论文介绍了一种规范化正态流方法，用于流形学习。通过可学习的可逆变换将数据嵌入到高维空间中，从而实现了在流形上计算概率密度并优化网络参数的目标。然而，当前的方法在学习到的流形表示中存在着与流形关联且退化的内在基函数的问题。 |
| [^7] | [Introduction to Online Nonstochastic Control.](http://arxiv.org/abs/2211.09619) | 介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。 |

# 详细

[^1]: 高维情况下多个均值向量的估计

    Estimation of multiple mean vectors in high dimension

    [https://arxiv.org/abs/2403.15038](https://arxiv.org/abs/2403.15038)

    通过凸组合的方法估计高维空间中不同概率分布的多维均值，引入了两种权重确定策略：一种通过测试程序识别低方差的相邻均值，提出了封闭形式插补公式；另一种通过最小化二次风险的上置信界确定权重，通过理论分析得出方法对经验均值的二次风险改进，在维度渐近的角度上渐近地接近 Oracle（Minimax）改进。

    

    我们致力于基于独立样本在一个共同空间中估计来自不同概率分布的多维均值。我们的方法是通过对这些样本导出的经验均值进行凸组合来形成估计量。我们引入了两种策略来找到适当的依赖于数据的凸组合权重：第一种利用测试程序来识别具有低方差的相邻均值，从而产生了一个关于权重的封闭形式插补公式；第二种通过最小化二次风险的上置信区间来确定权重。通过理论分析，我们评估了我们的方法相对于经验均值提供的二次风险改进。我们的分析集中在维度渐近的角度上，显示我们的方法在数据的有效维度增加时渐近地接近于一个 Oracle（Minimax）改进。我们展示了通过提出的方法在均值估计中的应用。

    arXiv:2403.15038v1 Announce Type: cross  Abstract: We endeavour to estimate numerous multi-dimensional means of various probability distributions on a common space based on independent samples. Our approach involves forming estimators through convex combinations of empirical means derived from these samples. We introduce two strategies to find appropriate data-dependent convex combination weights: a first one employing a testing procedure to identify neighbouring means with low variance, which results in a closed-form plug-in formula for the weights, and a second one determining weights via minimization of an upper confidence bound on the quadratic risk.Through theoretical analysis, we evaluate the improvement in quadratic risk offered by our methods compared to the empirical means. Our analysis focuses on a dimensional asymptotics perspective, showing that our methods asymptotically approach an oracle (minimax) improvement as the effective dimension of the data increases.We demonstrat
    
[^2]: 每个数据点泄露您隐私的程度有多大？量化每个数据点的成员泄露

    How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage

    [https://arxiv.org/abs/2402.10065](https://arxiv.org/abs/2402.10065)

    本论文研究了每个数据点的成员推断攻击，量化了每个数据点的成员泄露，并评估了两种隐私防御措施的效果。

    

    我们研究了每个数据点的成员推断攻击（MIAs），其中攻击者旨在推断出一个固定目标数据是否已包含在算法的输入数据集中，从而侵犯隐私。首先，我们定义每个数据点的成员泄露为最优对手辨识它的优势。然后，我们量化了经验均值的每个数据点的成员泄露，并表明它取决于目标数据点和数据生成分布之间的马氏距离。我们进一步评估了两种隐私防御措施的效果，即添加高斯噪声和子采样。我们准确地量化了它们都如何降低每个数据点的成员泄露。我们的分析建立在一个结合了似然比检验的Edgeworth展开和Lindeberg-Feller中心极限定理的新型证明技术上。我们的分析连接了现有的似然比和标量乘积攻击，并对这些攻击进行了论证。

    arXiv:2402.10065v1 Announce Type: new  Abstract: We study the per-datum Membership Inference Attacks (MIAs), where an attacker aims to infer whether a fixed target datum has been included in the input dataset of an algorithm and thus, violates privacy. First, we define the membership leakage of a datum as the advantage of the optimal adversary targeting to identify it. Then, we quantify the per-datum membership leakage for the empirical mean, and show that it depends on the Mahalanobis distance between the target datum and the data-generating distribution. We further assess the effect of two privacy defences, i.e. adding Gaussian noise and sub-sampling. We quantify exactly how both of them decrease the per-datum membership leakage. Our analysis builds on a novel proof technique that combines an Edgeworth expansion of the likelihood ratio test and a Lindeberg-Feller central limit theorem. Our analysis connects the existing likelihood ratio and scalar product attacks, and also justifies 
    
[^3]: CNN需要哪些频率？特征学习中的紧急瓶颈结构的出现

    Which Frequencies do CNNs Need? Emergent Bottleneck Structure in Feature Learning

    [https://arxiv.org/abs/2402.08010](https://arxiv.org/abs/2402.08010)

    本文描述了CNN中卷积瓶颈（CBN）结构的出现，网络在前几层将输入表示转换为在少数频率和通道上受支持的表示，然后通过最后几层映射回输出。CBN秩定义了保留在瓶颈中的频率的数量和类型，并部分证明了参数范数与深度和CBN秩的比例成正比。此外，我们还展示了网络的参数范数依赖于函数的规则性。我们发现任何具有接近最优参数范数的网络都会展示出CBN结构，这解释了下采样的常见实践；我们还验证了CBN结构在下采样下仍然成立。最后，我们使用CBN结构来解释...（摘要完整内容请见正文）

    

    我们描述了CNN中卷积瓶颈（CBN）结构的出现，网络使用其前几层将输入表示转换为仅在几个频率和通道上受支持的表示，然后使用最后几层将其映射回输出。我们定义了CBN秩，描述了保留在瓶颈内的频率的数量和类型，并在一定程度上证明了表示函数$f$所需的参数范数按深度乘以CBN秩$f$的比例缩放。我们还展示了参数范数在下一阶中依赖于$f$的正则性。我们展示了任何具有近乎最优参数范数的网络都会在权重和（在网络对大学习率稳定的假设下）激活中表现出CBN结构，这促使了下采样的常见做法；并且我们验证了CBN结构在下采样下仍然成立。最后，我们使用CBN结构来解释...

    We describe the emergence of a Convolution Bottleneck (CBN) structure in CNNs, where the network uses its first few layers to transform the input representation into a representation that is supported only along a few frequencies and channels, before using the last few layers to map back to the outputs. We define the CBN rank, which describes the number and type of frequencies that are kept inside the bottleneck, and partially prove that the parameter norm required to represent a function $f$ scales as depth times the CBN rank $f$. We also show that the parameter norm depends at next order on the regularity of $f$. We show that any network with almost optimal parameter norm will exhibit a CBN structure in both the weights and - under the assumption that the network is stable under large learning rate - the activations, which motivates the common practice of down-sampling; and we verify that the CBN results still hold with down-sampling. Finally we use the CBN structure to interpret the
    
[^4]: PQMass: 使用概率质量估计的生成模型质量的概率评估

    PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

    [https://arxiv.org/abs/2402.04355](https://arxiv.org/abs/2402.04355)

    PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。

    

    我们提出了一种全面的基于样本的方法来评估生成模型的质量。所提出的方法能够估计两个样本集合来自同一分布的概率，为评估单个生成模型的性能或比较在同一数据集上训练的多个竞争模型提供了一个统计上严格的方法。该比较可以通过将空间划分为非重叠的区域并比较每个区域中的数据样本数量来进行。该方法仅需要生成模型和测试数据的样本。它能够直接处理高维数据，无需降维。显著的是，该方法不依赖于关于真实分布密度的假设，并且不依赖于训练或拟合任何辅助模型。相反，它着重于近似计算密度的积分（概率质量）。

    We propose a comprehensive sample-based method for assessing the quality of generative models. The proposed approach enables the estimation of the probability that two sets of samples are drawn from the same distribution, providing a statistically rigorous method for assessing the performance of a single generative model or the comparison of multiple competing models trained on the same dataset. This comparison can be conducted by dividing the space into non-overlapping regions and comparing the number of data samples in each region. The method only requires samples from the generative model and the test data. It is capable of functioning directly on high-dimensional data, obviating the need for dimensionality reduction. Significantly, the proposed method does not depend on assumptions regarding the density of the true distribution, and it does not rely on training or fitting any auxiliary models. Instead, it focuses on approximating the integral of the density (probability mass) acros
    
[^5]: 小心使用手术刀：使用EMA改进梯度手术

    Careful with that Scalpel: Improving Gradient Surgery with an EMA

    [https://arxiv.org/abs/2402.02998](https://arxiv.org/abs/2402.02998)

    通过将训练损失梯度和辅助梯度在训练梯度方向上的正交投影结合起来，使用EMA（指数移动平均）可以改进梯度手术，提高深度学习估计管道的性能。

    

    在深度学习估计管道中，除了最小化单一的训练损失外，还依赖于辅助目标来量化和鼓励模型的可取属性（例如在另一个数据集上的表现，鲁棒性，与先前的一致性）。虽然将辅助损失与训练损失相加作为正则化的最简单方法，但最近的研究表明，通过混合梯度而不仅仅是简单相加，可以提高性能；这被称为梯度手术。我们将这个问题看作是一个约束最小化问题，其中辅助目标在训练损失的最小化集合中被最小化。为了解决这个双层问题，我们采用了一个参数更新方向，它将训练损失梯度和辅助梯度在训练梯度方向上的正交投影结合起来。在梯度来自小批次的情况下，我们解释了如何使用训练损失梯度的移动平均来维护。

    Beyond minimizing a single training loss, many deep learning estimation pipelines rely on an auxiliary objective to quantify and encourage desirable properties of the model (e.g. performance on another dataset, robustness, agreement with a prior). Although the simplest approach to incorporating an auxiliary loss is to sum it with the training loss as a regularizer, recent works have shown that one can improve performance by blending the gradients beyond a simple sum; this is known as gradient surgery. We cast the problem as a constrained minimization problem where the auxiliary objective is minimized among the set of minimizers of the training loss. To solve this bilevel problem, we follow a parameter update direction that combines the training loss gradient and the orthogonal projection of the auxiliary gradient to the training gradient. In a setting where gradients come from mini-batches, we explain how, using a moving average of the training loss gradients, we can carefully maintain
    
[^6]: 流形学习的规范化正态流

    Canonical normalizing flows for manifold learning. (arXiv:2310.12743v1 [stat.ML])

    [http://arxiv.org/abs/2310.12743](http://arxiv.org/abs/2310.12743)

    该论文介绍了一种规范化正态流方法，用于流形学习。通过可学习的可逆变换将数据嵌入到高维空间中，从而实现了在流形上计算概率密度并优化网络参数的目标。然而，当前的方法在学习到的流形表示中存在着与流形关联且退化的内在基函数的问题。

    

    流形学习流是一类生成建模技术，假设数据具有低维流形描述。通过可学习的可逆变换将这种流形嵌入到数据的高维空间中。因此，一旦通过重构损失正确对齐流形，流形上的概率密度就是可计算的，并且可以使用最大似然来优化网络参数。自然地，数据的低维表示需要是单射映射。最近的方法能够在建模的流形上对密度进行对准，并在嵌入到高维空间时高效计算密度体积变化项。然而，除非单射映射在解析上预定义，否则学习到的流形不一定是数据的有效表示。也就是说，这种模型的潜在维度经常会学习到与流形相关并且退化的内在基函数。

    Manifold learning flows are a class of generative modelling techniques that assume a low-dimensional manifold description of the data. The embedding of such manifold into the high-dimensional space of the data is achieved via learnable invertible transformations. Therefore, once the manifold is properly aligned via a reconstruction loss, the probability density is tractable on the manifold and maximum likelihood can be used optimize the network parameters. Naturally, the lower-dimensional representation of the data requires an injective-mapping. Recent approaches were able to enforce that density aligns with the modelled manifold, while efficiently calculating the density volume-change term when embedding to the higher-dimensional space. However, unless the injective-mapping is analytically predefined, the learned manifold is not necessarily an efficient representation of the data. Namely, the latent dimensions of such models frequently learn an entangled intrinsic basis with degenerat
    
[^7]: 在线非随机控制简介

    Introduction to Online Nonstochastic Control. (arXiv:2211.09619v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.09619](http://arxiv.org/abs/2211.09619)

    介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    

    本文介绍了一种新兴的动态系统控制与可微强化学习范式——在线非随机控制，并应用在线凸优化和凸松弛技术得到了具有可证明保证的新方法，在最佳和鲁棒控制方面取得了显著成果。与其他框架不同，该方法的目标是对抗性攻击，在无法预测扰动模型的情况下，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    This text presents an introduction to an emerging paradigm in control of dynamical systems and differentiable reinforcement learning called online nonstochastic control. The new approach applies techniques from online convex optimization and convex relaxations to obtain new methods with provable guarantees for classical settings in optimal and robust control.  The primary distinction between online nonstochastic control and other frameworks is the objective. In optimal control, robust control, and other control methodologies that assume stochastic noise, the goal is to perform comparably to an offline optimal strategy. In online nonstochastic control, both the cost functions as well as the perturbations from the assumed dynamical model are chosen by an adversary. Thus the optimal policy is not defined a priori. Rather, the target is to attain low regret against the best policy in hindsight from a benchmark class of policies.  This objective suggests the use of the decision making frame
    

