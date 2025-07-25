# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Classify and Generate Reciprocally: Simultaneous Positive-Unlabelled Learning and Conditional Generation with Extra Data](https://arxiv.org/abs/2006.07841) | 本论文提出了一种同时利用正数据-无标签学习和有条件生成的训练框架，以及额外无标签数据的方法。通过使用一个对噪声标签具有鲁棒性的分类器噪声不变有条件生成对抗网络来提高PU分类器的性能，并利用PU分类器预测的标签和额外数据来帮助生成。实验结果证明了该方法的有效性。 |
| [^2] | [Adversarially Robust Deep Learning with Optimal-Transport-Regularized Divergences.](http://arxiv.org/abs/2309.03791) | 本论文介绍了一种新的方法ARMOR_D来加强深度学习模型的对抗鲁棒性，该方法基于最优传输正则化差异，通过在分布的邻域上进行最大化期望损失来实现。实验证明，ARMOR_D方法在恶意软件检测和图像识别应用中能够优于现有方法，在对抗攻击下的鲁棒性方面具有较好的效果。 |
| [^3] | [Compressed and distributed least-squares regression: convergence rates with applications to Federated Learning.](http://arxiv.org/abs/2308.01358) | 本文研究了压缩对分布式和联邦学习中随机梯度算法的影响，通过比较不同的无偏压缩操作符的收敛速度，超越了经典的最坏情况分析。针对最小二乘回归，我们提出了一个随机逼近算法，并考虑了随机场的一般假设和噪声协方差的限制，以分析各种随机化机制。 |
| [^4] | [Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift.](http://arxiv.org/abs/2302.10160) | 该论文提出了一种关于核岭回归的协变量转移策略，通过使用伪标签进行模型选择，能够适应不同特征分布下的学习，实现均方误差最小化。 |

# 详细

[^1]: 同时进行正数据-无标签学习和有条件生成，利用额外数据来分类和生成

    Classify and Generate Reciprocally: Simultaneous Positive-Unlabelled Learning and Conditional Generation with Extra Data

    [https://arxiv.org/abs/2006.07841](https://arxiv.org/abs/2006.07841)

    本论文提出了一种同时利用正数据-无标签学习和有条件生成的训练框架，以及额外无标签数据的方法。通过使用一个对噪声标签具有鲁棒性的分类器噪声不变有条件生成对抗网络来提高PU分类器的性能，并利用PU分类器预测的标签和额外数据来帮助生成。实验结果证明了该方法的有效性。

    

    在许多机器学习问题中，标记类别数据的稀缺性是一个普遍存在的瓶颈。虽然存在丰富的无标签数据并提供潜在的解决方案，但利用它们是非常具有挑战性的。本文通过同时利用正数据-无标签（Positive-Unlabeled，PU）分类和有条件生成，以及额外的无标签数据，解决了这个问题。特别地，我们提出了一个新的训练框架，使得在面对额外数据（尤其是分布外的无标签数据）时，同时进行PU分类和有条件生成成为可能，通过探索它们之间的相互作用：1）通过一个对噪声标签具有鲁棒性的新型分类器噪声不变有条件生成对抗网络（Classifier-Noise-Invariant Conditional GAN，CNI-CGAN）来提高PU分类器的性能，2）利用PU分类器预测的标签和额外数据来帮助生成。从理论上，我们证明了CNI-CGAN的最优条件，并在实验中通过广泛的评估来验证了我们的方法。

    The scarcity of class-labeled data is a ubiquitous bottleneck in many machine learning problems. While abundant unlabeled data typically exist and provide a potential solution, it is highly challenging to exploit them. In this paper, we address this problem by leveraging Positive-Unlabeled~(PU) classification and the conditional generation with extra unlabeled data \emph{simultaneously}. In particular, we present a novel training framework to jointly target both PU classification and conditional generation when exposed to extra data, especially out-of-distribution unlabeled data, by exploring the interplay between them: 1) enhancing the performance of PU classifiers with the assistance of a novel Classifier-Noise-Invariant Conditional GAN~(CNI-CGAN) that is robust to noisy labels, 2) leveraging extra data with predicted labels from a PU classifier to help the generation. Theoretically, we prove the optimal condition of CNI-CGAN, and experimentally, we conducted extensive evaluations on
    
[^2]: 使用最优传输正则化差异来提高对抗性鲁棒深度学习

    Adversarially Robust Deep Learning with Optimal-Transport-Regularized Divergences. (arXiv:2309.03791v1 [cs.LG])

    [http://arxiv.org/abs/2309.03791](http://arxiv.org/abs/2309.03791)

    本论文介绍了一种新的方法ARMOR_D来加强深度学习模型的对抗鲁棒性，该方法基于最优传输正则化差异，通过在分布的邻域上进行最大化期望损失来实现。实验证明，ARMOR_D方法在恶意软件检测和图像识别应用中能够优于现有方法，在对抗攻击下的鲁棒性方面具有较好的效果。

    

    我们引入了ARMOR_D方法作为增强深度学习模型对抗性鲁棒性的创新方法。这些方法基于一种新的最优传输正则化差异类，通过信息差异和最优传输成本之间的infimal卷积构建。我们使用这些方法来增强对抗性鲁棒性，通过在分布的邻域上最大化期望损失，这被称为分布鲁棒优化技术。作为构建对抗样本的工具，我们的方法允许样本根据最优传输成本进行传输，并根据信息差异进行重新加权。我们在恶意软件检测和图像识别应用上证明了我们方法的有效性，并发现在增强对抗攻击鲁棒性方面，据我们所知，它优于现有方法。ARMOR_D在FGSM攻击下的robustified准确率达到98.29%，在其他攻击下达到98.18%。

    We introduce the $ARMOR_D$ methods as novel approaches to enhancing the adversarial robustness of deep learning models. These methods are based on a new class of optimal-transport-regularized divergences, constructed via an infimal convolution between an information divergence and an optimal-transport (OT) cost. We use these as tools to enhance adversarial robustness by maximizing the expected loss over a neighborhood of distributions, a technique known as distributionally robust optimization. Viewed as a tool for constructing adversarial samples, our method allows samples to be both transported, according to the OT cost, and re-weighted, according to the information divergence. We demonstrate the effectiveness of our method on malware detection and image recognition applications and find that, to our knowledge, it outperforms existing methods at enhancing the robustness against adversarial attacks. $ARMOR_D$ yields the robustified accuracy of $98.29\%$ against $FGSM$ and $98.18\%$ aga
    
[^3]: 压缩和分布式最小二乘回归：收敛速度及其在联邦学习中的应用

    Compressed and distributed least-squares regression: convergence rates with applications to Federated Learning. (arXiv:2308.01358v1 [cs.LG])

    [http://arxiv.org/abs/2308.01358](http://arxiv.org/abs/2308.01358)

    本文研究了压缩对分布式和联邦学习中随机梯度算法的影响，通过比较不同的无偏压缩操作符的收敛速度，超越了经典的最坏情况分析。针对最小二乘回归，我们提出了一个随机逼近算法，并考虑了随机场的一般假设和噪声协方差的限制，以分析各种随机化机制。

    

    本文研究了在机器学习中广泛应用的分布式和联邦学习中，压缩对随机梯度算法的影响。我们强调了几种无偏压缩操作符之间的收敛速度差异，这些操作符都满足相同的方差条件，从而超越了经典的最坏情况分析。为此，我们专注于最小二乘回归（LSR）的情况，并分析了一个依赖于随机场的最小二乘回归的随机逼近算法。我们对随机场的一般性假设进行了详细分析（特别是期望的Hölder正则性）并对噪声协方差进行了限制，以便分析各种随机化机制，包括压缩。然后，我们将结果扩展到联邦学习的情况下。具体而言，我们强调了对加性噪声的协方差𝖢𝖠𝖭𝖨𝖠对收敛性的影响。

    In this paper, we investigate the impact of compression on stochastic gradient algorithms for machine learning, a technique widely used in distributed and federated learning. We underline differences in terms of convergence rates between several unbiased compression operators, that all satisfy the same condition on their variance, thus going beyond the classical worst-case analysis. To do so, we focus on the case of least-squares regression (LSR) and analyze a general stochastic approximation algorithm for minimizing quadratic functions relying on a random field. We consider weak assumptions on the random field, tailored to the analysis (specifically, expected H\"older regularity), and on the noise covariance, enabling the analysis of various randomizing mechanisms, including compression. We then extend our results to the case of federated learning.  More formally, we highlight the impact on the convergence of the covariance $\mathfrak{C}_{\mathrm{ania}}$ of the additive noise induced 
    
[^4]: 核岭回归下伪标签的协变量转移策略

    Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift. (arXiv:2302.10160v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2302.10160](http://arxiv.org/abs/2302.10160)

    该论文提出了一种关于核岭回归的协变量转移策略，通过使用伪标签进行模型选择，能够适应不同特征分布下的学习，实现均方误差最小化。

    

    我们提出并分析了一种基于协变量转移的核岭回归方法。我们的目标是在目标分布上学习一个均方误差最小的回归函数，基于从目标分布采样的未标记数据和可能具有不同特征分布的已标记数据。我们将已标记数据分成两个子集，并分别进行核岭回归，以获得候选模型集合和一个填充模型。我们使用后者填充缺失的标签，然后相应地选择最佳的候选模型。我们的非渐近性过量风险界表明，在相当一般的情况下，我们的估计器能够适应目标分布以及协变量转移的结构。它能够实现渐近正态误差率直到对数因子的最小极限优化。在模型选择中使用伪标签不会产生主要负面影响。

    We develop and analyze a principled approach to kernel ridge regression under covariate shift. The goal is to learn a regression function with small mean squared error over a target distribution, based on unlabeled data from there and labeled data that may have a different feature distribution. We propose to split the labeled data into two subsets and conduct kernel ridge regression on them separately to obtain a collection of candidate models and an imputation model. We use the latter to fill the missing labels and then select the best candidate model accordingly. Our non-asymptotic excess risk bounds show that in quite general scenarios, our estimator adapts to the structure of the target distribution as well as the covariate shift. It achieves the minimax optimal error rate up to a logarithmic factor. The use of pseudo-labels in model selection does not have major negative impacts.
    

