# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustness Bounds on the Successful Adversarial Examples: Theory and Practice](https://arxiv.org/abs/2403.01896) | 本文提出了一个新的成功对抗样本概率上限的理论界限，取决于扰动范数、核函数以及训练数据集中最接近的不同标签对之间的距离，并且实验证明了该理论结果的有效性。 |
| [^2] | [Mixed-Output Gaussian Process Latent Variable Models](https://arxiv.org/abs/2402.09122) | 本文提出了一种基于高斯过程潜变量模型的贝叶斯非参数方法，可以用于信号分离，并且能够处理包含纯组分信号加权和的情况，适用于光谱学和其他领域的多种应用。 |
| [^3] | [Tailoring Mixup to Data using Kernel Warping functions.](http://arxiv.org/abs/2311.01434) | 本研究提出了一种利用核扭曲函数对Mixup数据进行个性化处理的方法，通过动态改变插值系数的概率分布来实现更频繁和更强烈的混合相似数据点。实验证明这种方法不仅提高了模型性能，还提高了模型的校准性。 |
| [^4] | [Multi-Armed Bandits and Quantum Channel Oracles.](http://arxiv.org/abs/2301.08544) | 本论文研究了量子算法在多臂赌博机问题中的应用，发现在可以查询奖励的随机性以及臂的叠加态时可以实现二次加速，但在只能有限地访问奖励的随机性时，查询复杂度与经典算法相同。 |

# 详细

[^1]: 成功对抗样本的强鲁棒性界限：理论与实践

    Robustness Bounds on the Successful Adversarial Examples: Theory and Practice

    [https://arxiv.org/abs/2403.01896](https://arxiv.org/abs/2403.01896)

    本文提出了一个新的成功对抗样本概率上限的理论界限，取决于扰动范数、核函数以及训练数据集中最接近的不同标签对之间的距离，并且实验证明了该理论结果的有效性。

    

    对抗样本（AE）是一种针对机器学习的攻击方法，通过对数据添加不可感知的扰动来诱使错分。本文基于高斯过程（GP）分类，研究了成功AE的概率上限。我们证明了一个新的上界，取决于AE的扰动范数、GP中使用的核函数以及训练数据集中具有不同标签的最接近对之间的距离。令人惊讶的是，该上限不受样本数据集分布的影响。我们通过使用ImageNet的实验验证了我们的理论结果。此外，我们展示了改变核函数参数会导致成功AE概率上限的变化。

    arXiv:2403.01896v1 Announce Type: new  Abstract: Adversarial example (AE) is an attack method for machine learning, which is crafted by adding imperceptible perturbation to the data inducing misclassification. In the current paper, we investigated the upper bound of the probability of successful AEs based on the Gaussian Process (GP) classification. We proved a new upper bound that depends on AE's perturbation norm, the kernel function used in GP, and the distance of the closest pair with different labels in the training dataset. Surprisingly, the upper bound is determined regardless of the distribution of the sample dataset. We showed that our theoretical result was confirmed through the experiment using ImageNet. In addition, we showed that changing the parameters of the kernel function induces a change of the upper bound of the probability of successful AEs.
    
[^2]: 混合输出高斯过程潜变量模型

    Mixed-Output Gaussian Process Latent Variable Models

    [https://arxiv.org/abs/2402.09122](https://arxiv.org/abs/2402.09122)

    本文提出了一种基于高斯过程潜变量模型的贝叶斯非参数方法，可以用于信号分离，并且能够处理包含纯组分信号加权和的情况，适用于光谱学和其他领域的多种应用。

    

    本文提出了一种贝叶斯非参数的信号分离方法，其中信号可以根据潜变量变化。我们的主要贡献是增加了高斯过程潜变量模型（GPLVMs），以包括每个数据点由已知数量的纯组分信号的加权和组成的情况，并观察多个输入位置。我们的框架允许使用各种关于每个观测权重的先验。这种灵活性使我们能够表示包括用于估计分数组成的总和为一约束和用于分类的二进制权重的用例。我们的贡献对于光谱学尤其相关，因为改变条件可能导致基础纯组分信号在样本之间变化。为了展示对光谱学和其他领域的适用性，我们考虑了几个应用：一个具有不同温度的近红外光谱数据集。

    arXiv:2402.09122v1 Announce Type: cross Abstract: This work develops a Bayesian non-parametric approach to signal separation where the signals may vary according to latent variables. Our key contribution is to augment Gaussian Process Latent Variable Models (GPLVMs) to incorporate the case where each data point comprises the weighted sum of a known number of pure component signals, observed across several input locations. Our framework allows the use of a range of priors for the weights of each observation. This flexibility enables us to represent use cases including sum-to-one constraints for estimating fractional makeup, and binary weights for classification. Our contributions are particularly relevant to spectroscopy, where changing conditions may cause the underlying pure component signals to vary from sample to sample. To demonstrate the applicability to both spectroscopy and other domains, we consider several applications: a near-infrared spectroscopy data set with varying temper
    
[^3]: 通过核扭曲函数定制Mixup数据

    Tailoring Mixup to Data using Kernel Warping functions. (arXiv:2311.01434v1 [cs.LG])

    [http://arxiv.org/abs/2311.01434](http://arxiv.org/abs/2311.01434)

    本研究提出了一种利用核扭曲函数对Mixup数据进行个性化处理的方法，通过动态改变插值系数的概率分布来实现更频繁和更强烈的混合相似数据点。实验证明这种方法不仅提高了模型性能，还提高了模型的校准性。

    

    数据增强是学习高效深度学习模型的重要基础。在所有提出的增强技术中，线性插值训练数据点（也称为Mixup）已被证明在许多应用中非常有效。然而，大多数研究都集中在选择合适的点进行混合，或者应用复杂的非线性插值，而我们则对更相似的点进行更频繁和更强烈的混合感兴趣。为此，我们提出了通过扭曲函数动态改变插值系数的概率分布的方法，取决于要组合的数据点之间的相似性。我们定义了一个高效而灵活的框架来实现这一点，以避免多样性的损失。我们进行了广泛的分类和回归任务实验，结果显示我们提出的方法既提高了模型的性能，又提高了模型的校准性。代码可在https://github.com/ENSTA-U2IS/torch-uncertainty上找到。

    Data augmentation is an essential building block for learning efficient deep learning models. Among all augmentation techniques proposed so far, linear interpolation of training data points, also called mixup, has found to be effective for a large panel of applications. While the majority of works have focused on selecting the right points to mix, or applying complex non-linear interpolation, we are interested in mixing similar points more frequently and strongly than less similar ones. To this end, we propose to dynamically change the underlying distribution of interpolation coefficients through warping functions, depending on the similarity between data points to combine. We define an efficient and flexible framework to do so without losing in diversity. We provide extensive experiments for classification and regression tasks, showing that our proposed method improves both performance and calibration of models. Code available in https://github.com/ENSTA-U2IS/torch-uncertainty
    
[^4]: 多臂赌博机和量子通道预测

    Multi-Armed Bandits and Quantum Channel Oracles. (arXiv:2301.08544v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2301.08544](http://arxiv.org/abs/2301.08544)

    本论文研究了量子算法在多臂赌博机问题中的应用，发现在可以查询奖励的随机性以及臂的叠加态时可以实现二次加速，但在只能有限地访问奖励的随机性时，查询复杂度与经典算法相同。

    

    多臂赌博机是强化学习理论的重要支柱之一。最近，人们开始研究用于多臂赌博机问题的量子算法，并发现当可以在叠加态中查询臂和奖励随机性时，可以实现二次加速（在查询复杂度上）。在这里，我们引入了进一步的赌博机模型，其中我们只能有限地访问奖励的随机性，但我们仍然可以在叠加态中查询臂。我们证明当如此时查询复杂度与经典算法相同。这推广了先前的结果，即当预测器具有正的失效概率时，对于未结构化搜索无法实现加速。

    Multi-armed bandits are one of the theoretical pillars of reinforcement learning. Recently, the investigation of quantum algorithms for multi-armed bandit problems was started, and it was found that a quadratic speed-up (in query complexity) is possible when the arms and the randomness of the rewards of the arms can be queried in superposition. Here we introduce further bandit models where we only have limited access to the randomness of the rewards, but we can still query the arms in superposition. We show that then the query complexity is the same as for classical algorithms. This generalizes the prior result that no speed-up is possible for unstructured search when the oracle has positive failure probability.
    

