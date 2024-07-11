# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benign overfitting in leaky ReLU networks with moderate input dimension](https://arxiv.org/abs/2403.06903) | 研究了在泄漏ReLU网络上使用铰链损失进行训练的过程中，信噪比（SNR）条件对于良性和非良性过拟合的影响，并发现高SNR值会导致良性过拟合，低SNR值则会导致有害过拟合。 |
| [^2] | [Lie Group Decompositions for Equivariant Neural Networks.](http://arxiv.org/abs/2310.11366) | 本论文提出了一种基于Lie群结构和几何特性的框架，可以处理非紧致非阿贝尔的Lie群，特别关注于$\text{GL}^{+}(n, \mathbb{R})$和$\text{SL}(n, \mathbb{R})$这两个Lie群。 |
| [^3] | [A Bias-Variance-Covariance Decomposition of Kernel Scores for Generative Models.](http://arxiv.org/abs/2310.05833) | 该论文首次引入了生成模型的核评分的偏差-方差-协方差分解，并提出了相应的量的无偏和一致估计器。通过应用在扩散模型上发现少数群体的模式坍缩是一种与过拟合相反的现象，并证明了方差和预测核熵是图像、音频和语言生成不确定性的可行度量。 |
| [^4] | [MCMC-Correction of Score-Based Diffusion Models for Model Composition.](http://arxiv.org/abs/2307.14012) | 本文提出了一种修正基于得分的扩散模型的方法，使其能够与各种MCMC方法结合，从而实现模型组合和进行更好的采样。 |
| [^5] | [Parameter estimation from an Ornstein-Uhlenbeck process with measurement noise.](http://arxiv.org/abs/2305.13498) | 本文研究了带有测量噪声的Ornstein-Uhlenbeck过程参数估计，提出了算法和方法能够分离热噪声和乘性噪声，并改善数据分析的参数估计精度。 |

# 详细

[^1]: 具有适度输入维度的泄漏ReLU网络中的良性过拟合问题

    Benign overfitting in leaky ReLU networks with moderate input dimension

    [https://arxiv.org/abs/2403.06903](https://arxiv.org/abs/2403.06903)

    研究了在泄漏ReLU网络上使用铰链损失进行训练的过程中，信噪比（SNR）条件对于良性和非良性过拟合的影响，并发现高SNR值会导致良性过拟合，低SNR值则会导致有害过拟合。

    

    良性过拟合问题探讨了一个模型是否能够完美地拟合嘈杂的训练数据，同时又能够很好地泛化。我们研究了在二层泄漏ReLU网络上使用铰链损失进行训练的良性过拟合问题，针对二分类任务。我们考虑输入数据，可以分解为一个共同信号和一个随机噪声成分的总和，这两者相互正交。我们表征了模型参数的信噪比（SNR）条件，导致了良性和非良性（有害）过拟合：特别是，如果SNR很高，则发生良性过拟合，相反，如果SNR很低，则发生有害过拟合。我们将良性和非良性过拟合归因于一个近似边界最大化性质，并展示了在铰链损失下使用梯度下降（GD）训练的泄漏ReLU网络满足这一性质。与以前的工作相比，我们不需要nea

    arXiv:2403.06903v1 Announce Type: new  Abstract: The problem of benign overfitting asks whether it is possible for a model to perfectly fit noisy training data and still generalize well. We study benign overfitting in two-layer leaky ReLU networks trained with the hinge loss on a binary classification task. We consider input data which can be decomposed into the sum of a common signal and a random noise component, which lie on subspaces orthogonal to one another. We characterize conditions on the signal to noise ratio (SNR) of the model parameters giving rise to benign versus non-benign, or harmful, overfitting: in particular, if the SNR is high then benign overfitting occurs, conversely if the SNR is low then harmful overfitting occurs. We attribute both benign and non-benign overfitting to an approximate margin maximization property and show that leaky ReLU networks trained on hinge loss with Gradient Descent (GD) satisfy this property. In contrast to prior work we do not require nea
    
[^2]: Lie Group Decompositions for Equivariant Neural Networks. (arXiv:2310.11366v1 [cs.LG]) (等变神经网络的Lie群分解)

    Lie Group Decompositions for Equivariant Neural Networks. (arXiv:2310.11366v1 [cs.LG])

    [http://arxiv.org/abs/2310.11366](http://arxiv.org/abs/2310.11366)

    本论文提出了一种基于Lie群结构和几何特性的框架，可以处理非紧致非阿贝尔的Lie群，特别关注于$\text{GL}^{+}(n, \mathbb{R})$和$\text{SL}(n, \mathbb{R})$这两个Lie群。

    

    在训练（卷积）神经网络模型时，对几何变换的不变性和等变性被证明是非常有用的归纳偏差，特别是在低数据环境下。大部分研究集中在使用的对称群为紧致或阿贝尔群，或者两者都是。最近的研究拓展了使用的变换类别到Lie群的情况，主要通过使用其Lie代数以及群的指数和对数映射。然而，这样的方法在适用于更大的变换群时受到限制，因为根据所关心的群$G$的不同，指数映射可能不满射。当$G$既不是紧致群也不是阿贝尔群时，还会遇到进一步的限制。我们利用Lie群及其齐次空间的结构和几何特性，提出了一个可以处理这类群的框架，主要关注Lie群$G = \text{GL}^{+}(n, \mathbb{R})$和$G = \text{SL}(n, \mathbb{R}$。

    Invariance and equivariance to geometrical transformations have proven to be very useful inductive biases when training (convolutional) neural network models, especially in the low-data regime. Much work has focused on the case where the symmetry group employed is compact or abelian, or both. Recent work has explored enlarging the class of transformations used to the case of Lie groups, principally through the use of their Lie algebra, as well as the group exponential and logarithm maps. The applicability of such methods to larger transformation groups is limited by the fact that depending on the group of interest $G$, the exponential map may not be surjective. Further limitations are encountered when $G$ is neither compact nor abelian. Using the structure and geometry of Lie groups and their homogeneous spaces, we present a framework by which it is possible to work with such groups primarily focusing on the Lie groups $G = \text{GL}^{+}(n, \mathbb{R})$ and $G = \text{SL}(n, \mathbb{R}
    
[^3]: 生成模型的核评分的偏差-方差-协方差分解

    A Bias-Variance-Covariance Decomposition of Kernel Scores for Generative Models. (arXiv:2310.05833v1 [cs.LG])

    [http://arxiv.org/abs/2310.05833](http://arxiv.org/abs/2310.05833)

    该论文首次引入了生成模型的核评分的偏差-方差-协方差分解，并提出了相应的量的无偏和一致估计器。通过应用在扩散模型上发现少数群体的模式坍缩是一种与过拟合相反的现象，并证明了方差和预测核熵是图像、音频和语言生成不确定性的可行度量。

    

    生成模型在我们日常生活中变得越来越重要，然而，尚不存在一个理论框架来评估它们的泛化行为和不确定性。特别是，不确定性估计问题通常以一种特定任务的临时解决方案来解决。例如，自然语言方法不能应用于图像生成。在本文中，我们首次引入了用于核评分及其相关熵的偏差-方差-协方差分解。我们提出了每个量的无偏和一致估计器，只需要生成样本而不需要底层模型本身。作为应用，我们提供了扩散模型的泛化评估，并发现少数群体的模式坍缩是一种与过拟合相反的现象。此外，我们证明了方差和预测核熵是图像、音频和语言生成不确定性的可行度量。具体来说，我们的方法使得可以通过样本生成评估生成模型的泛化性能，并且发现了不同模型类型下的不确定性现象。

    Generative models, like large language models, are becoming increasingly relevant in our daily lives, yet a theoretical framework to assess their generalization behavior and uncertainty does not exist. Particularly, the problem of uncertainty estimation is commonly solved in an ad-hoc manner and task dependent. For example, natural language approaches cannot be transferred to image generation. In this paper we introduce the first bias-variance-covariance decomposition for kernel scores and their associated entropy. We propose unbiased and consistent estimators for each quantity which only require generated samples but not the underlying model itself. As an application, we offer a generalization evaluation of diffusion models and discover how mode collapse of minority groups is a contrary phenomenon to overfitting. Further, we demonstrate that variance and predictive kernel entropy are viable measures of uncertainty for image, audio, and language generation. Specifically, our approach f
    
[^4]: MCMC-修正基于得分的扩散模型用于模型组合

    MCMC-Correction of Score-Based Diffusion Models for Model Composition. (arXiv:2307.14012v1 [stat.ML])

    [http://arxiv.org/abs/2307.14012](http://arxiv.org/abs/2307.14012)

    本文提出了一种修正基于得分的扩散模型的方法，使其能够与各种MCMC方法结合，从而实现模型组合和进行更好的采样。

    

    扩散模型可以用得分或能量函数来参数化。能量参数化具有更好的理论特性，主要是它可以通过在提议样本中总能量的变化基于Metropolis-Hastings修正步骤来进行扩展采样过程。然而，它似乎产生了稍微较差的性能，更重要的是，由于基于得分的扩散模型的普遍流行，现有的预训练能量参数化模型的可用性受到限制。这种限制削弱了模型组合的目的，即将预训练模型组合起来从新分布中进行采样。然而，我们的提议建议保留得分参数化，而是通过对得分函数进行线积分来计算基于能量的接受概率。这使我们能够重用现有的扩散模型，并将反向过程与各种马尔可夫链蒙特卡罗（MCMC）方法组合起来。

    Diffusion models can be parameterised in terms of either a score or an energy function. The energy parameterisation has better theoretical properties, mainly that it enables an extended sampling procedure with a Metropolis--Hastings correction step, based on the change in total energy in the proposed samples. However, it seems to yield slightly worse performance, and more importantly, due to the widespread popularity of score-based diffusion, there are limited availability of off-the-shelf pre-trained energy-based ones. This limitation undermines the purpose of model composition, which aims to combine pre-trained models to sample from new distributions. Our proposal, however, suggests retaining the score parameterization and instead computing the energy-based acceptance probability through line integration of the score function. This allows us to re-use existing diffusion models and still combine the reverse process with various Markov-Chain Monte Carlo (MCMC) methods. We evaluate our 
    
[^5]: 用于带测量噪声的Ornstein-Uhlenbeck过程参数估计

    Parameter estimation from an Ornstein-Uhlenbeck process with measurement noise. (arXiv:2305.13498v1 [stat.ML])

    [http://arxiv.org/abs/2305.13498](http://arxiv.org/abs/2305.13498)

    本文研究了带有测量噪声的Ornstein-Uhlenbeck过程参数估计，提出了算法和方法能够分离热噪声和乘性噪声，并改善数据分析的参数估计精度。

    

    本文旨在研究噪声对Ornstein-Uhlenbeck过程参数拟合的影响，重点考察了乘性噪声和热噪声对信号分离精度的影响。为了解决这些问题，我们提出了有效区分热噪声和乘性噪声、改善参数估计精度的算法和方法，探讨了乘性和热噪声对实际信号混淆的影响，并提出了解决方法。首先，我们提出了一种可以有效分离热噪声的算法，其性能可与Hamilton Monte Carlo (HMC)相媲美，但速度显著提高。随后，我们分析了乘性噪声，并证明了HMC无法隔离热噪声和乘性噪声。然而，我们展示了，在额外了解热噪声和乘性噪声之间比率的情况下，我们可以精确地估计参数和分离信号。

    This article aims to investigate the impact of noise on parameter fitting for an Ornstein-Uhlenbeck process, focusing on the effects of multiplicative and thermal noise on the accuracy of signal separation. To address these issues, we propose algorithms and methods that can effectively distinguish between thermal and multiplicative noise and improve the precision of parameter estimation for optimal data analysis. Specifically, we explore the impact of both multiplicative and thermal noise on the obfuscation of the actual signal and propose methods to resolve them. Firstly, we present an algorithm that can effectively separate thermal noise with comparable performance to Hamilton Monte Carlo (HMC) but with significantly improved speed. Subsequently, we analyze multiplicative noise and demonstrate that HMC is insufficient for isolating thermal and multiplicative noise. However, we show that, with additional knowledge of the ratio between thermal and multiplicative noise, we can accuratel
    

