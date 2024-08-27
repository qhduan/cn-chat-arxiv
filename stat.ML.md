# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit](https://arxiv.org/abs/2402.06388) | 该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。 |
| [^2] | [Asymptotic Dynamics of Alternating Minimization for Non-Convex Optimization](https://arxiv.org/abs/2402.04751) | 本文研究了交替极小化在非凸优化中的渐近动力学特性，通过复制方法追踪演化过程，发现其可用二维离散随机过程有效描述，存在记忆依赖性。该理论框架适用于分析各种迭代算法。 |
| [^3] | [When accurate prediction models yield harmful self-fulfilling prophecies](https://arxiv.org/abs/2312.01210) | 本研究通过调查预测模型的部署对决策产生有害影响的情况，发现这些模型可能会成为有害的自我实现预言。这些模型不会因为对某些患者造成更糟糕的结果而使其预测能力变无效。 |
| [^4] | [Random Vector Functional Link Networks for Function Approximation on Manifolds](https://arxiv.org/abs/2007.15776) | 本文提供了对于连续函数在紧致域上的通用逼近器的严格证明，填补了单层神经网络随机向量功能链接网络在实践中成功的理论缺口 |
| [^5] | [Randomized Kaczmarz with geometrically smoothed momentum.](http://arxiv.org/abs/2401.09415) | 本文研究了向随机Kaczmarz算法中添加几何平滑动量的效果，并证明了关于最小二乘损失矩阵奇异向量方向上的期望误差。数值示例证明了结果的实用性，并提出了几个问题。 |
| [^6] | [Continuum Limits of Ollivier's Ricci Curvature on data clouds: pointwise consistency and global lower bounds.](http://arxiv.org/abs/2307.02378) | 该论文研究了从数据云中构建的随机几何图与流形之间的曲率关系，并通过概率分析证明了点态一致性以及全局结构特性传承。研究结果对图上热核的收敛性和从数据云中学习流形具有重要的应用价值。 |
| [^7] | [Learning Causally Disentangled Representations via the Principle of Independent Causal Mechanisms.](http://arxiv.org/abs/2306.01213) | 本文通过定义独立因果机制，提出了ICM-VAE框架，使得学习因果解缠绕表示更准确 |
| [^8] | [Optimal signal propagation in ResNets through residual scaling.](http://arxiv.org/abs/2305.07715) | 本文为ResNets导出系统的有限尺寸理论，指出对于深层网络架构，缩放参数是优化信号传播和确保有效利用网络深度方面的关键。 |
| [^9] | [Bayesian neural networks via MCMC: a Python-based tutorial.](http://arxiv.org/abs/2304.02595) | 本文提供了一个基于Python的教程，介绍了贝叶斯神经网络的MCMC方法应用，通过教程使得深度学习开发者能够更好地应用贝叶斯推断进行参数估计和不确定性量化。 |
| [^10] | [Bridging the Usability Gap: Theoretical and Methodological Advances for Spectral Learning of Hidden Markov Models.](http://arxiv.org/abs/2302.07437) | 本文研究了隐马尔可夫模型谱学习中存在的问题，并提出了解决方案，包括提供了SHMM似然估计的误差渐近分布、提出投影SHMM算法可以减轻误差传播问题、并开发了SHMM和PSHMM的在线学习变体以适应潜在的非平稳性。研究结果表明PSHMM具有更好的性能表现。 |
| [^11] | [Causal Estimation of Exposure Shifts with Neural Networks: Evaluating the Health Benefits of Stricter Air Quality Standards in the US.](http://arxiv.org/abs/2302.02560) | 本研究提出了一种神经网络方法，利用其理论基础和实施的可行性，从而估计连续暴露/治疗的分布对政策相关结果的因果效应。我们将此方法应用于包含6800万个个体和2700万个美国境内死亡事件的数据中，通过评估美国国家环境保护局（EPA）对PM2.5的国家环境空气质量标准（NAAQS）进行修订后的健康效益。 |

# 详细

[^1]: 关于随机梯度下降（SGD）的收敛速度及其在修改的多臂赌博机上的策略梯度应用

    On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit

    [https://arxiv.org/abs/2402.06388](https://arxiv.org/abs/2402.06388)

    该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。

    

    我们提出了一个自包含的证明，证明了当学习速率遵循逆时间衰减规则时，随机梯度下降（SGD）的收敛速度；接下来，我们将这些结果应用于带有L2正则化的修改的策略梯度多臂赌博机（MAB）的收敛性分析。

    We present a self-contained proof of the convergence rate of the Stochastic Gradient Descent (SGD) when the learning rate follows an inverse time decays schedule; we next apply the results to the convergence of a modified form of policy gradient Multi-Armed Bandit (MAB) with $L2$ regularization.
    
[^2]: 非凸优化中Alternating Minimization的渐近动力学

    Asymptotic Dynamics of Alternating Minimization for Non-Convex Optimization

    [https://arxiv.org/abs/2402.04751](https://arxiv.org/abs/2402.04751)

    本文研究了交替极小化在非凸优化中的渐近动力学特性，通过复制方法追踪演化过程，发现其可用二维离散随机过程有效描述，存在记忆依赖性。该理论框架适用于分析各种迭代算法。

    

    本研究探讨了在具有正态分布协变量的双线性非凸函数优化中应用交替极小化的渐近动力学。我们采用统计物理学中的复制方法，通过多步方法精确追踪算法的演变。研究结果表明，动力学可以有效地用一个二维离散随机过程来描述，每一步都依赖于所有先前的时间步长，揭示了过程中的记忆依赖性。本文开发的理论框架广泛适用于各种迭代算法的分析，超越了交替极小化的范围。

    This study investigates the asymptotic dynamics of alternating minimization applied to optimize a bilinear non-convex function with normally distributed covariates. We employ the replica method from statistical physics in a multi-step approach to precisely trace the algorithm's evolution. Our findings indicate that the dynamics can be described effectively by a two--dimensional discrete stochastic process, where each step depends on all previous time steps, revealing a memory dependency in the procedure. The theoretical framework developed in this work is broadly applicable for the analysis of various iterative algorithms, extending beyond the scope of alternating minimization.
    
[^3]: 当准确的预测模型导致有害的自我实现预言

    When accurate prediction models yield harmful self-fulfilling prophecies

    [https://arxiv.org/abs/2312.01210](https://arxiv.org/abs/2312.01210)

    本研究通过调查预测模型的部署对决策产生有害影响的情况，发现这些模型可能会成为有害的自我实现预言。这些模型不会因为对某些患者造成更糟糕的结果而使其预测能力变无效。

    

    目标：预测模型在医学研究和实践中非常受欢迎。通过为特定患者预测感兴趣的结果，这些模型可以帮助决策困难的治疗决策，并且通常被誉为个性化的、数据驱动的医疗保健的杰出代表。许多预测模型在验证研究中基于其预测准确性而部署用于决策支持。我们调查这是否是一种安全和有效的方法。材料和方法：我们展示了使用预测模型进行决策可以导致有害的决策，即使在部署后这些预测表现出良好的区分度。这些模型是有害的自我实现预言：它们的部署损害了一群患者，但这些患者的更糟糕的结果并不使模型的预测能力无效。结果：我们的主要结果是对这些预测模型集合的形式化描述。接下来，我们展示了在部署前后都进行了良好校准的模型

    Objective: Prediction models are popular in medical research and practice. By predicting an outcome of interest for specific patients, these models may help inform difficult treatment decisions, and are often hailed as the poster children for personalized, data-driven healthcare. Many prediction models are deployed for decision support based on their prediction accuracy in validation studies. We investigate whether this is a safe and valid approach.   Materials and Methods: We show that using prediction models for decision making can lead to harmful decisions, even when the predictions exhibit good discrimination after deployment. These models are harmful self-fulfilling prophecies: their deployment harms a group of patients but the worse outcome of these patients does not invalidate the predictive power of the model.   Results: Our main result is a formal characterization of a set of such prediction models. Next we show that models that are well calibrated before and after deployment 
    
[^4]: 用于流形上函数逼近的随机向量功能链接网络

    Random Vector Functional Link Networks for Function Approximation on Manifolds

    [https://arxiv.org/abs/2007.15776](https://arxiv.org/abs/2007.15776)

    本文提供了对于连续函数在紧致域上的通用逼近器的严格证明，填补了单层神经网络随机向量功能链接网络在实践中成功的理论缺口

    

    feed-forward神经网络的学习速度因慢而著名，在深度学习应用中已经成为瓶颈数十年。为了应对这一问题，研究人员和实践者尝试引入随机性来减少学习需求。基于Igelnik和Pao的原始构造，具有随机输入到隐藏层权重和偏置的单层神经网络在实践中取得成功，但缺乏必要的理论证明。本文填补了这一理论空白。我们提供了一个（更正的）严格证明，证明Igelnik和Pao的构造是一个连续函数在紧致域上的通用逼近器，逼近误差像渐近衰减。

    arXiv:2007.15776v3 Announce Type: replace-cross  Abstract: The learning speed of feed-forward neural networks is notoriously slow and has presented a bottleneck in deep learning applications for several decades. For instance, gradient-based learning algorithms, which are used extensively to train neural networks, tend to work slowly when all of the network parameters must be iteratively tuned. To counter this, both researchers and practitioners have tried introducing randomness to reduce the learning requirement. Based on the original construction of Igelnik and Pao, single layer neural-networks with random input-to-hidden layer weights and biases have seen success in practice, but the necessary theoretical justification is lacking. In this paper, we begin to fill this theoretical gap. We provide a (corrected) rigorous proof that the Igelnik and Pao construction is a universal approximator for continuous functions on compact domains, with approximation error decaying asymptotically lik
    
[^5]: 具有几何平滑动量的随机Kaczmarz方法

    Randomized Kaczmarz with geometrically smoothed momentum. (arXiv:2401.09415v1 [math.NA])

    [http://arxiv.org/abs/2401.09415](http://arxiv.org/abs/2401.09415)

    本文研究了向随机Kaczmarz算法中添加几何平滑动量的效果，并证明了关于最小二乘损失矩阵奇异向量方向上的期望误差。数值示例证明了结果的实用性，并提出了几个问题。

    

    本文研究了向随机Kaczmarz算法中添加几何平滑动量的效果，该算法是线性最小二乘损失函数上随机梯度下降的实例。我们证明了关于定义最小二乘损失的矩阵的奇异向量方向上期望误差的结果。我们给出了几个数值示例来说明我们结果的实用性，并提出了几个问题。

    This paper studies the effect of adding geometrically smoothed momentum to the randomized Kaczmarz algorithm, which is an instance of stochastic gradient descent on a linear least squares loss function. We prove a result about the expected error in the direction of singular vectors of the matrix defining the least squares loss. We present several numerical examples illustrating the utility of our result and pose several questions.
    
[^6]: 在数据云中Ollivier的Ricci曲率的连续极限：点态一致性和全局下界

    Continuum Limits of Ollivier's Ricci Curvature on data clouds: pointwise consistency and global lower bounds. (arXiv:2307.02378v1 [math.DG])

    [http://arxiv.org/abs/2307.02378](http://arxiv.org/abs/2307.02378)

    该论文研究了从数据云中构建的随机几何图与流形之间的曲率关系，并通过概率分析证明了点态一致性以及全局结构特性传承。研究结果对图上热核的收敛性和从数据云中学习流形具有重要的应用价值。

    

    让$\mathcal{M} \subseteq \mathbb{R}^d$表示一个低维流形，$\mathcal{X}= \{ x_1, \dots, x_n \}$表示从$\mathcal{M}$均匀采样得到的一组点。我们研究了从$\mathcal{X}$构建的随机几何图与流形$\mathcal{M}$的曲率之间的关系，通过Ollivier的离散Ricci曲率的连续极限。我们证明了点态、非渐近一致性结果，并且还表明，如果$\mathcal{M}$的Ricci曲率从下面严格地被一个正常数界住，那么随机几何图将以高概率继承此全局结构特性。我们讨论全局离散曲率界限在图上热核的收敛性质以及对数据云上流形学习的影响。特别地，我们展示了一致性结果允许通过外禀曲率表征流形的内在曲率。

    Let $\mathcal{M} \subseteq \mathbb{R}^d$ denote a low-dimensional manifold and let $\mathcal{X}= \{ x_1, \dots, x_n \}$ be a collection of points uniformly sampled from $\mathcal{M}$. We study the relationship between the curvature of a random geometric graph built from $\mathcal{X}$ and the curvature of the manifold $\mathcal{M}$ via continuum limits of Ollivier's discrete Ricci curvature. We prove pointwise, non-asymptotic consistency results and also show that if $\mathcal{M}$ has Ricci curvature bounded from below by a positive constant, then the random geometric graph will inherit this global structural property with high probability. We discuss applications of the global discrete curvature bounds to contraction properties of heat kernels on graphs, as well as implications for manifold learning from data clouds. In particular, we show that the consistency results allow for characterizing the intrinsic curvature of a manifold from extrinsic curvature.
    
[^7]: 基于独立因果机制原则学习因果解缠绕表示

    Learning Causally Disentangled Representations via the Principle of Independent Causal Mechanisms. (arXiv:2306.01213v1 [cs.LG])

    [http://arxiv.org/abs/2306.01213](http://arxiv.org/abs/2306.01213)

    本文通过定义独立因果机制，提出了ICM-VAE框架，使得学习因果解缠绕表示更准确

    

    学习解缠绕的因果表示是一个具有挑战性的问题，近年来因其对提取下游任务的有意义信息而引起了广泛关注。本文从独立因果机制的角度定义了一种新的因果解缠绕概念。我们提出了ICM-VAE框架，通过因因果关系观察标签来监督学习因果解缠绕表示。我们使用可学习的基于流的微分同胚函数将噪声变量映射到潜在因果变量中来建模因果机制。此外，为了促进因果要素的解缠绕，我们提出了一种因果解缠绕先验，利用已知的因果结构来鼓励在潜在空间中学习因果分解分布。在相对温和的条件下，我们提供了理论结果，显示了因果要素和机制的可识别性，直到排列和逐元重参数化的限度。我们进行了实证研究...

    Learning disentangled causal representations is a challenging problem that has gained significant attention recently due to its implications for extracting meaningful information for downstream tasks. In this work, we define a new notion of causal disentanglement from the perspective of independent causal mechanisms. We propose ICM-VAE, a framework for learning causally disentangled representations supervised by causally related observed labels. We model causal mechanisms using learnable flow-based diffeomorphic functions to map noise variables to latent causal variables. Further, to promote the disentanglement of causal factors, we propose a causal disentanglement prior that utilizes the known causal structure to encourage learning a causally factorized distribution in the latent space. Under relatively mild conditions, we provide theoretical results showing the identifiability of causal factors and mechanisms up to permutation and elementwise reparameterization. We empirically demons
    
[^8]: 通过残差缩放实现ResNets的信号最优传递

    Optimal signal propagation in ResNets through residual scaling. (arXiv:2305.07715v1 [cond-mat.dis-nn])

    [http://arxiv.org/abs/2305.07715](http://arxiv.org/abs/2305.07715)

    本文为ResNets导出系统的有限尺寸理论，指出对于深层网络架构，缩放参数是优化信号传播和确保有效利用网络深度方面的关键。

    

    Residual网络（ResNets）在大深度上比前馈神经网络具有更好的训练能力和性能。引入跳过连接可以促进信号向更深层的传递。此外，先前的研究发现为残差分支添加缩放参数可以进一步提高泛化性能。尽管他们经验性地确定了这种缩放参数特别有利的取值范围，但其相关的性能提升及其在网络超参数上的普适性仍需要进一步理解。对于前馈神经网络（FFNets），有限尺寸理论在信号传播和超参数调节方面获得了重要洞见。我们在这里为ResNets导出了一个系统的有限尺寸理论，以研究信号传播及其对残差分支缩放的依赖性。我们导出响应函数的分析表达式，这是衡量网络对输入敏感性的一种指标，并表明对于深层网络架构，缩放参数在优化信号传播和确保有效利用网络深度方面发挥着至关重要的作用。

    Residual networks (ResNets) have significantly better trainability and thus performance than feed-forward networks at large depth. Introducing skip connections facilitates signal propagation to deeper layers. In addition, previous works found that adding a scaling parameter for the residual branch further improves generalization performance. While they empirically identified a particularly beneficial range of values for this scaling parameter, the associated performance improvement and its universality across network hyperparameters yet need to be understood. For feed-forward networks (FFNets), finite-size theories have led to important insights with regard to signal propagation and hyperparameter tuning. We here derive a systematic finite-size theory for ResNets to study signal propagation and its dependence on the scaling for the residual branch. We derive analytical expressions for the response function, a measure for the network's sensitivity to inputs, and show that for deep netwo
    
[^9]: 基于MCMC的贝叶斯神经网络：基于Python的教程

    Bayesian neural networks via MCMC: a Python-based tutorial. (arXiv:2304.02595v1 [stat.ML])

    [http://arxiv.org/abs/2304.02595](http://arxiv.org/abs/2304.02595)

    本文提供了一个基于Python的教程，介绍了贝叶斯神经网络的MCMC方法应用，通过教程使得深度学习开发者能够更好地应用贝叶斯推断进行参数估计和不确定性量化。

    

    贝叶斯推断为机器学习和深度学习提供了参数估计和不确定性量化的方法。变分推断和马尔科夫链蒙特卡罗（MCMC）采样技术用于实现贝叶斯推断。在过去三十年中，MCMC方法在适应更大的模型（如深度学习）和大数据问题方面面临了许多挑战。包括梯度的高级提议（例如Langevin提议分布）提供了一种解决MCMC采样中的一些限制的方法，此外，MCMC方法通常被限制在统计学家的使用范围内，并且仍不是深度学习研究人员的主流方法。我们提供了一个MCMC方法的教程，涵盖了简单的贝叶斯线性和逻辑模型，以及贝叶斯神经网络。这个教程的目的是通过编码来弥合理论和实现之间的差距，鉴于当前MCMC方法的普及程度仍然较低。

    Bayesian inference provides a methodology for parameter estimation and uncertainty quantification in machine learning and deep learning methods. Variational inference and Markov Chain Monte-Carlo (MCMC) sampling techniques are used to implement Bayesian inference. In the past three decades, MCMC methods have faced a number of challenges in being adapted to larger models (such as in deep learning) and big data problems. Advanced proposals that incorporate gradients, such as a Langevin proposal distribution, provide a means to address some of the limitations of MCMC sampling for Bayesian neural networks. Furthermore, MCMC methods have typically been constrained to use by statisticians and are still not prominent among deep learning researchers. We present a tutorial for MCMC methods that covers simple Bayesian linear and logistic models, and Bayesian neural networks. The aim of this tutorial is to bridge the gap between theory and implementation via coding, given a general sparsity of li
    
[^10]: 缩小可用性差距：隐马尔可夫模型谱学习的理论与方法学进展

    Bridging the Usability Gap: Theoretical and Methodological Advances for Spectral Learning of Hidden Markov Models. (arXiv:2302.07437v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.07437](http://arxiv.org/abs/2302.07437)

    本文研究了隐马尔可夫模型谱学习中存在的问题，并提出了解决方案，包括提供了SHMM似然估计的误差渐近分布、提出投影SHMM算法可以减轻误差传播问题、并开发了SHMM和PSHMM的在线学习变体以适应潜在的非平稳性。研究结果表明PSHMM具有更好的性能表现。

    

    Baum-Welch（B-W）算法是推断隐马尔可夫模型(HMM)最广泛接受的方法。 然而，它很容易陷入局部最优，而且对于许多实时应用来说速度太慢。文献中提出了一种基于矩法（MOM）的HMM的谱学习（SHMM），旨在克服这些障碍。尽管有这样的承诺，但SHMM的渐近理论一直很难得到，而SHMM的长期性能可能会由于误差的无限传播而降低。在本文中，我们(1)提供了SHMM似然估计的近似误差的渐近分布，(2)提出了一种新算法称为投影SHMM（PSHMM），它可以减轻误差传播问题，(3)开发了SHMM和PSHMM的在线学习变体，以适应潜在的非平稳性。我们在模拟数据和来自真实世界应用的数据上比较了SHMM、PSHMM和B-W算法的性能。

    The Baum-Welch (B-W) algorithm is the most widely accepted method for inferring hidden Markov models (HMM). However, it is prone to getting stuck in local optima, and can be too slow for many real-time applications. Spectral learning of HMMs (SHMM), based on the method of moments (MOM) has been proposed in the literature to overcome these obstacles. Despite its promises, asymptotic theory for SHMM has been elusive, and the long-run performance of SHMM can degrade due to unchecked propagation of error. In this paper, we (1) provide an asymptotic distribution for the approximate error of the likelihood estimated by SHMM, (2) propose a novel algorithm called projected SHMM (PSHMM) that mitigates the problem of error propagation, and (3) develop online learning variants of both SHMM and PSHMM that accommodate potential nonstationarity. We compare the performance of SHMM with PSHMM and estimation through the B-W algorithm on both simulated data and data from real world applications, and fin
    
[^11]: 神经网络在因果估计中的应用: 在美国评估更严格的空气质量标准的健康效益

    Causal Estimation of Exposure Shifts with Neural Networks: Evaluating the Health Benefits of Stricter Air Quality Standards in the US. (arXiv:2302.02560v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02560](http://arxiv.org/abs/2302.02560)

    本研究提出了一种神经网络方法，利用其理论基础和实施的可行性，从而估计连续暴露/治疗的分布对政策相关结果的因果效应。我们将此方法应用于包含6800万个个体和2700万个美国境内死亡事件的数据中，通过评估美国国家环境保护局（EPA）对PM2.5的国家环境空气质量标准（NAAQS）进行修订后的健康效益。

    

    在政策研究中，估计连续性暴露/治疗的分布对感兴趣的结果的因果效应是最关键的分析任务之一。我们称之为偏移-响应函数（SRF）估计问题。现有的涉及强健因果效应估计器的神经网络方法缺乏理论保证和实际实现，用于SRF估计。受公共卫生中的关键政策问题的启发，我们开发了一种神经网络方法及其理论基础，以提供具有强健性和效率保证的SRF估计。然后，我们将我们的方法应用于包含6800万个个体和2700万个美国境内死亡事件的数据中，以估计将美国国家环境保护局（EPA）最近提议从12 μg/m³改为9 μg/m³的PM2.5的美国国家环境空气质量标准（NAAQS）的修订对结果的因果效应。我们的目标是首次估计

    In policy research, one of the most critical analytic tasks is to estimate the causal effect of a policy-relevant shift to the distribution of a continuous exposure/treatment on an outcome of interest. We call this problem shift-response function (SRF) estimation. Existing neural network methods involving robust causal-effect estimators lack theoretical guarantees and practical implementations for SRF estimation. Motivated by a key policy-relevant question in public health, we develop a neural network method and its theoretical underpinnings to estimate SRFs with robustness and efficiency guarantees. We then apply our method to data consisting of 68 million individuals and 27 million deaths across the U.S. to estimate the causal effect from revising the US National Ambient Air Quality Standards (NAAQS) for PM 2.5 from 12 $\mu g/m^3$ to 9 $\mu g/m^3$. This change has been recently proposed by the US Environmental Protection Agency (EPA). Our goal is to estimate, for the first time, the 
    

