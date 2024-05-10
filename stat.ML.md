# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inference for Regression with Variables Generated from Unstructured Data](https://arxiv.org/abs/2402.15585) | 提出了一种使用联合上游和下游模型进行有效推断的一步策略，显著减少了偏误，在CEO时间利用数据的应用中产生了重要效果，适合应用研究人员。 |
| [^2] | [Riemann-Lebesgue Forest for Regression](https://arxiv.org/abs/2402.04550) | 提出了一种新颖的集成方法Riemann-Lebesgue Forest (RLF)用于回归问题，通过划分函数的值域为多个区间来逼近可测函数的思想，开发了一种新的树学习算法Riemann-Lebesgue Tree。通过Hoeffding分解和Stein方法推导了RLF在不同参数设置下的渐近性能，并在仿真数据和真实世界数据集上的实验中证明了RLF与原始随机森林相比具有竞争力的性能。 |
| [^3] | [Sliced Wasserstein with Random-Path Projecting Directions.](http://arxiv.org/abs/2401.15889) | 本研究提出了一种无需优化的切片分布方法，该方法能够快速进行蒙特卡洛期望估计。通过利用随机向量之间的归一化差异构建随机路径投影方向，从而得到了随机路径切片分布和两个切片瓦瑟斯坦的变种。这种方法在拓扑、统计和计算性质上有重要意义。 |
| [^4] | [Learning Causally Disentangled Representations via the Principle of Independent Causal Mechanisms.](http://arxiv.org/abs/2306.01213) | 本文通过定义独立因果机制，提出了ICM-VAE框架，使得学习因果解缠绕表示更准确 |
| [^5] | [Continual Learning of Multi-modal Dynamics with External Memory.](http://arxiv.org/abs/2203.00936) | 本文提出了一种新的连续学习方法，通过在记忆中维护遇到序列模式的描述符来实现，能够有效处理新的行为模式的连续出现。 |

# 详细

[^1]: 使用来自非结构化数据生成的变量进行回归的推断

    Inference for Regression with Variables Generated from Unstructured Data

    [https://arxiv.org/abs/2402.15585](https://arxiv.org/abs/2402.15585)

    提出了一种使用联合上游和下游模型进行有效推断的一步策略，显著减少了偏误，在CEO时间利用数据的应用中产生了重要效果，适合应用研究人员。

    

    分析非结构化数据的主要策略包括两个步骤。首先，使用上游信息检索模型估计感兴趣的潜在经济变量。其次，将估计值视为下游计量经济模型中的“数据”。我们建立了理论论点，解释为什么在实证合理的设置中，这种两步策略会导致偏误的推断。更具建设性的是，我们提出了一个有效推断的一步策略，该策略同时使用上游和下游模型。在模拟中，这一步策略(i) 显著减少了偏误；(ii) 在使用CEO时间利用数据的主要应用中产生了定量重要的效果；(iii) 可以很容易地被应用研究人员采用。

    arXiv:2402.15585v1 Announce Type: new  Abstract: The leading strategy for analyzing unstructured data uses two steps. First, latent variables of economic interest are estimated with an upstream information retrieval model. Second, the estimates are treated as "data" in a downstream econometric model. We establish theoretical arguments for why this two-step strategy leads to biased inference in empirically plausible settings. More constructively, we propose a one-step strategy for valid inference that uses the upstream and downstream models jointly. The one-step strategy (i) substantially reduces bias in simulations; (ii) has quantitatively important effects in a leading application using CEO time-use data; and (iii) can be readily adapted by applied researchers.
    
[^2]: Riemann-Lebesgue Forest回归方法的研究

    Riemann-Lebesgue Forest for Regression

    [https://arxiv.org/abs/2402.04550](https://arxiv.org/abs/2402.04550)

    提出了一种新颖的集成方法Riemann-Lebesgue Forest (RLF)用于回归问题，通过划分函数的值域为多个区间来逼近可测函数的思想，开发了一种新的树学习算法Riemann-Lebesgue Tree。通过Hoeffding分解和Stein方法推导了RLF在不同参数设置下的渐近性能，并在仿真数据和真实世界数据集上的实验中证明了RLF与原始随机森林相比具有竞争力的性能。

    

    我们提出了一种新颖的用于回归问题的集成方法，称为Riemann-Lebesgue Forest (RLF)。RLF的核心思想是通过将函数的值域划分为几个区间来模拟可测函数的逼近方式。基于这个思想，我们开发了一种新的树学习算法，称为Riemann-Lebesgue Tree，它在每个非叶节点上有机会从响应Y或特征空间X中的方向进行切割。我们通过Hoeffding分解和Stein方法来推导不同参数设置下RLF的渐近性能。当底层函数Y=f(X)遵循加法回归模型时，RLF与Scornet等人的论证（2014年）保持一致。通过在仿真数据和真实世界数据集上的实验证明，RLF与原始随机森林相比具有竞争力的性能。

    We propose a novel ensemble method called Riemann-Lebesgue Forest (RLF) for regression. The core idea of RLF is to mimic the way how a measurable function can be approximated by partitioning its range into a few intervals. With this idea in mind, we develop a new tree learner named Riemann-Lebesgue Tree which has a chance to split the node from response $Y$ or a direction in feature space $\mathbf{X}$ at each non-terminal node. We generalize the asymptotic performance of RLF under different parameter settings mainly through Hoeffding decomposition \cite{Vaart} and Stein's method \cite{Chen2010NormalAB}. When the underlying function $Y=f(\mathbf{X})$ follows an additive regression model, RLF is consistent with the argument from \cite{Scornet2014ConsistencyOR}. The competitive performance of RLF against original random forest \cite{Breiman2001RandomF} is demonstrated by experiments in simulation data and real world datasets.
    
[^3]: 带有随机路径投影方向的切片瓦瑟斯坦方法

    Sliced Wasserstein with Random-Path Projecting Directions. (arXiv:2401.15889v1 [stat.ML])

    [http://arxiv.org/abs/2401.15889](http://arxiv.org/abs/2401.15889)

    本研究提出了一种无需优化的切片分布方法，该方法能够快速进行蒙特卡洛期望估计。通过利用随机向量之间的归一化差异构建随机路径投影方向，从而得到了随机路径切片分布和两个切片瓦瑟斯坦的变种。这种方法在拓扑、统计和计算性质上有重要意义。

    

    在应用中，切片分布选择已被用作提高基于最小化切片瓦瑟斯坦距离的参数估计器性能的有效技术。先前的工作要么利用昂贵的优化来选择切片分布，要么使用需要昂贵的抽样方法的切片分布。在这项工作中，我们提出了一种无需优化的切片分布，可以快速进行蒙特卡洛期望估计的抽样。具体来说，我们引入了随机路径投影方向（RPD），它是通过利用两个输入测量中两个随机向量之间的归一化差异构建的。从RPD中，我们得到了随机路径切片分布（RPSD）和两个切片瓦瑟斯坦的变种，即随机路径投影切片瓦瑟斯坦（RPSW）和重要性加权随机路径投影切片瓦瑟斯坦（IWRPSW）。然后我们讨论了拓扑、统计和计算性质。

    Slicing distribution selection has been used as an effective technique to improve the performance of parameter estimators based on minimizing sliced Wasserstein distance in applications. Previous works either utilize expensive optimization to select the slicing distribution or use slicing distributions that require expensive sampling methods. In this work, we propose an optimization-free slicing distribution that provides a fast sampling for the Monte Carlo estimation of expectation. In particular, we introduce the random-path projecting direction (RPD) which is constructed by leveraging the normalized difference between two random vectors following the two input measures. From the RPD, we derive the random-path slicing distribution (RPSD) and two variants of sliced Wasserstein, i.e., the Random-Path Projection Sliced Wasserstein (RPSW) and the Importance Weighted Random-Path Projection Sliced Wasserstein (IWRPSW). We then discuss the topological, statistical, and computational propert
    
[^4]: 基于独立因果机制原则学习因果解缠绕表示

    Learning Causally Disentangled Representations via the Principle of Independent Causal Mechanisms. (arXiv:2306.01213v1 [cs.LG])

    [http://arxiv.org/abs/2306.01213](http://arxiv.org/abs/2306.01213)

    本文通过定义独立因果机制，提出了ICM-VAE框架，使得学习因果解缠绕表示更准确

    

    学习解缠绕的因果表示是一个具有挑战性的问题，近年来因其对提取下游任务的有意义信息而引起了广泛关注。本文从独立因果机制的角度定义了一种新的因果解缠绕概念。我们提出了ICM-VAE框架，通过因因果关系观察标签来监督学习因果解缠绕表示。我们使用可学习的基于流的微分同胚函数将噪声变量映射到潜在因果变量中来建模因果机制。此外，为了促进因果要素的解缠绕，我们提出了一种因果解缠绕先验，利用已知的因果结构来鼓励在潜在空间中学习因果分解分布。在相对温和的条件下，我们提供了理论结果，显示了因果要素和机制的可识别性，直到排列和逐元重参数化的限度。我们进行了实证研究...

    Learning disentangled causal representations is a challenging problem that has gained significant attention recently due to its implications for extracting meaningful information for downstream tasks. In this work, we define a new notion of causal disentanglement from the perspective of independent causal mechanisms. We propose ICM-VAE, a framework for learning causally disentangled representations supervised by causally related observed labels. We model causal mechanisms using learnable flow-based diffeomorphic functions to map noise variables to latent causal variables. Further, to promote the disentanglement of causal factors, we propose a causal disentanglement prior that utilizes the known causal structure to encourage learning a causally factorized distribution in the latent space. Under relatively mild conditions, we provide theoretical results showing the identifiability of causal factors and mechanisms up to permutation and elementwise reparameterization. We empirically demons
    
[^5]: 带外部记忆的多模态动态连续学习

    Continual Learning of Multi-modal Dynamics with External Memory. (arXiv:2203.00936v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2203.00936](http://arxiv.org/abs/2203.00936)

    本文提出了一种新的连续学习方法，通过在记忆中维护遇到序列模式的描述符来实现，能够有效处理新的行为模式的连续出现。

    

    本文研究了在新的行为模式连续出现时，如何将模型拟合到动态环境中。学习模型能够意识到新的模式出现，但它没有访问单个训练序列的真实模式的信息。目前的连续学习方法无法处理这种情况，因为参数传递受到灾难性干扰的影响，而情节记忆设计需要知道序列的真实模式。我们设计了一种新的连续学习方法，通过在神经情节记忆中维护遇到的序列模式的描述符来克服这两个限制。我们在记忆的注意权重上使用Dirichlet过程先验，以促进模式描述符的有效存储。通过检索先前任务相似模式的描述符，并将此描述符馈入其转移中，我们的方法通过在任务之间传递知识来执行连续学习。

    We study the problem of fitting a model to a dynamical environment when new modes of behavior emerge sequentially. The learning model is aware when a new mode appears, but it does not have access to the true modes of individual training sequences. The state-of-the-art continual learning approaches cannot handle this setup, because parameter transfer suffers from catastrophic interference and episodic memory design requires the knowledge of the ground-truth modes of sequences. We devise a novel continual learning method that overcomes both limitations by maintaining a descriptor of the mode of an encountered sequence in a neural episodic memory. We employ a Dirichlet Process prior on the attention weights of the memory to foster efficient storage of the mode descriptors. Our method performs continual learning by transferring knowledge across tasks by retrieving the descriptors of similar modes of past tasks to the mode of a current sequence and feeding this descriptor into its transitio
    

