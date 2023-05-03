# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sequence Modeling with Multiresolution Convolutional Memory.](http://arxiv.org/abs/2305.01638) | 本论文提出了一种新的用于序列建模的构建块，称为MultiresLayer，通过多分辨率卷积捕获输入序列中的多尺度趋势，既具有卷积网络的计算优势，又具有小波分解的有理论基础的动机。 |
| [^2] | [Revisiting Gradient Clipping: Stochastic bias and tight convergence guarantees.](http://arxiv.org/abs/2305.01588) | 本文提出了针对梯度剪切的收敛保证机制，不再需要特定的阈值和强噪声假设，同时可以独立于步长选择，从而提高了收敛的自由度。 |
| [^3] | [Unlocking the Power of Representations in Long-term Novelty-based Exploration.](http://arxiv.org/abs/2305.01521) | 本论文介绍了一种名为RECODE的非参数新颖性探索方法，它在深度强化学习中跟踪状态访问计数，并与新颖的逆动力学损失相结合，实现了在具有挑战性的任务中的最新最佳表现。 |
| [^4] | [On the properties of Gaussian Copula Mixture Models.](http://arxiv.org/abs/2305.01479) | 本文研究了高斯Copula混合模型（GCMM）的性质，开发了基于扩展期望最大算法的参数估计方法，并表明GCMM相比于GMM可以更好地拟合数据并实现更深入的数据挖掘。 |
| [^5] | [Stochastic Contextual Bandits with Graph-based Contexts.](http://arxiv.org/abs/2305.01470) | 本文提出了一种基于图上下文的随机情境赌博机问题，其中节点标签相同的节点共享相同的奖励分布。对于线图和树，我们提出了一种具有遗憾界的算法。 |
| [^6] | [Memory of recurrent networks: Do we compute it right?.](http://arxiv.org/abs/2305.01457) | 本文研究了线性回声状态网络的记忆容量计算问题。通过发现数值评估的不准确性主要源于数值方面的问题，提出了基于掩码矩阵MC相对于中立性的稳健数值方法，该方法可以解决数值评估中的误差问题。 |
| [^7] | [Unsupervised Feature Based Algorithms for Time Series Extrinsic Regression.](http://arxiv.org/abs/2305.01429) | 本研究提出了两种新的TSER算法：FreshPRINCE和DrCIF，它们分别由一组汇总特征和多个条件推理树构成，能够更好地在时间序列外源回归的问题上预测响应变量，比起以前的评估中使用的基线算法表现更佳。 |
| [^8] | [Are demographically invariant models and representations in medical imaging fair?.](http://arxiv.org/abs/2305.01397) | 医学影像模型编码患者人口统计信息，引发有关潜在歧视的担忧。研究表明，不编码人口属性的模型容易损失预测性能，而考虑人口统计属性的反事实模型不变性存在复杂性。人口统计学编码可以被认为是优势。 |
| [^9] | [LogSpecT: Feasible Graph Learning Model from Stationary Signals with Recovery Guarantees.](http://arxiv.org/abs/2305.01379) | 本文提出了一种新的图形学习模型LogSpecT及其实际公式rLogSpecT，以解决现有模型rSpecT敏感超参数选择、不可行的问题。本文提供了rLogSpecT的恢复保证，并提出了基于L-ADMM的高效算法。 |
| [^10] | [Random Function Descent.](http://arxiv.org/abs/2305.01377) | 本文提出了随机函数下降(RFD)算法，可以在随机环境中计算出步长并且与贝叶斯优化中的梯度下降算法相同。在合成基准测试中，RFD算法比未调整的Adam方法表现更好，提出的heuristic扩展可与调整后的Adam方法相媲美。 |
| [^11] | [Addressing Parameter Choice Issues in Unsupervised Domain Adaptation by Aggregation.](http://arxiv.org/abs/2305.01281) | 本文提出了一种使用线性聚合的方法来解决无监督领域适应中的参数选择问题，并且展示了该算法的目标误差渐近不劣于未知最佳聚合的两倍，大规模实证研究表明该方法优于深度嵌入验证。 |
| [^12] | [Unbounded Differentially Private Quantile and Maximum Estimation.](http://arxiv.org/abs/2305.01177) | 本文研究了如何对无上限数据进行差分隐私分位数和最大值的计算。调用基本的稀疏向量技术中的$\texttt{AboveThreshold}$子程序可以实现这个目标，可以提供更准确和稳健的最高分位数估计，从而应用于对于差分隐私求和和均值估计至关重要的数据剪切，该技术的隐私保障可以通过方法改进。 |
| [^13] | [Understanding the Generalization Ability of Deep Learning Algorithms: A Kernelized Renyi's Entropy Perspective.](http://arxiv.org/abs/2305.01143) | 本论文提出了一种新的信息理论度量方法——基于核化Renyi熵，用于在不假设Lipschitz或凸性条件的前提下对SGD / SGLD等下降学习算法进行分析，旨在提高当前泛化误差界限的优化水平。 |
| [^14] | [Performative Prediction with Bandit Feedback: Learning through Reparameterization.](http://arxiv.org/abs/2305.01094) | 本文提出一种新的在线反馈的实现式预测框架，解决了在模型部署自身改变数据分布的情况下优化准确性的问题。 |
| [^15] | [Model-agnostic Measure of Generalization Difficulty.](http://arxiv.org/abs/2305.01034) | 该论文提出了第一个无特定模型的、量化机器学习测试泛化难度的方法——归纳偏差复杂度度量。该方法量化了在任务上良好泛化所需的总信息量与数据提供的信息量之差，通常需要在许多维度上泛化的任务比涉及更少维度但要求更多细节的任务要困难得多。 |
| [^16] | [Spectral clustering in the Gaussian mixture block model.](http://arxiv.org/abs/2305.00979) | 本文首次研究了从高维高斯混合块模型中抽样的图聚类和嵌入问题。 |
| [^17] | [ContraNorm: A Contrastive Learning Perspective on Oversmoothing and Beyond.](http://arxiv.org/abs/2303.06562) | 本研究提出了一种新的规范化层——ContraNorm，针对图神经网络和变压器中的过度平滑问题，通过对比学习的方式在嵌入空间中破坏表示，缓解了完全塌陷和维度塌陷的现象，并在实验中表现出较高的精度。 |
| [^18] | [Exploring Numerical Priors for Low-Rank Tensor Completion with Generalized CP Decomposition.](http://arxiv.org/abs/2302.05881) | 本文提出了一种新的方法框架GCDTC，利用数值先验和广义CP分解实现了更高的低秩张量补全精度；同时介绍了一个算法SPTC，作为该框架的一个实现。在实验中，该方法表现出比现有技术更好的性能。 |
| [^19] | [Training Neural Networks for Sequential Change-point Detection.](http://arxiv.org/abs/2210.17312) | 本文介绍了一种使用神经网络进行在线变点检测的方法，通过训练神经网络逐步计算检测统计量的累积和来检测变点，并在合成和真实数据上证明了该方法的优越性和潜力。 |
| [^20] | [Transformers Learn Shortcuts to Automata.](http://arxiv.org/abs/2210.10749) | Transformer模型通过重新参数化其循环动态，可以使用比推理步骤更少的层数执行任何有限状态自动机的计算。多项式大小的 $O(\log T)$ 深度解决方案始终存在，而且$O(1)$深度模拟器是非常普遍的。 |
| [^21] | [Conditional Feature Importance for Mixed Data.](http://arxiv.org/abs/2210.03047) | 本研究提出了一种针对混合数据的条件特征重要性框架，使用条件预测影响和顺序knockoff抽样结合，以解决很少讨论的条件和边缘度量之间的重要区别，并揭示出为测试条件FI，目前只有少数方法可用且过去从业者由于数据要求不匹配而受到严重限制。 |
| [^22] | [A physics-based domain adaptation framework for modelling and forecasting building energy systems.](http://arxiv.org/abs/2208.09456) | 本文提出了一种基于物理学的领域自适应框架，将线性时不变状态空间模型与基于子空间的无监督降阶建模相结合，通过最小化模型在共享子空间中的表示差异，将在一个领域上训练过的模型适应到另一个领域，用于建筑能量系统的建模和预测，并在实验中表现出优异的预测性能。 |
| [^23] | [Boosted Off-Policy Learning.](http://arxiv.org/abs/2208.01148) | 我们提出了一种基于Boosting的离线策略学习算法，将基础学习器简化为监督学习，获得了广泛的实际效益；实验结果表明其应用能力优于深度神经网络的离线策略学习和简单回归方法。 |
| [^24] | [LogGENE: A smooth alternative to check loss for Deep Healthcare Inference Tasks.](http://arxiv.org/abs/2206.09333) | LogGENE采用分位数回归框架预测基因表达水平的完整条件分位数，从而为高通量基因组学提供了一种能提供解释和报告不确定性估计、鲁棒性强的推断方法。 |
| [^25] | [Learning Physics between Digital Twins with Low-Fidelity Models and Physics-Informed Gaussian Processes.](http://arxiv.org/abs/2206.08201) | 本文提出了一种新方法，通过低保真模型和物理知识驱动高斯过程进行数字孪生之间的学习， 并开发了贝叶斯分层建模框架允许多个数字孪生之间共享信息。 |
| [^26] | [Improving adversarial robustness by putting more regularizations on less robust samples.](http://arxiv.org/abs/2206.03353) | 本文提出了一种新的对抗训练算法，通过在容易受到对抗攻击的数据上施加更多正则化以提高对抗性鲁棒性，得到了在准确性和鲁棒性方面均为最优的表现。 |
| [^27] | [Bayesian Model Selection, the Marginal Likelihood, and Generalization.](http://arxiv.org/abs/2202.11678) | 本文回顾和探讨了边缘似然在构造约束和假设测试方面的有用性，强调了使用边缘似然作为泛化的代理的问题，并展示了其如何与神经架构搜索相关，可能导致超参数学习中的欠拟合和过拟合。 |
| [^28] | [CD-ROM: Complemented Deep-Reduced Order Model.](http://arxiv.org/abs/2202.10746) | 本文介绍了一种基于深度学习的闭合建模方法CD-ROM，用于经典的POD-Galerkin降阶模型，该方法可以显著提高降阶模型的准确性和稳定性。 |
| [^29] | [Non-asymptotic estimates for TUSLA algorithm for non-convex learning with applications to neural networks with ReLU activation function.](http://arxiv.org/abs/2107.08649) | 本文研究了非凸随机优化问题，提出了TUSLA算法在Wasserstein-1和Wasserstein-2距离上的非渐进误差界限，进而推导了期望过量风险的非渐进估计值。在ReLU神经网络中，理论和数值实验表明TUSLA算法能够高效且精确地解决此类优化问题。 |
| [^30] | [Word Embeddings: A Survey.](http://arxiv.org/abs/1901.09069) | 这篇综述介绍了一些主要的词向量构建策略，称为word embeddings，这些策略基于分布假设，编码了语法和语义信息，并被证明在很多NLP任务中是有用的额外特征。 |
| [^31] | [Pairwise Covariates-adjusted Block Model for Community Detection.](http://arxiv.org/abs/1807.03469) | 基于SBM模型，双重协变量调整的PCABM模型添加了关于节点间关系的附加信息。SCWA算法对PCABM模型进行了高效求解。模拟实验和实际数据分析表明PCABM模型具有优异的性能和预测能力。 |

# 详细

[^1]: 多分辨率卷积记忆的序列建模

    Sequence Modeling with Multiresolution Convolutional Memory. (arXiv:2305.01638v1 [cs.LG])

    [http://arxiv.org/abs/2305.01638](http://arxiv.org/abs/2305.01638)

    本论文提出了一种新的用于序列建模的构建块，称为MultiresLayer，通过多分辨率卷积捕获输入序列中的多尺度趋势，既具有卷积网络的计算优势，又具有小波分解的有理论基础的动机。

    

    有效地捕捉对于某个任务（如分类和生成建模）显著的顺序数据源中的长程模式是一个基本挑战。我们从基于小波的多分辨率分析中获得灵感，定义了一个新的用于序列建模的构建块，称为MultiresLayer。我们模型的关键组成部分是多分辨率卷积，以捕获输入序列中的多尺度趋势。我们的MultiresConv可以通过在扩张的因果卷积树上使用共享过滤器来实现。因此，它既具有卷积网络的计算优势，又具有小波分解的有理论基础的动机。

    Efficiently capturing the long-range patterns in sequential data sources salient to a given task -- such as classification and generative modeling -poses a fundamental challenge. Popular approaches in the space tradeoff between the memory burden of brute-force enumeration and comparison, as in transformers, the computational burden of complicated sequential dependencies, as in recurrent neural networks, or the parameter burden of convolutional networks with many or large filters. We instead take inspiration from wavelet-based multiresolution analysis to define a new building block for sequence modeling, which we call a MultiresLayer. The key component of our model is the multiresolution convolution, capturing multiscale trends in the input sequence. Our MultiresConv can be implemented with shared filters across a dilated causal convolution tree. Thus it garners the computational advantages of convolutional networks and the principled theoretical motivation of wavelet decompositions. 
    
[^2]: 重新审视梯度剪切：随机偏差和紧密收敛性保证。

    Revisiting Gradient Clipping: Stochastic bias and tight convergence guarantees. (arXiv:2305.01588v1 [cs.LG])

    [http://arxiv.org/abs/2305.01588](http://arxiv.org/abs/2305.01588)

    本文提出了针对梯度剪切的收敛保证机制，不再需要特定的阈值和强噪声假设，同时可以独立于步长选择，从而提高了收敛的自由度。

    

    梯度剪切是标准（随机）梯度下降的一种流行修改方法，每次迭代将梯度范数限制在某个值c>0。它被广泛用于稳定深度学习模型的训练( Goodfellow et al., 2016 )或强制实施差分隐私( Abadi et al., 2016 )。尽管剪切机制受欢迎且简单，但其收敛保证通常需要特定的$c$值和强噪声假设。在本文中，我们给出了收敛保证，显示了对任意剪辑阈值的精确依赖，并且表明我们的保证在确定性和随机梯度下都是紧密的。特别地，我们表明(i)对于确定性的梯度下降，剪辑阈值仅影响收敛的高阶项，(ii)在随机设置中，即使对于任意小的步长，也不能保证收敛到真正的最优解在标准的噪声假设下，我们给出了机器学习特定的随机噪声假设，在此假设下，收敛是保证的，剪切阈值$c$可以独立于步长选择。

    Gradient clipping is a popular modification to standard (stochastic) gradient descent, at every iteration limiting the gradient norm to a certain value $c >0$. It is widely used for example for stabilizing the training of deep learning models (Goodfellow et al., 2016), or for enforcing differential privacy (Abadi et al., 2016). Despite popularity and simplicity of the clipping mechanism, its convergence guarantees often require specific values of $c$ and strong noise assumptions.  In this paper, we give convergence guarantees that show precise dependence on arbitrary clipping thresholds $c$ and show that our guarantees are tight with both deterministic and stochastic gradients. In particular, we show that (i) for deterministic gradient descent, the clipping threshold only affects the higher-order terms of convergence, (ii) in the stochastic setting convergence to the true optimum cannot be guaranteed under the standard noise assumption, even under arbitrary small step-sizes. We give ma
    
[^3]: 揭秘长期基于新颖性探索中表示方法的威力

    Unlocking the Power of Representations in Long-term Novelty-based Exploration. (arXiv:2305.01521v1 [cs.LG])

    [http://arxiv.org/abs/2305.01521](http://arxiv.org/abs/2305.01521)

    本论文介绍了一种名为RECODE的非参数新颖性探索方法，它在深度强化学习中跟踪状态访问计数，并与新颖的逆动力学损失相结合，实现了在具有挑战性的任务中的最新最佳表现。

    

    我们介绍了一种名为RECODE（基于聚类的在线密度估计强化学习方法）的非参数新颖性探索方法，其根据在所选择的嵌入空间中的相似性聚合状态并估计其访问次数。通过将经典聚类方法适应于深度强化学习的非平稳性环境，RECODE可在数千个回合中有效地跟踪状态访问计数。我们进一步提出了一种新颖的逆动力学损失的泛化形式，它利用掩码变压器结构进行多步预测。RECODE与此相结合，在DM-Hard-8的一系列具有挑战性的3D探索任务中实现了最新的最佳表现。在困难的Atari游戏中，RECODE也创造了新的最佳表现，并成为首个成功通关"Pitfall!"的代理。

    We introduce Robust Exploration via Clustering-based Online Density Estimation (RECODE), a non-parametric method for novelty-based exploration that estimates visitation counts for clusters of states based on their similarity in a chosen embedding space. By adapting classical clustering to the nonstationary setting of Deep RL, RECODE can efficiently track state visitation counts over thousands of episodes. We further propose a novel generalization of the inverse dynamics loss, which leverages masked transformer architectures for multi-step prediction; which in conjunction with RECODE achieves a new state-of-the-art in a suite of challenging 3D-exploration tasks in DM-Hard-8. RECODE also sets new state-of-the-art in hard exploration Atari games, and is the first agent to reach the end screen in "Pitfall!".
    
[^4]: 高斯Copula混合模型的性质研究

    On the properties of Gaussian Copula Mixture Models. (arXiv:2305.01479v1 [cs.LG])

    [http://arxiv.org/abs/2305.01479](http://arxiv.org/abs/2305.01479)

    本文研究了高斯Copula混合模型（GCMM）的性质，开发了基于扩展期望最大算法的参数估计方法，并表明GCMM相比于GMM可以更好地拟合数据并实现更深入的数据挖掘。

    

    高斯Copula混合模型（GCMM）是使用Copula概念的高斯混合模型的推广。本文给出了其数学定义，并研究了似然函数的性质。基于这些属性，我们开发了扩展期望最大算法，用于估计混合Copula的参数，而每个组件对应的边际分布则使用单独的非参数统计方法进行估计。实验表明，相比于GMM，GCMM在相同数量的聚类情况下可以实现更好的拟合；此外，GCMM可以利用每个维度上的不同步数据实现更深入的数据挖掘。

    Gaussian copula mixture models (GCMM) are the generalization of Gaussian Mixture models using the concept of copula. Its mathematical definition is given and the properties of likelihood function are studied in this paper. Based on these properties, extended Expectation Maximum algorithms are developed for estimating parameters for the mixture of copulas while marginal distributions corresponding to each component is estimated using separate nonparametric statistical methods. In the experiment, GCMM can achieve better goodness-of-fitting given the same number of clusters as GMM; furthermore, GCMM can utilize unsynchronized data on each dimension to achieve deeper mining of data.
    
[^5]: 带有基于图的上下文的随机情境赌博机问题

    Stochastic Contextual Bandits with Graph-based Contexts. (arXiv:2305.01470v1 [cs.LG])

    [http://arxiv.org/abs/2305.01470](http://arxiv.org/abs/2305.01470)

    本文提出了一种基于图上下文的随机情境赌博机问题，其中节点标签相同的节点共享相同的奖励分布。对于线图和树，我们提出了一种具有遗憾界的算法。

    

    本文将在线图预测问题自然地推广到了随机情境赌博问题的一个版本，其中上下文是图中的节点，而图的结构提供了有关上下文相似性的信息。具体而言，我们给出一个图$G = (V，E)$，其节点集$V$表示具有未知顶点标签$y$的上下文。在我们的随机情境赌博机设置中，具有相同标签的顶点共享同一奖励分布。在图标签预测中，标准的实例难度概念是割大小$f$，即有不同标签结束点的边数。对于线图和树，我们提出了一种具有遗憾界的算法$O(T^{2/3}K^{1/3}f^{1/3})$，其中$K$是手臂数量。我们的算法依赖于Zimmert和Seldin~[AISTAT'19，JMLR'21]的最优随机赌徒算法。当最佳手臂的表现优于其他手臂时，遗憾界将改善为$\tilde{O}(\sqrt{KT\cdot f})$。

    We naturally generalize the on-line graph prediction problem to a version of stochastic contextual bandit problems where contexts are vertices in a graph and the structure of the graph provides information on the similarity of contexts. More specifically, we are given a graph $G=(V,E)$, whose vertex set $V$ represents contexts with {\em unknown} vertex label $y$. In our stochastic contextual bandit setting, vertices with the same label share the same reward distribution. The standard notion of instance difficulties in graph label prediction is the cutsize $f$ defined to be the number of edges whose end points having different labels. For line graphs and trees we present an algorithm with regret bound of $\tilde{O}(T^{2/3}K^{1/3}f^{1/3})$ where $K$ is the number of arms. Our algorithm relies on the optimal stochastic bandit algorithm by Zimmert and Seldin~[AISTAT'19, JMLR'21]. When the best arm outperforms the other arms, the regret improves to $\tilde{O}(\sqrt{KT\cdot f})$. The regret 
    
[^6]: 循环神经网络的记忆：我们计算得对吗？

    Memory of recurrent networks: Do we compute it right?. (arXiv:2305.01457v1 [cs.LG])

    [http://arxiv.org/abs/2305.01457](http://arxiv.org/abs/2305.01457)

    本文研究了线性回声状态网络的记忆容量计算问题。通过发现数值评估的不准确性主要源于数值方面的问题，提出了基于掩码矩阵MC相对于中立性的稳健数值方法，该方法可以解决数值评估中的误差问题。

    

    文献中对于循环神经网络的记忆容量（MC）的数值评估常常与已经建立的理论界限相矛盾。本文研究了线性回声状态网络的情况，对应的Kalman可控矩阵的秩已被证明等于总记忆容量。我们揭示了关于记忆不准确的数值评估的各种原因，并表明这些问题是纯粹数值方面上的，往往在近期文献中被忽视。更明确地说，我们证明了当线性MC的Krylov结构被忽略时，理论MC和它的经验值之间会存在差距。解决这一问题的方法是，利用MC相对于输入掩码矩阵的中立性，开发出稳健的数值方法。模拟结果显示，我们提出的方法得到的记忆曲线与理论完全一致。

    Numerical evaluations of the memory capacity (MC) of recurrent neural networks reported in the literature often contradict well-established theoretical bounds. In this paper, we study the case of linear echo state networks, for which the total memory capacity has been proven to be equal to the rank of the corresponding Kalman controllability matrix. We shed light on various reasons for the inaccurate numerical estimations of the memory, and we show that these issues, often overlooked in the recent literature, are of an exclusively numerical nature. More explicitly, we prove that when the Krylov structure of the linear MC is ignored, a gap between the theoretical MC and its empirical counterpart is introduced. As a solution, we develop robust numerical approaches by exploiting a result of MC neutrality with respect to the input mask matrix. Simulations show that the memory curves that are recovered using the proposed methods fully agree with the theory.
    
[^7]: 时间序列外源回归的无监督特征算法

    Unsupervised Feature Based Algorithms for Time Series Extrinsic Regression. (arXiv:2305.01429v1 [cs.LG])

    [http://arxiv.org/abs/2305.01429](http://arxiv.org/abs/2305.01429)

    本研究提出了两种新的TSER算法：FreshPRINCE和DrCIF，它们分别由一组汇总特征和多个条件推理树构成，能够更好地在时间序列外源回归的问题上预测响应变量，比起以前的评估中使用的基线算法表现更佳。

    

    时间序列外源回归（TSER）涉及使用一组训练时间序列来形成一个连续响应变量的预测模型，该变量与回归器序列没有直接关系。TSER存档用于比较算法于2022年发布，包括19个问题。我们将此存档的大小增加到63个问题，并重现以前算法的基准比较。然后，我们扩展比较，包括更广泛的标准回归器和以前研究中使用的最新版本的TSER模型。我们表明，以前评估的回归器都不能胜过标准分类器旋转森林的回归适应。我们引入了两个新的TSER算法，这些算法是从时间序列分类的相关工作中开发而来。FreshPRINCE是一个管道估计器，包括转换到各种汇总特征，然后是一个旋转森林回归器。DrCIF是一个树集合，它根据时间序列的随机增量创建汇总统计信息特征，然后使用多个条件推理树来预测响应变量。在扩展的TSER问题集上，FreshPRINCE和DrCIF都始终优于基准算法。

    Time Series Extrinsic Regression (TSER) involves using a set of training time series to form a predictive model of a continuous response variable that is not directly related to the regressor series. The TSER archive for comparing algorithms was released in 2022 with 19 problems. We increase the size of this archive to 63 problems and reproduce the previous comparison of baseline algorithms. We then extend the comparison to include a wider range of standard regressors and the latest versions of TSER models used in the previous study. We show that none of the previously evaluated regressors can outperform a regression adaptation of a standard classifier, rotation forest. We introduce two new TSER algorithms developed from related work in time series classification. FreshPRINCE is a pipeline estimator consisting of a transform into a wide range of summary features followed by a rotation forest regressor. DrCIF is a tree ensemble that creates features from summary statistics over random i
    
[^8]: 医学影像中的人口统计学不变模型和表示是否公平？

    Are demographically invariant models and representations in medical imaging fair?. (arXiv:2305.01397v1 [cs.LG])

    [http://arxiv.org/abs/2305.01397](http://arxiv.org/abs/2305.01397)

    医学影像模型编码患者人口统计信息，引发有关潜在歧视的担忧。研究表明，不编码人口属性的模型容易损失预测性能，而考虑人口统计属性的反事实模型不变性存在复杂性。人口统计学编码可以被认为是优势。

    

    研究表明，医学成像模型在其潜在表示中编码了有关患者人口统计学信息（年龄、种族、性别），这引发了有关其潜在歧视的担忧。在这里，我们询问是否可行和值得训练不编码人口属性的模型。我们考虑不同类型的与人口统计学属性的不变性，即边际、类条件和反事实模型不变性，并说明它们与算法公平的标准概念的等价性。根据现有理论，我们发现边际和类条件的不变性可被认为是实现某些公平概念的过度限制方法，导致显著的预测性能损失。关于反事实模型不变性，我们注意到对于人口统计学属性，定义医学图像反事实存在复杂性。最后，我们认为人口统计学编码甚至可以被认为是优势。

    Medical imaging models have been shown to encode information about patient demographics (age, race, sex) in their latent representation, raising concerns about their potential for discrimination. Here, we ask whether it is feasible and desirable to train models that do not encode demographic attributes. We consider different types of invariance with respect to demographic attributes marginal, class-conditional, and counterfactual model invariance - and lay out their equivalence to standard notions of algorithmic fairness. Drawing on existing theory, we find that marginal and class-conditional invariance can be considered overly restrictive approaches for achieving certain fairness notions, resulting in significant predictive performance losses. Concerning counterfactual model invariance, we note that defining medical image counterfactuals with respect to demographic attributes is fraught with complexities. Finally, we posit that demographic encoding may even be considered advantageou
    
[^9]: LogSpecT: 从平稳信号中学习可行的图形学习模型并具备恢复保证

    LogSpecT: Feasible Graph Learning Model from Stationary Signals with Recovery Guarantees. (arXiv:2305.01379v1 [stat.ML])

    [http://arxiv.org/abs/2305.01379](http://arxiv.org/abs/2305.01379)

    本文提出了一种新的图形学习模型LogSpecT及其实际公式rLogSpecT，以解决现有模型rSpecT敏感超参数选择、不可行的问题。本文提供了rLogSpecT的恢复保证，并提出了基于L-ADMM的高效算法。

    

    信号图形学习是图形信号处理（GSP）中的核心任务。学习平稳信号图形最常用的模型之一是SpecT。然而，它的实际公式rSpecT被认为对超参数选择敏感，并且更糟的是，容易无法实现。在本文中，我们首次给出保证rSpecT无法实现的条件，并设计了一种新模型（LogSpecT）及其实际公式（rLogSpecT）来解决这个问题。与rSpecT不同，新的实用模型rLogSpecT始终是可行的。此外，我们还提供了rLogSpecT的恢复保证，这些保证来自于与epi-converg​​ence相关的现代优化工具。这些工具对于各种学习问题都具有独立的利益和重要性。为了展示rLogSpecT在实践中的优点，我们提出了一种基于线性化交替方向乘子方法（L-ADMM）的高效算法。L-ADMM的子问题

    Graph learning from signals is a core task in Graph Signal Processing (GSP). One of the most commonly used models to learn graphs from stationary signals is SpecT. However, its practical formulation rSpecT is known to be sensitive to hyperparameter selection and, even worse, to suffer from infeasibility. In this paper, we give the first condition that guarantees the infeasibility of rSpecT and design a novel model (LogSpecT) and its practical formulation (rLogSpecT) to overcome this issue. Contrary to rSpecT, the novel practical model rLogSpecT is always feasible. Furthermore, we provide recovery guarantees of rLogSpecT, which are derived from modern optimization tools related to epi-convergence. These tools could be of independent interest and significant for various learning problems. To demonstrate the advantages of rLogSpecT in practice, a highly efficient algorithm based on the linearized alternating direction method of multipliers (L-ADMM) is proposed. The subproblems of L-ADMM a
    
[^10]: 随机函数下降法

    Random Function Descent. (arXiv:2305.01377v1 [math.OC])

    [http://arxiv.org/abs/2305.01377](http://arxiv.org/abs/2305.01377)

    本文提出了随机函数下降(RFD)算法，可以在随机环境中计算出步长并且与贝叶斯优化中的梯度下降算法相同。在合成基准测试中，RFD算法比未调整的Adam方法表现更好，提出的heuristic扩展可与调整后的Adam方法相媲美。

    

    虽然梯度下降方法在机器学习中十分常见，但是选择正确的步长经常需要进行“超参数调整”。这是因为回溯程序如Armijo's准则依赖于每个步骤中的质量评估，而这些评估在随机情况下不可用。由于优化方案可以用Taylor逼近来解释，我们将Taylor逼近替换为条件期望（最佳的$L^2$估计），提出了“随机函数下降”（RFD）。 在Bayesian优化中常见的一些轻微假设的情况下，我们证明了RFD与梯度下降算法是相同的，但是在随机情况下具有可计算的步长。我们在合成基准测试中比未调整的Adam方法表现更好。为了缩小与调整后的Adam算法之间的性能差距，我们提出了一种启发式扩展，可与调整后的Adam方法相媲美。

    While gradient based methods are ubiquitous in machine learning, selecting the right step size often requires "hyperparameter tuning". This is because backtracking procedures like Armijo's rule depend on quality evaluations in every step, which are not available in a stochastic context. Since optimization schemes can be motivated using Taylor approximations, we replace the Taylor approximation with the conditional expectation (the best $L^2$ estimator) and propose "Random Function Descent" (RFD). Under light assumptions common in Bayesian optimization, we prove that RFD is identical to gradient descent, but with calculable step sizes, even in a stochastic context. We beat untuned Adam in synthetic benchmarks. To close the performance gap to tuned Adam, we propose a heuristic extension competitive with tuned Adam.
    
[^11]: 通过聚合解决无监督领域适应中的参数选择问题

    Addressing Parameter Choice Issues in Unsupervised Domain Adaptation by Aggregation. (arXiv:2305.01281v1 [stat.ML])

    [http://arxiv.org/abs/2305.01281](http://arxiv.org/abs/2305.01281)

    本文提出了一种使用线性聚合的方法来解决无监督领域适应中的参数选择问题，并且展示了该算法的目标误差渐近不劣于未知最佳聚合的两倍，大规模实证研究表明该方法优于深度嵌入验证。

    

    本文研究了在无监督领域适应中选择算法超参数的问题，即在源域中有标记数据，在目标域中有来自不同输入分布的未标记数据。我们采用计算使用不同超参数的几个模型的策略，然后计算模型的线性聚合。虽然存在几个遵循这种策略的启发式方法，但是仍然缺少依赖于限制目标误差的彻底理论的方法。因此，我们提出了一种方法，将加权最小二乘法扩展到向量值函数（例如深度神经网络）。我们展示了所提出算法的目标误差渐近不劣于未知最佳聚合的两倍。我们还进行了大规模实证比较研究，包括文本、图像、脑电图、身体传感器信号和手机信号等多个数据集。我们的方法优于深度嵌入验证。

    We study the problem of choosing algorithm hyper-parameters in unsupervised domain adaptation, i.e., with labeled data in a source domain and unlabeled data in a target domain, drawn from a different input distribution. We follow the strategy to compute several models using different hyper-parameters, and, to subsequently compute a linear aggregation of the models. While several heuristics exist that follow this strategy, methods are still missing that rely on thorough theories for bounding the target error. In this turn, we propose a method that extends weighted least squares to vector-valued functions, e.g., deep neural networks. We show that the target error of the proposed algorithm is asymptotically not worse than twice the error of the unknown optimal aggregation. We also perform a large scale empirical comparative study on several datasets, including text, images, electroencephalogram, body sensor signals and signals from mobile phones. Our method outperforms deep embedded valid
    
[^12]: 无界差分隐私分位数和最大值估算

    Unbounded Differentially Private Quantile and Maximum Estimation. (arXiv:2305.01177v1 [cs.DS])

    [http://arxiv.org/abs/2305.01177](http://arxiv.org/abs/2305.01177)

    本文研究了如何对无上限数据进行差分隐私分位数和最大值的计算。调用基本的稀疏向量技术中的$\texttt{AboveThreshold}$子程序可以实现这个目标，可以提供更准确和稳健的最高分位数估计，从而应用于对于差分隐私求和和均值估计至关重要的数据剪切，该技术的隐私保障可以通过方法改进。

    

    本文考虑在数据的分位数计算中，尤其是最高分位数（如最大值），如何实现对无界数据的差分隐私计算。我们通过简单调用迭代调用基本的稀疏向量技术中的$\texttt{AboveThreshold}$子程序，即可高效地实现此目标，即使数据没有上限。特别地，我们展示出此过程可提供更准确和稳健的最高分位数估计，从而应用于对于差分隐私求和和均值估计至关重要的数据剪切。此外，我们展示了两个调用如何处理完全无界的数据设置。在我们的研究中，我们展示了改进$\texttt{AboveThreshold}$的分析方法可以提高广泛使用的稀疏向量技术的隐私保障（这是独立于本文的内容）。我们给出了更普遍的$\texttt{AboveThreshold}$隐私损失特征，并展示了差分隐私的标准组合规则可能会高估总体隐私损失。

    In this work we consider the problem of differentially private computation of quantiles for the data, especially the highest quantiles such as maximum, but with an unbounded range for the dataset. We show that this can be done efficiently through a simple invocation of $\texttt{AboveThreshold}$, a subroutine that is iteratively called in the fundamental Sparse Vector Technique, even when there is no upper bound on the data. In particular, we show that this procedure can give more accurate and robust estimates on the highest quantiles with applications towards clipping that is essential for differentially private sum and mean estimation. In addition, we show how two invocations can handle the fully unbounded data setting. Within our study, we show that an improved analysis of $\texttt{AboveThreshold}$ can improve the privacy guarantees for the widely used Sparse Vector Technique that is of independent interest. We give a more general characterization of privacy loss for $\texttt{AboveTh
    
[^13]: 深度学习算法的泛化能力理解：基于核化Renyi熵的视角

    Understanding the Generalization Ability of Deep Learning Algorithms: A Kernelized Renyi's Entropy Perspective. (arXiv:2305.01143v1 [stat.ML])

    [http://arxiv.org/abs/2305.01143](http://arxiv.org/abs/2305.01143)

    本论文提出了一种新的信息理论度量方法——基于核化Renyi熵，用于在不假设Lipschitz或凸性条件的前提下对SGD / SGLD等下降学习算法进行分析，旨在提高当前泛化误差界限的优化水平。

    

    最近，信息理论分析已成为理解深度神经网络泛化行为的流行框架。它允许对随机梯度/ Langevin下降（SGD / SGLD）学习算法进行直接分析，而无需诸如Lipschitz或凸性条件等强假设。然而，在这个框架内的当前泛化误差界限仍然远非最优，而对这些界限的实质性改进由于高维信息量的不可计算性而相当具有挑战性。为解决这个问题，我们首先提出了一种新的信息理。论衡量：基于核化Renyi熵，利用希尔伯特空间中的算子表示。它继承了香农熵的属性，可通过简单的随机抽样进行有效计算，同时保持独立于输入维度。然后，我们在核化Renyi熵下建立了SGD / SGLD的泛化误差界限，其中相互信息...

    Recently, information theoretic analysis has become a popular framework for understanding the generalization behavior of deep neural networks. It allows a direct analysis for stochastic gradient/Langevin descent (SGD/SGLD) learning algorithms without strong assumptions such as Lipschitz or convexity conditions. However, the current generalization error bounds within this framework are still far from optimal, while substantial improvements on these bounds are quite challenging due to the intractability of high-dimensional information quantities. To address this issue, we first propose a novel information theoretical measure: kernelized Renyi's entropy, by utilizing operator representation in Hilbert space. It inherits the properties of Shannon's entropy and can be effectively calculated via simple random sampling, while remaining independent of the input dimension. We then establish the generalization error bounds for SGD/SGLD under kernelized Renyi's entropy, where the mutual informati
    
[^14]: 通过重新参数化学习实现在线反馈的实现式预测

    Performative Prediction with Bandit Feedback: Learning through Reparameterization. (arXiv:2305.01094v1 [cs.LG])

    [http://arxiv.org/abs/2305.01094](http://arxiv.org/abs/2305.01094)

    本文提出一种新的在线反馈的实现式预测框架，解决了在模型部署自身改变数据分布的情况下优化准确性的问题。

    

    本文提出了在数据分布由模型部署自身改变的情形下预测的一个框架——实现式预测。现有研究的重点在于优化准确性，但是其假设往往难以在实践中得到满足。本文针对这类问题，提出了一种两层零阶优化算法，通过重新参数化实现式预测目标，从而将非凸的目标转化为凸的目标。

    Performative prediction, as introduced by Perdomo et al. (2020), is a framework for studying social prediction in which the data distribution itself changes in response to the deployment of a model. Existing work on optimizing accuracy in this setting hinges on two assumptions that are easily violated in practice: that the performative risk is convex over the deployed model, and that the mapping from the model to the data distribution is known to the model designer in advance. In this paper, we initiate the study of tractable performative prediction problems that do not require these assumptions. To tackle this more challenging setting, we develop a two-level zeroth-order optimization algorithm, where one level aims to compute the distribution map, and the other level reparameterizes the performative prediction objective as a function of the induced data distribution. Under mild conditions, this reparameterization allows us to transform the non-convex objective into a convex one and ac
    
[^15]: 无特定模型泛化难度度量

    Model-agnostic Measure of Generalization Difficulty. (arXiv:2305.01034v1 [cs.LG])

    [http://arxiv.org/abs/2305.01034](http://arxiv.org/abs/2305.01034)

    该论文提出了第一个无特定模型的、量化机器学习测试泛化难度的方法——归纳偏差复杂度度量。该方法量化了在任务上良好泛化所需的总信息量与数据提供的信息量之差，通常需要在许多维度上泛化的任务比涉及更少维度但要求更多细节的任务要困难得多。

    

    机器学习算法的度量是其可以执行的任务难度，足够困难的任务是强大机器学习模型的关键驱动因素。然而，量化机器学习测试的泛化难度一直是具有挑战性的。我们提出了据我们所知的第一个对任务固有泛化难度的无特定模型的度量。我们的归纳偏差复杂度度量量化了在任务上良好泛化所需的总信息量与数据提供的信息量之差。通过测量适合训练数据的假设在任务中泛化的分数占据的容积，来实现这一点。它与模型必须泛化的空间的内在维数成指数比例，但仅在每个维度的分辨率上呈多项式比例，表明需要在许多维度上泛化的任务比涉及更少维度的更多细节的任务要困难得多。

    The measure of a machine learning algorithm is the difficulty of the tasks it can perform, and sufficiently difficult tasks are critical drivers of strong machine learning models. However, quantifying the generalization difficulty of machine learning benchmarks has remained challenging. We propose what is to our knowledge the first model-agnostic measure of the inherent generalization difficulty of tasks. Our inductive bias complexity measure quantifies the total information required to generalize well on a task minus the information provided by the data. It does so by measuring the fractional volume occupied by hypotheses that generalize on a task given that they fit the training data. It scales exponentially with the intrinsic dimensionality of the space over which the model must generalize but only polynomially in resolution per dimension, showing that tasks which require generalizing over many dimensions are drastically more difficult than tasks involving more detail in fewer dimen
    
[^16]: 高斯混合块模型中的谱聚类

    Spectral clustering in the Gaussian mixture block model. (arXiv:2305.00979v1 [stat.ML])

    [http://arxiv.org/abs/2305.00979](http://arxiv.org/abs/2305.00979)

    本文首次研究了从高维高斯混合块模型中抽样的图聚类和嵌入问题。

    

    高斯混合块模型是用于模拟现代网络的图分布：对于这样的模型生成一个图，我们将每个顶点 $i$ 与一个从高斯混合中抽样到的潜在特征向量 $u_i \in \mathbb{R}^d$ 相关联，当且仅当特征向量足够相似，即 $\langle u_i,u_j \rangle \ge \tau$ 时，我们才会添加边 $(i,j)$。高斯混合的不同组成部分表示可能具有不同特征分布的不同类型的节点，例如在社交网络中，每个组成部分都表示独特社区的不同属性。这些网络涉及到的自然算法任务有嵌入（恢复潜在的特征向量）和聚类（通过其混合组分将节点分组）。本文开启了对从高维高斯混合块模型抽样的图进行聚类和嵌入研究。

    Gaussian mixture block models are distributions over graphs that strive to model modern networks: to generate a graph from such a model, we associate each vertex $i$ with a latent feature vector $u_i \in \mathbb{R}^d$ sampled from a mixture of Gaussians, and we add edge $(i,j)$ if and only if the feature vectors are sufficiently similar, in that $\langle u_i,u_j \rangle \ge \tau$ for a pre-specified threshold $\tau$. The different components of the Gaussian mixture represent the fact that there may be different types of nodes with different distributions over features -- for example, in a social network each component represents the different attributes of a distinct community. Natural algorithmic tasks associated with these networks are embedding (recovering the latent feature vectors) and clustering (grouping nodes by their mixture component).  In this paper we initiate the study of clustering and embedding graphs sampled from high-dimensional Gaussian mixture block models, where the
    
[^17]: ContraNorm: 对于过度平滑的对比学习视角和更多的研究

    ContraNorm: A Contrastive Learning Perspective on Oversmoothing and Beyond. (arXiv:2303.06562v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.06562](http://arxiv.org/abs/2303.06562)

    本研究提出了一种新的规范化层——ContraNorm，针对图神经网络和变压器中的过度平滑问题，通过对比学习的方式在嵌入空间中破坏表示，缓解了完全塌陷和维度塌陷的现象，并在实验中表现出较高的精度。

    

    过度平滑现象在各种图神经网络和变压器中普遍存在，当层数增加时，其性能会变差。我们从维度折叠的一个更一般的视角来描述过度平滑的现象，表示会聚到一个狭窄的锥形空间中，而不是表示会聚到一个点上。受到对抗性学习在防止维度折叠方面的有效性启发，我们提出了一种新的规范化层——ContraNorm。直观上，ContraNorm会在嵌入空间中隐式破坏表示，导致更均匀的分布和轻微的维度折叠。在理论分析中，我们证明了在某些条件下，ContraNorm可以缓解完全塌陷和维度塌陷的情况。我们提出的规范化层可以轻松地集成到GNNs和Transformers中，且参数开销很小。实验结果表明，我们的提议可以提高GNNs和Transformers的精度。

    Oversmoothing is a common phenomenon in a wide range of Graph Neural Networks (GNNs) and Transformers, where performance worsens as the number of layers increases. Instead of characterizing oversmoothing from the view of complete collapse in which representations converge to a single point, we dive into a more general perspective of dimensional collapse in which representations lie in a narrow cone. Accordingly, inspired by the effectiveness of contrastive learning in preventing dimensional collapse, we propose a novel normalization layer called ContraNorm. Intuitively, ContraNorm implicitly shatters representations in the embedding space, leading to a more uniform distribution and a slighter dimensional collapse. On the theoretical analysis, we prove that ContraNorm can alleviate both complete collapse and dimensional collapse under certain conditions. Our proposed normalization layer can be easily integrated into GNNs and Transformers with negligible parameter overhead. Experiments o
    
[^18]: 探索基于数值先验的广义CP分解低秩张量补全算法

    Exploring Numerical Priors for Low-Rank Tensor Completion with Generalized CP Decomposition. (arXiv:2302.05881v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.05881](http://arxiv.org/abs/2302.05881)

    本文提出了一种新的方法框架GCDTC，利用数值先验和广义CP分解实现了更高的低秩张量补全精度；同时介绍了一个算法SPTC，作为该框架的一个实现。在实验中，该方法表现出比现有技术更好的性能。

    

    张量补全在计算机视觉、数据分析和信号处理等领域中具有重要意义。最近，低秩张量补全这一类别的方法得到了广泛研究，对补全张量施加低秩结构。虽然这些方法取得了巨大成功，但尚未考虑到张量元素的数值先验信息。忽略数值先验将导致丢失关于数据的重要信息，因此阻止算法达到最优精度。本研究试图构建一个新的方法框架，名为GCDTC（广义CP分解张量补全），以利用数值先验并实现更高的张量补全精度。在这个新引入的框架中，将广义的CP分解应用于低秩张量补全。本文还提出了一种名为SPTC（平滑泊松张量补全）的算法，用于非负整数张量补全，作为GCDTC框架的一个实现。通过对合成和真实世界数据集的大量实验，证明所提出的方法相比于现有技术具有更优的张量补全性能。

    Tensor completion is important to many areas such as computer vision, data analysis, and signal processing. Enforcing low-rank structures on completed tensors, a category of methods known as low-rank tensor completion has recently been studied extensively. While such methods attained great success, none considered exploiting numerical priors of tensor elements. Ignoring numerical priors causes loss of important information regarding the data, and therefore prevents the algorithms from reaching optimal accuracy. This work attempts to construct a new methodological framework called GCDTC (Generalized CP Decomposition Tensor Completion) for leveraging numerical priors and achieving higher accuracy in tensor completion. In this newly introduced framework, a generalized form of CP Decomposition is applied to low-rank tensor completion. This paper also proposes an algorithm known as SPTC (Smooth Poisson Tensor Completion) for nonnegative integer tensor completion as an instantiation of the G
    
[^19]: 训练神经网络用于时序变点检测

    Training Neural Networks for Sequential Change-point Detection. (arXiv:2210.17312v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.17312](http://arxiv.org/abs/2210.17312)

    本文介绍了一种使用神经网络进行在线变点检测的方法，通过训练神经网络逐步计算检测统计量的累积和来检测变点，并在合成和真实数据上证明了该方法的优越性和潜力。

    

    检测数据流中的突变分布转换，即所谓的变点检测，是统计学和机器学习中的一个基本问题。我们引入了一种新颖的方法，使用神经网络进行在线变点检测。具体而言，我们的方法是训练神经网络来逐步计算检测统计量的累积和，当发生变点时，该量会显著变化。我们使用合成和真实世界数据证明了所提出的方法在检测变点方面的优越性和潜力。

    Detecting an abrupt distributional shift of a data stream, known as change-point detection, is a fundamental problem in statistics and machine learning. We introduce a novel approach for online change-point detection using neural networks. To be specific, our approach is training neural networks to compute the cumulative sum of a detection statistic sequentially, which exhibits a significant change when a change-point occurs. We demonstrated the superiority and potential of the proposed method in detecting change-point using both synthetic and real-world data.
    
[^20]: Transformers学会了自动机的快捷方式

    Transformers Learn Shortcuts to Automata. (arXiv:2210.10749v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.10749](http://arxiv.org/abs/2210.10749)

    Transformer模型通过重新参数化其循环动态，可以使用比推理步骤更少的层数执行任何有限状态自动机的计算。多项式大小的 $O(\log T)$ 深度解决方案始终存在，而且$O(1)$深度模拟器是非常普遍的。

    

    算法推理需要计算模型的循环能力，如图灵机等。然而，Transformer模型虽然缺乏循环能力，但能够使用比推理步骤更少的层数执行此类推理。这引发了一个问题：这些浅层次和非循环模型学到了什么解决方案？我们发现，低深度Transformer可以通过逐层重新参数化其循环动态，表示任何有限状态自动机（因此，任何有界内存算法）的计算。我们的理论结果表征了快捷解决方案，其中具有 $o(T)$ 层的Transformer可以精确复制自动机在长度为 $T$ 的输入序列上的计算。我们发现，多项式大小的 $O(\log T)$ 深度解决方案始终存在；此外，$O(1)$ 深度模拟器非常普遍，可以使用从 Krohn-Rhodes 理论和电路复杂度理论中的工具来理解。实证

    Algorithmic reasoning requires capabilities which are most naturally understood through recurrent models of computation, like the Turing machine. However, Transformer models, while lacking recurrence, are able to perform such reasoning using far fewer layers than the number of reasoning steps. This raises the question: what solutions are learned by these shallow and non-recurrent models? We find that a low-depth Transformer can represent the computations of any finite-state automaton (thus, any bounded-memory algorithm), by hierarchically reparameterizing its recurrent dynamics. Our theoretical results characterize shortcut solutions, whereby a Transformer with $o(T)$ layers can exactly replicate the computation of an automaton on an input sequence of length $T$. We find that polynomial-sized $O(\log T)$-depth solutions always exist; furthermore, $O(1)$-depth simulators are surprisingly common, and can be understood using tools from Krohn-Rhodes theory and circuit complexity. Empirical
    
[^21]: 混合数据的条件特征重要性

    Conditional Feature Importance for Mixed Data. (arXiv:2210.03047v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.03047](http://arxiv.org/abs/2210.03047)

    本研究提出了一种针对混合数据的条件特征重要性框架，使用条件预测影响和顺序knockoff抽样结合，以解决很少讨论的条件和边缘度量之间的重要区别，并揭示出为测试条件FI，目前只有少数方法可用且过去从业者由于数据要求不匹配而受到严重限制。

    

    尽管特征重要性（FI）在可解释的机器学习中很受欢迎，但这些方法的统计充分性很少被讨论。从统计角度看，一个主要区别是在调整协变量之前和之后分析变量的重要性，即“边缘”和“条件”度量之间。我们的工作引起了这种很少被承认但至关重要的区别的注意，并展示了其影响。此外，我们揭示了测试条件FI时只有少数方法可用，而从业者过去由于数据要求不匹配而受到严重限制。大多数现实世界的数据都表现出复杂的特征依赖性，并包含连续和分类数据（混合数据）。这些属性通常被条件FI度量所忽略。为了填补这一空白，我们提出将条件预测影响（CPI）框架与顺序knockoff抽样相结合。

    Despite the popularity of feature importance (FI) measures in interpretable machine learning, the statistical adequacy of these methods is rarely discussed. From a statistical perspective, a major distinction is between analyzing a variable's importance before and after adjusting for covariates i.e., between $\textit{marginal}$ and $\textit{conditional}$ measures. Our work draws attention to this rarely acknowledged, yet crucial distinction and showcases its implications. Further, we reveal that for testing conditional FI, only few methods are available and practitioners have hitherto been severely restricted in method application due to mismatching data requirements. Most real-world data exhibits complex feature dependencies and incorporates both continuous and categorical data (mixed data). Both properties are oftentimes neglected by conditional FI measures. To fill this gap, we propose to combine the conditional predictive impact (CPI) framework with sequential knockoff sampling. 
    
[^22]: 基于物理学的领域自适应框架用于建筑能量系统建模和预测

    A physics-based domain adaptation framework for modelling and forecasting building energy systems. (arXiv:2208.09456v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.09456](http://arxiv.org/abs/2208.09456)

    本文提出了一种基于物理学的领域自适应框架，将线性时不变状态空间模型与基于子空间的无监督降阶建模相结合，通过最小化模型在共享子空间中的表示差异，将在一个领域上训练过的模型适应到另一个领域，用于建筑能量系统的建模和预测，并在实验中表现出优异的预测性能。

    

    现代的基于机器学习的模型在建筑能源行为建模和预测方面已经成为一个流行的选择。然而，它们的结构通常并不具有与物理现象相关的机械结构相对应。因此，它们成功地泛化为未观测到的时间步取决于所观察到的系统动态在数据中表现的代表性，在数字孪生控制和能源管理等真实世界的工程问题中很难得到保证。为了解决这一问题，本文提出了一个框架，将线性时不变（LTI）状态空间模型（SSM）的成批参数模型与基于子空间的无监督降阶建模相结合，形成了一个子空间导向的领域适应（SDA）框架。SDA是一种转移学习方法，旨在通过最小化共享子空间中的模型表示的差异来将在一个领域上训练过的模型适应到另一个领域。在一个建筑能量系统案例研究中展示了我们提出的基于物理学的领域自适应框架，证明了它在短期和长期能量预测方面优于现有的基于机器学习的模型。

    State-of-the-art machine-learning-based models are a popular choice for modeling and forecasting energy behavior in buildings because given enough data, they are good at finding spatiotemporal patterns and structures even in scenarios where the complexity prohibits analytical descriptions. However, their architecture typically does not hold physical correspondence to mechanistic structures linked with governing physical phenomena. As a result, their ability to successfully generalize for unobserved timesteps depends on the representativeness of the dynamics underlying the observed system in the data, which is difficult to guarantee in real-world engineering problems such as control and energy management in digital twins. In response, we present a framework that combines lumped-parameter models in the form of linear time-invariant (LTI) state-space models (SSMs) with unsupervised reduced-order modeling in a subspace-based domain adaptation (SDA) framework. SDA is a type of transfer-lear
    
[^23]: 基于Boosting的离线策略学习算法

    Boosted Off-Policy Learning. (arXiv:2208.01148v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.01148](http://arxiv.org/abs/2208.01148)

    我们提出了一种基于Boosting的离线策略学习算法，将基础学习器简化为监督学习，获得了广泛的实际效益；实验结果表明其应用能力优于深度神经网络的离线策略学习和简单回归方法。

    

    我们提出了一种针对来自记录式赌博反馈的离线策略学习的Boosting算法。与现有的监督学习的Boosting方法不同，我们的算法直接优化了策略预期收益的估计。我们对该算法进行了分析，并证明如果基础学习器满足“弱”学习条件，那么每一轮Boosting都会减小过量经验风险（可能是指数级）。我们进一步展示了如何将基础学习器简化为监督学习，从而打开了广泛的基础学习器源，如决策树等，具有实际益处。实验结果表明，我们的算法继承了许多基于决策树的Boosting算法的优良性质（例如对特征缩放和超参数调整的鲁棒性），并且可以胜过基于深度神经网络的离线策略学习和只是回归观察到的奖励的方法。

    We propose the first boosting algorithm for off-policy learning from logged bandit feedback. Unlike existing boosting methods for supervised learning, our algorithm directly optimizes an estimate of the policy's expected reward. We analyze this algorithm and prove that the excess empirical risk decreases (possibly exponentially fast) with each round of boosting, provided a ''weak'' learning condition is satisfied by the base learner. We further show how to reduce the base learner to supervised learning, which opens up a broad range of readily available base learners with practical benefits, such as decision trees. Experiments indicate that our algorithm inherits many desirable properties of tree-based boosting algorithms (e.g., robustness to feature scaling and hyperparameter tuning), and that it can outperform off-policy learning with deep neural networks as well as methods that simply regress on the observed rewards.
    
[^24]: LogGENE: 一种用于深度医疗推理任务的平滑检查损失替代方法

    LogGENE: A smooth alternative to check loss for Deep Healthcare Inference Tasks. (arXiv:2206.09333v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.09333](http://arxiv.org/abs/2206.09333)

    LogGENE采用分位数回归框架预测基因表达水平的完整条件分位数，从而为高通量基因组学提供了一种能提供解释和报告不确定性估计、鲁棒性强的推断方法。

    

    在可靠的深度学习中，挖掘大型数据集并从中获得校准的预测具有即时相关性和实用性。本研究开发了基于深度神经网络的推断方法，适用于基因表达等大型数据集。然而，与典型的深度学习方法不同的是，我们的推断技术在准确性方面实现了最先进的性能，同时还能提供解释和报告不确定性估计。我们采用分位数回归框架来预测给定一组基因表达的完整条件分位数。条件分位数除了有助于提供预测的丰富解释外，还能够抵抗测量噪声。我们的技术在高通量基因组学中特别重要，这是一个正在引领个性化医疗、靶向药物设计和传递的新时代。然而，用于驱动估计过程的检查损失，在分位数回归中并无不同之处。

    Mining large datasets and obtaining calibrated predictions from tem is of immediate relevance and utility in reliable deep learning. In our work, we develop methods for Deep neural networks based inferences in such datasets like the Gene Expression. However, unlike typical Deep learning methods, our inferential technique, while achieving state-of-the-art performance in terms of accuracy, can also provide explanations, and report uncertainty estimates. We adopt the Quantile Regression framework to predict full conditional quantiles for a given set of housekeeping gene expressions. Conditional quantiles, in addition to being useful in providing rich interpretations of the predictions, are also robust to measurement noise. Our technique is particularly consequential in High-throughput Genomics, an area which is ushering a new era in personalized health care, and targeted drug design and delivery. However, check loss, used in quantile regression to drive the estimation process is not diffe
    
[^25]: 通过低保真模型和物理知识驱动高斯过程学习数字孪生之间的物理关系

    Learning Physics between Digital Twins with Low-Fidelity Models and Physics-Informed Gaussian Processes. (arXiv:2206.08201v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2206.08201](http://arxiv.org/abs/2206.08201)

    本文提出了一种新方法，通过低保真模型和物理知识驱动高斯过程进行数字孪生之间的学习， 并开发了贝叶斯分层建模框架允许多个数字孪生之间共享信息。

    

    数字孪生是一种代表个体的计算机模型，例如组件、患者或过程。 在许多情况下，我们想从数据中获取有关个体的知识，同时结合不完美的物理知识，并从其他个体的数据中学习。 本文介绍了一种全贝叶斯方法，用于在每个个体的物理参数引起兴趣的情况下学习数字孪生之间的关系。 在个性化模型的模型公式中引入了模型差异项，以解释低保真模型中缺失的物理现象。 为了允许个体之间共享信息，我们介绍了一个贝叶斯分层建模框架，其中通过新层将个体模型连接起来。 我们的方法在两个案例研究中进行了演示，一个是先前在文献中使用的玩具示例，扩展到更多个体的情况，一个是与治疗高血压相关的心血管模型。

    A digital twin is a computer model that represents an individual, for example, a component, a patient or a process. In many situations, we want to gain knowledge about an individual from its data while incorporating imperfect physical knowledge and also learn from data from other individuals. In this paper, we introduce a fully Bayesian methodology for learning between digital twins in a setting where the physical parameters of each individual are of interest. A model discrepancy term is incorporated in the model formulation of each personalized model to account for the missing physics of the low-fidelity model. To allow sharing of information between individuals, we introduce a Bayesian Hierarchical modelling framework where the individual models are connected through a new level in the hierarchy. Our methodology is demonstrated in two case studies, a toy example previously used in the literature extended to more individuals and a cardiovascular model relevant for the treatment of Hyp
    
[^26]: 在不稳健样本上施加更多正则化以提高对抗性鲁棒性

    Improving adversarial robustness by putting more regularizations on less robust samples. (arXiv:2206.03353v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2206.03353](http://arxiv.org/abs/2206.03353)

    本文提出了一种新的对抗训练算法，通过在容易受到对抗攻击的数据上施加更多正则化以提高对抗性鲁棒性，得到了在准确性和鲁棒性方面均为最优的表现。

    

    对抗性训练是提高对抗攻击鲁棒性的一种方法，在人类视觉无法察觉的数据扰动下，使给定的深度神经网络产生误判。本文提出了一种新的对抗训练算法，它在理论上得到很好的证明，并且在实践中表现优于其他现有的算法。该算法的一个新的特点是：对于容易受到对抗攻击的数据，比其他现有的正则化算法更多地应用正则化。理论上，我们证明了我们的算法可以被理解为一个最小化经验风险的正则化算法，它来自一个新的鲁棒风险上界的动机。数值实验表明，我们提出的算法同时提高了泛化性能(在例子上的准确性)和鲁棒性(在对抗攻击上的准确性)，达到了最先进的性能水平。

    Adversarial training, which is to enhance robustness against adversarial attacks, has received much attention because it is easy to generate human-imperceptible perturbations of data to deceive a given deep neural network. In this paper, we propose a new adversarial training algorithm that is theoretically well motivated and empirically superior to other existing algorithms. A novel feature of the proposed algorithm is to apply more regularization to data vulnerable to adversarial attacks than other existing regularization algorithms do. Theoretically, we show that our algorithm can be understood as an algorithm of minimizing the regularized empirical risk motivated from a newly derived upper bound of the robust risk. Numerical experiments illustrate that our proposed algorithm improves the generalization (accuracy on examples) and robustness (accuracy on adversarial attacks) simultaneously to achieve the state-of-the-art performance.
    
[^27]: 贝叶斯模型选择、边际似然和泛化

    Bayesian Model Selection, the Marginal Likelihood, and Generalization. (arXiv:2202.11678v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.11678](http://arxiv.org/abs/2202.11678)

    本文回顾和探讨了边缘似然在构造约束和假设测试方面的有用性，强调了使用边缘似然作为泛化的代理的问题，并展示了其如何与神经架构搜索相关，可能导致超参数学习中的欠拟合和过拟合。

    

    如何比较与观测完全一致的假设之间的区别？边际似然（亦称为贝叶斯证据）作为生成由先验得到观测结果的概率，为解决这个问题提供了一个独特的方法，自动编码奥卡姆剃刀原理。尽管已经观察到边际似然可能过拟合并且对先验假设很敏感，但其在超参数学习和离散模型比较方面的局限性尚未得到彻底研究。本文首先重温了边际似然的吸引人的特点，包括学习约束和假设测试。然后，我们强调了使用边际似然作为泛化的代理存在的概念和实际问题。我们展示了边际似然如何与泛化呈负相关，并对神经架构搜索产生影响，并且在超参数学习中可能导致欠拟合和过拟合。

    How do we compare between hypotheses that are entirely consistent with observations? The marginal likelihood (aka Bayesian evidence), which represents the probability of generating our observations from a prior, provides a distinctive approach to this foundational question, automatically encoding Occam's razor. Although it has been observed that the marginal likelihood can overfit and is sensitive to prior assumptions, its limitations for hyperparameter learning and discrete model comparison have not been thoroughly investigated. We first revisit the appealing properties of the marginal likelihood for learning constraints and hypothesis testing. We then highlight the conceptual and practical issues in using the marginal likelihood as a proxy for generalization. Namely, we show how marginal likelihood can be negatively correlated with generalization, with implications for neural architecture search, and can lead to both underfitting and overfitting in hyperparameter learning. We also re
    
[^28]: CD-ROM：补充深度减少阶模型

    CD-ROM: Complemented Deep-Reduced Order Model. (arXiv:2202.10746v4 [physics.flu-dyn] UPDATED)

    [http://arxiv.org/abs/2202.10746](http://arxiv.org/abs/2202.10746)

    本文介绍了一种基于深度学习的闭合建模方法CD-ROM，用于经典的POD-Galerkin降阶模型，该方法可以显著提高降阶模型的准确性和稳定性。

    

    通过POD-Galerkin方法进行模型阶数约简可以极大地提高求解物理问题的计算效率。然而，该方法在处理非线性高维动力系统如Navier-Stokes方程方面的适用性受到限制，产生不准确且有时不稳定的模型。本文提出了一种基于深度学习的闭合建模方法，用于经典的POD-Galerkin降阶模型(ROM)。所提出的方法在理论上是有基础的，使用神经网络来逼近研究得当的运算符。与大多数以前的工作相比，本文中的CD-ROM方法是基于可解释的连续记忆形式，由关于部分观测到的动力系统行为的简单假设导出。因此，修正后的模型可以使用大多数经典的时间步进模式进行模拟。CD-ROM方法的能力在计算流体力学的两个经典案例中得到了证明，表明它可以显著提高降阶模型的准确性和稳定性。

    Model order reduction through the POD-Galerkin method can lead to dramatic gains in terms of computational efficiency in solving physical problems. However, the applicability of the method to non linear high-dimensional dynamical systems such as the Navier-Stokes equations has been shown to be limited, producing inaccurate and sometimes unstable models. This paper proposes a deep learning based closure modeling approach for classical POD-Galerkin reduced order models (ROM). The proposed approach is theoretically grounded, using neural networks to approximate well studied operators. In contrast with most previous works, the present CD-ROM approach is based on an interpretable continuous memory formulation, derived from simple hypotheses on the behavior of partially observed dynamical systems. The final corrected models can hence be simulated using most classical time stepping schemes. The capabilities of the CD-ROM approach are demonstrated on two classical examples from Computational F
    
[^29]: 非凸学习中TUSLA算法的非渐进估计及其在ReLU神经网络中的应用

    Non-asymptotic estimates for TUSLA algorithm for non-convex learning with applications to neural networks with ReLU activation function. (arXiv:2107.08649v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2107.08649](http://arxiv.org/abs/2107.08649)

    本文研究了非凸随机优化问题，提出了TUSLA算法在Wasserstein-1和Wasserstein-2距离上的非渐进误差界限，进而推导了期望过量风险的非渐进估计值。在ReLU神经网络中，理论和数值实验表明TUSLA算法能够高效且精确地解决此类优化问题。

    

    本文考虑目标函数具有超线性增加和不连续随机梯度的非凸随机优化问题。针对这种情况，我们对Lovas等人（2020年）引入的tamed unadjusted stochastic Langevin algorithm（TUSLA）进行了非渐进性分析。特别地，我们在Wasserstein-1和Wasserstein-2距离上建立了TUSLA算法的非渐进误差界限。后一结果使我们能够进一步推导期望过量风险的非渐进估计值。为了说明主要结果的适用性，我们考虑了一个包含ReLU神经网络的迁移学习示例，该示例代表机器学习中的一个关键范例。我们为上述示例呈现了数值实验，支持了我们的理论发现。因此，在这种情况下，我们在理论和数值上都证明了TUSLA算法能够高效且精确地解决涉及ReLU激活函数的神经网络优化问题。

    We consider non-convex stochastic optimization problems where the objective functions have super-linearly growing and discontinuous stochastic gradients. In such a setting, we provide a non-asymptotic analysis for the tamed unadjusted stochastic Langevin algorithm (TUSLA) introduced in Lovas et al. (2020). In particular, we establish non-asymptotic error bounds for the TUSLA algorithm in Wasserstein-1 and Wasserstein-2 distances. The latter result enables us to further derive non-asymptotic estimates for the expected excess risk. To illustrate the applicability of the main results, we consider an example from transfer learning with ReLU neural networks, which represents a key paradigm in machine learning. Numerical experiments are presented for the aforementioned example which support our theoretical findings. Hence, in this setting, we demonstrate both theoretically and numerically that the TUSLA algorithm can solve the optimization problem involving neural networks with ReLU activati
    
[^30]: 词向量：一项综述

    Word Embeddings: A Survey. (arXiv:1901.09069v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/1901.09069](http://arxiv.org/abs/1901.09069)

    这篇综述介绍了一些主要的词向量构建策略，称为word embeddings，这些策略基于分布假设，编码了语法和语义信息，并被证明在很多NLP任务中是有用的额外特征。

    

    这项工作列出并描述了近期主要的策略，基于分布假设，用于构建单词的固定长度、密集和分布式表示。 这些表示现在通常被称为词向量，并且除了编码出令人惊讶的语法和语义信息外，在许多下游NLP任务中已被证明是有用的额外特征。

    This work lists and describes the main recent strategies for building fixed-length, dense and distributed representations for words, based on the distributional hypothesis. These representations are now commonly called word embeddings and, in addition to encoding surprisingly good syntactic and semantic information, have been proven useful as extra features in many downstream NLP tasks.
    
[^31]: 基于双重协变量调整的块模型用于社区检测

    Pairwise Covariates-adjusted Block Model for Community Detection. (arXiv:1807.03469v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/1807.03469](http://arxiv.org/abs/1807.03469)

    基于SBM模型，双重协变量调整的PCABM模型添加了关于节点间关系的附加信息。SCWA算法对PCABM模型进行了高效求解。模拟实验和实际数据分析表明PCABM模型具有优异的性能和预测能力。

    

    社区检测是网络研究中最基本的问题之一。随机块模型(SBM)是一种广泛应用的模型，已开发出各种估计方法并揭示了它们的社区检测一致性结果。但是，SBM受到一种假设的限制，即同一社区中的所有节点都是随机等价的，这可能不适用于实际应用。我们引入了一种基于双重协变量调整的随机块模型(PCABM)，即将双重协变量信息合并到SBM中。我们研究了协变量系数和社区分配的极大似然估计值。证明了在适当的稀疏条件下，协变量系数估计和社区分配均一致。介绍了一种带有调整的谱聚类（SCWA），以高效地解决PCABM问题。我们推导了SCWA检测社区的误差界限，证明了算法能够实现精确的社区恢复。数值模拟和实际数据分析表明PCABM优于现有方法。

    One of the most fundamental problems in network study is community detection. The stochastic block model (SBM) is a widely used model, for which various estimation methods have been developed with their community detection consistency results unveiled. However, the SBM is restricted by the strong assumption that all nodes in the same community are stochastically equivalent, which may not be suitable for practical applications. We introduce a pairwise covariates-adjusted stochastic block model (PCABM), a generalization of SBM that incorporates pairwise covariate information. We study the maximum likelihood estimates of the coefficients for the covariates as well as the community assignments. It is shown that both the coefficient estimates of the covariates and the community assignments are consistent under suitable sparsity conditions. Spectral clustering with adjustment (SCWA) is introduced to efficiently solve PCABM. Under certain conditions, we derive the error bound of community det
    

