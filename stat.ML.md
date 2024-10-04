# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benign overfitting in leaky ReLU networks with moderate input dimension](https://arxiv.org/abs/2403.06903) | 研究了在泄漏ReLU网络上使用铰链损失进行训练的过程中，信噪比（SNR）条件对于良性和非良性过拟合的影响，并发现高SNR值会导致良性过拟合，低SNR值则会导致有害过拟合。 |
| [^2] | [Signature Isolation Forest](https://arxiv.org/abs/2403.04405) | 介绍了一种新颖的异常检测算法"Signature Isolation Forest"，利用粗路径理论的签名变换去除了Functional Isolation Forest的线性内积和词典选择方面的限制。 |
| [^3] | [Latent Attention for Linear Time Transformers](https://arxiv.org/abs/2402.17512) | 提出了一种基于潜在向量定义注意力的方法，将标准transformer中的注意力机制的时间复杂度从二次方降低到与时间线性相关，表现与标准注意力媲美，但允许上下文窗口扩展到远远超出标准的范围。 |
| [^4] | [On the Curses of Future and History in Future-dependent Value Functions for Off-policy Evaluation](https://arxiv.org/abs/2402.14703) | 本文提出了针对POMDP结构的新颖覆盖假设，以解决未来依赖价值函数方法中的长度指数增长问题。 |
| [^5] | [Second Order Methods for Bandit Optimization and Control](https://arxiv.org/abs/2402.08929) | 本文提出了一种简单实用的二阶赌徒凸优化算法，并证明了其对于一类称之为$\kappa$-凸的凸函数实现了最优的后期损失界限。该算法在多个应用中表现出高效性能，包括赌徒逻辑回归。 |
| [^6] | [Solution of the Probabilistic Lambert Problem: Connections with Optimal Mass Transport, Schr\"odinger Bridge and Reaction-Diffusion PDEs.](http://arxiv.org/abs/2401.07961) | 这项研究将概率Lambert问题与最优质量传输、Schr\"odinger桥和反应-扩散偏微分方程等领域连接起来，从而解决了概率Lambert问题的解的存在和唯一性，并提供了数值求解的方法。 |
| [^7] | [Hebbian Learning from First Principles.](http://arxiv.org/abs/2401.07110) | 本文从第一原理中得到了赫布学习的明确表达式，并证明了这些学习规则在大数据极限下收敛到原始的存储方案。 |
| [^8] | [Machine learning a fixed point action for SU(3) gauge theory with a gauge equivariant convolutional neural network.](http://arxiv.org/abs/2401.06481) | 本研究使用具有精确规范不变性的卷积神经网络，通过机器学习方法找到了四维SU（3）规范理论的优秀固定点作用的参数化方法，为未来的蒙特卡洛模拟打下了必要的基础。 |
| [^9] | [Randomized Runge-Kutta-Nystr\"om.](http://arxiv.org/abs/2310.07399) | 本文介绍了5/2阶和7/2阶$L^2$-准确的随机Runge-Kutta-Nystr\"om方法，用于近似底层的哈密顿流，并展示了它在高维目标分布中的卓越效率。 |
| [^10] | [Optimal Learners for Realizable Regression: PAC Learning and Online Learning.](http://arxiv.org/abs/2307.03848) | 本论文研究了可实现回归问题的PAC学习和在线学习的统计复杂度，并提出了对于可学习性的必要条件和充分条件。 |
| [^11] | [On the Statistical Efficiency of Mean Field Reinforcement Learning with General Function Approximation.](http://arxiv.org/abs/2305.11283) | 本文研究了一般函数逼近下的均场控制(MFC)和均场博弈(MFG)中的强化学习的统计效率，提出了基于乐观最大似然估计的算法，并仅对转移动力学具有Lipschitz连续性的假设，最后建立了一个指数级的下界支持MFC设置。 |

# 详细

[^1]: 具有适度输入维度的泄漏ReLU网络中的良性过拟合问题

    Benign overfitting in leaky ReLU networks with moderate input dimension

    [https://arxiv.org/abs/2403.06903](https://arxiv.org/abs/2403.06903)

    研究了在泄漏ReLU网络上使用铰链损失进行训练的过程中，信噪比（SNR）条件对于良性和非良性过拟合的影响，并发现高SNR值会导致良性过拟合，低SNR值则会导致有害过拟合。

    

    良性过拟合问题探讨了一个模型是否能够完美地拟合嘈杂的训练数据，同时又能够很好地泛化。我们研究了在二层泄漏ReLU网络上使用铰链损失进行训练的良性过拟合问题，针对二分类任务。我们考虑输入数据，可以分解为一个共同信号和一个随机噪声成分的总和，这两者相互正交。我们表征了模型参数的信噪比（SNR）条件，导致了良性和非良性（有害）过拟合：特别是，如果SNR很高，则发生良性过拟合，相反，如果SNR很低，则发生有害过拟合。我们将良性和非良性过拟合归因于一个近似边界最大化性质，并展示了在铰链损失下使用梯度下降（GD）训练的泄漏ReLU网络满足这一性质。与以前的工作相比，我们不需要nea

    arXiv:2403.06903v1 Announce Type: new  Abstract: The problem of benign overfitting asks whether it is possible for a model to perfectly fit noisy training data and still generalize well. We study benign overfitting in two-layer leaky ReLU networks trained with the hinge loss on a binary classification task. We consider input data which can be decomposed into the sum of a common signal and a random noise component, which lie on subspaces orthogonal to one another. We characterize conditions on the signal to noise ratio (SNR) of the model parameters giving rise to benign versus non-benign, or harmful, overfitting: in particular, if the SNR is high then benign overfitting occurs, conversely if the SNR is low then harmful overfitting occurs. We attribute both benign and non-benign overfitting to an approximate margin maximization property and show that leaky ReLU networks trained on hinge loss with Gradient Descent (GD) satisfy this property. In contrast to prior work we do not require nea
    
[^2]: Signature Isolation Forest

    Signature Isolation Forest

    [https://arxiv.org/abs/2403.04405](https://arxiv.org/abs/2403.04405)

    介绍了一种新颖的异常检测算法"Signature Isolation Forest"，利用粗路径理论的签名变换去除了Functional Isolation Forest的线性内积和词典选择方面的限制。

    

    Functional Isolation Forest (FIF)是一种针对功能数据设计的最新一流异常检测(AD)算法。它依赖于一种树分区过程，通过将每个曲线观测投影到通过线性内积绘制的词典上来计算异常得分。本文通过引入“Signature Isolation Forest”，一种利用粗路径理论签名变换的新颖AD算法类，来解决这些挑战。我们的目标是通过提出两种算法来消除FIF施加的限制，这两种算法特别针对FIF内积的线性性和词典的选择。

    arXiv:2403.04405v1 Announce Type: cross  Abstract: Functional Isolation Forest (FIF) is a recent state-of-the-art Anomaly Detection (AD) algorithm designed for functional data. It relies on a tree partition procedure where an abnormality score is computed by projecting each curve observation on a drawn dictionary through a linear inner product. Such linear inner product and the dictionary are a priori choices that highly influence the algorithm's performances and might lead to unreliable results, particularly with complex datasets. This work addresses these challenges by introducing \textit{Signature Isolation Forest}, a novel AD algorithm class leveraging the rough path theory's signature transform. Our objective is to remove the constraints imposed by FIF through the proposition of two algorithms which specifically target the linearity of the FIF inner product and the choice of the dictionary. We provide several numerical experiments, including a real-world applications benchmark sho
    
[^3]: Latent Attention for Linear Time Transformers

    Latent Attention for Linear Time Transformers

    [https://arxiv.org/abs/2402.17512](https://arxiv.org/abs/2402.17512)

    提出了一种基于潜在向量定义注意力的方法，将标准transformer中的注意力机制的时间复杂度从二次方降低到与时间线性相关，表现与标准注意力媲美，但允许上下文窗口扩展到远远超出标准的范围。

    

    标准transformer中的注意力机制的时间复杂度随着序列长度的增加呈二次方增长。我们引入一种通过定义潜在向量的注意力来将其降低到与时间线性相关的方法。该方法可以轻松作为标准注意力机制的替代品。我们的“Latte Transformer”模型可用于双向和单向任务，因果版本允许一种在推理语言生成任务中内存和时间高效的递归实现。标准transformer的下一个标记预测随着序列长度线性增长，而Latte Transformer计算下一个标记所需的时间是恒定的。我们的方法的实证表现可与标准注意力媲美，但允许将上下文窗口扩展到远远超出标准注意力实际可行的范围。

    arXiv:2402.17512v1 Announce Type: new  Abstract: The time complexity of the standard attention mechanism in a transformer scales quadratically with the length of the sequence. We introduce a method to reduce this to linear scaling with time, based on defining attention via latent vectors. The method is readily usable as a drop-in replacement for the standard attention mechanism. Our "Latte Transformer" model can be implemented for both bidirectional and unidirectional tasks, with the causal version allowing a recurrent implementation which is memory and time-efficient during inference of language generation tasks. Whilst next token prediction scales linearly with the sequence length for a standard transformer, a Latte Transformer requires constant time to compute the next token. The empirical performance of our method is comparable to standard attention, yet allows scaling to context windows much larger than practical in standard attention.
    
[^4]: 在未来依赖价值函数中探讨未来和历史的诅咒在离线评估中的应用

    On the Curses of Future and History in Future-dependent Value Functions for Off-policy Evaluation

    [https://arxiv.org/abs/2402.14703](https://arxiv.org/abs/2402.14703)

    本文提出了针对POMDP结构的新颖覆盖假设，以解决未来依赖价值函数方法中的长度指数增长问题。

    

    我们研究了在部分可观测环境中复杂观测的离线评估(OPE)，旨在开发能够避免对时间跨度指数依赖的估计器。最近，Uehara等人（2022年）提出了未来依赖价值函数作为解决这一问题的一个有前途的框架。然而，该框架也取决于未来依赖价值函数的有界性以及其他相关数量，我们发现这些数量可能会随着长度呈指数增长，从而抹去该方法的优势。在本文中，我们发现了针对POMDP结构的新颖覆盖假设。

    arXiv:2402.14703v1 Announce Type: cross  Abstract: We study off-policy evaluation (OPE) in partially observable environments with complex observations, with the goal of developing estimators whose guarantee avoids exponential dependence on the horizon. While such estimators exist for MDPs and POMDPs can be converted to history-based MDPs, their estimation errors depend on the state-density ratio for MDPs which becomes history ratios after conversion, an exponential object. Recently, Uehara et al. (2022) proposed future-dependent value functions as a promising framework to address this issue, where the guarantee for memoryless policies depends on the density ratio over the latent state space. However, it also depends on the boundedness of the future-dependent value function and other related quantities, which we show could be exponential-in-length and thus erasing the advantage of the method. In this paper, we discover novel coverage assumptions tailored to the structure of POMDPs, such
    
[^5]: 二阶方法用于赌徒优化与控制

    Second Order Methods for Bandit Optimization and Control

    [https://arxiv.org/abs/2402.08929](https://arxiv.org/abs/2402.08929)

    本文提出了一种简单实用的二阶赌徒凸优化算法，并证明了其对于一类称之为$\kappa$-凸的凸函数实现了最优的后期损失界限。该算法在多个应用中表现出高效性能，包括赌徒逻辑回归。

    

    Bandit凸优化(BCO)是一种在不确定性下进行在线决策的通用框架。尽管已经建立了一般凸损失的紧束后期界限，但现有算法在高维数据上具有难以忍受的计算成本。在本文中，我们提出了一种受在线牛顿步骤算法启发的简单实用的BCO算法。我们证明了我们的算法对于一类我们称之为$\kappa$-凸的凸函数实现了最优(从层面上讲)的后期界限。这个类包含了一系列实际相关的损失函数，包括线性、二次和广义线性模型。除了最优的后期损失，这种方法也是一些经过深入研究的应用中已知的最高效的算法，包括赌徒逻辑回归。

    arXiv:2402.08929v1 Announce Type: new Abstract: Bandit convex optimization (BCO) is a general framework for online decision making under uncertainty. While tight regret bounds for general convex losses have been established, existing algorithms achieving these bounds have prohibitive computational costs for high dimensional data.   In this paper, we propose a simple and practical BCO algorithm inspired by the online Newton step algorithm. We show that our algorithm achieves optimal (in terms of horizon) regret bounds for a large class of convex functions that we call $\kappa$-convex. This class contains a wide range of practically relevant loss functions including linear, quadratic, and generalized linear models. In addition to optimal regret, this method is the most efficient known algorithm for several well-studied applications including bandit logistic regression.   Furthermore, we investigate the adaptation of our second-order bandit algorithm to online convex optimization with mem
    
[^6]: 概率Lambert问题的解决方案：与最优质量传输、Schr\"odinger桥和反应-扩散偏微分方程的连接

    Solution of the Probabilistic Lambert Problem: Connections with Optimal Mass Transport, Schr\"odinger Bridge and Reaction-Diffusion PDEs. (arXiv:2401.07961v1 [math.OC])

    [http://arxiv.org/abs/2401.07961](http://arxiv.org/abs/2401.07961)

    这项研究将概率Lambert问题与最优质量传输、Schr\"odinger桥和反应-扩散偏微分方程等领域连接起来，从而解决了概率Lambert问题的解的存在和唯一性，并提供了数值求解的方法。

    

    Lambert问题涉及通过速度控制在规定的飞行时间内将航天器从给定的初始位置转移到给定的终端位置，受到重力力场的限制。我们考虑了Lambert问题的概率变种，其中位置向量的端点约束的知识被它们各自的联合概率密度函数所替代。我们证明了具有端点联合概率密度约束的Lambert问题是一个广义的最优质量传输（OMT）问题，从而将这个经典的天体动力学问题与现代随机控制和随机机器学习的新兴研究领域联系起来。这个新发现的连接使我们能够严格建立概率Lambert问题的解的存在性和唯一性。同样的连接还帮助通过扩散正规化数值求解概率Lambert问题，即通过进一步的连接来利用。

    Lambert's problem concerns with transferring a spacecraft from a given initial to a given terminal position within prescribed flight time via velocity control subject to a gravitational force field. We consider a probabilistic variant of the Lambert problem where the knowledge of the endpoint constraints in position vectors are replaced by the knowledge of their respective joint probability density functions. We show that the Lambert problem with endpoint joint probability density constraints is a generalized optimal mass transport (OMT) problem, thereby connecting this classical astrodynamics problem with a burgeoning area of research in modern stochastic control and stochastic machine learning. This newfound connection allows us to rigorously establish the existence and uniqueness of solution for the probabilistic Lambert problem. The same connection also helps to numerically solve the probabilistic Lambert problem via diffusion regularization, i.e., by leveraging further connection 
    
[^7]: 从第一原理中的赫布学习

    Hebbian Learning from First Principles. (arXiv:2401.07110v1 [cond-mat.dis-nn])

    [http://arxiv.org/abs/2401.07110](http://arxiv.org/abs/2401.07110)

    本文从第一原理中得到了赫布学习的明确表达式，并证明了这些学习规则在大数据极限下收敛到原始的存储方案。

    

    最近，针对神经网络的Hopfield模型及其密集概化形式的原始存储方案已通过假设其哈密顿量的表达式为监督和无监督协议，成为真正的赫布学习规则。在本文中，我们首先依靠Jaynes的最大熵极值法得到了这些明确的表达式。除了形式上推导出这些赫布学习的规则，这个构建还突显了熵极值中的朗格朗日约束如何强制网络结果上的神经相关性：这些尝试模仿提供给网络进行训练的数据集中隐藏的经验支持，而且网络越密集，能够捕捉到的相关性时间越长。接下来，我们证明在大数据极限下，无论是否存在教师，这些赫布学习规则都会收敛到原始的存储方案。

    Recently, the original storage prescription for the Hopfield model of neural networks -- as well as for its dense generalizations -- has been turned into a genuine Hebbian learning rule by postulating the expression of its Hamiltonian for both the supervised and unsupervised protocols. In these notes, first, we obtain these explicit expressions by relying upon maximum entropy extremization \`a la Jaynes. Beyond providing a formal derivation of these recipes for Hebbian learning, this construction also highlights how Lagrangian constraints within entropy extremization force network's outcomes on neural correlations: these try to mimic the empirical counterparts hidden in the datasets provided to the network for its training and, the denser the network, the longer the correlations that it is able to capture. Next, we prove that, in the big data limit, whatever the presence of a teacher (or its lacking), not only these Hebbian learning rules converge to the original storage prescription o
    
[^8]: 使用具有规范等变性的卷积神经网络机器学习SU（3）规范理论的固定点作用

    Machine learning a fixed point action for SU(3) gauge theory with a gauge equivariant convolutional neural network. (arXiv:2401.06481v1 [hep-lat] CROSS LISTED)

    [http://arxiv.org/abs/2401.06481](http://arxiv.org/abs/2401.06481)

    本研究使用具有精确规范不变性的卷积神经网络，通过机器学习方法找到了四维SU（3）规范理论的优秀固定点作用的参数化方法，为未来的蒙特卡洛模拟打下了必要的基础。

    

    固定点的格子作用被设计成具有不受离散化效应影响的连续经典性质，并在量子层面上减少格子效应。它们提供了一种用较粗的格子来提取连续物理的可能方法，从而绕过与连续极限相关的临界减慢和拓扑冻结问题。实际应用的关键是找到一个精确且紧凑的固定点作用参数化方法，因为其许多性质只是隐含定义的。在这里，我们使用机器学习方法重新思考了如何参数化固定点作用的问题。特别地，我们使用具有精确规范不变性的卷积神经网络获得四维SU（3）规范理论的固定点作用。大的算子空间使我们能够找到比之前研究更好的参数化方法，这是未来蒙特卡洛模拟的必要第一步。

    Fixed point lattice actions are designed to have continuum classical properties unaffected by discretization effects and reduced lattice artifacts at the quantum level. They provide a possible way to extract continuum physics with coarser lattices, thereby allowing to circumvent problems with critical slowing down and topological freezing toward the continuum limit. A crucial ingredient for practical applications is to find an accurate and compact parametrization of a fixed point action, since many of its properties are only implicitly defined. Here we use machine learning methods to revisit the question of how to parametrize fixed point actions. In particular, we obtain a fixed point action for four-dimensional SU(3) gauge theory using convolutional neural networks with exact gauge invariance. The large operator space allows us to find superior parametrizations compared to previous studies, a necessary first step for future Monte Carlo simulations.
    
[^9]: 随机Runge-Kutta-Nystr\"om方法在非可逆马尔科夫链中的应用

    Randomized Runge-Kutta-Nystr\"om. (arXiv:2310.07399v1 [math.NA])

    [http://arxiv.org/abs/2310.07399](http://arxiv.org/abs/2310.07399)

    本文介绍了5/2阶和7/2阶$L^2$-准确的随机Runge-Kutta-Nystr\"om方法，用于近似底层的哈密顿流，并展示了它在高维目标分布中的卓越效率。

    

    本文介绍了5/2阶和7/2阶$L^2$-准确的随机Runge-Kutta-Nystr\"om方法，用于近似底层的哈密顿流，包括不调整的哈密顿蒙特卡洛和不调整的动力学朗之万链。通过在势能函数的梯度和海森矩阵的Lipschitz假设下提供了量化的5/2阶$L^2$-准确度上限。对于一些“良好行为”的高维目标分布，通过数值实验对应的马尔科夫链表现出很高的效率。

    We present 5/2- and 7/2-order $L^2$-accurate randomized Runge-Kutta-Nystr\"om methods to approximate the Hamiltonian flow underlying various non-reversible Markov chain Monte Carlo chains including unadjusted Hamiltonian Monte Carlo and unadjusted kinetic Langevin chains. Quantitative 5/2-order $L^2$-accuracy upper bounds are provided under gradient and Hessian Lipschitz assumptions on the potential energy function. The superior complexity of the corresponding Markov chains is numerically demonstrated for a selection of `well-behaved', high-dimensional target distributions.
    
[^10]: 可实现回归的最优学习算法：PAC学习和在线学习

    Optimal Learners for Realizable Regression: PAC Learning and Online Learning. (arXiv:2307.03848v1 [cs.LG])

    [http://arxiv.org/abs/2307.03848](http://arxiv.org/abs/2307.03848)

    本论文研究了可实现回归问题的PAC学习和在线学习的统计复杂度，并提出了对于可学习性的必要条件和充分条件。

    

    本研究旨在对可实现回归在PAC学习和在线学习的统计复杂度进行刻画。先前的研究已经证明了有限的fat shattering维度对于PAC学习的充分性以及有限的scaled Natarajan维度对于必要性的存在，但自从Simon 1997（SICOMP '97）的工作以来，对于更完整的刻画的进展甚少。为此，我们首先引入了一种最小化实例最优学习算法来对可实现回归进行学习，并提出了一种既定性又定量地刻画了哪些类的实数预测器可以被学习的新颖维度。然后，我们确定了一个与图维度相关的组合维度，该维度刻画了在可实现设置中的ERM可学习性。最后，我们根据与DS维度相关的组合维度建立了学习可行性的必要条件，并猜测它也可能是充分的。

    In this work, we aim to characterize the statistical complexity of realizable regression both in the PAC learning setting and the online learning setting.  Previous work had established the sufficiency of finiteness of the fat shattering dimension for PAC learnability and the necessity of finiteness of the scaled Natarajan dimension, but little progress had been made towards a more complete characterization since the work of Simon 1997 (SICOMP '97). To this end, we first introduce a minimax instance optimal learner for realizable regression and propose a novel dimension that both qualitatively and quantitatively characterizes which classes of real-valued predictors are learnable. We then identify a combinatorial dimension related to the Graph dimension that characterizes ERM learnability in the realizable setting. Finally, we establish a necessary condition for learnability based on a combinatorial dimension related to the DS dimension, and conjecture that it may also be sufficient in 
    
[^11]: 关于一般函数逼近下的均场强化学习的统计效率

    On the Statistical Efficiency of Mean Field Reinforcement Learning with General Function Approximation. (arXiv:2305.11283v1 [cs.LG])

    [http://arxiv.org/abs/2305.11283](http://arxiv.org/abs/2305.11283)

    本文研究了一般函数逼近下的均场控制(MFC)和均场博弈(MFG)中的强化学习的统计效率，提出了基于乐观最大似然估计的算法，并仅对转移动力学具有Lipschitz连续性的假设，最后建立了一个指数级的下界支持MFC设置。

    

    本文研究了一般函数逼近下的均场控制（MFC）和均场博弈（MFG）中强化学习的统计效率。引入了一种称为Mean-Field Model-Based Eluder Dimension (MBED)的新概念，包含了一系列丰富的均场强化学习问题。此外，我们提出了基于乐观最大似然估计的算法，可以返回一个$\epsilon$优的策略，适用于MFC或$\epsilon$纳什均衡策略适用于MFG，样本复杂度多项式与相关参数无关，与状态、动作和代理数量无关。值得注意的是，我们的结果仅对转移动力学具有Lipschitz连续性的假设，避免了以前的强结构假设。最后，在tabular设置下，假设有一个生成模型，我们建立了一个指数级的下界支持MFC设置，同时提供了一种新颖的样本高效的模型消除算法以逼近最优策略。

    In this paper, we study the statistical efficiency of Reinforcement Learning in Mean-Field Control (MFC) and Mean-Field Game (MFG) with general function approximation. We introduce a new concept called Mean-Field Model-Based Eluder Dimension (MBED), which subsumes a rich family of Mean-Field RL problems. Additionally, we propose algorithms based on Optimistic Maximal Likelihood Estimation, which can return an $\epsilon$-optimal policy for MFC or an $\epsilon$-Nash Equilibrium policy for MFG, with sample complexity polynomial w.r.t. relevant parameters and independent of the number of states, actions and the number of agents. Notably, our results only require a mild assumption of Lipschitz continuity on transition dynamics and avoid strong structural assumptions in previous work. Finally, in the tabular setting, given the access to a generative model, we establish an exponential lower bound for MFC setting, while providing a novel sample-efficient model elimination algorithm to approxim
    

