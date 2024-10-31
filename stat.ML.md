# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zeroth-Order Sampling Methods for Non-Log-Concave Distributions: Alleviating Metastability by Denoising Diffusion](https://arxiv.org/abs/2402.17886) | 本文提出了一种基于去噪扩散过程的零阶扩散蒙特卡洛算法，克服了非对数凹分布采样中的亚稳定性问题，并证明其采样精度具有倒多项式依赖。 |
| [^2] | [Learnability is a Compact Property](https://arxiv.org/abs/2402.10360) | 监督学习问题的困难性具有紧凑的有限特性表征。 |
| [^3] | [Tradeoffs of Diagonal Fisher Information Matrix Estimators](https://arxiv.org/abs/2402.05379) | 本研究探讨了使用对角费舍尔信息矩阵估计器的权衡。通过分析和数值研究，发现方差量取决于非线性与不同参数组之间的关系，应该在估计费舍尔信息时予以重视。 |
| [^4] | [Misspecification uncertainties in near-deterministic regression](https://arxiv.org/abs/2402.01810) | 该论文研究了近确定性回归中错误规范化的不确定性问题，并提出了一种组合模型，以准确预测和控制参数不确定性。 |
| [^5] | [Score-based Causal Representation Learning: Linear and General Transformations](https://arxiv.org/abs/2402.00849) | 这篇论文提出了一种基于得分的算法类，用于干预范围内的因果表示学习，涵盖了线性和一般转化。算法保证了可识别性和实现性，并且通过创造性地将得分函数与因果表示学习相结合。 |
| [^6] | [Fair Coresets via Optimal Transport](https://arxiv.org/abs/2311.05436) | 本研究提出了公平的Wasserstein核心集(FWC)，该方法通过最小化原始数据集与加权合成样本之间的Wasserstein距离，并强制实现人口平等，生成公平的合成代表性样本，可用于下游学习任务。 |
| [^7] | [A Survey Analyzing Generalization in Deep Reinforcement Learning.](http://arxiv.org/abs/2401.02349) | 本文调查了深度强化学习中的泛化性能。深度强化学习策略存在过拟合问题，限制了它们的鲁棒性和泛化能力。研究形式化和统一了提高泛化性和克服过拟合的不同解决方案。 |
| [^8] | [PAC-Bayes-Chernoff bounds for unbounded losses.](http://arxiv.org/abs/2401.01148) | 这篇论文提出了一种用于无界损失的高概率PAC-Bayes参考界限，并通过优化自由参数解决了一些开放问题，并通过灵活的假设产生了新的广义界限。 |
| [^9] | [When, Why and How Much? Adaptive Learning Rate Scheduling by Refinement.](http://arxiv.org/abs/2310.07831) | 该论文介绍了一种通过细化分析学习率调度来解决实践中学习率调整与理论的不一致的方法，通过对观察到的梯度范数进行分析，得到了适应于特定任务的细化调度。该方法能够改善优化算法的收敛性能。 |
| [^10] | [Solving Quadratic Systems with Full-Rank Matrices Using Sparse or Generative Priors.](http://arxiv.org/abs/2309.09032) | 本论文提出了一个方法，通过使用稀疏或生成的先验知识，解决了从全秩矩阵的二次系统中恢复信号的问题。其中，通过引入阈值Wirtinger流算法（TWF）来处理稀疏信号，并使用谱初始化和阈值梯度下降方法，在高维情况下实现了较小的测量数量。 |
| [^11] | [Efficient distributed representations beyond negative sampling.](http://arxiv.org/abs/2303.17475) | 本文介绍了一种高效的分布式表示（嵌入）学习方法，通过线性时间估计softmax归一化常数来实现学习过程，该方法优于负采样方法并在多项测试中验证了其有效性。 |

# 详细

[^1]: 用于非对数凹分布的零阶采样方法：通过去噪扩散缓解亚稳定性

    Zeroth-Order Sampling Methods for Non-Log-Concave Distributions: Alleviating Metastability by Denoising Diffusion

    [https://arxiv.org/abs/2402.17886](https://arxiv.org/abs/2402.17886)

    本文提出了一种基于去噪扩散过程的零阶扩散蒙特卡洛算法，克服了非对数凹分布采样中的亚稳定性问题，并证明其采样精度具有倒多项式依赖。

    

    这篇论文考虑了基于其非对数凹分布未归一化密度查询的采样问题。首先描述了一个基于模拟去噪扩散过程的框架，即扩散蒙特卡洛（DMC），其得分函数通过通用蒙特卡洛估计器逼近。DMC是一个基于神谕的元算法，其中神谕是假设可以访问生成蒙特卡洛分数估计器的样本的访问。然后，我们提供了一个基于拒绝采样的这个神谕的实现，这将DMC转化为一个真正的算法，称为零阶扩散蒙特卡洛（ZOD-MC）。我们通过首先构建一个通用框架，即DMC的性能保证，而不假设目标分布为对数凹或满足任何等周不等式，提供了收敛分析。然后我们证明ZOD-MC对所需采样精度具有倒多项式依赖，尽管仍然受到...

    arXiv:2402.17886v1 Announce Type: cross  Abstract: This paper considers the problem of sampling from non-logconcave distribution, based on queries of its unnormalized density. It first describes a framework, Diffusion Monte Carlo (DMC), based on the simulation of a denoising diffusion process with its score function approximated by a generic Monte Carlo estimator. DMC is an oracle-based meta-algorithm, where its oracle is the assumed access to samples that generate a Monte Carlo score estimator. Then we provide an implementation of this oracle, based on rejection sampling, and this turns DMC into a true algorithm, termed Zeroth-Order Diffusion Monte Carlo (ZOD-MC). We provide convergence analyses by first constructing a general framework, i.e. a performance guarantee for DMC, without assuming the target distribution to be log-concave or satisfying any isoperimetric inequality. Then we prove that ZOD-MC admits an inverse polynomial dependence on the desired sampling accuracy, albeit sti
    
[^2]: 学习性是一种紧凑性质

    Learnability is a Compact Property

    [https://arxiv.org/abs/2402.10360](https://arxiv.org/abs/2402.10360)

    监督学习问题的困难性具有紧凑的有限特性表征。

    

    最近关于学习的工作取得了一个引人注目的结果：各种问题的可学习性可能是不可判定的，或者与标准集合论ZFC公理无关。此外，这种问题的可学习性可能不是具有有限特性的属性：非正式地说，它不能通过检查问题的有限投影来检测。

    arXiv:2402.10360v1 Announce Type: new  Abstract: Recent work on learning has yielded a striking result: the learnability of various problems can be undecidable, or independent of the standard ZFC axioms of set theory. Furthermore, the learnability of such problems can fail to be a property of finite character: informally, it cannot be detected by examining finite projections of the problem.   On the other hand, learning theory abounds with notions of dimension that characterize learning and consider only finite restrictions of the problem, i.e., are properties of finite character. How can these results be reconciled? More precisely, which classes of learning problems are vulnerable to logical undecidability, and which are within the grasp of finite characterizations?   We demonstrate that the difficulty of supervised learning with metric losses admits a tight finite characterization. In particular, we prove that the sample complexity of learning a hypothesis class can be detected by ex
    
[^3]: 对角费舍尔信息矩阵估计器的权衡

    Tradeoffs of Diagonal Fisher Information Matrix Estimators

    [https://arxiv.org/abs/2402.05379](https://arxiv.org/abs/2402.05379)

    本研究探讨了使用对角费舍尔信息矩阵估计器的权衡。通过分析和数值研究，发现方差量取决于非线性与不同参数组之间的关系，应该在估计费舍尔信息时予以重视。

    

    费舍尔信息矩阵描述了神经网络参数空间中的局部几何性质，它提供了理论和工具来理解和优化神经网络。鉴于其计算成本高，实践者通常使用随机估计器，并仅评估对角线条目。我们研究了两种这样的估计器，其准确性和样本复杂性取决于它们关联的方差。我们推导了方差的界限，并在回归和分类网络中实例化它们。我们通过分析和数值研究来权衡这两个估计器。我们发现方差量取决于关于不同参数组的非线性，当估计费舍尔信息时不能忽视它们。

    The Fisher information matrix characterizes the local geometry in the parameter space of neural networks. It elucidates insightful theories and useful tools to understand and optimize neural networks. Given its high computational cost, practitioners often use random estimators and evaluate only the diagonal entries. We examine two such estimators, whose accuracy and sample complexity depend on their associated variances. We derive bounds of the variances and instantiate them in regression and classification networks. We navigate trade-offs of both estimators based on analytical and numerical studies. We find that the variance quantities depend on the non-linearity with respect to different parameter groups and should not be neglected when estimating the Fisher information.
    
[^4]: 近确定性回归中的错误规范化不确定性

    Misspecification uncertainties in near-deterministic regression

    [https://arxiv.org/abs/2402.01810](https://arxiv.org/abs/2402.01810)

    该论文研究了近确定性回归中错误规范化的不确定性问题，并提出了一种组合模型，以准确预测和控制参数不确定性。

    

    期望损失是模型泛化误差的上界，可用于学习的鲁棒PAC-Bayes边界。然而，损失最小化被认为忽略了错误规范化，即模型不能完全复制观测结果。这导致大数据或欠参数化极限下对参数不确定性的显著低估。我们分析近确定性、错误规范化和欠参数化替代模型的泛化误差，这是科学和工程中广泛相关的一个领域。我们证明后验分布必须覆盖每个训练点，以避免发散的泛化误差，并导出一个符合这个约束的组合模型。对于线性模型，这种高效的方法产生的额外开销最小。这种高效方法在模型问题上进行了演示，然后应用于原子尺度机器学习中的高维数据集。

    The expected loss is an upper bound to the model generalization error which admits robust PAC-Bayes bounds for learning. However, loss minimization is known to ignore misspecification, where models cannot exactly reproduce observations. This leads to significant underestimates of parameter uncertainties in the large data, or underparameterized, limit. We analyze the generalization error of near-deterministic, misspecified and underparametrized surrogate models, a regime of broad relevance in science and engineering. We show posterior distributions must cover every training point to avoid a divergent generalization error and derive an ensemble {ansatz} that respects this constraint, which for linear models incurs minimal overhead. The efficient approach is demonstrated on model problems before application to high dimensional datasets in atomistic machine learning. Parameter uncertainties from misspecification survive in the underparametrized limit, giving accurate prediction and boundin
    
[^5]: 基于得分的因果表示学习：线性和一般的转化

    Score-based Causal Representation Learning: Linear and General Transformations

    [https://arxiv.org/abs/2402.00849](https://arxiv.org/abs/2402.00849)

    这篇论文提出了一种基于得分的算法类，用于干预范围内的因果表示学习，涵盖了线性和一般转化。算法保证了可识别性和实现性，并且通过创造性地将得分函数与因果表示学习相结合。

    

    本篇论文针对一般非参数潜在因果模型和将潜在变量映射到观测变量的未知转化，研究了基于干预的因果表示学习（CRL）。研究了线性和一般的转化。这篇论文同时讨论了可识别性和实现性两个方面。可识别性是指确定算法不相关的条件，以确保恢复真实的潜在因果变量和潜在因果图。实现性是指算法方面，解决设计算法来实现可识别保证的问题。通过将得分函数（即密度函数对数的梯度）与CRL之间建立新联系，本文设计了一种得分为基础的算法类，确保了可识别性和实现性。首先，本文专注于线性转化，并展示了每个n个随机硬干预下该转化的因果表示可识别。

    This paper addresses intervention-based causal representation learning (CRL) under a general nonparametric latent causal model and an unknown transformation that maps the latent variables to the observed variables. Linear and general transformations are investigated. The paper addresses both the \emph{identifiability} and \emph{achievability} aspects. Identifiability refers to determining algorithm-agnostic conditions that ensure recovering the true latent causal variables and the latent causal graph underlying them. Achievability refers to the algorithmic aspects and addresses designing algorithms that achieve identifiability guarantees. By drawing novel connections between \emph{score functions} (i.e., the gradients of the logarithm of density functions) and CRL, this paper designs a \emph{score-based class of algorithms} that ensures both identifiability and achievability. First, the paper focuses on \emph{linear} transformations and shows that one stochastic hard intervention per n
    
[^6]: 通过最优传输实现公平的核心集

    Fair Coresets via Optimal Transport

    [https://arxiv.org/abs/2311.05436](https://arxiv.org/abs/2311.05436)

    本研究提出了公平的Wasserstein核心集(FWC)，该方法通过最小化原始数据集与加权合成样本之间的Wasserstein距离，并强制实现人口平等，生成公平的合成代表性样本，可用于下游学习任务。

    

    数据精炼和核心集已成为生成用于处理大规模数据集的下游学习任务的较小代表性样本集的流行方法。与此同时，机器学习越来越多地应用于社会层面的决策过程，使得模型构建者必须解决存在于数据中的子群体的固有偏见问题。当前方法通过优化相对于原始样本的局部属性来创建公平的合成代表性样本，但其对下游学习过程的影响尚未被探索。在这项工作中，我们提出了公平的Wasserstein核心集（FWC），一种新颖的核心集方法，它生成既具有公平性的合成代表性样本，又具有用于下游学习任务的样本级权重。FWC最小化原始数据集与加权合成样本之间的Wasserstein距离，同时强制实现人口平等。我们展示了FWC的无约束版本等价于通常的最优传输问题，并且通过实验证明了FWC的有效性和公平性。

    Data distillation and coresets have emerged as popular approaches to generate a smaller representative set of samples for downstream learning tasks to handle large-scale datasets. At the same time, machine learning is being increasingly applied to decision-making processes at a societal level, making it imperative for modelers to address inherent biases towards subgroups present in the data. Current approaches create fair synthetic representative samples by optimizing local properties relative to the original samples, but their effect on downstream learning processes has yet to be explored. In this work, we present fair Wasserstein coresets (FWC), a novel coreset approach which generates fair synthetic representative samples along with sample-level weights to be used in downstream learning tasks. FWC minimizes the Wasserstein distance between the original dataset and the weighted synthetic samples while enforcing demographic parity. We show that an unconstrained version of FWC is equiv
    
[^7]: 分析深度强化学习中泛化性能的调查

    A Survey Analyzing Generalization in Deep Reinforcement Learning. (arXiv:2401.02349v1 [cs.LG])

    [http://arxiv.org/abs/2401.02349](http://arxiv.org/abs/2401.02349)

    本文调查了深度强化学习中的泛化性能。深度强化学习策略存在过拟合问题，限制了它们的鲁棒性和泛化能力。研究形式化和统一了提高泛化性和克服过拟合的不同解决方案。

    

    利用深度神经网络解决高维状态或动作空间中的问题，强化学习研究在实践中取得了重要的成功和关注。尽管深度强化学习策略目前在许多领域中正在被应用，从医疗应用到自动驾驶车辆，但关于深度强化学习策略的泛化能力仍有许多待解答的问题。在本文中，我们将概述深度强化学习策略遇到过拟合问题的根本原因，限制了它们的鲁棒性和泛化能力。此外，我们将对提高泛化性和克服状态-动作值函数中的过拟合的不同解决方案进行形式化和统一。我们相信我们的研究可以为当前深度强化学习的进展提供一个简洁系统的统一分析，并有助于构建健壮的深度神经网络策略。

    Reinforcement learning research obtained significant success and attention with the utilization of deep neural networks to solve problems in high dimensional state or action spaces. While deep reinforcement learning policies are currently being deployed in many different fields from medical applications to self driving vehicles, there are still ongoing questions the field is trying to answer on the generalization capabilities of deep reinforcement learning policies. In this paper, we will outline the fundamental reasons why deep reinforcement learning policies encounter overfitting problems that limit their robustness and generalization capabilities. Furthermore, we will formalize and unify the diverse solution approaches to increase generalization, and overcome overfitting in state-action value functions. We believe our study can provide a compact systematic unified analysis for the current advancements in deep reinforcement learning, and help to construct robust deep neural policies 
    
[^8]: 无界损失的PAC-Bayes-Chernoff界限

    PAC-Bayes-Chernoff bounds for unbounded losses. (arXiv:2401.01148v1 [stat.ML])

    [http://arxiv.org/abs/2401.01148](http://arxiv.org/abs/2401.01148)

    这篇论文提出了一种用于无界损失的高概率PAC-Bayes参考界限，并通过优化自由参数解决了一些开放问题，并通过灵活的假设产生了新的广义界限。

    

    我们提出了一种新的用于无界损失的高概率PAC-Bayes参考界限。这个结果可以理解为Chernoff界限的PAC-Bayes版本。证明技巧依赖于通过Cramér变换对损失进行统一边界的尾部随机变量。我们强调了我们主要结果的两个应用。首先，我们证明了我们的界限解决了许多PAC-Bayes界限上的自由参数优化的开放问题。最后，我们证明了我们的方法允许在损失函数上进行灵活的假设，从而产生了广义了之前的界限，并且可以通过最小化来获得类似Gibbs的后验概率。

    We present a new high-probability PAC-Bayes oracle bound for unbounded losses. This result can be understood as a PAC-Bayes version of the Chernoff bound. The proof technique relies on uniformly bounding the tail of certain random variable based on the Cram\'er transform of the loss. We highlight two applications of our main result. First, we show that our bound solves the open problem of optimizing the free parameter on many PAC-Bayes bounds. Finally, we show that our approach allows working with flexible assumptions on the loss function, resulting in novel bounds that generalize previous ones and can be minimized to obtain Gibbs-like posteriors.
    
[^9]: 何时，为什么以及多少？通过细化进行的自适应学习率调度

    When, Why and How Much? Adaptive Learning Rate Scheduling by Refinement. (arXiv:2310.07831v1 [cs.LG])

    [http://arxiv.org/abs/2310.07831](http://arxiv.org/abs/2310.07831)

    该论文介绍了一种通过细化分析学习率调度来解决实践中学习率调整与理论的不一致的方法，通过对观察到的梯度范数进行分析，得到了适应于特定任务的细化调度。该方法能够改善优化算法的收敛性能。

    

    实践中使用的学习率调度与理论推荐的几乎完全不同。我们缩小了大部分理论与实践之间的差距，并因此能够推导出新的问题自适应学习率调度。我们的关键技术贡献是对广泛类别的优化算法（包括SGD）的学习率调度进行细化分析。与大多数前期研究只研究平均迭代的收敛性不同，我们研究最后一次迭代，这是大多数人在实践中使用的。当仅考虑最坏情况分析时，我们的理论预测最佳选择是线性衰减调度：这是一种实践中常用的选择，其将步长与当前迭代次数t和总步数T成比例地设置为1 - t/T。为了超越这种最坏情况分析，我们使用观察到的梯度范数来推导适应于特定任务的细化调度。这些细化调度表现出学习率逐渐增加和学习率迅速退火。

    Learning rate schedules used in practice bear little resemblance to those recommended by theory. We close much of this theory/practice gap, and as a consequence are able to derive new problem-adaptive learning rate schedules. Our key technical contribution is a refined analysis of learning rate schedules for a wide class of optimization algorithms (including SGD). In contrast to most prior works that study the convergence of the average iterate, we study the last iterate, which is what most people use in practice. When considering only worst-case analysis, our theory predicts that the best choice is the linear decay schedule: a popular choice in practice that sets the stepsize proportionally to $1 - t/T$, where $t$ is the current iteration and $T$ is the total number of steps. To go beyond this worst-case analysis, we use the observed gradient norms to derive schedules refined for any particular task. These refined schedules exhibit learning rate warm-up and rapid learning rate anneali
    
[^10]: 使用稀疏或生成的先验解决全秩矩阵的二次系统

    Solving Quadratic Systems with Full-Rank Matrices Using Sparse or Generative Priors. (arXiv:2309.09032v1 [cs.IT])

    [http://arxiv.org/abs/2309.09032](http://arxiv.org/abs/2309.09032)

    本论文提出了一个方法，通过使用稀疏或生成的先验知识，解决了从全秩矩阵的二次系统中恢复信号的问题。其中，通过引入阈值Wirtinger流算法（TWF）来处理稀疏信号，并使用谱初始化和阈值梯度下降方法，在高维情况下实现了较小的测量数量。

    

    从具有全秩矩阵的二次系统中恢复信号x在应用中经常出现，比如未分配的距离几何和亚波长成像。本文通过引入对x的先验知识，针对高维情况（m << n），使用独立同分布的标准高斯矩阵解决了该问题。首先，考虑k-稀疏的x，引入了TWF算法，该算法不需要稀疏水平k。TWF包括两个步骤：谱初始化，当m = O(k^2log n)时，确定了一个距离x足够近的点（可能会有符号翻转），以及具有很好初始化的阈值梯度下降，该下降产生了一个线性收敛到x的序列，用m = O(klog n)个测量。

    The problem of recovering a signal $\boldsymbol{x} \in \mathbb{R}^n$ from a quadratic system $\{y_i=\boldsymbol{x}^\top\boldsymbol{A}_i\boldsymbol{x},\ i=1,\ldots,m\}$ with full-rank matrices $\boldsymbol{A}_i$ frequently arises in applications such as unassigned distance geometry and sub-wavelength imaging. With i.i.d. standard Gaussian matrices $\boldsymbol{A}_i$, this paper addresses the high-dimensional case where $m\ll n$ by incorporating prior knowledge of $\boldsymbol{x}$. First, we consider a $k$-sparse $\boldsymbol{x}$ and introduce the thresholded Wirtinger flow (TWF) algorithm that does not require the sparsity level $k$. TWF comprises two steps: the spectral initialization that identifies a point sufficiently close to $\boldsymbol{x}$ (up to a sign flip) when $m=O(k^2\log n)$, and the thresholded gradient descent (with a good initialization) that produces a sequence linearly converging to $\boldsymbol{x}$ with $m=O(k\log n)$ measurements. Second, we explore the generative p
    
[^11]: 超越负采样的高效分布式表示方法

    Efficient distributed representations beyond negative sampling. (arXiv:2303.17475v1 [cs.LG])

    [http://arxiv.org/abs/2303.17475](http://arxiv.org/abs/2303.17475)

    本文介绍了一种高效的分布式表示（嵌入）学习方法，通过线性时间估计softmax归一化常数来实现学习过程，该方法优于负采样方法并在多项测试中验证了其有效性。

    

    本文介绍了一种高效的学习分布式表示（也称为嵌入）的方法。该方法通过最小化一个类似于Word2Vec算法中引入并在多个工作中采用的目标函数来实现。优化计算的瓶颈是softmax归一化常数的计算，这需要与样本大小呈二次比例的操作数。这种复杂度不适用于大型数据集，所以负采样是一个常见的解决方法，可以在与样本大小线性相关的时间内获得分布式表示。然而，负采样会改变损失函数，因此解决的是与最初提出的不同的优化问题。我们的贡献在于展示如何通过线性时间估计softmax归一化常数，从而设计了一种有效的优化策略来学习分布式表示。我们使用不同的数据集进行测试，并展示了我们的方法在嵌入质量和训练时间方面优于负采样。

    This article describes an efficient method to learn distributed representations, also known as embeddings. This is accomplished minimizing an objective function similar to the one introduced in the Word2Vec algorithm and later adopted in several works. The optimization computational bottleneck is the calculation of the softmax normalization constants for which a number of operations scaling quadratically with the sample size is required. This complexity is unsuited for large datasets and negative sampling is a popular workaround, allowing one to obtain distributed representations in linear time with respect to the sample size. Negative sampling consists, however, in a change of the loss function and hence solves a different optimization problem from the one originally proposed. Our contribution is to show that the sotfmax normalization constants can be estimated in linear time, allowing us to design an efficient optimization strategy to learn distributed representations. We test our ap
    

