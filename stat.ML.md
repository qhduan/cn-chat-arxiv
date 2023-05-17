# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Double Pessimism is Provably Efficient for Distributionally Robust Offline Reinforcement Learning: Generic Algorithm and Robust Partial Coverage.](http://arxiv.org/abs/2305.09659) | 本论文提出了一个名为P2MPO的算法框架，用于解决基于鲁棒离线RL的问题。该框架结合了灵活的模型估计子例程和双重悲观的策略优化步骤，采用双重悲观性原则以克服模型偏移等问题。研究表明，在模型准确性的假设下，该框架在拥有良好的鲁棒部分覆盖数据的情况下是具备高效性的。 |
| [^2] | [Balancing Risk and Reward: An Automated Phased Release Strategy.](http://arxiv.org/abs/2305.09626) | 本文提出了一种自动化的分阶段发布策略，能够在控制风险的同时最大化启动速度，通过一个受约束的分批赌博机问题模型来实现。 |
| [^3] | [The Power of Learned Locally Linear Models for Nonlinear Policy Optimization.](http://arxiv.org/abs/2305.09619) | 本文介绍了一种学习非线性系统动态的策略优化算法，该算法通过估计局部线性模型和执行类似于$\mathtt{iLQR}$的策略更新之间的迭代来实现，具有多项式的样本复杂度并克服了指数区间上的依赖性。 |
| [^4] | [Expressiveness Remarks for Denoising Diffusion Models and Samplers.](http://arxiv.org/abs/2305.09605) | 本文在漫扩扩散模型和采样器方面进行了表达能力的研究，通过将已知的神经网络逼近结果扩展到漫扩扩散模型和采样器来实现。 |
| [^5] | [Toward Falsifying Causal Graphs Using a Permutation-Based Test.](http://arxiv.org/abs/2305.09565) | 本文提出了一种通过构建节点置换基线的新型一致性度量方法，用于验证因果图的正确性并指导其在下游任务中的应用。 |
| [^6] | [Learning from Aggregated Data: Curated Bags versus Random Bags.](http://arxiv.org/abs/2305.09557) | 本文研究了两种自然的聚合方法：基于共同特征将数据点分组的精选包和将数据点随机分组的随机包，对于精选包设置和广泛的损失函数范围内，我们展示了可以通过梯度下降学习而不会导致数据聚合导致性能下降的情况。 |
| [^7] | [A Comparative Study of Methods for Estimating Conditional Shapley Values and When to Use Them.](http://arxiv.org/abs/2305.09536) | 本文研究了估计条件Shapley值的方法和应用场景，提出了新方法，扩展了之前的方法，并将这些方法分类，通过模拟研究评估了各类方法的精度和可靠性。 |
| [^8] | [Probabilistic Distance-Based Outlier Detection.](http://arxiv.org/abs/2305.09446) | 本文提出了一种将距离法异常检测分数转化为可解释的概率估计的通用方法，该方法使用与其他数据点的距离建模距离概率分布，将距离法异常检测分数转换为异常概率，提高了正常点和异常点之间的对比度，而不会影响检测性能。 |
| [^9] | [Lp- and Risk Consistency of Localized SVMs.](http://arxiv.org/abs/2305.09385) | 本文分析了局部支持向量机的一致性，证明了在非常弱的条件下，它们从全局SVM继承了$L_p$和风险一致性，即使底层区域随数据集大小的增加而变化。 |
| [^10] | [Errors-in-variables Fr\'echet Regression with Low-rank Covariate Approximation.](http://arxiv.org/abs/2305.09282) | 本论文提出了一种低秩协变量逼近的误差变量Frechet回归的方法，旨在提高回归估计器的效率和准确性，并实现在高维度和误差变量回归设置中更加有效的建模和估计。 |
| [^11] | [Transfer Causal Learning: Causal Effect Estimation with Knowledge Transfer.](http://arxiv.org/abs/2305.09126) | 本文提出了一个名为$\ell_1$-TCL的通用框架，它使用知识迁移和Lasso回归来提高因果效应估计精度。 |
| [^12] | [The Hessian perspective into the Nature of Convolutional Neural Networks.](http://arxiv.org/abs/2305.09088) | 本文基于Hessian映射，揭示了CNN结构和性质的本质，并证明了Hessian秩随着参数数量的增长而呈现出平方根增长。 |
| [^13] | [Convex optimization over a probability simplex.](http://arxiv.org/abs/2305.09046) | 这篇论文提出了一种新的迭代方案，用于求解概率单纯形上的凸优化问题。该方法具有收敛速度快且简单易行的特点。 |
| [^14] | [Scalable and Robust Tensor Ring Decomposition for Large-scale Data.](http://arxiv.org/abs/2305.09044) | 本文提出了一种可伸缩和健壮的张量环分解算法，能够自适应地填充缺失条目并识别异常值，在存储和计算复杂度上有显著降低，适用于处理大规模张量数据。 |
| [^15] | [SKI to go Faster: Accelerating Toeplitz Neural Networks via Asymmetric Kernels.](http://arxiv.org/abs/2305.09028) | 本论文提出使用非对称核（asymmetric kernels）实现Toeplitz神经网络（TNNs）的加速，通过稀疏加低秩Toeplitz矩阵分解、小型1D卷积和替换相对位置编码器（RPE）多层感知器（MLP）实现O（n）复杂度，针对因果模型，提出了“快速”因果屏蔽来抵消这种方法的限制。 |
| [^16] | [ELSA -- Enhanced latent spaces for improved collider simulations.](http://arxiv.org/abs/2305.07696) | 本文提出了多种增强模拟精确度的方法，包括在模拟链的末尾进行干预、在模拟链的开头进行干预和潜空间细化，通过 W+jets矩阵元代理模拟以及使用正则流和生成模型等策略进行研究，实验证明这些方法能显著提高精确度。 |
| [^17] | [Manifold Regularized Tucker Decomposition Approach for Spatiotemporal Traffic Data Imputation.](http://arxiv.org/abs/2305.06563) | 本文提出了一种基于流形正则化Tucker分解的时空交通数据填充方法，该方法利用稀疏正则化项改善了Tucker核的稀疏性，并引入流形正则化和时间约束项来优化张量的填充性能。 |
| [^18] | [LLT: An R package for Linear Law-based Feature Space Transformation.](http://arxiv.org/abs/2304.14211) | LLT是一个R包，用于线性定律特征空间变换，可以帮助对单变量和多变量时间序列进行分类。 |
| [^19] | [Classification of Superstatistical Features in High Dimensions.](http://arxiv.org/abs/2304.02912) | 本文利用经验风险最小化的方法，对高维超统计特征下的数据进行分类，并分析了正则化和分布尺度参数对分类的影响。 |
| [^20] | [Distributionally Robust Optimization using Cost-Aware Ambiguity Sets.](http://arxiv.org/abs/2303.09408) | 本文提出了一种新的用于分布鲁棒优化的模糊集，称为成本感知模糊集，它通过半空间定义，只排除那些预计对所获得的最坏情况成本产生重大影响的分布，实现了高置信度上界和一致估计。 |
| [^21] | [Synthetic Experience Replay.](http://arxiv.org/abs/2303.06614) | 本文提出了合成经验回放方法解决深度强化学习中数据匮乏问题，通过巧妙应用生成建模技术来扩充数据效果显著。 |
| [^22] | [Expressivity of Shallow and Deep Neural Networks for Polynomial Approximation.](http://arxiv.org/abs/2303.03544) | 本研究发现，浅层ReLU网络在表达具有随着输入维度增加的Lipschitz参数的函数时会遭受维度灾难，神经网络的表达能力更依赖于它们的深度而不是总体复杂度。 |
| [^23] | [Leveraging Demonstrations to Improve Online Learning: Quality Matters.](http://arxiv.org/abs/2302.03319) | 本篇论文探讨了离线演示数据如何改进在线学习的问题，提出了一种利用演示数据的TS算法，并给出了依赖于先验知识的贝叶斯遗憾界；研究发现，预训练可以大幅提高在线性能，改进程度随专家能力水平的提高而增加。 |
| [^24] | [Learning-Rate-Free Learning by D-Adaptation.](http://arxiv.org/abs/2301.07733) | D-Adaptation是一种可以自动设置学习率的方法，针对最小化凸性Lipschitz函数，用于实现最优收敛速率，而无需超参数，也无需额外对数因子改进，能够在各种机器学习问题中自动匹配手动调整的学习率。 |
| [^25] | [Combining datasets to increase the number of samples and improve model fitting.](http://arxiv.org/abs/2210.05165) | 本文提出了一种组合数据集的新框架ComImp，可以处理不同数据集之间存在不同特征的挑战，并利用PCA-ComImp进行维数降低。此外，此框架还可以用于数据预处理，填补缺失数据的条目。该方法在多个真实世界的数据集上得到了验证。 |
| [^26] | [Sample-and-Forward: Communication-Efficient Control of the False Discovery Rate in Networks.](http://arxiv.org/abs/2210.02555) | 该论文提出了通信高效的样本转发方法，可以在不同拓扑结构的网络中控制FDR，无需节点相互通信p值。方法经实验证明，拥有可证明的有限样本FDR控制和更强的检测功率。 |
| [^27] | [A moment-matching metric for latent variable generative models.](http://arxiv.org/abs/2111.00875) | 本文提出了一种用于比较和正则化潜变量生成模型的新型矩匹配度量方法，该方法通过研究数据矩和模型矩之间的差异来评估拟合模型质量。 |
| [^28] | [High-dimensional Inference for Dynamic Treatment Effects.](http://arxiv.org/abs/2110.04924) | 本文提出了一种新的 DR 方法，用于中间条件结果模型的 DR 表示，能够提供更优的稳健性保证，即使在面临高维混淆变量时也能实现一致性。 |
| [^29] | [MRCpy: A Library for Minimax Risk Classifiers.](http://arxiv.org/abs/2108.01952) | MRCpy是一种用于实现最小化风险分类器的Python库，它基于鲁棒风险最小化技术，可以利用0-1损失并提供了多种分类方法，其中一些提供了紧密的期望损失界限。 |
| [^30] | [Non-Parametric Manifold Learning.](http://arxiv.org/abs/2107.08089) | 该论文介绍了一种可求解紧致Riemann流形中距离的非参数估计器，并提出了该方法的一致性证明。该估计器是从Kontorovic对偶重构公式中的Connes距离公式推导而来。 |
| [^31] | [Graph neural networks-based Scheduler for Production planning problems using Reinforcement Learning.](http://arxiv.org/abs/2009.03836) | 本文提出了GraSP-RL框架，基于图神经网络来训练强化学习代理，以解决车间调度问题中状态空间难以处理、泛化能力较差的问题。 |
| [^32] | [Model Fusion via Optimal Transport.](http://arxiv.org/abs/1910.05653) | 本文提出一种基于最优传输的神经网络模型融合算法，能够成功地在不需要重新训练的情况下进行“单次”知识迁移，并且在独立同分布和非独立同分布的情况下比简单平均和集成模型更优。 |

# 详细

[^1]: 分布式鲁棒的离线强化学习：基于双重悲观性的通用算法和强健部分覆盖

    Double Pessimism is Provably Efficient for Distributionally Robust Offline Reinforcement Learning: Generic Algorithm and Robust Partial Coverage. (arXiv:2305.09659v1 [cs.LG])

    [http://arxiv.org/abs/2305.09659](http://arxiv.org/abs/2305.09659)

    本论文提出了一个名为P2MPO的算法框架，用于解决基于鲁棒离线RL的问题。该框架结合了灵活的模型估计子例程和双重悲观的策略优化步骤，采用双重悲观性原则以克服模型偏移等问题。研究表明，在模型准确性的假设下，该框架在拥有良好的鲁棒部分覆盖数据的情况下是具备高效性的。

    

    本文研究了分布式鲁棒的离线强化学习（鲁棒离线RL），其旨在从离线数据集中纯粹地找到一个能够在扰动环境中表现良好的最优强鲁棒策略。我们提出了一个名为P2MPO的算法框架，其中包含了灵活的模型估计子例程和双重悲观的策略优化步骤。双重悲观性原则对于克服由行为策略和目标策略家族之间的不匹配以及名义模型的扰动所引起的分布偏移至关重要。在对模型估计子例程进行一定准确性假设的情况下，我们证明了P2MPO算法在拥有良好的鲁棒部分覆盖数据的情况下是可证明有效的。

    We study distributionally robust offline reinforcement learning (robust offline RL), which seeks to find an optimal robust policy purely from an offline dataset that can perform well in perturbed environments. We propose a generic algorithm framework \underline{D}oubly \underline{P}essimistic \underline{M}odel-based \underline{P}olicy \underline{O}ptimization ($\texttt{P}^2\texttt{MPO}$) for robust offline RL, which features a novel combination of a flexible model estimation subroutine and a doubly pessimistic policy optimization step. The \emph{double pessimism} principle is crucial to overcome the distributional shift incurred by i) the mismatch between behavior policy and the family of target policies; and ii) the perturbation of the nominal model. Under certain accuracy assumptions on the model estimation subroutine, we show that $\texttt{P}^2\texttt{MPO}$ is provably efficient with \emph{robust partial coverage data}, which means that the offline dataset has good coverage of the d
    
[^2]: 平衡风险与收益：一种自动化的分阶段发布策略

    Balancing Risk and Reward: An Automated Phased Release Strategy. (arXiv:2305.09626v1 [stat.ML])

    [http://arxiv.org/abs/2305.09626](http://arxiv.org/abs/2305.09626)

    本文提出了一种自动化的分阶段发布策略，能够在控制风险的同时最大化启动速度，通过一个受约束的分批赌博机问题模型来实现。

    

    分阶段发布是科技行业中逐步发布新产品或更新的常见策略，通过一系列A/B测试，逐步增加处理单元的数量，直到完全部署或废弃。以原则性的方式执行分阶段发布需要以平衡不良影响的风险和迭代和快速学习的需求来选择分配给新发布的单位比例。在本文中，我们正式地阐述了这个问题，并提出了一种算法，在调度的每个阶段自动确定发布百分比，平衡控制风险和最大化启动速度的需求。我们的框架将这一挑战建模为一个受约束的分批赌博机问题，以确保我们预先指定的实验预算不会被高概率耗尽。我们提出的算法利用了自适应贝叶斯方法，其中将分配给处理的最大单元数由

    Phased releases are a common strategy in the technology industry for gradually releasing new products or updates through a sequence of A/B tests in which the number of treated units gradually grows until full deployment or deprecation. Performing phased releases in a principled way requires selecting the proportion of units assigned to the new release in a way that balances the risk of an adverse effect with the need to iterate and learn from the experiment rapidly. In this paper, we formalize this problem and propose an algorithm that automatically determines the release percentage at each stage in the schedule, balancing the need to control risk while maximizing ramp-up speed. Our framework models the challenge as a constrained batched bandit problem that ensures that our pre-specified experimental budget is not depleted with high probability. Our proposed algorithm leverages an adaptive Bayesian approach in which the maximal number of units assigned to the treatment is determined by
    
[^3]: 学习的局部线性模型在非线性策略优化中的威力

    The Power of Learned Locally Linear Models for Nonlinear Policy Optimization. (arXiv:2305.09619v1 [cs.LG])

    [http://arxiv.org/abs/2305.09619](http://arxiv.org/abs/2305.09619)

    本文介绍了一种学习非线性系统动态的策略优化算法，该算法通过估计局部线性模型和执行类似于$\mathtt{iLQR}$的策略更新之间的迭代来实现，具有多项式的样本复杂度并克服了指数区间上的依赖性。

    

    在基于学习的控制中，常见的流程是逐步估计系统动力学模型，并应用轨迹优化算法（例如$\mathtt{iLQR}$）在学习的模型上进行优化，以最小化目标成本。本文对一种简化版的此策略应用于一般非线性系统的情况进行了严格分析。我们分析了一种算法，该算法在估计非线性系统动态的局部线性模型和执行类似于$\mathtt{iLQR}$的策略更新之间进行迭代。我们证明该算法在相关问题参数中达到了多项式的样本复杂度，并通过合成局部稳定增益，克服了在问题区间上的指数依赖性。实验结果验证了我们算法的性能，并与自然的深度学习基线进行了比较。

    A common pipeline in learning-based control is to iteratively estimate a model of system dynamics, and apply a trajectory optimization algorithm e.g.~$\mathtt{iLQR}$ - on the learned model to minimize a target cost. This paper conducts a rigorous analysis of a simplified variant of this strategy for general nonlinear systems. We analyze an algorithm which iterates between estimating local linear models of nonlinear system dynamics and performing $\mathtt{iLQR}$-like policy updates. We demonstrate that this algorithm attains sample complexity polynomial in relevant problem parameters, and, by synthesizing locally stabilizing gains, overcomes exponential dependence in problem horizon. Experimental results validate the performance of our algorithm, and compare to natural deep-learning baselines.
    
[^4]: 漫扩扩散模型和采样器的表达能力研究

    Expressiveness Remarks for Denoising Diffusion Models and Samplers. (arXiv:2305.09605v1 [stat.ML])

    [http://arxiv.org/abs/2305.09605](http://arxiv.org/abs/2305.09605)

    本文在漫扩扩散模型和采样器方面进行了表达能力的研究，通过将已知的神经网络逼近结果扩展到漫扩扩散模型和采样器来实现。

    

    漫扩扩散模型是一类生成模型，在许多领域最近已经取得了最先进的结果。通过漫扩过程逐渐向数据中添加噪声，将数据分布转化为高斯分布。然后，通过模拟该漫扩的时间反演的逼近来获取生成模型的样本，刚开始这个漫扩模拟的初始值是高斯样本。最近的研究探索了将漫扩模型适应于采样和推断任务。本文基于众所周知的与F\"ollmer漂移类似的随机控制联系，将针对F\"ollmer漂移的已知神经网络逼近结果扩展到漫扩扩散模型和采样器。

    Denoising diffusion models are a class of generative models which have recently achieved state-of-the-art results across many domains. Gradual noise is added to the data using a diffusion process, which transforms the data distribution into a Gaussian. Samples from the generative model are then obtained by simulating an approximation of the time reversal of this diffusion initialized by Gaussian samples. Recent research has explored adapting diffusion models for sampling and inference tasks. In this paper, we leverage known connections to stochastic control akin to the F\"ollmer drift to extend established neural network approximation results for the F\"ollmer drift to denoising diffusion models and samplers.
    
[^5]: 基于置换检验的因果图假设验证方法

    Toward Falsifying Causal Graphs Using a Permutation-Based Test. (arXiv:2305.09565v1 [stat.ML])

    [http://arxiv.org/abs/2305.09565](http://arxiv.org/abs/2305.09565)

    本文提出了一种通过构建节点置换基线的新型一致性度量方法，用于验证因果图的正确性并指导其在下游任务中的应用。

    

    理解系统变量之间的因果关系对于解释和控制其行为至关重要。但是，从观察数据中推断因果图需要很多不总是现实的强假设。对于领域专家来说，很难表达因果图。因此，在将因果图用于下游任务之前，定量评估因果图的优劣的度量提供了有用的检查。现有的度量提供了一个绝对数量的因果图与观察数据之间的不一致性，而没有基础线，从业人员需要回答有多少这样的不一致性是可接受或预期的这一难题。在这里，我们提出了一种新的一致性度量方法，通过构建节点置换的替代基线。通过将不一致性的数量与替代基线上的数量进行比较，我们得出了一个可以解释的度量，捕捉有向无环图是否显著适合。

    Understanding the causal relationships among the variables of a system is paramount to explain and control its behaviour. Inferring the causal graph from observational data without interventions, however, requires a lot of strong assumptions that are not always realistic. Even for domain experts it can be challenging to express the causal graph. Therefore, metrics that quantitatively assess the goodness of a causal graph provide helpful checks before using it in downstream tasks. Existing metrics provide an absolute number of inconsistencies between the graph and the observed data, and without a baseline, practitioners are left to answer the hard question of how many such inconsistencies are acceptable or expected. Here, we propose a novel consistency metric by constructing a surrogate baseline through node permutations. By comparing the number of inconsistencies with those on the surrogate baseline, we derive an interpretable metric that captures whether the DAG fits significantly bet
    
[^6]: 大数据学习：精选包与随机包的对比研究

    Learning from Aggregated Data: Curated Bags versus Random Bags. (arXiv:2305.09557v1 [cs.LG])

    [http://arxiv.org/abs/2305.09557](http://arxiv.org/abs/2305.09557)

    本文研究了两种自然的聚合方法：基于共同特征将数据点分组的精选包和将数据点随机分组的随机包，对于精选包设置和广泛的损失函数范围内，我们展示了可以通过梯度下降学习而不会导致数据聚合导致性能下降的情况。

    

    保护用户隐私是许多机器学习系统部署的一个主要关注点，这些系统收集来自各种群体的数据。为了应对这种问题，一种方法是以聚合的形式收集和发布数据标签，从而可以将单个用户的信息与其他用户的信息组合起来。本文探讨了使用聚合数据标签而非单个标签来训练机器学习模型的可能性，具体来说，我们考虑了两种自然的聚合方法：基于共同特征将数据点分组的精选包和将数据点随机分组的随机包。对于精选包设置和广泛的损失函数范围内，我们展示了可以通过梯度下降学习而不会导致数据聚合导致性能下降的情况。我们的方法基于以下观察：损失函数的梯度之和可以表示为每个包的梯度的加权和，其中权重是包的大小。

    Protecting user privacy is a major concern for many machine learning systems that are deployed at scale and collect from a diverse set of population. One way to address this concern is by collecting and releasing data labels in an aggregated manner so that the information about a single user is potentially combined with others. In this paper, we explore the possibility of training machine learning models with aggregated data labels, rather than individual labels. Specifically, we consider two natural aggregation procedures suggested by practitioners: curated bags where the data points are grouped based on common features and random bags where the data points are grouped randomly in bag of similar sizes. For the curated bag setting and for a broad range of loss functions, we show that we can perform gradient-based learning without any degradation in performance that may result from aggregating data. Our method is based on the observation that the sum of the gradients of the loss functio
    
[^7]: 估计条件Shapley值的方法比较及其应用场景的研究

    A Comparative Study of Methods for Estimating Conditional Shapley Values and When to Use Them. (arXiv:2305.09536v1 [stat.ML])

    [http://arxiv.org/abs/2305.09536](http://arxiv.org/abs/2305.09536)

    本文研究了估计条件Shapley值的方法和应用场景，提出了新方法，扩展了之前的方法，并将这些方法分类，通过模拟研究评估了各类方法的精度和可靠性。

    

    Shapley值最早起源于合作博弈理论，但现在已经广泛应用于机器学习领域的模型无关解释框架中，用来解释复杂模型所做的预测。本文聚焦于预测模型的条件Shapley值的计算，探讨了不同的算法途径与应用场景，这些计算需要估计复杂的条件期望。文章提出了新的方法，扩展了之前提出的方法，并将这些方法分类、比较和评估。分类方式采用蒙特卡罗积分或回归对条件期望进行建模。作者通过广泛的模拟研究来衡量不同方法分类估计条件期望的精度和可靠性。

    Shapley values originated in cooperative game theory but are extensively used today as a model-agnostic explanation framework to explain predictions made by complex machine learning models in the industry and academia. There are several algorithmic approaches for computing different versions of Shapley value explanations. Here, we focus on conditional Shapley values for predictive models fitted to tabular data. Estimating precise conditional Shapley values is difficult as they require the estimation of non-trivial conditional expectations. In this article, we develop new methods, extend earlier proposed approaches, and systematize the new refined and existing methods into different method classes for comparison and evaluation. The method classes use either Monte Carlo integration or regression to model the conditional expectations. We conduct extensive simulation studies to evaluate how precisely the different method classes estimate the conditional expectations, and thereby the condit
    
[^8]: 概率距离法异常检测

    Probabilistic Distance-Based Outlier Detection. (arXiv:2305.09446v1 [cs.LG])

    [http://arxiv.org/abs/2305.09446](http://arxiv.org/abs/2305.09446)

    本文提出了一种将距离法异常检测分数转化为可解释的概率估计的通用方法，该方法使用与其他数据点的距离建模距离概率分布，将距离法异常检测分数转换为异常概率，提高了正常点和异常点之间的对比度，而不会影响检测性能。

    

    距离法异常检测方法的分数难以解释，因此在没有额外的上下文信息的情况下，很难确定正常点和异常点之间的截断阈值。我们描述了将距离法异常检测分数转化为可解释的概率估计的通用方法。该转换是排名稳定的，并增加了正常点和异常点之间的对比度。确定数据点之间的距离关系是识别数据中最近邻关系所必需的，然而大多数计算出的距离通常被丢弃。我们展示了可以使用与其他数据点的距离来建模距离概率分布，并随后使用这些分布将距离法异常检测分数转换为异常概率。我们的实验表明，概率转换不会影响众多表格和图像基准数据集上的检测性能，但会产生可解释性。

    The scores of distance-based outlier detection methods are difficult to interpret, making it challenging to determine a cut-off threshold between normal and outlier data points without additional context. We describe a generic transformation of distance-based outlier scores into interpretable, probabilistic estimates. The transformation is ranking-stable and increases the contrast between normal and outlier data points. Determining distance relationships between data points is necessary to identify the nearest-neighbor relationships in the data, yet, most of the computed distances are typically discarded. We show that the distances to other data points can be used to model distance probability distributions and, subsequently, use the distributions to turn distance-based outlier scores into outlier probabilities. Our experiments show that the probabilistic transformation does not impact detection performance over numerous tabular and image benchmark datasets but results in interpretable
    
[^9]: 局部支持向量机的$L_p$和风险一致性

    Lp- and Risk Consistency of Localized SVMs. (arXiv:2305.09385v1 [stat.ML])

    [http://arxiv.org/abs/2305.09385](http://arxiv.org/abs/2305.09385)

    本文分析了局部支持向量机的一致性，证明了在非常弱的条件下，它们从全局SVM继承了$L_p$和风险一致性，即使底层区域随数据集大小的增加而变化。

    

    基于核的正则化风险最小化器，又称为支持向量机（SVM），已知具有许多理想的属性，但在处理大型数据集时具有超线性的计算需求。可以通过使用局部SVM来解决这个问题，这种方法还提供了能够在不同的输入空间区域应用不同超参数的额外优势。本文分析了局部SVM的一致性。证明了它们在非常弱的情况下从全局SVM继承了$L_p$-以及风险-一致性，甚至可以在训练数据集大小增加时，允许底层的区域发生变化。

    Kernel-based regularized risk minimizers, also called support vector machines (SVMs), are known to possess many desirable properties but suffer from their super-linear computational requirements when dealing with large data sets. This problem can be tackled by using localized SVMs instead, which also offer the additional advantage of being able to apply different hyperparameters to different regions of the input space. In this paper, localized SVMs are analyzed with regards to their consistency. It is proven that they inherit $L_p$- as well as risk consistency from global SVMs under very weak conditions and even if the regions underlying the localized SVMs are allowed to change as the size of the training data set increases.
    
[^10]: 低秩协变量逼近的误差变量Frechet回归

    Errors-in-variables Fr\'echet Regression with Low-rank Covariate Approximation. (arXiv:2305.09282v1 [stat.ME])

    [http://arxiv.org/abs/2305.09282](http://arxiv.org/abs/2305.09282)

    本论文提出了一种低秩协变量逼近的误差变量Frechet回归的方法，旨在提高回归估计器的效率和准确性，并实现在高维度和误差变量回归设置中更加有效的建模和估计。

    

    Frechet回归已成为处理非欧几里得响应变量的回归分析的一种有前途的方法。然而，它依赖于理想情况下丰富和无噪声的协变量数据，因此其实际应用受到限制。本文提出了一种新的估计方法，通过利用协变量矩阵中固有的低秩结构来解决这些限制。我们的提出的框架结合了全局Frechet回归和主成分回归的概念，旨在提高回归估计器的效率和准确性。通过纳入低秩结构，我们的方法使得在高维度和误差变量回归设置中更加有效的建模和估计成为可能。我们对提议的估计器的大样本性质进行了理论分析，包括偏差、方差和由于测量误差引起的其他变化的全面率分析。此外，我们的数值实验验证了我们提出方法的性能。

    Fr\'echet regression has emerged as a promising approach for regression analysis involving non-Euclidean response variables. However, its practical applicability has been hindered by its reliance on ideal scenarios with abundant and noiseless covariate data. In this paper, we present a novel estimation method that tackles these limitations by leveraging the low-rank structure inherent in the covariate matrix. Our proposed framework combines the concepts of global Fr\'echet regression and principal component regression, aiming to improve the efficiency and accuracy of the regression estimator. By incorporating the low-rank structure, our method enables more effective modeling and estimation, particularly in high-dimensional and errors-in-variables regression settings. We provide a theoretical analysis of the proposed estimator's large-sample properties, including a comprehensive rate analysis of bias, variance, and additional variations due to measurement errors. Furthermore, our numeri
    
[^11]: 知识迁移下的因果效应估计: 转移因果学习

    Transfer Causal Learning: Causal Effect Estimation with Knowledge Transfer. (arXiv:2305.09126v1 [cs.LG])

    [http://arxiv.org/abs/2305.09126](http://arxiv.org/abs/2305.09126)

    本文提出了一个名为$\ell_1$-TCL的通用框架，它使用知识迁移和Lasso回归来提高因果效应估计精度。

    

    本文研究了一种新颖的问题，即在相同的协变量（或特征）空间设置下通过知识迁移来提高因果效应估计精度，即同类别迁移学习（TL），将其称为转移因果学习（TCL）问题。我们提出了一个通用的框架$\ell_1$-TCL，其中包含$\ell_1$正则化TL来进行苦事参数估计和下游插件ACE估计器，包括结果回归、逆概率加权和双重稳健估计器。最重要的是，借助于Lasso用于高维回归，我们建立了非渐近恢复保证。

    A novel problem of improving causal effect estimation accuracy with the help of knowledge transfer under the same covariate (or feature) space setting, i.e., homogeneous transfer learning (TL), is studied, referred to as the Transfer Causal Learning (TCL) problem. While most recent efforts in adapting TL techniques to estimate average causal effect (ACE) have been focused on the heterogeneous covariate space setting, those methods are inadequate for tackling the TCL problem since their algorithm designs are based on the decomposition into shared and domain-specific covariate spaces. To address this issue, we propose a generic framework called \texttt{$\ell_1$-TCL}, which incorporates $\ell_1$ regularized TL for nuisance parameter estimation and downstream plug-in ACE estimators, including outcome regression, inverse probability weighted, and doubly robust estimators. Most importantly, with the help of Lasso for high-dimensional regression, we establish non-asymptotic recovery guarantee
    
[^12]: 基于Hessian映射的卷积神经网络本质的新视角

    The Hessian perspective into the Nature of Convolutional Neural Networks. (arXiv:2305.09088v1 [cs.LG])

    [http://arxiv.org/abs/2305.09088](http://arxiv.org/abs/2305.09088)

    本文基于Hessian映射，揭示了CNN结构和性质的本质，并证明了Hessian秩随着参数数量的增长而呈现出平方根增长。

    

    尽管卷积神经网络(CNNs)一直被研究、应用和理论化，我们的目的是从它们的Hessian映射的角度提供一个稍微不同的观点，因为损失的Hessian捕捉了参数的成对交互，因此形成了一个自然的基础来探索CNN的架构方面如何表现出它的结构和性质。我们开发了一个依赖于CNN的Toeplitz表示的框架，并利用它来揭示Hessian结构，特别是它的秩。我们证明了紧密的上界（使用线性激活），它们紧密地遵循了Hessian秩的经验趋势，并在更一般的设置中保持在实践中。总的来说，我们的工作概括和确认了一个关键的洞见，即即使在CNNs中，Hessian秩随着参数数量的增长而呈现出平方根增长。

    While Convolutional Neural Networks (CNNs) have long been investigated and applied, as well as theorized, we aim to provide a slightly different perspective into their nature -- through the perspective of their Hessian maps. The reason is that the loss Hessian captures the pairwise interaction of parameters and therefore forms a natural ground to probe how the architectural aspects of CNN get manifested in its structure and properties. We develop a framework relying on Toeplitz representation of CNNs, and then utilize it to reveal the Hessian structure and, in particular, its rank. We prove tight upper bounds (with linear activations), which closely follow the empirical trend of the Hessian rank and hold in practice in more general settings. Overall, our work generalizes and establishes the key insight that, even in CNNs, the Hessian rank grows as the square root of the number of parameters.
    
[^13]: 概率单纯形上的凸优化

    Convex optimization over a probability simplex. (arXiv:2305.09046v1 [math.OC])

    [http://arxiv.org/abs/2305.09046](http://arxiv.org/abs/2305.09046)

    这篇论文提出了一种新的迭代方案，用于求解概率单纯形上的凸优化问题。该方法具有收敛速度快且简单易行的特点。

    

    我们提出了一种新的迭代方案——柯西单纯形来优化凸问题，使其满足概率单纯形上的限制条件，即$w\in\mathbb{R}^n$中$\sum_i w_i=1$，$w_i\geq0$。我们将单纯形映射到单位球的正四面体，通过梯度下降获得隐变量的解，并将结果映射回原始变量。该方法适用于高维问题，每次迭代由简单的操作组成，且针对凸函数证明了收敛速度为${O}(1/T)$。同时本文关注了信息理论（如交叉熵和KL散度）的应用。

    We propose a new iteration scheme, the Cauchy-Simplex, to optimize convex problems over the probability simplex $\{w\in\mathbb{R}^n\ |\ \sum_i w_i=1\ \textrm{and}\ w_i\geq0\}$. Other works have taken steps to enforce positivity or unit normalization automatically but never simultaneously within a unified setting. This paper presents a natural framework for manifestly requiring the probability condition. Specifically, we map the simplex to the positive quadrant of a unit sphere, envisage gradient descent in latent variables, and map the result back in a way that only depends on the simplex variable. Moreover, proving rigorous convergence results in this formulation leads inherently to tools from information theory (e.g. cross entropy and KL divergence). Each iteration of the Cauchy-Simplex consists of simple operations, making it well-suited for high-dimensional problems. We prove that it has a convergence rate of ${O}(1/T)$ for convex functions, and numerical experiments of projection 
    
[^14]: 大规模数据的可伸缩和健壮的张量环分解

    Scalable and Robust Tensor Ring Decomposition for Large-scale Data. (arXiv:2305.09044v1 [cs.LG])

    [http://arxiv.org/abs/2305.09044](http://arxiv.org/abs/2305.09044)

    本文提出了一种可伸缩和健壮的张量环分解算法，能够自适应地填充缺失条目并识别异常值，在存储和计算复杂度上有显著降低，适用于处理大规模张量数据。

    

    最近，张量环分解因其在高阶张量中的优越表现而受到越来越多的关注。然而，传统的张量环分解算法在面对大规模数据、缺失条目和异常值时，往往难以应用于现实世界的应用中。在本文中，我们提出了一种可扩展和健壮的张量环分解算法，能够处理大规模张量数据及其中的缺失条目和严重异常值。首先，我们开发了一种新颖的自适应加权最陡下降方法，在分解过程中能够自适应地填充缺失条目并识别异常值。此外，利用张量环模型，我们发展了一种新的快速 Gram 矩阵计算（FGMC）方法和一种随机子张量草图（RStS）策略，大幅度减少了存储和计算复杂度。实验结果表明，所提出的方法在张量环分解方面优于现有的方法。

    Tensor ring (TR) decomposition has recently received increased attention due to its superior expressive performance for high-order tensors. However, the applicability of traditional TR decomposition algorithms to real-world applications is hindered by prevalent large data sizes, missing entries, and corruption with outliers. In this work, we propose a scalable and robust TR decomposition algorithm capable of handling large-scale tensor data with missing entries and gross corruptions. We first develop a novel auto-weighted steepest descent method that can adaptively fill the missing entries and identify the outliers during the decomposition process. Further, taking advantage of the tensor ring model, we develop a novel fast Gram matrix computation (FGMC) approach and a randomized subtensor sketching (RStS) strategy which yield significant reduction in storage and computational complexity. Experimental results demonstrate that the proposed method outperforms existing TR decomposition met
    
[^15]: SKI加速Toeplitz神经网络：通过非对称核实现加速

    SKI to go Faster: Accelerating Toeplitz Neural Networks via Asymmetric Kernels. (arXiv:2305.09028v1 [stat.ML])

    [http://arxiv.org/abs/2305.09028](http://arxiv.org/abs/2305.09028)

    本论文提出使用非对称核（asymmetric kernels）实现Toeplitz神经网络（TNNs）的加速，通过稀疏加低秩Toeplitz矩阵分解、小型1D卷积和替换相对位置编码器（RPE）多层感知器（MLP）实现O（n）复杂度，针对因果模型，提出了“快速”因果屏蔽来抵消这种方法的限制。

    

    Toeplitz神经网络（TNNs）是最近出现并取得令人印象深刻结果的序列模型。它们需要O(n log n)的计算复杂度和O(n)的相对位置编码器（RPE）多层感知器（MLP）和衰减偏差调用。我们的目标是减少它们。我们首先指出，RPE是一个非对称正定核，而Toeplitz矩阵是伪格拉姆矩阵。此外：1）学习的核在主对角线附近显示出刺状行为，而在其他位置则表现出平滑行为；2）RPE MLP较慢。对于双向模型，这促使我们进行稀疏加低秩Toeplitz矩阵分解。对于稀疏组件的操作，我们进行小型1D卷积。对于低秩组件，我们将RPE MLP替换为线性插值，并使用非对称有结构的内核插值（SKI）（Wilson等，2015）以实现O（n）复杂度：我们提供了严格的误差分析。对于因果模型，“快速”因果屏蔽（Katharopoulos等，2020）抵消了SKI的好处。

    Toeplitz Neural Networks (TNNs) (Qin et. al. 2023) are a recent sequence model with impressive results. They require O(n log n) computational complexity and O(n) relative positional encoder (RPE) multi-layer perceptron (MLP) and decay bias calls. We aim to reduce both. We first note that the RPE is a non-SPD (symmetric positive definite) kernel and the Toeplitz matrices are pseudo-Gram matrices. Further 1) the learned kernels display spiky behavior near the main diagonals with otherwise smooth behavior; 2) the RPE MLP is slow. For bidirectional models, this motivates a sparse plus low-rank Toeplitz matrix decomposition. For the sparse component's action, we do a small 1D convolution. For the low rank component, we replace the RPE MLP with linear interpolation and use asymmetric Structured Kernel Interpolation (SKI) (Wilson et. al. 2015) for O(n) complexity: we provide rigorous error analysis. For causal models, "fast" causal masking (Katharopoulos et. al. 2020) negates SKI's benefits. 
    
[^16]: ELSA -- 提高碰撞模拟精确度的增强潜空间

    ELSA -- Enhanced latent spaces for improved collider simulations. (arXiv:2305.07696v1 [hep-ph] CROSS LISTED)

    [http://arxiv.org/abs/2305.07696](http://arxiv.org/abs/2305.07696)

    本文提出了多种增强模拟精确度的方法，包括在模拟链的末尾进行干预、在模拟链的开头进行干预和潜空间细化，通过 W+jets矩阵元代理模拟以及使用正则流和生成模型等策略进行研究，实验证明这些方法能显著提高精确度。

    

    模拟在碰撞物理中具有关键作用。我们探索机器学习增强模拟精确度的各种方法，包括在模拟链的末尾进行干预（重新加权）、在模拟链的开头进行干预（预处理）以及在末尾和开头之间建立联系（潜空间细化）。为了清晰地说明我们的方法，我们使用基于正则流的W+jets矩阵元代理模拟作为原型示例。首先，使用机器学习分类器在数据空间中确定权重。然后，我们将数据空间权重回推到潜空间以产生无权重的样本，并使用哈密顿蒙特卡罗使用潜空间细化（LASER）协议。另一种方法是增强正则流，该方法允许在潜空间和目标空间中具有不同的维度。我们研究了各种预处理策略，包括使用生成模型生成增强样本的新的通用方法。结果表明，这些方法显着提高了W+jets矩阵元模拟的精确度。

    Simulations play a key role for inference in collider physics. We explore various approaches for enhancing the precision of simulations using machine learning, including interventions at the end of the simulation chain (reweighting), at the beginning of the simulation chain (pre-processing), and connections between the end and beginning (latent space refinement). To clearly illustrate our approaches, we use W+jets matrix element surrogate simulations based on normalizing flows as a prototypical example. First, weights in the data space are derived using machine learning classifiers. Then, we pull back the data-space weights to the latent space to produce unweighted examples and employ the Latent Space Refinement (LASER) protocol using Hamiltonian Monte Carlo. An alternative approach is an augmented normalizing flow, which allows for different dimensions in the latent and target spaces. These methods are studied for various pre-processing strategies, including a new and general method f
    
[^17]: 基于流形正则化 Tucker 分解的时空交通数据填充方法

    Manifold Regularized Tucker Decomposition Approach for Spatiotemporal Traffic Data Imputation. (arXiv:2305.06563v1 [stat.ML])

    [http://arxiv.org/abs/2305.06563](http://arxiv.org/abs/2305.06563)

    本文提出了一种基于流形正则化Tucker分解的时空交通数据填充方法，该方法利用稀疏正则化项改善了Tucker核的稀疏性，并引入流形正则化和时间约束项来优化张量的填充性能。

    

    时空交通数据填充(STDI)是数据驱动智能交通系统中不可避免和具有挑战性的任务，在部分观测到的交通数据中估计丢失数据。由于交通数据具有多维和时空性质，我们将丢失数据填充视为张量完成问题。过去十年中，许多关于基于张量分解的 STDI 的研究已经展开。然而，如何利用时空相关性和核张量稀疏性来改善填充性能仍然需要解决。本文重新构造了3/4阶汉克尔张量，并提出了一种创新的流形正则化 Tucker 分解(maniRTD)模型用于STDI。明确地，我们通过引入多维延迟嵌入变换将传感交通状态数据表示为3/4阶张量。然后，ManiRTD使用稀疏正则化项改善了Tucker核的稀疏性，并使用流形正则化和时间约束项来优化张量的填充性能。

    Spatiotemporal traffic data imputation (STDI), estimating the missing data from partially observed traffic data, is an inevitable and challenging task in data-driven intelligent transportation systems (ITS). Due to traffic data's multidimensional and spatiotemporal properties, we treat the missing data imputation as a tensor completion problem. Many studies have been on STDI based on tensor decomposition in the past decade. However, how to use spatiotemporal correlations and core tensor sparsity to improve the imputation performance still needs to be solved. This paper reshapes a 3rd/4th order Hankel tensor and proposes an innovative manifold regularized Tucker decomposition (ManiRTD) model for STDI. Expressly, we represent the sensory traffic state data as the 3rd/4th tensors by introducing Multiway Delay Embedding Transforms. Then, ManiRTD improves the sparsity of the Tucker core using a sparse regularization term and employs manifold regularization and temporal constraint terms of f
    
[^18]: LLT：线性定律特征空间变换的R包

    LLT: An R package for Linear Law-based Feature Space Transformation. (arXiv:2304.14211v1 [cs.LG])

    [http://arxiv.org/abs/2304.14211](http://arxiv.org/abs/2304.14211)

    LLT是一个R包，用于线性定律特征空间变换，可以帮助对单变量和多变量时间序列进行分类。

    

    线性定律特征空间转换(LLT )算法的目标是帮助对单变量和多变量时间序列进行分类。LLT R包以灵活和用户友好的方式实现了该算法。该包将实例分为训练和测试集，并利用时延嵌入和谱分解技术，识别训练集中每个输入序列(初始特征)的控制模式(称为线性定律)。最后，它应用训练集的线性定律来转换测试集的初始特征。trainTest、trainLaw和testTrans三个单独的函数来执行这些步骤，它们需要预定义的数据结构;然而，为了快速计算，它们只使用内置函数。LLT R包和适当数据结构的示例数据集在GitHub上公开可用。

    The goal of the linear law-based feature space transformation (LLT) algorithm is to assist with the classification of univariate and multivariate time series. The presented R package, called LLT, implements this algorithm in a flexible yet user-friendly way. This package first splits the instances into training and test sets. It then utilizes time-delay embedding and spectral decomposition techniques to identify the governing patterns (called linear laws) of each input sequence (initial feature) within the training set. Finally, it applies the linear laws of the training set to transform the initial features of the test set. These steps are performed by three separate functions called trainTest, trainLaw, and testTrans. Their application requires a predefined data structure; however, for fast calculation, they use only built-in functions. The LLT R package and a sample dataset with the appropriate data structure are publicly available on GitHub.
    
[^19]: 高维超统计特征的分类方法

    Classification of Superstatistical Features in High Dimensions. (arXiv:2304.02912v1 [stat.ML])

    [http://arxiv.org/abs/2304.02912](http://arxiv.org/abs/2304.02912)

    本文利用经验风险最小化的方法，对高维超统计特征下的数据进行分类，并分析了正则化和分布尺度参数对分类的影响。

    

    在高维情况下，我们通过经验风险最小化的方法，对具有一般中心点的两个数据云的混合进行了学习，假设具有通用的凸损失和凸正则化。每个数据云是通过从可能是不可数的高斯分布叠加中进行采样来获得的，其方差具有通用的概率密度$\varrho$。我们的分析涵盖了大量的数据分布，包括没有协方差的幂律尾部分布的情况。我们研究了所得估计器的泛化性能，分析了正则化的作用以及分离转换与分布尺度参数的相关性。

    We characterise the learning of a mixture of two clouds of data points with generic centroids via empirical risk minimisation in the high dimensional regime, under the assumptions of generic convex loss and convex regularisation. Each cloud of data points is obtained by sampling from a possibly uncountable superposition of Gaussian distributions, whose variance has a generic probability density $\varrho$. Our analysis covers therefore a large family of data distributions, including the case of power-law-tailed distributions with no covariance. We study the generalisation performance of the obtained estimator, we analyse the role of regularisation, and the dependence of the separability transition on the distribution scale parameters.
    
[^20]: 使用成本感知的模糊集的分布鲁棒优化

    Distributionally Robust Optimization using Cost-Aware Ambiguity Sets. (arXiv:2303.09408v1 [math.OC])

    [http://arxiv.org/abs/2303.09408](http://arxiv.org/abs/2303.09408)

    本文提出了一种新的用于分布鲁棒优化的模糊集，称为成本感知模糊集，它通过半空间定义，只排除那些预计对所获得的最坏情况成本产生重大影响的分布，实现了高置信度上界和一致估计。

    

    我们提出了一种新的用于分布鲁棒优化（DRO）的模糊集类别，称为成本感知的模糊集，其定义为取决于在独立估计的最优解处评估成本函数的半空间，因此仅排除那些预计对所获得的最坏情况成本产生重大影响的分布。我们表明，由此产生的DRO方法提供了高置信度的上界和样本外预期成本的一致估计，并且经验证明，它与基于散度的模糊集相比，可以产生更少保守的解。

    We present a novel class of ambiguity sets for distributionally robust optimization (DRO). These ambiguity sets, called cost-aware ambiguity sets, are defined as halfspaces which depend on the cost function evaluated at an independent estimate of the optimal solution, thus excluding only those distributions that are expected to have significant impact on the obtained worst-case cost. We show that the resulting DRO method provides both a high-confidence upper bound and a consistent estimator of the out-of-sample expected cost, and demonstrate empirically that it results in less conservative solutions compared to divergence-based ambiguity sets.
    
[^21]: 合成经验回放：旨在用扩充数据来提高深度强化学习的效果

    Synthetic Experience Replay. (arXiv:2303.06614v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.06614](http://arxiv.org/abs/2303.06614)

    本文提出了合成经验回放方法解决深度强化学习中数据匮乏问题，通过巧妙应用生成建模技术来扩充数据效果显著。

    

    过去十年的一个关键主题是，当大型神经网络和大型数据集相结合时，它们可以产生令人惊异的结果。在深度强化学习中，这种范式通常通过经验回放实现，其中过去的经验数据集用于训练策略或值函数。然而，与监督学习或自监督学习不同，强化学习代理必须收集自己的数据，这通常是有限的。因此，利用深度学习的好处是具有挑战性的，即使是小型神经网络在训练开始时也可能出现过拟合现象。在这项工作中，我们利用了生成建模的巨大进步，并提出了合成经验回放（SynthER），一种基于扩散的方法来灵活地上采样代理收集的经验。我们证明了SynthER是一种有效的方法，可以在离线和在线设置下训练强化学习代理，无论是在感知环境还是在像素环境中。在离线设置中，我们观察到了显着的改进。

    A key theme in the past decade has been that when large neural networks and large datasets combine they can produce remarkable results. In deep reinforcement learning (RL), this paradigm is commonly made possible through experience replay, whereby a dataset of past experiences is used to train a policy or value function. However, unlike in supervised or self-supervised learning, an RL agent has to collect its own data, which is often limited. Thus, it is challenging to reap the benefits of deep learning, and even small neural networks can overfit at the start of training. In this work, we leverage the tremendous recent progress in generative modeling and propose Synthetic Experience Replay (SynthER), a diffusion-based approach to flexibly upsample an agent's collected experience. We show that SynthER is an effective method for training RL agents across offline and online settings, in both proprioceptive and pixel-based environments. In offline settings, we observe drastic improvements 
    
[^22]: 浅层神经网络和深层神经网络在多项式逼近中的表达能力。

    Expressivity of Shallow and Deep Neural Networks for Polynomial Approximation. (arXiv:2303.03544v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.03544](http://arxiv.org/abs/2303.03544)

    本研究发现，浅层ReLU网络在表达具有随着输入维度增加的Lipschitz参数的函数时会遭受维度灾难，神经网络的表达能力更依赖于它们的深度而不是总体复杂度。

    

    本研究探讨了要近似多元单项式所需的修正线性单元（ReLU）神经网络中神经元的数量。我们在一般紧致域上建立了任何浅层网络逼近乘积函数的指数下界。我们还证明了这个下界不适用于在单位立方体上的规范利普希茨单项式。这些发现表明，在表达具有随着输入维度增加的Lipschitz参数的函数时，浅层ReLU网络会遭受维度灾难，神经网络的表达能力更依赖于它们的深度而不是总体复杂度。

    This study explores the number of neurons required for a Rectified Linear Unit (ReLU) neural network to approximate multivariate monomials. We establish an exponential lower bound on the complexity of any shallow network approximating the product function over a general compact domain. We also demonstrate this lower bound doesn't apply to normalized Lipschitz monomials over the unit cube. These findings suggest that shallow ReLU networks experience the curse of dimensionality when expressing functions with a Lipschitz parameter scaling with the dimension of the input, and that the expressive power of neural networks is more dependent on their depth rather than overall complexity.
    
[^23]: 利用演示数据改进在线学习:质量至关重要

    Leveraging Demonstrations to Improve Online Learning: Quality Matters. (arXiv:2302.03319v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.03319](http://arxiv.org/abs/2302.03319)

    本篇论文探讨了离线演示数据如何改进在线学习的问题，提出了一种利用演示数据的TS算法，并给出了依赖于先验知识的贝叶斯遗憾界；研究发现，预训练可以大幅提高在线性能，改进程度随专家能力水平的提高而增加。

    

    我们研究了离线演示数据可以如何改进在线学习，自然而然地期望会有一定的改进，但问题在于如何改进以及可以改进多少？我们表明，改进的程度必须取决于演示数据的质量。为了生成可移植的见解，我们将重点放在了作为典型在线学习算法和模型的多臂赌博机上应用汤普森抽样（TS）。演示数据是由具有给定能力水平的专家生成的，这是我们引入的一个概念。我们提出了一种知情TS算法，通过贝叶斯定理以一致的方式利用演示数据并导出依赖于先验的贝叶斯遗憾界。这提供了洞见，即预训练如何极大地提高在线性能，以及改进程度随专家能力水平的提高而增加。我们还通过贝叶斯引导实现了实用的、近似的知情TS算法，并通过实验证明了实现了实质性的遗憾减少。

    We investigate the extent to which offline demonstration data can improve online learning. It is natural to expect some improvement, but the question is how, and by how much? We show that the degree of improvement must depend on the quality of the demonstration data. To generate portable insights, we focus on Thompson sampling (TS) applied to a multi-armed bandit as a prototypical online learning algorithm and model. The demonstration data is generated by an expert with a given competence level, a notion we introduce. We propose an informed TS algorithm that utilizes the demonstration data in a coherent way through Bayes' rule and derive a prior-dependent Bayesian regret bound. This offers insight into how pretraining can greatly improve online performance and how the degree of improvement increases with the expert's competence level. We also develop a practical, approximate informed TS algorithm through Bayesian bootstrapping and show substantial empirical regret reduction through exp
    
[^24]: 通过D适应实现学习率自由学习

    Learning-Rate-Free Learning by D-Adaptation. (arXiv:2301.07733v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.07733](http://arxiv.org/abs/2301.07733)

    D-Adaptation是一种可以自动设置学习率的方法，针对最小化凸性Lipschitz函数，用于实现最优收敛速率，而无需超参数，也无需额外对数因子改进，能够在各种机器学习问题中自动匹配手动调整的学习率。

    

    D适应是一种自动设置学习率的方法，可以渐近地实现最优收敛速率，用于最小化凸性Lipschitz函数，无需回溯或线性搜索，并且每步无需进行额外的函数值或梯度评估。我们的方法是这一类问题的第一个无超参数且收敛速率无需额外对数因子改进的方法。我们针对SGD和Adam变体展示了广泛的实验，其中该方法自动匹配手动调整的学习率，在十多个不同的机器学习问题中应用，包括大规模的视觉和语言问题。开源实现在 \url{https://github.com/facebookresearch/dadaptation}.

    D-Adaptation is an approach to automatically setting the learning rate which asymptotically achieves the optimal rate of convergence for minimizing convex Lipschitz functions, with no back-tracking or line searches, and no additional function value or gradient evaluations per step. Our approach is the first hyper-parameter free method for this class without additional multiplicative log factors in the convergence rate. We present extensive experiments for SGD and Adam variants of our method, where the method automatically matches hand-tuned learning rates across more than a dozen diverse machine learning problems, including large-scale vision and language problems.  An open-source implementation is available at \url{https://github.com/facebookresearch/dadaptation}.
    
[^25]: 合并数据集以增加样本数量并提高模型拟合

    Combining datasets to increase the number of samples and improve model fitting. (arXiv:2210.05165v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.05165](http://arxiv.org/abs/2210.05165)

    本文提出了一种组合数据集的新框架ComImp，可以处理不同数据集之间存在不同特征的挑战，并利用PCA-ComImp进行维数降低。此外，此框架还可以用于数据预处理，填补缺失数据的条目。该方法在多个真实世界的数据集上得到了验证。

    

    在许多使用情况下，将来自不同数据集的信息组合起来可以提高机器学习模型的性能，特别是当至少一个数据集的样本数量很少时。然而，在这种情况下可能存在一个潜在的挑战，即这些数据集的特征不完全相同，尽管它们中有一些共同的特征。为了解决这个挑战，我们提出了一种新的框架，称为基于插补的数据集组合（ComImp）。此外，我们提出了一种ComImp的变体，使用主成分分析（PCA），即PCA-ComImp，以在组合数据集之前降低维数。当数据集拥有大量不共享特征时，这非常有用。此外，我们的框架还可以用于数据预处理，即通过在组合不同数据集时填补缺失的条目来插补缺失数据。为了说明所提出的方法的能力和潜在用途，我们在多个真实世界的数据集上进行了广泛实验。

    For many use cases, combining information from different datasets can be of interest to improve a machine learning model's performance, especially when the number of samples from at least one of the datasets is small. However, a potential challenge in such cases is that the features from these datasets are not identical, even though there are some commonly shared features among the datasets. To tackle this challenge, we propose a novel framework called Combine datasets based on Imputation (ComImp). In addition, we propose a variant of ComImp that uses Principle Component Analysis (PCA), PCA-ComImp in order to reduce dimension before combining datasets. This is useful when the datasets have a large number of features that are not shared between them. Furthermore, our framework can also be utilized for data preprocessing by imputing missing data, i.e., filling in the missing entries while combining different datasets. To illustrate the power of the proposed methods and their potential us
    
[^26]: 《样本转发：网络中基于样本转发的虚警率控制》

    Sample-and-Forward: Communication-Efficient Control of the False Discovery Rate in Networks. (arXiv:2210.02555v2 [eess.SP] UPDATED)

    [http://arxiv.org/abs/2210.02555](http://arxiv.org/abs/2210.02555)

    该论文提出了通信高效的样本转发方法，可以在不同拓扑结构的网络中控制FDR，无需节点相互通信p值。方法经实验证明，拥有可证明的有限样本FDR控制和更强的检测功率。

    

    本论文旨在提出一种在通信限制下控制网络中虚警率（FDR）的方法。我们提出了一种灵活且通信高效的样本转发方法，它是Benjamini-Hochberg（BH）程序在具有一般拓扑结构的多跳网络中的版本。我们的方法证明了，为了在全局FDR控制约束下获得良好的统计功率，网络中的节点不需要将p值相互通信。考虑到一个总共有$m$个p值的网络，我们的方法首先对每个节点进行样本抽取，获取p值的经验CDF，然后向其邻居节点转发$\mathcal{O}(\log m)$位。在与原始BH程序相同的假设条件下，我们的方法具有可证明的有限样本FDR控制以及竞争性的经验检测功率，即使每个节点只有少量样本数据。我们提供了在p值混合模型假设下的功率渐近分析。

    This work concerns controlling the false discovery rate (FDR) in networks under communication constraints. We present sample-and-forward, a flexible and communication-efficient version of the Benjamini-Hochberg (BH) procedure for multihop networks with general topologies. Our method evidences that the nodes in a network do not need to communicate p-values to each other to achieve a decent statistical power under the global FDR control constraint. Consider a network with a total of $m$ p-values, our method consists of first sampling the (empirical) CDF of the p-values at each node and then forwarding $\mathcal{O}(\log m)$ bits to its neighbors. Under the same assumptions as for the original BH procedure, our method has both the provable finite-sample FDR control as well as competitive empirical detection power, even with a few samples at each node. We provide an asymptotic analysis of power under a mixture model assumption on the p-values.
    
[^27]: 一种用于潜变量生成模型的矩匹配度量方法

    A moment-matching metric for latent variable generative models. (arXiv:2111.00875v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2111.00875](http://arxiv.org/abs/2111.00875)

    本文提出了一种用于比较和正则化潜变量生成模型的新型矩匹配度量方法，该方法通过研究数据矩和模型矩之间的差异来评估拟合模型质量。

    

    面对无监督学习问题时，评估拟合模型的质量是困难的。潜变量模型，如变分自编码器和高斯混合模型，通常使用基于似然的方法进行训练。本文提出了一种新的用于模型比较或正则化的度量方法，该方法依赖于矩。其概念是使用矩范数（如弗罗贝尼乌斯范数）研究数据矩和模型矩之间的差异。我们展示了如何使用这个新的度量方法进行模型比较和正则化，并证明了该方法的可行性。

    It can be difficult to assess the quality of a fitted model when facing unsupervised learning problems. Latent variable models, such as variation autoencoders and Gaussian mixture models, are often trained with likelihood-based approaches. In scope of Goodhart's law, when a metric becomes a target it ceases to be a good metric and therefore we should not use likelihood to assess the quality of the fit of these models. The solution we propose is a new metric for model comparison or regularization that relies on moments. The concept is to study the difference between the data moments and the model moments using a matrix norm, such as the Frobenius norm. We show how to use this new metric for model comparison and then for regularization. It is common to draw samples from the fitted distribution when evaluating latent variable models and we show that our proposed metric is faster to compute and has a smaller variance that this alternative. We conclude this article with a proof of concept o
    
[^28]: 高维推断下的动态治疗效应估计

    High-dimensional Inference for Dynamic Treatment Effects. (arXiv:2110.04924v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2110.04924](http://arxiv.org/abs/2110.04924)

    本文提出了一种新的 DR 方法，用于中间条件结果模型的 DR 表示，能够提供更优的稳健性保证，即使在面临高维混淆变量时也能实现一致性。

    

    在因果推断中，估计动态治疗效应是一项重要的任务，尤其是当面临高维混淆变量时。双重稳健 (DR) 方法因其灵活性而成为估计治疗效应的有力工具。然而，我们展示了仅关注预期结果的 DR 传统方法可能无法提供最优结果。在本文中，我们提出了一种新的 DR 方法，用于中间条件结果模型的 DR 表示，从而提供了更优的稳健性保证。只要每个暴露时间和治疗路径都恰当地参数化了至少一个辅助函数，所提出的方法即使在面临高维混淆变量时也能实现一致性。我们的结果代表了一个重大的进步，因为它们提供了新的稳健性保证。实现这些结果的关键是我们的新 DR 表示，它在需要比以前的方法更弱的假设条件下提供了更好的推断性能。

    Estimating dynamic treatment effects is a crucial endeavor in causal inference, particularly when confronted with high-dimensional confounders. Doubly robust (DR) approaches have emerged as promising tools for estimating treatment effects due to their flexibility. However, we showcase that the traditional DR approaches that only focus on the DR representation of the expected outcomes may fall short of delivering optimal results. In this paper, we propose a novel DR representation for intermediate conditional outcome models that leads to superior robustness guarantees. The proposed method achieves consistency even with high-dimensional confounders, as long as at least one nuisance function is appropriately parametrized for each exposure time and treatment path. Our results represent a significant step forward as they provide new robustness guarantees. The key to achieving these results is our new DR representation, which offers superior inferential performance while requiring weaker ass
    
[^29]: MRCpy：一种用于最小化风险分类器的库

    MRCpy: A Library for Minimax Risk Classifiers. (arXiv:2108.01952v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2108.01952](http://arxiv.org/abs/2108.01952)

    MRCpy是一种用于实现最小化风险分类器的Python库，它基于鲁棒风险最小化技术，可以利用0-1损失并提供了多种分类方法，其中一些提供了紧密的期望损失界限。

    

    目前现有的监督分类库都是基于经验风险最小化和使用代理损失技术的。本文介绍MRCpy库，该库实现了基于鲁棒风险最小化的最小化风险分类器（MRC），并可利用0-1损失。这种技术产生了许多分类方法，可以提供紧密的期望损失界限。MRCpy为不同变量的MRC提供了统一的接口，并遵循流行Python库的标准。此外，MRCpy还提供了实现一些流行技术的功能，这些技术可以看作是MRC，例如L1正则化逻辑回归，0-1对抗性和最大熵机。此外，MRCpy还实现了最近的特征映射，如傅里叶，ReLU和阈值特征。该库采用面向对象的方法设计，方便协作者和用户。

    Existing libraries for supervised classification implement techniques that are based on empirical risk minimization and utilize surrogate losses. We present MRCpy library that implements minimax risk classifiers (MRCs) that are based on robust risk minimization and can utilize 0-1-loss. Such techniques give rise to a manifold of classification methods that can provide tight bounds on the expected loss. MRCpy provides a unified interface for different variants of MRCs and follows the standards of popular Python libraries. The presented library also provides implementation for popular techniques that can be seen as MRCs such as L1-regularized logistic regression, zero-one adversarial, and maximum entropy machines. In addition, MRCpy implements recent feature mappings such as Fourier, ReLU, and threshold features. The library is designed with an object-oriented approach that facilitates collaborators and users.
    
[^30]: 非参数流形学习

    Non-Parametric Manifold Learning. (arXiv:2107.08089v3 [math.ST] UPDATED)

    [http://arxiv.org/abs/2107.08089](http://arxiv.org/abs/2107.08089)

    该论文介绍了一种可求解紧致Riemann流形中距离的非参数估计器，并提出了该方法的一致性证明。该估计器是从Kontorovic对偶重构公式中的Connes距离公式推导而来。

    

    我们介绍了一种基于图拉普拉斯估计的Laplace-Beltrami算子的估计器，用于一个紧致的Riemann流形中的距离。我们通过图拉普拉斯估计的谱误差和几何特性来上界估计流形距离的误差，或者更准确地说是近年来在非交换几何中被关注的一类流形距离。此结果推出了（未截取的）流形距离的一致性证明。该估计器与Wasserstein距离的Kontorovic对偶重构公式中的Connes距离公式是相似的，事实上是从后者的收敛性质推导出来的。

    We introduce an estimator for distances in a compact Riemannian manifold based on graph Laplacian estimates of the Laplace-Beltrami operator. We upper bound the error in the estimate of manifold distances, or more precisely an estimate of a spectrally truncated variant of manifold distance of interest in non-commutative geometry (cf. [Connes and Suijelekom, 2020]), in terms of spectral errors in the graph Laplacian estimates and, implicitly, several geometric properties of the manifold. A consequence is a proof of consistency for (untruncated) manifold distances. The estimator resembles, and in fact its convergence properties are derived from, a special case of the Kontorovic dual reformulation of Wasserstein distance known as Connes' Distance Formula.
    
[^31]: 基于图神经网络的强化学习生产计划问题调度器

    Graph neural networks-based Scheduler for Production planning problems using Reinforcement Learning. (arXiv:2009.03836v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2009.03836](http://arxiv.org/abs/2009.03836)

    本文提出了GraSP-RL框架，基于图神经网络来训练强化学习代理，以解决车间调度问题中状态空间难以处理、泛化能力较差的问题。

    

    强化学习在车间调度问题中得到广泛应用。但对于车间调度问题，强化学习通常使用矢量化机器特征作为状态空间。这种方法存在三个主要问题：（1）机器单元和作业序列之间的关系没有完全捕获，（2）状态空间随着机器/作业数量的增加呈指数增长，（3）代理的泛化能力存在问题。本文提出了一种新的框架——GraSP-RL，基于图神经网络的强化学习生产计划问题调度器。它将车间调度问题表示为图形，并使用图神经网络(GNN)提取特征来训练强化学习代理。虽然图形本身在非欧几里德空间中，但使用GNN提取的特征在欧几里德空间中提供了当前生产状态的丰富编码，然后被强化学习代理用于选择下一个作业。此外，我们将调度问题视为一个基于图神经网络的流问题，使用回溯方法对此进行求解。

    Reinforcement learning (RL) is increasingly adopted in job shop scheduling problems (JSSP). But RL for JSSP is usually done using a vectorized representation of machine features as the state space. It has three major problems: (1) the relationship between the machine units and the job sequence is not fully captured, (2) exponential increase in the size of the state space with increasing machines/jobs, and (3) the generalization of the agent to unseen scenarios. We present a novel framework - GraSP-RL, GRAph neural network-based Scheduler for Production planning problems using Reinforcement Learning. It represents JSSP as a graph and trains the RL agent using features extracted using a graph neural network (GNN). While the graph is itself in the non-euclidean space, the features extracted using the GNNs provide a rich encoding of the current production state in the euclidean space, which is then used by the RL agent to select the next job. Further, we cast the scheduling problem as a de
    
[^32]: 基于最优传输的模型融合方法

    Model Fusion via Optimal Transport. (arXiv:1910.05653v6 [cs.LG] UPDATED)

    [http://arxiv.org/abs/1910.05653](http://arxiv.org/abs/1910.05653)

    本文提出一种基于最优传输的神经网络模型融合算法，能够成功地在不需要重新训练的情况下进行“单次”知识迁移，并且在独立同分布和非独立同分布的情况下比简单平均和集成模型更优。

    

    在机器学习应用中，结合不同的模型是一个广泛使用的范例。尽管最常见的方法是形成一个模型集合并平均它们的各自预测，但由于资源限制（以内存和计算的方式呈线性增长于模型数），这种方法常常无法实现。我们提出了一种基于层次的神经网络模型融合算法，利用最优传输来（软）对齐模型中的神经元，然后平均它们的相关参数。我们展示了这可以在异构非独立同分布数据上，成功地实现“单次”知识迁移（即不需要任何重新训练）在神经网络之间。在独立同分布和非独立同分布的情况下，我们示范了我们的方法明显优于简单平均以及如何在标准卷积网络（如VGG11）、残差网络上进行快速优化替代集成模型。

    Combining different models is a widely used paradigm in machine learning applications. While the most common approach is to form an ensemble of models and average their individual predictions, this approach is often rendered infeasible by given resource constraints in terms of memory and computation, which grow linearly with the number of models. We present a layer-wise model fusion algorithm for neural networks that utilizes optimal transport to (soft-) align neurons across the models before averaging their associated parameters.  We show that this can successfully yield "one-shot" knowledge transfer (i.e, without requiring any retraining) between neural networks trained on heterogeneous non-i.i.d. data. In both i.i.d. and non-i.i.d. settings , we illustrate that our approach significantly outperforms vanilla averaging, as well as how it can serve as an efficient replacement for the ensemble with moderate fine-tuning, for standard convolutional networks (like VGG11), residual networks
    

