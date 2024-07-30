# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Wasserstein perspective of Vanilla GANs](https://arxiv.org/abs/2403.15312) | 将普通GANs与水斯坦距离联系起来，扩展现有水斯坦GANs结果到普通GANs，获得了普通GANs的神谕不等式。 |
| [^2] | [Statistical Test for Generated Hypotheses by Diffusion Models](https://arxiv.org/abs/2402.11789) | 本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。 |
| [^3] | [Adaptive maximization of social welfare.](http://arxiv.org/abs/2310.09597) | 论文研究了通过适应性策略选择最大化社会福利的问题，并提供了关于遗憾的下界和算法的匹配上界。研究发现福利最大化比多臂老虎机问题更困难，但该算法达到了最优增长速率。 |
| [^4] | [Meta-Learning Operators to Optimality from Multi-Task Non-IID Data.](http://arxiv.org/abs/2308.04428) | 本文提出了从多任务非独立同分布数据中恢复线性操作符的方法，并发现现有的各向同性无关的元学习方法会对表示更新造成偏差，限制了表示学习的样本复杂性。为此，引入了去偏差和特征白化的适应方法。 |
| [^5] | [Structural restrictions in local causal discovery: identifying direct causes of a target variable.](http://arxiv.org/abs/2307.16048) | 这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。 |
| [^6] | [On Consistency of Signatures Using Lasso.](http://arxiv.org/abs/2305.10413) | 本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。 |
| [^7] | [Dropout Regularization in Extended Generalized Linear Models based on Double Exponential Families.](http://arxiv.org/abs/2305.06625) | 本论文研究了基于双指数族的扩展广义线性模型中的dropout正则化，dropout正则化偏好罕见但重要的特征，在均值和离散度方面都具有普适性。 |
| [^8] | [Generalization with quantum geometry for learning unitaries.](http://arxiv.org/abs/2303.13462) | 本文研究了量子机器学习模型的泛化能力，使用数据的量子费舍尔信息度量来评估成功训练和泛化所需的电路参数和训练数据的数量，并展示通过去除对称性来提高泛化能力，同时发现超出分布泛化能力可以比使用相同分布更优。 |
| [^9] | [Multitask Learning and Bandits via Robust Statistics.](http://arxiv.org/abs/2112.14233) | 本研究探讨了多任务学习以及Bandits方法的健壮统计学实现，提出了一种新颖的两阶段多任务学习估计器，该估计器以一种样本高效的方式利用共享全局参数和稀疏实例特定术语的结构。 |

# 详细

[^1]: 水斯坦视角下的普通 GANs

    A Wasserstein perspective of Vanilla GANs

    [https://arxiv.org/abs/2403.15312](https://arxiv.org/abs/2403.15312)

    将普通GANs与水斯坦距离联系起来，扩展现有水斯坦GANs结果到普通GANs，获得了普通GANs的神谕不等式。

    

    生成对抗网络(GANs)的实证成功引起了对理论研究日益增长的兴趣。统计文献主要集中在水斯坦GANs及其扩展上，特别是允许具有良好的降维特性。对于普通GANs，即原始优化问题，统计结果仍然相当有限，需要假设平滑激活函数和潜空间与周围空间的维度相等。为了弥合这一差距，我们将普通GANs与水斯坦距离联系起来。通过这样做，现有的水斯坦GANs结果可以扩展到普通GANs。特别是，在水斯坦距离中获得了普通GANs的神谕不等式。这个神谕不等式的假设旨在由实践中常用的网络架构满足，如前馈ReLU网络。

    arXiv:2403.15312v1 Announce Type: cross  Abstract: The empirical success of Generative Adversarial Networks (GANs) caused an increasing interest in theoretical research. The statistical literature is mainly focused on Wasserstein GANs and generalizations thereof, which especially allow for good dimension reduction properties. Statistical results for Vanilla GANs, the original optimization problem, are still rather limited and require assumptions such as smooth activation functions and equal dimensions of the latent space and the ambient space. To bridge this gap, we draw a connection from Vanilla GANs to the Wasserstein distance. By doing so, existing results for Wasserstein GANs can be extended to Vanilla GANs. In particular, we obtain an oracle inequality for Vanilla GANs in Wasserstein distance. The assumptions of this oracle inequality are designed to be satisfied by network architectures commonly used in practice, such as feedforward ReLU networks. By providing a quantitative resu
    
[^2]: 通过扩散模型生成的假设的统计检验

    Statistical Test for Generated Hypotheses by Diffusion Models

    [https://arxiv.org/abs/2402.11789](https://arxiv.org/abs/2402.11789)

    本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。

    

    AI的增强性能加速了其融入科学研究。特别是，利用生成式AI创建科学假设是很有前途的，并且正在越来越多地应用于各个领域。然而，当使用AI生成的假设进行关键决策（如医学诊断）时，验证它们的可靠性至关重要。在本研究中，我们考虑使用扩散模型生成的图像进行医学诊断任务，并提出了一种统计检验来量化其可靠性。所提出的统计检验的基本思想是使用选择性推断框架，我们考虑在生成的图像是由经过训练的扩散模型产生的这一事实条件下的统计检验。利用所提出的方法，医学图像诊断结果的统计可靠性可以以p值的形式量化，从而实现在控制错误率的情况下进行决策。

    arXiv:2402.11789v1 Announce Type: cross  Abstract: The enhanced performance of AI has accelerated its integration into scientific research. In particular, the use of generative AI to create scientific hypotheses is promising and is increasingly being applied across various fields. However, when employing AI-generated hypotheses for critical decisions, such as medical diagnoses, verifying their reliability is crucial. In this study, we consider a medical diagnostic task using generated images by diffusion models, and propose a statistical test to quantify its reliability. The basic idea behind the proposed statistical test is to employ a selective inference framework, where we consider a statistical test conditional on the fact that the generated images are produced by a trained diffusion model. Using the proposed method, the statistical reliability of medical image diagnostic results can be quantified in the form of a p-value, allowing for decision-making with a controlled error rate. 
    
[^3]: 自适应最大化社会福利

    Adaptive maximization of social welfare. (arXiv:2310.09597v1 [econ.EM])

    [http://arxiv.org/abs/2310.09597](http://arxiv.org/abs/2310.09597)

    论文研究了通过适应性策略选择最大化社会福利的问题，并提供了关于遗憾的下界和算法的匹配上界。研究发现福利最大化比多臂老虎机问题更困难，但该算法达到了最优增长速率。

    

    我们考虑了重复选择政策以最大化社会福利的问题。福利是个人效用和公共收入的加权和。早期的结果影响后续的政策选择。效用不可观测，但可以间接推断。响应函数通过实验学习获得。我们推导出了一个关于遗憾的下界，并且对于一种Exp3算法的匹配对策对立上界。累积遗憾以$T^{2/3}$的速率增长。这意味着(i)福利最大化比多臂老虎机问题更困难（对于有限的政策集来说，增长速率为$T^{1/2}$），和(ii)我们的算法实现了最优增长速率。对于随机设置，如果社会福利是凹的，我们可以使用二分搜索算法在连续政策集上实现$T^{1/2}$的速率。我们分析了非线性收入税扩展，并概述了商品税扩展。我们将我们的设置与垄断定价（更容易）和双边交易的定价进行了比较。

    We consider the problem of repeatedly choosing policies to maximize social welfare. Welfare is a weighted sum of private utility and public revenue. Earlier outcomes inform later policies. Utility is not observed, but indirectly inferred. Response functions are learned through experimentation.  We derive a lower bound on regret, and a matching adversarial upper bound for a variant of the Exp3 algorithm. Cumulative regret grows at a rate of $T^{2/3}$. This implies that (i) welfare maximization is harder than the multi-armed bandit problem (with a rate of $T^{1/2}$ for finite policy sets), and (ii) our algorithm achieves the optimal rate. For the stochastic setting, if social welfare is concave, we can achieve a rate of $T^{1/2}$ (for continuous policy sets), using a dyadic search algorithm.  We analyze an extension to nonlinear income taxation, and sketch an extension to commodity taxation. We compare our setting to monopoly pricing (which is easier), and price setting for bilateral tra
    
[^4]: 从多任务非独立同分布数据中元学习操作符到最优性

    Meta-Learning Operators to Optimality from Multi-Task Non-IID Data. (arXiv:2308.04428v1 [stat.ML])

    [http://arxiv.org/abs/2308.04428](http://arxiv.org/abs/2308.04428)

    本文提出了从多任务非独立同分布数据中恢复线性操作符的方法，并发现现有的各向同性无关的元学习方法会对表示更新造成偏差，限制了表示学习的样本复杂性。为此，引入了去偏差和特征白化的适应方法。

    

    机器学习中最近取得进展的一个强大概念是从异构来源或任务的数据中提取共同特征。直观地说，将所有数据用于学习共同的表示函数，既有助于计算效率，又有助于统计泛化，因为它可以减少要在给定任务上进行微调的参数数量。为了在理论上做出这些优点的根源，我们提出了从噪声向量测量$y = Mx + w$中回复线性操作符$M$的一般模型。其中，协变量$x$既可以是非独立同分布的，也可以是非各向同性的。我们证明了现有的各向同性无关的元学习方法会对表示更新造成偏差，这导致噪声项的缩放不再有利于源任务数量。这反过来会导致表示学习的样本复杂性受到单任务数据规模的限制。我们引入了一种方法，称为去偏差和特征白化。

    A powerful concept behind much of the recent progress in machine learning is the extraction of common features across data from heterogeneous sources or tasks. Intuitively, using all of one's data to learn a common representation function benefits both computational effort and statistical generalization by leaving a smaller number of parameters to fine-tune on a given task. Toward theoretically grounding these merits, we propose a general setting of recovering linear operators $M$ from noisy vector measurements $y = Mx + w$, where the covariates $x$ may be both non-i.i.d. and non-isotropic. We demonstrate that existing isotropy-agnostic meta-learning approaches incur biases on the representation update, which causes the scaling of the noise terms to lose favorable dependence on the number of source tasks. This in turn can cause the sample complexity of representation learning to be bottlenecked by the single-task data size. We introduce an adaptation, $\texttt{De-bias & Feature-Whiten}
    
[^5]: 局部因果发现中的结构限制: 识别目标变量的直接原因

    Structural restrictions in local causal discovery: identifying direct causes of a target variable. (arXiv:2307.16048v1 [stat.ME])

    [http://arxiv.org/abs/2307.16048](http://arxiv.org/abs/2307.16048)

    这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。

    

    我们考虑从观察联合分布中学习目标变量的一组直接原因的问题。学习表示因果结构的有向无环图(DAG)是科学中的一个基本问题。当完整的DAG从分布中可识别时，已知有一些结果，例如假设非线性高斯数据生成过程。通常，我们只对识别一个目标变量的直接原因（局部因果结构），而不是完整的DAG感兴趣。在本文中，我们讨论了对目标变量的数据生成过程的不同假设，该假设下直接原因集合可以从分布中识别出来。在这样做的过程中，我们对除目标变量之外的变量基本上没有任何假设。除了新的可识别性结果，我们还提供了两种从有限随机样本估计直接原因的实用算法，并在几个基准数据集上证明了它们的有效性。

    We consider the problem of learning a set of direct causes of a target variable from an observational joint distribution. Learning directed acyclic graphs (DAGs) that represent the causal structure is a fundamental problem in science. Several results are known when the full DAG is identifiable from the distribution, such as assuming a nonlinear Gaussian data-generating process. Often, we are only interested in identifying the direct causes of one target variable (local causal structure), not the full DAG. In this paper, we discuss different assumptions for the data-generating process of the target variable under which the set of direct causes is identifiable from the distribution. While doing so, we put essentially no assumptions on the variables other than the target variable. In addition to the novel identifiability results, we provide two practical algorithms for estimating the direct causes from a finite random sample and demonstrate their effectiveness on several benchmark dataset
    
[^6]: 使用Lasso的签名一致性研究

    On Consistency of Signatures Using Lasso. (arXiv:2305.10413v1 [stat.ML])

    [http://arxiv.org/abs/2305.10413](http://arxiv.org/abs/2305.10413)

    本文重新审视了Lasso回归对于签名变换的一致性问题，并发现对于不同的过程和时间序列，选择适当的签名定义和随机模型可以提高Lasso回归的一致性。

    

    签名变换是连续和离散时间序列数据的迭代路径积分，它们的普遍非线性通过线性化特征选择问题。本文在理论和数值上重新审视了Lasso回归对于签名变换的一致性问题。我们的研究表明，对于更接近布朗运动或具有较弱跨维度相关性的过程和时间序列，签名定义为It\^o积分的Lasso回归更具一致性；对于均值回归过程和时间序列，其签名定义为Stratonovich积分在Lasso回归中具有更高的一致性。我们的发现强调了在统计推断和机器学习中选择适当的签名和随机模型的重要性。

    Signature transforms are iterated path integrals of continuous and discrete-time time series data, and their universal nonlinearity linearizes the problem of feature selection. This paper revisits the consistency issue of Lasso regression for the signature transform, both theoretically and numerically. Our study shows that, for processes and time series that are closer to Brownian motion or random walk with weaker inter-dimensional correlations, the Lasso regression is more consistent for their signatures defined by It\^o integrals; for mean reverting processes and time series, their signatures defined by Stratonovich integrals have more consistency in the Lasso regression. Our findings highlight the importance of choosing appropriate definitions of signatures and stochastic models in statistical inference and machine learning.
    
[^7]: 基于双指数族的扩展广义线性模型中的Dropout正则化

    Dropout Regularization in Extended Generalized Linear Models based on Double Exponential Families. (arXiv:2305.06625v1 [stat.ML])

    [http://arxiv.org/abs/2305.06625](http://arxiv.org/abs/2305.06625)

    本论文研究了基于双指数族的扩展广义线性模型中的dropout正则化，dropout正则化偏好罕见但重要的特征，在均值和离散度方面都具有普适性。

    

    尽管dropout是一种流行的正则化技术，但其理论性质尚未被充分理解。本文研究了基于双指数族的扩展广义线性模型中的dropout正则化，其中离散参数可以随特征变化。理论分析表明，dropout正则化偏好罕见但重要的特征，在均值和离散度方面都具有普适性，这扩展了之前针对传统广义线性模型的结果 。采用自适应学习率的随机梯度下降进行训练。为了说明这一点，我们将dropout应用于自适应B样条平滑，其中均值和离散度参数都被灵活地建模。重要的B样条基础函数可以被认为是罕见的特征，我们在实验中证实，dropout是一种改善了罚最大似然方法的显式平滑性的均值和离散度参数的有效正则化形式。

    Even though dropout is a popular regularization technique, its theoretical properties are not fully understood. In this paper we study dropout regularization in extended generalized linear models based on double exponential families, for which the dispersion parameter can vary with the features. A theoretical analysis shows that dropout regularization prefers rare but important features in both the mean and dispersion, generalizing an earlier result for conventional generalized linear models. Training is performed using stochastic gradient descent with adaptive learning rate. To illustrate, we apply dropout to adaptive smoothing with B-splines, where both the mean and dispersion parameters are modelled flexibly. The important B-spline basis functions can be thought of as rare features, and we confirm in experiments that dropout is an effective form of regularization for mean and dispersion parameters that improves on a penalized maximum likelihood approach with an explicit smoothness p
    
[^8]: 利用量子几何进行学习幺正变换的泛化

    Generalization with quantum geometry for learning unitaries. (arXiv:2303.13462v1 [quant-ph])

    [http://arxiv.org/abs/2303.13462](http://arxiv.org/abs/2303.13462)

    本文研究了量子机器学习模型的泛化能力，使用数据的量子费舍尔信息度量来评估成功训练和泛化所需的电路参数和训练数据的数量，并展示通过去除对称性来提高泛化能力，同时发现超出分布泛化能力可以比使用相同分布更优。

    

    泛化是量子机器学习模型从训练数据学习准确预测新数据的能力。在这里，我们引入数据的量子费舍尔信息度量(DQFIM)来确定模型何时能够泛化。对于幺正变换的可变学习，DQFIM量化了成功训练和泛化所需的电路参数和训练数据的数量。我们应用DQFIM来解释何时恒定数量的训练状态和多项式数量的参数足以实现泛化。此外，通过从训练数据中删除对称性，可以提高泛化能力。最后，我们显示，使用不同数据分布进行训练和测试的超出分布泛化能力可以比使用相同分布的能力更优。我们的研究为提高量子机器学习中的泛化能力开辟了新的方法。

    Generalization is the ability of quantum machine learning models to make accurate predictions on new data by learning from training data. Here, we introduce the data quantum Fisher information metric (DQFIM) to determine when a model can generalize. For variational learning of unitaries, the DQFIM quantifies the amount of circuit parameters and training data needed to successfully train and generalize. We apply the DQFIM to explain when a constant number of training states and polynomial number of parameters are sufficient for generalization. Further, we can improve generalization by removing symmetries from training data. Finally, we show that out-of-distribution generalization, where training and testing data are drawn from different data distributions, can be better than using the same distribution. Our work opens up new approaches to improve generalization in quantum machine learning.
    
[^9]: 多任务学习和Bandits通过健壮统计学

    Multitask Learning and Bandits via Robust Statistics. (arXiv:2112.14233v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2112.14233](http://arxiv.org/abs/2112.14233)

    本研究探讨了多任务学习以及Bandits方法的健壮统计学实现，提出了一种新颖的两阶段多任务学习估计器，该估计器以一种样本高效的方式利用共享全局参数和稀疏实例特定术语的结构。

    

    决策者经常同时面对许多相关但异质的学习问题。在此工作中，我们研究了一种自然的设置，其中每个学习实例中的未知参数可以分解为共享全局参数加上稀疏的实例特定术语。我们提出了一种新颖的两阶段多任务学习估计器，以一种样本高效的方式利用这种结构，使用健壮统计学（在相似实例上学习）和LASSO回归（去偏差结果）的独特组合。我们的估计器提供了改进的样本复杂度界限。

    Decision-makers often simultaneously face many related but heterogeneous learning problems. For instance, a large retailer may wish to learn product demand at different stores to solve pricing or inventory problems, making it desirable to learn jointly for stores serving similar customers; alternatively, a hospital network may wish to learn patient risk at different providers to allocate personalized interventions, making it desirable to learn jointly for hospitals serving similar patient populations. Motivated by real datasets, we study a natural setting where the unknown parameter in each learning instance can be decomposed into a shared global parameter plus a sparse instance-specific term. We propose a novel two-stage multitask learning estimator that exploits this structure in a sample-efficient way, using a unique combination of robust statistics (to learn across similar instances) and LASSO regression (to debias the results). Our estimator yields improved sample complexity bound
    

