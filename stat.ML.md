# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Functional Bilevel Optimization for Machine Learning](https://arxiv.org/abs/2403.20233) | 介绍了机器学习中的函数双层优化问题，提出了不依赖于强凸假设的方法，并展示了在仪表回归和强化学习任务中使用神经网络的优势。 |
| [^2] | [Auditing Fairness under Unobserved Confounding](https://arxiv.org/abs/2403.14713) | 在未观测混杂因素的情况下，本文展示了即使在放宽或甚至在排除所有相关风险因素被观测到的假设的情况下，仍然可以给出对高风险个体分配率的信息丰富的界限。 |
| [^3] | [Early Directional Convergence in Deep Homogeneous Neural Networks for Small Initializations](https://arxiv.org/abs/2403.08121) | 本文研究了训练深度齐次神经网络时梯度流动力学的动态性，发现在足够小的初始化下，神经网络的权重在训练早期阶段保持较小规范，并且沿着神经相关函数的KKT点方向近似收敛。 |
| [^4] | [RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval](https://arxiv.org/abs/2402.18510) | 本文研究了RNNs和Transformer在处理算法问题时的表现能力差距，发现RNNs存在关键瓶颈，即无法完美地从上下文中检索信息，导致无法像Transformer那样轻松解决需要这种能力的任务。 |
| [^5] | [Learning Interpretable Concepts: Unifying Causal Representation Learning and Foundation Models](https://arxiv.org/abs/2402.09236) | 本研究将因果表示学习和基础模型相结合，研究了如何从数据中学习人类可解释的概念。实验证明了这一统一方法的实用性。 |
| [^6] | [Dynamic Incremental Optimization for Best Subset Selection](https://arxiv.org/abs/2402.02322) | 本文研究了一类$\ell_0$正则化问题的对偶形式，并提出了一种高效的原对偶算法，通过充分利用对偶范围估计和增量策略，提高了最佳子集选择问题的解决方案的效率和统计性质。 |
| [^7] | [A flexible Bayesian g-formula for causal survival analyses with time-dependent confounding](https://arxiv.org/abs/2402.02306) | 本文提出了一种更灵活的贝叶斯g形式估计器，用于具有时变混杂的因果生存分析。它采用贝叶斯附加回归树来模拟时变生成组件，并引入了纵向平衡分数以降低模型错误规范引起的偏差。 |
| [^8] | [Controlling Multiple Errors Simultaneously with a PAC-Bayes Bound](https://arxiv.org/abs/2202.05560) | 该研究提出了一种PAC-Bayes界限，能够同时控制多个错误，并提供丰富的信息，适用于回归中测试损失分布或分类中不同错误分类的概率。 |
| [^9] | [Comparative Study of Causal Discovery Methods for Cyclic Models with Hidden Confounders.](http://arxiv.org/abs/2401.13009) | 对于循环模型中含有隐藏因变量的因果发现，已经出现了能够处理这种情况的多种技术方法。 |
| [^10] | [Statistical Tests for Replacing Human Decision Makers with Algorithms.](http://arxiv.org/abs/2306.11689) | 本文提出了一种利用人工智能改善人类决策的统计框架，通过基准测试与机器预测，替换部分人类决策者的决策制定，并经过实验检验得出算法具有更高的真阳性率和更低的假阳性率，尤其是来自农村地区的医生的诊断更容易被替代。 |

# 详细

[^1]: 机器学习中的函数双层优化

    Functional Bilevel Optimization for Machine Learning

    [https://arxiv.org/abs/2403.20233](https://arxiv.org/abs/2403.20233)

    介绍了机器学习中的函数双层优化问题，提出了不依赖于强凸假设的方法，并展示了在仪表回归和强化学习任务中使用神经网络的优势。

    

    在本文中，我们介绍了针对机器学习中的双层优化问题的一种新的函数视角，其中内部目标在函数空间上被最小化。这些类型的问题通常通过在参数设置下开发的方法来解决，其中内部目标对于预测函数的参数强凸。函数视角不依赖于此假设，特别允许使用超参数化的神经网络作为内部预测函数。我们提出了可扩展和高效的算法来解决函数双层优化问题，并展示了我们方法在适合自然函数双层结构的仪表回归和强化学习任务上的优势。

    arXiv:2403.20233v1 Announce Type: cross  Abstract: In this paper, we introduce a new functional point of view on bilevel optimization problems for machine learning, where the inner objective is minimized over a function space. These types of problems are most often solved by using methods developed in the parametric setting, where the inner objective is strongly convex with respect to the parameters of the prediction function. The functional point of view does not rely on this assumption and notably allows using over-parameterized neural networks as the inner prediction function. We propose scalable and efficient algorithms for the functional bilevel optimization problem and illustrate the benefits of our approach on instrumental regression and reinforcement learning tasks, which admit natural functional bilevel structures.
    
[^2]: 在未观测混杂因素下审计公平性

    Auditing Fairness under Unobserved Confounding

    [https://arxiv.org/abs/2403.14713](https://arxiv.org/abs/2403.14713)

    在未观测混杂因素的情况下，本文展示了即使在放宽或甚至在排除所有相关风险因素被观测到的假设的情况下，仍然可以给出对高风险个体分配率的信息丰富的界限。

    

    决策系统中的一个基本问题是跨越人口统计线存在不公平性。然而，不公平性可能难以量化，特别是如果我们对公平性的理解依赖于难以衡量的风险等观念（例如，对于那些没有其治疗就会死亡的人平等获得治疗）。审计这种不公平性需要准确测量个体风险，而在未观测混杂的现实环境中，难以估计。在这些未观测到的因素“解释”明显差异的情况下，我们可能低估或高估不公平性。在本文中，我们展示了即使在放宽或（令人惊讶地）甚至在排除所有相关风险因素被观测到的假设的情况下，仍然可以对高风险个体的分配率给出信息丰富的界限。我们利用了在许多实际环境中（例如引入新型治疗）我们拥有在任何分配之前的数据的事实。

    arXiv:2403.14713v1 Announce Type: cross  Abstract: A fundamental problem in decision-making systems is the presence of inequity across demographic lines. However, inequity can be difficult to quantify, particularly if our notion of equity relies on hard-to-measure notions like risk (e.g., equal access to treatment for those who would die without it). Auditing such inequity requires accurate measurements of individual risk, which is difficult to estimate in the realistic setting of unobserved confounding. In the case that these unobservables "explain" an apparent disparity, we may understate or overstate inequity. In this paper, we show that one can still give informative bounds on allocation rates among high-risk individuals, even while relaxing or (surprisingly) even when eliminating the assumption that all relevant risk factors are observed. We utilize the fact that in many real-world settings (e.g., the introduction of a novel treatment) we have data from a period prior to any alloc
    
[^3]: 早期方向性收敛在深度齐次神经网络中进行小初始化时的分析

    Early Directional Convergence in Deep Homogeneous Neural Networks for Small Initializations

    [https://arxiv.org/abs/2403.08121](https://arxiv.org/abs/2403.08121)

    本文研究了训练深度齐次神经网络时梯度流动力学的动态性，发现在足够小的初始化下，神经网络的权重在训练早期阶段保持较小规范，并且沿着神经相关函数的KKT点方向近似收敛。

    

    本文研究了训练深度齐次神经网络时梯度流动力学的动态性，这些网络从小初始化开始。本文考虑到具有局部Lipschitz梯度和阶数严格大于两的神经网络。文章证明了对于足够小的初始化，在训练的早期阶段，神经网络的权重保持规范较小，并且在Karush-Kuhn-Tucker (KKT)点处近似沿着神经相关函数的方向收敛。此外，对于平方损失并在神经网络权重上进行可分离假设的情况下，还展示了在损失函数的某些鞍点附近梯度流动动态的类似方向性收敛。

    arXiv:2403.08121v1 Announce Type: new  Abstract: This paper studies the gradient flow dynamics that arise when training deep homogeneous neural networks, starting with small initializations. The present work considers neural networks that are assumed to have locally Lipschitz gradients and an order of homogeneity strictly greater than two. This paper demonstrates that for sufficiently small initializations, during the early stages of training, the weights of the neural network remain small in norm and approximately converge in direction along the Karush-Kuhn-Tucker (KKT) points of the neural correlation function introduced in [1]. Additionally, for square loss and under a separability assumption on the weights of neural networks, a similar directional convergence of gradient flow dynamics is shown near certain saddle points of the loss function.
    
[^4]: RNNs还不是Transformer：在上下文检索中的关键瓶颈

    RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval

    [https://arxiv.org/abs/2402.18510](https://arxiv.org/abs/2402.18510)

    本文研究了RNNs和Transformer在处理算法问题时的表现能力差距，发现RNNs存在关键瓶颈，即无法完美地从上下文中检索信息，导致无法像Transformer那样轻松解决需要这种能力的任务。

    

    本文探讨循环神经网络（RNNs）和Transformer在解决算法问题时的表示能力差距。我们重点关注RNNs是否能在处理长序列时，通过Chain-of-Thought (CoT)提示，与Transformer的性能相匹配。我们的理论分析显示CoT可以改进RNNs，但无法弥补与Transformer之间的差距。关键瓶颈在于RNNs无法完全从上下文中检索信息，即使经过CoT的增强：对于几个明确或隐式需要这种能力的任务，如联想召回和确定图是否为树，我们证明RNNs表达能力不足以解决这些任务，而Transformer可以轻松解决。相反，我们证明采用增强RNNs上下文检索能力的技术，包括

    arXiv:2402.18510v1 Announce Type: cross  Abstract: This paper investigates the gap in representation powers of Recurrent Neural Networks (RNNs) and Transformers in the context of solving algorithmic problems. We focus on understanding whether RNNs, known for their memory efficiency in handling long sequences, can match the performance of Transformers, particularly when enhanced with Chain-of-Thought (CoT) prompting. Our theoretical analysis reveals that CoT improves RNNs but is insufficient to close the gap with Transformers. A key bottleneck lies in the inability of RNNs to perfectly retrieve information from the context, even with CoT: for several tasks that explicitly or implicitly require this capability, such as associative recall and determining if a graph is a tree, we prove that RNNs are not expressive enough to solve the tasks while Transformers can solve them with ease. Conversely, we prove that adopting techniques to enhance the in-context retrieval capability of RNNs, inclu
    
[^5]: 学习可解释概念：统一因果表示学习与基础模型

    Learning Interpretable Concepts: Unifying Causal Representation Learning and Foundation Models

    [https://arxiv.org/abs/2402.09236](https://arxiv.org/abs/2402.09236)

    本研究将因果表示学习和基础模型相结合，研究了如何从数据中学习人类可解释的概念。实验证明了这一统一方法的实用性。

    

    构建智能机器学习系统有两种广泛的方法。一种方法是构建天生可解释的模型，这是因果表示学习领域的努力方向。另一种方法是构建高性能的基础模型，然后投入努力去理解它们的工作原理。本研究将这两种方法联系起来，研究如何从数据中学习人类可解释的概念。通过结合这两个领域的思想，我们正式定义了概念的概念，并展示了它们可以从多样的数据中被可靠地恢复出来。对于合成数据和大型语言模型的实验证明了我们统一方法的实用性。

    arXiv:2402.09236v1 Announce Type: cross Abstract: To build intelligent machine learning systems, there are two broad approaches. One approach is to build inherently interpretable models, as endeavored by the growing field of causal representation learning. The other approach is to build highly-performant foundation models and then invest efforts into understanding how they work. In this work, we relate these two approaches and study how to learn human-interpretable concepts from data. Weaving together ideas from both fields, we formally define a notion of concepts and show that they can be provably recovered from diverse data. Experiments on synthetic data and large language models show the utility of our unified approach.
    
[^6]: 动态增量优化用于最佳子集选择

    Dynamic Incremental Optimization for Best Subset Selection

    [https://arxiv.org/abs/2402.02322](https://arxiv.org/abs/2402.02322)

    本文研究了一类$\ell_0$正则化问题的对偶形式，并提出了一种高效的原对偶算法，通过充分利用对偶范围估计和增量策略，提高了最佳子集选择问题的解决方案的效率和统计性质。

    

    最佳子集选择被认为是稀疏学习问题的“黄金标准”。已经提出了各种优化技术来攻击这个非光滑非凸问题。本文研究了一类$\ell_0$正则化问题的对偶形式。基于原始问题和对偶问题的结构，我们提出了一种高效的原对偶算法。通过充分利用对偶范围估计和增量策略，我们的算法潜在地减少了冗余计算并改进了最佳子集选择的解决方案。理论分析和对合成和真实数据集的实验验证了所提出解决方案的效率和统计性质。

    Best subset selection is considered the `gold standard' for many sparse learning problems. A variety of optimization techniques have been proposed to attack this non-smooth non-convex problem. In this paper, we investigate the dual forms of a family of $\ell_0$-regularized problems. An efficient primal-dual algorithm is developed based on the primal and dual problem structures. By leveraging the dual range estimation along with the incremental strategy, our algorithm potentially reduces redundant computation and improves the solutions of best subset selection. Theoretical analysis and experiments on synthetic and real-world datasets validate the efficiency and statistical properties of the proposed solutions.
    
[^7]: 弹性贝叶斯g形式在具有时变混杂的因果生存分析中的应用

    A flexible Bayesian g-formula for causal survival analyses with time-dependent confounding

    [https://arxiv.org/abs/2402.02306](https://arxiv.org/abs/2402.02306)

    本文提出了一种更灵活的贝叶斯g形式估计器，用于具有时变混杂的因果生存分析。它采用贝叶斯附加回归树来模拟时变生成组件，并引入了纵向平衡分数以降低模型错误规范引起的偏差。

    

    在具有时间至事件结果的纵向观察性研究中，因果分析的常见目标是在研究群体中估计在假设干预情景下的因果生存曲线。g形式是这种分析的一个特别有用的工具。为了增强传统的参数化g形式方法，我们开发了一种更灵活的贝叶斯g形式估计器。该估计器同时支持纵向预测和因果推断。它在模拟时变生成组件的建模中引入了贝叶斯附加回归树，旨在减轻由于模型错误规范造成的偏差。具体而言，我们引入了一类更通用的离散生存数据g形式。这些公式可以引入纵向平衡分数，这在处理越来越多的时变混杂因素时是一种有效的降维方法。

    In longitudinal observational studies with a time-to-event outcome, a common objective in causal analysis is to estimate the causal survival curve under hypothetical intervention scenarios within the study cohort. The g-formula is a particularly useful tool for this analysis. To enhance the traditional parametric g-formula approach, we developed a more adaptable Bayesian g-formula estimator. This estimator facilitates both longitudinal predictive and causal inference. It incorporates Bayesian additive regression trees in the modeling of the time-evolving generative components, aiming to mitigate bias due to model misspecification. Specifically, we introduce a more general class of g-formulas for discrete survival data. These formulas can incorporate the longitudinal balancing scores, which serve as an effective method for dimension reduction and are vital when dealing with an expanding array of time-varying confounders. The minimum sufficient formulation of these longitudinal balancing
    
[^8]: 使用PAC-Bayes界限同时控制多个错误

    Controlling Multiple Errors Simultaneously with a PAC-Bayes Bound

    [https://arxiv.org/abs/2202.05560](https://arxiv.org/abs/2202.05560)

    该研究提出了一种PAC-Bayes界限，能够同时控制多个错误，并提供丰富的信息，适用于回归中测试损失分布或分类中不同错误分类的概率。

    

    当前的PAC-Bayes泛化界限仅限于性能的标量度量，如损失或错误率。我们提供了第一个能够提供丰富信息的PAC-Bayes界限，通过界定一组M种错误类型的经验概率与真实概率之间的Kullback-Leibler差异来控制可能结果的整个分布。

    arXiv:2202.05560v2 Announce Type: replace-cross  Abstract: Current PAC-Bayes generalisation bounds are restricted to scalar metrics of performance, such as the loss or error rate. However, one ideally wants more information-rich certificates that control the entire distribution of possible outcomes, such as the distribution of the test loss in regression, or the probabilities of different mis classifications. We provide the first PAC-Bayes bound capable of providing such rich information by bounding the Kullback-Leibler divergence between the empirical and true probabilities of a set of M error types, which can either be discretized loss values for regression, or the elements of the confusion matrix (or a partition thereof) for classification. We transform our bound into a differentiable training objective. Our bound is especially useful in cases where the severity of different mis-classifications may change over time; existing PAC-Bayes bounds can only bound a particular pre-decided w
    
[^9]: 循环模型中含有隐藏因变量的因果发现方法的比较研究

    Comparative Study of Causal Discovery Methods for Cyclic Models with Hidden Confounders. (arXiv:2401.13009v1 [cs.LG])

    [http://arxiv.org/abs/2401.13009](http://arxiv.org/abs/2401.13009)

    对于循环模型中含有隐藏因变量的因果发现，已经出现了能够处理这种情况的多种技术方法。

    

    如今，对因果发现的需求无处不在。理解系统中部分之间的随机依赖性以及实际的因果关系对科学的各个部分都至关重要。因此，寻找可靠的方法来检测因果方向的需求不断增长。在过去的50年里，出现了许多因果发现算法，但大多数仅适用于系统没有反馈环路并且具有因果充分性的假设，即没有未测量的子系统能够影响多个已测量变量。这是不幸的，因为这些限制在实践中往往不能假定。反馈是许多过程的一个重要特性，现实世界的系统很少是完全隔离和完全测量的。幸运的是，在最近几年中，已经发展了几种能够处理循环的、因果不充分的系统的技术。随着多种方法的出现，一种实际的应用方法开始变得可能。

    Nowadays, the need for causal discovery is ubiquitous. A better understanding of not just the stochastic dependencies between parts of a system, but also the actual cause-effect relations, is essential for all parts of science. Thus, the need for reliable methods to detect causal directions is growing constantly. In the last 50 years, many causal discovery algorithms have emerged, but most of them are applicable only under the assumption that the systems have no feedback loops and that they are causally sufficient, i.e. that there are no unmeasured subsystems that can affect multiple measured variables. This is unfortunate since those restrictions can often not be presumed in practice. Feedback is an integral feature of many processes, and real-world systems are rarely completely isolated and fully measured. Fortunately, in recent years, several techniques, that can cope with cyclic, causally insufficient systems, have been developed. And with multiple methods available, a practical ap
    
[^10]: 统计测试替代人类决策者的算法

    Statistical Tests for Replacing Human Decision Makers with Algorithms. (arXiv:2306.11689v1 [econ.EM])

    [http://arxiv.org/abs/2306.11689](http://arxiv.org/abs/2306.11689)

    本文提出了一种利用人工智能改善人类决策的统计框架，通过基准测试与机器预测，替换部分人类决策者的决策制定，并经过实验检验得出算法具有更高的真阳性率和更低的假阳性率，尤其是来自农村地区的医生的诊断更容易被替代。

    

    本文提出了一个统计框架，可以通过人工智能来改善人类的决策。首先将每个人类决策者的表现与机器预测进行基准测试；然后用所提出的人工智能算法的建议替换决策制定者的一个子集所做出的决策。利用全国大型孕产结果和繁殖年龄夫妇孕前检查的医生诊断数据集，我们试验了一种启发式高频率方法以及一种贝叶斯后验损失函数方法，并将其应用于异常出生检测。我们发现，我们的算法在一个测试数据集上的结果比仅由医生诊断的结果具有更高的总体真阳性率和更低的假阳性率。我们还发现，来自农村地区的医生的诊断更容易被替代，这表明人工智能辅助决策制定更容易提高精确度。

    This paper proposes a statistical framework with which artificial intelligence can improve human decision making. The performance of each human decision maker is first benchmarked against machine predictions; we then replace the decisions made by a subset of the decision makers with the recommendation from the proposed artificial intelligence algorithm. Using a large nationwide dataset of pregnancy outcomes and doctor diagnoses from prepregnancy checkups of reproductive age couples, we experimented with both a heuristic frequentist approach and a Bayesian posterior loss function approach with an application to abnormal birth detection. We find that our algorithm on a test dataset results in a higher overall true positive rate and a lower false positive rate than the diagnoses made by doctors only. We also find that the diagnoses of doctors from rural areas are more frequently replaceable, suggesting that artificial intelligence assisted decision making tends to improve precision more i
    

