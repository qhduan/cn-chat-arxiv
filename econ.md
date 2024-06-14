# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Macroeconomic Policies based on Microfoundations: A Stackelberg Mean Field Game Approach](https://arxiv.org/abs/2403.12093) | 本研究提出了基于Stackelberg Mean Field Game的方法，可以有效地学习宏观经济政策，并在模型预训练和无模型Stackelberg均场强化学习算法的基础上取得了实验结果表明其优越性。 |
| [^2] | [Asymptotic Theory for Two-Way Clustering](https://arxiv.org/abs/2301.03805) | 该论文证明了针对表现出两路依赖性和集群异质性样本的新中心极限定理，弥补了先前只适用于具有同质性要求的两路聚类推断理论，并在推断上对线性回归进行了验证。 |
| [^3] | [Local Identification in the Instrumental Variable Multivariate Quantile Regression Model.](http://arxiv.org/abs/2401.11422) | 提出了基于最优输运的多元分位数回归模型，考虑结果变量中条目之间的相关性，并提供了局部识别结果。结果表明，所需的仪器变量（IV）的支持大小与结果向量的维度无关，只需IV足够信息量。 |
| [^4] | [The Mean Squared Error of the Ridgeless Least Squares Estimator under General Assumptions on Regression Errors.](http://arxiv.org/abs/2305.12883) | 该论文研究了基于一般回归误差假设的无噪声回归最小二乘估计值的均方误差，并发现包含大量不重要的参数可以有效地降低估计器的均方误差。 |
| [^5] | [Covert learning and disclosure.](http://arxiv.org/abs/2304.02989) | 本研究研究了一个信息获取和传递的模型，在该模型中，发送者选择有选择性地忽视信息，而不是欺骗接收者。本文阐明了欺骗可能性如何决定发送者选择获取和传递的信息，并确定了发送者和接收者最优的伪造环境。 |
| [^6] | [Artificial Intelligence and Dual Contract.](http://arxiv.org/abs/2303.12350) | 本文通过实验研究了人工智能算法在双重合同问题中能够自主设计激励相容的合同，无需外部引导或通信，并且不同AI算法支持的委托人可以采用混合和零和博弈行为，更具智能的委托人往往会变得合作。 |
| [^7] | [Finding all stable matchings with assignment constraints.](http://arxiv.org/abs/2204.03989) | 本文提出了一个算法，可以寻找受分配约束的稳定匹配，并输出所有的稳定匹配。这为市场设计者提供了测试和实施具有分配约束的稳定匹配的工具。 |

# 详细

[^1]: 基于微观基础的宏观经济政策学习：一种斯塔克尔贝格均场博弈方法

    Learning Macroeconomic Policies based on Microfoundations: A Stackelberg Mean Field Game Approach

    [https://arxiv.org/abs/2403.12093](https://arxiv.org/abs/2403.12093)

    本研究提出了基于Stackelberg Mean Field Game的方法，可以有效地学习宏观经济政策，并在模型预训练和无模型Stackelberg均场强化学习算法的基础上取得了实验结果表明其优越性。

    

    有效的宏观经济政策在促进经济增长和社会稳定方面起着至关重要的作用。本文基于Stackelberg Mean Field Game（SMFG）模型，将最优宏观经济政策问题建模，其中政府作为政策制定的领导者，大规模家庭动态响应为追随者。这种建模方法捕捉了政府和大规模家庭之间的非对称动态博弈，并可以解释地评估基于微观基础的宏观经济政策效果，这是现有方法难以实现的。我们还提出了一种解决SMFG的方法，将真实数据进行预训练，并结合一种无模型的Stackelberg均场强化学习（SMFRL）算法，该算法可以独立于先前的环境知识和转变运行。我们的实验结果展示了SMFG方法在经济政策方面优于其他方法的优越性。

    arXiv:2403.12093v1 Announce Type: cross  Abstract: Effective macroeconomic policies play a crucial role in promoting economic growth and social stability. This paper models the optimal macroeconomic policy problem based on the \textit{Stackelberg Mean Field Game} (SMFG), where the government acts as the leader in policy-making, and large-scale households dynamically respond as followers. This modeling method captures the asymmetric dynamic game between the government and large-scale households, and interpretably evaluates the effects of macroeconomic policies based on microfoundations, which is difficult for existing methods to achieve. We also propose a solution for SMFGs, incorporating pre-training on real data and a model-free \textit{Stackelberg mean-field reinforcement learning }(SMFRL) algorithm, which operates independently of prior environmental knowledge and transitions. Our experimental results showcase the superiority of the SMFG method over other economic policies in terms 
    
[^2]: 两路聚类的渐近理论

    Asymptotic Theory for Two-Way Clustering

    [https://arxiv.org/abs/2301.03805](https://arxiv.org/abs/2301.03805)

    该论文证明了针对表现出两路依赖性和集群异质性样本的新中心极限定理，弥补了先前只适用于具有同质性要求的两路聚类推断理论，并在推断上对线性回归进行了验证。

    

    本文证明了对于表现出两路依赖性和集群异质性的样本，存在一种新的中心极限定理。至今，在具有两路依赖性和集群异质性情况下的统计推断一直是一个悬而未决的问题。现有的两路聚类推断理论要求集群间具有相同的分布（所谓的独立交换性假设所导出的），而在现有的一路聚类理论中并不需要这样的同质性要求。因此，新的结果从理论上证实了两路聚类是一路聚类的更稳健版本，并与应用实践一致。该结果应用于线性回归，显示了标准插补方差估计是有效的推断方法。

    arXiv:2301.03805v2 Announce Type: replace  Abstract: This paper proves a new central limit theorem for a sample that exhibits two-way dependence and heterogeneity across clusters. Statistical inference for situations with both two-way dependence and cluster heterogeneity has thus far been an open issue. The existing theory for two-way clustering inference requires identical distributions across clusters (implied by the so-called separate exchangeability assumption). Yet no such homogeneity requirement is needed in the existing theory for one-way clustering. The new result therefore theoretically justifies the view that two-way clustering is a more robust version of one-way clustering, consistent with applied practice. The result is applied to linear regression, where it is shown that a standard plug-in variance estimator is valid for inference.
    
[^3]: 仪器变量多元分位数回归模型中的局部识别

    Local Identification in the Instrumental Variable Multivariate Quantile Regression Model. (arXiv:2401.11422v1 [econ.EM])

    [http://arxiv.org/abs/2401.11422](http://arxiv.org/abs/2401.11422)

    提出了基于最优输运的多元分位数回归模型，考虑结果变量中条目之间的相关性，并提供了局部识别结果。结果表明，所需的仪器变量（IV）的支持大小与结果向量的维度无关，只需IV足够信息量。

    

    Chernozhukov和Hansen（2005）引入的仪器变量（IV）分位数回归模型是分析内生性情况下分位数处理效应的有用工具，但当结果变量是多维的时，它对每个变量不同维度的联合分布保持沉默。为了克服这个限制，我们提出了一个基于最优输运的考虑结果变量中条目之间相关性的多元分位数回归模型。然后，我们为模型提供了一个局部识别结果。令人惊讶的是，我们发现，所需的用于识别的IV的支持大小与结果向量的维度无关，只要IV足够信息量。我们的结果来自我们建立的一个具有独立理论意义的一般识别定理。

    The instrumental variable (IV) quantile regression model introduced by Chernozhukov and Hansen (2005) is a useful tool for analyzing quantile treatment effects in the presence of endogeneity, but when outcome variables are multidimensional, it is silent on the joint distribution of different dimensions of each variable. To overcome this limitation, we propose an IV model built on the optimal-transport-based multivariate quantile that takes into account the correlation between the entries of the outcome variable. We then provide a local identification result for the model. Surprisingly, we find that the support size of the IV required for the identification is independent of the dimension of the outcome vector, as long as the IV is sufficiently informative. Our result follows from a general identification theorem that we establish, which has independent theoretical significance.
    
[^4]: 基于一般回归误差假设来研究无噪声回归最小二乘估计值的均方误差

    The Mean Squared Error of the Ridgeless Least Squares Estimator under General Assumptions on Regression Errors. (arXiv:2305.12883v1 [math.ST])

    [http://arxiv.org/abs/2305.12883](http://arxiv.org/abs/2305.12883)

    该论文研究了基于一般回归误差假设的无噪声回归最小二乘估计值的均方误差，并发现包含大量不重要的参数可以有效地降低估计器的均方误差。

    

    近年来，最小$\ell_2$范数（无岭）插值最小二乘估计器的研究方兴未艾。然而，大多数分析都局限于简单的回归误差结构，假设误差是独立同分布的，具有零均值和相同的方差，与特征向量无关。此外，这些理论分析的主要重点是样本外预测风险。本文通过检查无岭插值最小二乘估计器的均方误差，允许更一般的回归误差假设，打破了现有文献的局限性。具体而言，我们研究过度参数化的潜在好处，通过描绘有限样本中的均方误差来表征均方误差。我们的研究结果表明，相对于样本量，包含大量不重要的参数可以有效地降低估计器的均方误差。

    In recent years, there has been a significant growth in research focusing on minimum $\ell_2$ norm (ridgeless) interpolation least squares estimators. However, the majority of these analyses have been limited to a simple regression error structure, assuming independent and identically distributed errors with zero mean and common variance, independent of the feature vectors. Additionally, the main focus of these theoretical analyses has been on the out-of-sample prediction risk. This paper breaks away from the existing literature by examining the mean squared error of the ridgeless interpolation least squares estimator, allowing for more general assumptions about the regression errors. Specifically, we investigate the potential benefits of overparameterization by characterizing the mean squared error in a finite sample. Our findings reveal that including a large number of unimportant parameters relative to the sample size can effectively reduce the mean squared error of the estimator. N
    
[^5]: 隐秘的学习和披露

    Covert learning and disclosure. (arXiv:2304.02989v1 [econ.TH])

    [http://arxiv.org/abs/2304.02989](http://arxiv.org/abs/2304.02989)

    本研究研究了一个信息获取和传递的模型，在该模型中，发送者选择有选择性地忽视信息，而不是欺骗接收者。本文阐明了欺骗可能性如何决定发送者选择获取和传递的信息，并确定了发送者和接收者最优的伪造环境。

    

    本研究研究了一个信息获取和传递的模型，在该模型中，发送者误报其发现的能力受到限制。在均衡状态下，发送者选择有选择性地忽视信息，而不是欺骗接收者。虽然不会产生欺骗，但我强调了欺骗可能性如何决定发送者选择获取和传递的信息。然后，本文转向比较静态分析，阐明了发送者如何从其声明更可验证中受益，并表明这类似于增加其承诺能力。最后，本文确定了发送者和接收者最优的伪造环境。

    I study a model of information acquisition and transmission in which the sender's ability to misreport her findings is limited. In equilibrium, the sender only influences the receiver by choosing to remain selectively ignorant, rather than by deceiving her about the discoveries. Although deception does not occur, I highlight how deception possibilities determine what information the sender chooses to acquire and transmit. I then turn to comparative statics, characterizing in which sense the sender benefits from her claims being more verifiable, showing this is akin to increasing her commitment power. Finally, I characterize sender- and receiver-optimal falsification environments.
    
[^6]: 人工智能与双重合同

    Artificial Intelligence and Dual Contract. (arXiv:2303.12350v1 [cs.AI])

    [http://arxiv.org/abs/2303.12350](http://arxiv.org/abs/2303.12350)

    本文通过实验研究了人工智能算法在双重合同问题中能够自主设计激励相容的合同，无需外部引导或通信，并且不同AI算法支持的委托人可以采用混合和零和博弈行为，更具智能的委托人往往会变得合作。

    

    随着人工智能算法的快速进步，人们希望算法很快就能在各个领域取代人类决策者，例如合同设计。我们通过实验研究了由人工智能（多智能体Q学习）驱动的算法在双重委托-代理问题的经典“双重合同”模型中的行为。我们发现，这些AI算法可以自主学习设计合适的激励相容合同，而无需外部引导或者它们之间的通信。我们强调，由不同AI算法支持的委托人可以采用混合和零和博弈行为。我们还发现，更具智能的委托人往往会变得合作，而智能较低的委托人则会出现内生性近视并倾向于竞争。在最优合同下，代理的较低合同激励由委托人之间的勾结策略维持。

    With the dramatic progress of artificial intelligence algorithms in recent times, it is hoped that algorithms will soon supplant human decision-makers in various fields, such as contract design. We analyze the possible consequences by experimentally studying the behavior of algorithms powered by Artificial Intelligence (Multi-agent Q-learning) in a workhorse \emph{dual contract} model for dual-principal-agent problems. We find that the AI algorithms autonomously learn to design incentive-compatible contracts without external guidance or communication among themselves. We emphasize that the principal, powered by distinct AI algorithms, can play mixed-sum behavior such as collusion and competition. We find that the more intelligent principals tend to become cooperative, and the less intelligent principals are endogenizing myopia and tend to become competitive. Under the optimal contract, the lower contract incentive to the agent is sustained by collusive strategies between the principals
    
[^7]: 寻找受分配约束的所有稳定匹配

    Finding all stable matchings with assignment constraints. (arXiv:2204.03989v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2204.03989](http://arxiv.org/abs/2204.03989)

    本文提出了一个算法，可以寻找受分配约束的稳定匹配，并输出所有的稳定匹配。这为市场设计者提供了测试和实施具有分配约束的稳定匹配的工具。

    

    本文考虑了受到分配约束的稳定匹配。这些匹配需要包含某些分配对，坚持排除其他某些分配对，并且是稳定的。我们的主要贡献是提出了一个算法来确定分配约束与稳定性的兼容性。只要存在与分配约束一致的稳定匹配，我们的算法将输出所有稳定匹配（每个解的多项式时间）。这为市场设计者提供了（i）测试具有分配约束的稳定匹配可行性的工具，以及（ii）实施它们的单独工具。

    In this paper we consider stable matchings that are subject to assignment constraints. These are matchings that require certain assigned pairs to be included, insist that some other assigned pairs are not, and, importantly, are stable. Our main contribution is an algorithm that determines when assignment constraints are compatible with stability. Whenever a stable matching consistent with the assignment constraints exists, our algorithm will output all of them (each in polynomial time per solution). This provides market designers with (i) a tool to test the feasibility of stable matchings with assignment constraints, and (ii) a separate tool to implement them.
    

