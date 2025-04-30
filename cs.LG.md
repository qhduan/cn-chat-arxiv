# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Twin Auto-Encoder Model for Learning Separable Representation in Cyberattack Detection](https://arxiv.org/abs/2403.15509) | 提出一种新型双自动编码器模型(TAE)，通过将潜在表示转换为可分离表示来解决网络攻击检测中混合表示的问题 |
| [^2] | [Intelligent Condition Monitoring of Industrial Plants: An Overview of Methodologies and Uncertainty Management Strategies.](http://arxiv.org/abs/2401.10266) | 本论文综述了工业厂房智能状态监测和故障检测和诊断方法，重点关注了Tennessee Eastman Process。调研总结了最流行和最先进的深度学习和机器学习算法，并探讨了算法的优劣势。还讨论了不平衡数据和无标记样本等挑战，以及深度学习模型如何应对。比较了不同算法在Tennessee Eastman Process上的准确性和规格。 |
| [^3] | [Settling the Sample Complexity of Online Reinforcement Learning.](http://arxiv.org/abs/2307.13586) | 本文解决了在线强化学习的样本复杂度问题，提出了一种基于模型的算法，它可以在有限时间不均匀马尔可夫决策问题中实现极小后悔的最优性。 |
| [^4] | [When Deep Learning Meets Polyhedral Theory: A Survey.](http://arxiv.org/abs/2305.00241) | 本文综述了深度学习与多面体理论的交叉领域。修正线性单元（ReLU）等函数使得一些神经网络结构能够通过多面体理论进行分析，应用线性和混合整数线性规划来实现网络修剪、鲁棒性分析和神经网络验证等任务。 |
| [^5] | [The Adaptive $\tau$-Lasso: Its Robustness and Oracle Properties.](http://arxiv.org/abs/2304.09310) | 本文提出了一种新型鲁棒的自适应 $\tau$-Lasso 估计器，同时采用自适应 $\ell_1$-范数惩罚项以降低真实回归系数的偏差。它具有变量选择一致性和真实支持下回归向量的渐近正态性的最优性质，假定已知真实回归向量的支持。 |
| [^6] | [Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answering.](http://arxiv.org/abs/2303.01903) | 本研究提出了一个名为Prophet的框架，使用答案启发式方式促使GPT-3解决基于知识的视觉问答问题。在特定的知识型VQA数据集上训练一个纯VQA模型，并从中提取出答案启发式，可提高模型的性能。 |

# 详细

[^1]: 双自动编码器模型用于学习网络攻击检测中的可分离表示

    Twin Auto-Encoder Model for Learning Separable Representation in Cyberattack Detection

    [https://arxiv.org/abs/2403.15509](https://arxiv.org/abs/2403.15509)

    提出一种新型双自动编码器模型(TAE)，通过将潜在表示转换为可分离表示来解决网络攻击检测中混合表示的问题

    

    表征学习在网络攻击检测等许多问题的成功中起着关键作用。大多数网络攻击检测的表征学习方法基于自动编码器（AE）模型的潜在向量。为了解决AEs表示中混合的问题，我们提出了一种称为双自动编码器（TAE）的新型模型。TAE将潜在表示确定地转换为更易区分的表示，即\textit{可分离表示}，并在输出端重建可分离表示。

    arXiv:2403.15509v1 Announce Type: cross  Abstract: Representation Learning (RL) plays a pivotal role in the success of many problems including cyberattack detection. Most of the RL methods for cyberattack detection are based on the latent vector of Auto-Encoder (AE) models. An AE transforms raw data into a new latent representation that better exposes the underlying characteristics of the input data. Thus, it is very useful for identifying cyberattacks. However, due to the heterogeneity and sophistication of cyberattacks, the representation of AEs is often entangled/mixed resulting in the difficulty for downstream attack detection models. To tackle this problem, we propose a novel mod called Twin Auto-Encoder (TAE). TAE deterministically transforms the latent representation into a more distinguishable representation namely the \textit{separable representation} and the reconstructsuct the separable representation at the output. The output of TAE called the \textit{reconstruction represe
    
[^2]: 工业厂房智能状态监测: 方法论和不确定性管理策略综述

    Intelligent Condition Monitoring of Industrial Plants: An Overview of Methodologies and Uncertainty Management Strategies. (arXiv:2401.10266v1 [cs.LG])

    [http://arxiv.org/abs/2401.10266](http://arxiv.org/abs/2401.10266)

    本论文综述了工业厂房智能状态监测和故障检测和诊断方法，重点关注了Tennessee Eastman Process。调研总结了最流行和最先进的深度学习和机器学习算法，并探讨了算法的优劣势。还讨论了不平衡数据和无标记样本等挑战，以及深度学习模型如何应对。比较了不同算法在Tennessee Eastman Process上的准确性和规格。

    

    状态监测在现代工业系统的安全性和可靠性中起着重要作用。人工智能（AI）方法作为一种在工业应用中日益受到学术界和行业关注的增长主题和一种强大的故障识别方式。本文概述了工业厂房智能状态监测和故障检测和诊断方法，重点关注开源基准Tennessee Eastman Process（TEP）。在这项调查中，总结了用于工业厂房状态监测、故障检测和诊断的最流行和最先进的深度学习（DL）和机器学习（ML）算法，并研究了每种算法的优点和缺点。还涵盖了不平衡数据、无标记样本以及深度学习模型如何处理这些挑战。最后，比较了利用Tennessee Eastman Process的不同算法的准确性和规格。

    Condition monitoring plays a significant role in the safety and reliability of modern industrial systems. Artificial intelligence (AI) approaches are gaining attention from academia and industry as a growing subject in industrial applications and as a powerful way of identifying faults. This paper provides an overview of intelligent condition monitoring and fault detection and diagnosis methods for industrial plants with a focus on the open-source benchmark Tennessee Eastman Process (TEP). In this survey, the most popular and state-of-the-art deep learning (DL) and machine learning (ML) algorithms for industrial plant condition monitoring, fault detection, and diagnosis are summarized and the advantages and disadvantages of each algorithm are studied. Challenges like imbalanced data, unlabelled samples and how deep learning models can handle them are also covered. Finally, a comparison of the accuracies and specifications of different algorithms utilizing the Tennessee Eastman Process 
    
[^3]: 解决在线强化学习的样本复杂度问题

    Settling the Sample Complexity of Online Reinforcement Learning. (arXiv:2307.13586v1 [cs.LG])

    [http://arxiv.org/abs/2307.13586](http://arxiv.org/abs/2307.13586)

    本文解决了在线强化学习的样本复杂度问题，提出了一种基于模型的算法，它可以在有限时间不均匀马尔可夫决策问题中实现极小后悔的最优性。

    

    在线强化学习的一个核心问题是数据效率。虽然最近的一些工作在在线强化学习中实现了渐近最小的后悔，但这些结果的最优性仅在“大样本”情况下得到保证，为了使其算法运行最佳，需要付出巨大的预燃成本。如何在不产生任何预燃成本的情况下实现极小后悔的最优性一直是强化学习理论中的一个开放问题。我们解决了有限时间不均匀马尔可夫决策问题的这个问题。具体地，我们证明了一种修改版的单调值传播(MVP)算法，该算法是由\cite{zhang2020reinforcement}提出的一种基于模型的算法，使得后悔的量级为(模除对数因子)\begin{equation *} \min\biggr\{ \sqrt{SAH^3K}，\，HK \biggr\}，\end{equation *}其中$S$是状态数，$A$是动作数，$H$是规划时域，$K$是总的回合数。这个后悔的量级与极小化后悔量级是相匹配的。

    A central issue lying at the heart of online reinforcement learning (RL) is data efficiency. While a number of recent works achieved asymptotically minimal regret in online RL, the optimality of these results is only guaranteed in a ``large-sample'' regime, imposing enormous burn-in cost in order for their algorithms to operate optimally. How to achieve minimax-optimal regret without incurring any burn-in cost has been an open problem in RL theory.  We settle this problem for the context of finite-horizon inhomogeneous Markov decision processes. Specifically, we prove that a modified version of Monotonic Value Propagation (MVP), a model-based algorithm proposed by \cite{zhang2020reinforcement}, achieves a regret on the order of (modulo log factors) \begin{equation*}  \min\big\{ \sqrt{SAH^3K}, \,HK \big\}, \end{equation*} where $S$ is the number of states, $A$ is the number of actions, $H$ is the planning horizon, and $K$ is the total number of episodes. This regret matches the minimax 
    
[^4]: 当深度学习遇见多面体理论：一项综述

    When Deep Learning Meets Polyhedral Theory: A Survey. (arXiv:2305.00241v1 [math.OC])

    [http://arxiv.org/abs/2305.00241](http://arxiv.org/abs/2305.00241)

    本文综述了深度学习与多面体理论的交叉领域。修正线性单元（ReLU）等函数使得一些神经网络结构能够通过多面体理论进行分析，应用线性和混合整数线性规划来实现网络修剪、鲁棒性分析和神经网络验证等任务。

    

    在过去的十年中，深度学习成为了预测建模的主要方法，得益于深度神经网络在计算机视觉和自然语言处理等任务中的显著准确性。与此同时，神经网络的结构回归到了基于分段常数和分段线性函数的简单表示，例如修正线性单元（ReLU），这种激活函数成为神经网络中最常用的类型。这使得某些类型的网络结构，如典型的全连接前馈神经网络，能够通过多面体理论进行分析，并应用线性规划（LP）和混合整数线性规划（MILP）等方法用于各种目的。本文综述了这个快速发展领域涌现的主要主题，为更详细地了解神经网络以及应用数学提供了新的视角。我们介绍了多面体理论的基础知识以及它与深度学习的关系，并回顾了该主题的最新进展，包括在网络修剪、鲁棒性分析和神经网络验证等任务中使用LP和MILP。最后，我们讨论了当前挑战和未来研究方向。

    In the past decade, deep learning became the prevalent methodology for predictive modeling thanks to the remarkable accuracy of deep neural networks in tasks such as computer vision and natural language processing. Meanwhile, the structure of neural networks converged back to simpler representations based on piecewise constant and piecewise linear functions such as the Rectified Linear Unit (ReLU), which became the most commonly used type of activation function in neural networks. That made certain types of network structure $\unicode{x2014}$such as the typical fully-connected feedforward neural network$\unicode{x2014}$ amenable to analysis through polyhedral theory and to the application of methodologies such as Linear Programming (LP) and Mixed-Integer Linear Programming (MILP) for a variety of purposes. In this paper, we survey the main topics emerging from this fast-paced area of work, which bring a fresh perspective to understanding neural networks in more detail as well as to app
    
[^5]: 自适应 $\tau$-Lasso：其健壮性和最优性质。

    The Adaptive $\tau$-Lasso: Its Robustness and Oracle Properties. (arXiv:2304.09310v1 [stat.ML])

    [http://arxiv.org/abs/2304.09310](http://arxiv.org/abs/2304.09310)

    本文提出了一种新型鲁棒的自适应 $\tau$-Lasso 估计器，同时采用自适应 $\ell_1$-范数惩罚项以降低真实回归系数的偏差。它具有变量选择一致性和真实支持下回归向量的渐近正态性的最优性质，假定已知真实回归向量的支持。

    

    本文介绍了一种用于分析高维数据集的新型正则化鲁棒 $\tau$-回归估计器，以应对响应变量和协变量的严重污染。我们称这种估计器为自适应 $\tau$-Lasso，它对异常值和高杠杆点具有鲁棒性，同时采用自适应 $\ell_1$-范数惩罚项来减少真实回归系数的偏差。具体而言，该自适应 $\ell_1$-范数惩罚项为每个回归系数分配一个权重。对于固定数量的预测变量 $p$，我们显示出自适应 $\tau$-Lasso 具有变量选择一致性和真实支持下回归向量的渐近正态性的最优性质，假定已知真实回归向量的支持。然后我们通过有限样本断点和影响函数来表征其健壮性。我们进行了广泛的模拟来比较不同的估计器的性能。

    This paper introduces a new regularized version of the robust $\tau$-regression estimator for analyzing high-dimensional data sets subject to gross contamination in the response variables and covariates. We call the resulting estimator adaptive $\tau$-Lasso that is robust to outliers and high-leverage points and simultaneously employs adaptive $\ell_1$-norm penalty term to reduce the bias associated with large true regression coefficients. More specifically, this adaptive $\ell_1$-norm penalty term assigns a weight to each regression coefficient. For a fixed number of predictors $p$, we show that the adaptive $\tau$-Lasso has the oracle property with respect to variable-selection consistency and asymptotic normality for the regression vector corresponding to the true support, assuming knowledge of the true regression vector support. We then characterize its robustness via the finite-sample breakdown point and the influence function. We carry-out extensive simulations to compare the per
    
[^6]: 用答案启发式方式促使大型语言模型解决基于知识的视觉问答问题

    Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answering. (arXiv:2303.01903v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2303.01903](http://arxiv.org/abs/2303.01903)

    本研究提出了一个名为Prophet的框架，使用答案启发式方式促使GPT-3解决基于知识的视觉问答问题。在特定的知识型VQA数据集上训练一个纯VQA模型，并从中提取出答案启发式，可提高模型的性能。

    

    基于知识的视觉问答需要超出图像范围的外部知识来回答问题。早期的研究从显式知识库（KBs）检索所需的知识，这经常会引入与问题无关的信息，从而限制了模型的性能。最近的研究试图将大型语言模型（即GPT-3）作为隐含式知识引擎来获取回答所需的必要知识。尽管这些方法取得了令人鼓舞的结果，但我们认为它们还没有充分发挥GPT-3的能力，因为提供的输入信息仍然不足。在本文中，我们提出了Prophet——一个概念上简单的框架，旨在通过回答启发式方式，促使GPT-3解决基于知识的VQA问题。具体来说，我们首先在特定的基于知识的VQA数据集上训练一个纯VQA模型，而不使用外部知识。之后，我们从模型中提取了两种互补的答案启发式：答案候选项。

    Knowledge-based visual question answering (VQA) requires external knowledge beyond the image to answer the question. Early studies retrieve required knowledge from explicit knowledge bases (KBs), which often introduces irrelevant information to the question, hence restricting the performance of their models. Recent works have sought to use a large language model (i.e., GPT-3) as an implicit knowledge engine to acquire the necessary knowledge for answering. Despite the encouraging results achieved by these methods, we argue that they have not fully activated the capacity of GPT-3 as the provided input information is insufficient. In this paper, we present Prophet -- a conceptually simple framework designed to prompt GPT-3 with answer heuristics for knowledge-based VQA. Specifically, we first train a vanilla VQA model on a specific knowledge-based VQA dataset without external knowledge. After that, we extract two types of complementary answer heuristics from the model: answer candidates 
    

