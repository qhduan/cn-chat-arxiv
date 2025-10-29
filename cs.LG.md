# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion Models Meet Contextual Bandits with Large Action Spaces](https://arxiv.org/abs/2402.10028) | 本文设计了一种利用预训练扩散模型的扩散汤普森采样方法，用于在大动作空间下进行高效的情境强化学习探索。实证评估结果表明了该方法的优越性能。 |
| [^2] | [Optimistically Tempered Online Learning](https://arxiv.org/abs/2301.07530) | 本文提出了一种乐观调节的在线学习框架和适应算法，挑战了对专家的信心假设，并通过动态遗憾界限的理论保证和实验证明了该方法的有效性。 |
| [^3] | [Minimax Optimal Transfer Learning for Kernel-based Nonparametric Regression.](http://arxiv.org/abs/2310.13966) | 本文主要研究了在再生核希尔伯特空间中的非参数回归的传递学习问题，提出了两种情况下的解决方法，并分别给出了统计性质和最优性结果。 |
| [^4] | [Exploit the antenna response consistency to define the alignment criteria for CSI data.](http://arxiv.org/abs/2310.06328) | 本论文提出了一个解决方案，利用天线响应一致性（ARC）来定义适当的对准标准，以解决在WiFi人体活动识别中的自我监督学习算法在CSI数据上无法达到预期性能的问题。 |
| [^5] | [Datasheets for Machine Learning Sensors.](http://arxiv.org/abs/2306.08848) | 本研究提出了一种用于机器学习传感器的标准数据表模板，并讨论了其主要组成部分。这些数据表可以促进对传感器数据在机器学习应用中的理解和利用，并提供了客观的性能评估指标。 |

# 详细

[^1]: 扩散模型与大动作空间情境强化学习的结合

    Diffusion Models Meet Contextual Bandits with Large Action Spaces

    [https://arxiv.org/abs/2402.10028](https://arxiv.org/abs/2402.10028)

    本文设计了一种利用预训练扩散模型的扩散汤普森采样方法，用于在大动作空间下进行高效的情境强化学习探索。实证评估结果表明了该方法的优越性能。

    

    由于动作空间较大，有效的探索是情境强化学习中的一个关键挑战。本文通过利用预训练的扩散模型来捕捉动作之间的相关性，设计了扩散汤普森采样（dTS）方法，实现了高效的探索。我们为dTS方法提供了理论和算法基础，并通过实证评估展示了它的优越性能。

    arXiv:2402.10028v1 Announce Type: cross  Abstract: Efficient exploration is a key challenge in contextual bandits due to the large size of their action space, where uninformed exploration can result in computational and statistical inefficiencies. Fortunately, the rewards of actions are often correlated and this can be leveraged to explore them efficiently. In this work, we capture such correlations using pre-trained diffusion models; upon which we design diffusion Thompson sampling (dTS). Both theoretical and algorithmic foundations are developed for dTS, and empirical evaluation also shows its favorable performance.
    
[^2]: 乐观调节的在线学习

    Optimistically Tempered Online Learning

    [https://arxiv.org/abs/2301.07530](https://arxiv.org/abs/2301.07530)

    本文提出了一种乐观调节的在线学习框架和适应算法，挑战了对专家的信心假设，并通过动态遗憾界限的理论保证和实验证明了该方法的有效性。

    

    乐观在线学习算法已经被开发出来，以利用专家意见，假设专家意见总是有用的。然而，我们可以合理地对这些意见与基于梯度的在线算法提供的学习信息的相关性提出质疑。在这项工作中，我们质疑对专家的信心假设，并开发了乐观调节（OT）在线学习框架以及在线算法的OT适应性。我们的算法具有动态遗憾界限的稳固理论保证，并最终验证了OT方法的有用性。

    arXiv:2301.07530v2 Announce Type: replace Abstract: Optimistic Online Learning algorithms have been developed to exploit expert advices, assumed optimistically to be always useful. However, it is legitimate to question the relevance of such advices \emph{w.r.t.} the learning information provided by gradient-based online algorithms. In this work, we challenge the confidence assumption on the expert and develop the \emph{optimistically tempered} (OT) online learning framework as well as OT adaptations of online algorithms. Our algorithms come with sound theoretical guarantees in the form of dynamic regret bounds, and we eventually provide experimental validation of the usefulness of the OT approach.
    
[^3]: 基于核非参数回归的最优极小化传递学习

    Minimax Optimal Transfer Learning for Kernel-based Nonparametric Regression. (arXiv:2310.13966v1 [stat.ML])

    [http://arxiv.org/abs/2310.13966](http://arxiv.org/abs/2310.13966)

    本文主要研究了在再生核希尔伯特空间中的非参数回归的传递学习问题，提出了两种情况下的解决方法，并分别给出了统计性质和最优性结果。

    

    近年来，传递学习在机器学习社区中受到了很大关注。它能够利用相关研究的知识来提高目标研究的泛化性能，使其具有很高的吸引力。本文主要研究在再生核希尔伯特空间中的非参数回归的传递学习问题，目的是缩小实际效果与理论保证之间的差距。具体考虑了两种情况：已知可传递的来源和未知的情况。对于已知可传递的来源情况，我们提出了一个两步核估计器，仅使用核岭回归。对于未知的情况，我们开发了一种基于高效聚合算法的新方法，可以自动检测并减轻负面来源的影响。本文提供了所需估计器的统计性质，并建立了该方法的最优性结果。

    In recent years, transfer learning has garnered significant attention in the machine learning community. Its ability to leverage knowledge from related studies to improve generalization performance in a target study has made it highly appealing. This paper focuses on investigating the transfer learning problem within the context of nonparametric regression over a reproducing kernel Hilbert space. The aim is to bridge the gap between practical effectiveness and theoretical guarantees. We specifically consider two scenarios: one where the transferable sources are known and another where they are unknown. For the known transferable source case, we propose a two-step kernel-based estimator by solely using kernel ridge regression. For the unknown case, we develop a novel method based on an efficient aggregation algorithm, which can automatically detect and alleviate the effects of negative sources. This paper provides the statistical properties of the desired estimators and establishes the 
    
[^4]: 利用天线响应一致性定义CSI数据的对准标准

    Exploit the antenna response consistency to define the alignment criteria for CSI data. (arXiv:2310.06328v1 [cs.LG])

    [http://arxiv.org/abs/2310.06328](http://arxiv.org/abs/2310.06328)

    本论文提出了一个解决方案，利用天线响应一致性（ARC）来定义适当的对准标准，以解决在WiFi人体活动识别中的自我监督学习算法在CSI数据上无法达到预期性能的问题。

    

    自我监督学习（SSL）用于基于WiFi的人体活动识别（HAR）由于能够解决标注数据不足的挑战而具有很大的潜力。然而，直接将原本设计用于其他领域的SSL算法，特别是对比学习，移植到CSI数据上往往无法达到预期的性能。我们将这个问题归因于对准标准不当，这破坏了特征空间和输入空间之间的语义距离一致性。为了解决这个挑战，我们引入了``Anetenna Response Consistency (ARC)''作为定义合适对准标准的解决方案。ARC的设计在保留输入空间的语义信息的同时，引入了对现实世界噪声的鲁棒性。我们从CSI数据结构的角度分析了ARC，并展示了其最优解导致了从输入CSI数据到特征映射中的动作向量的直接映射。

    Self-supervised learning (SSL) for WiFi-based human activity recognition (HAR) holds great promise due to its ability to address the challenge of insufficient labeled data. However, directly transplanting SSL algorithms, especially contrastive learning, originally designed for other domains to CSI data, often fails to achieve the expected performance. We attribute this issue to the inappropriate alignment criteria, which disrupt the semantic distance consistency between the feature space and the input space. To address this challenge, we introduce \textbf{A}netenna \textbf{R}esponse \textbf{C}onsistency (ARC) as a solution to define proper alignment criteria. ARC is designed to retain semantic information from the input space while introducing robustness to real-world noise. We analyze ARC from the perspective of CSI data structure, demonstrating that its optimal solution leads to a direct mapping from input CSI data to action vectors in the feature map. Furthermore, we provide extensi
    
[^5]: 机器学习传感器的数据表

    Datasheets for Machine Learning Sensors. (arXiv:2306.08848v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.08848](http://arxiv.org/abs/2306.08848)

    本研究提出了一种用于机器学习传感器的标准数据表模板，并讨论了其主要组成部分。这些数据表可以促进对传感器数据在机器学习应用中的理解和利用，并提供了客观的性能评估指标。

    

    机器学习（ML）传感器提供了一种新的感知范式，能够在边缘进行智能化，同时赋予终端用户更多对其数据的控制权。由于这些ML传感器在智能设备的发展中起着至关重要的作用，清晰地记录其规格、功能和限制非常关键。本文介绍了一种用于ML传感器的标准数据表模板，并讨论了其主要组成部分，包括系统的硬件、ML模型和数据集属性、端到端性能指标以及环境影响。我们提供了一个我们自己ML传感器的示例数据表，并详细讨论了每个部分。我们强调这些数据表如何促进对ML应用中传感器数据的更好理解和利用，并提供了客观的衡量系统性能的指标进行评估和比较。ML传感器及其数据表共同提供了更高的隐私、安全性、透明度、可解释性、可审计性和

    Machine learning (ML) sensors offer a new paradigm for sensing that enables intelligence at the edge while empowering end-users with greater control of their data. As these ML sensors play a crucial role in the development of intelligent devices, clear documentation of their specifications, functionalities, and limitations is pivotal. This paper introduces a standard datasheet template for ML sensors and discusses its essential components including: the system's hardware, ML model and dataset attributes, end-to-end performance metrics, and environmental impact. We provide an example datasheet for our own ML sensor and discuss each section in detail. We highlight how these datasheets can facilitate better understanding and utilization of sensor data in ML applications, and we provide objective measures upon which system performance can be evaluated and compared. Together, ML sensors and their datasheets provide greater privacy, security, transparency, explainability, auditability, and u
    

