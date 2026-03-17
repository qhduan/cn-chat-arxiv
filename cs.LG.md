# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Multi-Agent Mapping for Planetary Exploration](https://arxiv.org/abs/2404.02289) | 联邦学习在多智能体机器人探测中的应用，利用隐式神经映射和地球数据集上的元初始化，实现了对不同领域如火星地形和冰川的强泛化能力。 |
| [^2] | [Survey of Computerized Adaptive Testing: A Machine Learning Perspective](https://arxiv.org/abs/2404.00712) | 本文以机器学习视角综述了计算机自适应测试（CAT），重点解析其测试问题选择算法和如何优化认知诊断模型、题库构建和测试控制。 |
| [^3] | [PARAMANU-AYN: An Efficient Novel Generative and Instruction-tuned Language Model for Indian Legal Case Documents](https://arxiv.org/abs/2403.13681) | PARAMANU-AYN是一种基于印度法律案例文件的高效生成式语言模型，采用自回归解码器进行预训练，并经过面向指令的微调，在各种法律任务上取得了良好表现。 |
| [^4] | [ConjNorm: Tractable Density Estimation for Out-of-Distribution Detection](https://arxiv.org/abs/2402.17888) | 提出了一种新颖的理论框架，基于Bregman散度，通过引入共轭约束，提出了一种\textsc{ConjNorm}方法，以在给定数据集中搜索最佳规范系数$p$来重新构想密度函数设计。 |
| [^5] | [Combining Evidence Across Filtrations](https://arxiv.org/abs/2402.09698) | 这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。 |
| [^6] | [A Review of Deep Learning Methods for Photoplethysmography Data.](http://arxiv.org/abs/2401.12783) | 本综述系统地回顾了自2017年至2023年期间应用深度学习模型处理光电容积法数据的论文。研究发现，深度学习在个人健康管理和其他应用中具有显著成果。根据任务的不同，这些论文被分为医学相关和非医学相关两大类别，医学相关又细分为七个子组，包括血压分析... |
| [^7] | [Provable Tensor Completion with Graph Information.](http://arxiv.org/abs/2310.02543) | 本文提出了一个创新框架，系统地解决了具有图信息的动态图正则化张量补全问题，建立了严格的数学表示，并推导出了新的图导向补全模型。 |
| [^8] | [Faster Stochastic Algorithms for Minimax Optimization under Polyak--{\L}ojasiewicz Conditions.](http://arxiv.org/abs/2307.15868) | 该论文提出了基于Polyak-Lojasiewicz条件的极小极大优化的快速随机算法SPIDER-GDA，并证明了该算法能够在较低的复杂度下找到接近最优解的结果。 |
| [^9] | [Trainability barriers and opportunities in quantum generative modeling.](http://arxiv.org/abs/2305.02881) | 本文研究了量子生成模型的可训练性障碍，如荒芜高原和指数损失集中，使用隐式生成模型和明确损失会产生一种新的荒芜高原现象。最大均值差可以是低秩且可训练的或全局性且不可训练的。但是，可训练性所需的低秩损失通常不能区分高频和低频特征。 |
| [^10] | [Skeleton Regression: A Graph-Based Approach to Estimation with Manifold Structure.](http://arxiv.org/abs/2303.11786) | 这是一个处理低维流形数据的回归框架，首先通过构建图形骨架来捕捉潜在的流形几何结构，然后在其上运用非参数回归技术来估计回归函数，除了具有非参数优点之外，在处理多个流形数据，嘈杂观察时也表现出较好的鲁棒性。 |
| [^11] | [Offline Estimation of Controlled Markov Chains: Minimaxity and Sample Complexity.](http://arxiv.org/abs/2211.07092) | 本论文研究了离线估计有限控制马尔可夫链的转移概率矩阵的非参数估计器，并通过记录策略的混合特性建立了样本复杂性界限和最小化条件。结果表明，实现特定的统计风险界限涉及到混合特性的强度和样本数量之间微妙而有趣的权衡。还使用这些样本复杂性界限建立了离线评估恒定马尔可夫控制策略的相关界限。 |
| [^12] | [Quadratic Gradient: Combining Gradient Algorithms and Newton's Method as One.](http://arxiv.org/abs/2209.03282) | 本文提出了一种基于对角矩阵的二次梯度，可以加速梯度的收敛速度，在实验中表现良好。研究者还推测海森矩阵与学习率之间可能存在关系。 |

# 详细

[^1]: 行星探测的联邦多智能体建图

    Federated Multi-Agent Mapping for Planetary Exploration

    [https://arxiv.org/abs/2404.02289](https://arxiv.org/abs/2404.02289)

    联邦学习在多智能体机器人探测中的应用，利用隐式神经映射和地球数据集上的元初始化，实现了对不同领域如火星地形和冰川的强泛化能力。

    

    在多智能体机器人探测中，管理和有效利用动态环境产生的大量异构数据构成了一个重要挑战。联邦学习（FL）是一种有前途的分布式映射方法，它解决了协作学习中去中心化数据的挑战。FL使多个智能体之间可以进行联合模型训练，而无需集中化或共享原始数据，克服了带宽和存储限制。我们的方法利用隐式神经映射，将地图表示为由神经网络学习的连续函数，以便实现紧凑和适应性的表示。我们进一步通过在地球数据集上进行元初始化来增强这一方法，预训练网络以快速学习新的地图结构。这种组合在诸如火星地形和冰川等不同领域展现了较强的泛化能力。我们对这一方法进行了严格评估，展示了其有效性。

    arXiv:2404.02289v1 Announce Type: cross  Abstract: In multi-agent robotic exploration, managing and effectively utilizing the vast, heterogeneous data generated from dynamic environments poses a significant challenge. Federated learning (FL) is a promising approach for distributed mapping, addressing the challenges of decentralized data in collaborative learning. FL enables joint model training across multiple agents without requiring the centralization or sharing of raw data, overcoming bandwidth and storage constraints. Our approach leverages implicit neural mapping, representing maps as continuous functions learned by neural networks, for compact and adaptable representations. We further enhance this approach with meta-initialization on Earth datasets, pre-training the network to quickly learn new map structures. This combination demonstrates strong generalization to diverse domains like Martian terrain and glaciers. We rigorously evaluate this approach, demonstrating its effectiven
    
[^2]: 计算机自适应测试综述：机器学习视角

    Survey of Computerized Adaptive Testing: A Machine Learning Perspective

    [https://arxiv.org/abs/2404.00712](https://arxiv.org/abs/2404.00712)

    本文以机器学习视角综述了计算机自适应测试（CAT），重点解析其测试问题选择算法和如何优化认知诊断模型、题库构建和测试控制。

    

    计算机自适应测试（CAT）提供了一种高效、量身定制的评估考生熟练程度的方法，通过根据他们的表现动态调整测试问题。CAT广泛应用于教育、医疗、体育和社会学等多个领域，彻底改变了测试实践。然而，随着大规模测试的增加复杂性，CAT已经融合了机器学习技术。本文旨在提供一个以机器学习为重点的CAT综述，从新的角度解读这种自适应测试方法。通过研究CAT适应性核心的测试问题选择算法，我们揭示了其功能。此外，我们探讨了认知诊断模型、题库构建和CAT中的测试控制，探索了机器学习如何优化这些组成部分。通过对当前情况的分析，

    arXiv:2404.00712v1 Announce Type: cross  Abstract: Computerized Adaptive Testing (CAT) provides an efficient and tailored method for assessing the proficiency of examinees, by dynamically adjusting test questions based on their performance. Widely adopted across diverse fields like education, healthcare, sports, and sociology, CAT has revolutionized testing practices. While traditional methods rely on psychometrics and statistics, the increasing complexity of large-scale testing has spurred the integration of machine learning techniques. This paper aims to provide a machine learning-focused survey on CAT, presenting a fresh perspective on this adaptive testing method. By examining the test question selection algorithm at the heart of CAT's adaptivity, we shed light on its functionality. Furthermore, we delve into cognitive diagnosis models, question bank construction, and test control within CAT, exploring how machine learning can optimize these components. Through an analysis of curre
    
[^3]: PARAMANU-AYN：一种有效的新型生成式、面向印度法律案例文件的语言模型

    PARAMANU-AYN: An Efficient Novel Generative and Instruction-tuned Language Model for Indian Legal Case Documents

    [https://arxiv.org/abs/2403.13681](https://arxiv.org/abs/2403.13681)

    PARAMANU-AYN是一种基于印度法律案例文件的高效生成式语言模型，采用自回归解码器进行预训练，并经过面向指令的微调，在各种法律任务上取得了良好表现。

    

    在这篇论文中，我们介绍了PARAMANU-AYN，这是一个仅基于印度最高法院案例文件、印度宪法和印度刑法的语言模型。这种新颖的基于自回归（AR）解码器的模型是从头开始在上下文大小为8192的情况下进行预训练的。我们在困惑度指标上评估了我们的预训练法律模型。我们还对一组包括各种法律任务（如法律推理、判决解释、法律条款生成、法律草拟、法律合同草拟、案件摘要、宪法问题回答等）的10,763条指令进行了针对性训练。我们还通过GPT-3.5-Turbo对面向指令的模型的提示响应进行了在10分制度上的清晰度、相关性、完整性和法律推理指标的评估。我们的模型可以在CPU上运行，并实现每秒42.46个令牌的CPU推理速度。

    arXiv:2403.13681v1 Announce Type: new  Abstract: In this paper, we present PARAMANU-AYN, a language model based exclusively on case documents of the Supreme Court of India, the Constitution of India, and the Indian Penal Code. The novel Auto Regressive (AR) decoder based model is pretrained from scratch at a context size of 8192. We evaluated our pretrained legal model on perplexity metrics. We also instruction-tuned our pretrained model on a set of 10,763 instructions covering various legal tasks such as legal reasoning, judgement explanation, legal clause generation, legal drafting, legal contract drafting, case summarization, constitutional question-answering, etc. We also evaluated the responses of prompts for instruction-tuned models by GPT-3.5-Turbo on clarity, relevance, completeness, and legal reasoning metrics in a scale of 10. Our model can be run on CPU and achieved 42.46 tokens/sec CPU inference speed. We found that our models, despite not being pretrained on legal books, v
    
[^4]: ConjNorm：用于异常分布检测的可处理密度估计

    ConjNorm: Tractable Density Estimation for Out-of-Distribution Detection

    [https://arxiv.org/abs/2402.17888](https://arxiv.org/abs/2402.17888)

    提出了一种新颖的理论框架，基于Bregman散度，通过引入共轭约束，提出了一种\textsc{ConjNorm}方法，以在给定数据集中搜索最佳规范系数$p$来重新构想密度函数设计。

    

    后续异常分布（OOD）检测在可靠机器学习中受到密切关注。许多工作致力于推导基于logits、距离或严格数据分布假设的评分函数，以识别得分低的OOD样本。然而，这些估计得分可能无法准确反映真实数据密度或施加不切实际的约束。为了在基于密度得分设计方面提供一个统一的视角，我们提出了一个以Bregman散度为基础的新颖理论框架，该框架将分布考虑扩展到涵盖一系列指数族分布。利用我们定理中揭示的共轭约束，我们引入了一种\textsc{ConjNorm}方法，将密度函数设计重新构想为针对给定数据集搜索最佳规范系数$p$的过程。鉴于归一化的计算挑战，我们设计了一种无偏和解析可追踪的方法

    arXiv:2402.17888v1 Announce Type: cross  Abstract: Post-hoc out-of-distribution (OOD) detection has garnered intensive attention in reliable machine learning. Many efforts have been dedicated to deriving score functions based on logits, distances, or rigorous data distribution assumptions to identify low-scoring OOD samples. Nevertheless, these estimate scores may fail to accurately reflect the true data density or impose impractical constraints. To provide a unified perspective on density-based score design, we propose a novel theoretical framework grounded in Bregman divergence, which extends distribution considerations to encompass an exponential family of distributions. Leveraging the conjugation constraint revealed in our theorem, we introduce a \textsc{ConjNorm} method, reframing density function design as a search for the optimal norm coefficient $p$ against the given dataset. In light of the computational challenges of normalization, we devise an unbiased and analytically tract
    
[^5]: 合并不同过滤器中的证据

    Combining Evidence Across Filtrations

    [https://arxiv.org/abs/2402.09698](https://arxiv.org/abs/2402.09698)

    这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。

    

    在任何时刻有效的顺序推理中，已知任何可接受的推理方法必须基于测试鞅和它们的组合广义化，称为e进程，它们是非负进程，其在任何任意停时的期望上界不超过一。e进程量化了针对复合零假设的一系列结果的累积证据。本文研究了使用不同信息集（即过滤器）计算的e进程的合并方法，针对一个零假设。尽管在相同过滤器上构建的e进程可以轻松地合并（例如，通过平均），但在不同过滤器上构建的e进程不能那么容易地合并，因为它们在较粗的过滤器中的有效性不能转换为在更细的过滤器中的有效性。我们讨论了文献中三个具体例子：可交换性测试，独立性测试等。

    arXiv:2402.09698v1 Announce Type: cross  Abstract: In anytime-valid sequential inference, it is known that any admissible inference procedure must be based on test martingales and their composite generalization, called e-processes, which are nonnegative processes whose expectation at any arbitrary stopping time is upper-bounded by one. An e-process quantifies the accumulated evidence against a composite null hypothesis over a sequence of outcomes. This paper studies methods for combining e-processes that are computed using different information sets, i.e., filtrations, for a null hypothesis. Even though e-processes constructed on the same filtration can be combined effortlessly (e.g., by averaging), e-processes constructed on different filtrations cannot be combined as easily because their validity in a coarser filtration does not translate to validity in a finer filtration. We discuss three concrete examples of such e-processes in the literature: exchangeability tests, independence te
    
[^6]: 深度学习方法在光电容积法数据中的应用综述

    A Review of Deep Learning Methods for Photoplethysmography Data. (arXiv:2401.12783v1 [cs.AI])

    [http://arxiv.org/abs/2401.12783](http://arxiv.org/abs/2401.12783)

    本综述系统地回顾了自2017年至2023年期间应用深度学习模型处理光电容积法数据的论文。研究发现，深度学习在个人健康管理和其他应用中具有显著成果。根据任务的不同，这些论文被分为医学相关和非医学相关两大类别，医学相关又细分为七个子组，包括血压分析...

    

    光电容积法（PPG）是一种非常有前景的设备，因为它具有便携性、用户友好操作和非侵入性测量多种生理信息的能力。深度学习的最新进展，通过利用PPG信号，展示了在个人健康管理和其他多方面应用任务上取得了显著的成果。本综述系统地回顾了自2017年1月1日至2023年7月31日期间在Google学术、PubMed和Dimensions发表的应用深度学习模型处理PPG数据的论文。每篇论文从任务、模型和数据三个关键角度进行分析。最终提取了193篇论文，其中使用了不同的深度学习框架来处理PPG信号。根据这些论文所涉及的任务，我们将它们分为两大类别：医学相关和非医学相关。医学相关任务进一步分为七个子组，包括血压分析...

    Photoplethysmography (PPG) is a highly promising device due to its advantages in portability, user-friendly operation, and non-invasive capabilities to measure a wide range of physiological information. Recent advancements in deep learning have demonstrated remarkable outcomes by leveraging PPG signals for tasks related to personal health management and other multifaceted applications. In this review, we systematically reviewed papers that applied deep learning models to process PPG data between January 1st of 2017 and July 31st of 2023 from Google Scholar, PubMed and Dimensions. Each paper is analyzed from three key perspectives: tasks, models, and data. We finally extracted 193 papers where different deep learning frameworks were used to process PPG signals. Based on the tasks addressed in these papers, we categorized them into two major groups: medical-related, and non-medical-related. The medical-related tasks were further divided into seven subgroups, including blood pressure anal
    
[^7]: 具有图信息的可证明张量补全

    Provable Tensor Completion with Graph Information. (arXiv:2310.02543v1 [cs.LG])

    [http://arxiv.org/abs/2310.02543](http://arxiv.org/abs/2310.02543)

    本文提出了一个创新框架，系统地解决了具有图信息的动态图正则化张量补全问题，建立了严格的数学表示，并推导出了新的图导向补全模型。

    

    图被广泛应用作为变量之间相互关系的有效侧面信息，用于各种矩阵/张量恢复相关应用中的准确数据恢复。本文研究了具有图信息的张量补全问题。目前关于图正则化的张量补全的研究更倾向于特定任务，缺乏普适性和系统化的方法。此外，缺乏保证性能的恢复理论。此外，这些方法忽视了图的动态特性，在张量相关场景中将其视为类似矩阵的静态对象，即使图可能在时间上具有动态性。为了应对这些挑战，在本文中我们介绍了一个创新框架，系统地建立了解决动态图正则化张量补全问题的新模型、理论和算法。对于模型，我们建立了动态图的严格数学表示，基于该表示我们推导出了一个新的针对张量的图导向的补全模型。

    Graphs, depicting the interrelations between variables, has been widely used as effective side information for accurate data recovery in various matrix/tensor recovery related applications. In this paper, we study the tensor completion problem with graph information. Current research on graph-regularized tensor completion tends to be task-specific, lacking generality and systematic approaches. Furthermore, a recovery theory to ensure performance remains absent. Moreover, these approaches overlook the dynamic aspects of graphs, treating them as static akin to matrices, even though graphs could exhibit dynamism in tensor-related scenarios. To confront these challenges, we introduce a pioneering framework in this paper that systematically formulates a novel model, theory, and algorithm for solving the dynamic graph regularized tensor completion problem. For the model, we establish a rigorous mathematical representation of the dynamic graph, based on which we derive a new tensor-oriented g
    
[^8]: 基于Polyak-Lojasiewicz条件的极小极大优化的快速随机算法

    Faster Stochastic Algorithms for Minimax Optimization under Polyak--{\L}ojasiewicz Conditions. (arXiv:2307.15868v1 [math.OC])

    [http://arxiv.org/abs/2307.15868](http://arxiv.org/abs/2307.15868)

    该论文提出了基于Polyak-Lojasiewicz条件的极小极大优化的快速随机算法SPIDER-GDA，并证明了该算法能够在较低的复杂度下找到接近最优解的结果。

    

    本文考虑Polyak-Lojasiewicz (PL)条件下的随机一阶优化算法。我们提出了SPIDER-GDA算法来解决形式为$\min_x \max_y f(x,y)\triangleq \frac{1}{n} \sum_{i=1}^n f_i(x,y)$的有限和问题，其中目标函数$f(x,y)$在$x$方向上是$\mu_x$-PL的，在$y$方向上是$\mu_y$-PL的；每个$f_i(x,y)$都是$L$-平滑的。我们证明了SPIDER-GDA算法可以在${\mathcal O}\left((n + \sqrt{n}\,\kappa_x\kappa_y^2)\log (1/\epsilon)\right)$的随机一阶oracle (SFO)复杂度内找到一个$\epsilon$-最优解，这比现有方法的SFO上界${\mathcal O}\big((n + n^{2/3}\kappa_x\kappa_y^2)\log (1/\epsilon)\big)$更好，其中$\kappa_x\triangleq L/\mu_x$，$\kappa_y\triangleq L/\mu_y$。对于病态情况，我们提供了一种加速算法来进一步降低计算成本。它可以实现$\tilde{{\mathcal O}}\big((n+\sqrt{n}\,\kappa_x\kappa_y)\log^2 (1/\epsilon)\big)$的SFO上界。

    This paper considers stochastic first-order algorithms for minimax optimization under Polyak--{\L}ojasiewicz (PL) conditions. We propose SPIDER-GDA for solving the finite-sum problem of the form $\min_x \max_y f(x,y)\triangleq \frac{1}{n} \sum_{i=1}^n f_i(x,y)$, where the objective function $f(x,y)$ is $\mu_x$-PL in $x$ and $\mu_y$-PL in $y$; and each $f_i(x,y)$ is $L$-smooth. We prove SPIDER-GDA could find an $\epsilon$-optimal solution within ${\mathcal O}\left((n + \sqrt{n}\,\kappa_x\kappa_y^2)\log (1/\epsilon)\right)$ stochastic first-order oracle (SFO) complexity, which is better than the state-of-the-art method whose SFO upper bound is ${\mathcal O}\big((n + n^{2/3}\kappa_x\kappa_y^2)\log (1/\epsilon)\big)$, where $\kappa_x\triangleq L/\mu_x$ and $\kappa_y\triangleq L/\mu_y$. For the ill-conditioned case, we provide an accelerated algorithm to reduce the computational cost further. It achieves $\tilde{{\mathcal O}}\big((n+\sqrt{n}\,\kappa_x\kappa_y)\log^2 (1/\epsilon)\big)$ SFO u
    
[^9]: 量子生成建模中的可训练性障碍和机遇

    Trainability barriers and opportunities in quantum generative modeling. (arXiv:2305.02881v1 [quant-ph])

    [http://arxiv.org/abs/2305.02881](http://arxiv.org/abs/2305.02881)

    本文研究了量子生成模型的可训练性障碍，如荒芜高原和指数损失集中，使用隐式生成模型和明确损失会产生一种新的荒芜高原现象。最大均值差可以是低秩且可训练的或全局性且不可训练的。但是，可训练性所需的低秩损失通常不能区分高频和低频特征。

    

    量子生成模型提供了本质高效的采样策略，因此在量子硬件上实现近期优势有很大的潜力。然而，它们的可扩展性仍存在重要问题。本文研究量子生成模型的可训练性障碍，如荒芜高原和指数损失集中。我们探索了明确和隐含模型、损失之间的相互作用，并表明使用隐式生成模型（如基于量子电路的模型）和明确损失（如KL散度）会产生一种新的荒芜高原现象。相比之下，最大均值差（MMD），作为隐式损失的一个流行例子，可以看作是一个观测量的期望值，该观测量可能是低秩且可训练的，也可能是全局性且不可训练的，具体取决于核函数的选择。然而，我们同时强调，可训练性所需的低秩损失通常不能区分高频和低频特征。

    Quantum generative models, in providing inherently efficient sampling strategies, show promise for achieving a near-term advantage on quantum hardware. Nonetheless, important questions remain regarding their scalability. In this work, we investigate the barriers to the trainability of quantum generative models posed by barren plateaus and exponential loss concentration. We explore the interplay between explicit and implicit models and losses, and show that using implicit generative models (such as quantum circuit-based models) with explicit losses (such as the KL divergence) leads to a new flavour of barren plateau. In contrast, the Maximum Mean Discrepancy (MMD), which is a popular example of an implicit loss, can be viewed as the expectation value of an observable that is either low-bodied and trainable, or global and untrainable depending on the choice of kernel. However, in parallel, we highlight that the low-bodied losses required for trainability cannot in general distinguish hig
    
[^10]: Skeleton Regression：一种基于流形结构估计的基于图形的方法。

    Skeleton Regression: A Graph-Based Approach to Estimation with Manifold Structure. (arXiv:2303.11786v1 [cs.LG])

    [http://arxiv.org/abs/2303.11786](http://arxiv.org/abs/2303.11786)

    这是一个处理低维流形数据的回归框架，首先通过构建图形骨架来捕捉潜在的流形几何结构，然后在其上运用非参数回归技术来估计回归函数，除了具有非参数优点之外，在处理多个流形数据，嘈杂观察时也表现出较好的鲁棒性。

    

    我们引入了一个新的回归框架，旨在处理围绕低维流形的复杂数据。我们的方法首先构建一个图形表示，称为骨架，以捕获潜在的几何结构。然后，我们在骨架图上定义指标，应用非参数回归技术，以及基于图形的特征转换来估计回归函数。除了包括的非参数方法外，我们还讨论了一些非参数回归器在骨架图等一般度量空间方面的限制。所提出的回归框架使我们能够避开维度灾难，具有可以处理多个流形的并集并且鲁棒性能应对加性噪声和嘈杂观察的额外优势。我们为所提出的方法提供了统计保证，并通过模拟和实际数据示例证明了其有效性。

    We introduce a new regression framework designed to deal with large-scale, complex data that lies around a low-dimensional manifold. Our approach first constructs a graph representation, referred to as the skeleton, to capture the underlying geometric structure. We then define metrics on the skeleton graph and apply nonparametric regression techniques, along with feature transformations based on the graph, to estimate the regression function. In addition to the included nonparametric methods, we also discuss the limitations of some nonparametric regressors with respect to the general metric space such as the skeleton graph. The proposed regression framework allows us to bypass the curse of dimensionality and provides additional advantages that it can handle the union of multiple manifolds and is robust to additive noise and noisy observations. We provide statistical guarantees for the proposed method and demonstrate its effectiveness through simulations and real data examples.
    
[^11]: 离线估计控制马尔可夫链：最小化和样本复杂度

    Offline Estimation of Controlled Markov Chains: Minimaxity and Sample Complexity. (arXiv:2211.07092v4 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.07092](http://arxiv.org/abs/2211.07092)

    本论文研究了离线估计有限控制马尔可夫链的转移概率矩阵的非参数估计器，并通过记录策略的混合特性建立了样本复杂性界限和最小化条件。结果表明，实现特定的统计风险界限涉及到混合特性的强度和样本数量之间微妙而有趣的权衡。还使用这些样本复杂性界限建立了离线评估恒定马尔可夫控制策略的相关界限。

    

    在这项工作中，我们研究了有限控制马尔可夫链的转移概率矩阵的自然非参数估计器。我们考虑了一个固定数据集的离线设置，该数据集是使用所谓的记录策略收集的。我们为估计器开发了样本复杂性的界限，并建立了最小化的条件。我们的统计界限通过记录策略的混合特性来确定。我们表明，实现特定的统计风险界限涉及到混合特性的强度和样本数量之间微妙而有趣的权衡。我们在各种示例中验证了我们结果的有效性，包括遗传马尔可夫链，弱遗传非齐次马尔可夫链和具有非平稳马尔可夫、阶段性和贪婪控制的控制马尔可夫链。最后，我们使用这些样本复杂性界限来建立离线评估恒定马尔可夫控制策略的相关界限。

    In this work, we study a natural nonparametric estimator of the transition probability matrices of a finite controlled Markov chain. We consider an offline setting with a fixed dataset, collected using a so-called logging policy. We develop sample complexity bounds for the estimator and establish conditions for minimaxity. Our statistical bounds depend on the logging policy through its mixing properties. We show that achieving a particular statistical risk bound involves a subtle and interesting trade-off between the strength of the mixing properties and the number of samples. We demonstrate the validity of our results under various examples, such as ergodic Markov chains, weakly ergodic inhomogeneous Markov chains, and controlled Markov chains with non-stationary Markov, episodic, and greedy controls. Lastly, we use these sample complexity bounds to establish concomitant ones for offline evaluation of stationary Markov control policies.
    
[^12]: 二次梯度：将梯度算法和牛顿法融合为一体

    Quadratic Gradient: Combining Gradient Algorithms and Newton's Method as One. (arXiv:2209.03282v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2209.03282](http://arxiv.org/abs/2209.03282)

    本文提出了一种基于对角矩阵的二次梯度，可以加速梯度的收敛速度，在实验中表现良好。研究者还推测海森矩阵与学习率之间可能存在关系。

    

    使用一列与梯度相同大小的列向量，而不是仅使用一个浮点数来加速每个梯度元素的不同速率，可能对牛顿法的线搜索技术不足。此外，使用一个与海森矩阵大小相同的正方形矩阵来纠正海森矩阵可能是有用的。Chiang提出了一种介于列向量和正方形矩阵之间的东西，即对角矩阵，来加速梯度，并进一步提出了一种更快的梯度变体，称为二次梯度。在本文中，我们提出一种构建新版本的二次梯度的新方法。这个新的二次梯度不满足固定海森牛顿法的收敛条件。然而，实验结果显示，它有时比原始方法的收敛速度更快。此外，Chiang推测海森矩阵与学习率f之间可能存在关系。

    It might be inadequate for the line search technique for Newton's method to use only one floating point number. A column vector of the same size as the gradient might be better than a mere float number to accelerate each of the gradient elements with different rates. Moreover, a square matrix of the same order as the Hessian matrix might be helpful to correct the Hessian matrix. Chiang applied something between a column vector and a square matrix, namely a diagonal matrix, to accelerate the gradient and further proposed a faster gradient variant called quadratic gradient. In this paper, we present a new way to build a new version of the quadratic gradient. This new quadratic gradient doesn't satisfy the convergence conditions of the fixed Hessian Newton's method. However, experimental results show that it sometimes has a better performance than the original one in convergence rate. Also, Chiang speculates that there might be a relation between the Hessian matrix and the learning rate f
    

