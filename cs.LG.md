# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On a Neural Implementation of Brenier's Polar Factorization](https://arxiv.org/abs/2403.03071) | 提出了Brenier的极分解定理的神经实现，探讨了在机器学习中的应用，并通过神经网络参数化潜在函数$u$，从最新神经最优输运领域的进展中汲取灵感。 |
| [^2] | [An Inexact Halpern Iteration for with Application to Distributionally Robust Optimization](https://arxiv.org/abs/2402.06033) | 本文研究了确定性和随机环境下Halpern迭代算法的不精确变种，通过适当选择不精确的容差，这些变种展现出O(k^-1)的收敛速度，同时具有竞争性的收敛特性。并且我们还展示了这些方法在两类数据驱动Wasserstein分布鲁棒优化问题中的应用，以及在分布鲁棒学习中使用随机一阶方法进行不精确计算的能力。 |
| [^3] | [Retrieve to Explain: Evidence-driven Predictions with Language Models](https://arxiv.org/abs/2402.04068) | 检索以解释（R2E）是一种基于语言模型的检索方法，通过使用Shapley值确定证据的相对重要性，从而在黑盒模型中提供了可解释性，通过应用于药物靶点鉴定任务中，R2E模型在预测临床试验结果方面优于传统基因学方法。 |
| [^4] | [Dual-Directed Algorithm Design for Efficient Pure Exploration.](http://arxiv.org/abs/2310.19319) | 该论文研究了在有限备选方案集合中的纯探索问题。通过使用对偶变量，提出了一种新的算法设计原则，能够避免组合结构的复杂性，实现高效纯探索，从而准确回答查询问题。 |
| [^5] | [A Data-Centric Online Market for Machine Learning: From Discovery to Pricing.](http://arxiv.org/abs/2310.17843) | 这篇论文介绍了一种以数据为中心的在线市场，用于连接机器学习的供求匹配，并提出了解决这个市场设计中的两个核心挑战的新技术。 |
| [^6] | [Quantum Acceleration of Infinite Horizon Average-Reward Reinforcement Learning.](http://arxiv.org/abs/2310.11684) | 本研究探索了无限时域平均奖励强化学习中量子加速的潜力。我们提出了一种创新的量子框架，通过高效的量子均值估计技术，实现了指数级改进的遗憾保证。所提出的量子算法相较于经典算法，在遗憾界限上有显著改进。 |
| [^7] | [CLEVRER-Humans: Describing Physical and Causal Events the Human Way.](http://arxiv.org/abs/2310.03635) | CLEVRER-Humans是一个用于因果判断的视频推理数据集，通过人工标注来解决合成事件和合成语言描述的缺乏多样性问题，并通过迭代事件填空和神经语言生成模型提高数据收集效率。 |
| [^8] | [Distilling Inductive Bias: Knowledge Distillation Beyond Model Compression.](http://arxiv.org/abs/2310.00369) | 本文提出了一种超越模型压缩的知识蒸馏方法，通过从轻量级教师模型中提取归纳偏差，使Vision Transformers (ViTs) 的应用成为可能。这种方法包括使用一组不同架构的教师模型来指导学生Transformer，从而有效提高学生的性能。 |
| [^9] | [Graph topological property recovery with heat and wave dynamics-based features on graphsD.](http://arxiv.org/abs/2309.09924) | 本文提出了一种名为图微分方程网络（GDeNet）的方法，利用热和波动方程动力学特征来恢复图的拓扑属性，能够在各种下游任务中获得优秀的表现，同时在实际应用中也展现了较好的性能。 |
| [^10] | [Pointing the Way: Refining Radar-Lidar Localization Using Learned ICP Weights.](http://arxiv.org/abs/2309.08731) | 本文提出了一种深度学习方法，通过学习到的ICP权重优化雷达-激光雷达的定位，从而改善了雷达测量对激光雷达地图的定位效果。这一方法在保持高质量地图定位性能的同时，提高了在降水和大雾等恶劣天气条件下的定位准确性。 |
| [^11] | [Clustered Multi-Agent Linear Bandits.](http://arxiv.org/abs/2309.08710) | 本文研究了集群化的多智能体线性赌博机问题，提出了一种新颖的算法，通过智能体之间的协作来加速优化问题。通过理论分析和实证评估，证明了算法在遗憾最小化和聚类质量上的有效性。 |
| [^12] | [Learning under Selective Labels with Heterogeneous Decision-makers: An Instrumental Variable Approach.](http://arxiv.org/abs/2306.07566) | 本文提出了一种处理选择性标记数据的学习问题的方法。通过利用历史决策由一组异质决策者做出的事实，我们建立了一种有原理的工具变量框架，并提出了一种加权学习方法，用于学习预测规则。 |
| [^13] | [Time Series Classification for Detecting Parkinson's Disease from Wrist Motions.](http://arxiv.org/abs/2304.11265) | 该研究使用InceptionTime和ROCKET方法进行时间序列分类，以监测帕金森病患者的手腕运动。研究发现，所有方法都适用于估计震颤严重程度和肌肉强直的存在，但在检测运动障碍方面存在困难。具有岭分类器的InceptionTime方法展示了最先进的分类性能，显示时间序列分类在基于可穿戴设备的PD症状监测中具有潜力。 |

# 详细

[^1]: 论Brenier的极分解的神经实现

    On a Neural Implementation of Brenier's Polar Factorization

    [https://arxiv.org/abs/2403.03071](https://arxiv.org/abs/2403.03071)

    提出了Brenier的极分解定理的神经实现，探讨了在机器学习中的应用，并通过神经网络参数化潜在函数$u$，从最新神经最优输运领域的进展中汲取灵感。

    

    在1991年，Brenier证明了一个定理，将$QR$分解（分为半正定矩阵$\times$酉矩阵）推广到任意矢量场$F:\mathbb{R}^d\rightarrow \mathbb{R}^d$。这个被称为极分解定理的定理表明，任意场$F$都可以表示为凸函数$u$的梯度与保测度映射$M$的复合，即$F=\nabla u \circ M$。我们提出了这一具有深远理论意义的结果的实际实现，并探讨了在机器学习中可能的应用。该定理与最优输运（OT）理论密切相关，我们借鉴了神经最优输运领域的最新进展，将潜在函数$u$参数化为输入凸神经网络。映射$M$可以通过使用$u^*$，即$u$的凸共轭，逐点计算得到，即$M=\nabla u^* \circ F$，或者作为辅助网络学习得到。因为$M$在基因

    arXiv:2403.03071v1 Announce Type: cross  Abstract: In 1991, Brenier proved a theorem that generalizes the $QR$ decomposition for square matrices -- factored as PSD $\times$ unitary -- to any vector field $F:\mathbb{R}^d\rightarrow \mathbb{R}^d$. The theorem, known as the polar factorization theorem, states that any field $F$ can be recovered as the composition of the gradient of a convex function $u$ with a measure-preserving map $M$, namely $F=\nabla u \circ M$. We propose a practical implementation of this far-reaching theoretical result, and explore possible uses within machine learning. The theorem is closely related to optimal transport (OT) theory, and we borrow from recent advances in the field of neural optimal transport to parameterize the potential $u$ as an input convex neural network. The map $M$ can be either evaluated pointwise using $u^*$, the convex conjugate of $u$, through the identity $M=\nabla u^* \circ F$, or learned as an auxiliary network. Because $M$ is, in gene
    
[^2]: 不精确的Halpern迭代算法及其在分布鲁棒优化中的应用

    An Inexact Halpern Iteration for with Application to Distributionally Robust Optimization

    [https://arxiv.org/abs/2402.06033](https://arxiv.org/abs/2402.06033)

    本文研究了确定性和随机环境下Halpern迭代算法的不精确变种，通过适当选择不精确的容差，这些变种展现出O(k^-1)的收敛速度，同时具有竞争性的收敛特性。并且我们还展示了这些方法在两类数据驱动Wasserstein分布鲁棒优化问题中的应用，以及在分布鲁棒学习中使用随机一阶方法进行不精确计算的能力。

    

    Halpern迭代算法因其简单形式和吸引人的收敛性质，近年来在解决单调包含问题方面引起了越来越多的关注。本文研究了确定性和随机环境下该方案的不精确变种。我们进行了广泛的收敛性分析，并表明通过适当选择不精确的容差，不精确方案在（期望的）残差范数上具有O(k^-1)的收敛速度。我们的结果放宽了文献中采用的最新不精确性条件，同时具有相同的竞争性收敛特性。然后，我们演示了如何使用所提出的方法解决两类具有凸凹最小-最大优化重构的数据驱动Wasserstein分布鲁棒优化问题。我们强调了其在使用随机一阶方法进行分布鲁棒学习中的不精确计算能力。

    The Halpern iteration for solving monotone inclusion problems has gained increasing interests in recent years due to its simple form and appealing convergence properties. In this paper, we investigate the inexact variants of the scheme in both deterministic and stochastic settings. We conduct extensive convergence analysis and show that by choosing the inexactness tolerances appropriately, the inexact schemes admit an $O(k^{-1})$ convergence rate in terms of the (expected) residue norm. Our results relax the state-of-the-art inexactness conditions employed in the literature while sharing the same competitive convergence properties. We then demonstrate how the proposed methods can be applied for solving two classes of data-driven Wasserstein distributionally robust optimization problems that admit convex-concave min-max optimization reformulations. We highlight its capability of performing inexact computations for distributionally robust learning with stochastic first-order methods.
    
[^3]: 检索以解释：基于语言模型的证据驱动预测

    Retrieve to Explain: Evidence-driven Predictions with Language Models

    [https://arxiv.org/abs/2402.04068](https://arxiv.org/abs/2402.04068)

    检索以解释（R2E）是一种基于语言模型的检索方法，通过使用Shapley值确定证据的相对重要性，从而在黑盒模型中提供了可解释性，通过应用于药物靶点鉴定任务中，R2E模型在预测临床试验结果方面优于传统基因学方法。

    

    机器学习模型，尤其是语言模型，往往难以深入分析。黑盒模型可能掩盖了模型训练中的问题和有害偏差。对于人机协作过程来说，不透明的预测可能导致缺乏信任，限制模型的影响，即使模型的性能很好。为了解决这些问题，我们引入了检索以解释（Retrieve to Explain，简称R2E）。R2E是一种基于检索的语言模型，根据文档语料库中的证据，使用Shapley值来确定证据对最终预测的相对重要性，并根据自然语言模板将结构化数据纳入其中。R2E能够在不重新训练的情况下适应新的证据，并且能够通过模板化将结构化数据纳入到自然语言中。我们在通过分析已发表的科学文献进行药物靶点鉴定的实际案例中进行了评估，结果显示该模型在预测临床试验结果方面优于行业标准的基因学方法。

    Machine learning models, particularly language models, are notoriously difficult to introspect. Black-box models can mask both issues in model training and harmful biases. For human-in-the-loop processes, opaque predictions can drive lack of trust, limiting a model's impact even when it performs effectively. To address these issues, we introduce Retrieve to Explain (R2E). R2E is a retrieval-based language model that prioritizes amongst a pre-defined set of possible answers to a research question based on the evidence in a document corpus, using Shapley values to identify the relative importance of pieces of evidence to the final prediction. R2E can adapt to new evidence without retraining, and incorporate structured data through templating into natural language. We assess on the use case of drug target identification from published scientific literature, where we show that the model outperforms an industry-standard genetics-based approach on predicting clinical trial outcomes.
    
[^4]: 高效纯探索的双向算法设计

    Dual-Directed Algorithm Design for Efficient Pure Exploration. (arXiv:2310.19319v1 [stat.ML])

    [http://arxiv.org/abs/2310.19319](http://arxiv.org/abs/2310.19319)

    该论文研究了在有限备选方案集合中的纯探索问题。通过使用对偶变量，提出了一种新的算法设计原则，能够避免组合结构的复杂性，实现高效纯探索，从而准确回答查询问题。

    

    我们考虑在有限的备选方案集合中的随机顺序自适应实验的纯探索问题。决策者的目标是通过最小的测量工作以高置信度准确回答与备选方案相关的查询问题。一个典型的查询问题是确定表现最佳的备选方案，这在排名和选择问题以及机器学习文献中称为最佳臂识别问题。我们专注于固定精度的设定，并导出了一个与样本最优分配有强收敛性概念相关的优化条件的充分条件。使用对偶变量，我们刻画了一个分配是否最优的必要和充分条件。对偶变量的使用使我们能够绕过完全依赖于原始变量的最优条件的组合结构。值得注意的是，这些最优条件使得双向算法设计原则的扩展成为可能。

    We consider pure-exploration problems in the context of stochastic sequential adaptive experiments with a finite set of alternative options. The goal of the decision-maker is to accurately answer a query question regarding the alternatives with high confidence with minimal measurement efforts. A typical query question is to identify the alternative with the best performance, leading to ranking and selection problems, or best-arm identification in the machine learning literature. We focus on the fixed-precision setting and derive a sufficient condition for optimality in terms of a notion of strong convergence to the optimal allocation of samples. Using dual variables, we characterize the necessary and sufficient conditions for an allocation to be optimal. The use of dual variables allow us to bypass the combinatorial structure of the optimality conditions that relies solely on primal variables. Remarkably, these optimality conditions enable an extension of top-two algorithm design princ
    
[^5]: 一种以数据为中心的机器学习在线市场：从发现到定价

    A Data-Centric Online Market for Machine Learning: From Discovery to Pricing. (arXiv:2310.17843v1 [cs.LG])

    [http://arxiv.org/abs/2310.17843](http://arxiv.org/abs/2310.17843)

    这篇论文介绍了一种以数据为中心的在线市场，用于连接机器学习的供求匹配，并提出了解决这个市场设计中的两个核心挑战的新技术。

    

    数据是机器学习的动力 - 丰富和高质量的训练数据对于机器学习的成功至关重要。然而，要将机器学习从少数大型公司之间的竞赛转变为为众多普通用户的数据分析请求服务的可访问技术，仍然存在重要的挑战。我们观察到的一个差距是，许多机器学习用户可以从其他数据所有者拥有的新数据中受益，而这些数据所有者却坐在一堆数据上，不知道谁可以受益于它。这种差距为构建一个能够自动连接供求的在线市场创造了机会。虽然在线匹配市场很常见（例如，打车系统），但为机器学习设计一个以数据为中心的市场面临许多前所未有的挑战。本文开发了新的技术来解决设计这样一个市场中的两个核心挑战：（a）为了高效地将需求与供应匹配，我们设计了一种算法，可以从数千个数据池中自动发现任何机器学习任务所需的有用数据。

    Data fuels machine learning (ML) - rich and high-quality training data is essential to the success of ML. However, to transform ML from the race among a few large corporations to an accessible technology that serves numerous normal users' data analysis requests, there still exist important challenges. One gap we observed is that many ML users can benefit from new data that other data owners possess, whereas these data owners sit on piles of data without knowing who can benefit from it. This gap creates the opportunity for building an online market that can automatically connect supply with demand. While online matching markets are prevalent (e.g., ride-hailing systems), designing a data-centric market for ML exhibits many unprecedented challenges.  This paper develops new techniques to tackle two core challenges in designing such a market: (a) to efficiently match demand with supply, we design an algorithm to automatically discover useful data for any ML task from a pool of thousands o
    
[^6]: 无限时域平均奖励强化学习的量子加速

    Quantum Acceleration of Infinite Horizon Average-Reward Reinforcement Learning. (arXiv:2310.11684v1 [cs.LG])

    [http://arxiv.org/abs/2310.11684](http://arxiv.org/abs/2310.11684)

    本研究探索了无限时域平均奖励强化学习中量子加速的潜力。我们提出了一种创新的量子框架，通过高效的量子均值估计技术，实现了指数级改进的遗憾保证。所提出的量子算法相较于经典算法，在遗憾界限上有显著改进。

    

    本文研究量子加速在解决无限时域Markov决策过程（MDPs）中提高平均奖励结果的潜力。我们引入了一种创新的量子框架，用于代理与未知MDP的互动，扩展了传统的交互范式。我们的方法涉及设计一种基于乐观主导的具有量子信号的表格强化学习算法，通过高效的量子均值估计技术获取代理获取的量子信号。通过深入的理论分析，我们证明了量子均值估计的优势能够在无限时域强化学习中导致遗憾保证的指数进展。具体地，所提出的量子算法实现了一个遗憾界为$\tilde{\mathcal{O}}(1)$的性能，这是相对于经典对应算法所展示的$\tilde{\mathcal{O}}(\sqrt{T})$界限的显著改进。

    This paper investigates the potential of quantum acceleration in addressing infinite horizon Markov Decision Processes (MDPs) to enhance average reward outcomes. We introduce an innovative quantum framework for the agent's engagement with an unknown MDP, extending the conventional interaction paradigm. Our approach involves the design of an optimism-driven tabular Reinforcement Learning algorithm that harnesses quantum signals acquired by the agent through efficient quantum mean estimation techniques. Through thorough theoretical analysis, we demonstrate that the quantum advantage in mean estimation leads to exponential advancements in regret guarantees for infinite horizon Reinforcement Learning. Specifically, the proposed Quantum algorithm achieves a regret bound of $\tilde{\mathcal{O}}(1)$, a significant improvement over the $\tilde{\mathcal{O}}(\sqrt{T})$ bound exhibited by classical counterparts.
    
[^7]: CLEVRER-Humans: 用人类的方式描述物理和因果事件

    CLEVRER-Humans: Describing Physical and Causal Events the Human Way. (arXiv:2310.03635v1 [cs.AI])

    [http://arxiv.org/abs/2310.03635](http://arxiv.org/abs/2310.03635)

    CLEVRER-Humans是一个用于因果判断的视频推理数据集，通过人工标注来解决合成事件和合成语言描述的缺乏多样性问题，并通过迭代事件填空和神经语言生成模型提高数据收集效率。

    

    构建能够推理物理事件及其因果关系的机器对于与物理世界进行灵活互动非常重要。然而，现有的大多数物理和因果推理基准都仅基于合成事件和合成自然语言描述的因果关系。这种设计存在两个问题：一是事件类型和自然语言描述缺乏多样性；二是基于手动定义的启发式规则的因果关系与人类判断不一致。为了解决这两个问题，我们提出了CLEVRER-Humans基准，这是一个用人工标注的视频推理数据集，用于对物理事件的因果判断。我们采用了两种技术来提高数据收集效率：首先，一种新颖的迭代事件填空任务，以 eliciting 视频中事件的新表示方式，我们称之为因果事件图 (CEGs)；其次，一种基于神经语言生成模型的数据增强技术。

    Building machines that can reason about physical events and their causal relationships is crucial for flexible interaction with the physical world. However, most existing physical and causal reasoning benchmarks are exclusively based on synthetically generated events and synthetic natural language descriptions of causal relationships. This design brings up two issues. First, there is a lack of diversity in both event types and natural language descriptions; second, causal relationships based on manually-defined heuristics are different from human judgments. To address both shortcomings, we present the CLEVRER-Humans benchmark, a video reasoning dataset for causal judgment of physical events with human labels. We employ two techniques to improve data collection efficiency: first, a novel iterative event cloze task to elicit a new representation of events in videos, which we term Causal Event Graphs (CEGs); second, a data augmentation technique based on neural language generative models.
    
[^8]: 提炼归纳偏差：超越模型压缩的知识蒸馏

    Distilling Inductive Bias: Knowledge Distillation Beyond Model Compression. (arXiv:2310.00369v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.00369](http://arxiv.org/abs/2310.00369)

    本文提出了一种超越模型压缩的知识蒸馏方法，通过从轻量级教师模型中提取归纳偏差，使Vision Transformers (ViTs) 的应用成为可能。这种方法包括使用一组不同架构的教师模型来指导学生Transformer，从而有效提高学生的性能。

    

    随着计算机视觉的快速发展，Vision Transformers (ViTs) 提供了在视觉和文本领域中实现统一信息处理的诱人前景。但是由于ViTs缺乏固有的归纳偏差，它们需要大量的训练数据。为了使它们的应用实际可行，我们引入了一种创新的基于集成的蒸馏方法，从轻量级的教师模型中提取归纳偏差。以前的系统仅依靠基于卷积的教学方法。然而，这种方法将一组具有不同架构倾向的轻量级教师模型（例如卷积和非线性卷积）同时用于指导学生Transformer。由于这些独特的归纳偏差，教师模型可以从各种存储数据集中获得广泛的知识，从而提高学生的性能。我们提出的框架还涉及预先计算和存储logits，从根本上实现了非归一化的状态匹配。

    With the rapid development of computer vision, Vision Transformers (ViTs) offer the tantalizing prospect of unified information processing across visual and textual domains. But due to the lack of inherent inductive biases in ViTs, they require enormous amount of data for training. To make their applications practical, we introduce an innovative ensemble-based distillation approach distilling inductive bias from complementary lightweight teacher models. Prior systems relied solely on convolution-based teaching. However, this method incorporates an ensemble of light teachers with different architectural tendencies, such as convolution and involution, to instruct the student transformer jointly. Because of these unique inductive biases, instructors can accumulate a wide range of knowledge, even from readily identifiable stored datasets, which leads to enhanced student performance. Our proposed framework also involves precomputing and storing logits in advance, essentially the unnormalize
    
[^9]: 基于热和波动动力学特征的图拓扑属性恢复

    Graph topological property recovery with heat and wave dynamics-based features on graphsD. (arXiv:2309.09924v1 [cs.LG])

    [http://arxiv.org/abs/2309.09924](http://arxiv.org/abs/2309.09924)

    本文提出了一种名为图微分方程网络（GDeNet）的方法，利用热和波动方程动力学特征来恢复图的拓扑属性，能够在各种下游任务中获得优秀的表现，同时在实际应用中也展现了较好的性能。

    

    本文提出了一种名为图微分方程网络（GDeNet）的方法，利用图上的PDE解的表达能力，为各种下游任务获得连续的节点和图级表示。我们推导出了热和波动方程动力学与图的谱特性以及连续时间随机游走在图上行为之间的理论结果。我们通过恢复随机图生成参数、Ricci曲率和持久同调等方式实验证明了这些动力学能够捕捉到图形几何和拓扑的显著方面。此外，我们还展示了GDeNet在包括引用图、药物分子和蛋白质在内的真实世界数据集上的优越性能。

    In this paper, we propose Graph Differential Equation Network (GDeNet), an approach that harnesses the expressive power of solutions to PDEs on a graph to obtain continuous node- and graph-level representations for various downstream tasks. We derive theoretical results connecting the dynamics of heat and wave equations to the spectral properties of the graph and to the behavior of continuous-time random walks on graphs. We demonstrate experimentally that these dynamics are able to capture salient aspects of graph geometry and topology by recovering generating parameters of random graphs, Ricci curvature, and persistent homology. Furthermore, we demonstrate the superior performance of GDeNet on real-world datasets including citation graphs, drug-like molecules, and proteins.
    
[^10]: 指引的方法：利用学习到的ICP权重改进雷达-激光雷达定位

    Pointing the Way: Refining Radar-Lidar Localization Using Learned ICP Weights. (arXiv:2309.08731v1 [cs.RO])

    [http://arxiv.org/abs/2309.08731](http://arxiv.org/abs/2309.08731)

    本文提出了一种深度学习方法，通过学习到的ICP权重优化雷达-激光雷达的定位，从而改善了雷达测量对激光雷达地图的定位效果。这一方法在保持高质量地图定位性能的同时，提高了在降水和大雾等恶劣天气条件下的定位准确性。

    

    本文提出了一种基于深度学习的新方法，用于改进雷达测量对激光雷达地图的定位。虽然目前定位的技术水平是将激光雷达数据与激光雷达地图进行匹配，但是雷达被认为是一种有前途的替代方法，因为它对降水和大雾等恶劣天气具有更强的韧性。为了利用现有的高质量激光雷达地图，同时在恶劣天气下保持性能，将雷达数据与激光雷达地图进行匹配具有重要意义。然而，由于雷达测量中存在的独特伪影，雷达-激光雷达定位一直难以达到与激光雷达-激光雷达系统相媲美的性能，使其无法用于自动驾驶。本工作在基于ICP的雷达-激光雷达定位系统基础上，包括一个学习的预处理步骤，根据高层次的扫描信息对雷达点进行加权。将经过验证的分析方法与学习到的权重相结合，减小了雷达定位中的误差。

    This paper presents a novel deep-learning-based approach to improve localizing radar measurements against lidar maps. Although the state of the art for localization is matching lidar data to lidar maps, radar has been considered as a promising alternative, as it is potentially more resilient against adverse weather such as precipitation and heavy fog. To make use of existing high-quality lidar maps, while maintaining performance in adverse weather, matching radar data to lidar maps is of interest. However, owing in part to the unique artefacts present in radar measurements, radar-lidar localization has struggled to achieve comparable performance to lidar-lidar systems, preventing it from being viable for autonomous driving. This work builds on an ICP-based radar-lidar localization system by including a learned preprocessing step that weights radar points based on high-level scan information. Combining a proven analytical approach with a learned weight reduces localization errors in rad
    
[^11]: 集群化的多智能体线性赌博机

    Clustered Multi-Agent Linear Bandits. (arXiv:2309.08710v1 [cs.LG])

    [http://arxiv.org/abs/2309.08710](http://arxiv.org/abs/2309.08710)

    本文研究了集群化的多智能体线性赌博机问题，提出了一种新颖的算法，通过智能体之间的协作来加速优化问题。通过理论分析和实证评估，证明了算法在遗憾最小化和聚类质量上的有效性。

    

    本文针对多智能体线性随机赌博问题的一个特定实例，即集群化的多智能体线性赌博机进行了研究。在这个设置中，我们提出了一种新颖的算法，通过智能体之间的有效协作来加速整体优化问题。在这一贡献中，网络控制器负责估计网络的基本集群结构并优化同一组中智能体之间的经验分享。我们对遗憾最小化问题和聚类质量进行了理论分析。通过对合成数据和真实数据进行与最先进算法的实证评估，我们证明了我们方法的有效性：我们的算法显著改善了遗憾最小化，并成功恢复了真实的基本集群划分。

    We address in this paper a particular instance of the multi-agent linear stochastic bandit problem, called clustered multi-agent linear bandits. In this setting, we propose a novel algorithm leveraging an efficient collaboration between the agents in order to accelerate the overall optimization problem. In this contribution, a network controller is responsible for estimating the underlying cluster structure of the network and optimizing the experiences sharing among agents within the same groups. We provide a theoretical analysis for both the regret minimization problem and the clustering quality. Through empirical evaluation against state-of-the-art algorithms on both synthetic and real data, we demonstrate the effectiveness of our approach: our algorithm significantly improves regret minimization while managing to recover the true underlying cluster partitioning.
    
[^12]: 学习选择标签下的异质决策者：一种工具变量方法

    Learning under Selective Labels with Heterogeneous Decision-makers: An Instrumental Variable Approach. (arXiv:2306.07566v1 [stat.ML])

    [http://arxiv.org/abs/2306.07566](http://arxiv.org/abs/2306.07566)

    本文提出了一种处理选择性标记数据的学习问题的方法。通过利用历史决策由一组异质决策者做出的事实，我们建立了一种有原理的工具变量框架，并提出了一种加权学习方法，用于学习预测规则。

    

    我们研究了在选择性标记数据下的学习问题。这种问题在历史决策导致结果仅部分标记时出现。标记数据分布可能与整体人群有显著差异，特别是当历史决策和目标结果可以同时受某些未观察到的因素影响时。因此，仅基于标记数据进行学习可能会导致在整体人群中的严重偏差。我们的论文通过利用许多应用中历史决策由一组异质决策者做出的事实来解决此挑战。具体而言，我们在一个有原理的工具变量框架下分析了这种设置。我们建立了满足观察到的数据时任何给定预测规则的全体风险的点识别条件，并在点识别失败时提供了尖锐的风险界限。我们进一步提出了一种加权学习方法，用于学习预测规则。

    We study the problem of learning with selectively labeled data, which arises when outcomes are only partially labeled due to historical decision-making. The labeled data distribution may substantially differ from the full population, especially when the historical decisions and the target outcome can be simultaneously affected by some unobserved factors. Consequently, learning with only the labeled data may lead to severely biased results when deployed to the full population. Our paper tackles this challenge by exploiting the fact that in many applications the historical decisions were made by a set of heterogeneous decision-makers. In particular, we analyze this setup in a principled instrumental variable (IV) framework. We establish conditions for the full-population risk of any given prediction rule to be point-identified from the observed data and provide sharp risk bounds when the point identification fails. We further propose a weighted learning approach that learns prediction ru
    
[^13]: 手腕动作时间序列分类用于帕金森病检测

    Time Series Classification for Detecting Parkinson's Disease from Wrist Motions. (arXiv:2304.11265v1 [cs.LG])

    [http://arxiv.org/abs/2304.11265](http://arxiv.org/abs/2304.11265)

    该研究使用InceptionTime和ROCKET方法进行时间序列分类，以监测帕金森病患者的手腕运动。研究发现，所有方法都适用于估计震颤严重程度和肌肉强直的存在，但在检测运动障碍方面存在困难。具有岭分类器的InceptionTime方法展示了最先进的分类性能，显示时间序列分类在基于可穿戴设备的PD症状监测中具有潜力。

    

    帕金森病是一种神经退行性疾病，具有频繁变化的运动症状，持续的症状监测可以实现更有针对性的治疗。传统的时间序列分类和深度学习技术在使用可穿戴加速度计数据进行PD症状监测时性能有限，因为PD运动模式具有复杂性，但数据集很小。我们研究了InceptionTime和RandOm卷积核变换（ROCKET），因为它们是TSC的最新技术，并且对于PD症状监测非常有前景：InceptionTime的高学习能力适用于建模复杂运动模式，而ROCKET适用于小数据集。我们使用随机搜索找到了最高得分的InceptionTime结构，并将其与具有岭分类器和多层感知器（MLP）的ROCKET进行了比较，用于PD患者的手腕运动。我们发现，所有方法都适用于估计震颤严重程度和肌肉强直的存在，但在检测运动障碍方面存在困难。具有岭分类器的InceptionTime优于其他方法，并实现了最先进的分类性能，展示了TSC在基于可穿戴设备的PD症状监测中的潜力。

    Parkinson's disease (PD) is a neurodegenerative disease with frequently changing motor symptoms where continuous symptom monitoring enables more targeted treatment. Classical time series classification (TSC) and deep learning techniques have limited performance for PD symptom monitoring using wearable accelerometer data because PD movement patterns are complex, but datasets are small. We investigate InceptionTime and RandOm Convolutional KErnel Transform (ROCKET) because they are state-of-the-art for TSC and promising for PD symptom monitoring: InceptionTime's high learning capacity is suited to modeling complex movement patterns while ROCKET is suited to small datasets. We used a random search to find the highest-scoring InceptionTime architecture and compared it to ROCKET with a ridge classifier and a multi-layer perceptron (MLP) on wrist motions of PD patients. We find that all approaches are suitable for estimating tremor severity and bradykinesia presence but struggle with detecti
    

