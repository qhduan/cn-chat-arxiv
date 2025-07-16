# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PPA-Game: Characterizing and Learning Competitive Dynamics Among Online Content Creators](https://arxiv.org/abs/2403.15524) | 引入了PPA-Game模型来表征类似YouTube和TikTok平台上的内容创作者之间竞争动态，分析显示纯纳什均衡在大多数情况下是常见的，提出了一种在线算法用于最大化每个代理者的累积收益。 |
| [^2] | [Overcoming the Paradox of Certified Training with Gaussian Smoothing](https://arxiv.org/abs/2403.07095) | 通过使用高斯损失平滑方法，本研究提出了一种结合PGPE算法和不同凸放宽的认证训练方法，可以在训练神经网络时缓解紧凑凸松弛带来的问题，并获得更好性能的网络。 |
| [^3] | [FairTargetSim: An Interactive Simulator for Understanding and Explaining the Fairness Effects of Target Variable Definition](https://arxiv.org/abs/2403.06031) | FairTargetSim提供了一个交互式模拟器，展示了目标变量定义对公平性的影响，适用于算法开发者、研究人员和非技术利益相关者。 |
| [^4] | [TorchCP: A Library for Conformal Prediction based on PyTorch](https://arxiv.org/abs/2402.12683) | TorchCP是一个基于PyTorch的Python工具包，为深度学习模型上的合拟常规预测研究提供了实现后验和训练方法的多种工具，包括分类和回归任务。En_Tdlr: TorchCP is a Python toolbox built on PyTorch for conformal prediction research on deep learning models, providing various implementations for posthoc and training methods for classification and regression tasks, including multi-dimension output. |
| [^5] | [X Hacking: The Threat of Misguided AutoML](https://arxiv.org/abs/2401.08513) | 本文介绍了X黑客的概念，即利用自动化机器学习流程来操纵可解释AI（XAI）指标，从而产生所需解释的模型，而不降低其预测性能。研究者总结了X黑客现象的严重性，并提出了可能的检测和预防方法，同时探讨了对XAI研究可信度和可重现性的伦理影响。 |
| [^6] | [Beyond PCA: A Probabilistic Gram-Schmidt Approach to Feature Extraction](https://arxiv.org/abs/2311.09386) | 本研究提出了一种概率性Gram-Schmidt方法来进行特征提取，该方法可以检测和去除非线性依赖性，从而提取数据中的线性特征并去除非线性冗余。 |
| [^7] | [VFedMH: Vertical Federated Learning for Training Multi-party Heterogeneous Models.](http://arxiv.org/abs/2310.13367) | VFedMH是一种垂直联合学习方法，通过在前向传播过程中聚合参与者的嵌入来处理参与者之间的异构模型，解决了现有VFL方法面临的挑战。 |
| [^8] | [GRAPES: Learning to Sample Graphs for Scalable Graph Neural Networks.](http://arxiv.org/abs/2310.03399) | GRAPES是一种自适应图采样方法，通过学习识别在训练图神经网络分类器时具有影响力的节点集合，解决了可扩展图神经网络中的内存问题，并且在准确性和可扩展性方面表现出色。 |
| [^9] | [Data diversity and virtual imaging in AI-based diagnosis: A case study based on COVID-19.](http://arxiv.org/abs/2308.09730) | 本研究通过使用多样性的临床和虚拟生成的医学图像开发和评估了COVID-19诊断的AI模型，发现数据集特征对于AI性能具有重要影响，容易导致泛化能力较差，最高下降20％。 |

# 详细

[^1]: PPA-Game：表征和学习在线内容创作者之间的竞争动态

    PPA-Game: Characterizing and Learning Competitive Dynamics Among Online Content Creators

    [https://arxiv.org/abs/2403.15524](https://arxiv.org/abs/2403.15524)

    引入了PPA-Game模型来表征类似YouTube和TikTok平台上的内容创作者之间竞争动态，分析显示纯纳什均衡在大多数情况下是常见的，提出了一种在线算法用于最大化每个代理者的累积收益。

    

    我们引入了比例性收益分配游戏（PPA-Game）来模拟代理者如何竞争可分配资源和消费者的注意力，类似于YouTube和TikTok等平台上的内容创作者。根据异质权重为代理者分配收益，反映了创作者之间内容质量的多样性。我们的分析表明，纯纳什均衡（PNE）并不在每种情况下都有保证，但在我们的模拟中，通常会观察到，其缺乏情况是罕见的。除了分析静态收益外，我们进一步讨论了代理者关于资源收益的在线学习，将多玩家多臂老虎机框架整合在一起。我们提出了一种在线算法，在$T$轮中促进每个代理者累积收益的最大化。从理论上讲，我们建立了任何代理者的遗憾在任何$\eta > 0$下都受到$O(\log^{1 + \eta} T)$的限制。经验结果进一步验证了我们的算法的有效性。

    arXiv:2403.15524v1 Announce Type: cross  Abstract: We introduce the Proportional Payoff Allocation Game (PPA-Game) to model how agents, akin to content creators on platforms like YouTube and TikTok, compete for divisible resources and consumers' attention. Payoffs are allocated to agents based on heterogeneous weights, reflecting the diversity in content quality among creators. Our analysis reveals that although a pure Nash equilibrium (PNE) is not guaranteed in every scenario, it is commonly observed, with its absence being rare in our simulations. Beyond analyzing static payoffs, we further discuss the agents' online learning about resource payoffs by integrating a multi-player multi-armed bandit framework. We propose an online algorithm facilitating each agent's maximization of cumulative payoffs over $T$ rounds. Theoretically, we establish that the regret of any agent is bounded by $O(\log^{1 + \eta} T)$ for any $\eta > 0$. Empirical results further validate the effectiveness of ou
    
[^2]: 用高斯平滑克服认证培训的悖论

    Overcoming the Paradox of Certified Training with Gaussian Smoothing

    [https://arxiv.org/abs/2403.07095](https://arxiv.org/abs/2403.07095)

    通过使用高斯损失平滑方法，本研究提出了一种结合PGPE算法和不同凸放宽的认证训练方法，可以在训练神经网络时缓解紧凑凸松弛带来的问题，并获得更好性能的网络。

    

    尽管付出了大量努力，但训练神经网络以高认证准确度对抗对抗性示例仍然是一个悬而未决的问题。在训练中，尽管认证方法可以有效地利用紧凑的凸松弛进行界计算，但这些方法表现不如较松的松弛。先前的工作假设这是由这些更紧的松弛导致的损失表面的不连续性和扰动敏感性。在这项研究中，我们理论上展示了高斯损失平滑可以缓解这两个问题。我们通过提出一种结合PGPE的认证训练方法，该算法计算平滑损失的梯度，并使用不同的凸放宽来确认这一点。在使用这种训练方法时，我们观察到更紧密的界限确实导致更好的网络，可以在相同网络上胜过同类技术。尽管扩展基于PGPE的训练仍然具有挑战性。

    arXiv:2403.07095v1 Announce Type: new  Abstract: Training neural networks with high certified accuracy against adversarial examples remains an open problem despite significant efforts. While certification methods can effectively leverage tight convex relaxations for bound computation, in training, these methods perform worse than looser relaxations. Prior work hypothesized that this is caused by the discontinuity and perturbation sensitivity of the loss surface induced by these tighter relaxations. In this work, we show theoretically that Gaussian Loss Smoothing can alleviate both of these issues. We confirm this empirically by proposing a certified training method combining PGPE, an algorithm computing gradients of a smoothed loss, with different convex relaxations. When using this training method, we observe that tighter bounds indeed lead to strictly better networks that can outperform state-of-the-art methods on the same network. While scaling PGPE-based training remains challengin
    
[^3]: FairTargetSim：用于理解和解释目标变量定义公平性影响的交互式模拟器

    FairTargetSim: An Interactive Simulator for Understanding and Explaining the Fairness Effects of Target Variable Definition

    [https://arxiv.org/abs/2403.06031](https://arxiv.org/abs/2403.06031)

    FairTargetSim提供了一个交互式模拟器，展示了目标变量定义对公平性的影响，适用于算法开发者、研究人员和非技术利益相关者。

    

    机器学习需要为预测或决策定义目标变量，这个过程可能对公平性产生深远影响：偏见通常已经被编码在目标变量定义本身中，而不是在任何数据收集或训练之前。我们提出了一个交互式模拟器，FairTargetSim (FTS)，展示了目标变量定义如何影响公平性。FTS是一个有价值的工具，适用于算法开发者、研究人员和非技术利益相关者。FTS使用了算法招聘的案例研究，使用真实世界数据和用户定义的目标变量。FTS是开源的，可在以下网址找到：http://tinyurl.com/ftsinterface。本文附带的视频网址为：http://tinyurl.com/ijcaifts。

    arXiv:2403.06031v1 Announce Type: cross  Abstract: Machine learning requires defining one's target variable for predictions or decisions, a process that can have profound implications on fairness: biases are often encoded in target variable definition itself, before any data collection or training. We present an interactive simulator, FairTargetSim (FTS), that illustrates how target variable definition impacts fairness. FTS is a valuable tool for algorithm developers, researchers, and non-technical stakeholders. FTS uses a case study of algorithmic hiring, using real-world data and user-defined target variables. FTS is open-source and available at: http://tinyurl.com/ftsinterface. The video accompanying this paper is here: http://tinyurl.com/ijcaifts.
    
[^4]: TorchCP：基于PyTorch的一种适用于合拟常规预测的库

    TorchCP: A Library for Conformal Prediction based on PyTorch

    [https://arxiv.org/abs/2402.12683](https://arxiv.org/abs/2402.12683)

    TorchCP是一个基于PyTorch的Python工具包，为深度学习模型上的合拟常规预测研究提供了实现后验和训练方法的多种工具，包括分类和回归任务。En_Tdlr: TorchCP is a Python toolbox built on PyTorch for conformal prediction research on deep learning models, providing various implementations for posthoc and training methods for classification and regression tasks, including multi-dimension output.

    

    TorchCP是一个用于深度学习模型上的合拟常规预测研究的Python工具包。它包含了用于后验和训练方法的各种实现，用于分类和回归任务（包括多维输出）。TorchCP建立在PyTorch之上，并利用矩阵计算的优势，提供简洁高效的推理实现。该代码采用LGPL许可证，并在$\href{https://github.com/ml-stat-Sustech/TorchCP}{\text{this https URL}}$开源。

    arXiv:2402.12683v1 Announce Type: new  Abstract: TorchCP is a Python toolbox for conformal prediction research on deep learning models. It contains various implementations for posthoc and training methods for classification and regression tasks (including multi-dimension output). TorchCP is built on PyTorch (Paszke et al., 2019) and leverages the advantages of matrix computation to provide concise and efficient inference implementations. The code is licensed under the LGPL license and is open-sourced at $\href{https://github.com/ml-stat-Sustech/TorchCP}{\text{this https URL}}$.
    
[^5]: X黑客：误导的自动机器学习的威胁

    X Hacking: The Threat of Misguided AutoML

    [https://arxiv.org/abs/2401.08513](https://arxiv.org/abs/2401.08513)

    本文介绍了X黑客的概念，即利用自动化机器学习流程来操纵可解释AI（XAI）指标，从而产生所需解释的模型，而不降低其预测性能。研究者总结了X黑客现象的严重性，并提出了可能的检测和预防方法，同时探讨了对XAI研究可信度和可重现性的伦理影响。

    

    可解释的人工智能（XAI）和可解释的机器学习方法有助于建立对模型预测和派生见解的信任，但也为分析师提供了一种扭曲的动机，即操纵XAI指标以支持预先规定的结论。本文介绍了X黑客的概念，即将p-hacking应用于诸如Shap值之类的XAI指标。我们展示了如何利用自动化的机器学习流程来寻找“可辩护”的模型，这些模型可以产生所需的解释并在维持优越的预测性能时。我们将解释和准确性之间的权衡表述为一个多目标优化问题，并通过熟悉的真实世界数据集在经验上展示了X黑客的可行性和严重性。最后，我们提出了可能的检测和预防方法，并讨论了对XAI研究的可信度和可重现性的伦理影响。

    Explainable AI (XAI) and interpretable machine learning methods help to build trust in model predictions and derived insights, yet also present a perverse incentive for analysts to manipulate XAI metrics to support pre-specified conclusions. This paper introduces the concept of X-hacking, a form of p-hacking applied to XAI metrics such as Shap values. We show how an automated machine learning pipeline can be used to search for 'defensible' models that produce a desired explanation while maintaining superior predictive performance to a common baseline. We formulate the trade-off between explanation and accuracy as a multi-objective optimization problem and illustrate the feasibility and severity of X-hacking empirically on familiar real-world datasets. Finally, we suggest possible methods for detection and prevention, and discuss ethical implications for the credibility and reproducibility of XAI research.
    
[^6]: 超越PCA：一种概率性Gram-Schmidt方法的特征提取

    Beyond PCA: A Probabilistic Gram-Schmidt Approach to Feature Extraction

    [https://arxiv.org/abs/2311.09386](https://arxiv.org/abs/2311.09386)

    本研究提出了一种概率性Gram-Schmidt方法来进行特征提取，该方法可以检测和去除非线性依赖性，从而提取数据中的线性特征并去除非线性冗余。

    

    在无监督学习中，线性特征提取在数据中存在非线性依赖的情况下是一个基本挑战。我们提出使用概率性Gram-Schmidt (GS)类型的正交化过程来检测和映射出冗余维度。具体而言，通过在一族函数上应用GS过程，该族函数预计捕捉到数据中的非线性依赖性，我们构建了一系列协方差矩阵，可以用于识别新的大方差方向，或者将这些依赖性从主成分中去除。在前一种情况下，我们提供了熵减少的信息理论保证。在后一种情况下，我们证明在某些假设下，所得算法在所选择函数族的线性张成空间中可以检测和去除非线性依赖性。两种提出的方法都可以从数据中提取线性特征并去除非线性冗余。

    Linear feature extraction at the presence of nonlinear dependencies among the data is a fundamental challenge in unsupervised learning. We propose using a probabilistic Gram-Schmidt (GS) type orthogonalization process in order to detect and map out redundant dimensions. Specifically, by applying the GS process over a family of functions which presumably captures the nonlinear dependencies in the data, we construct a series of covariance matrices that can either be used to identify new large-variance directions, or to remove those dependencies from the principal components. In the former case, we provide information-theoretic guarantees in terms of entropy reduction. In the latter, we prove that under certain assumptions the resulting algorithms detect and remove nonlinear dependencies whenever those dependencies lie in the linear span of the chosen function family. Both proposed methods extract linear features from the data while removing nonlinear redundancies. We provide simulation r
    
[^7]: VFedMH: 垂直联合学习用于训练多参与方异构模型

    VFedMH: Vertical Federated Learning for Training Multi-party Heterogeneous Models. (arXiv:2310.13367v1 [cs.LG])

    [http://arxiv.org/abs/2310.13367](http://arxiv.org/abs/2310.13367)

    VFedMH是一种垂直联合学习方法，通过在前向传播过程中聚合参与者的嵌入来处理参与者之间的异构模型，解决了现有VFL方法面临的挑战。

    

    垂直联合学习（VFL）作为一种集成样本对齐和特征合并的新型训练范式，已经引起了越来越多的关注。然而，现有的VFL方法在处理参与者之间存在异构本地模型时面临挑战，这影响了优化收敛性和泛化能力。为了解决这个问题，本文提出了一种名为VFedMH的新方法，用于训练多方异构模型。VFedMH的重点是在前向传播期间聚合每个参与者知识的嵌入，而不是中间结果。主动方，拥有样本的标签和特征，在VFedMH中安全地聚合本地嵌入以获得全局知识嵌入，并将其发送给被动方。被动方仅拥有样本的特征，然后利用全局嵌入在其本地异构网络上进行前向传播。然而，被动方不拥有标签。

    Vertical Federated Learning (VFL) has gained increasing attention as a novel training paradigm that integrates sample alignment and feature union. However, existing VFL methods face challenges when dealing with heterogeneous local models among participants, which affects optimization convergence and generalization. To address this issue, this paper proposes a novel approach called Vertical Federated learning for training Multi-parties Heterogeneous models (VFedMH). VFedMH focuses on aggregating the embeddings of each participant's knowledge instead of intermediate results during forward propagation. The active party, who possesses labels and features of the sample, in VFedMH securely aggregates local embeddings to obtain global knowledge embeddings, and sends them to passive parties. The passive parties, who own only features of the sample, then utilize the global embeddings to propagate forward on their local heterogeneous networks. However, the passive party does not own the labels, 
    
[^8]: GRAPES: 学习用于可扩展图神经网络的图采样

    GRAPES: Learning to Sample Graphs for Scalable Graph Neural Networks. (arXiv:2310.03399v1 [cs.LG])

    [http://arxiv.org/abs/2310.03399](http://arxiv.org/abs/2310.03399)

    GRAPES是一种自适应图采样方法，通过学习识别在训练图神经网络分类器时具有影响力的节点集合，解决了可扩展图神经网络中的内存问题，并且在准确性和可扩展性方面表现出色。

    

    图神经网络（GNNs）通过以不同方式聚合周围信息来学习图中节点的表示。随着这些网络的加深，由于邻域尺寸的增加，它们的感受野呈指数增长，导致高内存消耗。图采样通过对图中节点进行抽样来解决GNNs中的内存问题。通过这种方式，GNNs可以扩展到更大的图。大多数采样方法专注于固定的采样启发式算法，这可能无法推广到不同的结构或任务。我们引入了GRAPES，一种自适应的图采样方法，该方法学习识别用于训练GNN分类器的一组具有影响力的节点。GRAPES使用GFlowNet来学习给定分类目标的节点采样概率。我们在几个小规模和大规模图基准上评估了GRAPES，并展示了其在准确性和可扩展性方面的有效性。与现有的采样方法相比，GRAPES即使在采样比例较低的情况下仍保持高准确性。

    Graph neural networks (GNNs) learn the representation of nodes in a graph by aggregating the neighborhood information in various ways. As these networks grow in depth, their receptive field grows exponentially due to the increase in neighborhood sizes, resulting in high memory costs. Graph sampling solves memory issues in GNNs by sampling a small ratio of the nodes in the graph. This way, GNNs can scale to much larger graphs. Most sampling methods focus on fixed sampling heuristics, which may not generalize to different structures or tasks. We introduce GRAPES, an adaptive graph sampling method that learns to identify sets of influential nodes for training a GNN classifier. GRAPES uses a GFlowNet to learn node sampling probabilities given the classification objectives. We evaluate GRAPES across several small- and large-scale graph benchmarks and demonstrate its effectiveness in accuracy and scalability. In contrast to existing sampling methods, GRAPES maintains high accuracy even with 
    
[^9]: 基于COVID-19的数据多样性和虚拟成像的AI诊断：以病例研究为基础

    Data diversity and virtual imaging in AI-based diagnosis: A case study based on COVID-19. (arXiv:2308.09730v1 [eess.IV])

    [http://arxiv.org/abs/2308.09730](http://arxiv.org/abs/2308.09730)

    本研究通过使用多样性的临床和虚拟生成的医学图像开发和评估了COVID-19诊断的AI模型，发现数据集特征对于AI性能具有重要影响，容易导致泛化能力较差，最高下降20％。

    

    许多研究已经调查了基于深度学习的人工智能（AI）模型在新型冠状病毒（COVID-19）的医学影像诊断中的应用，许多报道称其性能几乎完美。然而，性能的变异性和潜在的数据偏差引发了对临床适用性的担忧。本回顾性研究涉及使用临床多样性和虚拟生成的医学图像开发和评估COVID-19诊断的人工智能（AI）模型。此外，我们进行了一次虚拟成像试验，以评估AI性能受疾病范围、辐射剂量和计算机断层扫描（CT）和胸部放射摄影（CXR）成像模态等几个患者和物理性因素的影响。数据集特征（包括数量、多样性和患病率）强烈影响了AI的性能，导致接收者操作特征曲线下面积下降了高达20％，且泛化能力差。

    Many studies have investigated deep-learning-based artificial intelligence (AI) models for medical imaging diagnosis of the novel coronavirus (COVID-19), with many reports of near-perfect performance. However, variability in performance and underlying data biases raise concerns about clinical generalizability. This retrospective study involved the development and evaluation of artificial intelligence (AI) models for COVID-19 diagnosis using both diverse clinical and virtually generated medical images. In addition, we conducted a virtual imaging trial to assess how AI performance is affected by several patient- and physics-based factors, including the extent of disease, radiation dose, and imaging modality of computed tomography (CT) and chest radiography (CXR). AI performance was strongly influenced by dataset characteristics including quantity, diversity, and prevalence, leading to poor generalization with up to 20% drop in receiver operating characteristic area under the curve. Model
    

