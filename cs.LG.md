# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HyperFusion: A Hypernetwork Approach to Multimodal Integration of Tabular and Medical Imaging Data for Predictive Modeling](https://arxiv.org/abs/2403.13319) | 提出一种基于超网络的新框架，通过将图像处理条件设置在EHR的值和测量上，以整合临床成像和表格数据，旨在利用这些数据中的互补信息。 |
| [^2] | [Federated Transfer Learning with Differential Privacy](https://arxiv.org/abs/2403.11343) | 本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。 |
| [^3] | [Tractable Local Equilibria in Non-Concave Games](https://arxiv.org/abs/2403.08171) | 提出了一个新的解决概念，$(\varepsilon, \Phi(\delta))$-局部均衡，以解决在非凹游戏中局部均衡存在但难以处理的问题。 |
| [^4] | [CAS: A General Algorithm for Online Selective Conformal Prediction with FCR Control](https://arxiv.org/abs/2403.07728) | CAS框架允许在在线选择性预测中控制FCR，通过自适应选择和校准集构造输出符合预测区间 |
| [^5] | [Trading off Consistency and Dimensionality of Convex Surrogates for the Mode](https://arxiv.org/abs/2402.10818) | 通过在低维替代空间中的凸多面体顶点上嵌入结果，并探究单纯形中的一致性区域，权衡了替代损失维度、问题实例数量。 |
| [^6] | [Symmetry-Breaking Augmentations for Ad Hoc Teamwork](https://arxiv.org/abs/2402.09984) | 本研究提出了一种称为对称破缺增强的方法，通过增加训练队友的行为多样性来提高人工智能代理与新队友合作的性能。实验证明了该方法的有效性。 |
| [^7] | [Centaur: Federated Learning for Constrained Edge Devices](https://arxiv.org/abs/2211.04175) | Centaur提出了面向受限边缘设备的联邦学习框架，通过数据选择方案和基于分区的训练算法，实现了超限制设备在大型神经网络的高效参与，相比本地训练能获得更高准确性和节约能量。 |
| [^8] | [Vulnerabilities of Foundation Model Integrated Federated Learning Under Adversarial Threats.](http://arxiv.org/abs/2401.10375) | 本文研究基于基础模型集成的联邦学习在敌对威胁下的漏洞，提出了一种新的攻击策略，揭示了该模型在不同配置的联邦学习下对敌对威胁的高敏感性。 |
| [^9] | [Exploring the Frontiers of LLMs in Psychological Applications: A Comprehensive Review.](http://arxiv.org/abs/2401.01519) | 本文综述了大型语言模型（LLMs）在心理学应用中的前沿，包括如何模拟人类认知和行为、提供创新工具进行文献回顾、假设生成、实验设计等。 |
| [^10] | [Explainable Attention for Few-shot Learning and Beyond.](http://arxiv.org/abs/2310.07800) | 本研究介绍了一种用于少样本学习的解释性注意力框架，旨在通过识别输入数据的关键部分来提高模型性能。 |
| [^11] | [Three iterations of $(1-d)$-WL test distinguish non isometric clouds of $d$-dimensional points.](http://arxiv.org/abs/2303.12853) | 本文研究了WL测试在点云中的应用，结果发现三次迭代的$(d-1)$-WL测试可以区分$d$维欧几里得空间中的点云，且只需要一次迭代的$d$-WL测试就可以达到完整性。 |
| [^12] | [Enhanced Adaptive Gradient Algorithms for Nonconvex-PL Minimax Optimization.](http://arxiv.org/abs/2303.03984) | 本文提出了一类增强的基于动量的梯度下降上升方法（即MSGDA和AdaMSGDA）来解决非凸-PL极小极大问题，其中AdaMSGDA算法可以使用各种自适应学习率来更新变量$x$和$y$，而不依赖于任何全局和坐标自适应学习率。理论上，我们证明了我们的MSGDA和AdaMSGDA方法在找到$\epsilon$-稳定解时，只需要在每个循环中进行一次采样，就可以获得已知的最佳样本（梯度）复杂度$O(\epsilon^{-3})$。 |
| [^13] | [Preconditioned Gradient Descent for Overparameterized Nonconvex Burer--Monteiro Factorization with Global Optimality Certification.](http://arxiv.org/abs/2206.03345) | 本文提出了一种预条件梯度下降方法，使得超参数化情况下梯度下降的收敛速度恢复到线性，并在保证全局最优性证明有效的同时保持低廉的计算代价，该方法适用于强凸的代价函数 $\phi$。 |
| [^14] | [Composite Goodness-of-fit Tests with Kernels.](http://arxiv.org/abs/2111.10275) | 本文提出了一种基于核的假设检验方法，可以解决具有挑战性的复合检验问题，其核心思想是在正确的模型规范的零假设下，非参数地估计参数（或模拟器）分布。 |

# 详细

[^1]: HyperFusion：一种用于预测建模的多模态整合表格和医学成像数据的超网络方法

    HyperFusion: A Hypernetwork Approach to Multimodal Integration of Tabular and Medical Imaging Data for Predictive Modeling

    [https://arxiv.org/abs/2403.13319](https://arxiv.org/abs/2403.13319)

    提出一种基于超网络的新框架，通过将图像处理条件设置在EHR的值和测量上，以整合临床成像和表格数据，旨在利用这些数据中的互补信息。

    

    ARXIV: 2403.13319v1 公告类型: 交叉摘要: 整合各种临床模式，如医学成像和患者电子健康记录（EHR）获得的表格数据，是现代医疗保健的关键方面。多源数据的综合分析可以全面了解患者的状况，并可以增强诊断和治疗决策。深度神经网络（DNN）在医学领域的多模态任务中一直展示出出色的性能。然而，有效地将医学成像与以数字表格数据表示的临床、人口统计和遗传信息进行融合的复杂努力仍然是一个高度活跃的持续研究追求。

    arXiv:2403.13319v1 Announce Type: cross  Abstract: The integration of diverse clinical modalities such as medical imaging and the tabular data obtained by the patients' Electronic Health Records (EHRs) is a crucial aspect of modern healthcare. The integrative analysis of multiple sources can provide a comprehensive understanding of a patient's condition and can enhance diagnoses and treatment decisions. Deep Neural Networks (DNNs) consistently showcase outstanding performance in a wide range of multimodal tasks in the medical domain. However, the complex endeavor of effectively merging medical imaging with clinical, demographic and genetic information represented as numerical tabular data remains a highly active and ongoing research pursuit.   We present a novel framework based on hypernetworks to fuse clinical imaging and tabular data by conditioning the image processing on the EHR's values and measurements. This approach aims to leverage the complementary information present in these
    
[^2]: 具有差分隐私的联邦迁移学习

    Federated Transfer Learning with Differential Privacy

    [https://arxiv.org/abs/2403.11343](https://arxiv.org/abs/2403.11343)

    本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。

    

    联邦学习越来越受到欢迎，数据异构性和隐私性是两个突出的挑战。在本文中，我们在联邦迁移学习框架内解决了这两个问题，旨在通过利用来自多个异构源数据集的信息来增强对目标数据集的学习，同时遵守隐私约束。我们严格制定了\textit{联邦差分隐私}的概念，为每个数据集提供隐私保证，而无需假设有一个受信任的中央服务器。在这个隐私约束下，我们研究了三个经典的统计问题，即单变量均值估计、低维线性回归和高维线性回归。通过研究极小值率并确定这些问题的隐私成本，我们展示了联邦差分隐私是已建立的局部和中央模型之间的一种中间隐私模型。

    arXiv:2403.11343v1 Announce Type: new  Abstract: Federated learning is gaining increasing popularity, with data heterogeneity and privacy being two prominent challenges. In this paper, we address both issues within a federated transfer learning framework, aiming to enhance learning on a target data set by leveraging information from multiple heterogeneous source data sets while adhering to privacy constraints. We rigorously formulate the notion of \textit{federated differential privacy}, which offers privacy guarantees for each data set without assuming a trusted central server. Under this privacy constraint, we study three classical statistical problems, namely univariate mean estimation, low-dimensional linear regression, and high-dimensional linear regression. By investigating the minimax rates and identifying the costs of privacy for these problems, we show that federated differential privacy is an intermediate privacy model between the well-established local and central models of 
    
[^3]: 在非凹游戏中可处理的局部均衡

    Tractable Local Equilibria in Non-Concave Games

    [https://arxiv.org/abs/2403.08171](https://arxiv.org/abs/2403.08171)

    提出了一个新的解决概念，$(\varepsilon, \Phi(\delta))$-局部均衡，以解决在非凹游戏中局部均衡存在但难以处理的问题。

    

    虽然众所周知在线梯度下降和其他无悔学习程序可以有效地收敛到协调均衡，在每个Agent的效用对于其自身策略呈凹形的情况下，但当效用是非凹的时，这种情况在机器学习应用中很常见，其中Agent的策略由深度神经网络参数化，或者Agent的效用由神经网络计算，或两者兼而有之。实际上，非凹游戏存在一系列博弈论和优化挑战：(i) Nash均衡可能不存在；(ii) 局部Nash均衡存在但是不可处理；(iii) 混合Nash、协调和粗糙协调均衡在一般情况下具有无限支持，并且是不可处理的。为了避开这些挑战，我们提出了一个新的解决概念，称为$(\varepsilon, \Phi(\delta))$-局部均衡，该概念在非凹游戏中概括了局部Nash均衡。

    arXiv:2403.08171v1 Announce Type: cross  Abstract: While Online Gradient Descent and other no-regret learning procedures are known to efficiently converge to coarse correlated equilibrium in games where each agent's utility is concave in their own strategy, this is not the case when the utilities are non-concave, a situation that is common in machine learning applications where the agents' strategies are parameterized by deep neural networks, or the agents' utilities are computed by a neural network, or both. Indeed, non-concave games present a host of game-theoretic and optimization challenges: (i) Nash equilibria may fail to exist; (ii) local Nash equilibria exist but are intractable; and (iii) mixed Nash, correlated, and coarse correlated equilibria have infinite support in general, and are intractable. To sidestep these challenges we propose a new solution concept, termed $(\varepsilon, \Phi(\delta))$-local equilibrium, which generalizes local Nash equilibrium in non-concave games,
    
[^4]: CAS: 一种具有FCR控制的在线选择性符合预测的通用算法

    CAS: A General Algorithm for Online Selective Conformal Prediction with FCR Control

    [https://arxiv.org/abs/2403.07728](https://arxiv.org/abs/2403.07728)

    CAS框架允许在在线选择性预测中控制FCR，通过自适应选择和校准集构造输出符合预测区间

    

    我们研究了在线方式下后选择预测推断的问题。为了避免将资源耗费在不重要的单位上，在报告其预测区间之前对当前个体进行初步选择在在线预测任务中是常见且有意义的。由于在线选择导致所选预测区间中存在时间多重性，因此控制实时误覆盖陈述率（FCR）来测量平均误覆盖误差是重要的。我们开发了一个名为CAS（适应性选择后校准）的通用框架，可以包裹任何预测模型和在线选择规则，以输出后选择的预测区间。如果选择了当前个体，我们首先对历史数据进行自适应选择来构建校准集，然后为未观察到的标签输出符合预测区间。我们为校准集提供了可行的构造方式

    arXiv:2403.07728v1 Announce Type: cross  Abstract: We study the problem of post-selection predictive inference in an online fashion. To avoid devoting resources to unimportant units, a preliminary selection of the current individual before reporting its prediction interval is common and meaningful in online predictive tasks. Since the online selection causes a temporal multiplicity in the selected prediction intervals, it is important to control the real-time false coverage-statement rate (FCR) to measure the averaged miscoverage error. We develop a general framework named CAS (Calibration after Adaptive Selection) that can wrap around any prediction model and online selection rule to output post-selection prediction intervals. If the current individual is selected, we first perform an adaptive selection on historical data to construct a calibration set, then output a conformal prediction interval for the unobserved label. We provide tractable constructions for the calibration set for 
    
[^5]: 在模型的凸替代品的一致性和维度之间进行权衡

    Trading off Consistency and Dimensionality of Convex Surrogates for the Mode

    [https://arxiv.org/abs/2402.10818](https://arxiv.org/abs/2402.10818)

    通过在低维替代空间中的凸多面体顶点上嵌入结果，并探究单纯形中的一致性区域，权衡了替代损失维度、问题实例数量。

    

    在多类分类中，必须将结果嵌入到至少有$n-1$维的实数空间中，以设计一种一致的替代损失函数，这会导致"正确"的分类，而不受数据分布的影响。在信息检索和结构化预测任务等需要大量n时，优化n-1维替代常常是棘手的。我们研究了在多类分类中如何权衡替代损失维度、问题实例数量以及在单纯形上约束一致性区域的方法。我们跟随过去的研究，探讨了一种直观的嵌入过程，将结果映射到低维替代空间中的凸多面体的顶点上。我们展示了在每个点质量分布周围存在单纯形的全维子集，其中一致性成立，但是，少于n-1维度的情况下，存在一些分布，对于这些分布，一种现象性是

    arXiv:2402.10818v1 Announce Type: new  Abstract: In multiclass classification over $n$ outcomes, the outcomes must be embedded into the reals with dimension at least $n-1$ in order to design a consistent surrogate loss that leads to the "correct" classification, regardless of the data distribution. For large $n$, such as in information retrieval and structured prediction tasks, optimizing a surrogate in $n-1$ dimensions is often intractable. We investigate ways to trade off surrogate loss dimension, the number of problem instances, and restricting the region of consistency in the simplex for multiclass classification. Following past work, we examine an intuitive embedding procedure that maps outcomes into the vertices of convex polytopes in a low-dimensional surrogate space. We show that full-dimensional subsets of the simplex exist around each point mass distribution for which consistency holds, but also, with less than $n-1$ dimensions, there exist distributions for which a phenomeno
    
[^6]: 对于临时团队合作的对称破缺增强

    Symmetry-Breaking Augmentations for Ad Hoc Teamwork

    [https://arxiv.org/abs/2402.09984](https://arxiv.org/abs/2402.09984)

    本研究提出了一种称为对称破缺增强的方法，通过增加训练队友的行为多样性来提高人工智能代理与新队友合作的性能。实验证明了该方法的有效性。

    

    在许多协作环境中，人工智能（AI）代理必须能够适应使用未知或先前未观察到的策略的新队友。对于AI代理来说，这通常对人类来说很简单，但却是一项具有挑战性的任务。例如，如果一个AI代理在训练集中学会了与只在一侧道路上行驶的其他车辆并行驶，那么即使这些车辆的行为只是在左右对称上进行了翻转，它也可能难以适应与相反方向上行驶的驾驶员进行协调。为了解决这个问题，我们引入了对称破缺增强（SBA），通过应用对称翻转操作来增加训练队友的行为多样性。通过学习对增强后的队友的最佳响应，我们的代理能够接触到更广泛的行为约定，从而提高与新队友合作时的性能。我们在两个设置中进行了实验验证，并证明了我们的方法的有效性。

    arXiv:2402.09984v1 Announce Type: cross  Abstract: In many collaborative settings, artificial intelligence (AI) agents must be able to adapt to new teammates that use unknown or previously unobserved strategies. While often simple for humans, this can be challenging for AI agents. For example, if an AI agent learns to drive alongside others (a training set) that only drive on one side of the road, it may struggle to adapt this experience to coordinate with drivers on the opposite side, even if their behaviours are simply flipped along the left-right symmetry. To address this we introduce symmetry-breaking augmentations (SBA), which increases diversity in the behaviour of training teammates by applying a symmetry-flipping operation. By learning a best-response to the augmented set of teammates, our agent is exposed to a wider range of behavioural conventions, improving performance when deployed with novel teammates. We demonstrate this experimentally in two settings, and show that our a
    
[^7]: Centaur: 面向受限边缘设备的联邦学习

    Centaur: Federated Learning for Constrained Edge Devices

    [https://arxiv.org/abs/2211.04175](https://arxiv.org/abs/2211.04175)

    Centaur提出了面向受限边缘设备的联邦学习框架，通过数据选择方案和基于分区的训练算法，实现了超限制设备在大型神经网络的高效参与，相比本地训练能获得更高准确性和节约能量。

    

    联邦学习（FL）促进了在边缘设备上的新应用，尤其是对于可穿戴和物联网设备。这些设备捕获大量多样化的数据，但它们受到内存、计算、功耗和连接性约束，这些约束阻碍了它们参与FL。我们提出Centaur，一个多层FL框架，使超限制的设备能够高效地参与大型神经网络的FL。Centaur结合了两个主要的想法：（i）数据选择方案选择一部分样本加速学习，以及（ii）一个基于分区的训练算法，整合同一用户拥有的受限和强大设备。在四个基准神经网络和三个数据集上的评估显示，Centaur相比于在受限设备上的本地训练，能够获得约10\%更高的准确性，平均能节约约58\%的能量。我们的实验结果也表明了Centaur在处理时的卓越效率。

    arXiv:2211.04175v3 Announce Type: replace  Abstract: Federated learning (FL) facilitates new applications at the edge, especially for wearable and Internet-of-Thing devices. Such devices capture a large and diverse amount of data, but they have memory, compute, power, and connectivity constraints which hinder their participation in FL. We propose Centaur, a multitier FL framework, enabling ultra-constrained devices to efficiently participate in FL on large neural nets. Centaur combines two major ideas: (i) a data selection scheme to choose a portion of samples that accelerates the learning, and (ii) a partition-based training algorithm that integrates both constrained and powerful devices owned by the same user. Evaluations, on four benchmark neural nets and three datasets, show that Centaur gains ~10\% higher accuracy than local training on constrained devices with ~58\% energy saving on average. Our experimental results also demonstrate the superior efficiency of Centaur when dealing
    
[^8]: 基于基础模型集成的联邦学习在敌对威胁下的漏洞

    Vulnerabilities of Foundation Model Integrated Federated Learning Under Adversarial Threats. (arXiv:2401.10375v1 [cs.CR])

    [http://arxiv.org/abs/2401.10375](http://arxiv.org/abs/2401.10375)

    本文研究基于基础模型集成的联邦学习在敌对威胁下的漏洞，提出了一种新的攻击策略，揭示了该模型在不同配置的联邦学习下对敌对威胁的高敏感性。

    

    联邦学习是解决与数据隐私和安全相关的机器学习的重要问题，但在某些情况下存在数据不足和不平衡问题。基础模型的出现为现有联邦学习框架的局限性提供了潜在的解决方案，例如通过生成合成数据进行模型初始化。然而，由于基础模型的内在安全性问题，将基础模型集成到联邦学习中可能引入新的风险，这方面的研究尚属未开发。为了填补这一空白，我们首次研究基于基础模型集成的联邦学习在敌对威胁下的漏洞。基于基础模型集成的联邦学习的统一框架，我们引入了一种新的攻击策略，利用基础模型的安全性问题来破坏联邦学习客户端模型。通过在图像和文本领域中使用知名模型和基准数据集进行广泛实验，我们揭示了基于基础模型集成的联邦学习在不同配置的联邦学习下对这种新威胁的高敏感性。

    Federated Learning (FL) addresses critical issues in machine learning related to data privacy and security, yet suffering from data insufficiency and imbalance under certain circumstances. The emergence of foundation models (FMs) offers potential solutions to the limitations of existing FL frameworks, e.g., by generating synthetic data for model initialization. However, due to the inherent safety concerns of FMs, integrating FMs into FL could introduce new risks, which remains largely unexplored. To address this gap, we conduct the first investigation on the vulnerability of FM integrated FL (FM-FL) under adversarial threats. Based on a unified framework of FM-FL, we introduce a novel attack strategy that exploits safety issues of FM to compromise FL client models. Through extensive experiments with well-known models and benchmark datasets in both image and text domains, we reveal the high susceptibility of the FM-FL to this new threat under various FL configurations. Furthermore, we f
    
[^9]: 探索LLMs在心理学应用中的前沿：一份综述

    Exploring the Frontiers of LLMs in Psychological Applications: A Comprehensive Review. (arXiv:2401.01519v1 [cs.LG])

    [http://arxiv.org/abs/2401.01519](http://arxiv.org/abs/2401.01519)

    本文综述了大型语言模型（LLMs）在心理学应用中的前沿，包括如何模拟人类认知和行为、提供创新工具进行文献回顾、假设生成、实验设计等。

    

    本文探索了大型语言模型（LLMs）在心理学应用中的前沿。心理学经历了几次理论变革，当前人工智能（AI）和机器学习，特别是LLMs的使用有望开启新的研究方向。我们详细探讨了LLMs如ChatGPT在心理学研究中的转变。文章讨论了LLMs在认知与行为心理学、临床与咨询心理学、教育与发展心理学以及社会与文化心理学等心理学分支中的影响，强调了它们模拟人类认知和行为方面的潜力。本文深入探讨了这些模型模拟人类文本生成的能力，为心理学中的文献回顾、假设生成、实验设计、实验对象、数据分析、学术写作和同行评审等提供创新工具。虽然LLMs在推动研究方法学方面起着重要作用，

    This paper explores the frontiers of large language models (LLMs) in psychology applications. Psychology has undergone several theoretical changes, and the current use of Artificial Intelligence (AI) and Machine Learning, particularly LLMs, promises to open up new research directions. We provide a detailed exploration of how LLMs like ChatGPT are transforming psychological research. It discusses the impact of LLMs across various branches of psychology, including cognitive and behavioral, clinical and counseling, educational and developmental, and social and cultural psychology, highlighting their potential to simulate aspects of human cognition and behavior. The paper delves into the capabilities of these models to emulate human-like text generation, offering innovative tools for literature review, hypothesis generation, experimental design, experimental subjects, data analysis, academic writing, and peer review in psychology. While LLMs are essential in advancing research methodologie
    
[^10]: 解释性注意力用于少样本学习及其它领域

    Explainable Attention for Few-shot Learning and Beyond. (arXiv:2310.07800v1 [cs.AI])

    [http://arxiv.org/abs/2310.07800](http://arxiv.org/abs/2310.07800)

    本研究介绍了一种用于少样本学习的解释性注意力框架，旨在通过识别输入数据的关键部分来提高模型性能。

    

    注意力机制在增强学习模型中显示出了有希望的潜力，它可以通过识别输入数据中显著的部分来提高模型的性能。这在数据收集和标记方面存在挑战，导致训练样本有限的情况下尤其有价值。受人类认知过程启发，我们认为，如果将AI基线暴露于原始数据的关键部分而不是整个输入数据集，类似于人类的感知，那么它的性能可能会更准确、更可靠。然而，选择这些信息性数据部分的任务，即硬注意力寻找，是一个巨大的挑战。在少量训练样本的情况下，现有的研究很难找到这些信息性区域，原因是大量的训练参数无法从有限的样本中有效学习。在本研究中，我们介绍了一种实用的框架，用于实现可解释的硬注意力寻找。

    Attention mechanisms have exhibited promising potential in enhancing learning models by identifying salient portions of input data. This is particularly valuable in scenarios where limited training samples are accessible due to challenges in data collection and labeling. Drawing inspiration from human recognition processes, we posit that an AI baseline's performance could be more accurate and dependable if it is exposed to essential segments of raw data rather than the entire input dataset, akin to human perception. However, the task of selecting these informative data segments, referred to as hard attention finding, presents a formidable challenge. In situations with few training samples, existing studies struggle to locate such informative regions due to the large number of training parameters that cannot be effectively learned from the available limited samples. In this study, we introduce a novel and practical framework for achieving explainable hard attention finding, specifically
    
[^11]: 三次迭代的$(1-d)$-WL测试可以区分$d$维点云的非等距变换. (arXiv:2303.12853v1 [cs.LG])

    Three iterations of $(1-d)$-WL test distinguish non isometric clouds of $d$-dimensional points. (arXiv:2303.12853v1 [cs.LG])

    [http://arxiv.org/abs/2303.12853](http://arxiv.org/abs/2303.12853)

    本文研究了WL测试在点云中的应用，结果发现三次迭代的$(d-1)$-WL测试可以区分$d$维欧几里得空间中的点云，且只需要一次迭代的$d$-WL测试就可以达到完整性。

    

    Weisfeiler-Lehman (WL)测试是一个检查图同构的基本迭代算法。它被观察到是几种图神经网络体系结构设计的基础，这些网络的能力和性能可以用这个测试的表示能力来理解。受最近机器学习应用于涉及三维物体的数据集的发展启发，我们研究了当WL测试对完整的距离图表示的欧几里得点云是“完整的”时，它何时能够识别出任意一个任意点云.我们的主要结果是，$(d-1)$-维WL测试可以区分$d$维欧几里得空间中的点云，任何$d\ge 2$都可以，而且只需要进行三次测试。我们的结果对于$d=2,3$是紧的。我们还观察到$d$维WL测试只需要进行一次迭代就可以达到完整性。

    The Weisfeiler--Lehman (WL) test is a fundamental iterative algorithm for checking isomorphism of graphs. It has also been observed that it underlies the design of several graph neural network architectures, whose capabilities and performance can be understood in terms of the expressive power of this test. Motivated by recent developments in machine learning applications to datasets involving three-dimensional objects, we study when the WL test is {\em complete} for clouds of euclidean points represented by complete distance graphs, i.e., when it can distinguish, up to isometry, any arbitrary such cloud.  Our main result states that the $(d-1)$-dimensional WL test is complete for point clouds in $d$-dimensional Euclidean space, for any $d\ge 2$, and that only three iterations of the test suffice. Our result is tight for $d = 2, 3$. We also observe that the $d$-dimensional WL test only requires one iteration to achieve completeness.
    
[^12]: 非凸-PL极小极大优化的增强自适应梯度算法

    Enhanced Adaptive Gradient Algorithms for Nonconvex-PL Minimax Optimization. (arXiv:2303.03984v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2303.03984](http://arxiv.org/abs/2303.03984)

    本文提出了一类增强的基于动量的梯度下降上升方法（即MSGDA和AdaMSGDA）来解决非凸-PL极小极大问题，其中AdaMSGDA算法可以使用各种自适应学习率来更新变量$x$和$y$，而不依赖于任何全局和坐标自适应学习率。理论上，我们证明了我们的MSGDA和AdaMSGDA方法在找到$\epsilon$-稳定解时，只需要在每个循环中进行一次采样，就可以获得已知的最佳样本（梯度）复杂度$O(\epsilon^{-3})$。

    This paper proposes a class of enhanced momentum-based gradient descent ascent methods (MSGDA and AdaMSGDA) to solve nonconvex-PL minimax problems, where the AdaMSGDA algorithm can use various adaptive learning rates to update variables x and y without relying on any global and coordinate-wise adaptive learning rates. Theoretical analysis shows that MSGDA and AdaMSGDA methods have the best known sample (gradient) complexity of O(ε−3) in finding an ε-stationary solution.

    本文研究了一类非凸非凹的极小极大优化问题（即$\min_x\max_y f(x,y)$），其中$f(x,y)$在$x$上可能是非凸的，在$y$上是非凹的，并满足Polyak-Lojasiewicz（PL）条件。此外，我们提出了一类增强的基于动量的梯度下降上升方法（即MSGDA和AdaMSGDA）来解决这些随机非凸-PL极小极大问题。特别地，我们的AdaMSGDA算法可以使用各种自适应学习率来更新变量$x$和$y$，而不依赖于任何全局和坐标自适应学习率。理论上，我们提出了一种有效的收敛分析框架来解决我们的方法。具体而言，我们证明了我们的MSGDA和AdaMSGDA方法在找到$\epsilon$-稳定解（即$\mathbb{E}\|\nabla F(x)\|\leq \epsilon$，其中$F(x)=\max_y f(x,y)$）时，只需要在每个循环中进行一次采样，就可以获得已知的最佳样本（梯度）复杂度$O(\epsilon^{-3})$。

    In the paper, we study a class of nonconvex nonconcave minimax optimization problems (i.e., $\min_x\max_y f(x,y)$), where $f(x,y)$ is possible nonconvex in $x$, and it is nonconcave and satisfies the Polyak-Lojasiewicz (PL) condition in $y$. Moreover, we propose a class of enhanced momentum-based gradient descent ascent methods (i.e., MSGDA and AdaMSGDA) to solve these stochastic Nonconvex-PL minimax problems. In particular, our AdaMSGDA algorithm can use various adaptive learning rates in updating the variables $x$ and $y$ without relying on any global and coordinate-wise adaptive learning rates. Theoretically, we present an effective convergence analysis framework for our methods. Specifically, we prove that our MSGDA and AdaMSGDA methods have the best known sample (gradient) complexity of $O(\epsilon^{-3})$ only requiring one sample at each loop in finding an $\epsilon$-stationary solution (i.e., $\mathbb{E}\|\nabla F(x)\|\leq \epsilon$, where $F(x)=\max_y f(x,y)$). This manuscript 
    
[^13]: 针对超参数化的非凸Burer-Monteiro分解的预条件梯度下降与全局最优性证明

    Preconditioned Gradient Descent for Overparameterized Nonconvex Burer--Monteiro Factorization with Global Optimality Certification. (arXiv:2206.03345v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2206.03345](http://arxiv.org/abs/2206.03345)

    本文提出了一种预条件梯度下降方法，使得超参数化情况下梯度下降的收敛速度恢复到线性，并在保证全局最优性证明有效的同时保持低廉的计算代价，该方法适用于强凸的代价函数 $\phi$。

    

    本文探讨了使用梯度下降优化非凸函数$f(X)=\phi(XX^{T})$的方法，其中 $\phi$是一个平滑凸的$n\times n$矩阵上下文的代价函数。虽然仅有二阶停留点可以在合理时间内被证明找到，但如果 $X$ 的秩缺失，那么它的秩缺失将证明它是全局最优的。这种认证全局最优性的方法必然需要当前迭代$X$的搜索秩 $r$ 超过全局最小化器$X^{\star}$ 的秩$r^{\star}$。不幸的是，超参数化显著减慢了梯度下降的收敛速度，从 $r=r^{\star}$ 时的线性速度降为 $r>r^{\star}$ 时的亚线性速度，即使 $\phi$ 是强凸的情况下也是如此。在本文中，我们提出了一种廉价的预条件梯度下降方法，将超参数化情况下梯度下降的收敛速度恢复到线性，同时保证全局最优性证明依旧有效。这种方法只需要进行简单的矩阵乘法和求逆，并且适用于强凸的$φ$。我们通过仿真实验在合成数据和现实应用中验证了我们提出的方法。

    We consider using gradient descent to minimize the nonconvex function $f(X)=\phi(XX^{T})$ over an $n\times r$ factor matrix $X$, in which $\phi$ is an underlying smooth convex cost function defined over $n\times n$ matrices. While only a second-order stationary point $X$ can be provably found in reasonable time, if $X$ is additionally rank deficient, then its rank deficiency certifies it as being globally optimal. This way of certifying global optimality necessarily requires the search rank $r$ of the current iterate $X$ to be overparameterized with respect to the rank $r^{\star}$ of the global minimizer $X^{\star}$. Unfortunately, overparameterization significantly slows down the convergence of gradient descent, from a linear rate with $r=r^{\star}$ to a sublinear rate when $r>r^{\star}$, even when $\phi$ is strongly convex. In this paper, we propose an inexpensive preconditioner that restores the convergence rate of gradient descent back to linear in the overparameterized case, while
    
[^14]: 带有核的复合适合性检验方法

    Composite Goodness-of-fit Tests with Kernels. (arXiv:2111.10275v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2111.10275](http://arxiv.org/abs/2111.10275)

    本文提出了一种基于核的假设检验方法，可以解决具有挑战性的复合检验问题，其核心思想是在正确的模型规范的零假设下，非参数地估计参数（或模拟器）分布。

    

    模型错误说明可能会对概率模型的实现造成重大挑战，这促使开发出一些直接解决此问题的鲁棒方法。但是，这些更为复杂的方法是否需要取决于模型是否真的错误，目前缺乏通用的方法回答这个问题。在本文中，我们提出了一种方法。更具体地说，我们提出了基于核的假设检验方法，用于具有挑战性的复合检验问题，即我们是否感兴趣的数据来自某些参数模型族中的任何分布。我们的测试利用基于最大均值差异和核Stein差异的最小距离估计器。它们具有广泛的适用性，包括当参数模型的密度已知除标准化常数外，或者如果模型采用模拟器形式。作为我们的主要结果，我们展示了在正确的模型规范的零假设下，我们能够非参数地估计参数（或模拟器）分布。我们提供了建立我们方法有效性的理论，并通过模拟和异常检测应用案例演示了其性能。

    Model misspecification can create significant challenges for the implementation of probabilistic models, and this has led to development of a range of robust methods which directly account for this issue. However, whether these more involved methods are required will depend on whether the model is really misspecified, and there is a lack of generally applicable methods to answer this question. In this paper, we propose one such method. More precisely, we propose kernel-based hypothesis tests for the challenging composite testing problem, where we are interested in whether the data comes from any distribution in some parametric family. Our tests make use of minimum distance estimators based on the maximum mean discrepancy and the kernel Stein discrepancy. They are widely applicable, including whenever the density of the parametric model is known up to normalisation constant, or if the model takes the form of a simulator. As our main result, we show that we are able to estimate the param
    

