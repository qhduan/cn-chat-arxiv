# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Information-Theoretic Framework for Out-of-Distribution Generalization](https://arxiv.org/abs/2403.19895) | 提出了一个信息论框架用于机器学习中的超出分布泛化，可以自由插值并产生新的泛化界限，同时具有最优输运解释。 |
| [^2] | [Low-Dose CT Image Reconstruction by Fine-Tuning a UNet Pretrained for Gaussian Denoising for the Downstream Task of Image Enhancement](https://arxiv.org/abs/2403.03551) | 提出了一种通过精调UNet进行低剂量CT图像重建的方法，其中第二阶段的训练策略为CT图像增强阶段。 |
| [^3] | [Feudal Networks for Visual Navigation](https://arxiv.org/abs/2402.12498) | 使用封建学习的视觉导航，通过高级管理者、中级管理者和工作代理的分层结构，在不同空间和时间尺度上操作，具有独特模块来实现自监督学习记忆代理地图。 |
| [^4] | [Solid Waste Detection in Remote Sensing Images: A Survey](https://arxiv.org/abs/2402.09066) | 本文调查了固体废物在遥感图像中的检测方法。研究者利用地球观测卫星提供的高分辨率数据，通过遥感图像实现了固体废物处置场地的识别、监测和评估。 |
| [^5] | [Inverse Reinforcement Learning by Estimating Expertise of Demonstrators](https://arxiv.org/abs/2402.01886) | 本文介绍了一个新颖的框架，IRLEED，它通过估计演示者的专业知识来解决模仿学习中的次优和异质演示的问题。IRLEED通过结合演示者次优性的普适模型和最大熵IRL框架，有效地从多样的次优演示中得出最佳策略。 |
| [^6] | [Optimizing Heat Alert Issuance with Reinforcement Learning](https://arxiv.org/abs/2312.14196) | 本研究利用强化学习优化热预警系统，通过引入新颖强化学习环境和综合数据集，解决了气候和健康环境中的低信号效应和空间异质性。 |
| [^7] | [Sparse Portfolio Selection via Topological Data Analysis based Clustering.](http://arxiv.org/abs/2401.16920) | 本文使用拓扑数据分析工具提出了一种基于聚类的稀疏投资组合选择策略，通过利用股票价格波动的拓扑特征，在持续图和景观空间上引入新的距离度量，并与聚类算法相结合，显著提升了多样市场情景下稀疏投资组合的绩效。 |
| [^8] | [IGNITE: Individualized GeNeration of Imputations in Time-series Electronic health records.](http://arxiv.org/abs/2401.04402) | 个体化时间序列电子健康记录的生成模型IGNITE通过学习个体的动态特征，结合人口特征和治疗信息，生成个性化的真实值，为个体化医疗提供了有价值的方式。 |
| [^9] | [Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks.](http://arxiv.org/abs/2401.03350) | 提出了G-$\Delta$UQ，一种新的训练框架，旨在改善图神经网络（GNN）的内在不确定性估计。该框架通过图锚定策略将随机数据中心化应用于图数据，并且能够支持部分随机的GNN。 |
| [^10] | [Tackling Hybrid Heterogeneity on Federated Optimization via Gradient Diversity Maximization.](http://arxiv.org/abs/2310.02702) | 本文探讨了混合异构性如何影响联邦优化，并提出了一种通过最大化梯度多样性来减轻混合异构性负面影响的方法。 |
| [^11] | [Optimal Heterogeneous Collaborative Linear Regression and Contextual Bandits.](http://arxiv.org/abs/2306.06291) | 本文提出了一种新的估计器MOLAR，它利用协同线性回归和上下文臂问题中的稀疏异质性来提高估计精度，并且相比独立方法具有更好的表现。 |
| [^12] | [Catch-Up Distillation: You Only Need to Train Once for Accelerating Sampling.](http://arxiv.org/abs/2305.10769) | 本文提出了一种名为“追赶蒸馏”的方法，通过调整传统采样算法，让速度估计模型的当前时刻输出与其先前时刻输出和地面真实标签对齐，从而实现只需一次训练便能加速采样的效果。 |
| [^13] | [Towards the Characterization of Representations Learned via Capsule-based Network Architectures.](http://arxiv.org/abs/2305.05349) | 本研究旨在评估胶囊网络架构学习的表示方法及其可解释性，发现其编码的表示可能与部分-整体关系并不严格相关。 |
| [^14] | [Robust Dequantization of the Quantum Singular value Transformation and Quantum Machine Learning Algorithms.](http://arxiv.org/abs/2304.04932) | 本文研究了量子机器学习算法的鲁棒去量子化方法。我们提出了近似长度平方采样的概念，并展示了如何将随机线性代数技术适应到这种更弱的假设下。我们使用这些技术证明了最近的低秩去量子化框架和spa去量子化框架。 |
| [^15] | [Pre-Training Representations of Binary Code Using Contrastive Learning.](http://arxiv.org/abs/2210.05102) | 提出了一种使用对比学习预训练二进制代码表示的方法，可以将源代码和注释信息纳入二进制代码的表示学习中，对于反向工程和计算机安全任务有重要意义。 |

# 详细

[^1]: 一种信息论框架用于超出分布泛化

    An Information-Theoretic Framework for Out-of-Distribution Generalization

    [https://arxiv.org/abs/2403.19895](https://arxiv.org/abs/2403.19895)

    提出了一个信息论框架用于机器学习中的超出分布泛化，可以自由插值并产生新的泛化界限，同时具有最优输运解释。

    

    我们研究了机器学习中的超出分布（OOD）泛化，并提出了一个通用框架，提供了信息论泛化界限。我们的框架在Integral Probability Metric（IPM）和$f$-divergence之间自由插值，自然地恢复了一些已知结果（包括Wasserstein和KL-bound），并产生了新的泛化界限。此外，我们展示了我们的框架具有最优输运解释。在两个具体示例中评估时，所提出的界限在某些情况下严格改进了现有界限，或者恢复了现有OOD泛化界限中的最佳者。

    arXiv:2403.19895v1 Announce Type: cross  Abstract: We study the Out-of-Distribution (OOD) generalization in machine learning and propose a general framework that provides information-theoretic generalization bounds. Our framework interpolates freely between Integral Probability Metric (IPM) and $f$-divergence, which naturally recovers some known results (including Wasserstein- and KL-bounds), as well as yields new generalization bounds. Moreover, we show that our framework admits an optimal transport interpretation. When evaluated in two concrete examples, the proposed bounds either strictly improve upon existing bounds in some cases or recover the best among existing OOD generalization bounds.
    
[^2]: 通过微调预先为高斯降噪而训练的UNet进行低剂量CT图像重建，用于图像增强的下游任务

    Low-Dose CT Image Reconstruction by Fine-Tuning a UNet Pretrained for Gaussian Denoising for the Downstream Task of Image Enhancement

    [https://arxiv.org/abs/2403.03551](https://arxiv.org/abs/2403.03551)

    提出了一种通过精调UNet进行低剂量CT图像重建的方法，其中第二阶段的训练策略为CT图像增强阶段。

    

    计算机断层扫描（CT）是一种广泛使用的医学成像模态，由于其基于电离辐射，因此希望尽量减少辐射剂量。然而，降低辐射剂量会导致图像质量下降，从低剂量CT（LDCT）数据重建仍然是一个具有挑战性的任务，值得进行研究。根据LoDoPaB-CT基准，许多最先进的方法使用涉及UNet型架构的流程。具体来说，排名第一的方法ItNet使用包括滤波反投影（FBP）、在CT数据上训练的UNet和迭代细化步骤的三阶段流程。在本文中，我们提出了一种更简单的两阶段方法。第一阶段也使用了FBP，而新颖之处在于第二阶段的训练策略，特点是CT图像增强阶段。我们方法的关键点在于神经网络是预训练的。

    arXiv:2403.03551v1 Announce Type: cross  Abstract: Computed Tomography (CT) is a widely used medical imaging modality, and as it is based on ionizing radiation, it is desirable to minimize the radiation dose. However, a reduced radiation dose comes with reduced image quality, and reconstruction from low-dose CT (LDCT) data is still a challenging task which is subject to research. According to the LoDoPaB-CT benchmark, a benchmark for LDCT reconstruction, many state-of-the-art methods use pipelines involving UNet-type architectures. Specifically the top ranking method, ItNet, employs a three-stage process involving filtered backprojection (FBP), a UNet trained on CT data, and an iterative refinement step. In this paper, we propose a less complex two-stage method. The first stage also employs FBP, while the novelty lies in the training strategy for the second stage, characterized as the CT image enhancement stage. The crucial point of our approach is that the neural network is pretrained
    
[^3]: 封建网络用于视觉导航

    Feudal Networks for Visual Navigation

    [https://arxiv.org/abs/2402.12498](https://arxiv.org/abs/2402.12498)

    使用封建学习的视觉导航，通过高级管理者、中级管理者和工作代理的分层结构，在不同空间和时间尺度上操作，具有独特模块来实现自监督学习记忆代理地图。

    

    视觉导航遵循人类可以在没有详细地图的情况下导航的直觉。一种常见方法是在建立包含可用于规划的图像节点的拓扑图的同时进行交互式探索。最近的变体从被动视频中学习，并可以利用复杂的社交和语义线索进行导航。然而，需要大量的训练视频，利用大型图并且由于使用了里程计，场景不是未知的。我们引入了一种使用封建学习的视觉导航的新方法，该方法采用了由工作代理、中级管理者和高级管理者组成的分层结构。封建学习范式的关键在于，每个级别的代理看到任务的不同方面，并且在不同的空间和时间尺度上运作。在此框架中开发了两个独特的模块。对于高级管理者，我们自监督地学习一个记忆代理地图以记录

    arXiv:2402.12498v1 Announce Type: cross  Abstract: Visual navigation follows the intuition that humans can navigate without detailed maps. A common approach is interactive exploration while building a topological graph with images at nodes that can be used for planning. Recent variations learn from passive videos and can navigate using complex social and semantic cues. However, a significant number of training videos are needed, large graphs are utilized, and scenes are not unseen since odometry is utilized. We introduce a new approach to visual navigation using feudal learning, which employs a hierarchical structure consisting of a worker agent, a mid-level manager, and a high-level manager. Key to the feudal learning paradigm, agents at each level see a different aspect of the task and operate at different spatial and temporal scales. Two unique modules are developed in this framework. For the high- level manager, we learn a memory proxy map in a self supervised manner to record prio
    
[^4]: 遥感图像中的固体废物检测：一项调查

    Solid Waste Detection in Remote Sensing Images: A Survey

    [https://arxiv.org/abs/2402.09066](https://arxiv.org/abs/2402.09066)

    本文调查了固体废物在遥感图像中的检测方法。研究者利用地球观测卫星提供的高分辨率数据，通过遥感图像实现了固体废物处置场地的识别、监测和评估。

    

    识别和表征非法固体废物处置场地对环境保护至关重要，特别是应对污染和健康危害。不当管理的垃圾填埋场通过雨水渗透污染土壤和地下水，对动物和人类构成威胁。传统的填埋场辨识方法，如现场检查，耗时且昂贵。遥感技术是用于识别和监测固体废物处置场地的一种经济有效的解决方案，可以实现广泛覆盖和多次获取。地球观测（EO）卫星配备了一系列传感器和成像能力，几十年来一直提供高分辨率的数据。研究人员提出了专门的技术，利用遥感图像执行一系列任务，如废物场地检测、倾倒场监测和适宜位置评估。

    arXiv:2402.09066v1 Announce Type: cross Abstract: The detection and characterization of illegal solid waste disposal sites are essential for environmental protection, particularly for mitigating pollution and health hazards. Improperly managed landfills contaminate soil and groundwater via rainwater infiltration, posing threats to both animals and humans. Traditional landfill identification approaches, such as on-site inspections, are time-consuming and expensive. Remote sensing is a cost-effective solution for the identification and monitoring of solid waste disposal sites that enables broad coverage and repeated acquisitions over time. Earth Observation (EO) satellites, equipped with an array of sensors and imaging capabilities, have been providing high-resolution data for several decades. Researchers proposed specialized techniques that leverage remote sensing imagery to perform a range of tasks such as waste site detection, dumping site monitoring, and assessment of suitable locati
    
[^5]: 通过估计演示者的专业知识的逆向强化学习

    Inverse Reinforcement Learning by Estimating Expertise of Demonstrators

    [https://arxiv.org/abs/2402.01886](https://arxiv.org/abs/2402.01886)

    本文介绍了一个新颖的框架，IRLEED，它通过估计演示者的专业知识来解决模仿学习中的次优和异质演示的问题。IRLEED通过结合演示者次优性的普适模型和最大熵IRL框架，有效地从多样的次优演示中得出最佳策略。

    

    在模仿学习中，利用次优和异质的演示提出了一个重大挑战，因为现实世界数据的性质各不相同。然而，标准的模仿学习算法将这些数据集视为同质的，从而继承了次优演示的缺陷。先前处理这个问题的方法通常依赖于不切实际的假设，如高质量的数据子集、置信度排名或明确的环境知识。本文介绍了IRLEED（通过估计演示者的专业知识的逆向强化学习），这是一个新颖的框架，能够克服这些障碍，而不需要先前对演示者专业知识进行了解。IRLEED通过将演示者次优性的普适模型与最大熵IRL框架相结合，来处理奖励偏差和行动方差，从而有效地从多样的次优演示中得出最优策略。在在线和离线实验中进行了验证。

    In Imitation Learning (IL), utilizing suboptimal and heterogeneous demonstrations presents a substantial challenge due to the varied nature of real-world data. However, standard IL algorithms consider these datasets as homogeneous, thereby inheriting the deficiencies of suboptimal demonstrators. Previous approaches to this issue typically rely on impractical assumptions like high-quality data subsets, confidence rankings, or explicit environmental knowledge. This paper introduces IRLEED, Inverse Reinforcement Learning by Estimating Expertise of Demonstrators, a novel framework that overcomes these hurdles without prior knowledge of demonstrator expertise. IRLEED enhances existing Inverse Reinforcement Learning (IRL) algorithms by combining a general model for demonstrator suboptimality to address reward bias and action variance, with a Maximum Entropy IRL framework to efficiently derive the optimal policy from diverse, suboptimal demonstrations. Experiments in both online and offline I
    
[^6]: 用强化学习优化热预警的发布

    Optimizing Heat Alert Issuance with Reinforcement Learning

    [https://arxiv.org/abs/2312.14196](https://arxiv.org/abs/2312.14196)

    本研究利用强化学习优化热预警系统，通过引入新颖强化学习环境和综合数据集，解决了气候和健康环境中的低信号效应和空间异质性。

    

    社会适应气候变化的关键战略之一是利用预警系统减少极端高温事件的不利健康影响，以促使预防性行动。本文研究了强化学习（RL）作为优化此类系统效果的工具。我们的贡献有三个方面。首先，我们引入了一个新颖的强化学习环境，评估热预警政策的有效性，以减少与高温有关的住院人数。奖励模型基于历史天气、医疗保险健康记录以及社会经济/地理特征的全面数据集进行训练。我们使用变分贝叶斯技术解决了在气候和健康环境中常见的低信号效应和空间异质性。转换模型结合了真实的历史天气模式，并通过基于气候区域相似性的数据增强机制进行增强。

    arXiv:2312.14196v2 Announce Type: replace  Abstract: A key strategy in societal adaptation to climate change is the use of alert systems to reduce the adverse health impacts of extreme heat events by prompting preventative action. In this work, we investigate reinforcement learning (RL) as a tool to optimize the effectiveness of such systems. Our contributions are threefold. First, we introduce a novel RL environment enabling the evaluation of the effectiveness of heat alert policies to reduce heat-related hospitalizations. The rewards model is trained from a comprehensive dataset of historical weather, Medicare health records, and socioeconomic/geographic features. We use variational Bayesian techniques to address low-signal effects and spatial heterogeneity, which are commonly encountered in climate & health settings. The transition model incorporates real historical weather patterns enriched by a data augmentation mechanism based on climate region similarity. Second, we use this env
    
[^7]: 通过基于拓扑数据分析的聚类实现稀疏投资组合选择

    Sparse Portfolio Selection via Topological Data Analysis based Clustering. (arXiv:2401.16920v1 [q-fin.PM])

    [http://arxiv.org/abs/2401.16920](http://arxiv.org/abs/2401.16920)

    本文使用拓扑数据分析工具提出了一种基于聚类的稀疏投资组合选择策略，通过利用股票价格波动的拓扑特征，在持续图和景观空间上引入新的距离度量，并与聚类算法相结合，显著提升了多样市场情景下稀疏投资组合的绩效。

    

    本文使用拓扑数据分析工具，引入了一种针对稀疏投资组合构建的数据驱动聚类型股票选择策略。我们的资产选择策略利用股票价格波动的拓扑特征，选择一组拓扑类似（不同）的资产用于稀疏指数追踪（马科维茨）投资组合。我们引入了在持续图和景观空间上考虑时间成分的新距离度量，作为聚类算法的输入。我们对2009年至2020年的S\&P指数进行了实证分析，包括对COVID-19数据的研究，以验证我们方法的稳健性。我们将拓扑数据分析与聚类算法相结合的策略显著提升了不同市场情景下稀疏投资组合的综合绩效。

    This paper uses topological data analysis (TDA) tools and introduces a data-driven clustering-based stock selection strategy tailored for sparse portfolio construction. Our asset selection strategy exploits the topological features of stock price movements to select a subset of topologically similar (different) assets for a sparse index tracking (Markowitz) portfolio. We introduce new distance measures, which serve as an input to the clustering algorithm, on the space of persistence diagrams and landscapes that consider the time component of a time series. We conduct an empirical analysis on the S\&P index from 2009 to 2020, including a study on the COVID-19 data to validate the robustness of our methodology. Our strategy to integrate TDA with the clustering algorithm significantly enhanced the performance of sparse portfolios across various performance measures in diverse market scenarios.
    
[^8]: IGNITE: 个体化时间序列电子健康记录的生成模型

    IGNITE: Individualized GeNeration of Imputations in Time-series Electronic health records. (arXiv:2401.04402v1 [cs.LG])

    [http://arxiv.org/abs/2401.04402](http://arxiv.org/abs/2401.04402)

    个体化时间序列电子健康记录的生成模型IGNITE通过学习个体的动态特征，结合人口特征和治疗信息，生成个性化的真实值，为个体化医疗提供了有价值的方式。

    

    电子健康记录为推动个体化医疗提供了有价值的方式，可以根据个体差异量身定制治疗方案。为了实现这一目标，许多数据驱动的机器学习和统计模型借助丰富的纵向电子健康记录来研究患者的生理和治疗效果。然而，纵向电子健康记录往往稀疏且存在大量缺失，其中缺失的信息也可能反映患者的健康状况。因此，数据驱动模型在个体化医疗中的成功严重依赖于如何从生理数据、治疗以及数据中的缺失值来表示电子健康记录。为此，我们提出了一种新颖的深度学习模型，该模型可以在个体的人口特征和治疗的条件下，学习多变量数据的患者动态，并生成个性化的真实值。

    Electronic Health Records present a valuable modality for driving personalized medicine, where treatment is tailored to fit individual-level differences. For this purpose, many data-driven machine learning and statistical models rely on the wealth of longitudinal EHRs to study patients' physiological and treatment effects. However, longitudinal EHRs tend to be sparse and highly missing, where missingness could also be informative and reflect the underlying patient's health status. Therefore, the success of data-driven models for personalized medicine highly depends on how the EHR data is represented from physiological data, treatments, and the missing values in the data. To this end, we propose a novel deep-learning model that learns the underlying patient dynamics over time across multivariate data to generate personalized realistic values conditioning on an individual's demographic characteristics and treatments. Our proposed model, IGNITE (Individualized GeNeration of Imputations in
    
[^9]: 准确可扩展的图神经网络表观不确定性估计

    Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks. (arXiv:2401.03350v1 [cs.LG])

    [http://arxiv.org/abs/2401.03350](http://arxiv.org/abs/2401.03350)

    提出了G-$\Delta$UQ，一种新的训练框架，旨在改善图神经网络（GNN）的内在不确定性估计。该框架通过图锚定策略将随机数据中心化应用于图数据，并且能够支持部分随机的GNN。

    

    尽管图神经网络（GNN）广泛用于节点和图表示学习任务，但在分布变化下GNN不确定性估计的可靠性仍相对较少探索。事实上，虽然事后校准策略可以用于改善内部分布校准，但它们不一定也能改进分布变化下的校准。然而，产生更好的内部不确定性估计的技术尤其有价值，因为它们可以随后与事后策略结合使用。因此，在本研究中，我们提出了一种名为G-$\Delta$UQ的新型训练框架，旨在改善内在的GNN不确定性估计。我们的框架通过新颖的图锚定策略将随机数据中心化原则应用于图数据，并能够支持部分随机的GNN。虽然主流观点是为了获得可靠的估计，需要完全随机网络，但我们发现通过功能多样性引入的中观锚定可以在保证准确性的同时降低计算成本。

    While graph neural networks (GNNs) are widely used for node and graph representation learning tasks, the reliability of GNN uncertainty estimates under distribution shifts remains relatively under-explored. Indeed, while post-hoc calibration strategies can be used to improve in-distribution calibration, they need not also improve calibration under distribution shift. However, techniques which produce GNNs with better intrinsic uncertainty estimates are particularly valuable, as they can always be combined with post-hoc strategies later. Therefore, in this work, we propose G-$\Delta$UQ, a novel training framework designed to improve intrinsic GNN uncertainty estimates. Our framework adapts the principle of stochastic data centering to graph data through novel graph anchoring strategies, and is able to support partially stochastic GNNs. While, the prevalent wisdom is that fully stochastic networks are necessary to obtain reliable estimates, we find that the functional diversity induced b
    
[^10]: 通过最大化梯度多样性来解决联邦优化中的混合异构性

    Tackling Hybrid Heterogeneity on Federated Optimization via Gradient Diversity Maximization. (arXiv:2310.02702v1 [cs.LG])

    [http://arxiv.org/abs/2310.02702](http://arxiv.org/abs/2310.02702)

    本文探讨了混合异构性如何影响联邦优化，并提出了一种通过最大化梯度多样性来减轻混合异构性负面影响的方法。

    

    联邦学习是一种分布式机器学习范式，其中数据样本被分散和分布在多个客户端之间。这些样本可能表现出统计异质性，即数据分布在客户端之间不是独立和相同的。此外，系统异质性，即客户端计算能力的变化，会给联邦学习带来偏差。统计和系统异质性的综合效应可以显著降低联邦优化的效率。然而，混合异构性的影响并没有得到严谨的讨论。本文通过研究服务器端优化，探讨了混合异构性如何影响联邦优化。理论结果表明，在服务器更新方向上自适应地最大化梯度多样性可以帮助减轻混合异构性的潜在负面影响。为此，我们引入了一种新颖的基于服务器端梯度的优化器。

    Federated learning refers to a distributed machine learning paradigm in which data samples are decentralized and distributed among multiple clients. These samples may exhibit statistical heterogeneity, which refers to data distributions are not independent and identical across clients. Additionally, system heterogeneity, or variations in the computational power of the clients, introduces biases into federated learning. The combined effects of statistical and system heterogeneity can significantly reduce the efficiency of federated optimization. However, the impact of hybrid heterogeneity is not rigorously discussed. This paper explores how hybrid heterogeneity affects federated optimization by investigating server-side optimization. The theoretical results indicate that adaptively maximizing gradient diversity in server update direction can help mitigate the potential negative consequences of hybrid heterogeneity. To this end, we introduce a novel server-side gradient-based optimizer \
    
[^11]: 最优异构协同线性回归和上下文臂研究

    Optimal Heterogeneous Collaborative Linear Regression and Contextual Bandits. (arXiv:2306.06291v1 [stat.ML])

    [http://arxiv.org/abs/2306.06291](http://arxiv.org/abs/2306.06291)

    本文提出了一种新的估计器MOLAR，它利用协同线性回归和上下文臂问题中的稀疏异质性来提高估计精度，并且相比独立方法具有更好的表现。

    

    大型和复杂的数据集往往来自于几个可能是异构的来源。协同学习方法通过利用数据集之间的共性提高效率，同时考虑可能出现的差异。在这里，我们研究协同线性回归和上下文臂问题，其中每个实例的相关参数等于全局参数加上一个稀疏的实例特定术语。我们提出了一种名为MOLAR的新型二阶段估计器，它通过首先构建实例线性回归估计的逐项中位数，然后将实例特定估计值收缩到中位数附近来利用这种结构。与独立最小二乘估计相比，MOLAR提高了估计误差对数据维度的依赖性。然后，我们将MOLAR应用于开发用于稀疏异构协同上下文臂的方法，这些方法相比独立臂模型具有更好的遗憾保证。我们进一步证明了我们的贡献优于先前在文献中报道的算法。

    Large and complex datasets are often collected from several, possibly heterogeneous sources. Collaborative learning methods improve efficiency by leveraging commonalities across datasets while accounting for possible differences among them. Here we study collaborative linear regression and contextual bandits, where each instance's associated parameters are equal to a global parameter plus a sparse instance-specific term. We propose a novel two-stage estimator called MOLAR that leverages this structure by first constructing an entry-wise median of the instances' linear regression estimates, and then shrinking the instance-specific estimates towards the median. MOLAR improves the dependence of the estimation error on the data dimension, compared to independent least squares estimates. We then apply MOLAR to develop methods for sparsely heterogeneous collaborative contextual bandits, which lead to improved regret guarantees compared to independent bandit methods. We further show that our 
    
[^12]: 追赶蒸馏：加速采样只需一次训练

    Catch-Up Distillation: You Only Need to Train Once for Accelerating Sampling. (arXiv:2305.10769v1 [cs.LG])

    [http://arxiv.org/abs/2305.10769](http://arxiv.org/abs/2305.10769)

    本文提出了一种名为“追赶蒸馏”的方法，通过调整传统采样算法，让速度估计模型的当前时刻输出与其先前时刻输出和地面真实标签对齐，从而实现只需一次训练便能加速采样的效果。

    

    扩散概率模型在各种机器学习领域取得了令人瞩目的进展。然而，为了实现高质量的合成样本，通常需要执行大量的采样步骤，这阻碍了实时样本合成的可能性。传统的通过知识蒸馏加速采样的算法依赖于预训练的模型权重和离散时间步骤场景，需要额外的培训课程才能实现他们的目标。为了解决这些问题，我们提出了追赶蒸馏（CUD），它鼓励速度估计模型的当前时刻输出“追赶”其先前时刻输出。具体而言，CUD调整了原始的常微分方程（ODE）训练目标，以使当前时刻输出与地面真实标签和先前时刻输出对齐，利用基于龙格-库塔的多步对齐蒸馏进行精确的ODE估计，同时防止异步更新。

    Diffusion Probability Models (DPMs) have made impressive advancements in various machine learning domains. However, achieving high-quality synthetic samples typically involves performing a large number of sampling steps, which impedes the possibility of real-time sample synthesis. Traditional accelerated sampling algorithms via knowledge distillation rely on pre-trained model weights and discrete time step scenarios, necessitating additional training sessions to achieve their goals. To address these issues, we propose the Catch-Up Distillation (CUD), which encourages the current moment output of the velocity estimation model ``catch up'' with its previous moment output. Specifically, CUD adjusts the original Ordinary Differential Equation (ODE) training objective to align the current moment output with both the ground truth label and the previous moment output, utilizing Runge-Kutta-based multi-step alignment distillation for precise ODE estimation while preventing asynchronous updates
    
[^13]: 旨在表征基于胶囊网络架构学习的表示方法

    Towards the Characterization of Representations Learned via Capsule-based Network Architectures. (arXiv:2305.05349v1 [cs.LG])

    [http://arxiv.org/abs/2305.05349](http://arxiv.org/abs/2305.05349)

    本研究旨在评估胶囊网络架构学习的表示方法及其可解释性，发现其编码的表示可能与部分-整体关系并不严格相关。

    

    胶囊网络作为标准深度神经网络的一种更为紧凑和可解释的替代方法而重新引入。尽管最近的研究证明了其压缩能力，但至今尚未完全评估其可解释性质。在这里，我们进行了一项系统而原则性的研究，以评估这种类型网络的可解释性。此外，我们特别注意分析所学到的表示中是否确实编码了部分-整体关系的水平。在MNIST、SVHN、PASCAL-part和CelebA数据集中的分析表明，在CapsNets中编码的表示可能既不像文献中通常所述的那样分离，也不是严格与部分-整体关系相关的。

    Capsule Networks (CapsNets) have been re-introduced as a more compact and interpretable alternative to standard deep neural networks. While recent efforts have proved their compression capabilities, to date, their interpretability properties have not been fully assessed. Here, we conduct a systematic and principled study towards assessing the interpretability of these types of networks. Moreover, we pay special attention towards analyzing the level to which part-whole relationships are indeed encoded within the learned representation. Our analysis in the MNIST, SVHN, PASCAL-part and CelebA datasets suggest that the representations encoded in CapsNets might not be as disentangled nor strictly related to parts-whole relationships as is commonly stated in the literature.
    
[^14]: 量子奇异值变换及量子机器学习算法的鲁棒去量子化方法研究

    Robust Dequantization of the Quantum Singular value Transformation and Quantum Machine Learning Algorithms. (arXiv:2304.04932v1 [quant-ph])

    [http://arxiv.org/abs/2304.04932](http://arxiv.org/abs/2304.04932)

    本文研究了量子机器学习算法的鲁棒去量子化方法。我们提出了近似长度平方采样的概念，并展示了如何将随机线性代数技术适应到这种更弱的假设下。我们使用这些技术证明了最近的低秩去量子化框架和spa去量子化框架。

    

    在过去的几年中，已经有几种用于解决线性代数问题和特别是量子机器学习问题的量子算法被“去量子化”。这些去量子化结果通常在经典算法通过长度平方采样方法访问数据时成立。本文研究了这些去量子化结果的稳健性。我们引入了近似长度平方采样的概念，其中经典算法只能从接近理想分布的分布中进行采样。虽然量子算法在面对小扰动时本质上是鲁棒的，但当前的去量子化技术并不是。我们的主要技术贡献在于展示了如何将许多随机线性代数技术适应到这种更弱的假设下。然后，我们使用这些技术证明了最近由Chia、Gily\'en、Li、Lin、Tang和Wang（JACM 2022）提出的低秩去量子化框架和用于spa的去量子化框架。

    Several quantum algorithms for linear algebra problems, and in particular quantum machine learning problems, have been "dequantized" in the past few years. These dequantization results typically hold when classical algorithms can access the data via length-squared sampling. In this work we investigate how robust these dequantization results are. We introduce the notion of approximate length-squared sampling, where classical algorithms are only able to sample from a distribution close to the ideal distribution in total variation distance. While quantum algorithms are natively robust against small perturbations, current techniques in dequantization are not. Our main technical contribution is showing how many techniques from randomized linear algebra can be adapted to work under this weaker assumption as well. We then use these techniques to show that the recent low-rank dequantization framework by Chia, Gily\'en, Li, Lin, Tang and Wang (JACM 2022) and the dequantization framework for spa
    
[^15]: 使用对比学习预训练二进制代码表示

    Pre-Training Representations of Binary Code Using Contrastive Learning. (arXiv:2210.05102v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2210.05102](http://arxiv.org/abs/2210.05102)

    提出了一种使用对比学习预训练二进制代码表示的方法，可以将源代码和注释信息纳入二进制代码的表示学习中，对于反向工程和计算机安全任务有重要意义。

    

    编译后的软件以可执行的二进制代码形式交付。开发人员编写源代码来表达软件的语义，但编译器将其转换为CPU可以直接执行的二进制格式。因此，二进制代码分析对于反向工程和计算机安全任务等没有源代码的应用程序至关重要。然而，与包含丰富语义信息的源代码和自然语言不同，二进制代码通常难以理解和分析。虽然现有的工作使用AI模型辅助源代码分析，但很少有研究考虑二进制代码。在本文中，我们提出了一种将源代码和注释信息纳入二进制代码进行表示学习的对比学习模型，称为COMBO。具体而言，我们在COMBO中提出了三个组件：（1）用于冷启动预训练的主要对比学习方法，（2）用于将源代码和注释信息插入到二进制代码中的单纯插值方法。

    Compiled software is delivered as executable binary code. Developers write source code to express the software semantics, but the compiler converts it to a binary format that the CPU can directly execute. Therefore, binary code analysis is critical to applications in reverse engineering and computer security tasks where source code is not available. However, unlike source code and natural language that contain rich semantic information, binary code is typically difficult for human engineers to understand and analyze. While existing work uses AI models to assist source code analysis, few studies have considered binary code. In this paper, we propose a COntrastive learning Model for Binary cOde Analysis, or COMBO, that incorporates source code and comment information into binary code during representation learning. Specifically, we present three components in COMBO: (1) a primary contrastive learning method for cold-start pre-training, (2) a simplex interpolation method to incorporate so
    

