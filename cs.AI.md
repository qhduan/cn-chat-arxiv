# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Joint Problems in Learning Multiple Dynamical Systems](https://arxiv.org/abs/2311.02181) | 聚类时间序列的新问题，提出联合划分轨迹集并学习每个部分的线性动态系统模型，以最小化所有模型的最大误差 |
| [^2] | [Radio Map Estimation in the Real-World: Empirical Validation and Analysis.](http://arxiv.org/abs/2310.11036) | 本文通过对现有的无线电地图估计器进行经验证据的评估，研究了性能和复杂性之间的权衡以及快速衰落的影响。尽管基于深度神经网络的估计器表现最佳，但需要大量的训练数据。一种混合了传统方案和深度神经网络的新算法表现良好。 |
| [^3] | [Exploring Naming Conventions (and Defects) of Pre-trained Deep Learning Models in Hugging Face and Other Model Hubs.](http://arxiv.org/abs/2310.01642) | 本研究首次系统研究了预训练深度学习模型的命名惯例和相关缺陷，为我们了解研究到实践过程提供了知识和认识。 |
| [^4] | [LEGO: Learning and Graph-Optimized Modular Tracker for Online Multi-Object Tracking with Point Clouds.](http://arxiv.org/abs/2308.09908) | 本文提出了一个学习和图优化的模块化跟踪器LEGO，通过集成图优化和自注意力机制，提高了在线多目标跟踪中的数据关联性能。使用LiDAR单独进行跟踪的LEGO方法在KITTI目标跟踪评估中表现出了优秀的性能。 |

# 详细

[^1]: 学习多个动态系统中的联合问题

    Joint Problems in Learning Multiple Dynamical Systems

    [https://arxiv.org/abs/2311.02181](https://arxiv.org/abs/2311.02181)

    聚类时间序列的新问题，提出联合划分轨迹集并学习每个部分的线性动态系统模型，以最小化所有模型的最大误差

    

    时间序列的聚类是一个经过充分研究的问题，其应用范围从通过代谢产物浓度获得的定量个性化代谢模型到量子信息理论中的状态判别。我们考虑了一个变种，即给定一组轨迹和一些部分，我们联合划分轨迹集并学习每个部分的线性动态系统（LDS）模型，以使得所有模型的最大误差最小化。我们提出了全局收敛的方法和EM启发式算法，并附上了有前景的计算结果。

    arXiv:2311.02181v2 Announce Type: replace-cross  Abstract: Clustering of time series is a well-studied problem, with applications ranging from quantitative, personalized models of metabolism obtained from metabolite concentrations to state discrimination in quantum information theory. We consider a variant, where given a set of trajectories and a number of parts, we jointly partition the set of trajectories and learn linear dynamical system (LDS) models for each part, so as to minimize the maximum error across all the models. We present globally convergent methods and EM heuristics, accompanied by promising computational results.
    
[^2]: 现实世界中的无线电地图估计：经验证和分析

    Radio Map Estimation in the Real-World: Empirical Validation and Analysis. (arXiv:2310.11036v1 [eess.SP])

    [http://arxiv.org/abs/2310.11036](http://arxiv.org/abs/2310.11036)

    本文通过对现有的无线电地图估计器进行经验证据的评估，研究了性能和复杂性之间的权衡以及快速衰落的影响。尽管基于深度神经网络的估计器表现最佳，但需要大量的训练数据。一种混合了传统方案和深度神经网络的新算法表现良好。

    

    无线电地图在地理区域的每个点上量化了接收信号强度或其他无线电频率环境的大小。这些地图在无线网络规划、频谱管理和通信系统优化等众多应用中起着重要作用。然而，对现有的大量无线电地图估计器的经验证据非常有限。为了填补这一空白，使用自主无人机（UAV）收集了大量的测量数据，并对这些估计器的代表性子集进行了评估。在这些评估中，广泛研究了性能和复杂性之间的权衡以及快速衰落的影响。尽管基于深度神经网络（DNN）的复杂估计器表现最佳，但它们需要大量的训练数据才能相对传统方案提供实质性优势。一种混合了两种类型估计器的新算法被发现具有良好的性能。

    Radio maps quantify received signal strength or other magnitudes of the radio frequency environment at every point of a geographical region. These maps play a vital role in a large number of applications such as wireless network planning, spectrum management, and optimization of communication systems. However, empirical validation of the large number of existing radio map estimators is highly limited. To fill this gap, a large data set of measurements has been collected with an autonomous unmanned aerial vehicle (UAV) and a representative subset of these estimators were evaluated on this data. The performance-complexity trade-off and the impact of fast fading are extensively investigated. Although sophisticated estimators based on deep neural networks (DNNs) exhibit the best performance, they are seen to require large volumes of training data to offer a substantial advantage relative to more traditional schemes. A novel algorithm that blends both kinds of estimators is seen to enjoy th
    
[^3]: 探索Hugging Face和其他模型仓库中预训练深度学习模型的命名惯例（及缺陷）

    Exploring Naming Conventions (and Defects) of Pre-trained Deep Learning Models in Hugging Face and Other Model Hubs. (arXiv:2310.01642v1 [cs.SE])

    [http://arxiv.org/abs/2310.01642](http://arxiv.org/abs/2310.01642)

    本研究首次系统研究了预训练深度学习模型的命名惯例和相关缺陷，为我们了解研究到实践过程提供了知识和认识。

    

    随着深度学习的创新不断推进，许多工程师希望将预训练深度学习模型（PTMs）作为计算系统的组成部分。PTMs是研究到实践的流程的一部分：研究人员发布PTMs，工程师根据质量或性能进行调整并部署。如果PTM的作者为其选择适当的名称，可以促进模型的发现和复用。然而，先前的研究已经报道了模型名称并不总是选择得很好，有时甚至是错误的。PTM包的命名惯例和命名缺陷尚未得到系统的研究，了解它们将增加我们对PTM包的研究到实践过程运作方式的认识。本文报告了对PTM命名惯例及相关命名缺陷的首次研究。我们定义了PTM包名称的组成部分，包括元数据中的包名称和声明的架构。我们展示了第一项旨在描述PTM命名性质的研究。

    As innovation in deep learning continues, many engineers want to adopt Pre-Trained deep learning Models (PTMs) as components in computer systems. PTMs are part of a research-to-practice pipeline: researchers publish PTMs, which engineers adapt for quality or performance and then deploy. If PTM authors choose appropriate names for their PTMs, it could facilitate model discovery and reuse. However, prior research has reported that model names are not always well chosen, and are sometimes erroneous. The naming conventions and naming defects for PTM packages have not been systematically studied - understanding them will add to our knowledge of how the research-to-practice process works for PTM packages  In this paper, we report the first study of PTM naming conventions and the associated PTM naming defects. We define the components of a PTM package name, comprising the package name and claimed architecture from the metadata. We present the first study focused on characterizing the nature o
    
[^4]: LEGO: 对于基于点云的在线多目标跟踪的学习和图优化的模块化跟踪器

    LEGO: Learning and Graph-Optimized Modular Tracker for Online Multi-Object Tracking with Point Clouds. (arXiv:2308.09908v1 [cs.CV])

    [http://arxiv.org/abs/2308.09908](http://arxiv.org/abs/2308.09908)

    本文提出了一个学习和图优化的模块化跟踪器LEGO，通过集成图优化和自注意力机制，提高了在线多目标跟踪中的数据关联性能。使用LiDAR单独进行跟踪的LEGO方法在KITTI目标跟踪评估中表现出了优秀的性能。

    

    在线多目标跟踪（MOT）在自主系统中起着关键作用。现有的最先进方法通常采用跟踪-检测方法，数据关联起到了至关重要的作用。本文提出了一个学习和图优化（LEGO）的模块化跟踪器，以提高数据关联性能。所提出的LEGO跟踪器集成了图优化和自注意力机制，能够有效地制定关联评分图，从而实现准确高效的目标匹配。为了进一步增强状态更新过程，本文还添加了卡尔曼滤波器，通过将对象状态的时间连贯性纳入跟踪中，确保一致的跟踪。与其他在线跟踪方法（包括基于LiDAR和基于LiDAR-相机融合的方法）相比，我们提出的仅利用LiDAR的方法表现出了卓越性能。在提交结果至KITTI目标跟踪评估排行榜时，LEGO排名第一。

    Online multi-object tracking (MOT) plays a pivotal role in autonomous systems. The state-of-the-art approaches usually employ a tracking-by-detection method, and data association plays a critical role. This paper proposes a learning and graph-optimized (LEGO) modular tracker to improve data association performance in the existing literature. The proposed LEGO tracker integrates graph optimization and self-attention mechanisms, which efficiently formulate the association score map, facilitating the accurate and efficient matching of objects across time frames. To further enhance the state update process, the Kalman filter is added to ensure consistent tracking by incorporating temporal coherence in the object states. Our proposed method utilizing LiDAR alone has shown exceptional performance compared to other online tracking approaches, including LiDAR-based and LiDAR-camera fusion-based methods. LEGO ranked 1st at the time of submitting results to KITTI object tracking evaluation ranki
    

