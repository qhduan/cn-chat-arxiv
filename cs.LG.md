# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Are you a robot? Detecting Autonomous Vehicles from Behavior Analysis](https://arxiv.org/abs/2403.09571) | 提出了一个框架，通过监视车辆的行为和状态信息来自动识别自动驾驶车辆，无需车辆主动通知。 |
| [^2] | [Variational Inference of Parameters in Opinion Dynamics Models](https://arxiv.org/abs/2403.05358) | 通过将估计问题转化为可直接解决的优化任务，本研究提出了一种使用变分推断来估计意见动态ABM参数的方法。 |
| [^3] | [Tackling Missing Values in Probabilistic Wind Power Forecasting: A Generative Approach](https://arxiv.org/abs/2403.03631) | 本文提出了一种新的概率风力发电预测方法，通过生成模型估计特征和目标的联合分布，同时预测所有未知值，避免了预处理环节，在连续排名概率得分方面比传统方法表现更优。 |
| [^4] | [From Large to Small Datasets: Size Generalization for Clustering Algorithm Selection](https://arxiv.org/abs/2402.14332) | 通过引入尺寸泛化概念，研究了在半监督设置下的聚类算法选择问题，提出了能够在小实例上保证准确度最高的算法也将在原始大实例上拥有最高准确度的条件。 |
| [^5] | [Scheduling and Aggregation Design for Asynchronous Federated Learning over Wireless Networks.](http://arxiv.org/abs/2212.07356) | 本文提出了一种异步联邦学习的调度策略和聚合加权设计，通过采用基于信道感知数据重要性的调度策略和“年龄感知”的聚合加权设计来解决FL系统中的“拖沓”问题，并通过仿真证实了其有效性。 |

# 详细

[^1]: 你是机器人吗？从行为分析中检测自动驾驶车辆

    Are you a robot? Detecting Autonomous Vehicles from Behavior Analysis

    [https://arxiv.org/abs/2403.09571](https://arxiv.org/abs/2403.09571)

    提出了一个框架，通过监视车辆的行为和状态信息来自动识别自动驾驶车辆，无需车辆主动通知。

    

    自动驾驶技术的巨大热潮急切地呼唤新兴和创新技术，以支持先进的移动性使用案例。随着汽车制造商不断开发SAE 3级及以上系统来提高乘客的安全性和舒适性，交通管理机构需要建立新程序来管理从人工驾驶到完全自动驾驶车辆的过渡，并提供一个反馈机制来微调设想的自动驾驶系统。因此，自动对自动驾驶车辆进行自动分析并将其与人工驾驶车辆区分开是必要的。本文提出了一个完整的框架，通过监视使用摄像头图像和状态信息的活动车辆，以确定车辆是否为自动驾驶，而无需车辆主动通知。基本上，它依赖车辆之间的合作，这些车辆共享在道路上获取的数据，供机器学习使用。

    arXiv:2403.09571v1 Announce Type: cross  Abstract: The tremendous hype around autonomous driving is eagerly calling for emerging and novel technologies to support advanced mobility use cases. As car manufactures keep developing SAE level 3+ systems to improve the safety and comfort of passengers, traffic authorities need to establish new procedures to manage the transition from human-driven to fully-autonomous vehicles while providing a feedback-loop mechanism to fine-tune envisioned autonomous systems. Thus, a way to automatically profile autonomous vehicles and differentiate those from human-driven ones is a must. In this paper, we present a fully-fledged framework that monitors active vehicles using camera images and state information in order to determine whether vehicles are autonomous, without requiring any active notification from the vehicles themselves. Essentially, it builds on the cooperation among vehicles, which share their data acquired on the road feeding a machine learn
    
[^2]: 意见动态模型中参数的变分推断

    Variational Inference of Parameters in Opinion Dynamics Models

    [https://arxiv.org/abs/2403.05358](https://arxiv.org/abs/2403.05358)

    通过将估计问题转化为可直接解决的优化任务，本研究提出了一种使用变分推断来估计意见动态ABM参数的方法。

    

    尽管基于代理人的模型（ABMs）在研究社会现象中被频繁使用，但参数估计仍然是一个挑战，通常依赖于昂贵的基于模拟的启发式方法。本研究利用变分推断来估计意见动态ABM的参数，通过将估计问题转化为可直接解决的优化任务。

    arXiv:2403.05358v1 Announce Type: cross  Abstract: Despite the frequent use of agent-based models (ABMs) for studying social phenomena, parameter estimation remains a challenge, often relying on costly simulation-based heuristics. This work uses variational inference to estimate the parameters of an opinion dynamics ABM, by transforming the estimation problem into an optimization task that can be solved directly.   Our proposal relies on probabilistic generative ABMs (PGABMs): we start by synthesizing a probabilistic generative model from the ABM rules. Then, we transform the inference process into an optimization problem suitable for automatic differentiation. In particular, we use the Gumbel-Softmax reparameterization for categorical agent attributes and stochastic variational inference for parameter estimation. Furthermore, we explore the trade-offs of using variational distributions with different complexity: normal distributions and normalizing flows.   We validate our method on a
    
[^3]: 处理概率风力发电预测中的缺失值：一种生成方法

    Tackling Missing Values in Probabilistic Wind Power Forecasting: A Generative Approach

    [https://arxiv.org/abs/2403.03631](https://arxiv.org/abs/2403.03631)

    本文提出了一种新的概率风力发电预测方法，通过生成模型估计特征和目标的联合分布，同时预测所有未知值，避免了预处理环节，在连续排名概率得分方面比传统方法表现更优。

    

    机器学习技术已成功应用于概率风力发电预测。然而，由于传感器故障等原因导致数据集中存在缺失值的问题长期以来被忽视。尽管通常在模型估计和预测之前通过插补缺失值来解决这个问题是很自然的，但我们建议将缺失值和预测目标视为同等重要，并基于观测值同时预测所有未知值。本文通过基于生成模型估计特征和目标的联合分布，提出了一种有效的概率预测方法。这种方法无需预处理，避免引入潜在的错误。与传统的“插补，然后预测”流程相比，该方法在连续排名概率得分方面表现更好。

    arXiv:2403.03631v1 Announce Type: new  Abstract: Machine learning techniques have been successfully used in probabilistic wind power forecasting. However, the issue of missing values within datasets due to sensor failure, for instance, has been overlooked for a long time. Although it is natural to consider addressing this issue by imputing missing values before model estimation and forecasting, we suggest treating missing values and forecasting targets indifferently and predicting all unknown values simultaneously based on observations. In this paper, we offer an efficient probabilistic forecasting approach by estimating the joint distribution of features and targets based on a generative model. It is free of preprocessing, and thus avoids introducing potential errors. Compared with the traditional "impute, then predict" pipeline, the proposed approach achieves better performance in terms of continuous ranked probability score.
    
[^4]: 从大规模到小规模数据集：用于聚类算法选择的尺寸泛化

    From Large to Small Datasets: Size Generalization for Clustering Algorithm Selection

    [https://arxiv.org/abs/2402.14332](https://arxiv.org/abs/2402.14332)

    通过引入尺寸泛化概念，研究了在半监督设置下的聚类算法选择问题，提出了能够在小实例上保证准确度最高的算法也将在原始大实例上拥有最高准确度的条件。

    

    在聚类算法选择中，我们会得到一个大规模数据集，并要有效地选择要使用的聚类算法。我们在半监督设置下研究了这个问题，其中有一个未知的基准聚类，我们只能通过昂贵的oracle查询来访问。理想情况下，聚类算法的输出将与基本事实结构上接近。我们通过引入一种聚类算法准确性的尺寸泛化概念来解决这个问题。我们确定在哪些条件下我们可以（1）对大规模聚类实例进行子采样，（2）在较小实例上评估一组候选算法，（3）保证在小实例上准确度最高的算法将在原始大实例上拥有最高的准确度。我们为三种经典聚类算法提供了理论尺寸泛化保证：单链接、k-means++和Gonzalez的k中心启发式（一种平滑的变种）。

    arXiv:2402.14332v1 Announce Type: new  Abstract: In clustering algorithm selection, we are given a massive dataset and must efficiently select which clustering algorithm to use. We study this problem in a semi-supervised setting, with an unknown ground-truth clustering that we can only access through expensive oracle queries. Ideally, the clustering algorithm's output will be structurally close to the ground truth. We approach this problem by introducing a notion of size generalization for clustering algorithm accuracy. We identify conditions under which we can (1) subsample the massive clustering instance, (2) evaluate a set of candidate algorithms on the smaller instance, and (3) guarantee that the algorithm with the best accuracy on the small instance will have the best accuracy on the original big instance. We provide theoretical size generalization guarantees for three classic clustering algorithms: single-linkage, k-means++, and (a smoothed variant of) Gonzalez's k-centers heuris
    
[^5]: 异步联邦学习在无线网络中的调度和聚合设计

    Scheduling and Aggregation Design for Asynchronous Federated Learning over Wireless Networks. (arXiv:2212.07356v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.07356](http://arxiv.org/abs/2212.07356)

    本文提出了一种异步联邦学习的调度策略和聚合加权设计，通过采用基于信道感知数据重要性的调度策略和“年龄感知”的聚合加权设计来解决FL系统中的“拖沓”问题，并通过仿真证实了其有效性。

    

    联邦学习（FL）是一种协作的机器学习（ML）框架，它结合了设备上的训练和基于服务器的聚合来在分布式代理间训练通用的ML模型。本文中，我们提出了一种异步FL设计，采用周期性的聚合来解决FL系统中的“拖沓”问题。考虑到有限的无线通信资源，我们研究了不同调度策略和聚合设计对收敛性能的影响。基于降低聚合模型更新的偏差和方差的重要性，我们提出了一个调度策略，它同时考虑了用户设备的信道质量和训练数据表示。通过仿真验证了我们的基于信道感知数据重要性的调度策略相对于同步联邦学习提出的现有最新方法的有效性。此外，我们还展示了一种“年龄感知”的聚合加权设计可以显著提高学习性能。

    Federated Learning (FL) is a collaborative machine learning (ML) framework that combines on-device training and server-based aggregation to train a common ML model among distributed agents. In this work, we propose an asynchronous FL design with periodic aggregation to tackle the straggler issue in FL systems. Considering limited wireless communication resources, we investigate the effect of different scheduling policies and aggregation designs on the convergence performance. Driven by the importance of reducing the bias and variance of the aggregated model updates, we propose a scheduling policy that jointly considers the channel quality and training data representation of user devices. The effectiveness of our channel-aware data-importance-based scheduling policy, compared with state-of-the-art methods proposed for synchronous FL, is validated through simulations. Moreover, we show that an ``age-aware'' aggregation weighting design can significantly improve the learning performance i
    

