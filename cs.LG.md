# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decentralized Stochastic Subgradient Methods for Nonsmooth Nonconvex Optimization](https://arxiv.org/abs/2403.11565) | 该论文介绍了一种名为DSM的统一框架，用于分析去中心化随机次梯度方法的全局收敛性，证明了在温和条件下的全局收敛性，并展示其涵盖了各种现有高效的去中心化次梯度方法。 |
| [^2] | [Meta-learning the mirror map in policy mirror descent](https://arxiv.org/abs/2402.05187) | 该论文通过实证研究发现，传统的镜像映射选择（NPG）在标准基准环境中常常导致不理想的结果。通过元学习方法，找到了更高效的镜像映射，提升了性能。 |
| [^3] | [Arrival Time Prediction for Autonomous Shuttle Services in the Real World: Evidence from Five Cities.](http://arxiv.org/abs/2401.05322) | 本研究提出了一个自动穿梭巴士的到达时间预测系统，利用分别用于停留时间和运行时间预测的模型，并利用实际数据进行验证。通过集成空间数据和使用层次化模型处理绕过站点的情况，得到了可靠的AT预测结果。 |
| [^4] | [Time-Transformer: Integrating Local and Global Features for Better Time Series Generation.](http://arxiv.org/abs/2312.11714) | 本文提出了一种新的时间序列生成模型，通过时间变换器同时学习本地和全局特征，实现了对时间序列数据的更好生成能力。 |
| [^5] | [A new high-resolution indoor radon map for Germany using a machine learning based probabilistic exposure model.](http://arxiv.org/abs/2310.11143) | 本研究提出了一种基于机器学习的概率暴露模型，可以更准确地估计德国室内氡气分布，并具有更高的空间分辨率。 |

# 详细

[^1]: 基于去中心化随机次梯度法的非平滑非凸优化问题

    Decentralized Stochastic Subgradient Methods for Nonsmooth Nonconvex Optimization

    [https://arxiv.org/abs/2403.11565](https://arxiv.org/abs/2403.11565)

    该论文介绍了一种名为DSM的统一框架，用于分析去中心化随机次梯度方法的全局收敛性，证明了在温和条件下的全局收敛性，并展示其涵盖了各种现有高效的去中心化次梯度方法。

    

    在这篇论文中，我们关注具有非凸和非平滑目标函数的去中心化优化问题，特别是关注非平滑神经网络的去中心化训练。我们提出了一个统一的框架，称为DSM，用于分析去中心化随机次梯度法的全局收敛性。我们证明了在温和条件下，我们提出的框架的全局收敛性，通过建立生成序列渐近逼近其关联微分包含的轨迹。此外，我们证明了我们提出的框架涵盖了各种现有高效的去中心化次梯度方法，包括去中心化随机次梯度下降（DSGD），具有梯度跟踪技术的DSGD（DSGD-T）和带动量的DSGD（DSGDm）。此外，我们引入SignSGD，采用符号映射来正则化DSGDm中的更新方向，并表明它被包含在我们的提议中。

    arXiv:2403.11565v1 Announce Type: cross  Abstract: In this paper, we concentrate on decentralized optimization problems with nonconvex and nonsmooth objective functions, especially on the decentralized training of nonsmooth neural networks. We introduce a unified framework, named DSM, to analyze the global convergence of decentralized stochastic subgradient methods. We prove the global convergence of our proposed framework under mild conditions, by establishing that the generated sequence asymptotically approximates the trajectories of its associated differential inclusion. Furthermore, we establish that our proposed framework encompasses a wide range of existing efficient decentralized subgradient methods, including decentralized stochastic subgradient descent (DSGD), DSGD with gradient-tracking technique (DSGD-T), and DSGD with momentum (DSGDm). In addition, we introduce SignSGD employing the sign map to regularize the update directions in DSGDm, and show it is enclosed in our propos
    
[^2]: 在策略镜像下降中元学习镜像映射

    Meta-learning the mirror map in policy mirror descent

    [https://arxiv.org/abs/2402.05187](https://arxiv.org/abs/2402.05187)

    该论文通过实证研究发现，传统的镜像映射选择（NPG）在标准基准环境中常常导致不理想的结果。通过元学习方法，找到了更高效的镜像映射，提升了性能。

    

    策略镜像下降（PMD）是强化学习中的一种流行框架，作为一种统一视角，它包含了许多算法。这些算法是通过选择一个镜像映射而导出的，并且具有有限时间的收敛保证。尽管它很受欢迎，但对PMD的全面潜力的探索是有限的，大部分研究集中在一个特定的镜像映射上，即负熵，从而产生了著名的自然策略梯度（NPG）方法。目前的理论研究还不确定镜像映射的选择是否会对PMD的有效性产生重大影响。在我们的工作中，我们进行了实证研究，证明了传统的镜像映射选择（NPG）在几个标准基准环境中经常产生不理想的结果。通过应用元学习方法，我们确定了更高效的镜像映射，提高了性能，无论是平均性能还是最佳性能。

    Policy Mirror Descent (PMD) is a popular framework in reinforcement learning, serving as a unifying perspective that encompasses numerous algorithms. These algorithms are derived through the selection of a mirror map and enjoy finite-time convergence guarantees. Despite its popularity, the exploration of PMD's full potential is limited, with the majority of research focusing on a particular mirror map -- namely, the negative entropy -- which gives rise to the renowned Natural Policy Gradient (NPG) method. It remains uncertain from existing theoretical studies whether the choice of mirror map significantly influences PMD's efficacy. In our work, we conduct empirical investigations to show that the conventional mirror map choice (NPG) often yields less-than-optimal outcomes across several standard benchmark environments. By applying a meta-learning approach, we identify more efficient mirror maps that enhance performance, both on average and in terms of best performance achieved along th
    
[^3]: 在真实世界中为自动穿梭巴士服务的到达时间预测: 来自五个城市的证据

    Arrival Time Prediction for Autonomous Shuttle Services in the Real World: Evidence from Five Cities. (arXiv:2401.05322v1 [cs.LG])

    [http://arxiv.org/abs/2401.05322](http://arxiv.org/abs/2401.05322)

    本研究提出了一个自动穿梭巴士的到达时间预测系统，利用分别用于停留时间和运行时间预测的模型，并利用实际数据进行验证。通过集成空间数据和使用层次化模型处理绕过站点的情况，得到了可靠的AT预测结果。

    

    随着共享、连接和协作的自动驾驶车辆的出现，城市移动性正处于转型的边缘。然而，要想被客户接受，对它们的准点性的信任至关重要。许多试点项目没有固定时间表，因此可靠的到达时间（AT）预测的重要性得到了增强。本研究提出了一个针对自动穿梭巴士的AT预测系统，利用分别用于停留时间和运行时间预测的模型，并利用来自五个城市的实际数据进行验证。除了常用的方法如XGBoost外，我们还探索了使用图神经网络（GNN）集成空间数据的益处。为了准确处理穿梭巴士绕过站点的情况，我们提出了一个层次化模型，结合了随机森林分类器和GNN。最终的AT预测结果很有前景，即使预测数个站点之前也显示出较低的误差。然而，并没有单一的模型显露出普遍优势，我们提供了关于模型特征的见解。

    Urban mobility is on the cusp of transformation with the emergence of shared, connected, and cooperative automated vehicles. Yet, for them to be accepted by customers, trust in their punctuality is vital. Many pilot initiatives operate without a fixed schedule, thus enhancing the importance of reliable arrival time (AT) predictions. This study presents an AT prediction system for autonomous shuttles, utilizing separate models for dwell and running time predictions, validated on real-world data from five cities. Alongside established methods such as XGBoost, we explore the benefits of integrating spatial data using graph neural networks (GNN). To accurately handle the case of a shuttle bypassing a stop, we propose a hierarchical model combining a random forest classifier and a GNN. The results for the final AT prediction are promising, showing low errors even when predicting several stops ahead. Yet, no single model emerges as universally superior, and we provide insights into the chara
    
[^4]: 时间变换器：融合本地和全局特征以实现更好的时间序列生成

    Time-Transformer: Integrating Local and Global Features for Better Time Series Generation. (arXiv:2312.11714v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.11714](http://arxiv.org/abs/2312.11714)

    本文提出了一种新的时间序列生成模型，通过时间变换器同时学习本地和全局特征，实现了对时间序列数据的更好生成能力。

    

    生成时间序列数据是解决数据不足问题的一种有前景的方法。然而，由于时间序列数据的复杂时间特性，包括本地相关性和全局依赖性，使其成为具有挑战性的任务。大多数现有的生成模型未能有效学习时间序列数据的本地和全局特性。为了解决这个问题，我们提出了一种新颖的时间序列生成模型，命名为'时间变换器AAE'，它由一个对抗性自动编码器（AAE）和一个名为'时间变换器'的新设计架构组成。时间变换器首先通过层次并行设计同时学习本地和全局特征，结合了时间卷积网络和Transformer的能力，分别提取本地特征和全局依赖性。其次，提出了一个双向交叉注意力来在两个分支之间提供互补的引导，并实现本地特征和全局特征的合适融合。

    Generating time series data is a promising approach to address data deficiency problems. However, it is also challenging due to the complex temporal properties of time series data, including local correlations as well as global dependencies. Most existing generative models have failed to effectively learn both the local and global properties of time series data. To address this open problem, we propose a novel time series generative model named 'Time-Transformer AAE', which consists of an adversarial autoencoder (AAE) and a newly designed architecture named 'Time-Transformer' within the decoder. The Time-Transformer first simultaneously learns local and global features in a layer-wise parallel design, combining the abilities of Temporal Convolutional Networks and Transformer in extracting local features and global dependencies respectively. Second, a bidirectional cross attention is proposed to provide complementary guidance across the two branches and achieve proper fusion between loc
    
[^5]: 一种基于机器学习的概率暴露模型的德国高分辨率室内氡气地图

    A new high-resolution indoor radon map for Germany using a machine learning based probabilistic exposure model. (arXiv:2310.11143v1 [stat.ML])

    [http://arxiv.org/abs/2310.11143](http://arxiv.org/abs/2310.11143)

    本研究提出了一种基于机器学习的概率暴露模型，可以更准确地估计德国室内氡气分布，并具有更高的空间分辨率。

    

    室内氡气是一种致癌的放射性气体，可以在室内积累。通常情况下，全国范围内的室内氡暴露是基于广泛的测量活动估计得来的。然而，样本的特征往往与人口特征不同，这是由于许多相关因素，如地质源氡气的可用性或楼层水平。此外，样本大小通常不允许以高空间分辨率进行暴露估计。我们提出了一种基于模型的方法，可以比纯数据方法更加现实地估计室内氡分布，并具有更高的空间分辨率。我们采用了两阶段建模方法：1）应用分位数回归森林，使用环境和建筑数据作为预测因子，估计了德国每个住宅楼的每个楼层的室内氡概率分布函数；2）使用概率蒙特卡罗抽样技术使它们组合和。

    Radon is a carcinogenic, radioactive gas that can accumulate indoors. Indoor radon exposure at the national scale is usually estimated on the basis of extensive measurement campaigns. However, characteristics of the sample often differ from the characteristics of the population due to the large number of relevant factors such as the availability of geogenic radon or floor level. Furthermore, the sample size usually does not allow exposure estimation with high spatial resolution. We propose a model-based approach that allows a more realistic estimation of indoor radon distribution with a higher spatial resolution than a purely data-based approach. We applied a two-stage modelling approach: 1) a quantile regression forest using environmental and building data as predictors was applied to estimate the probability distribution function of indoor radon for each floor level of each residential building in Germany; (2) a probabilistic Monte Carlo sampling technique enabled the combination and
    

