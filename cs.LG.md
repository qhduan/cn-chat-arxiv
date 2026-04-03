# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Moonwalk: Inverse-Forward Differentiation](https://arxiv.org/abs/2402.14212) | Moonwalk引入了一种基于向量-逆-Jacobian乘积的新技术，加速前向梯度计算，显著减少内存占用，并在保持真实梯度准确性的同时，将计算时间降低了几个数量级。 |
| [^2] | [Strategic Client Selection to Address Non-IIDness in HAPS-enabled FL Networks.](http://arxiv.org/abs/2401.05308) | 该研究介绍了一种针对高空平台站（HAPS）使能的垂直异构网络中数据分布不均问题的战略客户选择策略，通过利用用户的网络流量行为预测和分类，优先选择数据呈现相似模式的客户参与，以提高联合学习（FL）模型的训练效果。 |
| [^3] | [Deep Kalman Filters Can Filter.](http://arxiv.org/abs/2310.19603) | 本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。 |
| [^4] | [Risk-Aware Linear Bandits: Theory and Applications in Smart Order Routing.](http://arxiv.org/abs/2208.02389) | 本论文研究了风险意识线性赌博机在智能订单路由中的应用，并提出了两种算法来最小化遗憾。分析表明，这些算法在近乎最优的情况下能够通过利用线性结构来提高性能。 |

# 详细

[^1]: Moonwalk：逆向-前向微分

    Moonwalk: Inverse-Forward Differentiation

    [https://arxiv.org/abs/2402.14212](https://arxiv.org/abs/2402.14212)

    Moonwalk引入了一种基于向量-逆-Jacobian乘积的新技术，加速前向梯度计算，显著减少内存占用，并在保持真实梯度准确性的同时，将计算时间降低了几个数量级。

    

    反向传播虽然在梯度计算方面有效，但在解决内存消耗和扩展性方面表现不佳。这项工作探索了前向梯度计算作为可逆网络中的一种替代方法，展示了它在减少内存占用的潜力，并不带来重大缺点。我们引入了一种基于向量-逆-Jacobian乘积的新技术，加速了前向梯度的计算，同时保留了减少内存和保持真实梯度准确性的优势。我们的方法Moonwalk在网络深度方面具有线性时间复杂度，与朴素前向的二次时间复杂度相比，在没有分配更多内存的情况下，从实证的角度减少了几个数量级的计算时间。我们进一步通过将Moonwalk与反向模式微分相结合来加速，以实现与反向传播相当的时间复杂度，同时保持更小的内存使用量。

    arXiv:2402.14212v1 Announce Type: cross  Abstract: Backpropagation, while effective for gradient computation, falls short in addressing memory consumption, limiting scalability. This work explores forward-mode gradient computation as an alternative in invertible networks, showing its potential to reduce the memory footprint without substantial drawbacks. We introduce a novel technique based on a vector-inverse-Jacobian product that accelerates the computation of forward gradients while retaining the advantages of memory reduction and preserving the fidelity of true gradients. Our method, Moonwalk, has a time complexity linear in the depth of the network, unlike the quadratic time complexity of na\"ive forward, and empirically reduces computation time by several orders of magnitude without allocating more memory. We further accelerate Moonwalk by combining it with reverse-mode differentiation to achieve time complexity comparable with backpropagation while maintaining a much smaller mem
    
[^2]: 面对HAPS使能的FL网络中的非独立同分布问题，战略客户选择的研究

    Strategic Client Selection to Address Non-IIDness in HAPS-enabled FL Networks. (arXiv:2401.05308v1 [cs.NI])

    [http://arxiv.org/abs/2401.05308](http://arxiv.org/abs/2401.05308)

    该研究介绍了一种针对高空平台站（HAPS）使能的垂直异构网络中数据分布不均问题的战略客户选择策略，通过利用用户的网络流量行为预测和分类，优先选择数据呈现相似模式的客户参与，以提高联合学习（FL）模型的训练效果。

    

    在由高空平台站（HAPS）使能的垂直异构网络中部署联合学习（FL）为各种不同通信和计算能力的客户提供了参与的机会。这种多样性不仅提高了FL模型的训练精度，还加快了其收敛速度。然而，在这些广阔的网络中应用FL存在显著的非独立同分布问题。这种数据异质性往往导致收敛速度较慢和模型训练性能的降低。我们的研究引入了一种针对此问题的客户选择策略，利用用户网络流量行为进行预测和分类。该策略通过战略性选择数据呈现相似模式的客户参与，同时优先考虑用户隐私。

    The deployment of federated learning (FL) within vertical heterogeneous networks, such as those enabled by high-altitude platform station (HAPS), offers the opportunity to engage a wide array of clients, each endowed with distinct communication and computational capabilities. This diversity not only enhances the training accuracy of FL models but also hastens their convergence. Yet, applying FL in these expansive networks presents notable challenges, particularly the significant non-IIDness in client data distributions. Such data heterogeneity often results in slower convergence rates and reduced effectiveness in model training performance. Our study introduces a client selection strategy tailored to address this issue, leveraging user network traffic behaviour. This strategy involves the prediction and classification of clients based on their network usage patterns while prioritizing user privacy. By strategically selecting clients whose data exhibit similar patterns for participation
    
[^3]: 深度卡尔曼滤波器可以进行滤波

    Deep Kalman Filters Can Filter. (arXiv:2310.19603v1 [cs.LG])

    [http://arxiv.org/abs/2310.19603](http://arxiv.org/abs/2310.19603)

    本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。

    

    深度卡尔曼滤波器（DKFs）是一类神经网络模型，可以从序列数据中生成高斯概率测度。虽然DKFs受卡尔曼滤波器的启发，但它们缺乏与随机滤波问题的具体理论关联，从而限制了它们在传统模型基础上的滤波问题的应用，例如数学金融中的债券和期权定价模型校准。我们通过展示一类连续时间DKFs，可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而解决了深度学习数学基础中的这个问题。我们的近似结果在路径的足够规则的紧致子集上一致成立，其中近似误差由在给定紧致路径集上均一地计算的最坏情况2-Wasserstein距离量化。

    Deep Kalman filters (DKFs) are a class of neural network models that generate Gaussian probability measures from sequential data. Though DKFs are inspired by the Kalman filter, they lack concrete theoretical ties to the stochastic filtering problem, thus limiting their applicability to areas where traditional model-based filters have been used, e.g.\ model calibration for bond and option prices in mathematical finance. We address this issue in the mathematical foundations of deep learning by exhibiting a class of continuous-time DKFs which can approximately implement the conditional law of a broad class of non-Markovian and conditionally Gaussian signal processes given noisy continuous-times measurements. Our approximation results hold uniformly over sufficiently regular compact subsets of paths, where the approximation error is quantified by the worst-case 2-Wasserstein distance computed uniformly over the given compact set of paths.
    
[^4]: 风险意识的线性赌博机：理论和在智能订单路由中的应用

    Risk-Aware Linear Bandits: Theory and Applications in Smart Order Routing. (arXiv:2208.02389v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.02389](http://arxiv.org/abs/2208.02389)

    本论文研究了风险意识线性赌博机在智能订单路由中的应用，并提出了两种算法来最小化遗憾。分析表明，这些算法在近乎最优的情况下能够通过利用线性结构来提高性能。

    

    受金融决策中机器学习的实际考虑（如风险厌恶和大型操作空间）的驱动，我们考虑了具有智能订单路由（SOR）应用的风险意识赌博机优化。具体来说，基于从纳斯达克ITCH数据集中对线性价格影响的初步观察，我们开展了风险意识线性赌博机的研究。在这种设置中，我们旨在在面对一组（最初）未知参数的线性函数作为奖励的行动时，通过使用均值方差度量来最小化遗憾，该度量反映了我们的表现与最优解之间的差距。基于方差最小化的全局最优（G-最优）设计，我们提出了独立于实例的全新的风险意识探索-承诺（RISE）算法和依赖于实例的风险意识连续淘汰（RISE++）算法。然后，我们通过严格分析它们近乎最优的遗憾上界，展示了通过利用线性结构，我们的算法的性能。

    Motivated by practical considerations in machine learning for financial decision-making, such as risk aversion and large action space, we consider risk-aware bandits optimization with applications in smart order routing (SOR). Specifically, based on preliminary observations of linear price impacts made from the NASDAQ ITCH dataset, we initiate the study of risk-aware linear bandits. In this setting, we aim at minimizing regret, which measures our performance deficit compared to the optimum's, under the mean-variance metric when facing a set of actions whose rewards are linear functions of (initially) unknown parameters. Driven by the variance-minimizing globally-optimal (G-optimal) design, we propose the novel instance-independent Risk-Aware Explore-then-Commit (RISE) algorithm and the instance-dependent Risk-Aware Successive Elimination (RISE++) algorithm. Then, we rigorously analyze their near-optimal regret upper bounds to show that, by leveraging the linear structure, our algorithm
    

