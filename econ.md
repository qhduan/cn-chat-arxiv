# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Three-Layer Data Markets](https://arxiv.org/abs/2402.09697) | 这篇论文研究了一个三层数据市场，其中用户通过与平台共享数据来获得服务，但会产生隐私损失。研究发现，当购买方对用户数据的价值较高时，所有平台都向用户提供服务，而当购买方对用户数据的价值较低时，只有低成本的大型平台能够服务用户。 |
| [^2] | [Market Design for Dynamic Pricing and Pooling in Capacitated Networks.](http://arxiv.org/abs/2307.03994) | 本研究提出了一种用于动态定价和汇聚网络的市场设计，通过设置边缘价格激励代理商共享有限网络容量。在考虑了整数和网络约束以及代理商异质偏好的情况下，我们提供了充分条件，保证市场均衡的存在和多项式时间计算，并识别了实现最大效用的特定市场均衡。 |

# 详细

[^1]: 关于三层数据市场的研究

    On Three-Layer Data Markets

    [https://arxiv.org/abs/2402.09697](https://arxiv.org/abs/2402.09697)

    这篇论文研究了一个三层数据市场，其中用户通过与平台共享数据来获得服务，但会产生隐私损失。研究发现，当购买方对用户数据的价值较高时，所有平台都向用户提供服务，而当购买方对用户数据的价值较低时，只有低成本的大型平台能够服务用户。

    

    我们研究了一个由用户（数据所有者）、平台和数据购买方组成的三层数据市场。每个用户通过共享数据获得平台服务，当他们的数据与购买方共享时，会发生隐私损失，尽管以噪声形式共享。用户选择与哪些平台分享数据，而平台在出售给购买方之前决定数据的噪声水平和定价。购买方选择从哪些平台购买数据。我们通过多阶段博弈模型来描述这些互动，重点研究子博弈纳什均衡。我们发现，当购买方对用户数据的价值较高（平台能够获取高价格）时，所有平台都向用户提供服务，并且用户加入并与每个平台分享数据。相反，当购买方对用户数据的估值较低时，只有低成本的大型平台能够为用户提供服务。在这种情况下，用户只加入并与这些低成本平台分享数据。有趣的是，增加平台数量会增加用户参与的意愿，但也会提高平台之间的竞争。

    arXiv:2402.09697v1 Announce Type: new  Abstract: We study a three-layer data market comprising users (data owners), platforms, and a data buyer. Each user benefits from platform services in exchange for data, incurring privacy loss when their data, albeit noisily, is shared with the buyer. The user chooses platforms to share data with, while platforms decide on data noise levels and pricing before selling to the buyer. The buyer selects platforms to purchase data from. We model these interactions via a multi-stage game, focusing on the subgame Nash equilibrium. We find that when the buyer places a high value on user data (and platforms can command high prices), all platforms offer services to the user who joins and shares data with every platform. Conversely, when the buyer's valuation of user data is low, only large platforms with low service costs can afford to serve users. In this scenario, users exclusively join and share data with these low-cost platforms. Interestingly, increased
    
[^2]: 动态定价和汇聚网络的市场设计

    Market Design for Dynamic Pricing and Pooling in Capacitated Networks. (arXiv:2307.03994v1 [cs.GT])

    [http://arxiv.org/abs/2307.03994](http://arxiv.org/abs/2307.03994)

    本研究提出了一种用于动态定价和汇聚网络的市场设计，通过设置边缘价格激励代理商共享有限网络容量。在考虑了整数和网络约束以及代理商异质偏好的情况下，我们提供了充分条件，保证市场均衡的存在和多项式时间计算，并识别了实现最大效用的特定市场均衡。

    

    我们研究了一种市场机制，通过设置边缘价格来激励战略性代理商组织旅行，以有效共享有限的网络容量。该市场允许代理商组成团队共享旅行，做出出发时间和路线选择的决策，并支付边缘价格和其他成本。我们发展了一种新的方法来分析市场均衡的存在和计算，建立在组合拍卖和动态网络流理论的基础上。我们的方法解决了市场均衡特征化中的挑战，包括：（a）共享有限边缘容量中旅行的动态流量所引发的整数和网络约束；（b）战略性代理商的异质和私人偏好。我们提供了关于网络拓扑和代理商偏好的充分条件，以确保市场均衡的存在和多项式时间计算。我们确定了一个特定的市场均衡，实现了所有代理商的最大效用，并且与经典的最大流最小割问题等价。

    We study a market mechanism that sets edge prices to incentivize strategic agents to organize trips that efficiently share limited network capacity. This market allows agents to form groups to share trips, make decisions on departure times and route choices, and make payments to cover edge prices and other costs. We develop a new approach to analyze the existence and computation of market equilibrium, building on theories of combinatorial auctions and dynamic network flows. Our approach tackles the challenges in market equilibrium characterization arising from: (a) integer and network constraints on the dynamic flow of trips in sharing limited edge capacity; (b) heterogeneous and private preferences of strategic agents. We provide sufficient conditions on the network topology and agents' preferences that ensure the existence and polynomial-time computation of market equilibrium. We identify a particular market equilibrium that achieves maximum utilities for all agents, and is equivalen
    

