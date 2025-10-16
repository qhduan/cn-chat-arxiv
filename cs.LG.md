# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PPA-Game: Characterizing and Learning Competitive Dynamics Among Online Content Creators](https://arxiv.org/abs/2403.15524) | 引入了PPA-Game模型来表征类似YouTube和TikTok平台上的内容创作者之间竞争动态，分析显示纯纳什均衡在大多数情况下是常见的，提出了一种在线算法用于最大化每个代理者的累积收益。 |
| [^2] | [Offline Fictitious Self-Play for Competitive Games](https://arxiv.org/abs/2403.00841) | 本文介绍了Off-FSP，这是竞争游戏的第一个实用的无模型离线RL算法，通过调整固定数据集的权重，使用重要性抽样，模拟与各种对手的互动。 |
| [^3] | [Optimistic Multi-Agent Policy Gradient for Cooperative Tasks.](http://arxiv.org/abs/2311.01953) | 本论文提出了一个通用的、简单的框架，在多智能体策略梯度方法中引入乐观更新，以缓解合作任务中的相对过度概括问题。通过使用一个泄漏化线性整流函数来重塑优势，我们的方法能够保持对潜在由其他代理引起的低回报个别动作的乐观态度。 |
| [^4] | [Zero-shot Task Preference Addressing Enabled by Imprecise Bayesian Continual Learning.](http://arxiv.org/abs/2305.14782) | 提出了零样本任务偏好的不精确贝叶斯继续学习（IBCL）算法，该算法更新模型参数分布凸壳形式的知识库，并使用零样本获取模型以满足不同的偏好，使得在具有大量任务偏好的情况下更加可扩展。 |
| [^5] | [HyFL: A Hybrid Framework For Private Federated Learning.](http://arxiv.org/abs/2302.09904) | HyFL是一种混合框架，它结合了安全多方计算技术和分层联合学习，并能够在分布式环境中保证数据和全局模型的隐私安全，有助于大规模部署。 |

# 详细

[^1]: PPA-Game：表征和学习在线内容创作者之间的竞争动态

    PPA-Game: Characterizing and Learning Competitive Dynamics Among Online Content Creators

    [https://arxiv.org/abs/2403.15524](https://arxiv.org/abs/2403.15524)

    引入了PPA-Game模型来表征类似YouTube和TikTok平台上的内容创作者之间竞争动态，分析显示纯纳什均衡在大多数情况下是常见的，提出了一种在线算法用于最大化每个代理者的累积收益。

    

    我们引入了比例性收益分配游戏（PPA-Game）来模拟代理者如何竞争可分配资源和消费者的注意力，类似于YouTube和TikTok等平台上的内容创作者。根据异质权重为代理者分配收益，反映了创作者之间内容质量的多样性。我们的分析表明，纯纳什均衡（PNE）并不在每种情况下都有保证，但在我们的模拟中，通常会观察到，其缺乏情况是罕见的。除了分析静态收益外，我们进一步讨论了代理者关于资源收益的在线学习，将多玩家多臂老虎机框架整合在一起。我们提出了一种在线算法，在$T$轮中促进每个代理者累积收益的最大化。从理论上讲，我们建立了任何代理者的遗憾在任何$\eta > 0$下都受到$O(\log^{1 + \eta} T)$的限制。经验结果进一步验证了我们的算法的有效性。

    arXiv:2403.15524v1 Announce Type: cross  Abstract: We introduce the Proportional Payoff Allocation Game (PPA-Game) to model how agents, akin to content creators on platforms like YouTube and TikTok, compete for divisible resources and consumers' attention. Payoffs are allocated to agents based on heterogeneous weights, reflecting the diversity in content quality among creators. Our analysis reveals that although a pure Nash equilibrium (PNE) is not guaranteed in every scenario, it is commonly observed, with its absence being rare in our simulations. Beyond analyzing static payoffs, we further discuss the agents' online learning about resource payoffs by integrating a multi-player multi-armed bandit framework. We propose an online algorithm facilitating each agent's maximization of cumulative payoffs over $T$ rounds. Theoretically, we establish that the regret of any agent is bounded by $O(\log^{1 + \eta} T)$ for any $\eta > 0$. Empirical results further validate the effectiveness of ou
    
[^2]: 竞争游戏的离线虚构自我对弈

    Offline Fictitious Self-Play for Competitive Games

    [https://arxiv.org/abs/2403.00841](https://arxiv.org/abs/2403.00841)

    本文介绍了Off-FSP，这是竞争游戏的第一个实用的无模型离线RL算法，通过调整固定数据集的权重，使用重要性抽样，模拟与各种对手的互动。

    

    离线强化学习（RL）因其在以前收集的数据集中改进策略而不需要在线交互的能力而受到重视。尽管在单一智能体设置中取得成功，但离线多智能体RL仍然是一个挑战，特别是在竞争游戏中。为了解决这些问题，本文介绍了Off-FSP，这是竞争游戏的第一个实用的无模型离线RL算法。我们首先通过调整固定数据集的权重，使用重要性抽样模拟与各种对手的互动。

    arXiv:2403.00841v1 Announce Type: cross  Abstract: Offline Reinforcement Learning (RL) has received significant interest due to its ability to improve policies in previously collected datasets without online interactions. Despite its success in the single-agent setting, offline multi-agent RL remains a challenge, especially in competitive games. Firstly, unaware of the game structure, it is impossible to interact with the opponents and conduct a major learning paradigm, self-play, for competitive games. Secondly, real-world datasets cannot cover all the state and action space in the game, resulting in barriers to identifying Nash equilibrium (NE). To address these issues, this paper introduces Off-FSP, the first practical model-free offline RL algorithm for competitive games. We start by simulating interactions with various opponents by adjusting the weights of the fixed dataset with importance sampling. This technique allows us to learn best responses to different opponents and employ
    
[^3]: 乐观的多智能体策略梯度在合作任务中的应用

    Optimistic Multi-Agent Policy Gradient for Cooperative Tasks. (arXiv:2311.01953v1 [cs.LG])

    [http://arxiv.org/abs/2311.01953](http://arxiv.org/abs/2311.01953)

    本论文提出了一个通用的、简单的框架，在多智能体策略梯度方法中引入乐观更新，以缓解合作任务中的相对过度概括问题。通过使用一个泄漏化线性整流函数来重塑优势，我们的方法能够保持对潜在由其他代理引起的低回报个别动作的乐观态度。

    

    在合作多智能体学习任务中，由于过拟合其他智能体的次优行为，导致智能体收敛到次优联合策略，出现了相对过度概括（RO）问题。早期研究表明，乐观主义可以缓解使用表格化Q学习时的RO问题。然而，对于复杂任务来说，利用函数逼近乐观主义可能加剧过估计，从而失败。另一方面，最近的深度多智能体策略梯度（MAPG）方法在许多复杂任务上取得了成功，但在严重的RO情况下可能失败。我们提出了一个通用而简单的框架，以在MAPG方法中实现乐观更新并缓解RO问题。具体而言，我们使用一个泄漏化线性整流函数，其中一个超参数选择乐观程度以在更新策略时重新塑造优势。直观地说，我们的方法对可能由其他代理引起的回报较低的个别动作保持乐观态度。

    \textit{Relative overgeneralization} (RO) occurs in cooperative multi-agent learning tasks when agents converge towards a suboptimal joint policy due to overfitting to suboptimal behavior of other agents. In early work, optimism has been shown to mitigate the \textit{RO} problem when using tabular Q-learning. However, with function approximation optimism can amplify overestimation and thus fail on complex tasks. On the other hand, recent deep multi-agent policy gradient (MAPG) methods have succeeded in many complex tasks but may fail with severe \textit{RO}. We propose a general, yet simple, framework to enable optimistic updates in MAPG methods and alleviate the RO problem. Specifically, we employ a \textit{Leaky ReLU} function where a single hyperparameter selects the degree of optimism to reshape the advantages when updating the policy. Intuitively, our method remains optimistic toward individual actions with lower returns which are potentially caused by other agents' sub-optimal be
    
[^4]: 零样本任务偏好的不精确贝叶斯继续学习

    Zero-shot Task Preference Addressing Enabled by Imprecise Bayesian Continual Learning. (arXiv:2305.14782v1 [cs.LG])

    [http://arxiv.org/abs/2305.14782](http://arxiv.org/abs/2305.14782)

    提出了零样本任务偏好的不精确贝叶斯继续学习（IBCL）算法，该算法更新模型参数分布凸壳形式的知识库，并使用零样本获取模型以满足不同的偏好，使得在具有大量任务偏好的情况下更加可扩展。

    

    类似于通用的多任务学习，继续学习也具有多目标优化的特性，因此需要在不同任务的性能之间进行平衡。也就是说，为了优化当前任务分布，可能需要在一些任务上牺牲性能以提高其他任务的性能。这意味着存在多个模型，每个模型在不同的时间都是最优的，每个模型都能够解决不同的任务-性能权衡问题。研究人员已经讨论如何训练特定的模型以满足交易偏好。然而，现有的算法需要额外的采样开销-在存在多个，可能是无限数量的偏好时会产生很大的负担。因此，我们提出了不精确贝叶斯继续学习（IBCL）。一旦有新任务，IBCL会（1）更新一个以模型参数分布凸壳形式存在的知识库，（2）并使用零样本获取特定模型以满足不同的偏好。也就是说，IBCL不需要任何额外的数据就能为一个特定的任务偏好生成新的模型，使得在具有大量任务偏好的情况下更加可扩展。

    Like generic multi-task learning, continual learning has the nature of multi-objective optimization, and therefore faces a trade-off between the performance of different tasks. That is, to optimize for the current task distribution, it may need to compromise performance on some tasks to improve on others. This means there exist multiple models that are each optimal at different times, each addressing a distinct task-performance trade-off. Researchers have discussed how to train particular models to address specific preferences on these trade-offs. However, existing algorithms require additional sample overheads -- a large burden when there are multiple, possibly infinitely many, preferences. As a response, we propose Imprecise Bayesian Continual Learning (IBCL). Upon a new task, IBCL (1) updates a knowledge base in the form of a convex hull of model parameter distributions and (2) obtains particular models to address preferences with zero-shot. That is, IBCL does not require any additi
    
[^5]: HyFL:一种用于私有联合学习的混合框架

    HyFL: A Hybrid Framework For Private Federated Learning. (arXiv:2302.09904v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.09904](http://arxiv.org/abs/2302.09904)

    HyFL是一种混合框架，它结合了安全多方计算技术和分层联合学习，并能够在分布式环境中保证数据和全局模型的隐私安全，有助于大规模部署。

    

    联合学习已经成为分布式机器学习的有效方法，通过在客户端设备上保留训练数据来确保数据隐私。然而，最近的研究强调了FL中的漏洞，包括通过单个模型更新甚至整个全局模型泄漏敏感信息。虽然关注点已放在客户端数据隐私上，但有限的研究解决了全局模型隐私问题。此外，客户端本地训练为恶意客户端启动强大的模型污染攻击开辟了途径。不幸的是，目前没有现有工作提供全面解决所有这些问题的解决方案。因此，我们介绍了HyFL，这是一种混合框架，可实现数据和全局模型隐私，并促进大规模部署。HyFL的基础是安全多方计算技术和分层联合学习的独特组合。在HyFL的训练过程中，客户端模型在多个抽象层次上进行安全聚合，以在分布式环境中提供隐私保护。实验证明，HyFL在大规模数据集上实现良好性能，同时确保客户端数据和全局模型的强大隐私和安全保障。

    Federated learning (FL) has emerged as an efficient approach for large-scale distributed machine learning, ensuring data privacy by keeping training data on client devices. However, recent research has highlighted vulnerabilities in FL, including the potential disclosure of sensitive information through individual model updates and even the aggregated global model. While much attention has been given to clients' data privacy, limited research has addressed the issue of global model privacy. Furthermore, local training at the client's side has opened avenues for malicious clients to launch powerful model poisoning attacks. Unfortunately, no existing work has provided a comprehensive solution that tackles all these issues. Therefore, we introduce HyFL, a hybrid framework that enables data and global model privacy while facilitating large-scale deployments. The foundation of HyFL is a unique combination of secure multi-party computation (MPC) techniques with hierarchical federated learnin
    

