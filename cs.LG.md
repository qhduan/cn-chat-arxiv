# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Goal-Conditioned Offline Reinforcement Learning via Metric Learning](https://arxiv.org/abs/2402.10820) | 通过度量学习的目标条件离线强化学习提出了一种新的方法来处理稀疏奖励、对称和确定性动作下的最优值函数近似，并展示了在学习次优离线数据集方面的显着优越性。 |
| [^2] | [Neural Rank Collapse: Weight Decay and Small Within-Class Variability Yield Low-Rank Bias](https://arxiv.org/abs/2402.03991) | 神经网络中的权重衰减和小的类内变化与低秩偏差现象有关 |
| [^3] | [Learning-based agricultural management in partially observable environments subject to climate variability.](http://arxiv.org/abs/2401.01273) | 本研究引入了一种将深度强化学习与循环神经网络相结合的创新框架，利用Gym-DSSAT模拟器训练智能agent来掌握最佳氮肥管理策略。研究强调了利用序列观测开发更高效氮肥输入策略的优势，并探讨了气候变异对农业管理的影响。 |
| [^4] | [Multi-modal Gaussian Process Variational Autoencoders for Neural and Behavioral Data.](http://arxiv.org/abs/2310.03111) | 该论文介绍了一种多模态高斯过程变分自编码器（GP-VAEs）的方法，用于描述神经和行为数据之间的关系。该方法通过将高斯过程因子分析（GPFA）和深度神经网络相结合，能够提取不同实验模态的共享和独立潜变量，并且具有解释能力。 |
| [^5] | [Optimal Estimates for Pairwise Learning with Deep ReLU Networks.](http://arxiv.org/abs/2305.19640) | 本文研究了深度ReLU网络中的成对学习，提出了一个针对一般损失函数的误差估计的尖锐界限，并基于成对最小二乘损失得出几乎最优的过度泛化误差界限。 |

# 详细

[^1]: 通过度量学习的目标条件离线强化学习

    Goal-Conditioned Offline Reinforcement Learning via Metric Learning

    [https://arxiv.org/abs/2402.10820](https://arxiv.org/abs/2402.10820)

    通过度量学习的目标条件离线强化学习提出了一种新的方法来处理稀疏奖励、对称和确定性动作下的最优值函数近似，并展示了在学习次优离线数据集方面的显着优越性。

    

    在这项工作中，我们解决了在目标条件下离线强化学习中从次优数据集中学习最优行为的问题。为此，我们提出了一种新颖的方法来近似处理稀疏奖励、对称且确定性动作下的目标条件离线RL问题的最优值函数。我们研究了一种表示恢复优化的属性，并提出了导致该属性的新优化目标。我们使用学习到的值函数以演员-评论者的方式指导策略的学习，这种方法被我们称为MetricRL。在实验中，我们展示了我们的方法如何始终优于其他离线RL基线在从次优离线数据集中学习方面的表现。此外，我们展示了我们的方法在处理高维观测和多目标任务中的有效性。

    arXiv:2402.10820v1 Announce Type: new  Abstract: In this work, we address the problem of learning optimal behavior from sub-optimal datasets in the context of goal-conditioned offline reinforcement learning. To do so, we propose a novel way of approximating the optimal value function for goal-conditioned offline RL problems under sparse rewards, symmetric and deterministic actions. We study a property for representations to recover optimality and propose a new optimization objective that leads to such property. We use the learned value function to guide the learning of a policy in an actor-critic fashion, a method we name MetricRL. Experimentally, we show how our method consistently outperforms other offline RL baselines in learning from sub-optimal offline datasets. Moreover, we show the effectiveness of our method in dealing with high-dimensional observations and in multi-goal tasks.
    
[^2]: 神经网络的权重衰减和类内变化小会导致低秩偏差

    Neural Rank Collapse: Weight Decay and Small Within-Class Variability Yield Low-Rank Bias

    [https://arxiv.org/abs/2402.03991](https://arxiv.org/abs/2402.03991)

    神经网络中的权重衰减和小的类内变化与低秩偏差现象有关

    

    近期在深度学习领域的研究显示了一个隐含的低秩偏差现象：深度网络中的权重矩阵往往近似为低秩，在训练过程中或从已经训练好的模型中去除相对较小的奇异值可以显著减小模型大小，同时保持甚至提升模型性能。然而，大多数关于神经网络低秩偏差的理论研究都涉及到简化的线性深度网络。在本文中，我们考虑了带有非线性激活函数和权重衰减参数的通用网络，并展示了一个有趣的神经秩崩溃现象，它将训练好的网络的低秩偏差与网络的神经崩溃特性联系起来：随着权重衰减参数的增加，网络中每一层的秩呈比例递减，与前面层的隐藏空间嵌入的类内变化成反比。我们的理论发现得到了支持。

    Recent work in deep learning has shown strong empirical and theoretical evidence of an implicit low-rank bias: weight matrices in deep networks tend to be approximately low-rank and removing relatively small singular values during training or from available trained models may significantly reduce model size while maintaining or even improving model performance. However, the majority of the theoretical investigations around low-rank bias in neural networks deal with oversimplified deep linear networks. In this work, we consider general networks with nonlinear activations and the weight decay parameter, and we show the presence of an intriguing neural rank collapse phenomenon, connecting the low-rank bias of trained networks with networks' neural collapse properties: as the weight decay parameter grows, the rank of each layer in the network decreases proportionally to the within-class variability of the hidden-space embeddings of the previous layers. Our theoretical findings are supporte
    
[^3]: 学习基于环境气候变异的部分可观测农业管理

    Learning-based agricultural management in partially observable environments subject to climate variability. (arXiv:2401.01273v1 [cs.LG])

    [http://arxiv.org/abs/2401.01273](http://arxiv.org/abs/2401.01273)

    本研究引入了一种将深度强化学习与循环神经网络相结合的创新框架，利用Gym-DSSAT模拟器训练智能agent来掌握最佳氮肥管理策略。研究强调了利用序列观测开发更高效氮肥输入策略的优势，并探讨了气候变异对农业管理的影响。

    

    农业管理在塑造作物产量、经济可盈利性和环境可持续性方面扮演着重要角色，特别关注施肥策略。然而，当面对极端天气条件（如热浪和干旱）时，传统指导方针的有效性减弱。本研究引入了一种创新框架，将深度强化学习（DRL）与循环神经网络（RNNs）相结合。利用Gym-DSSAT模拟器，我们训练了一个智能agent来掌握最佳氮肥管理。通过在爱荷华州玉米农作物上进行一系列模拟实验，我们比较了部分可观测马尔科夫决策过程（POMDP）模型和马尔科夫决策过程（MDP）模型。我们的研究强调了利用序列观测来开发更高效的氮肥输入策略的优势。此外，我们还探讨了气候的变异性对农业管理的影响。

    Agricultural management, with a particular focus on fertilization strategies, holds a central role in shaping crop yield, economic profitability, and environmental sustainability. While conventional guidelines offer valuable insights, their efficacy diminishes when confronted with extreme weather conditions, such as heatwaves and droughts. In this study, we introduce an innovative framework that integrates Deep Reinforcement Learning (DRL) with Recurrent Neural Networks (RNNs). Leveraging the Gym-DSSAT simulator, we train an intelligent agent to master optimal nitrogen fertilization management. Through a series of simulation experiments conducted on corn crops in Iowa, we compare Partially Observable Markov Decision Process (POMDP) models with Markov Decision Process (MDP) models. Our research underscores the advantages of utilizing sequential observations in developing more efficient nitrogen input policies. Additionally, we explore the impact of climate variability, particularly duri
    
[^4]: 多模态高斯过程变分自编码器用于神经和行为数据

    Multi-modal Gaussian Process Variational Autoencoders for Neural and Behavioral Data. (arXiv:2310.03111v1 [cs.LG])

    [http://arxiv.org/abs/2310.03111](http://arxiv.org/abs/2310.03111)

    该论文介绍了一种多模态高斯过程变分自编码器（GP-VAEs）的方法，用于描述神经和行为数据之间的关系。该方法通过将高斯过程因子分析（GPFA）和深度神经网络相结合，能够提取不同实验模态的共享和独立潜变量，并且具有解释能力。

    

    描述神经群体活动和行为数据之间的关系是神经科学的核心目标。虽然潜在变量模型（LVMs）在描述高维时间序列数据方面取得了成功，但它们通常只用于单一类型的数据，这使得难以识别不同实验数据模态之间的共享结构。在这里，我们通过提出一种无监督的LVM来解决这个缺点，该模型提取了不同、同时记录的实验模态的时间演化共享和独立潜变量。我们通过将高斯过程因子分析（GPFA），一种解释性的用于神经尖峰数据的LVM，并具有时间平滑潜空间，与高斯过程变分自编码器（GP-VAEs）相结合来实现这一点，GP-VAEs同样使用高斯先验来描述潜空间中的相关性，但由于深度神经网络映射到观测值具有丰富的表达能力。我们通过将潜变量分区来实现模型的可解释性。

    Characterizing the relationship between neural population activity and behavioral data is a central goal of neuroscience. While latent variable models (LVMs) are successful in describing high-dimensional time-series data, they are typically only designed for a single type of data, making it difficult to identify structure shared across different experimental data modalities. Here, we address this shortcoming by proposing an unsupervised LVM which extracts temporally evolving shared and independent latents for distinct, simultaneously recorded experimental modalities. We do this by combining Gaussian Process Factor Analysis (GPFA), an interpretable LVM for neural spiking data with temporally smooth latent space, with Gaussian Process Variational Autoencoders (GP-VAEs), which similarly use a GP prior to characterize correlations in a latent space, but admit rich expressivity due to a deep neural network mapping to observations. We achieve interpretability in our model by partitioning lat
    
[^5]: 深度ReLU网络中的成对学习最优估计

    Optimal Estimates for Pairwise Learning with Deep ReLU Networks. (arXiv:2305.19640v1 [stat.ML])

    [http://arxiv.org/abs/2305.19640](http://arxiv.org/abs/2305.19640)

    本文研究了深度ReLU网络中的成对学习，提出了一个针对一般损失函数的误差估计的尖锐界限，并基于成对最小二乘损失得出几乎最优的过度泛化误差界限。

    

    成对学习指的是在损失函数中考虑一对样本的学习任务。本文研究了深度ReLU网络中的成对学习，并估计了过度泛化误差。对于满足某些温和条件的一般损失函数，建立了误差估计的尖锐界限，其误差估计的阶数为O（（Vlog（n）/ n）1 /（2-β））。特别地，对于成对最小二乘损失，我们得到了过度泛化误差的几乎最优界限，在真实的预测器满足某些光滑性正则性时，最优界限达到了最小化界限，差距仅为对数项。

    Pairwise learning refers to learning tasks where a loss takes a pair of samples into consideration. In this paper, we study pairwise learning with deep ReLU networks and estimate the excess generalization error. For a general loss satisfying some mild conditions, a sharp bound for the estimation error of order $O((V\log(n) /n)^{1/(2-\beta)})$ is established. In particular, with the pairwise least squares loss, we derive a nearly optimal bound of the excess generalization error which achieves the minimax lower bound up to a logrithmic term when the true predictor satisfies some smoothness regularities.
    

