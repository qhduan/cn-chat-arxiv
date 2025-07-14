# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Analysis of Switchback Designs in Reinforcement Learning](https://arxiv.org/abs/2403.17285) | 本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。 |
| [^2] | [Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation](https://arxiv.org/abs/2403.06759) | 提出一种平均L1校准误差（mL1-ACE）作为辅助损失函数，用于改善图像分割中的像素级校准，减少了校准误差并引入了数据集可靠性直方图以提高校准评估。 |
| [^3] | [Signed Diverse Multiplex Networks: Clustering and Inference](https://arxiv.org/abs/2402.10242) | 保留边的符号在网络构建过程中提高了估计和聚类精度，有助于解决现实世界问题。 |
| [^4] | [Convergence for Natural Policy Gradient on Infinite-State Average-Reward Markov Decision Processes](https://arxiv.org/abs/2402.05274) | 本文针对无穷状态平均奖励马尔可夫决策过程中的自然策略梯度（NPG）算法进行了收敛性分析，证明了在良好的初始策略条件下，该算法能够以$O(1/\sqrt{T})$的收敛速率收敛。同时，对于一类大类排队MDPs，MaxWeight策略足以满足初始策略要求并实现收敛。 |
| [^5] | [Estimation of conditional average treatment effects on distributed data: A privacy-preserving approach](https://arxiv.org/abs/2402.02672) | 本论文提出了一种数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计条件平均治疗效果（CATE）模型。通过数值实验验证了该方法的有效性。该方法的三个主要贡献是：实现了对分布式数据上的非迭代通信的半参数CATE模型的估计和测试，提高了模型的鲁棒性。 |

# 详细

[^1]: 对强化学习中的往返设计进行的分析

    An Analysis of Switchback Designs in Reinforcement Learning

    [https://arxiv.org/abs/2403.17285](https://arxiv.org/abs/2403.17285)

    本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。

    

    本文提供了对A/B测试中往返设计的详细研究，这些设计随时间在基准和新策略之间交替。我们的目标是全面评估这些设计对其产生的平均处理效应（ATE）估计器准确性的影响。我们提出了一个新颖的“弱信号分析”框架，大大简化了这些ATE的均方误差（MSE）在马尔科夫决策过程环境中的计算。我们的研究结果表明：(i) 当大部分奖励误差呈正相关时，往返设计比每日切换策略的交替设计更有效。此外，增加政策切换的频率往往会降低ATE估计器的MSE。(ii) 然而，当误差不相关时，所有这些设计变得渐近等效。(iii) 在大多数误差为负相关时

    arXiv:2403.17285v1 Announce Type: cross  Abstract: This paper offers a detailed investigation of switchback designs in A/B testing, which alternate between baseline and new policies over time. Our aim is to thoroughly evaluate the effects of these designs on the accuracy of their resulting average treatment effect (ATE) estimators. We propose a novel "weak signal analysis" framework, which substantially simplifies the calculations of the mean squared errors (MSEs) of these ATEs in Markov decision process environments. Our findings suggest that (i) when the majority of reward errors are positively correlated, the switchback design is more efficient than the alternating-day design which switches policies in a daily basis. Additionally, increasing the frequency of policy switches tends to reduce the MSE of the ATE estimator. (ii) When the errors are uncorrelated, however, all these designs become asymptotically equivalent. (iii) In cases where the majority of errors are negative correlate
    
[^2]: 平均校准误差：一种可微损失函数，用于改善图像分割中的可靠性

    Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation

    [https://arxiv.org/abs/2403.06759](https://arxiv.org/abs/2403.06759)

    提出一种平均L1校准误差（mL1-ACE）作为辅助损失函数，用于改善图像分割中的像素级校准，减少了校准误差并引入了数据集可靠性直方图以提高校准评估。

    

    医学图像分割的深度神经网络经常产生与经验观察不一致的过于自信的结果，这种校准错误挑战着它们的临床应用。我们提出使用平均L1校准误差（mL1-ACE）作为一种新颖的辅助损失函数，以改善像素级校准而不会损害分割质量。我们展示了，尽管使用硬分箱，这种损失是直接可微的，避免了需要近似但可微的替代或软分箱方法的必要性。我们的工作还引入了数据集可靠性直方图的概念，这一概念推广了标准的可靠性图，用于在数据集级别聚合的语义分割中细化校准的视觉评估。使用mL1-ACE，我们将平均和最大校准误差分别降低了45%和55%，同时在BraTS 2021数据集上保持了87%的Dice分数。我们在这里分享我们的代码: https://github

    arXiv:2403.06759v1 Announce Type: cross  Abstract: Deep neural networks for medical image segmentation often produce overconfident results misaligned with empirical observations. Such miscalibration, challenges their clinical translation. We propose to use marginal L1 average calibration error (mL1-ACE) as a novel auxiliary loss function to improve pixel-wise calibration without compromising segmentation quality. We show that this loss, despite using hard binning, is directly differentiable, bypassing the need for approximate but differentiable surrogate or soft binning approaches. Our work also introduces the concept of dataset reliability histograms which generalises standard reliability diagrams for refined visual assessment of calibration in semantic segmentation aggregated at the dataset level. Using mL1-ACE, we reduce average and maximum calibration error by 45% and 55% respectively, maintaining a Dice score of 87% on the BraTS 2021 dataset. We share our code here: https://github
    
[^3]: 有符号多样化多重网络：聚类和推断

    Signed Diverse Multiplex Networks: Clustering and Inference

    [https://arxiv.org/abs/2402.10242](https://arxiv.org/abs/2402.10242)

    保留边的符号在网络构建过程中提高了估计和聚类精度，有助于解决现实世界问题。

    

    该论文介绍了一种有符号的广义随机点积图（SGRDPG）模型，这是广义随机点积图（GRDPG）的一个变种，其中边可以是正的也可以是负的。该设置被扩展为多重网络版本，其中所有层具有相同的节点集合并遵循SGRDPG。网络层的唯一公共特征是它们可以被划分为具有共同子空间结构的组，而其他情况下所有连接概率矩阵可能是完全不同的。上述设置非常灵活，并包括各种现有多重网络模型作为其特例。论文实现了两个目标。首先，它表明在网络构建过程中保留边的符号会导致更好的估计和聚类精度，因此有助于应对诸如大脑网络分析之类的现实问题。

    arXiv:2402.10242v1 Announce Type: cross  Abstract: The paper introduces a Signed Generalized Random Dot Product Graph (SGRDPG) model, which is a variant of the Generalized Random Dot Product Graph (GRDPG), where, in addition, edges can be positive or negative. The setting is extended to a multiplex version, where all layers have the same collection of nodes and follow the SGRDPG. The only common feature of the layers of the network is that they can be partitioned into groups with common subspace structures, while otherwise all matrices of connection probabilities can be all different. The setting above is extremely flexible and includes a variety of existing multiplex network models as its particular cases. The paper fulfills two objectives. First, it shows that keeping signs of the edges in the process of network construction leads to a better precision of estimation and clustering and, hence, is beneficial for tackling real world problems such as analysis of brain networks. Second, b
    
[^4]: 无穷状态平均奖励马尔可夫决策过程中的自然策略梯度收敛性

    Convergence for Natural Policy Gradient on Infinite-State Average-Reward Markov Decision Processes

    [https://arxiv.org/abs/2402.05274](https://arxiv.org/abs/2402.05274)

    本文针对无穷状态平均奖励马尔可夫决策过程中的自然策略梯度（NPG）算法进行了收敛性分析，证明了在良好的初始策略条件下，该算法能够以$O(1/\sqrt{T})$的收敛速率收敛。同时，对于一类大类排队MDPs，MaxWeight策略足以满足初始策略要求并实现收敛。

    

    无穷状态马尔可夫决策过程（MDPs）在建模和优化各种工程问题中起着重要作用。在强化学习（RL）环境中，已经开发了各种算法来学习和优化这些MDPs。在许多流行的基于策略梯度的学习算法中，如自然演员-评论家、TRPO和PPO，都基于自然策略梯度（NPG）算法。这些RL算法的收敛结果建立在NPG算法的收敛结果上。然而，所有现有的NPG算法收敛性结果均仅限于有限状态设置。我们证明了NPG算法在无穷状态平均奖励MDPs中的首个收敛速率界限，证明了$O(1/\sqrt{T})$的收敛速率，如果NPG算法以良好的初始策略进行初始化。此外，我们还展示了在大类排队MDPs的情况下，MaxWeight策略足够满足我们的初始策略要求，并实现了$O(1/...

    Infinite-state Markov Decision Processes (MDPs) are essential in modeling and optimizing a wide variety of engineering problems. In the reinforcement learning (RL) context, a variety of algorithms have been developed to learn and optimize these MDPs. At the heart of many popular policy-gradient based learning algorithms, such as natural actor-critic, TRPO, and PPO, lies the Natural Policy Gradient (NPG) algorithm. Convergence results for these RL algorithms rest on convergence results for the NPG algorithm. However, all existing results on the convergence of the NPG algorithm are limited to finite-state settings.   We prove the first convergence rate bound for the NPG algorithm for infinite-state average-reward MDPs, proving a $O(1/\sqrt{T})$ convergence rate, if the NPG algorithm is initialized with a good initial policy. Moreover, we show that in the context of a large class of queueing MDPs, the MaxWeight policy suffices to satisfy our initial-policy requirement and achieve a $O(1/\
    
[^5]: 对分布式数据的条件平均治疗效果估计：一种保护隐私的方法

    Estimation of conditional average treatment effects on distributed data: A privacy-preserving approach

    [https://arxiv.org/abs/2402.02672](https://arxiv.org/abs/2402.02672)

    本论文提出了一种数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计条件平均治疗效果（CATE）模型。通过数值实验验证了该方法的有效性。该方法的三个主要贡献是：实现了对分布式数据上的非迭代通信的半参数CATE模型的估计和测试，提高了模型的鲁棒性。

    

    在医学和社会科学等各个领域中，对条件平均治疗效果（CATEs）的估计是一个重要的课题。如果分布在多个参与方之间的数据可以集中，可以对CATEs进行高精度的估计。然而，如果这些数据包含隐私信息，则很难进行数据聚合。为了解决这个问题，我们提出了数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计CATE模型，并通过数值实验对该方法进行了评估。我们的贡献总结如下三点。首先，我们的方法能够在分布式数据上进行非迭代通信的半参数CATE模型的估计和测试。半参数或非参数的CATE模型能够比参数模型更稳健地进行估计和测试，对于模型偏差的鲁棒性更强。然而，据我们所知，目前还没有提出有效的通信方法来估计和测试这些模型。

    Estimation of conditional average treatment effects (CATEs) is an important topic in various fields such as medical and social sciences. CATEs can be estimated with high accuracy if distributed data across multiple parties can be centralized. However, it is difficult to aggregate such data if they contain privacy information. To address this issue, we proposed data collaboration double machine learning (DC-DML), a method that can estimate CATE models with privacy preservation of distributed data, and evaluated the method through numerical experiments. Our contributions are summarized in the following three points. First, our method enables estimation and testing of semi-parametric CATE models without iterative communication on distributed data. Semi-parametric or non-parametric CATE models enable estimation and testing that is more robust to model mis-specification than parametric models. However, to our knowledge, no communication-efficient method has been proposed for estimating and 
    

