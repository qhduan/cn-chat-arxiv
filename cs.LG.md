# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm](https://arxiv.org/abs/2403.05666) | 通过对ICP算法进行基于深度学习的攻击，在安全关键应用中评估其鲁棒性，重点在于找到可能的最大ICP姿势误差。 |
| [^2] | [Closed-Loop Unsupervised Representation Disentanglement with $\beta$-VAE Distillation and Diffusion Probabilistic Feedback](https://arxiv.org/abs/2402.02346) | 本文提出了闭环无监督表示解缠方法CL-Dis，使用扩散自动编码器（Diff-AE）和β-VAE共同提取语义解缠表示，以解决表示解缠面临的问题。 |
| [^3] | [Leveraging Public Representations for Private Transfer Learning.](http://arxiv.org/abs/2312.15551) | 该论文探讨了如何利用公共数据来改进私有学习的问题。研究发现，通过学习公共数据中的共享表示，可以在两种迁移学习场景中实现最优的学习效果。在单任务迁移场景中，算法在给定子空间范围内搜索线性模型，并实现了最优超额风险。在多任务个性化场景中，足够的公共数据可以消除私有协调需求，并通过纯局部学习达到相同的效用。 |
| [^4] | [Automated Bug Generation in the era of Large Language Models.](http://arxiv.org/abs/2310.02407) | 本论文探讨了在大型语言模型时代的自动缺陷生成问题，针对难以检测和难以修复的缺陷提出了解决方案，并分析了基于学习的技术中这两个目标的冲突。 |
| [^5] | [Efficient Methods for Non-stationary Online Learning.](http://arxiv.org/abs/2309.08911) | 这项工作提出了一种针对非平稳在线学习的高效方法，通过降低每轮投影的数量来优化动态遗憾和自适应遗憾的计算复杂性。 |
| [^6] | [Protect Federated Learning Against Backdoor Attacks via Data-Free Trigger Generation.](http://arxiv.org/abs/2308.11333) | 通过数据审计和触发器图像过滤等机制，我们提出了一种无数据生成触发器的防御方法来保护联邦学习免受后门攻击。该方法利用后门攻击特征来学习触发器，并生成具有新学习知识的图像。 |

# 详细

[^1]: 面对最坏情况：一种基于学习的对ICP算法鲁棒性分析的对抗攻击

    Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm

    [https://arxiv.org/abs/2403.05666](https://arxiv.org/abs/2403.05666)

    通过对ICP算法进行基于深度学习的攻击，在安全关键应用中评估其鲁棒性，重点在于找到可能的最大ICP姿势误差。

    

    这篇论文提出了一种通过深度学习攻击激光雷达点云来评估迭代最近点（ICP）算法鲁棒性的新方法。对于像自主导航这样的安全关键应用，确保算法在部署前的鲁棒性至关重要。ICP算法已成为基于激光雷达的定位的标准。然而，它产生的姿势估计可能会受到测量数据的影响。数据的污染可能来自各种场景，如遮挡、恶劣天气或传感器的机械问题。不幸的是，ICP的复杂和迭代特性使得评估其对污染的鲁棒性具有挑战性。虽然已经有人努力创建具有挑战性的数据集和开发仿真来经验性地评估ICP的鲁棒性，但我们的方法侧重于通过基于扰动的对抗攻击找到最大可能的ICP姿势误差。

    arXiv:2403.05666v1 Announce Type: cross  Abstract: This paper presents a novel method to assess the resilience of the Iterative Closest Point (ICP) algorithm via deep-learning-based attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms prior to deployments is of utmost importance. The ICP algorithm has become the standard for lidar-based localization. However, the pose estimate it produces can be greatly affected by corruption in the measurements. Corruption can arise from a variety of scenarios such as occlusions, adverse weather, or mechanical issues in the sensor. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP empirically, our method focuses on finding the maximum possible ICP pose error using perturbation-based adversarial
    
[^2]: 闭环无监督表示解缠的β-VAE蒸馏与扩散概率反馈

    Closed-Loop Unsupervised Representation Disentanglement with $\beta$-VAE Distillation and Diffusion Probabilistic Feedback

    [https://arxiv.org/abs/2402.02346](https://arxiv.org/abs/2402.02346)

    本文提出了闭环无监督表示解缠方法CL-Dis，使用扩散自动编码器（Diff-AE）和β-VAE共同提取语义解缠表示，以解决表示解缠面临的问题。

    

    表示解缠可能有助于AI根本上理解现实世界，从而使判别和生成任务受益。目前至少有三个未解决的核心问题：（i）过于依赖标签注释和合成数据-导致在自然情景下泛化能力较差；（ii）启发式/手工制作的解缠约束使得难以自适应地实现最佳训练权衡；（iii）缺乏合理的评估指标，特别是对于真实的无标签数据。为了解决这些挑战，我们提出了一种被称为CL-Dis的闭环无监督表示解缠方法。具体地，我们使用基于扩散的自动编码器（Diff-AE）作为骨干，并使用β-VAE作为副驾驶员来提取语义解缠的表示。扩散模型的强大生成能力和VAE模型的良好解缠能力是互补的。为了加强解缠，使用VAE潜变量。

    Representation disentanglement may help AI fundamentally understand the real world and thus benefit both discrimination and generation tasks. It currently has at least three unresolved core issues: (i) heavy reliance on label annotation and synthetic data -- causing poor generalization on natural scenarios; (ii) heuristic/hand-craft disentangling constraints make it hard to adaptively achieve an optimal training trade-off; (iii) lacking reasonable evaluation metric, especially for the real label-free data. To address these challenges, we propose a \textbf{C}losed-\textbf{L}oop unsupervised representation \textbf{Dis}entanglement approach dubbed \textbf{CL-Dis}. Specifically, we use diffusion-based autoencoder (Diff-AE) as a backbone while resorting to $\beta$-VAE as a co-pilot to extract semantically disentangled representations. The strong generation ability of diffusion model and the good disentanglement ability of VAE model are complementary. To strengthen disentangling, VAE-latent 
    
[^3]: 利用公共表示来进行私有迁移学习

    Leveraging Public Representations for Private Transfer Learning. (arXiv:2312.15551v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.15551](http://arxiv.org/abs/2312.15551)

    该论文探讨了如何利用公共数据来改进私有学习的问题。研究发现，通过学习公共数据中的共享表示，可以在两种迁移学习场景中实现最优的学习效果。在单任务迁移场景中，算法在给定子空间范围内搜索线性模型，并实现了最优超额风险。在多任务个性化场景中，足够的公共数据可以消除私有协调需求，并通过纯局部学习达到相同的效用。

    

    受到将公共数据纳入差分隐私学习的最新实证成功的启发，我们在理论上研究了从公共数据中学到的共享表示如何改进私有学习。我们探讨了线性回归的两种常见迁移学习场景，两者都假设公共任务和私有任务（回归向量）在高维空间中共享一个低秩子空间。在第一种单任务迁移场景中，目标是学习一个在所有用户之间共享的单一模型，每个用户对应数据集中的一行。我们提供了匹配的上下界，证明了我们的算法在给定子空间估计范围内搜索线性模型的算法类中实现了最优超额风险。在多任务模型个性化的第二种情景中，我们表明在有足够的公共数据情况下，用户可以避免私有协调，因为在给定子空间内纯粹的局部学习可以达到相同的效用。

    Motivated by the recent empirical success of incorporating public data into differentially private learning, we theoretically investigate how a shared representation learned from public data can improve private learning. We explore two common scenarios of transfer learning for linear regression, both of which assume the public and private tasks (regression vectors) share a low-rank subspace in a high-dimensional space. In the first single-task transfer scenario, the goal is to learn a single model shared across all users, each corresponding to a row in a dataset. We provide matching upper and lower bounds showing that our algorithm achieves the optimal excess risk within a natural class of algorithms that search for the linear model within the given subspace estimate. In the second scenario of multitask model personalization, we show that with sufficient public data, users can avoid private coordination, as purely local learning within the given subspace achieves the same utility. Take
    
[^4]: 在大型语言模型时代的自动缺陷生成

    Automated Bug Generation in the era of Large Language Models. (arXiv:2310.02407v1 [cs.SE])

    [http://arxiv.org/abs/2310.02407](http://arxiv.org/abs/2310.02407)

    本论文探讨了在大型语言模型时代的自动缺陷生成问题，针对难以检测和难以修复的缺陷提出了解决方案，并分析了基于学习的技术中这两个目标的冲突。

    

    缺陷在软件工程中是至关重要的；过去几十年的许多研究已经提出了检测、定位和修复软件系统中的缺陷的方法。评估这些技术的有效性需要复杂的缺陷，即那些很难通过测试和调试来检测和修复的缺陷。从传统软件工程的角度来看，难以修复的缺陷与正确的代码在多个位置上有所差异，这使得它们难以定位和修复。而难以检测的缺陷则在特定的测试输入和可达条件下展现出来。这两个目标，即生成难以检测和难以修复的缺陷，大多数是一致的；缺陷生成技术可以将多个语句更改为仅在特定输入集合下被覆盖。然而，对于基于学习的技术来说，这两个目标是相互冲突的：一个缺陷应该有与训练数据中的正确代码相似的代码表示，以挑战缺陷预测。

    Bugs are essential in software engineering; many research studies in the past decades have been proposed to detect, localize, and repair bugs in software systems. Effectiveness evaluation of such techniques requires complex bugs, i.e., those that are hard to detect through testing and hard to repair through debugging. From the classic software engineering point of view, a hard-to-repair bug differs from the correct code in multiple locations, making it hard to localize and repair. Hard-to-detect bugs, on the other hand, manifest themselves under specific test inputs and reachability conditions. These two objectives, i.e., generating hard-to-detect and hard-to-repair bugs, are mostly aligned; a bug generation technique can change multiple statements to be covered only under a specific set of inputs. However, these two objectives are conflicting for learning-based techniques: A bug should have a similar code representation to the correct code in the training data to challenge a bug predi
    
[^5]: 非平稳在线学习的高效方法

    Efficient Methods for Non-stationary Online Learning. (arXiv:2309.08911v1 [cs.LG])

    [http://arxiv.org/abs/2309.08911](http://arxiv.org/abs/2309.08911)

    这项工作提出了一种针对非平稳在线学习的高效方法，通过降低每轮投影的数量来优化动态遗憾和自适应遗憾的计算复杂性。

    

    非平稳在线学习近年来引起了广泛关注。特别是在非平稳环境中，动态遗憾和自适应遗憾被提出作为在线凸优化的两个原则性性能度量。为了优化它们，通常采用两层在线集成，由于非平稳性的固有不确定性，其中维护一组基学习器，并采用元算法在运行过程中跟踪最佳学习器。然而，这种两层结构引发了关于计算复杂性的担忧 -这些方法通常同时维护$\mathcal{O}(\log T)$个基学习器，对于一个$T$轮在线游戏，因此每轮执行多次投影到可行域上，当域很复杂时，这成为计算瓶颈。在本文中，我们提出了优化动态遗憾和自适应遗憾的高效方法，将每轮的投影次数从$\mathcal{O}(\log T)$降低到...

    Non-stationary online learning has drawn much attention in recent years. In particular, dynamic regret and adaptive regret are proposed as two principled performance measures for online convex optimization in non-stationary environments. To optimize them, a two-layer online ensemble is usually deployed due to the inherent uncertainty of the non-stationarity, in which a group of base-learners are maintained and a meta-algorithm is employed to track the best one on the fly. However, the two-layer structure raises the concern about the computational complexity -- those methods typically maintain $\mathcal{O}(\log T)$ base-learners simultaneously for a $T$-round online game and thus perform multiple projections onto the feasible domain per round, which becomes the computational bottleneck when the domain is complicated. In this paper, we present efficient methods for optimizing dynamic regret and adaptive regret, which reduce the number of projections per round from $\mathcal{O}(\log T)$ t
    
[^6]: 无数据生成触发器保护联邦学习免受后门攻击

    Protect Federated Learning Against Backdoor Attacks via Data-Free Trigger Generation. (arXiv:2308.11333v1 [cs.LG])

    [http://arxiv.org/abs/2308.11333](http://arxiv.org/abs/2308.11333)

    通过数据审计和触发器图像过滤等机制，我们提出了一种无数据生成触发器的防御方法来保护联邦学习免受后门攻击。该方法利用后门攻击特征来学习触发器，并生成具有新学习知识的图像。

    

    作为分布式机器学习范 paradigm，联邦学习 (FL) 可以使大规模客户端在不共享原始数据的情况下协同训练模型。然而，由于对不可信客户端的数据审计缺失，FL 易受污染攻击，特别是后门攻击。攻击者可以通过使用污染数据进行本地训练或直接更改模型参数，轻而易举地将后门注入模型，从而触发模型对图像中的目标模式进行错误分类。为解决这些问题，我们提出了一种基于两个后门攻击特征的新型无数据生成触发器防御方法：i) 触发器学习速度比普通知识更快，ii) 触发器模式对图像分类的影响大于普通类别模式。我们的方法通过识别旧和新全局模型之间的差异，生成具有新学习知识的图像，并通过评估方法过滤触发器图像。

    As a distributed machine learning paradigm, Federated Learning (FL) enables large-scale clients to collaboratively train a model without sharing their raw data. However, due to the lack of data auditing for untrusted clients, FL is vulnerable to poisoning attacks, especially backdoor attacks. By using poisoned data for local training or directly changing the model parameters, attackers can easily inject backdoors into the model, which can trigger the model to make misclassification of targeted patterns in images. To address these issues, we propose a novel data-free trigger-generation-based defense approach based on the two characteristics of backdoor attacks: i) triggers are learned faster than normal knowledge, and ii) trigger patterns have a greater effect on image classification than normal class patterns. Our approach generates the images with newly learned knowledge by identifying the differences between the old and new global models, and filters trigger images by evaluating the 
    

