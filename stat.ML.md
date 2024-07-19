# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Discrete Latent Graph Generative Modeling with Diffusion Bridges](https://arxiv.org/abs/2403.16883) | GLAD是一个在离散潜在空间上操作的图生成模型，通过适应扩散桥结构学习其离散潜在空间的先验，避免了依赖于原始数据空间的分解，在图生成任务中表现出优越性。 |
| [^2] | [PARMESAN: Parameter-Free Memory Search and Transduction for Dense Prediction Tasks](https://arxiv.org/abs/2403.11743) | 通过引入转导的概念，提出了PARMESAN，一种用于解决密集预测任务的无参数内存搜索和转导方法，实现了灵活性和无需连续训练的学习。 |
| [^3] | [Feedback Efficient Online Fine-Tuning of Diffusion Models](https://arxiv.org/abs/2402.16359) | 提出了一种反馈高效的在线微调扩散模型的强化学习程序 |
| [^4] | [Private Aggregation in Wireless Federated Learning with Heterogeneous Clusters.](http://arxiv.org/abs/2306.14088) | 本文探讨了在一个无线系统中，考虑到信息论隐私的条件下，通过基站连接到联合器的客户端，如何解决联邦学习中的隐私数据聚合问题。 |
| [^5] | [Bayesian sequential design of computer experiments for quantile set inversion.](http://arxiv.org/abs/2211.01008) | 本论文提出了一种基于贝叶斯策略的量化集反演方法，通过高斯过程建模和逐步不确定性减少原理，顺序选择评估函数的点，从而有效近似感兴趣的集合。 |

# 详细

[^1]: 带扩散桥的离散潜在图生成建模

    Discrete Latent Graph Generative Modeling with Diffusion Bridges

    [https://arxiv.org/abs/2403.16883](https://arxiv.org/abs/2403.16883)

    GLAD是一个在离散潜在空间上操作的图生成模型，通过适应扩散桥结构学习其离散潜在空间的先验，避免了依赖于原始数据空间的分解，在图生成任务中表现出优越性。

    

    学习潜在空间中的图生成模型相比于在原始数据空间上操作的模型受到较少关注，迄今表现出的性能乏善可陈。我们提出了GLAD，一个潜在空间图生成模型。与大多数先前的潜在空间图生成模型不同，GLAD在保留图结构的离散性质方面运行，无需进行诸如潜在空间连续性等不自然的假设。我们通过将扩散桥调整到其结构，来学习我们离散潜在空间的先验。通过在适当构建的潜在空间上操作，我们避免依赖于常用于在原始数据空间操作的模型中的分解。我们在一系列图基准数据集上进行实验，明显展示了离散潜在空间的优越性，并取得了最先进的图生成性能，使GLA

    arXiv:2403.16883v1 Announce Type: new  Abstract: Learning graph generative models over latent spaces has received less attention compared to models that operate on the original data space and has so far demonstrated lacklustre performance. We present GLAD a latent space graph generative model. Unlike most previous latent space graph generative models, GLAD operates on a discrete latent space that preserves to a significant extent the discrete nature of the graph structures making no unnatural assumptions such as latent space continuity. We learn the prior of our discrete latent space by adapting diffusion bridges to its structure. By operating over an appropriately constructed latent space we avoid relying on decompositions that are often used in models that operate in the original data space. We present experiments on a series of graph benchmark datasets which clearly show the superiority of the discrete latent space and obtain state of the art graph generative performance, making GLA
    
[^2]: PARMESAN: 用于密集预测任务的无参数内存搜索与转导

    PARMESAN: Parameter-Free Memory Search and Transduction for Dense Prediction Tasks

    [https://arxiv.org/abs/2403.11743](https://arxiv.org/abs/2403.11743)

    通过引入转导的概念，提出了PARMESAN，一种用于解决密集预测任务的无参数内存搜索和转导方法，实现了灵活性和无需连续训练的学习。

    

    在这项工作中，我们通过转导推理来解决深度学习中的灵活性问题。我们提出了PARMESAN（无参数内存搜索与转导），这是一种可扩展的转导方法，利用内存模块来解决密集预测任务。在推断过程中，内存中的隐藏表示被搜索以找到相应的示例。与其他方法不同，PARMESAN通过修改内存内容学习，而无需进行任何连续训练或微调可学习参数。我们的方法与常用的神经结构兼容。

    arXiv:2403.11743v1 Announce Type: new  Abstract: In this work we address flexibility in deep learning by means of transductive reasoning. For adaptation to new tasks or new data, existing methods typically involve tuning of learnable parameters or even complete re-training from scratch, rendering such approaches unflexible in practice. We argue that the notion of separating computation from memory by the means of transduction can act as a stepping stone for solving these issues. We therefore propose PARMESAN (parameter-free memory search and transduction), a scalable transduction method which leverages a memory module for solving dense prediction tasks. At inference, hidden representations in memory are being searched to find corresponding examples. In contrast to other methods, PARMESAN learns without the requirement for any continuous training or fine-tuning of learnable parameters simply by modifying the memory content. Our method is compatible with commonly used neural architecture
    
[^3]: 反馈高效在线微调扩散模型

    Feedback Efficient Online Fine-Tuning of Diffusion Models

    [https://arxiv.org/abs/2402.16359](https://arxiv.org/abs/2402.16359)

    提出了一种反馈高效的在线微调扩散模型的强化学习程序

    

    扩散模型在建模复杂数据分布方面表现出色，包括图像，蛋白质和小分子的分布。然而，在许多情况下，我们的目标是模拟最大化某些属性的分布的部分：例如，我们可能希望生成具有高审美质量的图像，或具有高生物活性的分子。自然地，我们可以将这视为一个强化学习（RL）问题，其目标是微调扩散模型以最大化与某些属性对应的奖励函数。即使可以访问地面真实奖励函数的在线查询，有效地发现高奖励样本也可能具有挑战性：它们在初始分布中的概率可能很低，并且可能存在许多不可行的样本，甚至没有定义良好的奖励（例如，不自然的图像或物理上不可能的分子）。在这项工作中，我们提出了一种新颖的强化学习程序，可以高效地发现高奖励样本。

    arXiv:2402.16359v1 Announce Type: cross  Abstract: Diffusion models excel at modeling complex data distributions, including those of images, proteins, and small molecules. However, in many cases, our goal is to model parts of the distribution that maximize certain properties: for example, we may want to generate images with high aesthetic quality, or molecules with high bioactivity. It is natural to frame this as a reinforcement learning (RL) problem, in which the objective is to fine-tune a diffusion model to maximize a reward function that corresponds to some property. Even with access to online queries of the ground-truth reward function, efficiently discovering high-reward samples can be challenging: they might have a low probability in the initial distribution, and there might be many infeasible samples that do not even have a well-defined reward (e.g., unnatural images or physically impossible molecules). In this work, we propose a novel reinforcement learning procedure that effi
    
[^4]: 非同质化集群下的无线联邦学习中的私有数据聚合

    Private Aggregation in Wireless Federated Learning with Heterogeneous Clusters. (arXiv:2306.14088v1 [cs.LG])

    [http://arxiv.org/abs/2306.14088](http://arxiv.org/abs/2306.14088)

    本文探讨了在一个无线系统中，考虑到信息论隐私的条件下，通过基站连接到联合器的客户端，如何解决联邦学习中的隐私数据聚合问题。

    

    联邦学习是通过多个参与客户端私有数据的协同训练神经网络的方法。在训练神经网络的过程中，使用一种著名并广泛使用的迭代优化算法——梯度下降算法。每个客户端使用本地数据计算局部梯度并将其发送给联合器以进行聚合。客户端数据的隐私是一个主要问题。实际上，观察到局部梯度就足以泄露客户端的数据。已研究了用于应对联邦学习中隐私问题的私有聚合方案，其中所有用户都彼此连接并与联合器连接。本文考虑了一个无线系统架构，其中客户端仅通过基站连接到联合器。当需要信息论隐私时，我们推导出通信成本的基本极限，并引入和分析了一种针对这种情况量身定制的私有聚合方案。

    Federated learning collaboratively trains a neural network on privately owned data held by several participating clients. The gradient descent algorithm, a well-known and popular iterative optimization procedure, is run to train the neural network. Every client uses its local data to compute partial gradients and sends it to the federator which aggregates the results. Privacy of the clients' data is a major concern. In fact, observing the partial gradients can be enough to reveal the clients' data. Private aggregation schemes have been investigated to tackle the privacy problem in federated learning where all the users are connected to each other and to the federator. In this paper, we consider a wireless system architecture where clients are only connected to the federator via base stations. We derive fundamental limits on the communication cost when information-theoretic privacy is required, and introduce and analyze a private aggregation scheme tailored for this setting.
    
[^5]: 基于贝叶斯序贯设计的计算机实验量化集反演

    Bayesian sequential design of computer experiments for quantile set inversion. (arXiv:2211.01008v2 [stat.ML] CROSS LISTED)

    [http://arxiv.org/abs/2211.01008](http://arxiv.org/abs/2211.01008)

    本论文提出了一种基于贝叶斯策略的量化集反演方法，通过高斯过程建模和逐步不确定性减少原理，顺序选择评估函数的点，从而有效近似感兴趣的集合。

    

    我们考虑一个未知的多元函数，它代表着一个系统，如一个复杂的数值模拟器，同时具有确定性和不确定性的输入。我们的目标是估计确定性输入集，这些输入导致的输出（就不确定性输入的分布而言）属于给定集合的概率小于给定阈值。这个问题被称为量化集反演（QSI），例如在稳健（基于可靠性）优化问题的背景下，当寻找满足约束条件且具有足够大概率的解集时会发生。为了解决QSI问题，我们提出了一种基于高斯过程建模和逐步不确定性减少（SUR）原理的贝叶斯策略，以顺序选择应该评估函数的点，以便高效近似感兴趣的集合。通过几个数值实验，我们展示了所提出的SUR策略的性能和价值

    We consider an unknown multivariate function representing a system-such as a complex numerical simulator-taking both deterministic and uncertain inputs. Our objective is to estimate the set of deterministic inputs leading to outputs whose probability (with respect to the distribution of the uncertain inputs) of belonging to a given set is less than a given threshold. This problem, which we call Quantile Set Inversion (QSI), occurs for instance in the context of robust (reliability-based) optimization problems, when looking for the set of solutions that satisfy the constraints with sufficiently large probability. To solve the QSI problem, we propose a Bayesian strategy based on Gaussian process modeling and the Stepwise Uncertainty Reduction (SUR) principle, to sequentially choose the points at which the function should be evaluated to efficiently approximate the set of interest. We illustrate the performance and interest of the proposed SUR strategy through several numerical experiment
    

