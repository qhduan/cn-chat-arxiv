# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hierarchical Randomized Smoothing.](http://arxiv.org/abs/2310.16221) | 分层随机平滑是一种在复杂数据上进行鲁棒性认证的解决方案，通过只在一个对象的子集上添加随机噪声，以更有针对性的方式提供了更强的鲁棒性保证和高准确性。 |
| [^2] | [Convergence of Sign-based Random Reshuffling Algorithms for Nonconvex Optimization.](http://arxiv.org/abs/2310.15976) | 该论文通过证明signSGD算法在非凸优化问题中的随机重排（SignRR）的收敛性，弥补了现有分析中的缺陷，提出了SignRVR和SignRVM算法，并且都以较快的收敛速度收敛于全局最优解。 |
| [^3] | [Random postprocessing for combinatorial Bayesian optimization.](http://arxiv.org/abs/2309.02842) | 针对组合贝叶斯优化，我们研究了一种随机后处理方法，严格禁止数据集中的重复样本，结果表明此方法显著减少了顺序步骤数，特别是在最大后验估计的情况下，为解决高维问题中贝叶斯优化的收敛速度慢提供了一种简单但通用的策略。 |
| [^4] | [Bounded P-values in Parametric Programming-based Selective Inference.](http://arxiv.org/abs/2307.11351) | 本研究提出了一种降低参数规划选择性推断计算成本的方法，通过计算p值的上界和下界来保证所需精度。 |
| [^5] | [Safe Model-Based Multi-Agent Mean-Field Reinforcement Learning.](http://arxiv.org/abs/2306.17052) | 本文提出了Safe-$\text{M}^3$-UCRL算法，通过使用模型中的认知不确定性和对数障碍方法，实现了在未知转移动态情况下达到安全策略的优化，成功解决了大规模多智能体协调问题。 |
| [^6] | [Leveraging Locality and Robustness to Achieve Massively Scalable Gaussian Process Regression.](http://arxiv.org/abs/2306.14731) | 该研究提出了一种新的思路，通过探索 GPnn 的鲁棒性和极限行为实现大规模高斯过程回归，即使在出现重大小错误的情况下只需要花费少量的工作进行参数估计即可实现高 MSE 准确性。同时，该研究成功解决了加性噪声方差带来的不确定度校准和 NLL 准确性问题。 |
| [^7] | [SimFBO: Towards Simple, Flexible and Communication-efficient Federated Bilevel Learning.](http://arxiv.org/abs/2305.19442) | SimFBO和其ShroFBO变体提出了一个简单、灵活且通信高效的FBO框架，可以应用于元学习和超参数优化任务。 |
| [^8] | [Embedding Inequalities for Barron-type Spaces.](http://arxiv.org/abs/2305.19082) | 本文测量了Barron空间和谱Barron空间之间的关系，并提供了嵌入不等式。 |
| [^9] | [Identification of Negative Transfers in Multitask Learning Using Surrogate Models.](http://arxiv.org/abs/2303.14582) | 本文提出了一种通过代理建模来解决多任务学习中负迁移问题的方法，能够识别哪些源任务的子集会对目标任务有帮助。 |
| [^10] | [How to Trust Your Diffusion Model: A Convex Optimization Approach to Conformal Risk Control.](http://arxiv.org/abs/2302.03791) | 本文提出了一种风险控制预测集（RCPS）程序的推广，称为$K$-RCPS，它允许为任何扩散模型提供逐个校准的未来样本间隔，并控制相对于基准真实图像的某种风险概念，同时保持最小平均区间长度。 |

# 详细

[^1]: 分层随机平滑

    Hierarchical Randomized Smoothing. (arXiv:2310.16221v1 [cs.LG])

    [http://arxiv.org/abs/2310.16221](http://arxiv.org/abs/2310.16221)

    分层随机平滑是一种在复杂数据上进行鲁棒性认证的解决方案，通过只在一个对象的子集上添加随机噪声，以更有针对性的方式提供了更强的鲁棒性保证和高准确性。

    

    真实世界的数据是复杂的，通常由可分解为多个实体的对象组成（例如，将图像分解为像素，将图形分解为相互连接的节点）。随机平滑是一种强大的框架，可以使模型在其输入的微小变化上具有证明的鲁棒性-通过在分类之前随机添加噪声来保证多数投票的鲁棒性。然而，当对手不是任意干扰整个对象（例如图像），而是对象的某个实体的子集（例如像素）时，通过随机平滑对这种复杂数据进行鲁棒性认证是具有挑战性的。作为解决方案，我们引入了分层随机平滑：我们通过仅在随机选择的实体子集上添加随机噪声来部分平滑对象。通过以比现有方法更有针对性的方式添加噪声，我们获得更强的鲁棒性保证，同时保持高准确性。我们使用不同的噪声分布初始化分层平滑，得到了新的鲁棒性保证。

    Real-world data is complex and often consists of objects that can be decomposed into multiple entities (e.g. images into pixels, graphs into interconnected nodes). Randomized smoothing is a powerful framework for making models provably robust against small changes to their inputs - by guaranteeing robustness of the majority vote when randomly adding noise before classification. Yet, certifying robustness on such complex data via randomized smoothing is challenging when adversaries do not arbitrarily perturb entire objects (e.g. images) but only a subset of their entities (e.g. pixels). As a solution, we introduce hierarchical randomized smoothing: We partially smooth objects by adding random noise only on a randomly selected subset of their entities. By adding noise in a more targeted manner than existing methods we obtain stronger robustness guarantees while maintaining high accuracy. We initialize hierarchical smoothing using different noising distributions, yielding novel robustness
    
[^2]: 非凸优化的基于符号随机重排算法的收敛性研究

    Convergence of Sign-based Random Reshuffling Algorithms for Nonconvex Optimization. (arXiv:2310.15976v1 [cs.LG])

    [http://arxiv.org/abs/2310.15976](http://arxiv.org/abs/2310.15976)

    该论文通过证明signSGD算法在非凸优化问题中的随机重排（SignRR）的收敛性，弥补了现有分析中的缺陷，提出了SignRVR和SignRVM算法，并且都以较快的收敛速度收敛于全局最优解。

    

    由于其通信效率较高，signSGD在非凸优化中很受欢迎。然而，现有对signSGD的分析基于假设每次迭代中的数据都是有放回采样的，这与实际实现中数据的随机重排和顺序馈送进算法的情况相矛盾。为了弥补这一差距，我们证明了signSGD在非凸优化中的随机重排（SignRR）的首个收敛结果。给定数据集大小$n$，数据迭代次数$T$，和随机梯度的方差限制$\sigma^2$，我们证明了SignRR的收敛速度与signSGD相同，为$O(\log(nT)/\sqrt{nT} + \|\sigma\|_1)$ \citep{bernstein2018signsgd}。接着，我们还提出了 SignRVR 和 SignRVM，分别利用了方差约减梯度和动量更新，都以$O(\log(nT)/\sqrt{nT})$的速度收敛。与signSGD的分析不同，我们的结果不需要每次迭代中极大的批次大小与同等数量的梯度进行比较。

    signSGD is popular in nonconvex optimization due to its communication efficiency. Yet, existing analyses of signSGD rely on assuming that data are sampled with replacement in each iteration, contradicting the practical implementation where data are randomly reshuffled and sequentially fed into the algorithm. We bridge this gap by proving the first convergence result of signSGD with random reshuffling (SignRR) for nonconvex optimization. Given the dataset size $n$, the number of epochs of data passes $T$, and the variance bound of a stochastic gradient $\sigma^2$, we show that SignRR has the same convergence rate $O(\log(nT)/\sqrt{nT} + \|\sigma\|_1)$ as signSGD \citep{bernstein2018signsgd}. We then present SignRVR and SignRVM, which leverage variance-reduced gradients and momentum updates respectively, both converging at $O(\log(nT)/\sqrt{nT})$. In contrast with the analysis of signSGD, our results do not require an extremely large batch size in each iteration to be of the same order a
    
[^3]: 针对组合贝叶斯优化的随机后处理方法

    Random postprocessing for combinatorial Bayesian optimization. (arXiv:2309.02842v1 [cs.LG])

    [http://arxiv.org/abs/2309.02842](http://arxiv.org/abs/2309.02842)

    针对组合贝叶斯优化，我们研究了一种随机后处理方法，严格禁止数据集中的重复样本，结果表明此方法显著减少了顺序步骤数，特别是在最大后验估计的情况下，为解决高维问题中贝叶斯优化的收敛速度慢提供了一种简单但通用的策略。

    

    基于模型的顺序方法用于离散的“黑盒”优化问题，包括贝叶斯优化技术，通常会对给定的目标函数访问多次相同的点，导致需要很多步骤才能找到全局最优解。在这里，我们对贝叶斯优化中的一种后处理方法进行了数值研究，该方法严格禁止数据集中的重复样本。我们发现后处理方法显著减少了找到全局最优解所需的顺序步骤数，特别是当采样函数是最大后验估计时。我们的结果为解决高维问题中贝叶斯优化的收敛速度慢提供了一种简单但通用的策略。

    Model-based sequential approaches to discrete "black-box" optimization, including Bayesian optimization techniques, often access the same points multiple times for a given objective function in interest, resulting in many steps to find the global optimum. Here, we numerically study the effect of a postprocessing method on Bayesian optimization that strictly prohibits duplicated samples in the dataset. We find the postprocessing method significantly reduces the number of sequential steps to find the global optimum, especially when the acquisition function is of maximum a posterior estimation. Our results provide a simple but general strategy to solve the slow convergence of Bayesian optimization for high-dimensional problems.
    
[^4]: 参数规划的选择性推断中的有界P值

    Bounded P-values in Parametric Programming-based Selective Inference. (arXiv:2307.11351v1 [stat.ML])

    [http://arxiv.org/abs/2307.11351](http://arxiv.org/abs/2307.11351)

    本研究提出了一种降低参数规划选择性推断计算成本的方法，通过计算p值的上界和下界来保证所需精度。

    

    选择性推断（SI）作为一种适用于数据驱动的假设检验的有前景的框架，一直受到研究关注。SI的基本思想是在一个假设被选中的事件的条件下进行推断。为了进行SI，必须以可追踪的形式对这个事件进行描述。当选择事件难以描述时，可以引入额外的条件以使其可处理。这些额外的条件往往会导致功效的损失，这一问题被称为过度条件化。基于参数规划的SI（PP-based SI）被提出作为解决过度条件化问题的一种方法。PP-based SI的主要问题是由于需要完全地探索数据空间而导致计算成本高。本研究引入了一种降低计算成本的过程，同时保证所需精度，通过提出计算p值的上界和下界的方法。我们还提出了三种类型的搜索策略。

    Selective inference (SI) has been actively studied as a promising framework for statistical hypothesis testing for data-driven hypotheses. The basic idea of SI is to make inferences conditional on an event that a hypothesis is selected. In order to perform SI, this event must be characterized in a traceable form. When selection event is too difficult to characterize, additional conditions are introduced for tractability. This additional conditions often causes the loss of power, and this issue is referred to as over-conditioning. Parametric programming-based SI (PP-based SI) has been proposed as one way to address the over-conditioning issue. The main problem of PP-based SI is its high computational cost due to the need to exhaustively explore the data space. In this study, we introduce a procedure to reduce the computational cost while guaranteeing the desired precision, by proposing a method to compute the upper and lower bounds of p-values. We also proposed three types of search str
    
[^5]: 安全的基于模型的多智能体均场强化学习

    Safe Model-Based Multi-Agent Mean-Field Reinforcement Learning. (arXiv:2306.17052v1 [cs.LG])

    [http://arxiv.org/abs/2306.17052](http://arxiv.org/abs/2306.17052)

    本文提出了Safe-$\text{M}^3$-UCRL算法，通过使用模型中的认知不确定性和对数障碍方法，实现了在未知转移动态情况下达到安全策略的优化，成功解决了大规模多智能体协调问题。

    

    许多应用，比如共享交通，需要协调大量的智能体。均场强化学习通过优化代表性智能体的策略来应对由此带来的可扩展性挑战。在本文中，我们解决了一个重要的泛化问题，即智能体分布存在全局约束的情况（例如需要满足容量约束或最小覆盖要求）。我们提出了Safe-$\text{M}^3$-UCRL，这是第一个能够在未知转移动态的情况下实现安全策略的基于模型的算法。作为一个关键因素，它在保证悲观约束满足的同时，利用转移模型中的认知不确定性来使用对数障碍方法确保高概率。我们在许多共享交通运营商面临的车辆重定位问题上展示了Safe-$\text{M}^3$-UCRL，并通过基于深圳出租车轨迹数据的仿真评估其性能。我们的算法能够有效满足关键需求。

    Many applications, e.g., in shared mobility, require coordinating a large number of agents. Mean-field reinforcement learning addresses the resulting scalability challenge by optimizing the policy of a representative agent. In this paper, we address an important generalization where there exist global constraints on the distribution of agents (e.g., requiring capacity constraints or minimum coverage requirements to be met). We propose Safe-$\text{M}^3$-UCRL, the first model-based algorithm that attains safe policies even in the case of unknown transition dynamics. As a key ingredient, it uses epistemic uncertainty in the transition model within a log-barrier approach to ensure pessimistic constraints satisfaction with high probability. We showcase Safe-$\text{M}^3$-UCRL on the vehicle repositioning problem faced by many shared mobility operators and evaluate its performance through simulations built on Shenzhen taxi trajectory data. Our algorithm effectively meets the demand in critica
    
[^6]: 利用本地性和鲁棒性实现大规模高斯过程回归

    Leveraging Locality and Robustness to Achieve Massively Scalable Gaussian Process Regression. (arXiv:2306.14731v1 [stat.ML])

    [http://arxiv.org/abs/2306.14731](http://arxiv.org/abs/2306.14731)

    该研究提出了一种新的思路，通过探索 GPnn 的鲁棒性和极限行为实现大规模高斯过程回归，即使在出现重大小错误的情况下只需要花费少量的工作进行参数估计即可实现高 MSE 准确性。同时，该研究成功解决了加性噪声方差带来的不确定度校准和 NLL 准确性问题。

    

    高斯过程回归所提供的精确预测和原则性不确定性测量会产生 O(n^3) 的成本，这对于现代大规模应用来说是难以承受的。因此，出现了大量关于计算效率的研究。我们通过探索 GP 最近邻预测(GPnn) 的鲁棒性和极限行为引入了一种新的视角。我们通过理论和模拟证明，随着数据量 n 的增加，估计参数和 GP 模型假设的准确性对 GPnn 预测准确性的影响逐渐减小。因此，为了实现高 MSE 准确性，即使在出现重大错误的情况下, 只需要花费少量的工作进行参数估计即可。相比之下，随着 n 趋近于无穷大，我们发现不确定度校准和 NLL 仍对一个参数敏感，即加性噪声方差；但我们证明可以纠正这种不准确性，并实现良好的不确定度校准和 NLL。

    The accurate predictions and principled uncertainty measures provided by GP regression incur O(n^3) cost which is prohibitive for modern-day large-scale applications. This has motivated extensive work on computationally efficient approximations. We introduce a new perspective by exploring robustness properties and limiting behaviour of GP nearest-neighbour (GPnn) prediction. We demonstrate through theory and simulation that as the data-size n increases, accuracy of estimated parameters and GP model assumptions become increasingly irrelevant to GPnn predictive accuracy. Consequently, it is sufficient to spend small amounts of work on parameter estimation in order to achieve high MSE accuracy, even in the presence of gross misspecification. In contrast, as n tends to infinity, uncertainty calibration and NLL are shown to remain sensitive to just one parameter, the additive noise-variance; but we show that this source of inaccuracy can be corrected for, thereby achieving both well-calibra
    
[^7]: SimFBO：简单、灵活且通信高效的联邦双层学习

    SimFBO: Towards Simple, Flexible and Communication-efficient Federated Bilevel Learning. (arXiv:2305.19442v1 [cs.LG])

    [http://arxiv.org/abs/2305.19442](http://arxiv.org/abs/2305.19442)

    SimFBO和其ShroFBO变体提出了一个简单、灵活且通信高效的FBO框架，可以应用于元学习和超参数优化任务。

    

    近来，由于元学习、微调、超参数调整等领域中嵌套优化结构的出现，联邦双层优化（FBO）在机器学习和边缘计算中显示了巨大的潜力。然而，现有的FBO算法往往涉及复杂的计算，并需要每次迭代多个子循环，每个子循环包含多个通信轮。在本文中，我们提出了一个名为SimFBO的简单灵活的FBO框架，它易于实现，不需要子循环，并包括一种广义的服务器端聚合和更新以提高通信效率。我们进一步提出了系统级异构鲁棒FBO（ShroFBO）作为SimFBO的变体，其对本地计算的异构有更强的鲁棒性。我们证明了在部分客户端参与和无替换的客户端采样下，SimFBO和ShroFBO可以实现线性收敛加速，同时改进了样本和通信复杂度。实验证明了它们在图像分类数据集的元学习和真实世界数据集上的超参数优化任务中的有效性。

    Federated bilevel optimization (FBO) has shown great potential recently in machine learning and edge computing due to the emerging nested optimization structure in meta-learning, fine-tuning, hyperparameter tuning, etc. However, existing FBO algorithms often involve complicated computations and require multiple sub-loops per iteration, each of which contains a number of communication rounds. In this paper, we propose a simple and flexible FBO framework named SimFBO, which is easy to implement without sub-loops, and includes a generalized server-side aggregation and update for improving communication efficiency. We further propose System-level heterogeneity robust FBO (ShroFBO) as a variant of SimFBO with stronger resilience to heterogeneous local computation. We show that SimFBO and ShroFBO provably achieve a linear convergence speedup with partial client participation and client sampling without replacement, as well as improved sample and communication complexities. Experiments demons
    
[^8]: Barron型空间的嵌入不等式

    Embedding Inequalities for Barron-type Spaces. (arXiv:2305.19082v1 [stat.ML])

    [http://arxiv.org/abs/2305.19082](http://arxiv.org/abs/2305.19082)

    本文测量了Barron空间和谱Barron空间之间的关系，并提供了嵌入不等式。

    

    深度学习理论中的一个基本问题是理解高维条件下两层神经网络的逼近和泛化性质。为了解决这个问题，研究人员引入了Barron空间$\mathcal{B}_s(\Omega)$和谱Barron空间$\mathcal{F}_s(\Omega)$，其中指数$s$表征了这些空间中函数的平滑性，$\Omega\subset\mathbb{R}^d$表示输入域。然而，两种类型的Barron空间之间的关系仍不清楚。本文通过以下不等式建立了这些空间之间的连续嵌入：对于任意$\delta\in(0,1),s\in\mathbb{N}^{+}$和$f:\Omega \mapsto \mathbb{R}$，都有\[ \delta\gamma^{\delta-s}_{\Omega}\|f\|_{\mathcal{F}_{s-\delta}(\Omega)}\lesssim_s \|f\|_{\mathcal{B}_s(\Omega)}\lesssim_s \|f\|_{\mathcal{F}_{s+1}(\Omega)}, \]其中$\gamma_{\Omega}=\sup_{\|v\|_2=1,x\in\Omega}|v^Tx|$，$\lesssim_s$表示仅与平滑参数$s$有关的常数。

    One of the fundamental problems in deep learning theory is understanding the approximation and generalization properties of two-layer neural networks in high dimensions. In order to tackle this issue, researchers have introduced the Barron space $\mathcal{B}_s(\Omega)$ and the spectral Barron space $\mathcal{F}_s(\Omega)$, where the index $s$ characterizes the smoothness of functions within these spaces and $\Omega\subset\mathbb{R}^d$ represents the input domain. However, it is still not clear what is the relationship between the two types of Barron spaces. In this paper, we establish continuous embeddings between these spaces as implied by the following inequality: for any $\delta\in (0,1), s\in \mathbb{N}^{+}$ and $f: \Omega \mapsto\mathbb{R}$, it holds that \[ \delta\gamma^{\delta-s}_{\Omega}\|f\|_{\mathcal{F}_{s-\delta}(\Omega)}\lesssim_s \|f\|_{\mathcal{B}_s(\Omega)}\lesssim_s \|f\|_{\mathcal{F}_{s+1}(\Omega)}, \] where $\gamma_{\Omega}=\sup_{\|v\|_2=1,x\in\Omega}|v^Tx|$ and notab
    
[^9]: 利用代理模型识别多任务学习中的负迁移

    Identification of Negative Transfers in Multitask Learning Using Surrogate Models. (arXiv:2303.14582v1 [cs.LG])

    [http://arxiv.org/abs/2303.14582](http://arxiv.org/abs/2303.14582)

    本文提出了一种通过代理建模来解决多任务学习中负迁移问题的方法，能够识别哪些源任务的子集会对目标任务有帮助。

    

    多任务学习广泛应用于通过增加多个相关源任务来训练低资源目标任务。然而，将所有源任务与目标任务简单组合并不总是能提高目标任务的预测性能，因为会存在负迁移。因此，多任务学习的一个关键问题是识别哪些源任务的子集会对目标任务有益。这个问题在计算上很具有挑战性，因为子集的数量随着源任务的数量呈指数级增长。在本文中，我们介绍了一种通过代理建模来解决此问题的有效方法。在代理建模中，我们对源任务进行采样（随机），并预先计算它们的多任务学习表现；然后，我们用线性回归模型来逼近预先计算的表现，该模型也可用于预测未采样的子集的表现。我们在几个合成示例和一个现实世界的多语言情感分析任务上证明了我们方法的有效性。

    Multitask learning is widely used in practice to train a low-resource target task by augmenting it with multiple related source tasks. Yet, naively combining all the source tasks with a target task does not always improve the prediction performance for the target task due to negative transfers. Thus, a critical problem in multitask learning is identifying subsets of source tasks that would benefit the target task. This problem is computationally challenging since the number of subsets grows exponentially with the number of source tasks; efficient heuristics for subset selection does not always capture the relationship between task subsets and multitask learning performances. In this paper, we introduce an efficient procedure to address this problem via surrogate modeling. In surrogate modeling, we sample (random) subsets of source tasks and precompute their multitask learning performances; Then, we approximate the precomputed performances with a linear regression model that can also be
    
[^10]: 如何信任您的扩散模型：一种凸优化方法应对符合风险控制的因式分解模型

    How to Trust Your Diffusion Model: A Convex Optimization Approach to Conformal Risk Control. (arXiv:2302.03791v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.03791](http://arxiv.org/abs/2302.03791)

    本文提出了一种风险控制预测集（RCPS）程序的推广，称为$K$-RCPS，它允许为任何扩散模型提供逐个校准的未来样本间隔，并控制相对于基准真实图像的某种风险概念，同时保持最小平均区间长度。

    

    基于分数的生成建模方法，简称扩散模型，在多个重要领域和任务中继续增长。尽管它们提供了来自经验分布的高质量和多样化样本，但在其负责任地用于关键场景方面的可靠性和可信度仍存在重要问题。收敛预测是一种现代工具，用于为任何黑盒子预测器构建有限样本、分布自由的不确定性保证。在这项工作中，我们专注于图像到图像回归任务，并提出了一种风险控制预测集（RCPS）程序的推广，我们称之为$K$-RCPS，它允许$(i)$为任何扩散模型提供逐个校准的未来样本间隔，并$(ii)$控制相对于基准真实图像的某种风险概念，同时保持最小平均区间长度。与现有的收敛风险控制过程不同，我们的过程依靠一种新型的凸优化公式，使其具有计算效率和易于实现的特点。我们在几个图像到图像回归任务上使用得分为基础的生成建模方法来说明我们的程序的有效性，展示了高度校准和良好控制的预测间隔。

    Score-based generative modeling, informally referred to as diffusion models, continue to grow in popularity across several important domains and tasks. While they provide high-quality and diverse samples from empirical distributions, important questions remain on the reliability and trustworthiness of these sampling procedures for their responsible use in critical scenarios. Conformal prediction is a modern tool to construct finite-sample, distribution-free uncertainty guarantees for any black-box predictor. In this work, we focus on image-to-image regression tasks and we present a generalization of the Risk-Controlling Prediction Sets (RCPS) procedure, that we term $K$-RCPS, which allows to $(i)$ provide entrywise calibrated intervals for future samples of any diffusion model, and $(ii)$ control a certain notion of risk with respect to a ground truth image with minimal mean interval length. Differently from existing conformal risk control procedures, ours relies on a novel convex opti
    

