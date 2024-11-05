# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Query-Efficient Correlation Clustering with Noisy Oracle](https://rss.arxiv.org/abs/2402.01400) | 本论文提出了一种低查询成本的聚类方法，利用纯在组合多臂赌博机探索范式实现在线学习，并设计了能在NP-hard情况下运行的多项式时间算法。 |
| [^2] | [Geometry of Polynomial Neural Networks](https://rss.arxiv.org/abs/2402.00949) | 本研究利用代数几何工具研究了具有单项式激活函数的多项式神经网络的表达性和学习过程，通过对神经流形的维度和学习度的研究，提供了网络表达能力和训练复杂度的度量，并给出了可学函数数量的上界。 |
| [^3] | [Efficient and Sharp Off-Policy Evaluation in Robust Markov Decision Processes](https://arxiv.org/abs/2404.00099) | 在对抗性环境中，本论文提出了一种可修改转移核密度的扰动模型，拓展了传统的边缘敏感性模型，对无限时间RL中策略价值进行了尖锐边界的刻画和估计。 |
| [^4] | [Do CLIPs Always Generalize Better than ImageNet Models?](https://arxiv.org/abs/2403.11497) | CLIP模型在面对分布转移时表现出良好的泛化能力，作者设计了CounterAnimal数据集来探究模型对虚假特征的依赖性。 |
| [^5] | [An SDP-based Branch-and-Cut Algorithm for Biclustering](https://arxiv.org/abs/2403.11351) | 提出了一个基于SDP的分支定界算法，用于解决$k$-最密不相交双团问题。 |
| [^6] | [Large-scale variational Gaussian state-space models](https://arxiv.org/abs/2403.01371) | 该论文介绍了一种针对具有高斯噪声驱动非线性动力学的状态空间模型的大规模变分算法和结构化逼近方法，可以有效评估ELBO和获取低方差的随机梯度估计，通过利用低秩蒙特卡罗逼近和推断网络的精度矩阵更新，将近似平滑问题转化为近似滤波问题。 |
| [^7] | [Near-Minimax-Optimal Distributional Reinforcement Learning with a Generative Model](https://arxiv.org/abs/2402.07598) | 本论文提出了一种基于生成模型的近最小极大分布式强化学习算法，该算法在使用生成模型近似回报分布方面具有极小极大优势，解决了一个开放问题，并提供了实验研究结果。 |
| [^8] | [Improving the Worst-Case Bidirectional Communication Complexity for Nonconvex Distributed Optimization under Function Similarity](https://arxiv.org/abs/2402.06412) | 本文提出了MARINA-P方法，通过引入一系列相关压缩器，优化了服务器到工作节点的通信复杂度。理论分析证明，MARINA-P在算法上优于现有方法，并可以作为支持双向压缩的起点。通过与上行压缩和动量步骤的结合，M3方法实现了双向压缩，并在总通信复杂度上改进。 |
| [^9] | [A Framework for Bilevel Optimization on Riemannian Manifolds](https://arxiv.org/abs/2402.03883) | 本论文提出了一个在黎曼流形上解决约束双层优化问题的框架，并提供了多种超梯度估计策略，并对其进行了研究。该框架不仅适用于确定性双层优化问题，还适用于随机双层优化问题，并且可以使用一般的回退。在各种应用中，该框架都具有很高的实用性。 |
| [^10] | [Consistency Models for Scalable and Fast Simulation-Based Inference](https://arxiv.org/abs/2312.05440) | 提出了一种新的神经后验估计的一致性模型，结合了标准化流和流匹配方法的优点，用于可扩展、快速和摊销推断，在多个实验中展示出优越性能。 |
| [^11] | [SASSL: Enhancing Self-Supervised Learning via Neural Style Transfer.](http://arxiv.org/abs/2312.01187) | SASSL提出了一种基于神经风格迁移的增强技术，通过解耦语义和风格属性，在自监督学习中生成多样化的增强样本，从而提升了图像分类性能。 |
| [^12] | [Sparse Bayesian Multidimensional Item Response Theory.](http://arxiv.org/abs/2310.17820) | 本文开发了一种可扩展的贝叶斯EM算法，用于从二元和有序项目响应中估计稀疏因子载荷，并通过贝叶斯非参数方法解决了未知潜在因子维度的问题。 |
| [^13] | [Quartile-Based Seasonality Decomposition for Time Series Forecasting and Anomaly Detection.](http://arxiv.org/abs/2306.05989) | 本文提出了一种名为QBSD的实时预测方法，以在时间序列异常检测中取得最佳平衡。 |
| [^14] | [Comparative Study on Semi-supervised Learning Applied for Anomaly Detection in Hydraulic Condition Monitoring System.](http://arxiv.org/abs/2306.02709) | 本研究比较了不同类型的半监督学习方法在液压状态监测系统中用于异常检测。深度学习模型表现最好，而集成模型可以进一步提高检测性能。 |
| [^15] | [DU-Shapley: A Shapley Value Proxy for Efficient Dataset Valuation.](http://arxiv.org/abs/2306.02071) | 本论文提出了一种称为DU-Shapley的方法，用于更有效地计算Shapley值，以实现机器学习中的数据集价值评估。 |

# 详细

[^1]: 低查询成本带噪声or同时聚类

    Query-Efficient Correlation Clustering with Noisy Oracle

    [https://rss.arxiv.org/abs/2402.01400](https://rss.arxiv.org/abs/2402.01400)

    本论文提出了一种低查询成本的聚类方法，利用纯在组合多臂赌博机探索范式实现在线学习，并设计了能在NP-hard情况下运行的多项式时间算法。

    

    我们研究了一个常见的聚类设置，其中我们需要对n个元素进行聚类，并且我们的目标是尽可能少地向返回两个元素相似性的有噪声的oracle查询。我们的设置涵盖了许多应用领域，在这些领域中，相似性函数计算起来成本高并且 inherently noisy。我们提出了两种基于纯在组合多臂赌博机探索范式(PE-CMAB)的在线学习问题的新颖表达方法固定置信度和固定预算设置。对于这两种设置，我们设计了将抽样策略与经典的相关聚类近似算法相结合的算法，并研究了它们的理论保证。我们的结果是这样的：这些算法是第一个在底层离线优化问题为NP-hard的情况下运行的多项式时间算法的例子。

    We study a general clustering setting in which we have $n$ elements to be clustered, and we aim to perform as few queries as possible to an oracle that returns a noisy sample of the similarity between two elements. Our setting encompasses many application domains in which the similarity function is costly to compute and inherently noisy. We propose two novel formulations of online learning problems rooted in the paradigm of Pure Exploration in Combinatorial Multi-Armed Bandits (PE-CMAB): fixed confidence and fixed budget settings. For both settings, we design algorithms that combine a sampling strategy with a classic approximation algorithm for correlation clustering and study their theoretical guarantees. Our results are the first examples of polynomial-time algorithms that work for the case of PE-CMAB in which the underlying offline optimization problem is NP-hard.
    
[^2]: 多项式神经网络的几何性质

    Geometry of Polynomial Neural Networks

    [https://rss.arxiv.org/abs/2402.00949](https://rss.arxiv.org/abs/2402.00949)

    本研究利用代数几何工具研究了具有单项式激活函数的多项式神经网络的表达性和学习过程，通过对神经流形的维度和学习度的研究，提供了网络表达能力和训练复杂度的度量，并给出了可学函数数量的上界。

    

    我们研究了具有单项式激活函数的多项式神经网络（PNN）的表达性和学习过程。网络的权重参数化了神经流形。在本文中，我们使用代数几何工具研究了某些神经流形：我们给出了半代数集的明确描述并特征化了它们的Zariski闭包，称为神经多样性。我们研究了它们的维度并将一个代数度量，学习度，与神经多样性相关联。维度作为网络表达能力的几何度量，学习度是训练网络的复杂度度量，并提供可学函数数量的上界。这些理论结果还伴随着实验证明。

    We study the expressivity and learning process for polynomial neural networks (PNNs) with monomial activation functions. The weights of the network parametrize the neuromanifold. In this paper, we study certain neuromanifolds using tools from algebraic geometry: we give explicit descriptions as semialgebraic sets and characterize their Zariski closures, called neurovarieties. We study their dimension and associate an algebraic degree, the learning degree, to the neurovariety. The dimension serves as a geometric measure for the expressivity of the network, the learning degree is a measure for the complexity of training the network and provides upper bounds on the number of learnable functions. These theoretical results are accompanied with experiments.
    
[^3]: 在强健马尔可夫决策过程中高效而尖锐的离线策略评估

    Efficient and Sharp Off-Policy Evaluation in Robust Markov Decision Processes

    [https://arxiv.org/abs/2404.00099](https://arxiv.org/abs/2404.00099)

    在对抗性环境中，本论文提出了一种可修改转移核密度的扰动模型，拓展了传统的边缘敏感性模型，对无限时间RL中策略价值进行了尖锐边界的刻画和估计。

    

    我们研究了在马尔可夫决策过程（MDP）中给定来自原始MDP的转移观察时，在最佳和最坏情况下评估策略，无论是在相同策略还是不同策略下。当存在历史和未来环境之间可能发生转变的可能性时，比如由于未测量的混杂、分布转移或对抗性环境。我们提出了一个扰动模型，可以将转移核密度修改至给定乘法因子或其倒数，这将经典的边际敏感性模型（MSM）扩展到无限时间 RL。我们描述了在这个模型下的策略价值的尖锐边界，即在给定来自原始MDP的转移观测时可能的最严格边界，我们研究了从这些转移观察中估计这些边界。我们开发了一个估计器，具有几个吸引人的特性。

    arXiv:2404.00099v1 Announce Type: new  Abstract: We study evaluating a policy under best- and worst-case perturbations to a Markov decision process (MDP), given transition observations from the original MDP, whether under the same or different policy. This is an important problem when there is the possibility of a shift between historical and future environments, due to e.g. unmeasured confounding, distributional shift, or an adversarial environment. We propose a perturbation model that can modify transition kernel densities up to a given multiplicative factor or its reciprocal, which extends the classic marginal sensitivity model (MSM) for single time step decision making to infinite-horizon RL. We characterize the sharp bounds on policy value under this model, that is, the tightest possible bounds given by the transition observations from the original MDP, and we study the estimation of these bounds from such transition observations. We develop an estimator with several appealing gua
    
[^4]: CLIP总是比ImageNet模型泛化更好吗？

    Do CLIPs Always Generalize Better than ImageNet Models?

    [https://arxiv.org/abs/2403.11497](https://arxiv.org/abs/2403.11497)

    CLIP模型在面对分布转移时表现出良好的泛化能力，作者设计了CounterAnimal数据集来探究模型对虚假特征的依赖性。

    

    大型视觉语言模型，例如CLIP，已经彻底改变了现代机器学习。CLIP展示了在分布转移下的良好泛化能力，得到了越来越多的文献支持。然而，CLIP的评估数据集主要是为ImageNet基准而设计的变种，可能不能完全反映CLIP在LAION等上进行预训练时对虚假相关性的稳健性。为了弥补这一差距，我们收集了一个真实世界数据集，名为CounterAnimal，其中包含动物照片中发现的现实虚假特征。CounterAnimal包括a）常见组：包括常见背景的动物，并且 b) 对照组：包括在不寻常背景下的动物。从常见组到对照组的性能下降量化了模型对虚假特征（即背景）预测动物的依赖性。我们发现，在LAION或OpenAI数据上进行训练的CLIP即没有

    arXiv:2403.11497v1 Announce Type: cross  Abstract: Large vision language models, such as CLIPs, have revolutionized modern machine learning. CLIPs have demonstrated great generalizability under distribution shifts, supported by an increasing body of literature. However, the evaluation datasets for CLIPs are variations primarily designed for ImageNet benchmarks, which may not fully reflect the extent to which CLIPs, e.g., pre-trained on LAION, robust to spurious correlations. To bridge the gap, we collect a real-world dataset called CounterAnimal that contains realistic spurious features found in animal photos. CounterAnimal consists of a) the common group: comprising animals on common backgrounds, and b) the counter group: including animals on unusual backgrounds. The performance drops from the common to counter groups quantify the reliance of models on spurious features (i.e., backgrounds) to predict the animals. We find that CLIPs trained on either LAION or the OpenAI data exhibit no
    
[^5]: 基于SDP的二分图聚类分支定界算法

    An SDP-based Branch-and-Cut Algorithm for Biclustering

    [https://arxiv.org/abs/2403.11351](https://arxiv.org/abs/2403.11351)

    提出了一个基于SDP的分支定界算法，用于解决$k$-最密不相交双团问题。

    

    二分图聚类，也称为共聚类、块聚类或双向聚类，涉及将数据矩阵的行和列同时聚类成不同的组，使得同一组内的行和列显示出相似的模式。作为二分图聚类的模型问题，我们考虑$k$-最密不相交双团问题，其目标是在给定加权完全二分图中识别 $k$ 个不相交的完全二部子图（称为双团），使它们的密度之和最大化。为了解决这个问题，我们提出了一个定制的分支定界算法。对于上界例程，我们考虑半定规划放松并提出了用于加强界限的有效不等式。我们使用一种一阶方法以切平面方式解决这个放松问题。对于下界，我们设计了一个利用解决方案的最大权匹配舍入过程。

    arXiv:2403.11351v1 Announce Type: cross  Abstract: Biclustering, also called co-clustering, block clustering, or two-way clustering, involves the simultaneous clustering of both the rows and columns of a data matrix into distinct groups, such that the rows and columns within a group display similar patterns. As a model problem for biclustering, we consider the $k$-densest-disjoint biclique problem, whose goal is to identify $k$ disjoint complete bipartite subgraphs (called bicliques) of a given weighted complete bipartite graph such that the sum of their densities is maximized. To address this problem, we present a tailored branch-and-cut algorithm. For the upper bound routine, we consider a semidefinite programming relaxation and propose valid inequalities to strengthen the bound. We solve this relaxation in a cutting-plane fashion using a first-order method. For the lower bound, we design a maximum weight matching rounding procedure that exploits the solution of the relaxation solved
    
[^6]: 大规模变分高斯状态空间模型

    Large-scale variational Gaussian state-space models

    [https://arxiv.org/abs/2403.01371](https://arxiv.org/abs/2403.01371)

    该论文介绍了一种针对具有高斯噪声驱动非线性动力学的状态空间模型的大规模变分算法和结构化逼近方法，可以有效评估ELBO和获取低方差的随机梯度估计，通过利用低秩蒙特卡罗逼近和推断网络的精度矩阵更新，将近似平滑问题转化为近似滤波问题。

    

    我们介绍了一种用于状态空间模型的嵌套变分推断算法和结构化变分逼近方法，其中非线性动力学由高斯噪声驱动。值得注意的是，所提出的框架允许在没有采用对角高斯逼近的情况下有效地评估ELBO和低方差随机梯度估计，通过利用（i）通过动力学对隐状态进行边缘化的蒙特卡罗逼近的低秩结构，（ii）一个推断网络，该网络通过低秩精度矩阵更新来近似更新步骤，（iii）将当前和未来观测编码为伪观测--将近似平滑问题转换为（更简单的）近似滤波问题。整体而言，必要的统计信息和ELBO可以在$O（TL（Sr+S^2+r^2））$时间内计算，其中$T$是系列长度，$L$是状态空间维数，$S$是用于逼近的样本数量。

    arXiv:2403.01371v1 Announce Type: cross  Abstract: We introduce an amortized variational inference algorithm and structured variational approximation for state-space models with nonlinear dynamics driven by Gaussian noise. Importantly, the proposed framework allows for efficient evaluation of the ELBO and low-variance stochastic gradient estimates without resorting to diagonal Gaussian approximations by exploiting (i) the low-rank structure of Monte-Carlo approximations to marginalize the latent state through the dynamics (ii) an inference network that approximates the update step with low-rank precision matrix updates (iii) encoding current and future observations into pseudo observations -- transforming the approximate smoothing problem into an (easier) approximate filtering problem. Overall, the necessary statistics and ELBO can be computed in $O(TL(Sr + S^2 + r^2))$ time where $T$ is the series length, $L$ is the state-space dimensionality, $S$ are the number of samples used to app
    
[^7]: 基于生成模型的近最小极大分布式强化学习算法

    Near-Minimax-Optimal Distributional Reinforcement Learning with a Generative Model

    [https://arxiv.org/abs/2402.07598](https://arxiv.org/abs/2402.07598)

    本论文提出了一种基于生成模型的近最小极大分布式强化学习算法，该算法在使用生成模型近似回报分布方面具有极小极大优势，解决了一个开放问题，并提供了实验研究结果。

    

    我们提出了一种新的基于模型的分布式强化学习算法，并证明了在使用生成模型近似回报分布方面，它是近似最小极大的（在对数因子上），从而解决了Zhang等人（2023）的一个开放问题。我们的分析为分布式强化学习中的分类方法提供了新的理论结果，并引入了一种新的分布式Bellman方程，即随机分类累积分布函数Bellman方程，我们认为这个方程也具有独立的研究意义。我们还进行了实验研究，比较了几种基于模型的分布式强化学习算法，并得出了对实践者有意义的几个结论。

    We propose a new algorithm for model-based distributional reinforcement learning (RL), and prove that it is minimax-optimal for approximating return distributions with a generative model (up to logarithmic factors), resolving an open question of Zhang et al. (2023). Our analysis provides new theoretical results on categorical approaches to distributional RL, and also introduces a new distributional Bellman equation, the stochastic categorical CDF Bellman equation, which we expect to be of independent interest. We also provide an experimental study comparing several model-based distributional RL algorithms, with several takeaways for practitioners.
    
[^8]: 提高非凸分布式优化在函数相似性下的最坏情况双向通信复杂性

    Improving the Worst-Case Bidirectional Communication Complexity for Nonconvex Distributed Optimization under Function Similarity

    [https://arxiv.org/abs/2402.06412](https://arxiv.org/abs/2402.06412)

    本文提出了MARINA-P方法，通过引入一系列相关压缩器，优化了服务器到工作节点的通信复杂度。理论分析证明，MARINA-P在算法上优于现有方法，并可以作为支持双向压缩的起点。通过与上行压缩和动量步骤的结合，M3方法实现了双向压缩，并在总通信复杂度上改进。

    

    服务器和工作节点之间的有效通信在分布式优化中起着关键作用。本文主要关注优化服务器到工作节点的通信，并揭示了当前流行的下行压缩方法中的低效性。首先考虑上行通信成本可忽略的纯粹情况下，我们引入MARINA-P，一种使用一系列相关压缩器的新型下行压缩方法。理论分析证明，使用排列压缩器的MARINA-P可以实现服务器到工作节点的通信复杂度随工作节点数量提高，因此在算法上可证明优于现有算法。我们进一步展示了MARINA-P可以作为支持双向压缩的方法的起点。我们介绍了M3，这是一种将MARINA-P与上行压缩和动量步骤组合的方法，能够实现双向压缩，并在总通信复杂度上证明了改进。

    Effective communication between the server and workers plays a key role in distributed optimization. In this paper, we focus on optimizing the server-to-worker communication, uncovering inefficiencies in prevalent downlink compression approaches. Considering first the pure setup where the uplink communication costs are negligible, we introduce MARINA-P, a novel method for downlink compression, employing a collection of correlated compressors. Theoretical analyses demonstrates that MARINA-P with permutation compressors can achieve a server-to-worker communication complexity improving with the number of workers, thus being provably superior to existing algorithms. We further show that MARINA-P can serve as a starting point for extensions such as methods supporting bidirectional compression. We introduce M3, a method combining MARINA-P with uplink compression and a momentum step, achieving bidirectional compression with provable improvements in total communication complexity as the number
    
[^9]: 在黎曼流形上进行双层优化的框架

    A Framework for Bilevel Optimization on Riemannian Manifolds

    [https://arxiv.org/abs/2402.03883](https://arxiv.org/abs/2402.03883)

    本论文提出了一个在黎曼流形上解决约束双层优化问题的框架，并提供了多种超梯度估计策略，并对其进行了研究。该框架不仅适用于确定性双层优化问题，还适用于随机双层优化问题，并且可以使用一般的回退。在各种应用中，该框架都具有很高的实用性。

    

    双层优化在各个领域的应用中越来越常见。在这项工作中，我们提出了一个在黎曼流形上约束双层优化问题变量的框架。我们提供了几种在流形上的超梯度估计策略，并研究了它们的估计误差。我们对流形上的超梯度下降算法提供了收敛性和复杂性分析。我们还将这些研究扩展到随机双层优化和使用一般的回退。我们展示了该框架在各种应用中的实用性。

    Bilevel optimization has seen an increasing presence in various domains of applications. In this work, we propose a framework for solving bilevel optimization problems where variables of both lower and upper level problems are constrained on Riemannian manifolds. We provide several hypergradient estimation strategies on manifolds and study their estimation error. We provide convergence and complexity analysis for the proposed hypergradient descent algorithm on manifolds. We also extend the developments to stochastic bilevel optimization and to the use of general retraction. We showcase the utility of the proposed framework on various applications.
    
[^10]: 可扩展和快速模拟推断的一致性模型

    Consistency Models for Scalable and Fast Simulation-Based Inference

    [https://arxiv.org/abs/2312.05440](https://arxiv.org/abs/2312.05440)

    提出了一种新的神经后验估计的一致性模型，结合了标准化流和流匹配方法的优点，用于可扩展、快速和摊销推断，在多个实验中展示出优越性能。

    

    仿真推断（SBI）不断寻找更具表现力的算法，以准确推断复杂模型的参数从嘈杂数据中。我们提出了神经后验估计的一致性模型（CMPE），这是一个用于可扩展、快速和摊销推断的新自由形式条件采样器，利用生成性神经网络。CMPE将标准化流和流匹配方法的优点结合到单个生成架构中：它本质上提炼了连续概率流，并能够利用无约束的结构快速进行少射推断，该结构可以定制到估计问题的结构。我们的实证评估表明，CMPE不仅在三个困难的低维问题上优于当前的最先进算法，而且在高维贝叶斯去噪实验和估计计算密集型多尺度中表现出有竞争力的性能。

    arXiv:2312.05440v2 Announce Type: replace-cross  Abstract: Simulation-based inference (SBI) is constantly in search of more expressive algorithms for accurately inferring the parameters of complex models from noisy data. We present consistency models for neural posterior estimation (CMPE), a new free-form conditional sampler for scalable, fast, and amortized SBI with generative neural networks. CMPE combines the advantages of normalizing flows and flow matching methods into a single generative architecture: It essentially distills a continuous probability flow and enables rapid few-shot inference with an unconstrained architecture that can be tailored to the structure of the estimation problem. Our empirical evaluation demonstrates that CMPE not only outperforms current state-of-the-art algorithms on three hard low-dimensional problems but also achieves competitive performance in a high-dimensional Bayesian denoising experiment and in estimating a computationally demanding multi-scale 
    
[^11]: SASSL:通过神经风格迁移增强自监督学习

    SASSL: Enhancing Self-Supervised Learning via Neural Style Transfer. (arXiv:2312.01187v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.01187](http://arxiv.org/abs/2312.01187)

    SASSL提出了一种基于神经风格迁移的增强技术，通过解耦语义和风格属性，在自监督学习中生成多样化的增强样本，从而提升了图像分类性能。

    

    自监督学习依赖于数据增强来从无标签图像中提取有意义的表征。现有的最先进的增强流水线包括了各种原始的转换，但通常忽略了自然图像的结构。因此，增强样本可能显示出退化的语义信息和低风格多样性，从而影响到自监督表征的下游性能。为了克服这个问题，我们提出了一种名为SASSL的新型增强技术，它基于神经风格迁移。该方法将图像中的语义和风格属性解耦，并仅对风格应用转换，保持内容，生成多样化的增强样本，更好地保留它们的语义属性。实验结果显示，与广为接受的MoCo v2相比，我们的技术在ImageNet上的top-1分类性能提升超过2%。

    Self-supervised learning relies heavily on data augmentation to extract meaningful representations from unlabeled images. While existing state-of-the-art augmentation pipelines incorporate a wide range of primitive transformations, these often disregard natural image structure. Thus, augmented samples can exhibit degraded semantic information and low stylistic diversity, affecting downstream performance of self-supervised representations. To overcome this, we propose SASSL: Style Augmentations for Self Supervised Learning, a novel augmentation technique based on Neural Style Transfer. The method decouples semantic and stylistic attributes in images and applies transformations exclusively to the style while preserving content, generating diverse augmented samples that better retain their semantic properties. Experimental results show our technique achieves a top-1 classification performance improvement of more than 2% on ImageNet compared to the well-established MoCo v2. We also measure
    
[^12]: 稀疏贝叶斯多维项目反应理论

    Sparse Bayesian Multidimensional Item Response Theory. (arXiv:2310.17820v1 [stat.ME])

    [http://arxiv.org/abs/2310.17820](http://arxiv.org/abs/2310.17820)

    本文开发了一种可扩展的贝叶斯EM算法，用于从二元和有序项目响应中估计稀疏因子载荷，并通过贝叶斯非参数方法解决了未知潜在因子维度的问题。

    

    多变量项目反应理论（MIRT）被应用研究人员广泛使用，以寻找问卷数据中响应模式背后的可解释（稀疏）解释。然而，在实践中，对于这种稀疏性发现工具的需求尚未得到满足。本文提出了一种用于二元和有序项目MIRT的贝叶斯平台，其需要最少的调整，并且由于其可并行化的特性，在相对较大的数据集上具有良好的可扩展性。MIRT模型的贝叶斯方法传统上依赖于MCMC模拟，在实践中可能既费时又难以通过额外的阈值设定实现精确的稀疏恢复。在本文中，我们开发了一种可扩展的贝叶斯EM算法，用于从二元和有序项目响应中估计稀疏因子载荷。我们利用贝叶斯非参数方法解决了未知潜在因子维度的看似不可逾越的问题，从而使得可以估计因子的数量。通过旋转可以实现稀疏性。

    Multivariate Item Response Theory (MIRT) is sought-after widely by applied researchers looking for interpretable (sparse) explanations underlying response patterns in questionnaire data. There is, however, an unmet demand for such sparsity discovery tools in practice. Our paper develops a Bayesian platform for binary and ordinal item MIRT which requires minimal tuning and scales well on relatively large datasets due to its parallelizable features. Bayesian methodology for MIRT models has traditionally relied on MCMC simulation, which cannot only be slow in practice, but also often renders exact sparsity recovery impossible without additional thresholding. In this work, we develop a scalable Bayesian EM algorithm to estimate sparse factor loadings from binary and ordinal item responses. We address the seemingly insurmountable problem of unknown latent factor dimensionality with tools from Bayesian nonparametrics which enable estimating the number of factors. Rotations to sparsity throug
    
[^13]: 基于四分位数的季节性分解用于时间序列预测和异常检测

    Quartile-Based Seasonality Decomposition for Time Series Forecasting and Anomaly Detection. (arXiv:2306.05989v1 [cs.LG])

    [http://arxiv.org/abs/2306.05989](http://arxiv.org/abs/2306.05989)

    本文提出了一种名为QBSD的实时预测方法，以在时间序列异常检测中取得最佳平衡。

    

    在电信领域，及时检测异常非常重要，因为这有助于识别和表征不规则模式、异常行为和网络异常，从而提高服务质量和操作效率。精确地预测和消除可预测的时间序列模式是时间序列异常检测的重要组成部分。本文提出了一种名为基于四分位数的季节性分解（QBSD）的实时预测方法，以在计算复杂度和预测准确率之间取得最佳平衡。本文比较了QBSD与现有预测方法的性能及其适用性。

    The timely detection of anomalies is essential in the telecom domain as it facilitates the identification and characterization of irregular patterns, abnormal behaviors, and network anomalies, contributing to enhanced service quality and operational efficiency. Precisely forecasting and eliminating predictable time series patterns constitutes a vital component of time series anomaly detection. While the state-of-the-art methods aim to maximize forecasting accuracy, the computational performance takes a hit. In a system composed of a large number of time series variables, e.g., cell Key Performance Indicators (KPIs), the time and space complexity of the forecasting employed is of crucial importance. Quartile-Based Seasonality Decomposition (QBSD) is a live forecasting method proposed in this paper to make an optimal trade-off between computational complexity and forecasting accuracy. This paper compares the performance of QBSD to the state-of-the-art forecasting methods and their applic
    
[^14]: 用于液压状态监测系统异常检测的半监督学习比较研究

    Comparative Study on Semi-supervised Learning Applied for Anomaly Detection in Hydraulic Condition Monitoring System. (arXiv:2306.02709v1 [cs.LG])

    [http://arxiv.org/abs/2306.02709](http://arxiv.org/abs/2306.02709)

    本研究比较了不同类型的半监督学习方法在液压状态监测系统中用于异常检测。深度学习模型表现最好，而集成模型可以进一步提高检测性能。

    

    基于状态的维护在液压系统中变得越来越重要。然而，这些系统的异常检测仍然具有挑战性，特别是由于异常数据很少，标记这些数据是费时费力甚至危险的。因此，建议使用无监督或半监督方法，特别是对于只有少量标签可用的情况下利用无监督学习作为特征提取机制来辅助监督学习的半监督学习方法。本研究系统地比较了在液压状态监测系统中应用的半监督学习方法用于异常检测。首先，进行了深入的数据分析和特征学习，以了解开源的液压状态监测数据集。然后，实施和评估了各种方法，包括传统的独立半监督学习模型（例如，一类支持向量机、鲁棒协方差）、集成模型（例如，孤立森林）和基于深度学习的模型（例如，自动编码器、图卷积网络）。结果表明，深度学习模型优于传统模型，而集成模型可以进一步提高检测性能。

    Condition-based maintenance is becoming increasingly important in hydraulic systems. However, anomaly detection for these systems remains challenging, especially since that anomalous data is scarce and labeling such data is tedious and even dangerous. Therefore, it is advisable to make use of unsupervised or semi-supervised methods, especially for semi-supervised learning which utilizes unsupervised learning as a feature extraction mechanism to aid the supervised part when only a small number of labels are available. This study systematically compares semi-supervised learning methods applied for anomaly detection in hydraulic condition monitoring systems. Firstly, thorough data analysis and feature learning were carried out to understand the open-sourced hydraulic condition monitoring dataset. Then, various methods were implemented and evaluated including traditional stand-alone semi-supervised learning models (e.g., one-class SVM, Robust Covariance), ensemble models (e.g., Isolation F
    
[^15]: DU-Shapley: 一种有效的数据集价值评估的Shapley值代理

    DU-Shapley: A Shapley Value Proxy for Efficient Dataset Valuation. (arXiv:2306.02071v1 [cs.AI])

    [http://arxiv.org/abs/2306.02071](http://arxiv.org/abs/2306.02071)

    本论文提出了一种称为DU-Shapley的方法，用于更有效地计算Shapley值，以实现机器学习中的数据集价值评估。

    

    许多机器学习问题需要进行数据集评估，即量化将一个单独的数据集与其他数据集聚合的增量收益，以某些相关预定义公用事业为基础。最近，Shapley值被提出作为实现这一目标的一种基本工具，因为它具有形式公理证明。由于其计算通常需要指数时间，因此考虑基于Monte Carlo积分的标准近似策略。然而，在某些情况下，这种通用近似方法仍然昂贵。本文利用数据集评估问题的结构知识，设计了更有效的Shapley值估计器。我们提出了一种新的Shapley值近似，称为离散均匀Shapley (DU-Shapley)，其表达为期望值

    Many machine learning problems require performing dataset valuation, i.e. to quantify the incremental gain, to some relevant pre-defined utility, of aggregating an individual dataset to others. As seminal examples, dataset valuation has been leveraged in collaborative and federated learning to create incentives for data sharing across several data owners. The Shapley value has recently been proposed as a principled tool to achieve this goal due to formal axiomatic justification. Since its computation often requires exponential time, standard approximation strategies based on Monte Carlo integration have been considered. Such generic approximation methods, however, remain expensive in some cases. In this paper, we exploit the knowledge about the structure of the dataset valuation problem to devise more efficient Shapley value estimators. We propose a novel approximation of the Shapley value, referred to as discrete uniform Shapley (DU-Shapley) which is expressed as an expectation under 
    

