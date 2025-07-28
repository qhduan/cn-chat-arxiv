# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Estimation of conditional average treatment effects on distributed data: A privacy-preserving approach](https://arxiv.org/abs/2402.02672) | 本论文提出了一种数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计条件平均治疗效果（CATE）模型。通过数值实验验证了该方法的有效性。该方法的三个主要贡献是：实现了对分布式数据上的非迭代通信的半参数CATE模型的估计和测试，提高了模型的鲁棒性。 |
| [^2] | [Noise Contrastive Estimation-based Matching Framework for Low-resource Security Attack Pattern Recognition.](http://arxiv.org/abs/2401.10337) | 该论文提出了一种基于噪声对比估计的低资源安全攻击模式识别匹配框架，通过直接语义相似度决定文本与攻击模式之间的关联，以降低大量类别、标签分布不均和标签空间复杂性带来的学习难度。 |
| [^3] | [Disentangled Latent Spaces Facilitate Data-Driven Auxiliary Learning.](http://arxiv.org/abs/2310.09278) | 本论文提出了一个新的框架，通过解耦过程来发现可以与主任务一起利用的不相关的分类任务和相关标签，从而在深度学习中促进辅助学习。 |
| [^4] | [Bounded KRnet and its applications to density estimation and approximation.](http://arxiv.org/abs/2305.09063) | 本文介绍了一种新的可逆映射B-KRnet，并将其应用于数据或PDE的密度估计/近似，由于其定义在有界域上，因此比KRnet更有效。 |
| [^5] | [Doubly Regularized Entropic Wasserstein Barycenters.](http://arxiv.org/abs/2303.11844) | 本文提出了一种双重正则化熵Wasserstein重心公式，具有好的正则化、逼近、稳定性和（无网格）优化特性; 其中，只有在$\tau=\lambda/2$的情况下是无偏差的。 |

# 详细

[^1]: 对分布式数据的条件平均治疗效果估计：一种保护隐私的方法

    Estimation of conditional average treatment effects on distributed data: A privacy-preserving approach

    [https://arxiv.org/abs/2402.02672](https://arxiv.org/abs/2402.02672)

    本论文提出了一种数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计条件平均治疗效果（CATE）模型。通过数值实验验证了该方法的有效性。该方法的三个主要贡献是：实现了对分布式数据上的非迭代通信的半参数CATE模型的估计和测试，提高了模型的鲁棒性。

    

    在医学和社会科学等各个领域中，对条件平均治疗效果（CATEs）的估计是一个重要的课题。如果分布在多个参与方之间的数据可以集中，可以对CATEs进行高精度的估计。然而，如果这些数据包含隐私信息，则很难进行数据聚合。为了解决这个问题，我们提出了数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计CATE模型，并通过数值实验对该方法进行了评估。我们的贡献总结如下三点。首先，我们的方法能够在分布式数据上进行非迭代通信的半参数CATE模型的估计和测试。半参数或非参数的CATE模型能够比参数模型更稳健地进行估计和测试，对于模型偏差的鲁棒性更强。然而，据我们所知，目前还没有提出有效的通信方法来估计和测试这些模型。

    Estimation of conditional average treatment effects (CATEs) is an important topic in various fields such as medical and social sciences. CATEs can be estimated with high accuracy if distributed data across multiple parties can be centralized. However, it is difficult to aggregate such data if they contain privacy information. To address this issue, we proposed data collaboration double machine learning (DC-DML), a method that can estimate CATE models with privacy preservation of distributed data, and evaluated the method through numerical experiments. Our contributions are summarized in the following three points. First, our method enables estimation and testing of semi-parametric CATE models without iterative communication on distributed data. Semi-parametric or non-parametric CATE models enable estimation and testing that is more robust to model mis-specification than parametric models. However, to our knowledge, no communication-efficient method has been proposed for estimating and 
    
[^2]: 基于噪声对比估计的低资源安全攻击模式识别匹配框架

    Noise Contrastive Estimation-based Matching Framework for Low-resource Security Attack Pattern Recognition. (arXiv:2401.10337v1 [cs.LG])

    [http://arxiv.org/abs/2401.10337](http://arxiv.org/abs/2401.10337)

    该论文提出了一种基于噪声对比估计的低资源安全攻击模式识别匹配框架，通过直接语义相似度决定文本与攻击模式之间的关联，以降低大量类别、标签分布不均和标签空间复杂性带来的学习难度。

    

    战术、技术和程序（TTPs）是网络安全领域中复杂的攻击模式，在文本知识库中有详细的描述。在网络安全写作中识别TTPs，通常称为TTP映射，是一个重要而具有挑战性的任务。传统的学习方法通常以经典的多类或多标签分类设置为目标。由于存在大量的类别（即TTPs），标签分布的不均衡和标签空间的复杂层次结构，这种设置限制了模型的学习能力。我们采用了一种不同的学习范式来解决这个问题，其中将文本与TTP标签之间的直接语义相似度决定为文本分配给TTP标签，从而减少了仅仅在大型标签空间上竞争的复杂性。为此，我们提出了一种具有有效的基于采样的学习比较机制的神经匹配架构，促进学习过程。

    Tactics, Techniques and Procedures (TTPs) represent sophisticated attack patterns in the cybersecurity domain, described encyclopedically in textual knowledge bases. Identifying TTPs in cybersecurity writing, often called TTP mapping, is an important and challenging task. Conventional learning approaches often target the problem in the classical multi-class or multilabel classification setting. This setting hinders the learning ability of the model due to a large number of classes (i.e., TTPs), the inevitable skewness of the label distribution and the complex hierarchical structure of the label space. We formulate the problem in a different learning paradigm, where the assignment of a text to a TTP label is decided by the direct semantic similarity between the two, thus reducing the complexity of competing solely over the large labeling space. To that end, we propose a neural matching architecture with an effective sampling-based learn-to-compare mechanism, facilitating the learning pr
    
[^3]: 解耦潜在空间促进数据驱动的辅助学习

    Disentangled Latent Spaces Facilitate Data-Driven Auxiliary Learning. (arXiv:2310.09278v1 [cs.LG])

    [http://arxiv.org/abs/2310.09278](http://arxiv.org/abs/2310.09278)

    本论文提出了一个新的框架，通过解耦过程来发现可以与主任务一起利用的不相关的分类任务和相关标签，从而在深度学习中促进辅助学习。

    

    在深度学习中，辅助目标常常被用来在数据稀缺或者主要任务非常复杂的情况下促进学习。这个想法主要受到同时解决多个任务带来的改进泛化能力的启发，从而产生更强大的共享表示。然而，找到能产生期望改进的最优辅助任务是一个关键问题，通常需要手动设计的技巧或者昂贵的元学习方法。本文提出了一个新颖的框架，称为Detaux，通过弱监督的解耦过程在任何多任务学习（MTL）模型中发现可以与主要任务一起利用的不相关的分类任务和相关标签。解耦过程在表示层面工作，将与主要任务相关的一个子空间与任意数量的正交子空间分离开来。

    In deep learning, auxiliary objectives are often used to facilitate learning in situations where data is scarce, or the principal task is extremely complex. This idea is primarily inspired by the improved generalization capability induced by solving multiple tasks simultaneously, which leads to a more robust shared representation. Nevertheless, finding optimal auxiliary tasks that give rise to the desired improvement is a crucial problem that often requires hand-crafted solutions or expensive meta-learning approaches. In this paper, we propose a novel framework, dubbed Detaux, whereby a weakly supervised disentanglement procedure is used to discover new unrelated classification tasks and the associated labels that can be exploited with the principal task in any Multi-Task Learning (MTL) model. The disentanglement procedure works at a representation level, isolating a subspace related to the principal task, plus an arbitrary number of orthogonal subspaces. In the most disentangled subsp
    
[^4]: 有界KRnet及其在密度估计和近似中的应用

    Bounded KRnet and its applications to density estimation and approximation. (arXiv:2305.09063v1 [cs.LG])

    [http://arxiv.org/abs/2305.09063](http://arxiv.org/abs/2305.09063)

    本文介绍了一种新的可逆映射B-KRnet，并将其应用于数据或PDE的密度估计/近似，由于其定义在有界域上，因此比KRnet更有效。

    

    本文在有界域上开发了一种可逆映射，称为B-KRnet，并将其应用于数据或PDE（例如福克-普朗克方程和Keller-Segel方程）的密度估计/近似。与KRnet类似，B-KRnet的结构将Knothe-Rosenblatt重排的三角形形式转化为归一化流模型。B-KRnet和KRnet之间的主要区别是B-KRnet定义在超立方体上，而KRnet定义在整个空间上，换句话说，我们在B-KRnet中引入了一种新的机制来保持精确的可逆性。将B-KRnet用作传输映射，我们获得了一个明确的概率密度函数（PDF）模型，该模型对应于先验（均匀）分布在超立方体上的推移。为了近似计算域上定义的PDF，B-KRnet比KRnet更有效。通过耦合KRnet和B-KRnet，我们还可以在高维域上定义一个深度生成模型。

    In this paper, we develop an invertible mapping, called B-KRnet, on a bounded domain and apply it to density estimation/approximation for data or the solutions of PDEs such as the Fokker-Planck equation and the Keller-Segel equation. Similar to KRnet, the structure of B-KRnet adapts the triangular form of the Knothe-Rosenblatt rearrangement into a normalizing flow model. The main difference between B-KRnet and KRnet is that B-KRnet is defined on a hypercube while KRnet is defined on the whole space, in other words, we introduce a new mechanism in B-KRnet to maintain the exact invertibility. Using B-KRnet as a transport map, we obtain an explicit probability density function (PDF) model that corresponds to the pushforward of a prior (uniform) distribution on the hypercube. To approximate PDFs defined on a bounded computational domain, B-KRnet is more effective than KRnet. By coupling KRnet and B-KRnet, we can also define a deep generative model on a high-dimensional domain where some di
    
[^5]: 双重正则化熵 Wasserstein 重心

    Doubly Regularized Entropic Wasserstein Barycenters. (arXiv:2303.11844v1 [math.OC])

    [http://arxiv.org/abs/2303.11844](http://arxiv.org/abs/2303.11844)

    本文提出了一种双重正则化熵Wasserstein重心公式，具有好的正则化、逼近、稳定性和（无网格）优化特性; 其中，只有在$\tau=\lambda/2$的情况下是无偏差的。

    

    我们研究了一种常规的正则化Wasserstein重心的公式，这个公式具有良好的正则化、逼近、稳定性和（无网格）优化特性。这个重心被定义为唯一一种最小化关于一族给定概率测度的熵最优输运（EOT）成本之和及熵项的概率测度。我们称之为$(\lambda,\tau)$-重心，其中，$\lambda$ 是内部正则化强度，$\tau$ 是外部正则化强度。这种公式恢复了已经提出的多种EOT重心，适合于不同的 $\lambda, \tau \geq 0$ 选择，并对它们进行了泛化。首先，尽管具有双重正则化，但在$\tau=\lambda/2$ 的情况下，我们证明了我们的公式是无偏的: 对于光滑密度，（未正则化的）Wasserstein 重心目标函数中的次优性是熵正则化强度$\lambda^2$的，而不是一般情况下的$\max \{\lambda, \tau\}$。

    We study a general formulation of regularized Wasserstein barycenters that enjoys favorable regularity, approximation, stability and (grid-free) optimization properties. This barycenter is defined as the unique probability measure that minimizes the sum of entropic optimal transport (EOT) costs with respect to a family of given probability measures, plus an entropy term. We denote it $(\lambda,\tau)$-barycenter, where $\lambda$ is the inner regularization strength and $\tau$ the outer one. This formulation recovers several previously proposed EOT barycenters for various choices of $\lambda,\tau \geq 0$ and generalizes them. First, in spite of -- and in fact owing to -- being \emph{doubly} regularized, we show that our formulation is debiased for $\tau=\lambda/2$: the suboptimality in the (unregularized) Wasserstein barycenter objective is, for smooth densities, of the order of the strength $\lambda^2$ of entropic regularization, instead of $\max\{\lambda,\tau\}$ in general. We discuss 
    

