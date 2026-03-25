# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Codesign of Scheduling and Parallelization for Large Model Training in Heterogeneous Clusters](https://arxiv.org/abs/2403.16125) | 在异构集群中，本文提出了Crius，一个用于有效调度多个大型模型的训练系统，引入了称为"Cell"的新调度粒度，以解决集群调度中低开销和准确性能数据获取之间的矛盾。 |
| [^2] | [MLFEF: Machine Learning Fusion Model with Empirical Formula to Explore the Momentum in Competitive Sports](https://arxiv.org/abs/2402.12149) | 本文旨在定义和量化动量，为网球比赛的实时分析提供基础，通过建立基于数据驱动和基于经验公式的模型，采用多种机器学习算法进行融合，以探索竞技体育中的动量。 |
| [^3] | [All you need is spin: SU(2) equivariant variational quantum circuits based on spin networks.](http://arxiv.org/abs/2309.07250) | 本文提出使用自旋网络构建SU(2)等价量子电路，通过编码群结构来限制优化空间， 具有旋转对称性，比其他已知的构造更直接实现在量子硬件上。 |
| [^4] | [Near-Optimal Nonconvex-Strongly-Convex Bilevel Optimization with Fully First-Order Oracles.](http://arxiv.org/abs/2306.14853) | 本文针对双层优化问题中底层问题为强凸的情况，提出了一种更加高效的算法，可以以接近最优的速率收敛。这种算法避免了在实践中可能无法获得或代价昂贵的Hessian-向量乘积预言机的使用。 |
| [^5] | [HD-Bind: Encoding of Molecular Structure with Low Precision, Hyperdimensional Binary Representations.](http://arxiv.org/abs/2303.15604) | HD-Bind是一种用于表示分子结构的超高维度二进制向量的方法，能够预测蛋白质中小分子的分子间结合亲和力。它显著减少了存储和计算所需的位数，并在药物发现中优于现有的方法。 |
| [^6] | [AttentionMixer: An Accurate and Interpretable Framework for Process Monitoring.](http://arxiv.org/abs/2302.10426) | AttentionMixer是一个旨在为能量变换厂建立准确且可解释的辐射监测框架的数据驱动方法，其技术创新点为空间和时间自适应消息传递块和注意力和可视化技术。 |

# 详细

[^1]: 一个用于异构集群中大型模型训练的调度和并行化代码设计

    A Codesign of Scheduling and Parallelization for Large Model Training in Heterogeneous Clusters

    [https://arxiv.org/abs/2403.16125](https://arxiv.org/abs/2403.16125)

    在异构集群中，本文提出了Crius，一个用于有效调度多个大型模型的训练系统，引入了称为"Cell"的新调度粒度，以解决集群调度中低开销和准确性能数据获取之间的矛盾。

    

    考虑调度和自适应并行性结合，为提高异构GPU集群上大型模型训练效率提供了巨大机遇。然而，将自适应并行性整合到集群调度器中扩展了集群调度空间。新空间是原始调度空间和自适应并行性的并行探索空间的乘积（还包括流水线，并行、数据并行和张量并行）。指数级扩大的调度空间和自适应并行性中不断变化的最佳并行计划共同导致了低开销和准确性能数据获取之间的矛盾，以实现高效的集群调度。

    arXiv:2403.16125v1 Announce Type: cross  Abstract: Joint consideration of scheduling and adaptive parallelism offers great opportunities for improving the training efficiency of large models on heterogeneous GPU clusters. However, integrating adaptive parallelism into a cluster scheduler expands the cluster scheduling space. The new space is the product of the original scheduling space and the parallelism exploration space of adaptive parallelism (also a product of pipeline, data, and tensor parallelism). The exponentially enlarged scheduling space and ever-changing optimal parallelism plan from adaptive parallelism together result in the contradiction between low-overhead and accurate performance data acquisition for efficient cluster scheduling. This paper presents Crius, a training system for efficiently scheduling multiple large models with adaptive parallelism in a heterogeneous cluster. Crius proposes a novel scheduling granularity called Cell. It represents a job with determinis
    
[^2]: MLFEF: 采用经验公式的机器学习融合模型探索竞技体育中的动量

    MLFEF: Machine Learning Fusion Model with Empirical Formula to Explore the Momentum in Competitive Sports

    [https://arxiv.org/abs/2402.12149](https://arxiv.org/abs/2402.12149)

    本文旨在定义和量化动量，为网球比赛的实时分析提供基础，通过建立基于数据驱动和基于经验公式的模型，采用多种机器学习算法进行融合，以探索竞技体育中的动量。

    

    网球非常受欢迎，教练和运动员对除了技能之外的因素，如动量，也感到好奇。本文将尝试定义和量化动量，为网球比赛的实时分析提供基础。基于近年来网球大满贯男子单打比赛的数据，我们构建了两个模型，一个是基于数据驱动的模型，另一个是基于经验公式的模型。对于数据驱动模型，我们首先找到了大量的公开数据，包括过去五年网球比赛的公开数据和球员的个人信息数据。然后对数据进行预处理和特征工程处理，并建立了一个SVM、Random Forrest算法和XGBoost的融合模型。对于机制分析模型，基于许多网球运动员和爱好者的建议，选择了重要特征，使用滑动窗口算法计算权重，和不同的met

    arXiv:2402.12149v1 Announce Type: new  Abstract: Tennis is so popular that coaches and players are curious about factors other than skill, such as momentum. This article will try to define and quantify momentum, providing a basis for real-time analysis of tennis matches. Based on the tennis Grand Slam men's singles match data in recent years, we built two models, one is to build a model based on data-driven, and the other is to build a model based on empirical formulas. For the data-driven model, we first found a large amount of public data including public data on tennis matches in the past five years and personal information data of players. Then the data is preprocessed, and feature engineered, and a fusion model of SVM, Random Forrest algorithm and XGBoost was established. For the mechanism analysis model, important features were selected based on the suggestions of many tennis players and enthusiasts, the sliding window algorithm was used to calculate the weight, and different met
    
[^3]: 你所需要的只是旋转：基于自旋网络的SU(2)等价变分量子电路

    All you need is spin: SU(2) equivariant variational quantum circuits based on spin networks. (arXiv:2309.07250v1 [quant-ph])

    [http://arxiv.org/abs/2309.07250](http://arxiv.org/abs/2309.07250)

    本文提出使用自旋网络构建SU(2)等价量子电路，通过编码群结构来限制优化空间， 具有旋转对称性，比其他已知的构造更直接实现在量子硬件上。

    

    变分算法要求将优化空间自然地限制在一个有效的范围内。在几何量子机器学习中，将群结构编码到参数化的量子电路中，将问题的对称性作为归纳偏置来考虑，可以实现这一点。然而，构建这样的电路是有挑战性的，因为尚未出现明确的指导原则。在本文中，我们提出使用自旋网络，一种在群变换下保持不变的有向张量网络形式，来设计SU(2)等价量子电路ansatz - 具有旋转对称性的电路。通过改变使SU(2)群作用块对角化的基础，这些网络为构建参数化等价量子电路提供了一个自然的构建模块。我们证明了我们的构造在数学上等效于其他已知的构造，例如基于交错和广义排列的构造，但在量子硬件上实现更直接。

    Variational algorithms require architectures that naturally constrain the optimisation space to run efficiently. In geometric quantum machine learning, one achieves this by encoding group structure into parameterised quantum circuits to include the symmetries of a problem as an inductive bias. However, constructing such circuits is challenging as a concrete guiding principle has yet to emerge. In this paper, we propose the use of spin networks, a form of directed tensor network invariant under a group transformation, to devise SU(2) equivariant quantum circuit ans\"atze -- circuits possessing spin rotation symmetry. By changing to the basis that block diagonalises SU(2) group action, these networks provide a natural building block for constructing parameterised equivariant quantum circuits. We prove that our construction is mathematically equivalent to other known constructions, such as those based on twirling and generalised permutations, but more direct to implement on quantum hardwa
    
[^4]: 近似最优非凸-强凸双层优化与全一阶预言机

    Near-Optimal Nonconvex-Strongly-Convex Bilevel Optimization with Fully First-Order Oracles. (arXiv:2306.14853v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2306.14853](http://arxiv.org/abs/2306.14853)

    本文针对双层优化问题中底层问题为强凸的情况，提出了一种更加高效的算法，可以以接近最优的速率收敛。这种算法避免了在实践中可能无法获得或代价昂贵的Hessian-向量乘积预言机的使用。

    

    双层优化在超参数调整、神经架构搜索和元学习等领域有着广泛应用。设计高效的双层优化算法是具有挑战性的，因为底层问题通过另一个优化问题隐式定义了一个可行性集。在这项工作中，我们考虑一种易于处理的情况，即底层问题是强凸的。最近的研究表明，通过Hessian-向量乘积预言机，可以在$\tilde{\mathcal{O}}(\epsilon^{-2})$个预言调用内可靠地找到一个$\epsilon$-一阶稳定点。然而，在实践中，Hessian-向量乘积可能无法获得或代价昂贵。Kwon等人（ICML 2023）通过提出一个一阶方法来解决这个问题，该方法可以以较慢的$\tilde{\mathcal{O}}(\epsilon^{-3})$的速率实现相同的目标。在这项工作中，我们提供了更严格的分析，证明这种方法可以以接近最优的$\tilde {\mathcal{O}}(\epsilon^{-2})$的速率像二阶方法一样收敛。

    Bilevel optimization has wide applications such as hyperparameter tuning, neural architecture search, and meta-learning. Designing efficient algorithms for bilevel optimization is challenging because the lower-level problem defines a feasibility set implicitly via another optimization problem. In this work, we consider one tractable case when the lower-level problem is strongly convex. Recent works show that with a Hessian-vector product oracle, one can provably find an $\epsilon$-first-order stationary point within $\tilde{\mathcal{O}}(\epsilon^{-2})$ oracle calls. However, Hessian-vector product may be inaccessible or expensive in practice. Kwon et al. (ICML 2023) addressed this issue by proposing a first-order method that can achieve the same goal at a slower rate of $\tilde{\mathcal{O}}(\epsilon^{-3})$. In this work, we provide a tighter analysis demonstrating that this method can converge at the near-optimal $\tilde {\mathcal{O}}(\epsilon^{-2})$ rate as second-order methods. Our a
    
[^5]: HD-Bind：使用低精度、超高维度二进制编码分子结构

    HD-Bind: Encoding of Molecular Structure with Low Precision, Hyperdimensional Binary Representations. (arXiv:2303.15604v1 [q-bio.BM])

    [http://arxiv.org/abs/2303.15604](http://arxiv.org/abs/2303.15604)

    HD-Bind是一种用于表示分子结构的超高维度二进制向量的方法，能够预测蛋白质中小分子的分子间结合亲和力。它显著减少了存储和计算所需的位数，并在药物发现中优于现有的方法。

    

    近年来，随着化学合成技术的进步，公开可用的类似药物的分子集合已经增长到数十亿个可能性。然而，传统的从大量药物候选中确定“命中”分子的方法依赖于生物物理理论来计算药物与其蛋白质靶标之间结合相互作用的吉布斯自由能的近似值。该方法的主要缺点是即使对于相对较小的分子集合，也需要出色的计算能力。Hyperdimensional Computing（HDC）是一种最近提出的学习范式，它能够利用低精度二进制向量算术来构建数据的有效表示，而不需要渐进式优化方法，这些优化方法在许多传统的机器学习和深度学习方法中是必需的。本文介绍了HD-Bind，一种将分子结构表示为超高维度二进制向量的方法，用于预测蛋白质中小分子的分子间结合亲和力。HD-Bind构建了超维二进制向量，表示分子结构，并能够捕获与蛋白质-配体结合相关的结构特征，同时显著减少了存储和计算所需的位数。我们的结果表明，HD-Bind优于现有的方法，并为在其他药物发现应用中使用HDC铺平了道路。

    Publicly available collections of drug-like molecules have grown to comprise 10s of billions of possibilities in recent history due to advances in chemical synthesis. Traditional methods for identifying ``hit'' molecules from a large collection of potential drug-like candidates have relied on biophysical theory to compute approximations to the Gibbs free energy of the binding interaction between the drug to its protein target. A major drawback of the approaches is that they require exceptional computing capabilities to consider for even relatively small collections of molecules.  Hyperdimensional Computing (HDC) is a recently proposed learning paradigm that is able to leverage low-precision binary vector arithmetic to build efficient representations of the data that can be obtained without the need for gradient-based optimization approaches that are required in many conventional machine learning and deep learning approaches. This algorithmic simplicity allows for acceleration in hardwa
    
[^6]: AttentionMixer：一个准确且可解释的过程监测框架

    AttentionMixer: An Accurate and Interpretable Framework for Process Monitoring. (arXiv:2302.10426v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2302.10426](http://arxiv.org/abs/2302.10426)

    AttentionMixer是一个旨在为能量变换厂建立准确且可解释的辐射监测框架的数据驱动方法，其技术创新点为空间和时间自适应消息传递块和注意力和可视化技术。

    

    在极端工作条件下运行的高效能转换工厂的安全性高度依赖于准确且可解释的自动监测系统。然而，目前可用的数据驱动监测系统在高准确性或可解释性方面往往无法满足要求，从而阻碍了它们在实践中的应用。为了克服这一限制，提出了一种基于广义消息传递框架的数据驱动方法——AttentionMixer，旨在为能量变换厂建立一个准确且可解释的辐射监测框架。为了提高模型的准确性，第一项技术贡献包括开发空间和时间自适应消息传递块，分别用于捕获空间和时间相关性；这两个块通过混合算子级联。为了增强模型可解释性，第二项技术贡献涉及实现注意力和可视化技术，使得可以对模型的预测做出解释。

    An accurate and explainable automatic monitoring system is critical for the safety of high efficiency energy conversion plants that operate under extreme working condition. Nonetheless, currently available data-driven monitoring systems often fall short in meeting the requirements for either high-accuracy or interpretability, which hinders their application in practice. To overcome this limitation, a data-driven approach, AttentionMixer, is proposed under a generalized message passing framework, with the goal of establishing an accurate and interpretable radiation monitoring framework for energy conversion plants. To improve the model accuracy, the first technical contribution involves the development of spatial and temporal adaptive message passing blocks, which enable the capture of spatial and temporal correlations, respectively; the two blocks are cascaded through a mixing operator. To enhance the model interpretability, the second technical contribution involves the implementation
    

