# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Community detection by spectral methods in multi-layer networks](https://arxiv.org/abs/2403.12540) | 改进的谱聚类算法在多层网络中实现了更好的社区检测性能，并证明多层网络对社区检测有优势。 |
| [^2] | [On Independent Samples Along the Langevin Diffusion and the Unadjusted Langevin Algorithm](https://arxiv.org/abs/2402.17067) | 在该论文中，我们研究了朗之凡扩散和未调整朗之凡算法中随机变量独立化速率的收敛性，证明了在目标函数强对数凹和平滑的情况下，互信息会以指数速率收敛于$0$。 |
| [^3] | [Partial Search in a Frozen Network is Enough to Find a Strong Lottery Ticket](https://arxiv.org/abs/2402.14029) | 提出一种方法，通过冻结随机子集的初始权重来减少强大的彩票票证（SLT）搜索空间，从而独立于所需SLT稀疏性降低了SLT搜索空间，保证了SLT在这种减少搜索空间中的存在。 |
| [^4] | [Understanding Generalization in the Interpolation Regime using the Rate Function.](http://arxiv.org/abs/2306.10947) | 本文利用大偏差理论，提出一种基于函数的平滑模型特征描述方法，解释了为什么一些插值器有很好的泛化能力以及现代学习技术为什么能够找到它们。 |
| [^5] | [Incentivizing Honesty among Competitors in Collaborative Learning and Optimization.](http://arxiv.org/abs/2305.16272) | 这项研究提出了一个模型来描述在协作学习中竞争对手的不诚实行为，提出了机制来激励诚实沟通，并确保学习质量与全面合作相当。 |
| [^6] | [Locally Adaptive Algorithms for Multiple Testing with Network Structure, with Application to Genome-Wide Association Studies.](http://arxiv.org/abs/2203.11461) | 本文提出了一种基于网络结构的局部自适应结构学习算法，可将LD网络数据和多个样本的辅助数据整合起来，通过数据驱动的权重分配方法实现对多重检验的控制，并在网络数据具有信息量时具有更高的功效。 |

# 详细

[^1]: 多层网络中基于谱方法的社区检测

    Community detection by spectral methods in multi-layer networks

    [https://arxiv.org/abs/2403.12540](https://arxiv.org/abs/2403.12540)

    改进的谱聚类算法在多层网络中实现了更好的社区检测性能，并证明多层网络对社区检测有优势。

    

    多层网络中的社区检测是网络分析中一个关键问题。本文分析了两种谱聚类算法在多层度校正随机块模型（MLDCSBM）框架下进行社区检测的性能。一种算法基于邻接矩阵的和，另一种利用了去偏和的平方邻接矩阵的和。我们在网络规模和/或层数增加时建立了这些方法在MLDCSBM下进行社区检测的一致性结果。我们的定理展示了利用多层进行社区检测的优势。此外，我们的分析表明，利用去偏和的平方邻接矩阵的谱聚类通常优于利用邻接矩阵的谱聚类。数值模拟证实了我们的算法，采用了去偏和的平方邻接矩阵。

    arXiv:2403.12540v1 Announce Type: cross  Abstract: Community detection in multi-layer networks is a crucial problem in network analysis. In this paper, we analyze the performance of two spectral clustering algorithms for community detection within the multi-layer degree-corrected stochastic block model (MLDCSBM) framework. One algorithm is based on the sum of adjacency matrices, while the other utilizes the debiased sum of squared adjacency matrices. We establish consistency results for community detection using these methods under MLDCSBM as the size of the network and/or the number of layers increases. Our theorems demonstrate the advantages of utilizing multiple layers for community detection. Moreover, our analysis indicates that spectral clustering with the debiased sum of squared adjacency matrices is generally superior to spectral clustering with the sum of adjacency matrices. Numerical simulations confirm that our algorithm, employing the debiased sum of squared adjacency matri
    
[^2]: 关于朗之凡扩散和未调整朗之凡算法中独立样本的研究

    On Independent Samples Along the Langevin Diffusion and the Unadjusted Langevin Algorithm

    [https://arxiv.org/abs/2402.17067](https://arxiv.org/abs/2402.17067)

    在该论文中，我们研究了朗之凡扩散和未调整朗之凡算法中随机变量独立化速率的收敛性，证明了在目标函数强对数凹和平滑的情况下，互信息会以指数速率收敛于$0$。

    

    我们研究了马尔可夫链中初始和当前随机变量独立化的速率，重点关注连续时间中的朗之凡扩散和离散时间中的未调整朗之凡算法（ULA）。我们通过它们的互信息度量随机变量之间的依赖关系。对于朗之凡扩散，我们展示了当目标函数强对数凹时，互信息以指数速率收敛于$0$，当目标函数弱对数凹时，以多项式速率收敛。这些速率类似于在类似条件下朗之凡扩散的混合时间。对于ULA，我们展示了当目标函数强对数凹且光滑时，互信息以指数速率收敛于$0$。我们通过发展这些马尔可夫链的互信息版本的混合时间分析来证明我们的结果。我们还提供了基于朗之凡扩散的强数据处理不等式的替代证明。

    arXiv:2402.17067v1 Announce Type: cross  Abstract: We study the rate at which the initial and current random variables become independent along a Markov chain, focusing on the Langevin diffusion in continuous time and the Unadjusted Langevin Algorithm (ULA) in discrete time. We measure the dependence between random variables via their mutual information. For the Langevin diffusion, we show the mutual information converges to $0$ exponentially fast when the target is strongly log-concave, and at a polynomial rate when the target is weakly log-concave. These rates are analogous to the mixing time of the Langevin diffusion under similar assumptions. For the ULA, we show the mutual information converges to $0$ exponentially fast when the target is strongly log-concave and smooth. We prove our results by developing the mutual version of the mixing time analyses of these Markov chains. We also provide alternative proofs based on strong data processing inequalities for the Langevin diffusion 
    
[^3]: 冻结网络中的部分搜索足以找到强大的彩票票证

    Partial Search in a Frozen Network is Enough to Find a Strong Lottery Ticket

    [https://arxiv.org/abs/2402.14029](https://arxiv.org/abs/2402.14029)

    提出一种方法，通过冻结随机子集的初始权重来减少强大的彩票票证（SLT）搜索空间，从而独立于所需SLT稀疏性降低了SLT搜索空间，保证了SLT在这种减少搜索空间中的存在。

    

    arXiv:2402.14029v1 公告类型：跨越 摘要：随机初始化的稠密网络包含可以在不进行权重学习的情况下实现高准确度的子网络--强大的彩票票证（SLTs）。最近，Gadhikar等人（2023年）在理论和实验证明，SLTs也可以在随机修剪的源网络中找到，从而减少SLT的搜索空间。然而，这限制了对甚至比源网络更稀疏的SLTs的搜索，导致由于意外的高稀疏性而准确度较差。本文提出了一种通过独立于所需SLT稀疏性的任意比率减少SLT搜索空间的方法。通过冻结一部分初始权重的随机子集，将其排除在搜索空间之外--即，通过永久修剪它们或将它们锁定为SLT的固定部分。事实上，通过我们与随机冻结变量的子集和逼近，在这种减少的搜索空间中，SLT的存在在理论上是得到保证的。除此之外，还可以减少...

    arXiv:2402.14029v1 Announce Type: cross  Abstract: Randomly initialized dense networks contain subnetworks that achieve high accuracy without weight learning -- strong lottery tickets (SLTs). Recently, Gadhikar et al. (2023) demonstrated theoretically and experimentally that SLTs can also be found within a randomly pruned source network, thus reducing the SLT search space. However, this limits the search to SLTs that are even sparser than the source, leading to worse accuracy due to unintentionally high sparsity. This paper proposes a method that reduces the SLT search space by an arbitrary ratio that is independent of the desired SLT sparsity. A random subset of the initial weights is excluded from the search space by freezing it -- i.e., by either permanently pruning them or locking them as a fixed part of the SLT. Indeed, the SLT existence in such a reduced search space is theoretically guaranteed by our subset-sum approximation with randomly frozen variables. In addition to reducin
    
[^4]: 使用速率函数理解插值区间的泛化

    Understanding Generalization in the Interpolation Regime using the Rate Function. (arXiv:2306.10947v1 [cs.LG])

    [http://arxiv.org/abs/2306.10947](http://arxiv.org/abs/2306.10947)

    本文利用大偏差理论，提出一种基于函数的平滑模型特征描述方法，解释了为什么一些插值器有很好的泛化能力以及现代学习技术为什么能够找到它们。

    

    本文基于大偏差理论的基本原理，提出了一种模型平滑度的新特征描述方法。与以往的工作不同，以往的工作通常用实数值（如权重范数）来表征模型的平滑度，我们表明可以用简单的实值函数来描述平滑度。基于模型平滑度的这一概念，我们提出了一个统一的理论解释，为什么一些插值器表现出非常好的泛化能力，以及为什么广泛使用的现代学习技术（如随机梯度下降，$\ell_2$-规范化，数据增强，不变的架构和超参数化）能够找到它们。我们得出的结论是，所有这些方法都提供了互补的过程，这些过程使优化器偏向于更平滑的插值器，而根据这种理论分析，更平滑的插值器是具有更好的泛化误差的插值器。

    In this paper, we present a novel characterization of the smoothness of a model based on basic principles of Large Deviation Theory. In contrast to prior work, where the smoothness of a model is normally characterized by a real value (e.g., the weights' norm), we show that smoothness can be described by a simple real-valued function. Based on this concept of smoothness, we propose an unifying theoretical explanation of why some interpolators generalize remarkably well and why a wide range of modern learning techniques (i.e., stochastic gradient descent, $\ell_2$-norm regularization, data augmentation, invariant architectures, and overparameterization) are able to find them. The emergent conclusion is that all these methods provide complimentary procedures that bias the optimizer to smoother interpolators, which, according to this theoretical analysis, are the ones with better generalization error.
    
[^5]: 在协同学习和优化中激励竞争对手诚实行为的研究

    Incentivizing Honesty among Competitors in Collaborative Learning and Optimization. (arXiv:2305.16272v1 [cs.LG])

    [http://arxiv.org/abs/2305.16272](http://arxiv.org/abs/2305.16272)

    这项研究提出了一个模型来描述在协作学习中竞争对手的不诚实行为，提出了机制来激励诚实沟通，并确保学习质量与全面合作相当。

    

    协同学习技术能够让机器学习模型的训练比仅利用单一数据源的模型效果更好。然而，在许多情况下，潜在的参与者是下游任务中的竞争对手，如每个都希望通过提供最佳推荐来吸引客户的公司。这可能会激励不诚实的更新，损害其他参与者的模型，从而可能破坏协作的好处。在这项工作中，我们制定了一个模型来描述这种交互，并在该框架内研究了两个学习任务：单轮均值估计和强凸目标的多轮 SGD。对于一类自然的参与者行为，我们发现理性的客户会被激励强烈地操纵他们的更新，从而防止学习。然后，我们提出了机制来激励诚实沟通，并确保学习质量与全面合作相当。最后，我们通过实验证明了这一点。

    Collaborative learning techniques have the potential to enable training machine learning models that are superior to models trained on a single entity's data. However, in many cases, potential participants in such collaborative schemes are competitors on a downstream task, such as firms that each aim to attract customers by providing the best recommendations. This can incentivize dishonest updates that damage other participants' models, potentially undermining the benefits of collaboration. In this work, we formulate a game that models such interactions and study two learning tasks within this framework: single-round mean estimation and multi-round SGD on strongly-convex objectives. For a natural class of player actions, we show that rational clients are incentivized to strongly manipulate their updates, preventing learning. We then propose mechanisms that incentivize honest communication and ensure learning quality comparable to full cooperation. Lastly, we empirically demonstrate the
    
[^6]: 基于网络结构的局部自适应多重检验算法，及其在全基因组关联研究中的应用

    Locally Adaptive Algorithms for Multiple Testing with Network Structure, with Application to Genome-Wide Association Studies. (arXiv:2203.11461v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2203.11461](http://arxiv.org/abs/2203.11461)

    本文提出了一种基于网络结构的局部自适应结构学习算法，可将LD网络数据和多个样本的辅助数据整合起来，通过数据驱动的权重分配方法实现对多重检验的控制，并在网络数据具有信息量时具有更高的功效。

    

    链接分析在全基因组关联研究中起着重要作用，特别是在揭示与疾病表型相关的连锁不平衡（LD）的SNP共同影响方面。然而，LD网络数据的潜力在文献中往往被忽视或未充分利用。在本文中，我们提出了一个局部自适应结构学习算法（LASLA），为整合网络数据或来自相关源域的多个样本的辅助数据提供了一个有原则且通用的框架；可能具有不同的维度/结构和不同的人群。LASLA采用$p$值加权方法，利用结构洞察力为各个检验点分配数据驱动的权重。理论分析表明，当主要统计量独立或弱相关时，LASLA可以渐近地控制FDR，并在网络数据具有信息量时实现更高的功效。通过各种合成实验和一个应用案例，展示了LASLA的效率。

    Linkage analysis has provided valuable insights to the GWAS studies, particularly in revealing that SNPs in linkage disequilibrium (LD) can jointly influence disease phenotypes. However, the potential of LD network data has often been overlooked or underutilized in the literature. In this paper, we propose a locally adaptive structure learning algorithm (LASLA) that provides a principled and generic framework for incorporating network data or multiple samples of auxiliary data from related source domains; possibly in different dimensions/structures and from diverse populations. LASLA employs a $p$-value weighting approach, utilizing structural insights to assign data-driven weights to individual test points. Theoretical analysis shows that LASLA can asymptotically control FDR with independent or weakly dependent primary statistics, and achieve higher power when the network data is informative. Efficiency again of LASLA is illustrated through various synthetic experiments and an applica
    

