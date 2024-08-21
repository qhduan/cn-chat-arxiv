# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inferring Dynamic Networks from Marginals with Iterative Proportional Fitting](https://arxiv.org/abs/2402.18697) | 通过识别一个生成网络模型，我们建立了一个设置，IPF可以恢复最大似然估计，揭示了关于在这种设置中使用IPF的隐含假设，并可以为IPF的参数估计提供结构相关的误差界。 |
| [^2] | [Shotgun crystal structure prediction using machine-learned formation energies.](http://arxiv.org/abs/2305.02158) | 本研究使用机器学习方法在多个结构预测标准测试中精确识别含有100个以上原子的许多材料的全局最小结构，并以单次能量评估为基础，取代了重复的第一原理能量计算过程。 |
| [^3] | [Valid Inference after Causal Discovery.](http://arxiv.org/abs/2208.05949) | 本研究开发了工具以实现因果发现后的有效推断，解决了使用相同数据运行因果发现算法后估计因果效应导致经典置信区间的覆盖保证无效问题。 |

# 详细

[^1]: 从边际推断动态网络的方法：迭代比例拟合

    Inferring Dynamic Networks from Marginals with Iterative Proportional Fitting

    [https://arxiv.org/abs/2402.18697](https://arxiv.org/abs/2402.18697)

    通过识别一个生成网络模型，我们建立了一个设置，IPF可以恢复最大似然估计，揭示了关于在这种设置中使用IPF的隐含假设，并可以为IPF的参数估计提供结构相关的误差界。

    

    来自现实数据约束的常见网络推断问题是如何从时间聚合的邻接矩阵和时间变化边际（即行向量和列向量之和）推断动态网络。先前的方法为了解决这个问题重新利用了经典的迭代比例拟合（IPF）过程，也称为Sinkhorn算法，并取得了令人满意的经验结果。然而，使用IPF的统计基础尚未得到很好的理解：在什么情况下，IPF提供了从边际准确估计动态网络的原则性，以及它在多大程度上估计了网络？在这项工作中，我们确定了这样一个设置，通过识别一个生成网络模型，IPF可以恢复其最大似然估计。我们的模型揭示了关于在这种设置中使用IPF的隐含假设，并使得可以进行新的分析，如有关IPF参数估计的结构相关误差界。当IPF失败时

    arXiv:2402.18697v1 Announce Type: cross  Abstract: A common network inference problem, arising from real-world data constraints, is how to infer a dynamic network from its time-aggregated adjacency matrix and time-varying marginals (i.e., row and column sums). Prior approaches to this problem have repurposed the classic iterative proportional fitting (IPF) procedure, also known as Sinkhorn's algorithm, with promising empirical results. However, the statistical foundation for using IPF has not been well understood: under what settings does IPF provide principled estimation of a dynamic network from its marginals, and how well does it estimate the network? In this work, we establish such a setting, by identifying a generative network model whose maximum likelihood estimates are recovered by IPF. Our model both reveals implicit assumptions on the use of IPF in such settings and enables new analyses, such as structure-dependent error bounds on IPF's parameter estimates. When IPF fails to c
    
[^2]: 使用机器学习的形成能量预测方法进行猎枪晶体结构预测

    Shotgun crystal structure prediction using machine-learned formation energies. (arXiv:2305.02158v1 [physics.comp-ph])

    [http://arxiv.org/abs/2305.02158](http://arxiv.org/abs/2305.02158)

    本研究使用机器学习方法在多个结构预测标准测试中精确识别含有100个以上原子的许多材料的全局最小结构，并以单次能量评估为基础，取代了重复的第一原理能量计算过程。

    

    可以通过找到原子构型能量曲面的全局或局部极小值来预测组装原子的稳定或亚稳定晶体结构。通常，这需要重复的第一原理能量计算，这在包含30个以上原子的大型系统中是不实际的。本研究使用简单但功能强大的机器学习工作流，使用机器学习辅助第一原理能量计算，对大量虚拟创建的晶体结构进行非迭代式单次筛选，从而在解决晶体结构预测问题方面取得了重大进展。

    Stable or metastable crystal structures of assembled atoms can be predicted by finding the global or local minima of the energy surface with respect to the atomic configurations. Generally, this requires repeated first-principles energy calculations that are impractical for large systems, such as those containing more than 30 atoms in the unit cell. Here, we have made significant progress in solving the crystal structure prediction problem with a simple but powerful machine-learning workflow; using a machine-learning surrogate for first-principles energy calculations, we performed non-iterative, single-shot screening using a large library of virtually created crystal structures. The present method relies on two key technical components: transfer learning, which enables a highly accurate energy prediction of pre-relaxed crystalline states given only a small set of training samples from first-principles calculations, and generative models to create promising and diverse crystal structure
    
[^3]: 因果发现后的有效推断

    Valid Inference after Causal Discovery. (arXiv:2208.05949v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.05949](http://arxiv.org/abs/2208.05949)

    本研究开发了工具以实现因果发现后的有效推断，解决了使用相同数据运行因果发现算法后估计因果效应导致经典置信区间的覆盖保证无效问题。

    

    因果发现和因果效应估计是因果推断中的两个基本任务。虽然已经针对每个任务单独开发了许多方法，但是同时应用这些方法时会出现统计上的挑战：在对相同数据运行因果发现算法后估计因果效应会导致"双重挑选"，从而使经典置信区间的覆盖保证无效。为此，我们开发了针对因果发现后有效的推断工具。通过实证研究，我们发现，天真组合因果发现算法和随后推断算法会导致高度膨胀的误覆盖率，而应用我们的方法则提供可靠的覆盖并实现比数据分割更准确的因果发现。

    Causal discovery and causal effect estimation are two fundamental tasks in causal inference. While many methods have been developed for each task individually, statistical challenges arise when applying these methods jointly: estimating causal effects after running causal discovery algorithms on the same data leads to "double dipping," invalidating the coverage guarantees of classical confidence intervals. To this end, we develop tools for valid post-causal-discovery inference. Across empirical studies, we show that a naive combination of causal discovery and subsequent inference algorithms leads to highly inflated miscoverage rates; on the other hand, applying our method provides reliable coverage while achieving more accurate causal discovery than data splitting.
    

