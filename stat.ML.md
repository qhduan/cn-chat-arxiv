# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [P-tensors: a General Formalism for Constructing Higher Order Message Passing Networks.](http://arxiv.org/abs/2306.10767) | P-tensors 提供了构建高阶消息传递网络的通用形式，其在分子等高度结构化的图形中具有优异的性能表现。 |
| [^2] | [Dictionary Learning under Symmetries via Group Representations.](http://arxiv.org/abs/2305.19557) | 本文研究在预定变换群下学习不变的字典问题。利用非阿贝尔傅里叶分析，提供了算法，建立了字典学习问题可以被有效地理解为某些矩阵优化问题的理论基础。 |

# 详细

[^1]: P张量：构建高阶消息传递网络的通用形式。

    P-tensors: a General Formalism for Constructing Higher Order Message Passing Networks. (arXiv:2306.10767v1 [stat.ML])

    [http://arxiv.org/abs/2306.10767](http://arxiv.org/abs/2306.10767)

    P-tensors 提供了构建高阶消息传递网络的通用形式，其在分子等高度结构化的图形中具有优异的性能表现。

    

    最近的几篇论文表明，高阶图神经网络在高度结构化的图形如分子中能够比其标准的消息传递对应物获得更好的准确性。这些模型通常通过考虑包含在给定图形中的子图的高阶表示，然后在它们之间执行某些线性映射来工作。我们将这些结构正式化为排列等变张量或P张量，并推导出所有线性映射之间的任意顺序等变P张量的基础。在实验中，我们展示了这种范式在几个基准数据集上达到了现有最先进性能。

    Several recent papers have recently shown that higher order graph neural networks can achieve better accuracy than their standard message passing counterparts, especially on highly structured graphs such as molecules. These models typically work by considering higher order representations of subgraphs contained within a given graph and then perform some linear maps between them. We formalize these structures as permutation equivariant tensors, or P-tensors, and derive a basis for all linear maps between arbitrary order equivariant P-tensors. Experimentally, we demonstrate this paradigm achieves state of the art performance on several benchmark datasets.
    
[^2]: 通过群表示学习对称下的字典学习

    Dictionary Learning under Symmetries via Group Representations. (arXiv:2305.19557v1 [math.OC])

    [http://arxiv.org/abs/2305.19557](http://arxiv.org/abs/2305.19557)

    本文研究在预定变换群下学习不变的字典问题。利用非阿贝尔傅里叶分析，提供了算法，建立了字典学习问题可以被有效地理解为某些矩阵优化问题的理论基础。

    

    字典学习问题可以被看作是一个数据驱动的过程，旨在学习一个合适的变换，以便通过示例数据直接表示数据的稀疏性。本文研究了在预定的变换群下学习不变的字典问题。自然的应用领域包括冷冻电镜、多目标跟踪、同步和姿态估计等。我们特别从数学表示理论的角度研究了这个问题。通过利用非阿贝尔傅里叶分析，我们为符合这些不变性的字典学习提供了算法。我们将自然界中的字典学习问题，其自然被建模为无限维度的问题，与相关的计算问题，这必然是有限维度的问题，联系起来。我们建立了字典学习问题可以被有效地理解为某些矩阵优化问题的理论基础。

    The dictionary learning problem can be viewed as a data-driven process to learn a suitable transformation so that data is sparsely represented directly from example data. In this paper, we examine the problem of learning a dictionary that is invariant under a pre-specified group of transformations. Natural settings include Cryo-EM, multi-object tracking, synchronization, pose estimation, etc. We specifically study this problem under the lens of mathematical representation theory. Leveraging the power of non-abelian Fourier analysis for functions over compact groups, we prescribe an algorithmic recipe for learning dictionaries that obey such invariances. We relate the dictionary learning problem in the physical domain, which is naturally modelled as being infinite dimensional, with the associated computational problem, which is necessarily finite dimensional. We establish that the dictionary learning problem can be effectively understood as an optimization instance over certain matrix o
    

