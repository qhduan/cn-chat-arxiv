# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Maximum Likelihood Estimation of Latent Variable Structural Equation Models: A Neural Network Approach.](http://arxiv.org/abs/2309.14073) | 本研究提出了一种新的图形结构，用于在线性和高斯性假设下稳定的潜变量结构方程模型。我们证明了计算该模型的最大似然估计等价于训练一个神经网络，并实现了一个基于GPU的算法来进行计算。 |
| [^2] | [Towards end-to-end ASP computation.](http://arxiv.org/abs/2306.06821) | 该论文提出了一种端到端方法，通过线性代数计算稳定模型满足给定的限制条件，同时没有使用符号ASP或SAT求解器，为加速通过并行技术提供了可能。 |
| [^3] | [A* Search Without Expansions: Learning Heuristic Functions with Deep Q-Networks.](http://arxiv.org/abs/2102.04518) | 本文提出了一种使用深度Q网络学习启发式函数，通过只进行一次前向传递计算相邻节点的转移成本和启发式值之和，并在不显式生成这些子节点的情况下指导搜索的Q*搜索算法，以大幅减少计算时间。在魔方问题上的实验表明，该方法能够高效地解决具有大动作空间的问题。 |

# 详细

[^1]: 潜变量结构方程模型的最大似然估计：一种神经网络方法

    Maximum Likelihood Estimation of Latent Variable Structural Equation Models: A Neural Network Approach. (arXiv:2309.14073v1 [stat.ML])

    [http://arxiv.org/abs/2309.14073](http://arxiv.org/abs/2309.14073)

    本研究提出了一种新的图形结构，用于在线性和高斯性假设下稳定的潜变量结构方程模型。我们证明了计算该模型的最大似然估计等价于训练一个神经网络，并实现了一个基于GPU的算法来进行计算。

    

    我们提出了一种在线性和高斯性假设下稳定的结构方程模型的图形结构。我们展示了计算这个模型的最大似然估计等价于训练一个神经网络。我们实现了一个基于GPU的算法来计算这些模型的最大似然估计。

    We propose a graphical structure for structural equation models that is stable under marginalization under linearity and Gaussianity assumptions. We show that computing the maximum likelihood estimation of this model is equivalent to training a neural network. We implement a GPU-based algorithm that computes the maximum likelihood estimation of these models.
    
[^2]: 迈向端到端ASP计算

    Towards end-to-end ASP computation. (arXiv:2306.06821v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2306.06821](http://arxiv.org/abs/2306.06821)

    该论文提出了一种端到端方法，通过线性代数计算稳定模型满足给定的限制条件，同时没有使用符号ASP或SAT求解器，为加速通过并行技术提供了可能。

    

    我们提出了一种端到端方法，用线性代数计算满足给定限制条件的稳定模型，以及ASP的计算。一种构造成矩阵化正常逻辑程序、Lin-Zhao定理中的循环公式和限制条件的代价函数的数值最小化的向量空间直接实现Lin-Zhao定理的思路，因此在我们的方法中没有使用符号ASP或SAT求解器。我们还提出了用于缩小程序大小的预计算和用于减少计算难度的循环公式启发式方法。我们用编程示例对我们的方法进行了实证测试，包括三色涂色问题和哈密顿环问题。由于我们的方法是纯粹数值方法，并且只包含向量/矩阵操作，因此可以通过并行技术（例如多核和GPU）进行加速。

    We propose an end-to-end approach for answer set programming (ASP) and linear algebraically compute stable models satisfying given constraints. The idea is to implement Lin-Zhao's theorem \cite{Lin04} together with constraints directly in vector spaces as numerical minimization of a cost function constructed from a matricized normal logic program, loop formulas in Lin-Zhao's theorem and constraints, thereby no use of symbolic ASP or SAT solvers involved in our approach. We also propose precomputation that shrinks the program size and heuristics for loop formulas to reduce computational difficulty. We empirically test our approach with programming examples including the 3-coloring and Hamiltonian cycle problems. As our approach is purely numerical and only contains vector/matrix operations, acceleration by parallel technologies such as many-cores and GPUs is expected.
    
[^3]: 不扩展的A*搜索：用深度Q网络学习启发式函数

    A* Search Without Expansions: Learning Heuristic Functions with Deep Q-Networks. (arXiv:2102.04518v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2102.04518](http://arxiv.org/abs/2102.04518)

    本文提出了一种使用深度Q网络学习启发式函数，通过只进行一次前向传递计算相邻节点的转移成本和启发式值之和，并在不显式生成这些子节点的情况下指导搜索的Q*搜索算法，以大幅减少计算时间。在魔方问题上的实验表明，该方法能够高效地解决具有大动作空间的问题。

    

    高效地使用 A* 搜索解决具有大动作空间的问题对于人工智能社区几十年来一直非常重要。这是因为 A* 搜索的计算和存储需求随着动作空间的大小呈线性增长。当 A* 搜索使用计算代价高昂的函数逼近器（如深度神经网络）学习启发式函数时，这种负担变得更加明显。为了解决这个问题，我们引入了 Q* 搜索，一种使用深度 Q 网络引导搜索的搜索算法，以利用一个事实，即在不显式生成这些子节点的情况下，一个节点的子节点的转移成本和启发式值之和可以通过单次前向传递计算。这显着降低了计算时间，并且每次迭代只需要生成一个节点。我们使用 Q* 搜索来解决魔方问题，并将其们表示为一个包含 1872 个元动作的大动作空间。

    Efficiently solving problems with large action spaces using A* search has been of importance to the artificial intelligence community for decades. This is because the computation and memory requirements of A* search grow linearly with the size of the action space. This burden becomes even more apparent when A* search uses a heuristic function learned by computationally expensive function approximators, such as deep neural networks. To address this problem, we introduce Q* search, a search algorithm that uses deep Q-networks to guide search in order to take advantage of the fact that the sum of the transition costs and heuristic values of the children of a node can be computed with a single forward pass through a deep Q-network without explicitly generating those children. This significantly reduces computation time and requires only one node to be generated per iteration. We use Q* search to solve the Rubik's cube when formulated with a large action space that includes 1872 meta-action
    

