# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Finding Decision Tree Splits in Streaming and Massively Parallel Models](https://arxiv.org/abs/2403.19867) | 提出了在数据流学习中计算决策树最佳分割点的算法，能够在流式计算和大规模并行模型中高效运行 |
| [^2] | [HyperMagNet: A Magnetic Laplacian based Hypergraph Neural Network](https://arxiv.org/abs/2402.09676) | HyperMagNet是一种基于磁度拉普拉斯的超图神经网络，通过将超图表示为非可逆的马尔可夫链并构建磁度拉普拉斯矩阵作为输入，它在节点分类任务中表现出优越性。 |
| [^3] | [Unveiling Molecular Moieties through Hierarchical Graph Explainability](https://arxiv.org/abs/2402.01744) | 本论文提出了一种使用图神经网络和分层可解释人工智能技术的方法，能够准确预测生物活性并找到与之相关的最重要的成分。 |
| [^4] | [Multi-Step Deductive Reasoning Over Natural Language: An Empirical Study on Out-of-Distribution Generalisation](https://arxiv.org/abs/2207.14000) | 提出了IMA-GloVe-GA，一个用于自然语言表达的多步推理的迭代神经推理网络，在超领域泛化方面具有更好的性能表现。 |
| [^5] | [Accelerating Generalized Linear Models by Trading off Computation for Uncertainty.](http://arxiv.org/abs/2310.20285) | 本论文提出了一种迭代方法，通过增加不确定性来降低计算量，并显著提高广义线性模型的训练速度。 |
| [^6] | [Asynchronous Graph Generators.](http://arxiv.org/abs/2309.17335) | 异步图生成器（AGG）是一种新型的图神经网络架构，通过节点生成进行数据插补，并隐式学习传感器测量的因果图表示，取得了state-of-the-art的结果。 |
| [^7] | [Learning from Similar Linear Representations: Adaptivity, Minimaxity, and Robustness.](http://arxiv.org/abs/2303.17765) | 本文提出了两种算法，适应相似性结构并对异常值任务具有稳健性，适用于表示多任务学习和迁移学习设置。 |
| [^8] | [Second-order Conditional Gradient Sliding.](http://arxiv.org/abs/2002.08907) | 提出了一种二阶条件梯度滑动（SOCGS）算法，可以高效解决约束二次凸优化问题，并在有限次线性收敛迭代后二次收敛于原始间隙。 |

# 详细

[^1]: 在流式和大规模并行模型中找到决策树分割点

    Finding Decision Tree Splits in Streaming and Massively Parallel Models

    [https://arxiv.org/abs/2403.19867](https://arxiv.org/abs/2403.19867)

    提出了在数据流学习中计算决策树最佳分割点的算法，能够在流式计算和大规模并行模型中高效运行

    

    在这项工作中，我们提出了一种数据流算法，用于计算决策树学习中的最优分割点。具体而言，给定观测数据流$x_i$及其标签$y_i$，目标是找到将数据分为两组的最佳分割点$j$，使得均方误差（回归问题）或误分类率（分类问题）最小化。我们提供了多种快速的数据流算法，这些算法在这些问题中使用亚线性空间和少量次数的遍历。这些算法还可以扩展到大规模并行计算模型中。尽管不能直接比较，但我们的工作与Domingos和Hulten的开创性工作（KDD 2000）相互补充。

    arXiv:2403.19867v1 Announce Type: cross  Abstract: In this work, we provide data stream algorithms that compute optimal splits in decision tree learning. In particular, given a data stream of observations $x_i$ and their labels $y_i$, the goal is to find the optimal split point $j$ that divides the data into two sets such that the mean squared error (for regression) or misclassification rate (for classification) is minimized. We provide various fast streaming algorithms that use sublinear space and a small number of passes for these problems. These algorithms can also be extended to the massively parallel computation model. Our work, while not directly comparable, complements the seminal work of Domingos and Hulten (KDD 2000).
    
[^2]: HyperMagNet:一种基于磁度拉普拉斯的超图神经网络

    HyperMagNet: A Magnetic Laplacian based Hypergraph Neural Network

    [https://arxiv.org/abs/2402.09676](https://arxiv.org/abs/2402.09676)

    HyperMagNet是一种基于磁度拉普拉斯的超图神经网络，通过将超图表示为非可逆的马尔可夫链并构建磁度拉普拉斯矩阵作为输入，它在节点分类任务中表现出优越性。

    

    在数据科学领域，超图是对展示多种关系的数据的自然模型，而图只能捕捉到两两之间的关系。然而，许多现有的超图神经网络通过对称矩阵表示将超图有效地简化为无向图，可能会丢失重要信息。我们提出了一种替代超图神经网络的方法，其中将超图表示为非可逆的马尔可夫链。我们使用该马尔可夫链构建了一个复数埃尔米特拉普拉斯矩阵 - 磁度拉普拉斯矩阵，该矩阵作为我们提出的超图神经网络的输入。我们研究了HyperMagNet在节点分类任务中的效果，并证明其在基于图简化的超图神经网络上的优越性。

    arXiv:2402.09676v1 Announce Type: new  Abstract: In data science, hypergraphs are natural models for data exhibiting multi-way relations, whereas graphs only capture pairwise. Nonetheless, many proposed hypergraph neural networks effectively reduce hypergraphs to undirected graphs via symmetrized matrix representations, potentially losing important information. We propose an alternative approach to hypergraph neural networks in which the hypergraph is represented as a non-reversible Markov chain. We use this Markov chain to construct a complex Hermitian Laplacian matrix - the magnetic Laplacian - which serves as the input to our proposed hypergraph neural network. We study HyperMagNet for the task of node classification, and demonstrate its effectiveness over graph-reduction based hypergraph neural networks.
    
[^3]: 通过分层图解释揭示分子成分

    Unveiling Molecular Moieties through Hierarchical Graph Explainability

    [https://arxiv.org/abs/2402.01744](https://arxiv.org/abs/2402.01744)

    本论文提出了一种使用图神经网络和分层可解释人工智能技术的方法，能够准确预测生物活性并找到与之相关的最重要的成分。

    

    背景：图神经网络（GNN）作为一种强大的工具，在支持体外虚拟筛选方面已经出现多年。在这项工作中，我们提出了一种使用图卷积架构实现高精度多靶标筛选的GNN。我们还设计了一种分层可解释人工智能（XAI）技术，通过利用信息传递机制，在原子、环和整个分子层面上直接捕获信息，从而找到与生物活性预测相关的最重要的成分。结果：我们在支持虚拟筛选方面的二十个细胞周期依赖性激酶靶标上报道了一种最先进的GNN分类器。我们的分类器超越了作者提出的先前最先进方法。此外，我们还设计了一个仅针对CDK1的高灵敏度版本的GNN，以使用我们的解释器来避免多类别模型固有的偏差。分层解释器已经由一位专家化学家在19个CDK1批准药物上进行了验证。

    Background: Graph Neural Networks (GNN) have emerged in very recent years as a powerful tool for supporting in silico Virtual Screening. In this work we present a GNN which uses Graph Convolutional architectures to achieve very accurate multi-target screening. We also devised a hierarchical Explainable Artificial Intelligence (XAI) technique to catch information directly at atom, ring, and whole molecule level by leveraging the message passing mechanism. In this way, we find the most relevant moieties involved in bioactivity prediction. Results: We report a state-of-the-art GNN classifier on twenty Cyclin-dependent Kinase targets in support of VS. Our classifier outperforms previous SOTA approaches proposed by the authors. Moreover, a CDK1-only high-sensitivity version of the GNN has been designed to use our explainer in order to avoid the inherent bias of multi-class models. The hierarchical explainer has been validated by an expert chemist on 19 approved drugs on CDK1. Our explainer 
    
[^4]: 自然语言上的多步演绎推理：基于超领域泛化的实证研究

    Multi-Step Deductive Reasoning Over Natural Language: An Empirical Study on Out-of-Distribution Generalisation

    [https://arxiv.org/abs/2207.14000](https://arxiv.org/abs/2207.14000)

    提出了IMA-GloVe-GA，一个用于自然语言表达的多步推理的迭代神经推理网络，在超领域泛化方面具有更好的性能表现。

    

    将深度学习与符号逻辑推理结合起来，旨在充分利用这两个领域的成功，并引起了越来越多的关注。受DeepLogic启发，该模型经过端到端训练，用于执行逻辑程序推理，我们介绍了IMA-GloVe-GA，这是一个用自然语言表达的多步推理的迭代神经推理网络。在我们的模型中，推理是使用基于RNN的迭代内存神经网络进行的，其中包含一个门关注机制。我们在PARARULES、CONCEPTRULES V1和CONCEPTRULES V2三个数据集上评估了IMA-GloVe-GA。实验结果表明，带有门关注机制的DeepLogic比DeepLogic和其他RNN基线模型能够实现更高的测试准确性。我们的模型在规则被打乱时比RoBERTa-Large实现了更好的超领域泛化性能。此外，为了解决当前多步推理数据集中推理深度不平衡的问题

    arXiv:2207.14000v2 Announce Type: replace-cross  Abstract: Combining deep learning with symbolic logic reasoning aims to capitalize on the success of both fields and is drawing increasing attention. Inspired by DeepLogic, an end-to-end model trained to perform inference on logic programs, we introduce IMA-GloVe-GA, an iterative neural inference network for multi-step reasoning expressed in natural language. In our model, reasoning is performed using an iterative memory neural network based on RNN with a gate attention mechanism. We evaluate IMA-GloVe-GA on three datasets: PARARULES, CONCEPTRULES V1 and CONCEPTRULES V2. Experimental results show DeepLogic with gate attention can achieve higher test accuracy than DeepLogic and other RNN baseline models. Our model achieves better out-of-distribution generalisation than RoBERTa-Large when the rules have been shuffled. Furthermore, to address the issue of unbalanced distribution of reasoning depths in the current multi-step reasoning datase
    
[^5]: 通过以计算为代价加速广义线性模型

    Accelerating Generalized Linear Models by Trading off Computation for Uncertainty. (arXiv:2310.20285v1 [cs.LG])

    [http://arxiv.org/abs/2310.20285](http://arxiv.org/abs/2310.20285)

    本论文提出了一种迭代方法，通过增加不确定性来降低计算量，并显著提高广义线性模型的训练速度。

    

    贝叶斯广义线性模型（GLMs）定义了一个灵活的概率框架，用于建模分类、有序和连续数据，并且在实践中被广泛使用。然而，对于大型数据集，GLMs的精确推断代价太高，因此需要在实践中进行近似。造成的近似误差对模型的可靠性产生不利影响，并且没有被考虑在预测的不确定性中。在这项工作中，我们引入了一系列迭代方法，明确地对这个误差建模。它们非常适合并行计算硬件，有效地回收计算并压缩信息，以减少GLMs的时间和内存需求。正如我们在一个实际的大型分类问题上展示的那样，我们的方法通过明确地将减少计算与增加不确定性进行权衡来显著加速训练。

    Bayesian Generalized Linear Models (GLMs) define a flexible probabilistic framework to model categorical, ordinal and continuous data, and are widely used in practice. However, exact inference in GLMs is prohibitively expensive for large datasets, thus requiring approximations in practice. The resulting approximation error adversely impacts the reliability of the model and is not accounted for in the uncertainty of the prediction. In this work, we introduce a family of iterative methods that explicitly model this error. They are uniquely suited to parallel modern computing hardware, efficiently recycle computations, and compress information to reduce both the time and memory requirements for GLMs. As we demonstrate on a realistically large classification problem, our method significantly accelerates training by explicitly trading off reduced computation for increased uncertainty.
    
[^6]: 异步图生成器

    Asynchronous Graph Generators. (arXiv:2309.17335v1 [cs.LG])

    [http://arxiv.org/abs/2309.17335](http://arxiv.org/abs/2309.17335)

    异步图生成器（AGG）是一种新型的图神经网络架构，通过节点生成进行数据插补，并隐式学习传感器测量的因果图表示，取得了state-of-the-art的结果。

    

    我们引入了异步图生成器（AGG），这是一种用于多通道时间序列的新型图神经网络架构。AGG将观测值建模为动态图上的节点，并通过转导式节点生成进行数据插补。AGG不依赖于循环组件或对时间规律的假设，使用可学习的嵌入将测量值、时间戳和元数据直接表示在节点中，并利用注意机制来学习变量之间的关系。这样，所提出的架构隐式地学习传感器测量的因果图表示，可以基于未见时间戳和元数据对新的测量进行预测。我们将所提出的AGG在概念和实证两方面与之前的工作进行了比较，并简要讨论了数据增强对AGG性能的影响。实验结果表明，AGG在t

    We introduce the asynchronous graph generator (AGG), a novel graph neural network architecture for multi-channel time series which models observations as nodes on a dynamic graph and can thus perform data imputation by transductive node generation. Completely free from recurrent components or assumptions about temporal regularity, AGG represents measurements, timestamps and metadata directly in the nodes via learnable embeddings, to then leverage attention to learn expressive relationships across the variables of interest. This way, the proposed architecture implicitly learns a causal graph representation of sensor measurements which can be conditioned on unseen timestamps and metadata to predict new measurements by an expansion of the learnt graph. The proposed AGG is compared both conceptually and empirically to previous work, and the impact of data augmentation on the performance of AGG is also briefly discussed. Our experiments reveal that AGG achieved state-of-the-art results in t
    
[^7]: 学习相似的线性表示：适应性、极小化、以及稳健性

    Learning from Similar Linear Representations: Adaptivity, Minimaxity, and Robustness. (arXiv:2303.17765v1 [stat.ML])

    [http://arxiv.org/abs/2303.17765](http://arxiv.org/abs/2303.17765)

    本文提出了两种算法，适应相似性结构并对异常值任务具有稳健性，适用于表示多任务学习和迁移学习设置。

    

    表示多任务学习和迁移学习在实践中取得了巨大的成功，然而对这些方法的理论理解仍然欠缺。本文旨在理解从具有相似但并非完全相同的线性表示的任务中学习，同时处理异常值任务。我们提出了两种算法，适应相似性结构并对异常值任务具有稳健性，适用于表示多任务学习和迁移学习设置，我们的算法在单任务或仅目标学习时表现优异。

    Representation multi-task learning (MTL) and transfer learning (TL) have achieved tremendous success in practice. However, the theoretical understanding of these methods is still lacking. Most existing theoretical works focus on cases where all tasks share the same representation, and claim that MTL and TL almost always improve performance. However, as the number of tasks grow, assuming all tasks share the same representation is unrealistic. Also, this does not always match empirical findings, which suggest that a shared representation may not necessarily improve single-task or target-only learning performance. In this paper, we aim to understand how to learn from tasks with \textit{similar but not exactly the same} linear representations, while dealing with outlier tasks. We propose two algorithms that are \textit{adaptive} to the similarity structure and \textit{robust} to outlier tasks under both MTL and TL settings. Our algorithms outperform single-task or target-only learning when
    
[^8]: 二阶条件梯度滑动

    Second-order Conditional Gradient Sliding. (arXiv:2002.08907v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2002.08907](http://arxiv.org/abs/2002.08907)

    提出了一种二阶条件梯度滑动（SOCGS）算法，可以高效解决约束二次凸优化问题，并在有限次线性收敛迭代后二次收敛于原始间隙。

    

    当需要高精度解决问题时，约束二阶凸优化算法是首选，因为它们具有局部二次收敛性。这些算法在每次迭代时需要解决一个约束二次子问题。我们提出了\emph{二阶条件梯度滑动}（SOCGS）算法，它使用一种无投影算法来近似解决约束二次子问题。当可行域是一个多面体时，该算法在有限次线性收敛迭代后二次收敛于原始间隙。进入二次收敛阶段后，SOCGS算法需通过$\mathcal{O}(\log(\log 1/\varepsilon))$次一阶和Hessian正交调用以及$\mathcal{O}(\log (1/\varepsilon) \log(\log1/\varepsilon))$次线性最小化正交调用来实现$\varepsilon$-最优解。当可行域只能通过线性优化正交调用高效访问时，此算法非常有用。

    Constrained second-order convex optimization algorithms are the method of choice when a high accuracy solution to a problem is needed, due to their local quadratic convergence. These algorithms require the solution of a constrained quadratic subproblem at every iteration. We present the \emph{Second-Order Conditional Gradient Sliding} (SOCGS) algorithm, which uses a projection-free algorithm to solve the constrained quadratic subproblems inexactly. When the feasible region is a polytope the algorithm converges quadratically in primal gap after a finite number of linearly convergent iterations. Once in the quadratic regime the SOCGS algorithm requires $\mathcal{O}(\log(\log 1/\varepsilon))$ first-order and Hessian oracle calls and $\mathcal{O}(\log (1/\varepsilon) \log(\log1/\varepsilon))$ linear minimization oracle calls to achieve an $\varepsilon$-optimal solution. This algorithm is useful when the feasible region can only be accessed efficiently through a linear optimization oracle, 
    

