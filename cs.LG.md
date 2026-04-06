# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Causal Graph Discovery Using Large Language Models](https://rss.arxiv.org/abs/2402.01207) | 提出了一个新的框架，利用大型语言模型进行高效的因果图发现，采用了广度优先搜索方法，只需要线性数量的查询，同时能轻松结合观察数据以提高性能，具有高效性和数据效率，并在真实因果图上取得了最先进的结果，展示了其在不同领域的广泛适用性潜力。 |
| [^2] | [An Optimal and Scalable Matrix Mechanism for Noisy Marginals under Convex Loss Functions.](http://arxiv.org/abs/2305.08175) | ResidualPlanner是一种用于带有高斯噪声的边缘的矩阵机制，既优化又可扩展，可以优化许多可以写成边际方差的凸函数的损失函数。 |

# 详细

[^1]: 使用大型语言模型的高效因果图发现

    Efficient Causal Graph Discovery Using Large Language Models

    [https://rss.arxiv.org/abs/2402.01207](https://rss.arxiv.org/abs/2402.01207)

    提出了一个新的框架，利用大型语言模型进行高效的因果图发现，采用了广度优先搜索方法，只需要线性数量的查询，同时能轻松结合观察数据以提高性能，具有高效性和数据效率，并在真实因果图上取得了最先进的结果，展示了其在不同领域的广泛适用性潜力。

    

    我们提出了一个新的框架，利用LLMs进行完整的因果图发现。之前基于LLM的方法采用了成对查询的方法，但这需要二次查询的数量，对于较大的因果图来说很快变得不可行。相反，提出的框架采用了广度优先搜索（BFS）的方法，只需要线性数量的查询。我们还展示了当有所观察数据可用时，提出的方法可以轻松地进行结合以提高性能。除了更具时间和数据效率外，提出的框架在不同大小的真实因果图上取得了最先进的结果。结果证明了提出方法在发现因果关系方面的有效性和效率，展示了其在不同领域的因果图发现任务中的广泛适用性潜力。

    We propose a novel framework that leverages LLMs for full causal graph discovery. While previous LLM-based methods have used a pairwise query approach, this requires a quadratic number of queries which quickly becomes impractical for larger causal graphs. In contrast, the proposed framework uses a breadth-first search (BFS) approach which allows it to use only a linear number of queries. We also show that the proposed method can easily incorporate observational data when available, to improve performance. In addition to being more time and data-efficient, the proposed framework achieves state-of-the-art results on real-world causal graphs of varying sizes. The results demonstrate the effectiveness and efficiency of the proposed method in discovering causal relationships, showcasing its potential for broad applicability in causal graph discovery tasks across different domains.
    
[^2]: 一种优化且可扩展的矩阵机制用于扰动边缘数据下凸损失函数。

    An Optimal and Scalable Matrix Mechanism for Noisy Marginals under Convex Loss Functions. (arXiv:2305.08175v1 [cs.DB])

    [http://arxiv.org/abs/2305.08175](http://arxiv.org/abs/2305.08175)

    ResidualPlanner是一种用于带有高斯噪声的边缘的矩阵机制，既优化又可扩展，可以优化许多可以写成边际方差的凸函数的损失函数。

    

    扰动的边缘数据是一种常见的保护数据隐私的形式，可用于诸如列联表分析、贝叶斯网络构建和合成数据生成等下游任务。我们提出了ResidualPlanner，这是一种用于带有高斯噪声的边缘的矩阵机制，既优化又可扩展。ResidualPlanner可以优化许多可以写成边际方差的凸函数的损失函数。此外，ResidualPlanner可以在几秒钟内优化大规模设置中的边缘准确性，即使之前的最先进技术（HDMM）也会占用过多的内存。甚至在具有100个属性的数据集上也可以在几分钟内运行。此外，ResidualPlanner还可以有效地计算每个边缘的方差/协方差值（之前的方法会很快失败）。

    Noisy marginals are a common form of confidentiality-protecting data release and are useful for many downstream tasks such as contingency table analysis, construction of Bayesian networks, and even synthetic data generation. Privacy mechanisms that provide unbiased noisy answers to linear queries (such as marginals) are known as matrix mechanisms.  We propose ResidualPlanner, a matrix mechanism for marginals with Gaussian noise that is both optimal and scalable. ResidualPlanner can optimize for many loss functions that can be written as a convex function of marginal variances (prior work was restricted to just one predefined objective function). ResidualPlanner can optimize the accuracy of marginals in large scale settings in seconds, even when the previous state of the art (HDMM) runs out of memory. It even runs on datasets with 100 attributes in a couple of minutes. Furthermore ResidualPlanner can efficiently compute variance/covariance values for each marginal (prior methods quickly
    

