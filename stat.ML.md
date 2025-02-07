# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rule Generation for Classification: Scalability, Interpretability, and Fairness.](http://arxiv.org/abs/2104.10751) | 这项研究介绍了一种新的基于规则的分类优化方法，利用列生成线性规划实现可扩展性，并通过分配成本系数和引入额外约束解决了解释性和公平性问题。该方法在局部解释性和公平性之间取得了良好的平衡。 |

# 详细

[^1]: 分类规则生成：可扩展性，解释性和公平性

    Rule Generation for Classification: Scalability, Interpretability, and Fairness. (arXiv:2104.10751v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2104.10751](http://arxiv.org/abs/2104.10751)

    这项研究介绍了一种新的基于规则的分类优化方法，利用列生成线性规划实现可扩展性，并通过分配成本系数和引入额外约束解决了解释性和公平性问题。该方法在局部解释性和公平性之间取得了良好的平衡。

    

    我们引入了一种新的基于规则的分类优化方法，具有约束条件。所提出的方法利用列生成线性规划，因此可扩展到大型数据集。所得定价子问题被证明是NP难问题。我们采用基于决策树的启发式方法，并解决了一个代理定价子问题以加速。该方法返回一组规则以及它们的最优权重，指示每个规则对学习的重要性。我们通过为规则分配成本系数和引入额外约束来解决解释性和公平性问题。具体而言，我们关注局部解释性，并将公平性的一般分离准则推广到多个敏感属性和类别。我们在一系列数据集上测试了所提出方法的性能，并提供了一个案例研究来详细阐述其不同方面。所提出的基于规则的学习方法在局部解释性和公平性之间达到了良好的平衡点。

    We introduce a new rule-based optimization method for classification with constraints. The proposed method leverages column generation for linear programming, and hence, is scalable to large datasets. The resulting pricing subproblem is shown to be NP-Hard. We recourse to a decision tree-based heuristic and solve a proxy pricing subproblem for acceleration. The method returns a set of rules along with their optimal weights indicating the importance of each rule for learning. We address interpretability and fairness by assigning cost coefficients to the rules and introducing additional constraints. In particular, we focus on local interpretability and generalize separation criterion in fairness to multiple sensitive attributes and classes. We test the performance of the proposed methodology on a collection of datasets and present a case study to elaborate on its different aspects. The proposed rule-based learning method exhibits a good compromise between local interpretability and fairn
    

