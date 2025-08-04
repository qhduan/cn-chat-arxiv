# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bagged Regularized $k$-Distances for Anomaly Detection](https://arxiv.org/abs/2312.01046) | 本文提出了一种称为Bagged Regularized $k$-Distances for Anomaly Detection (BRDAD)的基于距离的算法，通过将非监督异常检测问题转化为凸优化问题，成功解决了基于距离算法中超参数选择的敏感性挑战，并通过包集成方法解决了处理大规模数据集时的效率问题。 |

# 详细

[^1]: Bagged Regularized $k$-Distances用于异常检测

    Bagged Regularized $k$-Distances for Anomaly Detection

    [https://arxiv.org/abs/2312.01046](https://arxiv.org/abs/2312.01046)

    本文提出了一种称为Bagged Regularized $k$-Distances for Anomaly Detection (BRDAD)的基于距离的算法，通过将非监督异常检测问题转化为凸优化问题，成功解决了基于距离算法中超参数选择的敏感性挑战，并通过包集成方法解决了处理大规模数据集时的效率问题。

    

    本文考虑非监督异常检测的范式，即在没有标记的情况下识别数据集中的异常值。尽管基于距离的方法对于非监督异常检测具有较好的性能，但它们对最近邻数量的选择非常敏感。为此，我们提出了一种新的基于距离的算法，称为Bagged Regularized $k$-Distances for Anomaly Detection (BRDAD)，将非监督异常检测问题转化为凸优化问题。我们的BRDAD算法通过最小化替代风险（即经验风险的有限样本上界）来选择权重，以用于密度估计的带权重的$k$-distances。这种方法成功解决了基于距离算法中超参数选择的敏感性挑战。此外，在处理大规模数据集时，我们还可以通过包集成的方法来解决效率问题。

    We consider the paradigm of unsupervised anomaly detection, which involves the identification of anomalies within a dataset in the absence of labeled examples. Though distance-based methods are top-performing for unsupervised anomaly detection, they suffer heavily from the sensitivity to the choice of the number of the nearest neighbors. In this paper, we propose a new distance-based algorithm called bagged regularized $k$-distances for anomaly detection (BRDAD) converting the unsupervised anomaly detection problem into a convex optimization problem. Our BRDAD algorithm selects the weights by minimizing the surrogate risk, i.e., the finite sample bound of the empirical risk of the bagged weighted $k$-distances for density estimation (BWDDE). This approach enables us to successfully address the sensitivity challenge of the hyperparameter choice in distance-based algorithms. Moreover, when dealing with large-scale datasets, the efficiency issues can be addressed by the incorporated baggi
    

