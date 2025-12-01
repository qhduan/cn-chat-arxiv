# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sharp Generalization of Transductive Learning: A Transductive Local Rademacher Complexity Approach.](http://arxiv.org/abs/2309.16858) | 我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们利用变量的方差信息构建了TLRC，并将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。 |
| [^2] | [Strong consistency and optimality of spectral clustering in symmetric binary non-uniform Hypergraph Stochastic Block Model.](http://arxiv.org/abs/2306.06845) | 论文提出了非均匀超图随机块模型下谱聚类的强一致性信息理论阈值，并且在该阈值以下给出估计标签的期望“不匹配率”上界。并且，单步谱算法可以在超过该阈值时非常高的概率正确地给定每个顶点的标签。 |
| [^3] | [A Trio Neural Model for Dynamic Entity Relatedness Ranking.](http://arxiv.org/abs/1808.08316) | 这篇论文提出了一种基于神经网络的方法，通过动态评估实体相关性，利用集体注意作为监督，能学习到丰富而不同的实体表示，能在大规模数据集上比竞争基线获得更好的结果。 |

# 详细

[^1]: Transductive Learning的尖锐泛化：一种Transductive Local Rademacher Complexity方法

    Sharp Generalization of Transductive Learning: A Transductive Local Rademacher Complexity Approach. (arXiv:2309.16858v1 [stat.ML])

    [http://arxiv.org/abs/2309.16858](http://arxiv.org/abs/2309.16858)

    我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们利用变量的方差信息构建了TLRC，并将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。

    

    我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们的工作将传统的local rademacher complexity (LRC)的思想扩展到了transductive设置中，相对于典型的LRC方法在归纳设置中的分析有了相当大的变化。我们提出了一种基于Rademacher complex的局部化工具，可以应用于各种transductive learning问题，并在适当条件下得到了尖锐的界限。与LRC的发展类似，我们通过从独立变量的方差信息开始构建TLRC，将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。经过精心设计的...

    We introduce a new tool, Transductive Local Rademacher Complexity (TLRC), to analyze the generalization performance of transductive learning methods and motivate new transductive learning algorithms. Our work extends the idea of the popular Local Rademacher Complexity (LRC) to the transductive setting with considerable changes compared to the analysis of typical LRC methods in the inductive setting. We present a localized version of Rademacher complexity based tool wihch can be applied to various transductive learning problems and gain sharp bounds under proper conditions. Similar to the development of LRC, we build TLRC by starting from a sharp concentration inequality for independent variables with variance information. The prediction function class of a transductive learning model is then divided into pieces with a sub-root function being the upper bound for the Rademacher complexity of each piece, and the variance of all the functions in each piece is limited. A carefully designed 
    
[^2]: 对称二元非均匀超图随机块模型中谱聚类的强一致性与最优性

    Strong consistency and optimality of spectral clustering in symmetric binary non-uniform Hypergraph Stochastic Block Model. (arXiv:2306.06845v1 [math.ST])

    [http://arxiv.org/abs/2306.06845](http://arxiv.org/abs/2306.06845)

    论文提出了非均匀超图随机块模型下谱聚类的强一致性信息理论阈值，并且在该阈值以下给出估计标签的期望“不匹配率”上界。并且，单步谱算法可以在超过该阈值时非常高的概率正确地给定每个顶点的标签。

    

    本论文考虑了在非均匀超图随机块模型下，两个等大小的社区（n/2）中的随机超图上的无监督分类问题，其中每个边只依赖于其顶点的标签，边以一定概率独立出现。在这篇论文中，建立了强一致性的信息理论阈值，在该阈值以下，任何算法都有很高概率会误分类至少两个顶点，而特征向量估计量的期望“不匹配率”上界为$n$的阈值的负指数。另一方面，当超过该阈值时，尽管张量收缩引起了信息损失，但单步谱算法仅在给定收缩的邻接矩阵时，即使SDP在某些情况下失败，也可以非常高的概率正确地给定每个顶点分配标签。此外，强一致性可以通过对所有次优聚合信息实现。

    Consider the unsupervised classification problem in random hypergraphs under the non-uniform \emph{Hypergraph Stochastic Block Model} (HSBM) with two equal-sized communities ($n/2$), where each edge appears independently with some probability depending only on the labels of its vertices. In this paper, an \emph{information-theoretical} threshold for strong consistency is established. Below the threshold, every algorithm would misclassify at least two vertices with high probability, and the expected \emph{mismatch ratio} of the eigenvector estimator is upper bounded by $n$ to the power of minus the threshold. On the other hand, when above the threshold, despite the information loss induced by tensor contraction, one-stage spectral algorithms assign every vertex correctly with high probability when only given the contracted adjacency matrix, even if \emph{semidefinite programming} (SDP) fails in some scenarios. Moreover, strong consistency is achievable by aggregating information from al
    
[^3]: 一种三元神经模型用于动态实体相关性排名

    A Trio Neural Model for Dynamic Entity Relatedness Ranking. (arXiv:1808.08316v4 [cs.IR] UPDATED)

    [http://arxiv.org/abs/1808.08316](http://arxiv.org/abs/1808.08316)

    这篇论文提出了一种基于神经网络的方法，通过动态评估实体相关性，利用集体注意作为监督，能学习到丰富而不同的实体表示，能在大规模数据集上比竞争基线获得更好的结果。

    

    测量实体相关性是许多自然语言处理和信息检索应用的基本任务。之前的研究通常在静态设置和非监督方式下研究实体相关性。然而，现实世界中的实体往往涉及许多不同的关系，因此实体关系随时间变得非常动态。在这项工作中，我们提出了一种基于神经网络的方法来动态评估实体相关性，利用集体注意力作为监督。我们的模型能够在联合框架中学习丰富而不同的实体表示。通过对大规模数据集的广泛实验，我们证明了我们的方法比竞争基线获得了更好的结果。

    Measuring entity relatedness is a fundamental task for many natural language processing and information retrieval applications. Prior work often studies entity relatedness in static settings and an unsupervised manner. However, entities in real-world are often involved in many different relationships, consequently entity-relations are very dynamic over time. In this work, we propose a neural networkbased approach for dynamic entity relatedness, leveraging the collective attention as supervision. Our model is capable of learning rich and different entity representations in a joint framework. Through extensive experiments on large-scale datasets, we demonstrate that our method achieves better results than competitive baselines.
    

