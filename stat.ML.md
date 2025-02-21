# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty Quantification in Anomaly Detection with Cross-Conformal $p$-Values](https://arxiv.org/abs/2402.16388) | 针对异常检测系统中不确定性量化的需求，提出了一种新颖的框架，称为交叉一致异常检测，通过校准模型的不确定性提供统计保证。 |
| [^2] | [Hilbert's projective metric for functions of bounded growth and exponential convergence of Sinkhorn's algorithm.](http://arxiv.org/abs/2311.04041) | 本文对有界增长可积函数空间的Hilbert投影度量进行了研究，提出了某些松弛锥体内的内核积分算子具有压缩映射的性质，并应用于熵最优传输问题中，证明了Sinkhorn算法在边际分布的尾部与成本函数增长适当时呈指数收敛。 |
| [^3] | [Robust Classification of High-Dimensional Data using Data-Adaptive Energy Distance.](http://arxiv.org/abs/2306.13985) | 该论文提出了一种用于高维低样本量数据分类的稳健的数据自适应能量距离分类器，该分类器无需调参且在一定条件下可以实现完美分类，已在模拟研究和实际数据分析中得到证明比其他方法表现更优。 |

# 详细

[^1]: 具有交叉一致$p$-值的异常检测中的不确定性量化

    Uncertainty Quantification in Anomaly Detection with Cross-Conformal $p$-Values

    [https://arxiv.org/abs/2402.16388](https://arxiv.org/abs/2402.16388)

    针对异常检测系统中不确定性量化的需求，提出了一种新颖的框架，称为交叉一致异常检测，通过校准模型的不确定性提供统计保证。

    

    随着可靠、可信和可解释机器学习的重要性日益增加，对异常检测系统进行不确定性量化的要求变得愈发重要。在这种情况下，有效控制类型I错误率($\alpha$)而又不损害系统的统计功率($1-\beta$)可以建立信任，并减少与假发现相关的成本，特别是当后续程序昂贵时。利用符合预测原则的方法有望通过校准模型的不确定性为异常检测提供相应的统计保证。该工作引入了一个新颖的异常检测框架，称为交叉一致异常检测，建立在为预测任务设计的著名交叉一致方法之上。通过这种方法，他填补了在归纳一致异常检测环境中扩展先前研究的自然研究空白

    arXiv:2402.16388v1 Announce Type: cross  Abstract: Given the growing significance of reliable, trustworthy, and explainable machine learning, the requirement of uncertainty quantification for anomaly detection systems has become increasingly important. In this context, effectively controlling Type I error rates ($\alpha$) without compromising the statistical power ($1-\beta$) of these systems can build trust and reduce costs related to false discoveries, particularly when follow-up procedures are expensive. Leveraging the principles of conformal prediction emerges as a promising approach for providing respective statistical guarantees by calibrating a model's uncertainty. This work introduces a novel framework for anomaly detection, termed cross-conformal anomaly detection, building upon well-known cross-conformal methods designed for prediction tasks. With that, it addresses a natural research gap by extending previous works in the context of inductive conformal anomaly detection, rel
    
[^2]: Hilbert的投影度量用于有界增长函数和Sinkhorn算法的指数收敛

    Hilbert's projective metric for functions of bounded growth and exponential convergence of Sinkhorn's algorithm. (arXiv:2311.04041v2 [math.PR] UPDATED)

    [http://arxiv.org/abs/2311.04041](http://arxiv.org/abs/2311.04041)

    本文对有界增长可积函数空间的Hilbert投影度量进行了研究，提出了某些松弛锥体内的内核积分算子具有压缩映射的性质，并应用于熵最优传输问题中，证明了Sinkhorn算法在边际分布的尾部与成本函数增长适当时呈指数收敛。

    

    受无界环境中的熵最优传输问题的启发，我们研究了有界增长可积函数空间的Hilbert投影度量版本。这些Hilbert度量版本源自锥体，这些锥体是所有非负函数的松弛，即它们包括所有在与某些测试函数相乘时具有非负积分值的函数。我们证明了内核积分算子在适当的度量规范下是压缩映射，即使内核没有与零间隔，前提是内核趋向于零受控制。作为熵最优传输的应用，我们证明了在边际分布的尾部相对于成本函数的增长足够轻时，Sinkhorn算法呈指数收敛的性质。

    Motivated by the entropic optimal transport problem in unbounded settings, we study versions of Hilbert's projective metric for spaces of integrable functions of bounded growth. These versions of Hilbert's metric originate from cones which are relaxations of the cone of all non-negative functions, in the sense that they include all functions having non-negative integral values when multiplied with certain test functions. We show that kernel integral operators are contractions with respect to suitable specifications of such metrics even for kernels which are not bounded away from zero, provided that the decay to zero of the kernel is controlled. As an application to entropic optimal transport, we show exponential convergence of Sinkhorn's algorithm in settings where the marginal distributions have sufficiently light tails compared to the growth of the cost function.
    
[^3]: 使用数据自适应能量距离的高维数据稳健分类

    Robust Classification of High-Dimensional Data using Data-Adaptive Energy Distance. (arXiv:2306.13985v1 [stat.ML])

    [http://arxiv.org/abs/2306.13985](http://arxiv.org/abs/2306.13985)

    该论文提出了一种用于高维低样本量数据分类的稳健的数据自适应能量距离分类器，该分类器无需调参且在一定条件下可以实现完美分类，已在模拟研究和实际数据分析中得到证明比其他方法表现更优。

    

    在真实世界中，高维低样本量（HDLSS）数据的分类面临挑战，例如基因表达研究、癌症研究和医学成像等领域。本文提出了一些专门为HDLSS数据设计的分类器的开发和分析。这些分类器没有调节参数，并且是稳健的，因为它们不受底层数据分布的任何矩条件的影响。研究表明，在一些相当普遍的条件下，它们在HDLSS渐近区域内可以实现完美分类。还比较了所提出分类器的性能。我们的理论结果得到了广泛的模拟研究和实际数据分析的支持，证明了所提出分类技术优于几种广泛认可的方法的有希望优势。

    Classification of high-dimensional low sample size (HDLSS) data poses a challenge in a variety of real-world situations, such as gene expression studies, cancer research, and medical imaging. This article presents the development and analysis of some classifiers that are specifically designed for HDLSS data. These classifiers are free of tuning parameters and are robust, in the sense that they are devoid of any moment conditions of the underlying data distributions. It is shown that they yield perfect classification in the HDLSS asymptotic regime, under some fairly general conditions. The comparative performance of the proposed classifiers is also investigated. Our theoretical results are supported by extensive simulation studies and real data analysis, which demonstrate promising advantages of the proposed classification techniques over several widely recognized methods.
    

