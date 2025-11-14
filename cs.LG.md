# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Effector: A Python package for regional explanations](https://arxiv.org/abs/2404.02629) | Effector是一个专注于区域特征效果的Python软件包，通过引入区域效果来降低全局特征效果方法中可能的异质性。 |
| [^2] | [Spectral methods for Neural Integral Equations](https://arxiv.org/abs/2312.05654) | 本文引入了一个基于谱方法的神经积分方程框架，通过在谱域中学习算子，降低了计算成本，并保证了高插值精度。 |
| [^3] | [Investigating Feature and Model Importance in Android Malware Detection: An Implemented Survey and Experimental Comparison of ML-Based Methods](https://arxiv.org/abs/2301.12778) | 本文重新实现和评估了18个代表性的过去研究，并在包含124,000个应用程序的平衡、相关和最新数据集上进行了新实验，发现仅通过静态分析提取的特征就能实现高达96.8%的恶意软件检测准确性。 |
| [^4] | [Provably Scalable Black-Box Variational Inference with Structured Variational Families.](http://arxiv.org/abs/2401.10989) | 本文研究了均值场变分族和满秩变分族之间的理论中间地带：结构化变分族，并通过理论证明结构化变分族可以在迭代复杂性上表现更好，缩放效果更好。 |

# 详细

[^1]: Effector: 一个用于区域解释的Python软件包

    Effector: A Python package for regional explanations

    [https://arxiv.org/abs/2404.02629](https://arxiv.org/abs/2404.02629)

    Effector是一个专注于区域特征效果的Python软件包，通过引入区域效果来降低全局特征效果方法中可能的异质性。

    

    全局特征效果方法解释一个输出模型，每个特征对应一个图。该图显示特征对输出的平均效果，例如年龄对年收入的影响。然而，当由异质局部效果推导出平均效果时，平均效果可能具有误导性，即明显偏离平均值。为了减少异质性，区域效果为每个特征提供多个图，每个图代表特定子空间内的平均效果。为了可解释性，子空间被定义为由逻辑规则链定义的超矩形，例如年龄对男性和女性的年收入的影响，以及不同专业经验水平。我们介绍了Effector，一个致力于区域特征效果的Python库。Effector实现了一些成熟的全局效果方法，评估每种方法的异质性，并基于此提供区域效果。

    arXiv:2404.02629v1 Announce Type: new  Abstract: Global feature effect methods explain a model outputting one plot per feature. The plot shows the average effect of the feature on the output, like the effect of age on the annual income. However, average effects may be misleading when derived from local effects that are heterogeneous, i.e., they significantly deviate from the average. To decrease the heterogeneity, regional effects provide multiple plots per feature, each representing the average effect within a specific subspace. For interpretability, subspaces are defined as hyperrectangles defined by a chain of logical rules, like age's effect on annual income separately for males and females and different levels of professional experience. We introduce Effector, a Python library dedicated to regional feature effects. Effector implements well-established global effect methods, assesses the heterogeneity of each method and, based on that, provides regional effects. Effector automatica
    
[^2]: 神经积分方程的谱方法

    Spectral methods for Neural Integral Equations

    [https://arxiv.org/abs/2312.05654](https://arxiv.org/abs/2312.05654)

    本文引入了一个基于谱方法的神经积分方程框架，通过在谱域中学习算子，降低了计算成本，并保证了高插值精度。

    

    arXiv:2312.05654v3 公告类型：替换-跨交摘要：神经积分方程是基于积分方程理论的深度学习模型，其中模型由积分算子和通过优化过程学习的相应方程（第二种）组成。这种方法允许利用机器学习中积分算子的非局部特性，但计算成本很高。在本文中，我们介绍了基于谱方法的神经积分方程框架，该方法使我们能够在谱域中学习一个算子，从而降低计算成本，同时保证高插值精度。我们研究了我们方法的性质，并展示了关于模型近似能力和收敛到数值方法解的各种理论保证。我们提供了数值实验来展示所得模型的实际有效性。

    arXiv:2312.05654v3 Announce Type: replace-cross  Abstract: Neural integral equations are deep learning models based on the theory of integral equations, where the model consists of an integral operator and the corresponding equation (of the second kind) which is learned through an optimization procedure. This approach allows to leverage the nonlocal properties of integral operators in machine learning, but it is computationally expensive. In this article, we introduce a framework for neural integral equations based on spectral methods that allows us to learn an operator in the spectral domain, resulting in a cheaper computational cost, as well as in high interpolation accuracy. We study the properties of our methods and show various theoretical guarantees regarding the approximation capabilities of the model, and convergence to solutions of the numerical methods. We provide numerical experiments to demonstrate the practical effectiveness of the resulting model.
    
[^3]: 探究安卓恶意软件检测中的特征和模型重要性：一项实施调查和机器学习方法实验比较

    Investigating Feature and Model Importance in Android Malware Detection: An Implemented Survey and Experimental Comparison of ML-Based Methods

    [https://arxiv.org/abs/2301.12778](https://arxiv.org/abs/2301.12778)

    本文重新实现和评估了18个代表性的过去研究，并在包含124,000个应用程序的平衡、相关和最新数据集上进行了新实验，发现仅通过静态分析提取的特征就能实现高达96.8%的恶意软件检测准确性。

    

    Android的普及意味着它成为恶意软件的常见目标。多年来，各种研究发现机器学习模型能够有效区分恶意软件和良性应用程序。然而，随着操作系统的演进，恶意软件也在不断发展，对先前研究的发现提出了质疑，其中许多报告称使用小型、过时且经常不平衡的数据集能够获得非常高的准确性。在本文中，我们重新实现了18项具代表性的过去研究并使用包括124,000个应用程序的平衡、相关且最新的数据集对它们进行重新评估。我们还进行了新的实验，以填补现有知识中的空白，并利用研究结果确定在当代环境中用于安卓恶意软件检测的最有效特征和模型。我们表明，仅通过静态分析提取的特征即可实现高达96.8%的检测准确性。

    arXiv:2301.12778v2 Announce Type: replace  Abstract: The popularity of Android means it is a common target for malware. Over the years, various studies have found that machine learning models can effectively discriminate malware from benign applications. However, as the operating system evolves, so does malware, bringing into question the findings of these previous studies, many of which report very high accuracies using small, outdated, and often imbalanced datasets. In this paper, we reimplement 18 representative past works and reevaluate them using a balanced, relevant, and up-to-date dataset comprising 124,000 applications. We also carry out new experiments designed to fill holes in existing knowledge, and use our findings to identify the most effective features and models to use for Android malware detection within a contemporary environment. We show that high detection accuracies (up to 96.8%) can be achieved using features extracted through static analysis alone, yielding a mode
    
[^4]: 具有结构化变分族的可证伸缩性黑盒变分推断

    Provably Scalable Black-Box Variational Inference with Structured Variational Families. (arXiv:2401.10989v1 [stat.ML])

    [http://arxiv.org/abs/2401.10989](http://arxiv.org/abs/2401.10989)

    本文研究了均值场变分族和满秩变分族之间的理论中间地带：结构化变分族，并通过理论证明结构化变分族可以在迭代复杂性上表现更好，缩放效果更好。

    

    已知具有满秩协方差逼近的变分族在黑盒变分推断中表现不佳，无论是从实证上还是理论上。事实上，最近对黑盒变分推断的计算复杂性结果表明，与均值场变分族相比，满秩变分族在问题的维度上扩展得很差。这对具有本地变量的分层贝叶斯模型尤为关键，它们的维度随着数据集的大小而增加。因此，迭代复杂性对数据集大小N存在明确的O(N^2)依赖。在本文中，我们探索了均值场变分族和满秩变分族之间的理论中间地带：结构化变分族。我们严格证明了某些尺度矩阵结构可以实现更好的迭代复杂性O(N)，从而与N的缩放更好地匹配。我们在现实中验证了我们的理论结果

    Variational families with full-rank covariance approximations are known not to work well in black-box variational inference (BBVI), both empirically and theoretically. In fact, recent computational complexity results for BBVI have established that full-rank variational families scale poorly with the dimensionality of the problem compared to e.g. mean field families. This is particularly critical to hierarchical Bayesian models with local variables; their dimensionality increases with the size of the datasets. Consequently, one gets an iteration complexity with an explicit $\mathcal{O}(N^2)$ dependence on the dataset size $N$. In this paper, we explore a theoretical middle ground between mean-field variational families and full-rank families: structured variational families. We rigorously prove that certain scale matrix structures can achieve a better iteration complexity of $\mathcal{O}(N)$, implying better scaling with respect to $N$. We empirically verify our theoretical results on l
    

