# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Opening Up the Neural Network Classifier for Shap Score Computation.](http://arxiv.org/abs/2303.06516) | 本文提出了一种高效计算机器学习模型分类中Shap解释分数的方法，通过将二进制神经网络转换为布尔电路，并使用知识编译技术，将电路视为开放式模型，通过最近的高效算法计算Shap分数，相比于将BNN视为黑盒模型直接计算Shap，性能有了显著的提高。 |
| [^2] | [Interpretable Outlier Summarization.](http://arxiv.org/abs/2303.06261) | STAIR提出了一种可解释的异常值汇总方法，通过学习一组紧凑的人类可理解规则，以汇总和解释异常检测结果，具有强大的可解释性，以准确地总结检测结果。 |
| [^3] | [A Dataset for Learning Graph Representations to Predict Customer Returns in Fashion Retail.](http://arxiv.org/abs/2302.14096) | 该论文介绍了一个由ASOS收集的新型数据集，用于解决时尚零售生态系统中预测客户退货的挑战。研究者使用图表示学习方法，提高了退货预测分类任务的F1分数至0.792，这比其他模型有所改进。 |
| [^4] | [Separate and conquer heuristic allows robust mining of contrast sets in classification, regression, and survival data.](http://arxiv.org/abs/2204.00497) | 本文提出了一种基于分离征服的对比集挖掘算法RuleKit-CS，该算法通过多次通过伴随属性惩罚方案提供描述具有不同属性的相同示例的对比集，区别于标准的分离征服。该算法还被推广到回归和生存数据，允许识别标签属性/生存预测与预定义对比组的标签/预测一致的对比集。 |

# 详细

[^1]: 打开神经网络分类器以计算Shap分数

    Opening Up the Neural Network Classifier for Shap Score Computation. (arXiv:2303.06516v1 [cs.AI])

    [http://arxiv.org/abs/2303.06516](http://arxiv.org/abs/2303.06516)

    本文提出了一种高效计算机器学习模型分类中Shap解释分数的方法，通过将二进制神经网络转换为布尔电路，并使用知识编译技术，将电路视为开放式模型，通过最近的高效算法计算Shap分数，相比于将BNN视为黑盒模型直接计算Shap，性能有了显著的提高。

    This paper proposes an efficient method for computing Shap explanation scores in machine learning model classification by transforming binary neural networks into Boolean circuits and treating the resulting circuit as an open-box model, which leads to a significant improvement in performance compared to computing Shap directly on the BNN treated as a black-box model.

    我们解决了使用机器学习模型进行分类的Shap解释分数的高效计算问题。为此，我们展示了将二进制神经网络（BNN）转换为确定性和可分解的布尔电路，使用知识编译技术。所得到的电路被视为开放式模型，通过最近的高效算法计算Shap分数。详细的实验表明，与将BNN视为黑盒模型直接计算Shap相比，性能有了显著的提高。

    We address the problem of efficiently computing Shap explanation scores for classifications with machine learning models. With this goal, we show the transformation of binary neural networks (BNNs) for classification into deterministic and decomposable Boolean circuits, for which knowledge compilation techniques are used. The resulting circuit is treated as an open-box model, to compute Shap scores by means of a recent efficient algorithm for this class of circuits. Detailed experiments show a considerable gain in performance in comparison with computing Shap directly on the BNN treated as a black-box model.
    
[^2]: 可解释的异常值汇总

    Interpretable Outlier Summarization. (arXiv:2303.06261v1 [cs.LG])

    [http://arxiv.org/abs/2303.06261](http://arxiv.org/abs/2303.06261)

    STAIR提出了一种可解释的异常值汇总方法，通过学习一组紧凑的人类可理解规则，以汇总和解释异常检测结果，具有强大的可解释性，以准确地总结检测结果。

    STAIR proposes an interpretable outlier summarization method by learning a compact set of human understandable rules to summarize and explain the anomaly detection results, which has strong interpretability to accurately summarize the detection results.

    异常值检测在实际应用中是至关重要的，以防止金融欺诈、防御网络入侵或检测即将发生的设备故障。为了减少人力评估异常值检测结果的工作量，并有效地将异常值转化为可操作的见解，用户通常希望系统自动产生可解释的异常值检测结果的子组的汇总。然而，到目前为止，没有这样的系统存在。为了填补这一空白，我们提出了STAIR，它学习了一组紧凑的人类可理解规则，以汇总和解释异常检测结果。STAIR不使用经典的决策树算法来产生这些规则，而是提出了一个新的优化目标，以产生少量规则，具有最小的复杂性，因此具有强大的可解释性，以准确地总结检测结果。STAIR的学习算法通过迭代分割大规则来产生规则集，并在每个i中最大化这个目标，是最优的。

    Outlier detection is critical in real applications to prevent financial fraud, defend network intrusions, or detecting imminent device failures. To reduce the human effort in evaluating outlier detection results and effectively turn the outliers into actionable insights, the users often expect a system to automatically produce interpretable summarizations of subgroups of outlier detection results. Unfortunately, to date no such systems exist. To fill this gap, we propose STAIR which learns a compact set of human understandable rules to summarize and explain the anomaly detection results. Rather than use the classical decision tree algorithms to produce these rules, STAIR proposes a new optimization objective to produce a small number of rules with least complexity, hence strong interpretability, to accurately summarize the detection results. The learning algorithm of STAIR produces a rule set by iteratively splitting the large rules and is optimal in maximizing this objective in each i
    
[^3]: 一份用于学习图表示以预测时尚零售客户退货的数据集

    A Dataset for Learning Graph Representations to Predict Customer Returns in Fashion Retail. (arXiv:2302.14096v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.14096](http://arxiv.org/abs/2302.14096)

    该论文介绍了一个由ASOS收集的新型数据集，用于解决时尚零售生态系统中预测客户退货的挑战。研究者使用图表示学习方法，提高了退货预测分类任务的F1分数至0.792，这比其他模型有所改进。

    This paper introduces a novel dataset collected by ASOS for predicting customer returns in a fashion retail ecosystem. The researchers use Graph Representation Learning to improve the F1-score of the return prediction classification task to 0.792, outperforming other models.

    我们提出了一个由ASOS（一家主要的在线时尚零售商）收集的新型数据集，以解决在时尚零售生态系统中预测客户退货的挑战。通过发布这个庞大的数据集，我们希望激发研究社区和时尚行业之间的进一步合作。我们首先探讨了这个数据集的结构，重点关注图表示学习的应用，以利用自然数据结构并提供对数据中特定特征的统计洞察。除此之外，我们展示了一个退货预测分类任务的示例，其中包括一些基线模型（即没有中间表示学习步骤）和基于图表示的模型。我们展示了在下游退货预测分类任务中，使用图神经网络（GNN）可以找到F1分数为0.792，这比本文讨论的其他模型有所改进。除了这个增加的F1分数，我们还提出了一个l

    We present a novel dataset collected by ASOS (a major online fashion retailer) to address the challenge of predicting customer returns in a fashion retail ecosystem. With the release of this substantial dataset we hope to motivate further collaboration between research communities and the fashion industry. We first explore the structure of this dataset with a focus on the application of Graph Representation Learning in order to exploit the natural data structure and provide statistical insights into particular features within the data. In addition to this, we show examples of a return prediction classification task with a selection of baseline models (i.e. with no intermediate representation learning step) and a graph representation based model. We show that in a downstream return prediction classification task, an F1-score of 0.792 can be found using a Graph Neural Network (GNN), improving upon other models discussed in this work. Alongside this increased F1-score, we also present a l
    
[^4]: 分离征服启发式算法允许在分类、回归和生存数据中进行强大的对比集挖掘

    Separate and conquer heuristic allows robust mining of contrast sets in classification, regression, and survival data. (arXiv:2204.00497v3 [cs.DB] UPDATED)

    [http://arxiv.org/abs/2204.00497](http://arxiv.org/abs/2204.00497)

    本文提出了一种基于分离征服的对比集挖掘算法RuleKit-CS，该算法通过多次通过伴随属性惩罚方案提供描述具有不同属性的相同示例的对比集，区别于标准的分离征服。该算法还被推广到回归和生存数据，允许识别标签属性/生存预测与预定义对比组的标签/预测一致的对比集。

    This paper proposes a contrast set mining algorithm, RuleKit-CS, based on the separate and conquer heuristic, which provides contrast sets describing the same examples with different attributes through multiple passes accompanied with an attribute penalization scheme. The algorithm is also generalized for regression and survival data, allowing identification of contrast sets whose label attribute/survival prognosis is consistent with the label/prognosis for the predefined contrast groups.

    识别群体之间的差异是最重要的知识发现问题之一。该过程，也称为对比集挖掘，在医学、工业或经济等广泛领域中应用。在本文中，我们提出了RuleKit-CS，一种基于分离征服的对比集挖掘算法——一种用于决策规则归纳的成熟启发式算法。多次通过伴随属性惩罚方案提供描述具有不同属性的相同示例的对比集，区别于标准的分离征服。该算法还被推广到回归和生存数据，允许识别标签属性/生存预测与预定义对比组的标签/预测一致的对比集。这个特性，不是现有方法所提供的，进一步扩展了RuleKit-CS的可用性。在来自各个领域的130多个数据集上进行的实验和详细分析。

    Identifying differences between groups is one of the most important knowledge discovery problems. The procedure, also known as contrast sets mining, is applied in a wide range of areas like medicine, industry, or economics.  In the paper we present RuleKit-CS, an algorithm for contrast set mining based on separate and conquer - a well established heuristic for decision rule induction. Multiple passes accompanied with an attribute penalization scheme provide contrast sets describing same examples with different attributes, distinguishing presented approach from the standard separate and conquer. The algorithm was also generalized for regression and survival data allowing identification of contrast sets whose label attribute/survival prognosis is consistent with the label/prognosis for the predefined contrast groups. This feature, not provided by the existing approaches, further extends the usability of RuleKit-CS.  Experiments on over 130 data sets from various areas and detailed analys
    

