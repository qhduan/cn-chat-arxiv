# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tactile-based Object Retrieval From Granular Media](https://arxiv.org/abs/2402.04536) | 这项研究介绍了一种基于触觉反馈的机器人操作方法，用于在颗粒介质中检索埋藏的物体。通过模拟传感器噪声进行端到端训练，实现了自然出现的学习推动行为，并成功将其迁移到实际硬件上。 |
| [^2] | [Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier](https://arxiv.org/abs/2212.04382) | 本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。 |
| [^3] | [FLex&Chill: Improving Local Federated Learning Training with Logit Chilling.](http://arxiv.org/abs/2401.09986) | FLex&Chill 提出了一种通过Logit Chilling方法改进本地联合学习训练的方法，可以加快模型收敛并提高推理精度。 |
| [^4] | [Boosted Control Functions.](http://arxiv.org/abs/2310.05805) | 本研究通过建立同时方程模型和控制函数与分布概括的新连接，解决了在存在未观察到的混淆情况下，针对不同训练和测试分布的预测问题。 |
| [^5] | [Behavioral Machine Learning? Computer Predictions of Corporate Earnings also Overreact.](http://arxiv.org/abs/2303.16158) | 本文研究发现，机器学习算法可以更准确地预测公司盈利，但同样存在过度反应的问题，而传统培训的股市分析师和经过机器学习方法培训的分析师相比会产生较少的过度反应。 |

# 详细

[^1]: 基于触觉的从颗粒介质中检索物体的研究

    Tactile-based Object Retrieval From Granular Media

    [https://arxiv.org/abs/2402.04536](https://arxiv.org/abs/2402.04536)

    这项研究介绍了一种基于触觉反馈的机器人操作方法，用于在颗粒介质中检索埋藏的物体。通过模拟传感器噪声进行端到端训练，实现了自然出现的学习推动行为，并成功将其迁移到实际硬件上。

    

    我们介绍了一种名为GEOTACT的机器人操作方法，能够在颗粒介质中检索埋藏的物体。这是一项具有挑战性的任务，因为需要与颗粒介质进行交互，并且仅依靠触觉反馈来完成，因为一个埋藏的物体可能完全被视觉隐藏。在这种环境中，触觉反馈本身具有挑战性，因为需要与周围介质进行普遍接触，并且由触觉读数引起的固有噪声水平。为了解决这些挑战，我们使用了一种通过模拟传感器噪声进行端到端训练的学习方法。我们展示了我们的问题表述导致了学习推动行为的自然出现，操作器使用这些行为来减少不确定性并将物体引导到稳定的抓取位置，尽管存在假的和噪声的触觉读数。我们还引入了一种培训方案，可以在仿真中学习这些行为，并在实际硬件上进行零样本迁移。据我们所知，GEOTACT是第一个这样的方法。

    We introduce GEOTACT, a robotic manipulation method capable of retrieving objects buried in granular media. This is a challenging task due to the need to interact with granular media, and doing so based exclusively on tactile feedback, since a buried object can be completely hidden from vision. Tactile feedback is in itself challenging in this context, due to ubiquitous contact with the surrounding media, and the inherent noise level induced by the tactile readings. To address these challenges, we use a learning method trained end-to-end with simulated sensor noise. We show that our problem formulation leads to the natural emergence of learned pushing behaviors that the manipulator uses to reduce uncertainty and funnel the object to a stable grasp despite spurious and noisy tactile readings. We also introduce a training curriculum that enables learning these behaviors in simulation, followed by zero-shot transfer to real hardware. To the best of our knowledge, GEOTACT is the first meth
    
[^2]: 分类器边界的结构：朴素贝叶斯分类器的案例研究

    Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier

    [https://arxiv.org/abs/2212.04382](https://arxiv.org/abs/2212.04382)

    本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。

    

    无论基于模型、训练数据还是二者组合，分类器将（可能复杂的）输入数据归入相对较少的输出类别之一。本文研究在输入空间为图的情况下，边界的结构——那些被分类为不同类别的邻近点——的特性。我们的科学背景是基于模型的朴素贝叶斯分类器，用于处理由下一代测序仪生成的DNA读数。我们展示了边界既是巨大的，又具有复杂的结构。我们创建了一种新的不确定性度量，称为邻居相似度，它将一个点的结果与其邻居的结果分布进行比较。这个度量不仅追踪了贝叶斯分类器的两个固有不确定性度量，还可以在没有固有不确定性度量的分类器上实现，但需要计算成本。

    Whether based on models, training data or a combination, classifiers place (possibly complex) input data into one of a relatively small number of output categories. In this paper, we study the structure of the boundary--those points for which a neighbor is classified differently--in the context of an input space that is a graph, so that there is a concept of neighboring inputs, The scientific setting is a model-based naive Bayes classifier for DNA reads produced by Next Generation Sequencers. We show that the boundary is both large and complicated in structure. We create a new measure of uncertainty, called Neighbor Similarity, that compares the result for a point to the distribution of results for its neighbors. This measure not only tracks two inherent uncertainty measures for the Bayes classifier, but also can be implemented, at a computational cost, for classifiers without inherent measures of uncertainty.
    
[^3]: FLex&Chill：通过Logit Chilling改进本地联合学习训练

    FLex&Chill: Improving Local Federated Learning Training with Logit Chilling. (arXiv:2401.09986v1 [cs.LG])

    [http://arxiv.org/abs/2401.09986](http://arxiv.org/abs/2401.09986)

    FLex&Chill 提出了一种通过Logit Chilling方法改进本地联合学习训练的方法，可以加快模型收敛并提高推理精度。

    

    联合学习由于本地客户端的非iid分布式训练数据而受到数据异质性的阻碍。我们提出了一种新的联合学习模型训练方法FLex&Chill，利用了Logit Chilling方法。通过广泛的评估，我们证明在联合学习系统中固有的非iid数据特征存在的情况下，这种方法可以加快模型收敛并提高推理精度。从我们的实验中，我们观察到全局联合学习模型收敛时间提高了6倍，推理精度提高了3.37%。

    Federated learning are inherently hampered by data heterogeneity: non-iid distributed training data over local clients. We propose a novel model training approach for federated learning, FLex&Chill, which exploits the Logit Chilling method. Through extensive evaluations, we demonstrate that, in the presence of non-iid data characteristics inherent in federated learning systems, this approach can expedite model convergence and improve inference accuracy. Quantitatively, from our experiments, we observe up to 6X improvement in the global federated learning model convergence time, and up to 3.37% improvement in inference accuracy.
    
[^4]: 提升控制函数

    Boosted Control Functions. (arXiv:2310.05805v1 [stat.ML])

    [http://arxiv.org/abs/2310.05805](http://arxiv.org/abs/2310.05805)

    本研究通过建立同时方程模型和控制函数与分布概括的新连接，解决了在存在未观察到的混淆情况下，针对不同训练和测试分布的预测问题。

    

    现代机器学习方法和大规模数据的可用性为从大量的协变量中准确预测目标数量打开了大门。然而，现有的预测方法在训练和测试数据不同的情况下表现不佳，尤其是在存在隐藏混淆的情况下。虽然对因果效应估计（例如仪器变量）已经对隐藏混淆进行了深入研究，但对于预测任务来说并非如此。本研究旨在填补这一空白，解决在存在未观察到的混淆的情况下，针对不同训练和测试分布的预测问题。具体而言，我们在机器学习的分布概括领域，以及计量经济学中的同时方程模型和控制函数之间建立了一种新的联系。我们的贡献的核心是描述在一组分布转变下的数据生成过程的分布概括同时方程模型（SIMDGs）。

    Modern machine learning methods and the availability of large-scale data opened the door to accurately predict target quantities from large sets of covariates. However, existing prediction methods can perform poorly when the training and testing data are different, especially in the presence of hidden confounding. While hidden confounding is well studied for causal effect estimation (e.g., instrumental variables), this is not the case for prediction tasks. This work aims to bridge this gap by addressing predictions under different training and testing distributions in the presence of unobserved confounding. In particular, we establish a novel connection between the field of distribution generalization from machine learning, and simultaneous equation models and control function from econometrics. Central to our contribution are simultaneous equation models for distribution generalization (SIMDGs) which describe the data-generating process under a set of distributional shifts. Within thi
    
[^5]: 机器学习准确预测财报，但同样存在过度反应

    Behavioral Machine Learning? Computer Predictions of Corporate Earnings also Overreact. (arXiv:2303.16158v1 [q-fin.ST])

    [http://arxiv.org/abs/2303.16158](http://arxiv.org/abs/2303.16158)

    本文研究发现，机器学习算法可以更准确地预测公司盈利，但同样存在过度反应的问题，而传统培训的股市分析师和经过机器学习方法培训的分析师相比会产生较少的过度反应。

    

    大量证据表明，在金融领域中，机器学习算法的预测能力比人类更为准确。但是，文献并未测试算法预测是否更为理性。本文研究了几个算法（包括线性回归和一种名为Gradient Boosted Regression Trees的流行算法）对于公司盈利的预测结果。结果发现，GBRT平均胜过线性回归和人类股市分析师，但仍存在过度反应且无法满足理性预期标准。通过降低学习率，可最小程度上减少过度反应程度，但这会牺牲预测准确性。通过机器学习方法培训过的股市分析师比传统训练的分析师产生的过度反应较少。此外，股市分析师的预测反映出机器算法没有捕捉到的信息。

    There is considerable evidence that machine learning algorithms have better predictive abilities than humans in various financial settings. But, the literature has not tested whether these algorithmic predictions are more rational than human predictions. We study the predictions of corporate earnings from several algorithms, notably linear regressions and a popular algorithm called Gradient Boosted Regression Trees (GBRT). On average, GBRT outperformed both linear regressions and human stock analysts, but it still overreacted to news and did not satisfy rational expectation as normally defined. By reducing the learning rate, the magnitude of overreaction can be minimized, but it comes with the cost of poorer out-of-sample prediction accuracy. Human stock analysts who have been trained in machine learning methods overreact less than traditionally trained analysts. Additionally, stock analyst predictions reflect information not otherwise available to machine algorithms.
    

