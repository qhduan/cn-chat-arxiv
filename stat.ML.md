# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Random-Set Convolutional Neural Network (RS-CNN) for Epistemic Deep Learning.](http://arxiv.org/abs/2307.05772) | 这篇论文提出了一种新的随机集合卷积神经网络（RS-CNN）用于分类，通过预测信念函数而不是概率矢量集合，以表示模型的置信度和认识不确定性。基于认识论深度学习方法，该模型能够估计由有限训练集引起的认识不确定性。 |
| [^2] | [Sources of Uncertainty in Machine Learning -- A Statisticians' View.](http://arxiv.org/abs/2305.16703) | 本文讨论了机器学习中不确定性的来源和类型，从统计学家的视角出发，分类别介绍了随机性和认知性不确定性的概念，证明了不确定性来源各异，不可简单归为两类。同时，与统计学概念进行类比，探讨不确定性在机器学习中的作用。 |

# 详细

[^1]: 随机集合卷积神经网络（RS-CNN）用于认识论深度学习

    Random-Set Convolutional Neural Network (RS-CNN) for Epistemic Deep Learning. (arXiv:2307.05772v1 [cs.LG])

    [http://arxiv.org/abs/2307.05772](http://arxiv.org/abs/2307.05772)

    这篇论文提出了一种新的随机集合卷积神经网络（RS-CNN）用于分类，通过预测信念函数而不是概率矢量集合，以表示模型的置信度和认识不确定性。基于认识论深度学习方法，该模型能够估计由有限训练集引起的认识不确定性。

    

    机器学习越来越多地应用于安全关键领域，对抗攻击的鲁棒性至关重要，错误的预测可能导致潜在的灾难性后果。这突出了学习系统需要能够确定模型对其预测的置信度以及与之相关联的认识不确定性的手段，“知道一个模型不知道”。在本文中，我们提出了一种新颖的用于分类的随机集合卷积神经网络（RS-CNN），其预测信念函数而不是概率矢量集合，使用随机集合的数学，即对样本空间的幂集的分布。基于认识论深度学习方法，随机集模型能够表示机器学习中由有限训练集引起的“认识性”不确定性。我们通过近似预测信念函数相关联的置信集的大小来估计认识不确定性。

    Machine learning is increasingly deployed in safety-critical domains where robustness against adversarial attacks is crucial and erroneous predictions could lead to potentially catastrophic consequences. This highlights the need for learning systems to be equipped with the means to determine a model's confidence in its prediction and the epistemic uncertainty associated with it, 'to know when a model does not know'. In this paper, we propose a novel Random-Set Convolutional Neural Network (RS-CNN) for classification which predicts belief functions rather than probability vectors over the set of classes, using the mathematics of random sets, i.e., distributions over the power set of the sample space. Based on the epistemic deep learning approach, random-set models are capable of representing the 'epistemic' uncertainty induced in machine learning by limited training sets. We estimate epistemic uncertainty by approximating the size of credal sets associated with the predicted belief func
    
[^2]: 机器学习中的不确定性来源 -- 一个统计学家的视角

    Sources of Uncertainty in Machine Learning -- A Statisticians' View. (arXiv:2305.16703v1 [stat.ML])

    [http://arxiv.org/abs/2305.16703](http://arxiv.org/abs/2305.16703)

    本文讨论了机器学习中不确定性的来源和类型，从统计学家的视角出发，分类别介绍了随机性和认知性不确定性的概念，证明了不确定性来源各异，不可简单归为两类。同时，与统计学概念进行类比，探讨不确定性在机器学习中的作用。

    

    机器学习和深度学习已经取得了令人瞩目的成就，使我们能够回答几年前难以想象的问题。除了这些成功之外，越来越清晰的是，在纯预测之外，量化不确定性也是相关和必要的。虽然近年来已经出现了这方面的第一批概念和思想，但本文采用了一个概念性的视角，并探讨了可能的不确定性来源。通过采用统计学家的视角，我们讨论了与机器学习更常见相关的随机性和认知性不确定性的概念。本文旨在规范这两种类型的不确定性，并证明不确定性的来源各异，并且不总是可以分解为随机性和认知性。通过将统计概念与机器学习中的不确定性进行类比，我们也展示了统计学概念和机器学习中不确定性的作用。

    Machine Learning and Deep Learning have achieved an impressive standard today, enabling us to answer questions that were inconceivable a few years ago. Besides these successes, it becomes clear, that beyond pure prediction, which is the primary strength of most supervised machine learning algorithms, the quantification of uncertainty is relevant and necessary as well. While first concepts and ideas in this direction have emerged in recent years, this paper adopts a conceptual perspective and examines possible sources of uncertainty. By adopting the viewpoint of a statistician, we discuss the concepts of aleatoric and epistemic uncertainty, which are more commonly associated with machine learning. The paper aims to formalize the two types of uncertainty and demonstrates that sources of uncertainty are miscellaneous and can not always be decomposed into aleatoric and epistemic. Drawing parallels between statistical concepts and uncertainty in machine learning, we also demonstrate the rol
    

