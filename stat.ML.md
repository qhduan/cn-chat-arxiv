# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Representing Random Utility Choice Models with Neural Networks.](http://arxiv.org/abs/2207.12877) | 本论文提出了一种基于神经网络的离散选择模型类，RUMnets，可以近似表示任何随机效用最大化推导出的模型，并且在选择数据上有良好的预测能力。 |

# 详细

[^1]: 用神经网络表示随机效用选择模型

    Representing Random Utility Choice Models with Neural Networks. (arXiv:2207.12877v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.12877](http://arxiv.org/abs/2207.12877)

    本论文提出了一种基于神经网络的离散选择模型类，RUMnets，可以近似表示任何随机效用最大化推导出的模型，并且在选择数据上有良好的预测能力。

    

    在深度学习的成功之下，我们提出了一种基于神经网络的离散选择模型类，称为RUMnets，受随机效用最大化（RUM）框架的启发。该模型使用样本平均逼近来构建代理人的随机效用函数。我们证明了RUMnets可以对RUM离散选择模型类进行尖锐逼近：任何从随机效用最大化推导出的模型都可以被RUMnet无限接近地逼近。相反地，任何RUMnet都符合RUM原则。我们得到了在选择数据上拟合的RUMnet的泛化误差的上界，并且根据数据集和架构的关键参数，获得了关于其在新的未知数据上预测选择能力的理论洞见。通过利用神经网络的开源库，我们发现RUMnets在预测准确性方面与几种选择建模和机器学习方法具有竞争力。

    Motivated by the successes of deep learning, we propose a class of neural network-based discrete choice models, called RUMnets, inspired by the random utility maximization (RUM) framework. This model formulates the agents' random utility function using a sample average approximation. We show that RUMnets sharply approximate the class of RUM discrete choice models: any model derived from random utility maximization has choice probabilities that can be approximated arbitrarily closely by a RUMnet. Reciprocally, any RUMnet is consistent with the RUM principle. We derive an upper bound on the generalization error of RUMnets fitted on choice data, and gain theoretical insights on their ability to predict choices on new, unseen data depending on critical parameters of the dataset and architecture. By leveraging open-source libraries for neural networks, we find that RUMnets are competitive against several choice modeling and machine learning methods in terms of predictive accuracy on two rea
    

