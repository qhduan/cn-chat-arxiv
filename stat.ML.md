# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stacking as Accelerated Gradient Descent](https://arxiv.org/abs/2403.04978) | Stacking提出了一种理论解释，即实现了Nesterov的加速梯度下降形式，并证明对于某些深度线性残差网络，提供了加速训练。 |
| [^2] | [Understanding Self-Distillation and Partial Label Learning in Multi-Class Classification with Label Noise](https://arxiv.org/abs/2402.10482) | 自蒸馏在多类别分类中扮演着标签平均化的角色，有助于模型关注与特定实例相关的特征簇以预测标签，但随着蒸馏轮次增加，性能会降低。此外，在标签噪声情景下自蒸馏被证明是有效的，找到了实现100%分类准确率所需的最小蒸馏轮次。 |

# 详细

[^1]: Stacking作为加速梯度下降算法

    Stacking as Accelerated Gradient Descent

    [https://arxiv.org/abs/2403.04978](https://arxiv.org/abs/2403.04978)

    Stacking提出了一种理论解释，即实现了Nesterov的加速梯度下降形式，并证明对于某些深度线性残差网络，提供了加速训练。

    

    Stacking是一种启发式技术，通过逐渐增加层数并通过从旧层复制参数来初始化新层，用于训练深度残差网络，已经被证明在提高深度神经网络训练效率方面非常成功。本文提出了对于Stacking有效性的理论解释：即，Stacking实现了Nesterov的加速梯度下降的一种形式。该理论还涵盖了诸如提升方法中构建的加法集成等更简单的模型，并为每一轮提升过程中初始化新分类器的类似广泛使用的实用启发式提供了解释。我们还证明了对于某些深度线性残差网络，通过对Nesterov的加速梯度方法的一个新的潜能函数分析，Stacking确实提供了加速训练，从而允许更新中的误差。我们进行了概念验证实验来验证我们的理论。

    arXiv:2403.04978v1 Announce Type: new  Abstract: Stacking, a heuristic technique for training deep residual networks by progressively increasing the number of layers and initializing new layers by copying parameters from older layers, has proven quite successful in improving the efficiency of training deep neural networks. In this paper, we propose a theoretical explanation for the efficacy of stacking: viz., stacking implements a form of Nesterov's accelerated gradient descent. The theory also covers simpler models such as the additive ensembles constructed in boosting methods, and provides an explanation for a similar widely-used practical heuristic for initializing the new classifier in each round of boosting. We also prove that for certain deep linear residual networks, stacking does provide accelerated training, via a new potential function analysis of the Nesterov's accelerated gradient method which allows errors in updates. We conduct proof-of-concept experiments to validate our
    
[^2]: 理解带有标签噪音的多类别分类中的自蒸馏和部分标签学习

    Understanding Self-Distillation and Partial Label Learning in Multi-Class Classification with Label Noise

    [https://arxiv.org/abs/2402.10482](https://arxiv.org/abs/2402.10482)

    自蒸馏在多类别分类中扮演着标签平均化的角色，有助于模型关注与特定实例相关的特征簇以预测标签，但随着蒸馏轮次增加，性能会降低。此外，在标签噪声情景下自蒸馏被证明是有效的，找到了实现100%分类准确率所需的最小蒸馏轮次。

    

    自蒸馏（SD）是使用教师模型的输出训练学生模型的过程，两个模型共享相同的架构。我们的研究从理论上考察了使用交叉熵损失的多类别分类中的SD，探索了多轮SD和具有精炼教师输出的SD，这些灵感来自部分标签学习（PLL）。通过推导学生模型输出的封闭形式解，我们发现SD本质上是在具有高特征相关性的实例之间进行标签平均。最初有益的平均化有助于模型专注于与给定实例相关联的特征簇以预测标签。然而，随着蒸馏轮次的增加，性能会下降。此外，我们展示了SD在标签噪声情景中的有效性，并确定实现100%分类准确率所需的标签损坏条件和最小蒸馏轮次数。

    arXiv:2402.10482v1 Announce Type: new  Abstract: Self-distillation (SD) is the process of training a student model using the outputs of a teacher model, with both models sharing the same architecture. Our study theoretically examines SD in multi-class classification with cross-entropy loss, exploring both multi-round SD and SD with refined teacher outputs, inspired by partial label learning (PLL). By deriving a closed-form solution for the student model's outputs, we discover that SD essentially functions as label averaging among instances with high feature correlations. Initially beneficial, this averaging helps the model focus on feature clusters correlated with a given instance for predicting the label. However, it leads to diminishing performance with increasing distillation rounds. Additionally, we demonstrate SD's effectiveness in label noise scenarios and identify the label corruption condition and minimum number of distillation rounds needed to achieve 100% classification accur
    

