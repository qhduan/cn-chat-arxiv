# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Identifiable Latent Causal Content for Domain Adaptation under Latent Covariate Shift](https://arxiv.org/abs/2208.14161) | 提出了一种新的隐含协变量转移（LCS）范式，增加了领域间的可变性和适应性，并提供了恢复标签变量潜在原因的理论保证。 |
| [^2] | [Provable Guarantees for Nonlinear Feature Learning in Three-Layer Neural Networks.](http://arxiv.org/abs/2305.06986) | 本文研究了三层神经网络的特征学习能力，相比之下，它具有比两层网络更丰富的可证的特征学习能力，并提出了一个通用定理，限制了目标结构的样本复杂度和宽度，以实现低测试误差。 |
| [^3] | [Multi-mode fiber reservoir computing overcomes shallow neural networks classifiers.](http://arxiv.org/abs/2210.04745) | 多模光纤利用水库计算范例进行分类，精度高于直接训练原始图像和传统的传输矩阵模型。 |

# 详细

[^1]: 可识别的潜在因果内容用于隐含协变量转移下的领域自适应

    Identifiable Latent Causal Content for Domain Adaptation under Latent Covariate Shift

    [https://arxiv.org/abs/2208.14161](https://arxiv.org/abs/2208.14161)

    提出了一种新的隐含协变量转移（LCS）范式，增加了领域间的可变性和适应性，并提供了恢复标签变量潜在原因的理论保证。

    

    多源领域自适应（MSDA）解决了利用来自多个源域的标记数据和来自目标域的未标记数据来学习针对未标记目标领域的标签预测函数的挑战。我们提出了一种称为潜在协变量转移（LCS）的新范式，它引入了更大的领域间可变性和适应性。值得注意的是，它为恢复标签变量的潜在原因提供了理论保证。

    arXiv:2208.14161v3 Announce Type: replace  Abstract: Multi-source domain adaptation (MSDA) addresses the challenge of learning a label prediction function for an unlabeled target domain by leveraging both the labeled data from multiple source domains and the unlabeled data from the target domain. Conventional MSDA approaches often rely on covariate shift or conditional shift paradigms, which assume a consistent label distribution across domains. However, this assumption proves limiting in practical scenarios where label distributions do vary across domains, diminishing its applicability in real-world settings. For example, animals from different regions exhibit diverse characteristics due to varying diets and genetics.   Motivated by this, we propose a novel paradigm called latent covariate shift (LCS), which introduces significantly greater variability and adaptability across domains. Notably, it provides a theoretical assurance for recovering the latent cause of the label variable, w
    
[^2]: 三层神经网络中非线性特征学习的可证保证

    Provable Guarantees for Nonlinear Feature Learning in Three-Layer Neural Networks. (arXiv:2305.06986v1 [cs.LG])

    [http://arxiv.org/abs/2305.06986](http://arxiv.org/abs/2305.06986)

    本文研究了三层神经网络的特征学习能力，相比之下，它具有比两层网络更丰富的可证的特征学习能力，并提出了一个通用定理，限制了目标结构的样本复杂度和宽度，以实现低测试误差。

    

    深度学习理论中的一个核心问题是理解神经网络如何学习分层特征。深度网络提取显著特征的能力对其卓越的泛化能力和现代深度学习范式的预训练和微调至关重要。然而，从理论角度来看，这种特征学习过程仍然不够清晰，现有的分析主要局限于两层网络。在本文中，我们展示了三层神经网络具有证明的比两层网络更丰富的特征学习能力。我们分析了通过逐层梯度下降训练的三层网络学习的特征，并提出了一个通用定理，它上界了目标具有特定层次结构时实现低测试错误所需的样本复杂度和宽度。我们将我们的框架实例化到特定的统计学学习设置中——单指数模型和二次函数。

    One of the central questions in the theory of deep learning is to understand how neural networks learn hierarchical features. The ability of deep networks to extract salient features is crucial to both their outstanding generalization ability and the modern deep learning paradigm of pretraining and finetuneing. However, this feature learning process remains poorly understood from a theoretical perspective, with existing analyses largely restricted to two-layer networks. In this work we show that three-layer neural networks have provably richer feature learning capabilities than two-layer networks. We analyze the features learned by a three-layer network trained with layer-wise gradient descent, and present a general purpose theorem which upper bounds the sample complexity and width needed to achieve low test error when the target has specific hierarchical structure. We instantiate our framework in specific statistical learning settings -- single-index models and functions of quadratic 
    
[^3]: 多模光纤水库计算克服了浅层神经网络分类器

    Multi-mode fiber reservoir computing overcomes shallow neural networks classifiers. (arXiv:2210.04745v2 [physics.optics] UPDATED)

    [http://arxiv.org/abs/2210.04745](http://arxiv.org/abs/2210.04745)

    多模光纤利用水库计算范例进行分类，精度高于直接训练原始图像和传统的传输矩阵模型。

    

    在无序光子学领域中，常见的目标是对不透明材料进行表征，以控制光的传递或执行成像。在各种复杂的器件中，多模光纤以其成本效益高、易于操作的特点脱颖而出，使其在几个任务中具有吸引力。在这个背景下，我们利用水库计算范例，将这些光纤转化为随机硬件投影仪，将输入数据集转化为高维斑点图像集。我们的研究目标是证明，通过训练单个逻辑回归层对这些随机数据进行分类，可以提高精度，相比之下直接训练原始图像要更为准确。有趣的是，我们发现使用水库所达到的分类准确性也高于采用传统的传输矩阵模型，后者是描述通过无序器件传递光的广泛接受工具。我们发现，这种改进性能的原因在于水库的动力学具有更高的容量来捕捉复杂的输入输出映射，相对于传输矩阵的线性映射。

    In the field of disordered photonics, a common objective is to characterize optically opaque materials for controlling light delivery or performing imaging. Among various complex devices, multi-mode optical fibers stand out as cost-effective and easy-to-handle tools, making them attractive for several tasks. In this context, we leverage the reservoir computing paradigm to recast these fibers into random hardware projectors, transforming an input dataset into a higher dimensional speckled image set. The goal of our study is to demonstrate that using such randomized data for classification by training a single logistic regression layer improves accuracy compared to training on direct raw images. Interestingly, we found that the classification accuracy achieved using the reservoir is also higher than that obtained with the standard transmission matrix model, a widely accepted tool for describing light transmission through disordered devices. We find that the reason for such improved perfo
    

