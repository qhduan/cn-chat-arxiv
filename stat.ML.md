# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inference with Mondrian Random Forests.](http://arxiv.org/abs/2310.09702) | 本文在回归设置下给出了Mondrian随机森林的估计中心极限定理和去偏过程，使其能够进行统计推断和实现最小极大估计速率。 |
| [^2] | [Accelerating Generalized Random Forests with Fixed-Point Trees.](http://arxiv.org/abs/2306.11908) | 本文提出一种新的树生长规则，使广义随机森林在无梯度优化的情况下大大节省了时间。 |
| [^3] | [Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension.](http://arxiv.org/abs/2305.15203) | 本文通过研究神经网络的隐性偏差，着眼于其中涉及的傅里叶频率与图像分类和对抗性攻击之间的关系。研究提出了一种新方法，可以发现这些频率之间的非线性相关性。 |
| [^4] | [Self-Supervised Siamese Autoencoders.](http://arxiv.org/abs/2304.02549) | 本论文提出了一种新的自监督方法，名为SidAE，它结合了孪生架构和去噪自编码器的优点，可以更好地提取输入数据的特征，以在多个下游任务中获得更好的性能。 |
| [^5] | [Penalising the biases in norm regularisation enforces sparsity.](http://arxiv.org/abs/2303.01353) | 本研究表明，控制神经网络参数的范数可以获得良好的泛化性能。对神经网络中偏差项的范数进行惩罚可以实现稀疏估计量。 |

# 详细

[^1]: 带有Mondrian随机森林的推理

    Inference with Mondrian Random Forests. (arXiv:2310.09702v1 [math.ST])

    [http://arxiv.org/abs/2310.09702](http://arxiv.org/abs/2310.09702)

    本文在回归设置下给出了Mondrian随机森林的估计中心极限定理和去偏过程，使其能够进行统计推断和实现最小极大估计速率。

    

    随机森林是一种常用的分类和回归方法，在最近几年中提出了许多不同的变体。一个有趣的例子是Mondrian随机森林，其中底层树是根据Mondrian过程构建的。在本文中，我们给出了Mondrian随机森林在回归设置下的估计的中心极限定理。当与偏差表征和一致方差估计器相结合时，这允许进行渐近有效的统计推断，如构建置信区间，对未知的回归函数进行推断。我们还提供了一种去偏过程，用于Mondrian随机森林，使其能够在适当的参数调整下实现$\beta$-H\"older回归函数的最小极大估计速率，对于所有的$\beta$和任意维度。

    Random forests are popular methods for classification and regression, and many different variants have been proposed in recent years. One interesting example is the Mondrian random forest, in which the underlying trees are constructed according to a Mondrian process. In this paper we give a central limit theorem for the estimates made by a Mondrian random forest in the regression setting. When combined with a bias characterization and a consistent variance estimator, this allows one to perform asymptotically valid statistical inference, such as constructing confidence intervals, on the unknown regression function. We also provide a debiasing procedure for Mondrian random forests which allows them to achieve minimax-optimal estimation rates with $\beta$-H\"older regression functions, for all $\beta$ and in arbitrary dimension, assuming appropriate parameter tuning.
    
[^2]: 基于定点树的广义随机森林加速

    Accelerating Generalized Random Forests with Fixed-Point Trees. (arXiv:2306.11908v1 [stat.ML])

    [http://arxiv.org/abs/2306.11908](http://arxiv.org/abs/2306.11908)

    本文提出一种新的树生长规则，使广义随机森林在无梯度优化的情况下大大节省了时间。

    

    广义随机森林建立在传统随机森林的基础上，通过将其作为自适应核加权算法来构建估算器，并通过基于梯度的树生长过程来实现。我们提出了一种新的树生长规则，基于定点迭代近似表示梯度近似，实现了无梯度优化，并为此开发了渐近理论。这有效地节省了时间，尤其是在目标量的维度适中时。

    Generalized random forests arXiv:1610.01271 build upon the well-established success of conventional forests (Breiman, 2001) to offer a flexible and powerful non-parametric method for estimating local solutions of heterogeneous estimating equations. Estimators are constructed by leveraging random forests as an adaptive kernel weighting algorithm and implemented through a gradient-based tree-growing procedure. By expressing this gradient-based approximation as being induced from a single Newton-Raphson root-finding iteration, and drawing upon the connection between estimating equations and fixed-point problems arXiv:2110.11074, we propose a new tree-growing rule for generalized random forests induced from a fixed-point iteration type of approximation, enabling gradient-free optimization, and yielding substantial time savings for tasks involving even modest dimensionality of the target quantity (e.g. multiple/multi-level treatment effects). We develop an asymptotic theory for estimators o
    
[^3]: 通过内在维度将隐性偏见和对抗性攻击相关联

    Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension. (arXiv:2305.15203v1 [cs.LG])

    [http://arxiv.org/abs/2305.15203](http://arxiv.org/abs/2305.15203)

    本文通过研究神经网络的隐性偏差，着眼于其中涉及的傅里叶频率与图像分类和对抗性攻击之间的关系。研究提出了一种新方法，可以发现这些频率之间的非线性相关性。

    

    尽管神经网络在分类方面表现出色，但众所周知它们易受对抗性攻击的影响。这些攻击是针对模型的输入数据进行的小干扰，旨在欺骗模型。自然而然的问题是，模型的结构、设置或属性与攻击的性质之间可能存在潜在联系。在本文中，我们旨在通过关注神经网络的隐性偏差来解决这个问题，这指的是其固有倾向于支持特定模式或结果。具体而言，我们研究了隐性偏差的一个方面，其中包括进行准确图像分类所需的基本傅里叶频率。我们进行测试以评估这些频率与成功攻击所需的频率之间的统计关系。为了深入探讨这种关系，我们提出了一种新的方法，可以揭示坐标集之间的非线性相关性，在我们的情况下，这些坐标集就是前述的傅里叶频率。

    Despite their impressive performance in classification, neural networks are known to be vulnerable to adversarial attacks. These attacks are small perturbations of the input data designed to fool the model. Naturally, a question arises regarding the potential connection between the architecture, settings, or properties of the model and the nature of the attack. In this work, we aim to shed light on this problem by focusing on the implicit bias of the neural network, which refers to its inherent inclination to favor specific patterns or outcomes. Specifically, we investigate one aspect of the implicit bias, which involves the essential Fourier frequencies required for accurate image classification. We conduct tests to assess the statistical relationship between these frequencies and those necessary for a successful attack. To delve into this relationship, we propose a new method that can uncover non-linear correlations between sets of coordinates, which, in our case, are the aforementio
    
[^4]: 自监督的孪生自编码器

    Self-Supervised Siamese Autoencoders. (arXiv:2304.02549v1 [cs.LG])

    [http://arxiv.org/abs/2304.02549](http://arxiv.org/abs/2304.02549)

    本论文提出了一种新的自监督方法，名为SidAE，它结合了孪生架构和去噪自编码器的优点，可以更好地提取输入数据的特征，以在多个下游任务中获得更好的性能。

    

    完全监督的模型通常需要大量的标记训练数据，这往往是昂贵且难以获得的。相反，自监督表示学习减少了实现相同或更高下游性能所需的标记数据量。目标是在自监督任务上预先训练深度神经网络，以便网络能够从原始输入数据中提取有意义的特征。然后，将这些特征用作下游任务（如图像分类）中的输入。在先前的研究中，自编码器和孪生网络（如SimSiam）已成功应用于这些任务中。然而，仍然存在一些挑战，例如将特征的特性（例如，细节级别）与给定的任务和数据集匹配。在本文中，我们提出了一种结合了孪生架构和去噪自编码器优势的新自监督方法。我们展示了我们的模型，名为SidAE（孪生去噪自编码器），在多个下游任务上胜过了两个自监督最新基准。

    Fully supervised models often require large amounts of labeled training data, which tends to be costly and hard to acquire. In contrast, self-supervised representation learning reduces the amount of labeled data needed for achieving the same or even higher downstream performance. The goal is to pre-train deep neural networks on a self-supervised task such that afterwards the networks are able to extract meaningful features from raw input data. These features are then used as inputs in downstream tasks, such as image classification. Previously, autoencoders and Siamese networks such as SimSiam have been successfully employed in those tasks. Yet, challenges remain, such as matching characteristics of the features (e.g., level of detail) to the given task and data set. In this paper, we present a new self-supervised method that combines the benefits of Siamese architectures and denoising autoencoders. We show that our model, called SidAE (Siamese denoising autoencoder), outperforms two se
    
[^5]: 对正则化中的偏差进行惩罚将使稀疏化

    Penalising the biases in norm regularisation enforces sparsity. (arXiv:2303.01353v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2303.01353](http://arxiv.org/abs/2303.01353)

    本研究表明，控制神经网络参数的范数可以获得良好的泛化性能。对神经网络中偏差项的范数进行惩罚可以实现稀疏估计量。

    

    当训练神经网络时，通过控制参数的范数往往可以获得良好的泛化性能。然而，正则化参数的范数和所得估计量之间的关系在理论上尚未完全理解。本文针对具有单一隐藏层和一维数据的神经网络，展示了表示函数所需的参数范数由其二阶导数的总变差加权得到，其中所加权的因子为$\sqrt{1+x^2}$。值得注意的是，当不对偏差项的范数进行正则化时，这个加权因子会消失。这个额外的加权因子的存在非常重要，因为它被证明可以强制实现最小范数内插器的唯一性和稀疏性（在拐点数量上）。相反，省略偏差的范数则会导致非稀疏解。因此，在正则化中对偏差项进行惩罚，无论是显式还是隐式地，都会导致稀疏估计量。

    Controlling the parameters' norm often yields good generalisation when training neural networks. Beyond simple intuitions, the relation between regularising parameters' norm and obtained estimators remains theoretically misunderstood. For one hidden ReLU layer networks with unidimensional data, this work shows the parameters' norm required to represent a function is given by the total variation of its second derivative, weighted by a $\sqrt{1+x^2}$ factor. Notably, this weighting factor disappears when the norm of bias terms is not regularised. The presence of this additional weighting factor is of utmost significance as it is shown to enforce the uniqueness and sparsity (in the number of kinks) of the minimal norm interpolator. Conversely, omitting the bias' norm allows for non-sparse solutions. Penalising the bias terms in the regularisation, either explicitly or implicitly, thus leads to sparse estimators.
    

