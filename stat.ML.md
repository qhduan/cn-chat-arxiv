# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [InVA: Integrative Variational Autoencoder for Harmonization of Multi-modal Neuroimaging Data](https://arxiv.org/abs/2402.02734) | InVA是一种综合变分自编码器方法，利用多模态神经影像数据中不同来源的多个图像来进行预测推理，相较于传统的VAE方法具有更好的效果。 |
| [^2] | [Sufficient Invariant Learning for Distribution Shift.](http://arxiv.org/abs/2210.13533) | 本文研究了分布转移情况下的充分不变学习，观察到之前的工作只学习了部分不变特征，我们提出了学习充分不变特征的重要性，并指出在分布转移时，从训练集中学习的部分不变特征可能不适用于测试集，限制了性能提升。 |

# 详细

[^1]: InVA: 综合变分自编码器用于多模态神经影像数据的协调

    InVA: Integrative Variational Autoencoder for Harmonization of Multi-modal Neuroimaging Data

    [https://arxiv.org/abs/2402.02734](https://arxiv.org/abs/2402.02734)

    InVA是一种综合变分自编码器方法，利用多模态神经影像数据中不同来源的多个图像来进行预测推理，相较于传统的VAE方法具有更好的效果。

    

    在探索多个来自不同成像模式的图像之间的非线性关联方面具有重要意义。尽管有越来越多的文献研究基于多个图像来推断图像的预测推理，但现有方法在有效借用多个成像模式之间的信息来预测图像方面存在局限。本文建立在变分自编码器（VAEs）的文献基础上，提出了一种新颖的方法，称为综合变分自编码器（InVA）方法，它从不同来源获得的多个图像中借用信息来绘制图像的预测推理。所提出的方法捕捉了结果图像与输入图像之间的复杂非线性关联，并允许快速计算。数值结果表明，InVA相对于通常不允许借用输入图像之间信息的VAE具有明显的优势。

    There is a significant interest in exploring non-linear associations among multiple images derived from diverse imaging modalities. While there is a growing literature on image-on-image regression to delineate predictive inference of an image based on multiple images, existing approaches have limitations in efficiently borrowing information between multiple imaging modalities in the prediction of an image. Building on the literature of Variational Auto Encoders (VAEs), this article proposes a novel approach, referred to as Integrative Variational Autoencoder (\texttt{InVA}) method, which borrows information from multiple images obtained from different sources to draw predictive inference of an image. The proposed approach captures complex non-linear association between the outcome image and input images, while allowing rapid computation. Numerical results demonstrate substantial advantages of \texttt{InVA} over VAEs, which typically do not allow borrowing information between input imag
    
[^2]: 分布转移的充分不变学习

    Sufficient Invariant Learning for Distribution Shift. (arXiv:2210.13533v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.13533](http://arxiv.org/abs/2210.13533)

    本文研究了分布转移情况下的充分不变学习，观察到之前的工作只学习了部分不变特征，我们提出了学习充分不变特征的重要性，并指出在分布转移时，从训练集中学习的部分不变特征可能不适用于测试集，限制了性能提升。

    

    机器学习算法在各种应用中展现出了卓越的性能。然而，在训练集和测试集的分布不同的情况下，保证性能仍然具有挑战性。为了改善分布转移情况下的性能，已经提出了一些方法，通过学习跨组或领域的不变特征来提高性能。然而，我们观察到之前的工作只部分地学习了不变特征。虽然先前的工作侧重于有限的不变特征，但我们首次提出了充分不变特征的重要性。由于只有训练集是经验性的，从训练集中学习得到的部分不变特征可能不存在于分布转移时的测试集中。因此，分布转移情况下的性能提高可能受到限制。本文认为从训练集中学习充分的不变特征对于分布转移情况至关重要。

    Machine learning algorithms have shown remarkable performance in diverse applications. However, it is still challenging to guarantee performance in distribution shifts when distributions of training and test datasets are different. There have been several approaches to improve the performance in distribution shift cases by learning invariant features across groups or domains. However, we observe that the previous works only learn invariant features partially. While the prior works focus on the limited invariant features, we first raise the importance of the sufficient invariant features. Since only training sets are given empirically, the learned partial invariant features from training sets might not be present in the test sets under distribution shift. Therefore, the performance improvement on distribution shifts might be limited. In this paper, we argue that learning sufficient invariant features from the training set is crucial for the distribution shift case. Concretely, we newly 
    

