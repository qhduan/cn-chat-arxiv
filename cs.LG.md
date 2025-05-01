# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inferring the Langevin Equation with Uncertainty via Bayesian Neural Networks](https://rss.arxiv.org/abs/2402.01338) | 本研究提出了一个使用贝叶斯神经网络推断带有不确定性的朗之万方程的综合框架，可以应用于非线性和高维系统，通过提供预测的分布从而评估预测的不确定性。 |
| [^2] | [Neural Redshift: Random Networks are not Random Functions](https://arxiv.org/abs/2403.02241) | 本论文研究了未经训练的随机权重网络，发现即使简单的MLPs也具有强烈的归纳偏见，不同于传统观点的是，NNs并不具有固有的“简单偏见”，而是依赖于组件的作用。 |
| [^3] | [Shortcutting Cross-Validation: Efficiently Deriving Column-Wise Centered and Scaled Training Set $\mathbf{X}^\mathbf{T}\mathbf{X}$ and $\mathbf{X}^\mathbf{T}\mathbf{Y}$ Without Full Recomputation of Matrix Products or Statistical Moments.](http://arxiv.org/abs/2401.13185) | 本文提出了三种高效计算训练集$\mathbf{X}^\mathbf{T}\mathbf{X}$和$\mathbf{X}^\mathbf{T}\mathbf{Y}$的算法，相比于以前的工作，这些算法能够显著加速交叉验证，而无需重新计算矩阵乘积或统计量。 |
| [^4] | [EyePreserve: Identity-Preserving Iris Synthesis.](http://arxiv.org/abs/2312.12028) | 本论文提出了一种完全数据驱动的、保持身份的、瞳孔尺寸变化的虹膜图像合成方法，能够合成不同瞳孔尺寸的虹膜图像，代表不存在的身份，并能够在保持身份的同时进行非线性纹理变形。 |
| [^5] | [Multicollinearity Resolution Based on Machine Learning: A Case Study of Carbon Emissions in Sichuan Province.](http://arxiv.org/abs/2309.01115) | 本研究使用机器学习方法解决了多重共线性问题，针对四川省的碳排放情况进行了案例研究。研究结果确定了行业分组，评估了排放驱动因素，并提出了科学的减排策略，以改善情况。 |
| [^6] | [Semantic Positive Pairs for Enhancing Contrastive Instance Discrimination.](http://arxiv.org/abs/2306.16122) | 提出了一种名为语义正向对集合（SPPS）的方法，可以在表示学习过程中识别具有相似语义内容的图像，并将它们视为正向实例，从而减少丢弃重要特征的风险。在多个实验数据集上的实验证明了该方法的有效性。 |

# 详细

[^1]: 通过贝叶斯神经网络推断带有不确定性的朗之万方程

    Inferring the Langevin Equation with Uncertainty via Bayesian Neural Networks

    [https://rss.arxiv.org/abs/2402.01338](https://rss.arxiv.org/abs/2402.01338)

    本研究提出了一个使用贝叶斯神经网络推断带有不确定性的朗之万方程的综合框架，可以应用于非线性和高维系统，通过提供预测的分布从而评估预测的不确定性。

    

    在各个领域普遍存在的随机系统表现出从分子动力学到气候现象的过程中的波动。朗之万方程作为一种常见的数学模型，被用于研究这种系统，可以预测它们的时间演化以及分析热力学量，包括吸收热量、对系统做的功以及熵的产生。然而，从观测轨迹中推断朗之万方程仍然具有挑战性，尤其是对于非线性和高维系统。在本研究中，我们提出了一个全面的框架，利用贝叶斯神经网络来推断过阻尼和欠阻尼情况下的朗之万方程。我们的框架首先分别提供漂移力和扩散矩阵，然后将它们组合起来构建朗之万方程。通过提供预测的分布而不仅仅是一个单一的值，我们的方法允许我们评估预测的不确定性，这可以防止潜在的…

    Pervasive across diverse domains, stochastic systems exhibit fluctuations in processes ranging from molecular dynamics to climate phenomena. The Langevin equation has served as a common mathematical model for studying such systems, enabling predictions of their temporal evolution and analyses of thermodynamic quantities, including absorbed heat, work done on the system, and entropy production. However, inferring the Langevin equation from observed trajectories remains challenging, particularly for nonlinear and high-dimensional systems. In this study, we present a comprehensive framework that employs Bayesian neural networks for inferring Langevin equations in both overdamped and underdamped regimes. Our framework first provides the drift force and diffusion matrix separately and then combines them to construct the Langevin equation. By providing a distribution of predictions instead of a single value, our approach allows us to assess prediction uncertainties, which can prevent potenti
    
[^2]: 神经红移：随机网络并非随机函数

    Neural Redshift: Random Networks are not Random Functions

    [https://arxiv.org/abs/2403.02241](https://arxiv.org/abs/2403.02241)

    本论文研究了未经训练的随机权重网络，发现即使简单的MLPs也具有强烈的归纳偏见，不同于传统观点的是，NNs并不具有固有的“简单偏见”，而是依赖于组件的作用。

    

    我们对神经网络（NNs）的泛化能力的理解仍不完整。目前的解释基于梯度下降（GD）的隐含偏见，但无法解释梯度自由方法中模型的能力，也无法解释最近观察到的未经训练网络的简单偏见。本文寻找NNs中的其他泛化源。为了独立于GD理解体系结构提供的归纳偏见，我们研究未经训练的随机权重网络。即使是简单的MLPs也表现出强烈的归纳偏见：在权重空间中进行均匀抽样会产生一个非常偏向于复杂性的函数分布。但与常规智慧不同，NNs并不具有固有的“简单偏见”。这一特性取决于组件，如ReLU、残差连接和层归一化。可利用替代体系结构构建偏向于任何复杂性水平的偏见。Transformers也具有这一特性。

    arXiv:2403.02241v1 Announce Type: cross  Abstract: Our understanding of the generalization capabilities of neural networks (NNs) is still incomplete. Prevailing explanations are based on implicit biases of gradient descent (GD) but they cannot account for the capabilities of models from gradient-free methods nor the simplicity bias recently observed in untrained networks. This paper seeks other sources of generalization in NNs.   Findings. To understand the inductive biases provided by architectures independently from GD, we examine untrained, random-weight networks. Even simple MLPs show strong inductive biases: uniform sampling in weight space yields a very biased distribution of functions in terms of complexity. But unlike common wisdom, NNs do not have an inherent "simplicity bias". This property depends on components such as ReLUs, residual connections, and layer normalizations. Alternative architectures can be built with a bias for any level of complexity. Transformers also inher
    
[^3]: 简化交叉验证：高效地计算不需要全量重新计算矩阵乘积或统计量的列向中心化和标准化训练集$\mathbf{X}^\mathbf{T}\mathbf{X}$和$\mathbf{X}^\mathbf{T}\mathbf{Y}$

    Shortcutting Cross-Validation: Efficiently Deriving Column-Wise Centered and Scaled Training Set $\mathbf{X}^\mathbf{T}\mathbf{X}$ and $\mathbf{X}^\mathbf{T}\mathbf{Y}$ Without Full Recomputation of Matrix Products or Statistical Moments. (arXiv:2401.13185v1 [cs.LG])

    [http://arxiv.org/abs/2401.13185](http://arxiv.org/abs/2401.13185)

    本文提出了三种高效计算训练集$\mathbf{X}^\mathbf{T}\mathbf{X}$和$\mathbf{X}^\mathbf{T}\mathbf{Y}$的算法，相比于以前的工作，这些算法能够显著加速交叉验证，而无需重新计算矩阵乘积或统计量。

    

    交叉验证是一种广泛使用的评估预测模型在未知数据上表现的技术。许多预测模型，如基于核的偏最小二乘（PLS）模型，需要仅使用输入矩阵$\mathbf{X}$和输出矩阵$\mathbf{Y}$中的训练集样本来计算$\mathbf{X}^{\mathbf{T}}\mathbf{X}$和$\mathbf{X}^{\mathbf{T}}\mathbf{Y}$。在这项工作中，我们提出了三种高效计算这些矩阵的算法。第一种算法不需要列向预处理。第二种算法允许以训练集均值为中心化点进行列向中心化。第三种算法允许以训练集均值和标准差为中心化点和标准化点进行列向中心化和标准化。通过证明正确性和优越的计算复杂度，它们相比于直接交叉验证和以前的快速交叉验证工作，提供了显著的交叉验证加速，而无需数据泄露。它们适合并行计算。

    Cross-validation is a widely used technique for assessing the performance of predictive models on unseen data. Many predictive models, such as Kernel-Based Partial Least-Squares (PLS) models, require the computation of $\mathbf{X}^{\mathbf{T}}\mathbf{X}$ and $\mathbf{X}^{\mathbf{T}}\mathbf{Y}$ using only training set samples from the input and output matrices, $\mathbf{X}$ and $\mathbf{Y}$, respectively. In this work, we present three algorithms that efficiently compute these matrices. The first one allows no column-wise preprocessing. The second one allows column-wise centering around the training set means. The third one allows column-wise centering and column-wise scaling around the training set means and standard deviations. Demonstrating correctness and superior computational complexity, they offer significant cross-validation speedup compared with straight-forward cross-validation and previous work on fast cross-validation - all without data leakage. Their suitability for paralle
    
[^4]: EyePreserve: 保持身份的虹膜合成

    EyePreserve: Identity-Preserving Iris Synthesis. (arXiv:2312.12028v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.12028](http://arxiv.org/abs/2312.12028)

    本论文提出了一种完全数据驱动的、保持身份的、瞳孔尺寸变化的虹膜图像合成方法，能够合成不同瞳孔尺寸的虹膜图像，代表不存在的身份，并能够在保持身份的同时进行非线性纹理变形。

    

    在广泛的瞳孔尺寸范围内保持身份的同身份生物特征虹膜图像的合成是复杂的，因为它涉及到虹膜肌肉收缩机制，需要将虹膜非线性纹理变形模型嵌入到合成流程中。本论文提出了一种完全数据驱动的、保持身份的、瞳孔尺寸变化的虹膜图像合成方法。这种方法能够合成具有不同瞳孔尺寸的虹膜图像，代表不存在的身份，并能够在给定目标虹膜图像的分割掩膜下非线性地变形现有主体的虹膜图像纹理。虹膜识别实验表明，所提出的变形模型不仅在改变瞳孔尺寸时保持身份，而且在瞳孔尺寸有显著差异的同身份虹膜样本之间提供更好的相似度，与最先进的线性方法相比。

    Synthesis of same-identity biometric iris images, both for existing and non-existing identities while preserving the identity across a wide range of pupil sizes, is complex due to intricate iris muscle constriction mechanism, requiring a precise model of iris non-linear texture deformations to be embedded into the synthesis pipeline. This paper presents the first method of fully data-driven, identity-preserving, pupil size-varying s ynthesis of iris images. This approach is capable of synthesizing images of irises with different pupil sizes representing non-existing identities as well as non-linearly deforming the texture of iris images of existing subjects given the segmentation mask of the target iris image. Iris recognition experiments suggest that the proposed deformation model not only preserves the identity when changing the pupil size but offers better similarity between same-identity iris samples with significant differences in pupil size, compared to state-of-the-art linear an
    
[^5]: 基于机器学习的多重共线性解决方案：四川省碳排放案例研究

    Multicollinearity Resolution Based on Machine Learning: A Case Study of Carbon Emissions in Sichuan Province. (arXiv:2309.01115v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.01115](http://arxiv.org/abs/2309.01115)

    本研究使用机器学习方法解决了多重共线性问题，针对四川省的碳排放情况进行了案例研究。研究结果确定了行业分组，评估了排放驱动因素，并提出了科学的减排策略，以改善情况。

    

    本研究使用矩阵归一化对四川省46个关键产业2000-2019年的能源消耗数据进行预处理。DBSCAN聚类识别了16个特征类别以客观地分组行业。接下来，采用罚函数回归模型，以应对过拟合控制、高维数据处理和特征选择等复杂能源数据处理的优势。结果表明，煤炭周围的第二个聚类因生产需求而产生的排放量最高。以汽油和焦炭为中心的聚类的排放量也很显著。基于此，减排建议包括清洁煤技术、交通管理、钢铁中的煤电替代和行业标准化。该研究引入了无监督学习的方法来客观选择因素，并旨在探索新的减排途径。总而言之，本研究确定了行业分组，评估了排放驱动因素，并提出了科学的减排策略以进一步改善情况。

    This study preprocessed 2000-2019 energy consumption data for 46 key Sichuan industries using matrix normalization. DBSCAN clustering identified 16 feature classes to objectively group industries. Penalized regression models were then applied for their advantages in overfitting control, high-dimensional data processing, and feature selection - well-suited for the complex energy data. Results showed the second cluster around coal had highest emissions due to production needs. Emissions from gasoline-focused and coke-focused clusters were also significant. Based on this, emission reduction suggestions included clean coal technologies, transportation management, coal-electricity replacement in steel, and industry standardization. The research introduced unsupervised learning to objectively select factors and aimed to explore new emission reduction avenues. In summary, the study identified industry groupings, assessed emissions drivers, and proposed scientific reduction strategies to bette
    
[^6]: 增强对比实例区分的语义正向对

    Semantic Positive Pairs for Enhancing Contrastive Instance Discrimination. (arXiv:2306.16122v1 [cs.CV])

    [http://arxiv.org/abs/2306.16122](http://arxiv.org/abs/2306.16122)

    提出了一种名为语义正向对集合（SPPS）的方法，可以在表示学习过程中识别具有相似语义内容的图像，并将它们视为正向实例，从而减少丢弃重要特征的风险。在多个实验数据集上的实验证明了该方法的有效性。

    

    基于实例区分的自监督学习算法有效地防止表示坍缩，并在表示学习中产生了有希望的结果。然而，在嵌入空间中吸引正向对（即相同实例的两个视图）并排斥所有其他实例（即负向对），无论它们的类别，可能导致丢弃重要的特征。为了解决这个问题，我们提出了一种方法来识别具有相似语义内容的图像，并将它们视为正向实例，命名为语义正向对集合（SPPS），从而在表示学习中减少了丢弃重要特征的风险。我们的方法可以与任何对比实例区分框架（如SimCLR或MOCO）一起使用。我们在三个数据集（ImageNet、STL-10和CIFAR-10）上进行实验证明了我们的方法的有效性。实验结果表明，我们的方法始终优于基线方法vanilla。

    Self-supervised learning algorithms based on instance discrimination effectively prevent representation collapse and produce promising results in representation learning. However, the process of attracting positive pairs (i.e., two views of the same instance) in the embedding space and repelling all other instances (i.e., negative pairs) irrespective of their categories could result in discarding important features. To address this issue, we propose an approach to identifying those images with similar semantic content and treating them as positive instances, named semantic positive pairs set (SPPS), thereby reducing the risk of discarding important features during representation learning. Our approach could work with any contrastive instance discrimination framework such as SimCLR or MOCO. We conduct experiments on three datasets: ImageNet, STL-10 and CIFAR-10 to evaluate our approach. The experimental results show that our approach consistently outperforms the baseline method vanilla 
    

