# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Leveraging Nested MLMC for Sequential Neural Posterior Estimation with Intractable Likelihoods.](http://arxiv.org/abs/2401.16776) | 本研究提出一种嵌套APT方法来解决顺序神经后验估计中的嵌套期望计算问题，从而实现了收敛性分析。 |
| [^2] | [Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT.](http://arxiv.org/abs/2401.03302) | 本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。 |
| [^3] | [SOFARI: High-Dimensional Manifold-Based Inference.](http://arxiv.org/abs/2309.15032) | 本研究提出了一种基于高维流形的SOFAR推断（SOFARI）方法，通过结合Neyman近正交推断和SVD约束的Stiefel流形结构，实现了对多任务学习中潜在因子矩阵的准确推断。 |

# 详细

[^1]: 利用嵌套MLMC对具有难以处理的似然函数的顺序神经后验估计进行优化

    Leveraging Nested MLMC for Sequential Neural Posterior Estimation with Intractable Likelihoods. (arXiv:2401.16776v1 [stat.CO])

    [http://arxiv.org/abs/2401.16776](http://arxiv.org/abs/2401.16776)

    本研究提出一种嵌套APT方法来解决顺序神经后验估计中的嵌套期望计算问题，从而实现了收敛性分析。

    

    最近提出了顺序神经后验估计（SNPE）技术，用于处理具有难以处理的似然函数的基于模拟的模型。它们致力于通过使用基于神经网络的条件密度估计器自适应地生成的模拟来学习后验。作为一种SNPE技术，Greenberg等人（2019）提出的自动后验变换（APT）方法表现出色，并可应用于高维数据。然而，APT方法包含计算难以处理的归一化常数的对数的期望，即嵌套期望。尽管原子APT通过离散化归一化常数来解决这个问题，但分析学习的收敛性仍然具有挑战性。在本文中，我们提出了一种嵌套APT方法来估计相关的嵌套期望。这有助于建立收敛性分析。由于损失函数及其梯度的嵌套估计是有偏的，我们进行了

    Sequential neural posterior estimation (SNPE) techniques have been recently proposed for dealing with simulation-based models with intractable likelihoods. They are devoted to learning the posterior from adaptively proposed simulations using neural network-based conditional density estimators. As a SNPE technique, the automatic posterior transformation (APT) method proposed by Greenberg et al. (2019) performs notably and scales to high dimensional data. However, the APT method bears the computation of an expectation of the logarithm of an intractable normalizing constant, i.e., a nested expectation. Although atomic APT was proposed to solve this by discretizing the normalizing constant, it remains challenging to analyze the convergence of learning. In this paper, we propose a nested APT method to estimate the involved nested expectation instead. This facilitates establishing the convergence analysis. Since the nested estimators for the loss function and its gradient are biased, we make
    
[^2]: 行动中的现实主义：使用YOLOv8和DeiT从医学图像中诊断脑肿瘤的异常感知

    Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT. (arXiv:2401.03302v1 [eess.IV])

    [http://arxiv.org/abs/2401.03302](http://arxiv.org/abs/2401.03302)

    本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。

    

    在医学科学领域，由于脑肿瘤在患者中的罕见程度，可靠地检测和分类脑肿瘤仍然是一个艰巨的挑战。因此，在异常情况下检测肿瘤的能力对于确保及时干预和改善患者结果至关重要。本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤。来自国家脑映射实验室（NBML）的精选数据集包括81名患者，其中包括30例肿瘤病例和51例正常病例。检测和分类流程被分为两个连续的任务。检测阶段包括全面的数据分析和预处理，以修改图像样本和每个类别的患者数量，以符合真实世界场景中的异常分布（9个正常样本对应1个肿瘤样本）。此外，在测试中除了常见的评估指标外，我们还采用了... [摘要长度已达到上限]

    In the field of medical sciences, reliable detection and classification of brain tumors from images remains a formidable challenge due to the rarity of tumors within the population of patients. Therefore, the ability to detect tumors in anomaly scenarios is paramount for ensuring timely interventions and improved patient outcomes. This study addresses the issue by leveraging deep learning (DL) techniques to detect and classify brain tumors in challenging situations. The curated data set from the National Brain Mapping Lab (NBML) comprises 81 patients, including 30 Tumor cases and 51 Normal cases. The detection and classification pipelines are separated into two consecutive tasks. The detection phase involved comprehensive data analysis and pre-processing to modify the number of image samples and the number of patients of each class to anomaly distribution (9 Normal per 1 Tumor) to comply with real world scenarios. Next, in addition to common evaluation metrics for the testing, we emplo
    
[^3]: SOFARI:基于高维流形的推断

    SOFARI: High-Dimensional Manifold-Based Inference. (arXiv:2309.15032v1 [stat.ME])

    [http://arxiv.org/abs/2309.15032](http://arxiv.org/abs/2309.15032)

    本研究提出了一种基于高维流形的SOFAR推断（SOFARI）方法，通过结合Neyman近正交推断和SVD约束的Stiefel流形结构，实现了对多任务学习中潜在因子矩阵的准确推断。

    

    多任务学习是一种广泛使用的技术，用于从各种任务中提取信息。最近，基于系数矩阵中的稀疏奇异值分解（SVD）的稀疏正交因子回归（SOFAR）框架被引入到可解释的多任务学习中，可以发现不同层次之间有意义的潜在特征-响应关联网络。然而，由于稀疏SVD约束的正交性约束，对潜在因子矩阵进行精确推断仍然具有挑战性。在本文中，我们提出了一种新颖的方法，称为基于高维流形的SOFAR推断（SOFARI），借鉴了Neyman近正交推断，并结合了SVD约束所施加的Stiefel流形结构。通过利用潜在的Stiefel流形结构，SOFARI为潜在左因子向量和奇异值提供了偏差校正的估计量。

    Multi-task learning is a widely used technique for harnessing information from various tasks. Recently, the sparse orthogonal factor regression (SOFAR) framework, based on the sparse singular value decomposition (SVD) within the coefficient matrix, was introduced for interpretable multi-task learning, enabling the discovery of meaningful latent feature-response association networks across different layers. However, conducting precise inference on the latent factor matrices has remained challenging due to orthogonality constraints inherited from the sparse SVD constraint. In this paper, we suggest a novel approach called high-dimensional manifold-based SOFAR inference (SOFARI), drawing on the Neyman near-orthogonality inference while incorporating the Stiefel manifold structure imposed by the SVD constraints. By leveraging the underlying Stiefel manifold structure, SOFARI provides bias-corrected estimators for both latent left factor vectors and singular values, for which we show to enj
    

