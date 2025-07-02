# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fully Differentiable Lagrangian Convolutional Neural Network for Continuity-Consistent Physics-Informed Precipitation Nowcasting](https://arxiv.org/abs/2402.10747) | 提出了一种完全可微的拉格朗日卷积神经网络模型，实现了物理信息与数据驱动学习相结合，在降水预报中表现优秀，为其他拉格朗日机器学习模型提供了新思路。 |
| [^2] | [Selective Prediction for Semantic Segmentation using Post-Hoc Confidence Estimation and Its Performance under Distribution Shift](https://arxiv.org/abs/2402.10665) | 本文研究了在低资源环境中语义分割的选择性预测，提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性 |
| [^3] | [Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity](https://arxiv.org/abs/2310.05175) | 本研究发现LLMs中的激活异常值与网络层稀疏度的非均匀性相关，并提出了Outlier Weighed Layerwise Sparsity（OWL）作为剪枝LLMs到高稀疏度的秘密调味料。 |
| [^4] | [Leveraging Nested MLMC for Sequential Neural Posterior Estimation with Intractable Likelihoods.](http://arxiv.org/abs/2401.16776) | 本研究提出一种嵌套APT方法来解决顺序神经后验估计中的嵌套期望计算问题，从而实现了收敛性分析。 |
| [^5] | [Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT.](http://arxiv.org/abs/2401.03302) | 本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。 |
| [^6] | [Junk DNA Hypothesis: A Task-Centric Angle of LLM Pre-trained Weights through Sparsity.](http://arxiv.org/abs/2310.02277) | 本文研究通过稀疏性分析LLM预训练权重的任务中心角度，挑战了传统对于权重中冗余性的观点，并提出了"垃圾DNA假设"。 |

# 详细

[^1]: 完全可微的拉格朗日卷积神经网络用于连续一致物理信息降水预报

    Fully Differentiable Lagrangian Convolutional Neural Network for Continuity-Consistent Physics-Informed Precipitation Nowcasting

    [https://arxiv.org/abs/2402.10747](https://arxiv.org/abs/2402.10747)

    提出了一种完全可微的拉格朗日卷积神经网络模型，实现了物理信息与数据驱动学习相结合，在降水预报中表现优秀，为其他拉格朗日机器学习模型提供了新思路。

    

    本文提出了一种卷积神经网络模型，用于降水预报，结合了数据驱动学习和基于物理信息的领域知识。我们提出了LUPIN，即用于物理信息的拉格朗日双U-Net的现在预报，借鉴了现有的基于外推的预报方法，并以完全可微且GPU加速的方式实现了数据的拉格朗日坐标系转换，以允许实时端到端训练和推断。根据我们的评估，LUPIN与并超过了所选择基准的性能，为其他拉格朗日机器学习模型敞开了大门。

    arXiv:2402.10747v1 Announce Type: cross  Abstract: This paper presents a convolutional neural network model for precipitation nowcasting that combines data-driven learning with physics-informed domain knowledge. We propose LUPIN, a Lagrangian Double U-Net for Physics-Informed Nowcasting, that draws from existing extrapolation-based nowcasting methods and implements the Lagrangian coordinate system transformation of the data in a fully differentiable and GPU-accelerated manner to allow for real-time end-to-end training and inference. Based on our evaluation, LUPIN matches and exceeds the performance of the chosen benchmark, opening the door for other Lagrangian machine learning models.
    
[^2]: 使用事后置信度估计的选择性预测在语义分割中的性能及其在分布偏移下的表现

    Selective Prediction for Semantic Segmentation using Post-Hoc Confidence Estimation and Its Performance under Distribution Shift

    [https://arxiv.org/abs/2402.10665](https://arxiv.org/abs/2402.10665)

    本文研究了在低资源环境中语义分割的选择性预测，提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性

    

    语义分割在各种计算机视觉应用中扮演着重要角色，然而其有效性常常受到高质量标记数据的缺乏所限。为了解决这一挑战，一个常见策略是利用在不同种群上训练的模型，如公开可用的数据集。然而，这种方法导致了分布偏移问题，在兴趣种群上表现出降低的性能。在模型错误可能带来重大后果的情况下，选择性预测方法提供了一种减轻风险、减少对专家监督依赖的手段。本文研究了在资源匮乏环境下语义分割的选择性预测，着重于应用于在分布偏移下运行的预训练模型的事后置信度估计器。我们提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性。

    arXiv:2402.10665v1 Announce Type: new  Abstract: Semantic segmentation plays a crucial role in various computer vision applications, yet its efficacy is often hindered by the lack of high-quality labeled data. To address this challenge, a common strategy is to leverage models trained on data from different populations, such as publicly available datasets. This approach, however, leads to the distribution shift problem, presenting a reduced performance on the population of interest. In scenarios where model errors can have significant consequences, selective prediction methods offer a means to mitigate risks and reduce reliance on expert supervision. This paper investigates selective prediction for semantic segmentation in low-resource settings, thus focusing on post-hoc confidence estimators applied to pre-trained models operating under distribution shift. We propose a novel image-level confidence measure tailored for semantic segmentation and demonstrate its effectiveness through expe
    
[^3]: Outlier Weighed Layerwise Sparsity (OWL): 为剪枝LLMs达到高稀疏度提供缺失的秘密调味料

    Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity

    [https://arxiv.org/abs/2310.05175](https://arxiv.org/abs/2310.05175)

    本研究发现LLMs中的激活异常值与网络层稀疏度的非均匀性相关，并提出了Outlier Weighed Layerwise Sparsity（OWL）作为剪枝LLMs到高稀疏度的秘密调味料。

    

    大型语言模型（LLMs）以在各个领域展现出的卓越性能而闻名，在实际部署时由于模型庞大而面临挑战。为了解决这一挑战，人们努力将传统的网络剪枝技术应用于LLMs，发现可以在不影响性能的情况下一次性剪掉大量参数。现有的LLM剪枝策略一直坚持以等价稀疏度均匀剪裁所有层的做法，结果表现强劲。然而，这个观察结果与在视觉模型领域观察到的非均匀逐层稀疏的主流趋势相矛盾，后者通常会产生更好的结果。为了了解这种差异背后的原因，我们进行了全面研究，并发现与LLMs中异常值的出现强相关。

    arXiv:2310.05175v2 Announce Type: replace  Abstract: Large Language Models (LLMs), renowned for their remarkable performance across diverse domains, present a challenge when it comes to practical deployment due to their colossal model size. In response to this challenge, efforts have been directed toward the application of traditional network pruning techniques to LLMs, uncovering a massive number of parameters that can be pruned in one-shot without hurting performance. Prevailing LLM pruning strategies have consistently adhered to the practice of uniformly pruning all layers at equivalent sparsity, resulting in robust performance. However, this observation stands in contrast to the prevailing trends observed in the field of vision models, where non-uniform layerwise sparsity typically yields stronger results. To understand the underlying reasons for this disparity, we conduct a comprehensive study and discover a strong correlation with the emergence of activation outliers in LLMs. Ins
    
[^4]: 利用嵌套MLMC对具有难以处理的似然函数的顺序神经后验估计进行优化

    Leveraging Nested MLMC for Sequential Neural Posterior Estimation with Intractable Likelihoods. (arXiv:2401.16776v1 [stat.CO])

    [http://arxiv.org/abs/2401.16776](http://arxiv.org/abs/2401.16776)

    本研究提出一种嵌套APT方法来解决顺序神经后验估计中的嵌套期望计算问题，从而实现了收敛性分析。

    

    最近提出了顺序神经后验估计（SNPE）技术，用于处理具有难以处理的似然函数的基于模拟的模型。它们致力于通过使用基于神经网络的条件密度估计器自适应地生成的模拟来学习后验。作为一种SNPE技术，Greenberg等人（2019）提出的自动后验变换（APT）方法表现出色，并可应用于高维数据。然而，APT方法包含计算难以处理的归一化常数的对数的期望，即嵌套期望。尽管原子APT通过离散化归一化常数来解决这个问题，但分析学习的收敛性仍然具有挑战性。在本文中，我们提出了一种嵌套APT方法来估计相关的嵌套期望。这有助于建立收敛性分析。由于损失函数及其梯度的嵌套估计是有偏的，我们进行了

    Sequential neural posterior estimation (SNPE) techniques have been recently proposed for dealing with simulation-based models with intractable likelihoods. They are devoted to learning the posterior from adaptively proposed simulations using neural network-based conditional density estimators. As a SNPE technique, the automatic posterior transformation (APT) method proposed by Greenberg et al. (2019) performs notably and scales to high dimensional data. However, the APT method bears the computation of an expectation of the logarithm of an intractable normalizing constant, i.e., a nested expectation. Although atomic APT was proposed to solve this by discretizing the normalizing constant, it remains challenging to analyze the convergence of learning. In this paper, we propose a nested APT method to estimate the involved nested expectation instead. This facilitates establishing the convergence analysis. Since the nested estimators for the loss function and its gradient are biased, we make
    
[^5]: 行动中的现实主义：使用YOLOv8和DeiT从医学图像中诊断脑肿瘤的异常感知

    Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT. (arXiv:2401.03302v1 [eess.IV])

    [http://arxiv.org/abs/2401.03302](http://arxiv.org/abs/2401.03302)

    本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。

    

    在医学科学领域，由于脑肿瘤在患者中的罕见程度，可靠地检测和分类脑肿瘤仍然是一个艰巨的挑战。因此，在异常情况下检测肿瘤的能力对于确保及时干预和改善患者结果至关重要。本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤。来自国家脑映射实验室（NBML）的精选数据集包括81名患者，其中包括30例肿瘤病例和51例正常病例。检测和分类流程被分为两个连续的任务。检测阶段包括全面的数据分析和预处理，以修改图像样本和每个类别的患者数量，以符合真实世界场景中的异常分布（9个正常样本对应1个肿瘤样本）。此外，在测试中除了常见的评估指标外，我们还采用了... [摘要长度已达到上限]

    In the field of medical sciences, reliable detection and classification of brain tumors from images remains a formidable challenge due to the rarity of tumors within the population of patients. Therefore, the ability to detect tumors in anomaly scenarios is paramount for ensuring timely interventions and improved patient outcomes. This study addresses the issue by leveraging deep learning (DL) techniques to detect and classify brain tumors in challenging situations. The curated data set from the National Brain Mapping Lab (NBML) comprises 81 patients, including 30 Tumor cases and 51 Normal cases. The detection and classification pipelines are separated into two consecutive tasks. The detection phase involved comprehensive data analysis and pre-processing to modify the number of image samples and the number of patients of each class to anomaly distribution (9 Normal per 1 Tumor) to comply with real world scenarios. Next, in addition to common evaluation metrics for the testing, we emplo
    
[^6]: "垃圾DNA假设：通过稀疏性对LLM预训练权重进行任务中心角度分析"

    Junk DNA Hypothesis: A Task-Centric Angle of LLM Pre-trained Weights through Sparsity. (arXiv:2310.02277v1 [cs.LG])

    [http://arxiv.org/abs/2310.02277](http://arxiv.org/abs/2310.02277)

    本文研究通过稀疏性分析LLM预训练权重的任务中心角度，挑战了传统对于权重中冗余性的观点，并提出了"垃圾DNA假设"。

    

    传统对"垃圾DNA"的概念长期以来与人类基因组中的非编码片段相关联，占其组成的大约98%。然而，最近的研究揭示了一些这些看似无功能的DNA序列在细胞过程中起到的关键作用。有趣的是，深度神经网络中的权重与人类基因中观察到的冗余性有着显著的相似性。人们认为，庞大模型中的权重包含了过多的冗余，可以在不影响性能的情况下去除。本文通过提出一个令人信服的反论来挑战这个传统观点。我们使用稀疏性作为一种工具，来独立而准确地量化预训练大语言模型(LLM)中低幅度权重的细微重要性，从下游任务中心的角度理解它们包含的知识。我们提出了支持我们深入研究的"垃圾DNA假设"。

    The traditional notion of "Junk DNA" has long been linked to non-coding segments within the human genome, constituting roughly 98% of its composition. However, recent research has unveiled the critical roles some of these seemingly non-functional DNA sequences play in cellular processes. Intriguingly, the weights within deep neural networks exhibit a remarkable similarity to the redundancy observed in human genes. It was believed that weights in gigantic models contained excessive redundancy, and could be removed without compromising performance. This paper challenges this conventional wisdom by presenting a compelling counter-argument. We employ sparsity as a tool to isolate and quantify the nuanced significance of low-magnitude weights in pre-trained large language models (LLMs). Our study demonstrates a strong correlation between these weight magnitudes and the knowledge they encapsulate, from a downstream task-centric angle. we raise the "Junk DNA Hypothesis" backed by our in-depth
    

