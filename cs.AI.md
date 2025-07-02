# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quantifying analogy of concepts via ologs and wiring diagrams](https://rss.arxiv.org/abs/2402.01020) | 本文通过ologs和接线图的概念，建立了一个量化概念类比的框架，使得自主系统能够形成抽象概念，并通过图论和范畴论的技术进行比较和操作，同时使用接线图操作定义了度量和图编辑距离。 |
| [^2] | [Fully Differentiable Lagrangian Convolutional Neural Network for Continuity-Consistent Physics-Informed Precipitation Nowcasting](https://arxiv.org/abs/2402.10747) | 提出了一种完全可微的拉格朗日卷积神经网络模型，实现了物理信息与数据驱动学习相结合，在降水预报中表现优秀，为其他拉格朗日机器学习模型提供了新思路。 |
| [^3] | [Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT.](http://arxiv.org/abs/2401.03302) | 本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。 |
| [^4] | [Junk DNA Hypothesis: A Task-Centric Angle of LLM Pre-trained Weights through Sparsity.](http://arxiv.org/abs/2310.02277) | 本文研究通过稀疏性分析LLM预训练权重的任务中心角度，挑战了传统对于权重中冗余性的观点，并提出了"垃圾DNA假设"。 |
| [^5] | [Towards Enhanced Controllability of Diffusion Models.](http://arxiv.org/abs/2302.14368) | 本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。 |

# 详细

[^1]: 通过ologs和接线图量化概念的类比

    Quantifying analogy of concepts via ologs and wiring diagrams

    [https://rss.arxiv.org/abs/2402.01020](https://rss.arxiv.org/abs/2402.01020)

    本文通过ologs和接线图的概念，建立了一个量化概念类比的框架，使得自主系统能够形成抽象概念，并通过图论和范畴论的技术进行比较和操作，同时使用接线图操作定义了度量和图编辑距离。

    

    我们在Spivak和Kent创建的本体日志(ologs)理论的基础上，定义了一种称为接线图的概念。在本文中，接线图是一个有限的有向标记图。标记对应于olog中的类型；它们也可以被解释为自主系统中传感器的读数。因此，接线图可以用作自主系统形成抽象概念的框架。我们展示了基于骨架接线图的图形形成一个范畴。这使得骨架接线图可以使用图论和范畴论的技术进行比较和操作。我们还通过使用仅适用于接线图的操作将传统的图编辑距离定义扩展到接线图的情况，从而得到了所有骨架接线图集合上的度量。最后，我们给出了一个计算由接线图表示的两个概念之间距离的扩展示例，并解释了如何将我们的框架应用于任何应用领域。

    We build on the theory of ontology logs (ologs) created by Spivak and Kent, and define a notion of wiring diagrams. In this article, a wiring diagram is a finite directed labelled graph. The labels correspond to types in an olog; they can also be interpreted as readings of sensors in an autonomous system. As such, wiring diagrams can be used as a framework for an autonomous system to form abstract concepts. We show that the graphs underlying skeleton wiring diagrams form a category. This allows skeleton wiring diagrams to be compared and manipulated using techniques from both graph theory and category theory. We also extend the usual definition of graph edit distance to the case of wiring diagrams by using operations only available to wiring diagrams, leading to a metric on the set of all skeleton wiring diagrams. In the end, we give an extended example on calculating the distance between two concepts represented by wiring diagrams, and explain how to apply our framework to any applica
    
[^2]: 完全可微的拉格朗日卷积神经网络用于连续一致物理信息降水预报

    Fully Differentiable Lagrangian Convolutional Neural Network for Continuity-Consistent Physics-Informed Precipitation Nowcasting

    [https://arxiv.org/abs/2402.10747](https://arxiv.org/abs/2402.10747)

    提出了一种完全可微的拉格朗日卷积神经网络模型，实现了物理信息与数据驱动学习相结合，在降水预报中表现优秀，为其他拉格朗日机器学习模型提供了新思路。

    

    本文提出了一种卷积神经网络模型，用于降水预报，结合了数据驱动学习和基于物理信息的领域知识。我们提出了LUPIN，即用于物理信息的拉格朗日双U-Net的现在预报，借鉴了现有的基于外推的预报方法，并以完全可微且GPU加速的方式实现了数据的拉格朗日坐标系转换，以允许实时端到端训练和推断。根据我们的评估，LUPIN与并超过了所选择基准的性能，为其他拉格朗日机器学习模型敞开了大门。

    arXiv:2402.10747v1 Announce Type: cross  Abstract: This paper presents a convolutional neural network model for precipitation nowcasting that combines data-driven learning with physics-informed domain knowledge. We propose LUPIN, a Lagrangian Double U-Net for Physics-Informed Nowcasting, that draws from existing extrapolation-based nowcasting methods and implements the Lagrangian coordinate system transformation of the data in a fully differentiable and GPU-accelerated manner to allow for real-time end-to-end training and inference. Based on our evaluation, LUPIN matches and exceeds the performance of the chosen benchmark, opening the door for other Lagrangian machine learning models.
    
[^3]: 行动中的现实主义：使用YOLOv8和DeiT从医学图像中诊断脑肿瘤的异常感知

    Realism in Action: Anomaly-Aware Diagnosis of Brain Tumors from Medical Images Using YOLOv8 and DeiT. (arXiv:2401.03302v1 [eess.IV])

    [http://arxiv.org/abs/2401.03302](http://arxiv.org/abs/2401.03302)

    本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤，并解决了在罕见情况下的肿瘤检测问题。研究使用了来自国家脑映射实验室的数据集，通过修改样本数量和患者分布，使模型能够应对真实世界场景中的异常情况。

    

    在医学科学领域，由于脑肿瘤在患者中的罕见程度，可靠地检测和分类脑肿瘤仍然是一个艰巨的挑战。因此，在异常情况下检测肿瘤的能力对于确保及时干预和改善患者结果至关重要。本研究利用深度学习技术在具有挑战性的情况下检测和分类脑肿瘤。来自国家脑映射实验室（NBML）的精选数据集包括81名患者，其中包括30例肿瘤病例和51例正常病例。检测和分类流程被分为两个连续的任务。检测阶段包括全面的数据分析和预处理，以修改图像样本和每个类别的患者数量，以符合真实世界场景中的异常分布（9个正常样本对应1个肿瘤样本）。此外，在测试中除了常见的评估指标外，我们还采用了... [摘要长度已达到上限]

    In the field of medical sciences, reliable detection and classification of brain tumors from images remains a formidable challenge due to the rarity of tumors within the population of patients. Therefore, the ability to detect tumors in anomaly scenarios is paramount for ensuring timely interventions and improved patient outcomes. This study addresses the issue by leveraging deep learning (DL) techniques to detect and classify brain tumors in challenging situations. The curated data set from the National Brain Mapping Lab (NBML) comprises 81 patients, including 30 Tumor cases and 51 Normal cases. The detection and classification pipelines are separated into two consecutive tasks. The detection phase involved comprehensive data analysis and pre-processing to modify the number of image samples and the number of patients of each class to anomaly distribution (9 Normal per 1 Tumor) to comply with real world scenarios. Next, in addition to common evaluation metrics for the testing, we emplo
    
[^4]: "垃圾DNA假设：通过稀疏性对LLM预训练权重进行任务中心角度分析"

    Junk DNA Hypothesis: A Task-Centric Angle of LLM Pre-trained Weights through Sparsity. (arXiv:2310.02277v1 [cs.LG])

    [http://arxiv.org/abs/2310.02277](http://arxiv.org/abs/2310.02277)

    本文研究通过稀疏性分析LLM预训练权重的任务中心角度，挑战了传统对于权重中冗余性的观点，并提出了"垃圾DNA假设"。

    

    传统对"垃圾DNA"的概念长期以来与人类基因组中的非编码片段相关联，占其组成的大约98%。然而，最近的研究揭示了一些这些看似无功能的DNA序列在细胞过程中起到的关键作用。有趣的是，深度神经网络中的权重与人类基因中观察到的冗余性有着显著的相似性。人们认为，庞大模型中的权重包含了过多的冗余，可以在不影响性能的情况下去除。本文通过提出一个令人信服的反论来挑战这个传统观点。我们使用稀疏性作为一种工具，来独立而准确地量化预训练大语言模型(LLM)中低幅度权重的细微重要性，从下游任务中心的角度理解它们包含的知识。我们提出了支持我们深入研究的"垃圾DNA假设"。

    The traditional notion of "Junk DNA" has long been linked to non-coding segments within the human genome, constituting roughly 98% of its composition. However, recent research has unveiled the critical roles some of these seemingly non-functional DNA sequences play in cellular processes. Intriguingly, the weights within deep neural networks exhibit a remarkable similarity to the redundancy observed in human genes. It was believed that weights in gigantic models contained excessive redundancy, and could be removed without compromising performance. This paper challenges this conventional wisdom by presenting a compelling counter-argument. We employ sparsity as a tool to isolate and quantify the nuanced significance of low-magnitude weights in pre-trained large language models (LLMs). Our study demonstrates a strong correlation between these weight magnitudes and the knowledge they encapsulate, from a downstream task-centric angle. we raise the "Junk DNA Hypothesis" backed by our in-depth
    
[^5]: 实现扩展扩展扩散模型的可控性

    Towards Enhanced Controllability of Diffusion Models. (arXiv:2302.14368v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.14368](http://arxiv.org/abs/2302.14368)

    本文介绍了一种基于条件输入的扩散模型，利用两个潜在编码控制生成过程中的空间结构和语义风格，提出了两种通用采样技术和时间步相关的潜在权重调度，实现了对生成过程的更好控制。

    

    去噪扩散模型在生成逼真、高质量和多样化图像方面表现出卓越能力。然而，在生成过程中的可控程度尚未得到充分探讨。受基于GAN潜在空间的图像操纵技术启发，我们训练了一个条件于两个潜在编码、一个空间内容掩码和一个扁平的样式嵌入的扩散模型。我们依赖于扩散模型渐进去噪过程的感性偏置，在空间结构掩码中编码姿势/布局信息，在样式代码中编码语义/样式信息。我们提出了两种通用的采样技术来改善可控性。我们扩展了可组合的扩散模型，允许部分依赖于条件输入，以提高生成质量，同时还提供对每个潜在代码和它们的联合分布量的控制。我们还提出了时间步相关的内容和样式潜在权重调度，进一步提高了控制性。

    Denoising Diffusion models have shown remarkable capabilities in generating realistic, high-quality and diverse images. However, the extent of controllability during generation is underexplored. Inspired by techniques based on GAN latent space for image manipulation, we train a diffusion model conditioned on two latent codes, a spatial content mask and a flattened style embedding. We rely on the inductive bias of the progressive denoising process of diffusion models to encode pose/layout information in the spatial structure mask and semantic/style information in the style code. We propose two generic sampling techniques for improving controllability. We extend composable diffusion models to allow for some dependence between conditional inputs, to improve the quality of generations while also providing control over the amount of guidance from each latent code and their joint distribution. We also propose timestep dependent weight scheduling for content and style latents to further impro
    

