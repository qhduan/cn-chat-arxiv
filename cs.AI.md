# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DiffusionAct: Controllable Diffusion Autoencoder for One-shot Face Reenactment](https://arxiv.org/abs/2403.17217) | DiffusionAct是一种利用扩散模型进行神经人脸再现的新方法，能够编辑输入图像的面部姿势，实现身份和外观的保留，以及目标头部姿势和面部表情的转移。 |
| [^2] | [Measuring Non-Typical Emotions for Mental Health: A Survey of Computational Approaches](https://arxiv.org/abs/2403.08824) | 本调查是第一次同时探索用于分析压力、抑郁和投入度的计算方法，并讨论了最常用的数据集、输入模态、数据处理技术和信息融合方法。 |
| [^3] | [Morphological Symmetries in Robotics](https://arxiv.org/abs/2402.15552) | 形态对称性是机器人系统中的固有性质，通过对运动结构和质量的对称分布，延伸至机器人状态空间和传感器测量，进而影响机器人的运动方程和最优控制策略，并在机器人学建模、控制和设计中具有重要意义。 |
| [^4] | [Promoting Segment Anything Model towards Highly Accurate Dichotomous Image Segmentation](https://arxiv.org/abs/2401.00248) | 将段分离任意模型推进至高度准确的二元图像分割，通过提出DIS-SAM框架，成功改进SAM模型在细节方面的表现，实现了显著增强的分割精度。 |
| [^5] | [On Diffusion Modeling for Anomaly Detection.](http://arxiv.org/abs/2305.18593) | 本文研究了扩散建模在无监督和半监督异常检测中的应用，发现去噪扩散概率模型表现很好但计算成本高，因此提出了一种替代方法——扩散时间概率模型，该模型能够通过较大的时间步长上的高后验密度识别异常，并通过深度神经网络提高效率。 |
| [^6] | [Polysemanticity and Capacity in Neural Networks.](http://arxiv.org/abs/2210.01892) | 该论文通过分析特征容量来理解神经网络中的多义性现象，发现在最优的容量分配下，神经网络倾向于单义地表示重要特征，多义地表示次重要特征，并忽略最不重要的特征。多义性现象在输入具有更高的峰度或稀疏性时更为普遍，并且在某些体系结构中比其他体系结构更为普遍。此外，作者还发现了嵌入空间中的分块半正交结构，不同模型中的分块大小不同，突出了模型体系结构的影响。 |

# 详细

[^1]: DiffusionAct：可控扩散自编码器用于一次性人脸再现

    DiffusionAct: Controllable Diffusion Autoencoder for One-shot Face Reenactment

    [https://arxiv.org/abs/2403.17217](https://arxiv.org/abs/2403.17217)

    DiffusionAct是一种利用扩散模型进行神经人脸再现的新方法，能够编辑输入图像的面部姿势，实现身份和外观的保留，以及目标头部姿势和面部表情的转移。

    

    视频驱动的神经人脸再现旨在合成能成功保留源脸的身份和外观，同时转移目标头部姿势和面部表情的逼真面部图像。现有基于GAN的方法要么存在失真和视觉伪影，要么重构质量较差，即背景和一些重要的外观细节（如发型/颜色、眼镜和配饰）未被忠实重建。最近在扩散概率模型（DPMs）领域的进展使得生成高质量逼真图像成为可能。为此，本文提出了DiffusionAct，这是一种利用扩散模型的照片逼真图像生成来进行神经人脸再现的新方法。具体来说，我们提出控制Diffusion自编码器（DiffAE）的语义空间，以便编辑输入图像的面部姿势，定义为头部姿势方向。

    arXiv:2403.17217v1 Announce Type: cross  Abstract: Video-driven neural face reenactment aims to synthesize realistic facial images that successfully preserve the identity and appearance of a source face, while transferring the target head pose and facial expressions. Existing GAN-based methods suffer from either distortions and visual artifacts or poor reconstruction quality, i.e., the background and several important appearance details, such as hair style/color, glasses and accessories, are not faithfully reconstructed. Recent advances in Diffusion Probabilistic Models (DPMs) enable the generation of high-quality realistic images. To this end, in this paper we present DiffusionAct, a novel method that leverages the photo-realistic image generation of diffusion models to perform neural face reenactment. Specifically, we propose to control the semantic space of a Diffusion Autoencoder (DiffAE), in order to edit the facial pose of the input images, defined as the head pose orientation an
    
[^2]: 衡量非典型情绪对心理健康的影响：计算方法调查

    Measuring Non-Typical Emotions for Mental Health: A Survey of Computational Approaches

    [https://arxiv.org/abs/2403.08824](https://arxiv.org/abs/2403.08824)

    本调查是第一次同时探索用于分析压力、抑郁和投入度的计算方法，并讨论了最常用的数据集、输入模态、数据处理技术和信息融合方法。

    

    非典型情绪（如压力、抑郁和投入度分析）相较于经常讨论的情绪（如快乐、悲伤、恐惧和愤怒）来说，更为罕见且复杂。由于对心理健康和幸福的影响，人们对这些非典型情绪的重要性越来越认识到。压力和抑郁影响了日常任务的参与，突显了理解它们相互作用的必要性。本调查是第一次同时探索用于分析压力、抑郁和投入度的计算方法。我们讨论了用于计算分析压力、抑郁和投入度的最常用的数据集、输入模态、数据处理技术和信息融合方法。我们提供了非典型情绪分析方法的时间表和分类法，以及它们的通用流程和类别。随后，我们描述了最先进的计算方法。

    arXiv:2403.08824v1 Announce Type: cross  Abstract: Analysis of non-typical emotions, such as stress, depression and engagement is less common and more complex compared to that of frequently discussed emotions like happiness, sadness, fear, and anger. The importance of these non-typical emotions has been increasingly recognized due to their implications on mental health and well-being. Stress and depression impact the engagement in daily tasks, highlighting the need to understand their interplay. This survey is the first to simultaneously explore computational methods for analyzing stress, depression, and engagement. We discuss the most commonly used datasets, input modalities, data processing techniques, and information fusion methods used for the computational analysis of stress, depression and engagement. A timeline and taxonomy of non-typical emotion analysis approaches along with their generic pipeline and categories are presented. Subsequently, we describe state-of-the-art computa
    
[^3]: 机器人学中的形态对称性

    Morphological Symmetries in Robotics

    [https://arxiv.org/abs/2402.15552](https://arxiv.org/abs/2402.15552)

    形态对称性是机器人系统中的固有性质，通过对运动结构和质量的对称分布，延伸至机器人状态空间和传感器测量，进而影响机器人的运动方程和最优控制策略，并在机器人学建模、控制和设计中具有重要意义。

    

    我们提出了一个全面的框架来研究和利用机器人系统中的形态对称性。这些是机器人形态的固有特性，经常在动物生物学和机器人学中观察到，源于运动结构的复制和质量的对称分布。我们说明了这些对称性如何延伸到机器人的状态空间以及本体感知和外部感知传感器测量，导致机器人的运动方程和最优控制策略的等不变性。因此，我们认识到形态对称性作为一个相关且以前未被探索的受物理启示的几何先验，对机器人建模、控制、估计和设计中使用的数据驱动和分析方法都具有重要影响。对于数据驱动方法，我们演示了形态对称性如何提高机器学习模型的样本效率和泛化能力

    arXiv:2402.15552v1 Announce Type: cross  Abstract: We present a comprehensive framework for studying and leveraging morphological symmetries in robotic systems. These are intrinsic properties of the robot's morphology, frequently observed in animal biology and robotics, which stem from the replication of kinematic structures and the symmetrical distribution of mass. We illustrate how these symmetries extend to the robot's state space and both proprioceptive and exteroceptive sensor measurements, resulting in the equivariance of the robot's equations of motion and optimal control policies. Thus, we recognize morphological symmetries as a relevant and previously unexplored physics-informed geometric prior, with significant implications for both data-driven and analytical methods used in modeling, control, estimation and design in robotics. For data-driven methods, we demonstrate that morphological symmetries can enhance the sample efficiency and generalization of machine learning models 
    
[^4]: 将“段分离任意模型”推进至高度准确的二元图像分割

    Promoting Segment Anything Model towards Highly Accurate Dichotomous Image Segmentation

    [https://arxiv.org/abs/2401.00248](https://arxiv.org/abs/2401.00248)

    将段分离任意模型推进至高度准确的二元图像分割，通过提出DIS-SAM框架，成功改进SAM模型在细节方面的表现，实现了显著增强的分割精度。

    

    Segment Anything Model (SAM)代表了计算机视觉基础模型的重大突破，提供了大规模图像分割模型。然而，尽管SAM的零-shot表现，其分割蒙版缺乏细粒度细节，特别是在准确描绘对象边界方面。我们对SAM是否可以作为基础模型进一步改进以实现高度精确的对象分割（即称为二元图像分割DIS）抱有很高期望。为解决这一问题，我们提出了DIS-SAM，将SAM推进至DIS，具有极高的精确细节。DIS-SAM是一个专门为高度准确分割而设计的框架，保持了SAM的可促进设计。DIS-SAM采用了两阶段方法，将SAM与专门用于DIS的修改后的IS-Net集成在一起。尽管简单，DIS-SAM相比SAM和HQ-SA表现出显着增强的分割精度。

    arXiv:2401.00248v2 Announce Type: replace-cross  Abstract: The Segment Anything Model (SAM) represents a significant breakthrough into foundation models for computer vision, providing a large-scale image segmentation model. However, despite SAM's zero-shot performance, its segmentation masks lack fine-grained details, particularly in accurately delineating object boundaries. We have high expectations regarding whether SAM, as a foundation model, can be improved towards highly accurate object segmentation, which is known as dichotomous image segmentation (DIS). To address this issue, we propose DIS-SAM, which advances SAM towards DIS with extremely accurate details. DIS-SAM is a framework specifically tailored for highly accurate segmentation, maintaining SAM's promptable design. DIS-SAM employs a two-stage approach, integrating SAM with a modified IS-Net dedicated to DIS. Despite its simplicity, DIS-SAM demonstrates significantly enhanced segmentation accuracy compared to SAM and HQ-SA
    
[^5]: 关于扩散建模在异常检测中的应用

    On Diffusion Modeling for Anomaly Detection. (arXiv:2305.18593v1 [cs.LG])

    [http://arxiv.org/abs/2305.18593](http://arxiv.org/abs/2305.18593)

    本文研究了扩散建模在无监督和半监督异常检测中的应用，发现去噪扩散概率模型表现很好但计算成本高，因此提出了一种替代方法——扩散时间概率模型，该模型能够通过较大的时间步长上的高后验密度识别异常，并通过深度神经网络提高效率。

    

    扩散模型以其在生成建模中的优异性能而闻名，成为基于密度的异常检测的有吸引力的候选算法。本文研究了各种扩散建模方法在无监督和半监督异常检测中的应用。尤其是发现去噪扩散概率模型（DDPM）在异常检测方面具备很好的表现，但计算成本较高。通过简化DDPM在异常检测中的应用，我们自然地引出另一种称为扩散时间概率模型（DTPM）的替代方法。DTPM估计给定输入的扩散时间的后验分布，能够通过较大的时间步长上的高后验密度识别异常。我们导出了此后验分布的解析形式，并利用深度神经网络提高推理效率。通过在ADBenh基准测试中的实证评估，我们证明了所有基于扩散的异常检测方法的实用性。

    Known for their impressive performance in generative modeling, diffusion models are attractive candidates for density-based anomaly detection. This paper investigates different variations of diffusion modeling for unsupervised and semi-supervised anomaly detection. In particular, we find that Denoising Diffusion Probability Models (DDPM) are performant on anomaly detection benchmarks yet computationally expensive. By simplifying DDPM in application to anomaly detection, we are naturally led to an alternative approach called Diffusion Time Probabilistic Model (DTPM). DTPM estimates the posterior distribution over diffusion time for a given input, enabling the identification of anomalies due to their higher posterior density at larger timesteps. We derive an analytical form for this posterior density and leverage a deep neural network to improve inference efficiency. Through empirical evaluations on the ADBench benchmark, we demonstrate that all diffusion-based anomaly detection methods 
    
[^6]: 神经网络中的多义性和容量

    Polysemanticity and Capacity in Neural Networks. (arXiv:2210.01892v3 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2210.01892](http://arxiv.org/abs/2210.01892)

    该论文通过分析特征容量来理解神经网络中的多义性现象，发现在最优的容量分配下，神经网络倾向于单义地表示重要特征，多义地表示次重要特征，并忽略最不重要的特征。多义性现象在输入具有更高的峰度或稀疏性时更为普遍，并且在某些体系结构中比其他体系结构更为普遍。此外，作者还发现了嵌入空间中的分块半正交结构，不同模型中的分块大小不同，突出了模型体系结构的影响。

    

    神经网络中的单个神经元通常代表无关特征的混合。这种现象称为多义性，使解释神经网络变得更加困难，因此我们的目标是理解其原因。我们提出通过特征容量的视角来理解这一现象，特征容量是每个特征在嵌入空间中占用的分形维度。我们展示了在一个玩具模型中，最优的容量分配倾向于单义地表示最重要的特征，多义地表示次重要特征（与其对损失的影响成比例），并完全忽略最不重要的特征。当输入具有更高的峰度或稀疏性时，多义性更为普遍，并且在某些体系结构中比其他体系结构更为普遍。在得到最优容量分配后，我们进一步研究了嵌入空间的几何结构。我们发现了一个分块半正交的结构，不同模型中的分块大小不同，突出了模型体系结构的影响。

    Individual neurons in neural networks often represent a mixture of unrelated features. This phenomenon, called polysemanticity, can make interpreting neural networks more difficult and so we aim to understand its causes. We propose doing so through the lens of feature \emph{capacity}, which is the fractional dimension each feature consumes in the embedding space. We show that in a toy model the optimal capacity allocation tends to monosemantically represent the most important features, polysemantically represent less important features (in proportion to their impact on the loss), and entirely ignore the least important features. Polysemanticity is more prevalent when the inputs have higher kurtosis or sparsity and more prevalent in some architectures than others. Given an optimal allocation of capacity, we go on to study the geometry of the embedding space. We find a block-semi-orthogonal structure, with differing block sizes in different models, highlighting the impact of model archit
    

