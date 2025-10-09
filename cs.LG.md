# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Latent Dataset Distillation with Diffusion Models](https://arxiv.org/abs/2403.03881) | 这项研究提出了使用扩散模型进行潜在数据集蒸馏（LD3M），结合潜在空间中的扩散和数据集蒸馏的方法，以解决不同模型架构导致准确性下降和生成高分辨率图像的挑战。 |
| [^2] | [Train-Free Segmentation in MRI with Cubical Persistent Homology.](http://arxiv.org/abs/2401.01160) | 这是一种使用拓扑数据分析进行MRI图像分割的新方法，相比传统机器学习方法具有优势，无需大量注释数据集，提供更可解释和稳定的分割框架。 |
| [^3] | [Differential Privacy for Adaptive Weight Aggregation in Federated Tumor Segmentation.](http://arxiv.org/abs/2308.00856) | 本研究提出了一种针对联邦肿瘤分割中自适应权重聚合的差分隐私算法，通过扩展相似性权重聚合方法（SimAgg），提高了模型分割能力，并在保护隐私方面做出了额外改进。 |
| [^4] | [Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples.](http://arxiv.org/abs/2209.03358) | 这项研究主要关注于脉冲神经网络(SNNs)对抗性样本的鲁棒性和转移性。研究发现，成功的白盒对抗攻击SNNs在很大程度上依赖于替代梯度技术，并且非SNN架构创建的对抗样本往往不被SNNs误分类。 |

# 详细

[^1]: 使用扩散模型进行潜在数据集蒸馏

    Latent Dataset Distillation with Diffusion Models

    [https://arxiv.org/abs/2403.03881](https://arxiv.org/abs/2403.03881)

    这项研究提出了使用扩散模型进行潜在数据集蒸馏（LD3M），结合潜在空间中的扩散和数据集蒸馏的方法，以解决不同模型架构导致准确性下降和生成高分辨率图像的挑战。

    

    机器学习的有效性传统上依赖于越来越大的数据集的可用性。然而，大型数据集带来存储挑战，并且包含一些非影响力样本，在训练过程中可以被忽略而不影响模型最终的准确性。为了应对这些限制，出现了将数据集信息蒸馏成一组压缩样本（合成样本），即蒸馏数据集的概念。其中一个关键方面是选择用于连接原始和合成数据集的架构（通常是ConvNet）。然而，如果所使用的模型架构与蒸馏过程中使用的模型不同，则最终准确性会降低。另一个挑战是生成高分辨率图像，例如128x128及更高。

    arXiv:2403.03881v1 Announce Type: cross  Abstract: The efficacy of machine learning has traditionally relied on the availability of increasingly larger datasets. However, large datasets pose storage challenges and contain non-influential samples, which could be ignored during training without impacting the final accuracy of the model. In response to these limitations, the concept of distilling the information on a dataset into a condensed set of (synthetic) samples, namely a distilled dataset, emerged. One crucial aspect is the selected architecture (usually ConvNet) for linking the original and synthetic datasets. However, the final accuracy is lower if the employed model architecture differs from the model used during distillation. Another challenge is the generation of high-resolution images, e.g., 128x128 and higher. In this paper, we propose Latent Dataset Distillation with Diffusion Models (LD3M) that combine diffusion in latent space with dataset distillation to tackle both chal
    
[^2]: 无需训练的MRI立方持续同调分割方法

    Train-Free Segmentation in MRI with Cubical Persistent Homology. (arXiv:2401.01160v1 [eess.IV])

    [http://arxiv.org/abs/2401.01160](http://arxiv.org/abs/2401.01160)

    这是一种使用拓扑数据分析进行MRI图像分割的新方法，相比传统机器学习方法具有优势，无需大量注释数据集，提供更可解释和稳定的分割框架。

    

    我们描述了一种新的MRI扫描分割方法，使用拓扑数据分析（TDA），相比传统的机器学习方法具有几个优点。它分为三个步骤，首先通过自动阈值确定要分割的整个对象，然后检测一个已知拓扑结构的独特子集，最后推导出分割的各个组成部分。虽然调用了TDA的经典思想，但这样的算法从未与深度学习方法分离提出。为了实现这一点，我们的方法除了考虑图像的同调性外，还考虑了代表性周期的定位，这是在这种情况下似乎从未被利用过的信息。特别是，它提供了无需大量注释数据集进行分割的能力。TDA还通过将拓扑特征明确映射到分割组件来提供更可解释和稳定的分割框架。

    We describe a new general method for segmentation in MRI scans using Topological Data Analysis (TDA), offering several advantages over traditional machine learning approaches. It works in three steps, first identifying the whole object to segment via automatic thresholding, then detecting a distinctive subset whose topology is known in advance, and finally deducing the various components of the segmentation. Although convoking classical ideas of TDA, such an algorithm has never been proposed separately from deep learning methods. To achieve this, our approach takes into account, in addition to the homology of the image, the localization of representative cycles, a piece of information that seems never to have been exploited in this context. In particular, it offers the ability to perform segmentation without the need for large annotated data sets. TDA also provides a more interpretable and stable framework for segmentation by explicitly mapping topological features to segmentation comp
    
[^3]: 针对联邦肿瘤分割中自适应权重聚合的差分隐私研究

    Differential Privacy for Adaptive Weight Aggregation in Federated Tumor Segmentation. (arXiv:2308.00856v1 [cs.LG])

    [http://arxiv.org/abs/2308.00856](http://arxiv.org/abs/2308.00856)

    本研究提出了一种针对联邦肿瘤分割中自适应权重聚合的差分隐私算法，通过扩展相似性权重聚合方法（SimAgg），提高了模型分割能力，并在保护隐私方面做出了额外改进。

    

    联邦学习是一种分布式机器学习方法，通过创建一个公正的全局模型来保护个体客户数据的隐私。然而，传统的联邦学习方法在处理不同客户数据时可能引入安全风险，从而可能危及隐私和数据完整性。为了解决这些挑战，本文提出了一种差分隐私联邦深度学习框架，在医学图像分割中扩展了相似性权重聚合方法（SimAgg）到DP-SimAgg算法，这是一种针对多模态磁共振成像（MRI）中的脑肿瘤分割的差分隐私相似性加权聚合算法。我们的DP-SimAgg方法不仅提高了模型分割能力，还提供了额外的隐私保护层。通过广泛的基准测试和评估，以计算性能为主要考虑因素，证明了DP-SimAgg使..

    Federated Learning (FL) is a distributed machine learning approach that safeguards privacy by creating an impartial global model while respecting the privacy of individual client data. However, the conventional FL method can introduce security risks when dealing with diverse client data, potentially compromising privacy and data integrity. To address these challenges, we present a differential privacy (DP) federated deep learning framework in medical image segmentation. In this paper, we extend our similarity weight aggregation (SimAgg) method to DP-SimAgg algorithm, a differentially private similarity-weighted aggregation algorithm for brain tumor segmentation in multi-modal magnetic resonance imaging (MRI). Our DP-SimAgg method not only enhances model segmentation capabilities but also provides an additional layer of privacy preservation. Extensive benchmarking and evaluation of our framework, with computational performance as a key consideration, demonstrate that DP-SimAgg enables a
    
[^4]: 攻击脉冲：关于脉冲神经网络对抗性样本的可转移性与安全性的研究

    Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples. (arXiv:2209.03358v3 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2209.03358](http://arxiv.org/abs/2209.03358)

    这项研究主要关注于脉冲神经网络(SNNs)对抗性样本的鲁棒性和转移性。研究发现，成功的白盒对抗攻击SNNs在很大程度上依赖于替代梯度技术，并且非SNN架构创建的对抗样本往往不被SNNs误分类。

    

    脉冲神经网络(SNNs)因其高能效和最近在分类性能上的进展而受到广泛关注。然而，与传统的深度学习方法不同，对SNNs对抗性样本的鲁棒性的分析和研究仍然相对不完善。在这项工作中，我们关注于推进SNNs的对抗攻击方面，并做出了三个主要贡献。首先，我们展示了成功的白盒对抗攻击SNNs在很大程度上依赖于底层的替代梯度技术，即使在对抗性训练SNNs的情况下也一样。其次，利用最佳的替代梯度技术，我们分析了对抗攻击在SNNs和其他最先进的架构如Vision Transformers(ViTs)和Big Transfer Convolutional Neural Networks(CNNs)之间的可转移性。我们证明了非SNN架构创建的对抗样本往往不被SNNs误分类。第三，由于缺乏一个共性

    Spiking neural networks (SNNs) have attracted much attention for their high energy efficiency and for recent advances in their classification performance. However, unlike traditional deep learning approaches, the analysis and study of the robustness of SNNs to adversarial examples remain relatively underdeveloped. In this work, we focus on advancing the adversarial attack side of SNNs and make three major contributions. First, we show that successful white-box adversarial attacks on SNNs are highly dependent on the underlying surrogate gradient technique, even in the case of adversarially trained SNNs. Second, using the best surrogate gradient technique, we analyze the transferability of adversarial attacks on SNNs and other state-of-the-art architectures like Vision Transformers (ViTs) and Big Transfer Convolutional Neural Networks (CNNs). We demonstrate that the adversarial examples created by non-SNN architectures are not misclassified often by SNNs. Third, due to the lack of an ubi
    

