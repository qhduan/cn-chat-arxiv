# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AVicuna: Audio-Visual LLM with Interleaver and Context-Boundary Alignment for Temporal Referential Dialogue](https://arxiv.org/abs/2403.16276) | 介绍了一个新的框架AVicuna，生成了PU-VALOR数据集，解决了音频-视觉时间指代对话中的两个主要挑战：缺乏准确时间注释的数据集和整合复杂时间线索的方法。 |
| [^2] | [Do Large Language Model Understand Multi-Intent Spoken Language ?](https://arxiv.org/abs/2403.04481) | 该研究利用大型语言模型进行口语语言多目标理解，提出了改进实体槽和子目标指令的创新技术，并展示了LLMs在多目标SLU模型方面的潜力。 |
| [^3] | [Latent Dataset Distillation with Diffusion Models](https://arxiv.org/abs/2403.03881) | 这项研究提出了使用扩散模型进行潜在数据集蒸馏（LD3M），结合潜在空间中的扩散和数据集蒸馏的方法，以解决不同模型架构导致准确性下降和生成高分辨率图像的挑战。 |
| [^4] | [Is my Data in your AI Model? Membership Inference Test with Application to Face Images](https://arxiv.org/abs/2402.09225) | This paper introduces a novel approach called Membership Inference Test (MINT) to empirically assess if specific data was used during the training of AI models. Two MINT architectures based on MLP and CNN are proposed and evaluated on a challenging face recognition task, achieving promising results with up to 90% accuracy. |
| [^5] | [Inferring Capabilities from Task Performance with Bayesian Triangulation.](http://arxiv.org/abs/2309.11975) | 本研究提出了一种使用贝叶斯三角测量方法从任务表现中推断系统能力的方法，并利用该方法推断了不同认知特征的代理。这种能力导向的评估方法对于机器学习模型的表征具有潜在的应用价值。 |
| [^6] | [Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples.](http://arxiv.org/abs/2209.03358) | 这项研究主要关注于脉冲神经网络(SNNs)对抗性样本的鲁棒性和转移性。研究发现，成功的白盒对抗攻击SNNs在很大程度上依赖于替代梯度技术，并且非SNN架构创建的对抗样本往往不被SNNs误分类。 |

# 详细

[^1]: AVicuna：具有交错器和上下文边界对齐的音频-视觉LLM用于时间指代对话

    AVicuna: Audio-Visual LLM with Interleaver and Context-Boundary Alignment for Temporal Referential Dialogue

    [https://arxiv.org/abs/2403.16276](https://arxiv.org/abs/2403.16276)

    介绍了一个新的框架AVicuna，生成了PU-VALOR数据集，解决了音频-视觉时间指代对话中的两个主要挑战：缺乏准确时间注释的数据集和整合复杂时间线索的方法。

    

    在日常交流中，人类经常使用语音和手势来指代特定区域或对象，这个过程称为指代对话（RD）。尽管先前的研究已经通过大型语言模型（LLMs）或大型多模型模型（LMMs）在静态环境中调查了RD，但在音频-视觉媒体中探索时间指代对话（TRD）仍然有限。两个主要挑战阻碍了这一领域的进展：（1）缺乏具有精确时间注释的全面未修剪音频-视觉视频数据集，以及（2）需要有效整合复杂的时间听觉和视觉线索的方法。为了解决这些挑战，我们引入了一个新的框架，生成PU-VALOR，这是一个包含超过114,000个未修剪视频的广泛音频-视觉数据集，并介绍了AVicuna，具有音频-视觉令牌交错器（AVTI），确保了时间对齐。

    arXiv:2403.16276v1 Announce Type: cross  Abstract: In everyday communication, humans frequently use speech and gestures to refer to specific areas or objects, a process known as Referential Dialogue (RD). While prior studies have investigated RD through Large Language Models (LLMs) or Large Multimodal Models (LMMs) in static contexts, the exploration of Temporal Referential Dialogue (TRD) within audio-visual media remains limited. Two primary challenges hinder progress in this field: (1) the absence of comprehensive, untrimmed audio-visual video datasets with precise temporal annotations, and (2) the need for methods to integrate complex temporal auditory and visual cues effectively. To address these challenges, we introduce a novel framework to generate PU-VALOR, an extensive audio-visual dataset comprising over 114,000 untrimmed videos with accurate temporal demarcations. We also present AVicuna, featuring an Audio-Visual Tokens Interleaver (AVTI) that ensures the temporal alignment 
    
[^2]: 大型语言模型能理解多目标口语语言吗？

    Do Large Language Model Understand Multi-Intent Spoken Language ?

    [https://arxiv.org/abs/2403.04481](https://arxiv.org/abs/2403.04481)

    该研究利用大型语言模型进行口语语言多目标理解，提出了改进实体槽和子目标指令的创新技术，并展示了LLMs在多目标SLU模型方面的潜力。

    

    这项研究通过利用大型语言模型（LLMs）进行多目标口语语言理解（SLU）取得了重大进展，提出了一种在SLU环境中利用LLMs生成能力的独特方法。我们的创新技术重新配置了实体槽，专门用于LLMs在多目标SLU环境中的应用，并引入了子目标指令（SII）的概念，增强了对不同领域内复杂多目标交流的解剖和解释。由此产生的数据集，被称为LM-MixATIS和LM-MixSNIPS，是从现有基准中精心制作的。我们的研究表明，LLMs可以匹配并潜在地超越当前最先进的多目标SLU模型的能力。它进一步探讨了LLMs在各种意图配置和数据集比例下的有效性。此外，我们介绍了两个开创性的度量标准，即实体槽准确性（ESA）和Com

    arXiv:2403.04481v1 Announce Type: cross  Abstract: This study marks a significant advancement by harnessing Large Language Models (LLMs) for multi-intent spoken language understanding (SLU), proposing a unique methodology that capitalizes on the generative power of LLMs within an SLU context. Our innovative technique reconfigures entity slots specifically for LLM application in multi-intent SLU environments and introduces the concept of Sub-Intent Instruction (SII), enhancing the dissection and interpretation of intricate, multi-intent communication within varied domains. The resultant datasets, dubbed LM-MixATIS and LM-MixSNIPS, are crafted from pre-existing benchmarks. Our research illustrates that LLMs can match and potentially excel beyond the capabilities of current state-of-the-art multi-intent SLU models. It further explores LLM efficacy across various intent configurations and dataset proportions. Moreover, we introduce two pioneering metrics, Entity Slot Accuracy (ESA) and Com
    
[^3]: 使用扩散模型进行潜在数据集蒸馏

    Latent Dataset Distillation with Diffusion Models

    [https://arxiv.org/abs/2403.03881](https://arxiv.org/abs/2403.03881)

    这项研究提出了使用扩散模型进行潜在数据集蒸馏（LD3M），结合潜在空间中的扩散和数据集蒸馏的方法，以解决不同模型架构导致准确性下降和生成高分辨率图像的挑战。

    

    机器学习的有效性传统上依赖于越来越大的数据集的可用性。然而，大型数据集带来存储挑战，并且包含一些非影响力样本，在训练过程中可以被忽略而不影响模型最终的准确性。为了应对这些限制，出现了将数据集信息蒸馏成一组压缩样本（合成样本），即蒸馏数据集的概念。其中一个关键方面是选择用于连接原始和合成数据集的架构（通常是ConvNet）。然而，如果所使用的模型架构与蒸馏过程中使用的模型不同，则最终准确性会降低。另一个挑战是生成高分辨率图像，例如128x128及更高。

    arXiv:2403.03881v1 Announce Type: cross  Abstract: The efficacy of machine learning has traditionally relied on the availability of increasingly larger datasets. However, large datasets pose storage challenges and contain non-influential samples, which could be ignored during training without impacting the final accuracy of the model. In response to these limitations, the concept of distilling the information on a dataset into a condensed set of (synthetic) samples, namely a distilled dataset, emerged. One crucial aspect is the selected architecture (usually ConvNet) for linking the original and synthetic datasets. However, the final accuracy is lower if the employed model architecture differs from the model used during distillation. Another challenge is the generation of high-resolution images, e.g., 128x128 and higher. In this paper, we propose Latent Dataset Distillation with Diffusion Models (LD3M) that combine diffusion in latent space with dataset distillation to tackle both chal
    
[^4]: 我的数据在你的AI模型中吗？通过应用于人脸图像的成员推断测试

    Is my Data in your AI Model? Membership Inference Test with Application to Face Images

    [https://arxiv.org/abs/2402.09225](https://arxiv.org/abs/2402.09225)

    This paper introduces a novel approach called Membership Inference Test (MINT) to empirically assess if specific data was used during the training of AI models. Two MINT architectures based on MLP and CNN are proposed and evaluated on a challenging face recognition task, achieving promising results with up to 90% accuracy.

    

    这篇论文介绍了成员推断测试（MINT），一种用于经验性评估特定数据是否被用于训练人工智能（AI）模型的新方法。具体而言，我们提出了两种新颖的MINT架构，旨在学习在经过审计的模型暴露于其训练过程中使用的数据时出现的不同激活模式。第一个架构基于多层感知机（MLP）网络，第二个基于卷积神经网络（CNN）。所提出的MINT架构在具有挑战性的人脸识别任务上进行评估，考虑了三种最先进的人脸识别模型。使用六个公开可用的数据库进行实验，总共包含超过2200万张人脸图像。根据可用的AI模型测试的上下文，考虑了不同的实验场景。有希望的结果达到了90%的准确率。

    arXiv:2402.09225v1 Announce Type: cross Abstract: This paper introduces the Membership Inference Test (MINT), a novel approach that aims to empirically assess if specific data was used during the training of Artificial Intelligence (AI) models. Specifically, we propose two novel MINT architectures designed to learn the distinct activation patterns that emerge when an audited model is exposed to data used during its training process. The first architecture is based on a Multilayer Perceptron (MLP) network and the second one is based on Convolutional Neural Networks (CNNs). The proposed MINT architectures are evaluated on a challenging face recognition task, considering three state-of-the-art face recognition models. Experiments are carried out using six publicly available databases, comprising over 22 million face images in total. Also, different experimental scenarios are considered depending on the context available of the AI model to test. Promising results, up to 90% accuracy, are a
    
[^5]: 从任务表现中推断能力的贝叶斯三角测量方法

    Inferring Capabilities from Task Performance with Bayesian Triangulation. (arXiv:2309.11975v1 [cs.AI])

    [http://arxiv.org/abs/2309.11975](http://arxiv.org/abs/2309.11975)

    本研究提出了一种使用贝叶斯三角测量方法从任务表现中推断系统能力的方法，并利用该方法推断了不同认知特征的代理。这种能力导向的评估方法对于机器学习模型的表征具有潜在的应用价值。

    

    随着机器学习模型变得更加通用，我们需要以更丰富、更有意义的方式对其进行表征。我们描述了一种从多样化实验数据中推断系统的认知特征的方法。为此，我们引入了测量布局，模拟了任务实例特征如何与系统能力相互作用以影响性能。这些特征必须以复杂的方式进行三角测量，以便从非群体数据中推断能力，这对于传统的心理测量和推理工具是一个挑战。利用贝叶斯概率编程库PyMC，我们推断了两种情景中代理的不同认知特征：动物智能奥林匹克竞赛中的68名实际参赛选手和O-PIAAGETS的30个合成代理，其中O-PIAAGETS是一个物体恒常性测验。我们展示了以能力为导向的评估的潜力。

    As machine learning models become more general, we need to characterise them in richer, more meaningful ways. We describe a method to infer the cognitive profile of a system from diverse experimental data. To do so, we introduce measurement layouts that model how task-instance features interact with system capabilities to affect performance. These features must be triangulated in complex ways to be able to infer capabilities from non-populational data -- a challenge for traditional psychometric and inferential tools. Using the Bayesian probabilistic programming library PyMC, we infer different cognitive profiles for agents in two scenarios: 68 actual contestants in the AnimalAI Olympics and 30 synthetic agents for O-PIAAGETS, an object permanence battery. We showcase the potential for capability-oriented evaluation.
    
[^6]: 攻击脉冲：关于脉冲神经网络对抗性样本的可转移性与安全性的研究

    Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples. (arXiv:2209.03358v3 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2209.03358](http://arxiv.org/abs/2209.03358)

    这项研究主要关注于脉冲神经网络(SNNs)对抗性样本的鲁棒性和转移性。研究发现，成功的白盒对抗攻击SNNs在很大程度上依赖于替代梯度技术，并且非SNN架构创建的对抗样本往往不被SNNs误分类。

    

    脉冲神经网络(SNNs)因其高能效和最近在分类性能上的进展而受到广泛关注。然而，与传统的深度学习方法不同，对SNNs对抗性样本的鲁棒性的分析和研究仍然相对不完善。在这项工作中，我们关注于推进SNNs的对抗攻击方面，并做出了三个主要贡献。首先，我们展示了成功的白盒对抗攻击SNNs在很大程度上依赖于底层的替代梯度技术，即使在对抗性训练SNNs的情况下也一样。其次，利用最佳的替代梯度技术，我们分析了对抗攻击在SNNs和其他最先进的架构如Vision Transformers(ViTs)和Big Transfer Convolutional Neural Networks(CNNs)之间的可转移性。我们证明了非SNN架构创建的对抗样本往往不被SNNs误分类。第三，由于缺乏一个共性

    Spiking neural networks (SNNs) have attracted much attention for their high energy efficiency and for recent advances in their classification performance. However, unlike traditional deep learning approaches, the analysis and study of the robustness of SNNs to adversarial examples remain relatively underdeveloped. In this work, we focus on advancing the adversarial attack side of SNNs and make three major contributions. First, we show that successful white-box adversarial attacks on SNNs are highly dependent on the underlying surrogate gradient technique, even in the case of adversarially trained SNNs. Second, using the best surrogate gradient technique, we analyze the transferability of adversarial attacks on SNNs and other state-of-the-art architectures like Vision Transformers (ViTs) and Big Transfer Convolutional Neural Networks (CNNs). We demonstrate that the adversarial examples created by non-SNN architectures are not misclassified often by SNNs. Third, due to the lack of an ubi
    

