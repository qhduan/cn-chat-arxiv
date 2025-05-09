# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey on Consumer IoT Traffic: Security and Privacy](https://arxiv.org/abs/2403.16149) | 本调查针对消费者物联网（CIoT）流量分析从安全和隐私的角度出发，总结了CIoT流量分析的新特征、最新进展和挑战，认为通过流量分析可以揭示CIoT领域中的安全和隐私问题。 |
| [^2] | [Evaluating Fairness Metrics Across Borders from Human Perceptions](https://arxiv.org/abs/2403.16101) | 该研究通过国际调查评估了不同国家对其决策情景中各种公平度量标准的适用性。 |
| [^3] | [DyCE: Dynamic Configurable Exiting for Deep Learning Compression and Scaling](https://arxiv.org/abs/2403.01695) | 介绍了DyCE，一个动态可配置的提前退出框架，将设计考虑从彼此和基础模型解耦 |
| [^4] | [Connecting NTK and NNGP: A Unified Theoretical Framework for Neural Network Learning Dynamics in the Kernel Regime.](http://arxiv.org/abs/2309.04522) | 本文提出了一个马尔可夫近似学习模型，统一了神经切向核（NTK）和神经网络高斯过程（NNGP）核，用于描述无限宽度深层网络的学习动力学。 |
| [^5] | [An automated end-to-end deep learning-based framework for lung cancer diagnosis by detecting and classifying the lung nodules.](http://arxiv.org/abs/2305.00046) | 本文提出了一种基于深度学习的智能诊断框架，针对低资源环境实现早期检测和分类肺部结节，并在公共数据集上取得了较好的表现。 |
| [^6] | [Label-Efficient Deep Learning in Medical Image Analysis: Challenges and Future Directions.](http://arxiv.org/abs/2303.12484) | 近年来深度学习在医学图像分析中取得了最先进的性能，但这种方法的标记代价大，标记不足。因此发展了高效标记深度学习方法，充分利用未标记的和弱标记的数据。该综述总结了这方面的最新进展。 |

# 详细

[^1]: 消费者物联网流量的调查：安全与隐私

    A Survey on Consumer IoT Traffic: Security and Privacy

    [https://arxiv.org/abs/2403.16149](https://arxiv.org/abs/2403.16149)

    本调查针对消费者物联网（CIoT）流量分析从安全和隐私的角度出发，总结了CIoT流量分析的新特征、最新进展和挑战，认为通过流量分析可以揭示CIoT领域中的安全和隐私问题。

    

    在过去几年里，消费者物联网（CIoT）已经进入了公众生活。尽管CIoT提高了人们日常生活的便利性，但也带来了新的安全和隐私问题。我们尝试通过流量分析这一安全领域中的流行方法，找出研究人员可以从流量分析中了解CIoT安全和隐私方面的内容。本调查从安全和隐私角度探讨了CIoT流量分析中的新特征、CIoT流量分析的最新进展以及尚未解决的挑战。我们从2018年1月至2023年12月收集了310篇与CIoT流量分析有关的安全和隐私角度的论文，总结了识别了CIoT新特征的CIoT流量分析过程。然后，我们根据五个应用目标详细介绍了现有的研究工作：设备指纹识别、用户活动推断、恶意行为检测、隐私泄露以及通信模式识别。

    arXiv:2403.16149v1 Announce Type: cross  Abstract: For the past few years, the Consumer Internet of Things (CIoT) has entered public lives. While CIoT has improved the convenience of people's daily lives, it has also brought new security and privacy concerns. In this survey, we try to figure out what researchers can learn about the security and privacy of CIoT by traffic analysis, a popular method in the security community. From the security and privacy perspective, this survey seeks out the new characteristics in CIoT traffic analysis, the state-of-the-art progress in CIoT traffic analysis, and the challenges yet to be solved. We collected 310 papers from January 2018 to December 2023 related to CIoT traffic analysis from the security and privacy perspective and summarized the process of CIoT traffic analysis in which the new characteristics of CIoT are identified. Then, we detail existing works based on five application goals: device fingerprinting, user activity inference, malicious
    
[^2]: 跨国界评估公平度量标准：来自人类感知的视角

    Evaluating Fairness Metrics Across Borders from Human Perceptions

    [https://arxiv.org/abs/2403.16101](https://arxiv.org/abs/2403.16101)

    该研究通过国际调查评估了不同国家对其决策情景中各种公平度量标准的适用性。

    

    哪些公平度量标准适用于您的场景？即使结果符合已建立的公平度量标准，也可能存在关于公平感知的不一致情况。已进行了多项调查，评估了公平度量标准与人们对公平的感知。然而，这些调查范围有限，仅包括单个国家中数百名参与者。在这项研究中，我们进行了一项国际调查，以评估各种公平度量标准在决策场景中的适用性。我们分别从中国、法国、日本和美国的每个国家收集了1,000名参与者的回应，总计得到了4,000个回应，以分析公平度量标准的偏好。我们的调查包括三个不同场景，配备了四种公平度量标准，每个参与者在每种情况下选择其喜好的公平度量标准。该研究探讨了

    arXiv:2403.16101v1 Announce Type: new  Abstract: Which fairness metrics are appropriately applicable in your contexts? There may be instances of discordance regarding the perception of fairness, even when the outcomes comply with established fairness metrics. Several surveys have been conducted to evaluate fairness metrics with human perceptions of fairness. However, these surveys were limited in scope, including only a few hundred participants within a single country. In this study, we conduct an international survey to evaluate the appropriateness of various fairness metrics in decision-making scenarios. We collected responses from 1,000 participants in each of China, France, Japan, and the United States, amassing a total of 4,000 responses, to analyze the preferences of fairness metrics. Our survey consists of three distinct scenarios paired with four fairness metrics, and each participant answers their preference for the fairness metric in each case. This investigation explores the
    
[^3]: DyCE：用于深度学习压缩和扩展的动态可配置退出

    DyCE: Dynamic Configurable Exiting for Deep Learning Compression and Scaling

    [https://arxiv.org/abs/2403.01695](https://arxiv.org/abs/2403.01695)

    介绍了DyCE，一个动态可配置的提前退出框架，将设计考虑从彼此和基础模型解耦

    

    现代深度学习（DL）模型需要在资源受限环境中有效部署时，使用缩放和压缩技术。大多数现有技术，如修剪和量化，通常是静态的。另一方面，动态压缩方法（如提前退出）通过识别输入样本的困难程度并根据需要分配计算来降低复杂性。动态方法，尽管具有更高的灵活性和与静态方法共存的潜力，但在实现上面临重大挑战，因为动态部分的任何变化都会影响后续过程。此外，大多数当前的动态压缩设计都是单片的，与基础模型紧密集成，从而使其难以适应新颖基础模型。本文介绍了DyCE，一种动态可配置的提前退出框架，从而使设计考虑相互解耦以及与基础模型

    arXiv:2403.01695v1 Announce Type: cross  Abstract: Modern deep learning (DL) models necessitate the employment of scaling and compression techniques for effective deployment in resource-constrained environments. Most existing techniques, such as pruning and quantization are generally static. On the other hand, dynamic compression methods, such as early exits, reduce complexity by recognizing the difficulty of input samples and allocating computation as needed. Dynamic methods, despite their superior flexibility and potential for co-existing with static methods, pose significant challenges in terms of implementation due to any changes in dynamic parts will influence subsequent processes. Moreover, most current dynamic compression designs are monolithic and tightly integrated with base models, thereby complicating the adaptation to novel base models. This paper introduces DyCE, an dynamic configurable early-exit framework that decouples design considerations from each other and from the 
    
[^4]: 连接NTK和NNGP：神经网络学习动力学在核区域的统一理论框架

    Connecting NTK and NNGP: A Unified Theoretical Framework for Neural Network Learning Dynamics in the Kernel Regime. (arXiv:2309.04522v1 [cs.LG])

    [http://arxiv.org/abs/2309.04522](http://arxiv.org/abs/2309.04522)

    本文提出了一个马尔可夫近似学习模型，统一了神经切向核（NTK）和神经网络高斯过程（NNGP）核，用于描述无限宽度深层网络的学习动力学。

    

    人工神经网络近年来在机器学习领域取得了革命性的进展，但其学习过程缺乏一个完整的理论框架。对于无限宽度网络，已经取得了重大进展。在这个范式中，使用了两种不同的理论框架来描述网络的输出：一种基于神经切向核（NTK）的框架，假设了线性化的梯度下降动力学；另一种是基于神经网络高斯过程（NNGP）核的贝叶斯框架。然而，这两种框架之间的关系一直不明确。本文通过一个马尔可夫近似学习模型，统一了这两种不同的理论，用于描述随机初始化的无限宽度深层网络的学习动力学。我们推导出了在学习过程中和学习后的网络输入-输出函数的精确分析表达式，并引入了一个新的时间相关的神经动态核（NDK），这个核可以同时产生NTK和NNGP。

    Artificial neural networks have revolutionized machine learning in recent years, but a complete theoretical framework for their learning process is still lacking. Substantial progress has been made for infinitely wide networks. In this regime, two disparate theoretical frameworks have been used, in which the network's output is described using kernels: one framework is based on the Neural Tangent Kernel (NTK) which assumes linearized gradient descent dynamics, while the Neural Network Gaussian Process (NNGP) kernel assumes a Bayesian framework. However, the relation between these two frameworks has remained elusive. This work unifies these two distinct theories using a Markov proximal learning model for learning dynamics in an ensemble of randomly initialized infinitely wide deep networks. We derive an exact analytical expression for the network input-output function during and after learning, and introduce a new time-dependent Neural Dynamical Kernel (NDK) from which both NTK and NNGP
    
[^5]: 一种基于深度学习技术的肺癌诊断自动化端到端框架，用于检测和分类肺部结节

    An automated end-to-end deep learning-based framework for lung cancer diagnosis by detecting and classifying the lung nodules. (arXiv:2305.00046v1 [eess.IV])

    [http://arxiv.org/abs/2305.00046](http://arxiv.org/abs/2305.00046)

    本文提出了一种基于深度学习的智能诊断框架，针对低资源环境实现早期检测和分类肺部结节，并在公共数据集上取得了较好的表现。

    

    肺癌是全球癌症相关死亡的主要原因，在低资源环境中早期诊断对于改善患者疗效至关重要。本研究的目的是提出一种基于深度学习技术的自动化端到端框架，用于早期检测和分类肺部结节，特别是针对低资源环境。该框架由三个阶段组成：使用改进的3D Res-U-Net进行肺分割、使用YOLO-v5进行结节检测、使用基于Vision Transformer的架构进行分类。我们在开放的数据集LUNA16上对该框架进行了评估。所提出的框架的性能是使用各领域的评估指标进行衡量的。该框架在肺部分割dice系数上达到了98.82％，同时检测肺结节的平均准确度为0.76 mAP。

    Lung cancer is a leading cause of cancer-related deaths worldwide, and early detection is crucial for improving patient outcomes. Nevertheless, early diagnosis of cancer is a major challenge, particularly in low-resource settings where access to medical resources and trained radiologists is limited. The objective of this study is to propose an automated end-to-end deep learning-based framework for the early detection and classification of lung nodules, specifically for low-resource settings. The proposed framework consists of three stages: lung segmentation using a modified 3D U-Net named 3D Res-U-Net, nodule detection using YOLO-v5, and classification with a Vision Transformer-based architecture. We evaluated the proposed framework on a publicly available dataset, LUNA16. The proposed framework's performance was measured using the respective domain's evaluation matrices. The proposed framework achieved a 98.82% lung segmentation dice score while detecting the lung nodule with 0.76 mAP
    
[^6]: 医学图像分析中高效标记深度学习的挑战与未来方向

    Label-Efficient Deep Learning in Medical Image Analysis: Challenges and Future Directions. (arXiv:2303.12484v1 [cs.CV])

    [http://arxiv.org/abs/2303.12484](http://arxiv.org/abs/2303.12484)

    近年来深度学习在医学图像分析中取得了最先进的性能，但这种方法的标记代价大，标记不足。因此发展了高效标记深度学习方法，充分利用未标记的和弱标记的数据。该综述总结了这方面的最新进展。

    

    深度学习近年来得到了迅速发展，并在广泛应用中取得了最先进的性能。但是，训练模型通常需要收集大量标记数据，这需要昂贵耗时。特别是在医学图像分析（MIA）领域，数据有限，标签很难获得。因此，人们开发了高效标记深度学习方法，充分利用标记数据以及非标记和弱标记数据的丰富性。在本调查中，我们对近300篇论文进行了广泛调查，以全面概述最新进展的高效标记学习策略在MIA中的研究现状。我们首先介绍高效标记学习的背景，并将不同方案的方法归类。接下来，我们通过每种方案详细研究了目前最先进的方法。具体而言，我们进行了深入调查，覆盖了不仅是标准策略，还包括使用后处理和集合方法等方法。

    Deep learning has seen rapid growth in recent years and achieved state-of-the-art performance in a wide range of applications. However, training models typically requires expensive and time-consuming collection of large quantities of labeled data. This is particularly true within the scope of medical imaging analysis (MIA), where data are limited and labels are expensive to be acquired. Thus, label-efficient deep learning methods are developed to make comprehensive use of the labeled data as well as the abundance of unlabeled and weak-labeled data. In this survey, we extensively investigated over 300 recent papers to provide a comprehensive overview of recent progress on label-efficient learning strategies in MIA. We first present the background of label-efficient learning and categorize the approaches into different schemes. Next, we examine the current state-of-the-art methods in detail through each scheme. Specifically, we provide an in-depth investigation, covering not only canonic
    

