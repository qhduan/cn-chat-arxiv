# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cheating Suffix: Targeted Attack to Text-To-Image Diffusion Models with Multi-Modal Priors](https://rss.arxiv.org/abs/2402.01369) | 本文提出了一种名为MMP-Attack的有针对性攻击方法，通过整合文本和图像特征，该方法能够有效地攻击商业文本到图像模型，并且具有更高的普适性和可转移性。 |
| [^2] | [Parallelized Midpoint Randomization for Langevin Monte Carlo](https://arxiv.org/abs/2402.14434) | 探索在能够进行梯度平行评估的框架中的抽样问题，提出了并行化的随机中点方法，并通过新技术导出了对抽样和目标密度之间Wasserstein距离的上界，量化了并行处理单元带来的运行时改进。 |
| [^3] | [Evaluating Membership Inference Attacks and Defenses in Federated Learning](https://arxiv.org/abs/2402.06289) | 这篇论文评估了联邦学习中成员推断攻击和防御的情况。评估揭示了两个重要发现：多时序的模型信息有助于提高攻击的有效性，多空间的模型信息有助于提高攻击的效果。这篇论文还评估了两种防御机制的效用和隐私权衡。 |
| [^4] | [Scientific Language Modeling: A Quantitative Review of Large Language Models in Molecular Science](https://arxiv.org/abs/2402.04119) | 本研究提出了一种科学语言建模（SLM）的新方法，通过大型语言模型（LLM）来解决分子建模和设计中的挑战。通过多模态基准和实验评估，我们提供了关于模型与数据模态匹配的量化信息，同时也揭示了模型的知识学习偏好。 |
| [^5] | [Kernel PCA for Out-of-Distribution Detection](https://arxiv.org/abs/2402.02949) | 本论文提出了使用核PCA进行外分布检测的方法，通过在主成分子空间中引入非线性映射，实现了对内分布和外分布数据的有效区分。 |
| [^6] | [Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data.](http://arxiv.org/abs/2401.15113) | 本研究提出了一种使用深度学习和开放地球观测数据进行全球冰川制图的方法，通过新的模型和策略，在多种地形和传感器上实现了较高的准确性。通过添加合成孔径雷达数据，并报告冰川范围的校准置信度，提高了预测的可靠性和可解释性。 |
| [^7] | [Gradual Domain Adaptation: Theory and Algorithms.](http://arxiv.org/abs/2310.13852) | 本文研究了渐进域自适应中的渐进自训练算法，提出了一个改进的泛化界限，并指出了中间域在源域和目标域之间均匀放置的重要性。 |
| [^8] | [HPCR: Holistic Proxy-based Contrastive Replay for Online Continual Learning.](http://arxiv.org/abs/2309.15038) | HPCR是一种用于在线连续学习的新方法，该方法综合了基于代理和对比损失的重放方式。通过在对比损失中使用锚点-代理对替换锚点-样本对，HPCR能够减轻遗忘现象，并有效学习更细粒度的语义信息。实验证明，HPCR在多个任务上实现了最先进的性能。 |
| [^9] | [Ano-SuPs: Multi-size anomaly detection for manufactured products by identifying suspected patches.](http://arxiv.org/abs/2309.11120) | Ano-SuPs是一种通过识别可疑区块来进行制造产品的多尺度异常检测的两阶段策略方法。它可以解决图像背景复杂性和异常模式的挑战，并具有较高的准确性和鲁棒性。 |
| [^10] | [Provably Efficient Learning in Partially Observable Contextual Bandit.](http://arxiv.org/abs/2308.03572) | 本文研究了在部分可观察情境轮盘赌中的转移学习问题，提出了一种通过优化问题识别行为和奖励因果效应的方法，并利用因果约束来改进轮盘赌算法。 |
| [^11] | [Communication-Efficient Split Learning via Adaptive Feature-Wise Compression.](http://arxiv.org/abs/2307.10805) | 该论文提出了一个名为SplitFC的通信高效的分割学习框架，通过两种自适应压缩策略来减少中间特征和梯度向量的通信开销，这些策略分别是自适应特征逐渐掉落和自适应特征逐渐量化。 |
| [^12] | [Experimental Security Analysis of DNN-based Adaptive Cruise Control under Context-Aware Perception Attacks.](http://arxiv.org/abs/2307.08939) | 这项研究评估了基于深度神经网络的自适应巡航控制系统在隐蔽感知攻击下的安全性，并提出了一种上下文感知策略和基于优化的图像扰动生成方法。 |

# 详细

[^1]: 使用多模式先验的有针对性攻击文本到图像扩散模型

    Cheating Suffix: Targeted Attack to Text-To-Image Diffusion Models with Multi-Modal Priors

    [https://rss.arxiv.org/abs/2402.01369](https://rss.arxiv.org/abs/2402.01369)

    本文提出了一种名为MMP-Attack的有针对性攻击方法，通过整合文本和图像特征，该方法能够有效地攻击商业文本到图像模型，并且具有更高的普适性和可转移性。

    

    扩散模型已广泛应用于各种图像生成任务中，展现了图像和文本模态之间的卓越联系。然而，它们面临着被恶意利用的挑战，通过在原始提示后附加特定后缀来生成有害或敏感图像。现有作品主要关注使用单模态信息进行攻击，未能利用多模态特征，导致性能不尽如人意。在本工作中，我们提出了一种名为MMP-Attack的有针对性攻击方法，它将多模态先验（MMP）即文本和图像特征进行整合。具体而言，MMP-Attack的目标是在图像内容中添加目标对象的同时，同时移除原始对象。与现有作品相比，MMP-Attack具有更高的普适性和可转移性，在攻击商业文本到图像（T2I）模型（如DALL-E 3）方面表现出明显优势。据我们所知，这标志着当前最佳的技术水平。

    Diffusion models have been widely deployed in various image generation tasks, demonstrating an extraordinary connection between image and text modalities. However, they face challenges of being maliciously exploited to generate harmful or sensitive images by appending a specific suffix to the original prompt. Existing works mainly focus on using single-modal information to conduct attacks, which fails to utilize multi-modal features and results in less than satisfactory performance. Integrating multi-modal priors (MMP), i.e. both text and image features, we propose a targeted attack method named MMP-Attack in this work. Specifically, the goal of MMP-Attack is to add a target object into the image content while simultaneously removing the original object. The MMP-Attack shows a notable advantage over existing works with superior universality and transferability, which can effectively attack commercial text-to-image (T2I) models such as DALL-E 3. To the best of our knowledge, this marks 
    
[^2]: 并行中点随机化的 Langevin Monte Carlo

    Parallelized Midpoint Randomization for Langevin Monte Carlo

    [https://arxiv.org/abs/2402.14434](https://arxiv.org/abs/2402.14434)

    探索在能够进行梯度平行评估的框架中的抽样问题，提出了并行化的随机中点方法，并通过新技术导出了对抽样和目标密度之间Wasserstein距离的上界，量化了并行处理单元带来的运行时改进。

    

    我们探讨了在可以进行梯度的平行评估的框架中的抽样问题。我们的研究重点放在由平滑和强log-凹密度表征的目标分布上。我们重新审视了并行化的随机中点方法，并运用最近开发用于分析其纯顺序版本的证明技术。利用这些技术，我们得出了抽样和目标密度之间的Wasserstein距离的上界。这些界限量化了通过利用并行处理单元所实现的运行时改进，这可能是相当可观的。

    arXiv:2402.14434v1 Announce Type: cross  Abstract: We explore the sampling problem within the framework where parallel evaluations of the gradient of the log-density are feasible. Our investigation focuses on target distributions characterized by smooth and strongly log-concave densities. We revisit the parallelized randomized midpoint method and employ proof techniques recently developed for analyzing its purely sequential version. Leveraging these techniques, we derive upper bounds on the Wasserstein distance between the sampling and target densities. These bounds quantify the runtime improvement achieved by utilizing parallel processing units, which can be considerable.
    
[^3]: 在联邦学习中评估成员推断攻击和防御

    Evaluating Membership Inference Attacks and Defenses in Federated Learning

    [https://arxiv.org/abs/2402.06289](https://arxiv.org/abs/2402.06289)

    这篇论文评估了联邦学习中成员推断攻击和防御的情况。评估揭示了两个重要发现：多时序的模型信息有助于提高攻击的有效性，多空间的模型信息有助于提高攻击的效果。这篇论文还评估了两种防御机制的效用和隐私权衡。

    

    成员推断攻击(MIAs)对于隐私保护的威胁在联邦学习中日益增长。半诚实的攻击者，例如服务器，可以根据观察到的模型信息确定一个特定样本是否属于目标客户端。本文对现有的MIAs和相应的防御策略进行了评估。我们对MIAs的评估揭示了两个重要发现。首先，结合多个通信轮次的模型信息(多时序)相比于利用单个时期的模型信息提高了MIAs的整体有效性。其次，在非目标客户端(Multi-spatial)中融入模型显著提高了MIAs的效果，特别是当客户端的数据是同质的时候。这凸显了在MIAs中考虑时序和空间模型信息的重要性。接下来，我们通过隐私-效用权衡评估了两种类型的防御机制对MIAs的有效性。

    Membership Inference Attacks (MIAs) pose a growing threat to privacy preservation in federated learning. The semi-honest attacker, e.g., the server, may determine whether a particular sample belongs to a target client according to the observed model information. This paper conducts an evaluation of existing MIAs and corresponding defense strategies. Our evaluation on MIAs reveals two important findings about the trend of MIAs. Firstly, combining model information from multiple communication rounds (Multi-temporal) enhances the overall effectiveness of MIAs compared to utilizing model information from a single epoch. Secondly, incorporating models from non-target clients (Multi-spatial) significantly improves the effectiveness of MIAs, particularly when the clients' data is homogeneous. This highlights the importance of considering the temporal and spatial model information in MIAs. Next, we assess the effectiveness via privacy-utility tradeoff for two type defense mechanisms against MI
    
[^4]: 科学语言建模：分子科学中大型语言模型的定量评估

    Scientific Language Modeling: A Quantitative Review of Large Language Models in Molecular Science

    [https://arxiv.org/abs/2402.04119](https://arxiv.org/abs/2402.04119)

    本研究提出了一种科学语言建模（SLM）的新方法，通过大型语言模型（LLM）来解决分子建模和设计中的挑战。通过多模态基准和实验评估，我们提供了关于模型与数据模态匹配的量化信息，同时也揭示了模型的知识学习偏好。

    

    高效的分子建模和设计对于发现和探索新型分子至关重要，而深度学习方法的引入在这个领域中产生了革命性的影响。特别是，大型语言模型（LLM）以自然语言处理（NLP）的视角为科学问题提供了一种新的方法，引入了一种名为科学语言建模（SLM）的研究范式。然而，仍然存在两个关键问题：如何量化模型与数据模态之间的匹配以及如何识别模型的知识学习偏好。为了解决这些挑战，我们提出了一个名为ChEBI-20-MM的多模态基准，并进行了1263个实验，评估了模型与数据模态的兼容性和知识获取能力。通过模态转移概率矩阵，我们为任务提供了最合适的模态的见解。此外，我们还引入了一种统计可解释的方法，通过本地化来发现特定上下文的知识映射。

    Efficient molecular modeling and design are crucial for the discovery and exploration of novel molecules, and the incorporation of deep learning methods has revolutionized this field. In particular, large language models (LLMs) offer a fresh approach to tackle scientific problems from a natural language processing (NLP) perspective, introducing a research paradigm called scientific language modeling (SLM). However, two key issues remain: how to quantify the match between model and data modalities and how to identify the knowledge-learning preferences of models. To address these challenges, we propose a multi-modal benchmark, named ChEBI-20-MM, and perform 1263 experiments to assess the model's compatibility with data modalities and knowledge acquisition. Through the modal transition probability matrix, we provide insights into the most suitable modalities for tasks. Furthermore, we introduce a statistically interpretable approach to discover context-specific knowledge mapping by locali
    
[^5]: 外分布检测的核PCA

    Kernel PCA for Out-of-Distribution Detection

    [https://arxiv.org/abs/2402.02949](https://arxiv.org/abs/2402.02949)

    本论文提出了使用核PCA进行外分布检测的方法，通过在主成分子空间中引入非线性映射，实现了对内分布和外分布数据的有效区分。

    

    外分布（OoD）检测对于深度神经网络（DNN）的可靠性至关重要。现有的研究表明，直接应用于DNN特征的主成分分析（PCA）在检测来自内分布（InD）数据的OoD数据方面不足够。PCA的失败表明，仅通过在线性子空间中进行简单处理无法很好地将OoD和InD中的网络特征分离开来，而可以通过适当的非线性映射来解决。在这项工作中，我们利用核PCA（KPCA）框架进行OoD检测，寻找OoD和InD特征以显著不同的模式分配的子空间。我们设计了两种特征映射，在KPCA中引入非线性内核，以促进在主成分张成的子空间中InD和OoD数据之间的可分性。然后，通过在这种子空间中的重构误差，可以有效地得到$\mathcal{O}(1)$时间复杂度的检测结果。

    Out-of-Distribution (OoD) detection is vital for the reliability of Deep Neural Networks (DNNs). Existing works have shown the insufficiency of Principal Component Analysis (PCA) straightforwardly applied on the features of DNNs in detecting OoD data from In-Distribution (InD) data. The failure of PCA suggests that the network features residing in OoD and InD are not well separated by simply proceeding in a linear subspace, which instead can be resolved through proper nonlinear mappings. In this work, we leverage the framework of Kernel PCA (KPCA) for OoD detection, seeking subspaces where OoD and InD features are allocated with significantly different patterns. We devise two feature mappings that induce non-linear kernels in KPCA to advocate the separability between InD and OoD data in the subspace spanned by the principal components. Given any test sample, the reconstruction error in such subspace is then used to efficiently obtain the detection result with $\mathcal{O}(1)$ time comp
    
[^6]: 使用深度学习和开放地球观测数据实现全球冰川制图

    Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data. (arXiv:2401.15113v1 [cs.CV])

    [http://arxiv.org/abs/2401.15113](http://arxiv.org/abs/2401.15113)

    本研究提出了一种使用深度学习和开放地球观测数据进行全球冰川制图的方法，通过新的模型和策略，在多种地形和传感器上实现了较高的准确性。通过添加合成孔径雷达数据，并报告冰川范围的校准置信度，提高了预测的可靠性和可解释性。

    

    准确的全球冰川制图对于理解气候变化的影响至关重要。这个过程受到冰川多样性、难以分类的碎石和大数据处理的挑战。本文提出了Glacier-VisionTransformer-U-Net (GlaViTU)，一个卷积-Transformer深度学习模型，并提出了五种利用开放卫星影像进行多时相全球冰川制图的策略。空间、时间和跨传感器的泛化性能评估表明，我们的最佳策略在大多数情况下实现了IoU（交并比）> 0.85，并且在以冰雪为主的地区增加到了> 0.90，而在高山亚洲等碎石丰富的区域则降至> 0.75。此外，添加合成孔径雷达数据，即回波和干涉相干度，可以提高所有可用地区的准确性。报告冰川范围的校准置信度使预测更可靠和可解释。我们还发布了一个基准数据集。

    Accurate global glacier mapping is critical for understanding climate change impacts. It is challenged by glacier diversity, difficult-to-classify debris and big data processing. Here we propose Glacier-VisionTransformer-U-Net (GlaViTU), a convolutional-transformer deep learning model, and five strategies for multitemporal global-scale glacier mapping using open satellite imagery. Assessing the spatial, temporal and cross-sensor generalisation shows that our best strategy achieves intersection over union >0.85 on previously unobserved images in most cases, which drops to >0.75 for debris-rich areas such as High-Mountain Asia and increases to >0.90 for regions dominated by clean ice. Additionally, adding synthetic aperture radar data, namely, backscatter and interferometric coherence, increases the accuracy in all regions where available. The calibrated confidence for glacier extents is reported making the predictions more reliable and interpretable. We also release a benchmark dataset 
    
[^7]: 渐进域自适应：理论与算法

    Gradual Domain Adaptation: Theory and Algorithms. (arXiv:2310.13852v1 [cs.LG])

    [http://arxiv.org/abs/2310.13852](http://arxiv.org/abs/2310.13852)

    本文研究了渐进域自适应中的渐进自训练算法，提出了一个改进的泛化界限，并指出了中间域在源域和目标域之间均匀放置的重要性。

    

    无监督域自适应（UDA）是将模型从有标记的源域适应到无标记的目标域的一种一次性方法。尽管被广泛应用，但当源域和目标域之间的分布偏移较大时，UDA面临巨大挑战。渐进域自适应（GDA）通过使用中间域逐渐从源域适应到目标域来缓解这个限制。在这项工作中，我们首先从理论上分析了一种常见的GDA算法——渐进自训练，并提供了与Kumar等人（2020）相比显著改进的泛化界限。我们的理论分析得出一个有趣的观点：为了最小化目标域上的泛化误差，中间域的顺序应该均匀地放置在源域和目标域之间的Wasserstein测地线上。这个观点在中间域缺失或稀缺的情况下尤其有用，而这在现实世界的应用中经常出现。

    Unsupervised domain adaptation (UDA) adapts a model from a labeled source domain to an unlabeled target domain in a one-off way. Though widely applied, UDA faces a great challenge whenever the distribution shift between the source and the target is large. Gradual domain adaptation (GDA) mitigates this limitation by using intermediate domains to gradually adapt from the source to the target domain. In this work, we first theoretically analyze gradual self-training, a popular GDA algorithm, and provide a significantly improved generalization bound compared with Kumar et al. (2020). Our theoretical analysis leads to an interesting insight: to minimize the generalization error on the target domain, the sequence of intermediate domains should be placed uniformly along the Wasserstein geodesic between the source and target domains. The insight is particularly useful under the situation where intermediate domains are missing or scarce, which is often the case in real-world applications. Based
    
[^8]: HPCR: 基于代理的综合对比重放用于在线连续学习

    HPCR: Holistic Proxy-based Contrastive Replay for Online Continual Learning. (arXiv:2309.15038v1 [cs.LG])

    [http://arxiv.org/abs/2309.15038](http://arxiv.org/abs/2309.15038)

    HPCR是一种用于在线连续学习的新方法，该方法综合了基于代理和对比损失的重放方式。通过在对比损失中使用锚点-代理对替换锚点-样本对，HPCR能够减轻遗忘现象，并有效学习更细粒度的语义信息。实验证明，HPCR在多个任务上实现了最先进的性能。

    

    在线连续学习（OCL）旨在通过一次在线数据流传递持续学习新数据。然而，它通常会面临灾难性遗忘问题。现有的基于重放的方法通过以代理为基础或对比为基础的重放方式有效地缓解了这个问题。在本文中，我们对这两种重放方式进行了全面分析，并发现它们可以相互补充。受到这一发现的启发，我们提出了一种新颖的基于重放的方法称为代理对比重放（PCR），它将对比损失中的锚点-样本对替换为锚点-代理对，以减轻遗忘现象。基于PCR，我们进一步开发了一种更高级的方法，称为综合代理对比重放（HPCR），它由三个组件组成。对比组件在PCR的基础上条件性地将锚点-样本对纳入其中，通过大型训练批次学习更细粒度的语义信息。第二个组件是重放组件，它在样本选择上采用了多样性策略，以确保代理数据与当前任务具有更高的关联性。第三个组件是正则化组件，通过缩小样本空间，促进学习模型对任务特定特征的更好表示。实验证明，HPCR方法在多个在线连续学习任务上实现了最先进的性能。

    Online continual learning (OCL) aims to continuously learn new data from a single pass over the online data stream. It generally suffers from the catastrophic forgetting issue. Existing replay-based methods effectively alleviate this issue by replaying part of old data in a proxy-based or contrastive-based replay manner. In this paper, we conduct a comprehensive analysis of these two replay manners and find they can be complementary. Inspired by this finding, we propose a novel replay-based method called proxy-based contrastive replay (PCR), which replaces anchor-to-sample pairs with anchor-to-proxy pairs in the contrastive-based loss to alleviate the phenomenon of forgetting. Based on PCR, we further develop a more advanced method named holistic proxy-based contrastive replay (HPCR), which consists of three components. The contrastive component conditionally incorporates anchor-to-sample pairs to PCR, learning more fine-grained semantic information with a large training batch. The sec
    
[^9]: Ano-SuPs: 通过识别可疑的区块进行制造产品的多尺度异常检测

    Ano-SuPs: Multi-size anomaly detection for manufactured products by identifying suspected patches. (arXiv:2309.11120v1 [stat.ML])

    [http://arxiv.org/abs/2309.11120](http://arxiv.org/abs/2309.11120)

    Ano-SuPs是一种通过识别可疑区块来进行制造产品的多尺度异常检测的两阶段策略方法。它可以解决图像背景复杂性和异常模式的挑战，并具有较高的准确性和鲁棒性。

    

    基于图像的系统因其提供丰富的制造状态信息、低实施成本和高采集速度而受到欢迎。然而，图像背景的复杂性和各种异常模式给现有的矩阵分解方法带来了新的挑战，这些方法不足以满足建模需求。此外，异常的不确定性可能导致异常的污染问题，使得设计的模型和方法对外部干扰非常敏感。为了解决这些挑战，我们提出了一种通过识别可疑区块（Ano-SuPs）来检测异常的两阶段策略异常检测方法。具体来说，我们提出了通过两次重建输入图像来检测带有异常的区块的方法：第一步是通过去除那些可疑区块来获得一组正常区块，第二步是使用这些正常区块来优化对带有异常区块的识别。我们通过实验证明了这种方法的效果。

    Image-based systems have gained popularity owing to their capacity to provide rich manufacturing status information, low implementation costs and high acquisition rates. However, the complexity of the image background and various anomaly patterns pose new challenges to existing matrix decomposition methods, which are inadequate for modeling requirements. Moreover, the uncertainty of the anomaly can cause anomaly contamination problems, making the designed model and method highly susceptible to external disturbances. To address these challenges, we propose a two-stage strategy anomaly detection method that detects anomalies by identifying suspected patches (Ano-SuPs). Specifically, we propose to detect the patches with anomalies by reconstructing the input image twice: the first step is to obtain a set of normal patches by removing those suspected patches, and the second step is to use those normal patches to refine the identification of the patches with anomalies. To demonstrate its ef
    
[^10]: 在部分可观察情境轮盘赌中的可证效率学习

    Provably Efficient Learning in Partially Observable Contextual Bandit. (arXiv:2308.03572v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2308.03572](http://arxiv.org/abs/2308.03572)

    本文研究了在部分可观察情境轮盘赌中的转移学习问题，提出了一种通过优化问题识别行为和奖励因果效应的方法，并利用因果约束来改进轮盘赌算法。

    

    本文研究了在部分可观察情境轮盘赌中的转移学习问题，其中代理人仅有来自其他代理人的有限知识，并且对隐藏的混淆因素只有部分信息。我们将该问题转化为通过优化问题来识别或部分识别行为和奖励之间的因果效应。为了解决这些优化问题，我们将未知分布的原始功能约束离散化为线性约束，并通过顺序解线性规划来采样兼容的因果模型，以考虑估计误差得到因果约束。我们的采样算法为适当的采样分布提供了理想的收敛结果。然后，我们展示了如何将因果约束应用于改进经典的轮盘赌算法，并以行动集和函数空间规模为参考改变了遗憾值。值得注意的是，在允许我们处理一般情境分布的函数逼近任务中

    In this paper, we investigate transfer learning in partially observable contextual bandits, where agents have limited knowledge from other agents and partial information about hidden confounders. We first convert the problem to identifying or partially identifying causal effects between actions and rewards through optimization problems. To solve these optimization problems, we discretize the original functional constraints of unknown distributions into linear constraints, and sample compatible causal models via sequentially solving linear programmings to obtain causal bounds with the consideration of estimation error. Our sampling algorithms provide desirable convergence results for suitable sampling distributions. We then show how causal bounds can be applied to improving classical bandit algorithms and affect the regrets with respect to the size of action sets and function spaces. Notably, in the task with function approximation which allows us to handle general context distributions
    
[^11]: 通过自适应特征逐渐压缩实现高效的分割学习

    Communication-Efficient Split Learning via Adaptive Feature-Wise Compression. (arXiv:2307.10805v1 [cs.DC])

    [http://arxiv.org/abs/2307.10805](http://arxiv.org/abs/2307.10805)

    该论文提出了一个名为SplitFC的通信高效的分割学习框架，通过两种自适应压缩策略来减少中间特征和梯度向量的通信开销，这些策略分别是自适应特征逐渐掉落和自适应特征逐渐量化。

    

    本文提出了一种名为SplitFC的新颖的通信高效的分割学习（SL）框架，它减少了在SL培训过程中传输中间特征和梯度向量所需的通信开销。SplitFC的关键思想是利用矩阵的列所展示的不同的离散程度。SplitFC整合了两种压缩策略：（i）自适应特征逐渐掉落和（ii）自适应特征逐渐量化。在第一种策略中，中间特征向量根据这些向量的标准偏差确定自适应掉落概率进行掉落。然后，由于链式规则，与被丢弃的特征向量相关联的中间梯度向量也会被丢弃。在第二种策略中，非丢弃的中间特征和梯度向量使用基于向量范围确定的自适应量化级别进行量化。为了尽量减小量化误差，最优量化是。

    This paper proposes a novel communication-efficient split learning (SL) framework, named SplitFC, which reduces the communication overhead required for transmitting intermediate feature and gradient vectors during the SL training process. The key idea of SplitFC is to leverage different dispersion degrees exhibited in the columns of the matrices. SplitFC incorporates two compression strategies: (i) adaptive feature-wise dropout and (ii) adaptive feature-wise quantization. In the first strategy, the intermediate feature vectors are dropped with adaptive dropout probabilities determined based on the standard deviation of these vectors. Then, by the chain rule, the intermediate gradient vectors associated with the dropped feature vectors are also dropped. In the second strategy, the non-dropped intermediate feature and gradient vectors are quantized using adaptive quantization levels determined based on the ranges of the vectors. To minimize the quantization error, the optimal quantizatio
    
[^12]: 基于深度神经网络的自适应巡航控制在上下文感知攻击下的安全性实验分析

    Experimental Security Analysis of DNN-based Adaptive Cruise Control under Context-Aware Perception Attacks. (arXiv:2307.08939v1 [cs.CR])

    [http://arxiv.org/abs/2307.08939](http://arxiv.org/abs/2307.08939)

    这项研究评估了基于深度神经网络的自适应巡航控制系统在隐蔽感知攻击下的安全性，并提出了一种上下文感知策略和基于优化的图像扰动生成方法。

    

    自适应巡航控制（ACC）是一种广泛应用的驾驶员辅助功能，用于保持期望速度和与前方车辆的安全距离。本文评估基于深度神经网络（DNN）的ACC系统在隐蔽感知攻击下的安全性，该攻击会对摄像机数据进行有针对性的扰动，以导致前方碰撞事故。我们提出了一种基于知识和数据驱动的方法，设计了一种上下文感知策略，用于选择触发攻击最关键的时间点，并采用了一种新颖的基于优化的方法，在运行时生成适应性图像扰动。我们使用实际驾驶数据集和逼真的仿真平台评估了所提出攻击的有效性，该仿真平台使用了来自生产ACC系统的控制软件和物理世界驾驶模拟器，并考虑了驾驶员的干预以及自动紧急制动（AEB）和前向碰撞警示（FCW）等安全功能。

    Adaptive Cruise Control (ACC) is a widely used driver assistance feature for maintaining desired speed and safe distance to the leading vehicles. This paper evaluates the security of the deep neural network (DNN) based ACC systems under stealthy perception attacks that strategically inject perturbations into camera data to cause forward collisions. We present a combined knowledge-and-data-driven approach to design a context-aware strategy for the selection of the most critical times for triggering the attacks and a novel optimization-based method for the adaptive generation of image perturbations at run-time. We evaluate the effectiveness of the proposed attack using an actual driving dataset and a realistic simulation platform with the control software from a production ACC system and a physical-world driving simulator while considering interventions by the driver and safety features such as Automatic Emergency Braking (AEB) and Forward Collision Warning (FCW). Experimental results sh
    

