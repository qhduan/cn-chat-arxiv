# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Conjugate-Gradient-like Based Adaptive Moment Estimation Optimization Algorithm for Deep Learning](https://arxiv.org/abs/2404.01714) | 提出一种基于共轭梯度样式的新优化算法CG-like-Adam，用于深度学习，并在收敛分析和数值实验中展示了其优越性 |
| [^2] | [Comparing Bad Apples to Good Oranges: Aligning Large Language Models via Joint Preference Optimization](https://arxiv.org/abs/2404.00530) | 本研究提出了一种新的偏好获取方法，通过DOVE协议对指令-响应对的联合概率进行优化，以对齐大型语言模型。 |
| [^3] | [Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators](https://arxiv.org/abs/2403.16950) | 在大型语言模型评估中，通过引入成对偏好搜索方法PAIRS，成功解决了LLMs与人类判断不一致的问题，并取得了优于直接打分的最先进性能。 |
| [^4] | [A Survey on Consumer IoT Traffic: Security and Privacy](https://arxiv.org/abs/2403.16149) | 本调查针对消费者物联网（CIoT）流量分析从安全和隐私的角度出发，总结了CIoT流量分析的新特征、最新进展和挑战，认为通过流量分析可以揭示CIoT领域中的安全和隐私问题。 |
| [^5] | [Lemur: Log Parsing with Entropy Sampling and Chain-of-Thought Merging](https://arxiv.org/abs/2402.18205) | Lemur提出了一种先进的日志解析框架，采用熵抽样和思维链合并，解决了日志解析中存在的人工规则依赖和语义信息忽略等问题。 |
| [^6] | [Latent Neural PDE Solver: a reduced-order modelling framework for partial differential equations](https://arxiv.org/abs/2402.17853) | 提出了一种名为潜在神经PDE求解器（LNS）的框架，通过在潜在空间学习系统动态并使用较粗糙的离散化，可以大大简化神经PDE求解器的训练过程，降低计算成本。 |
| [^7] | [NeuralDiffuser: Controllable fMRI Reconstruction with Primary Visual Feature Guided Diffusion](https://arxiv.org/abs/2402.13809) | NeuralDiffuser引入主视觉特征引导，扩展了LDM方法的自下而上过程，以实现忠实的语义和细节。 |
| [^8] | [Forecasting high-impact research topics via machine learning on evolving knowledge graphs](https://arxiv.org/abs/2402.08640) | 通过机器学习预测未发布研究想法的影响力，我们使用一个由超过2100万篇科学论文构建的演化知识图谱，结合论文内容和历史引用的信息，高准确度预测未来的演化网络动态和新的研究方向的影响力。 |
| [^9] | [DEFormer: DCT-driven Enhancement Transformer for Low-light Image and Dark Vision.](http://arxiv.org/abs/2309.06941) | 该论文提出了一种新的DCT驱动增强Transformer（DEFormer），可以在低光图像中恢复丢失的细节，通过引入频率作为新的线索，通过可学习的频率分支（LFB）和基于曲率的频率增强（CFE）来实现。此外，还提出了交叉域融合（CDF）来减少领域之间的差异，DEFormer还可以作为暗部检测的预处理，有效提高了性能。 |
| [^10] | [Unlocking the Diagnostic Potential of ECG through Knowledge Transfer from Cardiac MRI.](http://arxiv.org/abs/2308.05764) | 该论文提出了一种通过从心脏MRI中的知识转移解锁心电图的诊断潜力的方法。通过将CMR图像中的领域特定信息转移到ECG嵌入中，该方法实现了仅根据ECG数据进行全面的心脏筛查，并能预测心血管疾病的个体风险和确定心脏表型。 |
| [^11] | [Offline Prioritized Experience Replay.](http://arxiv.org/abs/2306.05412) | 本文提出了离线优先经验重放（OPER）方法来解决离线强化学习中的分布偏移问题。通过设计一类优先级函数来对高回报的转换进行优先处理，从而改善行为策略，并在此改进的策略约束下优化离线强化学习算法的解决方案。对于离线强化学习，OPER方法是一种有效的解决方案。 |
| [^12] | [On Sequential Bayesian Inference for Continual Learning.](http://arxiv.org/abs/2301.01828) | 这篇论文研究了顺序贝叶斯推断在连续学习中的应用。研究表明，尽管使用了真实后验，这种方法仍无法防止神经网络中的灾难性遗忘。同时，模型误差和任务数据不平衡也会导致连续学习性能的下降。 |
| [^13] | [Differentiable Inductive Logic Programming in High-Dimensional Space.](http://arxiv.org/abs/2208.06652) | 本研究提出了一种在高维空间中进行可微归纳逻辑编程的扩展方法，通过大规模谓词发明来充分利用高维梯度下降的效能，以学习超出现有神经符号ILP系统能力的任务的解决方案。 |

# 详细

[^1]: 基于共轭梯度的自适应矩估计优化算法用于深度学习

    Conjugate-Gradient-like Based Adaptive Moment Estimation Optimization Algorithm for Deep Learning

    [https://arxiv.org/abs/2404.01714](https://arxiv.org/abs/2404.01714)

    提出一种基于共轭梯度样式的新优化算法CG-like-Adam，用于深度学习，并在收敛分析和数值实验中展示了其优越性

    

    训练深度神经网络是一项具有挑战性的任务。为加快培训速度并增强深度神经网络的性能，我们将传统的共轭梯度修正为共轭梯度样式，并将其并入通用Adam中，因此提出了一种名为CG-like-Adam的新优化算法用于深度学习。具体而言，通用Adam的一阶和二阶矩估计均由共轭梯度样式替换。收敛分析处理了一阶矩估计的指数移动平均系数为常数且一阶矩估计无偏的情况。数值实验显示了基于CIFAR10/100数据集的所提算法的优越性。

    arXiv:2404.01714v1 Announce Type: cross  Abstract: Training deep neural networks is a challenging task. In order to speed up training and enhance the performance of deep neural networks, we rectify the vanilla conjugate gradient as conjugate-gradient-like and incorporate it into the generic Adam, and thus propose a new optimization algorithm named CG-like-Adam for deep learning. Specifically, both the first-order and the second-order moment estimation of generic Adam are replaced by the conjugate-gradient-like. Convergence analysis handles the cases where the exponential moving average coefficient of the first-order moment estimation is constant and the first-order moment estimation is unbiased. Numerical experiments show the superiority of the proposed algorithm based on the CIFAR10/100 dataset.
    
[^2]: 将坏苹果与好橘子进行比较：通过联合优化偏好对齐大型语言模型

    Comparing Bad Apples to Good Oranges: Aligning Large Language Models via Joint Preference Optimization

    [https://arxiv.org/abs/2404.00530](https://arxiv.org/abs/2404.00530)

    本研究提出了一种新的偏好获取方法，通过DOVE协议对指令-响应对的联合概率进行优化，以对齐大型语言模型。

    

    一种常见的对齐大型语言模型（LLMs）的技术依赖于通过比较在固定上下文中条件生成的多个生成的人类偏好。然而，当这些生成放置在相同的上下文中时，这仅利用了成对比较。然而，这种条件排名通常无法捕获人类偏好的复杂和多维方面。在这项工作中，我们重新审视偏好获取的传统范式，并提出了一个基于在指令-响应对上联合引发偏好的新轴。虽然先前的偏好优化是针对条件排名协议（例如，DPO）设计的，但我们提出的偏好获取协议引入了DOVE，这是一个新的偏好优化目标，通过提升所选指令-响应对的联合概率来降低所拒绝指令-响应对的概率。

    arXiv:2404.00530v1 Announce Type: cross  Abstract: A common technique for aligning large language models (LLMs) relies on acquiring human preferences by comparing multiple generations conditioned on a fixed context. This only leverages the pairwise comparisons when the generations are placed in an identical context. However, such conditional rankings often fail to capture the complex and multidimensional aspects of human preferences. In this work, we revisit the traditional paradigm of preference acquisition and propose a new axis that is based on eliciting preferences jointly over the instruction-response pairs. While prior preference optimizations are designed for conditional ranking protocols (e.g., DPO), our proposed preference acquisition protocol introduces DOVE, a new preference optimization objective that upweights the joint probability of the chosen instruction-response pair over the rejected instruction-response pair. Interestingly, we find that the LLM trained with joint ins
    
[^3]: 与人类判断相一致：大型语言模型评估中成对偏好的作用

    Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators

    [https://arxiv.org/abs/2403.16950](https://arxiv.org/abs/2403.16950)

    在大型语言模型评估中，通过引入成对偏好搜索方法PAIRS，成功解决了LLMs与人类判断不一致的问题，并取得了优于直接打分的最先进性能。

    

    大型语言模型（LLMs）作为自动评估器在评估生成的自然语言质量方面表现出有希望的能力。然而，LLMs在评估中仍存在偏见，常常难以生成与人类评估一致的连贯评估。在这项工作中，我们首先对LLM评估器与人类判断之间的不一致进行系统研究，揭示现有旨在减轻偏见的校准方法不足以有效将LLM评估器对齐。受到RLHF中对偏好数据的使用的启发，我们将评估形式化为一个排序问题，并引入Pairwise-preference Search（PAIRS），这是一种以LLMs进行成对比较并有效对候选文本进行排序的基于不确定性引导的搜索方法。PAIRS在代表性评估任务上实现了最先进的性能，并且显示出比直接打分有显著改进。

    arXiv:2403.16950v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated promising capabilities as automatic evaluators in assessing the quality of generated natural language. However, LLMs still exhibit biases in evaluation and often struggle to generate coherent evaluations that align with human assessments. In this work, we first conduct a systematic study of the misalignment between LLM evaluators and human judgement, revealing that existing calibration methods aimed at mitigating biases are insufficient for effectively aligning LLM evaluators. Inspired by the use of preference data in RLHF, we formulate the evaluation as a ranking problem and introduce Pairwise-preference Search (PAIRS), an uncertainty-guided search method that employs LLMs to conduct pairwise comparisons and efficiently ranks candidate texts. PAIRS achieves state-of-the-art performance on representative evaluation tasks and demonstrates significant improvements over direct scoring. Furthe
    
[^4]: 消费者物联网流量的调查：安全与隐私

    A Survey on Consumer IoT Traffic: Security and Privacy

    [https://arxiv.org/abs/2403.16149](https://arxiv.org/abs/2403.16149)

    本调查针对消费者物联网（CIoT）流量分析从安全和隐私的角度出发，总结了CIoT流量分析的新特征、最新进展和挑战，认为通过流量分析可以揭示CIoT领域中的安全和隐私问题。

    

    在过去几年里，消费者物联网（CIoT）已经进入了公众生活。尽管CIoT提高了人们日常生活的便利性，但也带来了新的安全和隐私问题。我们尝试通过流量分析这一安全领域中的流行方法，找出研究人员可以从流量分析中了解CIoT安全和隐私方面的内容。本调查从安全和隐私角度探讨了CIoT流量分析中的新特征、CIoT流量分析的最新进展以及尚未解决的挑战。我们从2018年1月至2023年12月收集了310篇与CIoT流量分析有关的安全和隐私角度的论文，总结了识别了CIoT新特征的CIoT流量分析过程。然后，我们根据五个应用目标详细介绍了现有的研究工作：设备指纹识别、用户活动推断、恶意行为检测、隐私泄露以及通信模式识别。

    arXiv:2403.16149v1 Announce Type: cross  Abstract: For the past few years, the Consumer Internet of Things (CIoT) has entered public lives. While CIoT has improved the convenience of people's daily lives, it has also brought new security and privacy concerns. In this survey, we try to figure out what researchers can learn about the security and privacy of CIoT by traffic analysis, a popular method in the security community. From the security and privacy perspective, this survey seeks out the new characteristics in CIoT traffic analysis, the state-of-the-art progress in CIoT traffic analysis, and the challenges yet to be solved. We collected 310 papers from January 2018 to December 2023 related to CIoT traffic analysis from the security and privacy perspective and summarized the process of CIoT traffic analysis in which the new characteristics of CIoT are identified. Then, we detail existing works based on five application goals: device fingerprinting, user activity inference, malicious
    
[^5]: Lemur: 使用熵抽样和思维链合并进行日志解析

    Lemur: Log Parsing with Entropy Sampling and Chain-of-Thought Merging

    [https://arxiv.org/abs/2402.18205](https://arxiv.org/abs/2402.18205)

    Lemur提出了一种先进的日志解析框架，采用熵抽样和思维链合并，解决了日志解析中存在的人工规则依赖和语义信息忽略等问题。

    

    大型软件系统产生的日志对监视系统行为至关重要。先进的日志分析有助于检测、报警和诊断系统故障。日志解析是日志分析自动化的关键阶段，它涉及将原始日志消息转换为结构化模板。现有的日志解析器由于依赖于人工制定的规则而无法识别正确的模板。此外，这些方法侧重于统计特征，而忽略了日志消息中的语义信息。为了解决这些挑战，我们提出了一种先进的日志解析框架，采用熵抽样和思维链合并（Lemur）。具体而言，为了摆脱繁琐的手动规则，我们提出了一种受信息熵启发的新型抽样方法，能够有效地对典型日志进行聚类。此外，为了增强日志模板的合并，我们设计了一种思维链方法。

    arXiv:2402.18205v1 Announce Type: cross  Abstract: Logs produced by extensive software systems are integral to monitoring system behaviors. Advanced log analysis facilitates the detection, alerting, and diagnosis of system faults. Log parsing, which entails transforming raw log messages into structured templates, constitutes a critical phase in the automation of log analytics. Existing log parsers fail to identify the correct templates due to reliance on human-made rules. Besides, These methods focus on statistical features while ignoring semantic information in log messages. To address these challenges, we introduce a cutting-edge \textbf{L}og parsing framework with \textbf{E}ntropy sampling and Chain-of-Thought \textbf{M}erging (Lemur). Specifically, to discard the tedious manual rules. We propose a novel sampling method inspired by information entropy, which efficiently clusters typical logs. Furthermore, to enhance the merging of log templates, we design a chain-of-thought method f
    
[^6]: 潜在神经PDE求解器：用于偏微分方程的降阶建模框架

    Latent Neural PDE Solver: a reduced-order modelling framework for partial differential equations

    [https://arxiv.org/abs/2402.17853](https://arxiv.org/abs/2402.17853)

    提出了一种名为潜在神经PDE求解器（LNS）的框架，通过在潜在空间学习系统动态并使用较粗糙的离散化，可以大大简化神经PDE求解器的训练过程，降低计算成本。

    

    神经网络在加速由偏微分方程（PDEs）控制的系统的数值模拟方面显示出了巨大潜力。与许多现有的在高维离散化场上操作的神经网络代理不同，我们提议在潜在空间学习系统的动态，使用更粗糙的离散化。在我们提出的框架 - 潜在神经PDE求解器（LNS）中，首先训练一个非线性自动编码器，将系统的全阶表示投影到网格减少的空间中，接着训练一个时间模型来预测这个网格减少的空间中的未来状态。这种降阶过程通过大大减少伴随细粒度离散化的计算成本，简化了时间模型的训练。我们研究了提出的框架以及几种其他流行的神经PDE求解器在各种类型的系统上的能力，包括单相和多相流体系统。

    arXiv:2402.17853v1 Announce Type: cross  Abstract: Neural networks have shown promising potential in accelerating the numerical simulation of systems governed by partial differential equations (PDEs). Different from many existing neural network surrogates operating on high-dimensional discretized fields, we propose to learn the dynamics of the system in the latent space with much coarser discretizations. In our proposed framework - Latent Neural PDE Solver (LNS), a non-linear autoencoder is first trained to project the full-order representation of the system onto the mesh-reduced space, then a temporal model is trained to predict the future state in this mesh-reduced space. This reduction process simplifies the training of the temporal model by greatly reducing the computational cost accompanying a fine discretization. We study the capability of the proposed framework and several other popular neural PDE solvers on various types of systems including single-phase and multi-phase flows a
    
[^7]: NeuralDiffuser：具有主视觉特征引导扩散的可控fMRI重建

    NeuralDiffuser: Controllable fMRI Reconstruction with Primary Visual Feature Guided Diffusion

    [https://arxiv.org/abs/2402.13809](https://arxiv.org/abs/2402.13809)

    NeuralDiffuser引入主视觉特征引导，扩展了LDM方法的自下而上过程，以实现忠实的语义和细节。

    

    基于潜在扩散模型(LDM)从功能性磁共振成像(fMRI)中重建视觉刺激，为大脑提供了细粒度的检索。一个挑战在于重建细节的连贯对齐（如结构、背景、纹理、颜色等）。此外，即使在相同条件下，LDM也会生成不同的图像结果。因此，我们首先揭示了基于LDM的神经科学视角，即基于来自海量图像的预训练知识进行自上而下的创建，但缺乏基于细节驱动的自下而上感知，导致细节不忠实。我们提出了NeuralDiffuser，引入主视觉特征引导，以渐变形式提供细节线索，扩展了LDM方法的自下而上过程，以实现忠实的语义和细节。我们还开发了一种新颖的引导策略，以确保重复重建的一致性，而不是随机性。

    arXiv:2402.13809v1 Announce Type: cross  Abstract: Reconstructing visual stimuli from functional Magnetic Resonance Imaging (fMRI) based on Latent Diffusion Models (LDM) provides a fine-grained retrieval of the brain. A challenge persists in reconstructing a cohesive alignment of details (such as structure, background, texture, color, etc.). Moreover, LDMs would generate different image results even under the same conditions. For these, we first uncover the neuroscientific perspective of LDM-based methods that is top-down creation based on pre-trained knowledge from massive images but lack of detail-driven bottom-up perception resulting in unfaithful details. We propose NeuralDiffuser which introduces primary visual feature guidance to provide detail cues in the form of gradients, extending the bottom-up process for LDM-based methods to achieve faithful semantics and details. We also developed a novel guidance strategy to ensure the consistency of repeated reconstructions rather than a
    
[^8]: 通过机器学习在不断演化的知识图谱上预测高影响力的研究主题

    Forecasting high-impact research topics via machine learning on evolving knowledge graphs

    [https://arxiv.org/abs/2402.08640](https://arxiv.org/abs/2402.08640)

    通过机器学习预测未发布研究想法的影响力，我们使用一个由超过2100万篇科学论文构建的演化知识图谱，结合论文内容和历史引用的信息，高准确度预测未来的演化网络动态和新的研究方向的影响力。

    

    科学出版物的指数增长对人类研究者构成了严峻挑战。它迫使研究者将注意力集中在更狭窄的子领域上，使得发现其他领域的新颖且有影响力的研究想法和合作变得困难。虽然有办法预测科学论文未来的引用次数，但通常需要等到研究完成并且论文写成后才能进行评估，这样就错过了想法构思的早期阶段。在本文中，我们展示了如何预测从未被研究者发布的想法的影响力。为此，我们开发了一个大型的演化知识图谱，其中包含超过2100万篇科学论文。它结合了从论文内容中创建的语义网络和从历史引用中创建的影响网络。利用机器学习，我们可以高准确度地预测演化网络的动态情况，从而预测新的研究方向的影响力。我们预期这种能力将有助于研究者发现具有高影响力的研究主题。

    The exponential growth in scientific publications poses a severe challenge for human researchers. It forces attention to more narrow sub-fields, which makes it challenging to discover new impactful research ideas and collaborations outside one's own field. While there are ways to predict a scientific paper's future citation counts, they need the research to be finished and the paper written, usually assessing impact long after the idea was conceived. Here we show how to predict the impact of onsets of ideas that have never been published by researchers. For that, we developed a large evolving knowledge graph built from more than 21 million scientific papers. It combines a semantic network created from the content of the papers and an impact network created from the historic citations of papers. Using machine learning, we can predict the dynamic of the evolving network into the future with high accuracy, and thereby the impact of new research directions. We envision that the ability to 
    
[^9]: DEFormer: 用于低光图像和暗视觉的DCT驱动增强Transformer

    DEFormer: DCT-driven Enhancement Transformer for Low-light Image and Dark Vision. (arXiv:2309.06941v1 [cs.CV])

    [http://arxiv.org/abs/2309.06941](http://arxiv.org/abs/2309.06941)

    该论文提出了一种新的DCT驱动增强Transformer（DEFormer），可以在低光图像中恢复丢失的细节，通过引入频率作为新的线索，通过可学习的频率分支（LFB）和基于曲率的频率增强（CFE）来实现。此外，还提出了交叉域融合（CDF）来减少领域之间的差异，DEFormer还可以作为暗部检测的预处理，有效提高了性能。

    

    低光图像增强的目标是恢复图像的颜色和细节，对于自动驾驶中的高级视觉任务非常重要。然而，仅依靠RGB领域很难恢复暗区域的丢失细节。本文将频率作为网络的新线索，并提出了一种新颖的DCT驱动增强Transformer（DEFormer）。首先，我们提出了一个可学习的频率分支（LFB）用于频率增强，包括DCT处理和基于曲率的频率增强（CFE）。CFE计算每个通道的曲率以表示不同频率带的细节丰富度，然后我们将频率特征划分为更丰富纹理的频率带。此外，我们提出了一个交叉域融合（CDF）来减少RGB领域和频率领域之间的差异。我们还将DEFormer作为暗部检测的预处理，DEFormer有效提高了性能。

    The goal of low-light image enhancement is to restore the color and details of the image and is of great significance for high-level visual tasks in autonomous driving. However, it is difficult to restore the lost details in the dark area by relying only on the RGB domain. In this paper we introduce frequency as a new clue into the network and propose a novel DCT-driven enhancement transformer (DEFormer). First, we propose a learnable frequency branch (LFB) for frequency enhancement contains DCT processing and curvature-based frequency enhancement (CFE). CFE calculates the curvature of each channel to represent the detail richness of different frequency bands, then we divides the frequency features, which focuses on frequency bands with richer textures. In addition, we propose a cross domain fusion (CDF) for reducing the differences between the RGB domain and the frequency domain. We also adopt DEFormer as a preprocessing in dark detection, DEFormer effectively improves the performance
    
[^10]: 通过从心脏MRI中的知识转移解锁心电图的诊断潜力

    Unlocking the Diagnostic Potential of ECG through Knowledge Transfer from Cardiac MRI. (arXiv:2308.05764v1 [eess.SP])

    [http://arxiv.org/abs/2308.05764](http://arxiv.org/abs/2308.05764)

    该论文提出了一种通过从心脏MRI中的知识转移解锁心电图的诊断潜力的方法。通过将CMR图像中的领域特定信息转移到ECG嵌入中，该方法实现了仅根据ECG数据进行全面的心脏筛查，并能预测心血管疾病的个体风险和确定心脏表型。

    

    心电图 (ECG) 是一种广泛可用的诊断工具，可以快速和经济高效地评估心血管健康状况。然而，在心血管疾病的诊断中，通常更喜欢使用昂贵的心脏磁共振 (CMR) 成像进行更详细的检查。虽然 CMR 成像可以提供详细的心脏解剖可视化，但由于长时间扫描和高昂的费用，它并不广泛可用。为了解决这个问题，我们提出了一种第一种自监督对比方法，将CMR图像中的领域特定信息转移到ECG嵌入中。我们的方法将多模态对比学习与屏蔽数据建模相结合，实现了仅根据ECG数据进行全面的心脏筛查。在使用来自40044名UK Biobank受试者的数据进行的广泛实验证明了我们方法的实用性和可推广性。我们预测了各种心血管疾病的个体风险，并仅根据ECG数据确定了不同的心脏表型。

    The electrocardiogram (ECG) is a widely available diagnostic tool that allows for a cost-effective and fast assessment of the cardiovascular health. However, more detailed examination with expensive cardiac magnetic resonance (CMR) imaging is often preferred for the diagnosis of cardiovascular diseases. While providing detailed visualization of the cardiac anatomy, CMR imaging is not widely available due to long scan times and high costs. To address this issue, we propose the first self-supervised contrastive approach that transfers domain-specific information from CMR images to ECG embeddings. Our approach combines multimodal contrastive learning with masked data modeling to enable holistic cardiac screening solely from ECG data. In extensive experiments using data from 40,044 UK Biobank subjects, we demonstrate the utility and generalizability of our method. We predict the subject-specific risk of various cardiovascular diseases and determine distinct cardiac phenotypes solely from E
    
[^11]: 离线优先经验重放

    Offline Prioritized Experience Replay. (arXiv:2306.05412v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.05412](http://arxiv.org/abs/2306.05412)

    本文提出了离线优先经验重放（OPER）方法来解决离线强化学习中的分布偏移问题。通过设计一类优先级函数来对高回报的转换进行优先处理，从而改善行为策略，并在此改进的策略约束下优化离线强化学习算法的解决方案。对于离线强化学习，OPER方法是一种有效的解决方案。

    

    离线强化学习面临着分布偏移问题。为了解决这个问题，现有的工作主要集中在设计学习策略和行为策略之间的复杂约束。然而，这些约束通过均匀采样等方式被应用到表现良好和表现差的行动上，这可能会对学习策略产生负面影响。为了缓解这个问题，我们提出了离线优先经验重放（OPER），其中包括一类优先级函数，用于将高回报的转换置于更频繁的访问中。通过理论分析，我们证明了这类优先级函数能够引起行为策略的改善，当策略约束到这个改进的策略上时，离线强化学习算法很可能得到更好的解决方案。我们开发了两种实用策略来获得基于拟合值网络的优先权重（OPER-A）或者u

    Offline reinforcement learning (RL) is challenged by the distributional shift problem. To address this problem, existing works mainly focus on designing sophisticated policy constraints between the learned policy and the behavior policy. However, these constraints are applied equally to well-performing and inferior actions through uniform sampling, which might negatively affect the learned policy. To alleviate this issue, we propose Offline Prioritized Experience Replay (OPER), featuring a class of priority functions designed to prioritize highly-rewarding transitions, making them more frequently visited during training. Through theoretical analysis, we show that this class of priority functions induce an improved behavior policy, and when constrained to this improved policy, a policy-constrained offline RL algorithm is likely to yield a better solution. We develop two practical strategies to obtain priority weights by estimating advantages based on a fitted value network (OPER-A) or u
    
[^12]: 关于连续学习的顺序贝叶斯推断

    On Sequential Bayesian Inference for Continual Learning. (arXiv:2301.01828v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.01828](http://arxiv.org/abs/2301.01828)

    这篇论文研究了顺序贝叶斯推断在连续学习中的应用。研究表明，尽管使用了真实后验，这种方法仍无法防止神经网络中的灾难性遗忘。同时，模型误差和任务数据不平衡也会导致连续学习性能的下降。

    

    顺序贝叶斯推断可用于连续学习，以防止过去任务的灾难性遗忘，并在学习新任务时提供信息丰富的先验。我们重新审视顺序贝叶斯推断，并测试是否有访问真实后验的保证可以防止贝叶斯神经网络中的灾难性遗忘。为了做到这一点，我们使用哈密顿蒙特卡洛执行顺序贝叶斯推断。我们通过对哈密顿蒙特卡洛样本拟合密度估计器，将后验传播为新任务的先验。我们发现这种方法无法防止灾难性遗忘，证明了在神经网络中执行顺序贝叶斯推断的困难。然后，我们研究了顺序贝叶斯推断和连续学习的简单分析示例，并强调了模型错误说明问题，即使进行了准确的推断，也可能导致次优的连续学习性能。此外，我们讨论了任务数据不平衡可能导致遗忘的问题。

    Sequential Bayesian inference can be used for continual learning to prevent catastrophic forgetting of past tasks and provide an informative prior when learning new tasks. We revisit sequential Bayesian inference and test whether having access to the true posterior is guaranteed to prevent catastrophic forgetting in Bayesian neural networks. To do this we perform sequential Bayesian inference using Hamiltonian Monte Carlo. We propagate the posterior as a prior for new tasks by fitting a density estimator on Hamiltonian Monte Carlo samples. We find that this approach fails to prevent catastrophic forgetting demonstrating the difficulty in performing sequential Bayesian inference in neural networks. From there we study simple analytical examples of sequential Bayesian inference and CL and highlight the issue of model misspecification which can lead to sub-optimal continual learning performance despite exact inference. Furthermore, we discuss how task data imbalances can cause forgetting.
    
[^13]: 高维空间中可微归纳逻辑编程

    Differentiable Inductive Logic Programming in High-Dimensional Space. (arXiv:2208.06652v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2208.06652](http://arxiv.org/abs/2208.06652)

    本研究提出了一种在高维空间中进行可微归纳逻辑编程的扩展方法，通过大规模谓词发明来充分利用高维梯度下降的效能，以学习超出现有神经符号ILP系统能力的任务的解决方案。

    

    通过符号归纳逻辑编程（ILP）合成大型逻辑程序通常需要中间定义。然而，使用内涵谓词杂乱地占据假设空间通常会降低性能。相反，梯度下降提供了在这些高维空间中寻找解决方案的有效方法。到目前为止，神经符号ILP方法并没有充分利用这一点。我们提出扩展{\delta}ILP方法，以进行大规模谓词发明的归纳合成，从而允许我们利用高维梯度下降的效能。我们展示了大规模谓词发明通过梯度下降受益于可微归纳合成，并允许我们学习超出现有神经符号ILP系统能力的任务的解决方案。此外，我们在不指定解决方案的精确结构的语言偏差的情况下实现了这些结果。

    Synthesizing large logic programs through symbolic Inductive Logic Programming (ILP) typically requires intermediate definitions. However, cluttering the hypothesis space with intensional predicates typically degrades performance. In contrast, gradient descent provides an efficient way to find solutions within such high- dimensional spaces. Neuro-symbolic ILP approaches have not fully exploited this so far. We propose extending the {\delta}ILP approach to inductive synthesis with large-scale predicate invention, thus allowing us to exploit the efficacy of high-dimensional gradient descent. We show that large-scale predicate invention benefits differentiable inductive synthesis through gradient descent and allows one to learn solutions for tasks beyond the capabilities of existing neuro-symbolic ILP systems. Furthermore, we achieve these results without specifying the precise structure of the solution within the language bias.
    

