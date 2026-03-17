# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Survey of Computerized Adaptive Testing: A Machine Learning Perspective](https://arxiv.org/abs/2404.00712) | 本文以机器学习视角综述了计算机自适应测试（CAT），重点解析其测试问题选择算法和如何优化认知诊断模型、题库构建和测试控制。 |
| [^2] | [PARAMANU-AYN: An Efficient Novel Generative and Instruction-tuned Language Model for Indian Legal Case Documents](https://arxiv.org/abs/2403.13681) | PARAMANU-AYN是一种基于印度法律案例文件的高效生成式语言模型，采用自回归解码器进行预训练，并经过面向指令的微调，在各种法律任务上取得了良好表现。 |
| [^3] | [ConjNorm: Tractable Density Estimation for Out-of-Distribution Detection](https://arxiv.org/abs/2402.17888) | 提出了一种新颖的理论框架，基于Bregman散度，通过引入共轭约束，提出了一种\textsc{ConjNorm}方法，以在给定数据集中搜索最佳规范系数$p$来重新构想密度函数设计。 |
| [^4] | [A Review of Deep Learning Methods for Photoplethysmography Data.](http://arxiv.org/abs/2401.12783) | 本综述系统地回顾了自2017年至2023年期间应用深度学习模型处理光电容积法数据的论文。研究发现，深度学习在个人健康管理和其他应用中具有显著成果。根据任务的不同，这些论文被分为医学相关和非医学相关两大类别，医学相关又细分为七个子组，包括血压分析... |
| [^5] | [Deformation-Invariant Neural Network and Its Applications in Distorted Image Restoration and Analysis.](http://arxiv.org/abs/2310.02641) | 本文提出了一种弹性不变神经网络（DINN）用于处理受到几何形变影响的图像的图像处理任务。DINN通过融入拟保形变换网络（QCTN）来输出一致的潜在特征，使得具有相同原始对象或场景的几何形变图像能够更接近自然或良好图像分布。 |

# 详细

[^1]: 计算机自适应测试综述：机器学习视角

    Survey of Computerized Adaptive Testing: A Machine Learning Perspective

    [https://arxiv.org/abs/2404.00712](https://arxiv.org/abs/2404.00712)

    本文以机器学习视角综述了计算机自适应测试（CAT），重点解析其测试问题选择算法和如何优化认知诊断模型、题库构建和测试控制。

    

    计算机自适应测试（CAT）提供了一种高效、量身定制的评估考生熟练程度的方法，通过根据他们的表现动态调整测试问题。CAT广泛应用于教育、医疗、体育和社会学等多个领域，彻底改变了测试实践。然而，随着大规模测试的增加复杂性，CAT已经融合了机器学习技术。本文旨在提供一个以机器学习为重点的CAT综述，从新的角度解读这种自适应测试方法。通过研究CAT适应性核心的测试问题选择算法，我们揭示了其功能。此外，我们探讨了认知诊断模型、题库构建和CAT中的测试控制，探索了机器学习如何优化这些组成部分。通过对当前情况的分析，

    arXiv:2404.00712v1 Announce Type: cross  Abstract: Computerized Adaptive Testing (CAT) provides an efficient and tailored method for assessing the proficiency of examinees, by dynamically adjusting test questions based on their performance. Widely adopted across diverse fields like education, healthcare, sports, and sociology, CAT has revolutionized testing practices. While traditional methods rely on psychometrics and statistics, the increasing complexity of large-scale testing has spurred the integration of machine learning techniques. This paper aims to provide a machine learning-focused survey on CAT, presenting a fresh perspective on this adaptive testing method. By examining the test question selection algorithm at the heart of CAT's adaptivity, we shed light on its functionality. Furthermore, we delve into cognitive diagnosis models, question bank construction, and test control within CAT, exploring how machine learning can optimize these components. Through an analysis of curre
    
[^2]: PARAMANU-AYN：一种有效的新型生成式、面向印度法律案例文件的语言模型

    PARAMANU-AYN: An Efficient Novel Generative and Instruction-tuned Language Model for Indian Legal Case Documents

    [https://arxiv.org/abs/2403.13681](https://arxiv.org/abs/2403.13681)

    PARAMANU-AYN是一种基于印度法律案例文件的高效生成式语言模型，采用自回归解码器进行预训练，并经过面向指令的微调，在各种法律任务上取得了良好表现。

    

    在这篇论文中，我们介绍了PARAMANU-AYN，这是一个仅基于印度最高法院案例文件、印度宪法和印度刑法的语言模型。这种新颖的基于自回归（AR）解码器的模型是从头开始在上下文大小为8192的情况下进行预训练的。我们在困惑度指标上评估了我们的预训练法律模型。我们还对一组包括各种法律任务（如法律推理、判决解释、法律条款生成、法律草拟、法律合同草拟、案件摘要、宪法问题回答等）的10,763条指令进行了针对性训练。我们还通过GPT-3.5-Turbo对面向指令的模型的提示响应进行了在10分制度上的清晰度、相关性、完整性和法律推理指标的评估。我们的模型可以在CPU上运行，并实现每秒42.46个令牌的CPU推理速度。

    arXiv:2403.13681v1 Announce Type: new  Abstract: In this paper, we present PARAMANU-AYN, a language model based exclusively on case documents of the Supreme Court of India, the Constitution of India, and the Indian Penal Code. The novel Auto Regressive (AR) decoder based model is pretrained from scratch at a context size of 8192. We evaluated our pretrained legal model on perplexity metrics. We also instruction-tuned our pretrained model on a set of 10,763 instructions covering various legal tasks such as legal reasoning, judgement explanation, legal clause generation, legal drafting, legal contract drafting, case summarization, constitutional question-answering, etc. We also evaluated the responses of prompts for instruction-tuned models by GPT-3.5-Turbo on clarity, relevance, completeness, and legal reasoning metrics in a scale of 10. Our model can be run on CPU and achieved 42.46 tokens/sec CPU inference speed. We found that our models, despite not being pretrained on legal books, v
    
[^3]: ConjNorm：用于异常分布检测的可处理密度估计

    ConjNorm: Tractable Density Estimation for Out-of-Distribution Detection

    [https://arxiv.org/abs/2402.17888](https://arxiv.org/abs/2402.17888)

    提出了一种新颖的理论框架，基于Bregman散度，通过引入共轭约束，提出了一种\textsc{ConjNorm}方法，以在给定数据集中搜索最佳规范系数$p$来重新构想密度函数设计。

    

    后续异常分布（OOD）检测在可靠机器学习中受到密切关注。许多工作致力于推导基于logits、距离或严格数据分布假设的评分函数，以识别得分低的OOD样本。然而，这些估计得分可能无法准确反映真实数据密度或施加不切实际的约束。为了在基于密度得分设计方面提供一个统一的视角，我们提出了一个以Bregman散度为基础的新颖理论框架，该框架将分布考虑扩展到涵盖一系列指数族分布。利用我们定理中揭示的共轭约束，我们引入了一种\textsc{ConjNorm}方法，将密度函数设计重新构想为针对给定数据集搜索最佳规范系数$p$的过程。鉴于归一化的计算挑战，我们设计了一种无偏和解析可追踪的方法

    arXiv:2402.17888v1 Announce Type: cross  Abstract: Post-hoc out-of-distribution (OOD) detection has garnered intensive attention in reliable machine learning. Many efforts have been dedicated to deriving score functions based on logits, distances, or rigorous data distribution assumptions to identify low-scoring OOD samples. Nevertheless, these estimate scores may fail to accurately reflect the true data density or impose impractical constraints. To provide a unified perspective on density-based score design, we propose a novel theoretical framework grounded in Bregman divergence, which extends distribution considerations to encompass an exponential family of distributions. Leveraging the conjugation constraint revealed in our theorem, we introduce a \textsc{ConjNorm} method, reframing density function design as a search for the optimal norm coefficient $p$ against the given dataset. In light of the computational challenges of normalization, we devise an unbiased and analytically tract
    
[^4]: 深度学习方法在光电容积法数据中的应用综述

    A Review of Deep Learning Methods for Photoplethysmography Data. (arXiv:2401.12783v1 [cs.AI])

    [http://arxiv.org/abs/2401.12783](http://arxiv.org/abs/2401.12783)

    本综述系统地回顾了自2017年至2023年期间应用深度学习模型处理光电容积法数据的论文。研究发现，深度学习在个人健康管理和其他应用中具有显著成果。根据任务的不同，这些论文被分为医学相关和非医学相关两大类别，医学相关又细分为七个子组，包括血压分析...

    

    光电容积法（PPG）是一种非常有前景的设备，因为它具有便携性、用户友好操作和非侵入性测量多种生理信息的能力。深度学习的最新进展，通过利用PPG信号，展示了在个人健康管理和其他多方面应用任务上取得了显著的成果。本综述系统地回顾了自2017年1月1日至2023年7月31日期间在Google学术、PubMed和Dimensions发表的应用深度学习模型处理PPG数据的论文。每篇论文从任务、模型和数据三个关键角度进行分析。最终提取了193篇论文，其中使用了不同的深度学习框架来处理PPG信号。根据这些论文所涉及的任务，我们将它们分为两大类别：医学相关和非医学相关。医学相关任务进一步分为七个子组，包括血压分析...

    Photoplethysmography (PPG) is a highly promising device due to its advantages in portability, user-friendly operation, and non-invasive capabilities to measure a wide range of physiological information. Recent advancements in deep learning have demonstrated remarkable outcomes by leveraging PPG signals for tasks related to personal health management and other multifaceted applications. In this review, we systematically reviewed papers that applied deep learning models to process PPG data between January 1st of 2017 and July 31st of 2023 from Google Scholar, PubMed and Dimensions. Each paper is analyzed from three key perspectives: tasks, models, and data. We finally extracted 193 papers where different deep learning frameworks were used to process PPG signals. Based on the tasks addressed in these papers, we categorized them into two major groups: medical-related, and non-medical-related. The medical-related tasks were further divided into seven subgroups, including blood pressure anal
    
[^5]: 弹性不变神经网络及其在变形图像恢复和分析中的应用

    Deformation-Invariant Neural Network and Its Applications in Distorted Image Restoration and Analysis. (arXiv:2310.02641v1 [cs.CV])

    [http://arxiv.org/abs/2310.02641](http://arxiv.org/abs/2310.02641)

    本文提出了一种弹性不变神经网络（DINN）用于处理受到几何形变影响的图像的图像处理任务。DINN通过融入拟保形变换网络（QCTN）来输出一致的潜在特征，使得具有相同原始对象或场景的几何形变图像能够更接近自然或良好图像分布。

    

    受到几何形变影响的图像对于目标识别等图像处理和计算机视觉任务来说是一个重要的挑战。基于深度学习的图像模型通常无法对几何形变图像给出准确的性能。本文中，我们提出了一种弹性不变神经网络（DINN），用于解决几何形变图像的图像处理任务。DINN为几何形变图像输出一致的潜在特征，这些图像具有相同的原始对象或场景。DINN的思想是将一个简单的组件，称为拟保形变换网络（QCTN），融入到其他现有的深度网络中进行图像处理任务。QCTN是一个深度神经网络，它输出一个拟保形映射，可以将几何形变的图像转换为更接近自然或良好图像分布的改进版本。它首先输出一个贝尔特拉密系数，用于衡量拟保形映射的效果。

    Images degraded by geometric distortions pose a significant challenge to imaging and computer vision tasks such as object recognition. Deep learning-based imaging models usually fail to give accurate performance for geometrically distorted images. In this paper, we propose the deformation-invariant neural network (DINN), a framework to address the problem of imaging tasks for geometrically distorted images. The DINN outputs consistent latent features for images that are geometrically distorted but represent the same underlying object or scene. The idea of DINN is to incorporate a simple component, called the quasiconformal transformer network (QCTN), into other existing deep networks for imaging tasks. The QCTN is a deep neural network that outputs a quasiconformal map, which can be used to transform a geometrically distorted image into an improved version that is closer to the distribution of natural or good images. It first outputs a Beltrami coefficient, which measures the quasiconf
    

