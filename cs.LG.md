# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks](https://arxiv.org/abs/2403.04783) | 提出了一种基于响应过滤的多Agent防御框架AutoDefense，可以有效提高LLMs对抗越狱攻击的鲁棒性，同时保持正常用户请求的性能。 |
| [^2] | [IGUANe: a 3D generalizable CycleGAN for multicenter harmonization of brain MR images](https://arxiv.org/abs/2402.03227) | IGUANe是一种三维通用CycleGAN模型，通过集成多个域的训练实现了脑MR图像的多中心协调，使其成为通用生成器。 |
| [^3] | [Toward Green and Human-Like Artificial Intelligence: A Complete Survey on Contemporary Few-Shot Learning Approaches](https://arxiv.org/abs/2402.03017) | 本文全面调查了少样本学习领域的最新进展，探讨了该方法在解决深度学习在实际应用中的限制方面的潜力和挑战。 |
| [^4] | [Embedding Hardware Approximations in Discrete Genetic-based Training for Printed MLPs](https://arxiv.org/abs/2402.02930) | 本文将硬件近似嵌入到印刷多层感知器的训练过程中，通过离散遗传算法实现了最大化硬件近似的效益，在5%的精度损失下，相比基线，实现了超过5倍的面积和功耗的减少，并且超过了最先进的近似方法。 |
| [^5] | [Equivariant Symmetry Breaking Sets](https://arxiv.org/abs/2402.02681) | 这里是中文总结出的一句话要点: 该论文提出了一种全等变的对称破缺框架，通过引入对称破缺集来破坏等变神经网络中的对称性。这种方法通用且适用于任何群的等变性。 |
| [^6] | [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417) | 本文提出了一种新的通用视觉骨干模型Vim，通过使用双向状态空间模型和位置嵌入来高效表示视觉数据，相比于传统的视觉转换器如DeiT，在各种视觉任务上取得了更好的性能，并且实现了显著提升。 |
| [^7] | [Bespoke Approximation of Multiplication-Accumulation and Activation Targeting Printed Multilayer Perceptrons](https://arxiv.org/abs/2312.17612) | 本研究提出了一种针对印刷电子技术中的限制的自动化框架，用于设计超低功耗的多层感知机（MLP）分类器。 |
| [^8] | [The cell signaling structure function.](http://arxiv.org/abs/2401.02501) | 该论文提出了一个新的方法，在活体细胞显微镜捕捉到的五维视频中寻找细胞信号动力学时空模式，并且不需要任何先验的预期模式动力学和训练数据。该方法基于细胞信号结构函数（SSF），通过测量细胞信号状态和周围细胞质之间的核糖体强度，与当前最先进的核糖体与细胞核比值相比有了显著改进。通过归一化压缩距离（NCD）来识别相似的模式。该方法能够将输入的SSF构图表示为低维嵌入中的点，最优地捕捉模式。 |
| [^9] | [ResBit: Residual Bit Vector for Categorical Values.](http://arxiv.org/abs/2309.17196) | 本论文提出了一种名为ResBit的残差位向量方法，用于解决在深度学习中表示离散数据维度增加和无法恢复原始类别值的问题。 |
| [^10] | [Autobidders with Budget and ROI Constraints: Efficiency, Regret, and Pacing Dynamics.](http://arxiv.org/abs/2301.13306) | 本文提出了一个基于梯度的学习算法，可以在多种拍卖方式下满足预算和ROI约束，并达到个体后悔逐渐减小；结果表明，当各自竞争时，期望资金流动至少达到最优分配的期望流动的一半。 |

# 详细

[^1]: AutoDefense: 多Agent LLM 防御对抗越狱攻击

    AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks

    [https://arxiv.org/abs/2403.04783](https://arxiv.org/abs/2403.04783)

    提出了一种基于响应过滤的多Agent防御框架AutoDefense，可以有效提高LLMs对抗越狱攻击的鲁棒性，同时保持正常用户请求的性能。

    

    尽管在道德对齐方面进行了广泛的预训练和微调以防止在用户请求时生成有害信息，但大型语言模型（LLMs）仍然容易受到越狱攻击。 本文提出了一种基于响应过滤的多Agent防御框架AutoDefense，用于从LLMs中过滤有害回复。 此框架为LLM代理分配不同角色，并利用它们共同完成防御任务。 任务的划分增强了LLMs的整体遵循指令能力，并使其他防御组件作为工具集成成为可能。 AutoDefense 可以适应各种规模和种类的开源LLMs作为代理。 通过对大量有害和安全提示进行广泛实验，我们验证了所提出的AutoDefense在提高对抗越狱攻击的鲁棒性的同时，保持了正常用户请求的性能。

    arXiv:2403.04783v1 Announce Type: cross  Abstract: Despite extensive pre-training and fine-tuning in moral alignment to prevent generating harmful information at user request, large language models (LLMs) remain vulnerable to jailbreak attacks. In this paper, we propose AutoDefense, a response-filtering based multi-agent defense framework that filters harmful responses from LLMs. This framework assigns different roles to LLM agents and employs them to complete the defense task collaboratively. The division in tasks enhances the overall instruction-following of LLMs and enables the integration of other defense components as tools. AutoDefense can adapt to various sizes and kinds of open-source LLMs that serve as agents. Through conducting extensive experiments on a large scale of harmful and safe prompts, we validate the effectiveness of the proposed AutoDefense in improving the robustness against jailbreak attacks, while maintaining the performance at normal user request. Our code and 
    
[^2]: IGUANe: 一种适用于脑MR图像多中心协调的三维通用CycleGAN模型

    IGUANe: a 3D generalizable CycleGAN for multicenter harmonization of brain MR images

    [https://arxiv.org/abs/2402.03227](https://arxiv.org/abs/2402.03227)

    IGUANe是一种三维通用CycleGAN模型，通过集成多个域的训练实现了脑MR图像的多中心协调，使其成为通用生成器。

    

    在MRI研究中，来自多个采集点的图像数据的聚合可以增加样本大小，但可能引入阻碍后续分析一致性的与采集点相关的变异。图像翻译的深度学习方法已经成为协调MR图像跨站点的解决方案。在本研究中，我们引入了IGUANe（具有统一对抗网络的图像生成），这是一种原始的三维模型，它结合了域转换的优势和直接应用样式转移方法来实现多中心脑MR图像协调。IGUANe通过多对一策略，集成了任意数量的域进行训练，扩展了CycleGAN架构。在推断过程中，该模型可以应用于任何图像，甚至来自未知采集点，使其成为协调的通用生成器。在由11台不同扫描仪的T1加权图像组成的数据集上进行训练，IGUANe在未见站点的数据上进行了评估。

    In MRI studies, the aggregation of imaging data from multiple acquisition sites enhances sample size but may introduce site-related variabilities that hinder consistency in subsequent analyses. Deep learning methods for image translation have emerged as a solution for harmonizing MR images across sites. In this study, we introduce IGUANe (Image Generation with Unified Adversarial Networks), an original 3D model that leverages the strengths of domain translation and straightforward application of style transfer methods for multicenter brain MR image harmonization. IGUANe extends CycleGAN architecture by integrating an arbitrary number of domains for training through a many-to-one strategy. During inference, the model can be applied to any image, even from an unknown acquisition site, making it a universal generator for harmonization. Trained on a dataset comprising T1-weighted images from 11 different scanners, IGUANe was evaluated on data from unseen sites. The assessments included the
    
[^3]: 向绿色且类人的人工智能迈进：当代少样本学习方法的全面调查

    Toward Green and Human-Like Artificial Intelligence: A Complete Survey on Contemporary Few-Shot Learning Approaches

    [https://arxiv.org/abs/2402.03017](https://arxiv.org/abs/2402.03017)

    本文全面调查了少样本学习领域的最新进展，探讨了该方法在解决深度学习在实际应用中的限制方面的潜力和挑战。

    

    尽管深度学习取得了广泛的成功，但其对数据的需求和计算的昂贵性使其在许多数据受限的真实应用中不实用。少样本学习（FSL）旨在通过实现对新学习任务的快速适应来解决这些限制，并在近年来取得了显著发展。本调查提供了该领域最新进展的全面概述。首先，正式定义了FSL，并介绍了它与不同学习领域的关系。引入了一种新的分类法，扩展了以前提出的方法，并对经典和新领域中的实际应用进行了描述。最后，讨论了塑造该领域的最新趋势、突出挑战和有前途的未来研究方向。

    Despite deep learning's widespread success, its data-hungry and computationally expensive nature makes it impractical for many data-constrained real-world applications. Few-Shot Learning (FSL) aims to address these limitations by enabling rapid adaptation to novel learning tasks, seeing significant growth in recent years. This survey provides a comprehensive overview of the field's latest advancements. Initially, FSL is formally defined, and its relationship with different learning fields is presented. A novel taxonomy is introduced, extending previously proposed ones, and real-world applications in classic and novel fields are described. Finally, recent trends shaping the field, outstanding challenges, and promising future research directions are discussed.
    
[^4]: 将硬件近似嵌入离散基因训练中以用于印刷多层感知器

    Embedding Hardware Approximations in Discrete Genetic-based Training for Printed MLPs

    [https://arxiv.org/abs/2402.02930](https://arxiv.org/abs/2402.02930)

    本文将硬件近似嵌入到印刷多层感知器的训练过程中，通过离散遗传算法实现了最大化硬件近似的效益，在5%的精度损失下，相比基线，实现了超过5倍的面积和功耗的减少，并且超过了最先进的近似方法。

    

    印刷电子是一种有着低成本和灵活制造等独特特点的有望广泛应用于计算领域的技术。与传统的硅基技术不同，印刷电子可以实现可伸缩、可适应、非毒性的硬件。然而，由于印刷电子的特性尺寸较大，要实现复杂的电路如机器学习分类器是具有挑战性的。近似计算被证明可以降低机器学习电路（如多层感知器）的硬件成本。在本文中，我们通过将硬件近似嵌入到多层感知器的训练过程中来最大化近似计算的益处。由于硬件近似的离散性，我们提出并实现了一种基于遗传算法的硬件感知训练方法，专门为印刷多层感知器设计。在5%的精度损失下，相比基线，我们的多层感知器在面积和功耗上实现了超过5倍的减少，并且超过了最先进的近似方法。

    Printed Electronics (PE) stands out as a promisingtechnology for widespread computing due to its distinct attributes, such as low costs and flexible manufacturing. Unlike traditional silicon-based technologies, PE enables stretchable, conformal,and non-toxic hardware. However, PE are constrained by larger feature sizes, making it challenging to implement complex circuits such as machine learning (ML) classifiers. Approximate computing has been proven to reduce the hardware cost of ML circuits such as Multilayer Perceptrons (MLPs). In this paper, we maximize the benefits of approximate computing by integrating hardware approximation into the MLP training process. Due to the discrete nature of hardware approximation, we propose and implement a genetic-based, approximate, hardware-aware training approach specifically designed for printed MLPs. For a 5% accuracy loss, our MLPs achieve over 5x area and power reduction compared to the baseline while outperforming state of-the-art approximate
    
[^5]: 等变对称破缺集

    Equivariant Symmetry Breaking Sets

    [https://arxiv.org/abs/2402.02681](https://arxiv.org/abs/2402.02681)

    这里是中文总结出的一句话要点: 该论文提出了一种全等变的对称破缺框架，通过引入对称破缺集来破坏等变神经网络中的对称性。这种方法通用且适用于任何群的等变性。

    

    等变神经网络（ENN）已被证明在涉及潜在对称性的应用中非常有效。通过设计，ENN在给定更高对称性输入时无法产生较低对称性输出。然而，在许多物理系统中会发生自发对称破缺，我们可以从一个初始高度对称的状态获得一个较不对称的稳定状态。因此，我们必须了解如何系统地在ENN中破坏对称性。在这项工作中，我们提出了一种全等变的新型对称破缺框架。我们强调我们的方法是通用的，并适用于任何群的等变性。为了实现这一目标，我们引入了对称破缺集（SBS）的概念。我们不是重新设计现有的网络，而是设计了一组对称破缺对象，根据输入和输出的对称性将其输入到我们的网络中。我们展示了在这些集合上定义等变性的一种自然方式，它提供了额外的约束。通过最小化... (the abstract is incomplete and cut off)

    Equivariant neural networks (ENNs) have been shown to be extremely effective in applications involving underlying symmetries. By construction ENNs cannot produce lower symmetry outputs given a higher symmetry input. However, spontaneous symmetry breaking occurs in many physical systems and we may obtain a less symmetric stable state from an initial highly symmetric one. Hence, it is imperative that we understand how to systematically break symmetry in ENNs. In this work, we propose a novel symmetry breaking framework that is fully equivariant. We emphasize that our approach is general and applicable to equivariance under any group. To achieve this, we introduce the idea of symmetry breaking sets (SBS). Rather than redesign existing networks, we design sets of symmetry breaking objects which we feed into our network based on the symmetry of our inputs and outputs. We show there is a natural way to define equivariance on these sets, which gives an additional constraint. Minimizing the si
    
[^6]: Vision Mamba: 使用双向状态空间模型高效学习视觉表示

    Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model

    [https://arxiv.org/abs/2401.09417](https://arxiv.org/abs/2401.09417)

    本文提出了一种新的通用视觉骨干模型Vim，通过使用双向状态空间模型和位置嵌入来高效表示视觉数据，相比于传统的视觉转换器如DeiT，在各种视觉任务上取得了更好的性能，并且实现了显著提升。

    

    最近，具有高效硬件感知设计的状态空间模型（SSMs），即Mamba深度学习模型，在长序列建模方面展现出巨大潜力。与此同时，在SSMs上构建高效且通用的视觉骨干模型也是一种有吸引力的方向。然而，由于视觉数据的位置敏感性和对全局上下文的需求，对于SSMs来说，表示视觉数据是具有挑战性的。在本文中，我们展示了对于视觉表示学习来说，依赖自注意力并不是必要的，并提出了一种新的通用视觉骨干模型，即带有双向Mamba块（Vim），它使用位置嵌入标记图像序列，并使用双向状态空间模型压缩视觉表示。在ImageNet分类、COCO目标检测和ADE20k语义分割任务中，Vim相比于DeiT等经过良好验证的视觉转换器，实现了更高的性能，并且显示出了显著的改进。

    Recently the state space models (SSMs) with efficient hardware-aware designs, i.e., the Mamba deep learning model, have shown great potential for long sequence modeling. Meanwhile building efficient and generic vision backbones purely upon SSMs is an appealing direction. However, representing visual data is challenging for SSMs due to the position-sensitivity of visual data and the requirement of global context for visual understanding. In this paper, we show that the reliance on self-attention for visual representation learning is not necessary and propose a new generic vision backbone with bidirectional Mamba blocks (Vim), which marks the image sequences with position embeddings and compresses the visual representation with bidirectional state space models. On ImageNet classification, COCO object detection, and ADE20k semantic segmentation tasks, Vim achieves higher performance compared to well-established vision transformers like DeiT, while also demonstrating significantly improved
    
[^7]: 面向印刷多层感知机的定制近似乘积累加和激活技术

    Bespoke Approximation of Multiplication-Accumulation and Activation Targeting Printed Multilayer Perceptrons

    [https://arxiv.org/abs/2312.17612](https://arxiv.org/abs/2312.17612)

    本研究提出了一种针对印刷电子技术中的限制的自动化框架，用于设计超低功耗的多层感知机（MLP）分类器。

    

    印刷电子技术具有独特的特性，使其成为实现真正无处不在计算的重要技术。本研究提出了一种自动化框架，用于设计超低功耗的多层感知机（MLP）分类器，通过利用近似计算和定制化设计的原则来克服印刷电子技术中的限制。

    Printed Electronics (PE) feature distinct and remarkable characteristics that make them a prominent technology for achieving true ubiquitous computing. This is particularly relevant in application domains that require conformal and ultra-low cost solutions, which have experienced limited penetration of computing until now. Unlike silicon-based technologies, PE offer unparalleled features such as non-recurring engineering costs, ultra-low manufacturing cost, and on-demand fabrication of conformal, flexible, non-toxic, and stretchable hardware. However, PE face certain limitations due to their large feature sizes, that impede the realization of complex circuits, such as machine learning classifiers. In this work, we address these limitations by leveraging the principles of Approximate Computing and Bespoke (fully-customized) design. We propose an automated framework for designing ultra-low power Multilayer Perceptron (MLP) classifiers which employs, for the first time, a holistic approac
    
[^8]: 细胞信号传导结构和功能

    The cell signaling structure function. (arXiv:2401.02501v1 [cs.CV])

    [http://arxiv.org/abs/2401.02501](http://arxiv.org/abs/2401.02501)

    该论文提出了一个新的方法，在活体细胞显微镜捕捉到的五维视频中寻找细胞信号动力学时空模式，并且不需要任何先验的预期模式动力学和训练数据。该方法基于细胞信号结构函数（SSF），通过测量细胞信号状态和周围细胞质之间的核糖体强度，与当前最先进的核糖体与细胞核比值相比有了显著改进。通过归一化压缩距离（NCD）来识别相似的模式。该方法能够将输入的SSF构图表示为低维嵌入中的点，最优地捕捉模式。

    

    活体细胞显微镜捕捉到的五维$(x,y,z,channel,time)$视频显示了细胞运动和信号动力学的模式。我们在这里提出一种在五维活体细胞显微镜视频中寻找细胞信号动力学时空模式的方法，该方法独特之处在于不需要预先了解预期的模式动力学以及没有训练数据。所提出的细胞信号结构函数（SSF）是一种Kolmogorov结构函数，可以通过核心区域相对于周围细胞质的核糖体强度来最优地测量细胞信号状态，相比当前最先进的核糖体与细胞核比值有了显著的改进。通过度量归一化压缩距离（NCD）来识别相似的模式。NCD是一个用于表示输入的SSF构图在低维嵌入中作为点的Hilbert空间的再生核，可以最优地捕捉模式。

    Live cell microscopy captures 5-D $(x,y,z,channel,time)$ movies that display patterns of cellular motion and signaling dynamics. We present here an approach to finding spatiotemporal patterns of cell signaling dynamics in 5-D live cell microscopy movies unique in requiring no \emph{a priori} knowledge of expected pattern dynamics, and no training data. The proposed cell signaling structure function (SSF) is a Kolmogorov structure function that optimally measures cell signaling state as nuclear intensity w.r.t. surrounding cytoplasm, a significant improvement compared to the current state-of-the-art cytonuclear ratio. SSF kymographs store at each spatiotemporal cell centroid the SSF value, or a functional output such as velocity. Patterns of similarity are identified via the metric normalized compression distance (NCD). The NCD is a reproducing kernel for a Hilbert space that represents the input SSF kymographs as points in a low dimensional embedding that optimally captures the pattern
    
[^9]: ResBit: 基于残差位向量的离散值表示方法

    ResBit: Residual Bit Vector for Categorical Values. (arXiv:2309.17196v1 [cs.LG])

    [http://arxiv.org/abs/2309.17196](http://arxiv.org/abs/2309.17196)

    本论文提出了一种名为ResBit的残差位向量方法，用于解决在深度学习中表示离散数据维度增加和无法恢复原始类别值的问题。

    

    长期以来，独热编码向量一直广泛应用于机器学习中，作为一种简单且通用的表示离散数据的方法。然而，这种方法会导致维度随着要表示的离散数据线性增加，这在深度学习中视为空间计算复杂性的问题，而深度学习需要大量的数据。最近，基于扩散模型的高表达能力，提出了一种用位序列表示离散数据的方法，即Analog Bits。然而，由于在生成任务中要表示的类别类型数量不一定是2的幂次，导致Analog Bits能够表示的范围与类别数据的范围存在差异。如果生成了这样的值，问题就是无法恢复原始的类别值。为了解决这个问题，我们提出了残差位向量（ResBit），它是一种分层的位表示方法。

    The one-hot vector has long been widely used in machine learning as a simple and generic method for representing discrete data. However, this method increases the number of dimensions linearly with the categorical data to be represented, which is problematic from the viewpoint of spatial computational complexity in deep learning, which requires a large amount of data. Recently, Analog Bits, a method for representing discrete data as a sequence of bits, was proposed on the basis of the high expressiveness of diffusion models. However, since the number of category types to be represented in a generation task is not necessarily at a power of two, there is a discrepancy between the range that Analog Bits can represent and the range represented as category data. If such a value is generated, the problem is that the original category value cannot be restored. To address this issue, we propose Residual Bit Vector (ResBit), which is a hierarchical bit representation. Although it is a general-p
    
[^10]: 带有预算和ROI约束的自动出价算法：效率、后悔和节奏动态

    Autobidders with Budget and ROI Constraints: Efficiency, Regret, and Pacing Dynamics. (arXiv:2301.13306v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2301.13306](http://arxiv.org/abs/2301.13306)

    本文提出了一个基于梯度的学习算法，可以在多种拍卖方式下满足预算和ROI约束，并达到个体后悔逐渐减小；结果表明，当各自竞争时，期望资金流动至少达到最优分配的期望流动的一半。

    

    我们研究了自动出价算法在在线广告平台上进行博弈的情况。每个自动出价算法被赋予任务，在多轮重复拍卖中，最大化其广告主的总价值，同时受到预算和/或投资回报率约束。我们提出了一种基于梯度的学习算法，它可以保证满足所有约束条件，并达到逐渐减小的个体后悔。我们的算法仅使用自助反馈，并可与第一或第二价格拍卖以及任何“中间”拍卖格式一起使用。我们的主要结果是，当这些自动出价算法相互竞争时，所有轮次的期望资金流动 welfare 都至少达到了任何分配所实现的期望最优流动 welfare 的一半。这在出价动态是否收敛到均衡以及广告主估值之间的相关结构如何不同的情况下均成立。

    We study a game between autobidding algorithms that compete in an online advertising platform. Each autobidder is tasked with maximizing its advertiser's total value over multiple rounds of a repeated auction, subject to budget and/or return-on-investment constraints. We propose a gradient-based learning algorithm that is guaranteed to satisfy all constraints and achieves vanishing individual regret. Our algorithm uses only bandit feedback and can be used with the first- or second-price auction, as well as with any "intermediate" auction format. Our main result is that when these autobidders play against each other, the resulting expected liquid welfare over all rounds is at least half of the expected optimal liquid welfare achieved by any allocation. This holds whether or not the bidding dynamics converges to an equilibrium and regardless of the correlation structure between advertiser valuations.
    

