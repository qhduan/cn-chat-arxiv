# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [STARFlow: Spatial Temporal Feature Re-embedding with Attentive Learning for Real-world Scene Flow](https://arxiv.org/abs/2403.07032) | 提出了一种全局注意力流嵌入和空间时间特征重新嵌入模块相结合的方法，用于解决现实世界场景流预测中的局部依赖匹配和非刚性物体变形的挑战。 |
| [^2] | [Can LLMs Recognize Toxicity? Structured Toxicity Investigation Framework and Semantic-Based Metric](https://arxiv.org/abs/2402.06900) | 本研究提出了一种基于大型语言模型（LLMs）的自动度量方法，用于识别生成文本中的毒性。通过分析毒性因素和LLMs的内在毒性属性，该方法在测量毒性方面表现出众，比现有指标提升12个百分点。 |
| [^3] | [IGUANe: a 3D generalizable CycleGAN for multicenter harmonization of brain MR images](https://arxiv.org/abs/2402.03227) | IGUANe是一种三维通用CycleGAN模型，通过集成多个域的训练实现了脑MR图像的多中心协调，使其成为通用生成器。 |
| [^4] | [Toward Green and Human-Like Artificial Intelligence: A Complete Survey on Contemporary Few-Shot Learning Approaches](https://arxiv.org/abs/2402.03017) | 本文全面调查了少样本学习领域的最新进展，探讨了该方法在解决深度学习在实际应用中的限制方面的潜力和挑战。 |
| [^5] | [Equivariant Symmetry Breaking Sets](https://arxiv.org/abs/2402.02681) | 这里是中文总结出的一句话要点: 该论文提出了一种全等变的对称破缺框架，通过引入对称破缺集来破坏等变神经网络中的对称性。这种方法通用且适用于任何群的等变性。 |
| [^6] | [Uncovering communities of pipelines in the task-fMRI analytical space](https://arxiv.org/abs/2312.06231) | 本论文通过使用社群检测算法揭示了任务fMRI分析空间中的流程社群，并评估了不同背景下的流程关系的稳定性。研究表明，存在一些子集的流程给出相似的结果，特别是那些分享特定参数。这些结果对于参与者群体来说是稳定的，但在不同任务之间不稳定。流程空间的形成主要受到大脑激活区域大小和统计值规模的影响。 |
| [^7] | [Lifted Inference beyond First-Order Logic.](http://arxiv.org/abs/2308.11738) | 这项工作研究了超越一阶逻辑的提升推理问题，扩展了计数量词扩展的两个变量的一阶逻辑片段的域可提升性，并在限定了关系的情况下探索了不同属性的域可提升性。 |

# 详细

[^1]: STARFlow: 具有注意力学习的空间时间特征重新嵌入用于现实世界的场景流

    STARFlow: Spatial Temporal Feature Re-embedding with Attentive Learning for Real-world Scene Flow

    [https://arxiv.org/abs/2403.07032](https://arxiv.org/abs/2403.07032)

    提出了一种全局注意力流嵌入和空间时间特征重新嵌入模块相结合的方法，用于解决现实世界场景流预测中的局部依赖匹配和非刚性物体变形的挑战。

    

    场景流预测是理解动态场景中的关键任务，因为它提供了基本的运动信息。然而，当代场景流方法面临三大挑战。首先，仅基于局部感受野的流估计缺乏点对的长依赖匹配。为了解决这个问题，我们提出了全局注意力流嵌入，以匹配特征空间和欧几里得空间中的所有点对，提供局部细化之前的全局初始化。其次，在变形后存在非刚性物体的变形，导致连续帧之间的时空关系变化。为了更精确地估计残余流，设计了一个空间时间特征重新嵌入模块，以在变形后获取序列特征。此外，由于合成数据和真实数据之间的显著域差异，先前的方法表现出较差的泛化能力。

    arXiv:2403.07032v1 Announce Type: cross  Abstract: Scene flow prediction is a crucial underlying task in understanding dynamic scenes as it offers fundamental motion information. However, contemporary scene flow methods encounter three major challenges. Firstly, flow estimation solely based on local receptive fields lacks long-dependency matching of point pairs. To address this issue, we propose global attentive flow embedding to match all-to-all point pairs in both feature space and Euclidean space, providing global initialization before local refinement. Secondly, there are deformations existing in non-rigid objects after warping, which leads to variations in the spatiotemporal relation between the consecutive frames. For a more precise estimation of residual flow, a spatial temporal feature re-embedding module is devised to acquire the sequence features after deformation. Furthermore, previous methods perform poor generalization due to the significant domain gap between the synthesi
    
[^2]: LLM能够识别毒性吗？结构化毒性调查框架和基于语义的度量

    Can LLMs Recognize Toxicity? Structured Toxicity Investigation Framework and Semantic-Based Metric

    [https://arxiv.org/abs/2402.06900](https://arxiv.org/abs/2402.06900)

    本研究提出了一种基于大型语言模型（LLMs）的自动度量方法，用于识别生成文本中的毒性。通过分析毒性因素和LLMs的内在毒性属性，该方法在测量毒性方面表现出众，比现有指标提升12个百分点。

    

    在开发遵守社会标准的大型语言模型（LLMs）的过程中，识别生成文本中的毒性存在至关重要。现有的大多数毒性度量依赖于在特定毒性数据集上训练的编码模型。然而，这些编码器容易受到分布外的问题的影响，并且依赖于数据集中所假定的毒性定义。本文介绍了一种基于LLMs的自动鲁棒度量，用于区分模型回应是否具有毒性。我们首先分析了毒性因素，然后研究了LLMs的内在毒性属性，以确定它们作为评估器的适用性。随后，我们对评估数据集上的度量指标LLMs As ToxiciTy Evaluators（LATTE）进行了评估。实证结果表明，在不进行训练过程的情况下，我们的度量在测量毒性方面表现出色，F1得分比现有技术指标提高了12个百分点。我们还展示了上游毒性对度量结果的影响。

    In the pursuit of developing Large Language Models (LLMs) that adhere to societal standards, it is imperative to discern the existence of toxicity in the generated text. The majority of existing toxicity metrics rely on encoder models trained on specific toxicity datasets. However, these encoders are susceptible to out-of-distribution (OOD) problems and depend on the definition of toxicity assumed in a dataset. In this paper, we introduce an automatic robust metric grounded on LLMs to distinguish whether model responses are toxic. We start by analyzing the toxicity factors, followed by examining the intrinsic toxic attributes of LLMs to ascertain their suitability as evaluators. Subsequently, we evaluate our metric, LLMs As ToxiciTy Evaluators (LATTE), on evaluation datasets.The empirical results indicate outstanding performance in measuring toxicity, improving upon state-of-the-art metrics by 12 points in F1 score without training procedure. We also show that upstream toxicity has an 
    
[^3]: IGUANe: 一种适用于脑MR图像多中心协调的三维通用CycleGAN模型

    IGUANe: a 3D generalizable CycleGAN for multicenter harmonization of brain MR images

    [https://arxiv.org/abs/2402.03227](https://arxiv.org/abs/2402.03227)

    IGUANe是一种三维通用CycleGAN模型，通过集成多个域的训练实现了脑MR图像的多中心协调，使其成为通用生成器。

    

    在MRI研究中，来自多个采集点的图像数据的聚合可以增加样本大小，但可能引入阻碍后续分析一致性的与采集点相关的变异。图像翻译的深度学习方法已经成为协调MR图像跨站点的解决方案。在本研究中，我们引入了IGUANe（具有统一对抗网络的图像生成），这是一种原始的三维模型，它结合了域转换的优势和直接应用样式转移方法来实现多中心脑MR图像协调。IGUANe通过多对一策略，集成了任意数量的域进行训练，扩展了CycleGAN架构。在推断过程中，该模型可以应用于任何图像，甚至来自未知采集点，使其成为协调的通用生成器。在由11台不同扫描仪的T1加权图像组成的数据集上进行训练，IGUANe在未见站点的数据上进行了评估。

    In MRI studies, the aggregation of imaging data from multiple acquisition sites enhances sample size but may introduce site-related variabilities that hinder consistency in subsequent analyses. Deep learning methods for image translation have emerged as a solution for harmonizing MR images across sites. In this study, we introduce IGUANe (Image Generation with Unified Adversarial Networks), an original 3D model that leverages the strengths of domain translation and straightforward application of style transfer methods for multicenter brain MR image harmonization. IGUANe extends CycleGAN architecture by integrating an arbitrary number of domains for training through a many-to-one strategy. During inference, the model can be applied to any image, even from an unknown acquisition site, making it a universal generator for harmonization. Trained on a dataset comprising T1-weighted images from 11 different scanners, IGUANe was evaluated on data from unseen sites. The assessments included the
    
[^4]: 向绿色且类人的人工智能迈进：当代少样本学习方法的全面调查

    Toward Green and Human-Like Artificial Intelligence: A Complete Survey on Contemporary Few-Shot Learning Approaches

    [https://arxiv.org/abs/2402.03017](https://arxiv.org/abs/2402.03017)

    本文全面调查了少样本学习领域的最新进展，探讨了该方法在解决深度学习在实际应用中的限制方面的潜力和挑战。

    

    尽管深度学习取得了广泛的成功，但其对数据的需求和计算的昂贵性使其在许多数据受限的真实应用中不实用。少样本学习（FSL）旨在通过实现对新学习任务的快速适应来解决这些限制，并在近年来取得了显著发展。本调查提供了该领域最新进展的全面概述。首先，正式定义了FSL，并介绍了它与不同学习领域的关系。引入了一种新的分类法，扩展了以前提出的方法，并对经典和新领域中的实际应用进行了描述。最后，讨论了塑造该领域的最新趋势、突出挑战和有前途的未来研究方向。

    Despite deep learning's widespread success, its data-hungry and computationally expensive nature makes it impractical for many data-constrained real-world applications. Few-Shot Learning (FSL) aims to address these limitations by enabling rapid adaptation to novel learning tasks, seeing significant growth in recent years. This survey provides a comprehensive overview of the field's latest advancements. Initially, FSL is formally defined, and its relationship with different learning fields is presented. A novel taxonomy is introduced, extending previously proposed ones, and real-world applications in classic and novel fields are described. Finally, recent trends shaping the field, outstanding challenges, and promising future research directions are discussed.
    
[^5]: 等变对称破缺集

    Equivariant Symmetry Breaking Sets

    [https://arxiv.org/abs/2402.02681](https://arxiv.org/abs/2402.02681)

    这里是中文总结出的一句话要点: 该论文提出了一种全等变的对称破缺框架，通过引入对称破缺集来破坏等变神经网络中的对称性。这种方法通用且适用于任何群的等变性。

    

    等变神经网络（ENN）已被证明在涉及潜在对称性的应用中非常有效。通过设计，ENN在给定更高对称性输入时无法产生较低对称性输出。然而，在许多物理系统中会发生自发对称破缺，我们可以从一个初始高度对称的状态获得一个较不对称的稳定状态。因此，我们必须了解如何系统地在ENN中破坏对称性。在这项工作中，我们提出了一种全等变的新型对称破缺框架。我们强调我们的方法是通用的，并适用于任何群的等变性。为了实现这一目标，我们引入了对称破缺集（SBS）的概念。我们不是重新设计现有的网络，而是设计了一组对称破缺对象，根据输入和输出的对称性将其输入到我们的网络中。我们展示了在这些集合上定义等变性的一种自然方式，它提供了额外的约束。通过最小化... (the abstract is incomplete and cut off)

    Equivariant neural networks (ENNs) have been shown to be extremely effective in applications involving underlying symmetries. By construction ENNs cannot produce lower symmetry outputs given a higher symmetry input. However, spontaneous symmetry breaking occurs in many physical systems and we may obtain a less symmetric stable state from an initial highly symmetric one. Hence, it is imperative that we understand how to systematically break symmetry in ENNs. In this work, we propose a novel symmetry breaking framework that is fully equivariant. We emphasize that our approach is general and applicable to equivariance under any group. To achieve this, we introduce the idea of symmetry breaking sets (SBS). Rather than redesign existing networks, we design sets of symmetry breaking objects which we feed into our network based on the symmetry of our inputs and outputs. We show there is a natural way to define equivariance on these sets, which gives an additional constraint. Minimizing the si
    
[^6]: 揭示任务fMRI分析空间中的流程社群

    Uncovering communities of pipelines in the task-fMRI analytical space

    [https://arxiv.org/abs/2312.06231](https://arxiv.org/abs/2312.06231)

    本论文通过使用社群检测算法揭示了任务fMRI分析空间中的流程社群，并评估了不同背景下的流程关系的稳定性。研究表明，存在一些子集的流程给出相似的结果，特别是那些分享特定参数。这些结果对于参与者群体来说是稳定的，但在不同任务之间不稳定。流程空间的形成主要受到大脑激活区域大小和统计值规模的影响。

    

    功能磁共振成像中的分析工作流程具有高度灵活性，选择流程的最佳实践有限。尽管已经显示出使用不同流程可能导致不同的结果，但对于驱动这些差异的因素以及这些差异在不同背景下的稳定性仍然缺乏理解。我们使用社群检测算法探索流程空间，并评估不同背景下流程关系的稳定性。我们发现，存在一些子集的流程给出相似的结果，特别是那些分享特定参数（例如运动回归器的数量、软件包等）。这些流程与流程之间的模式在参与者群体中是稳定的，但在不同任务之间不稳定。通过可视化社群间的差异，我们发现流程空间主要受大脑激活区域的大小和统计值的规模的影响。

    Analytical workflows in functional magnetic resonance imaging are highly flexible with limited best practices as to how to choose a pipeline. While it has been shown that the use of different pipelines might lead to different results, there is still a lack of understanding of the factors that drive these differences and of the stability of these differences across contexts. We use community detection algorithms to explore the pipeline space and assess the stability of pipeline relationships across different contexts. We show that there are subsets of pipelines that give similar results, especially those sharing specific parameters (e.g. number of motion regressors, software packages, etc.). Those pipeline-to-pipeline patterns are stable across groups of participants but not across different tasks. By visualizing the differences between communities, we show that the pipeline space is mainly driven by the size of the activation area in the brain and the scale of statistic values in stati
    
[^7]: 超出一阶逻辑的提升推理

    Lifted Inference beyond First-Order Logic. (arXiv:2308.11738v1 [cs.AI])

    [http://arxiv.org/abs/2308.11738](http://arxiv.org/abs/2308.11738)

    这项工作研究了超越一阶逻辑的提升推理问题，扩展了计数量词扩展的两个变量的一阶逻辑片段的域可提升性，并在限定了关系的情况下探索了不同属性的域可提升性。

    

    在统计关系学习模型中，加权一阶模型计数(WFOMC)是概率推理的基础。由于WFOMC在一般情况下是不可计算的（$\#$P完全），因此能够在多项式时间内进行WFOMC的逻辑碎片非常有意义。这样的碎片被称为域可提升。最近的研究表明，在计数量词（$\mathrm{C^2}$）扩展的两个变量的一阶逻辑片段中，可以进行域提升。然而，许多真实世界数据的属性，如引用网络中的非循环性和社交网络中的连通性，不能在$\mathrm{C^2}$或一阶逻辑中建模。在这项工作中，我们扩展了$\mathrm{C^2}$的域可提升性，包括多个这样的属性。我们证明了在将$\mathrm{C^2}$句子的一个关系限定为表示有向无环图、连通图、树（或有向树）或森林（或有向森林）时，它仍然保持了域可提升性。所有我们的结果都是...

    Weighted First Order Model Counting (WFOMC) is fundamental to probabilistic inference in statistical relational learning models. As WFOMC is known to be intractable in general ($\#$P-complete), logical fragments that admit polynomial time WFOMC are of significant interest. Such fragments are called domain liftable. Recent works have shown that the two-variable fragment of first order logic extended with counting quantifiers ($\mathrm{C^2}$) is domain-liftable. However, many properties of real-world data, like acyclicity in citation networks and connectivity in social networks, cannot be modeled in $\mathrm{C^2}$, or first order logic in general. In this work, we expand the domain liftability of $\mathrm{C^2}$ with multiple such properties. We show that any $\mathrm{C^2}$ sentence remains domain liftable when one of its relations is restricted to represent a directed acyclic graph, a connected graph, a tree (resp. a directed tree) or a forest (resp. a directed forest). All our results r
    

