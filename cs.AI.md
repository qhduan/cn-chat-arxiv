# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond Joint Demonstrations: Personalized Expert Guidance for Efficient Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2403.08936) | 提出了个性化专家示范的概念，为每个智体或不同类型的智体提供针对个人目标的指导，解决了多智体强化学习中联合示范困难的问题。 |
| [^2] | [Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data](https://arxiv.org/abs/2402.14989) | 神经常微分方程（Neural ODEs）的扩展——神经随机微分方程（Neural SDEs）在处理不规则时间序列数据中的稳定性和性能方面提出了重要指导，需要谨慎设计漂移和扩散函数以保持稳定性。 |
| [^3] | [AdaFlow: Imitation Learning with Variance-Adaptive Flow-Based Policies](https://arxiv.org/abs/2402.04292) | AdaFlow是一个基于流模型的模仿学习框架，通过使用变异自适应ODE求解器，在保持多样性的同时，提供快速推理能力。 |
| [^4] | [Adaptive Communications in Collaborative Perception with Domain Alignment for Autonomous Driving.](http://arxiv.org/abs/2310.00013) | 这篇论文提出了一个通信的协同感知框架ACC-DA，通过动态调整通信图和自适应数据重构机制来增强自动驾驶中的感知能力。 |
| [^5] | [NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics.](http://arxiv.org/abs/2306.06202) | 本文介绍了神经成像领域的图机器学习基准测试NeuroGraph，并探讨了数据集生成的搜索空间。 |
| [^6] | [Visually-Grounded Descriptions Improve Zero-Shot Image Classification.](http://arxiv.org/abs/2306.06077) | 本文提出了一种称为V-GLOSS的新方法，它利用现代语言模型和语义知识库生成具有视觉基础的类别描述，提高了零样本图像分类的准确性，并引入了一个带有类别描述的银标准数据集。 |
| [^7] | [Artificial General Intelligence for Medical Imaging.](http://arxiv.org/abs/2306.05480) | 本文探讨了人工通用智能模型在医学成像中的应用，重点关注基础的大型语言、视觉和多模态模型。通过整合临床专业知识、领域知识和多模态能力来开发和部署AGI能够解决医学领域面临的挑战和问题。 |
| [^8] | [Deep ReLU Networks Have Surprisingly Simple Polytopes.](http://arxiv.org/abs/2305.09145) | 本文通过计算和分析ReLU网络多面体的单纯形直方图，发现在初始化和梯度下降时它们结构相对简单，这说明了一种新的隐式偏见。 |
| [^9] | [NeuroBench: Advancing Neuromorphic Computing through Collaborative, Fair and Representative Benchmarking.](http://arxiv.org/abs/2304.04640) | NeuroBench是由学术界和工业界成员共同开发的一套协作、公平和代表性的基准测试，可以解决神经形态计算中缺乏清晰标准的问题，推动该领域的发展。 |
| [^10] | [AI-Enhanced Intensive Care Unit: Revolutionizing Patient Care with Pervasive Sensing.](http://arxiv.org/abs/2303.06252) | 本文介绍了一种AI增强的重症监护室系统，通过普适感知和数据处理，可以改善患者的视觉监测和评估，提高护理质量。 |
| [^11] | [Huber-energy measure quantization.](http://arxiv.org/abs/2212.08162) | 该论文提出了一种Huber能量量化的算法，用于找到目标概率定律的最佳逼近，通过最小化原测度与量化版本之间的统计距离来实现。该算法已在多维高斯混合物、维纳空间魔方等几个数据库上进行了测试。 |

# 详细

[^1]: 超越联合示范：个性化专家指导用于高效多智体强化学习

    Beyond Joint Demonstrations: Personalized Expert Guidance for Efficient Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2403.08936](https://arxiv.org/abs/2403.08936)

    提出了个性化专家示范的概念，为每个智体或不同类型的智体提供针对个人目标的指导，解决了多智体强化学习中联合示范困难的问题。

    

    多智体强化学习算法面临有效探索的挑战，因为联合状态-动作空间的大小呈指数增长。虽然示范引导学习在单智体环境中已被证明是有益的，但其直接应用于多智体强化学习受到获得联合专家示范的实际困难的阻碍。在这项工作中，我们引入了个性化专家示范的新概念，针对每个单个智体或更广泛地说，团队中每种类型的智体进行了定制。这些示范仅涉及单智体行为以及每个智体如何实现个人目标，而不涉及任何合作元素，因此盲目模仿它们不会实现合作由于潜在冲突。为此，我们提出了一种方法，选择性地利用个性化专家示范作为指导，并允许智体学习协

    arXiv:2403.08936v1 Announce Type: cross  Abstract: Multi-Agent Reinforcement Learning (MARL) algorithms face the challenge of efficient exploration due to the exponential increase in the size of the joint state-action space. While demonstration-guided learning has proven beneficial in single-agent settings, its direct applicability to MARL is hindered by the practical difficulty of obtaining joint expert demonstrations. In this work, we introduce a novel concept of personalized expert demonstrations, tailored for each individual agent or, more broadly, each individual type of agent within a heterogeneous team. These demonstrations solely pertain to single-agent behaviors and how each agent can achieve personal goals without encompassing any cooperative elements, thus naively imitating them will not achieve cooperation due to potential conflicts. To this end, we propose an approach that selectively utilizes personalized expert demonstrations as guidance and allows agents to learn to coo
    
[^2]: 分析不规则时间序列数据中的稳定神经随机微分方程

    Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data

    [https://arxiv.org/abs/2402.14989](https://arxiv.org/abs/2402.14989)

    神经常微分方程（Neural ODEs）的扩展——神经随机微分方程（Neural SDEs）在处理不规则时间序列数据中的稳定性和性能方面提出了重要指导，需要谨慎设计漂移和扩散函数以保持稳定性。

    

    实际时间序列数据中的不规则采样间隔和缺失值对于假设一致间隔和完整数据的传统方法构成挑战。神经常微分方程（Neural ODEs）提供了一种替代方法，利用神经网络与常微分方程求解器结合，通过参数化向量场学习连续潜在表示。神经随机微分方程（Neural SDEs）通过引入扩散项扩展了神经常微分方程，然而在处理不规则间隔和缺失值时，这种添加并不是微不足道的。因此，仔细设计漂移和扩散函数对于保持稳定性和增强性能至关重要，而粗心的选择可能导致出现没有强解、随机破坏或不稳定的Euler离散化等不利的性质，显著影响神经随机微分方程的性能。

    arXiv:2402.14989v1 Announce Type: cross  Abstract: Irregular sampling intervals and missing values in real-world time series data present challenges for conventional methods that assume consistent intervals and complete data. Neural Ordinary Differential Equations (Neural ODEs) offer an alternative approach, utilizing neural networks combined with ODE solvers to learn continuous latent representations through parameterized vector fields. Neural Stochastic Differential Equations (Neural SDEs) extend Neural ODEs by incorporating a diffusion term, although this addition is not trivial, particularly when addressing irregular intervals and missing values. Consequently, careful design of drift and diffusion functions is crucial for maintaining stability and enhancing performance, while incautious choices can result in adverse properties such as the absence of strong solutions, stochastic destabilization, or unstable Euler discretizations, significantly affecting Neural SDEs' performance. In 
    
[^3]: AdaFlow: 变异自适应流策略的模仿学习

    AdaFlow: Imitation Learning with Variance-Adaptive Flow-Based Policies

    [https://arxiv.org/abs/2402.04292](https://arxiv.org/abs/2402.04292)

    AdaFlow是一个基于流模型的模仿学习框架，通过使用变异自适应ODE求解器，在保持多样性的同时，提供快速推理能力。

    

    基于扩散的模仿学习在多模态决策中改进了行为克隆（BC），但由于扩散过程中的递归而导致推理速度显著减慢。这促使我们设计高效的策略生成器，同时保持生成多样化动作的能力。为了解决这个挑战，我们提出了AdaFlow，这是一个基于流模型的模仿学习框架。AdaFlow使用状态条件的常微分方程（ODE）表示策略，这被称为概率流。我们揭示了它们训练损失的条件方差与ODE的离散化误差之间的有趣关系。基于这个观察，我们提出了一个变异自适应ODE求解器，在推理阶段可以调整步长，使AdaFlow成为一个自适应决策者，能够快速推理而不牺牲多样性。有趣的是，当动作分布被降低到一步生成器时，它自动退化到一个一步生成器。

    Diffusion-based imitation learning improves Behavioral Cloning (BC) on multi-modal decision-making, but comes at the cost of significantly slower inference due to the recursion in the diffusion process. It urges us to design efficient policy generators while keeping the ability to generate diverse actions. To address this challenge, we propose AdaFlow, an imitation learning framework based on flow-based generative modeling. AdaFlow represents the policy with state-conditioned ordinary differential equations (ODEs), which are known as probability flows. We reveal an intriguing connection between the conditional variance of their training loss and the discretization error of the ODEs. With this insight, we propose a variance-adaptive ODE solver that can adjust its step size in the inference stage, making AdaFlow an adaptive decision-maker, offering rapid inference without sacrificing diversity. Interestingly, it automatically reduces to a one-step generator when the action distribution i
    
[^4]: 自适应驾驶中基于领域匹配的协同感知的通信

    Adaptive Communications in Collaborative Perception with Domain Alignment for Autonomous Driving. (arXiv:2310.00013v1 [cs.AI])

    [http://arxiv.org/abs/2310.00013](http://arxiv.org/abs/2310.00013)

    这篇论文提出了一个通信的协同感知框架ACC-DA，通过动态调整通信图和自适应数据重构机制来增强自动驾驶中的感知能力。

    

    通过通信允许车辆交换补充信息，多个连接的自动驾驶车辆之间的协同感知可以极大地增强感知能力。尽管之前的方法取得了进展，但由于通道变化和协同车辆之间的数据异构性，仍然存在挑战。为了解决这些问题，我们提出了ACC-DA，一个通道感知的协同感知框架，它可以动态调整通信图并最小化平均传输延迟，同时减轻数据异构性带来的副作用。我们的创新点包括三个方面。首先，我们设计了一种最小化传输延迟的方法，根据不同的通道信息状态构建通信图并最小化传输延迟。然后，我们提出了一种自适应数据重构机制，可以动态调整码率-畸变折衷以增强感知效率。此外，它最小化了时域丢失。

    Collaborative perception among multiple connected and autonomous vehicles can greatly enhance perceptive capabilities by allowing vehicles to exchange supplementary information via communications. Despite advances in previous approaches, challenges still remain due to channel variations and data heterogeneity among collaborative vehicles. To address these issues, we propose ACC-DA, a channel-aware collaborative perception framework to dynamically adjust the communication graph and minimize the average transmission delay while mitigating the side effects from the data heterogeneity. Our novelties lie in three aspects. We first design a transmission delay minimization method, which can construct the communication graph and minimize the transmission delay according to different channel information state. We then propose an adaptive data reconstruction mechanism, which can dynamically adjust the rate-distortion trade-off to enhance perception efficiency. Moreover, it minimizes the temporal
    
[^5]: NeuroGraph:面向脑连接组学的图机器学习基准测试

    NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics. (arXiv:2306.06202v1 [cs.LG])

    [http://arxiv.org/abs/2306.06202](http://arxiv.org/abs/2306.06202)

    本文介绍了神经成像领域的图机器学习基准测试NeuroGraph，并探讨了数据集生成的搜索空间。

    

    机器学习为分析高维功能性神经成像数据提供了有价值的工具，已被证明对预测各种神经疾病、精神障碍和认知模式有效。在功能磁共振成像研究中，大脑区域之间的相互作用通常使用基于图的表示进行建模。图机器学习方法的有效性已在多个领域得到证实，标志着数据解释和预测建模中的一个转变步骤。然而，尽管有前景，但由于图形数据集构建的广泛预处理流水线和大参数搜索空间，在神经成像领域中应用这些技术的转换仍然受到意外的限制。本文介绍了NeuroGraph(一个基于图的神经成像数据集)，它涵盖了多个行为和认知特征类别。我们深入探讨了数据集生成搜索空间

    Machine learning provides a valuable tool for analyzing high-dimensional functional neuroimaging data, and is proving effective in predicting various neurological conditions, psychiatric disorders, and cognitive patterns. In functional Magnetic Resonance Imaging (MRI) research, interactions between brain regions are commonly modeled using graph-based representations. The potency of graph machine learning methods has been established across myriad domains, marking a transformative step in data interpretation and predictive modeling. Yet, despite their promise, the transposition of these techniques to the neuroimaging domain remains surprisingly under-explored due to the expansive preprocessing pipeline and large parameter search space for graph-based datasets construction. In this paper, we introduce NeuroGraph, a collection of graph-based neuroimaging datasets that span multiple categories of behavioral and cognitive traits. We delve deeply into the dataset generation search space by c
    
[^6]: 视觉词汇描述提升零样本图像分类

    Visually-Grounded Descriptions Improve Zero-Shot Image Classification. (arXiv:2306.06077v1 [cs.CV])

    [http://arxiv.org/abs/2306.06077](http://arxiv.org/abs/2306.06077)

    本文提出了一种称为V-GLOSS的新方法，它利用现代语言模型和语义知识库生成具有视觉基础的类别描述，提高了零样本图像分类的准确性，并引入了一个带有类别描述的银标准数据集。

    

    语言视觉模型如CLIP在零样本视觉任务（例如零样本图像分类ZSIC）方面取得了显著进展。然而，生成具体和富有表现力的类别描述仍然是一个主要挑战。现有方法存在粒度和标签歧义等问题。为了解决这些挑战，我们提出了一种新方法V-GLOSS：Visual Glosses，它利用现代语言模型和语义知识库来生成具有视觉基础的类别描述。我们通过在基准ZSIC数据集（包括ImageNet和STL-10）上实现最先进的结果来展示V-GLOSS的有效性。此外，我们引入了一个由V-GLOSS生成的带有类别描述的银标准数据集，并展示其用于视觉任务的有用性。我们提供了源代码和数据集。

    Language-vision models like CLIP have made significant progress in zero-shot vision tasks, such as zero-shot image classification (ZSIC). However, generating specific and expressive class descriptions remains a major challenge. Existing approaches suffer from granularity and label ambiguity issues. To tackle these challenges, we propose V-GLOSS: Visual Glosses, a novel method leveraging modern language models and semantic knowledge bases to produce visually-grounded class descriptions. We demonstrate V-GLOSS's effectiveness by achieving state-of-the-art results on benchmark ZSIC datasets including ImageNet and STL-10. In addition, we introduce a silver dataset with class descriptions generated by V-GLOSS, and show its usefulness for vision tasks. We make available our code and dataset.
    
[^7]: 医学成像的人工通用智能

    Artificial General Intelligence for Medical Imaging. (arXiv:2306.05480v1 [cs.AI])

    [http://arxiv.org/abs/2306.05480](http://arxiv.org/abs/2306.05480)

    本文探讨了人工通用智能模型在医学成像中的应用，重点关注基础的大型语言、视觉和多模态模型。通过整合临床专业知识、领域知识和多模态能力来开发和部署AGI能够解决医学领域面临的挑战和问题。

    

    本文探讨了人工通用智能模型在医疗保健中的潜在应用，重点关注基础的大型语言模型、大型视觉模型和大型多模态模型。我们强调将临床专业知识、领域知识和多模态能力整合到AGI模型中的重要性。此外，我们提出了指导医疗保健AGI模型开发和部署的路线图。在整个综述过程中，我们提供了关于在医学领域部署大规模AGI模型可能面临的潜在挑战和缺陷的关键观点。这篇综合性的综述旨在提供有关AGI对医学成像、医疗保健和其他领域未来影响的见解。

    In this review, we explore the potential applications of Artificial General Intelligence (AGI) models in healthcare, focusing on foundational Large Language Models (LLMs), Large Vision Models, and Large Multimodal Models. We emphasize the importance of integrating clinical expertise, domain knowledge, and multimodal capabilities into AGI models. In addition, we lay out key roadmaps that guide the development and deployment of healthcare AGI models. Throughout the review, we provide critical perspectives on the potential challenges and pitfalls associated with deploying large-scale AGI models in the medical field. This comprehensive review aims to offer insights into the future implications of AGI in medical imaging, healthcare and beyond.
    
[^8]: 深层ReLU网络的多面体异常简单

    Deep ReLU Networks Have Surprisingly Simple Polytopes. (arXiv:2305.09145v1 [cs.LG])

    [http://arxiv.org/abs/2305.09145](http://arxiv.org/abs/2305.09145)

    本文通过计算和分析ReLU网络多面体的单纯形直方图，发现在初始化和梯度下降时它们结构相对简单，这说明了一种新的隐式偏见。

    

    ReLU网络是一种多面体上的分段线性函数。研究这种多面体的性质对于神经网络的研究和发展至关重要。目前，对于多面体的理论和实证研究仅停留在计算数量的水平，这远远不能完整地描述多面体。为了将特征提升到一个新的水平，我们提出通过三角剖分多面体得出多面体的形状。通过计算和分析不同多面体的单纯形直方图，我们发现ReLU网络在初始化和梯度下降时具有相对简单的多面体结构，尽管这些多面体从理论上来说可以非常丰富和复杂。这一发现可以被认为是一种新的隐式偏见。随后，我们使用非平凡的组合推导来理论上解释为什么增加深度不会创建更复杂的多面体，通过限制每个维度的平均单纯形数量。

    A ReLU network is a piecewise linear function over polytopes. Figuring out the properties of such polytopes is of fundamental importance for the research and development of neural networks. So far, either theoretical or empirical studies on polytopes only stay at the level of counting their number, which is far from a complete characterization of polytopes. To upgrade the characterization to a new level, here we propose to study the shapes of polytopes via the number of simplices obtained by triangulating the polytope. Then, by computing and analyzing the histogram of simplices across polytopes, we find that a ReLU network has relatively simple polytopes under both initialization and gradient descent, although these polytopes theoretically can be rather diverse and complicated. This finding can be appreciated as a novel implicit bias. Next, we use nontrivial combinatorial derivation to theoretically explain why adding depth does not create a more complicated polytope by bounding the av
    
[^9]: NeuroBench：通过合作、公平和代表性基准测试推进神经形态计算

    NeuroBench: Advancing Neuromorphic Computing through Collaborative, Fair and Representative Benchmarking. (arXiv:2304.04640v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2304.04640](http://arxiv.org/abs/2304.04640)

    NeuroBench是由学术界和工业界成员共同开发的一套协作、公平和代表性的基准测试，可以解决神经形态计算中缺乏清晰标准的问题，推动该领域的发展。

    

    神经形态计算领域在遵循仿生学原理的基础上，具有推进计算效率和能力的巨大潜力。然而，神经形态研究中采用的技术多样性导致缺乏清晰的基准测试标准，阻碍了对神经形态方法与传统基于深度学习的方法的优劣势进行有效评估。本文提出了一个协作项目——NeuroBench，将学术界和工业界成员聚集起来为神经形态计算定义基准测试。NeuroBench的目标是成为社区开发的协作、公平和代表性的基准测试套件。本文讨论了基准测试神经形态解决方案面临的挑战，并概述了NeuroBench的关键特性。我们相信，NeuroBench将是定义能够统一神经形态计算目标的标准的重要一步。

    The field of neuromorphic computing holds great promise in terms of advancing computing efficiency and capabilities by following brain-inspired principles. However, the rich diversity of techniques employed in neuromorphic research has resulted in a lack of clear standards for benchmarking, hindering effective evaluation of the advantages and strengths of neuromorphic methods compared to traditional deep-learning-based methods. This paper presents a collaborative effort, bringing together members from academia and the industry, to define benchmarks for neuromorphic computing: NeuroBench. The goals of NeuroBench are to be a collaborative, fair, and representative benchmark suite developed by the community, for the community. In this paper, we discuss the challenges associated with benchmarking neuromorphic solutions, and outline the key features of NeuroBench. We believe that NeuroBench will be a significant step towards defining standards that can unify the goals of neuromorphic comput
    
[^10]: AI增强的重症监护室：通过普适感知革命性地改善患者护理

    AI-Enhanced Intensive Care Unit: Revolutionizing Patient Care with Pervasive Sensing. (arXiv:2303.06252v1 [cs.AI])

    [http://arxiv.org/abs/2303.06252](http://arxiv.org/abs/2303.06252)

    本文介绍了一种AI增强的重症监护室系统，通过普适感知和数据处理，可以改善患者的视觉监测和评估，提高护理质量。

    This paper introduces an AI-enhanced ICU system that improves patient visual monitoring and assessment, and ultimately enhances the quality of care, through pervasive sensing and data processing.

    重症监护室（ICU）是一个专门的医院空间，用于接受危重病人的密集护理和监测。全面的监测对于评估患者的病情，特别是病情的严重程度以及最终的护理质量至关重要。然而，由于时间限制和医护人员的工作量，ICU中的患者监测范围受到限制。目前，包括面部表情、姿势和活动能力等细节的视觉评估仅偶尔被捕捉到，或者根本没有被捕捉到。这些手动观察是主观的，容易出现文档错误，并给护理人员带来额外的工作量。人工智能（AI）系统由于其出色的学习能力，有潜力增强患者的视觉监测和评估。这样的系统需要强大的注释数据进行训练。为此，我们开发了普适感知和数据处理系统。

    The intensive care unit (ICU) is a specialized hospital space where critically ill patients receive intensive care and monitoring. Comprehensive monitoring is imperative in assessing patients conditions, in particular acuity, and ultimately the quality of care. However, the extent of patient monitoring in the ICU is limited due to time constraints and the workload on healthcare providers. Currently, visual assessments for acuity, including fine details such as facial expressions, posture, and mobility, are sporadically captured, or not captured at all. These manual observations are subjective to the individual, prone to documentation errors, and overburden care providers with the additional workload. Artificial Intelligence (AI) enabled systems has the potential to augment the patient visual monitoring and assessment due to their exceptional learning capabilities. Such systems require robust annotated data to train. To this end, we have developed pervasive sensing and data processing s
    
[^11]: Huber能量量化

    Huber-energy measure quantization. (arXiv:2212.08162v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2212.08162](http://arxiv.org/abs/2212.08162)

    该论文提出了一种Huber能量量化的算法，用于找到目标概率定律的最佳逼近，通过最小化原测度与量化版本之间的统计距离来实现。该算法已在多维高斯混合物、维纳空间魔方等几个数据库上进行了测试。

    

    我们描述了一种测量量化过程，即一种算法，它通过$Q$个狄拉克函数的总和（$Q$为量化参数），找到目标概率定律（更一般地，为有限变差测度）的最佳逼近。该过程通过将原测度与其量化版本之间的统计距离最小化来实现；该距离基于负定核构建，并且如果必要，可以实时计算并输入随机优化算法（如SGD，Adam等）。我们在理论上研究了最优测量量化器的存在的基本问题，并确定了需要保证合适行为的核属性。我们提出了两个最佳线性无偏（BLUE）估计器，用于平方统计距离，并将它们用于无偏程序HEMQ中，以找到最佳量化。我们在多维高斯混合物、维纳空间魔方等几个数据库上测试了HEMQ

    We describe a measure quantization procedure i.e., an algorithm which finds the best approximation of a target probability law (and more generally signed finite variation measure) by a sum of $Q$ Dirac masses ($Q$ being the quantization parameter). The procedure is implemented by minimizing the statistical distance between the original measure and its quantized version; the distance is built from a negative definite kernel and, if necessary, can be computed on the fly and feed to a stochastic optimization algorithm (such as SGD, Adam, ...). We investigate theoretically the fundamental questions of existence of the optimal measure quantizer and identify what are the required kernel properties that guarantee suitable behavior. We propose two best linear unbiased (BLUE) estimators for the squared statistical distance and use them in an unbiased procedure, called HEMQ, to find the optimal quantization. We test HEMQ on several databases: multi-dimensional Gaussian mixtures, Wiener space cub
    

