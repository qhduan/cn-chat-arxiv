# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PhysORD: A Neuro-Symbolic Approach for Physics-infused Motion Prediction in Off-road Driving](https://arxiv.org/abs/2404.01596) | PhysORD是一种神经符号方法，将物理定律融入神经模型中，显著提高了在越野驾驶中的运动预测泛化能力。 |
| [^2] | [Auxiliary CycleGAN-guidance for Task-Aware Domain Translation from Duplex to Monoplex IHC Images](https://arxiv.org/abs/2403.07389) | 通过引入新的训练设计，从而利用辅助的免疫荧光图像域，我们提出了一种用于从双向到单向IHC图像的任务感知域翻译的方法，该方法在下游分割任务中表现出比基线方法更好的效果。 |
| [^3] | [Ansible Lightspeed: A Code Generation Service for IT Automation](https://arxiv.org/abs/2402.17442) | Ansible Lightspeed是一种基于大型语言模型的服务，专注于将自然语言转换为Ansible代码，为IT自动化领域带来了创新。 |
| [^4] | [Generative Kaleidoscopic Networks](https://arxiv.org/abs/2402.11793) | 发现深层ReLU网络表现出过度泛化现象，利用这一特性设计了“生成万花筒网络”，通过递归映射随机输入噪声生成样本。 |
| [^5] | [DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models](https://arxiv.org/abs/2402.08777) | DNABERT-S是一种专门用于创建物种感知的DNA嵌入的基因组基础模型。为了提高对长读DNA序列的嵌入效果，引入了Manifold Instance Mixup (MI-Mix)对比目标方法来训练模型。 |
| [^6] | [Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning](https://arxiv.org/abs/2402.02500) | 通过广泛实验发现基于点云的方法在机器人学习中表现出更好的性能，特别是在各种预训练和泛化任务中。结果表明，点云观测模态对于复杂机器人任务是有价值的。 |
| [^7] | [Causal Fairness under Unobserved Confounding: A Neural Sensitivity Framework](https://arxiv.org/abs/2311.18460) | 分析了因果公平性对未观察到混杂的敏感性，推导出因果公平性指标的界限，提出神经框架用于学习公平预测，展示了框架的有效性 |
| [^8] | [Explaining Explanations in Probabilistic Logic Programming.](http://arxiv.org/abs/2401.17045) | 该论文介绍了基于概率逻辑编程的解释解释方法，以解决在不透明系统中生成合适解释的困难。 |
| [^9] | [Improving Reinforcement Learning from Human Feedback with Efficient Reward Model Ensemble.](http://arxiv.org/abs/2401.16635) | 本论文提出一种通过高效的奖励模型集成来改进人工反馈强化学习的方法，以解决由于奖励模型预测不准确而导致RLHF输出与人类价值观不一致的问题。 |
| [^10] | [StochGradAdam: Accelerating Neural Networks Training with Stochastic Gradient Sampling.](http://arxiv.org/abs/2310.17042) | StochGradAdam是一种利用随机梯度抽样加速神经网络训练的优化器，通过选择性梯度考虑，能够稳定收敛，提升鲁棒训练。在图像分类和分割任务中表现优异。 |
| [^11] | [Federated Learning with Neural Graphical Models.](http://arxiv.org/abs/2309.11680) | 本研究提出了一种名为FedNGMs的联邦学习框架，利用概率神经图模型来处理多个客户端的数据，并在保持训练数据私密性的同时提升模型准确性。 |
| [^12] | [Knowledge Propagation over Conditional Independence Graphs.](http://arxiv.org/abs/2308.05857) | 这项工作提出了在条件独立图上进行知识传播的算法，并通过在Cora和PubMed数据集上的实验证明了其优于现有方法的效果。 |
| [^13] | [SMARLA: A Safety Monitoring Approach for Deep Reinforcement Learning Agents.](http://arxiv.org/abs/2308.02594) | 本文提出了一种基于机器学习的安全监测方法SMARLA，用于深度强化学习智能体。该方法设计为黑盒子，利用状态抽象减少状态空间，实现对智能体状态的安全违规预测。经验证，SMARLA具有准确的违规预测能力，并可在智能体执行的早期阶段进行预测。 |
| [^14] | [On Existential First Order Queries Inference on Knowledge Graphs.](http://arxiv.org/abs/2304.07063) | 本文阐述了关于知识图谱中存在性一阶查询推理的新方法，提出了一个新数据集，并开发了一种来自模糊逻辑理论的新搜索算法，该算法能够解决新公式，并在现有公式中超过以前的方法。 |
| [^15] | [On the Power of Foundation Models.](http://arxiv.org/abs/2211.16327) | 本文通过范畴论探究了基础模型的能力，提出了具有最小所需能力的基础模型可以通过微调和足够的资源来解决前置任务所定义的类别中的下游任务，并且这种能力可以扩展到任何下游任务，只要允许微调且下游任务可在前置任务定义的范畴中表示。 |

# 详细

[^1]: PhysORD：一种神经符号方法用于越野驾驶中注入物理学的运动预测

    PhysORD: A Neuro-Symbolic Approach for Physics-infused Motion Prediction in Off-road Driving

    [https://arxiv.org/abs/2404.01596](https://arxiv.org/abs/2404.01596)

    PhysORD是一种神经符号方法，将物理定律融入神经模型中，显著提高了在越野驾驶中的运动预测泛化能力。

    

    运动预测对于自主越野驾驶至关重要，但与在道路上驾驶相比，它面临着更多挑战，主要是由于车辆与地形之间复杂的相互作用。传统的基于物理的方法在准确建模动态系统和外部干扰方面遇到困难。相反，基于数据驱动的神经网络需要大量数据集，并且难以明确捕捉基本的物理定律，这很容易导致泛化能力差。通过融合这两种方法的优势，神经符号方法提出了一个有前途的方向。这些方法将物理定律嵌入神经模型中，可能显著提高泛化能力。然而，以往的研究都没有在现实世界的越野驾驶环境中进行评估。为了弥合这一差距，我们提出 PhysORD，这是一种神经符号方法，集成了守恒定律，即欧拉-拉格朗日方程。

    arXiv:2404.01596v1 Announce Type: cross  Abstract: Motion prediction is critical for autonomous off-road driving, however, it presents significantly more challenges than on-road driving because of the complex interaction between the vehicle and the terrain. Traditional physics-based approaches encounter difficulties in accurately modeling dynamic systems and external disturbance. In contrast, data-driven neural networks require extensive datasets and struggle with explicitly capturing the fundamental physical laws, which can easily lead to poor generalization. By merging the advantages of both methods, neuro-symbolic approaches present a promising direction. These methods embed physical laws into neural models, potentially significantly improving generalization capabilities. However, no prior works were evaluated in real-world settings for off-road driving. To bridge this gap, we present PhysORD, a neural-symbolic approach integrating the conservation law, i.e., the Euler-Lagrange equa
    
[^2]: 辅助CycleGAN引导下的从双向到单向IHC图像的任务感知域翻译

    Auxiliary CycleGAN-guidance for Task-Aware Domain Translation from Duplex to Monoplex IHC Images

    [https://arxiv.org/abs/2403.07389](https://arxiv.org/abs/2403.07389)

    通过引入新的训练设计，从而利用辅助的免疫荧光图像域，我们提出了一种用于从双向到单向IHC图像的任务感知域翻译的方法，该方法在下游分割任务中表现出比基线方法更好的效果。

    

    生成模型使得从一个源图像域到一个在训练中未见过的目标域的转换成为可能。虽然Cycle生成对抗网络（GANs）已经被广泛应用，但其中的循环一致性约束依赖于两个域之间存在可逆映射的情况，而在染色单向和双向免疫组化（IHC）检测的图像之间的转换不是这样的。针对从后者到前者的转换，我们提出了一种新颖的训练设计，引入了一种新的约束，利用一组免疫荧光（IF）图像作为辅助的不配对图像域。在下游分割任务上的定量和定性结果显示，相比基线方法，所提出的方法带来了显著的好处。

    arXiv:2403.07389v1 Announce Type: cross  Abstract: Generative models enable the translation from a source image domain where readily trained models are available to a target domain unseen during training. While Cycle Generative Adversarial Networks (GANs) are well established, the associated cycle consistency constrain relies on that an invertible mapping exists between the two domains. This is, however, not the case for the translation between images stained with chromogenic monoplex and duplex immunohistochemistry (IHC) assays. Focusing on the translation from the latter to the first, we propose - through the introduction of a novel training design, an alternative constrain leveraging a set of immunofluorescence (IF) images as an auxiliary unpaired image domain. Quantitative and qualitative results on a downstream segmentation task show the benefit of the proposed method in comparison to baseline approaches.
    
[^3]: Ansible Lightspeed: 一种用于IT自动化的代码生成服务

    Ansible Lightspeed: A Code Generation Service for IT Automation

    [https://arxiv.org/abs/2402.17442](https://arxiv.org/abs/2402.17442)

    Ansible Lightspeed是一种基于大型语言模型的服务，专注于将自然语言转换为Ansible代码，为IT自动化领域带来了创新。

    

    大型语言模型（LLMs）的问世使得创建可提高开发者生产力的工具成为可能，集成开发环境（IDEs）常被用作与LLMs交互的接口。已发布许多这类工具，但几乎全部都专注于通用编程语言，很少关注对IT自动化至关重要的特定领域语言。Ansible是一种基于YAML的IT自动化特定语言。Red Hat Ansible Lightspeed与IBM Watson Code Assistant合作的Ansible Lightspeed是一种基于LLM的服务，专门用于将自然语言转换为Ansible代码。

    arXiv:2402.17442v1 Announce Type: cross  Abstract: The availability of Large Language Models (LLMs) which can generate code, has made it possible to create tools that improve developer productivity. Integrated development environments or IDEs which developers use to write software are often used as an interface to interact with LLMs. Although many such tools have been released, almost all of them focus on general-purpose programming languages. Domain-specific languages, such as those crucial for IT automation, have not received much attention. Ansible is one such YAML-based IT automation-specific language. Red Hat Ansible Lightspeed with IBM Watson Code Assistant, further referred to as Ansible Lightspeed, is an LLM-based service designed explicitly for natural language to Ansible code generation.   In this paper, we describe the design and implementation of the Ansible Lightspeed service and analyze feedback from thousands of real users. We examine diverse performance indicators, clas
    
[^4]: 生成万花筒网络

    Generative Kaleidoscopic Networks

    [https://arxiv.org/abs/2402.11793](https://arxiv.org/abs/2402.11793)

    发现深层ReLU网络表现出过度泛化现象，利用这一特性设计了“生成万花筒网络”，通过递归映射随机输入噪声生成样本。

    

    发现深层ReLU网络（或多层感知器架构）表现出“过度泛化”现象。也就是说，那些在训练过程中没有看到的输入的输出值被映射到了在学习过程中观察到的输出范围附近。换句话说，多层感知器学习了一对多的映射，这种效应在增加层数或多层感知器的深度时更为明显。我们利用了深层ReLU网络的这一特性来设计一个数据集万花筒，称为“生成万花筒网络”。简而言之，如果我们学习一个多层感知器将输入 $x\in\mathbb{R}^D$ 映射到自身 $f_\mathcal{N}(x)\rightarrow x$，那么“万花筒采样”过程将从随机输入噪声 $z\in\mathbb{R}^D$ 开始，并递归地应用 $f_\mathcal{N}(\cdots f_\mathcal{N}(z)\cdots )$。经过燃烧期后，我们开始观察来自输入分布的样本，我们发现更深的

    arXiv:2402.11793v1 Announce Type: cross  Abstract: We discovered that the Deep ReLU networks (or Multilayer Perceptron architecture) demonstrate an 'over-generalization' phenomenon. That is, the output values for the inputs that were not seen during training are mapped close to the output range that were observed during the learning process. In other words, the MLP learns a many-to-one mapping and this effect is more prominent as we increase the number of layers or depth of the MLP. We utilize this property of Deep ReLU networks to design a dataset kaleidoscope, termed as 'Generative Kaleidoscopic Networks'. Briefly, if we learn a MLP to map from input $x\in\mathbb{R}^D$ to itself $f_\mathcal{N}(x)\rightarrow x$, the 'Kaleidoscopic sampling' procedure starts with a random input noise $z\in\mathbb{R}^D$ and recursively applies $f_\mathcal{N}(\cdots f_\mathcal{N}(z)\cdots )$. After a burn-in period duration, we start observing samples from the input distribution and we found that deeper 
    
[^5]: DNABERT-S: 学习具有基因组基础模型的物种感知DNA嵌入

    DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models

    [https://arxiv.org/abs/2402.08777](https://arxiv.org/abs/2402.08777)

    DNABERT-S是一种专门用于创建物种感知的DNA嵌入的基因组基础模型。为了提高对长读DNA序列的嵌入效果，引入了Manifold Instance Mixup (MI-Mix)对比目标方法来训练模型。

    

    有效的DNA嵌入在基因组分析中仍然至关重要，特别是在缺乏用于模型微调的标记数据的情况下，尽管基因组基础模型已经取得了显著进展。一个典型的例子是宏基因组分箱，这是微生物组研究中的一个关键过程，旨在通过来自可能包含成千上万个不同的、通常没有经过表征的物种的复杂混合DNA序列的物种来对DNA序列进行分组。为了填补有效的DNA嵌入模型的缺陷，我们引入了DNABERT-S，这是一个专门用于创建物种感知的DNA嵌入的基因组基础模型。为了鼓励对易出错的长读DNA序列进行有效嵌入，我们引入了Manifold Instance Mixup(MI-Mix)，一种对比目标，它在随机选择的层次中混合DNA序列的隐藏表示，并训练模型以在输出层识别和区分这些混合比例。

    arXiv:2402.08777v1 Announce Type: cross Abstract: Effective DNA embedding remains crucial in genomic analysis, particularly in scenarios lacking labeled data for model fine-tuning, despite the significant advancements in genome foundation models. A prime example is metagenomics binning, a critical process in microbiome research that aims to group DNA sequences by their species from a complex mixture of DNA sequences derived from potentially thousands of distinct, often uncharacterized species. To fill the lack of effective DNA embedding models, we introduce DNABERT-S, a genome foundation model that specializes in creating species-aware DNA embeddings. To encourage effective embeddings to error-prone long-read DNA sequences, we introduce Manifold Instance Mixup (MI-Mix), a contrastive objective that mixes the hidden representations of DNA sequences at randomly selected layers and trains the model to recognize and differentiate these mixed proportions at the output layer. We further enha
    
[^6]: 点云问题:重新思考不同观测空间对机器人学习的影响

    Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning

    [https://arxiv.org/abs/2402.02500](https://arxiv.org/abs/2402.02500)

    通过广泛实验发现基于点云的方法在机器人学习中表现出更好的性能，特别是在各种预训练和泛化任务中。结果表明，点云观测模态对于复杂机器人任务是有价值的。

    

    在这项研究中，我们探讨了不同观测空间对机器人学习的影响，重点关注了三种主要模态：RGB，RGB-D和点云。通过在超过17个不同接触丰富的操作任务上进行广泛实验，涉及两个基准和仿真器，我们观察到了一个显著的趋势：基于点云的方法，即使是最简单的设计，通常在性能上超过了其RGB和RGB-D的对应物。这在从头开始训练和利用预训练的两种情况下都是一致的。此外，我们的研究结果表明，点云观测在相机视角、照明条件、噪声水平和背景外观等各种几何和视觉线索方面，都能提高策略零样本泛化能力。研究结果表明，三维点云是复杂机器人任务中有价值的观测模态。我们将公开所有的代码和检查点，希望我们的观点能帮助解决问题。

    In this study, we explore the influence of different observation spaces on robot learning, focusing on three predominant modalities: RGB, RGB-D, and point cloud. Through extensive experimentation on over 17 varied contact-rich manipulation tasks, conducted across two benchmarks and simulators, we have observed a notable trend: point cloud-based methods, even those with the simplest designs, frequently surpass their RGB and RGB-D counterparts in performance. This remains consistent in both scenarios: training from scratch and utilizing pretraining. Furthermore, our findings indicate that point cloud observations lead to improved policy zero-shot generalization in relation to various geometry and visual clues, including camera viewpoints, lighting conditions, noise levels and background appearance. The outcomes suggest that 3D point cloud is a valuable observation modality for intricate robotic tasks. We will open-source all our codes and checkpoints, hoping that our insights can help de
    
[^7]: 未观察到的混杂下的因果公平性：一种神经敏感性框架

    Causal Fairness under Unobserved Confounding: A Neural Sensitivity Framework

    [https://arxiv.org/abs/2311.18460](https://arxiv.org/abs/2311.18460)

    分析了因果公平性对未观察到混杂的敏感性，推导出因果公平性指标的界限，提出神经框架用于学习公平预测，展示了框架的有效性

    

    机器学习预测中的公平性由于法律、道德和社会原因在实践中被广泛要求。现有工作通常集中在没有未观察到的混杂的设置上，尽管未观察到的混杂可能导致严重违反因果公平性，从而产生不公平的预测。在这项工作中，我们分析了因果公平性对未观察到的混杂的敏感性。我们的贡献有三个方面。首先，我们推导出不同来源的未观察到混杂下因果公平性指标的界限。这使从业者能够检查其机器学习模型对在公平关键应用中的未观察到的混杂的敏感性。其次，我们提出了一种用于学习公平预测的新型神经框架，这使我们能够提供对因果公平性可能由于未观察到的混杂而受到违反的程度的最坏情况保证。第三，我们展示了我们框架的有效性。

    arXiv:2311.18460v2 Announce Type: replace-cross  Abstract: Fairness for machine learning predictions is widely required in practice for legal, ethical, and societal reasons. Existing work typically focuses on settings without unobserved confounding, even though unobserved confounding can lead to severe violations of causal fairness and, thus, unfair predictions. In this work, we analyze the sensitivity of causal fairness to unobserved confounding. Our contributions are three-fold. First, we derive bounds for causal fairness metrics under different sources of unobserved confounding. This enables practitioners to examine the sensitivity of their machine learning models to unobserved confounding in fairness-critical applications. Second, we propose a novel neural framework for learning fair predictions, which allows us to offer worst-case guarantees of the extent to which causal fairness can be violated due to unobserved confounding. Third, we demonstrate the effectiveness of our framewor
    
[^8]: 在概率逻辑编程中解释解释

    Explaining Explanations in Probabilistic Logic Programming. (arXiv:2401.17045v1 [cs.AI])

    [http://arxiv.org/abs/2401.17045](http://arxiv.org/abs/2401.17045)

    该论文介绍了基于概率逻辑编程的解释解释方法，以解决在不透明系统中生成合适解释的困难。

    

    基于人工智能的工具的出现也导致了产生人类可理解的解释的需求。在一些方法中，系统是不透明的（通常被称为“黑盒子”），这使得生成适当的解释变得困难。然而，在概率逻辑编程中，我们考虑了逻辑编程（用于知识表示）和概率（用于建模不确定性）的结合。在这个设置中，可以说模型是可以解释的，这方便了对模型的理解。然而，对于特定的查询，通常的“解释”的概念是与模型的每个随机变量的选择集相关联的。不幸的是，这个集合没有因果结构，实际上，一些选择实际上与所考虑的查询无关。为了克服这些缺点，我们提出了一种基于查询驱动推理定义的解释解释方法。

    The emergence of tools based on artificial intelligence has also led to the need of producing explanations which are understandable by a human being. In some approaches, the system is not transparent (often referred to as a "black box"), making it difficult to generate appropriate explanations. In this work, though, we consider probabilistic logic programming, a combination of logic programming (for knowledge representation) and probability (to model uncertainty). In this setting, one can say that models are interpretable, which eases its understanding. However, given a particular query, the usual notion of "explanation" is associated with a set of choices, one for each random variable of the model. Unfortunately, this set does not have a causal structure and, in fact, some of the choices are actually irrelevant to the considered query. In order to overcome these shortcomings, we present an approach to explaining explanations which is based on the definition of a query-driven inference
    
[^9]: 通过高效的奖励模型集成改进人工反馈强化学习

    Improving Reinforcement Learning from Human Feedback with Efficient Reward Model Ensemble. (arXiv:2401.16635v1 [cs.LG])

    [http://arxiv.org/abs/2401.16635](http://arxiv.org/abs/2401.16635)

    本论文提出一种通过高效的奖励模型集成来改进人工反馈强化学习的方法，以解决由于奖励模型预测不准确而导致RLHF输出与人类价值观不一致的问题。

    

    人工反馈强化学习（RLHF）是一种广泛使用的方法，用于将大型语言模型与人类价值观对齐。然而，RLHF依赖于通过有限的人类偏好数据训练的奖励模型，这可能导致不准确的预测。因此，RLHF可能产生与人类价值观不一致的输出。为了缓解这个问题，我们提出了一种奖励集成方法，可以使奖励模型做出更准确的预测。考虑到使用基于大型语言模型的奖励模型集成可能具有计算和资源昂贵的问题，我们探索了包括线性层集成和基于LoRA的集成在内的高效集成方法。实证上，我们使用我们的集成奖励模型运行Best-of-$n$和Proximal Policy Optimization，并验证我们的集成方法有助于改善RLHF输出的对齐性能。

    Reinforcement Learning from Human Feedback (RLHF) is a widely adopted approach for aligning large language models with human values. However, RLHF relies on a reward model that is trained with a limited amount of human preference data, which could lead to inaccurate predictions. As a result, RLHF may produce outputs that are misaligned with human values. To mitigate this issue, we contribute a reward ensemble method that allows the reward model to make more accurate predictions. As using an ensemble of large language model-based reward models can be computationally and resource-expensive, we explore efficient ensemble methods including linear-layer ensemble and LoRA-based ensemble. Empirically, we run Best-of-$n$ and Proximal Policy Optimization with our ensembled reward models, and verify that our ensemble methods help improve the alignment performance of RLHF outputs.
    
[^10]: StochGradAdam: 利用随机梯度抽样加速神经网络训练

    StochGradAdam: Accelerating Neural Networks Training with Stochastic Gradient Sampling. (arXiv:2310.17042v1 [cs.LG])

    [http://arxiv.org/abs/2310.17042](http://arxiv.org/abs/2310.17042)

    StochGradAdam是一种利用随机梯度抽样加速神经网络训练的优化器，通过选择性梯度考虑，能够稳定收敛，提升鲁棒训练。在图像分类和分割任务中表现优异。

    

    在深度学习优化领域中，本文介绍了StochGradAdam优化器，这是对广受赞誉的Adam算法的新颖改进。StochGradAdam的核心是其梯度抽样技术。该方法不仅确保稳定收敛，而且利用选择性梯度考虑的优势，通过减轻噪声或异常数据的影响和增强损失函数空间的探索，提升了鲁棒训练。在图像分类和分割任务中，StochGradAdam表现出优于传统Adam优化器的性能。通过在每次迭代中精心选择一部分梯度进行抽样，该优化器能够有效应对复杂模型的管理。本文从数学基础到偏差校正策略全面探讨了StochGradAdam的方法，展示了深度学习训练技术的可期进展。

    In the rapidly advancing domain of deep learning optimization, this paper unveils the StochGradAdam optimizer, a novel adaptation of the well-regarded Adam algorithm. Central to StochGradAdam is its gradient sampling technique. This method not only ensures stable convergence but also leverages the advantages of selective gradient consideration, fostering robust training by potentially mitigating the effects of noisy or outlier data and enhancing the exploration of the loss landscape for more dependable convergence. In both image classification and segmentation tasks, StochGradAdam has demonstrated superior performance compared to the traditional Adam optimizer. By judiciously sampling a subset of gradients at each iteration, the optimizer is optimized for managing intricate models. The paper provides a comprehensive exploration of StochGradAdam's methodology, from its mathematical foundations to bias correction strategies, heralding a promising advancement in deep learning training tec
    
[^11]: 具有神经图模型的联邦学习

    Federated Learning with Neural Graphical Models. (arXiv:2309.11680v1 [cs.LG])

    [http://arxiv.org/abs/2309.11680](http://arxiv.org/abs/2309.11680)

    本研究提出了一种名为FedNGMs的联邦学习框架，利用概率神经图模型来处理多个客户端的数据，并在保持训练数据私密性的同时提升模型准确性。

    

    联邦学习（FL）解决了在多个客户端保留对数据的独占控制的同时，基于专有数据创建模型的需求。近期提出的神经图模型（NGMs）是概率图模型，利用神经网络的表达能力学习输入特征之间的复杂非线性依赖关系。它们学会捕捉底层的数据分布，并具有高效的推理和采样算法。我们开发了一个FL框架，它维护一个全局的NGM模型，从本地NGM模型中学习到平均信息，同时保持训练数据在客户端的环境中。我们的设计FedNGMs避免了神经元匹配框架（如联邦匹配平均）中模型参数爆炸的缺点和不足。我们的全局模型大小在整个过程中保持不变。

    Federated Learning (FL) addresses the need to create models based on proprietary data in such a way that multiple clients retain exclusive control over their data, while all benefit from improved model accuracy due to pooled resources. Recently proposed Neural Graphical Models (NGMs) are Probabilistic Graphical models that utilize the expressive power of neural networks to learn complex non-linear dependencies between the input features. They learn to capture the underlying data distribution and have efficient algorithms for inference and sampling. We develop a FL framework which maintains a global NGM model that learns the averaged information from the local NGM models while keeping the training data within the client's environment. Our design, FedNGMs, avoids the pitfalls and shortcomings of neuron matching frameworks like Federated Matched Averaging that suffers from model parameter explosion. Our global model size remains constant throughout the process. In the cases where clients 
    
[^12]: 在条件独立图上的知识传播

    Knowledge Propagation over Conditional Independence Graphs. (arXiv:2308.05857v1 [cs.AI])

    [http://arxiv.org/abs/2308.05857](http://arxiv.org/abs/2308.05857)

    这项工作提出了在条件独立图上进行知识传播的算法，并通过在Cora和PubMed数据集上的实验证明了其优于现有方法的效果。

    

    条件独立（CI）图是一种特殊类型的概率图模型（PGM），其中特征连接使用无向图建模，边权重表示特征之间的部分相关性强度。由于CI图捕捉了特征之间的直接依赖关系，它们在研究社区中引起了越来越多的关注，特别是在发现领域拓扑方面。在这项工作中，我们提出了在CI图上执行知识传播的算法。我们的实验证明，我们的技术在公开的Cora和PubMed数据集上超过了最先进的方法。

    Conditional Independence (CI) graph is a special type of a Probabilistic Graphical Model (PGM) where the feature connections are modeled using an undirected graph and the edge weights show the partial correlation strength between the features. Since the CI graphs capture direct dependence between features, they have been garnering increasing interest within the research community for gaining insights into the systems from various domains, in particular discovering the domain topology. In this work, we propose algorithms for performing knowledge propagation over the CI graphs. Our experiments demonstrate that our techniques improve upon the state-of-the-art on the publicly available Cora and PubMed datasets.
    
[^13]: SMARLA：一种用于深度强化学习智能体的安全监测方法

    SMARLA: A Safety Monitoring Approach for Deep Reinforcement Learning Agents. (arXiv:2308.02594v1 [cs.LG])

    [http://arxiv.org/abs/2308.02594](http://arxiv.org/abs/2308.02594)

    本文提出了一种基于机器学习的安全监测方法SMARLA，用于深度强化学习智能体。该方法设计为黑盒子，利用状态抽象减少状态空间，实现对智能体状态的安全违规预测。经验证，SMARLA具有准确的违规预测能力，并可在智能体执行的早期阶段进行预测。

    

    深度强化学习算法(DRL)越来越多地应用于安全关键系统。确保DRL智能体的安全性在这种情况下是一个关键问题。然而，仅依靠测试是不足以确保安全性的，因为它不能提供保证。构建安全监测器是缓解这一挑战的一种解决方案。本文提出了SMARLA，一种基于机器学习的安全监测方法，专为DRL智能体设计。出于实际原因，SMARLA被设计为黑盒子(因为它不需要访问智能体的内部)，并利用状态抽象来减少状态空间，从而促进从智能体的状态学习安全违规预测模型。我们在两个知名的RL案例研究中验证了SMARLA。经验分析表明，SMARLA具有准确的违规预测能力，误报率低，并且可以在智能体执行的一半左右的早期阶段预测安全违规。

    Deep reinforcement learning algorithms (DRL) are increasingly being used in safety-critical systems. Ensuring the safety of DRL agents is a critical concern in such contexts. However, relying solely on testing is not sufficient to ensure safety as it does not offer guarantees. Building safety monitors is one solution to alleviate this challenge. This paper proposes SMARLA, a machine learning-based safety monitoring approach designed for DRL agents. For practical reasons, SMARLA is designed to be black-box (as it does not require access to the internals of the agent) and leverages state abstraction to reduce the state space and thus facilitate the learning of safety violation prediction models from agent's states. We validated SMARLA on two well-known RL case studies. Empirical analysis reveals that SMARLA achieves accurate violation prediction with a low false positive rate, and can predict safety violations at an early stage, approximately halfway through the agent's execution before 
    
[^14]: 关于知识图谱中存在性一阶查询推理的研究

    On Existential First Order Queries Inference on Knowledge Graphs. (arXiv:2304.07063v1 [cs.AI])

    [http://arxiv.org/abs/2304.07063](http://arxiv.org/abs/2304.07063)

    本文阐述了关于知识图谱中存在性一阶查询推理的新方法，提出了一个新数据集，并开发了一种来自模糊逻辑理论的新搜索算法，该算法能够解决新公式，并在现有公式中超过以前的方法。

    

    知识图谱推理是一项具有挑战性的任务，因为它利用观察到的信息来预测缺失的信息。特别地，回答一阶逻辑公式是特别感兴趣的，因为它具有清晰的语法和语义。最近，提出了查询嵌入方法，该方法学习了一组实体的嵌入，并将逻辑运算视为集合运算。尽管有很多研究遵循相同的方法，但它缺乏从逻辑角度进行系统检查的方法。在本文中，我们描述了先前研究调查的查询范围，并准确地确定了它与整个存在性公式家族之间的差距。此外，我们还开发了一个包含十个新公式的新数据集，并讨论了同时出现的新挑战。最后，我们提出了一种来自模糊逻辑理论的新搜索算法，该算法能够解决新公式，并在现有公式中超过以前的方法。

    Reasoning on knowledge graphs is a challenging task because it utilizes observed information to predict the missing one. Specifically, answering first-order logic formulas is of particular interest because of its clear syntax and semantics. Recently, the query embedding method has been proposed which learns the embedding of a set of entities and treats logic operations as set operations. Though there has been much research following the same methodology, it lacks a systematic inspection from the standpoint of logic. In this paper, we characterize the scope of queries investigated previously and precisely identify the gap between it and the whole family of existential formulas. Moreover, we develop a new dataset containing ten new formulas and discuss the new challenges coming simultaneously. Finally, we propose a new search algorithm from fuzzy logic theory which is capable of solving new formulas and outperforming the previous methods in existing formulas.
    
[^15]: 基础模型的能力探究

    On the Power of Foundation Models. (arXiv:2211.16327v3 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2211.16327](http://arxiv.org/abs/2211.16327)

    本文通过范畴论探究了基础模型的能力，提出了具有最小所需能力的基础模型可以通过微调和足够的资源来解决前置任务所定义的类别中的下游任务，并且这种能力可以扩展到任何下游任务，只要允许微调且下游任务可在前置任务定义的范畴中表示。

    

    如果基础模型具有无限高质量的数据点、无限计算能力、一个无限大的完美训练算法、以及在预设任务上保证零泛化误差，那么它可以用于一切吗？传统的表示、优化或泛化理论无法回答这个问题，因为它们主要探讨的问题在这里都是不存在的。本文提出范畴论提供了强大的理论工具，以回答这个问题。我们证明了三个结果，第一个限制了基于提示的学习的能力，即仅当任务可表示时，模型才能用提示解决下游任务；第二个结果表明，微调不受这个限制，因为一个具有最小所需能力（对称性）的基础模型可以通过微调和足够的资源来理论上解决前置任务所定义的类别中的下游任务。我们的最终结果可以看作是第二个结果的一般化，表明如果允许微调并且下游任务可在前置任务定义的范畴中表示，则基础模型的最小能力也足以解决任何下游任务。

    With infinitely many high-quality data points, infinite computational power, an infinitely large foundation model with a perfect training algorithm and guaranteed zero generalization error on the pretext task, can the model be used for everything? This question cannot be answered by the existing theory of representation, optimization or generalization, because the issues they mainly investigate are assumed to be nonexistent here. In this paper, we show that category theory provides powerful machinery to answer this question. We have proved three results. The first one limits the power of prompt-based learning, saying that the model can solve a downstream task with prompts if and only if the task is representable. The second one says fine tuning does not have this limit, as a foundation model with the minimum required power (up to symmetry) can theoretically solve downstream tasks for the category defined by pretext task, with fine tuning and enough resources. Our final result can be se
    

