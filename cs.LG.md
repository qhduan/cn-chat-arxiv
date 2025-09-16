# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semantic Augmentation in Images using Language](https://arxiv.org/abs/2404.02353) | 深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。 |
| [^2] | [LNPT: Label-free Network Pruning and Training](https://arxiv.org/abs/2403.12690) | 本文介绍了LNPT，一种无标签网络修剪和训练的新框架，通过引入学习差距的概念，强调其准确相关性，以解决在智能设备上确定修剪结构的难题。 |
| [^3] | [SEVEN: Pruning Transformer Model by Reserving Sentinels](https://arxiv.org/abs/2403.12688) | SEVEN通过保留梯度噪声较小的权重，在剪枝Transformer模型时取得了优异的效果。 |
| [^4] | [Early alignment in two-layer networks training is a two-edged sword.](http://arxiv.org/abs/2401.10791) | 本文研究了两层网络训练中的早期对齐现象，发现在小初始化和一个隐藏的ReLU层网络中，神经元会在训练的早期阶段向关键方向进行对齐，导致网络稀疏表示以及梯度流在收敛时的隐含偏好。然而，这种稀疏诱导的对齐也使得训练目标的最小化变得困难。 |
| [^5] | [Scalable manifold learning by uniform landmark sampling and constrained locally linear embedding.](http://arxiv.org/abs/2401.01100) | 通过均匀地标抽样和约束局部线性嵌入，提出了一种可伸缩的流形学习方法，可以有效处理大规模和高维数据，并解决全局结构失真和可伸缩性问题。 |
| [^6] | [Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling.](http://arxiv.org/abs/2310.06389) | 本研究提出了乐高积木，通过集成局部特征丰富和全局内容协调，实现了高效且可自适应的迭代细化扩散建模。这些积木可以堆叠在一起，用于在测试时根据需要进行重构，从而减少采样成本并生成高分辨率图像。 |
| [^7] | [Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models.](http://arxiv.org/abs/2309.01219) | 本文调查了大型语言模型中幻觉的检测、解释和缓解的最新研究，提出了幻觉现象和评估基准的分类，并讨论了未来研究的潜在方向。 |
| [^8] | [Sub-universal variational circuits for combinatorial optimization problems.](http://arxiv.org/abs/2308.14981) | 本研究提出了一种基于经典概率电路的变分电路，用于解决组合优化问题。通过对Max-Cut问题的数值研究，我们发现这种变分电路在多种图上表现出更好的性能，相比于量子近似优化算法。在评估量子变分电路的性能时，可以将其与具有子通用门集的变分电路进行比较，以识别量子变分电路的优势领域。 |
| [^9] | [Calibration in Deep Learning: A Survey of the State-of-the-Art.](http://arxiv.org/abs/2308.01222) | 本文回顾了深度学习中的校准方法的最新发展，并提供了对其原理的理解。研究表明，现代深度神经网络在预测能力上表现出色，但校准性较差，导致模型预测不可靠。因此，需要一些新的方法来改善模型的校准性。 |
| [^10] | [Data-Induced Interactions of Sparse Sensors.](http://arxiv.org/abs/2307.11838) | 本研究通过采用热力学观点，用统计物理学中的Ising模型来计算由训练数据引发的稀疏传感器之间的相互作用，从而优化传感器的空间配置和重构复杂系统的完整状态。 |
| [^11] | [Explaining Emergent In-Context Learning as Kernel Regression.](http://arxiv.org/abs/2305.12766) | 本文研究了为什么在预训练之后，基于Transformer的语言模型能够实现上下文学习，并提出了一种假设，认为LLMs在面对上下文示例时能够通过内部表示模拟核回归。 |
| [^12] | [When Deep Learning Meets Polyhedral Theory: A Survey.](http://arxiv.org/abs/2305.00241) | 本文综述了深度学习与多面体理论的交叉领域。修正线性单元（ReLU）等函数使得一些神经网络结构能够通过多面体理论进行分析，应用线性和混合整数线性规划来实现网络修剪、鲁棒性分析和神经网络验证等任务。 |
| [^13] | [Piecewise Deterministic Markov Processes for Bayesian Neural Networks.](http://arxiv.org/abs/2302.08724) | 本文介绍了基于分段确定性马尔可夫过程的贝叶斯神经网络推理方法，通过引入新的自适应稀疏方案，实现了对困难采样问题的加速处理。实验证明，这种方法在计算上可行，并能提高预测准确性、MCMC混合性能，并提供更有信息量的不确定性测量。 |

# 详细

[^1]: 利用语言在图像中进行语义增强

    Semantic Augmentation in Images using Language

    [https://arxiv.org/abs/2404.02353](https://arxiv.org/abs/2404.02353)

    深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。

    

    深度学习模型需要非常庞大的标记数据集进行监督学习，缺乏这些数据集会导致过拟合并限制其泛化到现实世界示例的能力。最近扩散模型的进展使得能够基于文本输入生成逼真的图像。利用用于训练这些扩散模型的大规模数据集，我们提出一种利用生成的图像来增强现有数据集的技术。本文探讨了各种有效数据增强策略，以提高深度学习模型的跨领域泛化能力。

    arXiv:2404.02353v1 Announce Type: cross  Abstract: Deep Learning models are incredibly data-hungry and require very large labeled datasets for supervised learning. As a consequence, these models often suffer from overfitting, limiting their ability to generalize to real-world examples. Recent advancements in diffusion models have enabled the generation of photorealistic images based on textual inputs. Leveraging the substantial datasets used to train these diffusion models, we propose a technique to utilize generated images to augment existing datasets. This paper explores various strategies for effective data augmentation to improve the out-of-domain generalization capabilities of deep learning models.
    
[^2]: LNPT：无标签网络修剪与训练

    LNPT: Label-free Network Pruning and Training

    [https://arxiv.org/abs/2403.12690](https://arxiv.org/abs/2403.12690)

    本文介绍了LNPT，一种无标签网络修剪和训练的新框架，通过引入学习差距的概念，强调其准确相关性，以解决在智能设备上确定修剪结构的难题。

    

    在训练之前修剪神经网络，使其能够部署在智能设备上。通过保留有助于泛化的权重，修剪后的网络可以在资源受限的智能设备上运行。我们提出了学习差距的概念，并强调它与泛化的准确相关性。实验表明，学习差距通过网络倒数第二层的特征图形式与泛化性能的变化相一致。我们提出了一种新的学习框架 LNPT，使得云端成熟网络能够提供在线指导。

    arXiv:2403.12690v1 Announce Type: new  Abstract: Pruning before training enables the deployment of neural networks on smart devices. By retaining weights conducive to generalization, pruned networks can be accommodated on resource-constrained smart devices. It is commonly held that the distance on weight norms between the initialized and the fully-trained networks correlates with generalization performance. However, as we have uncovered, inconsistency between this metric and generalization during training processes, which poses an obstacle to determine the pruned structures on smart devices in advance. In this paper, we introduce the concept of the learning gap, emphasizing its accurate correlation with generalization. Experiments show that the learning gap, in the form of feature maps from the penultimate layer of networks, aligns with variations of generalization performance. We propose a novel learning framework, LNPT, which enables mature networks on the cloud to provide online gui
    
[^3]: SEVEN: 通过保留哨兵来剪枝Transformer模型

    SEVEN: Pruning Transformer Model by Reserving Sentinels

    [https://arxiv.org/abs/2403.12688](https://arxiv.org/abs/2403.12688)

    SEVEN通过保留梯度噪声较小的权重，在剪枝Transformer模型时取得了优异的效果。

    

    大规模Transformer模型已经在各种任务中展现出卓越的性能。然而，由于其可观的参数规模，它们的适用性受到限制，尤其是在移动设备上。鉴于Transformer模型相对于卷积神经网络的梯度是动态且错综复杂的，常用的剪枝方法往往会保留具有较大梯度噪声的权重。这导致被剪枝的模型对稀疏性和数据集敏感，表现出次优性能。符号下降（SD）是一种用于训练和微调Transformer模型的通用方法。在本文中，我们试图通过SD的累积过程描述Transformer模型上的噪声批梯度序列。我们利用这一设计动态评估权重的重要性分数。我们引入了SEVEN，特别偏向于具有持续高敏感度的权重，即梯度噪声较小的权重。

    arXiv:2403.12688v1 Announce Type: new  Abstract: Large-scale Transformer models (TM) have demonstrated outstanding performance across various tasks. However, their considerable parameter size restricts their applicability, particularly on mobile devices. Due to the dynamic and intricate nature of gradients on TM compared to Convolutional Neural Networks, commonly used pruning methods tend to retain weights with larger gradient noise. This results in pruned models that are sensitive to sparsity and datasets, exhibiting suboptimal performance. Symbolic Descent (SD) is a general approach for training and fine-tuning TM. In this paper, we attempt to describe the noisy batch gradient sequences on TM through the cumulative process of SD. We utilize this design to dynamically assess the importance scores of weights.SEVEN is introduced by us, which particularly favors weights with consistently high sensitivity, i.e., weights with small gradient noise. These weights are tended to be preserved b
    
[^4]: 两层网络训练中的早期对齐是一把双刃剑

    Early alignment in two-layer networks training is a two-edged sword. (arXiv:2401.10791v1 [cs.LG])

    [http://arxiv.org/abs/2401.10791](http://arxiv.org/abs/2401.10791)

    本文研究了两层网络训练中的早期对齐现象，发现在小初始化和一个隐藏的ReLU层网络中，神经元会在训练的早期阶段向关键方向进行对齐，导致网络稀疏表示以及梯度流在收敛时的隐含偏好。然而，这种稀疏诱导的对齐也使得训练目标的最小化变得困难。

    

    使用一阶优化方法训练神经网络是深度学习成功的核心。初始化的规模是一个关键因素，因为小的初始化通常与特征学习模式相关，在这种模式下，梯度下降对简单解隐含偏好。本文提供了早期对齐阶段的普遍和量化描述，最初由Maennel等人提出。对于小初始化和一个隐藏的ReLU层网络，训练动态的早期阶段导致神经元向关键方向进行对齐。这种对齐引发了网络的稀疏表示，这与梯度流在收敛时的隐含偏好直接相关。然而，这种稀疏诱导的对齐是以在最小化训练目标方面遇到困难为代价的：我们还提供了一个简单的数据示例，其中超参数网络无法收敛到全局最小值。

    Training neural networks with first order optimisation methods is at the core of the empirical success of deep learning. The scale of initialisation is a crucial factor, as small initialisations are generally associated to a feature learning regime, for which gradient descent is implicitly biased towards simple solutions. This work provides a general and quantitative description of the early alignment phase, originally introduced by Maennel et al. (2018) . For small initialisation and one hidden ReLU layer networks, the early stage of the training dynamics leads to an alignment of the neurons towards key directions. This alignment induces a sparse representation of the network, which is directly related to the implicit bias of gradient flow at convergence. This sparsity inducing alignment however comes at the expense of difficulties in minimising the training objective: we also provide a simple data example for which overparameterised networks fail to converge towards global minima and
    
[^5]: 通过均匀地标抽样和约束局部线性嵌入实现可伸缩的流形学习

    Scalable manifold learning by uniform landmark sampling and constrained locally linear embedding. (arXiv:2401.01100v1 [cs.LG])

    [http://arxiv.org/abs/2401.01100](http://arxiv.org/abs/2401.01100)

    通过均匀地标抽样和约束局部线性嵌入，提出了一种可伸缩的流形学习方法，可以有效处理大规模和高维数据，并解决全局结构失真和可伸缩性问题。

    

    流形学习是机器学习和数据科学中的关键方法，旨在揭示高维空间中复杂非线性流形内在的低维结构。通过利用流形假设，已经开发了各种非线性降维技术来促进可视化、分类、聚类和获得关键洞察。虽然现有的流形学习方法取得了显著的成功，但仍然存在全局结构中的大量失真问题，这阻碍了对底层模式的理解。可伸缩性问题也限制了它们处理大规模数据的适用性。在这里，我们提出了一种可伸缩的流形学习(scML)方法，可以以有效的方式处理大规模和高维数据。它通过寻找一组地标来构建整个数据的低维骨架，然后将非地标引入地标空间中

    As a pivotal approach in machine learning and data science, manifold learning aims to uncover the intrinsic low-dimensional structure within complex nonlinear manifolds in high-dimensional space. By exploiting the manifold hypothesis, various techniques for nonlinear dimension reduction have been developed to facilitate visualization, classification, clustering, and gaining key insights. Although existing manifold learning methods have achieved remarkable successes, they still suffer from extensive distortions incurred in the global structure, which hinders the understanding of underlying patterns. Scalability issues also limit their applicability for handling large-scale data. Here, we propose a scalable manifold learning (scML) method that can manipulate large-scale and high-dimensional data in an efficient manner. It starts by seeking a set of landmarks to construct the low-dimensional skeleton of the entire data and then incorporates the non-landmarks into the landmark space based 
    
[^6]: 学习可堆叠和可跳过的乐高积木以实现高效、可重构和可变分辨率的扩散建模

    Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling. (arXiv:2310.06389v1 [cs.CV])

    [http://arxiv.org/abs/2310.06389](http://arxiv.org/abs/2310.06389)

    本研究提出了乐高积木，通过集成局部特征丰富和全局内容协调，实现了高效且可自适应的迭代细化扩散建模。这些积木可以堆叠在一起，用于在测试时根据需要进行重构，从而减少采样成本并生成高分辨率图像。

    

    扩散模型在生成真实感图像方面表现出色，但在训练和采样方面具有显著的计算成本。尽管有各种技术来解决这些计算挑战，但一个较少探索的问题是设计一个高效且适应性强的网络骨干，用于迭代细化。当前的选项如U-Net和Vision Transformer通常依赖于资源密集型的深度网络，缺乏在变量分辨率下生成图像或使用比训练中更小的网络所需的灵活性。本研究引入了乐高积木，它们无缝集成了局部特征丰富和全局内容协调。这些积木可以堆叠在一起，创建一个测试时可重构的扩散骨干，允许选择性跳过积木以减少采样成本，并生成比训练数据更高分辨率的图像。乐高积木通过MLP对局部区域进行丰富，并使用Transformer块进行变换，同时保持一致的全分辨率

    Diffusion models excel at generating photo-realistic images but come with significant computational costs in both training and sampling. While various techniques address these computational challenges, a less-explored issue is designing an efficient and adaptable network backbone for iterative refinement. Current options like U-Net and Vision Transformer often rely on resource-intensive deep networks and lack the flexibility needed for generating images at variable resolutions or with a smaller network than used in training. This study introduces LEGO bricks, which seamlessly integrate Local-feature Enrichment and Global-content Orchestration. These bricks can be stacked to create a test-time reconfigurable diffusion backbone, allowing selective skipping of bricks to reduce sampling costs and generate higher-resolution images than the training data. LEGO bricks enrich local regions with an MLP and transform them using a Transformer block while maintaining a consistent full-resolution i
    
[^7]: AI海洋中的妖怪之歌：大型语言模型中的幻觉调查

    Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models. (arXiv:2309.01219v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.01219](http://arxiv.org/abs/2309.01219)

    本文调查了大型语言模型中幻觉的检测、解释和缓解的最新研究，提出了幻觉现象和评估基准的分类，并讨论了未来研究的潜在方向。

    

    尽管大型语言模型（LLMs）在各种下游任务中展示出了卓越的能力，但人们对其产生幻觉的倾向表示担忧：LLMs有时会生成与用户输入不符、与先前生成的内容相矛盾或与已建立的世界知识不符的内容。这种现象对LLMs在现实场景中的可靠性构成了重大挑战。本文对关于幻觉检测、解释和缓解的最新研究进行了调查，重点探讨了LLMs所面临的独特挑战。我们提出了LLM幻觉现象和评估基准的分类，分析了现有的旨在缓解LLM幻觉的方法，并讨论了未来研究的潜在方向。

    While large language models (LLMs) have demonstrated remarkable capabilities across a range of downstream tasks, a significant concern revolves around their propensity to exhibit hallucinations: LLMs occasionally generate content that diverges from the user input, contradicts previously generated context, or misaligns with established world knowledge. This phenomenon poses a substantial challenge to the reliability of LLMs in real-world scenarios. In this paper, we survey recent efforts on the detection, explanation, and mitigation of hallucination, with an emphasis on the unique challenges posed by LLMs. We present taxonomies of the LLM hallucination phenomena and evaluation benchmarks, analyze existing approaches aiming at mitigating LLM hallucination, and discuss potential directions for future research.
    
[^8]: 子通用变分电路用于组合优化问题的翻译论文

    Sub-universal variational circuits for combinatorial optimization problems. (arXiv:2308.14981v1 [quant-ph])

    [http://arxiv.org/abs/2308.14981](http://arxiv.org/abs/2308.14981)

    本研究提出了一种基于经典概率电路的变分电路，用于解决组合优化问题。通过对Max-Cut问题的数值研究，我们发现这种变分电路在多种图上表现出更好的性能，相比于量子近似优化算法。在评估量子变分电路的性能时，可以将其与具有子通用门集的变分电路进行比较，以识别量子变分电路的优势领域。

    

    由于其在量子近似优化算法和量子机器学习研究中的应用，量子变分电路引起了广泛的关注。本研究引入了一种新颖的经典概率电路，用于生成对由二位随机矩阵构建的组合优化问题的近似解。通过数值研究，我们调查了我们提出的变分电路在解决不断增加规模的各种图的Max-Cut问题中的性能。我们的经典算法在多种类型的图上表现出更好的性能，相比于量子近似优化算法。我们的发现表明，将量子变分电路的性能与具有子通用门集的变分电路进行评估，是识别量子变分电路可突出优势领域的有价值的基准。

    Quantum variational circuits have gained significant attention due to their applications in the quantum approximate optimization algorithm and quantum machine learning research. This work introduces a novel class of classical probabilistic circuits designed for generating approximate solutions to combinatorial optimization problems constructed using two-bit stochastic matrices. Through a numerical study, we investigate the performance of our proposed variational circuits in solving the Max-Cut problem on various graphs of increasing sizes. Our classical algorithm demonstrates improved performance for several graph types to the quantum approximate optimization algorithm. Our findings suggest that evaluating the performance of quantum variational circuits against variational circuits with sub-universal gate sets is a valuable benchmark for identifying areas where quantum variational circuits can excel.
    
[^9]: 深度学习中的校准：最新研究综述

    Calibration in Deep Learning: A Survey of the State-of-the-Art. (arXiv:2308.01222v1 [cs.LG])

    [http://arxiv.org/abs/2308.01222](http://arxiv.org/abs/2308.01222)

    本文回顾了深度学习中的校准方法的最新发展，并提供了对其原理的理解。研究表明，现代深度神经网络在预测能力上表现出色，但校准性较差，导致模型预测不可靠。因此，需要一些新的方法来改善模型的校准性。

    

    在构建可靠、鲁棒的安全关键应用的人工智能系统中，深度神经模型的校准起着重要作用。最近的研究表明，具有高预测能力的现代神经网络的校准性较差，产生不可靠的模型预测。尽管深度学习模型在各种基准测试中取得了显著的性能，但对模型的校准性和可靠性的研究相对较少。理想的深度模型不仅应具有高预测性能，还应具有良好的校准性。最近提出了一些使用不同机制进行深度模型校准的方法。在本综述中，我们回顾了最新的校准方法，并解释了它们执行模型校准的原理。首先，我们从模型校准的定义开始，解释了模型校准不准确的根本原因。然后，我们介绍了可以衡量模型校准性的关键指标。接下来，我们总结了一些校准方法的方法和实践。

    Calibrating deep neural models plays an important role in building reliable, robust AI systems in safety-critical applications. Recent work has shown that modern neural networks that possess high predictive capability are poorly calibrated and produce unreliable model predictions. Though deep learning models achieve remarkable performance on various benchmarks, the study of model calibration and reliability is relatively underexplored. Ideal deep models should have not only high predictive performance but also be well calibrated. There have been some recent methods proposed to calibrate deep models by using different mechanisms. In this survey, we review the state-of-the-art calibration methods and provide an understanding of their principles for performing model calibration. First, we start with the definition of model calibration and explain the root causes of model miscalibration. Then we introduce the key metrics that can measure this aspect. It is followed by a summary of calibrat
    
[^10]: 稀疏传感器的数据引发的相互作用

    Data-Induced Interactions of Sparse Sensors. (arXiv:2307.11838v1 [cond-mat.stat-mech])

    [http://arxiv.org/abs/2307.11838](http://arxiv.org/abs/2307.11838)

    本研究通过采用热力学观点，用统计物理学中的Ising模型来计算由训练数据引发的稀疏传感器之间的相互作用，从而优化传感器的空间配置和重构复杂系统的完整状态。

    

    在科学和工程中，大维度的经验数据经常具有低秩结构，并且可以表示为仅由几个特征模式的组合。由于这种结构，我们可以使用仅有少数局部化的传感器测量来重新构建复杂系统的完整状态。这种重构的质量，特别是在传感器噪声存在的情况下，显著取决于传感器的空间配置。已经提出了多种基于缺失插值和QR分解的算法来优化传感器位置。在这里，我们采用热力学观点计算由训练数据引发的传感器相互作用的完整地形。该地形采用统计物理学中的Ising模型的形式，考虑到每个传感器位置捕获的数据方差以及传感器之间的串扰。绘制出这些数据引发的传感器相互作用的图景允许

    Large-dimensional empirical data in science and engineering frequently has low-rank structure and can be represented as a combination of just a few eigenmodes. Because of this structure, we can use just a few spatially localized sensor measurements to reconstruct the full state of a complex system. The quality of this reconstruction, especially in the presence of sensor noise, depends significantly on the spatial configuration of the sensors. Multiple algorithms based on gappy interpolation and QR factorization have been proposed to optimize sensor placement. Here, instead of an algorithm that outputs a singular "optimal" sensor configuration, we take a thermodynamic view to compute the full landscape of sensor interactions induced by the training data. The landscape takes the form of the Ising model in statistical physics, and accounts for both the data variance captured at each sensor location and the crosstalk between sensors. Mapping out these data-induced sensor interactions allow
    
[^11]: 将 Emergent In-Context Learning 解释为核回归

    Explaining Emergent In-Context Learning as Kernel Regression. (arXiv:2305.12766v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12766](http://arxiv.org/abs/2305.12766)

    本文研究了为什么在预训练之后，基于Transformer的语言模型能够实现上下文学习，并提出了一种假设，认为LLMs在面对上下文示例时能够通过内部表示模拟核回归。

    

    大型语言模型（LLMs）在迁移学习中引起了一场范式转变。与经典的预训练-微调过程相比，为了将LLMs用于下游预测任务，只需要提供一些示例，即上下文示例，而无需添加或更新现有的模型参数。LLMs的这种上下文学习能力非常有意思，但目前尚不完全了解预训练LLMs如何获得这种能力。本文通过提出一个假设，即当面临上下文示例时，LLMs能够通过内部表示模拟核回归，来研究为何基于Transformer的语言模型能够在预训练通用语料库之后实现上下文学习。具体来说，我们首先证明了上下文提示的贝叶斯推断在渐近情况下可以被理解为核回归 $\hat y = \sum_i y_i K(x, x_i)/\sum_i K(x, x_i)$，

    Large language models (LLMs) have initiated a paradigm shift in transfer learning. In contrast to the classic pretraining-then-finetuning procedure, in order to use LLMs for downstream prediction tasks, one only needs to provide a few demonstrations, known as in-context examples, without adding more or updating existing model parameters. This in-context learning (ICL) capability of LLMs is intriguing, and it is not yet fully understood how pretrained LLMs acquire such capabilities. In this paper, we investigate the reason why a transformer-based language model can accomplish in-context learning after pre-training on a general language corpus by proposing one hypothesis that LLMs can simulate kernel regression with internal representations when faced with in-context examples. More concretely, we first prove that Bayesian inference on in-context prompts can be asymptotically understood as kernel regression $\hat y = \sum_i y_i K(x, x_i)/\sum_i K(x, x_i)$ as the number of in-context demon
    
[^12]: 当深度学习遇见多面体理论：一项综述

    When Deep Learning Meets Polyhedral Theory: A Survey. (arXiv:2305.00241v1 [math.OC])

    [http://arxiv.org/abs/2305.00241](http://arxiv.org/abs/2305.00241)

    本文综述了深度学习与多面体理论的交叉领域。修正线性单元（ReLU）等函数使得一些神经网络结构能够通过多面体理论进行分析，应用线性和混合整数线性规划来实现网络修剪、鲁棒性分析和神经网络验证等任务。

    

    在过去的十年中，深度学习成为了预测建模的主要方法，得益于深度神经网络在计算机视觉和自然语言处理等任务中的显著准确性。与此同时，神经网络的结构回归到了基于分段常数和分段线性函数的简单表示，例如修正线性单元（ReLU），这种激活函数成为神经网络中最常用的类型。这使得某些类型的网络结构，如典型的全连接前馈神经网络，能够通过多面体理论进行分析，并应用线性规划（LP）和混合整数线性规划（MILP）等方法用于各种目的。本文综述了这个快速发展领域涌现的主要主题，为更详细地了解神经网络以及应用数学提供了新的视角。我们介绍了多面体理论的基础知识以及它与深度学习的关系，并回顾了该主题的最新进展，包括在网络修剪、鲁棒性分析和神经网络验证等任务中使用LP和MILP。最后，我们讨论了当前挑战和未来研究方向。

    In the past decade, deep learning became the prevalent methodology for predictive modeling thanks to the remarkable accuracy of deep neural networks in tasks such as computer vision and natural language processing. Meanwhile, the structure of neural networks converged back to simpler representations based on piecewise constant and piecewise linear functions such as the Rectified Linear Unit (ReLU), which became the most commonly used type of activation function in neural networks. That made certain types of network structure $\unicode{x2014}$such as the typical fully-connected feedforward neural network$\unicode{x2014}$ amenable to analysis through polyhedral theory and to the application of methodologies such as Linear Programming (LP) and Mixed-Integer Linear Programming (MILP) for a variety of purposes. In this paper, we survey the main topics emerging from this fast-paced area of work, which bring a fresh perspective to understanding neural networks in more detail as well as to app
    
[^13]: 基于分段确定性马尔可夫过程的贝叶斯神经网络研究

    Piecewise Deterministic Markov Processes for Bayesian Neural Networks. (arXiv:2302.08724v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.08724](http://arxiv.org/abs/2302.08724)

    本文介绍了基于分段确定性马尔可夫过程的贝叶斯神经网络推理方法，通过引入新的自适应稀疏方案，实现了对困难采样问题的加速处理。实验证明，这种方法在计算上可行，并能提高预测准确性、MCMC混合性能，并提供更有信息量的不确定性测量。

    

    现代贝叶斯神经网络（BNNs）的推理通常依赖于变分推断处理，这要求违反了独立性和后验形式的假设。传统的MCMC方法避免了这些假设，但由于无法适应似然的子采样，导致计算量增加。新的分段确定性马尔可夫过程（PDMP）采样器允许子采样，但引入了模型特定的不均匀泊松过程（IPPs），从中采样困难。本研究引入了一种新的通用自适应稀疏方案，用于从这些IPPs中进行采样，并展示了如何加速将PDMPs应用于BNNs推理。实验表明，使用这些方法进行推理在计算上是可行的，可以提高预测准确性、MCMC混合性能，并与其他近似推理方案相比，提供更有信息量的不确定性测量。

    Inference on modern Bayesian Neural Networks (BNNs) often relies on a variational inference treatment, imposing violated assumptions of independence and the form of the posterior. Traditional MCMC approaches avoid these assumptions at the cost of increased computation due to its incompatibility to subsampling of the likelihood. New Piecewise Deterministic Markov Process (PDMP) samplers permit subsampling, though introduce a model specific inhomogenous Poisson Process (IPPs) which is difficult to sample from. This work introduces a new generic and adaptive thinning scheme for sampling from these IPPs, and demonstrates how this approach can accelerate the application of PDMPs for inference in BNNs. Experimentation illustrates how inference with these methods is computationally feasible, can improve predictive accuracy, MCMC mixing performance, and provide informative uncertainty measurements when compared against other approximate inference schemes.
    

