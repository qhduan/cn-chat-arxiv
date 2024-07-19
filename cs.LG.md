# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NeRF-MAE : Masked AutoEncoders for Self Supervised 3D representation Learning for Neural Radiance Fields](https://arxiv.org/abs/2404.01300) | 通过使用Masked AutoEncoders，本文提出了NeRF-MAE用于自监督三维表示学习，利用标准的三维Vision Transformers适应NeRF的独特公式，将NeRF的体积网格作为密集输入，以产生有效的三维表示。 |
| [^2] | [NeuroPictor: Refining fMRI-to-Image Reconstruction via Multi-individual Pretraining and Multi-level Modulation](https://arxiv.org/abs/2403.18211) | NeuroPictor通过直接调制扩散模型的生成过程，实现了fMRI到图像的重建，在多个个体预训练和多层次的引导条件下，实现了更详细的图像控制。 |
| [^3] | [Discrete Latent Graph Generative Modeling with Diffusion Bridges](https://arxiv.org/abs/2403.16883) | GLAD是一个在离散潜在空间上操作的图生成模型，通过适应扩散桥结构学习其离散潜在空间的先验，避免了依赖于原始数据空间的分解，在图生成任务中表现出优越性。 |
| [^4] | [Simple Graph Condensation](https://arxiv.org/abs/2403.14951) | 提出了一种简化的图压缩方法，旨在减少图神经网络所带来的不必要复杂性。 |
| [^5] | [On Pretraining Data Diversity for Self-Supervised Learning](https://arxiv.org/abs/2403.13808) | 增加预训练数据多样性可以提高自监督学习性能，但仅在与下游数据的分布距离较小时有效。 |
| [^6] | [PARMESAN: Parameter-Free Memory Search and Transduction for Dense Prediction Tasks](https://arxiv.org/abs/2403.11743) | 通过引入转导的概念，提出了PARMESAN，一种用于解决密集预测任务的无参数内存搜索和转导方法，实现了灵活性和无需连续训练的学习。 |
| [^7] | [Not Just Change the Labels, Learn the Features: Watermarking Deep Neural Networks with Multi-View Data](https://arxiv.org/abs/2403.10663) | 通过使用多视角数据为深度神经网络添加水印，可以有效防御对源模型功能的窃取攻击 |
| [^8] | [Attention-based Class-Conditioned Alignment for Multi-Source Domain Adaptive Object Detection](https://arxiv.org/abs/2403.09918) | 提出了一种基于注意力的类别条件对齐方案，用于多源领域自适应目标检测，在跨领域对齐每个对象类别的实例。 |
| [^9] | [Transformers Get Stable: An End-to-End Signal Propagation Theory for Language Models](https://arxiv.org/abs/2403.09635) | 提出了一个统一的信号传播理论，提供了控制transformer模型信号传播的公式，提出了DeepScaleLM初始化和缩放方案，使得可以训练非常深的模型，并发现深层模型在多个任务和数据集上胜过浅层模型。 |
| [^10] | [Scalable Spatiotemporal Prediction with Bayesian Neural Fields](https://arxiv.org/abs/2403.07657) | 该论文提出了贝叶斯神经场（BayesNF），结合了深度神经网络和分层贝叶斯推断，用于处理大规模时空预测问题。 |
| [^11] | [FAX: Scalable and Differentiable Federated Primitives in JAX](https://arxiv.org/abs/2403.07128) | FAX是一个在JAX中嵌入联邦计算原语的库，支持大规模分布式计算，提供了联邦自动微分的实现，并可解释至现有的生产跨设备联邦计算系统。 |
| [^12] | [Semantic Residual Prompts for Continual Learning](https://arxiv.org/abs/2403.06870) | 通过引入语义剩余提示，作者提出了一种稳定的选择策略，利用两级适应机制来在持续学习中解决提示冲突的问题。 |
| [^13] | [GraphRCG: Self-conditioned Graph Generation via Bootstrapped Representations](https://arxiv.org/abs/2403.01071) | 提出了一种自条件图生成框架，通过自引导表示指导生成过程，明确建模和利用图分布，优于传统隐式捕获分布的方法。 |
| [^14] | [Self-Supervised Learning in Electron Microscopy: Towards a Foundation Model for Advanced Image Analysis](https://arxiv.org/abs/2402.18286) | 本文探讨了在电子显微镜中进行自监督学习的潜力，展示自监督预训练如何促进有效的微调，同时指出较低复杂度的模型在微调过程中始终优于更复杂的随机初始化模型。 |
| [^15] | [Feedback Efficient Online Fine-Tuning of Diffusion Models](https://arxiv.org/abs/2402.16359) | 提出了一种反馈高效的在线微调扩散模型的强化学习程序 |
| [^16] | [Mechanics-Informed Autoencoder Enables Automated Detection and Localization of Unforeseen Structural Damage](https://arxiv.org/abs/2402.15492) | 该研究提出了一种新颖的"部署和忘记"方法，结合廉价传感器和机械信息自编码器，实现了对结构损伤的自动检测和定位，仅需学习3小时数据即可自主识别和定位不同类型的未知损伤。 |
| [^17] | [MSPipe: Efficient Temporal GNN Training via Staleness-aware Pipeline](https://arxiv.org/abs/2402.15113) | 提出了MSPipe，一个通用而高效的MTGNNs框架，实现了最大化训练吞吐量同时保持模型准确性 |
| [^18] | [QuRating: Selecting High-Quality Data for Training Language Models](https://arxiv.org/abs/2402.09739) | QuRating是一种选择高质量数据用于训练语言模型的方法，它能够捕捉人类直观感知的文本的抽象特征。在实验中发现，平衡质量和多样性是很重要的。 |
| [^19] | [Information Complexity of Stochastic Convex Optimization: Applications to Generalization and Memorization](https://arxiv.org/abs/2402.09327) | 本论文研究了随机凸优化中记忆和学习之间的相互作用。通过量化学习算法对训练数据点揭示的信息来定义记忆，并准确定义了学习算法准确性与条件互信息（CMI）之间的权衡关系。在特定条件下，我们证明了学习算法的准确性与CMI之间的最佳边界。通过设计对手，我们进一步展示了记忆在随机凸优化中学习问题中的重要性。 |
| [^20] | [SynthCLIP: Are We Ready for a Fully Synthetic CLIP Training?](https://arxiv.org/abs/2402.01832) | SynthCLIP是一种新的框架，用于训练完全合成的CLIP模型，通过生成大规模的合成图片和标题数据集，在性能上可以与在真实数据上训练的CLIP模型相媲美。 |
| [^21] | [Receler: Reliable Concept Erasing of Text-to-Image Diffusion Models via Lightweight Erasers](https://arxiv.org/abs/2311.17717) | Receler提出了一种可靠概念擦除方法，通过轻量级橡皮擦实现对文本到图像扩散模型的概念擦除，具备鲁棒性和局部性，实验证明其优越性。 |
| [^22] | [LADDER: Revisiting the Cosmic Distance Ladder with Deep Learning Approaches and Exploring its Applications.](http://arxiv.org/abs/2401.17029) | LADDER是一个新颖的深度学习框架，通过学习宇宙的“距离梯度”，实现了预测宇宙距离并探索了多个宇宙学应用。这项研究表明在机器学习应用中需要进行有趣但谨慎的考虑。 |
| [^23] | [Can LLMs Patch Security Issues?.](http://arxiv.org/abs/2312.00024) | 本文提出了一种新的方法, Feedback-Driven Solution Synthesis (FDSS), 旨在通过将LLMs与静态代码分析工具Bandit结合，解决代码中的安全漏洞问题。该方法在现有方法的基础上有显著改进，并引入了一个新的数据集PythonSecurityEval。 |
| [^24] | [A Survey of Heterogeneous Transfer Learning.](http://arxiv.org/abs/2310.08459) | 异构迁移学习适用于源领域和目标领域具有不同特征、数据分布和标签空间的情况，通过处理这些差异来增强模型性能。 |
| [^25] | [Improved Membership Inference Attacks Against Language Classification Models.](http://arxiv.org/abs/2310.07219) | 在这篇论文中，我们提出了一个新的框架，用于对语言分类模型进行成员推理攻击。通过利用集成方法，生成多个专门的攻击模型，我们展示了这种方法在经典和语言分类任务上比单个攻击模型或每个类别标签的攻击模型更准确。 |
| [^26] | [Effective Multi-Graph Neural Networks for Illicit Account Detection on Cryptocurrency Transaction Networks.](http://arxiv.org/abs/2309.02460) | 本文介绍了一种新颖的多图神经网络模型DIAM，用于有效地检测加密货币交易网络上的非法账户。该模型通过自动学习节点表示并保留平行边的内在交易模式，在大型交易网络中取得了良好的效果。 |
| [^27] | [Structure and Gradient Dynamics Near Global Minima of Two-layer Neural Networks.](http://arxiv.org/abs/2309.00508) | 本论文通过分析两层神经网络在全局最小值附近的结构和梯度动力学，揭示了其泛化能力较强的原因。 |
| [^28] | [Improving Robustness and Accuracy of Ponzi Scheme Detection on Ethereum Using Time-Dependent Features.](http://arxiv.org/abs/2308.16391) | 这篇论文提出了一种基于交易的方法来提高以太坊上庞氏骗局的检测鲁棒性和准确性。现有的方法主要基于智能合约源代码或操作码进行检测，但缺乏鲁棒性。通过分析交易数据，可以更有效地识别庞氏骗局，因为交易更难伪装。 |
| [^29] | [Towards Realistic Unsupervised Fine-tuning with CLIP.](http://arxiv.org/abs/2308.12919) | 本论文针对无监督微调中可能出现的未知类别和超出分布范围的问题，提出了一种称为UEO的简单、高效、有效的微调方法，该方法能够同时提高对超出分布样本的检测能力和预定义类别实例的识别能力。 |
| [^30] | [Model Provenance via Model DNA.](http://arxiv.org/abs/2308.02121) | 本文介绍了模型来源证明的新概念模型DNA，通过编码模型的训练数据和输入输出信息作为紧凑全面的表示，来确定源模型是否作为目标模型的来源证明。 |
| [^31] | [Private Aggregation in Wireless Federated Learning with Heterogeneous Clusters.](http://arxiv.org/abs/2306.14088) | 本文探讨了在一个无线系统中，考虑到信息论隐私的条件下，通过基站连接到联合器的客户端，如何解决联邦学习中的隐私数据聚合问题。 |
| [^32] | [Towards Computational Architecture of Liberty: A Comprehensive Survey on Deep Learning for Generating Virtual Architecture in the Metaverse.](http://arxiv.org/abs/2305.00510) | 本文综述了当前最新的深度学习生成模型用于建筑形式的3D对象生成方法，强调了尚未充分探讨的问题，并提出了未来研究的重点议程。 |
| [^33] | [Policy Optimization for Personalized Interventions in Behavioral Health.](http://arxiv.org/abs/2303.12206) | 研究如何通过数字平台传递的行为健康介入最大化健康结果和治疗成本，提出了一个名为DecompPI的新算法，从离线数据进行预测任务，减轻了在线实验的需要，并在理论上证明了该算法的可扩展性和渐近收敛性。 |
| [^34] | [SIFT: Sparse Iso-FLOP Transformations for Maximizing Training Efficiency.](http://arxiv.org/abs/2303.11525) | 本研究提出了一种名为SIFT的方法，用于提高深度神经网络的训练效率、准确性和表示能力，通过稀疏等FLOP转换，缩短训练时间。 |
| [^35] | [Revolutionizing Genomics with Reinforcement Learning Techniques.](http://arxiv.org/abs/2302.13268) | 强化学习是一种革新的工具，可以在基因组学领域中解决自动数据分析和处理的问题。使用强化学习算法可以降低收集标记训练数据的成本，适用于基因组数据分析和解释。本调查重点关注在基因组研究领域中使用强化学习的应用，包括基因调控网络、基因组组装和序列比对。 |
| [^36] | [Language models show human-like content effects on reasoning tasks.](http://arxiv.org/abs/2207.07051) | 本研究探讨了语言模型在逻辑推理任务中是否像人类一样通过混入内容来影响答案，结果发现大型语言模型的先验期望能够捕捉到这种特征。 |

# 详细

[^1]: NeRF-MAE: 自监督三维表示学习中的Masked AutoEncoders

    NeRF-MAE : Masked AutoEncoders for Self Supervised 3D representation Learning for Neural Radiance Fields

    [https://arxiv.org/abs/2404.01300](https://arxiv.org/abs/2404.01300)

    通过使用Masked AutoEncoders，本文提出了NeRF-MAE用于自监督三维表示学习，利用标准的三维Vision Transformers适应NeRF的独特公式，将NeRF的体积网格作为密集输入，以产生有效的三维表示。

    

    由于神经场在计算机视觉和机器人领域的卓越能力，能够理解三维视觉世界，如推断语义、几何和动态等，本文探讨了神经场在从二维图像中密集表示三维场景的自监督预训练，具体使用Masked AutoEncoders的可能性。我们借鉴了将transformers扩展到新数据模态的令人惊讶的成功，利用标准的三维Vision Transformers来适应NeRF的独特公式。我们将NeRF的体积网格作为transformer的密集输入，与其他三维表示（如点云）进行对比，其信息密度可能不均匀，而表示是不规则的。由于将masked autoencoders应用于类似NeRF这样的隐式表示的困难，我们选择提取一个显式的表示。

    arXiv:2404.01300v1 Announce Type: cross  Abstract: Neural fields excel in computer vision and robotics due to their ability to understand the 3D visual world such as inferring semantics, geometry, and dynamics. Given the capabilities of neural fields in densely representing a 3D scene from 2D images, we ask the question: Can we scale their self-supervised pretraining, specifically using masked autoencoders, to generate effective 3D representations from posed RGB images. Owing to the astounding success of extending transformers to novel data modalities, we employ standard 3D Vision Transformers to suit the unique formulation of NeRFs. We leverage NeRF's volumetric grid as a dense input to the transformer, contrasting it with other 3D representations such as pointclouds where the information density can be uneven, and the representation is irregular. Due to the difficulty of applying masked autoencoders to an implicit representation, such as NeRF, we opt for extracting an explicit repres
    
[^2]: NeuroPictor: 通过多个个体的预训练和多层调制优化fMRI到图像的重建

    NeuroPictor: Refining fMRI-to-Image Reconstruction via Multi-individual Pretraining and Multi-level Modulation

    [https://arxiv.org/abs/2403.18211](https://arxiv.org/abs/2403.18211)

    NeuroPictor通过直接调制扩散模型的生成过程，实现了fMRI到图像的重建，在多个个体预训练和多层次的引导条件下，实现了更详细的图像控制。

    

    最近的fMRI到图像方法主要集中在将fMRI信号与预先训练的扩散模型的特定条件关联起来。相比之下，本文提出直接调制扩散模型的生成过程，将fMRI到图像过程分为三个步骤：i) fMRI校准编码，用于处理共享潜在空间的多个体预训练，以最小化个体差异并实现后续的跨主体训练；ii) fMRI到图像跨个体预训练，感知地学习如何引导不同个体之间高低层次条件的扩散模型；iii) fMRI到图像单个体细化，类似于步骤ii，但侧重于适应特定个体。

    arXiv:2403.18211v1 Announce Type: cross  Abstract: Recent fMRI-to-image approaches mainly focused on associating fMRI signals with specific conditions of pre-trained diffusion models. These approaches, while producing high-quality images, capture only a limited aspect of the complex information in fMRI signals and offer little detailed control over image creation. In contrast, this paper proposes to directly modulate the generation process of diffusion models using fMRI signals. Our approach, NeuroPictor, divides the fMRI-to-image process into three steps: i) fMRI calibrated-encoding, to tackle multi-individual pre-training for a shared latent space to minimize individual difference and enable the subsequent cross-subject training; ii) fMRI-to-image cross-subject pre-training, perceptually learning to guide diffusion model with high- and low-level conditions across different individuals; iii) fMRI-to-image single-subject refining, similar with step ii but focus on adapting to particula
    
[^3]: 带扩散桥的离散潜在图生成建模

    Discrete Latent Graph Generative Modeling with Diffusion Bridges

    [https://arxiv.org/abs/2403.16883](https://arxiv.org/abs/2403.16883)

    GLAD是一个在离散潜在空间上操作的图生成模型，通过适应扩散桥结构学习其离散潜在空间的先验，避免了依赖于原始数据空间的分解，在图生成任务中表现出优越性。

    

    学习潜在空间中的图生成模型相比于在原始数据空间上操作的模型受到较少关注，迄今表现出的性能乏善可陈。我们提出了GLAD，一个潜在空间图生成模型。与大多数先前的潜在空间图生成模型不同，GLAD在保留图结构的离散性质方面运行，无需进行诸如潜在空间连续性等不自然的假设。我们通过将扩散桥调整到其结构，来学习我们离散潜在空间的先验。通过在适当构建的潜在空间上操作，我们避免依赖于常用于在原始数据空间操作的模型中的分解。我们在一系列图基准数据集上进行实验，明显展示了离散潜在空间的优越性，并取得了最先进的图生成性能，使GLA

    arXiv:2403.16883v1 Announce Type: new  Abstract: Learning graph generative models over latent spaces has received less attention compared to models that operate on the original data space and has so far demonstrated lacklustre performance. We present GLAD a latent space graph generative model. Unlike most previous latent space graph generative models, GLAD operates on a discrete latent space that preserves to a significant extent the discrete nature of the graph structures making no unnatural assumptions such as latent space continuity. We learn the prior of our discrete latent space by adapting diffusion bridges to its structure. By operating over an appropriately constructed latent space we avoid relying on decompositions that are often used in models that operate in the original data space. We present experiments on a series of graph benchmark datasets which clearly show the superiority of the discrete latent space and obtain state of the art graph generative performance, making GLA
    
[^4]: 简单图压缩

    Simple Graph Condensation

    [https://arxiv.org/abs/2403.14951](https://arxiv.org/abs/2403.14951)

    提出了一种简化的图压缩方法，旨在减少图神经网络所带来的不必要复杂性。

    

    大规模图上繁重的训练成本已经引起了对图压缩的极大兴趣，涉及调整图神经网络（GNNs）在小尺度压缩图上的训练以在大规模原始图上使用。现有方法主要集中在调整压缩图和原始图之间的关键指标，如梯度、GNNs的分布和轨迹，从而在下游任务上实现了令人满意的性能。然而，这些复杂指标需要复杂的计算，可能会干扰压缩图的优化过程，使得压缩过程非常繁重和不稳定。在各个领域简化模型取得成功的背景下，我们提出了一种简化的图压缩中的指标对准方法，旨在减少从GNNs继承的不必要复杂性。在我们的方法中，我们消除外部参数，仅保留目标的压缩

    arXiv:2403.14951v1 Announce Type: cross  Abstract: The burdensome training costs on large-scale graphs have aroused significant interest in graph condensation, which involves tuning Graph Neural Networks (GNNs) on a small condensed graph for use on the large-scale original graph. Existing methods primarily focus on aligning key metrics between the condensed and original graphs, such as gradients, distribution and trajectory of GNNs, yielding satisfactory performance on downstream tasks. However, these complex metrics necessitate intricate computations and can potentially disrupt the optimization process of the condensation graph, making the condensation process highly demanding and unstable. Motivated by the recent success of simplified models in various fields, we propose a simplified approach to metric alignment in graph condensation, aiming to reduce unnecessary complexity inherited from GNNs. In our approach, we eliminate external parameters and exclusively retain the target conden
    
[^5]: 关于自监督学习的预训练数据多样性

    On Pretraining Data Diversity for Self-Supervised Learning

    [https://arxiv.org/abs/2403.13808](https://arxiv.org/abs/2403.13808)

    增加预训练数据多样性可以提高自监督学习性能，但仅在与下游数据的分布距离较小时有效。

    

    我们探讨了使用更多样化数据集对自监督学习(SSL)性能的影响，这些数据集的特征是唯一样本数量，在固定的计算预算下。我们的研究结果一致表明，增加预训练数据的多样性可以提高SSL性能，尽管只有当与下游数据的分布距离很小的时候才是如此。值得注意的是，即使通过网络爬虫或扩散生成的数据等方式实现了异常大的预训练数据多样性，分布转移仍然是一个挑战。我们的实验涵盖了七种SSL方法，使用了诸如ImageNet和YFCC100M等大规模数据集，总计超过200个GPU天。代码和训练模型将在https://github.com/hammoudhasan/DiversitySSL 上提供。

    arXiv:2403.13808v1 Announce Type: cross  Abstract: We explore the impact of training with more diverse datasets, characterized by the number of unique samples, on the performance of self-supervised learning (SSL) under a fixed computational budget. Our findings consistently demonstrate that increasing pretraining data diversity enhances SSL performance, albeit only when the distribution distance to the downstream data is minimal. Notably, even with an exceptionally large pretraining data diversity achieved through methods like web crawling or diffusion-generated data, among other ways, the distribution shift remains a challenge. Our experiments are comprehensive with seven SSL methods using large-scale datasets such as ImageNet and YFCC100M amounting to over 200 GPU days. Code and trained models will be available at https://github.com/hammoudhasan/DiversitySSL .
    
[^6]: PARMESAN: 用于密集预测任务的无参数内存搜索与转导

    PARMESAN: Parameter-Free Memory Search and Transduction for Dense Prediction Tasks

    [https://arxiv.org/abs/2403.11743](https://arxiv.org/abs/2403.11743)

    通过引入转导的概念，提出了PARMESAN，一种用于解决密集预测任务的无参数内存搜索和转导方法，实现了灵活性和无需连续训练的学习。

    

    在这项工作中，我们通过转导推理来解决深度学习中的灵活性问题。我们提出了PARMESAN（无参数内存搜索与转导），这是一种可扩展的转导方法，利用内存模块来解决密集预测任务。在推断过程中，内存中的隐藏表示被搜索以找到相应的示例。与其他方法不同，PARMESAN通过修改内存内容学习，而无需进行任何连续训练或微调可学习参数。我们的方法与常用的神经结构兼容。

    arXiv:2403.11743v1 Announce Type: new  Abstract: In this work we address flexibility in deep learning by means of transductive reasoning. For adaptation to new tasks or new data, existing methods typically involve tuning of learnable parameters or even complete re-training from scratch, rendering such approaches unflexible in practice. We argue that the notion of separating computation from memory by the means of transduction can act as a stepping stone for solving these issues. We therefore propose PARMESAN (parameter-free memory search and transduction), a scalable transduction method which leverages a memory module for solving dense prediction tasks. At inference, hidden representations in memory are being searched to find corresponding examples. In contrast to other methods, PARMESAN learns without the requirement for any continuous training or fine-tuning of learnable parameters simply by modifying the memory content. Our method is compatible with commonly used neural architecture
    
[^7]: 不仅改变标签，学习特征：使用多视角数据为深度神经网络添加水印

    Not Just Change the Labels, Learn the Features: Watermarking Deep Neural Networks with Multi-View Data

    [https://arxiv.org/abs/2403.10663](https://arxiv.org/abs/2403.10663)

    通过使用多视角数据为深度神经网络添加水印，可以有效防御对源模型功能的窃取攻击

    

    随着机器学习作为服务（MLaaS）平台的日益普及，越来越多关注深度神经网络（DNN）水印技术。这些方法用于验证目标DNN模型的所有权以保护知识产权。本文首先从特征学习的角度引入了一种新颖的基于触发集的水印方法。具体来说，我们表明通过选择展示多个特征的数据，也被称为$\textit{多视角数据}$，可以有效地防御...

    arXiv:2403.10663v1 Announce Type: cross  Abstract: With the increasing prevalence of Machine Learning as a Service (MLaaS) platforms, there is a growing focus on deep neural network (DNN) watermarking techniques. These methods are used to facilitate the verification of ownership for a target DNN model to protect intellectual property. One of the most widely employed watermarking techniques involves embedding a trigger set into the source model. Unfortunately, existing methodologies based on trigger sets are still susceptible to functionality-stealing attacks, potentially enabling adversaries to steal the functionality of the source model without a reliable means of verifying ownership. In this paper, we first introduce a novel perspective on trigger set-based watermarking methods from a feature learning perspective. Specifically, we demonstrate that by selecting data exhibiting multiple features, also referred to as $\textit{multi-view data}$, it becomes feasible to effectively defend 
    
[^8]: 基于注意力的多源领域自适应目标检测的类别条件对齐

    Attention-based Class-Conditioned Alignment for Multi-Source Domain Adaptive Object Detection

    [https://arxiv.org/abs/2403.09918](https://arxiv.org/abs/2403.09918)

    提出了一种基于注意力的类别条件对齐方案，用于多源领域自适应目标检测，在跨领域对齐每个对象类别的实例。

    

    目标检测（OD）的领域自适应方法致力于通过促进源域和目标域之间的特征对齐来缓解分布转移的影响。多源领域自适应（MSDA）允许利用多个带注释的源数据集和未标记的目标数据来提高检测模型的准确性和鲁棒性。大多数最先进的OD MSDA方法以一种与类别无关的方式执行特征对齐。最近提出的基于原型的方法提出了一种按类别对齐的方法，但由于嘈杂的伪标签而导致错误积累，这可能会对不平衡数据的自适应产生负面影响。为克服这些限制，我们提出了一种基于注意力的类别条件对齐方案，用于MSDA，该方案在跨领域对齐每个对象类别的实例。

    arXiv:2403.09918v1 Announce Type: cross  Abstract: Domain adaptation methods for object detection (OD) strive to mitigate the impact of distribution shifts by promoting feature alignment across source and target domains. Multi-source domain adaptation (MSDA) allows leveraging multiple annotated source datasets, and unlabeled target data to improve the accuracy and robustness of the detection model. Most state-of-the-art MSDA methods for OD perform feature alignment in a class-agnostic manner. This is challenging since the objects have unique modal information due to variations in object appearance across domains. A recent prototype-based approach proposed a class-wise alignment, yet it suffers from error accumulation due to noisy pseudo-labels which can negatively affect adaptation with imbalanced data. To overcome these limitations, we propose an attention-based class-conditioned alignment scheme for MSDA that aligns instances of each object category across domains. In particular, an 
    
[^9]: Transformers Get Stable: An End-to-End Signal Propagation Theory for Language Models

    Transformers Get Stable: An End-to-End Signal Propagation Theory for Language Models

    [https://arxiv.org/abs/2403.09635](https://arxiv.org/abs/2403.09635)

    提出了一个统一的信号传播理论，提供了控制transformer模型信号传播的公式，提出了DeepScaleLM初始化和缩放方案，使得可以训练非常深的模型，并发现深层模型在多个任务和数据集上胜过浅层模型。

    

    尽管transformer模型取得了巨大的成功，但在深度方面仍然很难扩展。本研究提出了一个统一的信号传播理论，并提供了控制transformer模型前向和反向信号矩的公式。我们的框架可以用于理解和缓解与高注意力分数相关的梯度消失/爆炸、秩坍缩和不稳定性。我们还提出了DeepScaleLM，一种初始化和缩放方案，通过该方案能够在模型中保持单位输出/梯度矩，从而使训练具有100多层的非常深模型成为可能。我们发现，transformer模型可以更深 - 我们的深层模型在语言建模、语音翻译和图像分类方面表现优异，包括仅编码器、仅解码器和编码器-解码器变体，适用于Pre-LN和Post-LN transformers，适用于多个数据集和模型大小。

    arXiv:2403.09635v1 Announce Type: cross  Abstract: In spite of their huge success, transformer models remain difficult to scale in depth. In this work, we develop a unified signal propagation theory and provide formulae that govern the moments of the forward and backward signal through the transformer model. Our framework can be used to understand and mitigate vanishing/exploding gradients, rank collapse, and instability associated with high attention scores. We also propose DeepScaleLM, an initialization and scaling scheme that conserves unit output/gradient moments throughout the model, enabling the training of very deep models with 100s of layers. We find that transformer models could be much deeper - our deep models with fewer parameters outperform shallow models in Language Modeling, Speech Translation, and Image Classification, across Encoder-only, Decoder-only and Encoder-Decoder variants, for both Pre-LN and Post-LN transformers, for multiple datasets and model sizes. These imp
    
[^10]: 使用贝叶斯神经场进行可扩展的时空预测

    Scalable Spatiotemporal Prediction with Bayesian Neural Fields

    [https://arxiv.org/abs/2403.07657](https://arxiv.org/abs/2403.07657)

    该论文提出了贝叶斯神经场（BayesNF），结合了深度神经网络和分层贝叶斯推断，用于处理大规模时空预测问题。

    

    时空数据集由空间参考的时间序列表示，广泛应用于许多科学和商业智能领域，例如空气污染监测，疾病跟踪和云需求预测。随着现代数据集规模和复杂性的不断增加，需要新的统计方法来捕捉复杂的时空动态并处理大规模预测问题。本研究介绍了Bayesian Neural Field (BayesNF)，这是一个用于推断时空域上丰富概率分布的通用领域统计模型，可用于包括预测、插值和变异分析在内的数据分析任务。BayesNF将用于高容量函数估计的新型深度神经网络架构与用于鲁棒不确定性量化的分层贝叶斯推断相结合。通过在定义先验分布方面进行序列化

    arXiv:2403.07657v1 Announce Type: cross  Abstract: Spatiotemporal datasets, which consist of spatially-referenced time series, are ubiquitous in many scientific and business-intelligence applications, such as air pollution monitoring, disease tracking, and cloud-demand forecasting. As modern datasets continue to increase in size and complexity, there is a growing need for new statistical methods that are flexible enough to capture complex spatiotemporal dynamics and scalable enough to handle large prediction problems. This work presents the Bayesian Neural Field (BayesNF), a domain-general statistical model for inferring rich probability distributions over a spatiotemporal domain, which can be used for data-analysis tasks including forecasting, interpolation, and variography. BayesNF integrates a novel deep neural network architecture for high-capacity function estimation with hierarchical Bayesian inference for robust uncertainty quantification. By defining the prior through a sequenc
    
[^11]: FAX: JAX中可扩展且可微分的联邦原语

    FAX: Scalable and Differentiable Federated Primitives in JAX

    [https://arxiv.org/abs/2403.07128](https://arxiv.org/abs/2403.07128)

    FAX是一个在JAX中嵌入联邦计算原语的库，支持大规模分布式计算，提供了联邦自动微分的实现，并可解释至现有的生产跨设备联邦计算系统。

    

    我们介绍了FAX，这是一个基于JAX设计的库，旨在支持数据中心和跨设备应用中的大规模分布式和联邦计算。FAX利用JAX的分片机制，实现了原生针对TPU和最先进的JAX运行时（包括Pathways）的定位。FAX将联邦计算的基本构件嵌入JAX中，带来了三个关键好处。首先，FAX的计算可以转换为XLA HLO。其次，FAX提供了联邦自动微分的完整实现，极大地简化了联邦计算的表达。最后，FAX的计算可以解释成现有的生产跨设备联邦计算系统。我们展示了FAX为数据中心中的联邦计算提供了易编程、高性能和可扩展的框架。FAX可在https://github.com/google-research/google-research/tree/master/fax 获取。

    arXiv:2403.07128v1 Announce Type: cross  Abstract: We present FAX, a JAX-based library designed to support large-scale distributed and federated computations in both data center and cross-device applications. FAX leverages JAX's sharding mechanisms to enable native targeting of TPUs and state-of-the-art JAX runtimes, including Pathways. FAX embeds building blocks for federated computations as primitives in JAX. This enables three key benefits. First, FAX computations can be translated to XLA HLO. Second, FAX provides a full implementation of federated automatic differentiation, greatly simplifying the expression of federated computations. Last, FAX computations can be interpreted out to existing production cross-device federated compute systems. We show that FAX provides an easily programmable, performant, and scalable framework for federated computations in the data center. FAX is available at https://github.com/google-research/google-research/tree/master/fax .
    
[^12]: 语义剩余提示用于持续学习

    Semantic Residual Prompts for Continual Learning

    [https://arxiv.org/abs/2403.06870](https://arxiv.org/abs/2403.06870)

    通过引入语义剩余提示，作者提出了一种稳定的选择策略，利用两级适应机制来在持续学习中解决提示冲突的问题。

    

    持续学习（CL）的提示调整方法冻结了一个大型预训练模型，并侧重于训练一些称为提示的参数向量。这些方法中的大多数将这些向量组织在一个键-值对池中，并使用输入图像作为查询来检索提示（值）。然而，随着任务的进行，由于键是学习的，提示选择策略本身也会面临灾难性遗忘，这是现有方法经常忽视的问题。为了使选择策略更加稳定，我们请求一个基础模型（CLIP）来在两级适应机制中选择我们的提示。具体而言，第一级利用标准文本提示来调整CLIP文本编码器，形成稳定的类原型。而第二级则将这些原型与查询图像一起用作键来索引一个s

    arXiv:2403.06870v1 Announce Type: new  Abstract: Prompt-tuning methods for Continual Learning (CL) freeze a large pre-trained model and focus training on a few parameter vectors termed prompts. Most of these methods organize these vectors in a pool of key-value pairs, and use the input image as query to retrieve the prompts (values). However, as keys are learned while tasks progress, the prompting selection strategy is itself subject to catastrophic forgetting, an issue often overlooked by existing approaches. For instance, prompts introduced to accommodate new tasks might end up interfering with previously learned prompts. To make the selection strategy more stable, we ask a foundational model (CLIP) to select our prompt within a two-level adaptation mechanism. Specifically, the first level leverages standard textual prompts for the CLIP textual encoder, leading to stable class prototypes. The second level, instead, uses these prototypes along with the query image as keys to index a s
    
[^13]: GraphRCG: 通过自引导表示的自条件图生成

    GraphRCG: Self-conditioned Graph Generation via Bootstrapped Representations

    [https://arxiv.org/abs/2403.01071](https://arxiv.org/abs/2403.01071)

    提出了一种自条件图生成框架，通过自引导表示指导生成过程，明确建模和利用图分布，优于传统隐式捕获分布的方法。

    

    图生成通常旨在创建与特定图分布密切对齐的新图。现有研究往往通过生成器的优化隐式捕获这种分布，可能忽视分布本身的复杂性。此外，这些方法通常忽略了学习到的分布对图生成的见解。相比之下，在这项工作中，我们提出了一种新颖的自条件图生成框架，旨在明确建模图分布并利用这些分布来指导生成过程。我们首先进行自条件建模，通过将每个图样本转换为低维表示，并优化一个表示生成器来捕获图分布并生成反映学习分布的新表示。随后，我们利用这些自引导表示作为自条件指导来...

    arXiv:2403.01071v1 Announce Type: cross  Abstract: Graph generation generally aims to create new graphs that closely align with a specific graph distribution. Existing works often implicitly capture this distribution through the optimization of generators, potentially overlooking the intricacies of the distribution itself. Furthermore, these approaches generally neglect the insights offered by the learned distribution for graph generation. In contrast, in this work, we propose a novel self-conditioned graph generation framework designed to explicitly model graph distributions and employ these distributions to guide the generation process. We first perform self-conditioned modeling to capture the graph distributions by transforming each graph sample into a low-dimensional representation and optimizing a representation generator to create new representations reflective of the learned distribution. Subsequently, we leverage these bootstrapped representations as self-conditioned guidance f
    
[^14]: 电子显微镜中的自监督学习：迈向高级图像分析基础模型

    Self-Supervised Learning in Electron Microscopy: Towards a Foundation Model for Advanced Image Analysis

    [https://arxiv.org/abs/2402.18286](https://arxiv.org/abs/2402.18286)

    本文探讨了在电子显微镜中进行自监督学习的潜力，展示自监督预训练如何促进有效的微调，同时指出较低复杂度的模型在微调过程中始终优于更复杂的随机初始化模型。

    

    在这项工作中，我们探讨了从无标签的电子显微镜数据集中进行自监督学习的潜力，迈出了构建该领域基础模型的一步。我们展示了自监督预训练如何促进有效的微调，以应用于一系列下游任务，包括语义分割、去噪、噪声与背景去除以及超分辨率。通过实验不同模型复杂度和感受野大小的变化，我们发现一个显著的现象，即微调过的较低复杂度模型始终胜过具有随机权重初始化的更复杂模型。我们展示了自监督预训练在电子显微镜背景下在各种下游任务中的多才多艺，使得快速收敛和更好的性能成为可能。我们得出结论，自监督预训练是一种强大的催化剂，特别在有限的注释数据可用时和 ef

    arXiv:2402.18286v1 Announce Type: cross  Abstract: In this work, we explore the potential of self-supervised learning from unlabeled electron microscopy datasets, taking a step toward building a foundation model in this field. We show how self-supervised pretraining facilitates efficient fine-tuning for a spectrum of downstream tasks, including semantic segmentation, denoising, noise & background removal, and super-resolution. Experimentation with varying model complexities and receptive field sizes reveals the remarkable phenomenon that fine-tuned models of lower complexity consistently outperform more complex models with random weight initialization. We demonstrate the versatility of self-supervised pretraining across various downstream tasks in the context of electron microscopy, allowing faster convergence and better performance. We conclude that self-supervised pretraining serves as a powerful catalyst, being especially advantageous when limited annotated data are available and ef
    
[^15]: 反馈高效在线微调扩散模型

    Feedback Efficient Online Fine-Tuning of Diffusion Models

    [https://arxiv.org/abs/2402.16359](https://arxiv.org/abs/2402.16359)

    提出了一种反馈高效的在线微调扩散模型的强化学习程序

    

    扩散模型在建模复杂数据分布方面表现出色，包括图像，蛋白质和小分子的分布。然而，在许多情况下，我们的目标是模拟最大化某些属性的分布的部分：例如，我们可能希望生成具有高审美质量的图像，或具有高生物活性的分子。自然地，我们可以将这视为一个强化学习（RL）问题，其目标是微调扩散模型以最大化与某些属性对应的奖励函数。即使可以访问地面真实奖励函数的在线查询，有效地发现高奖励样本也可能具有挑战性：它们在初始分布中的概率可能很低，并且可能存在许多不可行的样本，甚至没有定义良好的奖励（例如，不自然的图像或物理上不可能的分子）。在这项工作中，我们提出了一种新颖的强化学习程序，可以高效地发现高奖励样本。

    arXiv:2402.16359v1 Announce Type: cross  Abstract: Diffusion models excel at modeling complex data distributions, including those of images, proteins, and small molecules. However, in many cases, our goal is to model parts of the distribution that maximize certain properties: for example, we may want to generate images with high aesthetic quality, or molecules with high bioactivity. It is natural to frame this as a reinforcement learning (RL) problem, in which the objective is to fine-tune a diffusion model to maximize a reward function that corresponds to some property. Even with access to online queries of the ground-truth reward function, efficiently discovering high-reward samples can be challenging: they might have a low probability in the initial distribution, and there might be many infeasible samples that do not even have a well-defined reward (e.g., unnatural images or physically impossible molecules). In this work, we propose a novel reinforcement learning procedure that effi
    
[^16]: 机械信息自编码器实现自动检测和定位意外的结构损伤

    Mechanics-Informed Autoencoder Enables Automated Detection and Localization of Unforeseen Structural Damage

    [https://arxiv.org/abs/2402.15492](https://arxiv.org/abs/2402.15492)

    该研究提出了一种新颖的"部署和忘记"方法，结合廉价传感器和机械信息自编码器，实现了对结构损伤的自动检测和定位，仅需学习3小时数据即可自主识别和定位不同类型的未知损伤。

    

    结构健康监测（SHM）对于确保建筑物和桥梁等结构的安全性和长寿命至关重要。本文提出了一种基于机械信息自编码器的新颖的“部署和忘记”方法，用于自动检测和定位结构中的损伤。这种方法基于廉价传感器的全 pass 学习和机械信息自编码器的协同组合，在仅学习了 3 小时的数据后，就能自主检测和定位不同类型的意外损伤。

    arXiv:2402.15492v1 Announce Type: new  Abstract: Structural health monitoring (SHM) is vital for ensuring the safety and longevity of structures like buildings and bridges. As the volume and scale of structures and the impact of their failure continue to grow, there is a dire need for SHM techniques that are scalable, inexpensive, operate passively without human intervention, and customized for each mechanical structure without the need for complex baseline models. We present a novel "deploy-and-forget" approach for automated detection and localization of damages in structures. It is based on a synergistic combination of fully passive measurements from inexpensive sensors and a mechanics-informed autoencoder. Once deployed, our solution continuously learns and adapts a bespoke baseline model for each structure, learning from its undamaged state's response characteristics. After learning from just 3 hours of data, it can autonomously detect and localize different types of unforeseen dam
    
[^17]: MSPipe: 通过意识到陈旧性的管道实现高效的时间性GNN训练

    MSPipe: Efficient Temporal GNN Training via Staleness-aware Pipeline

    [https://arxiv.org/abs/2402.15113](https://arxiv.org/abs/2402.15113)

    提出了MSPipe，一个通用而高效的MTGNNs框架，实现了最大化训练吞吐量同时保持模型准确性

    

    记忆型时间性图神经网络（MTGNNs）是一类利用节点记忆模块捕获和保留长期时间依赖关系的时间性图神经网络，相对于无记忆的对应网络具有卓越的性能。然而，在MTGNNs中，为了获取最新的信息，记忆模块的迭代读取和更新过程需要遵循时间依赖关系，这引入了显著的开销并限制了训练吞吐量。现有静态GNNs的优化不适用于MTGNNs，因为两者在训练范式、模型架构和缺乏记忆模块上存在差异。此外，它们并未有效地解决时间依赖带来的挑战，使其对MTGNN训练无效。在本文中，我们提出了MSPipe，这是一个通用而高效的MTGNNs框架，可以最大化训练吞吐量同时保持模型准确性。

    arXiv:2402.15113v1 Announce Type: new  Abstract: Memory-based Temporal Graph Neural Networks (MTGNNs) are a class of temporal graph neural networks that utilize a node memory module to capture and retain long-term temporal dependencies, leading to superior performance compared to memory-less counterparts. However, the iterative reading and updating process of the memory module in MTGNNs to obtain up-to-date information needs to follow the temporal dependencies. This introduces significant overhead and limits training throughput. Existing optimizations for static GNNs are not directly applicable to MTGNNs due to differences in training paradigm, model architecture, and the absence of a memory module. Moreover, they do not effectively address the challenges posed by temporal dependencies, making them ineffective for MTGNN training. In this paper, we propose MSPipe, a general and efficient framework for MTGNNs that maximizes training throughput while maintaining model accuracy. Our design
    
[^18]: 选择高质量数据用于训练语言模型的QuRating方法

    QuRating: Selecting High-Quality Data for Training Language Models

    [https://arxiv.org/abs/2402.09739](https://arxiv.org/abs/2402.09739)

    QuRating是一种选择高质量数据用于训练语言模型的方法，它能够捕捉人类直观感知的文本的抽象特征。在实验中发现，平衡质量和多样性是很重要的。

    

    选择高质量的预训练数据对于创建能力强的语言模型很重要，但现有方法依赖简单的启发式方法。我们介绍了一种名为QuRating的方法，用于选择能够捕捉人类直观感知的文本的抽象特征的预训练文本数据。在本文中，我们研究了四个特征 - 写作风格、所需专业知识、事实和琐事以及教育价值。我们发现，语言模型能够辨别这些特征，并观察到它们在进行文本的配对判断方面比直接评估文本质量更好。我们训练了一个QuRater模型，从配对判断中学习标量评分，并使用它为260B的训练语料库中的每个标准进行质量评级注释。在实验中，我们根据不同的质量评级选择了30B个令牌，并在所选数据上训练了13亿参数的语言模型。我们发现在质量和多样性之间保持平衡是很重要的。

    arXiv:2402.09739v1 Announce Type: new  Abstract: Selecting high-quality pre-training data is important for creating capable language models, but existing methods rely on simple heuristics. We introduce QuRating, a method for selecting pre-training data that captures the abstract qualities of texts which humans intuitively perceive. In this paper, we investigate four qualities - writing style, required expertise, facts & trivia, and educational value. We find that LLMs are able to discern these qualities and observe that they are better at making pairwise judgments of texts than at rating the quality of a text directly. We train a QuRater model to learn scalar ratings from pairwise judgments, and use it to annotate a 260B training corpus with quality ratings for each of the four criteria. In our experiments, we select 30B tokens according to the different quality ratings and train 1.3B-parameter language models on the selected data. We find that it is important to balance quality and di
    
[^19]: 随机凸优化的信息复杂度：泛化和记忆的应用

    Information Complexity of Stochastic Convex Optimization: Applications to Generalization and Memorization

    [https://arxiv.org/abs/2402.09327](https://arxiv.org/abs/2402.09327)

    本论文研究了随机凸优化中记忆和学习之间的相互作用。通过量化学习算法对训练数据点揭示的信息来定义记忆，并准确定义了学习算法准确性与条件互信息（CMI）之间的权衡关系。在特定条件下，我们证明了学习算法的准确性与CMI之间的最佳边界。通过设计对手，我们进一步展示了记忆在随机凸优化中学习问题中的重要性。

    

    在这项工作中，我们研究了在随机凸优化（SCO）的背景下记忆和学习的相互作用。我们通过学习算法对其训练数据点揭示的信息来定义记忆。然后，我们使用Steinke和Zakynthinou（2020）提出的条件互信息（CMI）框架来量化这些信息。我们的主要结果是对学习算法的准确性和它的CMI之间的权衡的精确描述，回答了Livni（2023）提出的一个未解之问。我们证明，在$L^2$ Lipschitz-有界的设置和强凸性下，每个具有超额错误$\varepsilon$的学习算法的CMI下界分别被$\Omega(1/\varepsilon^2)$和$\Omega(1/\varepsilon)$所限制。我们进一步通过设计一个能够准确识别出大部分训练数据的对手来展示记忆在SCO中学习问题中的重要作用。

    arXiv:2402.09327v1 Announce Type: new Abstract: In this work, we investigate the interplay between memorization and learning in the context of \emph{stochastic convex optimization} (SCO). We define memorization via the information a learning algorithm reveals about its training data points. We then quantify this information using the framework of conditional mutual information (CMI) proposed by Steinke and Zakynthinou (2020). Our main result is a precise characterization of the tradeoff between the accuracy of a learning algorithm and its CMI, answering an open question posed by Livni (2023). We show that, in the $L^2$ Lipschitz--bounded setting and under strong convexity, every learner with an excess error $\varepsilon$ has CMI bounded below by $\Omega(1/\varepsilon^2)$ and $\Omega(1/\varepsilon)$, respectively. We further demonstrate the essential role of memorization in learning problems in SCO by designing an adversary capable of accurately identifying a significant fraction of the
    
[^20]: SynthCLIP: 我们准备好开始完全合成的CLIP训练了吗？

    SynthCLIP: Are We Ready for a Fully Synthetic CLIP Training?

    [https://arxiv.org/abs/2402.01832](https://arxiv.org/abs/2402.01832)

    SynthCLIP是一种新的框架，用于训练完全合成的CLIP模型，通过生成大规模的合成图片和标题数据集，在性能上可以与在真实数据上训练的CLIP模型相媲美。

    

    我们提出了SynthCLIP，一种新颖的用于训练完全合成的CLIP模型的框架，与之前依赖真实数据的方法有着显著区别。借助最近的文本到图像生成网络和大型语言模型，我们能够生成任意规模的图像和相应的标题的合成数据集，无需人为干预。通过大规模的训练，SynthCLIP实现了与在真实数据集上训练的CLIP模型相当的性能。我们还介绍了SynthCI-30M，一个纯粹合成的数据集，包含3000万张带标题的图片。我们的代码、训练模型和生成的数据已经在https://github.com/hammoudhasan/SynthCLIP发布。

    We present SynthCLIP, a novel framework for training CLIP models with entirely synthetic text-image pairs, significantly departing from previous methods relying on real data. Leveraging recent text-to-image (TTI) generative networks and large language models (LLM), we are able to generate synthetic datasets of images and corresponding captions at any scale, with no human intervention. With training at scale, SynthCLIP achieves performance comparable to CLIP models trained on real datasets. We also introduce SynthCI-30M, a purely synthetic dataset comprising 30 million captioned images. Our code, trained models, and generated data are released at https://github.com/hammoudhasan/SynthCLIP
    
[^21]: Receler: 通过轻量级橡皮擦可靠地擦除文本到图像扩散模型中的概念

    Receler: Reliable Concept Erasing of Text-to-Image Diffusion Models via Lightweight Erasers

    [https://arxiv.org/abs/2311.17717](https://arxiv.org/abs/2311.17717)

    Receler提出了一种可靠概念擦除方法，通过轻量级橡皮擦实现对文本到图像扩散模型的概念擦除，具备鲁棒性和局部性，实验证明其优越性。

    

    在文本到图像扩散模型中，概念擦除旨在禁用预训练的扩散模型生成与目标概念相关的图像。为了实现可靠的概念擦除，希望具备鲁棒性和局部性的属性。前者阻止模型为任何释义或学习提示生成与目标概念相关的图像，而后者保持其生成具有非目标概念的图像的能力。在本文中，我们提出了通过轻量级橡皮擦（Receler）来实现可靠的概念擦除。它学习了一个轻量级的橡皮擦来进行概念擦除，同时通过提出的概念定位正则化和对抗提示学习方案满足上述理想特性。通过对各种概念的全面实验验证了Receler相对于先前方法的优越性。我们的代码将在接受后提供。

    arXiv:2311.17717v2 Announce Type: replace-cross  Abstract: Concept erasure in text-to-image diffusion models aims to disable pre-trained diffusion models from generating images related to a target concept. To perform reliable concept erasure, the properties of robustness and locality are desirable. The former refrains the model from producing images associated with the target concept for any paraphrased or learned prompts, while the latter preserves its ability in generating images with non-target concepts. In this paper, we propose Reliable Concept Erasing via Lightweight Erasers (Receler). It learns a lightweight Eraser to perform concept erasing while satisfying the above desirable properties by proposed concept-localized regularization and adversarial prompt learning schemes. Comprehensive experiments with various concepts verify the superiority of Receler over previous methods. Our code will be available upon acceptance.
    
[^22]: LADDER: 深度学习方法重新探索宇宙距离梯度并探索其应用

    LADDER: Revisiting the Cosmic Distance Ladder with Deep Learning Approaches and Exploring its Applications. (arXiv:2401.17029v1 [astro-ph.CO])

    [http://arxiv.org/abs/2401.17029](http://arxiv.org/abs/2401.17029)

    LADDER是一个新颖的深度学习框架，通过学习宇宙的“距离梯度”，实现了预测宇宙距离并探索了多个宇宙学应用。这项研究表明在机器学习应用中需要进行有趣但谨慎的考虑。

    

    我们通过一种名为LADDER（深度学习算法用于距离估计和重建）的新颖深度学习框架，研究了使用“宇宙距离梯度”重建宇宙的前景。LADDER使用了来自Pantheon Type Ia超新星编译的视星等数据，并将数据点之间的全部协方差信息进行了融合，以生成具有相应误差的预测结果。通过对多个深度学习模型进行了多个验证实验后，我们选择了表现最佳的LADDER模型。然后，我们演示了我们的方法在宇宙学上的应用，包括作为独立于模型的一致性检查工具，用于其他数据集（如重子声学振荡）的校准，用于高红移数据集（如伽玛射线暴）的校准，以及用作未来探测的独立于模型的模拟目录生成器等等。我们的分析给出了关于在机器学习应用中要进行有趣而谨慎的考虑的支持。

    We investigate the prospect of reconstructing the ``cosmic distance ladder'' of the Universe using a novel deep learning framework called LADDER - Learning Algorithm for Deep Distance Estimation and Reconstruction. LADDER is trained on the apparent magnitude data from the Pantheon Type Ia supernovae compilation, incorporating the full covariance information among data points, to produce predictions along with corresponding errors. After employing several validation tests with a number of deep learning models, we pick LADDER as the best performing one. We then demonstrate applications of our method in the cosmological context, that include serving as a model-independent tool for consistency checks for other datasets like baryon acoustic oscillations, calibration of high-redshift datasets such as gamma ray bursts, use as a model-independent mock catalog generator for future probes, etc. Our analysis advocates for interesting yet cautious consideration of machine learning applications in 
    
[^23]: LLMs能够修复安全问题吗？

    Can LLMs Patch Security Issues?. (arXiv:2312.00024v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2312.00024](http://arxiv.org/abs/2312.00024)

    本文提出了一种新的方法, Feedback-Driven Solution Synthesis (FDSS), 旨在通过将LLMs与静态代码分析工具Bandit结合，解决代码中的安全漏洞问题。该方法在现有方法的基础上有显著改进，并引入了一个新的数据集PythonSecurityEval。

    

    大型语言模型(LLMs)在代码生成方面显示出了令人印象深刻的能力。然而，类似于人类开发者，这些模型可能会生成包含安全漏洞和缺陷的代码。编写安全代码仍然是一个重大挑战，因为漏洞通常在程序与外部系统或服务（如数据库和操作系统）之间的交互过程中出现。在本文中，我们提出了一种新颖的方法，即基于反馈的解决方案合成（FDSS），旨在探索使用LLMs接收来自静态代码分析工具Bandit的反馈，然后LLMs生成潜在解决方案来解决安全漏洞。每个解决方案以及易受攻击的代码随后被送回LLMs进行代码完善。我们的方法在基线上表现出显著改进，并优于现有方法。此外，我们引入了一个新的数据集PythonSecurityEval，该数据集收集了来自Stack Overflow的真实场景数据。

    Large Language Models (LLMs) have shown impressive proficiency in code generation. Nonetheless, similar to human developers, these models might generate code that contains security vulnerabilities and flaws. Writing secure code remains a substantial challenge, as vulnerabilities often arise during interactions between programs and external systems or services, such as databases and operating systems. In this paper, we propose a novel approach, Feedback-Driven Solution Synthesis (FDSS), designed to explore the use of LLMs in receiving feedback from Bandit, which is a static code analysis tool, and then the LLMs generate potential solutions to resolve security vulnerabilities. Each solution, along with the vulnerable code, is then sent back to the LLM for code refinement. Our approach shows a significant improvement over the baseline and outperforms existing approaches. Furthermore, we introduce a new dataset, PythonSecurityEval, collected from real-world scenarios on Stack Overflow to e
    
[^24]: 异构迁移学习综述

    A Survey of Heterogeneous Transfer Learning. (arXiv:2310.08459v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.08459](http://arxiv.org/abs/2310.08459)

    异构迁移学习适用于源领域和目标领域具有不同特征、数据分布和标签空间的情况，通过处理这些差异来增强模型性能。

    

    近年来，迁移学习的应用在很多实际场景中得到了广泛的应用，它利用源领域的知识来增强目标领域模型的性能。其成功的关键在于源领域和目标领域之间的共享知识，这是大多数迁移学习方法的前提条件。然而，这些方法通常假设两个领域具有相同的特征空间和标签空间，即同质迁移学习，但这并不总是现实合理的假设。通常，源领域和目标领域在特征空间、数据分布和标签空间上存在差异，这使得获取具有与目标领域相同特征和标签空间的源领域数据变得具有挑战性或昂贵。对这些差异进行随意的消除并不总是可行或最优的。因此，异构迁移学习作为一种应对这种差异的方法已经崭露头角，并在各种任务中显示出了巨大的潜力。

    The application of transfer learning, an approach utilizing knowledge from a source domain to enhance model performance in a target domain, has seen a tremendous rise in recent years, underpinning many real-world scenarios. The key to its success lies in the shared common knowledge between the domains, a prerequisite in most transfer learning methodologies. These methods typically presuppose identical feature spaces and label spaces in both domains, known as homogeneous transfer learning, which, however, is not always a practical assumption. Oftentimes, the source and target domains vary in feature spaces, data distributions, and label spaces, making it challenging or costly to secure source domain data with identical feature and label spaces as the target domain. Arbitrary elimination of these differences is not always feasible or optimal. Thus, heterogeneous transfer learning, acknowledging and dealing with such disparities, has emerged as a promising approach for a variety of tasks.
    
[^25]: 改进的对语言分类模型的成员推理攻击

    Improved Membership Inference Attacks Against Language Classification Models. (arXiv:2310.07219v1 [cs.LG])

    [http://arxiv.org/abs/2310.07219](http://arxiv.org/abs/2310.07219)

    在这篇论文中，我们提出了一个新的框架，用于对语言分类模型进行成员推理攻击。通过利用集成方法，生成多个专门的攻击模型，我们展示了这种方法在经典和语言分类任务上比单个攻击模型或每个类别标签的攻击模型更准确。

    

    人工智能系统在日常生活中普遍存在，具有零售、制造、健康等许多领域的用例。随着人工智能采用的增加，已经发现了相关的风险，包括对使用其数据训练模型的人的隐私风险。评估机器学习模型的隐私风险对于是否使用、部署或共享模型做出知情决策至关重要。隐私风险评估的一种常见方法是对模型进行一个或多个已知攻击，并测量它们的成功率。我们提出了一个新颖的框架，用于对分类模型进行成员推理攻击。我们的框架利用集成方法，为不同数据子集生成许多专门的攻击模型。我们展示了这种方法在经典和语言分类任务上比单个攻击模型或每个类别标签的攻击模型都实现了更高的准确性。

    Artificial intelligence systems are prevalent in everyday life, with use cases in retail, manufacturing, health, and many other fields. With the rise in AI adoption, associated risks have been identified, including privacy risks to the people whose data was used to train models. Assessing the privacy risks of machine learning models is crucial to enabling knowledgeable decisions on whether to use, deploy, or share a model. A common approach to privacy risk assessment is to run one or more known attacks against the model and measure their success rate. We present a novel framework for running membership inference attacks against classification models. Our framework takes advantage of the ensemble method, generating many specialized attack models for different subsets of the data. We show that this approach achieves higher accuracy than either a single attack model or an attack model per class label, both on classical and language classification tasks.
    
[^26]: 有效的多图神经网络用于加密货币交易网络上的非法账户检测

    Effective Multi-Graph Neural Networks for Illicit Account Detection on Cryptocurrency Transaction Networks. (arXiv:2309.02460v1 [cs.LG])

    [http://arxiv.org/abs/2309.02460](http://arxiv.org/abs/2309.02460)

    本文介绍了一种新颖的多图神经网络模型DIAM，用于有效地检测加密货币交易网络上的非法账户。该模型通过自动学习节点表示并保留平行边的内在交易模式，在大型交易网络中取得了良好的效果。

    

    我们研究了在线金融市场中日益重要的加密货币交易网络上的非法账户检测。在加密货币上的非法活动激增导致了普通用户数十亿的损失。现有的解决方案要么依赖于繁琐的特征工程来获得手工特征，要么不能充分利用加密货币交易数据中丰富的语义信息，从而导致亚优化的性能。在本文中，我们将非法账户检测问题定义为带有边属性的有向多图上的分类任务，并提出了DIAM，一种新颖的多图神经网络模型，用于在大型交易网络上有效地检测非法账户。首先，DIAM包含一个Edge2Seq模块，通过同时考虑边属性和有向边序列依赖关系，自动学习有效的节点表示，保留平行边的内在交易模式。然后利用t

    We study illicit account detection on transaction networks of cryptocurrencies that are increasi_testngly important in online financial markets. The surge of illicit activities on cryptocurrencies has resulted in billions of losses from normal users. Existing solutions either rely on tedious feature engineering to get handcrafted features, or are inadequate to fully utilize the rich semantics of cryptocurrency transaction data, and consequently, yield sub-optimal performance. In this paper, we formulate the illicit account detection problem as a classification task over directed multigraphs with edge attributes, and present DIAM, a novel multi-graph neural network model to effectively detect illicit accounts on large transaction networks. First, DIAM includes an Edge2Seq module that automatically learns effective node representations preserving intrinsic transaction patterns of parallel edges, by considering both edge attributes and directed edge sequence dependencies. Then utilizing t
    
[^27]: 两层神经网络全局最小值附近的结构和梯度动力学

    Structure and Gradient Dynamics Near Global Minima of Two-layer Neural Networks. (arXiv:2309.00508v1 [cs.LG])

    [http://arxiv.org/abs/2309.00508](http://arxiv.org/abs/2309.00508)

    本论文通过分析两层神经网络在全局最小值附近的结构和梯度动力学，揭示了其泛化能力较强的原因。

    

    在温和的假设下，我们研究了两层神经网络在全局最小值附近的损失函数表面的结构，确定了能够实现完美泛化的参数集，并完整描述了其周围的梯度流动态。通过新颖的技术，我们揭示了复杂的损失函数表面的一些简单方面，并揭示了模型、目标函数、样本和初始化对训练动力学的不同影响。基于这些结果，我们还解释了为什么（过度参数化的）神经网络可以很好地泛化。

    Under mild assumptions, we investigate the structure of loss landscape of two-layer neural networks near global minima, determine the set of parameters which give perfect generalization, and fully characterize the gradient flows around it. With novel techniques, our work uncovers some simple aspects of the complicated loss landscape and reveals how model, target function, samples and initialization affect the training dynamics differently. Based on these results, we also explain why (overparametrized) neural networks could generalize well.
    
[^28]: 提高以太坊上庞氏骗局检测的鲁棒性和准确性的方法

    Improving Robustness and Accuracy of Ponzi Scheme Detection on Ethereum Using Time-Dependent Features. (arXiv:2308.16391v1 [cs.CR])

    [http://arxiv.org/abs/2308.16391](http://arxiv.org/abs/2308.16391)

    这篇论文提出了一种基于交易的方法来提高以太坊上庞氏骗局的检测鲁棒性和准确性。现有的方法主要基于智能合约源代码或操作码进行检测，但缺乏鲁棒性。通过分析交易数据，可以更有效地识别庞氏骗局，因为交易更难伪装。

    

    区块链的快速发展导致越来越多的资金涌入加密货币市场，也吸引了近年来网络犯罪分子的兴趣。庞氏骗局作为一种老式的欺诈行为，现在也流行于区块链上，给许多加密货币投资者造成了巨大的财务损失。现有文献中已经提出了一些庞氏骗局检测方法，其中大多数是基于智能合约的源代码或操作码进行检测的。虽然基于合约代码的方法在准确性方面表现出色，但它缺乏鲁棒性：首先，大部分以太坊上的合约源代码并不公开可用；其次，庞氏骗局开发者可以通过混淆操作码或者创造新的分配逻辑来欺骗基于合约代码的检测模型（因为这些模型仅在现有的庞氏逻辑上进行训练）。基于交易的方法可以提高检测的鲁棒性，因为与智能合约不同，交易更加难以伪装。

    The rapid development of blockchain has led to more and more funding pouring into the cryptocurrency market, which also attracted cybercriminals' interest in recent years. The Ponzi scheme, an old-fashioned fraud, is now popular on the blockchain, causing considerable financial losses to many crypto-investors. A few Ponzi detection methods have been proposed in the literature, most of which detect a Ponzi scheme based on its smart contract source code or opcode. The contract-code-based approach, while achieving very high accuracy, is not robust: first, the source codes of a majority of contracts on Ethereum are not available, and second, a Ponzi developer can fool a contract-code-based detection model by obfuscating the opcode or inventing a new profit distribution logic that cannot be detected (since these models were trained on existing Ponzi logics only). A transaction-based approach could improve the robustness of detection because transactions, unlike smart contracts, are harder t
    
[^29]: 用CLIP实现真实的无监督微调

    Towards Realistic Unsupervised Fine-tuning with CLIP. (arXiv:2308.12919v1 [cs.CV])

    [http://arxiv.org/abs/2308.12919](http://arxiv.org/abs/2308.12919)

    本论文针对无监督微调中可能出现的未知类别和超出分布范围的问题，提出了一种称为UEO的简单、高效、有效的微调方法，该方法能够同时提高对超出分布样本的检测能力和预定义类别实例的识别能力。

    

    视觉-语言模型（VLM）如CLIP的出现推动了人们在下游监督学习任务中的应用研究。尽管一些之前的研究探索了CLIP的无监督微调，但它们常常依赖于与真实标签相关的类名等先验知识。本文中，我们探讨了一种真实的无监督微调情景，假设未标记的数据可能包含来自未知类别的超出分布范围的样本。此外，我们强调了在预定义类标签的识别之外，同时提高对超出分布检测能力的重要性。为了解决这个问题，我们提出了一种简单、高效、有效的微调方法，称为Universal Entropy Optimization (UEO)。UEO利用样本级置信度，以近似方式最小化置信实例的条件熵并最大化边缘熵。

    The emergence of vision-language models (VLMs), such as CLIP, has spurred a significant research effort towards their application for downstream supervised learning tasks. Although some previous studies have explored the unsupervised fine-tuning of CLIP, they often rely on prior knowledge in the form of class names associated with ground truth labels. In this paper, we delve into a realistic unsupervised fine-tuning scenario by assuming that the unlabeled data might contain out-of-distribution samples from unknown classes. Furthermore, we emphasize the importance of simultaneously enhancing out-of-distribution detection capabilities alongside the recognition of instances associated with predefined class labels.  To tackle this problem, we present a simple, efficient, and effective fine-tuning approach called Universal Entropy Optimization (UEO). UEO leverages sample-level confidence to approximately minimize the conditional entropy of confident instances and maximize the marginal entro
    
[^30]: 通过模型DNA的模型来源证明

    Model Provenance via Model DNA. (arXiv:2308.02121v1 [cs.LG])

    [http://arxiv.org/abs/2308.02121](http://arxiv.org/abs/2308.02121)

    本文介绍了模型来源证明的新概念模型DNA，通过编码模型的训练数据和输入输出信息作为紧凑全面的表示，来确定源模型是否作为目标模型的来源证明。

    

    了解机器学习（ML）模型的生命周期是一个有趣的研究领域（例如，了解模型的来源，训练方式以及使用方式）。本文聚焦于这一领域内的一个新问题，即模型来源证明（MP），该问题涉及目标模型与其预训练模型之间的关系，并旨在确定一个源模型是否作为目标模型的来源证明。这是一个重要的问题，对于确保机器学习模型的安全性和知识产权具有重要意义，但在文献中并没有得到很多关注。为了填补这一空白，我们引入了一个新概念，即模型DNA，它代表了机器学习模型的独特特征。我们利用数据驱动和模型驱动的表示学习方法，将模型的训练数据和输入输出信息编码为模型的紧凑且全面的表示（即DNA）。

    Understanding the life cycle of the machine learning (ML) model is an intriguing area of research (e.g., understanding where the model comes from, how it is trained, and how it is used). This paper focuses on a novel problem within this field, namely Model Provenance (MP), which concerns the relationship between a target model and its pre-training model and aims to determine whether a source model serves as the provenance for a target model. This is an important problem that has significant implications for ensuring the security and intellectual property of machine learning models but has not received much attention in the literature. To fill in this gap, we introduce a novel concept of Model DNA which represents the unique characteristics of a machine learning model. We utilize a data-driven and model-driven representation learning method to encode the model's training data and input-output information as a compact and comprehensive representation (i.e., DNA) of the model. Using this 
    
[^31]: 非同质化集群下的无线联邦学习中的私有数据聚合

    Private Aggregation in Wireless Federated Learning with Heterogeneous Clusters. (arXiv:2306.14088v1 [cs.LG])

    [http://arxiv.org/abs/2306.14088](http://arxiv.org/abs/2306.14088)

    本文探讨了在一个无线系统中，考虑到信息论隐私的条件下，通过基站连接到联合器的客户端，如何解决联邦学习中的隐私数据聚合问题。

    

    联邦学习是通过多个参与客户端私有数据的协同训练神经网络的方法。在训练神经网络的过程中，使用一种著名并广泛使用的迭代优化算法——梯度下降算法。每个客户端使用本地数据计算局部梯度并将其发送给联合器以进行聚合。客户端数据的隐私是一个主要问题。实际上，观察到局部梯度就足以泄露客户端的数据。已研究了用于应对联邦学习中隐私问题的私有聚合方案，其中所有用户都彼此连接并与联合器连接。本文考虑了一个无线系统架构，其中客户端仅通过基站连接到联合器。当需要信息论隐私时，我们推导出通信成本的基本极限，并引入和分析了一种针对这种情况量身定制的私有聚合方案。

    Federated learning collaboratively trains a neural network on privately owned data held by several participating clients. The gradient descent algorithm, a well-known and popular iterative optimization procedure, is run to train the neural network. Every client uses its local data to compute partial gradients and sends it to the federator which aggregates the results. Privacy of the clients' data is a major concern. In fact, observing the partial gradients can be enough to reveal the clients' data. Private aggregation schemes have been investigated to tackle the privacy problem in federated learning where all the users are connected to each other and to the federator. In this paper, we consider a wireless system architecture where clients are only connected to the federator via base stations. We derive fundamental limits on the communication cost when information-theoretic privacy is required, and introduce and analyze a private aggregation scheme tailored for this setting.
    
[^32]: 通向自由计算架构: 关于深度学习生成元宇宙虚拟建筑的综合调研

    Towards Computational Architecture of Liberty: A Comprehensive Survey on Deep Learning for Generating Virtual Architecture in the Metaverse. (arXiv:2305.00510v1 [cs.HC])

    [http://arxiv.org/abs/2305.00510](http://arxiv.org/abs/2305.00510)

    本文综述了当前最新的深度学习生成模型用于建筑形式的3D对象生成方法，强调了尚未充分探讨的问题，并提出了未来研究的重点议程。

    

    利用深度学习的3D形状生成技术正在受到计算机视觉和建筑设计两方的越来越多的关注。本综合调查旨在调查和比较当前最新的基于深度生成模型（DGMs）的3D对象生成方法，包括生成对抗网络（GANs）、变分自动编码器（VAEs）、3D感知图像和扩散模型。我们调查了187篇文章(占2018-2022年间发表文章的80.7%)，以回顾在虚拟环境下建筑生成可能性的领域，限于建筑形式。我们提供了建筑研究、虚拟环境和相关技术方法的概述，接着回顾了离散体素生成、由2D图像生成的3D模型以及条件参数的最近趋势。我们强调了3D生成和参数化控制中尚未充分探讨的问题值得进一步研究。此外，我们推测包括生成多样性、新型输出和嵌入式构建等四个研究议程可能会成为未来研究的重点。

    3D shape generation techniques utilizing deep learning are increasing attention from both computer vision and architectural design. This survey focuses on investigating and comparing the current latest approaches to 3D object generation with deep generative models (DGMs), including Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), 3D-aware images, and diffusion models. We discuss 187 articles (80.7% of articles published between 2018-2022) to review the field of generated possibilities of architecture in virtual environments, limited to the architecture form. We provide an overview of architectural research, virtual environment, and related technical approaches, followed by a review of recent trends in discrete voxel generation, 3D models generated from 2D images, and conditional parameters. We highlight under-explored issues in 3D generation and parameterized control that is worth further investigation. Moreover, we speculate that four research agendas including
    
[^33]: 行为健康个性化介入的政策优化

    Policy Optimization for Personalized Interventions in Behavioral Health. (arXiv:2303.12206v1 [cs.LG])

    [http://arxiv.org/abs/2303.12206](http://arxiv.org/abs/2303.12206)

    研究如何通过数字平台传递的行为健康介入最大化健康结果和治疗成本，提出了一个名为DecompPI的新算法，从离线数据进行预测任务，减轻了在线实验的需要，并在理论上证明了该算法的可扩展性和渐近收敛性。

    

    问题定义：通过数字平台传递的行为健康介入，通过教育，激励，提醒和外展，有望显着改善健康结果。我们研究了在介入具有成本和能力限制的情况下，优化患者个性化介入以最大化某种长期结果的问题。方法/结果：本文提供了一种无模型方法来解决这个问题。我们发现，来自增强学习文献的通用无模型方法对于医疗应用来说过于数据密集，而更简单的赌臂问题方法取得了进展，但忽略了长期患者动态。我们提出了一种新算法，称为DecompPI，它近似于一步政策迭代。实现DecompPI只需从离线数据进行预测任务，减轻了在线实验的需要。在理论上，我们展示了在一种自然的结构假设下，DecompPI可以获得算法复杂度的渐近收敛性，同时保持一个可扩展的模型.

    Problem definition: Behavioral health interventions, delivered through digital platforms, have the potential to significantly improve health outcomes, through education, motivation, reminders, and outreach. We study the problem of optimizing personalized interventions for patients to maximize some long-term outcome, in a setting where interventions are costly and capacity-constrained.  Methodology/results: This paper provides a model-free approach to solving this problem. We find that generic model-free approaches from the reinforcement learning literature are too data intensive for healthcare applications, while simpler bandit approaches make progress at the expense of ignoring long-term patient dynamics. We present a new algorithm we dub DecompPI that approximates one step of policy iteration. Implementing DecompPI simply consists of a prediction task from offline data, alleviating the need for online experimentation. Theoretically, we show that under a natural set of structural assu
    
[^34]: SIFT: 稀疏等FLOP转换以最大限度提高训练效率

    SIFT: Sparse Iso-FLOP Transformations for Maximizing Training Efficiency. (arXiv:2303.11525v1 [cs.LG])

    [http://arxiv.org/abs/2303.11525](http://arxiv.org/abs/2303.11525)

    本研究提出了一种名为SIFT的方法，用于提高深度神经网络的训练效率、准确性和表示能力，通过稀疏等FLOP转换，缩短训练时间。

    

    最近的研究探索了使用权重稀疏性来改善深度神经网络（DNN）的训练效率（与训练FLOPS相关的测试准确性）。 这些工作旨在减少训练FLOP，但使用稀疏权重进行训练通常会导致准确性损失或需要更长的训练周期，使得结果的训练效率不够清晰。 相比之下，我们专注于使用稀疏性提高准确性，同时使用与密集模型相同的FLOPS，并通过更高的准确性展示训练效率提高。 在本文中，我们介绍了SIFT，一组用作密集层的即插即用替代品来提高其表示能力和FLOP效率的稀疏等FLOP转换。 每个转换都由一个单一参数（稀疏级别）参数化，并提供更大的搜索空间以找到最佳的稀疏掩膜。

    Recent works have explored the use of weight sparsity to improve the training efficiency (test accuracy w.r.t training FLOPs) of deep neural networks (DNNs). These works aim to reduce training FLOPs but training with sparse weights often leads to accuracy loss or requires longer train schedules, making the resulting training efficiency less clear. In contrast, we focus on using sparsity to increase accuracy while using the same FLOPS as the dense model and show training efficiency gains through higher accuracy. In this work, we introduce SIFT, a family of Sparse Iso-FLOP Transformations which are used as drop-in replacements for dense layers to improve their representational capacity and FLOP efficiency. Each transformation is parameterized by a single parameter (sparsity level) and provides a larger search space to find optimal sparse masks. Without changing any training hyperparameters, replacing dense layers with SIFT leads to significant improvements across computer vision (CV) and
    
[^35]: 使用强化学习技术革新基因组学

    Revolutionizing Genomics with Reinforcement Learning Techniques. (arXiv:2302.13268v2 [q-bio.GN] UPDATED)

    [http://arxiv.org/abs/2302.13268](http://arxiv.org/abs/2302.13268)

    强化学习是一种革新的工具，可以在基因组学领域中解决自动数据分析和处理的问题。使用强化学习算法可以降低收集标记训练数据的成本，适用于基因组数据分析和解释。本调查重点关注在基因组研究领域中使用强化学习的应用，包括基因调控网络、基因组组装和序列比对。

    

    近年来，强化学习（RL）作为一种强大的工具出现在解决各种问题中，包括决策和基因组学。过去二十年的原始基因组数据指数增长已经超出了手动分析的能力，这导致对自动数据分析和处理的兴趣越来越大。RL算法能够在最小的人工监督下从经验中学习，使其非常适合基因组数据分析和解释。使用RL的一个关键好处是降低了收集标记训练数据的成本，这是监督学习所需的。虽然已经有许多研究探讨了机器学习在基因组学中的应用，但本调查仅专注于在各种基因组研究领域（包括基因调控网络，基因组组装和序列比对）中使用RL的情况。我们对现有研究的技术细节进行了全面的概述。

    In recent years, Reinforcement Learning (RL) has emerged as a powerful tool for solving a wide range of problems, including decision-making and genomics. The exponential growth of raw genomic data over the past two decades has exceeded the capacity of manual analysis, leading to a growing interest in automatic data analysis and processing. RL algorithms are capable of learning from experience with minimal human supervision, making them well-suited for genomic data analysis and interpretation. One of the key benefits of using RL is the reduced cost associated with collecting labeled training data, which is required for supervised learning. While there have been numerous studies examining the applications of Machine Learning (ML) in genomics, this survey focuses exclusively on the use of RL in various genomics research fields, including gene regulatory networks (GRNs), genome assembly, and sequence alignment. We present a comprehensive technical overview of existing studies on the applic
    
[^36]: 语言模型显示对推理任务具有类似人类的内容效应

    Language models show human-like content effects on reasoning tasks. (arXiv:2207.07051v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2207.07051](http://arxiv.org/abs/2207.07051)

    本研究探讨了语言模型在逻辑推理任务中是否像人类一样通过混入内容来影响答案，结果发现大型语言模型的先验期望能够捕捉到这种特征。

    

    抽象推理是智能系统的关键能力。大型语言模型在抽象推理任务上实现了高于随机的性能，但存在许多不完善之处。然而，人类的抽象推理也是不完美的。例如，人类推理受到我们对真实世界的知识和信念的影响，并表现出显著的“内容效应”；当问题的语义内容支持正确的逻辑推理时，人类更可靠地进行推理。这些内容纠缠的推理模式在关于人类智能基本性质的争论中起着核心作用。在这里，我们研究了语言模型是否以类似的方式混入内容来回答逻辑问题，这些语言模型的先验期望捕捉了一些人类知识的特征。我们在三个逻辑推理任务上探索了这个问题：自然语言推理、判断三段论的逻辑有效性和Wason选择任务。我们评估了最先进的大型语言模型的性能。

    Abstract reasoning is a key ability for an intelligent system. Large language models (LMs) achieve above-chance performance on abstract reasoning tasks, but exhibit many imperfections. However, human abstract reasoning is also imperfect. For example, human reasoning is affected by our real-world knowledge and beliefs, and shows notable "content effects"; humans reason more reliably when the semantic content of a problem supports the correct logical inferences. These content-entangled reasoning patterns play a central role in debates about the fundamental nature of human intelligence. Here, we investigate whether language models $\unicode{x2014}$ whose prior expectations capture some aspects of human knowledge $\unicode{x2014}$ similarly mix content into their answers to logical problems. We explored this question across three logical reasoning tasks: natural language inference, judging the logical validity of syllogisms, and the Wason selection task. We evaluate state of the art large 
    

