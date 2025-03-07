# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning truly monotone operators with applications to nonlinear inverse problems](https://arxiv.org/abs/2404.00390) | 通过新定义的惩罚损失学习单调神经网络，解决图像处理中的问题，并利用FBF算法提供收敛保证，以解决非线性逆问题。 |
| [^2] | [Estimation of multiple mean vectors in high dimension](https://arxiv.org/abs/2403.15038) | 通过凸组合的方法估计高维空间中不同概率分布的多维均值，引入了两种权重确定策略：一种通过测试程序识别低方差的相邻均值，提出了封闭形式插补公式；另一种通过最小化二次风险的上置信界确定权重，通过理论分析得出方法对经验均值的二次风险改进，在维度渐近的角度上渐近地接近 Oracle（Minimax）改进。 |
| [^3] | [Are Vision Language Models Texture or Shape Biased and Can We Steer Them?](https://arxiv.org/abs/2403.09193) | 本文研究了广泛应用的视觉语言模型中的纹理与形状偏见，发现这些模型通常比视觉编码器更偏向形状，暗示视觉偏见在一定程度上会受到文本的调节 |
| [^4] | [Data augmentation with automated machine learning: approaches and performance comparison with classical data augmentation methods](https://arxiv.org/abs/2403.08352) | 自动化机器学习的数据增强方法旨在自动化数据增强过程，为改善机器学习模型泛化性能提供了更高效的方式。 |
| [^5] | [On the Challenges and Opportunities in Generative AI](https://arxiv.org/abs/2403.00025) | 现代生成人工智能范例中存在关键的未解决挑战，如何解决这些挑战将进一步增强它们的能力、多功能性和可靠性，并为研究方向提供有价值的见解。 |
| [^6] | [GenCeption: Evaluate Multimodal LLMs with Unlabeled Unimodal Data](https://arxiv.org/abs/2402.14973) | 提出了一种名为GenCeption的新型MLLM评估框架，可以仅利用单模态数据评估跨模态语义一致性，并有效反映模型产生幻觉的倾向，具有较强的相关性和潜力于流行的MLLM基准结果。 |
| [^7] | [How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage](https://arxiv.org/abs/2402.10065) | 本论文研究了每个数据点的成员推断攻击，量化了每个数据点的成员泄露，并评估了两种隐私防御措施的效果。 |
| [^8] | [Which Frequencies do CNNs Need? Emergent Bottleneck Structure in Feature Learning](https://arxiv.org/abs/2402.08010) | 本文描述了CNN中卷积瓶颈（CBN）结构的出现，网络在前几层将输入表示转换为在少数频率和通道上受支持的表示，然后通过最后几层映射回输出。CBN秩定义了保留在瓶颈中的频率的数量和类型，并部分证明了参数范数与深度和CBN秩的比例成正比。此外，我们还展示了网络的参数范数依赖于函数的规则性。我们发现任何具有接近最优参数范数的网络都会展示出CBN结构，这解释了下采样的常见实践；我们还验证了CBN结构在下采样下仍然成立。最后，我们使用CBN结构来解释...（摘要完整内容请见正文） |
| [^9] | [Implicit Diffusion: Efficient Optimization through Stochastic Sampling](https://arxiv.org/abs/2402.05468) | 本文介绍了一种通过随机采样优化隐含分布的新算法，并提出了一种通用框架，用于在单个循环中同时进行优化和采样步骤。实验结果证明了该方法在真实环境中的有效性。 |
| [^10] | [PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation](https://arxiv.org/abs/2402.04355) | PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。 |
| [^11] | [Decentralized Sporadic Federated Learning: A Unified Methodology with Generalized Convergence Guarantees](https://arxiv.org/abs/2402.03448) | 本文提出了一种称为分散式间歇联邦学习（DSpodFL）的方法，它统一了分布式梯度下降（DGD）、随机闲话（RG）和分散式联邦平均（DFedAvg）等著名的分散优化方法。根据分析结果，DSpodFL能够在更一般的假设下达到几何收敛速率与最佳性差距的匹配。经过实验验证了该方法的有效性。 |
| [^12] | [Careful with that Scalpel: Improving Gradient Surgery with an EMA](https://arxiv.org/abs/2402.02998) | 通过将训练损失梯度和辅助梯度在训练梯度方向上的正交投影结合起来，使用EMA（指数移动平均）可以改进梯度手术，提高深度学习估计管道的性能。 |
| [^13] | [$\sigma$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples](https://arxiv.org/abs/2402.01879) | 该论文提出了一种新的基于梯度的$\ell_0$范数攻击方法$\sigma$-zero，其利用了$\ell_0$范数的可微近似和自适应投影运算符，能够在非凸和非可微的约束下优化，从而评估深度网络对稀疏$\ell_0$范数攻击的鲁棒性。 |
| [^14] | [Deep Augmentation: Self-Supervised Learning with Transformations in Activation Space](https://arxiv.org/abs/2303.14537) | 深度增强是一种利用dropout或PCA在神经网络中转换目标层的方法，有效改善性能和泛化能力。在对比学习任务中，在Transformers、ResNets和图神经网络等基础模型上，通过深度增强实现了显著的性能提升，但在监督问题上效果相反。 |
| [^15] | [Cross-Modal Prototype based Multimodal Federated Learning under Severely Missing Modality.](http://arxiv.org/abs/2401.13898) | 提出了一种适用于严重缺失模态的多模态联邦学习方法MFCPL，通过完整的原型提供多样的模态知识，解决了数据异质性和缺失模态带来的稳健性问题。 |
| [^16] | [Canonical normalizing flows for manifold learning.](http://arxiv.org/abs/2310.12743) | 该论文介绍了一种规范化正态流方法，用于流形学习。通过可学习的可逆变换将数据嵌入到高维空间中，从而实现了在流形上计算概率密度并优化网络参数的目标。然而，当前的方法在学习到的流形表示中存在着与流形关联且退化的内在基函数的问题。 |
| [^17] | [Deep Reinforcement Learning for Infinite Horizon Mean Field Problems in Continuous Spaces.](http://arxiv.org/abs/2309.10953) | 这项研究提出了一种基于深度强化学习的算法，通过将演员-评论家范式与均场分布表示配对，来解决连续空间中的均场博弈和均场控制问题，并使用朗之万动力学从分布中获取样本。该算法在渐近无限时域框架下使用线性二次基准进行评估。 |
| [^18] | [Introduction to Online Nonstochastic Control.](http://arxiv.org/abs/2211.09619) | 介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。 |
| [^19] | [Tutorial on amortized optimization.](http://arxiv.org/abs/2202.00665) | 该教程介绍了分摊优化的基础，并总结了其在变分推断、稀疏编码、元学习、控制、强化学习、凸优化、最优传输和深度平衡网络中的应用。 |

# 详细

[^1]: 学习真正单调算子及其在非线性逆问题中的应用

    Learning truly monotone operators with applications to nonlinear inverse problems

    [https://arxiv.org/abs/2404.00390](https://arxiv.org/abs/2404.00390)

    通过新定义的惩罚损失学习单调神经网络，解决图像处理中的问题，并利用FBF算法提供收敛保证，以解决非线性逆问题。

    

    本文介绍了一种通过新定义的惩罚损失来学习单调神经网络的新方法。该方法在解决一类变分问题中特别有效，特别是图像处理任务中常遇到的单调包含问题。采用前-后-前（FBF）算法来解决这些问题，在神经网络的Lipschitz常数未知的情况下也能提供解决方案。值得注意的是，FBF算法在学习算子单调的条件下提供收敛保证。借鉴即插即用的方法，我们的目标是将这些新学习的算子应用于解决非线性逆问题。为实现这一目标，我们首先将问题制定为一个变分包含问题，随后训练一个单调神经网络来逼近一个本质上可能不是单调的算子。利用FBF算法

    arXiv:2404.00390v1 Announce Type: cross  Abstract: This article introduces a novel approach to learning monotone neural networks through a newly defined penalization loss. The proposed method is particularly effective in solving classes of variational problems, specifically monotone inclusion problems, commonly encountered in image processing tasks. The Forward-Backward-Forward (FBF) algorithm is employed to address these problems, offering a solution even when the Lipschitz constant of the neural network is unknown. Notably, the FBF algorithm provides convergence guarantees under the condition that the learned operator is monotone. Building on plug-and-play methodologies, our objective is to apply these newly learned operators to solving non-linear inverse problems. To achieve this, we initially formulate the problem as a variational inclusion problem. Subsequently, we train a monotone neural network to approximate an operator that may not inherently be monotone. Leveraging the FBF al
    
[^2]: 高维情况下多个均值向量的估计

    Estimation of multiple mean vectors in high dimension

    [https://arxiv.org/abs/2403.15038](https://arxiv.org/abs/2403.15038)

    通过凸组合的方法估计高维空间中不同概率分布的多维均值，引入了两种权重确定策略：一种通过测试程序识别低方差的相邻均值，提出了封闭形式插补公式；另一种通过最小化二次风险的上置信界确定权重，通过理论分析得出方法对经验均值的二次风险改进，在维度渐近的角度上渐近地接近 Oracle（Minimax）改进。

    

    我们致力于基于独立样本在一个共同空间中估计来自不同概率分布的多维均值。我们的方法是通过对这些样本导出的经验均值进行凸组合来形成估计量。我们引入了两种策略来找到适当的依赖于数据的凸组合权重：第一种利用测试程序来识别具有低方差的相邻均值，从而产生了一个关于权重的封闭形式插补公式；第二种通过最小化二次风险的上置信区间来确定权重。通过理论分析，我们评估了我们的方法相对于经验均值提供的二次风险改进。我们的分析集中在维度渐近的角度上，显示我们的方法在数据的有效维度增加时渐近地接近于一个 Oracle（Minimax）改进。我们展示了通过提出的方法在均值估计中的应用。

    arXiv:2403.15038v1 Announce Type: cross  Abstract: We endeavour to estimate numerous multi-dimensional means of various probability distributions on a common space based on independent samples. Our approach involves forming estimators through convex combinations of empirical means derived from these samples. We introduce two strategies to find appropriate data-dependent convex combination weights: a first one employing a testing procedure to identify neighbouring means with low variance, which results in a closed-form plug-in formula for the weights, and a second one determining weights via minimization of an upper confidence bound on the quadratic risk.Through theoretical analysis, we evaluate the improvement in quadratic risk offered by our methods compared to the empirical means. Our analysis focuses on a dimensional asymptotics perspective, showing that our methods asymptotically approach an oracle (minimax) improvement as the effective dimension of the data increases.We demonstrat
    
[^3]: 视觉语言模型是纹理偏见还是形状偏见，我们可以引导它们吗？

    Are Vision Language Models Texture or Shape Biased and Can We Steer Them?

    [https://arxiv.org/abs/2403.09193](https://arxiv.org/abs/2403.09193)

    本文研究了广泛应用的视觉语言模型中的纹理与形状偏见，发现这些模型通常比视觉编码器更偏向形状，暗示视觉偏见在一定程度上会受到文本的调节

    

    arXiv:2403.09193v1 公告类型: 跨领域 摘要: 视觉语言模型（VLMs）在短短几年内彻底改变了计算机视觉模型的格局，开启了一系列新的应用，从零样本图像分类到图像字幕生成，再到视觉问答。与纯视觉模型不同，它们提供了通过语言提示访问视觉内容的直观方式。这种模型的广泛适用性引发我们思考它们是否也与人类视觉一致 - 具体来说，它们在多模态融合中有多大程度地采用了人类引导的视觉偏见，或者它们是否只是从纯视觉模型中继承了偏见。其中一个重要的视觉偏见是纹理与形状偏见，即局部信息的主导地位。在本文中，我们研究了一系列流行的VLMs中的这种偏见。有趣的是，我们发现VLMs通常比它们的视觉编码器更偏向于形状，这表明视觉偏见在一定程度上通过文本进行调节。

    arXiv:2403.09193v1 Announce Type: cross  Abstract: Vision language models (VLMs) have drastically changed the computer vision model landscape in only a few years, opening an exciting array of new applications from zero-shot image classification, over to image captioning, and visual question answering. Unlike pure vision models, they offer an intuitive way to access visual content through language prompting. The wide applicability of such models encourages us to ask whether they also align with human vision - specifically, how far they adopt human-induced visual biases through multimodal fusion, or whether they simply inherit biases from pure vision models. One important visual bias is the texture vs. shape bias, or the dominance of local over global information. In this paper, we study this bias in a wide range of popular VLMs. Interestingly, we find that VLMs are often more shape-biased than their vision encoders, indicating that visual biases are modulated to some extent through text
    
[^4]: 利用自动化机器学习的数据增强方法及与传统数据增强方法性能比较

    Data augmentation with automated machine learning: approaches and performance comparison with classical data augmentation methods

    [https://arxiv.org/abs/2403.08352](https://arxiv.org/abs/2403.08352)

    自动化机器学习的数据增强方法旨在自动化数据增强过程，为改善机器学习模型泛化性能提供了更高效的方式。

    

    数据增强被认为是常用于提高机器学习模型泛化性能的最重要的正则化技术。它主要涉及应用适当的数据转换操作，以创建具有所需属性的新数据样本。尽管其有效性，这一过程通常具有挑战性，因为手动创建和测试不同候选增强及其超参数需耗费大量时间。自动化数据增强方法旨在自动化这一过程。最先进的方法通常依赖于自动化机器学习（AutoML）原则。本研究提供了基于AutoML的数据增强技术的全面调查。我们讨论了使用AutoML实现数据增强的各种方法，包括数据操作、数据集成和数据合成技术。我们详细讨论了技术

    arXiv:2403.08352v1 Announce Type: cross  Abstract: Data augmentation is arguably the most important regularization technique commonly used to improve generalization performance of machine learning models. It primarily involves the application of appropriate data transformation operations to create new data samples with desired properties. Despite its effectiveness, the process is often challenging because of the time-consuming trial and error procedures for creating and testing different candidate augmentations and their hyperparameters manually. Automated data augmentation methods aim to automate the process. State-of-the-art approaches typically rely on automated machine learning (AutoML) principles. This work presents a comprehensive survey of AutoML-based data augmentation techniques. We discuss various approaches for accomplishing data augmentation with AutoML, including data manipulation, data integration and data synthesis techniques. We present extensive discussion of technique
    
[^5]: 关于生成人工智能中的挑战与机遇

    On the Challenges and Opportunities in Generative AI

    [https://arxiv.org/abs/2403.00025](https://arxiv.org/abs/2403.00025)

    现代生成人工智能范例中存在关键的未解决挑战，如何解决这些挑战将进一步增强它们的能力、多功能性和可靠性，并为研究方向提供有价值的见解。

    

    深度生成建模领域近年来增长迅速而稳定。随着海量训练数据的可用性以及可扩展的无监督学习范式的进步，最近的大规模生成模型展现出合成高分辨率图像和文本以及结构化数据（如视频和分子）的巨大潜力。然而，我们认为当前大规模生成人工智能模型没有充分解决若干基本问题，限制了它们在各个领域的广泛应用。在本工作中，我们旨在确定现代生成人工智能范例中的关键未解决挑战，以进一步增强它们的能力、多功能性和可靠性。通过识别这些挑战，我们旨在为研究人员提供有价值的见解，探索有益的研究方向，从而促进更加强大和可访问的生成人工智能的发展。

    arXiv:2403.00025v1 Announce Type: cross  Abstract: The field of deep generative modeling has grown rapidly and consistently over the years. With the availability of massive amounts of training data coupled with advances in scalable unsupervised learning paradigms, recent large-scale generative models show tremendous promise in synthesizing high-resolution images and text, as well as structured data such as videos and molecules. However, we argue that current large-scale generative AI models do not sufficiently address several fundamental issues that hinder their widespread adoption across domains. In this work, we aim to identify key unresolved challenges in modern generative AI paradigms that should be tackled to further enhance their capabilities, versatility, and reliability. By identifying these challenges, we aim to provide researchers with valuable insights for exploring fruitful research directions, thereby fostering the development of more robust and accessible generative AI so
    
[^6]: GenCeption：使用未标记的单模态数据评估多模态LLM

    GenCeption: Evaluate Multimodal LLMs with Unlabeled Unimodal Data

    [https://arxiv.org/abs/2402.14973](https://arxiv.org/abs/2402.14973)

    提出了一种名为GenCeption的新型MLLM评估框架，可以仅利用单模态数据评估跨模态语义一致性，并有效反映模型产生幻觉的倾向，具有较强的相关性和潜力于流行的MLLM基准结果。

    

    多模态大型语言模型（MLLMs）通常使用昂贵的带标注的多模态基准进行评估。然而，这些基准通常难以跟上MLLM评估的快速发展要求。我们提出了GenCeption，这是一个新颖的无需注释的MLLM评估框架，仅需要单模态数据来评估跨模态语义一致性，并反映出模型产生幻觉的倾向。类似于流行的DrawCeption游戏，GenCeption从一个非文本样本开始，并经历一系列迭代的描述和生成步骤。迭代之间的语义漂移使用GC@T指标进行量化。我们的实证发现验证了GenCeption的有效性，并显示出与流行的MLLM基准结果的强相关性。GenCeption可以通过利用普遍存在且以前未见的单模态数据来扩展，以减轻训练数据的污染。

    arXiv:2402.14973v1 Announce Type: cross  Abstract: Multimodal Large Language Models (MLLMs) are commonly evaluated using costly annotated multimodal benchmarks. However, these benchmarks often struggle to keep pace with the rapidly advancing requirements of MLLM evaluation. We propose GenCeption, a novel and annotation-free MLLM evaluation framework that merely requires unimodal data to assess inter-modality semantic coherence and inversely reflects the models' inclination to hallucinate. Analogous to the popular DrawCeption game, GenCeption initiates with a non-textual sample and undergoes a series of iterative description and generation steps. Semantic drift across iterations is quantified using the GC@T metric. Our empirical findings validate GenCeption's efficacy, showing strong correlations with popular MLLM benchmarking results. GenCeption may be extended to mitigate training data contamination by utilizing ubiquitous, previously unseen unimodal data.
    
[^7]: 每个数据点泄露您隐私的程度有多大？量化每个数据点的成员泄露

    How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage

    [https://arxiv.org/abs/2402.10065](https://arxiv.org/abs/2402.10065)

    本论文研究了每个数据点的成员推断攻击，量化了每个数据点的成员泄露，并评估了两种隐私防御措施的效果。

    

    我们研究了每个数据点的成员推断攻击（MIAs），其中攻击者旨在推断出一个固定目标数据是否已包含在算法的输入数据集中，从而侵犯隐私。首先，我们定义每个数据点的成员泄露为最优对手辨识它的优势。然后，我们量化了经验均值的每个数据点的成员泄露，并表明它取决于目标数据点和数据生成分布之间的马氏距离。我们进一步评估了两种隐私防御措施的效果，即添加高斯噪声和子采样。我们准确地量化了它们都如何降低每个数据点的成员泄露。我们的分析建立在一个结合了似然比检验的Edgeworth展开和Lindeberg-Feller中心极限定理的新型证明技术上。我们的分析连接了现有的似然比和标量乘积攻击，并对这些攻击进行了论证。

    arXiv:2402.10065v1 Announce Type: new  Abstract: We study the per-datum Membership Inference Attacks (MIAs), where an attacker aims to infer whether a fixed target datum has been included in the input dataset of an algorithm and thus, violates privacy. First, we define the membership leakage of a datum as the advantage of the optimal adversary targeting to identify it. Then, we quantify the per-datum membership leakage for the empirical mean, and show that it depends on the Mahalanobis distance between the target datum and the data-generating distribution. We further assess the effect of two privacy defences, i.e. adding Gaussian noise and sub-sampling. We quantify exactly how both of them decrease the per-datum membership leakage. Our analysis builds on a novel proof technique that combines an Edgeworth expansion of the likelihood ratio test and a Lindeberg-Feller central limit theorem. Our analysis connects the existing likelihood ratio and scalar product attacks, and also justifies 
    
[^8]: CNN需要哪些频率？特征学习中的紧急瓶颈结构的出现

    Which Frequencies do CNNs Need? Emergent Bottleneck Structure in Feature Learning

    [https://arxiv.org/abs/2402.08010](https://arxiv.org/abs/2402.08010)

    本文描述了CNN中卷积瓶颈（CBN）结构的出现，网络在前几层将输入表示转换为在少数频率和通道上受支持的表示，然后通过最后几层映射回输出。CBN秩定义了保留在瓶颈中的频率的数量和类型，并部分证明了参数范数与深度和CBN秩的比例成正比。此外，我们还展示了网络的参数范数依赖于函数的规则性。我们发现任何具有接近最优参数范数的网络都会展示出CBN结构，这解释了下采样的常见实践；我们还验证了CBN结构在下采样下仍然成立。最后，我们使用CBN结构来解释...（摘要完整内容请见正文）

    

    我们描述了CNN中卷积瓶颈（CBN）结构的出现，网络使用其前几层将输入表示转换为仅在几个频率和通道上受支持的表示，然后使用最后几层将其映射回输出。我们定义了CBN秩，描述了保留在瓶颈内的频率的数量和类型，并在一定程度上证明了表示函数$f$所需的参数范数按深度乘以CBN秩$f$的比例缩放。我们还展示了参数范数在下一阶中依赖于$f$的正则性。我们展示了任何具有近乎最优参数范数的网络都会在权重和（在网络对大学习率稳定的假设下）激活中表现出CBN结构，这促使了下采样的常见做法；并且我们验证了CBN结构在下采样下仍然成立。最后，我们使用CBN结构来解释...

    We describe the emergence of a Convolution Bottleneck (CBN) structure in CNNs, where the network uses its first few layers to transform the input representation into a representation that is supported only along a few frequencies and channels, before using the last few layers to map back to the outputs. We define the CBN rank, which describes the number and type of frequencies that are kept inside the bottleneck, and partially prove that the parameter norm required to represent a function $f$ scales as depth times the CBN rank $f$. We also show that the parameter norm depends at next order on the regularity of $f$. We show that any network with almost optimal parameter norm will exhibit a CBN structure in both the weights and - under the assumption that the network is stable under large learning rate - the activations, which motivates the common practice of down-sampling; and we verify that the CBN results still hold with down-sampling. Finally we use the CBN structure to interpret the
    
[^9]: 隐式扩散: 通过随机采样实现高效优化

    Implicit Diffusion: Efficient Optimization through Stochastic Sampling

    [https://arxiv.org/abs/2402.05468](https://arxiv.org/abs/2402.05468)

    本文介绍了一种通过随机采样优化隐含分布的新算法，并提出了一种通用框架，用于在单个循环中同时进行优化和采样步骤。实验结果证明了该方法在真实环境中的有效性。

    

    我们提出了一种通过参数化随机扩散隐式定义的分布来进行优化的新算法。通过优化这些参数，可以修改采样过程的结果分布。我们引入了一个针对这些过程的一阶优化的通用框架，通过在单个循环中进行优化和采样步骤来实现。这种方法受到双层优化和自动隐式微分的最新进展的启发，利用采样作为在概率分布空间上进行优化的视角。我们提供了关于我们方法性能的理论保证，以及在实际环境中证明其有效性的实验结果。

    We present a new algorithm to optimize distributions defined implicitly by parameterized stochastic diffusions. Doing so allows us to modify the outcome distribution of sampling processes by optimizing over their parameters. We introduce a general framework for first-order optimization of these processes, that performs jointly, in a single loop, optimization and sampling steps. This approach is inspired by recent advances in bilevel optimization and automatic implicit differentiation, leveraging the point of view of sampling as optimization over the space of probability distributions. We provide theoretical guarantees on the performance of our method, as well as experimental results demonstrating its effectiveness in real-world settings.
    
[^10]: PQMass: 使用概率质量估计的生成模型质量的概率评估

    PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

    [https://arxiv.org/abs/2402.04355](https://arxiv.org/abs/2402.04355)

    PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。

    

    我们提出了一种全面的基于样本的方法来评估生成模型的质量。所提出的方法能够估计两个样本集合来自同一分布的概率，为评估单个生成模型的性能或比较在同一数据集上训练的多个竞争模型提供了一个统计上严格的方法。该比较可以通过将空间划分为非重叠的区域并比较每个区域中的数据样本数量来进行。该方法仅需要生成模型和测试数据的样本。它能够直接处理高维数据，无需降维。显著的是，该方法不依赖于关于真实分布密度的假设，并且不依赖于训练或拟合任何辅助模型。相反，它着重于近似计算密度的积分（概率质量）。

    We propose a comprehensive sample-based method for assessing the quality of generative models. The proposed approach enables the estimation of the probability that two sets of samples are drawn from the same distribution, providing a statistically rigorous method for assessing the performance of a single generative model or the comparison of multiple competing models trained on the same dataset. This comparison can be conducted by dividing the space into non-overlapping regions and comparing the number of data samples in each region. The method only requires samples from the generative model and the test data. It is capable of functioning directly on high-dimensional data, obviating the need for dimensionality reduction. Significantly, the proposed method does not depend on assumptions regarding the density of the true distribution, and it does not rely on training or fitting any auxiliary models. Instead, it focuses on approximating the integral of the density (probability mass) acros
    
[^11]: 分散式间歇联邦学习：具有广义收敛保证的统一方法

    Decentralized Sporadic Federated Learning: A Unified Methodology with Generalized Convergence Guarantees

    [https://arxiv.org/abs/2402.03448](https://arxiv.org/abs/2402.03448)

    本文提出了一种称为分散式间歇联邦学习（DSpodFL）的方法，它统一了分布式梯度下降（DGD）、随机闲话（RG）和分散式联邦平均（DFedAvg）等著名的分散优化方法。根据分析结果，DSpodFL能够在更一般的假设下达到几何收敛速率与最佳性差距的匹配。经过实验验证了该方法的有效性。

    

    分散式联邦学习（DFL）近来受到了重要的研究关注，涵盖了模型更新和模型聚合这两个关键联邦学习过程都由客户端进行的设置。在本文中，我们提出了分散式间歇联邦学习（DSpodFL），这是一种DFL方法，它在这两个过程中广义化了间歇性的概念，建模了在实际DFL设置中出现的不同形式的异质性的影响。DSpodFL将许多着名的分散优化方法，如分布式梯度下降（DGD），随机闲话（RG）和分散式联邦平均（DFedAvg），统一到一个建模框架下。我们对DSpodFL的收敛行为进行了分析，显示出可以在更一般的假设下，将几何收敛速率与有限的最佳性差距相匹配。通过实验证明：

    Decentralized Federated Learning (DFL) has received significant recent research attention, capturing settings where both model updates and model aggregations -- the two key FL processes -- are conducted by the clients. In this work, we propose Decentralized Sporadic Federated Learning ($\texttt{DSpodFL}$), a DFL methodology which generalizes the notion of sporadicity in both of these processes, modeling the impact of different forms of heterogeneity that manifest in realistic DFL settings. $\texttt{DSpodFL}$ unifies many of the prominent decentralized optimization methods, e.g., distributed gradient descent (DGD), randomized gossip (RG), and decentralized federated averaging (DFedAvg), under a single modeling framework. We analytically characterize the convergence behavior of $\texttt{DSpodFL}$, showing, among other insights, that we can match a geometric convergence rate to a finite optimality gap under more general assumptions than in existing works. Through experiments, we demonstra
    
[^12]: 小心使用手术刀：使用EMA改进梯度手术

    Careful with that Scalpel: Improving Gradient Surgery with an EMA

    [https://arxiv.org/abs/2402.02998](https://arxiv.org/abs/2402.02998)

    通过将训练损失梯度和辅助梯度在训练梯度方向上的正交投影结合起来，使用EMA（指数移动平均）可以改进梯度手术，提高深度学习估计管道的性能。

    

    在深度学习估计管道中，除了最小化单一的训练损失外，还依赖于辅助目标来量化和鼓励模型的可取属性（例如在另一个数据集上的表现，鲁棒性，与先前的一致性）。虽然将辅助损失与训练损失相加作为正则化的最简单方法，但最近的研究表明，通过混合梯度而不仅仅是简单相加，可以提高性能；这被称为梯度手术。我们将这个问题看作是一个约束最小化问题，其中辅助目标在训练损失的最小化集合中被最小化。为了解决这个双层问题，我们采用了一个参数更新方向，它将训练损失梯度和辅助梯度在训练梯度方向上的正交投影结合起来。在梯度来自小批次的情况下，我们解释了如何使用训练损失梯度的移动平均来维护。

    Beyond minimizing a single training loss, many deep learning estimation pipelines rely on an auxiliary objective to quantify and encourage desirable properties of the model (e.g. performance on another dataset, robustness, agreement with a prior). Although the simplest approach to incorporating an auxiliary loss is to sum it with the training loss as a regularizer, recent works have shown that one can improve performance by blending the gradients beyond a simple sum; this is known as gradient surgery. We cast the problem as a constrained minimization problem where the auxiliary objective is minimized among the set of minimizers of the training loss. To solve this bilevel problem, we follow a parameter update direction that combines the training loss gradient and the orthogonal projection of the auxiliary gradient to the training gradient. In a setting where gradients come from mini-batches, we explain how, using a moving average of the training loss gradients, we can carefully maintain
    
[^13]: $\sigma$-zero: 基于梯度的$\ell_0$-范数对抗样本优化

    $\sigma$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples

    [https://arxiv.org/abs/2402.01879](https://arxiv.org/abs/2402.01879)

    该论文提出了一种新的基于梯度的$\ell_0$范数攻击方法$\sigma$-zero，其利用了$\ell_0$范数的可微近似和自适应投影运算符，能够在非凸和非可微的约束下优化，从而评估深度网络对稀疏$\ell_0$范数攻击的鲁棒性。

    

    评估深度网络对基于梯度攻击的对抗鲁棒性是具有挑战性的。虽然大多数攻击考虑$\ell_2$和$\ell_\infty$范数约束来制造输入扰动，但只有少数研究了稀疏的$\ell_1$和$\ell_0$范数攻击。特别是，由于在非凸且非可微约束上进行优化的固有复杂性，$\ell_0$范数攻击是研究最少的。然而，使用这些攻击评估对抗鲁棒性可以揭示在更传统的$\ell_2$和$\ell_\infty$范数攻击中未能测试出的弱点。在这项工作中，我们提出了一种新颖的$\ell_0$范数攻击，称为$\sigma$-zero，它利用了$\ell_0$范数的一个特殊可微近似来促进基于梯度的优化，并利用自适应投影运算符动态调整损失最小化和扰动稀疏性之间的权衡。通过在MNIST、CIFAR10和ImageNet数据集上进行广泛评估，包括...

    Evaluating the adversarial robustness of deep networks to gradient-based attacks is challenging. While most attacks consider $\ell_2$- and $\ell_\infty$-norm constraints to craft input perturbations, only a few investigate sparse $\ell_1$- and $\ell_0$-norm attacks. In particular, $\ell_0$-norm attacks remain the least studied due to the inherent complexity of optimizing over a non-convex and non-differentiable constraint. However, evaluating adversarial robustness under these attacks could reveal weaknesses otherwise left untested with more conventional $\ell_2$- and $\ell_\infty$-norm attacks. In this work, we propose a novel $\ell_0$-norm attack, called $\sigma$-zero, which leverages an ad hoc differentiable approximation of the $\ell_0$ norm to facilitate gradient-based optimization, and an adaptive projection operator to dynamically adjust the trade-off between loss minimization and perturbation sparsity. Extensive evaluations using MNIST, CIFAR10, and ImageNet datasets, involving
    
[^14]: 深度增强：在激活空间中使用自监督学习进行数据增强

    Deep Augmentation: Self-Supervised Learning with Transformations in Activation Space

    [https://arxiv.org/abs/2303.14537](https://arxiv.org/abs/2303.14537)

    深度增强是一种利用dropout或PCA在神经网络中转换目标层的方法，有效改善性能和泛化能力。在对比学习任务中，在Transformers、ResNets和图神经网络等基础模型上，通过深度增强实现了显著的性能提升，但在监督问题上效果相反。

    

    我们提出了一种称为深度增强的方法，通过使用辍学或PCA来转换神经网络中的目标层，以提高性能和泛化能力。我们通过在自然语言处理、计算机视觉和图学习中的对比学习任务上进行大量实验来展示深度增强。 我们观察到在对比学习的基础模型中，如Transformers、ResNets和图神经网络上深度增强能够带来显著的性能提升，但在相应的监督问题上观察到相反的效果。 我们的分析表明，深度增强减轻了层之间的相互适应，即"崩溃"形式的问题。 我们利用这一观察结果制定了一种选择目标层的方法；特别是，我们的实验表明，用深度增强定位更深层次的层要优于增强输入数据。 这种方法的简单网络和模态无关性使其

    arXiv:2303.14537v2 Announce Type: replace-cross  Abstract: We introduce Deep Augmentation, an approach to implicit data augmentation using dropout or PCA to transform a targeted layer within a neural network to improve performance and generalization. We demonstrate Deep Augmentation through extensive experiments on contrastive learning tasks in NLP, computer vision, and graph learning. We observe substantial performance gains with Transformers, ResNets, and Graph Neural Networks as the underlying models in contrastive learning, but observe inverse effects on the corresponding supervised problems. Our analysis suggests that Deep Augmentation alleviates co-adaption between layers, a form of "collapse." We use this observation to formulate a method for selecting which layer to target; in particular, our experimentation reveals that targeting deeper layers with Deep Augmentation outperforms augmenting the input data. The simple network- and modality-agnostic nature of this approach enables
    
[^15]: 跨模态原型基础的多模态联邦学习在严重缺失模态下的应用

    Cross-Modal Prototype based Multimodal Federated Learning under Severely Missing Modality. (arXiv:2401.13898v1 [cs.LG])

    [http://arxiv.org/abs/2401.13898](http://arxiv.org/abs/2401.13898)

    提出了一种适用于严重缺失模态的多模态联邦学习方法MFCPL，通过完整的原型提供多样的模态知识，解决了数据异质性和缺失模态带来的稳健性问题。

    

    多模态联邦学习（MFL）作为一种去中心化的机器学习范例已经出现，它允许具有不同模态的多个客户端在不共享私人数据的情况下合作训练机器学习模型，跨多样的数据源。然而，数据异质性和严重缺失模态等挑战给MFL的稳健性带来重要阻碍，严重影响全局模型的性能。在本文中，我们提出了一种适用于严重缺失模态下的多模态联邦学习的新方法，即多模态联邦交叉原型学习（MFCPL），通过对模态共享级别进行完整的原型来提供多样的模态知识。

    Multimodal federated learning (MFL) has emerged as a decentralized machine learning paradigm, allowing multiple clients with different modalities to collaborate on training a machine learning model across diverse data sources without sharing their private data. However, challenges, such as data heterogeneity and severely missing modalities, pose crucial hindrances to the robustness of MFL, significantly impacting the performance of global model. The absence of a modality introduces misalignment during the local training phase, stemming from zero-filling in the case of clients with missing modalities. Consequently, achieving robust generalization in global model becomes imperative, especially when dealing with clients that have incomplete data. In this paper, we propose Multimodal Federated Cross Prototype Learning (MFCPL), a novel approach for MFL under severely missing modalities by conducting the complete prototypes to provide diverse modality knowledge in modality-shared level with 
    
[^16]: 流形学习的规范化正态流

    Canonical normalizing flows for manifold learning. (arXiv:2310.12743v1 [stat.ML])

    [http://arxiv.org/abs/2310.12743](http://arxiv.org/abs/2310.12743)

    该论文介绍了一种规范化正态流方法，用于流形学习。通过可学习的可逆变换将数据嵌入到高维空间中，从而实现了在流形上计算概率密度并优化网络参数的目标。然而，当前的方法在学习到的流形表示中存在着与流形关联且退化的内在基函数的问题。

    

    流形学习流是一类生成建模技术，假设数据具有低维流形描述。通过可学习的可逆变换将这种流形嵌入到数据的高维空间中。因此，一旦通过重构损失正确对齐流形，流形上的概率密度就是可计算的，并且可以使用最大似然来优化网络参数。自然地，数据的低维表示需要是单射映射。最近的方法能够在建模的流形上对密度进行对准，并在嵌入到高维空间时高效计算密度体积变化项。然而，除非单射映射在解析上预定义，否则学习到的流形不一定是数据的有效表示。也就是说，这种模型的潜在维度经常会学习到与流形相关并且退化的内在基函数。

    Manifold learning flows are a class of generative modelling techniques that assume a low-dimensional manifold description of the data. The embedding of such manifold into the high-dimensional space of the data is achieved via learnable invertible transformations. Therefore, once the manifold is properly aligned via a reconstruction loss, the probability density is tractable on the manifold and maximum likelihood can be used optimize the network parameters. Naturally, the lower-dimensional representation of the data requires an injective-mapping. Recent approaches were able to enforce that density aligns with the modelled manifold, while efficiently calculating the density volume-change term when embedding to the higher-dimensional space. However, unless the injective-mapping is analytically predefined, the learned manifold is not necessarily an efficient representation of the data. Namely, the latent dimensions of such models frequently learn an entangled intrinsic basis with degenerat
    
[^17]: 基于深度强化学习的连续空间无限时域均场问题解决方法

    Deep Reinforcement Learning for Infinite Horizon Mean Field Problems in Continuous Spaces. (arXiv:2309.10953v1 [math.OC])

    [http://arxiv.org/abs/2309.10953](http://arxiv.org/abs/2309.10953)

    这项研究提出了一种基于深度强化学习的算法，通过将演员-评论家范式与均场分布表示配对，来解决连续空间中的均场博弈和均场控制问题，并使用朗之万动力学从分布中获取样本。该算法在渐近无限时域框架下使用线性二次基准进行评估。

    

    我们提出了一种强化学习算法，用于统一解决连续空间均场博弈（MFG）和均场控制（MFC）问题，并对其进行了分析和发展。所提出的方法将演员-评论家（AC）范式与通过参数化评分函数表示的均场分布配对，可以以在线方式有效地更新，并使用朗之万动力学从得到的分布中获得样本。AC代理和评分函数按迭代方式进行更新，以收敛到给定均场问题的MFG平衡或MFC最优解，具体取决于学习率的选择。算法的简单修改使我们能够解决混合均场控制博弈（MFCG）。我们使用渐近无限时域框架中的线性二次基准评估我们的算法性能。

    We present the development and analysis of a reinforcement learning (RL) algorithm designed to solve continuous-space mean field game (MFG) and mean field control (MFC) problems in a unified manner. The proposed approach pairs the actor-critic (AC) paradigm with a representation of the mean field distribution via a parameterized score function, which can be efficiently updated in an online fashion, and uses Langevin dynamics to obtain samples from the resulting distribution. The AC agent and the score function are updated iteratively to converge, either to the MFG equilibrium or the MFC optimum for a given mean field problem, depending on the choice of learning rates. A straightforward modification of the algorithm allows us to solve mixed mean field control games (MFCGs). The performance of our algorithm is evaluated using linear-quadratic benchmarks in the asymptotic infinite horizon framework.
    
[^18]: 在线非随机控制简介

    Introduction to Online Nonstochastic Control. (arXiv:2211.09619v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.09619](http://arxiv.org/abs/2211.09619)

    介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    

    本文介绍了一种新兴的动态系统控制与可微强化学习范式——在线非随机控制，并应用在线凸优化和凸松弛技术得到了具有可证明保证的新方法，在最佳和鲁棒控制方面取得了显著成果。与其他框架不同，该方法的目标是对抗性攻击，在无法预测扰动模型的情况下，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    This text presents an introduction to an emerging paradigm in control of dynamical systems and differentiable reinforcement learning called online nonstochastic control. The new approach applies techniques from online convex optimization and convex relaxations to obtain new methods with provable guarantees for classical settings in optimal and robust control.  The primary distinction between online nonstochastic control and other frameworks is the objective. In optimal control, robust control, and other control methodologies that assume stochastic noise, the goal is to perform comparably to an offline optimal strategy. In online nonstochastic control, both the cost functions as well as the perturbations from the assumed dynamical model are chosen by an adversary. Thus the optimal policy is not defined a priori. Rather, the target is to attain low regret against the best policy in hindsight from a benchmark class of policies.  This objective suggests the use of the decision making frame
    
[^19]: 关于分摊优化的教程

    Tutorial on amortized optimization. (arXiv:2202.00665v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.00665](http://arxiv.org/abs/2202.00665)

    该教程介绍了分摊优化的基础，并总结了其在变分推断、稀疏编码、元学习、控制、强化学习、凸优化、最优传输和深度平衡网络中的应用。

    

    优化是一种普遍的建模工具，经常在反复解决相同问题的情况下使用。分摊优化方法使用学习来预测这些设置中问题的解决方案，利用相似问题实例之间的共享结构。这些方法在变分推断和强化学习中至关重要，能够比不使用分摊的传统优化方法快几个数量级地解决优化问题。本次教程介绍了这些进步背后的分摊优化基础，并概述了它们在变分推断、稀疏编码、基于梯度的元学习、控制、强化学习、凸优化、最优传输和深度平衡网络中的应用。本教程的源代码可在https://github.com/facebookresearch/amortized-optimization-tutorial上获得。

    Optimization is a ubiquitous modeling tool and is often deployed in settings which repeatedly solve similar instances of the same problem. Amortized optimization methods use learning to predict the solutions to problems in these settings, exploiting the shared structure between similar problem instances. These methods have been crucial in variational inference and reinforcement learning and are capable of solving optimization problems many orders of magnitudes times faster than traditional optimization methods that do not use amortization. This tutorial presents an introduction to the amortized optimization foundations behind these advancements and overviews their applications in variational inference, sparse coding, gradient-based meta-learning, control, reinforcement learning, convex optimization, optimal transport, and deep equilibrium networks. The source code for this tutorial is available at https://github.com/facebookresearch/amortized-optimization-tutorial.
    

