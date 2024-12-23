# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [From Algorithms to Outcomes: Reviewing AI's Role in Non-Muscle-Invasive Bladder Cancer Recurrence Prediction](https://arxiv.org/abs/2403.10586) | 机器学习技术在非肌层侵袭性膀胱癌复发预测中具有潜在作用，可以提高准确性，降低治疗成本，并有效规划治疗方案 |
| [^2] | [Towards Adversarially Robust Dataset Distillation by Curvature Regularization](https://arxiv.org/abs/2403.10045) | 本文探讨了如何通过曲率正则化方法在精炼数据集中嵌入对抗鲁棒性，以保持模型高准确性并获得更好的对抗鲁棒性。 |
| [^3] | [Generalizing Denoising to Non-Equilibrium Structures Improves Equivariant Force Fields](https://arxiv.org/abs/2403.09549) | 将去噪方法推广到非平衡结构，从而改进等变力场的性能，提高了对原子间相互作用的理解以及在分子动力学和催化剂设计等领域的应用。 |
| [^4] | [Boosting, Voting Classifiers and Randomized Sample Compression Schemes](https://arxiv.org/abs/2402.02976) | 本研究提出了一种随机提升算法来解决传统提升算法的性能问题，并通过构建一个通用框架将样本压缩方法扩展到支持随机学习算法，实现了在样本大小上具有单对数依赖的泛化错误。 |
| [^5] | [Optimizing Heat Alert Issuance with Reinforcement Learning](https://arxiv.org/abs/2312.14196) | 本研究利用强化学习优化热预警系统，通过引入新颖强化学习环境和综合数据集，解决了气候和健康环境中的低信号效应和空间异质性。 |
| [^6] | [Protecting Sensitive Data through Federated Co-Training](https://arxiv.org/abs/2310.05696) | 提出了使用联合协同训练方法来保护敏感数据，通过在公共未标记数据集上共享硬标签代替模型参数，形成伪标签以结合私有数据训练本地模型，提高隐私保护效果并获得与联邦学习相媲美的模型质量。 |
| [^7] | [A survey on recent advances in named entity recognition.](http://arxiv.org/abs/2401.10825) | 这篇综述调查了最近的命名实体识别研究进展，并提供了对不同算法性能的深度比较，还探讨了数据集特征对方法行为的影响。 |
| [^8] | [Variational measurement-based quantum computation for generative modeling.](http://arxiv.org/abs/2310.13524) | 这项研究提出了一种基于测量的变分量子计算算法，将量子测量的随机性视为计算资源，并应用于生成建模任务。 |
| [^9] | [Residual Multi-Fidelity Neural Network Computing.](http://arxiv.org/abs/2310.03572) | 本研究提出了一种残差多保真计算框架，通过使用多保真信息构建神经网络代理模型，解决了低保真和高保真计算模型之间的相关性建模问题。这种方法训练了两个神经网络，利用残差函数进行模型训练，最终得到了高保真替代模型。 |
| [^10] | [Detecting Throat Cancer from Speech Signals Using Machine Learning: A Reproducible Literature Review.](http://arxiv.org/abs/2307.09230) | 本研究对使用机器学习和人工智能从语音记录中检测喉癌的文献进行了综述，发现了22篇相关论文，讨论了它们的方法和结果。研究使用了神经网络和梅尔频率倒谱系数提取音频特征，并通过迁移学习实现了分类，取得了一定的准确率。 |
| [^11] | [Learning ECG signal features without backpropagation.](http://arxiv.org/abs/2307.01930) | 该论文提出了一种用于生成时间序列数据表示的新方法，依靠理论物理的思想以数据驱动的方式构建紧凑的表示。该方法能够捕捉数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性，并可以在广义设置中应用。 |
| [^12] | [Layer-level activation mechanism.](http://arxiv.org/abs/2306.04940) | 去噪声更好，表现更好的分层级别激活机制 |
| [^13] | [Spatially-Aware Car-Sharing Demand Prediction.](http://arxiv.org/abs/2303.14421) | 本文提出了一种基于空间感知学习算法的方法来分析共享汽车站点的平均月度需求，利用了丰富的特征作为输入，可以提高预测性能。 |

# 详细

[^1]: 从算法到结果：审视人工智能在非肌层侵袭性膀胱癌复发预测中的作用

    From Algorithms to Outcomes: Reviewing AI's Role in Non-Muscle-Invasive Bladder Cancer Recurrence Prediction

    [https://arxiv.org/abs/2403.10586](https://arxiv.org/abs/2403.10586)

    机器学习技术在非肌层侵袭性膀胱癌复发预测中具有潜在作用，可以提高准确性，降低治疗成本，并有效规划治疗方案

    

    膀胱癌是英国每天造成15人死亡的领先泌尿道癌症。这种癌症主要表现为非肌层侵袭性膀胱癌（NMIBC），其特点是肿瘤还未渗透到膀胱壁的肌肉层。 NMIBC的复发率非常高，达到70-80％，因此治疗成本最高。目前用于预测复发的工具使用评分系统来高估风险，并具有较低的准确性。对复发的不准确和延迟预测显著提高了死亡的可能性。因此，准确预测复发对于成本效益的管理和治疗计划至关重要。这就是机器学习（ML）技术出现的地方，通过利用分子和临床数据预测NMIBC复发，成为一种有前途的方法。本次审查对预测NMIBC复发的ML方法进行了全面分析。我们的系统评估使

    arXiv:2403.10586v1 Announce Type: cross  Abstract: Bladder cancer, the leading urinary tract cancer, is responsible for 15 deaths daily in the UK. This cancer predominantly manifests as non-muscle-invasive bladder cancer (NMIBC), characterised by tumours not yet penetrating the muscle layer of the bladder wall. NMIBC is plagued by a very high recurrence rate of 70-80% and hence the costliest treatments. Current tools for predicting recurrence use scoring systems that overestimate risk and have poor accuracy. Inaccurate and delayed prediction of recurrence significantly elevates the likelihood of mortality. Accurate prediction of recurrence is hence vital for cost-effective management and treatment planning. This is where Machine learning (ML) techniques have emerged as a promising approach for predicting NMIBC recurrence by leveraging molecular and clinical data. This review provides a comprehensive analysis of ML approaches for predicting NMIBC recurrence. Our systematic evaluation de
    
[^2]: 通过曲率正则化实现对抗鲁棒性数据集精炼

    Towards Adversarially Robust Dataset Distillation by Curvature Regularization

    [https://arxiv.org/abs/2403.10045](https://arxiv.org/abs/2403.10045)

    本文探讨了如何通过曲率正则化方法在精炼数据集中嵌入对抗鲁棒性，以保持模型高准确性并获得更好的对抗鲁棒性。

    

    数据集精炼（DD）允许将数据集精炼为原始大小的分数，同时保留丰富的分布信息，使得在精炼数据集上训练的模型可以在节省显著计算负载的同时达到可比的准确性。最近在这一领域的研究集中在提高在精炼数据集上训练的模型的准确性。在本文中，我们旨在探索DD的一种新视角。我们研究如何在精炼数据集中嵌入对抗鲁棒性，以使在这些数据集上训练的模型保持高精度的同时获得更好的对抗鲁棒性。我们提出了一种通过将曲率正则化纳入到精炼过程中来实现这一目标的新方法，而这种方法的计算开销比标准的对抗训练要少得多。大量的实证实验表明，我们的方法不仅在准确性上优于标准对抗训练，同时在对抗性能方面也取得了显著改进。

    arXiv:2403.10045v1 Announce Type: new  Abstract: Dataset distillation (DD) allows datasets to be distilled to fractions of their original size while preserving the rich distributional information so that models trained on the distilled datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has been focusing on improving the accuracy of models trained on distilled datasets. In this paper, we aim to explore a new perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these datasets maintain the high accuracy and meanwhile acquire better adversarial robustness. We propose a new method that achieves this goal by incorporating curvature regularization into the distillation process with much less computational overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accur
    
[^3]: 将去噪推广到非平衡结构以改进等变力场

    Generalizing Denoising to Non-Equilibrium Structures Improves Equivariant Force Fields

    [https://arxiv.org/abs/2403.09549](https://arxiv.org/abs/2403.09549)

    将去噪方法推广到非平衡结构，从而改进等变力场的性能，提高了对原子间相互作用的理解以及在分子动力学和催化剂设计等领域的应用。

    

    理解原子间的相互作用，如3D原子体系中的力，对于许多应用如分子动力学和催化剂设计至关重要。然而，模拟这些相互作用需要计算密集的从头算计算，因此训练神经网络的数据有限。本文提出使用去噪非平衡结构（DeNS）作为辅助任务，以更好地利用训练数据并提高性能。在使用DeNS进行训练时，我们首先通过向其3D坐标添加噪声来破坏3D结构，然后预测噪声。不同于以往仅限于平衡结构的去噪工作，所提出的方法将去噪泛化到更大范围的非平衡结构。主要区别在于非平衡结构不对应于局部能量最小值，具有非零力，因此可能具有许多可能的原子位置。

    arXiv:2403.09549v1 Announce Type: cross  Abstract: Understanding the interactions of atoms such as forces in 3D atomistic systems is fundamental to many applications like molecular dynamics and catalyst design. However, simulating these interactions requires compute-intensive ab initio calculations and thus results in limited data for training neural networks. In this paper, we propose to use denoising non-equilibrium structures (DeNS) as an auxiliary task to better leverage training data and improve performance. For training with DeNS, we first corrupt a 3D structure by adding noise to its 3D coordinates and then predict the noise. Different from previous works on denoising, which are limited to equilibrium structures, the proposed method generalizes denoising to a much larger set of non-equilibrium structures. The main difference is that a non-equilibrium structure does not correspond to local energy minima and has non-zero forces, and therefore it can have many possible atomic posit
    
[^4]: 提升，投票分类器和随机采样压缩方案

    Boosting, Voting Classifiers and Randomized Sample Compression Schemes

    [https://arxiv.org/abs/2402.02976](https://arxiv.org/abs/2402.02976)

    本研究提出了一种随机提升算法来解决传统提升算法的性能问题，并通过构建一个通用框架将样本压缩方法扩展到支持随机学习算法，实现了在样本大小上具有单对数依赖的泛化错误。

    

    在提升中，我们旨在利用多个弱学习器来产生一个强学习器。这个范式的核心是将强学习器建模为一个投票分类器，它输出弱学习器的加权多数投票。尽管许多成功的提升算法，如标志性的AdaBoost，产生投票分类器，但它们的理论性能长期以来一直不够优化：迄今为止，已知的使投票分类器达到给定准确性所需的训练样本数的最佳界限总是至少包含至多两个对数因子，而这已经超过了一般的弱到强学习器所能实现的范围。在这项工作中，我们通过提出一种随机提升算法打破这一障碍，该算法输出的投票分类器在样本大小上包含单对数依赖的泛化错误。我们通过构建一个通用框架将样本压缩方法扩展到支持随机学习算法来获得这个结果。

    In boosting, we aim to leverage multiple weak learners to produce a strong learner. At the center of this paradigm lies the concept of building the strong learner as a voting classifier, which outputs a weighted majority vote of the weak learners. While many successful boosting algorithms, such as the iconic AdaBoost, produce voting classifiers, their theoretical performance has long remained sub-optimal: the best known bounds on the number of training examples necessary for a voting classifier to obtain a given accuracy has so far always contained at least two logarithmic factors above what is known to be achievable by general weak-to-strong learners. In this work, we break this barrier by proposing a randomized boosting algorithm that outputs voting classifiers whose generalization error contains a single logarithmic dependency on the sample size. We obtain this result by building a general framework that extends sample compression methods to support randomized learning algorithms ba
    
[^5]: 用强化学习优化热预警的发布

    Optimizing Heat Alert Issuance with Reinforcement Learning

    [https://arxiv.org/abs/2312.14196](https://arxiv.org/abs/2312.14196)

    本研究利用强化学习优化热预警系统，通过引入新颖强化学习环境和综合数据集，解决了气候和健康环境中的低信号效应和空间异质性。

    

    社会适应气候变化的关键战略之一是利用预警系统减少极端高温事件的不利健康影响，以促使预防性行动。本文研究了强化学习（RL）作为优化此类系统效果的工具。我们的贡献有三个方面。首先，我们引入了一个新颖的强化学习环境，评估热预警政策的有效性，以减少与高温有关的住院人数。奖励模型基于历史天气、医疗保险健康记录以及社会经济/地理特征的全面数据集进行训练。我们使用变分贝叶斯技术解决了在气候和健康环境中常见的低信号效应和空间异质性。转换模型结合了真实的历史天气模式，并通过基于气候区域相似性的数据增强机制进行增强。

    arXiv:2312.14196v2 Announce Type: replace  Abstract: A key strategy in societal adaptation to climate change is the use of alert systems to reduce the adverse health impacts of extreme heat events by prompting preventative action. In this work, we investigate reinforcement learning (RL) as a tool to optimize the effectiveness of such systems. Our contributions are threefold. First, we introduce a novel RL environment enabling the evaluation of the effectiveness of heat alert policies to reduce heat-related hospitalizations. The rewards model is trained from a comprehensive dataset of historical weather, Medicare health records, and socioeconomic/geographic features. We use variational Bayesian techniques to address low-signal effects and spatial heterogeneity, which are commonly encountered in climate & health settings. The transition model incorporates real historical weather patterns enriched by a data augmentation mechanism based on climate region similarity. Second, we use this env
    
[^6]: 通过联合协同训练保护敏感数据

    Protecting Sensitive Data through Federated Co-Training

    [https://arxiv.org/abs/2310.05696](https://arxiv.org/abs/2310.05696)

    提出了使用联合协同训练方法来保护敏感数据，通过在公共未标记数据集上共享硬标签代替模型参数，形成伪标签以结合私有数据训练本地模型，提高隐私保护效果并获得与联邦学习相媲美的模型质量。

    

    在许多应用中，敏感数据本质上是分布的，由于隐私问题可能无法汇总。联邦学习允许我们通过迭代地聚合本地模型的参数来协作训练模型，而无需合并数据。然而，可以通过共享模型参数推断出敏感数据。我们提出使用联合协同训练方法，在其中客户端分享公共未标记数据集上的硬标签，而不是模型参数。对共享标签的一致性形成了未标记数据集的伪标签，客户端将其与私有数据结合使用来训练本地模型。我们表明，共享硬标签大大提高了与共享模型参数相比的隐私保护。同时，联合协同训练实现了与联邦学习相媲美的模型质量。此外，它使我们能够使用像(梯度提升)决策树、规则集合等本地模型

    arXiv:2310.05696v2 Announce Type: replace  Abstract: In many applications, sensitive data is inherently distributed and may not be pooled due to privacy concerns. Federated learning allows us to collaboratively train a model without pooling the data by iteratively aggregating the parameters of local models. It is possible, though, to infer upon the sensitive data from the shared model parameters. We propose to use a federated co-training approach where clients share hard labels on a public unlabeled dataset instead of model parameters. A consensus on the shared labels forms a pseudo labeling for the unlabeled dataset that clients use in combination with their private data to train local models. We show that sharing hard labels substantially improves privacy over sharing model parameters. At the same time, federated co-training achieves a model quality comparable to federated learning. Moreover, it allows us to use local models such as (gradient boosted) decision trees, rule ensembles, 
    
[^7]: 最新进展的命名实体识别综述

    A survey on recent advances in named entity recognition. (arXiv:2401.10825v1 [cs.CL])

    [http://arxiv.org/abs/2401.10825](http://arxiv.org/abs/2401.10825)

    这篇综述调查了最近的命名实体识别研究进展，并提供了对不同算法性能的深度比较，还探讨了数据集特征对方法行为的影响。

    

    命名实体识别旨在从文本中提取出命名真实世界对象的子字符串，并确定其类型（例如，是否指人物或组织）。在本综述中，我们首先概述了最近流行的方法，同时还关注了基于图和变换器的方法，包括很少在其他综述中涉及的大型语言模型（LLMs）。其次，我们重点介绍了针对稀缺注释数据集设计的方法。第三，我们评估了主要命名实体识别实现在各种具有不同特征（领域、规模和类别数）的数据集上的性能。因此，我们提供了一种从未同时考虑的算法的深度比较。我们的实验揭示了数据集特征如何影响我们比较的方法的行为。

    Named Entity Recognition seeks to extract substrings within a text that name real-world objects and to determine their type (for example, whether they refer to persons or organizations). In this survey, we first present an overview of recent popular approaches, but we also look at graph- and transformer- based methods including Large Language Models (LLMs) that have not had much coverage in other surveys. Second, we focus on methods designed for datasets with scarce annotations. Third, we evaluate the performance of the main NER implementations on a variety of datasets with differing characteristics (as regards their domain, their size, and their number of classes). We thus provide a deep comparison of algorithms that are never considered together. Our experiments shed some light on how the characteristics of datasets affect the behavior of the methods that we compare.
    
[^8]: 基于测量的变分量子计算用于生成建模

    Variational measurement-based quantum computation for generative modeling. (arXiv:2310.13524v1 [quant-ph])

    [http://arxiv.org/abs/2310.13524](http://arxiv.org/abs/2310.13524)

    这项研究提出了一种基于测量的变分量子计算算法，将量子测量的随机性视为计算资源，并应用于生成建模任务。

    

    基于测量的量子计算（MBQC）提供了一种基本独特的范例来设计量子算法。在MBQC中，由于量子测量的固有随机性，自然的操作不是确定性和幺正的，而是通过概率附带的。然而，到目前为止，MBQC的主要算法应用是完全抵消这种概率性质，以模拟表达在电路模型中的幺正计算。在这项工作中，我们提出了设计MBQC算法的思路，该算法接受这种固有随机性，并将MBQC中的随机附带视为计算资源。我们考虑了随机性有益的自然应用，即生成建模，这是一个以生成复杂概率分布为中心的机器学习任务。为了解决这个任务，我们提出了一个具有控制参数的变分MBQC算法，可以直接调整允许在计算中引入的随机程度。

    Measurement-based quantum computation (MBQC) offers a fundamentally unique paradigm to design quantum algorithms. Indeed, due to the inherent randomness of quantum measurements, the natural operations in MBQC are not deterministic and unitary, but are rather augmented with probabilistic byproducts. Yet, the main algorithmic use of MBQC so far has been to completely counteract this probabilistic nature in order to simulate unitary computations expressed in the circuit model. In this work, we propose designing MBQC algorithms that embrace this inherent randomness and treat the random byproducts in MBQC as a resource for computation. As a natural application where randomness can be beneficial, we consider generative modeling, a task in machine learning centered around generating complex probability distributions. To address this task, we propose a variational MBQC algorithm equipped with control parameters that allow to directly adjust the degree of randomness to be admitted in the comput
    
[^9]: 多保真神经网络计算的残差方法

    Residual Multi-Fidelity Neural Network Computing. (arXiv:2310.03572v1 [cs.LG])

    [http://arxiv.org/abs/2310.03572](http://arxiv.org/abs/2310.03572)

    本研究提出了一种残差多保真计算框架，通过使用多保真信息构建神经网络代理模型，解决了低保真和高保真计算模型之间的相关性建模问题。这种方法训练了两个神经网络，利用残差函数进行模型训练，最终得到了高保真替代模型。

    

    在本研究中，我们考虑使用多保真信息构建神经网络代理模型的一般问题。给定一个廉价的低保真和一个昂贵的高保真计算模型，我们提出了一个残差多保真计算框架，将模型之间的相关性建模为一个残差函数，这是一个可能非线性的1）模型共享的输入空间和低保真模型输出之间的映射，以及2）两个模型输出之间的差异。为了实现这一点，我们训练了两个神经网络来协同工作。第一个网络在少量的高保真和低保真数据上学习残差函数。一旦训练完成，这个网络被用来生成额外的合成高保真数据，用于训练第二个网络。一旦训练完成，第二个网络作为我们对高保真感兴趣的量的替代模型。我们提供了三个数值例子来证明这种方法的能力。

    In this work, we consider the general problem of constructing a neural network surrogate model using multi-fidelity information. Given an inexpensive low-fidelity and an expensive high-fidelity computational model, we present a residual multi-fidelity computational framework that formulates the correlation between models as a residual function, a possibly non-linear mapping between 1) the shared input space of the models together with the low-fidelity model output and 2) the discrepancy between the two model outputs. To accomplish this, we train two neural networks to work in concert. The first network learns the residual function on a small set of high-fidelity and low-fidelity data. Once trained, this network is used to generate additional synthetic high-fidelity data, which is used in the training of a second network. This second network, once trained, acts as our surrogate for the high-fidelity quantity of interest. We present three numerical examples to demonstrate the power of th
    
[^10]: 使用机器学习从语音信号中检测喉癌：可复现的文献综述

    Detecting Throat Cancer from Speech Signals Using Machine Learning: A Reproducible Literature Review. (arXiv:2307.09230v1 [cs.LG])

    [http://arxiv.org/abs/2307.09230](http://arxiv.org/abs/2307.09230)

    本研究对使用机器学习和人工智能从语音记录中检测喉癌的文献进行了综述，发现了22篇相关论文，讨论了它们的方法和结果。研究使用了神经网络和梅尔频率倒谱系数提取音频特征，并通过迁移学习实现了分类，取得了一定的准确率。

    

    本文对使用机器学习和人工智能从语音记录中检测喉癌的当前文献进行了范围评估。我们找到了22篇相关论文，并讨论了它们的方法和结果。我们将这些论文分为两组 - 九篇进行二分类，13篇进行多类别分类。这些论文提出了一系列方法，其中最常见的是使用神经网络。在分类之前还从音频中提取了许多特征，其中最常见的是梅尔频率倒谱系数。在这次搜索中未找到任何带有代码库的论文，因此无法复现。因此，我们创建了一个公开可用的代码库来训练自己的分类器。我们在一个多类别问题上使用迁移学习，将三种病理和健康对照进行分类。使用这种技术，我们取得了53.54%的加权平均召回率、83.14%的敏感性和特异性。

    In this work we perform a scoping review of the current literature on the detection of throat cancer from speech recordings using machine learning and artificial intelligence. We find 22 papers within this area and discuss their methods and results. We split these papers into two groups - nine performing binary classification, and 13 performing multi-class classification. The papers present a range of methods with neural networks being most commonly implemented. Many features are also extracted from the audio before classification, with the most common bring mel-frequency cepstral coefficients. None of the papers found in this search have associated code repositories and as such are not reproducible. Therefore, we create a publicly available code repository of our own classifiers. We use transfer learning on a multi-class problem, classifying three pathologies and healthy controls. Using this technique we achieve an unweighted average recall of 53.54%, sensitivity of 83.14%, and specif
    
[^11]: 学习ECG信号特征的非反向传播方法

    Learning ECG signal features without backpropagation. (arXiv:2307.01930v1 [cs.LG])

    [http://arxiv.org/abs/2307.01930](http://arxiv.org/abs/2307.01930)

    该论文提出了一种用于生成时间序列数据表示的新方法，依靠理论物理的思想以数据驱动的方式构建紧凑的表示。该方法能够捕捉数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性，并可以在广义设置中应用。

    

    表示学习已经成为机器学习领域的一个关键研究领域，它旨在发现用于提高分类和预测等下游任务的原始数据的有效特征的有效方法。在本文中，我们提出了一种用于生成时间序列类型数据表示的新方法。这种方法依靠理论物理的思想以数据驱动的方式构建紧凑的表示，并可以捕捉到数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性。这个新方法旨在识别能够有效捕捉属于特定类别的样本之间共享特征的线性规律。通过随后利用这些规律在前向方式下生成一个与分类器无关的表示，它们可以在广义设置中应用。我们展示了我们方法的有效性。

    Representation learning has become a crucial area of research in machine learning, as it aims to discover efficient ways of representing raw data with useful features to increase the effectiveness, scope and applicability of downstream tasks such as classification and prediction. In this paper, we propose a novel method to generate representations for time series-type data. This method relies on ideas from theoretical physics to construct a compact representation in a data-driven way, and it can capture both the underlying structure of the data and task-specific information while still remaining intuitive, interpretable and verifiable. This novel methodology aims to identify linear laws that can effectively capture a shared characteristic among samples belonging to a specific class. By subsequently utilizing these laws to generate a classifier-agnostic representation in a forward manner, they become applicable in a generalized setting. We demonstrate the effectiveness of our approach o
    
[^12]: 分层级别激活机制

    Layer-level activation mechanism. (arXiv:2306.04940v1 [cs.LG])

    [http://arxiv.org/abs/2306.04940](http://arxiv.org/abs/2306.04940)

    去噪声更好，表现更好的分层级别激活机制

    

    本文提出了一种新颖的激活机制，旨在建立分层级别激活功能（LayerAct）。这些功能旨在通过减少输入偏移所导致的激活输出的分层级波动来降低传统元素级激活功能的噪音鲁棒性。此外，LayerAct功能实现了类似于零的平均激活输出，而不限制激活输出空间。我们进行了分析和实验，证明LayerAct功能在噪声鲁棒性方面优于元素级激活功能，并且经验证明这些功能的平均激活结果类似于零。在三个基准图像分类任务的实验结果表明，在处理嘈杂的图像数据集时，LayerAct功能比元素级激活功能表现更好，而在大多数情况下，清洁数据集的表现也是优越的。

    In this work, we propose a novel activation mechanism aimed at establishing layer-level activation (LayerAct) functions. These functions are designed to be more noise-robust compared to traditional element-level activation functions by reducing the layer-level fluctuation of the activation outputs due to shift in inputs. Moreover, the LayerAct functions achieve a zero-like mean activation output without restricting the activation output space. We present an analysis and experiments demonstrating that LayerAct functions exhibit superior noise-robustness compared to element-level activation functions, and empirically show that these functions have a zero-like mean activation. Experimental results on three benchmark image classification tasks show that LayerAct functions excel in handling noisy image datasets, outperforming element-level activation functions, while the performance on clean datasets is also superior in most cases.
    
[^13]: 空间感知的共享汽车需求预测

    Spatially-Aware Car-Sharing Demand Prediction. (arXiv:2303.14421v1 [cs.LG])

    [http://arxiv.org/abs/2303.14421](http://arxiv.org/abs/2303.14421)

    本文提出了一种基于空间感知学习算法的方法来分析共享汽车站点的平均月度需求，利用了丰富的特征作为输入，可以提高预测性能。

    

    近年来，共享汽车服务作为私人个人出行的可行替代品出现，承诺更可持续、资源利用效率更高，但仍然等同于私人出行。关于短期预测和优化方法的研究已经改善了共享汽车服务的运营和车队控制;然而，在文献中长期预测和空间分析是缺乏的。我们建议使用具有空间感知学习算法的平均月度需求来分析基于站点的共享汽车服务，这种算法既具有高预测性能，又具有可解释性。具体而言，我们比较了全球随机森林模型与空间感知方法来预测平均每个站点的月度需求。该研究利用了丰富的社会-人口学、基于位置的(例如POI)和共享汽车特定特征作为输入，这些特征来自一个大型的专有共享汽车数据集和公开可用的数据集。我们展示了全球随机森林模型通常表现最好，但在某些情况下，空间感知方法可以提高预测性能。

    In recent years, car-sharing services have emerged as viable alternatives to private individual mobility, promising more sustainable and resource-efficient, but still comfortable transportation. Research on short-term prediction and optimization methods has improved operations and fleet control of car-sharing services; however, long-term projections and spatial analysis are sparse in the literature. We propose to analyze the average monthly demand in a station-based car-sharing service with spatially-aware learning algorithms that offer high predictive performance as well as interpretability. In particular, we compare the spatially-implicit Random Forest model with spatially-aware methods for predicting average monthly per-station demand. The study utilizes a rich set of socio-demographic, location-based (e.g., POIs), and car-sharing-specific features as input, extracted from a large proprietary car-sharing dataset and publicly available datasets. We show that the global Random Forest 
    

