# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Review-Based Cross-Domain Recommendation via Hyperbolic Embedding and Hierarchy-Aware Domain Disentanglement](https://arxiv.org/abs/2403.20298) | 本文基于评论文本提出了一种双曲CDR方法，以应对推荐系统中的数据稀疏性挑战，避免传统基于距离的领域对齐技术可能引发的问题。 |
| [^2] | [Signed graphs in data sciences via communicability geometry](https://arxiv.org/abs/2403.07493) | 提出了符号图的可通信性几何概念，证明了其度量是欧几里德的和球形的，然后应用于解决符号图数据分析中的多个问题。 |
| [^3] | [Robustness Bounds on the Successful Adversarial Examples: Theory and Practice](https://arxiv.org/abs/2403.01896) | 本文提出了一个新的成功对抗样本概率上限的理论界限，取决于扰动范数、核函数以及训练数据集中最接近的不同标签对之间的距离，并且实验证明了该理论结果的有效性。 |
| [^4] | [Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A](https://arxiv.org/abs/2402.13213) | 多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。 |
| [^5] | [Mixed-Output Gaussian Process Latent Variable Models](https://arxiv.org/abs/2402.09122) | 本文提出了一种基于高斯过程潜变量模型的贝叶斯非参数方法，可以用于信号分离，并且能够处理包含纯组分信号加权和的情况，适用于光谱学和其他领域的多种应用。 |
| [^6] | [Tailoring Mixup to Data using Kernel Warping functions.](http://arxiv.org/abs/2311.01434) | 本研究提出了一种利用核扭曲函数对Mixup数据进行个性化处理的方法，通过动态改变插值系数的概率分布来实现更频繁和更强烈的混合相似数据点。实验证明这种方法不仅提高了模型性能，还提高了模型的校准性。 |
| [^7] | [Combat Urban Congestion via Collaboration: Heterogeneous GNN-based MARL for Coordinated Platooning and Traffic Signal Control.](http://arxiv.org/abs/2310.10948) | 本文提出了一种基于异构图多智能体强化学习和交通理论的创新解决方案，通过将车辆编队和交通信号控制作为不同的强化学习智能体，并结合图神经网络实现协调，以优化交通流量和缓解城市拥堵。 |
| [^8] | [Instant Complexity Reduction in CNNs using Locality-Sensitive Hashing.](http://arxiv.org/abs/2309.17211) | 该论文提出了一个名为HASTE的模块，通过使用局部敏感哈希技术，无需任何训练或精调即可实时降低卷积神经网络的计算成本，并且在压缩特征图时几乎不损失准确性。 |
| [^9] | [On the Need and Applicability of Causality for Fair Machine Learning.](http://arxiv.org/abs/2207.04053) | 本论文探讨了因果关系在公平机器学习中的必要性和适用性，强调了非因果预测的社会影响和法律反歧视过程依赖于因果主张。同时讨论了在实际场景中应用因果关系所面临的挑战和限制，并提出了可能的解决方案。 |

# 详细

[^1]: 基于双曲嵌入和层次感知域解耦的基于评论的跨领域推荐

    Review-Based Cross-Domain Recommendation via Hyperbolic Embedding and Hierarchy-Aware Domain Disentanglement

    [https://arxiv.org/abs/2403.20298](https://arxiv.org/abs/2403.20298)

    本文基于评论文本提出了一种双曲CDR方法，以应对推荐系统中的数据稀疏性挑战，避免传统基于距离的领域对齐技术可能引发的问题。

    

    数据稀疏性问题对推荐系统构成了重要挑战。本文提出了一种基于评论文本的算法，以应对这一问题。此外，跨领域推荐（CDR）吸引了广泛关注，它捕捉可在领域间共享的知识，并将其从更丰富的领域（源领域）转移到更稀疏的领域（目标领域）。然而，现有大多数方法假设欧几里德嵌入空间，在准确表示更丰富的文本信息和处理用户和物品之间的复杂交互方面遇到困难。本文倡导一种基于评论文本的双曲CDR方法来建模用户-物品关系。首先强调了传统的基于距离的领域对齐技术可能会导致问题，因为在双曲几何中对小修改造成的干扰会被放大，最终导致层次性崩溃。

    arXiv:2403.20298v1 Announce Type: cross  Abstract: The issue of data sparsity poses a significant challenge to recommender systems. In response to this, algorithms that leverage side information such as review texts have been proposed. Furthermore, Cross-Domain Recommendation (CDR), which captures domain-shareable knowledge and transfers it from a richer domain (source) to a sparser one (target), has received notable attention. Nevertheless, the majority of existing methodologies assume a Euclidean embedding space, encountering difficulties in accurately representing richer text information and managing complex interactions between users and items. This paper advocates a hyperbolic CDR approach based on review texts for modeling user-item relationships. We first emphasize that conventional distance-based domain alignment techniques may cause problems because small modifications in hyperbolic geometry result in magnified perturbations, ultimately leading to the collapse of hierarchical 
    
[^2]: 通过可通信性几何学在数据科学中的符号图

    Signed graphs in data sciences via communicability geometry

    [https://arxiv.org/abs/2403.07493](https://arxiv.org/abs/2403.07493)

    提出了符号图的可通信性几何概念，证明了其度量是欧几里德的和球形的，然后应用于解决符号图数据分析中的多个问题。

    

    符号图是表示多种存在冲突交互的数据的新兴方式，包括来自生物学、生态学和社会系统的数据。我们在这里提出了符号图的可通信性几何概念，证明了在这个空间中的度量，比如可通信性距离和角度，是欧几里德的和球形的。然后我们将这些度量应用于以统一方式解决符号图数据分析中的几个问题，包括符号图的分区、维度约简、找到符号网络中的联盟等级以及量化系统中现有派系之间极化程度的问题。

    arXiv:2403.07493v1 Announce Type: cross  Abstract: Signed graphs are an emergent way of representing data in a variety of contexts were conflicting interactions exist. These include data from biological, ecological, and social systems. Here we propose the concept of communicability geometry for signed graphs, proving that metrics in this space, such as the communicability distance and angles, are Euclidean and spherical. We then apply these metrics to solve several problems in data analysis of signed graphs in a unified way. They include the partitioning of signed graphs, dimensionality reduction, finding hierarchies of alliances in signed networks as well as the quantification of the degree of polarization between the existing factions in systems represented by this type of graphs.
    
[^3]: 成功对抗样本的强鲁棒性界限：理论与实践

    Robustness Bounds on the Successful Adversarial Examples: Theory and Practice

    [https://arxiv.org/abs/2403.01896](https://arxiv.org/abs/2403.01896)

    本文提出了一个新的成功对抗样本概率上限的理论界限，取决于扰动范数、核函数以及训练数据集中最接近的不同标签对之间的距离，并且实验证明了该理论结果的有效性。

    

    对抗样本（AE）是一种针对机器学习的攻击方法，通过对数据添加不可感知的扰动来诱使错分。本文基于高斯过程（GP）分类，研究了成功AE的概率上限。我们证明了一个新的上界，取决于AE的扰动范数、GP中使用的核函数以及训练数据集中具有不同标签的最接近对之间的距离。令人惊讶的是，该上限不受样本数据集分布的影响。我们通过使用ImageNet的实验验证了我们的理论结果。此外，我们展示了改变核函数参数会导致成功AE概率上限的变化。

    arXiv:2403.01896v1 Announce Type: new  Abstract: Adversarial example (AE) is an attack method for machine learning, which is crafted by adding imperceptible perturbation to the data inducing misclassification. In the current paper, we investigated the upper bound of the probability of successful AEs based on the Gaussian Process (GP) classification. We proved a new upper bound that depends on AE's perturbation norm, the kernel function used in GP, and the distance of the closest pair with different labels in the training dataset. Surprisingly, the upper bound is determined regardless of the distribution of the sample dataset. We showed that our theoretical result was confirmed through the experiment using ImageNet. In addition, we showed that changing the parameters of the kernel function induces a change of the upper bound of the probability of successful AEs.
    
[^4]: 软最大概率（大部分时候）在多项选择问答任务中预测大型语言模型的正确性

    Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A

    [https://arxiv.org/abs/2402.13213](https://arxiv.org/abs/2402.13213)

    多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。

    

    尽管大型语言模型（LLMs）在许多任务上表现出色，但过度自信仍然是一个问题。我们假设在多项选择问答任务中，错误答案将与最大softmax概率（MSPs）较小相关，相比之下正确答案较大。我们在十个开源LLMs和五个数据集上全面评估了这一假设，在表现良好的原始问答任务中发现了对我们假设的强有力证据。对于表现最佳的六个LLMs，从MSP导出的AUROC在59/60个实例中都优于随机机会，p < 10^{-4}。在这六个LLMs中，平均AUROC范围在60%至69%之间。利用这些发现，我们提出了一个带有弃权选项的多项选择问答任务，并展示通过根据初始模型响应的MSP有选择地弃权可以提高性能。我们还用预softmax logits而不是softmax进行了相同的实验。

    arXiv:2402.13213v1 Announce Type: cross  Abstract: Although large language models (LLMs) perform impressively on many tasks, overconfidence remains a problem. We hypothesized that on multiple-choice Q&A tasks, wrong answers would be associated with smaller maximum softmax probabilities (MSPs) compared to correct answers. We comprehensively evaluate this hypothesis on ten open-source LLMs and five datasets, and find strong evidence for our hypothesis among models which perform well on the original Q&A task. For the six LLMs with the best Q&A performance, the AUROC derived from the MSP was better than random chance with p < 10^{-4} in 59/60 instances. Among those six LLMs, the average AUROC ranged from 60% to 69%. Leveraging these findings, we propose a multiple-choice Q&A task with an option to abstain and show that performance can be improved by selectively abstaining based on the MSP of the initial model response. We also run the same experiments with pre-softmax logits instead of sof
    
[^5]: 混合输出高斯过程潜变量模型

    Mixed-Output Gaussian Process Latent Variable Models

    [https://arxiv.org/abs/2402.09122](https://arxiv.org/abs/2402.09122)

    本文提出了一种基于高斯过程潜变量模型的贝叶斯非参数方法，可以用于信号分离，并且能够处理包含纯组分信号加权和的情况，适用于光谱学和其他领域的多种应用。

    

    本文提出了一种贝叶斯非参数的信号分离方法，其中信号可以根据潜变量变化。我们的主要贡献是增加了高斯过程潜变量模型（GPLVMs），以包括每个数据点由已知数量的纯组分信号的加权和组成的情况，并观察多个输入位置。我们的框架允许使用各种关于每个观测权重的先验。这种灵活性使我们能够表示包括用于估计分数组成的总和为一约束和用于分类的二进制权重的用例。我们的贡献对于光谱学尤其相关，因为改变条件可能导致基础纯组分信号在样本之间变化。为了展示对光谱学和其他领域的适用性，我们考虑了几个应用：一个具有不同温度的近红外光谱数据集。

    arXiv:2402.09122v1 Announce Type: cross Abstract: This work develops a Bayesian non-parametric approach to signal separation where the signals may vary according to latent variables. Our key contribution is to augment Gaussian Process Latent Variable Models (GPLVMs) to incorporate the case where each data point comprises the weighted sum of a known number of pure component signals, observed across several input locations. Our framework allows the use of a range of priors for the weights of each observation. This flexibility enables us to represent use cases including sum-to-one constraints for estimating fractional makeup, and binary weights for classification. Our contributions are particularly relevant to spectroscopy, where changing conditions may cause the underlying pure component signals to vary from sample to sample. To demonstrate the applicability to both spectroscopy and other domains, we consider several applications: a near-infrared spectroscopy data set with varying temper
    
[^6]: 通过核扭曲函数定制Mixup数据

    Tailoring Mixup to Data using Kernel Warping functions. (arXiv:2311.01434v1 [cs.LG])

    [http://arxiv.org/abs/2311.01434](http://arxiv.org/abs/2311.01434)

    本研究提出了一种利用核扭曲函数对Mixup数据进行个性化处理的方法，通过动态改变插值系数的概率分布来实现更频繁和更强烈的混合相似数据点。实验证明这种方法不仅提高了模型性能，还提高了模型的校准性。

    

    数据增强是学习高效深度学习模型的重要基础。在所有提出的增强技术中，线性插值训练数据点（也称为Mixup）已被证明在许多应用中非常有效。然而，大多数研究都集中在选择合适的点进行混合，或者应用复杂的非线性插值，而我们则对更相似的点进行更频繁和更强烈的混合感兴趣。为此，我们提出了通过扭曲函数动态改变插值系数的概率分布的方法，取决于要组合的数据点之间的相似性。我们定义了一个高效而灵活的框架来实现这一点，以避免多样性的损失。我们进行了广泛的分类和回归任务实验，结果显示我们提出的方法既提高了模型的性能，又提高了模型的校准性。代码可在https://github.com/ENSTA-U2IS/torch-uncertainty上找到。

    Data augmentation is an essential building block for learning efficient deep learning models. Among all augmentation techniques proposed so far, linear interpolation of training data points, also called mixup, has found to be effective for a large panel of applications. While the majority of works have focused on selecting the right points to mix, or applying complex non-linear interpolation, we are interested in mixing similar points more frequently and strongly than less similar ones. To this end, we propose to dynamically change the underlying distribution of interpolation coefficients through warping functions, depending on the similarity between data points to combine. We define an efficient and flexible framework to do so without losing in diversity. We provide extensive experiments for classification and regression tasks, showing that our proposed method improves both performance and calibration of models. Code available in https://github.com/ENSTA-U2IS/torch-uncertainty
    
[^7]: 通过协作解决城市拥堵：基于异构GNN的协调编队和交通信号控制的多智能体强化学习方法

    Combat Urban Congestion via Collaboration: Heterogeneous GNN-based MARL for Coordinated Platooning and Traffic Signal Control. (arXiv:2310.10948v1 [cs.LG])

    [http://arxiv.org/abs/2310.10948](http://arxiv.org/abs/2310.10948)

    本文提出了一种基于异构图多智能体强化学习和交通理论的创新解决方案，通过将车辆编队和交通信号控制作为不同的强化学习智能体，并结合图神经网络实现协调，以优化交通流量和缓解城市拥堵。

    

    多年来，强化学习已经成为一种流行的方法，用于独立或分层方式开发信号控制和车辆编队策略。然而，在实时中联合控制这两者以减轻交通拥堵带来了新的挑战，如信号控制和编队之间固有的物理和行为异质性，以及它们之间的协调。本文提出了一种创新的解决方案来应对这些挑战，基于异构图多智能体强化学习和交通理论。我们的方法包括：1）将编队和信号控制设计为不同的强化学习智能体，具有自己的观测、动作和奖励函数，以优化交通流量；2）通过在多智能体强化学习中引入图神经网络来设计协调，以促进区域范围内智能体之间的无缝信息交换。我们通过SUMO模拟环境评估了我们的方法。

    Over the years, reinforcement learning has emerged as a popular approach to develop signal control and vehicle platooning strategies either independently or in a hierarchical way. However, jointly controlling both in real-time to alleviate traffic congestion presents new challenges, such as the inherent physical and behavioral heterogeneity between signal control and platooning, as well as coordination between them. This paper proposes an innovative solution to tackle these challenges based on heterogeneous graph multi-agent reinforcement learning and traffic theories. Our approach involves: 1) designing platoon and signal control as distinct reinforcement learning agents with their own set of observations, actions, and reward functions to optimize traffic flow; 2) designing coordination by incorporating graph neural networks within multi-agent reinforcement learning to facilitate seamless information exchange among agents on a regional scale. We evaluate our approach through SUMO simu
    
[^8]: 使用局部敏感哈希在CNN中实现即时复杂度降低

    Instant Complexity Reduction in CNNs using Locality-Sensitive Hashing. (arXiv:2309.17211v1 [cs.CV])

    [http://arxiv.org/abs/2309.17211](http://arxiv.org/abs/2309.17211)

    该论文提出了一个名为HASTE的模块，通过使用局部敏感哈希技术，无需任何训练或精调即可实时降低卷积神经网络的计算成本，并且在压缩特征图时几乎不损失准确性。

    

    为了在资源受限的设备上降低卷积神经网络（CNN）的计算成本，结构化剪枝方法已显示出有希望的结果，在不太大程度降低准确性的情况下大大减少了浮点运算（FLOPs）。然而，大多数最新的方法要求进行精调或特定的训练过程，以实现在保留准确性和降低FLOPs之间合理折衷。这引入了计算开销的额外成本，并需要可用的训练数据。为此，我们提出了HASTE（Hashing for Tractable Efficiency），它是一个无需参数和无需数据的模块，可以作为任何常规卷积模块的即插即用替代品。它能够在不需要任何训练或精调的情况下即时降低网络的测试推理成本。通过使用局部敏感哈希（LSH）来检测特征图中的冗余，我们能够大幅压缩潜在特征图而几乎不损失准确性。

    To reduce the computational cost of convolutional neural networks (CNNs) for usage on resource-constrained devices, structured pruning approaches have shown promising results, drastically reducing floating-point operations (FLOPs) without substantial drops in accuracy. However, most recent methods require fine-tuning or specific training procedures to achieve a reasonable trade-off between retained accuracy and reduction in FLOPs. This introduces additional cost in the form of computational overhead and requires training data to be available. To this end, we propose HASTE (Hashing for Tractable Efficiency), a parameter-free and data-free module that acts as a plug-and-play replacement for any regular convolution module. It instantly reduces the network's test-time inference cost without requiring any training or fine-tuning. We are able to drastically compress latent feature maps without sacrificing much accuracy by using locality-sensitive hashing (LSH) to detect redundancies in the c
    
[^9]: 论公平机器学习中因果关系的必要性和适用性

    On the Need and Applicability of Causality for Fair Machine Learning. (arXiv:2207.04053v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.04053](http://arxiv.org/abs/2207.04053)

    本论文探讨了因果关系在公平机器学习中的必要性和适用性，强调了非因果预测的社会影响和法律反歧视过程依赖于因果主张。同时讨论了在实际场景中应用因果关系所面临的挑战和限制，并提出了可能的解决方案。

    

    除了在流行病学、政治和社会科学中的常见应用案例外，事实证明因果关系在评估自动决策的公正性方面十分重要，无论是在法律上还是日常生活中。我们提供了关于为何因果关系对公平性评估尤为重要的论点和示例。特别是，我们指出了非因果预测的社会影响以及依赖因果主张的法律反歧视过程。我们最后讨论了应用因果关系在实际场景中的挑战和局限性，以及可能的解决方案。

    Besides its common use cases in epidemiology, political, and social sciences, causality turns out to be crucial in evaluating the fairness of automated decisions, both in a legal and everyday sense. We provide arguments and examples, of why causality is particularly important for fairness evaluation. In particular, we point out the social impact of non-causal predictions and the legal anti-discrimination process that relies on causal claims. We conclude with a discussion about the challenges and limitations of applying causality in practical scenarios as well as possible solutions.
    

