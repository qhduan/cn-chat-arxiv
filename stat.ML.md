# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Replicability and stability in learning.](http://arxiv.org/abs/2304.03757) | 该论文研究了机器学习中的可复制性和全局稳定性，并证明许多学习任务只能弱化地实现全局稳定性。 |
| [^2] | [Representer Theorems for Metric and Preference Learning: A Geometric Perspective.](http://arxiv.org/abs/2304.03720) | 该论文提出了度量学习和偏好学习的新的表现定理，解决了度量学习任务以三元组比较为基础的表现定理问题。这种表现定理可以用内积诱导的范数来表示。 |
| [^3] | [Compressed Regression over Adaptive Networks.](http://arxiv.org/abs/2304.03638) | 本文阐述了一个分布式智能体的网络如何合作解决回归问题，在通信约束、自适应和合作的情况下能够达到的性能，并探讨了分布式回归问题的基本属性与最优分配通信资源之间的定量关系。 |
| [^4] | [Supervised Contrastive Learning with Heterogeneous Similarity for Distribution Shifts.](http://arxiv.org/abs/2304.03440) | 本文提出了一种带有异构相似性的新的监督对比学习方法，用于解决分布偏移问题，防止过拟合影响模型性能。 |
| [^5] | [Dynamics of Finite Width Kernel and Prediction Fluctuations in Mean Field Neural Networks.](http://arxiv.org/abs/2304.03408) | 本研究分析了宽但有限的特征学习神经网络中有限宽度效应的动力学，提供了对网络权重随机初始化下DMFT序参数波动的表征以及特征学习如何动态地减少最终NTK和最终网络预测的方差。 |
| [^6] | [Scalable Causal Discovery with Score Matching.](http://arxiv.org/abs/2304.03382) | 该论文提出了一种利用分数匹配算法实现可扩展因果推断的方法，该算法可从非线性可加性高斯噪声模型的对数似然函数中发现整个因果图，并通过实现与当前最先进技术相当的准确性来降低了计算门槛。 |
| [^7] | [On the Learnability of Multilabel Ranking.](http://arxiv.org/abs/2304.03337) | 研究了一系列排名损失函数下多标签排名问题在批处理和在线设置下的可学习性，并首次给出基于可学习性的排名损失函数的等价类。 |
| [^8] | [Heavy-Tailed Regularization of Weight Matrices in Deep Neural Networks.](http://arxiv.org/abs/2304.02911) | 本文介绍了一种名为重尾部正则化的技术，在深度神经网络中通过明确提倡更重的重尾谱来提高泛化性能。与标准正则化技术相比，该方法在基准数据集上实现了显着的改进。 |
| [^9] | [Revisiting the Fragility of Influence Functions.](http://arxiv.org/abs/2303.12922) | 本文研究了影响函数的脆弱性，并提出在非凸条件下使用深层模型和更复杂数据集来解决这一问题。 |
| [^10] | [Modality-Agnostic Variational Compression of Implicit Neural Representations.](http://arxiv.org/abs/2301.09479) | 提出了一种无模态偏见的隐式神经表示变分压缩算法，能够在不同的数据模态上表现出卓越的压缩性能和效果。 |
| [^11] | [Bayesian community detection for networks with covariates.](http://arxiv.org/abs/2203.02090) | 本论文提出了一种带有协变量的网络贝叶斯社区检测模型，该模型具有灵活性，可以建模参数估计的不确定性，包括社区成员身份的不确定性。 |
| [^12] | [Likelihood-Free Frequentist Inference: Confidence Sets with Correct Conditional Coverage.](http://arxiv.org/abs/2107.03920) | 本文提出了无似然假设下的频率学派推断（LF2I）框架，通过结合经典统计和现代机器学习，实现了构建具有正确条件覆盖的置信区间的实用程序和诊断方法，在包括宇宙学参数推断在内的多个例子中都实现了覆盖性质得到大幅改善。 |
| [^13] | [Simultaneous Reconstruction and Uncertainty Quantification for Tomography.](http://arxiv.org/abs/2103.15864) | 本文提出了一个通过高斯过程建模灵活且明确地将样品特征和实验噪声的先验知识纳入重建和量化不确定性的方法。这个方法在各种图像中展示了与现有实用重建方法相媲美的重建结果，并具备不确定性量化的独特能力。 |

# 详细

[^1]: 学习中的可复制性和稳定性

    Replicability and stability in learning. (arXiv:2304.03757v1 [cs.LG])

    [http://arxiv.org/abs/2304.03757](http://arxiv.org/abs/2304.03757)

    该论文研究了机器学习中的可复制性和全局稳定性，并证明许多学习任务只能弱化地实现全局稳定性。

    

    可复制性是科学中的关键，因为它使我们能够验证和验证研究结果。Impagliazzo、Lei、Pitassi和Sorrell（'22）最近开始研究机器学习中的可复制性。如果同一算法在两个独立同分布输入上使用相同的内部随机性时通常产生相同的输出，则学习算法是可复制的。我们研究了一种不涉及固定随机性的可复制性变体。如果一个算法在两个独立同分布的输入上（不固定内部随机性）应用时通常产生相同的输出，则算法满足这种形式的可复制性。这个变种被称为全局稳定性，并在差分隐私的上下文中由Bun、Livni和Moran（'20）介绍。 Impagliazzo等人展示了如何提高任何可复制算法的效果，以使其产生的输出概率无限接近于1。相反，我们证明了对于许多学习任务，只能弱化地实现全局稳定性，这里输出只有相同的部分。

    Replicability is essential in science as it allows us to validate and verify research findings. Impagliazzo, Lei, Pitassi and Sorrell (`22) recently initiated the study of replicability in machine learning. A learning algorithm is replicable if it typically produces the same output when applied on two i.i.d. inputs using the same internal randomness. We study a variant of replicability that does not involve fixing the randomness. An algorithm satisfies this form of replicability if it typically produces the same output when applied on two i.i.d. inputs (without fixing the internal randomness). This variant is called global stability and was introduced by Bun, Livni and Moran (`20) in the context of differential privacy.  Impagliazzo et al. showed how to boost any replicable algorithm so that it produces the same output with probability arbitrarily close to 1. In contrast, we demonstrate that for numerous learning tasks, global stability can only be accomplished weakly, where the same o
    
[^2]: 度量学习与偏好学习的表现定理：基于几何的视角

    Representer Theorems for Metric and Preference Learning: A Geometric Perspective. (arXiv:2304.03720v1 [cs.LG])

    [http://arxiv.org/abs/2304.03720](http://arxiv.org/abs/2304.03720)

    该论文提出了度量学习和偏好学习的新的表现定理，解决了度量学习任务以三元组比较为基础的表现定理问题。这种表现定理可以用内积诱导的范数来表示。

    

    我们探讨了希尔伯特空间中的度量学习和偏好学习问题，并获得了一种新的度量学习和偏好学习的表现定理。我们的关键观察是，表现定理可以根据问题结构内在的内积所诱导的范数来表示。此外，我们展示了如何将我们的框架应用于三元组比较的度量学习任务，并展示它导致了一个简单且自包含的该任务的表现定理。在再生核希尔伯特空间(RKHS)的情况下，我们展示了学习问题的解可以使用类似于经典表现定理的核术语表示。

    We explore the metric and preference learning problem in Hilbert spaces. We obtain a novel representer theorem for the simultaneous task of metric and preference learning. Our key observation is that the representer theorem can be formulated with respect to the norm induced by the inner product inherent in the problem structure. Additionally, we demonstrate how our framework can be applied to the task of metric learning from triplet comparisons and show that it leads to a simple and self-contained representer theorem for this task. In the case of Reproducing Kernel Hilbert Spaces (RKHS), we demonstrate that the solution to the learning problem can be expressed using kernel terms, akin to classical representer theorems.
    
[^3]: 压缩回归与自适应网络

    Compressed Regression over Adaptive Networks. (arXiv:2304.03638v1 [cs.LG])

    [http://arxiv.org/abs/2304.03638](http://arxiv.org/abs/2304.03638)

    本文阐述了一个分布式智能体的网络如何合作解决回归问题，在通信约束、自适应和合作的情况下能够达到的性能，并探讨了分布式回归问题的基本属性与最优分配通信资源之间的定量关系。

    

    本文中，我们推导了一个分布式智能体网络在解决回归问题时，在通信约束、自适应和合作的情况下能够达到的性能。智能体使用了最近提出的 ACTC (adapt-compress-then-combine) 扩散策略，在这个策略中，邻近智能体交换的信号被随机不同压缩算子编码。我们详细阐述了均方估计误差的特征，其中包括了一项与没有通信约束的情况下智能体将要达到的误差有关的错误项，以及一项由于压缩而产生的误差项。分析揭示了分布式回归问题的基本属性，尤其是通过Perron特征向量引起的梯度噪声和网络拓扑结构（）。我们展示了知晓这些关系对于最优地分配智能体之间的通信资源是至关重要的。

    In this work we derive the performance achievable by a network of distributed agents that solve, adaptively and in the presence of communication constraints, a regression problem. Agents employ the recently proposed ACTC (adapt-compress-then-combine) diffusion strategy, where the signals exchanged locally by neighboring agents are encoded with randomized differential compression operators. We provide a detailed characterization of the mean-square estimation error, which is shown to comprise a term related to the error that agents would achieve without communication constraints, plus a term arising from compression. The analysis reveals quantitative relationships between the compression loss and fundamental attributes of the distributed regression problem, in particular, the stochastic approximation error caused by the gradient noise and the network topology (through the Perron eigenvector). We show that knowledge of such relationships is critical to allocate optimally the communication
    
[^4]: 带有异构相似性的监督对比学习用于分布偏移问题

    Supervised Contrastive Learning with Heterogeneous Similarity for Distribution Shifts. (arXiv:2304.03440v1 [cs.LG])

    [http://arxiv.org/abs/2304.03440](http://arxiv.org/abs/2304.03440)

    本文提出了一种带有异构相似性的新的监督对比学习方法，用于解决分布偏移问题，防止过拟合影响模型性能。

    

    数据的分布在训练和测试时发生变化会导致分布偏移问题，进而严重影响模型在实际应用中的性能表现。近期研究表明，过拟合是其原因之一，合适的正则化可以缓解这种影响，尤其适用于使用神经网络等高度具有代表性的模型。本文提出了一种新的监督对比学习方法，通过该方法可以防止过拟合，训练模型避免在分布偏移下性能退化。作者将对比损失中的余弦相似性扩展为更通用的相似性度量，并建议在比较样本与正样本或负样本时使用不同的参数，在理论上这一建议被证明可以作为对比损失中的一种边缘效应。实验在模拟分布偏移的基准数据集上进行，包括子种群偏移和...（原文未完成）

    Distribution shifts are problems where the distribution of data changes between training and testing, which can significantly degrade the performance of a model deployed in the real world. Recent studies suggest that one reason for the degradation is a type of overfitting, and that proper regularization can mitigate the degradation, especially when using highly representative models such as neural networks. In this paper, we propose a new regularization using the supervised contrastive learning to prevent such overfitting and to train models that do not degrade their performance under the distribution shifts. We extend the cosine similarity in contrastive loss to a more general similarity measure and propose to use different parameters in the measure when comparing a sample to a positive or negative example, which is analytically shown to act as a kind of margin in contrastive loss. Experiments on benchmark datasets that emulate distribution shifts, including subpopulation shift and do
    
[^5]: 有限宽度核和平均场神经网络中的预测波动动力学分析

    Dynamics of Finite Width Kernel and Prediction Fluctuations in Mean Field Neural Networks. (arXiv:2304.03408v1 [stat.ML])

    [http://arxiv.org/abs/2304.03408](http://arxiv.org/abs/2304.03408)

    本研究分析了宽但有限的特征学习神经网络中有限宽度效应的动力学，提供了对网络权重随机初始化下DMFT序参数波动的表征以及特征学习如何动态地减少最终NTK和最终网络预测的方差。

    

    我们分析了宽但有限的特征学习神经网络中有限宽度效应的动力学。与许多先前的分析不同，我们的结果是针对特征学习强度的非微扰有限宽度的结果。从无限宽深度神经网络核和预测动力学的动力学平均场理论（DMFT）描述开始，我们提供了对网络权重的随机初始化下DMFT序参数$\mathcal{O}(1/\sqrt{\text{width}})$波动的表征。在网络训练的懒惰极限中，所有核都是随机的但在时间上静止的，预测方差具有通用形式。然而，在富有特征学习的区域，核和预测的波动是动态耦合且方差可以被自洽计算。在两层网络中，我们展示了特征学习如何动态地减少最终NTK和最终网络预测的方差。我们还展示了如何进行初始化。

    We analyze the dynamics of finite width effects in wide but finite feature learning neural networks. Unlike many prior analyses, our results, while perturbative in width, are non-perturbative in the strength of feature learning. Starting from a dynamical mean field theory (DMFT) description of infinite width deep neural network kernel and prediction dynamics, we provide a characterization of the $\mathcal{O}(1/\sqrt{\text{width}})$ fluctuations of the DMFT order parameters over random initialization of the network weights. In the lazy limit of network training, all kernels are random but static in time and the prediction variance has a universal form. However, in the rich, feature learning regime, the fluctuations of the kernels and predictions are dynamically coupled with variance that can be computed self-consistently. In two layer networks, we show how feature learning can dynamically reduce the variance of the final NTK and final network predictions. We also show how initialization
    
[^6]: 基于分数匹配的可扩展因果推断

    Scalable Causal Discovery with Score Matching. (arXiv:2304.03382v1 [cs.LG])

    [http://arxiv.org/abs/2304.03382](http://arxiv.org/abs/2304.03382)

    该论文提出了一种利用分数匹配算法实现可扩展因果推断的方法，该算法可从非线性可加性高斯噪声模型的对数似然函数中发现整个因果图，并通过实现与当前最先进技术相当的准确性来降低了计算门槛。

    

    本文展示了如何在非线性可加性高斯噪声模型中利用对数似然函数的二阶导数来发现整个因果图。借助于可扩展的机器学习方法来逼近分数函数 $\nabla \log p(\mathbf{X})$，我们扩展了Rolland等人（2022）的工作，后者仅从分数中恢复拓扑顺序，并需要一个昂贵的修剪步骤来消除由此顺序允许的虚假边缘。我们的分析导致了DAS（即 Discovery At Scale，规模化发现）算法，它通过与图形大小成比例的因素减少修剪的复杂性。在实践中，DAS实现了与当前最先进技术相当的准确性，同时速度提升了一个数量级以上。总的来说，我们的方法实现了原则性和可扩展的因果推断，大大降低了计算门槛。

    This paper demonstrates how to discover the whole causal graph from the second derivative of the log-likelihood in observational non-linear additive Gaussian noise models. Leveraging scalable machine learning approaches to approximate the score function $\nabla \log p(\mathbf{X})$, we extend the work of Rolland et al. (2022) that only recovers the topological order from the score and requires an expensive pruning step removing spurious edges among those admitted by the ordering. Our analysis leads to DAS (acronym for Discovery At Scale), a practical algorithm that reduces the complexity of the pruning by a factor proportional to the graph size. In practice, DAS achieves competitive accuracy with current state-of-the-art while being over an order of magnitude faster. Overall, our approach enables principled and scalable causal discovery, significantly lowering the compute bar.
    
[^7]: 关于多标签排名的可学习性研究

    On the Learnability of Multilabel Ranking. (arXiv:2304.03337v1 [cs.LG])

    [http://arxiv.org/abs/2304.03337](http://arxiv.org/abs/2304.03337)

    研究了一系列排名损失函数下多标签排名问题在批处理和在线设置下的可学习性，并首次给出基于可学习性的排名损失函数的等价类。

    

    在机器学习中，多标签排名是一项重要任务，广泛应用于网络搜索、新闻报道、推荐系统等领域。但是，关于多标签排名设置中可学习性的最基本问题仍未解答。本文研究了一系列排名损失函数下多标签排名问题在批处理和在线设置下的可学习性，同时也首次给出了基于可学习性的排名损失函数的等价类。

    Multilabel ranking is a central task in machine learning with widespread applications to web search, news stories, recommender systems, etc. However, the most fundamental question of learnability in a multilabel ranking setting remains unanswered. In this paper, we characterize the learnability of multilabel ranking problems in both the batch and online settings for a large family of ranking losses. Along the way, we also give the first equivalence class of ranking losses based on learnability.
    
[^8]: 深度神经网络的重尾部正则化

    Heavy-Tailed Regularization of Weight Matrices in Deep Neural Networks. (arXiv:2304.02911v1 [stat.ML])

    [http://arxiv.org/abs/2304.02911](http://arxiv.org/abs/2304.02911)

    本文介绍了一种名为重尾部正则化的技术，在深度神经网络中通过明确提倡更重的重尾谱来提高泛化性能。与标准正则化技术相比，该方法在基准数据集上实现了显着的改进。

    

    深度神经网络成功和显著的泛化能力背后的原因仍然是一个巨大的挑战。从随机矩阵理论得到的最新信息，特别是涉及深度神经网络中权重矩阵的谱分析的信息，为解决这个问题提供了有价值的线索。一个关键发现是，神经网络的泛化性能与其权重矩阵的谱的重尾程度相关。为了利用这一发现，我们介绍了一种新的正则化技术，称为重尾部正则化，通过正则化明确提倡权重矩阵中更重的重尾谱。首先，我们采用加权阿尔法和稳定秩作为惩罚项，两者都可微分，从而可以直接计算它们的梯度。为了避免过度正则化，我们介绍了两种惩罚函数的变体。然后，采用贝叶斯统计视角，我们提出了重尾部正则化的概率解释，使我们能够将其效果理解为权重矩阵的先验。在多个基准数据集上的实证评估表明，与标准正则化技术相比，我们的方法明显提高了泛化性能。

    Unraveling the reasons behind the remarkable success and exceptional generalization capabilities of deep neural networks presents a formidable challenge. Recent insights from random matrix theory, specifically those concerning the spectral analysis of weight matrices in deep neural networks, offer valuable clues to address this issue. A key finding indicates that the generalization performance of a neural network is associated with the degree of heavy tails in the spectrum of its weight matrices. To capitalize on this discovery, we introduce a novel regularization technique, termed Heavy-Tailed Regularization, which explicitly promotes a more heavy-tailed spectrum in the weight matrix through regularization. Firstly, we employ the Weighted Alpha and Stable Rank as penalty terms, both of which are differentiable, enabling the direct calculation of their gradients. To circumvent over-regularization, we introduce two variations of the penalty function. Then, adopting a Bayesian statistics
    
[^9]: 重新审视影响函数的脆弱性

    Revisiting the Fragility of Influence Functions. (arXiv:2303.12922v1 [cs.LG])

    [http://arxiv.org/abs/2303.12922](http://arxiv.org/abs/2303.12922)

    本文研究了影响函数的脆弱性，并提出在非凸条件下使用深层模型和更复杂数据集来解决这一问题。

    

    最近几年有很多论文致力于解释深度学习模型的预测。然而，很少有方法被提出来验证这些解释的准确性或可信度。最近，影响函数被证明是一种评估深度神经网络在单个样本上的灵敏度的方法。但是，先前的研究表明影响函数易受噪声和数据分布不对称性影响，缺乏鲁棒性。本文旨在研究影响函数的脆弱性，通过探究影响函数背后的机理，从而为增强影响函数的鲁棒性提供新思路。

    In the last few years, many works have tried to explain the predictions of deep learning models. Few methods, however, have been proposed to verify the accuracy or faithfulness of these explanations. Recently, influence functions, which is a method that approximates the effect that leave-one-out training has on the loss function, has been shown to be fragile. The proposed reason for their fragility remains unclear. Although previous work suggests the use of regularization to increase robustness, this does not hold in all cases. In this work, we seek to investigate the experiments performed in the prior work in an effort to understand the underlying mechanisms of influence function fragility. First, we verify influence functions using procedures from the literature under conditions where the convexity assumptions of influence functions are met. Then, we relax these assumptions and study the effects of non-convexity by using deeper models and more complex datasets. Here, we analyze the k
    
[^10]: 无模态偏见的隐式神经表示变分压缩算法

    Modality-Agnostic Variational Compression of Implicit Neural Representations. (arXiv:2301.09479v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.09479](http://arxiv.org/abs/2301.09479)

    提出了一种无模态偏见的隐式神经表示变分压缩算法，能够在不同的数据模态上表现出卓越的压缩性能和效果。

    

    我们提出了一种基于数据的函数视图，并用隐式神经表示（INR）参数化的无模态神经压缩算法。我们通过软门控机制将非线性映射到紧凑的潜在表示中，从而弥合了潜在编码和稀疏性之间的差距。这允许每个数据项通过子网络选择来定制共享的INR网络的专业化。在获取这种潜在表示的数据集后，我们在无模态空间中直接优化速率/失真的折衷方案，使用神经压缩。隐式神经表示的变分压缩（VC-INR）在具有相同表示容量的量化之前显示出改进的性能，同时优于其他INR技术所使用的先前量化方案。我们的实验结果显示，使用相同的算法而不需要任何特定于模态的归纳偏差，可以在各种不同的模态上取得卓越的结果。我们展示了在图像、气候数据、文本和音频数据上的结果。

    We introduce a modality-agnostic neural compression algorithm based on a functional view of data and parameterised as an Implicit Neural Representation (INR). Bridging the gap between latent coding and sparsity, we obtain compact latent representations non-linearly mapped to a soft gating mechanism. This allows the specialisation of a shared INR network to each data item through subnetwork selection. After obtaining a dataset of such latent representations, we directly optimise the rate/distortion trade-off in a modality-agnostic space using neural compression. Variational Compression of Implicit Neural Representations (VC-INR) shows improved performance given the same representational capacity pre quantisation while also outperforming previous quantisation schemes used for other INR techniques. Our experiments demonstrate strong results over a large set of diverse modalities using the same algorithm without any modality-specific inductive biases. We show results on images, climate dat
    
[^11]: 带有协变量的网络贝叶斯社区发现

    Bayesian community detection for networks with covariates. (arXiv:2203.02090v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2203.02090](http://arxiv.org/abs/2203.02090)

    本论文提出了一种带有协变量的网络贝叶斯社区检测模型，该模型具有灵活性，可以建模参数估计的不确定性，包括社区成员身份的不确定性。

    

    在各个领域中，网络数据的普遍存在和从中提取出有用信息的需求，促进了相关模型和算法的快速发展。在众多针对网络数据的学习任务中，社区发现，即发现节点群集或"社区"，可能是学术界最受关注的。在许多实际应用中，网络数据通常附带有节点或边缘协变量等附加信息，理想情况下应该利用这些信息进行推断。在本文中，我们通过提出一种具有协变量依赖随机分区先验的贝叶斯随机块模型，为带有协变量的网络社区检测的有限文献增添了新的内容。在我们的先验下，协变量明确地表现为指定聚类成员的先验分布。我们的模型具有建模所有参数估计，包括社区成员身份的不确定性的灵活性。

    The increasing prevalence of network data in a vast variety of fields and the need to extract useful information out of them have spurred fast developments in related models and algorithms. Among the various learning tasks with network data, community detection, the discovery of node clusters or "communities," has arguably received the most attention in the scientific community. In many real-world applications, the network data often come with additional information in the form of node or edge covariates that should ideally be leveraged for inference. In this paper, we add to a limited literature on community detection for networks with covariates by proposing a Bayesian stochastic block model with a covariate-dependent random partition prior. Under our prior, the covariates are explicitly expressed in specifying the prior distribution on the cluster membership. Our model has the flexibility of modeling uncertainties of all the parameter estimates including the community membership. Im
    
[^12]: 无似然假设下基于频率学派推断：具有正确条件覆盖的置信区间

    Likelihood-Free Frequentist Inference: Confidence Sets with Correct Conditional Coverage. (arXiv:2107.03920v6 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2107.03920](http://arxiv.org/abs/2107.03920)

    本文提出了无似然假设下的频率学派推断（LF2I）框架，通过结合经典统计和现代机器学习，实现了构建具有正确条件覆盖的置信区间的实用程序和诊断方法，在包括宇宙学参数推断在内的多个例子中都实现了覆盖性质得到大幅改善。

    

    许多科学领域都广泛使用计算机模拟器以隐含复杂系统的似然函数。传统的统计方法并不适用于这些称为无似然假设下推断（LFI）的情况，尤其是在渐近和低维的条件下。虽然新的机器学习方法，如归一化流，已经革新了LFI方法的样本效率和容量，但它们是否能为小样本大小产生具有正确条件覆盖的置信区间，仍然是一个开放问题。本文将经典统计和现代机器学习相结合，提出了（i）具有有限样本保证名义覆盖的内曼区间建设的实用程序，以及（ii）估计整个参数空间的条件覆盖的诊断。我们将我们的框架称为无似然假设下的频率学派推断（LF2I）。我们的框架可以使用定义测试统计量的任何方法，如似然比，因此具有广泛的适用性。我们将我们的方法应用于几个合成和实际的例子，包括宇宙学参数推断，并证明与现有的LFI方法相比，覆盖性质得到了大幅改善。

    Many areas of science make extensive use of computer simulators that implicitly encode likelihood functions of complex systems. Classical statistical methods are poorly suited for these so-called likelihood-free inference (LFI) settings, particularly outside asymptotic and low-dimensional regimes. Although new machine learning methods, such as normalizing flows, have revolutionized the sample efficiency and capacity of LFI methods, it remains an open question whether they produce confidence sets with correct conditional coverage for small sample sizes. This paper unifies classical statistics with modern machine learning to present (i) a practical procedure for the Neyman construction of confidence sets with finite-sample guarantees of nominal coverage, and (ii) diagnostics that estimate conditional coverage over the entire parameter space. We refer to our framework as likelihood-free frequentist inference (LF2I). Any method that defines a test statistic, like the likelihood ratio, can 
    
[^13]: 同时重建和不确定性量化的层析成像

    Simultaneous Reconstruction and Uncertainty Quantification for Tomography. (arXiv:2103.15864v2 [stat.AP] UPDATED)

    [http://arxiv.org/abs/2103.15864](http://arxiv.org/abs/2103.15864)

    本文提出了一个通过高斯过程建模灵活且明确地将样品特征和实验噪声的先验知识纳入重建和量化不确定性的方法。这个方法在各种图像中展示了与现有实用重建方法相媲美的重建结果，并具备不确定性量化的独特能力。

    

    层析成像在各种应用中具有革命性的影响，但由于有限和噪声测量导致没有唯一解，因此其重建具有病态性质。因此，在没有基准真实值的情况下，量化解的质量非常有必要但又未充分探索。本文通过高斯过程建模解决了这一挑战，通过选择卷积核和噪声类型灵活且明确地将样品特征和实验噪声的先验知识纳入模型中。我们提出的方法不仅可以得到与现有实用重建方法（例如，反问题的规则迭代求解器）相媲美的重建结果，而且可以有效地量化解的不确定性。我们展示了所提出方法在各种图像中的能力，并展示了其在存在各种噪声情况下进行不确定性量化的独特能力。

    Tomographic reconstruction, despite its revolutionary impact on a wide range of applications, suffers from its ill-posed nature in that there is no unique solution because of limited and noisy measurements. Therefore, in the absence of ground truth, quantifying the solution quality is highly desirable but under-explored. In this work, we address this challenge through Gaussian process modeling to flexibly and explicitly incorporate prior knowledge of sample features and experimental noises through the choices of the kernels and noise models. Our proposed method yields not only comparable reconstruction to existing practical reconstruction methods (e.g., regularized iterative solver for inverse problem) but also an efficient way of quantifying solution uncertainties. We demonstrate the capabilities of the proposed approach on various images and show its unique capability of uncertainty quantification in the presence of various noises.
    

