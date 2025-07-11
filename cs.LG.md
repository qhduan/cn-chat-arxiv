# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unifews: Unified Entry-Wise Sparsification for Efficient Graph Neural Network](https://arxiv.org/abs/2403.13268) | Unifews通过统一逐条稀疏化的方式，联合边权重稀疏化以提高学习效率，适用于不同架构设计并具有逐渐增加稀疏度的自适应压缩。 |
| [^2] | [OVOR: OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning](https://arxiv.org/abs/2402.04129) | 这项研究提出了一种新的正则化方法，利用虚拟异常值来改善无需回顾的类增量学习过程中不同任务间的类别混淆问题，并且消除了额外的提示查询和组合计算开销。 |
| [^3] | [Don't Push the Button! Exploring Data Leakage Risks in Machine Learning and Transfer Learning.](http://arxiv.org/abs/2401.13796) | 本文讨论了机器学习中的数据泄露问题，即未预期的信息污染训练数据，影响模型性能评估，用户可能由于缺乏理解而忽视关键步骤，导致乐观的性能估计在实际场景中不成立。 |
| [^4] | [Spectral Estimators for Structured Generalized Linear Models via Approximate Message Passing.](http://arxiv.org/abs/2308.14507) | 本论文研究了针对广义线性模型的参数估计问题，提出了一种通过谱估计器进行预处理的方法。通过对测量进行特征协方差矩阵Σ表示，分析了谱估计器在结构化设计中的性能，并确定了最优预处理以最小化样本数量。 |
| [^5] | [Adversarial Defenses via Vector Quantization.](http://arxiv.org/abs/2305.13651) | 该论文提出了两种基于矢量量化的新对抗性防御方法，能够在高维空间中提供理论保证和实验上的表现优势。 |
| [^6] | [Implicit Counterfactual Data Augmentation for Deep Neural Networks.](http://arxiv.org/abs/2304.13431) | 本研究提出了隐式反事实数据增强（ICDA）方法，通过新的样本增强策略、易于计算的代理损失和具体方案，消除了虚假关联并进行了稳健预测。 |
| [^7] | [Interpretable Anomaly Detection via Discrete Optimization.](http://arxiv.org/abs/2303.14111) | 该论文提出了一个通过学习有限自动机进行异常检测的框架，并通过约束优化算法和新的正则化方案提高了可解释性。 |
| [^8] | [Don't Get Me Wrong: How to Apply Deep Visual Interpretations to Time Series.](http://arxiv.org/abs/2203.07861) | 该论文提出了一个针对时间序列分类和分割任务的框架，通过六个度量来评估基于梯度、传播或干扰的事后可视化解释方法。实验结果表明，这些方法对于时间序列的解释具有较高的可信度和有效性。 |

# 详细

[^1]: Unifews：用于高效图神经网络的统一逐条稀疏化

    Unifews: Unified Entry-Wise Sparsification for Efficient Graph Neural Network

    [https://arxiv.org/abs/2403.13268](https://arxiv.org/abs/2403.13268)

    Unifews通过统一逐条稀疏化的方式，联合边权重稀疏化以提高学习效率，适用于不同架构设计并具有逐渐增加稀疏度的自适应压缩。

    

    图神经网络（GNNs）在各种图学习任务中表现出了有希望的性能，但代价是资源密集型的计算。GNN更新的主要开销来自图传播和权重变换，两者都涉及对图规模矩阵的操作。先前的研究尝试通过利用图级别或网络级别的稀疏化技术来减少计算预算，从而产生缩小的图或权重。在这项工作中，我们提出了Unifews，它以逐个矩阵元素的方式统一了这两种操作，并进行联合边权重稀疏化以增强学习效率。Unifews的逐条设计使其能够在GNN层之间进行自适应压缩，稀疏度逐渐增加，并适用于各种架构设计，具有即时操作简化。在理论上，我们建立了一个新颖的框架来表征稀疏

    arXiv:2403.13268v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) have shown promising performance in various graph learning tasks, but at the cost of resource-intensive computations. The primary overhead of GNN update stems from graph propagation and weight transformation, both involving operations on graph-scale matrices. Previous studies attempt to reduce the computational budget by leveraging graph-level or network-level sparsification techniques, resulting in downsized graph or weights. In this work, we propose Unifews, which unifies the two operations in an entry-wise manner considering individual matrix elements, and conducts joint edge-weight sparsification to enhance learning efficiency. The entry-wise design of Unifews enables adaptive compression across GNN layers with progressively increased sparsity, and is applicable to a variety of architectural designs with on-the-fly operation simplification. Theoretically, we establish a novel framework to characterize spa
    
[^2]: OVOR：一种使用虚拟异常值正则化的OnePrompt方法，实现无需回顾的类增量学习

    OVOR: OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning

    [https://arxiv.org/abs/2402.04129](https://arxiv.org/abs/2402.04129)

    这项研究提出了一种新的正则化方法，利用虚拟异常值来改善无需回顾的类增量学习过程中不同任务间的类别混淆问题，并且消除了额外的提示查询和组合计算开销。

    

    最近的研究表明，利用大规模预训练模型和可学习的提示，在无需回顾的类增量学习（CIL）设置中可以实现比著名的基于回顾的方法更好的性能。无需回顾的CIL方法在区分不同任务的类别时遇到困难，因为它们并未一同训练。在这项研究中，我们提出了一种基于虚拟异常值的正则化方法，通过紧缩分类器的决策边界，减轻不同任务间类别的混淆。最近的基于提示的方法通常需要一个存储各任务特定提示的集合，以防止新任务的知识覆盖先前任务的知识，从而导致额外的查询和组合适当提示的计算开销。我们在论文中揭示，可以消除这种额外开销而不牺牲准确性。我们演示了简化的基于提示的方法可以达到与先前最新状态-of-the-art方法相当的结果。

    Recent works have shown that by using large pre-trained models along with learnable prompts, rehearsal-free methods for class-incremental learning (CIL) settings can achieve superior performance to prominent rehearsal-based ones. Rehearsal-free CIL methods struggle with distinguishing classes from different tasks, as those are not trained together. In this work we propose a regularization method based on virtual outliers to tighten decision boundaries of the classifier, such that confusion of classes among different tasks is mitigated. Recent prompt-based methods often require a pool of task-specific prompts, in order to prevent overwriting knowledge of previous tasks with that of the new task, leading to extra computation in querying and composing an appropriate prompt from the pool. This additional cost can be eliminated, without sacrificing accuracy, as we reveal in the paper. We illustrate that a simplified prompt-based method can achieve results comparable to previous state-of-the
    
[^3]: 不要按按钮！探索机器学习和迁移学习中的数据泄露风险

    Don't Push the Button! Exploring Data Leakage Risks in Machine Learning and Transfer Learning. (arXiv:2401.13796v1 [cs.LG])

    [http://arxiv.org/abs/2401.13796](http://arxiv.org/abs/2401.13796)

    本文讨论了机器学习中的数据泄露问题，即未预期的信息污染训练数据，影响模型性能评估，用户可能由于缺乏理解而忽视关键步骤，导致乐观的性能估计在实际场景中不成立。

    

    机器学习（ML）在各个领域取得了革命性的进展，为多个领域提供了预测能力。然而，随着ML工具的日益可获得性，许多从业者缺乏深入的ML专业知识，采用了“按按钮”方法，利用用户友好的界面而忽视了底层算法的深入理解。虽然这种方法提供了便利，但它引发了对结果可靠性的担忧，导致了错误的性能评估等挑战。本文解决了ML中的一个关键问题，即数据泄露，其中未预期的信息污染了训练数据，影响了模型的性能评估。由于缺乏理解，用户可能会无意中忽视关键步骤，从而导致在现实场景中可能不成立的乐观性能估计。评估性能与实际在新数据上的性能的差异是一个重要的关注点。本文特别将ML中的数据泄露分为不同类别，并讨论了相关解决方法。

    Machine Learning (ML) has revolutionized various domains, offering predictive capabilities in several areas. However, with the increasing accessibility of ML tools, many practitioners, lacking deep ML expertise, adopt a "push the button" approach, utilizing user-friendly interfaces without a thorough understanding of underlying algorithms. While this approach provides convenience, it raises concerns about the reliability of outcomes, leading to challenges such as incorrect performance evaluation. This paper addresses a critical issue in ML, known as data leakage, where unintended information contaminates the training data, impacting model performance evaluation. Users, due to a lack of understanding, may inadvertently overlook crucial steps, leading to optimistic performance estimates that may not hold in real-world scenarios. The discrepancy between evaluated and actual performance on new data is a significant concern. In particular, this paper categorizes data leakage in ML, discussi
    
[^4]: 通过近似传递消息实现结构化广义线性模型的谱估计器

    Spectral Estimators for Structured Generalized Linear Models via Approximate Message Passing. (arXiv:2308.14507v1 [math.ST])

    [http://arxiv.org/abs/2308.14507](http://arxiv.org/abs/2308.14507)

    本论文研究了针对广义线性模型的参数估计问题，提出了一种通过谱估计器进行预处理的方法。通过对测量进行特征协方差矩阵Σ表示，分析了谱估计器在结构化设计中的性能，并确定了最优预处理以最小化样本数量。

    

    我们考虑从广义线性模型中的观测中进行参数估计的问题。谱方法是一种简单而有效的估计方法：它通过对观测进行适当预处理得到的矩阵的主特征向量来估计参数。尽管谱估计器被广泛使用，但对于结构化（即独立同分布的高斯和哈尔）设计，目前仅有对谱估计器的严格性能表征以及对数据进行预处理的基本方法可用。相反，实际的设计矩阵具有高度结构化并且表现出非平凡的相关性。为解决这个问题，我们考虑了捕捉测量的非各向同性特性的相关高斯设计，通过特征协方差矩阵Σ进行表示。我们的主要结果是对于这种情况下谱估计器性能的精确渐近分析。然后，可以通过这一结果来确定最优预处理，从而最小化所需样本的数量。

    We consider the problem of parameter estimation from observations given by a generalized linear model. Spectral methods are a simple yet effective approach for estimation: they estimate the parameter via the principal eigenvector of a matrix obtained by suitably preprocessing the observations. Despite their wide use, a rigorous performance characterization of spectral estimators, as well as a principled way to preprocess the data, is available only for unstructured (i.e., i.i.d. Gaussian and Haar) designs. In contrast, real-world design matrices are highly structured and exhibit non-trivial correlations. To address this problem, we consider correlated Gaussian designs which capture the anisotropic nature of the measurements via a feature covariance matrix $\Sigma$. Our main result is a precise asymptotic characterization of the performance of spectral estimators in this setting. This then allows to identify the optimal preprocessing that minimizes the number of samples needed to meanin
    
[^5]: 基于矢量量化的对抗防御

    Adversarial Defenses via Vector Quantization. (arXiv:2305.13651v1 [cs.LG])

    [http://arxiv.org/abs/2305.13651](http://arxiv.org/abs/2305.13651)

    该论文提出了两种基于矢量量化的新对抗性防御方法，能够在高维空间中提供理论保证和实验上的表现优势。

    

    在随机离散化的基础上，我们在高维空间中利用矢量量化开发了两种新的对抗性防御方法，分别称为pRD和swRD。这些方法不仅在证明准确度方面提供了理论保证，而且通过大量实验表明，它们的表现与当前对抗防御技术相当甚至更优秀。这些方法可以扩展到一种版本，允许对目标分类器进行进一步训练，并展示出进一步改进的性能。

    Building upon Randomized Discretization, we develop two novel adversarial defenses against white-box PGD attacks, utilizing vector quantization in higher dimensional spaces. These methods, termed pRD and swRD, not only offer a theoretical guarantee in terms of certified accuracy, they are also shown, via abundant experiments, to perform comparably or even superior to the current art of adversarial defenses. These methods can be extended to a version that allows further training of the target classifier and demonstrates further improved performance.
    
[^6]: 深度神经网络的隐式反事实数据增强

    Implicit Counterfactual Data Augmentation for Deep Neural Networks. (arXiv:2304.13431v1 [cs.LG])

    [http://arxiv.org/abs/2304.13431](http://arxiv.org/abs/2304.13431)

    本研究提出了隐式反事实数据增强（ICDA）方法，通过新的样本增强策略、易于计算的代理损失和具体方案，消除了虚假关联并进行了稳健预测。

    

    机器学习模型易于捕捉非因果属性和类别之间的虚假相关性，使用反事实数据增强是破除这些虚假的联想的有效方法。然而，明确生成反事实数据很具挑战性，训练效率会降低。因此，本研究提出了一种隐式反事实数据增强（Implicit Counterfactual Data Augmentation，ICDA）方法来消除虚假关联并进行稳健预测。具体而言，首先，开发了一种新的样本增强策略，为每个样本生成在语义和反事实意义上有意义的深度特征，并具有不同的增强强度。其次，当增广样本数变为无穷大时，我们推导出对于增广特征集的易于计算的代理损失。第三，提出了两种具体的方案，包括直接量化和元学习，以确定鲁棒性损失的关键参数。此外，还从实验的角度解释了ICDA的作用。

    Machine-learning models are prone to capturing the spurious correlations between non-causal attributes and classes, with counterfactual data augmentation being a promising direction for breaking these spurious associations. However, explicitly generating counterfactual data is challenging, with the training efficiency declining. Therefore, this study proposes an implicit counterfactual data augmentation (ICDA) method to remove spurious correlations and make stable predictions. Specifically, first, a novel sample-wise augmentation strategy is developed that generates semantically and counterfactually meaningful deep features with distinct augmentation strength for each sample. Second, we derive an easy-to-compute surrogate loss on the augmented feature set when the number of augmented samples becomes infinite. Third, two concrete schemes are proposed, including direct quantification and meta-learning, to derive the key parameters for the robust loss. In addition, ICDA is explained from 
    
[^7]: 通过离散优化实现可解释性异常检测

    Interpretable Anomaly Detection via Discrete Optimization. (arXiv:2303.14111v1 [cs.LG])

    [http://arxiv.org/abs/2303.14111](http://arxiv.org/abs/2303.14111)

    该论文提出了一个通过学习有限自动机进行异常检测的框架，并通过约束优化算法和新的正则化方案提高了可解释性。

    

    异常检测在许多应用领域中都是必不可少的，例如网络安全、执法、医学和欺诈保护。然而，目前深度学习方法的决策过程往往难以理解，这通常限制了它们的实际应用性。为了克服这个限制，我们提出了一个学习框架，可以从序列数据中学习可解释性的异常检测器。具体来说，我们考虑从给定的未标记序列多重集中学习确定性有限自动机 （DFA）的任务。我们证明了这个问题是计算难题，并基于约束优化开发了两个学习算法。此外，我们为优化问题引入了新的正则化方案，以提高我们的DFA的整体可解释性。通过原型实现，我们证明我们的方法在准确性和F1分数方面表现出有望的结果。

    Anomaly detection is essential in many application domains, such as cyber security, law enforcement, medicine, and fraud protection. However, the decision-making of current deep learning approaches is notoriously hard to understand, which often limits their practical applicability. To overcome this limitation, we propose a framework for learning inherently interpretable anomaly detectors from sequential data. More specifically, we consider the task of learning a deterministic finite automaton (DFA) from a given multi-set of unlabeled sequences. We show that this problem is computationally hard and develop two learning algorithms based on constraint optimization. Moreover, we introduce novel regularization schemes for our optimization problems that improve the overall interpretability of our DFAs. Using a prototype implementation, we demonstrate that our approach shows promising results in terms of accuracy and F1 score.
    
[^8]: 不要误会我：如何将深度视觉解释应用于时间序列

    Don't Get Me Wrong: How to Apply Deep Visual Interpretations to Time Series. (arXiv:2203.07861v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2203.07861](http://arxiv.org/abs/2203.07861)

    该论文提出了一个针对时间序列分类和分割任务的框架，通过六个度量来评估基于梯度、传播或干扰的事后可视化解释方法。实验结果表明，这些方法对于时间序列的解释具有较高的可信度和有效性。

    

    在许多应用中，正确解释和理解深度学习模型非常重要。针对图像和自然语言处理的解释性视觉解释方法允许领域专家验证和理解几乎任何深度学习模型。然而，当推广到任意时间序列时，它们在本质上更加复杂和多样化。一个可视化解释是否解释了有效的推理或捕捉了实际特征是难以判断的。因此，我们需要客观评估来获得可信的质量指标，而不是盲目信任。我们提出了一个框架，包括六个正交度量，用于针对时间序列分类和分割任务的基于梯度、传播或干扰的事后视觉解释方法。实验研究包括了常见的时间序列神经网络架构和九种可视化解释方法。我们使用UCR r等多样的数据集评估了这些可视化解释方法。

    The correct interpretation and understanding of deep learning models are essential in many applications. Explanatory visual interpretation approaches for image, and natural language processing allow domain experts to validate and understand almost any deep learning model. However, they fall short when generalizing to arbitrary time series, which is inherently less intuitive and more diverse. Whether a visualization explains valid reasoning or captures the actual features is difficult to judge. Hence, instead of blind trust, we need an objective evaluation to obtain trustworthy quality metrics. We propose a framework of six orthogonal metrics for gradient-, propagation- or perturbation-based post-hoc visual interpretation methods for time series classification and segmentation tasks. An experimental study includes popular neural network architectures for time series and nine visual interpretation methods. We evaluate the visual interpretation methods with diverse datasets from the UCR r
    

