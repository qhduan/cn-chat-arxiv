# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diff-Def: Diffusion-Generated Deformation Fields for Conditional Atlases](https://arxiv.org/abs/2403.16776) | 使用扩散模型生成形变场，将一般人口图谱转变为特定子人口的图谱，确保结构合理性，避免幻觉。 |
| [^2] | [The Elements of Differentiable Programming](https://arxiv.org/abs/2403.14606) | 可微分编程是一个新的编程范式，使得复杂程序能够端对端地进行微分，实现基于梯度的参数优化。 |
| [^3] | [Align and Distill: Unifying and Improving Domain Adaptive Object Detection](https://arxiv.org/abs/2403.12029) | 引入了统一的基准测试和实现框架ALDI以及新的DAOD基准数据集CFC-DAOD，解决了领域自适应目标检测中的基准问题，并支持未来方法的发展。 |
| [^4] | [Preserving correlations: A statistical method for generating synthetic data](https://arxiv.org/abs/2403.01471) | 提出了一种方法来生成具有统计代表性的合成数据，能够在合成数据集中保持原始数据集中特征之间的相关性，并提供可调整的隐私级别。 |
| [^5] | [Primal Dual Alternating Proximal Gradient Algorithms for Nonsmooth Nonconvex Minimax Problems with Coupled Linear Constraints](https://arxiv.org/abs/2212.04672) | 提出了用于具有耦合线性约束的非光滑非凸极小极大问题的两种算法，分别具有迭代复杂度保证。 |
| [^6] | [Sum-of-Parts Models: Faithful Attributions for Groups of Features.](http://arxiv.org/abs/2310.16316) | Sum-of-Parts模型通过构造保证特征组归因的忠实性，将预测分解为可解释的分数之和，帮助天体物理学家发现了关于星系形成的新知识。 |
| [^7] | [DF2: Distribution-Free Decision-Focused Learning.](http://arxiv.org/abs/2308.05889) | DF2是一种无分布的决策焦点学习方法，特别解决了模型不匹配错误、样本平均逼近误差和梯度逼近误差三个瓶颈问题。 |
| [^8] | [Robust Twin Parametric Margin Support Vector Machine for Multiclass Classification.](http://arxiv.org/abs/2306.06213) | 提出了双参数边界支持向量机模型来解决多类分类问题，并通过鲁棒优化技术使其更加鲁棒。初步实验结果表明其具有良好的性能。 |

# 详细

[^1]: Diff-Def: 通过扩散生成的形变场进行有条件的图谱制作

    Diff-Def: Diffusion-Generated Deformation Fields for Conditional Atlases

    [https://arxiv.org/abs/2403.16776](https://arxiv.org/abs/2403.16776)

    使用扩散模型生成形变场，将一般人口图谱转变为特定子人口的图谱，确保结构合理性，避免幻觉。

    

    解剖图谱广泛应用于人口分析。有条件的图谱针对通过特定条件（如人口统计学或病理学）定义的特定子人口，并允许研究与年龄相关的形态学差异等细粒度解剖学差异。现有方法使用基于配准的方法或生成模型，前者无法处理大的解剖学变异，后者可能在训练过程中出现不稳定和幻觉。为了克服这些限制，我们使用潜在扩散模型生成形变场，将一个常规人口图谱转变为代表特定子人口的图谱。通过生成形变场，并将有条件的图谱注册到一组图像附近，我们确保结构的合理性，避免直接图像合成时可能出现的幻觉。我们将我们的方法与几种最先进的方法进行了比较。

    arXiv:2403.16776v1 Announce Type: cross  Abstract: Anatomical atlases are widely used for population analysis. Conditional atlases target a particular sub-population defined via certain conditions (e.g. demographics or pathologies) and allow for the investigation of fine-grained anatomical differences - such as morphological changes correlated with age. Existing approaches use either registration-based methods that are unable to handle large anatomical variations or generative models, which can suffer from training instabilities and hallucinations. To overcome these limitations, we use latent diffusion models to generate deformation fields, which transform a general population atlas into one representing a specific sub-population. By generating a deformation field and registering the conditional atlas to a neighbourhood of images, we ensure structural plausibility and avoid hallucinations, which can occur during direct image synthesis. We compare our method to several state-of-the-art 
    
[^2]: 可微分编程的要素

    The Elements of Differentiable Programming

    [https://arxiv.org/abs/2403.14606](https://arxiv.org/abs/2403.14606)

    可微分编程是一个新的编程范式，使得复杂程序能够端对端地进行微分，实现基于梯度的参数优化。

    

    人工智能最近取得了显著进展，这得益于大型模型、庞大数据集、加速硬件，以及可微分编程的变革性力量。这种新的编程范式使复杂计算机程序（包括具有控制流和数据结构的程序）能够进行端对端的微分，从而实现对程序参数的基于梯度的优化。不仅仅是程序的微分，可微分编程也包括了程序优化、概率等多个领域的概念。本书介绍了可微分编程所需的基本概念，并采用了优化和概率两个主要视角进行阐述。

    arXiv:2403.14606v1 Announce Type: new  Abstract: Artificial intelligence has recently experienced remarkable advances, fueled by large models, vast datasets, accelerated hardware, and, last but not least, the transformative power of differentiable programming. This new programming paradigm enables end-to-end differentiation of complex computer programs (including those with control flows and data structures), making gradient-based optimization of program parameters possible. As an emerging paradigm, differentiable programming builds upon several areas of computer science and applied mathematics, including automatic differentiation, graphical models, optimization and statistics. This book presents a comprehensive review of the fundamental concepts useful for differentiable programming. We adopt two main perspectives, that of optimization and that of probability, with clear analogies between the two. Differentiable programming is not merely the differentiation of programs, but also the t
    
[^3]: 对齐与提炼：统一和改进领域自适应目标检测

    Align and Distill: Unifying and Improving Domain Adaptive Object Detection

    [https://arxiv.org/abs/2403.12029](https://arxiv.org/abs/2403.12029)

    引入了统一的基准测试和实现框架ALDI以及新的DAOD基准数据集CFC-DAOD，解决了领域自适应目标检测中的基准问题，并支持未来方法的发展。

    

    目标检测器通常表现不佳于与其训练集不同的数据。最近，领域自适应目标检测（DAOD）方法已经展示了在应对这一挑战上的强大结果。遗憾的是，我们发现了系统化的基准测试陷阱，这些陷阱对过去的结果提出质疑并阻碍了进一步的进展：（a）由于基线不足导致性能高估，（b）不一致的实现实践阻止了方法的透明比较，（c）由于过时的骨干和基准测试缺乏多样性，导致缺乏普遍性。我们通过引入以下问题来解决这些问题：（1）一个统一的基准测试和实现框架，Align and Distill（ALDI），支持DAOD方法的比较并支持未来发展，（2）一个公平且现代的DAOD训练和评估协议，解决了基准测试的陷阱，（3）一个新的DAOD基准数据集，CFC-DAOD，能够在多样化的真实环境中进行评估。

    arXiv:2403.12029v1 Announce Type: cross  Abstract: Object detectors often perform poorly on data that differs from their training set. Domain adaptive object detection (DAOD) methods have recently demonstrated strong results on addressing this challenge. Unfortunately, we identify systemic benchmarking pitfalls that call past results into question and hamper further progress: (a) Overestimation of performance due to underpowered baselines, (b) Inconsistent implementation practices preventing transparent comparisons of methods, and (c) Lack of generality due to outdated backbones and lack of diversity in benchmarks. We address these problems by introducing: (1) A unified benchmarking and implementation framework, Align and Distill (ALDI), enabling comparison of DAOD methods and supporting future development, (2) A fair and modern training and evaluation protocol for DAOD that addresses benchmarking pitfalls, (3) A new DAOD benchmark dataset, CFC-DAOD, enabling evaluation on diverse real
    
[^4]: 保持相关性：一种用于生成合成数据的统计方法

    Preserving correlations: A statistical method for generating synthetic data

    [https://arxiv.org/abs/2403.01471](https://arxiv.org/abs/2403.01471)

    提出了一种方法来生成具有统计代表性的合成数据，能够在合成数据集中保持原始数据集中特征之间的相关性，并提供可调整的隐私级别。

    

    我们提出了一种方法来生成具有统计代表性的合成数据。主要目标是在合成数据集中保持原始数据集中存在的特征之间的相关性，同时提供一个舒适的隐私级别，可以根据特定客户需求进行调整。我们详细描述了我们用于分析原始数据集和生成合成数据点的算法。我们使用了一个大型能源相关数据集进行了测试。我们在定性（例如通过可视化相关性图）和定量（以适当的$\ell^1$类型误差范数作为评估指标）方面获得了良好的结果。所提出的方法论是一般的，不依赖于使用的测试数据集。我们期望它可适用于比此处指示的更广泛的情境。

    arXiv:2403.01471v1 Announce Type: new  Abstract: We propose a method to generate statistically representative synthetic data. The main goal is to be able to maintain in the synthetic dataset the correlations of the features present in the original one, while offering a comfortable privacy level that can be eventually tailored on specific customer demands.   We describe in detail our algorithm used both for the analysis of the original dataset and for the generation of the synthetic data points. The approach is tested using a large energy-related dataset. We obtain good results both qualitatively (e.g. via vizualizing correlation maps) and quantitatively (in terms of suitable $\ell^1$-type error norms used as evaluation metrics).   The proposed methodology is general in the sense that it does not rely on the used test dataset. We expect it to be applicable in a much broader context than indicated here.
    
[^5]: Primal Dual Alternating Proximal Gradient算法用于具有耦合线性约束的非光滑非凸极小极大问题

    Primal Dual Alternating Proximal Gradient Algorithms for Nonsmooth Nonconvex Minimax Problems with Coupled Linear Constraints

    [https://arxiv.org/abs/2212.04672](https://arxiv.org/abs/2212.04672)

    提出了用于具有耦合线性约束的非光滑非凸极小极大问题的两种算法，分别具有迭代复杂度保证。

    

    非凸极小极大问题近年来在机器学习、信号处理和许多其他领域引起了广泛关注。本文提出了一种用于解决非光滑非凸（强）凹和非凸线性极小极大问题的原始对偶交替近端梯度（PDAPG）算法和原始对偶近端梯度（PDPG-L）算法，分别用于具有耦合线性约束的情况。这两种算法的迭代复杂度证明为 $\mathcal{O}\left( \varepsilon ^{-2} \right)$ （对应 $\mathcal{O}\left( \varepsilon ^{-4} \right)$）在非凸强凹 （对应非凸凹）情况下，以及 $\mathcal{O}\left( \varepsilon ^{-3} \right)$ 在非凸线性情况下，分别达到 $\varepsilon$-稳态点。据我们所知，它们是用于解决具有耦合线性约束的非凸极小极大问题的第一批具有迭代复杂度保证的算法。

    arXiv:2212.04672v3 Announce Type: replace-cross  Abstract: Nonconvex minimax problems have attracted wide attention in machine learning, signal processing and many other fields in recent years. In this paper, we propose a primal-dual alternating proximal gradient (PDAPG) algorithm and a primal-dual proximal gradient (PDPG-L) algorithm for solving nonsmooth nonconvex-(strongly) concave and nonconvex-linear minimax problems with coupled linear constraints, respectively. The iteration complexity of the two algorithms are proved to be $\mathcal{O}\left( \varepsilon ^{-2} \right)$ (resp. $\mathcal{O}\left( \varepsilon ^{-4} \right)$) under nonconvex-strongly concave (resp. nonconvex-concave) setting and $\mathcal{O}\left( \varepsilon ^{-3} \right)$ under nonconvex-linear setting to reach an $\varepsilon$-stationary point, respectively. To our knowledge, they are the first two algorithms with iteration complexity guarantees for solving the nonconvex minimax problems with coupled linear const
    
[^6]: Sum-of-Parts模型：对特征组的忠实归因

    Sum-of-Parts Models: Faithful Attributions for Groups of Features. (arXiv:2310.16316v1 [cs.LG])

    [http://arxiv.org/abs/2310.16316](http://arxiv.org/abs/2310.16316)

    Sum-of-Parts模型通过构造保证特征组归因的忠实性，将预测分解为可解释的分数之和，帮助天体物理学家发现了关于星系形成的新知识。

    

    如果机器学习模型的解释准确反映了其决策过程，则被认为是“忠实”的解释。然而，例如深度学习的特征归因等解释并不能保证忠实，有可能产生具有误导性的解释。在这项工作中，我们开发了Sum-of-Parts（SOP）模型，它是一类模型，其预测具有通过构造保证忠实的特征组归因。该模型将预测分解为可解释的分数之和，每个分数直接归因于一组稀疏特征。我们使用标准可解释性指标对SOP进行评估，并在一个案例研究中，利用SOP提供的忠实解释帮助天体物理学家发现了关于星系形成的新知识。

    An explanation of a machine learning model is considered "faithful" if it accurately reflects the model's decision-making process. However, explanations such as feature attributions for deep learning are not guaranteed to be faithful, and can produce potentially misleading interpretations. In this work, we develop Sum-of-Parts (SOP), a class of models whose predictions come with grouped feature attributions that are faithful-by-construction. This model decomposes a prediction into an interpretable sum of scores, each of which is directly attributable to a sparse group of features. We evaluate SOP on benchmarks with standard interpretability metrics, and in a case study, we use the faithful explanations from SOP to help astrophysicists discover new knowledge about galaxy formation.
    
[^7]: DF2: 无分布的决策焦点学习

    DF2: Distribution-Free Decision-Focused Learning. (arXiv:2308.05889v1 [cs.LG])

    [http://arxiv.org/abs/2308.05889](http://arxiv.org/abs/2308.05889)

    DF2是一种无分布的决策焦点学习方法，特别解决了模型不匹配错误、样本平均逼近误差和梯度逼近误差三个瓶颈问题。

    

    最近决策焦点学习（DFL）作为一种强大的方法在解决预测-优化问题时，通过将预测模型定制到一个下游优化任务。然而，现有的端到端DFL方法受到三个重要瓶颈的制约：模型不匹配错误、样本平均逼近误差和梯度逼近误差。模型不匹配错误源于模型参数化的预测分布与真实概率分布之间的不协调。样本平均逼近误差是使用有限样本来近似期望优化目标时产生的。梯度逼近误差发生在DFL依靠KKT条件进行精确梯度计算时，而大多数方法在非凸目标中近似梯度进行反向传播。在本文中，我们提出DF2 - 第一个明确设计来解决这三个瓶颈的无分布决策焦点学习方法。

    Decision-focused learning (DFL) has recently emerged as a powerful approach for predict-then-optimize problems by customizing a predictive model to a downstream optimization task. However, existing end-to-end DFL methods are hindered by three significant bottlenecks: model mismatch error, sample average approximation error, and gradient approximation error. Model mismatch error stems from the misalignment between the model's parameterized predictive distribution and the true probability distribution. Sample average approximation error arises when using finite samples to approximate the expected optimization objective. Gradient approximation error occurs as DFL relies on the KKT condition for exact gradient computation, while most methods approximate the gradient for backpropagation in non-convex objectives. In this paper, we present DF2 -- the first \textit{distribution-free} decision-focused learning method explicitly designed to address these three bottlenecks. Rather than depending 
    
[^8]: 基于双参数边界支持向量机的多类分类鲁棒性模型

    Robust Twin Parametric Margin Support Vector Machine for Multiclass Classification. (arXiv:2306.06213v1 [cs.LG])

    [http://arxiv.org/abs/2306.06213](http://arxiv.org/abs/2306.06213)

    提出了双参数边界支持向量机模型来解决多类分类问题，并通过鲁棒优化技术使其更加鲁棒。初步实验结果表明其具有良好的性能。

    

    本文提出一种双参数边界支持向量机(TPMSVM)模型来解决多类分类问题。 对于每个类别，我们采用一对割平面的模式构建一个分类器。一旦确定了所有分类器，则将它们组合成一个综合的决策函数。我们考虑线性和非线性内核引起的分类器的情况。此外，我们通过鲁棒优化技术增强了所提出的方法的鲁棒性。 初步的计算实验表明了所提出的方法的良好性能。

    In this paper we present a Twin Parametric-Margin Support Vector Machine (TPMSVM) model to tackle the problem of multiclass classification. In the spirit of one-versus-all paradigm, for each class we construct a classifier by solving a TPMSVM-type model. Once all classifiers have been determined, they are combined into an aggregate decision function. We consider the cases of both linear and nonlinear kernel-induced classifiers. In addition, we robustify the proposed approach through robust optimization techniques. Indeed, in real-world applications observations are subject to measurement errors and noise, affecting the quality of the solutions. Consequently, data uncertainties need to be included within the model in order to prevent low accuracies in the classification process. Preliminary computational experiments on real-world datasets show the good performance of the proposed approach.
    

