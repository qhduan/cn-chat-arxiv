# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Resilient Multiple Choice Learning: A learned scoring scheme with application to audio scene analysis.](http://arxiv.org/abs/2311.01052) | 这项研究引入了韧性多选学习（rMCL）方法，通过使用基于Voronoi tessellations的数学框架和学习评分方案，在回归设置中实现了对于每个训练输入可能采样多个目标的条件分布估计。该方法在合成数据和声源定位问题上得到了实证验证和进一步评估，展示了其实际的有用性和解释的相关性。 |
| [^2] | [Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD.](http://arxiv.org/abs/2307.00310) | 本文开发了一种新的DP-SGD分析方法，可以在训练过程中对许多数据点的隐私泄漏进行更准确的评估。 |
| [^3] | [Improved Bayes Risk Can Yield Reduced Social Welfare Under Competition.](http://arxiv.org/abs/2306.14670) | 本文研究了机器学习模型在竞争环境下的行为，发现提高数据表示质量可能会导致供应商整体预测准确性降低，从而降低社会福利。 |
| [^4] | [Open Problem: Learning with Variational Objectives on Measures.](http://arxiv.org/abs/2306.11928) | 本文探讨了在测度上编写变分目标的动机，提出通过此类目标推导实用算法，以解决超出分布的泛化和弱监督学习等问题的开放问题。 |
| [^5] | [Conditional Matrix Flows for Gaussian Graphical Models.](http://arxiv.org/abs/2306.07255) | 本文为解决高斯图模型中变量的条件独立结构问题，提出了一种针对精度矩阵的$l_p$正则化的方法，并将频率学派和贝叶斯学派的优点融合在变分推理中，并引入了矩阵变量标准化流程来逼近后验。 |
| [^6] | [Representational Strengths and Limitations of Transformers.](http://arxiv.org/abs/2306.02896) | 本文研究了transformer的表示能力，正面说明了transformer在稀疏平均任务中的效率比循环网络和前馈网络更高，并展示了大嵌入维度在transformer中的必要性和作用；负面说明了注意力层的复杂度随输入大小线性缩放，但这种情况在实践中很少发生，可以使用替代的变体。 |
| [^7] | [Investigating the Corruption Robustness of Image Classifiers with Random Lp-norm Corruptions.](http://arxiv.org/abs/2305.05400) | 本研究探讨了使用随机Lp范数失真对图像分类器的训练和测试数据进行增强，并评估模型对不可感知随机失真的稳健性，发现稳健性可能会提高模型在随机失真方面的性能，但也可能会损害L∞范数的稳健性。 |
| [^8] | [Comparing Machine Learning Methods for Estimating Heterogeneous Treatment Effects by Combining Data from Multiple Randomized Controlled Trials.](http://arxiv.org/abs/2303.16299) | 本文研究了多个试验中利用数据估计个体化治疗效应的非参数方法，模拟表明直接允许试验间治疗效应的异质性的方法表现更好，单一研究方法的选择取决于治疗效应的功能形式。 |
| [^9] | [Learning to Reconstruct Signals From Binary Measurements.](http://arxiv.org/abs/2303.08691) | 该论文提出了一种新的自监督学习方法SSBM，它只需要二进制数据进行训练，并探索了从不完整的二进制观察中学习的极端情况。这为从二进制测量中恢复信号提供了必要和充分条件，并在一系列真实数据集上展示了SSBM的卓越表现。 |
| [^10] | [Bayesian Learning via Q-Exponential Process.](http://arxiv.org/abs/2210.07987) | 该论文研究了基于Q-指数过程的贝叶斯学习，通过推广Q-指数分布为Q-指数过程，来对函数的L_q正则化进行建模，并选择一致的多元q-指数分布。 |
| [^11] | [Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start.](http://arxiv.org/abs/2202.03397) | 本文针对一类双层问题，提出了无需warm-start也可实现最优样本复杂度的方法。 |

# 详细

[^1]: 韧性多选学习：用于音频场景分析的学习评分方案的引入

    Resilient Multiple Choice Learning: A learned scoring scheme with application to audio scene analysis. (arXiv:2311.01052v1 [stat.ML])

    [http://arxiv.org/abs/2311.01052](http://arxiv.org/abs/2311.01052)

    这项研究引入了韧性多选学习（rMCL）方法，通过使用基于Voronoi tessellations的数学框架和学习评分方案，在回归设置中实现了对于每个训练输入可能采样多个目标的条件分布估计。该方法在合成数据和声源定位问题上得到了实证验证和进一步评估，展示了其实际的有用性和解释的相关性。

    

    我们引入了韧性多选学习（rMCL），这是一种对于每个训练输入可能采样多个目标的回归设置下条件分布估计的MCL方法的扩展。多选学习是一个简单的框架，用于处理多模态密度估计，使用了一组假设的胜者全拿（WTA）损失。在回归设置中，现有的MCL变体主要集中在合并假设上，从而最终牺牲了预测的多样性。相反，我们的方法依赖于一个基于Voronoi tessellations的输出空间的数学框架支持的新颖的学习评分方案，我们可以从中得出概率解释。在对合成数据进行实证验证后，我们进一步评估了rMCL在声源定位问题上的优点，展示了其实际的有用性和解释的相关性。

    We introduce Resilient Multiple Choice Learning (rMCL), an extension of the MCL approach for conditional distribution estimation in regression settings where multiple targets may be sampled for each training input. Multiple Choice Learning is a simple framework to tackle multimodal density estimation, using the Winner-Takes-All (WTA) loss for a set of hypotheses. In regression settings, the existing MCL variants focus on merging the hypotheses, thereby eventually sacrificing the diversity of the predictions. In contrast, our method relies on a novel learned scoring scheme underpinned by a mathematical framework based on Voronoi tessellations of the output space, from which we can derive a probabilistic interpretation. After empirically validating rMCL with experiments on synthetic data, we further assess its merits on the sound source localization problem, demonstrating its practical usefulness and the relevance of its interpretation.
    
[^2]: 梯度相似：敏感度经常被过高估计在DP-SGD中

    Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD. (arXiv:2307.00310v1 [cs.LG])

    [http://arxiv.org/abs/2307.00310](http://arxiv.org/abs/2307.00310)

    本文开发了一种新的DP-SGD分析方法，可以在训练过程中对许多数据点的隐私泄漏进行更准确的评估。

    

    差分隐私随机梯度下降（DP-SGD）是私有深度学习的标准算法。虽然已知其隐私分析在最坏情况下是紧密的，但是一些实证结果表明，在常见的基准数据集上训练时，所得到的模型对许多数据点的隐私泄漏显著减少。在本文中，我们为DP-SGD开发了一种新的分析方法，捕捉到在数据集中具有相似邻居的点享受更好隐私性的直觉。形式上来说，这是通过修改从训练数据集计算得到的模型更新的每步隐私性分析来实现的。我们进一步开发了一个新的组合定理，以有效地利用这个新的每步分析来推理整个训练过程。总而言之，我们的评估结果表明，这种新颖的DP-SGD分析使我们能够正式地显示DP-SGD对许多数据点的隐私泄漏显著减少。

    Differentially private stochastic gradient descent (DP-SGD) is the canonical algorithm for private deep learning. While it is known that its privacy analysis is tight in the worst-case, several empirical results suggest that when training on common benchmark datasets, the models obtained leak significantly less privacy for many datapoints. In this paper, we develop a new analysis for DP-SGD that captures the intuition that points with similar neighbors in the dataset enjoy better privacy than outliers. Formally, this is done by modifying the per-step privacy analysis of DP-SGD to introduce a dependence on the distribution of model updates computed from a training dataset. We further develop a new composition theorem to effectively use this new per-step analysis to reason about an entire training run. Put all together, our evaluation shows that this novel DP-SGD analysis allows us to now formally show that DP-SGD leaks significantly less privacy for many datapoints. In particular, we ob
    
[^3]: 竞争环境下贝叶斯风险的提高可能导致社会福利的降低

    Improved Bayes Risk Can Yield Reduced Social Welfare Under Competition. (arXiv:2306.14670v1 [cs.GT])

    [http://arxiv.org/abs/2306.14670](http://arxiv.org/abs/2306.14670)

    本文研究了机器学习模型在竞争环境下的行为，发现提高数据表示质量可能会导致供应商整体预测准确性降低，从而降低社会福利。

    

    随着机器学习模型规模的增长，缩放定律等趋势预计会导致预测准确性的持续改进。然而，这些趋势只考虑了单个模型供应商的视角，而实际上供应商之间常常竞争用户。本文证明了竞争可以从根本上改变这些缩放趋势的行为，甚至可能造成整体预测准确性随着规模的增大而非单调或降低。我们定义了一个分类任务的竞争模型，并使用数据表示作为研究规模增加的影响的镜头。我们发现在一家市场上，改善数据表示质量（按贝叶斯风险计量）可能会降低竞争模型供应商的整体预测准确性（即社会福利）。我们的例子涵盖了简单设置中的封闭式公式到预训练的 CIFAR-10 模拟。

    As the scale of machine learning models increases, trends such as scaling laws anticipate consistent downstream improvements in predictive accuracy. However, these trends take the perspective of a single model-provider in isolation, while in reality providers often compete with each other for users. In this work, we demonstrate that competition can fundamentally alter the behavior of these scaling trends, even causing overall predictive accuracy across users to be non-monotonic or decreasing with scale. We define a model of competition for classification tasks, and use data representations as a lens for studying the impact of increases in scale. We find many settings where improving data representation quality (as measured by Bayes risk) decreases the overall predictive accuracy across users (i.e., social welfare) for a marketplace of competing model-providers. Our examples range from closed-form formulas in simple settings to simulations with pretrained representations on CIFAR-10. At
    
[^4]: 开放问题：基于变分目标的测度学习 (arXiv:2306.11928v1 [stat.ML])

    Open Problem: Learning with Variational Objectives on Measures. (arXiv:2306.11928v1 [stat.ML])

    [http://arxiv.org/abs/2306.11928](http://arxiv.org/abs/2306.11928)

    本文探讨了在测度上编写变分目标的动机，提出通过此类目标推导实用算法，以解决超出分布的泛化和弱监督学习等问题的开放问题。

    

    统计学习理论关注的是基于函数的变分目标。本文讨论了在测度上编写类似目标的动机，特别是讨论了超出分布的泛化和弱监督学习。这引发了一个自然的问题：能否将通常的统计学习结果转化为基于测量表达的目标？结果构建是否会导致新的实用算法？

    The theory of statistical learning has focused on variational objectives expressed on functions. In this note, we discuss motivations to write similar objectives on measures, in particular to discuss out-of-distribution generalization and weakly-supervised learning. It raises a natural question: can one cast usual statistical learning results to objectives expressed on measures? Does the resulting construction lead to new algorithms of practical interest?
    
[^5]: 针对高斯图模型的条件矩阵流

    Conditional Matrix Flows for Gaussian Graphical Models. (arXiv:2306.07255v1 [cs.LG])

    [http://arxiv.org/abs/2306.07255](http://arxiv.org/abs/2306.07255)

    本文为解决高斯图模型中变量的条件独立结构问题，提出了一种针对精度矩阵的$l_p$正则化的方法，并将频率学派和贝叶斯学派的优点融合在变分推理中，并引入了矩阵变量标准化流程来逼近后验。

    

    在少数观测变量中研究许多变量之间的条件独立结构是一项具有挑战性的任务。高斯图模型通过在$l_p$正则化中鼓励精度矩阵的稀疏性来解决此问题，其中$p \leq1$。然而，由于亚-$l_1$伪范数使目标高度非凸，因此大多数方法依赖于$l_1$范数。在这种情况下，频率学派方法允许优雅地计算作为收缩参数$\lambda$函数的解决方案路径。贝叶斯公式为精度矩阵引入了拉普拉斯先验，但是不同$\lambda$值的后验推断需要多次运行昂贵的吉布斯采样。我们提出了一个非常通用的框架，用于GGM的变分推理，它统一了频率学派和贝叶斯学派的优点。具体而言，我们建议用定义在s空间上的矩阵变量标准化流程来逼近后验。

    Studying conditional independence structure among many variables with few observations is a challenging task. Gaussian Graphical Models (GGMs) tackle this problem by encouraging sparsity in the precision matrix through an $l_p$ regularization with $p\leq1$. However, since the objective is highly non-convex for sub-$l_1$ pseudo-norms, most approaches rely on the $l_1$ norm. In this case frequentist approaches allow to elegantly compute the solution path as a function of the shrinkage parameter $\lambda$. Instead of optimizing the penalized likelihood, the Bayesian formulation introduces a Laplace prior on the precision matrix. However, posterior inference for different $\lambda$ values requires repeated runs of expensive Gibbs samplers. We propose a very general framework for variational inference in GGMs that unifies the benefits of frequentist and Bayesian frameworks. Specifically, we propose to approximate the posterior with a matrix-variate Normalizing Flow defined on the space of s
    
[^6]: Transformer的代表性优势和局限性

    Representational Strengths and Limitations of Transformers. (arXiv:2306.02896v1 [cs.LG])

    [http://arxiv.org/abs/2306.02896](http://arxiv.org/abs/2306.02896)

    本文研究了transformer的表示能力，正面说明了transformer在稀疏平均任务中的效率比循环网络和前馈网络更高，并展示了大嵌入维度在transformer中的必要性和作用；负面说明了注意力层的复杂度随输入大小线性缩放，但这种情况在实践中很少发生，可以使用替代的变体。

    

    注意力层常用于transformer中，是现代深度学习的支柱之一，但与其他网络结构相比，它们的好处和缺陷没有数学描述。在本研究中，我们对注意力层的表示能力进行了正面和负面的研究，并聚焦于内在复杂度参数，如宽度、深度和嵌入维度。在正面方面，我们提出了一项稀疏平均任务，其中循环网络和前馈网络的复杂度都随输入大小呈多项式缩放，而transformer仅呈对数缩放；此外，我们使用相同的构造来展示transformer中大嵌入维度的必要性和作用。在负面方面，我们提出了一个三元检测任务，其中注意力层的复杂度随输入大小呈线性缩放；由于这种情况在实践中似乎很少发生，因此我们还提出了可以替代的变体。

    Attention layers, as commonly used in transformers, form the backbone of modern deep learning, yet there is no mathematical description of their benefits and deficiencies as compared with other architectures. In this work we establish both positive and negative results on the representation power of attention layers, with a focus on intrinsic complexity parameters such as width, depth, and embedding dimension. On the positive side, we present a sparse averaging task, where recurrent networks and feedforward networks all have complexity scaling polynomially in the input size, whereas transformers scale merely logarithmically in the input size; furthermore, we use the same construction to show the necessity and role of a large embedding dimension in a transformer. On the negative side, we present a triple detection task, where attention layers in turn have complexity scaling linearly in the input size; as this scenario seems rare in practice, we also present natural variants that can be 
    
[^7]: 使用随机Lp范数失真探究图像分类器的腐败稳健性

    Investigating the Corruption Robustness of Image Classifiers with Random Lp-norm Corruptions. (arXiv:2305.05400v1 [cs.LG])

    [http://arxiv.org/abs/2305.05400](http://arxiv.org/abs/2305.05400)

    本研究探讨了使用随机Lp范数失真对图像分类器的训练和测试数据进行增强，并评估模型对不可感知随机失真的稳健性，发现稳健性可能会提高模型在随机失真方面的性能，但也可能会损害L∞范数的稳健性。

    

    稳健性是机器学习分类器实现安全和可靠的基本属性。在对图像分类模型的对抗稳健性和形式稳健性验证领域中，稳健性通常被定义为在Lp范数距离内对所有输入变化的稳定性。然而，对随机失真的稳健性通常通过在现实世界中观察到的变化来改进和评估，而很少考虑数学定义的Lp范数失真。本研究探讨了使用随机Lp范数失真来增强图像分类器的训练和测试数据。我们借鉴了对抗稳健性领域的方法来评估模型对不可感知随机失真的稳健性。我们实证和理论上研究了在不同Lp范数之间稳健性是否可转移，并得出结论，哪些Lp范数的失真应该用来训练和评估模型。我们发现训练数据增强可能会提高模型在随机失真方面的性能，但也可能会损害L∞范数的稳健性。

    Robustness is a fundamental property of machine learning classifiers to achieve safety and reliability. In the fields of adversarial robustness and formal robustness verification of image classification models, robustness is commonly defined as the stability to all input variations within an Lp-norm distance. However, robustness to random corruptions is usually improved and evaluated using variations observed in the real-world, while mathematically defined Lp-norm corruptions are rarely considered. This study investigates the use of random Lp-norm corruptions to augment the training and test data of image classifiers. We adapt an approach from the field of adversarial robustness to assess the model robustness to imperceptible random corruptions. We empirically and theoretically investigate whether robustness is transferable across different Lp-norms and derive conclusions on which Lp-norm corruptions a model should be trained and evaluated on. We find that training data augmentation wi
    
[^8]: 结合多个随机对照试验数据的机器学习方法比较异质性治疗效应估计

    Comparing Machine Learning Methods for Estimating Heterogeneous Treatment Effects by Combining Data from Multiple Randomized Controlled Trials. (arXiv:2303.16299v1 [stat.ME])

    [http://arxiv.org/abs/2303.16299](http://arxiv.org/abs/2303.16299)

    本文研究了多个试验中利用数据估计个体化治疗效应的非参数方法，模拟表明直接允许试验间治疗效应的异质性的方法表现更好，单一研究方法的选择取决于治疗效应的功能形式。

    

    个性化的治疗决策可以改善健康结果，但是使用数据以可靠、精确和有普遍意义的方式进行这些决策在一个单一数据集中是有挑战性的。利用多个随机对照试验可以组合具有非混杂性治疗分配的数据集，提高估计异质性治疗效应的能力。本文讨论了几种非参数方法，用于利用多个试验的数据估计异质性治疗效应。我们将单一研究的方法扩展到多个试验的场景中，并通过模拟研究探讨它们的性能，数据生成情景具有不同水平的跨试验异质性。模拟表明，直接允许试验间治疗效应的异质性的方法比不允许的方法表现更好，并且单一研究方法的选择取决于治疗效应的功能形式。最后，通过对减少住院重新入院干预的网络荟萃分析数据的应用，我们比较了实践中的方法，并讨论了对未来研究的影响。

    Individualized treatment decisions can improve health outcomes, but using data to make these decisions in a reliable, precise, and generalizable way is challenging with a single dataset. Leveraging multiple randomized controlled trials allows for the combination of datasets with unconfounded treatment assignment to improve the power to estimate heterogeneous treatment effects. This paper discusses several non-parametric approaches for estimating heterogeneous treatment effects using data from multiple trials. We extend single-study methods to a scenario with multiple trials and explore their performance through a simulation study, with data generation scenarios that have differing levels of cross-trial heterogeneity. The simulations demonstrate that methods that directly allow for heterogeneity of the treatment effect across trials perform better than methods that do not, and that the choice of single-study method matters based on the functional form of the treatment effect. Finally, w
    
[^9]: 从二进制测量中学习信号重构

    Learning to Reconstruct Signals From Binary Measurements. (arXiv:2303.08691v1 [eess.SP])

    [http://arxiv.org/abs/2303.08691](http://arxiv.org/abs/2303.08691)

    该论文提出了一种新的自监督学习方法SSBM，它只需要二进制数据进行训练，并探索了从不完整的二进制观察中学习的极端情况。这为从二进制测量中恢复信号提供了必要和充分条件，并在一系列真实数据集上展示了SSBM的卓越表现。

    

    无监督学习的最新进展突出了仅从噪声和不完整的线性测量中学习信号重构的可能性。这些方法在医学和科学成像以及传感中起到关键作用，其中地面真实数据经常稀缺或难以获得。然而，在实践中，测量不仅噪声和不完整，而且还被量化。在这里，我们探索从二进制观察中学习的极端情况，并提供了关于从不完整二进制数据中识别一组信号所需的测量数量的必要和充分条件。我们的结果是对从二进制测量中信号恢复现有界限的补充。此外，我们引入了一种新颖的自监督学习方法，我们将其命名为“SSBM”，它仅需要二进制数据进行训练。我们在一系列真实数据集上的实验证明SSBM与监督学习相当，并优于稀疏重构方法。

    Recent advances in unsupervised learning have highlighted the possibility of learning to reconstruct signals from noisy and incomplete linear measurements alone. These methods play a key role in medical and scientific imaging and sensing, where ground truth data is often scarce or difficult to obtain. However, in practice, measurements are not only noisy and incomplete but also quantized. Here we explore the extreme case of learning from binary observations and provide necessary and sufficient conditions on the number of measurements required for identifying a set of signals from incomplete binary data. Our results are complementary to existing bounds on signal recovery from binary measurements. Furthermore, we introduce a novel self-supervised learning approach, which we name SSBM, that only requires binary data for training. We demonstrate in a series of experiments with real datasets that SSBM performs on par with supervised learning and outperforms sparse reconstruction methods wit
    
[^10]: 基于Q-指数过程的贝叶斯学习

    Bayesian Learning via Q-Exponential Process. (arXiv:2210.07987v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2210.07987](http://arxiv.org/abs/2210.07987)

    该论文研究了基于Q-指数过程的贝叶斯学习，通过推广Q-指数分布为Q-指数过程，来对函数的L_q正则化进行建模，并选择一致的多元q-指数分布。

    

    正则化是优化、统计和机器学习中最基础的主题之一。为了在估计参数$u\in\mbR^d$时获得稀疏性，在目标函数中通常会添加$\ell_q$惩罚项，即$\Vert u\Vert_q$。这样的$\ell_q$惩罚对应的概率分布是什么？当我们对函数$u\in L^q$建模时，$\Vert u\Vert_q$对应的正确随机过程是什么？这对于统计建模大维度对象（例如图像）并保留确定性特性（例如图像边缘）的惩罚非常重要。在这项工作中，我们将$Q$-指数分布（密度正比于$\exp{(- \half|u|^q)}$）推广为一种称为\emph{$Q$-指数（Q-EP）过程}的随机过程，它对应于函数的$L_q$正则化。关键步骤是通过从大型椭圆轮廓分布族中选择来指定一致的多元$q$-指数分布。

    Regularization is one of the most fundamental topics in optimization, statistics and machine learning. To get sparsity in estimating a parameter $u\in\mbR^d$, an $\ell_q$ penalty term, $\Vert u\Vert_q$, is usually added to the objective function. What is the probabilistic distribution corresponding to such $\ell_q$ penalty? What is the correct stochastic process corresponding to $\Vert u\Vert_q$ when we model functions $u\in L^q$? This is important for statistically modeling large dimensional objects, e.g. images, with penalty to preserve certainty properties, e.g. edges in the image. In this work, we generalize the $q$-exponential distribution (with density proportional to) $\exp{(- \half|u|^q)}$ to a stochastic process named \emph{$Q$-exponential (Q-EP) process} that corresponds to the $L_q$ regularization of functions. The key step is to specify consistent multivariate $q$-exponential distributions by choosing from a large family of elliptic contour distributions. The work is closel
    
[^11]: 有下层压缩的双层优化: 无warm-start情况下最优样本复杂度分析

    Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start. (arXiv:2202.03397v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2202.03397](http://arxiv.org/abs/2202.03397)

    本文针对一类双层问题，提出了无需warm-start也可实现最优样本复杂度的方法。

    

    本文分析了一类一般的双层问题，其中上层问题是将一光滑目标函数最小化，下层问题是寻找一光滑收缩映射的不动点。这类问题包括元学习、均衡模型、超参数优化和数据污染对抗攻击的实例。我们展示了，即使没有warm-start，在某些情况下，如元学习和均衡模型，仍然可以实现顺序最优的样本复杂度。

    We analyse a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower-level problem, i.e. they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise (near) optimal
    

