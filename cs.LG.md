# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Serpent: Scalable and Efficient Image Restoration via Multi-scale Structured State Space Models](https://arxiv.org/abs/2403.17902) | Serpent提出了一种新的图像恢复架构，利用状态空间模型在全局感受野和计算效率之间取得平衡，实现了与最先进技术相当的重建质量，但计算量减少了数个数量级。 |
| [^2] | [Analyzing Male Domestic Violence through Exploratory Data Analysis and Explainable Machine Learning Insights](https://arxiv.org/abs/2403.15594) | 该研究是关于在孟加拉国背景下对男性家庭暴力进行开创性探索，揭示了男性受害者的存在、模式和潜在因素，填补了现有文献对男性受害者研究空白的重要性。 |
| [^3] | [Cartoon Hallucinations Detection: Pose-aware In Context Visual Learning](https://arxiv.org/abs/2403.15048) | 该研究提出了一种用于检测由TTI模型生成的卡通角色图像中视觉幻觉的系统，通过结合姿势感知上下文视觉学习和视觉语言模型，利用RGB图像和姿势信息，实现了更准确的决策，显著提高了视觉幻觉的识别能力，推动了TTI模型在非照片真实领域的发展。 |
| [^4] | [Optimal Transport for Domain Adaptation through Gaussian Mixture Models](https://arxiv.org/abs/2403.13847) | 通过高斯混合模型进行域自适应的最优输运，可以实现源域和目标域混合成分之间的匹配，从而在失效诊断中取得最先进的性能。 |
| [^5] | [Improving Cognitive Diagnosis Models with Adaptive Relational Graph Neural Networks](https://arxiv.org/abs/2403.05559) | 提出了一种利用自适应语义感知图神经网络改进认知诊断模型的方法，弥补了现有研究中对边缘内的异质性和不确定性的忽视。 |
| [^6] | [Fast Ergodic Search with Kernel Functions](https://arxiv.org/abs/2403.01536) | 提出了一种使用核函数的快速遍历搜索方法，其在搜索空间维度上具有线性复杂度，可以推广到李群，并且通过数值测试展示比现有算法快两个数量级。 |
| [^7] | [Coseparable Nonnegative Tensor Factorization With T-CUR Decomposition.](http://arxiv.org/abs/2401.16836) | 本文提出了一种基于T-CUR分解的可分离非负张量分解方法，用于在多维数据中提取有意义的特征。 |
| [^8] | [Bayesian Optimization with Hidden Constraints via Latent Decision Models.](http://arxiv.org/abs/2310.18449) | 本文介绍了一种基于潜在决策模型的贝叶斯优化方法，通过利用变分自编码器学习可行决策的分布，在原始空间和潜在空间之间实现了双向映射，从而解决了公共决策制定中的隐藏约束问题。 |
| [^9] | [Stochastic Submodular Bandits with Delayed Composite Anonymous Bandit Feedback.](http://arxiv.org/abs/2303.13604) | 本论文研究了具有随机次模收益和全赌徒延迟反馈的组合多臂赌博机问题，研究了三种延迟反馈模型并导出了后悔上限。研究结果表明，算法能够在考虑延迟组合匿名反馈时胜过其他全赌徒方法。 |

# 详细

[^1]: Serpent：通过多尺度结构化状态空间模型实现可扩展高效的图像恢复

    Serpent: Scalable and Efficient Image Restoration via Multi-scale Structured State Space Models

    [https://arxiv.org/abs/2403.17902](https://arxiv.org/abs/2403.17902)

    Serpent提出了一种新的图像恢复架构，利用状态空间模型在全局感受野和计算效率之间取得平衡，实现了与最先进技术相当的重建质量，但计算量减少了数个数量级。

    

    有效图像恢复架构的计算建筑块领域，主要由卷积处理和各种注意机制的组合所主导。然而，卷积滤波器本质上是局部的，因此在建模图像的长距离依赖性方面存在困难。另一方面，注意机制擅长捕获任意图像区域之间的全局相互作用，但对图像尺寸的二次成本较高。在这项工作中，我们提出了Serpent，这是一种利用最近在状态空间模型（SSMs）方面的进展作为其核心计算模块的架构。SSMs最初用于序列建模，可以通过有利的输入尺寸的线性缩放来维持全局感受野。我们的初步结果表明，Serpent可以实现与最先进技术相当的重建质量，同时需要数量级的计算量较少（在FLOPS上高达150倍的减少）。

    arXiv:2403.17902v1 Announce Type: cross  Abstract: The landscape of computational building blocks of efficient image restoration architectures is dominated by a combination of convolutional processing and various attention mechanisms. However, convolutional filters are inherently local and therefore struggle at modeling long-range dependencies in images. On the other hand, attention excels at capturing global interactions between arbitrary image regions, however at a quadratic cost in image dimension. In this work, we propose Serpent, an architecture that leverages recent advances in state space models (SSMs) in its core computational block. SSMs, originally introduced for sequence modeling, can maintain a global receptive field with a favorable linear scaling in input size. Our preliminary results demonstrate that Serpent can achieve reconstruction quality on par with state-of-the-art techniques, while requiring orders of magnitude less compute (up to $150$ fold reduction in FLOPS) an
    
[^2]: 通过探索性数据分析和可解释的机器学习洞见分析男性家庭暴力

    Analyzing Male Domestic Violence through Exploratory Data Analysis and Explainable Machine Learning Insights

    [https://arxiv.org/abs/2403.15594](https://arxiv.org/abs/2403.15594)

    该研究是关于在孟加拉国背景下对男性家庭暴力进行开创性探索，揭示了男性受害者的存在、模式和潜在因素，填补了现有文献对男性受害者研究空白的重要性。

    

    家庭暴力通常被视为一个关于女性受害者的性别问题，在近年来越来越受到关注。尽管有这种关注，孟加拉国特别是男性受害者仍然主要被忽视。我们的研究代表了在孟加拉国背景下对男性家庭暴力（MDV）这一未被充分探讨领域的开创性探索，揭示了其普遍性、模式和潜在因素。现有文献主要强调家庭暴力情境中女性的受害，导致对男性受害者的研究空白。我们从孟加拉国主要城市收集了数据，并进行了探索性数据分析以了解潜在动态。我们使用了11种传统机器学习模型（包括默认和优化的超参数）、2种深度学习和4种集成模型。尽管采用了各种方法，CatBoost由于其...

    arXiv:2403.15594v1 Announce Type: cross  Abstract: Domestic violence, which is often perceived as a gendered issue among female victims, has gained increasing attention in recent years. Despite this focus, male victims of domestic abuse remain primarily overlooked, particularly in Bangladesh. Our study represents a pioneering exploration of the underexplored realm of male domestic violence (MDV) within the Bangladeshi context, shedding light on its prevalence, patterns, and underlying factors. Existing literature predominantly emphasizes female victimization in domestic violence scenarios, leading to an absence of research on male victims. We collected data from the major cities of Bangladesh and conducted exploratory data analysis to understand the underlying dynamics. We implemented 11 traditional machine learning models with default and optimized hyperparameters, 2 deep learning, and 4 ensemble models. Despite various approaches, CatBoost has emerged as the top performer due to its 
    
[^3]: 卡通幻觉检测: 姿势感知上下文视觉学习

    Cartoon Hallucinations Detection: Pose-aware In Context Visual Learning

    [https://arxiv.org/abs/2403.15048](https://arxiv.org/abs/2403.15048)

    该研究提出了一种用于检测由TTI模型生成的卡通角色图像中视觉幻觉的系统，通过结合姿势感知上下文视觉学习和视觉语言模型，利用RGB图像和姿势信息，实现了更准确的决策，显著提高了视觉幻觉的识别能力，推动了TTI模型在非照片真实领域的发展。

    

    大规模文本到图像（TTI）模型已经成为各种生成领域中生成训练数据的常见方法。然而，视觉幻觉，尤其是在非照片真实风格如卡通人物中包含了感知上关键的缺陷，依然是一个令人担忧的问题。我们提出了一种新颖的用于检测TTI模型生成的卡通角色图像的视觉幻觉检测系统。我们的方法利用了姿势感知上下文视觉学习（PA-ICVL）与视觉语言模型（VLMs），同时利用RGB图像和姿势信息。通过从一个经过微调的姿势估计器中获得姿势指导，我们使VLM能够做出更准确的决策。实验结果表明，在识别视觉幻觉方面，与仅依赖于RGB图像的基线方法相比，取得了显著的改进。这项研究通过减轻视觉幻觉，推动了TTI模型在非照片真实领域的潜力。

    arXiv:2403.15048v1 Announce Type: cross  Abstract: Large-scale Text-to-Image (TTI) models have become a common approach for generating training data in various generative fields. However, visual hallucinations, which contain perceptually critical defects, remain a concern, especially in non-photorealistic styles like cartoon characters. We propose a novel visual hallucination detection system for cartoon character images generated by TTI models. Our approach leverages pose-aware in-context visual learning (PA-ICVL) with Vision-Language Models (VLMs), utilizing both RGB images and pose information. By incorporating pose guidance from a fine-tuned pose estimator, we enable VLMs to make more accurate decisions. Experimental results demonstrate significant improvements in identifying visual hallucinations compared to baseline methods relying solely on RGB images. This research advances TTI models by mitigating visual hallucinations, expanding their potential in non-photorealistic domains.
    
[^4]: 通过高斯混合模型进行域自适应的最优输运

    Optimal Transport for Domain Adaptation through Gaussian Mixture Models

    [https://arxiv.org/abs/2403.13847](https://arxiv.org/abs/2403.13847)

    通过高斯混合模型进行域自适应的最优输运，可以实现源域和目标域混合成分之间的匹配，从而在失效诊断中取得最先进的性能。

    

    在这篇论文中，我们探讨了通过最优输运进行域自适应的方法。我们提出了一种新颖的方法，即通过高斯混合模型对数据分布进行建模。这种策略使我们能够通过等价的离散问题解决连续最优输运。最优输运解决方案为我们提供了源域和目标域混合成分之间的匹配。通过这种匹配，我们可以在域之间映射数据点，或者将标签从源域组件转移到目标域。我们在失效诊断的两个域自适应基准测试中进行了实验，结果表明我们的方法具有最先进的性能。

    arXiv:2403.13847v1 Announce Type: cross  Abstract: In this paper we explore domain adaptation through optimal transport. We propose a novel approach, where we model the data distributions through Gaussian mixture models. This strategy allows us to solve continuous optimal transport through an equivalent discrete problem. The optimal transport solution gives us a matching between source and target domain mixture components. From this matching, we can map data points between domains, or transfer the labels from the source domain components towards the target domain. We experiment with 2 domain adaptation benchmarks in fault diagnosis, showing that our methods have state-of-the-art performance.
    
[^5]: 用自适应关系图神经网络改进认知诊断模型

    Improving Cognitive Diagnosis Models with Adaptive Relational Graph Neural Networks

    [https://arxiv.org/abs/2403.05559](https://arxiv.org/abs/2403.05559)

    提出了一种利用自适应语义感知图神经网络改进认知诊断模型的方法，弥补了现有研究中对边缘内的异质性和不确定性的忽视。

    

    认知诊断（CD）算法在智能教育中受到越来越多的研究关注。典型的CD算法通过推断学生的能力（即他们在各种知识概念上的熟练水平）来帮助学生。这种熟练水平可以进一步实现针对性的技能训练和个性化的练习建议，从而促进在线教育中学生的学习效率。最近，研究人员发现建立和整合学生-练习二部图对于增强诊断性能是有益的。然而，他们的研究仍然存在局限性。一方面，研究人员忽视了边缘内的异质性，即可能存在正确和错误的答案。另一方面，他们忽视了边缘内的不确定性，例如，正确的答案可能表示真正掌握或幸运猜测。为解决这些限制，我们提出了自适应语义感知图

    arXiv:2403.05559v1 Announce Type: cross  Abstract: Cognitive Diagnosis (CD) algorithms receive growing research interest in intelligent education. Typically, these CD algorithms assist students by inferring their abilities (i.e., their proficiency levels on various knowledge concepts). The proficiency levels can enable further targeted skill training and personalized exercise recommendations, thereby promoting students' learning efficiency in online education. Recently, researchers have found that building and incorporating a student-exercise bipartite graph is beneficial for enhancing diagnostic performance. However, there are still limitations in their studies. On one hand, researchers overlook the heterogeneity within edges, where there can be both correct and incorrect answers. On the other hand, they disregard the uncertainty within edges, e.g., a correct answer can indicate true mastery or fortunate guessing. To address the limitations, we propose Adaptive Semantic-aware Graph-ba
    
[^6]: 使用核函数的快速遍历搜索

    Fast Ergodic Search with Kernel Functions

    [https://arxiv.org/abs/2403.01536](https://arxiv.org/abs/2403.01536)

    提出了一种使用核函数的快速遍历搜索方法，其在搜索空间维度上具有线性复杂度，可以推广到李群，并且通过数值测试展示比现有算法快两个数量级。

    

    遍历搜索使得对信息分布进行最佳探索成为可能，同时保证了对搜索空间的渐近覆盖。然而，当前的方法通常在搜索空间维度上具有指数计算复杂度，并且局限于欧几里得空间。我们引入了一种计算高效的遍历搜索方法。我们的贡献是双重的。首先，我们开发了基于核的遍历度量，并将其从欧几里得空间推广到李群上。我们正式证明了所建议的度量与标准遍历度量一致，同时保证了在搜索空间维度上具有线性复杂度。其次，我们推导了非线性系统的核遍历度量的一阶最优性条件，这使得轨迹优化变得更加高效。全面的数值基准测试表明，所提出的方法至少比现有最先进的算法快两个数量级。

    arXiv:2403.01536v1 Announce Type: cross  Abstract: Ergodic search enables optimal exploration of an information distribution while guaranteeing the asymptotic coverage of the search space. However, current methods typically have exponential computation complexity in the search space dimension and are restricted to Euclidean space. We introduce a computationally efficient ergodic search method. Our contributions are two-fold. First, we develop a kernel-based ergodic metric and generalize it from Euclidean space to Lie groups. We formally prove the proposed metric is consistent with the standard ergodic metric while guaranteeing linear complexity in the search space dimension. Secondly, we derive the first-order optimality condition of the kernel ergodic metric for nonlinear systems, which enables efficient trajectory optimization. Comprehensive numerical benchmarks show that the proposed method is at least two orders of magnitude faster than the state-of-the-art algorithm. Finally, we d
    
[^7]: 基于T-CUR分解的可分离非负张量分解

    Coseparable Nonnegative Tensor Factorization With T-CUR Decomposition. (arXiv:2401.16836v1 [cs.LG])

    [http://arxiv.org/abs/2401.16836](http://arxiv.org/abs/2401.16836)

    本文提出了一种基于T-CUR分解的可分离非负张量分解方法，用于在多维数据中提取有意义的特征。

    

    非负矩阵分解(NMF)是一种重要的无监督学习方法，用于从数据中提取有意义的特征。为了在多项式时间框架内解决NMF问题，研究人员引入了可分离性假设，最近演变为可分离的概念。这一进展为原始数据提供了更高效的核心表示。然而，在现实世界中，数据更自然地被表示为多维数组，如图像或视频。将NMF应用于高维数据涉及向量化，会导致丢失关键的多维度相关性。为了保留数据中这些固有的相关性，我们转向张量(多维数组)并利用张量t乘积。这种方法将可分离的NMF扩展到张量设置，从而创建了我们所称的可分离非负张量分解(NTF)。在这项工作中，我们提供了一种交替索引选择方法来选择cos

    Nonnegative Matrix Factorization (NMF) is an important unsupervised learning method to extract meaningful features from data. To address the NMF problem within a polynomial time framework, researchers have introduced a separability assumption, which has recently evolved into the concept of coseparability. This advancement offers a more efficient core representation for the original data. However, in the real world, the data is more natural to be represented as a multi-dimensional array, such as images or videos. The NMF's application to high-dimensional data involves vectorization, which risks losing essential multi-dimensional correlations. To retain these inherent correlations in the data, we turn to tensors (multidimensional arrays) and leverage the tensor t-product. This approach extends the coseparable NMF to the tensor setting, creating what we term coseparable Nonnegative Tensor Factorization (NTF). In this work, we provide an alternating index selection method to select the cos
    
[^8]: 基于潜在决策模型的具有隐藏约束的贝叶斯优化方法

    Bayesian Optimization with Hidden Constraints via Latent Decision Models. (arXiv:2310.18449v1 [stat.ML])

    [http://arxiv.org/abs/2310.18449](http://arxiv.org/abs/2310.18449)

    本文介绍了一种基于潜在决策模型的贝叶斯优化方法，通过利用变分自编码器学习可行决策的分布，在原始空间和潜在空间之间实现了双向映射，从而解决了公共决策制定中的隐藏约束问题。

    

    贝叶斯优化（BO）已经成为解决复杂决策问题的强大工具，尤其在公共政策领域如警察划区方面。然而，由于定义可行区域的复杂性和决策的高维度，其在公共决策制定中的广泛应用受到了阻碍。本文介绍了一种新的贝叶斯优化方法——隐藏约束潜在空间贝叶斯优化（HC-LSBO），该方法集成了潜在决策模型。该方法利用变分自编码器来学习可行决策的分布，实现了原始决策空间与较低维度的潜在空间之间的双向映射。通过这种方式，HC-LSBO捕捉了公共决策制定中固有的隐藏约束的细微差别，在潜在空间中进行优化的同时，在原始空间中评估目标。我们通过对合成数据集和真实数据集进行数值实验来验证我们的方法，特别关注大规模问题。

    Bayesian optimization (BO) has emerged as a potent tool for addressing intricate decision-making challenges, especially in public policy domains such as police districting. However, its broader application in public policymaking is hindered by the complexity of defining feasible regions and the high-dimensionality of decisions. This paper introduces the Hidden-Constrained Latent Space Bayesian Optimization (HC-LSBO), a novel BO method integrated with a latent decision model. This approach leverages a variational autoencoder to learn the distribution of feasible decisions, enabling a two-way mapping between the original decision space and a lower-dimensional latent space. By doing so, HC-LSBO captures the nuances of hidden constraints inherent in public policymaking, allowing for optimization in the latent space while evaluating objectives in the original space. We validate our method through numerical experiments on both synthetic and real data sets, with a specific focus on large-scal
    
[^9]: 带有延迟组合匿名赌徒反馈的随机次模赌博算法

    Stochastic Submodular Bandits with Delayed Composite Anonymous Bandit Feedback. (arXiv:2303.13604v1 [cs.LG])

    [http://arxiv.org/abs/2303.13604](http://arxiv.org/abs/2303.13604)

    本论文研究了具有随机次模收益和全赌徒延迟反馈的组合多臂赌博机问题，研究了三种延迟反馈模型并导出了后悔上限。研究结果表明，算法能够在考虑延迟组合匿名反馈时胜过其他全赌徒方法。

    

    本文研究了组合多臂赌博机问题，其中包含了期望下的随机次模收益和全赌徒延迟反馈，延迟反馈被假定为组合和匿名。也就是说，延迟反馈是由过去行动的奖励组成的，这些奖励由子组件构成，其未知的分配方式。研究了三种延迟反馈模型：有界对抗模型、随机独立模型和随机条件独立模型，并针对每种延迟模型导出了后悔界。忽略问题相关参数，我们证明了所有延迟模型的后悔界为 $\tilde{O}(T^{2/3} + T^{1/3} \nu)$，其中 $T$ 是时间范围，$\nu$ 是三种情况下不同定义的延迟参数，因此展示了带有延迟的补偿项。所考虑的算法被证明能够胜过其他考虑了延迟组合匿名反馈的全赌徒方法。

    This paper investigates the problem of combinatorial multiarmed bandits with stochastic submodular (in expectation) rewards and full-bandit delayed feedback, where the delayed feedback is assumed to be composite and anonymous. In other words, the delayed feedback is composed of components of rewards from past actions, with unknown division among the sub-components. Three models of delayed feedback: bounded adversarial, stochastic independent, and stochastic conditionally independent are studied, and regret bounds are derived for each of the delay models. Ignoring the problem dependent parameters, we show that regret bound for all the delay models is $\tilde{O}(T^{2/3} + T^{1/3} \nu)$ for time horizon $T$, where $\nu$ is a delay parameter defined differently in the three cases, thus demonstrating an additive term in regret with delay in all the three delay models. The considered algorithm is demonstrated to outperform other full-bandit approaches with delayed composite anonymous feedbac
    

